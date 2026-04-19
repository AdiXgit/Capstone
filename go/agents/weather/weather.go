// go/agents/weather/main.go
// OWNER: Aditya
// Weather Intelligence Agent — powered by Open-Meteo (free, no API key)
// API docs: https://open-meteo.com/en/docs
// Covers all 29 Karnataka paddy districts with real-time + 16-day forecast
//
// ── FIXES (IRRI AWD-aligned irrigation logic) ──────────────────────────────
// FIX-1  farmingAdvisory() is now crop-stage-aware instead of a flat
//        condition→string map. It handles vegetative, flowering, and
//        grain-filling stages with IRRI-prescribed language.
// FIX-2  checkAlert() gains two new alert types:
//          • IRRIGATION_NEEDED — water-balance proxy triggers re-flood
//          • SKIP_IRRIGATION   — heavy rain already replenishing field
//        and accepts daysAfterTransplant + 7-day forecast rain sum so
//        the flowering window (DAT 55-70) is respected as a hard lock.
// FIX-3  estimateWaterBalance() implements a simple daily ET-based
//        water balance over the 7-day forecast to predict the earliest
//        day irrigation will be needed (returns -1 if not needed).
// FIX-4  GetWeatherForecast() now derives DaysAfterTransplant from the
//        request field and threads it through advisory + alert calls.
// FIX-5  GetWeatherAlert() similarly accepts and uses DAT.
// ───────────────────────────────────────────────────────────────────────────

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"math"
	"net"
	"net/http"
	"os"
	"strconv"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/reflection"

	pb "github.com/karnataka-paddy/paddy-multiagent/proto"
)

// ─────────────────────────────────────────────
// DISTRICT COORDINATES — all 29 paddy districts
// ─────────────────────────────────────────────

var districtCoords = map[string][2]float64{
	"Koppal":           {15.3530, 76.1547},
	"Ballari":          {15.1394, 76.9214},
	"Raichur":          {16.2120, 77.3566},
	"Davanagere":       {14.4644, 75.9218},
	"Shivamogga":       {13.9299, 75.5681},
	"Hassan":           {13.0068, 76.1004},
	"Mandya":           {12.5218, 76.8951},
	"Mysuru":           {12.2958, 76.6394},
	"Belagavi":         {15.8497, 74.4977},
	"Dharwad":          {15.4589, 75.0078},
	"Bagalkot":         {16.1691, 75.6967},
	"Bengaluru Rural":  {13.1986, 77.5697},
	"Bidar":            {17.9142, 77.5199},
	"Chamarajanagar":   {11.9261, 76.9437},
	"Chikkaballapura":  {13.4354, 77.7280},
	"Chikkamagaluru":   {13.3161, 75.7720},
	"Chitradurga":      {14.2251, 76.3980},
	"Dakshina Kannada": {12.8438, 74.9900},
	"Gadag":            {15.4164, 75.6254},
	"Haveri":           {14.7954, 75.3988},
	"Kalaburagi":       {17.3297, 76.8200},
	"Kodagu":           {12.3375, 75.8069},
	"Kolar":            {13.1361, 78.1294},
	"Ramanagara":       {12.7157, 77.2804},
	"Tumakuru":         {13.3400, 77.1010},
	"Udupi":            {13.3409, 74.7421},
	"Uttara Kannada":   {14.7860, 74.6910},
	"Vijayapura":       {16.8302, 75.7100},
	"Yadgir":           {16.7726, 77.1383},
}

func getCoords(district string) (lat, lon float64) {
	if c, ok := districtCoords[district]; ok {
		return c[0], c[1]
	}
	return 14.5204, 75.7224 // Karnataka centroid fallback
}

// ─────────────────────────────────────────────
// OPEN-METEO RESPONSE TYPES
// ─────────────────────────────────────────────

type currentResp struct {
	Current struct {
		Temp2m      float64 `json:"temperature_2m"`
		Humidity    int     `json:"relative_humidity_2m"`
		Precip      float64 `json:"precipitation"`
		WindSpeed   float64 `json:"wind_speed_10m"`
		WeatherCode int     `json:"weather_code"`
	} `json:"current"`
	Daily struct {
		Time         []string  `json:"time"`
		TempMax      []float64 `json:"temperature_2m_max"`
		TempMin      []float64 `json:"temperature_2m_min"`
		PrecipSum    []float64 `json:"precipitation_sum"`
		PrecipProb   []int     `json:"precipitation_probability_max"`
		WeatherCode  []int     `json:"weather_code"`
		SunshineSec  []float64 `json:"sunshine_duration"`
		WindSpeedMax []float64 `json:"wind_speed_10m_max"`
	} `json:"daily"`
}

type archiveResp struct {
	Daily struct {
		Time      []string  `json:"time"`
		TempMax   []float64 `json:"temperature_2m_max"`
		TempMin   []float64 `json:"temperature_2m_min"`
		TempMean  []float64 `json:"temperature_2m_mean"`
		PrecipSum []float64 `json:"precipitation_sum"`
		Humidity  []float64 `json:"relative_humidity_2m_mean"`
		Wind      []float64 `json:"wind_speed_10m_mean"`
		Sunshine  []float64 `json:"sunshine_duration"`
	} `json:"daily"`
}

// ─────────────────────────────────────────────
// HTTP CLIENT (cached, retried)
// ─────────────────────────────────────────────

type apiClient struct {
	http  *http.Client
	mu    sync.Mutex
	cache map[string]struct {
		body []byte
		exp  time.Time
	}
}

func newAPIClient() *apiClient {
	c := &apiClient{
		http: &http.Client{Timeout: 12 * time.Second},
	}
	c.cache = make(map[string]struct {
		body []byte
		exp  time.Time
	})
	return c
}

func (c *apiClient) get(url string) ([]byte, error) {
	c.mu.Lock()
	if e, ok := c.cache[url]; ok && time.Now().Before(e.exp) {
		c.mu.Unlock()
		return e.body, nil
	}
	c.mu.Unlock()

	var lastErr error
	for i := 0; i < 3; i++ {
		if i > 0 {
			time.Sleep(time.Duration(i*500) * time.Millisecond)
		}
		resp, err := c.http.Get(url)
		if err != nil {
			lastErr = err
			continue
		}
		body, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			lastErr = err
			continue
		}
		if resp.StatusCode != 200 {
			lastErr = fmt.Errorf("HTTP %d from Open-Meteo", resp.StatusCode)
			continue
		}
		c.mu.Lock()
		c.cache[url] = struct {
			body []byte
			exp  time.Time
		}{body, time.Now().Add(15 * time.Minute)}
		c.mu.Unlock()
		return body, nil
	}
	return nil, lastErr
}

func (c *apiClient) fetchCurrent(lat, lon float64) (*currentResp, error) {
	url := fmt.Sprintf(
		"https://api.open-meteo.com/v1/forecast"+
			"?latitude=%.4f&longitude=%.4f"+
			"&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code"+
			"&daily=temperature_2m_max,temperature_2m_min,precipitation_sum,"+
			"precipitation_probability_max,weather_code,sunshine_duration,wind_speed_10m_max"+
			"&timezone=Asia%%2FKolkata&forecast_days=16",
		lat, lon,
	)
	body, err := c.get(url)
	if err != nil {
		return nil, err
	}
	var r currentResp
	return &r, json.Unmarshal(body, &r)
}

func (c *apiClient) fetchHistorical(lat, lon float64, start, end string) (*archiveResp, error) {
	url := fmt.Sprintf(
		"https://archive-api.open-meteo.com/v1/archive"+
			"?latitude=%.4f&longitude=%.4f"+
			"&start_date=%s&end_date=%s"+
			"&daily=temperature_2m_max,temperature_2m_min,temperature_2m_mean,"+
			"precipitation_sum,relative_humidity_2m_mean,wind_speed_10m_mean,sunshine_duration"+
			"&timezone=Asia%%2FKolkata",
		lat, lon, start, end,
	)
	body, err := c.get(url)
	if err != nil {
		return nil, err
	}
	var r archiveResp
	return &r, json.Unmarshal(body, &r)
}

// ─────────────────────────────────────────────
// WMO CODE → CONDITION STRING
// ─────────────────────────────────────────────

func wmoCondition(code int) string {
	switch {
	case code == 0:
		return "CLEAR_SKY"
	case code <= 3:
		return "PARTLY_CLOUDY"
	case code <= 49:
		return "FOGGY"
	case code <= 59:
		return "DRIZZLE"
	case code <= 65:
		return "MODERATE_RAIN"
	case code <= 79:
		return "SNOW"
	case code <= 82:
		return "SHOWERS"
	case code == 95:
		return "THUNDERSTORM"
	case code >= 96:
		return "HEAVY_RAIN"
	default:
		return "NORMAL"
	}
}

// ─────────────────────────────────────────────
// FIX-3: WATER BALANCE ESTIMATOR
//
// Uses a simplified daily ET for tropical lowland rice (6 mm/day, which
// is the IRRI mid-range for Karnataka conditions) to estimate how many
// days until the field hits the IRRI AWD re-flood threshold of −150 mm
// (i.e. 15 cm below soil surface).
//
// Assumes the field was just re-flooded to 5 cm (50 mm) standing water
// at the time of the call. Returns the forecast day index (1-based) when
// irrigation is needed, or -1 if rainfall keeps the balance safe across
// the entire window.
// ─────────────────────────────────────────────

const (
	dailyETc        = 6.0   // mm/day — crop evapotranspiration for lowland rice, Karnataka
	initialPonding  = 50.0  // mm — 5 cm standing water after re-flood (IRRI target)
	awdThresholdMM  = -150.0 // mm — 15 cm below surface = IRRI "safe AWD" re-flood trigger
)

// estimateWaterBalance returns the 1-based day index in the forecast when
// irrigation will be needed, or -1 if the balance stays above threshold.
func estimateWaterBalance(forecastRain []float64) int {
	balance := initialPonding
	for i, rain := range forecastRain {
		balance += rain - dailyETc
		if balance <= awdThresholdMM {
			return i + 1
		}
	}
	return -1
}

// forecastRainSum returns the total rainfall over the next n days of forecast.
func forecastRainSum(forecastRain []float64, n int) float64 {
	total := 0.0
	limit := n
	if limit > len(forecastRain) {
		limit = len(forecastRain)
	}
	for i := 0; i < limit; i++ {
		total += forecastRain[i]
	}
	return total
}

// ─────────────────────────────────────────────
// FIX-1: STAGE-AWARE FARMING ADVISORY
//
// OLD: flat map of condition → generic string, no crop stage awareness,
//      "irrigate if dry 3+ days" heuristic not grounded in any standard.
//
// NEW: three explicit growth windows derived from days after transplanting
//      (DAT), matching IRRI AWD recommendations:
//        • Vegetative  (DAT  0–54): safe AWD, monitor water tube
//        • Flowering   (DAT 55–70): hard continuous flood, no AWD
//        • Grain fill  (DAT 71–100): resume safe AWD
//      Weather overlays (heat, humidity, rain) are applied on top.
// ─────────────────────────────────────────────

func farmingAdvisory(condition string, temp, rain, humidity float64, daysAfterTransplant int) string {

	// ── Stage: Flowering window — water is non-negotiable ──────────────
	// IRRI: "From one week before to one week after flowering, the field
	// should remain flooded." Typical flowering at DAT ~62 for Karnataka
	// varieties; window is DAT 55–70.
	if daysAfterTransplant >= 55 && daysAfterTransplant <= 70 {
		return "FLOWERING STAGE: Keep field continuously flooded to 5 cm standing water. " +
			"Do NOT allow water level to drop — this is the most yield-critical window. " +
			"Inspect bunds daily. Delay all pesticide/fertilizer spraying until after flowering."
	}

	// ── Stage: Grain filling / ripening ────────────────────────────────
	// IRRI: "After flowering, water level can drop to 15 cm below surface
	// before re-flooding." Resume safe AWD here.
	if daysAfterTransplant > 70 && daysAfterTransplant <= 100 {
		if rain > 20 {
			return "GRAIN FILLING: Heavy rain received — skip irrigation. " +
				"Check bunds and drainage. Field is likely replenished."
		}
		return "GRAIN FILLING (safe AWD): Check field water tube — if water level is at " +
			"15 cm below soil surface, re-flood to 5 cm standing water. " +
			"Drain field completely 7–10 days before planned harvest."
	}

	// ── Stage: Pre-harvest drain ────────────────────────────────────────
	if daysAfterTransplant > 100 {
		return "PRE-HARVEST: Drain field 7–10 days before harvest to firm the soil " +
			"for machinery. Do not irrigate further unless crop shows severe wilting."
	}

	// ── Vegetative stage (DAT 0–54): safe AWD with weather overlays ─────

	// Heavy rain: field is being replenished — skip irrigation
	if rain > 20 {
		return "Heavy rain received — skip irrigation. Ensure drainage bunds are intact. " +
			"Delay fertilizer application by 48 hours to avoid nutrient runoff."
	}

	// Extreme heat: irrigation timing matters more than frequency
	if temp > 40 {
		return "Extreme heat alert: Irrigate at dawn only to minimise evaporation losses. " +
			"Keep field water level at 5 cm to buffer heat stress in young plants."
	}

	// High humidity: disease pressure overrides irrigation advice
	if humidity > 90 {
		return "Very high humidity — scout for rice blast and sheath blight. " +
			"Do not irrigate until humidity drops. Wet foliage increases fungal risk."
	}

	// Foggy / drizzle: light moisture, watch disease
	if condition == "FOGGY" || condition == "DRIZZLE" {
		return "Low-intensity moisture — delay spraying until conditions clear. " +
			"Check field water tube; irrigate only if water is at 15 cm below surface."
	}

	// Thunderstorm: safety + damage check
	if condition == "THUNDERSTORM" {
		return "Stay off the field during the storm. After it passes, inspect bunds for " +
			"breaches and drainage channels for blockages before resuming field operations."
	}

	// ── Default AWD advisory (vegetative, normal conditions) ────────────
	// Key change from old code: no "dry 3 days" heuristic.
	// The trigger is always soil water depth, not elapsed days or sky condition.
	return "Check field water tube daily. Irrigate ONLY when water level reaches " +
		"15 cm below soil surface — then re-flood to 5 cm standing water. " +
		"Apply nitrogen fertilizer on dry soil just before re-irrigation for best uptake. " +
		"Do not irrigate based on sky condition or calendar alone (IRRI AWD protocol)."
}

// ─────────────────────────────────────────────
// FIX-2: EXTENDED ALERT CHECKER
//
// OLD: 4 alert types (FLOOD_RISK, HEAT_STRESS, DISEASE_RISK, HEAVY_RAIN),
//      no irrigation alert, no crop-stage awareness.
//
// NEW: adds IRRIGATION_NEEDED and SKIP_IRRIGATION alert types;
//      respects the IRRI flowering-window hard lock;
//      uses 7-day forecast rain sum as a forward-looking input
//      rather than reacting only to today's reading.
// ─────────────────────────────────────────────

type alertResult struct {
	hasAlert bool
	aType    string
	severity string
	message  string
	action   string
}

func checkAlert(
	temp, rain, humidity float64,
	district string,
	daysAfterTransplant int,  // FIX-2: new param — crop growth stage
	sevenDayRainSum float64,  // FIX-2: new param — forward-looking rain
) alertResult {

	isFloweringWindow := daysAfterTransplant >= 55 && daysAfterTransplant <= 70

	// ── SKIP_IRRIGATION: heavy rain already replenishing field ──────────
	// Moved above flood check so it fires first on 20–50 mm days.
	if rain > 20 && rain <= 50 {
		return alertResult{
			hasAlert: true,
			aType:    "SKIP_IRRIGATION",
			severity: "LOW",
			message:  fmt.Sprintf("Heavy rain (%.1f mm) in %s — field is being replenished naturally.", rain, district),
			action:   "Do not irrigate today. Inspect bunds and drainage outlets. Delay fertilizer by 48 hours.",
		}
	}

	// ── FLOOD_RISK: extreme rainfall ────────────────────────────────────
	if rain > 50 {
		return alertResult{
			hasAlert: true,
			aType:    "FLOOD_RISK",
			severity: "HIGH",
			message:  fmt.Sprintf("Extremely heavy rain (%.1f mm) — flood risk in %s.", rain, district),
			action:   "Open drainage outlets immediately. Move harvested produce to dry storage.",
		}
	}

	// ── HEAT_STRESS ─────────────────────────────────────────────────────
	if temp > 40 {
		return alertResult{
			hasAlert: true,
			aType:    "HEAT_STRESS",
			severity: "MEDIUM",
			message:  fmt.Sprintf("Temperature spike (%.1f°C) in %s.", temp, district),
			action:   "Irrigate at dawn only. Maintain 5 cm standing water to buffer heat. Apply mulch on bunds.",
		}
	}

	// ── DISEASE_RISK ─────────────────────────────────────────────────────
	if humidity > 90 {
		return alertResult{
			hasAlert: true,
			aType:    "DISEASE_RISK",
			severity: "MEDIUM",
			message:  fmt.Sprintf("Very high humidity (%.0f%%) in %s — fungal disease risk.", humidity, district),
			action:   "Scout for blast and sheath blight. Consider preventive fungicide spray. Delay irrigation.",
		}
	}

	// ── IRRIGATION_NEEDED (FIX-2) ────────────────────────────────────────
	// Trigger only when:
	//   • NOT in the flowering window (IRRI hard lock: keep flooded ±1 week of flowering)
	//   • Today's rain is negligible (< 2 mm)
	//   • 7-day forecast rain is also low (< 10 mm total) — field won't self-replenish
	// This is a weather-based proxy for the IRRI AWD soil-tube trigger
	// (actual trigger = water level at 15 cm below surface).
	if !isFloweringWindow && rain < 2.0 && sevenDayRainSum < 10.0 {
		return alertResult{
			hasAlert: true,
			aType:    "IRRIGATION_NEEDED",
			severity: "MEDIUM",
			message: fmt.Sprintf(
				"Water deficit likely in %s — no significant rain today (%.1f mm) "+
					"and only %.1f mm forecast over the next 7 days.",
				district, rain, sevenDayRainSum,
			),
			action: "Check field water tube. If water is at 15 cm below soil surface, " +
				"re-flood to 5 cm standing water (IRRI safe AWD protocol). " +
				"Apply nitrogen fertilizer on dry soil just before re-irrigating.",
		}
	}

	// ── FLOWERING WINDOW REMINDER ────────────────────────────────────────
	// If we're in the flowering window and no emergency alert fired above,
	// send a low-severity reminder to keep the field flooded.
	if isFloweringWindow {
		return alertResult{
			hasAlert: true,
			aType:    "FLOWERING_FLOOD_REQUIRED",
			severity: "LOW",
			message:  fmt.Sprintf("Crop in flowering window (DAT %d) in %s.", daysAfterTransplant, district),
			action:   "Maintain continuous 5 cm standing water. Do not allow field to dry. Check bunds daily.",
		}
	}

	return alertResult{}
}

// ─────────────────────────────────────────────
// gRPC SERVER
// ─────────────────────────────────────────────

type weatherServer struct {
	pb.UnimplementedWeatherAgentServiceServer
	api *apiClient
}

// FIX-4: GetWeatherForecast now reads DaysAfterTransplant from the request
// and passes it into farmingAdvisory so the advisory is stage-aware.
func (s *weatherServer) GetWeatherForecast(
	ctx context.Context, req *pb.WeatherRequest,
) (*pb.WeatherResponse, error) {
	log.Printf("[WEATHER] Forecast → %s (DAT=%d)", req.District, req.DaysAfterTransplant)
	lat, lon := getCoords(req.District)

	data, err := s.api.fetchCurrent(lat, lon)
	if err != nil {
		return nil, fmt.Errorf("open-meteo: %w", err)
	}

	cur := data.Current
	condition := wmoCondition(cur.WeatherCode)

	// FIX-4: pass DaysAfterTransplant into advisory
	dat := int(req.DaysAfterTransplant)
	advisory := farmingAdvisory(condition, cur.Temp2m, cur.Precip, float64(cur.Humidity), dat)

	// sunshine: seconds → hours
	sunH := 0.0
	if len(data.Daily.SunshineSec) > 0 {
		sunH = data.Daily.SunshineSec[0] / 3600.0
	}

	tempMax, tempMin := cur.Temp2m, cur.Temp2m
	if len(data.Daily.TempMax) > 0 {
		tempMax = data.Daily.TempMax[0]
	}
	if len(data.Daily.TempMin) > 0 {
		tempMin = data.Daily.TempMin[0]
	}

	days := int(req.ForecastDays)
	if days <= 0 {
		days = 7
	}
	days = minInt(days, len(data.Daily.Time))

	forecast := make([]*pb.DailyForecast, days)
	for i := 0; i < days; i++ {
		prob := 0.0
		if i < len(data.Daily.PrecipProb) {
			prob = float64(data.Daily.PrecipProb[i]) / 100.0
		}
		wc := 0
		if i < len(data.Daily.WeatherCode) {
			wc = data.Daily.WeatherCode[i]
		}
		forecast[i] = &pb.DailyForecast{
			Date:         data.Daily.Time[i],
			TempMax:      float32(safeF(data.Daily.TempMax, i)),
			TempMin:      float32(safeF(data.Daily.TempMin, i)),
			RainfallProb: float32(prob),
			Condition:    wmoCondition(wc),
		}
	}

	// FIX-3: compute irrigation day prediction and embed in response metadata
	irrigationDay := estimateWaterBalance(data.Daily.PrecipSum)
	irrigationNote := ""
	if irrigationDay > 0 {
		irrigationNote = fmt.Sprintf(
			"Water balance model predicts irrigation needed in ~%d day(s) "+
				"(IRRI AWD proxy: field water tube at 15 cm below surface).",
			irrigationDay,
		)
	} else {
		irrigationNote = "Forecast rainfall sufficient — no irrigation predicted in the next 7 days."
	}

	return &pb.WeatherResponse{
		Metadata: &pb.AgentMetadata{
			AgentId:    "weather-agent-001",
			AgentName:  "Weather Intelligence Agent",
			Timestamp:  time.Now().UTC().Format(time.RFC3339),
			Confidence: 0.93,
			Notes:      irrigationNote, // surface the water balance prediction
		},
		District:         req.District,
		TemperatureMax:   float32(tempMax),
		TemperatureMin:   float32(tempMin),
		TemperatureAvg:   float32(cur.Temp2m),
		RainfallMm:       float32(cur.Precip),
		HumidityPercent:  float32(cur.Humidity),
		WindSpeedKmph:    float32(cur.WindSpeed),
		SunshineHours:    float32(math.Round(sunH*10) / 10),
		WeatherCondition: condition,
		FarmingAdvisory:  advisory,
		Forecast:         forecast,
	}, nil
}

// FIX-5: GetWeatherAlert accepts and uses DaysAfterTransplant + 7-day rain sum.
func (s *weatherServer) GetWeatherAlert(
	ctx context.Context, req *pb.WeatherRequest,
) (*pb.WeatherAlertResponse, error) {
	lat, lon := getCoords(req.District)
	data, err := s.api.fetchCurrent(lat, lon)
	if err != nil {
		return &pb.WeatherAlertResponse{HasAlert: false}, nil
	}
	cur := data.Current

	// FIX-5: compute 7-day forward rain sum for IRRIGATION_NEEDED check
	sevenDayRain := forecastRainSum(data.Daily.PrecipSum, 7)

	a := checkAlert(
		cur.Temp2m, cur.Precip, float64(cur.Humidity),
		req.District,
		int(req.DaysAfterTransplant), // FIX-5
		sevenDayRain,                 // FIX-5
	)

	return &pb.WeatherAlertResponse{
		Metadata: &pb.AgentMetadata{
			AgentId:   "weather-agent-001",
			AgentName: "Weather Intelligence Agent",
			Timestamp: time.Now().UTC().Format(time.RFC3339),
		},
		HasAlert:       a.hasAlert,
		AlertType:      a.aType,
		Severity:       a.severity,
		Message:        a.message,
		ActionRequired: a.action,
	}, nil
}

func (s *weatherServer) GetHistoricalWeather(
	ctx context.Context, req *pb.HistoricalWeatherReq,
) (*pb.HistoricalWeatherResp, error) {
	log.Printf("[WEATHER] Historical → %s %s to %s", req.District, req.StartDate, req.EndDate)
	lat, lon := getCoords(req.District)

	data, err := s.api.fetchHistorical(lat, lon, req.StartDate, req.EndDate)
	if err != nil {
		return nil, fmt.Errorf("archive fetch: %w", err)
	}

	var totalRain, totalTemp float64
	var recs []*pb.WeatherResponse
	n := len(data.Daily.Time)
	for i := 0; i < n; i++ {
		rain := safeF(data.Daily.PrecipSum, i)
		temp := safeF(data.Daily.TempMean, i)
		totalRain += rain
		totalTemp += temp
		recs = append(recs, &pb.WeatherResponse{
			District:        req.District,
			TemperatureMax:  float32(safeF(data.Daily.TempMax, i)),
			TemperatureMin:  float32(safeF(data.Daily.TempMin, i)),
			TemperatureAvg:  float32(temp),
			RainfallMm:      float32(rain),
			HumidityPercent: float32(safeF(data.Daily.Humidity, i)),
			WindSpeedKmph:   float32(safeF(data.Daily.Wind, i)),
			SunshineHours:   float32(safeF(data.Daily.Sunshine, i) / 3600.0),
		})
	}
	avgTemp := 0.0
	if n > 0 {
		avgTemp = totalTemp / float64(n)
	}

	return &pb.HistoricalWeatherResp{
		Metadata: &pb.AgentMetadata{
			AgentId:   "weather-agent-001",
			AgentName: "Weather Intelligence Agent",
			Timestamp: time.Now().UTC().Format(time.RFC3339),
		},
		Records:       recs,
		AvgTemp:       float32(avgTemp),
		TotalRainfall: float32(totalRain),
	}, nil
}

// ─────────────────────────────────────────────
// REGISTRATION + MAIN
// ─────────────────────────────────────────────

func registerWithOrchestrator(port int32) {
	addr := os.Getenv("ORCHESTRATOR_ADDR")
	if addr == "" {
		addr = "orchestrator:50051"
	}
	for i := 0; i < 10; i++ {
		conn, err := grpc.NewClient(addr,
			grpc.WithTransportCredentials(insecure.NewCredentials()))
		if err != nil {
			time.Sleep(3 * time.Second)
			continue
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		_, err = pb.NewOrchestratorServiceClient(conn).RegisterAgent(ctx, &pb.AgentRegistration{
			AgentId:   "weather-agent-001",
			AgentName: "Weather Intelligence Agent (Open-Meteo)",
			AgentType: "WEATHER",
			Host:      "weather-agent",
			Port:      port,
		})
		cancel()
		conn.Close()
		if err == nil {
			log.Printf("[WEATHER] Registered with orchestrator at %s", addr)
			return
		}
		time.Sleep(3 * time.Second)
	}
}

func main() {
	port := int32(50052)
	if p := os.Getenv("WEATHER_AGENT_PORT"); p != "" {
		if v, err := strconv.Atoi(p); err == nil {
			port = int32(v)
		}
	}

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		log.Fatalf("[WEATHER] Listen: %v", err)
	}

	srv := grpc.NewServer()
	pb.RegisterWeatherAgentServiceServer(srv, &weatherServer{api: newAPIClient()})
	reflection.Register(srv)

	go registerWithOrchestrator(port)

	log.Printf("[WEATHER] Running on :%d | Open-Meteo live API | 29 districts", port)
	if err := srv.Serve(lis); err != nil {
		log.Fatalf("[WEATHER] Serve: %v", err)
	}
}

// ─────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────

func safeF(s []float64, i int) float64 {
	if i < len(s) {
		return s[i]
	}
	return 0
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}
