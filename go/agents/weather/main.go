// go/agents/weather/main.go
// OWNER: Aditya
// Weather Intelligence Agent — powered by Open-Meteo (free, no API key)
// API docs: https://open-meteo.com/en/docs
// Covers all 29 Karnataka paddy districts with real-time + 16-day forecast

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
	case code == 0:           return "CLEAR_SKY"
	case code <= 3:           return "PARTLY_CLOUDY"
	case code <= 49:          return "FOGGY"
	case code <= 59:          return "DRIZZLE"
	case code <= 65:          return "MODERATE_RAIN"
	case code <= 79:          return "SNOW"
	case code <= 82:          return "SHOWERS"
	case code == 95:          return "THUNDERSTORM"
	case code >= 96:          return "HEAVY_RAIN"
	default:                  return "NORMAL"
	}
}

// ─────────────────────────────────────────────
// FARMING ADVISORY + ALERTS
// ─────────────────────────────────────────────

func farmingAdvisory(condition string, temp, rain, humidity float64) string {
	m := map[string]string{
		"CLEAR_SKY":     "Good conditions for all field operations. Schedule irrigation if dry 3+ days.",
		"PARTLY_CLOUDY": "Suitable for regular farm operations including spraying.",
		"FOGGY":         "High humidity — watch for fungal diseases. Delay spraying until fog clears.",
		"DRIZZLE":       "Light rain — avoid fertilizer application. Check disease pressure.",
		"MODERATE_RAIN": "Good for transplanting. Ensure field bunds are intact.",
		"SHOWERS":       "Light showers. Safe for most operations. Monitor soil moisture.",
		"HEAVY_RAIN":    "Avoid field operations. Check drainage channels. Delay fertilizer 48h.",
		"THUNDERSTORM":  "Stay off the field. Secure farm equipment. Check damage after storm.",
		"NORMAL":        "Conditions suitable for regular field operations.",
	}
	if temp > 40 {
		return "Extreme heat — irrigate at dawn only. Watch for heat stress in young plants."
	}
	if humidity > 90 {
		return "Very high humidity — scout for blast and sheath blight. Consider fungicide."
	}
	if a, ok := m[condition]; ok {
		return a
	}
	return "Monitor field conditions. Consult local Krishi Vigyan Kendra if uncertain."
}

type alertResult struct {
	hasAlert bool
	aType    string
	severity string
	message  string
	action   string
}

func checkAlert(temp, rain, humidity float64, district string) alertResult {
	switch {
	case rain > 50:
		return alertResult{true, "FLOOD_RISK", "HIGH",
			fmt.Sprintf("Extremely heavy rain (%.1fmm) — flood risk in %s", rain, district),
			"Open drainage outlets immediately. Move harvested produce to dry storage."}
	case temp > 40:
		return alertResult{true, "HEAT_STRESS", "MEDIUM",
			fmt.Sprintf("Temperature spike (%.1f°C) in %s", temp, district),
			"Irrigate early morning. Apply mulch to reduce soil temperature."}
	case humidity > 90:
		return alertResult{true, "DISEASE_RISK", "MEDIUM",
			fmt.Sprintf("Very high humidity (%.0f%%) in %s — fungal disease risk", humidity, district),
			"Scout for blast and sheath blight. Consider preventive fungicide spray."}
	case rain > 20:
		return alertResult{true, "HEAVY_RAIN", "LOW",
			fmt.Sprintf("Heavy rain (%.1fmm) in %s", rain, district),
			"Ensure field drainage is working. Delay fertilizer application."}
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

func (s *weatherServer) GetWeatherForecast(
	ctx context.Context, req *pb.WeatherRequest,
) (*pb.WeatherResponse, error) {
	log.Printf("[WEATHER] Forecast → %s", req.District)
	lat, lon := getCoords(req.District)

	data, err := s.api.fetchCurrent(lat, lon)
	if err != nil {
		return nil, fmt.Errorf("open-meteo: %w", err)
	}

	cur := data.Current
	condition := wmoCondition(cur.WeatherCode)
	advisory := farmingAdvisory(condition, cur.Temp2m, cur.Precip, float64(cur.Humidity))

	// sunshine: seconds → hours
	sunH := 0.0
	if len(data.Daily.SunshineSec) > 0 {
		sunH = data.Daily.SunshineSec[0] / 3600.0
	}

	tempMax, tempMin := cur.Temp2m, cur.Temp2m
	if len(data.Daily.TempMax) > 0 { tempMax = data.Daily.TempMax[0] }
	if len(data.Daily.TempMin) > 0 { tempMin = data.Daily.TempMin[0] }

	days := int(req.ForecastDays)
	if days <= 0 { days = 7 }
	days = minInt(days, len(data.Daily.Time))

	forecast := make([]*pb.DailyForecast, days)
	for i := 0; i < days; i++ {
		prob := 0.0
		if i < len(data.Daily.PrecipProb) {
			prob = float64(data.Daily.PrecipProb[i]) / 100.0
		}
		wc := 0
		if i < len(data.Daily.WeatherCode) { wc = data.Daily.WeatherCode[i] }
		forecast[i] = &pb.DailyForecast{
			Date:         data.Daily.Time[i],
			TempMax:      float32(safeF(data.Daily.TempMax, i)),
			TempMin:      float32(safeF(data.Daily.TempMin, i)),
			RainfallProb: float32(prob),
			Condition:    wmoCondition(wc),
		}
	}

	return &pb.WeatherResponse{
		Metadata: &pb.AgentMetadata{
			AgentId:    "weather-agent-001",
			AgentName:  "Weather Intelligence Agent",
			Timestamp:  time.Now().UTC().Format(time.RFC3339),
			Confidence: 0.93,
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

func (s *weatherServer) GetWeatherAlert(
	ctx context.Context, req *pb.WeatherRequest,
) (*pb.WeatherAlertResponse, error) {
	lat, lon := getCoords(req.District)
	data, err := s.api.fetchCurrent(lat, lon)
	if err != nil {
		return &pb.WeatherAlertResponse{HasAlert: false}, nil
	}
	cur := data.Current
	a := checkAlert(cur.Temp2m, cur.Precip, float64(cur.Humidity), req.District)

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
	if n > 0 { avgTemp = totalTemp / float64(n) }

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
	if addr == "" { addr = "orchestrator:50051" }
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
		if v, err := strconv.Atoi(p); err == nil { port = int32(v) }
	}

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil { log.Fatalf("[WEATHER] Listen: %v", err) }

	srv := grpc.NewServer()
	pb.RegisterWeatherAgentServiceServer(srv, &weatherServer{api: newAPIClient()})
	reflection.Register(srv)

	go registerWithOrchestrator(port)

	log.Printf("[WEATHER] Running on :%d | Open-Meteo live API | 29 districts", port)
	if err := srv.Serve(lis); err != nil { log.Fatalf("[WEATHER] Serve: %v", err) }
}

func safeF(s []float64, i int) float64 {
	if i < len(s) { return s[i] }
	return 0
}
func minInt(a, b int) int {
	if a < b { return a }
	return b
}
