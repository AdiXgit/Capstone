package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net"
	"os"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/reflection"

	pb "github.com/AdiXgit/Capstone/proto"
)

// ─────────────────────────────────────────────
// ORCHESTRATOR SERVER
// ─────────────────────────────────────────────

type OrchestratorServer struct {
	pb.UnimplementedOrchestratorServiceServer
	mu               sync.RWMutex
	registeredAgents map[string]*AgentConnection
	queryCounter     int32
}

type AgentConnection struct {
	AgentID   string
	AgentName string
	AgentType string
	Host      string
	Port      int32
	Client    interface{} // holds the gRPC client stub
}

func NewOrchestratorServer() *OrchestratorServer {
	return &OrchestratorServer{
		registeredAgents: make(map[string]*AgentConnection),
	}
}

// RegisterAgent — agents call this on startup to announce themselves
func (s *OrchestratorServer) RegisterAgent(
	ctx context.Context, req *pb.AgentRegistration,
) (*pb.RegistrationAck, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	s.registeredAgents[req.AgentId] = &AgentConnection{
		AgentID:   req.AgentId,
		AgentName: req.AgentName,
		AgentType: req.AgentType,
		Host:      req.Host,
		Port:      req.Port,
	}
	log.Printf("[ORCHESTRATOR] Agent registered: %s (%s) at %s:%d",
		req.AgentName, req.AgentType, req.Host, req.Port)

	return &pb.RegistrationAck{Success: true, Message: "Registered successfully"}, nil
}

// GetSystemStatus — health check for dashboard
func (s *OrchestratorServer) GetSystemStatus(
	ctx context.Context, req *pb.StatusRequest,
) (*pb.SystemStatusResponse, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	agents := make([]string, 0, len(s.registeredAgents))
	for _, a := range s.registeredAgents {
		agents = append(agents, a.AgentName)
	}

	return &pb.SystemStatusResponse{
		OrchestratorStatus:  "HEALTHY",
		ActiveAgents:        agents,
		TotalQueriesHandled: s.queryCounter,
	}, nil
}

// RouteQuery — fan-out to all relevant agents concurrently, collect + synthesize results
func (s *OrchestratorServer) RouteQuery(
	ctx context.Context, req *pb.FarmerQuery,
) (*pb.OrchestratorResponse, error) {
	log.Printf("[ORCHESTRATOR] Routing query %s → type=%s district=%s DAT=%d",
		req.QueryId, req.QueryType, req.District, req.DaysAfterTransplant)

	s.mu.Lock()
	s.queryCounter++
	s.mu.Unlock()

	targetAgents := s.resolveAgents(req.QueryType)

	type agentWork struct {
		agentType string
		result    *pb.AgentResult
	}

	resultCh := make(chan agentWork, len(targetAgents))
	var wg sync.WaitGroup

	// ── Concurrent fan-out ────────────────────────────────────────────────
	for _, agentType := range targetAgents {
		wg.Add(1)
		go func(aType string) {
			defer wg.Done()
			result := s.callAgent(ctx, aType, req)
			resultCh <- agentWork{agentType: aType, result: result}
		}(agentType)
	}

	go func() {
		wg.Wait()
		close(resultCh)
	}()

	// ── Collect results ───────────────────────────────────────────────────
	var agentResults []*pb.AgentResult
	var totalConfidence float32
	for work := range resultCh {
		agentResults = append(agentResults, work.result)
		totalConfidence += work.result.Confidence
	}

	avgConf := float32(0)
	if len(agentResults) > 0 {
		avgConf = totalConfidence / float32(len(agentResults))
	}

	// FIX: synthesizeRecommendation now reads actual agent payload fields
	// instead of just printing "[AgentName: ready]" for every result.
	recommendation := s.synthesizeRecommendation(agentResults, req)

	return &pb.OrchestratorResponse{
		QueryId:           req.QueryId,
		Recommendation:    recommendation,
		AgentResults:      agentResults,
		OverallConfidence: avgConf,
		Timestamp:         time.Now().UTC().Format(time.RFC3339),
	}, nil
}

// resolveAgents maps query type → which agent types to fan-out to.
// FIX: added IRRIGATION query type to route to WEATHER + SOIL agents,
// consistent with the new IRRIGATION_NEEDED / SKIP_IRRIGATION alert types
// introduced in the weather agent.
func (s *OrchestratorServer) resolveAgents(queryType string) []string {
	routing := map[string][]string{
		"WEATHER":     {"WEATHER"},
		"SOIL":        {"SOIL"},
		"CROP_HEALTH": {"CROP_HEALTH", "WEATHER"},
		"FERTILIZER":  {"SOIL", "WEATHER", "CROP_HEALTH"},
		"IRRIGATION":  {"WEATHER", "SOIL"},                 // FIX: new — matches weather agent AWD alerts
		"FULL":        {"WEATHER", "SOIL", "CROP_HEALTH"},
	}
	if agents, ok := routing[queryType]; ok {
		return agents
	}
	// Default: fan out to all three core agents
	return []string{"WEATHER", "SOIL", "CROP_HEALTH"}
}

// callAgent establishes a gRPC call to the appropriate agent and returns
// a standardised AgentResult.
//
// NOTE: a new gRPC connection is created per call for simplicity.
// Production systems should maintain a pool of cached, reused connections.
func (s *OrchestratorServer) callAgent(
	ctx context.Context, agentType string, req *pb.FarmerQuery,
) *pb.AgentResult {
	s.mu.RLock()
	var conn *AgentConnection
	for _, a := range s.registeredAgents {
		if a.AgentType == agentType {
			conn = a
			break
		}
	}
	s.mu.RUnlock()

	// Fall back to env-configured addresses if agent hasn't registered yet
	if conn == nil {
		conn = &AgentConnection{
			AgentType: agentType,
			Host:      agentHostFromEnv(agentType),
			Port:      agentPortFromEnv(agentType),
		}
	}

	addr := fmt.Sprintf("%s:%d", conn.Host, conn.Port)
	grpcConn, err := grpc.NewClient(addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return errorResult(agentType+"_AGENT", err)
	}
	defer grpcConn.Close()

	callCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	switch agentType {

	case "WEATHER":
		client := pb.NewWeatherAgentServiceClient(grpcConn)
		// FIX (REQUIRED): pass DaysAfterTransplant from the farmer query through
		// to the weather agent. Without this, all the stage-aware advisory logic
		// (flowering lock, grain-fill AWD, irrigation prediction) introduced in
		// the weather agent fix defaults to DAT=0 and always hits the vegetative
		// path — making FIX-1, FIX-2, FIX-4, and FIX-5 in the weather agent
		// completely inert.
		resp, err := client.GetWeatherForecast(callCtx, &pb.WeatherRequest{
			District:            req.District,
			Date:                time.Now().Format("2006-01-02"),
			ForecastDays:        7,
			DaysAfterTransplant: req.DaysAfterTransplant, // FIX: was missing
		})
		if err != nil {
			return errorResult("WEATHER_AGENT", err)
		}
		return marshalResult("Weather Agent", resp, 0.88)

	case "SOIL":
		client := pb.NewSoilAgentServiceClient(grpcConn)
		resp, err := client.GetSoilHealth(callCtx, &pb.SoilRequest{
			District: req.District,
			Season:   req.Season,
		})
		if err != nil {
			return errorResult("SOIL_AGENT", err)
		}
		return marshalResult("Soil Agent", resp, 0.85)

	case "CROP_HEALTH":
		client := pb.NewCropHealthAgentServiceClient(grpcConn)
		resp, err := client.GetCropHealthStatus(callCtx, &pb.CropHealthRequest{
			District: req.District,
			Season:   req.Season,
			Year:     int32(time.Now().Year()),
		})
		if err != nil {
			return errorResult("CROP_HEALTH_AGENT", err)
		}
		return marshalResult("Crop Health Agent", resp, 0.82)
	}

	return &pb.AgentResult{Status: "UNKNOWN_AGENT"}
}

// synthesizeRecommendation builds a human-readable summary from all agent
// results by reading the actual advisory / recommendation fields out of
// each agent's JSON payload.
//
// FIX: old version was a stub that only printed "[AgentName: ready]" for
// every result and never inspected the actual data returned by the agents.
// New version parses each result's JSON and surfaces the advisory text,
// alert type, and recommendation where present.
func (s *OrchestratorServer) synthesizeRecommendation(
	results []*pb.AgentResult, req *pb.FarmerQuery,
) string {
	summary := fmt.Sprintf(
		"Advisory for farmer %s | District: %s | Season: %s | DAT: %d\n",
		req.FarmerId, req.District, req.Season, req.DaysAfterTransplant,
	)
	summary += "─────────────────────────────────────────\n"

	for _, r := range results {
		if r.Status != "OK" {
			summary += fmt.Sprintf("• %s: unavailable (status=%s)\n", r.AgentName, r.Status)
			continue
		}

		var payload map[string]interface{}
		if err := json.Unmarshal([]byte(r.ResultJson), &payload); err != nil {
			summary += fmt.Sprintf("• %s: could not parse response\n", r.AgentName)
			continue
		}

		line := fmt.Sprintf("• %s:\n", r.AgentName)

		// Weather agent fields
		if advisory, ok := payload["farming_advisory"].(string); ok && advisory != "" {
			line += fmt.Sprintf("    Advisory : %s\n", advisory)
		}
		if condition, ok := payload["weather_condition"].(string); ok && condition != "" {
			line += fmt.Sprintf("    Condition: %s\n", condition)
		}
		if rain, ok := payload["rainfall_mm"].(float64); ok {
			line += fmt.Sprintf("    Rainfall : %.1f mm today\n", rain)
		}

		// Metadata notes (carries our irrigation water-balance prediction)
		if meta, ok := payload["metadata"].(map[string]interface{}); ok {
			if notes, ok := meta["notes"].(string); ok && notes != "" {
				line += fmt.Sprintf("    Note     : %s\n", notes)
			}
		}

		// Soil agent fields
		if rec, ok := payload["recommendation"].(string); ok && rec != "" {
			line += fmt.Sprintf("    Recommendation: %s\n", rec)
		}
		if health, ok := payload["soil_health_score"].(float64); ok {
			line += fmt.Sprintf("    Soil Health Score: %.1f/10\n", health)
		}

		// Crop health agent fields
		if status, ok := payload["crop_status"].(string); ok && status != "" {
			line += fmt.Sprintf("    Crop Status: %s\n", status)
		}
		if disease, ok := payload["disease_risk"].(string); ok && disease != "" {
			line += fmt.Sprintf("    Disease Risk: %s\n", disease)
		}

		summary += line
	}

	summary += "─────────────────────────────────────────\n"
	summary += fmt.Sprintf("Overall confidence: %.0f%%\n",
		averageConfidence(results)*100)

	return summary
}

// ─────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────

func averageConfidence(results []*pb.AgentResult) float32 {
	if len(results) == 0 {
		return 0
	}
	var total float32
	for _, r := range results {
		total += r.Confidence
	}
	return total / float32(len(results))
}

func marshalResult(name string, v interface{}, conf float32) *pb.AgentResult {
	b, _ := json.Marshal(v)
	return &pb.AgentResult{
		AgentName:  name,
		ResultJson: string(b),
		Confidence: conf,
		Status:     "OK",
	}
}

func errorResult(name string, err error) *pb.AgentResult {
	return &pb.AgentResult{
		AgentName:  name,
		ResultJson: fmt.Sprintf(`{"error":"%s"}`, err.Error()),
		Confidence: 0,
		Status:     "ERROR",
	}
}

func agentHostFromEnv(agentType string) string {
	envKey := agentType + "_AGENT_HOST"
	if h := os.Getenv(envKey); h != "" {
		return h
	}
	defaults := map[string]string{
		"WEATHER":     "weather-agent",
		"SOIL":        "soil-agent",
		"CROP_HEALTH": "crop-health-agent",
	}
	if d, ok := defaults[agentType]; ok {
		return d
	}
	return "localhost"
}

func agentPortFromEnv(agentType string) int32 {
	defaults := map[string]int32{
		"WEATHER":     50052,
		"SOIL":        50053,
		"CROP_HEALTH": 50054,
	}
	if p, ok := defaults[agentType]; ok {
		return p
	}
	return 50060
}

// ─────────────────────────────────────────────
// MAIN
// ─────────────────────────────────────────────

func main() {
	port := os.Getenv("ORCHESTRATOR_PORT")
	if port == "" {
		port = "50051"
	}

	lis, err := net.Listen("tcp", ":"+port)
	if err != nil {
		log.Fatalf("[ORCHESTRATOR] Failed to listen: %v", err)
	}

	grpcServer := grpc.NewServer(
		grpc.MaxRecvMsgSize(16*1024*1024),
		grpc.MaxSendMsgSize(16*1024*1024),
	)

	orchestrator := NewOrchestratorServer()
	pb.RegisterOrchestratorServiceServer(grpcServer, orchestrator)
	reflection.Register(grpcServer) // enables grpcurl / debug introspection

	log.Printf("[ORCHESTRATOR] gRPC server listening on :%s", port)
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("[ORCHESTRATOR] Failed to serve: %v", err)
	}
}
