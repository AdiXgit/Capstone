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

// RegisterAgent — Python/Go agents call this on startup
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

// RouteQuery — fan-out to all relevant agents concurrently, collect results
func (s *OrchestratorServer) RouteQuery(
	ctx context.Context, req *pb.FarmerQuery,
) (*pb.OrchestratorResponse, error) {
	log.Printf("[ORCHESTRATOR] Routing query %s → type=%s district=%s",
		req.QueryId, req.QueryType, req.District)

	s.mu.Lock()
	s.queryCounter++
	s.mu.Unlock()

	// Determine which agents to fan-out to
	targetAgents := s.resolveAgents(req.QueryType)

	type agentWork struct {
		agentType string
		result    *pb.AgentResult
	}

	resultCh := make(chan agentWork, len(targetAgents))
	var wg sync.WaitGroup

	// ── Concurrent fan-out ────────────────────────────
	for _, agentType := range targetAgents {
		wg.Add(1)
		go func(aType string) {
			defer wg.Done()
			result := s.callAgent(ctx, aType, req)
			resultCh <- agentWork{agentType: aType, result: result}
		}(agentType)
	}

	// Close channel when all goroutines complete
	go func() {
		wg.Wait()
		close(resultCh)
	}()

	// ── Collect results ────────────────────────────────
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

	recommendation := s.synthesizeRecommendation(agentResults, req)

	return &pb.OrchestratorResponse{
		QueryId:           req.QueryId,
		Recommendation:    recommendation,
		AgentResults:      agentResults,
		OverallConfidence: avgConf,
		Timestamp:         time.Now().UTC().Format(time.RFC3339),
	}, nil
}

// resolveAgents maps query type → agent types to call
func (s *OrchestratorServer) resolveAgents(queryType string) []string {
	routing := map[string][]string{
		"WEATHER":     {"WEATHER"},
		"SOIL":        {"SOIL"},
		"CROP_HEALTH": {"CROP_HEALTH", "WEATHER"},
		"FERTILIZER":  {"SOIL", "WEATHER", "CROP_HEALTH"},
		"MARKET":      {"MARKET"},
		"PEST_RISK":   {"PEST_RISK", "WEATHER"},
		"FULL":        {"WEATHER", "SOIL", "CROP_HEALTH", "MARKET", "PEST_RISK"},
	}
	if agents, ok := routing[queryType]; ok {
		return agents
	}
	return []string{"WEATHER", "SOIL", "CROP_HEALTH", "MARKET", "PEST_RISK"}
}

// callAgent — establishes gRPC call to the appropriate agent
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

	if conn == nil {
		// Fallback: use env-configured agent addresses
		host := agentHostFromEnv(agentType)
		conn = &AgentConnection{AgentType: agentType, Host: host, Port: agentPortFromEnv(agentType)}
	}

	addr := fmt.Sprintf("%s:%d", conn.Host, conn.Port)
	grpcConn, err := grpc.NewClient(addr,
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		return &pb.AgentResult{
			AgentName:  agentType,
			ResultJson: `{"error":"connection failed"}`,
			Confidence: 0,
			Status:     "ERROR",
		}
	}
	defer grpcConn.Close()

	callCtx, cancel := context.WithTimeout(ctx, 10*time.Second)
	defer cancel()

	switch agentType {
	case "WEATHER":
		client := pb.NewWeatherAgentServiceClient(grpcConn)
		resp, err := client.GetWeatherForecast(callCtx, &pb.WeatherRequest{
			District:     req.District,
			Date:         time.Now().Format("2006-01-02"),
			ForecastDays: 7,
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

	case "MARKET":
		client := pb.NewMarketAgentServiceClient(grpcConn)
		resp, err := client.GetMarketAdvisory(callCtx, &pb.MarketRequest{
			District: req.District,
			Crop:     "Paddy",
		})
		if err != nil {
			return errorResult("MARKET_AGENT", err)
		}
		return marshalResult("Market Price Agent", resp, 0.80)

	case "PEST_RISK":
		client := pb.NewPestRiskAgentServiceClient(grpcConn)
		resp, err := client.GetPestRisk(callCtx, &pb.PestRiskRequest{
			District:    req.District,
			Season:      req.Season,
			GrowthStage: "Active Tillering", // TODO: pass actual stage once FarmerQuery includes it
			Temperature: 28.5,
			Humidity:    82,
			Rainfall:    10,
		})
		if err != nil {
			return errorResult("PEST_RISK_AGENT", err)
		}
		return marshalResult("Pest Risk Agent", resp, 0.85)
	}

	return &pb.AgentResult{Status: "UNKNOWN_AGENT"}
}

func (s *OrchestratorServer) synthesizeRecommendation(
	results []*pb.AgentResult, req *pb.FarmerQuery,
) string {
	parts := fmt.Sprintf("Advisory for %s (District: %s, Season: %s): ",
		req.FarmerId, req.District, req.Season)
	for _, r := range results {
		if r.Status == "OK" {
			parts += fmt.Sprintf("[%s: ready] ", r.AgentName)
		}
	}
	return parts
}

// ─────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────

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
		"MARKET":      "market-agent",
		"PEST_RISK":   "pest-risk-agent",
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
		"MARKET":      50055,
		"PEST_RISK":   50056,
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
	reflection.Register(grpcServer) // for grpcurl / debugging

	log.Printf("[ORCHESTRATOR] gRPC server listening on :%s", port)
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("[ORCHESTRATOR] Failed to serve: %v", err)
	}
}
