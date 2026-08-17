// go/agents/pest_risk/main.go
// OWNER: Aditya
// Pest Risk Agent — rule-based paddy pest advisory using Karnataka KVK guidelines

package main

import (
	"context"
	"fmt"
	"log"
	"net"
	"os"
	"strconv"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/reflection"

	pb "github.com/AdiXgit/Capstone/proto"
)

// ─────────────────────────────────────────────
// gRPC SERVER
// ─────────────────────────────────────────────

type pestRiskServer struct {
	pb.UnimplementedPestRiskAgentServiceServer
}

func (s *pestRiskServer) GetPestRisk(
	ctx context.Context, req *pb.PestRiskRequest,
) (*pb.PestRiskResponse, error) {
	log.Printf("[PEST_RISK] Query → district=%s season=%s stage=%s temp=%.1f humidity=%.1f",
		req.District, req.Season, req.GrowthStage, req.Temperature, req.Humidity)

	overallRisk, matches, advisory := ComputePestRisk(
		req.Season, req.GrowthStage, req.Temperature, req.Humidity, req.Rainfall,
	)

	var pestInfos []*pb.PestInfo
	for _, m := range matches {
		pestInfos = append(pestInfos, &pb.PestInfo{
			Name:         m.Pest,
			RiskLevel:    m.Risk,
			Signs:        m.Signs,
			Action:       m.Action,
			EtlThreshold: m.ETL,
		})
	}

	return &pb.PestRiskResponse{
		Metadata: &pb.AgentMetadata{
			AgentId:    "pest-risk-agent-001",
			AgentName:  "Pest Risk Advisory Agent",
			Timestamp:  time.Now().UTC().Format(time.RFC3339),
			Confidence: 0.90,
		},
		District:    req.District,
		OverallRisk: overallRisk,
		Pests:       pestInfos,
		Advisory:    advisory,
	}, nil
}

// ─────────────────────────────────────────────
// REGISTRATION + MAIN
// (identical pattern to weather.go — self-registers with orchestrator)
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
			AgentId:   "pest-risk-agent-001",
			AgentName: "Pest Risk Advisory Agent",
			AgentType: "PEST_RISK",
			Host:      "pest-risk-agent",
			Port:      port,
		})
		cancel()
		conn.Close()
		if err == nil {
			log.Printf("[PEST_RISK] Registered with orchestrator at %s", addr)
			return
		}
		time.Sleep(3 * time.Second)
	}
}

func main() {
	port := int32(50056)
	if p := os.Getenv("PEST_RISK_AGENT_PORT"); p != "" {
		if v, err := strconv.Atoi(p); err == nil {
			port = int32(v)
		}
	}

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", port))
	if err != nil {
		log.Fatalf("[PEST_RISK] Listen: %v", err)
	}

	srv := grpc.NewServer()
	pb.RegisterPestRiskAgentServiceServer(srv, &pestRiskServer{})
	reflection.Register(srv)

	go registerWithOrchestrator(port)

	log.Printf("[PEST_RISK] Running on :%d | KVK rule-based pest advisory", port)
	if err := srv.Serve(lis); err != nil {
		log.Fatalf("[PEST_RISK] Serve: %v", err)
	}
}
