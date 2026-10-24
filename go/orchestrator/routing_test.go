package main

import (
	"sort"
	"testing"
)

func TestResolveAgents(t *testing.T) {
	s := NewOrchestratorServer()
	cases := map[string][]string{
		"WEATHER":   {"WEATHER"},
		"MARKET":    {"MARKET"},
		"PEST_RISK": {"PEST_RISK", "WEATHER"},
		"INCOME":    {"INCOME"},
		"FULL":      {"CROP_HEALTH", "INCOME", "MARKET", "PEST_RISK", "SOIL", "WEATHER"},
		"UNKNOWN":   {"CROP_HEALTH", "INCOME", "MARKET", "PEST_RISK", "SOIL", "WEATHER"},
	}
	for q, want := range cases {
		got := s.resolveAgents(q)
		sort.Strings(got)
		sort.Strings(want)
		if len(got) != len(want) {
			t.Fatalf("%s: got %v want %v", q, got, want)
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("%s: got %v want %v", q, got, want)
			}
		}
	}
}

func TestAgentDefaults(t *testing.T) {
	ports := map[string]int32{"WEATHER": 50052, "SOIL": 50053, "CROP_HEALTH": 50054, "MARKET": 50055, "PEST_RISK": 50056, "INCOME": 50057}
	for a, p := range ports {
		if got := agentPortFromEnv(a); got != p {
			t.Errorf("%s port: got %d want %d", a, got, p)
		}
	}
}
