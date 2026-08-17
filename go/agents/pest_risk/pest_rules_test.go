package main

import "testing"

func TestComputePestRisk_HighRiskBPH(t *testing.T) {
	risk, matches, _ := ComputePestRisk("Kharif", "Active Tillering", 28.0, 85.0, 5.0)

	if risk != "High" {
		t.Errorf("expected High risk, got %s", risk)
	}

	found := false
	for _, m := range matches {
		if m.Pest == "Brown Plant Hopper" {
			found = true
		}
	}
	if !found {
		t.Error("expected Brown Plant Hopper to be flagged under hot+humid tillering conditions")
	}
}

func TestComputePestRisk_LowRiskCoolDry(t *testing.T) {
	risk, matches, _ := ComputePestRisk("Rabi", "Grain Filling", 20.0, 40.0, 0.0)

	if risk != "Low" {
		t.Errorf("expected Low risk in cool/dry conditions, got %s", risk)
	}
	if len(matches) != 0 {
		t.Errorf("expected no pest matches, got %d", len(matches))
	}
}

func TestCategorizeTemp(t *testing.T) {
	tests := []struct {
		temp     float32
		expected string
	}{
		{20, "cool"},
		{28, "warm"},
		{35, "hot"},
	}
	for _, tt := range tests {
		got := categorizeTemp(tt.temp)
		if got != tt.expected {
			t.Errorf("categorizeTemp(%.1f) = %s, want %s", tt.temp, got, tt.expected)
		}
	}
}
