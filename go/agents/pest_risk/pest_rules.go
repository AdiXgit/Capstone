// go/agents/pest_risk/pest_rules.go
// OWNER: Aditya
// Pest Risk Rulebook — encodes Karnataka KVK pest surveillance guidelines
// Pure rule-based lookup, no ML, no external dataset required

package main

// PestRule defines the trigger conditions and advisory for one pest.
type PestRule struct {
	Pest         string
	TempCats     []string // "cool" | "warm" | "hot"
	HumidityCats []string // "low" | "moderate" | "high"
	Stages       []string // growth stages this pest targets
	Risk         string   // "Low" | "Medium" | "High"
	Signs        string
	Action       string
	ETL          string // Economic Threshold Level
}

// PestRulebook — Karnataka KVK paddy pest calendar, encoded as rules.
var PestRulebook = []PestRule{
	{
		Pest:         "Brown Plant Hopper",
		TempCats:     []string{"warm"},
		HumidityCats: []string{"high"},
		Stages:       []string{"Active Tillering", "Maximum Tillering", "Panicle Initiation"},
		Risk:         "High",
		Signs:        "Honeydew deposits on lower leaves, hopper burn in circular patches",
		Action:       "Apply Imidacloprid 0.3ml/L. Drain standing water. Avoid excess Nitrogen.",
		ETL:          "5-10 hoppers per hill",
	},
	{
		Pest:         "Rice Blast",
		TempCats:     []string{"cool", "warm"},
		HumidityCats: []string{"high"},
		Stages:       []string{"Seedling", "Panicle Initiation", "Flowering"},
		Risk:         "High",
		Signs:        "Diamond-shaped lesions on leaves with grey centers",
		Action:       "Spray Tricyclazole 0.6g/L or Isoprothiolane 1.5ml/L preventively.",
		ETL:          "5% leaf area affected",
	},
	{
		Pest:         "Stem Borer",
		TempCats:     []string{"hot", "warm"},
		HumidityCats: []string{"moderate", "high"},
		Stages:       []string{"Active Tillering", "Maximum Tillering"},
		Risk:         "Medium",
		Signs:        "Dead hearts (vegetative stage) or white ears (reproductive stage)",
		Action:       "Apply Cartap Hydrochloride 1kg/acre or release Trichogramma cards.",
		ETL:          "5% dead hearts or 2% white ears",
	},
	{
		Pest:         "Leaf Folder",
		TempCats:     []string{"warm"},
		HumidityCats: []string{"high"},
		Stages:       []string{"Active Tillering", "Maximum Tillering", "Flowering"},
		Risk:         "Medium",
		Signs:        "Longitudinally folded leaves with white scraping damage inside",
		Action:       "Apply Chlorantraniliprole 0.3ml/L if >2 damaged leaves per hill.",
		ETL:          "2 damaged leaves per hill",
	},
	{
		Pest:         "Sheath Blight",
		TempCats:     []string{"warm", "hot"},
		HumidityCats: []string{"high"},
		Stages:       []string{"Maximum Tillering", "Panicle Initiation", "Flowering"},
		Risk:         "Medium",
		Signs:        "Irregular greenish-grey lesions on leaf sheath near water line",
		Action:       "Spray Hexaconazole 2ml/L or Validamycin 2ml/L.",
		ETL:          "Lesions reaching flag leaf on 20% of tillers",
	},
	{
		Pest:         "False Smut",
		TempCats:     []string{"warm"},
		HumidityCats: []string{"high"},
		Stages:       []string{"Flowering", "Grain Filling"},
		Risk:         "Low",
		Signs:        "Large velvety green spore balls replacing individual grains",
		Action:       "Spray Propiconazole 1ml/L at booting stage (preventive only).",
		ETL:          "Cosmetic — affects grain quality, not yield significantly",
	},
	{
		Pest:         "Gall Midge",
		TempCats:     []string{"warm"},
		HumidityCats: []string{"high"},
		Stages:       []string{"Seedling", "Active Tillering"},
		Risk:         "Medium",
		Signs:        "Silver shoots (onion-like tubular leaves) instead of normal tillers",
		Action:       "Apply Fipronil 0.3g/kg seed treatment. Remove silver shoots by hand.",
		ETL:          "5% silver shoots",
	},
}

// ─────────────────────────────────────────────
// CATEGORIZATION HELPERS
// ─────────────────────────────────────────────

func categorizeTemp(temp float32) string {
	if temp > 32 {
		return "hot"
	} else if temp > 25 {
		return "warm"
	}
	return "cool"
}

func categorizeHumidity(humidity float32) string {
	if humidity > 80 {
		return "high"
	} else if humidity > 60 {
		return "moderate"
	}
	return "low"
}

func contains(slice []string, item string) bool {
	for _, s := range slice {
		if s == item {
			return true
		}
	}
	return false
}

// ─────────────────────────────────────────────
// CORE LOGIC
// ─────────────────────────────────────────────

// ComputePestRisk matches current conditions against the rulebook and
// returns overall risk level, matched pest rules, and a human-readable advisory.
func ComputePestRisk(season, growthStage string, temp, humidity, rainfall float32) (string, []PestRule, string) {
	tempCat := categorizeTemp(temp)
	humCat := categorizeHumidity(humidity)

	var matches []PestRule
	riskOrder := map[string]int{"Low": 1, "Medium": 2, "High": 3}
	highestRisk := "Low"

	for _, rule := range PestRulebook {
		if contains(rule.TempCats, tempCat) &&
			contains(rule.HumidityCats, humCat) &&
			contains(rule.Stages, growthStage) {
			matches = append(matches, rule)
			if riskOrder[rule.Risk] > riskOrder[highestRisk] {
				highestRisk = rule.Risk
			}
		}
	}

	var advisory string
	if len(matches) == 0 {
		advisory = "No significant pest risk detected under current conditions. Continue routine field monitoring every 5-7 days."
	} else {
		var pestNames []string
		for _, m := range matches {
			pestNames = append(pestNames, m.Pest)
		}
		advisory = "Conditions favor: " + joinStrings(pestNames, ", ") +
			". Temperature (" + tempCat + ") and humidity (" + humCat + ") during " +
			growthStage + " stage create elevated pest pressure. Monitor fields every 3 days."
	}

	return highestRisk, matches, advisory
}

func joinStrings(items []string, sep string) string {
	result := ""
	for i, s := range items {
		if i > 0 {
			result += sep
		}
		result += s
	}
	return result
}
