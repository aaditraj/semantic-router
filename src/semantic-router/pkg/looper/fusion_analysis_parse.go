package looper

import (
	"encoding/json"
	"fmt"
	"strings"
)

// parseFusionAnalysis reads the judge's structured analysis out of its reply.
//
// The judge is the same tool-capable model that drives the loop, so its reply
// arrives wrapped in whatever scaffolding the chat template uses; the shared
// helpers strip that and enumerate the JSON readings worth trying.
func parseFusionAnalysis(content string) (*FusionAnalysis, error) {
	candidates := jsonObjectParseCandidates(stripReasoningAndToolScaffolding(content))
	if len(candidates) == 0 {
		return nil, fmt.Errorf("empty fusion analysis response")
	}
	var failures []string
	for _, candidate := range candidates {
		analysis, err := parseFusionAnalysisCandidate(candidate)
		if err == nil {
			return &analysis, nil
		}
		failures = append(failures, err.Error())
	}
	return nil, fmt.Errorf("%s", strings.Join(failures, "; "))
}

// parseFusionAnalysisCandidate requires at least one analysis key before
// accepting a candidate. Without that check any JSON object in the reply parses
// into an empty analysis, and an empty analysis is indistinguishable from a
// panel the judge found nothing to say about.
func parseFusionAnalysisCandidate(candidate string) (FusionAnalysis, error) {
	var payload map[string]json.RawMessage
	if err := json.Unmarshal([]byte(candidate), &payload); err != nil {
		return FusionAnalysis{}, err
	}

	analysis := FusionAnalysis{}
	fields := []struct {
		key    string
		target *[]string
	}{
		{"consensus", &analysis.Consensus},
		{"contradictions", &analysis.Contradictions},
		{"partial_coverage", &analysis.PartialCoverage},
		{"unique_insights", &analysis.UniqueInsights},
		{"blind_spots", &analysis.BlindSpots},
	}

	present := false
	for _, field := range fields {
		if _, ok := payload[field.key]; ok {
			present = true
			break
		}
	}
	if !present {
		return FusionAnalysis{}, fmt.Errorf("candidate is valid JSON but contains no Fusion analysis keys")
	}

	for _, field := range fields {
		values, err := decodeFusionAnalysisList(payload[field.key])
		if err != nil {
			return FusionAnalysis{}, fmt.Errorf("%s: %w", field.key, err)
		}
		*field.target = values
	}
	return analysis, nil
}

// decodeFusionAnalysisList accepts the shapes models actually produce for these
// fields: the requested list of strings, a bare string when the judge has only
// one point to make, and a list of lists when it groups points per panel.
func decodeFusionAnalysisList(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var single string
	if err := json.Unmarshal(raw, &single); err == nil {
		single = strings.TrimSpace(single)
		if single == "" {
			return []string{}, nil
		}
		return []string{single}, nil
	}
	var list []string
	if err := json.Unmarshal(raw, &list); err == nil {
		return list, nil
	}
	var nested [][]string
	if err := json.Unmarshal(raw, &nested); err == nil {
		out := make([]string, 0, len(nested))
		for _, group := range nested {
			for _, item := range group {
				item = strings.TrimSpace(item)
				if item != "" {
					out = append(out, item)
				}
			}
		}
		return out, nil
	}
	return nil, fmt.Errorf("expected []string or [][]string")
}
