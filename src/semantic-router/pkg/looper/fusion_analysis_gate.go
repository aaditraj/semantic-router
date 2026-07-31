package looper

import (
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"strings"
)

var fusionWordBoundaryRe = regexp.MustCompile(`[^a-z0-9_]+`)

type fusionAnalysisGateDecision struct {
	run    bool
	reason string
}

// shouldRunFusionAnalysis decides whether the judge's analysis call is worth
// making. It only ever answers "no" when the panel agreed on what to do, so the
// analysis it skips is one that had nothing to adjudicate.
//
// Enabled by algorithm.fusion.skip_analysis_on_agreement, which is off by
// default because skipping the call changes the model-call count and the trace
// shape a caller sees.
//
// Agreement is decided on the proposed tool calls, not on the surrounding
// prose. Comparing text measured the narration instead of the action: across a
// SWE-bench Verified run, panels emitting a byte-identical action were scored as
// disagreeing 49% of the time because they worded the explanation differently,
// while near-identical prose wrapping `git diff --cached` and `git diff` scored
// as agreement.
func shouldRunFusionAnalysis(panelResponses []*ModelResponse, skipOnAgreement bool) fusionAnalysisGateDecision {
	if !skipOnAgreement {
		return fusionAnalysisGateDecision{run: true}
	}
	if len(panelResponses) < 2 {
		return fusionAnalysisGateDecision{run: true, reason: "insufficient_panel_responses"}
	}
	if len(panelResponses) > 2 {
		// Two-panel Fusion is the path this was measured on. Agreement across a
		// larger panel needs a consensus rule rather than a pairwise one, so
		// those panels always run the analysis.
		return fusionAnalysisGateDecision{run: true, reason: "panel_count_gt_two"}
	}
	leftRaw, left := fusionGateContents(panelResponses[0])
	rightRaw, right := fusionGateContents(panelResponses[1])
	if left == "" || right == "" {
		return fusionAnalysisGateDecision{run: true, reason: "empty_panel_output"}
	}

	leftCalls := extractFusionToolCalls(leftRaw)
	rightCalls := extractFusionToolCalls(rightRaw)
	switch {
	case len(leftCalls) > 0 && len(rightCalls) > 0:
		if fusionToolCallsEquivalent(leftCalls, rightCalls) {
			return fusionAnalysisGateDecision{run: false, reason: "equivalent_tool_calls"}
		}
		return fusionAnalysisGateDecision{run: true, reason: "different_tool_calls"}
	case len(leftCalls) > 0 || len(rightCalls) > 0:
		// One panel wants to act and the other does not. That is a real
		// disagreement about the turn even if both texts read alike.
		return fusionAnalysisGateDecision{run: true, reason: "asymmetric_tool_call"}
	}

	// Neither panel proposed an action, so only the text is left. Exact equality
	// after normalization is the only claim worth making: a similarity score on
	// prose is what produced the 49% mismeasurement above.
	if left == right {
		return fusionAnalysisGateDecision{run: false, reason: "equivalent_panel_content"}
	}
	return fusionAnalysisGateDecision{run: true, reason: "prose_disagreement"}
}

// fusionGateContents returns the panel content the gate compares: the raw form
// for tool-call extraction, and a lowercased, punctuation-collapsed form for
// text equality. Both come off the same sanitizer the judge prompt uses, so the
// gate and the prompt can never disagree about what a panel said.
func fusionGateContents(resp *ModelResponse) (raw string, normalized string) {
	if resp == nil {
		return "", ""
	}
	raw = strings.TrimSpace(sanitizePanelContentForPrompt(resp.Content))
	if raw == "" {
		return "", ""
	}
	normalized = fusionWordBoundaryRe.ReplaceAllString(strings.ToLower(raw), " ")
	return raw, strings.Join(strings.Fields(normalized), " ")
}

// fusionJSONArgsEquivalent compares two argument payloads by value, so
// formatting differences such as {"a":1} against {"a": 1} do not read as
// different requests. Unparsable payloads fall back to string equality.
func fusionJSONArgsEquivalent(left string, right string) bool {
	var a any
	var b any
	if err := json.Unmarshal([]byte(left), &a); err != nil {
		return strings.TrimSpace(left) == strings.TrimSpace(right)
	}
	if err := json.Unmarshal([]byte(right), &b); err != nil {
		return strings.TrimSpace(left) == strings.TrimSpace(right)
	}
	return fusionDeepEqualJSON(a, b)
}

func fusionDeepEqualJSON(a any, b any) bool {
	switch av := a.(type) {
	case map[string]any:
		return fusionDeepEqualJSONObject(av, b)
	case []any:
		return fusionDeepEqualJSONArray(av, b)
	case float64:
		bf, ok := b.(float64)
		return ok && math.Abs(av-bf) <= 1e-9
	default:
		return fmt.Sprintf("%v", a) == fmt.Sprintf("%v", b)
	}
}

func fusionDeepEqualJSONObject(a map[string]any, b any) bool {
	other, ok := b.(map[string]any)
	if !ok || len(a) != len(other) {
		return false
	}
	for key, value := range a {
		counterpart, ok := other[key]
		if !ok || !fusionDeepEqualJSON(value, counterpart) {
			return false
		}
	}
	return true
}

func fusionDeepEqualJSONArray(a []any, b any) bool {
	other, ok := b.([]any)
	if !ok || len(a) != len(other) {
		return false
	}
	for i := range a {
		if !fusionDeepEqualJSON(a[i], other[i]) {
			return false
		}
	}
	return true
}
