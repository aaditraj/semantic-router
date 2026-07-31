package looper

import (
	"regexp"
	"strings"
)

// The panel runs tool-stripped, so it can only ever return prose. Judges read
// that absence as a finding and file contradictions like "neither panel edited
// any files", which then travel into synthesis as if the panel had disagreed
// about the work. The analysis prompt now states the panel cannot act, which
// removes most of these at the source; this filter is the backstop for the ones
// that still get through.
//
// The rule is a property of what a contradiction is, rather than a list of the
// sentences we happened to see: a contradiction contrasts the panels, naming
// what one holds against what the other holds. An item asserting that none of
// them did something makes one uniform claim about all of them, so there is no
// opposing pair in it to adjudicate.
//
// Only negated collective claims qualify. "Both panels agree the test passes but
// the log shows a failure" is also collective, yet it does carry a real
// disagreement, so the negation is what separates the two.
var collectivePanelSubjects = []string{
	"no panel",
	"none of the panels",
	"neither panel",
	"neither of the panels",
	"both panels",
	"either panel",
	"all panels",
	"all of the panels",
	"no response",
	"neither response",
	"both responses",
}

var negationRe = regexp.MustCompile(`(?i)\b(?:no|not|neither|none|never|nor|without|nothing|cannot)\b|n't`)

// filterNonContrastiveContradictions drops contradictions that assert a uniform
// absence across the panel, returning the kept items and how many were dropped.
// The count lands in the Fusion trace so the rate stays visible.
func filterNonContrastiveContradictions(in []string) ([]string, int) {
	if len(in) == 0 {
		return in, 0
	}
	out := make([]string, 0, len(in))
	dropped := 0
	for _, item := range in {
		if isNonContrastiveContradiction(item) {
			dropped++
			continue
		}
		out = append(out, item)
	}
	return out, dropped
}

func isNonContrastiveContradiction(item string) bool {
	lowered := strings.ToLower(strings.TrimSpace(item))
	if lowered == "" {
		return false
	}
	if !mentionsCollectivePanel(lowered) || !negationRe.MatchString(lowered) {
		return false
	}
	// A negated collective claim can still sit alongside a genuine contrast, as
	// in "neither panel ran tests, but panel 1 claims the fix is verified".
	return !mentionsIndividualPanel(lowered)
}

func mentionsCollectivePanel(lowered string) bool {
	for _, subject := range collectivePanelSubjects {
		if strings.Contains(lowered, subject) {
			return true
		}
	}
	return false
}

// mentionsIndividualPanel reports whether the text singles out a numbered panel
// or response, which is how the analysis prompt asks contradictions to be
// phrased.
func mentionsIndividualPanel(lowered string) bool {
	for _, subject := range []string{"panel ", "response "} {
		for digit := '1'; digit <= '9'; digit++ {
			if strings.Contains(lowered, subject+string(digit)) {
				return true
			}
		}
	}
	return false
}
