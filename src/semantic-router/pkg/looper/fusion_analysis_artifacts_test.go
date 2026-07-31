package looper

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestFilterNonContrastiveContradictionsDropsUniformAbsenceClaims(t *testing.T) {
	kept, dropped := filterNonContrastiveContradictions([]string{
		"No panel edited anything yet, so there is no concrete patch disagreement.",
		"Neither panel edited files in this turn.",
		"Panel 1 expects utils_tests.test_module while panel 2 expects tests.utils_tests.test_module",
		"Both panels agree on touching django/utils/autoreload.py",
	})

	assert.Equal(t, 2, dropped)
	assert.Equal(t, []string{
		"Panel 1 expects utils_tests.test_module while panel 2 expects tests.utils_tests.test_module",
		"Both panels agree on touching django/utils/autoreload.py",
	}, kept)
}

// A negated collective claim that also cites a panel individually still carries
// a contrast, so it has to survive.
func TestFilterNonContrastiveContradictionsKeepsContrastAlongsideAbsence(t *testing.T) {
	kept, dropped := filterNonContrastiveContradictions([]string{
		"Neither panel ran the tests, but panel 1 claims the fix is verified.",
	})

	assert.Equal(t, 0, dropped)
	assert.Len(t, kept, 1)
}

// A collective claim without negation can be a real disagreement with the
// evidence in the conversation rather than an artifact of the prose-only panel.
func TestFilterNonContrastiveContradictionsKeepsUnnegatedCollectiveClaims(t *testing.T) {
	kept, dropped := filterNonContrastiveContradictions([]string{
		"Both panels claim the suite passes, while the last test run in the conversation failed.",
	})

	assert.Equal(t, 0, dropped)
	assert.Len(t, kept, 1)
}

func TestAnalysisPromptTellsJudgeThePanelCannotAct(t *testing.T) {
	prompt := buildFusionAnalysisPrompt(fusionExecutionConfig{}, "Fix the failing test.", []*ModelResponse{
		{Model: "panel-a", Content: "Edit parser.py"},
	})

	assert.Contains(t, prompt, "could not call tools, edit files, or run commands")
}
