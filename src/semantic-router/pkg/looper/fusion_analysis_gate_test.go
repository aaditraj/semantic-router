package looper

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestShouldRunFusionAnalysisSkipsEquivalentPanelContent(t *testing.T) {
	decision := shouldRunFusionAnalysis([]*ModelResponse{
		{Model: "panel-a", Content: "Run pytest tests/test_core.py -k parser"},
		{Model: "panel-b", Content: "Run pytest tests/test_core.py -k parser"},
	})

	assert.False(t, decision.run)
	assert.Equal(t, "equivalent_panel_content", decision.reason)
}

func TestShouldRunFusionAnalysisSkipsEquivalentTaggedToolCalls(t *testing.T) {
	decision := shouldRunFusionAnalysis([]*ModelResponse{
		{Model: "panel-a", Content: `<tool_call>{"name":"bash","arguments":{"command":"pytest -k parser","cwd":"/testbed"}}</tool_call>`},
		{Model: "panel-b", Content: `<tool_call>{"name":"bash","arguments":{"cwd":"/testbed","command":"pytest -k parser"}}</tool_call>`},
	})

	assert.False(t, decision.run)
	assert.Equal(t, "equivalent_tagged_tool_call", decision.reason)
}

func TestShouldRunFusionAnalysisRunsForDifferentTaggedToolCalls(t *testing.T) {
	decision := shouldRunFusionAnalysis([]*ModelResponse{
		{Model: "panel-a", Content: `<tool_call>{"name":"bash","arguments":{"command":"pytest -k parser"}}</tool_call>`},
		{Model: "panel-b", Content: `<tool_call>{"name":"readfile","arguments":{"path":"parser.py"}}</tool_call>`},
	})

	assert.True(t, decision.run)
	assert.Equal(t, "different_tagged_tool_call", decision.reason)
}

func TestShouldRunFusionAnalysisRunsForLowOverlap(t *testing.T) {
	decision := shouldRunFusionAnalysis([]*ModelResponse{
		{Model: "panel-a", Content: "Edit parser.py then run pytest -k parser and inspect traceback"},
		{Model: "panel-b", Content: "Search for serializer rename and update docs before anything else"},
	})

	assert.True(t, decision.run)
	assert.Contains(t, decision.reason, "disagreement_")
}

func TestShouldRunFusionAnalysisRunsForThreePanelResponses(t *testing.T) {
	decision := shouldRunFusionAnalysis([]*ModelResponse{
		{Model: "panel-a", Content: "A"},
		{Model: "panel-b", Content: "A"},
		{Model: "panel-c", Content: "A"},
	})

	assert.True(t, decision.run)
	assert.Equal(t, "panel_count_gt_two", decision.reason)
}

func TestFilterFusionArtifactContradictions(t *testing.T) {
	kept, filtered := filterFusionArtifactContradictions([]string{
		"No panel edited anything yet, so there is no concrete patch disagreement.",
		"Neither panel edited files in this turn.",
		"Panel 1 expects utils_tests.test_module while panel 2 expects tests.utils_tests.test_module",
		"Both panels agree on touching django/utils/autoreload.py",
	})

	assert.Equal(t, 2, filtered)
	assert.Equal(t, []string{
		"Panel 1 expects utils_tests.test_module while panel 2 expects tests.utils_tests.test_module",
		"Both panels agree on touching django/utils/autoreload.py",
	}, kept)
}

