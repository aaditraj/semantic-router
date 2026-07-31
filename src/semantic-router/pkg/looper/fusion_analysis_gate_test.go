package looper

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestShouldRunFusionAnalysisAlwaysRunsWhenGateDisabled(t *testing.T) {
	decision := shouldRunFusionAnalysis([]*ModelResponse{
		{Model: "panel-a", Content: "Run pytest tests/test_core.py -k parser"},
		{Model: "panel-b", Content: "Run pytest tests/test_core.py -k parser"},
	}, false)

	assert.True(t, decision.run)
	assert.Empty(t, decision.reason)
}

func TestShouldRunFusionAnalysisSkipsEquivalentPanelContent(t *testing.T) {
	decision := shouldRunFusionAnalysis([]*ModelResponse{
		{Model: "panel-a", Content: "Run pytest tests/test_core.py -k parser"},
		{Model: "panel-b", Content: "Run pytest tests/test_core.py -k parser"},
	}, true)

	assert.False(t, decision.run)
	assert.Equal(t, "equivalent_panel_content", decision.reason)
}

func TestShouldRunFusionAnalysisSkipsEquivalentJSONToolCalls(t *testing.T) {
	decision := shouldRunFusionAnalysis([]*ModelResponse{
		{Model: "panel-a", Content: `<tool_call>{"name":"bash","arguments":{"command":"pytest -k parser","cwd":"/testbed"}}</tool_call>`},
		{Model: "panel-b", Content: `<tool_call>{"name":"bash","arguments":{"cwd":"/testbed","command":"pytest -k parser"}}</tool_call>`},
	}, true)

	assert.False(t, decision.run)
	assert.Equal(t, "equivalent_tool_calls", decision.reason)
}

func TestShouldRunFusionAnalysisRunsForDifferentJSONToolCalls(t *testing.T) {
	decision := shouldRunFusionAnalysis([]*ModelResponse{
		{Model: "panel-a", Content: `<tool_call>{"name":"bash","arguments":{"command":"pytest -k parser"}}</tool_call>`},
		{Model: "panel-b", Content: `<tool_call>{"name":"readfile","arguments":{"path":"parser.py"}}</tool_call>`},
	}, true)

	assert.True(t, decision.run)
	assert.Equal(t, "different_tool_calls", decision.reason)
}

// The models in use emit key/value XML rather than a JSON body. The gate read
// only the JSON shape, so across 8478 recorded tool calls the structured
// comparison never once ran and every decision fell through to prose overlap.
func TestShouldRunFusionAnalysisSkipsEquivalentXMLToolCalls(t *testing.T) {
	decision := shouldRunFusionAnalysis([]*ModelResponse{
		{Model: "panel-a", Content: "Let me look at the autodetector.\n<tool_call>grep<arg_key>path</arg_key><arg_value>/testbed/a.py</arg_value><arg_key>pattern</arg_key><arg_value>to_state</arg_value></tool_call>"},
		{Model: "panel-b", Content: "I'll search the file for the symbol first.\n<tool_call>grep<arg_key>pattern</arg_key><arg_value>to_state</arg_value><arg_key>path</arg_key><arg_value>/testbed/a.py</arg_value></tool_call>"},
	}, true)

	assert.False(t, decision.run)
	assert.Equal(t, "equivalent_tool_calls", decision.reason)
}

// Recorded as high_overlap_0.92 under the similarity gate and skipped, despite
// the two commands producing entirely different output.
func TestShouldRunFusionAnalysisRunsForNearIdenticalProseDifferentCommand(t *testing.T) {
	decision := shouldRunFusionAnalysis([]*ModelResponse{
		{Model: "panel-a", Content: "Now let me check the staged changes to confirm the patch applied.\n<tool_call>bash<arg_key>command</arg_key><arg_value>cd /testbed && git diff --cached</arg_value></tool_call>"},
		{Model: "panel-b", Content: "Now let me check the staged changes to confirm the patch applied.\n<tool_call>bash<arg_key>command</arg_key><arg_value>cd /testbed && git diff</arg_value></tool_call>"},
	}, true)

	assert.True(t, decision.run)
	assert.Equal(t, "different_tool_calls", decision.reason)
}

// Same read, different region. Text overlap called these equivalent at 0.93.
func TestShouldRunFusionAnalysisRunsForSameToolDifferentArguments(t *testing.T) {
	decision := shouldRunFusionAnalysis([]*ModelResponse{
		{Model: "panel-a", Content: "<tool_call>read<arg_key>filePath</arg_key><arg_value>/testbed/a.py</arg_value><arg_key>offset</arg_key><arg_value>100</arg_value></tool_call>"},
		{Model: "panel-b", Content: "<tool_call>read<arg_key>filePath</arg_key><arg_value>/testbed/a.py</arg_value><arg_key>offset</arg_key><arg_value>140</arg_value></tool_call>"},
	}, true)

	assert.True(t, decision.run)
	assert.Equal(t, "different_tool_calls", decision.reason)
}

// A cosmetic label must not read as disagreement: it appeared on 2633 of 2648
// recorded bash calls and the agent never acts on it.
func TestShouldRunFusionAnalysisIgnoresCosmeticDescriptionArgument(t *testing.T) {
	decision := shouldRunFusionAnalysis([]*ModelResponse{
		{Model: "panel-a", Content: "<tool_call>bash<arg_key>command</arg_key><arg_value>pytest -k parser</arg_value><arg_key>description</arg_key><arg_value>Check test without my changes</arg_value></tool_call>"},
		{Model: "panel-b", Content: "<tool_call>bash<arg_key>command</arg_key><arg_value>pytest -k parser</arg_value><arg_key>description</arg_key><arg_value>Check test without changes</arg_value></tool_call>"},
	}, true)

	assert.False(t, decision.run)
	assert.Equal(t, "equivalent_tool_calls", decision.reason)
}

// Identical prose framing, but only one panel actually proposes an action.
func TestShouldRunFusionAnalysisRunsWhenOnlyOnePanelProposesAnAction(t *testing.T) {
	decision := shouldRunFusionAnalysis([]*ModelResponse{
		{Model: "panel-a", Content: "The fix looks complete, I will verify the tests now.\n<tool_call>bash<arg_key>command</arg_key><arg_value>pytest</arg_value></tool_call>"},
		{Model: "panel-b", Content: "The fix looks complete, I will verify the tests now."},
	}, true)

	assert.True(t, decision.run)
	assert.Equal(t, "asymmetric_tool_call", decision.reason)
}

// Multi-step proposals must match as a sequence, not just on the first call.
func TestShouldRunFusionAnalysisComparesEveryProposedCall(t *testing.T) {
	decision := shouldRunFusionAnalysis([]*ModelResponse{
		{Model: "panel-a", Content: "<tool_call>bash<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call><tool_call>bash<arg_key>command</arg_key><arg_value>pytest -k a</arg_value></tool_call>"},
		{Model: "panel-b", Content: "<tool_call>bash<arg_key>command</arg_key><arg_value>ls</arg_value></tool_call><tool_call>bash<arg_key>command</arg_key><arg_value>pytest -k b</arg_value></tool_call>"},
	}, true)

	assert.True(t, decision.run)
	assert.Equal(t, "different_tool_calls", decision.reason)
}

func TestShouldRunFusionAnalysisRunsForDifferingProse(t *testing.T) {
	decision := shouldRunFusionAnalysis([]*ModelResponse{
		{Model: "panel-a", Content: "Edit parser.py then run pytest -k parser and inspect traceback"},
		{Model: "panel-b", Content: "Search for serializer rename and update docs before anything else"},
	}, true)

	assert.True(t, decision.run)
	assert.Equal(t, "prose_disagreement", decision.reason)
}

func TestShouldRunFusionAnalysisRunsForThreePanelResponses(t *testing.T) {
	decision := shouldRunFusionAnalysis([]*ModelResponse{
		{Model: "panel-a", Content: "A"},
		{Model: "panel-b", Content: "A"},
		{Model: "panel-c", Content: "A"},
	}, true)

	assert.True(t, decision.run)
	assert.Equal(t, "panel_count_gt_two", decision.reason)
}
