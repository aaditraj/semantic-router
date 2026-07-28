package looper

import (
	"encoding/json"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// agenticJudgeRequest mimics an OpenCode turn: a real task, tool-calling
// assistant turns, and tool results.
func agenticJudgeRequest(t *testing.T) *openai.ChatCompletionNewParams {
	t.Helper()
	raw := []byte(`{
		"model":"vllm-sr/fusion",
		"messages":[
			{"role":"system","content":"You are OpenCode."},
			{"role":"user","content":"Fix get_child_arguments in django/utils/autoreload.py."},
			{"role":"assistant","content":"","tool_calls":[
				{"id":"c1","type":"function","function":{"name":"bash","arguments":"{\"command\":\"pytest\"}"}}
			]},
			{"role":"tool","tool_call_id":"c1","content":"1 failed, 40 passed"}
		],
		"tools":[
			{"type":"function","function":{"name":"bash","description":"Run a shell command."}},
			{"type":"function","function":{"name":"todowrite","description":"Record the plan."}},
			{"type":"function","function":{"name":"edit","description":"Edit a file."}}
		],
		"tool_choice":"auto"
	}`)
	var req openai.ChatCompletionNewParams
	require.NoError(t, json.Unmarshal(raw, &req))
	return &req
}

func messagesOf(t *testing.T, req *openai.ChatCompletionNewParams) []interface{} {
	t.Helper()
	data, err := json.Marshal(req)
	require.NoError(t, err)
	var payload map[string]interface{}
	require.NoError(t, json.Unmarshal(data, &payload))
	messages, ok := payload["messages"].([]interface{})
	require.True(t, ok)
	return messages
}

func TestDetectJudgeToolAffordancesFindsPlanAndShellTools(t *testing.T) {
	aff := detectJudgeToolAffordances(agenticJudgeRequest(t))

	assert.True(t, aff.HasTools)
	assert.Equal(t, "todowrite", aff.PlanTool)
	assert.Equal(t, "bash", aff.ShellTool)
}

func TestDetectJudgeToolAffordancesOnToollessRequest(t *testing.T) {
	raw := []byte(`{"model":"m","messages":[{"role":"user","content":"Which option is correct?"}]}`)
	var req openai.ChatCompletionNewParams
	require.NoError(t, json.Unmarshal(raw, &req))

	aff := detectJudgeToolAffordances(&req)

	assert.False(t, aff.HasTools)
	assert.Empty(t, aff.PlanTool)
	assert.Empty(t, aff.ShellTool)
}

func TestFusionSynthesisStageRulesCoverPlanningAndVerification(t *testing.T) {
	rules := fusionSynthesisStageRules(detectJudgeToolAffordances(agenticJudgeRequest(t)))

	assert.Contains(t, rules, "acting agent in a multi-turn loop")
	assert.Contains(t, rules, "`todowrite`")
	assert.Contains(t, rules, "carry it forward")
	assert.Contains(t, rules, "only observed test output is")
	assert.Contains(t, rules, "`bash`")
}

// Fusion also serves plain question answering, where a judge with no tools must
// not be told to run tests or keep a plan.
func TestFusionSynthesisStageRulesEmptyWithoutTools(t *testing.T) {
	assert.Empty(t, fusionSynthesisStageRules(judgeToolAffordances{}))
}

func TestFusionSynthesisStageRulesScaleToAvailableTools(t *testing.T) {
	rules := fusionSynthesisStageRules(judgeToolAffordances{HasTools: true, ShellTool: "bash"})

	assert.Contains(t, rules, "`bash`")
	assert.NotContains(t, rules, "Keep the plan current")
}

// The prefix-cache contract: every stage must send the original conversation
// byte-identically and only add turns at the end. Rewriting or editing an
// earlier message diverges the prefix and forces a full re-prefill for that
// stage, which measured at 99.9% -> 17% prefix reuse on a 63-message
// conversation.
func TestAppendFusionStageTurnsOnlyExtendsTheConversation(t *testing.T) {
	base := agenticJudgeRequest(t)
	original := messagesOf(t, base)

	extended := messagesOf(t, appendFusionStageTurns(base, "stage rules", "stage prompt"))

	require.Len(t, extended, len(original)+2)
	for i, want := range original {
		assert.Equal(t, want, extended[i], "message %d must be untouched", i)
	}
	system := extended[len(extended)-2].(map[string]interface{})
	assert.Equal(t, "system", system["role"])
	assert.Equal(t, "stage rules", system["content"])
	user := extended[len(extended)-1].(map[string]interface{})
	assert.Equal(t, "user", user["role"])
	assert.Equal(t, "stage prompt", user["content"])
}

// With no rules to declare the stage keeps its previous single-message shape.
func TestAppendFusionStageTurnsSkipsEmptySystemTurn(t *testing.T) {
	base := agenticJudgeRequest(t)
	original := messagesOf(t, base)

	extended := messagesOf(t, appendFusionStageTurns(base, "   ", "stage prompt"))

	require.Len(t, extended, len(original)+1)
	last := extended[len(extended)-1].(map[string]interface{})
	assert.Equal(t, "user", last["role"])
}

func TestFusionAnalysisPromptAsksForCheckableContradictions(t *testing.T) {
	prompt := buildFusionAnalysisPrompt(fusionExecutionConfig{}, "Fix the parser.", []*ModelResponse{
		{Model: "panel-a", Content: "Return the dotted path."},
	})

	assert.Contains(t, prompt, "concrete value, state, or behaviour")
	assert.Contains(t, prompt, "reading a file or running a command would settle")
	assert.Contains(t, prompt, "partial_coverage, not a contradiction")
	// The JSON contract the parser and UI depend on must survive.
	assert.Contains(t, prompt, "consensus, contradictions, partial_coverage, unique_insights, blind_spots")
	assert.Contains(t, prompt, "Exact expected JSON structure:")
}
