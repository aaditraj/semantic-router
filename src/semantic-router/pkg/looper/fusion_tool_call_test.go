package looper

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"

	"github.com/openai/openai-go"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

type fusionToolCallObservation struct {
	model         string
	prompt        string
	hasTools      bool
	toolChoice    string
	hasToolResult bool
}

func TestFusionLooperAllowsFinalJudgeToolCallsOnly(t *testing.T) {
	var (
		mu           sync.Mutex
		observations []fusionToolCallObservation
	)
	server := newFusionToolCallServer(t, func(observation fusionToolCallObservation) {
		mu.Lock()
		observations = append(observations, observation)
		mu.Unlock()
	})
	defer server.Close()

	var params openai.ChatCompletionNewParams
	require.NoError(t, json.Unmarshal([]byte(`{
		"model":"vllm-sr/fusion",
		"messages":[
			{"role":"user","content":"search before answering"},
			{"role":"assistant","content":null,"tool_calls":[{
				"id":"call_existing",
				"type":"function",
				"function":{"name":"search","arguments":"{\"query\":\"existing\"}"}
			}]},
			{"role":"tool","tool_call_id":"call_existing","content":"existing search result"}
		],
		"tools":[{
			"type":"function",
			"function":{
				"name":"search",
				"description":"Search the web",
				"parameters":{"type":"object","properties":{"query":{"type":"string"}}}
			}
		}],
		"tool_choice":"auto"
	}`), &params))
	req := &Request{
		OriginalRequest: &params,
		DecisionName:    "fusion-test",
		Algorithm: &config.AlgorithmConfig{
			Type: "fusion",
			Fusion: &config.FusionAlgorithmConfig{
				Model:          "judge",
				AnalysisModels: []string{"panel-a", "panel-b"},
			},
		},
	}

	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, "application/json", resp.ContentType)
	assert.Equal(t, "fusion", resp.AlgorithmType)

	mu.Lock()
	defer mu.Unlock()
	// Two panels, the analysis judge, then the synthesis judge.
	assertFusionToolCallObservations(t, observations, 4)

	var body map[string]interface{}
	require.NoError(t, json.Unmarshal(resp.Body, &body))
	choices := body["choices"].([]interface{})
	message := choices[0].(map[string]interface{})["message"].(map[string]interface{})
	require.Len(t, message["tool_calls"], 1)
	assert.Equal(t, "tool_calls", choices[0].(map[string]interface{})["finish_reason"])
	assert.Contains(t, body, "fusion")

	var usageBody struct {
		Usage TokenUsage `json:"usage"`
	}
	require.NoError(t, json.Unmarshal(resp.Body, &usageBody))
	assert.Equal(t, TokenUsage{PromptTokens: 90, CompletionTokens: 13, TotalTokens: 103}, usageBody.Usage)
}

// With skip_analysis_on_agreement enabled and both panels proposing the same
// thing, the analysis judge call is dropped and the trace records why.
func TestFusionLooperSkipsAnalysisWhenPanelAgreesAndGateEnabled(t *testing.T) {
	var (
		mu           sync.Mutex
		observations []fusionToolCallObservation
	)
	server := newFusionToolCallServer(t, func(observation fusionToolCallObservation) {
		mu.Lock()
		observations = append(observations, observation)
		mu.Unlock()
	})
	defer server.Close()

	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:                   "judge",
			AnalysisModels:          []string{"panel-a", "panel-b"},
			SkipAnalysisOnAgreement: boolPtr(true),
		},
	}

	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
	require.NoError(t, err)

	mu.Lock()
	defer mu.Unlock()
	require.Len(t, observations, 3)

	var body struct {
		Fusion FusionTrace `json:"fusion"`
	}
	require.NoError(t, json.Unmarshal(resp.Body, &body))
	assert.Equal(t, "equivalent_panel_content", body.Fusion.AnalysisSkipReason)
}

// The skip reason describes a skip. A run that made the analysis call must not
// report one, whatever the gate observed on the way to deciding to run.
func TestFusionLooperOmitsSkipReasonWhenAnalysisRuns(t *testing.T) {
	server := newFusionToolCallServer(t, nil)
	defer server.Close()

	req := newFusionTestRequest()
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:                   "judge",
			AnalysisModels:          []string{"panel-a", "panel-b"},
			SkipAnalysisOnAgreement: boolPtr(true),
		},
	}
	req.CachedPanel = []*ModelResponse{
		{Model: "panel-a", Content: "Edit parser.py and rerun the failing test"},
		{Model: "panel-b", Content: "Rewrite the serializer docs instead"},
	}

	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
	require.NoError(t, err)

	var body struct {
		Fusion FusionTrace `json:"fusion"`
	}
	require.NoError(t, json.Unmarshal(resp.Body, &body))
	assert.Empty(t, body.Fusion.AnalysisSkipReason)
}

func TestFusionLooperStreamsFinalJudgeToolCalls(t *testing.T) {
	server := newFusionToolCallServer(t, nil)
	defer server.Close()

	req := newFusionTestRequest()
	req.IsStreaming = true
	req.Algorithm = &config.AlgorithmConfig{
		Type: "fusion",
		Fusion: &config.FusionAlgorithmConfig{
			Model:          "judge",
			AnalysisModels: []string{"panel-a"},
		},
	}

	resp, err := NewFusionLooper(&config.LooperConfig{Endpoint: server.URL}).Execute(context.Background(), req)
	require.NoError(t, err)
	assert.Equal(t, "text/event-stream", resp.ContentType)
	body := string(resp.Body)
	assert.Contains(t, body, `"tool_calls"`)
	assert.Contains(t, body, `"finish_reason":"tool_calls"`)
	assert.Contains(t, body, `"fusion"`)
	assert.Contains(t, body, "data: [DONE]")
}

func newFusionToolCallServer(
	t *testing.T,
	observe func(fusionToolCallObservation),
) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		assert.Equal(t, "1", r.Header.Get("x-vsr-fusion-depth"))
		payload := decodeFusionToolCallPayload(t, r)
		observation := fusionToolCallObservationFromPayload(payload)
		if observe != nil {
			observe(observation)
		}
		writeFusionToolCallFixtureResponse(t, w, observation.model, observation.prompt)
	}))
}

func decodeFusionToolCallPayload(t *testing.T, r *http.Request) map[string]interface{} {
	t.Helper()
	var payload map[string]interface{}
	require.NoError(t, json.NewDecoder(r.Body).Decode(&payload))
	return payload
}

func fusionToolCallObservationFromPayload(payload map[string]interface{}) fusionToolCallObservation {
	model, _ := payload["model"].(string)
	tools, _ := payload["tools"].([]interface{})
	toolChoice, _ := payload["tool_choice"].(string)
	observation := fusionToolCallObservation{
		model:      model,
		hasTools:   len(tools) > 0,
		toolChoice: toolChoice,
	}
	messages, ok := payload["messages"].([]interface{})
	if !ok || len(messages) == 0 {
		return observation
	}
	for _, rawMessage := range messages {
		message, ok := rawMessage.(map[string]interface{})
		if !ok {
			continue
		}
		role, _ := message["role"].(string)
		content, _ := message["content"].(string)
		if role == "tool" && strings.Contains(content, "existing search result") {
			observation.hasToolResult = true
		}
	}
	if message, ok := messages[len(messages)-1].(map[string]interface{}); ok {
		observation.prompt, _ = message["content"].(string)
	}
	return observation
}

func writeFusionToolCallFixtureResponse(t *testing.T, w http.ResponseWriter, model string, prompt string) {
	t.Helper()
	w.Header().Set("Content-Type", "application/json")
	if model == "judge" && strings.Contains(prompt, "Final answer:") {
		_ = json.NewEncoder(w).Encode(fusionToolCallCompletion(model))
		return
	}

	content := "panel answer"
	if model == "judge" && strings.Contains(prompt, "return only valid JSON") {
		content = `{"consensus":["panel"],"contradictions":[],"partial_coverage":[],"unique_insights":[],"blind_spots":[]}`
	}
	_ = json.NewEncoder(w).Encode(map[string]interface{}{
		"id":      "chatcmpl-test",
		"object":  "chat.completion",
		"created": 1,
		"model":   model,
		"choices": []map[string]interface{}{
			{
				"index":         0,
				"message":       map[string]interface{}{"role": "assistant", "content": content},
				"finish_reason": "stop",
			},
		},
		"usage": fusionTestUsage(model),
	})
}

func fusionToolCallCompletion(model string) map[string]interface{} {
	return map[string]interface{}{
		"id":      "chatcmpl-tool-call",
		"object":  "chat.completion",
		"created": 1,
		"model":   model,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": nil,
					"tool_calls": []map[string]interface{}{
						{
							"id":   "call_search",
							"type": "function",
							"function": map[string]interface{}{
								"name":      "search",
								"arguments": `{"query":"fusion"}`,
							},
						},
					},
				},
				"finish_reason": "tool_calls",
			},
		},
		"usage": fusionTestUsage(model),
	}
}

func assertFusionToolCallObservations(t *testing.T, observations []fusionToolCallObservation, expected int) {
	t.Helper()
	require.Len(t, observations, expected)
	finalJudgeWithTools := false
	for _, got := range observations {
		assert.True(t, got.hasToolResult, "%s should preserve prior tool results", got.model)
		if got.model == "judge" && strings.Contains(got.prompt, "Final answer:") {
			assert.True(t, got.hasTools)
			assert.Equal(t, "auto", got.toolChoice)
			finalJudgeWithTools = true
			continue
		}
		assert.False(t, got.hasTools, "%s should not receive tools for prompt %q", got.model, got.prompt)
		assert.Empty(t, got.toolChoice)
	}
	assert.True(t, finalJudgeWithTools)
}
