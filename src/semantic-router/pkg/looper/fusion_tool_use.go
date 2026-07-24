package looper

import (
	"encoding/json"
	"fmt"

	"github.com/openai/openai-go"
)

const fusionAnalysisStageSystemPrompt = "You are the Fusion analysis judge. Output exactly one valid JSON object with only these keys: consensus, contradictions, partial_coverage, unique_insights, blind_spots. Do not call tools. Do not emit markdown, XML tags, or extra prose."

func stripFusionToolUse(req *openai.ChatCompletionNewParams) *openai.ChatCompletionNewParams {
	if req == nil {
		return nil
	}
	stripped := cloneRequest(req)
	stripped.Tools = nil
	stripped.ToolChoice = openai.ChatCompletionToolChoiceOptionUnionParam{}
	stripped.Functions = nil
	stripped.FunctionCall = openai.ChatCompletionNewParamsFunctionCallUnion{}
	return stripped
}
func appendFusionStageMessage(req *openai.ChatCompletionNewParams, content string) *openai.ChatCompletionNewParams {
	if req == nil {
		return nil
	}
	data, err := json.Marshal(req)
	if err != nil {
		return req
	}
	var reqMap map[string]interface{}
	err = json.Unmarshal(data, &reqMap)
	if err != nil {
		return req
	}
	messages, ok := reqMap["messages"].([]interface{})
	if !ok {
		return req
	}
	reqMap["messages"] = append(messages, map[string]string{
		"role":    "user",
		"content": content,
	})
	data, err = json.Marshal(reqMap)
	if err != nil {
		return req
	}
	var appended openai.ChatCompletionNewParams
	err = json.Unmarshal(data, &appended)
	if err != nil {
		return req
	}
	return &appended
}

// buildFusionAnalysisStageRequest appends the analysis judge's system
// instruction and prompt to the conversation, leaving the existing turns
// untouched.
//
// The history must be extended, never rewritten. vLLM's prefix cache keys on
// the longest common *prefix* of the token sequence, so deleting a message in
// the middle invalidates every block after it: the panel and synthesis calls
// send the conversation verbatim, and an analysis request that dropped the
// assistant turns diverged at the first one and re-prefilled the remainder from
// scratch. Measured against a 63-message agentic conversation, that dropped
// prefix reuse from 99.9% to 17% and evicted the shared prefix the other stages
// depend on, while removing only ~7% of the context (OpenCode's assistant
// messages carry tool_calls, not text).
func buildFusionAnalysisStageRequest(req *openai.ChatCompletionNewParams, content string) *openai.ChatCompletionNewParams {
	if req == nil {
		return nil
	}
	data, err := json.Marshal(req)
	if err != nil {
		return req
	}
	var reqMap map[string]interface{}
	if err := json.Unmarshal(data, &reqMap); err != nil {
		return req
	}
	messages, ok := reqMap["messages"].([]interface{})
	if !ok {
		return req
	}
	extended := make([]interface{}, 0, len(messages)+2)
	extended = append(extended, messages...)
	extended = append(extended, map[string]string{
		"role":    "system",
		"content": fusionAnalysisStageSystemPrompt,
	})
	extended = append(extended, map[string]string{
		"role":    "user",
		"content": content,
	})
	reqMap["messages"] = extended
	data, err = json.Marshal(reqMap)
	if err != nil {
		return req
	}
	var prepared openai.ChatCompletionNewParams
	if err := json.Unmarshal(data, &prepared); err != nil {
		return req
	}
	return &prepared
}

func buildFusionStreamingToolCallSSE(
	id string,
	created int64,
	model string,
	raw []byte,
	cfg fusionExecutionConfig,
	trace *FusionTrace,
) ([]byte, error) {
	toolCalls, err := fusionToolCallDeltasFromRaw(raw)
	if err != nil {
		return nil, err
	}

	var body []byte
	roleChoice := map[string]interface{}{
		"index":         0,
		"delta":         map[string]interface{}{"role": "assistant"},
		"finish_reason": nil,
	}
	var extra map[string]interface{}
	if cfg.IncludeAnalysis || cfg.IncludeIntermediateResponses || len(trace.FailedModels) > 0 || trace.Grounding != nil {
		extra = map[string]interface{}{"fusion": trace}
	}
	body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, model, roleChoice, extra))

	toolChoice := map[string]interface{}{
		"index":         0,
		"delta":         map[string]interface{}{"tool_calls": toolCalls},
		"finish_reason": nil,
	}
	body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, model, toolChoice, nil))

	finalChoice := map[string]interface{}{
		"index":         0,
		"delta":         map[string]interface{}{},
		"finish_reason": "tool_calls",
	}
	body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, model, finalChoice, nil))
	return appendSSEDone(body), nil
}

func fusionToolCallDeltasFromRaw(raw []byte) ([]map[string]interface{}, error) {
	var completion struct {
		Choices []struct {
			Message struct {
				ToolCalls []map[string]interface{} `json:"tool_calls"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := json.Unmarshal(raw, &completion); err != nil {
		return nil, fmt.Errorf("failed to parse fusion streaming tool-call response: %w", err)
	}
	if len(completion.Choices) == 0 || len(completion.Choices[0].Message.ToolCalls) == 0 {
		return nil, fmt.Errorf("fusion streaming tool-call response did not contain tool_calls")
	}

	toolCalls := make([]map[string]interface{}, 0, len(completion.Choices[0].Message.ToolCalls))
	for i, toolCall := range completion.Choices[0].Message.ToolCalls {
		delta := make(map[string]interface{}, len(toolCall)+1)
		for key, value := range toolCall {
			delta[key] = value
		}
		delta["index"] = i
		toolCalls = append(toolCalls, delta)
	}
	return toolCalls, nil
}
