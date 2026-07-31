package looper

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strings"
)

var (
	toolArgPairRe = regexp.MustCompile(`(?is)<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>`)
	xmlTagRe      = regexp.MustCompile(`(?is)<[^>]+>`)
)

// Panels propose actions in two shapes. Some models emit a JSON body:
//
//	<tool_call>{"name":"bash","arguments":{"command":"pytest"}}</tool_call>
//
// others emit key/value XML, which is what the models in use today produce:
//
//	<tool_call>bash<arg_key>command</arg_key><arg_value>pytest</arg_value></tool_call>
//
// Reading only the first shape is why the analysis gate used to fall back to
// comparing prose: over 8478 recorded tool calls the JSON body never appeared
// once, so the structured comparison never ran. Both shapes are parsed here,
// through the single extractor the judge prompt also uses, so the two can no
// longer disagree about what a panel asked for.

// fusionToolArg keeps argument order for rendering; comparison ignores it.
type fusionToolArg struct {
	Key   string
	Value string
}

// fusionToolCall is a proposed action reduced to what decides its effect.
type fusionToolCall struct {
	Name string
	Args []fusionToolArg
	// RawArgsJSON is set for the JSON shape so existing rendering is unchanged.
	RawArgsJSON string
}

// fusionInertToolArgs are arguments the agent never acts on: free-text labels
// the model writes for a human reader. `description` rides along on 2633 of
// 2648 recorded bash calls, so counting it would report two panels running a
// byte-identical command as disagreeing whenever they worded the label
// differently.
var fusionInertToolArgs = map[string]struct{}{
	"description": {},
	"explanation": {},
	"reason":      {},
	"thought":     {},
}

// extractFusionToolCalls returns every complete tool call in a panel response,
// in order. Unterminated calls are already removed upstream by
// sanitizePanelContentForPrompt.
func extractFusionToolCalls(content string) []fusionToolCall {
	if !strings.Contains(strings.ToLower(content), "<tool_call>") {
		return nil
	}
	matches := toolCallBlockRe.FindAllStringSubmatch(content, -1)
	if len(matches) == 0 {
		return nil
	}
	calls := make([]fusionToolCall, 0, len(matches))
	for _, match := range matches {
		block := strings.TrimSpace(match[1])
		if block == "" {
			continue
		}
		if call, ok := parseFusionJSONToolCall(block); ok {
			calls = append(calls, call)
			continue
		}
		calls = append(calls, parseFusionXMLToolCall(block))
	}
	return calls
}

func parseFusionJSONToolCall(block string) (fusionToolCall, bool) {
	if !strings.HasPrefix(block, "{") {
		return fusionToolCall{}, false
	}
	var parsed struct {
		Name      string          `json:"name"`
		Arguments json.RawMessage `json:"arguments"`
	}
	if err := json.Unmarshal([]byte(block), &parsed); err != nil {
		return fusionToolCall{}, false
	}
	name := strings.TrimSpace(parsed.Name)
	if name == "" {
		return fusionToolCall{}, false
	}

	argsJSON := strings.TrimSpace(string(parsed.Arguments))
	if argsJSON == "" || argsJSON == "null" {
		argsJSON = "{}"
	} else if strings.HasPrefix(argsJSON, "\"") {
		// Some models double-encode the argument object as a JSON string.
		var decoded string
		if err := json.Unmarshal(parsed.Arguments, &decoded); err == nil {
			argsJSON = strings.TrimSpace(decoded)
		}
	}

	call := fusionToolCall{Name: name, RawArgsJSON: argsJSON}
	var fields map[string]any
	if err := json.Unmarshal([]byte(argsJSON), &fields); err != nil {
		// Unparsable arguments still compare, just as an opaque blob.
		call.Args = []fusionToolArg{{Key: "", Value: argsJSON}}
		return call, true
	}
	for key, value := range fields {
		call.Args = append(call.Args, fusionToolArg{Key: key, Value: stringifyToolArg(value)})
	}
	return call, true
}

func parseFusionXMLToolCall(block string) fusionToolCall {
	name := strings.TrimSpace(xmlTagRe.ReplaceAllString(strings.SplitN(block, "<arg_key>", 2)[0], ""))
	if name == "" {
		name = "unknown"
	}
	call := fusionToolCall{Name: name}
	for _, pair := range toolArgPairRe.FindAllStringSubmatch(block, -1) {
		key := strings.TrimSpace(xmlTagRe.ReplaceAllString(pair[1], ""))
		value := strings.TrimSpace(xmlTagRe.ReplaceAllString(pair[2], ""))
		if key != "" {
			call.Args = append(call.Args, fusionToolArg{Key: key, Value: value})
		}
	}
	return call
}

func stringifyToolArg(value any) string {
	switch v := value.(type) {
	case string:
		return v
	case nil:
		return ""
	default:
		encoded, err := json.Marshal(v)
		if err != nil {
			return fmt.Sprintf("%v", v)
		}
		return string(encoded)
	}
}

// actionableArgs drops arguments that cannot change what the tool does.
func (c fusionToolCall) actionableArgs() map[string]string {
	out := make(map[string]string, len(c.Args))
	for _, arg := range c.Args {
		if _, inert := fusionInertToolArgs[strings.ToLower(arg.Key)]; inert {
			continue
		}
		out[strings.ToLower(strings.TrimSpace(arg.Key))] = strings.TrimSpace(arg.Value)
	}
	return out
}

// fusionToolCallsEquivalent reports whether two panels proposed the same
// sequence of actions. Argument order is irrelevant, argument values are not:
// reading a file at offset 100 and at offset 140 are different requests even
// though the surrounding text is nearly identical.
func fusionToolCallsEquivalent(left []fusionToolCall, right []fusionToolCall) bool {
	if len(left) != len(right) || len(left) == 0 {
		return false
	}
	for i := range left {
		if !strings.EqualFold(strings.TrimSpace(left[i].Name), strings.TrimSpace(right[i].Name)) {
			return false
		}
		leftArgs := left[i].actionableArgs()
		rightArgs := right[i].actionableArgs()
		if len(leftArgs) != len(rightArgs) {
			return false
		}
		for key, leftValue := range leftArgs {
			rightValue, ok := rightArgs[key]
			if !ok {
				return false
			}
			if leftValue == rightValue {
				continue
			}
			// Fall back to JSON equality so {"a":1} and {"a": 1} agree.
			if !fusionJSONArgsEquivalent(leftValue, rightValue) {
				return false
			}
		}
	}
	return true
}
