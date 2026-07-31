package looper

import (
	"encoding/json"
	"fmt"
	"strings"
)

func buildFusionAnalysisPrompt(cfg fusionExecutionConfig, original string, responses []*ModelResponse) string {
	if cfg.AnalysisTemplate != "" {
		return renderFusionPrompt(cfg.AnalysisTemplate, original, responses, nil)
	}
	return fmt.Sprintf(
		"You are the Fusion analysis judge. Compare the panel responses and return only valid JSON.\n"+
			"Do not call tools. Do not emit tool_call blocks.\n"+
			"Return exactly one JSON object with these keys: consensus, contradictions, partial_coverage, unique_insights, blind_spots.\n"+
			"Each value must be an array with at most two concise strings.\n\n"+
			"Exact expected JSON structure:\n"+
			"```json\n"+
			"{\n"+
			"  \"consensus\": [\"point 1\", \"point 2\"],\n"+
			"  \"contradictions\": [\"point 1\"],\n"+
			"  \"partial_coverage\": [],\n"+
			"  \"unique_insights\": [\"point 1\"],\n"+
			"  \"blind_spots\": []\n"+
			"}\n"+
			"```\n\n"+
			"%s"+
			"%s\n"+
			"Original prompt:\n%s\n\n"+
			"Panel responses:\n%s",
		fusionPanelCapabilityNotice, fusionContradictionGuidance, original, formatPanelResponsesForAnalysis(responses),
	)
}

// fusionPanelCapabilityNotice tells the judge what the panel was able to do.
//
// The panel runs tool-stripped, so it can only return prose. Judges that are not
// told this read the absence of edits as a finding and report contradictions
// like "neither panel edited any files", which says nothing about the task. See
// filterNonContrastiveContradictions for the backstop.
const fusionPanelCapabilityNotice = "The panel could not call tools, edit files, or run commands; it could only write prose. " +
	"That a panel did not act is therefore never a finding.\n"

// fusionContradictionGuidance steers the judge toward disagreements that can be
// settled by evidence.
//
// Contradictions about a concrete value or test contract are 43.5% of the
// contradictions raised on solved SWE-bench tasks and 1.7% on failed ones,
// while open-ended "which approach is better" disagreements run the other way
// (280 on failures against 159 on solves). So the prompt asks for the first
// kind by construction and routes the second into partial_coverage.
const fusionContradictionGuidance = "State each contradiction as a claim about a concrete value, state, or behaviour: " +
	"name what one panel expects and what the other expects, such as \"panel 1 expects the call to return " +
	"'utils_tests.test_module', panel 2 expects 'tests.utils_tests.test_module'\". Prefer disagreements that " +
	"reading a file or running a command would settle. If the panels differ only in wording, style, or how much " +
	"detail they give, that is partial_coverage, not a contradiction.\n"

func buildFusionFinalPrompt(
	cfg fusionExecutionConfig,
	original string,
	outputContract string,
	responses []*ModelResponse,
	analysis *FusionAnalysis,
) string {
	if cfg.SynthesisTemplate != "" {
		return appendOutputContractForPrompt(
			renderFusionPrompt(cfg.SynthesisTemplate, original, responses, analysis),
			outputContract,
		)
	}
	analysisBlock := "No structured analysis is available. Synthesize directly from the panel responses."
	if analysis != nil && !analysis.ParseFailed {
		if data, err := json.MarshalIndent(analysis, "", "  "); err == nil {
			analysisBlock = string(data)
		}
	}
	prompt := fmt.Sprintf(`You are the Fusion calling model. Produce the final answer for the user using the panel responses and structured analysis. Resolve contradictions explicitly and do not mention internal model names unless the user asks.

Rules:
- Preserve the original output contract exactly.
- Do not reveal hidden reasoning, scratch work, panel reasoning, tool traces, or internal deliberation.
- Provide a concise explanation only when the original output contract asks for one.

Original prompt:
%s

Structured analysis:
%s

Panel responses:
%s

Final answer:`, original, analysisBlock, formatPanelResponses(responses))

	return appendOutputContractForPrompt(prompt, outputContract)
}

func renderFusionPrompt(template string, original string, responses []*ModelResponse, analysis *FusionAnalysis) string {
	replacer := strings.NewReplacer(
		"{{original}}", original,
		"{{responses}}", formatPanelResponses(responses),
		"{{analysis}}", formatFusionAnalysisForPrompt(analysis),
	)
	return replacer.Replace(template)
}

// truncatedPanelNotice replaces a panel response that consisted only of a tool
// call the model never finished emitting. Saying the panel was cut off is safer
// than showing the judge an empty block, which reads as "this panel had nothing
// to contribute".
const truncatedPanelNotice = "[panel response was cut off at the generation cap before producing a complete tool call]"

// stripUnterminatedToolCall removes a <tool_call> the generation never closed.
//
// Both judge prompts embed panel content, so an unterminated fragment teaches
// the judge to emit the same half-written tool call instead of a real one. Any
// panel cap makes this reachable, because a panel that falls into a repetition
// loop runs to the ceiling and stops mid-call: under a 1024-token cap 13.7% of
// generations ended that way (36.2% on django__django-13033) against 0.5%
// uncapped. The model also sometimes closes with </think> rather than
// </tool_call>, so anything from the first unclosed <tool_call> onward is
// discarded rather than trying to repair it.
func stripUnterminatedToolCall(content string) string {
	lower := strings.ToLower(content)
	for search := 0; ; {
		open := strings.Index(lower[search:], "<tool_call>")
		if open < 0 {
			return content
		}
		open += search
		closing := strings.Index(lower[open:], "</tool_call>")
		if closing < 0 {
			return strings.TrimSpace(content[:open])
		}
		search = open + closing + len("</tool_call>")
	}
}

// sanitizePanelContentForPrompt prepares panel output for embedding in a judge
// prompt. An empty response stays empty; one that survives only as a truncated
// fragment is reported as such.
func sanitizePanelContentForPrompt(content string) string {
	clean := strings.TrimSpace(content)
	if clean == "" {
		return ""
	}
	stripped := strings.TrimSpace(stripUnterminatedToolCall(clean))
	if stripped == "" {
		return truncatedPanelNotice
	}
	return stripped
}

func formatPanelResponses(responses []*ModelResponse) string {
	var b strings.Builder
	for i, resp := range responses {
		if resp == nil {
			continue
		}
		fmt.Fprintf(&b, "Response %d (%s):\n%s\n\n", i+1, resp.Model, sanitizePanelContentForPrompt(resp.Content))
		if reasoning := strings.TrimSpace(stripUnterminatedToolCall(resp.ReasoningContent)); reasoning != "" {
			fmt.Fprintf(&b, "Reasoning %d (%s):\n%s\n\n", i+1, resp.Model, reasoning)
		}
	}
	return strings.TrimSpace(b.String())
}

func formatPanelResponsesForAnalysis(responses []*ModelResponse) string {
	var b strings.Builder
	for i, resp := range responses {
		if resp == nil {
			continue
		}
		fmt.Fprintf(&b, "Response %d (%s):\n%s\n\n", i+1, resp.Model, normalizePanelResponseForAnalysis(resp.Content))
	}
	return strings.TrimSpace(b.String())
}

// normalizePanelResponseForAnalysis rewrites a panel's proposed tool call into a
// plain statement of what it wants to do.
//
// The analysis judge is asked for JSON while reading panel text that contains
// raw tool-call syntax, and it copies that syntax into its own reply: leaving the
// panel content verbatim produced unparsable analyses on most agentic turns.
// Stating the call in prose keeps the information and removes the pattern.
func normalizePanelResponseForAnalysis(content string) string {
	clean := sanitizePanelContentForPrompt(content)
	if clean == "" || clean == truncatedPanelNotice {
		return clean
	}
	if toolName, argsJSON, ok := parseTaggedToolCall(clean); ok {
		return fmt.Sprintf("Proposed tool call: %s\nArguments JSON: %s", toolName, strings.TrimSpace(argsJSON))
	}
	if !strings.Contains(clean, "<tool_call>") {
		return clean
	}
	// Same extractor the analysis gate uses, so the prompt and the gate can
	// never disagree about what a panel proposed.
	calls := extractFusionToolCalls(clean)
	if len(calls) == 0 {
		return clean
	}
	steps := make([]string, 0, len(calls))
	for _, call := range calls {
		var args []string
		for _, arg := range call.Args {
			args = append(args, fmt.Sprintf("%s=%q", arg.Key, arg.Value))
		}
		step := fmt.Sprintf("Proposed tool call: %s", call.Name)
		if len(args) > 0 {
			step += " (" + strings.Join(args, ", ") + ")"
		}
		steps = append(steps, step)
	}
	if len(steps) == 0 {
		return clean
	}
	normalized := strings.Join(steps, "\n")
	if idx := strings.Index(clean, "<tool_call>"); idx > 0 {
		preamble := strings.TrimSpace(clean[:idx])
		if preamble != "" {
			return preamble + "\n" + normalized
		}
	}
	return normalized
}

func formatFusionAnalysisForPrompt(analysis *FusionAnalysis) string {
	if analysis == nil {
		return ""
	}
	data, err := json.MarshalIndent(analysis, "", "  ")
	if err != nil {
		return analysis.Raw
	}
	return string(data)
}
