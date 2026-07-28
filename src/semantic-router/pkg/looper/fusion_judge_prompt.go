package looper

import (
	"fmt"
	"strings"

	"github.com/openai/openai-go"
)

// judgeToolAffordances records which tool families the judge can actually call
// for this request.
//
// The synthesis stage serves every fusion decision, agentic or not, so the
// rules below are emitted only for tools that are really on offer — telling a
// multiple-choice judge to run tests would be nonsense.
type judgeToolAffordances struct {
	HasTools  bool
	PlanTool  string
	ShellTool string
}

var (
	judgePlanToolNames  = []string{"todowrite", "todoread", "todo", "update_plan", "plan"}
	judgeShellToolNames = []string{"bash", "shell", "run_terminal_cmd", "terminal", "execute"}
)

func detectJudgeToolAffordances(req *openai.ChatCompletionNewParams) judgeToolAffordances {
	names := requestToolNames(req)
	aff := judgeToolAffordances{HasTools: len(names) > 0}
	for _, name := range names {
		lowered := strings.ToLower(strings.TrimSpace(name))
		if aff.PlanTool == "" && matchesToolName(judgePlanToolNames, lowered) {
			aff.PlanTool = name
		}
		if aff.ShellTool == "" && matchesToolName(judgeShellToolNames, lowered) {
			aff.ShellTool = name
		}
	}
	return aff
}

func matchesToolName(candidates []string, lowered string) bool {
	for _, candidate := range candidates {
		if lowered == candidate {
			return true
		}
	}
	return false
}

// requestToolNames lists callable tool names on a request, covering both the
// tools array and the legacy functions array.
func requestToolNames(req *openai.ChatCompletionNewParams) []string {
	reqMap, ok := requestAsMap(req)
	if !ok {
		return nil
	}
	var names []string
	if tools, ok := reqMap["tools"].([]interface{}); ok {
		for _, rawTool := range tools {
			tool, ok := rawTool.(map[string]interface{})
			if !ok {
				continue
			}
			fn, ok := tool["function"].(map[string]interface{})
			if !ok {
				continue
			}
			if name, ok := fn["name"].(string); ok && name != "" {
				names = append(names, name)
			}
		}
	}
	if functions, ok := reqMap["functions"].([]interface{}); ok {
		for _, rawFn := range functions {
			fn, ok := rawFn.(map[string]interface{})
			if !ok {
				continue
			}
			if name, ok := fn["name"].(string); ok && name != "" {
				names = append(names, name)
			}
		}
	}
	return names
}

// fusionSynthesisStageRules returns the standing rules for a judge that is
// driving an agent loop rather than answering a single question.
//
// Two behaviours motivated these, measured against single-endpoint runs using
// the same agent and dataset on SWE-bench Verified:
//
//   - Planning collapses. Two single endpoints roughly 4x apart in capability
//     both spend 2-3.5% of their actions on the plan tool; fusion spends 0.11%
//     (9 calls across 115 attempts against 264 and 274). The panel is
//     tool-stripped, so it only ever returns prose and never proposes a
//     bookkeeping call for the judge to adopt.
//   - Verification is displaced by deliberation. Fusion runs fewer tests than
//     both single endpoints on the large majority of shared tasks, so panel
//     agreement ends up standing in for measuring the fix.
func fusionSynthesisStageRules(aff judgeToolAffordances) string {
	if !aff.HasTools {
		return ""
	}
	rules := []string{
		"- You are the acting agent in a multi-turn loop, not a one-shot answerer. Calling a tool is a valid result for this turn.",
		"- The panel cannot call tools and can only return prose, so it will never propose an action. Treat a missing panel suggestion as no evidence either way, not as a reason to skip the action.",
	}
	if aff.PlanTool != "" {
		rules = append(rules, fmt.Sprintf(
			"- Keep the plan current with `%s`: record it before starting multi-step work and tick steps off as they land. If the conversation already contains a plan, carry it forward instead of dropping it.",
			aff.PlanTool,
		))
	}
	if aff.ShellTool != "" {
		rules = append(rules,
			"- Do not state that the work is complete, cor rect, or verified unless this conversation already shows a test run made after the most recent edit whose output passed. Panel agreement is not evidence of correctness; only observed test output is.",
			fmt.Sprintf(
				"- When that evidence is missing, spend this turn running the relevant tests with `%s` rather than declaring completion.",
				aff.ShellTool,
			),
		)
	}
	return "Standing rules while you drive this agent loop:\n" + strings.Join(rules, "\n")
}
