package looper

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"regexp"
	"strings"
	"time"

	"github.com/openai/openai-go"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// FusionLooper implements Fusion-style multi-model deliberation:
// parallel panel responses, judge analysis, then a final synthesized answer.
type FusionLooper struct {
	*BaseLooper
}

func NewFusionLooper(cfg *config.LooperConfig) *FusionLooper {
	return &FusionLooper{BaseLooper: NewBaseLooper(cfg)}
}

type fusionExecutionConfig struct {
	Model                        string
	AnalysisModels               []string
	AnalysisOverrides            map[string]config.FusionModelOverride
	MaxConcurrent                int
	MaxCompletionTokens          int
	RoundTimeoutSeconds          int
	MinSuccessfulResponses       int
	Temperature                  *float64
	IncludeAnalysis              bool
	IncludeIntermediateResponses bool
	OnError                      string
	AnalysisTemplate             string
	SynthesisTemplate            string
	JudgePromptVersion           string

	GroundingEnabled                 bool
	GroundingReference               string
	GroundingPolicy                  string
	GroundingMinScore                float64
	GroundingMinKeep                 int
	GroundingNLIContradictionPenalty float64
	GroundingOnError                 string
}

type FusionAnalysis struct {
	Consensus       []string `json:"consensus,omitempty"`
	Contradictions  []string `json:"contradictions,omitempty"`
	PartialCoverage []string `json:"partial_coverage,omitempty"`
	UniqueInsights  []string `json:"unique_insights,omitempty"`
	BlindSpots      []string `json:"blind_spots,omitempty"`
	Raw             string   `json:"raw,omitempty"`
	ParseFailed     bool     `json:"parse_failed,omitempty"`
}

type FusionPanelResponse struct {
	Model     string `json:"model"`
	Content   string `json:"content"`
	Reasoning string `json:"reasoning,omitempty"`
}

type FusionFailedModel struct {
	Model string `json:"model"`
	Error string `json:"error"`
}

type FusionTrace struct {
	Analysis            *FusionAnalysis       `json:"analysis,omitempty"`
	AnalysisSkipReason  string                `json:"analysis_skip_reason,omitempty"`
	AnalysisFilteredOut int                   `json:"analysis_filtered_out,omitempty"`
	Responses           []FusionPanelResponse `json:"responses,omitempty"`
	FailedModels        []FusionFailedModel   `json:"failed_models,omitempty"`
	JudgeModel          string                `json:"judge_model,omitempty"`
	AnalysisModels      []string              `json:"analysis_models,omitempty"`
	PromptVersion       string                `json:"prompt_version,omitempty"`
	Grounding           *FusionGroundingTrace `json:"grounding,omitempty"`
}

type fusionPanelResult struct {
	index int
	model string
	resp  *ModelResponse
	err   error
}

const fusionAnalysisSkipSimilarityThreshold = 0.92

type fusionAnalysisGateDecision struct {
	run    bool
	reason string
}

func (l *FusionLooper) Execute(ctx context.Context, req *Request) (*Response, error) {
	l.client.SetDecisionName(req.DecisionName)
	l.client.SetFusionDepth(1)
	defer l.client.SetFusionDepth(0)

	cfg := l.resolveFusionExecutionConfig(req)
	if len(cfg.AnalysisModels) == 0 {
		return nil, fmt.Errorf("fusion analysis_models cannot be empty")
	}
	if cfg.Model == "" {
		cfg.Model = cfg.AnalysisModels[0]
	}
	if err := validateFusionExecutionConfig(cfg); err != nil {
		return nil, err
	}
	if err := l.validateFusionModels(cfg); err != nil {
		return nil, err
	}

	logging.ComponentEvent("looper", "fusion_execution_started", map[string]interface{}{
		"decision":        req.DecisionName,
		"judge_model":     cfg.Model,
		"analysis_models": len(cfg.AnalysisModels),
		"streaming":       req.IsStreaming,
	})

	panelResponses, failedModels, err := l.executeFusionPanel(ctx, req, cfg)
	if err != nil {
		if cfg.OnError == config.FusionOnErrorFail || len(panelResponses) == 0 {
			return nil, err
		}
		logging.ComponentWarnEvent("looper", "fusion_panel_partial", map[string]interface{}{
			"decision":  req.DecisionName,
			"responses": len(panelResponses),
			"error":     err.Error(),
		})
	}

	// Grounding (optional) ranks/filters the panel before the judge. It makes no
	// model calls, so usage is summed from the full panel (the real cost paid).
	groundedPanel, groundingScores, groundingMode, err := l.applyGrounding(req, cfg, panelResponses)
	if err != nil {
		return nil, err
	}

	analysis := (*FusionAnalysis)(nil)
	analysisResp := (*ModelResponse)(nil)
	filteredContradictions := 0
	gate := shouldRunFusionAnalysis(groundedPanel)
	if gate.run {
		analysis, analysisResp = l.runFusionAnalysis(ctx, req, cfg, groundedPanel, groundingScores)
		if analysis != nil {
			analysis.Contradictions, filteredContradictions = filterFusionArtifactContradictions(analysis.Contradictions)
		}
	} else {
		logging.ComponentEvent("looper", "fusion_analysis_skipped", map[string]interface{}{
			"decision": req.DecisionName,
			"reason":   gate.reason,
			"panel":    len(groundedPanel),
		})
	}
	finalResp, err := l.runFusionFinal(ctx, req, cfg, groundedPanel, analysis, groundingScores)
	if err != nil {
		return nil, err
	}
	usage := SumUsage(panelResponses...).Add(analysisResp, finalResp)

	trace := buildFusionTrace(cfg, groundedPanel, failedModels, analysis, gate.reason, filteredContradictions, groundingMode, groundingScores)
	modelsUsed := orderedFusionModelsUsed(cfg.AnalysisModels, cfg.Model)
	iterations := len(cfg.AnalysisModels) + 2
	if !gate.run {
		iterations = len(cfg.AnalysisModels) + 1
	}

	if req.IsStreaming {
		return l.formatFusionStreamingResponse(finalResp, modelsUsed, iterations, cfg, trace, usage)
	}
	return l.formatFusionJSONResponse(finalResp, modelsUsed, iterations, cfg, trace, usage)
}

func (l *FusionLooper) validateFusionModels(cfg fusionExecutionConfig) error {
	for _, model := range append(append([]string{}, cfg.AnalysisModels...), cfg.Model) {
		for _, fusionName := range l.cfg.Fusion.EffectiveModelNames() {
			if model == fusionName {
				return fmt.Errorf("fusion model %q cannot be used as a judge or analysis model", model)
			}
		}
	}
	return nil
}

func (l *FusionLooper) executeFusionPanel(
	ctx context.Context,
	req *Request,
	cfg fusionExecutionConfig,
) ([]*ModelResponse, []FusionFailedModel, error) {
	// Paired multi-arm evaluation supplies the panel verbatim so every arm
	// synthesizes from a byte-identical panel (see bench/grounded_fusion). Skip
	// the live model calls and feed the cached panel straight into grounding +
	// synthesis, which are source-agnostic over []*ModelResponse.
	if len(req.CachedPanel) > 0 {
		return req.CachedPanel, nil, nil
	}

	panelCtx := ctx
	cancel := func() {}
	if cfg.RoundTimeoutSeconds > 0 {
		panelCtx, cancel = context.WithTimeout(ctx, time.Duration(cfg.RoundTimeoutSeconds)*time.Second)
	}
	defer cancel()

	results := make(chan fusionPanelResult, len(cfg.AnalysisModels))
	sem := make(chan struct{}, cfg.MaxConcurrent)
	for i, model := range cfg.AnalysisModels {
		go func(index int, modelName string) {
			select {
			case sem <- struct{}{}:
			case <-panelCtx.Done():
				results <- fusionPanelResult{index: index, model: modelName, err: panelCtx.Err()}
				return
			}
			defer func() { <-sem }()
			resp, err := l.callFusionModel(panelCtx, req, cfg, modelName, false, false, index+1, cfg.AnalysisOverrides[modelName])
			results <- fusionPanelResult{index: index, model: modelName, resp: resp, err: err}
		}(i, model)
	}

	collector := newFusionPanelCollector(cfg, cancel)
	for range cfg.AnalysisModels {
		select {
		case result := <-results:
			responses, err, done := collector.handleResult(result)
			if done {
				return responses, collector.failed, err
			}
		case <-panelCtx.Done():
			responses, err := collector.handleTimeout(panelCtx.Err())
			return responses, collector.failed, err
		}
	}

	responses := collector.responses()
	if len(responses) == 0 {
		return nil, collector.failed, fmt.Errorf("fusion panel failed: all %d analysis models failed", len(cfg.AnalysisModels))
	}
	return responses, collector.failed, nil
}

type fusionPanelCollector struct {
	cfg       fusionExecutionConfig
	cancel    context.CancelFunc
	ordered   []*ModelResponse
	failed    []FusionFailedModel
	successes int
}

func newFusionPanelCollector(cfg fusionExecutionConfig, cancel context.CancelFunc) *fusionPanelCollector {
	return &fusionPanelCollector{
		cfg:     cfg,
		cancel:  cancel,
		ordered: make([]*ModelResponse, len(cfg.AnalysisModels)),
	}
}

func (c *fusionPanelCollector) handleResult(result fusionPanelResult) ([]*ModelResponse, error, bool) {
	if result.err != nil {
		c.failed = append(c.failed, FusionFailedModel{Model: result.model, Error: result.err.Error()})
		if c.cfg.OnError == config.FusionOnErrorFail {
			c.cancel()
			return nil, fmt.Errorf("fusion panel model %q failed: %w", result.model, result.err), true
		}
		return nil, nil, false
	}
	c.ordered[result.index] = result.resp
	c.successes++
	if c.successes < c.cfg.MinSuccessfulResponses {
		return nil, nil, false
	}
	c.logQuorum()
	c.cancel()
	return c.responses(), nil, true
}

func (c *fusionPanelCollector) handleTimeout(err error) ([]*ModelResponse, error) {
	responses := c.responses()
	if len(responses) > 0 && c.cfg.OnError != config.FusionOnErrorFail {
		c.failed = append(c.failed, FusionFailedModel{Model: "panel", Error: err.Error()})
		return responses, err
	}
	return nil, err
}

func (c *fusionPanelCollector) responses() []*ModelResponse {
	return compactFusionPanelResponses(c.ordered)
}

func (c *fusionPanelCollector) logQuorum() {
	if c.successes >= len(c.cfg.AnalysisModels) {
		return
	}
	logging.ComponentEvent("looper", "fusion_panel_quorum_reached", map[string]interface{}{
		"responses": c.successes,
		"panel":     len(c.cfg.AnalysisModels),
	})
}

func compactFusionPanelResponses(ordered []*ModelResponse) []*ModelResponse {
	responses := make([]*ModelResponse, 0, len(ordered))
	for _, resp := range ordered {
		if resp != nil {
			responses = append(responses, resp)
		}
	}
	return responses
}

func shouldRunFusionAnalysis(panelResponses []*ModelResponse) fusionAnalysisGateDecision {
	if len(panelResponses) < 2 {
		return fusionAnalysisGateDecision{run: true, reason: "insufficient_panel_responses"}
	}
	if len(panelResponses) > 2 {
		// Keep the first version conservative. Two-panel fusion is the dominant
		// path today; for larger panels we can add a consensus gate later.
		return fusionAnalysisGateDecision{run: true, reason: "panel_count_gt_two"}
	}
	leftRaw, left := fusionGateContents(panelResponses[0])
	rightRaw, right := fusionGateContents(panelResponses[1])
	if left == "" || right == "" {
		return fusionAnalysisGateDecision{run: true, reason: "empty_panel_output"}
	}
	if left == right {
		return fusionAnalysisGateDecision{run: false, reason: "equivalent_panel_content"}
	}

	leftTool, leftArgs, leftTagged := parseTaggedToolCall(leftRaw)
	rightTool, rightArgs, rightTagged := parseTaggedToolCall(rightRaw)
	if leftTagged && rightTagged {
		if strings.EqualFold(strings.TrimSpace(leftTool), strings.TrimSpace(rightTool)) && fusionJSONArgsEquivalent(leftArgs, rightArgs) {
			return fusionAnalysisGateDecision{run: false, reason: "equivalent_tagged_tool_call"}
		}
		return fusionAnalysisGateDecision{run: true, reason: "different_tagged_tool_call"}
	}

	sim := fusionJaccardSimilarity(left, right)
	if sim >= fusionAnalysisSkipSimilarityThreshold {
		return fusionAnalysisGateDecision{run: false, reason: fmt.Sprintf("high_overlap_%.2f", sim)}
	}
	return fusionAnalysisGateDecision{run: true, reason: fmt.Sprintf("disagreement_%.2f", sim)}
}

func fusionGateContents(resp *ModelResponse) (raw string, normalized string) {
	if resp == nil {
		return "", ""
	}
	// Reuse the same prompt sanitizer the judge sees so the gate keys off the
	// same representation (e.g. truncated tool-call tails stripped).
	content := sanitizePanelContentForPrompt(resp.Content)
	raw = strings.TrimSpace(content)
	content = strings.ToLower(raw)
	if raw == "" {
		return "", ""
	}
	content = fusionWordBoundaryRe.ReplaceAllString(content, " ")
	content = strings.Join(strings.Fields(content), " ")
	return raw, content
}

func fusionJSONArgsEquivalent(left string, right string) bool {
	var a any
	var b any
	if err := json.Unmarshal([]byte(left), &a); err != nil {
		return strings.TrimSpace(left) == strings.TrimSpace(right)
	}
	if err := json.Unmarshal([]byte(right), &b); err != nil {
		return strings.TrimSpace(left) == strings.TrimSpace(right)
	}
	return fusionDeepEqualJSON(a, b)
}

func fusionDeepEqualJSON(a any, b any) bool {
	switch av := a.(type) {
	case map[string]any:
		bv, ok := b.(map[string]any)
		if !ok || len(av) != len(bv) {
			return false
		}
		for k, v := range av {
			other, ok := bv[k]
			if !ok || !fusionDeepEqualJSON(v, other) {
				return false
			}
		}
		return true
	case []any:
		bv, ok := b.([]any)
		if !ok || len(av) != len(bv) {
			return false
		}
		for i := range av {
			if !fusionDeepEqualJSON(av[i], bv[i]) {
				return false
			}
		}
		return true
	case float64:
		bf, ok := b.(float64)
		return ok && math.Abs(av-bf) <= 1e-9
	default:
		return fmt.Sprintf("%v", a) == fmt.Sprintf("%v", b)
	}
}

func fusionJaccardSimilarity(left string, right string) float64 {
	leftTokens := fusionTokenSet(left)
	rightTokens := fusionTokenSet(right)
	if len(leftTokens) == 0 || len(rightTokens) == 0 {
		return 0
	}
	intersection := 0
	union := make(map[string]struct{}, len(leftTokens)+len(rightTokens))
	for token := range leftTokens {
		union[token] = struct{}{}
		if _, ok := rightTokens[token]; ok {
			intersection++
		}
	}
	for token := range rightTokens {
		union[token] = struct{}{}
	}
	return float64(intersection) / float64(len(union))
}

func fusionTokenSet(content string) map[string]struct{} {
	out := map[string]struct{}{}
	for _, token := range strings.Fields(content) {
		if token == "" {
			continue
		}
		out[token] = struct{}{}
	}
	return out
}

var fusionArtifactContradictionPatterns = []string{
	"no panel edited",
	"neither panel edited",
	"none of the panels edited",
	"no edits were made by either panel",
	"no panel made any edits",
	"both panels made no edits",
	"neither panel made edits",
}

func filterFusionArtifactContradictions(in []string) ([]string, int) {
	if len(in) == 0 {
		return in, 0
	}
	out := make([]string, 0, len(in))
	filtered := 0
	for _, item := range in {
		lowered := strings.ToLower(strings.TrimSpace(item))
		isArtifact := false
		for _, pattern := range fusionArtifactContradictionPatterns {
			if strings.Contains(lowered, pattern) {
				isArtifact = true
				break
			}
		}
		if isArtifact {
			filtered++
			continue
		}
		out = append(out, item)
	}
	return out, filtered
}

func (l *FusionLooper) callFusionModel(
	ctx context.Context,
	req *Request,
	cfg fusionExecutionConfig,
	modelName string,
	allowTools bool,
	streaming bool,
	iteration int,
	override config.FusionModelOverride,
) (*ModelResponse, error) {
	callReq := cloneRequest(req.OriginalRequest)
	if !allowTools {
		callReq = stripFusionToolUse(callReq)
	}
	if override.Temperature != nil {
		callReq.Temperature = openai.Float(*override.Temperature)
	} else if cfg.Temperature != nil {
		callReq.Temperature = openai.Float(*cfg.Temperature)
	}
	if override.MaxCompletionTokens > 0 {
		callReq.MaxCompletionTokens = openai.Int(int64(override.MaxCompletionTokens))
	} else if cfg.MaxCompletionTokens > 0 {
		callReq.MaxCompletionTokens = openai.Int(int64(cfg.MaxCompletionTokens))
	}
	return l.client.CallModel(ctx, callReq, modelName, streaming, iteration, nil, accessKeyForModel(req, modelName))
}

func accessKeyForModel(req *Request, modelName string) string {
	if req == nil || req.ModelParams == nil {
		return ""
	}
	if params, ok := req.ModelParams[modelName]; ok {
		return params.AccessKey
	}
	for _, params := range req.ModelParams {
		for _, extID := range params.ExternalModelIDs {
			if extID == modelName {
				return params.AccessKey
			}
		}
	}
	return ""
}

func (l *FusionLooper) runFusionAnalysis(
	ctx context.Context,
	req *Request,
	cfg fusionExecutionConfig,
	panelResponses []*ModelResponse,
	groundingScores []groundingScore,
) (*FusionAnalysis, *ModelResponse) {
	prompt := buildFusionAnalysisPrompt(cfg, extractOriginalContent(req.OriginalRequest), panelResponses)
	if notes := formatGroundingNotes(groundingScores); notes != "" {
		prompt = prompt + "\n\n" + notes
	}
	analysisReq := buildFusionAnalysisStageRequest(req.OriginalRequest, prompt)
	analysisReq = stripFusionToolUse(analysisReq)
	resp, err := l.callFusionModel(ctx, &Request{OriginalRequest: analysisReq, ModelParams: req.ModelParams}, cfg, cfg.Model, false, false, len(panelResponses)+1, config.FusionModelOverride{})
	if err != nil {
		logging.ComponentWarnEvent("looper", "fusion_analysis_failed", map[string]interface{}{
			"judge_model": cfg.Model,
			"error":       err.Error(),
		})
		return nil, nil
	}
	analysis, parseErr := parseFusionAnalysis(resp.Content)
	if parseErr != nil {
		retryPrompt := prompt + "\n\n" +
			"Your previous response was invalid for parsing.\n" +
			"Return ONLY one valid JSON object matching the exact schema keys.\n" +
			"No prose. No markdown. No tool calls. No XML tags."
		retryReq := buildFusionAnalysisStageRequest(req.OriginalRequest, retryPrompt)
		retryReq = stripFusionToolUse(retryReq)
		retryResp, retryErr := l.callFusionModel(ctx, &Request{OriginalRequest: retryReq, ModelParams: req.ModelParams}, cfg, cfg.Model, false, false, len(panelResponses)+1, config.FusionModelOverride{})
		if retryErr == nil && retryResp != nil {
			if recovered, recoveredErr := parseFusionAnalysis(retryResp.Content); recoveredErr == nil {
				logging.ComponentEvent("looper", "fusion_analysis_parse_recovered", map[string]interface{}{
					"judge_model": cfg.Model,
				})
				return recovered, retryResp
			}
			resp = retryResp
		}
		logging.ComponentWarnEvent("looper", "fusion_analysis_parse_failed", map[string]interface{}{
			"judge_model": cfg.Model,
			"error":       parseErr.Error(),
		})
		return &FusionAnalysis{Raw: resp.Content, ParseFailed: true}, resp
	}
	return analysis, resp
}

func (l *FusionLooper) runFusionFinal(
	ctx context.Context,
	req *Request,
	cfg fusionExecutionConfig,
	panelResponses []*ModelResponse,
	analysis *FusionAnalysis,
	groundingScores []groundingScore,
) (*ModelResponse, error) {
	original := extractOriginalContent(req.OriginalRequest)
	outputContract := requestOutputContract(req.OriginalRequest, req.OutputContract)
	prompt := buildFusionFinalPrompt(cfg, original, outputContract, panelResponses, analysis)
	// Under weight/annotate policies the panel was not pruned, so the judge needs
	// the per-response groundedness signal at synthesis time to soft-weight.
	if notes := groundingSynthesisNotes(groundingScores, cfg.GroundingPolicy); notes != "" {
		prompt = prompt + "\n\n" + notes
	}
	// Affordances come off the original request; the synthesis call keeps its
	// tools, so these are the ones the judge can actually reach for. The rules
	// ride as a trailing system turn for the same prefix-cache reason the
	// analysis stage does: the shared conversation prefix stays byte-identical
	// across all three stages.
	stageRules := fusionSynthesisStageRules(detectJudgeToolAffordances(req.OriginalRequest))
	finalReq := appendFusionStageTurns(req.OriginalRequest, stageRules, prompt)
	resp, err := l.callFusionModel(ctx, &Request{OriginalRequest: finalReq, ModelParams: req.ModelParams}, cfg, cfg.Model, true, false, len(panelResponses)+2, config.FusionModelOverride{})
	if err != nil {
		return nil, fmt.Errorf("fusion final synthesis failed for judge model %q: %w", cfg.Model, err)
	}
	applyJSONActionOutputContract(req.OutputContractSpec, resp, panelResponses)
	applyFinalOutputContract(req.OutputContractSpec, resp)
	return resp, nil
}

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
			"%s\n"+
			"Original prompt:\n%s\n\n"+
			"Panel responses:\n%s",
		fusionContradictionGuidance, original, formatPanelResponsesForAnalysis(responses),
	)
}

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
	matches := fusionToolCallBlockRe.FindAllStringSubmatch(clean, -1)
	if len(matches) == 0 {
		return clean
	}
	steps := make([]string, 0, len(matches))
	for _, match := range matches {
		block := strings.TrimSpace(match[1])
		if block == "" {
			continue
		}
		toolName := strings.TrimSpace(fusionXMLTagRe.ReplaceAllString(strings.SplitN(block, "<arg_key>", 2)[0], ""))
		if toolName == "" {
			toolName = "unknown"
		}
		var args []string
		for _, pair := range fusionArgPairRe.FindAllStringSubmatch(block, -1) {
			key := strings.TrimSpace(fusionXMLTagRe.ReplaceAllString(pair[1], ""))
			value := strings.TrimSpace(fusionXMLTagRe.ReplaceAllString(pair[2], ""))
			if key != "" {
				args = append(args, fmt.Sprintf("%s=%q", key, value))
			}
		}
		step := fmt.Sprintf("Proposed tool call: %s", toolName)
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

func parseFusionAnalysis(content string) (*FusionAnalysis, error) {
	sanitized := sanitizeFusionAnalysisContent(content)
	candidates := jsonObjectParseCandidates(sanitized)
	for _, candidate := range extractBalancedJSONObjects(sanitized) {
		candidates = appendUniqueNonEmptyString(candidates, candidate)
	}
	if len(candidates) == 0 {
		return nil, fmt.Errorf("empty fusion analysis response")
	}
	var failures []string
	for _, candidate := range candidates {
		analysis, err := parseFusionAnalysisCandidate(candidate)
		if err == nil {
			return &analysis, nil
		}
		failures = append(failures, err.Error())
	}
	return nil, fmt.Errorf("%s", strings.Join(failures, "; "))
}

var (
	fusionThinkBlockRe    = regexp.MustCompile(`(?is)<think>.*?</think>`)
	fusionToolCallBlockRe = regexp.MustCompile(`(?is)<tool_call>\s*(.*?)\s*</tool_call>`)
	fusionActionBlockRe   = regexp.MustCompile(`(?is)<\|START_ACTION\|>.*?<\|END_ACTION\|>`)
	fusionArgPairRe       = regexp.MustCompile(`(?is)<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>\s*(.*?)\s*</arg_value>`)
	fusionXMLTagRe        = regexp.MustCompile(`(?is)<[^>]+>`)
	fusionWordBoundaryRe  = regexp.MustCompile(`[^a-z0-9_]+`)
)

func sanitizeFusionAnalysisContent(content string) string {
	trimmed := strings.TrimSpace(content)
	if trimmed == "" {
		return ""
	}
	withoutActionBlocks := strings.TrimSpace(fusionActionBlockRe.ReplaceAllString(trimmed, ""))
	withoutToolCallBlocks := strings.TrimSpace(fusionToolCallBlockRe.ReplaceAllString(withoutActionBlocks, ""))
	if idx := strings.Index(strings.ToLower(withoutToolCallBlocks), "<tool_call>"); idx >= 0 {
		withoutToolCallBlocks = strings.TrimSpace(withoutToolCallBlocks[:idx])
	}
	if idx := strings.Index(strings.ToLower(withoutToolCallBlocks), "<|start_action|>"); idx >= 0 {
		withoutToolCallBlocks = strings.TrimSpace(withoutToolCallBlocks[:idx])
	}
	withoutThinkBlocks := strings.TrimSpace(fusionThinkBlockRe.ReplaceAllString(withoutToolCallBlocks, ""))
	withoutThinkBlocks = strings.ReplaceAll(withoutThinkBlocks, "</think>", "\n")
	withoutThinkBlocks = strings.ReplaceAll(withoutThinkBlocks, "<think>", "\n")
	return strings.TrimSpace(withoutThinkBlocks)
}

func parseFusionAnalysisCandidate(candidate string) (FusionAnalysis, error) {
	var payload map[string]json.RawMessage
	if err := json.Unmarshal([]byte(candidate), &payload); err != nil {
		return FusionAnalysis{}, err
	}
	_, hasConsensus := payload["consensus"]
	_, hasContradictions := payload["contradictions"]
	_, hasPartial := payload["partial_coverage"]
	_, hasInsights := payload["unique_insights"]
	_, hasBlindSpots := payload["blind_spots"]
	if !hasConsensus && !hasContradictions && !hasPartial && !hasInsights && !hasBlindSpots {
		return FusionAnalysis{}, fmt.Errorf("candidate is valid JSON but contains no Fusion analysis keys")
	}

	analysis := FusionAnalysis{}
	if values, err := decodeFusionAnalysisList(payload["consensus"]); err != nil {
		return FusionAnalysis{}, fmt.Errorf("consensus: %w", err)
	} else {
		analysis.Consensus = values
	}
	if values, err := decodeFusionAnalysisList(payload["contradictions"]); err != nil {
		return FusionAnalysis{}, fmt.Errorf("contradictions: %w", err)
	} else {
		analysis.Contradictions = values
	}
	if values, err := decodeFusionAnalysisList(payload["partial_coverage"]); err != nil {
		return FusionAnalysis{}, fmt.Errorf("partial_coverage: %w", err)
	} else {
		analysis.PartialCoverage = values
	}
	if values, err := decodeFusionAnalysisList(payload["unique_insights"]); err != nil {
		return FusionAnalysis{}, fmt.Errorf("unique_insights: %w", err)
	} else {
		analysis.UniqueInsights = values
	}
	if values, err := decodeFusionAnalysisList(payload["blind_spots"]); err != nil {
		return FusionAnalysis{}, fmt.Errorf("blind_spots: %w", err)
	} else {
		analysis.BlindSpots = values
	}
	return analysis, nil
}

func decodeFusionAnalysisList(raw json.RawMessage) ([]string, error) {
	if len(raw) == 0 {
		return nil, nil
	}
	var single string
	if err := json.Unmarshal(raw, &single); err == nil {
		single = strings.TrimSpace(single)
		if single == "" {
			return []string{}, nil
		}
		return []string{single}, nil
	}
	var list []string
	if err := json.Unmarshal(raw, &list); err == nil {
		return list, nil
	}
	var nested [][]string
	if err := json.Unmarshal(raw, &nested); err == nil {
		out := make([]string, 0, len(nested))
		for _, group := range nested {
			for _, item := range group {
				item = strings.TrimSpace(item)
				if item != "" {
					out = append(out, item)
				}
			}
		}
		return out, nil
	}
	return nil, fmt.Errorf("expected []string or [][]string")
}

func extractBalancedJSONObjects(content string) []string {
	objects := []string{}
	start := -1
	depth := 0
	inString := false
	escaped := false
	for i := 0; i < len(content); i++ {
		ch := content[i]
		if inString {
			if escaped {
				escaped = false
				continue
			}
			if ch == '\\' {
				escaped = true
				continue
			}
			if ch == '"' {
				inString = false
			}
			continue
		}
		if ch == '"' {
			inString = true
			continue
		}
		if ch == '{' {
			if depth == 0 {
				start = i
			}
			depth++
			continue
		}
		if ch == '}' {
			if depth == 0 {
				continue
			}
			depth--
			if depth == 0 && start >= 0 && i >= start {
				objects = appendUniqueNonEmptyString(objects, strings.TrimSpace(content[start:i+1]))
				start = -1
			}
		}
	}
	return objects
}

func buildFusionTrace(
	cfg fusionExecutionConfig,
	panelResponses []*ModelResponse,
	failedModels []FusionFailedModel,
	analysis *FusionAnalysis,
	analysisSkipReason string,
	analysisFilteredOut int,
	groundingMode string,
	groundingScores []groundingScore,
) *FusionTrace {
	trace := &FusionTrace{
		JudgeModel:         cfg.Model,
		AnalysisModels:     append([]string(nil), cfg.AnalysisModels...),
		FailedModels:       failedModels,
		PromptVersion:      cfg.JudgePromptVersion,
		AnalysisSkipReason: analysisSkipReason,
		AnalysisFilteredOut: analysisFilteredOut,
	}
	if len(groundingScores) > 0 {
		trace.Grounding = &FusionGroundingTrace{
			ReferenceMode: groundingMode,
			Policy:        cfg.GroundingPolicy,
			Scores:        groundingScores,
		}
	}
	if cfg.IncludeAnalysis {
		trace.Analysis = analysis
	}
	if cfg.IncludeIntermediateResponses {
		trace.Responses = make([]FusionPanelResponse, 0, len(panelResponses))
		for _, resp := range panelResponses {
			trace.Responses = append(trace.Responses, FusionPanelResponse{
				Model:     resp.Model,
				Content:   resp.Content,
				Reasoning: resp.ReasoningContent,
			})
		}
	}
	return trace
}

func orderedFusionModelsUsed(analysisModels []string, judge string) []string {
	seen := map[string]bool{}
	models := make([]string, 0, len(analysisModels)+1)
	add := func(model string) {
		if model == "" || seen[model] {
			return
		}
		seen[model] = true
		models = append(models, model)
	}
	for _, model := range analysisModels {
		add(model)
	}
	add(judge)
	return models
}

func (l *FusionLooper) formatFusionJSONResponse(
	finalResp *ModelResponse,
	modelsUsed []string,
	iterations int,
	cfg fusionExecutionConfig,
	trace *FusionTrace,
	usage TokenUsage,
) (*Response, error) {
	if finalResp.HasToolCalls {
		return l.formatFusionToolCallJSONResponse(finalResp, modelsUsed, iterations, cfg, trace, usage)
	}

	completion := map[string]interface{}{
		"id":      fmt.Sprintf("chatcmpl-fusion-%d", time.Now().UnixNano()),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   finalResp.Model,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"message": map[string]interface{}{
					"role":    "assistant",
					"content": finalResp.Content,
				},
				"finish_reason": "stop",
			},
		},
		"usage": usage.Map(),
	}
	if cfg.IncludeAnalysis || cfg.IncludeIntermediateResponses || len(trace.FailedModels) > 0 || trace.Grounding != nil {
		completion["fusion"] = trace
	}
	body, err := json.Marshal(completion)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal fusion response: %w", err)
	}
	return &Response{
		Body:                  body,
		ContentType:           "application/json",
		Model:                 finalResp.Model,
		ModelsUsed:            modelsUsed,
		Iterations:            iterations,
		AlgorithmType:         "fusion",
		IntermediateResponses: trace,
		Usage:                 usage,
	}, nil
}

func (l *FusionLooper) formatFusionToolCallJSONResponse(
	finalResp *ModelResponse,
	modelsUsed []string,
	iterations int,
	cfg fusionExecutionConfig,
	trace *FusionTrace,
	usage TokenUsage,
) (*Response, error) {
	var completion map[string]interface{}
	if err := json.Unmarshal(finalResp.Raw, &completion); err != nil {
		return nil, fmt.Errorf("failed to parse fusion tool-call response: %w", err)
	}
	completion["id"] = fmt.Sprintf("chatcmpl-fusion-%d", time.Now().UnixNano())
	completion["model"] = finalResp.Model
	completion["usage"] = usage.Map()
	if cfg.IncludeAnalysis || cfg.IncludeIntermediateResponses || len(trace.FailedModels) > 0 || trace.Grounding != nil {
		completion["fusion"] = trace
	}
	body, err := json.Marshal(completion)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal fusion tool-call response: %w", err)
	}
	return &Response{
		Body:                  body,
		ContentType:           "application/json",
		Model:                 finalResp.Model,
		ModelsUsed:            modelsUsed,
		Iterations:            iterations,
		AlgorithmType:         "fusion",
		IntermediateResponses: trace,
		Usage:                 usage,
	}, nil
}

func (l *FusionLooper) formatFusionStreamingResponse(
	finalResp *ModelResponse,
	modelsUsed []string,
	iterations int,
	cfg fusionExecutionConfig,
	trace *FusionTrace,
	usage TokenUsage,
) (*Response, error) {
	timestamp := time.Now().Unix()
	id := fmt.Sprintf("chatcmpl-fusion-%d", timestamp)
	var (
		body []byte
		err  error
	)
	if finalResp.HasToolCalls {
		body, err = buildFusionStreamingToolCallSSE(id, timestamp, finalResp.Model, finalResp.Raw, cfg, trace)
		if err != nil {
			return nil, err
		}
	} else {
		body = buildFusionStreamingSSE(id, timestamp, finalResp.Model, finalResp.Content, cfg, trace)
	}
	resp := streamingLooperResponse(body, finalResp.Model, modelsUsed, iterations, "fusion")
	resp.IntermediateResponses = trace
	resp.Usage = usage
	return resp, nil
}

func buildFusionStreamingSSE(
	id string,
	created int64,
	model string,
	content string,
	cfg fusionExecutionConfig,
	trace *FusionTrace,
) []byte {
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
	for _, chunk := range splitIntoChunks(content, 50) {
		contentChoice := map[string]interface{}{
			"index":         0,
			"delta":         map[string]interface{}{"content": chunk},
			"finish_reason": nil,
		}
		body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, model, contentChoice, nil))
	}
	finalChoice := map[string]interface{}{
		"index":         0,
		"delta":         map[string]interface{}{},
		"finish_reason": "stop",
	}
	body = appendSSEDataLine(body, chatCompletionChunkPayload(id, created, model, finalChoice, nil))
	return appendSSEDone(body)
}
