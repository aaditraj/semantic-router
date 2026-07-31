package looper

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestParseFusionAnalysisAcceptsFencedJSON(t *testing.T) {
	analysis, err := parseFusionAnalysis("```json\n{\"consensus\":[\"agree\"],\"contradictions\":[],\"partial_coverage\":[],\"unique_insights\":[],\"blind_spots\":[]}\n```")

	require.NoError(t, err)
	require.NotNil(t, analysis)
	assert.Equal(t, []string{"agree"}, analysis.Consensus)
	assert.False(t, analysis.ParseFailed)
}

func TestParseFusionAnalysisAcceptsInlineFencedJSON(t *testing.T) {
	analysis, err := parseFusionAnalysis("```json {\"consensus\":[\"agree\"],\"contradictions\":[],\"partial_coverage\":[],\"unique_insights\":[],\"blind_spots\":[]} ```")

	require.NoError(t, err)
	require.NotNil(t, analysis)
	assert.Equal(t, []string{"agree"}, analysis.Consensus)
}

func TestParseFusionAnalysisExtractsJSONFromReasoningText(t *testing.T) {
	analysis, err := parseFusionAnalysis("The comparison is:\n{\"consensus\":[\"same fix\"],\"contradictions\":[],\"partial_coverage\":[],\"unique_insights\":[],\"blind_spots\":[]}\nDone.")

	require.NoError(t, err)
	require.NotNil(t, analysis)
	assert.Equal(t, []string{"same fix"}, analysis.Consensus)
}

func TestParseFusionAnalysisRepairsLooseJSONObject(t *testing.T) {
	analysis, err := parseFusionAnalysis("```json\n{`consensus`:[\"agree\",],`contradictions`:[],`partial_coverage`:[],`unique_insights`:[],`blind_spots`:[],}\n```")

	require.NoError(t, err)
	require.NotNil(t, analysis)
	assert.Equal(t, []string{"agree"}, analysis.Consensus)
}

func TestParseFusionAnalysisRepairsInvalidStringEscapes(t *testing.T) {
	analysis, err := parseFusionAnalysis(`{"consensus":["same \(escaped\) text"],"contradictions":[],"partial_coverage":[],"unique_insights":[],"blind_spots":[]}`)

	require.NoError(t, err)
	require.NotNil(t, analysis)
	assert.Equal(t, []string{`same \(escaped\) text`}, analysis.Consensus)
}

func TestParseFusionAnalysisRepairsUnclosedJSONObject(t *testing.T) {
	analysis, err := parseFusionAnalysis(`{"consensus":["fix signature"],"contradictions":[],"partial_coverage":[],"unique_insights":[],"blind_spots":[]`)

	require.NoError(t, err)
	require.NotNil(t, analysis)
	assert.Equal(t, []string{"fix signature"}, analysis.Consensus)
}

func TestParseFusionAnalysisStripsThinkTagPrefix(t *testing.T) {
	analysis, err := parseFusionAnalysis("<think>deliberation</think>\n{\"consensus\":[\"agree\"],\"contradictions\":[],\"partial_coverage\":[],\"unique_insights\":[],\"blind_spots\":[]}")

	require.NoError(t, err)
	require.NotNil(t, analysis)
	assert.Equal(t, []string{"agree"}, analysis.Consensus)
}

func TestParseFusionAnalysisKeepsJSONBeforeThinkAndToolCallTail(t *testing.T) {
	analysis, err := parseFusionAnalysis("{\"consensus\":[\"simplify signature\"],\"contradictions\":[],\"partial_coverage\":[],\"unique_insights\":[],\"blind_spots\":[]}</think><tool_call>bash<arg_key>command</arg_key><arg_value>cd /testbed && git diff HEAD</arg_value></tool_call>")

	require.NoError(t, err)
	require.NotNil(t, analysis)
	assert.Equal(t, []string{"simplify signature"}, analysis.Consensus)
}

func TestParseFusionAnalysisKeepsJSONBeforeActionBlockTail(t *testing.T) {
	analysis, err := parseFusionAnalysis("{\"consensus\":[\"fix target file\"],\"contradictions\":[],\"partial_coverage\":[],\"unique_insights\":[],\"blind_spots\":[]}<|START_ACTION|>[{\"name\":\"bash\",\"arguments\":{\"command\":\"pwd\"}}]<|END_ACTION|>")

	require.NoError(t, err)
	require.NotNil(t, analysis)
	assert.Equal(t, []string{"fix target file"}, analysis.Consensus)
}

func TestParseFusionAnalysisFlattensNestedArrays(t *testing.T) {
	analysis, err := parseFusionAnalysis(`{"consensus":[["a","b"]],"contradictions":[],"partial_coverage":[["c"]],"unique_insights":[],"blind_spots":[]}`)

	require.NoError(t, err)
	require.NotNil(t, analysis)
	assert.Equal(t, []string{"a", "b"}, analysis.Consensus)
	assert.Equal(t, []string{"c"}, analysis.PartialCoverage)
}

func TestParseFusionAnalysisAcceptsSingleStringValue(t *testing.T) {
	analysis, err := parseFusionAnalysis(`{"consensus":"Both models agree on the fix in astropy/html.py"}`)

	require.NoError(t, err)
	require.NotNil(t, analysis)
	assert.Equal(t, []string{"Both models agree on the fix in astropy/html.py"}, analysis.Consensus)
}

func TestParseFusionAnalysisRejectsPayloadWithNoFusionKeys(t *testing.T) {
	analysis, err := parseFusionAnalysis(`{"summary":"looks good","score":0.9}`)

	require.Error(t, err)
	assert.Nil(t, analysis)
	assert.Contains(t, err.Error(), "contains no Fusion analysis keys")
}

func TestParseFusionAnalysisSkipsNonFusionJSONObjectAndParsesNext(t *testing.T) {
	analysis, err := parseFusionAnalysis(`preface {"summary":"scratch"} suffix {"consensus":"agree","contradictions":[],"partial_coverage":[],"unique_insights":[],"blind_spots":[]}`)

	require.NoError(t, err)
	require.NotNil(t, analysis)
	assert.Equal(t, []string{"agree"}, analysis.Consensus)
}
