package looper

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestStripReasoningAndToolScaffoldingRemovesCompleteBlocks(t *testing.T) {
	content := "<think>weighing the options</think>\n{\"consensus\":[\"a\"]}\n" +
		"<tool_call>bash<arg_key>command</arg_key><arg_value>pytest</arg_value></tool_call>"

	assert.Equal(t, `{"consensus":["a"]}`, stripReasoningAndToolScaffolding(content))
}

// A generation that hit its cap mid-block leaves an opener with no closer, so
// everything from the opener on has to go: there is no way to know where the
// block would have ended.
func TestStripReasoningAndToolScaffoldingDropsUnclosedBlockTail(t *testing.T) {
	content := `{"consensus":["a"]}` + "\n<tool_call>bash<arg_key>command</arg_key><arg_val"

	assert.Equal(t, `{"consensus":["a"]}`, stripReasoningAndToolScaffolding(content))
}

// A brace inside a string value must not close the object early, which is what
// makes this usable on payloads carrying code snippets.
func TestExtractBalancedJSONObjectsIgnoresBracesInsideStrings(t *testing.T) {
	objects := extractBalancedJSONObjects(`{"keystrokes":"if x { y }","task_complete":false}`)

	assert.Equal(t, []string{`{"keystrokes":"if x { y }","task_complete":false}`}, objects)
}

func TestExtractBalancedJSONObjectsReturnsEachObjectInOrder(t *testing.T) {
	objects := extractBalancedJSONObjects(`prose {"first":1} more {"second":2} tail`)

	assert.Equal(t, []string{`{"first":1}`, `{"second":2}`}, objects)
}

// extractJSONObject spans the first brace to the last, so a reply holding an
// example object and then the real one collapses into an unparsable blob. The
// balanced scan is what recovers the individual objects.
func TestJSONObjectParseCandidatesIncludeIndividualObjects(t *testing.T) {
	candidates := jsonObjectParseCandidates(`Example: {"analysis":"x"} Actual: {"analysis":"y"}`)

	assert.Contains(t, candidates, `{"analysis":"x"}`)
	assert.Contains(t, candidates, `{"analysis":"y"}`)
	// The whole-string readings still come first.
	assert.Equal(t, `Example: {"analysis":"x"} Actual: {"analysis":"y"}`, candidates[0])
}
