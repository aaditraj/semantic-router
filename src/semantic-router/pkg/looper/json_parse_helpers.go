package looper

import (
	"regexp"
	"strings"
)

var (
	reasoningBlockRe = regexp.MustCompile(`(?is)<think>.*?</think>`)
	toolCallBlockRe  = regexp.MustCompile(`(?is)<tool_call>\s*(.*?)\s*</tool_call>`)
	actionBlockRe    = regexp.MustCompile(`(?is)<\|START_ACTION\|>.*?<\|END_ACTION\|>`)
)

func jsonObjectParseCandidates(content string) []string {
	candidate := stripMarkdownJSONFence(strings.TrimSpace(content))
	candidates := appendUniqueNonEmptyString(nil, candidate)

	extracted := extractJSONObject(candidate)
	candidates = appendUniqueNonEmptyString(candidates, extracted)
	candidates = appendUniqueNonEmptyString(candidates, repairLooseJSONObject(candidate))
	candidates = appendUniqueNonEmptyString(candidates, repairLooseJSONObject(extracted))
	candidates = appendUniqueNonEmptyString(candidates, repairUnclosedJSONObject(candidate))
	candidates = appendUniqueNonEmptyString(candidates, repairUnclosedJSONObject(extracted))
	candidates = appendUniqueNonEmptyString(candidates, repairLooseJSONObject(repairUnclosedJSONObject(candidate)))
	candidates = appendUniqueNonEmptyString(candidates, repairLooseJSONObject(repairUnclosedJSONObject(extracted)))
	// Appended last so the readings above keep priority. These only decide
	// anything when the reply carries more than one object, which is when
	// extractJSONObject's first-brace-to-last-brace span is unparsable.
	for _, object := range extractBalancedJSONObjects(candidate) {
		candidates = appendUniqueNonEmptyString(candidates, object)
	}
	return candidates
}

// stripReasoningAndToolScaffolding removes the wrappers a tool-capable model
// puts around an answer: <think> blocks, <tool_call> blocks, and the
// <|START_ACTION|> blocks some chat templates emit. Anything from an opener the
// generation never closed is dropped too, since there is no way to know where
// the unfinished block would have ended.
//
// Any stage that asks a tool-capable model for JSON needs this. The model
// answers correctly and then appends an action block, and a parser reading the
// whole string just sees invalid JSON.
func stripReasoningAndToolScaffolding(content string) string {
	trimmed := strings.TrimSpace(content)
	if trimmed == "" {
		return ""
	}
	stripped := strings.TrimSpace(actionBlockRe.ReplaceAllString(trimmed, ""))
	stripped = strings.TrimSpace(toolCallBlockRe.ReplaceAllString(stripped, ""))
	stripped = truncateAtUnclosedBlock(stripped, "<tool_call>", "<|start_action|>")
	stripped = strings.TrimSpace(reasoningBlockRe.ReplaceAllString(stripped, ""))
	// A reasoning block can also be closed without ever being opened, or left
	// open at the end of the generation; drop the markers and keep the text.
	stripped = strings.ReplaceAll(stripped, "</think>", "\n")
	stripped = strings.ReplaceAll(stripped, "<think>", "\n")
	return strings.TrimSpace(stripped)
}

func truncateAtUnclosedBlock(content string, openers ...string) string {
	lowered := strings.ToLower(content)
	for _, opener := range openers {
		if idx := strings.Index(lowered, opener); idx >= 0 {
			content = strings.TrimSpace(content[:idx])
			lowered = strings.ToLower(content)
		}
	}
	return content
}

// extractBalancedJSONObjects returns every top-level object in content, in
// order. String and escape state is tracked so a brace inside a string value
// does not close an object early, which matters as soon as the payload carries
// code snippets.
//
// It complements extractJSONObject: that one spans the first `{` to the last
// `}`, so a reply holding prose, an example object, then the real answer
// collapses into a single unparsable blob.
func extractBalancedJSONObjects(content string) []string {
	masked := maskJSONStringContents(content)
	objects := []string{}
	start := -1
	depth := 0
	for i := 0; i < len(masked); i++ {
		switch masked[i] {
		case '{':
			if depth == 0 {
				start = i
			}
			depth++
		case '}':
			if depth == 0 {
				continue
			}
			depth--
			if depth == 0 && start >= 0 {
				objects = appendUniqueNonEmptyString(objects, strings.TrimSpace(content[start:i+1]))
				start = -1
			}
		}
	}
	return objects
}

// maskJSONStringContents blanks out the inside of every string literal, keeping
// the quotes and every byte offset. Structural scans can then look at brackets
// without a brace inside a string value — a code snippet in a payload, say —
// reading as structure.
func maskJSONStringContents(content string) string {
	out := []byte(content)
	inString := false
	escaped := false
	for i, ch := range out {
		switch {
		case escaped:
			escaped = false
			out[i] = ' '
		case inString && ch == '\\':
			escaped = true
			out[i] = ' '
		case ch == '"':
			inString = !inString
		case inString:
			out[i] = ' '
		}
	}
	return string(out)
}

func appendUniqueNonEmptyString(candidates []string, value string) []string {
	value = strings.TrimSpace(value)
	if value == "" {
		return candidates
	}
	for _, existing := range candidates {
		if existing == value {
			return candidates
		}
	}
	return append(candidates, value)
}

func stripMarkdownJSONFence(value string) string {
	value = strings.TrimSpace(value)
	if !strings.HasPrefix(value, "```") {
		return value
	}
	if inner, ok := insideOutermostFence(value); ok {
		return inner
	}
	lines := strings.Split(value, "\n")
	if len(lines) > 0 && strings.HasPrefix(strings.TrimSpace(lines[0]), "```") {
		lines = lines[1:]
	}
	if len(lines) > 0 && strings.HasPrefix(strings.TrimSpace(lines[len(lines)-1]), "```") {
		lines = lines[:len(lines)-1]
	}
	return strings.TrimSpace(strings.Join(lines, "\n"))
}

// insideOutermostFence returns the text between the opening fence and the last
// fence marker, dropping a `json` language tag. It reports false when the value
// has no closing fence, leaving the caller to strip what it can line-wise.
func insideOutermostFence(value string) (string, bool) {
	if strings.Count(value, "```") < 2 {
		return "", false
	}
	rest := value[strings.Index(value, "```")+3:]
	if trimmed := strings.TrimSpace(rest); strings.HasPrefix(strings.ToLower(trimmed), "json") {
		rest = strings.TrimLeft(trimmed[4:], " \t\r\n")
	}
	end := strings.LastIndex(rest, "```")
	if end < 0 {
		return "", false
	}
	return strings.TrimSpace(rest[:end]), true
}

func extractJSONObject(value string) string {
	start := strings.Index(value, "{")
	end := strings.LastIndex(value, "}")
	if start >= 0 && end > start {
		return strings.TrimSpace(value[start : end+1])
	}
	return strings.TrimSpace(value)
}

func repairLooseJSONObject(value string) string {
	return removeJSONTrailingCommas(repairInvalidJSONEscapes(repairJSONBackticks(value)))
}

func repairUnclosedJSONObject(value string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return value
	}
	unclosed := unclosedJSONBrackets(maskJSONStringContents(value))
	if len(unclosed) == 0 {
		return value
	}
	var b strings.Builder
	b.Grow(len(value) + len(unclosed))
	b.WriteString(value)
	for i := len(unclosed) - 1; i >= 0; i-- {
		if unclosed[i] == '{' {
			b.WriteByte('}')
			continue
		}
		b.WriteByte(']')
	}
	return b.String()
}

// unclosedJSONBrackets returns the brackets still open at the end of content
// whose string literals have already been masked, outermost first.
func unclosedJSONBrackets(masked string) []byte {
	var stack []byte
	for i := 0; i < len(masked); i++ {
		switch masked[i] {
		case '{', '[':
			stack = append(stack, masked[i])
		case '}', ']':
			if len(stack) == 0 {
				continue
			}
			if closesJSONBracket(stack[len(stack)-1], masked[i]) {
				stack = stack[:len(stack)-1]
			}
		}
	}
	return stack
}

func closesJSONBracket(open byte, close byte) bool {
	return (open == '{' && close == '}') || (open == '[' && close == ']')
}

func repairJSONBackticks(value string) string {
	var b strings.Builder
	b.Grow(len(value))
	inString := false
	escaped := false
	for _, r := range value {
		if r == '"' && !escaped {
			inString = !inString
		}
		if r == '`' && !inString {
			b.WriteRune('"')
		} else {
			b.WriteRune(r)
		}
		if r == '\\' && !escaped {
			escaped = true
		} else {
			escaped = false
		}
	}
	return b.String()
}

func removeJSONTrailingCommas(value string) string {
	var b strings.Builder
	b.Grow(len(value))
	inString := false
	escaped := false
	for index, r := range value {
		if r == '"' && !escaped {
			inString = !inString
		}
		if r == ',' && !inString {
			remaining := strings.TrimLeft(value[index+len(string(r)):], " \t\r\n")
			if strings.HasPrefix(remaining, "}") || strings.HasPrefix(remaining, "]") {
				continue
			}
		}
		b.WriteRune(r)
		if r == '\\' && !escaped {
			escaped = true
		} else {
			escaped = false
		}
	}
	return b.String()
}

func repairInvalidJSONEscapes(value string) string {
	var b strings.Builder
	b.Grow(len(value))
	inString := false
	escaped := false
	for i := 0; i < len(value); i++ {
		ch := value[i]
		if !inString {
			if ch == '"' {
				inString = true
			}
			b.WriteByte(ch)
			continue
		}
		if escaped {
			b.WriteByte(ch)
			escaped = false
			continue
		}
		if ch == '\\' {
			if i+1 < len(value) && validJSONEscape(value, i+1) {
				b.WriteByte(ch)
				escaped = true
			} else {
				b.WriteString(`\\`)
			}
			continue
		}
		if ch == '"' {
			inString = false
		}
		b.WriteByte(ch)
	}
	return b.String()
}

func validJSONEscape(value string, index int) bool {
	switch value[index] {
	case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
		return true
	case 'u':
		if index+4 >= len(value) {
			return false
		}
		for i := index + 1; i <= index+4; i++ {
			if !isHexByte(value[i]) {
				return false
			}
		}
		return true
	default:
		return false
	}
}

func isHexByte(ch byte) bool {
	return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F')
}
