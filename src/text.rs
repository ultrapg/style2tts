pub struct SentenceChunk {
    pub text: String,
    pub pause_after_ms: u32,
}

pub fn split_text_into_chunks(input: &str, base_pause_ms: u32) -> Vec<SentenceChunk> {
    let mut chunks = Vec::new();
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return chunks;
    }

    // Split paragraphs first
    let paragraphs: Vec<&str> = trimmed.split("\n\n").collect();

    for (p_idx, paragraph) in paragraphs.iter().enumerate() {
        let is_last_paragraph = p_idx + 1 == paragraphs.len();
        let p_trimmed = paragraph.trim();
        if p_trimmed.is_empty() {
            continue;
        }

        // Split paragraph into sentences based on punctuation (. ! ?)
        let sentences = segment_sentences(p_trimmed);

        for (s_idx, s) in sentences.iter().enumerate() {
            let is_last_sentence = s_idx + 1 == sentences.len();
            let s_trimmed = s.trim();
            if s_trimmed.is_empty() {
                continue;
            }

            // Determine pause based on position and terminal punctuation
            let pause_ms = if is_last_sentence && !is_last_paragraph {
                // Paragraph boundary pause: longer
                base_pause_ms.max(500)
            } else if s_trimmed.ends_with('?') || s_trimmed.ends_with('!') {
                base_pause_ms.max(400)
            } else if s_trimmed.ends_with(',') || s_trimmed.ends_with(';') || s_trimmed.ends_with(':') {
                (base_pause_ms / 2).max(150)
            } else {
                base_pause_ms
            };

            chunks.push(SentenceChunk {
                text: s_trimmed.to_string(),
                pause_after_ms: pause_ms,
            });
        }
    }

    chunks
}

fn segment_sentences(text: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();

    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    while i < len {
        let c = chars[i];
        current.push(c);

        if c == '.' || c == '!' || c == '?' || c == '\n' {
            // Check for ellipsis (...)
            if c == '.' && i + 1 < len && chars[i + 1] == '.' {
                i += 1;
                continue;
            }

            // Check if next char is whitespace or end of string
            let next_is_space = i + 1 >= len || chars[i + 1].is_whitespace();
            if next_is_space {
                let s = current.trim().to_string();
                if !s.is_empty() {
                    result.push(s);
                }
                current.clear();
            }
        }
        i += 1;
    }

    let remainder = current.trim().to_string();
    if !remainder.is_empty() {
        result.push(remainder);
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sentence_segmentation() {
        let text = "Hello world! How are you doing? I am fine, thank you.";
        let chunks = split_text_into_chunks(text, 300);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].text, "Hello world!");
        assert_eq!(chunks[1].text, "How are you doing?");
        assert_eq!(chunks[2].text, "I am fine, thank you.");
    }
}
