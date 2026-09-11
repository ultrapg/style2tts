#[derive(Debug, Clone, PartialEq)]
pub struct SentenceChunk {
    pub text: String,
    pub pause_after_ms: u32,
}

/// Splits text into natural sentence chunks based on punctuation and paragraphs.
pub fn split_text_into_chunks(input: &str, base_pause_ms: u32) -> Vec<SentenceChunk> {
    let mut chunks = Vec::new();
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return chunks;
    }

    // Split paragraphs first
    let paragraphs: Vec<&str> = trimmed.split("\n\n").collect();
    let max_chars_per_chunk = 200; // StyleTTS2 fails around ~512 phonemes, ~250 chars is safe

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

            // Sub-segment long sentences on commas or spaces to avoid ONNX max sequence limits
            let sub_chunks = split_long_sentence(s_trimmed, max_chars_per_chunk);

            for (sub_idx, sub_str) in sub_chunks.iter().enumerate() {
                let is_last_sub = sub_idx + 1 == sub_chunks.len();
                let sub_trimmed = sub_str.trim();

                let pause_ms = if is_last_sub {
                    if is_last_sentence && !is_last_paragraph {
                        // Paragraph boundary pause: longer
                        base_pause_ms.max(500)
                    } else if sub_trimmed.ends_with('?') || sub_trimmed.ends_with('!') {
                        base_pause_ms.max(400)
                    } else if sub_trimmed.ends_with("...") || sub_trimmed.ends_with('…') {
                        (base_pause_ms as f32 * 1.5) as u32
                    } else if sub_trimmed.ends_with(',') || sub_trimmed.ends_with(';') || sub_trimmed.ends_with(':') {
                        (base_pause_ms / 2).max(150)
                    } else {
                        base_pause_ms
                    }
                } else {
                    // It's a mid-sentence break created by sub-segmentation
                    if sub_trimmed.ends_with(',') || sub_trimmed.ends_with(';') || sub_trimmed.ends_with(':') || sub_trimmed.ends_with("--") {
                        (base_pause_ms / 2).max(150) // natural pause for comma
                    } else {
                        // artificial break at space, short pause to simulate continuous speech
                        (base_pause_ms / 4).max(50)
                    }
                };

                chunks.push(SentenceChunk {
                    text: sub_trimmed.to_string(),
                    pause_after_ms: pause_ms,
                });
            }
        }
    }

    chunks
}

fn split_long_sentence(text: &str, max_chars: usize) -> Vec<String> {
    if text.len() <= max_chars {
        return vec![text.to_string()];
    }

    let mut result = Vec::new();
    let mut current = String::new();
    
    for word in text.split_whitespace() {
        if current.is_empty() {
            current.push_str(word);
        } else if current.len() + word.len() + 1 > max_chars {
            // Must break here to avoid exceeding max_chars
            result.push(current.clone());
            current = word.to_string();
        } else {
            current.push(' ');
            current.push_str(word);
        }

        // If chunk is getting moderately long and ends with a natural break, flush early
        if current.len() > 80 && (word.ends_with(',') || word.ends_with(';') || word.ends_with(':') || word.ends_with("--")) {
             result.push(current.clone());
             current.clear();
        }
    }
    
    if !current.is_empty() {
        result.push(current);
    }
    
    result
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

    #[test]
    fn test_punctuation_pauses() {
        let text = "Wait... What happened? Done!";
        let chunks = split_text_into_chunks(text, 300);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].pause_after_ms, 450); // 300 * 1.5
        assert_eq!(chunks[1].pause_after_ms, 400); // question mark pause
        assert_eq!(chunks[2].pause_after_ms, 400); // exclamation mark pause
    }
}
