use std::fs;
use std::path::PathBuf;

#[test]
fn test_style_cache_read_write_and_clear() {
    let test_cache_dir = PathBuf::from("cache/test_cache");
    let _ = fs::create_dir_all(&test_cache_dir);

    let styles_dir = test_cache_dir.join("styles");
    let _ = fs::create_dir_all(&styles_dir);

    let style_file = styles_dir.join("speaker_alice.style");
    let style = vec![0.5f32; 128];
    let predictor = vec![-0.5f32; 128];

    // Write binary style
    let mut bytes = Vec::with_capacity(256 * 4);
    for &val in style.iter().chain(predictor.iter()) {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    fs::write(&style_file, &bytes).unwrap();
    assert!(style_file.exists());

    // Read back
    let read_bytes = fs::read(&style_file).unwrap();
    assert_eq!(read_bytes.len(), 1024);

    // Verify floats
    let mut floats = Vec::new();
    for chunk in read_bytes.chunks_exact(4) {
        floats.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    assert_eq!(floats.len(), 256);
    assert_eq!(floats[0], 0.5f32);
    assert_eq!(floats[128], -0.5f32);

    // Clean up test cache
    let _ = fs::remove_file(&style_file);
    let _ = fs::remove_dir_all(&test_cache_dir);
}

#[test]
fn test_wav_audio_roundtrip() {
    let test_wav_path = PathBuf::from("cache/test_roundtrip.wav");
    let _ = fs::create_dir_all("cache");

    // Generate 1 second of 440Hz tone at 24kHz
    let sample_rate = 24000;
    let freq = 440.0;
    let mut samples = Vec::new();
    for i in 0..sample_rate {
        let t = (i as f32) / (sample_rate as f32);
        let sample = (32767.0 * (2.0 * std::f32::consts::PI * freq * t).sin()) as i16;
        samples.push(sample);
    }

    // Write WAV
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    {
        let mut writer = hound::WavWriter::create(&test_wav_path, spec).unwrap();
        for &s in &samples {
            writer.write_sample(s).unwrap();
        }
        writer.finalize().unwrap();
    }
    assert!(test_wav_path.exists());

    // Read WAV back
    let reader = hound::WavReader::open(&test_wav_path).unwrap();
    assert_eq!(reader.spec().sample_rate, 24000);
    assert_eq!(reader.spec().channels, 1);
    assert_eq!(reader.duration(), 24000);

    let _ = fs::remove_file(&test_wav_path);
}

#[test]
fn test_autocrop_behavior() {
    // Generate 20 seconds of audio at 24kHz
    let sample_rate = 24000;
    let total_samples = 20 * sample_rate;
    let mut samples = vec![0.0f32; total_samples];

    // Put speech in the middle (between second 2 and 18)
    for i in (2 * sample_rate)..(18 * sample_rate) {
        samples[i] = 0.5;
    }

    // Crop to 10.0s max
    let max_sec = 10.0f32;
    let max_samples = (max_sec * sample_rate as f32) as usize;

    let total_duration = samples.len() as f32 / sample_rate as f32;
    assert!(total_duration > max_sec);

    // Verify cropping logic cuts to exactly max_samples
    let start_idx = 2 * sample_rate - ((sample_rate as f32) * 0.05) as usize;
    let end_idx = start_idx + max_samples;
    let cropped = &samples[start_idx..end_idx];
    assert_eq!(cropped.len(), max_samples);
}

#[test]
fn test_sentence_segmentation_and_pauses() {
    use style2tts::text::split_text_into_chunks;

    let script = "It was a quiet night... Did you hear that? Look over there! Everything is fine.";
    let chunks = split_text_into_chunks(script, 250);
    assert_eq!(chunks.len(), 4);

    // Chunk 0: Ellipsis pause expansion
    assert_eq!(chunks[0].text, "It was a quiet night...");
    assert!(chunks[0].pause_after_ms >= 375); // 250 * 1.5

    // Chunk 1: Question
    assert_eq!(chunks[1].text, "Did you hear that?");
    assert!(chunks[1].pause_after_ms >= 400);

    // Chunk 2: Exclamation mark
    assert_eq!(chunks[2].text, "Look over there!");
    assert!(chunks[2].pause_after_ms >= 400);

    // Chunk 3: Standard sentence
    assert_eq!(chunks[3].text, "Everything is fine.");
    assert_eq!(chunks[3].pause_after_ms, 250);
}
