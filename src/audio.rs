use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use hound::{WavReader, WavSpec, WavWriter};

/// Reads audio from either WAV or MP3 file, converts to 24kHz mono float PCM in [-1.0, 1.0].
pub fn read_audio_mono_24k(path: &Path) -> Result<Vec<f32>, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();

    if ext == "mp3" {
        read_mp3_mono_24k(path)
    } else if ext == "wav" {
        read_wav_mono_24k(path)
    } else {
        // Unknown or missing extension: try WAV first, then MP3
        read_wav_mono_24k(path).or_else(|_| read_mp3_mono_24k(path))
    }
}

/// Decodes MP3 audio file into 24kHz mono float samples
pub fn read_mp3_mono_24k(path: &Path) -> Result<Vec<f32>, String> {
    let file = File::open(path)
        .map_err(|e| format!("Failed to open MP3 file {}: {}", path.display(), e))?;
    let mut decoder = minimp3::Decoder::new(file);

    let mut raw_samples = Vec::new();
    let mut sample_rate = 0;
    let mut channels = 0;

    loop {
        match decoder.next_frame() {
            Ok(minimp3::Frame {
                data,
                sample_rate: sr,
                channels: ch,
                ..
            }) => {
                sample_rate = sr;
                channels = ch;
                raw_samples.extend_from_slice(&data);
            }
            Err(minimp3::Error::Eof) => break,
            Err(e) => return Err(format!("Error decoding MP3 frame from {}: {:?}", path.display(), e)),
        }
    }

    if raw_samples.is_empty() || sample_rate == 0 || channels == 0 {
        return Err(format!("MP3 file {} contains no audio data or has an invalid header", path.display()));
    }

    // Convert i16 samples to f32 in [-1.0, 1.0] and average channels to mono
    let mono_samples: Vec<f32> = if channels > 1 {
        raw_samples
            .chunks_exact(channels)
            .map(|ch_samples| {
                let sum: f32 = ch_samples.iter().map(|&s| (s as f32) / 32768.0).sum();
                sum / (channels as f32)
            })
            .collect()
    } else {
        raw_samples.iter().map(|&s| (s as f32) / 32768.0).collect()
    };

    // Resample to 24000 Hz if needed
    if sample_rate == 24000 {
        Ok(mono_samples)
    } else {
        Ok(resample_linear(&mono_samples, sample_rate as u32, 24000))
    }
}

/// Decodes WAV audio file into 24kHz mono float samples
pub fn read_wav_mono_24k(path: &Path) -> Result<Vec<f32>, String> {
    let mut reader = WavReader::open(path)
        .map_err(|e| format!("Failed to open WAV file {}: {}", path.display(), e))?;

    let spec = reader.spec();
    let channels = spec.channels as usize;
    let sample_rate = spec.sample_rate;

    let raw_samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| (v as f32) / max_val))
                .collect::<Result<Vec<f32>, _>>()
                .map_err(|e| format!("Error decoding WAV samples: {}", e))?
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<f32>, _>>()
            .map_err(|e| format!("Error decoding float WAV samples: {}", e))?,
    };

    if raw_samples.is_empty() {
        return Err("WAV file contains no audio samples".to_string());
    }

    // Convert to mono
    let mono_samples: Vec<f32> = if channels > 1 {
        raw_samples
            .chunks_exact(channels)
            .map(|chunk| chunk.iter().sum::<f32>() / (channels as f32))
            .collect()
    } else {
        raw_samples
    };

    // Resample to 24000 Hz if needed
    if sample_rate == 24000 {
        Ok(mono_samples)
    } else {
        Ok(resample_linear(&mono_samples, sample_rate, 24000))
    }
}

/// Auto-crops reference audio if duration exceeds `max_duration_sec` (e.g. 10.0s).
/// Trims leading silence and extracts the optimal active speech window with an edge fade-out.
/// Returns (cropped_samples, was_cropped, original_duration_sec).
pub fn autocrop_reference_audio(
    samples: &[f32],
    sample_rate: u32,
    max_duration_sec: f32,
) -> (Vec<f32>, bool, f32) {
    let orig_duration = (samples.len() as f32) / (sample_rate as f32);
    if orig_duration <= max_duration_sec {
        return (samples.to_vec(), false, orig_duration);
    }

    let max_samples = (max_duration_sec * (sample_rate as f32)) as usize;

    // Detect leading silence threshold (amplitude < 0.015)
    let silence_threshold = 0.015f32;
    let mut start_idx = 0;
    for (i, &s) in samples.iter().enumerate() {
        if s.abs() > silence_threshold {
            // Keep 50ms pre-roll before first voiced sound
            let pre_roll = ((sample_rate as f32) * 0.05) as usize;
            start_idx = i.saturating_sub(pre_roll);
            break;
        }
    }

    // Ensure we don't start so late that fewer than max_samples remain
    if start_idx + max_samples > samples.len() {
        start_idx = samples.len().saturating_sub(max_samples);
    }

    let end_idx = (start_idx + max_samples).min(samples.len());
    let mut cropped = samples[start_idx..end_idx].to_vec();

    // Apply smooth 50ms fade-out at the end to prevent click/boundary artifacts
    let fade_samples = ((sample_rate as f32) * 0.05) as usize;
    let crop_len = cropped.len();
    if crop_len > fade_samples {
        let fade_start = crop_len - fade_samples;
        for i in 0..fade_samples {
            let fade_factor = 1.0 - (i as f32 / fade_samples as f32);
            cropped[fade_start + i] *= fade_factor;
        }
    }

    (cropped, true, orig_duration)
}

pub fn write_wav_24k(path: &Path, samples: &[i16]) -> Result<(), String> {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }

    let spec = WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let file = File::create(path)
        .map_err(|e| format!("Failed to create output file {}: {}", path.display(), e))?;
    let writer = BufWriter::new(file);
    let mut wav_writer = WavWriter::new(writer, spec)
        .map_err(|e| format!("Failed to initialize WAV writer: {}", e))?;

    for &sample in samples {
        wav_writer
            .write_sample(sample)
            .map_err(|e| format!("Failed to write WAV sample: {}", e))?;
    }

    wav_writer
        .finalize()
        .map_err(|e| format!("Failed to finalize WAV file: {}", e))?;

    Ok(())
}

pub fn generate_silence(duration_ms: u32, sample_rate: u32) -> Vec<i16> {
    let count = (duration_ms as u64 * sample_rate as u64 / 1000) as usize;
    vec![0i16; count]
}

fn resample_linear(input: &[f32], src_sr: u32, dst_sr: u32) -> Vec<f32> {
    if input.is_empty() || src_sr == dst_sr {
        return input.to_vec();
    }

    let ratio = (src_sr as f64) / (dst_sr as f64);
    let dst_len = ((input.len() as f64) / ratio).round() as usize;
    let mut output = Vec::with_capacity(dst_len);

    for i in 0..dst_len {
        let src_pos = (i as f64) * ratio;
        let idx = src_pos.floor() as usize;
        let frac = (src_pos - (idx as f64)) as f32;

        if idx + 1 < input.len() {
            let sample = input[idx] * (1.0 - frac) + input[idx + 1] * frac;
            output.push(sample);
        } else if idx < input.len() {
            output.push(input[idx]);
        }
    }

    output
}
