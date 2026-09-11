use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

use crate::engine::StyleTtsEngine;

pub struct StyleCache {
    cache_dir: PathBuf,
    memory_cache: HashMap<String, (Vec<f32>, Vec<f32>)>,
}

impl StyleCache {
    pub fn new(cache_dir: PathBuf) -> Self {
        let styles_dir = cache_dir.join("styles");
        let _ = fs::create_dir_all(&styles_dir);

        Self {
            cache_dir: styles_dir,
            memory_cache: HashMap::new(),
        }
    }

    pub fn get_or_extract(
        &mut self,
        engine: &StyleTtsEngine,
        ref_audio_path: &Path,
        audio_pcm: &[f32],
    ) -> Result<(Vec<f32>, Vec<f32>), String> {
        // 1. Calculate SHA-256 hash of reference audio file content
        let mut file = File::open(ref_audio_path)
            .map_err(|e| format!("Failed to open reference audio {}: {}", ref_audio_path.display(), e))?;
        let mut hasher = Sha256::new();
        let mut buffer = [0u8; 8192];
        loop {
            let n = file.read(&mut buffer).map_err(|e| e.to_string())?;
            if n == 0 {
                break;
            }
            hasher.update(&buffer[..n]);
        }
        let hash = format!("{:x}", hasher.finalize());

        // 2. Check in-memory cache
        if let Some(cached) = self.memory_cache.get(&hash) {
            println!(" [Cache] Using in-memory cached voice style for {}", ref_audio_path.display());
            return Ok(cached.clone());
        }

        // 3. Check disk cache
        let cache_file = self.cache_dir.join(format!("{}.style", hash));
        if cache_file.exists() {
            if let Ok(loaded) = Self::read_style_file(&cache_file) {
                println!(" [Cache] Loaded persistent style cache from {}", cache_file.display());
                self.memory_cache.insert(hash, loaded.clone());
                return Ok(loaded);
            }
        }

        // 4. Extract style using C++ engine Mel-spectrogram and style encoder
        println!(" [Voice Cloning] Analyzing acoustic features from {}...", ref_audio_path.display());
        let (style, predictor) = engine.extract_style(audio_pcm)?;

        // 5. Persist to disk cache
        let _ = Self::write_style_file(&cache_file, &style, &predictor);

        // Also save with filename stem for friendly voice selection
        if let Some(stem) = ref_audio_path.file_stem().and_then(|s| s.to_str()) {
            let named_file = self.cache_dir.join(format!("{}.style", stem));
            let _ = Self::write_style_file(&named_file, &style, &predictor);
        }

        println!(" [Cache] Saved persistent style cache to {}", cache_file.display());
        self.memory_cache.insert(hash, (style.clone(), predictor.clone()));

        Ok((style, predictor))
    }

    pub fn load_named_voice(&mut self, name: &str) -> Option<(Vec<f32>, Vec<f32>)> {
        // Check in-memory cache
        if let Some(cached) = self.memory_cache.get(name) {
            return Some(cached.clone());
        }

        // Check file by name
        let style_file = self.cache_dir.join(format!("{}.style", name));
        if style_file.exists() {
            if let Ok(loaded) = Self::read_style_file(&style_file) {
                self.memory_cache.insert(name.to_string(), loaded.clone());
                return Some(loaded);
            }
        }

        None
    }

    pub fn list_cached_voices(&self) -> Vec<String> {
        let mut voices = Vec::new();
        if let Ok(entries) = fs::read_dir(&self.cache_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() && path.extension().and_then(|s| s.to_str()) == Some("style") {
                    if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                        voices.push(stem.to_string());
                    }
                }
            }
        }
        voices.sort();
        voices
    }

    pub fn clear_cache(&self) -> Result<usize, String> {
        let mut count = 0;
        if let Ok(entries) = fs::read_dir(&self.cache_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_file() {
                    let _ = fs::remove_file(path);
                    count += 1;
                }
            }
        }
        Ok(count)
    }

    fn read_style_file(path: &Path) -> Result<(Vec<f32>, Vec<f32>), String> {
        let mut file = File::open(path).map_err(|e| e.to_string())?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes).map_err(|e| e.to_string())?;

        // Expect 256 floats (128 style + 128 predictor) = 1024 bytes
        if bytes.len() != 256 * 4 {
            return Err("Corrupted style cache file: invalid length".to_string());
        }

        let mut floats = Vec::with_capacity(256);
        for chunk in bytes.chunks_exact(4) {
            let val = f32::from_le_bytes(chunk.try_into().unwrap());
            floats.push(val);
        }

        let style = floats[0..128].to_vec();
        let predictor = floats[128..256].to_vec();

        Ok((style, predictor))
    }

    fn write_style_file(path: &Path, style: &[f32], predictor: &[f32]) -> Result<(), String> {
        if style.len() != 128 || predictor.len() != 128 {
            return Err("Style and predictor must both have 128 elements".to_string());
        }

        let mut file = File::create(path).map_err(|e| e.to_string())?;
        let mut bytes = Vec::with_capacity(256 * 4);

        for &val in style.iter().chain(predictor.iter()) {
            bytes.extend_from_slice(&val.to_le_bytes());
        }

        file.write_all(&bytes).map_err(|e| e.to_string())?;
        Ok(())
    }
}
