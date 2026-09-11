use std::ffi::{CStr, CString};
use std::path::Path;
use std::ptr;

use crate::ffi;

pub struct StyleTtsEngine {
    raw: *mut ffi::Style2TtsEngine,
}

unsafe impl Send for StyleTtsEngine {}
unsafe impl Sync for StyleTtsEngine {}

impl Drop for StyleTtsEngine {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe {
                ffi::style2tts_destroy(self.raw);
            }
            self.raw = ptr::null_mut();
        }
    }
}

fn get_last_error() -> String {
    unsafe {
        let err_ptr = ffi::style2tts_get_last_error();
        if err_ptr.is_null() {
            "Unknown engine error".to_string()
        } else {
            CStr::from_ptr(err_ptr).to_string_lossy().into_owned()
        }
    }
}

impl StyleTtsEngine {
    pub fn new(
        models_dir: &Path,
        espeak_dir: &Path,
        use_cuda: bool,
        num_threads: usize,
    ) -> Result<Self, String> {
        let models_str = models_dir.to_str().ok_or("Invalid UTF-8 in models directory path")?;
        let espeak_str = espeak_dir.to_str().ok_or("Invalid UTF-8 in espeak directory path")?;

        let c_models = CString::new(models_str).map_err(|e| e.to_string())?;
        let c_espeak = CString::new(espeak_str).map_err(|e| e.to_string())?;

        // Probe local application directory for downloaded libonnxruntime.so
        let base_dir = models_dir.parent().unwrap_or(Path::new("."));
        let local_so = base_dir.join("lib").join("libonnxruntime.so");
        let candidate_so = if local_so.exists() {
            Some(local_so)
        } else {
            let root_so = base_dir.join("libonnxruntime.so");
            if root_so.exists() {
                Some(root_so)
            } else {
                None
            }
        };

        if let Some(p) = candidate_so {
            if let Some(s) = p.to_str() {
                if let Ok(c_so) = CString::new(s) {
                    unsafe {
                        ffi::style2tts_load_onnxruntime(c_so.as_ptr());
                    }
                }
            }
        }

        let options = ffi::Style2TtsOptions {
            models_dir: c_models.as_ptr(),
            espeak_data_dir: c_espeak.as_ptr(),
            use_cuda: if use_cuda { 1 } else { 0 },
            num_threads: num_threads as std::os::raw::c_int,
        };

        let raw = unsafe { ffi::style2tts_create(&options) };
        if raw.is_null() {
            return Err(get_last_error());
        }

        Ok(Self { raw })
    }

    pub fn extract_style(&self, audio_pcm: &[f32]) -> Result<(Vec<f32>, Vec<f32>), String> {
        if audio_pcm.is_empty() {
            return Err("Cannot extract style from empty audio buffer".to_string());
        }

        let mut style = vec![0.0f32; 128];
        let mut predictor = vec![0.0f32; 128];

        let ret = unsafe {
            ffi::style2tts_extract_style(
                self.raw,
                audio_pcm.as_ptr(),
                audio_pcm.len(),
                style.as_mut_ptr(),
                style.len(),
                predictor.as_mut_ptr(),
                predictor.len(),
            )
        };

        if ret != 0 {
            return Err(get_last_error());
        }

        Ok((style, predictor))
    }

    pub fn get_default_embeddings(&self) -> Result<(Vec<f32>, Vec<f32>), String> {
        let mut style = vec![0.0f32; 128];
        let mut predictor = vec![0.0f32; 128];

        let ret = unsafe {
            ffi::style2tts_get_default_embeddings(
                self.raw,
                style.as_mut_ptr(),
                style.len(),
                predictor.as_mut_ptr(),
                predictor.len(),
            )
        };

        if ret != 0 {
            return Err(get_last_error());
        }

        Ok((style, predictor))
    }

    pub fn synthesize(
        &self,
        text: &str,
        style: Option<&[f32]>,
        predictor: Option<&[f32]>,
        speed: f32,
        steps: i32,
    ) -> Result<Vec<i16>, String> {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            return Ok(Vec::new());
        }

        let c_text = CString::new(trimmed).map_err(|e| format!("Text contains NUL byte: {}", e))?;

        let (style_ptr, style_len) = match style {
            Some(s) => (s.as_ptr(), s.len()),
            None => (ptr::null(), 0),
        };

        let (pred_ptr, pred_len) = match predictor {
            Some(p) => (p.as_ptr(), p.len()),
            None => (ptr::null(), 0),
        };

        let mut out_pcm: *mut i16 = ptr::null_mut();
        let mut out_samples: usize = 0;

        let ret = unsafe {
            ffi::style2tts_synthesize(
                self.raw,
                c_text.as_ptr(),
                style_ptr,
                style_len,
                pred_ptr,
                pred_len,
                speed,
                steps,
                &mut out_pcm,
                &mut out_samples,
            )
        };

        if ret != 0 {
            return Err(get_last_error());
        }

        if out_pcm.is_null() || out_samples == 0 {
            return Ok(Vec::new());
        }

        let samples = unsafe {
            let slice = std::slice::from_raw_parts(out_pcm, out_samples);
            let vec = slice.to_vec();
            ffi::style2tts_free_audio(out_pcm);
            vec
        };

        Ok(samples)
    }
}
