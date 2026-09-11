use std::os::raw::{c_char, c_int};

#[repr(C)]
pub struct Style2TtsEngine {
    _unused: [u8; 0],
}

#[repr(C)]
pub struct Style2TtsOptions {
    pub models_dir: *const c_char,
    pub espeak_data_dir: *const c_char,
    pub use_cuda: c_int,
    pub num_threads: c_int,
}

extern "C" {
    pub fn style2tts_load_onnxruntime(custom_so_path: *const c_char) -> c_int;
    pub fn style2tts_create(options: *const Style2TtsOptions) -> *mut Style2TtsEngine;
    pub fn style2tts_destroy(engine: *mut Style2TtsEngine);

    pub fn style2tts_extract_style(
        engine: *mut Style2TtsEngine,
        audio_pcm: *const f32,
        num_samples: usize,
        out_style: *mut f32,
        style_len: usize,
        out_predictor: *mut f32,
        predictor_len: usize,
    ) -> c_int;

    pub fn style2tts_synthesize(
        engine: *mut Style2TtsEngine,
        text: *const c_char,
        style_emb: *const f32,
        style_len: usize,
        predictor_emb: *const f32,
        predictor_len: usize,
        speed: f32,
        steps: c_int,
        out_pcm: *mut *mut i16,
        out_samples: *mut usize,
    ) -> c_int;

    pub fn style2tts_free_audio(pcm: *mut i16);

    pub fn style2tts_get_last_error() -> *const c_char;
}
