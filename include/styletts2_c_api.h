#ifndef STYLE2TTS_C_API_H
#define STYLE2TTS_C_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque engine pointer
typedef struct Style2TtsEngine Style2TtsEngine;

// Creation options
typedef struct {
    const char* models_dir;
    const char* espeak_data_dir;
    int use_cuda;           // 0 = CPU, 1 = CUDA
    int num_threads;        // 0 = auto
} Style2TtsOptions;

/**
 * Dynamically loads the ONNX Runtime shared library.
 * custom_so_path: optional path to libonnxruntime.so, or NULL to search standard paths.
 * Returns 0 on success, negative error code on failure.
 */
int style2tts_load_onnxruntime(const char* custom_so_path);

/**
 * Creates and initializes the StyleTTS2 engine.
 * Returns NULL on failure; call style2tts_get_last_error() for details.
 */
Style2TtsEngine* style2tts_create(const Style2TtsOptions* options);

/**
 * Destroys the StyleTTS2 engine and frees all allocated models and memory.
 */
void style2tts_destroy(Style2TtsEngine* engine);

/**
 * Extracts style embedding and predictor embedding from raw 24kHz float PCM audio.
 * out_style: pointer to float buffer of at least 128 elements.
 * out_predictor: pointer to float buffer of at least 128 elements.
 * Returns 0 on success, negative error code on failure.
 */
int style2tts_extract_style(
    Style2TtsEngine* engine,
    const float* audio_pcm,
    size_t num_samples,
    float* out_style,
    size_t style_len,
    float* out_predictor,
    size_t predictor_len
);

/**
 * Synthesizes speech for a single text clause or sentence.
 * style_emb: pointer to 128 float values (or NULL to use default/preset)
 * predictor_emb: pointer to 128 float values (or NULL to use default/preset)
 * speed: speech speed (e.g. 1.0f)
 * steps: inference step count
 * out_pcm: output pointer to allocated int16 PCM samples (caller must free with style2tts_free_audio)
 * out_samples: pointer to receive number of int16 samples generated
 * Returns 0 on success, negative error code on failure.
 */
int style2tts_synthesize(
    Style2TtsEngine* engine,
    const char* text,
    const float* style_emb,
    size_t style_len,
    const float* predictor_emb,
    size_t predictor_len,
    float speed,
    int steps,
    int16_t** out_pcm,
    size_t* out_samples
);

/**
 * Frees audio buffer allocated by style2tts_synthesize.
 */
void style2tts_free_audio(int16_t* pcm);

/**
 * Retrieves the last error message for the current thread.
 */
const char* style2tts_get_last_error(void);

#ifdef __cplusplus
}
#endif

#endif // STYLE2TTS_C_API_H
