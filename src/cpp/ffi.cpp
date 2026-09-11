#include "styletts2_c_api.h"
#include "styletts2_engine.h"

#include <string>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <dlfcn.h>

struct Style2TtsEngine {
    std::unique_ptr<StyleTTS2Engine> instance;
};

static thread_local std::string g_last_error = "";
static void* g_ort_lib_handle = nullptr;

static void set_error(const std::string& msg) {
    g_last_error = msg;
}

extern "C" {

int style2tts_load_onnxruntime(const char* custom_so_path) {
    if (Ort::Global<void>::api_ != nullptr) {
        return 0; // Already loaded and initialized
    }

    std::vector<std::string> candidates;
    if (custom_so_path && strlen(custom_so_path) > 0) {
        candidates.push_back(custom_so_path);
    }
    candidates.push_back("./lib/libonnxruntime.so");
    candidates.push_back("./libonnxruntime.so");
    candidates.push_back("libonnxruntime.so");
    candidates.push_back("/usr/lib/x86_64-linux-gnu/libonnxruntime.so.1.23");
    candidates.push_back("/usr/lib/x86_64-linux-gnu/libonnxruntime.so");
    candidates.push_back("/usr/local/lib/libonnxruntime.so");

    void* handle = nullptr;
    for (const auto& path : candidates) {
        handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (handle) {
            break;
        }
    }

    if (!handle) {
        const char* err = dlerror();
        set_error(std::string("Failed to load libonnxruntime.so: ") + (err ? err : "not found"));
        return -1;
    }

    typedef const OrtApiBase* (*OrtGetApiBaseFn)(void);
    OrtGetApiBaseFn get_api_base = reinterpret_cast<OrtGetApiBaseFn>(dlsym(handle, "OrtGetApiBase"));
    if (!get_api_base) {
        set_error("Symbol 'OrtGetApiBase' not found in loaded libonnxruntime.so");
        dlclose(handle);
        return -2;
    }

    const OrtApiBase* base = get_api_base();
    if (!base) {
        set_error("OrtGetApiBase() returned NULL");
        dlclose(handle);
        return -3;
    }

    const OrtApi* api = base->GetApi(ORT_API_VERSION);
    if (!api) {
        set_error("base->GetApi(ORT_API_VERSION) returned NULL");
        dlclose(handle);
        return -4;
    }

    Ort::InitApi(api);
    g_ort_lib_handle = handle;
    return 0;
}

Style2TtsEngine* style2tts_create(const Style2TtsOptions* options) {
    if (options == nullptr) {
        set_error("Options pointer is NULL");
        return nullptr;
    }

    if (style2tts_load_onnxruntime(nullptr) != 0) {
        return nullptr;
    }

    std::string modelsDir = options->models_dir ? options->models_dir : "models";
    std::string espeakDir = options->espeak_data_dir ? options->espeak_data_dir : "espeak-ng-data";
    bool useCuda = (options->use_cuda != 0);
    int numThreads = options->num_threads;

    try {
        auto engine = std::make_unique<Style2TtsEngine>();
        engine->instance = std::make_unique<StyleTTS2Engine>(
            modelsDir,
            espeakDir,
            useCuda,
            numThreads
        );
        return engine.release();
    } catch (const std::exception& ex) {
        set_error(std::string("Engine creation failed: ") + ex.what());
        return nullptr;
    } catch (...) {
        set_error("Engine creation failed with unknown exception");
        return nullptr;
    }
}

void style2tts_destroy(Style2TtsEngine* engine) {
    if (engine != nullptr) {
        delete engine;
    }
}

int style2tts_extract_style(
    Style2TtsEngine* engine,
    const float* audio_pcm,
    size_t num_samples,
    float* out_style,
    size_t style_len,
    float* out_predictor,
    size_t predictor_len
) {
    if (engine == nullptr || engine->instance == nullptr) {
        set_error("Engine is NULL");
        return -1;
    }

    try {
        engine->instance->extract_style(
            audio_pcm,
            num_samples,
            out_style,
            style_len,
            out_predictor,
            predictor_len
        );
        return 0;
    } catch (const std::exception& ex) {
        set_error(std::string("Style extraction failed: ") + ex.what());
        return -2;
    } catch (...) {
        set_error("Style extraction failed with unknown exception");
        return -3;
    }
}

int style2tts_get_default_embeddings(
    Style2TtsEngine* engine,
    float* out_style,
    size_t style_len,
    float* out_predictor,
    size_t predictor_len
) {
    if (engine == nullptr || engine->instance == nullptr) {
        set_error("Engine is NULL");
        return -1;
    }

    try {
        const auto& defStyle = engine->instance->get_default_style();
        const auto& defPred = engine->instance->get_default_predictor();

        if (out_style && style_len >= 128 && defStyle.size() >= 128) {
            std::copy(defStyle.begin(), defStyle.begin() + 128, out_style);
        }
        if (out_predictor && predictor_len >= 128 && defPred.size() >= 128) {
            std::copy(defPred.begin(), defPred.begin() + 128, out_predictor);
        }
        return 0;
    } catch (const std::exception& ex) {
        set_error(std::string("Failed to get default embeddings: ") + ex.what());
        return -2;
    } catch (...) {
        set_error("Failed to get default embeddings with unknown exception");
        return -3;
    }
}

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
) {
    if (engine == nullptr || engine->instance == nullptr) {
        set_error("Engine is NULL");
        return -1;
    }
    if (text == nullptr || out_pcm == nullptr || out_samples == nullptr) {
        set_error("Invalid parameter: text, out_pcm, or out_samples is NULL");
        return -2;
    }

    try {
        std::vector<int16_t> audio = engine->instance->synthesize(
            text,
            style_emb,
            style_len,
            predictor_emb,
            predictor_len,
            speed,
            steps
        );

        if (audio.empty()) {
            *out_pcm = nullptr;
            *out_samples = 0;
            return 0;
        }

        size_t count = audio.size();
        int16_t* buffer = static_cast<int16_t*>(std::malloc(count * sizeof(int16_t)));
        if (buffer == nullptr) {
            set_error("Memory allocation failed for output audio");
            return -4;
        }

        std::memcpy(buffer, audio.data(), count * sizeof(int16_t));
        *out_pcm = buffer;
        *out_samples = count;
        return 0;
    } catch (const std::exception& ex) {
        set_error(std::string("Synthesis failed: ") + ex.what());
        return -5;
    } catch (...) {
        set_error("Synthesis failed with unknown exception");
        return -6;
    }
}

void style2tts_free_audio(int16_t* pcm) {
    if (pcm != nullptr) {
        std::free(pcm);
    }
}

const char* style2tts_get_last_error(void) {
    return g_last_error.c_str();
}

} // extern "C"
