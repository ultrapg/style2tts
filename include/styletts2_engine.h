#ifndef STYLE2TTS_ENGINE_H
#define STYLE2TTS_ENGINE_H

#include <string>
#include <vector>
#include <memory>
#include <cstdint>

#ifndef ORT_API_MANUAL_INIT
#define ORT_API_MANUAL_INIT
#endif
#include "onnxruntime_cxx_api.h"
#include "phonemize.h"
#include "mel_spectrogram.h"

class StyleTTS2Engine {
public:
    StyleTTS2Engine(
        const std::string& modelsDir,
        const std::string& espeakDataDir,
        bool useCuda = false,
        int numThreads = 0
    );
    ~StyleTTS2Engine();

    // Extract 128-dim style embedding and 128-dim predictor embedding from 24kHz float PCM
    void extract_style(
        const float* audioPcm,
        size_t numSamples,
        float* outStyle,
        size_t styleLen,
        float* outPredictor,
        size_t predictorLen
    );

    // Synthesize speech from text
    std::vector<int16_t> synthesize(
        const std::string& text,
        const float* styleEmb,
        size_t styleLen,
        const float* predictorEmb,
        size_t predictorLen,
        float speed = 1.0f,
        int steps = 5
    );

    const std::vector<float>& get_default_style() const { return defaultStyle_; }
    const std::vector<float>& get_default_predictor() const { return defaultPredictor_; }

private:
    Ort::Env env_;
    Ort::SessionOptions sessionOptions_;

    std::unique_ptr<Ort::Session> plBert_;
    std::unique_ptr<Ort::Session> bertEncoder_;
    std::unique_ptr<Ort::Session> finalModel_;
    std::unique_ptr<Ort::Session> styleEncoder_;
    std::unique_ptr<Ort::Session> predictorEncoder_;

    PhonemizerEngine phonemizer_;
    MelSpectrogramExtractor melExtractor_;

    std::vector<float> defaultStyle_;
    std::vector<float> defaultPredictor_;

    void load_default_embeddings(const std::string& modelsDir);
};

#endif // STYLE2TTS_ENGINE_H
