#include "styletts2_engine.h"

#include <iostream>
#include <fstream>
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <cmath>
#if defined(__unix__) || defined(__APPLE__)
#include <unistd.h>
#include <fcntl.h>
#endif

static std::vector<float> load_binary_floats(const std::string& filepath, size_t expectedCount) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return {};
    }
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    size_t count = size / sizeof(float);
    std::vector<float> buffer(count);
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        if (expectedCount > 0 && buffer.size() < expectedCount) {
            buffer.resize(expectedCount, 0.0f);
        }
        return buffer;
    }
    return {};
}

StyleTTS2Engine::StyleTTS2Engine(
    const std::string& modelsDir,
    const std::string& espeakDataDir,
    bool useCuda,
    int numThreads
) : env_(ORT_LOGGING_LEVEL_ERROR, "StyleTTS2Engine"),
    sessionOptions_(),
    melExtractor_(24000, 2048, 1200, 300, 80, 0.0f, 12000.0f)
{
    // Initialize Phonemizer
    phonemizer_.Init("en-us", espeakDataDir);

    // Disable telemetry
    env_.DisableTelemetryEvents();

    if (numThreads > 0) {
        sessionOptions_.SetIntraOpNumThreads(numThreads);
        sessionOptions_.SetInterOpNumThreads(numThreads);
    }

    if (useCuda) {
        #ifdef USE_CUDA
        OrtCUDAProviderOptions cuda_options{};
        sessionOptions_.AppendExecutionProvider_CUDA(cuda_options);
        #endif
    }

    // Keep memory usage deterministic
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    sessionOptions_.DisableCpuMemArena();
    sessionOptions_.DisableMemPattern();


    // Model paths
    std::string plBertPath = modelsDir + "/plbert_simp.onnx";
    std::string bertEncoderPath = modelsDir + "/bert_encoder.onnx";
    std::string finalModelPath = modelsDir + "/final_simp.onnx";
    std::string styleEncoderPath = modelsDir + "/style_encoder_simp.onnx";
    std::string predictorEncoderPath = modelsDir + "/predictor_encoder_simp.onnx";

#if defined(__unix__) || defined(__APPLE__)
    int saved_stderr = dup(STDERR_FILENO);
    int dev_null = open("/dev/null", O_WRONLY);
    if (dev_null != -1) {
        dup2(dev_null, STDERR_FILENO);
        close(dev_null);
    }
#endif

    plBert_ = std::make_unique<Ort::Session>(env_, plBertPath.c_str(), sessionOptions_);
    bertEncoder_ = std::make_unique<Ort::Session>(env_, bertEncoderPath.c_str(), sessionOptions_);
    finalModel_ = std::make_unique<Ort::Session>(env_, finalModelPath.c_str(), sessionOptions_);

    // Voice cloning models (style & predictor encoders)
    std::ifstream fStyle(styleEncoderPath);
    if (fStyle.good()) {
        styleEncoder_ = std::make_unique<Ort::Session>(env_, styleEncoderPath.c_str(), sessionOptions_);
    }
    std::ifstream fPred(predictorEncoderPath);
    if (fPred.good()) {
        predictorEncoder_ = std::make_unique<Ort::Session>(env_, predictorEncoderPath.c_str(), sessionOptions_);
    }

#if defined(__unix__) || defined(__APPLE__)
    if (saved_stderr != -1) {
        dup2(saved_stderr, STDERR_FILENO);
        close(saved_stderr);
    }
#endif

    // Default embeddings
    load_default_embeddings(modelsDir);
}

StyleTTS2Engine::~StyleTTS2Engine() {
    // Unique pointers automatically cleanup Ort::Sessions
}

void StyleTTS2Engine::load_default_embeddings(const std::string& modelsDir) {
    defaultStyle_ = load_binary_floats(modelsDir + "/ref_s.bin", 128);
    defaultPredictor_ = load_binary_floats(modelsDir + "/ref_p.bin", 128);

    if (defaultStyle_.size() < 128) {
        defaultStyle_.assign(128, 0.0f);
    }
    if (defaultPredictor_.size() < 128) {
        defaultPredictor_.assign(128, 0.0f);
    }
}

void StyleTTS2Engine::extract_style(
    const float* audioPcm,
    size_t numSamples,
    float* outStyle,
    size_t styleLen,
    float* outPredictor,
    size_t predictorLen
) {
    if (!styleEncoder_ || !predictorEncoder_) {
        throw std::runtime_error("Style encoder or predictor encoder model is not loaded");
    }
    if (audioPcm == nullptr || numSamples == 0) {
        throw std::runtime_error("Invalid audio PCM input for style extraction");
    }
    if (styleLen < 128 || predictorLen < 128) {
        throw std::runtime_error("Style and predictor output buffers must be at least 128 floats");
    }

    int numFrames = 0;
    std::vector<float> mel = melExtractor_.compute_mel(audioPcm, numSamples, numFrames);
    if (numFrames == 0 || mel.empty()) {
        throw std::runtime_error("Failed to compute Mel spectrogram from audio PCM");
    }

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // PyTorch tensor shape for style encoder: [1, 1, 80, numFrames]
    std::vector<int64_t> melShape = {1, 1, 80, static_cast<int64_t>(numFrames)};
    Ort::Value melTensor = Ort::Value::CreateTensor<float>(
        memoryInfo,
        mel.data(),
        mel.size(),
        melShape.data(),
        melShape.size()
    );

    Ort::AllocatorWithDefaultOptions allocator;
    auto styleInName = styleEncoder_->GetInputNameAllocated(0, allocator);
    auto styleOutName = styleEncoder_->GetOutputNameAllocated(0, allocator);
    const char* styleInputNames[] = { styleInName.get() };
    const char* styleOutputNames[] = { styleOutName.get() };

    // 1. Run Style Encoder -> ref_s [1, 128]
    auto styleOut = styleEncoder_->Run(
        Ort::RunOptions{nullptr},
        styleInputNames,
        &melTensor,
        1,
        styleOutputNames,
        1
    );

    const float* sData = styleOut.front().GetTensorData<float>();
    std::copy(sData, sData + 128, outStyle);

    auto predInName = predictorEncoder_->GetInputNameAllocated(0, allocator);
    auto predOutName = predictorEncoder_->GetOutputNameAllocated(0, allocator);
    const char* predInputNames[] = { predInName.get() };
    const char* predOutputNames[] = { predOutName.get() };

    // 2. Run Predictor Encoder -> ref_p [1, 128]
    auto predictorOut = predictorEncoder_->Run(
        Ort::RunOptions{nullptr},
        predInputNames,
        &melTensor,
        1,
        predOutputNames,
        1
    );

    const float* pData = predictorOut.front().GetTensorData<float>();
    std::copy(pData, pData + 128, outPredictor);

    // Explicit cleanup
    melTensor.release();
    for (auto& t : styleOut) t.release();
    for (auto& t : predictorOut) t.release();
}

std::vector<int16_t> StyleTTS2Engine::synthesize(
    const std::string& text,
    const float* styleEmb,
    size_t styleLen,
    const float* predictorEmb,
    size_t predictorLen,
    float speed,
    int /*steps*/
) {
    if (text.empty()) {
        return {};
    }

    // 1. Phonemize
    std::vector<int64_t> tokens = phonemizer_.text_to_sequence(text);
    if (tokens.empty()) {
        return {};
    }
    // Prepend token 0 (start token)
    tokens.insert(tokens.begin(), 0);

    size_t seqLen = tokens.size();
    std::vector<int32_t> attentionMask(seqLen, 1);
    std::vector<int64_t> phonemeShape = {1, static_cast<int64_t>(seqLen)};

    auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // 2. plBert Session
    std::vector<Ort::Value> plBertInputs;
    plBertInputs.push_back(Ort::Value::CreateTensor<int64_t>(
        memoryInfo, tokens.data(), tokens.size(), phonemeShape.data(), phonemeShape.size()
    ));
    plBertInputs.push_back(Ort::Value::CreateTensor<int32_t>(
        memoryInfo, attentionMask.data(), attentionMask.size(), phonemeShape.data(), phonemeShape.size()
    ));

    const char* plBertInputNames[] = {"input_ids", "attention_mask"};
    const char* plBertOutputNames[] = {"bert_dur"};

    auto bertDur = plBert_->Run(
        Ort::RunOptions{nullptr},
        plBertInputNames,
        plBertInputs.data(),
        plBertInputs.size(),
        plBertOutputNames,
        1
    );

    const float* bertDurData = bertDur.front().GetTensorData<float>();
    std::vector<int64_t> bertDurShape = bertDur.front().GetTensorTypeAndShapeInfo().GetShape();
    int64_t totalElements = 1;
    for (auto dim : bertDurShape) {
        totalElements *= dim;
    }
    std::vector<float> bertDurVec(bertDurData, bertDurData + totalElements);

    // Release plBert input tensors immediately
    for (auto& t : plBertInputs) t.release();

    // 3. bertEncoder Session
    std::vector<Ort::Value> bertEncoderInputs;
    bertEncoderInputs.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, bertDurVec.data(), bertDurVec.size(), bertDurShape.data(), bertDurShape.size()
    ));

    const char* bertEncoderInputNames[] = {"input"};
    const char* bertEncoderOutputNames[] = {"d_en"};

    auto dEn = bertEncoder_->Run(
        Ort::RunOptions{nullptr},
        bertEncoderInputNames,
        bertEncoderInputs.data(),
        bertEncoderInputs.size(),
        bertEncoderOutputNames,
        1
    );

    const float* dEnData = dEn.front().GetTensorData<float>();
    std::vector<int64_t> dEnShape = dEn.front().GetTensorTypeAndShapeInfo().GetShape();

    int64_t batch = dEnShape[0];
    int64_t dim1 = dEnShape[1];
    int64_t dim2 = dEnShape[2];
    std::vector<float> dEnTransposed(batch * dim1 * dim2);

    // Transpose [b, i, j] -> [b, j, i]
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t i = 0; i < dim1; ++i) {
            for (int64_t j = 0; j < dim2; ++j) {
                int64_t oldIdx = b * (dim1 * dim2) + i * dim2 + j;
                int64_t newIdx = b * (dim2 * dim1) + j * dim1 + i;
                dEnTransposed[newIdx] = dEnData[oldIdx];
            }
        }
    }
    std::vector<int64_t> dEnTransposedShape = {batch, dim2, dim1};

    // Release bertEncoder intermediate tensors
    for (auto& t : bertEncoderInputs) t.release();
    for (auto& t : bertDur) t.release();

    // 4. Prepare Final Model (Vocoder / Generator)
    const float* activeStyle = (styleEmb != nullptr && styleLen >= 128) ? styleEmb : defaultStyle_.data();
    const float* activePred = (predictorEmb != nullptr && predictorLen >= 128) ? predictorEmb : defaultPredictor_.data();

    std::vector<int64_t> refShape = {1, 128};
    std::vector<float> speedVec = {speed};
    std::vector<int64_t> speedShape = {1};

    std::vector<Ort::Value> finalInputs;
    finalInputs.push_back(Ort::Value::CreateTensor<int64_t>(
        memoryInfo, tokens.data(), tokens.size(), phonemeShape.data(), phonemeShape.size()
    ));
    finalInputs.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, dEnTransposed.data(), dEnTransposed.size(), dEnTransposedShape.data(), dEnTransposedShape.size()
    ));
    finalInputs.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, const_cast<float*>(activePred), 128, refShape.data(), refShape.size()
    ));
    finalInputs.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, const_cast<float*>(activeStyle), 128, refShape.data(), refShape.size()
    ));
    finalInputs.push_back(Ort::Value::CreateTensor<float>(
        memoryInfo, speedVec.data(), speedVec.size(), speedShape.data(), speedShape.size()
    ));

    const char* finalInputNames[] = {"tokens", "d_en", "ref", "s", "speed"};
    const char* finalOutputNames[] = {"output_wav"};

    auto audioOutput = finalModel_->Run(
        Ort::RunOptions{nullptr},
        finalInputNames,
        finalInputs.data(),
        finalInputs.size(),
        finalOutputNames,
        1
    );

    const float* audioData = audioOutput.front().GetTensorData<float>();
    std::vector<int64_t> audioShape = audioOutput.front().GetTensorTypeAndShapeInfo().GetShape();
    int64_t numAudioSamples = audioShape[audioShape.size() - 1];

    std::vector<int16_t> audioBuffer;
    audioBuffer.reserve(numAudioSamples);

    const float MAX_WAV_VALUE = 32767.0f;
    for (int64_t i = 0; i < numAudioSamples; ++i) {
        float sample = audioData[i] * MAX_WAV_VALUE;
        float clamped = std::clamp(sample, -32768.0f, 32767.0f);
        audioBuffer.push_back(static_cast<int16_t>(clamped));
    }

    // Clean up transient ONNX tensors
    for (auto& t : finalInputs) t.release();
    for (auto& t : audioOutput) t.release();
    for (auto& t : dEn) t.release();

    return audioBuffer;
}
