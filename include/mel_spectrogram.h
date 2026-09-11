#ifndef MEL_SPECTROGRAM_H
#define MEL_SPECTROGRAM_H

#include <vector>
#include <cstddef>

class MelSpectrogramExtractor {
public:
    MelSpectrogramExtractor(
        int sampleRate = 24000,
        int nFft = 2048,
        int winLength = 1200,
        int hopLength = 300,
        int nMels = 80,
        float fMin = 0.0f,
        float fMax = 12000.0f
    );

    // Converts audio PCM samples (mono, 24kHz float in [-1.0, 1.0]) to Mel-Spectrogram
    // Result is flat vector of size (nMels * numFrames) in row-major order: [nMels, numFrames]
    std::vector<float> compute_mel(const float* pcm, size_t numSamples, int& outNumFrames);

    int get_n_mels() const { return nMels_; }

private:
    int sampleRate_;
    int nFft_;
    int winLength_;
    int hopLength_;
    int nMels_;
    float fMin_;
    float fMax_;

    std::vector<float> window_;
    std::vector<std::vector<float>> melFilters_; // [nMels_][nFft_/2 + 1]

    void init_window();
    void init_mel_filters();
    void fft2048(const float* in, float* outReal, float* outImag);
};

#endif // MEL_SPECTROGRAM_H
