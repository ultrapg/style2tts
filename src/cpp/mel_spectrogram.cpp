#include "mel_spectrogram.h"

#include <cmath>
#include <algorithm>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static inline float hz_to_mel(float hz) {
    return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

static inline float mel_to_hz(float mel) {
    return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

MelSpectrogramExtractor::MelSpectrogramExtractor(
    int sampleRate,
    int nFft,
    int winLength,
    int hopLength,
    int nMels,
    float fMin,
    float fMax
) : sampleRate_(sampleRate),
    nFft_(nFft),
    winLength_(winLength),
    hopLength_(hopLength),
    nMels_(nMels),
    fMin_(fMin),
    fMax_(fMax)
{
    init_window();
    init_mel_filters();
}

void MelSpectrogramExtractor::init_window() {
    window_.resize(winLength_);
    for (int i = 0; i < winLength_; ++i) {
        // Periodic Hanning window matching PyTorch / Torchaudio
        window_[i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * static_cast<float>(i) / static_cast<float>(winLength_)));
    }
}

void MelSpectrogramExtractor::init_mel_filters() {
    int numBins = nFft_ / 2 + 1;
    melFilters_.assign(nMels_, std::vector<float>(numBins, 0.0f));

    float melMin = hz_to_mel(fMin_);
    float melMax = hz_to_mel(fMax_);
    std::vector<float> melPoints(nMels_ + 2);

    for (int i = 0; i < nMels_ + 2; ++i) {
        melPoints[i] = melMin + (melMax - melMin) * static_cast<float>(i) / static_cast<float>(nMels_ + 1);
    }

    std::vector<float> binPoints(nMels_ + 2);
    for (int i = 0; i < nMels_ + 2; ++i) {
        float hz = mel_to_hz(melPoints[i]);
        binPoints[i] = (static_cast<float>(nFft_) * hz) / static_cast<float>(sampleRate_);
    }

    for (int m = 1; m <= nMels_; ++m) {
        float left = binPoints[m - 1];
        float center = binPoints[m];
        float right = binPoints[m + 1];

        for (int k = 0; k < numBins; ++k) {
            float kf = static_cast<float>(k);
            if (kf >= left && kf <= center && center > left) {
                melFilters_[m - 1][k] = (kf - left) / (center - left);
            } else if (kf >= center && kf <= right && right > center) {
                melFilters_[m - 1][k] = (right - kf) / (right - center);
            }
        }
    }
}

// 2048-point Radix-2 Cooley-Tukey in-place FFT
void MelSpectrogramExtractor::fft2048(const float* in, float* outReal, float* outImag) {
    const int N = 2048;
    // Bit reversal permutation
    for (int i = 0; i < N; ++i) {
        int rev = 0;
        int temp = i;
        for (int j = 0; j < 11; ++j) {
            rev = (rev << 1) | (temp & 1);
            temp >>= 1;
        }
        outReal[rev] = in[i];
        outImag[rev] = 0.0f;
    }

    // Cooley-Tukey stages
    for (int len = 2; len <= N; len <<= 1) {
        float angle = -2.0f * static_cast<float>(M_PI) / static_cast<float>(len);
        float wlenReal = std::cos(angle);
        float wlenImag = std::sin(angle);
        int halfLen = len >> 1;

        for (int i = 0; i < N; i += len) {
            float wReal = 1.0f;
            float wImag = 0.0f;
            for (int j = 0; j < halfLen; ++j) {
                int uIdx = i + j;
                int vIdx = i + j + halfLen;

                float uReal = outReal[uIdx];
                float uImag = outImag[uIdx];
                float vReal = outReal[vIdx] * wReal - outImag[vIdx] * wImag;
                float vImag = outReal[vIdx] * wImag + outImag[vIdx] * wReal;

                outReal[uIdx] = uReal + vReal;
                outImag[uIdx] = uImag + vImag;
                outReal[vIdx] = uReal - vReal;
                outImag[vIdx] = uImag - vImag;

                float nextWReal = wReal * wlenReal - wImag * wlenImag;
                float nextWImag = wReal * wlenImag + wImag * wlenReal;
                wReal = nextWReal;
                wImag = nextWImag;
            }
        }
    }
}

std::vector<float> MelSpectrogramExtractor::compute_mel(const float* pcm, size_t numSamples, int& outNumFrames) {
    if (numSamples == 0) {
        outNumFrames = 0;
        return {};
    }

    int pad = nFft_ / 2; // 1024
    size_t paddedLen = numSamples + 2 * pad;
    std::vector<float> padded(paddedLen);

    // Reflect padding (torch pad_mode='reflect')
    for (int i = 0; i < pad; ++i) {
        int idx = pad - i;
        if (idx >= static_cast<int>(numSamples)) idx = numSamples - 1;
        padded[i] = pcm[idx];
    }
    for (size_t i = 0; i < numSamples; ++i) {
        padded[pad + i] = pcm[i];
    }
    for (int i = 0; i < pad; ++i) {
        int idx = static_cast<int>(numSamples) - 2 - i;
        if (idx < 0) idx = 0;
        padded[pad + numSamples + i] = pcm[idx];
    }

    int numFrames = static_cast<int>((paddedLen - nFft_) / hopLength_) + 1;
    if (numFrames <= 0) {
        outNumFrames = 0;
        return {};
    }

    outNumFrames = numFrames;
    int numBins = nFft_ / 2 + 1; // 1025

    // Result in shape [nMels_, numFrames]
    std::vector<float> melOutput(nMels_ * numFrames, 0.0f);

    std::vector<float> frameBuffer(nFft_, 0.0f);
    std::vector<float> fftReal(nFft_);
    std::vector<float> fftImag(nFft_);
    std::vector<float> powerSpectrum(numBins);

    // Center window alignment offset
    int windowOffset = (nFft_ - winLength_) / 2;

    for (int t = 0; t < numFrames; ++t) {
        size_t startSample = t * hopLength_;

        std::fill(frameBuffer.begin(), frameBuffer.end(), 0.0f);
        for (int i = 0; i < winLength_; ++i) {
            size_t sampleIdx = startSample + windowOffset + i;
            if (sampleIdx < paddedLen) {
                frameBuffer[windowOffset + i] = padded[sampleIdx] * window_[i];
            }
        }

        fft2048(frameBuffer.data(), fftReal.data(), fftImag.data());

        // Power spectrum
        for (int k = 0; k < numBins; ++k) {
            powerSpectrum[k] = fftReal[k] * fftReal[k] + fftImag[k] * fftImag[k];
        }

        // Mel filterbank multiplication & log scaling
        for (int m = 0; m < nMels_; ++m) {
            float melEnergy = 0.0f;
            const float* filter = melFilters_[m].data();
            for (int k = 0; k < numBins; ++k) {
                melEnergy += filter[k] * powerSpectrum[k];
            }
            // Torchaudio normalization: (log(1e-5 + mel) - (-4.0)) / 4.0
            float logMel = (std::log(1e-5f + melEnergy) - (-4.0f)) / 4.0f;
            // Layout: [nMels_, numFrames] => row m, col t
            melOutput[m * numFrames + t] = logMel;
        }
    }

    return melOutput;
}
