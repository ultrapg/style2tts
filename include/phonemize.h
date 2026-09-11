#ifndef PHONEMIZE_H
#define PHONEMIZE_H

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>

class PhonemizerEngine {
public:
    PhonemizerEngine();
    ~PhonemizerEngine();

    // Initialize eSpeak-NG with voice name (e.g. "en-us") and local data path
    void Init(const std::string& voice, const std::string& espeakDataDir);

    // Converts input text to StyleTTS2 token ID sequence
    std::vector<int64_t> text_to_sequence(const std::string& text);

    // Phonemizes text to IPA string using eSpeak-NG
    std::string phonemize(const std::string& text);

private:
    std::unordered_map<char32_t, int> symbol_to_id_;
    bool initialized_ = false;

    void init_symbol_map();
};

#endif // PHONEMIZE_H
