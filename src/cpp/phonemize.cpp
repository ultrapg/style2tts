#include "phonemize.h"
#include "espeak-ng/speak_lib.h"

#include <stdexcept>
#include <iostream>
#include <algorithm>

// Simple UTF-8 to UTF-32 decoder
static std::vector<char32_t> utf8_to_utf32(const std::string& str) {
    std::vector<char32_t> out;
    size_t i = 0;
    while (i < str.size()) {
        unsigned char c = static_cast<unsigned char>(str[i]);
        char32_t cp = 0;
        int len = 0;

        if (c <= 0x7F) {
            cp = c;
            len = 1;
        } else if ((c & 0xE0) == 0xC0) {
            cp = c & 0x1F;
            len = 2;
        } else if ((c & 0xF0) == 0xE0) {
            cp = c & 0x0F;
            len = 3;
        } else if ((c & 0xF8) == 0xF0) {
            cp = c & 0x07;
            len = 4;
        } else {
            // invalid utf-8 byte, skip
            i++;
            continue;
        }

        if (i + len > str.size()) {
            break;
        }

        for (int k = 1; k < len; k++) {
            unsigned char next_c = static_cast<unsigned char>(str[i + k]);
            if ((next_c & 0xC0) != 0x80) {
                // Invalid sequence
                cp = 0;
                break;
            }
            cp = (cp << 6) | (next_c & 0x3F);
        }

        if (cp != 0) {
            out.push_back(cp);
        }
        i += len;
    }
    return out;
}

PhonemizerEngine::PhonemizerEngine() {
    init_symbol_map();
}

PhonemizerEngine::~PhonemizerEngine() {
    // Note: espeak_Terminate() can be called if initialized
}

void PhonemizerEngine::init_symbol_map() {
    // StyleTTS2 / LibriTTS mapping
    symbol_to_id_ = {
        {U'_', 0}, {U';', 1}, {U':', 2}, {U',', 3}, {U'.', 4}, {U'!', 5}, {U'?', 6}, {U'¡', 7}, {U'¿', 8}, {U'—', 9}, {U'…', 10},
        {U'"', 11}, {U'«', 12}, {U'»', 13}, {U'“', 14}, {U'”', 15}, {U' ', 16}, {U'A', 17}, {U'B', 18}, {U'C', 19}, {U'D', 20},
        {U'E', 21}, {U'F', 22}, {U'G', 23}, {U'H', 24}, {U'I', 25}, {U'J', 26}, {U'K', 27}, {U'L', 28}, {U'M', 29}, {U'N', 30},
        {U'O', 31}, {U'P', 32}, {U'Q', 33}, {U'R', 34}, {U'S', 35}, {U'T', 36}, {U'U', 37}, {U'V', 38}, {U'W', 39}, {U'X', 40},
        {U'Y', 41}, {U'Z', 42}, {U'a', 43}, {U'b', 44}, {U'c', 45}, {U'd', 46}, {U'e', 47}, {U'f', 48}, {U'g', 49}, {U'h', 50},
        {U'i', 51}, {U'j', 52}, {U'k', 53}, {U'l', 54}, {U'm', 55}, {U'n', 56}, {U'o', 57}, {U'p', 58}, {U'q', 59}, {U'r', 60},
        {U's', 61}, {U't', 62}, {U'u', 63}, {U'v', 64}, {U'w', 65}, {U'x', 66}, {U'y', 67}, {U'z', 68}, {U'ɑ', 69}, {U'ɐ', 70},
        {U'ɒ', 71}, {U'æ', 72}, {U'ɓ', 73}, {U'ʙ', 74}, {U'β', 75}, {U'ɔ', 76}, {U'ɕ', 77}, {U'ç', 78}, {U'ɗ', 79}, {U'ɖ', 80},
        {U'ð', 81}, {U'ʤ', 82}, {U'ə', 83}, {U'ɘ', 84}, {U'ɚ', 85}, {U'ɛ', 86}, {U'ɜ', 87}, {U'ɝ', 88}, {U'ɞ', 89}, {U'ɟ', 90},
        {U'ʄ', 91}, {U'ɡ', 92}, {U'ɠ', 93}, {U'ɢ', 94}, {U'ʛ', 95}, {U'ɦ', 96}, {U'ɧ', 97}, {U'ħ', 98}, {U'ɥ', 99}, {U'ʜ', 100},
        {U'ɨ', 101}, {U'ɪ', 102}, {U'ʝ', 103}, {U'ɭ', 104}, {U'ɬ', 105}, {U'ɫ', 106}, {U'ɮ', 107}, {U'ʟ', 108}, {U'ɱ', 109},
        {U'ɯ', 110}, {U'ɰ', 111}, {U'ŋ', 112}, {U'ɳ', 113}, {U'ɲ', 114}, {U'ɴ', 115}, {U'ø', 116}, {U'ɵ', 117}, {U'ɸ', 118},
        {U'θ', 119}, {U'œ', 120}, {U'ɶ', 121}, {U'ʘ', 122}, {U'ɹ', 123}, {U'ɺ', 124}, {U'ɾ', 125}, {U'ɻ', 126}, {U'ʀ', 127},
        {U'ʁ', 128}, {U'ɽ', 129}, {U'ʂ', 130}, {U'ʃ', 131}, {U'ʈ', 132}, {U'ʧ', 133}, {U'ʉ', 134}, {U'ʊ', 135}, {U'ʋ', 136},
        {U'ⱱ', 137}, {U'ʌ', 138}, {U'ɣ', 139}, {U'ɤ', 140}, {U'ʍ', 141}, {U'χ', 142}, {U'ʎ', 143}, {U'ʏ', 144}, {U'ʑ', 145},
        {U'ʐ', 146}, {U'ʒ', 147}, {U'ʔ', 148}, {U'ʡ', 149}, {U'ʕ', 150}, {U'ʢ', 151}, {U'ǀ', 152}, {U'ǁ', 153}, {U'ǂ', 154},
        {U'ǃ', 155}, {U'ˈ', 156}, {U'ˌ', 157}, {U'ː', 158}, {U'ˑ', 159}, {U'ʼ', 160}, {U'ʴ', 161}, {U'ʰ', 162}, {U'ʱ', 163},
        {U'ʲ', 164}, {U'ʷ', 165}, {U'ˠ', 166}, {U'ˤ', 167}, {U'˞', 168}, {U'↓', 169}, {U'↑', 170}, {U'→', 171}, {U'↗', 172},
        {U'↘', 173}, {U'̩', 175}, {U'ᵻ', 177}
    };
}

void PhonemizerEngine::Init(const std::string& voice, const std::string& espeakDataDir) {
    // AUDIO_OUTPUT_SYNCHRONOUS avoids initializing sound devices/pulseaudio
    int result = espeak_Initialize(AUDIO_OUTPUT_SYNCHRONOUS, 0, espeakDataDir.c_str(), 0);
    if (result < 0) {
        throw std::runtime_error("Failed to initialize eSpeak-NG with data path: " + espeakDataDir);
    }
    int setVoiceResult = espeak_SetVoiceByName(voice.c_str());
    if (setVoiceResult != 0) {
        throw std::runtime_error("Failed to set eSpeak-NG voice: " + voice);
    }
    initialized_ = true;
}

std::string PhonemizerEngine::phonemize(const std::string& text) {
    if (!initialized_) {
        throw std::runtime_error("PhonemizerEngine not initialized");
    }

    std::string textCopy = text;
    const char* inputTextPointer = textCopy.c_str();
    std::string res = "";

    while (inputTextPointer != nullptr && *inputTextPointer != '\0') {
        const char* clausePhonemes = espeak_TextToPhonemes(
            (const void**)&inputTextPointer,
            /*textmode*/ espeakCHARS_AUTO,
            /*phonememode = IPA*/ 0x02
        );
        if (clausePhonemes != nullptr) {
            res += clausePhonemes;
            res += ", ";
        }
    }

    if (res.size() >= 2) {
        res.pop_back();
        res.pop_back();
    }
    res += ".";
    return res;
}

std::vector<int64_t> PhonemizerEngine::text_to_sequence(const std::string& text) {
    std::string cleanText = phonemize(text);
    std::vector<char32_t> utf32Chars = utf8_to_utf32(cleanText);

    std::vector<int64_t> sequence;
    for (char32_t symbol : utf32Chars) {
        auto it = symbol_to_id_.find(symbol);
        if (it != symbol_to_id_.end()) {
            sequence.push_back(it->second);
        }
    }
    return sequence;
}
