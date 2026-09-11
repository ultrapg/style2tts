// Minimal stubs for unused eSpeak-NG audio output libraries (pcaudio & sonic)
// StyleTTS2 uses its own ONNX vocoder for all audio generation and only uses eSpeak-NG for Text-to-Phoneme translation.
// These stubs allow libespeak-ng.a to be 100% statically linked without pulling in libpulse, libasound, libpcaudio, or libsonic.

extern "C" {

void* create_audio_device_object(const char*, const char*) { return nullptr; }
int audio_object_open(void*, void*) { return -1; }
void audio_object_close(void*) {}
void audio_object_destroy(void*) {}
int audio_object_write(void*, const void*, int) { return 0; }
const char* audio_object_strerror(int) { return "No audio device (stub)"; }
int audio_object_drain(void*) { return 0; }
int audio_object_flush(void*) { return 0; }

void* sonicCreateStream(int, int) { return nullptr; }
void sonicDestroyStream(void*) {}
int sonicFlushStream(void*) { return 0; }
int sonicReadShortFromStream(void*, short*, int) { return 0; }
int sonicWriteShortToStream(void*, short*, int) { return 0; }
float sonicGetSpeed(void*) { return 1.0f; }
void sonicSetSpeed(void*, float) {}

}
