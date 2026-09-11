use std::path::Path;

fn main() {
    let project_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let project_path = Path::new(&project_dir);

    // Compile C++ core
    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .warnings(false)
        .include("include")
        .include("third_party/include")
        .include("third_party/include/espeak-ng")
        .include("third_party/include/onnxruntime")
        .file("src/cpp/phonemize.cpp")
        .file("src/cpp/mel_spectrogram.cpp")
        .file("src/cpp/styletts2_engine.cpp")
        .file("src/cpp/stubs.cpp")
        .file("src/cpp/ffi.cpp");

    build.compile("style2tts_core");

    // Link libraries: statically link eSpeak-NG, dynamically link libdl for runtime ONNX loading
    let third_party_lib = project_path.join("third_party/lib");
    println!("cargo:rustc-link-search=native={}", third_party_lib.display());
    println!("cargo:rustc-link-lib=static=espeak-ng");
    println!("cargo:rustc-link-lib=dl");

    // RPATH configuration for standalone portability (loads libonnxruntime.so from ./ or ./lib if placed locally)
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/lib");

    // Rerun triggers
    println!("cargo:rerun-if-changed=include/styletts2_c_api.h");
    println!("cargo:rerun-if-changed=include/phonemize.h");
    println!("cargo:rerun-if-changed=include/mel_spectrogram.h");
    println!("cargo:rerun-if-changed=include/styletts2_engine.h");
    println!("cargo:rerun-if-changed=src/cpp/phonemize.cpp");
    println!("cargo:rerun-if-changed=src/cpp/mel_spectrogram.cpp");
    println!("cargo:rerun-if-changed=src/cpp/styletts2_engine.cpp");
    println!("cargo:rerun-if-changed=src/cpp/ffi.cpp");
}
