use std::fs;
use std::io::{self, IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use clap::Parser;

use style2tts::audio::{self, generate_silence, write_wav_24k};
use style2tts::cache::StyleCache;
use style2tts::engine::StyleTtsEngine;
use style2tts::setup;
use style2tts::text::split_text_into_chunks;

#[derive(Parser, Debug)]
#[command(
    name = "style2tts",
    version = "0.1.0",
    about = "Fast, autonomous Text-to-Speech CLI with Voice Cloning (Rust & C++)",
    after_help = "Quick Examples:\n  style2tts \"Hello world!\"\n  style2tts \"How are you doing today? Look at that sunrise!\"\n  style2tts \"Hello world!\" -r voice.mp3 -o out.wav\n  style2tts document.txt -v my_voice -s 1.1\n  echo \"Piped text stream\" | style2tts"
)]
struct Args {
    /// Text to speak or path to a text file (positional argument)
    #[arg(value_name = "TEXT_OR_FILE")]
    input: Option<String>,

    /// Direct text input (alternative to positional argument)
    #[arg(short = 't', long = "text")]
    text: Option<String>,

    /// Read text from file (alternative to positional argument)
    #[arg(short = 'f', long = "file")]
    file: Option<PathBuf>,

    /// Reference audio file (.wav or .mp3) for voice cloning (auto-crops if >10s)
    #[arg(short = 'r', long = "reference", value_name = "AUDIO_FILE")]
    reference: Option<PathBuf>,

    /// Output WAV file path (default: <bin_dir>/output.wav)
    #[arg(short = 'o', long = "output", value_name = "WAV_FILE")]
    output: Option<PathBuf>,

    /// Voice preset name from cache
    #[arg(short = 'v', long = "voice", value_name = "NAME")]
    voice: Option<String>,

    /// Speech speed multiplier (0.5 to 2.0, default: 1.0)
    #[arg(short = 's', long = "speed", default_value_t = 1.0)]
    speed: f32,

    /// Synthesis compute steps (default: 5 = balanced speed & quality, 1-3 = fast, 10 = max quality)
    #[arg(long = "steps")]
    steps: Option<i32>,

    /// Base sentence pause duration in milliseconds (default: 300)
    #[arg(long = "pause", default_value_t = 300)]
    pause: u32,

    /// Directory containing ONNX model weights (default: <bin_dir>/models)
    #[arg(long = "models-dir", value_name = "DIR")]
    models_dir: Option<PathBuf>,

    /// Directory containing eSpeak-NG phonetic data (default: <bin_dir>/espeak-ng-data)
    #[arg(long = "espeak-dir", value_name = "DIR")]
    espeak_dir: Option<PathBuf>,

    /// Local cache directory (default: <bin_dir>/cache)
    #[arg(long = "cache-dir", value_name = "DIR")]
    cache_dir: Option<PathBuf>,

    /// List all cached voice styles
    #[arg(long = "list-voices")]
    list_voices: bool,

    /// Clear all cached voice styles
    #[arg(long = "clear-cache")]
    clear_cache: bool,

    /// Force verification and automatic download of missing assets
    #[arg(long = "setup")]
    setup: bool,

    /// Force CPU execution provider
    #[arg(long = "cpu")]
    cpu: bool,

    /// Enable GPU (CUDA) execution provider if available
    #[arg(long = "gpu")]
    gpu: bool,

    /// Number of CPU threads for inference (0 = auto)
    #[arg(long = "threads", default_value_t = 0)]
    threads: usize,
}

fn main() {
    let args = Args::parse();

    // 1. Determine base binary directory
    let exe_path = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("."));
    let bin_dir = exe_path.parent().unwrap_or_else(|| Path::new(".")).to_path_buf();

    // 2. Strict local filesystem policy: configure directories relative to binary
    let models_dir = args.models_dir.unwrap_or_else(|| bin_dir.join("models"));
    let espeak_dir = args.espeak_dir.unwrap_or_else(|| bin_dir.join("espeak-ng-data"));
    let cache_dir = args.cache_dir.unwrap_or_else(|| bin_dir.join("cache"));
    let output_path = args.output.unwrap_or_else(|| bin_dir.join("output.wav"));

    // Ensure cache directory exists and redirect all temp env vars to local cache
    let _ = fs::create_dir_all(&cache_dir);
    let cache_dir_str = cache_dir.to_string_lossy().to_string();
    std::env::set_var("TMPDIR", &cache_dir_str);
    std::env::set_var("TEMP", &cache_dir_str);
    std::env::set_var("TMP", &cache_dir_str);

    // 3. Initialize Cache
    let mut style_cache = StyleCache::new(cache_dir.clone());

    if args.list_voices {
        let voices = style_cache.list_cached_voices();
        println!("Cached voice styles (in {}):", cache_dir.join("styles").display());
        if voices.is_empty() {
            println!("  (No cached voices found)");
        } else {
            for v in voices {
                println!("  - {}", v);
            }
        }
        return;
    }

    if args.clear_cache {
        match style_cache.clear_cache() {
            Ok(count) => println!("Cleared {} cached voice styles.", count),
            Err(e) => eprintln!("Error clearing cache: {}", e),
        }
        return;
    }

    if args.setup {
        if let Err(e) = setup::ensure_assets(&bin_dir, &models_dir, &espeak_dir, &cache_dir) {
            eprintln!("Error during setup: {}", e);
            std::process::exit(1);
        }
        println!("StyleTTS2 assets are verified and ready.");
        return;
    }

    // 4. Retrieve input text (positional arg, -t, -f, or piped stdin)
    let text_content = if let Some(text) = args.text {
        text
    } else if let Some(file_path) = args.file {
        match fs::read_to_string(&file_path) {
            Ok(content) => content,
            Err(e) => {
                eprintln!("Error reading input file {}: {}", file_path.display(), e);
                std::process::exit(1);
            }
        }
    } else if let Some(pos) = args.input {
        let path = Path::new(&pos);
        if path.is_file() {
            match fs::read_to_string(path) {
                Ok(content) => content,
                Err(e) => {
                    eprintln!("Error reading input file {}: {}", path.display(), e);
                    std::process::exit(1);
                }
            }
        } else {
            pos
        }
    } else if !io::stdin().is_terminal() {
        let mut buffer = String::new();
        match io::stdin().read_to_string(&mut buffer) {
            Ok(_) => buffer,
            Err(e) => {
                eprintln!("Error reading from stdin: {}", e);
                std::process::exit(1);
            }
        }
    } else {
        eprintln!("Error: No text provided.");
        eprintln!("\nQuick usage:");
        eprintln!("  style2tts \"Your text to speak here\"");
        eprintln!("  style2tts \"How are you doing today? Look at that sunrise!\"");
        eprintln!("  style2tts document.txt");
        eprintln!("  style2tts \"Hello!\" -r voice.mp3 -o output.wav");
        eprintln!("\nRun with --help for all options.");
        std::process::exit(1);
    };

    if text_content.trim().is_empty() {
        eprintln!("Error: Input text is empty.");
        std::process::exit(1);
    }

    // 5. Verify & Auto-setup local assets (runs automatically on first run in empty folder)
    if let Err(e) = setup::ensure_assets(&bin_dir, &models_dir, &espeak_dir, &cache_dir) {
        eprintln!("Error initializing local assets: {}", e);
        std::process::exit(1);
    }

    let (effective_steps, steps_desc) = match args.steps {
        None => (5, "5 (Balanced Speed & Quality - Default)".to_string()),
        Some(s) if s <= 3 => (s, format!("{} (Fast / Low-Latency)", s)),
        Some(s) if s >= 10 => (s, format!("{} (High Quality)", s)),
        Some(s) => (s, format!("{}", s)),
    };

    println!("========================================================");
    println!(" StyleTTS2 Autonomous CLI (Rust & C++)");
    println!("========================================================");
    println!(" Models directory:     {}", models_dir.display());
    println!(" eSpeak data:          {}", espeak_dir.display());
    println!(" Local cache:          {}", cache_dir.display());
    println!(" Output target:        {}", output_path.display());
    println!(" Execution device:     {}", if args.gpu { "GPU (CUDA)" } else { "CPU" });
    println!(" Speech speed:         {:.2}x", args.speed);
    println!(" Synthesis steps:      {}", steps_desc);
    println!(" Sentence pause:       {} ms", args.pause);
    println!("--------------------------------------------------------");

    // 6. Initialize C++ Engine
    println!("Initializing TTS engine...");
    let init_start = Instant::now();
    let use_cuda = args.gpu && !args.cpu;
    let engine = match StyleTtsEngine::new(&models_dir, &espeak_dir, use_cuda, args.threads) {
        Ok(eng) => eng,
        Err(e) => {
            eprintln!("Engine initialization failed: {}", e);
            std::process::exit(1);
        }
    };
    let init_duration = init_start.elapsed();
    println!("Engine initialized in {:.2} seconds.", init_duration.as_secs_f64());

    // 7. Resolve Base Style & Predictor Vectors
    let (base_style, base_pred) = if let Some(ref_path) = &args.reference {
        println!("Analyzing reference audio for voice cloning: {}", ref_path.display());
        match audio::read_audio_mono_24k(ref_path) {
            Ok(ref_pcm) => {
                let (active_pcm, was_cropped, orig_dur) =
                    audio::autocrop_reference_audio(&ref_pcm, 24000, 10.0);
                if was_cropped {
                    println!(
                        " [Voice Cloning] Audio duration was {:.1}s. Auto-cropped to optimal 10.0s speech window.",
                        orig_dur
                    );
                }
                match style_cache.get_or_extract(&engine, ref_path, &active_pcm) {
                    Ok(vectors) => vectors,
                    Err(e) => {
                        eprintln!("Failed to extract voice style: {}", e);
                        std::process::exit(1);
                    }
                }
            }
            Err(e) => {
                eprintln!("Error reading reference audio: {}", e);
                std::process::exit(1);
            }
        }
    } else if let Some(voice_name) = &args.voice {
        if let Some(vectors) = style_cache.load_named_voice(voice_name) {
            println!("Using voice preset: '{}'", voice_name);
            vectors
        } else {
            eprintln!("Error: Voice '{}' not found in cache.", voice_name);
            eprintln!("Run with --list-voices to inspect available presets.");
            std::process::exit(1);
        }
    } else {
        println!("Using default StyleTTS2 voice embedding.");
        match engine.get_default_embeddings() {
            Ok(embeddings) => embeddings,
            Err(e) => {
                eprintln!("Failed to retrieve default embeddings from engine: {}", e);
                std::process::exit(1);
            }
        }
    };

    // 8. Sentence Chunking & Pure Native Synthesis
    let chunks = split_text_into_chunks(&text_content, args.pause);
    println!("Synthesizing speech ({} sentence chunks)...", chunks.len());

    let synth_start = Instant::now();
    let mut all_samples: Vec<i16> = Vec::new();

    for (idx, chunk) in chunks.iter().enumerate() {
        print!(" [{}/{}] \"{}\" ... ", idx + 1, chunks.len(), chunk.text);
        io::stdout().flush().unwrap();

        let chunk_start = Instant::now();
        let chunk_samples = match engine.synthesize(
            &chunk.text,
            Some(&base_style),
            Some(&base_pred),
            args.speed,
            effective_steps,
        ) {
            Ok(samples) => samples,
            Err(e) => {
                println!("FAILED!");
                eprintln!("Error synthesizing chunk: {}", e);
                std::process::exit(1);
            }
        };

        let chunk_duration = chunk_start.elapsed();
        let chunk_audio_secs = (chunk_samples.len() as f64) / 24000.0;
        println!("{:.2}s audio (took {:.2}s)", chunk_audio_secs, chunk_duration.as_secs_f64());

        all_samples.extend_from_slice(&chunk_samples);

        // Add pause silence between sentences
        if idx + 1 < chunks.len() && chunk.pause_after_ms > 0 {
            let silence = generate_silence(chunk.pause_after_ms, 24000);
            all_samples.extend_from_slice(&silence);
        }
    }

    let synth_duration = synth_start.elapsed();
    let total_audio_secs = (all_samples.len() as f64) / 24000.0;
    let rtf = synth_duration.as_secs_f64() / total_audio_secs.max(0.001);

    // 9. Write Output WAV file
    println!("Writing WAV file to {}...", output_path.display());
    if let Err(e) = write_wav_24k(&output_path, &all_samples) {
        eprintln!("Failed to save output WAV file: {}", e);
        std::process::exit(1);
    }

    println!("\n=== Synthesis Results ===");
    println!(" Initialization time:  {:.2} seconds", init_duration.as_secs_f64());
    println!(" Total inference time: {:.2} seconds", synth_duration.as_secs_f64());
    println!(" Generated audio:      {:.2} seconds ({} samples)", total_audio_secs, all_samples.len());
    println!(" Real-Time Factor:     {:.2}x ({})", rtf, if rtf < 1.0 { "Faster than real-time!" } else { "Slower than real-time" });
    println!(" Output file saved:    {}", output_path.display());
    println!("========================================================");
}
