use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct ModelCheckResult {
    pub all_present: bool,
    pub missing_files: Vec<String>,
}

pub fn check_models(models_dir: &Path) -> ModelCheckResult {
    let required_files = [
        "bert_encoder.onnx",
        "plbert_simp.onnx",
        "style_encoder_simp.onnx",
        "predictor_encoder_simp.onnx",
        "final_simp.onnx",
        "ref_s.bin",
        "ref_p.bin",
    ];

    let mut missing = Vec::new();
    for &file in &required_files {
        let path = models_dir.join(file);
        if !path.exists() || fs::metadata(&path).map(|m| m.len() == 0).unwrap_or(true) {
            missing.push(file.to_string());
        }
    }

    ModelCheckResult {
        all_present: missing.is_empty(),
        missing_files: missing,
    }
}

pub fn check_espeak_data(espeak_dir: &Path) -> bool {
    if !espeak_dir.exists() || !espeak_dir.is_dir() {
        return false;
    }

    if let Ok(entries) = fs::read_dir(espeak_dir) {
        entries.count() > 5
    } else {
        false
    }
}

pub fn check_onnxruntime(bin_dir: &Path) -> bool {
    let candidate_paths = [
        bin_dir.join("lib/libonnxruntime.so"),
        bin_dir.join("libonnxruntime.so"),
        PathBuf::from("/usr/lib/x86_64-linux-gnu/libonnxruntime.so.1.23"),
        PathBuf::from("/usr/lib/x86_64-linux-gnu/libonnxruntime.so"),
        PathBuf::from("/usr/lib/libonnxruntime.so"),
        PathBuf::from("/usr/local/lib/libonnxruntime.so"),
    ];

    candidate_paths.iter().any(|p| p.exists())
}

pub fn setup_onnxruntime(bin_dir: &Path, cache_dir: &Path) -> Result<(), String> {
    println!(" [AutoSetup] ONNX Runtime library not found on host system.");
    println!(" [AutoSetup] Downloading official standalone ONNX Runtime package (~7.5 MB)...");

    let lib_dir = bin_dir.join("lib");
    let _ = fs::create_dir_all(&lib_dir);

    let tar_url = "https://github.com/microsoft/onnxruntime/releases/download/v1.18.0/onnxruntime-linux-x64-1.18.0.tgz";
    let tar_file = cache_dir.join("onnxruntime.tgz");

    download_file_parallel(tar_url, &tar_file, 4)?;

    println!(" [AutoSetup] Extracting libonnxruntime.so to {}...", lib_dir.display());

    let tar_status = Command::new("tar")
        .args([
            "-xzf",
            tar_file.to_str().unwrap(),
            "-C",
            lib_dir.to_str().unwrap(),
            "--wildcards",
            "*/lib/libonnxruntime.so*",
            "--strip-components=2",
        ])
        .status();

    let _ = fs::remove_file(&tar_file);

    if let Ok(st) = tar_status {
        if st.success() {
            let unversioned = lib_dir.join("libonnxruntime.so");
            if !unversioned.exists() {
                let versioned = lib_dir.join("libonnxruntime.so.1.18.0");
                if versioned.exists() {
                    #[cfg(unix)]
                    let _ = std::os::unix::fs::symlink(&versioned, &unversioned);
                }
            }

            if unversioned.exists() || lib_dir.join("libonnxruntime.so.1.18.0").exists() {
                println!(" [AutoSetup] ONNX Runtime library successfully deployed!");
                return Ok(());
            }
        }
    }

    Err("Failed to extract ONNX Runtime library.".to_string())
}

pub fn ensure_assets(
    bin_dir: &Path,
    models_dir: &Path,
    espeak_dir: &Path,
    cache_dir: &Path,
) -> Result<(), String> {
    let model_check = check_models(models_dir);
    let espeak_present = check_espeak_data(espeak_dir);
    let ort_present = check_onnxruntime(bin_dir);

    if model_check.all_present && espeak_present && ort_present {
        return Ok(());
    }

    println!("========================================================");
    println!(" StyleTTS2 First-Run Auto-Setup");
    println!("========================================================");
    println!(" Empty folder or missing assets detected!");
    println!(" Automatically initializing local assets in program directory...");
    println!(" -> Program Dir: {}", bin_dir.display());
    println!(" -> Models:      {}", models_dir.display());
    println!(" -> eSpeak Data: {}", espeak_dir.display());
    println!("--------------------------------------------------------");

    // 1. Setup standalone ONNX Runtime if missing
    if !ort_present {
        setup_onnxruntime(bin_dir, cache_dir)?;
    }

    // 2. Setup eSpeak-NG data
    if !espeak_present {
        setup_espeak_data(espeak_dir, cache_dir)?;
    }

    // 3. Setup Models
    if !model_check.all_present {
        setup_models(models_dir, &model_check.missing_files, cache_dir)?;
    }

    println!("--------------------------------------------------------");
    println!(" [AutoSetup] All assets and libraries successfully initialized!");
    println!("========================================================\n");

    Ok(())
}

fn setup_espeak_data(espeak_dir: &Path, cache_dir: &Path) -> Result<(), String> {
    println!(" [AutoSetup] Setting up eSpeak-NG phonetic dictionary data...");

    // Check system directories first for fast local copy
    let system_candidates = [
        Path::new("/usr/lib/x86_64-linux-gnu/espeak-ng-data"),
        Path::new("/usr/share/espeak-ng-data"),
        Path::new("/usr/lib/espeak-ng-data"),
        Path::new("/usr/local/share/espeak-ng-data"),
    ];

    for &sys_path in &system_candidates {
        if sys_path.exists() && check_espeak_data(sys_path) {
            println!(" [AutoSetup] Found system eSpeak data at {}, copying locally...", sys_path.display());
            copy_dir_all(sys_path, espeak_dir)
                .map_err(|e| format!("Failed to copy espeak data: {}", e))?;
            if check_espeak_data(espeak_dir) {
                println!(" [AutoSetup] eSpeak-NG data successfully copied.");
                return Ok(());
            }
        }
    }

    // Fallback: download pre-packaged archive
    println!(" [AutoSetup] Downloading prebuilt espeak-ng-data package...");
    let tar_url = "https://github.com/thewh1teagle/espeakng-loader/releases/download/v1.51/espeak-ng-data.tar.gz";
    let tar_file = cache_dir.join("espeak-ng-data.tar.gz");

    let _ = fs::create_dir_all(cache_dir);
    let _ = fs::create_dir_all(espeak_dir);

    let curl_status = Command::new("curl")
        .args(["-L", "--fail", "-o", tar_file.to_str().unwrap(), tar_url])
        .status();

    if let Ok(status) = curl_status {
        if status.success() {
            let tar_status = Command::new("tar")
                .args(["-xzf", tar_file.to_str().unwrap(), "-C", espeak_dir.to_str().unwrap(), "--strip-components=1"])
                .status();

            let _ = fs::remove_file(&tar_file);

            if let Ok(ts) = tar_status {
                if ts.success() && check_espeak_data(espeak_dir) {
                    println!(" [AutoSetup] Extracted espeak-ng-data successfully.");
                    return Ok(());
                }
            }
        }
    }

    Err("Failed to set up eSpeak-NG data.".to_string())
}

fn setup_models(models_dir: &Path, missing_files: &[String], _cache_dir: &Path) -> Result<(), String> {
    fs::create_dir_all(models_dir)
        .map_err(|e| format!("Failed to create models directory {}: {}", models_dir.display(), e))?;

    let base_url = "https://huggingface.co/DDATT/StyleTTS2-ONNX-Cpp/resolve/main";

    for file in missing_files {
        let target = models_dir.join(file);
        let url = format!("{}/{}", base_url, file);

        println!(" [AutoSetup] Downloading {}...", file);
        download_file_parallel(&url, &target, 8)?;
    }

    Ok(())
}

fn download_file_parallel(url: &str, target: &Path, num_workers: usize) -> Result<(), String> {
    if let Some(parent) = target.parent() {
        let _ = fs::create_dir_all(parent);
    }

    // 1. Query Content-Length via HTTP HEAD request using curl
    let head_output = Command::new("curl")
        .args(["-sI", "-L", url])
        .output();

    let mut content_length: Option<u64> = None;
    if let Ok(output) = head_output {
        let header_str = String::from_utf8_lossy(&output.stdout);
        for line in header_str.lines() {
            let lower = line.to_lowercase();
            if lower.starts_with("content-length:") {
                if let Some(val_str) = line.split(':').nth(1) {
                    if let Ok(val) = val_str.trim().parse::<u64>() {
                        content_length = Some(val);
                    }
                }
            }
        }
    }

    // 2. For large files (> 20 MB), execute parallel multi-threaded chunk downloading in pure Rust
    if let Some(total) = content_length {
        if total > 20_000_000 {
            let workers = num_workers.clamp(2, 16);
            let chunk_size = total / (workers as u64);
            let temp_dir = target.with_extension("dl_parts");
            let _ = fs::create_dir_all(&temp_dir);

            println!(
                " [AutoSetup] Parallel downloading ({:.1} MB across {} Rust worker threads)...",
                total as f64 / (1024.0 * 1024.0),
                workers
            );

            let mut handles = Vec::new();
            for i in 0..workers {
                let start = (i as u64) * chunk_size;
                let end = if i == workers - 1 {
                    total - 1
                } else {
                    ((i + 1) as u64) * chunk_size - 1
                };

                let part_file = temp_dir.join(format!("part_{}", i));
                let part_file_clone = part_file.clone();
                let url_clone = url.to_string();

                let handle = std::thread::spawn(move || -> Result<(), String> {
                    let range_arg = format!("{}-{}", start, end);
                    let status = Command::new("curl")
                        .args([
                            "-s",
                            "-L",
                            "--fail",
                            "-r",
                            &range_arg,
                            "-o",
                            part_file_clone.to_str().unwrap(),
                            &url_clone,
                        ])
                        .status()
                        .map_err(|e| format!("curl error: {}", e))?;

                    if !status.success() {
                        return Err(format!("Chunk {}-{} failed", start, end));
                    }
                    Ok(())
                });

                handles.push((handle, part_file));
            }

            let mut failed = false;
            let mut part_paths = Vec::new();
            for (handle, part_file) in handles {
                part_paths.push(part_file);
                match handle.join() {
                    Ok(Ok(())) => {}
                    _ => {
                        failed = true;
                    }
                }
            }

            if !failed {
                // Stitch all chunk parts together in exact byte order
                let assemble_result = (|| -> io::Result<()> {
                    let mut out = fs::File::create(target)?;
                    for part_path in &part_paths {
                        let mut part = fs::File::open(part_path)?;
                        io::copy(&mut part, &mut out)?;
                    }
                    Ok(())
                })();

                let _ = fs::remove_dir_all(&temp_dir);

                if assemble_result.is_ok() {
                    if let Ok(meta) = fs::metadata(target) {
                        if meta.len() == total {
                            println!(" [AutoSetup] Successfully downloaded and verified {}.", target.file_name().unwrap().to_str().unwrap());
                            return Ok(());
                        }
                    }
                }
            } else {
                let _ = fs::remove_dir_all(&temp_dir);
            }

            println!(" [AutoSetup] Parallel download incomplete, falling back to standard single-stream download...");
        }
    }

    // 3. Direct single-stream fallback
    let status = Command::new("curl")
        .args(["-L", "--fail", "-o", target.to_str().unwrap(), url])
        .status()
        .map_err(|e| format!("Failed to execute curl: {}", e))?;

    if !status.success() {
        return Err(format!("Download failed for {}", url));
    }

    Ok(())
}

fn copy_dir_all(src: &Path, dst: &Path) -> io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let ty = entry.file_type()?;
        let from = entry.path();
        let to = dst.join(entry.file_name());
        if ty.is_dir() {
            copy_dir_all(&from, &to)?;
        } else {
            fs::copy(&from, &to)?;
        }
    }
    Ok(())
}
