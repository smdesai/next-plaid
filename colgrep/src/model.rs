use anyhow::{Context, Result};
use hf_hub::api::sync::ApiBuilder;
use std::path::PathBuf;

pub const DEFAULT_MODEL: &str = "lightonai/LateOn-Code-edge";
pub const MXBAI_COREML_MODEL: &str = "smdesai/MXBAIEdgeColbert-b8";
pub const MXBAI_COREML_QUERY_MODEL: &str = "smdesai/MXBAIEdgeColbert";
const MXBAI_BASE_MODEL: &str = "mixedbread-ai/mxbai-edge-colbert-v0-32m";
const NATIVE_COREML_MARKER: &str = ".colgrep-native-coreml";
const MXBAI_COREML_FILES: &[&str] = &[
    "MXBAIEdgeColbert-b8.mlmodelc/coremldata.bin",
    "MXBAIEdgeColbert-b8.mlmodelc/metadata.json",
    "MXBAIEdgeColbert-b8.mlmodelc/model.mil",
    "MXBAIEdgeColbert-b8.mlmodelc/analytics/coremldata.bin",
    "MXBAIEdgeColbert-b8.mlmodelc/weights/weight.bin",
];
const MXBAI_COREML_QUERY_FILES: &[&str] = &[
    "MXBAIEdgeColbert.mlmodelc/coremldata.bin",
    "MXBAIEdgeColbert.mlmodelc/metadata.json",
    "MXBAIEdgeColbert.mlmodelc/model.mil",
    "MXBAIEdgeColbert.mlmodelc/analytics/coremldata.bin",
    "MXBAIEdgeColbert.mlmodelc/weights/weight.bin",
];

/// Files required for ColBERT model
const REQUIRED_FILES: &[&str] = &[
    "tokenizer.json",
    "config_sentence_transformers.json",
    "config.json",
    "onnx_config.json",
];

/// Model weight files. At least one must be present, but neither is required on
/// its own: repos legitimately ship FP32 only, INT8 only, or both. Requiring
/// `model_int8.onnx` outright made FP32-only repos undownloadable even on CUDA
/// builds, which never load the quantized weights.
const WEIGHT_FILES: &[&str] = &["model.onnx", "model_int8.onnx"];

/// Load model from cache or download from HuggingFace.
/// Returns path to the model directory.
/// The `quiet` parameter is kept for API compatibility but no longer used
/// (output is now handled in IndexBuilder::ensure_model_created after ONNX runtime init).
pub fn ensure_model(model_id: Option<&str>, _quiet: bool) -> Result<PathBuf> {
    let model_id = model_id.unwrap_or(DEFAULT_MODEL);

    // Check if it's a local path
    let local_path = PathBuf::from(model_id);
    if local_path.exists() && local_path.is_dir() {
        if local_path.join(NATIVE_COREML_MARKER).is_file() {
            ensure_macos_15_apple_silicon()?;
        }
        return Ok(local_path);
    }

    let api = huggingface_api()?;
    if model_id == MXBAI_COREML_MODEL || model_id == MXBAI_COREML_QUERY_MODEL {
        ensure_macos_15_apple_silicon()?;
        return ensure_mxbai_coreml_model(&api, model_id == MXBAI_COREML_MODEL);
    }

    // Download from HuggingFace

    // Build API with token from environment variables or token file
    // Priority: HF_TOKEN > HUGGING_FACE_HUB_TOKEN > token file ($HF_HOME/token or ~/.cache/huggingface/token)
    let repo = api.model(model_id.to_string());

    // Download all required files (cached if already present)
    let mut model_dir = None;
    for file in REQUIRED_FILES {
        match repo.get(file) {
            Ok(path) => {
                if model_dir.is_none() {
                    model_dir = path.parent().map(|p| p.to_path_buf());
                }
            }
            Err(e) => {
                // config.json may not exist in all models, that's ok
                if *file != "config.json" {
                    return Err(e.into());
                }
            }
        }
    }

    // Fetch whichever weight files the repo publishes. Missing ones are fine as
    // long as at least one variant lands.
    let mut available_weights = Vec::new();
    for file in WEIGHT_FILES {
        if let Ok(path) = repo.get(file) {
            if model_dir.is_none() {
                model_dir = path.parent().map(|p| p.to_path_buf());
            }
            available_weights.push(*file);
        }
    }

    if available_weights.is_empty() {
        anyhow::bail!(
            "Model '{}' publishes neither model.onnx nor model_int8.onnx. \
             It does not look like an ONNX export; run `pylate-onnx-export {}` first.",
            model_id,
            model_id
        );
    }

    model_dir.ok_or_else(|| anyhow::anyhow!("Failed to determine model directory"))
}

fn huggingface_api() -> Result<hf_hub::api::sync::Api> {
    // Build API with token from environment variables or token file
    // Priority: HF_TOKEN > HUGGING_FACE_HUB_TOKEN > token file ($HF_HOME/token or ~/.cache/huggingface/token)
    let mut builder = ApiBuilder::from_env();
    let token_from_env = std::env::var("HF_TOKEN")
        .or_else(|_| std::env::var("HUGGING_FACE_HUB_TOKEN"))
        .ok()
        .map(|t| t.trim_matches('"').trim_matches('\'').to_string());
    if token_from_env.is_some() {
        builder = builder.with_token(token_from_env);
    }
    Ok(builder.build()?)
}

fn ensure_mxbai_coreml_model(api: &hf_hub::api::sync::Api, include_b8: bool) -> Result<PathBuf> {
    let model_id = if include_b8 {
        MXBAI_COREML_MODEL
    } else {
        MXBAI_COREML_QUERY_MODEL
    };
    let model_files = if include_b8 {
        MXBAI_COREML_FILES
    } else {
        MXBAI_COREML_QUERY_FILES
    };
    let coreml_repo = api.model(model_id.to_string());
    let mut model_dir = None;
    for file in model_files {
        let path = coreml_repo.get(file)?;
        if model_dir.is_none() {
            model_dir = path
                .parent()
                .and_then(|path| path.parent())
                .map(PathBuf::from);
        }
    }
    let model_dir = model_dir.context("Failed to determine CoreML model directory")?;
    if include_b8 {
        // The B8 document encoder has a fixed [8, 256] contract. Queries use the
        // matching single-item export so interactive search does not pad to eight.
        let query_repo = api.model(MXBAI_COREML_QUERY_MODEL.to_string());
        for file in MXBAI_COREML_QUERY_FILES {
            let source = query_repo.get(file)?;
            let destination = model_dir.join(file);
            std::fs::create_dir_all(
                destination
                    .parent()
                    .context("CoreML bundle file has no parent directory")?,
            )?;
            std::fs::copy(&source, &destination).with_context(|| {
                format!("Failed to stage {file} beside the B8 CoreML model bundle")
            })?;
        }
    }
    let base_repo = api.model(MXBAI_BASE_MODEL.to_string());
    for file in ["tokenizer.json", "onnx_config.json"] {
        let source = base_repo.get(file)?;
        std::fs::copy(&source, model_dir.join(file))
            .with_context(|| format!("Failed to stage {file} beside the CoreML model bundle"))?;
    }
    std::fs::write(model_dir.join(NATIVE_COREML_MARKER), model_id)?;

    Ok(model_dir)
}

fn ensure_macos_15_apple_silicon() -> Result<()> {
    if !cfg!(all(target_os = "macos", target_arch = "aarch64")) {
        anyhow::bail!(
            "{} requires Apple Silicon and macOS 15 or later; use an ONNX ColBERT model on this platform",
            MXBAI_COREML_MODEL
        );
    }
    let output = std::process::Command::new("sw_vers")
        .arg("-productVersion")
        .output()
        .context("Failed to determine the macOS version")?;
    let major = String::from_utf8_lossy(&output.stdout)
        .trim()
        .split('.')
        .next()
        .and_then(|part| part.parse::<u32>().ok());
    if !output.status.success() || major.is_none_or(|major| major < 15) {
        anyhow::bail!("{} requires macOS 15 or later", MXBAI_COREML_MODEL);
    }
    Ok(())
}

/// Whether `model_path` is a complete, explicitly identified native CoreML bundle.
///
/// A local model path replaces its Hub ID at the CLI boundary, so the marker records
/// the supported model identity rather than relying on the caller's display string.
pub fn uses_native_coreml(_model_id: &str, model_path: &std::path::Path) -> bool {
    cfg!(all(target_os = "macos", target_arch = "aarch64"))
        && (_model_id == MXBAI_COREML_MODEL
            || _model_id == MXBAI_COREML_QUERY_MODEL
            || PathBuf::from(_model_id).is_dir())
        && std::fs::read_to_string(model_path.join(NATIVE_COREML_MARKER)).is_ok_and(|model| {
            match model.trim() {
                MXBAI_COREML_MODEL => {
                    MXBAI_COREML_FILES
                        .iter()
                        .all(|file| model_path.join(file).is_file())
                        && MXBAI_COREML_QUERY_FILES
                            .iter()
                            .all(|file| model_path.join(file).is_file())
                }
                MXBAI_COREML_QUERY_MODEL => MXBAI_COREML_QUERY_FILES
                    .iter()
                    .all(|file| model_path.join(file).is_file()),
                _ => false,
            }
        })
}

/// Resolve the precision that is actually loadable from `model_dir`.
///
/// `requested` follows the user's `--fp32`/`--int8` preference (or the per-build
/// default), but a repo may ship only one variant. Falling back keeps a
/// FP32-only or INT8-only model usable instead of failing at session load.
pub fn resolve_quantized(model_dir: &std::path::Path, requested: bool) -> bool {
    let has_int8 = model_dir.join("model_int8.onnx").exists();
    let has_fp32 = model_dir.join("model.onnx").exists();
    match (requested, has_int8, has_fp32) {
        // Wanted INT8, only FP32 shipped.
        (true, false, true) => false,
        // Wanted FP32, only INT8 shipped.
        (false, true, false) => true,
        _ => requested,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mxbai_coreml_is_selected_only_for_its_complete_bundle() {
        let temp = tempfile::tempdir().unwrap();
        let bundle = temp.path().join("MXBAIEdgeColbert-b8.mlmodelc");
        std::fs::create_dir(&bundle).unwrap();
        assert!(!uses_native_coreml(MXBAI_COREML_MODEL, temp.path()));
        std::fs::write(temp.path().join(NATIVE_COREML_MARKER), MXBAI_COREML_MODEL).unwrap();
        for file in MXBAI_COREML_FILES {
            let path = temp.path().join(file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, []).unwrap();
        }
        for file in MXBAI_COREML_QUERY_FILES {
            let path = temp.path().join(file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, []).unwrap();
        }
        assert_eq!(
            uses_native_coreml(MXBAI_COREML_MODEL, temp.path()),
            cfg!(all(target_os = "macos", target_arch = "aarch64"))
        );
        assert!(!uses_native_coreml(DEFAULT_MODEL, temp.path()));
        std::fs::write(temp.path().join(NATIVE_COREML_MARKER), DEFAULT_MODEL).unwrap();
        assert!(!uses_native_coreml(MXBAI_COREML_MODEL, temp.path()));
    }

    #[test]
    fn standalone_query_coreml_bundle_is_recognized() {
        let temp = tempfile::tempdir().unwrap();
        std::fs::write(
            temp.path().join(NATIVE_COREML_MARKER),
            MXBAI_COREML_QUERY_MODEL,
        )
        .unwrap();
        for file in MXBAI_COREML_QUERY_FILES {
            let path = temp.path().join(file);
            std::fs::create_dir_all(path.parent().unwrap()).unwrap();
            std::fs::write(path, []).unwrap();
        }
        assert_eq!(
            uses_native_coreml(MXBAI_COREML_QUERY_MODEL, temp.path()),
            cfg!(all(target_os = "macos", target_arch = "aarch64"))
        );
    }
}
