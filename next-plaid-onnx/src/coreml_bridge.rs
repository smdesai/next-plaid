use anyhow::{Context, Result};
use std::ffi::{c_char, c_int, c_void, CString};
use std::path::{Path, PathBuf};

const NATIVE_COREML_MARKER: &str = ".colgrep-native-coreml";
const MXBAI_COREML_MODEL: &str = "smdesai/MXBAIEdgeColbert-b8";
const MXBAI_COREML_QUERY_MODEL: &str = "smdesai/MXBAIEdgeColbert";

pub(crate) enum CoreMlModels {
    QueryOnly(PathBuf),
    QueryAndB8 { query: PathBuf, document: PathBuf },
}

unsafe extern "C" {
    fn next_plaid_coreml_create(
        model_path: *const c_char,
        cpu_only: bool,
        error: *mut *mut c_char,
    ) -> *mut c_void;
    fn next_plaid_coreml_encode(
        session: *mut c_void,
        batch_size: usize,
        sequence_length: usize,
        is_query: bool,
        input_ids: *const i32,
        input_ids_len: usize,
        attention_mask: *const i32,
        attention_mask_len: usize,
        output: *mut f32,
        output_len: usize,
        embedding_dim: *mut usize,
        error: *mut *mut c_char,
    ) -> c_int;
    fn next_plaid_coreml_destroy(session: *mut c_void);
    fn next_plaid_coreml_free_error(error: *mut c_char);
}

pub(crate) struct CoreMlSession {
    query_path: PathBuf,
    document_path: Option<PathBuf>,
    force_cpu: bool,
    query_handle: Option<*mut c_void>,
    document_handle: Option<*mut c_void>,
}

// MLModel accepts prediction from arbitrary threads. Colbert serializes access
// through its session mutex, and ARC retains the native object until Drop.
unsafe impl Send for CoreMlSession {}

impl CoreMlSession {
    pub(crate) fn new(
        query_path: PathBuf,
        document_path: Option<PathBuf>,
        force_cpu: bool,
    ) -> Result<Self> {
        Ok(Self {
            query_path,
            document_path,
            force_cpu,
            query_handle: None,
            document_handle: None,
        })
    }

    fn handle(&mut self, is_query: bool) -> Result<*mut c_void> {
        let (path, handle, label) = if is_query {
            (&self.query_path, &mut self.query_handle, "query")
        } else {
            (
                self.document_path
                    .as_ref()
                    .context("This CoreML model does not include a B8 document encoder")?,
                &mut self.document_handle,
                "B8 document",
            )
        };
        if let Some(handle) = handle {
            return Ok(*handle);
        }

        let path = CString::new(path.to_string_lossy().as_bytes())?;
        let mut error = std::ptr::null_mut();
        let loaded = unsafe { next_plaid_coreml_create(path.as_ptr(), self.force_cpu, &mut error) };
        if loaded.is_null() {
            return Err(ffi_error(
                error,
                &format!("Failed to load the {label} CoreML model"),
            ));
        }
        *handle = Some(loaded);
        Ok(loaded)
    }

    pub(crate) fn document_batch_size(&self) -> usize {
        if self.document_path.is_some() {
            8
        } else {
            1
        }
    }

    pub(crate) fn run(
        &mut self,
        batch_size: usize,
        sequence_length: usize,
        input_ids: &[i64],
        attention_mask: &[i64],
        is_query: bool,
    ) -> Result<(usize, Vec<f32>)> {
        const EMBEDDING_DIM: usize = 64;
        const SEQUENCE_LENGTH: usize = 256;
        let expected_input_len = batch_size
            .checked_mul(sequence_length)
            .context("CoreML input dimensions overflow")?;
        if input_ids.len() != expected_input_len || attention_mask.len() != expected_input_len {
            anyhow::bail!("CoreML input dimensions do not match token and attention-mask buffers");
        }
        let use_query_model = is_query || self.document_path.is_none();
        if use_query_model && batch_size != 1 {
            anyhow::bail!("The native CoreML query encoder accepts one query at a time");
        }
        let input_ids: Vec<i32> = input_ids
            .iter()
            .map(|&value| i32::try_from(value).context("CoreML token IDs must fit int32"))
            .collect::<Result<_>>()?;
        let attention_mask: Vec<i32> = attention_mask
            .iter()
            .map(|&value| i32::try_from(value).context("CoreML attention masks must fit int32"))
            .collect::<Result<_>>()?;
        let output_len = batch_size
            .checked_mul(SEQUENCE_LENGTH)
            .and_then(|count| count.checked_mul(EMBEDDING_DIM))
            .context("CoreML output dimensions overflow")?;
        let mut output = vec![0.0; output_len];
        let mut embedding_dim = 0;
        let mut error = std::ptr::null_mut();
        let status = unsafe {
            next_plaid_coreml_encode(
                self.handle(use_query_model)?,
                batch_size,
                sequence_length,
                use_query_model,
                input_ids.as_ptr(),
                input_ids.len(),
                attention_mask.as_ptr(),
                attention_mask.len(),
                output.as_mut_ptr(),
                output.len(),
                &mut embedding_dim,
                &mut error,
            )
        };
        if status != 0 {
            return Err(ffi_error(error, "CoreML encoding failed"));
        }
        if embedding_dim != EMBEDDING_DIM {
            anyhow::bail!(
                "CoreML output has embedding dimension {embedding_dim}; expected {EMBEDDING_DIM}"
            );
        }
        let result_len = batch_size
            .checked_mul(SEQUENCE_LENGTH)
            .and_then(|count| count.checked_mul(embedding_dim))
            .context("CoreML output dimensions overflow")?;
        output.truncate(result_len);
        Ok((embedding_dim, output))
    }
}

impl Drop for CoreMlSession {
    fn drop(&mut self) {
        unsafe {
            if let Some(handle) = self.query_handle {
                next_plaid_coreml_destroy(handle);
            }
            if let Some(handle) = self.document_handle {
                next_plaid_coreml_destroy(handle);
            }
        };
    }
}

fn ffi_error(error: *mut c_char, fallback: &str) -> anyhow::Error {
    if error.is_null() {
        return anyhow::anyhow!(fallback.to_owned());
    }
    let message = unsafe { std::ffi::CStr::from_ptr(error) }
        .to_string_lossy()
        .into_owned();
    unsafe { next_plaid_coreml_free_error(error) };
    anyhow::anyhow!(message)
}

pub(crate) fn compiled_model_path(model_dir: &Path) -> Option<CoreMlModels> {
    let document_path = model_dir.join("MXBAIEdgeColbert-b8.mlmodelc");
    let query_path = model_dir.join("MXBAIEdgeColbert.mlmodelc");
    let has_b8 = [
        "coremldata.bin",
        "metadata.json",
        "model.mil",
        "analytics/coremldata.bin",
        "weights/weight.bin",
    ]
    .iter()
    .all(|file| document_path.join(file).is_file());
    let has_query = [
        "coremldata.bin",
        "metadata.json",
        "model.mil",
        "analytics/coremldata.bin",
        "weights/weight.bin",
    ]
    .iter()
    .all(|file| query_path.join(file).is_file());
    match std::fs::read_to_string(model_dir.join(NATIVE_COREML_MARKER))
        .ok()?
        .trim()
    {
        MXBAI_COREML_MODEL if has_b8 && has_query => Some(CoreMlModels::QueryAndB8 {
            query: query_path,
            document: document_path,
        }),
        MXBAI_COREML_QUERY_MODEL if has_query => Some(CoreMlModels::QueryOnly(query_path)),
        _ => None,
    }
}
