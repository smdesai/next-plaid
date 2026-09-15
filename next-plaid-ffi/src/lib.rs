//! UniFFI bindings over the `next-plaid` multi-vector search engine.
//!
//! Scope: **vector engine only**. Encoding (CoreML ColBERT) and tokenization
//! stay on the Swift side. This crate wraps [`next_plaid::MmapIndex`] behind a
//! thread-safe [`PlaidIndex`] object and exposes create / add / remove / search
//! / reconstruct over a flat, FFI-friendly embedding representation.
//!
//! Embeddings cross the boundary as [`EmbeddingMatrix`]: little-endian, row-major
//! `f32` bytes plus `rows`/`cols`. Callers must pass **unit-L2-normalized** rows;
//! the engine quantizes internally and does not normalize (see the plan's
//! "normalize at the Swift→Rust seam" note).

use std::sync::{Arc, Mutex, MutexGuard};

use ndarray::Array2;
use next_plaid::{IndexConfig, MmapIndex, QueryResult, SearchParameters, UpdateConfig};

uniffi::setup_scaffolding!();

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors surfaced across the FFI boundary. Each carries a human-readable
/// message; UniFFI renders this as a Swift `enum FfiError: Error`, so a failing
/// call becomes a Swift `throws`.
#[derive(Debug, thiserror::Error, uniffi::Error)]
pub enum FfiError {
    #[error("index creation failed: {0}")]
    Create(String),
    #[error("index load failed: {0}")]
    Load(String),
    #[error("search failed: {0}")]
    Search(String),
    #[error("update failed: {0}")]
    Update(String),
    #[error("delete failed: {0}")]
    Delete(String),
    #[error("shape error: {0}")]
    Shape(String),
    #[error("io error: {0}")]
    Io(String),
    /// Invalid argument or engine-level configuration/codec error.
    #[error("invalid argument: {0}")]
    Invalid(String),
}

impl From<next_plaid::Error> for FfiError {
    fn from(e: next_plaid::Error) -> Self {
        use next_plaid::Error as E;
        match e {
            E::IndexCreation(m) => FfiError::Create(m),
            E::IndexLoad(m) => FfiError::Load(m),
            E::Search(m) => FfiError::Search(m),
            E::Update(m) => FfiError::Update(m),
            E::Delete(m) => FfiError::Delete(m),
            E::Shape(m) => FfiError::Shape(m),
            E::Config(m) => FfiError::Invalid(m),
            E::Codec(m) => FfiError::Invalid(m),
            E::Filtering(m) => FfiError::Invalid(m),
            E::Io(err) => FfiError::Io(err.to_string()),
            E::Json(err) => FfiError::Invalid(err.to_string()),
            E::Sqlite(err) => FfiError::Invalid(err.to_string()),
            E::NpyRead(err) => FfiError::Io(err.to_string()),
            E::NpyWrite(err) => FfiError::Io(err.to_string()),
        }
    }
}

type FfiResult<T> = std::result::Result<T, FfiError>;

// ---------------------------------------------------------------------------
// Records crossing the boundary
// ---------------------------------------------------------------------------

/// A dense `[rows, cols]` `f32` matrix as little-endian, row-major bytes.
/// One matrix = one document's (or query's) per-token embeddings.
#[derive(uniffi::Record)]
pub struct EmbeddingMatrix {
    /// `rows * cols` `f32` values, little-endian, row-major.
    pub data: Vec<u8>,
    pub rows: u32,
    pub cols: u32,
}

/// One scored document for a query.
#[derive(uniffi::Record)]
pub struct SearchHit {
    pub doc_id: i64,
    pub score: f32,
}

/// Results for a single query in a batch.
#[derive(uniffi::Record)]
pub struct QueryHits {
    pub query_id: u64,
    pub hits: Vec<SearchHit>,
}

/// Outcome of a [`PlaidIndex::remove`] call. `deleted_ids_sorted` is the exact
/// set the engine compacted (in-range, deduped, ascending) so the Swift side can
/// reproduce the engine's renumbering: `new_id = old_id − count(deleted < old_id)`.
#[derive(uniffi::Record)]
pub struct RemoveOutcome {
    pub deleted_count: u64,
    pub deleted_ids_sorted: Vec<i64>,
}

/// Mirrors [`next_plaid::IndexConfig`]. UniFFI has no `usize`, so counts are `u64`.
/// `fts_tokenizer` is left at the engine default (metadata FTS is unused here).
#[derive(uniffi::Record)]
pub struct FfiIndexConfig {
    pub nbits: u64,
    pub batch_size: u64,
    pub seed: Option<u64>,
    pub kmeans_niters: u64,
    pub max_points_per_centroid: u64,
    pub n_samples_kmeans: Option<u64>,
    pub start_from_scratch: u64,
    pub force_cpu: bool,
    pub binary: bool,
}

impl From<FfiIndexConfig> for IndexConfig {
    fn from(c: FfiIndexConfig) -> Self {
        IndexConfig {
            nbits: c.nbits as usize,
            batch_size: c.batch_size as usize,
            seed: c.seed,
            kmeans_niters: c.kmeans_niters as usize,
            max_points_per_centroid: c.max_points_per_centroid as usize,
            n_samples_kmeans: c.n_samples_kmeans.map(|v| v as usize),
            start_from_scratch: c.start_from_scratch as usize,
            force_cpu: c.force_cpu,
            fts_tokenizer: Default::default(),
            binary: c.binary,
        }
    }
}

/// Mirrors [`next_plaid::UpdateConfig`]. Note `seed` is a plain `u64` here (the
/// engine's `UpdateConfig.seed` is not optional, unlike `IndexConfig.seed`).
#[derive(uniffi::Record)]
pub struct FfiUpdateConfig {
    pub batch_size: u64,
    pub kmeans_niters: u64,
    pub max_points_per_centroid: u64,
    pub n_samples_kmeans: Option<u64>,
    pub seed: u64,
    pub start_from_scratch: u64,
    pub buffer_size: u64,
    pub force_cpu: bool,
}

impl From<FfiUpdateConfig> for UpdateConfig {
    fn from(c: FfiUpdateConfig) -> Self {
        UpdateConfig {
            batch_size: c.batch_size as usize,
            kmeans_niters: c.kmeans_niters as usize,
            max_points_per_centroid: c.max_points_per_centroid as usize,
            n_samples_kmeans: c.n_samples_kmeans.map(|v| v as usize),
            seed: c.seed,
            start_from_scratch: c.start_from_scratch as usize,
            buffer_size: c.buffer_size as usize,
            force_cpu: c.force_cpu,
        }
    }
}

/// Mirrors [`next_plaid::SearchParameters`].
#[derive(uniffi::Record)]
pub struct FfiSearchParameters {
    pub batch_size: u64,
    pub n_full_scores: u64,
    pub top_k: u64,
    pub n_ivf_probe: u64,
    pub centroid_batch_size: u64,
    pub centroid_score_threshold: Option<f32>,
}

impl From<FfiSearchParameters> for SearchParameters {
    fn from(p: FfiSearchParameters) -> Self {
        SearchParameters {
            batch_size: p.batch_size as usize,
            n_full_scores: p.n_full_scores as usize,
            top_k: p.top_k as usize,
            n_ivf_probe: p.n_ivf_probe as usize,
            centroid_batch_size: p.centroid_batch_size as usize,
            centroid_score_threshold: p.centroid_score_threshold,
        }
    }
}

// ---------------------------------------------------------------------------
// Conversions
// ---------------------------------------------------------------------------

/// Decode an [`EmbeddingMatrix`] into an owned `Array2<f32>`, validating that the
/// byte length matches `rows * cols * 4`.
fn matrix_to_array2(m: &EmbeddingMatrix) -> FfiResult<Array2<f32>> {
    let rows = m.rows as usize;
    let cols = m.cols as usize;
    let expected = rows
        .checked_mul(cols)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| FfiError::Shape("embedding matrix dimensions overflow usize".into()))?;
    if m.data.len() != expected {
        return Err(FfiError::Shape(format!(
            "embedding byte length {} != rows*cols*4 = {} (rows={}, cols={})",
            m.data.len(),
            expected,
            rows,
            cols
        )));
    }
    let mut values = Vec::with_capacity(rows * cols);
    for chunk in m.data.chunks_exact(4) {
        values.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Array2::from_shape_vec((rows, cols), values)
        .map_err(|e| FfiError::Shape(format!("failed to build [{rows},{cols}] array: {e}")))
}

/// Encode an `Array2<f32>` back into an [`EmbeddingMatrix`] (little-endian,
/// row-major). `iter()` yields logical row-major order regardless of memory
/// layout, so reconstructed (possibly non-standard-layout) arrays are handled.
fn array2_to_matrix(a: &Array2<f32>) -> EmbeddingMatrix {
    let rows = a.nrows() as u32;
    let cols = a.ncols() as u32;
    let mut data = Vec::with_capacity(a.len() * 4);
    for &v in a.iter() {
        data.extend_from_slice(&v.to_le_bytes());
    }
    EmbeddingMatrix { data, rows, cols }
}

fn decode_all(matrices: &[EmbeddingMatrix]) -> FfiResult<Vec<Array2<f32>>> {
    matrices.iter().map(matrix_to_array2).collect()
}

/// Flatten the engine's parallel `passage_ids`/`scores` into `SearchHit`s.
fn zip_hits(res: QueryResult) -> Vec<SearchHit> {
    res.passage_ids
        .into_iter()
        .zip(res.scores)
        .map(|(doc_id, score)| SearchHit { doc_id, score })
        .collect()
}

// ---------------------------------------------------------------------------
// PlaidIndex
// ---------------------------------------------------------------------------

/// Thread-safe handle to an on-disk `MmapIndex`.
///
/// The engine's mutating methods (`update`/`delete`/`reload`) take `&mut self`
/// while UniFFI object methods receive `&self`, so the index lives behind a
/// `Mutex`. `MmapIndex` is `Send + Sync`, satisfying UniFFI's `Object` bound.
#[derive(uniffi::Object)]
pub struct PlaidIndex {
    inner: Mutex<MmapIndex>,
    /// Filesystem path of the index directory.
    path: String,
    /// Embedding dimension, captured at create/open (authoritative — not read
    /// from `Metadata.embedding_dim`, which is `#[serde(default)]` and may be 0).
    dim: u32,
}

impl PlaidIndex {
    fn lock(&self) -> FfiResult<MutexGuard<'_, MmapIndex>> {
        self.inner
            .lock()
            .map_err(|_| FfiError::Invalid("index lock poisoned by a prior panic".into()))
    }
}

#[uniffi::export]
impl PlaidIndex {
    /// Build a new index at `path` from `embeddings` (one matrix per document).
    /// Computes its own k-means centroids. Rejects an empty corpus.
    #[uniffi::constructor]
    pub fn create(
        path: String,
        embeddings: Vec<EmbeddingMatrix>,
        config: FfiIndexConfig,
    ) -> FfiResult<Arc<Self>> {
        if embeddings.is_empty() {
            return Err(FfiError::Invalid(
                "cannot create an index from an empty corpus".into(),
            ));
        }
        let arrays = decode_all(&embeddings)?;
        let dim = arrays[0].ncols();
        if let Some(bad) = arrays.iter().position(|a| a.ncols() != dim) {
            return Err(FfiError::Shape(format!(
                "inconsistent embedding dim: document {bad} has {} cols, expected {dim}",
                arrays[bad].ncols()
            )));
        }
        let cfg: IndexConfig = config.into();
        let index = MmapIndex::create_with_kmeans(&arrays, &path, &cfg)?;
        Ok(Arc::new(Self {
            inner: Mutex::new(index),
            path,
            dim: dim as u32,
        }))
    }

    /// Open an existing index directory.
    #[uniffi::constructor]
    pub fn open(path: String) -> FfiResult<Arc<Self>> {
        let index = MmapIndex::load(&path)?;
        let dim = index.embedding_dim() as u32;
        Ok(Arc::new(Self {
            inner: Mutex::new(index),
            path,
            dim,
        }))
    }

    /// Append documents. Returns the newly assigned internal ids.
    pub fn add(
        &self,
        embeddings: Vec<EmbeddingMatrix>,
        config: FfiUpdateConfig,
    ) -> FfiResult<Vec<i64>> {
        if embeddings.is_empty() {
            return Ok(Vec::new());
        }
        let arrays = decode_all(&embeddings)?;
        let cfg: UpdateConfig = config.into();
        let mut index = self.lock()?;
        Ok(index.update(&arrays, &cfg)?)
    }

    /// Delete documents by internal id, then reload so the handle reflects the
    /// compacted index. Ids are sanitized (in-range, deduped, ascending) before
    /// deletion; the returned set drives the caller's id-remap.
    pub fn remove(&self, ids: Vec<i64>) -> FfiResult<RemoveOutcome> {
        let mut index = self.lock()?;
        let n = index.num_documents() as i64;
        let mut sanitized: Vec<i64> = ids.into_iter().filter(|&id| id >= 0 && id < n).collect();
        sanitized.sort_unstable();
        sanitized.dedup();
        if sanitized.is_empty() {
            return Ok(RemoveOutcome {
                deleted_count: 0,
                deleted_ids_sorted: Vec::new(),
            });
        }
        // `MmapIndex::delete` deletes with `delete_metadata = true`, so it also
        // re-sequences the co-located `metadata.db` (and FTS) with the same
        // `new_id = old_id − count(deleted < old_id)` rule it applies to the
        // vectors — stored text stays tied to its embedding with no extra call.
        let deleted = index.delete(&sanitized)?;
        index.reload()?;
        Ok(RemoveOutcome {
            deleted_count: deleted as u64,
            deleted_ids_sorted: sanitized,
        })
    }

    /// Search with a single query. `subset`, if given, restricts candidates to
    /// those internal ids.
    pub fn search(
        &self,
        query: EmbeddingMatrix,
        params: FfiSearchParameters,
        subset: Option<Vec<i64>>,
    ) -> FfiResult<Vec<SearchHit>> {
        let q = matrix_to_array2(&query)?;
        let p: SearchParameters = params.into();
        let index = self.lock()?;
        let res = index.search(&q, &p, subset.as_deref())?;
        Ok(zip_hits(res))
    }

    /// Search a batch of queries. `parallel` runs them across threads.
    pub fn search_batch(
        &self,
        queries: Vec<EmbeddingMatrix>,
        params: FfiSearchParameters,
        parallel: bool,
        subset: Option<Vec<i64>>,
    ) -> FfiResult<Vec<QueryHits>> {
        let qs = decode_all(&queries)?;
        let p: SearchParameters = params.into();
        let index = self.lock()?;
        let results = index.search_batch(&qs, &p, parallel, subset.as_deref())?;
        Ok(results
            .into_iter()
            .map(|r| QueryHits {
                query_id: r.query_id as u64,
                hits: zip_hits(r),
            })
            .collect())
    }

    /// Reconstruct (dequantized) per-token embeddings for the given internal ids.
    pub fn reconstruct(&self, ids: Vec<i64>) -> FfiResult<Vec<EmbeddingMatrix>> {
        let index = self.lock()?;
        let arrays = index.reconstruct(&ids)?;
        Ok(arrays.iter().map(array2_to_matrix).collect())
    }

    pub fn num_documents(&self) -> FfiResult<u64> {
        Ok(self.lock()?.num_documents() as u64)
    }

    pub fn num_embeddings(&self) -> FfiResult<u64> {
        Ok(self.lock()?.num_embeddings() as u64)
    }

    /// Embedding dimension captured at create/open.
    pub fn embedding_dim(&self) -> u32 {
        self.dim
    }

    /// The index directory path.
    pub fn path(&self) -> String {
        self.path.clone()
    }
}

// ---------------------------------------------------------------------------
// Metadata / text store (SQLite `metadata.db` inside the index directory)
// ---------------------------------------------------------------------------
//
// Thin wrappers over `next_plaid::filtering`. Documents cross the boundary as
// JSON object strings (e.g. `{"documentName":..,"chunkText":..}`); the engine
// stores each keyed by its `doc_id` in the `_subset_` column, so a search hit's
// `doc_id` round-trips to its stored text. These are free functions (they take
// the index directory `path`) so metadata access does not require holding a
// live `PlaidIndex` handle.

/// Store `metadata_json[i]` for document `doc_ids[i]` in the index's
/// `metadata.db`. Creates the store on first call and appends thereafter, so it
/// can be invoked once per indexing batch. `doc_ids` must match the ids the
/// engine assigned (`create`/`add` order, 0,1,2,…) and align 1:1 with
/// `metadata_json`. Each JSON string must be a JSON object. Returns rows written.
#[uniffi::export]
pub fn store_documents(
    path: String,
    doc_ids: Vec<i64>,
    metadata_json: Vec<String>,
) -> FfiResult<u64> {
    if doc_ids.len() != metadata_json.len() {
        return Err(FfiError::Invalid(format!(
            "doc_ids length ({}) must match metadata_json length ({})",
            doc_ids.len(),
            metadata_json.len()
        )));
    }
    let values: Vec<serde_json::Value> = metadata_json
        .iter()
        .map(|s| serde_json::from_str(s))
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| FfiError::Invalid(format!("invalid metadata JSON: {e}")))?;

    // `create` wipes any existing db, so only use it for the first batch; append
    // with `update` once the store exists.
    let written = if next_plaid::filtering::exists(&path) {
        next_plaid::filtering::update(&path, values.as_slice(), doc_ids.as_slice())?
    } else {
        next_plaid::filtering::create(&path, values.as_slice(), doc_ids.as_slice())?
    };
    Ok(written as u64)
}

/// Fetch stored documents for `doc_ids`, as JSON object strings. Results are in
/// the same order as `doc_ids`; each row carries its `_subset_` (the doc_id).
/// Ids with no stored row are omitted, so callers should key results by
/// `_subset_` rather than by position. Returns an empty vec if no store exists.
#[uniffi::export]
pub fn get_documents(path: String, doc_ids: Vec<i64>) -> FfiResult<Vec<String>> {
    let rows = next_plaid::filtering::get(&path, None, &[], Some(doc_ids.as_slice()))?;
    rows.iter()
        .map(|v| serde_json::to_string(v))
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| FfiError::Invalid(format!("failed to serialize metadata row: {e}")))
}

/// Fetch stored documents matching a SQL `WHERE` `condition`, as JSON object
/// strings. The condition uses `?` placeholders bound, in order, to `params`
/// (each a JSON scalar string, e.g. `"\"report.txt\""` or `"3"`); it is
/// validated against the schema's columns and an allowlist grammar, so only
/// known columns and safe comparison/IN/BETWEEN/NULL operators are permitted.
/// Rows come back ordered by `_subset_` (the doc_id), each carrying it. Returns
/// an empty vec if no store exists.
///
/// Example: `condition = "documentName = ?"`, `params = ["\"report.txt\""]`
/// returns every chunk of that document, letting callers resolve a document's
/// ids without scanning the whole store.
#[uniffi::export]
pub fn query_documents(
    path: String,
    condition: String,
    params: Vec<String>,
) -> FfiResult<Vec<String>> {
    let values: Vec<serde_json::Value> = params
        .iter()
        .map(|s| serde_json::from_str(s))
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| FfiError::Invalid(format!("invalid query parameter JSON: {e}")))?;

    let rows = next_plaid::filtering::get(&path, Some(&condition), values.as_slice(), None)?;
    rows.iter()
        .map(|v| serde_json::to_string(v))
        .collect::<std::result::Result<_, _>>()
        .map_err(|e| FfiError::Invalid(format!("failed to serialize metadata row: {e}")))
}

/// Number of stored documents in the index's `metadata.db` (0 if none exists).
#[uniffi::export]
pub fn document_count(path: String) -> FfiResult<u64> {
    if !next_plaid::filtering::exists(&path) {
        return Ok(0);
    }
    Ok(next_plaid::filtering::count(&path)? as u64)
}

// ---------------------------------------------------------------------------
// Tests (host)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_matrix(positions: &[usize], dim: usize, rows: usize) -> EmbeddingMatrix {
        // Each of `rows` token rows is the same one-hot unit vector e_p (already
        // L2-normalized) for every position p in `positions` — here we use one
        // position per doc so a doc points in a single direction.
        assert_eq!(positions.len(), 1);
        let p = positions[0];
        let mut data = Vec::with_capacity(rows * dim * 4);
        for _ in 0..rows {
            for j in 0..dim {
                let v: f32 = if j == p { 1.0 } else { 0.0 };
                data.extend_from_slice(&v.to_le_bytes());
            }
        }
        EmbeddingMatrix {
            data,
            rows: rows as u32,
            cols: dim as u32,
        }
    }

    fn index_config() -> FfiIndexConfig {
        FfiIndexConfig {
            nbits: 2,
            batch_size: 50_000,
            seed: Some(42),
            kmeans_niters: 4,
            max_points_per_centroid: 256,
            n_samples_kmeans: None,
            start_from_scratch: 999,
            force_cpu: true,
            binary: false,
        }
    }

    fn update_config() -> FfiUpdateConfig {
        FfiUpdateConfig {
            batch_size: 50_000,
            kmeans_niters: 4,
            max_points_per_centroid: 256,
            n_samples_kmeans: None,
            seed: 42,
            start_from_scratch: 999,
            buffer_size: 100,
            force_cpu: true,
        }
    }

    // No centroid pruning / probe every cell: tiny indexes have few centroids.
    fn search_params(top_k: u64) -> FfiSearchParameters {
        FfiSearchParameters {
            batch_size: 2000,
            n_full_scores: 4096,
            top_k,
            n_ivf_probe: 1024,
            centroid_batch_size: 100_000,
            centroid_score_threshold: None,
        }
    }

    fn tmp_index_path(tag: &str) -> String {
        let mut p = std::env::temp_dir();
        let uniq = format!(
            "next_plaid_ffi_{tag}_{}_{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        p.push(uniq);
        p.to_string_lossy().into_owned()
    }

    const DIM: usize = 64;
    const TOKENS: usize = 4;

    #[test]
    fn create_and_search_returns_matching_doc() {
        let path = tmp_index_path("create_search");
        // Docs point along axes 0, 1, 2.
        let docs = vec![
            unit_matrix(&[0], DIM, TOKENS),
            unit_matrix(&[1], DIM, TOKENS),
            unit_matrix(&[2], DIM, TOKENS),
        ];
        let index = PlaidIndex::create(path, docs, index_config()).expect("create");
        assert_eq!(index.num_documents().unwrap(), 3);
        assert_eq!(index.embedding_dim(), DIM as u32);

        // Query along axis 1 -> doc 1 should top the ranking.
        let query = unit_matrix(&[1], DIM, 1);
        let hits = index.search(query, search_params(3), None).expect("search");
        assert!(!hits.is_empty(), "expected at least one hit");
        assert_eq!(hits[0].doc_id, 1, "top hit should be the axis-1 document");
    }

    #[test]
    fn add_appends_ids_without_renumbering_existing() {
        // Guards the plan's "add id stability" risk: adding more docs must not
        // renumber existing ones. With < start_from_scratch docs, `update` takes
        // the start-from-scratch rebuild path — the mode most likely to renumber.
        let path = tmp_index_path("add_stability");
        let docs = vec![
            unit_matrix(&[0], DIM, TOKENS),
            unit_matrix(&[1], DIM, TOKENS),
            unit_matrix(&[2], DIM, TOKENS),
        ];
        let index = PlaidIndex::create(path, docs, index_config()).expect("create");

        // Establish that axis-1 maps to id 1 before adding.
        let before = index
            .search(unit_matrix(&[1], DIM, 1), search_params(3), None)
            .expect("search before");
        assert_eq!(before[0].doc_id, 1);

        // Append two more docs along axes 3, 4.
        let new_ids = index
            .add(
                vec![unit_matrix(&[3], DIM, TOKENS), unit_matrix(&[4], DIM, TOKENS)],
                update_config(),
            )
            .expect("add");
        assert_eq!(index.num_documents().unwrap(), 5);
        // New ids are the appended tail range.
        assert_eq!(new_ids, vec![3, 4], "new ids should be the appended tail");

        // Existing content still resolves to its original id.
        let after = index
            .search(unit_matrix(&[1], DIM, 1), search_params(5), None)
            .expect("search after");
        assert_eq!(
            after[0].doc_id, 1,
            "axis-1 doc must keep id 1 after add (no renumber)"
        );
        // And a newly added doc is findable at its reported id.
        let new_hit = index
            .search(unit_matrix(&[3], DIM, 1), search_params(5), None)
            .expect("search new");
        assert_eq!(new_hit[0].doc_id, 3, "axis-3 doc should be id 3");
    }

    #[test]
    fn remove_middle_reports_sorted_deleted_and_compacts() {
        let path = tmp_index_path("remove_middle");
        let docs = vec![
            unit_matrix(&[0], DIM, TOKENS),
            unit_matrix(&[1], DIM, TOKENS),
            unit_matrix(&[2], DIM, TOKENS),
            unit_matrix(&[3], DIM, TOKENS),
        ];
        let index = PlaidIndex::create(path, docs, index_config()).expect("create");

        // Delete a middle doc (id 1); pass a duplicate and an out-of-range id to
        // exercise sanitization.
        let outcome = index.remove(vec![1, 1, 999, -5]).expect("remove");
        assert_eq!(outcome.deleted_count, 1);
        assert_eq!(outcome.deleted_ids_sorted, vec![1]);
        assert_eq!(index.num_documents().unwrap(), 3);

        // After compaction: old id 2 -> new id 1, old id 3 -> new id 2.
        // Axis-2 content now lives at id 1.
        let hit2 = index
            .search(unit_matrix(&[2], DIM, 1), search_params(3), None)
            .expect("search axis2");
        assert_eq!(hit2[0].doc_id, 1, "old id 2 should compact to id 1");
        let hit3 = index
            .search(unit_matrix(&[3], DIM, 1), search_params(3), None)
            .expect("search axis3");
        assert_eq!(hit3[0].doc_id, 2, "old id 3 should compact to id 2");
    }

    #[test]
    fn remove_suffix_keeps_ids_stable() {
        let path = tmp_index_path("remove_suffix");
        let docs = vec![
            unit_matrix(&[0], DIM, TOKENS),
            unit_matrix(&[1], DIM, TOKENS),
            unit_matrix(&[2], DIM, TOKENS),
        ];
        let index = PlaidIndex::create(path, docs, index_config()).expect("create");

        // Delete the last doc (suffix delete) -> survivors keep their ids.
        let outcome = index.remove(vec![2]).expect("remove");
        assert_eq!(outcome.deleted_ids_sorted, vec![2]);
        assert_eq!(index.num_documents().unwrap(), 2);

        let hit0 = index
            .search(unit_matrix(&[0], DIM, 1), search_params(2), None)
            .expect("search axis0");
        assert_eq!(hit0[0].doc_id, 0);
        let hit1 = index
            .search(unit_matrix(&[1], DIM, 1), search_params(2), None)
            .expect("search axis1");
        assert_eq!(hit1[0].doc_id, 1);
    }

    #[test]
    fn reconstruct_returns_matrices_of_expected_shape() {
        let path = tmp_index_path("reconstruct");
        let docs = vec![
            unit_matrix(&[0], DIM, TOKENS),
            unit_matrix(&[1], DIM, TOKENS),
        ];
        let index = PlaidIndex::create(path, docs, index_config()).expect("create");

        let mats = index.reconstruct(vec![0, 1]).expect("reconstruct");
        assert_eq!(mats.len(), 2);
        for m in &mats {
            assert_eq!(m.cols, DIM as u32);
            assert_eq!(m.rows, TOKENS as u32);
            assert_eq!(m.data.len(), (m.rows as usize) * (m.cols as usize) * 4);
        }
    }

    #[test]
    fn matrix_roundtrip_and_bad_length_rejected() {
        let m = unit_matrix(&[5], DIM, 2);
        let arr = matrix_to_array2(&m).expect("decode");
        assert_eq!(arr.shape(), &[2, DIM]);
        let back = array2_to_matrix(&arr);
        assert_eq!(back.data, m.data);

        let bad = EmbeddingMatrix {
            data: vec![0u8; 7], // not a multiple of rows*cols*4
            rows: 1,
            cols: 2,
        };
        assert!(matches!(matrix_to_array2(&bad), Err(FfiError::Shape(_))));
    }

    #[test]
    fn create_empty_corpus_is_rejected() {
        let path = tmp_index_path("empty");
        match PlaidIndex::create(path, vec![], index_config()) {
            Err(FfiError::Invalid(_)) => {}
            Err(other) => panic!("expected Invalid, got {other:?}"),
            Ok(_) => panic!("expected empty-corpus create to fail"),
        }
    }

    /// Helper: the `chunkText` stored for a single `doc_id`, via the metadata FFI.
    fn stored_text(path: &str, doc_id: i64) -> Option<String> {
        let rows = get_documents(path.to_string(), vec![doc_id]).expect("get_documents");
        let row: serde_json::Value = serde_json::from_str(rows.first()?).expect("row json");
        row.get("chunkText")
            .and_then(|v| v.as_str())
            .map(str::to_owned)
    }

    #[test]
    fn metadata_store_ties_text_to_doc_and_survives_renumber() {
        let path = tmp_index_path("meta_roundtrip");
        // Docs point along axes 0, 1, 2, 3.
        let docs = vec![
            unit_matrix(&[0], DIM, TOKENS),
            unit_matrix(&[1], DIM, TOKENS),
            unit_matrix(&[2], DIM, TOKENS),
            unit_matrix(&[3], DIM, TOKENS),
        ];
        let index = PlaidIndex::create(path.clone(), docs, index_config()).expect("create");

        // Store one text chunk per doc, keyed by the engine's doc ids 0..3.
        let doc_ids = vec![0_i64, 1, 2, 3];
        let metadata: Vec<String> = (0..4)
            .map(|i| format!(r#"{{"documentName":"doc{i}","chunkText":"text-{i}","chunkIndex":0}}"#))
            .collect();
        let written = store_documents(path.clone(), doc_ids, metadata).expect("store");
        assert_eq!(written, 4);
        assert_eq!(document_count(path.clone()).unwrap(), 4);

        // A search hit's doc_id round-trips to its stored text.
        let hit = index
            .search(unit_matrix(&[2], DIM, 1), search_params(4), None)
            .expect("search axis2");
        assert_eq!(hit[0].doc_id, 2);
        assert_eq!(stored_text(&path, hit[0].doc_id).as_deref(), Some("text-2"));

        // Delete the middle doc (id 1). Vectors compact (old 2->1, old 3->2) AND
        // the metadata store must renumber identically, so text stays tied.
        let outcome = index.remove(vec![1]).expect("remove");
        assert_eq!(outcome.deleted_ids_sorted, vec![1]);
        assert_eq!(document_count(path.clone()).unwrap(), 3);

        // Axis-2 content now lives at id 1 in BOTH stores.
        let hit2 = index
            .search(unit_matrix(&[2], DIM, 1), search_params(3), None)
            .expect("search axis2 after");
        assert_eq!(hit2[0].doc_id, 1, "old id 2 should compact to id 1");
        assert_eq!(
            stored_text(&path, hit2[0].doc_id).as_deref(),
            Some("text-2"),
            "text must follow the embedding through renumbering"
        );

        // Axis-3 content now lives at id 2.
        let hit3 = index
            .search(unit_matrix(&[3], DIM, 1), search_params(3), None)
            .expect("search axis3 after");
        assert_eq!(hit3[0].doc_id, 2, "old id 3 should compact to id 2");
        assert_eq!(stored_text(&path, hit3[0].doc_id).as_deref(), Some("text-3"));
    }

    #[test]
    fn query_documents_filters_by_column_and_returns_ids() {
        let path = tmp_index_path("meta_query");
        let docs = vec![
            unit_matrix(&[0], DIM, TOKENS),
            unit_matrix(&[1], DIM, TOKENS),
            unit_matrix(&[2], DIM, TOKENS),
            unit_matrix(&[3], DIM, TOKENS),
        ];
        let _index = PlaidIndex::create(path.clone(), docs, index_config()).expect("create");

        // Two documents: "A" owns chunks 0,1 ; "B" owns chunks 2,3.
        let doc_ids = vec![0_i64, 1, 2, 3];
        let metadata: Vec<String> = vec![
            r#"{"documentName":"A","chunkText":"A0","chunkIndex":0}"#.into(),
            r#"{"documentName":"A","chunkText":"A1","chunkIndex":1}"#.into(),
            r#"{"documentName":"B","chunkText":"B0","chunkIndex":0}"#.into(),
            r#"{"documentName":"B","chunkText":"B1","chunkIndex":1}"#.into(),
        ];
        store_documents(path.clone(), doc_ids, metadata).expect("store");

        // Filter by documentName; SQLite does the work, we get only A's chunks
        // ordered by _subset_ (the doc_id), each carrying it.
        let rows = query_documents(
            path.clone(),
            "documentName = ?".to_string(),
            vec![r#""A""#.to_string()],
        )
        .expect("query");
        let ids: Vec<i64> = rows
            .iter()
            .map(|s| {
                let v: serde_json::Value = serde_json::from_str(s).expect("row json");
                v.get("_subset_").and_then(|x| x.as_i64()).expect("_subset_")
            })
            .collect();
        assert_eq!(ids, vec![0, 1], "only A's chunk ids, ordered");

        // A condition on an unknown column is rejected (schema validation).
        assert!(query_documents(
            path.clone(),
            "nope = ?".to_string(),
            vec![r#""x""#.to_string()],
        )
        .is_err());
    }
}
