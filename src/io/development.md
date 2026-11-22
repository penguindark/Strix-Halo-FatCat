# src/io — development.md

## Goal
File and async I/O layer: GGUF v2/v3 reader (mmap-based, single + split
files), and later the `io_uring` queue used by `src/stream/` to pull cold
experts from NVMe.

## Status
**GGUF reader DONE (P1, 2026-11-01).** `gguf.c` + `include/fatcat/gguf.h`:

- Full GGUF v2/v3 parser: header, KV metadata (all 13 value types incl.
  arrays), tensor info table, alignment/data-start handling.
- **Split GGUF support with llama.cpp semantics**: each part is a
  self-contained GGUF; part 0 holds global metadata + `split.count`/`split.no`
  KVs; tensors carry a `part` tag and their offset is relative to their own
  part's data start. Validated against the real
  `Mistral-Large-4-1T-A52B-UD-IQ1_S` 6-part file (header) and
  `Mistral-Small-3.2-24B-UD-Q5_K_XL` (full single file).
- Bounds-checked everywhere (string/array length guards), mmap read-only.
- Quantization type table incl. new types (IQ1_S/IQ1_M, NVFP4, MXFP4, F8_E4M3).
- Unit tests pass (`tests/test_gguf.c`), including a synthetic split GGUF.
- **Streaming reads (2026-11-02, P1 profiler):** `fatcat_gguf_pread()` copies
  a mapped tensor range via `pread()` into a caller buffer (bypasses process
  page tables — used by the expert pool), and `fatcat_gguf_drop_range()`
  releases the backing page-cache range (`posix_fadvise(DONTNEED)`).
  Requires `#define _GNU_SOURCE`.

### Spec gotchas learned (keep in mind!)
- **GGUF `bool` is 1 byte** (not 4) — cost hours of debugging.
- Split GGUFs are NOT byte-concatenations: each part has its own header/KV/
  tensor table; part 0 may declare `n_tensors=0` with the tables in later parts.
- Tensor `offset` is relative to the *declaring part's* aligned data start.

## Scope
- In: GGUF parsing/mapping (done), tensor residency map, io_uring read queue
  (P3), hugepage-backed buffers.
- Out: GGUF writing/conversion (upstream quantizers produce files).

## Next steps
1. Tensor residency map: resident (mmap) vs streamed expert tensors.
2. io_uring ring wrapper with batch submit (P3).
3. `madvise(MADV_WILLNEED)`/`MADV_SEQUENTIAL` prefetch helpers for the streamer.
