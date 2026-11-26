# development_log.md — Append-only Development Log

Newest entries at the bottom. All entries in English (see `AGENT.md`).

---

## 2026-11-01 — Session 1: Project creation (P0)

- Discussed architecture: streaming MoE engine for Mistral-Large-4-1T-A52B
  (72 layers, 128+1 experts, top-4) on Strix Halo unified memory.
- Locked decisions: C23, CMake, GGUF input format, ROCm 10.1 gfx1151
  (`/home/user/rocm-10.1/install`), text-only first.
- Created project structure: root trackers (`AGENT.md`, `project.md`,
  `development_log.md`), `Manuals/` with AMD 57930 PPR 3.00 docs extracted
  from `AMD strix halo.7z`, source skeleton under `src/`, `tests/`, `tools/`,
  `benchmarks/`, `third_party/`, `docs/`.
- Added `development.md` to every code/script/test folder.
- Added CMake stub (C23, optional ROCm detection) and smoke `src/main.c`.
- Next: P1 — GGUF v3 reader in `src/io/` and expert-usage profiler in `tools/`.

## 2026-11-01 — Session 2: Performance doctrine adopted

- User set the governing objective: **maximum tokens/s** via fully optimized
  compute and the smartest possible expert cache; custom low-level matrix
  kernels (AVX-512 / HIP) explicitly in scope.
- Computed physical ceilings (4-bit): expert = 56.6 MB; ~26 GB active weights
  per token → decode ceiling ~9.8 tok/s @256 GB/s (realistic target ≥ 6 tok/s);
  cold NVMe expert miss = 8–19 ms → misses must always be prefetched.
- Added `docs/performance.md`: CPU/GPU kernel strategy, algorithmic multipliers
  (sparse indexer, speculative decoding), cache policy (LFU+recency, per-layer
  budgets, router-lookahead prefetch), measurement plan, kernel roadmap.
- Updated `project.md` (§3bis performance objective, decision log) and the
  `development.md` of `src/backend/cpu`, `src/backend/gpu`, `src/stream`.

## 2026-11-01 — Session 3: Architecture documentation gathered

- Downloaded ML4 architecture sources into `Manuals/model/`: upstream
  `config.json`, model card, HF release blog, vLLM day-0 blog (richest source:
  full architecture table, KV bytes/token, DSpark drafter, AMD/ROCm details),
  DSpark paper (arXiv 2607.05147), MLA/DeepSeek-V2 (2405.04434),
  DeepSeek-V3.2/DSA (2512.02556). Indexed in `Manuals/README.md`.
- Wrote `docs/model-architecture.md`: consolidated analysis — MLA+DSA stack,
  44 owned + 28 shared indexers, 9 layers reading layer 35's KV, 3:1
  RoPE/NoPE, LeRope, SSMax, built-in DSpark drafter (~2.9 tok/step, ~2×),
  KV ≈ 42–47 KB/token → 256K context ≈ 11 GB.

## 2026-11-01 — Session 4: Footprint & shared-machine constraints

- Locked: target **GGUF quants**, smallest footprint with best quality for
  **128 GB RAM / 256K context**; full model (260–600 GB by quant) streams
  from NVMe by design. Warm expert budget ≈ 90–95 GB → ~1,600 experts @Q4
  (~18%) or ~3,300 @Q2_K (~37%).
- Deferred step noted: GGUF repack for per-layer/tier **mixed quantization**.
- Locked: shared-machine workflow — early tests on **small model slices**
  (`tools/gguf-slice` planned); engine must support a **configurable hard RAM
  budget** with graceful degradation (slow but correct at tiny budgets) —
  budget-sweep curves are a primary characterization deliverable.
- Updated `project.md`, `docs/performance.md` (§3bis/3ter),
  `docs/model-architecture.md` (§6), `src/stream/`, `tools/` development.md.

## 2026-11-01 — Session 5: First test quant locked (UD-IQ1_S)

- Locked first test quant: `unsloth/Mistral-Large-4-1T-A52B-GGUF` →
  `UD-IQ1_S` (~1.7 bpw, 233 GB, 6 splits). Started background download to
  `/home/user/data/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/` via `hf`.
- At IQ1_S an expert ≈ 24 MB → warm budget (~90–95 GB) holds ≈ 3,900 experts
  (~43% of 9,088). IQ1_S dequant is table-based/slow — fine for concept
  validation; 1-bit quality is deliberately low (mechanism testing).
- Updated `project.md` decision table and `docs/model-architecture.md` §6.

## 2026-11-01 — Session 6: GGUF reader implemented (P1 core)

- Studied reference implementations on the net: `gguf-py` (GGUFReader) and
  llama.cpp `src/llama-model-loader.cpp` (split handling) + `ggml/src/gguf.cpp`.
- Implemented `src/io/gguf.c` + `include/fatcat/gguf.h` (C23, mmap,
  bounds-checked): full GGUF v2/v3 KV + tensor table parsing, quant size
  table (incl. IQ1_S/IQ1_M/NVFP4/MXFP4/F8_E4M3), name-sorted tensor lookup,
  single + split GGUF.
- Key discoveries (documented in `src/io/development.md`):
  - GGUF `bool` is **1 byte**, not 4 (cost a full debug cycle).
  - Split GGUFs are **not** byte concatenations: each part is a self-contained
    GGUF with own header/KV/tensor table; part 0 holds global metadata +
    `split.count`/`split.no`; tensor offsets are relative to their own part's
    data start. (ML4 IQ1_S part 1 declares `n_tensors=0`, `split.count=6`.)
- Built `tools/gguf-inspect` (header/KV/tensor dump, type histogram, expert
  analysis). Validated on Mistral-Small-3.2-24B-UD-Q5_K_XL (363 tensors,
  mixed UD quants visible) and on the IQ1_S part 1 header.
- `tests/test_gguf.c`: synthetic GGUF builder + single/split/garbage tests,
  always-on CHECK() macro (NDEBUG strips assert in RelWithDebInfo). PASS.
- CMake: `fatcat_io` static lib, `gguf-inspect` tool, `test_gguf` + CTest.
- IQ1_S download: ~142/233 GB at session end, still running.
- Next: full `gguf-inspect` pass on IQ1_S once download completes (expert
  tensor layout + real IQ1_S expert sizes), then `tools/gguf-slice`.

## 2026-11-01 — Session 7: IQ1_S download complete, real layout measured

- All 6 parts downloaded (233 GB) to
  `/home/user/data/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/UD-IQ1_S/`.
- Full `gguf-inspect` pass: 1,602 tensors, 233 GB, parsed across all 6 split
  parts by our reader.
- Key layout findings (recorded in `docs/model-architecture.md` §6):
  - Experts are STACKED tensors `blk.N.ffn_{gate,up,down}_exps.weight`
    (dims 6144×6144×128): expert `e` = contiguous slice at stride e×6144×6144
    elements → streaming an expert = 3 contiguous slice-reads.
  - UD mixed quant per tensor: 43× IQ2_XXS (9.73 MB/matrix/expert) +
    170× IQ1_S (7.37 MB) → avg expert ≈ 23.5 MB; routed total 214 GB.
  - Always-resident part (Q6_K/Q8_0/F32) measures 19 GB — matches budget.
  - Warm budget ⇒ ≈ 4,000 experts (≈44%) resident.
- Fixed gguf-inspect GB/layer display bug.
- Next: `tools/gguf-slice` (extract layers 0–2 + norms + lm_head into a small
  GGUF for shared-machine tests), then expert-usage profiler (P1).

## 2026-11-01 — Session 8: `gguf-slice` built, first ML4 slice extracted

- Implemented `tools/gguf-slice.c`: extracts `blk.0..blk.(N-1)` +
  `token_embd.weight`/`output.weight`/`output_norm.weight` from a (split)
  GGUF into a new single-file GGUF, preserving quant types exactly.
  - Rewrites `<arch>.block_count` to N and truncates every per-layer array
    KV (any array whose length == original block_count: `rope_pattern`,
    `kv_source_map`, `indexer.types`) — generic rule, no hardcoded keys.
  - Drops `split.count`/`split.no`/`split.tensors.count` from copied KV
    (gotcha: without this the reader rejects the single-file slice as a
    broken split model — caught by the built-in self-check).
  - Adds `fatcat.slice.source` / `fatcat.slice.orig_block_count` provenance;
    self-checks by reopening the output with our own reader (tensor count,
    block_count, boundary tensors, truncated arrays).
  - Options: `--keep SUBSTR`, `--no-tokenizer`, `--align N`.
- Smoke-tested on Mistral-Small-3.2-24B (dense llama, 2 layers), then
  produced the real slice:
  `/home/user/data/models/fatcat-slices/ML4-IQ1_S-layers0-2.gguf`
  — 77 tensors, 10.61 GB: layer 0 dense (Q8_0/Q6_K), layers 1–2 MoE
  (6 stacked expert tensors IQ2_XXS, 128 experts each), embeddings + lm_head
  (Q6_K) + full tokenizer. `block_count=3`, arrays truncated to 3.
- CMake: `gguf-slice` target.
- Next: expert-usage profiler (router traces → hotness distribution), then P2
  forward pass on this slice.

## 2026-11-02 — Session 9: GPU (HIP) backend for the ML4 forward pass

- New `src/gpu/`: `kernels.hip` (fused dequant+GEMV HIP kernels for F32/Q8_0/
  Q6_K/IQ1_S/IQ2_XXS, one block per output row, warpSize-aware block reduce —
  RDNA warp is 64 lanes, `__shfl_down_sync` mask must be 64-bit), `gpu.c`
  (thin C wrapper, `FATCAT_GPU=0` disables), `gpu_stub.c` (no-HIP builds).
  CMake compiles kernels.hip with hipcc (`--offload-arch=gfx1151`) and archives
  kernels.o into `fatcat_model`, links `amdhip64`.
- Design: CPU keeps control flow (rmsnorm/rope/router/attention/KV); GPU runs
  the big GEMVs on quantized weights (uploaded once); routed experts stream
  per token into 16 LRU-tagged 10 MB staging slots (tag = layer<<34|kind<<32|
  expert). Per-tensor CPU fallback if upload fails.
- Device verified: gfx1151, 20 CUs, 133 GB unified. `hipGetDeviceProperties`
  macro maps to R0600 (4 args) — use `hipDeviceGetAttribute`/`hipDeviceTotalMem`.
- ml4.c: `wt_upload()` for fixed weights, GPU paths in `gemv`/`gemv_expert`,
  per-token progress callback (`fatcat_ml4_set_token_cb`).

## 2026-11-02 — Session 10: GPU kernels debugged + profiler health logging

- Fixed kernel bugs (GPU logits were NaN):
  - IQ1_S: block/sub-block indices used `b` where `blk` was already block-offset
    (out-of-row reads → NaN) and covered only 1/4 of each row; rewrote as one
    thread per 8-value grid group mirroring `dequant.c` exactly.
  - IQ2_XXS: misaligned u32 reads at byte offset 2 (fine on CPU, garbage on
    GPU) → u16 pair reads; aux0/aux1 u16 indices must be `4*ib..4*ib+3`
    (byte offset 2+8·ib); element base was `b*32` instead of `b*256 + ib*32`.
  - `block_sum`: 64-bit shuffle mask for RDNA wavefront 64.
- `tests/gputest.c` (CMake target): GPU vs CPU GEMV on real tensors, all five
  types pass at norm-relative error ≈1.2e-7.
- End-to-end: full-model 1-token GPU logits vs llama.cpp: corr 0.9969,
  top-1000 overlap 931/1000 (CPU baseline was 0.9970/936) — GPU path correct.
- Sticky-error fix: a failed `hipMalloc` during model load surfaced later as a
  bogus "launch: out of memory"; gemv now clears sticky errors before launch,
  and upload failures print + count (`gpu stats: ... upload_fails=N`).
- Profiler health logging (so a manual run is self-verifying):
  - startup `selftest:` — GPU-vs-CPU GEMV per quant type on real tensors,
    aborts rc=3 on failure;
  - `health=ok|BAD` + `rss=` on every progress line (NaN/Inf scan of hidden
    state per token via new `fatcat_ml4_get_health()`);
  - footer `health: ... -> OK|FAIL`, `DONE rc=` reflects health (rc=4 on NaN);
  - build stamp in log header.
- `RUN.md`: single-command manual run instructions + health checklist.
- Next: user runs full 949-token profile after assistant shutdown → parse
  `profile-gpu.log` + `full-gpu.trace` → tier budgets in
  `docs/streaming-policy.md`.

## 2026-11-02 — Session 11: profiler memory cap (RSS self-limits at ~60 GB)

- Problem: first full run got ~0.45 tok/s then froze as RSS passed ~105 GB —
  mmap'd expert pages accumulate in the page cache (counted in RSS) and the
  system started thrashing NVMe re-reads.
- Fix in `ml4.c`:
  - `wt_upload()` now `madvise(MADV_DONTNEED)`s a fixed weight's mmap copy
    right after it is uploaded to the GPU (dead ~14 GB of page cache freed).
  - New policy: once RSS > `FATCAT_MEM_LIMIT_MB` (default 60000), every
    streamed expert slice is `MADV_DONTNEED`'d after use (page-aligned,
    fully-covered pages only). One-time `mem: rss > N MB ...` notice on stderr.
  - Verified with `FATCAT_MEM_LIMIT_MB=30000` on 24 tokens: RSS plateaus at
    the limit then settles ~17 GB floor (GPU copies + embeddings + KV),
    health=ok, DONE rc=0.
- `RUN.md` / `run_profile.sh` document the cap and the override.

## 2026-11-02 — Session 12: expert streaming via parallel pread (memory flat, no thrash)

- Problem (2nd freeze): even with the Session-11 madvise cap, RSS kept
  growing ~2.5 GB/token and the box froze at ~105 GB. smaps showed the RSS
  was Pss_File (gguf mmap pages, 44 GB) — GPU/TTM allocations are not in
  RSS, and readahead stragglers escaped the drop range. Separately, the
  mmap-fault reads were single-threaded at ~1 GB/s: the real speed wall
  (852 expert slices/token ≈ 6.8 GB/token).
- Fix: stop streaming experts through the process page tables entirely.
  - `src/io/gguf.c`: new `fatcat_gguf_pread()` (pread a mapped tensor range
    into a caller buffer) and `fatcat_gguf_drop_range()`
    (`posix_fadvise(DONTNEED)` on the backing file range). Declared in
    `include/fatcat/gguf.h`. Needed `#define _GNU_SOURCE`.
  - `ml4.c`: reusable expert pool (24 × 10 MB aligned slots, 240 MB);
    before each MoE layer's gemvs, its topk×3 slices are prefetched in
    parallel (`omp parallel for`, 8 threads) — deep NVMe queue instead of
    synchronous faults. `gemv_expert()` now takes a materialized `src`.
  - Memory policy now watches **system usage** (MemTotal−MemAvailable via
    /proc/meminfo), not RSS; default `FATCAT_MEM_LIMIT_MB=80000`; when
    exceeded, each expert's page-cache range is fadvise-dropped after use.
  - Profiler progress line reports `mem=<MB used>` instead of `rss=`.
- Verified (assistant LLM still running, 16 upload CPU-fallbacks included):
  - 40 tokens: 0.34 tok/s, `mem=47038MB` FLAT all run (was +2.5 GB/token),
    health=ok, DONE rc=0. Forced `FATCAT_MEM_LIMIT_MB=30000` on 12 tokens:
    still flat, rc=0.
  - Correctness unchanged: 1-token logits vs llama.cpp reference
    corr=0.9970, top-1000 overlap 931/1000, argmax matches.
  - `gpu_upload: hipMalloc(...) failed, CPU fallback` lines seen in diag
    logs are the graceful per-tensor fallback under memory pressure (the
    unified GPU allocates from system RAM); standalone runs should show
    upload_fails=0.
- `RUN.md` / `run_profile.sh` updated for the new mechanism and `mem=` field.
- Next: user runs full 949-token profile standalone → parse
  `profile-gpu.log` + `full-gpu.trace` → tier budgets in
  `docs/streaming-policy.md`.

## 2026-11-02 — Session 13: full 949-token GPU profile run + trace analysis

- User ran `./run_profile.sh` standalone: **949/949 tokens in 32:22
  (0.49 tok/s), health=ok every token, upload_fails=0, selftest OK,
  DONE rc=0**. Memory flat at 17.9–18.2 GB the whole run (min 17864 /
  max 18238 MB) — the Session-12 pread pool completely fixed the freeze.
  GPU stats: 808 975 staging copies, 6 361.6 GB read (~3.3 GB/s avg).
- Added `tools/analyze_trace.py` (FCT1 parser + greedy weight/byte tier
  sweep). Key findings written into `docs/streaming-policy.md`:
  - routed payload 221.3 GB; hotness is FLAT — median layer touches
    126/128 experts in 949 tokens; cov50/80/95/99 = 11/37/74/100 experts.
  - best-case warm hit rates: 16 GB→45.8%, 32 GB→61.6%, 48 GB→71.9%,
    64 GB→79.4% (independent-draw lower bound; LRU temporal locality adds
    more). No static pin set gets high hit rates → P3 must lean on LFU+recency
    + prefetch overlap (~7–9 ms/expert NVMe read).
  - stacked expert tensors are uniform-stride → 1 pread per matrix, no repack.
- P1 is now complete (reader + kernels + GPU backend + full-model profiler +
  measured policy). Next: P2 engine work + correctness vs llama.cpp on slice.

## 2026-11-17 — Session 14: P2a engine core + incremental decode; multi-token correctness hunt

- `src/core/engine.c` + `include/fatcat/engine.h`: engine context (open/reset/
  prefill/step/generate, greedy argmax, prefill/decode timing). `tools/fatcat-run.c`
  CLI generates N tokens from a token-id prompt.
- `ml4.c`: incremental decode — `seq_len` position tracking (forward continues
  from current position), `fatcat_ml4_reset()`, KV growth accounts for seq_len.
  Profiler now resets between independent input sequences (bug: its second
  `--check` pass ran at shifted positions).
- New diagnostics: `tools/logit-dump.c` (fatcat per-position logits via the
  incremental path), `tools/llama-ref-multi.cpp` (llama.cpp per-position
  reference; needs batch.logits set per position).
- **Bug found & fixed: YaRN mscale.** llama.cpp cancels the yarn magnitude
  mscale for this model (yarn_attn_factor = 1/(1+0.1*ln 2048) then rope_yarn
  multiplies (1+0.1*ln 2048) back -> net 1.0). fatcat applied mscale=1.7625
  to all cos/sin -> rope scores inflated ~3.1x. Invisible in the 1-token test
  (single-key softmax cancels score scaling!). 8-token corr 0.819 -> 0.870.
- Per-position vs llama.cpp (8-token prompt, after mscale fix):
  corr 0.997/0.964/0.937/0.949/0.918/0.887/0.871/0.876 — argmax matches at
  EVERY position (greedy generation stays on track), but a gradual divergence
  grows with position. FATCAT_KV_F16=1 collapses corr to ~0.3 => llama.cpp's
  MLA KV path here is effectively f32; fatcat's f32 default is correct.
- Open: identify the residual position-dependent gap (suspects: NoPE layer
  pattern interpretation, rope freq_factors application).

## 2026-11-17 — Session 15: P2c root cause found — fatcat is numerically correct

Deep per-layer bisect vs llama.cpp on the 8-token corpus. Four real bugs found
and fixed in `src/model/ml4.c`; the residual logit gap vs llama.cpp was then
proven to be llama.cpp's lossy fast kernels, not a fatcat error.

- **Bug 1: missing final logit softcap.** Model has
  `final_logit_softcapping=30`; llama.cpp applies `30*tanh(x/30)` after the
  output projection, fatcat did not. Logit stds were inflated ~15-30%.
- **Bug 2: wrong RMSNorm eps key.** llama.cpp reads
  `attention.layer_norm_epsilon` (=1e-6 here); fatcat read
  `attention.layer_norm_rms_epsilon` (absent) and silently used the 1e-5
  default. Now reads the llama.cpp key.
- **Bug 3: rope pairing convention.** ggml `GGML_ROPE_TYPE_NORM` (what
  MISTRAL_LARGE4 uses, see llama-model.cpp) rotates *adjacent* pairs
  (2j, 2j+1) via `rotate_pairs(n, n_offset=1, scale=1)`; fatcat applied the
  NEOX half-split (j, j+n_rot/2). `rope_apply` fixed to adjacent pairs.
- **Bug 4 (biggest): CPU gemv fallback silently zeroed wide rows.** The
  `tmp[8192]` dequant scratch had a guard `if (n_in > 8192) { y[i]=0; }`
  marked "never in ML4" — but the dense-FFN down projection has n_in=24576,
  so layer 0's entire FFN output was exactly zero. Fixed by dequantizing in
  8192-element chunks (every GGUF block size divides 8192), so any row width
  is exact. Layer-0 residual corr jumped 0.82 -> 0.99999.
- New diagnostics: `FATCAT_DUMP_HID` env hook in ml4.c dumps the residual
  stream per token (embd, then per layer attn_out/ffn_out/x_after as
  6144-f32 blocks); `tools/tensor-dump.c` dequantizes any GGUF tensor to raw
  f32 (now a CMake target); `tools/llama-layer-dump.cpp` dumps llama.cpp's
  per-layer residual via the public `llama-ext.h` embeddings_layer_inp API;
  `llama-ref-multi` gained `--seq` (one-token-per-decode) mode.
- **Golden verification (decisive).** Built a numpy-f64 layer-0 forward from
  bit-exact dequantized weights (fatcat dequant == ggml, reconfirmed):
  fatcat delta relerr vs golden = **4.2e-6**; llama.cpp delta relerr =
  **1.3e-2**. The engine gap is llama.cpp's Q8_0 int8-activation GEMV and
  f16-accumulating kernels, not fatcat. A Q8_0 int8-activation emulation was
  tried in fatcat (FATCAT_EMU_Q8X) and made agreement *worse* (per-layer
  median relerr 0.015 -> 0.021 mean 0.094) -> hypothesis "matching llama.cpp's
  quantization noise closes the gap" rejected; emulation code removed.
- Model is not chaotic at f32 noise level: fatcat f32-acc vs f64-acc builds
  give logit corr 1.0000 at all 8 positions; llama.cpp seq-vs-batch decode
  itself is flat ~0.998. f16 KV emulation (FATCAT_KV_F16) makes no
  difference (MLA KV path is f32 in llama.cpp here).
- Final 8-token logit corr vs llama.cpp: 0.9985/0.9834/0.9785/0.9749/0.9755/
  0.9640/0.9583/0.9595. Argmax matches at every position except pos 1, where
  llama.cpp's own top-2 margin is 0.062 (knife-edge; fatcat is closer to the
  golden forward there). Per-layer residual corr: 1.0 at embd, >=0.9993
  through layer 10, drifting to ~0.91 at layer 72 purely from llama.cpp's
  compounding kernel noise.
- **Verdict: P2c numerical correctness achieved.** fatcat's reference forward
  is more accurate than llama.cpp's; remaining logit correlation loss is
  llama.cpp-side noise amplified by depth. P2c closed.
- Next: P2b AVX-512 kernels (dot_f32, rmsnorm, fused dequant-GEMV for
  Q8_0/Q6_K/IQ1_S/IQ2_XXS) in `src/backend/cpu/`, then P3 streaming engine.

### Session 15 addendum: greedy generation validation (P2c closure)

- 19-token prompt ("The capital of France is Paris... Italy is"), 24-token
  greedy generation, fatcat-run vs llama-ref-multi --gen (new mode in
  tools/llama-ref-multi.cpp): **20/24 tokens identical**.
  fatcat: " Rome. The capital of Spain is Madrid. The capital of Portugal is
  Lisbon. The capital of Belgium is Brussels."
  llama.cpp: same through "...Lisbon. The capital of" then diverges to
  " the United Kingdom is". Both continuations coherent and factually
  correct — classic knife-edge greedy branch, not an engine error.
- fatcat speed on this run (GPU on, 40 GB mem cap): prefill 0.11 tok/s,
  decode 0.12 tok/s — expected for NVMe expert streaming without P3 cache.
- P2c closed: numerically correct (golden f64 check) AND behaviorally
  consistent with llama.cpp under greedy generation.

## 2026-11-17 — Session 16: P2b AVX-512 CPU kernels

- New backend `src/backend/cpu/` (`include/fatcat/cpu.h`): fused dequant+GEMV
  row kernels for Q8_0/Q6_K/IQ1_S/IQ2_XXS/F32 + `dot_f32` + `rmsnorm`, all
  AVX-512 (F/BW/VL/DQ). Runtime dispatch with scalar fallback
  (`fatcat_dequant` + naive dot); `FATCAT_CPU_SCALAR=1` forces scalar for A/B.
  Row-level OpenMP lives inside the kernels; `ml4.c` gemv/gemv_expert CPU
  paths, dot_f32 and rmsnorm now route through the backend.
- Design: exact-f32 (dequant to f32 registers, FMA) — decoded values are
  bit-exact with the scalar reference; only accumulation order differs. No
  int8-VNNI activation quantization (llama.cpp's lossy trick, rejected in
  P2c). Decoded values fold into the dot in 8–32 element register windows —
  no full-row f32 scratch (old path streamed a 32 KB tmp per row).
- Q6_K layout gotcha: within a 128-elem half, element e uses qh byte e%32
  (NOT e/4); nibble/2-bit-field interleave mirrors dequant.c exactly. Caught
  by the new test before it could poison the engine.
- `tests/test_cpu_kernels.c`: random quant blocks (f16 scales patched finite)
  vs fatcat_dequant+dot reference; all PASS at ~1e-6 relerr. Registered in
  ctest. `tools/cpu-bench` benchmarks any GGUF tensor.
- Measured (warm cache): single-thread speedups Q8_0 4.7x, Q6_K 7.6x,
  IQ1_S 5.2x, IQ2_XXS 28.8x (scalar IQ2_XXS per-element sign loop was the
  worst offender). 32-thread: Q8_0/Q6_K ~220-260 GB/s (L3-resident),
  IQ1_S ~33 GB/s, IQ2_XXS ~58 GB/s.
- End-to-end 8-token CPU forward (FATCAT_GPU=0): 84.4 s -> 54.8 s wall
  (1.54x; remainder is NVMe expert streaming). Logits bit-identical to the
  scalar path (corr 1.000000 all positions); GPU path unchanged (corr vs
  llama.cpp identical 0.9985..0.9595).
- Bench noise note: this box is shared with the assistant LLM; multi-thread
  numbers swing wildly under load — use best-of-many, trust 1T ratios.
- P2 complete (P2a engine core, P2b kernels, P2c correctness). Next: P3
  streaming engine (tiered expert cache + io_uring prefetch).

## 2026-11-17 — Session 17: P3 v1 warm tier

- New `src/stream/stream.c` + `include/fatcat/stream.h`: the warm tier. Fixed
  slot arena (default 10 MB slots >= largest expert matrix), slices keyed by
  (layer, matrix, expert). Open-addressing key hash with backward-shift delete;
  LFU+recency score `count / (1 + age_tokens/64)`, recency clock ticked once
  per generated token; victim = min score over a 16-slot random sample.
  `pin` = never-evict (P4 hot tier hook). Thread-safe: mutex + condvar
  coalesces concurrent gets of one key; fills run outside the lock.
  Budget `FATCAT_WARM_MB` (default 32768; 0 disables -> mmap fallback).
  On a miss the slice is pread() into its slot and the file range is
  fadvise(DONTNEED)-ed: the warm tier IS the cache, page cache stays small.
- `fatcat_gguf_resolve()` added (pointer -> backing fd + file offset).
  `ml4.c` MoE path: 12 slices/layer fetched via tier in a parallel pread
  loop, replacing the old fixed 24x10 MB round-robin pool (removed).
- Bugs found and fixed:
  1. `tab_erase` backward-shift left the vacated cell set (duplicate entries
     -> `tab_find` infinite loop). One-line fix: clear `tab[i]` after shift.
  2. **Use-after-evict**: caller held slot pointers across the expert GEMV
     loop while a later miss could evict + overwrite them (LFU ties make this
     likely under thrash). Caught because warm-tier logits differed from the
     mmap path at one position (corr 0.999715). Fix: per-slot use-count —
     `get()` takes a reference, new `fatcat_stream_release()` drops it,
     victim selection skips referenced slots; ml4.c releases after each
     layer's expert loop.
- `tests/test_stream.c` (1 MB tier, 64 KB slots, 16 slots < 64 blocks):
  data integrity, stats, capacity, LFU survival under one-shot flood, pin,
  use-reference survival, concurrent same-key coalescing, disabled tier.
  ALL PASS. Victim sample raised 7 -> 16 after 7 caused unlucky evictions of
  hot blocks in the test (and would in production).
- E2E (8 tokens, CPU, 40 GB mem cap, vs golden + mmap-path baselines):
  warm-tier logits now **bit-identical** to the no-tier path. 8 GB tier:
  13.1% hits (tier < 8.3 GB per-token working set -> intra-token thrash).
  16 GB tier: 29.2% hits on 8 tokens (union of experts across 8 tokens
  exceeds 1638 slots; policy-doc 45.8% is the 949-token steady state).
  Wall ~62 s (no tier, sys 9m36s of page-fault/cache churn) vs ~72 s
  (16 GB tier, sys 24 s): tier removes kernel overhead but v1's synchronous
  per-layer preads mean wall-time gains need either steady state (long runs)
  or P3 v2 async prefetch.
- **KV growth bug (found by the 512-token run):** the KV grow path sized the
  new capacity as `n_tokens + 64` (batch size!) instead of total sequence
  length; per-token decode kept cap=65 and position 65 wrote past the buffer
  -> heap corruption, SIGSEGV in memcpy. Fixed: cap covers seq_len+n_tokens+64
  and doubles. (Earlier runs were all <= 64 tokens or used the GPU path.)
- 512-token run (16 GB tier): 23.5% hits — below the policy doc's 45.8%
  static-allocation bound; LFU+recency needs tuning (aging window, sampling)
  and/or router-lookahead prefetch to close the gap.
- **Memory discipline (hard rule):** this box also hosts the assistant LLM
  (~68 GB RSS); a 16 GB tier + 40 GB cache cap OOM-killed it. All agent-run
  tests must use `FATCAT_WARM_MB<=6144 FATCAT_MEM_LIMIT_MB<=15000` and short
  corpora; long/full-budget runs are for the user standalone. logit-dump now
  prints peak RSS (getrusage) so every run's footprint is auditable.
- Next (P3 v2): io_uring (raw syscalls, no liburing) async prefetch queue;
  overlap layer L expert reads with layer L shared-expert/attention compute;
  router-lookahead hints. Then P4 GPU-pinned hot experts.

## 2026-11-17 — Session 18: P3 v2 io_uring async prefetch

- `src/stream/uring.c`: minimal raw io_uring wrapper (io_uring_setup/enter
  syscalls, no liburing): SQE queue, submit with GETEVENTS, CQE reap.
  `FATCAT_URING=0` disables the ring.
- stream tier: background fill worker + request queue. New API
  `fatcat_stream_prefetch(key, fd, off, len)`: claims a slot (LFU victim,
  marked in-flight), queues the read, returns immediately; a later get()
  hits or waits on the condvar. Ring batch = 64 reads; fallback worker does
  serial pread when the ring is unavailable. Stats semantics: a get() that
  had to wait for an in-flight fill now counts as a MISS (honest KPI:
  "fraction of expert requests that did not block on NVMe").
- ml4.c: layer L's 12 expert slices are prefetched immediately after its
  router, so NVMe streams while the shared-expert FFN computes; the get()
  loop after the FFN waits only for what hasn't landed.
- test_stream: + prefetch correctness (64 keys > 16 slots, waits + churn),
  + FATCAT_URING=0 fallback path. ALL PASS.
- E2E (8 tokens, 2 GB tier, mem cap 15 GB): logits bit-identical to the
  no-tier path; 44.9% of requests served without blocking; wall 44.7 s vs
  62 s no-tier baseline (-28%), sys 9m36s -> 18 s.
- **Memory reality check (peak RSS printout):** process baseline is ~16 GB
  RSS (mmap'd shared/attention/embedding weights) ON TOP of the warm tier.
  The 16 GB-tier runs were really ~32 GB RSS — that's what OOM-killed the
  host LLM. Agent runs: tier <= 4 GB (total ~20 GB RSS). Full-budget runs
  remain the user's standalone job.
- Next: policy tuning vs the 45.8% static bound, deeper prefetch lookahead
  (router speculation), per-layer budgets; then P4 GPU pinning.

## 2026-11-17 — Session 19: P3 v2.1 deferred per-slice waits + OMP spin fix

- ml4.c: replaced the batched 12-slice fetch with `expert_slice()` calls
  interleaved into the expert GEMV loop (1B): each get() blocks only on its
  own slice, so expert k's compute overlaps the layer's remaining in-flight
  reads. Tier refs released after the layer loop as before.
- stream worker: 200 us burst coalescing so a layer's 12 prefetches submit
  as one deep io_uring batch instead of serializing on early wakeups.
- **Big find — libgomp spin-wait cost 3x wall time.** With fine-grained
  waits, 31 OMP worker threads spin at every region end while the main
  thread blocks on I/O: user time ballooned to ~24 min for a 15 s run and
  the spinning saturated cores + DRAM, slowing real GEMVs and io_uring.
  `OMP_WAIT_POLICY=passive` fixes it, but libgomp reads env at LOAD time
  (constructor), so setenv inside the library is too late. logit-dump now
  does a one-time guarded self re-exec to default passive unless the user
  set a policy. (Library users must set the env themselves — documented.)
- E2E (8 tokens, 2 GB tier, 15 GB cap): 18.0 s wall vs 44.7 s (v2) and
  62 s (no-tier baseline) => 3.4x total; user 1m33s (was 24 min); logits
  bit-identical to no-tier path. Peak RSS 18.2 GB (baseline ~16 GB + tier).
- test_stream ALL PASS (incl. prefetch + fallback). Memory checks before
  every run; flash_serve (host LLM) RSS ~68 GB is mostly file-backed,
  MemAvailable ~90 GB held throughout.

## 2026-11-17 — Session 20: budget sweep tool + router-speculation prefetch (1A)

- `tools/budget-sweep.sh`: standalone tok/s + hit-rate vs tier budget sweep
  (default budgets 8/16/32/48 GB). Checks MemAvailable per budget and skips
  what would not fit (baseline ~16 GB RSS + budget). Core P3 deliverable
  per docs/performance.md §3bis — run by the user standalone.
- 1A router speculation: after each layer's attention residual, run the
  NEXT MoE layer's router on the current hidden state (missing this layer's
  FFN delta) and prefetch its top-4 slices — reads start one full layer
  earlier. Wrong guesses cost bandwidth, never correctness. Issued AFTER
  the current layer's critical-path reads (priority), works from dense
  layers too (covers the dense->MoE boundary). `FATCAT_SPEC=0` disables.
- Refactor: router softmax+top-k factored to `router_topk()`; prefetch loop
  to `prefetch_experts()`; counters via `fatcat_ml4_get_spec()`, printed by
  logit-dump.
- Measured (8 tokens, 2 GB tier): speculation accuracy 74.3% of actual
  top-4 experts pre-guessed; warm-tier hit rate 82.4% with spec off (the
  deferred per-slice waits let most prefetches land before their get());
  wall 16-18 s (bandwidth-bound at this tier size — speculation's win is
  larger at bigger budgets where reads overlap a whole layer). Logits
  bit-identical with spec on and off.

## 2026-11-17 — Session 21: P4 v1 GPU wired into the engine + zero-copy tier

- ml4.c now initializes the GPU (FATCAT_GPU_SLOTS staging slots, default 48;
  FATCAT_GPU=0 off) before weight uploads, and shuts it down on close. The
  GPU path had existed since P1 but was only reachable from the profiler.
- Mapped-pinned staging for GEMV in/out (hipHostMalloc mapped): 3 syncs per
  GEMV -> 1.
- **P4 zero-copy warm tier**: stream slots can be allocated as device-mapped
  pinned memory via a new allocator hook (`fatcat_stream_set_allocator`);
  on the APU (unified addressing probe) the GPU reads streamed expert
  slices directly from warm-tier slots — no GTT copies. gemv_expert wraps
  tier-slot pointers as weight handles (`fatcat_gpu_wrap`).
- logit-dump prints GPU stats (copies/GB, stage hits, upload fails).
- E2E (8 tokens, 2 GB tier, GPU on): copies 68.3 GB -> 14.8 GB with
  zero-copy; wall 55.7 s (copy-based) -> 28.0 s (zero-copy). Numerics in
  the established GPU band (corr vs golden 0.9985..0.9595; top-1 matches
  the CPU path at all positions).
- Honest status: GPU is still SLOWER than the optimized CPU path (28 s vs
  17 s). Causes: (a) per-GEMV launch+sync overhead (~14k calls/run) on
  TheRock, (b) GTT contention — flash_serve (the host LLM) shares the iGPU
  and its allocations make ~40 MB hipMallocs fail at blk.70/71 (CPU
  fallback handles it correctly). P4 v2: async GEMV pipelining (batch
  gate+up launches, one sync per expert), overlap staging with compute,
  and re-measure when the GPU is idle.
- test_stream flake fixed: LFU test now reinforces hot keys 200x so their
  score can never tie with one-shot blocks under random victim sampling.

## 2026-11-17 — Session 22: P4 v2 async GEMV pipelining

- kernels.hip: launch refactor (shared launch_gemv helper) + new async API:
  fatcat_gpu_gemv_dev() (launch-only, x/y are device-accessible pointers,
  no staging copies) and fatcat_gpu_wait() (device sync). Wrappers in
  gpu.c, stubs in gpu_stub.c, decls in gpu.h.
- ml4.c: forward scratch is now persistent (allocated once, freed on close)
  and device-mapped pinned when the GPU has unified addressing, so async
  kernels read x / write y in place. Expert loop restructured to 2 device
  syncs per MoE layer: launch all gate+up GEMVs async as slices arrive ->
  wait -> CPU silu + all down GEMVs async -> wait -> weighted accumulate.
  Per-expert fa[k]/fb[k]/fc[k] scratch (3*ML4_MAX_TOPK+1 slots). Staging
  fallback only when nothing is queued on the device (stage() could else
  evict a slot a pending kernel is reading). FATCAT_GPU_PIPE=0 A/B toggle.
- Validation: CPU path bit-identical to eight_fc_notier.bin after the loop
  restructure; GPU logits in the P2c band (0.9985..0.9579 vs golden), top-1
  matches CPU at all 8 positions; all unit tests pass.
- A/B on the shared box (6 runs each, min wall, 2 GB tier): pipelined
  25.3 s vs sync 25.6 s -> NEUTRAL. Honest analysis: batch-1 GEMV is
  memory-bandwidth-bound, and on an APU the GPU reads the same system RAM
  at the same bandwidth as the CPU while paying launch overhead on top;
  sync overhead (~10 calls/layer saved) is small next to kernel runtime,
  and the io_uring fill thread already overlapped NVMe during the old
  per-GEMV syncs. Pipelining is kept (default on): it strictly removes
  work and should pay off on an idle GPU / larger tiers. GPU still trails
  the optimized CPU path (17-19 s); the CPU path remains the mainline.

## 2026-11-17 — Session 23: HIP GEMV kernel optimization for gfx1151

- New tools/gpu-kbench.c: measures the practical device read ceiling
  (grid-stride sum kernel), a synthetic F32 GEMV (pure streaming, no
  dequant), and per-tensor achieved GB/s for GPU sync / async-dev /
  host-weight paths plus single-thread CPU, on real expert slices.
- Baseline finding: readbw ceiling ~236 GB/s, but the P1-era kernels ran
  at 3.6-21 GB/s (10-65x below). Two root causes:
  1. x was read from MAPPED HOST memory and re-read by every output row
     (6144 rows x 24KB = 147MB of uncached-ish traffic per GEMV). Fix:
     sync path stages x into device memory (g_x now hipMalloc); async
     path uses a new stream-ordered ring fatcat_gpu_xstage() - ml4 stages
     xn once per layer (shared by all gate+up GEMVs) and fa[k] per expert
     before down launches.
  2. scalar lane-strided x loads touched one cache line per lane per op.
     Fix: all 5 kernels vectorized: thread owns contiguous elems, x via
     float4, codebook rows via single u64 loads (iq1s/iq2xxs grids),
     IQ1_S remapped to one thread per 32-elem sub-block (header computed
     once per 32 elems, 4x ILP). Launcher enforces 16B alignment of x/y
     (CPU fallback otherwise).
- Achieved (kbench, sync path): IQ1_S 4 -> 51-74 GB/s, IQ2_XXS 3.6 ->
  57-86, Q8_0 10.8 -> 139-217, Q6_K 21 -> 97-142. Synthetic F32 215.
- E2E 8-token GPU best 22.9 s (was 24.6 s pre-rewrite; CPU mainline
  17-19 s). E2E on this box is dominated by NVMe streaming + contention
  with the host LLM sharing the iGPU (observer effect: runs while the
  assistant generates are 25-60 s). Numerics: corr band 0.9984..0.944,
  top-1 matches CPU at all positions; all unit tests pass.

## 2026-11-17 — Session 24: P5 v1 — DSA lightning indexer + sparse MLA attention

- Recon: indexer tensors exist on 44 "full" layers (indexer.types[]==1:
  0-16 + alternating 1,1,0,0 from 19); 9 tensors each (attn_q_b Q6_K
  6144x4096, attn_k Q8_0, proj F32, q/k LayerNorm params, SSMax).
  kv_source_map: 63 KV owners (layers 39,43..71 -> source 35). MLA KV =
  2304 B/token/owner ~ 38 GB at 256K (the long-context wall).
- Semantics (from llama.cpp mistral-large4.cpp, authoritative):
  indexer input = rmsnorm'd xn; q side = LayerNorm(xn; q_norm w+b,
  eps=layer_norm_epsilon=1e-6) -> attn_q_b -> per-head SSMax with
  lt=log(t+1) -> RoPE on first n_rot dims ([rope|nope] layout, ggml
  NORMAL rope with n_offs=0); k side = attn_k(xn) -> LayerNorm(k_norm)
  -> RoPE; score(p) = sum_h relu(q_h.k_p)*w_h, w = proj(xn) pre-scaled
  1/sqrt(nh*dk); top-k over 0..t incl. current pos; k cache written for
  every token from the start; selection dense while seq <= top_k;
  types[]==0 layers reuse the previous full layer's selection. The
  reference's Hadamard rotation is applied to BOTH q and k -> score-
  invariant (orthonormal) -> skipped in fatcat.
- Implementation (src/model/ml4.c): indexer fields in ml4_layer +
  fatcat_ml4 (ikv cache n_idx_own*cap*128 f32 = 512 B/token/layer,
  scores/sel scratch, grows with kv_cap); layernorm() + fatcat_topk_select()
  (deterministic quickselect + ascending sort); idx_select_layer();
  attention() takes struct ml4_idx (selection carried across layers,
  shared layers reuse) and does sparse MQA via m->idx_sel. Env knobs:
  FATCAT_IDX=0 disables (A/B), FATCAT_IDX_TOPK overrides top-k (dev).
- Bug fix (caught by new test): fatcat_topk_select's final qsort sorted
  the selected indices by score instead of by index -> out not ascending.
  Fixed to sort by index.
- Bug fix (pre-existing, Session 15 Bug 2 superseded): RMS norms must use
  attention.layer_norm_rms_epsilon (1e-5); attention.layer_norm_epsilon
  (1e-6) is only the indexer LayerNorm eps (llama.cpp
  mistral-large4.cpp lines 8-9; Session 15's finding was against an older
  llama.cpp checkout). fatcat now reads both keys correctly.
- New tests/test_topk.c: brute-force score-multiset comparison, ties,
  edge cases (k=0, k>=n, n=1), determinism; 5000-elem case. All 5 unit
  tests pass. gguf-inspect: BOOL array printing, arrays show up to 128.
- Validation (all CPU, 2 GB warm tier, peak RSS ~18-19 GB):
  * 8-token regression pre-eps-fix: bit-identical to eight_fc_notier.bin.
  * eps fix: top-1 unchanged at all 8 positions, cos >= 0.993 vs old
    baseline; new baseline = eight_fc_p5b.bin.
  * vs freshly rebuilt llama.cpp reference (8 tok): cos 0.970-0.9999,
    top-1 7/8 (pos 1 close call inside llama.cpp kernel noise); eps fix
    within noise band, as expected from P2c.
  * 103-token corpus, FATCAT_IDX_TOPK=64 vs FATCAT_IDX=0: all logits
    finite, cos 0.983-0.995 (sparse != dense), top-1 6/8 - sane for an
    aggressive k=64 on a 103-token seq (real top_k=2048).
- Next: long-context validation (>2048 tokens, expensive prefill), P5 v2
  KV memory (f16 KV storage halves 38 GB -> 19 GB at 256K), indexer
  scoring perf (AVX-512/GPU batched scores).

## 2026-11-17 — Session 25: long-context DSA validation; indexer-scorer perf probe (negative)

- logit-dump.c raised its hardcoded 512-token cap to a dynamic 8192 (the
  first long run silently truncated to 512, below top_k, so selection never
  fired). Side result from that truncated pair: sparse and dense outputs
  were BIT-IDENTICAL over 512 tokens -> the indexer k-cache writes do not
  perturb the dense path.
- Long-context validation (2100-token corpus from development_log.md, real
  top_k=2048, CPU, 2 GB warm tier, peak RSS ~19.8 GB):
  * positions 0..2047 bit-identical sparse vs dense (selection correctly
    inactive while seq <= top_k);
  * positions 2048..2099 all differ (selection activates at the exact
    threshold), all finite, cos 0.994..1.000 vs dense, top-1 matches 47/52.
    The 5 mismatches are close calls in the near-dense regime: at seq
    2049..2100 top-k=2048 drops only 1..52 least-relevant positions, so
    sparse ~= dense by construction (lossless-at-threshold, as intended).
  * Conclusion: DSA selection path is correct end-to-end at the real
    threshold; no crashes, memory stable.
- Indexer-scorer perf probe (reverted): built a transposed-cache AVX-512
  scorer (k stored dim-major so 16 positions vectorize contiguously, 8-head
  tiling) + tools/idx-bench.c, measured vs the REAL baseline (existing
  dot_f32-per-head, not naive scalar). Result: no win - 1.2x at n=2048,
  ~1x (slightly worse) at long n. The scorer is memory-bandwidth-bound, not
  compute-bound, so re-vectorizing the dot does not help; dot_f32 was
  already near the wall. Reverted (tree back to committed P5 v1). At the
  real operating point scoring is <1% of prefill cost. Revisit only with a
  GPU batched-scores kernel if long-context decode ever becomes scorer-bound.

## 2026-11-17 — Session 26: P5 v2 — real f16 MLA KV storage (opt-in, halves KV memory)

- Re-tested the Session 14 "f16 KV collapses corr to 0.3" finding with the
  now-correct code: f16-KV vs f32-KV on the 103-token corpus gives cos
  0.98..0.999, top-1 5/8. The 0.3 was an artifact of the buggy pre-eps-fix
  build; f16 KV is actually viable (same band as sparse-vs-dense and the
  GPU-vs-CPU accumulation noise).
- The old FATCAT_KV_F16 only EMULATED f16 numerics (rounded to f16 but
  stored f32) -> saved no memory. Replaced with REAL f16 storage: the MLA
  KV cache is now a uint16_t array when FATCAT_KV_F16=1 (kv is NULL),
  halving the dominant long-context allocation: 63 owners x 576 elems x
  4B = 2304 B/tok -> 1152 B/tok, i.e. ~38 GB -> ~19 GB at 256K.
- Two new AVX-512 F16C kernels for the read path (so the hot loop stays
  vectorized despite f16 storage): fatcat_cpu_dot_f32_f16 (score dot) and
  fatcat_cpu_kv_accum_f16 (value accumulate ac = ac*corr + w*k). Both
  unit-tested vs scalar (relerr 4.5e-6 / 4.0e-8). f32 path untouched.
- Validation (103-token corpus, CPU, peak RSS ~18.2 GB):
  * f32 path BIT-IDENTICAL to pre-change baseline (refactor is safe);
  * f16 storage active: real-f16 vs f32 cos 0.992..0.998 (f16 quant effect,
    acceptable quality for the memory win);
  * real-f16 vs old-emulation cos 0.98: both round KV to identical f16
    values, so the gap is pure dot-kernel accumulation order (4-acc vs
    2-acc) amplified chaotically over 72 layers x 103 pos - same class as
    the P4 GPU-vs-CPU band, not a semantic difference.
- Default stays f32 (bit-identical reference). FATCAT_KV_F16=1 is opt-in for
  256K deployments that need the headroom (f32 KV 38 GB + assistant 68 GB
  ~= 114 GB is tight on 128 GB with swap full; f16 gives ~95 GB).
- Next P5 memory items: hugepages (madvise, opt-in; forced compaction under
  memory pressure can stall so NOT default), NUMA/GTT carve-out sizing.

## 2026-11-17 — Session 27: P5 v3 — opt-in transparent hugepages for KV/indexer caches

- System THP is in `madvise` mode (0 anon hugepages by default), so the large
  KV/indexer caches got no hugepages. Added FATCAT_KV_HUGEPAGE=1 (opt-in):
  madvise(MADV_HUGEPAGE) on the KV (f32 or f16) and indexer k-cache after
  each growth. Cuts TLB/page-walk pressure on the long-context attention
  scan (2 MB pages vs 4 KB over a 19-38 GB working set).
- Default OFF: under memory pressure (defrag=madvise, swap full, assistant
  resident) fault-time compaction can stall, so this must be a deliberate
  choice, validated on an idle long-context run (like the GPU idle re-measure).
- Validation: FATCAT_KV_HUGEPAGE=1 output is BIT-IDENTICAL to the f32
  baseline (hugepages are numerically transparent); madvise returns 0. All
  unit tests pass. Perf benefit is context-length dependent (shows at 256K);
  deferred to a standalone idle run.

## 2026-11-18 — Session 28: P5 v4 — KV pre-allocation + system memory-tuning doc (P5 complete)

- Fixed the last 256K OOM hazard: lazy KV doubling left cap at up to 2x the
  sequence (~57 GB instead of 38.7 GB at 256K f32; +68 GB assistant > 128 GB).
  New FATCAT_KV_TARGET=<tokens> allocates the KV+indexer caches at a fixed
  capacity once (calloc is lazy: untouched capacity costs no RSS). The GGUF
  context_length (1M tokens = 154 GB f32) is deliberately NOT the default;
  without the knob the old doubling policy is unchanged.
- Growth now logs `kv: capacity N tokens, X GB (f32|f16)` for budget checks
  on real runs.
- Validation (94-token corpus, regenerated after /tmp cleanup): with
  FATCAT_KV_TARGET=4096 a single fixed allocation is made; without it the
  doubling path (65->130) is unchanged; outputs BIT-IDENTICAL between both
  paths (capacity is numerically transparent). All 5 unit tests pass.
- Wrote Manuals/io/memory-tuning.md: 256K memory budget table, GTT module
  knobs (gttsize, vm_fragment_size; HSA userptr needs no BIOS carve-out),
  NUMA (Strix Halo is single-node: binding is a no-op), THP validation
  procedure, misc vm.* settings. Manuals/README.md index updated.
- P5 is now complete (v1 indexer + sparse MQA, v2 f16 KV, v3 hugepages,
  v4 pre-allocation). Deferred to idle-box standalone runs: 256K end-to-end
  perf validation of f16 KV + hugepages together.

## 2026-11-18 — Session 29: end-to-end text-generation coherence check (passes) + 12 GB test watchdog

- New user rule: agent test RAM consumption must stay under 12 GB. Added
  tools/mem-run <limit_mb> <cmd...>: polls the child's ANONYMOUS memory
  (/proc smaps_rollup Anonymous, fallback RssAnon) every 0.2 s and SIGKILLs
  over the limit. Anonymous (not RSS/PSS) is the right metric: the first two
  attempts were killed at 12.6/13.0 GB PSS, but that was streamed model mmap
  as clean page cache, which the kernel reclaims before any OOM risk.
- fatcat-run now prints peak RSS at exit (audit parity with logit-dump).
- Coherence run: 11-token prompt ("Q: What is the capital of Australia? A:"),
  128 tokens greedy, full P5 stack (FATCAT_IDX=1 + FATCAT_KV_F16=1 +
  FATCAT_KV_TARGET=1024), CPU, warm 1024 MB, mem limit 8000 MB, under
  mem-run 12288: rc=0, peak ANON 998 MB (peak RSS 17.3 GB = mostly
  reclaimable page cache), prefill 0.34 tok/s, decode 0.41 tok/s (loaded box).
- Output is fully coherent: the model continues the few-shot Q/A format and
  answers Mars / Pacific / William Shakespeare correctly with well-formed
  structure. First answer "Sydney" (capital of Australia) is a model/quant
  quality point (IQ1_S), not an engine defect - the engine's job (coherent
  token streaming) is verified end-to-end.
- project.md agent test memory rule updated to the mem-run regime.

## 2026-11-18 — Session 30: pre-262K optimization pass — GPU+P5 validation, tuning env updated

- Corrected Manuals/io/memory-tuning.md: indexer k-cache at 256K is 5.9 GB
  (44 layers x 512 B/tok), not 2.4 GB (earlier per-group miscalc).
- Validated the GPU path with the FULL P5 stack (indexer + f16 KV + KV
  target) for the first time - P5 had only ever been tested CPU-only:
  logit-dump under mem-run 12288, warm 1024 MB, GPU on: rc=0, GPU active
  (copies=410, 14.7 GB, zero-copy path; upload_fails=13 = GPU-resident
  staging declining under assistant GPU contention, benign fallback),
  anon peak 1035 MB. GPU vs CPU logits: mean cos 0.982, min 0.849,
  top-1 93/94 - within the P4 accumulation-noise band. GPU-on is safe for
  the idle 256K run.
- Baked optimizations 1+2 into the standalone procedure (memory-tuning.md
  section 1): FATCAT_WARM_MB=65536 (biggest prefill lever when idle: every
  warm hit avoids ~2 MB NVMe traffic; budget 19.0+5.9+64 ~= 89 GB anon),
  GPU left ON, FATCAT_MEM_LIMIT_MB=0, no passive OMP on an idle box; plus
  an explicit GPU-vs-CPU A/B step for one ladder rung (settles the P4
  idle re-measure).
- Added tools/py/cmp_logits.py (row-wise cos/top-1 compare of logit dumps;
  vocab 204800) - the /tmp copy kept getting cleaned and had been
  re-derived with a wrong vocab once; now authoritative in-repo.

## 2026-11-18 — Session 31: fatcat-chat interactive utility (+ serve mode, context stats)

- New user-facing chat tool: tools/fatcat-chat (Python driver) + `fatcat-run
  --serve` (persistent-engine server: line protocol P=prefill, G=generate
  with EOS stop, I=info, R=reset, Q=quit). Model loads once; KV persists
  across turns, so each turn prefills only its own text.
- CLI: --system TEXT, --context TEXT, --context-file FILE (seeded as
  [SYSTEM_PROMPT]); -n max tokens/turn; --warm/--kv-tokens convenience
  env setters; --script FILE for non-interactive tests; --cpu.
- In-session commands (per user request): /help, /clear, /status, /exit
  (/quit alias); unknown /cmds suggest /help.
- Per-response stats line (per user request): [N tok | prefill s | decode s
  | tok/s | ctx seq/cap (%)] - context usage comes from the engine's real
  KV state via new fatcat_engine_context()/fatcat_ml4_kv_cap() accessors.
- Bug found by validation: driver broke out of the G read loop on EOS
  before consuming the GEN line; the stale line then poisoned the next P
  handshake ("server prefill failed: GEN 58"). Fixed: always drain to GEN.
  Streaming display switched from full-redraw to append-only suffix.
- Validation (scripted, under mem-run 12288, warm 1024, kv 4096): rc=0;
  /help + /status correct; 2 turns coherent; turn 2 recalled "teal" from
  turn 1 (cross-turn KV carry-over proven); ctx stats live (93/4096 ->
  174/4096); anon peak 232 MB. Speeds (0.18-0.20 tok/s) reflect assistant
  contention, not the engine.

## 2026-11-18 — Session 32: Phase A of hot-cache work - per-token expert sequence traces

- Backlog Phase A (prerequisite for stream-sim + keeping-quota): router
  decisions are now traceable per token.
  * ml4: fatcat_ml4_set_router_cb() - model-level default cb used by every
    forward path that passes none (forward, last_logits, engine, serve);
    engine pass-throughs fatcat_engine_set_router_cb + fatcat_engine_info.
  * tools/trace_seq.h: FCTS v1 format (header n_layers/n_experts/topk;
    record = token, layer, topk expert ids). ~1.4 MB per 1K tokens.
  * --trace-seq FILE on logit-dump (--out now optional) and fatcat-run
    (one-shot and serve mode: chat sessions trace too).
  * tools/py/read_trace.py: validator + stream-sim loader (load() ->
    (meta, records)); reports completeness + hotness skew.
- Sanity validation (94-token corpus, under mem-run 12288, anon 1001 MB):
  6674 records == 94 tokens x 71 MoE layers exactly; model dims read back
  correctly (72 layers, 128 experts, topk=4).
- KEY MEASUREMENT: top-10% experts carry mean 46% (30-66% range) of
  activations per layer vs 10% under uniform - strong empirical support
  for the keeping-quota hot-core design (backlog item 2).
- A2: recording a 1200-token mixed-domain trace (model docs + C source +
  Q&A) to /home/user/data/fatcat-traces/mixed1200.fcts for Phase B sim.
- All 5 unit tests pass. Next: Phase B tools/stream-sim (policy replay:
  flat-LFU vs LRU vs quota-partition keep=50/70/80/90).
- A2 result: the 1200-token mixed-domain trace was cut at 40-min shell
  timeout (my error: timeout set on a background job); the incrementally
  flushed file survived intact up to the last buffer - truncated to 977
  complete tokens and validated: 69367 records == 977x71 exactly.
  Saved at /home/user/data/fatcat-traces/mixed977.fcts (1.1 MB).
  Mixed-domain skew: top-10% experts carry mean 41% (25-59%) of
  activations - slightly lower than the single-domain corpus (46%) as
  expected with domain shifts, still 4x uniform. Phase A done.

## 2026-11-19 — Session 33: Phase B stream-sim - S3-FIFO wins; keeping-quota NOT supported by data

- tools/stream-sim: replays FCTS traces against cache policies at engine
  tier geometry (object = layer x expert ~23.5 MB, victim sampling width 16
  like stream.c). Policies: lru, fifo, s3fifo (fixed mid-impl: small-queue
  evictions must also feed the ghost queue), lfu decay sweep (16/64/256),
  quota-partition keep=80/90.
- Results on mixed977 trace (hit rate %, tier 8/16/32/64 GB):
    s3fifo   30.4  42.1  59.0  75.0   <- winner
    lru      21.1  40.5  57.1  74.5
    lfu64    17.2  34.3  54.5  74.1   <- engine today
    quota80  16.5  34.4  54.3  74.2   <- keeping-quota: no gain
  Ranking stable across trace thirds (domain segments): s3fifo > lru > lfu.
- Findings:
  1. The engine's current LFU/decay policy is WORSE than plain LRU at every
     size - frequency stickiness hurts on this workload; decay sweep shows
     more recency is better (lfu16 > lfu64 > lfu256).
  2. The keeping-quota (hot-core protection) does NOT beat flat LFU: the
     hot decile (~41-46% of traffic) still spans more objects than any
     realistic keep region, and LFU scoring already approximates protection.
     Honest negative result for the user-proposed design.
  3. S3-FIFO wins big at small tiers (+77% relative at 8 GB, +23% at 16 GB,
     +8% at 32 GB, ~tie at 64 GB where the working set nearly fits) and is
     CHEAPER to implement than today's score sampling (O(1) deques).
- Recommendation: Phase C pivots from FATCAT_WARM_KEEP to FATCAT_POLICY=
  s3fifo (default switch after e2e validation). Caveat: one trace (977
  tokens, 3 domains); more traces recommended before flipping the default.

## 2026-11-19 — Session 34: Phase C - S3-FIFO implemented in stream.c (FATCAT_POLICY)

- stream.c: S3-FIFO eviction behind FATCAT_POLICY=s3fifo (default stays
  lfu). Intrusive small FIFO (n_slots/10, one-hit-wonder filter) + main
  FIFO (freq-bit second chance) + ghost FIFO/open-addressed table of
  evicted keys; victim = queue head O(1) with busy-slot grace moves and a
  guard that falls back to the old LFU sampling if everything is pinned.
  Insert paths (get miss + prefetch) consult the ghost table; get() hits
  set the freq bit; fill failures + stale-short-entry removal unlink from
  queues. slot_t +12 B; ghost tables ~24 B/slot.
- Tests: 3 new s3fifo cases (integrity+fill, one-shot-flood absorption
  [deterministic: small queue eats cold keys, main residents untouched],
  prefetch path). ASan+UBSan+LSan clean. Fixed PRE-EXISTING flaky herd
  assertion (misses<=4 timing heuristic; old code failed 1/20 runs too)
  -> deterministic accounting check; 20/20 green. All 5 unit tests pass.
- E2E validation (100-token corpus, 1 GB tier, mem-run 12288):
  * NUMERIC TRANSPARENCY PASS: logits bit-identical lfu vs s3fifo
    (mean cos 1.000000, top1 100/100, maxdiff 0).
  * Hit rate at this tier: lfu 84.0% vs s3fifo 83.0% - no win at 1 GB.
    Consistent with the sim: at 1-4 GB tiers on mixed977 ALL policies
    score ~0-5% (working set >> tier); the s3fifo advantage measured in
    Session 33 lives at 8-32 GB, above the agent-test tier cap
    (FATCAT_WARM_MB<=1024). Engine-level A/B at 8-64 GB must run
    standalone (user-side): same command, FATCAT_WARM_MB=8192..65536.
- Decision: keep default=lfu until standalone runs at realistic tier sizes
  confirm the sim's predicted win; knob is ready and validated correct.
- Docs: src/stream/development.md (policy section + next steps),
  docs/streaming-policy.md (Policy (current)), project.md backlog item 2.

## 2026-11-19 — Session 35: FATCAT_MEM_BUDGET_GB - hard cap on the engine's own memory

- Backlog item 1 (user request 2026-11-18) implemented.
- stream: fatcat_stream_shrink(st, keep_bytes) - emergency reclaim that
  evicts resident slots (munmap payload) until used fits keep_bytes;
  victims follow the active policy (s3 queue heads / lfu sampled score),
  inflight/pinned/referenced slots skipped. Declared in stream.h.
- ml4 open: budget planner. kv_tok_bytes (KV+indexer bytes/token, all
  owners) computed once geometry is final; warm-tier open MOVED after
  layer/indexer setup so it can be planned. FATCAT_MEM_BUDGET_GB/_MB:
  KV target clamped to fit (KV wins over warm), warm tier gets
  budget - 1 GB reserve - kv_need; clamps printed to stderr.
- forward_int KV growth: refuse if new KV + reserve > budget (clear
  error, no OOM kill); shrink warm tier BEFORE the realloc copy since
  growth transiently holds old+new KV.
- Deviation from backlog design: deterministic accounting at the single
  KV-growth point instead of periodic smaps_rollup polling - KV capacity
  only changes there and the warm tier is bounded by its planned budget.
- engine: fatcat_engine_config.mem_budget_mb -> FATCAT_MEM_BUDGET_MB.
- Tests: shrink unit tests (lfu + s3fifo: payload freed, keep respected,
  pinned survives, post-shrink refills correct); ASan+UBSan clean; all 5
  unit tests pass. E2E under mem-run 12288: budget 2 GB warm clamp
  1024->1003 MB + logits BIT-IDENTICAL to unbudgeted run (top1 100/100);
  KV target clamp 100000->6403 tokens @ 2 GB (f32); refusal path at
  budget 1030 MB errors cleanly (rc=1).
- Docs: Manuals/io/memory-tuning.md new section; project.md backlog 1
  done + recent progress 28-35.

## 2026-11-19 — Session 36: standalone 18 GB GPU A/B -> S3-FIFO becomes the default policy

- First 18 GB attempt (CPU, agent-run): lfu completed (87.5% hit, 16.2 GB
  anon peak); s3fifo run killed at token 900 - page-cache growth from
  streaming with FATCAT_MEM_LIMIT_MB=0 pressured the box and the OOM
  killer took the assistant LLM; user stopped it. Lesson: never disable
  the fadvise safety (drop_experts) on this shared box; the 900 surviving
  logits were already bit-identical.
- tools/ab-policy.sh: standalone GPU A/B driver (report file, per-policy
  logs, periodic warm-tier stats every 100 tokens in logit-dump, logits
  comparison). Smoke-tested; GPU tier path requires the assistant stopped
  (hipMalloc fails otherwise -> silent mmap fallback, zero tier stats).
- RESULT (user-run, GPU, 18 GB tier, mixed1000 corpus, f16 KV):
    lfu     hit 74.6%  evictions 825277  wall 1154 s
    s3fifo  hit 77.6%  evictions 755197  wall 1019 s   <- wins all three
    logits bit-identical: top1 1000/1000, cos 1.0
  Win smaller than the expert-granularity sim predicted (+4% vs +23%
  relative) - engine slice granularity + prefetch/speculation smooth the
  gap - but consistent across hit rate, evictions (-8.5%) and wall (-8.5%).
- DECISION: flip default to S3-FIFO. stream.c: FATCAT_POLICY unset or any
  value != "lfu" -> s3fifo; FATCAT_POLICY=lfu restores legacy (it was
  ~1pt ahead at a 1 GB tier - escape hatch for tiny tiers).
- fatcat_stream_policy() getter; logit-dump prints the active policy in
  the warm-tier stats line. test_stream: legacy tests pinned to lfu
  explicitly; new default-policy + opt-out assertions. All tests pass,
  ASan/UBSan clean.
- Docs: streaming-policy.md (current policy), stream/development.md,
  project.md backlog item 2 closed. Hot-cache workstream (Phases A-C) DONE.

## 2026-11-20 — Session 37: prefill speed study vs llama.cpp-family (gufo) on Strix Halo

- Analyzed github.com/mikealanni/gufo (llama.cpp fork, Strix Halo GPU+NPU):
  630-4100 tok/s MoE prefill vs FatCat ~1 tok/s. New doc
  docs/prefill-speed-study.md with the decomposition and transfer plan.
- Root causes of the gap (in order): (1) their models fit fully in RAM
  (81 GiB on 128 GB) - zero per-token NVMe; (2) batched chunked prefill
  with per-expert grouped GEMMs amortizes weight loads across hundreds of
  tokens (we do 1 token/step, re-paying streaming+dequant every token);
  (3) they hold the fabric clock at max (pp_dpm_fclk) - free win we miss;
  (4) speculative decode (prompt-lookup is the lossless no-draft-model case).
- Ranked FatCat levers: D1 batched prefill (est. 10-30x prefill),
  D2 GPU staging fix (stage_hits=0 observed), D0 fabric clock hold (trivial),
  D3 prompt-lookup decode speculation. Non-transferring: their absolute
  numbers (4x fewer active params + full residency), NPU offload, MTP.

## 2026-11-20 — Session 38: D1 batched prefill implemented — bit-identical, ~3x at 1 GB tier

- docs/prefill-speed-study.md ranked batched prefill as the biggest lever
  (Session 37); implemented it as forward_batched() in ml4.c, opt-in via
  FATCAT_BATCH=<chunk tokens> (0 = off, sequential path untouched) and
  logit-dump --batch N.
- Design: layer-outer chunk loop. Per layer: attention per token (KV
  writes are positional), rmsnorm+router for the whole chunk (normalized
  inputs cached), token-slots grouped per expert via linked lists
  (head/nxt/g_tok/g_k), async union prefetch of the chunk's expert set
  before the shared-expert FFNs (NVMe streams during that compute), then
  ONE expert_slice load per expert per chunk serving all its tokens
  (held refs, released after the group). Per-token accumulation keeps the
  sequential order (shared first, routed in top-k order) -> bit-identity.
  v1 scope: sync GEMVs, no router speculation, no FATCAT_DUMP_HID (falls
  back to sequential).
- Validation (CPU, 1 GB warm tier, under mem-run 12288): policy100 final
  logits BIT-IDENTICAL batch 64 and 256 vs sequential (maxdiff 0.0000,
  cos 1.0, top1 match). Speed: 0.4 -> 1.23 (b64) / 1.31 (b256) tok/s
  (~3x) even at a tier far below the chunk expert union (hit 41-57% vs
  83% sequential - the union thrashes a 1 GB tier, yet amortization wins).
  Expert accesses per 100 tokens: 85k -> 17k (b256): the amortization.
- logit-dump: --batch N mode (single last_logits call, prints tok/s);
  rows_written accounting; +time.h.
- All 5 unit tests pass; build warning-free.
- Scaling sweep (CPU, 1 GB tier, mixed500): chunk 64/128/500 -> 1.23/1.30/
  1.23 tok/s; tier hit 57/41/29%. Chunk ~128 is the sweet spot when the
  tier is far below the chunk expert union; real chunk scaling needs a
  tier that fits the union (standalone GPU runs). perf-validation.sh
  gained PART 4: GPU sequential vs --batch 128/512 at 18 GB, with the
  bit-identity compare command printed. GPU smoke alongside the assistant
  LLM: every hipMalloc fails -> full CPU fallback, 0.21 tok/s (failed-
  attempt overhead) - confirms GPU tests require the standalone mode.
- STANDALONE GPU A/B RESULT (user-run, 18 GB tier, mixed500, report
  perf-validation-20261120-0324): sequential 1.0 tok/s (hit 78.3%) vs
  chunk 128 2.94 (98.4%) vs chunk 512 3.41 (100.0%, zero misses) - the
  chunk expert union fits the tier and NVMe streaming disappears.
  Logits bit-identical both chunks (maxdiff 0.0). DECISION: FATCAT_BATCH
  default flipped to 256 (0 = opt-out). fatcat_engine_prefill passes the
  whole prompt in one call, so fatcat-chat prefill batches automatically;
  fatcat-run smoke OK. GPU staging stats now copies=423 copy_GB=15.2
  (~30 MB/token, was ~7.4 GB/token in Session 36): the zero-copy wrap
  handles tier slots - D2 scope shrank to the non-resident tensors.

## 2026-11-20 — Session 38b: batched GEMM kernels (CPU + GPU) for the D1 grouped expert pass

- fatcat_cpu_gemm() (src/backend/cpu): AVX-512 IQ1_S/IQ2_XXS batched row
  kernels - each quantized row is decoded ONCE and FMA'd into up to 8
  per-token accumulators (token tiling beyond 8); per-token op order
  mirrors the gemv kernels exactly -> BIT-identical (test_cpu_kernels
  memcmp vs gemv loop, n_tok 1..9; scalar fallback same trick via
  dequant-once chunks). Other types loop gemv per token (no decode to
  amortize).
- fatcat_gpu_gemm() + k_gemm_iq1_s/k_gemm_iq2_xxs (kernels.hip): same
  design on device (codebook decode once per row, 8-token tiles, same
  block_sum per token -> bit-identical); sync staging with grow-on-demand
  device X + mapped-host Y (cap 512 MB); wrap->stage->CPU ladder in
  ml4.c expert_gemm(). Stubbed for non-HIP builds.
- forward_batched(): grouped pass now gathers token rows per expert
  (slots array), runs gate/up GEMM -> silu*mul -> down GEMM -> scatter
  when n_e >= 2; single-token experts keep the gemv path. New scratch
  buffers Xg/Yg/Yu/Zb/Yd (~5 x C x 6144 f32).
- Measured (CPU, 1 GB tier, assistant LLM competing): policy100 chunk 128
  1.23 -> 1.59 tok/s (+29%); mixed500 chunk 128 1.30 -> 1.53 (+18%).
  Final logits bit-identical both (maxdiff 0.0000). All 5 tests pass.
- GPU A/B left for standalone (hipMalloc needs the assistant stopped):
  tools/gpu-gemm-test.sh (chunk 128/512 @ 18 GB, bit-identity vs the
  p4_seq baseline from perf-validation-20261120-0324; GEMV-era refs
  2.94/3.41 tok/s).
- GPU GEMM A/B (user-run standalone, gpu-gemm-test-20261120-0406):
  chunk 128 2.92 tok/s, chunk 512 3.34 - statistically IDENTICAL to the
  GEMV-era numbers (2.94/3.41); logits bit-identical. Interpretation: on
  GPU the expert weight handling was NOT the wall (8x fewer weight reads
  changed nothing); the per-token dense ops (attention projections,
  shared FFN, router: ~10 GPU memcpy+launch+sync round-trips per layer
  per token x 71 layers) dominate GPU prefill time. CPU GEMM did win
  (+18-29%). Added gemm_calls/gemm_fails counters (stats line) to prove
  the GPU path engages vs silently falling back; next GPU lever is
  batching the per-token dense ops / cutting sync round-trips, not
  expert kernels.

## 2026-11-21 — Session 39: D4 stage 1 — batched dense per-token ops (GPU round-trip wall)

- Motivation: Session 38b GPU A/B proved expert weight handling is NOT the
  GPU prefill wall; ~700 synchronous memcpy+launch+sync round-trips per
  token are (wq/wkv_a/wo/router/shared-FFN, ~10 dense calls x 71 layers,
  one per token). D4 batches those over the chunk like D1 batched experts.
- CPU batched kernels (avx512.c): row_q8_0_nt / row_q6_k_nt - same
  construction as the IQ1_S pattern (one row decode, <=8 per-token
  accumulators fed in gemv order -> bit-identical); fatcat_cpu_gemm
  dispatch extended (F32 intentionally stays per-token gemv: no decode
  cost to amortize). Unit tests gemm_q8_0_bitid / gemm_q6_k_bitid
  (memcmp vs gemv loop, n_tok 1..9): PASS.
- GPU batched kernels (kernels.hip): k_gemm_q8_0 / k_gemm_q6_k mirror
  their gemv kernels' thread mapping + accumulation order exactly;
  launcher accepts the new types (same staging ladder, 512 MB cap).
- ml4.c: attention() split into gemv(wq) + attention_core() +
  [gemv(wkv_a)] + gemv(wout); attention_core is the sequential middle
  (ckv rmsnorm, RoPE, k-absorption, SSMax, KV publish, indexer, MQA
  scan, absorbed V) - positional, stays per-token. New gemm_res() =
  GPU gemm if weight uploaded else CPU gemm (mirrors gemv ladder).
  forward_batched(): per chunk now does batched rmsnorm -> wq GEMM ->
  wkv_a GEMM (owner layers) -> per-token attention_core -> wout GEMM;
  shared-expert FFN and the layer-0 dense FFN batched via gemm_res
  (gate/up GEMM -> silu*mul -> down GEMM). Router stays per-token.
  Round-trips/token ~700 -> ~4 at chunk 256. New scratch ~6 x C x 6144
  f32 + C x (n_ff_max=24576) f32 buffers; dense FFN n_ff=24576 >
  expert n_ff 6144 (Yg/Yu/Zb sized by max).
- Validation (CPU, 1 GB tier, assistant competing): all 5 unit tests
  green; policy100 e2e maxdiff 0.0000 vs lg-p100-seq-last baseline;
  mixed500 e2e (665 tok, shared-KV layers) maxdiff 0.0000 vs
  lg-m500-b500. Wall A/B policy100: seq 251.0 s -> batched 242.1 s
  (+3.7%: CPU was never the round-trip wall; the win is GPU-side).
  Pitfall logged: lg-p100-seq-last.bin is LAST-ROW ONLY (819200 B) -
  extract with tail -c 819200 before cmp_logits.py, else row 0 vs last
  row "mismatch" (cos 0.76) is a false alarm.
- Pending: standalone GPU A/B (user runs gpu-gemm-test.sh + fabric-clock
  hold @ 18 GB; refs 2.94/3.41 tok/s chunk 128/512); stage 2 (indexer
  gemv batching) optional.

## 2026-11-21 — Session 39b: D4 stage 1 GPU A/B — +6% prefill, bit-identity REGRESSION under investigation

- User standalone run (fabric-clock held, gpu-gemm-test-20261121-0112):
  chunk 128 3.19 tok/s (was 2.92), chunk 512 3.62 (was 3.41): +6% only,
  far from the ~2x hoped. gemm_calls=23181/53397, gemm_fails=0 -> GPU
  gemm path fully engaged (expert + dense). Interpretation: with the
  dense ops batched the remaining wall is the EXPERT gemm round-trips
  (~330 launches+syncs per layer per chunk: per-expert staging + 3 GEMMs
  each), not the dense ops; dense batching alone cannot pay for itself.
  Next lever for prefill: batch/cut the expert-side round-trips (async
  staging + multi-expert launches), or accept 3.6 as the 18 GB plateau.
- NEW PROBLEM: final logits no longer bit-identical to the sequential GPU
  baseline (cos 0.998669, top1 still matches; was cos=1.000000 pre-D4).
  CPU e2e IS bit-identical (policy100 + mixed500), so the divergence is
  GPU-only: either (a) k_gemm_q8_0/k_gemm_q6_k differ from their gemv
  twins despite line-by-line construction match, or (b) a restructure
  effect visible only on GPU. Added: gputest direct gemm-vs-gemv
  bit-identity check on real tensors (wired into gpu-gemm-test.sh),
  FATCAT_D4_OPS bitmask (bit0 attn projections, bit1 shared FFN,
  bit2 dense FFN; 0 = fully old path) to bisect on GPU.
- Escape-hatch validation: FATCAT_D4_OPS=0 CPU e2e reproduces the
  sequential baseline bit-for-bit (policy100, cos=1.000000) -> the
  restructure scaffolding is inert; GPU divergence is isolated to the
  dense gemm kernels or a GPU-only effect. Awaiting standalone run of
  the extended gpu-gemm-test.sh (gputest gemm-vs-gemv section bisects).

## 2026-11-21 — Session 39c: D4 GPU bit-identity root cause = FMA contraction drift in Q8_0 kernels

- gputest kernel-level bisect (gpu-gemm-test-20261121-0140): k_gemm_q6_k
  == k_gemv_q6_k BIT-IDENTICAL; k_gemm_q8_0 != k_gemv_q8_0 by ~1 ULP
  (relerr 5.5e-08). Cause: the gemm kernel hoists the quant decode out
  of the token loop, letting the compiler contract mul/add into FMAs
  differently than the gemv kernel -> last-bit drift, amplified through
  71 layers to cos 0.998669 on final logits (top1 still matched).
- Fix: pin the Q8_0 inner product with explicit __fmul_rn/__fmaf_rn
  (contraction-proof) in BOTH k_gemv_q8_0 and k_gemm_q8_0, same op
  order ((acc+p0)+p1)+p2)+p3 left-assoc, s = fma(d,acc,s). Q6_K needs
  no change (already bit-identical). NOTE: this shifts the GPU gemv
  baseline by ~1 ULP -> gpu-gemm-test.sh now captures a FRESH
  sequential baseline with the current build (gg-seq-$STAMP.bin)
  instead of the stale 11-20 ref, and compares last-row vs last-row.
- Session 39c cont.: run 0149 showed the fmaf-pin FIX WAS ITSELF BAD:
  this ROCm/clang build miscompiles the __fmaf_rn chain in k_gemv_q8_0
  (gemv relerr 11, sequential logits garbage, cos(seq,batched)=-0.003).
  Reverted both Q8_0 kernels to plain source; bit-identity is now
  enforced build-wide with -ffp-contract=off on kernels.hip (CMake),
  which makes gemv/gemm evaluate identically as written (the original
  1-ULP drift WAS contraction, confirmed by gputest). Awaiting rerun.
- Also fixed: test_cpu_kernels dot_f32 went FAIL because the two new
  gemm bitid cases shifted the shared RNG stream into data where an
  f32 naive reference sum has its own cancellation noise (threshold
  was data-lucky). Now: dedicated seed + double reference. ALL PASS.

## 2026-11-21 — Session 39d: D4 stage 1 CLOSED — GPU bit-identity restored

- Final standalone run (gpu-gemm-test-20261121-0219, -ffp-contract=off
  build): gputest gemm==gemv BIT-IDENTICAL for Q6_K AND Q8_0; batched
  prefill logits cos=1.000000 vs fresh sequential GPU baseline at both
  chunk 128 (3.11 tok/s) and chunk 512 (3.49). D4 stage 1 done:
  correct everywhere, +2-6% GPU prefill (3.41 -> ~3.5).
- Honest outcome vs forecast (~2x): the dense-op round-trips were NOT
  the dominant wall after all - with dense ops batched, the remaining
  time sits in the EXPERT-side per-(layer,expert) staging+launch+sync
  cycles (~330 per layer per chunk). Dense batching alone can't pay
  for itself. Next prefill lever if ever pursued: async expert staging
  / fewer syncs on the expert path (bigger surgery). Decode path
  untouched by D4 (bandwidth-bound as analyzed).
- Follow-up: gpu-gemm-test.sh gained a policy100 chunk-256-default vs
  sequential section (speed-study part 3, still unmeasured); D4 item in
  prefill-speed-study.md marked DONE with the honest +2-6% outcome and
  the -ffp-contract=off lesson. Next queued speed items: D3 prompt-
  lookup decode speculation; optional expert-side async staging.

## 2026-11-21 — Session 40: D3 prompt-lookup speculative decode implemented, lossless-verified (CPU)

- New: fatcat_ml4_forward_rows() (per-position logits via shared
  logits_row helper, same op order as last-row path), fatcat_ml4_rewind()
  (position-indexed KV => rewind is a counter move). Engine: prompt-lookup
  drafts (longest earlier suffix match, ngram 2..8, draft<=8), verification
  feeds [true_next, draft..] in ONE batched pass (D1/D4 path, bit-identical
  by construction), accepts longest argmax-matching prefix, rewinds rejects.
  Echo-contract pending buffer keeps step() API unchanged; flush() on early
  stop (EOS/n_new). FATCAT_SPEC=1 opt-in; tools/spec-decode-test.sh for GPU.
- CPU A/B (policy100, 48 tok, 1 GB tier): base / spec / spec+sequential
  fallback all emitted BYTE-IDENTICAL token sequences -> lossless verified.
- Speed on prose (policy100): base 0.101 tok/s vs spec 0.082 (-23%):
  acceptance only 1.26 tok/step (55 draft tok fed / 38 steps) - prose has
  no repetition for the lookup to exploit, verification overhead dominates.
  This is the known prompt-lookup profile; the win case is repetitive
  output. Next: repetitive-prompt A/B (prompt self-match), then GPU @ 18 GB
  where per-step round-trip overhead makes batching drafts much cheaper.
- rep200 A/B (policy100 x2 prompt): IDENTICAL again (3rd corpus), but
  acceptance only 1.12 tok/step (-14%): the model BREAKS OUT of the prompt
  repetition, so lookup drafts (the repeated pattern) get rejected. Lesson:
  prompt-lookup wins require repetitive OUTPUT, not repetitive prompt.
  Added adaptive gate: 4 consecutive fully-rejected verifications -> pause
  drafting 8 steps -> probe again (turns prose overhead near-neutral).
  Next CPU A/Bs: structured repetitive-output prompt (numbered list) +
  policy100 with gate.
- CPU D3 picture complete (all IDENTICAL, lossless):
  * replist (repetitive OUTPUT): acceptance 1.78 tok/step (27 steps/48 tok)
    but STILL -13% on CPU: tiny-batch verification on the CPU 1 GB tier
    costs ~20 s per 3-token pass (per-layer chunk overhead + drop_experts
    re-streaming), so drafts are not cheap enough to pay off.
  * policy100 with adaptive gate: 1.20 tok/step, -17% (gate rarely trips:
    prose gets partial acceptance, resetting the fail counter).
  Conclusion: CPU is the wrong venue — verification must be nearly free,
  which is exactly the GPU regime (D1: 256-token chunk = 0.3 s/tok vs
  1 s/tok decode @ 18 GB). spec-decode-test.sh extended: policy100 +
  replist, base vs spec8 vs spec16-rep, identity checks. GPU A/B decides.

## 2026-11-21 — Session 40b: D3 GPU verdict — NEGATIVE RESULT, stays opt-in off

- GPU @ 18 GB (spec-decode-test-20261121-0453): identity IDENTICAL x3
  (lossless holds on GPU too), but decode got WORSE:
  * policy100: base 0.44 tok/s -> spec8 0.24 (-45%)
  * replist:   base 0.44 -> spec8 0.17 (-61%) despite 1.78 tok/step
               spec16 0.13 (-70%)
- Root cause (the D3 premise was wrong): decode is not round-trip-bound,
  it is EXPERT-WEIGHT-STREAMING-bound. Verification does not amortize:
  m draft tokens route to ~m x topk DISTINCT experts per layer (union
  ~= sum, no reuse at m<=16 — unlike prefill's 512-token chunks where
  the union gives ~30x reuse). Measured: a verification token costs
  ~1.3 s vs 2.3 s for a decode token (only the dense/attention part
  amortizes), while EVERY fed token burns expert bytes — including
  rejected ones. Speculation therefore INCREASES weight traffic per
  emitted token; acceptance would need to beat the waste rate, and
  prompt-lookup acceptance (1.2-1.8) cannot.
- Same lesson as D4, one level deeper: in a streaming-MoE engine any
  technique that computes tokens that may be discarded is a loss,
  because the dominant cost is per-token expert bytes, not per-step
  overhead. D3 code stays (lossless, verified x7 corpora incl. GPU)
  but FATCAT_SPEC remains default OFF; it would only pay off if the
  warm tier ever held the full active set (not possible at 1T/128 GB).
- Speed workstream status: D0 done, D1 done (3.4x), D2 negligible,
  D4 stage 1 done (+2-6%), D3 negative. Remaining real lever:
  expert-side async staging / fewer syncs (reduces the constant on the
  streaming path itself), or bigger warm tiers.

## 2026-11-21 — Session 40c: first tier sweep INVALID (my script bug) + fixes

- decode-tier-sweep run 0517: 18/40/64 GB gave 0.53/0.58/0.61 tok/s —
  looked flat, but INVALID: script set FATCAT_MEM_LIMIT_MB=1 (copied from
  the prefill scripts), which trips drop_experts (mem_used > 1 MB always)
  -> every tier MISS became a full NVMe re-fault (mmap pages released
  after each use); tier itself filled (RSS 35->82 GB) but misses were
  maximally punished, and fatcat-run printed no hit-rate to reveal it.
- Fixes: sweep now uses MEM_LIMIT_MB=110000 (runaway guardrail only,
  never fires with assistant stopped); fatcat-run prints warm-tier stats
  (new fatcat_engine_stream() accessor). Re-run required before any
  tier-size conclusion. Lesson: standalone GPU scripts must NOT inherit
  the drop_experts guardrail meant for assistant-adjacent CPU runs.

## 2026-11-21 — Session 40d: VALID tier sweep — knee is BEHIND us; decode is round-trip-latency-bound

- decode-tier-sweep run 0527 (MEM_LIMIT guardrail fixed, tier stats on):
  * 18 GB: hit 91.7%, decode 0.53 tok/s
  * 40 GB: hit 92.4%, decode 0.57 tok/s
  * 64 GB: hit 93.3%, decode 0.61 tok/s
  3.5x tier buys +1.6pp hit and +15% decode. The tier-size curve is
  FLAT: at 18 GB the miss stream is already only ~8% of touches, and
  misses are NOT what decode waits on. Cache workstream: diminishing
  returns confirmed with data; 18 GB is a fine operating point.
- Arithmetic: 0.53 tok/s = 1.9 s/token, but streamed bytes at 91.7% hit
  are ~0.5 GB (0.1 s) + tier->GPU staging copies ~6 GB (0.12 s) + gemv
  compute ~5 ms. The remaining ~1.6 s/token is ~850 per-token GPU
  staging+launch+sync round-trips (284 experts x 3 gemvs, sequential
  decode path gets no D1/D4 batching). Decode is LATENCY-bound on
  per-expert round-trips, not bandwidth-bound.
- => The async-staging / multi-expert-launch work (D5) is the single
  lever that attacks BOTH decode (kill ~850 syncs/token) and prefill
  (the same wall identified in D4). Next: scope D5.
- Oddity noted: prefill slightly SLOWER at 64 GB tier (1.94) than 18 GB
  (2.73) — likely clock/noise; re-hold fabric clock before the next
  measurement session. Also today's decode 0.53 vs the ~1.0 tok/s
  recorded in perf-validation p4 — unexplained 2x, re-check in D5 A/B.

## 2026-11-21 — Session 41: D5a phase profiler for the decode path

- FATCAT_PROF=1: per-token seconds buckets in the sequential path (attn
  incl. indexer+scan, router, shared FFN, routed experts incl. waits,
  dense FFN, other). Prints via fatcat-run "prof/token:". Only fires on
  the sequential path = exactly the decode tokens. CPU sanity (1 GB tier):
  routed 6.2s + attn 2.5s + shffn 1.4s of 10.2 s/tok -> plumbing works.
- Purpose (D5): measure WHERE GPU decode's 1.9 s/token lives before
  restructuring; D3/D4 taught phase estimates can be wrong. GPU run next.
- GPU profile (policy100, 16 tok, 18 GB): attn 1.293s + routed 0.445 +
  shffn 0.059 of 1.815 s/tok -> the round-trip wall is the ATTENTION
  phase (71%), NOT the expert side (the P3v2 async pipe already hides
  expert staging). Theory: per-layer sync GPU gemvs (wq/wkv_a/wout +
  indexer) each pay memcpy+launch+blocking hipDeviceSynchronize.
- D5b: (1) hipSetDeviceFlags(hipDeviceScheduleSpin) at init
  (FATCAT_GPU_SPIN=0 opts out) - blocking sync latency is the suspect;
  (2) attention() queues wq+wkv_a as one stream-ordered batch (xstage
  once, 2 launches, 1 wait) - same kernels/order => bit-identical;
  (3) profiler split: attn = proj + core + wout. CPU sanity green.
- D5b GPU A/B: proj 0.022s (queueing worked: 2 sync gemvs/layer -> ~0),
  but total unchanged (1.82 s/tok): the wall is attention_core = 1.218s
  (17 ms/layer). wout (a sync gemv) is only 0.064s => sync overhead per
  se is NOT the story; something INSIDE attention_core costs ~17ms/layer.
- D5c: profiler split of the core: absorb (k-absorbed q, OMP CPU loop),
  scan (MQA online softmax, single-thread), wv (64 CPU gemvs, OMP),
  misc (rope/ssmax/publish/indexer). Sub-timers gated to decode-only
  (attention_core also runs in the batched prefill path; first version
  double-counted -> negative misc, fixed via sp param). CPU sanity:
  core 1.175 = absorb .475 + scan .022 + wv .432 + misc .246 ✓.
- D5d: indexer skip when t+1 <= top-k (selection would be "all" anyway;
  ist.n_sel=0 already means dense scan => bit-identical by construction).
  Kills most of the attn "misc" bucket (0.291 s/tok GPU; CPU misc 0.246->0.002).
- D5e: GPU-offload the two big CPU loops in attention_core. Probed first:
  wk_b/wv_b are Q8_0; standalone test proved dequant+dot_f32 != cpu gemv
  bit-wise (60-75% rows differ) => absorb CANNOT ride the gemv kernel.
  * k_absorb_q8_0: 1 thread/row, mirrors dequant_q8_0 + fatcat_dot_avx512
    exactly (4x16-lane fmaf accs, ((a0+a1)+(a2+a3)), _mm512_reduce_add_ps
    tree). Scalar mirror verified 0/256000 mismatches vs CPU absorb.
  * k_gemv_q8_0_batched: k_gemv_q8_0 body + per-item strides, 1 launch for
    the 64 absorbed-V gemvs; bit-identical to per-head gemv by construction.
  * wk_b/wv_b now wt_upload'd (~480 MB device total). attention_core gains
    devio flag (batched prefill passes 0: qb/ob are plain malloc there).
  * tests/gpu-absorb-test.c: kernel-level bit-identity vs the exact CPU
    reference patterns (dequant+dot_f32 / cpu gemv), random valid Q8_0.
- D5d/e GPU results (policy100, 16 tok, 18 GB tier): decode 0.55 -> 0.93
  tok/s (1.815 -> 1.053 s/tok, +69%), prefill 2.38 -> 2.60 tok/s.
  attn 1.250 -> 0.105 s/tok: misc 0.291->0.001 (indexer skip), absorb
  0.395->0.002 (GPU kernel), wv 0.454->0.040 (batched). core now 0.064
  (scan 0.021 = real KV scan). routed 0.445 -> 0.877: the CPU loops were
  hiding expert-pipe latency; routed is now 83% of decode = the true
  streamed-bytes + staging-latency cost. Next: attack the expert pipe
  (deeper in-flight window / fewer waits per layer) if worth it.
- D5e gate GREEN: absorb 0/32768 bit-mismatches vs CPU dequant+dot; wv
  batched 0/8192 vs per-head gpu gemv (max ULP 0). First FAIL was a test
  bug: fatcat_gpu_gemv handle is anchored at its base, so the per-head
  reference recomputed head 0 64x; batched kernel was right all along
  (within 3968 ULP of CPU gemv = normal GPU-vs-CPU gemv noise).
- Added FATCAT_GPU_ATT=0 gate (disables GPU absorb/wv -> CPU loops) for
  e2e A/B; tools/d5-validate.sh = kernel gate + logit-dump att1-vs-att0
  cos check on policy100.
- D5e e2e FAILED gate: cos(att1,att0)=0.915 on policy100 (top1 ok, pos0
  clean at 1.6e-5 ULP noise, pos1 already maxdiff 5.4). Standalone kernel
  tests pass => engine-context issue; bisecting with new FATCAT_GPU_ATT=2
  in-engine verify mode (GPU result vs CPU reference per layer/token,
  logs first 16 diffs with maxrel).
- FOUND+FIXED latent D5d bug: idx_select_layer also PUBLISHES the indexer
  key to ikv every token (needed once ctx > top-k); the t+1>topk skip
  punched holes in ikv for long contexts. Correct skip: kv_cap <= topk
  (cache can never exceed top-k -> dense forever). My first fix attempt
  had the condition inverted (caught by CPU misc bucket 0.002->0.235).
- ROOT CAUSE of D5e e2e FAIL found by in-engine verify (ATT=2): absorb
  kernel computed row stride as n_in/32*34=68B, but NoPE layers (L3,7,11,
  every 4th; rope off, rope slots zero-filled) store wk_b rows padded to
  the FULL qk=128 elems -> row_bytes=136. Row 0 matched, all others read
  misaligned weights (32767/32768 differ, maxrel 3e3). CPU loop was fine
  (strides by row_bytes). Fix: row_stride param through kernel/launcher/
  wrapper; standalone test now uses the padded 136B layout. wv verify
  showed only benign GPU-vs-CPU gemv noise (maxrel ~1e-3) as expected.

## 2026-11-21 — Session 41b: D5d/e ACCEPTED

- Re-validation after the NoPE stride fix: kernel gate ALL PASS (absorb
  0/32768 vs CPU incl. padded 136B rows; wv batched 0/8192 vs gpu gemv,
  max ULP 0). e2e cos(att1,att0) = 0.9915 at pos 99 (pos0 1.000000,
  gradual ULP accumulation through the KV cache), top-1 preserved.
  ACCEPTED: the project's GPU-vs-CPU bar is the P4 accumulation-noise
  band (mean cos ~0.98 accepted in P5 validation); my earlier
  cos=1.000000 expectation in d5-validate.sh was stricter than the
  engine's own precedent (GPU gemvs have never been bit-equal to CPU).
  Bit-identity holds where it must: new kernels == existing GPU kernels.
- FINAL D5d/e: decode 0.55 -> 0.93 tok/s (+69%), prefill 2.38 -> 2.60.
  attn phase 1.250 -> 0.105 s/tok. routed (0.877 s) is now 83% of decode
  = next target (D5f: expert pipe waits/in-flight depth).

## 2026-11-21 — Session 42: D5f profiler (routed split)

- D5a profiler extended with routed sub-buckets (FATCAT_PROF=1):
  slice (expert_slice staging waits), queue (gemv_expert issue: launch +
  xstage on GPU), wait (the two pipe fatcat_gpu_wait syncs), act (CPU
  silu), pre (prefetch_experts issue). Sub-ticks feed p_routed so the
  split sums exactly. CPU sanity: split sums to routed (6.349 s/tok at
  1 GB tier: slice 2.37 queue 3.96 act 0.01). GPU run will separate
  staging-latency (slice+wait) from launch overhead (queue).
- D5f GPU profile (18 GB tier, policy100, 16 tok): routed 0.895 s/tok
  splits as slice 0.795 / queue 0.003 / wait 0.093 / act 0.003. Launch
  overhead and pipe syncs are noise: the wall is expert weights not being
  resident when expert_slice() needs them. Stream code review: get() on a
  non-prefetched miss does a SYNCHRONOUS pread on the main thread (bypasses
  the io_uring worker's depth); prefetched-but-late waits and cold sync
  fills both counted as plain misses. Added stats split: wait_bytes
  (blocked on in-flight prefetch = issued too late) vs cold_bytes
  (sync fill = never issued / dropped by full tier), printed as
  "warm bytes:" line. CPU smoke: sync-fill dominates even at 1 GB tier.
- D5f stats @18 GB (GPU): misses are 100% late-prefetch (sync-fill ~0 ->
  prefetch coverage perfect, timing wrong). Tier fully saturated
  (1843/1843 slots, 27.9k evictions/16 tok). Added ring_bytes (true NVMe
  demand incl. wasted spec reads) + expert-spec accuracy print. CPU smoke
  is damning: ring-read 73 GB vs 11 GB consumed misses (6.6x waste from
  spec guesses + evict-reread), expert-spec accuracy 70%. Next: tier sweep
  18/40/64 GB with full stats (FATCAT_PROF now baked into sweep script).
- TIER SWEEP (post-D5e, GPU, policy100+24tok): decode 0.69 (18 GB) /
    1.07 (40 GB) / 1.20 tok/s (64 GB); slice 1.17/0.65/0.55 s. The old
    "18 GB knee" is OBSOLETE (measured pre-D5e when attention hid the byte
    wall): decode is NVMe-bandwidth-bound, curve still climbing at 64 GB.
    Standalone guidance stays FATCAT_WARM_MB=65536 (memory-tuning.md).
- D5f code: expert-spec accuracy only 69% -> wrong speculative prefetches
  burned NVMe bandwidth ahead of critical reads. Stream prefetch queue
  split into 2 classes: critical (current layer) drains strictly before
  speculative (next-layer guesses). prefetch() gains a spec flag; ml4.c
  marks end-of-layer guesses spec=1. test_stream updated, ALL PASS.
- D5f: priority queue A/B at 40 GB = FLAT (slice 0.649->0.637, decode
  1.07->1.08): ring depth 64 rarely backs up, ordering was not the wall.
  Kept (principled, free). Cumulative stats mix prefill's ~200 GB churn
  into every number -> added DECODE-ONLY measurement window: engine
  resets stream stats at the prefill->decode boundary and samples
  /proc/diskstats whole-disk sectors read around the decode loop
  ("decode nvme: X GB (Y GB/s)" line). Ground truth for whether decode
  slice time == NVMe saturation or exposed latency.
- D5f GROUND TRUTH (decode-only window, 40 GB tier): NVMe SATURATED at
  6.05 GB/s, 6.08 GB device reads/token vs only 1.6 GB/token of useful
  expert misses. Decomposition: ring reads 4.2 GB/tok + ~1.9 GB/tok mmap
  page-cache REFaults: stream pread fills push experts through the page
  cache (double-cached with the warm slots), evicting the hot mmap'd
  attn/shared weights which then refault. Decode is 100% bandwidth-bound;
  slice == NVMe queue wait. FIX (D5f-2): posix_fadvise(DONTNEED) on each
  slice right after its fill completes (ring + worker-pread + sync paths),
  FATCAT_FADV=0 escape hatch. CPU smoke: device reads 37.4->29.7 GB.
- D5f-3 hot-weight pinning: fadvise(DONTNEED) was FLAT (no-op: the GGUF is
  fully mmap'd, mapped pages cannot be reclaimed). O_DIRECT dead on arrival:
  GGUF data_start is 32B-aligned (general.alignment=32), never 512. Root
  cause re-read: the per-token hot set (attn + shared FFN + dense + indexer
  + router + lm head) is ~17 GB (these tensors are high-quant, row_bytes
  5-26 KB, NOT IQ1_S) -> cannot stay in page cache while 4.2 GB/token of
  expert preads churn it -> ~1.8 GB/token refaults at line rate. FIX: at
  load, memcpy the hot set into anonymous RAM (pin_wt/pin_f32, tracked in
  m->pins, charged to warm_room under MEM_BUDGET). Default ON only when
  warm tier >= 8 GB (standalone); agent tests (warm 1024) unaffected.
  FATCAT_PIN=0 escape hatch, FATCAT_PINDBG=1 per-tensor sizes.
- D5f-3 CRASH FIX: first GPU pin run aborted (glibc FORTIFY at Xg memcpy =
  heap corruption). Cause: attn_k_b/attn_v_b are 3-D tensors ([kv_lora,
  per-head, n_head]); wt_load took n_out=dims[1] (per-head rows), so
  pin_wt copied 34 KB instead of 2.2 MB and gemv read past the block.
  Fix: n_out = product of dims[1..]; wt_load now warns (FATCAT_PINDBG)
  when n_out*row_bytes != tensor n_bytes (all clean now). Verified:
  PIN=1 vs PIN=0 CPU output bit-identical; pin 16.46 GB.
- D5f pin A/B: FLAT-to-negative (0.93 vs 0.99 tok/s at 40 GB): the extra
  ~2 GB/token device traffic was NOT hot-weight refaults (pin removed the
  refault source, total unchanged; pin costs 16 GB RAM + page-cache
  pressure) -> FATCAT_PIN now default OFF (opt-in). Added per-process
  /proc/self/io probe (engine-only attribution; idle box: == diskstats).
  NEW SMOKING GUN: ring_spec_bytes counter -> speculative prefetches are
  ~74% of ring NVMe demand (13.4/18 GB CPU smoke). Fix: FATCAT_SPECP
  (default 2): speculate only top-2 experts by router weight (ids are in
  weight order); tail experts fall back to critical reads at router time
  (50 ms shffn window). Expect ring ~4.2 -> ~2.5 GB/tok at 40 GB tier.
- D5f RESULT (64 GB tier + SPECP=2): decode 1.34 tok/s (baseline 0.99 @40GB,
  1.20 @64GB pre-fix), slice 0.478, ring 2.45 GB/tok (spec share 74%->40%),
  engine-only == diskstats (no background noise). NVMe still saturated
  (5.06 GB/s x 0.733 s/tok == 3.71 GB/tok device bytes): decode is fully
  byte-bound. Remaining bytes: 2.45 ring + 1.34 non-ring (likely GPU
  staging fallback reading mmap'd experts when zero-copy wrap declines).
  Next: SPECP=0 probe at 64 GB; staging attribution via gpu stats line.
- D5f SPECP sweep @64 GB: SPECP=0: 1.34 tok/s, slice 0.461, ring 2.26
  GB/tok, device 3.44 GB/tok (4.61 GB/s, saturated). SPECP=2: 1.34 tok/s,
  slice 0.478, device 3.79 GB/tok. Speculation now neutral -> 0 saves 9%
  NVMe wear/bandwidth for free. Mystery remainder: 1.18 GB/tok non-ring
  device reads (~1.4 MB/slice ~ readahead window). Probe: FADV=0 (the
  post-fill DONTNEED may be killing readahead that later slices want).
- D5f FINAL: FADV=0 probe @64 GB: device bytes identical (82.9 vs 82.6) ->
  readahead theory dead; non-ring ~1.2 GB/tok is GPU-side migration, not
  user-attackable. FADV stays default ON. SPECP default flipped to 0
  (neutral speed, -9% NVMe bytes). memory-tuning.md updated (64 GB tier
  guidance, SPECP=0, PIN opt-in, MEM_LIMIT_MB=110000 never 0). project.md
  D5 entry: decode 0.55 -> 1.34-1.37 tok/s (+143%).
- User directive (Session 42b): standalone RAM target is 110 GB on this
  Strix Halo box (configurable, not a limit): warm tier guidance raised
  to ~96 GB short-context / ~86016 MB at 256K KV; tier sweep default now
  includes 98304. Expert-caching investigation started: does CODE routing
  concentrate enough to pin a domain expert set (prior mixed-text trace:
  cov95=74/128 flat; code may differ)? expert-profiler corpora built
  (code: fatcat C/py sources; prose: Manuals/devlog).

## 2026-11-21 — Session 42c: domain expert-caching hypothesis TESTED & REJECTED

- User idea: pin a domain (code) expert set to RAM for coding sessions.
  Measured with expert-profiler on code (C+HIP+py, 2831 tok) vs prose
  (docs, 2380 tok) GPU traces (expert-code/prose.trace, analyze_trace
  --compare): routing is domain-AGNOSTIC. Code cov50/80/95 med = 17/49/87
  experts per layer vs prose 15/46/83; distinct touched 127.3 vs 126.1
  of 128. cov80 pin-set ~= 85 GB (code) - the whole tier, no headroom;
  cov95 = 150 GB. Best-case hit @96 GB only 84% (code) / 85% (prose).
  Conclusion: NO static domain pin set exists; S3-FIFO already converges
  to the hot half (~26 GB cov50) within a session. Extends Session-13
  flat-hotness finding to the domain axis. EPFIX feature NOT built.
- Salvageable levers instead: (a) 96 GB tier (best-case 72->84%): sweep
  with the new 98304 rung measures the real decode gain; (b) optional
  warm-tier key persistence across restarts (skips cold-start churn;
  modest, revisit only if restart latency matters).

## 2026-11-21 — Session 42d: interactive chat driver (tools/py/chat.py)

- User asked to chat with the model. fatcat-run --serve protocol existed
  (P/G/I/R/Q, KV persists across turns) but the referenced driver was
  never written. Built tools/py/chat.py: spawns the engine, renders the
  official Mistral-Large-4 chat format (<s>[SYSTEM_PROMPT]..[/SYSTEM_PROMPT]
  [MODEL_SETTINGS]..[/MODEL_SETTINGS][INST]..[/INST]reply</s>) via the HF
  tokenizer, streams replies token-by-token (incremental decode), reports
  prefill/decode tok/s per turn. REPL: /reset /info /max /sys /quit.
  --ctx N sets FATCAT_KV_TARGET (context length; engine clamps to memory
  budget, /info shows real capacity); --reasoning off|low|medium|high
  controls the MODEL_SETTINGS header (off = concise replies).
  Key protocol detail: engine step-generate leaves the LAST generated token
  out of the KV (flush only drops spec pending), so the driver carries it
  and prepends it to the next turn's prefill.
- Validated against a protocol mock (real engine needs the GPU session).

## 2026-11-21 — Session 42e: chat.py first real use — reasoning split + fixes

- First real chat session (user, Italian prompts, --ctx 32000): model is
  GOOD (correct Italian, correct Python) but exposed three issues:
  1) The model ALWAYS thinks (reasoning_effort default high): raw output is
     [THINK]..[/THINK]answer; our skip_special_tokens decode stripped the
     markers and the thinking printed as plain answer text (82 tok for a
     greeting). Fix: chat.py now splits on THINK(34)/THINK_END(35) token
     ids, renders thinking dimmed, prints thinking/answer token split in
     the stats line; --reasoning default switched to `low` (model card:
     low = quick chat replies) and default --max 256->512.
  2) INFO at startup reports KV cap 0: KV allocates lazily at first
     prefill. chat.py now shows capacity after the first prefill.
  3) User noted CPU>GPU (42% vs 22%): expected, not a bug - decode is
     NVMe-bandwidth-bound (GPU waits on data ~78% of token time; engine
     has one cond-wait prefetch thread, no busy-spin). Load avg ~3.
- Measured chat perf (cold-ish tier): decode 0.92-1.03 tok/s, prefill
  0.4-0.8 tok/s (cold tier dominates early turns). reasoning=low should
  cut thinking tokens ~5-10x on chat replies.

## 2026-11-22 — Session 42f: 64GB tier benchmark = 1.41 tok/s; 96GB tier SWAPS (budget blind spots)

- decode-tier-sweep @64 GB (new defaults: SPECP=0): warm hit 77.2%
  (15794/20448), prefill 2.47 tok/s, decode 1.41 tok/s - best measured.
- 98304 MB tier aborted by user: heavy swap despite "122 GB available".
  Root cause: MEM_LIMIT_MB budget only counts tier+KV+1GB reserve; it
  does NOT count (a) hipMalloc device buffers - on Strix Halo GTT IS
  system RAM (expert slots, GEMM rings, host-mapped staging, ~6-8 GB),
  (b) GGUF page cache (mmap'd; fadvise DONTNEED no-op per D5f, grows
  unbounded over a long run), (c) rings/heap ~1-2 GB. 96+2.3+1+8+cache
  > 122 -> kernel evicts GGUF cache (refault storm) then swaps tier
  slots (anonymous) = thrash. Honest formula: tier <= avail - KV - 8 - 1.
  => short-context guidance revised 96 -> ~86 GB (matches long-ctx
  number). Retry at 86016 pending; if gain over 64 GB is small, 64
  stays default.

## 2026-11-22 — Session 42g: 86GB tier measured -> 64 GB stays the default

- decode-tier-sweep @86016 (warm page cache inherited): hit 79.5%
  (16259/20448), prefill 1.88, decode 1.43 tok/s vs 64 GB's 77.2% /
  1.41. +2.3pp hit buys +1.4% decode - within noise (64 GB run was
  tier-1 cold-cache worst case). Curve 18/40/64/86 = 0.69/1.07/1.34/
  1.43: FLATTENS at 64. Decision: 64 GB remains default operating
  point; 86 = optional max; 96 forbidden (swaps, Session 42f).
- memory-tuning.md guidance corrected accordingly (honest ceiling
  formula: avail - KV - ~9 GB for GPU GTT + rings + page-cache slack).
  decode-tier-sweep default TIERS now ends at 86016.

## 2026-11-22 — Session 42h: tools/chat-gpu.sh (one-command tuned chat)

- 32K-token batch prefill test abandoned by user (~4-5h prefill at
  1.9-2.5 tok/s: flat routing means batching can't amortize experts).
  Long-context work continues via interactive chat instead.
- chat-gpu.sh freezes the measured max-speed recipe: WARM_MB=65536
  (curve flat at 64), MEM_LIMIT=110000, KV_F16 + KV_HUGEPAGE + IDX=1
  (DSA keeps decode flat as the conversation grows to the 32K cap),
  SPECP=0 + reasoning=low already engine/chat defaults. Overridable
  via MODEL/MODEL/WARM_MB/CTX env and passthrough flags to chat.py.

## 2026-11-22 — Session 42i: chat.py mid-thinking cutoff fix

- User chat @ reasoning=medium exposed: /max 512 cutoff landed INSIDE an
  open [THINK] block (511 thinking + 0 answer). The KV stayed mid-thought,
  so EVERY later turn generated unmarked thinking-as-answer (512 tok,
  0 answer) - format poisoning, not a model bug.
- Fixes: (1) on cutoff while mode==think, carry now appends [/THINK] to
  close the block in KV + explicit warning; (2) warn when thinking ate
  the whole budget (0 answer); (3) default --max 512->1024 (medium/high
  think 500+); (4) startup no longer claims "clamped" when cap is just
  lazily unallocated. Mock-validated (cutoff mid-think -> next turn
  renders normally).
- Chat perf notes (cold-ish tier, medium): decode 0.87-1.03 tok/s,
  prefill 0.6-0.8 tok/s; consistent with cold tier + long generations.

## 2026-11-22 — Session 42j: unmarked-thinking renderer fix + serve loop guard

- Chat @ reasoning=low exposed: (a) model sometimes emits thinking with a
  BARE [/THINK] (no opener) - renderer showed it as answer and reprinted
  the prompt; thinking also drifted to Spanish (IQ1_S quirk, mitigate
  with --sys). (b) ASCII-art request fell into a greedy self-correction
  loop: 1536 tok (28 min), no answer, no exact repetition (semantic
  loop - uncatchable cheaply).
- Fixes: chat.py reclassifies a bare-[/THINK] segment as thinking (dim
  note + stats); fatcat-run --serve G now carries a generation history
  and stops on exact 24-token repetition ("GEN n LOOP" protocol flag,
  header updated); chat.py surfaces the LOOP notice. Mock-validated
  both paths; CPU smoke clean (serve G change is serve-only code).
- Chat perf this run (tier still warming): prefill 1.0-1.3, decode
  0.91-1.06 tok/s.

## 2026-11-22 — Session 42k: MIT license + repo made publish-ready

- User decision: open-source under MIT. Added LICENSE (MIT, "dario").
- Non-redistributable third-party docs (AMD PPR pdf/html/zip + extracts,
  arXiv papers, Mistral/vLLM posts) untracked via git rm --cached, kept
  on disk, added to .gitignore; Manuals/README.md + README.md license
  section document this. Mistral tokenizer/tekken/chat_template stay
  (Apache-2.0). No git remote exists yet.
- CAVEAT for publishing: the AMD docs live in git HISTORY (since
  a10b06f). Before pushing anywhere, either start the public repo from a
  fresh single commit or scrub history with git filter-repo.

## 2026-11-22 — Session 42l: public repo live + tools/publish.sh

- Public repo pushed: github.com/penguindark/Strix-Halo-FatCat (single
  snapshot commit from git archive of HEAD; clean history - no AMD docs,
  no keys). Public clone lives at /home/user/app/Strix-Halo-FatCat.
- .github_key was raw base64 without PEM armor (ssh: "error in libcrypto")
  - re-armed, verified fingerprint, backup .github_key.bak.
- LICENSE unified to MIT/penguindark in both repos (public repo's
  GitHub-created LICENSE adopted).
- tools/publish.sh: mirror dev HEAD -> public clone (wipe-except-.git,
  git archive, snapshot commit "sync: <head>", push with the ssh key).
  Guards: clean dev tree, deny-list of never-public paths, --dry-run.

## 2026-11-22 — Session 42m: author + copyright everywhere

- SPDX headers (MIT / "Dario Deledda") added to all 65 first-party source
  files: C/H/HIP/C++ (src, include, tests, tools), Python + shell tools
  (after shebang), CMakeLists.txt. Third-party artifacts (Manuals/model
  tokenizer/tekken/template) deliberately untouched.
- LICENSE copyright -> Dario Deledda; README top + License section and
  project.md header carry the author line. Build clean.

## 2026-11-25 — Session 43: H2 oracle spec-decode benchmark (tooling)

- Analysis of 6 external optimization proposals vs measured facts: H1
  (O_DIRECT unified pool) invalid (32B alignment, S42f budget math, flat
  tier curve); H4 (expert repack) low value (bandwidth-bound, not
  queue-depth-bound; priority-queue A/B was flat); H5 (GPU/IO overlap)
  already implemented (union prefetch + async GEMV, queue 0.003 s/tok);
  H6 (prefill anti-thrash) cheap A/B, small expected gain; H3 (cache-aware
  tail skip) needs FCTS v2 with router weights, ceiling ~1.4x; H2
  (multi-token spec decode) is the only structural lever on bytes/token.
- New: fatcat_engine_spec_oracle() + `fatcat-run --spec-oracle K,K,...` +
  tools/spec-oracle.sh (tuned standalone recipe). Phase A greedily
  generates N reference tokens; phase B rewinds to prompt end and refeds
  the SAME sequence single-token (B1) vs batched forward_rows M=K (B2),
  same warm-tier state -> amortization ratio R(K)=T_rows(K)/T_fwd(1) and
  oracle tok/s projection E(alpha,K)/T_rows(K). Losslessness guaranteed
  by construction (forward_rows rows are bit-identical to sequential
  decode, D3 property).
- Bug fixed during smoke: rows buffer must be kmax*n_vocab (first smoke
  double-freed with 1 row). CPU smoke (1 GB tier): R(2)=1.38, R(4)=1.61
  -> amortization real even cold; GPU verdict pending standalone run.

## 2026-11-25 — Session 43b: H2 oracle verdict — NEGATIVE, spec-decode archived

- Standalone GPU oracle run (474-tok prompt, 24 ref tok, 64 GB tier):
  baseline 1.109 s/fwd (0.90 tok/s, cold-ish tier + prefill churn; bench
  warm = 1.41). R(K)=T_rows(K)/T_fwd(1): K2 3.67 / K3 5.26 / K4 6.16 /
  K6 7.08. GB/fwd scales LINEARLY with K (3.72 -> 15.72 at K4, 4.2x).
- Root cause: forward_batched dedups expert loads per union, but reuse
  needs many tokens per chunk (measured ~30x at C=256); at verify-sized
  chunks (C=K<=6) 4 tokens x top-4 collide on ~1-2 experts/layer only ->
  union ~= 3x single-token bytes, and the sync-GEMV batched path (no GPU
  async pipe) adds time on top: R > K. Oracle (acceptance=100%) is
  SLOWER than plain decode at every K: even a perfect draft model cannot
  win on this engine as built.
- Even a hypothetical dedup+async verify path caps at ~E/union-ratio:
  K4 perfect -> ~1.6x; realistic alpha<=0.7 drafts -> ~1.2x. Not worth
  building (no strong small draft exists for Mistral-Large-4 either).
- DECISION: H2 rejected (mirrors D3). Speed work backlog: H3 (FCTS v2
  router weights + offline tail-skip sim, CPU-only) then H6 A/B.

## 2026-11-25 — Session 43c: H3 tooling built (FCTS v2 + skip-sim + engine FATCAT_SKIP)

- FCTS v2: trace_seq records now carry the normalized router weights
  (read_trace.py reads v1+v2; stream-sim adapted). New tools/skip-sim.py:
  replays v2 traces against the engine-mirror S3-FIFO tier and sweeps
  skip thresholds -> cold bytes saved vs router weight dropped.
- Engine: FATCAT_SKIP=<tau> (default 0=off) cache-aware tail-skip in the
  decode MoE path: cold-miss expert with w_k < tau*w_1 is dropped (no
  prefetch, no slice), weights renormalized; hits never dropped; prefill
  untouched. fatcat_stream_resident() probe added (no stats, no ref).
  Counters -> "tail-skip:" stats line.
- CPU smoke: SKIP=0.15 vs off -> identical 4-token output at 1 GB tier
  (21 experts dropped, avg weight 0.052); v2 trace parses clean.
- PRELIMINARY on the 113-token smoke trace @64 GB model tier: hit 79.9%,
  cold 1.34 GB/tok; tau=0.15 saves only 1.7% of cold bytes for 0.20%
  weight drop; tau=0.30 saves 4.9% for 1.06%. Cold bytes are only ~40%
  of device traffic -> H3 speed ceiling is small; needs a real-length
  trace before any tau decision. GPU capture instructions delivered.

## 2026-11-25 — Session 43d: H3 verdict — NEGATIVE, tail-skip archived (flag stays, default off)

- Real GPU-captured FCTS v2 trace (474-tok prompt + 64 decode tok, 64 GB
  tier): skip-sim @64 GB -> baseline hit 87.6%, cold only 0.83 GB/tok.
  Threshold sweep: tau=0.15 saves 0.7% of cold bytes (0.08% weight drop);
  tau=0.30 saves 2.6% (0.49% drop). Device traffic measured 3.4-3.7
  GB/tok, so cold expert bytes are ~25% of NVMe demand: even a perfect
  skip cannot move decode >1-2%. Quality price is real, speed gain is
  noise -> H3 REJECTED; FATCAT_SKIP remains opt-in (default 0) as a
  measurement tool only. No tau adopted.
- Why the hypothesis failed: at 64 GB the S3-FIFO tier already absorbs
  nearly all hot experts (hit 87.6%); what remains cold is dominated by
  first-touch/long-tail experts whose router weights are NOT small
  (that is why they got into top-4 at all) -> the cold-and-low-weight
  intersection the policy needs is tiny.
- Speed-work scoreboard: H1/H4/H5 invalid-or-done, H2 rejected (S43b),
  H3 rejected (this). Open backlog: H6 prefill no-promote A/B (hours,
  expected +1-3%), D4 stage 2 indexer batching (optional).

## 2026-11-26 — Session 43e: H6 prefill anti-thrash implemented (FATCAT_NOPROMO)

- stream.c: prefill admission gate. fatcat_stream_set_prefill() marks the
  batched-prefill window (ml4.c wraps forward_batched); while set, tier
  misses follow FATCAT_NOPROMO: 1 = never promote (caller uses mmap view,
  no tier slot consumed), 2 = promote only on a 2nd reference within the
  same prefill (64k-entry seen table, cleared per prefill). Default 0 =
  today's behavior. prefill_skips counter -> stats line. test_stream PASS.
- CPU smoke (1 GB tier): gate fires (10767 misses not promoted), decode
  sane; prefill itself slowed at the tiny tier (prefill reuses its own
  experts within chunks -> policy 2 may be the better trade). GPU A/B
  pending: baseline vs NOPROMO=1 vs NOPROMO=2, watch decode tok/s +
  decode hit% (stats reset at prefill->decode boundary) + prefill tok/s.

## 2026-11-26 — Session 43f: H6 A/B — ACCEPTED, FATCAT_NOPROMO=2 is the new default

- Standalone GPU A/B (474-tok prompt, 48 decode tok, 64 GB tier):
  base: hit 73.7% prefill 2.66 decode 0.85 tok/s
  NP=1: hit 74.3% prefill 2.79 decode 0.88 (+3.5%)
  NP=2: hit 74.3% prefill 2.90 decode 0.90 (+5.9%)
- NP=2 (promote prefill misses only on 2nd reference) wins on BOTH axes:
  decode +5.9%, prefill +9.0% (fewer one-shot promotions = fewer evictions
  of reused experts, cheaper fills). NP=1 strictly worse than NP=2:
  prefill does reuse its own experts within chunks.
- Caveat: single prompt, run-to-run decode variance is real (0.85-0.93
  seen for near-identical baselines); but both NP variants beat base and
  NP=2 >= NP=1 consistently, and prefill gain is large vs typical noise.
  Default flipped to 2 (FATCAT_NOPROMO=0 restores old behavior).
- Speed-work scoreboard FINAL: H6 accepted (+~4-6%), H1/H3/H4 rejected,
  H2 rejected, H5 already done. 64 GB tier stays the operating point.

## 2026-11-26 — Session 43g: static expert pruning (H7) — shelved, curve documented

- User proposal H7: profile the router (ml4 router_topk = softmax over 128
  experts/layer) offline, fix a static admit-list; low-weight experts are
  never loaded, never cached, their contribution dropped (weights
  renormalized). Static sibling of H3.
- Measured offline on the real 537-tok FCTS v2 trace (weights): keep-set
  size vs retained router mass per layer:
    keep 16 -> 55.5% | 23 -> 64.7% | 41 (~tier capacity: 64 GB / 71 MoE
    layers / 23.5 MB = ~41 experts/layer) -> ~80% | 64 -> 90.9% |
    96 -> 98.3% | 128 -> 100%.
- Verdict: routing mass is NOT compressible (S42c flatness from a new
  angle). Staying under ~2% mass loss needs ~96/128 experts = ~160 GB
  tier (2.5x available); at the 41/layer the tier can hold anyway, ~20%
  of FFN mass per token would be dropped - grossly visible on IQ1_S.
  And a static list can at best match what S3-FIFO already does
  dynamically (it selects the hot set per-interval, not on average).
- The only version that could work is model-side pruning WITH retraining
  (out of scope for an inference engine). SHELVED, not rejected outright:
  revisit if a smaller/pruned Mistral-Large-4 variant ever ships.

## 2026-11-26 — Session 44: Pista A investigation begins (staging audit counters)

- Literature sweep (SP-MoE 2510.10302, MoE-Lightning, AdapMoE, Specter,
  llama.cpp RFC #28248): most external ideas already tested/done here;
  RFC #28248 independently found our H6 prefill-thrash issue (+84% decode
  for them). New attackable lead = the D5f "mystery remainder": ~1.18
  GB/tok non-ring device reads attributed to GPU-side staging.
- Mechanism hypothesis: gemv_expert ladder = tier zero-copy (0 bytes) ->
  stage() fallback hipMemcpy from the GGUF mmap view -> CPU. expert_slice
  silently falls back to mmap when fatcat_stream_get returns NULL (no
  evictable slot / fill failure). Since D5f-2 fadvise(DONTNEED)s the page
  right after the ring fill, a staged read of those pages RE-READS NVMe:
  same bytes twice (ring + page fault). 88 layers x 4 experts x 3 slices
  x ~1 MB ~= 1.05-1.2 GB/tok == the mystery remainder.
- Instrumented (this commit): ml4 counters slice_mmap_n/bytes (tier could
  not serve -> mmap fallback) + stage_n/bytes (hipMemcpy'd into GPU slots)
  -> fatcat_engine_stage_stats -> "staging audit:" line in fatcat-run
  (also prints fatcat_gpu_stats copies/stage-hits). CPU smoke: counters
  live (10767 fallback slices at 1 GB tier, 0 GPU stage as expected).
- GPU verdict run pending: if gpu-stage bytes ~= 1.2 GB/tok the double-read
  is real -> fix candidates: (a) never fall back: wait for a slot instead
  of returning NULL when all slots busy; (b) skip fadvise for slices the
  GPU will stage; (c) enlarge staging slot ring. If ~0, mystery is
  elsewhere (re-open probe).

## 2026-11-26 — Session 44b: Pista A audit — staging theory DEAD, new suspect

- GPU audit run (48 tok, 64 GB tier): mmap-fallback 0, gpu-stage 0,
  gpu-copies 567/15.72 GB (resident uploads). The stage() double-read
  hypothesis is FALSE: the tier serves every decode slice.
- Reframed the mystery: ring submitted 122.42 GB but device did 191.14 GB
  in the decode window -> 68.7 GB (1.43 GB/tok) read OUTSIDE read()/
  io_uring accounting. Only mechanism left: mmap page REFaults (kernel
  reads pages for the GPU's zero-copy reads of the resident GGUF mmap
  views: attn/shared/dense/router weights, ~17 GB hot set squeezed out by
  the 65.5 GB pinned tier + KV + GTT). Matches D5f's 1.34 GB/tok.
- Added discriminator: decode-window majflt counter ("decode majflt:"
  line, /proc/self/stat field 12 x 4KB). If majflt ~= non-ring bytes,
  refault confirmed -> fix = pin the hot mmap set (FATCAT_PIN redux at 64
  GB, now with the staging audit to prove it) or madvise(WILL_NEED) the
  hot ranges. If majflt ~= 0, the reader is elsewhere (drive readahead?).

## 2026-11-26 — Session 44c: majflt parser was broken; refault suspect revived

- The "decode majflt: 0.00" verdict was a PARSER BUG: this kernel's
  /proc/<pid>/stat token order after comm does not match the proc(5) man
  page (verified empirically with getrusage(2) as ground truth: minflt is
  the 8th token after ')', majflt the 9th; man-page order reads a
  different field entirely). Fixed + validated: CPU smoke now reports
  0.75 GB decode-window refaults (plausible for the 1 GB-tier mmap
  fallback path).
- Also parsed the GGUF shards: expert tensors are stacked [6144,6144,128]
  IQ1_S/IQ2_XXS (~10 MB/expert slice); most tensor offsets are NOT
  512-aligned -> O_DIRECT stream path is mostly inactive (buffered reads
  + fadvise). Readahead math: ~10 MB random reads drag <=128 KB extra
  each -> readahead cannot explain the 68 GB non-ring remainder.
- Remaining suspects for the 1.41 GB/tok non-ring reader: (a) mmap
  refaults (now measurable with the FIXED parser), (b) unknown read site.
  Re-run the GPU audit: if majflt ~= non-ring bytes, refault confirmed.

## 2026-11-26 — Session 44d: refault DEAD (fixed parser); per-thread io sampler

- Re-run with FIXED majflt parser: decode-window refaults = 0.01 GB.
  mmap refault hypothesis is dead: the 68 GB non-ring reads are NOT page
  refaults, NOT staging, NOT sync-fills, NOT the pread fallback (ring
  alive), NOT gguf_pread (no callers) -> yet /proc/self/io read_bytes ==
  diskstats == 190 GB, i.e. charged to OUR process. Accounting paradox:
  ring submitted 122 GB; 68 GB came from read-family activity we cannot
  point at in code.
- New tool tools/io_sampler.py: per-task (pid+tid) read_bytes deltas for
  the WHOLE system during the run (validated: attributes 6 GB/s dd to
  its thread). Discriminates: fatcat threads (hidden read site),
  iou-wrk-* (io_uring io-wq accounting: ring bytes charged to worker,
  ring_bytes vs read_bytes measure different tasks), kworker (kernel
  reader: reclaim/readahead oddity).

## 2026-11-26 — Session 44e: PISTA A SOLVED — 4 MB kernel readahead was the sink

- io_sampler.py verdict: ONE thread (the fill worker) does ALL device
  reads at 3.4 GB/s; no iou-wrk, no kworker. So the non-ring bytes were
  ring reads amplified kernel-side.
- Root cause: /sys/block/nvme0n1/queue/read_ahead_kb = 4096 (4 MB, not
  the 128 KB default). Ring slices (~7.8 MB avg) are random offsets on a
  buffered fd (tensor offsets mostly not 512-aligned -> O_DIRECT off),
  so the kernel drags up to 4 MB of readahead after EVERY read:
  122.42 GB / ~7.8 MB ~= 15.7k reads x 4 MB ~= 62.8 GB == the observed
  62.9 GB non-ring remainder EXACTLY (and D5f's 1.18-1.34 GB/tok).
  The readahead is pure waste: adjacent file bytes belong to experts
  that are not routed next, and fadvise(DONTNEED) only covers the slice.
- Fix: fatcat_gguf_resolve() now hands the stream a dedicated fd opened
  with POSIX_FADV_RANDOM (fd_stream, lazily opened per part; mmap keeps
  default readahead so model load stays fast; O_DIRECT path unaffected).
  FATCAT_RAND=0 restores old behaviour for A/B. CPU smoke: non-ring
  device bytes fell 15.77 -> 5.00 GB; test_stream ALL PASS.
- Expected GPU effect: decode-window device reads 185 -> ~123 GB (-34%);
  NVMe was the bottleneck (3.4 GB/s saturated) -> decode up ~+30%.

## 2026-11-26 — Session 44f: Pista A fix validated — accounting perfect, decode unchanged

- FADV_RANDOM stream fd A/B (same 48-tok standalone GPU recipe):
  decode-window device reads 185.32 -> 122.31 GB (ring declared 122.42:
  accounting now closes to 0.1%). NVMe 3.34 -> 2.19 GB/s. Prefill 2.76
  tok/s (>= control). Decode 0.86 vs 0.87 tok/s: UNCHANGED.
- Honest conclusion: the 63 GB/tok-window of readahead was REAL waste
  (device wear, RAM pressure, prefill tax) but NOT the decode limiter.
  The drive had headroom; demand was stall-limited, not bandwidth-
  limited: late-prefetch == miss bytes means EVERY tier miss blocked
  the producer on its fill. Decode time == ring delivery time only
  because reads are issued on-demand, serial behind stalls.
- Reframes the speed story: bottleneck = prefetch lead-time/coverage
  (expert-spec guesses 65.3% used). Pista B (prefetch earlier/deeper)
  is the next lever; "NVMe-saturated, needs hardware" from S43f is
  RETRACTED — there is ~1.2 GB/s of drive headroom to exploit.

## 2026-11-26 — Session 45: Pista B — early speculative expert prefetch

- Code study: the next-layer router guess is computed right after each
  attention (FATCAT_SPEC=1, always on, accuracy 65.3% measured), but the
  speculative PREFETCH was off by default (FATCAT_SPECP=0, decided in the
  readahead era when bandwidth looked scarce) and, when on, was issued
  at END-of-layer (lead time = next attention only). Every tier miss
  stalls the producer (late-prefetch == miss): the decode lever is
  prefetch LEAD TIME, not bandwidth (Pista A proved ~1.2 GB/s headroom).
- Changes: (1) FATCAT_SPECE=1 (new default): issue the spec prefetch at
  the guess point -> lead time = this layer's FFN+GEMVs + next
  attention; legacy end-of-layer issue kept as FATCAT_SPECE=0.
  (2) stream worker caps speculative fills to 8 per batch (critical
  uncapped, merged critical-first): an early-issued deep guess can never
  delay a critical read by more than ~8 slice reads.
- Default behavior unchanged (SPECP=0 -> depth 0). A/B pending on GPU:
  SPECP=4 (early) vs SPECP=4 SPECE=0 (late) vs control SPECP=0
  (already measured: 122.31 GB device, 0.86 tok/s).

## 2026-11-26 — Session 45b: Pista B A/B — speculation ON +14%, late issue beats early

- Standalone GPU A/B (48 tok, 64 GB tier; control SPECP=0: 122.31 GB,
  0.86 tok/s):
  * SPECP=4 EARLY (issue at guess): ring 178.51 GB (spec 130.06),
    miss 50.26 GB, decode 0.90 tok/s (+4.7%), prefill 2.84.
  * SPECP=4 LATE (end-of-layer): ring 178.42 GB (spec 130.06),
    miss 66.04 GB, decode 0.98 tok/s (+14.0%), prefill 2.55.
- Counter-intuitive but structural: EARLY issue queues the guess BEFORE
  the current layer's critical reads; an idle worker starts a spec batch
  (cap 8 ~= 19 ms) and the critical reads that arrive behind it stall the
  GEMV pipeline every layer. LATE issue rides the guess behind critical
  in the same worker cycle: spec fills land during this layer's GEMVs,
  exactly when the next layer will need them. spec_early default flipped
  to 0 (FATCAT_SPECE=1 kept as A/B knob). Spec batch cap kept: it bounds
  the late-path worst case too.
- Cost: +56 GB ring per 48-tok window (~+1.17 GB/tok, 73% of ring is
  speculative) for +14% decode; device 3.62 GB/s still under the ~4.6
  measured ceiling. Default SPECP flip pending a confirmation re-run +
  SPECP=2 probe (half the bytes, maybe most of the gain).

## 2026-11-26 — Session 45c: Pista B CLOSED — FATCAT_SPECP=2 is the new default

- Confirmation runs (48 tok, 64 GB tier, late issue):
  * SPECP=4: decode 0.99 tok/s (replicates 0.98), ring 178.4 GB.
  * SPECP=2: decode 1.02 tok/s (+18.6% vs 0.86 control), ring 136.0 GB
    (spec 55.6 GB), device 2.89 GB/s — comfortable margin under the
    ~4.6 GB/s ceiling.
- Depth 2 BEATS depth 4 on speed AND costs 24% fewer bytes: deeper
  guesses pollute the tier (S3-FIFO demotion lag) and the marginal
  guess accuracy (65.3% top-4) does not pay for the interference.
- Default flipped: FATCAT_SPECP default 0 -> 2. README + project.md
  updated. Combined with Pista A (-34% dead device bytes) and H6, the
  474-tok standalone benchmark decode moved 0.86 -> 1.02 tok/s (+19%)
  this week; the "speed CLOSED, needs hardware" verdict from S43f is
  formally retracted — two software levers were still on the table.

## 2026-11-26 — Session 46: Specter-style proxy predictor REJECTED (offline study)

- Backlog item 2 was a learned routing predictor to beat the stale-router
  guess (65.3%). Studied offline first (CPU, free) with
  tools/py/predict-study.py on two real FCTS traces (h3-decode 537 tok,
  mixed977 977 tok): train an inter-layer affinity table
  W[(L-1,e)][(L,f)] += co-route count on prefill tokens, then predict
  layer L's experts from layer L-1's actual ids on decode tokens
  (no leakage).
- Result: affinity predictor = 23-25% engine-hit at every train/eval
  split and alpha (freq prior adds nothing); global-frequency 17-20%;
  shipped stale-router 65.3%. The proxy is 2.6x WORSE than what we ship.
- Why: the shipped guess reuses the REAL next-layer router on a nearly-
  fresh hidden state (only the current FFN delta missing), so it already
  captures token-specific routing content. Pure structural co-occurrence
  (which expert tends to follow which) is a much weaker signal — it
  cannot see the token. Specter's proxy predicts from EARLIER layers
  because it has no router access; we do, so the proxy is strictly
  dominated here.
- Verdict: no implementation, no GPU run. Saves a full build+measure
  cycle. The decode lever now sits at prefetch LEAD TIME (Pista B done)
  and hardware (Pista D second NVMe). Tool kept in tools/py/ for the
  record.
