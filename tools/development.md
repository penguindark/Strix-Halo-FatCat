# tools — development.md

## Goal
Developer tooling: GGUF inspector, expert-usage profiler (records which of the
128 routed experts fire per layer over real prompts), and the offline streaming
policy simulator that evaluates eviction/promotion strategies on recorded
router traces before touching the engine.

## Status
**P0 scaffold + `gguf-inspect` + `gguf-slice` + `expert-profiler` +
`analyze_trace.py` DONE (P1, complete 2026-11-02).**
- `expert-profiler.c`: full-model GPU-accelerated router-trace recorder.
  Startup GPU-vs-CPU kernel selftest (aborts rc=3), per-token progress line
  (pct/elapsed/ETA/tok·s⁻¹/`health=`/`mem=`), NaN/Inf health scan per token,
  incremental trace dump every 64 tokens (Ctrl-C safe), hotness table +
  coverage stats (cov50/80/95/99 per layer). Trace format **FCT1**: header +
  f64 weighted[nl][ne] + u64 raw[nl][ne]. Full 949-token run: 0.49 tok/s,
  flat ~18 GB, rc=0. `--check` dumps full logits for llama.cpp comparison
  (`tools/llama-ref.cpp` is the reference harness).
- `analyze_trace.py`: parses FCT1, computes per-layer coverage curves and
  greedy weight/byte warm-tier budget sweeps → feeds `docs/streaming-policy.md`.
- `gguf-inspect.c`: dumps header, KV metadata (`--kv`), tensor directory
  (`--tensors`), type histogram and routed-expert analysis (`_exps.` tensors,
  GB/layer); works on single and split GGUFs.
- `gguf-slice.c`: extracts `blk.0..blk.(N-1)` + `token_embd`/`output`/
  `output_norm` from a (split) GGUF into a new single-file GGUF. Rewrites
  `<arch>.block_count`, truncates every per-layer array KV (length ==
  original block_count: `rope_pattern`, `kv_source_map`, `indexer.types`),
  drops `split.*` bookkeeping KVs, adds `fatcat.slice.*` provenance, and
  self-checks by reopening the output with our reader. Options:
  `--keep SUBSTR`, `--no-tokenizer`, `--align N`.
  Produced `ML4-IQ1_S-layers0-2.gguf` (77 tensors, 10.61 GB) at
  `/home/user/data/models/fatcat-slices/`.
  Gotcha found: output must drop `split.count` from copied KV or the reader
  treats the single-file slice as a broken split model.

## Scope
- In: `gguf-inspect` (dump header/tensor directory) ✅, **`gguf-slice (extract a
  few layers into a small GGUF for shared-machine development tests)** ✅,
  `expert-profiler` (router-trace recorder), `stream-sim` (policy simulator
  over traces), `gpu-kbench` (HIP GEMV microbenchmark: read-bandwidth ceiling,
  synthetic F32 GEMV, per-tensor GPU sync/async/host-weight vs CPU GB/s) ✅,
  `mem-run` (hard memory watchdog: polls the child's anonymous memory and
  SIGKILLs over the limit; resident file pages excluded because clean page
  cache is reclaimable and does not threaten OOM) ✅,
  `fatcat-chat` (interactive chat REPL: Python driver + `fatcat-run --serve`
  persistent-engine server mode; chat template, `--system`/`--context[-file]`
  seeding, per-turn prefill with KV carried across turns, EOS stop,
  `--script` for non-interactive tests) ✅,
  `trace_seq.h` + `py/read_trace.py` (FCTS v1/v2 per-token expert sequence
  traces via `--trace-seq` on logit-dump/fatcat-run; v2 adds normalized
  router weights for the H3 tail-skip study; validator + sim loader) ✅,
  `skip-sim.py` (H3: replays FCTS v2 traces against the S3-FIFO tier model
  and sweeps FATCAT_SKIP thresholds — cold bytes saved vs router weight
  dropped, to pick tau before touching the engine) ✅,
  `spec-oracle.sh` (H2: batched-verify cost measurement, verdict archived
  in development_log S43b) ✅,
  `io_sampler.py` (Pista A: per-task read_bytes deltas system-wide during a
  run — attributes device reads to the exact thread; proved the fill
  worker alone reads at line rate, no io-wq/kworker) ✅,
  `py/predict-study.py` (Specter-style proxy study: trains an inter-layer
  affinity table on FCTS traces; verdict 23-25% vs shipped 65.3%, rejected,
  S46) ✅, `py/chat-prompt.py` (renders a multi-turn conversation JSON into
  the Mistral-Large-4 chat token stream for offline A/B) +
  `data/chat-real.json` (630-tok real-chat A/B prompt) + `chat-ab.sh`
  (external-validity A/B: current defaults vs pre-Pista on real chat,
  +16.3% decode confirmed, S48) ✅.
  Scripts allowed here (Python acceptable for analysis-only
  tools; engine itself stays C). Later: GGUF repacker for per-layer/tier mixed
  quantization (deferred step).
- Out: weight conversion (GGUF quants come from upstream quantizers).

## Next steps
1. `stream-sim` MVP: replay **FCTS v1 sequence traces** (extension now DONE:
   `--trace-seq` on logit-dump/fatcat-run, writer `trace_seq.h`, loader
   `py/read_trace.py::load`; recorded traces at
   `/home/user/data/fatcat-traces/`) against LRU / flat-LFU / quota-partition
   policies to tune the keeping-quota (`FATCAT_WARM_KEEP`, backlog) before
   touching `stream.c`.
2. Optional GGUF repacker for per-layer/tier mixed quantization (deferred).
