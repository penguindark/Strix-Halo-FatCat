# Running the expert profiler (full model, GPU)

The profiler streams the full Mistral-Large-4 model from NVMe and records
which routed experts each token activates. It needs the whole machine's RAM,
so **shut down the assistant LLM first**, then run this single command in a
terminal:

```bash
/home/user/app/fatcat/run_profile.sh
```

(the script wraps the profiler command below and tees everything to the log)

```bash
cd /home/user/app/fatcat && stdbuf -oL ./build/expert-profiler \
  /home/user/data/models/Mistral-Large-4-1T-A52B-UD-IQ1_S/UD-IQ1_S/Mistral-Large-4-1T-A52B-UD-IQ1_S-00001-of-00006.gguf \
  --tokens /home/user/data/prompts/profile.tokens \
  --trace  /home/user/data/prompts/full-gpu.trace \
  2>&1 | tee /home/user/data/prompts/profile-gpu.log
```

949 tokens are processed. Watch the live progress lines — each one shows
percent done, elapsed, **eta** and tokens/s:

```
[  123/949]  13.0%  elapsed 00:04:10  eta 00:28:05  0.49 tok/s  health=ok  mem=47034MB
```

`mem=` is **total system memory in use** (MemTotal − MemAvailable), which
counts everything that matters: GPU (unified) allocations, page cache, other
processes. It stays flat by design: streamed experts are read with parallel
`pread()` into a small reusable 240 MB buffer pool (never mapped into the
process), the mmap copies of fixed weights are dropped right after their GPU
upload, and if system usage ever passes **80 GB** each expert's page-cache
range is released after use. Measured: flat ~47 GB for the whole run, no
thrashing. Override the cap with e.g. `FATCAT_MEM_LIMIT_MB=70000`
(0 disables it).

## How to tell the run is healthy

| Log line                        | Healthy value                          |
|-------------------------------|------------------------------------------|
| `gpu: ...`                    | `gfx1151 (20 CUs, ...)` (not an error)   |
| `selftest: ...`               | 4 lines `ok` + `selftest: ALL KERNELS OK` |
| progress lines                | `health=ok` on every token               |
| `gpu stats: ... upload_fails=` | `0` ideal; >0 = some weights ran on CPU (valid but slower) |
| final line                    | `DONE rc=0 total_tokens=949 ...`         |

If the selftest fails the tool aborts immediately (`rc=3`) — nothing to wait
for. If any token produces NaN/Inf the run still completes but ends with
`health: ... -> FAIL` and `DONE rc=4`.

A partial trace is written every 64 tokens to `full-gpu.trace`, so **Ctrl-C
is safe** — data collected so far stays usable.

## After it finishes

Restart the assistant. The files it will read back:

- `/home/user/data/prompts/profile-gpu.log` — full console log (header,
  selftest, progress, hotness table, `DONE` marker)
- `/home/user/data/prompts/full-gpu.trace` — binary per-layer expert
  hotness (FCT1 format)
