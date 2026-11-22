# Strix Halo memory tuning for FatCat (256K-context target)

Notes gathered during P5 (sessions 25–28). English only, per repo rules.
Engine-side knobs are implemented; the system-side items here are host
configuration + validation procedures, not code.

## 1. Memory budget at 256K context

Fixed model facts (Mistral-Large-4-1T-A52B GGUF): 72 layers, MLA KV
512 + 64 rope elems = 576 per position, 63 KV-owning layers (KV sharing),
DSA indexer k-cache 32 heads x 16B = 512 B/token on 44 indexer layers.

| Component | f32 KV | f16 KV (`FATCAT_KV_F16=1`) |
|---|---|---|
| MLA KV @ 262144 tok (63 owners x 2304/1152 B) | 38.1 GB | 19.0 GB |
| DSA indexer k-cache @ 262144 tok (44 layers x 512 B) | 5.9 GB | 5.9 GB |
| Warm expert tier (`FATCAT_WARM_MB`) | tunable | tunable |
| Assistant LLM resident on this box | ~68 GB | ~68 GB |

With `FATCAT_KV_TARGET=262144` the KV+indexer caches are allocated once at
exactly the needed size (no doubling overshoot; `calloc` pages stay virtual
until touched). Without a target, lazy doubling can peak at up to ~2x the
sequence — ~57 GB instead of 38.1 GB for f32 — which plus the assistant
exceeds 128 GB. **For 256K runs always set `FATCAT_KV_TARGET`.**

### Hard cap: `FATCAT_MEM_BUDGET_GB` (engine's own memory)

`FATCAT_MEM_BUDGET_GB` (or `_MB`) caps what the engine itself allocates —
warm tier + KV/indexer + a 1 GB fixed reserve — independent of the
system-wide `FATCAT_MEM_LIMIT_MB` trigger. Behavior:

- **At open**: KV target bytes are computed from the model geometry; if the
  target doesn't fit, it is clamped (KV wins over the warm tier), and the
  warm tier gets the remainder (`FATCAT_WARM_MB` is lowered if needed).
- **At runtime**: KV growth first shrinks the warm tier (`fatcat_stream_-
  shrink`, coldest slots first, pinned/in-use slots untouched) to make room,
  and refuses to grow past the cap (clear error instead of an OOM kill).
- Caveat: KV growth transiently holds old + new KV during the copy, so leave
  headroom ≥ current KV size when growing without `FATCAT_KV_TARGET`.

Example — coexist with another large process, 24 GB hard cap:

```
FATCAT_MEM_BUDGET_GB=24 FATCAT_KV_TARGET=262144 FATCAT_KV_F16=1 \
FATCAT_WARM_MB=65536   # warm auto-clamped to ~3 GB
```

Recommended env for a 256K run on this 128 GB box, assistant stopped
(f16 KV, max warm tier, GPU on — Session 30 optimization pass, tier
retargeted to the 110 GB budget in Session 42):

```
FATCAT_KV_TARGET=263168 FATCAT_KV_F16=1 FATCAT_KV_HUGEPAGE=1 FATCAT_IDX=1 \
FATCAT_WARM_MB=86016 FATCAT_MEM_LIMIT_MB=110000
```

- **110 GB is the target budget on this Strix Halo box** (assistant
  stopped): KV 19.0 + indexer 5.9 + ~2 GB fixed + reserve leaves ~84 GB
  for the warm tier at 256K context (`FATCAT_WARM_MB=86016`). For
  short-context work the measured sweet spot is **64 GB**
  (`FATCAT_WARM_MB=65536`): 86 GB bought only +1.4% decode (1.41 -> 1.43
  tok/s, Session 42f) and **96 GB swaps** - the budget counts only
  tier+KV+reserve, not the GPU GTT buffers (on Strix Halo hipMalloc IS
  system RAM, ~6-8 GB) nor the GGUF page cache, so the honest ceiling is
  `avail - KV - ~9 GB`, not `budget - KV - 1 GB`. The tier is fully
  configurable - these are recommended operating points, not limits.

- `FATCAT_MEM_LIMIT_MB=110000`, never 0: 0 disables the guardrail and the
  page cache grew until the OOM killer took the assistant down (Session
  40c). 110000 keeps drop_experts off on an idle 128 GB box while still
  capping runaway growth.
- Decode (Session 42, D5f + 42f): decode is NVMe-bandwidth-bound; the
  warm-tier curve on policy100: 0.69 / 1.07 / 1.34 / 1.43 tok/s at 18 /
  40 / 64 / 86 GB (hit 77.2% @64, 79.5% @86) - the curve flattens at 64
  GB, so 64 is the default. Expert speculation prefetch is OFF by default
  (`FATCAT_SPECP=0`): at 64 GB it was speed-neutral and cost ~1 GB/token
  of NVMe bandwidth the critical reads need. `FATCAT_PIN=1` (pin ~16 GB
  hot weights to anonymous RAM) measured flat - leave off unless NVMe
  refaults ever dominate.

- `FATCAT_WARM_MB=65536`: with the assistant gone, cache ~64 GB of experts
  (model is 218 GB on disk, so this is the single biggest prefill lever:
  every warm hit is ~2 MB of NVMe traffic avoided). Budget: 19.0 KV + 5.9
  indexer + 64 warm ~= 89 GB anon - fits with headroom. Session 42f
  measured the last rung: 86 GB adds +2.3pp hit for +1.4% decode (within
  run-to-run noise) - stay at 64 unless you specifically want the extra
  hit margin; do NOT go to 96 (swaps, see above).
- GPU stays ON (do not set `FATCAT_GPU=0`): zero-copy warm tier via HSA
  userptr. Validated numerically with the full P5 stack (indexer + f16 KV,
  Session 30). The idle box is exactly the condition the P4 verdict needs
  re-measuring under - run the A/B below.
- No `OMP_WAIT_POLICY=passive` on an idle box (active wait is faster;
  passive only existed to yield cores to the assistant).

GPU vs CPU A/B (settles the deferred P4 idle re-measure): run one ladder
rung (e.g. 32K tokens) twice, `FATCAT_GPU=0` vs default, best of 2 each,
compare prefill + decode tok/s.

f32 KV at 256K (38.1 + 5.9 + 68 = ~112 GB) fits only with a small warm
tier and swap headroom; prefer f16 (cos 0.992–0.998 vs f32, Session 26).

## 2. GTT / GPU memory mapping (Strix Halo, unified memory)

Strix Halo has no dedicated VRAM: the gfx1151 iGPU shares system RAM.
Avoiding a BIOS "GTT carve-out" (STOLEN/UMA boot option) is fine — FatCat's
P4 path maps system pages to the GPU via HSA userptr (zero-copy), which
costs no carve-out. Relevant knobs if the GPU path is used:

- `amdgpu.gttsize=<MiB>` module parameter: sizes the GTT worker pool used
  for large non-pinned buffer mappings. If HIP large-buffer maps fail under
  memory pressure, raise it (e.g. `amdgpu.gttsize=32768`).
- `amdgpu.vm_fragment_size=9`: full-VM mode, cheaper page tables for many
  small mappings; harmless default for our usage.
- The GPU shares the same LPDDR5X bandwidth as the CPU; on a loaded box GPU
  timings are dominated by contention (P4 verdict). Re-measure GPU value
  only on an otherwise-idle system.

## 3. NUMA

Strix Halo presents a **single NUMA node** (`numactl --hardware` shows
node 0 only), so there is nothing to bind: NUMA pinning/interleave policies
are no-ops here. If FatCat is ever run on a multi-node host, pin the process
to the node holding the model page cache (`numactl --cpunodebind --membind`)
and keep the warm tier on one node; on this platform skip it.

## 4. Transparent hugepages — validation procedure

Host THP is in `madvise` mode (`/sys/kernel/mm/transparent_hugepage/enabled`),
so only `FATCAT_KV_HUGEPAGE=1` regions get hugepages (KV + indexer caches).
It is numerically transparent (bit-identical output, Session 27); the win is
reduced TLB/page-walk pressure over a 19–38 GB scan working set.

Validate on an idle box (forced compaction can stall under memory pressure
with `defrag=madvise`):

1. `echo madvise | sudo tee /sys/kernel/mm/transparent_hugepage/defrag`
   (or `defer+madvise` to bound stalls).
2. Run the 256K corpus with and without `FATCAT_KV_HUGEPAGE=1`.
3. During the run, watch `grep AnonHugePages /proc/meminfo` — it should
   climb toward the KV cache size (allow minutes: khugepaged promotes
   lazily; `MADV_HUGEPAGE` also promotes at fault time under `defrag`).
4. Compare decode tokens/s (best of 3 runs each; the observer effect on a
   loaded box swamps the effect).

If promotion stalls hurt more than TLB saves, leave it off — it is opt-in.

## 5. Misc host settings

- Keep swap enabled as overflow, but watch `si/so` in `vmstat 5`: sustained
  swap-in during decode means the budget is over-committed.
- `vm.min_free_kbytes=2097152` (2 GB) gives the kernel more room to compact
  for hugepages/GTT without OOM-triggering early.
- The engine prints `kv: capacity N tokens, X GB` at each KV growth and
  peak RSS at exit; use those lines to confirm the budget on real runs.

## 6. GPU fabric clock (D0 — free bandwidth on Strix Halo)

The fclk (fabric clock feeding the unified-memory path) idles down under
`auto` DPM, capping the bandwidth that FatCat's streaming GEMV/GEMM kernels
depend on. llama.cpp-family forks for gfx1151 document holding it at max as
required for full bandwidth.

```bash
sudo tools/fabric-clock.sh hold    # manual DPM, fclk -> 2000 MHz (top step)
tools/fabric-clock.sh status       # verify (no root needed)
sudo tools/fabric-clock.sh reset   # back to auto when done
```

Settings do not survive reboot. A/B with `tools/gpu-gemm-test.sh` (or any
GPU run — compare the `batched prefill ... tok/s` line). Expected effect:
raises the memory-bandwidth ceiling for both GPU prefill and decode; zero
engine changes, zero risk (sysfs only).
