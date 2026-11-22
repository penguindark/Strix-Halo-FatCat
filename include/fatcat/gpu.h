/* fatcat — GPU (ROCm/HIP) acceleration for the ML4 forward pass.
 *
 * Design (Strix Halo APU): CPU keeps control flow (rmsnorm, rope, router
 * softmax/top-k, attention scores, KV cache); the GPU runs fused
 * dequantize+GEMV kernels over the big weight matrices.
 *  - Fixed (non-expert) weights are uploaded once, still quantized, and
 *    dequantized inside the kernel (memory-bound; avoids f32 bloat).
 *  - Routed experts are streamed per token into small staging slots
 *    (LRU-tagged so repeats within/across tokens skip the copy).
 * All buffers live in unified memory; the OS page cache absorbs NVMe re-reads.
 *
 * Every entry point is safe to call when the GPU is unavailable: init returns
 * false and the caller falls back to the scalar CPU path. */
#ifndef FATCAT_GPU_H
#define FATCAT_GPU_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Initialize the GPU context and staging memory. Returns false on any
 * failure (no driver, no device, OOM). ggml_type ids match fatcat/gguf.h. */
bool fatcat_gpu_init(uint32_t expert_slots_per_tensor, char *errbuf, size_t errlen);
void fatcat_gpu_shutdown(void);
bool fatcat_gpu_enabled(void);
/* Human-readable device info for the log header. */
const char *fatcat_gpu_device_string(void);

/* Upload a fixed weight tensor (quantized bytes, verbatim from GGUF).
 * Returns an opaque device handle usable as `dev` in fatcat_gpu_gemv, or NULL. */
void *fatcat_gpu_upload(uint32_t type, const void *host, size_t bytes);

/* Stage one expert (a contiguous quantized slice) into a device slot.
 * Returns a handle for fatcat_gpu_gemv, or NULL on failure. The LRU keeps
 * `tag`s; a repeated tag reuses its slot without copying. */
void *fatcat_gpu_stage(uint32_t type, const void *host, size_t bytes, uint64_t tag);

/* Release a handle returned by upload/stage (device memory itself is owned
 * by the GPU module; this frees the small handle). */
void fatcat_gpu_release(void *handle);

/* y[n_out] = W @ x[n_in]; W is a handle from upload/stage; row layout is the
 * GGML row-major quantized form (n_in elements per row). Synchronous. */
bool fatcat_gpu_gemv(void *w, uint32_t type, const float *x, float *y,
                     int64_t n_out, int64_t n_in);

/* Batched GEMM (D1): Y[t*n_out+i] = dot(W_row_i, X + t*n_in), t < n_tok.
 * X/Y host pointers; sync. Bit-identical to per-token fatcat_gpu_gemv.
 * Returns 0 on success, non-zero -> caller falls back to CPU.
 * IQ1_S/IQ2_XXS only. */
int fatcat_gpu_gemm(void *w, uint32_t type, const float *X, float *Y,
                    int64_t n_out, int64_t n_in, int n_tok);

/* Counters for the GPU gemm path (diagnostics: fails > 0 means the batched
 * path silently fell back to CPU gemm). */
void fatcat_gpu_gemm_stats(uint64_t *calls, uint64_t *fails);

/* Statistics for the log: device-side copy bytes and staging cache hits. */
void fatcat_gpu_stats(uint64_t *copies, uint64_t *copy_bytes, uint64_t *stage_hits);
/* Number of fixed-weight uploads that failed (device OOM): those tensors
 * silently ran on the CPU. Non-zero means the run was slower but still valid. */
uint64_t fatcat_gpu_upload_fails(void);

/* ---- P4 zero-copy (APU unified memory) ----
 * On an APU GTT is system RAM, so copying warm-tier slots to device memory
 * is pure waste. The warm tier can allocate its slots via the two functions
 * below; the GPU then reads them in place. */

/* Allocate pinned, device-mapped host memory (NULL on failure). */
void *fatcat_gpu_host_alloc(size_t bytes);
void fatcat_gpu_host_free(void *p);
/* True when device pointers equal host pointers for mapped allocations
 * (unified addressing). Zero-copy path requires this. */
bool fatcat_gpu_unified(void);
/* Wrap an already device-mapped pointer as a GEMV weight handle (no copy). */
void *fatcat_gpu_wrap(void *dev_ptr, uint32_t type);

/* ---- P4 v2 async pipelining ----
 * Launch-only GEMV: x and y must be device-accessible pointers (memory from
 * fatcat_gpu_host_alloc or device memory). Multiple calls queue on the
 * device stream; fatcat_gpu_wait() blocks until all queued work finishes.
 * gemv_dev returns non-zero on launch failure (caller falls back to CPU). */
int fatcat_gpu_gemv_dev(void *wh, uint32_t type, const float *x, float *y,
                        long long n_out, long long n_in);
int fatcat_gpu_wait(void);
/* D5e batched per-head Q8_0 GEMV: for item i in [0,n_items),
 * y[i*y_stride + r] = dot(dequant_q8_0(wh + i*w_stride, row r), x + i*x_stride).
 * Bit-identical to n_items gemv_dev calls (same per-row kernel body).
 * x/y device-accessible; stream-ordered, sync with fatcat_gpu_wait(). */
int fatcat_gpu_gemv_q8_0_batched(void *wh, long long w_stride,
                                 const float *x, long long x_stride,
                                 float *y, long long y_stride,
                                 long long n_out, long long n_in,
                                 long long n_items);
/* D5e MLA k-absorption: y[row] = dot(dequant_q8_0(wh row), x + (row/rows_per_item)*x_stride).
 * row_stride = tensor row_bytes (NoPE layers pad wk_b rows to the full qk).
 * Bit-identical to the CPU absorb pattern (dequant + AVX-512 dot order). */
int fatcat_gpu_absorb_q8_0(void *wh, const float *x, float *y,
                           long long rows_per_item, long long n_items,
                           long long n_in, long long x_stride,
                           long long row_stride);
/* Stage x into device memory for async GEMVs (stream-ordered copy into a
 * ring slot; returns the device pointer to pass as x to gemv_dev, or NULL
 * if unavailable — then pass the mapped pointer and accept slower reads). */
void *fatcat_gpu_xstage(const void *x, size_t bytes);

/* Microbench: practical device read bandwidth in GB/s (bytes of GTT
 * streamed with grid-stride reads, iters times). -1 on failure. */
double fatcat_gpu_readbw(size_t bytes, int iters);

#endif /* FATCAT_GPU_H */
