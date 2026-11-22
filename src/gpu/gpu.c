/* fatcat GPU wrapper: thin C layer over the HIP implementation in
 * src/gpu/kernels.hip (compiled by hipcc). See include/fatcat/gpu.h. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fatcat/gpu.h"

/* implemented in kernels.hip (extern "C") */
extern int fatcat_gpu_init_impl(uint32_t slots, char *eb, size_t el);
extern void fatcat_gpu_shutdown_impl(void);
extern void fatcat_gpu_stats_impl(uint64_t *copies, uint64_t *bytes, uint64_t *hits);
extern uint64_t fatcat_gpu_upload_fails_impl(void);
extern const char *fatcat_gpu_devstr_impl(void);
extern void *fatcat_gpu_upload_impl(uint32_t type, const void *host, size_t bytes);
extern void *fatcat_gpu_stage_impl(uint32_t type, const void *host, size_t bytes, uint64_t tag);
extern void fatcat_gpu_free_handle(void *handle);
extern int fatcat_gpu_gemv_impl(void *wh, uint32_t type, const float *x, float *y,
                                long long n_out, long long n_in);
extern int fatcat_gpu_gemm_impl(void *wh, uint32_t type, const float *X, float *Y,
                                long long n_out, long long n_in, int n_tok);
extern void fatcat_gpu_gemm_stats_impl(uint64_t *calls, uint64_t *fails);

static bool g_enabled = false;
static char g_devstr[160] = "";

bool fatcat_gpu_init(uint32_t expert_slots_per_tensor, char *errbuf, size_t errlen)
{
    if (getenv("FATCAT_GPU") && atoi(getenv("FATCAT_GPU")) == 0) {
        snprintf(errbuf, errlen, "disabled by FATCAT_GPU=0");
        return false;
    }
    char eb[256] = {0};
    if (fatcat_gpu_init_impl(expert_slots_per_tensor, eb, sizeof(eb)) != 0) {
        snprintf(errbuf, errlen, "%s", eb);
        return false;
    }
    snprintf(g_devstr, sizeof(g_devstr), "%s", fatcat_gpu_devstr_impl());
    g_enabled = true;
    return true;
}

void fatcat_gpu_shutdown(void)
{
    if (!g_enabled) return;
    fatcat_gpu_shutdown_impl();
    g_enabled = false;
}

bool fatcat_gpu_enabled(void) { return g_enabled; }

const char *fatcat_gpu_device_string(void) { return g_devstr; }

void *fatcat_gpu_upload(uint32_t type, const void *host, size_t bytes)
{
    if (!g_enabled) return NULL;
    return fatcat_gpu_upload_impl(type, host, bytes);
}

void *fatcat_gpu_stage(uint32_t type, const void *host, size_t bytes, uint64_t tag)
{
    if (!g_enabled) return NULL;
    return fatcat_gpu_stage_impl(type, host, bytes, tag);
}

void fatcat_gpu_release(void *handle)
{
    if (handle) fatcat_gpu_free_handle(handle);
}

bool fatcat_gpu_gemv(void *w, uint32_t type, const float *x, float *y,
                     int64_t n_out, int64_t n_in)
{
    if (!g_enabled || !w) return false;
    return fatcat_gpu_gemv_impl(w, type, x, y, (long long)n_out, (long long)n_in) == 0;
}

int fatcat_gpu_gemm(void *w, uint32_t type, const float *X, float *Y,
                    int64_t n_out, int64_t n_in, int n_tok)
{
    if (!g_enabled || !w) return -1;
    return fatcat_gpu_gemm_impl(w, type, X, Y, (long long)n_out, (long long)n_in, n_tok);
}

void fatcat_gpu_gemm_stats(uint64_t *calls, uint64_t *fails)
{
    if (g_enabled) fatcat_gpu_gemm_stats_impl(calls, fails);
    else { *calls = 0; *fails = 0; }
}

void fatcat_gpu_stats(uint64_t *copies, uint64_t *copy_bytes, uint64_t *stage_hits)
{
    if (g_enabled) fatcat_gpu_stats_impl(copies, copy_bytes, stage_hits);
    else { *copies = *copy_bytes = *stage_hits = 0; }
}

uint64_t fatcat_gpu_upload_fails(void)
{
    return g_enabled ? fatcat_gpu_upload_fails_impl() : 0;
}

extern void *fatcat_gpu_host_alloc_impl(size_t bytes);
extern void fatcat_gpu_host_free_impl(void *p);
extern int fatcat_gpu_unified_impl(void);
extern void *fatcat_gpu_wrap_impl(void *dev_ptr, uint32_t type);

void *fatcat_gpu_host_alloc(size_t bytes)
{
    return g_enabled ? fatcat_gpu_host_alloc_impl(bytes) : NULL;
}

void fatcat_gpu_host_free(void *p)
{
    if (g_enabled) fatcat_gpu_host_free_impl(p);
}

bool fatcat_gpu_unified(void)
{
    return g_enabled && fatcat_gpu_unified_impl();
}

void *fatcat_gpu_wrap(void *dev_ptr, uint32_t type)
{
    if (!g_enabled) return NULL;
    return fatcat_gpu_wrap_impl(dev_ptr, type);
}

extern int fatcat_gpu_gemv_dev_impl(void *wh, uint32_t type, const float *x, float *y,
                                    long long n_out, long long n_in);
extern int fatcat_gpu_wait_impl(void);

int fatcat_gpu_gemv_dev(void *wh, uint32_t type, const float *x, float *y,
                        long long n_out, long long n_in)
{
    if (!g_enabled) return -1;
    return fatcat_gpu_gemv_dev_impl(wh, type, x, y, n_out, n_in);
}

int fatcat_gpu_wait(void)
{
    if (!g_enabled) return -1;
    return fatcat_gpu_wait_impl();
}

extern int fatcat_gpu_gemv_q8_0_batched_impl(void *wh, long long w_stride,
                                             const float *x, long long x_stride,
                                             float *y, long long y_stride,
                                             long long n_out, long long n_in,
                                             long long n_items);
int fatcat_gpu_gemv_q8_0_batched(void *wh, long long w_stride,
                                 const float *x, long long x_stride,
                                 float *y, long long y_stride,
                                 long long n_out, long long n_in,
                                 long long n_items)
{
    if (!g_enabled) return -1;
    return fatcat_gpu_gemv_q8_0_batched_impl(wh, w_stride, x, x_stride, y,
                                             y_stride, n_out, n_in, n_items);
}

extern int fatcat_gpu_absorb_q8_0_impl(void *wh, const float *x, float *y,
                                       long long rows_per_item, long long n_items,
                                       long long n_in, long long x_stride,
                                       long long row_stride);

int fatcat_gpu_absorb_q8_0(void *wh, const float *x, float *y,
                           long long rows_per_item, long long n_items,
                           long long n_in, long long x_stride,
                           long long row_stride)
{
    if (!g_enabled) return -1;
    return fatcat_gpu_absorb_q8_0_impl(wh, x, y, rows_per_item, n_items,
                                       n_in, x_stride, row_stride);
}

extern void *fatcat_gpu_xstage_impl(const void *x, size_t bytes);

void *fatcat_gpu_xstage(const void *x, size_t bytes)
{
    if (!g_enabled) return NULL;
    return fatcat_gpu_xstage_impl(x, bytes);
}

extern double fatcat_gpu_readbw_impl(size_t bytes, int iters);

double fatcat_gpu_readbw(size_t bytes, int iters)
{
    if (!g_enabled) return -1.0;
    return fatcat_gpu_readbw_impl(bytes, iters);
}
