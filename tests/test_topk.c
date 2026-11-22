/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Dario Deledda */
/* test_topk — DSA indexer top-k selection vs brute force.
 *
 * fatcat_topk_select picks the k largest scores with a quickselect; ties are
 * resolved arbitrarily but deterministically, so we compare the multiset of
 * selected SCORES against brute force, plus structural invariants (indices
 * sorted ascending, unique, in range) and determinism across runs. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "fatcat/ml4.h"

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    if (failures < 20) { fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
    fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); } failures++; } } while (0)

static uint64_t rng_state = 0xDEADBEEF12345ULL;
static uint32_t rnd(void)
{
    rng_state = rng_state * 6364136223846793005ULL + 1442695040888963407ULL;
    return (uint32_t)(rng_state >> 32);
}

static int cmp_f32_desc(const void *a, const void *b)
{
    float x = *(const float *)a, y = *(const float *)b;
    return (x < y) - (x > y);
}

/* brute force: sort all scores descending, take top k as a score multiset */
static void brute_topk(const float *s, int64_t n, int64_t k, float *out_sorted)
{
    float *tmp = malloc((size_t)n * sizeof(float));
    memcpy(tmp, s, (size_t)n * sizeof(float));
    qsort(tmp, (size_t)n, sizeof(float), cmp_f32_desc);
    memcpy(out_sorted, tmp, (size_t)k * sizeof(float));
    qsort(out_sorted, (size_t)k, sizeof(float), cmp_f32_desc);
    free(tmp);
}

static void check_case(const float *s, int64_t n, int64_t k)
{
    int64_t kk = k < n ? k : n;
    int64_t *out = malloc((size_t)(n ? n : 1) * sizeof(int64_t));
    int64_t *out2 = malloc((size_t)(n ? n : 1) * sizeof(int64_t));
    fatcat_topk_select(s, n, k, out);
    if (k >= n) {
        for (int64_t i = 0; i < n; i++) CHECK(out[i] == i, "k>=n: expected identity");
        free(out); free(out2); return;
    }
    /* structural: ascending, unique, in range */
    for (int64_t i = 0; i < kk; i++) {
        CHECK(out[i] >= 0 && out[i] < n, "index %lld out of range", (long long)out[i]);
        if (i) CHECK(out[i] > out[i - 1], "not strictly ascending at %lld", (long long)i);
    }
    /* score multiset matches brute force */
    float *got = malloc((size_t)(kk ? kk : 1) * sizeof(float));
    for (int64_t i = 0; i < kk; i++) got[i] = s[out[i]];
    qsort(got, (size_t)kk, sizeof(float), cmp_f32_desc);
    float *want = malloc((size_t)(kk ? kk : 1) * sizeof(float));
    brute_topk(s, n, kk, want);
    for (int64_t i = 0; i < kk; i++)
        CHECK(got[i] == want[i], "n=%lld k=%lld rank %lld: got %g want %g",
              (long long)n, (long long)k, (long long)i, got[i], want[i]);
    /* determinism */
    fatcat_topk_select(s, n, k, out2);
    CHECK(memcmp(out, out2, (size_t)kk * sizeof(int64_t)) == 0, "n=%lld k=%lld: not deterministic",
          (long long)n, (long long)k);
    free(out); free(out2); free(got); free(want);
}

int main(void)
{
    /* edge cases */
    float one[1] = { 3.5f };
    check_case(one, 1, 1);
    check_case(one, 1, 5);
    check_case(one, 1, 0);

    /* small hand case: distinct scores */
    float small[8] = { 5, -1, 9, 0, 9, 2, 7, 7 };
    check_case(small, 8, 3);   /* top: 9,9,7 */
    check_case(small, 8, 1);
    check_case(small, 8, 8);

    /* random cases, incl. many ties (values from a small set) */
    for (int trial = 0; trial < 200; trial++) {
        int64_t n = 1 + (int64_t)(rnd() % 500);
        int64_t k = (int64_t)(rnd() % (uint64_t)(n + 2));  /* may exceed n */
        float *s = malloc((size_t)n * sizeof(float));
        int ties = trial % 2;
        for (int64_t i = 0; i < n; i++)
            s[i] = ties ? (float)(rnd() % 5) : (float)rnd() / 8388608.0f - 256.0f;
        check_case(s, n, k);
        free(s);
    }

    /* larger case */
    {
        int64_t n = 5000, k = 2048;
        float *s = malloc((size_t)n * sizeof(float));
        for (int64_t i = 0; i < n; i++) s[i] = (float)rnd() / 8388608.0f - 256.0f;
        check_case(s, n, k);
        free(s);
    }

    if (failures) { fprintf(stderr, "test_topk: %d FAILURES\n", failures); return 1; }
    printf("test_topk: all checks passed\n");
    return 0;
}
