/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Dario Deledda */
/* fatcat — expert streaming tier manager (P3).
 *
 * The warm tier: a fixed budget of RAM slots holding expert slices
 * (one slice = one expert's one matrix, <= ~10 MB). Slices are keyed by
 * (layer, matrix, expert). On a miss the slice is pread() from NVMe into the
 * slot of the lowest-scoring victim; scores combine LFU count with recency
 * (exponential age decay in token ticks), matching docs/streaming-policy.md.
 *
 * On Strix Halo RAM and VRAM are the same physical memory, so tiers here are
 * a residency *policy*, not copies between devices. The cold tier (NVMe) is
 * read through pread (v1); io_uring prefetch lands in P3 v2.
 *
 * Thread-safe: metadata is mutex-guarded; slice fills run outside the lock.
 *
 * Eviction policy: DEFAULT S3-FIFO (small/main FIFOs + ghost table);
 * FATCAT_POLICY=lfu restores the legacy LFU+recency sampled-victim policy.
 * See fatcat_stream_policy() for the active name. */
#ifndef FATCAT_STREAM_H
#define FATCAT_STREAM_H

#include <stddef.h>
#include <stdint.h>

typedef struct fatcat_stream fatcat_stream;

typedef struct {
    uint64_t hits, misses, evictions;
    uint64_t hit_bytes, miss_bytes;
    /* miss breakdown: wait_bytes = get() blocked on an in-flight prefetch
     * (prefetch issued too late); cold_bytes = synchronous fill in get()
     * (no prefetch was in flight: never issued or dropped). */
    uint64_t wait_bytes, cold_bytes;
    uint64_t ring_bytes;    /* bytes submitted to io_uring (true NVMe demand) */
    uint64_t ring_spec_bytes;  /* ...of which speculative-class (guesses) */
    uint64_t n_slots, used_slots;
    size_t   slot_bytes;
} fatcat_stream_stats;

/* budget_mb = 0 disables the tier (every get is a miss -> caller fallback). */
fatcat_stream *fatcat_stream_open(size_t budget_mb, size_t slot_bytes,
                                  char *errbuf, size_t errlen);
void fatcat_stream_close(fatcat_stream *st);

/* Ensure len bytes of the slice `key` are resident; returns the pointer.
 * On a miss the bytes are read from (fd, file_off). len must be <=
 * slot_bytes. Returns NULL on oversize/IO failure (caller may fall back to
 * its own mapping). Thread-safe; concurrent gets of the same key block until
 * the in-flight fill completes.
 * Every successful get takes a use-reference on the slot: the slice cannot be
 * evicted until the caller calls fatcat_stream_release(key). */
const void *fatcat_stream_get(fatcat_stream *st, uint64_t key, int fd,
                              uint64_t file_off, size_t len, int *was_hit);

/* Drop one use-reference taken by fatcat_stream_get. Call once per get when
 * the returned pointer is no longer in use, or the slot is never evicted. */
void fatcat_stream_release(fatcat_stream *st, uint64_t key);

/* Queue an asynchronous read of `key` into a warm slot (io_uring worker).
 * Non-blocking: returns immediately; a later get() of the same key either
 * hits or waits for the in-flight fill. No-op if the key is already resident
 * or in flight, the tier is disabled, or no slot can be claimed.
 * spec != 0 marks a speculative (guessed) read: it queues BEHIND all
 * critical reads so a wrong guess can never delay a needed one. */
void fatcat_stream_prefetch(fatcat_stream *st, uint64_t key, int fd,
                            uint64_t file_off, size_t len, int spec);

/* Mark a slice as never-evict (hot tier placeholder; P4 GPU pinning). */
void fatcat_stream_pin(fatcat_stream *st, uint64_t key);

/* Override slot payload allocation (call before any get/prefetch). Used by
 * P4 to allocate slots as device-mapped pinned memory so the GPU reads
 * warm-tier bytes in place. alloc returns NULL on failure. */
typedef void *(*fatcat_stream_alloc_fn)(size_t bytes, void *ctx);
typedef void (*fatcat_stream_dealloc_fn)(void *ptr, size_t bytes, void *ctx);
void fatcat_stream_set_allocator(fatcat_stream *st, fatcat_stream_alloc_fn a,
                                 fatcat_stream_dealloc_fn f, void *ctx);

/* Advance the recency clock by one tick (call once per generated token). */
void fatcat_stream_tick(fatcat_stream *st);

/* Emergency reclaim: evict resident slots (releasing their payload pages)
 * until used capacity fits keep_bytes. In-flight, pinned and referenced
 * slots are skipped. Victim order follows the active policy (queue heads
 * for s3fifo, lowest score for lfu). Returns bytes of slot payload
 * released. Used by the FATCAT_MEM_BUDGET_GB hard cap when KV growth
 * squeezes the warm tier. */
size_t fatcat_stream_shrink(fatcat_stream *st, size_t keep_bytes);

/* Active eviction policy name: "s3fifo" (default) or "lfu". */
const char *fatcat_stream_policy(const fatcat_stream *st);

void fatcat_stream_get_stats(const fatcat_stream *st, fatcat_stream_stats *out);
/* Reset hit/miss counters (keep contents). */
void fatcat_stream_reset_stats(fatcat_stream *st);

#endif /* FATCAT_STREAM_H */
