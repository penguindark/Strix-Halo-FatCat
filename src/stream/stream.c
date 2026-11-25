/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Dario Deledda */
/* fatcat stream tier — warm-tier implementation (P3).
 *
 * Slot arena + open-addressing key table + LFU/recency scoring.
 * Score(slot) = count / (1 + age_tokens / 64): frequently used and recently
 * used experts survive; a slice used once 64 tokens ago scores half as much.
 * Victim selection samples up to 16 random candidates (avoids O(n) scans over
 * thousands of slots) and never evicts in-flight or pinned slots.
 *
 * Locking: one mutex guards the table + slot metadata. The pread() fill runs
 * with the slot marked in-flight and the lock released, so parallel misses
 * proceed concurrently. A second requester of an in-flight key waits on a
 * condition variable.
 *
 * P3 v2 — async prefetch: a fill worker thread drains a request queue and
 * reads through io_uring (raw syscalls, see uring.c; plain pread fallback if
 * the ring is unavailable). fatcat_stream_prefetch() claims a slot and queues
 * the read without blocking, so callers can overlap NVMe reads with compute.
 *
 * FATCAT_POLICY selects the eviction policy: the DEFAULT is S3-FIFO (Yang
 * et al., "Tiny Table, High Hit Rate", 2023) — a small FIFO (10% of slots)
 * filters one-hit wonders, a main FIFO (90%) holds promoted residents, and
 * a ghost FIFO of evicted keys (tiny open-addressed table) lets
 * re-referenced keys enter main directly. Victims are queue heads — O(1),
 * no sampling. Evidence: trace replay (tools/stream-sim, Session 33) and a
 * standalone GPU A/B at an 18 GB tier (Session 36: hit 77.6% vs 74.6%,
 * -8.5% evictions, -8.5% wall, logits bit-identical). FATCAT_POLICY=lfu
 * restores the legacy LFU/recency sampling (count/(1+age/64), 16-sample
 * victims); it slightly won at a 1 GB tier, so it remains the escape hatch
 * for tiers far below the working set. */
#define _GNU_SOURCE
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include "fatcat/stream.h"
#include "uring.h"

typedef struct {
    void    *data;      /* lazily mmap'd slot payload */
    uint64_t key;       /* 0 = empty slot */
    uint64_t last;      /* recency clock at last hit */
    uint64_t count;     /* LFU hits */
    uint32_t len;       /* valid bytes in data */
    uint16_t ref;       /* use-count: get() takes one, release() drops one */
    uint8_t  valid, inflight, pin;
    uint8_t  custom;    /* data came from the custom allocator */
    int32_t  prev, next;/* S3-FIFO intrusive list links, -1 = none */
    uint8_t  q;         /* S3-FIFO queue: 0 none, 1 small, 2 main */
    uint8_t  s3freq;    /* S3-FIFO frequency bit: hit since entering queue */
    int      fd;        /* source (fd, off): used for post-fill fadvise */
    uint64_t off;
} slot_t;

#define Q_NONE  0
#define Q_SMALL 1
#define Q_MAIN  2

typedef struct fillreq {
    struct fillreq *next;
    slot_t *s;
    int     fd;
    uint64_t off;
    size_t  len;
    int     spec;      /* queued on the speculative class (for stats) */
} fillreq;

/* queue classes: critical (this layer's routed experts) always drain before
 * speculative (next-layer guesses: ~30% wrong, must not delay real reads) */
#define FQ_N 2

struct fatcat_stream {
    slot_t  *slots;
    size_t   n_slots, used;
    size_t   slot_bytes;

    uint64_t *tab;          /* hash table: slot idx + 1, 0 = empty */
    size_t    tab_mask;

    pthread_mutex_t mu;
    pthread_cond_t  cv;

    /* async fill worker (P3 v2) */
    fatcat_uring *ring;         /* NULL -> worker uses plain pread */
    pthread_t     worker;
    int           worker_started, shutdown;
    fillreq      *q_head[FQ_N], *q_tail[FQ_N];
    pthread_cond_t q_cv;

    uint64_t clock;         /* token ticks */
    uint64_t rng;

    /* S3-FIFO state (FATCAT_POLICY=s3fifo) */
    int      use_s3;
    size_t   small_cap, main_cap;
    int32_t  small_head, small_tail, main_head, main_tail;
    size_t   small_len, main_len;
    uint64_t *gtab;         /* ghost key table: key+1 stored, 0 = empty */
    size_t   gtab_mask;
    uint64_t *gring;        /* ghost FIFO (pow2-sized ring of key+1) */
    size_t   gring_mask, ghead, gcount, gcap;

    fatcat_stream_alloc_fn alloc;
    fatcat_stream_dealloc_fn dealloc;
    void *alloc_ctx;

    uint64_t hits, misses, evictions, hit_bytes, miss_bytes;
    uint64_t wait_bytes, cold_bytes;
    uint64_t ring_bytes;    /* bytes actually submitted to io_uring by the
                             * fill worker (true NVMe demand, incl. spec
                             * prefetches that were never consumed) */
    uint64_t ring_spec_bytes;  /* ...of which speculative-class requests */
    int fadv;               /* POSIX_FADV_DONTNEED streamed slices (D5f) */

    /* H6 prefill anti-thrash (FATCAT_NOPROMO): while prefill_mode is set,
     * tier misses apply an admission policy so prefill's one-shot experts
     * do not evict decode-hot entries: 1 = never promote, 2 = promote only
     * on a second reference within the same prefill (seen table). */
    int prefill_mode, promo_policy;
    uint64_t *seen; size_t seen_mask;   /* second-ref table (policy 2) */
    uint64_t prefill_skips;             /* misses served without promoting */
};

/* ---------- helpers ---------- */

static size_t next_pow2(size_t n) { size_t p = 1; while (p < n) p <<= 1; return p; }

static inline uint64_t splitmix(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

static double slot_score(const slot_t *s, uint64_t now)
{
    double age = (double)(now - s->last);
    return (double)s->count / (1.0 + age / 64.0);
}

/* find table entry for key (or insertion point). caller holds mu. */
static size_t *tab_find(fatcat_stream *st, uint64_t key)
{
    size_t i = (size_t)(splitmix(key) & st->tab_mask);
    while (st->tab[i] && st->slots[st->tab[i] - 1].key != key)
        i = (i + 1) & st->tab_mask;
    return &st->tab[i];
}

static void tab_erase(fatcat_stream *st, size_t *e)
{
    /* backward-shift deletion for open addressing */
    size_t mask = st->tab_mask;
    size_t empty = (size_t)(e - st->tab);
    st->tab[empty] = 0;
    for (size_t i = empty;;) {
        i = (i + 1) & mask;
        uint64_t v = st->tab[i];
        if (!v) break;
        size_t home = (size_t)(splitmix(st->slots[v - 1].key) & mask);
        /* shift entry to `empty` iff empty lies on the probe path home->i */
        if (((i - home) & mask) >= ((i - empty) & mask)) {
            st->tab[empty] = v;
            st->tab[i] = 0;      /* the hole follows the shifted entry */
            empty = i;
        }
    }
}

/* ---------- S3-FIFO queues + ghost table (FATCAT_POLICY=s3fifo) ---------- */

static void q_push(fatcat_stream *st, slot_t *s, int q)
{
    int32_t idx = (int32_t)(s - st->slots);
    int32_t *head = q == Q_SMALL ? &st->small_head : &st->main_head;
    int32_t *tail = q == Q_SMALL ? &st->small_tail : &st->main_tail;
    size_t  *len  = q == Q_SMALL ? &st->small_len  : &st->main_len;
    s->prev = *tail; s->next = -1; s->q = (uint8_t)q;
    if (*tail >= 0) st->slots[*tail].next = idx; else *head = idx;
    *tail = idx; (*len)++;
}

static void q_unlink(fatcat_stream *st, slot_t *s)
{
    if (s->q == Q_NONE) return;
    int q = s->q;
    int32_t *head = q == Q_SMALL ? &st->small_head : &st->main_head;
    int32_t *tail = q == Q_SMALL ? &st->small_tail : &st->main_tail;
    size_t  *len  = q == Q_SMALL ? &st->small_len  : &st->main_len;
    if (s->prev >= 0) st->slots[s->prev].next = s->next; else *head = s->next;
    if (s->next >= 0) st->slots[s->next].prev = s->prev; else *tail = s->prev;
    s->prev = s->next = -1; s->q = Q_NONE; (*len)--;
}

/* ghost table: dedup membership for evicted keys; entries expire only via
 * the FIFO ring, so each key holds at most one ring slot. */
static void gtab_erase(fatcat_stream *st, size_t pos)
{
    size_t mask = st->gtab_mask;
    st->gtab[pos] = 0;
    for (size_t i = pos;;) {
        i = (i + 1) & mask;
        uint64_t v = st->gtab[i];
        if (!v) break;
        size_t home = (size_t)(splitmix(v - 1) & mask);
        if (((i - home) & mask) >= ((i - pos) & mask)) {
            st->gtab[pos] = v; st->gtab[i] = 0; pos = i;
        }
    }
}

static void ghost_add(fatcat_stream *st, uint64_t key)
{
    if (!st->gtab) return;
    size_t mask = st->gtab_mask, i = (size_t)(splitmix(key) & mask);
    while (st->gtab[i]) {
        if (st->gtab[i] == key + 1) return;   /* already ghosted */
        i = (i + 1) & mask;
    }
    st->gtab[i] = key + 1;
    if (st->gcount == st->gcap) {             /* ring full: expire oldest */
        uint64_t old = st->gring[st->ghead];
        st->ghead = (st->ghead + 1) & st->gring_mask; st->gcount--;
        size_t j = (size_t)(splitmix(old - 1) & mask);
        while (st->gtab[j] != old) j = (j + 1) & mask;
        gtab_erase(st, j);
    }
    st->gring[(st->ghead + st->gcount) & st->gring_mask] = key + 1;
    st->gcount++;
}

static int ghost_test(fatcat_stream *st, uint64_t key)
{
    if (!st->gtab) return 0;
    size_t mask = st->gtab_mask, i = (size_t)(splitmix(key) & mask);
    while (st->gtab[i]) {
        if (st->gtab[i] == key + 1) return 1;
        i = (i + 1) & mask;
    }
    return 0;
}

/* claim a slot for `key`: re-referenced keys (ghost hit) enter main, new
 * keys enter the small filter queue. caller holds mu, key already set. */
static void s3_insert(fatcat_stream *st, slot_t *s)
{
    if (!st->use_s3) return;
    s->s3freq = 0;
    q_push(st, s, ghost_test(st, s->key) ? Q_MAIN : Q_SMALL);
}

/* S3-FIFO victim: drain the over-capacity queue's head. freq bit set ->
 * promote (small->main) or reinsert (main tail) with the bit reset; else
 * evict and ghost the key. Busy (inflight/pinned/referenced) heads get a
 * grace period at the tail. NULL if everything is busy (caller falls back
 * to sampling). */
static slot_t *s3_pick(fatcat_stream *st)
{
    int guard = 4 * (int)(st->n_slots + 16);
    while (guard-- > 0) {
        int q;
        if (st->small_len > st->small_cap)      q = Q_SMALL;
        else if (st->main_len > st->main_cap)   q = Q_MAIN;
        else if (st->small_len)                 q = Q_SMALL;
        else if (st->main_len)                  q = Q_MAIN;
        else return NULL;
        int32_t head = q == Q_SMALL ? st->small_head : st->main_head;
        if (head < 0) {   /* chosen queue empty: try the other */
            q = Q_SMALL + Q_MAIN - q;
            head = q == Q_SMALL ? st->small_head : st->main_head;
            if (head < 0) return NULL;
        }
        slot_t *s = &st->slots[head];
        if (s->inflight || s->pin || s->ref) {
            q_unlink(st, s); q_push(st, s, q);
            continue;
        }
        q_unlink(st, s);
        if (s->s3freq) {
            s->s3freq = 0;
            q_push(st, s, Q_MAIN);   /* second chance in main */
            continue;
        }
        st->evictions++;
        size_t *e = tab_find(st, s->key);
        if (*e) tab_erase(st, e);
        ghost_add(st, s->key);
        s->valid = 0; s->key = 0; s->count = 0; s->pin = 0; s->ref = 0;
        s->inflight = 1;
        return s;
    }
    return NULL;
}

/* pick victim slot index (caller holds mu, must have used < n_slots or a
 * victim exists). returns slot idx; marks it inflight. */
static slot_t *pick_victim(fatcat_stream *st)
{
    if (st->used < st->n_slots) {   /* fresh slot first */
        for (size_t i = 0; i < st->n_slots; i++)
            if (!st->slots[i].valid && !st->slots[i].inflight) {
                st->used++;
                st->slots[i].inflight = 1;
                return &st->slots[i];
            }
    }
    if (st->use_s3) {
        slot_t *s = s3_pick(st);
        if (s) return s;
        /* queues all busy: fall through to score sampling */
    }
    slot_t *best = NULL;
    double best_s = 1e300;
    int tries = (int)(st->n_slots < 16 ? st->n_slots : 16);
    for (int t = 0; t < tries; t++) {
        st->rng = splitmix(st->rng);
        size_t i = (size_t)(st->rng % st->n_slots);
        slot_t *s = &st->slots[i];
        if (!s->valid || s->inflight || s->pin || s->ref) continue;
        double sc = slot_score(s, st->clock);
        if (sc < best_s) { best_s = sc; best = s; }
    }
    if (!best) {   /* pathological: all pinned/inflight — linear scan */
        for (size_t i = 0; i < st->n_slots; i++) {
            slot_t *s = &st->slots[i];
            if (!s->valid || s->inflight || s->pin || s->ref) continue;
            double sc = slot_score(s, st->clock);
            if (sc < best_s) { best_s = sc; best = s; }
        }
    }
    if (best) {
        st->evictions++;
        q_unlink(st, best);
        size_t *e = tab_find(st, best->key);
        if (*e) tab_erase(st, e);
        best->valid = 0; best->key = 0; best->count = 0; best->pin = 0;
        best->ref = 0;
        best->inflight = 1;
    }
    return best;
}

static void *slot_data(fatcat_stream *st, slot_t *s)
{
    if (!s->data) {
        if (st->alloc) {
            s->data = st->alloc(st->slot_bytes, st->alloc_ctx);
            s->custom = 1;
            return s->data;
        }
        s->data = mmap(NULL, st->slot_bytes, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    }
    return s->data == MAP_FAILED ? NULL : s->data;
}

void fatcat_stream_set_allocator(fatcat_stream *st, fatcat_stream_alloc_fn a,
                                 fatcat_stream_dealloc_fn f, void *ctx)
{
    if (!st) return;
    pthread_mutex_lock(&st->mu);
    st->alloc = a; st->dealloc = f; st->alloc_ctx = ctx;
    pthread_mutex_unlock(&st->mu);
}

/* ---------- async fill worker (P3 v2) ---------- */

/* finalize a queued fill (ring completion or worker pread). ok: bytes are
 * in s->data; the entry becomes resident with no reference (a later get()
 * takes one). On failure the entry is removed and waiters woken. */
static void fill_complete(fatcat_stream *st, slot_t *s, int ok)
{
    /* D5f: the bytes now live in the warm slot; drop the page-cache copy so
     * streamed experts do not evict the hot mmap'd weights (attention,
     * shared FFN) that gemv reads directly -> those refaults were ~2 GB of
     * redundant NVMe traffic per token at line rate. FATCAT_FADV=0 off. */
    if (ok && st->fadv && s->fd >= 0)
        posix_fadvise(s->fd, (off_t)s->off, s->len, POSIX_FADV_DONTNEED);
    pthread_mutex_lock(&st->mu);
    if (ok) {
        s->valid = 1;
        s->count = 1; s->last = st->clock;
    } else {
        size_t *e = tab_find(st, s->key);
        if (*e == (uint64_t)(s - st->slots) + 1) tab_erase(st, e);
        q_unlink(st, s);
        s->valid = 0; s->key = 0;
        st->used--;
    }
    s->inflight = 0;
    pthread_cond_broadcast(&st->cv);
    pthread_mutex_unlock(&st->mu);
}

static void ring_done(uint64_t user_data, int32_t res, void *ctx)
{
    fatcat_stream *st = ctx;
    slot_t *s = &st->slots[user_data - 1];
    fill_complete(st, s, res == (int32_t)s->len);
}

static void *fill_worker(void *arg)
{
    fatcat_stream *st = arg;
    for (;;) {
        pthread_mutex_lock(&st->mu);
        while (!st->q_head[0] && !st->q_head[1] && !st->shutdown)
            pthread_cond_wait(&st->q_cv, &st->mu);
        /* critical first, speculative behind it */
        fillreq *q = st->q_head[0];
        if (q) st->q_tail[0]->next = st->q_head[1];
        else   q = st->q_head[1];
        st->q_head[0] = st->q_tail[0] = st->q_head[1] = st->q_tail[1] = NULL;
        int sd = st->shutdown;
        if (q) {
            /* Burst coalescing: the producer queues a whole layer's slices
             * within microseconds, but may signal us mid-burst. Waiting a
             * short while for siblings keeps the NVMe queue deep instead of
             * serializing one-shot reads. */
            struct timespec ts;
            clock_gettime(CLOCK_REALTIME, &ts);
            ts.tv_nsec += 200 * 1000;
            if (ts.tv_nsec >= 1000000000L) { ts.tv_sec++; ts.tv_nsec -= 1000000000L; }
            pthread_cond_timedwait(&st->q_cv, &st->mu, &ts);
            /* append whatever arrived meanwhile: critical ahead of the
             * speculative tail we already hold */
            fillreq *tail = q;
            while (tail->next) tail = tail->next;
            if (st->q_head[0]) {
                tail->next = st->q_head[0];
                st->q_tail[0]->next = st->q_head[1];
            } else {
                tail->next = st->q_head[1];
            }
            st->q_head[0] = st->q_tail[0] = st->q_head[1] = st->q_tail[1] = NULL;
        }
        pthread_mutex_unlock(&st->mu);

        while (q) {
            if (!st->ring) {          /* fallback: serial pread */
                fillreq *r = q; q = r->next;
                void *dst = r->s->data;
                int ok = dst != NULL && dst != MAP_FAILED;
                size_t done = 0;
                while (ok && done < r->len) {
                    ssize_t p = pread(r->fd, (char *)dst + done,
                                      r->len - done, (off_t)r->off);
                    if (p < 0) { if (errno == EINTR) continue; ok = 0; break; }
                    if (p == 0) { ok = 0; break; }
                    done += (size_t)p;
                }
                fill_complete(st, r->s, ok);
                free(r);
                continue;
            }
            /* ring path: batch up to ring depth, submit, reap all */
            fillreq *batch[64];
            int n = 0, accepted = 0;
            while (q && n < 64) { batch[n++] = q; q = q->next; }
            fillreq *after = q;               /* first request not in batch */
            for (int i = 0; i < n; i++) {
                fillreq *rq = batch[i];
                void *dst = rq->s->data;
                int rc = -EINVAL;
                if (dst && dst != MAP_FAILED)
                    rc = fatcat_uring_read(st->ring, rq->fd, dst,
                                           (unsigned)rq->len, rq->off,
                                           (uint64_t)(rq->s - st->slots) + 1);
                if (rc < 0) fill_complete(st, rq->s, 0);
                else { accepted++;
                       pthread_mutex_lock(&st->mu); st->ring_bytes += rq->len;
                       if (rq->spec) st->ring_spec_bytes += rq->len;
                       pthread_mutex_unlock(&st->mu); }
            }
            if (accepted > 0) {
                fatcat_uring_submit(st->ring, 0);
                int done = 0;
                while (done < accepted) {
                    done += fatcat_uring_reap(st->ring, ring_done, st);
                    if (done < accepted) fatcat_uring_submit(st->ring, 1);
                }
            }
            for (fillreq *f = batch[0], *t; f && f != after; f = t) {
                t = f->next; free(f);
            }
        }
        if (sd) return NULL;
    }
}

void fatcat_stream_prefetch(fatcat_stream *st, uint64_t key, int fd,
                            uint64_t file_off, size_t len, int spec)
{
    if (!st || !st->worker_started || fd < 0 || len > st->slot_bytes) return;
    pthread_mutex_lock(&st->mu);
    if (tab_find(st, key)[0] != 0) {     /* resident or already in flight */
        pthread_mutex_unlock(&st->mu);
        return;
    }
    slot_t *s = pick_victim(st);
    if (!s) { pthread_mutex_unlock(&st->mu); return; }
    s->key = key; s->len = (uint32_t)len;
    s->fd = fd; s->off = file_off;
    *tab_find(st, key) = (uint64_t)(s - st->slots) + 1;
    s3_insert(st, s);
    if (!slot_data(st, s)) {             /* mmap failed: undo claim */
        size_t *e = tab_find(st, key);
        if (*e == (uint64_t)(s - st->slots) + 1) tab_erase(st, e);
        q_unlink(st, s);
        s->valid = 0; s->key = 0; s->inflight = 0; st->used--;
        pthread_mutex_unlock(&st->mu);
        return;
    }
    fillreq *r = calloc(1, sizeof *r);
    if (!r) {
        size_t *e = tab_find(st, key);
        if (*e == (uint64_t)(s - st->slots) + 1) tab_erase(st, e);
        q_unlink(st, s);
        s->valid = 0; s->key = 0; s->inflight = 0; st->used--;
        pthread_mutex_unlock(&st->mu);
        return;
    }
    *r = (fillreq){ NULL, s, fd, file_off, len, spec };
    const int c = spec ? 1 : 0;
    if (st->q_tail[c]) st->q_tail[c]->next = r; else st->q_head[c] = r;
    st->q_tail[c] = r;
    pthread_cond_signal(&st->q_cv);
    pthread_mutex_unlock(&st->mu);
}

/* ---------- public ---------- */

fatcat_stream *fatcat_stream_open(size_t budget_mb, size_t slot_bytes,
                                  char *errbuf, size_t errlen)
{
    if (!slot_bytes) slot_bytes = 10u << 20;
    fatcat_stream *st = calloc(1, sizeof *st);
    if (!st) { snprintf(errbuf, errlen, "stream: oom"); return NULL; }
    st->slot_bytes = slot_bytes;
    st->n_slots = budget_mb ? (budget_mb << 20) / slot_bytes : 0;
    st->rng = 0xC0FFEE123456789ULL;
    st->small_head = st->small_tail = st->main_head = st->main_tail = -1;
    pthread_mutex_init(&st->mu, NULL);
    pthread_cond_init(&st->cv, NULL);
    if (st->n_slots) {
        st->slots = calloc(st->n_slots, sizeof(slot_t));
        size_t tp = next_pow2(st->n_slots * 2);
        st->tab = calloc(tp, sizeof(uint64_t));
        st->tab_mask = tp - 1;
        const char *pol = getenv("FATCAT_POLICY");
        { const char *fa = getenv("FATCAT_FADV");    /* D5f page-cache guard */
          st->fadv = !fa || atoi(fa) != 0; }
        { const char *np = getenv("FATCAT_NOPROMO"); /* H6 prefill anti-thrash */
          st->promo_policy = np ? atoi(np) : 2;
          /* default 2 (promote on 2nd ref): A/B S43f — decode +5.9%,
           * prefill +9.0% vs off; never-promote (1) was worse on both. */
          if (st->promo_policy < 0 || st->promo_policy > 2) st->promo_policy = 0;
          if (st->promo_policy == 2) {
              size_t sp = 1u << 16;
              st->seen = calloc(sp, sizeof(uint64_t));
              st->seen_mask = sp - 1;
              if (!st->seen) st->promo_policy = 1;   /* degrade to never-promote */
          } }
        if (!pol || strcmp(pol, "lfu") != 0) {   /* default: s3fifo */
            st->use_s3 = 1;
            st->small_cap = st->n_slots / 10;
            if (!st->small_cap) st->small_cap = 1;
            st->main_cap = st->n_slots - st->small_cap;
            size_t gp = next_pow2(st->n_slots * 2);
            st->gtab = calloc(gp, sizeof(uint64_t));
            st->gtab_mask = gp - 1;
            st->gring = calloc(tp, sizeof(uint64_t));
            st->gring_mask = tp - 1;
            st->gcap = st->n_slots;
            if (!st->gtab || !st->gring) {   /* fall back to LFU, keep running */
                free(st->gtab); free(st->gring);
                st->gtab = st->gring = NULL;
                st->use_s3 = 0;
            }
        }
        if (!st->slots || !st->tab) {
            snprintf(errbuf, errlen, "stream: oom (%zu slots)", st->n_slots);
            fatcat_stream_close(st); return NULL;
        }
    }
    pthread_cond_init(&st->q_cv, NULL);
    if (st->n_slots) {
        const char *nu = getenv("FATCAT_URING");
        int use_ring = !nu || atoi(nu) != 0;    /* FATCAT_URING=0 disables */
        if (use_ring)
            fatcat_uring_open(&st->ring, 64);   /* NULL on failure -> pread */
        if (pthread_create(&st->worker, NULL, fill_worker, st) == 0)
            st->worker_started = 1;
    }
    return st;
}

void fatcat_stream_close(fatcat_stream *st)
{
    if (!st) return;
    if (st->worker_started) {
        pthread_mutex_lock(&st->mu);
        st->shutdown = 1;
        pthread_cond_broadcast(&st->q_cv);
        pthread_mutex_unlock(&st->mu);
        pthread_join(st->worker, NULL);
    }
    for (int c = 0; c < FQ_N; c++)
        while (st->q_head[c]) {   /* requests never processed */
            fillreq *r = st->q_head[c]; st->q_head[c] = r->next; free(r);
        }
    fatcat_uring_close(st->ring);
    if (st->slots)
        for (size_t i = 0; i < st->n_slots; i++) {
            void *d = st->slots[i].data;
            if (!d || d == MAP_FAILED) continue;
            if (st->slots[i].custom && st->dealloc)
                st->dealloc(d, st->slot_bytes, st->alloc_ctx);
            else
                munmap(d, st->slot_bytes);
        }
    free(st->slots); free(st->tab);
    free(st->gtab); free(st->gring);
    free(st->seen);
    pthread_mutex_destroy(&st->mu);
    pthread_cond_destroy(&st->cv);
    free(st);
}

static int seen_probe(fatcat_stream *st, uint64_t key);

const void *fatcat_stream_get(fatcat_stream *st, uint64_t key, int fd,
                              uint64_t file_off, size_t len, int *was_hit)
{
    if (was_hit) *was_hit = 0;
    if (!st || !st->n_slots || len > st->slot_bytes) return NULL;

    pthread_mutex_lock(&st->mu);
    int waited = 0;
    for (;;) {
        size_t *e = tab_find(st, key);
        if (*e) {
            slot_t *s = &st->slots[*e - 1];
            if (!s->inflight) {
                s->count++; s->last = st->clock; s->s3freq = 1;
                if (s->len < len && !s->ref) {   /* stale short entry: refetch */
                    tab_erase(st, e); s->valid = 0; s->key = 0; st->used--;
                    s->count = 0;
                    q_unlink(st, s);
                    break;
                }
                if (waited) {   /* served by an in-flight prefetch: cold bytes */
                    st->misses++; st->miss_bytes += len; st->wait_bytes += len;
                } else {
                    if (was_hit) *was_hit = 1;
                    st->hits++; st->hit_bytes += len;
                }
                s->ref++;
                void *d = s->data;
                pthread_mutex_unlock(&st->mu);
                return d;
            }
            /* in flight elsewhere: wait for it */
            waited = 1;
            pthread_cond_wait(&st->cv, &st->mu);
            continue;
        }
        break;
    }

    /* miss: H6 prefill admission gate (mu held) */
    if (st->prefill_mode && st->promo_policy) {
        int second = (st->promo_policy == 2) && seen_probe(st, key);
        if (!second) {
            st->prefill_skips++;
            st->misses++; st->miss_bytes += len; st->cold_bytes += len;
            pthread_mutex_unlock(&st->mu);
            return NULL;      /* caller falls back to its mmap view */
        }
    }

    /* miss: claim a slot, fill outside the lock */
    slot_t *s = pick_victim(st);
    if (!s) { pthread_mutex_unlock(&st->mu); return NULL; }
    s->key = key; s->len = 0;
    s->fd = fd; s->off = file_off;
    size_t *ins = tab_find(st, key);
    *ins = (uint64_t)(s - st->slots) + 1;
    s3_insert(st, s);
    void *dst = slot_data(st, s);
    pthread_mutex_unlock(&st->mu);

    int ok = 0;
    if (dst && fd >= 0) {
        size_t done = 0;
        ok = 1;
        while (done < len) {
            ssize_t r = pread(fd, (char *)dst + done, len - done, (off_t)(file_off + done));
            if (r < 0) { if (errno == EINTR) continue; ok = 0; break; }
            if (r == 0) { ok = 0; break; }
            done += (size_t)r;
        }
        if (ok && st->fadv)   /* D5f: slot has the bytes; drop page-cache copy */
            posix_fadvise(fd, (off_t)file_off, len, POSIX_FADV_DONTNEED);
    }

    pthread_mutex_lock(&st->mu);
    if (ok) {
        s->len = (uint32_t)len; s->valid = 1;
        s->count = 1; s->last = st->clock;
        s->ref = 1;               /* the get() caller holds one reference */
        st->misses++; st->miss_bytes += len; st->cold_bytes += len;
    } else {
        size_t *e = tab_find(st, key);
        if (*e == (uint64_t)(s - st->slots) + 1) tab_erase(st, e);
        q_unlink(st, s);
        s->valid = 0; s->key = 0; s->inflight = 0;
        st->used--;
        pthread_cond_broadcast(&st->cv);
        pthread_mutex_unlock(&st->mu);
        return NULL;
    }
    s->inflight = 0;
    pthread_cond_broadcast(&st->cv);
    void *d = s->data;
    pthread_mutex_unlock(&st->mu);
    return d;
}

void fatcat_stream_release(fatcat_stream *st, uint64_t key)
{
    if (!st || !st->n_slots) return;
    pthread_mutex_lock(&st->mu);
    size_t *e = tab_find(st, key);
    if (*e) {
        slot_t *s = &st->slots[*e - 1];
        if (s->ref) s->ref--;
    }
    pthread_mutex_unlock(&st->mu);
}

void fatcat_stream_pin(fatcat_stream *st, uint64_t key)
{
    if (!st || !st->n_slots) return;
    pthread_mutex_lock(&st->mu);
    size_t *e = tab_find(st, key);
    if (*e) st->slots[*e - 1].pin = 1;
    pthread_mutex_unlock(&st->mu);
}

void fatcat_stream_tick(fatcat_stream *st)
{
    if (!st) return;
    pthread_mutex_lock(&st->mu);
    st->clock++;
    pthread_mutex_unlock(&st->mu);
}

/* Non-intrusive residency probe (H3 cache-aware routing): 1 if key is
 * currently resident in the tier, else 0. Takes no reference, records no
 * hit/miss stats. A 0 may still be served from the GGUF mmap view. */
int fatcat_stream_resident(fatcat_stream *st, uint64_t key)
{
    if (!st) return 0;
    pthread_mutex_lock(&st->mu);
    size_t *e = tab_find(st, key);
    int res = (*e && st->slots[*e - 1].key == key) ? 1 : 0;
    pthread_mutex_unlock(&st->mu);
    return res;
}

/* H6: enter/leave prefill phase (admission policy FATCAT_NOPROMO applies
 * to misses while set). Entering clears the second-reference table. */
void fatcat_stream_set_prefill(fatcat_stream *st, int on)
{
    if (!st || st->promo_policy == 0) return;
    pthread_mutex_lock(&st->mu);
    if (on && st->seen) memset(st->seen, 0, (st->seen_mask + 1) * sizeof(uint64_t));
    st->prefill_mode = on ? 1 : 0;
    pthread_mutex_unlock(&st->mu);
}

/* record key in the prefill seen-set; returns 1 if it was already there
 * (second reference within this prefill). caller holds mu. */
static int seen_probe(fatcat_stream *st, uint64_t key)
{
    size_t i = (size_t)(splitmix(key) & st->seen_mask);
    while (st->seen[i] && st->seen[i] != key)
        i = (i + 1) & st->seen_mask;
    int was = st->seen[i] != 0;
    st->seen[i] = key ? key : 1;
    return was;
}

/* reclaim victim: policy-ordered evictable slot (busy slots rotated away).
 * Unlike pick_victim the slot is NOT marked inflight: shrink releases it. */
static slot_t *reclaim_victim(fatcat_stream *st)
{
    if (st->use_s3) {
        int q = st->main_len ? Q_MAIN : Q_SMALL;
        int guard = 2 * (int)(st->small_len + st->main_len) + 4;
        while (guard-- > 0) {
            int32_t h = q == Q_MAIN ? st->main_head : st->small_head;
            if (h < 0) { q = Q_SMALL + Q_MAIN - q; continue; }
            slot_t *s = &st->slots[h];
            if (s->inflight || s->pin || s->ref) {
                q_unlink(st, s); q_push(st, s, q);
                continue;
            }
            q_unlink(st, s);
            return s;
        }
        return NULL;
    }
    slot_t *best = NULL;
    double best_s = 1e300;
    int tries = (int)(st->n_slots < 16 ? st->n_slots : 16);
    for (int t = 0; t < tries; t++) {
        st->rng = splitmix(st->rng);
        slot_t *s = &st->slots[st->rng % st->n_slots];
        if (!s->valid || s->inflight || s->pin || s->ref) continue;
        double sc = slot_score(s, st->clock);
        if (sc < best_s) { best_s = sc; best = s; }
    }
    if (!best)
        for (size_t i = 0; i < st->n_slots; i++) {
            slot_t *s = &st->slots[i];
            if (!s->valid || s->inflight || s->pin || s->ref) continue;
            double sc = slot_score(s, st->clock);
            if (sc < best_s) { best_s = sc; best = s; }
        }
    return best;
}

size_t fatcat_stream_shrink(fatcat_stream *st, size_t keep_bytes)
{
    if (!st || !st->n_slots) return 0;
    pthread_mutex_lock(&st->mu);
    size_t freed = 0;
    while (st->used * st->slot_bytes > keep_bytes) {
        slot_t *s = reclaim_victim(st);
        if (!s) break;                       /* nothing evictable */
        size_t *e = tab_find(st, s->key);
        if (*e == (uint64_t)(s - st->slots) + 1) tab_erase(st, e);
        if (s->data && s->data != MAP_FAILED) {
            if (s->custom && st->dealloc)
                st->dealloc(s->data, st->slot_bytes, st->alloc_ctx);
            else if (!s->custom)
                munmap(s->data, st->slot_bytes);
            freed += st->slot_bytes;
        }
        s->data = NULL; s->custom = 0;
        s->valid = 0; s->key = 0; s->count = 0; s->len = 0;
        s->pin = 0; s->ref = 0; s->inflight = 0;
        st->used--;
        st->evictions++;
    }
    pthread_mutex_unlock(&st->mu);
    return freed;
}

const char *fatcat_stream_policy(const fatcat_stream *st)
{
    return (st && st->use_s3) ? "s3fifo" : "lfu";
}

void fatcat_stream_get_stats(const fatcat_stream *st, fatcat_stream_stats *out)
{
    memset(out, 0, sizeof *out);
    if (!st) return;
    pthread_mutex_lock((pthread_mutex_t *)&st->mu);
    out->hits = st->hits; out->misses = st->misses; out->evictions = st->evictions;
    out->hit_bytes = st->hit_bytes; out->miss_bytes = st->miss_bytes;
    out->wait_bytes = st->wait_bytes; out->cold_bytes = st->cold_bytes;
    out->ring_bytes = st->ring_bytes;
    out->ring_spec_bytes = st->ring_spec_bytes;
    out->prefill_skips = st->prefill_skips;
    out->n_slots = st->n_slots; out->used_slots = st->used;
    out->slot_bytes = st->slot_bytes;
    pthread_mutex_unlock((pthread_mutex_t *)&st->mu);
}

void fatcat_stream_reset_stats(fatcat_stream *st)
{
    if (!st) return;
    pthread_mutex_lock(&st->mu);
    st->hits = st->misses = st->evictions = st->hit_bytes = st->miss_bytes = 0;
    st->wait_bytes = st->cold_bytes = st->ring_bytes = 0;
    st->ring_spec_bytes = 0;
    pthread_mutex_unlock(&st->mu);
}
