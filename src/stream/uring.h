/* SPDX-License-Identifier: MIT */
/* SPDX-FileCopyrightText: 2026 Dario Deledda */
/* fatcat — minimal raw io_uring wrapper (no liburing dependency).
 * Internal to src/stream. Single-threaded use: one fill worker submits and
 * reaps; callers never touch the ring directly. */
#ifndef FATCAT_URING_H
#define FATCAT_URING_H

#include <stdint.h>

typedef struct fatcat_uring fatcat_uring;

/* entries: power-of-two ring depth. Returns 0 on success. */
int fatcat_uring_open(fatcat_uring **out, unsigned entries);
void fatcat_uring_close(fatcat_uring *r);

/* Queue a read: buf[0..len) <- fd at file offset off. user_data is returned
 * in the completion. Non-blocking; returns 0, or -EAGAIN if the SQ is full. */
int fatcat_uring_read(fatcat_uring *r, int fd, void *buf, unsigned len,
                      uint64_t off, uint64_t user_data);

/* Submit queued requests to the kernel. wait_nr: block until at least this
 * many (unconsumed) completions are posted. Returns submitted count or -errno. */
int fatcat_uring_submit(fatcat_uring *r, unsigned wait_nr);

/* Reap all posted completions, invoking cb(user_data, result, ctx) per CQE
 * (result: bytes read or -errno). Returns number reaped. */
typedef void (*fatcat_uring_cb)(uint64_t user_data, int32_t res, void *ctx);
int fatcat_uring_reap(fatcat_uring *r, fatcat_uring_cb cb, void *ctx);

#endif /* FATCAT_URING_H */
