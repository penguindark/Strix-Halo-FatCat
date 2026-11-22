/* fatcat — raw io_uring wrapper implementation (see uring.h).
 * Uses only io_uring_setup/io_uring_enter syscalls and the shared ring
 * mappings; no liburing. Straightforward SQ-producer flow:
 *   read(): write SQE at sq_tail, publish sq_tail
 *   submit(): io_uring_enter(to_submit, wait_nr, GETEVENTS)
 *   reap():   consume [cq_head, cq_tail) and advance cq_head
 * Overflow handling: if the CQ overflow flag is set we re-enter with
 * GETEVENTS until it clears (kernel 5.9+ never overflows when the app
 * consumes, but we stay correct anyway). */
#define _GNU_SOURCE
#include <errno.h>
#include <linux/io_uring.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "uring.h"

struct fatcat_uring {
    int fd;
    /* SQ */
    struct io_uring_sqe *sqes;
    uint32_t *sq_array;
    uint32_t *sq_head, *sq_tail, *sq_ring_mask, *sq_ring_entries, *sq_dropped;
    void *sq_map, *cq_map, *sqe_map;
    size_t sq_map_sz, cq_map_sz, sqe_map_sz;
    uint32_t sq_issued;      /* sq_tail value last handed to the kernel */
    /* CQ */
    struct io_uring_cqe *cqes;
    uint32_t *cq_head, *cq_tail, *cq_ring_mask, *cq_ring_entries, *cq_overflow;
};

int fatcat_uring_open(fatcat_uring **out, unsigned entries)
{
    struct io_uring_params p;
    memset(&p, 0, sizeof p);
    int fd = (int)syscall(__NR_io_uring_setup, entries, &p);
    if (fd < 0) return -errno;

    fatcat_uring *r = calloc(1, sizeof *r);
    if (!r) { close(fd); return -ENOMEM; }
    r->fd = fd;

    r->sq_map_sz = p.sq_off.array + p.sq_entries * sizeof(uint32_t);
    r->sq_map = mmap(NULL, r->sq_map_sz, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQ_RING);
    r->cq_map_sz = p.cq_off.cqes + p.cq_entries * sizeof(struct io_uring_cqe);
    r->cq_map = mmap(NULL, r->cq_map_sz, PROT_READ | PROT_WRITE,
                     MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_CQ_RING);
    r->sqe_map_sz = p.sq_entries * sizeof(struct io_uring_sqe);
    r->sqe_map = mmap(NULL, r->sqe_map_sz, PROT_READ | PROT_WRITE,
                      MAP_SHARED | MAP_POPULATE, fd, IORING_OFF_SQES);
    if (r->sq_map == MAP_FAILED || r->cq_map == MAP_FAILED ||
        r->sqe_map == MAP_FAILED) {
        fatcat_uring_close(r);
        return -ENOMEM;
    }

    r->sqes = r->sqe_map;
    r->sq_array = (uint32_t *)((char *)r->sq_map + p.sq_off.array);
    r->sq_head = (uint32_t *)((char *)r->sq_map + p.sq_off.head);
    r->sq_tail = (uint32_t *)((char *)r->sq_map + p.sq_off.tail);
    r->sq_ring_mask = (uint32_t *)((char *)r->sq_map + p.sq_off.ring_mask);
    r->sq_ring_entries = (uint32_t *)((char *)r->sq_map + p.sq_off.ring_entries);
    r->sq_dropped = (uint32_t *)((char *)r->sq_map + p.sq_off.dropped);
    r->cqes = (struct io_uring_cqe *)((char *)r->cq_map + p.cq_off.cqes);
    r->cq_head = (uint32_t *)((char *)r->cq_map + p.cq_off.head);
    r->cq_tail = (uint32_t *)((char *)r->cq_map + p.cq_off.tail);
    r->cq_ring_mask = (uint32_t *)((char *)r->cq_map + p.cq_off.ring_mask);
    r->cq_ring_entries = (uint32_t *)((char *)r->cq_map + p.cq_off.ring_entries);
    r->cq_overflow = (uint32_t *)((char *)r->cq_map + p.cq_off.overflow);
    *out = r;
    return 0;
}

void fatcat_uring_close(fatcat_uring *r)
{
    if (!r) return;
    if (r->sq_map != MAP_FAILED && r->sq_map) munmap(r->sq_map, r->sq_map_sz);
    if (r->cq_map != MAP_FAILED && r->cq_map) munmap(r->cq_map, r->cq_map_sz);
    if (r->sqe_map != MAP_FAILED && r->sqe_map) munmap(r->sqe_map, r->sqe_map_sz);
    if (r->fd >= 0) close(r->fd);
    free(r);
}

int fatcat_uring_read(fatcat_uring *r, int fd, void *buf, unsigned len,
                      uint64_t off, uint64_t user_data)
{
    uint32_t tail = *r->sq_tail, mask = *r->sq_ring_mask;
    if (tail - *r->sq_head >= *r->sq_ring_entries) return -EAGAIN;
    struct io_uring_sqe *sqe = &r->sqes[tail & mask];
    memset(sqe, 0, sizeof *sqe);
    sqe->opcode = IORING_OP_READ;
    sqe->fd = fd;
    sqe->off = (long long)off;
    sqe->addr = (unsigned long)buf;
    sqe->len = len;
    sqe->user_data = user_data;
    r->sq_array[tail & mask] = tail & mask;   /* identity indexing */
    atomic_store_explicit((atomic_uint *)r->sq_tail, tail + 1,
                          memory_order_release);
    return 0;
}

int fatcat_uring_submit(fatcat_uring *r, unsigned wait_nr)
{
    unsigned to_submit = *r->sq_tail - r->sq_issued;
    unsigned flags = wait_nr ? IORING_ENTER_GETEVENTS : 0;
    long rc = syscall(__NR_io_uring_enter, r->fd, to_submit, wait_nr, flags,
                      NULL, (size_t)0);
    if (rc < 0) return -errno;
    r->sq_issued = *r->sq_tail;
    return (int)rc;
}

int fatcat_uring_reap(fatcat_uring *r, fatcat_uring_cb cb, void *ctx)
{
    int n = 0;
    uint32_t head = *r->cq_head, tail = *r->cq_tail, mask = *r->cq_ring_mask;
    while (head != tail) {
        struct io_uring_cqe *c = &r->cqes[head & mask];
        cb(c->user_data, c->res, ctx);
        head++;
        n++;
    }
    atomic_store_explicit((atomic_uint *)r->cq_head, head, memory_order_release);
    return n;
}
