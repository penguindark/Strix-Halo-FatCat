#!/usr/bin/env python3
"""Pista A: attribute device reads to individual threads during a run.

Snapshots read_bytes from /proc/<pid>/task/<tid>/io for EVERY task on the
system (ours, iou-wrk io_uring workers, kworkers, desktop) and prints the
per-interval delta of the top readers. The mystery reader shows up here:
  - our fill worker thread  -> a read site the stream counters miss
  - iou-wrk-<pid>           -> io_uring reads are charged to the io-wq
                               worker, so ring_bytes and read_bytes
                               measure different tasks (no mystery at all)
  - some kworker            -> kernel-side reader (reclaim, readahead...)

usage: io_sampler.py WATCH_PID [INTERVAL_S]   (exits when WATCH_PID dies)
"""
import os, sys, time

def read_bytes_of(path):
    try:
        with open(path) as f:
            for line in f:
                if line.startswith("read_bytes:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return None

def comm_of(pid, tid=None):
    try:
        p = f"/proc/{pid}/task/{tid}/comm" if tid else f"/proc/{pid}/comm"
        with open(p) as f:
            return f.read().strip()
    except OSError:
        return "?"

def snapshot():
    snap = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        tdir = f"/proc/{pid}/task"
        try:
            tids = os.listdir(tdir)
        except OSError:
            continue
        for tid in tids:
            rb = read_bytes_of(f"{tdir}/{tid}/io")
            if rb:
                snap[(pid, tid)] = rb
    return snap

def alive(pid):
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False

def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    watch = sys.argv[1]
    iv = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
    prev = snapshot()
    t0 = time.time()
    while alive(watch):
        time.sleep(iv)
        cur = snapshot()
        rows = []
        for k, v in cur.items():
            d = v - prev.get(k, 0)
            if d > 1_000_000:
                rows.append((d, k[0], k[1]))
        rows.sort(reverse=True)
        stamp = time.time() - t0
        if rows:
            print(f"[{stamp:7.1f}s]", flush=True)
            for d, pid, tid in rows[:12]:
                print(f"    {d/1e6:9.1f} MB  pid {pid:>7} tid {tid:>7}  "
                      f"{comm_of(pid, tid)}", flush=True)
        prev = cur

if __name__ == "__main__":
    main()
