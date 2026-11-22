# Build Instructions

Status: P0 stub. Authoritative status: `../project.md`.

## Requirements
- OS: Linux on AMD Strix Halo (x86_64, Zen 5).
- Compiler: GCC/Clang with C23 support.
- CMake ≥ 3.28.
- Optional GPU: ROCm 10.1 TheRock gfx1151 at `/home/user/rocm-10.1/install`
  (override with `-DROCM_PATH=...`).
- Planned: `liburing-dev` (P3 streaming).

## Configure & build
```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build -j
./build/fatcat        # P0 smoke binary: prints version
```

GPU backend is auto-detected; disable with `-DFATCAT_WITH_GPU=OFF`.

## Notes
- All code compiles with `-march=znver5` (Strix Halo only).
- ROCm is linked from the local TheRock install; CMake links rocBLAS and
  hipBLASLt when present.
