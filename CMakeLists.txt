# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Dario Deledda
cmake_minimum_required(VERSION 3.28)

project(fatcat
  VERSION 0.0.1
  DESCRIPTION "Streaming MoE LLM engine for Mistral-Large-4-1T-A52B on AMD Strix Halo"
  LANGUAGES C)

# --- C23 ---
set(CMAKE_C_STANDARD 23)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

# --- Platform guard: Strix Halo only ---
if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
  message(WARNING "FatCat targets AMD Strix Halo (x86_64 + gfx1151); "
                  "other platforms are unsupported and untested.")
endif()

# --- Strix Halo CPU tuning (Zen 5: AVX-512, VNNI, BF16) ---
add_library(fatcat_flags INTERFACE)
target_compile_options(fatcat_flags INTERFACE
  -march=znver5 -mtune=znver5)
find_package(Threads REQUIRED)
find_package(OpenMP COMPONENTS C)
if(OpenMP_C_FOUND)
  target_link_libraries(fatcat_flags INTERFACE OpenMP::OpenMP_C)
endif()
target_link_libraries(fatcat_flags INTERFACE m)

# --- Optional ROCm 10.1 (TheRock gfx1151) ---
# Default location of the local TheRock install; override with -DROCM_PATH=...
set(ROCM_PATH "/home/user/rocm-10.1/install" CACHE PATH "ROCm installation prefix")
option(FATCAT_WITH_GPU "Build the gfx1151 GPU backend (rocBLAS/hipBLASLt)" ON)

set(FATCAT_HAS_ROCM OFF)
if(FATCAT_WITH_GPU)
  find_library(ROCM_ROCBLAS rocblas HINTS "${ROCM_PATH}/lib" NO_DEFAULT_PATH)
  find_library(ROCM_HIPBLASLT hipblaslt HINTS "${ROCM_PATH}/lib" NO_DEFAULT_PATH)
  find_path(ROCM_INCLUDE_DIR hip/hip_runtime.h HINTS "${ROCM_PATH}/include" NO_DEFAULT_PATH)
  if(ROCM_ROCBLAS AND ROCM_INCLUDE_DIR)
    set(FATCAT_HAS_ROCM ON)
    message(STATUS "ROCm found at ${ROCM_PATH} — GPU backend enabled")
  else()
    message(STATUS "ROCm not found at ${ROCM_PATH} — building CPU-only")
  endif()
endif()

# --- HIP kernels (hipcc, gfx1151) ---
set(HIPCC "${ROCM_PATH}/bin/hipcc")
set(FATCAT_HAS_HIP OFF)
if(FATCAT_WITH_GPU AND EXISTS "${HIPCC}")
  set(FATCAT_HAS_HIP ON)
  message(STATUS "hipcc found — compiling HIP kernels for gfx1151")
endif()

# --- Sources (files are added as phases progress) ---
add_library(fatcat_io STATIC
  src/io/gguf.c)
target_include_directories(fatcat_io PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(fatcat_io PRIVATE fatcat_flags)

add_library(fatcat_model STATIC
  src/model/dequant.c
  src/model/ml4.c
  src/model/quant_tables.c
  src/backend/cpu/cpu.c
  src/backend/cpu/avx512.c
  src/stream/stream.c
  src/stream/uring.c
  src/gpu/gpu.c)
target_include_directories(fatcat_model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(fatcat_model PRIVATE fatcat_flags fatcat_io Threads::Threads)

if(FATCAT_HAS_HIP)
  # compile HIP kernels with hipcc into an object, then archive it into
  # fatcat_model so intra-archive symbols (quant tables) resolve cleanly.
  # -ffp-contract=off: bit-identity between the gemv and batched gemm
  # kernels requires identical FP evaluation; leaving FMA contraction on
  # lets the compiler fuse mul/add differently in the two kernels
  # (observed 1-2 ULP drift on Q8_0, amplified to cos 0.9987 on logits).
  add_custom_command(OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/kernels.o
    COMMAND ${HIPCC} -O3 -ffp-contract=off --offload-arch=gfx1151 -x hip -c
            ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/kernels.hip
            -o ${CMAKE_CURRENT_BINARY_DIR}/kernels.o
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/kernels.hip
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})
  set_source_files_properties(${CMAKE_CURRENT_BINARY_DIR}/kernels.o
    PROPERTIES GENERATED TRUE)
  target_sources(fatcat_model PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/kernels.o)
  target_link_directories(fatcat_model INTERFACE "${ROCM_PATH}/lib")
  target_link_libraries(fatcat_model INTERFACE amdhip64)
else()
  target_sources(fatcat_model PRIVATE src/gpu/gpu_stub.c)
endif()

add_library(fatcat_core STATIC
  src/core/engine.c)
target_include_directories(fatcat_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(fatcat_core PRIVATE fatcat_flags fatcat_model fatcat_io)

add_executable(fatcat
  src/main.c)
target_include_directories(fatcat PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(fatcat PRIVATE fatcat_flags fatcat_io)

if(FATCAT_HAS_ROCM)
  target_compile_definitions(fatcat PRIVATE FATCAT_WITH_GPU=1)
  target_include_directories(fatcat PRIVATE ${ROCM_INCLUDE_DIR})
  target_link_directories(fatcat PRIVATE "${ROCM_PATH}/lib")
  if(ROCM_ROCBLAS)
    target_link_libraries(fatcat PRIVATE "${ROCM_ROCBLAS}")
  endif()
  if(ROCM_HIPBLASLT)
    target_link_libraries(fatcat PRIVATE "${ROCM_HIPBLASLT}")
  endif()
endif()

# --- Tools ---
add_executable(gguf-inspect tools/gguf-inspect.c)
target_link_libraries(gguf-inspect PRIVATE fatcat_io fatcat_flags)

add_executable(gguf-slice tools/gguf-slice.c)
target_link_libraries(gguf-slice PRIVATE fatcat_io fatcat_flags)

add_executable(expert-profiler tools/expert-profiler.c)
target_link_libraries(expert-profiler PRIVATE fatcat_model fatcat_io fatcat_flags)

add_executable(fatcat-run tools/fatcat-run.c)
target_link_libraries(fatcat-run PRIVATE fatcat_core fatcat_model fatcat_io fatcat_flags)

add_executable(logit-dump tools/logit-dump.c)
target_link_libraries(logit-dump PRIVATE fatcat_model fatcat_io fatcat_flags)

add_executable(tensor-dump tools/tensor-dump.c)
target_link_libraries(tensor-dump PRIVATE fatcat_model fatcat_io fatcat_flags)

add_executable(cpu-bench tools/cpu-bench.c)
target_link_libraries(cpu-bench PRIVATE fatcat_model fatcat_io fatcat_flags)

add_executable(gpu-kbench tools/gpu-kbench.c)
target_link_libraries(gpu-kbench PRIVATE fatcat_model fatcat_io fatcat_flags)

# --- Tests ---
enable_testing()
add_executable(test_gguf tests/test_gguf.c)
target_compile_definitions(test_gguf PRIVATE _GNU_SOURCE)  # mkdtemp
target_link_libraries(test_gguf PRIVATE fatcat_io fatcat_flags)
add_test(NAME gguf_reader COMMAND test_gguf)

add_executable(test_dequant tests/test_dequant.c)
target_link_libraries(test_dequant PRIVATE fatcat_model fatcat_flags)
add_test(NAME dequant_kernels COMMAND test_dequant ${CMAKE_CURRENT_SOURCE_DIR}/tests/data/dequant_golden.bin)

add_executable(test_cpu_kernels tests/test_cpu_kernels.c)
target_link_libraries(test_cpu_kernels PRIVATE fatcat_model fatcat_flags)
add_test(NAME cpu_kernels COMMAND test_cpu_kernels)

add_executable(test_stream tests/test_stream.c)
target_link_libraries(test_stream PRIVATE fatcat_model fatcat_flags Threads::Threads)
add_test(NAME stream_tier COMMAND test_stream)

add_executable(test_topk tests/test_topk.c)
target_link_libraries(test_topk PRIVATE fatcat_model fatcat_flags)
add_test(NAME topk_select COMMAND test_topk)

add_executable(gputest tests/gputest.c)
target_link_libraries(gputest PRIVATE fatcat_model fatcat_io fatcat_flags)

add_executable(gpu-absorb-test tests/gpu-absorb-test.c)
target_link_libraries(gpu-absorb-test PRIVATE fatcat_model fatcat_io fatcat_flags)
