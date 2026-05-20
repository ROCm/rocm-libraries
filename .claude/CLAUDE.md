# CLAUDE.md

  

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

  

## Project Overview

  

Composable Kernel (CK) is AMD's high-performance GPU kernel library for machine learning workloads. It uses HIP C++ with a tile-based programming model and tensor coordinate transformation techniques.

  

**Two implementations exist:**

- **CK Tile** (`include/ck_tile/`) - Modern tile-programming API, preferred for new development

- **Legacy CK** (`include/ck/`) - Older implementation, still supported

  

## Git Commit Rules

  

- **禁止在 commit 中留下 Claude Code 痕迹**：所有 commit message 不得包含 `Co-Authored-By: Claude`、`🤖 Generated with Claude Code` 等任何 AI 工具署名

- **"把代码提一下"** = commit + push（不只是 commit，要同时 push 到远端）

  

## Build Commands

  

### Standard Build (Development)

  

```bash

cd /root/rocm-libraries/projects/composablekernel

rm -rf build && mkdir build && cd build

cmake --preset dev -DGPU_TARGETS="gfx950" -DBUILD_TESTING=OFF -G Ninja ..

cmake --build . --target <target> -j$(nproc)

```

  

必须用 `--preset dev`（设了关键 flags：`-fPIE -Wno-gnu-line-marker -fbracket-depth=1024`、`CMAKE_HIP_COMPILER`、`BUILD_DEV=ON`）。不要用裸 `cmake ..`。

  

**必须用 `-G Ninja`**：Make 的 `cmake_depends` 是单线程的，CK 项目规模下即使 nothing to do 也要 11 分钟。Ninja 只需 2 秒。

  

始终使用 ROCm 7.2.2 / clang 22.0.0 编译。

  

### Key CMake Options

  

```bash

-D GPU_TARGETS="gfx950"          # Required to build tests/examples

-D CMAKE_BUILD_TYPE=Release      # or Debug, RelWithDebInfo

-D BUILD_DEV=ON                  # Enables -Werror (strict mode)

-D DTYPES="fp16;fp32;fp8"        # Restrict data type instances

-D BUILD_TESTING=OFF             # Skip test instances

```

  

### SA3 开发加速：只生成 sageattnv3 实例

  

```bash

# 取消注释 example/ck_tile/01_fmha/CMakeLists.txt 中的 --filter 行:

#   --filter \\*sageattn\\*

# 然后重新 cmake + build

```

  

### Running Tests

  

```bash

./bin/test_ck_tile_fmha_fwd_fp16          # 直接运行

ctest -R test_ck_tile_fmha_fwd_fp16       # CTest

make smoke                                 # 快速测试 (<30s)

make regression                            # 长测试 (>=30s)

```

  

### Code Formatting

  

```bash

# CK 有自己的 pre-commit config:

pre-commit run --config projects/composablekernel/.pre-commit-config.yaml --files <files>

# hooks: clang-format, ruff, copyright header, remod

```

  

## Workflow Preferences

  

- **直接行动，无需确认**：编译、运行测试、修改代码等操作直接执行

- **编译 timeout**：CK 编译耗时长，编译和测试命令使用 `run_in_background=true` 后台运行

- **编译并行度**：始终使用 `-j$(nproc)` 全核编译

  

## Debugging Rules

  

- **先读代码推导，后编译验证**：constexpr 链可以从模板参数 + 头文件直接算出，不需要 printf。用 `static_assert` 验证数值，编译失败即知结果

- **二分法隔离**：遇到 bug 用 constexpr if / flag 快速隔离到最小子系统，确认根因后再精修

- **不要凭记忆假设 constexpr 值**：必须从代码或 static_assert 验证。基于错误假设的分析会浪费大量时间

- **保存 baseline 二进制**：实验前 `cp bin/xxx /tmp/baseline_xxx`，A/B 对比直接用副本，不反复 rebuild

- **debug 问题必须详细记录**：每个遗留问题写入 memory 需包含：具体现象、触发条件、根因分析、当前 workaround、失败方案、后续方向。记到"下次打开就能接着干"的程度

- **测试跑多轮**：UT / 正确性验证至少跑 3-5 轮，不同 batch/head/seqlen 组合。单次 pass 不能确认正确性，竞态 bug 偶发。关注 occ >= 2 场景

  

## Optimization Rules

  

- **优化前确认目标硬件**：先查明 GPU 型号（gfx942/950/951）、VGPR/LDS/CU/HBM/MFMA 参数，用具体数值推导 occupancy/LDS 预算/bandwidth bound，再决定方向

- **每种优化建独立 branch**：commit 源码 + 编译出的二进制，比较时直接切 branch 跑，不反复 revert + rebuild

  

## Profiling Rules

  

- **profiling 跳过正确性验证**：benchmark/PMC/ATT 运行时用 `-v=0`（仅性能数据），正确性验证单独跑

- **benchmark 前检查 GPU 占用**：跑性能前先 `rocm-smi --showuse` 确认各 GPU 利用率。如果 >10% 说明有其他任务在抢资源，性能数据不可靠。性能报告中必须附上当时的 GPU 占用情况

- **rocprofv3 --att 禁止加 `-o`**：加 `-o` 只输出原始 .att 文件，RCV 打不开。不加 `-o` 时自动生成 `ui_output_agent_*/` 目录结构

  ```bash

  rocprofv3 --att --att-activity 8 --att-target-cu 1 --att-buffer-size 0x10000000 \

    -- ./<binary> <args>

  tar czf <output>.tar.gz ui_output_agent_*_dispatch_*/

  ```

  

## Architecture

  

### Four-Layer Operator Hierarchy

  

Every CK Tile operator (GEMM, FMHA, etc.) follows this hierarchy:

  

1. **Warp** (`ops/<op>/warp/`): Single-warp MMA operations wrapping hardware instructions (MFMA on gfx9, WMMA on gfx11/12)

2. **Block** (`ops/<op>/block/`): Multi-warp tile computation. Name encodes memory placement: `areg_bsmem_creg` = A in registers, B in shared memory, C in registers

3. **Pipeline** (`ops/<op>/pipeline/`): Main loop and data movement strategy. FMHA pipelines: `qr_ks_vs` = Q in registers, K/V in shared memory

4. **Kernel** (`ops/<op>/kernel/`): Top-level HIP kernel template. Composes pipeline + epilogue, defines grid/block and `Kargs`

  

### Problem + Shape + Traits + Policy Pattern

  

- **Problem**: Aggregates all data types and compile-time config

- **Shape**: Compile-time tile dimensions as `sequence<>` values (`kM0`, `kN0`, `kK0`)

- **Traits**: Boolean compile-time feature flags (padding, bias, dropout, etc.)

- **Policy**: Controls implementation details (load alignment, async copy, unroll)

  

### Core Abstractions (`include/ck_tile/core/`)

  

- **`sequence<Ns...>`**: Compile-time integer sequence for shapes, strides

- **`tile_distribution`**: Describes how a tile is distributed across threads

- **`static_distributed_tensor`**: A tile held in registers

- **`tile_window`**: View into global/shared memory with coordinate transforms

- **Tile APIs**: `load_tile()`, `store_tile()`, `shuffle_tile()`, `slice_tile()`, `sweep_tile()`

  

### Data Types

  

Custom numeric types in `include/ck_tile/core/numeric/`:

- `fp16_t`, `bf16_t`, `fp32_t`, `int8_t`, `fp8_t`, `e8m0_t`, `pk_fp4_t` (packed fp4)

- MX types require gfx950: guarded with `CK_USE_NATIVE_MX_SUPPORT`

  

### Architecture-Specific Guards

  

- `CK_USE_XDL` — MFMA instructions (gfx9)

- `CK_USE_WMMA` — WMMA instructions (gfx11/12)

- `CK_USE_GFX950` — gfx950-specific paths

- `CK_USE_NATIVE_MX_SUPPORT` — Native MX hardware on gfx950

  

## Coding Conventions

  

- **Indent**: 4 spaces, no tabs; **column limit**: 100 characters

- **Header guard**: `#pragma once`

- **Template parameters**: `PascalCase_` with trailing underscore; resolved aliases drop underscore

- **Compile-time constants**: `kPascalCase` prefix (`kBlockSize`, `kPadSeqLenQ`)

- **Namespace**: `ck_tile` (all lowercase)

- **Host/device functions**: `CK_TILE_HOST_DEVICE` macro

- Every file starts with AMD copyright + MIT SPDX identifier

  

## Key Files for Common Tasks

  

| Task | Key Files |

|------|-----------|

| Add a new FMHA variant | `ops/fmha/pipeline/`, `ops/fmha/kernel/fmha_fwd_kernel.hpp`, `example/ck_tile/01_fmha/` |

| Add a new warp GEMM shape | `ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp` |

| Add a new data type | `core/numeric/`, update `DTYPES` filter in `test/CMakeLists.txt` |

| Add a new block GEMM variant | `ops/gemm/block/block_gemm_*` |

| Reference CPU implementation | `include/ck_tile/host/reference/` |

| Add a new test | `test/ck_tile/<operation>/` |