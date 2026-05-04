# Example 06: FMHA BWD via rocm_ck

Demonstrates shipping the full CK Tile FMHA backward pipeline as device-only
code objects via kpack archives. Three kernel families cover the complete
backward pass:

1. **OGradDotO** -- `D = rowSum(dO * O) * p_undrop` (per-token scalar)
2. **DqDkDv** -- main backward (5 GEMMs: dQ, dK, dV gradients)
3. **ConvertDQ** -- split-K dQ accumulation reduce + type conversion (deterministic mode)

The pipeline runs: OGradDotO -> DqDkDv -> (ConvertDQ if deterministic).

## Dependencies

- **ROCm** with HIP support (`/opt/rocm`)
- **System packages**: `libmsgpack-cxx-dev`, `libzstd-dev`
- **Python packages**: `msgpack` (`pip install msgpack`)

## Build

```bash
cmake -B build -S . -G Ninja \
    -DCMAKE_CXX_COMPILER=/llvm-project/build/bin/clang++ \
    -DCMAKE_PREFIX_PATH=/opt/rocm \
    -DGPU_TARGETS="gfx942" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
ninja -C build
```

## Run

```bash
./build/kpack_rocm_ck_fmha_bwd build/kernels.kpack
```

## Variant Surface (19 kernels)

### OGradDotO (5 variants)

| Variant | dtype | hdim_v | mode  | padding | Notes |
|---------|-------|--------|-------|---------|-------|
| `fp16_d128_batch`      | FP16 | 128 | batch | padded  | Most common config |
| `bf16_d128_batch`      | BF16 | 128 | batch | padded  | BF16 dtype axis |
| `fp16_d64_batch`       | FP16 | 64  | batch | padded  | Smaller hdim axis |
| `fp16_d128_group`      | FP16 | 128 | group | padded  | Group mode axis |
| `fp16_d128_batch_npad` | FP16 | 128 | batch | none    | No-padding axis |

### DqDkDv (10 variants)

| Variant | dtype | hdim | mode  | features | Notes |
|---------|-------|------|-------|----------|-------|
| `fp16_d128_batch`             | FP16 | 128 | batch | plain                | Baseline (numerically verified) |
| `bf16_d128_batch`             | BF16 | 128 | batch | plain                | dtype axis (numerically verified) |
| `fp16_d128_batch_cmask`       | FP16 | 128 | batch | causal mask          | Compilation proof |
| `fp16_d128_batch_det`         | FP16 | 128 | batch | deterministic        | Compilation proof (host runner does not yet invoke the OGradDotO -> DqDkDv -> ConvertDQ pipeline end-to-end) |
| `fp16_d128_group`             | FP16 | 128 | group | plain                | Compilation proof (group-mode host runner not implemented; runtime path skipped) |
| `fp16_d128_batch_ebias`       | FP16 | 128 | batch | elementwise bias     | Compilation proof |
| `fp16_d128_batch_alibi`       | FP16 | 128 | batch | ALiBi bias           | Compilation proof |
| `fp16_d128_batch_ebias_dbias` | FP16 | 128 | batch | bias + bias gradient | Compilation proof |
| `fp16_d128_batch_dropout`     | FP16 | 128 | batch | dropout              | Compilation proof |
| `fp16_d128_batch_cmask_det`   | FP16 | 128 | batch | causal + det         | Compilation proof |

### ConvertDQ (4 variants)

| Variant | dtype | hdim | mode  | Notes |
|---------|-------|------|-------|-------|
| `fp16_d128_batch_det` | FP16 | 128 | batch | Paired with det DqDkDv |
| `fp16_d128_group_det` | FP16 | 128 | group | Paired with det DqDkDv |
| `bf16_d128_batch_det` | BF16 | 128 | batch | BF16 dtype axis |
| `bf16_d128_group_det` | BF16 | 128 | group | BF16 + group axis |

## Design

### Signature / Algorithm Split

- **Signature** = problem shape: dtype, head dimensions, batch/group mode
- **Algorithm** = feature flags + tuning: bias, mask, dropout, deterministic,
  padding mode, occupancy hint
- **Config** = Signature + Algorithm (user-facing)
- **Spec** = validated descriptor (structural type, NTTP-safe)

Each kernel family has its own Signature/Algorithm/Config/Spec structs.
`makeSpec()` is overloaded per Config type (unambiguous).

### Generic Args with Named Slot Constants

All kernel families use the same `rocm_ck::Args` struct (1408 bytes). Tensor
pointers, dimensions, and strides are packed into `Args::tensors[]` slots.
Problem-level scalars (scale, num_head_q, etc.) use `Args::scalars[]` slots.

Named slot constants (e.g., `fmha_bwd_dqdkdv_slots::Q = 0`,
`::K = 1`, `::V = 2`, ...) prevent off-by-one errors in the 50-parameter
DqDkDv kernel argument mapping.

### CK Tile Template Chains

**OGradDotO:**
```
FmhaBwdOGradDotOKernel<Pipeline>
  -> BlockFmhaBwdOGradDotO<PipelineProblem>
    -> TileFmhaBwdOGradDotOTraits<spad, dvpad, block_per_cu>
```

**DqDkDv:**
```
FmhaBwdDQDKDVKernel<Pipeline, KGradEpi, VGradEpi, QGradEpi>
  -> BlockFmhaBwdDQDKDVPipeline<PipelineProblem>
    -> BlockFmhaBwdPipelineProblem<15 data types, Shape, Mask, Dropout, Traits>
      -> TileFmhaBwdTraits<padQ, padV, BiasEnum, hasBiasGrad, blockPerCu>
      -> TileFmhaBwdShape<BlockTile, 5x(BlockWarps, WarpTile), maxSeqLenQ>
```

**ConvertDQ:**
```
FmhaBwdConvertQGradKernel<Pipeline>
  -> BlockFmhaBwdConvertQGrad<PipelineProblem>
    -> TileFmhaBwdConvertQGradTraits<spad, dpad, block_per_cu>
```

## File Structure

The kernel-family headers live under `include/rocm_ck/ops/fmha_bwd/` and are
shared by tests and other examples; only the example-specific files (registry,
.hip instantiations, runner, build glue) live in this directory.

```
include/rocm_ck/ops/fmha_bwd/
  common.hpp                -- Shared types: FmhaMode, FmhaBiasType, padding docs
  ograd_dot_o_spec.hpp      -- OGradDotO: Signature/Algorithm/slots/makeSpec (SHARED)
  dqdkdv_spec.hpp           -- DqDkDv: Signature/Algorithm/slots/makeSpec (SHARED)
  convert_dq_spec.hpp       -- ConvertDQ: Signature/Algorithm/slots/makeSpec (SHARED)
  ograd_dot_o_api.hpp       -- OGradDotO: grid_size + validateArgs (HOST ONLY)
  dqdkdv_api.hpp            -- DqDkDv: grid_size + validateArgs (HOST ONLY)
  convert_dq_api.hpp        -- ConvertDQ: grid_size + validateArgs (HOST ONLY)
  ograd_dot_o_dev.hpp       -- OGradDotO device bridge (DEVICE ONLY, CK Tile dep)
  dqdkdv_dev.hpp            -- DqDkDv device bridge (DEVICE ONLY, CK Tile dep)
  convert_dq_dev.hpp        -- ConvertDQ device bridge (DEVICE ONLY, CK Tile dep)

examples/06_rocm_ck_fmha_bwd/
  rocm_fmha_bwd_registry.hpp            -- 3 variant arrays + 3 findVariant() overloads
  fmha_bwd_ograd_dot_o_*.hip            -- 5 OGradDotO variant instantiations
  fmha_bwd_dqdkdv_*.hip                 -- 10 DqDkDv variant instantiations
  fmha_bwd_convert_dq_*.hip             -- 4 ConvertDQ variant instantiations
  main.cpp                              -- 3-kernel pipeline runner + CPU reference
  CMakeLists.txt                        -- Build: compile .hip -> .hsaco, pack, build host exe
  pack.py                               -- Pack .hsaco files into kpack archive
  INSIGHTS.md                           -- Design decisions and lessons learned
```
