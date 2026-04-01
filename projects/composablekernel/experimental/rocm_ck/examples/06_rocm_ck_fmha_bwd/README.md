# Example 06: FMHA BWD via rocm_ck

Demonstrates shipping the full CK Tile FMHA backward pipeline as device-only
code objects via kpack archives. Three kernel families cover the complete
backward pass:

1. **OGradDotO** — `D = rowSum(dO * O) * p_undrop` (per-token scalar)
2. **DqDkDv** — main backward (5 GEMMs: dQ, dK, dV gradients)
3. **ConvertDQ** — split-K dQ accumulation reduce + type conversion (deterministic mode)

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

## Variant Surface (12 kernels)

### OGradDotO (5 variants)

| Variant | dtype | hdim_v | mode  | padding | Notes |
|---------|-------|--------|-------|---------|-------|
| `fp16_d128_batch`      | FP16 | 128 | batch | padded  | Most common config |
| `bf16_d128_batch`      | BF16 | 128 | batch | padded  | BF16 dtype axis |
| `fp16_d64_batch`       | FP16 | 64  | batch | padded  | Smaller hdim axis |
| `fp16_d128_group`      | FP16 | 128 | group | padded  | Group mode axis |
| `fp16_d128_batch_npad` | FP16 | 128 | batch | none    | No-padding axis |

### DqDkDv (5 variants)

| Variant | dtype | hdim | mode  | features | Notes |
|---------|-------|------|-------|----------|-------|
| `fp16_d128_batch`       | FP16 | 128 | batch | plain        | Baseline (numerically verified) |
| `bf16_d128_batch`       | BF16 | 128 | batch | plain        | dtype axis (numerically verified) |
| `fp16_d128_batch_cmask` | FP16 | 128 | batch | causal mask  | Compilation proof |
| `fp16_d128_batch_det`   | FP16 | 128 | batch | deterministic| Compilation proof |
| `fp16_d128_group`       | FP16 | 128 | group | plain        | Group mode (bridge incomplete) |

### ConvertDQ (2 variants)

| Variant | dtype | hdim | mode  | Notes |
|---------|-------|------|-------|-------|
| `fp16_d128_batch_det` | FP16 | 128 | batch | Paired with det DqDkDv |
| `fp16_d128_group_det` | FP16 | 128 | group | Paired with det DqDkDv |

## Design

### Signature / Algorithm Split

- **Signature** = problem shape: dtype, head dimensions, batch/group mode
- **Algorithm** = feature flags + tuning: bias, mask, dropout, deterministic,
  padding mode, occupancy hint
- **Config** = Signature + Algorithm (user-facing)
- **Spec** = validated descriptor (structural type, NTTP-safe)

Each kernel family has its own Signature/Algorithm/Config/Spec structs.
`make_spec()` is overloaded per Config type (unambiguous).

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

```
rocm_fmha_bwd_common.hpp              — Shared types: FmhaMode, FmhaBiasType, padding docs
rocm_fmha_bwd_ograd_dot_o_spec.hpp    — OGradDotO: Signature/Algorithm/slots/make_spec (SHARED)
rocm_fmha_bwd_dqdkdv_spec.hpp         — DqDkDv: Signature/Algorithm/slots/make_spec (SHARED)
rocm_fmha_bwd_convert_dq_spec.hpp     — ConvertDQ: Signature/Algorithm/slots/make_spec (SHARED)
rocm_fmha_bwd_ograd_dot_o_api.hpp     — OGradDotO: grid_size helper (HOST ONLY)
rocm_fmha_bwd_dqdkdv_api.hpp          — DqDkDv: grid_size helper (HOST ONLY)
rocm_fmha_bwd_convert_dq_api.hpp      — ConvertDQ: grid_size helper (HOST ONLY)
rocm_fmha_bwd_api.hpp                 — Unified include-all for host code (HOST ONLY)
rocm_fmha_bwd_ograd_dot_o_dev.hpp     — OGradDotO device bridge (DEVICE ONLY, CK Tile dep)
rocm_fmha_bwd_dqdkdv_dev.hpp          — DqDkDv device bridge (DEVICE ONLY, CK Tile dep)
rocm_fmha_bwd_convert_dq_dev.hpp      — ConvertDQ device bridge (DEVICE ONLY, CK Tile dep)
rocm_fmha_bwd_registry.hpp            — 3 variant arrays + 3 findVariant() overloads
fmha_bwd_ograd_dot_o_*.hip            — 5 OGradDotO variant instantiations
fmha_bwd_dqdkdv_*.hip                 — 5 DqDkDv variant instantiations
fmha_bwd_convert_dq_*.hip             — 2 ConvertDQ variant instantiations
main.cpp                              — 3-kernel pipeline runner + CPU reference
CMakeLists.txt                         — Build: compile .hip -> .hsaco, pack, build host exe
pack.py                                — Pack .hsaco files into kpack archive
INSIGHTS.md                            — Design decisions and lessons learned
```
