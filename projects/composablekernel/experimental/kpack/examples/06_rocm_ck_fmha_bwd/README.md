# Example 06: FMHA BWD OGradDotO via kpack

Demonstrates shipping the CK Tile FMHA backward `OGradDotO` kernel family
(`D = rowSum(dO * O) * p_undrop`) as device-only code objects via kpack archives.

This is the simplest FMHA backward kernel — it computes a per-token scalar from
the output gradient and output tensors. It validates the kpack pattern for
multi-parameter CK Tile kernels with both batch and group (variable-length) modes.

## Dependencies

- **ROCm** with HIP support (`/opt/rocm`)
- **System packages**: `libmsgpack-cxx-dev`, `libzstd-dev`
- **Python packages**: `msgpack` (`pip install msgpack`)

## Build

```bash
cmake -B build -S . -G Ninja \
    -DCMAKE_CXX_COMPILER=/opt/rocm/llvm/bin/clang++ \
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

## Variant Surface

| Variant | dtype | hdim_v | mode  | pad_s | pad_dv | Notes |
|---------|-------|--------|-------|-------|--------|-------|
| `fp16_d128_batch`      | FP16 | 128 | batch | yes | yes | Most common config |
| `bf16_d128_batch`      | BF16 | 128 | batch | yes | yes | BF16 dtype axis |
| `fp16_d64_batch`       | FP16 | 64  | batch | yes | yes | Smaller hdim axis |
| `fp16_d128_group`      | FP16 | 128 | group | yes | yes | Group mode axis |
| `fp16_d128_batch_npad` | FP16 | 128 | batch | no  | no  | No-padding axis |

## Design

### Signature / Algorithm Split

- **Signature** (`FmhaBwdOGradDotOSignature`): WHAT — dtype, hdim_v, mode
- **Algorithm** (`FmhaBwdOGradDotOAlgorithm`): HOW — padding flags, occupancy hint, block size
- **Config** = Signature + Algorithm (user-facing)
- **Kernel** = validated descriptor (structural type, NTTP-safe)

### CK Tile Template Chain

```
FmhaBwdOGradDotOKernel<Pipeline>
  -> BlockFmhaBwdOGradDotO<PipelineProblem>
    -> BlockFmhaBwdOGradDotOPipelineProblem<OType, dOType, DType, bm0, hdim_v, is_group, Traits>
      -> TileFmhaBwdOGradDotOTraits<spad, dvpad, block_per_cu>
```

### Two Args Structs

Batch and group modes use separate flat argument structs
(`FmhaBwdOGradDotOBatchArgs` / `FmhaBwdOGradDotOGroupArgs`) that match CK Tile's
internal Kargs layout exactly. ABI is verified by `static_assert` on sizeof/alignof
in the device header.

## File Structure

```
rocm_fmha_bwd_api.hpp         — Host+device API (NO CK Tile dependency)
rocm_fmha_bwd_dev.hpp         — Device-side: config NTTP -> CK Tile template chain
rocm_fmha_bwd_registry.hpp    — Variant table + findVariant()
fmha_bwd_ograd_dot_o_*.hip    — Per-variant kernel instantiations (~15 lines each)
main.cpp                       — Host loader + verification
CMakeLists.txt                 — Build: compile .hip -> .hsaco, pack, build host exe
pack.py                        — Pack .hsaco files into kpack archive
```
