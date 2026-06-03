# MatMulNBits gfx1151 Plan

This page captures the implementation plan for adding `MatMulNBits` support to
CK_DSL for the Qwen3.5-9B fp16-int4 group-size-32 workload. It is a planning
document, not a statement of shipped support.

## Target Workload

The model image lists 249 `MatMulNBits` calls, all with `M = seq_len` except
the decode-only `lm_head` case:

| Role | Count | M | N | K | Quant |
|---|---:|---|---:|---:|---|
| FFN gate/up projection | 64 | `seq_len` | 12288 | 4096 | int4 / g32 |
| attention out/in projection | 56 | `seq_len` | 4096 | 4096 | int4 / g32 |
| linear attention input | 48 | `seq_len` | 32 | 4096 | int4 / g32 |
| FFN down projection | 32 | `seq_len` | 4096 | 12288 | int4 / g32 |
| qkv projection | 32 | `seq_len` | 8192 | 4096 | int4 / g32 |
| full-attention kv projection | 16 | `seq_len` | 1024 | 4096 | int4 / g32 |
| lm_head decode-only | 1 | 1 | 248320 | 4096 | int4 / g32 |

The dense FP16 GEMM rows in the same image are already covered by
`universal_gemm`. The 3D patch-embedding convolution is not an exact match for
the existing 2D CK_DSL convolution instances.

## Contract

Add a new GEMM-family instance with a gfx1151-only v1 validator:

```text
C[M, N] = A[M, K] @ dequant(B[N, K], scales[N, K / 32])^T

A:      fp16, row-major [M, K]
B:      packed int4 weights, logical [N, K]
Scales: fp16 or fp32, [N, K / 32]
C:      fp16, row-major [M, N]
M:      runtime
N, K:   compile-time specialization fields
group:  compile-time, fixed to 32 for v1
```

The v1 kernel should support signed symmetric int4 first. If the exported model
requires unsigned int4 with zero-points, add optional zero-point parameters after
the symmetric path is correct and benchmarked.

## Why A New Instance

`block_scale_gemm` is close in spirit but is not the right starting point for
gfx1151:

- it currently rejects `quant_mode != "abquant"`;
- it rejects i4 mantissas in the shipped MFMA path;
- its optimized quantized path is CDNA/MFMA-oriented;
- gfx1151 CK_DSL already has verified fp16/bf16 WMMA, not int4 WMMA.

The gfx1151 path should therefore dequantize int4 weights to fp16 fragments and
feed `wmma_f32_16x16x16_f16`. This mirrors the high-level structure of the
existing C++ CK f16-i4 WMMA instances while staying inside CK_DSL's current
target-neutral MMA contract.

## Placement

Use `instances/common/matmul_nbits.py` only if the implementation is genuinely
target-neutral in the CK_DSL sense: resolve the MMA op from `ArchTarget`, use
the op's A/B/C layout maps for fragment coordinates, derive block size through
the GEMM-family helpers, and let `is_valid_spec(spec, arch)` reject every target
except gfx1151 until another backend path is implemented.

If the first implementation hard-codes gfx1151 lane math or directly resembles
the standalone reference `instances/gfx1151/wmma_gemm.py`, put it under
`instances/gfx1151/matmul_nbits.py` instead. The `common/` placement is a
conscious choice only for a unified-body kernel with a narrow initial validator,
not a place for gfx1151-specific fragment arithmetic.

## Kernel Families

Start with three explicit specializations instead of one generic slow path.

### Large-N Matmul

Use for:

- `N=12288, K=4096`
- `N=4096, K=4096`
- `N=4096, K=12288`
- `N=8192, K=4096`
- `N=1024, K=4096`

Initial geometry candidates:

```text
tile_m=64, tile_n=128, tile_k=32
tile_m=64, tile_n=128, tile_k=64
wave_size=32
atom=wmma_f32_16x16x16_f16
```

The hot loop should be specialized for `seq_len_tile = 64` rows. For dynamic
`seq_len`, drive the kernel over an outer M loop in multiples of 64:

```text
for m_outer in range(0, seq_len, 64):
  M_tile = min(64, seq_len - m_outer)
  run or enter hot loop for A[m_outer : m_outer + M_tile, :]
```

In the common case where `seq_len` is a multiple of 64, the hot loop stays
tail-free on M. The final partial tile should use the same bounds predicates as
the GEMM epilogue for correctness, but tuning and benchmarking should report the
multiple-of-64 path separately.

Load `A` as fp16 and load packed `B` as bytes. Reuse the existing
`helpers/i4_dequant.py` primitives (`unpack_i4_byte_to_pair_f32`,
`unpack_i4_byte_to_pair_i8`, and the packed-byte load conventions) instead of
duplicating nibble extraction. The new code should only add the WMMA-specific
conversion/staging step: f32 dequantized i4 values multiplied by the
`k // 32` scale, converted to fp16 fragments or staged fp16 LDS values for
WMMA consumption.

### Skinny-N Matmul

Use for `N=32, K=4096`.

Avoid wasting a 128-wide N tile. Start with a `64x32x64`-style geometry inspired
by the C++ f16-i4 WMMA configurations, then tune M tiling based on prompt sizes.

### Decode-Only GEMV

Use for `M=1, N=248320, K=4096`.

The first milestone can route this through the large-N WMMA path for
correctness, but a performant implementation should be a dedicated GEMV-style
kernel. One or more waves compute a row slice, stream packed weights and scales,
reduce across K, and store a contiguous span of logits.

## Packing And Layout

Require a CK_DSL-prepacked B layout for the fast path. The test harness should
include a host packer that accepts a logical `[N, K]` int4 matrix and writes the
layout consumed by the kernel.

For the first implementation, keep the layout simple enough to verify:

- pack two signed int4 values per byte;
- keep K contiguous inside each N row;
- group scales by `(n, k_group)` where `k_group = k // 32`;
- add a separate optimized prepack layout only after the baseline passes.

If matching CK's `permute_vectors_i4x4_b` layout becomes necessary for
bandwidth, add a second `packing="ck_i4x4"` spec mode and keep the simple layout
as the reference/debug path.

## Public Surface

Suggested spec fields:

```python
@dataclass(frozen=True)
class MatMulNBitsSpec(WarpTileBlockSizeMixin):
    name: str
    N: int
    K: int
    tile: TileSpec
    group_size: int = 32
    seq_len_tile: int = 64
    wave_size: int = 32
    block_size: int = 0
    scale_dtype: str = "fp16"
    zero_points: bool = False
    packing: str = "row_k_contiguous"
    family: str = "large_n"  # "large_n" | "skinny_n" | "decode_gemv"

    def __post_init__(self) -> None:
        self._init_block_size()
```

`tile` should use the existing GEMM `TileSpec` shape (`tile_m`, `tile_n`,
`tile_k`, `warp_m`, `warp_n`, `warp_k`, `warp_tile_m`, `warp_tile_n`,
`warp_tile_k`) so validation, kernel naming, grid helpers, and occupancy math
follow the same path as `UniversalGemmSpec`, `GroupedGemmSpec`, `BatchedGemmSpec`,
and `FlatMMSpec`. For the large-N and skinny-N families, `tile.tile_m` should be
64 for the first tuned path and `seq_len_tile` should remain 64.

Suggested helpers:

```text
is_valid_spec(spec, arch="gfx1151") -> (bool, str)
build_matmul_nbits(spec, arch="gfx1151") -> KernelDef
matmul_nbits_signature(spec)
matmul_nbits_grid(M, spec)
matmul_nbits_outer_tiles(seq_len, spec)
pack_i4_weights_for_matmul_nbits(weights, spec)
```

## Test Harness

Add a gfx1151 example harness:

```text
python/ck_dsl/examples/gfx1151/matmul_nbits_verify.py
```

Required behavior:

- generate random fp16 `A`;
- generate random int4 `B` and per-group scales;
- pack `B` into the kernel layout;
- compile and launch the selected specialization;
- compare against a torch/numpy reference:
  `A @ dequant(B, scales, zero_points).T`;
- support `--shape all`, `--m`, `--n`, `--k`, `--group-size 32`,
  `--family`, `--arch gfx1151`, `--seq-len-tile 64`, and `--verify`. Use
  lowercase `--m/--n/--k` to match the existing remote builder shape parser and
  the other gfx1151 harnesses (e.g. `wmma_gemm_verify.py`);
- the first tuned shape is a static `m=128, n=4096, k=4096`;
- verify both `M % 64 == 0` and tail cases where the outer M loop has a final
  partial tile.

The harness must also satisfy the remote-test example contract:

```text
python -m ck_dsl.examples.gfx1151.matmul_nbits_verify \
  --no-verify --output-dir <stage> --m 128 --n 4096 --k 4096
```

In `--no-verify` mode it should build locally and write one `*.hsaco` plus
`manifest.json` under `<stage>` without launching a GPU kernel. Verification
then happens through `ck_dsl.run_manifest` on a remote gfx1151 node.

Pytest coverage:

```text
python/test/test_ck_dsl.py
  - IR/lowering smoke tests for all three families

python/test/test_ck_dsl_numeric.py
  - gfx1151 numeric tests gated on available hardware
  - at least one shape from each family
```

## Remote Testing

Use the remote-test orchestrator in `benchmark/remote_test` for gfx1151 GPU
validation. The workflow builds artifacts locally, rsyncs them to the slurm
login node, and runs `ck_dsl.run_manifest --verify` under `srun` on a node with
the `GFX1151&MARKHAM` constraint.

Commands from the Composable Kernel Python root:

```bash
export PYTHONPATH=/workspace/rocm-libraries/projects/composablekernel/python

python -m ck_dsl.benchmark.remote_test.cli probe
python -m ck_dsl.benchmark.remote_test.cli all --arch gfx1151
```

Before MatMulNBits can use this flow, update
`benchmark/remote_test/config.py::ARCHES["gfx1151"]` to point at the new harness:

```python
"gfx1151": ArchProfile(
    arch="gfx1151",
    example_module="ck_dsl.examples.gfx1151.matmul_nbits_verify",
    example_args=[
        "--m", "128",
        "--n", "4096",
        "--k", "4096",
        "--group-size", "32",
        "--family", "large_n",
        "--seq-len-tile", "64",
    ],
    slurm_constraint="GFX1151&MARKHAM",
)
```

The harness uses lowercase `--m/--n/--k` so the existing remote builder shape
parser records the hint without changes. Keep the generated `run_spec.json`
shape aligned with the manifest so the remote runner invokes:

```text
python3 -m ck_dsl.run_manifest <hsaco> <manifest> --shape M,N,K --verify
```

Remote test matrix for the first pass:

```text
large_n hot loop: M=64,  N=4096,  K=4096
large_n multi:    M=128, N=4096,  K=4096
large_n tail:     M=96,  N=4096,  K=4096
skinny_n:         M=64,  N=32,    K=4096
decode_gemv:      M=1,   N=248320,K=4096
```

Start with one registered remote profile for the smoke shape. For the full
matrix, either run the CLI repeatedly with extra args or extend the remote-test
config to support named gfx1151 suites. Do not make remote testing a replacement
for local IR/lowering tests; it is the required GPU numeric gate for gfx1151.

## Benchmarking

Add:

```text
python/ck_dsl/examples/gfx1151/bench_matmul_nbits.py
```

Run the seven model shape groups with representative `M` values:

```text
decode:       M=1
hot loop:     M=64
medium:       M=128, 512
prefill:      M=1024+ (outer-looped in multiples of 64)
tail checks:  M=16, 32, 96 only for correctness / tail overhead
```

Report:

- median latency in microseconds;
- effective TOPS using `2 * M * N * K`;
- packed-weight, scale, A, and C bandwidth estimates;
- selected specialization and tile geometry;
- number of 64-row outer-loop tiles and whether a tail tile ran;
- CSV and JSON output paths.

When a CK C++ f16-i4 binary is available, optionally benchmark it as a reference
baseline, but do not make that a hard dependency for CK_DSL tests.

The benchmark script should also be runnable through the same remote staging
path, but benchmark execution should be a separate explicit mode from
`run_manifest --verify` so CI-style remote tests stay deterministic and short.

## Milestones

1. Add `MatMulNBitsSpec` using `WarpTileBlockSizeMixin`, signature/grid helpers,
   64-row outer-loop helpers, and a gfx1151-only validator.
2. Implement the simple packed layout and host-side pack/reference helpers.
   Reuse `helpers/i4_dequant.py` for packed-i4 unpack/dequant; add only the
   WMMA-specific f32-to-fp16 fragment/staging path.
3. Land a correctness-first WMMA tiled kernel for one large-N shape.
4. Cover all seven model shape groups with numeric tests, including `M=64`,
   multi-tile `M`, and a final partial M tile.
5. Wire the gfx1151 harness into `benchmark/remote_test` and pass remote
   `run_manifest --verify` for the hot-loop, multi-tile, tail, skinny-N, and
   decode smoke shapes.
6. Add skinny-N and decode-GEMV specializations.
7. Add benchmark script and record first gfx1151 remote results.
8. Update `SUPPORT_MATRIX.md`, `instances/index.md`, and `reference/file_index.md`
   once the instance ships.

## Open Questions

- Does the model export use signed symmetric int4, or unsigned int4 with
  zero-points?
- Are scales stored as fp16 or fp32?
- Is the weight layout fixed by ONNX Runtime `MatMulNBits`, or can the runtime
  prepack once into a CK_DSL-private layout?
- What prompt-length distribution should drive tuning for dynamic `seq_len`?
- Should the decode `lm_head` path return fp16 logits directly, or accumulate
  and store fp32 for downstream sampling?
