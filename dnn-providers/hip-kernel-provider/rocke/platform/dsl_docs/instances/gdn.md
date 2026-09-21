# Gated DeltaNet

The gfx950 Gated DeltaNet (GDN) instance implements single-token decode over a
paged recurrent state. It supports `bf16` and `f16` activation and state
dtypes. Prefill is a separate part of the GDN family and is not implemented by
this instance.

This page is the public instance reference, following the same structure as
[`kda.md`](kda.md). For the equations and GPU thread mapping, see
[`library/builders/gfx950/gdn/ALGORITHM.md`](../../../library/builders/gfx950/gdn/ALGORITHM.md).
For build, correctness, benchmark, and tuning commands, see
[`library/builders/gfx950/gdn/README.md`](../../../library/builders/gfx950/gdn/README.md).

## Source

| Area | Path |
|---|---|
| Kernel spec, validator, and emitter | `library/kernels/gfx950/gdn_decode.py` |
| Host driver and fp32 reference | `library/builders/gfx950/gdn/gdn_decode.py` |
| Tile sweep | `library/builders/gfx950/gdn/tune.py` |
| Dispatch | `library/dispatch/gdn/` |

## Tensor contract

All tensors are contiguous and row-major.

| Tensor | Shape | Dtype |
|---|---|---|
| `query`, `key` | `[B, 1, num_k_heads, head_k_dim]` | `dtype` |
| `value`, `out` | `[B, 1, num_v_heads, head_v_dim]` | `dtype` |
| `a`, `b` | `[B, 1, num_v_heads]` | `dtype` |
| `dt_bias` | `[num_v_heads]` | `dtype` |
| `A_log` | `[num_v_heads]` | `f32` |
| `read_indices`, `write_indices` | `[B]` | `i32` |
| `state` | `[pool, num_v_heads, head_v_dim, head_k_dim]` | `state_dtype` |

The sequence length is one. `read_indices` and `write_indices` address the
state pool; `-1` marks an idle batch entry. The kernel writes `out` and updates
`state` in place.

## Spec and validation

`GdnDecodeSpec` carries the head geometry, dtypes, `use_qk_l2norm`, and three
tile knobs: `num_warps`, `warp_threads_k`, and `blocks_per_v_dim`.
`simple=True` selects the one-thread-per-state-row reference implementation.

`is_valid_spec(spec, arch)` is the shared authority for direct builds and
dispatch. It checks the target wave size, data types, head grouping and
alignment, workgroup size, and divisibility of the K and V tiles.
`kernel_name()` includes every field that changes generated code.

## Dispatch

```python
from dispatch.gdn import GdnDecodeRequest, dispatch_gdn_decode

result = dispatch_gdn_decode(GdnDecodeRequest(batch=16, arch="gfx950"))
```

The result contains the selected `GdnDecodeSpec`, kernel identity, signature,
and launch grid and block. An explicit `spec_id` selects a registered tile for
benchmarking or replay. Candidate admission ends in `is_valid_spec()`, so
dispatch cannot offer a tile that the kernel rejects.

## Validation

Run from `dnn-providers/hip-kernel-provider/rocke`:

```bash
PYTHONPATH=library:platform/python python3 -m pytest \
  library/tests/test_gdn_decode_spec.py \
  library/tests/test_gdn_decode_golden.py \
  library/tests/dispatch/gdn/test_gfx950_wiring.py
```

The on-device output and recurrent-state checks are in
`library/tests/test_gdn_decode_gfx950_numeric.py`.
