# Gated DeltaNet

The gfx950 Gated DeltaNet (GDN) family implements single-token decode and
chunkwise prefill. It supports `bf16` and `f16` activation and state dtypes.

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


## Prefill

Many tokens at once, for the sequence that has just arrived. This is a
different shape of problem from decode: there is no single-token step to
serialize, so the work is done **chunkwise** and the recurrent state is carried
across chunks by a scan.

GDN prefill is **not a separate kernel**. It is the shared KDA chunkwise pair
run in a different gate mode:

- `library/kernels/gfx950/kda_chunkwise.py` -- emitter, `gate_kind="gdn"`
- `library/builders/gfx950/kda/gdn_prefill.py` -- host driver and fp64 oracle
- `library/dispatch/gdn/prefill_gfx950.py` -- the two candidates
- `library/dispatch/gdn/prefill_common.py` -- request, ABI version, vocabulary
- `library/benchmarks/gfx950/gdn/sweep_prefill_value_splits.py` -- `value_splits` sweep

### Why one emitter, not two

The chunkwise algorithm is identical for KDA and GDN -- chunk factorization, triangular
solve, state scan. Only the decay gate differs: KDA's is per channel and floored, GDN's is a
scalar per `(token, head)` broadcast across `DK` and unbounded below. Both are log-domain;
the equations, their ranges and the derivation are owned by
[`ALGORITHM.md`](../../../library/builders/gfx950/gdn/ALGORITHM.md) SS2.2-2.3 and are
deliberately not restated here.

Forking the emitter would have duplicated the chunk factorization, the triangular solve and
the state scan -- three pieces of real algebra -- to vary one expression.

`gate_kind` defaults to `"kda"`, so every KDA spec emits byte-identical code:
the golden fixture gains two GDN cases and **no existing case SHA moves**.

The unbounded GDN gate is the reason `test_gdn_prefill_decay_guard.py` exists:
KDA's gate is floored by `lower_bound`, GDN's is not, so the supported envelope
is enforced rather than assumed. The bound is `EXP2_CLAMP / (log2(e) * chunk/2)`
-- **5.46** per token at `chunk=32` -- computed by `decay_limit_for_chunk` in
[`gdn_prefill.py`](../../../library/builders/gfx950/kda/gdn_prefill.py) and
raised on at launch. It is the point the hardware `exp2` clamp begins to
saturate, i.e. where the result stops being exact; past it the kernel returns a
bounded, finite, WRONG answer, and the final state stays clean while the output
degrades, so a loop validating only its carried state sees nothing.

### Two launches, no fused default

`dispatch_gdn_prefill` rejects `algorithm="auto"`. A GDN prefill is
`chunk_prep` then `chunk_scan`; resolving `auto` to one half would dispatch
half a computation and return successfully. The caller pins each half:

```python
prep = dispatch_gdn_prefill(GdnPrefillRequest(..., algorithm="chunk_prep"))
scan = dispatch_gdn_prefill(GdnPrefillRequest(..., algorithm="chunk_scan"))
```

`value_splits` bands a head's value extent across workgroups, turning a `BH`-wide
grid into `BH x value_splits`. It buys parallelism when `batch_heads` is small,
which is when the scan's natural grid starves.

The bands do not overlap and need **no reduction** afterwards -- there is no
cross-workgroup traffic. What a split costs is **redundant reads**: the per-chunk
tiles are addressed by chunk only, so every band re-reads the full tile set for
every chunk. The quantity that grows is roughly `tile_bytes x value_splits x
num_chunks` per `(batch, head)`, which is why the win reverses as `batch_heads`
grows and the sweep script exists to find the crossover.
