# MoE Rank-Staged Reduction Epilogues

## Contents

- [Scope](#scope)
- [Reduction and RMSNorm](#reduction-and-rmsnorm)
- [Reduction and Scatter](#reduction-and-scatter)
- [Dispatch and verification](#dispatch-and-verification)

## Scope

[`moe_rank_reduce.py`](../../python/rocke/instances/common/moe_rank_reduce.py)
contains the single-device arithmetic half of two communication epilogues. The
input is a contiguous, locally addressable workspace with layout
`[world_size, rows, width]`.

The kernels do **not** transport data between devices. rocKE has no collective
runtime or peer-memory protocol, so a caller must fill the rank-major workspace
before launch. Keeping that precondition in the public contract avoids treating
a local reduction as an all-reduce or reduce-scatter implementation.

## Reduction and RMSNorm

`build_moe_rank_reduce_rmsnorm` sums each row across the baked
`world_size`, performs one workgroup-wide sum-of-squares reduction, and writes:

```text
reduced[row, column] = sum_rank partials[rank, row, column]
output[row, column] =
    reduced[row, column]
    * rsqrt(sum_column(reduced**2) / width + epsilon)
    * gamma[column]
```

`MoeRankReduceRMSNormSpec.fp32_internal=False` rounds the rank sum to the
storage dtype before the square and normalization. This reproduces a narrow
collective output consumed by a separate RMSNorm. The opt-in true mode keeps
the sum in f32 through normalization.

ABI:

```text
(Partials, Gamma, Y, rows, width, eps)
```

## Reduction and Scatter

`build_moe_rank_reduce_scatter` reads only the destination rank's contiguous
column shard from every staged partial:

```text
shard_width = width / world_size
Y[row, local_column] =
    sum_source_rank Partials[
        source_rank,
        row,
        rank * shard_width + local_column
    ]
```

The caller validates the runtime `rank` argument. No atomic operation is
needed because each output element has one workgroup owner.

ABI:

```text
(Partials, Y, rows, width, rank)
```

## Dispatch and verification

[`moe_rank_reduce.py`](../../python/rocke/dispatch/families/moe_rank_reduce.py)
registers separate `rmsnorm` and `scatter` candidates. Dispatch selects the
widest legal vector geometry for the requested full row or output shard and
records that collective transport is caller-owned in the result explanation.

CPU structure and lowering coverage:

```text
python -m pytest tests/instances/test_moe_rank_reduce.py
python -m pytest tests/dispatch/dispatch_tests/moe/test_moe_rank_reduce.py
```

The Python and C engine emitters are paired in
`tests/instances/parity/moe_rank_reduce_emit.{py,c}` for byte-identity
verification.
