# Chunkwise KDA Instances

Kimi Delta Attention prefill: a **gated delta-rule linear recurrence**, not
softmax attention. Each step applies a per-channel forget gate to a
`(head_k, head_v)` state matrix, then a rank-1 delta-rule write scaled by a
per-token `beta`. The chunkwise form factors that per-token recurrence into
dense `bf16` matmuls over fixed-length chunks, so it runs on the matrix core
instead of a token loop.

Like the attention family, this lives under `library/` (`library → platform`
one-way): kernel defs in `library/kernels/`, host drivers in
`library/builders/`, dispatch in `library/dispatch/`.

- `library/kernels/gfx942/kda_chunkwise.py` -- all three kernels
- `library/builders/gfx942/kda/` -- host drivers (see below)
- `library/dispatch/kda/` -- candidate registration

## Two paths, one recurrence

| Kernel | Builder | Grid | What it does |
|---|---|---|---|
| fused | `build_kda_chunk_fused` | `BH * parts` | Whole prefill in one kernel: builds each chunk's tiles in LDS and consumes them immediately, so only `q/k/g/beta` in and `o` plus the final state out touch HBM. |
| prep | `build_kda_chunk_prep` | `BH * parts * NC` | Split path phase 1. State-independent, so it is fully parallel over the sequence: one workgroup per chunk, six tiles per chunk written to HBM. |
| scan | `build_kda_chunk_scan` | `BH * parts` | Split path phase 2. Walks one stream's chunks in order, staging each chunk's tiles from HBM into LDS and running the same recurrence. |

`BH = batch * num_heads`, `parts` is the V partition below, `NC = seqlen / chunk`.

The two paths compute the same function and trade HBM traffic against
parallelism: fused avoids writing the tiles at all but serializes the tile work
inside one workgroup per stream, while prep+scan pays the round trip to get a
tile phase that is parallel over the whole sequence. Which one wins depends on
whether `BH * parts` alone fills the device. **There is no measured crossover
yet**, so dispatch defaults to fused and the split halves are opt-in rather
than heuristically routed.

## The V partition

gfx942 has 64 KiB of LDS per CU. The default scan spec uses 37,376 B and the
fused spec 62,016 B, so neither fits twice, and a DV=128 state mirror plus the
C16 tile builder does not fit once. One workgroup therefore owns
`KDA_PARTITION_HEAD_V = 64` value channels, and the host splits a logical
DV=128 head into `parts = 2` workgroups that share `q/k/g/beta` and each take
half of `v`/`o`. This is why the grids above carry a `parts` factor and why the
specs' `head_v` is 64 while the model's head is 128.

## ABI

```text
fused: (q, k, g, beta, v, o, h0, ht, scale: f32, nc: i32)
prep:  (q, k, g, beta, a, gk, gq, aqk, kt, dec, scale: f32)
scan:  (a, gk, gq, aqk, kt, dec, v, o, h0, ht, scale: f32, nc: i32)
```

`q/k/v/o` and the six intermediate tiles are `bf16`; `g`, `beta`, `dec`, `h0`
and `ht` are `f32`. `h0` is read only when `has_initial_state`, `ht` written
only when `store_final_state`; both are compile-time spec flags, not runtime
predicates. Every shape is baked into the kernel as a constant, so the ABI
carries pointers, the softmax scale, and the chunk count.

## Coverage

gfx942 only, `bf16` only. All three validators (`is_valid_spec`,
`is_valid_fused_spec`, `is_valid_scan_spec`) reject any other `arch` rather
than emitting a wrong-layout kernel: CDNA3 lacks the K-packed `bf16` atoms and
the `ds_read_*_tr_*` transposing loads that a gfx950 path would use, so the
swizzled LDS store/load pairing here is specific to `a_per_lane == 4`.

- `chunk` in `{16, 32}` (`KDA_CHUNK_SIZES`); 32 exceeds the LDS budget for the
  default prep spec, so 16 is what ships
- `head_k` a multiple of the MFMA K step, and `block_size` a multiple of
  `head_k` at least twice over, for the grouped cumulative sum
- `head_v` a multiple of 64 (the partition above)
- `seqlen` a multiple of `chunk`. **No varlen path**: a ragged batch must be
  padded by the caller
- prefill only; there is no KDA decode kernel here

## Host drivers

`library/builders/gfx942/kda/` carries two host lanes over the *same* packed
layout, so either can validate the other:

- `kda_chunk_fused.py`, `kda_chunk_prep.py`, `kda_chunk_split.py` -- torch GPU
  experiments: parity sweeps and local benchmarks against a torch reference
- `hostpack.py` -- torch-free numpy pack/unpack plus a float64 token-serial
  oracle for the gated delta rule, independent of the fused factorization
- `manifest.py` -- registers the `kda_chunk_fused_bf16` manifest-runner kind, so
  `python -m rocke.run_manifest <hsaco> <manifest> --verify` runs the pack and
  the correctness check on a remote node with no torch installed

## Dispatch

`library/dispatch/kda/` registers one candidate per kernel in `KDA_REGISTRY`:

| Candidate | algorithm | priority |
|---|---|---|
| `kda_gfx942_chunk_fused` | `chunk_fused` | 10 (default) |
| `kda_gfx942_chunk_prep` | `chunk_prep` | 20 (opt-in) |
| `kda_gfx942_chunk_scan` | `chunk_scan` | 20 (opt-in) |

`KdaRequest` carries the *logical* `head_v`; the candidates apply the partition
when they build the spec and when they report their grid. The capabilities
declare arch, dtype, chunk length and the partition relation as data; anything
the validators compute from a spec -- the LDS budget, the cumsum row-group
divisibility, the wave/panel cover, the scan partition -- stays in the residual
predicate. `require_binding` is off: the launch-side pack lives in `hostpack.py`
behind the manifest runner, and lifting it to a `ProblemBinding` is separate
work.

## Tests

| File | Needs |
|---|---|
| `library/tests/test_kda_chunkwise_gfx942_spec.py` | comgr (skips without) |
| `library/tests/test_kda_gfx942_golden.py` | nothing -- pins emitted IR SHA |
| `library/tests/test_kda_hostpack.py` | numpy -- oracle + pack round trip |
| `library/tests/dispatch/kda/` | nothing -- registration, gates, geometry |
| `library/tests/test_kda_chunkwise_gfx942_numeric.py` | a local gfx942 device |
