# SDPA Reference Implementations

The SDPA (scaled dot-product attention) references are the oracles every hipDNN attention engine is
checked against. This document says what they implement, how they are built, and how we know they are
right, so they can be trusted as the thing everything else is compared to.

There are two references, written independently and checked against each other and against PyTorch:

| | CPU reference | GPU reference |
|---|---|---|
| Code | `projects/hipdnn/test_sdk/include/hipdnn_test_sdk/utilities/CpuFpReferenceSdpa.hpp` (dense), `CpuFpReferenceSdpaRagged.hpp` (packed) | `dnn-providers/integration-tests/gpu-ref/include/hipdnn-gpu-ref/GpuFpReferenceSdpa.hpp`, `src/GpuFpReferenceSdpa.cpp`, kernels `gpu-ref/kernels/sdpa/` |
| Graph plans | `test_sdk/.../cpu_graph_executor/detail/SdpaFwdPlan.hpp`, `SdpaBwdPlan.hpp`, `SdpaRaggedFwdPlan.hpp` | `integration-tests/src/harness/gpu-graph-executor/detail/GpuSdpa*Plan.hpp` |
| Executes | host C++, one parallel task per query row | hipRTC kernels, one thread per output element |
| Speed | small and medium shapes (the golden lane caps it at 4M-element Q/K, quick tier) | every shape the bundles use |

The contract is cuDNN's. Where a use case disagrees with cuDNN, the decision is recorded in
[SdpaSemanticsRegister.md](SdpaSemanticsRegister.md) (SEM-xxx entries), and the references and their
tests encode it.

## 1. Design

### 1.1 One pipeline, named options

Each reference is one forward function and one backward function, `forward`/`backward` (CPU) and
`fprop`/`bprop` (GPU). Every feature is a named field of an options struct (`SdpaFwdOptions`,
`SdpaBwdOptions`, and their GPU twins), off by default. No feature has its own copy of the softmax.
The forward computes, for each query row `(b, h, sq)`:

1. **Score** of each live key (`skv < seq_len_kv`): `s = (q · k) * scale`, then softcap, `+ bias`,
   `+ ALiBi`. A key that the diagonal band or the block mask removes is `-inf`. This step is
   `scoreRow` on the CPU and `sdpa_common::scoreFromDot` / `bandMasked` / `blockMasked` on the GPU
   (`kernels/sdpa/GpuRefSdpaCommon.hpp`). The backward uses the same code.
2. **Softmax** over the row, with an optional sink logit in the max and the denominator:
   `Max = max(max_j s_j, sink)`, `Sum_exp = Σ exp(s_j - Max) (+ exp(sink - Max))`,
   `LSE = Max + log(Sum_exp)`. A row with no live key and no sink gives `O = 0` and `LSE = -inf`.
3. **P @ V**: `P_j = exp(s_j - Max) / Sum_exp`, optionally stored as fp8 (cuDNN `sdpa_fp8`, see
   below), then times the dropout mask and scale.

Everything is computed in FP32, directly and without tiling. The softmax runs as three passes over the
row (max, sum, weighted sum), so no reference ever depends on a tile size or a rescaling trick an
engine might use.

### 1.2 Features layered around the core

Features that change the inputs or outputs, rather than the math, are thin layers around the core
call:

| Layer | What it does | Where |
|---|---|---|
| fp8 dequantize | fp8 Q/K/V times their descales (scalar or `[B, H_kv, 1, 1]`) to FP32, then the float core runs | CPU `forward`, GPU `fprop` + `sdpaDequantRef` |
| fp8 output scaling | core writes FP32 `O_f`; `amax_o = max \|O_f\|`, `O = O_f * scale_o` in O's dtype | CPU `forward`, GPU `sdpaScaleOutRef` |
| Paged K/V | CPU: gather pages into dense K/V (`gatherPagedKv`). GPU: the kernel reads pages in place through the table | plans / kernel |
| Ragged (RFC 0014) | native packed forward (bf16, fp8); every other ragged graph is rewritten to its dense padded form, run dense, and packed back (`RaggedSdpaDensify.hpp`) | both executors |
| Philox dropout | the engine's `rng_dump` keep mask is applied as an explicit dropout mask with scale `1 / (1 - p)` | plans (`RngDumpHandoff.hpp` in the harness) |

### 1.3 Things deliberately not shared

The CPU and GPU references do not share code: they share only the semantics in the register. That is
what makes comparing them meaningful. Within the GPU reference, the forward and backward kernels share
`GpuRefSdpaCommon.hpp`, so a score feature is written once per reference.

## 2. What is supported

The table lists every attribute of the hipDNN SDPA graph nodes (`sdpa_attributes.fbs`,
`sdpa_backward_attributes.fbs`). Both references support everything marked Yes. The plans decline
(report "not applicable") anything else, field by field. They never ignore an attribute.

### 2.1 Forward (`SdpaAttributes`)

| Feature | Support (CPU and GPU) | Semantics |
|---|---|---|
| Q/K/V/O dtypes | FP32; FP16; BF16 (O the same, or FP32 for FP16/BF16 inputs); FP8 E4M3, E4M3 FNUZ, E5M2, E5M2 FNUZ inputs with BF16 output or output in the input's fp8 format | compute in FP32 |
| Layout | any strides: BHSD, BSHD, interleaved K/V rows | |
| GQA / MQA | `H_q` divisible by `H_k` and `H_v` (independent) | |
| `D_qk != D_v` | Yes | |
| Scale (`attn_scale_value` or scale tensor, device or pass-by-value) | Yes; absent = 1.0 | SEM-013 |
| Additive bias (`attn_mask`) | FLOAT, HALF, BF16; rank 1-4, broadcast on size-1 dims | SEM-006 |
| Diagonal band (`left_bound`, `right_bound`, `diagonal_alignment`; deprecated causal flags) | Yes; bottom-right measured from each batch's live lengths | SEM-001, 003, 004, 005 |
| Padding (`seq_len_q/kv`, `padding_mask`) | INT32 `[B, 1, 1, 1]`, both; pad rows give `O = 0`, `LSE = -inf` | SEM-008, 010 |
| Paged K/V (`page_table_k/v`, `max_seq_len_kv`) | K, V or both paged, each with its own block size; table INT32 `[B, 1, ceil(max/block), 1]`; extent inferred as cuDNN does when unset; needs padding | SEM-009 |
| Ragged tensors (`ragged_offset` on any primary) | native forward for BF16 and FP8 E4M3; everything else (other dtypes, bias, sinks, paged K/V, backward) through the dense rewrite | RFC 0014 |
| Attention sink (`sink_token`) | FLOAT `[1, H_q, 1, 1]` | SEM-022 |
| ALiBi (`alibi_mask`) | cuDNN slopes, `+ (skv - sq) * slope(h)`; top-left causal only (as cuDNN) | SEM-021 |
| Block-sparse mask (`block_mask`) | UINT8, one bit per 128 x 128 tile | SEM-025 |
| Dropout, explicit mask (`dropout_mask`, `dropout_scale`) | FLOAT `[B or 1, H or 1, Sq, Skv]` | SEM-015 |
| Dropout, Philox (`dropout_probability`, seed, offset) | only with an `rng_dump` output (the engine's keep mask is applied) | SEM-015 |
| fp8 Q/K/V descales | scalar or `[B, H_kv, 1, 1]` FLOAT (a paged side needs a scalar on the GPU) | |
| fp8 softmax / output scaling (`scale_s`, `descale_s`, `scale_o`, `amax_s`, `amax_o`) | one-element FLOAT; `scale_s` with `descale_s`; fp8 inputs only; not with dropout | SEM-024 |
| Outputs: `stats` (LSE), `Max`, `Sum_exp` | FLOAT `[B, H, Sq, 1]` (packed when ragged) | SEM-011, 023 |
| `generate_stats`, `mma_core_mode`, `implementation` | accepted; engine hints with no meaning for an FP32 oracle | |

### 2.2 Backward (`SdpaBackwardAttributes`)

| Feature | Support (CPU and GPU) |
|---|---|
| dtypes | FP32, FP16, BF16 (all tensors one dtype); bias FLOAT/HALF/BF16 |
| Q, K, V, O, dO, LSE (rank 3 or 4) to dQ, dK, dV | Yes; `D = rowsum(dO * O)`; P from the LSE (or recomputed when absent) |
| GQA / MQA, `D_qk != D_v` | Yes (dK/dV accumulate over the query heads) |
| Scale, band, padding, ALiBi | as forward |
| Bias and `dbias` | `dbias` = unscaled dS summed over every broadcast dim (SEM-014) |
| Dropout, explicit mask (`dropout_mask`, `dropout_scale`, `dropout_scale_inv`) | cuDNN's composition (SEM-015) |
| Ragged | through the dense rewrite |
| Philox dropout | **No**: the backward node has no `rng_dump` (needs an API change, below) |

### 2.3 Reference-only extensions

These are not expressible in the graph API. The reference functions take them so tests and adapters can use them.

- **Softcap** `c * tanh(s / c)`, forward and backward (SEM-020). An attribute is pending a team decision.
- **GPU `probabilityMode`** (bf16 RTNE/RTZ rounding of P before P @ V). This models specific provider kernels. The graph plans always use FP32 P (SEM-012).

### 2.4 Not supported, and why

Everything below needs a schema or frontend change first:

| Missing | cuDNN | Blocked on |
|---|---|---|
| Backward Philox dropout | backward `RNG_DUMP` | an `rng_dump` field on `SdpaBackwardAttributes` |
| Backward sinks (`dsink`) | `set_sink_token` / `set_dsink_token` | backward schema fields |
| fp8 backward | `sdpa_fp8_backward` node | a node with the fp8 descale/scale/amax fields |
| `score_mod` (FlexAttention) | `set_score_mod` | a scope decision |

## 3. How we know the references are right

No single check is enough, so the evidence is layered. Each layer catches a different kind of mistake.

### 3.1 Simple by construction

The references compute the definition directly:
- FP32 throughout.
- No tiling, no online softmax, no early exit.
- One code path per feature, in the order the register gives.

A reviewer can check each step against the cuDNN reference code it cites (cudnn-frontend
`test/python/sdpa/fp16_ref.py`, `fp8_ref.py`). The fp8 layers wrap the float core instead of changing
it. When a semantics question comes up (left_bound, ALiBi sign, sink in the LSE, fp8 tiling), the
answer goes into the register first and then into the code and its tests.

### 3.2 Closed-form oracles

Inputs whose exact output is known analytically, in `TestCpuFpReferenceSdpaOracles.cpp`:
- uniform scores (O = mean of V);
- one raised score (known LSE);
- a single live key (O copies V);
- dead rows (O = 0, LSE = -inf);
- sinks on uniform and fully masked rows;
- fp8 scaling with one key (catches a skipped P quantization);
- amax_s with and without a dominating sink.

`TestCpuFpReferenceSdpa.cpp` adds hand-computed FP64 cases for scale, masks, bands, softcap, ALiBi and dropout.

### 3.3 Properties (metamorphic checks)

Two graphs that must give the same result because one feature is defined in terms of another:
- GQA = MHA with repeated K/V heads.
- Band = `-inf` bias.
- Block mask = `-inf` bias on its cleared tiles.
- Scale = pre-scaled Q.
- Padding = truncated batches.
- Paged = dense padded.
- Sink = an extra zero-value key.
- Ragged = per-batch dense (and the dense rewrite = the native ragged plan).
- The deprecated causal flags = explicit bounds.

These pin how features compose. A single closed form cannot do that.

### 3.4 Finite-difference gradient checks (backward)

`TestCpuFpReferenceSdpaGradCheck*` checks dQ, dK, dV and dBias against central differences of the
forward. It covers plain, GQA, band, padding, bias, dropout, softcap and ALiBi. The backward is
therefore checked against the forward, not just against another backward.

### 3.5 CPU vs GPU parity

The two references are independent implementations: host C++ and HIP kernels, with no shared code.
- **Quick:** every feature is compared at small shapes, with outputs pre-filled with NaN so an
  unwritten element fails. Files: `tests/gpu-ref/TestGpuFpReferenceSdpa*.cpp`, `TestGpuSdpa*Plan.cpp`,
  `TestGpuRaggedSdpaDensify.cpp`.
- **Standard:** medium shapes, with hundreds to over a thousand keys and several features combined
  (`Standard/TestGpuSdpaFwdSweepFp32`, `Standard/TestGpuSdpaBwdFp32`).
- **fp8 device codecs:** the device decoders and encoders are checked bit-for-bit against the host
  types, on every code and on rounding ties.

### 3.6 External oracle: PyTorch golden bundles

Golden bundles hold inputs and outputs computed by PyTorch. Each reference runs the bundle graph and
must match.

- **Generators** (`dnn-providers/integration-tests/reference-data-scripts/`):
  - `generate_sdpa_fwd_golden.py`: PyTorch MATH backend.
  - `generate_sdpa_bwd_golden.py`: explicit FP32 backward, cross-checked by FP64 autograd.
  - `generate_sdpa_fwd_paged_golden.py`: gather-then-dense, with a naive per-row loop check in its tests.
  - All share masks and helpers through `utilities/sdpa_common.py`.
- **Exactness checks against cuDNN's own reference code:**
  - The fp8-scaling mode reproduces cuDNN's `fp8_ref.compute_ref` bit for bit (checked while the mode was written).
  - The block-mask expansion and the ALiBi slopes are copied from cuDNN's `fp16_ref.py`.

What the generators cover, and the gaps:

| Feature | PyTorch oracle | How |
|---|---|---|
| dtypes FP16/BF16, GQA, bands, padding, Sq != Skv, stats | yes | `F.scaled_dot_product_attention` (MATH) + additive masks |
| bias + dBias, explicit dropout (backward) | yes | explicit FP32 backward + FP64 autograd check |
| ragged | yes | pack/unpack around dense |
| paged K/V, sinks | yes | gather pages; sink as an extra logit column |
| fp8 descales, fp8 scaling (E4M3, E5M2) | yes | dequantize; cuDNN `fp8_ref` formula in torch |
| block mask, ALiBi | yes | as an additive bias (cuDNN's expansion / slopes) |
| Max / Sum_exp | no bundle; properties + CPU/GPU parity only | (simple; could be added to the forward generator) |
| softcap | no; not in the graph API | closed forms, gradcheck, parity |
| Philox dropout | no oracle by design | the engine's RNG stream is engine-defined; the reference applies the engine's `rng_dump` |
| backward ALiBi, backward ragged | no bundle | gradcheck + parity; the backward generator rejects them today |

### 3.7 Mutation testing

The tests are themselves checked by injecting plausible bugs into the references and confirming some
test fails. Examples:
- bottom-right offset taken from the maximum lengths instead of the live ones;
- swapped alignments;
- GQA mapped to the wrong head;
- dBias scaled;
- `dropout_scale_inv` dropped or squared;
- dS sign flipped;
- the dead-row LSE check removed.

A mutant that survives means a missing test, and a test is added. Each new feature commit also checks
that its tests fail on the code before the change.

### 3.8 Applicability: decline, never ignore

Each plan reads and gates every schema field it could be given: dtype, shape, rank, presence and
combination. Each rule has a decline test, and the CPU and GPU plans are tested with the same graphs
(`SdpaFwdTestGraphs.hpp`). A graph a reference cannot run exactly is reported "not applicable", never
run approximately.

## 4. Test tiers

Tiers are GTest name prefixes, mapped to CTest labels by
`dnn-providers/integration-tests/test_categories.yaml`. Golden bundles take their tier from their
directory (`integration-test-bundles/{quick,standard,comprehensive,full}`). Labels cascade:
`ctest -L standard` runs quick and standard.

| Tier | When | SDPA reference content |
|---|---|---|
| **Quick** | every PR (pre-commit, CI) | All CPU reference tests (`hipdnn_test_sdk_tests`): closed forms, properties, gradchecks, plan applicability. All small-shape CPU-vs-GPU parity tests (`hipdnn_gpu_ref_tests`, `hipdnn_integration_tests_unit_tests`). Every `quick/` golden bundle on the GPU lane, and on the CPU lane when Q and K have at most 4M elements. Generator Python tests (skip without PyTorch). |
| **Standard** | PR / pre-check-in | Medium-shape CPU-vs-GPU sweeps (`Standard/TestGpuSdpaFwdSweepFp32`, `Standard/TestGpuSdpaBwdFp32`). `standard/` golden bundles (GPU lane). |
| **Comprehensive** | nightly | `comprehensive/` golden bundles (GPU lane): longer sequences, more head/dim combinations |
| **Full** | release / on demand | `full/` golden bundles (GPU lane) |

Why it is split this way:
- The CPU reference is too slow for long rows, so it is checked exhaustively at small sizes and against
  the GPU at medium sizes.
- The GPU reference is checked against PyTorch at every size the bundles hold.
- Correctness at large sizes therefore rests on the GPU reference plus PyTorch, with the GPU reference
  itself pinned to the CPU reference at the sizes the CPU can afford.

## 5. Limits

- The CPU reference is `O(B · H · Sq · Skv · D)` per pass, which is minutes for 4K-token GQA shapes.
  The golden lane skips such bundles on the CPU, with a recorded reason, rather than time out.
- The GPU reference uses one thread per output element and recomputes the score row for every output
  column. It is correct but slow at very long rows; bundles beyond the full tier would need a faster
  oracle.
- Some bundles quantize with cuDNN's fp8 scaling and have more than 128 keys. There they match cuDNN
  up to the tiling artifact described in SEM-024.

## 6. Adding a feature

1. **Contract.** Find cuDNN's definition (cudnn-frontend `include/` and `test/python/sdpa/*_ref.py`).
   Add or update a register entry when a use case differs.
2. **Code.** Add one options field and one setter on each reference. Put the computation in the shared
   score step or the layer it belongs to, never in a copy of the pipeline.
3. **Plans.** Gate every field of the new attribute in both plans, with a decline test per field
   (shared graphs in `SdpaFwdTestGraphs.hpp`).
4. **Tests.** Add a closed form or property (CPU), CPU-vs-GPU parity (quick, small), a
   generator mode plus at least one quick golden bundle, and a mutation check that the new tests fail
   without the feature.
5. **Docs.** Update the support tables here and in the operation-support documents.
