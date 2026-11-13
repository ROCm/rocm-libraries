# Lane: engine applicability (`APPL`)

Find graphs an engine accepts but cannot serve: undefined behavior, wrong results, crashes or errors that surface late. The
contract: an engine accepts a graph only if it serves every property that can reach it, and declines the rest before plan build.
Applicability means `isApplicable`, a matcher or `graph_match`, and a plan builder's own checks.

## Reachability first

A property counts only if it reaches the changed engine. All three layers must pass:

1. **Plugin version.** The plugin's advertised `HIPDNN_PLUGIN_API_VERSION` must clear the graph's floor
   (`projects/hipdnn/plugin_sdk/include/hipdnn_plugin_sdk/PluginVersionConstants.hpp`, `computeMinimumEnginePluginApiVersion`;
   per-property floors are in the checklist below). An override-shape plugin must also export the override-execute symbol. The
   host-side plugin-version filter in the hipDNN backend is outside this lane's trigger; changes to it belong to `CORR` or `API`.
2. **Provider schema guard.** In hip-kernel-provider, `src/core/GraphSchema.hpp` and `ApplicabilityManager` skip an engine whose
   declared schema is below the graph floor.
3. **Engine-declared version.** Ingestor engines declare `sdk_version` in the UED; `GenericPlanBuilder::understandsGraph`
   declines below the floor.

A gap below the floor is `N/A-version`: report it at most as `FUTURE WORK` (latent), never as a defect. Frontend `validate()` is
no guard: it does not run on the raw backend API, deserialized or JSON graphs, or the cuDNN-compat lowering.

## Method

1. List the properties (table below) that reach each changed engine, builder or matcher.
2. For each, find the read in applicability, or a proven delegation. An external library (MIOpen, hipBLASLt) counts only when it
   is shown to return an error before launch and a test pins that; documented-unsupported means the provider declines.
3. Compare sibling builders, arches and forward/backward paths for divergent gates.
4. Check every path that builds, sizes or restores a plan without the gate: `buildPlan`, `getMaxWorkspaceSize`, planless
   execution contexts, deserialized plans.
5. Review any raise of an advertised or declared version (plugin version macro, UED `sdk_version`, provider schema opt-in) over
   every newly reachable property.
6. Decline tests: one per property value with boundary rows (below, at, above the limit), each a legal graph, failing when the
   guard is removed. Offer the guard removal to `VER` as the mutation.
7. State, for every claimed hazard, the smallest legal graph mutation the engine accepts.
8. When support depends on a contract the tree cannot show (prebuilt code objects, vendored or
   artifact-only kernels, a library's internals), the missing positive gate is the hazard. A
   reachable value or feature combination (GQA ratio, mask mode, dtype pair, alignment) that the
   implementation path does not visibly handle is an applicability finding, not a `CORR` handoff.
   Tier it by its worst plausible outcome, mark the kernel side `[INFERENCE]` in **Proof**, and put
   the device check that would settle it under **Approval requests**. Never lower the tier because
   the kernel cannot be read. Such a finding's verdict stays `UNVERIFIED` until the device check under **Approval requests**
   runs.

## Hazard classes

Closed set; name each finding's class in `Actual`.

| Class | Look for |
|---|---|
| `never-read` | A reachable property applicability never reads |
| `loose-check` | Read, but admits values the kernel cannot serve (divisibility, limits, 32-bit narrowing) |
| `sibling-drift` | One builder, arch or direction checks it; a sibling does not |
| `accept-then-fail` | Applicability passes; plan build or launch fails |
| `version-mismatch` | A version raised before every engine declines or reads the new fields |
| `bypass-path` | A plan built or run, or a tensor handled, outside the gate |
| `blackbox-delegation` | A library trusted to reject that does not, or is unproven |
| `over-decline` | Reverse: a servable graph refused |

## Property checklist

"Checked" means the engine reads the property and declines unsupported values. Standard decline test unless stated: from a legal
served graph change only the property, assert decline, and assert the unchanged graph is accepted.

| Property | Floor | Checked means | Decline test |
|---|---|---|---|
| Tensor `data_type` | 1.0 | Every operand, aux and virtual intermediate dtype is in the served set and consistent; node `compute_data_type` supported | One row per unsupported dtype per role |
| Dims and extents | 1.0 | Rank, cross-tensor relations (`Hq % Hkv`), head-dim tables, divisibility; 32-bit narrowing of dims, strides and products range-checked; backend does not check positivity | Faulting shape; rows at 2^31 and 2^32 |
| Strides and layout | 1.0 | The kernel's addressing formula: innermost stride, packing, order, 0 strides, pitch, derived strides, output overlap | Stride 0, short row, padded pitch, permuted order |
| `alignment` | 1.3 if any `!= 16` | Declines every value that is not a multiple of what the kernel needs, including 0 and negatives, on every operand incl. aux | 1, 2, 4, 8, 24 decline; 16, 32, 256 serve |
| `virtual` | 1.0 | Operands the engine reads or writes are non-virtual; intermediates it does not write are virtual | Flip per role |
| `is_runtime_pass_by_value` | 1.2 | A host-pointer override is read, or the role is declined; compile-time `value` tensors are read, not expected as buffers | Pass-by-value per non-scalar role |
| Ragged offset tensor | 1.3 | Non-SDPA engines decline ragged; SDPA declares supported combinations (aux dtype and shape, shared offsets, multiplier equals sequence stride, no ragged Q over paged K/V) | Ragged on each operand; sibling arches agree |
| Ragged offset multiplier | 1.4 if any `!= 1` | Read on every tensor incl. non-ragged, or declined | Multiplier 2 with and without matching stride |
| Override shapes | 1.1 + export | `is_override_shape_enabled` declined until override execute exists | Flag on a served graph |
| Graph-level data types | 1.0 | Node and tensor types are checked, not the graph defaults | As `data_type` |
| `device_arch` | none | Every launch configuration is keyed on the actual arch, siblings included; known per-arch library bugs declined | Per-arch rows; arch-query failure declines |
| Topology, node count | none | Node count, order and fusion shape match the builder; data flow matches by uid; backend does not check order | Extra, reordered, unlinked node |
| Optional tensors, attribute combinations | none | Couplings the engine relies on checked; optional tensors it would drop are declined | Drop or add each optional tensor |
| Aliasing | none | One uid in two roles or overlapping outputs declined where kernels assume `__restrict__` | Same uid in two roles |
| SDPA attributes | none | Every attribute read or declined: scale (finite, positive, normal) or tensor, dropout, masks and bounds, sinks, stats, paging, padding and seq lens, `mma_core_mode`, `implementation`, FP8 descale and amax | Paired graph with and without each |
| Convolution | none | `conv_mode`, groups, dilation, asymmetric padding, dtype triples, layout | Each value outside the served set |
| Norms | none | Epsilon and momentum presence and dtype, running-stat completeness, saved mean and inv_variance, phase, activation fusion | Subsets of running stats; drop epsilon |
| Pointwise modes and clips | none | Mode support; clips, slope and betas map to exactly the served function; operand vs output dtype; scalar operands | `{lower 0, slope 0.01}` on a ReLU-only builder |
| Matmul and MX dequant | none | `is_negative_scale`, dequantize `block_size`, scale, bias and batch strides, dtype pairs | One row per unread attribute |

## Severity

Per [../policy.md](../policy.md) tiers; kind `defect` unless noted.

| Situation | Tier |
|---|---|
| Reachable hazard giving UB, a memory fault, wrong results or a crash | `BLOCKING` |
| Reachable `accept-then-fail`, `over-decline`, or a version raise whose newly reachable properties are not all handled | `IMPORTANT` (`BLOCKING` when the raise exposes UB or wrong results) |
| `N/A-version` latent hazard, or one in code the change does not touch | `FUTURE WORK`, kind `source-claim` |

A reachable claim at `IMPORTANT` or above carries its graph mutation and the three-layer reachability argument in **Proof**.

## Hazard knowledge directory

When the `Hazard knowledge:` field of `dispatch-context.md` (or the dispatch `Overrides:`) names a directory, read its README, its
property checklist and the page for the changed provider. Cite matching history (PR numbers) in findings and say whether a finding matches
an open candidate row there by ID, and write `Hazard knowledge: <directory read>` under **Coverage**. Otherwise run from this brief and write `Hazard knowledge: none supplied` under **Coverage** (a
non-file line, free of `|`).

## Handoffs

- Missing decline tests: `TESTB`. Weak decline tests (illegal graph, boundary rows missing): `TESTQ`.
- Mutation proofs: `VER`.
- Correctness faults that are not applicability: `CORR`.
