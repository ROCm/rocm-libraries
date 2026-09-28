# Architecture index

Use this page to navigate rocKE's current architecture documentation. Instance
builders are cataloged separately in [`../instances/index.md`](../instances/index.md).

## Start here

- [`mental_model.md`](mental_model.md) — how specs, builders, IR, lowering,
  compilation, and launch fit together.
- [`authoring_model.md`](authoring_model.md) — the operation-to-`KernelDef`
  workflow and the reusable helper/core boundaries.
- [`kernel_taxonomy.md`](kernel_taxonomy.md) — which primitive families current
  kernels use and why.

## IR and engines

- [`engines_and_switching.md`](engines_and_switching.md) — Python and C++
  engine selection.
- [`dual_backend_unification_rfc.md`](dual_backend_unification_rfc.md) — broader
  dual-backend migration proposal; its serialized-IR seam is implemented, while
  other phases remain proposed.
- [`ir_serialization_format.md`](ir_serialization_format.md) — serialized IR
  format consumed across that seam.
- [`portable_ir_schema.md`](portable_ir_schema.md) — the JSON/CBOR artifact
  schemas the C++ engine replays at runtime, and the byte-identity and device
  gates on that path.
- [`portable_ir_production_readiness.md`](portable_ir_production_readiness.md) —
  assessment of that path against production requirements: measured evidence,
  gap analysis, and the acceptance plan for the implementation story.

## Addressing and layout

- [`transform_dag.md`](transform_dag.md) — coordinate-transform semantics.
- [`coordinate_address_planning.md`](coordinate_address_planning.md) — proposed
  transform-aware physical-address planning layer.
- [`multi_arch_data_layout.md`](multi_arch_data_layout.md) — current architecture
  facts plus forward design for matrix catalogs, layout maps, ISA backends, and
  family policy.

## Code-generation controls

- [`backend_support_agpr_res.md`](backend_support_agpr_res.md) — implemented
  engine-level AGPR allocation control, current scope, and remaining validation.

## Kernel optimization design

- [`kernel_opt_design.md`](kernel_opt_design.md) — proposal for combining current
  gfx950 tiled-2D attention optimization controls.
- [`wavescope_integration.md`](wavescope_integration.md) — how source locations
  reach an ATT trace, and how to use the viewer during an optimization pass.
  Opens with a glossary of the LLVM and DWARF terms the rest of it assumes.
- [`attention_thread_block_mapping.md`](attention_thread_block_mapping.md) — which
  workgroup gets which (query-block, head, batch) tuple in dense attention. Frames
  every mapping as a mixed-radix decomposition over four digits, giving three
  independent factors (digit order, query-block traversal, assignment policy) and a
  144-combination space of which eight are measured. Records the hypotheses tested
  against it and which held, why one locality class leads at every batch size, and why
  the persistent and non-persistent paths differ in *who assigns work* rather than in
  what they can express. Its "In flight" section covers the generalized-ordering sweep
  that made the whole space reachable behind experimental spec fields, including the
  measured correction that the two slowest digits are **not** interchangeable — and
  "Resolved" records what that sweep could not see, because the shipped decode's own
  traversal had been pruned from it. Ends with an end-to-end comparison of the branch
  against `develop` over the real LLM shape list, and one negative result (the kv-phase
  split) kept because it explains why the highest-locality orders lose. A prior-art
  section maps what AITER, AOTriton, FlyDSL and FlashAttention do onto the same digit
  notation, and notes the two ideas none of the sweeps have tested (GQA packing and a
  dynamic work queue).

Experiment summaries are historical evidence tied to their stated hardware,
toolchain, and configuration. They are not current performance promises.
