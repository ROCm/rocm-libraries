# Architecture index

Use this page to navigate rocKE's current architecture documentation. Instance
builders are cataloged separately in [`../instances/index.md`](../instances/index.md).

## Start here

- [`mental_model.md`](mental_model.md) — how specs, builders, IR, lowering,
  compilation, and launch fit together.
- [`authoring_model.md`](authoring_model.md) — authoring boundaries and the
  kernel-instance, helper, and core ownership split.
- [`kernel_taxonomy.md`](kernel_taxonomy.md) — which primitive families current
  kernels use and why.

## IR and engines

- [`engines_and_switching.md`](engines_and_switching.md) — Python and C++
  engine selection.
- [`dual_backend_unification_rfc.md`](dual_backend_unification_rfc.md) — design
  record for the implemented serialized-IR backend seam.
- [`ir_serialization_format.md`](ir_serialization_format.md) — serialized IR
  format consumed across that seam.

## Addressing and layout

- [`transform_dag.md`](transform_dag.md) — coordinate-transform semantics.
- [`coordinate_address_planning.md`](coordinate_address_planning.md) — current
  offset, magic-unmerge, and incremental-move lowering.
- [`multi_arch_data_layout.md`](multi_arch_data_layout.md) — architecture facts,
  matrix catalogs, layout maps, ISA backends, and family policy.

## Code-generation controls

- [`backend_support_agpr_res.md`](backend_support_agpr_res.md) — implemented
  AGPR allocation control and its tradeoffs.

## Kernel optimization design

- [`kernel_opt_design.md`](kernel_opt_design.md) — current gfx950 tiled-2D
  attention optimization controls and guards.

Experiment summaries are historical evidence tied to their stated hardware,
toolchain, and configuration. They are not current performance promises.
