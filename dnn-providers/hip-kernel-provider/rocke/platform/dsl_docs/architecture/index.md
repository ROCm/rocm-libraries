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

Experiment summaries are historical evidence tied to their stated hardware,
toolchain, and configuration. They are not current performance promises.
