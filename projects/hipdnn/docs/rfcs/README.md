# hipDNN RFCs

This directory holds the design documents (RFCs) for hipDNN. Each RFC has one status. The status shows how much of the design exists in the code on `develop`.

## Status values

| Status | Meaning |
|--------|---------|
| `Planned` | The RFC is proposed or accepted. No implementation is merged. |
| `In progress` | Part of the implementation is merged, or it is merged behind a flag or phase. Some design parts are not built and the RFC marks them `Not yet built.` |
| `Implemented` | The feature the RFC proposes is merged, and the RFC text describes the current design. If a major part of the design is not built, use `In progress`. |
| `Rejected / Replaced` | The design was not built, was dropped, or another RFC replaced it. The RFC links to the replacement when one exists. |

Choose the status from the code, not from the RFC text.

## Header block

Put this block directly under the title and the contributors line, before the table of contents:

```markdown
- **Status**: Implemented
- **Status notes**: <1-3 short sentences: what shipped, what did not, key PR numbers or file paths>
- **Last verified**: YYYY-MM-DD against `develop` (<short commit hash>)
```

For a replaced RFC, write `- **Status**: Rejected / Replaced by [RFC NNNN](./NNNN_Name.md)`.

Write each RFC to describe the current design, as if it were a first draft. Do not add notes that compare the code with an earlier version of the RFC. Git history keeps those changes. Start the section of a design part that is not built with the sentence `Not yet built.` Use `> [!NOTE]` only for unexpected behavior that a reader must watch for.

Update the status, the notes, and the verification date in the same change that changes the implementation state.

## Index

| RFC | Title | Status |
|-----|-------|--------|
| [0001](./0001_CpuGraphExecutorDesign.md) | CPU graph executor | Implemented |
| [0002](./0002_PluginSdkDesign.md) | Plugin SDK | Implemented |
| [0003](./0003_EngineIdDesign.md) | Engine ID | Implemented |
| [0004](./0004_EngineConfigKnobs.md) | Engine configuration knobs | Implemented |
| [0005](./0005_Versioning.md) | Versioning | In progress |
| [0006](./0006_PluginAgnosticIntegrationTests.md) | Plugin-agnostic integration tests | Implemented |
| [0007](./0007_EngineSelectionHeuristicsFramework.md) | Engine selection and heuristics framework | Implemented |
| [0008](./0008_OverridableTensorShapesDesign.md) | Overridable tensor shapes | In progress |
| [0009](./0009_CompiledPlanSerialization.md) | Execution plan serialization | Implemented |
| [0010](./0010_BehaviorNotes.md) | Per-engine behavior notes | Implemented |
| [0011](./0011_BundleTestFramework.md) | Bundle test framework | In progress |
| [0012](./0012_CuDNN_Shim.md) | cuDNN shim (reference: [0012 reference](./0012_CuDNN_Shim_Reference.md)) | In progress |
| [0013](./0013_Autotune.md) | Autotuning | Implemented |
| [0014](./0014_RaggedTensors.md) | Ragged tensors | In progress |
| [0015](./0015_EngineSupportClaims.md) | Per-graph engine support claims | In progress |
| [0016](./0016_RuntimePassByValueTensors.md) | Runtime pass-by-value tensors | Implemented |
| [0017](./0017_UniversalKernelDescriptor.md) | Universal kernel descriptors (example: [worked example](./examples/0017_UniversalKernelDescriptor_WorkedExample.md)) | In progress |
| [0018](./0018_UniversalMatchDescriptor.md) | Universal match descriptor (UMD criteria) | In progress |
| [0019](./0019_UniversalHeuristicDescriptor.md) | Universal heuristic descriptor (UHD) | In progress |
| [0020](./0020_UniversalEngineDescriptor.md) | Universal engine descriptor (UED) | In progress |
