# KernelFromAnywhere discovery and implemented contract

KernelFromAnywhere (KFA) is the common-metadata direction for just-in-time (JIT)
generation. General matrix multiplication (GEMM) is the current operation profile;
other operation profiles remain proposed. This note uses API for application
programming interface and ABI for application binary interface.

Convergence means the same versioned metadata schema and execution semantics across
producers, not identical symbols, layouts, or tuning values.

The approved [target design](../JIT.md#target-design) makes the explicit JIT entry
points internal, adds rocRoller as a future independent backend behind the Jit
interface, and has JIT-generated solutions supply the heuristic query after the
pre-tuned lookup. The sections below reflect that design. The
[roadmap](../JIT.md#roadmap) records which steps are implemented.

**The infrastructure already exists in the source tree as Gemm-From-Anywhere (GFA) V1.**
Its guide is [CustomKernels/README.md](../tensilelite/Tensile/CustomKernels/README.md).
GFA V1 provides the metadata format, generic Tensile-host argument dispatch, and
checked-in demos from external sources. It does not include full production kernel
sets, hipKittens, or **JIT generation**, so the current source is not a standalone
universal KFA runtime.

## Intended lifecycle and reusable seams

1. A backend produces AMD graphics processing unit (GPU) assembly (`.s`), with an entry point and normal `.amdgpu_metadata`. GFA's checked-in origin directories are `tensile`, `aiter`, `ck`, `rocroller`, `wave`, and `triton`. Directories are organizational, not runtime backend registration.
2. Embed a top-level **`custom.config` YAML (YAML Ain't Markup Language) mapping inside that metadata section**, carrying the higher-level Tensile-side interface and provenance. `Tensile.AddCustomConfig` can mechanically extract it from a benchmark YAML and inject it into the assembly. `--dry-run` previews it; a provenance-only injection is insufficient to make an external kernel usable. The tool refuses duplicate custom.config insertion. This is source preparation, not an automatic decoder for arbitrary precompiled binaries.
3. A logic-file solution or benchmark `CustomKernel`/`CustomKernels` request resolves the assembly by name. `CustomKernels.getCustomKernelConfig` reads it, validates recognized parameters, forces assembly language and the selected kernel name, and supplies the CustomKernel mapping. `BenchmarkProblems._getCustomKernelSolutionObj` constructs **one Solution without benchmarking**, making it a concrete future singleton-ingestion seam. The broader enumeration helper warning-skips bad custom metadata; a JIT adapter should preserve fail-closed request behavior instead.
4. Existing generator/build machinery reads the custom source, assembles/links normal code objects, and serializes the Tensile solution library. Existing selection and device-library loading then find the appropriate solution/code object. Runtime `generateCustomCall` binds its declared argument semantics from the concrete GEMM request. Built-in solve logic still owns applicable helper sequence, workspace and scheduling behavior.

Source anchors (all under `projects/hipblaslt/tensilelite`): `Tensile/CustomKernels.py:135,187,368`; `Tensile/AddCustomConfig.py:78,207,344`; `Tensile/BenchmarkProblems.py:306,359`; `Tensile/LibraryIO.py:743`; `Tensile/KernelWriterAssembly.py:172,185`; `Tensile/Toolchain/Assembly.py:86`; `src/ContractionSolution.cpp:2783`.

Do not confuse the separate `Tensile/backends/base.py` BackendFactory with an external-codegen registry: it selects optimization strategies (Tensile enumeration or Ductile genetic algorithms), whose interface drives candidate/benchmark loops. Ductile explicitly does not honor build-only requests (`ductile_backend.py:192`). It is not the requested non-benchmarking external JIT interface.

## Metadata contract: what is actually described

| Layer / field | Implemented meaning | Consequence for reuse |
|---|---|---|
| AMD `.amdgpu_metadata` / `amdhsa.kernels` | Low-level code-object facts such as kernel symbol, arg metadata, resource usage and hardware ABI. | Necessary ABI facts; does not describe GEMM support, argument semantics or helper ordering by itself. |
| `custom.config.Source.Origin`, optional `Repository`; top-level `Version`; `Features` mapping | Kernel provenance/declared descriptive capabilities. The README explicitly says these are presence-checked provenance and are not used as runtime gates. | Reuse names/lineage, but never trust `SupportsBias` etc. as a substitute for executable predicates. `Version` defaults to `1.0.0` in AddCustomConfig and is a kernel version string, not negotiated GFA schema version. |
| `InternalSupportParams.KernArgsVersion` | Required Tensile kernel-argument ABI discriminator. | Separate from provenance Version, AMD code-object version, and the JIT bundle's schema2/TLJIT001 envelope. The source has no explicit independent GFA schema negotiation. |
| `ProblemType`, `MatrixInstruction`, tuning/layout/predicate parameters | Contraction types, transpose/batching/epilogue modes, implementation shape and accepted problem constraints in the Tensile Solution system. | Reuse actual Solution/hardware/problem/workspace predicates. Do not infer support from source origin or an example's datatype alone. |
| `CustomKernel.name`, `args` | Primary symbol/name and ordered typed GEMM semantics; each arg has type, semantic, optional padding/index. | Stable name must resolve in the compiled module. Explicit padding and semantic order matter. This is a semantic packing contract, not arbitrary object-ABI reflection. |
| `macrotile`, `threads`, `grid` | Tile shape, workgroup shape and enum-based grid formulas. | Expresses supported GEMM launch patterns, not an arbitrary launch program. |
| `workspaceType`, per-element workspace sizes | Existing modes None, SplitK, StreamK, StreamKWithReduction and sizes per C/bias element. | Runtime workspace/flags/helper handling remains coordinated with Tensile solve and device resources. |
| `generated` | Distinguishes metadata auto-populated for Tensile-generated kernels from external/handwritten custom kernels. | Preserve the runtime's generated/custom dispatch distinction; metadata presence does not authorize rerouting all generated kernels. |

The C++ shape is `include/Tensile/ContractionSolution.hpp:58–228`: semantic examples include A/B/C/D/scale/bias/workspace/flag pointers, independent C/D strides, byte strides for packed operands, Alpha/Beta, shape dimensions, split-K/StreamK scheduling values, constants and epilogue arguments. Grid choices include One, TilesX/Y, Batch, TilesXY, TilesXYBatch, StreamKWith/NoBatch and TilesXYBatchGSU. Serialization is in `include/Tensile/Serialization/ContractionSolution.hpp:80`; packing is in `include/Tensile/KernelArguments.hpp:496`.

External metadata must contain `Source.Origin`, `Version`, a `Features` mapping, `InternalSupportParams.KernArgsVersion`, `ProblemType`, `MatrixInstruction`, and a `CustomKernel` mapping with `args`, `macrotile`, `threads`, `grid` (`CustomKernels.py:493–520`). Tensile-generated sources need only KernArgsVersion because consuming logic/test YAML supplies additional solution state. The metadata validator mostly checks presence and mapping shape; it is not a complete proof that a binary matches its declared argument ABI.

For generated kernels lacking an explicit CustomKernel mapping, `_buildCustomKernelFromMetadata` at292 reads the **first** `amdhsa.kernels` entry, infers known argument-name semantics, derives tile/grid defaults, and may reorder universal-argument headers. Unknown arg names raise an actionable error requesting an explicit mapping. This inference does not generically interpret all possible backend ABIs or arbitrary multi-entry code objects. Explicit external mappings can select a symbol in a file containing multiple kernels; the Composable Kernel (CK) example includes many compiled symbols but explicitly names its selected GEMM.

## Layout, launch and multi-kernel limits

The current contract is **GEMM-specific and AMD/HIP-specific**, using the existing Tensile runtime. HIP is the Heterogeneous-compute Interface for Portability. This is not a portable backend-neutral dispatch ABI or independent module/lifetime API.

- Physical layout requires accurate ProblemType plus per-argument semantics and support predicates. Packed 4-bit floating-point (FP4) examples distinguish byte leading strides (`StrideA0Bytes`/`StrideB0Bytes`), packed K (`SizeSumDiv2`), scale strides and fixed unit strides. Those adapters must agree with actual tensor/scale storage.
- The documented Triton integration rewrites dynamic local data share (LDS) to a static group segment because the current custom-call launcher uses `sharedMemBytes=0`. Unused Triton scratch pointers are explicitly represented by trailing padding. A backend requiring dynamic LDS or live scratch allocation cannot simply copy the demo metadata.
- Target instruction set architecture (ISA), wavefront and valid problem sizes remain enforced through assembler, Solution state, predicates and the normal device runtime, rather than a new GFA device negotiation layer. Checked-in demos target gfx942/gfx950 only; there is no gfx1250 or Windows demo.
- A CustomKernel record describes a main GEMM entry and recognized workspace modes. There is no arbitrary helper-kernel directed acyclic graph (DAG), per-stage workspace-lifetime graph, or backend-owned host callback encoded in it. Existing Tensile solve can still produce needed pre/post kernels for supported modes. Preserve its full sequence, all-symbol preflight, storage ownership and stream/synchronizer behavior when composing with JIT.
- Runtime code deliberately keeps Tensile-generated kernels on their established generated call path; external/handwritten kernels use the generic custom call. The host mapping documents the specific guard and scheduling/layout reasons. A universal metadata-based replacement would be a broader change requiring separate validation.

## Actual examples and validation status

Current checked-in test inputs use 32-bit floating-point (FP32) computation. Data
formats below include 16-bit floating point (FP16), `bfloat16` (BF16),
and microscaled 4-bit floating point (MXFP4):

| Test YAML under `Tensile/Tests/custom/` | Current data/output | Target |
|---|---|---|
| `custom_aiter_bf16.yaml` | BF16 → BF16 | gfx950 |
| `custom_aiter_f4.yaml` | MXFP4 → BF16, prescribed swizzle/scale layout | gfx950 |
| `custom_ck.yaml` | FP16 → FP16 | gfx942 |
| `custom_rr.yaml` | FP16 → FP16 | gfx942 |
| `custom_wave_bf16.yaml` | BF16 → FP32 | gfx950 |
| `custom_triton_f4.yaml` | MXFP4 → BF16, packed-stride mapping | gfx950 |
| `custom_tensile_sk.yaml` | FP16 → FP16, StreamK | gfx942 |

These are concrete integration fixtures, not proof that every operation/layout from each source backend is supported. The CK fixture's YAML sets DataType/DestDataType `h` (FP16).

Reusable tests cover AddCustomConfig injection, strict/non-strict metadata command-line interface (CLI) behavior, malformed/absent metadata, parameter/predicate propagation, semantic inference, source discovery, and logic-file round trips: `test_CustomKernelMetadata.py`, `test_custom_kernel_cli.py`, `test_library_io_custom_kernel.py`, `test_custom_kernel_host.py`, `test_custom_kernel_occupancy.py`. `Tensile.ValidateMetadata --strict` is available as a failing gate; non-strict mode warns with successful exit. Build-time `ValidateMetadata` is off by default and warns while continuing. No `.github`, tox or CMake file invokes it directly, so it is available but not enforced by continuous integration (CI).

## Producer-first convergence

**KFA metadata is the target common kernel encoding. TensileLite production must emit a complete compatible representation before both origins share the same argument/launch path.** The ingestion facts above describe the existing implementation; do not introduce a competing opaque runtime to avoid the KFA contract.

Generated Tensile kernels already populate a `CustomKernel` record in `KernelWriter._getKernelSource:12444–12474`. `_registerKernelArgs:12302` records ordered typed semantics, and the record includes symbol, tile, threads and grid. `Contractions.Solution.FromOriginalState:940–1007` serializes it together with **ProblemType, SizeMapping, hardware/problem/task predicates and InternalArgsSupport**. Those surrounding fields are still essential executable metadata. `SingleSolution._build:409–425` constructs the existing `MasterSolutionLibrary.BenchmarkingLibrary`, applies names and writes its normal YAML/MessagePack library. This is already partial convergence at the producer and library levels. `TensileCreateLibrary/Run.py:262–273,341–351` explicitly carries the generated CustomKernel mapping from worker results back onto original/serialized solutions. However, assembly emission in `rocisa/rocisa/include/code.hpp:1524–1526` currently embeds only `InternalSupportParams.KernArgsVersion` within `custom.config`; normal kernel ABI metadata is also present. The complete semantic description lives in solution state/library, not in self-contained generated assembly. Producer convergence must close that packaging gap too.

It is not complete equivalence. Concrete source gaps are:

| Concern | Existing generated behavior | Required common-contract work |
|---|---|---|
| Packed argument completeness | `Signature.py:445–452` and `generateSingleCall:2751–2756` append four 64-bit D/C/A/B batch offsets. `_registerKernelArgs` omits them; the current CustomArgSemantic enum has no corresponding fields. Optional fused all-to-all tail at Signature:457–467 is also absent. | Derive signature and semantic layout from one authoritative description, including exact order, sizes, offsets/alignment, optional tails and pointer-array semantics. Unsupported profiles must reject explicitly. |
| Workspace and helpers | Generated CustomKernel currently hard-codes `workspaceType=None`, zero per-element sizes and `generated=True`; `requiredWorkspaceSize:5400` deliberately recovers real policy from SizeMapping. `solve:4850–5030` inserts beta-only, conversion and reduction calls. | Preserve complete workspace/synchronizer policy and helper sequence in the common GEMM profile; the existing main-entry record alone cannot replace it. A general DAG is unnecessary for the first supported profiles. |
| Launch geometry | `generateSingleCall:2600–2680` handles real M/N cluster grids and rounds appropriate grids to cluster multiples; current generic custom dispatch has no equivalent rounding. | Normalize workgroup/grid/cluster and LDS requirements; compare final launch descriptors, including special schedules. |
| Stream-K semantics | Generated kernels use resolved whole-problem schedules; custom `StreamKWithBatch` explicitly multiplies a per-batch grid by batch count. Runtime gates also distinguish tile counts and dynamic queue/work-stealing support. | Encode a precise schedule profile and preserve device/compute-die/synchronizer constraints; do not equate identically named metadata with identical meaning. |
| Support/selection | ProblemType alone is insufficient. The full Solution also carries hardware/problem/task predicates, size mapping, scale physical layout and internal-argument support. | Make those constraints an explicit, versioned GEMM profile associated with KFA metadata, reusing the existing predicate machinery. Source/Features provenance is not a capability gate. |
| Grouped and optional modes | Grouped execution uses a distinct existing call path; stochastic seed, debug, packed operands, activation, scaling and adaptive accumulation have profile-specific layout. | Enumerate supported profiles and test exact packing/sequence equivalence for each before switching its runtime dispatch. |

`solve:4977–4996` documents the temporary `generated` gate: subtile, gfx950 and work-stealing generated layouts have not been validated through `generateCustomCall`. This is a real compatibility boundary. Enrichment must let those kernels join the shared path; changing a Boolean does not prove compatibility.

Roadmap step 3 of the target design has generators emit assembly or HIP source plus metadata only, and hipBLASLt builds raw code objects through AMD comgr. KFA already embeds `custom.config` in the assembly's `.amdgpu_metadata` section, so it is a natural carrier for the "metadata" output of assembly-emitting backends. How HIP-source output carries the same metadata is not yet defined. The plan of record does not yet select the metadata format; the packaging gap above still applies whichever format is chosen.

## Existing selection and execution that both origins reuse

For ordinary installed libraries, KFA custom kernels are ordinary ContractionSolutions. `MasterSolutionLibrary.FromOriginalState:400–604` builds layers in this order: hardware, operation identifier, performance metric, problem/type predicates, optional lazy placeholder, then equality/range/grid/prediction selection. Runtime `getSolutions` in `tensile_host.cpp:4850–4880` calls `library->findTopSolutions`; `SingleSolutionLibrary.hpp` applies hardware and software predicates. Chosen solutions use existing `solve` and HIP SolutionAdapter loading/launching. Origin is not a separate selector backend.

Current JIT supplies an explicitly chosen singleton instead of searching the installed multi-solution library. `loadGeneratedBundle` (`hipblaslt-jit-tensilelite.cpp:143–204`) reads the private envelope, loads the normal serialized library via `LoadLibraryData`, loads code-object bytes with SolutionAdapter and resolves the primary symbol. `Bundle::support/prepare` (`tensile_host.cpp:6033–6131`) reuses hardware/problem/task/software predicates, required workspace, ConstructTensileProblem, GetTensileInputs, bindFlagRegion and `solution->solve`. It resolves **every** returned helper symbol before publishing a PreparedLaunch; execution calls `adapter->launchKernels`.

Therefore the current JIT implementation does not contain a whole second kernel execution engine. KFA convergence can unify the generated/custom descriptor and argument/launch branches, then reduce private artifact translation. It does not remove operation capture, module ownership, complete-bundle preparation, concurrency/device rules, process launch or operation adaptation.

In the target design, the JIT solution library is loaded at runtime as a second master library (roadmap step 4). Because it mimics the TensileLibrary layout of a library file plus code objects, its entries are expected to remain serialized solutions that keep the existing predicates, workspace rules and `solve` sequence. That library is then another consumer of the same executable contract, not a new selector backend.

## Reusing problem and solution types across producers

The [current request payload](../library/src/amd_detail/hipblaslt-jit-gemm-internal.hpp)
already embeds `RocblasltContractionProblem`. `GemmRequest` adds the operation tag
and owns captured alpha/beta values; it does not own application buffers.
Public `GemmProblemType` describes less than this full payload. Another generator
does not require a second public GEMM problem model.

The [Tensile base interfaces](../tensilelite/include/Tensile/Tensile.hpp) already
define `Problem`, `ProblemInputs`, and `Solution`. They are generic but skeletal,
not complete compilation identities or executable owners. Preserve
`ContractionProblemGemm` and `ContractionSolution` for their existing GEMM predicates,
workspace rules, and ordered preparation; assess extensions against missing semantics.

The [private backend interface](../library/src/amd_detail/hipblaslt-jit-backend.hpp)
keeps `CompiledSolution` context for backend, target, request, workspace, and bundle
lifetime. The current opaque `Solution` can remain a thin owner around reusable
operation solutions and modules. The matmul algorithm is an operation-specific token;
it cannot by itself replace that general ownership. Likewise, `Request` can retain
the existing operation payload without becoming a backend-specific recipe structure.

For non-GEMM work, first define the operation payload, support checks, argument
binding, ordered helpers, workspace initialization, and lifetime. Reuse
`KernelArguments`, `KernelInvocation`, and HIP `SolutionAdapter` where sufficient.
Shared generic handles do not establish a non-GEMM KFA profile or execution adapter.
`getJitAlgo(device, request, backend, maxWorkspaceBytes, solution, diagnostics)`,
`makeGemmRequest`, both `getGemmAlgo` functions and `createBackend` are not public
API. Their headers are internal and used by unit tests, and Jit takes over backend
invocation in roadmap step 2. Evaluate these reuse choices against that internal
contract.

## rocRoller: existing integration and proposed KFA adaptation

The current [rocRoller host route](../library/src/amd_detail/rocblaslt/src/rocroller/rocroller_host.cpp)
derives `KernelType`, obtains Origami-ranked configurations, and checks a handle-owned
kernel cache. On a miss it calls `RocRollerGemmKernel::generate`; the
[kernel implementation](../library/src/amd_detail/rocblaslt/src/rocroller/gemm.cpp)
uses `CommandKernel::generateKernel` and `loadKernel`, then binds `CommandArguments`,
checks predicates, and calls `launchKernel`. The
[configuration selector](../library/src/amd_detail/rocblaslt/src/rocroller/solution_selection.cpp)
and [cache](../library/src/amd_detail/rocblaslt/src/rocroller/solution_cache.cpp)
are separate responsibilities. This route precedes Tensile solution lookup and
does not currently pass through generic `getJitAlgo` or the Tensile KFA consumer.

The checked-in [rocRoller KFA fixture](../tensilelite/Tensile/Tests/custom/custom_rr.yaml)
demonstrates one assembly artifact entering existing custom-kernel ingestion. It
does not prove automatic KFA export from runtime generation. The rocRoller dispatch
branch also supports precompiled custom code objects with handwritten argument
packing; [custom_kernels.cpp](../library/src/amd_detail/rocblaslt/src/rocroller/custom_kernels.cpp)
includes kernels from other producers, so this route does not identify a kernel's producer.

| TensileLite proposal | rocRoller proposal | Other-generator proposal |
| --- | --- | --- |
| Complete generated KFA metadata and use the library-owned consumer for proven profiles. | Export or normalize supported generated artifacts into that same schema and consumer. | Implement the same producer contract; add an operation adapter only for missing operation semantics. |

The contract includes binary/symbol/target compatibility, full argument ABI and
physical layout, predicates, grid/workgroup/cluster units and dynamic shared memory,
every ordered helper, workspace and synchronization initialization, lifetime, and
diagnostics. Preserve rocRoller's `ZeroedBeforeAndAfter` Stream-K scratch contract:
caller-visible workspace is not the complete synchronization requirement. The
TensileLite path also obtains handle-owned synchronization state during preparation;
applications own buffers/workspace and follow existing handle/stream rules.

All three producers should share the library-owned validation and execution
consumer for each supported profile. Origami ranking remains distinct from code
generation. This assessment is based on source inspection; it is not a runtime or
interoperability result.

In the target design, rocRoller is a future independent backend behind the Jit
interface, alongside TensileLite and other generators; it does not route through
TensileLite. That backend is outside the six planned roadmap steps. Until it
exists, the runtime route described above stays in place. `HIPBLASLT_JIT=2`
skips that early route so that JIT is the only source of solutions.

## Selection and JIT-generated solutions

The current [library construction](../tensilelite/Tensile/SolutionLibrary.py)
normally orders Equality, Range, Prediction, GridBased, FreeSize, and TruePred
selectors below hardware/operation/problem predicates. Equality is matching with
equality distance; Prediction uses Origami to rank existing solutions. Modes and
available branches affect traversal. Provider-private prediction of new recipes
is a different task and remains outside the existing-solution ranking contract.

The approved target design in [JIT.md](../JIT.md#heuristic-integration-and-hipblaslt_jit)
places JIT generation outside the pre-tuned library:

- JIT is not a leaf row inside the pre-tuned library. With `HIPBLASLT_JIT=1`, the
  trigger is evaluated after the complete pre-tuned lookup and the existing
  `getAllSolutions` shortfall fill. That placement covers an absent root library or
  operation branch, which a leaf could not catch.
- The trigger also runs after the retry that repeats an xf32 lookup with FP32 math
  inside `getBestSolutions`. Results from rocRoller's early route count toward
  `requestedAlgoCount`.
- The trigger fires when the result is empty **or** contains fewer than
  `requestedAlgoCount` solutions. The query then consults the JIT solution library
  and generates as many solutions as are needed to reach the requested count. Filling
  top-N is intended, and the trigger sits outside the
  [ExactLogicLibrary::findTopSolutions](../tensilelite/include/Tensile/ExactLogicLibrary.hpp)
  accumulation.
- With `HIPBLASLT_JIT=2`, JIT is the only source: Equality, Prediction, the other
  pre-tuned libraries and the rocRoller early route are skipped. The JIT solution
  library is consulted before generation.
- The explicit entry points are internal. Deterministic backend choice and
  prewarming remain available to unit tests through the internal headers.

JIT.md also defines the failure rules (JIT failures are always reported),
build-time tool-path defaults with `HIPBLASLT_JIT_*` overrides for the heuristic
path, and the cache key under which mismatched libraries are ignored rather than
deleted. Still to settle before step 5: compilation latency, concurrency and stream
capture inside a heuristic query, whether the existing heuristic contract allows
fewer than `requestedAlgoCount` results, how unsupported and failed outcomes map to
statuses, and enumeration/index-query behavior for reserved indices. A GEMM-templated
library does not become operation-independent by adding a type.

## Reviewable implementation sequence

1. **Emission:** complete generated metadata for a small explicit GEMM profile; derive signature and semantic arguments from one source; include necessary target, support, launch and workspace/helper facts. Use KFA as the canonical representation while preserving existing supported files.
2. **Validation:** define a distinct schema/profile version and backward-compatible reader. Validate required executable fields, recognized semantics, argument offsets/types/sizes, symbols, targets and complete artifact contents. Do not overload provenance Version or weaken artifact containment/integrity checks. Add strict validation at this producer/consumer boundary.
3. **Equivalence:** compare old and metadata-driven argument bytes, complete launch descriptors, workspace requirements and full helper sequence for the same concrete requests. Then run representative numerical/runtime tests on their actual GPUs, including nontrivial C/D strides, batching, split-K/Stream-K, scaling/epilogues, and malformed/unsupported metadata. Source inspection alone is not a runtime-equivalence result.
4. **Shared path:** gate each proven profile onto KFA-driven execution, retaining existing support predicates and HIP adapter. Unproven profiles continue on their established path until extended and validated. An existing external fixture and equivalent generated kernel should exercise the same consumer.
5. **Simplification:** consolidate/delete only the code now demonstrably redundant. `SingleSolution._writeLoaderEnvelope:478–502`, `readEnvelope` in `hipblaslt-jit-tensilelite-artifacts.hpp`, and `loadGeneratedBundle`'s private manifest-to-library identity plumbing are concrete transport candidates once common KFA artifacts cover their requirements. The planned JIT solution library, which mimics the TensileLibrary layout, is another candidate replacement for that private transport. Much validation migrates rather than vanishes. Retire generated/custom packing forks only when no supported profile depends on them.

This sequence is independent of the six target-design roadmap steps in [JIT.md](../JIT.md#roadmap). KFA convergence is future work.
