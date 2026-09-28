# Request and execute JIT solutions

This is a source guide to the current just-in-time (JIT) application programming
interface (API) for general matrix multiplication (GEMM), maintained under the
existing hipBLASLt code/documentation reviewer rules. See the
[roadmap](JIT_ROADMAP.md) for ownership, source publication and deferred release
documentation integration.

The [direct TensileLite API](JIT_TENSILELITE.md) accepts an explicit
YAML (YAML Ain't Markup Language) recipe
and returns a GEMM algorithm in one call. That API and the
[direct sample](clients/samples/29_hipblaslt_jit_gemm/README.md) remain available.
This guide describes the optional generic interface layered above that path.
Both interfaces share one provider implementation and algorithm registry.
The direct entry point always requires a recipe and explicitly selects TensileLite.

The installed `hipblaslt/hipblaslt-jit.hpp` exposes a backend-neutral request API.
`getJitAlgo` accepts an operation request and configured backend and returns an
owned solution bundle. The bundle includes the kernels and helpers needed for
that operation. GEMM is implemented; an attention request adapter and provider
remain future work.

An application configures the TensileLite provider through
`jit::tensilelite::createBackend`, using options from
`hipblaslt/hipblaslt-jit-tensilelite.hpp`. Python, compiler, recipe and output
paths belong to this provider. It then uses `jit::makeGemmRequest` to capture
existing GEMM descriptors and host scalars, and `jit::getJitAlgo` to compile on
the selected device. Finally, `jit::getGemmAlgo` adapts the solution to the
algorithm accepted by `hipblasLtMatmul` and `hipblaslt_ext::Gemm`.

The application owns its buffers and workspace. The request owns descriptor
values and host scalars; it does not take ownership of device pointers.
Compilation and support checks finish before graphics processing unit (GPU) work is submitted.

Internally, the GEMM request already reuses `RocblasltContractionProblem` with
owned scalar values. The generic `Solution` retains executable ownership around
existing GEMM support and execution machinery. The
[design discussion](jit-design/confluence-roadmap-draft.md) assesses further reuse
of existing problem/solution types and KernelFromAnywhere (KFA) metadata across
TensileLite, rocRoller, and possible other generators. Those are proposals: the
current generic production provider is TensileLite, and existing rocRoller runtime
generation uses a separate path. An ordinary matmul does not initiate this new
TensileLite/generic JIT path; rocRoller's existing path can already generate code.

## Origami modeled inputs

An empty TensileLite `Options::configPath` requests Origami prediction. The private
`origami.gemm.dp.v1` contract covers the existing data-parallel candidate domain.
Origami ranks caller-supplied configurations; it does not synthesize their fields
or choose whether to enable Stream-K. `StreamK=0` and occupancy 1 remain explicit
model inputs. Every applicable prediction is transferred; defaults supply only
settings that this model does not predict.

The inventory below follows `shared/origami/include/origami/{origami,gemm,streamk,types}.hpp`
and their implementations. A selected configuration is an output of ranking even
though its fields originate in the caller's candidate catalog.

| Origami output | Generator input or use | Conditions |
| --- | --- | --- |
| `rank_configs` / `select_config`: ordered configuration and latency | Nine-value `MatrixInstruction` recipe (MI plus retained wave topology), macro tile, `DepthU`, `NonTemporalA/B`; latency and order in provenance | Estimation ranks the existing target instruction/tile/depth/cache-hint catalog. No kernel benchmarking. |
| `select_workgroup_mapping`: signed `wgm` | `WorkGroupMapping` | Preserved exactly; zero and values outside the runtime range are rejected. |
| `select_workgroup_mapping`: `wgmxcc` | `WorkGroupMappingXCC`, with `WorkGroupMappingXCCGroup=0` | Origami 0/1 both mean identity and translate to Tensile 1. Larger values require a supported power of two and a divisible grid for equivalent whole-grid grouping. |
| `select_workgroup_mapping`: `wgmxccchunk`, `wgmxccsplitk` | Retained in every candidate's `modeled.workgroup_mapping` | Nonzero values require the Stream-K mapping ABI and reject the current data-parallel candidate; neither is substituted with `WorkGroupMappingXCCGroup`. |
| `select_staggerU`: `staggerU`, `staggerUMapping` | `StaggerU`, `StaggerUMapping` | All results, including zero, are supplied and checked after derivation. Origami currently returns zero for batches, K splitting, and several no-benefit conditions. |
| `select_staggerU`: `staggerUStrideShift` | `StaggerUStride = DepthU × Tensile DataType bytes × 2^shift` | Check the derived `_staggerStrideShift`; zero stagger may normalize the byte stride without changing its meaning. |
| `gemm::compute_launch_parameters`: reduction, grid, active CUs, timesteps, split factor | `modeled.launch`; `StreamK=0`, `GlobalSplitU=1` | Data parallel derives `none`, output-tile grid and split factor 1. Active CUs/timesteps describe the model, not kernel tuning fields. |
| `streamk::select_reduction`, `select_grid_size` | Mode-dependent prediction APIs | Applicable when a caller enables Stream-K. The data-parallel domain has no reduction/grid tuning prediction to default. Adding Stream-K candidates requires preserving these outputs through Tensile's workspace and launch reconciliation. |
| `streamk::select_hybrid_mode` | Static/dynamic schedule within StreamK=5 | Does not select Stream-K enablement. Inapplicable to the current data-parallel domain. |
| `gemm::predict_workgroup_mapping` | Internal latency-estimation approximation | Alternative fast mapping estimate, not an additional kernel field. The generator receives the full `select_workgroup_mapping` result. |
| Hardware `get_recommended_matrix_instruction` | Alternative throughput-based MI choice | The provider uses the full instruction catalog plus ranking, preserving the selected MI. |
| GEMM/Formocast performance and resource estimates | Scores/diagnostics | These APIs estimate latency/utilization/resource costs; they do not predict new vector widths, occupancy, or backend tuning settings. |

Unpredicted inputs include wave topology, occupancy, Stream-K enablement and grid
policy, workspace limits, vector widths, subtile/main-loop choice, prefetch and
scheduling, direct-to-LDS/VGPR settings, load coalescing, swizzle/layout and
split-U policy. Some are fixed by the request or candidate domain; others retain
Tensile defaults and derivation. Formocast consumes additional backend settings
to estimate cost; it does not fill them in. Epilogue overhead is not modeled.

The selector rejects missing modeled fields, unsupported translations, and any
derived recipe that changes a modeled value or cannot carry it at runtime.
Rejection advances to the next ranked candidate; exhausting the ranking fails
with reasons and emits no selected recipe. A CU budget smaller than the device's
XCD count is rejected before calling the mapping selectors. Diagnostic manifests
retain raw outputs, translated parameters, defaults, and rejections. Explicit
recipes continue through `Tensile.SingleSolution` without this prediction contract.

## Build

`HIPBLASLT_ENABLE_JIT` is disabled by default. The declarations remain
available when disabled, and selection returns `HIPBLAS_STATUS_NOT_SUPPORTED`.
The enabled implementation requires the host library and ROCm. The TensileLite
provider also requires its Python dependencies and the local rocisa extension.
Provider processes use POSIX spawning on Linux and `CreateProcessW` on Windows;
additional Python import paths use the platform separator (`:` or `;`).
Using an existing configured build and Python environment:

```bash
cmake -S "$project_root/projects/hipblaslt" -B "$project_build" \
  -DHIPBLASLT_ENABLE_JIT=ON -DHIPBLASLT_ENABLE_HOST=ON \
  -DHIPBLASLT_BUILD_TESTING=ON \
  -DHIPBLASLT_ENABLE_DEVICE=OFF -DGPU_TARGETS=gfx950 \
  -DPython_EXECUTABLE="$project_python" -DPython3_EXECUTABLE="$project_python"
cmake --build "$project_build" --target _rocisa hipblaslt-jit-generic-api-test --parallel
export PYTHONPATH="$project_build/tensilelite/rocisa:$project_build/tensilelite:$project_root/projects/hipblaslt/tensilelite"
```

Use the compiler and target appropriate for the local device. Generated
bundles do not depend on a prebuilt hipBLASLt device library.

The `jit` CMake preset enables this feature for a new configuration. The
commands above show how to enable it in an existing build. The shared workflow
configures native gfx90a, gfx942, gfx950 and gfx1250 runners for both API variants
and an independent test backend. Local integration execution is validated on
gfx950; the gfx1250 SIA4 fixture also has separate compilation evidence.
Configured targets do not imply completed native runs.

## Algorithm lifetime and validation

The returned heuristic result contains the required workspace size. Supply
that workspace and follow the same handle, stream, and workspace sharing rules
as `hipblasLtMatmul` and `Gemm` calls using prebuilt algorithms. All helper entrypoints are resolved before submission.
Stream-K uses the handle's stream-specific synchronization region;
MultipleBufferSingleKernel and output-amax use its shared synchronization
storage. Registry synchronization protects algorithm lookup; it does not protect
application buffers or make simultaneous calls on one `Gemm` object safe.

Copies of an algorithm remain usable on its generating device within the same
process. Its modules are retained until process exit. Reuse within one program invocation needs no recompilation. A different program
invocation compiles its own solution: no persistent cache or bundle reload is
provided. Save recipes and diagnostic manifests for reproduction; never persist
the opaque algorithm bytes or treat them as a prebuilt library index.

An empty GEMM output (M=0 or N=0) returns NOT_SUPPORTED from the request factory
without compilation. K=0 can use a recipe that implements beta*C. TensileLite
owns its datatype, instruction and scale-layout restrictions. The library
propagates provider support failures, including a mismatch between the supplied
physical MX scale layout and the compiled solution.

`clients/tests/jit/test_helper_failures.py` removes helper modules or symbols from a
valid split-K bundle, checks that C/extension paths leave D and workspace
untouched, and verifies that failed reinitialization preserves the previous
extension algorithm. `clients/tests/jit/test_bundle_failures.py` checks malformed
envelopes, missing code, mismatched solution identity and unsupported problems
through the same public API. Both scripts run in the shared JIT workflow.

Explicit recipes use TensileLite's target and solution
validators. Output-amax currently requires one batch, GlobalSplitU=1 and
StreamK=0. Generation never benchmarks recipes or substitutes another recipe
when the supplied one fails.

The [single-solution documentation](tensilelite/SINGLE_SOLUTION.md)
describes the recipe, bundle, and Python builder contracts.

The private provider loader reads `loader.bin`, a bounded, versioned envelope
published alongside the human-readable `manifest.json`. Corruption tests modify
the consumed envelope or code objects; JavaScript Object Notation (JSON) is a diagnostic record.

A searchable JIT solution library and persistent code cache are future work.
Such a library could index solutions by problem description and look up compatible
code before requesting generation. The current API retains explicitly selected
algorithms in one process; it does not search a JIT collection or reuse code from
an earlier program invocation. The design discussion assesses a fallback library
that generates only after complete existing lookup yields zero compatible results,
including Equality and Origami-based selection. It is not implemented and would
share generation/validation with the explicit API, which remains useful for
backend choice and prewarming. Neither adding a library type nor generic handles
alone provides a non-GEMM operation adapter.


The [component roadmap](JIT_ROADMAP.md) describes the complete flow and remaining planning, search and cache work.
