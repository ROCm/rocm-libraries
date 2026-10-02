# hipBLASLt just-in-time (JIT) GEMM generation

This source guide describes just-in-time (JIT) kernel generation for general
matrix multiplication (GEMM) in hipBLASLt. It is written for hipBLASLt and
TensileLite contributors and integration developers, and it is maintained under
the existing @ROCm/hipblaslt-reviewers and @ROCm/hipblaslt-docs-reviewers rules
in [.github/CODEOWNERS](../../.github/CODEOWNERS). It is not a released
application programming interface (API) or support statement. Release-document
integration is undecided.

It describes what the code implements today. "Implemented" means present in the
source, not released or approved as product naming. A JIT backend is the
generator that turns a request into kernel sources; this page describes the
parts of hipBLASLt around any backend. The
[JIT test guide](clients/tests/jit/README.md) covers building and running the
tests.

## Summary

Applications reach JIT generation through the existing heuristic query. In a
build with `HIPBLASLT_ENABLE_JIT=ON`, the environment variable `HIPBLASLT_JIT`
lets `hipblasLtMatmulAlgoGetHeuristic`, `GemmInstance::algoGetHeuristic` and
`hipblasLtMatmul` without an algorithm return solutions from a persistent JIT
solution library on disk. Solutions that the library lacks are generated
by the build's JIT backends, built with comgr, published into the library, and
returned by solution index, so any later process runs them without generating
again. In fallback mode JIT is
consulted after the pre-tuned Equality results and before the other pre-tuned
libraries; in forced mode JIT is the only source.
See [heuristic integration](#heuristic-integration).

The JIT test binaries also call internal entry points directly: they pass GEMM
descriptors, hipBLASLt generates and builds one solution through a backend, and
the test passes the returned algorithm to `hipblasLtMatmul` or
`hipblaslt_ext::Gemm`, or publishes generated solutions into the JIT solution
library. These entry points are not part of the public API.

## Current behavior

### Entry points

The internal generic entry point generates through the Jit component, keeps
generated algorithms in one process-local registry, and returns an algorithm
for the existing C/C++ GEMM execution APIs.

| Entry point | Header | Behavior |
| --- | --- | --- |
| Generic request/backend/solution | `library/src/amd_detail/hipblaslt-jit.hpp` | A backend factory, such as `jit::mock::createBackend`, returns a `Backend` handle that owns a Jit configured with that backend. `jit::makeGemmRequest` captures existing GEMM descriptors and host scalars, `jit::getJitAlgo` compiles on the selected device and returns an owned `Solution`, and `jit::getGemmAlgo` adapts it to the algorithm accepted by `hipblasLtMatmul` and `Gemm`. The `hipblaslt-jit-api-test` binary exercises it. `jit::getLibraryAlgos` instead returns solution indices from the [JIT solution library](#persistent-solution-library), generating and publishing the solutions it lacks; `hipblaslt_ext::getAlgosFromIndex` turns them into algorithms. |

The header is internal, as are `hipblaslt-jit-mock.hpp` and
`hipblaslt-jit-gemm-internal.hpp`: they are not installed and
`hipblaslt-ext.hpp` does not include them. `libhipblaslt.so` exports four
functions and one type from them with `HIPBLASLT_EXPORT` for the JIT test
binaries, which link against the shared library: `jit::makeGemmRequest`,
`jit::getJitAlgo`, `jit::getGemmAlgo`, `jit::getLibraryAlgos` and the
`jit::detail::GemmRequest` request type. A build with
`HIPBLASLT_JIT_TESTING=ON` also exports `jit::mock::createBackend`. No
installed header declares them, and they are not a supported API.

For these entry points, the backend's configuration belongs to the options of
its factory; heuristic queries use the process's backend instead, as
[heuristic integration](#heuristic-integration) describes. The application
owns its buffers and workspace. The
request owns descriptor values and host scalars; it does not take ownership of
device pointers. Compilation and support checks finish before graphics
processing unit (GPU) work is submitted; call the entry points before stream
capture. The Jit interfaces are for
compiled-in implementations and do not establish a stable external plugin
application binary interface (ABI). GEMM is the implemented operation; an
attention request adapter and backend remain future work.

Internally, the GEMM request reuses `RocblasltContractionProblem` with owned
scalar values. The generic `Solution` and private `CompiledSolution` retain
the Jit, device target, request, workspace and bundle lifetime around the
existing GEMM support and execution machinery. A matmul algorithm is an
adaptation token, not a general owning executable object.

### Components

| Component | Current input, output and connection |
| --- | --- |
| Jit | `hipblaslt-jit-component.{hpp,cpp}`. For one request and device target, `Jit::generate` runs the predictor when the backend consumes a prediction, asks the backend for solutions in a private scratch directory, builds each solution's code objects, checks support, and loads the supported ones as process-local bundles. With a solution store, it publishes them instead and loads them only when publishing fails; `getLibraryAlgos` configures the JIT solution library as the store. Each failure records its stage (configure, predict, generate, build, support, load or publish), and `getJitAlgo` reports the first one. The scratch directory is removed on success and kept after a failure that left files in it. |
| Origami predictor | `hipblaslt-jit-origami-predictor.cpp` expands the tuning knowledge's candidate seeds across the target's matrix instructions, ranks them with Origami, and emits the `origami.gemm.dp.v1` modeled contract (workgroup mapping, stagger and launch outputs), which a backend that consumes the contract receives as the request's prediction. Jit runs it only for such a backend. |
| Catalog knowledge | `hipblaslt-jit-catalog-knowledge.cpp` (`makeCatalogKnowledge()`, id `catalog.v1`) is the tuning knowledge: 11 tile shapes, two DepthU rules, and cache hints that are only the defaults on gfx90a and gfx1250. It supplies no values for unmodeled knobs, which keep the generator's defaults. |
| Code-object builder | `hipblaslt-jit-builder.cpp` over `hipblaslt-jit-code-object.cpp`. The comgr builder assembles the main kernels, compiles the helper source and links both into one raw executable code object for the device's target ID, then checks that the object targets that ID and defines the entry's kernel. See [code-object construction with comgr](#code-object-construction-with-comgr). |
| Loader | `hipblaslt-jit-loader.cpp`. It reads source bundles by directory convention for the backends. The Tensile loader parses the entry, checks support and workspace with TensileLite's predicates, and loads the code object into a `TensileBundle`. |
| JIT solution library | `hipblaslt-jit-library.cpp`, with `hipblaslt-jit-msgpack.cpp` writing the library files and `hipblaslt-jit-fs.cpp` providing the directory checks, file lock and atomic replacement. It publishes built solutions as a standard lazy TensileLite library on disk, looks them up by exact problem, and resolves their reserved solution indices for `tensile_host.cpp`. See [persistent solution library](#persistent-solution-library). |
| Generic entry point and adapters | `makeGemmRequest`, `getJitAlgo` and `getGemmAlgo` connect Jit to existing execution. `hipblaslt-jit-api-test` covers this flow on replayed solutions. |
| Heuristic integration | `hipblaslt-jit-mode.cpp` reads `HIPBLASLT_JIT`. `hipblaslt-jit-process-backend.cpp` configures one Jit per process for each backend that `HIPBLASLT_JIT_BACKENDS` selects, from the backend, predictor and tuning knowledge that `makeDefaultProcessBackend` or an opt-in provider returns, with the JIT solution library as its store, and `hipblaslt-jit-backend.cpp` looks a problem up in that library and generates what it lacks, backend by backend. `hipblaslt-jit-report.cpp` prints failures. `rocblaslt_auxiliary.cpp` calls them from both heuristic queries, and `tensile_host.cpp` from `hipblasLtMatmul` without an algorithm. See [heuristic integration](#heuristic-integration). |
| Benchmark | `hipblaslt-bench` has no JIT option. With `HIPBLASLT_JIT=2`, its ordinary heuristic query returns generated solutions, so selection and compilation finish before correctness checks and execution timing. |
| Mock backend | `hipblaslt-jit-mock-backend.cpp` replays a list of source bundles, without a generator or a subprocess. A generation returns, in list order, up to the requested count of bundles whose predicates accept the device and problem, skipping excluded kernels; the comgr builder still builds them. Its faults fail generation and leave a log in the scratch directory, replace the main kernel assembly with an invalid instruction so the build fails, append the request to a file and fail, or abort the process. Given a modeled contract it consumes the predictions of the predictor for that contract. Tests reach it through `jit::mock::createBackend` in `hipblaslt-jit-mock.hpp`, and builds with `HIPBLASLT_JIT_TESTING=ON` and no generator backend use it for heuristic queries; it is not a production backend. |

Each build links one definition of `makeDefaultProcessBackend`, chosen when
hipBLASLt is configured. Without a generator backend it comes from
`hipblaslt-jit-no-backend.cpp`, whose configure failure says that hipBLASLt was
built without a JIT generator backend, or, with `HIPBLASLT_JIT_TESTING=ON`,
from `hipblaslt-jit-test-backend.cpp`, which replays bundles through the mock
backend; the [JIT test guide](clients/tests/jit/README.md) describes its
settings. A build that configures a generator backend links that backend's
definition instead. A build also links one definition of
`optInProcessBackends`, the backends that serve heuristic queries only when
`HIPBLASLT_JIT_BACKENDS` names them: `hipblaslt-jit-no-opt-in-backend.cpp`
lists none, and a build that configures an opt-in backend links its list
instead.

### Origami modeled inputs

Jit runs the Origami predictor for a backend that consumes the
`origami.gemm.dp.v1` contract. This private contract covers the existing data-parallel candidate
domain. Origami ranks caller-supplied configurations; it does not synthesize
their fields or choose whether to enable Stream-K. Origami's `stream_k=0` and
occupancy 1 remain explicit model inputs, and the recipe keeps TensileLite's
non-persistent `TileProcessingStrategy=None`. Every applicable prediction is transferred;
defaults supply only settings that this model does not predict.

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
| `gemm::compute_launch_parameters`: reduction, grid, active CUs, timesteps, split factor | `modeled.launch`; `TileProcessingStrategy=None`, `GlobalSplitU=1` | Data parallel derives `none`, output-tile grid and split factor 1. Active CUs/timesteps describe the model, not kernel tuning fields. |
| `streamk::select_reduction`, `select_grid_size` | Mode-dependent prediction APIs | Applicable when a caller enables Stream-K. The data-parallel domain has no reduction/grid tuning prediction to default. Adding Stream-K candidates requires preserving these outputs through Tensile's workspace and launch reconciliation. |
| `streamk::select_hybrid_mode` | Static/dynamic schedule of a Stream-K kernel with `WorkAssignment=Hybrid` | Does not select Stream-K enablement. Inapplicable to the current data-parallel domain. |
| `gemm::predict_workgroup_mapping` | Internal latency-estimation approximation | Alternative fast mapping estimate, not an additional kernel field. The generator receives the full `select_workgroup_mapping` result. |
| Hardware `get_recommended_matrix_instruction` | Alternative throughput-based MI choice | The predictor uses the full instruction catalog plus ranking, preserving the selected MI. |
| GEMM/Formocast performance and resource estimates | Scores/diagnostics | These APIs estimate latency/utilization/resource costs; they do not predict new vector widths, occupancy, or backend tuning settings. |

Unpredicted inputs include wave topology, occupancy, Stream-K enablement and grid
policy, workspace limits, vector widths, subtile/main-loop choice, prefetch and
scheduling, direct-to-LDS/VGPR settings, load coalescing, swizzle/layout and
split-U policy. Some are fixed by the request or candidate domain; others retain
Tensile defaults and derivation. Formocast consumes additional backend settings
to estimate cost; it does not fill them in. Epilogue overhead is not modeled.

A backend that consumes the contract rejects missing modeled fields,
unsupported translations, and any derived recipe that changes a modeled value
or cannot carry it at runtime.
Rejection advances to the next ranked candidate; exhausting the ranking fails
with reasons and emits no selected recipe. A CU budget smaller than the device's
XCD count is rejected before calling the mapping selectors. Diagnostic manifests
retain raw outputs, translated parameters, defaults, and rejections.

### Pre-tuned heuristic selection

The installed library answers `hipblasLtMatmulAlgoGetHeuristic` and
`GemmInstance::algoGetHeuristic` from pre-tuned TensileLite libraries. Default
construction orders the Equality, Range, Prediction, GridBased and FreeSize
selection libraries, followed by TruePred rows, beneath hardware, operation and
problem predicates. Modes and available branches affect traversal. Equality is
matching with equality distance.

The Prediction library type (C++ `ProblemPredictionLibrary`) is referred to as
the **OrigamiLibrary** in the target design; it is not a new type. It ships 507
pre-tuned solution YAML files in per-architecture `Origami` directories under
`library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/`: 489 for gfx950,
7 each for gfx1250 and gfx1250v0, and 4 for navi32. At runtime it ranks those
existing solutions with `origami::rank_configs`. Origami does not construct
kernels at package time. This existing-solution ranking is distinct from the JIT
predictor, which ranks candidate recipes that may not exist in any library.

When a problem that requests xf32 math (`rocblaslt_compute_f32_fast_xf32`)
finds no solution, `getBestSolutions` in `tensile_host.cpp` repeats the lookup
with FP32 math. When `getBestSolutions` returns fewer results than
`requestedAlgoCount`, the existing heuristic code in `rocblaslt_auxiliary.cpp`
calls `getAllSolutions`, excluding the GridBased and Prediction libraries that
were already consulted, and appends supported, non-duplicate solutions until the
count is reached. For some eligible problems, another hipBLASLt route answers
`getBestSolutions` and `getAllSolutions` before the Tensile lookup. With
`HIPBLASLT_JIT` set, the [heuristic integration](#heuristic-integration)
extends or replaces this lookup.

### Build

`HIPBLASLT_ENABLE_JIT` is disabled by default. A disabled build compiles and
exports no JIT entry points. The enabled implementation requires the host
library, ROCm and ROCm's `amd_comgr` CMake package, which only a JIT build
links. comgr compiles the helper source against the host's C and C++ standard
library headers, so those must be installed where JIT runs. A backend can need
more; its configure failure names what is missing. From the repository root:

```bash
project_root="$PWD"
project_build="$project_root/projects/hipblaslt/build/release"
cmake -S "$project_root/projects/hipblaslt" -B "$project_build" \
  -DHIPBLASLT_ENABLE_JIT=ON -DHIPBLASLT_ENABLE_HOST=ON \
  -DHIPBLASLT_BUILD_TESTING=ON \
  -DHIPBLASLT_ENABLE_DEVICE=OFF -DGPU_TARGETS=gfx950
cmake --build "$project_build" --parallel
export LD_LIBRARY_PATH="$project_build/library:$project_build/tensilelite${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Use the compiler and target appropriate for the local device. A ROCm installation can ship its own `libhipblaslt` and
`libtensilelite-host`; keep the build tree's `library` and `tensilelite`
directories ahead of them on `LD_LIBRARY_PATH`, or the loader picks the prebuilt
copies. Generated bundles do not depend on a prebuilt hipBLASLt device library.
The `jit` CMake preset enables this feature for a new configuration. The
[JIT test guide](clients/tests/jit/README.md) lists the test targets and the
validation commands.

### Code-object construction with comgr

Generators emit assembly or HIP source plus metadata only; they do not assemble,
link or bundle. hipBLASLt C++ builds code objects in process through AMD comgr
(`hipblaslt-jit-code-object.cpp`). The builder uses three comgr actions:

- Assembly: `AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE`, after the
  `.amdgcn_target` directive is rewritten to the device's full target ID.
- HIP helper source: `AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE` with
  `--rocm-path` and a content-derived `-cuid`, so helper objects link together.
- Link: `AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE` joins the main kernel
  and helper relocatables into one code object per solution, with
  `-Xlinker --build-id=sha1`.

The generator and the builder use the same code-object version, which
`GenerationRequest::codeObjectVersion` carries (4 by default). The output is a
raw, uncompressed executable code object. comgr cannot bundle or compress it,
and `hipModuleLoadData` accepts raw executable and linkable format (ELF)
objects. A generator needs no offload bundler.

comgr's own on-disk cache (`~/.cache/comgr`) keeps its default in every
`HIPBLASLT_JIT` mode: hipBLASLt never sets `AMD_COMGR_CACHE`. That cache is
separate from the JIT solution library and serves a different purpose. It holds
the results of comgr actions for every comgr user in the process, such as
hipRTC, and comgr reads its setting once per process. The JIT solution library
holds the solutions that hipBLASLt generated, and its
[cache key](#persistent-solution-library) ignores the variables that control
only comgr's cache.

### Algorithm lifetime and failures

The returned heuristic result contains the required workspace size. Supply that
workspace and follow the same handle, stream and workspace sharing rules as
`hipblasLtMatmul` and `Gemm` calls using prebuilt algorithms. All helper
entrypoints are resolved before submission. Stream-K uses the handle's
stream-specific synchronization region; MultipleBufferSingleKernel and
output-amax use its shared synchronization storage. Registry synchronization
protects algorithm lookup; it does not protect application buffers or make
simultaneous calls on one `Gemm` object safe.

Copies of an algorithm remain usable on its generating device within the same
process, and its modules are retained until process exit. Reuse within one
program invocation needs no recompilation. The opaque algorithm bytes that
`getJitAlgo` and `getGemmAlgo` return are not a library index, and a different
program invocation cannot use them. To reuse a solution across processes,
publish it with `getLibraryAlgos` and keep its solution index. Save the
bundle manifests for reproduction.

An empty GEMM output (M=0 or N=0) returns `HIPBLAS_STATUS_NOT_SUPPORTED` from
the request factory without compilation. K=0 can use a solution that
implements beta*C. The backend owns its datatype, instruction and scale-layout
restrictions. The library propagates support failures, including a mismatch
between the supplied physical MX scale layout and the compiled solution. Jit
never benchmarks generated solutions.

hipBLASLt reads a source bundle by directory convention:
`library/TensileLibrary.*` is the one-solution entry, `sources/*.s` are the
main kernels, `sources/Kernels.cpp` holds the helper kernels, and the other
files in `sources/` are the headers it includes. The file count and sizes are
bounded, and symbolic links must stay inside the bundle. The JavaScript Object
Notation (JSON) `manifest.json` is a provenance record that hipBLASLt does not
read. `clients/tests/jit/test_helper_failures.py` and `test_bundle_failures.py`
damage valid bundles and check that the public C and extension paths reject
them without writing output or workspace. A failed build names the retained
`comgr.log` in its message.

The tests that need no generator replay gfx950 source bundles committed in
`clients/tests/jit/data`. Their manifests record the kernel-argument and
persistent-loop argument layout versions of the generator that wrote them, and
the `jit-bundle-freshness` test fails when those differ from the ones in
`tensilelite/Tensile/Common/GlobalParameters.py`,
when the code-object version differs from the builder's, or when a bundle no
longer reads or builds. [Their README](clients/tests/jit/data/README.md) gives
the commands that regenerate them.

### Persistent solution library

The JIT solution library keeps generated solutions on disk so that later
processes run them without generating again. `jit::getLibraryAlgos` is its
entry point: for one request it returns up to the requested number of solution
indices, first the published solutions that match the request, in the order
they were first published, then solutions that Jit generates with the supplied
backend and publishes. Any process then passes an index to
`hipblaslt_ext::getAlgosFromIndex`, `hipblasLtMatmul` and `Gemm` as it would a
prebuilt index. With `HIPBLASLT_JIT` set, heuristic queries and
`hipblasLtMatmul` without an algorithm use the same lookup and publication; see
[heuristic integration](#heuristic-integration).

**Location and permissions.** The root is `HIPBLASLT_JIT_LIBRARY_PATH` or, when
that is unset or empty, `/tmp/hipblaslt-jit-<uid>/` on Linux and
`%TEMP%\hipblaslt-jit-<user>` on Windows. hipBLASLt creates missing directories
with mode 0700. On Linux it refuses the root, and each directory it uses below
the root, when the path is a symbolic link or not a directory, is owned by
another user, or is writable by group or others; a readable directory such as
0755 is accepted. Windows checks only for reparse points and non-directories.
There is no setting that accepts a shared, writable directory. A refused
directory, a privileged (setuid or setgid) process and a build that reads YAML
libraries disable the library for the process, and each use then fails with
`JIT solution library disabled: <reason>`. A prebuilt library that already uses
the reserved index range also disables it.

**Layout.** Under a `v1` schema directory, each cache key has its own
directory, named after the ISA and a 64-bit hash of the key. Each key directory
is a standard lazy TensileLite library that the stock loader reads: a master
file, the index mapping, and one entry (`.dat`) and code object (`.co`) per
solution. For example:

```text
$HIPBLASLT_JIT_LIBRARY_PATH/
└── v1/
    ├── allocator.dat                  next free solution index
    ├── lock                           publication lock for the whole root
    └── gfx950-677afeae9fbc7f02/       one directory per cache key
        ├── cache-key.json
        ├── TensileLibrary_lazy_gfx950.dat
        ├── TensileLiteLibrary_lazy_gfx950_Mapping.dat
        ├── TensileLibrary_JIT_ad7857c9cccf2981_555e7809bfd8786c.dat
        ├── TensileLibrary_JIT_ad7857c9cccf2981_555e7809bfd8786c.co
        └── staging/                   temporary files before rename
```

An entry name is `TensileLibrary_JIT_<ProblemType hash>_<kernel and size hash>`,
so publishers of the same solution for the same problem choose the same files.
When a name already holds another kernel, the publisher appends `_1`, `_2` and
so on. Each entry holds one solution, with its solution index rewritten to the
allocated one. Each master row is
`And(SizeEqual(M), SizeEqual(N), SizeEqual(batch), SizeEqual(K), <the entry's
problem predicate>)` pointing to a placeholder for the entry, so a lookup
matches only the exact sizes within the full ProblemType. The solution's own
hardware, problem and task predicates, including its workspace requirement,
still run on every match.

**Cache key.** `cache-key.json` records, in canonical JSON:

| Field | Contents |
| --- | --- |
| `target` | Full target ID with features (for example `gfx950:sramecc+:xnack-`), ISA, TensileLite library architecture and wavefront size |
| `backend` | Backend identifier and version. Each backend defines its version to change whenever its output can; the mock backend's is a hash of its replayed bundles. |
| `comgr` | comgr version, and on Linux the path, size and modification time of the loaded comgr library |
| `code_object_version` | The code-object version that the generator and the builder use (4) |
| `rocm_path` | The ROCm path that the builder passes to comgr |
| `compiler_environment` | `HIP_PATH`, `LLVM_PATH`, and every set `AMD_COMGR_*` variable, such as `AMD_COMGR_DRIVER_OPTIONS_APPEND` and `AMD_COMGR_HOTSWAP_*`, except those that control only comgr's cache, logging, temporary files and statistics |
| `schema` | The library schema version (1) |

A process uses only the directory whose name and `cache-key.json` both match
its key exactly. It ignores other directories and never modifies or deletes
them, and a directory whose `cache-key.json` holds a different key fails the
lookup or publication with a message that says so. To resolve an index it has
not looked up, a process searches the directories for its device whose key
matches everything except the backend, so a later process can run the solution
without configuring the backend that generated it.

**Indices.** JIT solutions use solution indices from 2^30 to `INT32_MAX`.
Prebuilt libraries stay below that range: `TensileCreateLibrary` stops with an
error when a library's solution indices would reach 2^30. `allocator.dat` holds
the next index, so
indices stay unique across every key directory under a root and are never
reused while the root exists. Publication fails once the range is exhausted.
`tensile_host.cpp` routes reserved indices to the JIT solution library, with its
own solution adapter whose code-object directory is the key directory, and all
other indices to the prebuilt library. An index that no JIT library holds
resolves to an empty library, so callers report their usual missing-solution
error. For an index that names no solution in either range,
`hipblaslt_ext::Gemm::initialize` and `GroupedGemm::initialize` return
`HIPBLAS_STATUS_INVALID_VALUE`, and `hipblasLtMatmul` returns
`HIPBLAS_STATUS_INTERNAL_ERROR`; problems that another hipBLASLt route answers
before the Tensile lookup do not reach this check. The `AlgoErrors` tests in
`hipblaslt-test` check these statuses with index 2^30 − 1, the last index below
the reserved range; see [validation](#validation).

**Publication and refresh.** A publisher holds `lock` (waiting up to 120
seconds) while it allocates indices and writes, in this order: `allocator.dat`,
the code objects, the entries, the mapping and the master. Each file is written
under `staging/` and renamed into place, so readers never see a partial file.
A process that stops at any point leaves a library that loads, whose master
refers only to complete entries; the next publisher reuses or overwrites what
it left. Republishing a published solution writes nothing and returns its
index. Each lookup reloads the master when another process has replaced the
master or mapping file, and resolving an unknown index reloads it too. A
solution already resolved from an earlier master stays valid.

**Clearing.** hipBLASLt never deletes entries. To clear the library, delete
the root directory, or one key directory, while no process is using it.

### Heuristic integration

`HIPBLASLT_JIT` selects the JIT mode for the process. hipBLASLt reads it once,
when the first handle is created.

| `HIPBLASLT_JIT` | Mode | Behavior |
| --- | --- | --- |
| unset, empty or `0` | Off | Heuristic queries and `hipblasLtMatmul` behave as in a build without JIT. |
| `1` | Fallback | JIT comes after the Equality results: a query takes the Equality results, then JIT solutions, then the results of the other pre-tuned libraries and the `getAllSolutions` fill, each only for what is still missing. |
| `2` | Forced | JIT is the only source. The query skips the override file, every pre-tuned library, every other hipBLASLt source and the `getAllSolutions` fill. |

Any other value leaves JIT off and prints
`hipblaslt warning: HIPBLASLT_JIT=<value> is not 0, 1 or 2; JIT is off` once.
A build without JIT ignores a nonzero value and prints
`hipblaslt warning: HIPBLASLT_JIT=<value> is ignored: hipBLASLt was built without HIPBLASLT_ENABLE_JIT`
once.

**Order.** In fallback mode, `hipblasLtMatmulAlgoGetHeuristic` and
`GemmInstance::algoGetHeuristic` take results from these sources in turn, each
only for what is still missing from `requestedAlgoCount`:

1. The override file.
2. The Equality rows of the pre-tuned libraries.
3. JIT. It looks the problem up in the JIT solution library under the process's
   cache key. If that is still short, Jit generates the rest: the predictor
   ranks candidates when the backend consumes a prediction, the backend
   generates solutions, comgr builds them, and the library publishes them.
   Every kernel that the query has already returned is excluded by name, and
   the backend returns only kernels it has not already returned for the
   request, so each new result is a different kernel.
4. The other pre-tuned libraries: the Range, Prediction (Origami), GridBased and
   FreeSize rows and the MLP rows.
5. The `getAllSolutions` fill.

JIT results count toward the request, so a query that the Equality results
fill does not consult JIT. Its results can still differ from those with JIT
off. The pre-tuned library walks its hardware branches in order, a branch that
names a device or CU count before the generic one. With JIT off, a single walk
takes each branch's Equality rows and then its other rows, and stops when the
request is full. The fallback-mode Equality pass takes the Equality rows of
every branch first, so when the first matching branch's Equality rows do not
fill the request, an Equality result of the generic branch can take a place
that, with JIT off, goes to a result of the specific branch's other rows. The
results are those of JIT off when only one branch matches or when the first
matching branch's Equality rows fill the request. The sources after JIT skip
the kernels its results use. When the override entry names a JIT solution, the
JIT results do not repeat its kernel. When neither pre-tuned
pass finds a solution for an xf32 problem, both repeat with FP32 math; JIT runs
once for the query. When another hipBLASLt route answers the problem before the
Tensile lookup, its results come first and JIT fills only what is still missing
after the `getAllSolutions` fill.

In forced mode, the JIT lookup and generation are the whole query. Each JIT
result passes the same support and workspace checks as a `getAllSolutions`
result. Its solution index is in the reserved JIT range.

**Backends.** `HIPBLASLT_JIT_BACKENDS` lists the backends that serve the JIT
step, by identifier and separated by commas, in the order they serve it.
Unset or empty, the build's default backend serves alone, and opt-in backends
are not configured. hipBLASLt reads it when the first query reaches the JIT
step. An identifier the build lacks prints
`hipblaslt warning: JIT backend <id> in HIPBLASLT_JIT_BACKENDS is not in this build; ignored`
once; when the list names no backend of the build, every query reports a
configure failure. Each backend has its own cache key, so its solutions sit in
their own key directory of the JIT solution library. In the JIT step a
backend that does not serve the problem or the device is skipped without a
report. The others serve in turn, each looking up and generating what is still
missing, less one result kept for each later one, and excluding the kernels of
the earlier ones; a backend that fails or falls short leaves its share to the
next. A query therefore returns a result of every such backend only when it
requests at least as many results as there are backends, and `hipblasLtMatmul`
without an algorithm runs a solution of the first one that returns one. A
backend whose
configuration fails is reported like any other failure, once per problem, and
the others still serve. When no backend serves the problem, the query reports
`JIT generate failed for <problem>: no enabled JIT backend supports this problem`.
With several backends, each failure message names its backend, and a
shortfall warning lists each backend's count and summary.

**Return count.** `hipblasLtMatmulAlgoGetHeuristic` sets `*returnAlgoCount` to 0
before it validates the request, so a rejected request also reports no results:
a `requestedAlgoCount` below 1 returns `HIPBLAS_STATUS_INVALID_VALUE` with a
count of 0. Only a null argument, and in builds with fused all-to-all a
rejected all-to-all epilogue, return before the count is set. Returning
fewer results than requested, including none, is success, as it already was for
the pre-tuned lookup. In fallback mode, a query whose pre-tuned lookup failed,
for example because no pre-tuned library could be loaded, succeeds when JIT adds
a result and otherwise keeps its error. In forced mode the query succeeds even
when JIT fails; it then returns no results and reports the failure.

**`hipblasLtMatmul` without an algorithm.** In fallback mode it runs the first
solution of the same order: an Equality result, else a JIT solution, else a
result of the other pre-tuned libraries. When another hipBLASLt route answers the
problem before the Tensile lookup, it uses a JIT solution only when that route
finds none. In forced mode it uses only
JIT solutions. When no solution is found it returns
`HIPBLAS_STATUS_INTERNAL_ERROR`.

**Failure reporting.** JIT problems are printed on stderr without any
`HIPBLASLT_LOG_LEVEL` setting, once per distinct message in a process, and are
also passed to the existing error or info log. A failure is an error when the
query gets no JIT result and a warning when JIT still added one. A shortfall
without a failure is a warning. Each line names the stage (configure, predict,
generate, build, support, load, lookup or publish), the problem and the cause.
A generation or build failure names its kept log:

```text
hipblaslt error: JIT configure failed for GEMM M=256 N=128 K=512 batch=1 opA=OP_N opB=OP_N A=R_16F B=R_16F C=R_16F D=R_16F compute=COMPUTE_32F epilogue=EPILOGUE_DEFAULT: hipBLASLt was built without a JIT generator backend
hipblaslt error: JIT predict failed for GEMM M=256 N=128 K=0 ... EPILOGUE_DEFAULT: JIT GEMM prediction: No Origami ranking: no finite positive-latency candidates for this request
hipblaslt error: JIT build failed for GEMM M=256 N=128 K=512 ... EPILOGUE_DEFAULT: comgr could not assemble: /tmp/comgr-3939069-4-b17d31/input/0-<kernel>.s:1:1: error: invalid instruction; see /tmp/hipblaslt-jit-Vr6yTk/comgr.log
```

A shortfall warning, such as `hipblaslt warning: JIT returned 1 of 2 requested
solutions for GEMM ...`, ends with the backend's summary of its generation.

Once a backend's generation falls short for a problem, later queries for it in
the same process look it up in that backend's entries but do not generate with
it again. Queries for the
same problem and workspace limit in one process generate one at a time, so the
second one finds what the first published. Separate processes can generate the
same solutions at once; they publish under the library lock, which keeps one
entry and one index per solution, so every process returns the same indices.
An empty output (M=0 or N=0) gets no JIT result. A problem that Origami cannot
rank, such as K=0, reports a predict failure. Grouped GEMM is not supported and
reports an error. Fused GEMM and all-to-all, in builds with
`HIPBLASLT_ENABLE_GEMM_A2A_FUSION`, is not implemented: the JIT ProblemType has
no fused all-to-all, so JIT neither looks such a problem up nor generates or
publishes a solution for it. The query reports one lookup failure per problem;
in fallback mode the pre-tuned sources still answer it, and in forced mode it
returns no results. A JIT algorithm passed for a fused problem fails the
support check, because its solution requires a problem without fused
all-to-all.

**Stream capture.** `hipblasLtMatmul` without an algorithm checks its stream
with `hipStreamIsCapturing`; the null and legacy streams never capture. While
the stream captures, in any capture mode, JIT only looks up solutions already
published to the JIT solution library: it starts no generation, comgr build or
publication, and the other sources keep their order. A published solution
whose code object the process has not loaded yet is loaded during the capture,
as pre-tuned code objects are; HIP allows module loads in every capture mode,
and the capture stays valid. When nothing is found the call returns
`HIPBLAS_STATUS_INTERNAL_ERROR`, as outside a capture, and reports one line:

```text
hipblaslt error: JIT generation skipped during stream capture for GEMM M=256 N=128 K=576 ... EPILOGUE_DEFAULT
```

When a pre-tuned solution is found instead, the skipped generation goes only to
the info log.

The heuristic queries take no stream and may generate during a capture: they
follow the mode as they do outside one. Generating inside a capture was verified
to keep the capture valid in global, thread-local and relaxed capture modes
with HIP 7.17. Generation takes seconds, so warm the JIT solution library or run
the query before the capture: run the same queries beforehand, in this process
or an earlier one, or run the query first and pass the returned algorithm to
`hipblasLtMatmul` inside the capture.

#### Scratch files

A query that generates waits for the whole generation, which takes seconds to
minutes per problem. Generation runs in a new directory under the system
temporary directory (`TMPDIR` on Linux). It is removed after success and kept
after a failure, whose report names the log inside it.

### Diagnostics with `HIPBLASLT_JIT_DEBUG`

In a JIT build with `HIPBLASLT_JIT` set to `1` or `2`, `HIPBLASLT_JIT_DEBUG`
prints where JIT time goes and what JIT is doing. Its value is a
comma-separated list of category names, in any case:

| Name | Lines |
| --- | --- |
| `timing` | One line when each heuristic query, `hipblasLtMatmul` call, generation and generated solution finishes, with the duration of each step |
| `progress` | One line per step as it happens: lookups, waits, generation stages, builds and publication |
| `all` | Every category, including categories added later |

Unset or empty prints nothing. A number, including `0` and `1`, or an unknown
name prints `hipblaslt warning: HIPBLASLT_JIT_DEBUG=<value>: ignoring <names>;
the value is timing, progress or all, comma-separated` once and is ignored; the
names beside it still apply. hipBLASLt reads the variable once per process.
With `HIPBLASLT_JIT` off, and in a build without JIT, it prints nothing and no
warning.

The lines go to stderr. `HIPBLASLT_JIT_DEBUG_FILE` names a file to append them
to instead, with `%i` replaced by the process ID; the file is created readable
and writable by its owner only. Each line is written whole, so threads and
processes that share a file never split a line. A file that cannot be opened
prints one warning, and the lines go to stderr. The lines are not copied into
the hipBLASLt log.

Every line is the prefix `hipblaslt jit-debug ` followed by one JSON object,
so `grep '^hipblaslt jit-debug '` separates the lines and one JSON parser reads
them. The object starts with these keys:

| Key | Value |
| --- | --- |
| `v` | Schema version, `1`. New keys keep the version; a changed meaning increments it |
| `cat` | `timing` or `progress` |
| `ev` | The event |
| `pid`, `tid` | The process ID and a thread number counted from 1 in each process |
| `t_ms` | Milliseconds on the monotonic clock since the process's first line |
| `q` | The query the line belongs to, `<pid>.<n>`, or `null` |
| `gen` | Inside a generation, the generation, `<pid>.g<n>` |

Durations are nanoseconds in an `ns` object, in the order the steps ran, and
each includes the steps nested in it. A string longer than 512 bytes is cut and
the line gets `"truncated":true`; most generated kernel names are, and the
solution index identifies the solution. A line longer than 4 KiB keeps only the
keys above, with `"truncated":true` and its full size as `oversize`.

`timing` lines:

| `ev` | When and what |
| --- | --- |
| `process` | First line of the process: the mode, the categories, the destination, the wall-clock time and `AMD_COMGR_CACHE`. It is a `progress` line when only `progress` is on |
| `setup` | The first JIT use in the process: its `status`, and the time to open the JIT solution library (`store`) and create the components. A backend can add its own fields and steps, such as creating itself |
| `library.init` | A device's pre-tuned library initialization |
| `query` | Each heuristic query, `api` `c` or `cpp`: `requested`, `returned`, the problem, `from` (the results each source added: `override`; `best` from the pre-tuned query, split into `equality`, `jit` and `others` when JIT runs between them; `all` from the `getAllSolutions` fill; otherwise `jit` after them; in forced mode only `jit`), `jit` (results `needed` and found as `hits`, `hits_after_wait`, `kept` and `dropped` by the support check, and `waited_on`, the generation another thread ran while this one waited), `gen` when it generated, and the step durations |
| `matmul` | `hipblasLtMatmul` without an algorithm or with a JIT solution: the first call for each problem and algorithm, and every call that generated, loaded a code object (`loaded` is `now`) or skipped generation during a capture, with the selection, library lookup, preparation, code-object load and launch durations |
| `generation` | Each generation: the requested and candidate counts, failures, the problem, the generated, `fresh`, `reused` and published counts, and the durations of prediction, scratch, the backend, building, support checks, publication (lock wait, time holding the lock, refresh) and loading, with `other` the rest. A backend can add its own fields and the durations of its own steps within `backend` |
| `solution` | One per generated solution: rank, kernel, `outcome` (`built`, `build_failed`, `unsupported`, `published`, `publish_failed`, `loaded` or `load_failed`), index, message, assembly and HIP unit counts with each HIP unit's compile time as `hip_units`, and the metadata, assembly, HIP compile, link, build and support durations |
| `query.aggregate`, `matmul.aggregate` | At most once per second, and at exit: the calls not printed in full, per API or per problem and algorithm, as `calls`, `ns.sum` and `ns.max` |

`progress` lines:

| `ev` | When and what |
| --- | --- |
| `query.start`, `query.end` | A heuristic query or `hipblasLtMatmul` without an algorithm starts and ends |
| `lookup` | The JIT solution library lookup: `result` (`hit`, `partial` or `miss`), `found` and `needed` |
| `capture.skip` | Generation skipped because the stream is being captured |
| `generation.wait` | The query waited for another thread's generation of the same problem, named in `waited_on` |
| `generation.repeated` | The problem fell short in an earlier generation in this process, so it is not generated again |
| `generation.start`, `generation.end` | A generation starts, with the requested and candidate counts and the problem, and ends, with `outcome` (`ok`, `partial`, `failed` or `empty`) and the generated, published and loaded counts |
| `build.start`, `build.end` | Each solution's code-object build, with its `outcome` |
| `publish.start`, `publish.done` | Publication into the JIT solution library, with `fresh` and `reused` entries |
| `load.done` | Solutions loaded without publication |
| `failure` | A failed stage and its message |

`query.start`, `query.end`, `lookup` and `query` lines share a budget of 50
lines that refills at 10 per second. Lines over the budget are counted and
reported by a `suppressed` line in their category before the next line that is
written, and a dropped `query` line is added to `query.aggregate`. Repeated
`matmul` calls go to `matmul.aggregate`. Generation lines are never dropped.

A cache hit with `HIPBLASLT_JIT_DEBUG=timing`:

```text
hipblaslt jit-debug {"v":1,"cat":"timing","ev":"query","pid":4242,"tid":1,"t_ms":5258.696,"q":"4242.4","api":"cpp","mode":1,"requested":2,"returned":2,"problem":"GEMM M=256 N=128 K=512 batch=1 opA=OP_N opB=OP_N A=R_16F B=R_16F C=R_16F D=R_16F compute=COMPUTE_32F epilogue=EPILOGUE_DEFAULT","from":{"equality":0,"jit":2,"others":0,"best":2},"jit":{"needed":2,"hits":2,"kept":2,"dropped":0},"ns":{"total":277874,"jit_target":3660,"lookup_attach":14369,"lookup_refresh":6900,"lookup_scan":231155,"jit_lookup":256484,"jit_support":3850,"jit":271284,"others":1370,"get_best":276604}}
```

The start of a generation with `HIPBLASLT_JIT_DEBUG=progress`:

```text
hipblaslt jit-debug {"v":1,"cat":"progress","ev":"lookup","pid":4242,"tid":1,"t_ms":155.350,"q":"4242.1","result":"miss","found":0,"needed":2}
hipblaslt jit-debug {"v":1,"cat":"progress","ev":"generation.start","pid":4242,"tid":1,"t_ms":155.791,"q":"4242.1","gen":"4242.g1","requested":2,"candidates":72,"problem":"GEMM M=256 N=128 K=512 batch=1 opA=OP_N opB=OP_N A=R_16F B=R_16F C=R_16F D=R_16F compute=COMPUTE_32F epilogue=EPILOGUE_DEFAULT"}
```

Unset, empty, with `HIPBLASLT_JIT` off, or in a build without JIT, the
variable adds no lines, files or clock reads. With any
value, the results, return statuses, JIT solution library contents and the
other stderr output are the same as without it, and a line that cannot be
written never fails a call.

### Validation

The [JIT test guide](clients/tests/jit/README.md) describes the test binaries
and their CTest tests, which replay the committed source bundles instead of
generating. `ctest -L jit-cpu` runs the ones that need no GPU, and `-L jit-gpu`
the ones that run on a gfx950 GPU. In a build whose backend is the test
backend, the heuristic routes cover each mode, reuse of the library by later
processes and by `getAlgosFromIndex` with JIT off, distinct kernels when
several solutions are requested, the order of a device library's Equality
results, JIT solutions and other pre-tuned results, a tuning override that
names a JIT solution, `hipblasLtMatmul` without an algorithm and heuristic
queries that generate during stream capture in each capture mode, the same
query from several threads and processes at once, a problem the predictor
cannot rank, failure reports, and the `HIPBLASLT_JIT_DEBUG` lines: timing that
adds up, progress in order, no change when it is off, and debug files. In
any build with `HIPBLASLT_JIT_TESTING=ON`, the multi-backend routes replace
the backends with mock backends and cover their order and selection, opt-in
backends, a backend whose configuration fails, the result kept for each later
backend, a backend that does not serve the problem, kernels two backends share,
a backend that fell short, and stream capture. The
[`AlgoErrors` tests](clients/tests/jit/README.md#algorithm-error-status-tests)
in `hipblaslt-test` check the statuses for an index that names no solution and
for a rejected heuristic query.

Numerical results require execution on the target GPU, and cross-compilation
establishes compilation only. The tests do not cover native Windows execution.
