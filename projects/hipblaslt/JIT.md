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

hipBLASLt runs a GEMM with a kernel from its pre-tuned library, so a problem
that the library serves poorly, or not at all, has no better kernel available.
JIT generation produces kernels for a problem when they are needed. The `Jit`
component defines the stages of that generation and the order in which they
run. Its interfaces, a comgr code-object builder and a TensileLite solution
loader are implemented. Internal entry points let the JIT test binaries
generate a GEMM solution through a backend and run it with `hipblasLtMatmul`
or `hipblaslt_ext::Gemm`, or publish generated solutions into a persistent
JIT solution library on disk, whose solution indices any later process runs
without generating again. For a backend that consumes a prediction, an Origami
predictor first ranks candidate configurations. The only backend is a test mock
that replays pre-generated source bundles. No generator backend is implemented yet, and no
public API reaches JIT.

## Current behavior

### Entry points

The internal entry points, in `library/src/amd_detail/hipblaslt-jit.hpp`,
generate through `Jit`, keep generated algorithms in one process-local
registry, and return an algorithm for the existing C and C++ GEMM execution
APIs:

- A backend factory, such as `jit::mock::createBackend`, returns a `Backend`
  handle that owns a `Jit` configured with that backend.
- `jit::makeGemmRequest` captures existing GEMM descriptors and host scalars.
- `jit::getJitAlgo` compiles on the selected device and returns an owned
  `Solution`.
- `jit::getGemmAlgo` adapts the solution to the algorithm that
  `hipblasLtMatmul` and `Gemm` accept.
- `jit::getLibraryAlgos` instead returns solution indices from the
  [JIT solution library](#persistent-solution-library), generating and
  publishing the solutions it lacks; `hipblaslt_ext::getAlgosFromIndex` turns
  them into algorithms.

The header is internal, as are `hipblaslt-jit-mock.hpp` and
`hipblaslt-jit-gemm-internal.hpp`: they are not installed and
`hipblaslt-ext.hpp` does not include them. `libhipblaslt.so` exports four
functions and one type from them with `HIPBLASLT_EXPORT` for the JIT test
binaries, which link against the shared library: `jit::makeGemmRequest`,
`jit::getJitAlgo`, `jit::getGemmAlgo`, `jit::getLibraryAlgos` and the
`jit::detail::GemmRequest` request type. A build with `HIPBLASLT_JIT_TESTING=ON` also exports
`jit::mock::createBackend`. No installed header declares them, and they are
not a supported API.

The backend's configuration belongs to the options of its factory. The
application owns its buffers and workspace. The request owns descriptor values
and host scalars; it does not take ownership of device pointers. Compilation
and support checks finish before graphics processing unit (GPU) work is
submitted; call the entry points before stream capture. GEMM is the
implemented operation.

Internally, the GEMM request reuses `RocblasltContractionProblem` with owned
scalar values. The generic `Solution` and private `CompiledSolution` retain
the `Jit`, device target, request, workspace and bundle lifetime around the
existing GEMM support and execution machinery. A matmul algorithm is an
adaptation token, not a general owning executable object.

### Components

`Jit`, in `library/src/amd_detail/hipblaslt-jit-component.{hpp,cpp}`, runs one
request for one device target through these stages. Each stage is an
interface, so that each implementation can be replaced and tested on its own.

| Stage | Interface | Contract |
| --- | --- | --- |
| Predict | `Predictor::predict(PredictionRequest, TuningKnowledge, Prediction&)` | Only for a backend that consumes a prediction. Ranks candidates built from the tuning knowledge's seeds, best first, each naming its modeled contract. |
| Generate | `Backend::generate(GenerationRequest, std::vector<GeneratedSolution>&)` | Returns up to `GenerationRequest::count` solutions, best first, and builds and loads nothing. Each `GeneratedSolution` holds a one-solution TensileLite library entry, its main kernel name, and the source units to build. `NotSupported` means the request is outside the backend's domain. |
| Build | `CodeObjectBuilder::build(GeneratedSolution, GenerationRequest, BuiltSolution&)` | Builds a solution's units into code objects. `GeneratedSolution` has no code-object field; only `BuiltSolution` adds the main code object and its helpers. |
| Support | `SolutionLoader::support` | Evaluates the entry's predicates and workspace for the request, and loads no code. |
| Publish | `SolutionStore::publish` | Stores built solutions and returns one library index per solution, in order. `SolutionStore::lookup` returns the indices of stored solutions for exactly a request. |
| Load | `SolutionLoader::load` | Loads the code objects into a process-local executable `KernelBundle`. |

`Jit::generate` is the only code that sequences these stages. When the backend
consumes a prediction, it first runs the predictor and passes on only the
candidates whose contract the backend transports. It then asks the
backend for at most the requested count of solutions in a private scratch
directory, forwarding the workspace limit and the kernels the caller already
has. It then builds each solution and checks its support until the count is
reached. With a store it publishes the supported solutions, and loads them only
when publishing fails; without a store it loads them. A failure in one
solution's build or support skips that solution and keeps the rest.

Each failure is recorded with its stage (configure, predict, generate, build,
support, load or publish) in `Jit::Outcome::failures`, in the order it happened.
The scratch directory is created under the temporary directory, removed when
the call succeeds, and kept after a failure that left files in it. `Jit`
requires a backend, a builder and a loader; the store is optional. One `Jit`
can generate from several threads at once.

**Contracts and versions.** `BackendInfo::contracts` lists the modeled
contracts a backend transports; an empty set means it consumes no prediction.
`Predictor::modeledContracts()` lists those a predictor emits. A backend with
contracts requires a predictor that shares at least one, and tuning knowledge;
`Jit` rejects the components otherwise. The `Jit`'s version is the backend's
version, followed for such a backend by
`|predictor=<id>;contracts=<shared contracts, sorted>|knowledge=<id>@<version>`.
`Jit` creates its store from `Components::store`, a factory it calls once with
the backend's information and that version, so the cache key changes whenever
the predictor, the shared contracts or the knowledge version change.

**Candidates.** Each `Candidate` names its modeled contract (empty means the
prediction's `modeledContract`), the index of the seed it came from, and
provenance JSON that the backend records. `parameters` are forwarded to the
backend in that contract's vocabulary; `modeled` values are recorded, not
forwarded. A consumer ignores candidate fields it does not know, so contracts
can add fields without breaking older consumers. `ExecutionPolicy` names how a kernel covers
the output tiles: `strategy` is none, data-parallel or Stream-K, and
`assignment` is a static grid, a dynamic work queue or hybrid.

`OperationRequest` names only its kind, so `Jit` sees no GEMM. The interfaces
are for compiled-in implementations and do not establish a stable external
plugin application binary interface (ABI).

The implementations are:

- Predictor: `makeOrigamiPredictor()`, in `hipblaslt-jit-origami-predictor.cpp`,
  expands the tuning knowledge's candidate seeds across the target's matrix
  instructions, ranks them with Origami, and emits the `origami.gemm.dp.v1`
  contract; see [Origami modeled inputs](#origami-modeled-inputs).
- Tuning knowledge: `makeCatalogKnowledge()`, in
  `hipblaslt-jit-catalog-knowledge.cpp` (id `catalog.v1`): 11 tile shapes, two
  DepthU rules, cache hints that are only the defaults on gfx90a and gfx1250,
  and policy none for every seed. It supplies no values for unmodeled knobs,
  which keep the backend's defaults.
- Code-object builder: `makeComgrBuilder()`; see
  [building generated sources](#building-generated-sources).
- Loader: `makeTensileLoader()`, in `hipblaslt-jit-loader.cpp`, parses the
  entry, checks support and workspace with TensileLite's predicates, and loads
  the code object into a process-local `TensileBundle`.
- Solution store: the JIT solution library, in `hipblaslt-jit-library.cpp`,
  with `hipblaslt-jit-msgpack.cpp` writing the library files and
  `hipblaslt-jit-fs.cpp` providing the directory checks, file lock and atomic
  replacement. `getLibraryAlgos` sets it as the store of its `Jit`; see
  [persistent solution library](#persistent-solution-library).
- Mock backend: `hipblaslt-jit-mock-backend.cpp` replays a list of source
  bundles without a generator. A generation returns, in list order, up to the
  requested count of bundles whose predicates accept the device and problem,
  skipping excluded kernels; the comgr builder still builds them. Its faults
  fail generation and leave a log in the scratch directory, replace the main
  kernel assembly with an invalid instruction so the build fails, append the
  request to a file and fail, or abort the process. Given modeled contracts it
  consumes the predictions of the Origami predictor and the catalog knowledge.
  Tests reach it through `jit::mock::createBackend` in `hipblaslt-jit-mock.hpp`. Only builds with
  `HIPBLASLT_JIT_TESTING=ON` compile it; it is not a production backend.

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

### Build

`HIPBLASLT_ENABLE_JIT` is disabled by default. A disabled build compiles and
exports no JIT entry points. The enabled build requires the host library, ROCm
and ROCm's `amd_comgr` CMake package, which only a JIT build links. comgr
compiles helper sources against the host's C and C++ standard library headers,
so those must be installed where JIT runs. From the repository root:

```bash
project_root="$PWD"
project_build="$project_root/projects/hipblaslt/build/release"
cmake -S "$project_root/projects/hipblaslt" -B "$project_build" \
  -DHIPBLASLT_ENABLE_JIT=ON -DHIPBLASLT_ENABLE_HOST=ON \
  -DHIPBLASLT_BUILD_TESTING=ON \
  -DHIPBLASLT_ENABLE_DEVICE=OFF -DGPU_TARGETS=gfx950
cmake --build "$project_build" --parallel
```

The `jit` CMake preset enables this feature for a new configuration.
`HIPBLASLT_JIT_TESTING`, off by default, also compiles the mock backend into
the library and adds the tests that use it. The
[JIT test guide](clients/tests/jit/README.md) lists the test targets and the
validation commands.

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
`getJitAlgo` and `getGemmAlgo` return are not a library index:
`hipblaslt_ext::getIndexFromAlgo` returns -1 for them, and a different program
invocation cannot use them. Save the bundle manifests for reproduction.

An empty GEMM output (M=0 or N=0) returns `HIPBLAS_STATUS_NOT_SUPPORTED` from
the request factory without compilation. K=0 can use a solution that
implements beta*C. The backend owns its datatype, instruction and scale-layout
restrictions. The library propagates support failures, including a mismatch
between the supplied physical MX scale layout and the compiled solution. Jit
never benchmarks generated solutions. A failed build names the retained
`comgr.log` in its message.

### Persistent solution library

The JIT solution library keeps generated solutions on disk so that later
processes run them without generating again. `jit::getLibraryAlgos` is its
entry point: for one request it returns up to the requested number of solution
indices, first the published solutions that match the request, in the order
they were first published, then solutions that `Jit` generates with the
supplied backend and publishes. Any process then passes an index to
`hipblaslt_ext::getAlgosFromIndex`, `hipblasLtMatmul` and `Gemm` as it would a
prebuilt index.

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
| `backend` | Backend identifier and version. Each backend defines its version to change whenever its output can; the mock backend's is a hash of its replayed bundles. For a backend that consumes predictions, the version is followed by `\|predictor=<id>;contracts=<contracts>\|knowledge=<id>@<version>`: the predictor, the modeled contracts that both it and the backend support, sorted, and the tuning knowledge with its version. |
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
the next index, so indices stay unique across every key directory under a root
and are never reused while the root exists. Publication fails once the range is
exhausted. `tensile_host.cpp` routes reserved indices to the JIT solution
library, with its own solution adapter whose code-object directory is the key
directory, and all other indices to the prebuilt library. An index that no JIT
library holds resolves to an empty library, so callers report their usual
missing-solution error.

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

### Building generated sources

Generators emit assembly or HIP source plus metadata only; they do not assemble,
link or bundle. `makeComgrBuilder()` returns the `CodeObjectBuilder` that builds
code objects in process through AMD comgr (`hipblaslt-jit-builder.cpp` and
`hipblaslt-jit-code-object.cpp`). It uses three comgr actions:

- Assembly: `AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE`. A unit's
  `.amdgcn_target` directive must name the device's processor and only
  features the device has; it is then rewritten to the device's full target
  ID. Assembly units that declare different wavefront sizes are rejected.
- HIP helper source: `AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE` with
  `--rocm-path` and a content-derived `-cuid`, so helper objects link together.
  The ROCm path is `HIP_PATH` when set, otherwise the prefix of the loaded HIP
  runtime.
- Link: `AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE` joins the main kernel
  and helper relocatables into one code object per solution, with
  `-Xlinker --build-id=sha1`.

The generator and the builder use the same code-object version, which
`GenerationRequest::codeObjectVersion` carries (4 by default). The output is a
raw, uncompressed executable code object. comgr cannot bundle or compress it,
and `hipModuleLoadData` accepts raw executable and linkable format (ELF)
objects. A generator needs no offload bundler.

After linking, the builder reads the code object's metadata and checks that its
instruction set architecture (ISA) is the device's and that it defines the
solution's main kernel. A build failure has the build stage. Its message
carries the first error line of the comgr log, and the builder appends the full
log to `comgr.log` in the request's scratch directory and names that file in
the message.

comgr's own on-disk cache (`~/.cache/comgr`) keeps its default: hipBLASLt never
sets `AMD_COMGR_CACHE`. That cache holds the results of comgr actions for every
comgr user in the process, such as hipRTC, and comgr reads its setting once per
process.

### Source bundle format

A source bundle is a generated solution stored as a directory.
`source_bundle::readSourceBundle`, in `hipblaslt-jit-source-bundle.hpp`, reads
one by directory convention:

| Path | Contents |
| --- | --- |
| `library/TensileLibrary.dat.zlib`, `.dat` or `.yaml` | The one-solution library entry; exactly one of them, decoded when compressed |
| `sources/*.s` | The main kernel assembly; at least one |
| `sources/Kernels.cpp` | The helper kernels, when the solution needs helpers |
| Other files in `sources/` | Headers that the helper source includes |
| `manifest.json` | A JavaScript Object Notation (JSON) provenance record that hipBLASLt does not read |

Artifact paths must be relative and stay inside the bundle, including through
symbolic links, and `sources/` may hold only regular files. The reader bounds
the file count (1024), each file (64 MiB), the decoded library (64 MiB) and the
sources in total (256 MiB).

The JIT tests use gfx950 source bundles committed in `clients/tests/jit/data`.
Their manifests record the kernel-argument and persistent-loop argument layout
versions of the generator that wrote them, and the `jit-bundle-freshness` test
fails when those differ from the ones in
`tensilelite/Tensile/Common/GlobalParameters.py`, when the code-object version
differs from the builder's, or when a bundle no longer reads or builds.
[Their README](clients/tests/jit/data/README.md) gives the commands that
regenerate them.
