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
without generating again. The only backend is a test backend that replays
pre-generated source directories. No generator backend is implemented yet.
`hipblasLtMatmulAlgoGetHeuristic` and `Gemm::algoGetHeuristic` consult JIT
when the library is built with JIT support and `HIPBLASLT_JIT` is `1` or `2`.
Those queries, and `jit::getJitAlgo`, publish into the JIT solution library
and reuse a hit, so a later process asking the same problem gets the same
index and does not generate. The algorithms are library indices from 2^30.
`getIndexFromAlgo` returns that index.

## Current behavior

### Entry points

The internal entry points, in `library/src/amd_detail/hipblaslt-jit.hpp`,
generate through `Jit`, publish into the JIT solution library, and return an
algorithm for the existing C and C++ GEMM execution APIs:

- A backend factory, such as `jit::replay::createBackend`, returns a `Backend`
  handle that owns a `Jit` configured with that backend.
- `jit::makeGemmRequest` captures existing GEMM descriptors and host scalars.
- `jit::getJitAlgo` looks the solution up in the JIT solution library, publishes
  it when it is absent, and returns an owned `Solution` for the first index.
  A hit does not generate.
- `jit::getGemmAlgo` adapts the solution to the library index that
  `hipblasLtMatmul` and `Gemm` accept.
- `jit::getLibraryAlgos` returns solution indices from the same lookup and
  publish, described under
  [persistent solution library](#persistent-solution-library).
  `hipblaslt_ext::getAlgosFromIndex` turns them into algorithms.

The header is internal, as are `hipblaslt-jit-replay.hpp` and
`hipblaslt-jit-gemm-internal.hpp`: they are not installed and
`hipblaslt-ext.hpp` does not include them. `libhipblaslt.so` exports five
functions and one type from them with `HIPBLASLT_EXPORT` for the JIT test
binaries, which link against the shared library: `jit::makeGemmRequest`,
`jit::getJitAlgo`, `jit::getGemmAlgo`, `jit::getLibraryAlgos` and the
`jit::detail::GemmRequest` request type, and `jit::replay::createBackend`. No
installed header declares them, and they are not a supported API.

The backend's configuration belongs to the options of its factory. The
application owns its buffers and workspace. The request owns descriptor values
and host scalars; it does not take ownership of device pointers. Compilation
and support checks finish before graphics processing unit (GPU) work is
submitted. A capturing stream may return a solution already in the JIT
solution library and does not start a build. GEMM is the implemented
operation.

Internally, the GEMM request reuses `RocblasltContractionProblem` with owned
scalar values. A matmul algorithm is a JIT solution library index from 2^30.
`hipblasLtMatmul` runs that index the way it runs a prebuilt solution.

### Heuristic queries

`HIPBLASLT_JIT` chooses whether the C and C++ heuristic queries consult the
JIT library. The value is read once per process. A build without JIT support
does not read it. Anything other than unset, empty, `0`, `1` or `2` warns
once and leaves the mode off.

| Value | Lookup |
| --- | --- |
| unset, empty or `0` | Heuristic queries do not consult JIT. `getBestSolutions` runs as it does without JIT. |
| `1` | The override file, then the Equality provider rows, then the JIT library, then the other provider rows, then the `getAllSolutions` fill. Each source fills only the remaining request, and a kernel already returned is skipped. |
| `2` | The JIT library only. The override file is not read. A problem the library does not solve returns success with no algorithms. |

A JIT build with no HipKittens backend replays the committed kernels under
`clients/tests/jit/data` when that tree is compiled in. Those queries publish
into the JIT solution library, and a later query in the process reuses the
index. A build with no such kernels warns once: mode `1` leaves the query
unchanged, and mode `2` returns no algorithms. The replay backend implements
generation only.

The algorithms those queries return are JIT solution library indices, from
2^30, and they run through the same path as a prebuilt index.
`getIndexFromAlgo` returns the index. A second process that queries the same
problem receives that index and does not generate. The library is the only
cache: nothing is kept in a process-local map. A capturing stream may return a
hit and does not start a build.

Grouped GEMM is not filled from the JIT library. Mode `1` leaves that query on
its existing path. Mode `2` returns success with no algorithms.

### Components

`Jit`, in `library/src/amd_detail/hipblaslt-jit-component.{hpp,cpp}`, runs one
request for one device target through these stages. Each stage is an
interface, so that each implementation can be replaced and tested on its own.

| Stage | Interface | Contract |
| --- | --- | --- |
| Generate | `Backend::generate(GenerationRequest, std::vector<SourceCodeBundle>&)` | Returns up to `GenerationRequest::count` source bundles, best first, and builds and loads nothing. A `SourceCodeBundle` is the main kernel, HIP helpers and headers that become one solution. It is not a TensileLite library entry. `NotSupported` means the request is outside the backend's domain. |
| Build | `CodeObjectBuilder::build(SourceCodeBundle, BuildRequest, CompiledCode&)` | Compiles the main kernel with comgr into one code-object `Artifact`, and each HIP helper into its own code object. |
| Support | `SolutionLoader::support` | Evaluates the predicates and workspace of the record's solutions for the request, returns the local indices of those that support it, best first, and loads no code. |
| Publish | `SolutionStore::publish` | Stores the supported code objects and returns one library index per solution, in order. `JitLibrary` writes the TensileLite solution record. `SolutionStore::lookup` returns the indices of stored solutions for exactly a request. |
| Load | `SolutionLoader::load` | Loads the code objects once and returns a process-local executable `KernelBundle` for each supported solution. |

`Jit::generate` is the only code that sequences these stages. It asks the
backend for at most the requested count of solutions in a private scratch
directory, forwarding the workspace limit and the kernels the caller already
has. It then builds each source bundle and checks the support of its solutions
until the count of supported solutions is reached. With a store it publishes the
supported solutions, and loads them only when publishing fails; without a
store it loads them. A failure in one bundle's build, support or load skips
that bundle and keeps the rest.

Each failure is recorded with its stage (configure, generate, build, support,
load or publish) in `Jit::Outcome::failures`, in the order it happened.
The scratch directory is created under the temporary directory, removed when
the call succeeds, and kept after a failure that left files in it. `Jit`
requires a backend, a builder and a loader; the store is optional. One `Jit`
can generate from several threads at once.

`Jit` takes a GEMM request. The interfaces are for compiled-in implementations
and do not establish a stable external plugin application binary interface (ABI).

The implementations are:

- Code-object builder: `makeComgrBuilder()`; see
  [building generated sources](#building-generated-sources).
- Loader: `makeTensileLoader()` checks support with a local TensileLite load of
  the record (see
  [loading a published solution](#loading-a-published-solution)) and returns a
  `TensileGemmBundle` that refers to the `ContractionSolution`. The bundle does
  not own a library, hardware description, or adapter. Published solutions are
  launched through `JitLibrary`. The loader is in `rocblaslt/src/tensile_host.cpp`,
  next to the GEMM problem translation that the support check reuses.
- Solution store: the JIT solution library, in `hipblaslt-jit-library.cpp`,
  with `hipblaslt-jit-msgpack.cpp` writing the library files and
  `hipblaslt-jit-fs.cpp` providing the directory checks, file lock and atomic
  replacement. `getJitAlgo`, `getLibraryAlgos` and heuristic queries share one
  lookup-then-publish helper that sets this store; see
  [persistent solution library](#persistent-solution-library). A backend
  implements `generate` only.
- Replay backend: `hipblaslt-jit-replay-backend.cpp` replays a list of source
  bundles without a generator. A generation returns, in list order, the
  bundles with a solution whose predicates accept the device and problem, until
  they hold the requested count of such solutions. The predicate check is a
  local TensileLite load inside that function. It skips a bundle when every
  such solution's kernel is excluded; the comgr builder still builds them.
  It implements `Backend`, so `Jit` builds and loads its sources with the same
  builder and loader as any other backend. Tests reach it through
  `jit::replay::createBackend` in `hipblaslt-jit-replay.hpp`, and every JIT
  build compiles it.

### Build

`HIPBLASLT_ENABLE_JIT` is disabled by default. A disabled build compiles and
exports no JIT entry points. The enabled build requires the host library, ROCm
and ROCm's `amd_comgr` CMake package, which only a JIT build links. comgr
compiles HIP sources against the host's C and C++ standard library headers,
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

`GPU_TARGETS` is `gfx90a`, `gfx942` or `gfx950`. The command above uses
`gfx950`. The [JIT test guide](clients/tests/jit/README.md) lists the test
targets and the validation commands.

### Algorithm lifetime and failures

The returned heuristic result contains the required workspace size. Supply that
workspace and follow the same handle, stream and workspace sharing rules as
`hipblasLtMatmul` and `Gemm` calls using prebuilt algorithms. The algorithm
carries its solution's local index, resolves to its bundle's library and
adapter, and then runs through the same launch path as a prebuilt algorithm,
including its synchronization storage. Registry synchronization
protects algorithm lookup; it does not protect application buffers or make
simultaneous calls on one `Gemm` object safe.

`getJitAlgo` and `getGemmAlgo` return a JIT solution library index from 2^30.
`hipblaslt_ext::getIndexFromAlgo` returns that index, and another process that
can read the library runs it without generating. Save the bundle manifests
for reproduction.

An empty GEMM output (M=0 or N=0) returns `HIPBLAS_STATUS_NOT_SUPPORTED` from
the request factory without compilation. K=0 can use a solution that
implements beta*C. The backend owns its datatype, instruction and scale-layout
restrictions. The library propagates support failures, including a mismatch
between the supplied physical MX scale layout and the compiled solution. Jit
never benchmarks generated solutions. A failed build names the retained
`comgr.log` in its message.

### Persistent solution library

The JIT solution library keeps generated solutions on disk so that later
processes run them without generating again. It is the only cache. Every
backend publishes what it builds. `jit::getLibraryAlgos`, `jit::getJitAlgo`
and the heuristic queries share one lookup-then-publish helper: for one
request it returns up to the requested number of solution indices, first the
published solutions that match the request, in the order they were first
published, then solutions that `Jit` generates with the supplied backend and
publishes. A hit skips generation. Any process then passes an index to
`hipblaslt_ext::getAlgosFromIndex`, `hipblasLtMatmul` and `Gemm` as it would a
prebuilt index. Heuristic results are those indices. A second process gets
the same index and does not generate.

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

**Layout.** Under a `v1` schema directory, each device and backend has its own
directory. The directory name is a filesystem-safe literal of the ISA, target
ID, library architecture, wavefront size, backend id and backend version. It
does not include the comgr path or the ROCm path. A query opens one slice of
that directory. The slice key is TensileLite's lazy placeholder grain: the
operation identifier, the datatypes, the scale suffix (`_SAB` or `_SABV`, and
MX block and type) and the epilogue flags that change the kernel family. The
slice is a standard lazy TensileLite library that the stock loader reads: a
master file, the index mapping, and one entry (`.dat`) and code object (`.co`)
per solution. Other slices stay unread. Code objects load for the chosen index.
A HIP helper is compiled once per backend id and helper name and stored beside
the slices; an entry records the helper name. The files the library reads and
writes are MessagePack. For example:

```text
$HIPBLASLT_JIT_LIBRARY_PATH/
└── v1/
    ├── allocator.dat                  next free solution index
    ├── lock                           publication lock for the whole root
    └── gfx950_..._replay_1/           one directory per device and backend
        ├── cache-key.msgpack
        ├── indices.msgpack            solution index to slice key
        ├── helpers/<backend>/<name>.co
        ├── slices/<slice key>/
        │   ├── TensileLibrary_lazy_gfx950.dat
        │   ├── TensileLiteLibrary_lazy_gfx950_Mapping.dat
        │   ├── k/<kernel>/t/<problem>/s/<sizes>.dat
        │   └── k/<kernel>/t/<problem>/s/<sizes>.co
        └── staging/                   temporary files before rename
```

An entry name is a filesystem-safe literal of the kernel, the problem type and
the sizes. Publishers of that same identity choose the same files and reuse the
existing index. Each entry holds one supported solution of a generated entry,
with its solution index rewritten to the allocated one and the generated
entry's code object beside it. Each master row is
`And(SizeEqual(M), SizeEqual(N), SizeEqual(batch), SizeEqual(K), <the entry's
problem predicate>)` pointing to a placeholder for the entry, so a lookup
matches only the exact sizes within the full ProblemType. The solution's own
hardware, problem and task predicates, including its workspace requirement,
still run on every match. On load, a solution whose code-object version or
kernarg version this build cannot run is skipped. The rest of the library
still loads. A different schema integer, comgr path or ROCm path does not
reject the directory.

**Cache key.** `cache-key.msgpack` records:

| Field | Contents |
| --- | --- |
| `target_id`, `isa`, `library_arch`, `wavefront_size` | Full target ID with features (for example `gfx950:sramecc+:xnack-`), ISA, TensileLite library architecture and wavefront size |
| `backend_id`, `backend_version` | Backend identifier and integer version. The replay backend increments its version constant when a replayed solution can change. |
| `comgr` | comgr version, and on Linux the path, size and modification time of the loaded comgr library |
| `code_object_version` | The code-object version that the generator and the builder use (4) |
| `rocm_path` | The ROCm path that the builder passes to comgr |
| `compiler_environment` | `HIP_PATH`, `LLVM_PATH`, and every set `AMD_COMGR_*` variable, including `AMD_COMGR_CACHE`, `AMD_COMGR_CACHE_DIR`, `AMD_COMGR_CACHE_POLICY` and `AMD_COMGR_USE_VFS`. Logging, `SAVE_TEMPS`, `SAVE_LLVM_TEMPS` and `TIME_STATISTICS` are omitted because they do not change code-object bytes |
| `schema` | The library schema version (1). A different integer does not reject the directory. |

A process uses the directory named for its device and backend. A present device
or backend field that disagrees with that directory fails the lookup or
publication. The comgr identity, the ROCm path, the schema integer and the
code-object version are recorded and do not. To resolve an index it has not
looked up, a process searches the directories for its device, so a later
process can run the solution without configuring the backend that generated it.

**Indices.** JIT solutions use solution indices from 2^30 to `INT32_MAX`.
Prebuilt libraries stay below that range; one that reaches it disables the JIT
solution library, as above. `allocator.dat` holds
the next index, so indices stay unique across every key directory under a root
and are never reused while the root exists. Publication fails once the range is
exhausted. `tensile_host.cpp` routes reserved indices to the JIT solution
library, with its own solution adapter whose code-object directory is the
slice, and all other indices to the prebuilt library. An index that no JIT
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

- Assembly: `AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE`. The main kernel's
  `.amdgcn_target` directive must name the device's processor and only
  features the device has; it is then rewritten to the device's full target
  ID. The declared wavefront size is the one the assembler uses.
- HIP source: `AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE` with
  `--rocm-path` and a `-cuid` taken from the source name. A HIP helper is its
  own code object. `JitLibrary` is the helper cache, keyed by backend id and
  helper name. A hit reuses the code object stored in the library. A miss
  compiles it once, stores it in the library, and then compiles the main
  kernel. A build with no library compiles each helper and does not keep a
  second cache. comgr's `AMD_COMGR_CACHE` stays comgr's; it cannot be stored
  in the library. The ROCm path is `HIP_PATH` when set, otherwise the prefix
  of the loaded HIP runtime.
- Link: `AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE` links the main kernel
  into one code object, with `-Xlinker --build-id=sha1`. The helper code
  objects and the main kernel are ready to add to `JitLibrary`.

The generator and the builder use the same code-object version, which
`GenerationRequest::codeObjectVersion` and `BuildRequest::codeObjectVersion`
carry (4 by default). The output is a
raw, uncompressed executable code object. comgr cannot bundle or compress it,
and `hipModuleLoadData` accepts raw executable and linkable format (ELF)
objects. A generator needs no offload bundler.

After linking, the builder reads the code object's metadata and checks that its
instruction set architecture (ISA) is the device's and that it defines every
kernel named by the build request. Its message
carries the first error line of the comgr log, and the builder appends the full
log to `comgr.log` in `BuildRequest::scratch` and names that file in the
message.

### Loading a published solution

`JitLibrary` is a TensileLite `SolutionLibrary`. After publish, hipBLASLt
launches that solution the same way it launches Equality and Origami solutions.
The stored record is a `ContractionSolution` with the code object, kernel name,
`ContractionSolution::ProblemType`, and the predicates those libraries already
expect. `TensileGemmBundle` does not own a `MasterSolutionLibrary`, `Hardware`,
or `SolutionAdapter`. An in-process handle for one index refers to the solution
inside `JitLibrary`.

`loadGemmLibrary` reads a solution record with TensileLite and requires local
solutions 0 to N-1. It does not require the record to hold exactly one
solution. A predicate check calls it as a local load inside the replay or
HipKittens backend function. That load is not a field of a bundle type and not
a second library type. `readTensileSourceBundle` reads a source directory into
a `SourceCodeBundle` and the kernel name. A backend contributes one
`SourceCodeBundle`. Publishing adds that solution to the library, which holds
many.

The record's solutions are listed best first, each with its own `Problem` row.
Their hardware and problem predicates decide which problems each one serves,
so `findBestSolution` on the loaded library selects the first solution that
serves a problem, and nothing for a problem none of them was generated for.

### Source bundle format

A source bundle is one kernel's sources stored as a directory.
`source_bundle::readSourceBundle`, in `hipblaslt-jit-source-bundle.hpp`, reads
the sources into a `SourceCodeBundle` and requires one main kernel.
`readTensileSourceBundle` also reads the library entry for the kernel name.

| Path | Contents |
| --- | --- |
| `library/TensileLibrary.dat` | The TensileLite solution record, uncompressed MsgPack. Not part of `SourceCodeBundle` |
| `sources/*.s` | The assembly main kernel, at most one |
| `sources/*.hip`, `sources/*.cpp` | The HIP main kernel when there is no assembly, otherwise HIP helpers |
| Other files in `sources/` | Headers |
| `manifest.json` | A JavaScript Object Notation (JSON) provenance record that hipBLASLt does not read |

Artifact paths must be relative and stay inside the bundle, including through
symbolic links, and `sources/` may hold only regular files. The reader bounds
the file count (1024), each file (64 MiB) and the sources in total (256 MiB).

The JIT tests use one kernel, generated for gfx90a, gfx942 and gfx950, in
`clients/tests/jit/data`, of which only the assembly is committed. The assembly
records the kernel-argument and persistent-loop argument layout versions of the
generator that wrote it. `jit-publish` builds that assembly for the current
device and publishes it into the JIT solution library: `plain` as one solution
and `plain-pair` as two. `jit-loader` reads that library and resolves the
pre-generated kernel. Replay tests stage a temporary source directory from the
same descriptions, which is the input `readTensileSourceBundle` already
accepts, and do not publish a second format.
[The data README](clients/tests/jit/data/README.md) gives the command that
generated the assembly.
