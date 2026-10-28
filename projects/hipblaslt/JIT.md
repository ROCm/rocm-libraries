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
run. Its interfaces, a comgr code-object builder, a TensileLite source bundle
reader and the JIT solution library are implemented. Internal entry points let
the JIT test binaries generate a GEMM solution through a backend and run it
with `hipblasLtMatmul` or `hipblaslt_ext::Gemm`, or publish generated solutions
into a persistent JIT solution library on disk, whose solution indices any
later process runs without generating again. The only backend is a test
backend that replays pre-generated TensileLite kernels. No generator backend
is implemented yet. `hipblasLtMatmulAlgoGetHeuristic` and
`Gemm::algoGetHeuristic` consult JIT when the library is built with JIT
support and `HIPBLASLT_JIT` is `1` or `2`. Those queries, and
`jit::getJitAlgo`, publish into the JIT solution library and reuse a hit, so a
later process asking the same problem gets the same index and does not
generate. The algorithms are library indices in [2^30, 0x60000000).
`getIndexFromAlgo` returns that index.

## Current behavior

### Entry points

The internal entry points, in `library/src/amd_detail/hipblaslt-jit.hpp`,
generate through `Jit`, publish into the JIT solution library, and return an
algorithm for the existing C and C++ GEMM execution APIs:

- A backend factory, such as `jit::replay::createBackend`, returns the `Jit`
  configured with that backend.
- `jit::makeGemmRequest` captures existing GEMM descriptors and host scalars.
- `jit::getJitAlgo` looks the solution up in the JIT solution library, publishes
  it when it is absent, and sets a `hipblasLtMatmulHeuristicResult_t` for the
  first index: its `algo` carries that index, which `hipblasLtMatmul` and
  `Gemm` accept, and its `workspaceSize` is the published solution's
  workspace. One requested count of one problem type and size is generated
  at most once per process; a later query for that same count returns what
  that generation published. A different K, datatype, epilogue or count is
  generated on its own.
- `jit::getLibraryAlgos` returns solution indices from the same lookup and
  publish, described under
  [persistent solution library](#persistent-solution-library).
  `hipblaslt_ext::getAlgosFromIndex` turns them into algorithms.

The header is internal, as are `hipblaslt-jit-replay.hpp` and
`hipblaslt-jit-gemm-internal.hpp`: they are not installed and
`hipblaslt-ext.hpp` does not include them. `libhipblaslt.so` exports four
functions and one type from them with `HIPBLASLT_EXPORT` for the JIT test
binaries, which link against the shared library: `jit::makeGemmRequest`,
`jit::getJitAlgo`, `jit::getLibraryAlgos` and the `jit::detail::GemmRequest`
request type, and `jit::replay::createBackend`. No
installed header declares them, and they are not a supported API.

The backend's configuration belongs to the options of its factory. The
application owns its buffers and workspace. The request owns descriptor values
and host scalars; it does not take ownership of device pointers. Compilation
and support checks finish before graphics processing unit (GPU) work is
submitted. Heuristic queries do not carry a stream, so modes 1 and 2 may
build during stream capture. GEMM is the implemented
operation.

Internally, the GEMM request reuses `RocblasltContractionProblem` with owned
scalar values. A matmul algorithm is a JIT solution library index in
[2^30, 0x60000000). `hipblasLtMatmul` runs that index the way it runs a
prebuilt solution.

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

The algorithms those queries return are JIT solution library indices, in
[2^30, 0x60000000), and they run through the same path as a prebuilt index.
`getIndexFromAlgo` returns the index. A second process that queries the same
problem receives that index and does not generate. Each code object is loaded
once per device until that slice's library file changes. Heuristic queries
have no stream, so they may build during stream capture.

Grouped GEMM is not filled from the JIT library. Mode `1` leaves that query on
its existing path. Mode `2` returns success with no algorithms.

### Components

`Jit`, in `library/src/amd_detail/hipblaslt-jit-component.{hpp,cpp}`, runs one
request for one device target through these stages. Each stage is an
interface, so that each implementation can be replaced and tested on its own.

| Stage | Interface | Contract |
| --- | --- | --- |
| Generate | `Backend::generate(GenerationRequest, std::vector<SourceCodeBundle>&)` | Returns up to `GenerationRequest::count` source bundles, best first, each carrying the `ContractionSolution`s its executable runs, and links and publishes nothing. `NotSupported` means the request is outside the backend's domain. |
| Support | `supportingSolutions` | Keeps the bundle's solutions whose predicates and workspace accept the request, best first, and loads no code. |
| Build | `CodeObjectBuilder::build(SourceCodeBundle, BuildKind, Artifact&)` | Builds the bundle into one `Artifact`. Helper relocatables come from the helper library when the caller has one. |
| Publish | `SolutionStore::publish` | Stores one executable with its supported `ContractionSolution`s and returns one library index per solution, in order. `JitLibrary` writes them with TensileLite's writer. `SolutionStore::lookup` returns the indices of stored solutions for exactly a request. |

An `Artifact` is one file of one kind: `Assembly` or `Hip` source, a
`Relocatable` (compiled and not linked) or an `Executable` (a linked code
object). A `SourceCodeBundle` is the main kernel, helpers and headers that
become one executable, with the build inputs: the device's `Target`, the
code-object version, the `ContractionSolution`s the executable runs, best
first, and any other kernel names, such as helpers', the executable must
define. It is not a TensileLite library entry.
`CodeObjectBuilder::build(bundle, kind, out)` builds that bundle with comgr
into exactly one `Artifact`: an `Executable` for `BuildKind::Executable`, or
the main kernel alone as a `Relocatable` for `BuildKind::Relocatable`, which
is how a helper is built.

A `TensileliteAssembly` is a TensileLite kernel's `Assembly` with its `Target`
and the `ContractionSolution`s it implements. `TensileliteAssembly::make`
assembles the kernel alone into a `Relocatable`, reads the solution state
TensileLite keeps in `custom.config` of the kernel's `.amdgpu_metadata` from
that object's metadata note (`code_object::metadataNote`), and builds the
solution with `TensileLite::solutionFromState`. The kernel name and the
kernel-argument versions are the solution's; the target is a field because a
solution states its target only as a hardware predicate. An `Executable` build
accepts that `Relocatable` as its main kernel and links it without assembling
again. `readTensileSourceBundle` reads a source bundle directory and does this
for its assembly, so the bundle's main kernel is the `Relocatable` and its
`solutions` are the kernel's. A HIP main kernel stays a plain `Artifact`, and
its backend builds the bundle's `ContractionSolution`s itself.

A `Target` holds the TensileLite `AMDGPU::Processor`, an `Any`, `On` or `Off`
setting for each of `sramecc` and `xnack`, and the device's ASIC revision.
`Target::fromTargetId` parses a target ID, such as
`hipDeviceProp_t::gcnArchName` or the ID in an assembly `.amdgcn_target`
directive, and rejects an unknown processor, including `gfx000`, and any other
feature. Strings come only from `processorName()`, `targetId()`, `isaName()`
(the comgr ISA name) and `libraryArch()` (the GEMM library subtree, `gfx1250v0`
for a gfx1250 A0 part), so library directory names and `cache-key.msgpack`
fields are the strings earlier builds wrote. `DeviceTarget::make` parses the
device's `gcnArchName` and takes its `asicRevision`; a device whose target ID
is unknown is not supported.

`Jit::generate` is the only code that sequences these stages. It asks the
backend for at most the requested count of solutions in a private scratch
directory, forwarding the workspace limit and the kernels the caller already
has. Then, for each source bundle until the count of published solutions is
reached, it keeps the solutions `supportingSolutions` accepts, builds the
bundle and publishes the executable with those solutions through its
`SolutionStore`. A HIP helper whose kernel the bundle names in `kernelNames`
comes from the helper library under that name, built as a `Relocatable` and
stored there when the library does not hold it; the bundle is then built as
one `Executable` linked with the stored relocatables. A failed build appends
the comgr log to `comgr.log` in the scratch directory. The JIT solution
library is the only place a JIT solution lives: nothing is loaded in process,
and a bundle whose support check, build or publish fails contributes no
solution and the next bundle is tried.

Each failure is recorded with its stage (configure, generate, support, build
or publish) in `Jit::Outcome::failures`, in the order it happened.
The scratch directory is created under the temporary directory, removed when
the call succeeds, and kept after a failure that left files in it. `Jit`
requires a backend, a builder and a store. One `Jit` can generate from
several threads at once.

`Jit` takes a GEMM request. The interfaces are for compiled-in implementations
and do not establish a stable external plugin application binary interface (ABI).

The implementations are:

- Code-object builder: `makeComgrBuilder()`; see
  [building generated sources](#building-generated-sources).
- Support check: `supportingSolutions`; see
  [loading a published solution](#loading-a-published-solution). The GEMM
  problem it checks is the request's, translated by
  `RocblasltToTensileContractionProblemGemm` in `rocblaslt/src/tensile_host.cpp`.
- Solution store: the JIT solution library, in `hipblaslt-jit-library.cpp`,
  with `hipblaslt-jit-msgpack.cpp` writing its own documents and
  `hipblaslt-jit-fs.cpp` providing the directory checks, file lock and atomic
  replacement. `getJitAlgo`, `getLibraryAlgos` and heuristic queries share one
  lookup-then-publish helper that sets this store; see
  [persistent solution library](#persistent-solution-library). A backend
  implements `generate` only.
- Replay backend: `hipblaslt-jit-replay-backend.cpp` replays a list of
  TensileLite source bundles without a generator. It reads each bundle whose
  assembly targets the device with `readTensileSourceBundle` and replays the
  kernel's solution as plain-pair: two copies with typed edits, the first only
  for K a multiple of 512 with WorkGroupMapping 8, the second for any K with
  WorkGroupMapping 1. A generation returns, in list order, the bundles with a
  solution whose predicates accept the device and problem, until they hold the
  requested count of such solutions. It skips a bundle when every such
  solution's kernel is excluded. It implements `Backend`, so `Jit` checks,
  builds and publishes its bundles as it does any other backend's. Tests reach
  it through `jit::replay::createBackend` in `hipblaslt-jit-replay.hpp`, and
  every JIT build compiles it. Each bundle it returns names the request's
  target and code-object version.

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
`hipblasLtMatmul` and `Gemm` calls using prebuilt algorithms. A heuristic
result is a JIT solution library index, in [2^30, 0x60000000), and it runs
through the same launch path as a prebuilt index. Another process that queries
the same problem receives that index.

`getJitAlgo` returns a JIT solution library index in [2^30, 0x60000000).
`hipblaslt_ext::getIndexFromAlgo` returns that index, and another process that
can read the library runs it without generating. Save the bundle manifests
for reproduction.

An empty GEMM output (M=0 or N=0) returns `HIPBLAS_STATUS_NOT_SUPPORTED` from
the request factory without compilation. K=0 can use a solution that
implements beta*C. The backend owns its datatype, instruction and scale-layout
restrictions, which its solutions' predicates state. A request no generated
solution supports returns `HIPBLAS_STATUS_NOT_SUPPORTED`. Jit never benchmarks
generated solutions. A failed build names the retained
`comgr.log` in its message.

### Persistent solution library

The JIT solution library keeps generated solutions on disk so that later
processes run them without generating again. It is the only cache. Every
backend publishes what it builds. `jit::getLibraryAlgos`, `jit::getJitAlgo`
and the heuristic queries share one lookup-then-publish helper: for one
request it returns up to the requested number of solution indices, first the
published solutions that match the request, in the order they were first
published, then solutions that `Jit` generates with the supplied backend and
publishes. One requested count of one problem type and size is generated
at most once in the process; a later query for that same count returns the
published indices. A different K, datatype, epilogue or count is generated
on its own. Any process then passes an index to
`hipblaslt_ext::getAlgosFromIndex`, `hipblasLtMatmul` and `Gemm` as it would a
prebuilt index. Heuristic results are those indices. A second process gets
the same index and does not generate.

**Location and permissions.** `JitLibrary` takes its root from the caller. The
heuristic queries and `jit::getJitAlgo` use one process library whose root is
`HIPBLASLT_JIT_LIBRARY_PATH` or, when that is unset or empty,
`/tmp/hipblaslt-jit-<uid>/` on Linux and `%TEMP%\hipblaslt-jit-<user>` on
Windows. hipBLASLt creates missing directories
with mode 0700. On Linux it refuses the root, and each directory it uses below
the root, when the path is a symbolic link or not a directory, is owned by
another user, or is writable by group or others; a readable directory such as
0755 is accepted. Windows checks only for reparse points and non-directories.
There is no setting that accepts a shared, writable directory. A refused
directory, a privileged (setuid or setgid) process, a build that reads YAML
libraries and a build without a git revision disable the library for the
process, and each use then fails with
`JIT solution library disabled: <reason>`. A prebuilt library with an index in
the reserved JIT range [2^30, 0x60000000) also disables it.

**Layout.** Under a `v1` schema directory, each hipBLASLt build has its own
directory, named by the git revision `hipblasLtGetGitRevision` reports. A build
reads and writes only its own directory, so it never loads kernels that another
build published, even when that build packs kernel arguments differently.
Within it, each device and backend has its own directory. The directory name is a filesystem-safe literal of the ISA, target
ID, library architecture, wavefront size, backend id and backend version. It
does not include the comgr path or the ROCm path. A query opens one slice of
that directory. The slice key is TensileLite's lazy placeholder grain: the
operation identifier, the datatypes, the scale suffix (`_SAB` or `_SABV`, and
MX block and type) and the epilogue flags that change the kernel family. The
slice is a standard lazy TensileLite library that the stock loader reads: a
master file, the index mapping, and one entry (`.dat`) and code object (`.co`)
per published executable. Other slices stay unread. A code object loads once
per device and stays loaded when a later publish refreshes that slice. The directory also holds `helpers/`, one relocatable per helper
kernel, named for the helper kernel's symbol. A solution's code object is the
fully linked executable, produced by linking the main kernel with those helper
relocatables. The files the library reads and writes are MessagePack.
TensileLite's writer (`MessagePackWriteLibraryData` and
`MessagePackWriteLibraryMapping`) writes the master, the mapping and the
entries from library objects built in C++. The JIT library's own documents are
`cache-key.msgpack`, the index allocator and the index-to-slice map. For
example:

```text
$HIPBLASLT_JIT_LIBRARY_PATH/
└── v1/
    └── <git revision>/                    one directory per hipBLASLt build
        ├── allocator.dat                  next free solution index
        ├── lock                           publication lock for this build
        └── gfx950_..._replay_1/           one directory per device and backend
            ├── cache-key.msgpack
            ├── indices.msgpack            solution index to slice key
            ├── helpers/
            │   └── <helper kernel>.o      relocatable linked into a solution
            ├── slices/<slice key>/
            │   ├── TensileLibrary_lazy_gfx950.dat
            │   ├── TensileLiteLibrary_lazy_gfx950_Mapping.dat
            │   ├── k/<kernels>/t/<problem>/s/<sizes>.dat
            │   └── k/<kernels>/t/<problem>/s/<sizes>.co
            └── staging/                   temporary files before rename
```

An entry name is a filesystem-safe literal of the kernels, the problem type and
the sizes. An entry holds the supported solutions published with one
executable, each under its own allocated index, and that executable beside it
as the code object. Publishers of that same identity choose the same files,
and a solution already in the entry keeps its index. Each solution has its own
master row, `And(SizeEqual(M), SizeEqual(N), SizeEqual(batch), SizeEqual(K),
<the solution's problem predicate>)`, pointing to a placeholder for the entry,
so a lookup matches only the exact sizes within the full ProblemType. The solution's own
hardware, problem and task predicates, including its workspace requirement,
still run on every match. An entry this build cannot read fails the lookup
instead of being skipped, because only this build wrote it. A different comgr
path or ROCm path does not reject the directory.

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

**Indices.** JIT solutions use solution indices from `jitIndexBase` (2^30) up
to, but not including, `jitIndexEnd` (0x60000000): 2^29 indices. Indices from
0x60000000 to `INT32_MAX` are left to runtime user kernel registration and are
never JIT indices. Prebuilt libraries stay below 2^30; one with an index in the
JIT range disables the JIT solution library, as above, and one with indices
from 0x60000000 up does not. `allocator.dat` holds the next index, so indices
stay unique across every key directory under a root and are never reused while
the root exists. Publication fails once the range is exhausted, without
allocating `jitIndexEnd`. `tensile_host.cpp` routes indices in the JIT range
to the JIT solution library, with its own solution adapter whose code-object
directory is the slice, and all other indices to the prebuilt library. An
index that no JIT library holds resolves to an empty library, so callers
report their usual missing-solution error.

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
  ID. The main kernel's declared wavefront size is the one the assembler uses. An
  `Executable` build also accepts a main kernel already assembled into a
  `Relocatable`.
- HIP source: `AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE` with
  `--rocm-path` and a `-cuid` taken from the source name. A HIP helper
  compiles to its own `Relocatable`. `JitLibrary::acquireHelper` stores that
  relocatable as a file in the cache-key directory, named for the helper
  kernel's symbol, because a source file name such as a TensileLite
  `Kernels.cpp` can differ per solution. A hit returns it and does not compile.
  A miss compiles it once and stores it under the publish lock. A bundle's
  helpers are HIP sources, which an `Executable` build compiles, or
  relocatables, which it links as they are; the build keeps no second cache.
  comgr's `AMD_COMGR_CACHE` stays comgr's; it cannot be stored in the library.
  The ROCm path is `HIP_PATH` when set, otherwise the prefix of the loaded HIP
  runtime.
- Link: `AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE` links the main kernel
  and its helpers into one `Executable`, with `-Xlinker --build-id=sha1`. The
  solution library stores that executable.

The generator and the builder use the same code-object version, which
`GenerationRequest::codeObjectVersion` and `SourceCodeBundle::codeObjectVersion`
carry (4 by default). The output is a
raw, uncompressed executable code object. comgr cannot bundle or compress it,
and `hipModuleLoadData` accepts raw executable and linkable format (ELF)
objects. A generator needs no offload bundler.

After linking, the builder reads the executable's metadata and checks that its
instruction set architecture (ISA) is the device's, that its code-object
version is the bundle's, and that it defines the kernel of every solution in
`SourceCodeBundle::solutions` and every name in `SourceCodeBundle::kernelNames`.
The builder writes no file. A failure's
message carries the first error line of the comgr log, and `Status::log`
carries the full log. A caller that owns a scratch directory passes the status
to `keepBuildLog`, which appends the log to `comgr.log` there and names that
file in the message.

### Loading a published solution

`JitLibrary` is a TensileLite `SolutionLibrary`. After publish, hipBLASLt
launches that solution the same way it launches Equality and Origami solutions.
The stored record is a `ContractionSolution` with the code object, kernel name,
`ContractionSolution::ProblemType`, and the predicates those libraries already
expect. Nothing else holds a published solution in process.

`supportingSolutions`, in `hipblaslt-jit-library.hpp`, decides which of a
bundle's solutions serve a problem before they are published. TensileLite's
`selectionPredicate` evaluates each solution's problem, task and hardware
predicates, with the workspace limit as the problem's workspace, and the
solution's `requiredWorkspaceSize` must fit that limit. It keeps the bundle's
order, best first. A backend contributes one `SourceCodeBundle`. `publish`
takes the executable built from it and the solutions `supportingSolutions`
kept, and adds them as one entry.

Each solution of an entry has its own `Problem` row. Their hardware and
problem predicates decide which problems each one serves, so a lookup lists,
in index order, every solution of a matching entry that serves the problem,
and nothing for a problem none of them was generated for.

### Source bundle format

A source bundle is one kernel's sources stored as a directory.
`source_bundle::readSourceBundle`, in `hipblaslt-jit-source-bundle.hpp`, reads
the sources into a `SourceCodeBundle` and requires one main kernel. The caller
sets the target and solutions, which the directory does not hold.
`readTensileSourceBundle` reads the solutions from the assembly's
`custom.config`, as above. A bundle with a `library/` entry is rejected,
because a kernel's solutions come from the kernel itself.

| Path | Contents |
| --- | --- |
| `sources/*.s` | The assembly main kernel, at most one |
| `sources/*.hip`, `sources/*.cpp` | The HIP main kernel when there is no assembly, otherwise HIP helpers |
| Other files in `sources/` | Headers |
| `manifest.json` | A JavaScript Object Notation (JSON) provenance record that hipBLASLt does not read |

Artifact paths must be relative and stay inside the bundle, including through
symbolic links, and `sources/` may hold only regular files. The reader bounds
the file count (1024), each file (64 MiB) and the sources in total (256 MiB).

The JIT tests use one kernel, generated for gfx90a, gfx942 and gfx950, in
`clients/tests/jit/data`, of which only the assembly is committed. The assembly
keeps the kernel's TensileLite solution state in `custom.config`, which
`TensileliteAssembly::make` reads. `jit-publish` builds that assembly for the
current device and publishes it into a JIT solution library: `plain` as one
solution and `plain-pair` as two solutions in one entry. `jit-loader` reads
that library and finds each solution at its K. The replay backend reads the
same bundles in place.
[The data README](clients/tests/jit/data/README.md) gives the command that
generated the assembly.
