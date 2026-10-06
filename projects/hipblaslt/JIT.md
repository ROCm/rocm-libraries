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
pre-generated source bundles. No generator backend is implemented yet.
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
solution library and does not start a build. A null stream and the legacy
stream are not capturing. GEMM is the implemented operation.

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

The process backend replays source bundles when `HIPBLASLT_JIT_TEST_REPLAY`
names them, a whitespace-separated list read once per process. When that
variable is unset, a build with the HipKittens backend uses it instead. A JIT
build with neither warns once: mode `1` leaves the query unchanged, and mode
`2` returns no algorithms. Either backend implements generation only. Every
query publishes through the JIT solution library.

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
| Generate | `Backend::generate(GenerationRequest, std::vector<GeneratedSolution>&)` | Returns entries holding up to `GenerationRequest::count` solutions, best first, and builds and loads nothing. Each `GeneratedSolution` holds a TensileLite library entry for one processor with one or more solutions, the names of their main kernels, and the source units to build. `NotSupported` means the request is outside the backend's domain. |
| Build | `CodeObjectBuilder::build(GeneratedSolution, BuildRequest, BuiltSolution&)` | Builds an entry's units into one code object. `GeneratedSolution` has no code-object field; only `BuiltSolution` adds the code object. |
| Support | `SolutionLoader::support` | Evaluates the predicates and workspace of the entry's solutions for the request, returns the local indices of those that support it, best first, and loads no code. |
| Publish | `SolutionStore::publish` | Stores the supported solutions of built entries and returns one library index per solution, in order. `SolutionStore::lookup` returns the indices of stored solutions for exactly a request. |
| Load | `SolutionLoader::load` | Loads the code objects once and returns a process-local executable `KernelBundle` for each supported solution. |

`Jit::generate` is the only code that sequences these stages. It asks the
backend for at most the requested count of solutions in a private scratch
directory, forwarding the workspace limit and the kernels the caller already
has. It then builds each entry and checks the support of its solutions until
the count of supported solutions is reached. With a store it publishes the
supported solutions, and loads them only when publishing fails; without a
store it loads them. A failure in one entry's build, support or load skips
that entry and keeps the rest.

Each failure is recorded with its stage (configure, generate, build, support,
load or publish) in `Jit::Outcome::failures`, in the order it happened.
The scratch directory is created under the temporary directory, removed when
the call succeeds, and kept after a failure that left files in it. `Jit`
requires a backend, a builder and a loader; the store is optional. One `Jit`
can generate from several threads at once.

`OperationRequest` names only its kind, so `Jit` sees no GEMM. The interfaces
are for compiled-in implementations and do not establish a stable external
plugin application binary interface (ABI).

The implementations are:

- Code-object builder: `makeComgrBuilder()`; see
  [building generated sources](#building-generated-sources).
- Loader: `makeTensileLoader()` reads the entry with the TensileLite loader (see
  [loading a built solution](#loading-a-built-solution)), checks the support
  and workspace of each solution with TensileLite's predicates, in index order,
  and returns a process-local `TensileGemmBundle` for each supported solution.
  The bundles of one entry share its library and adapter. It is in
  `rocblaslt/src/tensile_host.cpp`, next to the GEMM problem translation that
  the support check reuses.
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
  they hold the requested count of such solutions. It skips a bundle when every
  such solution's kernel is excluded; the comgr builder still builds them.
  It implements `Backend`, so `Jit` builds and loads its entries with the same
  builder and loader as any other backend. Tests reach it through
  `jit::replay::createBackend` in `hipblaslt-jit-replay.hpp`, and every JIT
  build compiles it.
- HipKittens backend: `hipblaslt-jit-hipkittens.cpp`, built when
  `HIPBLASLT_JIT_ENABLE_HIPKITTENS` is on and `GPU_TARGETS` includes gfx950.
  It compiles the gfx950 TN templates in `library/src/amd_detail/hipkittens`
  with comgr's generic HIP path. The HipKittens headers are staged next to
  the library; `HIPBLASLT_JIT_HIPKITTENS_PATH` overrides that directory.
  `HIPBLASLT_JIT_TEST_REPLAY` still selects the replay backend. A build
  without the option keeps replay as the only process backend.

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
`gfx950`. `HIPBLASLT_JIT_ENABLE_HIPKITTENS=ON` adds the gfx950 HipKittens
backend. It needs a network fetch of the pinned HipKittens commit unless
`FETCHCONTENT_SOURCE_DIR_HIPKITTENS` points at an unpacked archive. The
[JIT test guide](clients/tests/jit/README.md) lists the test targets and the
validation commands.

### Algorithm lifetime and failures

The returned heuristic result contains the required workspace size. Supply that
workspace and follow the same handle, stream and workspace sharing rules as
`hipblasLtMatmul` and `Gemm` calls using prebuilt algorithms. A heuristic
result is a JIT solution library index, from 2^30, and it runs through the
same launch path as a prebuilt index. Another process that queries the same
problem receives that index.

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
so on. Each entry holds one supported solution of a generated entry, with its
solution index rewritten to the allocated one and the generated entry's code
object beside it. Each master row is
`And(SizeEqual(M), SizeEqual(N), SizeEqual(batch), SizeEqual(K), <the entry's
problem predicate>)` pointing to a placeholder for the entry, so a lookup
matches only the exact sizes within the full ProblemType. The solution's own
hardware, problem and task predicates, including its workspace requirement,
still run on every match.

**Cache key.** `cache-key.json` records, in canonical JSON:

| Field | Contents |
| --- | --- |
| `target` | Full target ID with features (for example `gfx950:sramecc+:xnack-`), ISA, TensileLite library architecture and wavefront size |
| `backend` | Backend identifier and version. Each backend defines its version to change whenever its output can; the replay backend's is a hash of its replayed bundles. |
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
Prebuilt libraries stay below that range; one that reaches it disables the JIT
solution library, as above. `allocator.dat` holds
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
- HIP source: `AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE` with
  `--rocm-path` and a content-derived `-cuid`, so HIP objects link together.
  The ROCm path is `HIP_PATH` when set, otherwise the prefix of the loaded HIP
  runtime.
- Link: `AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE` joins the assembly
  and HIP relocatables into one code object per solution, with
  `-Xlinker --build-id=sha1`.

The generator and the builder use the same code-object version, which
`GenerationRequest::codeObjectVersion` and `BuildRequest::codeObjectVersion`
carry (4 by default). The output is a
raw, uncompressed executable code object. comgr cannot bundle or compress it,
and `hipModuleLoadData` accepts raw executable and linkable format (ELF)
objects. A generator needs no offload bundler.

After linking, the builder reads the code object's metadata and checks that its
instruction set architecture (ISA) is the device's and that it defines every
main kernel of the entry's solutions. Its message
carries the first error line of the comgr log, and the builder appends the full
log to `comgr.log` in `BuildRequest::scratch` and names that file in the
message.

### Loading a built solution

`hipblaslt-jit-loader.{hpp,cpp}` turns a built TensileLite entry into a loaded
library of its solutions:

- `readTensileSourceBundle` reads a source bundle into a `TensileSource`: a
  `GeneratedSolution` (the library entry, the main kernel names of its
  solutions, each `sources/*.s` as an assembly unit, and each HIP source with
  the bundle's headers as a HIP unit) and the TensileLite library parsed from
  that entry. `loadGemmLibrary` is that parse. A caller that needs the library
  keeps the one `readTensileSourceBundle` returned.
- `parseTensileBundle` reads a built solution's entry into a TensileLite
  `MasterSolutionLibrary` for the device's hardware. It requires local
  solutions 0 to N-1, each naming one of the built kernels and every built
  kernel named by one of them, and loads no code.
- `loadTensileBundle` loads the built code object into a new
  `SolutionAdapter` and resolves every main kernel.

The entry's solutions are listed best first, each with its own `Problem` row.
Their hardware and problem predicates decide which problems each one serves,
so `findBestSolution` on the loaded library selects the first solution that
serves a problem, and nothing for a problem none of them was generated for.

### Source bundle format

A source bundle is a generated library entry stored as a directory.
`source_bundle::readSourceBundle`, in `hipblaslt-jit-source-bundle.hpp`, reads
one by directory convention and requires at least one kernel source:

| Path | Contents |
| --- | --- |
| `library/TensileLibrary.dat` | The library entry, stored as uncompressed MsgPack |
| `sources/*.s` | Kernel assembly |
| `sources/*.hip`, `sources/*.cpp` | HIP kernel sources |
| Other files in `sources/` | Headers that the HIP sources include |
| `manifest.json` | A JavaScript Object Notation (JSON) provenance record that hipBLASLt does not read |

Artifact paths must be relative and stay inside the bundle, including through
symbolic links, and `sources/` may hold only regular files. The reader bounds
the file count (1024), each file (64 MiB) and the sources in total (256 MiB).

The JIT tests use one kernel, generated for gfx90a, gfx942 and gfx950, in
`clients/tests/jit/data`, of which only the assembly is committed. The assembly
records the kernel-argument and persistent-loop argument layout versions of the
generator that wrote it. The `jit-bundles` test writes source bundles from it
for each architecture, each with the library entry and manifest built from the
assembly and a description of each solution: `plain` with one solution and
`plain-pair` with two. The tests that need a GPU run the bundles of its
architecture.
[The data README](clients/tests/jit/data/README.md) gives the command that
generated the assembly.
