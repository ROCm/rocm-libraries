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
run. Its interfaces and a comgr code-object builder are implemented; no
backend, loader or store is implemented yet, and nothing in hipBLASLt calls
`Jit`.

## Current behavior

### Components

`Jit`, in `library/src/amd_detail/hipblaslt-jit-component.{hpp,cpp}`, runs one
request for one device target through these stages. Each stage is an
interface, so that each implementation can be replaced and tested on its own.

| Stage | Interface | Contract |
| --- | --- | --- |
| Generate | `Backend::generate(GenerationRequest, std::vector<GeneratedSolution>&)` | Returns up to `GenerationRequest::count` solutions, best first, and builds and loads nothing. Each `GeneratedSolution` holds a one-solution TensileLite library entry, its main kernel name, and the source units to build. `NotSupported` means the request is outside the backend's domain. |
| Build | `CodeObjectBuilder::build(GeneratedSolution, GenerationRequest, BuiltSolution&)` | Builds a solution's units into code objects. `GeneratedSolution` has no code-object field; only `BuiltSolution` adds the main code object and its helpers. |
| Support | `SolutionLoader::support` | Evaluates the entry's predicates and workspace for the request, and loads no code. |
| Publish | `SolutionStore::publish` | Stores built solutions and returns one library index per solution, in order. `SolutionStore::lookup` returns the indices of stored solutions for exactly a request. |
| Load | `SolutionLoader::load` | Loads the code objects into a process-local executable `KernelBundle`. |

`Jit::generate` is the only code that sequences these stages. It asks the
backend for at most the requested count of solutions in a private scratch
directory, forwarding the workspace limit and the kernels the caller already
has. It then builds each solution and checks its support until the count is
reached. With a store it publishes the supported solutions, and loads them only
when publishing fails; without a store it loads them. A failure in one
solution's build or support skips that solution and keeps the rest.

Each failure is recorded with its stage (configure, generate, build, support,
load or publish) in `Jit::Outcome::failures`, in the order it happened.
The scratch directory is created under the temporary directory, removed when
the call succeeds, and kept after a failure that left files in it. `Jit`
requires a backend, a builder and a loader; the store is optional. One `Jit`
can generate from several threads at once.

`OperationRequest` names only its kind, so `Jit` sees no GEMM. The interfaces
are for compiled-in implementations and do not establish a stable external
plugin application binary interface (ABI).

### Build

`HIPBLASLT_ENABLE_JIT` is disabled by default. A disabled build compiles none of
the JIT sources. The enabled build requires the host library, ROCm and ROCm's
`amd_comgr` CMake package, which only a JIT build links. comgr compiles helper
sources against the host's C and C++ standard library headers, so those must
be installed where JIT runs. From the repository root:

```bash
project_root="$PWD"
project_build="$project_root/projects/hipblaslt/build/release"
cmake -S "$project_root/projects/hipblaslt" -B "$project_build" \
  -DHIPBLASLT_ENABLE_JIT=ON -DHIPBLASLT_ENABLE_HOST=ON \
  -DHIPBLASLT_BUILD_TESTING=ON \
  -DHIPBLASLT_ENABLE_DEVICE=OFF -DGPU_TARGETS=gfx950
cmake --build "$project_build" --parallel
```

The `jit` CMake preset enables this feature for a new configuration. The
[JIT test guide](clients/tests/jit/README.md) lists the test targets and the
validation commands.

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
