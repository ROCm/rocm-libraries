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
parts of hipBLASLt that build and load what a backend generates, and the
JIT solution library that stores a built kernel for a later load. The
[JIT test guide](clients/tests/jit/README.md) covers building and running the
tests.

## Summary

hipBLASLt runs a GEMM with a kernel from its pre-tuned library, so a problem
that the library serves poorly, or not at all, has no better kernel available.
JIT generation produces kernels for a problem when they are needed. This page
describes the parts that turn generated kernel sources into a loaded
TensileLite solution: a comgr code-object builder, a source bundle reader, a
TensileLite loader and a JIT solution library. The library does not call them
yet. The JIT tests build the committed assembly and publish it into the JIT
solution library, and the loader test reads that library back. No generator
backend is implemented, and no API reaches JIT.

## Current behavior

The interfaces are in `library/src/amd_detail/hipblaslt-jit-component.hpp`. An
`Artifact` is one file: assembly, HIP source, or a code object. A
`SourceCodeBundle` is the main kernel, HIP helpers and headers that become one
solution. It is not a TensileLite library entry. `CodeObjectBuilder::build`
compiles that bundle with comgr into a code object `Artifact`. `JitLibrary`
writes the TensileLite solution record when it publishes the code object. The
interfaces are for compiled-in implementations and do not establish a stable
external plugin application binary interface (ABI).

### Build

`HIPBLASLT_ENABLE_JIT` is disabled by default. A disabled build compiles no
JIT code. The enabled build requires the host library, ROCm and ROCm's
`amd_comgr` CMake package, which only a JIT build links. comgr compiles HIP
sources against the host's C and C++ standard library headers, so those must be
installed where JIT runs. From the repository root:

```bash
project_root="$PWD"
project_build="$project_root/projects/hipblaslt/build/release"
cmake -S "$project_root/projects/hipblaslt" -B "$project_build" \
  -DHIPBLASLT_ENABLE_JIT=ON -DHIPBLASLT_ENABLE_HOST=ON \
  -DHIPBLASLT_BUILD_TESTING=ON \
  -DHIPBLASLT_ENABLE_DEVICE=OFF -DGPU_TARGETS=gfx950
cmake --build "$project_build" --parallel
```

The [JIT test guide](clients/tests/jit/README.md) lists the test targets and
the validation commands.

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
`BuildRequest::codeObjectVersion` carries (4 by default). The output is a
raw, uncompressed executable code object. comgr cannot bundle or compress it,
and `hipModuleLoadData` accepts raw executable and linkable format (ELF)
objects. A generator needs no offload bundler.

After linking, the builder reads the code object's metadata and checks that its
instruction set architecture (ISA) is the device's and that it defines every
main kernel of the entry's solutions. Its message
carries the first error line of the comgr log, and the builder appends the full
log to `comgr.log` in `BuildRequest::scratch` and names that file in the
message.

### JIT solution library

`JitLibrary` in `hipblaslt-jit-library.{hpp,cpp}` is a TensileLite
lazy-loading library on disk. Indices start at `2^30` (`jitIndexBase`). Each
directory holds a master library, an index mapping, one entry file and one
code object per published solution, all MessagePack, plus `cache-key.msgpack`.
Directory and entry names are filesystem-safe literals of the device, backend,
kernel, problem type and sizes. A row matches the solution's problem type and
its exact sizes. A solution whose code-object version or kernarg version this
build cannot run is skipped; the rest of the library still loads. Publishing
replaces files atomically, in an order that never leaves a reference to a
missing file. `readTensileSourceBundle` reads a source directory into a
`SourceCodeBundle` and the kernel name. A caller that checks predicates loads
the TensileLite entry with `loadGemmLibrary`. The tests publish and load the
JIT solution library instead of that directory.

### Loading a published solution

`JitLibrary` loads a published solution as a TensileLite `ContractionSolution`
and a `SolutionAdapter`. `loadGemmLibrary` reads a solution record and requires
local solutions 0 to N-1. `hipblaslt-jit-loader.{hpp,cpp}` does not wrap that
library in another type.

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
device and publishes it into a JIT solution library: `plain` as one solution
and `plain-pair` as two. `jit-loader` reads that library and resolves the
pre-generated kernel.
[The data README](clients/tests/jit/data/README.md) gives the command that
generated the assembly.
