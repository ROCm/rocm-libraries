# hipBLASLt just-in-time (JIT) GEMM generation

This source guide describes just-in-time (JIT) kernel generation for general
matrix multiplication (GEMM) in hipBLASLt. It is written for hipBLASLt and
TensileLite contributors and integration developers, and it is maintained under
the existing code-owner rules in [.github/CODEOWNERS](../../.github/CODEOWNERS).
It is not a released application programming interface (API) or support
statement, and it describes what the source implements today. The
[JIT test guide](clients/tests/jit/README.md) covers building and running the
tests.

## Summary

hipBLASLt runs a GEMM with a kernel from its pre-tuned library, so a problem
that the library serves poorly, or not at all, has no better kernel available.
JIT generation produces kernels for a problem when they are needed. A comgr
code-object builder, a TensileLite source bundle reader and the JIT solution
library are implemented. They build generated kernel sources into code objects
and publish them as TensileLite solutions into a persistent library on disk,
whose solutions a later process loads without building again. A JIT backend,
the generator that turns a request into kernel sources, is not part of this
page.

## Components

The interfaces are in `library/src/amd_detail/hipblaslt-jit-component.hpp`.
They are for compiled-in implementations and are not a stable plugin
application binary interface (ABI).

An `Artifact` is one file: `Assembly` or `Hip` source, a `Relocatable` or an
`Executable` code object. A `SourceCodeBundle` is the main kernel, helpers and
headers that become one executable, with the device's `Target`, the code-object
version, the `ContractionSolution`s the executable runs, best first, and any
other kernel names it must define. `CodeObjectBuilder::build` turns a bundle
into an `Executable`, or into the main kernel alone as a `Relocatable`, which
is how a helper is built.

A `TensileliteAssembly` is a TensileLite kernel's assembly with the solutions
built, through `TensileLite::solutionFromState`, from the solution state in
`custom.config` of its `.amdgpu_metadata`. `readTensileSourceBundle` reads a
source bundle and makes its main kernel that assembled `Relocatable`, which an
`Executable` build links without assembling again.

A `Target` is a TensileLite `AMDGPU::Processor` with an `Any`, `On` or `Off`
setting for `sramecc` and `xnack`. `Target::fromTargetId` parses a target ID,
such as `hipDeviceProp_t::gcnArchName`, and rejects an unknown processor or
feature.

## Build

`HIPBLASLT_ENABLE_JIT` is off by default, and a build without it compiles and
exports no JIT entry points. The enabled build needs the host library and ROCm's `amd_comgr`
CMake package. comgr compiles HIP sources against the host's C and C++
standard library headers, so those must be installed where JIT runs. From the
repository root:

```bash
project_root="$PWD"
project_build="$project_root/projects/hipblaslt/build/release"
cmake -S "$project_root/projects/hipblaslt" -B "$project_build" \
  -DHIPBLASLT_ENABLE_JIT=ON -DHIPBLASLT_ENABLE_HOST=ON \
  -DHIPBLASLT_BUILD_TESTING=ON \
  -DHIPBLASLT_ENABLE_DEVICE=OFF -DGPU_TARGETS=gfx950
cmake --build "$project_build" --parallel
```

`GPU_TARGETS` is `gfx942` or `gfx950`.

## Persistent solution library

`JitLibrary`, in `hipblaslt-jit-library.hpp`, is a TensileLite
`SolutionLibrary` and the only cache of generated solutions. `publish` adds
the executable built from one bundle as one entry, holding the solutions
`supportingSolutions` keeps for the problem. `lookup` returns up to a requested
number of the indices published for exactly a problem, in the order they were
first published, and `findTopSolutions` returns those solutions for the query
`beginQuery` sets on the calling thread. A published solution launches through
the TensileLite solution adapter, the same way Equality and Origami solutions
do.

**Location and permissions.** The caller chooses the root. hipBLASLt creates
each directory it uses with mode 0700 on Linux, and refuses one that is a
symbolic link, not a directory, owned by another user, or writable by group or
others. Windows checks only for reparse points and non-directories. A refused
directory, a privileged (setuid or setgid) process, a build that reads YAML
libraries and a build without a git revision disable the library, and each use
then fails with `JIT solution library disabled: <reason>`.

**Layout.** Each hipBLASLt build reads and writes only its own directory, named
by the git revision `hipblasLtGetGitRevision` reports, so it never loads
kernels that a build packing kernel arguments differently published. Within
it, each device and backend has a directory named by a filesystem-safe literal
of the target ID, wavefront size, backend id and backend version. The
toolchain is not part of that name. A query opens one slice: a standard lazy
TensileLite library, keyed by TensileLite's lazy placeholder grain, with one
entry (`.dat`) and code object (`.co`) per published executable. All files are
MessagePack. TensileLite's writer writes the master, the mapping and the
entries; the JIT library writes only the index allocator and the index-to-slice
map.

```text
<root>/
└── v1/
    └── <git revision>/                    one directory per hipBLASLt build
        ├── allocator.dat                  next free solution index
        ├── lock                           publication lock for this build
        └── gfx950_..._<backend>_<version>/ one directory per device and backend
            ├── indices.msgpack            solution index to slice key
            ├── helpers/<helper kernel>.o  relocatable linked into a solution
            ├── slices/<slice key>/
            │   ├── TensileLibrary_lazy_gfx950.dat
            │   ├── TensileLiteLibrary_lazy_gfx950_Mapping.dat
            │   ├── k/<kernels>/t/<problem>/s/<sizes>.dat
            │   └── k/<kernels>/t/<problem>/s/<sizes>.co
            └── staging/                   temporary files before rename
```

An entry is named for its kernels, problem type and sizes, so a second publish
of the same identity reuses its files and its solutions' indices. Each solution
has its own master row, `And(SizeEqual(M), SizeEqual(N), SizeEqual(batch),
SizeEqual(K), <the solution's problem predicate>)`, so a lookup matches only
the exact sizes, and the solution's own predicates still run on every match.
An entry this build cannot read fails the lookup instead of being skipped.

**Indices.** JIT solutions use the indices TensileLite reserves for them in
`LocalKernelIndex.hpp`, from `JitKernelIndexBase` (2^30) up to, but not
including, `LocalKernelIndexBase` (0x60000000). `allocator.dat` holds the next
index, so indices are unique across a build's directories and never reused.
Publication fails once the range is exhausted.

**Publication and refresh.** A publisher holds `lock` while it allocates
indices and writes `allocator.dat`, the code objects, the entries, the mapping
and the master, in that order. Each file is written under `staging/` and
renamed into place, so a reader never sees a partial file or a reference to a
missing one, and takes no lock. A publisher waits at most 15 seconds for the
lock; builds and helper compiles run before it takes it. A lookup reloads a
slice when another process has replaced its master or mapping, and resolving an
unknown index reloads it too. Each code object is loaded once per device.

**Clearing.** hipBLASLt never deletes entries. To clear the library, delete the
root, or one device and backend directory, while no process uses it.

**Scale and limits.** Nothing bounds the number of entries or their disk use;
the index range is the only limit. An entry holds one exact (kernels, problem
type, M, N, K, batch), so each new size of a kernel adds one. Measured on gfx950,
on ext4 over NVMe, publishing the committed 16x16x16 `plain` kernel with one
solution per entry into one slice:

- Disk: an entry adds its 45,296-byte code object, its 8,836-byte `.dat`, and
  per solution about 2.6 KB of master, 1.1 KB of mapping and 52 bytes of
  `indices.msgpack`, 57.9 KB in all. 4,000 entries took 232 MB. Paths relative
  to the root reach 1,220 characters, in components of at most 120.
- Publication: a publish rewrites the slice's master and mapping and the
  directory's `indices.msgpack`, so its cost grows with the entries already
  there. It took 4 ms for the first entry, a median of 37 ms at 1,000 and
  169 ms at 4,000. Publishing n entries into one slice writes O(n²) bytes.
  Reading and assembling the bundle took about 12 ms and linking under 1 ms,
  both before the lock.
- Lock: publishers of one build take turns. 1,000 publications took 17 to 20
  seconds from 1, 4 or 8 processes. The slowest single publication took 3.2
  seconds, lock wait included.
- Lookup: the first lookup of a slice in a process reads its master, which took
  0.3 ms at 20 entries and 52 ms at 4,000.
- Device memory: a resolved code object stays loaded on its device until the
  process exits, 2 MiB each. The first load in a process took 144 MiB more.

## Building generated sources

Generators emit assembly or HIP source; they do not assemble, link or bundle.
`makeComgrBuilder()` builds in process through comgr
(`hipblaslt-jit-builder.cpp` and `hipblaslt-jit-code-object.cpp`):

- Assembly is assembled to a `Relocatable`. Its `.amdgcn_target` directive must
  name the device's processor and only features the device has, and is then
  rewritten to the device's full target ID.
- HIP source is compiled to a `Relocatable` with `--rocm-path` (`HIP_PATH`
  when set, otherwise the prefix of the loaded HIP runtime) and a `-cuid` taken
  from the source name. `JitLibrary::acquireHelper` stores a helper's
  relocatable under the helper kernel's symbol, and compiles it only on a miss.
- The main kernel and its helpers are linked into one raw, uncompressed
  `Executable` with `-Xlinker --build-id=sha1`, which `hipModuleLoadData`
  accepts as is.

The builder then checks that the executable's ISA is the device's, that its
code-object version is the bundle's (4 by default), and that it defines the
kernel of every solution and every name in `kernelNames`. It writes no file. A
failure's message carries the first error line of the comgr log and
`Status::log` the full log; `keepBuildLog` appends that log to `comgr.log` in a
caller's scratch directory.

## Source bundle format

A source bundle is one kernel's sources stored as a directory.
`source_bundle::readSourceBundle`, in `hipblaslt-jit-source-bundle.hpp`, reads
it into a `SourceCodeBundle`. The caller sets the target and solutions, or
`readTensileSourceBundle` reads the solutions from the assembly's
`custom.config`. A bundle with a `library/` entry is rejected.

| Path | Contents |
| --- | --- |
| `sources/*.s` | The assembly main kernel, at most one |
| `sources/*.hip`, `sources/*.cpp` | The HIP main kernel when there is no assembly, otherwise HIP helpers |
| Other files in `sources/` | Headers |
| `manifest.json` | A JSON provenance record that hipBLASLt does not read |

Artifact paths must be relative and stay inside the bundle, including through
symbolic links, and `sources/` may hold only regular files. The reader bounds
the file count (1024), each file (64 MiB) and the sources in total (256 MiB).
