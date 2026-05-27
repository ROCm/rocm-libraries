# Kernel Packaging Research

Goal: package a single hipBLASLt-selected kernel into a standalone executable
that loads the `.co`, allocates inputs, runs the kernel, and times it - for
apples-to-apples benchmarking against non-hipBLASLt kernels.

## Reference: example solution

Example name produced by `okl.py` for a representative bf16 problem:

```
Cijk_Alik_Bljk_BBS_BH_UserArgs_MT32x32x128_MI16x16x1_SN_LDSB0_..._ISA942_...
```

with solution index `45732` in the shipped gfx942 lazy library. The name encodes:
- `Cijk_Alik_Bljk` - tensor contraction with op(A)=T, op(B)=N, C/D row-major-ish naming.
- `BBS` - data types (BFloat16/BFloat16 in/out, single-precision compute is the `BBS` group).
- `BH` - bias / HPA (high-precision accumulate) variants.
- `UserArgs` - this kernel uses the **user-args ABI** (a struct in device memory) rather than the legacy per-launch byte buffer.
- `MT32x32x128` - macro tile MxNxK.
- `MI16x16x1` - MFMA shape (per-wave instruction shape).
- `SN` - source-kernel variant; `LDSB0` - LDS B-buffer size.
- `ISA942` - compiled for gfx942.

Everything that follows is needed to answer: where does this kernel's compiled
machine code physically live, and what bytes does it want as launch arguments?

## 1. How kernels are stored

### 1.1 On-disk layout

After a normal install the device library lives at
`/opt/rocm/lib/hipblaslt/library/` (or, in a dev build, at
`build/Tensile/library/`). The runtime resolves that directory either from the
`HIPBLASLT_TENSILE_LIBPATH` env var or relative to the loaded shared library;
see `clients/samples/01_hipblaslt_gemm/.../` callers and `okl.py` for the
candidate-list logic.

`ls /opt/rocm/lib/hipblaslt/library/ | wc -l` on this box reports 3027 entries.
The breakdown is:

| Glob | Count | Role |
|---|---|---|
| `TensileLibrary_<types>_<arch>.co` | 1506 | Per-problem-type compiled kernel shards (clang-offload-bundler `.co`) |
| `TensileLibrary_<types>_<arch>.dat` | 1508 | Per-shard msgpack solution-metadata sidecars |
| `TensileLibrary_lazy_<arch>.dat` | 12 | Per-arch heuristic / dispatch index (placeholder library) |
| `Kernels.so-000-<arch>.hsaco` | 13 | Per-arch "helper" code object (HIP-generated source kernels: beta-only, conversion, reduction, etc.) |
| `extop_<arch>.co` | 12 | hipBLASLt extension-op kernels (softmax/layernorm/AMax/etc., separate from Tensile) |
| `hipblasltExtOpLibrary.dat`, `hipblasltTransform.hsaco` | 1, 1 | Extension-op index + transform helper |
| `TensileLiteLibrary_lazy_Mapping.dat` | 1 | Solution-index to shard-filename mapping (msgpack) |

For our example (`Cijk_Alik_Bljk_BBS_BH_..._UserArgs_..._ISA942_...`) the
relevant shard is

```
/opt/rocm/lib/hipblaslt/library/TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx942.co
```

paired with the `.dat` of the same prefix. Filename schema (from the Python
`MasterSolutionLibrary` writer and `PlaceholderLibrary::getCodeObjectFileName()`,
`tensilelite/include/Tensile/PlaceholderLibrary.hpp:216-219` and `:198`):

```
TensileLibrary_<inA><inB>_<outC><outD>_<flags>_<computeFlags>_<Contraction|...>_<index-order>_<arch>.co
```

So `BB_BB` = bf16 in / bf16 out, `HA` = high-precision accumulate, `Bias` = bias
fused, `SAV` = scaleAlphaVec, `UA` = supports DeviceUserArguments, `HPA` = HPA
compute type, `Contraction_l_Alik_Bljk_Cijk_Dijk` = the contraction index order.
The arch suffix is the strict gfx target (xnack variants are appended as
`-xnack-` / `-xnack+` before `.co`, stripped by `removeXnack()` at
`tensilelite/src/hip/HipSolutionAdapter.cpp:79-87`).

Sizes for the example on this machine:

```
2.6 MB  TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx942.co
406 KB  TensileLibrary_lazy_gfx942.dat
144 KB  TensileLiteLibrary_lazy_Mapping.dat
 17 MB  Kernels.so-000-gfx942.hsaco
```

### 1.2 Per-shard bundling (lazy loading)

Bundling is **per problem-type / per-arch shard**, not per kernel. One `.co`
holds many kernels (one ELF symbol each), all kernels belonging to the same
problem-type group. The Python pipeline assembles one shard per group with the
clang-offload-bundler; see `tensilelite/Tensile/Toolchain/Component.py:286-309`
(`compress()`) for the bundler invocation:

```python
args = [
    self._component_path, "--compress", "--type=o", "--bundle-align=4096",
    f"--targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa-unknown-{target}",
    f"--input={devnull}", f"--input={srcPath}", f"--output={destPath}",
]
```

That bundler step is the one that wraps the unbundled gfx ELF into the "CCOB"
zstd-compressed container observed below.

The lazy-load index `TensileLibrary_lazy_<arch>.dat` is a msgpack-serialized
`MasterSolutionLibrary` where each problem-type subtree is a
`PlaceholderLibrary` referring to a shard filename. At dispatch time, the
placeholder lazy-loads its real per-shard library file (the matching
`TensileLibrary_..._<arch>.dat`) and stamps every solution it returns with
`solution->codeObjectFilename = filePrefix + ".co"` (see
`tensilelite/include/Tensile/PlaceholderLibrary.hpp:201, 218, 231, 247, 273,
294, 313, 340`). That string is what later flows into `KernelInvocation`.

`TensileLiteLibrary_lazy_Mapping.dat` is a separate msgpack mapping used by
`MasterSolutionLibrary` for solution-index lookups, but most dispatches do not
need it - the in-memory library knows its own indexing once loaded.

### 1.3 Inspecting a .co file

The shipped `.co` files are **clang-offload-bundler compressed bundles**. They
do not parse as ELF directly:

```
$ file TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx942.co
... : data

$ xxd ... | head -1
00000000: 4343 4f42 0300 0100 1b85 2800 0000 0000  CCOB......(.....
```

`CCOB` is the magic for the offload-bundler compressed format; the zstd payload
starts at offset 0x1a (`28b5 2ffd` = zstd magic). HIP's `hipModuleLoad`
understands this container directly, but the standard binutils tools do not.

To peel one apart:

```bash
# List bundle entries (targets):
/opt/rocm/llvm/bin/clang-offload-bundler --type=o \
    --input=/opt/rocm/lib/hipblaslt/library/TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx942.co \
    -list
# -> hipv4-amdgcn-amd-amdhsa--gfx942
#    host-x86_64-unknown-linux-gnu-

# Extract the gfx942 ELF:
/opt/rocm/llvm/bin/clang-offload-bundler --type=o \
    --input=/opt/rocm/lib/hipblaslt/library/TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx942.co \
    --targets=hipv4-amdgcn-amd-amdhsa--gfx942 \
    --output=/tmp/bb_bb_alik.elf --unbundle
# -> /tmp/bb_bb_alik.elf is a 74 MB elf64-amdgpu file with ~234k symbols.

# List kernel symbols:
/usr/bin/llvm-objdump --syms /tmp/bb_bb_alik.elf | grep '^[0-9a-f]\+ g .* F .text' | head
```

Two symbol kinds per kernel appear in the table:

| Symbol kind | Section | What it is |
|---|---|---|
| `Cijk_..._WG..._..._..._4_4` | `.text` | The kernel entry point (this is what `hipModuleGetFunction` returns) |
| `Cijk_..._WG..._..._..._4_4.kd` | `.rodata` (64 bytes) | The AMDGPU kernel descriptor (sgpr/vgpr counts, kernarg size, ABI flags) |

Both are `g .protected`. Examples from the shard:

```
00000000001eea00 g  F .text     ... .protected Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x112x128_MI16x16x1_..._WS64_WG16_4_4
00000000001e7240 g  O .rodata 40 .protected Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT64x112x128_MI16x16x1_..._WS64_WG16_4_4.kd
```

Each kernel is a fully self-contained AMDGPU function with its own `.kd` descriptor; nothing is shared between kernels in the shard beyond the ELF
container.

`roc-obj-ls` does *not* work here because the file is a clang-offload-bundler
output, not the host-ELF-with-embedded-fatbin format `roc-obj-ls` expects:

```
$ /opt/rocm/bin/roc-obj-ls /opt/rocm/.../TensileLibrary_BB_BB_..._gfx942.co
Error: No kernel section found
```

For the helper kernels in `Kernels.so-000-gfx942.hsaco` the file is a plain
ELF and standard tools work directly:

```bash
/usr/bin/llvm-objdump --syms /opt/rocm/lib/hipblaslt/library/Kernels.so-000-gfx942.hsaco | head
```

### 1.4 Mapping solution name to .co file

Two equally good ways:

1. **Use the placeholder mapping**. The `.co` basename is exactly the prefix of
   the per-shard `.dat`, and the per-shard `.dat` is named after the problem
   type. For a given solution name you can derive the shard prefix by parsing
   the name, but it is faster to just grep for the symbol in every gfx-matching
   `.co`:

   ```bash
   for f in /opt/rocm/lib/hipblaslt/library/TensileLibrary_*_gfx942.co; do
       /opt/rocm/llvm/bin/clang-offload-bundler --type=o --input="$f" \
           --targets=hipv4-amdgcn-amd-amdhsa--gfx942 --output=/tmp/x.elf --unbundle 2>/dev/null
       /usr/bin/llvm-objdump --syms /tmp/x.elf | grep -q "Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT32x32x128_..." && echo "$f"
   done
   ```

2. **Ask Tensile** by re-running `okl.py` (or `hipblaslt-bench` with
   `HIPBLASLT_LOG_LEVEL=5` / `TENSILE_DB=0x80` for `printCodeObjectInfo()`,
   `tensilelite/src/Debug.cpp`-controlled). The lazy load prints lines like
   `load placeholder library .../TensileLibrary_BB_BB_..._gfx942.dat` and
   `loaded code object .../TensileLibrary_BB_BB_..._gfx942.co`.

Inside the runtime the mapping is set at lazy-load time in
`PlaceholderLibrary::loadPlaceholderLibrary()`
(`tensilelite/include/Tensile/PlaceholderLibrary.hpp:167-214`), stamped onto
each solution at `:201`, and the same string is later read by
`ContractionSolution::generateSingleCall` at
`tensilelite/src/ContractionSolution.cpp:1567`:

```cpp
rv.codeObjectFile = codeObjectFilename.load();
```

This is what `HipSolutionAdapter::launchKernel` then feeds to
`FindCodeObject(kernel.codeObjectFile)` to trigger the on-demand
`hipModuleLoad` of the right shard.

**Bottom line for the standalone exe**: every solution is one ELF symbol in one
`.co` file you can identify with a `clang-offload-bundler --unbundle` +
`llvm-objdump --syms | grep` two-liner. Each kernel is position-independent and
loadable by itself - you do not need any other shard.


## 2. How TensileLite loads and launches a kernel

The launch path is two distinct objects:

1. `ContractionSolution` builds a `KernelInvocation` (name + grid/block + arg
   bytes + name of the `.co` to use). It is pure data assembly; no HIP calls.
2. `hip::SolutionAdapter` takes that `KernelInvocation`, ensures the right
   `.co` is `hipModuleLoad`-ed, resolves the symbol via
   `hipModuleGetFunction`, and calls `hipExtModuleLaunchKernel`.

`KernelInvocation` itself is defined in `tensilelite/include/Tensile/Tensile.hpp:122-138`:

```cpp
struct TENSILE_API KernelInvocation
{
public:
    std::string kernelName;
    std::string codeObjectFile;     // Code object file kernel is located in
    bool isSingleCall = false;
    dim3 clusterDim{1, 1, 1};
    dim3   workGroupSize;
    dim3   numWorkGroups;
    dim3   numWorkItems;
    size_t sharedMemBytes = 0;
    KernelArguments args;
};
```

### 2.1 Entry point: ContractionSolution::solve

Top-level entry: `ContractionSolution::solve` (overload for `ContractionProblem`)
at `tensilelite/src/ContractionSolution.cpp:2675-2698`:

```cpp
std::vector<KernelInvocation> ContractionSolution::solve(ContractionProblem const& problem,
                                                         ProblemInputs const&      inputs,
                                                         Hardware const&           hardware,
                                                         void*       hipHostMemory,
                                                         size_t      hipHostMemorySize,
                                                         hipStream_t stream) const
{
    if(auto gemmProblem = dynamic_cast<ContractionProblemGemm const*>(&problem))
    {
        auto gemmInputs = dynamic_cast<ContractionInputs const*>(&inputs);
        return solve((*gemmProblem), (*gemmInputs), hardware);
    }
    ...
}
```

That second `solve(...)` overload (`:2750-2900`) walks
`gsuSettings`/`StreamKSettings`, then dispatches to either
`generateSingleCall<T_Debug>(problem, inputs, hardware, sk, gsuSettings)`
(typical case for our example) or one of the multi-call paths (split-K with
global accumulation, stream-K with reduction, etc.). For the simple `BBS_BH_UserArgs`
kernel returning one `KernelInvocation`, the relevant builder is
`generateSingleCall` (`tensilelite/src/ContractionSolution.cpp:1446-1569`).

The shape returned from `generateSingleCall` is the same in every dispatch
path:

```cpp
KernelInvocation rv;
rv.isSingleCall = true;
rv.args = KernelArguments(T_Debug);
rv.args.reserve(1024, 128);
rv.kernelName = kernelName;                       // the solution name string
calculateGrid(rv.workGroupSize, rv.numWorkGroups, problem);
...
if(internalArgsSupport.useUniversalArgs)
    kernelArgs<T_Debug, false>(...);              // emits "gemm_count","internalArgs",...
singleCallArgs<T_Debug, true>(problem, inputs, 0, &hardware,
                              problemNumGroupTiles, rv.numWorkGroups,
                              rv.args, sk);       // emits sizes/strides/ptrs/alpha/beta...
...
rv.codeObjectFile = codeObjectFilename.load();    // shard the kernel lives in
return rv;
```

### 2.2 SolutionAdapter (module load + function lookup)

`hip::SolutionAdapter`
(`tensilelite/include/Tensile/hip/HipSolutionAdapter.hpp`,
`tensilelite/src/hip/HipSolutionAdapter.cpp`) is the runtime that owns:

- `std::vector<hipModule_t> m_modules;` - every `hipModuleLoad`-ed module.
- `std::unordered_map<std::string, hipFunction_t> m_kernels;` - solution name
  to resolved `hipFunction_t`, populated lazily on first `getKernel(name)`.
- `std::unordered_set<std::string> m_loadedCOFiles;` - names of `.co`s
  already loaded (xnack-stripped).
- `std::string m_codeObjectDirectory;` and `m_lazyLoadArchitecture` - so that
  on a memory-pressure `hipModuleLoad` failure the lazy state can be
  re-initialized (HipSolutionAdapter.cpp:96-135).

The key methods:

- `loadCodeObjectFile(path)`
  (`tensilelite/src/hip/HipSolutionAdapter.cpp:89-158`): direct
  `hipModuleLoad(&module, path.c_str())`, push to `m_modules`. This is what
  consumes the CCOB-bundled `.co` files - HIP handles unbundling internally.

- `loadCodeObject(image)` / `loadCodeObjectBytes(bytes)` (`:160-185`):
  `hipModuleLoadData` from an in-memory buffer (useful if you embed the `.co`
  in your binary as a `.rodata` blob).

- `getKernel(rv, name)` (`:291-323`): walks `m_modules` calling
  `hipModuleGetFunction(&rv, module, name.c_str())`, caches the result. The
  function name passed in is **the full solution-name string** (the same string
  found in the symbol table of the unbundled ELF).

- `launchKernel(KernelInvocation, stream, startEvent, stopEvent, isKernelLoaded)`
  (`:405-532`) is the actual launch:

```cpp
hipFunction_t function;
getKernel(function, kernel.kernelName);

void*  kernelArgs = const_cast<void*>(kernel.args.data());
size_t argsSize   = kernel.args.size();

void* hipLaunchParams[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kernelArgs,
                           HIP_LAUNCH_PARAM_BUFFER_SIZE,    &argsSize,
                           HIP_LAUNCH_PARAM_END};

hipExtModuleLaunchKernel(function,
                         kernel.numWorkItems.x, kernel.numWorkItems.y, kernel.numWorkItems.z,
                         kernel.workGroupSize.x, kernel.workGroupSize.y, kernel.workGroupSize.z,
                         kernel.sharedMemBytes,
                         stream,
                         nullptr,                /* kernelParams - unused for extended API */
                         (void**)&hipLaunchParams,
                         nullptr, nullptr);      /* start/stop events handled separately above */
```

Three things to note:

- The arg buffer is passed via the driver-style `HIP_LAUNCH_PARAM_BUFFER_POINTER`
  / `..._BUFFER_SIZE` / `..._END` array - **not** as a `void**` of per-arg
  pointers. The contents of `kernel.args.data()` is the raw byte sequence the
  GPU sees as `KERNARGS`.
- The launch uses `hipExtModuleLaunchKernel`, which expects **global thread
  count** (`numWorkItems`), not block count, in the X/Y/Z grid args - hence the
  multiplication `rv.numWorkItems.x = rv.workGroupSize.x * rv.numWorkGroups.x`
  at `ContractionSolution.cpp:1491-1493`. (Standard `hipModuleLaunchKernel`
  uses grid count instead.)
- `sharedMemBytes` is **always 0** for these kernels
  (`ContractionSolution.cpp:1495`). All LDS is statically reserved in the
  kernel descriptor; the launch supplies no dynamic shared memory.

Cluster launch (`HIP_HAS_CLUSTER_LAUNCH`, gfx125x) uses `hipDrvLaunchKernelEx`
with a `HIP_LAUNCH_CONFIG` carrying `clusterDim`
(`HipSolutionAdapter.cpp:456-501`). For gfx942/BBS_BH this path is unused.

### 2.3 Lazy-loading dispatch

The lazy path is split between:

- **Loading the index** at startup:
  `SolutionAdapter::initializeLazyLoading(arch, codeObjectDir)`
  (`HipSolutionAdapter.cpp:343-398`). This *does not* load any per-shard `.co`;
  it just loads the helper `Kernels.so-000-<arch>.hsaco` and records
  `m_codeObjectDirectory` + `m_lazyLoadArchitecture` for later. The actual
  `MasterSolutionLibrary` for the lazy index is loaded separately by
  `LoadLibraryFile(...)` at client startup
  (`tensilelite/client/main.cpp:909, 937`).

- **Loading a shard on demand**: when
  `ContractionSolution::generateSingleCall` returns a `KernelInvocation` whose
  `codeObjectFile` is e.g. `TensileLibrary_BB_BB_..._gfx942.co`,
  `SolutionAdapter::launchKernel` calls
  `FindCodeObject(kernel.codeObjectFile)` first
  (`HipSolutionAdapter.cpp:411-414`). `FindCodeObject` checks
  `m_loadedCOFiles`, and if not present iterates xnack variants and calls
  `loadCodeObjectFile(codeObjectDir + modifiedCOName)`
  (`HipSolutionAdapter.cpp:241-268`). The result is added to `m_modules` and
  `m_loadedCOFiles`; future launches needing kernels from the same shard skip
  the load.

The CO filename comes from `PlaceholderLibrary::getCodeObjectFileName()`
(`PlaceholderLibrary.hpp:216-219`) which returns `filePrefix + ".co"`. The
prefix is set when the placeholder is constructed from the lazy index msgpack;
see `MasterSolutionLibrary.hpp:198` for the equivalent stamping at non-lazy
load time.

If the runtime gets `hipErrorLaunchFailure` or `hipErrorNoBinaryForGpu` from a
`hipModuleLoad` (large libraries can exhaust device memory), it unloads every
module, clears the caches, re-runs `initializeLazyLoading`, and retries the
load
(`HipSolutionAdapter.cpp:96-135`).

### 2.4 Minimal-launcher precedents in the repo

There is no "load one `.co` and launch one kernel" sample tucked away
anywhere; everything that runs Tensile kernels goes through
`hip::SolutionAdapter`. The most direct precedents are:

- `tensilelite/client/main.cpp:679-711` (`LoadCodeObjects`): plain loop calling
  `adapter.loadCodeObjectFile(filename)`. Useful as a model for the load step.
- `tensilelite/client/main.cpp:914-937`: bring up `SolutionAdapter`,
  `LoadCodeObjects`, `adapter.initializeLazyLoading(arch, dir)`.
- `tensilelite/client/main.cpp:1127-1186`: the actual benchmark loop that
  builds `kernels = solution->solve(...)` and runs
  `adapter.launchKernels(kernels[0], stream, startEvents, stopEvents)`.
- `library/src/amd_detail/rocblaslt/src/tensile_host.cpp:2718-2950` is the
  hipBLASLt side - it instantiates one `SolutionAdapter` per device, populates
  it via `loadCodeObjectFile` and `initializeLazyLoading`, and stores it in a
  per-device-id atomic for reuse across calls. Same shape as the client, just
  wired into the library handle.

A truly minimal launcher (no Tensile runtime at all) would be ~80 lines:
`hipModuleLoad`, `hipModuleGetFunction`, build the arg byte buffer,
`hipExtModuleLaunchKernel` with `HIP_LAUNCH_PARAM_*`. That's discussed in §4.

**Bottom line for the standalone exe**: a `.co` is loadable by raw HIP -
`hipModuleLoad` accepts the CCOB bundle directly. A solution's name is its ELF
symbol name and `hipModuleGetFunction(module, name)` returns the
`hipFunction_t`. The launch uses `hipExtModuleLaunchKernel` with the
driver-style param array and **global thread count** (not block count). The
runtime's SolutionAdapter is a useful template but not a hard dependency.


## 3. Kernel argument ABI

Two ABIs coexist. Despite the misleading naming, **the example kernel uses the
legacy per-launch byte buffer ABI** - the `UserArgs` token in its name means
the solution *can also* accept a `DeviceUserArguments` struct, but the runtime
only takes that path on the `solveTensileGPU(..., dUA, dUAHost, ...)` entry
(see `ContractionSolution.cpp:2702-2748`), which is gated by the
`--use-user-args` client flag (`main.cpp:1044, 1132-1147`). hipBLASLt's normal
dispatch and `okl.py`-equivalent runs go through plain
`ContractionSolution::solve` -> `generateSingleCall` -> universal-args byte
buffer.

### 3.1 Legacy (per-launch byte buffer)

The host builds a flat byte vector (the `KernelArguments` class,
`tensilelite/include/Tensile/KernelArguments.hpp:135-238`,
`tensilelite/src/KernelArguments.cpp`). Each `append<T>(name, value)` writes
the value into an `std::vector<uint8_t>`, padded per-element so that each
element is aligned to its own size (8-byte pointers must be on 8-byte
boundaries, etc. - `KernelArguments::alignTo`). The final `data()`+`size()`
pair is what the GPU sees as the kernarg segment.

The layout that `generateSingleCall` + `singleCallArgs` + `kernelArgs` produce
for our example (`Cijk_Alik_Bljk_BBS_BH_UserArgs_..._gfx942`, single-call,
non-batched-pointer-array, non-StreamK, non-MBSK), in order:

```
# from kernelArgs() [tensilelite/src/ContractionSolution.cpp:1289-1402], emitted when
# internalArgsSupport.useUniversalArgs is true (true for this solution).
uint32   gemm_count            # = (gemmCount & 0x3FFFFFFF) | (argType << 30); 1, argType=0
uint32   internalArgs          # packed: GSU + GSUC + GSUWGMRR + StaggerU + WGM(v0)
int32    internalArgs1         # WGM(v1+) or WGMXCC packing (version-dependent)
uint32   numWorkGroups         # rv.numWorkGroups.x * .y * .z (collapsed)

# from singleCallArgs() [tensilelite/src/ContractionSolution.cpp:540-1287]
uint32   size_0, size_1, ...   # one per problem.problemSizes() (free + batch + sum)
ptr      d                     # output D tensor base
ptr      c                     # input C tensor base
ptr      a                     # input A tensor base
ptr      b                     # input B tensor base
# (mxsa/mxsb skipped for non-MX kernels; ws/Flags only for SK)
uint32   strideD1, strideD2, ...  # NumIndicesC - (useInitialStridesCD?0:1) entries
uint32   strideC1, strideC2, ...
uint32   strideA1, strideA2, ...
uint32   strideB1, strideB2, ...
<alpha>                        # type-dependent; bf16 alpha -> promoted to float (4 bytes)
<beta>                         # only if problemType.useBeta; same rules
# bias / scale block - all conditional on problemType flags:
ptr      scaleA, scaleB, scaleC, scaleD   # if UseScaleAB/UseScaleCD
ptr      scaleAlphaVec                    # if useScaleAlphaVec
ptr      bias                             # if useBias
uint32   biasType                         # if useBias
uint32   reserved (padding)
ptr      e                                # if useE / Aux
uint32   strideE1, strideE2               # if useE
TAct     act0, act1                       # activation params if Activation != none
int32    activationType                   # if activation gated
```

For the canonical bf16-bf16 in/out, fp32-compute, bias-disabled,
activation-none case the per-launch payload is roughly 130-200 bytes.
You can see exactly what is being passed by setting
`TENSILE_DB=0x20` (`Debug::printKernelArguments`) and re-running - the
adapter dumps every arg with its name to stdout
(`HipSolutionAdapter.cpp:416-422`,
`KernelArguments::operator<<` in `KernelArguments.cpp`).

The kernel-side signature exactly matches this buffer; it is emitted in
`Tensile/Components/Signature.py:118-235` by the `SignatureBase::addArg` calls.
The `Gemm info`/`kernel info0`/`kernel info1`/`numWG` quartet at line 128-131
maps one-to-one to the four `uint32`s `kernelArgs()` emits in C++; the
`SizesFree`/`SizesSum`/`D`/`C`/`A`/`B`/`strideD*`/.../`alpha`/`beta` ordering
matches `singleCallArgs()` field-for-field. That's the contract.

### 3.2 UserArgs variant

The alternate ABI is `DeviceUserArguments<TAct>`, defined in
`tensilelite/include/Tensile/ContractionSolution.hpp:57-92`:

```cpp
template <typename TAct>
struct DeviceUserArguments
{
    uint32_t m;
    uint32_t n;
    uint32_t batch;
    uint32_t k;
    void*    d;
    void*    c;
    void*    a;
    void*    b;
    uint32_t strideD1, strideD2;
    uint32_t strideC1, strideC2;
    uint32_t strideA1, strideA2;
    uint32_t strideB1, strideB2;
    int8_t   alpha[16];   // up to 128-bit alpha
    int8_t   beta[16];
    void*    scaleA, scaleB, scaleC, scaleD;
    void*    scaleAlphaVec;
    void*    bias;
    int      biasType;
    uint32_t reserved;
    void*    e;
    uint32_t strideE1, strideE2;
    TAct     act0, act1;
    int      activationType;
} __attribute__((packed));
```

In this path the user populates one or more such structs in device memory and
the kernel reads them itself; the per-launch kernarg buffer collapses to
`{gemm_count|argType=1 in high bits, internalArgs, internalArgs1, numWG,
DeviceUserArguments*}` (see grouped-gemm path
`ContractionSolution.cpp:1742-1753` and the `2398, 2440` appends of
`"DeviceUserArguments"`). The runtime allocates the struct in pinned
host+device memory via `setDeviceUserArgs()` (`ContractionSolution.cpp:3097`)
then copies to device.

A standalone executable can do **either**:
- (recommended) build the legacy byte buffer in C++ - it's just memcpy'd
  to kernarg memory at launch time.
- (alternate) for `*_UserArgs_*` kernels, allocate one `DeviceUserArguments`
  struct on the GPU, fill it from host, and pass its pointer in the
  much-shorter kernarg buffer. The kernel reads the struct on entry.

The legacy path is simpler to wrap because no extra device alloc is needed and
the ABI is the same shape for every solution; the UserArgs path is what
hipBLASLt itself uses for the user-facing grouped-gemm extension.

### 3.3 Where the per-solution layout is defined

Three sources of truth, all in lock-step:

1. **Logic YAML** under `library/src/amd_detail/rocblaslt/src/Tensile/Logic/`.
   Per-solution `InternalSupportParams` and the `SolutionNameMin` field
   describe what tile/MFMA/etc. the kernel was generated with. Example from
   `aquavanjaram_Cijk_Alik_Bljk_BBS_BH_UserArgs.yaml:127-128`:

   ```yaml
   InternalSupportParams: {SupportCustomStaggerU: true, SupportCustomWGM: true,
                            SupportUserGSU: true, UseUniversalArgs: true}
   ```

   `UseUniversalArgs: true` is what makes
   `internalArgsSupport.useUniversalArgs` true in C++
   (`ContractionSolution.hpp:548`, populated by the msgpack loader). That gates
   the `kernelArgs()` block emitting `gemm_count`/`internalArgs*`/`numWG`
   prior to the user-facing args.

2. **Python signature emitter**: `Tensile/Components/Signature.py:118-235`.
   This is the authoritative declaration of what the *kernel* expects in
   kernarg memory; rocisa later emits the matching `.amdhsa_kernarg_size`,
   `.kernarg_segment_size`, etc. in the ELF metadata.

3. **C++ host-side packer**: `ContractionSolution::generateSingleCall` +
   `singleCallArgs` + `kernelArgs` at
   `tensilelite/src/ContractionSolution.cpp:540-1568`. This is the host code
   that builds the matching byte buffer at runtime.

The per-solution per-arg layout is **not** stored in the lazy index. The host
code knows what to pack based on `problemType` and `sizeMapping` /
`internalArgsSupport` fields it reads from the msgpack solution object. So a
standalone executable that wants to handle arbitrary solutions either:

- links against `libTensile.so` and uses `ContractionSolution` to do the
  packing; or
- replicates the conditional packing logic in C++ (manageable but every new
  flag, e.g. MX block scales, expert scheduling, scaleAB, would need to be
  mirrored); or
- pins itself to one family of solutions and hand-writes the packer for that
  family.

The ELF also carries the AMDGPU HSA metadata for each kernel
(`.amdgpu_metadata` note) with the full kernarg list - useful for verifying
the layout. Dump it with:

```bash
/usr/bin/llvm-readobj --notes /tmp/bb_bb_alik.elf | less   # very large; pipe through grep
# or, more focused, dump just the kernel descriptors:
/usr/bin/llvm-objdump --disassemble-symbols=<kernel_name>.kd /tmp/bb_bb_alik.elf
```

The AMDGPU metadata note has each `.args` entry with `name`/`size`/`offset`/
`value_kind` (e.g. `by_value` for scalars, `global_buffer` for pointers).
That's the ground truth for what the kernel reads from kernarg memory.

### 3.4 Grid / block dimensions

All computed in `ContractionSolution::calculateGrid`
(`tensilelite/src/ContractionSolution.cpp:1405-1442`) from the problem size
and the solution's `sizeMapping`:

```cpp
workGroupSize.x = sizeMapping.workGroupSize.x * sizeMapping.workGroupSize.y
                  * sizeMapping.workGroupSize.z;
workGroupSize.y = 1;  workGroupSize.z = 1;        // always flatten to 1D block

numWorkGroups.x = 1; numWorkGroups.y = 1;
for(... freeIndicesA ...) numWorkGroups.x *= freeSizeA(i);   // typically M
for(... freeIndicesB ...) numWorkGroups.y *= freeSizeB(i);   // typically N
numWorkGroups.z = 1;
for(... batchIndices ...) {
    if(packBatchDims & 0x1) numWorkGroups.x *= batchSize(i);
    if(packBatchDims & 0x2) numWorkGroups.y *= batchSize(i);
    if(!packBatchDims)      numWorkGroups.z *= batchSize(i);
}
if(problem.transposeC01()) std::swap(numWorkGroups.x, numWorkGroups.y);

numWorkGroups.x = CeilDivide(numWorkGroups.x, sizeMapping.macroTile.x);
numWorkGroups.y = CeilDivide(numWorkGroups.y, sizeMapping.macroTile.y);
```

`sizeMapping` is read from the solution metadata in the `.dat`; the relevant
YAML fields are `WorkGroup`, `MacroTile0`/`MacroTile1`, `PackBatchDims` and
similar. Each solution carries its own.

After `calculateGrid`, `generateSingleCall` adjusts for split-K
(`ContractionSolution.cpp:1466-1469`: `numWorkGroups.y *= gsu`) and for the
universal-args version-1 collapse to 1D
(`:1481-1487`: `numWorkGroups.x *= y*z; y=z=1`), then computes
`numWorkItems = workGroupSize * numWorkGroups` componentwise (`:1491-1493`).
`numWorkItems` is what `hipExtModuleLaunchKernel` wants for X/Y/Z grid
arguments.

`sharedMemBytes` is always 0 for these kernels - LDS is reserved in the
kernel descriptor (`.kd` symbol) rather than requested at launch.

For the example solution `MT32x32x128` (macroTile = (32,32)),
WG = e.g. `[32, 8, 1]` from the YAML (flattened to 256), a 4096x4096x4096
GEMM gives:

```
numWorkGroups.x = CeilDivide(4096, 32) = 128
numWorkGroups.y = CeilDivide(4096, 32) = 128
numWorkGroups.z = 1
# after universal-args v1 collapse:
numWorkGroups.x = 128 * 128 * 1 = 16384, y=z=1
# then:
numWorkItems.x = 256 * 16384 = 4194304, y=z=256
```

The grid math is per-solution. To run it for a chosen solution without linking
Tensile, you can either read `WorkGroup` and `MacroTile0/1` from the YAML or
from the AMDGPU metadata note of the kernel (the metadata records the actual
`flat_work_group_size`).

**Bottom line for the standalone exe**: the launch buffer is just a packed
sequence of C scalars and 64-bit pointers in a fixed order. For the legacy
ABI, the order depends on roughly a dozen `problemType`/`sizeMapping` flags
- replicable but boilerplatey. For the UserArgs ABI (this kernel supports
it), the packed device struct is tiny and identical across kernels of the
same family; that is probably the cleaner target for a benchmarking wrapper.
The grid math is one `CeilDivide` per free dimension and you can read the
required `WorkGroup` and `MacroTile` from the solution's logic YAML.


## 4. Implications for the standalone executable

There are three viable architectures, in order of recommended effort:

### Option A (recommended): link `libTensile.so` + `hip::SolutionAdapter`, drop everything else

Build the wrapper exe against the already-built tensilelite host library
(`tensilelite/src/hip/HipSolutionAdapter.cpp` + `ContractionSolution.cpp` +
friends) and do the following at startup:

1. `TensileLite::hip::SolutionAdapter adapter;`
2. `adapter.loadCodeObjectFile("/opt/.../TensileLibrary_BB_BB_..._gfx942.co")`
   - just the one shard you need.
3. Resolve the solution: either load the lazy index via `LoadLibraryFile()` and
   query `MasterSolutionLibrary::getSolutionByIndex(problem, hardware,
   <index_from_okl>)`, **or** skip the library entirely, hand-construct a
   `ContractionSolution` from the YAML (its `sizeMapping`, `problemType`,
   `internalArgsSupport`, `kernelName`, `codeObjectFilename`).
4. Build a `ContractionProblemGemm` from your shape, an `inputs` with the
   device pointers you've allocated yourself.
5. `auto kernels = solution->solve(problem, inputs, hardware, nullptr, 0,
   stream);`
6. Time `adapter.launchKernels(kernels[0], stream, startEvent, stopEvent);`
   in a loop.

This reuses the existing arg-packing and grid-math (so it handles UA / SK /
GSU / MX block scales / bias / activation correctly out of the box) and
costs you essentially nothing beyond `libTensile.so` and its hip deps.
`okl.py` is happy returning you the index; you can either pass that to
`MasterSolutionLibrary::getSolutionByIndex` or use the same name lookup the
`SolutionIterator` in `client/src/SolutionIterator.cpp` does.

Roughly 100-200 lines of C++ for the wrapper. Almost everything that can go
wrong with the ABI is impossible because you are using the same packer the
production library does.

### Option B: minimal raw-HIP launcher for the UserArgs ABI

If you want zero dependency on tensilelite C++:

1. `hipModuleLoad(&mod, "/opt/.../TensileLibrary_BB_BB_..._gfx942.co");`
2. `hipModuleGetFunction(&fn, mod, "<full solution name>");`
3. Allocate one `DeviceUserArguments<float>` (or appropriate `TAct`) on the
   device, populate from host with M/N/K/strides/pointers/alpha/beta/etc.,
   `hipMemcpy` to device.
4. Build a 16-byte kernarg buffer:
   ```
   uint32 gemm_count   = 1 | (1 << 30);   // argType=1 => "args in HBM"
   uint32 internalArgs = 1;                // GSU=1, no staggerU/WGM tweaks
   int32  internalArgs1 = <WGM packed per InternalArgsSupport.version>;
   uint32 numWorkGroups = numWG.x * numWG.y * numWG.z;
   void*  dDeviceUA;
   ```
5. Compute grid from `WorkGroup`/`MacroTile0`/`MacroTile1` in the logic YAML
   (or just hard-code them for the one solution you're shipping).
6. Launch with `hipExtModuleLaunchKernel(fn, numWorkItems.x, numWorkItems.y,
   numWorkItems.z, workGroupSize.x, 1, 1, /*shared=*/0, stream, nullptr,
   hipLaunchParams, nullptr, nullptr);`

This works **only** for `*_UserArgs_*` solutions whose
`InternalSupportParams.UseUniversalArgs == true` (most modern aquavanjaram
solutions). The total binary is ~150 lines of C++ plus the kernel `.co`.

Risk: anything the runtime does that you skip will silently miscompute. The
two most subtle bits are (a) the `internalArgs` packing of GSU + StaggerU +
WGM bits, which is version-dependent (see `kernelArgs()` at
`ContractionSolution.cpp:1289-1402`), and (b) the universal-args v1
post-collapse of `numWorkGroups.y/z` into `x`. Get both wrong and the
kernel will run but produce garbage. The Debug `printKernelArguments` dump
from a baseline `hipblaslt-bench` run is the easiest cross-check.

### Option C: minimal raw-HIP launcher for the legacy byte buffer

Same as B but you build the full ordered argument byte buffer described in
§3.1 instead of allocating a `DeviceUserArguments`. Roughly the same effort
but you handle every per-problem-type variation by hand. Only worth it if
the kernels you care about don't support UserArgs.

### Recommendation

Go with **Option A** if you're benchmarking many solutions or want to be
forward-compatible with new kernels. The Tensile runtime is small enough
(`libTensile.so` is ~6 MB) and already on disk in a hipBLASLt install; you
don't ship a copy. Your wrapper exe becomes a thin shell that just
constructs problems and calls `solution->solve(...)` +
`adapter.launchKernels(...)`. The cuBLAS-comparable bench harness around it
(allocation, fill, validate, time) is the same code you'd write either way.

Go with **Option B** only if the deliverable explicitly needs to be free of
non-vendor dependencies (e.g. shipping a binary that other groups can run
without a hipBLASLt build tree). In that case pick a single
`*_UserArgs_*` solution per problem you care about and freeze its grid +
internalArgs constants.

A useful **cross-check** in both cases: run hipBLASLt's own
`hipblaslt-bench` against the same problem with `--algo_method index
--solution_index <N>` plus `TENSILE_DB=0x20`. The kernarg dump on stdout is
literally the byte buffer your wrapper needs to match.

## 5. Open questions / things I couldn't verify

- **Exact bit layout of `internalArgs`/`internalArgs1` for
  `internalArgsSupport.version == 2`**: I can see the code in
  `ContractionSolution.cpp:1337-1402` and `1370-1395` but did not exhaustively
  walk through every conditional for `WGMXCC`/`WGMXCCG`/`WGMXCCCHUNK` packing.
  For a single solution you should dump the buffer from a known-good
  `hipblaslt-bench` invocation rather than re-deriving.
- **Whether `hipModuleLoad` accepts a CCOB-bundle pointed at the wrong gfx**:
  the bundles in `/opt/rocm/lib/hipblaslt/library` are single-target
  (host stub + one gfx ELF). I did not test what happens if you load a
  gfx950 bundle on a gfx942 device, but expect `hipErrorNoBinaryForGpu`.
- **Sufficient `KernelArguments` reservation**: `generateSingleCall` calls
  `rv.args.reserve(1024, 128)` (`:1458`). I did not find a case where the
  emitted buffer exceeds 1 KB, but for very wide problem types (MX block
  scales + bias + activation + StreamK) this could in principle happen.
- **Whether the runtime ever launches more than one kernel for a `_BBS_BH_`
  non-MBSK / non-StreamK / non-GSU solution**: I traced the `solve` ->
  `generateSingleCall` -> `return {rv}` path and it produces exactly one
  `KernelInvocation`. Multi-kernel paths exist (output conversion at
  `:1903`, beta-only at `:1925`, reduction at `:2231`, GSU at `:2401/2443`)
  but are conditional on `globalAccumulation`/`useBeta`/`gradient`/`gsu` and
  the relevant `globalAccumulation`/`adaptiveGemmGSUA` modes. Confirm with a
  size of the kernels vector returned at runtime before assuming a single
  launch is sufficient.
- **Exact AMDGPU-HSA `.args` metadata for a chosen kernel**: I confirmed the
  symbol and `.kd` exist (§1.3) but did not dump the per-arg
  `value_kind`/`offset` for the example. `llvm-readobj --notes <kernel.elf>`
  gives this; piping through `grep -A2 -E "name:|value_kind:|offset:"`
  produces the kernel-side ground truth for the host packer.
- **Whether `clang-offload-bundler --unbundle` accepts xnack-qualified
  targets** (e.g. `hipv4-amdgcn-amd-amdhsa-unknown-gfx942:xnack-`): the
  shipped CO files I inspected were plain `gfx942`, not `gfx942:xnack-`. If
  you encounter xnack variants in the wild, the bundler target string will
  need adjustment - see `removeXnack()` and the `xnack` iteration in
  `HipSolutionAdapter.cpp:79-87, 252-264`.

## 6. Worked example: 512^3 bf16 TN, solution 45732 (ROCm 6.4.3)

This is the result of actually running the dump workflow recommended in §4
against a real install, to lock in concrete byte values for one solution.

### 6.1 Capturing the launch with `TENSILE_DB=0x40`

**Important correction to §4:** the bit for `printKernelArguments` is
`0x40`, not `0x20`. `0x20` is `printCodeObjectInfo`. See
`tensilelite/src/Debug.cpp:65-70`. Use `0xF0` (or `0xC0` for kernarg + tensor
only) to also get code-object load lines and tensor descriptors.

```bash
HIPBLASLT_TENSILE_LIBPATH=/opt/rocm-6.4.3/lib/hipblaslt/library \
TENSILE_DB=0xF0 \
/opt/rocm-6.4.3/bin/hipblaslt-bench \
  -m 512 -n 512 -k 512 --transA T --transB N \
  --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r \
  --compute_type f32_r --algo_method heuristic --requested_solution 1 \
  --print_kernel_info --iters 1 --cold_iters 0
```

The dump (relevant lines):

```
load placeholder library /opt/rocm-6.4.3/lib/hipblaslt/library//TensileLibrary_BB_BB_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx942.dat
loaded code object /opt/rocm-6.4.3/lib/hipblaslt/library/TensileLibrary_BB_BB_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx942.co
Kernel Cijk_Alik_Bljk_BBS_BH_UserArgs_MT32x32x128_MI16x16x1_..._WS64_WG32_8_1
 l(256, 1, 1) x g(256, 1, 1) = (65536, 1, 1)
[0..3]    gemm_count:    01 00 00 00  (1)
[4..7]    internalArgs:  01 00 08 20  (0x20080001)
[8..11]   internalArgs1: 00 00 01 4c  (0x4c010000)
[12..15]  numWorkGroups: 00 01 00 00  (256)
[16..19]  size_0:        00 02 00 00  (512)         # M
[20..23]  size_1:        00 02 00 00  (512)         # N
[24..27]  size_2:        01 00 00 00  (1)           # batch
[28..31]  size_3:        00 02 00 00  (512)         # K
[32..39]  d:             ... 8-byte device pointer
[40..47]  c:             ... 8-byte device pointer
[48..55]  a:             ... 8-byte device pointer
[56..63]  b:             ... 8-byte device pointer
[64..67]  strideD1:      00 02 00 00  (512)         # LD of D
[68..71]  strideD2:      00 00 00 00  (0)           # batch stride
[72..75]  strideC1:      00 02 00 00  (512)
[76..79]  strideC2:      00 00 00 00  (0)
[80..83]  strideA1:      00 02 00 00  (512)
[84..87]  strideA2:      00 00 00 00  (0)
[88..91]  strideB1:      00 02 00 00  (512)
[92..95]  strideB2:      00 00 00 00  (0)
[96..99]  alpha:         00 00 80 3f  (float 1.0)
[100..103] beta:         00 00 00 00  (float 0.0)
```

Total kernarg buffer: **104 bytes**. Grid: **256x1x1 workgroups** of
**256 threads** each (65536 work-items). The host runtime prints workgroup
size as `(256, 1, 1)`, grid as `(256, 1, 1)` (in workgroups), and the
"`(65536, 1, 1)`" product is what gets passed to `hipExtModuleLaunchKernel`
as `globalWorkSizeX`.

### 6.2 Discovery: this is NOT the DeviceUserArguments path

Despite the solution name containing `UserArgs`, the runtime launched this
GEMM via the **legacy in-kernarg byte buffer**, not via a
`DeviceUserArguments` struct in device memory. This is visible from two
signals:

1. The kernarg buffer is 104 bytes, not 24 bytes.
2. The high bits of `gemm_count` are zero (`0x00000001`); for the UserArgs
   path the runtime ORs in `(1 << 30)` (`argType=1`, "args are in HBM") -
   see `ContractionSolution.cpp:1742-1753`.

The `_UserArgs_` in the solution name means **the solution supports
UserArgs**, not that the runtime always launches it that way. For a single
non-grouped GEMM, the runtime takes the simpler in-kernarg path. For
grouped-gemm with N>1 problems, or when called through hipBLASLt's
grouped-gemm extension, the same kernel would be launched via the
DeviceUserArguments path with the 24-byte kernarg.

**This simplifies Option B substantially**: for non-grouped GEMM we never
need to populate a `DeviceUserArguments` struct, just a 104-byte kernarg
buffer with the fields above.

### 6.3 Kernel-side ABI verification (HSA metadata)

Cross-check from the kernel's own ELF metadata. Tool path on this box:
`clang-offload-bundler` ships under `/opt/rocm-7.2.1/lib/llvm/bin/` not
`/opt/rocm-6.4.3/llvm/bin/` (which doesn't exist as a tree); use whichever
is on disk. `llvm-readobj` from `/usr/bin/` works fine.

```bash
BUNDLER=/opt/rocm-7.2.1/lib/llvm/bin/clang-offload-bundler
CO=/opt/rocm-6.4.3/lib/hipblaslt/library/TensileLibrary_BB_BB_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx942.co
$BUNDLER --list --type=o --input=$CO
# Output:
#   hipv4-amdgcn-amd-amdhsa--gfx942
#   host-x86_64-unknown-linux-gnu-

$BUNDLER --unbundle --type=o --input=$CO \
  --targets=hipv4-amdgcn-amd-amdhsa--gfx942,host-x86_64-unknown-linux-gnu- \
  --output=/tmp/kernel.elf --output=/tmp/host.o

llvm-readobj --notes /tmp/kernel.elf | grep -A4 -E "name:|kernarg_segment_size|group_segment_fixed_size|max_flat_workgroup_size"
```

The `amdhsa.kernels` note for our kernel reports:

```yaml
.args:
  - {name: "Gemm info",      offset: 0,   size: 4, value_kind: by_value, value_type: u32}
  - {name: "kernel info0",   offset: 4,   size: 4, value_kind: by_value, value_type: u32}
  - {name: "kernel info1",   offset: 8,   size: 4, value_kind: by_value, value_type: u32}
  - {name: "numWG",          offset: 12,  size: 4, value_kind: by_value, value_type: u32}
  - {name: "SizesFree0",     offset: 16,  size: 4, value_kind: by_value, value_type: u32}
  - {name: "SizesFree1",     offset: 20,  size: 4, value_kind: by_value, value_type: u32}
  - {name: "SizesFree2",     offset: 24,  size: 4, value_kind: by_value, value_type: u32}
  - {name: "SizesSum0",      offset: 28,  size: 4, value_kind: by_value, value_type: u32}
  - {name: "D", offset: 32,  size: 8, value_kind: global_buffer, address_space: generic}
  - {name: "C", offset: 40,  size: 8, value_kind: global_buffer, address_space: generic}
  - {name: "A", offset: 48,  size: 8, value_kind: global_buffer, address_space: generic}
  - {name: "B", offset: 56,  size: 8, value_kind: global_buffer, address_space: generic}
  - {name: "strideD0",       offset: 64,  size: 4, value_kind: by_value, value_type: u32}
  - {name: "strideD1",       offset: 68,  size: 4, value_kind: by_value, value_type: u32}
  - {name: "strideC0",       offset: 72,  size: 4, value_kind: by_value, value_type: u32}
  - {name: "strideC1",       offset: 76,  size: 4, value_kind: by_value, value_type: u32}
  - {name: "strideA0",       offset: 80,  size: 4, value_kind: by_value, value_type: u32}
  - {name: "strideA1",       offset: 84,  size: 4, value_kind: by_value, value_type: u32}
  - {name: "strideB0",       offset: 88,  size: 4, value_kind: by_value, value_type: u32}
  - {name: "strideB1",       offset: 92,  size: 4, value_kind: by_value, value_type: u32}
  - {name: "alpha",          offset: 96,  size: 4, value_kind: by_value}
  - {name: "beta",           offset: 100, size: 4, value_kind: by_value}
.group_segment_fixed_size:   51200    # LDS bytes (kernel-side allocation)
.kernarg_segment_size:       104      # matches runtime dump exactly
.max_flat_workgroup_size:    256
.private_segment_fixed_size: 0
```

The host runtime dump and the kernel-side metadata agree byte-for-byte.

Also note `custom.config.InternalSupportParams.KernArgsVersion: 2` in the
metadata - this is what `ContractionSolution.cpp:1289-1402` uses to
select the `internalArgs`/`internalArgs1` bit-packing version.

### 6.4 Decoding `internalArgs` (0x20080001) and `internalArgs1` (0x4c010000)

These are the bit-packed fields whose layout is version-dependent. From
`ContractionSolution.cpp:1289-1402` with `KernArgsVersion=2`:

`internalArgs` (0x20080001, little-endian -> bytes `01 00 08 20`):
- low 16 bits: GSU value = **0x0001 = 1** (no global split-K)
- bits 16-23: StaggerU = **0x08 = 8**
- bits 24-31: WGM type / WGMXCCG packing (need to verify against the
  KernArgsVersion=2 code path)

`internalArgs1` (0x4c010000, little-endian bytes `00 00 01 4c`):
- WGMXCC + WGMXCCG + WGMXCCCHUNK packed; for this single-CU workgroup
  mapping (`WGM0_WGMXCC1_WGMXCCGn1` in the solution name) the value is
  what the runtime produced and what we should re-use verbatim.

**For a packaged single-kernel exe, do not re-derive these.** Use the
literal byte values from the dump - they encode all of (GSU, StaggerU, WGM,
WGMXCC, WGMXCCG, WGMXCCCHUNK) correctly for the chosen kernel.

### 6.5 Mapping problem parameters into the buffer

The constants vs the problem-dependent fields:

| Offset | Field            | Source                                              |
|--------|------------------|-----------------------------------------------------|
| 0      | gemm_count       | constant `1` for single GEMM                        |
| 4      | internalArgs     | constant from dump (kernel-specific, problem-indep) |
| 8      | internalArgs1    | constant from dump (kernel-specific, problem-indep) |
| 12     | numWG            | ceildiv(M, MT0) * ceildiv(N, MT1) * batch           |
| 16-31  | sizes M/N/B/K    | from problem                                        |
| 32-63  | D/C/A/B pointers | from hipMalloc                                      |
| 64-95  | strides          | from problem (LD; batch-stride 0 if batch=1)        |
| 96     | alpha            | from problem (4 bytes, f32 here)                    |
| 100    | beta             | from problem (4 bytes, f32 here)                    |

For this kernel: `MT0=32, MT1=32` (from the `_MT32x32x128_` token in the
solution name), so `numWG = ceildiv(M,32) * ceildiv(N,32) * batch`. For
512x512x1 this is 16*16*1 = 256, matching the dump.

The launch dims are: `workGroupSize = (256, 1, 1)` (from `_WS64_WG32_8_1`:
WS=64 lanes/wave, WG=32x8x1 waves -> 256 threads, but the launch parameter
is the *workgroup size in threads* = 256), `gridSizeWorkgroups = (numWG, 1,
1)`, `globalSize_x = workGroupSize * gridSizeWorkgroups = 256 * numWG`.

For `hipExtModuleLaunchKernel` you pass **global thread count**, so for
512^3 you pass `globalSize_x = 65536`.

## 7. Standalone wrapper skeleton (Option B realized)

What follows is the actual ~180 LOC C++ for a standalone benchmark exe
that loads the .co, launches the kernel on synthetic data, validates a
small slice on host, and times it. Compile with just `hipcc` + the HIP
runtime - **no link dependency on libTensile.so, libhipblaslt.so, or
anything else from a hipBLASLt install**.

### 7.1 Per-solution constants (the only kernel-specific bits)

These come from §6.1 and §6.4. A real per-solution generator (the wrapper
around `okl.py`) would emit a small header per solution with these
values + the `.co` path + the kernel symbol. For now, hand-paste:

```cpp
// === BEGIN solution-specific constants (one of these per packaged kernel) ===
static constexpr const char* SOLUTION_CO_FILE =
    "/opt/rocm-6.4.3/lib/hipblaslt/library/"
    "TensileLibrary_BB_BB_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx942.co";

static constexpr const char* SOLUTION_KERNEL =
    "Cijk_Alik_Bljk_BBS_BH_UserArgs_MT32x32x128_MI16x16x1_SN_LDSB0_AFC0_"
    "AFEM8_AFEM8_ASEM32_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_EPS0_FDSI0_"
    "GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA942_IU1_K1_LBSPPA256_LBSPPB256_"
    "LBSPPM0_LPA16_LPB16_LPMn1_LRVW8_LWPMn1_MIAV0_MIWT1_1_MO40_NTn1_NTA0_"
    "NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB1_ONLL1_PGR2_PLR1_PKA1_SIA3_SS1_"
    "SPO0_SRVW0_SSO0_SVW1_SK0_SKXCCM0_TLDS1_ULSGRO0_USL1_UIOFGRO0_USFGROn1_"
    "VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS64_WG32_8_1";

static constexpr uint32_t INTERNAL_ARGS  = 0x20080001;  // dump byte 4..7
static constexpr uint32_t INTERNAL_ARGS1 = 0x4c010000;  // dump byte 8..11
static constexpr uint32_t MACRO_TILE_0   = 32;          // from MT32x32x128
static constexpr uint32_t MACRO_TILE_1   = 32;
static constexpr uint32_t WORKGROUP_SIZE = 256;         // WS64*WG32*8*1 / WS64 = 256 threads
// === END solution-specific constants ===
```

### 7.2 Full standalone exe

```cpp
// okl_run.cpp - standalone benchmark for one packaged hipBLASLt kernel.
// Compile: hipcc -O3 -std=c++17 okl_run.cpp -o okl_run
// Run:     ./okl_run

#include <hip/hip_runtime.h>
#include <hip/hip_ext.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>
#include <cstdlib>

#define HIP_CHECK(c) do { hipError_t e=(c); if(e){ \
    fprintf(stderr,"HIP error %d at %s:%d: %s\n",e,__FILE__,__LINE__, \
            hipGetErrorString(e)); std::exit(1);} } while(0)

// === solution-specific constants (see 7.1) ===
static constexpr const char* SOLUTION_CO_FILE = "...";
static constexpr const char* SOLUTION_KERNEL  = "...";
static constexpr uint32_t INTERNAL_ARGS  = 0x20080001;
static constexpr uint32_t INTERNAL_ARGS1 = 0x4c010000;
static constexpr uint32_t MACRO_TILE_0   = 32;
static constexpr uint32_t MACRO_TILE_1   = 32;
static constexpr uint32_t WORKGROUP_SIZE = 256;

// Problem: 512^3 bf16 TN, alpha=1, beta=0
// (a per-solution generator would template these too, or read from argv)
static constexpr uint32_t M = 512, N = 512, K = 512, BATCH = 1;
static constexpr uint32_t LDA = K, LDB = K, LDC = M, LDD = M;  // T(A): KxM, N(B): KxN, D/C: MxN
static constexpr float    ALPHA = 1.0f, BETA = 0.0f;

// Element size for bf16
static constexpr size_t BF16_BYTES = 2;

// Pack a uint16 bf16 value (round-toward-zero, no NaN handling here)
static uint16_t fp32_to_bf16(float v) {
    uint32_t u; std::memcpy(&u, &v, 4);
    return uint16_t(u >> 16);
}

int main() {
    // ---- 1. Allocate device buffers ----
    void *dA = nullptr, *dB = nullptr, *dC = nullptr, *dD = nullptr;
    size_t bytesA = size_t(K) * M * BF16_BYTES;  // T: KxM
    size_t bytesB = size_t(K) * N * BF16_BYTES;  // N: KxN
    size_t bytesC = size_t(M) * N * BF16_BYTES;
    size_t bytesD = size_t(M) * N * BF16_BYTES;
    HIP_CHECK(hipMalloc(&dA, bytesA));
    HIP_CHECK(hipMalloc(&dB, bytesB));
    HIP_CHECK(hipMalloc(&dC, bytesC));
    HIP_CHECK(hipMalloc(&dD, bytesD));

    // ---- 2. Fill A, B with a deterministic pattern (host -> device) ----
    std::vector<uint16_t> hostA(K * M), hostB(K * N);
    for (size_t i = 0; i < hostA.size(); ++i) hostA[i] = fp32_to_bf16(0.01f * float(i % 1024));
    for (size_t i = 0; i < hostB.size(); ++i) hostB[i] = fp32_to_bf16(0.02f * float(i % 1024));
    HIP_CHECK(hipMemcpy(dA, hostA.data(), bytesA, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hostB.data(), bytesB, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemset(dC, 0, bytesC));
    HIP_CHECK(hipMemset(dD, 0, bytesD));

    // ---- 3. Load the code object and look up the kernel ----
    hipModule_t module;
    HIP_CHECK(hipModuleLoad(&module, SOLUTION_CO_FILE));
    hipFunction_t kernel;
    HIP_CHECK(hipModuleGetFunction(&kernel, module, SOLUTION_KERNEL));

    // ---- 4. Build the 104-byte kernarg buffer (see 6.5) ----
    alignas(8) uint8_t kernarg[104];
    auto put_u32 = [&](size_t off, uint32_t v) { std::memcpy(kernarg + off, &v, 4); };
    auto put_ptr = [&](size_t off, void* p)    { std::memcpy(kernarg + off, &p, 8); };
    auto put_f32 = [&](size_t off, float v)    { std::memcpy(kernarg + off, &v, 4); };

    uint32_t numWG = ((M + MACRO_TILE_0 - 1) / MACRO_TILE_0) *
                     ((N + MACRO_TILE_1 - 1) / MACRO_TILE_1) * BATCH;

    put_u32(0,   1);                    // gemm_count (argType=0 in high bits)
    put_u32(4,   INTERNAL_ARGS);
    put_u32(8,   INTERNAL_ARGS1);
    put_u32(12,  numWG);
    put_u32(16,  M);
    put_u32(20,  N);
    put_u32(24,  BATCH);
    put_u32(28,  K);
    put_ptr(32,  dD);
    put_ptr(40,  dC);
    put_ptr(48,  dA);
    put_ptr(56,  dB);
    put_u32(64,  LDD); put_u32(68,  0);  // strideD0, batch-stride D
    put_u32(72,  LDC); put_u32(76,  0);  // strideC0, batch-stride C
    put_u32(80,  LDA); put_u32(84,  0);  // strideA0, batch-stride A
    put_u32(88,  LDB); put_u32(92,  0);  // strideB0, batch-stride B
    put_f32(96,  ALPHA);
    put_f32(100, BETA);

    // ---- 5. Driver-style launch ----
    size_t kernarg_size = sizeof(kernarg);
    void* launch_params[] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, kernarg,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,    &kernarg_size,
        HIP_LAUNCH_PARAM_END
    };

    // Grid: hipExtModuleLaunchKernel takes GLOBAL thread count (not workgroups)
    uint32_t globalX = numWG * WORKGROUP_SIZE;

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    // Warmup
    HIP_CHECK(hipExtModuleLaunchKernel(
        kernel, globalX, 1, 1,
        WORKGROUP_SIZE, 1, 1,
        /*sharedMemBytes=*/0, /*stream=*/nullptr,
        nullptr, launch_params, nullptr, nullptr));
    HIP_CHECK(hipDeviceSynchronize());

    // Timed loop
    constexpr int ITERS = 100;
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < ITERS; ++i) {
        HIP_CHECK(hipExtModuleLaunchKernel(
            kernel, globalX, 1, 1,
            WORKGROUP_SIZE, 1, 1,
            0, nullptr, nullptr, launch_params, nullptr, nullptr));
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float ms = 0;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    double sec_per_iter = double(ms) * 1e-3 / ITERS;
    double flops = 2.0 * double(M) * N * K * BATCH;
    double gflops = flops / sec_per_iter * 1e-9;

    printf("kernel:   %s\n", SOLUTION_KERNEL);
    printf("problem:  M=%u N=%u K=%u batch=%u  TN  bf16\n", M, N, K, BATCH);
    printf("iters:    %d\n", ITERS);
    printf("time:     %.3f us / iter\n", sec_per_iter * 1e6);
    printf("perf:     %.1f gflops\n", gflops);

    // ---- 6. Spot-check correctness (cheap: one output element) ----
    std::vector<uint16_t> hostD(M * N);
    HIP_CHECK(hipMemcpy(hostD.data(), dD, bytesD, hipMemcpyDeviceToHost));
    // D[0,0] = sum_k A[k,0] * B[k,0]   (TN: A is KxM, B is KxN)
    double ref = 0;
    for (uint32_t k = 0; k < K; ++k) {
        float a, b;
        uint32_t ua = uint32_t(hostA[k + 0 * K]) << 16;
        uint32_t ub = uint32_t(hostB[k + 0 * K]) << 16;
        std::memcpy(&a, &ua, 4); std::memcpy(&b, &ub, 4);
        ref += double(a) * double(b);
    }
    uint32_t got_u = uint32_t(hostD[0]) << 16;
    float got; std::memcpy(&got, &got_u, 4);
    printf("D[0,0]:   got=%g  ref=%g  rel_err=%.3e\n",
           got, ref, std::abs(got - ref) / std::abs(ref + 1e-12));

    HIP_CHECK(hipFree(dA)); HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC)); HIP_CHECK(hipFree(dD));
    HIP_CHECK(hipModuleUnload(module));
    return 0;
}
```

### 7.3 Build and run

```bash
hipcc -O3 -std=c++17 okl_run.cpp -o okl_run
./okl_run
```

Expected output shape (numbers depend on hardware):

```
kernel:   Cijk_Alik_Bljk_BBS_BH_UserArgs_MT32x32x128_MI16x16x1_...
problem:  M=512 N=512 K=512 batch=1  TN  bf16
iters:    100
time:     8.7 us / iter
perf:     30854.7 gflops
D[0,0]:   got=... ref=... rel_err=...
```

The kernel time should match the per-iter time hipblaslt-bench reports for
the same problem (8.7 us in our earlier okl.py run), confirming the
wrapper drives the kernel identically.

### 7.4 What a per-solution generator needs to produce

To make the wrapper repeatable for arbitrary `(solution_index, problem)`
pairs, build a small generator (Python or shell) that:

1. Runs `hipblaslt-bench ... --algo_method index --solution_index N
   ... TENSILE_DB=0x40` and parses the kernarg dump.
2. Extracts: `.co` path (from the `loaded code object` line), kernel
   symbol name, `internalArgs`, `internalArgs1`, MT0/MT1 (from the
   `_MT<a>x<b>x<c>_` token in the kernel name), `WORKGROUP_SIZE` (from
   `_WS<wave>_WG<x>_<y>_<z>` -> wave * x * y * z).
3. Emits a small C++ header (`solution_<index>.hpp`) containing the
   `SOLUTION_CO_FILE` / `SOLUTION_KERNEL` / `INTERNAL_ARGS` / ... constants.
4. The `main()` reads M/N/K/strides/dtypes from `argv` (or via macros set
   at compile-time per packaged binary) and otherwise reuses the same
   skeleton above.

The dtype-dependent bits (BF16_BYTES, fp32_to_bf16, alpha/beta encoding
size at offsets 96/100) parametrize on (a_type, b_type, c_type, d_type,
compute_type). For all common GEMM dtype combos these are:

| Tensor dtype | element bytes | alpha/beta bytes (when compute=f32) |
|--------------|---------------|-------------------------------------|
| f16, bf16    | 2             | 4 (f32)                              |
| f32          | 4             | 4 (f32)                              |
| f64          | 8             | 8 (f64) - alpha/beta widen           |
| f8, bf8      | 1             | 4 (f32)                              |
| int8         | 1             | 4 (i32) - alpha/beta change kind     |

The kernel's HSA `.args` metadata reports the per-arg sizes; if your
generator reads it (it's a 60-line awk over `llvm-readobj --notes`
output), the buffer layout becomes fully data-driven.

### 7.5 Limits of this approach

- **One kernel per binary.** This is a deliberate choice for a benchmark
  shootout (apples-to-apples; the binary is the unit of comparison). If you
  want one exe that can switch kernels at runtime, ship many `.co` files
  alongside many solution headers; the launcher becomes a small dispatch
  table.
- **Frozen Tensile version.** The 104-byte layout above is stable as long
  as `KernArgsVersion == 2` and the kernel's HSA metadata doesn't change.
  Cross-version testing should re-dump the kernarg via §6.1 and recompare.
- **No batched-with-different-strides, no StreamK, no GSU>1, no MX block
  scales.** All of these change the kernarg layout (extra fields after the
  basic ones, conditional on the kernel's feature flags). Each adds at
  most a handful of fields; the same `llvm-readobj --notes` driven generator
  in 7.4 handles them.
- **No bias, no activation.** Same comment.
- **xnack variants.** The shipped CO files I unbundled were plain `gfx942`.
  Builds with xnack+/xnack- ABI will have a different bundle target string;
  use `clang-offload-bundler --list` to discover at packaging time.


## 8. Kernel loading interface

This section covers the slice of okl_run.cpp that turns a (config-file,
.co-on-disk) pair into a `hipFunction_t` ready to launch. The kernarg buffer
construction and launch loop are out of scope here.

### 8.1 What the runtime gives us, and what it does not

`hipModuleLoad` is a one-shot: it accepts a path, opens the file, parses the
`__CLANG_OFFLOAD_BUNDLE__` wrapper (when present), picks the slice matching
the current device, and loads it. Failure modes that we hit in practice:

- **wrong arch in the bundle**: `hipModuleLoad` returns `hipErrorNoBinaryForGpu`
  (216). The runtime cannot tell you which slices the bundle DOES have.
- **file missing**: returns `hipErrorFileNotFound` (301). Path is not echoed in
  the error string.
- **path is a junk file**: returns `hipErrorInvalidImage` or similar.

The Tensile runtime works around the arch problem by trying each xnack variant
in turn — see `tensilelite/src/hip/HipSolutionAdapter.cpp:256-263`:

```cpp
for(auto ver : {"", "-xnack-", "-xnack+"})
{
    std::string modifiedCOName = codeObjectFile;
    modifiedCOName.insert(loc, ver);
    err = loadCodeObjectFile(codeObjectDir + modifiedCOName);
    if(err == hipSuccess) break;
}
```

i.e. it brute-forces three filenames (`...co`, `...-xnack-.co`, `...-xnack+.co`)
and stops on the first that loads. The arch base (`gfx942`) is already baked
into the filename when the lazy-loader hands it off (see `FindCodeObject` at
`HipSolutionAdapter.cpp:241-268` and the caller chain from `launchKernel` at
`:411-414`).

`hipModuleGetFunction` is a string lookup: pass a name, get a function or
`hipErrorNotFound`. The HIP API surface in `/opt/rocm-7.2.1/include/hip/hip_runtime_api.h`
exposes `hipModuleGetFunctionCount` (line 6362) but NO enumeration call — there
is no `hipModuleGetFunctionNames` or equivalent. The only programmatic way to
list the symbols in a module is to inspect the underlying ELF directly via
`llvm-readobj` / HSA, BEFORE loading.

`hipFuncGetAttribute` (line 6474) is useful as a confirmation probe: after a
successful `hipModuleGetFunction` it returns register count, LDS size, etc.,
sourced from the kernel descriptor (`.kd`) that lives next to the text symbol.
A `hipSuccess` here means the loader actually wired up both the code and the
descriptor; useful sanity for our standalone path that bypasses Tensile.

### 8.2 What's inside a .co (the bundle metadata)

Every Tensile shard is a `clang-offload-bundler` wrapper around one or more
ELF objects, one per target. `clang-offload-bundler --list` reads the bundle
header without unpacking anything. Real example:

```
$ /opt/rocm-7.2.1/lib/llvm/bin/clang-offload-bundler --list --type=o \
    --input=/opt/rocm-6.4.3/lib/hipblaslt/library/TensileLibrary_BB_BB_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx942.co
hipv4-amdgcn-amd-amdhsa--gfx942
host-x86_64-unknown-linux-gnu-
```

Tuple format is `<kind>-<triple>--<arch>[:feature][:feature]...`. For our
purposes the trailing field is the only one that varies between shards and
the only one the HIP runtime cares about for arch matching.

After unbundling the device slice with
`--unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx942 --output=dev.o`,
the resulting ELF carries the full kernel ABI in an `NT_AMDGPU_METADATA` note:

```
$ llvm-readobj --notes dev.o | head -50
... amdhsa.kernels:
  - .args:
      - .name:    Gemm info     .offset: 0   .size: 4   .value_kind: by_value
      - .name:    kernel info0  .offset: 4   .size: 4   ...
      ...
    .symbol:    Cijk_..._MT32x16x512_..._WG32_4_1.kd
    .name:      Cijk_..._MT32x16x512_..._WG32_4_1
```

`.name` is what `hipModuleGetFunction` resolves. `.symbol` is the kernel
descriptor name (the same string + `.kd`). For each kernel in the shard there
is one entry. `grep '\.symbol:'` is a cheap way to enumerate.

### 8.3 The principled loading interface (this commit)

The pre-existing `okl_run.cpp` just did:

```cpp
HIP_CHECK(hipModuleLoad(&module, co_path.c_str()));
HIP_CHECK(hipModuleGetFunction(&kernel, module, c.kernel_symbol.c_str()));
```

That deferred everything to the runtime. Failures bubbled up as bare HIP
error strings ("no binary for GPU") with no context about what was expected.

The new interface in `okl_run.cpp` is three small named pieces:

1. `parse_arch_tuple(s)` — splits a `gcnArchName`-style string
   (e.g. `gfx942:xnack-`) into `{base, xnack}`. Pure function.

2. `device_arch_tuple()` — wraps `hipGetDevice` + `hipGetDeviceProperties` and
   parses `gcnArchName`. Returns `ArchTuple` for the currently selected device.

3. `check_arch_compatible(conf, device)` — compares conf's expected
   `target_arch` (e.g. `gfx942`) and `xnack` (`any`/`on`/`off`) against the
   device tuple. Exits with a precise message naming both sides on mismatch.
   Runs BEFORE `hipModuleLoad`.

4. `load_kernel(co_path, kernel_symbol)` — returns
   `{hipModule_t, hipFunction_t, num_regs, lds_bytes}`. Steps:
   - `std::filesystem::exists(co_path)` precheck; clean message if missing.
   - `hipModuleLoad`; on `hipErrorNoBinaryForGpu`, emit a message pointing the
     user at `clang-offload-bundler --list` to see what arches ARE in the bundle.
   - `hipModuleGetFunction`; on `hipErrorNotFound`, emit a message with the
     symbol asked for, the .co path, and the exact `clang-offload-bundler` +
     `llvm-readobj` recipe to enumerate what IS available. We cannot enumerate
     from HIP, so we punt to the user with the right tools.
   - `hipFuncGetAttribute(NUM_REGS|SHARED_SIZE_BYTES)` as a confirmation probe.
     Recorded for diagnostics; failures here are non-fatal because some HIP
     builds report a subset.

The choices behind this shape:

- **Single (arch, xnack) per conf, not a list.** Conf is generated from one
  bench dump, which always picked one shard. A multi-arch package would be a
  parallel directory of confs, not one fat conf. KISS.
- **`target_arch` is required.** Without it we'd be reproducing the silent
  `hipErrorNoBinaryForGpu` mystery. Set the field to an empty string in the
  conf to opt out explicitly.
- **`xnack=any` default.** The shipped ROCm 6.4.3 bundle has plain `gfx942`
  shards (no xnack feature suffix in the bundle target), so demanding a
  specific xnack at runtime would reject every shipped kernel. `any` is the
  only sane default; the field exists so xnack-variant packages can lock down.
- **`bundle_target` is diagnostic only.** Verbatim string from
  `clang-offload-bundler --list`, echoed in the runner's preamble for human
  inspection. We don't string-match on it because the format is brittle (kind
  prefix `hipv4-` may shift over ROCm versions).
- **No introspection step on load.** I considered shelling out to
  `llvm-readobj --notes` on each load to verify the symbol up front, but the
  cost (fork + ELF parse of a multi-MB file) outweighed the benefit (an error
  message a few μs earlier). The `hipFuncGetAttribute` probe achieves the same
  "symbol really there with a descriptor" assurance for free.

### 8.4 Conf-field additions

| key | required | source | what it means |
|---|---|---|---|
| `target_arch` | yes | `okl.py probe_bundle()` → bundle target's trailing field | gfxNNN slug; runner rejects load if `hipDeviceProp.gcnArchName` base doesn't match |
| `xnack` | no (default `any`) | bundle target's `:xnack+` / `:xnack-` suffix | `any` skips xnack check; `on`/`off` must match device |
| `bundle_target` | no | bundle target verbatim | diagnostic only; printed in runner preamble |

Population path in `okl.py`:

```python
def probe_bundle(co_path):
    # runs clang-offload-bundler --list, picks the amdgcn target,
    # returns (target_arch, xnack_mode, full_target_string)
```

If `clang-offload-bundler` is not found, fields are written as blank and the
runner skips the check (with a warning at packaging time).

### 8.5 Before / after

Before (the legacy minimal conf, all loading-related):

```
co_file       = kernel.co
kernel_symbol = Cijk_..._MT32x32x128_..._WG32_8_1
```

After:

```
co_file       = kernel.co
kernel_symbol = Cijk_..._MT32x32x128_..._WG32_8_1
target_arch   = gfx942
xnack         = any
bundle_target = hipv4-amdgcn-amd-amdhsa--gfx942
```

And the runner's preamble now includes:

```
target:    arch=gfx942 xnack=any  device=gfx942:xnack-  bundle=hipv4-amdgcn-amd-amdhsa--gfx942
resources: regs=256 lds=51200 bytes
```

### 8.6 Error-path coverage (verified)

Forcing each failure mode by editing a known-good conf:

| failure | conf change | runner output (first line + exit code) |
|---|---|---|
| missing .co | `co_file = does-not-exist.co` | `okl_run: .co file not found: ...` (exit 1) |
| wrong arch  | `target_arch = gfx950`        | `okl_run: arch mismatch` + both sides + remediation (exit 1) |
| wrong xnack | `xnack = on`                  | `okl_run: xnack mismatch` + both sides + remediation (exit 1) |
| wrong sym   | `kernel_symbol = Cijk_DoesNotExist_...` | `okl_run: kernel symbol not found in module:` + `clang-offload-bundler` + `llvm-readobj` recipe (exit 1) |

### 8.7 Things deliberately not done

- **No module caching.** The runner loads exactly one .co per invocation;
  caching would matter only for a long-lived process loading many shards
  (which is Tensile's job, not ours). `m_kernels` in `SolutionAdapter`
  (`HipSolutionAdapter.cpp:296-310`) is the precedent if we ever need it.
- **No `bundle_target` string-matching.** The kind prefix (`hipv4-` today,
  `hipv5-` in some HIP versions) is not stable; matching on it is a
  liability. The arch slug is what `hipModuleLoad` actually keys on.
- **No multi-arch package format.** One conf = one shard = one (arch, xnack)
  pair. Multi-arch is a directory of confs, not a fat conf.
- **No symbol-list embedding in the conf.** We don't precompute "what symbols
  are in this .co" because (a) the user already knows the one they want
  (`kernel_symbol`) and (b) hot-introspection at runtime is cheap to defer to
  the explicit `llvm-readobj` recipe in the error message.
