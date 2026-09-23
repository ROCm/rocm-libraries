# JIT GEMM through TensileLite

JIT GEMM compiles a matrix-multiplication solution when an application requests
it. `hipblaslt_ext::experimental::getJitGemmAlgo`, declared in
`hipblaslt/hipblaslt-ext.hpp`, returns an algorithm that can be passed to the
C `hipblasLtMatmul` API or the C++ extension `Gemm` API. Generation finishes
before the application submits GPU work.

`GenerateOptions::configPath` supplies an explicit YAML recipe for
`Tensile.SingleSolution` to compile.

The sample also demonstrates the separate `JitGemm` owner class. Its `prepare`
method generates and loads one solution, and `run` submits its kernels without
regeneration. This class owns the loaded modules and private synchronization
storage until its destruction; its header remains internal to the sample.

## Build

JIT GEMM is a build-time opt-in feature enabled by
`HIPBLASLT_ENABLE_JIT_GEMM`. It is disabled by default. The installed extension
header declares `getJitGemmAlgo` in either build configuration; without JIT
enabled, the function returns `HIPBLAS_STATUS_NOT_SUPPORTED`.

The enabled implementation requires the host library, Linux, ROCm, Boost
headers, TensileLite's Python dependencies, and the local rocisa extension.
Using an existing configured build and Python environment:

```bash
cmake -S "$project_root/projects/hipblaslt" -B "$project_build" \
  -DHIPBLASLT_ENABLE_JIT_GEMM=ON -DHIPBLASLT_ENABLE_HOST=ON \
  -DHIPBLASLT_ENABLE_DEVICE=OFF -DGPU_TARGETS=gfx950 \
  -DPython_EXECUTABLE="$project_python" -DPython3_EXECUTABLE="$project_python"
cmake --build "$project_build" --target _rocisa hipblaslt-jit-gemm --parallel
export PYTHONPATH="$project_build/tensilelite/rocisa:$project_build/tensilelite:$project_root/projects/hipblaslt/tensilelite"
```

Use the compiler and target appropriate for the local device. Generated
bundles do not depend on a prebuilt hipBLASLt device library.

## Generate and run an explicit recipe

The following example uses a split-K recipe on gfx950. Split-K divides the K
reduction across GPU workgroups; this recipe combines their partial results
with an output-conversion kernel. The output path must be new and its parent
directory must exist. Compiler and generator diagnostics are retained in
`<output>.log`, with process scratch files in `<output>.cwd`.

```bash
fixtures="$project_root/projects/hipblaslt/tensilelite/Tensile/Tests/unit/test_data"

"$project_build/clients/staging/hipblaslt-jit-gemm" \
  "$project_python" "$project_root/projects/hipblaslt/tensilelite" \
  "$PYTHONPATH" "$fixtures/single_solution_splitk.yaml" /tmp/jit-gemm-splitk \
  gfx950:sramecc+:xnack- /opt/rocm/bin/amdclang++ \
  --k 512 --expect-configured-gsu 4 --expect-min-gsu 4 \
  --expect-accumulation multiple-buffer --expect-kernels 2 --min-workspace 1
```

This invocation uses the `JitGemm` owner. Adding `--normal-api both` exercises
the same explicit recipe through `hipblasLtMatmul` and the C++ extension `Gemm`
instead. These APIs use the handle's existing synchronization storage:
Stream-K uses a stream-specific region, while MultipleBufferSingleKernel and
output-amax use shared handle storage. Their existing handle and concurrency
requirements also apply to generated algorithms. The sample owner provides
private synchronization buffers when that ownership is needed.

The sample defaults to M=256, N=128, K=128, column-major NN FP16 input/output,
FP32 accumulation, alpha=1.25, and beta=0.5. `--m`, `--n`, and `--k` change its
shape. Problem sizes in the YAML are examples for the generator's existing
schema; they do not select or time a kernel in this entry point.

| Recipe | Shape M,N,K | Kernel sequence |
| --- | --- | --- |
| `single_solution.yaml` | 256,128,128 | One main kernel; no split-K |
| `single_solution_splitk.yaml` | 256,128,512 | Split-K main kernel, then output conversion |
| `single_solution_singlebuffer.yaml` | 256,128,512 | Beta initialization, split-K main kernel, then output conversion |
| `single_solution_adaptive.yaml` | 128,64,1024 or 128,64,4096 | Runtime-selected split count and accumulation mode |

`DispatchInfo` reports the configured and selected split count, accumulation
mode, and invoked kernel names. `--expect-kernels`, `--expect-accumulation`, and
`--min-workspace` let the sample check a recipe's expected behavior. A solution
can contain more helper variants than a particular problem launches, so helper
counts in the manifest are not launch counts.

To generate the bundle without GPU execution, use the Python entry point:

```bash
python -m Tensile.SingleSolution "$fixtures/single_solution_splitk.yaml" \
  /tmp/jit-solution-python --architecture gfx950 \
  --cxx-compiler /opt/rocm/bin/amdclang++
```

The [single-solution documentation](../../../tensilelite/docs/single-solution.md)
describes the YAML contract, artifact layout, and Python return value.

## Preparation, execution, and reuse

`getJitGemmAlgo` returns a process-local algorithm after generation and support
checks. The application supplies the reported workspace and uses the C or C++
extension execution API. Copies of the algorithm can be reused on the same
device without generating another kernel. Its modules remain registered until
process exit.

The separate sample owner has an explicit lifetime:

1. Construct `JitGemm` with a live handle, then call `setProblem` with matmul
   descriptors, matrix storage, and host alpha/beta scalars.
2. Call `prepare` with the YAML and generation options. It compiles and loads
   the complete solution, checks support, and reports the required workspace.
3. Allocate workspace and call `initialize` with the workspace and stream.
   This packs the kernel arguments and resolves every entrypoint before the
   owner becomes ready. Reinitialization waits for earlier work; failure
   invalidates readiness.
4. Call `run` on the initialized stream as often as needed. It enqueues the
   complete kernel sequence without compiling or synchronizing with the host.
5. Keep the handle, matrices, workspace, and stream alive until work completes.
   Calls on each owner must be serialized. Destruction waits for submitted
   work before unloading modules and releasing private synchronization storage.

A fresh output path is needed only for generation, not for each execution.
Neither API currently loads a retained disk bundle in a later process or
maintains a persistent kernel cache. That interface is deferred to future JIT
library work. Save the YAML and manifest to reproduce a recipe; an opaque
algorithm value cannot be persisted as a library entry.

## Validation and current limits

The sample independently computes `alpha * A * B + beta * C` on the CPU with
FP32 accumulation, then compares every GPU output element with that reference.
It rejects nonfinite results and resets D to NaNs between runs so a missing
write cannot pass by preserving an earlier result. With `--amax 1` and an
`OutputAmaxD: true` recipe, it also compares output-amax with the CPU reference.

The sample checks repeated execution, workspace errors, invalidated
initialization, two owners sharing a handle on separate streams, and
destruction immediately after submission. The accompanying
`test_jit_bundle_failures.py` and `test_jit_normal_helper_failures.py` cover
invalid manifests, metadata, modules, and symbols before GPU submission.

Explicit YAML uses TensileLite's target and solution validators. Compiling
kernels for a target checks generator and compiler support; checking numerical
correctness requires execution on that GPU. Both runtime routes preserve the
physical MX scale layout: gfx950 requires pre-swizzled block32 UE8M0 descriptors
and a `HostPreSwizzle` solution; gfx1250 requires ordinary block-scale descriptors
and an `InMemorySwizzle` solution. Incompatible descriptor and solution layouts
are rejected. Natural gfx950 scale loads need an implementation in the shared
subtile generator before that layout can be accepted.

Output-amax currently requires one batch, `GlobalSplitU: 1`, and `StreamK: 0`.
Its reduction needs final output and does not combine batch offsets, so other
combinations require changes to the reduction and runtime checks. The sample
owner does not support stream capture; preparation and lifetime handling for
capture are outside its initial scope. Architecture aliases that do not match
the device's HIP name also require additional target resolution. Neither API
tunes kernels or provides a fallback when generation or support checks fail.
