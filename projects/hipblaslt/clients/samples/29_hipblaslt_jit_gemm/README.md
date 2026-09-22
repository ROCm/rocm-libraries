# JIT GEMM through TensileLite

`hipblaslt_ext::experimental::getJitGemmAlgo` generates one solution and returns
an algorithm for normal `hipblasLtMatmul` or extension `Gemm` execution. An empty
`GenerateOptions::configPath` invokes Origami parameter prediction; an explicit
YAML path invokes `Tensile.SingleSolution` directly. The benchmark
[`--jit-gemm` option](../../bench/README.jit.md) uses the automatic route.

The sample option `--normal-api both` exercises an explicit recipe through both
normal APIs, including copied algorithms and repeated calls with changed input.
Normal execution supplies the existing Stream-K and output-amax synchronization
bindings. Stream-K and output-amax follow the same generated-solution predicates
as other normal algorithms. Generated algorithms and modules remain registered
until process exit. They are valid only on their
original device in that process and have no reusable prebuilt solution index.
Normal execution follows the handle's existing synchronization ownership:
Stream-K binds a stream-specific region, while MBSK and output-amax use shared
handle storage. The normal algorithm API does not provide the standalone
owner's private synchronization buffers for concurrent calls.

The separate owner API below provides explicit preparation and module lifetime:

`hipblaslt_ext::experimental::JitGemm` calls TensileLite synchronously during
`prepare` to generate and build one GEMM **solution**, including its helper
kernels and support code. It loads every returned code object into a private
adapter. The existing Tensile runtime checks support, computes workspace,
selects fixed or automatic split-K behavior, packs arguments, and returns the
complete ordered invocation sequence. `run` executes that sequence without
regenerating or benchmarking.

This build-only experiment is a step toward JIT GEMM from hipBLASLt. Enable
`HIPBLASLT_ENABLE_JIT_GEMM` to build it and the `hipblaslt-jit-gemm` numerical
executable. It is disabled by default; its header at
`library/src/amd_detail/hipblaslt-jit-gemm.hpp` is not installed and has no stable
ABI commitment. It requires Linux, Boost headers, ROCm, source-built TensileLite
host/rocisa, and TensileLite's third-party Python dependencies. It requires no
shipped hipBLASLt device library or installed project package.

## Build from a prepared checkout

From the repository root, reuse the existing `.venv` and release build used for
the local rocisa extension:

```bash
project_root="$PWD"
project_build="$project_root/projects/hipblaslt/build/release"
source "$project_root/.venv/bin/activate"
export PATH="/opt/rocm/bin:/opt/rocm/llvm/bin:$PATH"
export PYTHONPATH="$project_build/tensilelite/rocisa:$project_build/tensilelite:$project_root/projects/hipblaslt/tensilelite"
export TENSILE_DISABLE_HELPER_CACHE=1
export TENSILE_HELPER_CACHE_DIR="$project_build/helper-cache"

cmake -S "$project_root/projects/hipblaslt" -B "$project_build" \
  -DHIPBLASLT_ENABLE_JIT_GEMM=ON
cmake --build "$project_build" --target _rocisa hipblaslt-jit-gemm -j 8
```

A minimal initial configuration also sets AMD C/C++ compilers, the ROCm prefix,
`GPU_TARGETS`, both CMake Python executable variables, and these options:

```text
-DHIPBLASLT_ENABLE_HOST=ON -DTENSILELITE_ENABLE_HOST=ON
-DHIPBLASLT_ENABLE_DEVICE=OFF -DHIPBLASLT_ENABLE_CLIENT=OFF
-DHIPBLASLT_ENABLE_EXTOPS=OFF -DHIPBLASLT_ENABLE_MATRIX_TRANSFORM=OFF
-DHIPBLASLT_ENABLE_ROCROLLER=OFF -DHIPBLASLT_ENABLE_MARKER=OFF
-DHIPBLASLT_ENABLE_MXDATAGENERATOR=OFF -DHIPBLASLT_ENABLE_YAML=OFF
-DTENSILELITE_ENABLE_CLIENT=OFF -DTENSILELITE_ENABLE_AUTOBUILD=OFF
-DHIPBLASLT_BUILD_TESTING=OFF -DTENSILELITE_BUILD_TESTING=OFF
-DCMAKE_SKIP_INSTALL_RULES=ON
```

For a third-party nanobind wheel, set
`FETCHCONTENT_TRY_FIND_PACKAGE_MODE=ALWAYS` and `nanobind_DIR` to the output of
`python -m nanobind --cmake_dir`. Keep local native build directories on
`PYTHONPATH`; do not install the project to make imports work.

## Generate, prepare and execute

Each request requires a fresh output path and existing parent directory. Logs
and tool scratch files are retained in `<output>.log` and `<output>.cwd`. Choose
the HIP architecture and matching feature qualifiers for the executing device.
The examples below target MI355X:

```bash
fixtures="$project_root/projects/hipblaslt/tensilelite/Tensile/Tests/unit/test_data"

# Standalone reusable Python API/CLI; no GPU execution or benchmarking.
python -m Tensile.SingleSolution "$fixtures/single_solution_splitk.yaml" \
  /tmp/jit-solution-python --architecture gfx950 \
  --cxx-compiler /opt/rocm/bin/amdclang++

# Full C++ path: prepare itself invokes Python generation and compilation.
"$project_build/clients/staging/hipblaslt-jit-gemm" \
  "$VIRTUAL_ENV/bin/python" "$project_root/projects/hipblaslt/tensilelite" \
  "$PYTHONPATH" "$fixtures/single_solution_splitk.yaml" /tmp/jit-gemm-splitk \
  gfx950:sramecc+:xnack- /opt/rocm/bin/amdclang++ \
  --k 512 --expect-configured-gsu 4 --expect-min-gsu 4 \
  --expect-accumulation multiple-buffer --expect-kernels 2 --min-workspace 1

# Malformed-main/helper artifacts and unsupported descriptors must fail before launch.
python "$project_root/projects/hipblaslt/clients/samples/29_hipblaslt_jit_gemm/test_jit_bundle_failures.py" \
  "$project_build/clients/staging/hipblaslt-jit-gemm" \
  /tmp/jit-gemm-splitk/bundle /tmp/jit-gemm-negative \
  --architecture gfx950:sramecc+:xnack- --k 512

# The same explicit recipe selected as an ordinary C/extension algorithm.
"$project_build/clients/staging/hipblaslt-jit-gemm" \
  "$VIRTUAL_ENV/bin/python" "$project_root/projects/hipblaslt/tensilelite" \
  "$PYTHONPATH" "$fixtures/single_solution_splitk.yaml" /tmp/jit-normal-splitk \
  gfx950:sramecc+:xnack- /opt/rocm/bin/amdclang++ \
  --normal-api both --k 512

# Missing helper modules/symbols must fail before normal C/extension submission,
# while leaving a previously initialized extension algorithm runnable.
python "$project_root/projects/hipblaslt/clients/samples/29_hipblaslt_jit_gemm/test_jit_normal_helper_failures.py" \
  "$project_build/clients/staging/hipblaslt-jit-gemm" \
  /tmp/jit-normal-splitk/bundle /tmp/jit-normal-helper-negative
```

The sample defaults to M=256, N=128, K=128, column-major NN FP16 input/output,
FP32 accumulation, alpha=1.25 and beta=0.5. `--m`, `--n`, and `--k` change its
shape independently of the recipe's benchmark-size examples; those YAML sizes
are not benchmarked. Suggested additional cases use fresh output paths:

| Recipe | Shape M,N,K | Expected dispatch |
| --- | --- | --- |
| `single_solution.yaml` | 256,128,128 | GSU=1, MBSK, one invocation |
| `single_solution_singlebuffer.yaml` | 256,128,512 | GSU=4, SingleBuffer, beta initialization then main then conversion |
| `single_solution_adaptive.yaml` | 128,64,1024 | configured GSU=-1, resolved GSU>1, MBSK, one invocation |
| `single_solution_adaptive.yaml` | 128,64,4096 | configured GSU=-1, resolved GSU>1, MultipleBuffer, main plus conversion |

Use `--expect-configured-gsu -1 --expect-min-gsu 2` for the automatic cases,
`--expect-accumulation multiple-buffer-single-kernel` or `multiple-buffer`,
`--expect-kernels 1`, `2`, or `3`, and `--min-workspace 1` to make dispatch/workspace
expectations executable assertions. `DispatchInfo` reports configured and
resolved GSU, accumulation mode and actual Tensile invocation names. Helper
writer counts in the manifest are not invocation counts: one writer can emit
several entrypoints, and support headers/device functions are not launches.

The independent CPU triple loop compares every output element and explicitly
rejects nonfinite results. Inputs are nonuniform and D is reset to NaNs between
runs. The harness checks repeated runs with one generation, insufficient
nonzero workspace and invalidated readiness, capture/foreign-device rejection,
two owners sharing one handle on different streams with distinct data/workspace,
immediate destruction after enqueue, and reuse of the remaining owner. Zero
workspace and single-visible-device cases are reported as skipped where their
negative test is inapplicable. The fault script checks missing main/helper
modules and symbols, corrupt metadata, manifest schema/duplicates/paths and
unsupported descriptors, without launching an invalid solution. With `--amax 1`
and a matching `OutputAmaxD: true` recipe, the sample also checks output-amax
against its independent CPU reference across repeated and separate-owner runs.

## Owner contract and current scope

1. Construct `JitGemm` with a live handle and call `setProblem` with the usual
   matmul descriptors, matrices and host scalar pointers. The existing canonical
   translator validates descriptors and captures scalar values.
2. Call `prepare` with explicit Python/source/import/YAML/output/target/tool
   options. It waits for generation, validates manifest v2 and the sole local
   solution, loads all code objects, checks support and reports required workspace.
3. Allocate that workspace and call `initialize(workspace, bytes, stream)`.
   It validates stream ownership/capture, waits for prior work, packs the
   complete sequence with the normal solver, and resolves every invocation
   symbol before reporting readiness. Failed reinitialization invalidates it.
4. Call `run` on that stream. It enqueues the solution sequence and records a
   completion event; successful runs do not synchronize or regenerate. Private
   synchronization storage for MBSK, Stream-K, or output-amax is reset on the
   stream before dispatch. Separate owners do not alias these counters or flags.
5. Keep the handle, matrices, workspace and stream alive through completion and
   serialize calls on each owner. Reinitialization may change streams after
   waiting. Destruction waits before unloading all private modules and freeing
   private synchronization storage. Unrecoverable device/drain failure retains
   these resources with a diagnostic rather than releasing executing code/data.

Fixed GSU, automatic GSU (`GlobalSplitU: -1`) and adaptive accumulation
(`AdaptiveGemmGSUA: 1`) use the ordinary runtime decisions and workspace rules.
`AdaptiveGemm` is a separate store-width choice. Stream-K uses private queue/flag
storage in the standalone owner and the normal stream-specific binding through
`getJitGemmAlgo`. Output-amax similarly uses private owner storage or the normal
handle binding. Output-amax currently requires `GlobalSplitU: 1`, `StreamK: 0`,
and one batch: its reduction must see final output, and its workgroup reduction
does not include batch offsets. The shared generator validation and runtime
predicates enforce these limits for both API routes.

Automatic prediction supports gfx90a, gfx942, and gfx950. Explicit YAML uses the
normal target ISA and solution validators. Cross-compilation tests cover FP16
and FP32 recipes for all three architectures; runtime correctness must also be
checked on each GPU. Stream capture in the standalone owner and architecture
aliases not represented by the device's HIP name remain unsupported.
The owner API has no cache, tuning/benchmarking, transparent fallback, or package
installation. The normal algorithm route uses a private tag in the existing
opaque algorithm storage; the public algorithm type and size remain unchanged.
