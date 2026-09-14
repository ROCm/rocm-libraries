# The hipRTC envelope and the target device

[RUNBOOK.md](RUNBOOK.md) owns execution order; this page owns what the source may
contain and what the target architecture actually provides. Paths are relative to
the repository root; kernel paths shown bare (`hip_mlops_engine/…`,
`kernel_ingestor_engine/…`) sit under
`dnn-providers/hip-kernel-provider/src/engines/`.

## What a hipRTC kernel compiles against

hipRTC compiles a **string**, against a **caller-supplied list of virtual headers**.
There is no filesystem include search and no `-I`: whatever the host code does not
hand to `hiprtcCreateProgram` does not exist. In the provider path, the list is the
CMake-embedded header set plus, for a bundle-sourced kernel, the bundle's own `.h` /
`.hpp` / `.cuh` files one level deep, sorted by name
(`dnn-providers/hip-kernel-provider/src/engines/kernel_ingestor_engine/IngestorKernelCode.hpp`,
`collectKernelHeaders`). A bundle header whose name collides with an embedded one is
a load error, not a shadow.

| Available | Evidence |
|---|---|
| `__global__`, `extern "C"`, `__device__`, `__forceinline__` | every in-tree kernel |
| `__shared__`, including compile-time-sized arrays | `hip_mlops_engine/kernels/batchnorm/ReductionFunctions.hpp` |
| `warpSize`, `__shfl_sync`, `__shfl_down_sync` | same file |
| `float`, `double`, `_Float16`, `__half`, `__bf16` as device types, with no header | `hip_mlops_engine/kernels/common/HipKernelMath.hpp` |
| `__ocml_*` math (`__ocml_exp_f16`, `__ocml_sqrt_f16`, …), `__hfma` | same file |
| `__builtin_amdgcn_exp2f`, `__builtin_amdgcn_rcpf`, `__builtin_copysignf`, `__builtin_elementwise_fma` | same file |
| `<cstdint>` and the constrained device-usable standard headers hipRTC bundles | `kernel_ingestor_engine/kernels/ConvFwd.cpp` |

| Not available | Evidence |
|---|---|
| `<hip/hip_fp16.h>` | guarded out by `#ifndef __HIPCC_RTC__` in `hip_mlops_engine/kernels/types/VectorTypes.hpp` and `batchnorm/StaticUnroll.hpp`. Use `_Float16` / `__bf16` directly. |
| `<hip/hip_runtime.h>` from device source | no in-tree kernel includes it |
| Host STL (`<vector>`, `<string>`, `<iostream>`, …) | appears only in host-side files, never in a compiled kernel source |

**MFMA builtins (`__builtin_amdgcn_mfma_*`) have no precedent in this provider tree.**
Neither their availability nor their unavailability under this hipRTC path is
established. If the algorithm needs them, prove a minimal compile on the target
architecture first, and report that proof — do not assume either way. The matrix-core
kernels that do exist in tree (`hip_flash2_engine`, `asm_sdpa_engine`) are compiled
ahead of time with `hipcc` into code objects and are **not** hipRTC precedents; their
flags (`-O3`, `--cuda-device-only`, rocWMMA includes) do not transfer.

## Compile options

`dnn-providers/hip-kernel-provider/src/compilation/KernelCompileOptions.hpp` is the
production set, in order:

1. `-std=c++17`
2. `--offload-arch=<gcnArchName>` — the raw name, suffixes intact, e.g.
   `gfx942:sramecc+:xnack-`
3. `-D` pairs only.

The defines always present: `HIP_PLUGIN_USE_FP32` / `FP16` / `BFP16` (exactly one
set per tensor dtype), `HIP_PLUGIN_USE_RNE_BFLOAT16=1`, `HIP_PLUGIN_USE_FPMIX=0`,
`HIP_PLUGIN_USE_BFPMIX=0`, `HIP_PLUGIN_LAYOUT_NHWC`, `HIP_PLUGIN_USE_AMDGCN=0`, and
the `HIP_PLUGIN_GFX103X` / `110X` / `115X` / `120X` prefix flags. Callers add their
own with `.add(name, value)`.

**No optimization level, no fast-math and no warning flags are added by the
production path.** `-O3` appears only in ad hoc test fixtures and in the AOT `hipcc`
build. If your kernel's correctness or its measured behaviour depends on `-O3`, say
so explicitly — the integration path will not supply it for you.

Two consequences for authoring:

- A macro the source needs must be bound by whoever compiles it. Guard each with
  `#ifndef <NAME>` / `#error`.
- Downstream, only `bool`, `int` and `string` metadata can be rendered into a `-D`;
  `float` and integer lists are rejected, because a float has no single textual
  spelling and `-DALPHA=1` and `-DALPHA=1.0` are different types in device code. Pass
  floats and lists as kernel arguments, not macros.

## Device facts

The in-tree device query is thin: `gcnArchName`, `warpSize`, `multiProcessorCount`,
`totalGlobalMem`. **There is no in-tree table of LDS size, CU count or MFMA
inventory per architecture** — that gap is confirmed, not assumed. Anything richer
comes from a live device or from AMD's published ISA documentation
([prior-art.md](prior-art.md)), and an ISA-document fact used for sizing should be
cross-checked against a live `rocminfo -a` before it is trusted.

Local architecture:

```bash
rocminfo                 # agent properties: wavefront size, LDS/CU, SIMD count, CU count
rocminfo -a              # fuller dump
hipconfig --version
```

`projects/hipdnn/tools/IngestorGenerator/tools/device_probe.py --mode early --arch
"$ARCH" --sweep-root <existing-writable-dir>` confirms the requested architecture
string is present and a scratch root is writable. It is a feasibility gate; it
reports no capability numbers.

## Authoring for an architecture you do not have

This is the common case and it is supported: the kernel is authored against the
target's facts, and the correctness proof is run on that target through the
scheduler.

Use the workspace's `alola-gpu-test` skill to run anything — a `rocminfo` probe, the
compile, the harness — on the exact GPU class:

```bash
.claude/skills/alola-gpu-test/launch-gpu-test.sh run \
  --gpu gfx942-mi300x --constraint MARKHAM \
  --image /cluster/images/hipdnn/hipdnn_latest_gfx942.sqsh \
  --time 00:30:00 --name kernel-probe \
  --command 'hostname -s; rocminfo | head -40'
```

Three constraints that are load-bearing, from `Knowledge/alola/`:

- `--constraint MARKHAM` is required whenever the command reads `$HOME`: each site
  has an unrelated `/home`, and the worktrees exist only on Markham's.
- Every `gfx950-mi355x` node is AUSTIN, so a worktree-reading payload can never be
  scheduled on one. Probe it container-only with
  `--partition miopen --account miopen --no-constraint`, or stage sources to the
  node's own `/tmp`.
- A worktree reached through a symlink into login-node-local `/var/tmp` is invisible
  from every compute node. Stage the actual checkout to shared storage or to the
  node.

Two architectures means two compiles and two runs. A result on `gfx942` says nothing
about `gfx950`; report each separately, and do not let one stand in for the other.

## Facts worth recording per target

Architecture name (with suffixes), wavefront size, CUs, LDS bytes per workgroup,
maximum threads per block, register budget, and — once a kernel exists — its actual
resource usage. Occupancy and spill claims are `llvm-objdump` / profiler
observations; the workspace's `gpu-profile` skill owns them. They are optional
context for a correctness deliverable and must be labelled as measured or not.
