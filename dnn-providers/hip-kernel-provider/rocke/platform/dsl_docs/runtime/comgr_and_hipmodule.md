# COMGR And HIP Module Deep Dive

`runtime/comgr.py` and `runtime/hip_module.py` are the lowest-level pieces of the DSL: a ctypes wrapper over `libamd_comgr.so` (LLVM IR -> HSACO) and `libamdhip64.so` (HSACO -> launchable kernel). This page explains how each works in detail.

## COMGR

### Installing the compiler

The `rocke` wheel contains no COMGR, LLVM or Clang binaries. Its base install
requires only NumPy; the optional `sdk-core` extra installs the shared
`rocm-sdk-core` package. Neither discovery nor COMGR loading imports torch or
calls `rocm_sdk.initialize_process` to initialize HIP.

From the `platform` directory, install the extra using an AMD package index:

```sh
python -m pip install -e '.[sdk-core]' --extra-index-url https://stable.repo.amd.com/rocm/whl-next/
```

For an already published wheel, replace `-e '.[sdk-core]'` with
`'rocke[sdk-core]'`. PyPI's `rocm-sdk-core==0.1.0` placeholder contains no
compiler and does not satisfy the extra's version requirement. Choose the
same SDK release/index as any other ROCm framework in the environment; the
extra does not force a separate private compiler version.

Package availability and rocKE's compiler baseline are separate:

| SDK wheel version | Linux x86-64 COMGR | Windows x86-64 COMGR | Meaning |
| --- | --- | --- | --- |
| `6.5.0rc20250601` | `_rocm_sdk_core/lib/libamd_comgr.so.3` | No Windows wheel in the retained index at this version | Earliest retained Linux nightly inspected; ships LLVM 19 |
| `7.0.0rc20250701` | Linux nightlies also published | `_rocm_sdk_core/bin/amd_comgr0605.dll` | Earliest retained Windows nightly inspected |
| `7.9.0rc1` | `_rocm_sdk_core/lib/libamd_comgr.so.3` | `_rocm_sdk_core/bin/amd_comgr0701.dll` | Minimum SDK package version requested by `rocke[sdk-core]`; Linux ships LLVM 20 |
| `7.10.0` | `_rocm_sdk_core/lib/libamd_comgr.so.3` | `_rocm_sdk_core/bin/amd_comgr0702.dll` | Later stable SDK package; Linux ships LLVM 22 |

These are TheRock SDK **package versions**, not a claim that identically
numbered HIP components or distribution releases introduced Python packaging.
The early development wheels demonstrate that SDK packaging predates the
`7.9` series. The extra deliberately uses the common `7.9.0rc1` floor on Linux
and Windows rather than selecting the oldest available wheel on each OS:
older packages can predate the LLVM 20 emission baseline. AUTO still queries
the loaded compiler, and compilation checks the IR/compiler flavor contract;
the package floor does not establish support for every target or compiler build.

The historical wheel contents and versions can be inspected at AMD's
[legacy nightly index](https://rocm.nightlies.amd.com/v2/gfx1151/rocm-sdk-core/)
and [legacy release index](https://repo.amd.com/rocm/whl/gfx1151/rocm-sdk-core/).
Legacy multi-arch packages start at `7.13` in the
[multi-arch release index](https://repo.amd.com/rocm/whl-multi-arch/rocm-sdk-core/).
For current stable releases, use the
[stable SDK index](https://stable.repo.amd.com/rocm/whl-next/rocm-sdk-core/).
TheRock documents the current versus historical index split in
[its release guide](https://github.com/ROCm/TheRock/blob/main/RELEASES.md).

Discovery reads the installed `rocm-sdk-core` distribution's file metadata
without importing `rocm_sdk` or a payload package. It locates Linux
`libamd_comgr.so` and numeric SONAME variants under the payload's `lib`
directory, or Windows `amd_comgr.dll` and numeric versioned DLL names under
`bin`. Both `_rocm_sdk_core` and `_rocm_sdk_core_<suffix>` payloads are
supported. Windows registers the selected DLL directory for its dependencies.
Wheel inspection establishes package contents; it is not a Windows runtime
compilation result.

An existing system ROCm/HIP SDK compiler remains usable without the extra by
setting `ROCKE_COMGR_LIB` to that library. `ROCM_PATH`, `ROCM_HOME`, and
`HIP_PATH` do not select COMGR. To author
LLVM IR without any COMGR installed, pass an explicit `llvm_flavor` to lowering
or set `ROCKE_LLVM_FLAVOR`. GPU launch still requires a suitable HIP runtime,
GPU and driver; the `sdk-core` extra does not provide the driver.

### Compilation pipeline

`libamd_comgr.so` is the ROCm compiler-support library. Its API lets a process compile LLVM IR text to a HSA code object (HSACO) without spawning hipcc or clang. The DSL drives the following chain:

```text
LLVM IR text (utf-8)
  -> AMD_COMGR_DATA_KIND_SOURCE  (lang = LLVM_IR)
  -> AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC            -> bitcode
  -> AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE       -> relocatable ELF
  -> AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE  -> HSACO bytes
```

Constants (`runtime/comgr.py`):

```text
AMD_COMGR_DATA_KIND_SOURCE       = 0x1
AMD_COMGR_DATA_KIND_BC           = 0x6
AMD_COMGR_DATA_KIND_RELOCATABLE  = 0x7
AMD_COMGR_DATA_KIND_EXECUTABLE   = 0x8
AMD_COMGR_DATA_KIND_BYTES        = 0x9
AMD_COMGR_LANGUAGE_LLVM_IR       = 0x4
AMD_COMGR_ACTION_COMPILE_SOURCE_TO_BC            = 0x2
AMD_COMGR_ACTION_LINK_BC_TO_BC                   = 0x3
AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE       = 0x4
AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE  = 0x7
AMD_COMGR_STATUS_SUCCESS                         = 0
```

Library loading is deferred until the first compiler query or compilation.
Both Python and native C++ select COMGR with this policy:

1. A nonempty `$ROCKE_COMGR_LIB` selects that explicit library.
2. Otherwise, select COMGR owned by the installed `rocm-sdk-core` distribution.

An unloadable override fails rather than falling back to the SDK. Missing SDK
metadata, a missing COMGR payload, or a missing compiler dependency produces an
actionable error. Neither torch imports, mapped foreign libraries, system ROCm
roots, `ROCM_PATH`/`ROCM_HOME`/`HIP_PATH`, nor bare library names supply an
automatic COMGR fallback. An empty override is equivalent to unset.
Successful loads are retained for the lifetime of the process; later imports
or environment changes do not replace that compiler. AUTO flavor selection
queries its LLVM version, independently of package release metadata.

Python obtains payload ownership from `importlib.metadata`. Native standalone
AUTO reads distribution metadata under installation locations adjacent to the
loaded engine or executable, including `site-packages`, Linux
`lib/python*/site-packages` and `lib64/python*/site-packages`, and Windows
`Lib/site-packages`. An installed shared engine or executable and its SDK can
move together without a build-machine SDK path. CMake also writes a relative
`rocke_sdk_site.txt` descriptor for its selected Python interpreter, allowing
build-tree binaries to locate that interpreter's SDK. This descriptor stays in
the build tree and is never installed; reconfigure when changing interpreters.
Standalone consumers outside those installation locations and the configured
build tree must select COMGR with `ROCKE_COMGR_LIB`. Python bindings continue
delegating AUTO to Python's resolver. The native adapter queries compiler version; it does not implement
Python's COMGR-to-HSACO compilation pipeline.

The compatibility hook `comgr.prefer_bundled_lib()` remains callable. It uses
the same resolver without importing torch and returns informational package
release metadata.

On Linux, before opening the selected COMGR, both loaders reject a distinct
mapped COMGR file. Aliases of the same file are accepted. The ordinary TheRock
torch/rocKE shared SDK case therefore reuses one compiler. A different system
or legacy torch compiler already mapped is a conflict, even in a torch-free
process, and is never adopted automatically. Loading another COMGR can also
load another LLVM and abort before an error is returned. Compile in a separate
process before loading the conflicting framework, then hand the HSACO bytes
to its launch process. Mapped-file checks establish COMGR file identity; they
do not prove isolation of every LLVM dependency. An unavailable mapping
inventory cannot establish that another compiler is absent.

HIP keeps its independent runtime policy, described below. Sharing the HIP
runtime used by torch matters because a second HIP instance has disjoint module
state; COMGR's env-or-SDK policy does not change HIP selection.

## Entry Point

```python
hsaco, timings = build_hsaco_from_llvm_ir(
    ir_text,
    isa="amdgcn-amd-amdhsa--gfx950",
    options=["-O3"],
)
```

Returns `(bytes, ComgrTimings)`. `ComgrTimings(bc, link, relocatable, executable)` carries per-stage seconds; `.total` is the sum. Construct it by keyword.

`build_hsaco_from_llvm_modules(ir_texts, isa=..., options=...)` compiles several modules into one code object: each is compiled to bitcode, the bitcode is linked (`LINK_BC_TO_BC`, timed as `link`; 0.0 for a single module) and code-generated once. `compile_module(KernelModule(...))` in `helpers/compile.py` uses it to ship companion kernels beside a primary.

Errors from any stage raise `ComgrError(f"{where}: status={s} ({status_string(s)})")`. The actual error message comes from `amd_comgr_status_string`.

## What `-O3` Does

The COMGR backend respects standard clang options. The default `-O3` enables:

- LLVM SSA optimization passes (DCE, SROA, instcombine, loop opt, etc.);
- AMDGPU backend's scheduling and register-allocation heuristics;
- LLVM's vectorizer.

For the kernels in `rocke`, most performance comes from IR-level decisions (atom, tile, LDS, async, cshuffle), not from `-O*` flags. The runbook explicitly says: do not assume compiler flags explain a gap until ISA / resource evidence supports it.

## Adding Compiler Options

For durable scheduler tuning, attach a validated `CodegenPolicy` to the kernel
and compile through `compile_kernel()`. See
[`../optimization/scheduler-policy.md`](../optimization/scheduler-policy.md).
The policy is emitted as an `amdgpu-sched-strategy` LLVM function attribute and
survives the serialized-IR handoff to either LLVM lowerer.

`build_hsaco_from_llvm_ir(ir_text, options=[...])` passes any list of strings to `amd_comgr_action_info_set_option_list`. Useful examples:

```text
"-O3"
"-mllvm", "-amdgpu-early-inline-all=false"
"-mllvm", "-amdgpu-prelink"
"-Wl,--strip-debug"
```

Arbitrary options are intentionally not exposed through `compile_kernel`; call
`build_hsaco_from_llvm_ir` directly for isolated compiler diagnostics. Raw
options are not automatically validated or included in artifact identity.

## Compiling In A Fresh Process

An in-process compile is not always a function of the IR alone. LLVM's AMDGPU
IGroupLP strategy behind `llvm.amdgcn.iglp.opt(i32 1)` (the single-wave
small-GEMM strategy) keeps its DS-write counts in file-scope statics that it adds
to and never resets, so the schedule of an `iglp_opt(1)` kernel depends on every
`iglp_opt(1)` kernel the same process compiled before it (its own assertion,
compiled out in release builds, expects the counts to be zero). Measured on the
attention backward kernels: the same IR gives another code object, with a
different resource-gate verdict, after other compiles.

`build_hsaco_from_llvm_ir(..., fresh_process=True)` compiles the module in a new
Python process that compiles nothing else
(`rocke.runtime._comgr_fresh_process`): the comgr library this process resolved
(passed as `ROCKE_COMGR_LIB`), this rocke package first on `PYTHONPATH`, and
comgr's own on-disk cache off. A compile error raises `ComgrError` with the
child's message; a child that dies raises `ComgrProcessDied` (a `ComgrError`).
`compile_kernel` does this for a kernel whose builder set the kernel attribute
`COMGR_FRESH_PROCESS_ATTR` (`"comgr_fresh_process"`, compile policy only: no
lowerer reads it, so the LLVM IR is unchanged). The attention backward main
kernel sets it whenever it emits `iglp_opt(1)`; other families compile in-process
as before. The cost is one short process start per such kernel. `compile_module`
links its modules in-process, so it refuses a kernel carrying the attribute
(`ValueError`); compile such a kernel with `compile_kernel`.

## HIP Module Layer

`runtime/hip_module.py` is the matching ctypes wrapper for `libamdhip64.so` (the HIP runtime). It exposes only what the DSL needs.

### Library loading

The HIP module loader retains its runtime discovery order:
`$ROCKE_HIP_LIB` override, then, if torch is already imported, the
`libamdhip64` torch uses (an unambiguous mapped copy, else
`<torch>/lib/libamdhip64.so`, else `_rocm_sdk_core/lib/libamdhip64.so*` when
`rocm_sdk` is imported), then the ROCm installs (`libamdhip64.so` and the
`.so.7` SONAME), then bare `libamdhip64.so` via the dynamic linker.
Preferring the copy torch loaded keeps rocke and torch on the same HIP
runtime instance. Linux HIP does not add a standalone SDK tier; Windows also
looks for SDK runtime DLLs and under `HIP_PATH`, `ROCM_PATH`, or `ROCM_HOME`
before a bare DLL name. COMGR's env-or-SDK policy does not change these HIP
candidates.

Failure raises `HipError`.

### Constants

```text
HIP_LAUNCH_PARAM_BUFFER_POINTER = ctypes.c_void_p(1)
HIP_LAUNCH_PARAM_BUFFER_SIZE    = ctypes.c_void_p(2)
HIP_LAUNCH_PARAM_END            = ctypes.c_void_p(3)
hipMemcpyHostToDevice = 1
hipMemcpyDeviceToHost = 2
HIP_SUCCESS          = 0
HIP_ERROR_NOT_READY  = 600
```

### Bound HIP Functions

```text
hipGetErrorString
hipModuleLoadData / hipModuleUnload / hipModuleGetFunction
hipModuleLaunchKernel
hipMalloc / hipFree / hipMemcpy / hipMemset
hipDeviceSynchronize / hipStreamSynchronize
hipEventCreate / hipEventDestroy / hipEventRecord
hipEventSynchronize / hipEventQuery / hipEventElapsedTime
```

### `Runtime`

```python
rt = Runtime()
mod = rt.load_module(hsaco_bytes)        # hipModuleLoadData
fn  = mod.get_function(kernel_name)      # hipModuleGetFunction
rt.launch(fn, grid, block, args_bytes, shared_bytes=0, stream=0, record_event=True)
rt.launch_blocking(fn, grid, block, args_bytes, ...)
rt.sync()                                # hipDeviceSynchronize
rt.wait_stream(stream)                   # hipStreamSynchronize + release retained
```

`Runtime` owns:

- a per-stream FIFO of `(refs_tuple, completion_event_or_None)`;
- the bound function table.

The per-stream FIFO solves two real correctness problems:

1. **HIP "extra" launch path does not promise to copy the packed args buffer at enqueue time.** The GPU command processor has been observed reading the buffer after `hipModuleLaunchKernel` returns. The FIFO keeps each packed-args bytes object alive until the stream is sync'd.
2. **Raw HIP launches are invisible to torch's caching allocator.** Tensors passed by raw pointer can have their storage recycled mid-launch unless explicitly retained. `Runtime.retain_for_stream(stream, *values)` merges tensor refs into the most recent FIFO entry.

`Runtime.release_pending_for_stream(stream)` drops retained refs after external synchronization; `Runtime.wait_stream(stream)` is the canonical "drain + release" primitive on ROCm (more reliable than `torch.cuda.synchronize()` for raw ctypes launches).

### Launch Mechanism

`hipModuleLaunchKernel` takes args via the "extra" buffer:

```text
extra[] = {
    HIP_LAUNCH_PARAM_BUFFER_POINTER, &args_bytes[0],
    HIP_LAUNCH_PARAM_BUFFER_SIZE,    &args_size,
    HIP_LAUNCH_PARAM_END,
}
```

`args_bytes` is the packed arg buffer (built by `runtime/packing.py::pack_args` from the signature dict list). For an `(A, B, C, M, N, K)` GEMM:

```text
struct.pack("<QQQiii", A_dev, B_dev, C_dev, M, N, K)
```

Three pointers (8 bytes each), three i32 scalars (4 bytes each), 36 bytes total.

For pointer arg correctness, the pointer values must be valid device pointers (`hipMalloc`-style or torch tensor `.data_ptr()`); the AMDGPU side decodes them through the `make_buffer_rsrc` intrinsic (for buffer paths) or via direct GEP (for `addrspace(1)` paths).

### `Event`

```text
Event.record(stream=0) / synchronize() / query() / elapsed_to(other) / destroy()
```

`query()` is non-blocking: returns True iff `hipEventQuery` reports `hipSuccess`. Used by `Runtime._reap_completed` to drop FIFO entries whose kernels have finished.

## When To Use The Low Level

For everyday DSL use, `KernelLauncher` and `compile_kernel` are the right abstraction. Reach into `Runtime` / `build_hsaco_from_llvm_ir` directly when:

- You want to pass non-default COMGR options.
- You're writing a manifest runner outside torch.
- You need fine-grained event control across multiple streams.
- You're debugging a launch failure and want to inspect packed args + grid + block + stream manually.

Numpy-only paths (the manifest runner) go through `Runtime.alloc` / `memcpy_h2d` / `launch` / `memcpy_d2h` directly with `struct.pack`-built args. Torch paths go through `runtime/packing.pack_args` + `KernelLauncher`. Both end up at the same `hipModuleLaunchKernel`.

## Common Failure Modes

- `ComgrError: do_action(COMPILE_SOURCE_TO_BC): status=4`: malformed LLVM IR (unknown intrinsic, wrong type, wrong address space). Inspect `art.llvm_text` and `core/lower_llvm.py::_INTRINSIC_DECLS`.
- `ComgrError: do_action(COMPILE_SOURCE_TO_BC): status=1`: most commonly an intrinsic signature mismatch between the toolchain and the emitted IR (e.g. `make.buffer.rsrc.p1` vs `.p8.p1` between LLVM 20 and LLVM 21+). The DSL picks the right flavor automatically — see "LLVM Intrinsic Flavor" in `dsl_docs/ir_lowering/backend_details.md` for the override knob.
- `ComgrError: ... status=5`: undefined intrinsic at link time. Likely a bf16 path that needs the `_1k` variant (see `mfma.f32.16x16x16bf16.1k` in `_INTRINSIC_DECLS`).
- `HipError: hipModuleLoadData: hipError(700)`: HSACO failed to load — usually an ISA mismatch (HSACO built for gfx950, runtime device is gfx940/gfx942).
- `HipError: hipModuleGetFunction: hipError(500)`: function name mismatch. Check `art.kernel_name` matches what the launcher was constructed with.
- `HipError: hipModuleLaunchKernel: hipError(701)`: launch failed. Most often: more threads than `kernel.attrs["max_workgroup_size"]`, or LDS over budget. Check both before chasing harder bugs.
- `HipError: ... hipError(2)`: out of memory; check device pointers and sizes.

## Verifying The Stack End-To-End

A minimal repro that touches every layer:

```python
from rocke import (
    IRBuilder, F16, I32, PtrType, compile_kernel,
)
from rocke.runtime.launcher import KernelLauncher, LaunchConfig

b = IRBuilder("smoke_copy")
b.kernel.attrs["max_workgroup_size"] = 64
X = b.param("X", PtrType(F16, "global"), readonly=True, align=2)
Y = b.param("Y", PtrType(F16, "global"), writeonly=True, align=2)
N = b.param("N", I32)
tid = b.thread_id_x()
v = b.global_load_f16(X, tid)
b.global_store(Y, tid, v)

art = compile_kernel(b.kernel)

launcher = KernelLauncher(
    hsaco=art.hsaco,
    kernel_name=art.kernel_name,
    signature=[
        {"name": "X", "type": "ptr<f16, global>"},
        {"name": "Y", "type": "ptr<f16, global>"},
        {"name": "N", "type": "i32"},
    ],
)

import torch
X = torch.arange(64, dtype=torch.float16, device="cuda")
Y = torch.empty_like(X)
launcher({"X": X, "Y": Y, "N": 64},
         config=LaunchConfig(grid=(1, 1, 1), block=(64, 1, 1)))

assert torch.equal(X, Y), "smoke copy failed"
print("end-to-end smoke ok")
```

If this works, COMGR + HIP module load + ctypes pack + launch all work on the box.
