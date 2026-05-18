# COMGR And HIP Module Deep Dive

`runtime/comgr.py` and `runtime/hip_module.py` are the lowest-level pieces of the DSL: a ctypes wrapper over `libamd_comgr.so` (LLVM IR -> HSACO) and `libamdhip64.so` (HSACO -> launchable kernel). This page explains how each works in detail.

## COMGR

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

Library loading: the runtime checks the default ROCm library locations, then
falls back to bare `libamd_comgr.so` via dlopen. Failure raises `ComgrError`.

## Entry Point

```python
hsaco, timings = build_hsaco_from_llvm_ir(
    ir_text,
    isa="amdgcn-amd-amdhsa--gfx950",
    options=["-O3"],
)
```

Returns `(bytes, ComgrTimings)`. `ComgrTimings(bc, relocatable, executable)` carries per-stage seconds; `.total` is the sum.

Errors from any stage raise `ComgrError(f"{where}: status={s} ({status_string(s)})")`. The actual error message comes from `amd_comgr_status_string`.

## What `-O3` Does

The COMGR backend respects standard clang options. The default `-O3` enables:

- LLVM SSA optimization passes (DCE, SROA, instcombine, loop opt, etc.);
- AMDGPU backend's scheduling and register-allocation heuristics;
- LLVM's vectorizer.

For the kernels in `ck_dsl`, most performance comes from IR-level decisions (atom, tile, LDS, async, cshuffle), not from `-O*` flags. The runbook explicitly says: do not assume compiler flags explain a gap until ISA / resource evidence supports it.

## Adding Compiler Options

`build_hsaco_from_llvm_ir(ir_text, options=[...])` passes any list of strings to `amd_comgr_action_info_set_option_list`. Useful examples:

```text
"-O3"
"-mllvm", "-amdgpu-early-inline-all=false"
"-mllvm", "-amdgpu-prelink"
"-Wl,--strip-debug"
```

This is not currently exposed through `compile_kernel`; call `build_hsaco_from_llvm_ir` directly when you need custom options.

## HIP Module Layer

`runtime/hip_module.py` is the matching ctypes wrapper for `libamdhip64.so` (the HIP runtime). It exposes only what the DSL needs.

### Library loading

The HIP module loader follows the same pattern: default ROCm library
locations first, then bare `libamdhip64.so` via dlopen.

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

`args_bytes` is the packed arg buffer (built by `runtime/torch_module.py::pack_args` from the signature dict list). For an `(A, B, C, M, N, K)` GEMM:

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

Numpy-only paths (the manifest runner) go through `Runtime.alloc` / `memcpy_h2d` / `launch` / `memcpy_d2h` directly with `struct.pack`-built args. Torch paths go through `runtime/torch_module.pack_args` + `KernelLauncher`. Both end up at the same `hipModuleLaunchKernel`.

## Common Failure Modes

- `ComgrError: do_action(COMPILE_SOURCE_TO_BC): status=4`: malformed LLVM IR (unknown intrinsic, wrong type, wrong address space). Inspect `art.llvm_text` and `core/lower_llvm.py::_INTRINSIC_DECLS`.
- `ComgrError: ... status=5`: undefined intrinsic at link time. Likely a bf16 path that needs the `_1k` variant (see `mfma.f32.16x16x16bf16.1k` in `_INTRINSIC_DECLS`).
- `HipError: hipModuleLoadData: hipError(700)`: HSACO failed to load — usually an ISA mismatch (HSACO built for gfx950, runtime device is gfx940/gfx942).
- `HipError: hipModuleGetFunction: hipError(500)`: function name mismatch. Check `art.kernel_name` matches what the launcher was constructed with.
- `HipError: hipModuleLaunchKernel: hipError(701)`: launch failed. Most often: more threads than `kernel.attrs["max_workgroup_size"]`, or LDS over budget. Check both before chasing harder bugs.
- `HipError: ... hipError(2)`: out of memory; check device pointers and sizes.

## Verifying The Stack End-To-End

A minimal repro that touches every layer:

```python
from ck_dsl import (
    IRBuilder, F16, I32, PtrType, compile_kernel,
)
from ck_dsl.runtime.launcher import KernelLauncher, LaunchConfig

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
