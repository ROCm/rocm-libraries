# Compile, Launch, And Manifest Runtime

This page follows a kernel from `KernelDef` through HSACO to a torch-aware launch.

## Compile Entry Point

File: `helpers/compile.py`.

```python
from ck_dsl.helpers import compile_kernel
art = compile_kernel(
    kernel,
    isa="amdgcn-amd-amdhsa--gfx950",
    capture_ir_text=True,
    optimize_ir=False,
) -> KernelArtifact
```

`KernelArtifact` fields:

```text
kernel       : KernelDef (the input)
ir_text      : str        # MLIR-style dump (empty if capture_ir_text=False)
llvm_text    : str        # AMDGPU LLVM IR text
hsaco        : bytes      # HSA code object
timings      : Dict[str, float]   # per-stage ms
pass_stats   : PassStats          # constants_folded, common_subexpressions, dead_ops_removed
isa          : str
kernel_name  : str
hsaco_bytes  : int
```

Timing keys:

```text
ir_opt              # only when optimize_ir=True
ir_build            # print_ir
ir_lower_llvm       # lower_kernel_to_llvm
comgr_bc            # LLVM IR -> bitcode
comgr_relocatable   # bitcode -> relocatable ELF
comgr_executable    # relocatable -> HSACO
total
```

## COMGR Build

File: `runtime/comgr.py`.

```python
hsaco, timings = build_hsaco_from_llvm_ir(
    ir_text,
    isa="amdgcn-amd-amdhsa--gfx950",
    options=["-O3"],
)
```

The driver loads `libamd_comgr.so` via ctypes from `/opt/rocm/lib/libamd_comgr.so`, `.so.3`, or bare lookup; passes the IR text through `COMPILE_SOURCE_TO_BC -> CODEGEN_BC_TO_RELOCATABLE -> LINK_RELOCATABLE_TO_EXECUTABLE`; and extracts the executable bytes.

`ComgrTimings(bc, relocatable, executable)` returns per-stage seconds. `compile_kernel` converts to milliseconds for `KernelArtifact.timings`.

## HIP Runtime Layer

File: `runtime/hip_module.py`. Loads `libamdhip64.so` via ctypes. Exposes:

```text
Runtime()
Runtime.alloc(nbytes)             # hipMalloc
Runtime.free(ptr)                 # hipFree
Runtime.memcpy_h2d / memcpy_d2h / memset / sync / wait_stream
Runtime.load_module(blob) -> Module
Runtime.event() -> Event
Runtime.launch(fn, grid, block, args_bytes, shared_bytes=0, stream=0, record_event=True)
Runtime.launch_blocking(fn, grid, block, args_bytes, shared_bytes=0, stream=0)
Runtime.retain_for_stream(stream, *objs)
Runtime.release_pending_for_stream(stream)

Module.get_function(name) -> _HipFunctionHandle
Event.record(stream=0) / synchronize() / query() / elapsed_to(other) / destroy()
```

Important lifetime detail: `Runtime` owns a per-stream pending-args queue (`_pending_args[stream]`). Raw `hipModuleLaunchKernel` calls go through ctypes and are invisible to torch's allocator. Two failure modes follow without this queue:

1. The `HIP_LAUNCH_PARAM_BUFFER_POINTER` ("extra") path is not required to copy the packed args buffer at enqueue time. The GPU command processor has been observed reading the buffer after `hipModuleLaunchKernel` returns. Retaining the args buffer per launch + draining on stream sync fixes it.
2. torch tensors passed by raw pointer are not seen by the caching allocator. The pool can recycle their storage while a kernel is still reading them. `retain_for_stream(stream, *values)` keeps the tensor references alive until the stream is sync'd.

Both fixes are automatic when launches go through `KernelLauncher`.

## Torch Runtime Layer

File: `runtime/torch_module.py`.

```text
pack_args(signature, values) -> bytes      # packs kernel args in declaration order
pack_args_kernelparams(signature, values)  # same, for KernelParams ABI
resolve_stream(stream) -> int              # 0 -> torch.cuda.current_stream().cuda_stream
empty_workspace(shape, dtype, device) -> torch.Tensor
launch_torch_kernel(...)                   # back-compat shim
```

Stream correctness matters. Launching on the wrong stream can race the torch caching allocator or surrounding PyTorch work; this is why `LaunchConfig.stream=0` is auto-resolved to torch's current stream.

## KernelLauncher

File: `runtime/launcher.py`.

```python
from ck_dsl.runtime.launcher import KernelLauncher, LaunchConfig

launcher = KernelLauncher(
    hsaco=art.hsaco,
    kernel_name=art.kernel_name,
    signature=gemm_args_signature(),     # list of {"name": ..., "type": ...} dicts
    cache_key=("gemm", spec.tile, ...),  # optional, semantic key for caches
)

values = {"A": A, "B": B, "C": C, "M": M, "N": N, "K": K}
launcher(values, config=LaunchConfig(
    stream=0,                     # 0 -> resolve to torch current stream
    grid=(grid_x, grid_y, grid_z),
    block=(block_size, 1, 1),
    shared_bytes=0,
    fence=True,                   # default; HIP-event sync after launch
))
```

Construct **once** per (problem-shape, problem-dtype). The HIP module is loaded eagerly in `__init__` and held for the lifetime of the launcher. The `Module._blob` reference keeps the HSACO bytes alive.

`LaunchConfig.fence=True` (default) makes the launcher synchronize on the launch's completion before returning (`hipStreamSynchronize`). This matches CK Tile's `launch_kernel` contract; the per-call cost is ~0.3 us. `fence=False` is fire-and-forget; the caller must drain (`time_launches` does this under the hood).

The `no_fence()` context manager forces `fence=False` for every nested launcher call regardless of per-call config; `time_launches` uses it to wrap a timed loop.

## PipelineLauncher

```python
pipeline = PipelineLauncher([segment_launcher, reduce_launcher])
pipeline(
    values_per_stage=[seg_vals, reduce_vals],
    configs_per_stage=[seg_cfg, reduce_cfg],
    stream=0,
)
```

All stages run on the same stream. Same-stream FIFO ordering already guarantees stage N+1 observes stage N's writes, so intermediate stages do **not** fence; only the last stage honors `cfg.fence`. Used by:

- split-KV attention (`attention_tiled_3d_segment` -> `attention_tiled_3d_reduce`);
- any future fixup kernel chains (k-fixup GEMM, im2col + GEMM + col2im).

## WorkspacePool

```python
from ck_dsl.runtime.launcher import WorkspacePool, WorkspaceSpec

pool = WorkspacePool()
ws = pool.get(
    name="segm_output",
    shape=(num_q, num_h, num_segments, head_size),
    dtype=torch.float32,
    device="cuda",
)
```

Or declarative:

```python
specs = [
    WorkspaceSpec("segm_output", (Q, H, S, D), torch.float32, "cuda"),
    WorkspaceSpec("segm_max",    (Q, H, S),    torch.float32, "cuda"),
    WorkspaceSpec("segm_expsum", (Q, H, S),    torch.float32, "cuda"),
]
tensors = pool.prepare(specs)
```

Behavior:

- slots are keyed by `name`;
- re-requesting with the same shape returns the same tensor;
- re-requesting with a smaller shape returns a view of the existing storage;
- re-requesting with a larger shape grows the underlying tensor in place;
- different `dtype` or `device` reallocates.

The pool fixes the workspace-lifetime race: `torch.empty(..., device=q.device)` returns to the caching allocator when the dispatch frame returns; raw HIP launches don't see that, so the allocator can hand the storage to another kernel mid-flight. Pool-owned tensors outlive the dispatch.

`WorkspacePool.required_nbytes(specs)` reports total spec bytes; `capacity_nbytes()` reports current physical capacity.

## DeviceMem

`runtime/launcher.py::DeviceMem(nbytes)` is RAII over `hipMalloc`/`hipFree` for non-torch flows. The `run_manifest` runner uses it to allocate the numpy-backed problem buffers.

## time_launches

```python
from ck_dsl.runtime.launcher import time_launches

ms = time_launches(
    lambda: launcher(values, config=cfg),
    warmup=5,
    iters=100,
    stream=0,
)
```

Returns mean per-call wall time in **milliseconds**. Internally:

1. `warmup` cold runs without timing.
2. `rt.sync()` to drain.
3. Record `e0` on the resolved stream.
4. Run `iters` calls under `no_fence()` (fire-and-forget, no per-launch event).
5. Record `e1`.
6. `e1.synchronize()`; `ms = e0.elapsed_to(e1) / iters`.
7. Drain `Runtime._pending_args[stream]` via `wait_stream`.

The timer is the only event-creating primitive in the runtime layer. Production dispatch does not branch on timing mode.

## Manifest Flow

File: `helpers/manifest.py`. Schema version: `ck.dsl.example.manifest/v1`. Full schema reference: `runtime/manifest_schema.md`.

Common entry points:

```text
gemm_args_signature(*, with_bytes=False)
conv_args_signature()
attention_args_signature(*, path="2d" | "reduce")

make_gemm_manifest(...)
make_conv_manifest(...)
make_attention_manifest(...)
make_simple_op_manifest(...)         # elementwise/reduce/norm/transpose

write_artifact(artifact, out_dir, manifest,
               write_ir_text=True, write_llvm_text=True)
```

`write_artifact` produces:

```text
<out_dir>/<kernel_name>.hsaco
<out_dir>/<kernel_name>.ir.txt     # MLIR-style debug dump
<out_dir>/<kernel_name>.ll         # AMDGPU LLVM IR text
<out_dir>/manifest.json
```

Runner: `python -m ck_dsl.run_manifest <hsaco> <manifest> [--shape MxNxK] [--verify]`. The runner allocates problem buffers, packs args from the signature, launches via `time_launches`, optionally verifies with a numpy / torch reference, and prints a `Perf: <ms>, <TFlops>, <GB/s>` line.

## Sweep Flow

`sweep.py` builds many specs in parallel, content-hash caches HSACO, and writes a sweep manifest. `sweep_bench.py` consumes the sweep manifest and benchmarks each entry with median + spread + CSV output. See `helpers/autotune.py::Autotuner` for the in-process autotuning API.

## Analysis Flow

```python
from ck_dsl import analyze_llvm_ir, analyze_hsaco

ir_stats   = analyze_llvm_ir(art.llvm_text)
hsaco_stats = analyze_hsaco(
    hsaco_path,
    objdump="/opt/rocm/llvm/bin/llvm-objdump",
    readelf="/opt/rocm/llvm/bin/llvm-readelf",
)
print(hsaco_stats.isa.as_dict())
print(hsaco_stats.resources.as_dict())
```

`LlvmIrStats` counts intrinsic occurrences (MFMA shapes, async LDS loads, vector loads/stores, waitcnt, barriers).

`HsacoAnalysis` carries `isa` (`IsaStats`: instruction counts from disassembly) and `resources` (`ResourceInfo`: VGPR, SGPR, static LDS bytes from ELF notes).

`compare_variant_reports(*reports)` compares two `VariantReport`s for a controlled-variant experiment.

## Common Patterns

### One-shot compile + inspect

```python
kernel = build_universal_gemm(spec)
art = compile_kernel(kernel)
print(f"codegen {art.timings['total']:.2f} ms, hsaco {art.hsaco_bytes} bytes")
```

### Persistent torch launch

```python
launcher = KernelLauncher(hsaco=art.hsaco, kernel_name=art.kernel_name,
                          signature=gemm_args_signature())
cfg = LaunchConfig(grid=(gx, gy, gz), block=(bs, 1, 1))
for _ in range(N):
    launcher({"A": A, "B": B, "C": C, "M": M, "N": N, "K": K}, config=cfg)
```

### Pipeline launch

```python
pipeline = PipelineLauncher([seg, red])
pipeline([seg_vals, red_vals], [seg_cfg, red_cfg], stream=0)
```

### Manifest execute

```bash
PYTHONPATH=python python -m ck_dsl.run_manifest out.hsaco manifest.json --verify
```

### Sweep + benchmark

```bash
PYTHONPATH=python python example/ck_tile/dsl/07_gemm_universal_sweep/gen.py \
    --output-dir /tmp/sweep --subset compute --parallel 16
PYTHONPATH=python python -m ck_dsl.sweep_bench /tmp/sweep/sweep_manifest.json \
    --attempts 3 --csv /tmp/sweep/results.csv
```

## Runtime Failure Modes

- Measuring compile time as kernel time (use `time_launches`, not wall-clock around `compile_kernel + launch`).
- Reloading HSACO on every benchmark iteration (construct `KernelLauncher` once).
- Launching on a different stream from torch producer tensors (use `stream=0` or pass torch's current stream handle).
- Packed args buffer freed too early (use `KernelLauncher` or `Runtime.retain_for_stream`).
- Workspace tensors freed or reallocated before pipeline completes (use `WorkspacePool`).
- Grid/block config in manifest does not match kernel attrs (`max_workgroup_size` must be >= block size).
- ABI byte size mismatch (`A_bytes` is i32 byte count, not element count).
- Benchmark compares graph/pipeline amortized timing with cold single launch.
