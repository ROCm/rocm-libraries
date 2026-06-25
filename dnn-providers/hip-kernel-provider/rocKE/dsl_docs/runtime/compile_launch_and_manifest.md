# Compile, Launch, And Manifest Runtime

This page follows a kernel from `KernelDef` through HSACO to a torch-aware launch.

## Compile Entry Point

File: `helpers/compile.py`.

```python
from rocke.helpers import compile_kernel
art = compile_kernel(
    kernel,
    arch="gfx950",                       # gfx942 / gfx950 / gfx1151 / gfx1201; takes precedence over isa
    isa="amdgcn-amd-amdhsa--gfx950",     # raw comgr triple, kept for back-compat (gfx950 default)
    capture_ir_text=True,
    optimize_ir=False,
    backend=None,                        # None -> ROCKE_BACKEND (default "cpp", falls back to native python)
) -> KernelArtifact
```

`backend` selects which engine produces the lowered AMDGPU `.ll`: `"python"` (native lowerer), `"cpp"` (the C++ engine binding), or `"both"` (lower with both and assert byte-equality). When unset it follows `core/backend.py::resolve_backend` — `ROCKE_BACKEND` env, else the package default `"cpp"`, which auto-falls back to the native Python lowerer when the `rocke_engine` binding isn't built. The two engines emit byte-identical IR.

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

The driver loads `libamd_comgr.so` via ctypes, preferring the
torch-bundled `<torch>/lib/libamd_comgr.so` when torch is imported (so
rocke and torch share one HIP runtime), then the default ROCm library
locations / dynamic linker search path (see `runtime/hip_module.py::_candidate_lib_paths`);
passes the IR text through
`COMPILE_SOURCE_TO_BC -> CODEGEN_BC_TO_RELOCATABLE -> LINK_RELOCATABLE_TO_EXECUTABLE`;
and extracts the executable bytes.

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
from rocke.runtime.launcher import KernelLauncher, LaunchConfig

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

## CK-Tile-style multi-kernel launch

`runtime/launcher.py` also exposes a low-level CK-Tile-shaped pair of primitives that mirror `ck_tile::launch_kernel` and `ck_tile::make_kernel` (`include/ck_tile/host/kernel_launch.hpp` lines 118-286) field-for-field. Use these when you want the C++ shape directly: each callable is a closure that has already baked in `(values, grid, block, lds_bytes)`, and `launch_kernel` is variadic over closures with optional cold-warmup + timed-iters wrapping.

```python
from rocke import StreamConfig, launch_kernel, make_kernel

# Production (no timing): runs each callable once on the stream
# under no_fence, returns 0.0. No implicit final sync -- the caller
# is responsible (via wait_stream_and_release or torch's stream-aware
# next-read).
launch_kernel(
    StreamConfig(stream_id=stream),
    make_kernel(seg_launcher, seg_vals, seg_grid, seg_block),
    make_kernel(red_launcher, red_vals, red_grid, red_block),
)

# Benchmark: cold_niters warmup + nrepeat timed iters wrapped by
# one HIP-event pair. Returns mean per-iteration ms.
ms = launch_kernel(
    StreamConfig(stream_id=stream, time_kernel=True,
                 cold_niters=5, nrepeat=100),
    make_kernel(seg_launcher, seg_vals, seg_grid, seg_block),
    make_kernel(red_launcher, red_vals, red_grid, red_block),
)
```

`StreamConfig` mirrors `ck_tile::stream_config`: `stream_id`, `time_kernel`, `log_level`, `cold_niters`, `nrepeat`, `is_gpu_timer`, `flush_cache`. `stream_id=0` auto-resolves to torch's current stream.

`make_kernel(launcher, values, grid, block, *, lds_bytes=0)` returns a `Callable[[StreamConfig], None]` that:

- Captures `values` via `dict(values)` and freezes `(grid, block, lds_bytes)` so caller mutations after closure construction do not affect the closure.
- Reads `stream_id` from its `StreamConfig` argument **at call time** (not at construction time), so the same closure can be replayed on different streams.
- Always launches with `fence=False`. The only sync points are the timing-loop boundary inside `launch_kernel` (when `time_kernel=True`), the caller's explicit drain, and torch's stream-aware next-read.

Bare host lambdas with the same `Callable[[StreamConfig], None]` shape compose freely with `make_kernel` closures -- mirrors the C++ `MOR_SORTING_MP_DISPATCH_` pattern at `example/ck_tile/15_fused_moe/instances/fused_moesorting_api.cpp:481`:

```python
def maybe_clear_workspace(s):
    Runtime().memset(ws_ptr, 0, ws_bytes, stream=s.stream_id)

launch_kernel(
    StreamConfig(stream_id=stream),
    maybe_clear_workspace,                 # bare lambda
    make_kernel(seg_launcher, seg_vals, ...),
    make_kernel(red_launcher, red_vals, ...),
)
```

### When to use which

| Need | Use |
| --- | --- |
| Fixed stage list per problem-shape, hot dispatch, implicit last-stage fence | `PipelineLauncher` |
| One-shot benchmark drivers, autotuner harnesses, mixed kernel + host callables | `launch_kernel` + `make_kernel` |
| Group of timed launches reported as a single ms (cold+warmup wraps the whole group) | `launch_kernel(StreamConfig(time_kernel=True, ...))` |
| Fire-and-forget single-launch | `KernelLauncher(values, config=cfg)` directly |

### Migration note for `PipelineLauncher` callers

`PipelineLauncher` honours `cfg.fence` on the **last** stage and therefore implicitly fences the host on pipeline completion. `launch_kernel` does not -- the closures returned by `make_kernel` are always `fence=False`, and `launch_kernel` itself never inserts an implicit final sync. Code paths that read an output tensor on the host immediately after `pipeline(...)` returns relied on the last-stage fence. When migrating to `launch_kernel`, add an explicit `wait_stream_and_release(stream)` (or rely on torch's stream-aware next-read sync) before the host read.

### Macro-style shorthand convention for instance authors

CK Tile's `MOE_SORTING_MP_*` macros at `example/ck_tile/15_fused_moe/instances/fused_moesorting_api.cpp` lines 201-309 each expand to a self-invoking lambda that returns a `make_kernel` closure. The Python equivalent for instance authors is a per-phase factory function:

```python
def moe_sort_histogram_callable(spec, args, *, launcher=None):
    """CK-style per-phase callable factory for launch_kernel(s, ...)."""
    if launcher is None:
        launcher = _get_or_build_hist_launcher(spec)   # cached per spec
    grid  = moe_sort_histogram_grid(spec, args.num_tokens)
    block = (spec.block_size, 1, 1)
    return make_kernel(launcher, args.histogram_values(), grid, block,
                       lds_bytes=moe_sort_histogram_lds(spec))
```

A CK-style chained launch then reads as:

```python
launch_kernel(
    StreamConfig(stream_id=stream),
    moe_sort_histogram_callable(spec, args),
    moe_sort_scan_callable(spec, args),
    moe_sort_scatter_callable(spec, args),
)
```

This is the recipe for converting any multi-phase instance (moe_sorting's three builders, fused_moe's gather / silu_mul / topk_weighted_reduce, etc.) into a CK-style chain.

## WorkspacePool

```python
from rocke.runtime.launcher import WorkspacePool, WorkspaceSpec

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
from rocke.runtime.launcher import time_launches

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

Runner: `python -m rocke.run_manifest <hsaco> <manifest> [--shape M,N,K] [--verify]`. The runner allocates problem buffers, packs args from the signature, launches via `time_launches`, optionally verifies with a numpy / torch reference, and prints a `Perf: <ms>, <TFlops>, <GB/s>` line.

## Sweep Flow

`sweep.py` builds many specs in parallel, content-hash caches HSACO, and writes a sweep manifest. `sweep_bench.py` consumes the sweep manifest and benchmarks each entry with median + spread + CSV output. See `helpers/autotune.py::Autotuner` for the in-process autotuning API.

## Analysis Flow

```python
from rocke import analyze_llvm_ir, analyze_hsaco

ir_stats   = analyze_llvm_ir(art.llvm_text)
hsaco_stats = analyze_hsaco(
    hsaco_path,
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

### CK-style multi-kernel launch

```python
from rocke import StreamConfig, launch_kernel, make_kernel

ms = launch_kernel(
    StreamConfig(stream_id=stream, time_kernel=True,
                 cold_niters=5, nrepeat=100),
    make_kernel(seg, seg_vals, seg_grid, seg_block),
    make_kernel(red, red_vals, red_grid, red_block),
)
```

### Manifest execute

```bash
PYTHONPATH=Python python -m rocke.run_manifest out.hsaco manifest.json --verify
```

### Sweep + benchmark

```bash
OUT_DIR="${OUT_DIR:-$(mktemp -d)}"
PYTHONPATH=Python python example/ck_tile/dsl/07_gemm_universal_sweep/gen.py \
    --output-dir "$OUT_DIR" --subset compute --parallel 16
PYTHONPATH=Python python -m rocke.sweep_bench "$OUT_DIR"/sweep_manifest.json \
    --attempts 3 --csv "$OUT_DIR"/results.csv
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
