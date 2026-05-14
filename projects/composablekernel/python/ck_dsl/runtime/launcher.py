# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Long-lived launcher abstractions for CK DSL kernels.

Why this exists
---------------

CK DSL kernels are produced as HSACO blobs compiled in-process and
launched through a thin ctypes wrapper over `hipModuleLaunchKernel`
(see :mod:`ck_dsl.runtime.hip_module`). The naive launch path
("compile, load, launch, unload") has *correctness* problems on top of
the obvious performance problems:

1.  **Workspace lifetime race.** Multi-kernel pipelines like
    split-KV attention need intermediate buffers (segm_*). If the
    caller allocates them with `torch.empty(...)` and drops them
    when the dispatcher returns, torch's caching allocator can
    recycle the storage while a second kernel is still reading it.
    CK Tile's `fmha_bwd_launcher` solves this by allocating
    workspace **once** at launcher-construction time and reusing it
    across every call -- the workspace outlives every launch.

2.  **Module reload tax.** Reloading the same HSACO module per
    call costs ~500us on ROCm and exposes us to module-cache
    aliasing if we re-use Python object ids. CK Tile compiles
    once per problem shape and caches the function handle; we
    should do the same.

3.  **Stream / allocator desync.** A `torch.empty(..., device=q.device)`
    workspace allocation is tagged with torch's *current* stream
    in the caching allocator. If we then launch on raw HIP stream 0
    (legacy default), the allocator never sees our launch and may
    free the workspace prematurely. Resolution: launches always go
    on `torch.cuda.current_stream().cuda_stream` unless the caller
    asked otherwise.

4.  **Packed-args ABI race.** The `HIP_LAUNCH_PARAM_BUFFER_POINTER`
    ("extra") launch path does **not** promise to copy the packed
    args buffer at enqueue time. The GPU command processor has
    been observed reading the buffer *after*
    `hipModuleLaunchKernel` returns. Resolution: keep every
    launch's args ctypes buffer alive in
    `Runtime._pending_args[stream]` until the stream is sync'd.
    See :mod:`ck_dsl.runtime.hip_module` for the runtime side.

5.  **Cross-instance cache-key collisions.** Keying the loaded
    module cache by `id(hsaco_bytes)` is fragile -- Python may
    re-use object ids across distinct bytes objects. The launcher
    keys its cache by a *semantic* key supplied at construction.

This module supplies three small primitives that, together, fix
all five categories above by construction:

* :class:`KernelLauncher`: owns one compiled-and-loaded HSACO module,
  packs and issues args for one kernel, manages stream + args lifetime.

* :class:`PipelineLauncher`: a sequence of :class:`KernelLauncher`\\s
  that share a stream (and optionally shared workspace), launched
  one-after-the-other on the same stream. Mirrors CK Tile's
  ``ck_tile::launch_kernel(stream_config, k0, k1)`` chained-callable
  semantics for split-KV / segment-then-reduce pipelines.

* :class:`WorkspacePool`: a thin owner of named torch workspace
  tensors keyed by (shape, dtype, device). Lazily allocates and
  reuses across launches; equivalent to CK Tile's
  ``DeviceMem ws_buf(ws_size);`` held over the whole problem
  lifetime.

Each kernel-emitting instance (attention 2D, attention 3D, GEMM,
conv, ...) should construct exactly one :class:`KernelLauncher`
per (problem-shape, problem-dtype) tuple at first-use time, cache
it on the dispatch side, and call ``launcher(values, stream=...)``
on every subsequent dispatch.

A migration of ``attention_unified`` to use this API is the
in-tree reference (see ``_get_3d_pipeline``, ``_get_2d_launcher``,
and ``_get_scalar_launcher`` in
``ck_dsl/instances/attention_unified.py``); the same template
applies to ``gemm``, ``grouped_gemm``, ``conv``, and any future op.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from .hip_module import Runtime
from .torch_module import pack_args, resolve_stream

__all__ = [
    "DeviceMem",
    "KernelLauncher",
    "LaunchConfig",
    "LaunchSummary",
    "PipelineLauncher",
    "WorkspacePool",
    "time_launches",
]


@dataclass(frozen=True)
class LaunchSummary:
    """Returned by every launcher call. Currently just records the
    number of launches that were issued; per-launch wall time should
    be measured by the *caller* using `torch.cuda.Event` so the
    launcher itself stays free of timing-mode branching.
    """

    launches: int


@dataclass
class LaunchConfig:
    """Per-call options for a launcher.

    The values here are the *only* knobs that affect a launch; the
    launcher's own state (HSACO, function handle, workspace,
    signature) is immutable after construction.
    """

    stream: int = 0
    """HIP stream handle. ``0`` is auto-resolved to
    ``int(torch.cuda.current_stream().cuda_stream)`` via
    :func:`resolve_stream` so torch's caching allocator can see the
    launch. Pass an explicit non-zero handle (or
    ``torch.cuda.current_stream().cuda_stream`` itself) to override.
    """

    grid: Tuple[int, int, int] = (1, 1, 1)
    """3D launch grid (number of CTAs in each dim)."""

    block: Tuple[int, int, int] = (64, 1, 1)
    """3D block dim (threads per CTA). Default is a single wave64."""

    shared_bytes: int = 0
    """Dynamic LDS bytes requested at launch (in addition to the
    kernel's statically-declared LDS)."""


# Module-global Runtime. There's no per-Runtime state worth instancing
# (everything lives on HIP itself); subclassing Runtime to add hooks
# would still share the singleton.
_HIP_RUNTIME: Optional[Runtime] = None


def _runtime() -> Runtime:
    global _HIP_RUNTIME
    if _HIP_RUNTIME is None:
        _HIP_RUNTIME = Runtime()
    return _HIP_RUNTIME


class KernelLauncher:
    """Owns one compiled HSACO module + one kernel function entry point.

    Construct **once** per (problem-shape, problem-dtype). The HIP
    module is loaded eagerly (in ``__init__``) and held on the
    instance for the lifetime of the launcher; the underlying
    `Module._blob` reference keeps the HSACO bytes alive so the
    loaded code object cannot be reclaimed.

    Calling the launcher::

        launcher(values, config=LaunchConfig(grid=..., block=..., stream=...))

    packs ``values`` against the launcher's signature, resolves the
    stream to a torch-tracked handle, and issues a single bare
    ``hipModuleLaunchKernel``. The packed args ctypes buffer is
    handed off to :class:`Runtime` (which keeps it alive via
    :attr:`Runtime._pending_args` until the stream is sync'd).

    The launcher does **not** sync inside the call. The caller (or
    a downstream `torch.cuda.Event.synchronize()`) is responsible
    for observing the output. This matches Triton's launch semantics
    and lets a benchmarking harness do its own event-based timing.
    """

    def __init__(
        self,
        *,
        hsaco: bytes,
        kernel_name: str,
        signature: Sequence[Mapping[str, Any]],
        cache_key: Optional[Tuple] = None,
    ) -> None:
        self._hsaco = hsaco
        self._kernel_name = kernel_name
        self._signature = list(signature)
        self._cache_key = cache_key
        rt = _runtime()
        self._module = rt.load_module(hsaco)
        self._fn = self._module.get_function(kernel_name)

    @property
    def kernel_name(self) -> str:
        return self._kernel_name

    @property
    def signature(self) -> Sequence[Mapping[str, Any]]:
        return tuple(self._signature)

    def __call__(
        self,
        values: Mapping[str, Any],
        *,
        config: LaunchConfig,
    ) -> LaunchSummary:
        rt = _runtime()
        args = pack_args(self._signature, values)
        stream = resolve_stream(config.stream)
        rt.launch(
            self._fn,
            config.grid,
            config.block,
            args,
            shared_bytes=config.shared_bytes,
            stream=stream,
        )
        return LaunchSummary(launches=1)

    def __repr__(self) -> str:
        key_str = f", cache_key={self._cache_key!r}" if self._cache_key else ""
        return f"KernelLauncher({self._kernel_name!r}{key_str})"


class PipelineLauncher:
    """A sequence of :class:`KernelLauncher`\\s that share a stream.

    Mirrors CK Tile's
    ``ck_tile::launch_kernel(stream_config, k0, k1, ...)`` chained
    callable form: all kernels run on the same ``stream_id_``, in
    declaration order, with same-stream FIFO ordering as the only
    correctness primitive (no events, no record_stream needed for
    correctness -- only for timing).

    Each kernel has its own ``LaunchConfig`` (grid, block, shared
    bytes), but ``stream`` is forced to a single value for the whole
    pipeline. Use this for split-KV attention (segment + reduce),
    grouped-GEMM (k-fixup), conv (im2col then GEMM then col2im),
    etc.
    """

    def __init__(self, launchers: Sequence[KernelLauncher]) -> None:
        if not launchers:
            raise ValueError("PipelineLauncher requires at least one stage")
        self._stages = tuple(launchers)

    @property
    def stages(self) -> Tuple[KernelLauncher, ...]:
        return self._stages

    def __call__(
        self,
        values_per_stage: Sequence[Mapping[str, Any]],
        configs_per_stage: Sequence[LaunchConfig],
        *,
        stream: int = 0,
    ) -> LaunchSummary:
        if len(values_per_stage) != len(self._stages):
            raise ValueError(
                f"PipelineLauncher: got {len(values_per_stage)} values "
                f"but pipeline has {len(self._stages)} stages"
            )
        if len(configs_per_stage) != len(self._stages):
            raise ValueError(
                f"PipelineLauncher: got {len(configs_per_stage)} configs "
                f"but pipeline has {len(self._stages)} stages"
            )
        resolved_stream = resolve_stream(stream)
        total = 0
        for stage, vals, cfg in zip(self._stages, values_per_stage, configs_per_stage):
            # Force the same stream across all stages -- this is the
            # whole point of a pipeline launcher; we override the
            # per-stage config's stream field.
            stage_cfg = LaunchConfig(
                stream=resolved_stream,
                grid=cfg.grid,
                block=cfg.block,
                shared_bytes=cfg.shared_bytes,
            )
            s = stage(vals, config=stage_cfg)
            total += s.launches
        return LaunchSummary(launches=total)


@dataclass
class _Slot:
    """One named workspace slot inside a :class:`WorkspacePool`."""

    name: str
    tensor: Any  # torch.Tensor; not typed to avoid the import at module-load
    shape: Tuple[int, ...]
    dtype: Any
    device: Any


class WorkspacePool:
    """Long-lived owner of named torch workspace tensors.

    Solves the workspace-lifetime race described in the module
    docstring: a tensor returned by :meth:`get` is owned by the
    pool, not by the caller, so it survives the dispatch function's
    stack frame. The next dispatch (or the next iteration of a
    benchmark timing loop) sees the same tensor at the same address
    and re-uses it directly -- no allocation, no torch caching-
    allocator race.

    Slots are keyed by ``name``. Re-requesting a slot with the same
    name but a larger shape grows the underlying tensor in place
    (lazy realloc); a smaller shape just returns a view of the
    existing buffer. Re-requesting with a different ``dtype`` or
    ``device`` reallocates.

    The pool is the natural place to hang onto split-KV / segment-
    reduce intermediates in the attention dispatcher: one pool per
    cached :class:`KernelLauncher`.
    """

    def __init__(self) -> None:
        self._slots: Dict[str, _Slot] = {}

    def get(
        self,
        name: str,
        shape: Sequence[int],
        *,
        dtype: Any,
        device: Any,
    ) -> Any:
        import torch  # local to keep ck_dsl.core import-time torch-free

        shape_t = tuple(int(x) for x in shape)
        nbytes_needed = 1
        for s in shape_t:
            nbytes_needed *= s
        if name in self._slots:
            slot = self._slots[name]
            same_dtype = slot.dtype == dtype
            same_device = slot.device == device
            existing_elems = 1
            for s in slot.shape:
                existing_elems *= s
            if same_dtype and same_device and existing_elems >= nbytes_needed:
                if slot.shape == shape_t:
                    return slot.tensor
                # Reshape view into the existing allocation.
                return slot.tensor.flatten()[:nbytes_needed].view(shape_t)
            # Outgrow or change dtype/device: drop the old slot. The
            # tensor's storage is freed by torch when the slot's
            # reference dies (and any pending kernel keeps the
            # underlying memory alive via Runtime._pending_args +
            # same-stream FIFO).
            del self._slots[name]
        t = torch.empty(shape_t, dtype=dtype, device=device)
        self._slots[name] = _Slot(name, t, shape_t, dtype, device)
        return t

    def drop(self, name: str) -> None:
        self._slots.pop(name, None)

    def clear(self) -> None:
        self._slots.clear()

    def __contains__(self, name: str) -> bool:
        return name in self._slots

    def __repr__(self) -> str:
        return f"WorkspacePool(slots={list(self._slots)})"


class DeviceMem:
    """RAII over :meth:`Runtime.alloc` / :meth:`Runtime.free`.

    The numpy-flavored ``run_manifest`` benchmark path doesn't have a
    torch caching allocator to lean on, so it allocates raw device
    memory via ``hipMalloc``. The naive style (``ptr = rt.alloc(n); ...
    rt.free(ptr)``) is leak-prone (every exception path or early
    return needs an explicit free) and exactly the kind of bookkeeping
    CK Tile's ``ck_tile::DeviceMem`` was designed to encapsulate.

    Construct with a byte size; the buffer is freed when this object
    is garbage-collected or when the caller goes out of scope. Use
    :meth:`ptr` to fetch the raw device pointer (an ``int``) to pass
    into kernel-arg packing.

    For nbytes==0 the constructor is a no-op (``ptr() == 0``); this
    matches CK Tile's contract for optional workspaces.
    """

    def __init__(self, nbytes: int) -> None:
        self._nbytes = int(nbytes)
        if self._nbytes > 0:
            self._ptr = _runtime().alloc(self._nbytes)
        else:
            self._ptr = 0

    def ptr(self) -> int:
        return self._ptr

    def nbytes(self) -> int:
        return self._nbytes

    def realloc(self, nbytes: int) -> None:
        """Drop the current buffer and allocate ``nbytes`` instead."""
        if self._ptr:
            _runtime().free(self._ptr)
            self._ptr = 0
        self._nbytes = int(nbytes)
        if self._nbytes > 0:
            self._ptr = _runtime().alloc(self._nbytes)

    def __del__(self) -> None:
        try:
            if getattr(self, "_ptr", 0):
                _runtime().free(self._ptr)
        except Exception:
            # Garbage-collection-time errors are uncatchable by callers
            # and HIP may not even have a context anymore (interpreter
            # teardown), so swallow them.
            pass
        self._ptr = 0

    def __repr__(self) -> str:
        return f"DeviceMem(ptr=0x{self._ptr:x}, nbytes={self._nbytes})"


def time_launches(
    fn: Callable[[], None],
    *,
    warmup: int = 5,
    iters: int = 100,
    stream: int = 0,
) -> float:
    """Benchmark ``fn`` (which is expected to issue one or more
    launches on ``stream``) using HIP events, returning average
    per-call wall time in milliseconds.

    Equivalent to CK Tile's ``gpu_timer`` / Triton's autotuner
    timing loop. Does NOT recompile or reload modules: ``fn`` should
    capture whatever :class:`KernelLauncher` or :class:`PipelineLauncher`
    you want to measure and just call it.

    Replaces both ``run_manifest._launch_timed`` (which built its own
    event scaffolding around `rt.launch`) and the timing branch of
    ``launch_torch_kernel`` (which also reloaded the module per call
    on top of timing -- a separate bug). Keeping the timer out of the
    launcher lets the launcher stay overhead-free for production code
    paths that aren't benchmarking.
    """
    rt = _runtime()
    for _ in range(int(warmup)):
        fn()
    rt.sync()
    resolved = resolve_stream(stream)
    e0 = rt.event()
    e1 = rt.event()
    e0.record(stream=resolved)
    for _ in range(int(iters)):
        fn()
    e1.record(stream=resolved)
    e1.synchronize()
    ms = e0.elapsed_to(e1) / int(iters)
    e0.destroy()
    e1.destroy()
    return ms
