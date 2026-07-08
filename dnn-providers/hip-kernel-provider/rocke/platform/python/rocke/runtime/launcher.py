# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Long-lived launcher abstractions for CK DSL kernels.

Why this exists
---------------

CK DSL kernels are produced as HSACO blobs compiled in-process and
launched through a thin ctypes wrapper over `hipModuleLaunchKernel`
(see :mod:`rocke.runtime.hip_module`). The naive launch path
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
    See :mod:`rocke.runtime.hip_module` for the runtime side.

5.  **Cross-instance cache-key collisions.** Keying the loaded
    module cache by `id(hsaco_bytes)` is fragile -- Python may
    re-use object ids across distinct bytes objects. The launcher
    keys its cache by a *semantic* key supplied at construction.

This module supplies a small set of primitives that, together, fix
the categories above by construction:

* :class:`KernelLauncher`: owns one compiled-and-loaded HSACO module,
  packs and issues args for one kernel, manages stream + args lifetime.

* :class:`PipelineLauncher`: a sequence of :class:`KernelLauncher`\\s
  that share a stream (and optionally shared workspace), launched
  one-after-the-other on the same stream. Mirrors CK Tile's
  ``ck_tile::launch_kernel(stream_config, k0, k1)`` chained-callable
  semantics for split-KV / segment-then-reduce pipelines.

* :class:`WorkspaceSpec` + :class:`WorkspacePool`: declarative
  workspace sizing plus a thin owner of named torch workspace tensors
  keyed by (shape, dtype, device). Lazily allocates and reuses across
  launches; equivalent to CK Tile's ``workspace_size`` plus
  ``DeviceMem ws_buf(ws_size);`` held over the whole problem lifetime.

* :class:`DeviceMem`: RAII over `hipMalloc`/`hipFree` for numpy /
  manifest flows that do not use torch tensors.

* :func:`time_launches`: the event-timing loop, kept separate from
  launchers so production dispatch has no benchmarking branches.

Each kernel-emitting instance (attention 2D, attention 3D, GEMM,
conv, ...) should construct exactly one :class:`KernelLauncher`
per (problem-shape, problem-dtype) tuple at first-use time, cache
it on the dispatch side, and call ``launcher(values, stream=...)``
on every subsequent dispatch.

A migration of ``attention_unified`` to use this API is the
in-tree reference (see ``_get_3d_pipeline``, ``_get_2d_launcher``,
and ``_get_scalar_launcher`` in
``rocke/instances/common/attention_unified.py``); the same template
applies to ``gemm``, ``grouped_gemm``, ``conv``, and any future op.
"""

from __future__ import annotations

import contextvars
import time as _time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, Mapping, Optional, Sequence, Tuple

from .hip_module import Runtime
from .torch_module import pack_args, resolve_stream

__all__ = [
    "DeviceMem",
    "KernelLauncher",
    "LaunchConfig",
    "LaunchSummary",
    "PipelineLauncher",
    "StreamConfig",
    "WorkspaceSpec",
    "WorkspacePool",
    "launch_kernel",
    "make_kernel",
    "no_fence",
    "release_retained_for_stream",
    "synchronize_and_release",
    "time_launches",
]


# ---------------------------------------------------------------------
# Launch-fence override
# ---------------------------------------------------------------------
#
# ``LaunchConfig.fence`` controls whether a single ``KernelLauncher``
# call event-synchronizes on its launch's completion before returning
# (mirroring CK Tile's ``launch_kernel`` contract, which always ends
# with ``hipStreamSynchronize``). Batch wrappers like
# :func:`time_launches` need to suppress that per-call sync to time
# many launches inside one outer event-timed region. They do so via
# the :func:`no_fence` context manager, which forces the resolved
# fence to ``False`` for any ``KernelLauncher`` call made within its
# scope -- even ones whose ``LaunchConfig.fence`` is True.
_fence_override: "contextvars.ContextVar[Optional[bool]]" = contextvars.ContextVar(
    "rocke_launcher_fence_override", default=None
)


@contextmanager
def no_fence() -> "Iterator[None]":
    """Context: every :class:`KernelLauncher` call inside is fire-and-forget.

    Use only when the surrounding code does its own stream/event sync
    (e.g. :func:`time_launches` records start/end events around an
    iteration block). Outside this context the default
    ``LaunchConfig.fence=True`` policy applies and every launcher
    call event-synchronizes before returning.
    """
    token = _fence_override.set(False)
    try:
        yield
    finally:
        _fence_override.reset(token)


def _resolved_fence(config_fence: bool) -> bool:
    """Combine the per-call ``LaunchConfig.fence`` with the active
    :func:`no_fence` override. The override wins when set."""
    override = _fence_override.get()
    return bool(config_fence) if override is None else bool(override)


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

    fence: bool = True
    """Event-synchronize on this launch's completion before returning.

    Mirrors CK Tile's ``launch_kernel`` contract: every launch is
    paired with a stream-bound synchronization primitive
    (``hipEventSynchronize`` here, ``hipStreamSynchronize`` there) so
    the host never observes a half-finished kernel and tensors /
    args buffers can be released immediately on return.

    Why default ``True``: on ROCm, ``torch.cuda.synchronize()`` does
    not reliably drain raw ``hipModuleLaunchKernel`` work queued
    through ctypes, so a fire-and-forget launch followed by a torch
    sync is unsafe (the host may observe an output buffer that the
    kernel has not yet written to). A per-launch HIP event is the
    minimum safe primitive.

    Set ``fence=False`` only when the caller wraps multiple launches
    in an outer event-timed region (e.g. :func:`time_launches` or a
    multi-stage pipeline that ends with its own
    :meth:`Runtime.wait_stream`). The :func:`no_fence` context
    manager forces this off for any nested launcher call regardless
    of the per-call value.
    """


@dataclass(frozen=True)
class StreamConfig:
    """CK-Tile-shaped stream configuration for :func:`launch_kernel`.

    Field-for-field mirror of ``ck_tile::stream_config`` in
    ``include/ck_tile/host/stream_config.hpp``: one stream handle
    plus a small bag of timing knobs that select the
    non-timing / GPU-event-timer / CPU-timer dispatch path inside
    :func:`launch_kernel`.

    Field semantics
    ---------------
    ``stream_id`` -- HIP stream handle (``int``). ``0`` (the default)
    auto-resolves to ``int(torch.cuda.current_stream().cuda_stream)``
    via :func:`resolve_stream` inside the closure produced by
    :func:`make_kernel`, so torch's caching allocator sees the launch.
    The handle is read at call time, not closure-construction time, so
    a single closure can be replayed on different streams.

    ``time_kernel`` -- when False, :func:`launch_kernel` is a thin
    "submit each callable in order under :func:`no_fence`" wrapper and
    returns ``0.0``. When True, the callables are run for
    ``cold_niters`` warmup iterations followed by ``nrepeat`` timed
    iterations, and :func:`launch_kernel` returns the average wall
    time per iteration in milliseconds.

    ``cold_niters`` / ``nrepeat`` -- analogous to CK Tile's
    ``cold_niters_`` / ``nrepeat_``. Warmup iterations run the same
    callable list under :func:`no_fence` but their wall time is not
    measured.

    ``is_gpu_timer`` -- when True (default), the timed loop is wrapped
    by two :class:`Event` records on ``stream_id`` and the elapsed
    time is computed via ``hipEventElapsedTime``. When False, an
    end-to-end ``Runtime.wait_stream`` + :func:`time.perf_counter`
    pair is used (CPU timer; less precise but does not require a
    valid HIP context for the timer itself).

    ``log_level`` and ``flush_cache`` are accepted for parity with
    the C++ struct; v1 of this primitive ignores them. They are
    documented here so callers writing portable benchmark drivers
    can construct one ``StreamConfig`` for both the C++ and Python
    paths without per-language massaging.
    """

    stream_id: int = 0
    time_kernel: bool = False
    log_level: int = 0
    cold_niters: int = 3
    nrepeat: int = 10
    is_gpu_timer: bool = True
    flush_cache: bool = False


@dataclass(frozen=True)
class WorkspaceSpec:
    """Declarative workspace requirement, CK Tile-style.

    This is the Python analogue of CK Tile's `workspace_size` contract:
    the op declares the exact named workspaces it needs (shape + dtype +
    device), and a `WorkspacePool` turns those specs into long-lived
    tensors. The spec exposes `numel()` and `nbytes()` so callers can
    report and validate total scratch usage before launching.
    """

    name: str
    shape: Tuple[int, ...]
    dtype: Any
    device: Any

    def numel(self) -> int:
        n = 1
        for dim in self.shape:
            n *= int(dim)
        return int(n)

    def nbytes(self) -> int:
        return self.numel() * _dtype_element_size(self.dtype)


# Module-global Runtime. There's no per-Runtime state worth instancing
# (everything lives on HIP itself); subclassing Runtime to add hooks
# would still share the singleton.
_HIP_RUNTIME: Optional[Runtime] = None


def _runtime() -> Runtime:
    global _HIP_RUNTIME
    if _HIP_RUNTIME is None:
        _HIP_RUNTIME = Runtime()
    return _HIP_RUNTIME


def release_retained_for_stream(stream: int = 0) -> None:
    """Drop retained args/tensors for a stream after external synchronization.

    Use this when the caller has already synchronized the stream via
    some other mechanism (an external HIP event sync,
    ``hipStreamSynchronize``, etc.) and only needs the bucket
    bookkeeping to be cleared. For the common case of "wait and then
    release", prefer :func:`wait_stream_and_release` which does both
    in one event-based call.
    """
    _runtime().release_pending_for_stream(resolve_stream(stream))


def wait_stream_and_release(stream: int = 0) -> None:
    """Event-synchronize on ``stream`` and release all retained refs.

    The CK Tile-shaped per-stream drain: equivalent to
    ``hipEventSynchronize`` on the stream's most-recent launch
    event, followed by destroying every event in the stream's
    bucket. This is the *correct* primitive on ROCm for raw HIP
    launches queued through ctypes -- ``torch.cuda.synchronize()``
    does not reliably drain that queue.

    Use this when isolating a single stream's work in a benchmark
    harness or between dispatcher lanes; use
    :func:`synchronize_and_release` for whole-device drain.
    """
    _runtime().wait_stream(resolve_stream(stream))


def synchronize_and_release(stream: int = 0) -> None:
    """Synchronize the device and release all retained launch resources.

    Device-wide drain (``hipDeviceSynchronize`` + ref release). Safe
    when the caller is on the legacy HIP null stream or has work
    spread across multiple streams. Benchmark harnesses call this
    between independent lanes (e.g. Triton 2D -> CK 2D -> Triton 3D
    -> CK 3D) when they want strong isolation rather than maximum
    overlap.

    Prefer :func:`wait_stream_and_release` when you know the target
    stream -- a per-stream event wait avoids stalling unrelated work.
    """
    _runtime().sync()


def _dtype_element_size(dtype: Any) -> int:
    """Return element size in bytes for a torch/numpy-like dtype."""
    if hasattr(dtype, "itemsize"):
        return int(dtype.itemsize)
    try:
        import torch

        return int(torch.empty((), dtype=dtype).element_size())
    except Exception:
        # Common torch dtype repr fallback for docs/tests that don't import torch.
        name = str(dtype)
        if any(x in name for x in ("float16", "bfloat16", "int16")):
            return 2
        if any(x in name for x in ("float32", "int32")):
            return 4
        if any(x in name for x in ("float64", "int64")):
            return 8
        raise TypeError(f"cannot determine element size for dtype {dtype!r}")


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
        fence = _resolved_fence(config.fence)

        if fence:
            # Synchronous-launch fast path -- mirrors CK Tile's
            # ``launch_kernel`` (single ``hipStreamSynchronize`` after
            # the launch). Empirical cost on ROCm 7: ~0.3 us per call
            # vs ~43 us for an event-based fence. The stream sync is
            # both the GPU completion wait and the host-side
            # args-buffer-drain barrier; nothing parks in
            # :attr:`Runtime._pending_args` for fenced launches, so
            # there's no bucket growth and no need to attach tensor
            # refs to a completion event.
            rt.launch_blocking(
                self._fn,
                config.grid,
                config.block,
                args,
                shared_bytes=config.shared_bytes,
                stream=stream,
            )
            return LaunchSummary(launches=1)

        # Asynchronous path (e.g. inside :func:`time_launches` under
        # the :func:`no_fence` override): retain refs in
        # :attr:`Runtime._pending_args` with no per-launch event.
        # The caller's outer ``time_launches`` event timer (or a later
        # :meth:`Runtime.wait_stream` / :func:`synchronize_and_release`)
        # is the drain point. ``wait_stream`` uses
        # ``hipStreamSynchronize`` so per-launch events are dead weight
        # in this path -- skipping them is the difference between ~0.3
        # us and ~1 us per launch in tight benchmark loops, and
        # multiple-percent on tiny kernels like the conv bake-off.
        rt.launch(
            self._fn,
            config.grid,
            config.block,
            args,
            shared_bytes=config.shared_bytes,
            stream=stream,
            record_event=False,
        )
        rt.retain_for_stream(stream, *values.values())
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
        n_stages = len(self._stages)
        total = 0
        for idx, (stage, vals, cfg) in enumerate(
            zip(self._stages, values_per_stage, configs_per_stage)
        ):
            # Force the same stream across all stages -- this is the
            # whole point of a pipeline launcher; we override the
            # per-stage config's stream field.
            #
            # Same-stream FIFO ordering already guarantees stage N+1
            # observes stage N's writes, so intermediate stages do NOT
            # need to fence -- a per-stage ``hipEventSynchronize`` would
            # serialize the host on every stage and defeat the whole
            # purpose of chaining. Only the LAST stage honors
            # ``cfg.fence`` so the host sees a fully-finished pipeline
            # on return.
            is_last = idx == n_stages - 1
            stage_cfg = LaunchConfig(
                stream=resolved_stream,
                grid=cfg.grid,
                block=cfg.block,
                shared_bytes=cfg.shared_bytes,
                fence=bool(cfg.fence) and is_last,
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
    capacity_numel: int
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
    (lazy realloc); a smaller shape returns a view of the existing
    allocation. Re-requesting with a different ``dtype`` or ``device``
    reallocates. The pool tracks capacity separately from the requested
    shape, so size accounting is explicit and stable.

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
        import torch  # local to keep rocke.core import-time torch-free

        shape_t = tuple(int(x) for x in shape)
        nbytes_needed = 1
        for s in shape_t:
            nbytes_needed *= s
        required_numel = int(nbytes_needed)
        if name in self._slots:
            slot = self._slots[name]
            same_dtype = slot.dtype == dtype
            same_device = slot.device == device
            if same_dtype and same_device and slot.capacity_numel >= required_numel:
                if slot.shape == shape_t:
                    return slot.tensor
                # Reshape view into the existing allocation.
                return slot.tensor.flatten()[:required_numel].view(shape_t)
            # Outgrow or change dtype/device: drop the old slot. The
            # tensor's storage is freed by torch when the slot's
            # reference dies (and any pending kernel keeps the
            # underlying memory alive via Runtime._pending_args +
            # same-stream FIFO).
            del self._slots[name]
        t = torch.empty(shape_t, dtype=dtype, device=device)
        self._slots[name] = _Slot(name, t, shape_t, required_numel, dtype, device)
        return t

    def get_spec(self, spec: WorkspaceSpec) -> Any:
        return self.get(spec.name, spec.shape, dtype=spec.dtype, device=spec.device)

    def prepare(self, specs: Sequence[WorkspaceSpec]) -> Dict[str, Any]:
        """Allocate/reuse every spec and return a name->tensor mapping."""
        return {spec.name: self.get_spec(spec) for spec in specs}

    @staticmethod
    def required_nbytes(specs: Sequence[WorkspaceSpec]) -> int:
        return sum(spec.nbytes() for spec in specs)

    def capacity_nbytes(self) -> int:
        total = 0
        for slot in self._slots.values():
            total += slot.capacity_numel * _dtype_element_size(slot.dtype)
        return total

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

    Internally runs ``fn`` under :func:`no_fence` so each launcher
    call inside the timed iteration stays fire-and-forget; the two
    outer events record start/end and the trailing
    ``hipEventSynchronize`` is the single drain point that bounds
    the elapsed-time measurement. After timing, the bucket retained
    for ``stream`` is reaped via :meth:`Runtime.wait_stream`.
    """
    rt = _runtime()
    resolved = resolve_stream(stream)
    with no_fence():
        for _ in range(int(warmup)):
            fn()
        rt.sync()
        e0 = rt.event()
        e1 = rt.event()
        try:
            e0.record(stream=resolved)
            for _ in range(int(iters)):
                fn()
            e1.record(stream=resolved)
            e1.synchronize()
            ms = e0.elapsed_to(e1) / int(iters)
        finally:
            # Destroy both events even if ``fn`` raised mid-loop, so a failed
            # timing run does not leak two HIP events.
            e0.destroy()
            e1.destroy()
    # Drain the per-launch events accumulated during the timed loop.
    rt.wait_stream(resolved)
    return ms


# ---------------------------------------------------------------------
# CK-Tile-style multi-kernel launch primitive
# ---------------------------------------------------------------------
#
# Python parity with ``ck_tile::make_kernel`` and
# ``ck_tile::launch_kernel`` from
# ``include/ck_tile/host/kernel_launch.hpp``. The C++ pattern bakes
# everything (grid, block, LDS bytes, kernel args) into a closure
# returned by ``make_kernel(K{}, grid, block, lds, kargs)``; the
# variadic ``launch_kernel(stream_config, c0, c1, ...)`` then invokes
# each closure on the same stream in order with optional timing.
#
# The Python equivalent provided here:
#
# - :func:`make_kernel` -- returns a ``Callable[[StreamConfig], None]``
#   that, on call, dispatches one launch through a pre-built
#   :class:`KernelLauncher` (no compile, no module load -- both
#   already happened when the launcher was constructed). The closure
#   reads ``stream_id`` from its :class:`StreamConfig` argument so the
#   same closure can be replayed on different streams. The closure
#   always launches with ``fence=False``: the only sync point is
#   :func:`launch_kernel`'s timing-loop boundary (or the caller's
#   external drain).
#
# - :func:`launch_kernel` -- variadic over the closure callables.
#   With ``time_kernel=False``, runs each callable once on the stream
#   under :func:`no_fence` and returns ``0.0``. With
#   ``time_kernel=True``, runs ``cold_niters`` warmup iterations of
#   the whole callable group, then ``nrepeat`` timed iterations, and
#   returns the per-iteration average wall time in milliseconds. The
#   timer is GPU-event by default (``is_gpu_timer=True``) or CPU-side
#   :func:`time.perf_counter` between :meth:`Runtime.wait_stream`
#   calls when ``is_gpu_timer=False``.
#
# Coexistence with :class:`PipelineLauncher`
# ------------------------------------------
# :class:`PipelineLauncher` is the higher-level cached form for hot
# dispatch paths: it owns N pre-built :class:`KernelLauncher`\\s and
# accepts ``values_per_stage`` + ``configs_per_stage`` arrays each
# call. It enforces "only the last stage fences" because its callers
# rely on the implicit final sync.
#
# :func:`launch_kernel` here is the low-level building block: callers
# bake (values, grid, block, lds) into a closure via
# :func:`make_kernel` and pass arbitrary callables (including bare
# host lambdas, e.g. ``maybe_clear_workspace``). The closures never
# fence, and :func:`launch_kernel` does not implicitly fence at the
# end -- the caller is responsible for the final sync (or relying on
# torch's stream-aware next-read sync). This is strictly more
# flexible (a single timing event around the whole group is the
# dominant case in benchmark drivers) but production callers
# migrating from :class:`PipelineLauncher` MUST add an explicit
# :func:`wait_stream_and_release` (or accept torch's next-read sync)
# after :func:`launch_kernel` returns.
#
# Macro-style shorthand convention (for instance authors)
# -------------------------------------------------------
# CK Tile's ``MOE_SORTING_MP_*`` macros at
# ``example/ck_tile/15_fused_moe/instances/fused_moesorting_api.cpp``
# each expand to a self-invoking lambda that returns a
# ``make_kernel`` closure. The Python equivalent for instance
# authors is a per-phase factory function::
#
#     def moe_sort_histogram_callable(spec, args, *, launcher=None):
#         if launcher is None:
#             launcher = _get_or_build_hist_launcher(spec)
#         grid  = moe_sort_histogram_grid(spec, args.num_tokens)
#         block = (spec.block_size, 1, 1)
#         return make_kernel(launcher, args.histogram_values(),
#                            grid, block,
#                            lds_bytes=moe_sort_histogram_lds(spec))
#
# Then a CK-style chained launch reads as::
#
#     launch_kernel(StreamConfig(stream_id=stream),
#                   moe_sort_histogram_callable(spec, args),
#                   moe_sort_scan_callable(spec, args),
#                   moe_sort_scatter_callable(spec, args))


def make_kernel(
    launcher: "KernelLauncher",
    values: Mapping[str, Any],
    grid: Tuple[int, int, int],
    block: Tuple[int, int, int],
    *,
    lds_bytes: int = 0,
) -> Callable[["StreamConfig"], None]:
    """Bake (values, grid, block, lds_bytes) into a CK-style launch closure.

    Returns a ``Callable[[StreamConfig], None]`` that, on call,
    dispatches one launch through ``launcher``. The returned closure:

    - Reads ``stream_id`` from its :class:`StreamConfig` argument at
      call time (not at closure construction time), so the same
      closure can be replayed on different streams. This matches CK
      Tile's ``ck_tile::make_kernel`` (the chevron launch in the
      generated lambda references ``s.stream_id_``, not a captured
      stream).
    - Always uses ``fence=False`` regardless of any outer
      :func:`no_fence` context. The only host-side sync points in
      this primitive are:
        * the timing-loop boundary inside :func:`launch_kernel` when
          ``StreamConfig.time_kernel=True``;
        * any explicit :func:`wait_stream_and_release` /
          :func:`synchronize_and_release` that the caller issues
          after :func:`launch_kernel` returns;
        * torch's stream-aware next-read sync.
    - Holds an immutable copy of ``values`` (via ``dict(values)``)
      and ``(grid, block, lds_bytes)`` so the closure is stable
      against caller mutations of the input dict.

    Construct ``launcher`` once per (problem-shape, problem-dtype)
    via the standard
    ``compile_kernel(...)`` -> ``KernelLauncher(hsaco, kernel_name,
    signature)`` flow and reuse the same launcher across every
    :func:`make_kernel` call. There is no per-closure compile, no
    per-closure module load.

    Mirrors ``ck_tile::make_kernel`` at
    ``include/ck_tile/host/kernel_launch.hpp`` lines 118-133.
    """
    captured_values = dict(values)
    captured_grid = (int(grid[0]), int(grid[1]), int(grid[2]))
    captured_block = (int(block[0]), int(block[1]), int(block[2]))
    captured_lds = int(lds_bytes)

    def _closure(s: "StreamConfig") -> None:
        launcher(
            captured_values,
            config=LaunchConfig(
                stream=int(s.stream_id),
                grid=captured_grid,
                block=captured_block,
                shared_bytes=captured_lds,
                fence=False,
            ),
        )

    return _closure


def _launch_and_check(
    s: "StreamConfig",
    callables: Sequence[Callable[["StreamConfig"], None]],
) -> None:
    """Run each callable on ``s`` in declaration order under ``no_fence``.

    Mirrors ``ck_tile::launch_and_check`` at
    ``include/ck_tile/host/kernel_launch.hpp`` lines 135-143. The C++
    side does an explicit ``hipPeekAtLastError`` between callables and
    short-circuits via a fold expression; the Python side relies on
    :class:`Runtime` raising :class:`HipError` immediately when
    ``hipModuleLaunchKernel`` returns non-success, which gives the
    same short-circuit behavior (subsequent callables are not invoked
    after a raise).
    """
    with no_fence():
        for c in callables:
            c(s)


def _timing_loop(
    s: "StreamConfig",
    callables: Sequence[Callable[["StreamConfig"], None]],
) -> float:
    """Run ``callables`` for cold + timed iterations and return ms/iter.

    Mirrors ``ck_tile::timing_loop_impl`` at
    ``include/ck_tile/host/kernel_launch.hpp`` lines 204-236: a
    ``cold_niters`` warmup pass over the whole callable group
    followed by ``nrepeat`` timed passes, with the timer wrapping
    only the timed region. ``s.is_gpu_timer`` selects between an
    event-bracketed measurement (default; matches CK Tile's
    ``gpu_timer``) and a CPU-side :func:`time.perf_counter` window
    between :meth:`Runtime.wait_stream` syncs.
    """
    rt = _runtime()
    resolved_stream = resolve_stream(int(s.stream_id))
    cold = max(0, int(s.cold_niters))
    iters = max(1, int(s.nrepeat))

    with no_fence():
        for _ in range(cold):
            for c in callables:
                c(s)

        if s.is_gpu_timer:
            # Drain any pending warmup work so the start event marks
            # the first timed launch's submission, not warmup
            # completion.
            rt.wait_stream(resolved_stream)
            e0 = rt.event()
            e1 = rt.event()
            e0.record(stream=resolved_stream)
            for _ in range(iters):
                for c in callables:
                    c(s)
            e1.record(stream=resolved_stream)
            e1.synchronize()
            ms_total = e0.elapsed_to(e1)
            e0.destroy()
            e1.destroy()
        else:
            rt.wait_stream(resolved_stream)
            t0 = _time.perf_counter()
            for _ in range(iters):
                for c in callables:
                    c(s)
            rt.wait_stream(resolved_stream)
            ms_total = (_time.perf_counter() - t0) * 1000.0

    # Final drain: the timed-loop body ran under :func:`no_fence`
    # which does not retain per-launch events, so the
    # :attr:`Runtime._pending_args` bucket for this stream still
    # holds args buffers + tensor refs from the timed launches. The
    # ``e1.synchronize()`` above (or the trailing ``wait_stream`` in
    # the CPU-timer branch) drained the GPU side; this drains the
    # host bookkeeping.
    rt.wait_stream(resolved_stream)
    return float(ms_total) / iters


def launch_kernel(
    s: "StreamConfig",
    *callables: Callable[["StreamConfig"], None],
) -> float:
    """CK-Tile-style multi-kernel launch.

    Run each callable on ``s.stream_id`` in declaration order. With
    ``s.time_kernel=False`` (default), the call is fire-and-forget on
    the same stream and returns ``0.0`` -- the caller is responsible
    for any final sync (via :func:`wait_stream_and_release` or
    torch's stream-aware next-read). With ``s.time_kernel=True``,
    runs ``s.cold_niters`` warmup iterations of the whole callable
    group followed by ``s.nrepeat`` timed iterations, and returns the
    per-iteration average in milliseconds.

    Each callable must accept a :class:`StreamConfig` argument:

    - The most common shape is the closure returned by
      :func:`make_kernel`, which dispatches one
      :class:`KernelLauncher` launch with baked
      ``(values, grid, block, lds_bytes)``.
    - Bare host lambdas are also accepted -- e.g. for a
      ``maybe_clear_workspace`` step that issues a ``hipMemset`` on
      the stream:
      ``lambda sc: rt.memset(ws_ptr, 0, ws_bytes, stream=sc.stream_id)``.
      This mirrors the C++ ``MOR_SORTING_MP_DISPATCH_`` form at
      ``example/ck_tile/15_fused_moe/instances/fused_moesorting_api.cpp:481``.

    Stream / sync model
    -------------------
    All callables share ``s.stream_id``. Same-stream FIFO ordering on
    HIP guarantees callable ``i+1`` observes callable ``i``'s writes
    -- no events, no host barriers between them. The closures
    produced by :func:`make_kernel` always launch with ``fence=False``,
    so :func:`launch_kernel` itself never inserts a per-callable
    sync. In the timing path, the only sync points are the
    :class:`Event` synchronize bracketing the timed iterations (or
    :meth:`Runtime.wait_stream` for the CPU-timer path) plus a final
    :meth:`Runtime.wait_stream` to drain
    :attr:`Runtime._pending_args` bookkeeping. In the non-timing
    path, there is no implicit sync at all.

    Migration note for :class:`PipelineLauncher` callers
    ----------------------------------------------------
    Unlike :class:`PipelineLauncher` -- which honours ``cfg.fence``
    on the *last* stage and therefore implicitly fences the host on
    pipeline completion -- :func:`launch_kernel` never fences. Code
    paths that returned a :class:`LaunchSummary` from
    ``pipeline(...)`` and then immediately read an output tensor on
    the host side relied on the last-stage fence; if you migrate
    them to :func:`launch_kernel`, add an explicit
    :func:`wait_stream_and_release` (or rely on torch's stream-aware
    next-read sync) before the host read.

    Returns
    -------
    Average per-iteration wall time in milliseconds when
    ``s.time_kernel=True``; ``0.0`` otherwise. Matches the C++
    ``ck_tile::launch_kernel`` at
    ``include/ck_tile/host/kernel_launch.hpp`` lines 265-286.

    Raises
    ------
    ValueError if no callables were supplied (parity with the C++
    ``static_assert(sizeof...(callables) > 0)``).
    """
    if not callables:
        raise ValueError("launch_kernel requires at least one callable")

    if not s.time_kernel:
        _launch_and_check(s, callables)
        return 0.0

    return _timing_loop(s, callables)
