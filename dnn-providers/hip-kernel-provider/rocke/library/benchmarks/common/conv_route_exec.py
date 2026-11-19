# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Bind a production route (:func:`dispatch.conv_routes.dispatch_conv_route`) to device buffers.

Torch-free: buffers are raw device pointers from :class:`rocke.runtime.hip_module.Runtime`
(torch tensors work too, through their ``data_ptr``). The pinned implicit-GEMM step of a
route is bound exactly as ``benchmark_implicit_gemm_conv --run-from-cache`` binds a cached
binary (:func:`benchmark_implicit_gemm_conv._cached_bind_batch`: kernargs, grid, launch
scratch, zeroing, Stage 2); the route's own kernels (pad copy, Winograd relayout / main /
split-K reduction) are compiled from their ``RouteStep.build`` (memoised per process) and
bound with ``KernelLauncher.bind``. Workspaces are allocated once per binding (or shared
through ``workspaces=``, a :class:`SharedWorkspaces`) and the ``zero_once`` ones zeroed
at allocation. A shared workspace is reused only by a plan that declares it the same
way (same size and ``zero_once``; a ``zero_once`` one only by the same route): anything
else is refused, never bound to a buffer it would overrun or find dirty.

Usage::

    plan = dispatch_conv_route(req, routes)
    rb = RouteBinding(plan, cache, rt, "gfx942", ptrs=(x, w, y), sizes=(nx, nw, ny))
    rb.run()          # enqueue the whole call on the null stream
    rb.free()         # release the workspaces / launch scratch (after a sync)
"""

from __future__ import annotations

_KERNELS: dict = {}  # (kernel name, arch) -> (hsaco, name)


def compiled_step(step, arch: str):
    """``(hsaco, kernel_name)`` of a ``RouteStep`` of kind ``kernel`` (memoised)."""
    from rocke.helpers import compile_kernel

    kd = step.build(arch)
    key = (kd.name, arch)
    hit = _KERNELS.get(key)
    if hit is None:
        hit = _KERNELS[key] = (compile_kernel(kd, arch=arch).hsaco, kd.name)
    return hit


def pinned_launch(plan, cache, B):
    """``(ident, meta, launcher, split_k)`` of the plan's pinned kernel from ``cache``."""
    from benchmarks.common.kernel_cache import KernelIdentity
    from rocke.runtime.launcher import KernelLauncher

    spec = plan.kernel.spec
    ident = KernelIdentity.from_label(spec.name)
    lp = spec.launch_params()
    split_k = lp.get("split_k")
    if split_k is not None and "tail_full" in lp:
        split_k = B.TailSplit(int(split_k), int(lp["tail_full"]))
    entry = cache.get(ident)
    if entry is None:
        raise LookupError(f"{plan.case}: {spec.name} is not in the cache")
    blob, meta = entry
    dtype = str(plan.request.dtype)
    launcher = KernelLauncher(
        hsaco=blob,
        kernel_name=meta["kernel_name"],
        signature=B._signature_for_identity(ident, dtype),
    )
    return ident, meta, launcher, split_k


def _route_key(plan) -> tuple:
    """What a ``zero_once`` workspace's never-written part depends on: the
    route's launches and kernels (its layout), not only its size."""
    return (
        plan.route,
        plan.kernel_label,
        plan.launches,
        repr(plan.pad_copy),
        repr(plan.winograd),
        tuple(plan.workspaces),
    )


class SharedWorkspaces(dict):
    """Route workspaces shared by several bindings: ``{name: device pointer}``
    plus, per name, how it was allocated (``meta[name] = (nbytes, zero_once,
    route key)``), so a later binding can check the buffer fits its plan."""

    def __init__(self):
        super().__init__()
        self.meta = {}

    def compatible(self, plan) -> bool:
        """True when every workspace of ``plan`` already here can be reused by it."""
        return all(
            self._reason(w, plan) is None for w in plan.workspaces if w.name in self
        )

    def _reason(self, w, plan):
        nbytes, zero_once, key = self.meta[w.name]
        if nbytes < w.nbytes:
            return f"shared buffer of {nbytes} B < {w.nbytes} B needed"
        if zero_once != w.zero_once:
            return "zero_once differs from the shared buffer's"
        if w.zero_once and key != _route_key(plan):
            return "zero_once buffer of another route (its halo may not be zero)"
        return None


class RouteBinding:
    """One call of a routed case bound to buffers ``ptrs`` = (A, B, D): fwd (X, W, Y),
    dgrad (dY, W, dX), wgrad (dY, X, dW); ``sizes`` their byte sizes.

    ``cfg_cls``: the ``LaunchConfig`` class to launch with (a stream-binding wrapper
    for multi-stream replays). ``workspaces``: a :class:`SharedWorkspaces` to share
    route workspaces between bindings of one case (a replay binds one call per
    buffer set; the workspaces are scratch, reused call after call); a workspace
    already there that does not fit this plan raises ``ValueError``."""

    def __init__(
        self,
        plan,
        cache,
        rt,
        arch,
        ptrs,
        sizes,
        *,
        stage2_kernels=None,
        cfg_cls=None,
        workspaces=None,
        live_extras=None,
        B=None,
        alloc=None,
    ):
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig

        if B is None:
            from benchmarks.common import benchmark_implicit_gemm_conv as B
        self.plan, self.rt = plan, rt
        self.ptrs, self.sizes = tuple(ptrs), tuple(sizes)
        cfg_cls = cfg_cls or LaunchConfig
        self.live_extras = live_extras if live_extras is not None else []
        self._owned = []
        if workspaces is not None and not isinstance(workspaces, SharedWorkspaces):
            raise TypeError("workspaces= takes a SharedWorkspaces")
        ws = workspaces if workspaces is not None else SharedWorkspaces()
        self._keep = []
        for w in plan.workspaces:
            if w.name in ws:
                why = ws._reason(w, plan)
                if why is not None:
                    raise ValueError(
                        f"{plan.case}: route workspace {w.name}: {why}; "
                        "bind with its own SharedWorkspaces"
                    )
            else:
                if (
                    alloc is not None
                ):  # caller's allocator: alloc(nbytes) -> (ptr, keepalive)
                    p, keep = alloc(w.nbytes)
                    self._keep.append(keep)
                else:
                    p = rt.alloc(w.nbytes)
                    if workspaces is None:
                        self._owned.append(p)
                if w.zero_once:
                    rt.memset(p, 0, w.nbytes)
                ws[w.name] = p
                ws.meta[w.name] = (w.nbytes, w.zero_once, _route_key(plan))
        self.workspaces = ws
        bufs = {"x": self.ptrs[0], "w": self.ptrs[1], "y": self.ptrs[2], **ws}
        nbytes = {
            "x": self.sizes[0],
            "w": self.sizes[1],
            "y": self.sizes[2],
            # the plan's sizes: a shared buffer is checked to be at least that
            **{w.name: w.nbytes for w in plan.workspaces},
        }
        runs = []
        for step in plan.steps:
            am = step.arg_map
            if step.kind == "pinned":
                ident, meta, launcher, split_k = pinned_launch(plan, cache, B)
                kp = B._launch_problem(plan.direction, plan.kernel_problem)
                chunks = B._batch_chunks(plan.direction, kp, str(plan.request.dtype))
                op = tuple(bufs[am[k]] for k in ("A", "B", "D"))
                osz = tuple(nbytes[am[k]] for k in ("A", "B", "D"))
                import rocke.runtime.launcher as _L

                saved = _L.LaunchConfig
                _L.LaunchConfig = cfg_cls
                try:
                    bound = B._cached_bind_batch(
                        chunks,
                        plan.direction,
                        kp,
                        ident,
                        meta,
                        rt,
                        launcher,
                        str(plan.request.dtype),
                        split_k,
                        op,
                        osz,
                        arch,
                        stage2_kernels if stage2_kernels is not None else {},
                        KernelLauncher,
                        cfg_cls,
                        self.live_extras,
                    )
                finally:
                    _L.LaunchConfig = saved
                if bound is None:
                    raise ValueError(
                        f"{plan.case}: {plan.kernel_label} cannot launch {kp}"
                    )
                runs.append(bound[0])
            else:
                hsaco, name = compiled_step(step, arch)
                kl = KernelLauncher(
                    hsaco=hsaco, kernel_name=name, signature=list(step.signature)
                )
                runs.append(
                    kl.bind(
                        {p: bufs[b] for p, b in step.args},
                        config=cfg_cls(grid=tuple(step.grid), block=tuple(step.block)),
                    )
                )
        self.runs = tuple(runs)
        if len(runs) == 1:
            self.run = runs[0]

    def run(self):
        for r in self.runs:
            r()

    def free(self, B=None):
        if B is None:
            from benchmarks.common import benchmark_implicit_gemm_conv as B
        B._free_extras(self.rt, self.live_extras)
        for p in self._owned:
            self.rt.free(p)
        self._owned = []
        self._keep = []
