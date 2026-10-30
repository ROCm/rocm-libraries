# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Launch plans of the Winograd forward convolution (gfx1250, opt-in route).

A Winograd forward is several kernels, each built for the problem (the
transform kernels bake the extents in): the plan kinds of
:mod:`kernels.common.conv_winograd_select` map onto launch sequences as

* ``unfused2d`` (3x3, :mod:`kernels.common.conv_winograd_fwd`): K1 filter
  transform (merged with the first chunk's input transform when the plan asks
  for it and :func:`merge_worthwhile` agrees), K2a (split prescale only), K2
  input transform, K3 batched GEMM (:mod:`kernels.common.conv_winograd_gemm_wmma`)
  and K4 output transform; a batch split runs K2 / K3 / K4 once per chunk of
  ``N / chunks`` images on byte offsets of x and y, sharing U;
* ``unfused2d_rt`` (3x3, :mod:`kernels.common.conv_winograd_fwd_rt`): the same
  pipeline (without the merged launch) on shape-generic transform binaries,
  every problem extent a kernarg, so each binary is compiled once ahead of
  time and named by its kernel name;
* ``unfused_gen`` (1-D, 7x1, 5x5, segmented long filters,
  :mod:`kernels.common.conv_winograd_gen`): K1, K2, K3 with ``taps`` segments,
  K4;
* ``fused1d`` (:mod:`kernels.common.conv_winograd_fused_1d_wmma`): the
  generalized K1 writes U, then one fused kernel goes from x to y.

The steps are :class:`ConvLaunchStep` records with no conv-kernel identity: each
carries the kernel to build (``kernel.build(arch)``) and its kernargs by name,
resolved by :meth:`ConvLaunchPlan.values` against the buffers ``x`` / ``w`` /
``y`` and the plan's workspaces (``U``, ``V``, ``M``, ``e``). Transform knobs,
operand dtype and the prescale follow the defaults of the Winograd benchmark
drivers; K3 takes the plan's recommended configuration over every batch.
"""

from __future__ import annotations

import dataclasses
import functools
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

from kernels.common._conv_implicit_gemm_common import ConvProblem
from kernels.common.conv_winograd_select import WinogradPlan

__all__ = [
    "ConvWinogradSpec",
    "WinogradKernel",
    "winograd_flavor_reason",
    "winograd_launch_plan",
    "winograd_steps",
]


@dataclass(frozen=True)
class WinogradKernel:
    """The kernel of one plan step: ``build(arch) -> KernelDef``.

    ``llvm_flavors`` lists the LLVM flavors the kernel lowers at; ``None``
    means every flavor."""

    name: str
    build: Callable[[str], object]
    llvm_flavors: Optional[Tuple[str, ...]] = None


@dataclass(frozen=True)
class ConvWinogradSpec:
    """A Winograd plan selected for one forward convolution.

    ``plan`` is the selector's :class:`WinogradPlan`; ``block`` the workgroup
    of the plan's last kernel (the one :class:`DispatchResult` describes).
    """

    name: str
    plan: WinogradPlan
    problem: ConvProblem
    dtype: str
    arch: str
    block: Tuple[int, int, int] = (256, 1, 1)

    direction = "fwd"
    algorithm = "winograd"
    is_direct = False
    helpers: Tuple = ()

    def kernel_name(self) -> str:
        return self.name

    def identity(self) -> dict:
        p = self.plan
        return {
            "name": self.name,
            "arch": self.arch,
            "dtype": self.dtype,
            "problem": dataclasses.asdict(self.problem),
            "plan": {
                "kind": p.kind,
                "tile": list(p.tile),
                "variant": p.variant,
                "k3": p.k3,
                "chunks": p.chunks,
                "merge": p.merge,
                "fused_cfg": [[k, v] for k, v in p.fused_cfg],
            },
        }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _k3_knobs(label: str) -> dict:
    """``WinogradGemmWmmaSpec`` knobs of a K3 label such as
    ``256x256x64_w2x4_f8x4_s3_tdm`` (the selector's recommendation)."""
    toks = label.split("_")
    tile_k = int(toks[0].split("x")[2])
    knobs = dict(tile_k=tile_k)
    for t in toks[1:]:
        if t.startswith("w") and "x" in t:
            a, b = t[1:].split("x")
            knobs.update(waves_m=int(a), waves_n=int(b))
        elif t.startswith("f") and "x" in t:
            a, b = t[1:].split("x")
            knobs.update(frags_m=int(a), frags_n=int(b))
        elif t.startswith("s") and t[1:].isdigit():
            knobs["stages"] = int(t[1:])
        elif t in ("tdm", "direct"):
            knobs["epilogue"] = t
        else:
            raise ValueError(f"K3 label {label!r}: unknown token {t!r}")
    return knobs


def _gen_axes(problem: ConvProblem, variant: str):
    """``(ax_y, ax_x)`` of a generalized-pipeline variant string (``4s3``,
    ``6``, ``2x2``), as the Winograd benchmark drivers parse it."""
    from kernels.common.conv_winograd_gen import WinoAxis

    my = 0
    text = variant
    if "x" in text:
        a, text = text.split("x", 1)
        my = int(a)
    seg = 1
    if "s" in text:
        text, s = text.split("s", 1)
        seg = int(s)
    mx = int(text)
    ay = WinoAxis(1, 1) if problem.Y == 1 else WinoAxis(my or mx, problem.Y)
    if problem.X == 1:
        ax = WinoAxis(1, 1)
        if problem.Y > 1 and not my:
            ay = WinoAxis(mx, problem.Y)
    else:
        ax = WinoAxis(mx, problem.X, seg)
    return ay, ax


# Transform-kernel argument -> buffer of the plan ("<arg>_bytes" is its size).
_BUF_OF = {"w": "w", "u": "U", "x": "x", "v": "V", "e": "e", "mm": "M", "m": "M"}
_BUF_OF.update(y="y")


def _transform_kernargs(signature, sizes: dict, offsets: dict) -> tuple:
    """Kernargs of a transform-kernel step from its signature: pointers at the
    chunk's byte offset, ``*_bytes`` the chunk's buffer size."""
    out = []
    for a in signature:
        nm = a["name"]
        if nm.endswith("_bytes"):
            out.append(("i32", int(sizes[_BUF_OF[nm[: -len("_bytes")]]])))
        else:
            buf = _BUF_OF[nm]
            out.append(("buf", buf, int(offsets.get(buf, 0))))
    return tuple(out)


def _step(name, build, grid, block, signature, kernargs, llvm_flavors=None):
    from .grouped_convolution import ConvLaunchStep

    bufs = [k[1] for k in kernargs if k[0] == "buf"]
    return ConvLaunchStep(
        identity=None,
        grid=tuple(int(g) for g in grid),
        block=(int(block), 1, 1),
        signature=tuple(signature),
        buffers=(bufs[0], bufs[1] if len(bufs) > 2 else None, bufs[-1]),
        kernel=WinogradKernel(name, build, llvm_flavors),
        kernargs=tuple(kernargs),
    )


def _k3_step(spec, prob, v_off=0, m_off=0):
    """The batched-GEMM step over every batch of ``prob`` (splits 1)."""
    from kernels.common.conv_winograd_gemm_wmma import (
        LLVM_FLAVORS,
        build_winograd_gemm_wmma,
        winograd_gemm_problem_reason,
        winograd_gemm_reason,
        winograd_gemm_signature,
    )

    for why in (
        winograd_gemm_reason(spec),
        winograd_gemm_problem_reason(spec, prob),
    ):
        if why:
            raise ValueError(f"K3 {spec.label()}: {why}")
    vals = spec.launch_values(prob, 1, 2, 3)
    ptr = {"v": ("V", v_off), "u": ("U", 0), "d": ("M", m_off)}
    sig = winograd_gemm_signature(spec)
    kernargs = []
    for a in sig:
        nm = a["name"]
        if nm in ptr:
            kernargs.append(("buf", ptr[nm][0], ptr[nm][1]))
        else:
            kernargs.append(("i32", int(vals[nm])))
    return _step(
        "K3",
        functools.partial(_build_with_arch, build_winograd_gemm_wmma, spec),
        spec.grid(prob),
        spec.threads_per_block,
        sig,
        kernargs,
        llvm_flavors=LLVM_FLAVORS,
    )


def _build_with_arch(fn, spec, arch):
    return fn(spec, arch=arch)


# ---------------------------------------------------------------------------
# plan kinds
# ---------------------------------------------------------------------------


def _unfused2d(problem: ConvProblem, dtype: str, plan: WinogradPlan):
    from kernels.common import conv_winograd_fwd as W
    from kernels.common.conv_winograd_gemm_wmma import (
        WinogradGemmProblem,
        WinogradGemmWmmaSpec,
    )

    mh, mw = plan.tile
    chunks = max(1, int(plan.chunks))
    if problem.N % chunks:
        raise ValueError(f"batch split {chunks} does not divide N={problem.N}")
    prescale = dtype == "bf16" and max(mh, mw) >= 4
    operand = "bf16" if ((mh, mw) == (2, 2) and dtype == "bf16") else "fp16"
    xf = W.tuned_transform_options((mh, mw), prescale, problem)
    sub = dataclasses.replace(problem, N=problem.N // chunks)
    spec = W.WinogradFwdSpec(
        problem=sub,
        m=mh,
        m_w=mw,
        io_dtype=dtype,
        operand_dtype=operand,
        prescale=prescale,
        **xf,
    )
    sizes = dict(spec.buffer_bytes())
    if not spec.prescale:
        sizes["e"] = 4  # never read; keeps a valid pointer
    stages = {st.name: st for st in W.transform_stages(spec)}
    merge: Tuple[str, ...] = ()
    if plan.merge and W.merge_worthwhile(spec):
        names = [s.strip() for s in plan.merge.split(",")]
        first = "K2a" if (spec.prescale and spec.prescale_split) else "K2"
        merge = tuple(first if s == "K2" else s for s in names)
    pre = [k for k in ("K2a", "K2") if k in stages]
    x_chunk, y_chunk = sizes["x"], sizes["y"]

    def stage_step(name, chunk):
        st = stages[name]
        offs = {"x": chunk * x_chunk, "y": chunk * y_chunk}
        return _step(
            f"{name}" + (f"[{chunk}]" if chunks > 1 else ""),
            functools.partial(_build_with_arch, st.build, spec),
            st.grid,
            st.block,
            st.signature,
            _transform_kernargs(st.signature, sizes, offs),
        )

    steps = []
    if merge:
        lay = W.merged_stage(spec, merge)
        mspec = W.WinogradMergedSpec(spec, merge)
        steps.append(
            _step(
                "MERGED_" + "_".join(merge),
                functools.partial(_build_with_arch, W.build_winograd_merged, mspec),
                lay.grid,
                lay.block,
                lay.signature,
                _transform_kernargs(lay.signature, sizes, {"x": 0, "y": 0}),
            )
        )
    else:
        steps.append(stage_step("K1", 0))
    k3 = WinogradGemmWmmaSpec(dtype=operand, **_k3_knobs(plan.k3))
    prob = WinogradGemmProblem.from_conv(sub, (mh, mw))
    for i in range(chunks):
        for k in pre:
            if i == 0 and k in merge:
                continue
            steps.append(stage_step(k, i))
        steps.append(_k3_step(k3, prob))
        steps.append(stage_step("K4", i))
    workspaces = {k: int(max(sizes[k], 4)) for k in ("U", "V", "M", "e")}
    return tuple(steps), workspaces


def _rt_kernargs(signature, vals: dict, sizes: dict, offsets: dict) -> tuple:
    """Kernargs of a shape-generic transform step: its scalar values from
    ``stage_launch``, ``*_bytes`` the chunk's buffer sizes, pointers at the
    chunk's byte offsets."""
    out = []
    for a in signature:
        nm = a["name"]
        if nm in vals:
            out.append(("i32", int(vals[nm])))
        elif nm.endswith("_bytes"):
            out.append(("i32", int(sizes[_BUF_OF[nm[: -len("_bytes")]]])))
        else:
            buf = _BUF_OF[nm]
            out.append(("buf", buf, int(offsets.get(buf, 0))))
    return tuple(out)


def _unfused2d_rt(problem: ConvProblem, dtype: str, plan: WinogradPlan):
    """The 3x3 pipeline on the shape-generic (AOT) transform binaries of
    :mod:`kernels.common.conv_winograd_fwd_rt`: each step is named by its
    binary's kernel name (its AOT identity) and its problem extents are
    kernargs. The binaries are :func:`default_stage_specs`; K3 is the plan's
    WMMA GEMM, as in :func:`_unfused2d`."""
    from kernels.common import conv_winograd_fwd_rt as R
    from kernels.common.conv_winograd_gemm_wmma import (
        WinogradGemmProblem,
        WinogradGemmWmmaSpec,
    )

    mh, mw = plan.tile
    chunks = max(1, int(plan.chunks))
    if problem.N % chunks:
        raise ValueError(f"batch split {chunks} does not divide N={problem.N}")
    sub = dataclasses.replace(problem, N=problem.N // chunks)
    geo = R.RtGeometry(sub, mh, mw)
    operand, prescale = R.pipeline_dtypes((mh, mw), dtype)
    why = geo.check(prescale)
    if why:
        raise ValueError(why)
    specs = R.default_stage_specs(sub, (mh, mw), dtype)
    sizes = dict(geo.buffer_bytes(prescale))
    x_chunk, y_chunk = sizes["x"], sizes["y"]

    def stage_step(stage, chunk):
        spec = specs[stage]
        grid, block, vals = R.stage_launch(spec, geo)
        sig = R.stage_signature(spec)
        offs = {"x": chunk * x_chunk, "y": chunk * y_chunk}
        return _step(
            spec.kernel_name(),
            functools.partial(_build_with_arch, R.build_stage, spec),
            grid,
            block,
            sig,
            _rt_kernargs(sig, vals, sizes, offs),
        )

    steps = [stage_step("K1", 0)]
    k3 = WinogradGemmWmmaSpec(dtype=operand, **_k3_knobs(plan.k3))
    prob = WinogradGemmProblem.from_conv(sub, (mh, mw))
    for i in range(chunks):
        for k in ("K2a", "K2"):
            if k in specs:
                steps.append(stage_step(k, i))
        steps.append(_k3_step(k3, prob))
        steps.append(stage_step("K4", i))
    workspaces = {k: int(max(sizes[k], 4)) for k in ("U", "V", "M", "e")}
    return tuple(steps), workspaces


def _unfused_gen(problem: ConvProblem, dtype: str, plan: WinogradPlan):
    from kernels.common import conv_winograd_gen as WG
    from kernels.common.conv_winograd_gemm_wmma import WinogradGemmWmmaSpec

    if int(plan.chunks) != 1:
        raise ValueError("the generalized pipeline runs without a batch split")
    ay, ax = _gen_axes(problem, plan.variant)
    spec = WG.WinogradGenSpec(problem=problem, ax_y=ay, ax_x=ax, io_dtype=dtype)
    sizes = dict(spec.buffer_bytes())
    prob = spec.gemm_problem()
    steps = []
    stages = {st.name: st for st in WG.transform_stages(spec)}
    for name in ("K1", "K2"):
        st = stages[name]
        steps.append(
            _step(
                name,
                functools.partial(_build_with_arch, st.build, spec),
                st.grid,
                st.block,
                st.signature,
                _transform_kernargs(st.signature, sizes, {}),
            )
        )
    k3 = WinogradGemmWmmaSpec(dtype="fp16", taps=spec.S, **_k3_knobs(plan.k3))
    steps.append(_k3_step(k3, prob))
    st = stages["K4"]
    steps.append(
        _step(
            "K4",
            functools.partial(_build_with_arch, st.build, spec),
            st.grid,
            st.block,
            st.signature,
            _transform_kernargs(st.signature, sizes, {}),
        )
    )
    workspaces = {k: int(max(sizes[k], 4)) for k in ("U", "V", "M")}
    return tuple(steps), workspaces


def _fused1d(problem: ConvProblem, dtype: str, plan: WinogradPlan):
    from kernels.common import conv_winograd_gen as WG
    from kernels.common.conv_winograd_fused_1d_wmma import (
        FusedWino1DSpec,
        build_fused_wino_1d,
        fused_signature,
    )

    if dtype != "fp16":
        raise ValueError("the fused 1-D kernel takes fp16 I/O only")
    ay, ax = _gen_axes(problem, plan.variant)
    gspec = WG.WinogradGenSpec(problem=problem, ax_y=ay, ax_x=ax)
    cfg = dict(plan.fused_cfg)
    try:
        spec = FusedWino1DSpec(problem=problem, ax=ax, **cfg)
    except ValueError as e:
        if "not exact in fp16" not in str(e):
            raise
        # B^T not fp16-exact: the VALU transform (as the benchmark driver).
        spec = FusedWino1DSpec(problem=problem, ax=ax, **dict(cfg, xf="valu"))
    gsizes = gspec.buffer_bytes()
    sizes = dict(spec.buffer_bytes())
    if sizes["u"] != gsizes["U"]:
        raise ValueError("U layout of the fused kernel differs from K1's")
    k1 = WG.transform_stages(gspec)[0]
    steps = [
        _step(
            "K1",
            functools.partial(_build_with_arch, k1.build, gspec),
            k1.grid,
            k1.block,
            k1.signature,
            _transform_kernargs(k1.signature, gsizes, {}),
        )
    ]
    sig = fused_signature(spec)
    fused_buf = {"x": "x", "u": "U", "y": "y"}
    kernargs = []
    for a in sig:
        nm = a["name"]
        if nm.endswith("_bytes"):
            kernargs.append(("i32", int(sizes[nm[: -len("_bytes")]])))
        else:
            kernargs.append(("buf", fused_buf[nm], 0))
    steps.append(
        _step(
            "FUSED",
            functools.partial(_build_with_arch, build_fused_wino_1d, spec),
            (spec.grid(), 1, 1),
            spec.waves * 32,
            sig,
            kernargs,
        )
    )
    return tuple(steps), {"U": int(max(gsizes["U"], 4))}


_KINDS = {
    "unfused2d": _unfused2d,
    "unfused2d_rt": _unfused2d_rt,
    "unfused_gen": _unfused_gen,
    "fused1d": _fused1d,
}


@functools.lru_cache(maxsize=256)
def winograd_steps(problem: ConvProblem, dtype: str, plan: WinogradPlan) -> tuple:
    """``(steps, workspaces)`` of ``plan`` on ``problem`` (memoised).

    Raises ``ValueError`` when a kernel of the plan cannot run the problem.
    """
    fn = _KINDS.get(plan.kind)
    if fn is None:
        raise ValueError(f"no launch plan for Winograd kind {plan.kind!r}")
    steps, workspaces = fn(problem, dtype, plan)
    return steps, tuple(sorted(workspaces.items()))


def winograd_flavor_reason(
    problem: ConvProblem, dtype: str, plan: WinogradPlan, llvm_flavor: str
) -> Optional[str]:
    """Why a kernel of ``plan`` does not lower at ``llvm_flavor``, or ``None``.

    Raises ``ValueError`` like :func:`winograd_steps`."""
    steps, _ = winograd_steps(problem, dtype, plan)
    for step in steps:
        flavors = step.kernel.llvm_flavors
        if flavors is not None and llvm_flavor not in flavors:
            return (
                f"{step.kernel.name} needs LLVM flavor {' or '.join(flavors)}, "
                f"the request asks for {llvm_flavor}"
            )
    return None


def winograd_launch_plan(
    problem: ConvProblem, dtype: str, plan: WinogradPlan, arch: Optional[str] = None
):
    """The :class:`ConvLaunchPlan` of ``plan`` on ``problem`` (buffers
    ``x`` / ``w`` / ``y`` plus the plan's workspaces). ``arch`` is only
    checked: the kernels are built per step with the caller's arch."""
    from .grouped_convolution import ConvLaunchPlan

    if arch is not None and arch != "gfx1250":
        raise ValueError(f"the Winograd forward kernels target gfx1250, not {arch}")
    steps, workspaces = winograd_steps(problem, dtype, plan)
    return ConvLaunchPlan(steps, dict(workspaces))
