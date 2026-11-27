# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Two-launch (reduce + apply) runner for the dynamic split batchnorm2d candidates.

Shared by the AOT benchmark and the validators so the workspace, kernarg and
grid conventions live in one place.
"""

from __future__ import annotations

from typing import Any

from rocke.instances.common.batchnorm2d import (
    batchnorm2d_split_dyn_grid,
    batchnorm2d_split_dyn_workspace_floats,
)
from rocke.runtime.launcher import KernelLauncher, LaunchConfig


def split_plan_label(cand: Any, plan: Any) -> str:
    return f"{cand.name}_sn{plan.sn}_sh{plan.sh}_nc{plan.n_chunk}_hc{plan.hw_chunk}"


class SplitRun:
    """Launches one compiled split pair with a fixed plan on fixed buffers.

    ``bufs`` holds device tensors: ``x gamma beta y rm rv`` and, when the spec
    saves statistics, ``mean invstd``; ``ws`` is the f32 workspace of at least
    ``batchnorm2d_split_dyn_workspace_floats(c, plan)`` elements.
    """

    def __init__(
        self,
        cand,
        plan,
        art_reduce,
        art_apply,
        *,
        n,
        c,
        h,
        w,
        bufs,
        ws,
        eps=1e-5,
        momentum=0.1,
    ):
        self.cand, self.plan = cand, plan
        # Keep the allocations alive: only raw addresses are retained below.
        self._keep = (bufs, ws)
        assert ws.numel() >= batchnorm2d_split_dyn_workspace_floats(c, plan)
        geom = {
            "N": n,
            "C": c,
            "H": h,
            "W": w,
            "NChunk": plan.n_chunk,
            "HwChunk": plan.hw_chunk,
            "SplitHw": plan.sh,
            "SplitN": plan.sn,
        }
        self.v_red = {"X": bufs["x"].data_ptr(), "Workspace": ws.data_ptr(), **geom}
        self.v_app = {
            "X": bufs["x"].data_ptr(),
            "Workspace": ws.data_ptr(),
            "Gamma": bufs["gamma"].data_ptr(),
            "Beta": bufs["beta"].data_ptr(),
            "Y": bufs["y"].data_ptr(),
            "RunningMean": bufs["rm"].data_ptr(),
            "RunningVar": bufs["rv"].data_ptr(),
            "eps": eps,
            "momentum": momentum,
            **geom,
        }
        if "mean" in bufs:
            self.v_app.update(
                {"Mean": bufs["mean"].data_ptr(), "InvStd": bufs["invstd"].data_ptr()}
            )
        self.l_red = KernelLauncher.from_artifact(art_reduce)
        self.l_app = KernelLauncher.from_artifact(art_apply)
        grid = batchnorm2d_split_dyn_grid(c, plan)
        self.cfg = LaunchConfig(grid=grid, block=(cand.block_size, 1, 1))

    def __call__(self) -> None:
        self.l_red(self.v_red, config=self.cfg)
        self.l_app(self.v_app, config=self.cfg)


class SplitBwdRun:
    """Two-launch runner of one compiled dynamic split *backward* pair.

    ``bufs`` holds device tensors ``x dy scale mean invstd dx dscale dbias``;
    ``ws`` is the f32 workspace (at least ``cand.workspace_floats(c, plan)``).
    """

    def __init__(self, cand, plan, art_reduce, art_apply, *, n, c, h, w, bufs, ws):
        self.cand, self.plan = cand, plan
        # Keep the allocations alive: only raw addresses are retained below.
        self._keep = (bufs, ws)
        assert ws.numel() >= cand.workspace_floats(c, plan)
        geom = {
            "N": n,
            "C": c,
            "H": h,
            "W": w,
            "NChunk": plan.n_chunk,
            "HwChunk": plan.hw_chunk,
            "SplitHw": plan.sh,
            "SplitN": plan.sn,
        }
        self.v_red = {
            "X": bufs["x"].data_ptr(),
            "dY": bufs["dy"].data_ptr(),
            "Mean": bufs["mean"].data_ptr(),
            "InvStd": bufs["invstd"].data_ptr(),
            "Workspace": ws.data_ptr(),
            **geom,
        }
        self.v_app = {
            "X": bufs["x"].data_ptr(),
            "dY": bufs["dy"].data_ptr(),
            "Scale": bufs["scale"].data_ptr(),
            "Mean": bufs["mean"].data_ptr(),
            "InvStd": bufs["invstd"].data_ptr(),
            "Workspace": ws.data_ptr(),
            "dX": bufs["dx"].data_ptr(),
            "dScale": bufs["dscale"].data_ptr(),
            "dBias": bufs["dbias"].data_ptr(),
            **geom,
        }
        self.l_red = KernelLauncher.from_artifact(art_reduce)
        self.l_app = KernelLauncher.from_artifact(art_apply)
        self.cfg = LaunchConfig(
            grid=batchnorm2d_split_dyn_grid(c, plan), block=(cand.block_size, 1, 1)
        )

    def __call__(self) -> None:
        self.l_red(self.v_red, config=self.cfg)
        self.l_app(self.v_app, config=self.cfg)


def bwd_values(bufs, *, n, c, h, w):
    """Kernel argument dict of the single-launch backward kernels."""
    return {
        "X": bufs["x"].data_ptr(),
        "dY": bufs["dy"].data_ptr(),
        "Scale": bufs["scale"].data_ptr(),
        "Mean": bufs["mean"].data_ptr(),
        "InvStd": bufs["invstd"].data_ptr(),
        "dX": bufs["dx"].data_ptr(),
        "dScale": bufs["dscale"].data_ptr(),
        "dBias": bufs["dbias"].data_ptr(),
        "N": n,
        "C": c,
        "H": h,
        "W": w,
    }


def nhwc_plan_label(cand: Any, plan: Any) -> str:
    return f"{cand.name}_rc{plan.rows_per_cta}_s{plan.splits}"


class NhwcRun:
    """Launches one compiled NHWC candidate (one or two kernels) with a fixed plan.

    ``bufs`` holds device tensors in NHWC memory order: infer / train ``x gamma
    beta y rm rv`` (+ ``mean invstd`` for train); backward ``x dy scale mean
    invstd dx dscale dbias``.  ``ws`` is the f32 workspace of a split plan."""

    def __init__(
        self, cand, plan, arts, *, n, c, h, w, bufs, ws=None, eps=1e-5, momentum=0.1
    ):
        from rocke.instances.common.batchnorm2d import batchnorm2d_nhwc_grid

        self.cand, self.plan = cand, plan
        # Keep the allocations alive: only raw addresses are retained below.
        self._keep = (bufs, ws)
        dims = {"N": n, "C": c, "H": h, "W": w}
        p = lambda k: bufs[k].data_ptr()  # noqa: E731
        geom = {"RowsPerCta": plan.rows_per_cta}
        if cand.variant == "split":
            assert ws is not None and ws.numel() >= cand.workspace_floats(c, plan)
            geom_apply = {**geom, "Splits": plan.splits}
            wsp = {"Workspace": ws.data_ptr()}
        else:
            geom_apply, wsp = {}, {}
        if cand.mode == "infer":
            vals = [
                {
                    "X": p("x"),
                    "Gamma": p("gamma"),
                    "Beta": p("beta"),
                    "Y": p("y"),
                    "RunningMean": p("rm"),
                    "RunningVar": p("rv"),
                    "eps": eps,
                    "momentum": momentum,
                    **dims,
                    **geom,
                }
            ]
        elif cand.mode == "train":
            full = {
                "X": p("x"),
                "Gamma": p("gamma"),
                "Beta": p("beta"),
                "Y": p("y"),
                "RunningMean": p("rm"),
                "RunningVar": p("rv"),
                "Mean": p("mean"),
                "InvStd": p("invstd"),
                "eps": eps,
                "momentum": momentum,
                **dims,
            }
            vals = (
                [full]
                if cand.variant == "fused"
                else [
                    {"X": p("x"), **wsp, **dims, **geom},
                    {**full, **wsp, **geom_apply},
                ]
            )
        else:
            full = {
                "X": p("x"),
                "dY": p("dy"),
                "Scale": p("scale"),
                "Mean": p("mean"),
                "InvStd": p("invstd"),
                "dX": p("dx"),
                "dScale": p("dscale"),
                "dBias": p("dbias"),
                **dims,
            }
            vals = (
                [full]
                if cand.variant == "fused"
                else [
                    {
                        "X": p("x"),
                        "dY": p("dy"),
                        "Mean": p("mean"),
                        "InvStd": p("invstd"),
                        **wsp,
                        **dims,
                        **geom,
                    },
                    {**full, **wsp, **geom_apply},
                ]
            )
        self.launchers = [KernelLauncher.from_artifact(a) for a in arts]
        self.vals = vals
        self.cfg = LaunchConfig(
            grid=batchnorm2d_nhwc_grid(c, plan), block=(cand.block_size, 1, 1)
        )

    def __call__(self) -> None:
        for l, v in zip(self.launchers, self.vals):
            l(v, config=self.cfg)


class McwRun:
    """Launches one compiled multi-channel-per-wave candidate (single kernel).

    ``bufs`` holds device tensors: train ``x gamma beta y rm rv mean invstd``;
    backward ``x dy scale mean invstd dx dscale dbias`` (NCHW memory order)."""

    def __init__(self, cand, art, *, n, c, h, w, bufs, eps=1e-5, momentum=0.1):
        from rocke.instances.common.batchnorm2d import batchnorm2d_mcw_grid

        self.cand = cand
        self._keep = bufs  # keep allocations alive (raw addresses only below)
        p = lambda k: bufs[k].data_ptr()  # noqa: E731
        dims = {"N": n, "C": c, "H": h, "W": w}
        if cand.mode == "train":
            self.vals = {
                "X": p("x"),
                "Gamma": p("gamma"),
                "Beta": p("beta"),
                "Y": p("y"),
                "RunningMean": p("rm"),
                "RunningVar": p("rv"),
                "Mean": p("mean"),
                "InvStd": p("invstd"),
                "eps": eps,
                "momentum": momentum,
                **dims,
            }
        else:
            self.vals = bwd_values(bufs, n=n, c=c, h=h, w=w)
        self.launcher = KernelLauncher.from_artifact(art)
        self.cfg = LaunchConfig(
            grid=batchnorm2d_mcw_grid(c, cand.spec("fp16", "gfx950")),
            block=(cand.block_size, 1, 1),
        )

    def __call__(self) -> None:
        self.launcher(self.vals, config=self.cfg)
