# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Drop-in call of the 3-D yx Winograd kernels (opt-in; not wired into any default path).

:class:`Winograd3dDropin` turns a :class:`~kernels.common.conv_winograd3d_select.Winograd3dPlan`
(``mode="dropin"``) for a 3-D conv problem into the launches of one NDHWC-in / NDHWC-out call:

1. one auxiliary launch (``build_yx_aux``): X (FWD) or dY (BWD-data) relayout NDHWC -> c8 into
   a workspace with the zero halo the problem needs (halo form: 0 FWD / 2 BWD-data; adapter
   pad (1,1,1): 1 / 1; pad (0,1,1): z 0 / 2, y x 1), the per-block amax map of the prescale
   (``bps=1``) and the weight transform (flipped for BWD-data), side by side;
2. the main kernel (``build_winograd_yx``), writing Y / dX as NDHWC bf16;
3. with split-K (``ksplit``), the ordered f32 reduction (``build_ks_reduce``).

A plan whose spec reads NDHWC directly (``xin="ndhwc"``: the zero halo by predication
(``xpad`` / ``xpad_z``), the prescale in-kernel (``bps=3``)) skips the relayout: its only
auxiliary launch is the weight transform. The fill's 16-byte lane loads then touch one
128-byte line each (TA-bound), so the selector never produces them; they are built from an
explicit spec.

Everything is deterministic (plain stores, ordered split-K reduction). Tensors are torch
tensors on the device (the launcher takes their data pointers); the halo of the c8 workspace
is zeroed once, when the workspace is allocated. Usage::

    plan = select_winograd3d(problem, "fwd", mode="dropin")
    if plan.kind == "yx":
        call = Winograd3dDropin(problem, "fwd", plan)
        y = call(x, w)            # x [D, H, W, C] bf16, w [K, 3, 3, 3, C] bf16 -> [Do, Ho, Wo, K]
        dx = Winograd3dDropin(problem, "dgrad", plan_d)(dy, w)   # dY -> dX, same forward filter
"""

from __future__ import annotations

from typing import Optional


class Winograd3dDropin:
    """Built kernels + workspaces of one drop-in yx Winograd call (see the module doc)."""

    def __init__(
        self,
        problem,
        direction: str,
        plan,
        *,
        arch: str = "gfx942",
        device: str = "cuda",
    ):
        import torch

        from kernels.common import conv_winograd3d as cw
        from kernels.common import conv_winograd3d_select as sel
        from rocke.helpers import compile_kernel
        from rocke.helpers.spec import SignatureBuilder
        from rocke.runtime.launcher import KernelLauncher, LaunchConfig

        if plan.kind != "yx" or plan.mode != "dropin":
            raise ValueError(f"needs a yx drop-in plan, got {plan.kind} / {plan.mode}")
        if direction != plan.direction:
            raise ValueError(f"plan is for {plan.direction}, not {direction}")
        self.direction = direction
        hz, h = sel._relayout_halos(problem, direction)
        Di, Hi, Wi, C, K = sel._winograd_problem(problem, direction)
        kw = plan.spec_dict
        self.spec = spec = cw.WinogradYxSpec(Di=Di, Hi=Hi, Wi=Wi, C=C, K=K, **kw)
        ok, why = cw.is_valid_yx_spec(spec)
        if not ok:
            raise ValueError(f"invalid spec: {why}")
        self.bps = spec.bps
        # NDHWC-direct main kernel (xin "ndhwc"): reads X / dY in place, the zero halo by
        # predication, the prescale in-kernel (bps=3) -- no relayout, only the weight transform
        self.ndhwc = spec.xin == "ndhwc"
        if self.ndhwc:
            if self.bps not in (0, 3) or spec.oscale or spec.yc8h:
                raise ValueError("drop-in NDHWC: bps 0 / 3, no oscale / yc8h")
            if cw._xpad(spec) != (hz, h):
                raise ValueError("xpad does not match the problem's zero halo")
        elif self.bps not in (0, 1) or spec.oscale or spec.yc8h:
            raise ValueError("drop-in: bps 0 / 1, no oscale / yc8h")
        if self.bps == 1 and (spec.bps_halo != h or cw._bps_hz(spec) != hz):
            raise ValueError("bps halo does not match the relayout halo")
        # source (unpadded) dims of the relayout, the main kernel's output dims
        self.src = (Di - 2 * hz, Hi - 2 * h, Wi - 2 * h, C)
        self.out = (spec.Do, spec.Ho, spec.Wo, K)
        halo = h if hz == h else (hz, h, h)
        odt = "f16" if spec.op_dtype == "f16" else "bf16"
        wtv = plan.wtv or 1
        wt3 = (
            wtv in (3, 4)
            and odt == "f16"
            and spec.stage == "lds16"
            and not spec.b16
            and not spec.bc
        )
        dgrad = direction == "dgrad"
        if self.ndhwc:
            if wt3:
                ka = cw.build_yx_weight_transform_v3(
                    K, C, flip=dgrad, swz=wtv == 4, my=spec.my
                )
                ga = cw.yx_weight_transform_v3_grid(K, C)
            else:
                ka = cw.build_yx_weight_transform(
                    K,
                    C,
                    out_dtype=odt,
                    fold=spec.stage in ("reg", "wlds"),
                    u16=spec.b16,
                    uc=spec.bc,
                    flip=dgrad,
                )
                ga = cw.yx_weight_transform_grid(K, C)
            sga = SignatureBuilder().ptr("W", "bf16").ptr("U", odt)
            self._caux = LaunchConfig(grid=ga, block=(256, 1, 1))
        else:
            ka = cw.build_yx_aux(
                *self.src,
                K,
                out_dtype=odt,
                fold=spec.stage in ("reg", "wlds"),
                rev=spec.xrev,
                halo=halo,
                flip=dgrad,
                wtv=(wtv if wt3 else 1),
                amax=self.bps == 1,
                my=spec.my,
            )
            sga = (
                SignatureBuilder()
                .ptr("X", "bf16")
                .ptr("X8", "bf16")
                .ptr("W", "bf16")
                .ptr("U", odt)
            )
            if self.bps == 1:
                sga = sga.ptr("E", "i32")
            self._caux = LaunchConfig(
                grid=cw.yx_aux_grid(*self.src, K, wtv=(wtv if wt3 else 1)),
                block=(256, 1, 1),
            )
        self._aux = KernelLauncher(
            hsaco=compile_kernel(ka, arch=arch).hsaco,
            kernel_name=ka.name,
            signature=sga.build(),
        )
        km = cw.build_winograd_yx(spec)
        self.ks = spec.ksplit
        sgm = (
            SignatureBuilder()
            .ptr("X", "bf16")
            .ptr("U", odt)
            .ptr("Y", "f32" if self.ks > 1 else "bf16")
        )
        if self.bps in (1, 2):
            sgm = sgm.ptr("E", "i32")
        self._main = KernelLauncher(
            hsaco=compile_kernel(km, arch=arch).hsaco,
            kernel_name=km.name,
            signature=sgm.build(),
        )
        self._cmain = LaunchConfig(
            grid=cw.winograd_yx_grid(spec), block=(spec.block_size, 1, 1)
        )
        self.kernel_names = [ka.name, km.name]
        nel = spec.Do * spec.Ho * spec.Wo * K
        if self.ks > 1:
            kr = cw.build_ks_reduce(nel, self.ks)
            self._red = KernelLauncher(
                hsaco=compile_kernel(kr, arch=arch).hsaco,
                kernel_name=kr.name,
                signature=SignatureBuilder().ptr("P", "f32").ptr("Y", "bf16").build(),
            )
            self._cred = LaunchConfig(grid=cw.ks_reduce_grid(nel), block=(256, 1, 1))
            self.kernel_names.append(kr.name)
        # workspaces: c8 input with a zero halo (zeroed once; the relayout writes the interior),
        # transformed weights, amax map, split-K partials
        self.X8 = (
            None
            if self.ndhwc
            else torch.zeros(C // 8, Di, Hi, Wi, 8, device=device, dtype=torch.bfloat16)
        )
        udt = torch.float16 if odt == "f16" else torch.bfloat16
        self.U = torch.empty(3 * spec.nbp * 4 * C * K, device=device, dtype=udt)
        self.E = (
            torch.zeros(
                cw.x_to_c8_lds_grid(*self.src)[0], device=device, dtype=torch.int32
            )
            if self.bps == 1
            else None
        )
        self.P = (
            torch.empty(self.ks * nel, device=device, dtype=torch.float32)
            if self.ks > 1
            else None
        )

    def __call__(self, x, w, y: Optional[object] = None):
        """x: [D, H, W, C] (or [1, D, H, W, C]) bf16 -- X for FWD, dY for BWD-data; w: the
        forward filter [K, 3, 3, 3, C] bf16 (both directions); y: optional output
        [Do, Ho, Wo, K] bf16 (Y / dX), allocated when None. Returns y."""
        import torch

        x = x.reshape(self.src) if x.dim() == 5 else x
        if (
            tuple(x.shape) != self.src
            or x.dtype != torch.bfloat16
            or not x.is_contiguous()
        ):
            raise ValueError(
                f"x: contiguous bf16 {self.src}, got {tuple(x.shape)} {x.dtype}"
            )
        if y is None:
            y = torch.empty(self.out, device=x.device, dtype=torch.bfloat16)
        ea = {"E": self.E} if self.bps == 1 else {}
        if self.ndhwc:
            self._aux({"W": w, "U": self.U}, config=self._caux)
        else:
            self._aux(
                {"X": x, "X8": self.X8, "W": w, "U": self.U, **ea}, config=self._caux
            )
        xin = x if self.ndhwc else self.X8
        if self.ks > 1:
            self._main({"X": xin, "U": self.U, "Y": self.P, **ea}, config=self._cmain)
            self._red({"P": self.P, "Y": y}, config=self._cred)
        else:
            self._main({"X": xin, "U": self.U, "Y": y, **ea}, config=self._cmain)
        return y
