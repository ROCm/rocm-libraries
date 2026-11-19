# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Interior-only dX for halo-form dgrad (``KernelIdentity.dx_interior``).

A *halo-form* convolution reads an input that already carries its padding:
pad 0, stride 1, dilation 1 and an odd filter, so the input is the output
grown by ``(F - 1) / 2`` voxels on every side (3-D UNet
layers: ``(Do + 2, Ho + 2, Wo + 2)`` for a 3x3x3 filter). Its dgrad writes
dX over the whole input, the halo shell included. When the model discards the
shell's gradient (it came from a pad whose backward is a slice), only the
interior is needed.

Two launch forms, both without a binary change:

* ``pp_boxes`` kernels: the records (pp_box_sub_gemms(interior=True)) cover
  the interior voxels only -- the box runs / pixel classes start and end
  ``(F - 1) / 2`` inside every axis. The problem, its dense dX store and the
  pp_box_det stage 2 are the halo problem's own.
* every other kernel (the x-halo loop): the view problem below.

The interior is itself a dgrad: interior voxel ``(d + hd, h + hh, w + hw)`` of
the halo problem reads exactly the dY pixels voxel ``(d, h, w)`` of the
*same-size* problem reads -- input extents = output extents, pads
``(hd, hh, hw)``, the same dY and W. So an interior launch is the binary's own
launch on that **view problem**, with the dX pointer moved to voxel
``(hd, hh, hw)`` of the full tensor and the full tensor's dX strides in the
kernargs (the x-halo kernels take their dX strides as kernargs; the pp_boxes box
epilogue does not, hence its record form). ``dx_interior`` is a launch-time
identity field, like ``pp_box_cus``.

* The x-halo loops (``tapk_xhalo``, ``tapk_pp_xhalo``) serve ``pW == 0`` only,
  so their view keeps the full W rows: pads ``(hd, hh, 0)``, extents
  ``(Do, Ho, Wi)``, offset ``(hd, hh, 0)``. The ``hw`` columns at each end of
  an interior row are computed as before (exact values); the D and H shell is
  skipped.
* (:func:`view_pixel_map` maps view pixel indices to full-tensor ones, for a
  dense-index stage such as the pp_box_det stage 2 under a view launch.)

Modes (``dx_interior``): 1 = skip (the shell is not written), 2 = zero-fill
(after the main launch, :func:`build_dgrad_halo_zero` writes zeros over the
whole shell, the x-halo kernels' exact w columns included).
"""

from __future__ import annotations

from dataclasses import dataclass, replace as dc_replace
from typing import Callable, Dict, Tuple

from rocke.core.ir import I32, IRBuilder, KernelDef, PtrType
from rocke.helpers.spec import SignatureBuilder, kernel_name_join
from rocke.helpers.tensor_view import make_buffer_resource

DXI_SKIP = 1
DXI_ZERO = 2
DXI_MODES = (DXI_SKIP, DXI_ZERO)
# dx_interior_axes bits (0 = all): which axes' halo shells are skipped.
DXA_W, DXA_H, DXA_D = 1, 2, 4
DXA_ALL = DXA_W | DXA_H | DXA_D


def dx_interior_axes(identity) -> int:
    """The axes ``identity``'s interior launch drops the shell of (bit mask)."""
    return int(getattr(identity, "dx_interior_axes", 0) or 0) or DXA_ALL


def dx_interior_uses_view(identity) -> bool:
    """Is ``identity``'s interior launched as the view problem? (pp_boxes:
    no -- its records are restricted to the interior of the halo problem.)"""
    return not getattr(identity, "pp_boxes", 0)


def dx_interior_keeps_w(identity) -> bool:
    """Does ``identity``'s interior launch write the full W rows of interior
    (d, h) rows? (The x-halo loop serves pW == 0 only, so its view keeps them;
    so does any launch whose axes leave W out.)"""
    if not dx_interior_axes(identity) & DXA_W:
        return dx_interior_uses_view(identity)
    return dx_interior_uses_view(identity) and bool(
        getattr(identity, "tapk_xhalo", False) or getattr(identity, "tapk_pp_xhalo", 0)
    )


def dx_interior_writes_w(identity) -> bool:
    """Does the interior launch still store the w shell columns of interior
    rows? (The full-W view without the x-halo store mask, xh_dx_wmask.)"""
    return dx_interior_keeps_w(identity) and not getattr(identity, "xh_dx_wmask", False)


def wmask_view(p):
    """``xh_dx_wmask`` on a ``pW = (X-1)/2`` problem: the x-halo loop runs the
    W-extended view (``Wi + 2 hw``, ``pW = 0``; same dY), its dX origin ``hw``
    pixels before dX's (those columns are masked out of the stores).
    Returns ``(view problem, element offset (negative), dX stride kernargs)``."""
    hd, hh, hw = halo_widths(p)
    v = dc_replace(p, Wi=int(p.Wi) + 2 * hw, pW=0)
    s = _full_strides(p)
    vals = {
        "p_dX_stride_n": s["n"],
        "p_dX_stride_hi": s["hi"],
        "p_dX_stride_wi": s["wi"],
    }
    if p.is_3d:
        vals["p_dX_stride_di"] = s["di"]
    return v, -hw * s["wi"], vals


def launch_view(identity, p):
    """``(problem, dX element offset, dX stride kernargs or {}, pixel map or None)``
    an interior / W-masked launch of ``identity`` on ``p`` binds."""
    if getattr(identity, "dx_interior", 0):
        if not dx_interior_uses_view(identity):
            return p, 0, {}, None
        keep_w = dx_interior_keeps_w(identity)
        ax = dx_interior_axes(identity)
        off, vals = interior_view_dx(p, keep_w, ax)
        return (
            interior_view_problem(p, keep_w, ax),
            off,
            vals,
            view_pixel_map(p, keep_w, ax),
        )
    if getattr(identity, "xh_dx_wmask", False):
        v, off, vals = wmask_view(p)
        return v, off, vals, None
    return p, 0, {}, None


def halo_widths(p, axes: int = DXA_ALL) -> Tuple[int, int, int]:
    """``(hd, hh, hw)``: the halo shell's width per axis ((F - 1) / 2; hd = 0 in
    2-D); 0 on an axis ``axes`` leaves out."""
    hd = (int(p.Z) - 1) // 2 if (p.is_3d and axes & DXA_D) else 0
    hh = (int(p.Y) - 1) // 2 if axes & DXA_H else 0
    hw = (int(p.X) - 1) // 2 if axes & DXA_W else 0
    return hd, hh, hw


def dx_interior_problem_ok(p, mask: int = DXA_ALL) -> Tuple[bool, str]:
    """Is ``p`` a halo-form problem (pad 0, stride 1, dilation 1, odd filter)
    on the axes of ``mask``, stride 1 / dilation 1 on all?"""
    axes = [("Y", "sH", "pH", "dH", DXA_H), ("X", "sW", "pW", "dW", DXA_W)]
    if p.is_3d:
        axes.append(("Z", "sD", "pD", "dD", DXA_D))
    for f, s, pad, dil, bit in axes:
        if int(getattr(p, s)) != 1 or int(getattr(p, dil)) != 1:
            return False, "dx_interior serves stride 1 / dilation 1 only"
        if not mask & bit:
            continue
        if int(getattr(p, pad)) != 0:
            return False, "dx_interior serves the halo form (pad 0) only"
        if int(getattr(p, f)) % 2 == 0:
            return False, "dx_interior needs an odd filter"
    if int(p.groups) != 1:
        return False, "dx_interior serves groups == 1 only"
    if p.Ho < 1 or p.Wo < 1 or (p.is_3d and p.Do < 1):
        return False, "dx_interior: empty output"
    return True, "ok"


def dx_interior_launch_conflict(identity, p) -> str:
    """Why a dgrad binary (``identity``: a KernelIdentity or any object with
    its fields) cannot run the interior / W-masked launch on ``p``, or ``""``
    (the launch rules of identity_supports_problem; the view or record problem
    is then checked against the binary itself). C++ twin:
    rocke_dgrad_dx_interior_launch_ok."""
    dxi = int(getattr(identity, "dx_interior", 0) or 0)
    if getattr(identity, "xh_dx_wmask", False) and not dxi:
        hw = (int(getattr(p, "X", 1)) - 1) // 2
        if not identity.tapk_xhalo or hw < 1 or int(getattr(p, "pW", 0)) != hw:
            return "xh_dx_wmask serves pW == (X-1)/2 problems (or a dx_interior launch)"
        if int(getattr(p, "sW", 1)) != 1 or int(getattr(p, "dW", 1)) != 1:
            return "xh_dx_wmask serves stride 1 / dilation 1 W"
        return ""
    if not dxi:
        return ""
    if dxi not in DXI_MODES:
        return f"dx_interior={dxi} is not a mode"
    if not identity.tap_outer_k:
        return "dx_interior serves the stride-1 tap-outer kernels only"
    if identity.split_k > 1 and not (identity.pp_boxes and identity.pp_box_det):
        return "dx_interior: split-K only as pp_box_det (no dX zeroing)"
    if identity.pp_streamk or getattr(identity, "tapk_pp_persist", 0) == 2:
        return "dx_interior: no stream-K (it zeroes dX)"
    ax = int(getattr(identity, "dx_interior_axes", 0) or 0)
    if ax < 0 or ax > 7:
        return "dx_interior_axes is a bit mask 0..7"
    if getattr(identity, "xh_dx_wmask", False) and ax and not ax & 1:
        return "xh_dx_wmask skips the W shell: dx_interior_axes must include W"
    ok, why = dx_interior_problem_ok(p, ax or DXA_ALL)
    return "" if ok else why


def interior_view_problem(p, keep_w: bool, axes: int = DXA_ALL):
    """The (padded) problem whose dX is the halo problem's interior along ``axes``."""
    hd, hh, hw = halo_widths(p, axes)
    kw = {}
    if hh:
        kw.update(Hi=int(p.Hi) - 2 * hh, pH=hh)
    if hw and not keep_w:
        kw.update(Wi=int(p.Wi) - 2 * hw, pW=hw)
    if hd:
        kw.update(Di=int(p.Di) - 2 * hd, pD=hd)
    return dc_replace(p, **kw)


def _full_strides(p) -> Dict[str, int]:
    C = int(p.C)
    sw, sh = C, int(p.Wi) * C
    sd = int(p.Hi) * sh
    sn = (int(p.Di) * sd) if p.is_3d else sd
    return {"n": sn, "di": sd, "hi": sh, "wi": sw}


def interior_view_dx(
    p, keep_w: bool, axes: int = DXA_ALL
) -> Tuple[int, Dict[str, int]]:
    """``(element offset of the view's origin in dX, dX stride kernargs)``."""
    hd, hh, hw = halo_widths(p, axes)
    s = _full_strides(p)
    off = (
        hd * s["di"] * (1 if p.is_3d else 0)
        + hh * s["hi"]
        + (0 if keep_w else hw) * s["wi"]
    )
    vals = {
        "p_dX_stride_n": s["n"],
        "p_dX_stride_hi": s["hi"],
        "p_dX_stride_wi": s["wi"],
    }
    if p.is_3d:
        vals["p_dX_stride_di"] = s["di"]
        if int(p.N) == 1:
            # One image: its stride is never used; the view's own (Do planes)
            # keeps the n-stride check of the plane-gapped linear store
            # (xh_dx_wmask with xh_epi_lin) true.
            vals["p_dX_stride_n"] = s["di"] * (int(p.Di) - 2 * hd)
    return off, vals


def view_pixel_map(p, keep_w: bool, axes: int = DXA_ALL) -> Callable[[int], int]:
    """View dX pixel index ``(n*Di' + d)*Hi'*Wi' + h*Wi' + w`` -> the full
    tensor's ``(n*Di + d + hd)*Hi*Wi + (h + hh)*Wi + w + hw``."""
    v = interior_view_problem(p, keep_w, axes)
    hd, hh, hw = halo_widths(p, axes)
    if keep_w:
        hw = 0
    Dv = int(v.Di) if v.is_3d else 1
    D = int(p.Di) if p.is_3d else 1
    Hv, Wv, H, W = int(v.Hi), int(v.Wi), int(p.Hi), int(p.Wi)

    def f(pix: int) -> int:
        nd, r = divmod(pix, Hv * Wv)
        h, w = divmod(r, Wv)
        n, d = divmod(nd, Dv)
        return ((n * D + d + hd) * H + h + hh) * W + w + hw

    return f


def interior_mask_np(p, axes: int = DXA_ALL):
    """Boolean ``N[D]HW`` mask of the interior voxels (numpy)."""
    import numpy as np

    hd, hh, hw = halo_widths(p, axes)
    shape = (p.N, p.Di, p.Hi, p.Wi) if p.is_3d else (p.N, p.Hi, p.Wi)
    m = np.zeros(shape, dtype=bool)
    if p.is_3d:
        m[:, hd : p.Di - hd, hh : p.Hi - hh, hw : p.Wi - hw] = True
    else:
        m[:, hh : p.Hi - hh, hw : p.Wi - hw] = True
    return m


def written_mask_np(p, keep_w: bool, axes: int = DXA_ALL):
    """The voxels an interior launch writes in skip mode: the interior, plus
    (``keep_w``) the w columns of interior rows."""
    import numpy as np

    hd, hh, hw = halo_widths(p, axes)
    if not keep_w:
        return interior_mask_np(p, axes)
    shape = (p.N, p.Di, p.Hi, p.Wi) if p.is_3d else (p.N, p.Hi, p.Wi)
    m = np.zeros(shape, dtype=bool)
    if p.is_3d:
        m[:, hd : p.Di - hd, hh : p.Hi - hh, :] = True
    else:
        m[:, hh : p.Hi - hh, :] = True
    return m


# ---------------------------------------------------------------------------
# Zero-fill of the halo shell (dx_interior 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DgradHaloZeroSpec:
    """The shell as three flat sections of 16-byte units, one unit per
    thread (``C * elem_bytes`` a multiple of 16; a pixel is ``pu`` units, a
    row ``ru``, a plane ``Hi * ru``):

    1. the ``hd`` first and last planes of every image (2 runs of hd planes);
    2. the ``hh`` first and last rows of every interior plane;
    3. the ``hw`` first and last pixels of every interior row.

    Each section starts on a workgroup boundary (its block range is a
    kernarg), so the section branch is uniform."""

    block: int = 256
    name: str = "conv_dgrad_halo_zero"

    def kernel_name(self) -> str:
        return kernel_name_join(self.name, f"b{self.block}", "v2")


def build_dgrad_halo_zero(
    spec: DgradHaloZeroSpec, *, arch: str = "gfx942"
) -> KernelDef:
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block
    dx_ptr = b.param(
        "dx", PtrType(I32, "global"), noalias=True, writeonly=True, align=16
    )
    dx_bytes = b.param("dx_bytes", I32)
    names = (
        "Di",
        "Do",
        "Hi",
        "Ho",
        "ru",
        "hd",
        "hh",
        "e",
        "L1",
        "L2",
        "L3",
        "n1",
        "n2",
        "u1",
        "u2",
        "u3",
    )
    k = {n: b.param(n, I32) for n in names}
    c0, c1 = b.const_i32(0), b.const_i32(1)
    blk = b.block_id_x()
    tid = b.thread_id_x()
    cb = b.const_i32(spec.block)
    plane = b.mul(k["Hi"], k["ru"])

    def sec1(t):
        # (n, side, offset) over L1 = 2 * hd * plane units per image
        n = b.div(t, k["L1"])
        r = b.sub(t, b.mul(n, k["L1"]))
        half = b.mul(k["hd"], plane)
        side = b.div(r, half)
        o = b.sub(r, b.mul(side, half))
        last = b.mul(b.sub(k["Di"], k["hd"]), plane)
        return b.add(b.add(b.mul(b.mul(n, k["Di"]), plane), b.mul(side, last)), o)

    def sec2(t):
        g = b.div(t, k["L2"])
        r = b.sub(t, b.mul(g, k["L2"]))
        half = b.mul(k["hh"], k["ru"])
        side = b.div(r, half)
        o = b.sub(r, b.mul(side, half))
        n = b.div(g, k["Do"])
        d = b.sub(g, b.mul(n, k["Do"]))
        base = b.mul(b.add(b.mul(n, k["Di"]), b.add(d, k["hd"])), plane)
        last = b.mul(b.sub(k["Hi"], k["hh"]), k["ru"])
        return b.add(b.add(base, b.mul(side, last)), o)

    def sec3(t):
        g = b.div(t, k["L3"])
        r = b.sub(t, b.mul(g, k["L3"]))
        side = b.div(r, k["e"])
        o = b.sub(r, b.mul(side, k["e"]))
        nd = b.div(g, k["Ho"])
        h = b.sub(g, b.mul(nd, k["Ho"]))
        n = b.div(nd, k["Do"])
        d = b.sub(nd, b.mul(n, k["Do"]))
        row = b.add(
            b.mul(b.add(b.mul(n, k["Di"]), b.add(d, k["hd"])), k["Hi"]),
            b.add(h, k["hh"]),
        )
        return b.add(b.add(b.mul(row, k["ru"]), b.mul(side, b.sub(k["ru"], k["e"]))), o)

    in2 = b.cmp_ge(blk, k["n1"])
    in3 = b.cmp_ge(blk, k["n2"])
    sec_blk = b.select(
        in3, b.sub(blk, k["n2"]), b.select(in2, b.sub(blk, k["n1"]), blk)
    )
    t = b.add(b.mul(sec_blk, cb), tid)
    cnt = b.select(in3, k["u3"], b.select(in2, k["u2"], k["u1"]))
    # Each section's decode under its own uniform branch (their divisors may
    # be 0 placeholders for an empty section, which is never entered).
    dx = make_buffer_resource(b, dx_ptr, num_bytes=dx_bytes)
    zero = b.zero_vec(I32, 4)
    oob = b.const_i32(0x7FFFFFF0)
    c16 = b.const_i32(16)
    ok = b.cmp_lt(t, cnt)

    def store(unit):
        off = b.select(ok, b.mul(unit, c16), oob)
        b.buffer_store_vN(dx.rsrc, off, c0, zero)

    with b.scf_if_else(in3) as (then3, else3):
        with then3:
            store(sec3(t))
        with else3:
            with b.scf_if_else(in2) as (then2, else2):
                with then2:
                    store(sec2(t))
                with else2:
                    store(sec1(t))
    del c1
    return b.kernel


def dgrad_halo_zero_signature(spec: DgradHaloZeroSpec) -> list:
    sb = SignatureBuilder().ptr("dx", "i32").scalar("dx_bytes", "i32")
    for n in (
        "Di",
        "Do",
        "Hi",
        "Ho",
        "ru",
        "hd",
        "hh",
        "e",
        "L1",
        "L2",
        "L3",
        "n1",
        "n2",
        "u1",
        "u2",
        "u3",
    ):
        sb = sb.scalar(n, "i32")
    return sb.build()


def halo_zero_units(p, elem_bytes: int, axes: int = DXA_ALL) -> Dict[str, int]:
    """Section sizes (16-byte units) of the shell of ``p``'s dX along ``axes``
    (see DgradHaloZeroSpec; ``Do`` / ``Ho`` are the interior plane / row counts)."""
    hd, hh, hw = halo_widths(p, axes)
    pix = int(p.C) * elem_bytes
    if pix % 16:
        raise ValueError("dx_interior zero-fill needs C * elem_bytes % 16 == 0")
    pu = pix // 16
    N = int(p.N)
    Di = int(p.Di) if p.is_3d else 1
    Do = Di - 2 * hd
    Hi, ru = int(p.Hi), int(p.Wi) * pu
    Ho = Hi - 2 * hh
    e = hw * pu
    L1, L2, L3 = 2 * hd * Hi * ru, 2 * hh * ru, 2 * e
    return dict(
        Di=Di,
        Do=Do,
        Hi=Hi,
        Ho=Ho,
        ru=ru,
        hd=hd,
        hh=hh,
        e=e,
        L1=L1,
        L2=L2,
        L3=L3,
        u1=N * L1,
        u2=N * Do * L2,
        u3=N * Do * Ho * L3,
    )


def halo_zero_launch(
    p,
    elem_bytes: int,
    dx_ptr: int,
    dx_bytes: int,
    spec: DgradHaloZeroSpec,
    axes: int = DXA_ALL,
):
    """``(values, grid, block)`` of the shell zero-fill of the full dX at ``dx_ptr``."""
    u = halo_zero_units(p, elem_bytes, axes)
    B = spec.block
    b1, b2, b3 = (-(-u[k] // B) for k in ("u1", "u2", "u3"))
    values = {"dx": dx_ptr, "dx_bytes": dx_bytes}
    for k in ("Di", "Do", "Hi", "Ho", "ru", "hd", "hh"):
        values[k] = u[k]
    # Placeholder divisors of an empty section. A non-empty section is the
    # only one its blocks enter, but a fully empty shell (1-tap filters) still
    # launches block 0, which takes the section-3 branch with every store out
    # of bounds: its divisors (e, L3) must not be 0 either.
    for k in ("e", "L1", "L2", "L3"):
        values[k] = max(1, u[k])
    values.update(n1=b1, n2=b1 + b2, u1=u["u1"], u2=u["u2"], u3=u["u3"])
    return values, (max(1, b1 + b2 + b3), 1, 1), (B, 1, 1)
