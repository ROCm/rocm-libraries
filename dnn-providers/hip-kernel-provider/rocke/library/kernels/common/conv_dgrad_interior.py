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


def dx_interior_uses_view(identity) -> bool:
    """Is ``identity``'s interior launched as the view problem? (pp_boxes:
    no -- its records are restricted to the interior of the halo problem.)"""
    return not getattr(identity, "pp_boxes", 0)


def dx_interior_keeps_w(identity) -> bool:
    """Does ``identity``'s interior launch write the full W rows of interior
    (d, h) rows? (The x-halo loop serves pW == 0 only, so its view keeps them.)"""
    return dx_interior_uses_view(identity) and bool(getattr(identity, "tapk_xhalo", False))


def halo_widths(p) -> Tuple[int, int, int]:
    """``(hd, hh, hw)``: the halo shell's width per axis ((F - 1) / 2; hd = 0 in 2-D)."""
    hd = (int(p.Z) - 1) // 2 if p.is_3d else 0
    return hd, (int(p.Y) - 1) // 2, (int(p.X) - 1) // 2


def dx_interior_problem_ok(p) -> Tuple[bool, str]:
    """Is ``p`` a halo-form problem (pad 0, stride 1, dilation 1, odd filter)?"""
    axes = [("Y", "sH", "pH", "dH"), ("X", "sW", "pW", "dW")]
    if p.is_3d:
        axes.append(("Z", "sD", "pD", "dD"))
    for f, s, pad, dil in axes:
        if int(getattr(p, s)) != 1 or int(getattr(p, dil)) != 1:
            return False, "dx_interior serves stride 1 / dilation 1 only"
        if int(getattr(p, pad)) != 0:
            return False, "dx_interior serves the halo form (pad 0) only"
        if int(getattr(p, f)) % 2 == 0:
            return False, "dx_interior needs an odd filter"
    if int(p.groups) != 1:
        return False, "dx_interior serves groups == 1 only"
    if p.Ho < 1 or p.Wo < 1 or (p.is_3d and p.Do < 1):
        return False, "dx_interior: empty output"
    return True, "ok"


def interior_view_problem(p, keep_w: bool):
    """The same-size (padded) problem whose dX is the halo problem's interior."""
    hd, hh, hw = halo_widths(p)
    kw = dict(Hi=p.Ho, pH=hh)
    if not keep_w:
        kw.update(Wi=p.Wo, pW=hw)
    if p.is_3d:
        kw.update(Di=p.Do, pD=hd)
    return dc_replace(p, **kw)


def _full_strides(p) -> Dict[str, int]:
    C = int(p.C)
    sw, sh = C, int(p.Wi) * C
    sd = int(p.Hi) * sh
    sn = (int(p.Di) * sd) if p.is_3d else sd
    return {"n": sn, "di": sd, "hi": sh, "wi": sw}


def interior_view_dx(p, keep_w: bool) -> Tuple[int, Dict[str, int]]:
    """``(element offset of the view's origin in dX, dX stride kernargs)``."""
    hd, hh, hw = halo_widths(p)
    s = _full_strides(p)
    off = hd * s["di"] * (1 if p.is_3d else 0) + hh * s["hi"] + (0 if keep_w else hw) * s["wi"]
    vals = {
        "p_dX_stride_n": s["n"],
        "p_dX_stride_hi": s["hi"],
        "p_dX_stride_wi": s["wi"],
    }
    if p.is_3d:
        vals["p_dX_stride_di"] = s["di"]
    return off, vals


def view_pixel_map(p, keep_w: bool) -> Callable[[int], int]:
    """View dX pixel index ``(n*Di' + d)*Hi'*Wi' + h*Wi' + w`` -> the full
    tensor's ``(n*Di + d + hd)*Hi*Wi + (h + hh)*Wi + w + hw``."""
    v = interior_view_problem(p, keep_w)
    hd, hh, hw = halo_widths(p)
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


def interior_mask_np(p):
    """Boolean ``N[D]HW`` mask of the interior voxels (numpy)."""
    import numpy as np

    hd, hh, hw = halo_widths(p)
    shape = (p.N, p.Di, p.Hi, p.Wi) if p.is_3d else (p.N, p.Hi, p.Wi)
    m = np.zeros(shape, dtype=bool)
    if p.is_3d:
        m[:, hd:p.Di - hd, hh:p.Hi - hh, hw:p.Wi - hw] = True
    else:
        m[:, hh:p.Hi - hh, hw:p.Wi - hw] = True
    return m


def written_mask_np(p, keep_w: bool):
    """The voxels an interior launch writes in skip mode: the interior, plus
    (``keep_w``) the w columns of interior rows."""
    import numpy as np

    hd, hh, hw = halo_widths(p)
    if not keep_w:
        return interior_mask_np(p)
    shape = (p.N, p.Di, p.Hi, p.Wi) if p.is_3d else (p.N, p.Hi, p.Wi)
    m = np.zeros(shape, dtype=bool)
    if p.is_3d:
        m[:, hd:p.Di - hd, hh:p.Hi - hh, :] = True
    else:
        m[:, hh:p.Hi - hh, :] = True
    return m


# ---------------------------------------------------------------------------
# Zero-fill of the halo shell (dx_interior 2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DgradHaloZeroSpec:
    """One workgroup per dX row ``(n, d, h)``: a shell row (d or h in the
    halo) is zeroed whole, an interior row only its ``hw`` pixels at each
    end. 16-byte stores; the row and pixel extents are kernargs in 16-byte
    units (``C * elem_bytes`` must be a multiple of 16)."""

    block: int = 64
    name: str = "conv_dgrad_halo_zero"

    def kernel_name(self) -> str:
        return kernel_name_join(self.name, f"b{self.block}")


def build_dgrad_halo_zero(spec: DgradHaloZeroSpec) -> KernelDef:
    b = IRBuilder(spec.kernel_name())
    b.kernel.attrs["max_workgroup_size"] = spec.block
    dx_ptr = b.param("dx", PtrType(I32, "global"), noalias=True, writeonly=True, align=16)
    dx_bytes = b.param("dx_bytes", I32)
    c_Di = b.param("Di", I32)
    c_Hi = b.param("Hi", I32)
    row_u = b.param("row_units", I32)  # Wi * C * eb / 16
    edge_u = b.param("edge_units", I32)  # hw * C * eb / 16 (one end)
    c_hd = b.param("hd", I32)
    c_hh = b.param("hh", I32)
    c0 = b.const_i32(0)
    tid = b.thread_id_x()
    rid = b.block_id_x()
    nd = b.div(rid, c_Hi)
    h = b.sub(rid, b.mul(nd, c_Hi))
    d = b.sub(nd, b.mul(b.div(nd, c_Di), c_Di))
    shell = b.lor(
        b.lor(b.cmp_lt(d, c_hd), b.cmp_ge(d, b.sub(c_Di, c_hd))),
        b.lor(b.cmp_lt(h, c_hh), b.cmp_ge(h, b.sub(c_Hi, c_hh))),
    )
    two_e = b.add(edge_u, edge_u)
    n_units = b.select(shell, row_u, two_e)
    # Units of the right edge start at row_u - 2*edge (the left edge's i < edge
    # maps to itself): unit(i) = i + (i >= edge ? row_u - 2*edge : 0).
    right_shift = b.select(shell, c0, b.sub(row_u, two_e))
    base = b.mul(rid, row_u)
    dx = make_buffer_resource(b, dx_ptr, num_bytes=dx_bytes)
    zero = b.zero_vec(I32, 4)
    c_blk = b.const_i32(spec.block)
    oob = b.const_i32(0x7FFFFFF0)
    trips = b.div(b.add(n_units, b.const_i32(spec.block - 1)), c_blk)
    loop = b.scf_for(c0, trips, b.const_i32(1), iv_name="zt")
    with loop as j:
        i = b.add(b.mul(j, c_blk), tid)
        u = b.add(i, b.select(b.cmp_ge(i, edge_u), right_shift, c0))
        off = b.mul(b.add(base, u), b.const_i32(16))
        off = b.select(b.cmp_lt(i, n_units), off, oob)
        b.buffer_store_vN(dx.rsrc, off, c0, zero)
    return b.kernel


def dgrad_halo_zero_signature(spec: DgradHaloZeroSpec) -> list:
    return (
        SignatureBuilder()
        .ptr("dx", "i32")
        .scalar("dx_bytes", "i32")
        .scalar("Di", "i32")
        .scalar("Hi", "i32")
        .scalar("row_units", "i32")
        .scalar("edge_units", "i32")
        .scalar("hd", "i32")
        .scalar("hh", "i32")
        .build()
    )


def halo_zero_launch(p, elem_bytes: int, dx_ptr: int, dx_bytes: int, spec: DgradHaloZeroSpec):
    """``(values, grid, block)`` of the shell zero-fill of the full dX at ``dx_ptr``."""
    hd, hh, hw = halo_widths(p)
    pix = int(p.C) * elem_bytes
    if pix % 16:
        raise ValueError("dx_interior zero-fill needs C * elem_bytes % 16 == 0")
    D = int(p.Di) if p.is_3d else 1
    rows = int(p.N) * D * int(p.Hi)
    values = {
        "dx": dx_ptr,
        "dx_bytes": dx_bytes,
        "Di": D,
        "Hi": int(p.Hi),
        "row_units": int(p.Wi) * pix // 16,
        "edge_units": hw * pix // 16,
        "hd": hd,
        "hh": hh,
    }
    return values, (rows, 1, 1), (spec.block, 1, 1)
