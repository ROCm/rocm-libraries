# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Shape-generic ring GETT: rocKE's native tensor contraction body with its index maps as kernel
arguments (or, for hipTensor's JIT, as constants).

``rocke.instances.common.tensor.gett`` computes ``D = alpha * sum_K A B (+ beta * C)`` (f16 /
bf16 data, f32 accumulation, gfx950) in one launch of the 256 x 256 direct-to-LDS ring GEMM body,
reading both operands in place through per-lane mode offsets and writing D through a permuted
epilogue. Its index maps (the mixed-radix digits of the GEMM rows of each operand, of the K space
and of D's rows and columns) are compile-time constants there: one kernel per problem. Here the
same body takes them from kernel arguments (the hooks' digit functions use magic-number
division), so one kernel serves every problem of its class, and hipTensor's kernel pack carries
the classes. With ``geom`` the builder makes them constants again: the problem-specific kernel of
one problem (the class recipe is rolled over every geometry value; hipTensor's JIT does not build
it).

Class (:class:`RgettSpec`): data type, bilinear, the DTL lane width of each operand (``vec_a`` /
``vec_b``: 8 elements = 16-byte lanes when the operand's fastest run and every other stride are
multiples of 8, else 2 elements = 4-byte lanes), the D vector width ``vec_d`` (1, 2, 4) and
``epi_pair`` (16-byte D stores, 8-element D runs), ``wide_d`` (D past 2^31 bytes), ``slots`` (the
digits per digit space the kernel decomposes: a class for problems of at most 2 or 4 digits per
space skips the magic-number division of the unused slots, which costs about a tenth of the main
loop on the doc rows when every slot is decomposed), ``c_prefetch`` (bilinear: C loads issued
two atom rows ahead) and the ring body's levers (K tile, DTL stages, chiplet raster).

Geometry (i32 kernel arguments; the host plan :func:`.gett_host.rgett_plan`): ``ra*`` / ``rb*``
the GEMM rows of the kernel's A and B operands (extent ``e``, magic pair ``m, s`` of every digit
but the last, operand stride ``S``), ``dr*`` / ``dc*`` the same rows as D's rows and columns (D
strides), ``kl*`` the in-tile K digits and ``kt*`` the K tile digits (strides ``A`` / ``B`` of the
two operands), ``kps`` / ``kpl`` the padded in-tile K digit (``kps`` -1: none) and its real
extent. The GEMM's ``M``, ``N``, ``K`` arguments are the row counts and the padded K. D (and C)
must span less than 2^31 bytes, each operand less than the DMA buffer range.

Kernel ABI ``hiptensor-aot-rgett/v1``: ``A, B, C`` (D, the output), ``M, N, K``, ``alpha``
[``Cin, beta``], then the geometry. Launch: grid ``(ceil(N / 256), ceil(M / 256), 1)``, block
256. Mirrored by the C++ engine (``rocke_build_htaot_rgett``).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from rocke.core.ir import I32, IRBuilder, KernelDef, Value
from rocke.helpers.magic_division import do_magic_division_u31_dynamic
from rocke.instances.common.tensor.gett import (
    MAX_D_BYTES,
    NARROW_D_BYTES,
    OOB,
    GettHooks,
    GettKnobs,
    body_spec,
)
from rocke.instances.common.tensor.gett_plan import GettPlan, OpGeom

from .common import require_bool

RGETT_ABI = "hiptensor-aot-rgett/v1"

#: digit slots of the GEMM rows of an operand (and of D's rows / columns), the in-tile K digits
#: and the K tile digits
SLOTS_ROW = 8
SLOTS_KL = 6
SLOTS_KT = 8

#: (prefix, stride names) of every digit space of the geometry
SPACES = (
    ("ra", ("S",), SLOTS_ROW),
    ("rb", ("S",), SLOTS_ROW),
    ("dr", ("S",), SLOTS_ROW),
    ("dc", ("S",), SLOTS_ROW),
    ("kl", ("A", "B"), SLOTS_KL),
    ("kt", ("A", "B"), SLOTS_KT),
)

ARCHS = ("gfx950",)


@dataclass(frozen=True)
class RgettSpec:
    dtype: str = "f16"  # f16 | bf16 (A, B, C and D)
    bilinear: bool = False
    vec_a: int = 8  # DTL lane width (elements) of the kernel's A operand: 8 or 2
    vec_b: int = 8
    vec_d: int = 4  # D elements per store: 1, 2 or 4
    epi_pair: bool = False  # 16-byte D stores (D runs of 8, vec_d 4)
    # D (and C) past 2^31 bytes (up to MAX_D_BYTES): the epilogue masks the summed offset
    wide_d: bool = False
    # bilinear: C loads issued this many atom rows ahead of their use (0: per row)
    c_prefetch: int = 0
    # digits per digit space the kernel decomposes (0: every slot); the geometry arguments are
    # the same, the slots past ``slots`` must have extent 1
    slots: int = 0
    tile_k: int = 32
    dtl_stages: int = 3
    chiplet_wgm: int = 8
    chiplet_chunk_size: int = 64
    #: the geometry constants of a problem-specific kernel (``(name, value)`` for every name of
    #: :func:`geometry_names`; :meth:`with_geom`), None: the class kernel (geometry arguments)
    geom: Optional[Tuple[Tuple[str, int], ...]] = None

    def with_geom(self, geom: Optional[Dict[str, int]]) -> "RgettSpec":
        """The class with the geometry values ``geom`` (None: the class kernel)."""
        g = (
            None
            if geom is None
            else tuple(sorted((k, int(v)) for k, v in geom.items()))
        )
        return dataclasses.replace(self, geom=g)

    def check(self) -> None:
        require_bool(self, "bilinear", "epi_pair", "wide_d")
        if self.dtype not in ("f16", "bf16"):
            raise ValueError("dtype must be f16 or bf16")
        if self.vec_a not in (8, 2) or self.vec_b not in (8, 2):
            raise ValueError("vec_a / vec_b must be 8 or 2")
        if self.vec_d not in (1, 2, 4):
            raise ValueError("vec_d must be 1, 2 or 4")
        if self.epi_pair and self.vec_d != 4:
            raise ValueError("epi_pair needs vec_d 4")
        if self.tile_k not in (32, 64):
            raise ValueError("tile_k must be 32 or 64")
        if self.c_prefetch and not self.bilinear:
            raise ValueError("c_prefetch needs bilinear")
        if self.slots and not 1 <= self.slots <= min(SLOTS_KL, SLOTS_KT, SLOTS_ROW):
            raise ValueError("slots must be 0 or 1 .. the smallest digit space")

    def kernel_name(self) -> str:
        return (
            f"hiptensor_aot_rgett_{self.dtype}_{'bil' if self.bilinear else 'scl'}"
            f"_a{self.vec_a}_b{self.vec_b}_d{self.vec_d}{'p' if self.epi_pair else ''}"
            f"_t{self.tile_k}s{self.dtl_stages}w{self.chiplet_wgm}c{self.chiplet_chunk_size}"
            + ("_wd" if self.wide_d else "")
            + (f"_cp{self.c_prefetch}" if self.c_prefetch else "")
            + (f"_n{self.slots}" if self.slots else "")
        )

    def key(self) -> str:
        """The JIT recipe key of the class."""
        return "rgett:" + self.kernel_name().replace(
            "hiptensor_aot_rgett", "hiptensor_jit_rgett", 1
        )


def geometry_names() -> Tuple[str, ...]:
    """Every geometry argument, in kernel argument order."""
    out = []
    for prefix, strides, n in SPACES:
        for i in range(n):
            out.append(f"{prefix}e{i}")
            if i < n - 1:
                out += [f"{prefix}m{i}", f"{prefix}s{i}"]
            out += [f"{prefix}{s}{i}" for s in strides]
    return tuple(out) + ("kps", "kpl")


class _Rt:
    """A digit space of the runtime geometry (stands in a GettPlan digit list)."""

    def __init__(self, prefix: str, stride: str):
        self.prefix, self.stride = prefix, stride

    def __len__(self):  # the base hooks only test emptiness
        return 1


class RtGettHooks(GettHooks):
    """The ring GETT hooks with the index maps of the class geometry: kernel arguments (``geom``
    None) or the constants ``geom``."""

    def __init__(
        self,
        plan,
        knobs,
        spec,
        dtype,
        geom: Optional[Dict[str, int]],
        wide: bool = False,
        slots: int = 0,
    ):
        super().__init__(plan, knobs, spec, dtype=dtype)
        self.wide = wide
        self.slots = slots
        self.geom = geom
        self.g: Dict[str, Value] = {}

    # ------------------------------------------------------------------ params
    def add_params(self, b: IRBuilder) -> None:
        super().add_params(b)
        self.dims = {"M": b.get_param("M"), "N": b.get_param("N")}
        if self.geom is None:
            for name in geometry_names():
                self.g[name] = b.param(name, I32)

    def _v(self, b: IRBuilder, name: str) -> Value:
        if self.geom is None:
            return self.g[name]
        return b.const_i32(int(self.geom[name]))

    def _space(self, b: IRBuilder, x: Value, prefix: str, strides):
        """(offset per stride name, digit values) of index ``x`` over a digit space."""
        n = dict((p, k) for p, _s, k in SPACES)[prefix]
        if self.slots:
            n = min(n, self.slots)
        offs = {s: None for s in strides}
        digits = []
        rest = x
        for i in range(n):
            e = self._v(b, f"{prefix}e{i}")
            if i == n - 1:
                d, q = rest, None
            elif self.geom is None:
                # magic-number division (common.Emit.udiv)
                m, s = self._v(b, f"{prefix}m{i}"), self._v(b, f"{prefix}s{i}")
                q = do_magic_division_u31_dynamic(b, rest, m, s)
                d = b.sub(rest, b.mul(q, e))
            else:
                q = b.div(rest, e)
                d = b.mod(rest, e)
            digits.append(d)
            for st in strides:
                t = b.mul(d, self._v(b, f"{prefix}{st}{i}"))
                offs[st] = t if offs[st] is None else b.add(offs[st], t)
            rest = q
        return offs, digits

    def _digits(self, b, x, digits):
        if isinstance(digits, _Rt):
            return self._space(b, x, digits.prefix, (digits.stride,))[0][digits.stride]
        return super()._digits(b, x, digits)

    # ------------------------------------------------------------------ operand addressing
    def lane_split(self, nm: str) -> bool:
        return False

    def _klocal_expr(self, b, nm, k):
        return self._space(b, k, "kl", (nm,))[0][nm]

    def _k_valid(self, b, k):
        return self._kpad_ok(b, self._space(b, k, "kl", ())[1])

    def _kpad_ok(self, b, digits):
        """The in-tile K digits ``digits`` are not padding."""
        kps, kpl = self._v(b, "kps"), self._v(b, "kpl")
        ok = None
        for i, d in enumerate(digits):
            t = b.lor(b.cmp_ne(kps, b.const_i32(i)), b.cmp_lt(d, kpl))
            ok = t if ok is None else b.land(ok, t)
        return ok

    def tile_voffs(self, b, nm, mn0):
        op = self.plan.ops[nm]
        rows = self.dims["M" if nm == "A" else "N"]
        np_ = self.passes(nm)
        groups: Dict[Tuple[int, ...], int] = {}
        rep = []
        for p in range(np_):
            key = tuple(self._lane_geom(nm, p, t)[1] for t in range(self.bs))
            if key not in groups:
                groups[key] = p
            rep.append(groups[key])
        rowoff: Dict[int, Tuple[Value, Value]] = {}
        out = []
        for p in range(np_):
            k_v, r_v = self._lane_expr(b, nm, p)
            gp = rep[p]
            if gp not in rowoff:
                if gp != p:
                    _k, r_v = self._lane_expr(b, nm, gp)
                row = b.add(mn0, r_v)
                ok = b.cmp_lt(row, rows)
                ro = self._digits(b, b.select(ok, row, b.const_i32(0)), op.row_digits)
                rowoff[gp] = (ro, ok)
            ro, ok = rowoff[gp]
            offs, digits = self._space(b, k_v, "kl", (nm,))
            ok = b.land(ok, self._kpad_ok(b, digits))
            v = b.mul(b.add(ro, offs[nm]), b.const_i32(2))
            out.append(b.select(ok, v, b.const_i32(OOB)))
        return out

    def ktile_off_bytes(self, b, nm, k_ld):
        kt = b.div(k_ld, b.const_i32(self.T))
        off = self._space(b, kt, "kt", (nm,))[0][nm]
        return b.to_sgpr_u32(b.mul(off, b.const_i32(2)))

    # ------------------------------------------------------------------ epilogue
    def d_bytes(self) -> int:
        # narrow: every in-range D offset is below 2^31 bytes and the row / column sentinel OOB
        # is the buffer range, so rows and columns past the tensor are dropped; wide: the summed
        # element offset is masked (the base hooks' wide-D epilogue)
        return MAX_D_BYTES if self.wide else NARROW_D_BYTES

    def wide_d(self) -> bool:
        return self.wide

    def _d_part(self, b, idx, digits, lim, big):
        lim = self.dims["M" if digits is self.plan.d_row_digits else "N"]
        e = self._digits(b, idx, digits)
        if self.wide:
            return (e, b.cmp_lt(idx, lim))
        off = b.mul(e, b.const_i32(2))
        return b.select(b.cmp_lt(idx, lim), off, big)


def _plan(spec: RgettSpec) -> GettPlan:
    """The class's stand-in plan: its structure (lane widths, D vector width) with runtime
    digit spaces; the row counts are not known (every tile is treated as ragged)."""
    ops = {
        "A": OpGeom("A", [], 1, _Rt("ra", "S"), spec.vec_a, {}, 0),
        "B": OpGeom("B", [], 1, _Rt("rb", "S"), spec.vec_b, {}, 0),
    }
    return GettPlan(
        modes_a=[],
        modes_b=[],
        modes_d=[],
        ext={},
        swapped=False,
        M=1,
        N=1,
        K=1,
        tile_k=spec.tile_k,
        k_order=[],
        k_local=[],
        k_tile=[],
        ops=ops,
        d_row_digits=_Rt("dr", "S"),
        d_col_digits=_Rt("dc", "S"),
        vec_d=spec.vec_d,
        strides_d={},
        vec_d8=spec.epi_pair,
        k_pad=None,
        kc=False,
        vec_dc8=False,
    )


def build_rgett(spec: RgettSpec, *, arch: str = "gfx950") -> KernelDef:
    """The class kernel (``spec.geom`` None: geometry arguments) or its problem-specific form
    (:meth:`RgettSpec.with_geom`)."""
    from rocke.instances.common.gemm_universal import build_universal_gemm

    spec.check()
    geom = None if spec.geom is None else dict(spec.geom)
    if arch not in ARCHS:
        raise ValueError(f"the ring GETT body needs {ARCHS}")
    name = spec.kernel_name()
    if geom is not None:
        name = name.replace("hiptensor_aot_rgett", "hiptensor_jit_rgett", 1)
    body = body_spec(
        "fp16" if spec.dtype == "f16" else "bf16",
        tile_k=spec.tile_k,
        dtl_stages=spec.dtl_stages,
        chiplet_wgm=spec.chiplet_wgm,
        chiplet_chunk_size=spec.chiplet_chunk_size,
    )
    body = dataclasses.replace(body, name=name)
    knobs = GettKnobs(
        bilinear=spec.bilinear, epi_pair=spec.epi_pair, c_prefetch=spec.c_prefetch
    )
    hooks = RtGettHooks(
        _plan(spec), knobs, body, spec.dtype, geom, spec.wide_d, spec.slots
    )
    object.__setattr__(body, "_gett", hooks)
    return build_universal_gemm(body, arch=arch)


#: digit-slot buckets of the pack's classes (0: every slot)
SLOT_BUCKETS = (2, 4, 0)
#: bilinear classes: C loads issued this many atom rows ahead
BILINEAR_C_PREFETCH = 2


def rgett_classes(dtypes=("f16", "bf16")):
    """The classes hipTensor's kernel pack carries: every lane / D width, each in the digit-slot
    buckets :data:`SLOT_BUCKETS`, and the D-past-2^31-bytes classes of packed operands. The
    bilinear 16-byte-store classes decomposing every slot are left out (their row offsets per
    atom row cost more than the 8-byte stores save; the planner uses the 8-byte class).
    """
    out = []
    for dt in dtypes:
        for bil in (False, True):
            cpf = BILINEAR_C_PREFETCH if bil else 0
            for va in (8, 2):
                for vb in (8, 2):
                    for vd, pair in ((4, True), (4, False), (2, False), (1, False)):
                        for n in SLOT_BUCKETS:
                            if bil and pair and n == 0:
                                continue
                            for wide in (False, True) if va == vb == 8 else (False,):
                                # wide: D past 2^31 bytes (packed operands have 8-element
                                # runs)
                                out.append(
                                    RgettSpec(
                                        dt,
                                        bil,
                                        va,
                                        vb,
                                        vd,
                                        pair,
                                        wide_d=wide,
                                        c_prefetch=cpf,
                                        slots=n,
                                    )
                                )
    return out


__all__ = [
    "ARCHS",
    "RGETT_ABI",
    "RgettSpec",
    "RtGettHooks",
    "SLOTS_KL",
    "SLOTS_KT",
    "SLOTS_ROW",
    "SLOT_BUCKETS",
    "BILINEAR_C_PREFETCH",
    "build_rgett",
    "geometry_names",
    "rgett_classes",
]
