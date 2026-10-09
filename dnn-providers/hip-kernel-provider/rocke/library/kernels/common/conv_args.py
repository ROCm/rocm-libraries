# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Host-side argument computation for AOT convolution launches.

An AOT conv kernel takes its problem shape as kernel arguments, so the host
has to compute every extent, stride and magic-division constant the kernel
would otherwise have had folded in as a compile-time constant.

Each class here produces the ``values`` dict that
:class:`~rocke.runtime.launcher.KernelLauncher` consumes, keyed by the names
in :mod:`kernels.common.conv_abi` — the same list the kernel builders emit
their params from and the launch signature is derived from. ``to_launch_values``
asserts that the dict covers exactly that list, so a missing or stray key is a
loud error here rather than silent argument corruption on the GPU.

Lives in ``kernels/`` (the lowest library layer) so that every layer above —
``dispatch/``, ``builders/``, ``benchmarks/``, ``tests/`` — can import it
without introducing a layering cycle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from kernels.common.conv_abi import ArgSpec, conv_arg_names, conv_direct_arg_names
from rocke.helpers.transforms import calculate_magic_numbers

__all__ = [
    "ConvArgs",
    "ConvGeometry",
]


def _magic_as_i32(mult: int) -> int:
    """Reinterpret a uint32 magic multiplier as the signed i32 the packer wants.

    ``pack_args`` packs i32 kernargs with ``struct`` format ``'i'``, which
    rejects anything at or above 2**31. The kernel reads the value back with
    an unsigned mul-hi, so the two's-complement bit pattern is what it needs.
    """
    return mult - (1 << 32) if mult >= (1 << 31) else mult


def _magic_pair(name_prefix: str, divisor: int) -> Dict[str, int]:
    """The ``(mult, shift)`` kernarg pair for one unmerge divisor."""
    mult, shift = calculate_magic_numbers(max(int(divisor), 1))
    return {
        f"{name_prefix}_mult": _magic_as_i32(mult),
        f"{name_prefix}_shift": shift,
    }


def _ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


@dataclass(frozen=True)
class ConvGeometry:
    """Extents shared by every direction, derived once from a ``ConvProblem``.

    Kept separate from the per-direction argument classes because all three
    need the same output extents and per-group channel counts, and deriving
    them twice is how the 2-D and 3-D paths drift apart.
    """

    N: int
    Di: int
    Hi: int
    Wi: int
    C: int
    K: int
    Z: int
    Y: int
    X: int
    sD: int
    sH: int
    sW: int
    pD: int
    pH: int
    pW: int
    dD: int
    dH: int
    dW: int
    groups: int
    is_3d: bool

    @staticmethod
    def _out(extent: int, pad: int, dil: int, filt: int, stride: int) -> int:
        return (extent + 2 * pad - dil * (filt - 1) - 1) // stride + 1

    @property
    def Do(self) -> int:
        if not self.is_3d:
            return 1
        return self._out(self.Di, self.pD, self.dD, self.Z, self.sD)

    @property
    def Ho(self) -> int:
        return self._out(self.Hi, self.pH, self.dH, self.Y, self.sH)

    @property
    def Wo(self) -> int:
        return self._out(self.Wi, self.pW, self.dW, self.X, self.sW)

    @property
    def cpg(self) -> int:
        return self.C // self.groups

    @property
    def kpg(self) -> int:
        return self.K // self.groups

    @property
    def Zc(self) -> int:
        """Filter depth as used in products (1 when 2-D)."""
        return self.Z if self.is_3d else 1

    @classmethod
    def from_problem(cls, problem: object) -> "ConvGeometry":
        """Build from a ``ConvProblem`` without importing library code."""
        is_3d = bool(getattr(problem, "is_3d", False))
        return cls(
            N=problem.N,
            Di=getattr(problem, "Di", 1) or 1,
            Hi=problem.Hi,
            Wi=problem.Wi,
            C=problem.C,
            K=problem.K,
            Z=getattr(problem, "Z", 1) or 1,
            Y=problem.Y,
            X=problem.X,
            sD=getattr(problem, "sD", 1) or 1,
            sH=problem.sH,
            sW=problem.sW,
            pD=getattr(problem, "pD", 0) or 0,
            pH=problem.pH,
            pW=problem.pW,
            dD=getattr(problem, "dD", 1) or 1,
            dH=problem.dH,
            dW=problem.dW,
            groups=problem.groups,
            is_3d=is_3d,
        )

    @classmethod
    def from_direct_problem(cls, problem: object) -> "ConvGeometry":
        """Build from a ``DirectConvProblem``.

        Direct conv names the same quantities differently (``H``/``W`` for the
        input extents, ``KH``/``KW`` for the filter, one ``PAD`` and one
        ``stride`` for both axes, ``dil_h``/``dil_w`` for the dilation, which
        only the non-grouped forward kernel computes). Mapping it onto the
        shared geometry here means the output-extent formula and the NHWC /
        NHWK stride layouts exist once for both algorithms rather than once
        per algorithm -- they were drifting apart by construction before.
        """
        return cls(
            N=problem.N,
            Di=1,
            Hi=problem.H,
            Wi=problem.W,
            C=problem.groups * problem.cpg,
            K=problem.groups * problem.kpg,
            Z=1,
            Y=problem.KH,
            X=problem.KW,
            sD=1,
            sH=problem.stride,
            sW=problem.stride,
            pD=0,
            pH=problem.PAD,
            pW=problem.PAD,
            dD=1,
            dH=problem.dil_h,
            dW=problem.dil_w,
            groups=problem.groups,
            is_3d=False,
        )

    # ---- tensor strides (row-major, in elements) ----

    def nhwc_strides(self) -> Dict[str, int]:
        """Activation-layout strides: N[D]HWC."""
        out = {
            "n": (
                self.Di * self.Hi * self.Wi * self.C
                if self.is_3d
                else self.Hi * self.Wi * self.C
            ),
            "hi": self.Wi * self.C,
            "wi": self.C,
        }
        if self.is_3d:
            out["di"] = self.Hi * self.Wi * self.C
        return out

    def nhwk_strides(self) -> Dict[str, int]:
        """Output-layout strides: N[D]HWK."""
        out = {
            "n": (
                self.Do * self.Ho * self.Wo * self.K
                if self.is_3d
                else self.Ho * self.Wo * self.K
            ),
            "ho": self.Wo * self.K,
            "wo": self.K,
        }
        if self.is_3d:
            out["do"] = self.Ho * self.Wo * self.K
        return out

    def filter_strides(self) -> Dict[str, int]:
        """Weight-layout strides: K[Z]YXC with the per-group channel extent."""
        out = {
            "k": self.Zc * self.Y * self.X * self.cpg,
            "y": self.X * self.cpg,
            "x": self.cpg,
        }
        if self.is_3d:
            out["z"] = self.Y * self.X * self.cpg
        return out

    # ---- magic-number blocks ----

    def spatial_magic(self, prefix: str) -> Dict[str, int]:
        """Divisors for the ``(n, [do,] ho, wo)`` unmerge."""
        out: Dict[str, int] = {}
        if self.is_3d:
            out.update(_magic_pair(f"p_magic_{prefix}Do", self.Do))
        out.update(_magic_pair(f"p_magic_{prefix}Ho", self.Ho))
        out.update(_magic_pair(f"p_magic_{prefix}Wo", self.Wo))
        return out

    def channel_magic(self, prefix: str, cdim: int) -> Dict[str, int]:
        """Divisors for the ``([z,] y, x, c)`` unmerge."""
        out: Dict[str, int] = {}
        if self.is_3d:
            out.update(_magic_pair(f"p_magic_{prefix}Y", self.Y))
        out.update(_magic_pair(f"p_magic_{prefix}X", self.X))
        out.update(_magic_pair(f"p_magic_{prefix}cpg", cdim))
        return out

    def dims_block(self) -> Dict[str, int]:
        """The ``p_N .. p_kpg`` extent block common to every direction."""
        out = {
            "p_N": self.N,
            "p_Hi": self.Hi,
            "p_Wi": self.Wi,
            "p_C": self.C,
            "p_K": self.K,
            "p_Y": self.Y,
            "p_X": self.X,
            "p_sH": self.sH,
            "p_sW": self.sW,
            "p_pH": self.pH,
            "p_pW": self.pW,
            "p_dH": self.dH,
            "p_dW": self.dW,
            "p_groups": self.groups,
            "p_Ho": self.Ho,
            "p_Wo": self.Wo,
            "p_cpg": self.cpg,
            "p_kpg": self.kpg,
        }
        if self.is_3d:
            out.update(
                {
                    "p_Z": self.Z,
                    "p_Di": self.Di,
                    "p_sD": self.sD,
                    "p_pD": self.pD,
                    "p_dD": self.dD,
                    "p_Do": self.Do,
                }
            )
        return out


def _finalize(
    values: Dict[str, int], arg_names: Sequence[ArgSpec], pointers: Dict[str, int]
) -> Dict[str, int]:
    """Merge pointers into ``values`` and check the dict against the ABI list.

    Kernargs are packed positionally, so a key the signature does not name is
    dead weight and a name the signature *does* have but the dict lacks would
    surface far away as ``KeyError: missing kernel arg``. Checking both
    directions here points straight at the mismatch instead.
    """
    values = dict(values)
    values.update(pointers)
    expected = {name for name, _ in arg_names}
    missing = expected - set(values)
    extra = set(values) - expected
    if missing or extra:
        raise ValueError(
            "AOT launch values do not match the kernarg ABI: "
            f"missing={sorted(missing)} unexpected={sorted(extra)}"
        )
    return values


# ---------------------------------------------------------------------------
# Launch args
# ---------------------------------------------------------------------------

# The kernarg names are self-describing, which is what lets one routine fill
# every ABI: a stride is ``p_<tensor>_stride_<axis>`` and the tensor fixes the
# layout; a magic pair is ``p_magic_<m|k|n>_<divisor>_<mult|shift>`` and only
# the divisor affects the value; every other scalar name means one thing in
# every ABI that uses it.
_STRIDE_RE = re.compile(r"p_(A|B|D|X|W|dX|dY|dW)_stride_(\w+)")
_MAGIC_RE = re.compile(r"p_magic_[mkn]_(\w+)_(mult|shift)")

# Which layout each operand tensor has. A tensor keeps its name across
# directions -- ``dY`` is NHWK whether wgrad or dgrad reads it -- so the
# stride names never need the direction to be decoded.
_LAYOUT_OF_TENSOR = {
    "A": "nhwc",
    "X": "nhwc",
    "dX": "nhwc",
    "D": "nhwk",
    "dY": "nhwk",
    "B": "filter",
    "W": "filter",
    "dW": "filter",
}

# Host state the kernel reads but that cannot be derived from the problem:
# the caller has to hand it over, and forgetting to is an error, not a zero.
_CALLER_SUPPLIED = ("ws_ptr", "ws_bytes", "sub_gemm_buf", "num_sub_gemms")


# Which directions each algorithm has a kernel for.
_DIRECTIONS = {
    "implicit_gemm": ("fwd", "wgrad", "dgrad"),
    "direct": ("fwd", "dgrad", "wgrad"),
}

# The GEMM axes of each implicit-GEMM direction, as the names of the scalars
# that hold them: (M, N, K). The three directions contract different tensors,
# so each has its own view of which extent is which axis -- and that is the
# only thing about the tile counts and the launch grid that differs.
_GEMM_AXES = {
    "fwd": ("p_M", "p_kpg", "p_K_gemm"),
    "wgrad": ("p_wg_M", "p_wg_N", "p_wg_K"),
    "dgrad": ("p_dg_M", "p_dg_N", "p_dg_K"),
}

# The forward implicit-GEMM loaders decode the filter-channel reduction index
# k -> (y, x, c) with 24-bit multiplies (``mul_u24``), and every operand of
# those products is bounded by the reduction extent. The K loop runs up to one
# tile past that extent, so the bound keeps a bit of headroom below 2**24.
MUL24_REDUCTION_LIMIT = 1 << 23


@dataclass
class ConvArgs:
    """Launch values for one conv kernel -- any algorithm, any direction.

    One type covers implicit GEMM in all three directions and direct conv in
    both of its. What distinguishes them is data, not type:

    * ``algorithm`` picks the ABI family. It is inferred from the problem:
      a ``DirectConvProblem`` (which bakes the filter geometry into the
      kernel) gives ``"direct"``, a ``ConvProblem`` gives ``"implicit_gemm"``.
    * ``direction`` picks the ABI within the family and, for implicit GEMM,
      which extents are the GEMM axes.
    * ``tile_m`` / ``tile_n`` (and ``tile_k`` for wgrad) are the tile the
      implicit-GEMM binary was built with. Direct conv has no tile and must
      leave them at 0 -- passing one would be silently meaningless.

    :meth:`to_launch_values` walks the kernel's ABI and computes each
    argument from its name, which works for every variant because the names
    are self-describing (see ``_STRIDE_RE`` / ``_MAGIC_RE``).
    """

    geom: ConvGeometry
    direction: str = "fwd"
    algorithm: str = "implicit_gemm"
    # The tile the *kernel* was built with. It is baked into the ISA, so an
    # args object is only meaningful paired with one binary -- carrying it
    # here rather than re-passing it to every call keeps the two from being
    # mismatched by accident.
    tile_m: int = 0
    tile_n: int = 0
    # wgrad only: the granularity the split-K slice width is rounded to.
    tile_k: int = 0

    def __post_init__(self) -> None:
        allowed = _DIRECTIONS.get(self.algorithm)
        if allowed is None:
            raise ValueError(f"unknown conv algorithm {self.algorithm!r}")
        if self.direction not in allowed:
            raise ValueError(
                f"{self.algorithm} conv has no {self.direction!r} kernel; "
                f"expected one of {allowed}"
            )
        tiles = (self.tile_m, self.tile_n, self.tile_k)
        if self.algorithm == "direct":
            if any(tiles):
                raise ValueError(
                    "direct conv has no GEMM tile; tile_m/tile_n/tile_k must be 0"
                )
            return
        if self.tile_m <= 0 or self.tile_n <= 0:
            raise ValueError(
                "implicit-GEMM launch args need the kernel's tile_m and tile_n: "
                "the tile counts the kernel decodes its workgroup id against "
                "are derived from them"
            )
        if self.direction == "wgrad" and self.tile_k <= 0:
            raise ValueError(
                "wgrad launch args need the kernel's tile_k: the split-K slice "
                "width is rounded to it"
            )

    @classmethod
    def from_problem(
        cls,
        problem: object,
        *,
        direction: str = "fwd",
        tile_m: int = 0,
        tile_n: int = 0,
        tile_k: int = 0,
    ) -> "ConvArgs":
        """Build from a ``ConvProblem`` or a ``DirectConvProblem``.

        The problem type decides the algorithm: direct conv's problem names
        its filter ``KH``/``KW`` because those are baked into the kernel, and
        that is what tells the two apart.
        """
        if hasattr(problem, "KH"):
            geom, algorithm = ConvGeometry.from_direct_problem(problem), "direct"
        else:
            geom, algorithm = ConvGeometry.from_problem(problem), "implicit_gemm"
        return cls(geom, direction, algorithm, tile_m, tile_n, tile_k)

    # ---- the ABI ----

    def arg_names(self, *, two_stage: bool = False) -> List[ArgSpec]:
        """The ordered kernarg ABI of the kernel these args launch."""
        if self.algorithm == "direct":
            if two_stage:
                raise ValueError("direct conv has no two-stage variant")
            return conv_direct_arg_names(direction=self.direction)
        return conv_arg_names(
            direction=self.direction, is_3d=self.geom.is_3d, two_stage=two_stage
        )

    # ---- GEMM view (implicit GEMM only) ----

    def _require_gemm(self) -> None:
        if self.algorithm != "implicit_gemm":
            raise ValueError(
                "direct conv is not a GEMM: it has no GEMM axes, tile counts "
                "or GEMM launch grid"
            )

    @property
    def gemm_m(self) -> int:
        """Extent along the GEMM's M axis for this direction."""
        self._require_gemm()
        return self._named_values()[_GEMM_AXES[self.direction][0]]

    @property
    def gemm_n(self) -> int:
        """Extent along the GEMM's N axis for this direction."""
        self._require_gemm()
        return self._named_values()[_GEMM_AXES[self.direction][1]]

    @property
    def gemm_k(self) -> int:
        """Extent along the GEMM's reduction axis for this direction."""
        self._require_gemm()
        return self._named_values()[_GEMM_AXES[self.direction][2]]

    @property
    def num_pid_m(self) -> int:
        """M-axis tile count; the kernel reads this as ``p_num_pid_m``."""
        return _ceil_div(self.gemm_m, self.tile_m)

    @property
    def num_pid_n(self) -> int:
        """N-axis tile count; the kernel reads this as ``p_num_pid_n``."""
        return _ceil_div(self.gemm_n, self.tile_n)

    def grid(self, split_k: int = 1) -> Tuple[int, int, int]:
        """``(gx, gy, gz)``: N tiles by M tiles by groups (times the split).

        ``split_k`` is a launch parameter, so it stays an argument. The dgrad
        tilde path overrides this with the sub-GEMM tile total; see the
        benchmark's grid helper.
        """
        return (self.num_pid_n, self.num_pid_m, self.geom.groups * split_k)

    # ---- values ----

    def _named_values(self) -> Dict[str, int]:
        """Every scalar whose name alone fixes its value, for this geometry."""
        g = self.geom
        spatial_out = g.N * g.Do * g.Ho * g.Wo if g.is_3d else g.N * g.Ho * g.Wo
        spatial_in = g.N * g.Di * g.Hi * g.Wi if g.is_3d else g.N * g.Hi * g.Wi
        filter_cpg = g.Zc * g.Y * g.X * g.cpg
        filter_kpg = g.Zc * g.Y * g.X * g.kpg
        values = dict(g.dims_block())
        values.update(
            {
                # direct conv's names for the total channel counts
                "p_total_c": g.C,
                "p_total_k": g.K,
                # forward GEMM: output positions x output channels over
                # filter-channels
                "p_M": spatial_out,
                "p_K_gemm": filter_cpg,
                # wgrad GEMM: output channels x filter-channels over output
                # positions
                "p_wg_M": g.kpg,
                "p_wg_N": filter_cpg,
                "p_wg_K": spatial_out,
                # dgrad GEMM: input positions x input channels over
                # filter-output-channels
                "p_dg_M": spatial_in,
                "p_dg_N": g.cpg,
                "p_dg_K": filter_kpg,
            }
        )
        return values

    def _variant_values(self, split_k: int) -> Dict[str, int]:
        """Scalars whose meaning depends on the variant, not just the name."""
        if self.algorithm == "direct" or self.direction != "wgrad":
            if split_k != 1:
                raise ValueError(
                    f"split_k={split_k}: only implicit-GEMM wgrad splits the "
                    f"reduction ({self.algorithm} {self.direction} has no "
                    f"split-K arguments)"
                )
        if self.algorithm == "direct":
            return {}
        values = {"p_num_pid_m": self.num_pid_m, "p_num_pid_n": self.num_pid_n}
        if self.direction == "wgrad":
            # Slice width, rounded up to a whole number of K tiles so every
            # slice is the same width and the loop bound stays tile-aligned.
            # The tail past wg_K reads zero through the descriptor's bounds,
            # which is what lets all ``split_k`` slices share one uniform
            # bound computation.
            values["ks"] = _ceil_div(self.gemm_k, self.tile_k * split_k) * self.tile_k
            values["ks_count"] = split_k
        return values

    def _divisor(self, name: str) -> int:
        g = self.geom
        return {
            "Do": g.Do,
            "Ho": g.Ho,
            "Wo": g.Wo,
            "Di": g.Di,
            "Hi": g.Hi,
            "Wi": g.Wi,
            "Y": g.Y,
            "X": g.X,
            "cpg": g.cpg,
        }[name]

    def _strides(self, layout: str) -> Dict[str, int]:
        g = self.geom
        return {
            "nhwc": g.nhwc_strides,
            "nhwk": g.nhwk_strides,
            "filter": g.filter_strides,
        }[layout]()

    def _resolve(self, name: str, named: Dict[str, int]) -> int:
        if name in named:
            return named[name]
        m = _STRIDE_RE.fullmatch(name)
        if m:
            return self._strides(_LAYOUT_OF_TENSOR[m.group(1)])[m.group(2)]
        m = _MAGIC_RE.fullmatch(name)
        if m:
            mult, shift = calculate_magic_numbers(
                max(int(self._divisor(m.group(1))), 1)
            )
            return _magic_as_i32(mult) if m.group(2) == "mult" else shift
        if name in _CALLER_SUPPLIED:
            raise ValueError(
                f"{name} must be passed to to_launch_values: it is host state "
                f"this kernel reads, not something the problem determines"
            )
        raise ValueError(f"no host-side rule for kernarg {name!r}")

    def to_launch_values(
        self,
        a_ptr: int,
        b_ptr: int,
        d_ptr: int,
        a_bytes: int,
        b_bytes: int,
        d_bytes: int,
        *,
        split_k: int = 1,
        ws_ptr: Optional[int] = None,
        ws_bytes: Optional[int] = None,
        sub_gemm_buf: Optional[int] = None,
        num_sub_gemms: Optional[int] = None,
    ) -> Dict[str, int]:
        """Return the ``values`` dict for ``KernelLauncher.__call__()``.

        The six positional arguments are the three operand pointers and their
        byte sizes, bound in ABI order to whatever the direction calls them
        (``A/B/D``, ``dY/X/dW``, ``dY/W/dX``). The keywords are the only
        other per-launch inputs:

        * ``split_k`` -- wgrad's reduction split degree. A launch parameter,
          never baked in, so one binary serves any degree.
        * ``ws_ptr`` / ``ws_bytes`` -- wgrad's two-stage workspace. Passing
          them selects the two-stage ABI.
        * ``sub_gemm_buf`` / ``num_sub_gemms`` -- dgrad's tilde record table,
          which the kernel always reads.

        Anything the ABI needs but was not passed, and anything passed that
        the ABI does not have, is an error here rather than a misread on the
        GPU.
        """
        if split_k < 1:
            raise ValueError(f"split_k must be >= 1 (got {split_k})")
        if (
            self.algorithm == "implicit_gemm"
            and self.direction == "fwd"
            and self.gemm_k >= MUL24_REDUCTION_LIMIT
        ):
            raise ValueError(
                f"implicit-GEMM fwd needs a reduction extent (Z*Y*X*cpg) below "
                f"2**23 for its 24-bit address products; got {self.gemm_k}"
            )
        abi = self.arg_names(two_stage=ws_ptr is not None)
        names = [n for n, _ in abi]
        values: Dict[str, int] = dict(
            zip(names[:6], (a_ptr, b_ptr, d_ptr, a_bytes, b_bytes, d_bytes))
        )
        supplied = {
            "ws_ptr": ws_ptr,
            "ws_bytes": (ws_bytes or 0) if ws_ptr is not None else ws_bytes,
            "sub_gemm_buf": sub_gemm_buf,
            "num_sub_gemms": num_sub_gemms,
        }
        supplied = {k: v for k, v in supplied.items() if v is not None}
        named = self._named_values()
        named.update(self._variant_values(split_k))
        for name in names[6:]:
            values[name] = (
                supplied.pop(name) if name in supplied else self._resolve(name, named)
            )
        if supplied:
            raise ValueError(
                f"{sorted(supplied)} are not arguments of the "
                f"{self.algorithm} {self.direction} kernel"
            )
        return _finalize(values, abi, {})
