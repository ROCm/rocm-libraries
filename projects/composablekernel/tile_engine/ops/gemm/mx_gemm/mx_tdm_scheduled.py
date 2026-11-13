# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""comp_tdm_scheduled (the rocKE gfx1250 mx_gemm kernel) in Python: spec, validator, names, tuning.

The Python side of MxGemmPipelineAgBgCrCompTDMScheduledPolicy's compile-time rules (in
include/ck_tile/ops/gemm/pipeline/mx_gemm_pipeline_ag_bg_cr_comp_tdm_scheduled_policy.hpp) and
the pipeline's shape rules, both ports of rocKE's MxGemmSpec (instances/gfx1250/mx_gemm.py at
b32989439dc).

Field names and defaults are rocKE's. The tuning knobs (TUNING_DEFAULTS) are what tile_engine's
``tuning_config`` section and the dispatcher's config carry; the rest comes from the tile config
and the data types.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Optional, Tuple

ATOM_K = 128
WAVE = 32
WAVES = 4
BLOCK_K = 32
DEFAULT_SIGNAL = 6
DATA_PAD_BYTES = 16
BANK_CYCLE_BYTES = 256
MXS_BLOCK_BYTES = 256
MXS_PAD_BYTES = 16
REGION_ALIGN_BITS = 2048
SEGMENT_BYTES = 1 << 16
TDM_PAD_PERIOD_MAX = 1024
TDM_PAD_MAX_BYTES = 128 * 4
LDS_PAD_STEP_BYTES = 8
I32_SPAN = 1 << 31
LDS_CAPACITY_BYTES = 327680
VGPRS = 1024
CHUNK_BYTES = 16
LANE_GROUPS = 2

MX_DTYPES = ("fp4", "fp8", "bf8")
OUT_DTYPES = ("fp8e4m3", "fp16", "bf16")

#: rocKE's tuning knobs and defaults; cluster_signal_wmmas None means derived.
TUNING_DEFAULTS: Dict[str, object] = {
    "cluster_dim": (4, 4, 1),
    "gl2_prefetch_distance": 4,
    "cluster_signal_wmmas": None,
    "local_prefetch": 1,
    "half_plr": "b",
    "lds_segment_interleave": True,
}

#: Shape-dependent code paths of a CK instance (the Problem's MxTdmScheduledGemmTraits). rocKE derives
#: them from its compile-time M, N and K; a CK instance fixes them and runs every problem they
#: cover. The defaults run any M and N with no K tail; store_width 0 means the widest the tile
#: allows. Every instance runs any tile count (the pipeline's TailNumber picks the parity).
SHAPE_DEFAULTS: Dict[str, object] = {
    "edge": True,
    "tail_steps": 0,
    "wide": False,
    "store_width": 0,
}

# The short reason codes, in validate()'s order (MxTdmScheduledReason).
REASONS = (
    "ok",
    "shape_positive",
    "k_scale_block",
    "out_dtype",
    "tile_mn",
    "dtype",
    "dtype_width",
    "tile_k",
    "wave_group",
    "mma",
    "tile_atoms",
    "scale_span",
    "k_short",
    "local_prefetch",
    "half_plr",
    "half_plr_b",
    "signal_wmmas",
    "gl2_distance",
    "cluster_dim",
    "cluster_size",
    "cluster_grid",
    "lds_pad",
    "accumulators",
    "lsi_repeat",
    "lsi_baseline_split",
    "lsi_capacity",
    "lds_capacity",
)


def _round_up(v: int, m: int) -> int:
    return (v + m - 1) // m * m


def _ceil_div(v: int, d: int) -> int:
    return (v + d - 1) // d


def _pow2_ceil(v: int) -> int:
    p = 1
    while p < v:
        p <<= 1
    return p


def _is_pow2(v: int) -> bool:
    return v > 0 and (v & (v - 1)) == 0


@dataclass(frozen=True)
class Pad:
    """pad_bytes of LDS padding after every block_bytes written."""

    pad_bytes: int = 0
    block_bytes: int = 0

    @property
    def on(self) -> bool:
        return self.pad_bytes > 0


@dataclass(frozen=True)
class Operand:
    """One operand's place in an LDS buffer (MxTdmScheduledOperand)."""

    rows: int = 0
    base: int = 0
    stride: int = 0
    pad_bytes: int = 0
    block_bytes: int = 0
    iterate_rows: int = 0
    iterate_stride: int = 0

    def as_list(self) -> List[int]:
        return [
            self.rows,
            self.base,
            self.stride,
            self.pad_bytes,
            self.block_bytes,
            self.iterate_rows,
            self.iterate_stride,
        ]


@dataclass(frozen=True)
class LdsLayout:
    reason: str = "ok"
    row_bytes: int = 0
    k_steps: int = 0
    a: Operand = Operand()
    b: Operand = Operand()
    a_scale: Operand = Operand()
    b_scale: Operand = Operand()
    buffer_bytes: int = 0
    buffer_stride: int = 0
    lds_bytes: int = 0


@dataclass(frozen=True)
class _Placement:
    reason: str = "ok"
    a_base: int = 0
    a_stride: int = 0
    b_base: int = 0
    b_stride: int = 0
    scale_base: int = 0
    aligned: bool = False
    span: int = 0


def _place(
    first_is_b, comp_first, stride, second_stride, scale_base, aligned=False, span=0
):
    if first_is_b:
        return _Placement(
            "ok", comp_first, second_stride, 0, stride, scale_base, aligned, span
        )
    return _Placement(
        "ok", 0, stride, comp_first, second_stride, scale_base, aligned, span
    )


def _drop_half_b_order(fields: Dict, what: str) -> None:
    """Removes rocKE's half_b_order: "ahead" is the only order; "late" is not supported."""
    order = str(fields.pop("half_b_order", "ahead")).lower()
    if order != "ahead":
        raise ValueError(
            f"half_b_order {order!r} in the {what} is not supported (only 'ahead')"
        )


@dataclass(frozen=True)
class MxTdmScheduledSpec:
    """One comp_tdm_scheduled kernel: problem, tile and tuning, with rocKE's field names and defaults."""

    M: int = 0
    N: int = 0
    K: int = 0
    out_dtype: str = "fp8e4m3"
    tile_m: int = 256
    tile_n: int = 256
    tile_k: int = 512
    wave_group: Tuple[int, int] = (2, 2)
    cluster_dim: Tuple[int, int, int] = (4, 4, 1)
    gl2_prefetch_distance: int = 4
    cluster_signal_wmmas: Optional[int] = None
    local_prefetch: int = 1
    half_plr: str = "b"
    lds_segment_interleave: bool = True
    mma: str = "32x16"
    a_dtype: str = "fp4"
    b_dtype: str = "fp4"

    @classmethod
    def from_dict(cls, d: Dict) -> "MxTdmScheduledSpec":
        fields = dict(d)
        _drop_half_b_order(fields, "spec")
        for key in ("wave_group", "cluster_dim"):
            if key in fields:
                fields[key] = tuple(fields[key])
        return cls(**fields)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["wave_group"] = list(self.wave_group)
        d["cluster_dim"] = list(self.cluster_dim)
        return d

    # The WMMA atom.
    @property
    def warp_m(self) -> int:
        return self.wave_group[0]

    @property
    def warp_n(self) -> int:
        return self.wave_group[1]

    @property
    def atom_m(self) -> int:
        return 32 if self.mma == "32x16" else 16

    atom_n = 16

    @property
    def c_frag_len(self) -> int:
        return 16 if self.mma == "32x16" else 8

    @property
    def a_rows(self) -> int:
        return 2 if self.mma == "32x16" else 1

    @staticmethod
    def dtype_bits(t: str) -> int:
        return 4 if t == "fp4" else 8

    @property
    def elem_bits(self) -> int:
        return self.dtype_bits(self.a_dtype)

    @property
    def k_per_byte(self) -> int:
        return 8 // self.elem_bits

    @property
    def step_bytes(self) -> int:
        return ATOM_K // self.k_per_byte

    @property
    def chunks_per_lane(self) -> int:
        return self.step_bytes // (LANE_GROUPS * CHUNK_BYTES)

    @property
    def wave_repeat_m(self) -> int:
        return self.tile_m // (self.warp_m * self.atom_m)

    @property
    def wave_repeat_n(self) -> int:
        return self.tile_n // (self.warp_n * self.atom_n)

    @property
    def k_steps(self) -> int:
        return self.tile_k // ATOM_K

    @property
    def tail_steps(self) -> int:
        return _ceil_div(self.K % self.tile_k, ATOM_K)

    @property
    def edge_m(self) -> bool:
        return self.M % self.tile_m != 0

    @property
    def edge_n(self) -> bool:
        return self.N % self.tile_n != 0

    @property
    def packed_k(self) -> int:
        return self.K // self.k_per_byte

    @property
    def wide_a(self) -> bool:
        return _round_up(self.M, self.tile_m) * self.packed_k > I32_SPAN

    @property
    def wide_b(self) -> bool:
        return _round_up(self.N, self.tile_n) * self.packed_k > I32_SPAN

    @property
    def wide_d(self) -> bool:
        return (
            _round_up(self.M, self.tile_m) * _round_up(self.N, self.tile_n) > I32_SPAN
        )

    @property
    def out_bytes(self) -> int:
        return 1 if self.out_dtype == "fp8e4m3" else 2

    @property
    def max_store_width(self) -> int:
        return min(self.wave_repeat_n, 16 // self.out_bytes)

    @property
    def store_width(self) -> int:
        width = self.max_store_width
        while width > 1 and self.N % width:
            width //= 2
        return width

    @property
    def has_cluster(self) -> bool:
        return not (self.cluster_dim[0] == 1 and self.cluster_dim[1] == 1)

    @property
    def half_b(self) -> bool:
        return self.local_prefetch == 1 and self.half_plr == "b"

    @property
    def signal_limit(self) -> int:
        per_step = self.wave_repeat_m * self.wave_repeat_n
        return per_step // 2 if self.half_b else per_step

    @property
    def signal_wmmas(self) -> int:
        if self.cluster_signal_wmmas is not None:
            return self.cluster_signal_wmmas
        return min(DEFAULT_SIGNAL, self.signal_limit)

    @property
    def fp8_convert_width(self) -> int:
        return 8 if self.wave_repeat_n % 8 == 0 else 4

    @property
    def a_scale_reads(self) -> int:
        return max(1, self.wave_repeat_m // 4)

    @property
    def b_scale_reads(self) -> int:
        return max(1, self.wave_repeat_n // 4)

    @property
    def reads_per_step(self) -> int:
        return (
            self.a_scale_reads
            + self.b_scale_reads
            + self.chunks_per_lane
            * (self.a_rows * self.wave_repeat_m + self.wave_repeat_n)
        )

    @property
    def late_pending_reads(self) -> int:
        return self.reads_per_step if self.local_prefetch else 0

    @property
    def grid(self) -> Tuple[int, int]:
        """Workgroups along x (N tiles) and y (M tiles)."""
        return _ceil_div(self.N, self.tile_n), _ceil_div(self.M, self.tile_m)

    # LDS layout.
    def derived_pads(self) -> Dict[str, Pad]:
        k_bytes = self.tile_k // self.k_per_byte

        def data_block(vector_width: int) -> int:
            block = _round_up(k_bytes, BANK_CYCLE_BYTES)
            if k_bytes * vector_width > BANK_CYCLE_BYTES:
                block = _round_up(k_bytes * vector_width, BANK_CYCLE_BYTES)
            return block

        def scale_pad(vector_width: int) -> Pad:
            per_block = (ATOM_K // 32) * vector_width
            if vector_width < 4 or per_block % 16 or (per_block // 16) & 1:
                return Pad()
            return Pad(MXS_PAD_BYTES, MXS_BLOCK_BYTES)

        rm, rn = self.wave_repeat_m, self.wave_repeat_n
        return {
            "a": Pad(DATA_PAD_BYTES, data_block(rm)),
            "b": Pad(DATA_PAD_BYTES, data_block(rn)),
            "a_scale": scale_pad(rm),
            "b_scale": scale_pad(rn),
        }

    def _interleave_placement(
        self, comp_a, comp_b, data_a, data_b, scales
    ) -> _Placement:
        seg = SEGMENT_BYTES
        lanes = min(self.atom_m, self.atom_n)

        def coarse(is_b: bool) -> bool:
            tile = self.tile_n if is_b else self.tile_m
            waves = self.warp_n if is_b else self.warp_m
            repeat = self.wave_repeat_n if is_b else self.wave_repeat_m
            return lanes * repeat >= tile // waves

        if self.warp_m == 1 or self.warp_n == 1:
            active_b = self.warp_n > 1
            comp_act = comp_b if active_b else comp_a
            comp_share = comp_a if active_b else comp_b
            data_act = data_b if active_b else data_a
            if not coarse(active_b):
                return _Placement("lsi_repeat")
            if active_b and data_b <= seg and comp_b >= seg:
                return _Placement("lsi_baseline_split")
            stride = comp_act + 2 * comp_share
            if stride // seg > (data_act - 1) // seg:
                return _place(active_b, comp_act, stride, comp_share, stride + comp_act)
            pre = _round_up(stride, seg)
            span = pre + comp_act
            return _place(
                active_b, comp_act, pre, comp_share, span, True, span + scales
            )
        if not coarse(True):
            return _Placement("lsi_repeat")
        a_cols = self.atom_m * self.warp_m * self.wave_repeat_m
        a_readable = coarse(False) or (self.tile_m // 2) % a_cols == 0
        if not a_readable:
            stride = comp_b + 2 * comp_a
            if stride >= seg:
                return _place(True, comp_b, stride, comp_a, stride + comp_b)
            pre = _round_up(stride, seg)
            span = pre + comp_b
            return _place(True, comp_b, pre, comp_a, span, True, span + scales)
        pair = comp_a + comp_b
        if pair < seg:
            span = seg + pair
            return _place(True, comp_b, seg, seg, span, True, span + scales)
        return _place(True, comp_b, pair, pair, 2 * pair)

    def lds_layout(self) -> LdsLayout:
        """The LDS layout of a spec whose tile, types and wave group are valid."""
        pads = self.derived_pads()
        row_bytes = self.tile_k // self.k_per_byte
        steps = self.k_steps
        a_rows, b_rows = self.tile_m // 2, self.tile_n // 2

        def data_component(rows: int, pad: Pad) -> Tuple[int, int, int]:
            if pad.on and pad.block_bytes > TDM_PAD_PERIOD_MAX:
                iterate_rows = pad.block_bytes // row_bytes
                stride = pad.block_bytes + pad.pad_bytes
                return _ceil_div(rows, iterate_rows) * stride, iterate_rows, stride
            size = rows * row_bytes
            if pad.on:
                size += size // pad.block_bytes * pad.pad_bytes
            return size, 0, 0

        def scale_region(rows: int, pad: Pad) -> int:
            size = 2 * rows * steps * 4
            return (
                size // pad.block_bytes * (pad.block_bytes + pad.pad_bytes)
                if pad.on
                else size
            )

        def scale_component(rows: int, pad: Pad) -> int:
            size = steps * rows * 4
            return size + (size // pad.block_bytes * pad.pad_bytes if pad.on else 0)

        a_bytes, a_it_rows, a_it_stride = data_component(a_rows, pads["a"])
        b_bytes, b_it_rows, b_it_stride = data_component(b_rows, pads["b"])
        align = REGION_ALIGN_BITS // self.elem_bits
        b_scale_region = _round_up(scale_region(b_rows, pads["b_scale"]), align)
        a_scale_region = _round_up(scale_region(a_rows, pads["a_scale"]), align)
        b_end = _round_up(2 * b_bytes, align)
        a_base = b_end + b_scale_region + a_scale_region
        placement = _Placement("ok", a_base, a_bytes, 0, b_bytes, b_end)
        baseline_span = a_base + 2 * a_bytes
        natural_stride = a_base + _round_up(2 * a_bytes, align)
        if self.lds_segment_interleave:
            placement = self._interleave_placement(
                a_bytes,
                b_bytes,
                a_rows * row_bytes,
                b_rows * row_bytes,
                b_scale_region + a_scale_region,
            )
            if placement.reason != "ok":
                return LdsLayout(reason=placement.reason)
        if placement.aligned:
            buffer_stride = _pow2_ceil(max(natural_stride, placement.span))
            if 2 * buffer_stride > LDS_CAPACITY_BYTES:
                return LdsLayout(reason="lsi_capacity")
            lds_bytes = max(
                buffer_stride + baseline_span, buffer_stride + placement.span
            )
        else:
            buffer_stride = natural_stride
            if natural_stride + _pow2_ceil(natural_stride) <= LDS_CAPACITY_BYTES:
                buffer_stride = _pow2_ceil(natural_stride)
            lds_bytes = buffer_stride + baseline_span
        b_scale_base = placement.scale_base
        a_scale_base = b_scale_base + b_scale_region
        a_scale_comp = scale_component(a_rows, pads["a_scale"])
        b_scale_comp = scale_component(b_rows, pads["b_scale"])
        pa, pb, pas, pbs = pads["a"], pads["b"], pads["a_scale"], pads["b_scale"]
        return LdsLayout(
            reason="ok",
            row_bytes=row_bytes,
            k_steps=steps,
            a=Operand(
                a_rows,
                placement.a_base,
                placement.a_stride,
                pa.pad_bytes,
                pa.block_bytes,
                a_it_rows,
                a_it_stride,
            ),
            b=Operand(
                b_rows,
                placement.b_base,
                placement.b_stride,
                pb.pad_bytes,
                pb.block_bytes,
                b_it_rows,
                b_it_stride,
            ),
            a_scale=Operand(
                a_rows, a_scale_base, a_scale_comp, pas.pad_bytes, pas.block_bytes
            ),
            b_scale=Operand(
                b_rows, b_scale_base, b_scale_comp, pbs.pad_bytes, pbs.block_bytes
            ),
            buffer_bytes=max(
                placement.a_base + placement.a_stride + a_bytes,
                placement.b_base + placement.b_stride + b_bytes,
                a_scale_base + 2 * a_scale_comp,
                b_scale_base + 2 * b_scale_comp,
            ),
            buffer_stride=buffer_stride,
            lds_bytes=lds_bytes,
        )

    # Validation.
    @staticmethod
    def _pad_reason(pad: Pad, row_bytes: int, rows: int, chunk: int) -> str:
        if pad.pad_bytes < 0 or pad.block_bytes < 0:
            return "lds_pad"
        if not pad.on:
            return "ok"
        if pad.pad_bytes % LDS_PAD_STEP_BYTES or pad.pad_bytes > TDM_PAD_MAX_BYTES:
            return "lds_pad"
        block = pad.block_bytes
        if block <= TDM_PAD_PERIOD_MAX:
            if block < 8 or not _is_pow2(block):
                return "lds_pad"
        elif row_bytes == 0:
            return "lds_pad"
        elif block % row_bytes or rows % (block // row_bytes):
            return "lds_pad"
        if block % chunk:
            return "lds_pad"
        return "ok"

    def validate(self, check_shape: bool = True) -> str:
        """rocKE's validate(): "ok" if the spec builds, else the code of the first rule broken.

        check_shape: check the rules that need M, N and K (off for a tile-only instance)."""
        if check_shape:
            if self.M <= 0 or self.N <= 0 or self.K <= 0:
                return "shape_positive"
            if self.K % BLOCK_K:
                return "k_scale_block"
        if self.out_dtype not in OUT_DTYPES:
            return "out_dtype"
        valid_mn = (16, 32, 64, 128, 256)
        if self.tile_m not in valid_mn or self.tile_n not in valid_mn:
            return "tile_mn"
        if self.a_dtype not in MX_DTYPES or self.b_dtype not in MX_DTYPES:
            return "dtype"
        if self.dtype_bits(self.b_dtype) != self.elem_bits:
            return "dtype_width"
        fp4 = self.a_dtype == "fp4"
        if not (self.tile_k in (128, 256) or (fp4 and self.tile_k in (512, 1024))):
            return "tile_k"
        if tuple(self.wave_group) not in ((2, 2), (4, 1), (1, 4)):
            return "wave_group"
        if not (self.mma == "16x16" or (fp4 and self.mma == "32x16")):
            return "mma"
        if self.tile_m % (self.warp_m * self.atom_m) or self.tile_n % (
            self.warp_n * self.atom_n
        ):
            return "tile_atoms"
        rm, rn = self.wave_repeat_m, self.wave_repeat_n
        if check_shape:
            mn_span = max(
                _round_up(self.M, self.tile_m), _round_up(self.N, self.tile_n)
            )
            if mn_span * _ceil_div(self.K, ATOM_K) * 4 > I32_SPAN:
                return "scale_span"
            if self.K < self.tile_k:
                return "k_short"
        if self.local_prefetch not in (0, 1):
            return "local_prefetch"
        if self.half_plr not in ("b", "none"):
            return "half_plr"
        if self.half_plr == "b" and (self.local_prefetch != 1 or rn % 2):
            return "half_plr_b"
        if self.signal_wmmas < 0 or self.signal_wmmas > self.signal_limit:
            return "signal_wmmas"
        if self.gl2_prefetch_distance != 0 and not 2 <= self.gl2_prefetch_distance <= 8:
            return "gl2_distance"
        cx, cy, cz = self.cluster_dim
        if cx < 1 or cy < 1 or cz < 1:
            return "cluster_dim"
        if cz != 1 or cx * cy > 16:
            return "cluster_size"
        gx, gy = self.grid
        if check_shape and (gx % cx or gy % cy):
            return "cluster_grid"
        pads = self.derived_pads()
        row_bytes = self.tile_k // self.k_per_byte
        checks = (
            (pads["a"], self.tile_m // 2, 16, True),
            (pads["b"], self.tile_n // 2, 16, True),
            (pads["a_scale"], self.tile_m // 2, min(16, 4 * rm), False),
            (pads["b_scale"], self.tile_n // 2, min(16, 4 * rn), False),
        )
        for pad, rows, chunk, data in checks:
            reason = self._pad_reason(pad, row_bytes if data else 0, rows, chunk)
            if reason != "ok":
                return reason
            if (
                data
                and pad.on
                and pad.block_bytes <= TDM_PAD_PERIOD_MAX
                and (pad.block_bytes % row_bytes)
            ):
                return "lds_pad"
        if rm * rn * self.c_frag_len > VGPRS:
            return "accumulators"
        layout = self.lds_layout()
        if layout.reason != "ok":
            return layout.reason
        if layout.lds_bytes > LDS_CAPACITY_BYTES:
            return "lds_capacity"
        return "ok"

    # Names.
    def tuning_name(self, with_shape_parts: bool = True) -> str:
        """rocKE's name after the tile: cluster, then each non-default value (the policy's GetTuningName)."""
        cx, cy, _ = self.cluster_dim
        name = f"_c{cx}x{cy}"
        if with_shape_parts and self.out_dtype != "fp8e4m3":
            name += f"_{self.out_dtype}"
        if with_shape_parts and tuple(self.wave_group) != (2, 2):
            name += f"_wg{self.warp_m}x{self.warp_n}"
        if self.gl2_prefetch_distance != 4:
            name += f"_pf{self.gl2_prefetch_distance}"
        if self.cluster_signal_wmmas is not None:
            name += f"_cs{self.cluster_signal_wmmas}"
        if self.local_prefetch == 0:
            name += "_lp0"
        if self.half_plr == "none":
            name += "_hplrnone"
        if not self.lds_segment_interleave:
            name += "_lsi0"
        if with_shape_parts and self.mma == "16x16":
            name += "_mma16"
        return name

    def kernel_name(self, prefix: str = "mx_gemm") -> str:
        """rocKE's MxGemmSpec.kernel_name()."""
        tag = (
            "mx" + self.a_dtype + (self.b_dtype if self.b_dtype != self.a_dtype else "")
        )
        return (
            f"{prefix}_{tag}_M{self.M}N{self.N}K{self.K}"
            f"_t{self.tile_m}x{self.tile_n}x{self.tile_k}{self.tuning_name()}"
        )

    def tuning(self) -> Dict[str, object]:
        return {key: getattr(self, key) for key in TUNING_DEFAULTS}


_NAME = re.compile(
    r"^(?P<prefix>.+?)_mx(?P<a>fp4|fp8|bf8)(?P<b>fp4|fp8|bf8)?"
    r"(?:_M(?P<M>\d+)N(?P<N>\d+)K(?P<K>\d+))?"
    r"_t(?P<tm>\d+)x(?P<tn>\d+)x(?P<tk>\d+)_c(?P<cx>\d+)x(?P<cy>\d+)(?P<rest>(?:_\w+?)*)$"
)
_FLAG = re.compile(
    r"_(?:(?P<out>fp16|bf16)|wg(?P<wm>\d)x(?P<wn>\d)|pf(?P<pf>\d+)|cs(?P<cs>\d+)"
    r"|(?P<lp0>lp0)|(?P<hplrnone>hplrnone)|(?P<hblate>hblate)|(?P<lsi0>lsi0)|(?P<mma16>mma16))"
)


def parse_kernel_name(name: str) -> Tuple[str, MxTdmScheduledSpec]:
    """(prefix, spec) of a rocKE kernel name; M, N and K are 0 when the name has none."""
    m = _NAME.match(name)
    if not m:
        raise ValueError(f"not a rocKE mx_gemm kernel name: {name!r}")
    fields: Dict[str, object] = {
        "a_dtype": m["a"],
        "b_dtype": m["b"] or m["a"],
        "M": int(m["M"] or 0),
        "N": int(m["N"] or 0),
        "K": int(m["K"] or 0),
        "tile_m": int(m["tm"]),
        "tile_n": int(m["tn"]),
        "tile_k": int(m["tk"]),
        "cluster_dim": (int(m["cx"]), int(m["cy"]), 1),
    }
    rest = m["rest"]
    pos = 0
    while pos < len(rest):
        f = _FLAG.match(rest, pos)
        if not f:
            raise ValueError(f"unknown part {rest[pos:]!r} in {name!r}")
        pos = f.end()
        if f["out"]:
            fields["out_dtype"] = f["out"]
        elif f["wm"]:
            fields["wave_group"] = (int(f["wm"]), int(f["wn"]))
        elif f["pf"]:
            fields["gl2_prefetch_distance"] = int(f["pf"])
        elif f["cs"]:
            fields["cluster_signal_wmmas"] = int(f["cs"])
        elif f["lp0"]:
            fields["local_prefetch"] = 0
        elif f["hplrnone"]:
            fields["half_plr"] = "none"
        elif f["hblate"]:
            raise ValueError(f"half-B late (hblate) is not supported: {name!r}")
        elif f["lsi0"]:
            fields["lds_segment_interleave"] = False
        elif f["mma16"]:
            fields["mma"] = "16x16"
    return m["prefix"], MxTdmScheduledSpec(**fields)


# tile_engine / dispatcher configs.


def normalize_tuning(tuning: Optional[Dict]) -> Dict[str, object]:
    """A complete tuning dict: rocKE's defaults for missing keys; unknown keys are an error."""
    tuning = dict(tuning or {})
    unknown = set(tuning) - set(TUNING_DEFAULTS) - set(SHAPE_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown comp_tdm_scheduled tuning keys: {sorted(unknown)}")
    out = dict(TUNING_DEFAULTS)
    out.update(SHAPE_DEFAULTS)
    out.update(tuning)
    out["cluster_dim"] = tuple(out["cluster_dim"])
    if len(out["cluster_dim"]) == 2:
        out["cluster_dim"] = (*out["cluster_dim"], 1)
    if out["cluster_signal_wmmas"] is not None and out["cluster_signal_wmmas"] < 0:
        out["cluster_signal_wmmas"] = None
    out["half_plr"] = str(out["half_plr"]).lower()
    out["lds_segment_interleave"] = bool(out["lds_segment_interleave"])
    out["edge"] = bool(out["edge"])
    out["wide"] = bool(out["wide"])
    return out


def is_default_tuning(tuning: Optional[Dict]) -> bool:
    t = normalize_tuning(tuning)
    return all(t[k] == normalize_tuning({})[k] for k in t)


def schedule_tunings() -> List[Dict[str, object]]:
    """Tunings that reach every tile-dependent outcome of validate() (the defaults first): the
    knobs a tile's validity turns on are half PLR, local prefetch and segment interleave."""
    return [
        normalize_tuning(
            {"half_plr": plr, "local_prefetch": lp, "lds_segment_interleave": lsi}
        )
        for plr, lp, lsi in itertools.product(("b", "none"), (1, 0), (True, False))
    ]


def encode_tuning(tuning: Optional[Dict]) -> str:
    """A tuning as a ``key=value,...`` token (no characters CMake or a shell treat specially)."""
    t = normalize_tuning(tuning)
    parts = []
    for key in sorted(t):
        value = t[key]
        if isinstance(value, tuple):
            value = "x".join(str(v) for v in value)
        elif value is None:
            value = "none"
        elif isinstance(value, bool):
            value = "1" if value else "0"
        parts.append(f"{key}={value}")
    return ",".join(parts)


def decode_tuning(token: str) -> Dict[str, object]:
    """The inverse of encode_tuning."""
    raw = dict(part.split("=", 1) for part in token.split(",") if part)
    out: Dict[str, object] = {}
    for key, value in raw.items():
        default = {**TUNING_DEFAULTS, **SHAPE_DEFAULTS}.get(key)
        if key == "cluster_dim":
            out[key] = tuple(int(v) for v in value.split("x"))
        elif key == "cluster_signal_wmmas":
            out[key] = None if value == "none" else int(value)
        elif isinstance(default, bool):
            out[key] = value == "1"
        elif isinstance(default, int):
            out[key] = int(value)
        else:
            out[key] = value
    return normalize_tuning(out)


def expand_tuning_config(tuning_config: Optional[Dict]) -> List[Dict[str, object]]:
    """The product of a ``tuning_config`` section ({"key": {"values": [...]}}), normalized."""
    tuning_config = tuning_config or {}
    keys = sorted(tuning_config)
    values = []
    for key in keys:
        entry = tuning_config[key]
        vals = entry.get("values") if isinstance(entry, dict) else entry
        if not isinstance(vals, list) or not vals:
            raise ValueError(
                f"tuning_config.{key} must be {{'values': [...]}} with a value"
            )
        values.append(vals)
    return [
        normalize_tuning(dict(zip(keys, combo))) for combo in itertools.product(*values)
    ]


#: tile_engine / dispatcher datatype names to rocKE's.
TE_DTYPES = {"fp4": "fp4", "fp8": "fp8", "bf8": "bf8"}
TE_OUT_DTYPES = {"fp8": "fp8e4m3", "fp16": "fp16", "bf16": "bf16"}
CPP_TYPES = {
    "fp4": "ck_tile::pk_fp4_t",
    "fp8": "ck_tile::fp8_t",
    "bf8": "ck_tile::bf8_t",
    "fp8e4m3": "ck_tile::fp8_t",
    "fp16": "ck_tile::half_t",
    "bf16": "ck_tile::bf16_t",
}


def spec_from_tile(
    tile_config: Dict,
    tuning: Optional[Dict],
    a_dtype: str = "fp4",
    b_dtype: Optional[str] = None,
    out_dtype: str = "fp8e4m3",
    M: int = 0,
    N: int = 0,
    K: int = 0,
) -> MxTdmScheduledSpec:
    """The spec of a tile_engine tile config with a tuning (M, N, K 0: the instance alone)."""
    t = normalize_tuning(tuning)
    if tile_config.get("warp_k", 1) != 1:
        raise ValueError("comp_tdm_scheduled needs warp_k 1")
    if (
        tile_config.get("warp_tile_n", 16) != 16
        or tile_config.get("warp_tile_k", 128) != 128
    ):
        raise ValueError(
            "comp_tdm_scheduled needs a warp tile of 32x16x128 or 16x16x128"
        )
    mma = {32: "32x16", 16: "16x16"}.get(tile_config.get("warp_tile_m"))
    if mma is None:
        raise ValueError("comp_tdm_scheduled needs warp_tile_m 32 or 16")
    return MxTdmScheduledSpec(
        M=M,
        N=N,
        K=K,
        out_dtype=out_dtype,
        tile_m=tile_config["tile_m"],
        tile_n=tile_config["tile_n"],
        tile_k=tile_config["tile_k"],
        wave_group=(tile_config["warp_m"], tile_config["warp_n"]),
        cluster_dim=t["cluster_dim"],
        gl2_prefetch_distance=t["gl2_prefetch_distance"],
        cluster_signal_wmmas=t["cluster_signal_wmmas"],
        local_prefetch=t["local_prefetch"],
        half_plr=t["half_plr"],
        lds_segment_interleave=t["lds_segment_interleave"],
        mma=mma,
        a_dtype=a_dtype,
        b_dtype=b_dtype or a_dtype,
    )


def tile_from_spec(spec: MxTdmScheduledSpec) -> Dict[str, int]:
    """The tile_engine tile config of a spec."""
    return {
        "tile_m": spec.tile_m,
        "tile_n": spec.tile_n,
        "tile_k": spec.tile_k,
        "warp_m": spec.warp_m,
        "warp_n": spec.warp_n,
        "warp_k": 1,
        "warp_tile_m": spec.atom_m,
        "warp_tile_n": 16,
        "warp_tile_k": 128,
    }


def instance_store_width(spec: MxTdmScheduledSpec, tuning: Optional[Dict]) -> int:
    t = normalize_tuning(tuning)
    width = int(t["store_width"]) or spec.max_store_width
    return width


def instance_name_suffix(spec: MxTdmScheduledSpec, tuning: Optional[Dict]) -> str:
    """What a comp_tdm_scheduled instance adds to the tile_engine kernel name: rocKE's tuning suffix without the
    parts the name already carries, then the shape traits that are not the defaults."""
    t = normalize_tuning(tuning)
    suffix = spec.tuning_name(with_shape_parts=False)
    if not t["edge"]:
        suffix += "_noedge"
    if t["tail_steps"]:
        suffix += f"_tail{t['tail_steps']}"
    if t["wide"]:
        suffix += "_wide"
    if t["store_width"]:
        suffix += f"_w{t['store_width']}"
    return suffix


def validate_instance(spec: MxTdmScheduledSpec, tuning: Optional[Dict]) -> str:
    """ "ok" if the tile and tuning make a comp_tdm_scheduled instance (the compile-time rules plus the traits)."""
    reason = spec.validate(check_shape=False)
    if reason != "ok":
        return reason
    t = normalize_tuning(tuning)
    width = instance_store_width(spec, t)
    if (
        width < 1
        or width > spec.max_store_width
        or spec.wave_repeat_n % width
        or not _is_pow2(width)
    ):
        return "store_width"
    if not 0 <= t["tail_steps"] < spec.k_steps:
        return "tail_steps"
    return "ok"


def instance_unsupported_reason(
    spec: MxTdmScheduledSpec, tuning: Optional[Dict]
) -> str:
    """MxGemmKernel::GetUnsupportedReason of the comp_tdm_scheduled instance on spec's M, N and K: "" if it
    runs them, else the rocKE reason code or the shape trait it lacks."""
    reason = spec.validate()
    if reason != "ok":
        return reason
    t = normalize_tuning(tuning)
    if (spec.edge_m or spec.edge_n) and not t["edge"]:
        return "edge"
    if (spec.wide_a or spec.wide_b or spec.wide_d) and not t["wide"]:
        return "wide"
    if spec.tail_steps != t["tail_steps"]:
        return "tail_steps"
    if spec.N % instance_store_width(spec, t):
        return "store_width"
    return ""


def cpp_cluster_sequence(tuning: Optional[Dict]) -> str:
    """The ClusterTileGemmShape cluster of a tuning: (M tiles, N tiles, 1) = (y, x, 1)."""
    cx, cy, _ = normalize_tuning(tuning)["cluster_dim"]
    return f"ck_tile::sequence<{cy}, {cx}, 1>"


def cpp_policy_type(tuning: Optional[Dict]) -> str:
    """The MxGemmPipelineAgBgCrCompTDMScheduledPolicy<...> spelling of a tuning's other knobs."""
    t = normalize_tuning(tuning)
    signal = -1 if t["cluster_signal_wmmas"] is None else t["cluster_signal_wmmas"]
    plr = "B" if t["half_plr"] == "b" else "None"
    lsi = "true" if t["lds_segment_interleave"] else "false"
    return (
        f"ck_tile::MxGemmPipelineAgBgCrCompTDMScheduledPolicy<{t['gl2_prefetch_distance']}, "
        f"{signal}, {t['local_prefetch']}, ck_tile::MxTdmScheduledHalfPlr::{plr}, {lsi}>"
    )


def cpp_kernel_traits(spec: MxTdmScheduledSpec, tuning: Optional[Dict]) -> str:
    """The MxTdmScheduledGemmTraits<...> spelling of a tuning's shape traits."""
    t = normalize_tuning(tuning)
    edge = "true" if t["edge"] else "false"
    wide = "true" if t["wide"] else "false"
    return (
        f"ck_tile::MxTdmScheduledGemmTraits<{edge}, {edge}, {wide}, {wide}, {wide}, "
        f"{t['tail_steps']}, {instance_store_width(spec, t)}>"
    )


def te_tuning_from_spec(
    spec: MxTdmScheduledSpec, shape_traits: bool = False
) -> Dict[str, object]:
    """The tuning dict of a spec; with shape_traits, the traits its M, N and K need exactly."""
    t: Dict[str, object] = {k: getattr(spec, k) for k in TUNING_DEFAULTS}
    t["cluster_dim"] = list(spec.cluster_dim)
    if shape_traits and spec.M and spec.N and spec.K:
        t.update(
            edge=spec.edge_m or spec.edge_n,
            tail_steps=spec.tail_steps,
            wide=spec.wide_a or spec.wide_b or spec.wide_d,
            store_width=0
            if spec.store_width == spec.max_store_width
            else spec.store_width,
        )
    return t


def tuning_config_section(
    tunings: Iterable[Dict[str, object]],
) -> Dict[str, Dict[str, list]]:
    """A tuning_config section holding exactly these values per key (for one tuning: an exact
    instance; for several, their product)."""
    section: Dict[str, Dict[str, list]] = {}
    for t in tunings:
        for key, value in t.items():
            value = list(value) if isinstance(value, tuple) else value
            vals = section.setdefault(key, {"values": []})["values"]
            if value not in vals:
                vals.append(value)
    return section


__all__ = [
    "MxTdmScheduledSpec",
    "LdsLayout",
    "Operand",
    "Pad",
    "REASONS",
    "SHAPE_DEFAULTS",
    "TUNING_DEFAULTS",
    "cpp_cluster_sequence",
    "cpp_kernel_traits",
    "cpp_policy_type",
    "decode_tuning",
    "encode_tuning",
    "expand_tuning_config",
    "instance_name_suffix",
    "instance_store_width",
    "instance_unsupported_reason",
    "is_default_tuning",
    "normalize_tuning",
    "parse_kernel_name",
    "schedule_tunings",
    "spec_from_tile",
    "te_tuning_from_spec",
    "tile_from_spec",
    "tuning_config_section",
    "validate_instance",
]
