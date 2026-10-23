# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Dense SDPA backward (dQ, dK, dV): shared spec, validation and kernarg ABI.

The dense backward serves contiguous BSHD / BHSD problems with fixed lengths
whose shape is baked at build time: sequence lengths, head counts, head size,
layout, mask and every element stride are compile-time constants; the batch is
a runtime kernarg (grid z) unless the spec pins it. It is a routing choice
inside cells the layout-general family already declares (it never adds or
removes a cell), so it has no predicate of its own: the general predicate
decides, :func:`kernels.common.attention_bwd_plan.dense_bwd_eligibility`
decides eligibility, and the tuned dense pack decides the route.

This module owns:

* :class:`AttentionDenseBwdSpec`, the architecture-neutral base. Problem shape
  fields are required; every other field is defaulted.
  :class:`ArchAttentionDenseBwdSpec` adds the target's body knob
  ``body_variant`` and is parameterised by the target: the per-arch specs
  (``kernels.gfx942.attention_dense_bwd.Gfx942AttentionDenseBwdSpec`` and the
  gfx950 one) are that class bound to their target, next to their builders.
* :func:`validate_attn_dense_bwd_spec`: arch checks plus the body spec, an
  :class:`~kernels.common.attention_bwd.AttnBwdSpec` resolved and validated by
  the layout-general family's own validator. The dense family runs the shared
  backward body (tile step, mask policy, stats view, dQ sink, dK / dV
  epilogues); ``body_knobs`` carries any of the general family's tuning knobs
  to it, so every body lever reaches the dense kernels unchanged.
* :func:`attn_dense_bwd_params`: the ordered kernargs of the
  ``rocke.attn_dense_bwd.v1`` ABI. Prep and convert are the shared aux kernels
  of the general family (their own ABI).

Dense-only knobs (compile-time fields, defaults first):

* ``epilogue``: ``"lds_staged"``, ``"lds_staged_pair"``, ``"direct"`` (the
  general family's dK / dV epilogues; every value stores the same values);
* ``oob``: ``"buffer_oob"`` (buffer resources per (batch, head) slab whose
  ``num_records`` ends at the last row of the sequence: rows past the
  sequence read as zero) or ``"clamp"`` (rows clamped to the last row, as the
  general family does); both give the same dK / dV;
* ``edge_tiles``: mask-free interior q tiles (compile-time band
  classification, see :mod:`kernels.common._attention_dense_bwd_shell`);
* ``ws_layout``: the dQ (and kv) workspace layout, shared with the aux
  kernels;
* ``dkv_mode``: ``"direct"`` (one kv head per CTA, dK / dV stored once) or
  ``"atomic"`` (the runtime ``g_split`` splits the query-head group; dK / dV
  go through the fp32 workspace and two converts);
* ``persistent`` / ``persist_decode``: a grid of ``num_persistent`` CTAs (a
  kernarg) striding over the work items, in kv-tile-major or kv-head-major
  order;
* ``xcd_remap``: the runtime chiplet remap of the CTA index (kernargs
  ``xcd_n``, ``xcd_chunk``; ``xcd_n = 0`` is the identity);
* ``dq_mode``: ``"atomic"`` (G4 in the main kernel, fp32 dQ atomics, a dQ
  convert) or ``"split"`` (the main kernel computes dK / dV only; dQ comes
  from the dense Q-parallel dQ kernel of
  :mod:`kernels.common.attention_dense_bwd_dq`, on the targets of
  ``SPLIT_DQ_ARCHS``);
* ``body_variant`` (per-arch subclasses): ``"general"`` (the shared body), or
  the per-arch bodies registered but not built yet.

Kernel names carry the base key (dtype, head size, layout, shape, heads, mask,
ragged) and a salt over every non-default compile-time field.

Shape pinning (explicit, by design). The dense family is compile-time shaped:
``AttentionDenseBwdSpec.SHAPE_BINDING`` is ``"compile_time"`` and
:meth:`AttentionDenseBwdSpec.shape_key` is the pinned problem shape (the
fields of ``SHAPE_FIELDS``: sequence lengths, head counts, head size, layout,
and the batch when the spec pins it). Every one of them is baked into the code
object: the strides, the per-slab buffer ranges, the kv-tile count and the
compile-time band classification (which removes the edge or interior loops a
shape proves empty) all fold the shape into constants, so two shapes are two
different kernels, one instance per shape key. That is the family's purpose:
the dense pack serves only the exact shape keys it was tuned for, and every
other request takes the layout-general family (runtime shapes). A request is
checked against the instance's pinned shape on the host
(``attention_bwd_run.check_attn_dense_bwd_tensors``).

The shape therefore belongs to the instance identity, and the kernel name
spells it in readable tokens (``sq``, ``sk``, ``hq``, ``hk``, ``d``, the
layout, ``b`` for a pinned batch) rather than in a digest: the name is the
code object's symbol, so it must differ between shapes; the tokens keep
profiles, traces and resource-gate reports attributable to a shape (the
gfx942 dense forward family names its kernels the same way); and the IR
golden table, the on-disk code-object cache and the recorded resource-gate
rows are keyed by these names, so a digest would change the bytes of every
dense kernel and re-key every artifact without changing any kernel. A
runtime-shape dense body (shape out of the name, lengths as kernargs) would be
a new body, not a renaming, and is not built.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, fields
from types import MappingProxyType
from typing import ClassVar, Optional

from rocke.helpers.spec import non_default_fields, salted_kernel_name

from kernels.common.attention_bwd import (
    BWD_DTYPES,
    WS_LAYOUTS,
    AttnBwdSpec,
    _not_built,
    validate_attn_bwd_spec,
)

__all__ = [
    "ATTN_DENSE_BWD_ABI",
    "DENSE_BODY_KNOBS",
    "DENSE_BWD_ARCHS",
    "DENSE_BWD_BODY_VARIANTS",
    "DENSE_BWD_HEAD_SIZES",
    "DENSE_DKV_MODES",
    "DENSE_EPILOGUES",
    "DENSE_LAYOUTS",
    "DENSE_OOB_MODES",
    "DENSE_PERSIST_DECODES",
    "ArchAttentionDenseBwdSpec",
    "AttentionDenseBwdSpec",
    "attn_dense_bwd_params",
    "check_dense_bwd_spec_class",
    "dense_bwd_band",
    "dense_bwd_not_built",
    "dense_bwd_not_built_reasons",
    "dense_layout_strides",
    "validate_attn_dense_bwd_spec",
]

ATTN_DENSE_BWD_ABI = "rocke.attn_dense_bwd.v1"
DENSE_BWD_ARCHS = ("gfx942", "gfx950")
DENSE_BWD_HEAD_SIZES = (64, 128)
DENSE_LAYOUTS = ("bshd", "bhsd")
DENSE_EPILOGUES = ("lds_staged", "lds_staged_pair", "direct")
DENSE_OOB_MODES = ("buffer_oob", "clamp")
DENSE_DKV_MODES = ("direct", "atomic")
DENSE_PERSIST_DECODES = ("kv_tile_major", "hkv_major")
DENSE_DQ_MODES = ("atomic", "split")

# Fields of the general family's spec the dense spec owns itself (problem
# fields, or dense knobs that map onto a body field or replace it: the dense
# grid remap is ``AttentionDenseBwdSpec.xcd_remap``); every other tuning field
# of AttnBwdSpec may ride in ``body_knobs``. ``static_scale`` and
# ``lb_interleave`` are body knobs the dense plan derives (as the general
# plan does) from the scale placement and the load-balance order.
_BODY_OWNED = frozenset(
    {
        "head_size",
        "dtype",
        "seq_mode",
        "dkv_mode",
        "stage_vec",
        "mask_class",
        "dq_mode",
        "ws_layout",
        "dkv_epilogue",
        "edge_tiles",
        "head_pack",
        "xcd_remap",
        "name",
    }
)
DENSE_BODY_KNOBS = tuple(
    f.name for f in fields(AttnBwdSpec) if f.name not in _BODY_OWNED
)
# Declared defaults of the body knobs: a ``body_knobs`` entry equal to its
# default is dropped, so one kernel has one spec (and one name).
_BODY_DEFAULTS = {
    f.name: f.default
    for f in fields(AttnBwdSpec)
    if f.name in DENSE_BODY_KNOBS and f.default is not dataclasses.MISSING
}

# Largest in-slab byte offset a buffer voffset (i32) may reach: rows up to one
# tile past the sequence end, at the tensor's token stride.
_VOFFSET_LIMIT = 2**31
_TILE_SLACK_ROWS = 256


def _check_in(name, value, allowed):
    if value not in allowed:
        raise ValueError(f"{name}={value!r} not in {tuple(allowed)}")


def _is_body_default(name, value) -> bool:
    if name not in _BODY_DEFAULTS:
        return False
    d = _BODY_DEFAULTS[name]
    return type(value) is type(d) and value == d


def _canonical_knobs(knobs) -> tuple:
    """Sorted ``((name, value), ...)`` with lists turned into tuples."""
    items = knobs.items() if isinstance(knobs, dict) else knobs
    out = {}
    for item in items:
        if not (isinstance(item, (tuple, list)) and len(item) == 2):
            raise ValueError(f"body_knobs entry {item!r} is not a (name, value) pair")
        k, v = item
        if k in out:
            raise ValueError(f"body_knobs sets {k!r} twice")
        out[k] = tuple(v) if isinstance(v, list) else v
    return tuple(sorted(out.items()))


def dense_layout_strides(layout: str, heads: int, seq: int, d: int) -> tuple:
    """``(batch, head, token)`` element strides of a contiguous tensor."""
    if layout == "bshd":
        return (seq * heads * d, d, heads * d)
    if layout == "bhsd":
        return (heads * seq * d, seq * d, d)
    raise ValueError(f"layout={layout!r} not in {DENSE_LAYOUTS}")


@dataclass(frozen=True)
class AttentionDenseBwdSpec:
    """Compile-time spec of a dense backward main kernel (shared by both arches).

    Problem fields (``seqlen_q``, ``seqlen_kv``, ``num_query_heads``,
    ``num_kv_heads``, ``head_size``) are required. ``batch = None`` makes the
    batch a runtime kernarg; an integer pins it (debug and A/B only).

    Mask: ``causal`` (keys ``k <= q + off``) with ``causal_bottom_right``
    choosing ``off = seqlen_kv - seqlen_q`` instead of 0, and
    ``sliding_window = W > 0`` keeping the ``W`` keys ``q + off - W + 1 ..
    q + off``. ``window_left`` / ``window_right`` (both or one set, ``-1`` =
    unbounded on that side) give any two-sided band of the general family
    instead (``causal`` must then be False and ``sliding_window`` 0;
    ``causal_bottom_right`` still picks the diagonal).

    ``ragged = False`` needs ``seqlen_q % block_m == 0`` and ``seqlen_kv %
    block_n == 0`` (checked against the resolved tile); ``ragged = True`` is
    its own kernel for any lengths (tail tiles masked, rows past the sequence
    read as zero, partial rows dropped by guarded stores).

    The problem shape is pinned at compile time (``SHAPE_BINDING =
    "compile_time"``; :meth:`shape_key`, see the module docstring): one
    instance per shape key, the shape spelled in the kernel name.
    """

    # The dense family bakes the problem shape into the code object (see the
    # module docstring); "runtime" (lengths as kernargs) is not built.
    SHAPE_BINDING: ClassVar[str] = "compile_time"
    # Fields of the pinned shape, in shape_key order (the batch only when the
    # spec pins it; it is a runtime kernarg otherwise).
    SHAPE_FIELDS: ClassVar[tuple] = (
        "seqlen_q",
        "seqlen_kv",
        "num_query_heads",
        "num_kv_heads",
        "head_size",
        "layout",
        "batch",
    )
    # The target of a per-arch subclass (None: the architecture-neutral base).
    ARCH: ClassVar[Optional[str]] = None

    seqlen_q: int
    seqlen_kv: int
    num_query_heads: int
    num_kv_heads: int
    head_size: int
    batch: Optional[int] = None
    causal: bool = True
    causal_bottom_right: bool = False
    dtype: str = "bf16"
    layout: str = "bshd"
    sliding_window: int = 0
    window_left: Optional[int] = None
    window_right: Optional[int] = None
    ragged: bool = False
    dkv_mode: str = "direct"
    dq_mode: str = "atomic"
    epilogue: str = "lds_staged"
    oob: str = "buffer_oob"
    edge_tiles: bool = True
    ws_layout: str = "head_major"
    persistent: bool = False
    persist_decode: str = "kv_tile_major"
    xcd_remap: bool = False
    body_knobs: tuple = ()
    name: str = "rocke_attn_dense_bwd"

    # Values of the per-arch ``body_variant`` field; the base has none.
    BODY_VARIANTS = ("general",)

    def __post_init__(self) -> None:
        for fname in ("seqlen_q", "seqlen_kv", "num_query_heads", "num_kv_heads"):
            v = getattr(self, fname)
            if isinstance(v, bool) or not isinstance(v, int) or v < 1:
                raise ValueError(f"{fname}={v!r} must be a positive int")
        _check_in("head_size", self.head_size, DENSE_BWD_HEAD_SIZES)
        if self.batch is not None and (
            isinstance(self.batch, bool)
            or not isinstance(self.batch, int)
            or self.batch < 1
        ):
            raise ValueError(f"batch={self.batch!r} must be None or a positive int")
        if self.num_query_heads % self.num_kv_heads:
            raise ValueError(
                f"num_query_heads={self.num_query_heads} is not a multiple of "
                f"num_kv_heads={self.num_kv_heads}"
            )
        _check_in("dtype", self.dtype, BWD_DTYPES)
        _check_in("layout", self.layout, DENSE_LAYOUTS)
        _check_in("dkv_mode", self.dkv_mode, DENSE_DKV_MODES)
        _check_in("dq_mode", self.dq_mode, DENSE_DQ_MODES)
        _check_in("epilogue", self.epilogue, DENSE_EPILOGUES)
        _check_in("oob", self.oob, DENSE_OOB_MODES)
        _check_in("ws_layout", self.ws_layout, WS_LAYOUTS)
        _check_in("persist_decode", self.persist_decode, DENSE_PERSIST_DECODES)
        for fname in (
            "causal",
            "causal_bottom_right",
            "ragged",
            "edge_tiles",
            "persistent",
            "xcd_remap",
        ):
            if not isinstance(getattr(self, fname), bool):
                raise ValueError(f"{fname} must be a bool")
        sw = self.sliding_window
        if isinstance(sw, bool) or not isinstance(sw, int) or sw < 0:
            raise ValueError(f"sliding_window={sw!r} must be an int >= 0")
        if sw and not self.causal:
            raise ValueError("sliding_window > 0 requires causal=True")
        explicit = self.window_left is not None or self.window_right is not None
        if explicit:
            if self.causal or sw:
                raise ValueError(
                    "window_left / window_right give the band themselves: "
                    "causal must be False and sliding_window 0"
                )
            for fname in ("window_left", "window_right"):
                v = getattr(self, fname)
                if v is not None and (
                    isinstance(v, bool) or not isinstance(v, int) or v < -1
                ):
                    raise ValueError(f"{fname}={v!r} must be None or an int >= -1")
        if self.persistent and self.xcd_remap:
            raise ValueError("xcd_remap applies to the non-persistent grid only")
        if self.persist_decode != "kv_tile_major" and not self.persistent:
            raise ValueError("persist_decode applies to persistent=True only")
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(f"name={self.name!r} must be a non-empty identifier")
        object.__setattr__(self, "body_knobs", _canonical_knobs(self.body_knobs))
        unknown = [k for k, _ in self.body_knobs if k not in DENSE_BODY_KNOBS]
        if unknown:
            raise ValueError(
                f"body_knobs {unknown} are not tuning knobs of the shared body "
                f"(dense-owned or problem fields are spec fields)"
            )
        object.__setattr__(
            self,
            "body_knobs",
            tuple((k, v) for k, v in self.body_knobs if not _is_body_default(k, v)),
        )
        _check_in("body_variant", self.variant(), type(self).BODY_VARIANTS)
        # The body spec checks every knob value (arch-independent rules now;
        # the arch rules in validate_attn_dense_bwd_spec).
        self.body_spec()

    # -- derived -----------------------------------------------------------

    @property
    def group(self) -> int:
        return self.num_query_heads // self.num_kv_heads

    def shape_key(self) -> tuple:
        """The pinned problem shape: the ``SHAPE_FIELDS`` values in order."""
        return tuple(getattr(self, f) for f in self.SHAPE_FIELDS)

    def variant(self) -> str:
        """The body variant: the per-arch subclass's ``body_variant`` field;
        the architecture-neutral base has the shared body only."""
        if "body_variant" in self.__dataclass_fields__:
            return self.body_variant
        return "general"

    def band(self) -> tuple[int, int, bool]:
        """``(left, right, bottom_right)`` of the mask (``-1`` = unbounded)."""
        return dense_bwd_band(self)

    @property
    def masked(self) -> bool:
        left, right, _ = self.band()
        return not (left == -1 and right == -1)

    def body_spec(self) -> AttnBwdSpec:
        """The general family's spec of the shared body (not resolved)."""
        return AttnBwdSpec(
            head_size=self.head_size,
            dtype=self.dtype,
            seq_mode="batched",
            dkv_mode=self.dkv_mode,
            stage_vec=8,
            mask_class="band" if self.masked else "none",
            dq_mode=self.dq_mode,
            ws_layout=self.ws_layout,
            dkv_epilogue=self.epilogue,
            edge_tiles=self.edge_tiles,
            head_pack=False,
            **dict(self.body_knobs),
        )

    def strides(self, tensor: str) -> tuple[int, int, int]:
        """``(batch, head, token)`` element strides of ``tensor``."""
        q_side = tensor in ("q", "o", "do", "dq")
        heads = self.num_query_heads if q_side else self.num_kv_heads
        seq = self.seqlen_q if q_side else self.seqlen_kv
        return dense_layout_strides(self.layout, heads, seq, self.head_size)

    def non_default_fields(self) -> tuple[tuple[str, object], ...]:
        """Compile-time fields that differ from their declared default."""
        return non_default_fields(self, skip=("name",))

    def _mask_token(self) -> str:
        left, right, br = self.band()
        diag = "br" if br else "tl"
        if left == -1 and right == -1:
            return "none"
        if left == -1 and right == 0:
            return f"causal{diag}"
        return f"win{left}x{right}{diag}"

    def kernel_name(self) -> str:
        """Base key plus a salt over every non-default compile-time knob."""
        parts = [
            self.name,
            self.dtype,
            f"d{self.head_size}",
            self.layout,
            f"sq{self.seqlen_q}",
            f"sk{self.seqlen_kv}",
            f"hq{self.num_query_heads}",
            f"hk{self.num_kv_heads}",
            self._mask_token(),
        ]
        if self.ragged:
            parts.append("ragged")
        if self.batch is not None:
            parts.append(f"b{self.batch}")
        keyed = (
            "batch",
            "causal",
            "causal_bottom_right",
            "dtype",
            "layout",
            "sliding_window",
            "window_left",
            "window_right",
            "ragged",
        )
        return salted_kernel_name(
            "_".join(parts),
            [(k, v) for k, v in self.non_default_fields() if k not in keyed],
        )


# ``body_variant`` values of each target: the shared body (``"general"``,
# built) and the target's own body, registered but not built (``"xt"`` on
# gfx942: K, K^T and V register resident with XT LDS transposes; ``"trload"``
# on gfx950: x4 LDS DMA with ``ds_read_tr16_b64`` operand reads).
DENSE_BWD_BODY_VARIANTS = MappingProxyType(
    {
        "gfx942": ("general", "xt"),
        "gfx950": ("general", "trload"),
    }
)


@dataclass(frozen=True)
class ArchAttentionDenseBwdSpec(AttentionDenseBwdSpec):
    """The dense backward spec of one target: the shared spec plus the
    target's body knob ``body_variant`` (see :data:`DENSE_BWD_BODY_VARIANTS`).

    A per-arch spec names its target in the class statement,
    ``class Gfx942AttentionDenseBwdSpec(ArchAttentionDenseBwdSpec,
    arch="gfx942")``, which sets ``ARCH`` and ``BODY_VARIANTS``; the per-arch
    module keeps the class (descriptors name it there) and its builder.
    """

    body_variant: str = "general"

    def __init_subclass__(cls, *, arch: Optional[str] = None, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if arch is not None:
            _check_in("arch", arch, DENSE_BWD_ARCHS)
            cls.ARCH = arch
            cls.BODY_VARIANTS = DENSE_BWD_BODY_VARIANTS[arch]


def check_dense_bwd_spec_class(spec: AttentionDenseBwdSpec, spec_cls: type) -> None:
    """``TypeError`` unless ``spec`` is a ``spec_cls`` (a per-arch builder's
    own spec class)."""
    if not isinstance(spec, spec_cls):
        raise TypeError(
            f"build_attention_dense_bwd on {spec_cls.ARCH} takes a "
            f"{spec_cls.__name__}, got {type(spec).__name__}"
        )


def dense_bwd_band(spec: AttentionDenseBwdSpec) -> tuple[int, int, bool]:
    """``(left, right, bottom_right)`` of a dense spec's mask."""
    br = bool(spec.causal_bottom_right)
    if spec.window_left is not None or spec.window_right is not None:
        left = -1 if spec.window_left is None else int(spec.window_left)
        right = -1 if spec.window_right is None else int(spec.window_right)
        return left, right, br
    if not spec.causal:
        return -1, -1, br
    if spec.sliding_window:
        return spec.sliding_window - 1, 0, br
    return -1, 0, br


def dense_bwd_not_built(spec: AttentionDenseBwdSpec) -> list[str]:
    """Dense knob values whose emission is not built (registered levers)."""
    out = []
    variant = spec.variant()
    if variant != "general":
        out.append(f"body_variant={variant!r}")
    return out


def validate_attn_dense_bwd_spec(spec: AttentionDenseBwdSpec, arch: str) -> AttnBwdSpec:
    """Arch checks of a dense spec; returns the resolved, validated body spec.

    Raises ``ValueError`` for an illegal spec on ``arch`` (wrong arch, a tile
    that does not divide the lengths of a non-ragged kernel, a slab past the
    32-bit buffer offset) and every error of the general validator.
    """
    _check_in("arch", arch, DENSE_BWD_ARCHS)
    owner = type(spec).ARCH
    if owner is not None and owner != arch:
        raise ValueError(f"{type(spec).__name__} is a {owner} spec, not {arch}")
    # the general main kernel's recorded resource failures (RESOURCE_REFUSED)
    # are of another code object: the dense kernel's own gate rows cover it
    body = validate_attn_bwd_spec(spec.body_spec(), arch, main_kernel_refusals=False)
    if not spec.ragged:
        if spec.seqlen_q % body.block_m or spec.seqlen_kv % body.block_n:
            raise ValueError(
                f"ragged=False needs seqlen_q % {body.block_m} == 0 and seqlen_kv % "
                f"{body.block_n} == 0 (got {spec.seqlen_q}, {spec.seqlen_kv}); use "
                "ragged=True"
            )
    for tensor in ("q", "k"):
        _sb, _sh, st = spec.strides(tensor)
        seq = spec.seqlen_q if tensor == "q" else spec.seqlen_kv
        if (seq + _TILE_SLACK_ROWS) * st * 2 >= _VOFFSET_LIMIT:
            raise ValueError(
                f"{tensor} slab of {seq} rows at token stride {st} passes the 32-bit "
                "buffer offset"
            )
    if spec.batch is not None:
        rows = spec.batch * max(spec.seqlen_q, spec.seqlen_kv)
        if rows * spec.num_query_heads * spec.head_size >= _VOFFSET_LIMIT:
            raise ValueError("batch * rows * heads * head_size passes 2^31 elements")
    return body


# Body levers the dense shell does not build (their defaults only): the slice
# sink (its slice slabs, reduce launches and workspace terms are the general
# plan's and shell's), the buffer-bounded workspace stats (the dense shell
# forms its own stats view), the per-slot quad epilogue, the user-tensor
# access width cap and the carried Q / dO slab offsets (the dense addressing
# is its own).
_DENSE_SHELL_DEFAULTS = (
    ("dkv_sink", "atomic"),
    ("ws_bounds", "clamp"),
    ("dkv_cell_vec", 1),
    ("narrow_vec", 1),
    ("slab_carry", False),
)


def _shell_not_built(body: AttnBwdSpec) -> list[str]:
    """Resolved body values that need the general main-kernel shell, which the
    dense shell does not reproduce: the mirror pass (``lb_mirror``, a second
    kv tile per CTA) and the levers of ``_DENSE_SHELL_DEFAULTS``. The Q / dO
    prefetch ring (register or DMA path, every depth) is built by the dense
    shell as by the general one."""
    out = []
    if body.lb_mirror:
        # the paired second kv-tile pass is part of the general shell
        out.append("lb_mirror=True (dense shell)")
    for name, default in _DENSE_SHELL_DEFAULTS:
        value = getattr(body, name)
        if value != default:
            out.append(f"{name}={value!r} (dense shell)")
    return out


def _not_built_all(spec: AttentionDenseBwdSpec, body: AttnBwdSpec) -> list[str]:
    return dense_bwd_not_built(spec) + _not_built(body) + _shell_not_built(body)


def dense_bwd_not_built_reasons(spec: AttentionDenseBwdSpec) -> list[str]:
    """Not-built values of a per-arch dense spec: its dense knobs and the
    knobs of its resolved body (``ValueError`` for an illegal spec)."""
    arch = type(spec).ARCH
    if arch is None:
        raise ValueError("dense_bwd_not_built_reasons needs a per-arch dense spec")
    return _not_built_all(spec, validate_attn_dense_bwd_spec(spec, arch))


# ---------------------------------------------------------------------------
# kernarg ABI rocke.attn_dense_bwd.v1
# ---------------------------------------------------------------------------
#
# Order: pointers, f32 scalars, i32 scalars (every pointer 8 bytes, so the
# packed bytes need no padding). O and LSE are part of the ABI for symmetry
# with the prep launch; the main kernel never dereferences them.

_PTRS = "Q K V O dO LSE dK dV WS_LSE2 WS_DSUM WS_DQ WS_DK WS_DV".split()
_F32 = ("scale_log2", "ds_mult", "dk_mult", "dq_mult")
_I32 = "batch g_split lb_order pair xcd_n xcd_chunk num_persistent".split()


def attn_dense_bwd_params(
    spec: Optional[AttentionDenseBwdSpec] = None,
) -> tuple[tuple[str, str], ...]:
    """Ordered ``(name, struct format)`` kernargs of the dense main kernel.

    ``"Q"`` is a 64-bit global pointer, ``"f"`` an f32, ``"i"`` an i32. The
    list does not depend on the spec; ``spec`` is accepted so callers can pass
    the instance they launch.
    """
    return (
        tuple((p, "Q") for p in _PTRS)
        + tuple((f, "f") for f in _F32)
        + tuple((i, "i") for i in _I32)
    )
