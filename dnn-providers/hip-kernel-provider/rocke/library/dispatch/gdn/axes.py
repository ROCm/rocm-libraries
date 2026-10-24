# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The gfx950 GDN/KDA decode tuning space as data: the knob axes, the
``waves_per_eu`` values, and the static ``auto`` default of every case.

Nothing here walks or builds a spec: the axis helpers and the walk are the
shared :mod:`rocke.dispatch.tuning`, and :mod:`.rules` decides which axes a
request reads and what is legal.

Every knob is a :class:`~kernels.gfx950.gdn_decode.GdnDecodeSpec` field whose
dataclass default is the code the kernel emitted before the knob existed, so
each registered tile's base configuration (no knobs) is that code. ``auto``
runs one ``AUTO_DEFAULTS`` entry per case: a tile plus knobs on top of its
base. What each knob does is documented on the spec field (ALGORITHM.md
§4.10); where it changes code is the kernel's ``*_applies`` predicates and
``is_valid_spec``:

* ``state_load_hint`` / ``state_store_hint``: cache policy of the
  recurrent-state loads and stores (``"default"`` or ``"streaming"``);
* ``dpp_reduce``: the in-group sum's xor-4 / xor-8 stages on DPP row mirrors;
* ``xcd_remap``: one contiguous range of work per XCD;
* ``stream_rows``: decay and update each owned state row in its own iteration;
* ``interleave_cols``: interleaved 16 B column runs for an f32 state;
* ``conv_once`` (``fuse_conv``): each conv channel computed once per
  workgroup and shared through LDS;
* ``norm_gate_once`` (``conv_once`` + ``fuse_out_norm``): the norm gate times
  its weight computed once per output row into LDS;
* ``out_lds`` (``fuse_out_norm``): packed output stores through LDS;
* ``waves_per_eu`` (the space's outer knob, ``0`` = compiler's choice): the
  ``amdgpu-waves-per-eu`` occupancy floor.
"""

from __future__ import annotations

import dataclasses as dc
from types import MappingProxyType
from typing import Mapping, Tuple

from kernels.gfx950.gdn_decode import _MAX_WAVES_PER_EU, STATE_HINTS, GdnDecodeSpec
from rocke.dispatch.tuning.axes import KnobAxis, flag, values

_DEFAULT_STATE_HINT = {f.name: f.default for f in dc.fields(GdnDecodeSpec)}[
    "state_load_hint"
]

# Ordered prerequisites first: ``norm_gate_once`` is legal only with
# ``conv_once`` on, so it follows it and the walk prunes it while conv_once is
# off.
STATE_LOAD_HINT_AXIS = values("state_load_hint", _DEFAULT_STATE_HINT, STATE_HINTS)
STATE_STORE_HINT_AXIS = values("state_store_hint", _DEFAULT_STATE_HINT, STATE_HINTS)
CONV_ONCE_AXIS = flag("conv_once")
NORM_GATE_ONCE_AXIS = flag("norm_gate_once")
DPP_REDUCE_AXIS = flag("dpp_reduce")
XCD_REMAP_AXIS = flag("xcd_remap")
STREAM_ROWS_AXIS = flag("stream_rows")
INTERLEAVE_COLS_AXIS = flag("interleave_cols")
OUT_LDS_AXIS = flag("out_lds")

AXES: Tuple[KnobAxis, ...] = (
    STATE_LOAD_HINT_AXIS,
    STATE_STORE_HINT_AXIS,
    CONV_ONCE_AXIS,
    NORM_GATE_ONCE_AXIS,
    DPP_REDUCE_AXIS,
    XCD_REMAP_AXIS,
    STREAM_ROWS_AXIS,
    INTERLEAVE_COLS_AXIS,
    OUT_LDS_AXIS,
)

# The outer knob's values per sweep level, swept on top of every knob set. The
# first, 0 (no attribute: the compiler's choice), is the kernel default and is
# what each tile's base configuration uses. As for attention
# (``dispatch/attention/waves.py``), ``production`` -- what
# ``dispatch_gdn_decode_all`` lists unless a sweep selects ``full`` -- is a
# short list and ``full`` is wider. Production adds only 4: the one non-zero
# value the gfx950 knob numeric suite runs on every configured tile
# (``test_gdn_decode_knobs_gfx950_numeric.py``). Full tries every value the
# kernel accepts (``is_valid_spec``: 0 to ``_MAX_WAVES_PER_EU``). Any legal value
# is pinnable at either level; the level only decides what is listed.
PRODUCTION_WAVES_PER_EU: Tuple[int, ...] = (0, 4)
FULL_WAVES_PER_EU: Tuple[int, ...] = tuple(range(_MAX_WAVES_PER_EU + 1))


@dc.dataclass(frozen=True)
class AutoDefault:
    """One case's ``auto`` configuration: a registered tile and the knobs set
    on top of that tile's base configuration (every other knob stays at its
    dataclass default)."""

    tile: Tuple[int, int, int]
    knobs: Mapping[str, object]


def _auto(tile: Tuple[int, int, int], **knobs) -> AutoDefault:
    return AutoDefault(tile, MappingProxyType(dict(_STREAMING_STATE, **knobs)))


# Every case streams the state: nontemporal state loads and stores (the state
# is read once and written once per call). Each row lists its other knobs;
# a knob it does not list stays at its default (off, waves_per_eu 0).
_STREAMING_STATE = {"state_load_hint": "streaming", "state_store_hint": "streaming"}

# The state width an ``auto`` request reads its row with. f16 state was not
# measured; it moves the same bytes as bf16, so it runs the bf16 row.
AUTO_STATE_ROW: Mapping[str, str] = MappingProxyType(
    {"bf16": "bf16", "f16": "bf16", "f32": "f32"}
)

# The static ``auto`` default per case: (gate_kind, fused, state row), where
# fused means fuse_conv or fuse_out_norm. One entry per case, never keyed on
# batch, head count or work: the batch only changes the grid. Chosen on gfx950
# in cold memory at head_k_dim = head_v_dim = 128 over three head geometries
# per case and batch 1..256 (ALGORITHM.md §4.6 names the shapes; the fused
# cases were timed with both fuse_conv and fuse_out_norm): a screen of every
# legal tile x knob configuration, then a retime of the leaders (arms
# interleaved, three runs). Rule: the lowest geomean against each shape's
# fastest configuration; among the configurations near-tied with it, the lower
# worst-shape ratio, near-tied worst shapes counting as tied; then fewer
# non-default knobs; then the lower geomean. gfx942 is not measured.
#
# A knob a request cannot read is dropped for it, never refused (rules.py
# scope): the fused-only knobs on a request with only one fused flag (and
# norm_gate_once with conv_once, which the kernel requires for it), and
# dpp_reduce / stream_rows / interleave_cols where the tile's ``*_applies``
# predicate is False. To revisit, sweep the case's tiles x knobs cold, then
# change the row here, the literal table in test_gfx950_wiring.py and the
# table in ALGORITHM.md §4.6 together.
AUTO_DEFAULTS: Mapping[Tuple[str, bool, str], AutoDefault] = MappingProxyType(
    {
        ("gdn", False, "bf16"): _auto((2, 16, 8), dpp_reduce=True),
        ("gdn", False, "f32"): _auto(
            (2, 16, 16), dpp_reduce=True, interleave_cols=True
        ),
        ("gdn", True, "bf16"): _auto(
            (4, 16, 1),
            dpp_reduce=True,
            stream_rows=True,
            conv_once=True,
            norm_gate_once=True,
            out_lds=True,
        ),
        ("gdn", True, "f32"): _auto(
            (8, 16, 1),
            dpp_reduce=True,
            stream_rows=True,
            interleave_cols=True,
            conv_once=True,
            out_lds=True,
            waves_per_eu=4,
        ),
        ("kda", False, "bf16"): _auto((1, 16, 8), dpp_reduce=True, waves_per_eu=4),
        ("kda", False, "f32"): _auto(
            (1, 16, 16), dpp_reduce=True, stream_rows=True, interleave_cols=True
        ),
        ("kda", True, "bf16"): _auto(
            (4, 8, 1), dpp_reduce=True, conv_once=True, out_lds=True
        ),
        # An override of the rule, chosen for batch 1 (the latency case): the
        # rule picks (16, 16, 1) with interleave_cols, conv_once, out_lds and
        # waves_per_eu 4; this row is the (4, 8, 1) configuration with the
        # best selection score among the (4, 8, 1) arms, near-tied with the
        # rule's pick on that score.
        ("kda", True, "f32"): _auto(
            (4, 8, 1),
            dpp_reduce=True,
            stream_rows=True,
            interleave_cols=True,
            conv_once=True,
            out_lds=True,
        ),
    }
)

# Each row's knobs are knobs of the space and its values values of their axes
# (waves_per_eu: an int the kernel accepts), so a mistyped row fails at
# import; a well-formed stack the space refuses on a request raises at
# dispatch (``auto_knobs``), never runs another configuration.
_AXIS_VALUES = {
    axis.name: {value for choice in axis.choices for _, value in choice}
    for axis in AXES
}
_AXIS_VALUES["waves_per_eu"] = set(FULL_WAVES_PER_EU)
for _case, _default in AUTO_DEFAULTS.items():
    for _name, _value in _default.knobs.items():
        _allowed = _AXIS_VALUES.get(_name, set())
        assert _value in _allowed and type(_value) in {type(v) for v in _allowed}, (
            _case,
            _name,
            _value,
        )
    # Fused requests run one workgroup per head.
    assert not _case[1] or _default.tile[2] == 1, _case
