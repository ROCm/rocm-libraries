# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The gfx950 GDN/KDA decode tuning space as data: the knob axes and the
``waves_per_eu`` values.

Nothing here walks or builds a spec: the axis helpers and the walk are the
shared :mod:`rocke.dispatch.tuning`, and :mod:`.rules` decides which axes a
request reads and what is legal.

Every knob is a :class:`~kernels.gfx950.gdn_decode.GdnDecodeSpec` field whose
dataclass default is the code the kernel emitted before the knob existed, so
each registered tile's base configuration (no knobs) is the kernel ``auto``
selected before the registry existed. What each knob does is documented on the
spec field (ALGORITHM.md §4.10); where it changes code is the kernel's
``*_applies`` predicates and ``is_valid_spec``:

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
from typing import Tuple

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
