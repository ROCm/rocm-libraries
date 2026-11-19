# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``dispatch.sdpa``'s family-level sweep entry points.

``registered_sdpa_combos`` / ``sdpa_sweep_space`` / ``dispatch_sdpa_all``
return the ``dispatch.attention`` combos of the family an SDPA gate admits a
request to -- the unified tuning variants when ``sdpa_unified_candidates`` is
non-empty, the candidates ``sdpa_dense_candidates`` names -- and nothing for a
request both gates decline.
"""

from __future__ import annotations

from dataclasses import replace
from functools import cache

import pytest

from dispatch.attention.unified_rules import TUNING_ALGORITHM
from dispatch.sdpa import (
    SdpaBias,
    SdpaDropout,
    dispatch_sdpa_all,
    registered_sdpa_combos,
    sdpa_sweep_space,
)
from dispatch.sdpa.bind_dense import sdpa_dense_candidates
from dispatch.sdpa.bind_unified import sdpa_unified_candidates

from tests.sdpa.request_cases import make_request, packed, paged

ARCHES = ("gfx942", "gfx950")


def _request(arch="gfx950", d=128, dtype="fp16", **overrides):
    return make_request(arch=arch, d=d, dtype=dtype, **overrides)


def _gate_names(req):
    return set(sdpa_unified_candidates(req)) | set(sdpa_dense_candidates(req))


def _expected(req, candidate, spec):
    if candidate.algorithm == TUNING_ALGORITHM:
        return bool(sdpa_unified_candidates(req))
    return candidate.name in sdpa_dense_candidates(req)


# One admitted request per arch: gfx950 serves full attention, gfx942 serves
# top-left causal.
_ADMITTED = {
    "gfx942": _request(arch="gfx942", right_bound=0),
    "gfx950": _request(),
}


@cache
def _admitted_combos(arch):
    """``registered_sdpa_combos`` for ``_ADMITTED[arch]``, computed once: it
    enumerates every attention combo, and three tests compare against it."""
    return registered_sdpa_combos(_ADMITTED[arch])


@pytest.mark.parametrize("arch", ARCHES)
def test_combos_are_exactly_the_gate_admitted_families(arch):
    """The gate-admitted families are reachable and nothing else is."""
    from dispatch.attention import registered_attention_combos
    from dispatch.sdpa.bind_shared import to_attention_request

    req = _ADMITTED[arch]
    assert _gate_names(req)
    want = tuple(
        (c, spec)
        for c, spec in registered_attention_combos(to_attention_request(req))
        if _expected(req, c, spec)
    )
    assert want
    assert _admitted_combos(arch) == want
    dense = {c.name for c, _ in want if c.algorithm != TUNING_ALGORITHM}
    assert dense == set(sdpa_dense_candidates(req))


def test_gfx942_full_attention_reaches_dense_but_not_the_unified_tuning_variants():
    """A concrete, named outcome rather than one mirrored from ``_serves``:
    gfx942 full attention (NO_MASK) fails the unified gate's band check (no
    top-left kernel there, see ``bind_unified._band_reject``), but a square
    128x128 shape still clears the dense candidate's own tiling check. The
    family split must drop the tuning variants here and nothing else."""
    req = _request(arch="gfx942", s_q=128, s_kv=128)
    assert not sdpa_unified_candidates(req)
    assert sdpa_dense_candidates(req) == ("attention_gfx942_dense",)
    combos = registered_sdpa_combos(req)
    assert combos
    assert {c.algorithm for c, _ in combos} == {"attention_dense"}


def test_gfx950_padded_reaches_unified_but_not_dense():
    """A concrete, named outcome rather than one mirrored from ``_serves``:
    the dense gate declines ``padded`` outright (the dense candidates take one
    sequence length per batch), while the unified gate carries it. The family
    split must drop every dense combo here and nothing else."""
    req = _request(
        arch="gfx950",
        padding_mask=True,
        seq_len_q=packed((2, 1, 1, 1), "int32"),
        seq_len_kv=packed((2, 1, 1, 1), "int32"),
    )
    assert sdpa_unified_candidates(req)
    assert not sdpa_dense_candidates(req)
    combos = registered_sdpa_combos(req)
    assert combos
    assert {c.algorithm for c, _ in combos} == {TUNING_ALGORITHM}


def test_lse_decode_shape_route_admits_only_split_kv():
    """The production route (auto dispatch) only ever selects the 3D path for
    a decode shape (one query token) -- independently of LSE, since a 2D route
    candidate declines a shape its own ``select_path`` routes to 3D (see
    ``generic.py``'s per-candidate gate)."""
    req = _request(s_q=1, s_kv=64, stats=packed((2, 4, 1, 1)))
    assert sdpa_unified_candidates(req) == ("attention_unified_3d",)


def test_lse_tuning_sweep_includes_the_tiled_2d_path_on_a_prefill_shape():
    """A prefill shape on gfx942/gfx950 writes LSE through the tiled 2D path
    too (``_reject_lse`` in ``kernels/common/attention_unified.py`` allows it
    there, not only on split-KV): an explicit tuning variant for that path
    must stay reachable through the sweep entry points, since it is not
    restricted to the shape the auto route would naturally pick."""
    for arch in ARCHES:
        extra = {"right_bound": 0} if arch == "gfx942" else {}
        req = _request(
            arch=arch, s_q=128, s_kv=128, stats=packed((2, 4, 128, 1)), **extra
        )
        assert sdpa_unified_candidates(req)
        unified = [
            spec
            for c, spec in registered_sdpa_combos(req)
            if c.algorithm == TUNING_ALGORITHM
        ]
        assert unified
        assert "2d" in {spec.path for spec in unified}


@pytest.mark.parametrize("arch", ARCHES)
def test_sweep_space_is_the_specs_of_the_combos(arch):
    req = _ADMITTED[arch]
    assert tuple(sdpa_sweep_space(req)) == tuple(
        spec for _, spec in _admitted_combos(arch)
    )


@pytest.mark.parametrize("arch", ARCHES)
def test_dispatch_all_matches_registered_combos(arch):
    req = _ADMITTED[arch]
    results = dispatch_sdpa_all(req)
    assert results
    assert [(r.candidate, r.spec) for r in results] == list(_admitted_combos(arch))


def _declined_by_both(arch):
    base = _ADMITTED[arch]
    return {
        "bias": replace(base, bias=SdpaBias(packed((2, 4, 8, 8)))),
        "dropout": replace(base, dropout=SdpaDropout(0.1)),
        "block_mask": replace(base, block_mask=packed((2, 4, 1, 1), "uint8")),
        "paged": paged(base),
    }


_DECLINED_PAIRS = [
    (arch, feature) for arch in ARCHES for feature in sorted(_declined_by_both(arch))
]


@pytest.mark.parametrize(
    "arch,feature", _DECLINED_PAIRS, ids=[f"{a}-{f}" for a, f in _DECLINED_PAIRS]
)
def test_all_three_are_empty_when_both_gates_decline(arch, feature):
    """A request the plain request's kernels would serve must not reach them
    through the entry points once a feature the gates decline is added."""
    req = _declined_by_both(arch)[feature]
    assert feature in req.features()
    assert not _gate_names(req)
    assert registered_sdpa_combos(req) == ()
    assert tuple(sdpa_sweep_space(req)) == ()
    assert dispatch_sdpa_all(req) == ()


def test_all_three_are_empty_for_something_that_is_not_an_sdpa_request():
    assert registered_sdpa_combos("not a request") == ()
    assert tuple(sdpa_sweep_space("not a request")) == ()
    assert dispatch_sdpa_all("not a request") == ()
