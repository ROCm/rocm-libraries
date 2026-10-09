# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU contract tests for the gfx950 GDN decode candidate registry."""

from __future__ import annotations

from dataclasses import asdict, replace

import pytest

from dispatch.gdn import (
    GdnDecodeRequest,
    dispatch_gdn_decode,
    dispatch_gdn_decode_all,
    gdn_candidates,
)
from dispatch.gdn.gfx950 import ARCH, CONFIGURED_TILES, DEFAULT_TILE, make_spec
from kernels.gfx950.gdn_decode import is_valid_spec
from rocke.dispatch.core import stable_json_hash


def _req(batch: int = 16, **changes) -> GdnDecodeRequest:
    return GdnDecodeRequest(batch=batch, arch=ARCH, **changes)


def _tile(spec) -> tuple[int, int, int]:
    return spec.num_warps, spec.warp_threads_k, spec.blocks_per_v_dim


def _legal_results(req: GdnDecodeRequest | None = None):
    return dispatch_gdn_decode_all(req or _req())


def test_configured_count_and_identity_are_stable_and_unique():
    candidates = tuple(
        candidate
        for candidate in gdn_candidates()
        if not candidate.spec_id.startswith("kda_")
    )
    expected_ids = tuple(
        f"nw{nw}_wtk{wtk}_bpv{bpv}" for nw, wtk, bpv in CONFIGURED_TILES
    )
    assert len(CONFIGURED_TILES) == 180
    assert len(candidates) == 180
    assert tuple(candidate.spec_id for candidate in candidates) == expected_ids
    assert len({candidate.name for candidate in candidates}) == 180
    assert all(
        candidate.name == f"gdn_decode_gfx950_{candidate.spec_id}"
        for candidate in candidates
    )


def test_legal_d128_count_is_54_and_validator_is_authority():
    req = _req()
    results = _legal_results(req)
    expected = {
        tile
        for tile in CONFIGURED_TILES
        if is_valid_spec(make_spec(req, tile), arch=ARCH)[0]
    }
    assert len(expected) == 54
    assert {_tile(result.spec) for result in results} == expected
    assert len(results) == 54
    assert all(is_valid_spec(result.spec, arch=ARCH)[0] for result in results)


def test_legal_compile_identities_are_unique():
    results = _legal_results()
    identity_sets = (
        {result.candidate.name for result in results},
        {result.candidate.spec_id for result in results},
        {stable_json_hash(asdict(result.spec), n=16) for result in results},
        {result.spec.kernel_name() for result in results},
        {result.kernel_id.compile_key for result in results},
    )
    assert all(len(identities) == 54 for identities in identity_sets)


def test_explicit_representatives_select_exact_tile_at_every_batch():
    legal = _legal_results()
    selected = (legal[0], legal[len(legal) // 2], legal[-1])
    for batch in (1, 16, 64, 256):
        for expected in selected:
            result = dispatch_gdn_decode(
                _req(
                    batch,
                    algorithm=expected.candidate.algorithm,
                    spec_id=expected.candidate.spec_id,
                )
            )
            assert result.candidate.spec_id == expected.candidate.spec_id
            assert _tile(result.spec) == _tile(expected.spec)


def test_every_legal_pin_round_trips_and_illegal_pin_fails_loudly():
    req = _req()
    for expected in _legal_results(req):
        result = dispatch_gdn_decode(
            replace(
                req,
                algorithm=expected.candidate.algorithm,
                spec_id=expected.candidate.spec_id,
            )
        )
        assert result.candidate.spec_id == expected.candidate.spec_id
        assert _tile(result.spec) == _tile(expected.spec)

    with pytest.raises(ValueError, match="nw4_wtk16_bpv8"):
        dispatch_gdn_decode(
            _req(
                head_k_dim=64,
                algorithm="warp_tiled",
                spec_id="nw4_wtk16_bpv8",
            )
        )
    with pytest.raises(ValueError, match="does_not_exist"):
        dispatch_gdn_decode(_req(algorithm="warp_tiled", spec_id="does_not_exist"))


def test_auto_is_static_default_independent_of_batch():
    outcomes = []
    for batch in (1, 16, 64, 256):
        repeated = [dispatch_gdn_decode(_req(batch)) for _ in range(20)]
        assert {_tile(result.spec) for result in repeated} == {DEFAULT_TILE}
        assert len({result.candidate.spec_id for result in repeated}) == 1
        outcomes.append(repeated[0].candidate.spec_id)
    assert len(set(outcomes)) == 1


def test_gdn_auto_ignores_custom_rankers_when_default_is_legal():
    def reverse(request, candidates):
        return tuple(reversed(candidates))

    result = dispatch_gdn_decode(_req(), ranker=reverse)

    assert _tile(result.spec) == DEFAULT_TILE


def test_d64_auto_falls_back_to_a_validator_approved_candidate():
    result = dispatch_gdn_decode(_req(head_k_dim=64))
    assert is_valid_spec(result.spec, arch=ARCH)[0]


def test_registry_and_benchmark_enumeration_have_identity_bijection():
    from benchmarks.gfx950.gdn.benchmark_gdn_decode import registered_results

    req = _req()
    registry = _legal_results(req)
    benchmark = registered_results(req)

    def identities(results):
        return {
            (
                result.candidate.name,
                result.candidate.spec_id,
                result.kernel_id.spec_hash,
                result.spec.kernel_name(),
                result.kernel_id.compile_key,
            )
            for result in results
        }

    assert len(registry) == len(benchmark) == 54
    assert identities(registry) == identities(benchmark)


def test_benchmark_reports_missing_candidates_before_selecting_auto(
    monkeypatch, capsys
):
    import sys

    from benchmarks.gfx950.gdn import benchmark_gdn_decode

    monkeypatch.setattr(benchmark_gdn_decode, "registered_results", lambda request: ())
    monkeypatch.setattr(
        benchmark_gdn_decode,
        "dispatch_gdn_decode",
        lambda request: pytest.fail("auto must not be selected without candidates"),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["benchmark_gdn_decode.py", "--batches", "1"],
    )

    assert benchmark_gdn_decode.main() == 1
    assert (
        "batch 1: expected 54 legal registry candidates, got 0"
        in capsys.readouterr().err
    )
