# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# The canonical two-stage wgrad Stage 2 of the cached launcher
# (benchmark_implicit_gemm_conv.stage2_policy) and its env overrides: the
# default environment builds the canonical Stage 2, an override is visible to
# the replay / route checks. Host-only (compile_kernel is stubbed).

from __future__ import annotations

import pytest

from benchmarks.common import benchmark_implicit_gemm_conv as B


def _problem(**kw):
    from kernels.common._conv_implicit_gemm_common import ConvProblem

    base = {"N": 1, "Hi": 18, "Wi": 34, "C": 64, "K": 64, "Y": 3, "X": 3}
    base.update({"pH": 0, "pW": 0}, **kw)
    return ConvProblem(**base)


@pytest.fixture
def clean_env(monkeypatch):
    for k in B.STAGE2_ENV:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_policy_defaults():
    p = _problem()
    assert B.stage2_policy(p, 8) == {"vec": 4}
    assert B.stage2_policy(p, 1024) == {"vec": 4, **B.STAGE2_FOLD}
    assert B.stage2_policy(p, 1024, rezero=True) == {"vec": 4, "rezero": 1}
    assert B.stage2_policy(p, 8, ws_bf16=True) == {"vec": 4, "ws_bf16": True}
    assert (
        B.stage2_policy(_problem(C=64, K=64, groups=2), 1024).get("fold_loop") is None
    )
    odd = _problem(C=3, K=3, X=1, Y=1)
    assert B.stage2_policy(odd, 1024) == {}
    with pytest.raises(ValueError, match="vector Stage 2"):
        B.stage2_policy(odd, 8, ws_bf16=True)


class _Art:
    hsaco, kernel_name = b"", "k"


def _spec_under_env(monkeypatch, ws_replicas, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr("rocke.compile_kernel", lambda *a, **k: _Art())
    _, spec = B._stage2_reduce(
        _problem(), "bf16", "gfx942", {}, lambda **kw: None, ws_replicas
    )
    return spec


@pytest.mark.parametrize("reps", [8, 1024])
def test_default_env_builds_the_canonical_stage2(clean_env, reps):
    assert B.stage2_env_overrides() == {}
    spec = _spec_under_env(clean_env, reps)
    for k, v in B.stage2_policy(_problem(), reps).items():
        assert getattr(spec, k) == v


def test_overrides_are_reported(clean_env):
    spec = _spec_under_env(clean_env, 1024, ROCKE_WGRAD_S2_FOLD_U="2")
    assert spec.fold_loop == 2
    assert B.stage2_env_overrides() == {"ROCKE_WGRAD_S2_FOLD_U": "2"}
    clean_env.setenv("ROCKE_PBD_S2_MSPLIT", "1")
    assert "ROCKE_PBD_S2_MSPLIT" in B.stage2_env_overrides()


def test_route_check_refuses_stage2_overrides(clean_env, tmp_path):
    pytest.importorskip("torch")
    from benchmarks.common import conv_route_check

    clean_env.setenv("ROCKE_WGRAD_S2_VEC", "1")
    argv = ["--routes", str(tmp_path / "none.json"), "--cache", str(tmp_path)]
    assert conv_route_check.main(argv) == 2
