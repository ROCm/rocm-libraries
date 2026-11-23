# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Every UHD model hip-kernel-provider ships scores the graphs that leave an optional feature
unbound.

A model that reads an optional feature bare refuses every valid graph that leaves it unset, so
the engine falls back to declared order or loses its estimate for that graph (RFC 0019 §6.7,
absent is not zero; §6.8 item 9). ``uhd_gen promote`` refuses such a signature, but a model
committed by hand or retrained outside promote is not checked, and the descriptor validator
only warns. This test lints every shipped ``predict_engine`` and ``sort_kernel_catalog``
model of every engine (kernel-ingestor bundles of any pack source and the native engines)
through hipDNN's feature evaluator, the same check promote applies. A model that fails is
fixed with ``python -m uhd_gen rewrite-signature``. Host-only; reads the signatures in the
source tree. Most are committed in git; the models of a bundle whose whole ``heuristics``
directory is a DVC directory (the conv AOT rankers) are present, and linted, only after
``dvc pull``, which the superbuild CI jobs run first.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.quick

HKP = Path(__file__).resolve().parents[2]
ENGINES = HKP / "src" / "engines"
ROLES = ("predict_engine", "sort_kernel_catalog")


def _shipped_models() -> list[Path]:
    return sorted(
        path
        for path in ENGINES.rglob("*.uhd.json")
        if any(role in path.parts for role in ROLES)
    )


def _lint():
    """uhd_gen's lint and a runnable evaluator. Skips without them, as uhd_gen's suites do,
    unless the build requires those suites (HIPDNN_REQUIRE_UHD_GEN_TESTS)."""
    sys.path.insert(0, str(HKP.parents[1] / "projects" / "hipdnn" / "tools"))
    required = bool(os.environ.get("HIPDNN_REQUIRE_UHD_GEN_TESTS"))
    if required:
        from uhd_gen.features import resolve_feature_evaluator
        from uhd_gen.optional import lint
    else:
        lint = pytest.importorskip("uhd_gen.optional").lint
        resolve_feature_evaluator = pytest.importorskip(
            "uhd_gen.features"
        ).resolve_feature_evaluator
    try:
        evaluator = resolve_feature_evaluator()
    except ValueError as error:
        if required:
            raise
        pytest.skip(f"no hipdnn_uhd_features evaluator: {error}")
    return lint, evaluator


def test_the_shipped_models_are_found():
    models = _shipped_models()
    roles = {role for path in models for role in ROLES if role in path.parts}
    assert roles == set(
        ROLES
    ), f"expected both roles under {ENGINES}, found {sorted(roles)}"
    assert any(
        "asm_sdpa_engine" in path.parts for path in models
    ), "the native ASM SDPA models are not under the searched tree"


@pytest.mark.parametrize(
    "model", _shipped_models(), ids=lambda path: path.relative_to(ENGINES).as_posix()
)
def test_every_optional_read_has_a_default(model):
    lint, evaluator = _lint()
    descriptor = json.loads(model.read_text(encoding="utf-8"))
    unguarded = lint(
        descriptor["features_signature"],
        descriptor.get("score", {}).get("metric"),
        evaluator,
    )
    assert not unguarded, (
        f"{model.relative_to(HKP).as_posix()} reads features a valid graph may leave unbound, "
        f"so such a graph is unscorable: {[found['reference'] for found in unguarded]}; "
        "rewrite it with `python -m uhd_gen rewrite-signature`"
    )
