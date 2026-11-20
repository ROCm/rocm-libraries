# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The dense attention C++ tests check the hand-restated launch grid, fast division and
cold-score policies against golden tables generated from rocKE (and, for the score, from
the gfx942 manifest's full enumeration). A table nobody regenerates goes stale silently:
the C++ keeps passing against it after rocKE or the manifest moves. Each generator here is
rendered in memory and compared with its checked-in .inc.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import registry_enumerator  # noqa: E402

_REPO = Path(__file__).resolve().parents[5]
_PROVIDER = _REPO / "dnn-providers" / "hip-kernel-provider"
_ROCKE = _PROVIDER / "rocke"
_PACKS = _PROVIDER / "src/tests/engines/kernel_ingestor_engine/packs"
_KNOBS = _PROVIDER / "src/integration_tests/kernel_ingestor_engine"

# (directory, generator, table): the persistent grid / fast-division goldens (shared with
# gfx950), the gfx942 cold-score policy goldens over the full enumeration, and the gfx942
# knob suite's per-kernel rows from the shipped selection.
_GOLDENS = (
    (
        _PACKS,
        "gen_attention_dense_persistent_goldens.py",
        "TestAttentionDensePersistentGoldens.inc",
    ),
    (
        _PACKS,
        "gen_gfx942_attention_dense_score_goldens.py",
        "TestGfx942AttentionDenseScoreGoldens.inc",
    ),
    (
        _KNOBS,
        "gen_gfx942_attention_dense_knob_classes.py",
        "IntegrationGpuGfx942AttentionDenseKnobClasses.inc",
    ),
)


def _load(generator: Path):
    # The persistent generator imports rocKE at module scope, so bind before exec.
    registry_enumerator.bind_provider(str(_PROVIDER))
    spec = importlib.util.spec_from_file_location(
        f"_golden_{generator.stem}", generator
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "directory,generator,table", _GOLDENS, ids=[g for _, g, _ in _GOLDENS]
)
def test_the_checked_in_golden_table_is_what_its_generator_renders(
    directory, generator, table
):
    if not (_ROCKE / "library").is_dir():
        pytest.skip(f"rocKE library not in this checkout: {_ROCKE / 'library'}")
    module = _load(directory / generator)
    assert module.OUT == directory / table
    assert module.render() == (directory / table).read_text(
        encoding="utf-8"
    ), f"{table} is stale: rerun {generator} (see its docstring) and commit the table"
