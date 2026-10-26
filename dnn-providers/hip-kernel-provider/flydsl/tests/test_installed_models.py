# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The generator binds the trained models installed beside an op.

``uhd_gen promote`` files a model at ``heuristics/<ued-id>/<role>/<arch>/<metric>/``;
``gen_descriptors.py`` owns the UED, so it must bind every such model under its role and
arch key -- otherwise a regeneration drops the model and ``--check`` refuses a tree that
ships one. Host-only, no ``dvc pull``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
import gen_descriptors  # noqa: E402

OP = "rmsnorm_bwd"
UED_ID = gen_descriptors._uuid(f"ued/{OP}")
FIELDS = {"dtype": "string", "N": "int"}


def _install(op_dir: Path, role: str, arch: str, model_id: str, **overrides) -> Path:
    doc = {
        "id": model_id,
        "score": {"metric": "tflops"},
        "trained_against": {"ued": {"id": UED_ID}},
    } | overrides
    folder = op_dir / "heuristics" / UED_ID / role / arch / "tflops"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{model_id}.uhd.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def _ued(op_dir: Path) -> dict:
    installed = gen_descriptors.installed_models(op_dir, UED_ID)
    docs = gen_descriptors.OpDescriptors(OP, FIELDS, installed).documents()
    return docs[f"flydsl_{OP}.ued.json"]


def test_an_arch_model_binds_under_its_arch_and_keeps_the_default(
    tmp_path: Path,
) -> None:
    _install(tmp_path, "predict_engine", "gfx950", "model-950")
    _install(tmp_path, "sort_kernel_catalog", "gfx942", "ranker-942")
    ops = gen_descriptors.OpDescriptors(OP, FIELDS)
    ued = _ued(tmp_path)
    assert ued["predict_engine"] == {
        "default": ops.estimate_uhd_id,
        "gfx950": "model-950",
    }
    assert ued["sort_kernel_catalog"] == {"default": ops.uhd_id, "gfx942": "ranker-942"}


def test_a_default_model_replaces_only_its_own_metric(tmp_path: Path) -> None:
    # The estimate slot declares tflops, so a tflops estimate takes its place; the native
    # ranker declares no metric, so a tflops ranker joins it.
    _install(tmp_path, "predict_engine", "default", "model")
    _install(tmp_path, "sort_kernel_catalog", "default", "ranker")
    ued = _ued(tmp_path)
    assert ued["predict_engine"] == {"default": "model"}
    native = gen_descriptors.OpDescriptors(OP, FIELDS).uhd_id
    assert ued["sort_kernel_catalog"] == {"default": [native, "ranker"]}


def test_no_installed_model_leaves_the_engine_as_generated(tmp_path: Path) -> None:
    assert (
        _ued(tmp_path)
        == gen_descriptors.OpDescriptors(OP, FIELDS).documents()[
            f"flydsl_{OP}.ued.json"
        ]
    )


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"trained_against": {"ued": {"id": "another-engine"}}}, "trained against UED"),
        ({"score": {"metric": "time"}}, "filed under metric 'tflops'"),
    ],
)
def test_a_misfiled_model_is_refused(
    tmp_path: Path, overrides: dict, match: str
) -> None:
    _install(tmp_path, "predict_engine", "gfx950", "model", **overrides)
    with pytest.raises(gen_descriptors.HkpPackError, match=match):
        gen_descriptors.installed_models(tmp_path, UED_ID)


def test_two_models_for_one_metric_on_one_key_are_refused(tmp_path: Path) -> None:
    _install(tmp_path, "predict_engine", "gfx950", "first")
    _install(tmp_path, "predict_engine", "gfx950", "second")
    with pytest.raises(gen_descriptors.HkpPackError, match="second 'tflops' model"):
        gen_descriptors.installed_models(tmp_path, UED_ID)
