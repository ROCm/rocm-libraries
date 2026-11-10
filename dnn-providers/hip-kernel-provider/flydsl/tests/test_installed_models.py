# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The generator binds the trained models installed beside an op, as ``uhd_gen promote`` does.

``uhd_gen promote`` files a model at ``heuristics/<ued-id>/<role>/<arch>/<metric>/`` (a
metric-less one directly under ``<role>/<arch>/``) and binds it in the UED as a list per arch
key, for every role. ``gen_descriptors.py`` owns
the UED, so it must bind every such model the same way -- otherwise a regeneration drops the
model, rewrites promote's binding, or ``--check`` refuses a tree that ships one. Host-only, no
``dvc pull``.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import uuid
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
import gen_descriptors  # noqa: E402

OP = "rmsnorm_bwd"
FIELDS = {"dtype": "string", "N": "int"}
UED_ID = gen_descriptors.ued_id(OP, FIELDS)


def _install(
    op_dir: Path,
    role: str,
    arch: str,
    model_id: str,
    metric: str | None = "tflops",
    **overrides,
) -> Path:
    doc = (
        {
            "id": model_id,
            "trained_against": {"ued": {"id": UED_ID}},
        }
        | ({"score": {"metric": metric}} if metric else {})
        | overrides
    )
    folder = op_dir / "heuristics" / UED_ID / role / arch
    if metric:
        folder /= metric
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
    # gfx942's ranker gives the estimate a gfx942 slot, so the pipeline can train one there;
    # gfx950's estimate adds no ranker key, which would change the selector revision.
    assert ued["predict_engine"] == {
        "default": ops.estimate_uhd_id,
        "gfx942": ops.estimate_uhd_id,
        "gfx950": ["model-950"],
    }
    assert ued["sort_kernel_catalog"] == {
        "default": ops.uhd_id,
        "gfx942": ["ranker-942"],
    }


def test_a_default_model_replaces_only_its_own_metric(tmp_path: Path) -> None:
    # The estimate slot declares tflops, so a tflops estimate takes its place; the native
    # ranker declares no metric, so a tflops ranker joins it.
    _install(tmp_path, "predict_engine", "default", "model")
    _install(tmp_path, "sort_kernel_catalog", "default", "ranker")
    ued = _ued(tmp_path)
    assert ued["predict_engine"] == {"default": ["model"]}
    native = gen_descriptors.OpDescriptors(OP, FIELDS).uhd_id
    assert ued["sort_kernel_catalog"] == {"default": [native, "ranker"]}


def test_no_installed_model_leaves_the_engine_as_generated(tmp_path: Path) -> None:
    assert (
        _ued(tmp_path)
        == gen_descriptors.OpDescriptors(OP, FIELDS).documents()[
            f"flydsl_{OP}.ued.json"
        ]
    )


def test_a_metric_less_model_binds_from_its_arch_folder(tmp_path: Path) -> None:
    _install(tmp_path, "sort_kernel_catalog", "gfx942", "ranker", metric=None)
    _install(
        tmp_path, "predict_applicable_kernels", "gfx950", "applicable", metric=None
    )
    ued = _ued(tmp_path)
    native = gen_descriptors.OpDescriptors(OP, FIELDS).uhd_id
    assert ued["sort_kernel_catalog"] == {"default": native, "gfx942": ["ranker"]}
    # The candidate generators (future JIT) are a list like every role.
    assert ued["predict_applicable_kernels"] == {"gfx950": ["applicable"]}


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"trained_against": {"ued": {"id": "another-engine"}}}, "trained against UED"),
        ({"score": {"metric": "time"}}, "filed under metric 'tflops'"),
        ({"score": {}}, "filed under metric 'tflops'"),
    ],
)
def test_a_misfiled_model_is_refused(
    tmp_path: Path, overrides: dict, match: str
) -> None:
    _install(tmp_path, "predict_engine", "gfx950", "model", **overrides)
    with pytest.raises(gen_descriptors.HkpPackError, match=match):
        gen_descriptors.installed_models(tmp_path, UED_ID)


def test_a_metric_model_outside_its_metric_folder_is_refused(tmp_path: Path) -> None:
    path = _install(tmp_path, "predict_engine", "gfx950", "model", metric=None)
    doc = json.loads(path.read_text(encoding="utf-8")) | {"score": {"metric": "tflops"}}
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(gen_descriptors.HkpPackError, match="filed under metric None"):
        gen_descriptors.installed_models(tmp_path, UED_ID)


def test_two_models_for_one_metric_on_one_key_are_refused(tmp_path: Path) -> None:
    _install(tmp_path, "predict_engine", "gfx950", "first")
    _install(tmp_path, "predict_engine", "gfx950", "second")
    with pytest.raises(gen_descriptors.HkpPackError, match="second tflops model"):
        gen_descriptors.installed_models(tmp_path, UED_ID)


@pytest.mark.parametrize(
    "changed",
    [
        {"dtype": "string", "N": "int", "has_dropout": "int"},  # a field added
        {"dtype": "string"},  # a field removed
        {"dtype": "string", "N": "string"},  # a field retyped
        {"N": "int", "dtype": "string"},  # the order its instances declare them in
    ],
)
def test_the_engine_id_moves_exactly_when_its_kernel_fields_do(changed: dict) -> None:
    # A model is trained against one field list, and the UHD pipeline files measurements by
    # engine id: the same fields keep the id (and every model trained against it), other
    # fields get a new id and so new datasets.
    assert gen_descriptors.ued_id(OP, dict(FIELDS)) == UED_ID
    assert gen_descriptors.ued_id(OP, changed) != UED_ID
    assert gen_descriptors.OpDescriptors(OP, FIELDS).ued_id == UED_ID


def test_two_metric_less_models_on_one_key_are_refused(tmp_path: Path) -> None:
    _install(tmp_path, "predict_applicable_kernels", "gfx950", "first", metric=None)
    _install(tmp_path, "predict_applicable_kernels", "gfx950", "second", metric=None)
    with pytest.raises(
        gen_descriptors.HkpPackError, match=r"second \(metric-less\) model"
    ):
        gen_descriptors.installed_models(tmp_path, UED_ID)


def _promote():
    """uhd_gen's promote module, so a binding is checked against promote rather than a copy
    of its rules. Skips without uhd_gen's requirements, as uhd_gen's suites do, unless the
    build requires them (HIPDNN_REQUIRE_UHD_GEN_TESTS)."""
    sys.path.insert(0, str(FLYDSL.parents[2] / "projects" / "hipdnn" / "tools"))
    if os.environ.get("HIPDNN_REQUIRE_UHD_GEN_TESTS"):
        from uhd_gen import promote

        return promote
    return pytest.importorskip("uhd_gen.promote")


@pytest.mark.parametrize(
    ("role", "arch", "metric"),
    [
        ("predict_engine", "gfx950", "tflops"),  # a new key
        ("predict_engine", "default", "tflops"),  # replaces the estimate slot
        ("sort_kernel_catalog", "gfx942", "tflops"),  # a new key
        (
            "sort_kernel_catalog",
            "default",
            "tflops",
        ),  # joins the metric-less native ranker
        ("sort_kernel_catalog", "default", None),  # replaces the native ranker
        ("predict_applicable_kernels", "gfx950", None),  # the candidate generators
    ],
)
def test_a_binding_is_what_promote_writes(
    tmp_path: Path, role: str, arch: str, metric: str | None
) -> None:
    promote = _promote()
    before = _ued(tmp_path)
    # promote reads the metric of every UHD a key already binds from its installed file.
    installed_ids = {}
    for name, doc in gen_descriptors.OpDescriptors(OP, FIELDS).documents().items():
        if name.endswith(".uhd.json"):
            path = tmp_path / name
            path.write_text(json.dumps(doc), encoding="utf-8")
            installed_ids[doc["id"]] = path
    model = str(uuid.uuid4())
    installed_ids[model] = _install(tmp_path, role, arch, model, metric=metric)
    holder = copy.deepcopy(before)
    written, _ = promote._rebind(holder, role, arch, model, metric, installed_ids)
    assert _ued(tmp_path)[role][arch] == written
