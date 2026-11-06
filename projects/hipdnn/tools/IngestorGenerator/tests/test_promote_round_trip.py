# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""`engine.heuristics` regenerates exactly the UED the real `uhd_gen promote` writes.

The test runs promote itself on a generated bundle, then regenerates the UED from a
config that records the same binding. A copied fixture of promote's output cannot catch
promote and the generator disagreeing about a role's shape (RFC 0019 §3.1: the metric
roles are lists, `predict_applicable_kernels` is one id); this can."""

import json
import sys
from pathlib import Path

import pytest
import yaml

from codegen.config_loader import load_config
from codegen.generator import _dump, build_ued
from tests.helpers import read_docs

MODEL = "cf37fa30-32dc-4a21-a008-68ef5e0d30a6"
#: Features per role: a catalog ranker scores candidates by their knobs; an engine
#: predictor sees the graph only (RFC 0019 L1), so promote refuses a kernel feature there.
KERNEL_SIGNATURE = ["$kernel.block_size"]
GRAPH_SIGNATURE = ["$graph.work"]

# Each case is one `uhd_gen promote --role R --arch A`, then the binding that
# `engine.heuristics` must hold for the generator to write the same UED.
CASES = [
    pytest.param("sort_kernel_catalog", "gfx942", [MODEL], id="sort_kernel_catalog"),
    pytest.param("predict_engine", "gfx942", [MODEL], id="predict_engine"),
    pytest.param(
        "predict_applicable_kernels", "gfx942", MODEL, id="predict_applicable_kernels"
    ),
    pytest.param(
        "predict_applicable_kernels", "default", MODEL, id="predict_applicable_default"
    ),
]


def _uhd_gen():
    """The promote entry points, or a skip when this environment cannot run them.

    `uhd_gen` lives beside this tool; its models are FlatBuffers whose `features_hash` only
    the built `hipdnn_uhd_features` evaluator can compute (it has no Python fallback).
    """
    # The same three imports uhd_gen's own promote tests require: the shipping converter
    # and the promote module import them at load.
    for dependency in ("flatbuffers", "lightgbm", "pandas"):
        pytest.importorskip(dependency)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from uhd_gen import features, lgbm_to_flatbuffer, promote, provenance

    try:
        evaluator = features.resolve_feature_evaluator()
    except ValueError as error:
        pytest.skip(str(error))
    return evaluator, features, lgbm_to_flatbuffer, promote, provenance


def _write_model(root: Path, tree: Path, role: str, arch: str, uhd_gen) -> Path:
    """A tree_data model the runtime would load, built as uhd_gen's own tests build it."""
    _, features, lgbm_to_flatbuffer, _, provenance = uhd_gen
    signature = GRAPH_SIGNATURE if role == "predict_engine" else KERNEL_SIGNATURE
    features_hash = features.compute_features_hash(signature, None)
    artifact = lgbm_to_flatbuffer.build_gbdt_model(
        {
            "tree_info": [{"tree_structure": {"leaf_value": 1.0}}],
            "max_feature_idx": 0,
            "objective": "regression",
        },
        features_hash,
    )
    trained = provenance.snapshot_provenance(
        tree, arch=arch if arch != "default" else "gfx942"
    )
    document = {
        "version": "1.0",
        "id": MODEL,
        "name": "model",
        "adapter": "tree_data",
        "objective": "max",
        "features_signature": signature,
        "features_hash": features_hash,
        "trained_against": trained,
        "tree_data": {"artifact": "model.bin"},
    }
    if role == "predict_engine":
        # An engine predictor must say what it predicts; promote refuses a metric-less one.
        document["score"] = {
            "metric": "tflops",
            "calibrated": True,
            "transform": "log1p",
        }
    if role == "predict_engine":
        # The runtime compares this with the engine provider's current selector revision.
        trained["selector_revision"] = "provider-1"
    root.mkdir(parents=True)
    (root / "model.bin").write_bytes(artifact)
    (root / "heuristic.uhd.json").write_text(
        json.dumps(document, indent=2) + "\n", encoding="utf-8"
    )
    (root / "train_manifest.json").write_text(
        json.dumps({"training_arches": ["gfx942"], "trained_against": trained}),
        encoding="utf-8",
    )
    return root


@pytest.mark.allow_skip(
    reason="needs flatbuffers, lightgbm, pandas and the built hipdnn_uhd_features evaluator, which a "
    "build without the hipDNN heuristics tooling does not have"
)
@pytest.mark.parametrize("role, arch, binding", CASES)
def test_promoted_ued_regenerates_from_engine_heuristics(
    role, arch, binding, generator, scale_add_config, configs_dir, tmp_path
):
    uhd_gen = _uhd_gen()
    evaluator, _, _, promote, _ = uhd_gen
    assert scale_add_config.engine.heuristics == {}

    generator.render(scale_add_config, tmp_path)
    tree = tmp_path / scale_add_config.descriptor_dir
    [ued_path] = tree.glob("*.ued.json")
    generated = json.loads(ued_path.read_text(encoding="utf-8"))
    ids = {
        "ued": generated["id"],
        "kmd": generated["metadata"],
        "uhd": generated["sort_kernel_catalog"]["default"],
    }
    assert read_docs(tree)[ued_path.name] == generated

    model = _write_model(tmp_path / "model", tree, role, arch, uhd_gen)
    plan = promote.build_plan(
        model, tree, role=role, arch=arch, feature_evaluator=evaluator
    )
    promote._apply(plan)

    # Through the loader, so its shape rules are part of what the promoted file checks.
    raw = yaml.safe_load((configs_dir / "scale_add.yaml").read_text())
    raw["engine"]["heuristics"] = {role: {arch: binding}}
    config_path = tmp_path / "bound.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    bound = load_config(config_path)
    # Text, not bytes: the generator writes LF, while an autocrlf checkout holds the
    # promoted file as CRLF (as test_a_promoted_ued_regenerates_byte_identical notes).
    assert _dump(build_ued(bound, ids)) == ued_path.read_text(encoding="utf-8")
