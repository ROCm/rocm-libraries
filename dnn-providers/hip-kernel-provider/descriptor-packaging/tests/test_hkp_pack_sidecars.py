# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests that the packer resolves and carries the model file a trained UHD names."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from hkp_pack.descriptors import HkpPackError, load_flat_input
from hkp_pack.pipeline import compile_intermediate

pytestmark = pytest.mark.quick


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _model_uhd(artifact: str) -> dict:
    return {
        "version": "1.0",
        "id": "133a8b19-8e34-4f74-86d6-b6495a6483f3",
        "name": "Trained heuristic",
        "adapter": "tree_data",
        "features_signature": ["$kernel.tile_m"],
        "features_hash": "sha256:" + "0" * 16,
        "trained_against": {
            "ued": {"id": "699a8b19-8e34-4f74-86d6-b6495a6483f3", "revision": "1.0"},
            "kmd": {"id": "799a8b19-8e34-4f74-86d6-b6495a6483f3", "revision": "1.0"},
            "umd": [],
        },
        "objective": "max",
        "tree_data": {"artifact": artifact},
    }


def _native_uhd() -> dict:
    return {
        "version": "1.0",
        "id": "233a8b19-8e34-4f74-86d6-b6495a6483f3",
        "name": "Native heuristic",
        "adapter": "native",
        "objective": "max",
        "native": {"symbol": "hipkernel.pointwise.score"},
    }


def _root_with_model_uhd(tmp_path: Path, artifact: str = "model.bin") -> Path:
    """A minimal root: one model UHD and the artifact it names."""
    root = tmp_path / "src"
    _write_json(root / "pack" / "heuristic.uhd.json", _model_uhd(artifact))
    path = root / "pack" / "model.bin"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"HGBM-stub")
    return root


class TestResolution:
    def test_the_named_artifact_becomes_a_sidecar(self, tmp_path: Path):
        flat = load_flat_input(_root_with_model_uhd(tmp_path), log=lambda *_: None)

        (uhd,) = [d for d in flat.descriptors if d.type == "uhd"]
        (sidecar,) = uhd.sidecars
        assert sidecar.name == "model.bin"
        assert sidecar.rel_dir == Path("pack")
        assert sidecar.source.read_bytes() == b"HGBM-stub"

    def test_a_native_uhd_names_a_symbol_not_a_file(self, tmp_path: Path):
        root = tmp_path / "src"
        _write_json(root / "pack" / "native.uhd.json", _native_uhd())

        flat = load_flat_input(root, log=lambda *_: None)

        (uhd,) = [d for d in flat.descriptors if d.type == "uhd"]
        assert uhd.sidecars == []

    def test_a_native_uhd_carries_nothing_from_its_folder(self, tmp_path: Path):
        root = tmp_path / "src"
        _write_json(root / "pack" / "native.uhd.json", _native_uhd())
        (root / "pack" / "unrelated.bin").write_bytes(b"not mine")

        flat = load_flat_input(root, log=lambda *_: None)

        (uhd,) = [d for d in flat.descriptors if d.type == "uhd"]
        assert uhd.sidecars == []

    def test_an_unnamed_file_beside_a_model_uhd_is_not_carried(self, tmp_path: Path):
        """Only the named file travels, not training data or notes beside it."""
        root = _root_with_model_uhd(tmp_path)
        (root / "pack" / "training_data.csv").write_text("a,b\n1,2\n", encoding="utf-8")

        flat = load_flat_input(root, log=lambda *_: None)

        (uhd,) = [d for d in flat.descriptors if d.type == "uhd"]
        assert [s.name for s in uhd.sidecars] == ["model.bin"]

    def test_the_artifact_resolves_relative_to_its_own_descriptor(self, tmp_path: Path):
        """No root-relative fallback: a typo must not bind to a same-named file."""
        root = _root_with_model_uhd(tmp_path)
        (root / "model.bin").write_bytes(b"decoy")
        (root / "pack" / "model.bin").write_bytes(b"correct")

        flat = load_flat_input(root, log=lambda *_: None)

        (uhd,) = [d for d in flat.descriptors if d.type == "uhd"]
        assert uhd.sidecars[0].source.read_bytes() == b"correct"

    def test_a_shared_artifact_keeps_its_authored_position(self, tmp_path: Path):
        """`../shared/x` keeps its relative position so the path still resolves."""
        root = tmp_path / "src"
        _write_json(
            root / "rocKE" / "attn" / "heuristic.uhd.json",
            _model_uhd("../shared/model.bin"),
        )
        shared = root / "rocKE" / "shared" / "model.bin"
        shared.parent.mkdir(parents=True, exist_ok=True)
        shared.write_bytes(b"shared")

        flat = load_flat_input(root, log=lambda *_: None)

        (uhd,) = [d for d in flat.descriptors if d.type == "uhd"]
        (sidecar,) = uhd.sidecars
        assert sidecar.rel_dir == Path("rocKE/shared")
        assert sidecar.name == "model.bin"


class TestRejection:
    def test_missing_artifact_is_an_error_not_a_warning(self, tmp_path: Path):
        """The runtime drops an engine missing its artifact, so fail at pack time."""
        root = tmp_path / "src"
        _write_json(root / "pack" / "heuristic.uhd.json", _model_uhd("absent.bin"))

        with pytest.raises(HkpPackError, match="payload source not found"):
            load_flat_input(root, log=lambda *_: None)

    def test_an_adapter_that_reads_a_model_must_name_one(self, tmp_path: Path):
        root = tmp_path / "src"
        doc = _model_uhd("model.bin")
        del doc["tree_data"]
        _write_json(root / "pack" / "heuristic.uhd.json", doc)

        with pytest.raises(HkpPackError):
            load_flat_input(root, log=lambda *_: None)

    def test_artifact_outside_the_root_is_rejected(self, tmp_path: Path):
        """Mirrors the hip source check and the runtime's treeRoot bound."""
        root = tmp_path / "src"
        _write_json(
            root / "pack" / "heuristic.uhd.json", _model_uhd("../../outside.bin")
        )
        (tmp_path.parent / "outside.bin").write_bytes(b"outside")

        with pytest.raises(HkpPackError, match="payload escapes the source root"):
            load_flat_input(root, log=lambda *_: None)

    def test_escape_is_checked_before_existence(self, tmp_path: Path):
        root = tmp_path / "src"
        _write_json(
            root / "pack" / "heuristic.uhd.json", _model_uhd("../../nowhere.bin")
        )

        with pytest.raises(HkpPackError, match="payload escapes the source root"):
            load_flat_input(root, log=lambda *_: None)


class TestIntermediateStaging:
    """compile_intermediate mirrors the authored tree before anything is pruned."""

    def test_sidecar_is_mirrored_into_the_intermediate_tree(self, tmp_path: Path):
        root = _root_with_model_uhd(tmp_path)
        flat = load_flat_input(root, log=lambda *_: None)
        inter_dir = tmp_path / "inter" / "gfx942"

        compile_intermediate(
            flat,
            root,
            "gfx942",
            hipcc=None,
            inter_arch_dir=inter_dir,
            log=lambda *_: None,
        )

        assert (inter_dir / "pack" / "heuristic.uhd.json").is_file()
        staged = inter_dir / "pack" / "model.bin"
        assert staged.is_file(), "the artifact the UHD names must ride with it"
        assert staged.read_bytes() == b"HGBM-stub"

    def test_sidecar_keeps_its_authored_subpath(self, tmp_path: Path):
        root = tmp_path / "src"
        _write_json(
            root / "rocKE" / "attn" / "heuristic.uhd.json",
            _model_uhd("../shared/model.bin"),
        )
        shared = root / "rocKE" / "shared" / "model.bin"
        shared.parent.mkdir(parents=True, exist_ok=True)
        shared.write_bytes(b"shared")
        flat = load_flat_input(root, log=lambda *_: None)
        inter_dir = tmp_path / "inter" / "gfx942"

        compile_intermediate(
            flat,
            root,
            "gfx942",
            hipcc=None,
            inter_arch_dir=inter_dir,
            log=lambda *_: None,
        )

        assert (inter_dir / "rocKE" / "shared" / "model.bin").read_bytes() == b"shared"
        assert not (inter_dir / "rocKE" / "attn" / "model.bin").exists()


@pytest.mark.parametrize(
    "missing",
    [
        "objective",
        "features_signature",
        "features_hash",
        "trained_against",
        "tree_data",
    ],
)
def test_feature_models_require_complete_headers_before_packaging(tmp_path, missing):
    root = _root_with_model_uhd(tmp_path)
    doc = _model_uhd("model.bin")
    del doc[missing]
    _write_json(root / "pack" / "heuristic.uhd.json", doc)
    with pytest.raises(HkpPackError):
        load_flat_input(root, log=lambda *_: None)


def test_custom_library_uses_library_and_carries_the_shared_object(tmp_path):
    root = tmp_path / "src"
    doc = _native_uhd()
    del doc["native"]
    doc["adapter"] = "custom_library"
    doc["custom_library"] = {"library": "lib/model.so", "symbol": "score"}
    _write_json(root / "custom.uhd.json", doc)
    (root / "lib").mkdir()
    (root / "lib" / "model.so").write_bytes(b"shared object")
    flat = load_flat_input(root, log=lambda *_: None)
    assert flat.descriptors[0].sidecars[0].source.read_bytes() == b"shared object"
    doc["custom_library"]["artifact"] = doc["custom_library"].pop("library")
    _write_json(root / "custom.uhd.json", doc)
    with pytest.raises(HkpPackError):
        load_flat_input(root, log=lambda *_: None)


def test_explicit_semantic_revision_does_not_change_format_admission(tmp_path):
    root = tmp_path / "src"
    _write_json(
        root / "metadata.kmd.json",
        {
            "version": "1.0",
            "revision": "12.34",
            "id": "799a8b19-8e34-4f74-86d6-b6495a6483f3",
            "name": "metadata",
            "fields": [{"name": "block_size", "type": "int"}],
        },
    )
    flat = load_flat_input(root, log=lambda *_: None)
    assert flat.descriptors[0].doc["revision"] == "12.34"
    doc = _native_uhd()
    doc["version"] = "2.0"
    _write_json(root / "native.uhd.json", doc)
    with pytest.raises(HkpPackError):
        load_flat_input(root, log=lambda *_: None)


_UHD_SCHEMA = "projects/hipdnn/plugin_sdk/schemas/uhd.schema.json"


def _repo_root() -> Path:
    return next(
        parent
        for parent in Path(__file__).resolve().parents
        if (parent / _UHD_SCHEMA).is_file()
    )


def _canonical_uhd_validator():
    """The published schema, which the runtime loader (UhdParser.hpp) mirrors."""
    jsonschema = pytest.importorskip("jsonschema")
    return jsonschema.Draft7Validator(
        json.loads((_repo_root() / _UHD_SCHEMA).read_text(encoding="utf-8"))
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "valid",
        "missing_objective",
        "missing_provenance",
        "legacy_provenance",
        "wrong_body",
        "two_bodies",
        "legacy_derived",
        "bare_feature",
        "empty_feature",
        "invalid_hash",
        "unknown_header",
        "unsupported_format",
        "tflops_metric",
        "time_metric",
        "legacy_units",
        "unregistered_metric",
        "calibrated_without_metric",
        "metric_objective_mismatch",
        "feature_semantics",
        "feature_semantics_zero",
        "feature_semantics_text",
    ],
)
def test_packaging_and_canonical_schema_agree_on_uhd_headers(tmp_path, mutation):
    validator = _canonical_uhd_validator()
    doc = _model_uhd("model.bin")
    valid = mutation in ("valid", "tflops_metric", "time_metric", "feature_semantics")
    if mutation == "missing_objective":
        del doc["objective"]
    elif mutation == "missing_provenance":
        del doc["trained_against"]
    elif mutation == "legacy_provenance":
        doc["trained_against"] = {"ued": "1.0", "kmd": "1.0", "umd": "1.0"}
    elif mutation == "wrong_body":
        doc["table"] = doc.pop("tree_data")
    elif mutation == "two_bodies":
        doc["native"] = {"symbol": "score"}
    elif mutation == "legacy_derived":
        doc["derived"] = {"tiles": {"ceil_div": [100, "$kernel.tile_m"]}}
    elif mutation == "bare_feature":
        doc["features_signature"] = ["kernel.tile_m"]
    elif mutation == "empty_feature":
        doc["features_signature"] = []
    elif mutation == "invalid_hash":
        doc["features_hash"] = "sha256:not-hex"
    elif mutation == "unknown_header":
        doc["unknown"] = True
    elif mutation == "unsupported_format":
        doc["version"] = "1.1"
    elif mutation == "tflops_metric":
        doc["score"] = {"metric": "tflops", "calibrated": True, "transform": "log1p"}
    elif mutation == "time_metric":
        # RFC 0019 §4.4: the time metric requires objective `min`.
        doc["objective"] = "min"
        doc["score"] = {"metric": "time", "calibrated": True}
    elif mutation == "legacy_units":
        doc["score"] = {"units": "tflops", "calibrated": True}
    elif mutation == "unregistered_metric":
        doc["score"] = {"metric": "bandwidth"}
    elif mutation == "calibrated_without_metric":
        doc["score"] = {"calibrated": True}
    elif mutation == "metric_objective_mismatch":
        doc["score"] = {"metric": "time"}
    elif mutation == "feature_semantics":
        doc["trained_against"]["feature_semantics_revision"] = 2
    elif mutation == "feature_semantics_zero":
        doc["trained_against"]["feature_semantics_revision"] = 0
    elif mutation == "feature_semantics_text":
        doc["trained_against"]["feature_semantics_revision"] = "1"
    root = tmp_path / "src"
    _write_json(root / "heuristic.uhd.json", doc)
    (root / "model.bin").write_bytes(b"artifact")
    assert validator.is_valid(doc) == valid
    if valid:
        flat = load_flat_input(root, log=lambda *_: None)
        assert flat.descriptors[0].sidecars[0].source.read_bytes() == b"artifact"
    else:
        with pytest.raises(HkpPackError):
            load_flat_input(root, log=lambda *_: None)


def test_schema_admits_the_extension_namespaces_the_loader_ignores():
    """RFC 0019 §4.1: `x-`, `_` and root `provenance` keys are legal at every level."""
    validator = _canonical_uhd_validator()
    doc = _model_uhd("model.bin")
    doc["x-trained-by"] = "uhd_gen 3.2"
    doc["_internal"] = {"ticket": "SWDEV-000000"}
    doc["provenance"] = {"dataset": "nightly", "rows": 4096}
    doc["score"] = {"metric": "tflops", "x-sampler": "sobol"}
    doc["trained_against"]["_run"] = 17
    doc["tree_data"]["x-bytes"] = 2048

    assert validator.is_valid(doc), [
        error.message for error in validator.iter_errors(doc)
    ]

    # Any other unknown name is still refused, at the root and nested alike.
    stray_root = _model_uhd("model.bin")
    stray_root["notes"] = "not an extension key"
    assert not validator.is_valid(stray_root)
    stray_nested = _model_uhd("model.bin")
    stray_nested["tree_data"]["bytes"] = 2048
    assert not validator.is_valid(stray_nested)


def test_every_in_tree_uhd_conforms_to_the_canonical_schema():
    validator = _canonical_uhd_validator()
    root = _repo_root()
    paths = sorted(
        path
        for tree in ("projects/hipdnn", "dnn-providers")
        for path in (root / tree).rglob("*.uhd.json")
    )
    # A wrong root would make the check vacuous.
    assert paths
    errors = {
        str(path.relative_to(root)): [
            error.message
            for error in validator.iter_errors(
                json.loads(path.read_text(encoding="utf-8"))
            )
        ]
        for path in paths
    }
    assert not {path: found for path, found in errors.items() if found}


def _custom_library_uhd() -> dict:
    doc = _native_uhd()
    del doc["native"]
    doc["adapter"] = "custom_library"
    doc["custom_library"] = {"library": "lib/model.so", "symbol": "score", "config": {}}
    return doc


def _bounded_uhd() -> dict:
    """A model UHD sitting exactly on the parser's int32 code and revision bounds."""
    doc = _model_uhd("model.bin")
    doc["features_signature"] = ["$kernel.layout"]
    doc["categorical_encoding"] = {
        "$kernel.layout": {"nhwc": -(2**31), "nchw": 2**31 - 1}
    }
    doc["trained_against"]["kmd"]["revision"] = "999999999.999999999"
    doc["trained_against"]["feature_semantics_revision"] = 2**63 - 1
    doc["provenance"] = {"dataset": "nightly"}
    return doc


def _set(*path_and_value):
    """A mutation assigning the last argument at the key path given by the others."""
    *path, key, value = path_and_value

    def mutate(doc: dict) -> None:
        target = doc
        for step in path:
            target = target[step]
        target[key] = value

    return mutate


@pytest.mark.parametrize(
    "build, mutate",
    [
        pytest.param(
            _custom_library_uhd,
            _set("custom_library", "config", {"threads": 4}),
            id="custom_library_config_not_empty",
        ),
        pytest.param(
            _bounded_uhd,
            _set("categorical_encoding", "kernel.dtype", {"fp16": 0}),
            id="categorical_field_not_a_reference",
        ),
        pytest.param(
            _bounded_uhd,
            _set("categorical_encoding", "$kernel.layout", "chwn", 2**31),
            id="categorical_code_above_int32",
        ),
        pytest.param(
            _bounded_uhd,
            _set("categorical_encoding", "$kernel.layout", "chwn", -(2**31) - 1),
            id="categorical_code_below_int32",
        ),
        pytest.param(
            _bounded_uhd,
            _set("trained_against", "ued", "revision", "1000000000.0"),
            id="revision_major_over_nine_digits",
        ),
        pytest.param(
            _bounded_uhd,
            _set("trained_against", "ued", "revision", "1.1000000000"),
            id="revision_minor_over_nine_digits",
        ),
        pytest.param(
            _bounded_uhd,
            _set("trained_against", "feature_semantics_revision", 2**63),
            id="feature_semantics_revision_over_int64",
        ),
        pytest.param(
            _bounded_uhd,
            _set("trained_against", "provenance", {"dataset": "nightly"}),
            id="provenance_below_the_root",
        ),
        pytest.param(
            _bounded_uhd,
            _set("tree_data", "provenance", {"dataset": "nightly"}),
            id="provenance_in_the_body",
        ),
    ],
)
def test_schema_refuses_what_the_runtime_parser_refuses(build, mutate):
    validator = _canonical_uhd_validator()
    doc = build()
    assert validator.is_valid(doc), [
        error.message for error in validator.iter_errors(doc)
    ]
    mutate(doc)
    assert not validator.is_valid(doc)


def _static_order_uhd(body: dict) -> dict:
    doc = _native_uhd()
    doc["adapter"] = "static_order"
    del doc["native"]
    doc["static_order"] = body
    return doc


def test_packaging_refuses_static_order_criteria(tmp_path):
    """static_order takes no parameters; UhdParser refuses a declared `order`."""
    root = tmp_path / "src"
    _write_json(root / "order.uhd.json", _static_order_uhd({}))
    load_flat_input(root, log=lambda *_: None)
    _write_json(
        root / "order.uhd.json", _static_order_uhd({"order": ["priority", "id"]})
    )
    with pytest.raises(HkpPackError, match="static_order.order is not supported"):
        load_flat_input(root, log=lambda *_: None)


def test_schema_refuses_static_order_criteria():
    validator = _canonical_uhd_validator()
    empty = _static_order_uhd({"x-note": "extension keys stay legal"})
    assert validator.is_valid(empty), [
        error.message for error in validator.iter_errors(empty)
    ]
    assert not validator.is_valid(_static_order_uhd({"order": ["priority", "id"]}))


def test_schema_ties_the_objective_to_the_score_metric():
    """RFC 0019 §4.4: a metric fixes `objective`; with no metric, either is legal."""
    validator = _canonical_uhd_validator()
    doc = _model_uhd("model.bin")
    doc["score"] = {"metric": "tflops", "calibrated": True}

    doc["objective"] = "max"
    assert validator.is_valid(doc), [
        error.message for error in validator.iter_errors(doc)
    ]
    doc["objective"] = "min"
    assert not validator.is_valid(doc)

    doc["score"] = {"metric": "time", "calibrated": False}
    assert validator.is_valid(doc), [
        error.message for error in validator.iter_errors(doc)
    ]
    doc["objective"] = "max"
    assert not validator.is_valid(doc)

    doc["score"] = {"calibrated": False}
    assert validator.is_valid(doc), [
        error.message for error in validator.iter_errors(doc)
    ]
    doc["objective"] = "min"
    assert validator.is_valid(doc), [
        error.message for error in validator.iter_errors(doc)
    ]


def test_schema_requires_the_signature_a_categorical_encoding_encodes():
    """RFC 0019 §4.1: a `categorical_encoding` requires a `features_signature`."""
    validator = _canonical_uhd_validator()
    doc = _native_uhd()
    doc["categorical_encoding"] = {"$kernel.layout": {"nhwc": 0, "nchw": 1}}
    assert not validator.is_valid(doc)

    doc["features_signature"] = ["$kernel.layout"]
    doc["features_hash"] = "sha256:" + "0" * 16
    doc["trained_against"] = {"selector_revision": "hkp-2026.09"}
    assert validator.is_valid(doc), [
        error.message for error in validator.iter_errors(doc)
    ]


def _metric_uhd(identity: str, metric: str | None) -> dict:
    """A native UHD scoring in @p metric, or a metric-less ordering for None."""
    doc = _native_uhd()
    doc["id"] = identity
    if metric is not None:
        doc["objective"] = {"tflops": "max", "time": "min"}[metric]
        doc["score"] = {"metric": metric, "calibrated": True}
    return doc


_TFLOPS_ID = "433a8b19-8e34-4f74-86d6-b6495a6483f3"
_TIME_ID = "533a8b19-8e34-4f74-86d6-b6495a6483f3"
_OTHER_TFLOPS_ID = "633a8b19-8e34-4f74-86d6-b6495a6483f3"
_ORDERING_ID = "733a8b19-8e34-4f74-86d6-b6495a6483f3"


def _root_with_roles(tmp_path: Path, main_fixture: Path, **roles) -> Path:
    """The main fixture plus per-metric UHDs, with its UED's roles set from @p roles."""
    root = tmp_path / "src"
    shutil.copytree(main_fixture, root)
    for identity, metric in (
        (_TFLOPS_ID, "tflops"),
        (_TIME_ID, "time"),
        (_OTHER_TFLOPS_ID, "tflops"),
        (_ORDERING_ID, None),
    ):
        _write_json(
            root / f"model_{identity[:3]}.uhd.json", _metric_uhd(identity, metric)
        )
    ued_path = root / "pointwise.ued.json"
    ued = json.loads(ued_path.read_text(encoding="utf-8"))
    ued.update(roles)
    _write_json(ued_path, ued)
    return root


def test_all_role_and_arch_models_remain_reachable_when_packaging(
    tmp_path, main_fixture
):
    from hkp_pack.descriptors import reachable_generic_ids

    root = _root_with_roles(
        tmp_path,
        main_fixture,
        predict_engine={"gfx942": [_TFLOPS_ID, _TIME_ID]},
        predict_applicable_kernels={"default": _ORDERING_ID},
    )
    ued_path = root / "pointwise.ued.json"
    ued = json.loads(ued_path.read_text(encoding="utf-8"))
    original = ued["sort_kernel_catalog"]["default"]
    ued["sort_kernel_catalog"]["gfx950"] = [_OTHER_TFLOPS_ID, original]
    _write_json(ued_path, ued)
    flat = load_flat_input(root, log=lambda *_: None)
    retained = reachable_generic_ids(flat, flat.kdps())
    assert {original, _TFLOPS_ID, _TIME_ID, _OTHER_TFLOPS_ID, _ORDERING_ID} <= retained


@pytest.mark.parametrize(
    "roles, message",
    [
        # RFC 0019 §3.1: at most one model per metric for an architecture.
        (
            {"sort_kernel_catalog": {"gfx942": [_TFLOPS_ID, _OTHER_TFLOPS_ID]}},
            "metric 'tflops'",
        ),
        (
            {"predict_engine": {"default": [_TFLOPS_ID, _OTHER_TFLOPS_ID]}},
            "metric 'tflops'",
        ),
        (
            {
                "sort_kernel_catalog": {
                    "default": ["bb58374f-2972-57b1-a9cb-c358bddef2e5", _ORDERING_ID]
                }
            },
            "no metric",
        ),
        ({"predict_engine": {"gfx942": _ORDERING_ID}}, "declares no score.metric"),
    ],
)
def test_packaging_rejects_ambiguous_per_metric_models(
    tmp_path, main_fixture, roles, message
):
    root = _root_with_roles(tmp_path, main_fixture, **roles)
    with pytest.raises(HkpPackError, match=message):
        load_flat_input(root, log=lambda *_: None)


def test_one_metric_per_architecture_is_per_architecture(tmp_path, main_fixture):
    """Metric uniqueness is per (architecture, metric)."""
    root = _root_with_roles(
        tmp_path,
        main_fixture,
        predict_engine={"gfx942": _TFLOPS_ID, "gfx950": [_OTHER_TFLOPS_ID, _TIME_ID]},
    )
    load_flat_input(root, log=lambda *_: None)


@pytest.mark.parametrize(
    "legacy",
    [
        {"heuristic": "233a8b19-8e34-4f74-86d6-b6495a6483f3"},
        {"sort_kernel_catalog": "233a8b19-8e34-4f74-86d6-b6495a6483f3"},
    ],
)
def test_packaging_rejects_legacy_heuristic_spellings(tmp_path, legacy):
    root = tmp_path / "src"
    _write_json(
        root / "engine.ued.json",
        {
            "version": "1.0",
            "id": "699a8b19-8e34-4f74-86d6-b6495a6483f3",
            "name": "test:engine",
            **legacy,
        },
    )
    with pytest.raises(HkpPackError):
        load_flat_input(root, log=lambda *_: None)


@pytest.mark.parametrize(
    "roles, message",
    [
        ({"predict_engine_tflops": {"gfx942": _TFLOPS_ID}}, "unknown fields"),
        ({"predict_engine": {"gfx942": []}}, "nonempty list"),
        (
            {"sort_kernel_catalog": {"gfx942": [_TFLOPS_ID, _TFLOPS_ID.upper()]}},
            "twice",
        ),
        ({"predict_engine": {"gfx942": [_TFLOPS_ID, "not-a-uuid"]}}, "requires a UUID"),
        # A candidate generator has no metric, so no list form (RFC 0019 §3.1).
        (
            {"predict_applicable_kernels": {"default": [_ORDERING_ID]}},
            "requires a UUID",
        ),
    ],
)
def test_packaging_rejects_malformed_role_values(
    tmp_path, main_fixture, roles, message
):
    root = _root_with_roles(tmp_path, main_fixture, **roles)
    with pytest.raises(HkpPackError, match=message):
        load_flat_input(root, log=lambda *_: None)
