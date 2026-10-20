# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU checks for the SDPA reference's accuracy and artifact contracts."""

from __future__ import annotations

import ast
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from sdpa_reference.architectures import ARCHITECTURES, baseline_lock, get_architecture
from sdpa_reference.architectures.gfx942 import CASES
from sdpa_reference.cli import load_bundle, verify_case
from sdpa_reference.contract import (
    INPUT_GENERATOR,
    SCHEMA_VERSION,
    checked_inputs,
    make_inputs,
    Case,
    independent_reference,
)
from reference_common.numeric import (
    ErrorBudget,
    array_digest,
    file_digest,
    payload_digests,
    write_json,
)


def test_cohort_preserves_existing_sdpa_parameterizations():
    source = Path(__file__).with_name("test_attention_dense_gfx942_numeric.py")
    module = ast.parse(source.read_text())
    original = next(
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_COHORT" for t in node.targets)
    )
    assert [
        (c.dtype, c.head_dim, c.query_heads, c.kv_heads, c.persistent, c.causal)
        for c in CASES
    ] == original
    assert len({case.id for case in CASES}) == 8


def test_gfx942_source_adapter_uses_the_current_tuning_dispatch_api():
    import kernels

    target = get_architecture("gfx942")
    library_root = Path(kernels.__file__).resolve().parent.parent
    for case in CASES:
        spec = target.prepare(case, str(library_root))
        assert spec.persistent is case.persistent
        assert spec.dtype == case.dtype
        assert spec.head_size == case.head_dim


@pytest.mark.parametrize("causal", [False, True])
def test_independent_sdpa_matches_analytic_uniform_attention(causal):
    case = Case("fp16", 2, 4, 2, False, causal, batch=2, sequence_length=2)
    q = np.zeros(case.shape, dtype=np.float16)
    k = np.zeros((2, 2, 2, 2), dtype=np.float16)
    v = np.array([[[[2, 4], [6, 8]], [[10, 12], [14, 16]]]] * 2, np.float16)
    actual = independent_reference(case, {"q": q, "k": k, "v": v})
    expected = np.repeat(v.astype(np.float64), 2, axis=2)
    expected[:, 1] = (expected[:, 0] + expected[:, 1]) / 2
    if not causal:
        expected[:, 0] = expected[:, 1]
    np.testing.assert_array_equal(actual, expected)
    assert actual.dtype == np.float64


def _bundle(tmp_path):
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "fixture").write_bytes(b"artifact fixture")
    manifest = {
        "schema": SCHEMA_VERSION,
        "baseline_revision": "a" * 40,
        "files": payload_digests(payload),
        "input_generation": INPUT_GENERATOR,
        "cases": {
            case.id: {
                "case": asdict(case),
                "device_target": "gfx942:sramecc+:xnack-",
                "budget": asdict(ErrorBudget(case.tolerance, 0.001, case.margin)),
                "comparison_limit": ErrorBudget(
                    case.tolerance, 0.001, case.margin
                ).comparison_limit,
                "input_digests": {},
                "output_digest": "not-a-qualified-output",
                "kernel": {},
            }
            for case in CASES
        },
    }
    _lock(tmp_path, manifest)
    return manifest


def _lock(tmp_path, manifest):
    write_json(tmp_path / "manifest.json", manifest)
    write_json(
        tmp_path / "lock.json",
        {
            "schema": manifest["schema"],
            "baseline_revision": "a" * 40,
            "manifest_sha256": file_digest(tmp_path / "manifest.json"),
        },
    )


def test_corrupt_artifact_and_changed_manifest_are_rejected(tmp_path):
    manifest = _bundle(tmp_path)
    assert load_bundle(tmp_path, tmp_path / "lock.json") == manifest
    (tmp_path / "payload/fixture").write_bytes(b"changed artifact")
    with pytest.raises(ValueError, match="payload"):
        load_bundle(tmp_path, tmp_path / "lock.json")
    (tmp_path / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="pinned lock"):
        load_bundle(tmp_path, tmp_path / "lock.json")


@pytest.mark.parametrize(
    "change", ["missing_case", "changed_shape", "looser_tolerance"]
)
def test_even_a_new_lock_cannot_silently_change_the_test_contract(tmp_path, change):
    manifest = _bundle(tmp_path)
    entry = manifest["cases"][CASES[0].id]
    if change == "missing_case":
        del manifest["cases"][CASES[0].id]
    elif change == "changed_shape":
        entry["case"]["sequence_length"] = 128
    else:
        budget = ErrorBudget(0.2, 0.001, CASES[0].margin)
        entry["budget"] = asdict(budget)
        entry["comparison_limit"] = budget.comparison_limit
    _lock(tmp_path, manifest)
    with pytest.raises(ValueError, match="cohort|contract|tolerance"):
        load_bundle(tmp_path, tmp_path / "lock.json")


def test_unknown_old_output_cannot_inherit_a_qualified_bound(tmp_path, monkeypatch):
    from sdpa_reference import cli

    manifest = _bundle(tmp_path)
    manifest["cases"][CASES[0].id]["input_digests"] = {
        name: array_digest(a) for name, a in make_inputs(CASES[0]).items()
    }
    calls = []

    def worker(request, **kwargs):
        calls.append(request["mode"])
        return [np.zeros(CASES[0].shape, np.float16)], {}

    monkeypatch.setattr(cli, "_worker", worker)
    with pytest.raises(AssertionError, match="reference qualification failure"):
        verify_case(CASES[0], bundle=tmp_path, manifest=manifest)
    assert calls == ["replay"]


def test_worker_prefers_selected_library_over_test_packages(tmp_path, monkeypatch):
    from sdpa_reference.cli import _worker

    runner = tmp_path / "tests"
    library = tmp_path / "library"
    platform = tmp_path / "platform"
    platform.mkdir()
    for root, value in ((runner, "test-only"), (library, "production")):
        package = root / "dispatch"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(f"origin = {value!r}\n")
    support = runner / "sdpa_reference"
    support.mkdir()
    (support / "__init__.py").write_text("")
    (support / "worker.py").write_text(
        "import json, sys\n"
        "from pathlib import Path\n"
        "import numpy as np\n"
        "import dispatch\n"
        "assert dispatch.origin == 'production', dispatch.__file__\n"
        "output = Path(sys.argv[2])\n"
        "np.savez(output / 'outputs.npz', out_0=np.ones(1))\n"
        "(output / 'report.json').write_text(json.dumps({'launches': 1}))\n"
    )
    monkeypatch.setenv("PYTHONPATH", str(runner))
    outputs, report = _worker(
        {"mode": "source", "repetitions": 1},
        runner=runner,
        platform=platform,
        library=library,
        work=tmp_path / "work",
    )
    np.testing.assert_array_equal(outputs[0], np.ones(1))
    assert report["launches"] == 1


def test_architecture_enrollment_and_locks():
    assert ARCHITECTURES == ("gfx942",)
    assert get_architecture("gfx942").CASES == CASES
    assert baseline_lock("gfx942").is_file()
    for unsupported in ("gfx950", "gfx1151", "../gfx942"):
        with pytest.raises(ValueError, match="not enrolled"):
            get_architecture(unsupported)


def test_bundle_cannot_be_used_for_a_different_architecture(tmp_path):
    manifest = _bundle(tmp_path)
    manifest["cases"][CASES[0].id]["device_target"] = "gfx950:sramecc+:xnack-"
    _lock(tmp_path, manifest)
    with pytest.raises(ValueError, match="different architecture"):
        load_bundle(tmp_path, tmp_path / "lock.json", architecture="gfx942")


def test_generated_input_mismatch_prevents_any_worker(tmp_path, monkeypatch):
    from sdpa_reference import cli

    manifest = _bundle(tmp_path)
    calls = []
    monkeypatch.setattr(cli, "_worker", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(ValueError, match="generated SDPA input digest mismatch"):
        verify_case(CASES[0], bundle=tmp_path, manifest=manifest)
    assert not calls


def test_changed_generator_contract_is_rejected(tmp_path):
    manifest = _bundle(tmp_path)
    manifest["input_generation"] = dict(INPUT_GENERATOR, seed=1)
    _lock(tmp_path, manifest)
    with pytest.raises(ValueError, match="generator contract"):
        load_bundle(tmp_path, tmp_path / "lock.json")


@pytest.mark.parametrize("fail", [False, True])
def test_generated_inputs_are_temporary_and_shared_by_both_workers(
    tmp_path, monkeypatch, fail
):
    from sdpa_reference import cli

    manifest = _bundle(tmp_path)
    case = CASES[0]
    entry = manifest["cases"][case.id]
    expected = make_inputs(case)
    entry["input_digests"] = {name: array_digest(a) for name, a in expected.items()}
    out = np.zeros(case.shape, dtype="<f2")
    entry["output_digest"] = array_digest(out)
    paths = []

    def worker(request, **kwargs):
        path = Path(request["inputs"])
        assert not path.is_relative_to(tmp_path)
        with np.load(path, allow_pickle=False) as stored:
            assert {name: array_digest(stored[name]) for name in stored.files} == entry[
                "input_digests"
            ]
        paths.append(path)
        if fail:
            raise RuntimeError("deliberate worker failure")
        return [out], {
            "device_target": entry["device_target"],
            "launches": 1,
            "torch_imported": False,
        }

    monkeypatch.setattr(cli, "_worker", worker)
    if fail:
        with pytest.raises(RuntimeError, match="deliberate worker failure"):
            verify_case(
                case,
                bundle=tmp_path,
                manifest=manifest,
                current_root=tmp_path,
                repetitions=1,
            )
        assert len(paths) == 1
    else:
        verify_case(
            case,
            bundle=tmp_path,
            manifest=manifest,
            current_root=tmp_path,
            repetitions=1,
        )
        assert len(paths) == 2 and paths[0] == paths[1]
    assert not paths[0].exists()
    assert not list(tmp_path.rglob("*.npz"))


@pytest.mark.parametrize(
    "dtype,digests",
    [
        (
            "fp16",
            {
                "q": "4e6b5317919934be8a9be759a993227d32ce61bb6d8e6bc2143a4c43cc3f92b3",
                "k": "6e43cc840eecfb9a676cfb287995f12a987029818926cceac0f314fe69da0a98",
                "v": "b1ed43c87a483c5c64a275477883780b51f37640483068c89b929ed1bf7e2e04",
            },
        ),
        (
            "bf16",
            {
                "q": "21af38eaafd51c77969f1545295ba274c684991ec11f7105ae54166848480ba3",
                "k": "55c94dfb4faaa66e9678982ad9c2436cf77d2d302e2675632820d30db047ac29",
                "v": "9067bc6081e97cecee433c3369d2a52a3996fb1ab3a36a6dbc5083f76284d3fb",
            },
        ),
    ],
)
def test_versioned_generator_preserves_stream_order_and_quantized_bits(dtype, digests):
    case = Case(dtype, 2, 4, 2, False, True, sequence_length=2)
    checked_inputs(case, digests)
    # Other generator instances must not change the freshly seeded corpus.
    np.random.default_rng(123).standard_normal(99)
    checked_inputs(case, digests)
