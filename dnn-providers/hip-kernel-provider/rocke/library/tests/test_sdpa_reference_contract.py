# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU checks for the SDPA reference's accuracy and artifact contracts."""

from __future__ import annotations

import ast
import json
import re
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pytest

from sdpa_reference.architectures import ARCHITECTURES, baseline_lock, get_architecture
from sdpa_reference.architectures.gfx942 import CASES
from sdpa_reference.cli import OPERATION, load_bundle, verify_case
from sdpa_reference.contract import (
    INPUT_GENERATOR,
    SCHEMA_VERSION,
    make_inputs,
    Case,
    independent_reference,
)
from reference_common import cli as common_cli
from reference_common.numeric import (
    ErrorBudget,
    array_digest,
    file_digest,
    payload_digests,
    write_json,
)


# The Torch-gated gfx950 dense test has no mask column; it runs its rows causal.
_LEGACY_MASK = {"gfx942": {}, "gfx950": {"causal": True}}


def _legacy_source(architecture):
    return Path(__file__).with_name(f"test_attention_dense_{architecture}_numeric.py")


def _legacy_cases(architecture):
    """Base ``_COHORT`` rows of a Torch-gated test; it runs them at B=1, S=512."""
    module = ast.parse(_legacy_source(architecture).read_text())
    original = next(
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_COHORT" for t in node.targets)
    )
    fixed = dict(_LEGACY_MASK[architecture], batch=1, sequence_length=512)
    return tuple(Case(*row, **fixed) for row in original)


def _assert_covers_legacy_cohort(architecture):
    """Keep every legacy parameterization pinned; pinned-only cases need no edit."""
    cases = get_architecture(architecture).CASES
    missing = [case.id for case in _legacy_cases(architecture) if case not in cases]
    assert not missing, (
        f"{architecture} SDPA cohort no longer covers {missing} from "
        f"{_legacy_source(architecture).name}::_COHORT. Restore them; cohort "
        "changes require requalifying the complete cohort (see "
        "reference_common/docs/gpu-reference-workflow.md)."
    )


@pytest.mark.parametrize("architecture", ["gfx942", "gfx950"])
def test_cohort_preserves_existing_sdpa_parameterizations(architecture):
    _assert_covers_legacy_cohort(architecture)


@pytest.mark.parametrize("architecture", ["gfx942", "gfx950"])
def test_legacy_coverage_allows_pinned_only_cases_and_names_dropped_ones(
    monkeypatch, architecture
):
    target = get_architecture(architecture)
    # Start from exactly the legacy rows so this checks the guard, not the cohort.
    cases = _legacy_cases(architecture)
    monkeypatch.setattr(target, "CASES", (*cases, replace(cases[0], batch=2)))
    _assert_covers_legacy_cohort(architecture)
    monkeypatch.setattr(target, "CASES", cases[1:])
    with pytest.raises(AssertionError, match=re.escape(f"['{cases[0].id}']")):
        _assert_covers_legacy_cohort(architecture)


def test_gfx942_source_adapter_uses_the_current_tuning_dispatch_api():
    import kernels

    target = get_architecture("gfx942")
    library_root = Path(kernels.__file__).resolve().parent.parent
    for case in CASES:
        spec = target.prepare(case, str(library_root))
        assert spec.persistent is case.persistent
        assert spec.dtype == case.dtype
        assert spec.head_size == case.head_dim


def test_gfx942_source_adapter_supports_published_dispatch_api(monkeypatch):
    from dataclasses import dataclass

    import dispatch.attention as attention
    import kernels

    # Published source snapshots predate the tuning API and select persistence
    # through the request. Exercise both sides of that request contract.
    @dataclass(frozen=True)
    class LegacyRequest(attention.AttentionRequest):
        dense_persistent: str = "auto"

    tuning = attention.tuning_spec_with_knobs

    def legacy_factory(request):
        assert request.dense_persistent in ("on", "off")
        return tuning(
            request,
            "gfx942_dense",
            {"persistent": request.dense_persistent == "on"},
        ).kernel_spec

    monkeypatch.setattr(attention, "AttentionRequest", LegacyRequest)
    monkeypatch.setattr(
        attention, "dense_spec_for_request", legacy_factory, raising=False
    )
    monkeypatch.delattr(attention, "tuning_spec_with_knobs")
    root = str(Path(kernels.__file__).resolve().parent.parent)
    for case in CASES:
        spec = get_architecture("gfx942").prepare(case, root)
        assert spec.persistent is case.persistent
        assert spec.dtype == case.dtype
        assert spec.head_size == case.head_dim
        assert (spec.num_query_heads, spec.num_kv_heads) == (
            case.query_heads,
            case.kv_heads,
        )
        assert spec.causal is case.causal


def test_gfx942_source_adapter_reports_a_renamed_current_api(monkeypatch):
    """A renamed current API fails on its own name, not in the published path."""
    import kernels
    from dispatch import attention

    # A symbol rename moves the function but leaves this adapter's call behind.
    monkeypatch.setattr(
        attention,
        "attention_spec_with_knobs",
        attention.tuning_spec_with_knobs,
        raising=False,
    )
    monkeypatch.delattr(attention, "tuning_spec_with_knobs")
    root = str(Path(kernels.__file__).resolve().parent.parent)
    with pytest.raises(AttributeError, match="no attribute 'tuning_spec_with_knobs'"):
        get_architecture("gfx942").prepare(CASES[0], root)


@pytest.mark.parametrize("case", get_architecture("gfx950").CASES, ids=lambda c: c.id)
@pytest.mark.parametrize("causal", [False, True], ids=["full", "causal"])
def test_gfx950_source_adapter_preserves_dispatch_and_replay_abi(case, causal):
    import kernels
    from kernels.gfx950.attention_dense import attention_dense_signature

    target = get_architecture("gfx950")
    library_root = Path(kernels.__file__).resolve().parent.parent
    # Exercise both mask branches without expanding the pinned GPU cohort.
    case = replace(case, causal=causal)
    spec = target.prepare(case, str(library_root))
    assert spec.persistent is case.persistent
    assert spec.wide_lds_dma is (
        case.persistent and case.head_dim == 128 and case.causal
    )
    assert spec.dtype == case.dtype
    assert spec.head_size == case.head_dim
    assert (spec.num_query_heads, spec.num_kv_heads) == (
        case.query_heads,
        case.kv_heads,
    )
    assert spec.causal is case.causal
    assert not spec.use_sinks and not spec.sliding_window
    assert spec.runtime_shape is (not case.persistent), (
        f"{case.id} selected runtime_shape={spec.runtime_shape}. Dense replay "
        "binds shape scalars only for runtime-shape grid kernels; persistent "
        "kernels bake the shape. Use a sequence length the aligned (non-ragged) "
        "grid kernel accepts, or extend the dense replay ABI first."
    )
    # Replay supplies shape scalars only for the runtime-shape members of this cohort.
    signature = attention_dense_signature(spec)
    names = [arg["name"] for arg in signature]
    expected = ["q_ptr", "k_ptr", "v_ptr", "o_ptr", "scale"]
    if not case.persistent:
        expected += ["batch", "seqlen_q", "seqlen_kv"]
    assert names == expected


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


def _bundle(tmp_path, architecture="gfx942"):
    target = get_architecture(architecture)
    payload = tmp_path / "payload"
    payload.mkdir()
    (payload / "fixture").write_bytes(b"artifact fixture")
    manifest = {
        "schema": SCHEMA_VERSION,
        "budget_policy": "max-reference-bound-v1",
        "baseline_revision": "a" * 40,
        "files": payload_digests(payload),
        "input_generation": INPUT_GENERATOR,
        "cases": {
            case.id: {
                "case": asdict(case),
                "device_target": f"{architecture}:sramecc+:xnack-",
                "budget": asdict(ErrorBudget(case.tolerance, 0.001)),
                "comparison_limit": ErrorBudget(case.tolerance, 0.001).comparison_limit,
                "input_digests": {},
                "output_digest": "not-a-qualified-output",
                "kernel": {
                    "compiler_target": getattr(target, "COMPILER_TARGET", architecture)
                },
            }
            for case in target.CASES
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
        budget = ErrorBudget(0.2, 0.001)
        entry["budget"] = asdict(budget)
        entry["comparison_limit"] = budget.comparison_limit
    _lock(tmp_path, manifest)
    with pytest.raises(ValueError, match="cohort|contract|tolerance"):
        load_bundle(tmp_path, tmp_path / "lock.json")


def test_case_ids_name_batch_and_sequence_length():
    case = CASES[0]
    assert case.id == "fp16-d128-h16-kv4-default-causal-b1-s512"
    variants = (case, replace(case, batch=2), replace(case, sequence_length=1024))
    assert len({variant.id for variant in variants}) == len(variants)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_every_enrolled_case_has_a_distinct_id(architecture):
    from reference_common.cases import index_cases

    # Adapters index ids on import, so a collision fails instead of dropping a case.
    target = get_architecture(architecture)
    assert list(target.CASE_BY_ID.values()) == list(target.CASES)
    with pytest.raises(ValueError, match="duplicate SDPA case id"):
        index_cases((*target.CASES, target.CASES[0]), label="SDPA")


@pytest.mark.parametrize("change", ["missing", "extra", "duplicate"])
def test_bundle_must_record_the_enrolled_cohort_one_to_one(tmp_path, change):
    manifest = _bundle(tmp_path)
    case = CASES[0]
    entry = manifest["cases"][case.id]
    if change == "missing":
        del manifest["cases"][case.id]
        expected = rf"missing: {case.id} \(.*sequence_length=512\); extra: none"
    elif change == "extra":
        variant = asdict(replace(case, sequence_length=1024))
        manifest["cases"]["variant"] = dict(entry, case=variant)
        expected = r"missing: none; extra: variant \(.*sequence_length=1024\)"
    else:
        manifest["cases"]["copy"] = entry
        expected = rf"records case {case.id} more than once: copy, {case.id}$"
    _lock(tmp_path, manifest)
    with pytest.raises(ValueError, match=expected):
        load_bundle(tmp_path, tmp_path / "lock.json")


def _published_id(case):
    """The id format of the published bundles, which omits batch and length."""
    grid = "persistent" if case.persistent else "default"
    mask = "causal" if case.causal else "full"
    return (
        f"{case.dtype}-d{case.head_dim}-h{case.query_heads}"
        f"-kv{case.kv_heads}-{grid}-{mask}"
    )


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_bundle_keyed_by_published_ids_loads_and_replays(
    tmp_path, monkeypatch, architecture
):
    from sdpa_reference import cli

    manifest = _bundle(tmp_path, architecture)
    cases = get_architecture(architecture).CASES
    # Published ids omit batch and length. Cases that would share one were never
    # published under it, so they keep their full ids.
    published = [_published_id(case) for case in cases]
    keys = {
        case.id: key if published.count(key) == 1 else case.id
        for case, key in zip(cases, published)
    }
    manifest["cases"] = {keys[case.id]: manifest["cases"][case.id] for case in cases}
    _lock(tmp_path, manifest)
    assert (
        load_bundle(tmp_path, tmp_path / "lock.json", architecture=architecture)
        == manifest
    )
    case = next(c for c in reversed(cases) if keys[c.id] != c.id)
    key = keys[case.id]
    assert cli.manifest_key(manifest, case, architecture=architecture) == key
    entry = manifest["cases"][key]
    entry["input_digests"] = {
        name: array_digest(a) for name, a in make_inputs(case).items()
    }
    out = np.zeros(case.shape, dtype="<f2" if case.dtype == "fp16" else "<u2")
    entry["output_digest"] = array_digest(out)
    requests = []

    def worker(request, **kwargs):
        requests.append(request)
        return [out], {
            "device_target": entry["device_target"],
            "launches": 1,
            "torch_imported": False,
        }

    monkeypatch.setattr(common_cli, "run_worker", worker)
    verify_case(
        case,
        bundle=tmp_path,
        manifest=manifest,
        current_root=tmp_path,
        repetitions=1,
        architecture=architecture,
    )
    # Frozen runners receive the request protocol they were qualified with.
    replay, source = requests
    assert source == {
        "architecture": architecture,
        "case": asdict(case),
        "input_digests": entry["input_digests"],
        "repetitions": 1,
        "inputs": source["inputs"],
        "mode": "source",
    }
    assert replay == dict(
        source,
        mode="replay",
        kernel=entry["kernel"],
        hsaco=str(tmp_path / "payload/cases" / key / "kernel.hsaco"),
    )


def test_unknown_old_output_cannot_inherit_a_qualified_bound(tmp_path, monkeypatch):
    manifest = _bundle(tmp_path)
    manifest["cases"][CASES[0].id]["input_digests"] = {
        name: array_digest(a) for name, a in make_inputs(CASES[0]).items()
    }
    calls = []

    def worker(request, **kwargs):
        calls.append(request["mode"])
        return [np.zeros(CASES[0].shape, np.float16)], {}

    monkeypatch.setattr(common_cli, "run_worker", worker)
    with pytest.raises(AssertionError, match="reference qualification failure"):
        verify_case(CASES[0], bundle=tmp_path, manifest=manifest)
    assert calls == ["replay"]


def test_worker_prefers_selected_library_over_test_packages(tmp_path, monkeypatch):
    from reference_common.runner import run_worker

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
    outputs, report = run_worker(
        {"mode": "source", "repetitions": 1},
        runner=runner,
        platform=platform,
        library=library,
        work=tmp_path / "work",
        module=OPERATION.worker_module,
    )
    np.testing.assert_array_equal(outputs[0], np.ones(1))
    assert report["launches"] == 1


def test_architecture_enrollment_and_locks():
    assert ARCHITECTURES == ("gfx942", "gfx950", "gfx1151", "gfx1200", "gfx1201")
    assert get_architecture("gfx942").CASES == CASES
    assert baseline_lock("gfx942").is_file()
    for unsupported in ("gfx1100", "gfx1250", "../gfx942"):
        with pytest.raises(ValueError, match="not enrolled"):
            get_architecture(unsupported)


def test_bundle_cannot_be_used_for_a_different_architecture(tmp_path):
    manifest = _bundle(tmp_path)
    manifest["cases"][CASES[0].id]["device_target"] = "gfx950:sramecc+:xnack-"
    _lock(tmp_path, manifest)
    with pytest.raises(ValueError, match="different architecture"):
        load_bundle(tmp_path, tmp_path / "lock.json", architecture="gfx942")


@pytest.mark.parametrize("architecture", ["gfx1200", "gfx1201"])
@pytest.mark.parametrize(
    "change",
    [None, "other-processor", "generic-device", "specific-code", "missing-target"],
)
def test_generic_wmma_bundle_keeps_processor_qualification(
    tmp_path, architecture, change
):
    """Generic compilation must not transfer a certificate between processors."""
    manifest = _bundle(tmp_path, architecture)
    entry = next(iter(manifest["cases"].values()))
    if change == "other-processor":
        entry["device_target"] = "gfx1201" if architecture == "gfx1200" else "gfx1200"
    elif change == "generic-device":
        entry["device_target"] = "gfx12-generic"
    elif change == "specific-code":
        entry["kernel"]["compiler_target"] = architecture
    elif change == "missing-target":
        del entry["kernel"]["compiler_target"]
    _lock(tmp_path, manifest)
    if change is None:
        assert (
            load_bundle(tmp_path, tmp_path / "lock.json", architecture=architecture)
            == manifest
        )
    else:
        with pytest.raises(
            ValueError, match="different architecture|different compiler target"
        ):
            load_bundle(tmp_path, tmp_path / "lock.json", architecture=architecture)


@pytest.mark.parametrize("architecture", ["gfx1200", "gfx1201"])
@pytest.mark.parametrize("mismatch", ["qualification", "current-features"])
def test_generic_wmma_replay_requires_qualified_processor(
    tmp_path, monkeypatch, architecture, mismatch
):
    manifest = _bundle(tmp_path, architecture)
    case = get_architecture(architecture).CASES[0]
    entry = manifest["cases"][case.id]
    entry["input_digests"] = {
        name: array_digest(a) for name, a in make_inputs(case).items()
    }
    out = np.zeros(case.shape, dtype="<f2")
    entry["output_digest"] = array_digest(out)

    def worker(request, **kwargs):
        device = entry["device_target"]
        if mismatch == "qualification":
            device = "gfx1201" if architecture == "gfx1200" else "gfx1200"
        elif request["mode"] == "source":
            device = architecture + ":xnack+"
        return [out], {"device_target": device, "launches": 1, "torch_imported": False}

    monkeypatch.setattr(common_cli, "run_worker", worker)
    with pytest.raises(ValueError, match="qualified target|different target features"):
        verify_case(
            case,
            bundle=tmp_path,
            manifest=manifest,
            architecture=architecture,
            current_root=tmp_path,
        )


def test_generated_input_mismatch_prevents_any_worker(tmp_path, monkeypatch):
    manifest = _bundle(tmp_path)
    calls = []
    monkeypatch.setattr(
        common_cli, "run_worker", lambda *args, **kwargs: calls.append(args)
    )
    with pytest.raises(ValueError, match="generated SDPA input digest mismatch"):
        verify_case(CASES[0], bundle=tmp_path, manifest=manifest)
    assert not calls


@pytest.mark.parametrize("declared", ["sdpa", "conv-fwd"])
def test_declared_operation_must_be_sdpa(tmp_path, declared):
    # Published SDPA manifests omit the field; one that declares it must agree.
    manifest = _bundle(tmp_path)
    assert "operation" not in manifest
    manifest["operation"] = declared
    _lock(tmp_path, manifest)
    if declared == "sdpa":
        assert load_bundle(tmp_path, tmp_path / "lock.json") == manifest
    else:
        with pytest.raises(ValueError, match="wrong reference operation"):
            load_bundle(tmp_path, tmp_path / "lock.json")


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

    monkeypatch.setattr(common_cli, "run_worker", worker)
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
    OPERATION.checked_inputs(case, digests)
    # Other generator instances must not change the freshly seeded corpus.
    np.random.default_rng(123).standard_normal(99)
    OPERATION.checked_inputs(case, digests)


def _wmma_grid(prepared):
    """One CTA per query tile, query head, and batch of the prepared case (flat 1-D grid)."""
    case = prepared.case
    tiles = -(-case.sequence_length // prepared.spec.bm)
    return (case.batch * tiles * case.query_heads, 1, 1)


@pytest.mark.parametrize(
    "architecture,case",
    [
        (architecture, case)
        for architecture in ("gfx1151", "gfx1200", "gfx1201")
        for case in get_architecture(architecture).CASES
    ],
    ids=lambda value: value.id if isinstance(value, Case) else value,
)
def test_wmma_reference_matches_production_kernel_abi(architecture, case):
    import kernels
    from sdpa_reference.implementations import wmma

    target = get_architecture(architecture)
    root = Path(kernels.__file__).resolve().parent.parent
    prepared = target.prepare(case, str(root))
    assert prepared.abi == "wmma-strided-v3"
    kernel = wmma.build(prepared)
    assert [(p.name, p.type.name) for p in kernel.params] == [
        (arg["name"], arg["type"].replace(" ", "")) for arg in wmma.signature(prepared)
    ]
    assert wmma.grid(prepared) == _wmma_grid(prepared)
    assert prepared.spec.threads == 2 * prepared.spec.bm
    assert prepared.spec.head_dim == case.head_dim
    assert prepared.spec.mask == ("causal" if case.causal else "none")
    assert prepared.spec.layout == "dense"
    values = wmma.scalars(case, case.scale)
    assert values["seqlen_q"] == values["seqlen_k"] == case.sequence_length
    assert values["num_query_heads"] == case.query_heads
    assert values["num_kv_heads"] == case.kv_heads
    assert values["bottom_right"] == values["write_lse"] == 0
    # The "causal" class takes the diagonal from the runtime bounds; the unmasked one
    # ignores them.
    assert values["window_left"] == -1
    assert values["window_right"] == (0 if case.causal else -1)
    assert values["batch"] == case.batch
    assert values["num_v_heads"] == case.kv_heads
    assert values["head_dim_q"] == values["head_dim_v"] == case.head_dim
    assert values["stride_lse_batch"] == 0
    assert values["stride_lse_token"] == values["stride_lse_head"] == 0
    assert values["scale_log2"] == pytest.approx(case.scale * np.log2(np.e))
    for name, heads in (
        ("q", case.query_heads),
        ("k", case.kv_heads),
        ("v", case.kv_heads),
        ("o", case.query_heads),
    ):
        assert values[f"stride_{name}_batch"] == (
            case.sequence_length * heads * case.head_dim
        )
        assert values[f"stride_{name}_token"] == heads * case.head_dim
        assert values[f"stride_{name}_head"] == case.head_dim
    buffers = {name: object() for name in ("q", "k", "v", "out")}
    bound = wmma.replay_values({"abi": prepared.abi, "scalars": values}, buffers)
    assert bound["Q"] is buffers["q"]
    assert bound["K"] is buffers["k"]
    assert bound["V"] is buffers["v"]
    assert bound["O"] is buffers["out"]
    assert bound["LSE"] == 0
    assert {arg["name"] for arg in wmma.signature(prepared)} == set(bound)
    with pytest.raises(RuntimeError, match="unlaunched"):
        target.exported_kernel(prepared)


@pytest.mark.parametrize("architecture", ["gfx1151", "gfx1200", "gfx1201"])
def test_wmma_launch_and_export_preserve_runtime_arguments(architecture, monkeypatch):
    from types import SimpleNamespace

    import kernels
    import rocke.helpers
    import rocke.runtime
    from rocke.runtime.packing import pack_args
    from sdpa_reference.implementations import wmma
    from sdpa_reference.worker import replay_values

    # Batched, causal GQA exercises the distinct K/V strides.
    case = next(
        (
            c
            for c in reversed(wmma.CASES)
            if c.batch > 1 and c.causal and c.kv_heads < c.query_heads
        ),
        None,
    )
    assert case, "the WMMA cohort needs a batched, causal GQA case"
    root = Path(kernels.__file__).resolve().parent.parent
    target = get_architecture(architecture)
    compiler_target = "gfx1151" if architecture == "gfx1151" else "gfx12-generic"
    prepared = target.prepare(case, str(root))
    compiled = []
    launches = []

    def compile_kernel(kernel, **kwargs):
        assert kwargs["arch"] == compiler_target
        compiled.append(kernel)
        return SimpleNamespace(hsaco=b"code-object", kernel_name=kernel.name)

    class Launcher:
        def __init__(self, *, hsaco, kernel_name, signature):
            self._hsaco = hsaco
            self.kernel_name = kernel_name
            self.signature = signature

        def __call__(self, values, *, config):
            launches.append((pack_args(self.signature, values), config))

    monkeypatch.setattr(rocke.helpers, "compile_kernel", compile_kernel)
    monkeypatch.setattr(rocke.runtime, "KernelLauncher", Launcher)
    buffers = dict(q=0x1000, k=0x2000, v=0x3000, out=0x4000)
    for _ in range(2):
        wmma.launch(prepared, buffers, case.scale)
    code, metadata = wmma.exported_kernel(prepared)
    assert code == b"code-object"
    assert metadata["abi"] == "wmma-strided-v3"
    assert metadata["compiler_target"] == compiler_target
    assert len(compiled) == 1 and len(launches) == 2
    expected = pack_args(metadata["signature"], replay_values(metadata, buffers, case))
    for arguments, config in launches:
        assert arguments == expected
        assert config.grid == metadata["grid"] == _wmma_grid(prepared)
        assert config.block == metadata["block"] == (prepared.spec.threads, 1, 1)


def test_wmma_reference_cohort_and_rejections():
    from dataclasses import replace
    import kernels
    from sdpa_reference.implementations import wmma

    assert len(wmma.CASES) == len({case.id for case in wmma.CASES})
    required = {
        (d, kv, causal) for d in (64, 128) for kv in (1, 4) for causal in (False, True)
    }
    missing = required - {(c.head_dim, c.kv_heads, c.causal) for c in wmma.CASES}
    assert not missing, (
        f"WMMA cohort no longer covers (head_dim, kv_heads, causal) {sorted(missing)}. "
        "gfx1151, gfx1200, and gfx1201 share it; cohort changes require "
        "requalifying each of them (see "
        "reference_common/docs/gpu-reference-workflow.md)."
    )
    root = str(Path(kernels.__file__).resolve().parent.parent)
    for architecture in ("gfx1151", "gfx1200", "gfx1201"):
        target = get_architecture(architecture)
        for changed in (
            replace(wmma.CASES[0], dtype="bf16"),
            replace(wmma.CASES[0], persistent=True),
        ):
            with pytest.raises(ValueError, match="outside"):
                target.prepare(changed, root)
        with pytest.raises(RuntimeError, match="outside the selected library"):
            target.prepare(wmma.CASES[0], str(Path(root) / "wrong-tree"))


@pytest.mark.parametrize("architecture", ["gfx942", "gfx950"])
@pytest.mark.parametrize("runtime_shape", [False, True])
def test_legacy_dense_replay_arguments(architecture, runtime_shape):
    from sdpa_reference.worker import replay_values

    case = get_architecture(architecture).CASES[0]
    buffers = {name: object() for name in ("q", "k", "v", "out")}
    expected = dict(
        q_ptr=buffers["q"],
        k_ptr=buffers["k"],
        v_ptr=buffers["v"],
        o_ptr=buffers["out"],
        scale=case.scale,
    )
    if runtime_shape:
        expected.update(
            batch=case.batch,
            seqlen_q=case.sequence_length,
            seqlen_kv=case.sequence_length,
        )
    assert replay_values({"runtime_shape": runtime_shape}, buffers, case) == expected
    assert (
        replay_values(
            {"abi": "dense-v1", "runtime_shape": runtime_shape}, buffers, case
        )
        == expected
    )


@pytest.mark.parametrize(
    "abi", ["wmma-strided-v1", "wmma-strided-v2", "wmma-strided-v3"]
)
def test_worker_wmma_replay_uses_frozen_scalars_and_rejects_unknown_abi(abi):
    from sdpa_reference.worker import replay_values
    from sdpa_reference.implementations import wmma

    case = wmma.CASES[0]
    buffers = {name: object() for name in ("q", "k", "v", "out")}
    frozen = wmma.scalars(case, case.scale / 2, abi=abi)
    if abi == "wmma-strided-v1":
        assert set(frozen) == {
            "scale_log2",
            "seqlen_q",
            "seqlen_k",
            *(
                f"stride_{tensor}_{axis}"
                for tensor in ("q", "k", "v", "o")
                for axis in ("token", "head")
            ),
        }
    metadata = {"abi": abi, "scalars": frozen}
    result = replay_values(metadata, buffers, case)
    expected = dict(
        frozen, Q=buffers["q"], K=buffers["k"], V=buffers["v"], O=buffers["out"]
    )
    if abi in ("wmma-strided-v2", "wmma-strided-v3"):
        expected["LSE"] = 0
    assert result == expected
    assert metadata == {"abi": abi, "scalars": frozen}
    with pytest.raises(ValueError, match="unsupported SDPA replay ABI"):
        replay_values({"abi": "unknown"}, buffers, case)


@pytest.mark.parametrize("architecture", ARCHITECTURES)
def test_reference_adapter_rejects_wrong_source_tree(architecture):
    import kernels

    target = get_architecture(architecture)
    wrong_root = Path(kernels.__file__).resolve().parent / "wrong-tree"
    with pytest.raises(RuntimeError, match="outside the selected library"):
        target.prepare(target.CASES[0], str(wrong_root))


@pytest.mark.parametrize("architecture", ["gfx942", "gfx950"])
def test_dense_request_preserves_tensor_and_mask_contract(architecture):
    from sdpa_reference.architectures.common import attention_request

    for case in get_architecture(architecture).CASES:
        request = attention_request(case, architecture)
        assert request.arch == architecture
        assert request.batch == case.batch
        assert request.nhead_q == case.query_heads
        assert request.nhead_k == case.kv_heads
        assert request.seqlen_q == request.seqlen_k == case.sequence_length
        assert request.hdim_q == request.hdim_v == case.head_dim
        assert request.dtype == case.dtype
        assert request.mask_type == (1 if case.causal else 0)


@pytest.mark.parametrize("architecture", ["gfx942", "gfx950"])
def test_dense_export_uses_selected_production_launcher(architecture, monkeypatch):
    import importlib
    from types import SimpleNamespace

    import kernels
    from kernels.common.attention_dense_spec import attention_dense_cache_key

    target = get_architecture(architecture)
    module = importlib.import_module(f"kernels.{architecture}.attention_dense")
    root = str(Path(kernels.__file__).resolve().parent.parent)
    for case in target.CASES:
        spec = target.prepare(case, root)
        launcher = SimpleNamespace(
            _hsaco=b"executed-code-object",
            kernel_name="executed_kernel",
            signature=module.attention_dense_signature(spec),
        )
        monkeypatch.setattr(
            module,
            "_DENSE_LAUNCHER_CACHE",
            {attention_dense_cache_key(spec, arch=architecture): launcher},
        )
        code, metadata = target.exported_kernel(spec)
        assert code is launcher._hsaco
        if architecture == "gfx942":
            scalars = module.attention_dense_runtime_args(spec)
        else:
            scalars = (
                dict(
                    batch=case.batch,
                    seqlen_q=case.sequence_length,
                    seqlen_kv=case.sequence_length,
                )
                if spec.runtime_shape
                else {}
            )
        assert metadata == {
            "abi": "dense-v2",
            "name": launcher.kernel_name,
            "signature": launcher.signature,
            "grid": module.attention_dense_grid(spec),
            "block": module.attention_dense_block(spec),
            "runtime_shape": spec.runtime_shape,
            "scalars": scalars,
        }


@pytest.mark.parametrize(
    "architecture,case",
    [
        (arch, case)
        for arch in ("gfx942", "gfx950")
        for case in get_architecture(arch).CASES
    ],
    ids=lambda value: value.id if isinstance(value, Case) else value,
)
def test_dense_export_replays_production_launch_arguments(
    architecture, case, monkeypatch
):
    import importlib

    import kernels
    from kernels.common.attention_dense_spec import attention_dense_cache_key
    from rocke.runtime.packing import pack_args
    from sdpa_reference.worker import replay_values

    target = get_architecture(architecture)
    module = importlib.import_module(f"kernels.{architecture}.attention_dense")
    spec = target.prepare(case, str(Path(kernels.__file__).resolve().parent.parent))
    buffers = dict(q=0x1000, k=0x2000, v=0x3000, out=0x4000)
    launched = []

    class RecordingLauncher:
        _hsaco = b"executed-code-object"
        kernel_name = "executed_kernel"
        signature = module.attention_dense_signature(spec)

        def __call__(self, values, *, config):
            launched.append(pack_args(self.signature, values))

    launcher = RecordingLauncher()
    monkeypatch.setattr(
        module,
        "_DENSE_LAUNCHER_CACHE",
        {attention_dense_cache_key(spec, arch=architecture): launcher},
    )
    target.launch(spec, buffers, case.scale)
    _, metadata = target.exported_kernel(spec)
    metadata = json.loads(json.dumps(metadata))
    replayed = replay_values(metadata, buffers, case)
    assert launched == [pack_args(metadata["signature"], replayed)]
    assert metadata["abi"] == "dense-v2"
    assert set(replayed) == {arg["name"] for arg in metadata["signature"]}
    # Runtime dimensions must come from the certificate, not the replay caller.
    assert (
        replay_values(metadata, buffers, replace(case, batch=case.batch + 1))
        == replayed
    )
    # An ABI expansion must be frozen completely or fail qualification.
    launcher.signature = [
        *launcher.signature,
        {"name": "unbound_scalar", "type": "i32"},
    ]
    with pytest.raises(ValueError, match="scalar.*signature"):
        target.exported_kernel(spec)


def test_frozen_runner_imports_all_adapters_without_source_checkout(tmp_path):
    import subprocess
    import sys

    runner = tmp_path / "runner"
    OPERATION.freeze_runner(runner)
    assert not list(runner.rglob("baseline_lock.json"))
    assert not list(runner.rglob("__pycache__"))
    for name in ("pinned_tests.py", "pytest_support.py"):
        assert not (runner / "reference_common" / name).exists()
    # Isolated mode ignores inherited PYTHONPATH and the source working directory.
    # Only the frozen runner is added; production modules must remain lazy.
    script = """
import importlib
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
from sdpa_reference.architectures import ARCHITECTURES, get_architecture
from sdpa_reference import worker

assert ARCHITECTURES == tuple(sys.argv[2:])
for architecture in ARCHITECTURES:
    adapter = get_architecture(architecture)
    assert adapter.CASES
    assert callable(adapter.prepare)
    assert callable(adapter.launch)
    assert callable(adapter.exported_kernel)
for name in ("dense", "wmma"):
    importlib.import_module("sdpa_reference.implementations." + name)
for name, module in tuple(sys.modules.items()):
    if name.split(".")[0] in ("sdpa_reference", "reference_common"):
        assert Path(module.__file__).resolve().is_relative_to(root), name
assert "reference_common.worker" in sys.modules
assert "kernels" not in sys.modules
assert "dispatch" not in sys.modules
assert "torch" not in sys.modules
assert "pytest" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(runner), *ARCHITECTURES],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
