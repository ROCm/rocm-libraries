# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU checks for the SDPA reference's accuracy and artifact contracts."""

from __future__ import annotations

import ast
import json
from dataclasses import asdict, replace
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


def test_gfx950_cohort_preserves_existing_sdpa_parameterizations():
    source = Path(__file__).with_name("test_attention_dense_gfx950_numeric.py")
    module = ast.parse(source.read_text())
    original = next(
        ast.literal_eval(node.value)
        for node in module.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_COHORT" for t in node.targets)
    )
    cases = get_architecture("gfx950").CASES
    assert [
        (c.dtype, c.head_dim, c.query_heads, c.kv_heads, c.persistent) for c in cases
    ] == original
    assert len({case.id for case in cases}) == 7
    assert all(c.causal and c.batch == 1 and c.sequence_length == 512 for c in cases)


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
    assert spec.runtime_shape is (not case.persistent)
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
        "baseline_revision": "a" * 40,
        "files": payload_digests(payload),
        "input_generation": INPUT_GENERATOR,
        "cases": {
            case.id: {
                "case": asdict(case),
                "device_target": f"{architecture}:sramecc+:xnack-",
                "budget": asdict(ErrorBudget(case.tolerance, 0.001, case.margin)),
                "comparison_limit": ErrorBudget(
                    case.tolerance, 0.001, case.margin
                ).comparison_limit,
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
    from sdpa_reference import cli

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

    monkeypatch.setattr(cli, "_worker", worker)
    with pytest.raises(ValueError, match="qualified target|different target features"):
        verify_case(
            case,
            bundle=tmp_path,
            manifest=manifest,
            architecture=architecture,
            current_root=tmp_path,
        )


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


@pytest.mark.parametrize("architecture", ["gfx1151", "gfx1200", "gfx1201"])
@pytest.mark.parametrize("case_index", range(8))
def test_wmma_reference_matches_production_kernel_abi(architecture, case_index):
    import kernels
    from sdpa_reference.implementations import wmma

    target = get_architecture(architecture)
    case = target.CASES[case_index]
    root = Path(kernels.__file__).resolve().parent.parent
    prepared = target.prepare(case, str(root))
    assert prepared.abi == "wmma-strided-v2"
    kernel = wmma.build(prepared)
    assert [(p.name, p.type.name) for p in kernel.params] == [
        (arg["name"], arg["type"].replace(" ", "")) for arg in wmma.signature(prepared)
    ]
    assert wmma.grid(prepared) == (4, 4, 2)
    assert prepared.spec.block_size == 32
    assert prepared.spec.head_size == case.head_dim
    assert prepared.spec.mask_mode == ("causal" if case.causal else "none")
    assert prepared.spec.layout == "dense"
    values = wmma.scalars(case, case.scale)
    assert values["seqlen_q"] == values["seqlen_k"] == 64
    assert values["num_query_heads"] == 4
    assert values["num_kv_heads"] == case.kv_heads
    assert values["bottom_right"] == values["write_lse"] == 0
    assert values["window_left"] == values["window_right"] == -1
    assert values["stride_lse_batch"] == 0
    assert values["stride_lse_token"] == values["stride_lse_head"] == 0
    assert values["scale_log2"] == pytest.approx(case.scale * np.log2(np.e))
    for name, heads in (("q", 4), ("k", case.kv_heads), ("v", case.kv_heads), ("o", 4)):
        assert values[f"stride_{name}_batch"] == 64 * heads * case.head_dim
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

    case = wmma.CASES[-1]  # Batched, causal GQA exercises the distinct K/V strides.
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
    assert metadata["abi"] == "wmma-strided-v2"
    assert metadata["compiler_target"] == compiler_target
    assert len(compiled) == 1 and len(launches) == 2
    expected = pack_args(metadata["signature"], replay_values(metadata, buffers, case))
    for arguments, config in launches:
        assert arguments == expected
        assert config.grid == metadata["grid"] == (4, 4, 2)
        assert config.block == metadata["block"] == (32, 1, 1)


def test_wmma_reference_cohort_and_rejections():
    from dataclasses import replace
    import kernels
    from sdpa_reference.implementations import wmma

    assert len(wmma.CASES) == len({case.id for case in wmma.CASES}) == 8
    assert {(c.head_dim, c.kv_heads, c.causal) for c in wmma.CASES} == {
        (d, kv, causal) for d in (64, 128) for kv in (1, 4) for causal in (False, True)
    }
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


@pytest.mark.parametrize("abi", ["wmma-strided-v1", "wmma-strided-v2"])
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
    if abi == "wmma-strided-v2":
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
        assert metadata == {
            "name": launcher.kernel_name,
            "signature": launcher.signature,
            "grid": module.attention_dense_grid(spec),
            "block": module.attention_dense_block(spec),
            "runtime_shape": spec.runtime_shape,
        }


def test_frozen_runner_imports_all_adapters_without_source_checkout(tmp_path):
    import subprocess
    import sys

    from sdpa_reference.cli import _freeze_runner

    runner = tmp_path / "runner"
    _freeze_runner(runner)
    assert not list(runner.rglob("baseline_lock.json"))
    assert not list(runner.rglob("__pycache__"))
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
assert "kernels" not in sys.modules
assert "dispatch" not in sys.modules
assert "torch" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", script, str(runner), *ARCHITECTURES],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
