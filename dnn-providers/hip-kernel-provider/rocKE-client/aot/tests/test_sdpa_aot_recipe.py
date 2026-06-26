# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import importlib.util
import json
import types
from pathlib import Path

import pytest

from rocke_client_aot.instance_schema import InstanceError, parse_instance
from rocke_client_aot.sidecar import emit_sidecar


CLIENT_ROOT = Path(__file__).resolve().parents[1]
KERNEL_DIR = CLIENT_ROOT.parent / "kernels" / "sdpa" / "fmha_fwd_mfma"
INSTANCE_ROOT = KERNEL_DIR / "instances"
TOOLS_DIR = CLIENT_ROOT / "tools"

EXPECTED_BASENAME = "sdpa_fwd_fmha_fwd_mfma_fp16_bshd_{arch}_q64_k64_hq4_hkv4_d64_none"
EXPECTED_COMPILE_SPEC = {
    "dtype": "fp16",
    "canonical_layout": "BSHD",
    "seqlen_q": 64,
    "seqlen_k": 64,
    "num_query_heads": 4,
    "num_kv_heads": 4,
    "head_size": 64,
    "block_size_q": 16,
    "block_size_k": 64,
    "mask_mode": "none",
}


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _instance_path(arch: str) -> Path:
    name = EXPECTED_BASENAME.format(arch=arch)
    return INSTANCE_ROOT / arch / f"{name}.instance.json"


def _copy_instance(tmp_path: Path, arch: str, *, name: str | None = None) -> Path:
    src = _instance_path(arch)
    dst = tmp_path / (name or src.name)
    _write_json(dst, _read_json(src))
    return dst


def _load_tool(name: str):
    path = TOOLS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _eval_grid_formula(formula: dict, *, batch: int, instance: dict) -> list[int]:
    compile_spec = instance["compile_spec"]
    values = {**compile_spec, "batch": batch}

    def eval_axis(axis):
        if isinstance(axis, int):
            return axis
        if isinstance(axis, str):
            return int(values[axis])
        if isinstance(axis, dict) and "ceil_div" in axis:
            numerator, denominator = axis["ceil_div"]
            n = int(values[numerator] if isinstance(numerator, str) else numerator)
            d = int(
                values[denominator] if isinstance(denominator, str) else denominator
            )
            return (n + d - 1) // d
        raise AssertionError(f"unsupported grid formula axis: {axis!r}")

    return [eval_axis(formula[axis]) for axis in ("x", "y", "z")]


@pytest.mark.parametrize("arch", ["gfx1151", "gfx942"])
def test_checked_in_sdpa_instance_parses_and_filename_is_deterministic(arch):
    expected_name = EXPECTED_BASENAME.format(arch=arch)
    instance_path = _instance_path(arch)

    assert instance_path.name == f"{expected_name}.instance.json"
    assert instance_path.parent.name == arch

    parsed = parse_instance(instance_path)
    data = parsed.data

    assert data["schema"] == "ck.rocke.aot.instance/v1"
    assert data["name"] == expected_name
    assert data["op"] == "sdpa_fwd"
    assert data["family"] == "fmha_fwd_mfma"
    assert data["arch"] == arch
    assert data["compile_spec"] == EXPECTED_COMPILE_SPEC
    assert data["selection"]["batch"] == {"min": 1, "max": 64}
    assert data["test_profiles"] == [{"batch": 2}]
    assert sorted(instance_path.parent.glob("*.instance.json")) == [instance_path]


@pytest.mark.parametrize("alias", ["fp16", "f16", "half"])
def test_dtype_aliases_normalize_to_external_fp16(tmp_path, alias):
    instance_path = _copy_instance(tmp_path, "gfx1151")
    instance = _read_json(instance_path)
    instance["compile_spec"]["dtype"] = alias
    _write_json(instance_path, instance)

    parsed = parse_instance(instance_path)

    assert parsed.data["compile_spec"]["dtype"] == "fp16"
    assert "_fp16_" in instance_path.name


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc.__setitem__("schema", "bad.instance/v1"),
        lambda doc: doc.__setitem__("op", "gemm"),
        lambda doc: doc.__setitem__("family", "other_family"),
    ],
)
def test_schema_op_and_family_rejections(tmp_path, mutate):
    instance_path = _copy_instance(tmp_path, "gfx1151")
    instance = _read_json(instance_path)
    mutate(instance)
    _write_json(instance_path, instance)

    with pytest.raises(InstanceError):
        parse_instance(instance_path)


def test_invalid_shape_rejected_during_instance_parsing(tmp_path):
    instance_path = _copy_instance(tmp_path, "gfx1151")
    instance = _read_json(instance_path)
    instance["compile_spec"]["seqlen_q"] = 63
    _write_json(instance_path, instance)

    with pytest.raises(InstanceError, match="seqlen_q"):
        parse_instance(instance_path)


def test_instance_name_must_match_compile_spec(tmp_path):
    instance_path = _copy_instance(tmp_path, "gfx1151")
    instance = _read_json(instance_path)
    instance["compile_spec"]["head_size"] = 128
    _write_json(instance_path, instance)

    with pytest.raises(InstanceError, match="compile spec basename"):
        parse_instance(instance_path)


@pytest.mark.parametrize(
    ("arch", "expected_block"),
    [("gfx1151", [32, 1, 1]), ("gfx942", [64, 1, 1])],
)
def test_sidecar_required_fields_launch_signature_and_hashes(arch, expected_block):
    instance_path = _instance_path(arch)
    parsed = parse_instance(instance_path)
    hsaco_bytes = b"not-a-real-hsaco-for-sidecar-unit-tests"
    hsaco_filename = f"{parsed.data['name']}.hsaco"
    artifact = types.SimpleNamespace(
        kernel_name="rocke_fmha_fwd_mfma_unit_test",
        hsaco=hsaco_bytes,
        hsaco_bytes=len(hsaco_bytes),
        timings={},
        isa=f"amdgcn-amd-amdhsa--{arch}",
    )

    sidecar = emit_sidecar(parsed, parsed.spec, artifact, hsaco_filename)

    assert sidecar["schema"] == "ck.rocke.aot.sidecar/v1"
    assert sidecar["kernel_id"]["op"] == "sdpa_fwd"
    assert sidecar["kernel_id"]["family"] == "fmha_fwd_mfma"
    assert sidecar["kernel_id"]["arch"] == arch
    assert sidecar["artifact"]["hsaco_filename"] == hsaco_filename
    assert sidecar["artifact"]["symbol"] == "rocke_fmha_fwd_mfma_unit_test"
    assert (
        sidecar["artifact"]["hsaco_sha256"] == hashlib.sha256(hsaco_bytes).hexdigest()
    )
    assert sidecar["artifact"]["hsaco_size"] == len(hsaco_bytes)
    assert sidecar["selection"]["dtypes"] == {
        "q": "fp16",
        "k": "fp16",
        "v": "fp16",
        "o": "fp16",
        "acc": "fp32",
    }
    assert _eval_grid_formula(
        sidecar["launch"]["grid_formula"], batch=2, instance=parsed.data
    ) == [4, 4, 2]
    assert sidecar["launch"]["block"] == expected_block

    signature = sidecar["args_signature"]
    assert [arg["name"] for arg in signature[:4]] == ["Q", "K", "V", "O"]
    pointer_args = [arg for arg in signature if arg["kind"] == "pointer"]
    assert pointer_args
    assert all(arg["type"] == "ptr<f16, global>" for arg in pointer_args)
    assert all(arg["size_bytes"] == 8 and arg["alignment"] == 8 for arg in pointer_args)
    scalar_args = [arg for arg in signature if arg["kind"] == "scalar"]
    assert scalar_args
    assert all(arg["size_bytes"] == 4 and arg["alignment"] == 4 for arg in scalar_args)


def test_build_cli_uses_python_comgr_path_and_writes_sidecar(tmp_path, monkeypatch):
    instance_path = _copy_instance(tmp_path, "gfx1151")
    instance_data = _read_json(instance_path)
    build_module = _load_tool("rocke_aot_build")
    calls = []

    fake_spec = object()
    fake_parsed = types.SimpleNamespace(
        path=instance_path,
        data=instance_data,
        spec=fake_spec,
        validation_reason="ok",
    )

    def fake_parse_instance(path):
        assert Path(path) == instance_path
        return fake_parsed

    def fake_build_fmha_fwd_mfma(spec, *, arch):
        assert spec is fake_spec
        assert arch == "gfx1151"
        return types.SimpleNamespace(name="fake_kernel")

    def fake_compile_kernel(kernel, **kwargs):
        calls.append(kwargs)
        assert kernel.name == "fake_kernel"
        return types.SimpleNamespace(
            kernel_name="fake_kernel",
            hsaco=b"fake-hsaco",
            hsaco_bytes=len(b"fake-hsaco"),
            timings={},
            isa="amdgcn-amd-amdhsa--gfx1151",
        )

    def fake_emit_sidecar(instance, spec, artifact, hsaco_filename):
        assert getattr(instance, "data", instance) == instance_data
        assert spec is fake_spec
        assert artifact.kernel_name == "fake_kernel"
        assert hsaco_filename == f"{instance_data['name']}.hsaco"
        return {
            "schema": "ck.rocke.aot.sidecar/v1",
            "artifact": {
                "hsaco_filename": hsaco_filename,
                "symbol": artifact.kernel_name,
                "hsaco_sha256": hashlib.sha256(artifact.hsaco).hexdigest(),
                "hsaco_size": len(artifact.hsaco),
            },
        }

    monkeypatch.setattr(
        build_module, "parse_instance", fake_parse_instance, raising=False
    )
    monkeypatch.setattr(
        build_module, "build_fmha_fwd_mfma", fake_build_fmha_fwd_mfma, raising=False
    )
    monkeypatch.setattr(
        build_module, "compile_kernel", fake_compile_kernel, raising=False
    )
    monkeypatch.setattr(build_module, "emit_sidecar", fake_emit_sidecar, raising=False)

    import rocke.helpers as rocke_helpers
    import rocke.instances.common.fmha_mfma as fmha_mfma
    import rocke_client_aot.instance_schema as instance_schema
    import rocke_client_aot.sidecar as sidecar

    monkeypatch.setattr(
        instance_schema, "parse_instance", fake_parse_instance, raising=False
    )
    monkeypatch.setattr(
        fmha_mfma, "build_fmha_fwd_mfma", fake_build_fmha_fwd_mfma, raising=False
    )
    monkeypatch.setattr(
        rocke_helpers, "compile_kernel", fake_compile_kernel, raising=False
    )
    monkeypatch.setattr(sidecar, "emit_sidecar", fake_emit_sidecar, raising=False)

    assert build_module.main(["--artifact-dir", str(tmp_path)]) == 0

    assert calls
    assert calls[0]["arch"] == "gfx1151"
    assert calls[0]["backend"] == "python"
    assert calls[0]["capture_ir_text"] is False
    hsaco_path = instance_path.with_name(f"{instance_data['name']}.hsaco")
    assert hsaco_path.read_bytes() == b"fake-hsaco"
    sidecar = _read_json(
        instance_path.with_name(f"{instance_data['name']}.sidecar.json")
    )
    assert (
        sidecar["artifact"]["hsaco_sha256"] == hashlib.sha256(b"fake-hsaco").hexdigest()
    )
    assert sidecar["artifact"]["hsaco_size"] == len(b"fake-hsaco")
