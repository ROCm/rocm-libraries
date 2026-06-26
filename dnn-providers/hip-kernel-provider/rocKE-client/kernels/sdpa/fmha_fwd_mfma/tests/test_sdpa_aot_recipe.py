# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import importlib.util
import json
import types
from pathlib import Path

import pytest

from rocke_client_aot.instance_schema import (
    InstanceError,
    KernelInstanceActions,
    attributes_match_constraints,
    normalize_attribute_constraints,
    parse_instance,
)
from rocke_client_aot.json_schema import load_json_schema, validate_json_schema


KERNEL_DIR = Path(__file__).resolve().parents[1]
SCHEMA_DIR = KERNEL_DIR / "schemas"
CLIENT_ROOT = Path(__file__).resolve().parents[4]
INSTANCE_ROOT = KERNEL_DIR / "instances"
TOOLS_DIR = CLIENT_ROOT / "aot" / "tools"

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
EXPECTED_ATTRIBUTE_CONSTRAINTS = {
    "mask_mode": {"equals": "none"},
    "dropout_probability": {"equals": 0.0},
    "scale_policy": {"equals": "default_1_over_sqrt_d"},
    "padding_mask": {"equals": False},
    "alibi_mask": {"equals": False},
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


def _load_sdpa_schema(name: str) -> tuple[dict, Path]:
    schema_path = SCHEMA_DIR / name
    return load_json_schema(schema_path), schema_path


def _eval_grid_formula(formula: dict, *, batch: int, instance: dict) -> list[int]:
    values = {**instance["compile_spec"], "batch": batch}

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

    assert data["schema"] == "rocke.aot.instance/v1"
    assert data["name"] == expected_name
    assert data["op"] == "sdpa_fwd"
    assert data["family"] == "fmha_fwd_mfma"
    assert data["arch"] == arch
    assert data["compile_spec"] == EXPECTED_COMPILE_SPEC
    assert data["selection"]["batch"] == {"min": 1, "max": 64}
    assert data["selection"]["attribute_constraints"] == EXPECTED_ATTRIBUTE_CONSTRAINTS
    assert data["test_profiles"] == [{"batch": 2}]
    assert parsed.compile_spec is data["compile_spec"]
    assert parsed.selection is data["selection"]
    assert parsed.test_profiles == data["test_profiles"]
    assert sorted(instance_path.parent.glob("*.instance.json")) == [instance_path]


@pytest.mark.parametrize("arch", ["gfx1151", "gfx942"])
def test_checked_in_sdpa_instance_matches_json_schema(arch):
    schema, schema_path = _load_sdpa_schema("instance.schema.json")

    validate_json_schema(
        _read_json(_instance_path(arch)), schema, schema_path=schema_path
    )


@pytest.mark.parametrize("alias", ["fp16", "f16", "half"])
def test_dtype_aliases_normalize_to_external_fp16(tmp_path, alias):
    instance_path = _copy_instance(tmp_path, "gfx1151")
    instance = _read_json(instance_path)
    instance["compile_spec"]["dtype"] = alias
    _write_json(instance_path, instance)

    parsed = parse_instance(instance_path, kernel_dir=KERNEL_DIR)

    assert parsed.data["compile_spec"]["dtype"] == "fp16"
    assert "_fp16_" in instance_path.name


def test_attribute_constraints_match_checked_in_sidecar_selection():
    constraints = normalize_attribute_constraints(EXPECTED_ATTRIBUTE_CONSTRAINTS)

    assert attributes_match_constraints(
        {
            "mask_mode": "none",
            "dropout_probability": 0.0,
            "scale_policy": "default_1_over_sqrt_d",
            "padding_mask": False,
            "alibi_mask": False,
        },
        constraints,
    )
    assert not attributes_match_constraints(
        {
            "mask_mode": "none",
            "dropout_probability": 0.0,
        },
        constraints,
    )


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
        parse_instance(instance_path, kernel_dir=KERNEL_DIR)


def test_invalid_shape_rejected_during_instance_parsing(tmp_path):
    instance_path = _copy_instance(tmp_path, "gfx1151")
    instance = _read_json(instance_path)
    instance["compile_spec"]["seqlen_q"] = 63
    _write_json(instance_path, instance)

    with pytest.raises(InstanceError, match="seqlen_q"):
        parse_instance(instance_path, kernel_dir=KERNEL_DIR)


def test_instance_name_must_match_compile_spec(tmp_path):
    instance_path = _copy_instance(tmp_path, "gfx1151")
    instance = _read_json(instance_path)
    instance["compile_spec"]["head_size"] = 128
    _write_json(instance_path, instance)

    with pytest.raises(InstanceError, match="SDPA FMHA MFMA basename"):
        parse_instance(instance_path, kernel_dir=KERNEL_DIR)


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

    sidecar = parsed.actions.emit_sidecar(parsed, parsed.spec, artifact, hsaco_filename)
    schema, schema_path = _load_sdpa_schema("sidecar.schema.json")
    validate_json_schema(sidecar, schema, schema_path=schema_path)

    assert set(sidecar) == {
        "schema",
        "cache_key",
        "artifact",
        "selection",
        "launch",
        "args_signature",
    }
    assert sidecar["schema"] == "rocke.aot.sidecar/v1"
    assert sidecar["cache_key"].startswith(
        f"sdpa_fwd:fmha_fwd_mfma:fmha_fwd_mfma:dense_fmha_fwd:"
    )
    assert sidecar["artifact"]["hsaco_filename"] == hsaco_filename
    assert sidecar["artifact"]["symbol"] == "rocke_fmha_fwd_mfma_unit_test"
    assert (
        sidecar["artifact"]["hsaco_sha256"] == hashlib.sha256(hsaco_bytes).hexdigest()
    )
    assert sidecar["artifact"]["hsaco_size"] == len(hsaco_bytes)

    selection = sidecar["selection"]
    assert selection["op"] == "sdpa_fwd"
    assert selection["arch"] == arch
    assert selection["dtypes"] == {
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

    def fake_build_kernel(spec, *, arch):
        assert spec is fake_spec
        assert arch == "gfx1151"
        return types.SimpleNamespace(name="fake_kernel")

    def fake_emit_sidecar(instance, spec, artifact, hsaco_filename):
        assert getattr(instance, "data", instance) == instance_data
        assert spec is fake_spec
        assert artifact.kernel_name == "rocke_fmha_fwd_mfma_fake_kernel"
        assert hsaco_filename == f"{instance_data['name']}.hsaco"
        digest = "0" * 64
        return {
            "schema": "rocke.aot.sidecar/v1",
            "cache_key": (
                "sdpa_fwd:fmha_fwd_mfma:fmha_fwd_mfma:dense_fmha_fwd:"
                "fp16_bshd_blockq16_blockk64:gfx1151:"
                f"{digest}:{digest}"
            ),
            "artifact": {
                "hsaco_filename": hsaco_filename,
                "symbol": artifact.kernel_name,
                "hsaco_sha256": hashlib.sha256(artifact.hsaco).hexdigest(),
                "hsaco_size": len(artifact.hsaco),
            },
            "selection": {
                "op": "sdpa_fwd",
                "arch": "gfx1151",
                "dtypes": {
                    "q": "fp16",
                    "k": "fp16",
                    "v": "fp16",
                    "o": "fp16",
                    "acc": "fp32",
                },
                "canonical_layout": "BSHD",
                "shape_constraints": {
                    "batch": {"min": 1, "max": 64},
                    "seqlen_q": {"equals": 64, "multiple_of": 16},
                    "seqlen_k": {"equals": 64, "multiple_of": 16},
                    "num_query_heads": {"equals": 4},
                    "num_kv_heads": {"equals": 4},
                    "head_size": {"equals": 64},
                },
                "attribute_constraints": EXPECTED_ATTRIBUTE_CONSTRAINTS,
            },
            "launch": {
                "shared_mem_bytes": 0,
                "grid_formula": {
                    "x": {"ceil_div": ["seqlen_q", 16]},
                    "y": "num_query_heads",
                    "z": "batch",
                },
                "block": [32, 1, 1],
                "tile_sizes": {
                    "block_q": 16,
                    "block_k": 64,
                    "head_size": 64,
                    "wave_size": 32,
                },
            },
            "args_signature": [
                {
                    "name": "Q",
                    "type": "ptr<f16, global>",
                    "kind": "pointer",
                    "size_bytes": 8,
                    "alignment": 8,
                },
            ],
        }

    fake_parsed = types.SimpleNamespace(
        path=instance_path,
        data=instance_data,
        compile_spec=instance_data["compile_spec"],
        selection=instance_data["selection"],
        test_profiles=instance_data["test_profiles"],
        spec=fake_spec,
        validation_reason="ok",
        actions=KernelInstanceActions(
            build_kernel=fake_build_kernel,
            emit_sidecar=fake_emit_sidecar,
        ),
    )

    def fake_parse_instance(path, *, kernel_dir=None):
        assert Path(path) == instance_path
        assert Path(kernel_dir) == KERNEL_DIR
        return fake_parsed

    def fake_compile_kernel(kernel, **kwargs):
        calls.append(kwargs)
        assert kernel.name == "fake_kernel"
        return types.SimpleNamespace(
            kernel_name="rocke_fmha_fwd_mfma_fake_kernel",
            hsaco=b"fake-hsaco",
            hsaco_bytes=len(b"fake-hsaco"),
            timings={},
            isa="amdgcn-amd-amdhsa--gfx1151",
        )

    monkeypatch.setattr(
        build_module, "parse_instance", fake_parse_instance, raising=False
    )
    monkeypatch.setattr(
        build_module, "compile_kernel", fake_compile_kernel, raising=False
    )

    import rocke.helpers as rocke_helpers
    import rocke_client_aot.instance_schema as instance_schema

    monkeypatch.setattr(
        instance_schema, "parse_instance", fake_parse_instance, raising=False
    )
    monkeypatch.setattr(
        rocke_helpers, "compile_kernel", fake_compile_kernel, raising=False
    )

    assert (
        build_module.main(
            ["--artifact-dir", str(tmp_path), "--kernel-dir", str(KERNEL_DIR)]
        )
        == 0
    )

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


def test_build_cli_validates_instance_schema_before_parse(
    tmp_path, monkeypatch, capsys
):
    instance_path = _copy_instance(tmp_path, "gfx1151")
    instance_data = _read_json(instance_path)
    instance_data["extra"] = True
    _write_json(instance_path, instance_data)
    build_module = _load_tool("rocke_aot_build")

    def fail_parse_instance(*_args, **_kwargs):
        pytest.fail("instance schema validation should run before parse_instance")

    monkeypatch.setattr(
        build_module, "parse_instance", fail_parse_instance, raising=False
    )

    result = build_module.main(
        ["--artifact-dir", str(tmp_path), "--kernel-dir", str(KERNEL_DIR)]
    )

    assert result == 1
    assert "Additional properties" in capsys.readouterr().err


def test_build_cli_rejects_hipcc_environment(tmp_path, monkeypatch, capsys):
    build_module = _load_tool("rocke_aot_build")
    monkeypatch.setenv("ROCKE_COMPILE_BACKEND", "hipcc")

    result = build_module.main(
        ["--artifact-dir", str(tmp_path), "--kernel-dir", str(KERNEL_DIR)]
    )

    assert result == 1
    assert "would request a hipcc compile path" in capsys.readouterr().err


def test_build_cli_rejects_empty_artifact_dir(tmp_path, capsys):
    build_module = _load_tool("rocke_aot_build")

    result = build_module.main(
        ["--artifact-dir", str(tmp_path), "--kernel-dir", str(KERNEL_DIR)]
    )

    assert result == 1
    assert "no .instance.json files found" in capsys.readouterr().err
