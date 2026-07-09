# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import hashlib
import sys
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
    parse_instance_list,
)
from rocke_client_aot.json_schema import (
    SchemaValidationError,
    load_json_schema,
    validate_json_schema,
)


AOT_DIR = Path(__file__).resolve().parents[1]
LIBRARY_DIR = AOT_DIR.parent
KERNELS_DIR = LIBRARY_DIR / "kernels"
HANDLER = KERNELS_DIR / "common" / "fmha_mfma_aot.py"
SCHEMA_DIR = AOT_DIR / "schemas" / "fmha_fwd_mfma"
TOOLS_DIR = AOT_DIR / "tools"

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


def _aot_list_path(arch: str) -> Path:
    return KERNELS_DIR / arch / "fmha_fwd_mfma" / "aot_list.json"


def _copy_list(dst_dir: Path, arch: str) -> Path:
    dst = dst_dir / "aot_list.json"
    _write_json(dst, _read_json(_aot_list_path(arch)))
    return dst


def _load_tool(name: str):
    path = TOOLS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_numeric_tool():
    path = Path(__file__).resolve().parent / "sdpa_aot_numeric.py"
    spec = importlib.util.spec_from_file_location("sdpa_aot_numeric", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_kernel_handler():
    path = HANDLER
    spec = importlib.util.spec_from_file_location("sdpa_aot_instance", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_numeric_verifier():
    path = Path(__file__).resolve().parent / "sdpa_aot_numeric.py"
    spec = importlib.util.spec_from_file_location("sdpa_aot_numeric", path)
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
def test_checked_in_sdpa_instance_parses_and_name_is_deterministic(arch):
    expected_name = EXPECTED_BASENAME.format(arch=arch)
    list_path = _aot_list_path(arch)

    assert list_path.name == "aot_list.json"
    assert list_path.parent.name == "fmha_fwd_mfma"
    assert list_path.parent.parent.name == arch

    parsed = parse_instance_list(list_path, handler_path=HANDLER)
    assert len(parsed) == 1
    data = parsed[0].data

    assert data["schema"] == "rocke.aot.instance/v1"
    assert data["name"] == expected_name
    assert data["op"] == "sdpa_fwd"
    assert data["family"] == "fmha_fwd_mfma"
    assert data["arch"] == arch
    assert data["compile_spec"] == EXPECTED_COMPILE_SPEC
    assert data["selection"]["batch"] == {"min": 1, "max": 64}
    assert data["selection"]["attribute_constraints"] == EXPECTED_ATTRIBUTE_CONSTRAINTS
    assert data["test_profiles"] == [{"batch": 1}, {"batch": 2}, {"batch": 64}]
    assert parsed[0].compile_spec is data["compile_spec"]
    assert parsed[0].selection is data["selection"]
    assert parsed[0].test_profiles == data["test_profiles"]


@pytest.mark.parametrize("arch", ["gfx1151", "gfx942"])
def test_checked_in_sdpa_instance_matches_json_schema(arch):
    schema, schema_path = _load_sdpa_schema("instance.schema.json")

    for instance in _read_json(_aot_list_path(arch)):
        validate_json_schema(instance, schema, schema_path=schema_path)


@pytest.mark.parametrize("alias", ["fp16", "f16", "half"])
def test_dtype_aliases_normalize_to_external_fp16(tmp_path, alias):
    list_path = _copy_list(tmp_path, "gfx1151")
    instances = _read_json(list_path)
    instances[0]["compile_spec"]["dtype"] = alias
    _write_json(list_path, instances)

    parsed = parse_instance_list(list_path, handler_path=HANDLER)[0]

    assert parsed.data["compile_spec"]["dtype"] == "fp16"
    assert "_fp16_" in parsed.data["name"]


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
    list_path = _copy_list(tmp_path, "gfx1151")
    instances = _read_json(list_path)
    mutate(instances[0])
    _write_json(list_path, instances)

    with pytest.raises(InstanceError):
        parse_instance_list(list_path, handler_path=HANDLER)


def test_invalid_shape_rejected_during_instance_parsing(tmp_path):
    list_path = _copy_list(tmp_path, "gfx1151")
    instances = _read_json(list_path)
    instances[0]["compile_spec"]["seqlen_q"] = 63
    _write_json(list_path, instances)

    with pytest.raises(InstanceError, match="seqlen_q"):
        parse_instance_list(list_path, handler_path=HANDLER)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda spec: spec.__setitem__("dtype", "bf16"), "unsupported dtype"),
        (lambda spec: spec.__setitem__("canonical_layout", "BHSD"), "canonical_layout"),
        (lambda spec: spec.__setitem__("mask_mode", "causal"), "mask_mode"),
        (lambda spec: spec.__setitem__("block_size_q", 0), "block_size_q"),
        (lambda spec: spec.__setitem__("block_size_q", 32), "must be 16"),
        (lambda spec: spec.__setitem__("seqlen_k", 63), "seqlen_k"),
        (lambda spec: spec.__setitem__("head_size", 96), "head_size"),
        (
            lambda spec: (
                spec.__setitem__("num_query_heads", 3),
                spec.__setitem__("num_kv_heads", 2),
            ),
            "num_query_heads",
        ),
    ],
)
def test_invalid_compile_spec_fields_are_rejected(tmp_path, mutate, message):
    list_path = _copy_list(tmp_path, "gfx1151")
    instances = _read_json(list_path)
    mutate(instances[0]["compile_spec"])
    _write_json(list_path, instances)

    with pytest.raises(InstanceError, match=message):
        parse_instance_list(list_path, handler_path=HANDLER)


def test_parse_instance_rejects_arch_invalid_spec(tmp_path, monkeypatch):
    import kernels.common.fmha_mfma as fmha_mfma

    list_path = _copy_list(tmp_path, "gfx1151")
    monkeypatch.setattr(
        fmha_mfma,
        "is_valid_spec",
        lambda _spec, _arch: (False, "unit-test rejection"),
    )

    with pytest.raises(InstanceError, match="unit-test rejection"):
        parse_instance_list(list_path, handler_path=HANDLER)


def test_instance_name_must_match_compile_spec(tmp_path):
    list_path = _copy_list(tmp_path, "gfx1151")
    instances = _read_json(list_path)
    instances[0]["compile_spec"]["head_size"] = 128
    _write_json(list_path, instances)

    with pytest.raises(InstanceError, match="SDPA FMHA MFMA basename"):
        parse_instance_list(list_path, handler_path=HANDLER)


@pytest.mark.parametrize(
    ("arch", "expected_block"),
    [("gfx1151", [32, 1, 1]), ("gfx942", [64, 1, 1])],
)
def test_sidecar_required_fields_launch_signature_and_hashes(arch, expected_block):
    parsed = parse_instance_list(_aot_list_path(arch), handler_path=HANDLER)[0]
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
        "sdpa_fwd:fmha_fwd_mfma:fmha_fwd_mfma:dense_fmha_fwd:"
    )
    assert ":hipkg-sdpa-fwd-fmha-mfma/v1:" in sidecar["cache_key"]
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
    assert sidecar["launch"]["tile_sizes"]["block_q"] == 16

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
    list_path = _copy_list(tmp_path, "gfx1151")
    instance_data = _read_json(list_path)[0]
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
                "hipkg-sdpa-fwd-fmha-mfma/v1:"
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
        path=list_path,
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

    def fake_parse_instance_list(path, *, handler_path=None, expected_arch=None):
        assert Path(path) == list_path
        assert Path(handler_path) == HANDLER
        assert expected_arch == "gfx1151"
        return [fake_parsed]

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
        build_module, "parse_instance_list", fake_parse_instance_list, raising=False
    )
    monkeypatch.setattr(
        build_module, "compile_kernel", fake_compile_kernel, raising=False
    )

    assert (
        build_module.main(
            [
                "--artifact-dir",
                str(tmp_path),
                "--handler",
                str(HANDLER),
                "--schema-dir",
                str(SCHEMA_DIR),
                "--arch",
                "gfx1151",
            ]
        )
        == 0
    )

    assert calls
    assert calls[0]["arch"] == "gfx1151"
    assert calls[0]["backend"] == "python"
    assert calls[0]["capture_ir_text"] is False
    hsaco_path = tmp_path / f"{instance_data['name']}.hsaco"
    assert hsaco_path.read_bytes() == b"fake-hsaco"
    sidecar = _read_json(tmp_path / f"{instance_data['name']}.sidecar.json")
    assert (
        sidecar["artifact"]["hsaco_sha256"] == hashlib.sha256(b"fake-hsaco").hexdigest()
    )
    assert sidecar["artifact"]["hsaco_size"] == len(b"fake-hsaco")


@pytest.mark.parametrize(
    ("failure_mode", "expected_error"),
    [("emit", RuntimeError), ("validate", SchemaValidationError)],
)
def test_build_one_cleans_artifacts_when_sidecar_fails(
    tmp_path, monkeypatch, failure_mode, expected_error
):
    list_path = _copy_list(tmp_path, "gfx1151")
    instance_data = _read_json(list_path)[0]
    build_module = _load_tool("rocke_aot_build")
    fake_spec = object()

    def fake_build_kernel(spec, *, arch):
        assert spec is fake_spec
        assert arch == "gfx1151"
        return types.SimpleNamespace(name="fake_kernel")

    def fake_emit_sidecar(_instance, _spec, _artifact, _hsaco_filename):
        if failure_mode == "emit":
            raise RuntimeError("sidecar generation failed")
        return {"schema": "rocke.aot.sidecar/v1"}

    fake_parsed = types.SimpleNamespace(
        data=instance_data,
        spec=fake_spec,
        actions=KernelInstanceActions(
            build_kernel=fake_build_kernel,
            emit_sidecar=fake_emit_sidecar,
        ),
    )

    monkeypatch.setattr(
        build_module,
        "compile_kernel",
        lambda *_args, **_kwargs: types.SimpleNamespace(
            kernel_name="rocke_fmha_fwd_mfma_fake_kernel",
            hsaco=b"partial-hsaco",
            hsaco_bytes=len(b"partial-hsaco"),
            timings={},
            isa="amdgcn-amd-amdhsa--gfx1151",
        ),
    )

    with pytest.raises(expected_error):
        build_module._build_one(
            fake_parsed,
            tmp_path,
            SCHEMA_DIR / "sidecar.schema.json",
        )

    assert not (tmp_path / f"{instance_data['name']}.hsaco").exists()
    assert not (tmp_path / f"{instance_data['name']}.sidecar.json").exists()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["aot_list.json"]


def test_build_module_internal_fallbacks_and_stale_cleanup(tmp_path):
    build_module = _load_tool("rocke_aot_build")

    data = {"name": "direct"}
    assert build_module._parsed_instance_data(data) is data

    fallback_data = {"name": "fallback"}
    assert (
        build_module._parsed_instance_data(
            types.SimpleNamespace(instance=fallback_data)
        )
        is fallback_data
    )
    with pytest.raises(TypeError, match="parsed instance data"):
        build_module._parsed_instance_data(object())

    fallback_spec = object()
    assert (
        build_module._parsed_spec(types.SimpleNamespace(fmha_spec=fallback_spec))
        is fallback_spec
    )
    with pytest.raises(TypeError, match="spec or fmha_spec"):
        build_module._parsed_spec(object())

    stale_hsaco = tmp_path / "old.hsaco"
    stale_sidecar = tmp_path / "old.sidecar.json"
    stale_dir = tmp_path / "dir.hsaco"
    keep = tmp_path / "keep.txt"
    stale_hsaco.write_bytes(b"old")
    stale_sidecar.write_text("{}", encoding="utf-8")
    stale_dir.mkdir()
    keep.write_text("keep", encoding="utf-8")

    build_module._clean_stale_outputs(tmp_path)

    assert not stale_hsaco.exists()
    assert not stale_sidecar.exists()
    assert stale_dir.is_dir()
    assert keep.is_file()

    # Branch: overlay dir present but missing the family file -> shared fallback.
    fallback_kernel_dir = tmp_path / "kernel"
    fallback_kernel_dir.mkdir()
    assert build_module._schema_path(fallback_kernel_dir, "instance").name == (
        "instance.schema.json"
    )
    # Branch: --schema-dir omitted (schema_dir is None) -> shared schema. This is
    # the default CMake path when a family ships no overlay.
    for _name in ("instance", "sidecar"):
        _shared = build_module._schema_path(None, _name)
        assert _shared.name == f"{_name}.schema.json"
        assert _shared.is_file()
        assert _shared.parent.name == "schemas"


def test_build_cli_help_returns_argparse_exit_code(capsys):
    build_module = _load_tool("rocke_aot_build")

    assert build_module.main(["--help"]) == 0
    assert (
        "Build rocKE client AOT HSACO and sidecar artifacts" in capsys.readouterr().out
    )


def test_build_cli_reports_argument_and_schema_errors(tmp_path, capsys, monkeypatch):
    build_module = _load_tool("rocke_aot_build")

    missing_artifact = tmp_path / "missing-artifacts"
    result = build_module.main(
        [
            "--artifact-dir",
            str(missing_artifact),
            "--handler",
            str(HANDLER),
            "--schema-dir",
            str(SCHEMA_DIR),
            "--arch",
            "gfx1151",
        ]
    )
    assert result == 1
    assert "artifact directory does not exist" in capsys.readouterr().err

    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    missing_handler = tmp_path / "missing-handler.py"
    result = build_module.main(
        [
            "--artifact-dir",
            str(artifact_dir),
            "--handler",
            str(missing_handler),
            "--arch",
            "gfx1151",
        ]
    )
    assert result == 1
    assert "handler module does not exist" in capsys.readouterr().err

    _copy_list(artifact_dir, "gfx1151")
    monkeypatch.setattr(
        build_module,
        "_schema_path",
        lambda _kernel_dir, name: tmp_path / f"missing-{name}.schema.json",
    )
    result = build_module.main(
        [
            "--artifact-dir",
            str(artifact_dir),
            "--handler",
            str(HANDLER),
            "--schema-dir",
            str(SCHEMA_DIR),
            "--arch",
            "gfx1151",
        ]
    )
    assert result == 1
    assert "schema file does not exist" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("name", "", "instance name"),
        ("arch", "", "instance arch"),
    ],
)
def test_build_one_requires_parsed_name_and_arch(tmp_path, field, value, message):
    build_module = _load_tool("rocke_aot_build")
    list_path = _copy_list(tmp_path, "gfx1151")
    instance_data = _read_json(list_path)[0]
    instance_data[field] = value

    fake_parsed = types.SimpleNamespace(
        data=instance_data,
        spec=object(),
        actions=types.SimpleNamespace(),
    )

    with pytest.raises(ValueError, match=message):
        build_module._build_one(
            fake_parsed,
            tmp_path,
            SCHEMA_DIR / "sidecar.schema.json",
        )


def test_build_cli_validates_instance_schema_before_parse(
    tmp_path, monkeypatch, capsys
):
    list_path = _copy_list(tmp_path, "gfx1151")
    instances = _read_json(list_path)
    instances[0]["extra"] = True
    _write_json(list_path, instances)
    build_module = _load_tool("rocke_aot_build")

    def fail_parse_instance_list(*_args, **_kwargs):
        pytest.fail("instance schema validation should run before parse_instance_list")

    monkeypatch.setattr(
        build_module, "parse_instance_list", fail_parse_instance_list, raising=False
    )

    result = build_module.main(
        [
            "--artifact-dir",
            str(tmp_path),
            "--handler",
            str(HANDLER),
            "--schema-dir",
            str(SCHEMA_DIR),
            "--arch",
            "gfx1151",
        ]
    )

    assert result == 1
    assert "Additional properties" in capsys.readouterr().err


def test_sidecar_accepts_instance_mapping_and_instance_attr_fallback():
    parsed = parse_instance_list(_aot_list_path("gfx1151"), handler_path=HANDLER)[0]
    artifact = types.SimpleNamespace(
        kernel_name="rocke_fmha_fwd_mfma_unit_test",
        hsaco=b"hsaco",
        hsaco_bytes=len(b"hsaco"),
        timings={},
        isa="amdgcn-amd-amdhsa--gfx1151",
    )

    direct = parsed.actions.emit_sidecar(
        parsed.data, parsed.spec, artifact, f"{parsed.data['name']}.hsaco"
    )
    fallback = parsed.actions.emit_sidecar(
        types.SimpleNamespace(instance=parsed.data),
        parsed.spec,
        artifact,
        f"{parsed.data['name']}.hsaco",
    )

    assert direct == fallback


def test_sidecar_rejects_unsupported_kernel_id():
    parsed = parse_instance_list(_aot_list_path("gfx1151"), handler_path=HANDLER)[0]
    data = dict(parsed.data)
    data["op"] = "gemm"
    artifact = types.SimpleNamespace(
        kernel_name="rocke_fmha_fwd_mfma_unit_test",
        hsaco=b"hsaco",
        hsaco_bytes=len(b"hsaco"),
        timings={},
        isa="amdgcn-amd-amdhsa--gfx1151",
    )

    with pytest.raises(ValueError, match="unsupported sidecar kernel id"):
        parsed.actions.emit_sidecar(
            data, parsed.spec, artifact, f"{data['name']}.hsaco"
        )


@pytest.mark.parametrize(
    ("signature", "message"),
    [
        ([{"name": 1, "type": "i32"}], "string name and type"),
        ([{"name": "Q", "type": "i64"}], "unsupported scalar ABI type"),
        (
            [
                {"name": "A", "type": "ptr<f16, global>"},
                {"name": "K", "type": "ptr<f16, global>"},
                {"name": "V", "type": "ptr<f16, global>"},
                {"name": "O", "type": "ptr<f16, global>"},
            ],
            "must start with Q/K/V/O",
        ),
        (
            [
                {"name": "Q", "type": "ptr<i32, global>"},
                {"name": "K", "type": "ptr<f16, global>"},
                {"name": "V", "type": "ptr<f16, global>"},
                {"name": "O", "type": "ptr<f16, global>"},
            ],
            "unexpected type",
        ),
    ],
)
def test_enrich_args_signature_rejects_invalid_abi(monkeypatch, signature, message):
    handler = _load_kernel_handler()
    monkeypatch.setattr(handler, "fmha_fwd_mfma_signature", lambda _spec: signature)

    with pytest.raises((TypeError, ValueError), match=message):
        handler.enrich_args_signature(object())


def test_build_kernel_delegates_to_rocke_builder(monkeypatch):
    import kernels.common.fmha_mfma as fmha_mfma

    handler = _load_kernel_handler()
    fake_spec = object()
    fake_kernel = object()

    def fake_build(spec, *, arch):
        assert spec is fake_spec
        assert arch == "gfx1151"
        return fake_kernel

    monkeypatch.setattr(fmha_mfma, "build_fmha_fwd_mfma", fake_build)

    assert handler.build_kernel(fake_spec, arch="gfx1151") is fake_kernel


def test_validate_compile_spec_reports_future_required_fields(monkeypatch):
    handler = _load_kernel_handler()
    monkeypatch.setattr(
        handler,
        "_COMPILE_FIELDS",
        (*handler._COMPILE_FIELDS, "future_required_field"),
    )

    with pytest.raises(InstanceError, match="future_required_field"):
        handler._validate_compile_spec(EXPECTED_COMPILE_SPEC, context="compile_spec")


def test_private_mapping_helper_rejects_non_mapping():
    handler = _load_kernel_handler()

    with pytest.raises(TypeError, match="must be a mapping"):
        handler._as_mapping([], "value")


def test_build_cli_rejects_hipcc_environment(tmp_path, monkeypatch, capsys):
    build_module = _load_tool("rocke_aot_build")
    monkeypatch.setenv("ROCKE_COMPILE_BACKEND", "hipcc")

    result = build_module.main(
        [
            "--artifact-dir",
            str(tmp_path),
            "--handler",
            str(HANDLER),
            "--schema-dir",
            str(SCHEMA_DIR),
            "--arch",
            "gfx1151",
        ]
    )

    assert result == 1
    assert "would request a hipcc compile path" in capsys.readouterr().err


def test_build_cli_rejects_missing_aot_list(tmp_path, capsys):
    build_module = _load_tool("rocke_aot_build")

    result = build_module.main(
        [
            "--artifact-dir",
            str(tmp_path),
            "--handler",
            str(HANDLER),
            "--schema-dir",
            str(SCHEMA_DIR),
            "--arch",
            "gfx1151",
        ]
    )

    assert result == 1
    assert "no aot_list.json found" in capsys.readouterr().err


def _numeric_compile_spec(**overrides):
    spec = {
        "dtype": "fp16",
        "canonical_layout": "BSHD",
        "seqlen_q": 16,
        "seqlen_k": 16,
        "num_query_heads": 2,
        "num_kv_heads": 1,
        "head_size": 32,
        "block_size_q": 16,
        "block_size_k": 64,
        "mask_mode": "none",
    }
    spec.update(overrides)
    return spec


def _numeric_sidecar():
    scalar_names = [
        "seqlen_q",
        "seqlen_k",
        "stride_q_token",
        "stride_q_head",
        "stride_k_token",
        "stride_k_head",
        "stride_v_token",
        "stride_v_head",
        "stride_o_token",
        "stride_o_head",
    ]
    return {
        "artifact": {
            "hsaco_filename": "kernel.hsaco",
            "symbol": "kernel_symbol",
            "hsaco_sha256": hashlib.sha256(b"hsaco").hexdigest(),
            "hsaco_size": len(b"hsaco"),
        },
        "launch": {
            "grid_formula": {
                "x": {"ceil_div": ["seqlen_q", 16]},
                "y": "num_query_heads",
                "z": "batch",
            },
            "block": [32, 1, 1],
            "shared_mem_bytes": 0,
        },
        "args_signature": [
            {
                "name": name,
                "kind": "pointer",
                "type": "ptr<f16, global>",
                "size_bytes": 8,
            }
            for name in ("Q", "K", "V", "O")
        ]
        + [{"name": "scale_log2", "kind": "scalar", "type": "f32", "size_bytes": 4}]
        + [
            {"name": name, "kind": "scalar", "type": "i32", "size_bytes": 4}
            for name in scalar_names
        ],
    }


class _FakeHipModule:
    def __init__(self):
        self.unloaded = False

    def get_function(self, name):
        assert name == "kernel_symbol"
        return object()

    def unload(self):
        self.unloaded = True


class _FakeRuntime:
    def __init__(self, *, fail_after=None):
        self.fail_after = fail_after
        self.alloc_calls = 0
        self.next_ptr = 1000
        self.buffers = {}
        self.freed = []
        self.module = _FakeHipModule()

    def load_module(self, hsaco):
        assert hsaco == b"hsaco"
        return self.module

    def alloc(self, size):
        if self.fail_after is not None and self.alloc_calls >= self.fail_after:
            raise RuntimeError("allocation failed")
        self.alloc_calls += 1
        ptr = self.next_ptr
        self.next_ptr += 1
        self.buffers[ptr] = bytearray(size)
        return ptr

    def memcpy_h2d(self, ptr, host, size):
        self.buffers[ptr][:size] = bytes(host[:size])

    def memset(self, ptr, value, size):
        self.buffers[ptr][:size] = bytes([value]) * size

    def launch(self, function, grid, block, packed, *, shared_bytes):
        assert function is not None
        assert grid == (1, 2, 1)
        assert block == (32, 1, 1)
        assert packed
        assert shared_bytes == 0

    def sync(self):
        pass

    def memcpy_d2h(self, host, ptr, size):
        for index, byte in enumerate(self.buffers[ptr][:size]):
            host[index] = byte

    def free(self, ptr):
        self.freed.append(ptr)


def test_numeric_helpers_validate_paths_json_grid_and_args(tmp_path):
    numeric = _load_numeric_tool()

    sample = tmp_path / "sample.json"
    sample.write_text('{"ok": true}', encoding="utf-8")
    assert numeric._load_json(sample) == {"ok": True}

    bad_json = tmp_path / "array.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        numeric._load_json(bad_json)

    aot_list = tmp_path / "aot_list.json"
    aot_list.write_text('[{"name": "x"}]', encoding="utf-8")
    assert numeric._load_aot_list(aot_list) == [{"name": "x"}]
    empty_list = tmp_path / "empty_list.json"
    empty_list.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty JSON array"):
        numeric._load_aot_list(empty_list)
    object_list = tmp_path / "object_list.json"
    object_list.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="non-empty JSON array"):
        numeric._load_aot_list(object_list)

    assert numeric._eval_grid_formula(
        {"x": {"ceil_div": ["n", 16]}, "y": "heads", "z": 3},
        {"n": 17, "heads": 2},
    ) == (2, 2, 3)
    with pytest.raises(ValueError, match="invalid ceil_div"):
        numeric._eval_grid_formula({"x": {"ceil_div": ["n"]}, "y": 1, "z": 1}, {"n": 1})
    with pytest.raises(ValueError, match="unsupported"):
        numeric._eval_grid_formula({"x": [], "y": 1, "z": 1}, {})

    packed = numeric._pack_args(
        [
            {
                "name": "ptr",
                "kind": "pointer",
                "type": "ptr<f16, global>",
                "size_bytes": 8,
            },
            {"name": "f", "kind": "scalar", "type": "f32", "size_bytes": 4},
            {"name": "i", "kind": "scalar", "type": "i32", "size_bytes": 4},
            {"name": "q", "kind": "scalar", "type": "i64", "size_bytes": 8},
        ],
        {"ptr": 7, "f": 1.5, "i": -2, "q": 9},
    )
    assert len(packed) == 24
    with pytest.raises(ValueError, match="pointer arg"):
        numeric._pack_args(
            [
                {
                    "name": "ptr",
                    "kind": "pointer",
                    "type": "ptr<f16, global>",
                    "size_bytes": 4,
                }
            ],
            {"ptr": 7},
        )
    with pytest.raises(ValueError, match="unsupported arg"):
        numeric._pack_args(
            [{"name": "flag", "kind": "scalar", "type": "bool", "size_bytes": 1}],
            {"flag": True},
        )


def test_numeric_reference_and_host_buffer():
    import numpy as np

    numeric = _load_numeric_tool()
    q = np.ones((2, 1, 4), dtype=np.float16)
    k = np.ones((3, 1, 4), dtype=np.float16)
    v = np.arange(12, dtype=np.float16).reshape(3, 1, 4)

    reference = numeric._ref_attention(q, k, v)

    assert reference.shape == q.shape
    assert numeric._host_buffer(np.zeros((2,), dtype=np.float16))._length_ == 4


def test_numeric_verify_profile_success_and_allocation_cleanup(monkeypatch):
    import numpy as np

    numeric = _load_numeric_tool()
    fake_runtime = _FakeRuntime()
    monkeypatch.setitem(
        sys.modules,
        "rocke.runtime.hip_module",
        types.SimpleNamespace(Runtime=lambda: fake_runtime),
    )
    monkeypatch.setattr(
        numeric,
        "_ref_attention",
        lambda q, _k, _v: np.zeros(q.shape, dtype=np.float32),
    )
    instance = {"compile_spec": _numeric_compile_spec()}

    assert numeric._verify_profile(instance, _numeric_sidecar(), b"hsaco", batch=1)
    assert len(fake_runtime.freed) == 4
    assert fake_runtime.module.unloaded

    failing_runtime = _FakeRuntime(fail_after=1)
    monkeypatch.setitem(
        sys.modules,
        "rocke.runtime.hip_module",
        types.SimpleNamespace(Runtime=lambda: failing_runtime),
    )
    with pytest.raises(RuntimeError, match="allocation failed"):
        numeric._verify_profile(instance, _numeric_sidecar(), b"hsaco", batch=1)
    assert failing_runtime.freed == [1000]
    assert failing_runtime.module.unloaded

    bad_mask = {"compile_spec": _numeric_compile_spec(mask_mode="causal")}
    with pytest.raises(ValueError, match="mask_mode"):
        numeric._verify_profile(bad_mask, _numeric_sidecar(), b"hsaco", batch=1)


def test_numeric_verify_instance_digest_profiles_and_main(
    tmp_path, monkeypatch, capsys
):
    numeric = _load_numeric_tool()
    hsaco_path = tmp_path / "kernel.hsaco"
    sidecar_path = tmp_path / "sample.sidecar.json"
    hsaco_path.write_bytes(b"hsaco")
    sidecar = _numeric_sidecar()
    _write_json(sidecar_path, sidecar)

    no_profiles = {
        "name": "sample",
        "compile_spec": _numeric_compile_spec(),
        "test_profiles": [],
    }
    assert numeric._verify_instance(no_profiles, tmp_path)
    assert "SKIP instance without test profiles" in capsys.readouterr().out

    bad_sidecar = dict(sidecar)
    bad_sidecar["artifact"] = dict(sidecar["artifact"], hsaco_sha256="0" * 64)
    _write_json(sidecar_path, bad_sidecar)
    with pytest.raises(ValueError, match="digest/size mismatch"):
        numeric._verify_instance(no_profiles, tmp_path)

    _write_json(sidecar_path, sidecar)
    with_profile = {
        "name": "sample",
        "compile_spec": _numeric_compile_spec(),
        "test_profiles": [{"batch": 7}],
    }
    batches = []
    monkeypatch.setattr(
        numeric,
        "_verify_profile",
        lambda _instance, _sidecar, _hsaco, batch: batches.append(batch) or True,
    )
    assert numeric._verify_instance(with_profile, tmp_path)
    assert batches == [7]

    _write_json(tmp_path / "aot_list.json", [with_profile])

    monkeypatch.setattr(numeric, "_device_arch", lambda: None)
    assert numeric.main(["--arch", "gfx1151", "--artifact-dir", str(tmp_path)]) == 77
    assert "does not match" in capsys.readouterr().out

    monkeypatch.setattr(numeric, "_device_arch", lambda: "gfx1151")
    monkeypatch.setattr(
        numeric, "_verify_instance", lambda _instance, _artifact_dir: False
    )
    assert numeric.main(["--arch", "gfx1151", "--artifact-dir", str(tmp_path)]) == 1
    monkeypatch.setattr(
        numeric, "_verify_instance", lambda _instance, _artifact_dir: True
    )
    assert numeric.main(["--arch", "gfx1151", "--artifact-dir", str(tmp_path)]) == 0

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    with pytest.raises(SystemExit, match="no aot_list.json"):
        numeric.main(["--arch", "gfx1151", "--artifact-dir", str(empty_dir)])


def test_numeric_device_arch_handles_runtime_query_failure(monkeypatch, capsys):
    numeric = _load_numeric_tool()

    def fail_arch():
        raise RuntimeError("no hip")

    monkeypatch.setitem(
        sys.modules,
        "rocke.runtime.hip_module",
        types.SimpleNamespace(get_device_arch=fail_arch),
    )

    assert numeric._device_arch() is None
    assert "unable to query HIP device arch" in capsys.readouterr().out
