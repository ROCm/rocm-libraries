# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Round-trip and manifest tests for the rocKE client AOT kpack packer.

``rocm_kpack`` is consumed from source via PYTHONPATH in the build; where it is
unavailable (a kpack-less dev checkout) the packing tests skip, mirroring how the
numeric tests skip without a matching device.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

# The build environment guarantees rocm_kpack (CMake configure gate). When the
# CTest run sets ROCKE_KPACK_REQUIRED, a failed import is a real regression in
# the test env plumbing and must fail loud rather than silently skip the whole
# kpack leg. In a bare dev checkout (flag unset) we skip, like the numeric tests.
if os.environ.get("ROCKE_KPACK_REQUIRED"):
    import rocm_kpack  # noqa: F401
else:
    pytest.importorskip("rocm_kpack")

from rocke_client_aot.json_schema import (  # noqa: E402
    SchemaValidationError,
    load_json_schema,
    validate_json_schema,
)

AOT_DIR = Path(__file__).resolve().parents[1]
SCHEMA_DIR = AOT_DIR / "schemas"
TOOLS_DIR = AOT_DIR / "tools"
ARCH = "gfx942"


def _load_tool(name: str):
    path = TOOLS_DIR / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sidecar(name: str, hsaco: bytes, *, symbol: str, cache_key: str) -> dict:
    return {
        "schema": "rocke.aot.sidecar/v1",
        "cache_key": cache_key,
        "artifact": {
            "hsaco_filename": f"{name}.hsaco",
            "symbol": symbol,
            "hsaco_sha256": hashlib.sha256(hsaco).hexdigest(),
            "hsaco_size": len(hsaco),
        },
        "selection": {
            "op": "sdpa_fwd",
            "arch": ARCH,
            "dtypes": {
                "q": "fp16",
                "k": "fp16",
                "v": "fp16",
                "o": "fp16",
                "acc": "fp32",
            },
            "canonical_layout": "BSHD",
            "shape_constraints": {"head_size": {"equals": 64}},
            "attribute_constraints": {"mask_mode": {"equals": "none"}},
        },
        "launch": {
            "shared_mem_bytes": 0,
            "grid_formula": {
                "x": {"ceil_div": ["seqlen_q", 16]},
                "y": "num_query_heads",
                "z": "batch",
            },
            "block": [64, 1, 1],
            "tile_sizes": {
                "block_q": 16,
                "block_k": 64,
                "head_size": 64,
                "wave_size": 64,
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
            {
                "name": "scale_log2",
                "type": "f32",
                "kind": "scalar",
                "size_bytes": 4,
                "alignment": 4,
            },
        ],
    }


def _instance(name: str) -> dict:
    return {
        "schema": "rocke.aot.instance/v1",
        "name": name,
        "op": "sdpa_fwd",
        "family": "fmha_fwd_mfma",
        "arch": ARCH,
        "compile_spec": {"dtype": "fp16"},
        "selection": {"attribute_constraints": {"mask_mode": {"equals": "none"}}},
        "test_profiles": [{"batch": 1}],
    }


def _build_artifact_dir(tmp_path: Path, count: int = 2) -> tuple[Path, dict[str, dict]]:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    instances = []
    sidecars: dict[str, dict] = {}
    for i in range(count):
        name = f"sdpa_fwd_fmha_fwd_mfma_fp16_bshd_{ARCH}_q64_k64_hq4_hkv4_d64_none_{i}"
        hsaco = f"hsaco-bytes-{i}".encode() * (i + 3)
        (artifact_dir / f"{name}.hsaco").write_bytes(hsaco)
        sc = _sidecar(
            name,
            hsaco,
            symbol=f"rocke_fmha_fwd_mfma_kernel_{i}",
            cache_key=f"sdpa_fwd:fmha_fwd_mfma:{i}",
        )
        _write_json(artifact_dir / f"{name}.sidecar.json", sc)
        instances.append(_instance(name))
        sidecars[name] = sc
    _write_json(artifact_dir / "aot_list.json", instances)
    return artifact_dir, sidecars


def _bundle_schema():
    schema_path = SCHEMA_DIR / "bundle.schema.json"
    return load_json_schema(schema_path), schema_path


def test_toc_key_rule_is_deterministic():
    packer = _load_tool("rocke_kpack_pack")
    assert (
        packer.toc_key("sdpa_fwd", "fmha_fwd_mfma", "inst")
        == "rocke/sdpa_fwd/fmha_fwd_mfma/inst"
    )


def test_pack_arch_round_trips_every_kernel(tmp_path):
    from rocm_kpack.kpack import PackedKernelArchive

    packer = _load_tool("rocke_kpack_pack")
    artifact_dir, sidecars = _build_artifact_dir(tmp_path)
    out_dir = tmp_path / "out"

    kpack_path, manifest_path = packer.pack_arch(
        artifact_dir=artifact_dir,
        arch=ARCH,
        out_dir=out_dir,
        engine_build_id="test-build-id",
        llvm_flavor="llvm22",
        bundle_schema_path=SCHEMA_DIR / "bundle.schema.json",
    )
    assert kpack_path.name == f"rocke_client_{ARCH}.kpack"
    assert manifest_path.name == f"rocke_client_{ARCH}.json"

    archive = PackedKernelArchive.read(kpack_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert len(manifest["entries"]) == len(sidecars)
    for entry in manifest["entries"]:
        name = entry["toc_key"].rsplit("/", 1)[1]
        sidecar = sidecars[name]
        packed = archive.get_kernel(entry["toc_key"], ARCH)
        assert packed is not None
        assert hashlib.sha256(packed).hexdigest() == sidecar["artifact"]["hsaco_sha256"]
        # kpack TOC metadata carries the identity fields.
        toc_meta = archive.toc[entry["toc_key"]][ARCH]["metadata"]
        assert toc_meta["cache_key"] == entry["cache_key"] == sidecar["cache_key"]
        assert toc_meta["symbol"] == entry["symbol"] == sidecar["artifact"]["symbol"]


def test_manifest_validates_and_carries_sidecar_fields_unchanged(tmp_path):
    packer = _load_tool("rocke_kpack_pack")
    artifact_dir, sidecars = _build_artifact_dir(tmp_path)
    out_dir = tmp_path / "out"
    schema, schema_path = _bundle_schema()

    _, manifest_path = packer.pack_arch(
        artifact_dir=artifact_dir,
        arch=ARCH,
        out_dir=out_dir,
        engine_build_id="test-build-id",
        llvm_flavor="llvm22",
        bundle_schema_path=schema_path,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    validate_json_schema(manifest, schema, schema_path=schema_path)
    assert manifest["schema"] == "rocke.aot.bundle/v1"
    assert manifest["producer"] == "rocKE"
    assert manifest["engine_build_id"] == "test-build-id"
    assert manifest["llvm_flavor"] == "llvm22"
    assert manifest["arch"] == ARCH
    assert manifest["kpack"] == f"rocke_client_{ARCH}.kpack"

    for entry in manifest["entries"]:
        name = entry["toc_key"].rsplit("/", 1)[1]
        sidecar = sidecars[name]
        # Single source of truth: nothing is re-derived from the sidecar.
        assert entry["selection"] == sidecar["selection"]
        assert entry["launch"] == sidecar["launch"]
        assert entry["args_signature"] == sidecar["args_signature"]
        assert entry["cache_key"] == sidecar["cache_key"]
        assert entry["symbol"] == sidecar["artifact"]["symbol"]


def test_pack_arch_rejects_hsaco_digest_mismatch(tmp_path):
    packer = _load_tool("rocke_kpack_pack")
    artifact_dir, sidecars = _build_artifact_dir(tmp_path, count=1)
    name = next(iter(sidecars))
    (artifact_dir / f"{name}.hsaco").write_bytes(b"tampered-bytes")

    with pytest.raises(ValueError, match="sha256"):
        packer.pack_arch(
            artifact_dir=artifact_dir,
            arch=ARCH,
            out_dir=tmp_path / "out",
            engine_build_id="id",
            llvm_flavor="llvm20",
            bundle_schema_path=SCHEMA_DIR / "bundle.schema.json",
        )


def test_pack_arch_rejects_missing_hsaco(tmp_path):
    packer = _load_tool("rocke_kpack_pack")
    artifact_dir, sidecars = _build_artifact_dir(tmp_path, count=1)
    name = next(iter(sidecars))
    (artifact_dir / f"{name}.hsaco").unlink()

    with pytest.raises(ValueError, match="HSACO does not exist"):
        packer.pack_arch(
            artifact_dir=artifact_dir,
            arch=ARCH,
            out_dir=tmp_path / "out",
            engine_build_id="id",
            llvm_flavor="llvm20",
            bundle_schema_path=SCHEMA_DIR / "bundle.schema.json",
        )


def test_pack_arch_rejects_arch_mismatch(tmp_path):
    packer = _load_tool("rocke_kpack_pack")
    artifact_dir, _ = _build_artifact_dir(tmp_path, count=1)

    with pytest.raises(ValueError, match="!= packer arch"):
        packer.pack_arch(
            artifact_dir=artifact_dir,
            arch="gfx1151",
            out_dir=tmp_path / "out",
            engine_build_id="id",
            llvm_flavor="llvm20",
            bundle_schema_path=SCHEMA_DIR / "bundle.schema.json",
        )


def test_main_cli_success_and_error_paths(tmp_path, capsys):
    packer = _load_tool("rocke_kpack_pack")
    artifact_dir, _ = _build_artifact_dir(tmp_path, count=1)
    out_dir = tmp_path / "out"

    assert (
        packer.main(
            [
                "--artifact-dir",
                str(artifact_dir),
                "--arch",
                ARCH,
                "--out-dir",
                str(out_dir),
                "--engine-build-id",
                "cli-id",
            ]
        )
        == 0
    )
    assert (out_dir / f"rocke_client_{ARCH}.kpack").is_file()
    manifest_file = out_dir / f"rocke_client_{ARCH}.json"
    assert manifest_file.is_file()
    # Lock the CLI-shipped contract: --engine-build-id flows through and the
    # default llvm_flavor is recorded.
    cli_manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert cli_manifest["engine_build_id"] == "cli-id"
    assert cli_manifest["llvm_flavor"]
    assert cli_manifest["arch"] == ARCH

    missing = tmp_path / "missing"
    assert (
        packer.main(
            ["--artifact-dir", str(missing), "--arch", ARCH, "--out-dir", str(out_dir)]
        )
        == 1
    )
    assert "artifact directory does not exist" in capsys.readouterr().err

    assert (
        packer.main(
            [
                "--artifact-dir",
                str(artifact_dir),
                "--arch",
                ARCH,
                "--out-dir",
                str(out_dir),
                "--bundle-schema",
                str(tmp_path / "nope.json"),
            ]
        )
        == 1
    )
    assert "bundle schema does not exist" in capsys.readouterr().err


def test_manifest_missing_field_fails_schema_validation(tmp_path):
    schema, schema_path = _bundle_schema()
    bad = {
        "schema": "rocke.aot.bundle/v1",
        "producer": "rocKE",
        "engine_build_id": "id",
        "llvm_flavor": "llvm20",
        "arch": ARCH,
        "kpack": "rocke_client_gfx942.kpack",
        "entries": [{"cache_key": "x", "toc_key": "rocke/a/b/c", "symbol": "s"}],
    }
    with pytest.raises(SchemaValidationError):
        validate_json_schema(bad, schema, schema_path=schema_path)


def _pack(packer, artifact_dir, tmp_path):
    return packer.pack_arch(
        artifact_dir=artifact_dir,
        arch=ARCH,
        out_dir=tmp_path / "out",
        engine_build_id="id",
        llvm_flavor="llvm20",
        bundle_schema_path=SCHEMA_DIR / "bundle.schema.json",
    )


def test_pack_arch_rejects_missing_aot_list(tmp_path):
    packer = _load_tool("rocke_kpack_pack")
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no aot_list.json"):
        _pack(packer, empty, tmp_path)


def test_pack_arch_rejects_non_array_aot_list(tmp_path):
    packer = _load_tool("rocke_kpack_pack")
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    _write_json(artifact_dir / "aot_list.json", [])
    with pytest.raises(ValueError, match="non-empty JSON array"):
        _pack(packer, artifact_dir, tmp_path)


def test_pack_arch_rejects_missing_sidecar(tmp_path):
    packer = _load_tool("rocke_kpack_pack")
    artifact_dir, sidecars = _build_artifact_dir(tmp_path, count=1)
    name = next(iter(sidecars))
    (artifact_dir / f"{name}.sidecar.json").unlink()
    with pytest.raises(FileNotFoundError):
        _pack(packer, artifact_dir, tmp_path)
