#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Pack per-arch rocKE client AOT HSACO into a .kpack + bundle manifest.

Runs after ``rocke_aot_build.py`` has produced the loose per-instance HSACO and
``rocke.aot.sidecar/v1`` sidecars for one architecture. For that architecture it:

1. Reads the copied ``aot_list.json`` and each instance's sidecar.
2. Verifies every HSACO's SHA256 matches its sidecar ``artifact.hsaco_sha256``.
3. Packs each HSACO into one ``rocke_client_<arch>.kpack`` under a deterministic
   ``toc_key`` (``rocke/<op>/<family>/<name>``; Plan 3's loader recomputes it).
4. Emits a ``hipdnn.rocke.bundle/v1`` manifest aggregating the sidecars (single
   source of truth: ``selection``/``launch``/``args_signature`` are carried
   through unchanged) and validates it against ``bundle.schema.json``.

``rocm_kpack`` is consumed from source via ``PYTHONPATH`` (``ROCKE_KPACK_PYTHON_DIR``);
it is never a pip dependency. ``zstandard``/``msgpack`` (its imports) and
``jsonschema`` come from ``library/aot/requirements.txt`` in the rocke-pyenv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from rocke_client_aot.instance_schema import AOT_LIST_FILENAME
from rocke_client_aot.json_schema import load_json_schema, validate_json_schema

BUNDLE_SCHEMA = "hipdnn.rocke.bundle/v1"
PRODUCER = "rocKE"
GROUP_NAME = "rocke_client"
_DEFAULT_LLVM_FLAVOR = "llvm20"


def _parser() -> argparse.ArgumentParser:
    """Create the command-line parser for the rocKE client AOT kpack packer."""

    parser = argparse.ArgumentParser(
        description="Pack rocKE client AOT HSACO into a .kpack and bundle manifest."
    )
    parser.add_argument(
        "--artifact-dir",
        required=True,
        type=Path,
        help="Per-arch build dir holding aot_list.json, HSACO, and sidecars.",
    )
    parser.add_argument(
        "--arch",
        required=True,
        help="Architecture the packed HSACO target; also the kpack arch family.",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=Path,
        help="Directory to write rocke_client_<arch>.kpack and its manifest.",
    )
    parser.add_argument(
        "--engine-build-id",
        default="unknown",
        help="rocKE engine build id recorded in the bundle manifest.",
    )
    parser.add_argument(
        "--llvm-flavor",
        default=os.environ.get("ROCKE_LLVM_FLAVOR", _DEFAULT_LLVM_FLAVOR),
        help="LLVM IR flavor recorded in the bundle manifest (e.g. llvm20/llvm22).",
    )
    parser.add_argument(
        "--bundle-schema",
        type=Path,
        default=None,
        help="Path to bundle.schema.json; defaults to the shared AOT schema.",
    )
    return parser


def toc_key(op: str, family: str, name: str) -> str:
    """Return the deterministic, provider-owned kpack table-of-contents key.

    Plan 3's loader recomputes/reads the same key, so the rule is stable:
    ``rocke/<op>/<family>/<name>``.
    """

    return f"rocke/{op}/{family}/{name}"


def _load_json(path: Path) -> Any:
    """Load a JSON document from disk."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _require_str(value: Any, context: str) -> str:
    """Return a non-empty string or raise with contextual information."""

    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _entry_from_sidecar(
    instance: dict[str, Any], sidecar: dict[str, Any]
) -> tuple[dict[str, Any], str]:
    """Build a manifest entry and its toc_key from an instance + its sidecar.

    ``selection``/``launch``/``args_signature`` are carried through verbatim from
    the ``rocke.aot.sidecar/v1`` sidecar; nothing is re-derived here.
    """

    op = _require_str(instance.get("op"), "instance op")
    family = _require_str(instance.get("family"), "instance family")
    name = _require_str(instance.get("name"), "instance name")

    artifact = sidecar.get("artifact")
    if not isinstance(artifact, dict):
        raise ValueError(f"{name}: sidecar artifact must be an object")
    key = toc_key(op, family, name)
    entry = {
        "cache_key": _require_str(
            sidecar.get("cache_key"), f"{name}: sidecar cache_key"
        ),
        "toc_key": key,
        "symbol": _require_str(artifact.get("symbol"), f"{name}: artifact symbol"),
        "selection": sidecar["selection"],
        "launch": sidecar["launch"],
        "args_signature": sidecar["args_signature"],
    }
    return entry, key


def _read_verified_hsaco(
    artifact_dir: Path, sidecar: dict[str, Any], name: str
) -> bytes:
    """Return the instance HSACO bytes after checking the sidecar digest."""

    artifact = sidecar["artifact"]
    hsaco_path = artifact_dir / _require_str(
        artifact.get("hsaco_filename"), f"{name}: artifact hsaco_filename"
    )
    if not hsaco_path.is_file():
        raise ValueError(f"{name}: HSACO does not exist: {hsaco_path}")
    hsaco = hsaco_path.read_bytes()
    actual_sha = hashlib.sha256(hsaco).hexdigest()
    expected_sha = _require_str(
        artifact.get("hsaco_sha256"), f"{name}: artifact hsaco_sha256"
    )
    if actual_sha != expected_sha:
        raise ValueError(
            f"{name}: packed HSACO sha256 {actual_sha} != sidecar "
            f"artifact.hsaco_sha256 {expected_sha}"
        )
    return hsaco


def _bundle_schema_path(explicit: Path | None) -> Path:
    """Resolve the bundle schema path, defaulting to the shared AOT schema."""

    if explicit is not None:
        return explicit
    return Path(__file__).resolve().parents[1] / "schemas" / "bundle.schema.json"


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    """Write bytes to ``path`` atomically via a sibling temp file."""

    handle, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "wb") as fh:
            fh.write(data)
        os.replace(temp_name, path)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write the bundle manifest using the repository's stable JSON formatting."""

    text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    _atomic_write_bytes(path, text.encode("utf-8"))


def pack_arch(
    *,
    artifact_dir: Path,
    arch: str,
    out_dir: Path,
    engine_build_id: str,
    llvm_flavor: str,
    bundle_schema_path: Path,
) -> tuple[Path, Path]:
    """Pack one architecture's HSACO into a .kpack and write its bundle manifest."""

    from rocm_kpack.compression import ZstdCompressor
    from rocm_kpack.kpack import PackedKernelArchive

    aot_list_path = artifact_dir / AOT_LIST_FILENAME
    if not aot_list_path.is_file():
        raise ValueError(f"no {AOT_LIST_FILENAME} found in {artifact_dir}")
    instances = _load_json(aot_list_path)
    if not isinstance(instances, list) or not instances:
        raise ValueError(f"{aot_list_path} must be a non-empty JSON array")

    archive = PackedKernelArchive(
        group_name=GROUP_NAME,
        gfx_arch_family=arch,
        gfx_arches=[arch],
        compressor=ZstdCompressor(compression_level=3),
    )

    entries: list[dict[str, Any]] = []
    verify_specs: list[tuple[str, str]] = []
    for instance in instances:
        if not isinstance(instance, dict):
            raise ValueError(f"{aot_list_path}: each instance must be an object")
        name = _require_str(instance.get("name"), "instance name")
        if instance.get("arch") != arch:
            raise ValueError(
                f"{name}: instance arch {instance.get('arch')!r} != packer arch {arch!r}"
            )
        sidecar = _load_json(artifact_dir / f"{name}.sidecar.json")
        if not isinstance(sidecar, dict):
            raise ValueError(f"{name}: sidecar must be a JSON object")
        entry, key = _entry_from_sidecar(instance, sidecar)
        hsaco = _read_verified_hsaco(artifact_dir, sidecar, name)
        prepared = archive.prepare_kernel(
            relative_path=key,
            gfx_arch=arch,
            hsaco_data=hsaco,
            metadata={"cache_key": entry["cache_key"], "symbol": entry["symbol"]},
        )
        archive.add_kernel(prepared)
        entries.append(entry)
        verify_specs.append((key, hashlib.sha256(hsaco).hexdigest()))

    out_dir.mkdir(parents=True, exist_ok=True)
    kpack_name = f"{GROUP_NAME}_{arch}.kpack"
    manifest_name = f"{GROUP_NAME}_{arch}.json"
    kpack_path = out_dir / kpack_name

    manifest = {
        "schema": BUNDLE_SCHEMA,
        "producer": PRODUCER,
        "engine_build_id": engine_build_id,
        "llvm_flavor": llvm_flavor,
        "arch": arch,
        "kpack": kpack_name,
        "entries": entries,
    }
    # Validate the manifest BEFORE writing any archive bytes so a schema failure
    # never leaves an orphan .kpack.
    validate_json_schema(
        manifest, load_json_schema(bundle_schema_path), schema_path=bundle_schema_path
    )

    archive.finalize_archive()
    archive.write(kpack_path)

    # Produce-time round-trip self-check: read the written archive back and
    # confirm each kernel's bytes match the sidecar digest, catching zstd/msgpack
    # round-trip or TOC regressions here rather than at runtime hipModuleLoadData.
    written = PackedKernelArchive.read(kpack_path)
    for key, expected_sha in verify_specs:
        packed = written.get_kernel(key, arch)
        actual = hashlib.sha256(packed).hexdigest() if packed is not None else None
        if actual != expected_sha:
            raise ValueError(
                f"{key}: packed archive sha256 {actual} != expected {expected_sha}"
            )

    _write_manifest(out_dir / manifest_name, manifest)
    return kpack_path, out_dir / manifest_name


def main(argv: Sequence[str] | None = None) -> int:
    """Pack the AOT artifacts described by the command-line arguments."""

    try:
        args = _parser().parse_args(argv)
        if not args.artifact_dir.is_dir():
            raise ValueError(f"artifact directory does not exist: {args.artifact_dir}")
        bundle_schema_path = _bundle_schema_path(args.bundle_schema)
        if not bundle_schema_path.is_file():
            raise ValueError(f"bundle schema does not exist: {bundle_schema_path}")
        pack_arch(
            artifact_dir=args.artifact_dir,
            arch=args.arch,
            out_dir=args.out_dir,
            engine_build_id=args.engine_build_id,
            llvm_flavor=args.llvm_flavor,
            bundle_schema_path=bundle_schema_path,
        )
    except SystemExit as exc:
        code = exc.code
        return code if isinstance(code, int) else 2
    except Exception as exc:
        print(f"rocke_kpack_pack: error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
