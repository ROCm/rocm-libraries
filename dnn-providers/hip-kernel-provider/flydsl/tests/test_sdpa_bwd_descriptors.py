# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Descriptors ``gen_descriptors.py`` derives for ``sdpa_bwd_merged``.

The engine reads only the KMD metadata the manifest's knobs spell, and autotune sees
only the UED ``knobs`` list. These tests run ``collect`` and ``build_documents`` over a
synthetic merged manifest: the schema is the 4A contract, the UED knobs are exactly the
tuning fields (schedule and launch geometry), and the op-level documents have the keys of a sibling op's.
Host-only: no object is compiled and no content tree is needed.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

FLYDSL = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(FLYDSL))
import gen_descriptors as gd  # noqa: E402

ARCH = "gfx950"

# The merged KMD fields, in manifest order, with their metadata type tokens.
MERGED_KNOBS = {
    "head_dim": 128,
    "head_dim_v": 128,
    "causal": 1,
    "has_bias": 1,
    "dropout": 0,
    "padded": 0,
    "hints": 1,
    "floor": "tail",
    "dform": "bk",
    "role_order": "auto",
    "store_db": -1,
    "gqa_pack": 1,
    "block_threads": 512,
    "block_kv": 128,
    "block_m": 64,
    "reduce_symbol": "fmha_bwd_dbias_reduce_gfx950_kernel_1",
}
CONTRACT_FIELDS = {
    "dtype": "string",
    "head_dim": "int",
    "head_dim_v": "int",
    "causal": "int",
    "has_bias": "int",
    "dropout": "int",
    "padded": "int",
    "hints": "int",
    "floor": "string",
    "dform": "string",
    "role_order": "string",
    "store_db": "int",
    "gqa_pack": "int",
    "block_threads": "int",
    "block_kv": "int",
    "block_m": "int",
    "reduce_symbol": "string",
}
SIBLING_KNOBS = {"head_dim": 128, "causal": 0, "block_threads": 256, "block_m": 64}


def _write_manifest(content: Path, op: str, knobs: dict, symbol: str) -> None:
    folder = content / op / ARCH
    folder.mkdir(parents=True)
    data = op.encode()
    (folder / "k.hsaco").write_bytes(data)
    record = {
        "name": f"{op}_k",
        "file": "k.hsaco",
        "sha256": hashlib.sha256(data).hexdigest(),
        "symbol": symbol,
        "priority": 0,
        "dtype": "bf16",
        "knobs": knobs,
    }
    manifest = {
        "op": op,
        "arch": ARCH,
        "toolchain": {"inputs_sha256": "synthetic"},
        "instances": [record],
    }
    (folder / gd.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def documents(tmp_path, monkeypatch):
    """``build_documents`` over a merged and a sibling attention manifest."""
    monkeypatch.setattr(gd, "inputs_digest", lambda op, arch: "synthetic")
    monkeypatch.setattr(gd, "describe", lambda data, where, symbol: None)
    monkeypatch.setattr(gd, "verify_arch", lambda *args: None)
    monkeypatch.setattr(gd, "verify_generic", lambda *args: None)
    _write_manifest(tmp_path, "sdpa_bwd_merged", MERGED_KNOBS, "fmha_bwd_merged")
    _write_manifest(tmp_path, "sdpa_cdna", SIBLING_KNOBS, "sdpa_cdna_kernel")
    folders = gd.build_documents(gd.collect(tmp_path, ARCH), ARCH)
    return folders


def _doc(folders, op: str, suffix: str) -> dict:
    return folders[Path(op)][f"flydsl_{op}.{suffix}.json"]


def test_kmd_fields_are_the_contract(documents):
    kmd = _doc(documents, "sdpa_bwd_merged", "kmd")
    assert {f["name"]: f["type"] for f in kmd["fields"]} == CONTRACT_FIELDS


def test_ued_knobs_are_the_tuning_fields_only(documents):
    ued = _doc(documents, "sdpa_bwd_merged", "ued")
    assert set(ued["knobs"]) == {
        "role_order",
        "hints",
        "block_threads",
        "block_kv",
        "block_m",
    }


def test_op_level_documents_have_a_sibling_ops_shape(documents):
    merged, sibling = "sdpa_bwd_merged", "sdpa_cdna"
    assert set(documents[Path(merged)]) == {
        n.replace(sibling, merged) for n in documents[Path(sibling)]
    }
    for suffix in ("ued", "uhd"):
        assert set(_doc(documents, merged, suffix)) == set(
            _doc(documents, sibling, suffix)
        )

    uhd = _doc(documents, merged, "uhd")
    assert uhd["adapter"] == "native"
    assert uhd["native"] == {"symbol": "hipkernel.flydsl_sdpa_bwd_merged.score"}
    ued = _doc(documents, merged, "ued")
    assert ued["name"] == "hipkernel:flydsl_sdpa_bwd_merged"
    assert ued["sort_kernel_catalog"] == {"default": uhd["id"]}
    assert ued["sdk_version"] == _doc(documents, sibling, "ued")["sdk_version"]


def test_kernel_metadata_carries_every_manifest_knob(documents):
    kdp = documents[Path("sdpa_bwd_merged") / ARCH]["flydsl_sdpa_bwd_merged.kdp.json"]
    (kernel,) = kdp["kernelDescriptors"]
    assert kernel["metadata"] == {"dtype": "bf16", **MERGED_KNOBS}
