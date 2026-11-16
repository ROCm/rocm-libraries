# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Descriptors ``gen_descriptors.py`` derives for ``sdpa_bwd_merged``.

The engine reads only the KMD metadata the manifest's knobs spell, and autotune sees
only the UED ``knobs`` list. These tests run ``collect`` and ``build_documents`` over a
synthetic merged manifest: the schema is the one the engine reads (slot metadata), the UED knobs are exactly the
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
    "has_dropout": 0,
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
    "block_q": 64,
    "reduce_symbol": "fmha_bwd_dbias_reduce_gfx950_kernel_1",
}
CONTRACT_FIELDS = {
    "dtype": "string",
    "head_dim": "int",
    "head_dim_v": "int",
    "causal": "int",
    "has_bias": "int",
    "has_dropout": "int",
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
    "block_q": "int",
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
    if op in gd.MAIN_LAUNCH:
        # The tail-zero companion every dQ and merged row launches last.
        (folder / "tz.hsaco").write_bytes(b"tailzero")
        manifest["companions"] = [
            {
                "hash": "a3287842cb25",
                "spec": "tailzero,bf16",
                "file": "tz.hsaco",
                "symbol": "fmha_bwd_tailzero_gfx950_kernel_0",
                "sha256": hashlib.sha256(b"tailzero").hexdigest(),
            }
        ]
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


def test_ued_knobs_are_the_tuning_fields_and_the_alternatives(documents):
    ued = _doc(documents, "sdpa_bwd_merged", "ued")
    # The alternatives are the graph-determined fields merged's kernels differ in while
    # serving one graph: the tile-32 exact and padded rows, packed and unpacked GQA, dB store.
    assert set(ued["knobs"]) == {
        "padded",
        "store_db",
        "gqa_pack",
        "role_order",
        "hints",
        "block_threads",
        "block_kv",
        "block_m",
        "block_q",
    }


def test_op_level_documents_have_a_sibling_ops_shape(documents):
    merged, sibling = "sdpa_bwd_merged", "sdpa_cdna"
    # The dispatch of the rows that launch the dBias reduce is the one document merged has
    # that the sibling lacks.
    assert set(documents[Path(merged)]) == {
        *(n.replace(sibling, merged) for n in documents[Path(sibling)]),
        "flydsl_sdpa_bwd_merged_reduce.udd.json",
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
    # The backward engines read ragged operands and offset multipliers, so they declare the
    # schema of the last of them: 1.4.0, the engine's K_MAX_SUPPORTED_API_VERSION, so a
    # literal, not the generator's table.
    assert ued["sdk_version"] == "1.4.0"


def test_kernel_metadata_carries_every_manifest_knob(documents):
    kdp = documents[Path("sdpa_bwd_merged") / ARCH]["gfx950.reduce.kdp.json"]
    (kernel,) = kdp["kernelDescriptors"]
    assert kernel["metadata"] == {"dtype": "bf16", **MERGED_KNOBS}


def test_a_merged_row_that_stores_db_launches_its_own_object_as_the_reduce(documents):
    folder = documents[Path("sdpa_bwd_merged") / ARCH]
    # The row names a reduce symbol, so it is the `reduce` pack's: no row is left in the plain pack.
    assert folder["flydsl_sdpa_bwd_merged.kdp.json"]["kernelDescriptors"] == []
    pack = folder["gfx950.reduce.kdp.json"]
    (kernel,) = pack["kernelDescriptors"]
    assert list(kernel["sources"]) == ["merged", "dbias_reduce", "tailzero"]
    # The reduce is a second kernel of the row's own object.
    assert kernel["sources"]["dbias_reduce"] == {
        "kind": "hsaco",
        "file": kernel["sources"]["merged"]["file"],
        "symbol": MERGED_KNOBS["reduce_symbol"],
    }
    assert kernel["sources"]["tailzero"]["file"] == "tz.hsaco"
    assert kernel["metadata"]["block_q"] == 64
    udd = documents[Path("sdpa_bwd_merged")]["flydsl_sdpa_bwd_merged_reduce.udd.json"]
    assert pack["dispatch"] == udd["id"]
    assert udd["dispatch_symbol"] == "hipkernel.flydsl_sdpa_bwd_merged_reduce.dispatch"


# ---------------------------------------------------------------------------
# collect(): the companions a row launches, by its bias dtype and its value-head role
# ---------------------------------------------------------------------------

TAILZERO = "fmha_bwd_tailzero_gfx950_kernel_0"
TAILZERO_HV = "fmha_bwd_tailzero_hv_gfx950_kernel_0"
DQ_U2 = "fmha_bwd_dq_u2_gfx950_kernel_0"
REDUCE_16 = "fmha_bwd_dbias_reduce_gfx11_kernel_0"
REDUCE_F32 = "fmha_bwd_dbias_reduce_f32_gfx11_kernel_0"


def _dq_manifest(content: Path, rows: list[dict], companions: list[dict]) -> None:
    """A dQ manifest of @p rows (``name``, ``spec``, ``knobs``) and @p companions (``spec``, ``symbol``), with an object each."""
    folder = content / "sdpa_bwd_dq" / ARCH
    folder.mkdir(parents=True)
    instances = []
    for row in rows:
        data = row["name"].encode()
        (folder / f"{row['name']}.hsaco").write_bytes(data)
        instances.append(
            {
                "name": row["name"],
                "file": f"{row['name']}.hsaco",
                "sha256": hashlib.sha256(data).hexdigest(),
                "symbol": row.get("symbol", f"sym_{row['name']}"),
                "priority": 0,
                "dtype": row.get("dtype", "bf16"),
                "spec": row["spec"],
                "knobs": row["knobs"],
            }
        )
    entries = []
    for index, companion in enumerate(companions):
        data = f"companion{index}".encode()
        (folder / f"c{index}.hsaco").write_bytes(data)
        entries.append(
            {
                "hash": f"c{index}",
                "spec": companion["spec"],
                "file": f"c{index}.hsaco",
                "symbol": companion["symbol"],
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    manifest = {
        "op": "sdpa_bwd_dq",
        "arch": ARCH,
        "toolchain": {"inputs_sha256": "synthetic"},
        "instances": instances,
        "companions": entries,
    }
    (folder / gd.MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


@pytest.fixture
def synthetic_objects(monkeypatch):
    monkeypatch.setattr(gd, "inputs_digest", lambda op, arch: "synthetic")
    monkeypatch.setattr(gd, "describe", lambda data, where, symbol: None)
    monkeypatch.setattr(gd, "verify_arch", lambda *args: None)
    monkeypatch.setattr(gd, "verify_generic", lambda *args: None)


def _dq_knobs(**fields) -> dict:
    return {
        "head_dim": 64,
        "has_bias": 0,
        "reduce_symbol": "",
        "bias_f32": -1,
        "u2": "auto",
        **fields,
    }


COMPANIONS = [
    {"spec": "tailzero,bf16", "symbol": TAILZERO},
    {"spec": "tailzero_hv,bf16", "symbol": TAILZERO_HV},
    {"spec": "dbias_reduce,bf16", "symbol": REDUCE_16},
    {"spec": "dbias_reduce,fp32", "symbol": REDUCE_F32},
]


def test_a_value_head_dq_row_launches_the_v_head_tailzero_and_the_others_the_standard_one(
    tmp_path, synthetic_objects
) -> None:
    _dq_manifest(
        tmp_path,
        [
            {
                "name": "std",
                "spec": "dq,bf16,64,plain,bk,0,0,tail,0,1",
                "knobs": _dq_knobs(),
            },
            {
                "name": "hv",
                "spec": "dq,bf16,64,plain,bk,0,0,tail,0,1;u2=dq",
                "symbol": DQ_U2,
                "knobs": _dq_knobs(u2="dq"),
            },
        ],
        COMPANIONS,
    )
    std, hv = gd.collect(tmp_path, ARCH)
    assert std.companions["tailzero"]["symbol"] == TAILZERO
    assert hv.companions["tailzero"]["symbol"] == TAILZERO_HV
    assert gd.REDUCE_ROLE not in std.companions and gd.REDUCE_ROLE not in hv.companions


def test_a_float_bias_row_launches_the_float_dbias_reduce(
    tmp_path, synthetic_objects
) -> None:
    bias = "dq,bf16,64,bias,bk,0,0,tail,0,1;bias_f32=1"
    _dq_manifest(
        tmp_path,
        [
            {
                "name": "f32bias",
                "spec": bias,
                "knobs": _dq_knobs(has_bias=1, bias_f32=1, reduce_symbol=REDUCE_F32),
            },
            {
                "name": "bias16",
                "spec": "dq,bf16,64,bias,bk,0,0,tail,0,1",
                "knobs": _dq_knobs(has_bias=1, reduce_symbol=REDUCE_16),
            },
        ],
        COMPANIONS,
    )
    f32, sixteen = gd.collect(tmp_path, ARCH)
    assert f32.companions[gd.REDUCE_ROLE]["symbol"] == REDUCE_F32
    assert sixteen.companions[gd.REDUCE_ROLE]["symbol"] == REDUCE_16


def test_a_float_bias_row_naming_the_16_bit_reduce_is_refused(
    tmp_path, synthetic_objects
) -> None:
    _dq_manifest(
        tmp_path,
        [
            {
                "name": "f32bias",
                "spec": "dq,bf16,64,bias,bk,0,0,tail,0,1;bias_f32=1",
                "knobs": _dq_knobs(has_bias=1, bias_f32=1, reduce_symbol=REDUCE_16),
            }
        ],
        COMPANIONS,
    )
    with pytest.raises(ValueError, match="reduce symbol"):
        gd.collect(tmp_path, ARCH)


def test_a_value_head_dq_row_without_the_v_head_tailzero_is_refused(
    tmp_path, synthetic_objects
) -> None:
    _dq_manifest(
        tmp_path,
        [
            {
                "name": "hv",
                "spec": "dq,bf16,64,plain,bk,0,0,tail,0,1;u2=dq",
                "symbol": DQ_U2,
                "knobs": _dq_knobs(u2="dq"),
            }
        ],
        [c for c in COMPANIONS if c["spec"] != "tailzero_hv,bf16"],
    )
    with pytest.raises(ValueError, match="tailzero_hv"):
        gd.collect(tmp_path, ARCH)


def test_a_value_head_row_that_defines_another_kernels_symbol_is_refused(
    tmp_path, synthetic_objects
) -> None:
    """A u2 object has a kernel of its own: the plain dQ symbol on a ``u2=dq`` row is a build that did not take the u2 path."""
    _dq_manifest(
        tmp_path,
        [
            {
                "name": "hv",
                "spec": "dq,bf16,64,plain,bk,0,0,tail,0,1;u2=dq",
                "symbol": "fmha_bwd_dq_gfx950_kernel_0",
                "knobs": _dq_knobs(u2="dq"),
            }
        ],
        COMPANIONS,
    )
    with pytest.raises(ValueError, match=DQ_U2):
        gd.collect(tmp_path, ARCH)


def test_a_value_head_dq_row_ships_in_the_dq_u2_pack_and_the_standard_row_in_the_plain_one(
    tmp_path, synthetic_objects
) -> None:
    """Collected from a manifest, the pair of rows packs apart: the V-head row on the ``dq_u2`` dispatch as ``{dq, tailzero}``."""
    _dq_manifest(
        tmp_path,
        [
            {
                "name": "std",
                "spec": "dq,bf16,64,plain,bk,0,0,tail,0,1",
                "knobs": _dq_knobs(),
            },
            {
                "name": "hv",
                "spec": "dq,bf16,64,plain,bk,0,0,tail,0,1;u2=dq",
                "symbol": DQ_U2,
                "knobs": _dq_knobs(u2="dq"),
            },
        ],
        COMPANIONS,
    )
    entries = gd.collect(tmp_path, ARCH)
    ops = {op: [e for e in entries if e.op == op] for op in gd.SPLIT_SLOT_OPS}
    assert [gd.extra_pack(e) for e in ops["sdpa_bwd_dq"]] == [None, "u2"]
    pack = gd._arch_documents(
        ops["sdpa_bwd_dq"],
        gd.OpDescriptors("sdpa_bwd_dq", gd._fields_for_op(ops["sdpa_bwd_dq"])),
        ARCH,
    )
    assert sorted(pack) == ["flydsl_sdpa_bwd_dq.kdp.json", f"{ARCH}.dq_u2.kdp.json"]
    (hv,) = pack[f"{ARCH}.dq_u2.kdp.json"]["kernelDescriptors"]
    assert list(hv["sources"]) == ["dq", "tailzero"]
    assert hv["sources"]["dq"]["symbol"] == DQ_U2
    assert hv["sources"]["tailzero"]["symbol"] == TAILZERO_HV
    assert hv["metadata"]["u2_role"] == "dq"
