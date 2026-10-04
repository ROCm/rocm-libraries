# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import dataclasses
import os
import struct
import zlib
from pathlib import Path

import numpy as np
import pytest

from lib import features as fs
from lib import mlrec
from lib.hardware import DATATYPE_VALUE, arch_constants
from lib.subcells import SplitRule, split_tree_from_labels

WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / "hipblaslt"
LFS_POINTER = b"version https://git-lfs"

LABELS = [
    "Large|Large|LargeK|Bnone",
    "Mid|Mid|MidK|Bnone#M<=300",
    "Mid|Mid|MidK|Bnone#M>300#N<=200",
    "Mid|Mid|MidK|Bnone#M>300#N>200",
    "Tiny|Small|TinyK|Bany",
]


def random_bundle(labels, seed=0, dims=(4, 6, 3), signatures=None):
    """In-memory bundle with random weights and statistics for `labels`."""
    rng = np.random.default_rng(seed)
    q, i, x = (
        len(n)
        for n in (
            fs.query_feature_names(),
            fs.item_feature_names(),
            fs.interaction_feature_names(),
        )
    )
    embed, hidden, inter = dims
    models = {}
    for label in labels:
        shapes = mlrec.tensor_shapes(q, i, x, embed, hidden, inter)
        sd = {k: rng.standard_normal(s).astype(np.float32) for k, s in shapes.items()}
        sd["temperature"] = np.asarray([0.5 + rng.random()], dtype=np.float32)
        norms = {}
        for key, n in (("q", q), ("i", i), ("x", x)):
            norms[f"{key}_mean"] = rng.standard_normal(n).tolist()
            norms[f"{key}_std"] = (0.5 + rng.random(n)).tolist()
        models[label] = {
            "state_dict": sd,
            "feature_norms": norms,
            "embed_dim": embed,
            "hidden_dim": hidden,
            "inter_hidden": inter,
            "smart_k_signatures": (signatures or {}).get(
                label,
                [[128, 128, 64, 16, 16, 32, 0, 0], [64, 64, 64, 16, 16, 32, 0, 4]],
            ),
        }
    return {
        "q_names": fs.query_feature_names(),
        "i_names": fs.item_feature_names(),
        "x_names": fs.interaction_feature_names(),
        "models": models,
    }


def expected_weight(a, dtype):
    """Reference dequantized values, written independently of lib/mlrec."""
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    if dtype == "fp32":
        return a
    if dtype == "bf16":
        import torch

        return torch.from_numpy(a.copy()).to(torch.bfloat16).to(torch.float32).numpy()
    levels = 127 if dtype == "int8" else 7
    amax = float(np.abs(a).max()) if a.size else 0.0
    scale = (amax / levels) if amax > 0 else 1.0
    q = np.clip(np.round(a / scale), -levels, levels).astype(np.int32)
    return np.float32(scale) * q.astype(np.float32)


def v1_from_v2(data, trailer=True):
    """The MLREC_v1 file holding a v2 file's payload."""
    h = mlrec.read_header(data)
    header_size = struct.unpack_from("<I", data, 12)[0]
    payload = data[header_size:-8]
    out = b"MLREC_v1" + struct.pack("<IIB", 1, 0x01020304, h["weight_dtype"])
    for s in (h["feature_catalog_hash"], h["arch"]):
        out += struct.pack("<I", len(s)) + s.encode()
    out += struct.pack("<IIIII", *h["dims"], h["n_cells"], h["n_splits"])
    return out + payload + (b"MLRECEND" if trailer else b"")


def shipped_models():
    out = []
    for p in sorted(WEIGHTS_DIR.glob("*/*/*.tilewright.bin")):
        data = p.read_bytes()
        if not data.startswith(LFS_POINTER):
            out.append((p, data))
    return out


@pytest.mark.parametrize("dtype", ["fp32", "bf16", "int8", "int4"])
def test_round_trip(dtype):
    bundle = random_bundle(LABELS, seed=1)
    consts = arch_constants("gfx950")
    data = mlrec.write_model(bundle, None, "gfx950", consts, dtype)
    p = mlrec.read_model(data)
    assert p["version"] == 2 and p["weight_dtype_name"] == dtype
    assert p["arch"] == "gfx950"
    assert p["feature_catalog_hash"] == fs.feature_names_hash()
    assert p["dims"] == (55, 12, 37)
    assert p["constants"] == consts
    assert [c["label"] for c in p["cells"]] == sorted(LABELS)
    expected_tree = split_tree_from_labels(LABELS)
    assert {r.cell: r for r in p["splits"]} == expected_tree
    for c in p["cells"]:
        src = bundle["models"][c["label"]]
        for k, v in src["feature_norms"].items():
            np.testing.assert_array_equal(c["feature_norms"][k], np.float32(v))
        assert [list(s) for s in c["smart_k_signatures"]] == src["smart_k_signatures"]
        for name in mlrec.WEIGHT_ORDER:
            got = c["state_dict"][name].reshape(-1)
            want = (
                expected_weight(src["state_dict"][name], dtype)
                if name.endswith(".weight")
                else np.asarray(src["state_dict"][name], np.float32).reshape(-1)
            )
            assert got.view(np.uint32).tolist() == want.view(np.uint32).tolist(), name
        assert c["temperature"] == float(src["state_dict"]["temperature"][0])
    again = mlrec.write_model(mlrec.bundle_from_model(p), None, "gfx950", consts, dtype)
    assert again == data


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 63, 64])
@pytest.mark.parametrize("dtype", ["int4", "int8", "bf16"])
def test_weight_codec_lengths(n, dtype):
    a = np.random.default_rng(n).standard_normal(n).astype(np.float32)
    code = mlrec.WEIGHT_DTYPES[dtype]
    buf = mlrec.encode_weight(a, code)
    assert len(buf) == mlrec.encoded_weight_size(n, code)
    got = mlrec.decode_weight(buf, n, code)
    assert (
        got.view(np.uint32).tolist()
        == expected_weight(a, dtype).view(np.uint32).tolist()
    )


def test_int4_nibble_layout():
    a = np.asarray([7.0, -7.0, 0.0], dtype=np.float32)
    buf = mlrec.encode_weight(a, mlrec.WEIGHT_DTYPES["int4"])
    assert struct.unpack_from("<f", buf)[0] == 1.0
    assert buf[4:] == bytes([(1 << 4) | 15, 8])


def test_bf16_rounding_matches_torch():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    a = rng.standard_normal(4096).astype(np.float32)
    halves = (np.arange(64, dtype=np.uint32) << 16) | 0x8000
    a = np.concatenate([a, halves.view(np.float32), np.float32([0.0, -0.0, 3.0e38])])
    want = torch.from_numpy(a.copy()).to(torch.bfloat16).view(torch.uint16).numpy()
    assert mlrec.bf16_bits(a).tolist() == want.tolist()


def test_header_layout():
    data = mlrec.write_model(
        random_bundle(LABELS[:2]),
        None,
        "gfx1250v0",
        arch_constants("gfx1250v0"),
        "int8",
    )
    assert data[:8] == b"MLREC_v2" and data[-8:] == b"MLRECEND"
    endian, header_size, payload_size, crc, wdt = struct.unpack_from("<IIQIB", data, 8)
    assert endian == 0x01020304 and wdt == 2 and data[29:32] == b"\0\0\0"
    assert header_size + payload_size + 8 == len(data)
    assert zlib.crc32(data[header_size:-8]) == crc
    off = 52
    for expect in (fs.feature_names_hash(), "gfx1250v0"):
        (n,) = struct.unpack_from("<H", data, off)
        assert data[off + 2 : off + 2 + n].decode() == expect
        off += 2 + n
    pmc, c0, c1, c2, mi_default, n_mi = struct.unpack_from("<dddddI", data, off)
    assert (pmc, c0, c1, c2, mi_default, n_mi) == (4.0, 0.0, 0.016, 0.0, 32.0, 19)
    entries = [
        struct.unpack_from("<IIIid", data, off + 44 + 24 * j) for j in range(n_mi)
    ]
    assert entries == sorted(entries)
    assert (16, 16, 128, DATATYPE_VALUE["float4"], 4.0) in entries
    assert off + 44 + 24 * n_mi == header_size


def test_read_rejects_corruption():
    data = mlrec.write_model(
        random_bundle(LABELS[:1]), None, "gfx950", arch_constants("gfx950"), "fp32"
    )
    bad = [
        data[:-1],
        data[:-8] + b"MLRECENX",
        data[:-20] + bytes([data[-20] ^ 1]) + data[-19:],
        data[:30] + b"\1" + data[31:],
        b"MLREC_v3" + data[8:],
        data[:40] + b"\x01\x00\x00\x00" + data[44:],
    ]
    for b in bad:
        with pytest.raises(mlrec.FormatError):
            mlrec.read_model(b)


def test_write_rejects_bad_bundles():
    consts = arch_constants("gfx950")

    def write(b):
        return mlrec.write_model(b, None, "gfx950", consts, "bf16")

    b = random_bundle(LABELS[:1])
    b["q_names"] = list(reversed(b["q_names"]))
    with pytest.raises(ValueError):
        write(b)
    b = random_bundle(LABELS[:1])
    b["models"][LABELS[0]]["state_dict"]["q_proj.0.weight"][0, 0] = np.nan
    with pytest.raises(ValueError):
        write(b)
    b = random_bundle(LABELS[:1])
    b["models"][LABELS[0]]["state_dict"]["i_proj.2.bias"] = np.zeros(2, np.float32)
    with pytest.raises(ValueError):
        write(b)
    b = random_bundle(LABELS[:1], signatures={LABELS[0]: [[1, 2, 3]]})
    with pytest.raises(ValueError):
        write(b)
    with pytest.raises(ValueError):
        write(random_bundle(["Tiny|Tiny|TinyK|Bnone#M<=x"]))
    with pytest.raises(ValueError):
        mlrec.write_model(random_bundle(LABELS[:1]), None, "gfx950", consts, "fp16")


def _constants(**kw):
    return dataclasses.replace(arch_constants("gfx950"), **kw)


def _with_weight(value, name="q_proj.0.weight"):
    b = random_bundle(LABELS[:1])
    b["models"][LABELS[0]]["state_dict"][name][0, 0] = value
    return b


def _with_dim(value):
    b = random_bundle(LABELS[:1])
    b["models"][LABELS[0]]["embed_dim"] = value
    return b


SPLIT = SplitRule("A", "M", 10, "A#M<=10", "A#M>10")
BAD_WRITES = {
    "arch with a space": dict(arch="gfx 950"),
    "empty arch": dict(arch=""),
    "non-ASCII arch": dict(arch="gfx950\u00e9"),
    "parallel_mi_cu 0": dict(constants=_constants(parallel_mi_cu=0.0)),
    "mi_default 0": dict(constants=_constants(mi_default=0.0)),
    "mi_default nan": dict(constants=_constants(mi_default=float("nan"))),
    "MI cycles 0": dict(constants=_constants(mi_table={(16, 16, 4, "float"): 0.0})),
    "MI cycles inf": dict(
        constants=_constants(mi_table={(16, 16, 4, "float"): float("inf")})
    ),
    "fnuz MI dtype": dict(
        constants=_constants(mi_table={(16, 16, 32, "float8_fnuz"): 16.0})
    ),
    "unknown MI dtype": dict(
        constants=_constants(mi_table={(16, 16, 32, "fp8"): 16.0})
    ),
    "negative MI shape": dict(
        constants=_constants(mi_table={(-16, 16, 32, "float8"): 16.0})
    ),
    "too many MI entries": dict(
        constants=_constants(mi_table={(i, 16, 4, "float"): 8.0 for i in range(1025)})
    ),
    "cell label with a space": dict(bundle=random_bundle(["Tiny Tiny"])),
    "layer width 0": dict(bundle=_with_dim(0)),
    "layer width above 65536": dict(bundle=_with_dim(65537)),
    "bf16 overflow": dict(bundle=_with_weight(3.4e38), dtype="bf16"),
    "int8 scale overflow": dict(bundle=_with_weight(3.4e38), dtype="int8"),
    "int4 scale overflow": dict(bundle=_with_weight(3.0e38), dtype="int4"),
    "infinite weight": dict(bundle=_with_weight(np.inf), dtype="fp32"),
    "split threshold beyond i32": dict(
        splits=[dataclasses.replace(SPLIT, threshold=2**31)]
    ),
    "split label with a space": dict(
        splits=[dataclasses.replace(SPLIT, lo_label="A#M<=10 ")]
    ),
    "cyclic split tree": dict(
        splits=[SPLIT, SplitRule("A#M<=10", "N", 5, "A", "A#M<=10#N>5")]
    ),
}


@pytest.mark.parametrize("case", sorted(BAD_WRITES))
def test_writer_rejects_what_the_loader_rejects(case):
    kw = dict(BAD_WRITES[case])
    with pytest.raises(ValueError):
        mlrec.write_model(
            kw.pop("bundle", random_bundle(LABELS[:1])),
            kw.pop("splits", None),
            kw.pop("arch", "gfx950"),
            kw.pop("constants", arch_constants("gfx950")),
            kw.pop("dtype", "bf16"),
        )


def test_writer_accepts_the_loader_limits():
    b = _with_weight(3.0e38)
    for dtype in ("fp32", "int8"):
        mlrec.write_model(b, None, "gfx950", arch_constants("gfx950"), dtype)
    big = _constants(mi_table={(i, 16, 4, "float"): 8.0 for i in range(1024)})
    mlrec.write_model(random_bundle(LABELS[:1]), None, "gfx950", big, "fp32")
    labelled = random_bundle(["!|~#M<=1"])
    mlrec.write_model(labelled, None, "gfx950:xnack+", arch_constants("gfx950"), "fp32")


def _patched_header(data, old, new):
    """`data` with the header bytes `old` replaced by `new` (same length; the
    payload CRC does not cover the header)."""
    header_size = struct.unpack_from("<I", data, 12)[0]
    head = data[:header_size]
    assert len(old) == len(new) and head.count(old) == 1
    return head.replace(old, new) + data[header_size:]


def test_engine_rejects_the_header_values_the_writer_refuses():
    tw = pytest.importorskip("tilewright")
    consts = _constants(mi_table={(16, 16, 32, "float8"): 16.0})
    data = mlrec.write_model(random_bundle(LABELS[:1]), None, "gfx950", consts, "fp32")
    tw.load_model_from_memory(data)
    float8, fnuz = DATATYPE_VALUE["float8"], DATATYPE_VALUE["float8_fnuz"]
    for old, new in (
        (b"gfx950", b"gfx 50"),
        (struct.pack("<dI", 32.0, 1), struct.pack("<dI", 0.0, 1)),
        (struct.pack("<id", float8, 16.0), struct.pack("<id", fnuz, 16.0)),
        (struct.pack("<id", float8, 16.0), struct.pack("<id", float8, 0.0)),
    ):
        with pytest.raises(ValueError):
            tw.load_model_from_memory(_patched_header(data, old, new))


def test_converter_checks_the_payload_and_catalog():
    consts = arch_constants("gfx950")
    v1 = v1_from_v2(
        mlrec.write_model(random_bundle(LABELS[:1]), None, "gfx950", consts, "fp32")
    )
    label = LABELS[0].encode()
    spaced = v1.replace(label, label[:-1] + b" ")
    with pytest.raises(ValueError, match="cell label"):
        mlrec.convert_v1_to_v2(spaced, consts)
    stale = v1.replace(fs.feature_names_hash().encode(), b"0123456789abcdef")
    with pytest.raises(ValueError, match="catalog hash"):
        mlrec.convert_v1_to_v2(stale, consts)


@pytest.mark.parametrize("trailer", [True, False])
def test_v1_read_and_convert(trailer):
    consts = arch_constants("gfx950")
    v2 = mlrec.write_model(
        random_bundle(LABELS, seed=4), None, "gfx950", consts, "int4"
    )
    v1 = v1_from_v2(v2, trailer=trailer)
    p1, p2 = mlrec.read_model(v1), mlrec.read_model(v2)
    assert p1["version"] == 1 and p1["has_trailer"] is trailer
    assert p1["payload_crc32"] == p2["payload_crc32"]
    assert [c["label"] for c in p1["cells"]] == [c["label"] for c in p2["cells"]]
    assert mlrec.convert_v1_to_v2(v1, consts) == v2
    with pytest.raises(mlrec.FormatError):
        mlrec.convert_v1_to_v2(v2, consts)


def test_engine_loads_written_models():
    tw = pytest.importorskip("tilewright")
    for dtype in ("fp32", "bf16", "int8", "int4"):
        data = mlrec.write_model(
            random_bundle(LABELS, seed=2),
            None,
            "gfx950",
            arch_constants("gfx950"),
            dtype,
        )
        info = tw.describe(tw.load_model_from_memory(data))
        assert info.arch == "gfx950" and info.n_cells == len(LABELS)
        assert info.n_splits == len(split_tree_from_labels(LABELS))
        assert int(info.weight_type) == mlrec.WEIGHT_DTYPES[dtype]


def test_shipped_models_convert_and_rewrite_byte_for_byte():
    models = shipped_models()
    if not models:
        pytest.skip("shipped model files are not materialized")
    for path, data in models:
        parsed = mlrec.read_model(data)
        consts = arch_constants(parsed["arch"])
        assert parsed["constants"] == consts, path.name
        assert mlrec.convert_v1_to_v2(v1_from_v2(data), consts) == data, path.name
        rewritten = mlrec.write_model(
            mlrec.bundle_from_model(parsed),
            None,
            parsed["arch"],
            consts,
            parsed["weight_dtype"],
        )
        assert rewritten == data, path.name


def test_convert_original_v1_files():
    root = os.environ.get("TILEWRIGHT_V1_MODELS_DIR")
    if not root:
        pytest.skip("set TILEWRIGHT_V1_MODELS_DIR to <dir>/<arch>/*.tilewright.bin")
    n = 0
    for v1 in sorted(Path(root).glob("*/*.tilewright.bin")):
        v2 = WEIGHTS_DIR / v1.parent.name / v1.parent.name / v1.name
        if not v2.exists() or v2.read_bytes().startswith(LFS_POINTER):
            continue
        data = v1.read_bytes()
        consts = arch_constants(mlrec.read_header(data)["arch"])
        assert mlrec.convert_v1_to_v2(data, consts) == v2.read_bytes(), v1.name
        n += 1
    assert n > 0
