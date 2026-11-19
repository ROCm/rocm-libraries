# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""tools/ground_truth.py layouts: strided Q/K/V/O graphs for the catalog sweep.

The strides are pinned three ways that do not share code with the formulas: hand
computed values, the rocKE stride vectors file (the C++ twin's oracle), and the
rocKE host admission predicate. A layout the kernels would decline is a sweep that
times nothing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))

import ground_truth as gt  # noqa: E402
import registry_enumerator as re_  # noqa: E402
import sweep as sweep_tool  # noqa: E402

_CONFIGS = _ROOT / "configs"
_MANIFESTS = {
    a: _CONFIGS / f"{a}_attention_dense.engine.yaml" for a in ("gfx950", "gfx942")
}
_ROCKE = re_._REPO / "dnn-providers" / "hip-kernel-provider" / "rocke"
_VECTORS = (
    _ROCKE / "platform" / "tests" / "golden" / "attention_dense_stride_vectors.json"
)
_BSHD_FIXTURES = _ROOT / "tests" / "fixtures" / "ground_truth"

_TENSORS = {"q": "query", "k": "key", "v": "value", "o": "output"}
_LAYOUTS = sorted(gt.LAYOUTS)


def _request(**overrides):
    request = dict(batch=2, nhead_q=8, nhead_k=2, seqlen_q=64, seqlen_k=64)
    request |= dict(hdim_q=128, hdim_v=128, dtype="bf16", mask_type="NO_MASK")
    return request | overrides


def _row(number=1, layout=None, **overrides):
    row = dict(request=_request(**overrides))
    row["_provenance"] = {"source": "layouts", "row": number}
    return row if layout is None else row | {"layout": layout}


@pytest.fixture(scope="module")
def template():
    manifest = gt.load_manifest(str(_MANIFESTS["gfx950"]))
    return gt.ground_truth_block(manifest, _MANIFESTS["gfx950"])


def _strides(template, layout, **overrides) -> dict:
    graph = gt.instantiate(*template, _row(layout=layout, **overrides))
    return {t["name"]: t["strides"] for t in graph["tensors"]}


# --- the template is unchanged for BSHD ------------------------------------------------


@pytest.mark.parametrize(
    "fixture, number, request_fields",
    [
        (
            "bshd_gqa_d128_causal",
            1,
            dict(
                batch=2,
                nhead_q=32,
                nhead_k=8,
                seqlen_q=2048,
                seqlen_k=2048,
                mask_type="TOP_LEFT_CAUSAL",
                dtype="bf16",
                hdim_q=128,
                hdim_v=128,
            ),
        ),
        (
            "bshd_mha_d64_plain",
            2,
            dict(
                batch=1,
                nhead_q=16,
                nhead_k=16,
                seqlen_q=512,
                seqlen_k=512,
                mask_type="NO_MASK",
                dtype="fp16",
                hdim_q=64,
                hdim_v=64,
            ),
        ),
    ],
)
def test_a_row_without_a_layout_gives_the_graph_bytes_it_gave_before(
    tmp_path, template, fixture, number, request_fields
):
    """The fixture is `graphs` output of the code before `layouts` existed."""
    row = dict(request=request_fields, time_ms=1.0, accuracy={}, device="gfx950")
    row["_provenance"] = {"source": "ground_truth", "row": number}
    gt_path, out = tmp_path / "GT.json", tmp_path / "graphs"
    gt_path.write_text(json.dumps([row]))
    argv = f"graphs --gt {gt_path} --manifest {_MANIFESTS['gfx950']} --out {out}"
    assert gt.main(argv.split()) == 0
    expected = (_BSHD_FIXTURES / f"{fixture}.graph.json").read_text()
    assert (out / f"ground_truth_r{number:04d}.json").read_text() == expected


@pytest.mark.parametrize("shape", [dict(), dict(nhead_k=8, seqlen_k=96, hdim_v=32)])
def test_the_bshd_layout_is_the_templates_own_packed_bshd(template, shape):
    plain = gt.instantiate(*template, _row(**shape))
    named = gt.instantiate(*template, _row(layout="bshd", **shape))
    assert json.dumps(plain, indent=4) == json.dumps(named, indent=4)


@pytest.mark.parametrize("layout", [None, *_LAYOUTS])
def test_no_graph_carries_the_template_only_layout_role(template, layout):
    graph = gt.instantiate(*template, _row(layout=layout))
    assert all(
        "layout_role" not in t and "stride_order" not in t for t in graph["tensors"]
    )
    assert all(isinstance(t["strides"], list) for t in graph["tensors"])


# --- hand-computed strides: B2 Hq8 Hkv2 S64 D128 ----------------------------------------

_BSHD_Q, _BSHD_K = [65536, 128, 1024, 1], [16384, 128, 256, 1]
_HAND = {
    "bshd": dict(query=_BSHD_Q, key=_BSHD_K, value=_BSHD_K, output=_BSHD_Q),
    "bhsd": dict(
        query=[65536, 8192, 128, 1],
        key=[16384, 8192, 128, 1],
        value=[16384, 8192, 128, 1],
        output=[65536, 8192, 128, 1],
    ),
    # one [2, 64, 12, 128] buffer: 8 + 2 * 2 heads, token stride 12 * 128
    "packed_qkv": dict(
        query=[98304, 128, 1536, 1],
        key=[98304, 128, 1536, 1],
        value=[98304, 128, 1536, 1],
        output=_BSHD_Q,
    ),
    # one [2, 64, 4, 128] buffer for K and V
    "packed_kv": dict(
        query=_BSHD_Q,
        key=[32768, 128, 512, 1],
        value=[32768, 128, 512, 1],
        output=_BSHD_Q,
    ),
    # the odd batches of [4, ...]: batch stride doubled
    "batch_slice": dict(
        query=[131072, 128, 1024, 1],
        key=[32768, 128, 256, 1],
        value=[32768, 128, 256, 1],
        output=_BSHD_Q,
    ),
}


def test_every_layout_has_a_hand_computed_case():
    assert sorted(_HAND) == _LAYOUTS


@pytest.mark.parametrize("layout", _LAYOUTS)
def test_layout_strides_match_the_hand_computed_values(template, layout):
    assert _strides(template, layout) == _HAND[layout]


def test_packed_qkv_of_equal_heads_is_the_b_s_3_h_d_buffer(template):
    got = _strides(template, "packed_qkv", nhead_q=8, nhead_k=8)
    assert (
        got["query"]
        == got["key"]
        == got["value"]
        == [3 * 64 * 8 * 128, 128, 3 * 8 * 128, 1]
    )


def test_batch_slice_doubles_only_the_batch_stride_of_q_k_v(template):
    base, sliced = _strides(template, "bshd"), _strides(template, "batch_slice")
    for name in ("query", "key", "value"):
        assert sliced[name][0] == 2 * base[name][0]
        assert sliced[name][1:] == base[name][1:]
    assert sliced["output"] == base["output"]


# --- the rocKE vectors file: strides the C++ twin is pinned to -------------------------

_VECTOR_CASES = {
    "admit_packed_bshd_gqa_d128_batch2": "bshd",
    "admit_bhsd_all_four": "bhsd",
    "admit_bhsd_all_four_batch1": "bhsd",
    "admit_packed_qkv_one_buffer": "packed_qkv",
    "admit_packed_qkv_one_buffer_batch1": "packed_qkv",
    "admit_packed_kv_one_buffer_gqa": "packed_kv",
}


@pytest.mark.parametrize("case_name, layout", _VECTOR_CASES.items())
def test_layouts_reproduce_the_vectors_files_admitted_strides(
    template, case_name, layout
):
    if not _VECTORS.is_file():
        pytest.skip(f"rocKE stride vectors not in this checkout: {_VECTORS}")
    case = next(
        c for c in json.loads(_VECTORS.read_text())["cases"] if c["name"] == case_name
    )
    problem = case["problem"]
    got = _strides(
        template,
        layout,
        batch=problem["batch"],
        nhead_q=problem["num_query_heads"],
        nhead_k=problem["num_kv_heads"],
        seqlen_q=problem["seqlen_q"],
        seqlen_k=problem["seqlen_kv"],
        hdim_q=problem["head_size"],
        hdim_v=problem["head_size"],
    )
    assert got == {name: case[role]["strides"] for role, name in _TENSORS.items()}


# --- the rocKE host admission predicate ------------------------------------------------

_SHAPES = {
    "gqa_d128": dict(batch=2, nhead_q=32, nhead_k=8, seqlen_q=2048, seqlen_k=2048),
    "mha_d128": dict(batch=1, nhead_q=32, nhead_k=32, seqlen_q=4096, seqlen_k=4096),
    "tp_d64": dict(
        batch=2,
        nhead_q=16,
        nhead_k=2,
        seqlen_q=1024,
        seqlen_k=1024,
        hdim_q=64,
        hdim_v=64,
    ),
    "mqa_d64": dict(
        batch=3, nhead_q=8, nhead_k=1, seqlen_q=256, seqlen_k=256, hdim_q=64, hdim_v=64
    ),
}


@pytest.fixture(scope="module")
def admit():
    """rocKE's `kernels.common._attention_dense_strides`, or skip."""
    paths = [str(_ROCKE / "library"), str(_ROCKE / "platform" / "python")]
    if not all(Path(p).is_dir() for p in paths):
        pytest.skip(f"rocKE not in this checkout: {_ROCKE}")
    with pytest.MonkeyPatch.context() as patch:
        for path in paths:
            patch.syspath_prepend(path)
        try:
            import kernels.common._attention_dense_strides as module
        except ImportError as exc:
            pytest.skip(f"rocKE strides module does not import: {exc}")
    return module


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("layout", _LAYOUTS)
def test_every_layout_is_admitted_by_the_rocke_predicate(
    template, admit, layout, shape
):
    request = _request(**_SHAPES[shape])
    graph = gt.instantiate(*template, _row(layout=layout, **_SHAPES[shape]))
    spec = SimpleNamespace(
        batch=request["batch"],
        seqlen_q=request["seqlen_q"],
        seqlen_kv=request["seqlen_k"],
        num_query_heads=request["nhead_q"],
        num_kv_heads=request["nhead_k"],
        head_size=request["hdim_q"],
        block_n=64,
    )
    operands = {
        role: admit.DenseOperand(tuple(t["dims"]), tuple(t["strides"]))
        for role, t in zip("qkvo", graph["tensors"])
    }
    verdict = admit.dense_strides_admit(spec, operands)
    assert verdict.ok, (verdict.reason, verdict.detail)
    # a layout that only canonicalizes to the BSHD twin is a duplicate measurement
    twin = admit.packed_launch_strides(spec)
    if layout == "bshd":
        assert dict(verdict.strides) == twin
    elif layout != "batch_slice" or request["batch"] > 1:
        assert dict(verdict.strides) != twin


# --- refusals --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "layout, overrides, message",
    [
        ("diagonal", {}, "layout 'diagonal' is not in"),
        ("packed_qkv", dict(seqlen_k=96), "needs one seqlen and head size in qkv"),
        ("packed_qkv", dict(hdim_v=64), "needs one seqlen and head size in qkv"),
        ("packed_kv", dict(hdim_v=64), "needs one seqlen and head size in kv"),
    ],
)
def test_an_unbuildable_layout_is_refused_by_name(template, layout, overrides, message):
    with pytest.raises(gt.GroundTruthError, match=message):
        gt.instantiate(*template, _row(layout=layout, **overrides))


def test_a_template_without_roles_cannot_take_a_layout(template):
    graph_template, value_map = template
    bare = json.loads(json.dumps(graph_template))
    for tensor in bare["tensors"]:
        del tensor["layout_role"]
    with pytest.raises(gt.GroundTruthError, match="no q/k/v/o layout_role"):
        gt.instantiate(bare, value_map, _row(layout="bhsd"))
    assert gt.instantiate(bare, value_map, _row())["tensors"][0]["strides"] == _BSHD_Q


# --- the `layouts` subcommand -----------------------------------------------------------


def _spec(**overrides) -> dict:
    spec = dict(
        dtypes=["bf16", "fp16"],
        head_dims=[64, 128],
        mask_types=["NO_MASK", "TOP_LEFT_CAUSAL"],
        layouts=list(gt.LAYOUTS),
        shapes=[
            dict(id="gqa", batch=2, nhead_q=32, nhead_k=8, seqlen_q=2048),
            dict(id="mha", batch=1, nhead_q=32, nhead_k=32, seqlen_q=4096),
            dict(id="tp", batch=2, nhead_q=16, nhead_k=2, seqlen_q=1024),
        ],
    )
    return spec | overrides


def _layouts(tmp_path, arch="gfx950", **overrides):
    shapes, out = tmp_path / "shapes.yaml", tmp_path / "graphs"
    shapes.write_text(yaml.safe_dump(_spec(**overrides)))
    argv = f"layouts --manifest {_MANIFESTS[arch]} --shapes {shapes} --out {out}"
    return gt.main(argv.split()), out


@pytest.mark.parametrize("arch", ["gfx950", "gfx942"])
def test_the_default_cell_set_gives_120_graphs_and_a_join_map(tmp_path, arch):
    rc, out = _layouts(tmp_path, arch)
    assert rc == 0
    mapping = yaml.safe_load(gt.layouts_map_path(out).read_text())
    assert len(mapping) == len(list(out.glob("*.json"))) == 2 * 2 * 2 * 3 * 5
    assert [m["graph_name"] for m in mapping] == [
        f"layouts_r{n:04d}" for n in range(1, 121)
    ]
    graphs = {p.stem: json.loads(p.read_text()) for p in out.glob("*.json")}
    for entry in mapping:
        graph = graphs[entry["graph_name"]]
        assert entry["semantic_identity"] == sweep_tool.semantic_identity(graph)
        assert entry["strides"] == {t["name"]: t["strides"] for t in graph["tensors"]}
        assert graph["_provenance"]["source"] == "layouts"


def test_the_map_names_family_shape_and_layout_of_every_graph(tmp_path):
    _, out = _layouts(tmp_path)
    mapping = yaml.safe_load(gt.layouts_map_path(out).read_text())
    cells = {
        (m["family"]["dtype"], m["family"]["head_size"], m["family"]["mask_type"])
        for m in mapping
    }
    assert len(cells) == 8
    per_cell = {
        c: [
            (m["shape"], m["layout"])
            for m in mapping
            if (
                m["family"]["dtype"],
                m["family"]["head_size"],
                m["family"]["mask_type"],
            )
            == c
        ]
        for c in cells
    }
    assert all(len(set(v)) == 15 for v in per_cell.values())
    first = mapping[0]
    assert first["request"]["hdim_q"] == first["family"]["head_size"] == 64
    assert (first["shape"], first["layout"]) == ("gqa", "bshd")


def test_two_graphs_share_an_identity_only_when_the_map_says_they_repeat_the_twin(
    tmp_path,
):
    _, out = _layouts(tmp_path)
    mapping = yaml.safe_load(gt.layouts_map_path(out).read_text())
    by_identity = {}
    for m in mapping:
        by_identity.setdefault(m["semantic_identity"], []).append(m)
    for group in by_identity.values():
        if len(group) > 1:
            assert {m["layout"] for m in group} == {"bshd", "batch_slice"}
            assert {m["shape"] for m in group} == {"mha"}  # batch 1: one slab, no step
    flagged = {(m["shape"], m["layout"]) for m in mapping if m["same_as_bshd"]}
    assert flagged == {
        ("gqa", "bshd"),
        ("mha", "bshd"),
        ("tp", "bshd"),
        ("mha", "batch_slice"),
    }


def test_the_graphs_are_a_valid_sweep_corpus(tmp_path):
    _, out = _layouts(tmp_path)
    inventory = sweep_tool.corpus_inventory(
        dict(path=str(out), expected_graphs=120), "none"
    )
    assert len(inventory) == 120 and len({g["graph_name"] for g in inventory}) == 120


def test_layouts_refuse_a_directory_that_already_holds_files(tmp_path, capsys):
    (tmp_path / "graphs").mkdir()
    (tmp_path / "graphs" / "old.json").write_text("{}")
    rc, _ = _layouts(tmp_path)
    assert rc == 2 and "is not empty" in capsys.readouterr().err


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(layouts=["bshd", "zigzag"]), "unknown layout(s) ['zigzag']"),
        (
            dict(shapes=[dict(id="a", batch=1, nhead_q=8, nhead_k=8, seqlen_q=64)] * 2),
            "not unique",
        ),
        (
            dict(shapes=[dict(id="a", batch=1, nhead_q=8, seqlen_q=64)]),
            "bad or missing ['nhead_k']",
        ),
        (
            dict(
                shapes=[
                    dict(id="a", batch=1, nhead_q=8, nhead_k=8, seqlen_q=64, extra=1)
                ]
            ),
            "bad or missing ['extra']",
        ),
    ],
)
def test_a_malformed_shapes_file_is_refused(tmp_path, capsys, overrides, message):
    rc, out = _layouts(tmp_path, **overrides)
    assert rc == 2 and message in capsys.readouterr().err
    assert not out.exists()


def test_a_shapes_file_with_a_missing_or_extra_key_is_refused(tmp_path, capsys):
    spec = _spec()
    del spec["layouts"]
    shapes = tmp_path / "shapes.yaml"
    shapes.write_text(yaml.safe_dump(spec))
    argv = f"layouts --manifest {_MANIFESTS['gfx950']} --shapes {shapes} --out {tmp_path / 'g'}"
    assert gt.main(argv.split()) == 2
    assert "exactly" in capsys.readouterr().err


def test_seqlen_k_defaults_to_seqlen_q_and_can_differ(tmp_path):
    shapes = [dict(id="a", batch=1, nhead_q=8, nhead_k=8, seqlen_q=64)]
    shapes += [dict(id="b", batch=1, nhead_q=8, nhead_k=8, seqlen_q=64, seqlen_k=128)]
    _, out = _layouts(
        tmp_path,
        shapes=shapes,
        layouts=["bhsd"],
        dtypes=["bf16"],
        head_dims=[64],
        mask_types=["NO_MASK"],
    )
    mapping = yaml.safe_load(gt.layouts_map_path(out).read_text())
    assert [m["request"]["seqlen_k"] for m in mapping] == [64, 128]
