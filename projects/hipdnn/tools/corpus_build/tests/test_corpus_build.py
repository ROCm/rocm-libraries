# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""What a corpus has to be true of before anyone spends cluster time measuring it.

Four properties, and each of them has a specific way of going wrong silently:

  deduplication   the same problem arriving from two sources is one problem. Emitted
                  twice it is measured twice, which inflates the corpus and quietly
                  doubles the weight of whichever regime it lands in.
  regime tagging  a graph that does not say which population it belongs to leaves
                  `uhd_gen evaluate` reporting RFC 0019.13 §11.2's per-regime table as
                  UNAVAILABLE, which is the state this tool exists to end.
  determinism     a corpus that cannot be rebuilt cannot be audited, and a training
                  run whose problem set cannot be reproduced cannot be compared to the
                  next one.
  graph validity  a graph the backend refuses at `from_binary` looks fine on disk. The
                  round trip here reads every emitted graph back with the repository's
                  OTHER hipDNN graph reader (`mine_shapes.from_graph_corpus`) and
                  checks it describes the shape the manifest claims -- an independent
                  parser, so agreement is evidence rather than a restatement.
"""
from __future__ import annotations

import collections
import json
import uuid
from pathlib import Path

import pytest

from corpus_build import assemble, build as pipeline, graphs, kernels, model_shapes, sweep
from corpus_build.__main__ import main
from corpus_build.shapes import Filter, Shape

REPO = pipeline.REPO
SHIPPED = REPO / "dnn-providers/integration-tests/integration-test-bundles/quick/SdpaFwd/bshd"

#: Geometries for the pack fixture: (dtype, head_size, hq, hkv, sq, skv, batch, causal,
#: how many kernels claim it). The last one is below `--min-candidates` on purpose.
_PACK_GEOMETRIES = [
    ("BF16", 64, 8, 8, 512, 512, 1, 0, 3),
    ("BF16", 128, 32, 8, 2048, 2048, 1, 1, 4),
    ("FP16", 64, 16, 1, 1, 4096, 8, 1, 3),
    ("BF16", 64, 4, 4, 512, 256, 2, 0, 3),
    ("BF16", 64, 8, 8, 128, 128, 1, 0, 2),
]

_CATALOG = """# A catalog

### SDPA drill-down

| model | D | q/kv heads | causal | routes now? | if not, why |
|---|:--:|:--:|:--:|:--:|---|
| tinybert | 64 | 12/12 | no | yes | - |
| tinyllama | 128 | 32/**8** | **yes** | yes | - |
| mystery | 64 | 8/8 | no | yes | - |

## Working entries

### `tinybert_ab.py` - encoder
- **Validated (dim=768 H12 seq256 L6, bf16):** `Sq=Skv=256, D=64`.

### `tinyllama_ab.py` - decoder
- **Validated (dim=2048 H32/KV8 seq512 L4, bf16 + f16):** routes.
"""


def _pack(path: Path) -> Path:
    descriptors = []
    for dtype, head_size, hq, hkv, sq, skv, batch, causal, count in _PACK_GEOMETRIES:
        for index in range(count):
            descriptors.append({
                "version": "1.0", "id": f"{dtype}-{head_size}-{sq}-{index}",
                "name": f"kernel_{len(descriptors)}", "priority": 0,
                "metadata": {"dtype": dtype, "head_size": head_size,
                             "num_query_heads": hq, "num_kv_heads": hkv,
                             "seqlen_q": sq, "seqlen_kv": skv, "batch": batch,
                             "causal": causal, "block_m": 128 + index}})
    path.write_text(json.dumps({"version": "1.0", "id": "pack", "name": "test pack",
                                "arch": ["gfx942"], "kernelDescriptors": descriptors}),
                    encoding="utf-8")
    return path


@pytest.fixture
def sources(tmp_path):
    """A pack and a catalog small enough to reason about, and a place for shape files."""
    root = tmp_path / "packs"
    root.mkdir()
    _pack(root / "test_attention.kdp.json")
    catalog = tmp_path / "MODEL_CATALOG.md"
    catalog.write_text(_CATALOG, encoding="utf-8")
    shapes_dir = tmp_path / "model-shapes"
    shapes_dir.mkdir()
    return {"kdp_roots": [root], "catalog": catalog, "shapes": shapes_dir}


def _build(tmp_path, sources, **overrides):
    options = dict(count=40, seed=0, kdp_roots=sources["kdp_roots"],
                   catalog=sources["catalog"], batches=(1,))
    options.update(overrides)
    out = tmp_path / options.pop("name", "corpus")
    return pipeline.build(out, **options), out


# --------------------------------------------------------------------- deduplication


def test_a_shape_two_sources_describe_is_emitted_once(tmp_path, sources):
    """The pack's `bf16 b1 hq8 hkv8 sq512 skv512 d64` geometry, published again as a
    model shape. It is one problem: one graph, one manifest row, and the recorded
    provenance is the model's -- `assemble.SOURCES` puts an observation ahead of a
    packed geometry, because that is the more informative thing to have written down.
    """
    (sources["shapes"] / "published.csv").write_text(
        "model,batch,heads_q,heads_kv,seq_q,seq_kv,head_dim,dtype,mask\n"
        "twin,1,8,8,512,512,64,bf16,none\n", encoding="utf-8")
    manifest, out = _build(tmp_path, sources, shape_dirs=[sources["shapes"]])

    twins = [record for record in manifest["graphs"]
             if (record["batch"], record["heads_q"], record["seqlen_q"],
                 record["head_dim"], record["dtype"]) == (1, 8, 512, 64, "bf16")
             and not record["causal"] and record["seqlen_kv"] == 512]
    assert len(twins) == 1, f"the same shape was emitted twice: {twins}"
    assert twins[0]["source"] == "model"
    assert manifest["duplicates_dropped"]["kernel"] == 1
    assert len(list((out / "graphs").glob("*.json"))) == manifest["emitted"]


def test_every_emitted_shape_tuple_is_distinct(tmp_path, sources):
    """The corpus-wide invariant behind the test above, over all three sources at once."""
    manifest, _ = _build(tmp_path, sources, count=200)
    keys = [(row["op"], row["dtype"], row["batch"], row["heads_q"], row["heads_kv"],
             row["seqlen_q"], row["seqlen_kv"], row["head_dim"], row["causal"],
             row["alignment"])
            for row in manifest["graphs"]]
    assert len(set(keys)) == len(keys)
    assert len({row["name"] for row in manifest["graphs"]}) == len(keys)


# -------------------------------------------------------------------- regime tagging


@pytest.mark.parametrize("shape,expected,causal", [
    (dict(seqlen_q=1, seqlen_kv=512, heads_q=32, heads_kv=8), "decode_short_gqa", True),
    (dict(seqlen_q=1, seqlen_kv=32768, heads_q=32, heads_kv=1), "decode_long_mqa", True),
    (dict(seqlen_q=512, seqlen_kv=512, heads_q=12, heads_kv=12), "prefill_short_mha", True),
    (dict(seqlen_q=8192, seqlen_kv=8192, heads_q=64, heads_kv=8), "prefill_long_gqa", True),
    (dict(seqlen_q=256, seqlen_kv=4096, heads_q=16, heads_kv=16), "append_long_mha", True),
    # Cross attention is never causal: the two sequences come from different tensors, so
    # there is no diagonal to mask against -- and Shape refuses the combination.
    (dict(seqlen_q=512, seqlen_kv=256, heads_q=8, heads_kv=8), "cross_short_mha", False),
])
def test_the_regime_names_the_population_a_problem_belongs_to(shape, expected, causal):
    """Phase, context length and head grouping, which are the three axes a heuristic
    can be excellent on one side of and useless on the other. `2048` is the boundary
    because it is the middle bucket of the declaration's own `seqlen_k` regimes."""
    assert Shape(dtype="bf16", batch=1, head_dim=128, causal=causal, **shape).regime == expected


def test_a_causal_cross_attention_shape_is_refused_rather_than_emitted_unlabellable():
    """`sdpa_fwd.opmeta.json` counts a causal problem's work as Sq*Sk - Sq*(Sq-1)/2,
    which goes non-positive once the queries outrun the keys. The graph is legal and an
    engine will run it; what cannot exist is a training row, because the label is derived
    from that count -- "full-graph graph.flops must be a positive finite number" ended
    AITER's 843-graph collection (run 67929589) on exactly these shapes."""
    with pytest.raises(ValueError, match="causal cross attention"):
        Shape(dtype="bf16", batch=1, head_dim=128, causal=True,
              seqlen_q=512, seqlen_kv=256, heads_q=8, heads_kv=8)


def test_the_regime_travels_on_the_graph_as_well_as_the_manifest(tmp_path, sources):
    """In the name, because a graph gets separated from its manifest -- quoted in a
    bench invocation, named in a log, joined to a result row by `benchmark` -- and a
    graph that cannot say which population it is in is a row §11.2 cannot bucket."""
    manifest, out = _build(tmp_path, sources)
    assert manifest["regimes"], "no regime was recorded at all"
    for record in manifest["graphs"]:
        assert record["regime"] in record["name"]
        assert (out / record["file"]).is_file()
        assert json.loads((out / record["file"]).read_text())["name"] == record["name"]
    assert sum(manifest["regimes"].values()) == manifest["emitted"]


def test_the_manifest_column_is_one_evaluate_will_find(tmp_path, sources):
    """`uhd_gen evaluate` discovers the regime column by name and reports the
    per-regime table as UNAVAILABLE when none of the names it knows is present. A
    manifest spelling it `corpus_regime` or `sdpa_regime` would be silently useless."""
    evaluate = pytest.importorskip("uhd_gen.evaluate")
    manifest, out = _build(tmp_path, sources)
    header = (out / "manifest.csv").read_text(encoding="utf-8").splitlines()[0].split(",")
    assert "regime" in header
    assert "benchmark" in header
    assert set(header) & set(evaluate.REGIME_COLUMN_CANDIDATES)
    assert len(manifest["graphs"]) == len(
        (out / "manifest.csv").read_text(encoding="utf-8").strip().splitlines()) - 1


# ----------------------------------------------------------------------- determinism


def test_the_same_seed_rebuilds_the_same_corpus(tmp_path, sources):
    first, first_out = _build(tmp_path, sources, count=60, name="a")
    second, second_out = _build(tmp_path, sources, count=60, name="b")
    assert first["graphs"] == second["graphs"]
    assert (sorted(path.name for path in (first_out / "graphs").glob("*.json"))
            == sorted(path.name for path in (second_out / "graphs").glob("*.json")))
    for record in first["graphs"]:
        assert ((first_out / record["file"]).read_bytes()
                == (second_out / record["file"]).read_bytes())


def test_a_different_seed_moves_the_sampled_shapes_and_nothing_else(tmp_path, sources):
    """The seed is the sampler's, not the corpus's: the packed geometries and the
    recorded model shapes are enumerated, not drawn, so they must not move. A seed
    that reshuffled them would make two corpora incomparable for no reason."""
    first, _ = _build(tmp_path, sources, count=60, seed=0, name="a")
    second, _ = _build(tmp_path, sources, count=60, seed=7, name="b")

    def by_source(manifest, source):
        return [row["name"] for row in manifest["graphs"] if row["source"] == source]

    assert by_source(first, "kernel") == by_source(second, "kernel")
    assert by_source(first, "model") == by_source(second, "model")
    assert by_source(first, "sweep") != by_source(second, "sweep")


# --------------------------------------------------------------------- facet filter


def test_a_filtered_corpus_carries_only_the_facets_it_was_asked_for(tmp_path, sources):
    """Every source obeys one gate. The pack fixture publishes `d64` geometries and the
    declaration sweeps both dtypes, so a corpus asked for bf16/d128 that still contains
    a packed `d64` row would be filtering by provenance rather than by shape -- and the
    engine it was narrowed for would decline exactly those rows."""
    (sources["shapes"] / "published.csv").write_text(
        "model,batch,heads_q,heads_kv,seq_q,seq_kv,head_dim,dtype,mask\n"
        "wide,1,8,8,512,512,128,bf16,none\n"
        "narrow,1,8,8,512,512,64,bf16,none\n", encoding="utf-8")
    manifest, out = _build(tmp_path, sources, count=60, shape_dirs=[sources["shapes"]],
                           keep=Filter(dtypes=("bf16",), head_dims=(128,)))

    assert manifest["emitted"] > 0
    assert {(row["dtype"], row["head_dim"]) for row in manifest["graphs"]} == {("bf16", 128)}
    assert {row["source"] for row in manifest["graphs"]} >= {"model", "sweep"}
    assert manifest["reports"]["filtered_out"]["model"] >= 1
    assert manifest["reports"]["filtered_out"]["kernel"] >= 1
    assert manifest["reports"]["filter"] == {"dtypes": ["bf16"], "head_dims": [128], "causal": []}


def test_a_filtered_sweep_still_fills_the_count_it_was_given(tmp_path, sources):
    """The filter is applied while drawing, not after. A narrow gate cut post-hoc would
    hand back a fraction of the requested corpus -- here roughly an eighth, one dtype of
    two against one head dim of four -- which is how a 24-graph validation corpus turns
    into three graphs without anything reporting an error."""
    wide, _ = _build(tmp_path, sources, count=40, name="wide")
    narrow, _ = _build(tmp_path, sources, count=40, name="narrow",
                       keep=Filter(dtypes=("bf16",), head_dims=(128,)))

    assert narrow["emitted"] == wide["emitted"] == 40
    assert narrow["reports"]["sweep"]["filtered"] > 0
    assert narrow["reports"]["sweep"]["shapes"] >= wide["reports"]["sweep"]["shapes"]


# --------------------------------------------------------------------- graph validity


@pytest.fixture(scope="module")
def real_corpus(tmp_path_factory):
    """A corpus from the tree's real inputs: the shipped packs, catalog and declaration."""
    out = tmp_path_factory.mktemp("real")
    return pipeline.build(out, count=120, seed=0), out


def test_no_emitted_graph_pins_the_mma_core_mode(real_corpus):
    """AITER's `SdpaFwdPlanBuilder::isApplicable` declines any graph that sets
    `mma_core_mode` -- in its own words, "mma_core_mode must be unset". The shipped
    bundles carry `"float"`, and a corpus that inherited it reported that engine as
    serving 0 of 24 bf16/d128 graphs its own kernel table serves (run 67928822). An
    engine-agnostic corpus must not decide the contest in the graph document."""
    _, out = real_corpus
    for path in sorted((out / "graphs").glob("*.json")):
        node = json.loads(path.read_text(encoding="utf-8"))["nodes"][0]
        assert node["attributes"]["mma_core_mode"] is None, path.name


def test_every_causal_shape_is_carried_at_both_diagonal_anchors(real_corpus):
    """A causal problem is served by different kernels depending on where its diagonal
    is anchored -- AITER's gfx942 forward table has bottom-right causal kernels and no
    top-left ones, and served 0 of 15 causal graphs in run 67928906 for exactly that
    reason. Both anchors must reach the corpus, and the graph document must say which:
    `SdpaPlanUtils::getMaskType` reads `diagonal_alignment` off the bounds trio."""
    manifest, out = real_corpus
    causal = [row for row in manifest["graphs"] if row["causal"]]
    assert causal, "the tree's own inputs carry causal shapes"
    anchors = collections.Counter(row["alignment"] for row in causal)
    assert anchors["top_left"] > 0 and anchors["bottom_right"] > 0

    written = {row["alignment"]: json.loads(
        (out / row["file"]).read_text(encoding="utf-8"))["nodes"][0]["attributes"]
        for row in causal}
    assert written["bottom_right"]["diagonal_alignment"] == "BOTTOM_RIGHT"
    assert written["top_left"]["diagonal_alignment"] == "TOP_LEFT"
    for attributes in written.values():
        assert (attributes["left_bound"], attributes["right_bound"]) == (-1, 0)


def test_every_emitted_graph_reads_back_as_the_shape_it_claims(real_corpus):
    """Read back by `mine_shapes.from_graph_corpus`, which is the repository's other
    hipDNN graph reader and derives causality from `left_bound`/`causal_mask` rather
    than from the file name. A graph this parser cannot read, or reads as a different
    shape, is one the corpus is lying about."""
    manifest, out = real_corpus
    mined = model_shapes.mine.from_graph_corpus(out / "graphs")
    assert len(mined) == manifest["emitted"]

    by_name = {record["name"]: record for record in manifest["graphs"]}
    for record in mined:
        claimed = by_name[record["_provenance"]["graph"]]
        assert (record["batch"], record["nhead_q"], record["nhead_k"], record["seqlen_q"],
                record["seqlen_k"], record["hdim_q"], record["dtype"]) == (
            claimed["batch"], claimed["heads_q"], claimed["heads_kv"],
            claimed["seqlen_q"], claimed["seqlen_kv"], claimed["head_dim"],
            claimed["dtype"])
        assert record["mask_type"] == (1 if claimed["causal"] else 0)


@pytest.mark.skipif(not SHIPPED.is_dir(), reason="shipped SdpaFwd bundles not present")
def test_emitted_graphs_carry_the_shipped_bundles_own_keys(real_corpus):
    """Key presence, not just values: `json::to<Graph>` rejects an SDPA node whose
    attention-window keys are absent rather than null, and a rejected bundle is logged
    as INVALID_GRAPH_SCHEMA and silently not registered -- a hole in a sweep that
    looked like it ran everything."""
    _, out = real_corpus
    shipped = json.loads(next(SHIPPED.rglob("*.json")).read_text())
    expected_attributes = set(shipped["nodes"][0]["attributes"])
    expected_top = set(shipped) | {"name"}

    for path in sorted((out / "graphs").glob("*.json")):
        graph = json.loads(path.read_text())
        assert set(graph) == expected_top, path.name
        node = graph["nodes"][0]
        assert node["type"] == shipped["nodes"][0]["type"]
        assert set(node["attributes"]) == expected_attributes, path.name
        assert set(node["inputs"]) == set(shipped["nodes"][0]["inputs"])
        assert set(node["outputs"]) == set(shipped["nodes"][0]["outputs"])
        assert graph["io_data_type"] in ("bfloat16", "half"), (
            "the FlatBuffer enum spells fp16 `half`; `float16` produces a graph the "
            "backend accepts on disk and refuses at from_binary")


def test_the_manifest_id_survives_the_write_and_read_uhd_gen_performs(real_corpus):
    """`uhd_gen generate` reads an ID-less JSON graph off disk, canonicalises what it
    read and mints a UUID5 of it -- that id becomes the corpus's `benchmark` column,
    which is what the manifest is joined on. The id is computed here from the document
    before it is written, so the property that matters is that the written file
    canonicalises to the same thing: a float that reprs differently after a round trip
    would silently break every join."""
    manifest, out = real_corpus
    for record in manifest["graphs"][:50]:
        reread = json.loads((out / record["file"]).read_text(encoding="utf-8"))
        canonical = json.dumps(reread, sort_keys=True, separators=(",", ":"),
                               allow_nan=False)
        assert record["benchmark"] == str(
            uuid.uuid5(uuid.NAMESPACE_URL, "hipdnn:graph:" + canonical))


# --------------------------------------------------------------------------- sources


def test_the_pack_supplies_only_geometries_worth_ranking(tmp_path, sources):
    """A geometry two kernels claim has nothing to rank between, so it teaches a
    ranking model nothing and is not worth the measurement."""
    found, [stats] = kernels.collect(
        kernels.discover(sources["kdp_roots"]), min_candidates=3, max_bytes=2 ** 31)
    assert stats["geometries"] == len(_PACK_GEOMETRIES)
    assert stats["too_few_candidates"] == 1
    assert len(found) == len(_PACK_GEOMETRIES) - 1
    assert all(shape.seqlen_q != 128 for shape in (item.shape for item in found))


def test_the_catalog_yields_the_geometries_it_records():
    """Against the real `MODEL_CATALOG.md`, because that is the file that will be
    edited. Every value asserted is one the catalog states: llama's 32/8 GQA at D=64
    causal, gpt2 validated in both dtypes, and the two entries that record no usable
    geometry reported as skipped rather than guessed into existence."""
    found, stats = model_shapes.from_catalog(
        REPO / model_shapes.DEFAULT_CATALOG, batches=(1,))
    shapes = {candidate.shape for candidate in found}

    assert Shape(dtype="bf16", batch=1, heads_q=32, heads_kv=8, seqlen_q=512,
                 seqlen_kv=512, head_dim=64, causal=True) in shapes, "llama prefill"
    assert Shape(dtype="bf16", batch=1, heads_q=32, heads_kv=8, seqlen_q=1,
                 seqlen_kv=512, head_dim=64, causal=True) in shapes, "llama decode"
    assert Shape(dtype="fp16", batch=1, heads_q=12, heads_kv=12, seqlen_q=512,
                 seqlen_kv=512, head_dim=64, causal=True) in shapes, "gpt2 in fp16"
    assert not any(shape.causal and shape.seqlen_q == 1 and shape.heads_q == 12
                   and not shape.causal for shape in shapes)
    # bert is an encoder: no autoregressive phase, so no decode shape.
    assert not any(shape.seqlen_q == 1 and not shape.causal for shape in shapes)
    assert {entry["model"] for entry in stats["skipped"]} == {"sdxl", "sd15"}


def test_shape_files_are_read_in_every_form_a_cluster_publishes_them_in(tmp_path):
    """The kernel team's CSV, aiter-style JSON records, `key=value` lines, and a
    hipDNN graph. Four spellings of the same fields; a directory that mines as zero
    rows is indistinguishable from one nobody pointed at."""
    root = tmp_path / "model-shapes"
    root.mkdir()
    (root / "published.csv").write_text(
        "model,batch,heads_q,heads_kv,seq_q,seq_kv,head_dim,dtype,mask\n"
        "csv,2,16,4,1024,1024,128,bfloat16,causal\n", encoding="utf-8")
    (root / "model_shapes.json").write_text(json.dumps([
        {"model": "records", "batch": 1, "nhead_q": 64, "nhead_k": 8, "seqlen_q": 1,
         "seqlen_k": 4096, "hdim_q": 128, "dtype": "fp16", "mask": "no_mask"}]),
        encoding="utf-8")
    (root / "hand.txt").write_text(
        "# a comment\nbatch=4 hq=8 hkv=8 sq=256 skv=256 d=64 dtype=half causal=true\n",
        encoding="utf-8")
    graph_shape = Shape(dtype="bf16", batch=3, heads_q=10, heads_kv=2, seqlen_q=128,
                        seqlen_kv=512, head_dim=64, causal=False)
    graphs.write(root, graph_shape)

    found, stats = model_shapes.from_shape_dir(root)
    shapes = {candidate.shape for candidate in found}
    assert graph_shape in shapes
    assert Shape(dtype="bf16", batch=2, heads_q=16, heads_kv=4, seqlen_q=1024,
                 seqlen_kv=1024, head_dim=128, causal=True) in shapes
    assert Shape(dtype="fp16", batch=1, heads_q=64, heads_kv=8, seqlen_q=1,
                 seqlen_kv=4096, head_dim=128, causal=False) in shapes
    assert Shape(dtype="fp16", batch=4, heads_q=8, heads_kv=8, seqlen_q=256,
                 seqlen_kv=256, head_dim=64, causal=True) in shapes
    assert stats["graphs"] == 1


@pytest.mark.parametrize("spelling", ["banded", "bottom_right"])
def test_an_unreadable_mask_spelling_is_refused_rather_than_guessed(tmp_path, spelling):
    """A guessed mask puts a differently-masked problem in the corpus under the row's
    name, and every number measured against it is attributed to a shape nobody ran.

    `bottom_right` is the dangerous one: it looks like a synonym for causal and is
    not. Aligning the causal diagonal to the bottom right masks a different triangle
    whenever seqlen_q != seqlen_k, and the graph written here is always TOP_LEFT.
    """
    root = tmp_path / "model-shapes"
    root.mkdir()
    (root / "published.csv").write_text(
        f"batch,heads_q,seq_q,seq_kv,head_dim,mask\n1,8,512,512,64,{spelling}\n",
        encoding="utf-8")
    with pytest.raises(SystemExit, match="unknown mask spelling"):
        model_shapes.from_shape_dir(root)


def test_the_sweep_obeys_the_constraint_the_declaration_states():
    """`seqlen_q <= seqlen_k` is declared, not assumed: the sampler reads the clause
    rather than restating the relation, so a declaration that changes it changes the
    sample. Every drawn point satisfies it and none repeats."""
    declaration = sweep.load(REPO / sweep.DEFAULT_DECLARATION)
    found, stats = sweep.sample(declaration, 200, seed=0, max_bytes=2 ** 31)
    assert len(found) == 200
    assert len({candidate.shape.key for candidate in found}) == 200
    for candidate in found:
        assert candidate.shape.seqlen_q <= candidate.shape.seqlen_kv
        assert candidate.shape.dtype in declaration["parameters"]["dtype"]["values"]
        assert candidate.shape.head_dim in declaration["regimes"]["head_dim"]["buckets"]
    assert all(stats["by_kind"][kind] for kind in declaration["mixture"]), (
        "every declared mixture component must contribute; a component that never "
        "draws is a share of the corpus the declaration asked for and did not get")


def test_the_sweep_fills_the_grouping_axis_the_packs_leave_thin():
    """GQA is the axis the declaration cannot express (its own `llama3_70b_gqa`
    archetype says so: "declared once the builder exposes the KV head count") and the
    one that decides whether decode is memory bound. A sweep of MHA-only shapes would
    leave the corpus's decode regime describing a problem nobody serves."""
    declaration = sweep.load(REPO / sweep.DEFAULT_DECLARATION)
    found, _ = sweep.sample(declaration, 200, seed=0, max_bytes=2 ** 31)
    groupings = {candidate.shape.grouping for candidate in found}
    assert {"mha", "gqa"} <= groupings
    assert all(candidate.shape.heads_q % candidate.shape.heads_kv == 0
               for candidate in found)


# ------------------------------------------------------------------------ allocation


def test_a_source_that_cannot_fill_its_share_hands_the_rest_back():
    """Otherwise a corpus asked for 1000 graphs returns however many the smallest
    source had, and the shortfall is invisible."""
    allocation = assemble.allocate(
        100, {"model": 5, "kernel": 400, "sweep": 400}, assemble.DEFAULT_SHARES)
    assert sum(allocation.values()) == 100
    assert allocation["model"] == 5


def test_every_source_is_represented_even_in_a_small_corpus():
    """A corpus of 30 that is 30 packed geometries has none of the shapes the
    heuristic exists to get right in it."""
    allocation = assemble.allocate(
        30, {"model": 50, "kernel": 500, "sweep": 500}, assemble.DEFAULT_SHARES)
    assert all(count > 0 for count in allocation.values())


# ------------------------------------------------------------------------------- cli


def test_the_command_line_produces_a_corpus_offline(tmp_path, capsys):
    """The acceptance run, at a size that keeps the suite quick: no GPU, no plugin, no
    network -- every input is a file in this tree."""
    out = tmp_path / "corpus"
    assert main(["--out", str(out), "--count", "50", "--seed", "1"]) == 0
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["emitted"] == 50
    assert len(list((out / "graphs").glob("*.json"))) == 50
    assert (out / "manifest.csv").is_file()
    assert set(manifest["mix"]) == {"model", "kernel", "sweep"}
    assert manifest["seed"] == 1
    assert all(entry["sha256"] for entry in manifest["inputs"])
    printed = capsys.readouterr().out
    assert "regimes" in printed


def test_a_corpus_can_be_restricted_to_one_mask(tmp_path, sources):
    """An engine can live entirely on one side of causality: AITER's gfx950 forward table is
    two kernels, both unmasked, so it declines every causal graph on that architecture. A
    corpus drawn without the facet comes out ~80% causal, which is ~80% wasted measurement
    when the corpus exists to train that engine."""
    manifest, _ = _build(tmp_path, sources, count=60, keep=Filter(causal=(False,)))

    assert manifest["emitted"] > 0
    assert {row["causal"] for row in manifest["graphs"]} == {False}
    assert manifest["reports"]["sweep"]["filtered"] > 0
