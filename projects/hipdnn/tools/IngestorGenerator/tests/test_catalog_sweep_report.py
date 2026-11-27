# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""tools/catalog_sweep_report.py and tools/ranker_check.py on synthetic sweep outputs.

Nothing here times a kernel: the winners are written by hand so that every verdict
has a number a reader can recompute. The real catalogs are used only to pin the
kernel-identity join (the shipped binaries of each arch, split by the additive-bias
feature; see ``_SHIPPED`` and ``_ARM``).
"""

from __future__ import annotations

import base64
import json
import struct
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import yaml

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tools"))

import catalog_sweep_report as rep  # noqa: E402
import ground_truth as gt  # noqa: E402
import ranker_check as rk  # noqa: E402
import registry_enumerator  # noqa: E402
import variant_reduce  # noqa: E402

_DESCRIPTORS = (
    _ROOT.parents[3]
    / "dnn-providers"
    / "hip-kernel-provider"
    / "src"
    / "engines"
    / "kernel_ingestor_engine"
    / "descriptors"
    / "rocKE"
)
_CONFIGS = _ROOT / "configs"
_DEVICE = dict(
    gcn_arch_name="gfx950",
    warp_size=64,
    multi_processor_count=256,
    total_global_mem=1,
    memory_bus_width=1,
    memory_clock_rate=1,
    shared_mem_per_block=1,
)


def graph_content(name, graph_id=bytes(16)):
    """A Graph flatbuffer holding only ``name`` (field 0) and ``id`` (field 9)."""
    vtable = struct.pack("<HH10H", 24, 24, 20, *([0] * 8), 4)
    table = struct.pack("<i", 24) + graph_id + struct.pack("<I", 4)
    text = struct.pack("<I", len(name)) + name.encode() + b"\0"
    raw = struct.pack("<I", 28) + vtable + table + text
    return base64.b64encode(raw).decode()


# --- a synthetic catalog: 6 binaries fanned out over two head-count cells -------------

_BINARIES = {  # label -> (block_m, block_n, persistent)
    "A": (256, 64, False),
    "B": (128, 64, False),
    "C": (256, 128, False),
    "D": (128, 128, False),
    "E": (64, 64, False),
    "P": (256, 64, True),
}
_CELLS = {"gqa": (32, 8), "mha": (32, 32)}


def _entry(label, cell, **extra):
    hq, kv = _CELLS[cell]
    bm, bn, persistent = _BINARIES[label]
    spec = dict(dtype="bf16", head_size=128, causal=True, block_m=bm, block_n=bn)
    spec |= dict(persistent=persistent, num_query_heads=hq, num_kv_heads=kv)
    spec |= dict(batch=1, seqlen_q=512, seqlen_kv=512) | extra.pop("spec", {})
    meta = dict(head_size=128, num_query_heads=hq, num_kv_heads=kv, block_m=bm)
    meta |= dict(block_n=bn, batch=1, seqlen_q=512, seqlen_kv=512)
    source = dict(kind="rocke", source="k.py", builder="b", spec=spec)
    out = dict(id=f"id-{label}-{cell}", name=f"attn_{label}_hq{hq}_kv{kv}")
    return out | dict(kernel_source=source, metadata=meta, priority=0) | extra


def _kdp(tmp_path, file, labels, cells=("gqa", "mha"), kdp_id="PACK"):
    entries = [_entry(label, cell) for cell in cells for label in labels]
    doc = dict(id=kdp_id, dispatch="DISP", kernelDescriptors=entries)
    path = tmp_path / file
    path.write_text(json.dumps(doc))
    return str(path)


def _winner(label, cell, time_ms):
    ids = dict(pack_id="PACK", dispatch_id="DISP", kernel_id=f"id-{label}-{cell}")
    return dict(time_ms=time_ms, **ids)


class Sweep:
    """Writes a sweep.py-shaped output directory."""

    def __init__(self, root):
        self.root, self.count = root, 0

    def phase(self, arm, kind, round_, *, graphs=None, ledger=(), status="success"):
        """``graphs``: {graph name: [winner entries]} written as one winners shard."""
        self.count += 1
        attempt = (
            self.root / "attempts" / f"{kind}__c__{arm}__r{round_}-{self.count:04d}"
        )
        shard = attempt / "winners" / "ingestor-winners" / "1.0.0" / "e" / "gfx950"
        shard.mkdir(parents=True)
        lines = ["1.0.0"]
        for name, entries in (graphs or {}).items():
            content = {"content_base64": graph_content(name, bytes([self.count]) * 16)}
            doc = dict(v=2, device=_DEVICE, graph=content, entries=entries)
            lines.append(json.dumps(doc, sort_keys=True))
        (shard / "winners.jsonl").write_text("\n".join(lines) + "\n")
        phase = dict(kind=kind, corpus="c", arm=arm, round=round_)
        doc = dict(phase=phase, status=status, ledger=list(ledger))
        (attempt / "phase.json").write_text(json.dumps(doc))

    def timing(self, arm, round_, times: dict, cell="gqa", served=None):
        """``times``: {graph: {label: ms}} (entries best first); ``served``: ledger
        means."""
        graphs = {
            g: [
                _winner(lab, cell, t)
                for lab, t in sorted(by.items(), key=lambda e: e[1])
            ]
            for g, by in times.items()
        }
        ledger = [
            dict(graph_name=g, outcome="served", mean_ms=ms)
            for g, ms in (served or {}).items()
        ]
        self.phase(arm, "timing", round_, graphs=graphs, ledger=ledger)


def _entry_map(
    shape="gqa", layout="bhsd", n=1, family=("bf16", 128, "NO_MASK"), **extra
):
    fam = dict(dtype=family[0], head_size=family[1], mask_type=family[2])
    return dict(
        graph_name=f"g_{layout}_{shape}",
        family=fam,
        shape=shape,
        layout=layout,
        **extra,
    )


@pytest.fixture
def world(tmp_path):
    arm = _kdp(tmp_path, "arm.kdp.json", "ABCDE")
    shipped = _kdp(tmp_path, "shipped.kdp.json", "AB", cells=("gqa",))
    return dict(tmp=tmp_path, arm=arm, shipped=shipped, sweep=Sweep(tmp_path / "sweep"))


def _rounds(world, mapping, rounds):
    catalog = rep.Catalog([world["arm"]])
    names = {m["graph_name"] for m in mapping}
    return rep.winner_rounds(world["sweep"].root, "full", catalog, names)


# --- kernel identity ------------------------------------------------------------------


def test_identity_ignores_shape_fields_names_and_ids():
    one, other = _entry("A", "gqa"), _entry("A", "mha")
    other["kernel_source"]["spec"] |= dict(batch=4, seqlen_q=2048, seqlen_kv=2048)
    assert one["name"] != other["name"] and one["id"] != other["id"]
    assert rep.kernel_identity(one) == rep.kernel_identity(other)


@pytest.mark.parametrize(
    "edit",
    [
        lambda e: e["kernel_source"]["spec"].update(block_n=32),
        lambda e: e["kernel_source"]["spec"].update(causal=False),
        lambda e: e["kernel_source"]["spec"].update(persistent=True),
        lambda e: e["kernel_source"].update(builder="other"),
        lambda e: e.update(arch=["gfx942"]),
    ],
    ids=["block_n", "causal", "persistent", "builder", "arch"],
)
def test_identity_changes_with_every_binary_field(edit):
    changed = _entry("A", "gqa")
    edit(changed)
    assert rep.kernel_identity(changed) != rep.kernel_identity(_entry("A", "gqa"))


def test_identity_is_the_binary_not_the_selection_metadata_or_priority():
    other = _entry("A", "gqa")
    other["metadata"]["block_m"] = 64
    other["priority"] = 7
    assert rep.kernel_identity(other) == rep.kernel_identity(_entry("A", "gqa"))


def test_omitted_spec_fields_are_completed_with_their_defaults():
    """A shipped entry omits a knob it holds at default; a full-space one carries it."""
    defaults = dict(wpe=2, lazy=True, num_query_heads=99, num_kv_heads=99, batch=9)
    shipped = _entry("A", "gqa")
    full = _entry("A", "mha")
    full["kernel_source"]["spec"] |= dict(wpe=2, lazy=True)
    off = _entry("A", "mha")
    off["kernel_source"]["spec"] |= dict(wpe=4, lazy=True)
    assert rep.kernel_identity(shipped) != rep.kernel_identity(full)
    assert rep.kernel_identity(shipped, defaults) == rep.kernel_identity(full, defaults)
    assert rep.kernel_identity(shipped, defaults) != rep.kernel_identity(off, defaults)


def test_a_shipped_catalog_with_no_binary_in_the_arm_is_refused(world, capsys):
    path, _ = _cli_world(world)
    elsewhere = _kdp(world["tmp"], "other.kdp.json", "E")
    argv = _argv("report", world, path, world["tmp"] / "r.yaml")
    argv[argv.index("--shipped-kdp") + 1] = elsewhere
    rows = json.loads(Path(elsewhere).read_text())
    for row in rows["kernelDescriptors"]:
        row["kernel_source"]["spec"]["block_m"] = 8
    Path(elsewhere).write_text(json.dumps(rows))
    assert rep.main(argv) == 2 and "pass --manifest" in capsys.readouterr().err


def test_a_multi_launch_entry_is_completed_launch_by_launch():
    base = _entry("A", "gqa")
    one = dict(id="x", name="n", sources=dict(main=base["kernel_source"]), metadata={})
    two = json.loads(json.dumps(one))
    two["sources"]["main"]["spec"]["knob"] = 5
    assert rep.kernel_identity(one, dict(knob=5)) == rep.kernel_identity(
        two, dict(knob=5)
    )
    assert rep.kernel_identity(one, dict(knob=6)) != rep.kernel_identity(
        two, dict(knob=6)
    )


def test_identity_of_a_multi_launch_entry_keeps_every_launch():
    base = _entry("A", "gqa")
    entry = dict(
        id="x", name="n", sources=dict(main=base["kernel_source"]), metadata={}
    )
    twin = json.loads(json.dumps(entry))
    twin["sources"]["main"]["spec"]["block_m"] = 64
    assert rep.kernel_identity(entry) != rep.kernel_identity(twin)


def test_catalog_collapses_cells_and_flags_persistent(tmp_path):
    path = _kdp(tmp_path, "k.json", "AP")
    catalog = rep.Catalog([path])
    assert len(catalog.identities) == 2 and sum(map(len, catalog.names.values())) == 4
    assert len(catalog.persistent) == 1
    assert catalog.of(_winner("A", "mha", 1.0)) in catalog.identities
    assert catalog.of(dict(pack_id="P", dispatch_id="D", kernel_id="k")) is None


def test_ids_are_matched_case_insensitively(tmp_path):
    path = _kdp(tmp_path, "k.json", "A", cells=("gqa",), kdp_id="PACK")
    upper = dict(pack_id="pack", dispatch_id="disp", kernel_id="ID-A-GQA")
    assert rep.Catalog([path]).of(upper) is not None


# Distinct binaries (``kernel_identity``) of the shipped dense-attention catalogs, in
# the slice the sweep recipe measures (no LSE twin, no sliding window), split by the
# additive-bias feature. "plain" is the design count: the 52 gfx950 kernels the catalog
# collapse left, and the 72 gfx942 ones once the LSE twins and the four window sizes
# are set aside. "bias" is what the additive-bias kernels add: one cell per head size x
# dtype x mask, at one tile on gfx950 (128/64) and at three on gfx942 (256/64, 64/64,
# 256/32).
_BIAS_CELLS = 2 * 2 * 2
_SHIPPED = {
    "gfx950": dict(plain=52, bias=_BIAS_CELLS * 1),
    "gfx942": dict(plain=72, bias=_BIAS_CELLS * 3),
}
# Distinct binaries of the full-space arm the F recipe builds (see ``_scratch_arm``).
# Only a run of the registry yields them: the plain part is every bias-off production
# spec of the cut cells, the bias part the tile sweep of the bias cells.
_ARM = {
    "gfx950": dict(plain=1304, bias=616),
    "gfx942": dict(plain=256, bias=44),
}


def _shipped_kdp(arch: str) -> Path:
    """The DVC-pulled dense-attention KDP of ``arch``; skips when it was not pulled."""
    folder = _DESCRIPTORS / f"{arch}_attention_dense"
    path = folder / "artifacts" / f"{arch}_attention_dense.kdp.json"
    if not path.is_file():
        pytest.skip(f"{path} not pulled (dvc pull in {path.parents[1]})")
    return path


def _in_recipe_slice(entry: dict) -> bool:
    """Whether the sweep recipe measures the entry: no LSE twin, no sliding window."""
    spec = entry["kernel_source"]["spec"]
    return not spec.get("emit_lse") and not spec["sliding_window"]


def _binaries(entries, defaults: dict | None = None) -> dict:
    """The distinct binaries of ``entries`` as ``plain`` and ``bias`` kernel sets."""
    out = dict(plain=set(), bias=set())
    for entry in entries:
        biased = entry["kernel_source"]["spec"].get("use_additive_bias")
        out["bias" if biased else "plain"].add(rep.kernel_identity(entry, defaults))
    return out


def _sizes(binaries: dict) -> dict:
    """The number of binaries in each kernel set of ``_binaries``."""
    return {kind: len(kernels) for kind, kernels in binaries.items()}


@pytest.mark.parametrize("arch", ["gfx950", "gfx942"])
def test_the_real_shipped_catalogs_collapse_to_the_binaries_the_design_counted(arch):
    """Needs the DVC-pulled KDP. The shipped binaries are the bias-off kernels the
    design counted plus the additive-bias kernels, two disjoint sets. A regenerated
    catalog (a new ``artifacts.dvc`` pointer) must update ``_SHIPPED``."""
    entries = json.loads(_shipped_kdp(arch).read_text())["kernelDescriptors"]
    shipped = _binaries(e for e in entries if _in_recipe_slice(e))
    assert _sizes(shipped) == _SHIPPED[arch]
    assert not shipped["plain"] & shipped["bias"]


def _registry_unavailable(text: str) -> bool:
    """Whether ``text`` (an enumerator error) says rocKE could not be imported."""
    words = ("ImportError", "ModuleNotFoundError", "not importable", "cannot import")
    return any(word in text for word in words)


@pytest.mark.parametrize(
    "text, unavailable",
    [
        ("ModuleNotFoundError: No module named 'numpy'", True),
        ("FAIL: rocKE dispatch is not importable (is the provider root bound?)", True),
        ("FAIL: cannot import the rocKE dispatcher registry module 'x'", True),
        ("FAIL: enumerate.cells: a cell field is unknown", False),
        ("KeyError: 'use_attn_bias'", False),
        ("", False),
    ],
)
def test_only_an_import_failure_counts_as_rocke_being_unavailable(text, unavailable):
    assert _registry_unavailable(text) is unavailable


@pytest.mark.parametrize(
    "stderr, outcome, message",
    [
        (
            "ModuleNotFoundError: No module named 'numpy'",
            pytest.skip.Exception,
            "rocKE registry not importable",
        ),
        (
            "FAIL: enumerate.cells: a cell field is unknown",
            pytest.fail.Exception,
            "enumerator exit 2: FAIL: enumerate.cells",
        ),
    ],
)
def test_the_scratch_arm_skips_for_a_missing_rocke_and_fails_for_anything_else(
    tmp_path, monkeypatch, stderr, outcome, message
):
    def refused(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 2, "", stderr)

    monkeypatch.setattr(subprocess, "run", refused)
    with pytest.raises(outcome, match=message):
        _scratch_arm(tmp_path, "gfx950")


def _scratch_arm(tmp_path, arch):
    """Kernels and spec defaults of the full-space arm the F recipe builds, with the
    spec defaults of its manifest. The recipe cuts the head cells to two head pairs,
    keeps only the window-free mask cells (a window is a baked field, so each size is a
    binary of its own) and drops the LSE twins; the additive-bias cells stay.
    """
    manifest = yaml.safe_load(
        (_CONFIGS / f"{arch}_attention_dense.engine.yaml").read_text()
    )
    enum = manifest["enumerate"]
    enum |= dict(space="full", max_kernels=100000)
    manifest["reduction"]["aot_budget_per_arch"] = 100000
    block = manifest["ground_truth"]
    block["graph_template"] = str(_CONFIGS / block["graph_template"])
    for group in enum["cells"]:
        fields = group["fields"]
        if "nhead_q" in fields:
            group["values"] = [[64, 64, 32, 8], [128, 128, 32, 8]]
        if "sliding_window" in fields:
            at = fields.index("sliding_window")
            group["values"] = [v for v in group["values"] if v[at] == 0]
        if fields == ["emit_lse"]:
            group["values"] = [False]
    path = tmp_path / f"{arch}.engine.yaml"
    path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    out = tmp_path / f"{arch}.config.yaml"
    run = subprocess.run(
        [sys.executable, str(_ROOT / "tools" / "registry_enumerator.py")]
        + f"--manifest {path} --out {out}".split(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    if run.returncode:
        if _registry_unavailable(run.stderr):
            pytest.skip(f"rocKE registry not importable: {run.stderr[-200:]}")
        pytest.fail(f"enumerator exit {run.returncode}: {run.stderr[-400:]}")
    kernels = [
        k for p in yaml.safe_load(out.read_text())["packs"] for k in p["kernels"]
    ]
    return kernels, rep.spec_defaults(path)


@pytest.mark.parametrize("arch", ["gfx950", "gfx942"])
def test_the_full_space_arm_holds_every_shipped_binary_once_defaults_are_completed(
    tmp_path, arch
):
    """Needs the DVC-pulled KDP and rocKE. A change of the rocKE enumerated space or of
    the manifest's cells must update ``_ARM``; a regenerated catalog ``_SHIPPED``."""
    entries = json.loads(_shipped_kdp(arch).read_text())["kernelDescriptors"]
    in_slice = [e for e in entries if _in_recipe_slice(e)]
    kernels, defaults = _scratch_arm(tmp_path, arch)
    arm = _binaries(kernels, defaults)
    shipped = _binaries(in_slice, defaults)
    assert _sizes(arm) == _ARM[arch]
    assert _sizes(shipped) == _SHIPPED[arch]
    assert shipped["plain"] <= arm["plain"] and shipped["bias"] <= arm["bias"]
    # the shipped slice omits knobs the full space carries: without the defaults the
    # two catalogs share no binary on gfx950
    bare = {rep.kernel_identity(e) for e in in_slice} & {
        rep.kernel_identity(k) for k in kernels
    }
    assert bool(bare) == (arch == "gfx942")


# --- winners back to graphs -----------------------------------------------------------


def test_graph_name_is_read_from_the_flatbuffer_whatever_its_id():
    for graph_id in (bytes(16), bytes(range(16))):
        assert (
            rep.graph_name_of(graph_content("layouts_r0007", graph_id))
            == "layouts_r0007"
        )
    same = variant_reduce.graph_identity(
        {"content_base64": graph_content("g", bytes(range(16)))}, "x"
    )
    assert (
        rep.graph_name_of(same) == "g"
    ), "the id-zeroed form the reducer uses still names the graph"


@pytest.mark.parametrize(
    "junk", ["", "AAAA", base64.b64encode(b"\x01" * 40).decode(), "!!"]
)
def test_graph_name_is_none_for_anything_but_a_named_graph(junk):
    assert rep.graph_name_of(junk) is None


def test_winner_rounds_reads_rounds_graphs_and_identities(world):
    mapping = [_entry_map()]
    world["sweep"].timing("full", 1, {"g_bhsd_gqa": {"A": 2.0, "B": 1.0}})
    world["sweep"].timing("full", 2, {"g_bhsd_gqa": {"A": 2.2, "B": 1.1}})
    rounds, problems, _ = _rounds(world, mapping, None)
    catalog = rep.Catalog([world["arm"]])
    a, b = (catalog.of(_winner(x, "gqa", 0)) for x in "AB")
    assert rounds == {
        1: {"g_bhsd_gqa": {a: 2.0, b: 1.0}},
        2: {"g_bhsd_gqa": {a: 2.2, b: 1.1}},
    }
    assert not problems


def test_winner_rounds_counts_what_it_cannot_join(world):
    world["sweep"].timing("full", 1, {"g_bhsd_gqa": {"A": 1.0}, "stranger": {"A": 1.0}})
    bogus = dict(pack_id="PACK", dispatch_id="DISP", kernel_id="nope", time_ms=1.0)
    world["sweep"].phase(
        "full", "timing", 2, graphs={"g_bhsd_gqa": [bogus, _winner("A", "gqa", 3.0)]}
    )
    rounds, problems, lost = _rounds(world, [_entry_map()], None)
    assert problems == {"unmapped_records": 1, "unknown_kernel_entries": 1}
    assert lost == {"g_bhsd_gqa": 1}, "a dropped entry is charged to its graph"
    assert list(rounds) == [1, 2] and len(rounds[2]["g_bhsd_gqa"]) == 1


def test_winner_rounds_skips_failed_phases_other_arms_and_correctness(world):
    world["sweep"].phase(
        "full",
        "timing",
        1,
        graphs={"g_bhsd_gqa": [_winner("A", "gqa", 1.0)]},
        status="failed",
    )
    world["sweep"].phase(
        "other", "timing", 1, graphs={"g_bhsd_gqa": [_winner("A", "gqa", 1.0)]}
    )
    world["sweep"].phase(
        "full", "correctness", 1, graphs={"g_bhsd_gqa": [_winner("A", "gqa", 1.0)]}
    )
    rounds, problems, _ = _rounds(world, [_entry_map()], None)
    assert rounds == {} and not problems


def test_one_identity_shipped_in_two_cells_keeps_its_fastest_time(world):
    both = [_winner("A", "gqa", 3.0), _winner("A", "mha", 2.0)]
    world["sweep"].phase("full", "timing", 1, graphs={"g_bhsd_gqa": both})
    rounds, _, _ = _rounds(world, [_entry_map()], None)
    assert list(rounds[1]["g_bhsd_gqa"].values()) == [2.0]


# --- the A/A' band and the drift ------------------------------------------------------


def _served(world, arm, round_, values):
    world["sweep"].phase(
        arm,
        "timing",
        round_,
        ledger=[
            dict(graph_name=f"g{i}", outcome="served", mean_ms=v)
            for i, v in enumerate(values)
        ],
    )


def test_control_band_is_the_p95_of_the_graphs_aa_prime_deviation(world):
    base = [1.0] * 20
    _served(world, "A", 1, base)
    _served(world, "A2", 1, [1 + 0.001 * (i + 1) for i in range(20)])
    # nearest rank: ceil(0.95 * 20) = 19th of the sorted deviations
    assert rep.control_band(world["sweep"].root, "A", "A2") == pytest.approx(0.019)


def test_control_band_takes_the_median_over_rounds_and_ignores_unshared_graphs(world):
    for r, (a, b) in enumerate([(1.0, 1.10), (1.0, 1.30), (1.0, 1.12)], start=1):
        _served(world, "A", r, [a, 5.0])
        _served(world, "A2", r, [b])
    assert rep.control_band(world["sweep"].root, "A", "A2") == pytest.approx(0.12)
    assert rep.control_band(world["sweep"].root, "A", "nothing") is None


def test_cell_drift_needs_two_rounds_and_is_the_median_spread(world):
    rounds = {1: {"g": {"x": 1.0, "y": 2.0}}, 2: {"g": {"x": 1.0, "y": 2.4}}}
    assert rep.cell_drift({1: rounds[1]}, {"g"}) is None
    # y/best: 2.0 then 2.4/1.0 -> (2.4 - 2.0) / 2.2; x/best: 1, 1 -> 0;
    # median of {0, 0.1818}
    assert rep.cell_drift(rounds, {"g"}) == pytest.approx(0.4 / 2.2 / 2)
    assert rep.cell_drift(rounds, {"other"}) is None


# --- verdicts -------------------------------------------------------------------------


def _catalog_and_shipped(world):
    catalog = rep.Catalog([world["arm"]])
    return catalog, rep.Catalog([world["shipped"]]).identities


def _report(world, mapping, per_round, band=None, checks=None):
    """per_round: [{graph: {label: ms}}] -> build_report on shipped = {A, B}."""
    for r, times in enumerate(per_round, start=1):
        world["sweep"].timing("full", r, times)
    rounds, _, lost = _rounds(world, mapping, None)
    catalog, shipped = _catalog_and_shipped(world)
    return rep.build_report(mapping, rounds, shipped, catalog, band, checks, lost)


def _two_shapes():
    return [_entry_map(shape="gqa"), _entry_map(shape="mha")]


def _times(gqa, mha):
    return {"g_bhsd_gqa": gqa, "g_bhsd_mha": mha}


def test_a_graph_ratio_is_best_shipped_over_best_enumerated(world):
    report = _report(
        world, _two_shapes(), [_times(dict(A=1.2, B=1.5, C=1.0), dict(A=1.0, E=1.0))]
    )
    first, second = report["graphs"]
    assert (
        first["ratio"] == pytest.approx(1.2)
        and first["best_enumerated"]["time_ms"] == 1.0
    )
    assert first["candidates"] == 3 and first["shipped_candidates"] == 2
    assert second["ratio"] == pytest.approx(1.0)


def test_the_shipped_set_is_matched_by_binary_not_by_cell(world):
    """The shipped KDP holds A and B in the gqa cell only; the mha graph sees them."""
    report = _report(
        world, [_entry_map(shape="mha")], [{"g_bhsd_mha": dict(A=1.0, C=0.5)}]
    )
    assert report["graphs"][0]["shipped_candidates"] == 1


def test_a_cell_passes_when_its_geomean_is_inside_the_band(world):
    report = _report(
        world, _two_shapes(), [_times(dict(A=1.02, C=1.0), dict(A=1.02, C=1.0))]
    )
    (cell,) = report["cells"]
    assert cell["verdict"] == "PASS" and cell["geomean_ratio"] == pytest.approx(1.02)
    assert cell["band"] == pytest.approx(rep.BAND_FLOOR) and cell["watch"] == []


def test_a_cell_fails_above_the_band_and_names_the_shapes_above_it(world):
    report = _report(
        world, _two_shapes(), [_times(dict(A=1.2, C=1.0), dict(A=1.0, C=1.0))]
    )
    (cell,) = report["cells"]
    assert cell["verdict"] == "FAIL" and cell["geomean_ratio"] == pytest.approx(
        1.2**0.5
    )
    assert cell["watch"] == ["gqa"]


def test_a_shape_exactly_on_the_band_is_not_a_watch(world):
    edge = 1 + rep.BAND_FLOOR
    report = _report(world, [_entry_map()], [{"g_bhsd_gqa": dict(A=edge, C=1.0)}])
    assert report["cells"][0]["watch"] == []


def test_a_pass_still_lists_a_shape_above_the_band_as_watch(world):
    report = _report(
        world, _two_shapes(), [_times(dict(A=1.05, C=1.0), dict(A=1.0, C=1.0))]
    )
    (cell,) = report["cells"]
    assert cell["verdict"] == "PASS" and cell["watch"] == ["gqa"]


def test_the_band_edge_belongs_to_the_pass(world):
    edge = 1 + rep.BAND_FLOOR
    report = _report(world, [_entry_map()], [{"g_bhsd_gqa": dict(A=edge, C=1.0)}])
    assert report["cells"][0]["verdict"] == "PASS"
    report = _report(
        world,
        [_entry_map(layout="bshd")],
        [{"g_bshd_gqa": dict(A=edge + 0.001, C=1.0)}],
    )
    assert report["cells"][0]["verdict"] == "FAIL"


@pytest.mark.parametrize(
    "band, drift_times, expected",
    [
        (None, False, rep.BAND_FLOOR),
        (0.08, False, 0.08),
        (0.01, True, "drift"),
    ],
    ids=["floor", "control", "drift"],
)
def test_the_band_is_the_largest_of_floor_control_and_twice_the_drift(
    world, band, drift_times, expected
):
    rounds = [{"g_bhsd_gqa": dict(A=1.0, C=1.0, E=1.5)}]
    if drift_times:
        rounds = [{"g_bhsd_gqa": dict(A=1.0, C=1.0, E=1.5, D=1.6)}]
        rounds += [{"g_bhsd_gqa": dict(A=1.0, C=1.3, E=1.9, D=2.0)}]
    report = _report(world, [_entry_map()], rounds, band=band)
    (cell,) = report["cells"]
    if expected == "drift":
        assert cell["drift"] > 0 and cell["band"] == pytest.approx(2 * cell["drift"])
        assert cell["band"] > rep.BAND_FLOOR
    else:
        assert cell["band"] == pytest.approx(expected) and cell["drift"] is None


def test_a_cell_with_a_graph_that_has_no_shipped_kernel_is_incomplete(world):
    report = _report(
        world, _two_shapes(), [_times(dict(A=1.0, C=1.0), dict(C=1.0, D=1.1))]
    )
    (cell,) = report["cells"]
    assert cell["verdict"] == "INCOMPLETE" and cell["missing"] == ["g_bhsd_mha"]


def test_a_graph_nobody_timed_makes_its_cell_incomplete(world):
    report = _report(world, _two_shapes(), [{"g_bhsd_gqa": dict(A=1.0, C=1.0)}])
    assert report["cells"][0]["verdict"] == "INCOMPLETE"


def test_a_graph_that_lost_winner_entries_makes_its_cell_incomplete(world):
    """A winner entry no --arm-kdp kernel matches is dropped; if it was the fastest one
    the best enumerated time looks slower and the cell would pass on partial data."""
    bogus = dict(pack_id="PACK", dispatch_id="DISP", kernel_id="nope", time_ms=0.1)
    entries = [bogus, _winner("A", "gqa", 1.0), _winner("C", "gqa", 1.0)]
    world["sweep"].phase("full", "timing", 1, graphs={"g_bhsd_gqa": entries})
    mapping = [_entry_map()]
    rounds, _, lost = _rounds(world, mapping, None)
    catalog, shipped = _catalog_and_shipped(world)
    report = rep.build_report(mapping, rounds, shipped, catalog, None, None, lost)
    (cell,) = report["cells"]
    assert cell["verdict"] == "INCOMPLETE" and cell["lost_graphs"] == ["g_bhsd_gqa"]
    assert report["graphs"][0]["lost_entries"] == 1
    clean = rep.build_report([_entry_map()], rounds, shipped, catalog)
    assert clean["cells"][0]["verdict"] == "PASS", "without `lost` the ratio is 1.0"


def test_a_layout_that_repeats_the_twin_is_flagged_and_left_out_of_the_cell(world):
    mapping = [
        _entry_map(layout="batch_slice", same_as_bshd=True),
        _entry_map(layout="batch_slice", shape="mha"),
    ]
    report = _report(
        world,
        mapping,
        [
            {
                "g_batch_slice_gqa": dict(A=9.0, C=1.0),
                "g_batch_slice_mha": dict(A=1.0, C=1.0),
            }
        ],
    )
    (cell,) = report["cells"]
    assert cell["graphs"] == 1 and cell["verdict"] == "PASS"
    assert [g["repeats_twin"] for g in report["graphs"]] == [True, False]


def test_cells_split_by_family_and_layout(world):
    mapping = [
        _entry_map(layout="bhsd"),
        _entry_map(layout="packed_qkv"),
        _entry_map(layout="bhsd", family=("fp16", 128, "NO_MASK"))
        | dict(graph_name="g_other"),
    ]
    times = {
        "g_bhsd_gqa": dict(A=1.0),
        "g_packed_qkv_gqa": dict(A=1.0),
        "g_other": dict(A=1.0),
    }
    report = _report(world, mapping, [times])
    assert sorted(tuple(c["cell"]) for c in report["cells"]) == [
        ("bf16", 128, "NO_MASK", "bhsd"),
        ("bf16", 128, "NO_MASK", "packed_qkv"),
        ("fp16", 128, "NO_MASK", "bhsd"),
    ]


def test_each_best_kernel_reports_whether_it_was_numerically_checked(world):
    mapping = [_entry_map()]
    times = {"g_bhsd_gqa": dict(A=1.1, C=1.0)}
    catalog = rep.Catalog([world["arm"]])
    a, c = (catalog.of(_winner(x, "gqa", 0)) for x in "AC")
    for n, (checks, enumerated, shipped) in enumerate(
        [
            ({"g_bhsd_gqa": {c: True}}, "pass", "not checked"),
            ({"g_bhsd_gqa": {c: False, a: True}}, "FAIL", "pass"),
            ({}, "not checked", "not checked"),
        ]
    ):
        report = _report(world, mapping, [times], checks=checks)
        row = report["graphs"][0]
        assert (row["best_enumerated"]["numeric"], row["best_shipped"]["numeric"]) == (
            enumerated,
            shipped,
        )
        world["sweep"] = Sweep(world["tmp"] / f"s{n}")


def test_a_best_shipped_kernel_that_fails_its_numeric_check_fails_the_cell(world):
    """A fast, wrong shipped kernel has the best time and ratio 1.0."""
    catalog = rep.Catalog([world["arm"]])
    a, c = (catalog.of(_winner(x, "gqa", 0)) for x in "AC")
    times = [{"g_bhsd_gqa": dict(A=1.0, C=1.0)}]
    report = _report(world, [_entry_map()], times, checks={"g_bhsd_gqa": {a: False}})
    (cell,) = report["cells"]
    assert cell["geomean_ratio"] == pytest.approx(1.0)
    assert cell["verdict"] == "FAIL" and cell["numeric_failed"] == ["g_bhsd_gqa"]
    # a failing best *enumerated* kernel is not a shipped kernel's fault
    world["sweep"] = Sweep(world["tmp"] / "other")
    report = _report(world, [_entry_map()], times, checks={"g_bhsd_gqa": {c: False}})
    assert report["cells"][0]["verdict"] == "PASS"
    assert report["cells"][0]["numeric_failed"] == []


def test_a_best_shipped_kernel_nobody_checked_is_listed_not_failed(world):
    catalog = rep.Catalog([world["arm"]])
    a = catalog.of(_winner("A", "gqa", 0))
    mapping = _two_shapes()
    times = [_times(dict(A=1.0, C=1.0), dict(A=1.0, C=1.0))]
    report = _report(world, mapping, times, checks={"g_bhsd_gqa": {a: True}})
    (cell,) = report["cells"]
    assert cell["verdict"] == "PASS" and cell["not_checked"] == ["g_bhsd_mha"]


@pytest.mark.parametrize("order", [["one", "two"], ["two", "one"]])
def test_a_failing_numeric_check_is_not_overwritten_by_a_passing_one(world, order):
    catalog = rep.Catalog([world["arm"]])
    graphs = {"g": [_winner("A", "gqa", 1.0)]}
    for arm, ok in (("one", False), ("two", True)):
        correctness = dict(passed=ok, execution_success=True, tolerance_match=ok)
        ledger = [dict(graph_name="g", outcome="served", correctness=correctness)]
        world["sweep"].phase(arm, "correctness", 1, graphs=graphs, ledger=ledger)
    got = rep.numeric_checks(world["sweep"].root, order, catalog, {"g"})
    assert got == {"g": {catalog.of(_winner("A", "gqa", 0)): False}}


def test_numeric_checks_pair_the_served_kernel_with_its_verdict(world):
    catalog = rep.Catalog([world["arm"]])
    ledger = [
        dict(
            graph_name="g_bhsd_gqa",
            outcome="served",
            correctness=dict(passed=True, execution_success=True, tolerance_match=True),
        ),
        dict(
            graph_name="g_bhsd_mha",
            outcome="served",
            correctness=dict(
                passed=True, execution_success=True, tolerance_match=False
            ),
        ),
        dict(
            graph_name="g_packed_qkv_gqa",
            outcome="served",
            correctness=dict(passed=True, execution_success=True, tolerance_match=True),
            reference=dict(usable=False),
        ),
    ]
    graphs = {
        "g_bhsd_gqa": [_winner("B", "gqa", 2.0), _winner("C", "gqa", 1.0)],
        "g_bhsd_mha": [_winner("A", "mha", 1.0)],
        "g_packed_qkv_gqa": [_winner("A", "gqa", 1.0)],
    }
    world["sweep"].phase("full", "correctness", 1, graphs=graphs, ledger=ledger)
    names = set(graphs)
    got = rep.numeric_checks(world["sweep"].root, ["full"], catalog, names)
    c, a = (catalog.of(_winner(x, "gqa", 0)) for x in "CA")
    assert got == {"g_bhsd_gqa": {c: True}, "g_bhsd_mha": {a: False}}


# --- kernels the engine cannot launch, and the screening stage ------------------------


def test_persistent_kernels_of_the_full_enumeration_are_listed_apart(tmp_path):
    arm = rep.Catalog([_kdp(tmp_path, "arm.json", "AB")])
    full = rep.Catalog([_kdp(tmp_path, "full.json", "ABP")])
    listed = rep.not_launchable(arm, full)
    assert listed["specs"] == 1 and listed["persistent"] == 1
    assert listed["examples"] == ["attn_P_hq32_kv8"]
    # a spec missing from the arm for another reason is not called persistent
    wider = rep.not_launchable(rep.Catalog([_kdp(tmp_path, "one.json", "A")]), full)
    assert wider["specs"] == 2 and wider["persistent"] == 1


def _screen_rounds(world):
    world["sweep"].timing(
        "screen",
        1,
        {
            "g1": dict(A=1.0, B=1.01, C=1.03, D=1.5, E=2.0),
            "g2": dict(E=1.0, D=1.2, C=1.9, B=2.0, A=3.0),
        },
    )
    catalog = rep.Catalog([world["arm"]])
    names = {"g1", "g2"}
    rounds, _, _ = rep.winner_rounds(world["sweep"].root, "screen", catalog, names)
    return catalog, rounds


def _labels(catalog, identities):
    back = {catalog.of(_winner(x, "gqa", 0)): x for x in "ABCDE"}
    return {back[i] for i in identities}


def test_screen_keeps_the_top_k_the_near_ties_and_the_shipped(world):
    catalog, rounds = _screen_rounds(world)
    ship = {catalog.of(_winner("E", "gqa", 0))}
    assert _labels(
        catalog, rep.screen_candidates(rounds, set(), top_k=1, margin=0.0)
    ) == {"A", "E"}
    assert _labels(
        catalog, rep.screen_candidates(rounds, set(), top_k=1, margin=0.02)
    ) == {"A", "B", "E"}
    assert _labels(
        catalog, rep.screen_candidates(rounds, ship, top_k=1, margin=0.0)
    ) == {"A", "E"}
    worst = {catalog.of(_winner("D", "gqa", 0))}  # never top-1, never a near tie
    assert _labels(
        catalog, rep.screen_candidates(rounds, worst, top_k=1, margin=0.0)
    ) == {"A", "D", "E"}
    only_a = {catalog.of(_winner("A", "gqa", 0))}
    assert _labels(
        catalog, rep.screen_candidates(rounds, only_a, top_k=2, margin=0.0)
    ) == {"A", "B", "E", "D"}
    assert _labels(catalog, rep.screen_candidates(rounds, only_a, top_k=5)) == set(
        "ABCDE"
    )


# --- the command line, end to end on the real `layouts` map ---------------------------


def _layout_map(tmp_path):
    shapes = tmp_path / "shapes.yaml"
    spec = dict(
        dtypes=["bf16"],
        head_dims=[128],
        mask_types=["NO_MASK"],
        layouts=["bshd", "bhsd"],
    )
    spec["shapes"] = [dict(id="gqa", batch=2, nhead_q=32, nhead_k=8, seqlen_q=256)]
    shapes.write_text(yaml.safe_dump(spec))
    manifest = _ROOT / "configs" / "gfx950_attention_dense.engine.yaml"
    out = tmp_path / "graphs"
    assert (
        gt.main(f"layouts --manifest {manifest} --shapes {shapes} --out {out}".split())
        == 0
    )
    return str(gt.layouts_map_path(out)), [p.stem for p in sorted(out.glob("*.json"))]


def _cli_world(world, ratio_bhsd=1.0):
    path, names = _layout_map(world["tmp"])
    for r in (1, 2):
        times = {names[0]: dict(A=1.0, C=0.99), names[1]: dict(A=ratio_bhsd, C=1.0)}
        served = {n: 1.0 for n in names}
        world["sweep"].timing("full", r, times, served=served)
        world["sweep"].timing("full2", r, times, served=served)
    return path, names


def _argv(command, world, path, out, *extra):
    base = f"{command} --sweep-output {world['sweep'].root} --arm full"
    base += f" --arm-kdp {world['arm']}"
    return (
        base
        + f" --shipped-kdp {world['shipped']} --map {path} --out {out} "
        + " ".join(extra)
    ).split()


def test_report_cli_writes_cells_and_graphs_for_the_layouts_map(world, capsys):
    path, names = _cli_world(world)
    out = world["tmp"] / "report.yaml"
    assert (
        rep.main(
            _argv(
                "report",
                world,
                path,
                out,
                "--control-arm full2",
                f"--full-kdp {world['arm']}",
            )
        )
        == 0
    )
    result = yaml.safe_load(out.read_text())
    assert [c["verdict"] for c in result["cells"]] == ["PASS", "PASS"]
    assert [g["graph"] for g in result["graphs"]] == names
    assert result["band_basis"]["control_p95"] == pytest.approx(0.0)
    assert result["not_launchable"]["specs"] == 0 and result["problems"] == {}
    assert "PASS" in capsys.readouterr().out


def test_report_cli_completes_both_catalogs_with_the_manifests_defaults(world):
    """--manifest fills omitted spec fields of the arm, shipped and full catalogs."""
    manifest = _CONFIGS / "gfx950_attention_dense.engine.yaml"
    try:
        rep.spec_defaults(manifest)
    except (ImportError, registry_enumerator.EnumerationError) as exc:
        if not _registry_unavailable(f"{type(exc).__name__}: {exc}"):
            raise
        pytest.skip(f"rocKE registry not importable: {exc}")
    path, _ = _cli_world(world)
    out = world["tmp"] / "report.yaml"
    argv = _argv("report", world, path, out, f"--manifest {manifest}")
    assert rep.main(argv + f"--full-kdp {world['arm']}".split()) == 0
    result = yaml.safe_load(out.read_text())
    assert [c["verdict"] for c in result["cells"]] == ["PASS", "PASS"]
    assert result["not_launchable"]["specs"] == 0


def test_report_cli_strict_exits_1_on_a_failing_cell_and_2_on_bad_input(world, capsys):
    path, _ = _cli_world(world, ratio_bhsd=1.5)
    out = world["tmp"] / "report.yaml"
    assert rep.main(_argv("report", world, path, out)) == 0
    assert rep.main(_argv("report", world, path, out, "--strict")) == 1
    bad = _argv("report", world, path, out)
    bad[bad.index("--arm") + 1] = "no-such-arm"
    assert rep.main(bad) == 2 and "no timing winners" in capsys.readouterr().err


def test_report_cli_strict_refuses_an_incomplete_cell(world):
    path, names = _layout_map(world["tmp"])
    world["sweep"].timing("full", 1, {names[0]: dict(A=1.0, C=1.0)})
    out = world["tmp"] / "report.yaml"
    assert rep.main(_argv("report", world, path, out)) == 0
    cells = yaml.safe_load(out.read_text())["cells"]
    assert {c["verdict"] for c in cells} == {"PASS", "INCOMPLETE"}
    assert rep.main(_argv("report", world, path, out, "--strict")) == 1


def _correctness_phase(world, names, passed=True):
    """The arm's correctness phase served kernel A, each graph's best shipped one."""
    result = dict(passed=passed, execution_success=True, tolerance_match=passed)
    ledger = [dict(graph_name=n, outcome="served", correctness=result) for n in names]
    graphs = {n: [_winner("A", "gqa", 1.0)] for n in names}
    world["sweep"].phase("full", "correctness", 1, graphs=graphs, ledger=ledger)


def test_report_cli_strict_exits_0_only_when_every_best_shipped_kernel_passed(
    world, capsys
):
    path, names = _cli_world(world)
    out = world["tmp"] / "report.yaml"
    strict = _argv("report", world, path, out, "--strict")
    assert rep.main(strict) == 1, "nothing checked the best shipped kernels"
    assert "no passing numeric check" in capsys.readouterr().err
    _correctness_phase(world, names)
    assert rep.main(strict) == 0
    cells = yaml.safe_load(out.read_text())["cells"]
    assert all(c["not_checked"] == [] for c in cells)


def test_report_cli_fails_a_cell_whose_best_shipped_kernel_is_wrong(world):
    path, names = _cli_world(world)
    _correctness_phase(world, names, passed=False)
    out = world["tmp"] / "report.yaml"
    assert rep.main(_argv("report", world, path, out)) == 0
    cells = yaml.safe_load(out.read_text())["cells"]
    assert [c["verdict"] for c in cells] == ["FAIL", "FAIL"]
    assert all(c["numeric_failed"] for c in cells)
    assert rep.main(_argv("report", world, path, out, "--strict")) == 1


def test_report_cli_strict_refuses_winner_entries_no_arm_kernel_matches(world, capsys):
    """Winners the KDPs do not list drop out, which can only flatter the shipped set."""
    path, names = _cli_world(world)
    _correctness_phase(world, names)
    bogus = dict(pack_id="PACK", dispatch_id="DISP", kernel_id="nope", time_ms=0.1)
    graphs = {names[0]: [bogus, _winner("A", "gqa", 1.0), _winner("C", "gqa", 0.99)]}
    world["sweep"].phase("full", "timing", 3, graphs=graphs)
    out = world["tmp"] / "report.yaml"
    assert rep.main(_argv("report", world, path, out)) == 0
    result = yaml.safe_load(out.read_text())
    assert result["problems"] == {"unknown_kernel_entries": 1}
    assert result["cells"][0]["verdict"] == "INCOMPLETE"
    assert rep.main(_argv("report", world, path, out, "--strict")) == 1
    assert "1 winner entries match no --arm-kdp kernel" in capsys.readouterr().err


def test_strict_refuses_unmatched_winner_entries_on_their_own():
    """Not through a cell: a graph that repeats its BSHD twin is in no cell."""
    result = dict(cells=[dict(not_checked=[])])
    clean = rep._strict_refusals(result, Counter(PASS=1), Counter())
    assert clean == []
    (reason,) = rep._strict_refusals(
        result, Counter(PASS=1), Counter(unknown_kernel_entries=2)
    )
    assert "2 winner entries match no --arm-kdp kernel" in reason


@pytest.mark.parametrize("tool", ["report", "ranker"])
def test_a_malformed_map_is_unusable_input_not_a_crash(world, capsys, tool):
    bad = world["tmp"] / "bad.yaml"
    bad.write_text("graph_name: [unclosed\n")
    out = world["tmp"] / "out.yaml"
    if tool == "report":
        argv = _argv("report", world, bad, out)
        code = rep.main(argv)
    else:
        argv = f"--sweep-output x --ranker-arm a --best-arm b --map {bad} --out {out}"
        code = rk.main(argv.split())
    assert code == 2 and "not a readable YAML file" in capsys.readouterr().err


@pytest.mark.parametrize(
    "text, message",
    [
        ("graph_name: g\n", "not a `layouts` map"),
        ("- a string\n", "not a `layouts` map"),
        ("- {graph_name: g, layout: l, shape: s}\n", "entry 1 (g) lacks ['family']"),
        ("- {layout: l, shape: s, family: {}}\n", "entry 1 (?) lacks ['graph_name', "),
        (
            "- {graph_name: g, layout: l, shape: s, family: {dtype: bf16}}\n",
            "lacks ['family.head_size', 'family.mask_type']",
        ),
        (
            "- {graph_name: g, layout: l, shape: s, family: bf16}\n",
            "lacks ['family.dtype/head_size/mask_type']",
        ),
    ],
)
def test_a_map_entry_without_a_key_the_tools_read_is_named(
    world, capsys, text, message
):
    bad = world["tmp"] / "bad.yaml"
    bad.write_text(text)
    out = world["tmp"] / "out.yaml"
    assert rep.main(_argv("report", world, bad, out)) == 2
    assert message in capsys.readouterr().err
    argv = f"--sweep-output x --ranker-arm a --best-arm b --map {bad} --out {out}"
    assert rk.main(argv.split()) == 2
    assert message in capsys.readouterr().err


def test_a_bug_inside_the_report_is_a_traceback_not_unusable_input(world, monkeypatch):
    """main reports only input problems (exit 2); a TypeError is the tool's own bug."""
    path, _ = _cli_world(world)

    def broken(*args, **kwargs):
        raise TypeError("a bug")

    monkeypatch.setattr(rep, "build_report", broken)
    with pytest.raises(TypeError, match="a bug"):
        rep.main(_argv("report", world, path, world["tmp"] / "report.yaml"))


@pytest.mark.parametrize(
    ("descriptors", "message"),
    [
        ([dict(id="k")], "entry 1 (?) lacks ['name', 'kernel_source or sources']"),
        ([dict(id="k", sources=dict(a=dict(spec={})))], "entry 1 (?) lacks ['name']"),
        ([dict(name="n", kernel_source=dict(spec={}))], "entry 1 (n) lacks ['id']"),
        ([dict(id="k", name="n", sources={})], "lacks ['kernel_source or sources']"),
        (
            [dict(id="k", name="n", kernel_source=dict(spec=[1]))],
            "entry 1 (n): a kernel source is not an object with an object spec",
        ),
        (["k"], "entry 1 (?) is not an object"),
        (dict(k=1), "kernelDescriptors is not a list"),
    ],
)
def test_a_malformed_kdp_entry_is_unusable_input_naming_the_entry(
    world, capsys, descriptors, message
):
    """A KDP is input: an entry the kernel identity cannot read exits 2 naming the KDP
    and the entry, as a malformed map entry does. A traceback would exit 1, the code
    ``--strict`` gives a failing cell."""
    path, _ = _cli_world(world)
    bad = world["tmp"] / "bad.kdp.json"
    doc = dict(id="PACK", dispatch="DISP", kernelDescriptors=descriptors)
    bad.write_text(json.dumps(doc))
    argv = _argv("report", world, path, world["tmp"] / "out.yaml", "--strict")
    argv[argv.index("--arm-kdp") + 1] = str(bad)
    assert rep.main(argv) == 2
    err = capsys.readouterr().err
    assert err.startswith(f"FAIL: {bad}: ") and message in err, err


@pytest.mark.parametrize(
    ("ledger", "message"),
    [
        (["g"], "'ledger' is not a list of objects"),
        ([dict(outcome="served", mean_ms=1.0)], "graph_name=None, mean_ms=1.0"),
        ([dict(graph_name="g", outcome="served", mean_ms="1")], "mean_ms='1'"),
        ([dict(graph_name="g", outcome="served", mean_ms=-1.0)], "mean_ms=-1.0"),
    ],
)
def test_a_malformed_ledger_is_unusable_input_naming_the_phase(
    world, capsys, ledger, message
):
    """A served entry is joined to its graph by ``graph_name`` and timed by
    ``mean_ms``; a ledger that lacks them is input to refuse (exit 2, naming the
    phase record), not a KeyError traceback."""
    path = world["tmp"] / "map.yaml"
    path.write_text(yaml.safe_dump(_ranker_map()))
    world["sweep"].phase("ranked", "timing", 1, ledger=ledger)
    out = world["tmp"] / "ranker.yaml"
    argv = f"--sweep-output {world['sweep'].root} --ranker-arm ranked --best-arm best"
    assert rk.main(f"{argv} --map {path} --out {out}".split()) == 2
    err = capsys.readouterr().err
    assert err.startswith("FAIL: ") and "phase.json: " in err and message in err, err


def test_screen_cli_writes_every_descriptor_name_of_every_kept_binary(world):
    path, _ = _cli_world(world)
    out, names = world["tmp"] / "screen.yaml", world["tmp"] / "keep.txt"
    assert rep.main(_argv("screen", world, path, out, f"--out-names {names}")) == 0
    kept = names.read_text().split()
    assert (
        sorted(kept) == sorted(set(kept))
        and "attn_A_hq32_kv8" in kept
        and "attn_A_hq32_kv32" in kept
    )
    assert (
        "verdict" not in out.read_text()
    ), "a screen gives candidates, never a verdict"


# --- the ranker check -----------------------------------------------------------------


def _ranker_map():
    out = []
    for layout in ("bshd", "bhsd"):
        for shape in ("gqa", "mha"):
            out.append(_entry_map(shape=shape, layout=layout))
    return out


def _ratios(bshd, bhsd):
    names = ["g_bshd_gqa", "g_bshd_mha", "g_bhsd_gqa", "g_bhsd_mha"]
    return dict(zip(names, [*bshd, *bhsd]))


def test_regret_is_ranker_time_over_best_time_for_the_shared_mapped_graphs():
    ranker = {"g_bshd_gqa": 1.1, "g_bhsd_gqa": 2.0, "x": 5.0}
    best = {"g_bshd_gqa": 1.0, "g_bhsd_gqa": 1.0, "g_bhsd_mha": 1.0}
    assert rk.regrets(ranker, best, _ranker_map()) == pytest.approx(
        {"g_bshd_gqa": 1.1, "g_bhsd_gqa": 2.0}
    )


def test_a_cell_is_ok_when_it_costs_no_more_than_its_twin():
    (cell,) = rk.check(_ratios((1.05, 1.05), (1.06, 1.06)), _ranker_map())
    assert cell["verdict"] == "OK" and cell["delta"] == pytest.approx(0.01)
    assert cell["regret"] == pytest.approx(0.06) and cell[
        "twin_regret"
    ] == pytest.approx(0.05)


def test_a_cell_is_mis_ranked_when_it_regrets_more_than_three_points_over_its_twin():
    (cell,) = rk.check(_ratios((1.00, 1.00), (1.04, 1.04)), _ranker_map())
    assert cell["verdict"] == "MIS-RANKED" and cell["graphs_over_limit"] == []
    (cell,) = rk.check(_ratios((1.00, 1.00), (1.029, 1.029)), _ranker_map())
    assert cell["verdict"] == "OK"


def test_one_strided_graph_with_regret_above_ten_percent_is_enough():
    (cell,) = rk.check(_ratios((1.08, 1.08), (1.0, 1.11)), _ranker_map())
    assert cell["verdict"] == "MIS-RANKED" and cell["graphs_over_limit"] == [
        "g_bhsd_mha"
    ]
    assert cell["delta"] < rk.REGRET_DELTA_LIMIT


def test_a_twin_that_regrets_a_lot_is_not_blamed_on_strides():
    (cell,) = rk.check(_ratios((1.09, 1.09), (1.09, 1.09)), _ranker_map())
    assert cell["verdict"] == "OK" and cell["regret"] == pytest.approx(0.09)


def test_a_missing_twin_or_graph_leaves_the_cell_incomplete():
    ratios = _ratios((1.0, 1.0), (1.0, 1.0))
    del ratios["g_bhsd_mha"]
    assert rk.check(ratios, _ranker_map())[0]["verdict"] == "INCOMPLETE"
    assert (
        rk.check({k: v for k, v in ratios.items() if "bshd" not in k}, _ranker_map())[
            0
        ]["verdict"]
        == "INCOMPLETE"
    )


def test_a_layout_that_repeats_the_twin_is_not_a_ranker_cell():
    mapping = _ranker_map() + [
        _entry_map(shape="one", layout="batch_slice", same_as_bshd=True)
    ]
    cells = rk.check(
        _ratios((1.0, 1.0), (1.0, 1.0)) | {"g_batch_slice_one": 9.0}, mapping
    )
    assert [c["cell"][-1] for c in cells] == ["bhsd"]
    assert rk.repeats_twin(mapping) == ["g_batch_slice_one"]


def _paired_map():
    """BSHD on shapes a and b; batch_slice on both, repeating the twin on b."""
    return [
        _entry_map(shape="a", layout="bshd"),
        _entry_map(shape="b", layout="bshd"),
        _entry_map(shape="a", layout="batch_slice"),
        _entry_map(shape="b", layout="batch_slice", same_as_bshd=True),
    ]


def test_the_twin_is_the_bshd_graphs_of_the_same_shapes_only():
    """Shape b repeats its twin, so it is in neither side of the comparison; a twin
    regret averaged over b would hide the 5 points shape a loses to its strides."""
    ratios = {"g_bshd_a": 1.0, "g_bshd_b": 1.2, "g_batch_slice_a": 1.05}
    (cell,) = rk.check(ratios, _paired_map())
    assert cell["verdict"] == "MIS-RANKED" and cell["delta"] == pytest.approx(0.05)
    assert cell["twin_regret"] == pytest.approx(0.0)
    assert cell["graphs"] == 1 and cell["measured"] == 1


def test_a_cell_whose_paired_twin_was_not_measured_is_incomplete():
    mapping = [
        _entry_map(shape="a", layout="bshd"),
        _entry_map(shape="b", layout="bshd"),
        _entry_map(shape="a", layout="bhsd"),
        _entry_map(shape="b", layout="bhsd"),
    ]
    ratios = {"g_bshd_a": 1.0, "g_bhsd_a": 1.0, "g_bhsd_b": 1.0}
    (cell,) = rk.check(ratios, mapping)
    assert cell["verdict"] == "INCOMPLETE" and cell["measured"] == 2
    (cell,) = rk.check(ratios | {"g_bshd_b": 1.0}, mapping)
    assert cell["verdict"] == "OK"


def test_ranker_cli_does_not_count_a_repeating_layout_against_strict(world):
    mapping = _ranker_map() + [
        _entry_map(shape="one", layout="batch_slice", same_as_bshd=True)
    ]
    path = world["tmp"] / "map.yaml"
    path.write_text(yaml.safe_dump(mapping))
    ledger = [
        dict(graph_name=m["graph_name"], outcome="served", mean_ms=1.0) for m in mapping
    ]
    for arm in ("ranked", "best"):
        world["sweep"].phase(arm, "timing", 1, ledger=ledger)
    out = world["tmp"] / "ranker.yaml"
    argv = f"--sweep-output {world['sweep'].root} --ranker-arm ranked --best-arm best"
    argv += f" --map {path} --out {out} --strict"
    assert rk.main(argv.split()) == 0
    result = yaml.safe_load(out.read_text())
    assert [c["verdict"] for c in result["cells"]] == ["OK"]
    assert result["repeats_twin"] == ["g_batch_slice_one"]


def test_ranker_cli_reads_the_ledgers_of_both_arms(world, capsys):
    mapping = _ranker_map()
    path = world["tmp"] / "map.yaml"
    path.write_text(yaml.safe_dump(mapping))
    for r in (1, 2):
        world["sweep"].phase(
            "best",
            "timing",
            r,
            ledger=[
                dict(graph_name=m["graph_name"], outcome="served", mean_ms=1.0)
                for m in mapping
            ],
        )
        ranker = [
            dict(
                graph_name=m["graph_name"],
                outcome="served",
                mean_ms=1.0 if m["layout"] == "bshd" else 1.2,
            )
            for m in mapping
        ]
        world["sweep"].phase("ranked", "timing", r, ledger=ranker)
    out = world["tmp"] / "ranker.yaml"
    argv = f"--sweep-output {world['sweep'].root} --ranker-arm ranked --best-arm best"
    argv += f" --map {path} --out {out}"
    assert rk.main(argv.split()) == 0
    result = yaml.safe_load(out.read_text())
    assert result["cells"][0]["verdict"] == "MIS-RANKED"
    assert result["graph_ratios"]["g_bhsd_gqa"] == pytest.approx(1.2)
    assert rk.main((argv + " --strict").split()) == 1
    assert rk.main(argv.replace("--best-arm best", "--best-arm none").split()) == 2
    assert "no mapped graph in common" in capsys.readouterr().err


def test_the_pre_registered_bars_are_the_approved_values():
    """Fixed before any data was seen; changing one is a decision, not a tuning."""
    assert (rep.BAND_FLOOR, rep.DRIFT_MULTIPLIER) == (0.03, 2)
    assert (rk.REGRET_DELTA_LIMIT, rk.REGRET_GRAPH_LIMIT) == (0.03, 0.10)
