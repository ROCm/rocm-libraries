# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import hashlib
import json
import os
import re
import stat
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

import correctness_fakes as cf
from lib import bank_pool as bp
from lib import correctness as cx
from lib import correctness_gate as cg
from lib import gpu_lease
from lib import kernel_identity as ki
from lib.dat import kernel_dat_info
from test_run_orchestrator import arg, calls, env, make_config, run, stubs


@pytest.fixture
def world(tmp_path, monkeypatch):
    """The fake library, harness and spec, with device 0 the physical GPU 2."""
    lib = cf.write_library(tmp_path / "lib")
    harness = cf.write_harness(tmp_path / "errmap")
    (tmp_path / "state").mkdir()
    spec = {
        "names": cf.names(),
        "log": str(tmp_path / "calls.jsonl"),
        "state_dir": str(tmp_path / "state"),
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    monkeypatch.setenv("FAKE_ERRMAP_SPEC", str(path))
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "2,3")
    for var in ("HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"):
        monkeypatch.delenv(var, raising=False)

    class World:
        def __init__(self):
            self.base = tmp_path
            self.lib = lib
            self.harness = harness
            self.spec = spec
            self.log = tmp_path / "calls.jsonl"

        def set(self, **changes):
            self.spec.update(changes)
            path.write_text(json.dumps(self.spec))

        def cfg(self, **correctness):
            return cf.config(harness=str(self.harness), **correctness)

        def run(self, out="gate", cache="cache", cfg=None, **kw):
            spec = cg.Spec(
                cfg=cfg or self.cfg(),
                dat_dir=self.lib,
                library_stem=cf.STEM,
                out_dir=self.base / out,
                cache_dir=self.base / cache,
                devices=kw.pop("devices", [0]),
                **kw,
            )
            return cg.run_gate(spec)

        def blocked(self, out="gate"):
            doc = json.loads((self.base / out / cx.BLOCKLIST_FILE).read_text())
            return sorted(e["name"] for e in doc["entries"])

        def calls(self):
            return cf.harness_calls(self.log)

    return World()


def name(sid):
    return cf.names()[str(sid)]


# ── blocklists ───────────────────────────────────────────────────────────────


def test_lists_name_kernels_by_name_with_an_optional_class(tmp_path):
    p = tmp_path / "l.tsv"
    p.write_text("# a comment\n\nA\tCijk_one  # trailing\nCijk_two\n")
    bl = cx.read_list(p)
    assert [(e.name, e.reason) for e in bl.entries] == [
        ("Cijk_one", "A"),
        ("Cijk_two", ""),
    ]
    assert bl.blocks_name("Cijk_two") and not bl.blocks_name("Cijk_three")
    p.write_text("A\tB\tCijk_one\n")
    with pytest.raises(ValueError, match="l.tsv:1"):
        cx.read_list(p)
    with pytest.raises(ValueError, match="cannot read"):
        cx.read_list(tmp_path / "missing.tsv")


def test_a_blocklist_file_round_trips_and_ids_match_by_name_hash(tmp_path):
    k = kernel_dat_info(cf.solution(101, cf.TABLE[101]))
    bl = cx.resolved(cx.Blocklist([cx.Entry(k["kernel_name"], "C", source="g")]), [k])
    (e,) = bl.entries
    assert e.kernel_id == ki.kernel_id(k) and e.signature == ki.signature_label(k)
    cx.write_blocklist(tmp_path / "b.json", bl, device="x")
    back = cx.read_list(tmp_path / "b.json")
    assert back.entries == bl.entries
    assert back.blocks_id(ki.kernel_id(k)) and back.blocks(k)
    assert back.blocks({"kernel_id": ki.kernel_id(k)})
    other = kernel_dat_info(cf.solution(102, cf.TABLE[102]))
    assert not back.blocks(other) and not back.blocks_id(ki.kernel_id(other))
    assert back.blocked_indices({7: ki.kernel_id(k), 8: ki.kernel_id(other)}) == {7}
    (tmp_path / "bad.json").write_text('{"schema": "other"}')
    with pytest.raises(ValueError, match="bad.json"):
        cx.read_list(tmp_path / "bad.json")


def test_a_blocked_name_blocks_its_alias_group_whole(tmp_path):
    """A kernel whose operands the tensor data mover loads and its THA/THB
    twin run the same code: a blocklist naming either blocks both, by name,
    by kernel id, in the pool and in the sidecar stage06 stages; the same
    hints on kernels without such operands tell kernels apart as before."""
    import bench_fakes as bf
    import stage06_deploy_weights as s6
    from test_kernel_identity import tdm_solution

    sols = [tdm_solution(1, 0), tdm_solution(2, 1, tha=1, thb=1)]
    sols.append(tdm_solution(3, 2, thd=1))
    base, twin, other = (kernel_dat_info(s) for s in sols)
    for blocked in (base, twin):
        bl = cx.Blocklist([cx.Entry(blocked["kernel_name"], "C")])
        assert bl.blocks(base) and bl.blocks(twin) and not bl.blocks(other)
        assert bl.blocks_name(base["kernel_name"])
        assert bl.blocks_name(twin["kernel_name"])
        assert bl.blocks_id(ki.kernel_id(twin))
        assert not bl.blocks_id(ki.kernel_id(other))
        assert bl.kept([base, twin, other]) == [other]
    loose = [kernel_dat_info(tdm_solution(i, i, tdm=0, tha=i)) for i in (0, 1)]
    assert cx.Blocklist([cx.Entry(loose[1]["kernel_name"])]).kept(loose) == [loose[0]]
    lib = tmp_path / "lib"
    lib.mkdir()
    bf.write_logic(lib / f"{bf.LIBRARY_STEM}.dat", sols)
    lst = tmp_path / "l.tsv"
    lst.write_text(f"C\t{twin['kernel_name']}\n")
    cfg = {"arch": "gfx950", "correctness": {"lists": [str(lst)]}}
    assert s6.blocklist_names(cfg, bf.LIBRARY_STEM, lib, 0) == sorted(
        [base["kernel_name"], twin["kernel_name"]]
    )
    assert bp.config_pool(cfg, lib, bf.LIBRARY_STEM).indices == [3]


def test_shipped_lists_are_named_without_path_or_suffix(tmp_path):
    assert cx.list_path("gfx950_bbs_tn") == cx.LISTS_DIR / "gfx950_bbs_tn.tsv"
    assert cx.list_path("my.tsv") == Path("my.tsv")
    assert cx.list_path(str(tmp_path / "x")) == tmp_path / "x"


def test_the_shipped_gfx950_list_holds_names_and_classes_only():
    path = cx.LISTS_DIR / "gfx950_bbs_tn.tsv"
    bl = cx.read_list(path)
    assert len(bl) == len(set(bl.names)) > 0
    assert {e.reason for e in bl.entries} <= {"A", "B", "C", "D"}
    assert all(n.startswith(("Cijk_", "Custom_Cijk_")) for n in bl.names)
    text = path.read_text()
    assert "%" not in text and "/home" not in text and "/tmp" not in text


def test_config_blocklist_is_the_union_of_lists_and_the_gate_blocklist(tmp_path):
    a = tmp_path / "a.tsv"
    a.write_text("A\tCijk_a\n")
    b = tmp_path / "b.json"
    cx.write_blocklist(b, cx.Blocklist([cx.Entry("Cijk_b", "gate")]))
    assert cx.config_blocklist({}) is None
    cfg = {"correctness": {"lists": [str(a)], "blocklist": str(b)}}
    assert cx.config_blocklist(cfg).names == {"Cijk_a", "Cijk_b"}
    a.write_text("A\tCijk_c\n")
    os.utime(a, ns=(1, 1))
    assert cx.config_blocklist(cfg).names == {"Cijk_c", "Cijk_b"}
    with pytest.raises(ValueError, match="cannot be read"):
        cx.config_blocklist({"correctness": {"lists": [str(tmp_path / "gone")]}})


def test_a_gate_config_without_its_blocklist_warns_once(tmp_path, capsys):
    cx._WARNED.clear()
    a = tmp_path / "a.tsv"
    a.write_text("Cijk_a\n")
    cfg = {"correctness": {"enabled": True, "lists": [str(a)]}}
    assert cx.config_blocklist(cfg).names == {"Cijk_a"}
    assert cx.config_blocklist(cfg).names == {"Cijk_a"}
    assert capsys.readouterr().err.count("names no gate blocklist") == 1


def test_config_pool_leaves_blocked_kernels_out_of_table_and_bank(tmp_path):
    cf.write_library(tmp_path / "lib")
    lst = tmp_path / "l.tsv"
    lst.write_text(f"A\t{name(101)}\nB\t{name(200)}\nC\tCijk_not_in_the_pool\n")
    cfg = cf.config(lists=[str(lst)])
    full = bp.config_pool(cfg, tmp_path / "lib", cf.STEM, apply_blocklist=False)
    kept = bp.config_pool(cfg, tmp_path / "lib", cf.STEM)
    assert full.indices == [100, 101, 102, 103, 200, 201]
    assert kept.indices == [100, 102, 103, 201]
    assert kept.bank.stems == full.bank.stems == [cf.BANK]
    assert bp.bank_library_stems(cfg, tmp_path / "lib", cf.STEM) == [cf.BANK]


def test_sidecar_text_reads_back_as_names():
    text = cx.sidecar_text("Stem", ["b", "a", "b"])
    assert text.startswith("# ") and text.endswith("a\nb\n")


def test_sidecars_are_read_as_tensilelite_reads_them(tmp_path):
    p = tmp_path / ("Stem" + cx.SIDECAR_SUFFIX)
    p.write_text("# head\n\n  Cijk_a  \n#Cijk_b\nCijk_c\n")
    assert cx.read_sidecar(p) == ["Cijk_a", "Cijk_c"]


# ── the gate ─────────────────────────────────────────────────────────────────


def test_harness_problem_types():
    problem = cg.harness_problem(cf.config())
    args = problem.args()
    assert args[args.index("--compute") + 1] == "f32"
    assert args[args.index("--trans-a") + 1] == "T"
    assert args[args.index("--trans-b") + 1] == "N"
    assert (problem.trans_a, problem.trans_b) == (True, False)
    tf32 = cf.config()
    tf32["hipblaslt"].update(
        a_type="f32_r", b_type="f32_r", c_type="f32_r", d_type="f32_r"
    )
    tf32["hipblaslt"].update(compute_type="c_xf32_r", transA="N", transB="T")
    problem = cg.harness_problem(tf32)
    assert "xf32" in problem.args() and (problem.trans_a, problem.trans_b) == (
        False,
        True,
    )
    f8 = cf.config()
    f8["hipblaslt"]["a_type"] = "f8_r"
    with pytest.raises(cg.GateError, match="a_type"):
        cg.harness_problem(f8)
    scaled = cf.config()
    scaled["hipblaslt"]["scaleA"] = 3
    with pytest.raises(cg.GateError, match="scaled"):
        cg.harness_problem(scaled)


@pytest.mark.parametrize(
    "a_type, b_type, bits",
    [
        ("f8_r", "f8_r", (8, 8)),
        ("bf8_r", "bf8_r", (8, 8)),
        ("f4_r", "f4_r", (4, 4)),
        ("f8_r", "f4_r", (8, 4)),
    ],
)
def test_block_scaled_8_and_4_bit_inputs_run_their_own_shapes(a_type, b_type, bits):
    problem = cg.harness_problem(cf.mx_config(a_type, b_type))
    args = problem.args()
    assert args[args.index("--a-type") + 1] == cg.HARNESS_MX_TYPES[a_type][0]
    assert args[args.index("--b-type") + 1] == cg.HARNESS_MX_TYPES[b_type][0]
    assert args[args.index("--scale-a") + 1] == "3"
    assert args[args.index("--scale-b") + 1] == "3"
    assert problem.mx_bits == bits
    assert cg.shapes_of(problem) == cg.mx_shapes(*bits)
    assert cg.shapes_of(cg.harness_problem(cf.config())) == cg.SHAPES
    assert cg.shapes_of(None) == cg.SHAPES


@pytest.mark.parametrize(
    "change, match",
    [
        ({"scaleA": 1, "scaleB": 1}, "scaled"),
        ({"scaleB": 0}, "scaled"),
        ({"scaleA": 0, "scaleB": 0}, "a_type"),
        ({"a_type": "bf16_r", "b_type": "bf16_r"}, "a_type"),
        ({"a_type": "f6_r", "b_type": "f6_r"}, "a_type"),
        ({"c_type": "f8_r", "d_type": "f8_r"}, "c_type"),
    ],
)
def test_scales_and_types_the_harness_does_not_run_are_refused(change, match):
    cfg = cf.mx_config()
    cfg["hipblaslt"].update(change)
    with pytest.raises(cg.GateError, match=match):
        cg.harness_problem(cfg)


def _digest(doc, **kw):
    return hashlib.sha256(json.dumps(doc, **kw).encode()).hexdigest()[:16]


def test_the_bf16_and_f32_gate_is_byte_identical():
    """The protocol, every harness command and the summary's shapes of bf16
    and f32 problems are those of the gate before it ran 8- and 4-bit inputs
    (the digests were taken from it), so a verdict of such a kernel means
    what it meant; only the harness digest follows the harness source."""
    assert cg.protocol(cg.Settings(), "harness-id") == "629690fa63793e23"
    custom = cg.Settings(repetitions=4, trigger_repetitions=12, first_launch="all")
    assert cg.protocol(custom, "x") == "32fb9ce61321b1a8"
    cmds = []
    for layout in ("TN", "NN", "NT", "TT"):
        cfg = cf.config()
        cfg["hipblaslt"].update(transA=layout[0], transB=layout[1])
        p = cg.harness_problem(cfg)
        for s in cg.SHAPES:
            cmds.append(cg.Task(s.name, s, [1, 2], 3).command(Path("h"), p))
        kernels = {7: {"mt_k": 64}, 8: {"mt_k": 256}}
        unsupported = {s.name: {7, 8} for s in cg.SHAPES}
        for t in cg.roundup_tasks(unsupported, kernels, cg.Settings()):
            cmds.append(t.command(Path("h"), p))
        for t in cg.main_tasks([1, 2, 3, 4, 5], cg.Settings(), 2):
            cmds.append([t.tag, t.reps] + t.command(Path("h"), p))
    assert len(cmds) == 312 and _digest(cmds) == "e7d63c96d80847bd"
    shapes = [dict(s.record(), text=s.text()) for s in cg.SHAPES]
    assert _digest(shapes, sort_keys=True) == "4ed32c7f2ad731d8"


def _prime(n):
    return n > 1 and all(n % d for d in range(2, int(n**0.5) + 1))


@pytest.mark.parametrize("bits", [(8, 8), (4, 4), (8, 4)])
def test_mx_shapes_fit_block_scales_and_reach_their_edges(bits):
    shapes = cg.mx_shapes(*bits)
    by = {s.name: s for s in shapes}
    assert len(by) == len(shapes)
    about_k = {"small_m_long_n_k_aligned", "k_128", "k_256"}
    for s in shapes:
        assert s.k % 32 == 0, s.name
        assert s.name in about_k or s.k // 32 % 2, f"{s.name}: K a multiple of 64"
        assert s.m % 8, f"{s.name}: M a multiple of 8"
    assert sum(_prime(s.m) for s in shapes) > len(shapes) // 2
    for name in ("a_over_4gib", "a_over_4gib_batched"):
        s = by[name]
        assert s.m * s.k * s.batch * bits[0] // 8 >= 1 << 32
        assert s.m * (s.k - 64) * s.batch * bits[0] // 8 < 1 << 32, "the least K"
    for name in ("b_over_4gib", "b_over_4gib_batched"):
        s = by[name]
        assert s.n * s.k * s.batch * bits[1] // 8 >= 1 << 32
        assert s.n * (s.k - 64) * s.batch * bits[1] // 8 < 1 << 32, "the least K"
    assert by["a_over_4gib_batched"].batch > 1 and by["b_over_4gib_batched"].batch > 1
    bias = [s for s in shapes if s.bias]
    assert 0 < len(bias) < len(shapes)
    assert any(s.trigger and s.batch > 1 for s in bias)
    assert any(s.separate_d and s.beta for s in bias)
    assert any(s.name.startswith("a_over_4gib") for s in bias)
    assert by["small_m_batched_bias"] == replace(
        by["small_m_batched"], name="small_m_batched_bias", bias=True
    )


@pytest.mark.parametrize("layout", ["TN", "NN", "NT", "TT"])
def test_mx_leading_dimensions_are_padded_past_the_stored_k(layout):
    """The harness stores a block-scaled A and B with K rounded up to a
    multiple of 256 (zeros past K, which the scale tensor's padding
    multiplies), so a padded leading dimension counts from there wherever K
    is the stored rows."""
    cfg = cf.mx_config()
    cfg["hipblaslt"].update(transA=layout[0], transB=layout[1])
    problem = cg.harness_problem(cfg)
    assert problem.stored_k == cg.MX_STORED_K == 256
    assert cg.harness_problem(cf.config()).stored_k == 1
    shape = next(s for s in cg.shapes_of(problem) if s.name == "alpha_beta_padded")
    args = cg.Task("t", shape, [1], 3).command(Path("h"), problem)[1:]
    ld = {f: int(args[args.index(f) + 1]) for f in ("--lda", "--ldb", "--ldc")}
    k = -(-shape.k // 256) * 256
    assert k > shape.k
    assert ld["--lda"] == (k if layout[0] == "T" else shape.m) + shape.pad_a
    assert ld["--ldb"] == (shape.n if layout[1] == "T" else k) + shape.pad_b
    assert ld["--ldc"] == shape.m + shape.pad_c


@pytest.mark.parametrize("layout", ["TN", "NN", "NT", "TT"])
@pytest.mark.parametrize("a_type", ["f8_r", "bf8_r", "f4_r"])
def test_the_harness_takes_every_mx_task_in_every_layout(layout, a_type):
    """Every command of the gate's passes satisfies what the harness checks
    of block-scaled inputs: K a multiple of 32, an even leading dimension of
    a packed f4 operand, C, D and the bias of bf16 or f32."""
    cfg = cf.mx_config(a_type)
    cfg["hipblaslt"].update(transA=layout[0], transB=layout[1])
    problem = cg.harness_problem(cfg)
    shapes = cg.shapes_of(problem)
    kernels = {7: {"mt_k": 512}, 8: {"mt_k": 128}}
    tasks = cg.main_tasks([7, 8], cg.Settings(), 1, shapes)
    unsupported = {s.name: {7, 8} for s in shapes}
    tasks += cg.roundup_tasks(unsupported, kernels, cg.Settings(), shapes)
    tasks += cg.first_launch_tasks(kernels, [7, 8], cg.Settings())
    for t in tasks:
        args = t.command(Path("h"), problem)[1:]
        opts = dict(zip(args[4::2], args[5::2]))
        m, n, k = (int(x) for x in args[:3])
        assert not cf._type_error(k, opts), t.tag
        assert not cf._leading_dimension_error(m, n, k, opts), t.tag
        assert ("--bias" in opts) == t.shape.bias


def test_the_gate_runs_block_scaled_problems_on_their_shapes(world):
    world.set(fail={"101": "shape:13x65"})
    s = world.run(cfg=cf.mx_config("f4_r", harness=str(world.harness)))
    assert s["verdicts"] == {"fail": 1, "pass": 5}
    assert world.blocked() == [name(101)]
    shapes = cg.mx_shapes(4, 4)
    calls = world.calls()
    ran = {tuple(c["args"][:4]) for c in calls}
    assert {(str(x.m), str(x.n), str(x.k), str(x.batch)) for x in shapes} <= ran
    assert ("13", "65", "383", "2048") not in ran, "a bf16 shape"
    for c in calls:
        a = c["args"]
        assert a[a.index("--a-type") + 1] == a[a.index("--b-type") + 1] == "f4"
        assert a[a.index("--scale-a") + 1] == a[a.index("--scale-b") + 1] == "3"
    biased = {tuple(c["args"][:4]) for c in calls if "--bias" in c["args"]}
    assert biased == {
        (str(x.m), str(x.n), str(x.k), str(x.batch)) for x in shapes if x.bias
    }
    v = json.loads((world.base / "gate" / cg.VERDICTS_FILE).read_text())
    by_sid = {k["solution_index"]: k for k in v["kernels"]}
    assert by_sid[101]["evidence"]["failing_shapes"] == {
        "small_m_batched": [10, 10],
        "small_m_batched_bias": [10, 10],
    }
    summary = json.loads((world.base / "gate" / cg.SUMMARY_FILE).read_text())
    assert [x["name"] for x in summary["shapes"]] == [x.name for x in shapes]
    assert [x.get("bias", False) for x in summary["shapes"]] == [x.bias for x in shapes]
    bf16 = world.run(out="gate_bf16", cache="cache_bf16")
    assert bf16["protocol"] != s["protocol"]
    assert all("bias" not in x for x in bf16["shapes"])


@pytest.mark.parametrize("layout", ["TN", "NN", "NT", "TT"])
def test_padded_leading_dimensions_follow_the_layout(layout):
    ta, tb = layout[0] == "T", layout[1] == "T"
    shape = next(s for s in cg.SHAPES if s.name == "alpha_beta_padded")
    args = shape.args(ta, tb)
    ld = {f: int(args[args.index(f) + 1]) for f in ("--lda", "--ldb", "--ldc")}
    rows_a = shape.k if ta else shape.m
    rows_b = shape.n if tb else shape.k
    assert ld == {
        "--lda": rows_a + shape.pad_a,
        "--ldb": rows_b + shape.pad_b,
        "--ldc": shape.m + shape.pad_c,
    }
    assert int(args[args.index("--ldd") + 1]) == ld["--ldc"]
    assert min(shape.pad_a, shape.pad_b, shape.pad_c) > 0
    kernels = {7: {"mt_k": 64}}
    (grown,) = cg.roundup_tasks({"alpha_beta_padded": {7}}, kernels, cg.Settings())
    assert grown.shape.k == 576 and grown.shape.pad_a == shape.pad_a
    args = grown.shape.args(ta, tb)
    assert int(args[args.index("--lda") + 1]) == (576 if ta else shape.m) + shape.pad_a
    assert int(args[args.index("--ldb") + 1]) == (shape.n if tb else 576) + shape.pad_b


@pytest.mark.parametrize("layout", ["NN", "NT", "TT"])
def test_the_gate_runs_every_layout(world, layout):
    world.set(fail={"101": "always"})
    cfg = world.cfg()
    cfg["hipblaslt"].update(transA=layout[0], transB=layout[1])
    s = world.run(cfg=cfg)
    assert s["verdicts"] == {"fail": 1, "pass": 5}
    assert world.blocked() == [name(101)]
    for c in world.calls():
        a = c["args"]
        assert (a[a.index("--trans-a") + 1], a[a.index("--trans-b") + 1]) == (
            layout[0],
            layout[1],
        )


def test_settings_need_three_repetitions():
    assert cg.Settings.of(cf.config()) == cg.Settings()
    with pytest.raises(cg.GateError, match="at least 3"):
        cg.Settings.of(cf.config(repetitions=2))
    with pytest.raises(cg.GateError, match="first_launch"):
        cg.Settings.of(cf.config(first_launch="some"))


def test_the_gate_blocks_failing_kernels_and_keeps_the_rest(world):
    world.set(fail={"101": "always", "201": "shape:65x131071"})
    s = world.run()
    assert (
        s["pool"] == 6
        and s["checked"] == 6
        and s["verdicts"]
        == {
            "fail": 2,
            "pass": 4,
        }
    )
    assert world.blocked() == [name(101), name(201)]
    v = json.loads((world.base / "gate" / cg.VERDICTS_FILE).read_text())
    by_sid = {k["solution_index"]: k for k in v["kernels"]}
    assert by_sid[201]["evidence"]["failing_shapes"] == {
        "small_m_long_n": [10, 10],
        "small_m_long_n_k_aligned": [10, 10],
    }
    assert by_sid[100]["verdict"] == "pass" and by_sid[100]["source"] == "gate"
    assert by_sid[101]["kernel_id"] == ki.kernel_id(
        kernel_dat_info(cf.solution(101, cf.TABLE[101]))
    )
    calls = world.calls()
    shapes = {tuple(c["args"][:4]) for c in calls}
    assert {(str(s.m), str(s.n), str(s.k), str(s.batch)) for s in cg.SHAPES} <= shapes
    reps = {c["args"][c["args"].index("--reps") + 1] for c in calls}
    assert reps == {"1", "3", "10"}, "first launches of the hand-written 103 at 1"
    assert all("--trans-a" in c["args"] and "--compute" in c["args"] for c in calls)


def test_a_second_gate_checks_nothing_and_takes_no_gpu(world, monkeypatch):
    world.set(fail={"102": "always"})
    world.run()
    n = len(world.calls())
    held = []
    monkeypatch.setattr(gpu_lease, "hold", lambda *a, **k: held.append(a))
    s = world.run(out="gate2")
    assert s["checked"] == 0 and s["cached"] == 6 and held == []
    assert len(world.calls()) == n
    assert world.blocked("gate2") == [name(102)]
    v = json.loads((world.base / "gate2" / cg.VERDICTS_FILE).read_text())
    assert {k["source"] for k in v["kernels"]} == {"cache"}


def test_listed_kernels_are_blocked_unchecked(world, tmp_path):
    lst = tmp_path / "known.tsv"
    lst.write_text(f"D\t{name(103)}\n")
    world.set(fail={"103": "always"})
    s = world.run(cfg=world.cfg(lists=[str(lst)]))
    assert s["listed"] == 1 and s["checked"] == 5
    assert world.blocked() == [name(103)]
    assert all(
        "103" not in c["args"][c["args"].index("--ids") + 1].split(",")
        for c in world.calls()
    )


def test_hand_written_kernels_get_a_first_launch_and_k_rounded_up(world):
    world.set(fail={"103": "first"}, depth_u={"103": 64})
    s = world.run()
    assert world.blocked() == [name(103)]
    calls = world.calls()
    alone = [c for c in calls if c["args"][c["args"].index("--ids") + 1] == "103"]
    first = [c for c in alone if c["args"][:4] == ["625", "77", "64", "1"]]
    assert len(first) == 3, "processes of its own at K = DepthU"
    assert all(c["args"][c["args"].index("--reps") + 1] == "1" for c in first)
    grown = {c["args"][2] for c in alone}
    assert {"384", "896", "4416", "29760"} <= grown and "383" not in grown
    v = json.loads((world.base / "gate" / cg.VERDICTS_FILE).read_text())
    ev = next(k for k in v["kernels"] if k["solution_index"] == 103)["evidence"]
    assert "first_launch_k64" in ev["failing_shapes"]
    assert "small_m_batched" in ev["unsupported_shapes"]
    assert s["verdicts"]["fail"] == 1


def test_every_first_launch_is_a_fresh_process(world):
    # 103 fails only on the first GEMM of the second process that launches
    # it first: one process with several repetitions would never see it. On
    # two GPUs no task of the main pass starts with it.
    world.set(fail={"103": "first_in:2"}, depth_u={"103": 64})
    world.run(devices=[0, 1])
    assert world.blocked() == [name(103)]
    v = json.loads((world.base / "gate" / cg.VERDICTS_FILE).read_text())
    ev = next(k for k in v["kernels"] if k["solution_index"] == 103)["evidence"]
    assert ev["failing_shapes"] == {"first_launch_k64": [1, 3]}


def test_first_launch_policy_all_and_none(world):
    world.set(fail={"101": "first"})
    world.run(cfg=world.cfg(first_launch="none"))
    assert world.blocked() == []
    world.run(out="gate2", cache="cache2", cfg=world.cfg(first_launch="all"))
    assert world.blocked("gate2") == [name(101)]


def test_failures_are_followed_up_in_their_family(world):
    # 102 fails on its fifth repetition of a shape the main pass runs 3 times.
    world.set(fail={"101": "shape:524287x3", "102": "rate:5@524287x3"})
    world.run()
    assert world.blocked() == [name(101), name(102)]
    (follow,) = [
        c for c in world.calls() if c["args"][c["args"].index("--ids") + 1] == "102"
    ]
    assert follow["args"][:2] == ["524287", "3"]
    assert follow["args"][follow["args"].index("--reps") + 1] == "10"
    v = json.loads((world.base / "gate" / cg.VERDICTS_FILE).read_text())
    ev = next(k for k in v["kernels"] if k["solution_index"] == 102)["evidence"]
    assert list(ev["failing_shapes"]) == ["a_over_4gib_followup"]


def test_a_crash_that_repeats_alone_blocks_its_kernel_and_the_others_go_on(world):
    world.set(crash=[101])
    s = world.run()
    assert world.blocked() == [name(101)]
    assert s["harness_crashes"] > 0 and s["unconfirmed_crashes"] > 0
    v = json.loads((world.base / "gate" / cg.VERDICTS_FILE).read_text())
    verdicts = {k["solution_index"]: k["verdict"] for k in v["kernels"]}
    assert verdicts == {
        100: "pass",
        101: "crash",
        102: "pass",
        103: "pass",
        200: "pass",
        201: "pass",
    }
    assert "crashed or hung, also alone" in json.dumps(v)
    ids = [c["args"][c["args"].index("--ids") + 1] for c in world.calls()]
    assert ids.count("101") == len(cg.SHAPES), "each crash again alone"
    # A crash holds under its protocol only: another one checks it again.
    s = world.run(out="gate2")
    assert s["checked"] == 0 and world.blocked("gate2") == [name(101)]
    world.set(crash=[])
    s = world.run(out="gate3", cfg=world.cfg(repetitions=4))
    assert s["checked"] == 6 and world.blocked("gate3") == []


def test_a_crash_that_does_not_repeat_alone_blames_no_kernel(world):
    world.set(crash_once=[102])
    s = world.run()
    assert world.blocked() == [] and s["verdicts"] == {"pass": 6}
    assert s["unconfirmed_crashes"] == 1
    v = json.loads((world.base / "gate" / cg.VERDICTS_FILE).read_text())
    ev = next(k for k in v["kernels"] if k["solution_index"] == 102)["evidence"]
    assert ev["unconfirmed_crashes"] == 1 and ev["crashes"] == 0


def test_a_hang_is_a_crash_after_the_timeout(world):
    world.set(hang=[200], hang_shape="524287x3")
    world.run(cfg=world.cfg(timeout_s=1.0))
    assert world.blocked() == [name(200)]
    v = json.loads((world.base / "gate" / cg.VERDICTS_FILE).read_text())
    ev = next(k for k in v["kernels"] if k["solution_index"] == 200)["evidence"]
    assert ev["failing_shapes"] == {"a_over_4gib": [1, 1]}
    assert ev["hangs"] == ev["crashes"] == 1


def test_a_hang_holds_only_while_the_timeout_is_no_longer(world):
    """A kernel that hung under a short timeout may finish under a longer
    one: its cached crash holds for timeouts up to the one it hung under."""
    world.set(hang=[200], hang_shape="524287x3")
    world.run(cfg=world.cfg(timeout_s=1.0))
    (rec,) = [
        r
        for r in cg.VerdictCache(world.base / "cache").records()
        if r["verdict"] == "crash"
    ]
    assert rec["timeout_s"] == 1.0
    s = world.run(out="gate2", cfg=world.cfg(timeout_s=0.5))
    assert s["checked"] == 0 and world.blocked("gate2") == [name(200)]
    world.set(hang=[])
    s = world.run(out="gate3", cfg=world.cfg(timeout_s=600.0))
    assert s["checked"] == 1 and world.blocked("gate3") == []
    # A crash that is no hang holds whatever the timeout.
    world.set(crash=[101])
    world.run(out="gate4", cache="cache4")
    s = world.run(out="gate5", cache="cache4", cfg=world.cfg(timeout_s=1200.0))
    assert s["checked"] == 0 and world.blocked("gate5") == [name(101)]


def test_the_timeout_bounds_the_silence_of_a_process_not_its_length(world):
    # Each process of one shape takes far longer than the timeout but prints
    # a line more often than that.
    world.set(delay=0.3, delay_shape="524287x3")
    s = world.run(cfg=world.cfg(timeout_s=1.0, first_launch="none"))
    assert s["verdicts"] == {"pass": 6} and s["harness_crashes"] == 0
    assert sum(t > 2.0 for t in _walls(world)) == 2


def _walls(world):
    out = []
    for p in (world.base / "gate" / "runs").glob("gpu*.jsonl"):
        for line in p.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("task_done"):
                out.append(rec["wall_s"])
    return out


def test_a_process_killed_from_outside_blames_no_kernel(world):
    world.set(killed=[200], fail={"101": "always"})
    with pytest.raises(cg.GateError, match="killed by SIGKILL from outside"):
        world.run()
    cached = cg.VerdictCache(world.base / "cache").records()
    assert {r["name"]: r["verdict"] for r in cached} == {name(101): "fail"}


def test_the_culprit_is_a_kernel_the_harness_found(world):
    world.set(missing=[101], crash=[102])
    s = world.run()
    assert world.blocked() == [name(102)] and s["not_found"] == 1
    v = json.loads((world.base / "gate" / cg.VERDICTS_FILE).read_text())
    verdicts = {k["solution_index"]: k["verdict"] for k in v["kernels"]}
    assert verdicts[101] == "untested" and verdicts[102] == "crash"


def test_a_harness_that_finds_none_of_its_kernels_is_refused(world):
    world.set(missing=sorted(int(s) for s in cf.names()))
    with pytest.raises(cg.HarnessMismatch, match="finds none"):
        world.run()
    assert not (world.base / "cache" / cg.CACHE_FILE).exists()


def test_a_first_launch_of_a_kernel_found_nowhere_leaves_it_untested(world):
    """The hand-written 103's first launches hold it alone: a harness that
    finds it nowhere leaves it untested, as the main pass does, while one
    that finds it among other kernels but not alone is refused."""
    world.set(missing=[103], depth_u={"103": 64})
    s = world.run()
    v = json.loads((world.base / "gate" / cg.VERDICTS_FILE).read_text())
    verdicts = {k["solution_index"]: k["verdict"] for k in v["kernels"]}
    assert verdicts[103] == "untested" and world.blocked() == []
    assert s["not_found"] == 1
    firsts = [
        c
        for c in world.calls()
        if c["args"][c["args"].index("--ids") + 1] == "103"
        and c["args"][:4] == ["625", "77", "64", "1"]
    ]
    assert len(firsts) == 3
    world.set(missing=[], missing_alone=[103])
    with pytest.raises(cg.HarnessMismatch, match="as the only GEMM of a process"):
        world.run(out="gate2", cache="cache2")
    assert not (world.base / "cache2" / cg.CACHE_FILE).exists()


def test_a_harness_that_fails_before_any_kernel_stops_the_gate(world):
    world.set(broken=True)
    with pytest.raises(cg.GateError, match="before it ran a kernel"):
        world.run()
    assert not (world.base / "cache" / cg.CACHE_FILE).exists()


def test_a_kernel_no_shape_supports_is_untested(world):
    world.set(unsupported=[100])
    s = world.run()
    assert s["verdicts"] == {"pass": 5, "untested": 1} and world.blocked() == []


def test_a_harness_of_another_build_is_refused_at_its_first_line(world):
    names = cf.names()
    names["100"] = "Cijk_someone_else"
    world.set(names=names, delay=0.2)
    with pytest.raises(cg.HarnessMismatch, match="solution 100"):
        world.run()
    assert len(world.calls()) == 1
    assert max(_walls(world) or [0.0]) < 1.0
    assert not (world.base / "cache" / cg.CACHE_FILE).exists()


def test_verdicts_no_pass_can_change_are_cached_before_the_follow_ups(world):
    # 101 fails a shape; its family's 102 passed it, so a follow-up runs
    # 102 alone, which this harness cannot.
    world.set(fail={"101": "shape:524287x3"}, broken_ids=["102"])
    with pytest.raises(cg.GateError, match="before it ran a kernel"):
        world.run()
    cached = {
        r["name"]: r["verdict"] for r in cg.VerdictCache(world.base / "cache").records()
    }
    assert cached == {
        name(100): "pass",
        name(101): "fail",
        name(103): "pass",
        name(200): "pass",
        name(201): "pass",
    }
    world.set(broken_ids=[])
    s = world.run(out="gate2")
    assert s["cached"] == 5 and s["checked"] == 1
    assert world.blocked("gate2") == [name(101)]


def test_another_protocol_checks_passes_again_but_keeps_failures(world):
    world.set(fail={"101": "always"})
    world.run()
    world.set(fail={})
    s = world.run(out="gate2", cfg=world.cfg(repetitions=4))
    assert s["cached"] == 1 and s["checked"] == 5
    assert world.blocked("gate2") == [name(101)]


def test_a_rebuilt_library_file_is_checked_again(world):
    world.run()
    (world.lib / f"{cf.BANK}.co").write_bytes(b"new code")
    s = world.run(out="gate2")
    assert s["cached"] == 4 and s["checked"] == 2


def test_a_dry_run_checks_nothing(world):
    s = world.run(dry_run=True)
    assert s["checked"] == 0 and world.calls() == []
    v = json.loads((world.base / "gate" / cg.VERDICTS_FILE).read_text())
    assert {k["verdict"] for k in v["kernels"]} == {"unchecked"}


def test_a_type_the_harness_does_not_run_fails_only_when_a_kernel_needs_it(world):
    """A config the harness cannot check reads its pool and writes its
    records under the bf16 protocol as before; it fails once a kernel needs
    a check."""
    scalar = cf.mx_config("f8_r", harness=str(world.harness))
    scalar["hipblaslt"].update(scaleA=1, scaleB=1)
    s = world.run(cfg=scalar, dry_run=True)
    assert s["checked"] == 0 and world.calls() == []
    assert [x["name"] for x in s["shapes"]] == [x.name for x in cg.SHAPES]
    with pytest.raises(cg.GateError, match="scaled"):
        world.run(cfg=scalar, out="gate2")
    assert world.calls() == []


def test_the_gate_runs_under_the_lease_of_its_gpus(world, monkeypatch):
    seen = []
    real = gpu_lease.hold_usable

    def hold(devices, purpose, *a, **k):
        seen.append((list(devices), purpose))
        return real(devices, purpose, *a, **k)

    monkeypatch.setattr(gpu_lease, "hold_usable", hold)
    s = world.run(devices=[0, 1])
    assert seen == [([0, 1], "correctness gate")] and s["gpus"] == [2, 3]
    rocr = {
        json.loads(line)["gpu"]
        for line in (world.base / "gate" / "runs" / "gpu2.jsonl")
        .read_text()
        .splitlines()
    }
    assert rocr == {2}


def test_devices_of_another_kind_are_refused(world, monkeypatch):
    from lib import hardware

    real = hardware.device_n_cu

    def n_cu(cfg, device=0, topology=None):
        return 304 if device == 1 else real(cfg, device, topology)

    cfg = world.cfg()
    cfg["hardware"].pop("n_cu")
    monkeypatch.setattr(hardware, "device_n_cu", n_cu)
    monkeypatch.setattr(
        hardware,
        "device_hardware",
        lambda c, d=0, t=None: hardware.DeviceHardware(
            n_cu=304 if d == 1 else 256, lds_bytes=163840, l2_bytes=4194304
        ),
    )
    with pytest.raises(cg.GateError, match="one kind of device"):
        world.run(cfg=cfg, devices=[0, 1])


def test_the_harness_is_built_once_against_the_build(tmp_path, monkeypatch):
    build = tmp_path / "build"
    (build / "library").mkdir(parents=True)
    (build / "library" / "libhipblaslt.so").write_bytes(b"lib")
    (build / "CMakeCache.txt").write_text(
        f"hipblaslt_SOURCE_DIR:STATIC={tmp_path / 'src'}\n"
    )
    compiler = tmp_path / "cxx"
    compiler.write_text(
        f"#!{sys.executable}\n"
        "import sys, pathlib\n"
        "out = sys.argv[sys.argv.index('-o') + 1]\n"
        "pathlib.Path(out).write_text('built')\n"
        f"open({str(tmp_path / 'builds')!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
    )
    compiler.chmod(compiler.stat().st_mode | stat.S_IXUSR)
    h = cg.harness(tmp_path / "cache", build, compiler=str(compiler))
    assert h.built and h.path.read_text() == "built"
    cmd = (tmp_path / "builds").read_text()
    assert f"-I{tmp_path / 'src' / 'library' / 'include'}" in cmd
    assert f"-L{build / 'library'}" in cmd and "-Wl,-rpath," in cmd
    again = cg.harness(tmp_path / "cache", build, compiler=str(compiler))
    assert not again.built and again.path == h.path
    assert len((tmp_path / "builds").read_text().splitlines()) == 1
    (build / "library" / "libhipblaslt.so").write_bytes(b"rebuilt")
    assert cg.harness(tmp_path / "cache", build, compiler=str(compiler)).built
    compiler.write_text(f"#!{sys.executable}\nimport sys\nsys.exit(1)\n")
    (build / "library" / "libhipblaslt.so").write_bytes(b"again")
    with pytest.raises(cg.GateError, match="building the harness failed"):
        cg.harness(tmp_path / "cache", build, compiler=str(compiler))


def test_the_harness_source_builds_with_the_documented_options():
    text = cg.HARNESS_SOURCE.read_text()
    for flag in (
        "--ids",
        "--reps",
        "--alpha",
        "--beta",
        "--sep",
        "--trans-a",
        "--compute",
        "--scale-a",
        "--scale-b",
        "--bias",
        "--bias-type",
        "--corrupt-ref",
    ):
        assert f'"{flag}"' in text
    for attribute in (
        "HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0",
        "HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER",
        "HIPBLASLT_MATMUL_DESC_B_SCALE_MODE",
        "HIPBLASLT_EPILOGUE_BIAS",
        "HIPBLASLT_MATMUL_DESC_BIAS_DATA_TYPE",
    ):
        assert attribute in text
    problem = cg.ProblemType(("--x", "y"), True, False)
    args = cg.Task("t", cg.SHAPES[11], [1, 2], 3).command(Path("h"), problem)
    assert args[:5] == ["h", "1537", "515", "517", "1"]
    assert "--lda" in args and "--beta" in args
    assert args[-6:] == ["--x", "y", "--trans-a", "T", "--trans-b", "N"]
    assert '"sids"' in text, "the ready line names the solutions it found"


def _decode(kind, code):
    """The value of an E4M3 (f8), E5M2 (bf8) or E2M1 (f4) code."""
    e_bits, m_bits = {"f8": (4, 3), "bf8": (5, 2), "f4": (2, 1)}[kind]
    sign = code >> (e_bits + m_bits)
    exp = (code >> m_bits) & ((1 << e_bits) - 1)
    man = code & ((1 << m_bits) - 1)
    bias = (1 << (e_bits - 1)) - 1
    if exp == 0:
        value = man / (1 << m_bits) * 2.0 ** (1 - bias)
    else:
        value = (1 + man / (1 << m_bits)) * 2.0 ** (exp - bias)
    return -value if sign else value


def test_the_harness_writes_8_and_4_bit_integers_exactly():
    """Its codes of -2 .. 2 decode to those integers, and the scales of A
    and B are distinct powers of 2 other than 1, so a kernel that drops a
    scale or applies one operand's to both is wrong."""
    text = cg.HARNESS_SOURCE.read_text()
    for kind in ("f8", "bf8", "f4"):
        found = re.search(rf"\b{kind}\[5\]\s*=\s*\{{([^}}]*)\}}", text)
        codes = [int(x, 16) for x in found.group(1).split(",")]
        assert [_decode(kind, c) for c in codes] == [-2, -1, 0, 1, 2], kind
    exps = [int(re.search(rf"kScaleExp{op}\s*=\s*(\d+)", text).group(1)) for op in "AB"]
    assert 0 not in exps and exps[0] != exps[1]


def test_the_cache_keeps_a_failure_under_any_protocol(tmp_path):
    cache = cg.VerdictCache(tmp_path)
    cache.add(
        [
            {
                "kernel_id": "k",
                "device": "d",
                "code": "c",
                "protocol": "p1",
                "verdict": "pass",
            }
        ]
    )
    assert cache.lookup("k", "d", "c", "p1")["verdict"] == "pass"
    assert cache.lookup("k", "d", "c", "p2") is None
    assert cache.lookup("k", "other", "c", "p1") is None
    cache.add(
        [
            {
                "kernel_id": "k",
                "device": "d",
                "code": "c",
                "protocol": "p2",
                "verdict": "fail",
            }
        ]
    )
    assert cache.lookup("k", "d", "c", "p1")["verdict"] == "fail"
    (tmp_path / cg.CACHE_FILE).write_text(
        (tmp_path / cg.CACHE_FILE).read_text() + '{"truncated'
    )
    assert cg.VerdictCache(tmp_path).lookup("k", "d", "c", "p3")["verdict"] == "fail"


def test_stage00_writes_the_run_blocklist(world, tmp_path, capsys):
    import stage00_correctness_gate as s00

    world.set(fail={"201": "always"})
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(world.cfg(cache_dir=str(tmp_path / "shared"))))
    argv = ["--config-yaml", str(cfg_path), "--out-dir", str(tmp_path / "out")]
    argv += ["--dat-dir", str(world.lib), "--devices", "0"]
    assert s00.main(argv) == 0
    assert world.blocked("out") == [name(201)]
    assert (tmp_path / "shared" / cg.CACHE_FILE).is_file()
    assert "MT128x128x32" in capsys.readouterr().err
    bad = world.cfg()
    bad["hipblaslt"]["a_type"] = "f8_r"
    cfg_path.write_text(yaml.safe_dump(bad))
    assert s00.main(argv) == 1
    mx = cf.mx_config("f8_r", harness=str(world.harness))
    cfg_path.write_text(yaml.safe_dump(mx))
    argv[argv.index("--out-dir") + 1] = str(tmp_path / "out_mx")
    assert s00.main(argv) == 0
    assert world.blocked("out_mx") == [name(201)]
    mx["hipblaslt"].update(scaleA=1, scaleB=1)
    cfg_path.write_text(yaml.safe_dump(mx))
    argv[argv.index("--out-dir") + 1] = str(tmp_path / "out_scalar")
    assert s00.main(argv) == 1


def test_stage00_reports_a_busy_lease_with_2(world, tmp_path, monkeypatch):
    import stage00_correctness_gate as s00

    def busy(*a, **k):
        raise gpu_lease.LeaseBusy(2, None, tmp_path / "gpu2.lock")

    monkeypatch.setattr(gpu_lease, "hold_usable", busy)
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(world.cfg()))
    argv = ["--config-yaml", str(cfg_path), "--out-dir", str(tmp_path / "out")]
    assert s00.main([*argv, "--dat-dir", str(world.lib), "--quiet"]) == 2


# ── where the blocklist applies ──────────────────────────────────────────────


@pytest.fixture
def rows(tmp_path):
    """Enriched rows of the stage05 test pool with their record, its library,
    and a blocklist of the kernel that wins most GEMMs."""
    import collections

    from test_kernel_identity import pool_library, record_rows
    from test_stage05_train import CELLS3, kernel_pool, shapes_for, write_round

    pool = kernel_pool()
    ident = pool_library(tmp_path / "lib", pool, stem=cf.STEM)
    data = write_round(tmp_path / "rows", shapes_for(3, 6, CELLS3), pool)
    record_rows(data, ident, cf.STEM)
    import stage05_train as s5

    cells, _ = s5.load_enriched_chunks(str(data), True, identity=ident)
    gemms = [g for gs in cells.values() for g in gs]
    wins = collections.Counter(
        min((c for c in g["candidates"] if not c["is_skip"]), key=lambda c: c["us"])[
            "sol_idx_global"
        ]
        for g in gemms
    )
    top = wins.most_common(1)[0][0]
    name_of = {sid: k["kernel_name"] for sid, k in ident.kernels.items()}
    bl = cx.Blocklist([cx.Entry(name_of[top], "C")])
    return data, ident, bl, top, gemms


def _winner_sids(gemms):
    return {
        (g["m"], g["n"], g["k"], g["batch_count"]): min(
            (c for c in g["candidates"] if not c["is_skip"]), key=lambda c: c["us"]
        )["sol_idx_global"]
        for g in gemms
    }


def test_stage05_loads_no_row_of_a_blocked_kernel(rows):
    import stage05_train as s5

    data, ident, bl, top, before = rows
    for identity in (ident, None):
        cells, stats = s5.load_enriched_chunks(
            str(data), True, identity=identity, blocklist=bl
        )
        gemms = [g for gs in cells.values() for g in gs]
        assert stats["rows_blocked"] > 0
        assert all(c["sol_idx_global"] != top for g in gemms for c in g["candidates"])
        for g in gemms:
            tested = [c["us"] for c in g["candidates"] if not c["is_skip"]]
            assert g["winner_us"] == min(tested)
    assert top in _winner_sids(before).values()


def test_deploy_loss_loads_the_rows_stage05_trained_on(rows, tmp_path):
    """deploy_loss.py reads stage05's inputs as stage05 did: through the
    blocklist of the config snapshot its metrics.json names."""
    import argparse
    import pickle

    import deploy_loss as dl
    from test_stage05_train import CELLS3, HARDWARE

    data, _, bl, top, before = rows
    assert top in {c["sol_idx_global"] for g in before for c in g["candidates"]}
    lst = tmp_path / "l.tsv"
    lst.write_text("".join(f"C\t{e.name}\n" for e in bl.entries))
    snapshot = tmp_path / "snapshot.yaml"
    snapshot.write_text(
        yaml.safe_dump({"arch": "gfx950", "correctness": {"lists": [str(lst)]}})
    )
    train = tmp_path / "stage05"
    train.mkdir()
    args = {
        "config_yaml": str(snapshot),
        "enriched_csv_dir": [str(data)],
        "validation_frac": 0.5,
        "seed": 5,
        "repeat_aggregation": "median",
        "hardware_device": 0,
    }
    metrics = {
        "args": args,
        "catalog": "v2",
        "hardware": dict(HARDWARE),
        "per_cell_val_sel_eff": {c: 1.0 for c in CELLS3},
    }
    (train / "metrics.json").write_text(json.dumps(metrics))
    work = tmp_path / "work"
    ns = argparse.Namespace(
        stage05_dir=str(train), work_dir=str(work), hardware=None, pci_chip_id=None
    )
    dl.prepare(ns)
    prepared = pickle.loads((work / dl.PREPARED).read_bytes())
    seen = {
        c["sol_idx_global"]
        for gs in prepared["val"].values()
        for g in gs
        for c in g["candidates"]
    }
    assert seen and top not in seen
    assert prepared["load_stats"]["rows_blocked"] > 0


def test_stage05_names_the_blocklists_that_filtered_its_labels(tmp_path):
    import stage05_train as s5

    a = cx.Blocklist([cx.Entry("Cijk_a")])
    b = cx.Blocklist([cx.Entry("Cijk_a"), cx.Entry("Cijk_b")])
    assert cx.names_digest(a) != cx.names_digest(b)
    assert cx.names_digest(cx.Blocklist()) is None
    assert cx.names_digest(cx.Blocklist([cx.Entry("Cijk_a", "C", source="x")])) == (
        cx.names_digest(a)
    )
    folders = []
    for i, rec in enumerate((cx.applied_record(a), cx.applied_record(None), None)):
        d = tmp_path / f"r{i}"
        d.mkdir()
        doc = {"rows": 1} if rec is None else {"blocklist": rec}
        (d / "enrich_summary.json").write_text(json.dumps(doc))
        folders.append(d)
    got = s5.labels_blocklist(b, folders)
    assert got == {
        "digest": cx.names_digest(b),
        "kernels": 2,
        "enriched": [None, cx.names_digest(a)],
        "unknown": 1,
    }
    assert s5.labels_blocklist(None, [])["digest"] is None


def test_timings_and_bench_analysis_leave_a_blocked_kernel_out(rows):
    from lib import evaluate as ev
    from lib import run_analysis as ra

    data, ident, bl, top, before = rows
    plain = ev.load_timings([data], identity=ident)
    timings = ev.load_timings([data], identity=ident, blocklist=bl)
    assert set(timings) == set(plain)
    assert all(
        c["sol_idx_global"] != top for g in timings.values() for c in g["candidates"]
    )
    assert any(timings[k]["winner_us"] > plain[k]["winner_us"] for k in plain)
    assert ev.load_timings([data], blocklist=bl).keys() == plain.keys()
    for identity in (ident, None):
        bench = ra.load_bench([data], identity=identity, blocklist=bl)
        assert bench.stats["rows_blocked"] > 0 and top not in bench.configs
        assert all(top not in set(g.sids.tolist()) for g in bench.gemms)


def test_stage04_never_writes_a_row_of_a_blocked_kernel(tmp_path, monkeypatch):
    import stage04_convert_to_enriched_dataset as s04
    from bench_fakes import LIBRARY_STEM, kernel_name, write_logic
    from test_stage04_convert_to_enriched_dataset import (
        GA,
        MT,
        _config,
        _gemm,
        _library_solutions,
        _log_lines,
        _rows,
    )

    lib = tmp_path / "library"
    lib.mkdir()
    write_logic(lib / (LIBRARY_STEM + ".dat.zlib"), _library_solutions())
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "block_0000_lines1-6_gpu0.log").write_text("\n".join(_log_lines()) + "\n")
    lst = tmp_path / "l.tsv"
    lst.write_text(f"C\t{kernel_name(MT[11])}\n")
    cfg = _config(tmp_path / "c.yaml")
    doc = yaml.safe_load(cfg.read_text())
    doc["correctness"] = {"lists": [str(lst)]}
    cfg.write_text(yaml.safe_dump(doc))
    argv = ["stage04", "--log-dir", logs, "--out-dir", tmp_path / "out"]
    argv += ["--dat-dir", lib, "--library-stem", LIBRARY_STEM, "--quiet"]
    monkeypatch.setattr(sys, "argv", [str(a) for a in [*argv, "--config-yaml", cfg]])
    assert s04.main() == 0
    a = {int(r["sol_idx_global"]): r for r in _rows(tmp_path / "out")[_gemm(GA)]}
    assert 11 not in a and a[10]["is_winner"] == "True"
    summary = json.loads((tmp_path / "out" / "enrich_summary.json").read_text())
    assert summary["kernels_blocked"] == 1
    assert summary["blocklist"] == cx.applied_record(cx.read_list(lst))
    assert summary["counters"]["tested_rows_blocked"] == 1
    attrs = json.loads((tmp_path / "out" / "kernels.json").read_text())["kernels"]
    assert "11" not in attrs


def _deploy_cfg(path, lst):
    cfg = cf.config(lists=[str(lst)])
    cfg["correctness"].pop("enabled")
    cfg["deploy"] = {"weight_dtype": "int8"}
    path.write_text(yaml.safe_dump(cfg))
    return path


def _staged(out):
    import stage06_deploy_weights as s6

    return out / "staged_for_deploy" / s6.WEIGHTS_REL / "gfx950" / "gfx950"


def test_stage06_stages_the_sidecar_of_the_served_pool(tmp_path, make_bundle):
    import stage06_deploy_weights as s6

    lib = cf.write_library(tmp_path / "lib")
    lst = tmp_path / "l.tsv"
    lst.write_text(f"C\t{name(102)}\nA\t{name(200)}\nD\tCijk_of_another_library\n")
    cfg = _deploy_cfg(tmp_path / "cfg.yaml", lst)
    train = tmp_path / "round_1" / "stage05"
    make_bundle(train, ["Tiny|Tiny|TinyK|Bnone"])
    out = tmp_path / "out"
    rc = s6.deploy(
        train_dir=train, config_yaml=cfg, out_dir=out, quiet=True, library_dir=lib
    )
    assert rc == 0
    sidecar = _staged(out) / cx.sidecar_name(cf.STEM)
    assert cx.read_sidecar(sidecar) == sorted([name(102), name(200)])
    rec = json.loads((out / "deploy_record.json").read_text())
    assert rec["blocklist"]["kernels"] == 2
    assert rec["targets"][0]["blocklist"] == cx.sidecar_name(cf.STEM)
    # The record names the blocklist the deploy applied, the one that
    # filtered the bundle's labels (none recorded by this bundle) and each
    # target's sidecar.
    assert rec["blocklist"]["digest"] == cx.names_digest(cx.read_list(lst))
    assert rec["blocklist"]["trained_with"] is None
    digest = hashlib.sha256(sidecar.read_bytes()).hexdigest()
    assert rec["targets"][0]["blocklist_sha256"] == digest
    install = (out / "staged_for_deploy" / "INSTALL.md").read_text()
    assert f"Copy each `{cx.SIDECAR_SUFFIX}` file" in install
    assert "a bundle without one needs that file removed" in install
    out2 = tmp_path / "out2"
    assert s6.deploy(train_dir=train, config_yaml=cfg, out_dir=out2, quiet=True) == 0
    names = cx.read_sidecar(_staged(out2) / cx.sidecar_name(cf.STEM))
    assert "Cijk_of_another_library" in names and len(names) == 3
    # A bundle without a sidecar says which file a former deploy left.
    lst.write_text("# nothing blocked\n")
    out3 = tmp_path / "out3"
    assert s6.deploy(train_dir=train, config_yaml=cfg, out_dir=out3, quiet=True) == 0
    assert not (_staged(out3) / cx.sidecar_name(cf.STEM)).exists()
    install = (out3 / "staged_for_deploy" / "INSTALL.md").read_text()
    assert f"This bundle ships no `{cx.SIDECAR_SUFFIX}` file" in install
    assert cx.sidecar_name(cf.STEM) in install
    rec = json.loads((out3 / "deploy_record.json").read_text())
    assert rec["blocklist"]["digest"] is None
    assert rec["targets"][0]["blocklist_sha256"] is None


def test_stage06_warns_when_the_labels_had_another_blocklist(
    tmp_path, make_bundle, capsys
):
    import torch

    import stage06_deploy_weights as s6

    lib = cf.write_library(tmp_path / "lib")
    lst = tmp_path / "l.tsv"
    lst.write_text(f"C\t{name(102)}\n")
    cfg = _deploy_cfg(tmp_path / "cfg.yaml", lst)
    train = tmp_path / "round_1" / "stage05"
    make_bundle(train, ["Tiny|Tiny|TinyK|Bnone"])
    bundle = torch.load(train / "models.pt", weights_only=True)
    trained = cx.applied_record(cx.read_list(lst))
    torch.save(
        {**bundle, "blocklist": {**trained, "enriched": [], "unknown": 0}},
        train / "models.pt",
    )
    out = tmp_path / "out"
    assert (
        s6.deploy(train_dir=train, config_yaml=cfg, out_dir=out, library_dir=lib) == 0
    )
    assert "labels were filtered by another blocklist" not in capsys.readouterr().err
    rec = json.loads((out / "deploy_record.json").read_text())
    assert rec["blocklist"]["trained_with"]["digest"] == trained["digest"]
    lst.write_text(f"C\t{name(102)}\nA\t{name(200)}\n")
    assert (
        s6.deploy(train_dir=train, config_yaml=cfg, out_dir=out, library_dir=lib) == 0
    )
    assert "labels were filtered by another blocklist" in capsys.readouterr().err


def test_stage06_installs_and_removes_the_sidecar_with_the_model(
    tmp_path, make_bundle, monkeypatch
):
    import stage06_deploy_weights as s6
    from test_stage06_deploy_weights import COPY_BODY, fake_cmake, weights_dir

    pytest.importorskip("tilewright")
    lib = cf.write_library(tmp_path / "lib")
    repo = tmp_path / "repo"
    weights_dir(repo, "gfx950").mkdir(parents=True)
    build = tmp_path / "build"
    blib = build / "Tensile" / "library" / "gfx950"
    blib.mkdir(parents=True)
    src = repo / "shared" / "tilewright" / "weights" / "hipblaslt"
    fake_cmake(tmp_path, monkeypatch, COPY_BODY.format(src=str(src)))
    lst = tmp_path / "l.tsv"
    lst.write_text(f"C\t{name(102)}\n")
    cfg = _deploy_cfg(tmp_path / "cfg.yaml", lst)
    train = tmp_path / "round_1" / "stage05"
    make_bundle(train, ["Tiny|Tiny|TinyK|Bnone"])

    def deploy(out):
        return s6.deploy(
            train_dir=train,
            config_yaml=cfg,
            out_dir=tmp_path / out,
            quiet=True,
            library_dir=lib,
            repo=repo,
            apply_patch=True,
            build_after_apply=True,
            build_dir=build,
        )

    side = cx.sidecar_name(cf.STEM)
    assert deploy("out") == 0
    assert cx.read_sidecar(weights_dir(repo, "gfx950") / side) == [name(102)]
    assert cx.read_sidecar(blib / side) == [name(102)]
    rec = json.loads((tmp_path / "out" / "deploy_record.json").read_text())
    assert rec["colocation"] == "verified"

    lst.write_text("# nothing blocked\n")
    assert deploy("out2") == 0
    assert not (weights_dir(repo, "gfx950") / side).exists()
    assert not (blib / side).exists()

    lst.write_text(f"C\t{name(102)}\n")
    skipping = COPY_BODY.format(src=str(src)).replace(
        'if not f.name.startswith("."):',
        'if not f.name.startswith(".") and not f.name.endswith(".blocklist"):',
    )
    fake_cmake(tmp_path, monkeypatch, skipping)
    assert deploy("out3") == 1
    rec = json.loads((tmp_path / "out3" / "deploy_record.json").read_text())
    assert "does not copy" in rec["error"] and rec["deployed"] is False
    assert not (weights_dir(repo, "gfx950") / side).exists()


# ── the orchestrator ─────────────────────────────────────────────────────────


def test_resolve_correctness_names_the_invocations_blocklist(tmp_path, monkeypatch):
    import run_orchestrator as ro

    monkeypatch.chdir(tmp_path)
    run_dir = tmp_path / "run"
    cfg = {"correctness": {"enabled": True, "lists": ["gfx950_bbs_tn", "mine.tsv"]}}
    out = ro.resolve_correctness(cfg, run_dir, "inv1")["correctness"]
    assert out["lists"] == ["gfx950_bbs_tn", str(tmp_path / "mine.tsv")]
    assert out["cache_dir"] == str(run_dir / "correctness" / "cache")
    assert out["blocklist"] == str(run_dir / "correctness" / "blocklists" / "inv1.json")
    assert cfg["correctness"] == {
        "enabled": True,
        "lists": ["gfx950_bbs_tn", "mine.tsv"],
    }
    shared = ro.resolve_correctness(
        {"correctness": {"enabled": True, "cache_dir": "c"}}, run_dir, "inv2"
    )
    assert shared["correctness"]["cache_dir"] == str(tmp_path / "c")
    canary_off = {"enabled": False}
    off = ro.resolve_correctness(
        {"correctness": {"lists": ["a.tsv"]}, "hazards": canary_off}, run_dir, "i"
    )
    assert "blocklist" not in off["correctness"]
    plain = {"arch": "x", "hazards": canary_off}
    assert ro.resolve_correctness(plain, run_dir, "i") == plain
    canary = ro.resolve_correctness({"arch": "x"}, run_dir, "i")
    assert canary["correctness"] == {
        "blocklist": str(run_dir / "correctness" / "blocklists" / "i.json")
    }
    assert canary["hazards"] == {"cache_dir": str(run_dir / "hazards" / "cache")}
    gate_off = ro.resolve_correctness({"correctness": {"lists": []}}, run_dir, "i")
    assert gate_off["correctness"]["blocklist"] == canary["correctness"]["blocklist"]
    assert "cache_dir" not in gate_off["correctness"]
    stale = {"correctness": {"blocklist": "old.json"}}
    unneeded = ro.resolve_correctness(stale, run_dir, "i", canary=False)
    assert unneeded["correctness"] == {"lists": []}
    assert unneeded["hazards"]["cache_dir"] == canary["hazards"]["cache_dir"]


def test_the_preflight_reads_the_pool_before_the_gate_wrote_a_blocklist(tmp_path):
    import run_orchestrator as ro
    from bench_fakes import solution
    from test_run_orchestrator import UNLISTED, _unlisted_config

    queue = dict(tileProcessingStrategy="StreamK", workAssignment="DynamicWorkQueue")
    static = dict(tileProcessingStrategy="StreamK", workAssignment="StaticGrid")
    kernels = [solution(1, 1, **static), solution(3, 3, **queue)]
    cfg, paths = _unlisted_config(tmp_path, kernels)
    cfg["correctness"] = {"enabled": True}
    cfg = ro.resolve_correctness(cfg, tmp_path / "run", "inv")
    assert not Path(cfg["correctness"]["blocklist"]).exists()
    (err,) = ro._queue_count_errors(cfg, paths)
    assert err.startswith("1 of the 2 kernels of the pool (e.g. ")
    assert f"unknown for {UNLISTED}: set hardware.num_xcd" in err


def test_the_audit_set_benches_every_kernel_whatever_a_run_blocks(tmp_path):
    from lib import audit as al

    cfg = cf.config(lists=["gfx950_bbs_tn"])
    assert "correctness" not in al.bench_config(cfg)
    assert cfg["correctness"]["lists"] == ["gfx950_bbs_tn"]


def test_the_audit_set_checks_read_every_kernel_of_the_pool(tmp_path):
    import types

    import run_orchestrator as ro
    from lib import audit as al
    from lib import run_layout

    lib = cf.write_library(tmp_path / "lib")
    lst = tmp_path / "l.tsv"
    lst.write_text(f"A\t{name(101)}\n")
    cfg = ro.resolve_correctness(cf.config(lists=[str(lst)]), tmp_path / "run", "inv")
    assert not Path(cfg["correctness"]["blocklist"]).exists()
    paths = types.SimpleNamespace(library_dir=lib)
    digests = ro.config_pool_digests(cfg, paths)
    assert sorted(digests) == [100, 101, 102, 103, 200, 201]
    audit = tmp_path / "audit"
    run_layout.settle(audit)
    (audit / "stage02").mkdir(parents=True)
    (audit / "stage02" / ro.STAGE_MARKER).write_text("{}")
    planned = {str(s): d for s, d in digests.items()}
    m = {"schema": al.SCHEMA_VERSION, "generations": [], "planned_kernels": planned}
    (audit / al.MANIFEST).write_text(json.dumps(m))
    assert ro.audit_build_refusal(audit, ro.config_pool_digests(cfg, paths)) is None


def test_config_checks_of_the_correctness_section(tmp_path):
    import run_orchestrator as ro

    good = tmp_path / "l.tsv"
    good.write_text("A\tCijk_x\n")
    cfg, _ = ro.load_config(make_config(tmp_path, **{"correctness.lists": [str(good)]}))
    assert ro.validate_config(cfg)[0] == []
    cfg["correctness"] = {"lists": [str(tmp_path / "gone.tsv")], "enabled": True}
    cfg["correctness"]["harness"] = str(tmp_path / "no_harness")
    cfg["hipblaslt"]["a_type"] = "f8_r"
    errors = "\n".join(ro.validate_config(cfg)[0])
    assert "correctness.lists: cannot read the blocklist" in errors
    assert (
        "correctness.enabled: the exact harness does not run hipblaslt.a_type" in errors
    )
    assert "correctness.harness" in errors
    cfg["correctness"] = {"enabled": True}
    cfg["hipblaslt"].update(a_type="f4_r", b_type="f4_r", scaleA=3, scaleB=3)
    assert not [e for e in ro.validate_config(cfg)[0] if "correctness" in e]
    cfg["hipblaslt"].update(scaleA=1, scaleB=1)
    errors = "\n".join(ro.validate_config(cfg)[0])
    assert "correctness.enabled: the exact harness does not run scaled" in errors
    cfg["correctness"] = {"repetitions": 2}
    assert any("repetitions" in e for e in ro.validate_config(cfg)[0])


def test_a_gate_blocklist_without_the_gate_is_refused(tmp_path, stubs, env, capsys):
    """A snapshot of a gated run with the gate turned off would apply the
    blocklist an earlier invocation's gate wrote, or fail every stage once it
    is gone: the config check refuses it before anything runs."""
    import run_orchestrator as ro

    old = tmp_path / "blocklist.json"
    cx.write_blocklist(old, cx.Blocklist([cx.Entry(name="Cijk_x")]))
    cfg, _ = ro.load_config(make_config(tmp_path))
    cfg["hazards"] = {"enabled": False}
    cfg["correctness"] = {"enabled": False, "blocklist": str(old)}
    (err,) = ro.validate_config(cfg)[0]
    assert err.startswith("correctness.blocklist: ") and "correctness.lists" in err
    cfg["correctness"] = {"enabled": False, "lists": [str(old)]}
    assert ro.validate_config(cfg)[0] == []
    gated = ro.resolve_correctness(
        {**cfg, "correctness": {"enabled": True, "blocklist": str(old)}},
        tmp_path / "run",
        "inv",
    )
    assert gated["correctness"]["blocklist"] != str(old)
    assert ro.validate_config(gated)[0] == []
    doc = yaml.safe_load(make_config(tmp_path, **{"full.total_rounds": 1}).read_text())
    doc["correctness"] = {"blocklist": str(old)}
    doc["hazards"] = {"enabled": False}
    path = tmp_path / "snapshot_without_gate.yaml"
    path.write_text(yaml.safe_dump(doc))
    assert run(stubs, path, tmp_path / "run2", "--mode", "initial") == 2
    assert "correctness.blocklist" in capsys.readouterr().err
    assert calls(env) == []
    # With the hazard canary on as well, a gate's blocklist stays refused.
    del doc["hazards"]
    path.write_text(yaml.safe_dump(doc))
    assert run(stubs, path, tmp_path / "run3", "--mode", "initial") == 2
    assert "correctness.blocklist" in capsys.readouterr().err
    assert calls(env) == []
    # --mode audit benches without a correctness section: nothing applies it.
    del cfg["hazards"]
    cfg["correctness"] = {"enabled": False, "blocklist": str(old)}
    assert ro.validate_config(cfg)[0] and not ro.validate_config(cfg, "audit")[0]


def test_the_canary_blocklist_of_a_snapshot_without_the_gate_is_replaced(
    tmp_path, stubs, env
):
    """The snapshot of an invocation whose canary ran without the gate names
    the blocklist the orchestrator wrote of the canary's entries alone: the
    next invocation names its own, or none when its pool needs no canary,
    and the canary's store keeps what the old one said."""
    import run_orchestrator as ro
    from lib import hazards as hz

    canary = tmp_path / "canary.json"
    entry = cx.Entry(name="Cijk_hang", reason="hazard canary: hang", source=hz.SOURCE)
    cx.write_blocklist(canary, cx.Blocklist([entry]), source=hz.SOURCE)
    doc = yaml.safe_load(make_config(tmp_path, **{"full.total_rounds": 1}).read_text())
    doc["correctness"] = {"blocklist": str(canary)}
    assert ro.validate_config(doc)[0] == []
    assert ro.validate_config(dict(doc, hazards={"enabled": False}))[0]
    mixed = tmp_path / "mixed.json"
    cx.write_blocklist(mixed, cx.Blocklist([entry, cx.Entry(name="Cijk_wrong")]))
    assert ro.validate_config(dict(doc, correctness={"blocklist": str(mixed)}))[0]
    gone = dict(doc, correctness={"blocklist": str(tmp_path / "gone.json")})
    assert ro.validate_config(gone)[0]
    path = tmp_path / "snapshot_of_the_canary.yaml"
    path.write_text(yaml.safe_dump(doc))
    assert run(stubs, path, tmp_path / "run", "--mode", "initial") == 0
    snapshot = Path(arg(calls(env)[0], "--config-yaml"))
    assert "blocklist" not in yaml.safe_load(snapshot.read_text())["correctness"]


def _gate_config(tmp_path, **more):
    harness = tmp_path / "errmap"
    harness.write_text("")
    lst = tmp_path / "l.tsv"
    lst.write_text("A\tCijk_x\n")
    over = {
        "correctness.enabled": True,
        "correctness.lists": [str(lst)],
        "correctness.harness": str(harness),
        "correctness.devices": [1],
        "full.total_rounds": 1,
        **more,
    }
    return make_config(tmp_path, **over)


def _snapshots(run_dir):
    return sorted(
        p
        for p in (run_dir / "configs").glob("*.yaml")
        if not p.name.endswith(".source.yaml")
    )


def test_the_orchestrator_runs_the_gate_first_and_every_stage_reads_its_blocklist(
    tmp_path, stubs, env, monkeypatch
):
    import run_orchestrator as ro

    monkeypatch.setenv("STUB_BLOCKED", json.dumps(["Cijk_y"]))
    run_dir = tmp_path / "run"
    cfg = _gate_config(tmp_path)
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    seen = calls(env)
    assert seen[0]["stage"] == "stage00"
    gate = seen[0]
    assert arg(gate, "--out-dir") == str(run_dir / "correctness" / "gate.new")
    assert arg(gate, "--cache-dir") == str(run_dir / "correctness" / "cache")
    assert arg(gate, "--devices") == "1" and arg(gate, "--hardware-device") == "0"
    assert arg(gate, "--dat-dir") == str(
        ro.derive_paths(
            yaml.safe_load(Path(arg(gate, "--config-yaml")).read_text())
        ).library_dir
    )
    assert gate["leases"] == {}
    snapshot = Path(arg(gate, "--config-yaml"))
    resolved = yaml.safe_load(snapshot.read_text())
    first = Path(resolved["correctness"]["blocklist"])
    assert first.parent == run_dir / "correctness" / "blocklists"
    assert cx.read_list(first).names == {"Cijk_y"}
    assert cx.config_blocklist(resolved).names == {"Cijk_x", "Cijk_y"}
    assert all(
        arg(c, "--config-yaml") == str(snapshot)
        for c in seen
        if "--config-yaml" in c["argv"]
    )
    gate_dir = run_dir / "correctness" / "gate"
    assert (gate_dir / ".stage_complete.json").is_file()
    assert not (run_dir / "correctness" / "gate.new").exists()
    marker = json.loads((run_dir / "round_0" / "stage01" / ro.STAGE_MARKER).read_text())
    assert marker["blocklist"] == cx.blocklist_record(first)
    assert marker["blocklist"]["kernels"] == 1

    # The next invocation applies what its own gate wrote; the first one's
    # snapshot still names the blocklist its stages applied.
    monkeypatch.setenv("STUB_BLOCKED", json.dumps(["Cijk_y", "Cijk_z"]))
    run(stubs, cfg, run_dir, "--mode", "validate")
    gates = [c for c in calls(env) if c["stage"] == "stage00"]
    assert len(gates) == 2
    newest = yaml.safe_load(Path(arg(gates[1], "--config-yaml")).read_text())
    second = Path(newest["correctness"]["blocklist"])
    assert second != first and cx.read_list(second).names == {"Cijk_y", "Cijk_z"}
    assert cx.config_blocklist(resolved).names == {"Cijk_x", "Cijk_y"}
    assert (run_dir / "correctness" / "stale").is_dir()


def test_a_failed_gate_leaves_the_last_blocklist_in_place(
    tmp_path, stubs, env, monkeypatch, capsys
):
    run_dir = tmp_path / "run"
    cfg = _gate_config(tmp_path)
    monkeypatch.setenv("STUB_BLOCKED", json.dumps(["Cijk_y"]))
    assert run(stubs, cfg, run_dir, "--mode", "initial") == 0
    capsys.readouterr()
    monkeypatch.setenv("STUB_FAIL", "stage00")
    assert run(stubs, cfg, run_dir, "--mode", "validate") == 1
    gate = run_dir / "correctness" / "gate"
    assert cx.read_list(gate / cx.BLOCKLIST_FILE).names == {"Cijk_y"}
    assert (run_dir / "correctness" / "gate.new" / "partial.txt").is_file()
    newest = yaml.safe_load(_snapshots(run_dir)[-1].read_text())
    assert newest["config_id"] and cx.config_blocklist(newest).names == {
        "Cijk_x",
        "Cijk_y",
    }
    assert "holds the blocklist of the last gate that succeeded" in (
        capsys.readouterr().err
    )
    # A gate that fails again does not lose it either, and one that runs
    # again replaces it.
    assert run(stubs, cfg, run_dir, "--mode", "validate") == 1
    assert cx.read_list(gate / cx.BLOCKLIST_FILE).names == {"Cijk_y"}
    monkeypatch.delenv("STUB_FAIL")
    monkeypatch.setenv("STUB_BLOCKED", "[]")
    run(stubs, cfg, run_dir, "--mode", "validate")
    assert cx.read_list(gate / cx.BLOCKLIST_FILE).names == set()


def test_the_run_pins_the_contents_of_its_lists(tmp_path, stubs, env, capsys):
    import run_orchestrator as ro
    from lib import pinning

    lst = tmp_path / "l.tsv"
    lst.write_text("A\tCijk_x\n")
    cfg_path = make_config(
        tmp_path,
        **{"correctness.lists": [str(lst), "gfx950_bbs_tn"], "full.total_rounds": 1},
    )
    cfg, _ = ro.load_config(cfg_path)
    before = ro.pinned_config(cfg)
    assert set(before["correctness.lists_sha256"]) == {str(lst), "gfx950_bbs_tn"}
    assert "correctness.lists_sha256" not in ro.pinned_config(cfg, audit=True)
    run_dir = tmp_path / "run"
    assert run(stubs, cfg_path, run_dir, "--mode", "initial") == 0
    lst.write_text("A\tCijk_x\nB\tCijk_w\n")
    changes = pinning.differences(
        {"config": pinning.config_record(before)},
        {"config": pinning.config_record(ro.pinned_config(cfg))},
    )
    assert [c.split(":")[0] for c in changes] == ["config correctness.lists_sha256"]
    capsys.readouterr()
    assert run(stubs, cfg_path, run_dir, "--mode", "initial") == 2
    assert "correctness.lists_sha256" in capsys.readouterr().err
    run(stubs, cfg_path, run_dir, "--mode", "initial", "--allow-code-change")
    assert "accepted (--allow-code-change): config correctness.lists_sha256" in (
        capsys.readouterr().err
    )


def test_stage07_serves_the_staged_sidecar_from_its_library_directory(
    tmp_path, stubs, env, monkeypatch
):
    import shutil

    from test_run_orchestrator import STEM as OSTEM

    release = tmp_path / "bench" / "projects" / "hipblaslt" / "build" / "release"
    lib = release / "Tensile" / "library" / "gfx950"
    lib.mkdir(parents=True)
    (lib / f"{OSTEM}.dat").write_bytes(b"logic")
    (lib / "tilewright_index").write_text(f"{OSTEM}\tshipped.bin\n")
    (lib / cx.sidecar_name(OSTEM)).write_text("Cijk_of_an_earlier_deploy\n")
    v3 = tmp_path / "v3build"
    cfg = _gate_config(
        tmp_path,
        **{
            "full.total_rounds": 2,
            "stage07.build_dir": str(v3),
            "deploy.rocm_libraries_to_deploy": None,
            "deploy.apply_patch": False,
            "deploy.build_after_apply": False,
        },
    )
    monkeypatch.setenv("STUB_STAGED", "gfx950")
    monkeypatch.setenv("STUB_BLOCKED", json.dumps(["Cijk_y"]))
    monkeypatch.setenv("STUB_SIDECAR", cx.sidecar_text(OSTEM, ["Cijk_x", "Cijk_y"]))
    run_dir = tmp_path / "run"
    assert run(stubs, cfg, run_dir, "--mode", "full") == 0
    libdir = run_dir / "validate" / "stage07_lib" / "gfx950"
    side = libdir / cx.sidecar_name(OSTEM)
    assert not side.is_symlink() and cx.read_sidecar(side) == ["Cijk_x", "Cijk_y"]
    rec = json.loads(
        (run_dir / "round_1" / "stage06" / "deploy_record.json").read_text()
    )
    staged = Path(rec["targets"][0]["staged"]).parent / rec["targets"][0]["blocklist"]
    doc = json.loads((libdir / ".serving_libdir.json").read_text())
    assert doc["blocklist"] == str(staged) and doc["blocklist_sha256"]

    shutil.rmtree(run_dir / "validate")
    assert run(stubs, cfg, run_dir, "--mode", "validate") == 0
    assert cx.read_sidecar(side) == ["Cijk_x", "Cijk_y"]
    (s07,) = [c for c in calls(env) if c["stage"] == "stage07"][-1:]
    assert arg(s07, "--library-dir") == str(libdir)


def test_a_bundle_serves_with_the_blocklist_of_its_own_deploy(tmp_path):
    """A bundle's blocklist: the sidecar its stage06 staged (none staged:
    none), else the blocklist its stage05 applied, through the config
    snapshot its metrics.json names; unknown when neither can be read."""
    bundle = tmp_path / "round_2"
    train, s06 = bundle / "stage05", bundle / "stage06"
    train.mkdir(parents=True)
    assert cx.bundle_blocklist(train, cf.STEM)[0] is None
    lst = tmp_path / "l.tsv"
    lst.write_text("A\tCijk_a\n")
    snapshot = tmp_path / "snapshot.yaml"
    snapshot.write_text(yaml.safe_dump({"correctness": {"lists": [str(lst)]}}))
    metrics = {"args": {"config_yaml": str(snapshot)}}
    (train / "metrics.json").write_text(json.dumps(metrics))
    found, where = cx.bundle_blocklist(train, cf.STEM)
    assert found.names == {"Cijk_a"} and str(snapshot) in where
    snapshot.write_text(yaml.safe_dump({"arch": "gfx950"}))
    found, _ = cx.bundle_blocklist(train, cf.STEM)
    assert found is not None and not found
    staged = s06 / "staged_for_deploy"
    staged.mkdir(parents=True)
    (staged / "m.bin").write_bytes(b"model")
    (staged / cx.sidecar_name(cf.STEM)).write_text(cx.sidecar_text(cf.STEM, ["Cijk_b"]))
    target = {"library_stem": cf.STEM, "staged": str(staged / "m.bin")}
    record = {"train_dir": str(train), "blocklist": {"kernels": 1}}
    record["targets"] = [dict(target, blocklist=cx.sidecar_name(cf.STEM))]
    (s06 / "deploy_record.json").write_text(json.dumps(record))
    found, where = cx.bundle_blocklist(train, cf.STEM)
    assert found.names == {"Cijk_b"} and "deploy_record.json" in where
    record["targets"] = [dict(target, blocklist=None)]
    (s06 / "deploy_record.json").write_text(json.dumps(record))
    found, _ = cx.bundle_blocklist(train, cf.STEM)
    assert found is not None and not found
    # A record from before sidecars says nothing: the snapshot does.
    (s06 / "deploy_record.json").write_text(json.dumps({"targets": [target]}))
    assert "snapshot.yaml" in cx.bundle_blocklist(train, cf.STEM)[1]
    record["targets"] = [dict(target, blocklist="gone.blocklist")]
    (s06 / "deploy_record.json").write_text(json.dumps(record))
    found, why = cx.bundle_blocklist(train, cf.STEM)
    assert found is None and "gone.blocklist" in why


def test_stage08_evaluates_a_bundle_with_its_own_blocklist(tmp_path, capsys):
    import argparse

    import stage08_optional_validate_selection_efficiency as s08

    train = tmp_path / "round_2" / "stage05"
    train.mkdir(parents=True)
    own = tmp_path / "own.tsv"
    own.write_text("A\tCijk_a\n")
    snapshot = tmp_path / "snapshot.yaml"
    snapshot.write_text(yaml.safe_dump({"correctness": {"lists": [str(own)]}}))
    (train / "metrics.json").write_text(
        json.dumps({"args": {"config_yaml": str(snapshot)}})
    )
    now = tmp_path / "now.tsv"
    now.write_text("A\tCijk_a\nB\tCijk_b\n")
    cfg = {"correctness": {"lists": [str(now)]}}
    args = argparse.Namespace(train_dir=train, library_stem=cf.STEM, quiet=False)
    assert s08.bundle_blocklist_of(args, cfg).names == {"Cijk_a"}
    assert "the bundle serves with its own blocklist" in capsys.readouterr().err
    (train / "metrics.json").unlink()
    assert s08.bundle_blocklist_of(args, cfg).names == {"Cijk_a", "Cijk_b"}
    assert "the bundle's blocklist is unknown" in capsys.readouterr().err


def test_a_serving_libdir_never_takes_the_librarys_own_sidecar(tmp_path):
    from lib import pool_libdir as pl

    src = tmp_path / "src"
    src.mkdir()
    stem = "TensileLibrary_BB_BB_Contraction_l_Alik_Bljk_ID75a0_gfx950"
    other = "TensileLibrary_HH_HH_Contraction_l_Alik_Bljk_ID75a0_gfx950"
    for s_ in (stem, other):
        (src / f"{s_}.dat").write_bytes(b"logic")
        (src / cx.sidecar_name(s_)).write_text(f"Cijk_old_{s_}\n")
    (src / "tilewright_index").write_text(f"{stem}\told.bin\n{other}\to.bin\n")
    model = tmp_path / "model.bin"
    model.write_bytes(b"MLREC_v3")
    dst = tmp_path / "dst"
    summary = pl.build_serving_libdir(src, dst, stem, "new.bin", model)
    assert not (dst / cx.sidecar_name(stem)).exists()
    assert (dst / cx.sidecar_name(other)).is_symlink()
    assert summary["blocklist"] is None
    side = tmp_path / "staged.blocklist"
    side.write_text(cx.sidecar_text(stem, ["Cijk_a"]))
    summary = pl.build_serving_libdir(src, dst, stem, "new.bin", model, blocklist=side)
    assert cx.read_sidecar(dst / cx.sidecar_name(stem)) == ["Cijk_a"]
    assert summary["blocklist"] == str(side)


def test_stage07_needs_a_tensilelite_that_reads_blocklists(tmp_path):
    import fake_hipblaslt as fh
    import run_orchestrator as ro
    from test_run_orchestrator import STEM as OSTEM

    release = tmp_path / "bench" / "projects" / "hipblaslt" / "build" / "release"
    fh.write_fake_build(release, "gfx950", OSTEM)
    v3 = tmp_path / "v3"
    fh.write_fake_build(v3, "gfx950", OSTEM, capabilities=("MLREC_v3",))
    cfg, _ = ro.load_config(
        make_config(tmp_path, **{"stage07.build_dir": str(v3), "train.catalog": "v3"})
    )
    paths = ro.derive_paths(cfg)
    assert ro.stage07_build_errors(cfg, paths) == []
    pool = bp.config_pool(cfg, paths.library_dir, OSTEM, apply_blocklist=False)
    lst = tmp_path / "l.tsv"
    lst.write_text("A\tCijk_not_in_the_pool\n")
    cfg["correctness"] = {"lists": [str(lst)]}
    assert ro.stage07_build_errors(cfg, paths) == []
    lst.write_text(f"A\t{pool.kernels[0]['kernel_name']}\n")
    (err,) = ro.stage07_build_errors(cfg, paths)
    assert "does not read the blocklist sidecar" in err
    assert f"no {cx.SIDECAR_SUFFIX!r}" in err
    gated = ro.resolve_correctness(
        {**cfg, "correctness": {"enabled": True}}, tmp_path / "run", "inv"
    )
    (err,) = ro.stage07_build_errors(gated, paths)
    assert "does not read the blocklist sidecar" in err
    fh.write_fake_build(
        v3, "gfx950", OSTEM, capabilities=("MLREC_v3", cx.SIDECAR_SUFFIX)
    )
    assert ro.stage07_build_errors(gated, paths) == []
    assert ro.stage07_build_errors(cfg, paths) == []


def test_an_unreadable_blocklist_is_a_note_of_the_analysis(tmp_path):
    from lib import run_analysis as ra

    info = ra.ModelInfo(labels=[], whitelists={})
    gone = str(tmp_path / "gone.json")
    info.library_cfg = {"arch": "gfx950", "correctness": {"blocklist": gone}}
    assert ra.model_blocklist(info) is None
    assert "blocklist cannot be read" in info.notes[0]
    assert info.library_cfg == {"arch": "gfx950"}
    lst = tmp_path / "l.tsv"
    lst.write_text("Cijk_a\n")
    info.library_cfg["correctness"] = {"lists": [str(lst)]}
    assert ra.model_blocklist(info).names == {"Cijk_a"} and len(info.notes) == 1


def test_without_the_gate_no_stage00_runs(tmp_path, stubs, env):
    lst = tmp_path / "l.tsv"
    lst.write_text("A\tCijk_x\n")
    cfg = make_config(
        tmp_path, **{"correctness.lists": [str(lst)], "full.total_rounds": 1}
    )
    assert run(stubs, cfg, tmp_path / "run", "--mode", "initial") == 0
    assert "stage00" not in [c["stage"] for c in calls(env)]


def test_the_v2_winner_is_the_fastest_median_of_the_pool():
    from lib import audit as al
    from lib import audit_v2 as v2

    mm = v2.Measurement(0, [1, 2, 3], {1: 4.0, 2: 5.0, 3: 6.0}, False, False)
    assert al.pool_medians(mm, {1, 2, 3}) is mm
    kept = al.pool_medians(mm, {2, 3})
    assert kept.winner_us == 5.0 and mm.winner_us == 4.0
    assert al.pool_medians(None, {1}) is None
    assert al.pool_medians(mm, set()).winner_us is None
