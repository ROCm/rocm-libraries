# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import hashlib
import json
import math
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
            cfg = cfg or self.cfg()
            spec = cg.Spec(
                cfg=cfg,
                dat_dir=self.lib,
                library_stem=cfg["hipblaslt"]["library_stem"],
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


# (name, key) as TensileLite's
# PredictionLibraryTest.TilewrightBlocklistKeysDropTheIaAndPkaTokens pins them.
BLOCKLIST_KEYS = [
    ("Cijk_Ailk_MT64x64x32_IA0_K1_PKA1_SU0", "Cijk_Ailk_MT64x64x32_K1_SU0"),
    ("Cijk_Ailk_MT64x64x32_K1_PKA0_SU0", "Cijk_Ailk_MT64x64x32_K1_SU0"),
    ("IA2_Cijk_IA10_PKAn1", "Cijk"),
    (
        "Cijk_SIA3_IA_PKA_IAx1_PKA1x_MIA0_IAn_ia0",
        "Cijk_SIA3_IA_PKA_IAx1_PKA1x_MIA0_IAn_ia0",
    ),
    ("Cijk__IA0_", "Cijk__"),
    ("PKA1", ""),
    ("", ""),
    (
        "Custom_Cijk_Ailk_Bljk_S_SK3_UserArgs_MT256x256x32_shortname0",
        "Custom_Cijk_Ailk_Bljk_S_SK3_UserArgs_MT256x256x32_shortname0",
    ),
]
TENSILELITE = Path(__file__).resolve().parents[4] / "projects/hipblaslt/tensilelite"


@pytest.mark.parametrize("name, key", BLOCKLIST_KEYS)
def test_blocklist_keys_drop_the_ia_and_pka_tokens(name, key):
    assert cx.blocklist_key(name) == key


@pytest.mark.skipif(
    not (TENSILELITE / "include").is_dir(), reason="no TensileLite tilewright adapter"
)
def test_tensilelite_reads_sidecars_by_the_same_keys():
    hpp = (TENSILELITE / "include/Tensile/TilewrightRanker.hpp").read_text()
    tokens = re.search(r"BlocklistIgnoredTokens\{([^}]*)\}", hpp)
    assert tokens is not None
    assert tuple(re.findall(r'"(\w+)"', tokens.group(1))) == cx.IGNORED_NAME_TOKENS
    tests = (TENSILELITE / "tests/PredictionLibrary_test.cpp").read_text()
    body = re.search(
        r"TilewrightBlocklistKeysDropTheIaAndPkaTokens\)\n\{(.*?)\n\}", tests, re.S
    )
    assert body is not None
    pinned = re.findall(r'\{"([^"]*)",\s*"([^"]*)"\}', body.group(1))
    assert pinned == BLOCKLIST_KEYS


def test_a_name_blocks_the_kernels_whose_ia_and_pka_tokens_differ(tmp_path):
    """A sidecar written from a build that names kernels with PKA0 and no IA
    token blocks them, by name, in a build that names them with IA0 and
    PKA1, as TensileLite reads it; a kernel that differs from a listed one in
    those tokens alone is blocked with it, while other tokens still tell
    kernels apart. A kernel id hashes the name with its tokens, so it
    matches only the build that wrote the name."""
    side = tmp_path / ("Stem" + cx.SIDECAR_SUFFIX)
    side.write_text(cx.sidecar_text("Stem", ["Cijk_a_K1_PKA0_SU0", "Custom_b"]))
    bl = cx.sidecar_blocklist(side)
    assert bl.names == {"Cijk_a_K1_PKA0_SU0", "Custom_b"}
    names = [
        "Cijk_a_IA0_K1_PKA1_SU0",
        "Cijk_a_IA0_K1_PKA0_SU0",
        "Cijk_a_IA0_K1_PKA1_SU1",
        "Custom_b",
        "Cijk_c_IA0_K1_PKA1_SU0",
    ]
    kept = bl.kept([{"kernel_name": n} for n in names])
    assert [k["kernel_name"] for k in kept] == [names[2], names[4]]
    assert bl.blocks_name("Cijk_a_IA2_K1_PKAn1_SU0")
    assert not bl.blocks_name("Cijk_a_K1_SU1") and not bl.blocks_name("Custom")
    kid = f"MT64x64x32_MI16x16x8_NTA0_NTB0:{ki.name_hash(names[0])}"
    assert not bl.blocks_id(kid)
    assert cx.Blocklist([cx.Entry(names[0])]).blocks_id(kid)
    # A tensor-data-mover alias of a listed kernel, spelled by another build.
    tdm = cx.Blocklist([cx.Entry("Cijk_d_TDMI1_THA3_K1_PKA0")])
    assert tdm.blocks_name("Cijk_d_TDMI1_THA0_IA0_K1_PKA1")
    assert not tdm.blocks_name("Cijk_d_TDMI2_THA0_IA0_K1_PKA1")


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
        ({"scaleA": 2, "scaleB": 2}, "scaled"),
        ({"scaleA": 1, "scaleB": 3}, "scaled"),
        ({"scaleB": 0}, "scaled"),
        ({"scaleA": 0, "scaleB": 0}, "a_type"),
        ({"a_type": "bf16_r", "b_type": "bf16_r"}, "a_type"),
        ({"a_type": "f6_r", "b_type": "f6_r"}, "a_type"),
        ({"c_type": "f8_r", "d_type": "f8_r"}, "c_type"),
        ({"scaleA": 1, "scaleB": 1, "a_type": "f4_r"}, "a_type 'f4_r' with one f32"),
        ({"scaleA": 1, "scaleB": 1, "b_type": "bf16_r"}, "b_type 'bf16_r' with one"),
        ({"scaleA": 1, "scaleB": 0}, "scaled"),
        ({"scaleA": 1, "scaleB": 1, "d_type": "f32_r"}, "equal C and D"),
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


def test_the_block_scaled_gate_is_byte_identical():
    """The protocols, every harness command and the summary's shapes of
    block-scaled problems are those of the gate before it ran one f32 scale
    per tensor (the digests were taken from it)."""
    custom = cg.Settings(repetitions=4, trigger_repetitions=12, first_launch="all")
    golden = {
        (8, 8): ("631ba23aceb1316d", "317d8028e4b177cb", "cf69fdab024a8b93"),
        (4, 4): ("bcbaa772e3b708bb", "03d24900daf956e1", "73d5f89ad318bb8d"),
        (8, 4): ("15985e175f8dff15", "18186500ce481829", "149bdd3e72b4a9ac"),
    }
    for bits, (proto, proto_custom, records) in golden.items():
        shapes = cg.mx_shapes(*bits)
        assert cg.protocol(cg.Settings(), "harness-id", shapes) == proto
        assert cg.protocol(custom, "x", shapes) == proto_custom
        summary = [dict(s.record(), text=s.text()) for s in shapes]
        assert _digest(summary, sort_keys=True) == records
    cmds = []
    kernels = {7: {"mt_k": 128}, 8: {"mt_k": 512}}
    for layout in ("TN", "NN", "NT", "TT"):
        for a, b in (
            ("f8_r", "f8_r"),
            ("bf8_r", "bf8_r"),
            ("f4_r", "f4_r"),
            ("f8_r", "f4_r"),
        ):
            cfg = cf.mx_config(a, b)
            cfg["hipblaslt"].update(transA=layout[0], transB=layout[1])
            p = cg.harness_problem(cfg)
            shapes = cg.shapes_of(p)
            for s in shapes:
                cmds.append(cg.Task(s.name, s, [1, 2], 3).command(Path("h"), p))
            unsupported = {s.name: {7, 8} for s in shapes}
            for t in cg.roundup_tasks(unsupported, kernels, cg.Settings(), shapes):
                cmds.append(t.command(Path("h"), p))
            for t in cg.main_tasks([1, 2, 3, 4, 5], cg.Settings(), 2, shapes):
                cmds.append([t.tag, t.reps] + t.command(Path("h"), p))
            for t in cg.first_launch_tasks(kernels, [7, 8], cg.Settings()):
                cmds.append(t.command(Path("h"), p))
    assert len(cmds) == 1796 and _digest(cmds) == "0fd8d20728102fb9"


@pytest.mark.parametrize(
    "a_type, b_type", [("f8_r", "f8_r"), ("bf8_r", "bf8_r"), ("f8_r", "bf8_r")]
)
def test_8_bit_inputs_with_one_f32_scale_per_tensor_run_their_own_shapes(
    a_type, b_type
):
    cfg = cf.scalar_config(a_type, b_type)
    problem = cg.harness_problem(cfg)
    args = problem.args()
    assert args[args.index("--a-type") + 1] == cg.HARNESS_SCALAR_TYPES[a_type]
    assert args[args.index("--b-type") + 1] == cg.HARNESS_SCALAR_TYPES[b_type]
    assert args[args.index("--scale-a") + 1] == "1"
    assert args[args.index("--scale-b") + 1] == "1"
    assert problem.scalar_scales and problem.mx_bits == (0, 0)
    assert problem.stored_k == 1, "K stored as it is"
    assert cg.shapes_of(problem) == cg.SCALAR_SHAPES
    assert not cg.harness_problem(cf.mx_config()).scalar_scales
    assert not cg.harness_problem(cf.config()).scalar_scales


def test_scalar_shapes_keep_the_k_and_alignment_rules_of_the_bf16_shapes():
    """F8BS kernels take any K, M, N and leading dimension (no predicate of
    their libraries asks for a multiple or an alignment), as bf16 kernels do:
    every problem of SHAPES runs as it is, but the operands over 4 GiB, which
    take the least odd K that reaches 4 GiB at one byte per element (alone
    and as many batches); K of one iteration of each DepthU from 64 to 1024,
    and of 512 and 1024 also with an odd tail on the same M and N; the bias
    epilogue runs beside its twin without it."""
    shapes = cg.SCALAR_SHAPES
    by = {s.name: s for s in shapes}
    assert len(by) == len(shapes)
    over = {"a_over_4gib", "b_over_4gib"}
    for s in cg.SHAPES:
        if s.name in over:
            continue
        assert by[s.name] == s, s.name
    for name in ("a_over_4gib", "a_over_4gib_batched"):
        s = by[name]
        assert s.m * s.k * s.batch >= 1 << 32 > s.m * (s.k - 2) * s.batch, name
    for name in ("b_over_4gib", "b_over_4gib_batched"):
        s = by[name]
        assert s.n * s.k * s.batch >= 1 << 32 > s.n * (s.k - 2) * s.batch, name
    assert by["a_over_4gib_batched"].batch > 1 and by["b_over_4gib_batched"].batch > 1
    one_iteration = {f"k_{du}": du for du in (64, 128, 256, 512, 1024)}
    about_k = {"small_m_long_n_k_aligned", "long_k", *one_iteration}
    assert all(s.k % 64 for s in shapes if s.name not in about_k)
    assert {n: by[n].k for n in one_iteration} == one_iteration
    for du in (512, 1024):
        # One iteration, then a tail of 7 for every DepthU up to this one.
        tail = by[f"k_{du}_tail"]
        assert tail == replace(by[f"k_{du}"], name=tail.name, k=du + 7)
        assert all(tail.k % d == 7 for d in one_iteration.values() if d <= du)
    assert by["alpha_beta_padded"].pad_a % 2 and by["alpha_beta_padded"].pad_b % 2
    bias = [s for s in shapes if s.bias]
    assert {s.bias_type for s in bias} == {"", "f32"}
    assert any(s.trigger and s.batch > 1 for s in bias)
    assert any(s.separate_d and s.beta for s in bias)
    assert any(s.name.startswith("a_over_4gib") for s in bias)
    for s in bias:
        if s.name.endswith("_bias") or s.name.endswith("_bias_f32"):
            twin = s.name[: s.name.index("_bias")]
            assert replace(by[twin], name=s.name, bias=True, bias_type=s.bias_type) == s
    with pytest.raises(ValueError, match="bias type without a bias"):
        cg.Shape("x", 1, 1, 1, bias_type="f32")


@pytest.mark.parametrize("layout", ["TN", "NN", "NT", "TT"])
@pytest.mark.parametrize("a_type", ["f8_r", "bf8_r"])
def test_the_harness_takes_every_scalar_scaled_task_in_every_layout(layout, a_type):
    """Every command of the gate's passes of a scalar-scaled problem is one
    the harness takes, gives both operands their scale, stores K as it is
    and names the bias type only where the shape asks for f32."""
    cfg = cf.scalar_config(a_type)
    cfg["hipblaslt"].update(transA=layout[0], transB=layout[1])
    problem = cg.harness_problem(cfg)
    shapes = cg.shapes_of(problem)
    kernels = {7: {"mt_k": 1024}, 8: {"mt_k": 128}}
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
        assert opts["--scale-a"] == opts["--scale-b"] == "1"
        assert ("--bias" in opts) == t.shape.bias
        assert opts.get("--bias-type", "") == t.shape.bias_type
        if "--lda" in opts:
            rows = k if layout[0] == "T" else m
            assert int(opts["--lda"]) == rows + t.shape.pad_a


@pytest.mark.parametrize(
    "a_type, scale_a, b_type, scale_b, error",
    [
        ("f8", "1", "f8", "1", ""),
        ("bf8", "1", "f8", "1", ""),
        ("f8", "3", "f8", "3", ""),
        ("bf16", "1", "bf16", "1", "8-bit"),
        ("f4", "1", "f4", "1", "8-bit"),
        ("f8", "0", "f8", "0", "8-bit"),
        ("f8", "1", "bf16", "0", ""),
        ("f8", "1", "f8", "3", "one format"),
        ("f8", "2", "f8", "2", "scaling format 2"),
    ],
)
def test_the_fake_harness_refuses_what_the_harness_refuses(
    a_type, scale_a, b_type, scale_b, error
):
    opts = {"--a-type": a_type, "--scale-a": scale_a}
    opts.update({"--b-type": b_type, "--scale-b": scale_b})
    found = cf._type_error(384, opts)
    assert (error in found and found) if error else not found
    text = cg.HARNESS_SOURCE.read_text()
    assert "one f32 scale only when it" in text and "A and B take scales of one" in text


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


def _shape_of(args):
    return tuple(args[:4])


def test_the_gate_runs_an_f8bs_config_end_to_end(world, tmp_path):
    """gfx950's F8BS problem type (f8 A and B with one f32 scale per tensor,
    bf16 C and D, a bias library) gates its library's table on
    SCALAR_SHAPES and the tiny shapes: every command runs f8 data with both
    scales, K as the shape gives it, the bias epilogue exactly where the
    shape asks for it, fresh operands on the tiny shapes alone; stage 0
    writes the blocklist of the failing kernel."""
    import stage00_correctness_gate as s00

    cf.write_f8bs_library(world.lib)
    names = cf.f8bs_names()
    world.set(names={**cf.names(), **names}, fail={"301": "shape:3x257"})
    cfg = cf.f8bs_config(harness=str(world.harness))
    s = world.run(cfg=cfg)
    assert s["pool"] == 4 and s["verdicts"] == {"fail": 1, "pass": 3}
    assert world.blocked() == [names["301"]]
    calls = world.calls()
    shapes = cg.SCALAR_SHAPES
    assert cg.tiny_shapes_of(cg.harness_problem(cfg)) == cg.TINY_SHAPES
    assert {_shape_of(c["args"]) for c in calls} == {
        (str(x.m), str(x.n), str(x.k), str(x.batch)) for x in (*shapes, *cg.TINY_SHAPES)
    }
    tiny = {(str(x.m), str(x.n), str(x.k), str(x.batch)) for x in cg.TINY_SHAPES}
    biased = set()
    for c in calls:
        opts = dict(zip(c["args"][4::2], c["args"][5::2]))
        assert (opts.get("--fresh") == "1") == (_shape_of(c["args"]) in tiny)
        assert opts["--a-type"] == opts["--b-type"] == "f8"
        assert opts["--scale-a"] == opts["--scale-b"] == "1"
        assert (opts["--c-type"], opts["--d-type"], opts["--compute"]) == (
            "bf16",
            "bf16",
            "f32",
        )
        assert (opts["--trans-a"], opts["--trans-b"]) == ("T", "N")
        if "--bias" in opts:
            biased.add(_shape_of(c["args"]) + (opts.get("--bias-type", ""),))
    assert biased == {
        (str(x.m), str(x.n), str(x.k), str(x.batch), x.bias_type)
        for x in shapes
        if x.bias
    }
    v = json.loads((world.base / "gate" / cg.VERDICTS_FILE).read_text())
    by_sid = {k["solution_index"]: k for k in v["kernels"]}
    assert by_sid[301]["evidence"]["failing_shapes"] == {"b_over_4gib_batched": [3, 3]}
    assert [x["name"] for x in s["shapes"]] == [x.name for x in shapes]
    assert [x.get("bias_type", "") for x in s["shapes"]] == [
        x.bias_type for x in shapes
    ]
    hid = cg.harness_digest(world.harness, None)
    assert s["protocol"] == cg.protocol(cg.Settings(), hid, shapes)
    others = (cg.SHAPES, cg.mx_shapes(8, 8))
    assert s["protocol"] not in {cg.protocol(cg.Settings(), hid, x) for x in others}
    # Stage 0 takes the config and, its verdicts cached, writes the
    # blocklist without a harness run.
    n = len(calls)
    cfg_path = tmp_path / "f8bs.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    argv = ["--config-yaml", str(cfg_path), "--out-dir", str(tmp_path / "out")]
    argv += ["--dat-dir", str(world.lib), "--devices", "0"]
    assert s00.main(argv + ["--cache-dir", str(world.base / "cache")]) == 0
    assert len(world.calls()) == n
    blocked = cx.read_list(tmp_path / "out" / cx.BLOCKLIST_FILE)
    assert sorted(blocked.names) == [names["301"]]


def test_a_kernel_that_ignores_its_scale_fails(world):
    """A kernel that leaves out the scale of A or of B computes what an
    unscaled problem expects, so the bf16 gate passes it; the gate of 8-bit
    data with one f32 scale per tensor gives both operands their scales (2
    and 4) in every command and blocks it on every shape."""
    world.set(ignore_scale={"101": "A", "200": "B"})
    s = world.run()
    assert s["verdicts"] == {"pass": 6} and world.blocked() == []
    cfg = cf.scalar_config(harness=str(world.harness))
    s = world.run(out="gate_scalar", cache="cache_scalar", cfg=cfg)
    assert s["verdicts"] == {"fail": 2, "pass": 4}
    assert world.blocked("gate_scalar") == sorted([name(101), name(200)])
    v = json.loads((world.base / "gate_scalar" / cg.VERDICTS_FILE).read_text())
    by_sid = {k["solution_index"]: k for k in v["kernels"]}
    for sid in (101, 200):
        failing = by_sid[sid]["evidence"]["failing_shapes"]
        assert set(failing) == {x.name for x in cg.SCALAR_SHAPES}
        assert all(bad == n for bad, n in failing.values())
    assert cf.HARNESS_SCALES["a"] != 1.0 != cf.HARNESS_SCALES["b"]


def test_one_wrong_element_fails(world):
    """A kernel that returns one wrong element in one repetition of one
    shape is blocked, and the kernel of its family that passed that shape
    runs it again."""
    cf.write_f8bs_library(world.lib)
    names = cf.f8bs_names()
    world.set(
        names={**cf.names(), **names},
        fail={"303": "rate:10@13x65"},
        nbad={"303": 1},
    )
    s = world.run(cfg=cf.f8bs_config(harness=str(world.harness)))
    assert s["pool"] == 4 and s["verdicts"] == {"fail": 1, "pass": 3}
    assert world.blocked() == [names["303"]]
    v = json.loads((world.base / "gate" / cg.VERDICTS_FILE).read_text())
    ev = next(k for k in v["kernels"] if k["solution_index"] == 303)["evidence"]
    assert ev["failing_runs"] == 2
    assert ev["failing_shapes"] == {
        "small_m_batched": [1, 10],
        "small_m_batched_bias": [1, 10],
    }
    again = [
        c for c in world.calls() if c["args"][c["args"].index("--ids") + 1] == "302"
    ]
    assert len(again) == 2
    assert {_shape_of(c["args"]) for c in again} == {("13", "65", "383", "2048")}
    assert {c["args"][c["args"].index("--reps") + 1] for c in again} == {"10"}


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


def test_an_audit_set_takes_a_failure_the_canary_keeps_out_later(world, tmp_path):
    """The gate finds 101 computing wrong results. In the run's next
    invocation the hazard canary keeps 101 out unchecked (its run alone
    killed from outside), and 102 too, which the gate passed; the gate
    lists both with the canary's entries alone. An audit set built for the
    run takes 101, from the run's newest blocklist and from that snapshot as
    its config, since an older snapshot names it for the gate's finding,
    and leaves 102 to its own canary."""
    from lib import audit as al
    from lib import hazards as hz

    world.set(fail={"101": "always"})
    world.run(out="inv1")
    why = "its bench was killed by SIGKILL from outside"
    reason = hz.unchecked_reason(["prefetch_gl2_2"], why)
    quarantine = tmp_path / "quarantine.json"
    cuts = [cx.Entry(name(s), reason, source=hz.SOURCE) for s in (101, 102)]
    cx.write_blocklist(quarantine, cx.Blocklist(cuts))
    world.run(out="inv2", quarantine=quarantine)
    first = cx.read_list(world.base / "inv1" / cx.BLOCKLIST_FILE)
    second = cx.read_list(world.base / "inv2" / cx.BLOCKLIST_FILE)
    assert [(e.name, e.source) for e in first.entries] == [
        (name(101), "correctness gate")
    ]
    assert {e.name for e in second.entries} == {name(101), name(102)}
    assert all(hz.kept_out_unchecked(e) for e in second.entries)
    run = tmp_path / "run"
    for stamp, out in (("20261019_000001", "inv1"), ("20261019_000002", "inv2")):
        copy = run / "correctness" / "blocklists" / f"{stamp}_full.json"
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes((world.base / out / cx.BLOCKLIST_FILE).read_bytes())
        snap = run / "configs" / f"{stamp}_full.yaml"
        snap.parent.mkdir(parents=True, exist_ok=True)
        sec = {"enabled": True, "blocklist": str(copy)}
        snap.write_text(yaml.safe_dump(dict(world.cfg(), correctness=sec)))
    for cfg in (yaml.safe_load(snap.read_text()), world.cfg()):
        taken, _ = al.source_blocklist(cfg, [run])
        assert taken.names == {name(101)}


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
    n_shapes = len(cg.SHAPES) + len(cg.TINY_SHAPES)
    assert ids.count("101") == n_shapes, "each crash again alone, in both stages"
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


def test_lines_with_nan_or_infinite_elements_are_read():
    """The harness prints a NaN or an infinity as printf does (an element a
    kernel did not write keeps a NaN pattern); such a line is a record, read
    with those values, and not a line to skip."""
    line = (
        '{"sid": 6, "rep": 0, "supported": true, "ws_need": 0, "nbad": 3, '
        '"nnan": 1, "maxabs": inf, "bad": [[60,36,0,-nan,88,0,""], '
        '[1,2,0,nan,4,0,"extra chunks [0,1]"], [3,4,1,-inf,2,0,""]], '
        '"name": "Cijk_nan_inf_K6"}'
    )
    rec = cg._parse_line(line)
    assert (rec["sid"], rec["nbad"], rec["nnan"], rec["name"]) == (
        6,
        3,
        1,
        "Cijk_nan_inf_K6",
    )
    assert rec["maxabs"] == float("inf")
    assert [math.isnan(b[3]) for b in rec["bad"]] == [True, True, False]
    assert (
        rec["bad"][2][3] == float("-inf") and rec["bad"][1][6] == "extra chunks [0,1]"
    )
    assert cg._parse_line('{"sid": 6, "nbad": nan-ish}') is None
    assert cg._parse_line('{"ready": true, "algos": 1, "sids": [6]}')["ready"]


def test_a_kernel_that_leaves_elements_unwritten_fails(world):
    """Its every line holds a NaN and an infinity: each is a failing
    repetition, so the kernel is blocked rather than left without records."""
    world.set(unwritten=[102])
    s = world.run()
    assert s["verdicts"] == {"fail": 1, "pass": 5}
    assert world.blocked() == [name(102)]
    v = json.loads((world.base / "gate" / cg.VERDICTS_FILE).read_text())
    ev = next(k for k in v["kernels"] if k["solution_index"] == 102)["evidence"]
    assert set(ev["failing_shapes"]) == {x.name for x in cg.SHAPES}
    assert ev["failing_runs"] == ev["runs"]


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
    records = cg.VerdictCache(world.base / "cache").records()
    cached = {
        r["name"]: r["verdict"] for r in records if r.get("stage", cg.MAIN) == cg.MAIN
    }
    assert cached == {
        name(100): "pass",
        name(101): "fail",
        name(103): "pass",
        name(200): "pass",
        name(201): "pass",
    }
    tiny = {r["name"]: r["verdict"] for r in records if r.get("stage") == cg.TINY}
    assert tiny == {name(s): "pass" for s in (100, 101, 103, 200, 201)}
    world.set(broken_ids=[])
    s = world.run(out="gate2")
    assert s["cached"] == 5 and s["checked"] == 1
    assert s["tiny"]["cached"] == 5 and s["tiny"]["checked"] == 1
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
    vector = cf.mx_config("f8_r", harness=str(world.harness))
    vector["hipblaslt"].update(scaleA=2, scaleB=2)
    s = world.run(cfg=vector, dry_run=True)
    assert s["checked"] == 0 and world.calls() == []
    assert [x["name"] for x in s["shapes"]] == [x.name for x in cg.SHAPES]
    with pytest.raises(cg.GateError, match="scaled"):
        world.run(cfg=vector, out="gate2")
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
        "--drop-scale",
    ):
        assert f'"{flag}"' in text
    for attribute in (
        "HIPBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0",
        "HIPBLASLT_MATMUL_MATRIX_SCALE_SCALAR_32F",
        "HIPBLASLT_MATMUL_DESC_A_SCALE_POINTER",
        "HIPBLASLT_MATMUL_DESC_B_SCALE_POINTER",
        "HIPBLASLT_MATMUL_DESC_A_SCALE_MODE",
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
    and B, block-32 or one f32 per tensor, are distinct powers of 2 other
    than 1, so a kernel that drops a scale or applies one operand's to both
    is wrong; the fake harness gives the same scales."""
    text = cg.HARNESS_SOURCE.read_text()
    for kind in ("f8", "bf8", "f4"):
        found = re.search(rf"\b{kind}\[5\]\s*=\s*\{{([^}}]*)\}}", text)
        codes = [int(x, 16) for x in found.group(1).split(",")]
        assert [_decode(kind, c) for c in codes] == [-2, -1, 0, 1, 2], kind
    exps = [int(re.search(rf"kScaleExp{op}\s*=\s*(\d+)", text).group(1)) for op in "AB"]
    assert 0 not in exps and exps[0] != exps[1]
    assert cf.HARNESS_SCALES == {"a": 2.0 ** exps[0], "b": 2.0 ** exps[1]}
    assert cf.HARNESS_SCALES == {"a": 2.0, "b": 4.0}
    formats = {
        name: int(re.search(rf"\bk{name}\s*=\s*(\d+);", text).group(1))
        for name in ("Scalar", "Block32")
    }
    assert formats == {"Scalar": cg.SCALAR_SCALING, "Block32": cg.MX_SCALING}
    assert "std::ldexp(1.f, exp)" in text, "the f32 scale is 2^exp"


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
    mx["hipblaslt"].update(scaleA=2, scaleB=2)
    cfg_path.write_text(yaml.safe_dump(mx))
    argv[argv.index("--out-dir") + 1] = str(tmp_path / "out_vector")
    assert s00.main(argv) == 1


def test_stage00_counts_the_tiny_stage_in_its_summary_line(world, tmp_path, capsys):
    """A gate whose cache holds every kernel's verdict of the shapes but not
    of the tiny shapes runs the tiny shapes alone, and its summary line
    says so; with both stages cached it has nothing to check."""
    import stage00_correctness_gate as s00

    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(yaml.safe_dump(world.cfg(cache_dir=str(tmp_path / "shared"))))
    argv = ["--config-yaml", str(cfg_path), "--dat-dir", str(world.lib)]
    argv += ["--devices", "0", "--out-dir"]
    assert s00.main([*argv, str(tmp_path / "first")]) == 0
    first = capsys.readouterr().out
    assert "6 checked now on the shapes and 6 on the tiny shapes" in first
    cache = tmp_path / "shared" / cg.CACHE_FILE
    kept = [
        line
        for line in cache.read_text().splitlines()
        if json.loads(line).get("stage") != cg.TINY
    ]
    cache.write_text("\n".join(kept) + "\n")
    assert s00.main([*argv, str(tmp_path / "tiny")]) == 0
    tiny = capsys.readouterr().out
    assert "0 checked now on the shapes and 6 on the tiny shapes" in tiny
    assert "nothing to check" not in tiny
    assert s00.main([*argv, str(tmp_path / "cached")]) == 0
    assert "nothing to check" in capsys.readouterr().out


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


def test_the_audit_config_lists_the_blocklist_of_the_runs_it_is_built_for(tmp_path):
    from lib import audit as al

    cfg = cf.config(lists=["gfx950_bbs_tn"])
    assert "correctness" not in al.bench_config(cfg)
    built = al.bench_config(cfg, None, tmp_path / "source.json")
    assert built["correctness"] == {"lists": [str(tmp_path / "source.json")]}
    assert cfg["correctness"]["lists"] == ["gfx950_bbs_tn"]


def test_the_audit_set_checks_read_the_kernels_a_run_does_not_block(tmp_path):
    """The kernels an audit plan's solution indices named are the pool's,
    blocked kernels too; the kernels the set must hold for a run are those
    its blocklist leaves in: the lists before the gate wrote the
    invocation's blocklist, both after."""
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
    required = ro.config_pool_digests(cfg, paths, required=True)
    assert sorted(required) == [100, 102, 103, 200, 201]
    gate = cx.Blocklist([cx.Entry(name(102), source="correctness gate")])
    cx.write_blocklist(Path(cfg["correctness"]["blocklist"]), gate)
    required = ro.config_pool_digests(cfg, paths, required=True)
    assert sorted(required) == [100, 103, 200, 201]
    audit = tmp_path / "audit"
    run_layout.settle(audit)
    (audit / "stage02").mkdir(parents=True)
    (audit / "stage02" / ro.STAGE_MARKER).write_text("{}")
    planned = {str(s): d for s, d in required.items()}
    m = {"schema": al.SCHEMA_VERSION, "generations": [], "planned_kernels": planned}
    (audit / al.MANIFEST).write_text(json.dumps(m))
    assert ro.audit_build_refusal(audit, digests, required) is None
    m["generations"] = [{"kernels": planned}]
    (audit / al.MANIFEST).write_text(json.dumps(m))
    assert ro.audit_build_refusal(audit, digests, required) is None
    assert al.unbenched(m, digests) == [101, 102]
    m["left_out"] = {digests[101]: name(101), digests[102]: name(102)}
    assert al.unbenched(m, digests) == [], "a kernel the set left out is covered"


def test_a_run_finds_complete_an_audit_set_that_lacks_only_what_it_blocks(tmp_path):
    """A kernel the run's blocklist names, which the audit set neither
    benched nor left out (it entered the pool after the set was built for
    another run), keeps no run from finding the set complete; one the run
    does not block does."""
    import types

    import run_orchestrator as ro
    from lib import audit as al

    lib = cf.write_library(tmp_path / "lib")
    audit = tmp_path / "audit"
    audit.mkdir()
    lst = tmp_path / "l.tsv"
    lst.write_text(f"A\t{name(101)}\n")
    cfg = cf.config(lists=[str(lst)])
    cfg["audit"] = {"dir": str(audit)}
    paths = types.SimpleNamespace(library_dir=lib)
    digests = ro.config_pool_digests(cfg, paths)
    benched = {str(s): d for s, d in digests.items() if s != 101}
    m = {"schema": al.SCHEMA_VERSION, "generations": [{"kernels": benched}]}
    (audit / al.MANIFEST).write_text(json.dumps(m))
    (audit / al.COMPLETE_MARKER).write_text("{}")
    assert ro.audit_status(cfg, paths) == ("ok", None)
    del cfg["correctness"]
    st, why = ro.audit_status(cfg, paths)
    assert st == "incomplete" and "1 kernel(s)" in why


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
    assert "the exact harness does not run hipblaslt.a_type 'f4_r' with one" in errors
    cfg["hipblaslt"].update(a_type="f8_r", b_type="f8_r")
    assert not [e for e in ro.validate_config(cfg)[0] if "correctness" in e]
    cfg["hipblaslt"].update(scaleA=2, scaleB=2)
    errors = "\n".join(ro.validate_config(cfg)[0])
    assert "correctness.enabled: the exact harness does not run scaled" in errors
    cfg["correctness"] = {"repetitions": 2}
    assert any("repetitions" in e for e in ro.validate_config(cfg)[0])
    cfg["hipblaslt"].update(scaleA=1, scaleB=1)
    cfg["correctness"] = {"enabled": True, "custom_kernels": str(tmp_path / "gone")}
    errors = "\n".join(ro.validate_config(cfg)[0])
    assert "correctness.custom_kernels:" in errors and "is no directory" in errors
    cfg["correctness"]["custom_kernels"] = str(tmp_path)
    cfg["correctness"]["static_check"] = False
    assert not [e for e in ro.validate_config(cfg)[0] if "correctness" in e]
    assert ro.cfg_get(cf.config(), "correctness.static_check") is True


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


def test_stage08_takes_names_that_differ_in_ia_and_pka_tokens_for_one(tmp_path, capsys):
    import argparse

    import stage08_optional_validate_selection_efficiency as s08

    train = tmp_path / "round_2" / "stage05"
    train.mkdir(parents=True)
    own = tmp_path / "own.tsv"
    own.write_text("A\tCijk_a_K1_PKA0\n")
    snapshot = tmp_path / "snapshot.yaml"
    snapshot.write_text(yaml.safe_dump({"correctness": {"lists": [str(own)]}}))
    (train / "metrics.json").write_text(
        json.dumps({"args": {"config_yaml": str(snapshot)}})
    )
    now = tmp_path / "now.tsv"
    now.write_text("A\tCijk_a_IA0_K1_PKA1\n")
    cfg = {"correctness": {"lists": [str(now)]}}
    args = argparse.Namespace(train_dir=train, library_stem=cf.STEM, quiet=False)
    assert s08.bundle_blocklist_of(args, cfg).names == {"Cijk_a_K1_PKA0"}
    assert "the bundle serves with its own blocklist" not in capsys.readouterr().err


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
    fh.write_fake_build(
        v3, "gfx950", OSTEM, capabilities=("MLREC_v3", ro.CU_FLOOR_MARKER)
    )
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
        v3,
        "gfx950",
        OSTEM,
        capabilities=("MLREC_v3", ro.CU_FLOOR_MARKER, cx.SIDECAR_SUFFIX),
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


# ── the static check of hand-written kernels and the tiny shapes ────────────


def _custom_source(directory, name, widths=(4, 4), extra=""):
    """A hand-written kernel's source in `directory` whose custom.config
    declares global-read vector widths `widths` (None: undeclared)."""
    keys = "".join(
        f"  GlobalReadVectorWidth{op}: {w}\n"
        for op, w in zip("AB", widths)
        if w is not None
    )
    path = Path(directory) / f"{name}.s"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "/* GlobalReadVectorWidthA=1, GlobalReadVectorWidthB=1 */\n"
        ".amdgpu_metadata\n---\ncustom.config:\n"
        "  InternalSupportParams:\n    KernArgsVersion: 2\n"
        f"{keys}{extra}"
        "amdhsa.version:\n  - 1\n  - 1\n...\n.end_amdgpu_metadata\n"
    )
    return path


def _with_predicates(lib, sid, predicates, trans=None):
    """Give solution `sid` of the fake library's table the problem
    predicates `predicates` (an And of them) and the layout `trans`."""
    import msgpack

    path = lib / f"{cf.STEM}.dat"
    doc = msgpack.unpackb(path.read_bytes(), raw=False, strict_map_key=False)
    for sol in doc["solutions"]:
        if sol["index"] == sid:
            sol["problemPredicate"] = {"type": "And", "value": list(predicates)}
            if trans is not None:
                sol["problemType"] = {"transA": trans[0], "transB": trans[1]}
    path.write_bytes(msgpack.packb(doc))


def _lf(m, n):
    return [
        {"type": "LeadingFree0SizesGreaterOrEqual", "value": m},
        {"type": "LeadingFree1SizesGreaterOrEqual", "value": n},
    ]


def test_the_custom_config_is_read_from_the_metadata_of_the_source(tmp_path):
    src = _custom_source(tmp_path, "Custom_k", (4, 8))
    config = cg.custom_config(src)
    assert config["GlobalReadVectorWidthA"] == 4
    assert config["GlobalReadVectorWidthB"] == 8
    bare = tmp_path / "bare.s"
    bare.write_text(".amdgpu_metadata\n---\namdhsa.version: [1, 1]\n...\n")
    with pytest.raises(ValueError, match="no custom.config"):
        cg.custom_config(bare)
    broken = tmp_path / "broken.s"
    broken.write_text("---\ncustom.config: [\n...\n")
    with pytest.raises(ValueError, match="not YAML"):
        cg.custom_config(broken)


def test_sources_are_found_as_tensilelite_finds_them(tmp_path):
    root = tmp_path / "CustomKernels"
    _custom_source(root / "b", "K1", (8, 8))
    _custom_source(root / "a", "K1", (2, 2))
    _custom_source(root, "K2")
    _custom_source(root / "a", "K2", (1, 1))
    found = cg.custom_kernel_sources(root)
    assert found["K1"] == root / "a" / "K1.s", "the first of a sorted walk"
    assert found["K2"] == root / "K2.s", "the top of the directory first"
    build = tmp_path / "build"
    build.mkdir()
    src = tmp_path / "src"
    (build / "CMakeCache.txt").write_text(f"hipblaslt_SOURCE_DIR:STATIC={src}\n")
    d, why = cg.custom_kernels_dir({}, build)
    assert d is None and "has no" in why
    (src / cg.CUSTOM_KERNELS_DIR).mkdir(parents=True)
    d, why = cg.custom_kernels_dir({}, build)
    assert d == src / cg.CUSTOM_KERNELS_DIR and "build's hipBLASLt source" in why
    given = {"correctness": {"custom_kernels": str(root)}}
    assert cg.custom_kernels_dir(given, build) == (root, "correctness.custom_kernels")
    gone = {"correctness": {"custom_kernels": str(tmp_path / "gone")}}
    assert cg.custom_kernels_dir(gone, build)[0] is None
    assert cg.custom_kernels_dir({}, None)[0] is None


def test_the_least_free_size_counts_every_size_predicate():
    preds = cg.predicate_leaves(
        {
            "type": "And",
            "value": [
                {"type": "LeadingFree0SizesGreaterOrEqual", "value": 2},
                {"type": "And", "value": [{"type": "Free0SizeMultiple", "value": 8}]},
                {"type": "LeadingFree1SizesGreaterOrEqual", "value": 1},
                {"type": "SizeGreaterThan", "index": 1, "value": 5},
                {"type": "BoundSizeMultiple", "index": -1, "value": 64},
                {"type": "Or", "value": [{"type": "Free1SizeMultiple", "value": 99}]},
            ],
        }
    )
    assert cg.free_size_floor(preds, 0) == (
        8,
        ["LeadingFree0SizesGreaterOrEqual 2", "Free0SizeMultiple 8"],
    )
    assert cg.free_size_floor(preds, 1) == (
        6,
        ["LeadingFree1SizesGreaterOrEqual 1", "SizeGreaterThan 5"],
    )
    assert cg.free_size_floor([], 0) == (1, [])
    equal = [{"type": "SizeEqual", "index": 0, "value": 256}]
    assert (
        cg.free_size_floor(equal, 0)[0] == 256 and cg.free_size_floor(equal, 1)[0] == 1
    )


@pytest.mark.parametrize(
    "trans, sides",
    [
        ((False, False), ["A"]),
        ((False, True), ["A", "B"]),
        ((True, False), []),
        ((True, True), ["B"]),
    ],
)
def test_a_hand_written_kernel_is_unguarded_where_it_reads_along_a_free_size(
    tmp_path, trans, sides
):
    """The TensileLite custom-kernel path builds GlobalReadVectorWidth 1 and
    LeadingFree predicates of 1, while the kernel's custom.config reads A
    and B 4 wide: along M of a non-transposed A and N of a transposed B,
    the predicates admit sizes the kernel reads outside the operand at."""
    src = _custom_source(tmp_path, "Custom_k")
    sol = {
        "problemType": {"transA": trans[0], "transB": trans[1]},
        "problemPredicate": {"type": "And", "value": _lf(1, 1)},
        "sizeMapping": {"grvwA": 1, "grvwB": 1},
    }
    f = cg.check_custom_kernel(7, sol, "Custom_k", src)
    assert f.blocks == bool(sides)
    assert f.status == (cg.UNGUARDED if sides else "guarded")
    for op in sides:
        dim = "M" if op == "A" else "N"
        assert (
            f"admit {dim} of 1 to 3, below the global-read vector width 4" in f.reason
        )
        assert f.sides[op]["declared_width"] == 4 and f.sides[op]["built_width"] == 1
    assert f.sha256 == cg.sha256_file(src) and f.source == str(src)
    guarded = dict(sol, problemPredicate={"type": "And", "value": _lf(4, 4)})
    assert cg.check_custom_kernel(7, guarded, "Custom_k", src).status == "guarded"


def test_the_static_check_names_what_it_cannot_check(tmp_path):
    sol = {"problemType": {"transA": False, "transB": False}}
    undeclared = _custom_source(tmp_path, "U", (None, 4))
    f = cg.check_custom_kernel(1, sol, "U", undeclared)
    assert f.status == "undeclared" and not f.blocks and "for A" in f.reason
    f = cg.check_custom_kernel(1, sol, "N", None, missing="no sources here")
    assert (f.status, f.reason) == ("unchecked", "no sources here")
    broken = tmp_path / "broken.s"
    broken.write_text("nothing")
    assert cg.check_custom_kernel(1, sol, "B", broken).status == "unchecked"
    # The config's layout stands for a solution that records none.
    tn = cg.check_custom_kernel(
        1, {}, "U", _custom_source(tmp_path, "T"), (True, False)
    )
    assert tn.status == "guarded"
    nn = cg.check_custom_kernel(1, {}, "T", tmp_path / "T.s", (False, False))
    assert nn.status == cg.UNGUARDED


def test_only_hand_written_kernels_are_checked_statically(world, tmp_path):
    lib = world.lib
    _with_predicates(lib, 103, _lf(1, 1), trans=(False, False))
    root = tmp_path / "ck"
    _custom_source(root, cf.solution(103, cf.TABLE[103])["name"])
    sources = cg.custom_kernel_sources(root)
    found = cg.static_check(lib, [cf.STEM, cf.BANK], [100, 101, 103, 200], sources)
    assert list(found) == [103] and found[103].blocks
    generated = {"customKernel": {"name": "x", "generated": True}}
    assert cg.custom_kernel_name(generated) is None
    assert cg.custom_kernel_name({"customKernel": {"name": "x"}}) == "x"
    assert cg.custom_kernel_name({"customKernel": {"generated": False}}) is None
    nothing = cg.static_check(lib, [cf.STEM], [103], None, missing="why")
    assert nothing[103].status == "unchecked" and nothing[103].reason == "why"


def _nn_world(world, tmp_path, widths=(4, 4), **correctness):
    """The world's library in NN with the hand-written 103 under
    LeadingFree predicates of 1 and a source that declares `widths`."""
    _with_predicates(world.lib, 103, _lf(1, 1), trans=(False, False))
    root = tmp_path / "CustomKernels"
    _custom_source(root, name(103), widths)
    cfg = world.cfg(custom_kernels=str(root), **correctness)
    cfg["hipblaslt"].update(transA="N", transB="N")
    return cfg


def test_the_gate_blocks_an_unguarded_kernel_statically_and_never_runs_it(
    world, tmp_path
):
    cfg = _nn_world(world, tmp_path)
    s = world.run(cfg=cfg)
    assert world.blocked() == [name(103)]
    assert s["verdicts"] == {"pass": 5, cg.UNGUARDED: 1}
    assert s["static"]["statuses"] == {cg.UNGUARDED: 1}
    assert s["static"]["unguarded"] == [{"solution_index": 103, "name": name(103)}]
    assert s["static"]["sources"] == str(tmp_path / "CustomKernels")
    assert all(
        "103" not in c["args"][c["args"].index("--ids") + 1].split(",")
        for c in world.calls()
    )
    v = json.loads((world.base / "gate" / cg.VERDICTS_FILE).read_text())
    row = next(k for k in v["kernels"] if k["solution_index"] == 103)
    assert row["verdict"] == cg.UNGUARDED and row["source"] == "static check"
    assert row["static"]["sides"]["A"]["leading_free"] is True
    assert "below the global-read vector width 4 its custom.config" in row["reason"]
    doc = json.loads((world.base / "gate" / cx.BLOCKLIST_FILE).read_text())
    (entry,) = doc["entries"]
    assert entry["source"] == "correctness gate (static check)"
    assert entry["reason"].startswith("unguarded in the static check")
    assert entry["kernel_id"] == row["kernel_id"]
    # A gate with the check off runs it, and the tiny shapes find it.
    world.set(reads_before={"103": {"A": 4}}, depth_u={"103": 64})
    s = world.run(
        out="gate_off",
        cache="cache_off",
        cfg=dict(cfg, correctness=dict(cfg["correctness"], static_check=False)),
    )
    assert s["static"]["enabled"] is False and world.blocked("gate_off") == [name(103)]
    v = json.loads((world.base / "gate_off" / cg.VERDICTS_FILE).read_text())
    row = next(k for k in v["kernels"] if k["solution_index"] == 103)
    assert row["verdict"] == "crash" and row["tiny"]["verdict"] == "crash"
    assert set(row["tiny"]["evidence"]["failing_shapes"]) == {
        f"tiny_m_{m}_k384" for m in (1, 2, 3)
    }


def test_tiny_shapes_reach_every_free_size_below_a_vector_width():
    for shapes, k in ((cg.TINY_SHAPES, 383), (cg.MX_TINY_SHAPES, 416)):
        assert [(s.m, s.n) for s in shapes] == [
            (1, 257),
            (2, 257),
            (3, 257),
            (257, 1),
            (257, 2),
            (257, 3),
        ]
        assert all(s.fresh and s.stage == cg.TINY and s.k == k for s in shapes)
        assert all(s.batch == 1 and not s.trigger for s in shapes)
    assert cg.MX_TINY_SHAPES[0].k % 32 == 0 and cg.MX_TINY_SHAPES[0].k // 32 % 2 == 1
    assert all(cg.TINY_SHAPES[0].k % d for d in (2, 16, 32, 64, 128, 256))
    for shapes in (cg.SHAPES, cg.SCALAR_SHAPES, cg.mx_shapes(8, 4)):
        assert not any(s.fresh for s in shapes)
    assert cg.tiny_shapes_of(cg.harness_problem(cf.mx_config())) == cg.MX_TINY_SHAPES
    assert cg.tiny_shapes_of(cg.harness_problem(cf.config())) == cg.TINY_SHAPES
    assert cg.tiny_shapes_of(None) == cg.TINY_SHAPES
    rec = cg.TINY_SHAPES[0].record()
    assert rec["fresh"] is True and "fresh" not in cg.SHAPES[0].record()
    assert cg.TINY_SHAPES[0].text() == "1x257x383x1 fresh operands"


def test_only_tiny_shapes_ask_for_fresh_operands_and_only_of_a_harness_that_can():
    p = cg.harness_problem(cf.config())
    tiny = cg.Task("t", cg.TINY_SHAPES[0], [1], 3)
    assert tiny.stage == cg.TINY
    cmd = tiny.command(Path("h"), p, fresh=True)
    assert cmd[1:5] == ["1", "257", "383", "1"]
    assert cmd.index("--fresh") < cmd.index("--ids")
    assert "--fresh" not in tiny.command(Path("h"), p)
    main = cg.Task("m", cg.SHAPES[0], [1], 3)
    assert main.stage == cg.MAIN
    assert main.command(Path("h"), p, fresh=True) == main.command(Path("h"), p)
    args = tiny.shape.args(True, False, fresh=True)
    assert args[args.index("--fresh") + 1] == "1"


@pytest.mark.parametrize("layout", ["TN", "NN", "NT", "TT"])
@pytest.mark.parametrize(
    "make", [cf.config, cf.scalar_config, cf.mx_config, lambda: cf.mx_config("f4_r")]
)
def test_the_harness_takes_every_tiny_task_in_every_layout(layout, make):
    cfg = make()
    cfg["hipblaslt"].update(transA=layout[0], transB=layout[1])
    problem = cg.harness_problem(cfg)
    shapes = cg.tiny_shapes_of(problem)
    kernels = {7: {"mt_k": 64}, 8: {"mt_k": 256}}
    tasks = cg.main_tasks([7, 8], cg.Settings(), 1, shapes)
    unsupported = {s.name: {7, 8} for s in shapes}
    tasks += cg.roundup_tasks(unsupported, kernels, cg.Settings(), shapes)
    assert all(t.stage == cg.TINY for t in tasks)
    for t in tasks:
        args = t.command(Path("h"), problem, fresh=True)[1:]
        opts = dict(zip(args[4::2], args[5::2]))
        m, n, k = (int(x) for x in args[:3])
        assert not cf._type_error(k, opts), t.tag
        assert not cf._leading_dimension_error(m, n, k, opts), t.tag
        assert opts["--fresh"] == "1"


def test_tiny_shapes_are_a_stage_with_a_protocol_of_its_own():
    s = cg.Settings()
    main = cg.protocol(s, "harness-id", cg.SHAPES)
    tiny = cg.protocol(s, "harness-id", cg.TINY_SHAPES, cg.TINY)
    assert main == cg.protocol(s, "harness-id", cg.SHAPES, cg.MAIN)
    assert main == "629690fa63793e23", "the main stage's protocol is unchanged"
    assert tiny != cg.protocol(s, "harness-id", cg.TINY_SHAPES) != main
    assert tiny != cg.protocol(s, "other", cg.TINY_SHAPES, cg.TINY)


def test_a_cached_gate_runs_only_the_tiny_shapes(world):
    """A cache of the shapes before the tiny stage existed (its records
    without the tiny ones) keeps its verdicts; the next gate runs every
    kernel on the tiny shapes alone, and no first launch."""
    world.set(fail={"101": "always"})
    world.run()
    cache = world.base / "cache" / cg.CACHE_FILE
    kept = [
        line
        for line in cache.read_text().splitlines()
        if json.loads(line).get("stage") != cg.TINY
    ]
    cache.write_text("\n".join(kept) + "\n")
    n = len(world.calls())
    s = world.run(out="gate2")
    assert s["checked"] == 0 and s["cached"] == 6
    assert s["tiny"]["checked"] == 5 and s["tiny"]["cached"] == 1, "101's failure"
    later = world.calls()[n:]
    tiny = {(str(x.m), str(x.n), str(x.k), str(x.batch)) for x in cg.TINY_SHAPES}
    assert later and all(
        _shape_of(c["args"]) in tiny or c["args"][2] == "384" for c in later
    )
    assert all(c["args"][c["args"].index("--fresh") + 1] == "1" for c in later)
    assert world.blocked("gate2") == [name(101)]
    assert s["tiny"]["harness_fresh"] is True
    assert s["tiny"]["tasks"] == s["tiny"]["fresh_tasks"] > 0


def test_a_kernel_with_a_verdict_of_one_stage_alone_is_unchecked(world):
    """A kernel whose cached verdicts hold one stage's and not the other's
    (here the main stage's, of a cache from before the tiny stage, read by
    a dry run) is unchecked, as one with neither, never passed; a failure of
    either stage decides it all the same."""
    world.set(fail={"101": "always"})
    world.run()
    cache = world.base / "cache" / cg.CACHE_FILE
    kept = [
        line
        for line in cache.read_text().splitlines()
        if json.loads(line).get("stage") != cg.TINY
    ]
    cache.write_text("\n".join(kept) + "\n")
    s = world.run(out="dry", dry_run=True)
    assert s["cached"] == 6 and s["tiny"]["cached"] == 1, "101's failure"
    v = json.loads((world.base / "dry" / cg.VERDICTS_FILE).read_text())
    verdicts = {k["solution_index"]: k["verdict"] for k in v["kernels"]}
    assert verdicts.pop(101) == "fail"
    assert set(verdicts.values()) == {"unchecked"}
    assert s["verdicts"] == {"fail": 1, "unchecked": 5}
    assert world.blocked("dry") == [name(101)]
    passed, crashed = {"verdict": "pass"}, {"verdict": "crash"}
    for main, tiny in ((passed, None), (None, passed), (None, None)):
        assert cg.kernel_verdict(main, tiny) == ("unchecked", None)
    assert cg.kernel_verdict(None, crashed) == ("crash", crashed)
    assert cg.kernel_verdict(passed, passed) == ("pass", passed)
    assert cg.kernel_verdict({"verdict": "untested"}, passed)[0] == "pass"


def test_a_read_before_an_operand_is_found_only_at_fresh_allocations(world):
    """101 reads A 4 wide along M without a guard: at M of 1 to 3 it reads
    before A. With the operands at fresh allocations the read faults, a
    crash that blocks it on the tiny shapes; on a harness without --fresh,
    or a device that cannot place them so, it reads another buffer and
    passes, and the gate says that the tiny shapes ran on usual
    allocations."""
    cfg = world.cfg()
    cfg["hipblaslt"].update(transA="N", transB="N")
    world.set(reads_before={"101": {"A": 4}})
    s = world.run(cfg=cfg)
    assert world.blocked() == [name(101)]
    v = json.loads((world.base / "gate" / cg.VERDICTS_FILE).read_text())
    row = next(k for k in v["kernels"] if k["solution_index"] == 101)
    assert row["verdict"] == "crash" and row["tiny"]["verdict"] == "crash"
    assert set(row["tiny"]["evidence"]["failing_shapes"]) == {
        f"tiny_m_{m}" for m in (1, 2, 3)
    }
    assert row["tiny"]["evidence"]["fresh"] is True
    assert row["evidence"]["failing_shapes"] == {}, "no main shape is below 4"
    assert s["verdicts"] == {"crash": 1, "pass": 5}
    for caps, alloc, out in ((None, True, "old"), (["fresh"], False, "novmm")):
        world.set(caps=caps, fresh_alloc=alloc)
        s = world.run(out=out, cache=f"cache_{out}", cfg=cfg)
        assert world.blocked(out) == [], out
        assert s["tiny"]["harness_fresh"] is (caps is not None)
        assert s["tiny"]["fresh_tasks"] == 0 and s["tiny"]["tasks"] > 0
        v = json.loads((world.base / out / cg.VERDICTS_FILE).read_text())
        row = next(k for k in v["kernels"] if k["solution_index"] == 101)
        assert row["tiny"]["evidence"]["fresh"] is False


def test_the_gate_warns_when_the_tiny_shapes_run_on_usual_allocations(world):
    warned = []
    world.set(caps=None)
    spec = cg.Spec(
        cfg=world.cfg(),
        dat_dir=world.lib,
        library_stem=cf.STEM,
        out_dir=world.base / "g",
        cache_dir=world.base / "c",
        devices=[0],
    )
    cg.run_gate(spec, warn=warned.append)
    assert any("takes no --fresh" in w for w in warned)
    assert not any("--fresh" in c["args"] for c in world.calls())
    world.set(caps=["fresh"], fresh_alloc=False)
    warned.clear()
    cg.run_gate(
        replace(spec, out_dir=world.base / "g2", cache_dir=world.base / "c2"),
        warn=warned.append,
    )
    assert any(
        "could not place their operands at fresh allocations" in w for w in warned
    )
    assert any("cannot be checked statically" in w for w in warned)


def test_a_tiny_pass_without_fresh_operands_is_checked_again(world):
    """A tiny-shape pass made without fresh operands is never taken from the
    cache: where the device could not place them, the harness's --caps
    failed, or the processes of the tiny M shapes alone could not, 101's
    read before A goes unseen and it passes; the next gate on the same
    cache, which can, runs every such kernel on the tiny shapes again and
    blocks it. 201's crash stands wherever it happened."""
    cfg = world.cfg()
    cfg["hipblaslt"].update(transA="N", transB="N")
    world.set(reads_before={"101": {"A": 4}}, crash=[201])
    tiny_m = [f"{s.m}x{s.n}" for s in cg.TINY_SHAPES if s.name.startswith("tiny_m")]
    for cache, before, after in (
        ("c_novmm", {"fresh_alloc": False}, {"fresh_alloc": True}),
        ("c_caps", {"caps": None}, {"caps": ["fresh"]}),
        ("c_mixed", {"no_fresh_shapes": tiny_m}, {"no_fresh_shapes": []}),
    ):
        world.set(**before)
        s = world.run(out=f"{cache}_1", cache=cache, cfg=cfg)
        assert world.blocked(f"{cache}_1") == [name(201)], cache
        assert s["tiny"]["fresh_tasks"] < s["tiny"]["tasks"], cache
        v = json.loads((world.base / f"{cache}_1" / cg.VERDICTS_FILE).read_text())
        row = next(k for k in v["kernels"] if k["solution_index"] == 101)
        assert row["tiny"]["verdict"] == "pass", cache
        assert row["tiny"]["evidence"]["fresh"] is False, cache
        world.set(**after)
        n = len(world.calls())
        s = world.run(out=f"{cache}_2", cache=cache, cfg=cfg)
        assert world.blocked(f"{cache}_2") == [name(101), name(201)], cache
        assert s["checked"] == 0 and s["tiny"]["checked"] == 5, cache
        assert s["tiny"]["cached"] == 1 and s["tiny"]["cached_not_fresh"] == 5
        ids = [c["args"][c["args"].index("--ids") + 1] for c in world.calls()[n:]]
        assert ids and not any("201" in i.split(",") for i in ids), cache
    for verdict in ("pass", "untested"):
        rec = {"stage": cg.TINY, "verdict": verdict, "evidence": {"fresh": False}}
        assert cg.unfresh(rec), rec
        assert cg.unfresh(dict(rec, evidence={})), "its evidence does not say"
        assert cg.unfresh(dict(rec, evidence="garbled")), "its evidence does not say"
    for rec in (
        {"stage": cg.TINY, "verdict": "crash", "evidence": {"fresh": False}},
        {"stage": cg.TINY, "verdict": "pass", "evidence": {"fresh": True}},
        {"stage": cg.TINY, "verdict": "untested", "evidence": {"fresh": None}},
        {"stage": cg.MAIN, "verdict": "pass", "evidence": {"fresh": False}},
    ):
        assert not cg.unfresh(rec), rec


def test_harness_caps_lists_what_the_harness_takes(world, tmp_path):
    assert cg.harness_caps(world.harness) == {"fresh"}
    world.set(caps=None)
    assert cg.harness_caps(world.harness) == set()
    world.set(caps=["fresh", "x"])
    assert cg.harness_caps(world.harness) == {"fresh", "x"}
    assert cg.harness_caps(tmp_path / "gone") == set()


def test_the_harness_source_places_operands_at_fresh_allocations():
    """The harness's --fresh maps every buffer of the problem at the start
    of an allocation of its own behind an unmapped reserved range, falls
    back to hipMalloc without virtual memory management, says which in its
    ready line, and lists the option in --caps without a HIP call."""
    text = cg.HARNESS_SOURCE.read_text()
    assert '"--fresh"' in text and '"--caps"' in text
    assert '\\"caps\\": [\\"fresh\\"]' in text
    assert '\\"fresh\\": %s' in text
    for call in (
        "hipDeviceAttributeVirtualMemoryManagementSupported",
        "hipMemGetAllocationGranularity",
        "hipMemAddressReserve",
        "hipMemCreate",
        "hipMemMap",
        "hipMemSetAccess",
        "hipMemUnmap",
        "hipMemRelease",
        "hipMemAddressFree",
    ):
        assert call in text, call
    main = text[text.index("int main(") :]
    caps = main.index('"--caps"')
    assert caps < main.index("parse(argc, argv)") < main.index("hipblasLtCreate")
    for buf in ("p.a", "p.b", "p.c", "p.scale_a", "p.scale_b", "p.bias"):
        assert f"memory.copy({buf})" in main, buf
    assert "memory.alloc(p.d.size())" in main and "hipMalloc(&dev_a" not in main
