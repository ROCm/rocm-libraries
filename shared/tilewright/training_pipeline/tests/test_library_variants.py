# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Output-type variants (lib/library_variants.py) and the index rows stage06
gives them."""

import json
from pathlib import Path

import pytest
import yaml

import stage06_deploy_weights as s6
from lib import library_variants as lv

ARCH = "gfx950"
FAMILY = "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx950"
OUTPUT = "TensileLibrary_BB_SB_HA_Bias_SAV_UA_Type_BS_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx950"
OUTPUT_CU = OUTPUT.replace("_gfx950", "_CU64_gfx950")
INPUT = "TensileLibrary_HH_HH_HA_Bias_SAV_UA_Type_HH_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx950"
SCALE = "TensileLibrary_BB_SB_HA_Bias_SABV_SAV_UA_Type_BS_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx950"
LAYOUT = "TensileLibrary_BB_SB_HA_Bias_SAV_UA_Type_BS_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk_gfx950"
TAGGED_ONLY = "TensileLibrary_BB_HB_HA_Bias_SAV_UA_Type_BH_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_CU64_gfx950"
LOGIC = "gfx950_Cijk_Alik_Bljk_BBS_BH_BiasSB_HAS_SAV_UserArgs"
MODEL = f"{LOGIC}.tilewright.bin"

BASE = {
    "aType": "BFloat16",
    "bType": "BFloat16",
    "cType": "BFloat16",
    "dType": "BFloat16",
    "eType": "BFloat16",
    "computeInputTypeA": "BFloat16",
    "computeInputTypeB": "BFloat16",
    "computeType": "Float",
    "biasDataTypeWhiteList": ["Float", "BFloat16"],
    "useScaleAB": "",
    "useScaleAlphaVec": 1,
    "transA": True,
    "transB": False,
    "operationIdentifier": "Contraction_l_Alik_Bljk_Cijk_Dijk",
}
FLOAT_OUT = dict(BASE, cType="Float", dType="Float", eType="Float")
HALF_OUT = dict(
    BASE, cType="Half", dType="Half", eType="Half", biasDataTypeWhiteList=["Float", "Half"]
)
HALF_IN = dict(
    BASE, aType="Half", bType="Half", computeInputTypeA="Half", computeInputTypeB="Half"
)
VECTOR_SCALE = dict(FLOAT_OUT, useScaleAB="Vector")
OTHER_LAYOUT = dict(
    FLOAT_OUT, transA=False, operationIdentifier="Contraction_l_Ailk_Bljk_Cijk_Dijk"
)


def write_library(
    d, stem, pt, *, pool=None, order=None, offset=0, table=True, extra=()
):
    """`pool` (test_stage05_train.kernel_pool entries; default all of it) as
    the library file `stem` in `d`, its solutions of problem type `pt` and
    numbered `sol_idx_global + offset`, its Prediction table (with `table`)
    in the order `order` (indices of `pool`), and the solutions `extra`
    outside the table."""
    import bench_fakes as bf
    from test_stage05_train import kernel_pool, library_solutions

    pool = kernel_pool() if pool is None else pool
    picked = [pool[i] for i in (order or range(len(pool)))]
    picked = [dict(k, sol_idx_global=k["sol_idx_global"] + offset) for k in picked]
    sols = [
        dict(s, problemType=dict(pt), name=f"{s['name']}_K{k['sol_idx_global']}")
        for s, k in zip(library_solutions(picked), picked)
    ]
    held = [s["index"] for s in sols]
    d.mkdir(parents=True, exist_ok=True)
    bf.write_logic(d / f"{stem}.dat", sols + list(extra), table=held if table else None)
    return d


def pool_keys(lib, stem, cfg):
    """The view-key hashes of the pool stage06 reads for `stem` from `lib`."""
    from lib import drift_fallback
    from lib import evaluate as ev
    from lib import features as fs

    pool = ev.library_pool(lib, stem, cfg)
    return set(drift_fallback.pool_hashes(pool, fs.DEFAULT_ATTRIBUTE_SPEC))


# ── problem types and variants ──────────────────────────────────────────────


def test_a_library_has_the_problem_type_of_its_pool(tmp_path):
    import bench_fakes as bf

    other = dict(bf.solution(999, 99), problemType=dict(HALF_IN))
    write_library(tmp_path, FAMILY, BASE, extra=[other])
    assert lv.problem_type(tmp_path, FAMILY) == BASE
    write_library(tmp_path, OUTPUT, FLOAT_OUT, table=False)
    assert lv.problem_type(tmp_path, OUTPUT) == FLOAT_OUT
    write_library(tmp_path, INPUT, BASE, table=False, extra=[other])
    with pytest.raises(ValueError, match="2 problem types"):
        lv.problem_type(tmp_path, INPUT)
    with pytest.raises(FileNotFoundError):
        lv.problem_type(tmp_path, SCALE)


def test_variants_differ_from_the_library_in_the_output_fields_alone(tmp_path):
    """Another output type is a variant, listed with every file of its key;
    another input type, scaling or operation is not, nor a key without an
    untagged file. The relation holds both ways."""
    for stem, pt in (
        (FAMILY, BASE),
        (OUTPUT, FLOAT_OUT),
        (OUTPUT_CU, FLOAT_OUT),
        (INPUT, HALF_IN),
        (SCALE, VECTOR_SCALE),
        (LAYOUT, OTHER_LAYOUT),
        (TAGGED_ONLY, HALF_OUT),
    ):
        write_library(tmp_path, stem, pt)
    (tmp_path / "TensileLibrary_lazy_gfx950.dat").write_bytes(b"not a contraction")
    found = lv.variants(tmp_path, FAMILY)
    assert [(v.key, v.files, v.differs) for v in found] == [
        (OUTPUT, sorted([OUTPUT, OUTPUT_CU]), ["cType", "dType", "eType"])
    ]
    assert [v.key for v in lv.variants(tmp_path, OUTPUT_CU)] == [FAMILY]
    assert lv.family(BASE) == lv.family(FLOAT_OUT) == lv.family(HALF_OUT)
    assert lv.family(BASE) != lv.family(HALF_IN)


def test_the_pool_gate_needs_coverage_both_ways_the_order_and_no_sidecars():
    full = lv.coverage(range(1, 11), range(1, 11))
    assert full == {
        "n_view_keys": 10,
        "n_trained_keys": 10,
        "trained_in_pool": 10,
        "share_pool_keys": 1.0,
        "share_model_in_pool": 1.0,
    }
    assert lv.pool_gate(full, True) == []
    # The least shares pass; a hash is read as an unsigned 64-bit value.
    edge = lv.coverage(list(range(1, 10)) + [-1], list(range(1, 10)) + [2**64 - 1])
    assert edge["trained_in_pool"] == 10
    nine = lv.coverage(range(1, 11), range(2, 11))
    assert nine["share_pool_keys"] == lv.MIN_SHARE and lv.pool_gate(nine, True) == []
    small = lv.coverage(range(1, 5), range(2, 6))
    reasons = lv.pool_gate(small, True)
    assert reasons == [
        "the model was trained on 3 of its pool's 4 view keys",
        "its pool holds 3 of the model's 4 trained view keys",
    ]
    assert lv.pool_gate(full, False) == [
        "its pool lists the candidate keys in another order than the library's"
    ]
    (sidecars,) = lv.pool_gate(full, True, ["a blocklist", "kernel guards"])
    assert "a blocklist and kernel guards" in sidecars
    assert lv.pool_gate(lv.coverage(range(3), []), True) == [
        "the model lists no trained view keys"
    ]
    assert lv.pool_gate(lv.coverage([], range(3)), True) == ["its pool holds no kernel"]


def test_a_record_of_measured_gates_is_read_by_variant_key_and_model(tmp_path):
    sha, other = "ab" * 32, "cd" * 32
    path = tmp_path / "measured.yaml"
    path.write_text(
        yaml.safe_dump(
            [
                {"variant": OUTPUT_CU, "model_sha256": sha, "passed": True, "n": 3},
                {"variant": OUTPUT, "model_sha256": sha, "passed": True},
                {"variant": OUTPUT, "model_sha256": other, "passed": False},
            ]
        )
    )
    assert lv.read_measured(path) == {(OUTPUT, sha): True, (OUTPUT, other): False}
    for bad in (
        {"variant": OUTPUT, "model_sha256": sha[:63], "passed": True},
        {"variant": OUTPUT, "model_sha256": sha.upper(), "passed": True},
        {"variant": OUTPUT, "model_sha256": sha, "passed": "yes"},
        {"variant": "", "model_sha256": sha, "passed": True},
        "not an entry",
    ):
        path.write_text(yaml.safe_dump([bad]))
        with pytest.raises(ValueError, match="entry 1 is not"):
            lv.read_measured(path)
    path.write_text(
        yaml.safe_dump(
            [
                {"variant": OUTPUT, "model_sha256": sha, "passed": True},
                {"variant": OUTPUT_CU, "model_sha256": sha, "passed": False},
            ]
        )
    )
    with pytest.raises(ValueError, match="disagree"):
        lv.read_measured(path)
    path.write_text(yaml.safe_dump({"variant": OUTPUT}))
    with pytest.raises(ValueError, match="not a list"):
        lv.read_measured(path)


def test_a_variant_gets_a_row_with_its_pool_gate_and_a_measured_pass_or_a_waiver():
    assert lv.decide([], True, False) == (True, "the measured gate passed")
    assert lv.decide([], None, True) == (True, "waived: no measured gate")
    assert lv.decide([], None, False) == (False, "no measured gate passed for this model")
    assert lv.decide([], False, True) == (False, "the measured gate failed")
    assert lv.decide(["a", "b"], True, True) == (False, "a; b")


# ── stage06 ─────────────────────────────────────────────────────────────────


def write_cfg(path, **variants):
    from test_stage06_deploy_weights import write_cfg as base

    cfg = yaml.safe_load(base(path, arch=ARCH, stem=FAMILY, logic=LOGIC).read_text())
    if variants:
        cfg["deploy"]["variants"] = variants
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture
def world(tmp_path):
    """A library directory with the family library and its variant (both
    files of the variant's key), a checkout with an index of another
    library, and a v3 bundle trained on every key of the family's pool."""
    from test_stage06_deploy_weights import LABELS, write_v3_bundle

    lib = tmp_path / "lib"
    write_library(lib, FAMILY, BASE)
    write_library(lib, OUTPUT, FLOAT_OUT, offset=1000)
    write_library(lib, OUTPUT_CU, FLOAT_OUT, offset=2000)
    repo = tmp_path / "repo"
    (repo / s6.wl.LOGIC_REL / ARCH).mkdir(parents=True)
    wd = repo / s6.weights_rel(ARCH)
    wd.mkdir(parents=True)
    (wd / "tilewright_index").write_text("# index\nOther_gfx950\tother.tilewright.bin\n")
    (wd / "other.tilewright.bin").write_bytes(b"other model")
    cfg = yaml.safe_load(write_cfg(tmp_path / "keys.yaml").read_text())
    keys = pool_keys(lib, FAMILY, cfg)
    train = tmp_path / "round_1" / "stage05"
    write_v3_bundle(train, LABELS, trained_view_keys=sorted(keys))
    return {"lib": lib, "repo": repo, "wd": wd, "train": train, "tmp": tmp_path}


def run(w, name, cfg, **kw):
    out = w["tmp"] / name
    rc = s6.deploy(
        train_dir=w["train"], config_yaml=cfg, out_dir=out, quiet=True,
        library_dir=w["lib"], **kw,
    )  # fmt: skip
    rec = json.loads((out / "deploy_record.json").read_text()) if rc == 0 else None
    return rc, rec, out


def variant_of(rec):
    (target,) = rec["variants"]["targets"]
    (variant,) = target["variants"]
    return variant


def index_of(w):
    return s6.parse_index((w["wd"] / "tilewright_index").read_text()).as_dict()


def test_a_variant_without_a_measured_pass_gets_no_row(world):
    rc, rec, out = run(world, "out", write_cfg(world["tmp"] / "cfg.yaml"))
    assert rc == 0
    v = variant_of(rec)
    assert v["key"] == OUTPUT and v["files"] == sorted([OUTPUT, OUTPUT_CU])
    assert v["differs_in"] == ["cType", "dType", "eType"]
    assert v["coverage"]["share_pool_keys"] == v["coverage"]["share_model_in_pool"] == 1.0
    assert v["same_order"] and v["pool_gate"] and v["measured"] is None
    assert not v["row"] and v["reason"] == "no measured gate passed for this model"
    assert rec["variants"]["measured"] is None and rec["variants"]["waived"] == []
    fragment = out / "staged_for_deploy" / s6.weights_rel(ARCH) / "tilewright_index.fragment"
    assert fragment.read_text() == f"{s6.bank_pool.problem_type_key(FAMILY)}\t{MODEL}\n"


def test_a_variant_with_a_measured_pass_for_the_model_gets_a_row(world, capsys):
    """The row names the deployed model under the variant's key, in the
    checkout, the staged index and its fragment; a pass of other model bytes
    does not count."""
    pytest.importorskip("tilewright")
    _, rec, _ = run(world, "first", write_cfg(world["tmp"] / "plain.yaml"))
    sha = rec["sha256"]
    measured = world["tmp"] / "gates" / "measured.yaml"
    measured.parent.mkdir()
    measured.write_text(
        yaml.safe_dump([{"variant": OUTPUT_CU, "model_sha256": "0" * 64, "passed": True}])
    )
    cfg = write_cfg(world["tmp"] / "gates" / "cfg.yaml", measured="measured.yaml")
    rc, rec, _ = run(world, "other", cfg, repo=world["repo"], apply_patch=True)
    assert rc == 0 and not variant_of(rec)["row"]
    assert index_of(world) == {FAMILY: MODEL, "Other_gfx950": "other.tilewright.bin"}
    measured.write_text(
        yaml.safe_dump([{"variant": OUTPUT_CU, "model_sha256": sha, "passed": True}])
    )
    capsys.readouterr()
    rc, rec, out = run(world, "pass", cfg, repo=world["repo"], apply_patch=True)
    assert rc == 0 and rec["sha256"] == sha
    v = variant_of(rec)
    assert v["row"] and v["measured"] == "passed"
    assert v["reason"] == "the measured gate passed"
    assert Path(rec["variants"]["measured"]) == measured.resolve()
    assert rec["targets"][0]["variant_rows"] == [OUTPUT]
    assert index_of(world) == {
        FAMILY: MODEL,
        OUTPUT: MODEL,
        "Other_gfx950": "other.tilewright.bin",
    }
    staged = out / "staged_for_deploy" / s6.weights_rel(ARCH)
    assert (staged / "tilewright_index").read_text() == (
        world["wd"] / "tilewright_index"
    ).read_text()
    assert (staged / "tilewright_index.fragment").read_text() == (
        f"{FAMILY}\t{MODEL}\n{OUTPUT}\t{MODEL}\n"
    )
    assert f"{OUTPUT} (an output-type variant)" in (
        out / "staged_for_deploy" / "INSTALL.md"
    ).read_text()


def test_a_waiver_gives_a_row_but_never_over_a_failed_gate_or_a_row_of_its_own(world):
    pytest.importorskip("tilewright")
    cfg = write_cfg(world["tmp"] / "waived.yaml", waived=[OUTPUT_CU, "Unknown_gfx950"])
    rc, rec, _ = run(world, "waived", cfg, repo=world["repo"], apply_patch=True)
    assert rc == 0 and rec["variants"]["waived"] == sorted([OUTPUT, "Unknown_gfx950"])
    v = variant_of(rec)
    assert v["row"] and v["waived"] and v["reason"] == "waived: no measured gate"
    assert index_of(world)[OUTPUT] == MODEL
    sha = rec["sha256"]
    measured = world["tmp"] / "failed.yaml"
    measured.write_text(
        yaml.safe_dump([{"variant": OUTPUT, "model_sha256": sha, "passed": False}])
    )
    cfg = write_cfg(
        world["tmp"] / "failed_cfg.yaml", waived=[OUTPUT], measured=str(measured)
    )
    rc, rec, _ = run(world, "failed", cfg, repo=world["repo"], apply_patch=True)
    assert rc == 0 and not variant_of(rec)["row"]
    assert variant_of(rec)["reason"] == "the measured gate failed"
    assert OUTPUT not in index_of(world)
    assert rec["targets"][0]["variant_rows_dropped"] == [OUTPUT]
    # A variant key whose row names a model of its own keeps it.
    wd = world["wd"]
    (wd / "own.tilewright.bin").write_bytes(b"own model")
    text = (wd / "tilewright_index").read_text()
    (wd / "tilewright_index").write_text(text + f"{OUTPUT}\town.tilewright.bin\n")
    cfg = write_cfg(world["tmp"] / "own.yaml", waived=[OUTPUT])
    rc, rec, _ = run(world, "own", cfg, repo=world["repo"], apply_patch=True)
    assert rc == 0 and variant_of(rec)["row"]
    assert rec["targets"][0]["variant_own_rows"] == [OUTPUT]
    assert rec["targets"][0]["variant_rows"] == []
    assert index_of(world)[OUTPUT] == "own.tilewright.bin"
    assert (wd / "own.tilewright.bin").read_bytes() == b"own model"


def test_a_redeploy_drops_the_row_of_a_variant_that_no_longer_passes(world):
    """The row an earlier deploy wrote names the model, which is no error for
    the next deploy of the library: it plans the row anew."""
    pytest.importorskip("tilewright")
    cfg = write_cfg(world["tmp"] / "waived.yaml", waived=[OUTPUT])
    assert run(world, "one", cfg, repo=world["repo"], apply_patch=True)[0] == 0
    assert index_of(world)[OUTPUT] == MODEL
    assert run(world, "two", cfg, repo=world["repo"], apply_patch=True)[0] == 0
    assert index_of(world)[OUTPUT] == MODEL
    plain = write_cfg(world["tmp"] / "plain.yaml")
    rc, rec, _ = run(world, "three", plain, repo=world["repo"], apply_patch=True)
    assert rc == 0 and rec["targets"][0]["variant_rows_dropped"] == [OUTPUT]
    assert index_of(world) == {FAMILY: MODEL, "Other_gfx950": "other.tilewright.bin"}
    assert sorted(p.name for p in world["wd"].iterdir()) == [
        MODEL,
        "other.tilewright.bin",
        "tilewright_index",
    ]


@pytest.mark.parametrize("change", ["order", "subset", "blocklist"])
def test_a_variant_whose_pool_fails_the_gate_gets_no_row_even_when_waived(
    world, monkeypatch, change
):
    from test_stage05_train import kernel_pool

    pool = kernel_pool()
    if change == "order":
        backwards = list(range(len(pool)))[::-1]
        write_library(world["lib"], OUTPUT, FLOAT_OUT, offset=1000, order=backwards)
        want = "in another order than the library's"
    elif change == "subset":
        write_library(world["lib"], OUTPUT, FLOAT_OUT, pool=pool[:-2], offset=1000)
        want = "trained view keys"
    else:
        monkeypatch.setattr(s6, "blocklist_names", lambda *a: ["Cijk_Blocked_K1"])
        monkeypatch.setattr(s6, "served_guards", lambda *a: {})
        want = "ships with a blocklist"
    cfg = write_cfg(world["tmp"] / "waived.yaml", waived=[OUTPUT])
    rc, rec, _ = run(world, "out", cfg)
    assert rc == 0
    v = variant_of(rec)
    assert not v["row"] and not v["pool_gate"] and v["waived"]
    assert want in v["reason"], v["reason"]


def test_a_record_of_measured_gates_that_cannot_be_read_refuses_the_deploy(world, capsys):
    measured = world["tmp"] / "measured.yaml"
    measured.write_text(yaml.safe_dump([{"variant": OUTPUT, "passed": True}]))
    cfg = write_cfg(world["tmp"] / "cfg.yaml", measured=str(measured))
    assert run(world, "bad", cfg)[0] == 1
    assert "deploy.variants.measured" in capsys.readouterr().err
    cfg = write_cfg(world["tmp"] / "missing.yaml", measured="missing.yaml")
    assert run(world, "missing", cfg)[0] == 1


def test_the_variant_settings_change_no_pin(config_env, pipeline_dir):
    import run_orchestrator as ro

    cfg, _ = ro.load_config(pipeline_dir / "configs" / "gfx1250_bbs_tn_v3.yaml")
    pinned = ro.pinned_config(cfg)
    assert not [k for k in pinned if k.startswith("deploy.variants")]
    cfg["deploy"]["variants"] = {"measured": "measured.yaml", "waived": [OUTPUT]}
    assert ro.pinned_config(cfg) == pinned
    assert ro.cfg_get({}, "deploy.variants.waived") == []
    assert ro.cfg_get({}, "deploy.variants.measured") is None
