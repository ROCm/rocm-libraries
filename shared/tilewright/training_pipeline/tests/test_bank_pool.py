# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
import zlib
from pathlib import Path

import msgpack
import pytest

import check_pick_log as cpl
from bench_fakes import solution
from lib import bank_pool as bp
from lib import dat
from lib import evaluate as ev
from lib.planner import Catalog

PREFIX = (
    "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk"
)
STEM = f"{PREFIX}_ID75a0_gfx950"
CU256 = f"{PREFIX}_CU256_ID75a0_gfx950"
CU304 = f"{PREFIX}_CU304_ID75a0_gfx950"
A3A2 = f"{PREFIX}_ID75a3-75a2_gfx950"
A8 = f"{PREFIX}_ID75a8_gfx950"
OTHER_TYPE = (
    "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk"
    "_CU256_ID75a0_gfx950"
)


def amdgpu(*terms):
    return {"type": "AMDGPU", "value": {"type": "And", "value": list(terms)}}


def proc(name="gfx950"):
    return {"type": "Processor", "value": name}


def chip(value):
    return {"type": "PciChipId", "value": value}


def cus(value):
    return {"type": "CUCount", "value": value}


def sol(index, hw):
    s = solution(index, index % 7, (64 * (1 + index % 4), 64, 64))
    s["hardwarePredicate"] = hw
    return s


def equality(*indices):
    return {
        "predicate": {"type": "EqualityMatching"},
        "library": {
            "type": "Matching",
            "table": [{"key": [1, i, 1, 1], "index": i, "speed": 0.0} for i in indices],
        },
    }


def problem(*rows):
    return {"type": "Problem", "rows": list(rows)}


def write(path, solutions, library):
    raw = msgpack.packb({"solutions": solutions, "library": library})
    path.write_bytes(zlib.compress(raw) if path.name.endswith(".zlib") else raw)


A0 = amdgpu(proc(), chip(0x75A0))
A0_256 = amdgpu(proc(), cus(256), chip(0x75A0))


@pytest.fixture
def libdir(tmp_path):
    """Files of one problem type plus one of another type:

    STEM    Prediction table [30, 10, 20]; nested Equality row {20, 41}
    CU256   nested Equality row {44, 40, 10, 41}, a GridBased row {50}
    CU304   Equality row {80} for 304 CUs
    A3A2    Equality row {60, 61} for chip 0x75a3 or 0x75a2
    A8      Equality row {70} for chip 0x75a8
    OTHER   Equality row {90} accepted everywhere (another layout)
    """
    d = tmp_path / "lib"
    d.mkdir()
    write(
        d / f"{STEM}.dat",
        [sol(i, A0) for i in (10, 20, 30, 41)],
        problem(
            {
                "predicate": {"type": "PredictionMatching"},
                "library": {"type": "Prediction", "table": [30, 10, 20]},
            },
            {"predicate": {"type": "TruePred"}, "library": problem(equality(20, 41))},
        ),
    )
    write(
        d / f"{CU256}.dat.zlib",
        [sol(i, A0_256) for i in (44, 40, 10, 41, 50)],
        problem(
            {
                "predicate": {"type": "TruePred"},
                "library": problem(equality(44, 40, 10, 41)),
            },
            {
                "predicate": {"type": "GridBasedMatching"},
                "library": {"type": "Matching", "table": [{"key": [1], "index": 50}]},
            },
        ),
    )
    write(
        d / f"{CU304}.dat",
        [sol(80, amdgpu(proc(), cus(304), chip(0x75A0)))],
        problem(equality(80)),
    )
    either = {"type": "Or", "value": [chip(0x75A3), chip(0x75A2)]}
    write(
        d / f"{A3A2}.dat",
        [sol(i, amdgpu(proc(), either)) for i in (61, 60)],
        problem(equality(61, 60)),
    )
    write(
        d / f"{A8}.dat", [sol(70, amdgpu(proc(), chip(0x75A8)))], problem(equality(70))
    )
    write(
        d / f"{OTHER_TYPE}.dat", [sol(90, {"type": "TruePred"})], problem(equality(90))
    )
    (d / "TensileLibrary_lazy_gfx950.dat").write_bytes(msgpack.packb({}))
    return d


def device(chip_id=0x75A0, cu_count=256, processor="gfx950"):
    return bp.BankDevice(processor=processor, cu_count=cu_count, chip_id=chip_id)


@pytest.mark.parametrize(
    "stem, key",
    [
        (STEM, f"{PREFIX}_gfx950"),
        (CU256, f"{PREFIX}_gfx950"),
        (A3A2, f"{PREFIX}_gfx950"),
        (f"{PREFIX}_CU256_gfx950", f"{PREFIX}_gfx950"),
        (f"{PREFIX}_gfx950", f"{PREFIX}_gfx950"),
        (f"{PREFIX}_ID-_gfx1250", f"{PREFIX}_gfx1250"),
        ("no_arch", "no_arch"),
        ("plain", "plain"),
    ],
)
def test_problem_type_key(stem, key):
    assert bp.problem_type_key(stem) == key


def test_same_type_stems(libdir):
    assert bp.same_type_stems(libdir, STEM) == sorted([STEM, CU256, CU304, A3A2, A8])


@pytest.mark.parametrize(
    "gpu, sol_chip, ok",
    [
        (0x75A0, 0x75A0, True),
        (0x75B0, 0x75A0, True),
        (0x75A2, 0x75A0, True),
        (0x75B2, 0x75A0, True),
        (0x75A3, 0x75A0, True),
        (0x75B3, 0x75A0, True),
        (0x75A8, 0x75A0, True),
        (0x75B8, 0x75A0, True),
        (0x75A0, 0x75A3, False),
        (0x75A3, 0x75A2, False),
        (0x75B0, 0x75B3, False),
        (None, 0x75A0, False),
    ],
)
def test_chip_id_fallbacks(gpu, sol_chip, ok):
    assert bp.chip_id_accepts(device(gpu), sol_chip) is ok
    assert bp.hardware_accepts(amdgpu(proc(), chip(sol_chip)), device(gpu)) is ok


def test_chip_id_predicates_need_a_chip_id_processor():
    gfx1250 = device(0x75A0, processor="gfx1250")
    assert not bp.chip_id_accepts(gfx1250, 0x75A0)
    assert bp.hardware_accepts(amdgpu(proc("gfx1250")), gfx1250)


def test_hardware_predicate_terms():
    d = device()
    assert bp.hardware_accepts(None, d)
    assert bp.hardware_accepts({"type": "TruePred"}, d)
    assert not bp.hardware_accepts({"type": "FalsePred"}, d)
    assert not bp.hardware_accepts(amdgpu(proc("gfx942")), d)
    assert not bp.hardware_accepts(amdgpu(cus(304)), d)
    assert bp.hardware_accepts(
        {"type": "AMDGPU", "value": {"type": "Not", "value": cus(304)}}, d
    )
    assert bp.hardware_accepts(
        {"type": "Not", "value": {"type": "AMDGPU", "value": cus(304)}}, d
    )
    target = {"type": "AMDGPU", "value": {"type": "TargetProcessor", "value": ""}}
    for value, ok in (("gfx950", True), ("gfx900", True), ("gfx942", False)):
        target["value"]["value"] = value
        assert bp.hardware_accepts(target, d) is ok, value
    target["value"]["value"] = "gfx1250"
    assert not bp.hardware_accepts(target, d)
    assert bp.runs_kernel_targeting("gfx90a", "gfx900")
    assert not bp.runs_kernel_targeting("gfx803", "gfx900")


@pytest.mark.parametrize(
    "pred, message",
    [
        ({"type": "Hostname", "value": "x"}, "unknown hardware predicate"),
        ({"type": "AMDGPU", "value": {"type": "XCCCount"}}, "unknown AMDGPU predicate"),
        (amdgpu({"type": "TargetProcessor", "value": "gfxABC"}), "unknown processor"),
    ],
)
def test_unknown_predicates_raise(pred, message):
    with pytest.raises(ValueError, match=message):
        bp.hardware_accepts(pred, device())


def test_equality_indices_walk_nested_problem_libraries():
    lib = problem(
        equality(3, 1),
        {"predicate": {"type": "TruePred"}, "library": problem(equality(7))},
        {
            "predicate": {"type": "EqualityMatching"},
            "library": {"type": "Matching", "table": [{"value": {"index": 9}}]},
        },
        {
            "predicate": {"type": "EqualityMatching"},
            "library": {"type": "Single", "index": 11},
        },
        {
            "predicate": {"type": "GridBasedMatching"},
            "library": {"type": "Matching", "table": [{"index": 5}]},
        },
    )
    assert bp.equality_indices(lib) == {1, 3, 7, 9, 11}
    assert bp.equality_indices({"type": "Prediction", "table": [1]}) == set()
    assert bp.equality_indices(None) == set()
    with pytest.raises(ValueError, match="holds a 'Prediction' library"):
        bp.equality_indices(
            problem(
                {
                    "predicate": {"type": "EqualityMatching"},
                    "library": {"type": "Prediction", "table": [1]},
                }
            )
        )


@pytest.mark.parametrize(
    "dev, bank, stems",
    [
        (device(0x75A0), [40, 41, 44], [CU256]),
        (device(0x75B8), [40, 41, 44], [CU256]),
        (device(0x75A3), [40, 41, 44, 60, 61], [A3A2, CU256]),
        (device(0x75A8, cu_count=304), [41, 70, 80], [A8, CU304, STEM]),
        (device(0x75A2, cu_count=128), [41, 60, 61], [A3A2, STEM]),
        (device(None), [], []),
        (device(0x75A0, processor="gfx942"), [], []),
    ],
)
def test_merged_pool(libdir, dev, bank, stems):
    pool = bp.library_pool(libdir, STEM, dev)
    assert [k["sol_idx_global"] for k in pool.table] == [30, 10, 20]
    assert [k["sol_idx_global"] for k in pool.bank.kernels] == bank
    assert pool.indices == [30, 10, 20] + bank
    assert pool.bank.stems == sorted(stems)
    assert pool.kernels[3:] == [dat.kernel_dat_info(sol(i, A0)) for i in bank]


def test_a_solution_in_two_files_counts_for_the_first_file(libdir):
    pool = bp.library_pool(libdir, STEM, device())
    assert pool.indices.count(41) == 1
    assert pool.bank.stems == [CU256]
    only_stem = bp.library_pool(libdir, STEM, device(), [STEM])
    assert only_stem.indices == [30, 10, 20, 41]
    assert only_stem.bank.stems == [STEM]


def test_explicit_stems_apply_the_same_rule(libdir):
    pool = bp.library_pool(libdir, STEM, device(), [CU256])
    assert pool.indices == [30, 10, 20, 40, 41, 44]
    assert bp.library_pool(libdir, STEM, device(), [A8]).indices == [30, 10, 20]
    assert bp.library_pool(libdir, STEM).indices == [30, 10, 20]


def test_catalog_and_evaluation_pools_follow_the_merged_order(libdir):
    pool = bp.library_pool(libdir, STEM, device(0x75A3))
    cat = Catalog.build(pool.table, pool.bank.kernels)
    assert cat.pool_ids + cat.bank_ids == pool.indices
    cfg = config(bank="auto")
    assert [
        k["sol_idx_global"] for k in ev.library_pool(libdir, STEM, cfg)
    ] == bp.config_pool(cfg, libdir, STEM).indices
    assert [k["sol_idx_global"] for k in ev.library_pool(libdir, STEM)] == [30, 10, 20]


def config(enabled=True, chip_id=0x75A0, **planner):
    hardware = {"n_cu": 256, "lds_bytes": 160 * 1024, "l2_bytes": 4 << 20}
    if chip_id is not None:
        hardware["pci_chip_id"] = chip_id
    return {
        "arch": "gfx950",
        "hardware": hardware,
        "planner": {"enabled": enabled, **planner},
    }


def test_bank_source():
    assert not bp.bank_source(None).enabled
    assert not bp.bank_source(config()).enabled
    assert not bp.bank_source(config(enabled=False, bank="auto")).enabled
    assert bp.bank_source(config(bank="auto")) == bp.BankSource(auto=True)
    both = config(bank="auto", bank_library_stems=[CU256])
    assert bp.bank_source(both) == bp.BankSource(stems=(CU256,))
    with pytest.raises(ValueError, match="planner.bank"):
        bp.bank_source(config(bank="all"))


def test_config_pool_and_bench_stems(libdir):
    auto = config(bank="auto")
    assert bp.config_pool(auto, libdir, STEM).indices == [30, 10, 20, 40, 41, 44]
    assert bp.bank_library_stems(auto, libdir, STEM) == [CU256]
    a8 = config(bank="auto", chip_id=0x75A8)
    assert bp.bank_library_stems(a8, libdir, STEM) == [CU256, A8]
    explicit = config(bank="auto", bank_library_stems=[A8])
    assert bp.config_pool(explicit, libdir, STEM).indices == [30, 10, 20]
    assert bp.bank_library_stems(explicit, libdir, STEM) == [A8]
    off = config()
    assert bp.config_pool(off, libdir, STEM).indices == [30, 10, 20]
    assert bp.bank_library_stems(off, libdir, STEM) == []


def test_the_device_comes_from_the_config_when_gpus_are_hidden(tmp_path):
    absent = tmp_path / "absent"
    dev = bp.bank_device(config(), 0, absent)
    assert dev == bp.BankDevice("gfx950", 256, 0x75A0)
    with pytest.raises(RuntimeError, match="hardware.pci_chip_id"):
        bp.bank_device(config(chip_id=None), 0, absent)
    strict = dict(config(chip_id=None), arch="gfx1250-strict")
    strict["hardware"]["n_cu"] = 7
    assert bp.bank_device(strict, 0, absent) == bp.BankDevice("gfx1250", 7)
    with pytest.raises(ValueError, match="no arch"):
        bp.bank_device({**config(), "arch": ""}, 0, absent)


# ── pick log checker ─────────────────────────────────────────────────────────


def pick(m, index, n_configs=6):
    return (
        f"[TILEWRIGHT_PICK] m={m} n=64 k=32 b=1 tA=T tB=N leaf=L top1_sig=(mt_m=64,"
        f"mt_n=64,mt_k=64,mi_m=16,mi_n=16,mi_k=32,cha=0,chb=0) top1_score=0.5 "
        f"top1_index={index} n_configs={n_configs}"
    )


def ran(index):
    return f"    --Solution index: {index}"


def test_check_picks():
    pool = [30, 10, 20, 40, 41, 44]
    lines = [pick(1, 0), "Solution selection time: 1 us", ran(30)]
    lines += [pick(2, 4), ran(41), ran(41)]
    lines += [pick(3, 2), ran(30), pick(4, 9), ran(44)]
    out = cpl.check_picks(lines, pool)
    assert [p.top1_index for p in out.picks] == [0, 4, 2, 9]
    assert out.without_pick == 1
    assert out.n_configs == [6]
    assert [(p.shape[0], named) for p, named in out.mismatches] == [(3, 20), (4, None)]
    assert not out.ok


def test_check_pick_log_main(libdir, tmp_path, capsys):
    log = tmp_path / "bench.log"
    log.write_text("\n".join([pick(1, 0), ran(30), pick(2, 5), ran(44)]) + "\n")
    base = [str(log), "--library-dir", str(libdir), "--library-stem", STEM]
    dev = ["--processor", "gfx950", "--cu-count", "256", "--chip-id", "0x75a0"]
    assert cpl.main(base + dev) == 0
    assert "mismatches: 0" in capsys.readouterr().out
    assert cpl.main(base + dev + ["--bank-stem", A8]) == 1
    assert "n_configs differs" in capsys.readouterr().out
    cfg = tmp_path / "cfg.yaml"
    import yaml

    cfg.write_text(yaml.safe_dump(config(bank="auto")))
    assert cpl.main(base + ["--config-yaml", str(cfg)]) == 0
    log.write_text("\n".join([pick(1, 1), ran(30)]) + "\n")
    assert cpl.main(base + dev) == 1
    assert "pool names 10, the bench ran 30" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        cpl.main(base + ["--processor", "gfx950", "--cu-count", "x"])
    with pytest.raises(SystemExit):
        cpl.main(base)


def test_check_pick_log_leaves_out_what_the_runtime_left_out(libdir, tmp_path, capsys):
    """The runtime ranks the pool without the kernels of the sidecar next to
    the model, so the positions of its picks count without them too."""
    from lib import correctness as cx

    names = {
        k["sol_idx_global"]: k["kernel_name"]
        for k in bp.library_pool(libdir, STEM, device()).kernels
    }
    log = tmp_path / "bench.log"
    log.write_text("\n".join([pick(1, 0, 5), ran(30), pick(2, 4, 5), ran(44)]) + "\n")
    base = [str(log), "--library-dir", str(libdir), "--library-stem", STEM]
    dev = ["--processor", "gfx950", "--cu-count", "256", "--chip-id", "0x75a0"]
    assert cpl.main(base + dev) == 1
    capsys.readouterr()
    (libdir / cx.sidecar_name(STEM)).write_text(cx.sidecar_text(STEM, [names[41]]))
    assert cpl.main(base + dev) == 0
    out = capsys.readouterr().out
    assert "pool: 5 kernels" in out and "1 kernel(s) left out" in out
    nothing = tmp_path / "none.tsv"
    nothing.write_text("# no kernel\n")
    assert cpl.main(base + dev + ["--blocklist", str(nothing)]) == 1
    assert "pool: 6 kernels" in capsys.readouterr().out


def test_check_pick_log_reads_the_guards_of_a_model_trained_with_them_only(
    libdir, tmp_path, capsys
):
    """The runtime reads the guards sidecar next to the model only for a
    model trained with kernel guards, which keeps a guarded kernel in its
    pool; it serves any other model with the blocklist sidecar's lines
    whole. The check reads the guards next to the model by the same rule,
    and any given with --guards."""
    from test_v3 import LARGE, v3_bundle, write

    from lib import correctness as cx
    from lib import mlrec
    from lib import tilewright_index as twi

    names = {
        k["sol_idx_global"]: k["kernel_name"]
        for k in bp.library_pool(libdir, STEM, device()).kernels
    }
    (libdir / cx.sidecar_name(STEM)).write_text(cx.sidecar_text(STEM, [names[41]]))
    guards = libdir / cx.guards_name(STEM)
    guards.write_text(cx.guards_text(STEM, {names[41]: (cx.Guard("M", 4),)}))
    model = write(v3_bundle(labels=[LARGE]))
    (libdir / "m.tilewright.bin").write_bytes(model)
    (libdir / twi.INDEX_NAME).write_text(f"{STEM}\tm.tilewright.bin\n")
    log = tmp_path / "bench.log"
    base = [str(log), "--library-dir", str(libdir), "--library-stem", STEM]
    dev = ["--processor", "gfx950", "--cu-count", "256", "--chip-id", "0x75a0"]
    whole = [pick(1, 0, 5), ran(30), pick(2, 4, 5), ran(44)]
    guarded = [pick(8, 4), ran(41), pick(2, 4), ran(44), pick(9, 5), ran(44)]
    log.write_text("\n".join(whole) + "\n")
    assert not cpl.served_with_guards(libdir, STEM, 0x75A0)
    assert cpl.main(base + dev) == 0
    assert "pool: 5 kernels" in capsys.readouterr().out
    log.write_text("\n".join(guarded) + "\n")
    assert cpl.main(base + dev) == 1
    capsys.readouterr()
    assert cpl.main(base + dev + ["--guards", str(guards)]) == 0
    assert "passed over below its guard: 1" in capsys.readouterr().out
    (libdir / "m.tilewright.bin").write_bytes(mlrec.with_kernel_guards(model))
    assert cpl.served_with_guards(libdir, STEM, 0x75A0)
    assert cpl.main(base + dev) == 0
    out = capsys.readouterr().out
    assert "pool: 6 kernels" in out and "passed over below its guard: 1" in out
    log.write_text("\n".join(whole) + "\n")
    assert cpl.main(base + dev) == 1
    (libdir / twi.INDEX_NAME).write_text(f"{STEM}\tgone.tilewright.bin\n")
    assert not cpl.served_with_guards(libdir, STEM, 0x75A0)
    assert cpl.main(base + dev) == 0


# ── the live library ─────────────────────────────────────────────────────────

LIVE_DIR = os.environ.get("TILEWRIGHT_GFX950_LIBRARY_DIR")


@pytest.mark.skipif(
    not LIVE_DIR,
    reason="set TILEWRIGHT_GFX950_LIBRARY_DIR to a gfx950 Tensile library directory",
)
def test_bbs_tn_pool_on_a_live_library():
    """bf16 TN on 0x75a0 with 256 CUs: the rule gives the pool the planner
    built from the explicit `_CU256_` bank (945 table + 2,371 bank kernels)."""
    libdir = Path(LIVE_DIR)
    cfg = {
        "arch": "gfx950",
        "hardware": {"n_cu": 256, "pci_chip_id": 0x75A0},
        "planner": {"enabled": True, "bank": "auto"},
    }
    cfg["hardware"].update(lds_bytes=160 * 1024, l2_bytes=4 << 20)
    pool = bp.config_pool(cfg, libdir, STEM)
    assert (len(pool.table), len(pool.bank.kernels)) == (945, 2371)
    assert pool.bank.stems == [CU256]
    current = Catalog.build(
        dat.load_library_kernels(libdir, STEM), dat.load_library_kernels(libdir, CU256)
    )
    assert pool.indices == current.pool_ids + current.bank_ids
    assert bp.bank_library_stems(cfg, libdir, STEM) == [CU256]
