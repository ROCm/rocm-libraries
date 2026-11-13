# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import os
from pathlib import Path

import msgpack
import pytest

from bench_fakes import solution
from lib import bank_pool as bp
from lib import correctness as cx
from lib import library_pool as lp
from lib.kernel_identity import library_identity

PREFIX = (
    "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk"
)
STEM = f"{PREFIX}_ID75a0_gfx950"
CU256 = f"{PREFIX}_CU256_ID75a0_gfx950"
A3A2 = f"{PREFIX}_ID75a3-75a2_gfx950"
ROWS = f"{PREFIX}_gfx950"
NEAR = "TensileLibrary_BB_BB_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_gfx950"
OTHER_LAYOUT = (
    "TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Ailk_Bljk_Cijk_Dijk"
    "_ID75a0_gfx950"
)
HARDWARE = {"n_cu": 256, "lds_bytes": 160 * 1024, "l2_bytes": 4 << 20}


def amdgpu(*terms):
    return {"type": "AMDGPU", "value": {"type": "And", "value": list(terms)}}


def proc(name="gfx950"):
    return {"type": "Processor", "value": name}


def chip(value):
    return {"type": "PciChipId", "value": value}


A0 = amdgpu(proc(), chip(0x75A0))
A0_256 = amdgpu(proc(), {"type": "CUCount", "value": 256}, chip(0x75A0))
A3_OR_A2 = amdgpu(proc(), {"type": "Or", "value": [chip(0x75A3), chip(0x75A2)]})


def name_of(index):
    return f"Cijk_Alik_Bljk_BBS_BH_Bias_HA_S_SAV_UserArgs_MT{index}_SN_TEST"


def sol(index, hw, name=None):
    s = solution(index, index % 7, (64 * (1 + index % 4), 64, 64))
    s["name"] = name or name_of(index)
    s["hardwarePredicate"] = hw
    return s


def matching(predicate, indices, distance="Equality"):
    table = [{"key": [1, i, 1, 1], "index": i, "speed": 0.0} for i in indices]
    return {
        "predicate": {"type": predicate},
        "library": {"type": "Matching", "distance": distance, "table": table},
    }


def equality(*indices):
    return matching("EqualityMatching", indices)


def gridbased(*indices):
    return matching("GridBasedMatching", indices, "GridBased")


def freesize(*indices):
    return {
        "predicate": {"type": "FreeSizeMatching"},
        "library": {"type": "FreeSize", "table": [{"index": i} for i in indices]},
    }


def prediction(*indices):
    return {
        "predicate": {"type": "PredictionMatching"},
        "library": {"type": "Prediction", "table": list(indices)},
    }


def problem(*rows):
    return {"type": "Problem", "rows": list(rows)}


def nested(*rows):
    return {"predicate": {"type": "TruePred"}, "library": problem(*rows)}


def write(libdir, stem, solutions, library):
    path = Path(libdir) / f"{stem}.dat"
    path.write_bytes(msgpack.packb({"solutions": solutions, "library": library}))
    return path


@pytest.fixture
def libdir(tmp_path):
    """Files of the host's problem type, one near it and one of another
    layout:

    STEM          Prediction table [30, 10, 20]; nested Equality row {20, 41}
    CU256         nested Equality row {44, 40, 10, 41} for 0x75a0 at 256 CUs
    A3A2          Equality row {61, 60} for chip 0x75a3 or 0x75a2
    NEAR          Equality row {90}: the data types and layout, no epilogue
    OTHER_LAYOUT  Equality row {95}
    """
    d = tmp_path / "lib"
    d.mkdir()
    write(
        d,
        STEM,
        [sol(i, A0) for i in (10, 20, 30, 41)],
        problem(prediction(30, 10, 20), nested(equality(20, 41))),
    )
    write(
        d,
        CU256,
        [sol(i, A0_256) for i in (44, 40, 10, 41)],
        problem(nested(equality(44, 40, 10, 41))),
    )
    write(d, A3A2, [sol(i, A3_OR_A2) for i in (61, 60)], problem(equality(61, 60)))
    write(d, NEAR, [sol(90, None)], problem(equality(90)))
    write(d, OTHER_LAYOUT, [sol(95, None)], problem(equality(95)))
    (d / "TensileLibrary_lazy_gfx950.dat").write_bytes(msgpack.packb({}))
    return d


def device(chip_id=0x75A0, cu_count=256):
    return lp.PoolDevice("gfx950", cu_count, chip_id)


def config(chip_id=0x75A0, **sections):
    hardware = dict(HARDWARE, pci_chip_id=chip_id)
    return {"arch": "gfx950", "hardware": hardware, **sections}


def new_form(kind, **library):
    return config(library={"kind": kind, **library}, train={"catalog": "v3"})


def ids(kernels):
    return [k["sol_idx_global"] for k in kernels]


# ── configuration ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "cfg, source",
    [
        (None, lp.PoolSource(legacy=True)),
        ({}, lp.PoolSource(legacy=True)),
        ({"planner": {"bank": "auto"}}, lp.PoolSource(legacy=True)),
        (
            {"planner": {"enabled": True, "bank": "auto"}},
            lp.PoolSource(("origami", "bok"), ("equality",), legacy=True),
        ),
        (
            {
                "planner": {
                    "enabled": True,
                    "bank": "auto",
                    "bank_library_stems": [A3A2],
                }
            },
            lp.PoolSource(("origami", "bok"), ("equality",), (A3A2,), legacy=True),
        ),
        ({"planner": {"enabled": True, "bank": "none"}}, lp.PoolSource(legacy=True)),
        ({"library": {}}, lp.PoolSource(legacy=True)),
        ({"library": {"kind": "origami"}}, lp.PoolSource()),
        ({"library": {"bok_rows": "equality"}}, lp.PoolSource(bok_rows=("equality",))),
        (
            {"library": {"kind": ["bok", "origami"], "bok_rows": ["gridbased"]}},
            lp.PoolSource(("bok", "origami"), ("gridbased",)),
        ),
        (
            {"library": {"kind": "bok"}, "planner": {"enabled": True}},
            lp.PoolSource(("bok",), ("freesize", "equality")),
        ),
    ],
)
def test_pool_sources(cfg, source):
    assert lp.pool_source(cfg) == source


@pytest.mark.parametrize(
    "cfg, message",
    [
        ({"library": {"kind": "all"}}, "library.kind must be one of"),
        ({"library": {"kind": []}}, "library.kind must be one of"),
        ({"library": {"kind": ["bok", "bok"]}}, "distinct"),
        ({"library": {"bok_rows": ["range"]}}, "library.bok_rows must be one of"),
        ({"planner": {"enabled": True, "bank": "all"}}, "planner.bank must be one of"),
        (
            {"library": {"kind": "origami"}, "planner": {"bank": "none"}},
            "library and planner.bank both name the kernel pool",
        ),
        (
            {"library": {"kind": "bok"}, "planner": {"bank_library_stems": []}},
            "delete planner.bank_library_stems",
        ),
    ],
)
def test_pool_source_errors(cfg, message):
    with pytest.raises(ValueError, match=message):
        lp.pool_source(cfg)
    errors, _ = lp.config_problems(cfg)
    assert any(message in e for e in errors)


def test_config_problems():
    v3 = {"catalog": "v3"}
    assert lp.config_problems({"planner": {"enabled": True, "bank": "auto"}}) == (
        [],
        [],
    )
    assert lp.config_problems({"library": {"kind": "origami"}}) == ([], [])
    errors, warnings = lp.config_problems({"library": {"kind": ["origami", "bok"]}})
    assert errors == [
        "library.kind bok needs train.catalog v3: only a v3 model tells TensileLite "
        "to rank the bank it was trained on"
    ]
    no_gate = {"catalog": "v3", "trained_key_gate": False}
    errors, _ = lp.config_problems({"library": {"kind": "bok"}, "train": no_gate})
    assert len(errors) == 1 and "train.trained_key_gate" in errors[0]
    _, warnings = lp.config_problems({"library": {"kind": "bok"}, "train": v3})
    assert len(warnings) == 1 and "cannot be served yet" in warnings[0]
    _, warnings = lp.config_problems(
        {"library": {"kind": ["bok", "origami"]}, "train": v3}
    )
    assert len(warnings) == 1 and "lists bok before origami" in warnings[0]
    _, warnings = lp.config_problems(
        {"library": {"kind": ["origami", "bok"], "bok_rows": "gridbased"}, "train": v3}
    )
    (warning,) = warnings
    assert warning.startswith(
        "library.bok_rows gridbased: a bridge for a bank of kernels built as "
        "GridBased rows, which TensileLite does not rank with the model."
    )
    assert "Equality, Range, Prediction, GridBased, FreeSize" in warning
    assert "sits behind that table's row" in warning
    assert "TENSILE_TILEWRIGHT_BANK takes Equality rows only" in warning
    assert "before tilewright ranks" not in warning
    assert lp.config_problems({"library": {"bok_rows": ["equality"]}}) == (
        [],
        ["library.bok_rows is ignored: library.kind has no bok"],
    )


def test_stems_and_problem_types():
    assert lp.problem_type_key(CU256) == lp.problem_type_key(A3A2) == f"{PREFIX}_gfx950"
    assert lp.type_tail(NEAR) == lp.type_tail(STEM) != lp.type_tail(OTHER_LAYOUT)
    assert lp.type_tail("no_type_gfx950") is None
    assert [lp.is_cu_file(s) for s in (CU256, STEM, A3A2, ROWS)] == [
        True,
        False,
        False,
        False,
    ]
    assert lp.is_cu_file(f"{PREFIX}_CU64_gfx1250") and not lp.is_cu_file("plain")


# ── resolving a pool ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "dev",
    [device(), device(0x75A3), device(0x75A8, 128), device(None)],
    ids=["75a0", "75a3", "75a8-128cu", "no-chip-id"],
)
def test_the_legacy_bank_and_library_kind_are_one_pool(libdir, dev):
    """`planner.bank: auto` is `library.kind [origami, bok]` with bok rows
    equality: same kernels, order and files on every device."""
    legacy = bp.library_pool(libdir, STEM, dev)
    new = lp.resolve(libdir, STEM, lp.PoolSource(("origami", "bok")), dev)
    assert ids(new.table) == ids(legacy.table) == [30, 10, 20]
    assert ids(new.bank.kernels) == ids(legacy.bank.kernels)
    assert new.bank.stems == legacy.bank.stems
    assert new.kernels == legacy.kernels


def test_config_forms_resolve_alike(libdir):
    legacy = config(planner={"enabled": True, "bank": "auto"})
    explicit = config(planner={"enabled": True, "bank_library_stems": [CU256]})
    new = new_form(["origami", "bok"])
    pools = [bp.config_pool(c, libdir, STEM) for c in (legacy, explicit, new)]
    assert [p.indices for p in pools] == [[30, 10, 20, 40, 41, 44]] * 3
    assert {tuple(p.bank.stems) for p in pools} == {(CU256,)}
    for c in (legacy, explicit, new):
        assert bp.bank_source(c).enabled
        assert bp.bank_library_stems(c, libdir, STEM) == [CU256]
        assert bp.config_identity(c, libdir, STEM).digest == (
            library_identity(libdir, [STEM, CU256]).digest
        )
    assert lp.pool_source(legacy).text() == (
        "planner.bank auto: [origami, bok] with bok rows equality"
    )
    assert lp.pool_source(new).text() == (
        "library.kind [origami, bok] with bok rows freesize, equality"
    )


def test_the_planner_gate_holds_for_the_legacy_keys_only(libdir):
    off = {"enabled": False, "bank": "auto"}
    assert bp.config_pool(config(planner=off), libdir, STEM).indices == [30, 10, 20]
    new = dict(new_form(["origami", "bok"]), planner={"enabled": False})
    assert bp.config_pool(new, libdir, STEM).indices == [30, 10, 20, 40, 41, 44]


def test_kinds_are_taken_in_order_and_the_first_keeps_an_index(libdir):
    pool = bp.config_pool(new_form(["bok", "origami"]), libdir, STEM)
    assert [p.kind for p in pool.parts] == ["bok", "origami"]
    assert ids(pool.table) == [10, 20, 40, 41, 44]
    assert ids(pool.bank.kernels) == [30] and pool.bank.stems == [STEM]
    assert pool.parts[1].duplicates == 2
    assert pool.indices == ids(pool.parts[0].kernels) + ids(pool.parts[1].kernels)
    assert bp.bank_source(new_form(["bok", "origami"])).enabled


def test_a_bank_of_kernels_alone(libdir):
    cfg = new_form("bok")
    pool = bp.config_pool(cfg, libdir, STEM)
    assert ids(pool.table) == [10, 20, 40, 41, 44] and not pool.bank.kernels
    assert pool.files == [CU256, STEM]
    assert bp.bank_library_stems(cfg, libdir, STEM) == [CU256]
    assert not bp.bank_source(cfg).enabled
    origami = bp.config_pool(new_form("origami"), libdir, STEM)
    assert origami.indices == [30, 10, 20] and origami.device is None


def test_bok_rows_choose_the_row_kinds(libdir):
    write(
        libdir,
        ROWS,
        [sol(i, amdgpu(proc())) for i in (70, 71, 72, 73)],
        problem(freesize(71, 70), gridbased(72), equality(73)),
    )

    def bank(cfg):
        return bp.config_pool(cfg, libdir, STEM).bank

    assert ids(bank(new_form(["origami", "bok"])).kernels) == [40, 41, 44, 70, 71, 73]
    rows = new_form(["origami", "bok"], bok_rows=["gridbased"])
    assert ids(bank(rows).kernels) == [72] and bank(rows).stems == [ROWS]
    legacy = config(planner={"enabled": True, "bank": "auto"})
    assert ids(bank(legacy).kernels) == [40, 41, 44, 73]
    pool = bp.config_pool(new_form(["origami", "bok"]), libdir, STEM)
    assert pool.parts[1].rows == {"equality": 4, "freesize": 2}
    warns = [line for level, line in lp.describe(pool) if level == "warn"]
    assert warns == [
        "bok: 2 kernels come from freesize rows, which TensileLite's bank does not "
        "take: it ranks a smaller pool than the model's"
    ]


def test_hardware_predicates_choose_the_files(libdir):
    both = lp.PoolSource(("origami", "bok"))
    pool = lp.resolve(libdir, STEM, both, device())
    assert pool.parts[1].idle == {
        STEM: "the pool holds every kernel of it already",
        A3A2: "the device's hardware predicate rejects its 2 kernels",
    }
    assert pool.parts[1].duplicates == 3
    pool = lp.resolve(libdir, STEM, both, device(0x75A3))
    assert ids(pool.bank.kernels) == [40, 41, 44, 60, 61]
    assert pool.bank.stems == [CU256, A3A2]
    pool = lp.resolve(libdir, STEM, both, device(cu_count=128))
    assert ids(pool.bank.kernels) == [41]
    assert pool.parts[1].idle[CU256] == (
        "the device's hardware predicate rejects the 3 kernels of it the pool does "
        "not hold already"
    )


def cu(count):
    return {"type": "CUCount", "value": count}


@pytest.mark.parametrize(
    "pred, specificity, of_device",
    [
        (None, (0, 0), (0, 0)),
        (amdgpu(proc()), (0, 0), (0, 0)),
        (amdgpu(proc("gfx1250"), cu(96)), (0, 96), (0, 96)),
        (A0, (1, 0), (2, 0)),
        (A0_256, (1, 256), (2, 256)),
        (A3_OR_A2, (1, 0), (1, 0)),
    ],
)
def test_predicate_specificity_orders_the_hardware_rows(pred, specificity, of_device):
    """A row naming the device's own chip id comes first, then one naming
    the chip id it falls back to, then the higher CU count."""
    assert lp.predicate_specificity(pred) == specificity
    assert lp.predicate_specificity(pred, device()) == of_device


def test_a_cu_count_class_with_a_prediction_table_of_its_own_ranks_with_it(tmp_path):
    """A device of the CU count of a `_CU<n>_` file of the problem type that
    holds a Prediction table ranks with that table, ahead of the generic
    file's; a `_CU<n>_` file without one does not take the device's
    Prediction ranking."""
    d = tmp_path / "lib"
    d.mkdir()
    generic = f"{PREFIX}_gfx1250"
    cu96, cu192 = f"{PREFIX}_CU96_gfx1250", f"{PREFIX}_CU192_gfx1250"
    hw = amdgpu(proc("gfx1250"))
    hw96, hw192 = amdgpu(proc("gfx1250"), cu(96)), amdgpu(proc("gfx1250"), cu(192))
    write(d, generic, [sol(i, hw) for i in (1, 2)], problem(prediction(2, 1)))
    write(d, cu96, [sol(i, hw96) for i in (1, 2)], problem(prediction(2, 1)))
    write(d, cu192, [sol(5, hw192)], problem(equality(5)))
    tables = lp.prediction_predicates(d, generic)
    assert tables == {cu96: [hw96], generic: [hw]}
    assert lp.prediction_predicates(d, cu192) == tables

    def ahead(stem, cus):
        return lp.prediction_stems_ahead(tables, stem, lp.PoolDevice("gfx1250", cus))

    assert ahead(generic, 96) == [cu96]
    assert ahead(cu96, 96) == []
    for cus in (192, 256):
        assert ahead(generic, cus) == []
        assert ahead(cu96, cus) == [generic]
    assert ahead(cu192, 96) == []


def test_a_chip_with_a_prediction_table_of_its_own_ranks_with_it(libdir):
    """The gfx950 files of the fixture hold one Prediction table, so no device
    ranks with another; a file whose table names a chip id takes the devices
    of that chip ahead of a table of a chip they fall back to."""
    assert lp.prediction_predicates(libdir, STEM) == {STEM: [A0]}
    a8 = f"{PREFIX}_ID75a8_gfx950"
    hw = amdgpu(proc(), chip(0x75A8))
    write(libdir, a8, [sol(i, hw) for i in (70, 71)], problem(prediction(71, 70)))
    tables = lp.prediction_predicates(libdir, STEM)
    assert sorted(tables) == [STEM, a8]
    assert lp.prediction_stems_ahead(tables, STEM, device()) == []
    assert lp.prediction_stems_ahead(tables, STEM, device(0x75A8)) == [a8]
    assert lp.prediction_stems_ahead(tables, a8, device(0x75A8)) == []
    assert lp.prediction_stems_ahead(tables, a8, device()) == [STEM]
    assert lp.prediction_stems_ahead(tables, NEAR, device()) == []


def test_solution_names_never_merge_kernels(libdir):
    """TensileLite ranks every solution index: a solution that repeats the
    name of another stays in the pool, and the composition counts it."""
    write(
        libdir,
        f"{PREFIX}_ID75b0_gfx950",
        [sol(80, None, name_of(30)), sol(81, None, name_of(44))],
        problem(equality(81, 80)),
    )
    for cfg in (new_form(["origami", "bok"]), config(planner={"bank": "auto"})):
        cfg.setdefault("planner", {})["enabled"] = True
        pool = bp.config_pool(cfg, libdir, STEM)
        assert ids(pool.bank.kernels) == [40, 41, 44, 80, 81]
        s = lp.summary(pool)
        assert (s["repeated_names"], s["repeated_across_kinds"]) == (2, 1)
    lines = dict((line, level) for level, line in lp.describe(pool))
    repeats = (
        "2 kernels repeat the solution name of an earlier pool kernel under "
        "another solution index (1 that of another kind): TensileLite ranks each "
        "of them, so each is benched and ranked as a kernel of its own"
    )
    assert lines[repeats] == "warn"


def test_origami_needs_the_prediction_table(tmp_path):
    d = tmp_path / "lib"
    d.mkdir()
    write(d, STEM, [sol(i, A0) for i in (5, 3, 4)], problem(equality(4, 3)))
    with pytest.raises(ValueError, match="no Prediction table.*library.kind bok"):
        bp.config_pool(new_form("origami"), d, STEM)
    legacy = bp.config_pool(config(), d, STEM)
    assert legacy.indices == [5, 3, 4] and legacy.parts[0].rows == {"file": 3}
    assert any(
        level == "warn" and "has no Prediction table" in line
        for level, line in lp.describe(legacy)
    )
    assert bp.config_pool(new_form("bok"), d, STEM).indices == [3, 4]


def test_rows_are_read_by_solution_index_never_by_size(libdir, monkeypatch):
    """Only the solution indices of a row are read, never the problem sizes
    its table is keyed by."""

    class Sizeless(dict):
        def __getitem__(self, key):
            assert key != "key", "a row's problem sizes were read"
            return super().__getitem__(key)

        def get(self, key, default=None):
            assert key != "key", "a row's problem sizes were read"
            return super().get(key, default)

    def guarded(node):
        if isinstance(node, dict):
            out = {k: guarded(v) for k, v in node.items()}
            return Sizeless(out) if "key" in node else out
        if isinstance(node, list):
            return [guarded(v) for v in node]
        return node

    read = lp.read_tensile_logic
    monkeypatch.setattr(lp, "read_tensile_logic", lambda path: guarded(read(path)))
    pool = bp.config_pool(new_form(["origami", "bok"]), libdir, STEM)
    assert pool.indices == [30, 10, 20, 40, 41, 44]
    guarded_rows = guarded(problem(freesize(7), gridbased(8), equality(9)))
    assert lp.row_indices(guarded_rows, lp.ROW_PREDICATES) == {
        7: "freesize",
        8: "gridbased",
        9: "equality",
    }


@pytest.mark.parametrize(
    "library, message",
    [
        (
            problem(
                {
                    "predicate": {"type": "FreeSizeMatching"},
                    "library": {"type": "Prediction", "table": [1]},
                }
            ),
            "a FreeSize row holds a 'Prediction' library",
        ),
        (
            problem(
                {
                    "predicate": {"type": "FreeSizeMatching"},
                    "library": {"type": "FreeSize", "table": [{"speed": 1}]},
                }
            ),
            "a FreeSize table entry names no solution index",
        ),
    ],
)
def test_rows_the_pipeline_cannot_follow(library, message):
    with pytest.raises(ValueError, match=message):
        lp.row_indices(library, ("freesize",))


def test_solution_rows_name_the_row_of_every_index():
    range_row = {
        "predicate": {"type": "RangeMatching"},
        "library": {
            "type": "Matching",
            "table": [{"key": [1], "value": {"index": 11}}],
        },
    }
    library = problem(
        equality(5),
        nested(gridbased(6, 5), prediction(7, 8)),
        freesize(9),
        range_row,
        {"predicate": {"type": "TruePred"}, "library": {"type": "MLPClassification"}},
        {"predicate": {"type": "TruePred"}, "library": {"type": "Single", "index": 12}},
    )
    assert lp.solution_rows(library) == {
        5: "Equality",
        6: "GridBased",
        7: "Prediction",
        8: "Prediction",
        9: "FreeSize",
        11: "Range",
        12: "other",
    }
    assert lp.solution_rows(None) == {}
    assert lp.solution_rows({"type": "Matching", "table": []}) == {}


def test_the_rows_of_the_data_types_come_from_every_file_of_them(libdir):
    rows = lp.type_rows(libdir, STEM)
    assert rows[30] == rows[20] == (STEM, "Prediction")
    assert rows[44] == (CU256, "Equality") and rows[61] == (A3A2, "Equality")
    # An index several files hold is the first file's, in stem order.
    assert rows[10] == (CU256, "Equality")
    # The file of the data types and layout without the epilogue is read, the
    # file of another layout is not.
    assert rows[90] == (NEAR, "Equality") and 95 not in rows
    (libdir / f"{NEAR}.dat").write_bytes(b"\xc1")
    with pytest.raises(ValueError, match=NEAR):
        lp.type_rows(libdir, STEM)


SIZES = [
    {"type": "FreeSizeA", "index": 0},
    {"type": "FreeSizeB", "index": 0},
    {"type": "BatchSize", "index": 0},
    {"type": "BoundSize", "index": 0},
]


def test_equality_rows_answer_the_sizes_their_keys_hold():
    """An Equality row's entries carry the (M, N, batch, K) their keys hold
    under the row's properties, in any order; a size no property names is
    not compared, and a row of another kind answers every size."""
    keyed = {
        "predicate": {"type": "EqualityMatching"},
        "library": {
            "type": "Matching",
            "properties": [SIZES[3], SIZES[0], SIZES[1], SIZES[2]],
            "table": [
                {"key": [64, 256, 128, 1], "index": 3},
                {"key": [32, 8, 8, 2], "value": {"type": "Single", "index": 4}},
            ],
        },
    }
    partial = {
        "predicate": {"type": "EqualityMatching"},
        "library": {
            "type": "Matching",
            "properties": [SIZES[0], {"type": "AIGreaterThanEqual", "index": 0}],
            "table": [{"key": [512, 7], "index": 5}],
        },
    }
    entries = lp.row_entries(
        problem(keyed, nested(partial), equality(6), gridbased(7)), "file"
    )
    assert [(e.stem, e.kind, e.index, e.size) for e in entries] == [
        ("file", "Equality", 3, (256, 128, 1, 64)),
        ("file", "Equality", 4, (8, 8, 2, 32)),
        ("file", "Equality", 5, (512, None, None, None)),
        # No properties: no size of the key is known.
        ("file", "Equality", 6, (None, None, None, None)),
        ("file", "GridBased", 7, None),
    ]
    first, _, partly, unkeyed, grid = entries
    assert first.answers(256, 128, 1, 64) and not first.answers(256, 128, 2, 64)
    assert not first.answers_every_size()
    assert partly.answers(512, 1, 9, 3) and not partly.answers(511, 1, 9, 3)
    assert unkeyed.answers_every_size() and unkeyed.answers(1, 2, 3, 4)
    assert grid.answers_every_size() and grid.answers(1, 2, 3, 4)


def test_the_type_rows_hold_every_row_and_its_kernels(libdir):
    rows = lp.read_type_rows(libdir, STEM)
    assert rows.rows == lp.type_rows(libdir, STEM)
    # Every row of every file, an index in several of them included.
    assert {(e.stem, e.index) for e in rows.entries if e.index == 10} == {
        (STEM, 10),
        (CU256, 10),
    }
    assert {e.kind for e in rows.entries if e.stem == STEM} == {
        "Prediction",
        "Equality",
    }
    # The kernel of a solution several files hold is the first file's.
    assert rows.kernels[10]["kernel_name"] == name_of(10)
    assert rows.kernels[44]["mt_m"] == 64 * (1 + 44 % 4)
    assert set(rows.kernels) == {10, 20, 30, 40, 41, 44, 60, 61, 90}
    # A solution whose sizeMapping TensileLite cannot read keeps its name.
    broken = sol(70, None)
    broken["sizeMapping"]["streamK"] = "x"
    write(libdir, NEAR, [broken], problem(equality(70)))
    assert lp.read_type_rows(libdir, STEM).kernels[70] == {
        "sol_idx_global": 70,
        "kernel_name": name_of(70),
    }


def test_the_device_is_read_for_a_bank_only(tmp_path, libdir):
    absent = tmp_path / "absent"
    no_chip = {"arch": "gfx950", "hardware": dict(HARDWARE), "library": {}}
    no_chip["library"]["kind"] = "origami"
    assert bp.config_pool(no_chip, libdir, STEM, 0, absent).indices == [30, 10, 20]
    no_chip["library"]["kind"] = ["origami", "bok"]
    with pytest.raises(RuntimeError, match="hardware.pci_chip_id"):
        bp.config_pool(no_chip, libdir, STEM, 0, absent)
    with pytest.raises(ValueError, match="needs the bench device"):
        lp.resolve(libdir, STEM, lp.PoolSource(("bok",)))


def test_blocked_kernels_leave_their_kind(libdir):
    blocklist = cx.Blocklist([cx.Entry(name_of(41)), cx.Entry(name_of(30))])
    whole = bp.config_pool(new_form(["origami", "bok"]), libdir, STEM)
    kept = lp.without_blocked(whole, blocklist)
    assert kept.indices == [10, 20, 40, 44]
    assert [p.blocked for p in kept.parts] == [1, 1]
    assert kept.files == whole.files == [STEM, CU256]
    assert lp.without_blocked(whole, None) is whole


# ── composition ──────────────────────────────────────────────────────────────


def test_summary_and_printout(libdir):
    pool = bp.config_pool(new_form(["origami", "bok"]), libdir, STEM)
    s = lp.summary(pool)
    assert (s["kernels"], s["kinds"], s["bok_rows"]) == (
        6,
        ["origami", "bok"],
        ["freesize", "equality"],
    )
    assert s["device"] == "gfx950 with 256 CUs, chip 0x75a0" and not s["legacy"]
    origami, bok = s["parts"]
    assert (origami["kernels"], origami["rows"], origami["files"]) == (
        3,
        {"prediction": 3},
        [STEM],
    )
    assert (bok["kernels"], bok["duplicates"], sorted(bok["idle_files"])) == (
        3,
        3,
        sorted([STEM, A3A2]),
    )
    assert s["near_type_files"] == [NEAR]
    assert s["keys"] == len(
        {tuple(k[f] for f in lp.SIGNATURE_FIELDS) for k in pool.kernels}
    )
    lines = lp.describe(pool)
    assert lines[0] == (
        "info",
        "library.kind [origami, bok] with bok rows freesize, equality, on the bench "
        "device (gfx950 with 256 CUs, chip 0x75a0)",
    )
    text = "\n".join(line for _, line in lines)
    assert f"origami: 3 kernels in {origami['keys']} keys" in text
    assert f"bok: 3 kernels in {bok['keys']} keys" in text
    assert f"bok: 1 file holds kernels of one CU count ({CU256})" in text
    assert (
        f"not read, of another problem type with the library's data types and "
        f"operation: {NEAR}"
    ) in text
    assert text.splitlines()[-1].startswith("pool: 6 kernels in")
    assert all(level == "info" for level, _ in lines)


def test_printout_of_a_legacy_bank_of_another_problem_type(libdir):
    cfg = config(planner={"enabled": True, "bank_library_stems": [OTHER_LAYOUT]})
    pool = bp.config_pool(cfg, libdir, STEM)
    assert pool.indices == [30, 10, 20, 95]
    warns = [line for level, line in lp.describe(pool) if level == "warn"]
    assert warns == [
        f"planner.bank_library_stems names {OTHER_LAYOUT}, of another problem type "
        f"than the library; TensileLite's bank takes the library's problem type only"
    ]
    empty = bp.config_pool(
        new_form(["origami", "bok"], bok_rows="gridbased"), libdir, STEM
    )
    assert ("warn", "bok: no kernel of the bank on this device") in lp.describe(empty)


# ── the live library ─────────────────────────────────────────────────────────

LIVE_DIR = os.environ.get("TILEWRIGHT_GFX950_LIBRARY_DIR")


@pytest.mark.skipif(
    not LIVE_DIR,
    reason="set TILEWRIGHT_GFX950_LIBRARY_DIR to a gfx950 Tensile library directory",
)
def test_library_kind_on_a_live_library():
    """bf16 TN on 0x75a0 with 256 CUs: the live run's legacy bank
    (`planner.bank_library_stems`, the `_CU256_` file) and
    `library.kind [origami, bok]` are one pool, kernel for kernel, and
    `library.kind origami` is the Prediction table."""
    libdir = Path(LIVE_DIR)
    legacy = config(planner={"enabled": True, "bank_library_stems": [CU256]})
    new = new_form(["origami", "bok"])
    a = bp.config_pool(legacy, libdir, STEM, apply_blocklist=False)
    b = bp.config_pool(new, libdir, STEM, apply_blocklist=False)
    assert a.kernels == b.kernels and a.bank.stems == b.bank.stems == [CU256]
    assert lp.summary(a)["keys"] == lp.summary(b)["keys"]
    assert bp.config_identity(legacy, libdir, STEM).digest == (
        bp.config_identity(new, libdir, STEM).digest
    )
    table = bp.config_pool(new_form("origami"), libdir, STEM)
    assert table.kernels == a.table
