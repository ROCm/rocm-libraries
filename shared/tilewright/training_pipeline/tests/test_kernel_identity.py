# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Kernel identity across hipBLASLt builds (lib/kernel_identity.py).

A rebuild renumbers the solutions of the pool and the bank. Every join of a
persisted artifact (enriched rows, plans, timings, bench logs, audit
generations and v2 records) with the library read afterwards must follow the
kernels by their ids or refuse; none may bind a solution index to the
kernel another build gave it."""

import json
import os
import random
import shutil
import subprocess
import sys
import types
import zlib
from pathlib import Path

import msgpack
import pytest
import yaml

import bench_fakes as bf
import fake_hipblaslt as fh
import run_orchestrator as ro
import stage04_convert_to_enriched_dataset as s04
import stage05_train as s05
import test_audit as ta
from test_run_orchestrator import layout_of_the_past
from bench_fakes import (
    BANNER,
    LIBRARY_STEM,
    Gemm,
    measured_lines,
    preamble,
    skip,
    solution,
    winner,
)
from lib import audit as al
from lib import audit_v2 as v2
from lib import bank_pool as bp
from lib import evaluate as ev
from lib import kernel_identity as ki
from lib import run_analysis as ra
from lib import run_layout
from lib.dat import KERNELS_FILE
from lib.hardware import DeviceHardware, arch_constants

BANK_STEM = LIBRARY_STEM.replace("_Dijk_", "_Dijk_CU256_")
STEMS = [LIBRARY_STEM, BANK_STEM]
# Kernels 0..5 form the Prediction table, 6..11 the bank. Kernels 2 and 3,
# 4 and 9, and 10 and 11 share a signature and differ in workgroup mapping.
SPECS = [
    ((128, 128, 64), (16, 16, 128, 1), 0, 0, 8),
    ((256, 128, 64), (16, 16, 128, 1), 0, 0, 8),
    ((64, 64, 64), (32, 32, 64, 1), 0, 0, 8),
    ((64, 64, 64), (32, 32, 64, 1), 0, 0, 4),
    ((128, 256, 64), (16, 16, 128, 1), 0, 4, 8),
    ((256, 256, 64), (32, 32, 64, 1), 4, 0, 8),
    ((32, 64, 128), (16, 16, 128, 1), 0, 0, 8),
    ((64, 128, 128), (16, 16, 128, 1), 0, 0, 8),
    ((128, 64, 64), (32, 32, 64, 1), 0, 0, 8),
    ((128, 256, 64), (16, 16, 128, 1), 0, 4, 4),
    ((256, 64, 64), (16, 16, 128, 1), 0, 0, 8),
    ((256, 64, 64), (16, 16, 128, 1), 0, 0, 2),
]
# Kernel 3 retuned: another macro tile under its own solution index.
RETUNED = ((128, 64, 128), (32, 32, 64, 1), 0, 0, 4)
N_TABLE = 6
GEMMS = [Gemm(1024, 512, 256), Gemm(96, 2048, 512), Gemm(4096, 4096, 128, 2)]


def name_of(spec) -> str:
    (m, n, d), mi, nta, ntb, wgm = spec
    return (
        f"Cijk_Alik_Bljk_F8BS_MT{m}x{n}x{d}_MI{mi[0]}x{mi[1]}x1_NTA{nta}_NTB{ntb}"
        f"_WGM{wgm}_TEST"
    )


def sol(spec, index: int, local: int):
    mt, mi, nta, ntb, wgm = spec
    s = solution(index, local, mt, mi=mi, nta=nta, ntb=ntb, workGroupMapping=wgm)
    return dict(s, name=name_of(spec))


def write_build(
    d: Path, ids, *, drop=(), twin=None, table_order=None, specs=SPECS
) -> Path:
    """The pool and bank logic files of a build numbering kernel k `ids[k]`;
    `drop` leaves kernels out, `twin` (kernel, index) adds a copy of a bank
    kernel under another index, `table_order` reorders the table, `specs`
    replaces the kernels."""
    d.mkdir(parents=True, exist_ok=True)
    table = [k for k in (table_order or range(N_TABLE)) if k not in drop]
    sols = [sol(specs[k], ids[k], local) for local, k in enumerate(sorted(table))]
    doc = {
        "solutions": sols,
        "library": {
            "type": "Problem",
            "rows": [
                {
                    "predicate": {"type": "PredictionMatching"},
                    "library": {"type": "Prediction", "table": [ids[k] for k in table]},
                }
            ],
        },
    }
    (d / f"{LIBRARY_STEM}.dat").write_bytes(msgpack.packb(doc, use_bin_type=True))
    bank = [k for k in range(N_TABLE, len(specs)) if k not in drop]
    sols = [sol(specs[k], ids[k], j) for j, k in enumerate(bank)]
    if twin is not None:
        k, index = twin
        sols.append(sol(specs[k], index, len(sols)))
    rows = [
        {
            "predicate": {"type": "EqualityMatching"},
            "library": {
                "type": "Matching",
                "table": [{"key": [1, 1, 1, 1], "index": s["index"]} for s in sols],
            },
        }
    ]
    doc = {"solutions": sols, "library": {"type": "Problem", "rows": rows}}
    raw = zlib.compress(msgpack.packb(doc, use_bin_type=True))
    (d / f"{BANK_STEM}.dat.zlib").write_bytes(raw)
    return d


IDS_A = [100 + k for k in range(N_TABLE)] + [10 + j for j in range(6)]
# A library added before the pool's in the global numbering.
IDS_SHIFT = [i + 37 for i in IDS_A]
IDS_SHUFFLE = random.Random(5).sample(range(500, 512), len(SPECS))
# Build A's indices, each naming another kernel.
IDS_ROTATE = IDS_A[1:] + IDS_A[:1]
SPECS_RETUNED = SPECS[:3] + [RETUNED] + SPECS[4:]


def latency(g: Gemm, k: int) -> float:
    return round(5.0 + (g.m * 7 + g.n * 3 + g.k + 13 * k) % 97 / 10.0, 3)


def log_lines(ids, gemms=GEMMS, kernels=range(len(SPECS)), specs=SPECS):
    """A bench log of `gemms` over `kernels` run with a build numbering
    kernel k `ids[k]`; the last kernel of each GEMM is skip-screened."""
    lines = list(BANNER)
    for g in gemms:
        ks = list(kernels)
        lines += preamble(len(ks))
        for rank, k in enumerate(ks[:-1]):
            mt = specs[k][0]
            lines += measured_lines(
                rank, g, ids[k], latency(g, k), mt, with_identity=False
            )
            lines += [
                f"    --Solution index: {ids[k]}",
                f"    --Solution name:  {name_of(specs[k])}",
                f"    --kernel name:    {name_of(specs[k])}",
            ]
        lines.append(skip(len(ks) - 1, ids[ks[-1]], "90.00"))
        best = min(ks[:-1], key=lambda k: latency(g, k))
        lines += winner(ks.index(best), g, ids[best], latency(g, best))
    return lines


def write_logs(d: Path, ids, bench_library=None, **kw) -> Path:
    """`d/logs/block_*.log` of `log_lines`; with `bench_library` (a
    LibraryIdentity) the bench's stage summary records it, as stage03
    does."""
    logs = d / "logs"
    logs.mkdir(parents=True)
    (logs / "block_0000_lines1-3_gpu0.log").write_text("\n".join(log_lines(ids, **kw)))
    if bench_library is not None:
        summary = {ki.RECORD_KEY: ki.record(bench_library, ())}
        (d / "stage03_summary.json").write_text(json.dumps(summary))
    return logs


def plan_doc(identity, ids, listed=(0, 2, 6, 9)):
    shapes = [
        {
            "m": g.m,
            "n": g.n,
            "k": g.k,
            "batch": g.batch,
            "role": "planned" if listed is not None else "exhaustive",
            "kernels": [ids[k] for k in listed] if listed is not None else None,
        }
        for g in GEMMS
    ]
    doc = {"version": 1, "shapes": shapes}
    if identity is not None:
        doc[ki.RECORD_KEY] = ki.record(identity)
    return doc


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Build A's library, a log benched with it, its plan, and the rows
    stage04 enriched from them; builds B renumbering the kernels."""
    lib_a = write_build(tmp_path / "lib_a", IDS_A)
    ident_a = ki.library_identity(lib_a, STEMS)
    logs = write_logs(tmp_path / "bench_a", IDS_A)
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps(plan_doc(ident_a, IDS_A)))
    out = tmp_path / "stage04"
    assert run_stage04(monkeypatch, logs, lib_a, out, plan) == 0
    return types.SimpleNamespace(
        tmp=tmp_path, lib_a=lib_a, ident_a=ident_a, logs=logs, plan=plan, rows=out
    )


def run_stage04(monkeypatch, logs, lib, out, plan=None, stems=True):
    argv = ["stage04", "--log-dir", logs, "--out-dir", out, "--dat-dir", lib]
    if stems:
        argv += ["--library-stem", LIBRARY_STEM, "--extra-library-stem", BANK_STEM]
    argv += ["--plan", plan] if plan is not None else []
    monkeypatch.setattr(sys, "argv", [str(a) for a in argv + ["--quiet"]])
    return s04.main()


def pool_library(
    d: Path, pool, offset: int = 0, stem: str = LIBRARY_STEM, order=None
) -> ki.LibraryIdentity:
    """`pool` (test_stage05_train.kernel_pool entries, which stage tests
    write rows of) as the library file `stem` of a build in `d` numbering
    each kernel `sol_idx_global + offset`, in the order `order` (indices of
    `pool`) when given; every kernel named after its index in `pool`, so
    that builds agree on its kernel id. Returns its identity."""
    from test_stage05_train import library_solutions

    d.mkdir(parents=True, exist_ok=True)
    picked = [pool[i] for i in (order or range(len(pool)))]
    sols = library_solutions(
        [dict(k, sol_idx_global=k["sol_idx_global"] + offset) for k in picked]
    )
    sols = [
        dict(s, name=f"{s['name']}_K{k['sol_idx_global']}")
        for s, k in zip(sols, picked)
    ]
    bf.write_logic(d / f"{stem}.dat", sols)
    return ki.library_identity(d, [stem])


def record_rows(rows: Path, identity: ki.LibraryIdentity, stem: str) -> Path:
    """Enriched rows written without stage04, given the kernels.json stage04
    writes: the kernel attributes and the record of `identity`'s kernels."""
    from lib.dat import write_kernel_attributes

    write_kernel_attributes(
        Path(rows) / KERNELS_FILE,
        dict(identity.kernels),
        stem,
        extra={ki.RECORD_KEY: ki.record(identity)},
    )
    return Path(rows)


def unrecorded(rows: Path, dst: Path, keep_attributes: bool = True) -> Path:
    """A copy of enriched rows as stage04 wrote them before kernel identities
    were recorded."""
    shutil.copytree(rows, dst)
    doc = json.loads((dst / KERNELS_FILE).read_text())
    for key in (ki.RECORD_KEY, "signatures"):
        doc.pop(key)
    if keep_attributes:
        (dst / KERNELS_FILE).write_text(json.dumps(doc))
    else:
        (dst / KERNELS_FILE).unlink()
    return dst


# ── ids, digests, remaps ─────────────────────────────────────────────────────


def test_a_kernel_id_is_its_signature_and_a_hash_of_its_name(tmp_path):
    ident = ki.library_identity(write_build(tmp_path, IDS_A), STEMS)
    kid2, kid3 = ident.ids[IDS_A[2]], ident.ids[IDS_A[3]]
    assert kid2.split(":")[0] == kid3.split(":")[0] == "MT64x64x64_MI32x32x64_NTA0_NTB0"
    assert kid2 != kid3 and len(kid2.split(":")[1]) == ki.NAME_HASH_DIGITS
    assert ki.label_of_id(kid2) == "MT64x64x64 MI32x32x64 NTA0 NTB0"
    assert ki.signature_label(ident.kernels[IDS_A[2]]) == ki.label_of_id(kid2)
    shifted = ki.library_identity(write_build(tmp_path / "b", IDS_SHIFT), STEMS)
    assert [ident.ids[i] for i in IDS_A] == [shifted.ids[i] for i in IDS_SHIFT]
    assert len(set(ident.ids.values())) == len(SPECS)


def test_the_digest_follows_the_index_of_every_kernel_and_the_table_order(tmp_path):
    a = ki.library_identity(write_build(tmp_path / "a", IDS_A), STEMS)
    again = ki.library_identity(write_build(tmp_path / "a2", IDS_A), STEMS)
    assert a.digest == again.digest and a.ids == again.ids
    shifted = ki.library_identity(write_build(tmp_path / "s", IDS_SHIFT), STEMS)
    reordered = write_build(tmp_path / "o", IDS_A, table_order=[1, 0, 2, 3, 4, 5])
    order = ki.library_identity(reordered, STEMS)
    assert len({a.digest, shifted.digest, order.digest}) == 3
    assert order.ids == a.ids and order.files[BANK_STEM] == a.files[BANK_STEM]
    assert order.files[LIBRARY_STEM] != a.files[LIBRARY_STEM]
    assert order.index_digests == a.index_digests
    assert a.index_digests[BANK_STEM] == a.files[BANK_STEM]
    assert shifted.index_digests[LIBRARY_STEM] != a.index_digests[LIBRARY_STEM]
    rec = ki.record(a, ())
    assert rec[ki.INDEX_DIGESTS] == dict(a.index_digests)
    assert ki.changed_files(rec, order) == []
    assert ki.changed_files(rec, shifted) == STEMS
    old = dict(rec)
    old.pop(ki.INDEX_DIGESTS)
    assert ki.changed_files(old, order) == [LIBRARY_STEM]
    swapped = ki.library_identity(tmp_path / "a", [BANK_STEM, LIBRARY_STEM])
    assert (swapped.digest, swapped.ids, swapped.files) == (a.digest, a.ids, a.files)


def test_a_record_maps_its_indices_to_their_kernels_in_a_rebuild(tmp_path):
    a = ki.library_identity(write_build(tmp_path / "a", IDS_A), STEMS)
    rec = ki.record(a, [IDS_A[0], IDS_A[7], 999])
    assert rec["files"] == dict(a.files)
    assert rec["kernel_ids"] == {
        LIBRARY_STEM: {str(IDS_A[0]): a.ids[IDS_A[0]]},
        BANK_STEM: {str(IDS_A[7]): a.ids[IDS_A[7]]},
    }
    rec = ki.record(a)
    assert ki.recorded_ids(rec) == dict(a.ids)
    shuffled = ki.library_identity(write_build(tmp_path / "b", IDS_SHUFFLE), STEMS)
    remap = ki.Remap.of(rec, shuffled, "rows")
    assert not remap.same and remap.n_moved == len(SPECS) and not remap.unresolved
    assert len(remap.changed) == len(SPECS)
    assert [remap.index(i) for i in IDS_A] == IDS_SHUFFLE
    assert remap.index(-3) == -3 and remap.index(12345) is None
    text = remap.explain()
    assert "hipBLASLt build changed" in text and "12 of the 12" in text
    first = IDS_A.index(min(IDS_A))
    assert f"{a.ids[IDS_A[first]]}: solution {IDS_A[first]} then, " in text
    assert f"then, {IDS_SHUFFLE[first]} now" in text
    with pytest.raises(ki.BuildChangedError, match="hipBLASLt build changed"):
        remap.require_same()
    same = ki.Remap.of(rec, a, "rows")
    assert same.same and same.index(IDS_A[4]) == IDS_A[4]
    same.require_same()
    old = ki.Remap.of(None, shuffled, "rows")
    assert old.same and not old.recorded and old.index(IDS_A[1]) == IDS_A[1]
    assert "records no kernel identities" in old.explain()


def test_kernels_a_rebuild_drops_or_doubles_are_unresolved(tmp_path):
    a = ki.library_identity(write_build(tmp_path / "a", IDS_A), STEMS)
    rec = ki.record(a)
    gone = write_build(tmp_path / "gone", IDS_SHIFT, drop=[5])
    remap = ki.Remap.of(rec, ki.library_identity(gone, STEMS), "rows")
    assert remap.unresolved == [IDS_A[5]] and remap.index(IDS_A[5]) is None
    assert remap.index(IDS_A[0]) == IDS_SHIFT[0]
    assert a.ids[IDS_A[5]] in remap.explain()
    twin = write_build(tmp_path / "twin", IDS_SHIFT, twin=(11, 999))
    remap = ki.Remap.of(rec, ki.library_identity(twin, STEMS), "rows")
    assert remap.unresolved == [IDS_A[11]]
    grown = write_build(tmp_path / "grown", IDS_A, twin=(10, 999))
    remap = ki.Remap.of(rec, ki.library_identity(grown, STEMS), "rows")
    assert remap.same and remap.digest_then != remap.digest_now
    assert remap.files_changed == [BANK_STEM]
    assert "still names the same kernel" in remap.explain()


def test_a_kernel_retuned_under_its_own_index_counts_as_changed(tmp_path):
    a = ki.library_identity(write_build(tmp_path / "a", IDS_A), STEMS)
    retuned = ki.library_identity(
        write_build(tmp_path / "r", IDS_A, specs=SPECS_RETUNED), STEMS
    )
    remap = ki.Remap.of(ki.record(a), retuned, "plan p")
    assert remap.changed == [(IDS_A[3], a.ids[IDS_A[3]])]
    assert remap.unresolved == [IDS_A[3]] and remap.n_moved == 0
    assert remap.index(IDS_A[2]) == IDS_A[2] and remap.index(IDS_A[3]) is None
    text = remap.explain()
    assert "1 of the 12 solution indices it names now name another kernel" in text
    assert (
        f"not in the library this stage reads under one solution index (e.g. "
        f"{a.ids[IDS_A[3]]})" in text
    )
    assert f"library file(s) {LIBRARY_STEM} changed" in text


def test_a_library_read_without_some_recorded_files_is_not_a_rebuild(tmp_path):
    """A stage reading the pool alone meets rows recorded with the pool and
    the bank: their bank indices are left as they are while the pool is
    unchanged, and left out when it changed."""
    a = ki.library_identity(write_build(tmp_path / "a", IDS_A), STEMS)
    rec = ki.record(a)
    pool_only = ki.library_identity(tmp_path / "a", [LIBRARY_STEM])
    remap = ki.Remap.of(rec, pool_only, "rows")
    assert remap.same and remap.files_outside == [BANK_STEM]
    assert remap.index(IDS_A[7]) == IDS_A[7]
    remap.require_same()
    assert "build changed" not in remap.explain()
    assert f"{BANK_STEM}, which this stage does not read" in remap.explain()
    shifted = write_build(tmp_path / "s", IDS_SHIFT)
    remap = ki.Remap.of(rec, ki.library_identity(shifted, [LIBRARY_STEM]), "rows")
    assert remap.index(IDS_A[0]) == IDS_SHIFT[0] and remap.index(IDS_A[7]) is None
    assert remap.unresolved == sorted(IDS_A[N_TABLE:])
    text = remap.explain()
    assert f"library file(s) {LIBRARY_STEM} changed" in text
    assert f"{BANK_STEM}, which this stage does not read" in text


def test_an_unread_files_index_another_read_file_holds_is_unresolved(tmp_path):
    """Rows recorded with the pool and bank A, read with the pool and bank
    B: the pool is unchanged, but bank B holds two of bank A's indices with
    other kernels, so those two are unresolved; an index bank B holds with
    the same kernel, or does not hold, keeps itself."""
    a = ki.library_identity(write_build(tmp_path / "a", IDS_A), STEMS)
    rec = ki.record(a)
    bank_b = BANK_STEM.replace("_CU256_", "_CU128_")
    lib = shutil.copytree(tmp_path / "a", tmp_path / "b")
    bf.write_logic(
        lib / f"{bank_b}.dat",
        [sol(SPECS[6], IDS_A[6], 0), sol(SPECS[8], IDS_A[7], 1)]
        + [sol(SPECS[7], IDS_A[8], 2)],
    )
    remap = ki.Remap.of(rec, ki.library_identity(lib, [LIBRARY_STEM, bank_b]), "rows")
    assert remap.files_outside == [BANK_STEM] and remap.files_changed == []
    assert not remap.same and remap.unresolved == [IDS_A[7], IDS_A[8]]
    assert remap.changed == [(s, a.ids[s]) for s in (IDS_A[7], IDS_A[8])]
    assert [remap.index(s) for s in (IDS_A[0], IDS_A[6], IDS_A[9])] == [
        IDS_A[0],
        IDS_A[6],
        IDS_A[9],
    ]
    text = remap.explain()
    assert "build changed" not in text
    assert f"{BANK_STEM}, which this stage does not read" in text
    assert "hold other kernels under 2 of the solution indices it names" in text
    assert f"solution {IDS_A[7]}, which named {a.ids[IDS_A[7]]}" in text
    with pytest.raises(ki.BuildChangedError, match="hold other kernels"):
        remap.require_same()


def test_a_miss_record_names_each_pick_and_winner_next_to_its_index(tmp_path):
    from lib import miss_rebench as mr

    a = ki.library_identity(write_build(tmp_path / "a", IDS_A), STEMS)
    measured = {"m": 1, "pick": IDS_A[2], "status": "measured", "winner_sol": IDS_A[7]}
    failed = {"m": 2, "pick": 999, "status": "failed", "ratio": None}
    rec = {"runs": 3, "gemms": [measured, failed]}
    out = mr.named_record(rec, a)
    assert out["library_digest"] == a.digest and out["runs"] == 3
    first, second = out["gemms"]
    assert list(first) == [
        "m",
        "pick",
        "pick_signature",
        "pick_kernel_id",
        "status",
        "winner_sol",
        "winner_signature",
        "winner_kernel_id",
    ]
    assert first["pick_kernel_id"] == a.ids[IDS_A[2]]
    assert first["pick_signature"] == "MT64x64x64 MI32x32x64 NTA0 NTB0"
    assert first["winner_kernel_id"] == a.ids[IDS_A[7]]
    assert (second["pick_signature"], second["pick_kernel_id"]) == (None, None)
    assert "winner_sol" not in second and list(rec["gemms"][0]) == list(measured)
    assert mr.named_record(None, None) == {"library_digest": None}
    assert mr.named_record(rec, None)["gemms"][0]["pick_kernel_id"] is None


def test_member_orders_follow_kernels_not_indices(tmp_path):
    device = bp.BankDevice(processor="gfx950", cu_count=256, chip_id=0x75A0)

    def orders(lib):
        pool = bp.library_pool(lib, LIBRARY_STEM, device, [BANK_STEM]).kernels
        return ki.member_orders(pool, ra.signature_of)

    a = orders(write_build(tmp_path / "a", IDS_A))
    keys = {k.split("|")[0]: k for k in a}
    assert set(keys) == {
        "MT64x64x64_MI32x32x64_NTA0_NTB0",
        "MT128x256x64_MI16x16x128_NTA0_NTB4",
        "MT256x64x64_MI16x16x128_NTA0_NTB0",
    }
    assert keys["MT128x256x64_MI16x16x128_NTA0_NTB4"].endswith(
        "|128,256,64,16,16,128,0,4"
    )
    assert orders(write_build(tmp_path / "t", IDS_SHIFT)) == a
    swapped = write_build(tmp_path / "o", IDS_A, table_order=[0, 1, 3, 2, 4, 5])
    moved = orders(swapped)
    assert [k for k in a if a[k] != moved[k]] == [
        keys["MT64x64x64_MI32x32x64_NTA0_NTB0"]
    ]
    bank_swap = IDS_A[:10] + [IDS_A[11], IDS_A[10]]
    moved = orders(write_build(tmp_path / "b", bank_swap))
    assert [k for k in a if a[k] != moved[k]] == [
        keys["MT256x64x64_MI16x16x128_NTA0_NTB0"]
    ]


# ── stage04 ──────────────────────────────────────────────────────────────────


def test_stage04_records_the_kernels_its_rows_name(world):
    doc = json.loads((world.rows / KERNELS_FILE).read_text())
    rec = ki.read_record(doc)
    assert rec["digest"] == world.ident_a.digest
    assert ki.recorded_ids(rec) == {i: world.ident_a.ids[i] for i in IDS_A}
    assert doc["signatures"] == {
        str(i): ki.label_of_id(world.ident_a.ids[i]) for i in IDS_A
    }
    summary = json.loads((world.rows / "enrich_summary.json").read_text())
    assert summary["library_digest"] == world.ident_a.digest
    assert summary["bench_records"] == 0


def test_stage04_drops_rows_of_another_build_and_then_fails(world, monkeypatch):
    lib_b = write_build(world.tmp / "lib_b", IDS_ROTATE)
    one = write_logs(world.tmp / "one", IDS_A, gemms=GEMMS[:1], kernels=range(4))
    assert run_stage04(monkeypatch, one, lib_b, world.tmp / "o1") == 2
    summary = json.loads((world.tmp / "o1" / "enrich_summary.json").read_text())
    assert summary["counters"]["problems_naming_other_kernels"] == 1
    assert summary["name_mismatches"] == 3
    with pytest.raises(RuntimeError, match="hipBLASLt build changed") as e:
        run_stage04(monkeypatch, world.logs, lib_b, world.tmp / "o2")
    import re

    sid = int(re.search(r"e\.g\. solution (\d+): bench", str(e.value)).group(1))
    k = IDS_A.index(sid)
    ident_b = ki.library_identity(lib_b, STEMS)
    ran = f"bench {ident_b.ids[IDS_ROTATE[k]]} (solution {IDS_ROTATE[k]} here)"
    assert ran in str(e.value) and f".dat {ident_b.ids[sid]})" in str(e.value)


def test_stage04_fails_on_the_first_other_name_of_a_recorded_bench(world, monkeypatch):
    logs = write_logs(world.tmp / "b", IDS_A, bench_library=world.ident_a)
    text = (logs / "block_0000_lines1-3_gpu0.log").read_text()
    wrong = name_of(SPECS[1])
    (logs / "block_0000_lines1-3_gpu0.log").write_text(
        text.replace(
            f"--Solution name:  {name_of(SPECS[0])}", f"--Solution name:  {wrong}", 1
        )
    )
    with pytest.raises(RuntimeError, match="although the bench recorded") as e:
        run_stage04(monkeypatch, logs, world.lib_a, world.tmp / "o")
    ids = world.ident_a.ids
    assert str(e.value).startswith(
        f"solution {IDS_A[0]}: the bench ran {ids[IDS_A[1]]} (solution {IDS_A[1]} "
        f"here), the .dat names {ids[IDS_A[0]]},"
    )
    alien = "Cijk_Alik_Bljk_F8BS_MT64x32x16_MI16x16x1_NTA0_NTB0_ELSEWHERE"
    (logs / "block_0000_lines1-3_gpu0.log").write_text(
        text.replace(
            f"--Solution name:  {name_of(SPECS[0])}", f"--Solution name:  {alien}", 1
        )
    )
    with pytest.raises(RuntimeError, match="which this library does not hold") as e:
        run_stage04(monkeypatch, logs, world.lib_a, world.tmp / "o3")
    assert f"MT64x32x16 (name {ki.name_hash(alien)})" in str(e.value)
    good = write_logs(world.tmp / "g", IDS_A, bench_library=world.ident_a)
    assert run_stage04(monkeypatch, good, world.lib_a, world.tmp / "o2") == 0
    summary = json.loads((world.tmp / "o2" / "enrich_summary.json").read_text())
    assert summary["bench_records"] == 1


def test_stage04_refuses_logs_a_rebuild_renumbered_even_by_their_skip_rows(
    world, monkeypatch, capsys
):
    """Kernel 11, the skip-screened kernel of every GEMM, retuned under its
    own index: the tested rows still print the .dat's names, so only the
    bench's record shows that the skip rows name another kernel now."""
    specs = SPECS[:11] + [((256, 32, 64), (16, 16, 128, 1), 0, 0, 2)]
    lib_c = write_build(world.tmp / "lib_c", IDS_A, specs=specs)
    plain = write_logs(world.tmp / "plain", IDS_A)
    assert run_stage04(monkeypatch, plain, lib_c, world.tmp / "o1") == 0
    recorded = write_logs(world.tmp / "rec", IDS_A, bench_library=world.ident_a)
    capsys.readouterr()
    assert run_stage04(monkeypatch, recorded, lib_c, world.tmp / "o2") == 1
    err = capsys.readouterr().err
    assert "ran with other library files" in err and BANK_STEM in err


def test_stage04_reads_a_recorded_bench_after_a_rebuild_of_the_table_alone(
    world, monkeypatch, capsys
):
    """A rebuild that only reorders the Prediction table changes the pool
    file's digest, not what its indices name: logs whose bench recorded the
    old build are enriched. A record written before index digests were
    recorded cannot tell, and stays refused."""
    lib_t = write_build(world.tmp / "lib_t", IDS_A, table_order=[1, 0, 2, 3, 4, 5])
    assert ki.library_identity(lib_t, STEMS).digest != world.ident_a.digest
    logs = write_logs(world.tmp / "t", IDS_A, bench_library=world.ident_a)
    assert run_stage04(monkeypatch, logs, lib_t, world.tmp / "o") == 0
    summary = json.loads((world.tmp / "o" / "enrich_summary.json").read_text())
    assert summary["bench_records"] == 1
    want = [r for p in sorted(world.rows.glob("chunk_*.csv")) for r in csv_rows(p)]
    got = [
        r for p in sorted((world.tmp / "o").glob("chunk_*.csv")) for r in csv_rows(p)
    ]
    assert len(got) == len(want)
    assert {(r["m"], r["sol_idx_global"], r["us"]) for r in got} == {
        (r["m"], r["sol_idx_global"], r["us"]) for r in want
    }
    old = json.loads((logs.parent / "stage03_summary.json").read_text())
    old[ki.RECORD_KEY].pop(ki.INDEX_DIGESTS)
    (logs.parent / "stage03_summary.json").write_text(json.dumps(old))
    capsys.readouterr()
    assert run_stage04(monkeypatch, logs, lib_t, world.tmp / "o2") == 1
    err = capsys.readouterr().err
    assert "written before index digests were recorded" in err
    assert "so its solution indices name other kernels" not in err


def test_stage03b_merges_logs_of_a_build_whose_table_alone_changed(world):
    """stage03b's check of the library stage03 recorded follows the same
    rule: a reordered table passes, other kernels under an index do not."""
    import stage03b_rebench_near_ties as s03b

    s03 = world.tmp / "s03"
    write_logs(s03, IDS_A, bench_library=world.ident_a)
    lib_t = write_build(world.tmp / "lib_t", IDS_A, table_order=[1, 0, 2, 3, 4, 5])
    lib_r = write_build(world.tmp / "lib_r", IDS_A, specs=SPECS_RETUNED)

    def args(lib):
        return types.SimpleNamespace(
            identity_dir=lib, identity_stem=STEMS, stage03_dir=s03
        )

    now, rec = s03b.library_record(args(lib_t))
    assert now == ki.library_identity(lib_t, STEMS) and rec == ki.record(now, ())
    with pytest.raises(ValueError, match="name other kernels than a re-bench"):
        s03b.library_record(args(lib_r))
    unrecorded_s03 = world.tmp / "s03_plain"
    write_logs(unrecorded_s03, IDS_A)
    plain = types.SimpleNamespace(
        identity_dir=world.lib_a, identity_stem=STEMS, stage03_dir=unrecorded_s03
    )
    assert s03b.library_record(plain) == (None, None)


def test_stage04_reads_a_recorded_plan_at_its_kernels_current_indices(
    world, monkeypatch, capsys
):
    lib_b = write_build(world.tmp / "lib_b", IDS_SHUFFLE)
    logs = write_logs(world.tmp / "b", IDS_SHUFFLE)
    out = world.tmp / "o"
    assert run_stage04(monkeypatch, logs, lib_b, out, world.plan) == 0
    listed = {IDS_SHUFFLE[k] for k in (0, 2, 6, 9)}
    rows = [r for p in sorted(out.glob("chunk_*.csv")) for r in csv_rows(p)]
    assert {r["n_planned"] for r in rows} == {"4"}
    assert {int(r["sol_idx_global"]) for r in rows if r["is_planned"] == "1"} == listed
    assert "current indices" in capsys.readouterr().out

    gone = write_build(world.tmp / "gone", IDS_SHIFT, drop=[9])
    assert (
        run_stage04(
            monkeypatch,
            write_logs(world.tmp / "c", IDS_SHIFT, kernels=range(9)),
            gone,
            world.tmp / "o3",
            world.plan,
        )
        == 1
    )
    err = capsys.readouterr().err
    assert "of the kernels its lists name are among them" in err
    assert world.ident_a.ids[IDS_A[9]] in err


def test_stage04_reads_an_exhaustive_plan_after_any_rebuild(world, monkeypatch):
    plan = world.tmp / "exhaustive.json"
    plan.write_text(json.dumps(plan_doc(world.ident_a, IDS_A, listed=None)))
    lib_r = write_build(world.tmp / "lib_r", IDS_A, specs=SPECS_RETUNED)
    logs = write_logs(world.tmp / "r", IDS_A, specs=SPECS_RETUNED)
    assert run_stage04(monkeypatch, logs, lib_r, world.tmp / "o", plan) == 0
    lib_s = write_build(world.tmp / "lib_s", IDS_SHIFT)
    logs = write_logs(world.tmp / "s", IDS_SHIFT)
    assert run_stage04(monkeypatch, logs, lib_s, world.tmp / "o2", plan) == 0
    rows = [
        r for p in sorted((world.tmp / "o2").glob("chunk_*.csv")) for r in csv_rows(p)
    ]
    assert {(r["is_planned"], r["n_planned"], r["is_origami_pick"]) for r in rows} == {
        ("1", "", "0")
    }


def test_stage04_takes_a_plan_without_a_record_with_a_warning(
    world, monkeypatch, capsys
):
    """A plan written before kernel identities were recorded lists kernels
    by indices nothing can check: stage04 takes them as they are and says
    so; an exhaustive one lists none."""
    old = world.tmp / "old_plan.json"
    old.write_text(json.dumps(plan_doc(None, IDS_A)))
    capsys.readouterr()
    assert run_stage04(monkeypatch, world.logs, world.lib_a, world.tmp / "o", old) == 0
    assert f"plan {old} records no kernel identities" in capsys.readouterr().err
    rows = [
        r for p in sorted((world.tmp / "o").glob("chunk_*.csv")) for r in csv_rows(p)
    ]
    listed = {str(IDS_A[k]) for k in (0, 2, 6, 9)}
    assert {r["sol_idx_global"] for r in rows if r["is_planned"] == "1"} == listed
    exhaustive = world.tmp / "old_exhaustive.json"
    exhaustive.write_text(json.dumps(plan_doc(None, IDS_A, listed=None)))
    out = world.tmp / "o2"
    assert run_stage04(monkeypatch, world.logs, world.lib_a, out, exhaustive) == 0
    assert "records no kernel identities" not in capsys.readouterr().err


def test_stage04_without_a_stem_records_only_the_files_its_rows_name(
    world, monkeypatch
):
    """Directory mode reads every library of the directory; the identity
    covers the files whose rows were written, and a Prediction table it
    cannot follow there is left out of the digest instead of failing."""
    lib = world.tmp / "dir"
    shutil.copytree(world.lib_a, lib)
    other = LIBRARY_STEM.replace("_Dijk_", "_Dijk_ID1234_")
    bf.write_logic(lib / f"{other}.dat", [sol(SPECS[0], 900, 0)])
    pool = lib / f"{LIBRARY_STEM}.dat"
    doc = msgpack.unpackb(pool.read_bytes(), raw=False)
    rows = doc["library"]["rows"]
    rows.append(json.loads(json.dumps(rows[0])))
    rows[1]["library"]["table"] = rows[1]["library"]["table"][::-1]
    pool.write_bytes(msgpack.packb(doc, use_bin_type=True))
    out = world.tmp / "o"
    assert run_stage04(monkeypatch, world.logs, lib, out, stems=False) == 0
    rec = ki.read_record(json.loads((out / KERNELS_FILE).read_text()))
    assert sorted(rec["files"]) == sorted(STEMS)
    assert ki.recorded_ids(rec) == {i: world.ident_a.ids[i] for i in IDS_A}
    assert run_stage04(monkeypatch, world.logs, lib, world.tmp / "o2") == 1


def csv_rows(path):
    import csv

    with path.open() as f:
        return list(csv.DictReader(f))


# ── joins that follow the kernels ────────────────────────────────────────────


HW = DeviceHardware(n_cu=100, lds_bytes=65536, l2_bytes=4 << 20)


def kernel_of(identity, sid):
    return identity.ids[int(sid)]


def measured(gemms, identity):
    """{(shape, kernel id): (us, skip, attributes)} of loaded GEMMs."""
    out = {}
    for g in gemms:
        for c in g["candidates"]:
            key = (
                (g["m"], g["n"], g["k"], g["batch_count"]),
                kernel_of(identity, c["sol_idx_global"]),
            )
            out[key] = (c["us"], c["is_skip"], c.get("attributes"))
    return out


def load(rows, identity):
    cells, stats = s05.load_enriched_chunks(str(rows), True, identity=identity)
    return [g for gs in cells.values() for g in gs], stats


@pytest.mark.parametrize("ids", [IDS_SHIFT, IDS_SHUFFLE])
def test_enriched_rows_keep_their_kernels_in_a_rebuild(world, ids, capsys):
    before, _ = load(world.rows, world.ident_a)
    lib_b = write_build(world.tmp / "lib_b", ids)
    ident_b = ki.library_identity(lib_b, STEMS)
    after, stats = load(world.rows, ident_b)
    assert measured(after, ident_b) == measured(before, world.ident_a)
    assert stats.get("rows_kernel_not_in_library", 0) == 0
    assert "hipBLASLt build changed" in capsys.readouterr().err
    unmapped, _ = load(world.rows, None)
    assert {c["sol_idx_global"] for g in unmapped for c in g["candidates"]} == set(
        IDS_A
    )


def test_rows_of_a_kernel_the_rebuild_dropped_are_left_out(world):
    lib_b = write_build(world.tmp / "lib_b", IDS_SHIFT, drop=[3])
    ident_b = ki.library_identity(lib_b, STEMS)
    after, stats = load(world.rows, ident_b)
    assert stats["rows_kernel_not_in_library"] == len(GEMMS)
    expected = {
        key: v
        for key, v in measured(
            load(world.rows, world.ident_a)[0], world.ident_a
        ).items()
        if key[1] != world.ident_a.ids[IDS_A[3]]
    }
    assert measured(after, ident_b) == expected


def test_plans_and_timings_keep_their_kernels_in_a_rebuild(world, capsys):
    ident_b = ki.library_identity(write_build(world.tmp / "lib_b", IDS_SHUFFLE), STEMS)
    gemms, _ = load(world.rows, ident_b)
    assert ev.attach_plans(gemms, [ev.read_plan(world.plan)], identity=ident_b) == 3
    want = {IDS_SHUFFLE[k] for k in (0, 2, 6, 9)}
    assert all(g["listed"] == want for g in gemms)
    timings = ev.load_timings([world.rows], identity=ident_b)
    by_kernel = {
        (key[:4], kernel_of(ident_b, c["sol_idx_global"])): c["us"]
        for key, g in timings.items()
        for c in g["candidates"]
    }
    assert by_kernel == {k: v[0] for k, v in measured(gemms, ident_b).items()}
    unrecorded_plan = dict(plan_doc(None, IDS_A), path="old/plan.json")
    capsys.readouterr()
    ev.attach_plans(gemms, [unrecorded_plan], identity=ident_b)
    assert all(g["listed"] == {IDS_A[k] for k in (0, 2, 6, 9)} for g in gemms)
    assert "old/plan.json records no kernel identities" in capsys.readouterr().err


def test_rows_without_a_record_are_checked_by_engine_config_and_attributes(
    world, capsys
):
    """Rows enriched before identities were recorded: kept when every index
    still holds a kernel of their engine config and attributes, else the
    kernels that differ are left out, with a count that names them."""
    legacy = unrecorded(world.rows, world.tmp / "legacy")
    before, _ = load(world.rows, world.ident_a)
    capsys.readouterr()
    kept, stats = load(legacy, world.ident_a)
    assert measured(kept, world.ident_a) == measured(before, world.ident_a)
    assert stats.get("rows_kernel_not_in_library", 0) == 0
    out = capsys.readouterr().out
    assert "the 12 kernels it names match the current library's" in out
    ident_r = ki.library_identity(write_build(world.tmp / "lib_r", IDS_ROTATE), STEMS)
    gone, stats = load(legacy, ident_r)
    assert gone == []
    assert stats["rows_kernel_not_in_library"] == stats["rows_total"]
    err = capsys.readouterr().err
    assert "12 of the 12 kernels it names differ" in err
    assert f"solution {min(IDS_A)} held " in err
    assert f"{ident_r.ids[min(IDS_A)]} now" in err
    no_attributes = unrecorded(world.rows, world.tmp / "bare", keep_attributes=False)
    _, stats = load(no_attributes, ident_r)
    # Kernels 2 / 3 and 10 / 11 differ only in workgroup mapping, an
    # attribute: rotated onto each other, their engine configs agree.
    per_kernel = stats["rows_total"] // len(SPECS)
    assert stats["rows_kernel_not_in_library"] == stats["rows_total"] - 2 * per_kernel
    timings = ev.load_timings([legacy], identity=ident_r)
    assert not any(g["candidates"] for g in timings.values())
    assert ev.load_timings([legacy], identity=world.ident_a) == ev.load_timings(
        [world.rows], identity=world.ident_a
    )


def test_an_unreadable_kernels_json_is_a_warning_without_an_identity(world, capsys):
    broken = world.tmp / "broken"
    shutil.copytree(world.rows, broken)
    (broken / KERNELS_FILE).write_text("{not json")
    gemms, _ = load(broken, None)
    assert {c["sol_idx_global"] for g in gemms for c in g["candidates"]} == set(IDS_A)
    assert "carry no attributes" in capsys.readouterr().err
    assert len(ev.load_timings([broken])) == len(GEMMS)
    kept, stats = load(broken, world.ident_a)
    assert stats.get("rows_kernel_not_in_library", 0) == 0
    assert "checked by their rows alone" in capsys.readouterr().err


def test_deployed_picks_are_the_same_kernels_after_a_rebuild(world):
    """The engine's picks on the rows of build A, evaluated against build
    B's pool, are the kernels it picks against build A's pool."""
    pytest.importorskip("tilewright")
    from test_mlrec import random_bundle

    from lib import mlrec
    from lib.grid import cell_key

    device = bp.BankDevice(processor="gfx950", cu_count=256, chip_id=0x75A0)
    signatures = sorted(
        {
            (*SPECS[k][0], *SPECS[k][1][:3], SPECS[k][2], SPECS[k][3])
            for k in range(len(SPECS))
        }
    )
    picks = {}
    for name, ids in (("a", IDS_A), ("b", IDS_SHIFT)):
        lib = world.lib_a if name == "a" else write_build(world.tmp / "lib_b", ids)
        identity = ki.library_identity(lib, STEMS)
        gemms, _ = load(world.rows, identity)
        pool = ev.pool_from_kernels(
            bp.library_pool(lib, LIBRARY_STEM, device, [BANK_STEM]).kernels
        )
        labels = sorted(
            {cell_key(g["m"], g["n"], g["k"], g["batch_count"]) for g in gemms}
        )
        bundle = random_bundle(
            labels,
            seed=3,
            signatures={c: [list(s) for s in signatures] for c in labels},
        )
        data = mlrec.write_model(
            bundle, None, "gfx950", arch_constants("gfx950"), "fp32"
        )
        out = ev.evaluate_model(
            data, gemms, hardware=HW, pools=ev.pools_for(gemms, pool), labels=labels
        )
        picks[name] = sorted(
            (
                (gemms[r.index]["m"], gemms[r.index]["n"], gemms[r.index]["k"]),
                kernel_of(identity, r.pick_sol_idx),
                r.sel_eff,
            )
            for r in out.results
        )
    assert picks["a"] == picks["b"]
    assert all(p[1] is not None for p in picks["a"])


# ── the run analysis behind the report ───────────────────────────────────────


def two_builds(world, monkeypatch):
    """round_0's rows enriched with build A, round_1's with build B (every
    index of A names another kernel in B), and B's identity."""
    lib_b = write_build(world.tmp / "lib_b", IDS_ROTATE)
    r1 = world.tmp / "r1"
    logs = write_logs(world.tmp / "b", IDS_ROTATE)
    assert run_stage04(monkeypatch, logs, lib_b, r1) == 0
    return world.rows, r1, ki.library_identity(lib_b, STEMS)


class PicklingPool:
    """ProcessPoolExecutor in this process: one worker, its initializer's
    arguments and every task's arguments and result passed through pickle
    as to and from a worker process. `crossed` collects (what, object) of
    everything that crossed in the latest pool."""

    crossed = []

    def __init__(self, max_workers=None, initializer=None, initargs=()):
        self.initializer, self.initargs = initializer, initargs
        PicklingPool.crossed = []

    def __enter__(self):
        if self.initializer is not None:
            self.initializer(*self.through("initargs", self.initargs))
        return self

    def __exit__(self, *exc):
        return False

    @classmethod
    def through(cls, what, obj):
        import pickle

        cls.crossed.append((what, obj))
        return pickle.loads(pickle.dumps(obj))

    def map(self, fn, *iterables):
        return [
            self.through("result", fn(*self.through("args", args)))
            for args in zip(*iterables)
        ]


def chunk_per_gemm(rows: Path, dst: Path) -> Path:
    """A copy of an enriched folder with each GEMM's rows in a chunk of its
    own."""
    import csv

    shutil.copytree(rows, dst)
    by_gemm = {}
    for p in sorted(dst.glob("chunk_*.csv")):
        with p.open(newline="") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames
            for r in reader:
                by_gemm.setdefault((r["m"], r["n"], r["k"], r["batch_count"]), [])
                by_gemm[(r["m"], r["n"], r["k"], r["batch_count"])].append(r)
        p.unlink()
    for i, rs in enumerate(by_gemm.values()):
        with (dst / f"chunk_{i:04d}.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            w.writerows(rs)
    return dst


@pytest.mark.parametrize("jobs", [1, 2])
def test_the_run_analysis_reads_two_builds_at_their_kernels(world, monkeypatch, jobs):
    monkeypatch.setattr(ra, "ProcessPoolExecutor", PicklingPool)
    monkeypatch.setattr(ra, "_WORKER_REMAPS", [])
    monkeypatch.setattr(PicklingPool, "crossed", [])
    r0, r1, ident_b = two_builds(world, monkeypatch)
    bench = ra.load_bench([r0, r1], jobs=jobs, identity=ident_b)
    for g in bench.gemms:
        by_kernel = {kernel_of(ident_b, s): u for s, u in zip(g.sids, g.us)}
        shape = Gemm(*(int(v) for v in g.key[:3]), int(g.key[3]))
        for k in range(len(SPECS) - 1):
            assert by_kernel[ki.kernel_id(ident_b.kernels[IDS_ROTATE[k]])] == latency(
                shape, k
            )
    for sid, cfg in bench.configs.items():
        kid = kernel_of(ident_b, sid)
        assert ki.signature_text(cfg) == kid.split(":")[0]
        assert cfg["attributes"] == ident_b.kernels[sid]["attributes"]
    assert any("hipBLASLt build changed" in n for n in bench.notes)
    with pytest.raises(ki.BuildChangedError, match="different hipBLASLt builds"):
        ra.load_bench([r0, r1], jobs=jobs)
    legacy = unrecorded(r0, world.tmp / "legacy")
    bench = ra.load_bench([legacy, r1], jobs=jobs, identity=ident_b)
    assert bench.stats["rows_kernel_not_in_library"] > 0
    assert {int(s) for g in bench.gemms for s in g.sids} == set(IDS_ROTATE)
    assert any("records no kernel identities" in n for n in bench.notes)
    with pytest.raises(ki.BuildChangedError):
        ra.load_bench([legacy, r1], identity=ident_b, strict=True)
    for planner in ("stage02", "stage01b"):
        rd = world.tmp / f"run_{planner}" / "round_0"
        (rd / planner).mkdir(parents=True)
        shutil.copy(world.plan, rd / planner / "plan.json")
        assert set(ra.round_plan(rd, ident_b).values()) == {
            frozenset(IDS_ROTATE[k] for k in (0, 2, 6, 9))
        }
        assert set(ra.round_plan(rd).values()) == {
            frozenset(IDS_A[k] for k in (0, 2, 6, 9))
        }


@pytest.mark.parametrize("jobs", [1, 2])
def test_without_an_identity_an_unrecorded_folder_must_match_the_records(
    world, monkeypatch, jobs
):
    """Without the library's identity a folder enriched before identities
    were recorded merges with recorded folders only when its kernels match
    the ones their records name under the same indices: by signature (its
    rows) and by attributes (its kernels.json)."""
    monkeypatch.setattr(ra, "ProcessPoolExecutor", PicklingPool)
    monkeypatch.setattr(ra, "_WORKER_REMAPS", [])
    monkeypatch.setattr(PicklingPool, "crossed", [])
    r0, r1, _ = two_builds(world, monkeypatch)
    legacy_a = unrecorded(r0, world.tmp / "legacy_a")
    with pytest.raises(ki.BuildChangedError, match="records no kernel identities") as e:
        ra.load_bench([legacy_a, r1], jobs=jobs)
    assert "different hipBLASLt builds" in str(e.value)
    assert f"{world.tmp / 'legacy_a'} records" in str(e.value)
    legacy_b = unrecorded(r1, world.tmp / "legacy_b")
    alone = ra.load_bench([r1], jobs=jobs)
    both = ra.load_bench([legacy_b, r1], jobs=jobs)
    assert both.notes == [] and both.stats["gemms"] == alone.stats["gemms"]
    assert [(g.key, g.sids.tolist(), g.us.tolist()) for g in both.gemms] == [
        (g.key, g.sids.tolist(), g.us.tolist()) for g in alone.gemms
    ]
    swapped = shutil.copytree(legacy_b, world.tmp / "swapped")
    doc = json.loads((swapped / KERNELS_FILE).read_text())
    a, b = str(IDS_ROTATE[2]), str(IDS_ROTATE[3])
    doc["kernels"][a], doc["kernels"][b] = doc["kernels"][b], doc["kernels"][a]
    (swapped / KERNELS_FILE).write_text(json.dumps(doc))
    with pytest.raises(ki.BuildChangedError, match="a kernel with workgroup_mapping"):
        ra.load_bench([swapped, r1], jobs=jobs)


def test_load_bench_workers_get_the_folder_remaps_once(world, monkeypatch):
    """The parallel load_bench gives each worker the folders' remaps once,
    through the executor's initializer, without the kernel ids only
    explanations read. A chunk's task carries its path and folder and
    returns what its rows showed beyond the worker's earlier chunks, never a
    remap; the parent ends with the serial path's rows, counts and notes."""
    monkeypatch.setattr(ra, "ProcessPoolExecutor", PicklingPool)
    monkeypatch.setattr(ra, "_WORKER_REMAPS", [])
    monkeypatch.setattr(PicklingPool, "crossed", [])
    bare = unrecorded(world.rows, world.tmp / "bare", keep_attributes=False)
    legacy = chunk_per_gemm(bare, world.tmp / "split")
    ident_r = ki.library_identity(write_build(world.tmp / "lib_r", IDS_ROTATE), STEMS)

    def loaded(jobs):
        b = ra.load_bench([legacy, world.rows], jobs=jobs, identity=ident_r)
        rows = sorted(
            (g.key, sorted(zip(g.sids.tolist(), g.us.tolist()))) for g in b.gemms
        )
        return rows, b.stats, b.notes, sorted(b.configs)

    serial = loaded(1)
    assert PicklingPool.crossed == []
    assert serial[1]["rows_kernel_not_in_library"] > 0 and len(serial[2]) == 2
    assert loaded(2) == serial
    crossed = PicklingPool.crossed
    ((_, (workers,)),) = [c for c in crossed if c[0] == "initargs"]
    assert [type(r) for r in workers] == [ki.Remap, ki.Remap]
    assert all(not r.kernels_then and not r.changed for r in workers)
    assert workers[0].legacy is not None and not workers[0].legacy.ids
    tasks = [o for what, o in crossed if what == "args"]
    assert tasks == [(p, 0) for p in sorted(legacy.glob("chunk_*.csv"))] + [
        (p, 1) for p in sorted(world.rows.glob("chunk_*.csv"))
    ]
    results = [o for what, o in crossed if what == "result"]
    assert all(isinstance(r[3], ki.Found) for r in results)
    assert not any(isinstance(x, ki.Remap) for r in results for x in r)
    decided = [len(r[3].differ) + len(r[3].agree) for r in results[:3]]
    assert decided == [len(SPECS), 0, 0]


def test_the_report_says_why_it_cannot_merge_two_builds(world, monkeypatch, tmp_path):
    import make_report as mr

    r0, r1, _ = two_builds(world, monkeypatch)
    run = tmp_path / "run"
    for i, src in enumerate((r0, r1)):
        shutil.copytree(src, run / f"round_{i}" / "stage04")
    notes = []
    assert mr._analysis(str(run), 1, False, notes, "warmup", False) is None
    assert notes and notes[0].startswith("bench analysis skipped:")
    assert "different hipBLASLt builds" in notes[0]


def test_the_reports_stage08_decomposition_maps_each_dataset(
    world, monkeypatch, tmp_path
):
    """The report decomposes a stage08 dataset with its folder mapped to the
    library of the run's model, and notes a dataset it cannot map."""
    import make_report as mr

    run = tmp_path / "run"
    s5 = run / "round_0" / "stage05"
    s5.mkdir(parents=True)
    (s5 / "cells.json").write_text(json.dumps({"cells": []}))
    args = {"library_dir": str(world.lib_a), "library_stem": LIBRARY_STEM}
    (s5 / "metrics.json").write_text(json.dumps({"args": dict(args, arch="gfx950")}))
    rows = [{"name": "ds", "dataset": {"path": str(world.rows), "summary": {}}}]
    seen = []
    real = ra.load_bench

    def spy(dirs, **kw):
        seen.append(kw.get("identity"))
        return real(dirs, **kw)

    monkeypatch.setattr(ra, "load_bench", spy)
    notes = []
    out = mr._stage08_decomposition(str(run), rows, 1, False, notes, "warmup")
    assert set(out) == {"ds"} and notes == []
    assert [i.digest for i in seen] == [
        ki.library_identity(world.lib_a, [LIBRARY_STEM]).digest
    ]

    def refuse(dirs, **kw):
        raise ki.BuildChangedError("cannot map")

    monkeypatch.setattr(ra, "load_bench", refuse)
    assert mr._stage08_decomposition(str(run), rows, 1, False, notes, "warmup") == {}
    assert notes == ["stage 8 decomposition of ds skipped: cannot map"]


# ── joins that refuse ────────────────────────────────────────────────────────


@pytest.mark.parametrize("planner", ["stage02", "stage01b"])
def test_the_orchestrator_does_not_bench_plans_of_another_build(world, capsys, planner):
    rd = world.tmp / "run" / "round_0"
    (rd / planner / "seed").mkdir(parents=True)
    (rd / planner / "plan.json").write_text(world.plan.read_text())
    cfg = {"planner": {"enabled": True, "bank_library_stems": [BANK_STEM]}}

    def pipeline(lib):
        p = types.SimpleNamespace(planner=True, cfg=cfg, stem=LIBRARY_STEM)
        p.paths, p.devices = types.SimpleNamespace(library_dir=lib), [0]
        p.layout = run_layout.of(rd.parent)
        p.stage_name = lambda key: ro.Pipeline.stage_name(p, key)
        p.plan_dir = lambda owner: ro.Pipeline.plan_dir(p, owner)
        p.round_plans = lambda r: ro.Pipeline.round_plans(p, r)
        return p

    assert ro.Pipeline.plan_identity_errors(pipeline(world.lib_a), rd) == []
    lib_b = write_build(world.tmp / "lib_b", IDS_SHUFFLE)
    (err,) = ro.Pipeline.plan_identity_errors(pipeline(lib_b), rd)
    assert "hipBLASLt build changed" in err and "--from-stage 2" in err
    plan = rd / planner / "plan.json"
    plan.write_text(json.dumps(plan_doc(None, IDS_A)))
    capsys.readouterr()
    assert ro.Pipeline.plan_identity_errors(pipeline(lib_b), rd) == []
    assert f"plan {plan} records no kernel identities" in capsys.readouterr().err


@pytest.mark.parametrize("rebuilt", [False, True])
def test_a_round_whose_plans_name_other_kernels_stops_before_any_bench(
    world, monkeypatch, rebuilt
):
    lib = write_build(world.tmp / "lib_b", IDS_SHUFFLE) if rebuilt else world.lib_a
    cfg = {
        "arch": "gfx950",
        "hipblaslt": {"library_stem": LIBRARY_STEM},
        "runtime": {"devices": [0]},
        "planner": {"enabled": True, "bank_library_stems": [BANK_STEM]},
    }
    paths = ro.Paths(world.tmp, world.tmp, world.tmp / "bench", lib, None, None)
    p = ro.Pipeline(
        cfg=cfg,
        snapshot=world.tmp / "cfg.yaml",
        run_dir=world.tmp / "run",
        paths=paths,
        stages_dir=world.tmp,
        timing=ro.Timing(),
        force_deploy=False,
    )
    rd = world.tmp / "run" / "round_0"
    for key in ("stage01", "stage02"):
        (rd / key).mkdir(parents=True)
        ro.mark_stage_complete(rd / key, ["stub"])
    (rd / "stage02" / "plan.json").write_text(world.plan.read_text())
    ran = []

    def run_stage(self, owner_dir, key, args, round_idx, out_dir=None):
        ran.append(key)
        return 7

    monkeypatch.setattr(ro.Pipeline, "run_stage", run_stage)
    step = ro.RoundStep(0, "initial", [], None, False)
    result = p._run_round_stages(rd, step, 3.0)
    if rebuilt:
        assert result.rc == 2 and ran == []
    else:
        assert ran == ["stage03"]
    ran.clear()
    ro.mark_stage_complete(rd / "stage03", ["stub"])
    p._run_round_stages(rd, step, 4.0)
    assert ran == ["stage04"]


def test_audit_v2_never_mixes_builds_and_maps_its_medians(tmp_path):
    ident_a = ki.library_identity(write_build(tmp_path / "a", IDS_A), STEMS)
    lib_b = write_build(tmp_path / "b", IDS_SHUFFLE)
    ident_b = ki.library_identity(lib_b, STEMS)
    shape = (256, 256, 256, 1)
    cands = [(IDS_A[0], 10.0), (IDS_A[7], 10.4), (IDS_A[4], 30.0)]
    g = {
        "m": 256,
        "n": 256,
        "k": 256,
        "batch_count": 1,
        "winner_us": 10.0,
        "candidates": [
            {"sol_idx_global": s, "us": us, "is_skip": False} for s, us in cands
        ],
    }
    gemms = {("exclusive", shape): g}
    lines = {shape: "- {M: 256, N: 256, K: 256, batch_count: 1}"}
    parts = {"exclusive": [[shape]]}
    audit = tmp_path / "audit"
    d = v2.v2_dir(audit)
    m1 = v2.Model("m1", "1" * 64, "x", {("exclusive", shape): IDS_A[7]})
    assert (
        v2.plan(
            d, gemms=gemms, lines=lines, parts=parts, new_models=[m1], identity=ident_a
        )
        == 0
    )
    m = v2.manifest(d)
    for doc in (m["canary"], v2.plans(d)[0], v2.models(d)["m1"]):
        assert ki.read_record(doc)["digest"] == ident_a.digest
    m2 = v2.Model("m2", "2" * 64, "x", {("exclusive", shape): IDS_SHUFFLE[4]})
    with pytest.raises(v2.V2Error, match="hipBLASLt build changed"):
        v2.plan(
            d, gemms=gemms, lines=lines, parts=parts, new_models=[m2], identity=ident_b
        )
    m["bench"] = {"library": {"dir": str(tmp_path / "a"), "stems": STEMS}}
    v2.check_bench_library(d, m)
    m["bench"]["library"]["dir"] = str(lib_b)
    with pytest.raises(v2.V2Error, match="hipBLASLt build changed"):
        v2.check_bench_library(d, m)
    v2._write(d / v2.MANIFEST, m)
    proc = subprocess.run(
        [
            sys.executable,
            str(ta.PIPELINE / "scripts" / "audit_v2.py"),
            "bench",
            "--audit-dir",
            str(audit),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "its bench lines would time other kernels" in proc.stdout + proc.stderr
    medians = {str(IDS_A[0]): 10.1, str(IDS_A[7]): 10.2}
    result = {
        "gemms": [
            {
                "shape": list(shape),
                "kernels": [IDS_A[0], IDS_A[7]],
                "median": medians,
                "flagged": False,
            }
        ],
    }
    v2._write(v2.part_path(d, 0, "exclusive", 0) / v2.RESULT, result)
    (meas,) = v2.measurements(d, ident_b).values()
    assert meas.median == {IDS_SHUFFLE[0]: 10.1, IDS_SHUFFLE[7]: 10.2}
    assert meas.kernels == [IDS_SHUFFLE[0], IDS_SHUFFLE[7]]
    for same in (v2.measurements(d, ident_a), v2.measurements(d)):
        assert same[("exclusive", shape)].median == {IDS_A[0]: 10.1, IDS_A[7]: 10.2}


def test_audit_v2_records_without_identities_follow_the_audit_digests(tmp_path):
    """A v2 measurement planned before kernel identities were recorded, met
    after a rebuild that gives build A's indices to other kernels: a new
    pick's index lies in an old set, where it named another kernel. By the
    audit set's kernel digests the medians are read at their kernels' new
    indices and plan refuses; with the same build nothing moves."""
    ident_a = ki.library_identity(write_build(tmp_path / "a", IDS_A), STEMS)
    ident_b = ki.library_identity(write_build(tmp_path / "b", IDS_ROTATE), STEMS)
    digests = {
        name: al.pool_digests(i.kernels.values())
        for name, i in (("a", ident_a), ("b", ident_b))
    }
    audit_m = {
        "generations": [{"kernels": {str(s): x for s, x in digests["a"].items()}}]
    }
    shape, key = (256, 256, 256, 1), ("exclusive", (256, 256, 256, 1))

    def gemm(ids, extra=()):
        rows = [(ids[0], 10.0), (ids[7], 10.4), (ids[4], 30.0), *extra]
        cands = [{"sol_idx_global": s, "us": u, "is_skip": False} for s, u in rows]
        return {
            "m": 256,
            "n": 256,
            "k": 256,
            "batch_count": 1,
            "winner_us": 10.0,
            "candidates": cands,
        }

    lines = {shape: "- {M: 256, N: 256, K: 256, batch_count: 1}"}
    parts = {"exclusive": [[shape]]}
    d = v2.v2_dir(tmp_path / "audit")
    m1 = v2.Model("m1", "1" * 64, "x", {key: IDS_A[7]})
    assert (
        v2.plan(d, gemms={key: gemm(IDS_A)}, lines=lines, parts=parts, new_models=[m1])
        == 0
    )
    assert ki.read_record(v2.plans(d)[0]) is None
    medians = {str(IDS_A[0]): 10.1, str(IDS_A[7]): 10.2}
    result = {"gemms": [{"shape": list(shape), "kernels": [IDS_A[0], IDS_A[7]]}]}
    result["gemms"][0].update(median=medians, flagged=False)
    v2._write(v2.part_path(d, 0, "exclusive", 0) / v2.RESULT, result)

    then_b = v2.AuditKernels.of(audit_m, digests["b"])
    assert IDS_ROTATE[6] == IDS_A[7]
    assert v2.measurements(d, ident_b)[key].median[IDS_A[7]] == 10.2
    meas = v2.measurements(d, ident_b, then_b)[key]
    assert meas.median == {IDS_ROTATE[0]: 10.1, IDS_ROTATE[7]: 10.2}
    assert meas.kernels == [IDS_ROTATE[0], IDS_ROTATE[7]]
    m2 = v2.Model("m2", "2" * 64, "x", {key: IDS_ROTATE[6]})
    gemms_b = {key: gemm(IDS_ROTATE, [(IDS_ROTATE[6], 12.0)])}
    with pytest.raises(v2.V2Error, match="by the audit set's kernel digests") as e:
        v2.plan(
            d,
            gemms=gemms_b,
            lines=lines,
            parts=parts,
            new_models=[m2],
            identity=ident_b,
            kernels=then_b,
        )
    first = min(IDS_A[0], IDS_A[7])
    now = IDS_ROTATE[IDS_A.index(first)]
    assert f"solution {first} named {ident_b.ids[now]}, at solution {now}" in str(
        e.value
    )
    then_a = v2.AuditKernels.of(audit_m, digests["a"])
    assert v2.measurements(d, ident_a, then_a)[key].median == {
        IDS_A[0]: 10.1,
        IDS_A[7]: 10.2,
    }
    m3 = v2.Model("m3", "3" * 64, "x", {key: IDS_A[4]})
    assert (
        v2.plan(
            d,
            gemms={key: gemm(IDS_A)},
            lines=lines,
            parts=parts,
            new_models=[m3],
            identity=ident_a,
            kernels=then_a,
        )
        == 1
    )
    twice = {
        "generations": audit_m["generations"] + [{"kernels": {str(IDS_A[0]): "x"}}]
    }
    assert v2.AuditKernels.of(twice, digests["a"]).index(IDS_A[0]) is None


def test_audit_v2_records_without_identities_follow_the_generations_of_their_plan(
    tmp_path,
):
    """Generation 0 benched X at solution 100 and Z at 101, and a v2 record
    without identities measured both there. A later increment benched a new
    kernel Y at 100, and the build now holds Y at 100, X at 101 and Z at 102.
    The record meant the generations finished before its plan: X's median
    follows X, so a pick of Z scores against X. Planned after an increment
    that benched the build's renumbered kernels, it means that build. When
    its plan time is unknown, 100 names X or Y: the GEMM is not scored, and
    plan and the bench refuse it."""
    X, Y, Z = "x" * 16, "y" * 16, "z" * 16
    shape, key = (256, 256, 256, 1), ("exclusive", (256, 256, 256, 1))
    rows = [(100, 10.0), (101, 12.0), (102, 30.0)]
    g = {"m": 256, "n": 256, "k": 256, "batch_count": 1, "winner_us": 10.0}
    g["candidates"] = [
        {"sol_idx_global": s, "us": u, "is_skip": False} for s, u in rows
    ]
    d = v2.v2_dir(tmp_path / "audit")
    m1 = v2.Model("m1", "1" * 64, "x", {key: 101})
    lines = {shape: "- {M: 256, N: 256, K: 256, batch_count: 1}"}
    parts = {"exclusive": [[shape]]}
    assert v2.plan(d, gemms={key: g}, lines=lines, parts=parts, new_models=[m1]) == 0
    m = v2.manifest(d)
    m["generations"][0]["created"] = "2000-06-01T00:00:00"
    v2._write(d / v2.MANIFEST, m)
    with pytest.raises(v2.V2Error, match="records no bench library"):
        v2.check_bench_library(d, m)
    result = {"gemms": [{"shape": list(shape), "kernels": [100, 101]}]}
    result["gemms"][0].update(median={"100": 10.0, "101": 12.0}, flagged=False)
    v2._write(v2.part_path(d, 0, "exclusive", 0) / v2.RESULT, result)
    v2.check_bench_library(d, m)

    def audit(increment, finished):
        gen0 = {"kernels": {"100": X, "101": Z}, "finished": "2000-01-01T00:00:00"}
        inc = {"kernels": increment, "finished": finished}
        return v2.AuditKernels.of(
            {"generations": [gen0, inc]}, {100: Y, 101: X, 102: Z}
        )

    after = audit({"100": Y}, "2000-07-01T00:00:00")
    meas = v2.measurements(d, None, after)[key]
    assert meas.median == {101: 10.0, 102: 12.0} and not meas.unplaced
    assert meas.winner_us / meas.median[102] == pytest.approx(10.0 / 12.0)
    assert after.at("2000-06-30T23:59:59").then == {100: X, 101: Z}
    assert after.at("2000-07-01T00:00:00").then == {100: Y, 101: Z}

    renumbered = audit({"100": Y, "101": X, "102": Z}, "2000-05-01T00:00:00")
    assert v2.measurements(d, None, renumbered)[key].median == {100: 10.0, 101: 12.0}
    v2.require_same_build(d, None, renumbered)
    with pytest.raises(v2.V2Error, match="do not name the same kernel now"):
        v2.require_same_build(d, None, after)

    del m["generations"][0]["created"]
    v2._write(d / v2.MANIFEST, m)
    meas = v2.measurements(d, None, after)[key]
    assert meas.unplaced == [100] and meas.median == {102: 12.0}
    why = "the audit generations do not tell which kernel solution 100 named"
    with pytest.raises(v2.V2Error, match=why):
        v2.require_same_build(d, None, after)
    assert after.unplaced([100, 101]) == [100]
    gone = v2.AuditKernels.of({"generations": [{"kernels": {"7": X}}]}, {8: Z})
    assert gone.index(7) is None and gone.unplaced([7]) == []
    assert "which the pool no longer holds" in gone.explain("r", [7], None)


# ── the audit set across a rebuild (fake bench) ──────────────────────────────


def build_audit(tmp_path, monkeypatch):
    """An audit set (exclusive context only) benched with the fake build;
    returns its namespace."""
    pytest.importorskip("tilewright")
    fh.unhide_devices(monkeypatch)
    kernels = fh.kernels_grid()
    root = tmp_path / "bench"
    build = root / "projects" / "hipblaslt" / "build" / "release"
    fh.write_fake_build(build, ta.ARCH, LIBRARY_STEM, kernels)
    libdir = build / "Tensile" / "library" / ta.ARCH
    monkeypatch.setenv("FAKE_LIBRARY_DIR", str(libdir))
    monkeypatch.setenv("FAKE_LIBRARY_STEM", LIBRARY_STEM)
    audit_dir = tmp_path / "audit_set"
    cfg = ta.base_config(root, audit_dir)
    cfg["contexts"] = {"list": [{"cu_budget": 0}], "holdout": []}
    cfg_path = tmp_path / "train.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))
    cmd = ["--config", str(cfg_path), "--mode", "audit", "--run-dir", str(audit_dir)]
    return types.SimpleNamespace(
        dir=audit_dir, cfg=cfg, cmd=cmd, libdir=libdir, kernels=kernels
    )


def current(a):
    return al.pool_digests(bp.config_pool(a.cfg, a.libdir, LIBRARY_STEM, 0).kernels)


def by_digest(gemms, digests):
    return {
        (g["m"], g["n"], g["k"], digests[c["sol_idx_global"]]): c["us"]
        for g in gemms
        for c in g["candidates"]
    }


@pytest.mark.parametrize(
    "change, planner",
    [("renumber", "stage02"), ("retune", "stage02"), ("renumber", "stage01b")],
)
def test_an_audit_increment_after_a_rebuild_benches_only_changed_kernels(
    tmp_path, monkeypatch, change, planner
):
    """Also for an audit set built before the planner became stage02: the
    increment reads its plan in stage01b/ and plans nothing again."""
    a = build_audit(tmp_path, monkeypatch)
    with layout_of_the_past(planner):
        assert ro.main(a.cmd) == 0
    before = current(a)
    gen0 = by_digest(al.load_gemms(a.dir, current=before), before)
    plan = json.loads((a.dir / planner / "plan.json").read_text())
    assert ki.read_record(plan) is not None
    planned = (a.dir / planner / ro.STAGE_MARKER).read_text()
    if change == "renumber":
        new = [dict(k, index=int(k["index"]) + 50) for k in a.kernels]
        new.append(solution(60, 10, mt=(64, 128, 64), mi=(16, 16, 128, 1)))
    else:
        new = [dict(k) for k in a.kernels]
        new[0] = solution(0, 0, mt=(64, 128, 64), mi=(16, 16, 128, 1))
    bf.write_logic(a.libdir / f"{LIBRARY_STEM}.dat", new)
    now = current(a)
    assert al.status(a.dir, now)[0] == "incomplete"
    assert al.unbenched(al.manifest(a.dir), now) == [60 if change == "renumber" else 0]
    assert ro.main(a.cmd) == 0
    m = al.manifest(a.dir)
    assert len(m["generations"]) == 2
    assert list(m["generations"][1]["kernels"]) == [
        "60" if change == "renumber" else "0"
    ]
    assert len({g["library_digest"] for g in m["generations"]}) == 2
    assert al.status(a.dir, now) == ("ok", None)
    gemms = al.load_gemms(a.dir, current=now)
    assert all(len(g["candidates"]) == len(now) for g in gemms)
    after = by_digest(gemms, now)
    new_digest = now[60 if change == "renumber" else 0]
    assert {k: v for k, v in after.items() if k[3] != new_digest} == {
        k: v for k, v in gen0.items() if k[3] in set(now.values())
    }
    width = 64
    for g in gemms:
        (c,) = [c for c in g["candidates"] if now[c["sol_idx_global"]] == new_digest]
        sm = new[-1 if change == "renumber" else 0]["sizeMapping"]
        assert c["us"] == pytest.approx(
            fh.kernel_time_us(g["m"], g["n"], g["k"], g["batch_count"], sm, width),
            rel=1e-5,
        )
    with pytest.raises(ValueError, match="different hipBLASLt builds"):
        al.load_gemms(a.dir)
    inc = json.loads((a.dir / "increments" / "inc_001" / "increment.json").read_text())
    assert ki.recorded_ids(ki.read_record(inc)) == {
        sid: kid
        for sid, kid in ki.library_identity(a.libdir, [LIBRARY_STEM]).ids.items()
        if sid == (60 if change == "renumber" else 0)
    }
    assert (a.dir / planner / ro.STAGE_MARKER).read_text() == planned
    assert sorted(p.name for p in a.dir.glob("stage0*")) == sorted(
        ["stage01", "stage01.log", planner, f"{planner}.log"]
    )
    assert run_layout.of(a.dir).planner == planner


@pytest.mark.parametrize("planner", ["stage02", "stage01b"])
def test_an_audit_refuses_to_bench_lines_planned_with_another_build(
    tmp_path, monkeypatch, capsys, planner
):
    a = build_audit(tmp_path, monkeypatch)
    real = ro.Pipeline.run_stage

    def interrupted(self, owner_dir, key, args, round_idx, out_dir=None):
        if Path(owner_dir).name == "part_0001":
            raise ro.Interrupted(15)
        return real(self, owner_dir, key, args, round_idx, out_dir)

    monkeypatch.setattr(ro.Pipeline, "run_stage", interrupted)
    with layout_of_the_past(planner):
        assert ro.main(a.cmd) != 0
    monkeypatch.setattr(ro.Pipeline, "run_stage", real)
    plan = ki.recorded_ids(
        ki.read_record(json.loads((a.dir / planner / "plan.json").read_text()))
    )
    shifted = [dict(k, index=int(k["index"]) + 50) for k in a.kernels]
    bf.write_logic(a.libdir / f"{LIBRARY_STEM}.dat", shifted)
    capsys.readouterr()
    assert ro.main(a.cmd) == 2
    out = capsys.readouterr()
    text = out.out + out.err
    assert "since the audit set was planned" in text
    assert f"solution 0 named {plan[0]} then and names none now" in text

    b = build_audit(tmp_path / "b", monkeypatch)
    assert ro.main(b.cmd) == 0
    grown = b.kernels + [solution(10, 10, mt=(64, 128, 64), mi=(16, 16, 128, 1))]
    bf.write_logic(b.libdir / f"{LIBRARY_STEM}.dat", grown)

    def interrupted_increment(self, owner_dir, key, args, round_idx, out_dir=None):
        if "increments" in str(owner_dir) and key == "stage03":
            raise ro.Interrupted(15)
        return real(self, owner_dir, key, args, round_idx, out_dir)

    monkeypatch.setattr(ro.Pipeline, "run_stage", interrupted_increment)
    assert ro.main(b.cmd) != 0
    monkeypatch.setattr(ro.Pipeline, "run_stage", real)
    inc = json.loads((b.dir / "increments" / "inc_001" / "increment.json").read_text())
    kid = ki.recorded_ids(ki.read_record(inc))[10]
    bf.write_logic(
        b.libdir / f"{LIBRARY_STEM}.dat",
        [dict(k, index=int(k["index"]) + 5) for k in grown],
    )
    capsys.readouterr()
    assert ro.main(b.cmd) == 2
    out = capsys.readouterr()
    text = out.out + out.err
    assert "since increment 1" in text
    assert f"solution 10 named {kid} then and names" in text


# ── identity rule tdm_th_v1 ──────────────────────────────────────────────────


def tdm_solution(index, local, tdm=3, tha=0, thb=0, thd=0, mt=(128, 128, 64)):
    """A solution like the tensor-data-mover kernels of a gfx1250 library:
    TDMInst `tdm`, the temporal hints `tha` / `thb` / `thd`, spelled in its
    name as TensileLite spells them."""
    m, n, d = mt
    s = solution(
        index,
        local,
        mt,
        mi=(16, 16, 128, 1),
        hasTemporalHint=True,
        temporalHintA=tha,
        temporalHintB=thb,
    )
    s["name"] = (
        f"Cijk_Alik_Bljk_BBS_BH_MT{m}x{n}x{d}_MI16x16x1_NTA0_NTB0_TDMI{tdm}"
        f"_TDMIM0_THn1_THA{tha}_THB{thb}_THD{thd}_THMXSA0_WGM8"
    )
    return s


def twin_build(d: Path, offset=0, tdm=3, table=(0, 1, 2)) -> ki.LibraryIdentity:
    """A pool of three kernels in table order `table` (positions in the
    file): the base, its THA/THB twin, and a variant that stores D with
    another hint, numbered from `offset`."""
    sols = [
        tdm_solution(offset, 0, tdm),
        tdm_solution(offset + 1, 1, tdm, tha=1, thb=1),
        tdm_solution(offset + 2, 2, tdm, thd=1),
    ]
    d.mkdir(parents=True, exist_ok=True)
    bf.write_logic(d / f"{LIBRARY_STEM}.dat", sols, table=[offset + t for t in table])
    return ki.library_identity(d, [LIBRARY_STEM])


def test_a_th_twin_of_a_tdm_kernel_is_one_kernel_one_key_benched_once(tmp_path):
    from lib import dat
    from lib import features as fs
    from lib import planner as pl

    def view_keys(pool):
        return {
            fs.v3_view_key(fs.config_kwargs_from_row(k), k["attributes"], ())
            for k in pool
        }

    tdm = twin_build(tmp_path / "tdm")
    plain = twin_build(tmp_path / "plain", tdm=0)
    pool = dat.load_library_kernels(tmp_path / "tdm", LIBRARY_STEM)
    before = dat.load_library_kernels(tmp_path / "plain", LIBRARY_STEM)
    assert [(k["cache_hints_a"], k["cache_hints_b"]) for k in before] == [
        (0, 0),
        (4, 4),
        (0, 0),
    ]
    assert len(view_keys(before)) == 2 and len(view_keys(pool)) == 1
    assert len(set(plain.ids.values())) == 3 and not plain.aliases
    base, twin, thd = (tdm.ids[i] for i in (0, 1, 2))
    assert base == twin != thd
    assert tdm.aliases == {1: 0} and tdm.index_of(base) == 0
    assert pl.Catalog.build(before).pool_ids == [0, 1, 2]
    cat = pl.Catalog.build(pool)
    assert cat.pool_ids == [0, 2] and cat.aliases == {1: 0}
    assert sorted(cat.kernels) == [0, 2] and list(cat.families.values()) == [[0, 2]]
    rec = ki.record(tdm, cat.pool_ids, cat.aliases)
    assert rec["rule"] == ki.IDENTITY_RULE == "tdm_th_v1"
    assert rec[ki.ALIASES] == {"1": 0}
    assert ki.read_record({ki.RECORD_KEY: rec}) == rec
    assert ki.ALIASES not in ki.record(plain, [0, 1, 2])
    assert ki.ALIASES not in ki.record(tdm, [2])


def test_the_first_member_in_pool_order_stands_for_its_alias_group(tmp_path):
    """The table orders the pool: the group's first member there is the one
    the runtime serves, the planner benches and a rebuild maps to."""
    reordered = twin_build(tmp_path / "a", table=(1, 0, 2))
    assert reordered.aliases == {0: 1} and reordered.index_of(reordered.ids[0]) == 1
    a = twin_build(tmp_path / "b")
    rec = ki.record(a)
    shifted = twin_build(tmp_path / "c", offset=40)
    remap = ki.Remap.of(rec, shifted, "rows")
    assert remap.index(0) == 40 and remap.index(1) == 40 and remap.index(2) == 42
    assert not remap.unresolved


def test_the_kernel_ids_of_kernels_without_tdm_operands_stay_as_they_were(tmp_path):
    """Names and hints of kernels the tensor data mover loads no operand
    of, whether their names spell TDMI0 or no TDMI at all, give the kernel
    id they gave before the identity rule: the signature and a hash of the
    name as written."""
    from lib import dat

    ident = ki.library_identity(write_build(tmp_path, IDS_A), STEMS)
    plain = [dat.kernel_dat_info(tdm_solution(i, i, 0, tha=i, thb=3)) for i in (1, 3)]
    for k in list(ident.kernels.values()) + plain:
        assert ki.tdm_operands(k) == ()
        assert ki.kernel_id(k) == (
            f"{ki.signature_text(k)}:{ki.name_hash(k['kernel_name'])}"
        )
        assert ki.identity_attributes(k) == k["attributes"]
    assert [(k["cache_hints_a"], k["cache_hints_b"]) for k in plain] == [
        (4, 4),
        (4, 4),
    ]
    assert not ident.aliases and not ki.alias_map(plain)


def test_alias_groups_are_only_kernels_with_tdm_operands(tmp_path):
    """Solutions named alike without a TDM-loaded operand stay separate
    kernels, as they were; with one they are an alias group."""
    from lib import dat

    same = [dict(tdm_solution(i, i, 0), name="Cijk_SAME_TDMI0") for i in (5, 6)]
    loaded = [dict(tdm_solution(i, i, 3), name="Cijk_SAME_TDMI3") for i in (7, 8)]
    pool = [dat.kernel_dat_info(s) for s in same + loaded]
    assert ki.alias_map(pool) == {8: 7}
    ident = ki.identity_of_files([(LIBRARY_STEM, pool, None)])
    assert ident.index_of(ident.ids[5]) is None and ident.index_of(ident.ids[7]) == 7


def test_a_plan_that_lists_a_group_lists_its_aliases(tmp_path):
    """A bench of an alias group's first member measured the group: with the
    library's identity a GEMM's list holds the aliases of the members it
    lists, so an evaluation passes over an alias without a row as over its
    member, and no alias is drawn as an unmeasured kernel."""
    ident = twin_build(tmp_path)
    shape = {"m": 64, "n": 64, "k": 64, "batch": 1, "kernels": [0, 2]}
    plan = {"version": 1, "shapes": [shape], ki.RECORD_KEY: ki.record(ident, [0, 2])}
    gemm = {"m": 64, "n": 64, "k": 64, "batch_count": 1, "candidates": []}
    assert ev.attach_plans([gemm], [plan], identity=ident) == 1
    assert gemm["listed"] == {0, 1, 2}
    ev.attach_plans([gemm], [plan])
    assert gemm["listed"] == {0, 2}
    assert ki.with_aliases(frozenset({2}), ident.aliases) == {2}


def test_the_audit_digests_an_alias_group_once(tmp_path):
    """The audit set benches a group's first member: the pool's digests
    leave its aliases out, which share its digest, and the digest of a
    kernel without TDM-loaded operands is what it was before the rule."""
    import hashlib

    from lib import dat
    from lib import features as fs

    twin_build(tmp_path)
    pool = dat.load_library_kernels(tmp_path, LIBRARY_STEM)
    digests = al.pool_digests(pool)
    assert sorted(digests) == [0, 2]
    assert al.kernel_digest(pool[1]) == digests[0] != digests[2]
    plain = dat.kernel_dat_info(tdm_solution(9, 0, 0, tha=1))
    body = {
        "name": plain["kernel_name"],
        "config": fs.config_kwargs_from_row(plain),
        "attributes": plain["attributes"],
    }
    old = hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    assert al.kernel_digest(plain) == old


def test_the_audit_digest_of_a_hand_written_kernel_is_what_it_was():
    """A kernel written by hand carries the launch rule's hand-written
    attribute, which its name already tells apart: its identity and audit
    digest are those it had before the launch rule read the attribute."""
    import hashlib

    from lib import dat
    from lib import features as fs

    sol = solution(9, 0, tileProcessingStrategy="StreamK")
    sol["customKernel"] = {"name": sol["name"], "generated": False}
    hand = dat.kernel_dat_info(sol)
    assert hand["attributes"][dat.HAND_WRITTEN] == 1
    before = {k: v for k, v in hand["attributes"].items() if k != dat.HAND_WRITTEN}
    assert ki.identity_attributes(hand) == before
    body = {
        "name": hand["kernel_name"],
        "config": fs.config_kwargs_from_row(hand),
        "attributes": before,
    }
    old = hashlib.sha1(json.dumps(body, sort_keys=True).encode()).hexdigest()[:16]
    assert al.kernel_digest(hand) == old
    assert ki.kernel_id(hand) == ki.kernel_id(dict(hand, attributes=before))


LIVE_DIR = os.environ.get("TILEWRIGHT_GFX950_LIBRARY_DIR")


@pytest.mark.skipif(
    not LIVE_DIR,
    reason="set TILEWRIGHT_GFX950_LIBRARY_DIR to a gfx950 Tensile library directory",
)
def test_the_identity_rule_keeps_every_kernel_id_of_a_live_gfx950_library():
    from lib import dat

    files = 0
    for path in sorted(Path(LIVE_DIR).iterdir()):
        if not dat.is_tensile_contraction_logic(path.name):
            continue
        files += 1
        data = dat.read_tensile_logic(path)
        for sol in data.get("solutions") or []:
            k = dat.kernel_dat_info(sol)
            assert ki.tdm_operands(k) == ()
            assert ki.kernel_id(k) == (
                f"{ki.signature_text(k)}:{ki.name_hash(k['kernel_name'])}"
            )
    assert files
