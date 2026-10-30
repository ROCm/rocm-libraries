# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import json
import sys
from pathlib import Path

import pytest

from lib import correctness as cx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import check_guarded_picks as cgp  # noqa: E402

SK3 = "Custom_Cijk_Ailk_Bljk_S_SK3_UserArgs_MT256x256x32_MI16x16x1_shortname0_gfx950"
OTHER = "Cijk_Ailk_Bljk_S_MT128x64x32_MI16x16x1_PKA1_SU0"


def _guards(tmp_path, free="M"):
    path = tmp_path / "S.tilewright.guards"
    path.write_text(cx.guards_text("S", {SK3: (cx.Guard(free, 4),)}))
    return path


def _table(tmp_path, rows, name_column="served_solution_name"):
    path = tmp_path / "picks.tsv"
    head = f"#shape\tm\tn\tk\tb\tcu\t{name_column}\tserved_kernel_name"
    body = [
        f"{i}\t{m}\t{n}\t64\t1\t0\t{name}\t{name}"
        for i, (m, n, name) in enumerate(rows)
    ]
    path.write_text("\n".join([head, *body]) + "\n")
    return path


@pytest.mark.parametrize("any_free_size", [False, True])
def test_picks_of_a_guarded_kernel_are_denied_below_its_guard(tmp_path, any_free_size):
    """Synthetic picks of the guarded kernel at M and N of 1 to 8: denied
    where M is below 4, or with --any-free-size where M or N is (the rule
    for rankings that read no guards sidecar); other kernels never."""
    guards = cx.read_guards(_guards(tmp_path))
    picks = [
        cgp.Pick(m, n, 64, 1, name)
        for m in range(1, 9)
        for n in range(1, 9)
        for name in (SK3, OTHER, SK3.replace("SK3", "SK3_PKA0"))
    ]
    denied = cgp.denied_picks(picks, guards, any_free_size)
    got = {(p.m, p.n) for p, _ in denied if p.name == SK3}
    want = {
        (m, n)
        for m in range(1, 9)
        for n in range(1, 9)
        if m < 4 or (any_free_size and n < 4)
    }
    assert got == want
    assert all(p.name != OTHER for p, _ in denied)
    applied = {g for _, gs in denied for g in gs}
    if any_free_size:
        assert applied <= {cx.Guard("M", 4), cx.Guard("N", 4)}
    else:
        assert applied == {cx.Guard("M", 4)}


def test_the_script_reads_pick_tables_and_writes_the_denied(tmp_path, capsys):
    guards = _guards(tmp_path)
    rows = [(2, 256, SK3), (4, 2, SK3), (8, 8, SK3), (1, 1, OTHER)]
    table = _table(tmp_path, rows)
    out = tmp_path / "denied.json"
    rc = cgp.main(
        ["--guards", str(guards), "--picks", str(table), "--denied-out", str(out)]
    )
    assert rc == 1
    assert [(d["m"], d["n"]) for d in json.loads(out.read_text())] == [(2, 256)]
    rc = cgp.main(
        ["--guards", str(guards), "--picks", str(table), "--any-free-size",
         "--denied-out", str(out)]
    )  # fmt: skip
    assert rc == 1
    doc = json.loads(out.read_text())
    assert [(d["m"], d["n"]) for d in doc] == [(2, 256), (4, 2)]
    assert doc[1]["guards"] == [
        {"free": "M", "minimum": 4},
        {"free": "N", "minimum": 4},
    ]
    printed = capsys.readouterr().out
    assert "denied: 2" in printed and "M or N" in printed
    clean = _table(tmp_path, [(8, 8, SK3), (1, 1, OTHER)], name_column="solution_name")
    assert cgp.main(["--guards", str(guards), "--picks", str(clean)]) == 0
    bad = tmp_path / "bad.tsv"
    bad.write_text("#m\tn\tk\tb\n1\t1\t1\t1\n")
    assert cgp.main(["--guards", str(guards), "--picks", str(bad)]) == 2
    malformed = tmp_path / "bad.guards"
    malformed.write_text(f"{SK3} M 1\n")
    assert cgp.main(["--guards", str(malformed), "--picks", str(clean)]) == 2


def test_the_script_reads_the_solutions_a_bench_ran(tmp_path):
    from bench_fakes import BANNER, measured_lines, preamble

    guards = _guards(tmp_path)
    lines = list(BANNER) + preamble(1)
    lines += measured_lines(0, _gemm(2, 256), 10, 20.0, (256, 256, 32))
    lines += preamble(1)
    lines += measured_lines(1, _gemm(8, 256), 11, 21.0, (256, 256, 32))
    log = tmp_path / "bench.log"
    log.write_text("\n".join(lines) + "\n")
    picks = cgp.read_bench_picks(log)
    assert [(p.m, p.n) for p in picks] == [(2, 256), (8, 256)]
    named = {p.name for p in picks}
    renamed = {n: (cx.Guard("M", 4),) for n in named}
    denied = cgp.denied_picks(picks, renamed)
    assert [(p.m, p.n) for p, _ in denied] == [(2, 256)]
    assert cgp.denied_picks(picks, cx.read_guards(guards)) == []


def _gemm(m, n):
    from bench_fakes import Gemm

    return Gemm(m, n, 64)
