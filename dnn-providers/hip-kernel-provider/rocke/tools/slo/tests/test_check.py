# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Baseline compare, CLI exit codes, output formats, git --base."""
import json
import subprocess
import xml.etree.ElementTree as ET

import pytest
from conftest import kernel_text, src, write

import slo
import slo_check as K
import slo_measure as MS

KERNEL = "library/kernels/common/k.py"


# ---------- compare ----------


def measure(root):
    return MS.measure(root)


def test_unchanged_tree_has_no_findings(fake_rocke):
    write(fake_rocke, KERNEL, kernel_text())
    cur = measure(fake_rocke)
    base = {"files": {p: K.tracked(r) for p, r in cur["files"].items()}}
    assert K.compare(cur, base) == []


def test_worse_and_better_are_reported(fake_rocke):
    write(fake_rocke, KERNEL, kernel_text(n_raw=3, extra_fn_lines=10))
    base = {"files": {p: K.tracked(r) for p, r in measure(fake_rocke)["files"].items()}}
    write(
        fake_rocke, KERNEL, kernel_text(n_raw=6, extra_fn_lines=0)
    )  # more raw ops, shorter fn
    f = {(x["metric"], x["kind"]) for x in K.compare(measure(fake_rocke), base)}
    assert ("raw_per_100", "worse") in f
    assert ("longest_fn", "better") in f
    assert ("sloc", "better") in f


def test_new_file_is_judged_against_targets(fake_rocke):
    write(fake_rocke, KERNEL, kernel_text(n_raw=50))
    f = K.compare(measure(fake_rocke), {"files": {}})
    kinds = {(x["check"], x["kind"]) for x in f}
    assert ("SLO103", "over_target") in kinds  # 50 raw ops in ~52 sloc
    assert all(x["kind"] == "over_target" for x in f)


def test_rename_carries_the_baseline_over(fake_rocke):
    write(fake_rocke, KERNEL, kernel_text(n_raw=50))
    base = {"files": {p: K.tracked(r) for p, r in measure(fake_rocke)["files"].items()}}
    (fake_rocke / KERNEL).rename(fake_rocke / "library/kernels/common/moved.py")
    cur = measure(fake_rocke)
    assert (
        K.compare(cur, base, renamed={"library/kernels/common/moved.py": KERNEL}) == []
    )
    assert K.compare(cur, base) != []  # without the rename it looks new


def test_report_only_metric_never_enforces(fake_rocke):
    write(
        fake_rocke,
        KERNEL,
        src(
            """
        from rocke.helpers.attention import mask
        def f(b):
            mask(b, 1)
            b.add(1, 2)
    """
        ),
    )
    base = {"files": {p: K.tracked(r) for p, r in measure(fake_rocke)["files"].items()}}
    write(
        fake_rocke,
        KERNEL,
        src(
            """
        from rocke.helpers.attention import mask
        def f(b):
            b.add(1, 2)
    """
        ),
    )
    hcr = [x for x in K.compare(measure(fake_rocke), base) if x["metric"] == "hcr"]
    assert hcr and hcr[0]["kind"] == "worse" and hcr[0]["enforce"] is False


# ---------- CLI: exit codes and outputs ----------


def cli(root, *args):
    return slo.main([*args[:1], "--root", str(root), *args[1:]])


def test_exit_codes(fake_rocke, tmp_path, capsys):
    bl = tmp_path / "bl.json"
    write(fake_rocke, KERNEL, kernel_text(n_raw=3))
    assert cli(fake_rocke, "baseline", "--update", "--baseline", str(bl)) == 0
    assert cli(fake_rocke, "check", "--baseline", str(bl)) == 0
    write(fake_rocke, KERNEL, kernel_text(n_raw=9))
    assert cli(fake_rocke, "check", "--baseline", str(bl)) == 0  # warn only
    assert cli(fake_rocke, "check", "--baseline", str(bl), "--fail-on", "increase") == 1
    write(fake_rocke, KERNEL, "def broken(:\n")
    assert cli(fake_rocke, "check", "--baseline", str(bl)) == 2  # tool can't measure
    assert slo.main(["check", "--root", str(tmp_path)]) == 2  # not a rocke folder
    assert cli(fake_rocke, "baseline", "--baseline", str(bl)) == 2  # forgot --update


def test_hcr_only_change_does_not_fail_even_when_enforcing(fake_rocke, tmp_path):
    bl = tmp_path / "bl.json"
    write(
        fake_rocke,
        KERNEL,
        "from rocke.helpers.attention import mask\ndef f(b):\n    mask(b, 1)\n    mask(b, 2)\n    b.add(1, 2)\n",
    )
    cli(fake_rocke, "baseline", "--update", "--baseline", str(bl))
    write(
        fake_rocke,
        KERNEL,
        "from rocke.helpers.attention import mask\ndef f(b):\n    mask(b, 1)\n    b.add(1, 2)\n    y = 0\n",
    )
    assert cli(fake_rocke, "check", "--baseline", str(bl), "--fail-on", "increase") == 0


def test_outputs(fake_rocke, tmp_path, capsys):
    bl = tmp_path / "bl.json"
    write(fake_rocke, KERNEL, kernel_text(n_raw=3))
    cli(fake_rocke, "baseline", "--update", "--baseline", str(bl))
    write(fake_rocke, KERNEL, kernel_text(n_raw=9))
    j, x, m = tmp_path / "o.json", tmp_path / "o.xml", tmp_path / "o.md"
    capsys.readouterr()
    rc = cli(
        fake_rocke,
        "check",
        "--baseline",
        str(bl),
        "--json",
        str(j),
        "--junit",
        str(x),
        "--markdown",
        str(m),
        "--github",
        "--path-prefix",
        "rocke/",
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert (
        "::warning file=rocke/library/kernels/common/k.py,line=1,title=SLO103::" in out
    )
    assert "WARN SLO103" in out
    assert any(f["check"] == "SLO103" for f in json.loads(j.read_text())["findings"])
    suite = ET.parse(x).getroot()
    assert suite.get("failures") == "0"  # warn only: no red tests
    assert "raw_per_100" in "".join(suite.itertext())
    assert "| SLO103 |" in m.read_text()
    cli(
        fake_rocke,
        "check",
        "--baseline",
        str(bl),
        "--junit",
        str(x),
        "--fail-on",
        "increase",
    )
    assert ET.parse(x).getroot().get("failures") != "0"


# ---------- git --base ----------


def git(root, *a):
    subprocess.run(["git", "-C", str(root), *a], check=True, capture_output=True)


@pytest.fixture
def repo(fake_rocke):
    top = fake_rocke.parent
    git(top, "init", "-q", "-b", "main")
    git(top, "config", "user.email", "t@t")
    git(top, "config", "user.name", "t")
    write(fake_rocke, KERNEL, kernel_text(n_raw=3))
    write(fake_rocke, "library/kernels/common/other.py", kernel_text(n_raw=30))
    git(top, "add", "-A")
    git(top, "commit", "-qm", "base")
    return fake_rocke


def test_base_limits_check_to_changed_files(repo, tmp_path, capsys):
    import slo_git

    bl = tmp_path / "bl.json"
    write(repo, KERNEL, kernel_text(n_raw=4))
    ch = slo_git.changes(repo, "HEAD")
    assert set(ch["changed"]) == {KERNEL}
    assert ch["changed"][KERNEL] == {5}  # the one new line, by new-file number
    capsys.readouterr()
    assert cli(repo, "check", "--baseline", str(bl), "--base", "HEAD") == 0
    out = capsys.readouterr().out
    assert "other.py" not in out  # untouched file is not judged
    assert "k.py" in out  # no baseline: changed file judged as new


def test_base_detects_renames(repo):
    import slo_git

    git(repo.parent, "mv", f"rocke/{KERNEL}", "rocke/library/kernels/common/k2.py")
    ch = slo_git.changes(repo, "HEAD")
    assert ch["renamed"] == {"library/kernels/common/k2.py": KERNEL}


def test_bad_base_ref_is_a_tool_error(repo, tmp_path):
    assert (
        cli(
            repo,
            "check",
            "--base",
            "no-such-ref",
            "--baseline",
            str(tmp_path / "b.json"),
        )
        == 2
    )
