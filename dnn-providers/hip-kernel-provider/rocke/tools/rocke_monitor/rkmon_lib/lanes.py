# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Where a test runs in CI, for `rk lane-of` and the fix-line premise (rkmon_lib/fixline.py): tools/lane_of.py for this
configuration, run in a PR runner's slot (it configures the tree and collects its tests: PR code), and its
CI facts (config `ci_test_tiers`, `ci_absent_modules`). And the Windows lane simulated on Linux, `rk winsim`
(tools/winsim.py): the PR's tests run as a Windows runner would run them, and the C/C++ it adds scanned for windows.h's
macro names.

The listings tools/lane_of.py makes (a commit's CTest entries) are cached in <cache>/lane_of, which the monitor writes
and the runners read: a run reads the cache, and a listing it makes goes to its slot's `out`; the monitor
keeps the one for the commit it asked about, read as data (`keep_listing`).
"""

from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

from . import gitops, paths, prcode, prtree, slots, whole_file

# The modes of what a PR's job reads are worker/perms.py's, read when first asked (paths.worker_module), never as this
# module loads: docker/rkmon's deploy path imports it (rkmon_lib/control.py, through review.py), from a package folder
# that holds no worker/.

# The Windows simulation's time box for its test runs (each file twice, plus the attribution of what it finds).
WINSIM_BUDGET_S = 300
LANE_OF_TIMEOUT_S = 900  # the fix-line premise's wait for an answer


def absent_modules(cfg: dict) -> list[str]:
    """The modules CI's test environment lacks: config ci_absent_modules."""
    return [m for m in cfg.get("ci_absent_modules") or () if m]


def _venv_python(cfg: dict) -> str:
    try:
        py = f"{paths.get(cfg, 'venv')}/bin/python"
    except (KeyError, paths.PathsError):
        return sys.executable
    return py if os.path.exists(py) else sys.executable


def lane_of_cache(cfg: dict) -> str:
    """The cache of tools/lane_of.py's listings, one JSON a commit: the monitor's, read by the runners."""
    return f"{paths.get(cfg, 'cache')}/{paths.CACHE_LANE_OF}"


def lane_of_argv(
    cfg: dict,
    targets: list[str],
    rev: str | None = None,
    collect: bool = False,
    log: str | None = None,
    as_json: bool = False,
    cache_out: str | None = None,
) -> list[str]:
    """tools/lane_of.py's command line for this configuration: the clone, the rocKE path, the checks' venv (PyYAML for
    the tiers, its pytest for --collect), the CI facts and the cache; `cache_out`, where a listing it makes goes.
    """
    py = _venv_python(cfg)
    cmd = [
        "nice",
        "-n",
        "10",
        sys.executable,
        "-I",
        "-B",
        str(paths.PACKAGE / "tools" / "lane_of.py"),
        "--repo",
        paths.get(cfg, "repo"),
        "--rocke-path",
        paths.repository_folder(cfg),
        "--python",
        py,
        "--collect-python",
        py,
        "--absent",
        ",".join(absent_modules(cfg)),
        "--cache",
        lane_of_cache(cfg),
    ]
    if cache_out:
        cmd += ["--cache-out", cache_out]
    if cfg.get("ci_test_tiers"):
        cmd += ["--ci-tiers", json.dumps(cfg["ci_test_tiers"])]
    if rev:
        cmd += ["--rev", rev]
    if collect:
        cmd.append("--collect")
    if log:
        cmd += ["--log", log]
    if as_json:
        cmd.append("--json")
    return cmd + ["--", *targets]


def lane_of_env(cfg: dict) -> dict:
    """Its environment: the checks' PATH (the venv's cmake and ctest) and the scratch folder. PR code keeps only what
    worker/pr_env.py lists of it (prcode.Slot)."""
    env = dict(os.environ)
    try:
        env["TMPDIR"] = paths.get(cfg, "tmp")
    except (KeyError, paths.PathsError):
        pass
    try:
        env["PATH"] = f"{paths.get(cfg, 'venv')}/bin:{env.get('PATH', '')}"
    except (KeyError, paths.PathsError):
        pass
    return env


def keep_listing(cache: str, sha: str, slot) -> str:
    """Keep the listing a run made for commit `sha` (`<sha>.json` in its slot's `out`; no other name is read: a listing
    for another commit is never taken from a run) in `cache`, as data: read as JSON by the result reader
    (worker/pr_results.py read_json), a JSON object, written whole (whole_file), a file of the monitor's that the
    runners read (0640). "" when kept, or when the run made none (the cache had it already); else why it was not kept.
    """
    name = f"{sha}.json"
    if not os.path.lexists(os.path.join(slot.out, name)):
        return ""
    try:
        got = slot.read_json(name)
    except OSError as exc:  # the reader's Refused: no file its rules allow, or no JSON
        return f"the listing for {sha[:10]} was not kept ({str(exc)[:160]})"
    if not isinstance(got, dict):
        return f"the listing for {sha[:10]} was not kept (not a JSON object)"
    try:
        os.makedirs(cache, exist_ok=True)
        whole_file.write(
            os.path.join(cache, name),
            json.dumps(got),
            paths.worker_module("perms").TREE_FILE_MODE,
        )
    except OSError as exc:
        return f"the listing for {sha[:10]} was not kept ({exc.strerror or exc})"
    return ""


def run_lane_of(
    cfg: dict,
    targets: list[str],
    rev: str | None = None,
    collect: bool = False,
    log: str | None = None,
    as_json: bool = False,
    capture: bool = True,
    timeout: float | None = LANE_OF_TIMEOUT_S,
    environ=None,
    run=None,
    records=None,
) -> subprocess.CompletedProcess:
    """tools/lane_of.py for the targets at `rev` (default develop) in a PR runner's slot of its own (prcode.Slot), after
    its `--fetch` (prcode.fetch_for: the blobs of the clone it reads), a CI shard `log` handed to it to read; the
    listing it makes for that commit kept in the cache (`keep_listing`; a line on stderr when it is not). Its output
    captured unless `capture` is false. Exit 3 when no slot could be taken (the lines on its stderr).
    """
    repo, cache = paths.get(cfg, "repo"), lane_of_cache(cfg)
    try:
        sha = gitops.git(
            repo, "rev-parse", "--verify", f"{rev or prtree.DEVELOP}^{{commit}}"
        ).strip()
    except gitops.GitError:
        sha = ""  # the tool says why it cannot read the commit
    prcode.fetch_for(
        lane_of_argv(cfg, targets, rev=rev),
        lane_of_env(cfg),
        timeout=timeout or prcode.FETCH_S,
    )
    try:
        with prcode.Slot(
            environ=environ, run=run, records=records, tmp=paths.get(cfg, "tmp")
        ) as s:
            argv = lane_of_argv(
                cfg,
                targets,
                rev=rev,
                collect=collect,
                log=s.give(log) if log else None,
                as_json=as_json,
                cache_out=s.out,
            )
            env = prcode.with_trees(lane_of_env(cfg), repo)
            p = s.run(argv, env=env, timeout=timeout, capture=capture)
            why = keep_listing(cache, sha, s) if sha else ""
    except prcode.NoSlot as exc:
        return subprocess.CompletedProcess([], slots.REFUSED, "", "\n".join(exc.lines))
    if why:
        print(f"lane-of: {why}", file=sys.stderr)
    return p


def lane_of(
    cfg: dict, targets: list[str], rev: str | None = None, collect: bool = False
) -> dict:
    """tools/lane_of.py's JSON answer for the targets; {"error": ...} when it gave none."""
    try:
        p = run_lane_of(cfg, targets, rev=rev, collect=collect, as_json=True)
    except (OSError, KeyError, paths.PathsError) as exc:
        return {"error": f"lane-of did not run ({exc})", "results": []}
    try:
        return json.loads(p.stdout)
    except ValueError:
        tail = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["no output"]
        return {"error": f"lane-of gave no answer ({tail[0][:160]})", "results": []}


def changed_tests(repo: str, rev: str, base: str, rocke: str) -> list[str]:
    """The test files (test_*.py, *_test.py) `rev` adds or changes against merge-base(base, rev), relative to the
    rocKE tree. prtree.NotRun or gitops.GitError when git cannot tell."""
    k = rocke.rstrip("/") + "/"
    out = []
    for c in prtree.changes(repo, prtree.merge_base(repo, base, rev), rev):
        f = c["new"] or ""
        name = f.rsplit("/", 1)[-1]
        if (
            f.startswith(k)
            and (name.startswith("test_") or name.endswith("_test.py"))
            and name.endswith(".py")
        ):
            out.append(f[len(k) :])
    return out


def winsim(
    cfg: dict,
    targets: list[str],
    rev: str,
    base: str,
    scan: bool = True,
    only: str | None = None,
    budget: float | None = None,
    run=None,
    records=None,
    environ=None,
) -> int:
    """`rk winsim`: the rocKE tree at `rev` written to a scratch folder of the monitor's that the PR runners read and
    write none of (git archive, removed after); with `scan`, the C/C++ lines `rev` adds against merge-base(base, rev)
    scanned for windows.h's macro names, a lead where the file sees a Windows header in that tree (git and the tree's
    text only, as the monitor); then tools/winsim.py runs `targets` (default: the test files `rev` adds or changes)
    plainly and under the Windows simulation, in a PR runner's slot of its own (prcode.Slot). Its exit code; 3 when the
    tree was not written or no slot could be taken.
    """
    repo, k = paths.get(cfg, "repo"), paths.repository_folder(cfg)
    run = run or subprocess.run
    tool = str(paths.PACKAGE / "tools" / "winsim.py")
    env = lane_of_env(cfg)
    tmp = paths.get(cfg, "tmp")
    os.makedirs(tmp, exist_ok=True)
    scratch = pathlib.Path(tempfile.mkdtemp(prefix="winsim_", dir=tmp))
    try:
        why = prcode.tree(scratch)  # the runners read the tree and the scan's report
        if why:
            print(f"WINSIM skip NOT RUN: the scratch folder was not made ({why})")
            return 3
        if not targets:
            try:
                targets = changed_tests(repo, rev, base, k)
            except (prtree.NotRun, gitops.GitError) as exc:
                print(f"the test files {rev[:10]} changes: not read ({str(exc)[:160]})")
                targets = []
            print(
                f"the test files {rev[:10]} adds or changes: {' '.join(targets) or 'none'}"
            )
        tree = scratch / k
        cmd = [sys.executable, "-I", "-B", tool, "run", "--tree", str(tree)]
        cmd += ["--python", _venv_python(cfg)]
        if targets or scan:
            try:
                prtree.extract(repo, rev, [k], scratch)
            except RuntimeError as exc:
                print(
                    f"WINSIM skip NOT RUN: the tree at {rev[:10]} was not written ({str(exc)[:160]})"
                )
                return 3
        if scan:
            report = scratch / "scan.txt"
            argv = [sys.executable, "-I", "-B", tool, "scan", "--repo", repo]
            argv += ["--head", rev, "--base", base, "--rocke-path", k]
            try:
                with open(report, "w", encoding="utf-8") as fh:
                    run(
                        argv + ["--tree", str(tree)],
                        stdout=fh,
                        stderr=subprocess.STDOUT,
                        env=env,
                        timeout=900,
                        stdin=subprocess.DEVNULL,
                    )
            except (OSError, subprocess.SubprocessError) as exc:
                report.write_text(f"SCAN NOT RUN: {exc}\n", encoding="utf-8")
            os.chmod(report, paths.worker_module("perms").TREE_FILE_MODE)
            cmd += ["--scan-report", str(report)]
        if only:
            cmd += ["--only", only]
        if budget is not None:
            cmd += ["--budget", str(budget)]
        try:
            with prcode.Slot(environ=environ, run=run, records=records, tmp=tmp) as s:
                return s.run(
                    ["nice", "-n", "10", *cmd, "--", *targets],
                    env=env,
                    timeout=None,
                    capture=False,
                ).returncode
        except prcode.NoSlot as exc:
            print("\n".join(exc.lines))
            return 3
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
