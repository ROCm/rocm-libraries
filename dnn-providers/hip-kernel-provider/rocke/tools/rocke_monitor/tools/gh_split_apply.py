#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""On a host: make gh's own folder hold a copy of the read login, so a gh that names no folder (a Claude session started
by hand, its agents, anything run in a shell) only reads, and the write login live only in its own folder, where rk's
guarded writes look.

  tools/gh_split_apply.py [--force]
  tools/gh_split_rollback.py [--no-probe]

  apply     before: gh's own folder (~/.config/gh, or $XDG_CONFIG_HOME/gh) = the write login; the read login's folder
            (<SECRETS_DIR>/gh-read) = the read login. After: the write login's folder (<SECRETS_DIR>/gh) = the write
            login; the read login's folder = the read login still, and gh's own folder = a copy of it.
  rollback  the other way: the copy goes, and gh's own folder has the write login again (tools/gh_split_rollback.py
            runs this file's `main` with action "rollback").
Each login's folder is the one the monitor reads it from (the login_dirs of rkmon_lib/gh.py: `rk gh-logins` shows
them); apply refuses while there is no read login (`rkmon login github_read` sets it), and a read login set later
reaches gh's own folder by rollback, then apply. Folders end 700 and files 600. Each login is probed before and after
(the probe of rkmon_lib/gh.py: a comment with no body and a branch on a commit that does not exist, each refused either
way, so nothing is made): the write login must be able to comment, the read login to make neither. A check that fails
before the move changes nothing (exit 2); one that fails after it puts the old layout back (exit 1). apply refuses while
an rk command runs (--force skips that check). rollback removes the copy only when its probe says that it reads only
(--no-probe: no probe, before or after). GH_SPLIT_REPO and GH_SPLIT_ISSUE name the probe's repo and issue (default:
config repo and gh.probe_issue). In the container the split is the write user's and needs none of this.
"""

import argparse
import functools
import os
import pathlib
import shutil
import subprocess
import sys

sys.dont_write_bytecode = True
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from rkmon_lib import config, doctor, gh, ghlogins, steps  # noqa: E402


class Refused(RuntimeError):
    pass


def say(text: str) -> None:
    print(f"gh_split: {text}", flush=True)


def default_dir() -> pathlib.Path:
    x = os.environ.get("XDG_CONFIG_HOME")
    return pathlib.Path(x) / "gh" if x else pathlib.Path.home() / ".config" / "gh"


@functools.cache
def monitor_config() -> dict:
    """The monitor's configuration, as its commands read it (config.load, unchecked)."""
    try:
        return config.load(check=False)
    except config.ConfigError as exc:
        raise Refused(f"the monitor's configuration cannot be read: {exc}") from None


def folders() -> tuple[pathlib.Path, pathlib.Path]:
    """(the read login's folder, the write login's folder): where the monitor's configuration reads each
    (gh.login_dirs)."""
    dirs = gh.login_dirs(monitor_config())
    if dirs.get("unsplit"):
        raise Refused(
            f"there is no read login yet: {doctor.gh_login_fix(gh.READ)} sets it; nothing changed"
        )
    if not dirs["read"] or not dirs["write"]:
        raise Refused(
            "the configuration knows no secrets folder (paths.secrets_dir), so the logins have no folders of "
            "their own; nothing changed"
        )
    return pathlib.Path(dirs["read"]), pathlib.Path(dirs["write"])


def target() -> tuple[str, str]:
    """(repo, issue) of the probe."""
    if os.environ.get("GH_SPLIT_REPO") and os.environ.get("GH_SPLIT_ISSUE"):
        return os.environ["GH_SPLIT_REPO"], os.environ["GH_SPLIT_ISSUE"]
    cfg = monitor_config()
    return cfg["repo"], str(gh.probe_issue(cfg))


def probe(folder: pathlib.Path, repo: str, issue: str) -> dict:
    """The login in `folder` probed (gh.probe), gh's own folder by its name (default_dir)."""
    return gh.probe(str(folder), repo, issue)


def writes(p: dict) -> bool:
    """Whether the login of probe P is a write login: it may comment."""
    return p["can_comment"] is True


def reads_only(p: dict) -> bool:
    """Whether the login of probe P is a read login: it may neither comment nor push."""
    return p["can_write"] is False


def tighten(folder: pathlib.Path) -> None:
    folder.chmod(0o700)
    for f in folder.rglob("*"):
        if f.is_file():
            f.chmod(0o600)


def apply(args) -> int:
    d = default_dir()
    if os.environ.get("GH_CONFIG_DIR"):
        raise Refused(
            f"GH_CONFIG_DIR is set in this shell ({os.environ['GH_CONFIG_DIR']}): unset it first"
        )
    rd, wr = folders()
    if not (d / gh.HOSTS_FILE).is_file():
        raise Refused(f"no gh login in {d}")
    if not (rd / gh.HOSTS_FILE).is_file():
        raise Refused(
            f"no read login in {rd}: {doctor.gh_login_fix(gh.READ)} sets it; nothing changed"
        )
    if wr.exists():
        raise Refused(f"{wr} exists already: applied before? (rollback undoes it)")
    if not args.force:
        busy = subprocess.run(
            ["pgrep", "-af", r"rkmon\.py"], capture_output=True, text=True
        ).stdout
        busy = [
            ln
            for ln in busy.splitlines()
            if "rkmon.py serve" not in ln and "pgrep" not in ln
        ]
        if busy:
            raise Refused(
                f"an rk command is running; run this between ticks (or --force): {busy[0][:160]}"
            )
    repo, issue = target()
    a, b = probe(d, repo, issue), probe(rd, repo, issue)
    if not writes(a):
        raise Refused(
            f"the login in {d} should be the write login, which may comment: {ghlogins.may(a)} ({a['why']}); "
            "nothing changed"
        )
    if not reads_only(b):
        raise Refused(
            f"the login in {rd} should be the read login, which may neither comment nor push: {ghlogins.may(b)} "
            f"({b['why']}); nothing changed"
        )
    say(f"before: {d} {ghlogins.may(a, gh.WRITE)}, {rd} {ghlogins.may(b)}")
    wr.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.move(str(d), str(wr))
    try:
        shutil.copytree(str(rd), str(d))
    except OSError:
        shutil.rmtree(str(d), ignore_errors=True)
        shutil.move(str(wr), str(d))
        raise Refused(f"copying {rd} to {d} failed; {wr} moved back") from None
    for f in (d, wr):
        tighten(f)
    c, w = probe(d, repo, issue), probe(wr, repo, issue)
    if not reads_only(c) or not writes(w):
        print(
            f"gh_split: after the move the default login ({d}) is {ghlogins.may(c)} (want read-only) and {wr} "
            f"{ghlogins.may(w)} (want one that may comment): rolling back",
            file=sys.stderr,
        )
        args.no_probe = True
        rollback(args)
        return 1
    say(
        f"after: the default login ({d}) is {ghlogins.may(c)}, a copy of {rd}'s; {wr} {ghlogins.may(w, gh.WRITE)}"
    )
    say(steps.done_line("`rk gh-logins` shows both sides from the toolkit"))
    return 0


def rollback(args) -> int:
    d = default_dir()
    rd, wr = folders()
    if not (wr / gh.HOSTS_FILE).is_file():
        raise Refused(f"no write login in {wr}: nothing to roll back")
    repo = issue = ""
    if not args.no_probe:
        repo, issue = target()
        c = probe(d, repo, issue) if d.exists() else None
        if c is not None and not reads_only(c):
            raise Refused(
                f"the login in {d} should be the read login's copy, which may neither comment nor push: "
                f"{ghlogins.may(c)} ({c['why']}); nothing changed"
            )
    if d.exists():
        shutil.rmtree(str(d))
    try:
        shutil.move(str(wr), str(d))
    except OSError:
        shutil.copytree(str(rd), str(d))
        raise Refused(
            f"moving {wr} to {d} failed; {d} has a copy of the read login again"
        ) from None
    tighten(d)
    say(steps.done_line(f"{d} has the write login again; {rd} keeps the read login"))
    if args.no_probe:
        return 0
    a, b = probe(d, repo, issue), probe(rd, repo, issue)
    say(
        f"probes: {d} {ghlogins.may(a, gh.WRITE)} (want one that may comment), {rd} {ghlogins.may(b)} "
        "(want read-only)"
    )
    if not writes(a) or not reads_only(b):
        say("WARNING: a probe is not what the layout expects")
        return 1
    return 0


def main(argv: list[str], action: str = "apply") -> int:
    ap = argparse.ArgumentParser(
        prog=f"gh_split_{action}.py", description=__doc__.split("\n\n")[0]
    )
    if action == "apply":
        ap.add_argument("--force", action="store_true")
    else:
        ap.add_argument("--no-probe", action="store_true")
    args = ap.parse_args(argv)
    try:
        return apply(args) if action == "apply" else rollback(args)
    except Refused as exc:
        print(f"gh_split_{action}: REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
