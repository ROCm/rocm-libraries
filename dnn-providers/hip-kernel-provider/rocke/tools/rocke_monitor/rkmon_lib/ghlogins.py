# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Which GitHub login each side uses, and whether the split holds (`rk status`, `rk gh-logins`, `rk doctor`, every
tick); and the read login set from a token, never one that can write (`rk read-login`: set_read, which `rkmon login
github_read` runs; the Claude session may not run it, rkmon_lib/outward.py).

Up to three folders are probed (gh.probe: a comment with no body and a branch with no name, each refused either way,
so nothing is made; 422 for a login that may make it, 403 for one that may not):
- read: the read login (gh.login_dirs); it must make neither;
- write: the write login, judged by need (`write_lacks`): it must be able to comment while something that is on writes
  to GitHub (secrets_registry.needed_by: nothing does while review automation and the merge queue are in prototype),
  and to push while something that is on turns auto-merge on or off (secrets_registry.PUSH_NEEDS). One
  that may comment but not push says what that leaves: reviews post, but auto-merge cannot be turned on or off
  (`may`). In the container its folder belongs to the writer, so this side is probed through the broker (`rk gh-logins
  --side write` as the writer: writer.WRITE_PROBE);
- session: the folder the session's gh uses: gh's own rule (gh.default_dir) in the environment the session runs
  with, which names the read login's folder whenever there is one (gh.settings); probed only when it is not the read
  folder, and it must not be able to write either.
A probe runs at most once an hour (CACHE_MIN), or at once when a folder or its login file changes (a login made or
rotated); the result is kept in state/gh_logins.json. A `!! GH_LOGIN` line stands while the cached result breaks the
split, with its fix as the operator types it here (doctor.gh_login_fix). A folder with no login in it (gh.probe's
`held`) is named "none". With no split, nothing is probed and it is said once, no alarm: a configuration that names no
folders (gh's own folder for both), or one with no read login yet (gh.login_dirs "unsplit": the write login makes the
reads and the writes until `rkmon login github_read` sets the read login). The result is written whole (whole_file), at
the state's shared mode (worker/perms.py).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile

from . import gh, paths, whole_file, writer
from .store import minutes_since, now

CACHE_FILE = "gh_logins.json"
CACHE_MIN = 60


def folders(cfg: dict) -> dict:
    d = gh.login_dirs(cfg)
    if d.get("unsplit") or (not d["read"] and not d["write"]):
        return {}
    out = {"read": d["read"], "write": d["write"]}
    s = gh.default_dir(dict(os.environ, **gh.settings(cfg)[1]))
    if os.path.realpath(s) != os.path.realpath(out["read"] or s + "/x"):
        out["session"] = s
    return out


def _stamps(f: dict) -> dict:
    """Each folder's login file as (inode, mtime): a login made or rotated (`gh auth login`) changes it, and is probed
    again at once. A folder this user cannot read has no stamp."""
    out = {}
    for side, folder in f.items():
        try:
            s = os.stat(os.path.join(folder, gh.HOSTS_FILE))
            out[side] = [s.st_ino, s.st_mtime_ns]
        except OSError:
            out[side] = None
    return out


def _cached(store) -> dict | None:
    if store is None:
        return None
    try:
        c = json.loads((store.dir / CACHE_FILE).read_text())
        return c if isinstance(c, dict) and isinstance(c.get("sides"), dict) else None
    except (OSError, ValueError):
        return None


def probe_side(cfg: dict, side: str, folder: str, run=subprocess.run) -> dict:
    """One side's probe: here, or the write side through the broker when this user cannot read its folder."""
    argv = writer.routed(["gh-logins", "--side", side, "--json"])
    if side == "write" and argv and not os.access(folder, os.R_OK | os.X_OK):
        try:
            p = run(argv, capture_output=True, text=True, timeout=180)
            r = json.loads(p.stdout or "{}")
            if isinstance(r, dict) and "can_write" in r:
                r["via"] = "the broker"
                return r
            why = (p.stderr or p.stdout or f"exit {p.returncode}").strip()[:200]
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            why = str(exc)[:200]
        return {
            "dir": folder,
            "login": None,
            "http": None,
            "can_comment": None,
            "can_push": None,
            "can_write": None,
            "why": f"the broker did not answer: {why}",
        }
    return gh.probe(folder, cfg["repo"], gh.probe_issue(cfg))


def check(cfg: dict, store=None, force: bool = False) -> tuple[dict, bool]:
    """(the probe result, whether it was probed now): the cached one while it is under CACHE_MIN old, for the same
    folders and the same login files (_stamps), unless `force`. `store` None: no cache (rk doctor).
    """
    f = folders(cfg)
    stamps = _stamps(f)
    c = _cached(store)
    if (
        c
        and not force
        and c.get("folders") == f
        and c.get("stamps") == stamps
        and minutes_since(c.get("at") or "2000-01-01T00:00:00Z") < CACHE_MIN
    ):
        return c, False
    c = {
        "at": now(),
        "repo": cfg.get("repo"),
        "folders": f,
        "unsplit": gh.login_dirs(cfg).get("unsplit", ""),
        "stamps": stamps,
        "sides": {side: probe_side(cfg, side, folder) for side, folder in f.items()},
    }
    if store is not None:
        whole_file.write(
            store.dir / CACHE_FILE,
            json.dumps(c, indent=1),
            paths.worker_module("perms").SHARED_FILE_MODE,
        )
    return c, True


# Each of the probe's answers (gh.probe) as what the login may do, in a word.
ABILITY = {"can_comment": "comment", "can_push": "push"}
# What a write login that may comment but not push leaves, in plain words.
NO_PUSH = "reviews post, but auto-merge cannot be turned on or off"


def may(r: dict, side: str = "") -> str:
    """What the login of probe R (gh.probe) may do, in words: "can comment and push", "read-only" (neither), "can
    comment but not push", on the write SIDE with what that leaves (NO_PUSH), or "can push but not comment"; with an
    answer GitHub did not give, what is known."""
    c, p = r.get("can_comment"), r.get("can_push")
    if c is True and p is True:
        return "can comment and push"
    if c is False and p is False:
        return "read-only"
    if c is True and p is False:
        return "can comment but not push" + (f": {NO_PUSH}" if side == gh.WRITE else "")
    if c is False and p is True:
        return "can push but not comment"
    said = [
        (
            f"{'can' if v else 'cannot'} {ABILITY[k]}"
            if v is not None
            else f"whether it can {ABILITY[k]} is not known"
        )
        for k, v in (("can_comment", c), ("can_push", p))
    ]
    return "not known" if c is None and p is None else "; ".join(said)


def write_needs(cfg: dict) -> list:
    """The probe's answers (gh.probe's keys) the write login is judged by: can_comment, for the posting every write
    login is for, and can_push while something that is on turns auto-merge on or off (secrets_registry.PUSH_NEEDS).
    """
    from . import secrets_registry as sr

    pushes = sr.needed_by(cfg, sr.entry(cfg, "github_write"), among=sr.PUSH_NEEDS)
    return ["can_comment"] + (["can_push"] if pushes else [])


def write_lacks(cfg: dict, r: dict) -> tuple[list, list]:
    """(what the write login of probe R, gh.probe, may not do, what is not known of it): of `write_needs`' answers,
    those GitHub refused, and those it did not give."""
    needs = write_needs(cfg)
    return [k for k in needs if r.get(k) is False], [
        k for k in needs if r.get(k) is None
    ]


def blocked(refused: list) -> str:
    """What the write login leaves undone while it may not do REFUSED (write_lacks'), in words: while it may not
    comment, every write; while it may not push alone, NO_PUSH."""
    return NO_PUSH if refused == ["can_push"] else "rk post and every other write fail"


def who(r: dict) -> str:
    """A side's login in a word, from its probe (gh.probe): its name; "none" when its folder has no login; "?" when
    it is not known."""
    if r.get("login"):
        return r["login"]
    return "none" if r.get("held") is False else "?"


def _part(side: str, r: dict) -> str:
    """A side in the summary: its login, what it may do and its folder; a folder that holds no login, "none" alone."""
    does = "" if r.get("held") is False else f" {may(r, side)}"
    return f"{side} {who(r)}{does} ({r['dir']})"


def summary(c: dict) -> str:
    """One line: each side's login, what it may do, and its folder."""
    s = c["sides"]
    if not s and c.get("unsplit"):
        from . import doctor

        return (
            f"GH_LOGINS no read/write split yet: the login in {c['unsplit']} reads and writes (no read login); "
            f"{doctor.wrapper()} login github_read adds a read-only one"
        )
    if not s:
        return (
            "GH_LOGINS no read/write split: this configuration uses gh's own login for reads and writes "
            f"({doctor.wrapper()} login github_read makes the split)"
        )
    parts = [_part(side, s[side]) for side in ("read", "write", "session") if side in s]
    if "session" not in s:
        parts.append("session default = the read login")
    return f"GH_LOGINS {'; '.join(parts)}; probed {c['at']}"


def alarms(cfg: dict, c: dict) -> list[str]:
    """The `!! GH_LOGIN` lines, each with its fix as the operator types it here: the read or the session login can
    write; the write login may not do what it is judged by (write_lacks) while something that is on writes to GitHub
    (secrets_registry.needed_by: in prototype nothing does). Stable text (no time), so the tick filter
    wakes the orchestrator once while one holds."""
    from . import doctor, secrets_registry as sr

    s, out = c["sides"], []
    fixes = {
        side: doctor.gh_login_fix(gh.READ) for side in ("read", "session")
    }  # the session and its agents read with the read login
    for side, what in (
        ("read", "the read login"),
        (
            "session",
            "the session's default gh login (the orchestrator's and its agents')",
        ),
    ):
        r = s.get(side)
        if r and r.get("can_write") is True:
            out.append(
                f"!! GH_LOGIN {what} can write: {who(r)} in {r['dir']} ({r.get('why')}); "
                f"GitHub writes are not limited to rk: log it in with a read-only token ({fixes[side]})"
            )
    w = s.get("write")
    refused = write_lacks(cfg, w)[0] if w else []
    if refused and sr.needed_by(cfg, sr.entry(cfg, "github_write")):
        cannot = " or ".join(ABILITY[k] for k in refused)
        out.append(
            f"!! GH_LOGIN the write login cannot {cannot}: {who(w)} in {w['dir']} ({w.get('why')}); "
            f"{blocked(refused)} until it is fixed: {doctor.gh_login_fix(gh.WRITE)}"
        )
    return out


def notes(cfg: dict, c: dict) -> list[str]:
    """Advisory lines for a side the probe could not settle: the read and the session side while whether it can write
    is not known, the write side while an answer it is judged by is not known (write_lacks).
    """
    out = []
    for side, r in c["sides"].items():
        if side == gh.WRITE:
            unknown = write_lacks(cfg, r)[1]
        else:
            unknown = r.get("can_write") is None
        if unknown:
            out.append(f"(gh login {side} not known: {r.get('why') or 'no answer'})")
    return out


def status_lines(cfg: dict, store, force: bool = False) -> list[str]:
    """`rk status` and `rk gh-logins`: the summary, the alarms and the notes."""
    try:
        c, _ = check(cfg, store, force)
    except Exception as exc:  # noqa: BLE001 - a status read-out never fails on it
        return [f"(gh logins not checked: {str(exc)[:200]})"]
    return [summary(c)] + alarms(cfg, c) + notes(cfg, c)


def cached_lines(cfg: dict, store) -> list[str]:
    """`rk status` (local files only): the last probe's summary and alarms, as the cache holds them."""
    c = _cached(store)
    if not c:
        return [
            "(gh logins not probed yet: the next tick or `rk gh-logins` probes them)"
        ]
    return [summary(c)] + alarms(cfg, c)


def tick_lines(cfg: dict, store) -> list[str]:
    """Every tick: the summary and notes when it probed (hourly), the alarms every tick while one holds. Never fails a
    tick."""
    try:
        c, fresh = check(cfg, store)
    except Exception as exc:  # noqa: BLE001
        return [f"(gh logins not checked: {str(exc)[:200]})"]
    return (
        ([summary(c)] if fresh else [])
        + alarms(cfg, c)
        + (notes(cfg, c) if fresh else [])
    )


def side_json(cfg: dict, side: str) -> str:
    """`rk gh-logins --side SIDE --json`: one side's probe, here (the broker runs it as the write user)."""
    f = gh.login_dirs(cfg)
    return json.dumps(gh.probe(f[side], cfg["repo"], gh.probe_issue(cfg)))


def _removed(paths) -> list:
    """Each of PATHS removed, a folder whole, never through a link: (path, why) for each one that stays."""
    kept = []
    for path in paths:
        try:
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path)
            else:
                os.unlink(path)
        except FileNotFoundError:
            continue
        except OSError as exc:
            kept.append((path, exc.strerror or str(exc)))
    return kept


def _swap_in(src: str, folder: str) -> list:
    """The probed login in the folder SRC as the folder FOLDER, whole (set_read): SRC's regular files are copied into a
    new folder beside FOLDER, on its volume (`.NAME.<unique>.new`: 700, its files 600, synced), which is then renamed
    into FOLDER's place: the folder there is moved aside first (`.NAME.<unique>.old`), put back when the rename fails,
    then removed. So FOLDER holds that login alone: gh keeps an entry for each account in its folder, and a login over
    an earlier one would keep that account's token. The new and set-aside folders an earlier swap left (killed between
    these steps) are removed first. OSError when the copy or a rename fails, FOLDER then as it was; else (path, why)
    for each folder of an earlier login that could not be removed."""
    parent, name = os.path.split(folder)
    os.makedirs(parent, mode=0o700, exist_ok=True)
    kept = _removed(
        [
            os.path.join(parent, n)
            for n in os.listdir(parent)
            if n.startswith(f".{name}.") and n.endswith((".new", ".old"))
        ]
    )
    new = tempfile.mkdtemp(prefix=f".{name}.", suffix=".new", dir=parent)
    old = ""
    try:
        with os.scandir(src) as entries:
            for e in entries:
                if e.is_file(follow_symlinks=False):
                    with open(e.path, "rb") as f:
                        data = f.read()
                    fd = os.open(os.path.join(new, e.name), whole_file.FLAGS, 0o600)
                    whole_file.fill(fd, data, 0o600, durable=True)
        whole_file.sync_folder(new)
        if os.path.lexists(folder):
            # renamed onto an empty folder made for it: a name no other writer takes
            old = tempfile.mkdtemp(prefix=f".{name}.", suffix=".old", dir=parent)
            try:
                os.rename(folder, old)
            except OSError:
                os.rmdir(old)
                raise
        try:
            os.rename(new, folder)
        except OSError:
            if old:  # the earlier login, back in its place
                os.rename(old, folder)
            raise
    except BaseException:
        _removed([new])
        raise
    whole_file.sync_folder(parent)
    return kept + (_removed([old]) if old else [])


def read_token_fix(repo: str, environ=os.environ):
    """The fix of a read login whose token cannot read what the monitor reads, or can write: on GitHub, another token,
    the one github_read's made_by names (secrets_registry.READ_TOKEN, for the repository REPO), then the read login's
    own command (doctor.gh_login_fix)."""
    from . import checks, doctor, secrets_registry as sr

    make = f"on GitHub, make {sr.for_repo(sr.READ_TOKEN, repo)}"
    return checks.Steps((make, ", then", doctor.gh_login_fix(gh.READ, environ)))


def probe_not_found_fix(environ=os.environ):
    """The fix of GitHub's 404 to set_read's probe, which another token never cures: the monitor watches public
    repositories only, so the repository the deploy settings name (repository.name) is set to a public one, as GitHub
    names it, then confirmed and applied; when it is right, the issue the probe comments on (gh.probe_issue) is gone,
    and the package's maintainers name another in the repository's defaults (rkmon_lib/repository_defaults.py).
    """
    from . import checks, doctor

    named = checks.Described("the public repository as GitHub names it, <org>/<repo>")
    return checks.Steps(
        (
            *doctor.settings_parts("repo", named, environ),
            "; when repository.name is right, the probe's issue is gone: the package's maintainers name another in "
            "the repository's defaults",
        )
    )


def set_read(cfg: dict, token: str, out=None, login=None, probe=None) -> int:
    """`rk read-login`: TOKEN, a GitHub token (`rkmon login github_read` gives it on stdin), as the read login, never
    one that can write. It is logged in to a temporary folder of its own first (gh.login_with_token), where
    the probe asks whether it can write (gh.probe); only a token that reads only reaches the read login's own folder
    (secrets_registry.path: the folder its variable names, else its place; never the older login gh.login_dirs reads
    with while there is no read login), whole: the probed folder's files take that folder's place (_swap_in), with no
    second login, so no earlier account stays in it. The temporary folder goes either way. REFUSED, exit 4, nothing
    changed: no token, no repository to probe, a token GitHub does not take or does not know, one that can write, or
    GitHub's 404 (repository.name names no public repository, the only kind the monitor watches, or no probe issue,
    gh.probe_issue, is in it); for the last two, a FAIL row: for one that can write, its fix is another token, the one
    github_read's made_by names (secrets_registry.READ_TOKEN, for the repository), then the command again
    (doctor.gh_login_fix); for the 404, another token cures nothing, and its fix is probe_not_found_fix's. ERROR, exit
    1, nothing changed: whether it can write is not known (any other answer: a 5xx, no answer in time, no network, which
    running it again may get past), gh fails, there is no temporary folder to try it in, or its login cannot be put in
    place. Else `github_read: set`, with the folder and its login, then a line for each folder of an earlier login that
    could not be removed, exit 0. The token is never printed. LOGIN and PROBE stand for gh's in a test.
    """
    from . import checks, paths, secrets_registry as sr

    out = out or sys.stdout
    login, probe = login or gh.login_with_token, probe or gh.probe
    (name,) = [n for n, side in sr.GH_SIDES.items() if side == gh.READ]
    token = token.strip()

    def hidden(text: str) -> str:
        """TEXT with the token never in it (gh's words included)."""
        return text.replace(token, "(the token)") if token else text

    def said(word: str, text: str) -> int:
        """WORD and TEXT (hidden) as the outcome line: REFUSED's exit code, 4, or ERROR's, 1."""
        out.write(f"{word}: {name}: {hidden(text)}\n")
        return 4 if word == "REFUSED" else 1

    def refused(found: str, fix=None) -> int:
        """REFUSED, exit 4, nothing changed: a FAIL row, FOUND (hidden), whose fix is FIX, else another token, the
        one github_read's made_by names, for the repository, then this command again (read_token_fix).
        """
        row = checks.Check(
            name, checks.Status.FAIL, hidden(found), fix or read_token_fix(repo)
        )
        out.write(f"REFUSED: {name}: nothing changed:\n{checks.render([row])}\n")
        return 4

    if not token:
        return said("REFUSED", "no token on stdin; nothing changed")
    repo = cfg.get("repo")
    if not repo:
        return said(
            "REFUSED",
            "the configuration names no repository, where the probe asks whether the token can write; nothing "
            "changed",
        )
    try:
        folder = sr.path(cfg, sr.entry(cfg, name))
    except paths.PathsError as exc:
        return said("REFUSED", f"nothing changed: {exc}")
    try:
        own = tempfile.mkdtemp(prefix="rk-read-login-")
    except OSError as exc:
        return said(
            "ERROR",
            f"no temporary folder to try the token in ({str(exc)[:200]}); nothing changed",
        )

    def tried() -> int:
        """The token logged in to OWN and probed there; one that reads only, put in FOLDER whole (_swap_in)."""
        issue = gh.probe_issue(cfg)
        try:
            why = login(own, token)
            p = {} if why else probe(own, repo, issue)
        except (OSError, subprocess.SubprocessError) as exc:
            return said("ERROR", f"gh did not run ({str(exc)[:160]}); nothing changed")
        if why:
            return said(
                "REFUSED", f"GitHub did not take the token ({why}); nothing changed"
            )
        who, can = p.get("login"), p.get("can_write")
        if can is True:
            return refused(
                f"{who or 'the token'} can write to {repo} ({p.get('why')}); the read login must not"
            )
        if p.get("http") == 404:
            # the same answer each time, whatever the token: repository.name names no public repository (the monitor
            # watches public ones only, so a private one is a wrong name, not a wrong token), or the issue is gone
            return refused(
                f"repository.name, {repo}, names no public repository, or GitHub finds no issue {issue} in it, "
                f"which the probe comments on ({p.get('why')})",
                probe_not_found_fix(),
            )
        if can is None:
            return said(
                "ERROR",
                f"whether {who or 'the token'} can write to {repo} is not known ({p.get('why') or 'no answer'}); "
                "nothing changed",
            )
        if not who:
            return said(
                "REFUSED",
                f"GitHub does not know the token ({p.get('why') or 'no login'}); nothing changed",
            )
        try:
            kept = _swap_in(own, os.path.realpath(folder))
        except OSError as exc:
            return said(
                "ERROR",
                f"{who} reads only, but its login could not be put in {folder} ({str(exc)[:160]}); nothing "
                "changed",
            )
        out.write(f"{name}: set ({folder}; {who} reads only)\n")
        for path, why in kept:
            out.write(
                f"{name}: {path}, left by an earlier login, could not be removed ({why})\n"
            )
        return 0

    try:
        return tried()
    finally:
        shutil.rmtree(own, ignore_errors=True)
