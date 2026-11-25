# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk logins`: every login and secret a deployment may need, in the secrets registry's order
(rkmon_lib/secrets_registry.py, `walk_entries`), each with what needs it among what is on (`needed_by`), its state and
the command that sets it (`set_command`). `docker/rkmon login` reads it for the login walk's list
(confirm.walk_list, which takes what needs each one from your confirmed settings on the host), offers each missing
one in turn, and ends with a doctor run.

States: SET (in place and right, or, until the one-time move ends, under an older name the move copies to its place:
`older_copy`), MISSING, WRONG (set, but it fails its check: the read login can write, the write login may not do what
it is judged by, comment, and push while something that is on turns auto-merge on or off (ghlogins.write_lacks), an
atlassian site the login does not reach, the ssh key of a host whose own key the state does not record:
host_keys.recorded, its fix `login`, which shows that key and records it once you accept it), NOT NEEDED (nothing that
is on uses it), UNKNOWN (it could not be checked here). Never a secret's value: only whether it is set, and where.

- `look(cfg, entry, environ=None)`: whether this process finds the entry in its place (secrets_registry.path, in
  ENVIRON): {"set": whether it holds it (secrets_registry.held); "where": that place, or why there is
  none; "mark": when it last changed, its file's (a login's first file's) time and inode, never its size (a secret's
  length is a part of it), "" when it is not set}.
- `here(entry, environ=None)`: whether this process reads the entry itself: every entry but, in the container, one
  the writer keeps, whose place is the writer's volume, which only the writer reads (writer.routing).
- `found(cfg, entry, environ=None)`: the same as `look`, from the one that may look: `look` for an entry this process
  reads (`here`), else the writer's answer (writer.kept), with "set" None and NOT_SAID in "where" when it gives none.
- `said(got)`: `found`'s answer in a word: set, missing, or not known.
- `writer_secrets(cfg)`: `rk writer-secrets`, the writer's answer: {name: look(...)} for each entry the writer keeps.
"""

from __future__ import annotations

import json
import pathlib

from . import secrets_registry as sr

NOT_SAID = "the writer did not say; rk writer-secrets asks it"


def look(cfg: dict, entry, environ=None) -> dict:
    """Whether this process finds the entry in its place, where, and when it last changed (the module's docstring)."""
    try:
        p = pathlib.Path(sr.path(cfg, entry, environ))
    except Exception as exc:  # noqa: BLE001 - no secrets folder known yet
        return {"set": False, "where": str(exc), "mark": ""}
    files = [p / f for f in sr.held_files(entry) or ()] or [p]
    try:
        ok = sr.held(str(p), sr.held_files(entry))
        s = files[0].stat() if ok else None
    except OSError:
        ok, s = False, None
    return {
        "set": ok,
        "where": str(p),
        "mark": f"{s.st_mtime_ns}:{s.st_ino}" if s else "",
    }


def here(entry, environ=None) -> bool:
    """Whether this process reads the entry itself (the module's docstring)."""
    from . import writer

    return entry.keeper != sr.WRITER or not writer.routing(environ)


def found(cfg: dict, entry, environ=None) -> dict:
    """`look`'s answer from the one that may look (the module's docstring)."""
    from . import writer

    if here(entry, environ):
        return look(cfg, entry, environ)
    got = writer.kept().get(entry.name)
    if isinstance(got, dict) and isinstance(got.get("set"), bool):
        return {
            "set": got["set"],
            "where": str(got.get("where") or ""),
            "mark": str(got.get("mark") or ""),
        }
    return {"set": None, "where": NOT_SAID, "mark": ""}


def said(got: dict) -> str:
    """`found`'s answer in a word: set, missing, or not known."""
    return {True: "set", False: "missing"}.get(got["set"], "not known")


def writer_secrets(cfg: dict) -> dict:
    """`rk writer-secrets`: {name: look(...)} for each entry the writer keeps (the module's docstring)."""
    return {e.name: look(cfg, e) for e in sr.entries(cfg) if e.keeper == sr.WRITER}


def _gh_sides(cfg: dict, probe) -> dict:
    if not cfg.get("repo"):
        return {}
    try:
        c, _ = probe(cfg)
        return c.get("sides") or {}
    except Exception:
        return {}


def older_copy(cfg: dict, entry, environ=None) -> str:
    """Where this process finds the entry under an older name, or in an older place, that the one-time move copies to
    its place (secrets_registry.older_places, each `copied`), in words; "" when it finds none, and once the move has
    ended in the monitor's home (one_time_move.ended): an older copy the end kept sets nothing.
    """
    from . import config, one_time_move, writer

    if one_time_move.ended(config.home()):
        return ""
    try:
        places = sr.older_places(cfg, entry, environ)
    except Exception:  # noqa: BLE001 - no secrets folder known yet
        return ""
    for older, place in places:
        if not older.copied or (
            older.keeper == sr.WRITER and writer.routing(environ)
        ):  # the writer's volume: no other user reads it
            continue
        if sr.held(place, sr.held_files(entry, older)):
            how = (
                "its older place"
                if older.name == (entry.place or entry.name)
                else f"under its older name {older.name}"
            )
            return f"{place}, {how}: the one-time move copies it to its place"
    return ""


def state(cfg: dict, entry, gh_sides: dict | None = None) -> tuple[str, str]:
    """(state, what was found) for one entry of the secrets registry. MISSING from its own place, an entry this process
    finds under an older name, or in an older place, that the one-time move copies is set, until the move ends
    (`older_copy`, state 7): SET, or WRONG as a key set in its place is (`_set`).
    """
    st, found_ = _state(cfg, entry, gh_sides)
    if st == "MISSING":
        older = older_copy(cfg, entry)
        if older:
            return _set(cfg, entry, older)
    return st, found_


def _state(cfg: dict, entry, gh_sides: dict | None = None) -> tuple[str, str]:
    """(state, what was found) for one entry, from its own place."""
    gh_side = sr.GH_SIDES.get(entry.name)
    if gh_side:
        side = gh_sides.get(gh_side) if gh_sides else None
        if not side:
            return (
                ("UNKNOWN", "no repository in the configuration yet")
                if not cfg.get("repo")
                else (
                    "MISSING",
                    "no login",
                )
            )
        login, can = side.get("login"), side.get("can_write")
        if (
            not login
        ):  # no login in that folder: GitHub answers an anonymous probe as for a read-only one
            return (
                "MISSING",
                f"no login ({side.get('why') or side.get('dir') or 'none'})",
            )
        if gh_side == "write":
            from . import ghlogins

            refused, unknown = ghlogins.write_lacks(cfg, side)
            if refused:
                return (
                    "WRONG",
                    f"{login} {ghlogins.may(side)}: {ghlogins.blocked(refused)} "
                    f"({side.get('why') or 'GitHub refused'})",
                )
            if not unknown:
                return "SET", f"{login} {ghlogins.may(side, gh_side)}"
        else:
            if can is False:
                return "SET", f"{login} reads only"
            if can is True:
                return "WRONG", f"{login or '?'} can write: log a read-only token in"
        return (
            ("MISSING", "no login")
            if not login
            else ("UNKNOWN", side.get("why") or "not known")
        )
    if entry.name == "atlassian":
        from . import atlassian, doctor, writer

        s = atlassian.status(cfg)
        if not s["found"]:
            return (
                "MISSING",
                f"the image has no atlassian CLI (fix: {doctor.image_fix()})",
            )
        if not s["logged_in"]:
            return "MISSING", "not logged in"
        site = atlassian.settings(cfg)["site"]
        if site and not s["site_ok"]:
            return (
                "WRONG",
                f"logged in, but {atlassian.site_key()} {site} does not answer "
                f"({doctor.rk_command(writer.ATLASSIAN, 'site', 'list')} names yours)",
            )
        wrong = atlassian.wrong_sites(cfg)
        if wrong:
            return "WRONG", "; ".join(wrong)
        return "SET", atlassian.login_words(s) + (f"; site {site}" if site else "")
    f = found(cfg, entry)
    where = f["where"]
    if f["set"] is None:
        return "UNKNOWN", where
    if not f["set"]:
        return "MISSING", where
    return _set(cfg, entry, where)


def _set(cfg: dict, entry, where: str) -> tuple[str, str]:
    """(state, what was found) for an entry that is set, at WHERE: SET; for the ssh key of a host that something that
    is on reaches (host_keys.host, secrets_registry.needed_by), WRONG while the state records none of the host's own
    keys, the only keys ssh trusts it by (host_keys.recorded), else SET saying they are.
    """
    from . import host_keys

    h = host_keys.host(entry.name)
    if h is None or not sr.needed_by(cfg, entry):
        return "SET", where
    if not host_keys.recorded(cfg, h):
        return "WRONG", f"the key is set; {h.words}'s own key is not recorded"
    return "SET", f"{where}; {h.words}'s own key is recorded"


def rows(cfg: dict, probe=None) -> list[dict]:
    """Each entry the walk lists, in order: {name, what, needed_by, state, found, command, host_key}. COMMAND: the
    words, after the wrapper's name, of the command that fixes it, the one that sets it but for HOST_KEY: whether it
    is the ssh key of a host whose own key is not recorded (WRONG), which `login` fixes: it shows that key and records
    it once you accept it (host_keys.record_fix)."""
    from . import ghlogins, host_keys

    probe = probe or (lambda c: ghlogins.check(c, None, force=True))
    sides = _gh_sides(cfg, probe)
    out = []
    for entry in sr.walk_entries(cfg):
        needs = sr.needed_by(cfg, entry)
        st, found = state(cfg, entry, sides)
        if not needs and st != "SET":
            st = "NOT NEEDED"
        host_key = st == "WRONG" and host_keys.host(entry.name) is not None
        out.append(
            {
                "name": entry.name,
                "what": entry.what,
                "needed_by": needs,
                "state": st,
                "found": found,
                "command": "login" if host_key else " ".join(sr.set_command(entry)),
                "host_key": host_key,
            }
        )
    return out


def text(rs: list[dict], wrapper: str = "rkmon") -> str:
    from . import host_keys

    lines = []
    for i, r in enumerate(rs, 1):
        needs = ", ".join(r["needed_by"]) or "nothing that is on"
        lines.append(
            f"{i}. {r['state']:<10} {r['name']}: {r['what']}; needed by {needs}"
        )
        if r["found"]:
            lines.append(f"   {r['found']}")
        if r["state"] in ("MISSING", "WRONG"):
            fix = (
                host_keys.record_fix(wrapper)
                if r.get("host_key")
                else f"{wrapper} {r['command']}"
            )
            lines.append(f"   fix: {fix}")
    todo = [r for r in rs if r["state"] in ("MISSING", "WRONG")]
    lines.append(
        f"\n{len(todo)} to do"
        + (
            ": " + ", ".join(r["name"] for r in todo)
            if todo
            else ": every login the settings need is in"
        )
    )
    return "\n".join(lines)


def main(cfg: dict, as_json: bool = False, wrapper: str = "") -> int:
    from . import doctor

    wrapper = wrapper or doctor.wrapper_name()
    rs = rows(cfg)
    print(json.dumps(rs, indent=1) if as_json else text(rs, wrapper))
    return 1 if any(r["state"] in ("MISSING", "WRONG") for r in rs) else 0
