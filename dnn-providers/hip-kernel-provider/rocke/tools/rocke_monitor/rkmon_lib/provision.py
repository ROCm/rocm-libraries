# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk provision`: the work volume set up from your confirmed settings, with no question asked.
Every `rkmon start` runs it in a one-off container, once it has put your confirmed settings in place and before the
monitor starts: the first sets the work volume up, the next finishes one cut short, and a later one puts back what has
gone missing and changes nothing that is in place, so a doctor row that finds a piece of the set-up missing gives the
fix `<wrapper> stop --drain`, then `<wrapper> start`. It is not a command for you.

Its steps, in order, each a row (checks.render) printed as soon as it is done, and shown at once (`_say`):

1. folders: the working folder's layout (deploy.layout).
2. worker tools: the package's worker tools (deploy.install_worker_tools).
3. clone: the monitor's clone of the repository and its develop worktree (deploy.clone_commands,
   deploy.clone_settings_commands, deploy.wtdev_commands), back to the clone's start date, which the internal state
   keeps: deploy.default_since, recorded there before the clone is made when the state holds none, so that a run cut
   short clones from the same date; while the one-time move is due (one_time_move.due), nothing is recorded: the move
   carries the older install's own date there. git forgets a worktree whose folder is gone before the develop worktree
   is made at its place, both under the clone's lock (gitops.clone_lock), which the monitor's fetches and stagings take,
   so that a run while the monitor uses the clone waits for them. A clone or a worktree that is there but not whole
   (deploy.clone_check finds the clone wrong, or the worktree is not checked out: a run cut short left it part-made, or
   the clone is another repository's) is removed and made again while the work volume has no record, when it is this
   command's own. Once the record is there, nothing is removed: the row is a WARN whose fix removes it by hand, then the
   stop and the start (doctor.clone_fix, doctor.remove_then_start).
4. code owners: the code-owner areas and their teams, from the clone's CODEOWNERS (`codeowner_teams`), into the
   internal state when it holds none.
5. checks' virtual environment (deploy.venv_commands): the image's, or one made in the working folder when the image
   has none, in place once its install has finished (deploy.venv_in_place): one a run cut short left is made again.
6. agentic files: the skill, the team members and the agent types, rendered with the configuration and installed in
   its Claude workspace (agentic.refresh, its first install too), unless the render installed there is this
   configuration's already, as the manifest of the install records it. Then a folder that render went into and that
   lacks one of its files (the review book, the offload passes' instructions) gets the render again, and each
   installed file that is gone is put back from it, SKILL.md and the dispatch table's rows too, while every file there
   stays as it is, edited or not (agentic.put_back). After this, the orchestrator renders them again whenever the
   configuration changes.
7. local review loop's server: the checks of the local review loop's model (models.check), from the container, with its
   key and the settings the reviews send (config.model_for(cfg, "draft"), their context and max output too): the model
   list, the test chat and the tool-call probe, on an Ollama server or an OpenAI-compatible one. What it finds goes to
   the internal state (FOUND), each replacing or clearing what was there: where the server's model list and chat route
   answer when they are not the default places (models.apply_found), and the name its list gives the model when it
   differs from the model's id (`_shown_as`); then the probe's record (PROBE). OFF without a local review loop. Once the
   record is there, the server is probed only while `_probes_again` says so; else the row says the last probe passed,
   with no request. When the last probe was of another address or model, the server is probed from the binding alone, so
   that what was found for the other one is not used; a probe of this address and model with another key, or with the
   key sent another way, uses what was found, as the server is the same. A start judges no model itself:
   config.probe_failure, the one judge (the doctor and rk features name it), reads the record. A test chat or a
   tool-call probe with no HTTP answer (a connection error, a timeout or a dropped connection: the server may be down,
   or its model still loading), which the client says (reviewer_client.NO_ANSWER), is recorded as "not checked: " and
   why, as a rate limit's is, which refuses no model; any HTTP answer is recorded as it is, so a refusal, an error or a
   model the server does not list refuses the model for its role, as a failed tool-call probe does. Each is a WARN that
   says what the doctor makes of the record: a refused model's, that the doctor fails on it, its fix what puts that
   failure right, as the doctor's reviewer row gives it (doctor.probe_fix); one not checked, that the doctor reads it
   and the next start probes the server again, its fix `<wrapper> stop --drain`, then `<wrapper> start` (a start while
   the monitor runs does nothing). A start probes it again until a probe of this address and model, with this key sent
   this way (models.key_sent), passes both (`_probes_again`); while the test chat gets no answer, the monitor starts on
   the places found before, else the default places, which the row names. The checks can take about two minutes: a line
   before them says what they are (`_checking`).

What it finds goes to the internal state through state_store; it writes no config.json. A step that is in place is left
as it is, so a run after one that finished changes nothing, nor a run after one cut short in what that one had done;
once the record is there, a step makes only what is missing and removes nothing (`_kept`). A step that fails is a FAIL
row, whose fix names what puts it right: for a failure one of your settings causes, that setting in your deploy settings
file, then `<wrapper> confirm` and `<wrapper> apply` (a clone of a repository that git does not find, or of one with no
develop: the repository's name, NO_REPOSITORY; agentic files with no operator login: the login); for any other (the
network, the disk), the start again, which runs the steps again once that is put right. The steps after it do not run,
and no record is written.

The record: `provisioned` in the internal state (RECORD), `package_version` and then `at`, written when every
step has run: when the last run finished, and with which package. It is this work volume's: `rkmon adopt` leaves an
earlier monitor's behind, and records its adoption in the section instead (ADOPTED_AT, its time: adopt.stage). A work
volume an older package set up has no record, but has the config.json that package's interview wrote, or the file the
one-time move renames it to (OLDER_SETUP, in the monitor's home): it counts as provisioned, and gets its record when
it is first read (`provisioned`), with OLDER_INSTALL true. That is the one test of whether the work volume is an older
install's (`older_setup`), and the answer is decided once, at that first read, by this package's first start: a
config.json written later on a work volume this package set up, by a tool or a person, changes it no more. An
adopted work volume is set up by the next start's run, whatever config.json adopt copied there, and its answer is
adopt's: only `rkmon adopt`, which copies an earlier monitor's config.json, sets OLDER_INSTALL again
(one_time_move.due_again).

- `provisioned(home) -> dict | None`: the record of HOME, the monitor's home: {"at", "package_version"},
  "older_install" true when it is written first for a work volume an older package set up or adopt has set it, and
  "adopted_at" once adopt has filled the volume; None while the volume is not provisioned, an adopted one too until
  the next start's run sets it up. state_store.StateError when the internal state cannot be read or written.
- `older_setup(home) -> bool`: whether HOME holds what an older package's setup left there (OLDER_SETUP).
- `older_install(home) -> bool`: whether the work volume was an older install's when this package first read it: the
  record's OLDER_INSTALL once there is a record (or OLDER_INSTALL is true); before the record, `older_setup`, as the
  first read decides it, but false on an adopted volume, whose answer is adopt's; `older_setup` too while the internal
  state cannot be read. It only reads.
- `words(rec) -> str`: the record REC in a line.
- `codeowner_teams(cfg) -> dict`: the repository's code-owner areas under the configuration's folder and their teams,
  from the clone's CODEOWNERS on develop: {area: [team, ...]}, the teams of the code-owner organization only; {}
  without a clone or a folder.
- `run(cfg, home, out=print) -> bool`: the steps on the configuration CFG for the home HOME, each row through OUT as it
  is done, then the record; whether the record is written (no step failed).
- `main(check=False, out=print) -> int`: `rk provision [--check]`, for the monitor's home (config.home()), the record
  read first (`provisioned`). With --check: the record's line (`words`), exit 0, or NOT_YET, exit 1, while there is
  none. Without: the configuration loaded (config.load), `run`, then DONE, exit 0, or exit 1 after a FAIL row.
  REFUSED, exit 4, when the configuration or the internal state cannot be read (applied.refuse: why, and the fix).
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import sys
from typing import NamedTuple

from . import (
    checks,
    config,
    deploy,
    gitops,
    image,
    one_time_move,
    paths,
    registry,
    runtime_config,
    state_store,
)
from .checks import Check, Command, Status, Steps

# What an older package's interview left in the monitor's home: its config.json, or that file renamed by the one-time
# move's end.
OLDER_SETUP = (runtime_config.CONFIG_JSON, one_time_move.MIGRATED)
# The record in the internal state: its section, the same in the file, and its names (registry STATE keys).
RECORD = "provisioned"
AT, PACKAGE_VERSION = f"{RECORD}.at", f"{RECORD}.package_version"
# The record's answer to whether the work volume was an older install's when this package first read it.
OLDER_INSTALL = f"{RECORD}.older_install"
# When rkmon adopt last filled the work volume with an earlier monitor's state (adopt.stage).
ADOPTED_AT = f"{RECORD}.adopted_at"
# The last probe of the local review loop's server (models.probe_record), a registry STATE key.
PROBE = "worker.probe"
# Step 7's row: the checks of the local review loop's model on its server.
SERVER = "local review loop's server"
NOT_YET = "not set up yet: the first start ({program} start) sets the work volume up"
DONE = "provisioned: the work volume is set up from your confirmed settings"
# Why a row leaves a piece that is there but not whole as it is, once the record is there.
REMOVES_NOTHING = "once the work volume is set up, a start removes nothing"


def _record(state: dict) -> dict | None:
    rec = state.get(RECORD) or {}
    return dict(rec) if rec.get("at") else None


def _adopted(state: dict) -> bool:
    """Whether STATE, the internal state, records an adoption (ADOPTED_AT)."""
    at = registry.get(state, ADOPTED_AT)
    return at is not registry.MISSING and bool(at)


def _kept(home) -> bool:
    """Whether the work volume of HOME is set up, its record there: a run then makes only what is missing and removes
    nothing."""
    return _record(state_store.read(home)) is not None


def _write_record(home, older: bool = False) -> dict:
    """The record, now: the package first, then OLDER_INSTALL for an older package's work volume (OLDER), the time
    last, so that a record with its time is whole."""
    from .store import now

    state_store.set(home, PACKAGE_VERSION, image.package_version(paths.PACKAGE))
    if older:
        state_store.set(home, OLDER_INSTALL, True)
    state_store.set(home, AT, now())
    return _record(state_store.read(home)) or {}


def older_setup(home) -> bool:
    """Whether HOME holds what an older package's setup left there: see the module's docstring."""
    return any(os.path.lexists(pathlib.Path(home) / name) for name in OLDER_SETUP)


def provisioned(home) -> dict | None:
    """The record of HOME, written first for an older package's work volume: see the module's docstring."""
    home = pathlib.Path(home)
    state = state_store.read(home)
    rec = _record(state)
    if rec is not None:
        return rec
    if _adopted(state) or not older_setup(home):
        return None
    return _write_record(home, older=True)


def older_install(home) -> bool:
    """Whether the work volume was an older install's when this package first read it: see the module's docstring."""
    try:
        state = state_store.read(home)
    except state_store.StateError:
        return older_setup(home)
    if registry.get(state, OLDER_INSTALL) is True:
        return True
    return _record(state) is None and not _adopted(state) and older_setup(home)


def words(rec: dict) -> str:
    """The record REC in a line."""
    return (
        f"provisioned at {rec.get('at')} (package {rec.get('package_version') or '?'})"
    )


def _program() -> str:
    # a call-time import: doctor is a large module
    from .doctor import wrapper_name

    return wrapper_name()


def _again() -> Command:
    """A failed step's fix: the start again, which runs the steps again (those in place stay as they are)."""
    return Command((_program(), "start"))


def _settings_fix(internal: str) -> Steps:
    """A failed step's fix when one of your settings causes it: the configuration key INTERNAL put right in your deploy
    settings file (where its value comes from), then `<wrapper> confirm` and `<wrapper> apply` (doctor.settings_parts,
    which ends as doctor.apply_commands does).
    """
    # Call-time imports: doctor is a large module, and settings_file reads the schemas.
    from . import deploy_tree, doctor, settings_file

    value = settings_file.source_of(deploy_tree.node(deploy_tree.name(internal))[1])
    return Steps(doctor.settings_parts(internal, value))


class _Cause(NamedTuple):
    """A cause of a step's failure that one of your settings puts right: what the failed command says (PATTERN), the
    cause in plain words (WORDS), and the configuration key whose value causes it (KEY).
    """

    pattern: re.Pattern
    words: str
    key: str


# The clone's: git finds no repository at the address the repository's name gives (GitHub asks a clone with no login
# for one where it shows none; a path or a file:// address with none), or one with no develop branch.
NO_REPOSITORY = _Cause(
    re.compile(
        r"repository not found|repository '[^']*' (?:not found|does not exist)|could not read username"
        r"|does not appear to be a git repository|remote branch develop not found",
        re.IGNORECASE,
    ),
    "git finds no such repository, or one with no develop branch",
    "repo",
)


# --- the code owners --------------------------------------------------------------------------------------------------


def codeowner_teams(cfg: dict) -> dict:
    """The code-owner areas and their teams: see the module's docstring."""
    k = paths.repository_folder(cfg)
    repo = paths.resolve(cfg)["repo"]
    if not k or not repo:
        return {}
    p = subprocess.run(
        ["git", "-C", repo, "show", "refs/remotes/origin/develop:.github/CODEOWNERS"],
        capture_output=True,
        text=True,
    )
    if p.returncode:
        return {}
    org = (cfg.get("codeowner_org") or (cfg.get("repo") or "").split("/")[0]).lower()
    out = {}
    for line in p.stdout.splitlines():
        parts = line.split("#", 1)[0].split()
        if len(parts) < 2 or not parts[0].strip("/").startswith(k):
            continue
        rest = parts[0].strip("/")[len(k) :].strip("/")
        area = rest.split("/")[0] if rest else "root"
        teams = [
            t.split("/", 1)[1]
            for t in parts[1:]
            if t.startswith("@") and "/" in t and t[1:].split("/")[0].lower() == org
        ]
        if teams:
            out[area] = sorted(set(out.get(area, [])) | set(teams))
    return out


# --- the steps --------------------------------------------------------------------------------------------------------


class _Failed(Exception):
    """A step failed: its FAIL row."""

    def __init__(self, row: Check):
        super().__init__(row.found)
        self.row = row


def _run(name: str, what: str, cmds: list, causes: tuple = ()) -> None:
    """CMDS, run (deploy.run); a failure is the step NAME's FAIL row, WHAT not done: its fix is the setting of the
    first of CAUSES whose pattern the failure's words match, else the start again."""
    try:
        deploy.run(cmds)
    except RuntimeError as exc:
        cause = next((c for c in causes if c.pattern.search(str(exc))), None)
        found = f"{what}: {cause.words}: {exc}" if cause else f"{what}: {exc}"
        fix = _settings_fix(cause.key) if cause else _again()
        raise _Failed(Check(name, Status.FAIL, found, fix)) from None


def _folders(cfg: dict, home) -> Check:
    made = [line for line in deploy.layout(cfg) if line.startswith("mkdir ")]
    where = paths.get(cfg, "working_dir")
    return Check(
        "folders",
        Status.PASS,
        f"{len(made)} made in {where}" if made else f"in place in {where}",
    )


def _worker_tools(cfg: dict, home) -> Check:
    dest = paths.get(cfg, "worker_tools")
    stale = deploy.stale_worker_tools(cfg)
    if not stale:
        return Check("worker tools", Status.PASS, f"in place in {dest}")
    deploy.install_worker_tools(cfg)
    return Check("worker tools", Status.PASS, f"installed in {dest}")


def _git_ok(*args) -> bool:
    return (
        subprocess.run(
            ["git", *map(str, args)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def _checked_out(wt: pathlib.Path) -> bool:
    """WT is a worktree whose checkout finished: its index is its HEAD's."""
    return (wt / ".git").is_file() and _git_ok(
        "-C", wt, "diff-index", "--quiet", "--cached", "HEAD", "--"
    )


def _clone(cfg: dict, home) -> Check:
    since = cfg.get("clone_since") or ""
    if not since:
        since = deploy.default_since()
        # recorded before the clone is made, so that a run cut short clones from the same date; not while the one-time
        # move is due, which carries the older install's own date into the internal state
        if not one_time_move.due(home):
            state_store.set(home, "clone_since", since)
    cfg = dict(cfg, clone_since=since)
    repo = pathlib.Path(paths.get(cfg, "repo"))
    wt = pathlib.Path(paths.worktree(cfg, "wtdev"))
    url = deploy.clone_url(cfg)
    bad = deploy.clone_check(cfg)
    kept = _kept(home)
    # A clone or a worktree that is there but not whole is this command's own on a volume with no record, made again;
    # once the record is there, it stays as it is, and its row's fix removes it by hand.
    if bad is not None and os.path.lexists(repo):
        if kept:
            from . import doctor

            return Check(
                "clone",
                Status.WARN,
                f"{repo}: {bad}; {REMOVES_NOTHING}",
                doctor.clone_fix(cfg),
            )
        shutil.rmtree(repo)
    if bad is None:
        _run("clone", f"the settings of {repo}", deploy.clone_settings_commands(cfg))
        if _checked_out(wt):
            return Check(
                "clone",
                Status.PASS,
                f"in place: {repo}, back to {since}, with its develop worktree {wt}",
            )
        clone = f"{repo} in place, back to {since}"
    else:
        _run(
            "clone",
            f"{url} not cloned into {repo}",
            deploy.clone_commands(cfg, url=url),
            (NO_REPOSITORY,),
        )
        clone = f"{url} cloned into {repo}, back to {since}"
    if os.path.lexists(wt):  # its checkout did not finish, or its clone is gone
        if kept:
            from . import doctor

            return Check(
                "clone",
                Status.WARN,
                f"{clone}, but the develop worktree {wt} there is not a checkout of it; {REMOVES_NOTHING}",
                doctor.remove_then_start(f"the worktree {wt}"),
            )
        shutil.rmtree(wt)
    # git keeps a worktree's record when its folder is gone: forgotten first, so that the develop worktree can be made
    # at its place again. Both hold the clone's lock, which the monitor's fetches and stagings take: a run while the
    # monitor uses the clone waits for them.
    with gitops.clone_lock(str(repo)):
        _run(
            "clone",
            f"the worktrees of {repo}",
            [["git", "-C", str(repo), "worktree", "prune"]],
        )
        _run("clone", f"the develop worktree {wt} not made", deploy.wtdev_commands(cfg))
    return Check("clone", Status.PASS, f"{clone}, with its develop worktree {wt} made")


def _code_owners(cfg: dict, home) -> Check:
    have = cfg.get("codeowner_teams") or {}
    if any(have.values()):
        return Check(
            "code owners",
            Status.PASS,
            f"in the internal state: {len(have)} code-owner areas",
        )
    teams = codeowner_teams(cfg)
    if not teams:
        return Check(
            "code owners",
            Status.PASS,
            f"the clone's CODEOWNERS names no team for {paths.repository_folder(cfg)}",
        )
    state_store.set(home, "codeowner_teams", teams)
    shown = "; ".join(f"{a}: {', '.join(t)}" for a, t in teams.items())
    return Check("code owners", Status.PASS, f"from CODEOWNERS, recorded: {shown}")


def _venv(cfg: dict, home) -> Check:
    name = "checks' virtual environment"
    venv = paths.get(cfg, "venv")
    cmds = deploy.venv_commands(cfg, "checks")
    if not cmds:
        return Check(name, Status.PASS, f"in place: {venv}")
    _run(name, f"{venv} not made", cmds)
    return Check(name, Status.PASS, f"made: {venv}")


def _agentic(cfg: dict, home) -> Check:
    from . import agentic, wrapping

    name = "agentic files"
    if not (paths.PACKAGE / "agentic" / "skills").is_dir():
        return Check(name, Status.OFF, "this package has none (agentic/skills/)")
    ws = str(cfg.get("claude_workspace") or "").strip()
    if not ws:
        return Check(name, Status.OFF, "no Claude workspace is set (claude_workspace)")
    claude = pathlib.Path(os.path.expanduser(ws)) / ".claude"
    out = pathlib.Path(paths.get(cfg, "agentic"))
    try:
        done = agentic.refresh(cfg, first=True)
        # The render installed is this configuration's, but its folder may lack a file of it that is read from there
        # (the review book, the offload passes' instructions): rendered again there, as agentic.refresh renders it.
        bundle, team = paths.PACKAGE / "agentic", agentic.team_skills()
        gone = (
            []
            if done
            else [
                r
                for r, _src in agentic.sources(bundle, team)
                if not (out / r).is_file()
            ]
        )
        if gone:
            agentic.render(bundle, config.normalize(cfg), out, team)
        # And the workspace may lack a file installed from it: put back, every file there left as it is.
        back = [] if done else agentic.put_back(out, claude.parent)
    except ValueError as exc:
        if exc.args == (agentic.LOGIN_MISSING,):
            raise _Failed(
                Check(
                    name,
                    Status.FAIL,
                    "not rendered: the skill names you by your GitHub username, which the settings do not give",
                    _settings_fix("operator.github_login"),
                )
            ) from None
        raise _Failed(
            Check(name, Status.FAIL, f"not rendered: {exc}", _again())
        ) from None
    if done:
        return Check(name, Status.PASS, f"{done[0]}, and installed in {claude}")

    def few(rels: list) -> str:
        return wrapping.joined(rels) if len(rels) <= 3 else f"{len(rels)} of its files"

    found = f"put back in {claude}: {few(back)}" if back else f"in place in {claude}"
    if gone:
        found += f"; rendered again into {out}, which lacked {few(gone)}"
    return Check(name, Status.PASS, found)


# What step 7 finds out about the local review loop's server besides the probe's record, each a registry STATE key
# (worker.NAME), and what it is in words, for step 7's row.
FOUND = {
    "models_url": "where its model list answers",
    "chat_path": "its chat route",
    "model_name": "the name its model list gives the model",
}


def _last_probe(w: dict) -> dict:
    """The last probe's record (PROBE) in the local review loop's settings W: {} for none."""
    last = w.get("probe")
    return last if isinstance(last, dict) else {}


def _of_binding(w: dict, last: dict) -> bool:
    """Whether LAST, a probe's record, is of the address and the model of W, the local review loop's settings."""
    return (last.get("url"), last.get("model")) == (w["url"], w["model"])


def _probes_again(cfg: dict) -> bool:
    """Whether a start probes the local review loop's server again once the work volume is set up: while its last probe
    (PROBE) is missing or of another address or model, did not pass both the test chat and the tool-call probe, or sent
    another key, or sent it another way (models.key_sent), than the settings the probe sends (`_worker`).
    """
    from . import bindings, models

    if not bindings.local_worker_set(cfg):
        return False
    w = _worker(cfg)
    last = _last_probe(w)
    return not (
        _of_binding(w, last)
        and last.get("chat") == last.get("tools") == "ok"
        and last.get("key_sent") == models.key_sent(w)
    )


def _probes(cfg: dict, home) -> bool:
    """Whether this run probes the local review loop's server: always while the work volume of HOME has no record, else
    while `_probes_again` says so."""
    return not _kept(home) or _probes_again(cfg)


def _shown_as(asked: dict, rec: dict) -> str:
    """The name the server's model list gives the model, when it differs from the model's id (worker.model_name), for
    ASKED, the settings probed, and REC, the probe's record, which has the place where the list answered for an
    OpenAI-compatible server only (an Ollama server names a model by its id): ASKED's own when it has none, or the list
    does not answer."""
    from . import models

    if rec.get("models_url"):
        listed, err, _where = models.openai_list(
            dict(asked, models_url=rec["models_url"])
        )
        if not err:
            return next(
                (
                    m["display"]
                    for m in listed
                    if m["name"] == asked["model"] and m["display"] != m["name"]
                ),
                "",
            )
    return asked.get("model_name") or ""


def _worker(cfg: dict) -> dict:
    """The local review loop's settings as the reviews send them (config.env's RKMON_WORKER_JSON), their context and max
    output among them, and beside them the last probe's record (PROBE), which no review gets: step 7's checks send the
    settings and compare the record."""
    draft = config.model_for(cfg, "draft")
    return (
        {k: v for k, v in draft.items() if k != "by"}
        if draft["by"] == "worker"
        else config.worker(cfg)
    )


def _checking(cfg: dict, home) -> str:
    """The line step 7 prints before its checks, which can take about two minutes: what they are, of which model, where;
    "" without a local review loop, or when this run does not probe it (`_probes`)."""
    from . import bindings

    if not bindings.local_worker_set(cfg) or not _probes(cfg, home):
        return ""
    w = _worker(cfg)
    return (
        f"{SERVER}: checking {w['model']} at {config.safe_url(w['url'])} from the container: its model list, a test "
        "chat and a tool-call probe, which can take about two minutes"
    )


def _worker_server(cfg: dict, home) -> Check:
    from . import bindings, doctor, models, steps, wrapping

    name = SERVER
    if not bindings.local_worker_set(cfg):
        return Check(name, Status.OFF, "no local review loop is set up")
    w = _worker(cfg)
    where = config.safe_url(w["url"])
    if not _probes(cfg, home):
        return Check(
            name,
            Status.PASS,
            f"in place: {where}: {w['model']} passed the test chat and the tool-call probe at its last probe",
        )
    last = _last_probe(w)
    # What a probe of another address or model found is that one's: the server is probed from the binding alone.
    asked = (
        dict(w, **dict.fromkeys(FOUND, "")) if last and not _of_binding(w, last) else w
    )
    rec = models.check(asked)
    found = dict(models.apply_found(asked, rec), model_name=_shown_as(asked, rec))
    recorded = [key for key in FOUND if (found.get(key) or "") != (w.get(key) or "")]
    for key in recorded:
        state_store.set(home, f"worker.{key}", found.get(key) or "")
    lc = models.client()
    if rec["chat"].startswith(lc.NO_ANSWER):
        # A test chat with no HTTP answer judges nothing (the server may be down, or its model still loading): it is
        # recorded as not checked, which config.probe_failure never refuses. An HTTP answer is recorded as it is.
        rec["chat"] = f"{lc.NOT_CHECKED}: {rec['chat']}"
    elif rec["tools"].startswith(lc.NO_ANSWER):
        # So is a tool-call probe with no HTTP answer.
        rec["tools"] = f"{lc.NOT_CHECKED}: {rec['tools']}"
    # Last, so that a probe recorded as answered has what it found recorded before it.
    state_store.set(home, PROBE, rec)
    program = _program()
    refused = config.probe_failure(dict(w, probe=rec))
    if refused:
        # A refused model: what puts its failure right, as the doctor's reviewer row gives it.
        judged = f"so the reviews refuse {w['model']} until a probe passes, and {program} doctor fails on this record"
        fix = doctor.probe_fix(cfg, dict(w, probe=rec))
    else:
        # Not checked: a start after a stop probes it again (_probes_again).
        judged = f"that refuses no model, and {program} doctor reads this record; the next start probes it again"
        fix = Steps(steps.restart_parts(program))
    if rec["chat"] != "ok":
        # What stands in for what was not found: what was found before, else the default places.
        stand_in = (
            "; until the test chat is answered, the monitor takes its model list to be at "
            f"{config.safe_url(lc.models_urls(found)[0])} and its chat route to be {lc.chat_path(found)}"
            if not refused and asked["backend"] == "openai"
            else ""
        )
        return Check(
            name, Status.WARN, f"{where}: {rec['chat']}; {judged}{stand_in}", fix
        )
    places = (
        f"; its model list answers at {config.safe_url(rec['models_url'])}, and its chat route at {rec['chat_path']}"
        if asked["backend"] == "openai"
        else ""
    ) + (
        f"; recorded: {wrapping.joined(FOUND[key] for key in recorded)}"
        if recorded
        else ""
    )
    if rec["tools"] != "ok":
        return Check(
            name,
            Status.WARN,
            f"{where}: {w['model']} answers the test chat, but not the tool-call probe: {rec['tools']}{places}; "
            f"{judged}",
            fix,
        )
    return Check(
        name,
        Status.PASS,
        f"{where}: {w['model']} answers the test chat and the tool-call probe{places}",
    )


STEPS = (
    _folders,
    _worker_tools,
    _clone,
    _code_owners,
    _venv,
    _agentic,
    _worker_server,
)


# The line a step that can take minutes prints before it starts, from the configuration and the home ("" for none).
FIRST = {_worker_server: _checking}


def _say(out, text: str) -> None:
    """TEXT through OUT, then stdout flushed, so that it shows at once when stdout is a pipe: rkmon start runs rk
    provision through `docker compose run -T` when it is not on a terminal."""
    out(text)
    sys.stdout.flush()


def _step(step, cfg: dict, home, out) -> bool:
    """STEP on the configuration CFG for the home HOME, after the line it prints first (FIRST), its row through OUT,
    each shown at once (`_say`): whether it did not fail."""
    first = FIRST[step](cfg, home) if step in FIRST else ""
    if first:
        _say(out, first)
    try:
        row = step(cfg, home)
    except _Failed as exc:
        _say(out, checks.render([exc.row]))
        return False
    except OSError as exc:
        name = step.__name__.strip("_").replace("_", " ")
        _say(out, checks.render([Check(name, Status.FAIL, f"{exc}", _again())]))
        return False
    _say(out, checks.render([row]))
    return True


def run(cfg: dict, home, out=print) -> bool:
    """The steps, then the record: see the module's docstring."""
    if not all(_step(step, cfg, home, out) for step in STEPS):
        return False
    _write_record(home)
    return True


def main(check: bool = False, out=print) -> int:
    """`rk provision [--check]`: see the module's docstring."""
    from . import applied  # a call-time import: only a refusal and a run read it

    home = config.home()
    try:
        # read first: a work volume an older package set up gets its record here, so that the run removes nothing
        rec = provisioned(home)
        if check:
            out(words(rec) if rec is not None else NOT_YET.format(program=_program()))
            return 0 if rec is not None else 1
        cfg = config.load()
        if not run(cfg, home, out):
            return 1
    except config.ConfigError as exc:
        return applied.refuse(exc, out)
    out(DONE)
    return 0
