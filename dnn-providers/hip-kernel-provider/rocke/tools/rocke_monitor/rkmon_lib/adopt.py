# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Moving from an earlier monitor: its state into this deployment, with nothing machine-specific in the code.

  rkmon adopt --from DIR [--map OLD=NEW]... [--dry-run] [--yes]

DIR is a copy of the earlier monitor's working folder: its container's /work (`docker cp NAME:/work DIR`), or a host
install's <WORKING_DIR>. Its configuration (DIR/monitor/config.json) names where each of its folders is (paths.*); a
folder it does not name is the usual one under DIR. Three steps, each its own function:

- stage (on the host, reading the earlier monitor only): what moves is copied into a staging folder, with a manifest of
  where each part came from. Left behind: every secret (logins, webhooks, keys, tokens: each file or folder whose name
  the secrets registry says may hold one, by its name or an older one, secrets_registry.holds_secret; installed again
  with `rkmon login` and `rkmon secret`), the Claude session on record (state/orchestrator.json: a Claude conversation
  stays on the host that made it, so this deployment's orchestrator starts a new one), every lock, pid file and log of a
  running loop, backups and temporary files, the job folders. In state.json the work in flight is cleared (reviews ready
  or staging, the pipes, the jobs, the rechecks, the draft pools: none survives a move, and the plan picks each PR again
  whose head our review does not cover). The internal state leaves the earlier work volume's setup record behind
  (rkmon_lib/provision.py's RECORD): this deployment's start sets its own work volume up, before its first start too.
  Standby is set in it (standby.on, its reason REASON, since the staging's time: the deployment starts in standby, and
  sends nothing of what the earlier monitor sends until `rkmon standby off`), and the adoption
  recorded beside it (provision.ADOPTED_AT, the same time), in the staged copy (state_store.staged, as the host has no
  worker/); an earlier internal state that cannot be read is refused (AdoptError), as standby could not be set in it,
  its fix the state's own, then adopt again. The earlier monitor's config.json comes along (CONFIG_JSON, a file, never a
  link), for the one-time move to split by owner.
- plan (in a one-off container of this deployment, read-only): what apply would replace and merge, record by record,
  where config.json goes, and that the deployment starts in standby.
- apply (the same, writing): the state folder, then the probes, are replaced. The new copy is built in a folder
  inside the one it replaces (BUILD), which a build that fails removes, nothing changed; then, for the state folder,
  the mark of the adoption is written (below), and the copy is swapped in: the folder's own entries go aside into
  <WORKING_DIR>/adopt-backup-<time>/, but the container's scaler state (KEEP_CONTAINER), and the built ones take
  their place, the internal state first, so that the folder is in standby from the first of them on. Then the
  lessons, caches, post copies, daily and spend folders are merged, each file of the same name going aside first;
  then the earlier monitor's config.json goes into the monitor's home (this deployment's own aside) and, the internal
  state in place, the one-time move is made due again (one_time_move.due_again), so that the next start splits it;
  last, the mark says that apply finished. A file apply writes keeps its times and takes the mode its folder gives a
  file the monitor makes (the umask, the folder's default ACL and set-group-ID bit: worker/perms.py), never the
  stage's; one it writes over keeps its own. Every old path in a .json or .jsonl file, config.json's too, is
  rewritten to this deployment's: the pairs come from the two configurations (each old folder -> this deployment's
  folder of the same part) and from --map. An OSError, or a move that cannot be made due again, stops apply (Stopped):
  the error, with its fix (`stopped_by`), the lines of what apply did, and where that leaves the state: NOTHING,
  PART_WAY or PLACED.

The mark of an adoption (MARK_FILE, of the kind MARK_KIND, in the monitor's home, outside the state folder) keeps the
monitor in standby (features.standby) from before apply moves anything until `rkmon standby off` removes it, whatever
the internal state says, and with none: an apply that stops once the state folder went aside, or an internal state
moved aside, never ends standby. Its `applied_at` is "" until apply has finished, then the time it did.
- `marked(home) -> bool`: whether the monitor's home HOME holds a mark (`mark_file(home)`, its path).
- `unfinished(home) -> config.ConfigError | None`: why the adoption HOME's mark records has not finished: rkmon adopt
  stopped part-way, or the mark cannot be read; its fix adopt again, from the same copy. None when HOME holds no mark,
  or one that says apply finished. `rkmon standby off` refuses while it says why (rkmon_lib/standby.py).
- `unmark(home)`: the mark removed, which standby off does last, after the move's end and the clear; a
  config.ConfigError with its fix, HOME made writable for the monitor, then standby off, when it cannot be.

docker/rkmon refuses to adopt while the container runs, shows the plan first, and asks before it applies.
"""

from __future__ import annotations

import contextlib
import datetime as dt
import errno
import json
import os
import pathlib
import re
import shutil

from . import (
    commands,
    config,
    one_time_move,
    paths,
    program,
    provision,
    runtime_config,
    state_store,
    steps,
    store,
)
from .checks import Steps
from .secrets_registry import holds_secret

# The parts of a monitor's working folder: (label, how apply puts it in place, its config key, its folder under the
# working folder when the configuration names none). REPLACE: the container's copy goes aside whole, its folder staying
# (`_swap`); MERGE: files of the same name are replaced (the container's go aside), others are kept; MAP: no file
# moves, only its paths are rewritten.
REPLACE, MERGE, MAP = "replace", "merge", "map"
PARTS = (
    ("state", REPLACE, "", f"monitor/{paths.STATE}"),
    ("probes", REPLACE, "paths.probes", "probes"),
    ("lessons", MERGE, "paths.lessons", "lessons"),
    ("cache", MERGE, "paths.cache", "cache"),
    ("posts", MERGE, "paths.posts", "scratch/post"),
    ("posted", MERGE, "paths.posted_dir", ""),
    ("daily", MERGE, "paths.daily_dir", "rocke_daily"),
    ("spend", MERGE, "paths.spend_dir", "rocke_spend"),
    ("home", MAP, "", "monitor"),
    ("workers", MAP, "paths.workers", "workers"),
    ("repo", MAP, "paths.repo", "repo"),
    ("scratch", MAP, "paths.scratch", "scratch"),
    ("tmp", MAP, "paths.tmp", "tmp"),
    ("outputs", MAP, "paths.outputs_dir", ""),
    ("root", MAP, "paths.working_dir", ""),
)
CACHE_DIRS = (
    "mutants",
)  # the cache's subfolders that move (the others are rebuilt by the next scans)
POSTED = re.compile(r"PR_.*\.md$")  # the posted copies in the outputs folder
# Never moved: locks, pid files and the logs of a running loop, the login cache, the Claude session on record (the
# orchestrator's, rkmon_lib/orchestrator.py STATE: a Claude conversation stays on the host that made it), backups,
# temporary files.
SKIP = re.compile(
    r"^(lock|.*\.lock|.*\.pid|tick_loop\..*|web\.(log|pid|lock)|gh_logins\.json|mirror_feed\.jsonl|orchestrator\.json)$"
    r"|\.bak-|\.prev$|\.tmp$"
)
SKIP_DIRS = {
    "jobs",
    "__pycache__",
    ".git",
}  # in the state folder: the job folders (the jobs are cleared)
NO_REWRITE = {
    "spend_cache.json"
}  # keyed by the old Claude transcripts' paths: kept as history, as it is
IN_FLIGHT = (
    "review_ready",
    "review_pipes",
    "rechecks",
    "draft_pool",
    "shadow_pool",
    "analysis_wait",
)
# The state folder's files the container keeps its own of, when it has one (`_swap`).
KEEP_CONTAINER = ("scaler_state.json",)
MANIFEST = "manifest.json"
# The earlier monitor's configuration file, staged in the folder of the home part (staged_config).
CONFIG_JSON = runtime_config.CONFIG_JSON
# Standby's reason in the adopted state: the deployment starts in standby.
REASON = "adopted from an earlier monitor"
# A replaced folder's new copy is built in this folder inside it, so that what apply writes there takes the modes the
# folder gives the monitor's files (`_build`), then swapped in (`_swap`).
BUILD = ".adopt-incoming"
# Where a stopped apply leaves the state (Stopped.where).
NOTHING = "nothing changed"
PART_WAY = "the state folder is part-way between this deployment's state and the earlier monitor's"
PLACED = "the earlier monitor's state is in place, in standby"
# The mark of an adoption, in the monitor's home (see the module's text): its file and its kind.
MARK_FILE = "adoption.json"
MARK_KIND = "adoption"


class AdoptError(RuntimeError):
    """What adopt refuses, nothing changed: MESSAGE says why, and FIX is the fix it names (a checks.Command, Edit or
    Steps), None when the message says it; `fix(program_name)` gives it, as config.ConfigError's does.
    """

    def __init__(self, message: str, fix=None):
        super().__init__(message)
        self._fix = fix

    def fix(self, program_name: str):
        """The fix this refusal names, as the host's command PROGRAM_NAME runs it; None when it names none."""
        return self._fix


class Stopped(Exception):
    """apply stopped: ERROR, the config.ConfigError that stopped it, with its fix; DONE, the lines of what apply did,
    as it returns them; WHERE, where that leaves the state (NOTHING, PART_WAY or PLACED).
    """

    def __init__(self, error: config.ConfigError, done, where: str):
        super().__init__(str(error))
        self.error, self.done, self.where = error, tuple(done), where


def _get(cfg: dict, dotted: str) -> str:
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict):
            return ""
        node = node.get(part)
    return str(node or "").strip()


def old_folders(root, cfg: dict | None) -> dict:
    """{label: the old monitor's folder of that part} from its configuration (paths.*), else the usual place under
    ROOT; its home, ROOT/monitor. The posted copies' folder is paths.posted_dir, else paths.outputs_dir, else the
    dashboard file's folder.
    """
    root = pathlib.Path(os.path.expanduser(str(root))).resolve()
    cfg = cfg or {}
    home = root / "monitor"

    def p(v: str) -> pathlib.Path:
        return pathlib.Path(os.path.expanduser(v))

    out = {}
    for label, _how, key, under in PARTS:
        v = _get(cfg, key) if key else ""
        if label == "state":
            out[label] = home / paths.STATE
        elif label == "home":
            out[label] = home
        elif label == "root":
            out[label] = p(v) if v else root
        elif v:
            out[label] = p(v)
        elif under:
            out[label] = root / under
    outputs = _get(cfg, "paths.outputs_dir")
    dash = _get(cfg, "paths.dashboard")
    if "outputs" not in out:
        out["outputs"] = p(outputs) if outputs else (p(dash).parent if dash else root)
    if "posted" not in out:
        out["posted"] = out["outputs"]
    if not _get(cfg, "paths.daily_dir") and (out["outputs"] / "rocke_daily").is_dir():
        out["daily"] = out["outputs"] / "rocke_daily"
    if not _get(cfg, "paths.spend_dir") and (out["outputs"] / "rocke_spend").is_dir():
        out["spend"] = out["outputs"] / "rocke_spend"
    return {k: str(v) for k, v in out.items()}


def parse_maps(items) -> list:
    """--map OLD=NEW arguments: [(old, new)], each an absolute path."""
    out = []
    for it in items or ():
        old, sep, new = str(it).partition("=")
        if not sep or not old.startswith("/") or not new.startswith("/"):
            raise AdoptError(f"--map {it}: OLD=NEW, two absolute paths")
        out.append((old.rstrip("/") or "/", new.rstrip("/") or "/"))
    return out


def _moved(name: str) -> bool:
    """A file of this name moves: no lock, backup or temporary file, and never a secret's file, wherever it is."""
    return not (SKIP.search(name) or holds_secret(name))


def _copy(
    src: pathlib.Path,
    dst: pathlib.Path,
    left: list,
    top: bool = True,
    state: bool = False,
) -> int:
    """Copy a folder's files that move (secrets, locks and backups left, and in the state folder its job folders),
    links never followed. How many."""
    n = 0
    if not src.is_dir():
        return 0
    for entry in sorted(src.iterdir()):
        if entry.is_symlink():
            continue
        if entry.is_dir():
            if (
                holds_secret(entry.name)
                or (state and top and entry.name in SKIP_DIRS)
                or entry.name
                in (
                    "__pycache__",
                    ".git",
                )
            ):
                left.append(str(entry) + "/")
                continue
            n += _copy(entry, dst / entry.name, left, top=False, state=state)
        elif entry.is_file():
            if not _moved(entry.name):
                left.append(str(entry))
                continue
            dst.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry, dst / entry.name)
            n += 1
    return n


def clear_in_flight(st: dict) -> dict:
    """state.json's work in flight, cleared in place: {what: what it held}."""
    cleared = {}
    for k in IN_FLIGHT:
        v = st.get(k)
        if v:
            cleared[k] = (
                [r.get("pr") for r in v]
                if isinstance(v, list) and v and isinstance(v[0], dict)
                else sorted(v) if isinstance(v, dict) else v
            )
            st[k] = type(v)()
    if st.get("review_pipeline"):
        cleared["review_pipeline"] = (st["review_pipeline"] or {}).get("pr")
    if "review_pipeline" in st:
        st["review_pipeline"] = None
    jobs = st.get("jobs") or {}
    for k, v in list(jobs.items()):
        if k != "history" and v:
            cleared[f"jobs.{k}"] = v.get("id") if isinstance(v, dict) else v
            jobs[k] = None
    if st.get("drift_job"):
        cleared["drift_job"] = st["drift_job"]
        st["drift_job"] = None
    if st.pop("mirror", None) is not None:
        cleared["mirror"] = "its mirror record"
    for k in ("stopped", "draining"):
        if st.pop(k, None) is not None:
            cleared[k] = "the old monitor's"
    return cleared


def leave_setup_record(state_dir: pathlib.Path) -> None:
    """The internal state staged in STATE_DIR without the earlier work volume's setup record (provision.RECORD), which
    would tell this deployment's start that its work volume is set up. A file that cannot be read stays as it is: the
    monitor's first read names its fix."""
    p = state_dir / state_store.STATE_FILE
    try:
        body = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if isinstance(body, dict) and body.pop(provision.RECORD, None) is not None:
        p.write_text(json.dumps(body, ensure_ascii=False, indent=1), encoding="utf-8")


def staged_config(stage_dir) -> pathlib.Path:
    """Where the stage in STAGE_DIR keeps the earlier monitor's config.json: in the folder of the home part."""
    return pathlib.Path(stage_dir) / "home" / CONFIG_JSON


def _stand_by(old_home, out: pathlib.Path) -> None:
    """Standby set in the internal state staged in OUT, and the adoption recorded beside it: see the module's text.
    AdoptError when the earlier monitor's internal state, in its home OLD_HOME, cannot be read, its fix the state's
    own, then adopt again.
    """
    when = store.now()
    values = {
        "standby.on": True,
        "standby.reason": REASON,
        "standby.since": when,
        provision.ADOPTED_AT: when,
    }
    try:
        state_store.read(old_home)
        # the stage's state folder is the state part's, whose label is the state folder's name (paths.STATE)
        state_store.staged(out, values)
    except state_store.StateError as exc:
        raise AdoptError(
            f"{exc}, so standby cannot be set in the adopted state",
            _then_again(exc.fix(program.name())),
        ) from None


def stage(old: dict, out, extra_maps=()) -> dict:
    """Copy what moves from the old monitor's folders (old_folders) into OUT (made afresh, mode 700), its config.json
    too (staged_config), set standby in the staged internal state (`_stand_by`), and write the manifest. The manifest:
    {sources, maps, cleared, left, counts, records, config_json, staged_at}; CONFIG_JSON, the config.json staged, ""
    when the earlier monitor's home has none."""
    out = pathlib.Path(out)
    state_json = pathlib.Path(old["state"]) / "state.json"
    if not state_json.is_file():
        raise AdoptError(
            f"no state.json in {old['state']}: name the old monitor's working folder (the one holding monitor/state/)"
        )
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    out.chmod(0o700)
    left, counts = [], {}
    for label, how, _key, _under in PARTS:
        if how == MAP or label not in old:
            continue
        src = pathlib.Path(old[label])
        if label == "posted":
            n = 0
            for f in sorted(src.glob("PR_*.md")) if src.is_dir() else []:
                if f.is_file() and POSTED.match(f.name):
                    (out / label).mkdir(exist_ok=True)
                    shutil.copy2(f, out / label / f.name)
                    n += 1
            counts[label] = n
        elif label == "cache":
            n = 0
            for f in sorted(src.iterdir()) if src.is_dir() else []:
                if f.is_file() and not f.is_symlink() and _moved(f.name):
                    (out / label).mkdir(exist_ok=True)
                    shutil.copy2(f, out / label / f.name)
                    n += 1
            for d in CACHE_DIRS:
                n += _copy(src / d, out / label / d, left)
            counts[label] = n
        else:
            counts[label] = _copy(src, out / label, left, state=label == "state")
    st = json.loads(state_json.read_text(encoding="utf-8"))
    cleared = clear_in_flight(st)
    (out / "state").mkdir(exist_ok=True)
    (out / "state" / "state.json").write_text(
        json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    leave_setup_record(out / "state")
    _stand_by(old["home"], out)
    source, config_json = pathlib.Path(old["home"]) / CONFIG_JSON, ""
    if source.is_file() and not source.is_symlink():
        staged_config(out).parent.mkdir()
        shutil.copy2(source, staged_config(out))
        config_json = str(source)
    prs = st.get("prs") or {}
    records = {
        "prs": len(prs),
        "with_our_review": sum(
            1 for r in prs.values() if (r.get("ours") or {}).get("head")
        ),
        "open": sum(1 for r in prs.values() if r.get("status") == "open"),
        "holds": len(st.get("holds") or {}),
        "incidents": len(st.get("incidents") or {}),
        "last_tick": st.get("last_tick"),
        "enabled": st.get("enabled"),
    }
    manifest = {
        "sources": {k: v for k, v in old.items()},
        "maps": [list(m) for m in extra_maps],
        "cleared": cleared,
        "left": sorted(left),
        "counts": counts,
        "records": records,
        "config_json": config_json,
        "staged_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    (out / MANIFEST).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    for d, _dirs, files in os.walk(
        out
    ):  # the one-off container reads it as this deployment's user
        os.chmod(d, 0o755)
        for f in files:
            os.chmod(os.path.join(d, f), 0o644)
    return manifest


def stage_report(m: dict) -> list:
    lines = ["What moves (files):"]
    lines += [f"  {k}: {n} from {m['sources'].get(k)}" for k, n in m["counts"].items()]
    lines.append(
        f"  config.json: from {m['config_json']}"
        if m["config_json"]
        else "  config.json: none in the copy"
    )
    lines += [
        f"Left behind: every secret and login (set them again: {program.name()} login), the Claude session on "
        "record (a Claude conversation stays on the host that made it: the orchestrator here starts a new one), the "
        f"locks, pid files and loop logs, backups, the job folders ({len(m['left'])} item(s))",
        "Work in flight, cleared (the plan picks these PRs again):",
    ]
    lines += [f"  {k}: {v}" for k, v in m["cleared"].items()] or ["  none"]
    r = m["records"]
    lines.append(
        f"The state: {r['prs']} PR records ({r['open']} open, {r['with_our_review']} with our review), "
        f"{r['holds']} PRs on hold, {r['incidents']} incidents; last tick {r['last_tick']}"
    )
    return lines


def rewrites(m: dict, targets: dict) -> list:
    """The path rewrites, longest old path first: each old folder of a part -> this deployment's folder of the same
    part (targets), then the --map pairs."""
    pairs = {}
    for label, old in m["sources"].items():
        new = targets.get(label)
        if old and new and old != new:
            pairs[old.rstrip("/")] = new.rstrip("/")
    for old, new in m.get("maps") or []:
        pairs[old] = new
    return sorted(pairs.items(), key=lambda kv: -len(kv[0]))


def rewrite(text: str, pairs: list) -> tuple:
    n = 0
    for old, new in pairs:
        c = text.count(old)
        if c:
            text, n = text.replace(old, new), n + c
    return text, n


def _count(p: pathlib.Path) -> int:
    return sum(1 for x in p.rglob("*") if x.is_file()) if p.is_dir() else 0


def plan(stage_dir, targets: dict) -> list:
    """What apply would do, in words (nothing written)."""
    from . import states  # a call-time import: states reads the settings files' module

    stage_dir = pathlib.Path(stage_dir)
    m = load_manifest(stage_dir)
    cur_p = pathlib.Path(targets["state"]) / "state.json"
    try:
        cur = json.loads(cur_p.read_text(encoding="utf-8")) if cur_p.is_file() else {}
    except (OSError, ValueError):
        cur = {}
    new = json.loads((stage_dir / "state" / "state.json").read_text(encoding="utf-8"))
    heads = lambda prs: sum(
        1 for r in prs.values() if (r.get("ours") or {}).get("head")
    )  # noqa: E731
    lines = [
        f"state.json: this deployment's {len(cur.get('prs') or {})} PR records ({heads(cur.get('prs') or {})} with "
        f"our review) -> the old monitor's {len(new.get('prs') or {})} ({heads(new.get('prs') or {})} with our review)"
    ]
    for label, how, _key, _under in PARTS:
        if how == MAP or not (stage_dir / label).exists():
            continue
        t = pathlib.Path(targets.get(label) or "")
        if not targets.get(label):
            lines.append(
                f"{label}: no folder for it in this deployment's configuration: left out"
            )
        elif how == REPLACE:
            lines.append(
                f"{t}: its {_count(t)} files go aside; the old monitor's {_count(stage_dir / label)} come in"
            )
        else:
            lines.append(
                f"{t}: + {_count(stage_dir / label)} files (same names replaced, this deployment's go aside)"
            )
    here = pathlib.Path(targets["home"]) / CONFIG_JSON
    if staged_config(stage_dir).is_file():
        aside = ", this deployment's going aside" if os.path.lexists(here) else ""
        lines.append(
            f"{here}: the earlier monitor's comes in{aside}; the next start splits it by owner, once (the one-time "
            "move)"
        )
    else:
        lines.append(
            "config.json: none in the earlier monitor's copy, so no one-time move"
        )
    lines.append(f"the deployment starts {states.in_standby()}")
    pairs = rewrites(m, targets)
    lines.append(f"paths rewritten in .json and .jsonl files: {len(pairs)} pair(s)")
    lines += [f"  {o} -> {n}" for o, n in pairs]
    lines.append(f"kept from this deployment's state: {', '.join(KEEP_CONTAINER)}")
    return lines


def load_manifest(stage_dir) -> dict:
    p = pathlib.Path(stage_dir) / MANIFEST
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise AdoptError(
            f"no staged state at {stage_dir} ({exc}): stage it first"
        ) from exc


def _put(src: pathlib.Path, dst: pathlib.Path, pairs: list) -> int:
    """One file into place, with SRC's times: a .json or .jsonl file with its old paths rewritten, any other copied. A
    new file takes the mode its folder gives a file the monitor makes, never SRC's; a file written over keeps its own.
    How many old paths it rewrote."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    if src.suffix in (".json", ".jsonl") and src.name not in NO_REWRITE:
        text, n = rewrite(
            src.read_text(encoding="utf-8", errors="surrogateescape"), pairs
        )
        dst.write_text(text, encoding="utf-8", errors="surrogateescape")
    else:
        shutil.copyfile(src, dst)
    st = src.stat()
    os.utime(dst, ns=(st.st_atime_ns, st.st_mtime_ns))
    return n


def _put_config(stage_dir, home, backup: pathlib.Path, pairs: list) -> str:
    """The staged config.json into the monitor's home HOME, its old paths rewritten (PAIRS), this deployment's own
    moved aside into BACKUP first. Its line."""
    dst = pathlib.Path(home) / CONFIG_JSON
    aside = ""
    if os.path.lexists(dst):
        (backup / "home").mkdir(parents=True, exist_ok=True)
        shutil.move(str(dst), str(backup / "home" / CONFIG_JSON))
        aside = "; this deployment's is set aside"
    n = _put(staged_config(stage_dir), dst, pairs)
    return f"{dst}: the earlier monitor's, {n} old path(s) rewritten in it{aside}"


def _again() -> tuple:
    """The words and the command, after a fix, that adopt again from the copy, which stages it afresh: the command in
    the form `rkmon help` names it (commands.form_of)."""
    return ("run", steps.command_line(commands.form_of("adopt")), "again")


def _then_again(fix) -> Steps:
    """FIX (a checks.Command, Edit or Steps; None for none), then adopt again (`_again`)."""
    if fix is None:
        return Steps(_again())
    parts = fix.parts if isinstance(fix, Steps) else (fix,)
    return Steps((*parts, ", then", *_again()))


def mark_file(home) -> pathlib.Path:
    """The mark of an adoption in the monitor's home HOME: see the module's text."""
    return pathlib.Path(home) / MARK_FILE


def marked(home) -> bool:
    """Whether HOME holds the mark of an adoption: see the module's text."""
    return os.path.lexists(mark_file(home))


def _mark(home, applied_at: str = "") -> None:
    """The mark of an adoption written whole in HOME (state_store.write_file), APPLIED_AT the time apply finished,
    "" until it has. Its OSError names the mark's file."""
    p = mark_file(home)
    try:
        state_store.write_file(p, {"kind": MARK_KIND, "applied_at": applied_at})
    except OSError as exc:
        raise OSError(exc.errno, exc.strerror, str(p)) from None


def unfinished(home) -> config.ConfigError | None:
    """Why the adoption HOME's mark records has not finished, with its fix; None when it has, or there is no mark: see
    the module's text."""
    p = mark_file(home)
    if not os.path.lexists(p):
        return None
    try:
        mark = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        why = getattr(exc, "strerror", None) or exc
        return config.ConfigError(
            f"{p} cannot be read as the mark of an adoption: {why}", Steps(_again())
        )
    if (
        not isinstance(mark, dict)
        or mark.get("kind") != MARK_KIND
        or not isinstance(mark.get("applied_at"), str)
    ):
        return config.ConfigError(f"{p} is no mark of an adoption", Steps(_again()))
    if mark["applied_at"]:
        return None
    return config.ConfigError(
        f"{program.name()} adopt stopped part-way", Steps(_again())
    )


def unmark(home) -> None:
    """The mark of an adoption removed from HOME, as standby off does last: see the module's text."""
    p = mark_file(home)
    try:
        p.unlink(missing_ok=True)
    except OSError as exc:
        why = exc.strerror or exc
        off = ("run", steps.command_line("standby off"))
        raise config.ConfigError(
            f"the mark of the adoption, {p}, cannot be removed: {why}",
            one_time_move.writable(p.parent, off),
        ) from None


def stopped_by(exc: OSError, stage_dir, at=None) -> config.ConfigError:
    """EXC, the OSError that stopped plan or apply, in words, with its fix. What it names is the file EXC names (of
    two, the second: the one a copy or a rename writes), one in a build where it was going (BUILD left out), else AT,
    the folder apply was changing. A staged copy that cannot be read is staged afresh, by adopt again (`_again`); else
    the folder named is given space (ENOSPC, EDQUOT) or made writable for the monitor (one_time_move.writable), then
    adopt again."""
    why = exc.strerror or str(exc)
    named = exc.filename2 or exc.filename
    if isinstance(named, (str, bytes)):
        parts = pathlib.Path(os.fsdecode(named)).parts
        place = pathlib.Path(*(p for p in parts if p != BUILD))
    else:
        place = pathlib.Path(at) if at else None
    if place is None:
        return config.ConfigError(f"adopt stopped: {why}", Steps(_again()))
    if place.is_relative_to(stage_dir):
        return config.ConfigError(
            f"the staged copy {place} cannot be read: {why}", Steps(_again())
        )
    folder = place if os.path.isdir(place) else place.parent
    if exc.errno in (errno.ENOSPC, errno.EDQUOT):
        fix = Steps((f"free space on the volume of {folder}, then", *_again()))
    else:
        fix = one_time_move.writable(folder, _again())
    return config.ConfigError(f"adopt stopped at {place}: {why}", fix)


def _build(src: pathlib.Path, t: pathlib.Path, pairs: list) -> tuple:
    """The staged part SRC built inside the folder T it replaces, in T/BUILD, each file `_put` there, so that it
    takes the modes T gives the files the monitor makes: (its files, the old paths rewritten in them). An OSError
    removes the build first, so that nothing of T has changed."""
    t.mkdir(parents=True, exist_ok=True)
    build = t / BUILD
    files = rewritten = 0
    try:
        if os.path.lexists(build):
            shutil.rmtree(build)  # what an apply cut short left
        build.mkdir()
        for f in sorted(src.rglob("*")):
            if f.is_file():
                rewritten += _put(f, build / f.relative_to(src), pairs)
                files += 1
    except OSError:
        shutil.rmtree(build, ignore_errors=True)
        raise
    return files, rewritten


def _swap(t: pathlib.Path, aside: pathlib.Path, keep=()) -> None:
    """The copy built in T/BUILD (`_build`) in place of T's own entries: each of those but KEEP's moved into ASIDE,
    then each built one into T, the internal state first, so that T is in standby from the first of them on, a KEEP
    name T kept leaving the built one out; then the build folder removed. T itself stays, with its modes.
    """
    build = t / BUILD
    own = [e for e in sorted(t.iterdir()) if e.name != BUILD and e.name not in keep]
    if own:
        aside.mkdir(parents=True, exist_ok=True)
    for e in own:
        shutil.move(str(e), str(aside / e.name))
    for e in sorted(
        build.iterdir(), key=lambda e: (e.name != state_store.STATE_FILE, e.name)
    ):
        if e.name in keep and os.path.lexists(t / e.name):
            e.unlink()
        else:
            e.rename(t / e.name)
    build.rmdir()


def apply(stage_dir, targets: dict, backup_root, now: str | None = None) -> list:
    """Put the staged state in place (see the module's text). The lines to show; Stopped when an OSError, or a move
    that cannot be made due again, stops it."""
    stage_dir = pathlib.Path(stage_dir)
    m = load_manifest(stage_dir)
    pairs = rewrites(m, targets)
    stamp = now or dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = pathlib.Path(backup_root) / f"adopt-backup-{stamp}"
    kept = f"this deployment's own copies are in {backup}"
    home = pathlib.Path(targets["home"])
    out, where, at, made = [], NOTHING, backup.parent, False
    try:
        backup.mkdir(parents=True)
        made = True
        rewritten = 0
        for label, how, _key, _under in PARTS:
            src = stage_dir / label
            if how == MAP or not src.exists() or not targets.get(label):
                continue
            t = at = pathlib.Path(targets[label])
            if how == REPLACE:
                files, n = _build(src, t, pairs)
                rewritten += n
                # the state folder, the first part: the mark first, then its swap, where the state changes
                is_state = label == "state"
                if is_state:
                    try:
                        _mark(home)
                    except OSError:
                        shutil.rmtree(t / BUILD, ignore_errors=True)
                        raise
                    where = PART_WAY
                _swap(t, backup / label, KEEP_CONTAINER if is_state else ())
                if is_state:
                    where = PLACED
                out.append(f"{t}: replaced ({files} files)")
            else:
                files = 0
                for f in sorted(src.rglob("*")):
                    if not f.is_file():
                        continue
                    rel = f.relative_to(src)
                    if (t / rel).exists():
                        (backup / label / rel).parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(t / rel, backup / label / rel)
                    rewritten += _put(f, t / rel, pairs)
                    files += 1
                out.append(f"{t}: {files} files merged in")
        state_t = at = pathlib.Path(targets["state"])
        for sub in ("jobs", "gates", "incidents", config.FOLLOWUPS):
            (state_t / sub).mkdir(parents=True, exist_ok=True)
        out.append(f"{rewritten} old path(s) rewritten")
        at = home
        if staged_config(stage_dir).is_file():
            out.append(_put_config(stage_dir, home, backup, pairs))
            out += one_time_move.due_again(home, _again())
        _mark(home, store.now())
    except OSError as exc:
        error = stopped_by(exc, stage_dir, at)
        if where == NOTHING:
            if made:
                with contextlib.suppress(OSError):
                    backup.rmdir()
            raise Stopped(error, out, NOTHING) from None
        raise Stopped(error, [*out, kept], where) from None
    except one_time_move.MoveError as exc:
        raise Stopped(exc, [*out, kept], where) from None
    out.append(f"APPLIED: {kept}")
    return out
