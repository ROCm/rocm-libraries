# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The one writer of what a deployment runs: docker/.env, the compose override and the
delivered layer with the team's skills, as a plan makes them (rkmon_lib/deploy_plan.py), each kept as it was for
`rkmon rollback`, and the fingerprint of each file it writes kept in this folder's records, so that a hand edit shows
(state 9).

- `deliver(plan, keep_previous=True, *, package_dir, compose)`: PLAN into place, in this order.
  1. The checks, before anything is written: docker/.env, the override and their kept copies (KEPT, `.prev`) are,
     where they exist, regular files, not links, in a package folder outside cloud-synced folders (private_files); a
     file an older package wrote at a looser mode is set to 600 first (private_files.tighten). A refusal is
     private_files.Refused, and nothing is written. While the plan's layer records the team's skills (its `skills`),
     the folder its settings name is read (deploy_plan.skill_files), nothing written when it fails: SkillsChanged
     when its files are not the ones the layer records (a file changed, added or removed since the confirm), Failed
     (its code 0) when one is no UTF-8 text.
  2. With KEEP_PREVIOUS, the two files as they are: each kept copy written through private_files at 600, or removed
     when its file is not there, so that putting back gives back exactly what was there. When one cannot be written,
     the kept copies are put back as they were.
  3. The layer, through a one-off container (COMPOSE runs `docker compose` with LAYER_RUN's arguments, the program on
     its input: `layer_program`) of the plan's image in the plan's compose project: both are given to docker compose,
     `-p` and the plan's variables in its environment, never taken from the files in the docker folder, which may name
     another deployment's or none. The container is the compose file's LAYER_SERVICE, which mounts the settings volume
     alone, with no network and one capability (container_layout.LAYER_WRITE_CAPABILITIES). The layer is a new file,
     root's, readable by the monitor's group (the plan's RKMON_GID) and nobody else (0640), renamed into place as
     LAYER_FILE in the settings volume, whose folder is then root's and that group's (0750). Beside it, SKILLS_FOLDER
     holds the team's skills step 1 read, each file that group's (0640) in folders as the team's folder has them
     (0750); none while the layer records none. With KEEP_PREVIOUS, the layer it replaces is kept as LAYER_KEPT, and
     the skills as SKILLS_KEPT (each none, when there was none). The program keeps a journal beside them, so that the
     volume holds either its layers and skills as they were or the new ones: the layer's rename into place is the
     one step that changes them, and a run that stopped before it is undone, one that stopped after it finished,
     when the next run starts. Once the run ends well, the layer's fingerprint is kept (`wrote`), and with
     KEEP_PREVIOUS the one it replaced as the kept layer's.
  4. docker/.env and the override, through private_files at 600, each a new file renamed into place, then its
     fingerprint kept; with no override in the plan, the one there goes.
  KEEP_PREVIOUS False leaves every kept copy as it is, so that the kept set stays the one from before the last
  change: keep_previous decides it.
  A delivery that does not finish raises Failed, or Interrupted for an interrupt from step 3 on, whose `changed` says
  what it changed ("" for nothing):
  - the program stopped before the rename (its exit code UNCHANGED): nothing changed; the kept copies are put back.
  - the run ended any other way (docker failed, or the container was stopped): the settings volume has either its
    layers as they were or the new one; docker/.env and the override are as they were, and their kept copies are
    the files as they were. A delivery of the same plan then, keeping the previous ones as keep_previous decides,
    ends where this one would have ended.
  - docker/.env, the override or a fingerprint (in records.FILE) cannot be written after the run: the new layer is in
    place, and each file written before the one that failed.
- `keep_previous(package_dir, plan, read) -> bool`: whether a delivery of PLAN keeps what is in place (its
  KEEP_PREVIOUS), the rule's one home: not when this folder's records say the layer deliver last wrote is the plan's
  (records.Written.layer: the plan delivered again after a run that stopped, or a hand edit put back, state 9),
  nor when the settings volume holds the plan's layer (a run that stopped after the layer's rename, before its
  fingerprint was kept); else it does. READ, a function of nothing, reads the layer in the settings volume, only when
  the records do not say: (its text, ""), None for none there, or (None, why) when it cannot be read: Failed then
  (its code 0), nothing changed, never a keep.
- `put_back(variables, *, package_dir, compose) -> bool`: what deliver kept, back in place: the layer (LAYER_KEPT as
  LAYER_FILE) with the team's skills kept (SKILLS_KEPT as SKILLS_FOLDER; none, when none are kept), then docker/.env
  and the override (an override with no kept copy goes), after the checks of step 1,
  the one-off container in the compose project and with the image VARIABLES name (NAME -> text), given to docker
  compose as deliver gives the plan's: the deployment's, never docker/.env's, which a hand edit may have changed (a
  ValueError, nothing changed, when they name no project); each one's fingerprint kept as it is in place (`wrote`).
  The kept copies stay, so a put back that stopped part-way runs again to the same end. False, nothing changed, when
  nothing was kept (`kept`). Failed or Interrupted as for deliver: NOTHING_KEPT when the volume keeps no layer,
  nothing changed.
- `kept(package_dir)`: whether a delivery kept what was there before it (docker/.env's kept copy is there).
- `kept_differs(package_dir) -> bool`: whether what a delivery kept differs from what is in place, so that putting it
  back changes something (`rkmon rollback`): docker/.env or the override from its kept copy, or the layer the records
  say is in place from the kept one (records.Written: layer and kept_layer, which count as differing while either is
  not known); False when nothing is kept.
- `layer_program(call) -> bytes`: what the one-off container runs on its input: LAYER_PROGRAM, with the values it
  reads before it and CALL (Python, `deliver(...)` or `put_back()`) after it.
- `on_host(package_dir) -> (env, override)`: docker/.env's and the override's texts as they are in the package
  folder, None for a file that is not there, each checked first as every reader of a private file checks it
  (private_files.checked: Refused, with the fix; a reader changes no permissions).
- `kept_on_host(package_dir) -> (env, override)`: docker/.env's and the override's kept copies' texts (KEPT), what
  a put back puts in place, read as on_host reads the files: None for a copy that is not there.
- `skills_with_layer(skills) -> Path | None`: for a reader of the settings volume (agentic.team_skills), the team's
  skills that go with the layer in place, SKILLS being the volume's SKILLS_FOLDER: as the next run's journal would
  leave them, so that no reader takes the skills of one delivery with the layer of another. After the commit of a run
  that stopped before it finished, the skills it staged (SKILLS and JOURNAL_NEW), or none when it marked them as none
  (and JOURNAL_NONE); else SKILLS. None for no folder, or a link.
- `texts(plan) -> (env, override, layer)`: what deliver writes for PLAN, by WRITES: docker/.env's text, the
  override's (None: no file) and the layer's (deploy_plan.layer_text).
- `differing(made, env, override, layer) -> tuple`: the names in WRITES of what deliver writes whose whole text
  differs from what it writes for a plan, MADE (`texts`): docker/.env's text ENV, the override's OVERRIDE and the
  delivered layer's LAYER, each as it is, None for none: what apply would change, the change plan's rule
  (rkmon_lib/change_plan.py).
- `fingerprint(text) -> str`: the fingerprint of a file whose text is TEXT, as deliver writes it or finds it:
  `sha256:` and the SHA-256 of its UTF-8 bytes; NO_FILE for None, no file there.
- `wrote(package_dir, keep=False, put_back=False, **texts)`: what a delivery or a put-back wrote, kept by fingerprint
  in this folder's records (records.change_written), so that a hand edit shows (state 9). TEXTS, by the field
  of records.Written each file has: env, override and layer, each the text written (None: no file there); KEEP: the
  layer written replaced the one the settings volume keeps now, whose fingerprint kept_layer takes; PUT_BACK: the
  layer the settings volume kept is back in place, and layer takes kept_layer's fingerprint. deliver and put_back
  call it as they write; a TypeError for a name that is no such field.
- `changed_outside(written, env, override, layer) -> tuple`: the names in WRITES of the files that differ from what
  deliver last wrote there (state 9): docker/.env's text ENV, the override's OVERRIDE and the delivered layer's
  LAYER, each as it is (None for none), against the fingerprints WRITTEN keeps (records.Written; None while none
  are). A file with no fingerprint kept (an older package wrote it) counts as unchanged.
- COMPOSE: a function of (the arguments of `docker compose`, the bytes for its input, the variables set in its
  environment over this process's: NAME -> text) to its exit code, which runs docker compose in this package folder's
  docker folder (docker/rkmon's).
- ENV_FILE and OVERRIDE_FILE: the two files' places in the package folder, the first two of
  private_files.PACKAGE_FILES, which lists each kept copy after them (its name ending in KEPT). LAYER_NAME: the layer
  as messages name it; WRITES: the names of the three. SETTINGS_VOLUME: the settings volume, which the container
  mounts read-only; LAYER_WRITE_DIR: where the one-off container mounts it to write it; LAYER_FILE: the layer's file
  there; SKILLS_FOLDER: the team's skills' folder there (the four container_layout's). LAYER_KEPT and SKILLS_KEPT:
  what a delivery keeps of the two. LAYER_SERVICE: that container's compose service. LAYER_PROGRAM: what it runs
  (Python, standard library), as root. JOURNAL_NEW and JOURNAL_NONE: its journal's ends: what a run stages is named
  as what it becomes and JOURNAL_NEW, and a mark that it is none adds JOURNAL_NONE to that.
- `Failed`: a delivery or a put back that did not finish: `code`, the one-off container's exit code (0 when it ended
  well, and a file on this host could not be written after it, or when keep_previous could not read the layer in
  place), `changed`, and `above`: what the outcome shows above itself first, "" for none: a private file's refusal
  (private_files.Refused) as it shows itself, its rows then where the rules are, whose words then say that the lines
  above say why. `SkillsChanged`, a Failed of its own kind (its code 0, nothing changed): step 1's, the team's skills
  changed since the confirm, which a confirm of them comes before (state 5). `Interrupted`, a
  KeyboardInterrupt: `changed`.

Standard library only, on the host's Python (deployment.HOST_PYTHON): rkmon runs it on the host.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import pathlib

from . import container_layout, deploy_plan, private_files, records

ENV_FILE, OVERRIDE_FILE = private_files.PACKAGE_FILES[:2]
KEPT = ".prev"
SETTINGS_VOLUME = container_layout.SETTINGS_VOLUME
LAYER_WRITE_DIR = container_layout.LAYER_WRITE_DIR
LAYER_FILE = container_layout.LAYER_FILE
LAYER_KEPT = "deploy.previous.json"
SKILLS_FOLDER = container_layout.SKILLS_FOLDER
SKILLS_KEPT = "skills.previous"
JOURNAL_NEW = ".new"
JOURNAL_NONE = ".none"
LAYER_NAME = f"the settings volume's {LAYER_FILE}"
WRITES = (ENV_FILE, OVERRIDE_FILE, LAYER_NAME)
# the field of records.Written that keeps each file's fingerprint, by its name in WRITES
_FIELDS = {ENV_FILE: "env", OVERRIDE_FILE: "override", LAYER_NAME: "layer"}
NO_FILE = "none"  # the fingerprint of no file (fingerprint)
NOTHING_KEPT = 3
UNCHANGED = 4
LAYER_SERVICE = "deliver"
# docker compose's arguments for the one-off container that writes the settings volume, after `-p PROJECT`: its
# service mounts the volume; the program comes on its input (python3 -I -), then the folder and the owner's id
LAYER_RUN = (
    "run",
    "--rm",
    "-T",
    "--no-deps",
    "--user",
    "0:0",
    "--entrypoint",
    "python3",
    LAYER_SERVICE,
    "-I",
    "-",
    LAYER_WRITE_DIR,
    "0",
)
LAYER_PROGRAM = '''\
import errno
import os
import shutil
import signal
import stat
import sys

FOLDER, OWNER = sys.argv[1], int(sys.argv[2])
CURRENT = os.path.join(FOLDER, LAYER_FILE)
PREVIOUS = os.path.join(FOLDER, LAYER_KEPT)
SKILLS = os.path.join(FOLDER, SKILLS_FOLDER)
SKILLS_PREVIOUS = os.path.join(FOLDER, SKILLS_KEPT)
# The journal: what a run stages is named as what it becomes, and NEW. STAGED, the layer that becomes CURRENT in one
# rename, the commit, is there only before it: it is made before the rest of the journal, and goes after it when a run
# is undone. SKILLS_STAGED, the team's skills that become SKILLS, is there from before the commit until it is SKILLS.
# With what is in place kept, WAS (a second name of the layer in CURRENT) and SKILLS_WAS (second names of SKILLS'
# files, in folders as its folders are) are there from before the commit until they are PREVIOUS and SKILLS_PREVIOUS.
# Each of these three (FINISHED, by what it becomes) is either there or marked as none (its name and NONE): then, after
# the commit, nothing is at the place it becomes.
STAGED = CURRENT + NEW
WAS = PREVIOUS + NEW
SKILLS_STAGED = SKILLS + NEW
SKILLS_WAS = SKILLS_PREVIOUS + NEW
FINISHED = (PREVIOUS, SKILLS_PREVIOUS, SKILLS)
signal.signal(signal.SIGINT, signal.SIG_IGN)  # an interrupt at the terminal lets the run end as it would


def regular(path):
    return os.path.isfile(path) and not os.path.islink(path)


def folder(path):
    return os.path.isdir(path) and not os.path.islink(path)


def failed(exc):
    raise exc


def cleared(path):
    """PATH, with nothing there."""
    if folder(path):
        shutil.rmtree(path)
    elif os.path.lexists(path):
        os.unlink(path)
    return path


def synced(path=FOLDER):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def synced_tree(top):
    for path, _, _ in os.walk(top, topdown=False, onerror=failed):
        synced(path)


def marked(path):
    """PATH, a new empty file: a mark."""
    os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600))


def moved(target):
    """After the commit: TARGET made from what was staged for it, or with nothing there when that is marked as
    none."""
    if os.path.lexists(target + NEW):
        cleared(target)
        os.replace(target + NEW, target)
    elif os.path.lexists(target + NEW + NONE):
        cleared(target)
        cleared(target + NEW + NONE)


def recover():
    """What a run that stopped left: undone before its commit, finished after it; the same again when it stops."""
    if os.path.lexists(STAGED):
        for target in FINISHED:
            cleared(target + NEW)
            cleared(target + NEW + NONE)
        synced()
        cleared(STAGED)
    for target in FINISHED:
        moved(target)
    synced()


def written(path, text, group):
    """TEXT as the new file PATH, OWNER's and GROUP's, 0640, on disk."""
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o640)
    with os.fdopen(fd, "wb") as f:
        os.fchown(f.fileno(), OWNER, group)
        os.fchmod(f.fileno(), 0o640)
        f.write(text.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def made(path, uid, gid, perms):
    os.mkdir(path, 0o700)
    os.chown(path, uid, gid)
    os.chmod(path, perms)


def linked(source, target):
    """TARGET, a new folder of second names of SOURCE's files, in folders as SOURCE's are, on disk."""
    for path, _, files in os.walk(source, onerror=failed):
        to = os.path.normpath(os.path.join(target, os.path.relpath(path, source)))
        info = os.lstat(path)
        made(to, info.st_uid, info.st_gid, stat.S_IMODE(info.st_mode))
        for name in files:
            os.link(os.path.join(path, name), os.path.join(to, name), follow_symlinks=False)
    synced_tree(target)


def change(stage, keep, group):
    """STAGE() makes STAGED, then SKILLS_STAGED; with KEEP, the journal says what is in place; then the commit, and
    the journal finished. An error before the commit undoes the journal and exits UNCHANGED. After it, the folder is
    OWNER's and GROUP's (with GROUP None, the new CURRENT's group), 0750."""
    try:
        recover()
        stage()
        if keep:
            if regular(CURRENT):
                os.link(CURRENT, WAS)
            else:
                marked(WAS + NONE)
            if folder(SKILLS):
                linked(SKILLS, SKILLS_WAS)
            else:
                marked(SKILLS_WAS + NONE)
        synced()
        os.replace(STAGED, CURRENT)
    except OSError as exc:
        print(f"the settings volume: {exc}", file=sys.stderr)
        try:
            recover()
        except OSError:
            pass  # the next run's recover() undoes it
        sys.exit(UNCHANGED)
    recover()
    os.chown(FOLDER, OWNER, os.stat(CURRENT).st_gid if group is None else group)
    os.chmod(FOLDER, 0o750)
    synced()


def deliver(text, group, keep, skills):
    """TEXT as the layer, and SKILLS (each file's path in the folder, with forward slashes -> its text; None: none)
    as the team's skills, both GROUP's."""

    def stage():
        written(STAGED, text, group)
        synced()
        if skills is None:
            marked(SKILLS_STAGED + NONE)
            return
        made(SKILLS_STAGED, OWNER, group, 0o750)
        for rel, body in sorted(skills.items()):
            parts = rel.split("/")
            if any(part in ("", ".", "..") or chr(0) in part for part in parts):
                raise OSError(errno.EINVAL, "no path in the team's skills folder", rel)
            path = SKILLS_STAGED
            for part in parts[:-1]:
                path = os.path.join(path, part)
                if not folder(path):
                    made(path, OWNER, group, 0o750)
            written(os.path.join(path, parts[-1]), body, group)
        synced_tree(SKILLS_STAGED)

    change(stage, keep, group)


def put_back():
    """PREVIOUS as the layer, and SKILLS_PREVIOUS (none, when none is kept) as the team's skills."""

    def stage():
        if not regular(PREVIOUS):
            sys.exit(NOTHING_KEPT)
        os.link(PREVIOUS, STAGED)
        synced()
        if folder(SKILLS_PREVIOUS):
            linked(SKILLS_PREVIOUS, SKILLS_STAGED)
        else:
            marked(SKILLS_STAGED + NONE)

    change(stage, False, None)
'''


class Failed(RuntimeError):
    """A delivery or a put back that did not finish: see the module's docstring. WHY: what happened, the exit code's
    words by default; ABOVE: what the outcome shows above itself first ("" for none)."""

    def __init__(self, code: int, changed: str = "", why: str = "", above: str = ""):
        self.code, self.changed, self.above = code, changed, above
        if not why:
            if code == NOTHING_KEPT:
                why = "the settings volume keeps no earlier layer to put back"
            elif code == UNCHANGED:
                why = "the one-off container that writes the settings volume could not change it"
            else:
                why = f"the one-off container that writes the settings volume ended with exit code {code}"
        super().__init__(f"{why}: {changed or 'nothing changed'}")


class SkillsChanged(Failed):
    """Step 1's failure while the team's skills are not the ones the plan's layer records (changed since the
    confirm): nothing written. WHY: which, in words."""

    def __init__(self, why: str):
        super().__init__(0, why=why)


class Interrupted(KeyboardInterrupt):
    """An interrupt during a delivery's or a put back's run, or after it: CHANGED, what changed, in words."""

    def __init__(self, changed: str):
        self.changed = changed
        super().__init__(changed)


def _kept(path: pathlib.Path) -> pathlib.Path:
    return path.with_name(path.name + KEPT)


def _paths(pkg: pathlib.Path) -> tuple:
    return tuple(pkg / rel for f in (ENV_FILE, OVERRIDE_FILE) for rel in (f, f + KEPT))


def _check(pkg: pathlib.Path) -> None:
    """Step 1's checks of the four paths: Refused, nothing written, when one breaks a rule."""
    for path in _paths(pkg):
        private_files.write(path, "", pkg, replace=True, dry_run=True)


def _team(plan: deploy_plan.DeployPlan):
    """Step 1's read of the team's skills PLAN's layer records: each file's path -> its text, read from the folder
    the plan's settings name; None while the layer records none. SkillsChanged when they are not the files the layer
    records; Failed (its code 0) when one is no UTF-8 text."""
    confirmed = plan.layer.get("skills")
    if confirmed is None:
        return None
    values = plan.layer["settings"]
    folder = deploy_plan.skills_path(values)
    found = deploy_plan.skill_files(values) or {}
    now = deploy_plan.digests(found)
    both = set(now) & set(confirmed)
    differ = sorted(
        (set(now) ^ set(confirmed)) | {r for r in both if now[r] != confirmed[r]}
    )
    if differ:
        names = ", ".join(differ[:3]) + (
            f" and {len(differ) - 3} more" if len(differ) > 3 else ""
        )
        raise SkillsChanged(
            f"the team's skills in {folder} changed since your confirm ({names})"
        )
    team = {}
    for rel, data in found.items():
        try:
            team[rel] = data.decode("utf-8")
        except UnicodeDecodeError:
            raise Failed(
                0, why=f"the team's skill {folder / rel} is no UTF-8 text"
            ) from None
    return team


def _text(path: pathlib.Path, pkg: pathlib.Path):
    """The private file PATH's text, checked first; None when it is not there."""
    if not os.path.lexists(path):
        return None
    p = private_files.checked(path, pkg)
    fd = os.open(p, os.O_RDONLY | private_files.NOFOLLOW)
    with os.fdopen(fd, "rb") as f:
        return f.read().decode("utf-8", "replace")


def _read(path: pathlib.Path, pkg: pathlib.Path):
    """The private file PATH's text (None when it is not there), set to 600 first when it is looser, then checked."""
    if os.path.lexists(path):
        private_files.tighten(path)
    return _text(path, pkg)


def _put(path: pathlib.Path, text, pkg: pathlib.Path) -> None:
    """TEXT as the private file PATH, through private_files at 600; None: no file there."""
    if text is None:
        if os.path.lexists(path):
            os.unlink(path)
    else:
        private_files.write(path, text, pkg, replace=True)


def _put_kept(pkg: pathlib.Path, texts: dict) -> None:
    """TEXTS (a file's path -> its text) as the files' kept copies."""
    for path, text in texts.items():
        _put(_kept(path), text, pkg)


def _changed(volume: str, keep: bool, written: int = 0, which: str = "") -> str:
    """What a delivery or a put back that stopped changed, in words: VOLUME, what the settings volume has; KEEP,
    whether the kept copies are the files as they were; WRITTEN, how many of docker/.env and the override, in that
    order, are the WHICH ones already ("new", or "kept")."""
    if written >= 2:
        files = f"{ENV_FILE} and the override are the {which} ones"
    elif written:
        files = f"{ENV_FILE} is the {which} one, and the override is as it was"
    else:
        files = f"{ENV_FILE} and the override are as they were"
    kept = "; their kept copies (.prev) are the files as they were" if keep else ""
    return f"the settings volume has {volume}; {files}{kept}"


def layer_program(call: str) -> bytes:
    """What the one-off container runs on its input: see the module's docstring."""
    head = (
        f"LAYER_FILE = {LAYER_FILE!r}\nLAYER_KEPT = {LAYER_KEPT!r}\n"
        f"SKILLS_FOLDER = {SKILLS_FOLDER!r}\nSKILLS_KEPT = {SKILLS_KEPT!r}\n"
        f"NEW = {JOURNAL_NEW!r}\nNONE = {JOURNAL_NONE!r}\n"
        f"NOTHING_KEPT = {NOTHING_KEPT}\nUNCHANGED = {UNCHANGED}\n"
    )
    return (head + LAYER_PROGRAM + call + "\n").encode("utf-8")


def _layer_run(compose, variables: dict, call: str, unknown: str) -> int:
    """LAYER_PROGRAM's CALL through COMPOSE in the compose project VARIABLES name, with them in docker compose's
    environment: its exit code; Interrupted, UNKNOWN what changed, for an interrupt."""
    args = ["-p", variables["COMPOSE_PROJECT_NAME"], *LAYER_RUN]
    try:
        return compose(args, layer_program(call), dict(variables))
    except KeyboardInterrupt:
        raise Interrupted(unknown) from None


def _after_run(what: str, changed: str, call, *args, **kwargs) -> None:
    """CALL(*ARGS, **KWARGS), which writes WHAT after a run that ended well; when it cannot, Failed (Interrupted, for
    an interrupt) says CHANGED, what changed."""
    try:
        call(*args, **kwargs)
    except KeyboardInterrupt:
        raise Interrupted(changed) from None
    except private_files.Refused as exc:
        # it shows itself above the outcome (Failed's `above`): its rows, then where the rules are
        why = f"{what} could not be written (the lines above say why)"
        raise Failed(0, changed, why, above=str(exc)) from exc
    except (OSError, ValueError) as exc:
        raise Failed(0, changed, f"{what} could not be written ({exc})") from exc


def _put_after(
    pkg: pathlib.Path, texts: dict, volume: str, which: str, keep: bool
) -> None:
    """TEXTS (docker/.env's place in the package folder -> its text, then the override's) into place after a run
    that ended well, the settings volume holding VOLUME, each file's fingerprint kept once it is written (`wrote`);
    when a file or its fingerprint cannot be written, Failed (Interrupted, for an interrupt) says what changed, the
    files written being the WHICH ones."""
    for done, (rel, text) in enumerate(texts.items()):
        _after_run(rel, _changed(volume, keep, done, which), _put, pkg / rel, text, pkg)
        field = {_FIELDS[rel]: text}
        changed = _changed(volume, keep, done + 1, which)
        _after_run(records.FILE, changed, wrote, pkg, **field)


def deliver(
    plan: deploy_plan.DeployPlan, keep_previous: bool = True, *, package_dir, compose
) -> None:
    """PLAN into place: see the module's docstring."""
    pkg = pathlib.Path(package_dir)
    files = (pkg / ENV_FILE, pkg / OVERRIDE_FILE)
    _check(pkg)
    env, override, layer = texts(plan)
    team = _team(plan)
    group = int(plan.variables["RKMON_GID"])
    was = {}  # each kept copy as it was, put back when the run changes nothing
    if keep_previous:
        now = {f: _read(f, pkg) for f in files}
        was = {f: _read(_kept(f), pkg) for f in files}
        try:
            _put_kept(pkg, now)
        except BaseException:
            _put_kept(pkg, was)
            raise
    if keep_previous:
        volume = "the new layer, the earlier one kept"
        unknown = "either its layers as they were or the new one, the earlier one kept"
    else:
        volume, unknown = "the new layer", "either its layer as it was or the new one"
    call = f"deliver({layer!r}, {group}, {bool(keep_previous)}, {team!r})"
    code = _layer_run(compose, plan.variables, call, _changed(unknown, keep_previous))
    if code == UNCHANGED:
        _put_kept(pkg, was)
        raise Failed(code)
    if code:
        raise Failed(code, _changed(unknown, keep_previous))
    changed = _changed(volume, keep_previous)
    _after_run(records.FILE, changed, wrote, pkg, layer=layer, keep=keep_previous)
    texts_after = {ENV_FILE: env, OVERRIDE_FILE: override}
    _put_after(pkg, texts_after, volume, "new", keep_previous)


def keep_previous(package_dir, plan: deploy_plan.DeployPlan, read) -> bool:
    """Whether a delivery of PLAN keeps what is in place: see the module's docstring."""
    layer = texts(plan)[2]
    written = records.read(package_dir).written
    if written is not None and written.layer == fingerprint(layer):
        return False
    found, why = read()
    if why:
        raise Failed(
            0, why=f"{LAYER_NAME} could not be read, to keep it for rollback ({why})"
        )
    return found != layer


def kept(package_dir) -> bool:
    """Whether a delivery kept what was there before it."""
    return os.path.lexists(pathlib.Path(package_dir) / (ENV_FILE + KEPT))


def kept_differs(package_dir) -> bool:
    """Whether what a delivery kept differs from what is in place: see the module's docstring."""
    pkg = pathlib.Path(package_dir)
    if not kept(pkg):
        return False
    for rel in (ENV_FILE, OVERRIDE_FILE):
        if _text(pkg / rel, pkg) != _text(_kept(pkg / rel), pkg):
            return True
    w = records.read(pkg).written
    return w is None or "" in (w.layer, w.kept_layer) or w.layer != w.kept_layer


def on_host(package_dir) -> tuple:
    """docker/.env's and the override's texts as they are: see the module's docstring."""
    pkg = pathlib.Path(package_dir)
    return _text(pkg / ENV_FILE, pkg), _text(pkg / OVERRIDE_FILE, pkg)


def kept_on_host(package_dir) -> tuple:
    """docker/.env's and the override's kept copies' texts: see the module's docstring."""
    pkg = pathlib.Path(package_dir)
    return _text(_kept(pkg / ENV_FILE), pkg), _text(_kept(pkg / OVERRIDE_FILE), pkg)


def skills_with_layer(skills) -> pathlib.Path | None:
    """The team's skills that go with the layer in place: see the module's docstring."""
    skills = pathlib.Path(skills)
    staged = skills.with_name(skills.name + JOURNAL_NEW)

    def folder(path: pathlib.Path) -> bool:
        return path.is_dir() and not path.is_symlink()

    if not os.path.lexists(skills.with_name(LAYER_FILE + JOURNAL_NEW)):
        if folder(staged):
            return staged
        if os.path.lexists(staged.with_name(staged.name + JOURNAL_NONE)):
            return None
    return skills if folder(skills) else None


def texts(plan: deploy_plan.DeployPlan) -> tuple:
    """What deliver writes for PLAN: see the module's docstring."""
    return plan.env, plan.override or None, deploy_plan.layer_text(plan.layer)


def differing(made: deploy_plan.DeployPlan, env, override, layer) -> tuple:
    """What deliver writes that differs from what it writes for MADE: see the module's docstring."""
    found = (env, override, layer)
    return tuple(n for n, a, b in zip(WRITES, found, texts(made)) if a != b)


def fingerprint(text) -> str:
    """The fingerprint of a file whose text is TEXT: see the module's docstring."""
    if text is None:
        return NO_FILE
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def wrote(package_dir, *, keep: bool = False, put_back: bool = False, **texts) -> None:
    """What a delivery or a put-back wrote, kept by fingerprint: see the module's docstring."""
    fields = set(_FIELDS.values())
    if not set(texts) <= fields:
        unknown = ", ".join(sorted(set(texts) - fields))
        raise TypeError(f"no file deliver writes is kept as {unknown}")

    def change(w: records.Written) -> records.Written:
        new = {field: fingerprint(text) for field, text in texts.items()}
        if keep:
            new["kept_layer"] = w.layer
        if put_back:
            new["layer"] = w.kept_layer
        return dataclasses.replace(w, **new)

    records.change_written(package_dir, change)


def changed_outside(written, env, override, layer) -> tuple:
    """The files that differ from what deliver last wrote there: see the module's docstring."""
    if written is None:
        return ()
    found = dict(zip(WRITES, (env, override, layer)))
    return tuple(
        name
        for name in WRITES
        if getattr(written, _FIELDS[name]) not in ("", fingerprint(found[name]))
    )


def put_back(variables: dict, *, package_dir, compose) -> bool:
    """What deliver kept, back in place: see the module's docstring."""
    pkg = pathlib.Path(package_dir)
    if not kept(pkg):
        return False
    if not variables.get("COMPOSE_PROJECT_NAME"):
        raise ValueError(
            "no compose project (COMPOSE_PROJECT_NAME) is given to put back in: nothing changed"
        )
    _check(pkg)
    kept_texts = {
        rel: _read(_kept(pkg / rel), pkg) for rel in (ENV_FILE, OVERRIDE_FILE)
    }
    unknown = _changed("either its layer as it was or the kept one", False)
    code = _layer_run(compose, variables, "put_back()", unknown)
    if code in (NOTHING_KEPT, UNCHANGED):
        raise Failed(code)
    if code:
        raise Failed(code, unknown)
    volume = "the kept layer"
    _after_run(records.FILE, _changed(volume, False), wrote, pkg, put_back=True)
    _put_after(pkg, kept_texts, volume, "kept", False)
    return True
