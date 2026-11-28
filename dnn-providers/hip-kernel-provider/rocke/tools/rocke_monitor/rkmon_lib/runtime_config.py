# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The running configuration: each key of the registry (rkmon_lib/registry.py) from
its owner's layer, with the layer it came from.

Each key's value, by its owner:
- a deploy key: the delivered deploy layer's, through the carry from the deploy settings into the configuration
  (deploy_tree.to_config); else the key's default.
- a package key: the repository defaults' (those for the configuration's repository and folder,
  `repository_for`), else config.defaults.json's.
- a state key: the internal state's (rkmon_lib/state_store.py), else the key's default.
- a derived key: its function's (rkmon_lib/derived.py), over the keys before it.

config.json is one more source while the load reads no layer; beside one it is never read, and the one-time move
splits it by owner, once (rkmon_lib/one_time_move.py). A key it holds keeps its value, but a state key the
internal state holds, and a package key the repository defaults set that it holds blank or at config.defaults.json's
value, which a config.json written in full holds for every package key (the repository defaults' value wins:
repository_defaults.own); its derived keys are given to their functions (derived.Layout). A key it holds that the
registry does not know is not read.

- SOURCES: where a value comes from: DEPLOY, REPOSITORY, PACKAGE, STATE and DERIVED (registry's owners and REPOSITORY),
  and CONFIG_JSON. A deploy key comes from DEPLOY when the delivered layer holds it, from CONFIG_JSON when no layer is
  delivered and config.json holds it, and else, at its default, from PACKAGE; which of the layer's keys your file
  sets is `deploy_source`'s. A state key at its default comes from STATE; a derived key that keeps the value
  config.json gives it comes from CONFIG_JSON.
- `Resolved(values, source, problems, layer)`: VALUES, the configuration, nested and strict (registry.strict: reading
  a key the registry does not know raises); SOURCE, dotted key -> its source; PROBLEMS, what keeps the configuration
  from running (config.problems), one line each; LAYER, the layer `load` read (the delivered one, or the one handed to
  the container in its place), None when it read none.
- `resolve(package, repository, deploy, state, layout, carried=None) -> Resolved`: pure: no network, no clock and no
  file but the package's own. PACKAGE, REPOSITORY and STATE hold their keys nested, under the configuration's names;
  DEPLOY is the configuration the delivered layer's settings make (deploy_tree.to_config), None when no layer is
  delivered; LAYOUT is derived.Layout; CARRIED is config.json's values (`carried`), None without one, whose deploy
  keys are not read while DEPLOY is given.
- `repository_for(deploy, carried) -> dict`: the values of the repository defaults that apply to the configuration's
  repository and folder (repository_defaults.find), nested; {} when none do. DEPLOY and CARRIED as `resolve` takes
  them.
- `from_settings(settings, state=None, carried=None, container=True, image_venv="") -> Resolved`: the configuration a
  deploy layer's SETTINGS make, in the container by default: what the container's load runs, and what the host's
  confirm shows it will run.
- `load(home=None, settings_dir=None, config_file=None) -> Resolved`: the configuration the monitor runs, from the
  delivered layer (`read_layer`: SETTINGS_DIR's in the container, which is where it is delivered, and in its place the
  layer handed to a one-off container of the logins' walk, `handed_layer`; on a host, only SETTINGS_DIR's when given),
  the internal state of HOME (the monitor's home, config.home() by default) and, while it reads no layer, config.json
  (CONFIG_FILE, HOME's by default). config.NoConfiguration (a ConfigError) when neither the layer nor config.json is
  there, and ConfigError when config.json is no JSON object; LayerError when the layer cannot run, and when that
  SETTINGS_DIR has none while HOME's monitor has run one, as its applied record says, or while that record cannot be
  read (the monitor never goes back to config.json's deploy keys); HandedError when the handed layer is not read;
  state_store.StateError when the internal state cannot be read or breaks its schema. Each names its fix
  (config.ConfigError.fix): your deploy settings confirmed and applied where there is no configuration
  (doctor.apply_commands), and each other the one its function names. It only reads: the problems are the caller's
  to refuse (config.load), and the applied record is rk start's and `rk applied --load`'s to write (`record_applied`).
- `folders(home=None, config_file=None) -> Resolved`: the configuration that names the folders, which the emergency
  brake runs with (config.folders): config.json, as `load` reads it, and in the container its layout, over the
  package's defaults; never the delivered layer or the internal state, so that neither keeps the monitor from
  stopping. In the container it reads config.json only while no layer is delivered there (`delivered`).
  config.NoConfiguration on a host without config.json, and ConfigError when config.json is no JSON object, as `load`
  says.
- `config_json(path, data=None) -> dict | None`: config.json at PATH as it is written: DATA, its bytes when they are
  read already, else PATH's; None when there is no file. ConfigError when it is no JSON object, its fix to mend it or
  to move it aside, then to confirm and apply your deploy settings (settings_file.aside_fix, doctor.apply_commands).
- `carried(path, notes=None, data=None) -> dict | None`: config.json at PATH (`config_json`, DATA as it takes it) as
  the configuration reads it: its older keys renamed (renames.migrate) and its first review pass filled in as today's
  code reads it (config.first_pass). NOTES, a list, gets renames.migrate's lines, which name the file.

The delivered deploy layer: LAYER_FILE in SETTINGS_DIR, of the kind LAYER_KIND at LAYER_SCHEMA_VERSION:
`package_version`, `code_sha256` and `deploy_schema_version`, the package (its version, and the SHA-256 of its code,
image.code_sha256) and the deploy settings' layout it was confirmed for; `confirmed_at`, when they were confirmed;
`fingerprint`, the confirmed settings' fingerprint; `settings`, every key of the deploy settings at its effective
value (settings_file.effective); `set_in_file`, the keys of those settings your file sets (deploy_plan.set_in_file),
each other one being at its default; `container`, the container's project, ports, address and image. Only `deliver`
writes it: at a first start, `rkmon apply` and `rkmon rollback`.

- `read_layer(settings_dir=SETTINGS_DIR) -> dict | None`: the delivered layer, checked; None when none is
  delivered. LayerError when it cannot be read, a link to no file among them, or `layer_problems` finds a problem.
- `delivered(environ=None) -> bool`: whether your confirmed settings are delivered here, the one test of it: in the
  container (doctor.deployment_image, in ENVIRON: the process's environment when None), the layer's file is in
  SETTINGS_DIR, which tells it unread; a link there to no file counts, and the load stops on it (`read_layer`): nothing
  falls back to config.json. The folders read config.json only while it is False, and the one-time move splits
  config.json only while it is True (rkmon_lib/one_time_move.py).
- `layer_problems(layer, package_version) -> list[str]`: what keeps LAYER from running on the package PACKAGE_VERSION,
  one line each: a layer of another kind or layout, one with no settings or no list of the keys your file sets, one
  confirmed for another package or another layout of the deploy settings.
- `deploy_source(layer, name) -> str`: the source `rk config show` gives NAME, a key of the deploy settings file, in
  LAYER, a delivered layer: DEPLOY when your file sets it, that is when it, or the map that holds it, is among
  the layer's `set_in_file`; else PACKAGE_SOURCE, the default your file leaves out or blank. As the layer holds them:
  those of the settings the first start or the last apply delivered.
- `LayerError(found)`: a ConfigError; FOUND says what is wrong, and `fix(program_name)` is the fix, which the host's
  command PROGRAM_NAME runs: `apply` (this package's code, delivered with the settings) or `rollback`. rk start and
  `rk applied --load` say what is wrong and the fix (applied.cannot_run), and record nothing as applied; the monitor
  runs no tick (`rk features --runs` refuses every part: docker/entrypoint.sh), and the doctor shows a FAIL with the
  fix.

The layer handed to a one-off container of the logins' walk (deploy step 5) or of `rkmon web-host install`:
before the first start no layer is delivered, and after a confirm that is not applied yet the delivered one is not the
one confirmed. So the host's wrapper hands each of those one-off containers (docker/rkmon walk_inside) the layer your
confirmed settings make, the text the first start delivers, and the entry point's root phase writes it, root's, as
LAYER_FILE in HANDED_DIR, in that container's own file system (container_layout.HANDED_LAYER_DIR). The file exists in
those one-off containers alone, never in the running container nor in a start's: what reports on the running monitor
there (`rk applied`, the status) never reads it.

- `handed_layer(folder=None) -> dict | None`: that layer, in FOLDER (HANDED_DIR by default), which the container's
  load reads in place of the delivered one, checked as `read_layer` checks that one; None when none is handed.
  HandedError when FOLDER or the file is a link, is not root's (ROOT_UID's), or may be written by a user other than
  root (a write bit for its group or for others: an ACL that lets another user write sets its group's), or the file
  is no regular file; and when the layer cannot be read or `layer_problems` finds a problem.
- `HandedError(found, untrusted=True)`: a ConfigError; FOUND says what is wrong, in plain words, and
  `fix(program_name)` is the fix: for a folder or file that is not root's alone (UNTRUSTED), an image this package
  built (image.own_image_fix), in whose own file system the root phase alone makes them; else the host's image check
  (`image check`): the layer is the host's package's, the code that reads it the image's.

The applied record: APPLIED_FILE in the home's state folder (config.state_dir), of the kind APPLIED_KIND at
APPLIED_SCHEMA_VERSION: the fingerprint and the package version of the layer the monitor runs, and when it first
started on it.

- `applied(home) -> dict | None`: the record; None when there is none. RecordError when it cannot be read, or is no
  applied record.
- `holds(record, layer) -> bool`: whether the applied record RECORD is LAYER's: both carry the same fingerprint and
  package version (APPLIED_MARK). False when either is None.
- `record_applied(home, resolved, now=None) -> bool`: the record of the layer RESOLVED was loaded from, at the time
  NOW (store.now() by default), written whole. Two commands write it as they load that layer: rk start and
  `rk applied --load` (rkmon_lib/applied.py). ConfigError, its fix doctor.cannot_run_fix, when RESOLVED has
  problems: a configuration that cannot run is never recorded. False, with nothing written, when RESOLVED was loaded
  from no layer, and when the record is the layer's already (`holds`); a record that cannot be read is written anew.
  RecordError, with nothing written, when the state folder cannot be written.
- `RecordError(found, first="")`: the applied record cannot be read or written; FOUND names its file and says why,
  and `fix()` is the fix, as the operator types it here (doctor.rk_command): `rk applied --load`, which writes the
  record again, after FIRST when given.
"""

from __future__ import annotations

import contextlib
import copy
import errno
import json
import os
import pathlib
import stat
from typing import NamedTuple

from . import (
    config,
    container_layout,
    deploy_tree,
    derived,
    file_types,
    image,
    registry,
    renames,
    repository_defaults,
    state_store,
)
from .checks import Command, Steps
from .paths import PACKAGE

DEPLOY, REPOSITORY, PACKAGE_SOURCE, STATE, DERIVED = (
    registry.DEPLOY,
    registry.REPOSITORY,
    registry.PACKAGE,
    registry.STATE,
    registry.DERIVED,
)
CONFIG_JSON = "config.json"
SOURCES = (DEPLOY, REPOSITORY, PACKAGE_SOURCE, STATE, DERIVED, CONFIG_JSON)

# Where the container mounts the settings volume, read-only, and the layer's file, kind and layout there:
# container_layout's
SETTINGS_DIR, LAYER_FILE = container_layout.SETTINGS_DIR, container_layout.LAYER_FILE
LAYER_KIND = container_layout.LAYER_KIND
LAYER_SCHEMA_VERSION = container_layout.LAYER_SCHEMA_VERSION
# The folder of the layer handed to a one-off container of the logins, in its own file system (container_layout's),
# and the one owner its read takes for that folder and its file: root.
HANDED_DIR = container_layout.HANDED_LAYER_DIR
ROOT_UID = 0
APPLIED_FILE = "applied_settings.json"
APPLIED_KIND = "applied_settings"
APPLIED_SCHEMA_VERSION = 1
APPLIED_MARK = (
    "fingerprint",
    "package_version",
)  # what the record and the layer it holds both carry


class Resolved(NamedTuple):
    values: dict
    source: dict
    problems: list
    layer: dict | None = None


class LayerError(config.ConfigError):
    """The delivered deploy layer cannot run: see the module's docstring."""

    def __init__(self, found: str):
        super().__init__(found)
        self.found = found

    @staticmethod
    def fix(program_name: str):
        """The fix, as the host's command PROGRAM_NAME runs it."""
        return Steps(
            (
                Command((program_name, "apply")),
                "(this package's code, delivered with your settings), or",
                Command((program_name, "rollback")),
            )
        )


class HandedError(config.ConfigError):
    """The layer handed to this container is not read: see the module's docstring."""

    def __init__(self, found: str, untrusted: bool = True):
        super().__init__(found)
        self.found, self.untrusted = found, untrusted

    def fix(self, program_name: str):
        """The fix, as the host's command PROGRAM_NAME runs it."""
        if self.untrusted:
            from .image import own_image_fix  # a call-time import, as RecordError's

            return own_image_fix(program_name)
        return Command((program_name, "image", "check"))


class RecordError(Exception):
    """The applied record cannot be read or written: see the module's docstring."""

    def __init__(self, found: str, first: str = ""):
        super().__init__(found)
        self.found, self.first = found, first

    def fix(self):
        """The fix, as the operator types it here."""
        # a call-time import: doctor reads config, which reads this module
        from .doctor import rk_command

        then = f"{self.first}, then run" if self.first else "run"
        return Steps((then, rk_command("applied", "--load")))


# --- resolve ----------------------------------------------------------------------------------------------------------


def _deploy_value(key, deploy, carried):
    if deploy is not None:
        v = registry.get(deploy, key.name)
        if v is registry.MISSING:
            return key.default, PACKAGE_SOURCE
        return v, DEPLOY
    v = registry.get(carried, key.name)
    if v is not registry.MISSING:
        return v, CONFIG_JSON
    return key.default, PACKAGE_SOURCE


def _package_value(key, package, repository, carried, theirs):
    v = registry.get(carried, key.name)
    if v is not registry.MISSING and (
        key.name not in theirs
        or repository_defaults.own(v, registry.get(package, key.name))
    ):
        return v, CONFIG_JSON
    for layer, source in ((repository, REPOSITORY), (package, PACKAGE_SOURCE)):
        v = registry.get(layer, key.name)
        if v is not registry.MISSING:
            return v, source
    return key.default, PACKAGE_SOURCE


def _state_value(key, state, carried):
    for layer, source in ((state, STATE), (carried, CONFIG_JSON)):
        v = registry.get(layer, key.name)
        if v is not registry.MISSING:
            return v, source
    return key.default, STATE


def given(carried) -> dict:
    """The derived keys' values config.json gives, by dotted key (derived.Layout's GIVEN)."""
    out = {}
    for key in registry.keys():
        if key.owner == registry.DERIVED:
            v = registry.get(carried or {}, key.name)
            if v is not registry.MISSING:
                out[key.name] = v
    return out


def resolve(package, repository, deploy, state, layout, carried=None) -> Resolved:
    """Each key's value and source, and the configuration's problems: see the module's docstring."""
    carried, repository = carried or {}, repository or {}
    theirs = set(repository_defaults.keys_of(repository))
    values, source = {}, {}
    for key in registry.keys():
        if key.owner == registry.DEPLOY:
            v, s = _deploy_value(key, deploy, carried)
        elif key.owner == registry.PACKAGE:
            v, s = _package_value(key, package or {}, repository, carried, theirs)
        elif key.owner == registry.STATE:
            v, s = _state_value(key, state or {}, carried)
        else:
            v = derived.FUNCTIONS[key.name](values, layout)
            kept = (layout.given or {}).get(key.name)
            s = CONFIG_JSON if kept not in (None, "") and v == kept else DERIVED
        if v is registry.MISSING:
            continue
        values[key.name], source[key.name] = v, s
    nested = registry.strict(registry.nest(values))
    return Resolved(nested, source, config.problems(nested))


def repository_for(deploy, carried) -> dict:
    """The values of the repository defaults for the configuration's repository and folder: see the module's
    docstring."""
    repo, folder = (
        _deploy_value(registry.key(name), deploy, carried or {})[0]
        for name in ("repo", "rocke_path")
    )
    found = repository_defaults.find(repo, folder)
    return copy.deepcopy(found.values) if found is not None else {}


def from_settings(
    settings, state=None, carried=None, container=True, image_venv=""
) -> Resolved:
    """The configuration a deploy layer's settings make: see the module's docstring."""
    layout = derived.Layout(
        container=container, given=given(carried), image_venv=image_venv
    )
    deploy = deploy_tree.to_config(settings or {})
    return resolve(
        registry.package_defaults(),
        repository_for(deploy, carried),
        deploy,
        state or {},
        layout,
        carried,
    )


# --- the delivered layer ----------------------------------------------------------------------------------------------


def layer_problems(layer, package_version: str) -> list:
    """What keeps LAYER from running on the package PACKAGE_VERSION: see the module's docstring."""
    if not isinstance(layer, dict):
        return ["it is no JSON object"]
    if (
        layer.get("kind") != LAYER_KIND
        or layer.get("schema_version") != LAYER_SCHEMA_VERSION
    ):
        return [
            f"it is of the kind {layer.get('kind')!r} at layout {layer.get('schema_version')!r}, not "
            f"{LAYER_KIND!r} at layout {LAYER_SCHEMA_VERSION}"
        ]
    out = []
    if not isinstance(layer.get("settings"), dict):
        out.append("it has no settings")
    set_keys = layer.get("set_in_file")
    if not isinstance(set_keys, list) or not all(isinstance(k, str) for k in set_keys):
        out.append("it does not list the keys your file sets")
    if layer.get("package_version") != package_version:
        out.append(
            f"it was confirmed for the package {layer.get('package_version')!r}, and this is the package "
            f"{package_version!r}"
        )
    if layer.get("deploy_schema_version") != file_types.DEPLOY.version:
        out.append(
            f"it was confirmed for layout {layer.get('deploy_schema_version')!r} of the deploy settings, and this "
            f"package reads layout {file_types.DEPLOY.version}"
        )
    return out


def deploy_source(layer: dict, name: str) -> str:
    """The source rk config show gives the deploy settings file's key NAME in LAYER: see the module's docstring."""
    set_keys = set(layer.get("set_in_file") or ())
    parts = name.split(".")
    if any(".".join(parts[:n]) in set_keys for n in range(1, len(parts) + 1)):
        return DEPLOY
    return PACKAGE_SOURCE


def _checked(data: bytes, what: str, error) -> dict:
    """The layer DATA holds, WHAT naming it, checked: ERROR(why), raised, when it is no JSON or
    `layer_problems` finds a problem."""
    try:
        layer = json.loads(data.decode("utf-8"))
    except ValueError as exc:
        raise error(f"{what} are no JSON: {exc}") from None
    problems = layer_problems(layer, image.package_version(PACKAGE))
    if problems:
        raise error(f"{what} cannot run: " + "; ".join(problems))
    return layer


def read_layer(settings_dir=SETTINGS_DIR) -> dict | None:
    """The delivered layer, checked; None when none is delivered: see the module's docstring."""
    path = pathlib.Path(settings_dir) / LAYER_FILE
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        # a link to no file is delivered all the same (`delivered`): the load stops on it, and never reads config.json
        if os.path.lexists(path):
            raise LayerError(
                f"your confirmed settings at {path} cannot be read: it is a link to a file that is not there"
            ) from None
        return None
    except OSError as exc:
        raise LayerError(
            f"your confirmed settings at {path} cannot be read: {exc.strerror or exc}"
        ) from None
    return _checked(data, f"your confirmed settings at {path}", LayerError)


def handed_layer(folder=None) -> dict | None:
    """The layer handed to this container, checked; None when none is: see the module's docstring. Only a one-off
    container of the logins' walk has it, never the running container: what reports on the running monitor there (rk
    applied, the status) never reads it."""
    folder = pathlib.Path(HANDED_DIR if folder is None else folder)
    path = folder / LAYER_FILE
    what = f"the confirmed deploy settings handed to this container ({path})"
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    try:
        fd = os.open(folder, flags | os.O_DIRECTORY)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _not_opened(what, folder, exc) from None
    try:
        _root_alone(fd, folder, what)
        try:
            file_fd = os.open(LAYER_FILE, flags | os.O_NONBLOCK, dir_fd=fd)
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise _not_opened(what, path, exc) from None
        try:  # checked first: a folder in its place is no file, which a file object would not open
            _root_alone(file_fd, path, what, regular=True)
        except BaseException:
            os.close(file_fd)
            raise
        with os.fdopen(file_fd, "rb") as f:
            data = f.read()
    except OSError as exc:
        raise HandedError(
            f"{what} cannot be read: {exc.strerror or exc}", untrusted=False
        ) from None
    finally:
        os.close(fd)
    return _checked(data, what, lambda why: HandedError(why, untrusted=False))


def _not_opened(what: str, path, exc: OSError) -> HandedError:
    """The refusal of the handed layer's folder or file PATH, which did not open (EXC): a link, or no folder, is not
    root's alone; else it cannot be read."""
    with contextlib.suppress(OSError):
        if stat.S_ISLNK(os.lstat(path).st_mode):
            return HandedError(f"{what} are not read: {path} is a link")
    if exc.errno == errno.ENOTDIR:
        return HandedError(f"{what} are not read: {path} is no folder")
    return HandedError(f"{what} cannot be read: {exc.strerror or exc}", untrusted=False)


def _root_alone(fd: int, path, what: str, regular: bool = False) -> None:
    """HandedError unless PATH, which FD has open, is root's alone: ROOT_UID's, with no write bit for its group or
    for others; with REGULAR, a regular file too."""
    s = os.fstat(fd)
    if regular and not stat.S_ISREG(s.st_mode):
        raise HandedError(f"{what} are not read: {path} is no file")
    if s.st_uid != ROOT_UID:
        raise HandedError(
            f"{what} are not read: {path} is not root's (user {s.st_uid} owns it)"
        )
    if s.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
        raise HandedError(
            f"{what} are not read: users other than root may write {path} (permissions "
            f"{stat.S_IMODE(s.st_mode):o})"
        )


def delivered(environ=None) -> bool:
    """Whether your confirmed settings are delivered here: see the module's docstring."""
    # a call-time import: doctor reads config, which reads this module
    from . import doctor

    container = doctor.deployment_image(os.environ if environ is None else environ)
    return container and os.path.lexists(pathlib.Path(SETTINGS_DIR) / LAYER_FILE)


# --- the applied record -----------------------------------------------------------------------------------------------


def _applied_path(home) -> pathlib.Path:
    return config.state_dir(home) / APPLIED_FILE


def applied(home) -> dict | None:
    """The applied record: see the module's docstring."""
    path = _applied_path(home)
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        why = exc.strerror or exc
        raise RecordError(f"the applied record {path} cannot be read: {why}") from None
    try:
        rec = json.loads(data.decode("utf-8"))
    except ValueError as exc:
        raise RecordError(f"the applied record {path} is no JSON: {exc}") from None
    if not isinstance(rec, dict) or rec.get("kind") != APPLIED_KIND:
        raise RecordError(
            f"{path} is no applied record: it is not of the kind {APPLIED_KIND!r}"
        )
    return rec


def holds(record: dict | None, layer: dict | None) -> bool:
    """Whether the applied record RECORD is LAYER's: see the module's docstring."""
    if not record or not layer:
        return False
    return all(record.get(k) == layer.get(k) for k in APPLIED_MARK)


def record_applied(home, resolved: Resolved, now: str | None = None) -> bool:
    """The applied record of the layer RESOLVED was loaded from: see the module's docstring."""
    if resolved.problems:
        # a call-time import: doctor reads config, which reads this module
        from .doctor import cannot_run_fix

        raise config.ConfigError(
            "the configuration cannot run, so its layer is not recorded as applied: "
            + "; ".join(resolved.problems),
            cannot_run_fix(),
        )
    layer = resolved.layer
    if layer is None:
        return False
    if now is None:
        from . import (
            store,
        )  # a call-time import: store reads config, which reads this module

        now = store.now()
    try:
        was = applied(home)
    except RecordError:  # one that cannot be read is written anew
        was = None
    if holds(was, layer):
        return False
    record = {
        "kind": APPLIED_KIND,
        "schema_version": APPLIED_SCHEMA_VERSION,
        **{k: layer.get(k) for k in APPLIED_MARK},
        "applied_at": now,
    }
    path = _applied_path(home)
    try:
        state_store.write_file(path, record)
    except OSError as exc:
        folder = path.parent
        raise RecordError(
            f"the applied record cannot be written in {folder}: {exc.strerror or exc}",
            first=f"make {folder} writable",
        ) from None
    return True


# --- load -------------------------------------------------------------------------------------------------------------


def config_json(path, data: bytes | None = None) -> dict | None:
    """config.json at PATH as it is written: see the module's docstring."""
    path = pathlib.Path(path)
    if data is None:
        try:
            data = path.read_bytes()
        except FileNotFoundError:
            return None
    try:
        raw = json.loads(data.decode("utf-8"))
    except ValueError as exc:
        fix = _anew(path, mend=True)
        raise config.ConfigError(f"{path} is not valid JSON: {exc}", fix) from exc
    if not isinstance(raw, dict):
        raise config.ConfigError(f"{path} is no JSON object", _anew(path))
    return raw


def carried(path, notes: list | None = None, data: bytes | None = None) -> dict | None:
    """config.json at PATH as the configuration reads it: see the module's docstring."""
    raw = config_json(path, data)
    if raw is None:
        return None
    raw, lines = renames.migrate(raw, pathlib.Path(path).name)
    if notes is not None:
        notes.extend(lines)
    return dict(raw, reviewer=config.first_pass(raw))


def _anew(path, mend: bool = False) -> Steps:
    """The fix of the config.json at PATH that is no configuration: mend it (MEND), or move it aside, then your deploy
    settings confirmed and applied (doctor.apply_commands).
    """
    # call-time imports: doctor reads config, which reads this module; settings_file is a large module
    from . import doctor, settings_file

    then = (", then", *doctor.apply_commands())
    return settings_file.aside_fix(path, mend=mend, then=then)


def _missing_layer(home, settings_dir) -> None:
    """LayerError, with no layer delivered in SETTINGS_DIR, when HOME's monitor has run one, as its applied record
    says, or that record cannot be read: such a layer cannot run either."""
    missing = f"your confirmed settings at {pathlib.Path(settings_dir) / LAYER_FILE} are missing"
    try:
        ran = applied(home)
    except RecordError as exc:
        raise LayerError(
            f"{missing}, and whether this monitor has run your confirmed settings before is not known: {exc}"
        ) from None
    if ran is not None:
        raise LayerError(
            f"{missing}, and this monitor has run your confirmed settings before (applied at {ran.get('applied_at')})"
        )


def _config_file(home, config_file) -> tuple:
    """HOME (config.home() by default) and its config.json: CONFIG_FILE when given."""
    home = pathlib.Path(home) if home is not None else config.home()
    path = pathlib.Path(config_file) if config_file is not None else home / CONFIG_JSON
    return home, path


def _none_at(path) -> config.NoConfiguration:
    """No configuration at PATH: its fix is your deploy settings confirmed and applied (doctor.apply_commands)."""
    # a call-time import: doctor reads config, which reads this module
    from . import doctor

    return config.NoConfiguration(
        f"no configuration at {path}", Steps(doctor.apply_commands())
    )


def load(home=None, settings_dir=None, config_file=None) -> Resolved:
    """The configuration the monitor runs: see the module's docstring."""
    from . import (
        doctor,
    )  # a call-time import: doctor reads config, which reads this module

    home, path = _config_file(home, config_file)
    container = doctor.deployment_image()
    image_venv, image_llvm_bin, layer = "", "", None
    if container:
        settings_dir = SETTINGS_DIR if settings_dir is None else settings_dir
        venv = container_layout.IMAGE_VENV
        image_venv = venv if os.path.isdir(venv) else ""
        image_llvm_bin = container_layout.rocm_llvm_bin()
        layer = handed_layer()  # only a one-off container of the logins' walk has one
    if layer is None and settings_dir is not None:
        layer = read_layer(settings_dir)
    # With a layer, config.json is never read: the one-time move splits it (rkmon_lib/one_time_move.py).
    from_file = carried(path) if layer is None else None
    if (
        layer is None and settings_dir is not None
    ):  # a host given no SETTINGS_DIR reads no layer
        _missing_layer(home, settings_dir)
    if layer is None and from_file is None:
        raise _none_at(path)
    deploy = deploy_tree.to_config(layer["settings"]) if layer is not None else None
    resolved = resolve(
        registry.package_defaults(),
        repository_for(deploy, from_file),
        deploy,
        state_store.read(home),
        derived.Layout(
            container=container,
            given=given(from_file),
            image_venv=image_venv,
            image_llvm_bin=image_llvm_bin,
        ),
        from_file,
    )
    return resolved._replace(layer=layer)


def folders(home=None, config_file=None) -> Resolved:
    """The configuration that names the folders: see the module's docstring."""
    # a call-time import: doctor reads config, which reads this module
    from . import doctor

    path = _config_file(home, config_file)[1]
    container = doctor.deployment_image()
    from_file = None if delivered() else carried(path)
    if from_file is None and not container:
        raise _none_at(path)
    layout = derived.Layout(container=container, given=given(from_file))
    return resolve(registry.package_defaults(), {}, None, {}, layout, from_file)
