# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""What confirmed deploy settings make on this host: docker/.env, the compose override
and the delivered layer, with the settings' fingerprint. The plan is pure: `deliver` writes it; the change plan makes
it again from the layer that runs, and compares it with what is on this host.

- `Host(uid, gid, confirmed_at)`: what the plan reads besides the settings: your user's and your group's ids on this
  host, which a blank host.users.monitor_uid or monitor_gid stands for (the schema's "your own user id on this host"),
  and when the settings were confirmed, as this folder's records keep it.
- `plan(doc, package, code, host) -> DeployPlan(env, override, layer, fingerprint, variables)`: DOC, a deploy
  settings document in the layout this package reads (a ValueError for any other document); PACKAGE, this package's
  version (image.package_version); CODE, the SHA-256 of its code (image.code_sha256).
  - `variables`: docker/.env's variables, NAME -> text (`env_values`).
  - `env`: docker/.env's whole text: its comment lines (ENV_HEADER), then `NAME=value` for each of the variables.
  - `override`: docker/compose.override.yaml's whole text, or "" when it would add nothing to compose.yaml (then
    there is no such file). It adds to the compose service SERVICE the team dashboard's port while the container
    serves it (`served`): the address and the host port docker/.env gives (RKMON_TEAM_HOST, RKMON_TEAM_PORT; the
    address is features.team_host's: in prototype this host's own, whatever serve.address says), to the port the
    dashboard listens on inside (container_layout.TEAM_PORT); and while host.resources.ramdisk_gib is above 0, the RAM
    disk: a tmpfs at container_layout.RAMDISK of that many GiB, rounded up to a whole one.
  - `layer`: the delivered layer, a mapping: its `kind` and `schema_version` (container_layout's LAYER_KIND and
    LAYER_SCHEMA_VERSION); the deploy schema's version; PACKAGE; CODE (`code_sha256`); the confirm's time (HOST's);
    the fingerprint; the `settings` (`settings(doc)`); `set_in_file` (`set_in_file(doc)`), the keys of the settings
    your file sets, so that `rk config show --source` tells them from the defaults; and the `container`: its
    compose project (container.name), the host port of each dashboard it publishes (`admin`, and `team` while it is
    served), the admin dashboard's host address (container_layout.LOOPBACK: this host only), and its image
    (`registry`, `name`, `tag`); and while skills_folder is set, `skills`: the team's skills folder's files, by path
    and SHA-256 (`skills`), which the fingerprint covers and `deliver` checks the files it copies against.
    `layer_text(layer)`: the layer's file: canonical JSON (settings_file.canonical, keys sorted), indented, so that
    the same settings make the same text whatever the order of a map's entries in your file.
  - `fingerprint`: `fingerprint(doc, package, code)`.
- `from_layer(layer, uid, gid) -> DeployPlan`: the plan a delivered layer LAYER (a mapping, as `layer` is) was made
  by, made again from what it holds: its `settings` and `set_in_file`, its deploy schema's version, its package, its
  code and its confirm's time, on a host whose ids are UID and GID. A layer as deliver wrote it gives its own text
  again (`layer_text`); one edited by hand does not, since its fingerprint is made again from its settings.
- `read_layer(text) -> dict`: the delivered layer whose file is TEXT, as a mapping, as `layer` is (but for one edited
  by hand); {} for None, no file, or a text that is no JSON object.
- `settings(doc)`: every deploy key at its effective value (settings_file.effective), the sections LEFT_OUT names left
  out: `updates`, which only the host reads.
- `set_in_file(doc)`: the keys of `settings(doc)` DOC's file sets (settings_file.keys_set), LEFT_OUT left out. Each
  other key is at its default.
- `fingerprint(doc, package, code)`: "sha256:…" through settings_file.fingerprint, over the deploy schema's
  version, PACKAGE, CODE and `settings(doc)`, and while skills_folder is set, `skills`: other code under the same
  version is a change too, and so is a changed, added or removed file of the team's skills. It leaves out the
  package's and the repository's defaults, state, derived values (the ids a blank key stands for among them),
  secrets, the files the plan makes, the confirm's time, `updates`, and `set_in_file(doc)`: a key your file sets at
  its default runs as it does left blank.
- `skills_path(values)`: the team's skills folder the settings VALUES name (skills_folder), its ~ expanded;
  None while the key is blank.
- `skill_files(values)`: that folder's files, read now: each regular file's path in it, with forward slashes, -> its
  bytes (`regular_bytes`). A link, any other kind of file and a file that cannot be read are left out: check 18 names
  each, and its FAIL stops the confirm. None while the key is blank.
- `regular_bytes(path)`: PATH's bytes when it is a regular file, opened without following a link or waiting on a
  pipe; None for any other kind of file, or one that cannot be read. Check 18 reads the folder's files with it too.
- `skills(values)`: `digests(skill_files(values))`; None while the key is blank. `digests(files)`: FILES (path ->
  bytes) -> each one's SHA-256, as the layer's `skills` keeps them.
- `file_sha256(doc)`: "sha256:…" over the file's own values (each blank key as the file leaves it), LEFT_OUT left out,
  and while skills_folder is set, the team's skills (`skills`):
  the records keep it at a confirm, so that a change of your file is told from a change of this package (state 5).
- `served(values)`: whether the container serves the team dashboard, by the settings VALUES (`settings(doc)`), so
  that its port opens on this host: the dashboard is on, by the rule the monitor reads (deploy_tree.rules()), in
  either mode (in prototype the page is made and served as in live), and its delivery is "serve".
- `env_values(values, host)`: docker/.env's variables for the settings VALUES on HOST, NAME -> text, in ENV_KEYS'
  order: the team dashboard's two only while it is served; for a blank id, HOST's.
- ENV_KEYS: each variable docker/.env may hold, in its order, with the deploy keys it is made from (those that decide
  whether it is there among them); OVERRIDE_KEYS: the deploy keys the override is made from. Docker reads both files
  only when it makes the container.
- SERVICE: the compose service in docker/compose.yaml, the monitor's container.

Standard library only, on the host's Python (deployment.HOST_PYTHON): rkmon runs it on the host.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import pathlib
import stat
from typing import NamedTuple

from . import container_layout, deploy_tree, features, file_types, settings_file
from .coherence import TEAM
from .from_layout1 import UPDATES

_NOFOLLOW = getattr(os, "O_NOFOLLOW", 0)  # a link at the path fails the open
_NONBLOCK = getattr(os, "O_NONBLOCK", 0)  # a pipe at the path does not hold the open
SERVICE = "rkmon"
LEFT_OUT = (UPDATES,)
SKILLS_KEY = "features.review_automation.skills_folder"
TEAM_SWITCH = (f"{TEAM}.enabled", f"{TEAM}.mode", f"{TEAM}.delivery")
RAMDISK_GIB = "host.resources.ramdisk_gib"
ENV_KEYS = {
    "COMPOSE_PROJECT_NAME": ("container.name",),
    "RKMON_IMAGE": ("image.name",),
    "RKMON_TAG": ("image.tag",),
    "RKMON_ADMIN_HOST": (),
    "RKMON_PORT": ("features.dashboards.admin.port",),
    "RKMON_TEAM_HOST": (f"{TEAM}.serve.address", *TEAM_SWITCH),
    "RKMON_TEAM_PORT": (f"{TEAM}.serve.port", *TEAM_SWITCH),
    "RKMON_CPUS": ("host.resources.cpus",),
    "RKMON_MEMORY": ("host.resources.memory_gib",),
    "RKMON_UID": ("host.users.monitor_uid",),
    "RKMON_GID": ("host.users.monitor_gid",),
}
OVERRIDE_KEYS = (RAMDISK_GIB, *TEAM_SWITCH)

# The comment lines both files' headers end with.
_EDIT_HERE = (
    "# Change your deploy settings file instead, then run rkmon confirm and rkmon apply: an edit here shows as",
    '# "changed outside your file" in rkmon status, and rkmon apply puts it back.',
)
ENV_HEADER = (
    "# docker/.env: the values docker compose reads for this deployment, made from your confirmed deploy settings.",
    "# rkmon writes it at the deployment's first start, and at each rkmon apply and rkmon rollback (the previous file",
    "# kept as .env.prev).",
    *_EDIT_HERE,
)
OVERRIDE_HEADER = (
    "# docker/compose.override.yaml: what docker compose adds to compose.yaml for this deployment, made from your",
    "# confirmed deploy settings: the team dashboard's port while the container serves it, and the RAM disk, a tmpfs",
    "# that counts against the container's memory limit. rkmon writes it with docker/.env (the previous file kept as",
    "# compose.override.yaml.prev).",
    *_EDIT_HERE,
)


class Host(NamedTuple):
    """What the plan reads besides the settings: see the module's docstring."""

    uid: int
    gid: int
    confirmed_at: str


class DeployPlan(NamedTuple):
    """What confirmed deploy settings make: see the module's docstring."""

    env: str
    override: str
    layer: dict
    fingerprint: str
    variables: dict


def _without(values: dict) -> dict:
    return {k: v for k, v in values.items() if k not in LEFT_OUT}


def settings(doc: settings_file.Document) -> dict:
    """Every deploy key at its effective value, LEFT_OUT left out."""
    return _without(settings_file.effective(doc))


def set_in_file(doc: settings_file.Document) -> list:
    """The keys of `settings(doc)` DOC's file sets."""
    return [k for k in settings_file.keys_set(doc) if k.split(".")[0] not in LEFT_OUT]


def skills_path(values: dict) -> pathlib.Path | None:
    """The team's skills folder the settings VALUES name (SKILLS_KEY): see the module's docstring."""
    folder = _get(values, SKILLS_KEY)
    if not isinstance(folder, str) or not folder.strip():
        return None
    return pathlib.Path(os.path.expanduser(folder.strip()))


def regular_bytes(path: pathlib.Path) -> bytes | None:
    """PATH's bytes when it is a regular file, or None: see the module's docstring."""
    try:
        fd = os.open(path, os.O_RDONLY | _NOFOLLOW | _NONBLOCK)
    except OSError:
        return None
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            return None
        chunks = []
        while True:
            chunk = os.read(fd, 1 << 16)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    except OSError:
        return None
    finally:
        os.close(fd)


def skill_files(values: dict) -> dict | None:
    """The files of the team's skills folder the settings VALUES name, read now: see the module's docstring."""
    top = skills_path(values)
    if top is None:
        return None
    found = {}
    for f in sorted(top.rglob("*")):
        data = regular_bytes(f)
        if data is not None:
            found[f.relative_to(top).as_posix()] = data
    return found


def digests(files: dict) -> dict:
    """FILES (path -> bytes) -> each one's SHA-256: see the module's docstring."""
    return {rel: hashlib.sha256(data).hexdigest() for rel, data in files.items()}


def skills(values: dict) -> dict | None:
    """The team's skills folder the settings VALUES name (SKILLS_KEY), read now: each of its
    files' path -> its SHA-256 (`skill_files`); None while the key is blank."""
    files = skill_files(values)
    return None if files is None else digests(files)


def _fingerprint(
    values: dict, schema_version: int, package: str, code: str, team=None
) -> str:
    """The fingerprint of the settings VALUES in the deploy schema's version SCHEMA_VERSION with PACKAGE, whose
    code's SHA-256 is CODE, and while skills_folder is set, TEAM: its files by path and SHA-256 (`skills`).
    """
    made = {
        "deploy_schema_version": schema_version,
        "package_version": package,
        "code_sha256": code,
        "settings": values,
    }
    return settings_file.fingerprint(made if team is None else {**made, "skills": team})


def fingerprint(doc: settings_file.Document, package: str, code: str) -> str:
    """The fingerprint of DOC's settings with this package, PACKAGE, whose code's SHA-256 is CODE, and the files of
    the skills folder they name, read now (`skills`)."""
    values = settings(doc)
    return _fingerprint(values, doc.file_type.version, package, code, skills(values))


def file_sha256(doc: settings_file.Document) -> str:
    """The SHA-256 of the file's own values, LEFT_OUT left out, and while skills_folder is set, of the team's skills
    folder's files, read now (`skills`), so that a changed skill is a change of your settings too (state 5).
    """
    own, team = _without(doc.values), skills(settings(doc))
    return settings_file.fingerprint(
        own if team is None else {"values": own, "skills": team}
    )


def _get(values: dict, dotted: str):
    value = deploy_tree.get(values, dotted)
    return None if value is deploy_tree.MISSING else value


def served(values: dict) -> bool:
    """Whether the container serves the team dashboard, by the settings VALUES."""
    on = deploy_tree.rules()[f"{TEAM}.enabled"].value(values)
    return on is True and _get(values, f"{TEAM}.delivery") == "serve"


def _number(value) -> str:
    return f"{value:g}"


def env_values(values: dict, host: Host) -> dict:
    """docker/.env's variables for the settings VALUES on HOST, in ENV_KEYS' order."""
    uid, gid = (_get(values, f"host.users.monitor_{k}") for k in ("uid", "gid"))
    out = {
        "COMPOSE_PROJECT_NAME": _get(values, "container.name"),
        "RKMON_IMAGE": _get(values, "image.name"),
        "RKMON_TAG": _get(values, "image.tag"),
        "RKMON_ADMIN_HOST": container_layout.LOOPBACK,
        "RKMON_PORT": _get(values, "features.dashboards.admin.port"),
        "RKMON_CPUS": _number(_get(values, "host.resources.cpus")),
        "RKMON_MEMORY": _number(_get(values, "host.resources.memory_gib")) + "g",
        "RKMON_UID": host.uid if uid is None else uid,
        "RKMON_GID": host.gid if gid is None else gid,
    }
    if served(values):
        address = _get(values, f"{TEAM}.serve.address")
        out["RKMON_TEAM_HOST"] = features.team_host(values, address)
        out["RKMON_TEAM_PORT"] = _get(values, f"{TEAM}.serve.port")
    return {name: str(out[name]) for name in ENV_KEYS if name in out}


def _env_text(variables: dict) -> str:
    lines = [*ENV_HEADER, *(f"{name}={value}" for name, value in variables.items())]
    return "\n".join(lines) + "\n"


def _override_text(values: dict) -> str:
    lines = []
    if served(values):
        lines += [
            "    ports:",
            f"      - target: {container_layout.TEAM_PORT}",
            '        published: "${RKMON_TEAM_PORT}"',
            '        host_ip: "${RKMON_TEAM_HOST}"',
        ]
    size_gib = math.ceil(_get(values, RAMDISK_GIB) or 0)
    if size_gib:
        lines += [
            "    tmpfs:",
            f"      - {container_layout.RAMDISK}:size={size_gib}g,mode=1777,exec",
        ]
    if not lines:
        return ""
    return "\n".join([*OVERRIDE_HEADER, "services:", f"  {SERVICE}:", *lines]) + "\n"


def _layer(
    schema_version: int,
    package: str,
    code: str,
    host: Host,
    values: dict,
    set_keys: list,
    fp: str,
    variables: dict,
    team=None,
) -> dict:
    ports = {"admin": int(variables["RKMON_PORT"])}
    if "RKMON_TEAM_PORT" in variables:
        ports["team"] = int(variables["RKMON_TEAM_PORT"])
    layer = {
        "kind": container_layout.LAYER_KIND,
        "schema_version": container_layout.LAYER_SCHEMA_VERSION,
        "deploy_schema_version": schema_version,
        "package_version": package,
        "code_sha256": code,
        "confirmed_at": host.confirmed_at,
        "fingerprint": fp,
        "settings": values,
        "set_in_file": list(set_keys),
        "container": {
            "project": _get(values, "container.name"),
            "ports": ports,
            "address": container_layout.LOOPBACK,
            "image": {
                part: _get(values, f"image.{part}")
                for part in ("registry", "name", "tag")
            },
        },
    }
    return layer if team is None else {**layer, "skills": team}


def layer_text(layer: dict) -> str:
    """The delivered layer's file: LAYER as canonical JSON, indented."""
    return (
        json.dumps(
            settings_file.canonical(layer),
            sort_keys=True,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    )


def read_layer(text) -> dict:
    """The delivered layer whose file is TEXT, as a mapping: see the module's docstring."""
    try:
        layer = json.loads(text) if text is not None else None
    except ValueError:
        return {}
    return layer if isinstance(layer, dict) else {}


def plan(
    doc: settings_file.Document, package: str, code: str, host: Host
) -> DeployPlan:
    """What DOC makes with this package, PACKAGE, whose code's SHA-256 is CODE, on HOST: see the module's
    docstring."""
    deploy = file_types.DEPLOY
    if doc.file_type.kind != deploy.kind or doc.version != deploy.version:
        raise ValueError(
            f"the plan reads {deploy.noun} in layout {deploy.version}, not {doc.file_type.noun} in layout "
            f"{doc.version}"
        )
    values, set_keys = settings(doc), set_in_file(doc)
    return _made(
        values, set_keys, doc.file_type.version, package, code, host, skills(values)
    )


def from_layer(layer: dict, uid: int, gid: int) -> DeployPlan:
    """The plan the delivered layer LAYER was made by, made again: see the module's docstring."""
    host = Host(uid, gid, layer["confirmed_at"])
    schema_version = layer["deploy_schema_version"]
    return _made(
        layer["settings"],
        layer["set_in_file"],
        schema_version,
        layer["package_version"],
        layer["code_sha256"],
        host,
        layer.get("skills"),
    )


def _made(
    values: dict,
    set_keys: list,
    schema_version: int,
    package: str,
    code: str,
    host: Host,
    team=None,
) -> DeployPlan:
    """What the settings VALUES (`settings(doc)`), of which the file sets SET_KEYS (`set_in_file(doc)`), in the deploy
    schema's version SCHEMA_VERSION make with PACKAGE, whose code's SHA-256 is CODE, on HOST.
    """
    fp = _fingerprint(values, schema_version, package, code, team)
    variables = env_values(values, host)
    return DeployPlan(
        env=_env_text(variables),
        override=_override_text(values),
        layer=_layer(
            schema_version, package, code, host, values, set_keys, fp, variables, team
        ),
        fingerprint=fp,
        variables=variables,
    )
