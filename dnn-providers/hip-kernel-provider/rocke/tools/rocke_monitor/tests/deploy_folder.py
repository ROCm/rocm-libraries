# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A deployment's package folder for the tests that run docker/rkmon against the stand-in docker
(tests/fakes/fake_docker): the deploy path's (tests/test_deploy_path_<part>.py) and tests/test_wrapper.py.

- `copy(dest) -> pathlib.Path`: DEST made a copy of what docker/rkmon reads of the package (docker/, rkmon_lib/,
  schema/, agentic/, whose files check 18 names a team's skills replace, the quick deploy guide, the package's
  defaults, VERSION and the .dockerignore that says which of its files `rkmon apply` installs as the container's
  code), with no worker/, as the package folder the deploy path runs from holds none (rkmon_lib/paths.py
  worker_module: a test fails on any worker/ module a deploy command imports), with none of a deployment's own files
  (docker/.env, the compose override, the host's records, a certificate), and the copy the test's to change.
- `write_settings(path, port, ...)`: a deploy settings file at PATH, yours only, as settings_file.dump writes it: the
  container NAME, the admin dashboard on PORT, `features` over FEATURES (a feature that is off as its switch alone),
  `integrations` over the GitHub CLI's login name, `bindings` its model_bindings, then the text `more` (a part the
  writer does not write).
- `HOST_SIZE`: the environment that gives docker/rkmon a host of HOST_MEMORY_GIB GiB of memory and HOST_CPUS
  CPUs (deployment.host_size), room for the container's share write_settings gives it by default, so no test
  depends on the machine it runs on.
- `confirm(package_dir, path, home) -> records.Confirmed`: the settings file PATH confirmed in the package folder's
  records, as `rkmon confirm` records it for a user whose home folder is HOME.
- `plan(package_dir, path) -> deploy_plan.DeployPlan`: what the confirmed settings file PATH makes, as the first
  start delivers it: with the package's version, its code's SHA-256 and the time the confirm recorded, on your
  user's and group's ids.
- `delivered(package_dir, path) -> list`: what the first start delivers for the confirmed settings, the file PATH
  (rkmon_lib/deliver.py), in place without a start: docker/.env and the compose override as deliver writes them,
  yours only, with the fingerprints deliver keeps in the records (deliver.wrote); and the stand-in docker's answers,
  to put first, that the settings volume has the layer deliver writes and the running monitor has loaded it (runs).
- `runs(made) -> list`: the stand-in docker's answers, to put first, that the settings volume has the layer of the
  plan MADE and the running monitor has loaded it (`rk applied --json`).
- `secrets_volumes(here, unheld=()) -> list`: the stand-in docker's answers, to put first, about the volumes that
  keep the logins and secrets before the first start: docker compose config names them in NAME's compose
  project; HERE, both are on this host, and the one-off container that reads them finds that those named UNHELD hold
  nothing; else neither is here yet.
- `volumes(labeled=(), unlabeled=(), env=True) -> list`: the stand-in docker's answers, to put first (before
  secrets_volumes', whose listing of volumes by name they answer too), about NAME's volumes on this host (docker/rkmon:
  own_volumes): LABELED, the names of those docker compose made for NAME's compose project, and UNLABELED, of those
  made otherwise; docker compose config, read as own_volumes reads it while docker/.env is there (ENV: with its
  variables put in) or while it is not (with none, --no-interpolate), and answered for that read alone, declares
  docker/compose.yaml's volumes (VOLUME_KEYS) under their names in that project (VOLUMES) and mounts those that keep
  the logins and secrets as secrets_volumes' answer does, and a listing by name gives each volume here, whatever the
  names asked, as docker's name filter would if it were not exact (it also matches a part of a name): docker/rkmon
  keeps the names it asked for alone.
- `image_code(package_dir, changed=()) -> list`: the stand-in docker's answer, to put first, that the image's own
  copies of the files an image keeps (code_volume.IMAGE_BOUND) are the package folder PACKAGE_DIR's, by the
  fingerprints a one-off container of the image prints (rkmon_lib/confirm.py: CODE_PROBE), but for those CHANGED
  names, whose copies are another's. IMAGE_CODE: its pattern, which finds the call of that one-off container by
  its arguments, the folder of the image's own code and the first file of IMAGE_BOUND.
"""

import dataclasses
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import sandbox  # noqa: E402
from rkmon_lib import (  # noqa: E402
    code_volume,
    container_layout,
    deliver,
    deploy_plan,
    file_types,
    identity,
    image,
    records,
    settings_file,
)
from rkmon_lib.confirm import CODE_PROBE  # noqa: E402

# The container name of the deploy settings the tests write: its compose project too.
NAME = "rkmon-test"
# The volumes docker/compose.yaml declares, its last top-level key, by their keys there, in its order; their names in
# NAME's compose project
(_DECLARED,) = re.findall(
    r"(?ms)^volumes:\n(.*)",
    (ROOT / "docker" / "compose.yaml").read_text(encoding="utf-8"),
)
VOLUME_KEYS = tuple(re.findall(r"(?m)^  ([\w-]+):", _DECLARED))
VOLUMES = tuple(f"{NAME}_{key}" for key in VOLUME_KEYS)
# The host the tests give docker/rkmon (HOST_SIZE): its memory in GiB and its CPUs
HOST_MEMORY_GIB, HOST_CPUS = 64, 16
HOST_SIZE = {"RKMON_HOST_SIZE": f"{HOST_MEMORY_GIB},{HOST_CPUS}"}
# The features of the deploy settings file the tests write, as a deployment starts: review automation and the
# merge queue in prototype, the admin dashboard on, the rest off.
FEATURES = {
    "review_automation": {
        "enabled": True,
        "mode": "prototype",
        "parallel_reviews": 1,
        "remote_gpu_tests": {"enabled": False},
        "spend": {"enabled": False},
    },
    "merge_queue": {
        "enabled": True,
        "mode": "prototype",
        "auto_merge": {"enabled": False},
        "auto_hold": {"enabled": True},
        "queue_notices": {"enabled": True},
        "drift_notices": {"enabled": True},
        "develop_guard": {"enabled": True},
    },
    "dashboards": {"admin": {"enabled": True}, "team": {"enabled": False}},
    "alerts": {"develop_breaks": {"enabled": False}, "needs_you": {"enabled": False}},
    "ticket_creation": {
        "follow_up_tickets": {"enabled": False},
        "found_issue_tickets": {"enabled": False},
    },
    "weekly_report": {"enabled": False},
}


def merged(a: dict, b: dict) -> dict:
    """B laid over A, mappings merged key by key."""
    out = dict(a)
    for k, v in b.items():
        both = isinstance(v, dict) and isinstance(out.get(k), dict)
        out[k] = merged(out[k], v) if both else v
    return out


def folded(section):
    """A feature that is off as its switch alone, as the templates write it."""
    if not isinstance(section, dict):
        return section
    if section.get("enabled") is False:
        return {"enabled": False}
    return {k: folded(v) for k, v in section.items()}


def copy(dest: pathlib.Path) -> pathlib.Path:
    """DEST, a copy of what docker/rkmon reads of the package: see the module's docstring."""
    shutil.copytree(
        ROOT / "docker",
        dest / "docker",
        ignore=shutil.ignore_patterns(
            "__pycache__", ".env*", "compose.override.yaml*", "*.crt", ".rkmon"
        ),
    )
    shutil.copytree(
        ROOT / "rkmon_lib",
        dest / "rkmon_lib",
        ignore=shutil.ignore_patterns("__pycache__"),
    )
    shutil.copytree(ROOT / "schema", dest / "schema")
    shutil.copytree(ROOT / "agentic", dest / "agentic")
    for name in (
        "QUICK_DEPLOY_GUIDE.md",
        "config.defaults.json",
        "VERSION",
        ".dockerignore",
    ):
        shutil.copy(ROOT / name, dest / name)
    return sandbox.writable(dest)


def write_settings(
    path: pathlib.Path,
    port: int,
    features=None,
    image_tag="rkmon-test",
    registry="",
    image_name="",
    memory_gib=12,
    ramdisk_gib=4,
    bindings=None,
    more="",
    integrations=None,
    cpus=2,
) -> None:
    """The deploy settings file at PATH: see the module's docstring."""
    image_part = {"tag": image_tag}
    if image_name:
        image_part["name"] = image_name
    if registry:
        image_part["registry"] = registry
    admin = {"dashboards": {"admin": {"port": port}}}
    values = {
        "container": {"name": NAME},
        "image": image_part,
        "repository": {
            "name": "acme/widgets",
            "folder": "widgets/rocke",
            "scope_labels": ["widgets"],
        },
        "host": {
            "resources": {
                "cpus": cpus,
                "memory_gib": memory_gib,
                "ramdisk_gib": ramdisk_gib,
            },
            "users": {"monitor_uid": 4242, "monitor_gid": 4343},
        },
        "features": folded(merged(merged(FEATURES, admin), features or {})),
        "integrations": merged(
            {"github_cli": {"username": "op-login"}}, integrations or {}
        ),
    }
    if bindings:
        values["model_bindings"] = bindings
    doc = settings_file.Document(file_types.DEPLOY, file_types.DEPLOY.version, values)
    path.write_text(settings_file.dump(doc) + more, encoding="utf-8")
    path.chmod(0o600)


def confirm(package_dir: pathlib.Path, path: pathlib.Path, home) -> records.Confirmed:
    """The settings file PATH confirmed in PACKAGE_DIR's records: see the module's docstring."""
    doc = settings_file.read(path, file_types.DEPLOY)
    version, code = image.package_version(package_dir), image.code_sha256(package_dir)
    host = deploy_plan.Host(os.getuid(), os.getgid(), "")
    made = deploy_plan.plan(doc, version, code, host)
    named = made.layer["container"]["image"]
    confirmed = records.Confirmed(
        settings_file=str(path),
        fingerprint=made.fingerprint,
        at=records.timestamp(),
        file_sha256=deploy_plan.file_sha256(doc),
        package_version=version,
        code_sha256=code,
        container_name=made.variables["COMPOSE_PROJECT_NAME"],
        image=records.Image(named["registry"], named["name"], named["tag"]),
        changes=0,
    )
    with mock.patch.dict(os.environ, {"HOME": str(home)}):
        records.confirm(package_dir, confirmed)
        shown = settings_file.shown(path)
    return dataclasses.replace(confirmed, settings_file=shown)


def plan(package_dir: pathlib.Path, path: pathlib.Path) -> deploy_plan.DeployPlan:
    """What the confirmed settings file PATH makes: see the module's docstring."""
    c = records.read(package_dir).confirmed
    doc = settings_file.read(path, file_types.DEPLOY)
    host = deploy_plan.Host(os.getuid(), os.getgid(), c.at)
    return deploy_plan.plan(doc, c.package_version, c.code_sha256, host)


# The volumes that keep the logins and secrets, by their keys in docker/compose.yaml: the folder the service mounts
# each at
_SECRETS_FOLDERS = {
    "rkmon-secrets": container_layout.SECRETS_DIR,
    "rkmon-secrets-write": container_layout.WRITE_SECRETS_DIR,
}


def _config(keys) -> str:
    """docker compose config's answer for NAME's compose project, as JSON: the service's mounts of the volumes that
    keep the logins and secrets (_SECRETS_FOLDERS), and the volumes KEYS declared, each under its name in that
    project."""
    mounts = [
        {"type": "volume", "source": k, "target": t}
        for k, t in _SECRETS_FOLDERS.items()
    ]
    config = {
        "name": NAME,
        "services": {"rkmon": {"volumes": mounts}},
        "volumes": {key: {"name": f"{NAME}_{key}"} for key in keys},
    }
    return json.dumps(config)


def secrets_volumes(here: bool, unheld=()) -> list:
    """The stand-in docker's answers about the volumes that keep the logins and secrets: see the module's
    docstring."""
    names = [f"{NAME}_{key}" for key in _SECRETS_FOLDERS]
    return [
        [r"^compose config --format json$", _config(_SECRETS_FOLDERS), 0],
        [
            r"^volume ls (--filter name=\S+ )+--format \{\{\.Name\}\}$",
            "\n".join(names) if here else "",
            0,
        ],
        [
            r"^run --rm --network none .*--mount type=volume,",
            json.dumps(list(unheld)),
            0,
        ],
    ]


def volumes(labeled=(), unlabeled=(), env: bool = True) -> list:
    """The stand-in docker's answers about NAME's volumes on this host: see the module's docstring."""
    by_label = re.escape(f"label={identity.PROJECT}={NAME}")
    return [
        [
            f"^volume ls --filter {by_label} " + r"--format \{\{\.Name\}\}$",
            "\n".join(labeled),
            0,
        ],
        [
            "^compose config "
            + ("" if env else "--no-interpolate ")
            + "--format json$",
            _config(VOLUME_KEYS),
            0,
        ],
        [
            r"^volume ls (--filter name=\S+ )+--format \{\{\.Name\}\}$",
            "\n".join([*labeled, *unlabeled]),
            0,
        ],
    ]


IMAGE_CODE = f"{container_layout.IMAGE_CODE} {code_volume.IMAGE_BOUND[0]} "


def image_code(package_dir: pathlib.Path, changed=()) -> list:
    """The image's own copies of the files it keeps, PACKAGE_DIR's but CHANGED: see the module's docstring."""
    printed = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            CODE_PROBE,
            str(package_dir),
            *code_volume.IMAGE_BOUND,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    held = json.loads(printed)
    for rel in changed:
        held[rel] = hashlib.sha256(b"another copy\n").hexdigest()
    return [[IMAGE_CODE, json.dumps(held), 0]]


def delivered(package_dir: pathlib.Path, path: pathlib.Path) -> list:
    """What the first start delivers for the confirmed settings file PATH: see the module's docstring."""
    made = plan(package_dir, path)
    env, override, layer = deliver.texts(made)
    for rel, written in ((deliver.ENV_FILE, env), (deliver.OVERRIDE_FILE, override)):
        if written is not None:
            (package_dir / rel).write_text(written, encoding="utf-8")
            (package_dir / rel).chmod(0o600)
    deliver.wrote(package_dir, env=env, override=override, layer=layer)
    return runs(made)


def runs(made) -> list:
    """The stand-in docker's answers that the plan MADE's layer is in the settings volume and runs: see the module's
    docstring."""
    applied = {"applied": {"fingerprint": made.fingerprint}, "paused": False}
    layer_file = f"{deliver.LAYER_WRITE_DIR}/{deliver.LAYER_FILE}"
    return [
        [re.escape(layer_file) + "$", deliver.texts(made)[2], 0],
        [r"rk applied --json", json.dumps(applied), 0],
    ]
