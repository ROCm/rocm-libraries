# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The container's fixed layout: its volumes' folders, the dashboards' ports and listen address inside it, and its
users with their ids.

These are the package's values, not settings: nothing outside the container's volumes depends on them, so no
deployment changes them. The image (docker/Dockerfile), the compose file (docker/compose.yaml) and the scripts the
image runs as root (docker/entrypoint.sh, docker/rk_*.py) set them up. Python code reads them here, the scripts in
docker/rk_*.py too (from the image's code); tests/test_container_layout.py checks that the image, the compose file, the
entry point and the settings' defaults agree with them.

- WORK_DIR, SECRETS_DIR, WRITE_SECRETS_DIR: the folders of the three volumes of the users' files: the work volume (the
  state, the logs, the clone); the secrets the monitor reads (the read logins, the keys); the writer's secrets, which
  only the writer reads (the GitHub write login, the Teams webhook, the web host's key and the atlassian CLI's login:
  rkmon_lib/secrets_registry.py, each entry's keeper). VOLUMES: the three, in that order. WRITE_LOGIN_DIR: the write
  login's folder, in the write volume. MONITOR_HOME: the monitor's home (RKMON_HOME: its configuration and its state),
  in the work volume. LOGS_FOLDER: the name of the container's logs' folder in the work volume (WORK_DIR/LOGS_FOLDER:
  the entry point's LOGS, and the write broker's log, docker/rk_write.py and rkmon_lib/writer_run.py LOG).
  SECRETS_VOLUME: the volume at SECRETS_DIR, as the compose file names it, with no name of its own, so docker's name
  for it is <container.name>_SECRETS_VOLUME: rkmon confirm finds it so (docker/rkmon, own_secrets_volume).
- CODE_VOLUME: the code volume, at CODE_DIR: the releases of the package's code that root installs over the image's
  own (rkmon_lib/code_volume.py), root's and read-only to every other user. No user's file is in it, so it is none of
  VOLUMES. IMAGE_CODE: the image's own code, which docker/Dockerfile copies there, root's and read-only;
  IMAGE_ID_FILE: the digest of it, which the image's build writes.
- SETTINGS_VOLUME: the settings volume, the delivered deploy layer's: LAYER_FILE, of the kind LAYER_KIND
  at LAYER_SCHEMA_VERSION, which rkmon_lib/deploy_plan.py makes and rkmon_lib/runtime_config.py reads. The container
  mounts it read-only at SETTINGS_DIR, so nothing in the container writes it; the monitor and the writer read the
  layer by the monitor's group, and no PR runner does. Only rkmon_lib/deliver.py writes it, as root, through a one-off
  container, a service of the compose file's own, which mounts it alone, at LAYER_WRITE_DIR, with no network and
  only LAYER_WRITE_CAPABILITIES: CHOWN, to give the layer the monitor's group. Beside the layer, SKILLS_FOLDER is the
  folder of a team's own review skills, laid out as the package's agentic/ folder: the session's render lays its
  files over the package's (rkmon_lib/agentic.py).
- HANDED_LAYER_DIR: the folder of the layer your confirmed settings make, as LAYER_FILE, in a one-off container of the
  logins' walk (deploy step 5) or of `rkmon web-host install` alone, which the host's wrapper (docker/rkmon
  walk_inside) runs so. The wrapper hands that container the layer by the variable HANDED_LAYER_VARIABLE, its name alone
  on docker's command line and its value in docker compose's environment: HANDED_LAYER_MAX bytes at most, what one
  variable carries (Linux's limit on a string of the environment, NAME=value and its end, on the host's docker command
  and in the container), and the wrapper refuses a longer layer before any one-off container. The layer holds no secret
  (each secret is a file of its own in the secrets volumes): docker inspect shows the variable, and the container's
  first process, tini, keeps it for the one-off's run. The entry point takes the variable out of its environment first,
  so the command does not inherit it, and its root phase writes the layer there (docker/rk_root.py hand-layer: the
  folder root's, group rkmon, mode 750, the file mode 640, in the container's own file system, in no volume), where rk
  reads it in place of the settings volume's layer (rkmon_lib/runtime_config.py handed_layer). The compose file passes
  no variable of that name, so neither the running container nor a start's one-off container has the folder.
- LEAVE_VOLUMES_VARIABLE: the variable, set and not empty, by which the host's wrapper (docker/rkmon) tells the entry
  point that a one-off container's command only reads (a question, adopt's plan): its root phase then gives the users
  their ids alone and the container's own temporary folders their permissions, as for the logins' walk
  (docker/rk_ids.py --users, docker/rk_layout.py --system-tmp), so that it walks no file of the volumes and writes no
  ids record. The entry point takes the variable out of its environment first, and the start of `run` has its whole
  root phase whatever the variable says.
- WRAPPER_VARIABLE: the variable by which the host's wrapper (docker/rkmon) gives each command in the container the name
  it was run by (RKMON_WRAPPER), which the fixes show (rkmon_lib/doctor.py wrapper_name). `shown_wrapper(environ)`: that
  name as the writer's rk gets it, which the writer's broker (docker/rk_write.py) and the writer
  (rkmon_lib/writer_run.py) each check before they pass it on, in the fixed environment they give it: ENVIRON's value
  when it is a path that ends in WRAPPER_NAME, the wrapper's own name, made of POSIX's portable file name characters
  (letters, digits, `.`, `_`, `-`) and `/` alone, not beginning with `-`, at most WRAPPER_MAX characters long; else,
  unset or any other value, WRAPPER_NAME. The writer only shows it, in a fix's command: it is never run, and never used
  as a path.
- IMAGE_VENV: the image's checks venv (docker/Dockerfile), which the review checks and the suite run with in the
  container while the configuration names no other.
- ROCM_RECORD: the image's ROCm record, which its build writes (docker/install_rocm.sh): the ROCm release it installed
  and the folder of that release's LLVM tools, `llvm_bin`. `rocm_record(path="")`: the record at PATH, ROCM_RECORD by
  default; {} where there is none or it cannot be read. `rocm_llvm_bin(path="")`: the LLVM tools' folder the record
  names, "" without one: in the container, the value of paths.rocm_llvm_bin wherever the record names one, over a
  folder config.json gives (rkmon_lib/derived.py), and the folder rk doctor's "ROCm release (image)" row shows.
- GH_COUNTER_DIR: the image's folder of the GitHub request cap's counter, docker/rk_gh.py linked there as gh
  (docker/Dockerfile): the first folder of every PATH in the container, the image's, which the monitor, the session,
  its agents and the jobs inherit, and the writer's (rkmon_lib/writer_run.py BASE_ENV), so that each gh call there
  counts (rkmon_lib/gh_cap.py). GH_MISSING_FIX: what brings back the image's own gh, which every gh call there runs
  behind the counter, when it is missing or too old: the counter names it (docker/rk_gh.py).
- ADMIN_PORT, TEAM_PORT: the ports the admin dashboard and the team dashboard listen on inside the container. The
  compose file publishes each on a host port of the deployment's choosing.
- LISTEN_ADDRESS: the address both listen on inside the container, every address the container has, so the published
  ports reach them; which host addresses reach them is the published ports' choice.
- EVERY_ADDRESS: the IPv4 address that stands for every address a machine has (a listener bound to it listens on
  all of them). LISTEN_ADDRESS is it; the package names it from here wherever it means every address of a host too:
  a dashboard's bind, a probe of one, the bind `rkmon web-host install` gives the team web host's request service.
- LOOPBACK: the IPv4 address of a machine's loopback, which that machine alone reaches: the host address the
  deployment publishes the admin dashboard on, its only one (rkmon_lib/deploy_plan.py, deployment.env_values,
  the move from layout 1), and a host install's default bind of it (webserve.HOST). The package names it from
  here wherever it means that address.
- LOCAL_HOSTS: the names a URL gives this machine by: LOOPBACK, localhost, and IPv6's loopback, bare (as urlsplit
  reads a URL's host) and in brackets (as a URL writes it). config.is_local_url, the reviewer model's client
  (worker/reviewer_client.py, through worker/rkenv.py) and tools/worker_grace.py read it.
- `User(name, uid, group, gid, groups)`: a user of the image: its name and id, its own group's name and id, and the
  other groups it is in. MONITOR runs the monitor (the session, its agents, the jobs); the start gives it, and its
  group, the deployment's ids (docker/rk_ids.py). WRITER runs the writer (rkmon_lib/writer.py), the only user that reads
  the writer's secrets; its group is the monitor's (rkmon), so the monitor and the writer share what they write, and its
  id is the deployment's too. USERS: the monitor, the writer and the PR runners.
- PR_RUNNERS: the users that run pull requests' code, one per slot (SLOTS of them): rkpr1 to rkprN, with these ids
  in every deployment. Their shared group, PR_GROUP (PR_GID), reads the trees PR code runs from (the clone, the
  worktrees, the virtual environments) and writes nothing of the monitor's; no runner is in the monitor's group, so
  none reads the secrets volumes or the monitor's state. The monitor is in PR_GROUP: it gives those trees that group.
  Each runner's own group, of its name, is its primary one, which no other user is in: what a runner makes outside
  its slot (in SHM_DIR and MQUEUE_DIR) no other runner reads by its group. Those groups' ids are a range of their own,
  below the runners' user ids (PR_GID is the first runner's user id). PR_RUNNER: the first runner.
- RAMDISK: the RAM disk's folder: the deployment's tmpfs is mounted there (the default of config `ramdisk.path`:
  rkmon_lib/ramdisk.py); without one, it is a folder of the container's own. Each start gives it to the monitor, group
  rkmon, mode 2771 (docker/rk_layout.py): a PR runner passes through it to its slot and to the trees it reads, and
  writes nothing at its top. The slots are in it, so the jobs' builds and temporary files are in memory when the
  deployment has a RAM disk; what the slots hold goes with the container, and with each of its starts on a RAM disk.
- SLOTS_DIR: the slots' folder, in RAMDISK, root's; SLOTS_DIR/<runner> is a slot's own, the runner's, group rkmon (the
  monitor reads it), mode SLOT_MODE, 2751 (another runner reaches a kept slot's build in it, and reads nothing else):
  the start makes them so (docker/rk_layout.py), and the launcher refuses a slot that is not (docker/rk_prcode.py).
  SLOT_FOLDERS: the folders the launcher makes fresh in it for each job: the job's results (SLOT_OUT), its builds
  (SLOT_BUILD), its temporary files (SLOT_TMP) and its home (SLOT_HOME). SLOT_LOCKS: the monitor's folder of the slots'
  records (which job holds which slot: rkmon_lib/slots.py). SLOT_INPUTS: the monitor's folder of what it hands each
  slot's job to read, SLOT_INPUTS/<runner>, group PR_GROUP; INPUTS is its name, which a host's slot folder holds too
  (rkmon_lib/prcode.py).
- SYSTEM_TMP: the system's temporary folders, /tmp and /var/tmp: root's, group rkmon, sticky, mode 1770, which each
  start gives them (docker/rk_layout.py), so the monitor, the writer and root use them and no PR runner writes or reads
  them; a job's temporary files go to its slot's SLOT_TMP (its TMPDIR). SHM_DIR, /dev/shm, and MQUEUE_DIR,
  /dev/mqueue, stay every user's, as System V IPC does: POSIX semaphores, shared memory and message queues live there
  (Python's multiprocessing locks need /dev/shm), and the container cannot give a runner its own; the launcher
  (docker/rk_prcode.py) removes a runner's own entries in both, and its System V objects, each time it begins, keeps
  or ends the runner's slot. A PR job can still make an object of a fixed name there for a develop job that runs at the
  same time to open; the runners' own groups narrow that to an object the job opens to every user.
- CAPABILITIES: the Linux capabilities the container keeps; the compose file drops every other one. Only a process
  that runs as root has them: the start's root phase (docker/entrypoint.sh), where CHOWN, DAC_OVERRIDE, FOWNER and
  FSETID give the users their ids and the folders their owners and modes (docker/rk_ids.py, docker/rk_layout.py,
  docker/rk_root.py), and SETUID and SETGID drop to a user (setpriv); sudo, which needs SETUID and SETGID to run a
  helper as another user, and AUDIT_WRITE to write its audit record of each run (without it, sudo says on each call
  that it could not); and tini, the first process, which needs KILL to pass a stop signal on to the monitor's
  processes. The image test runs every stage with these and no other (tests/container/image.py), and a stage of it
  fails without any one of them (its --without-cap).
  PIDS_LIMIT: the most processes the container runs at once. RUN_LIMITS: both as `docker run` flags, for a container
  the compose file does not start.

Standard library only: the package's own worker/ imports it too (worker/pr_user.py); an installed copy of the worker
tools reads PR_GROUP and SLOT_HOME, which its install wrote beside it (rkmon_lib/deploy.py), and runs no file.
"""

from __future__ import annotations

import json
import re
from typing import NamedTuple

WORK_DIR = "/work"
SECRETS_DIR = "/secrets"
WRITE_SECRETS_DIR = "/secrets-write"
VOLUMES = (WORK_DIR, SECRETS_DIR, WRITE_SECRETS_DIR)
SECRETS_VOLUME = "rkmon-secrets"
WRITE_LOGIN_DIR = f"{WRITE_SECRETS_DIR}/gh"
MONITOR_HOME = f"{WORK_DIR}/monitor"
LOGS_FOLDER = "logs"
CODE_VOLUME = "rkmon-code"
CODE_DIR = "/code"
IMAGE_CODE = "/opt/rocke_monitor"
IMAGE_ID_FILE = "/etc/rkmon/image_code_id"

SETTINGS_VOLUME = "rkmon-settings"
SETTINGS_DIR = "/etc/rkmon/settings"
LAYER_WRITE_DIR = "/mnt/rkmon-settings"
LAYER_FILE = "deploy.json"
SKILLS_FOLDER = "skills"
LAYER_KIND = "confirmed_deploy_settings"
LAYER_SCHEMA_VERSION = 1
LAYER_WRITE_CAPABILITIES = ("CHOWN",)
HANDED_LAYER_DIR = "/run/rkmon-confirmed"
HANDED_LAYER_VARIABLE = "RKMON_CONFIRMED_LAYER"
# Linux's limit on one string of a process's environment, NAME=value and its end (MAX_ARG_STRLEN), is 128 KiB.
HANDED_LAYER_MAX = 128 * 1024 - len(HANDED_LAYER_VARIABLE) - 2
LEAVE_VOLUMES_VARIABLE = "RKMON_LEAVE_VOLUMES"
WRAPPER_VARIABLE = "RKMON_WRAPPER"
WRAPPER_NAME = "rkmon"  # docker/rkmon's: rkmon_lib/program.py NAME
WRAPPER_MAX = 255
WRAPPER_SHOWN = re.compile(rf"(?!-)(?:[A-Za-z0-9._/-]*/)?{re.escape(WRAPPER_NAME)}")

IMAGE_VENV = "/opt/rkmon/venv"
ROCM_RECORD = "/etc/rkmon/rocm.json"
GH_COUNTER_DIR = "/opt/rkmon/gh-counter"
GH_MISSING_FIX = "build the image again (QUICK_BUILD_GUIDE.md) and roll it out by its new tag (QUICK_DEPLOY_GUIDE.md)"

ADMIN_PORT = 18613
TEAM_PORT = 18615
EVERY_ADDRESS = "0.0.0.0"
LISTEN_ADDRESS = EVERY_ADDRESS
LOOPBACK = "127.0.0.1"
LOCAL_HOSTS = (LOOPBACK, "localhost", "::1", "[::1]")


def shown_wrapper(environ) -> str:
    """The name the host's wrapper was run by, as the writer's rk gets it: see the module's docstring."""
    value = environ.get(WRAPPER_VARIABLE) or ""
    if len(value) <= WRAPPER_MAX and WRAPPER_SHOWN.fullmatch(value):
        return value
    return WRAPPER_NAME


def rocm_record(path="") -> dict:
    """The image's ROCm record: see the module's docstring."""
    try:
        with open(path or ROCM_RECORD, encoding="utf-8") as f:
            rec = json.load(f)
    except (OSError, ValueError):
        return {}
    return rec if isinstance(rec, dict) else {}


def rocm_llvm_bin(path="") -> str:
    """The LLVM tools' folder the image's ROCm record names: see the module's docstring."""
    return str(rocm_record(path).get("llvm_bin") or "").strip()


class User(NamedTuple):
    """A user of the image, with the image's own ids."""

    name: str
    uid: int
    group: str
    gid: int
    groups: tuple = ()


SLOTS = 16
PR_GROUP = "rkpr"
PR_GID = 1998
# A runner's user id counts down from PR_GID; its own group's id, from just below the last runner's user id.
PR_RUNNERS = tuple(
    User(
        f"{PR_GROUP}{i}",
        PR_GID + 1 - i,
        f"{PR_GROUP}{i}",
        PR_GID + 1 - SLOTS - i,
        (PR_GROUP,),
    )
    for i in range(1, SLOTS + 1)
)
PR_RUNNER = PR_RUNNERS[0]
MONITOR = User("rkmon", 1000, "rkmon", 1000, (PR_GROUP,))
WRITER = User("rkwrite", 1999, MONITOR.group, MONITOR.gid)
USERS = (MONITOR, WRITER, *PR_RUNNERS)

RAMDISK = "/ramdisk"
SLOTS_DIR = f"{RAMDISK}/slots"
SLOT_MODE = 0o2751
SLOT_FOLDERS = ("out", "build", "tmp", "home")
SLOT_OUT, SLOT_BUILD, SLOT_TMP, SLOT_HOME = SLOT_FOLDERS
SLOT_LOCKS = f"{SLOTS_DIR}/locks"
INPUTS = "inputs"
SLOT_INPUTS = f"{SLOTS_DIR}/{INPUTS}"

SYSTEM_TMP = ("/tmp", "/var/tmp")
SHM_DIR = "/dev/shm"
MQUEUE_DIR = "/dev/mqueue"

CAPABILITIES = (
    "CHOWN",
    "DAC_OVERRIDE",
    "FOWNER",
    "FSETID",
    "KILL",
    "SETUID",
    "SETGID",
    "AUDIT_WRITE",
)
PIDS_LIMIT = 4096
RUN_LIMITS = (
    "--cap-drop",
    "ALL",
    *(flag for c in CAPABILITIES for flag in ("--cap-add", c)),
    "--pids-limit",
    str(PIDS_LIMIT),
)
