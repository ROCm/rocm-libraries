# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The container's two jobs on the host, each a numbered path with its guards: building an image (from a build
settings file, on any machine) and deploying a container from one (from a deploy settings file, on any machine, which
never builds). docker/rkmon runs them, so this module uses the standard library only, on the host's Python
(HOST_PYTHON). Nothing here runs docker: the wrapper asks docker and hands the answers in (Facts), so every guard is a
plain function the tests call.

Each path's steps are rkmon_lib/steps.py's (QUICK_BUILD_GUIDE.md, QUICK_DEPLOY_GUIDE.md); their guards are here.
The build path's: build_checks, plan_checks, ca_checks and the TLS checks. The deploy path's: preflight
(step 2's check of the host and the deploy settings file), the file's own rows (read) and the settings' checks
(settings_checks), the rows rkmon confirm adds to the file's own (confirm_rows), and the image's (no_image_row,
image_tls_checks: step 4; whether an image suits this package is confirm.image_rows').

The deploy settings file (`kind: deploy_settings`: schema/deploy_settings.schema.json, read through
rkmon_lib/deploy_tree.py) is the one source of the container (container.*: its name; image.*: its image), the host's
share (host.*: CPUs, memory, the RAM disk, the monitor's user and group), the dashboards' ports, and the monitor's
configuration. docker/.env and compose.override.yaml are made from the confirmed file (rkmon_lib/deploy_plan.py) and
have no internal value. Every check here reads the configuration the file makes (config.json's keys) and names each
key as the file does (deploy_tree.name); the features' checks read the configuration the container runs with it
(`configuration`).
"""

from __future__ import annotations

import os
import pathlib
import re
import shutil
import socket
import ssl
import stat
from dataclasses import dataclass, field
from typing import NamedTuple

from . import (
    checks,
    config,
    container_layout,
    deploy_tree,
    identity,
    image,
    paths,
    program,
    registry,
    states,
    steps,
    wrapping,
)
from .checks import DELETE, Command, Described, Edit, Status, Steps

# The oldest Python docker/rkmon runs on, the host's (the YAML reader, rkmon_lib/answers.py, needs it), and
# tools/make_package.py's, which builds the package there: step 2's check refuses an older one, and
# QUICK_DEPLOY_GUIDE.md states it from here, in what the host needs; QUICK_BUILD_GUIDE.md names that guide for it. The
# monitor's own floor, in the container and on a host install, is doctor.PY_MIN.
HOST_PYTHON = (3, 9)

# --- checks -----------------------------------------------------------------------------------------------------------


# The values of a fix's edit that only you know, in words (checks.Described).
ASKED = Described("what the line above asks for")
OWN = Described("your own value")
CA_FILES = Described(
    "your network's root certificate files, as PEM (Base-64, .crt), outside the package folder"
)


def run_on(command: str, file: str) -> tuple:
    """The parts of a fix's steps that run COMMAND (as steps.py writes one: "check --settings FILE") on FILE: the
    command with FILE's path; for a settings file named in words (deploy_tree.FILE, image.BUILD_FILE), with FILE and
    the words that say what it is."""
    if file in (deploy_tree.FILE, image.BUILD_FILE):
        return (steps.command_line(command), ", FILE being that file")
    return (steps.command_line(command, FILE=file),)


def check_again(file: str) -> tuple:
    """The parts of a fix's steps that run the check of FILE again (run_on)."""
    return run_on("check --settings FILE", file)


# --- the settings file ------------------------------------------------------------------------------------------------

# The container's memory for the work itself, beside the RAM disk (check 8), and the RAM disk's first
# estimates of a review and a build (ramdisk.SEED_GB): the package's defaults (config.defaults.json, through the
# registry). The most reviews and builds at once are config's (config.pipes_max, config.checks_max).
WORK_GB = registry.package_section("resources")["work_memory_gib"]
PIPE_GB = registry.package_section("ramdisk")["review_estimate_gib"]
HEAVY_GB = registry.package_section("ramdisk")["build_estimate_gib"]
RAM_DISK_ROW = "RAM disk"  # the name of the RAM disk's rows on the deploy settings: sizing_checks', coherence's
PUBLIC_CA = (  # the issuers of public sites' certificates: any other one names a TLS-inspecting proxy
    "DigiCert",
    "Sectigo",
    "GlobalSign",
    "Let's Encrypt",
    "ISRG",
    "Google Trust Services",
    "Amazon",
    "Microsoft",
    "Entrust",
    "GoDaddy",
    "Starfield",
    "Baltimore",
    "USERTrust",
    "COMODO",
    "IdenTrust",
    "Certainly",
    "SSL.com",
    "HARICA",
    "Buypass",
)


def get(cfg: dict, dotted: str, default=None):
    node = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return default if node is None else node


def read(path) -> tuple:
    """(the configuration a deploy settings file makes, the file's FAIL and WARN rows, each with its fix, and check 7's
    PASS row: rkmon_lib/checks.py rows): deploy_tree.read. A file the reader refuses raises settings_file.Refused, with
    its rows; one that cannot be opened, its OSError."""
    from . import deploy_tree

    return deploy_tree.read(path)


def N(internal: str) -> str:
    """A configuration key as the deploy settings file names it."""
    from . import deploy_tree

    return deploy_tree.name(internal)


def settings_kind(path) -> str:
    """The kind of settings file PATH is by its header, its first line after the comments: "deploy" (`kind:
    deploy_settings`, or the earliest layout's `rkmon_settings: 1`), "build" (`kind: build_settings`, or
    `rkmon_build_settings: 1`), or "" for any other file, or one that cannot be read as text. The kinds and the
    earliest layout's first keys are file_types'."""
    from . import (
        file_types,
    )  # which imports this module (through coherence): read at the call

    try:
        text = pathlib.Path(path).read_text(encoding="utf-8")
    except (OSError, ValueError):
        return ""
    for line in text.lstrip("\ufeff").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s == "---":
            continue
        for kind, old, header in (
            (
                "deploy",
                file_types.DEPLOY_HEADER_BEFORE_LAYOUT_1,
                file_types.DEPLOY.kind,
            ),
            ("build", file_types.BUILD_HEADER_BEFORE_LAYOUT_1, file_types.BUILD.kind),
        ):
            if re.match(
                rf"""\{{?\s*"?({re.escape(old)}"?\s*:|kind"?\s*:\s*['"]?{re.escape(header)}\b)""",
                s,
            ):
                return kind
        return ""
    return ""


# What the scan for stray private files never looks into, wherever it is: git's own files, and the folders a
# developer's copy leaves wherever it runs (paths.COPY_OWN_LEFTOVERS: the caches).
STRAY_SKIP = (
    ".git",
    *(p.rstrip("/") for p in paths.COPY_OWN_LEFTOVERS if p.endswith("/")),
)
SETTINGS_MAX_BYTES = 2_000_000  # a settings file is smaller: the scan reads no larger file for its header


def stray_private_files(package_dir, in_use: bool = True) -> list:
    """The private files in the package folder PACKAGE_DIR that are not its own, as paths relative to it, sorted: each
    file and folder whose name the ignore lists hold (secrets_registry.ignore_patterns, matched as git matches them: a
    pattern with a trailing / on a folder's name only; a folder found is named with its trailing / and not looked
    into), and each settings file of either kind by its header (settings.header_line), whatever its name. IN_USE:
    PACKAGE_DIR is a package folder in use, whose own private files lie in their places (private_files.in_own_places)
    and whose top holds the folders a developer's copy makes (paths.COPY_OWN_FOLDERS, which git and the image's build
    context leave out): neither is looked into. False: PACKAGE_DIR is a package's code that comes in (an update's new
    tree), which holds none of the package's own private files, so any it has is a stray too, and is looked into
    whole. Never looked into: STRAY_SKIP. A link is judged by its name alone, never followed.
    """
    import fnmatch

    from . import private_files, secrets_registry

    root = pathlib.Path(package_dir)
    patterns = secrets_registry.ignore_patterns()
    file_patterns = [x for x in patterns if not x.endswith("/")]
    folder_patterns = file_patterns + [x[:-1] for x in patterns if x.endswith("/")]

    def named(name: str, among: list) -> bool:
        return any(fnmatch.fnmatchcase(name, x) for x in among)

    def own(rel: str) -> bool:
        return in_use and private_files.in_own_places(rel)

    out = []
    for dirpath, dirnames, files in os.walk(root):
        here = pathlib.PurePosixPath(pathlib.Path(dirpath).relative_to(root).as_posix())
        look = []
        for name in dirnames:
            rel = (here / name).as_posix()
            if (
                name in STRAY_SKIP
                or own(rel)
                or (in_use and here.name == "" and name in paths.COPY_OWN_FOLDERS)
            ):
                continue
            if named(name, folder_patterns):
                out.append(f"{rel}/")
            else:
                look.append(name)
        dirnames[:] = look
        for name in files:
            rel = (here / name).as_posix()
            if not own(rel) and (
                named(name, file_patterns)
                or _settings_file(pathlib.Path(dirpath) / name)
            ):
                out.append(rel)
    return sorted(out)


def _settings_file(p: pathlib.Path) -> bool:
    """P is a settings file of either kind by its header (settings.header_line): a regular file, not a link, smaller
    than SETTINGS_MAX_BYTES, read as UTF-8 text."""
    from . import settings

    try:
        if p.is_symlink() or not p.is_file() or p.stat().st_size >= SETTINGS_MAX_BYTES:
            return False
        text = p.read_text(encoding="utf-8")
    except (OSError, ValueError):
        return False
    return settings.header_line(text) > 0


def stray_rows(package_dir, again, source: str = "") -> list:
    """The row of a package folder that holds private files that are not its own (stray_private_files): a FAIL that
    names each, its fix to move each out, then AGAIN (a fix's command: the command run again); none when it holds none.
    PACKAGE_DIR is the package folder in use, which a build, a commit and an update would each carry such a file off;
    with SOURCE, it is a new package's code copied from SOURCE (a package update), which holds none of the package's own
    private files either, and the fix takes each out of SOURCE."""
    stray = stray_private_files(package_dir, in_use=not source)
    if not stray:
        return []
    if source:
        return [
            checks.Check(
                "the new package",
                Status.FAIL,
                "private files inside it, which no package carries: "
                + ", ".join(stray),
                Steps((f"take each out of {source}, then run", again)),
            )
        ]
    return [
        checks.Check(
            "the package folder",
            Status.FAIL,
            "private files inside it: " + ", ".join(stray),
            Steps(
                (
                    "move each out of the package folder (a build, a commit or an update would carry it), then run",
                    again,
                )
            ),
        )
    ]


# The build settings key of the root certificates the image trusts.
CA_KEY = "operating_system.ca_certificates"


def _not_yours(f: pathlib.Path, st, file: str):
    """The FAIL row of the rule that the root certificate file F (ST, its stat) is yours, or None: you or root own it,
    and no one else may change it, since the image trusts whatever it holds. No write permission for others, nor for
    its group unless the group is your own (your primary group, which holds only you where each user has a group of
    their own, as Ubuntu makes them). FILE is the build settings file that names F."""
    if st.st_uid not in (os.getuid(), 0):
        return checks.Check(
            "CA certificate",
            Status.FAIL,
            f"{f} belongs to another user, who could change the certificates the image trusts",
            Steps(("copy it to a file of yours, then", Edit(CA_KEY, file, CA_FILES))),
        )
    mode = stat.S_IMODE(st.st_mode)
    if mode & 0o002 or (mode & 0o020 and st.st_gid != os.getgid()):
        sudo = ("sudo",) if st.st_uid != os.getuid() else ()
        return checks.Check(
            "CA certificate",
            Status.FAIL,
            f"{f} has permissions {mode:03o}: others on this machine could change the certificates the image trusts",
            Command((*sudo, "chmod", "go-w", str(f))),
        )
    return None


def ca_checks(files, file: str = image.BUILD_FILE) -> list:
    """The root certificate files the build settings name (CA_KEY; FILES, a build plan's ca_files): each one is there,
    is yours (`_not_yours`) and is a PEM certificate; the fixes edit FILE, the build settings file that names them.
    """
    if not files:
        return [
            checks.Check(
                "CA certificate",
                Status.PASS,
                f"none named ({CA_KEY}): right for a network that does not inspect TLS",
            )
        ]
    out = []
    for f in files:
        try:
            st = f.stat()
            text = f.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            out.append(
                checks.Check(
                    "CA certificate",
                    Status.FAIL,
                    f"{f}: {exc.strerror}",
                    Steps(
                        (
                            f"put your network's root certificate at {f}, or",
                            Edit(CA_KEY, file, CA_FILES),
                        )
                    ),
                )
            )
            continue
        wrong = _not_yours(f, st, file)
        if wrong:
            out.append(wrong)
        elif "-----BEGIN CERTIFICATE-----" not in text:
            out.append(
                checks.Check(
                    "CA certificate",
                    Status.FAIL,
                    f"{f} has no PEM certificate",
                    Edit(CA_KEY, file, CA_FILES),
                )
            )
        else:
            out.append(checks.Check("CA certificate", Status.PASS, f"{f}"))
    return out


def stage_ca(files, folder) -> int:
    """Copy each root certificate file of FILES (a build plan's ca_files) into FOLDER, a private folder the build takes
    as its build context build_plan.CA_CONTEXT: as ca-1.crt, ca-2.crt and on, in the settings' order (mode 644; the
    base image trusts *.crt there), so no name of yours reaches the build. The number copied.
    """
    for n, f in enumerate(files, 1):
        d = pathlib.Path(folder) / f"ca-{n}.crt"
        shutil.copyfile(f, d)
        d.chmod(0o644)
    return len(files)


def resources(cfg: dict) -> dict:
    """The container's sizing: {cpus, memory_gb, ramdisk, ramdisk_gb, ramdisk_path, pipes, checks}."""

    def num(key):
        return registry.number(cfg, key, float, registry.FALSY)

    rd = get(cfg, "ramdisk.enabled", False) is True
    return {
        "cpus": num("resources.cpus"),
        "memory_gb": num("resources.memory_gb"),
        "ramdisk": rd,
        "ramdisk_gb": num("ramdisk.size_gb") if rd else 0.0,
        "ramdisk_path": str(registry.value(cfg, "ramdisk.path", registry.FALSY)),
        "pipes": int(num("review.pipes")),
        "checks": int(num("checks.limit")),
    }


def admin_public_port(cfg: dict) -> int:
    """The host port docker publishes the admin dashboard on: admin_dashboard.public_port (blank or 0: its default)."""
    return registry.number(cfg, "admin_dashboard.public_port", int, registry.FALSY)


def _g(x: float) -> str:
    return f"{x:g}"


def sizing_checks(cfg: dict, file: str = deploy_tree.FILE) -> list:
    """The RAM disk's room for the most reviews and builds that may run at once (config.pipes_max, config.checks_max;
    the reviews while review automation is on): a WARN while it holds the starting ones but not all of them, else a
    PASS. No row without a RAM disk, or while it is smaller than the starting ones need: the deploy file's own rules on
    the container's share of this host (coherence.resources, checks 8 and 9), which rkmon check and rkmon confirm
    run beside these rows, warn of that, and judge the memory and the CPUs. The numbers are coherence.least's. The
    fixes edit FILE.
    """
    from . import coherence, features  # coherence imports this module

    r = resources(cfg)
    if not r["ramdisk"]:
        return []
    on = features.enabled(cfg, "review_automation")
    start = coherence.least(reviews=r["pipes"] if on else 0, builds=r["checks"])
    if r["ramdisk_gb"] < start.need:
        return []
    most_reviews, most_builds = (config.pipes_max() if on else 0), config.checks_max()
    full = coherence.least(reviews=most_reviews, builds=most_builds)
    if r["ramdisk_gb"] >= full.need:
        found = f"{_g(r['ramdisk_gb'])} GiB at {r['ramdisk_path']}"
        return [checks.Check(RAM_DISK_ROW, Status.PASS, found)]
    ram, mem = (deploy_tree.name(k) for k in ("ramdisk.size_gb", "resources.memory_gb"))
    room = coherence.least(
        full.about
    ).memory  # the memory a RAM disk for all of them takes, with the work's
    more = (
        (", and", Edit(mem, file, Described(f"{_g(room)} or more")))
        if r["memory_gb"] < room
        else ()
    )
    them = (
        f"all {most_reviews} parallel reviews and {most_builds} parallel builds"
        if on
        else f"all {most_builds} parallel builds"
    )
    return [
        checks.Check(
            RAM_DISK_ROW,
            Status.WARN,
            f"{N('ramdisk.size_gb')} {_g(r['ramdisk_gb'])} fits the starting reviews and builds, and no more run at "
            f"once than fit on it ({them} need {_g(full.need)} GiB)",
            Steps(
                (
                    "fine while fewer run at once; for all of them,",
                    Edit(ram, file, full.about),
                    *more,
                )
            ),
        )
    ]


def _unresolved_fix(key: str, file: str, source):
    """The fix in FILE of a value the build cannot have (build_plan.Unresolved), by its build settings KEY: the keys
    that pick another value. SOURCE(key): where a key's value comes from (settings_file.source_of).
    """
    if key == "rocm.version":
        return Steps(
            (
                Edit(key, file, Described("a release that folder lists")),
                ", or",
                Edit("rocm.package_url", file, Described("the installer's full URL")),
            )
        )
    if key == "github_cli.package_url":
        return Steps(
            (
                Edit(key, file, Described("the gh release archive's full URL")),
                ", and",
                Edit(
                    "github_cli.package_sha256",
                    file,
                    source("github_cli.package_sha256"),
                ),
            )
        )
    if key == "github_cli.version":
        return Edit(key, file, Described("the gh release to install"))
    if key == "claude_agent_sdk.version":
        return Steps(
            (
                Edit(key, file, Described("a release the package index has")),
                ", or",
                Edit(
                    "claude_agent_sdk.package_url",
                    file,
                    source("claude_agent_sdk.package_url"),
                ),
                ", and",
                Edit(
                    "claude_agent_sdk.package_sha256",
                    file,
                    source("claude_agent_sdk.package_sha256"),
                ),
            )
        )
    if key in ("github_cli.package_sha256", "rocm.package_sha256"):
        return Edit(key, file, source(key))
    return Edit(key, file, ASKED)


def plan_checks(
    p, file_checks: list, file: str = image.BUILD_FILE, here: bool = False
) -> list:
    """The build's own rows of P, a build plan that build_plan.resolve has read, the fixes in FILE: a FAIL for each
    value the build cannot have (an Unresolved, named by its key; a key whose row in FILE_CHECKS, the file's own,
    fails already, and the root certificates, which ca_checks reads, left out); a WARN for each blank version the build
    takes as the newest release, as two builds of the file can then differ; and, when HERE (its NAME:TAG is on this
    machine already), a WARN that the build would replace that image under the same tag.
    """
    from . import (
        file_types,
        settings_file,
    )  # which import this module (through coherence): read at the call

    schema = file_types.BUILD.load()

    def source(key: str):
        return settings_file.source_of(settings_file.schema_at(schema, key))

    failing = {c.name for c in file_checks if c.status == Status.FAIL}
    out = [
        checks.Check(u.key, Status.FAIL, u.why, _unresolved_fix(u.key, file, source))
        for u in p.unresolved
        if u.key not in failing and u.key != CA_KEY
    ]
    out += [
        checks.Check(
            key,
            Status.WARN,
            f"not pinned; this build would take the newest release, {version}",
            Steps(
                (Edit(key, file, version), ", so a later build makes the same image")
            ),
        )
        for key, version in p.newest
    ]
    if here:
        out.append(
            checks.Check(
                "image.tag",
                Status.WARN,
                f"{p.image.ref} is on this machine already; this build would replace it under the same tag",
                Edit(
                    "image.tag",
                    file,
                    Described(f'a tag of its own, such as "{p.image.tag}-2"'),
                ),
            )
        )
    return out


def image_ref(cfg: dict) -> str:
    """The image the deploy settings name, <image.name>:<image.tag> (the name rocke-monitor by default, the tag
    container.name)."""
    ref = str(get(cfg, "deployment.image", "") or "").strip()
    name, _, tag = ref.rpartition(":") if ":" in ref else ("", "", "")
    return f"{name or image.DEFAULT_NAME}:{tag or project(cfg)}"


def image_name(cfg: dict) -> str:
    return image_ref(cfg).rpartition(":")[0]


def image_tag(cfg: dict) -> str:
    return image_ref(cfg).rpartition(":")[2]


def image_key(part: str) -> str:
    """The deploy settings file's key of the image's PART, "name" or "tag": that key of the section that names the
    image (deploy_tree.name of deployment.image)."""
    return f"{N('deployment.image')}.{part}"


def rk_ids():
    """docker/rk_ids.py, the entry point's mapping of the users' ids: the one home of their rules."""
    import importlib.util

    path = pathlib.Path(__file__).resolve().parent.parent / "docker" / "rk_ids.py"
    spec = importlib.util.spec_from_file_location("rk_ids", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def container_checks(cfg: dict, file: str = deploy_tree.FILE) -> list:
    """The monitor's user and group (host.users, or whoever runs rkmon for the ones left out) are ids the entry point
    maps beside the package's own users, the GitHub writer and the PR runners, whose ids are container_layout's
    (docker/rk_ids.py's rules: it refuses the rest at each start): the monitor's user and group take none of theirs, and
    no row names a setting for them. The form of the image's name and tag is the file's to check: their keys' patterns
    (settings_file.validate). The fixes edit FILE.
    """
    ids_rules = rk_ids()
    out = []

    def key_of(key: str) -> str:
        return deploy_tree.name("container." + key)

    ids = {}
    for key in ("uid", "gid"):
        v = get(cfg, f"container.{key}", None)
        if v in (None, ""):
            continue
        if not str(v).isdigit() or not 1 <= int(v) <= ids_rules.MAX_ID:
            out.append(
                checks.Check(
                    "container ids",
                    Status.FAIL,
                    f"{N('container.' + key)} is {v!r}: not a user id",
                    Steps(
                        (
                            Edit(key_of(key), file, Described("a user id")),
                            ", or",
                            Edit(key_of(key), file, DELETE),
                        )
                    ),
                )
            )
            continue
        ids[key] = int(v)
    want = {"uid": ids.get("uid", os.getuid()), "gid": ids.get("gid", os.getgid())}
    # Left out: whoever runs rkmon (root's 0 is no id the entry point maps).
    for key in ("uid", "gid"):
        if key not in ids and not 1 <= want[key] <= ids_rules.MAX_ID:
            out.append(
                checks.Check(
                    "container ids",
                    Status.FAIL,
                    f"{N('container.' + key)} is left out, and yours ({want[key]}) is not an id the container's "
                    "users take",
                    Steps(
                        (
                            Edit(key_of(key), file, Described("your own id")),
                            "(",
                            Command(("id", "-u" if key == "uid" else "-g")),
                            "prints it, run as yourself)",
                        )
                    ),
                )
            )
    # The clashes the entry point refuses (docker/rk_ids.py, the one home of the ids' rules), worded here: an id out of
    # its range, refused above, is checked against nothing; the package's own ids never clash, so each clash is the
    # monitor's uid's or gid's.
    known = {k: v if 1 <= v <= ids_rules.MAX_ID else None for k, v in want.items()}
    for c in ids_rules.clashes(ids_rules.settings_ids(known["uid"], known["gid"])):
        key = key_of(c.key)
        given = (
            f"{key} is {want[c.key]}"
            if c.key in ids
            else f"your {'user' if c.key == 'uid' else 'group'} id ({want[c.key]})"
        )
        out.append(
            checks.Check(
                "container ids",
                Status.FAIL,
                f"{given}: {c.why}, which the container's entry point refuses at each start",
                Edit(key, file, Described(c.what)),
            )
        )
    return out


def project(cfg: dict) -> str:
    """The container name, which is the compose project, CFG gives (container.name: a deploy settings file's
    configuration holds its default when the file leaves it out); "" for none, which project_checks refuses.
    """
    return str(get(cfg, "deployment.project", "") or "").strip()


def project_checks(cfg: dict, file: str = deploy_tree.FILE) -> list:
    p = project(cfg)
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", p):
        return [
            checks.Check(
                "container name",
                Status.FAIL,
                f"container.name {p!r}: docker compose takes lowercase letters, digits, - and _ only",
                Edit(
                    deploy_tree.name("deployment.project"),
                    file,
                    Described(
                        "a name of lowercase letters, digits, - and _, such as rkmon-test"
                    ),
                ),
            )
        ]
    return []


def placeholder_checks(cfg: dict, file: str = deploy_tree.FILE) -> list:
    """Values a template leaves for you to fill in (<...>: settings_file.PLACEHOLDER), each fix an edit of FILE."""
    from . import (
        settings_file,
    )  # which imports this module (through coherence): read at the call

    out = []

    def placeholder(v) -> str:
        held = v.strip() if isinstance(v, str) else ""
        return held if settings_file.PLACEHOLDER.fullmatch(held) else ""

    def walk(node, prefix=""):
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{prefix}{k}.")
        elif isinstance(node, list):  # a list the template leaves to you: [<label>]
            held = next((placeholder(x) for x in node if placeholder(x)), "")
            if held:
                out.append((prefix.rstrip("."), held))
        elif placeholder(node):
            out.append((prefix.rstrip("."), placeholder(node)))

    walk(cfg)
    return [
        checks.Check(
            "settings file",
            Status.FAIL,
            f"{N(k)} is still the template's {held}",
            Edit(deploy_tree.name(k), file, OWN),
        )
        for k, held in out
    ]


# The keys every deployment needs that have no default (config.REQUIRED, but paths.working_dir, which the container
# sets): (configuration key, what it is, its value in a fix's words).
REQUIRED_SETTINGS = (
    (
        "repo",
        "the GitHub repository rocKE is in, as owner/name",
        "the repository, as owner/name",
    ),
    (
        "rocke_path",
        "rocKE's folder inside the repository",
        "rocKE's folder in it",
    ),
    (
        "operator.github_login",
        "the GitHub login the monitor posts as",
        "that login",
    ),
    (
        "scope.labels",
        "the PR labels that put a PR in scope (at least one)",
        "a list of those labels",
    ),
)


def required_checks(cfg: dict, file: str = deploy_tree.FILE) -> list:
    """Each required key the settings leave blank (a template's <...> is placeholder_checks'), its fix the key's edit
    of FILE."""
    out = []
    for key, what, value in REQUIRED_SETTINGS:
        v = get(cfg, key, "")
        if isinstance(v, (list, tuple)):
            blank = not [x for x in v if str(x).strip()]
        else:
            blank = not str(v).strip()
        if blank:
            out.append(
                checks.Check(
                    "settings file",
                    Status.FAIL,
                    f"{N(key)} is blank: {what}",
                    Edit(deploy_tree.name(key), file, Described(value)),
                )
            )
    return out


def feature_checks(cfg: dict, file: str = deploy_tree.FILE) -> list:
    """The features' combinations (features.coherence): each one that cannot run (FAIL) or does less than it seems to
    (WARN), with its fix in FILE; PASS with each feature's state (features.headline_parts), which the row wraps before
    a "·" (wrapping.marked)."""
    from . import features

    eff = configuration(cfg)
    bad = [
        checks.Check("features", status, what, fix)
        for status, what, fix, _s in features.coherence(eff, file)
    ]
    return bad or [
        checks.Check(
            "features", Status.PASS, wrapping.marked(features.headline_parts(eff))
        )
    ]


def settings_checks(cfg: dict, file: str = deploy_tree.FILE) -> list:
    """Every guard on the deploy settings themselves (no host fact needed), their fixes in FILE."""
    return (
        feature_checks(cfg, file)
        + placeholder_checks(cfg, file)
        + required_checks(cfg, file)
        + container_checks(cfg, file)
        + sizing_checks(cfg, file=file)
        + project_checks(cfg, file)
    )


def confirm_rows(cfg: dict, file: str, package_dir) -> list:
    """The rows rkmon confirm judges the deploy settings file FILE by after its own (read), CFG being the
    configuration it makes: the settings' checks (settings_checks), the container's share of this host, with its
    memory and CPUs (coherence.resources, on host_size: checks 8 and 9), and the team's skills folder the file
    names, against the package folder PACKAGE_DIR (coherence.skills: check 18). docker/rkmon's confirm and the
    container tests' set-up both take this list."""
    from . import (
        coherence,
        file_types,
        settings_file,
    )  # which import this module (through coherence): read at the call

    doc = settings_file.read(file, file_types.DEPLOY)
    return [
        *settings_checks(cfg, file),
        *coherence.resources(doc, *host_size()),
        *coherence.skills(doc, package_dir),
    ]


# The build machine's free disk, as build step 2's disk row judges it: a build with ROCm takes about
# BUILD_DISK_NEED_GIB (the image and its stages' cache), so less is a FAIL, and less than BUILD_DISK_WARN_GIB a WARN.
BUILD_DISK_NEED_GIB = 40
BUILD_DISK_WARN_GIB = 80


def build_checks(p, facts, file_checks: list, file: str = image.BUILD_FILE) -> list:
    """The build path's step 2: docker and buildx, the build settings file's own rows (FILE_CHECKS), then, for P, the
    build plan the file makes once build_plan.resolve has read it (None when the file's rows refused it), its root
    certificates, this network's TLS against them, the build's own rows (plan_checks; FACTS.image_here: its NAME:TAG
    is on this machine already), and the disk (BUILD_DISK_NEED_GIB, BUILD_DISK_WARN_GIB). FILE is the build settings
    file as the fixes name it.
    """
    out = []
    if facts.docker:
        out.append(checks.Check("docker", Status.PASS, f"server {facts.docker}"))
    else:
        out.append(
            checks.Check(
                "docker",
                Status.FAIL,
                facts.docker_error or "docker does not answer",
                Steps(
                    (
                        "install Docker (or Podman: CONTAINER_ENGINE=podman) and start its daemon, then run",
                        *check_again(file),
                    )
                ),
            )
        )
    out.append(
        checks.Check("docker buildx", Status.PASS, facts.buildx)
        if facts.buildx
        else checks.Check(
            "docker buildx",
            Status.FAIL,
            "no BuildKit (`docker buildx version` fails): the build's stages and secrets need it",
            Steps(
                (
                    "install the docker buildx plugin (docker-buildx-plugin), then run",
                    *check_again(file),
                )
            ),
        )
    )
    out += file_checks
    if p is None:
        return out
    out += (
        ca_checks(p.ca_files, file)
        + tls_checks(p, facts.tls, file)
        + plan_checks(p, file_checks, file, facts.image_here)
    )
    if facts.disk_free_gb:
        st = (
            Status.FAIL
            if facts.disk_free_gb < BUILD_DISK_NEED_GIB
            else (
                Status.WARN if facts.disk_free_gb < BUILD_DISK_WARN_GIB else Status.PASS
            )
        )
        out.append(
            checks.Check(
                "disk",
                st,
                f"{facts.disk_free_gb:.0f} GiB free at {facts.disk_where} (a build with ROCm takes about "
                f"{BUILD_DISK_NEED_GIB} GiB: the image and its stages' cache)",
                (
                    None
                    if st == Status.PASS
                    else Steps(
                        (
                            "free space there:",
                            Command(("docker", "system", "df")),
                            "shows what takes it;",
                            Command(("docker", "builder", "prune")),
                            "frees the build cache",
                        )
                    )
                ),
            )
        )
    return out


# --- docker/.env ------------------------------------------------------------------------------------------------------


def ports(values: dict) -> list:
    """(address, port) of each dashboard port docker/.env publishes, by its variables VALUES (deploy_plan.env_values):
    the team dashboard's while VALUES has one; a blank address is this host only (container_layout.LOOPBACK).
    """
    here = container_layout.LOOPBACK
    out = [(values.get("RKMON_ADMIN_HOST") or here, int(values["RKMON_PORT"]))]
    if values.get("RKMON_TEAM_PORT"):
        out.append(
            (
                values.get("RKMON_TEAM_HOST") or here,
                int(values["RKMON_TEAM_PORT"]),
            )
        )
    return out


def configuration(cfg: dict) -> dict:
    """The configuration the container runs with CFG, the configuration a deploy settings file makes (`read`):
    runtime_config.resolve in the container's layout, over the package's defaults and the repository defaults for its
    repository and folder.
    """
    # call-time imports: runtime_config reads file_types, which reads this module (through coherence)
    from . import derived, runtime_config

    return runtime_config.resolve(
        registry.package_defaults(),
        runtime_config.repository_for(cfg, None),
        cfg,
        {},
        derived.Layout(container=True),
    ).values


def no_image_row(cfg: dict, file: str = deploy_tree.FILE):
    """The deploy path's image step while the image CFG names is not on this host: a FAIL with its fix (FILE: the
    deploy settings file that names the image). Whether an image here suits this package is confirm.image_rows'.
    """
    name, tag = image_key("name"), image_key("tag")
    return checks.Check(
        "image",
        Status.FAIL,
        f"no image {image_ref(cfg)} here ({name} and {tag} name it)",
        Steps(
            (
                steps.command_line("image load IMAGE_FILE"),
                ", IMAGE_FILE being the image's file (or",
                steps.command_line("image pull"),
                f", from {N('deployment.pull_registry')}); or, for an image here under another name or tag (",
                steps.command_line("image ls"),
                "lists them),",
                Edit(name, file, Described("its name")),
                "and",
                Edit(tag, file, Described("its tag")),
                ", then run",
                steps.command_line("confirm"),
            )
        ),
    )


# A TLS probe of a public site, run in the image (`python3 -c`): it prints `TLS ok`, `TLS untrusted <why>` or
# `TLS unreachable <why>`; rkmon image check reads it (image_tls_checks).
TLS_PROBE = (
    "import socket, ssl\n"
    "try:\n"
    "    with socket.create_connection(('github.com', 443), timeout=10) as raw:\n"
    "        with ssl.create_default_context().wrap_socket(raw, server_hostname='github.com'):\n"
    "            print('TLS ok')\n"
    "except ssl.SSLCertVerificationError as exc:\n"
    "    print('TLS untrusted', exc.verify_message)\n"
    "except OSError as exc:\n"
    "    print('TLS unreachable', exc)\n"
)


def image_tls_checks(out: str) -> list:
    """The image's own trust on this network (TLS_PROBE's answer): a network that inspects TLS with a root the image
    does not trust fails every GitHub, PyPI and model call of the monitor."""
    if "TLS ok" in out:
        return [
            checks.Check(
                "TLS in the image",
                Status.PASS,
                "the image trusts this network's TLS (github.com)",
            )
        ]
    if "TLS untrusted" in out:
        return [
            checks.Check(
                "TLS in the image",
                Status.FAIL,
                "this network inspects TLS, and the image does not trust its root certificate (github.com: "
                + out.split("TLS untrusted", 1)[1].strip()[:120]
                + ")",
                Steps(
                    (
                        Edit(CA_KEY, image.BUILD_FILE, CA_FILES),
                        ", build the image again (QUICK_BUILD_GUIDE.md), then",
                        steps.command_line("image load IMAGE_FILE"),
                    )
                ),
            )
        ]
    return [
        checks.Check(
            "TLS in the image",
            Status.WARN,
            "github.com did not answer from the image here: "
            + (
                out.split("TLS unreachable", 1)[1].strip()[:120]
                if "TLS unreachable" in out
                else out.strip()[:120]
            ),
            Steps(
                (
                    "the monitor needs GitHub: check this host's network, then run",
                    steps.command_line("image check"),
                )
            ),
        )
    ]


# --- host facts and the deploy path's check -----------------------------------------------------------------------------


@dataclass
class Facts:
    """What the wrapper asked docker and the host (read-only); the tests make their own."""

    docker: str = ""  # the server's version, "" when docker does not answer
    docker_error: str = ""
    compose: str = ""  # docker compose's version
    buildx: str = ""  # docker buildx's version (the build secrets need BuildKit)
    disk_free_gb: float = 0.0
    disk_where: str = ""
    host_gb: float = 0.0
    host_cpus: int = 0
    # (address, port) of each port the deployment opens that another program or container uses; None when not
    # asked: the deploy settings file's own rows FAIL, so the ports it makes are not known yet
    busy_ports: list | None = field(default_factory=list)
    # identity's answer about container.name on this host: NONE, OURS or OTHER; "" when docker was not asked
    name_owner: str = ""
    name_unknown: str = (
        ""  # why whose container.name is cannot be told (docker's answer), when it cannot
    )
    project_volumes: list = field(
        default_factory=list
    )  # this project's volumes that exist already
    first_settings: bool = True  # this folder has recorded no settings file yet
    tls: dict = field(default_factory=dict)  # tls_probe's answer
    image_here: bool = (
        False  # the image the build settings name is on this machine already
    )
    python: tuple = ()


def host_memory_gb(meminfo: str = "/proc/meminfo") -> float:
    try:
        for line in pathlib.Path(meminfo).read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                return round(int(line.split()[1]) / 2**20, 1)
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


# This host's memory and CPUs, which the container's share of it is judged against (coherence.resources,
# checks 8 and 9). RKMON_HOST_SIZE in the environment, "GIB,CPUS" (such as "64,16"), sets both: the tests,
# which judge their files the same on every machine.
_HOST_SIZE = re.compile(r"(\d+(?:\.\d+)?),(\d+)")


def host_size(environ=os.environ, meminfo: str = "/proc/meminfo") -> tuple:
    """(memory in GiB, CPUs) of this host: RKMON_HOST_SIZE's, else MEMINFO's MemTotal (host_memory_gb) and
    os.cpu_count(); 0 for a figure that cannot be read, which is then not compared."""
    m = _HOST_SIZE.fullmatch(str(environ.get("RKMON_HOST_SIZE") or ""))
    if m:
        return float(m[1]), int(m[2])
    return host_memory_gb(meminfo), os.cpu_count() or 0


def port_free(host: str, port: int) -> bool:
    """Nothing listens on host:port here (a bind succeeds)."""
    fam = socket.AF_INET6 if ":" in host else socket.AF_INET
    s = socket.socket(fam, socket.SOCK_STREAM)
    try:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((host or container_layout.EVERY_ADDRESS, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


def busy_port_row(host: str, port: int, fix) -> checks.Check:
    """The FAIL row of the dashboard port HOST:PORT, which another program or container uses, with FIX: each of
    busy_port_checks's, and those of the settings rollback puts back, whose fix is rollback's own.
    """
    return checks.Check(
        "port",
        Status.FAIL,
        f"{host}:{port} is in use by another program or container",
        fix,
    )


def busy_port_checks(cfg: dict, busy: list, file: str = deploy_tree.FILE) -> list:
    """A FAIL for each dashboard port of BUSY, the (address, port) pairs that another program or container uses
    (the check's, preflight, and the start's), with its fix: the key of FILE, the deploy settings file, that sets it.
    """
    admin_port = admin_public_port(cfg)
    return [
        busy_port_row(
            host,
            port,
            Edit(
                deploy_tree.name(
                    "admin_dashboard.public_port"
                    if port == admin_port
                    else "team_dashboard.public_port"
                ),
                file,
                Described("a port that is free on this host"),
            ),
        )
        for host, port in busy
    ]


# The public site tls_probe asks, (host, port): github.com, where the build downloads from. RKMON_TLS_SITE in the
# environment (host:port) moves it: the tests, which reach no public site.
TLS_SITE = ("github.com", 443)


def tls_site(environ=os.environ) -> tuple:
    """(host, port) of the site tls_probe asks: RKMON_TLS_SITE's host:port, else TLS_SITE."""
    host, _, port = str(environ.get("RKMON_TLS_SITE") or "").rpartition(":")
    return (host, int(port)) if host and port.isdigit() else TLS_SITE


def tls_probe(cafiles: list, timeout: float = 8, environ=os.environ) -> dict:
    """Whether this network inspects TLS: the certificate a public site (tls_site) presents here, checked with this
    host's own trust store ({"ok", "issuer", "error"}), and with the settings' CA certificates alone when there are
    some ({"with_ca"}: "root" when they verify it up to a root, "partial" when they verify it only from an issuing
    certificate, False when they do not verify it, None when none are named)."""
    host, port = tls_site(environ)
    out = {"host": host, "ok": False, "issuer": "", "error": "", "with_ca": None}

    def fetch(ctx):
        with socket.create_connection((host, port), timeout=timeout) as raw:
            with ctx.wrap_socket(raw, server_hostname=host) as s:
                return s.getpeercert()

    try:
        cert = fetch(ssl.create_default_context())
        out["ok"] = True
        issuer = dict(x[0] for x in cert.get("issuer", ()) if x)
        out["issuer"] = issuer.get("organizationName") or issuer.get("commonName") or ""
    except ssl.SSLCertVerificationError as exc:
        out["error"] = f"the certificate is not trusted ({exc.verify_message})"
    except (OSError, ssl.SSLError) as exc:
        out["error"] = f"no connection ({exc})"
    if cafiles:
        # The settings' certificates alone: as a full chain up to a root, the one tls_checks passes (pip trusts a
        # chain only up to a root), else as a partial chain, which tells issuing certificates without their root from
        # certificates that do not verify this network at all.
        out["with_ca"] = False
        for partial in (False, True):
            try:
                ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
                if partial:
                    if not hasattr(ssl, "VERIFY_X509_PARTIAL_CHAIN"):
                        break
                    ctx.verify_flags |= ssl.VERIFY_X509_PARTIAL_CHAIN
                for f in cafiles:
                    ctx.load_verify_locations(cafile=str(f))
                fetch(ctx)
                out["with_ca"] = "partial" if partial else "root"
                break
            except (OSError, ssl.SSLError, ValueError):
                continue
    return out


def inspected(probe: dict) -> bool:
    """The probe met a TLS-inspecting proxy: an untrusted chain, or a trusted one from no public issuer."""
    if not probe:
        return False
    if probe.get("error", "").startswith("the certificate is not trusted"):
        return True
    issuer = probe.get("issuer") or ""
    return bool(
        probe.get("ok")
        and issuer
        and not any(p.lower() in issuer.lower() for p in PUBLIC_CA)
    )


def tls_checks(p, probe: dict, file: str = image.BUILD_FILE) -> list:
    """This network's TLS (PROBE, tls_probe's answer) against the root certificates build plan P names (CA_KEY), the
    fixes in FILE. Only a full chain up to a root passes, on every base: pip, which installs the image's Python
    packages, trusts a chain only up to a root. No row names the issuer of the certificate the network presents.
    """
    named = bool(p.ca_files)
    if not probe:
        return []
    if probe.get("error", "").startswith("no connection"):
        return [
            checks.Check(
                "network",
                Status.FAIL,
                f"{probe['host']}: {probe['error']}",
                Steps(
                    (
                        "the build downloads from GitHub, PyPI and the ROCm installer's host: fix this host's network "
                        "or proxy, then run",
                        *check_again(file),
                    )
                ),
            )
        ]
    if inspected(probe):
        if named and probe.get("with_ca") == "partial":
            return [
                checks.Check(
                    "TLS",
                    Status.FAIL,
                    f"this network inspects TLS, and {CA_KEY} names only issuing certificates of its chain, not its "
                    "root: pip, which installs the image's Python packages, needs the root",
                    Steps(
                        (
                            "get your network's root certificate, the top certificate of its chain (the one that "
                            "signs itself): ask your IT team for it, or export the top certificate from your "
                            "browser's certificate viewer; then",
                            Edit(CA_KEY, file, CA_FILES),
                        )
                    ),
                )
            ]
        if named and probe.get("with_ca"):
            return [
                checks.Check(
                    "TLS",
                    Status.PASS,
                    f"this network inspects TLS; {CA_KEY} covers it",
                )
            ]
        if named:
            return [
                checks.Check(
                    "TLS",
                    Status.FAIL,
                    f"this network inspects TLS, and the certificate {CA_KEY} names does not verify it",
                    Edit(
                        CA_KEY,
                        file,
                        Described(
                            "your network's ROOT certificate file, not an intermediate's"
                        ),
                    ),
                )
            ]
        return [
            checks.Check(
                "TLS",
                Status.FAIL,
                "this network inspects TLS, and no root certificate is named",
                Edit(CA_KEY, file, CA_FILES),
            )
        ]
    return [
        checks.Check(
            "TLS",
            Status.PASS,
            f"{probe['host']} answers with a public certificate",
        )
    ]


def preflight(
    cfg: dict | None,
    facts: Facts,
    file_checks: list,
    file: str = deploy_tree.FILE,
    share: list = (),
) -> list:
    """The deploy path's step 2, in order: docker and compose, the deploy settings file and its settings, disk, memory
    and CPUs (SHARE: the file's rows on the container's share of this host, coherence.resources with FACTS' memory and
    CPUs), the ports, the container name; the fixes edit FILE, the deploy settings file. (This network's TLS is the
    image's to trust: step 4 checks it inside the image.)"""
    out = []
    if facts.docker:
        out.append(checks.Check("docker", Status.PASS, f"server {facts.docker}"))
    else:
        out.append(
            checks.Check(
                "docker",
                Status.FAIL,
                facts.docker_error or "docker does not answer",
                Steps(
                    (
                        "install Docker (or Podman: CONTAINER_ENGINE=podman) and start its daemon (Docker's install "
                        "guide for your distribution), then run",
                        *check_again(file),
                    )
                ),
            )
        )
    out.append(
        checks.Check("docker compose", Status.PASS, facts.compose)
        if facts.compose
        else checks.Check(
            "docker compose",
            Status.FAIL,
            "no compose v2 (`docker compose version` fails)",
            Steps(
                (
                    "install the docker compose plugin (docker-compose-plugin), then run",
                    *check_again(file),
                )
            ),
        )
    )
    if facts.python and facts.python < HOST_PYTHON:
        floor = ".".join(map(str, HOST_PYTHON))
        out.append(
            checks.Check(
                "host Python",
                Status.FAIL,
                f"{'.'.join(map(str, facts.python))}: {program.name()} needs Python {floor} or newer",
                Steps(
                    (
                        f"run it with Python {floor} or newer, such as",
                        Command(
                            (
                                f"python{floor}",
                                "docker/rkmon",
                                *steps.typed(
                                    "check --settings FILE",
                                    **(
                                        {}
                                        if file == deploy_tree.FILE
                                        else {"FILE": file}
                                    ),
                                ),
                            )
                        ),
                        "in the package folder"
                        + (
                            ", FILE being that file" if file == deploy_tree.FILE else ""
                        ),
                    )
                ),
            )
        )
    out += file_checks
    if cfg is None:
        return out
    out += settings_checks(cfg, file)
    if facts.disk_free_gb:
        st = (
            Status.FAIL
            if facts.disk_free_gb < 30
            else Status.WARN if facts.disk_free_gb < 60 else Status.PASS
        )
        out.append(
            checks.Check(
                "disk",
                st,
                f"{facts.disk_free_gb:.0f} GiB free at {facts.disk_where} (the image is about 20 GiB with ROCm, its file "
                "a third of that while it is loaded; the clone, worktrees and builds tens more)",
                (
                    None
                    if st == Status.PASS
                    else Steps(
                        (
                            "free space there:",
                            Command(("docker", "system", "df")),
                            "shows what takes it;",
                            Command(("docker", "image", "prune")),
                            "removes the images no container uses",
                        )
                    )
                ),
            )
        )
    out += share
    from . import features

    if not features.on(configuration(cfg or {}), "admin-dashboard"):
        out.append(
            checks.Check(
                "ports",
                Status.PASS,
                "not needed: features.dashboards.admin is off, so no web server runs",
            )
        )
    elif facts.busy_ports:
        out += busy_port_checks(cfg, facts.busy_ports, file)
    elif facts.busy_ports is not None:
        out.append(
            checks.Check(
                "ports",
                Status.PASS,
                "the dashboards' ports are free (or this deployment's)",
            )
        )
    name = project(cfg)
    taken = name_row(name, facts.name_owner, facts.name_unknown, file)
    if taken is not None:
        out.append(taken)
    elif facts.first_settings and facts.project_volumes:
        out.append(
            checks.Check(
                "container name",
                Status.WARN,
                f"volumes of {name} exist already ({', '.join(facts.project_volumes)}): this deployment would take "
                "over their state and logins",
                Steps(
                    (
                        "keep them only if they are this deployment's; else",
                        Edit(
                            deploy_tree.name("deployment.project"),
                            file,
                            Described("a name of its own"),
                        ),
                    )
                ),
            )
        )
    elif facts.name_owner in (identity.NONE, identity.OURS):
        out.append(
            checks.Check(
                "container name",
                Status.PASS,
                f"{name}: no other deployment on this host uses it",
            )
        )
    return out


def name_row(name: str, owner: str, unknown: str, file: str = deploy_tree.FILE):
    """The container name's FAIL row when NAME is another deployment's on this host (OWNER, identity's answer, is
    OTHER), or whose it is cannot be told (UNKNOWN says why); None otherwise. The row never says which folder the
    other deployment runs from. FILE is the deploy settings file the fix edits."""
    if unknown:
        return checks.Check(
            "container name",
            Status.FAIL,
            f"whose {name} is on this host is unknown: {unknown}",
            Steps(
                (
                    "run",
                    Command(("docker", "ps", "--all")),
                    "to see docker's own answer; once it answers, run",
                    *check_again(file),
                )
            ),
        )
    if owner == identity.OTHER:
        return checks.Check(
            "container name",
            Status.FAIL,
            f"{name} is used by another deployment on this host",
            states.name_fix(name, file),
        )
    return None


# --- the doctor runs of apply and rollback -----------------------------------------------------------------------------


DOCTOR_ROW = re.compile(
    r"^(?:  )?(" + "|".join(s.value for s in Status) + r")\s+(.+?):\s"
)  # a row's first line in rk doctor's report: its status (one of checks.Status's words), then its name


class DoctorRow(NamedTuple):
    """A row of rk doctor's report (doctor_rows): its status (one of checks.Status's words), its name, and its lines.
    It reads as a check row does (status, name), so outcome.tally counts a report's rows.
    """

    status: str
    name: str
    lines: list


def doctor_rows(text: str) -> list:
    """rk doctor's report as its rows (DoctorRow): the row's line and the lines it goes on over (the rest of what it
    found, and its fix), as checks.render prints them (`  FAIL  NAME: ...`, the next lines from column 8) or as an
    older image prints them (`FAIL  NAME: ...`, the next lines from column 6). Any other line ends a row.
    """
    rows, row = [], None
    for line in (text or "").splitlines():
        m = DOCTOR_ROW.match(line)
        if m:
            row = DoctorRow(m.group(1), m.group(2).strip(), [line])
            rows.append(row)
        elif row is not None and line.startswith(" " * 6):
            row.lines.append(line)
        else:
            row = None
    return rows


def new_failures(before: list, after: list) -> list:
    """The checks that fail after a change (rkmon apply's doctor after it, and rollback's) and did not before it, the
    baseline. A name read from an older image's report may be cut at its first colon (`session login` for
    `session login: gh`): such a name covers the longer ones it starts.
    """

    def known(x: str) -> bool:
        return any(
            x == b or x.startswith(b + ":") or b.startswith(x + ":") for b in before
        )

    return [x for x in after if not known(x)]
