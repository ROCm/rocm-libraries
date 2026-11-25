# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The container image: its name and its title, the bases it builds on, the build context of the package's code
with the quick guides' names, the package's version, and the steps to another image.

Building and deploying are two jobs, on any machines. The build is rkmon_lib/build_plan.py's: a BUILD settings file
(`kind: build_settings`, QUICK_BUILD_GUIDE.md) becomes one `docker build`, and the image carries the labels and the
build record build_plan lists. The deploy machine never builds: its DEPLOY settings file (QUICK_DEPLOY_GUIDE.md) names
the image to run, and `rkmon image check`, confirm and start refuse an image not built from this package's image
recipe (rkmon_lib/confirm.py image_rows).

The image's code is the package's build context as docker reads the package's .dockerignore: `dockerignore(root)`,
its patterns in their order; `ignored(patterns, path)`, whether they leave PATH (a file or a folder, relative to the
package folder) out; `code_files(root)`, the package's files that make the image's code, which `rkmon apply` and
`rkmon start` install as a release (docker/rk_root.py code-install) so that the same code makes the same release;
`code_sha256(root)`, the SHA-256 of that code, "sha256:…": code_volume's digest of those files (their paths, bytes
and execute bits), which the settings' fingerprint covers, so that other code under the same VERSION is
a change to confirm too. The three that read ROOT raise an OSError when it has no .dockerignore.

This module runs on the host too (docker/rkmon, on the host's Python: deployment.HOST_PYTHON), so it uses the
standard library only.
"""

import json
import os
import pathlib
import re

from . import build_plan

# The image's name when the settings leave it blank (image.name in the build and in the deploy settings: their
# schemas' default). The image is <name>:<tag>.
DEFAULT_NAME = "rocke-monitor"
# The title label every image of this package carries (build_plan's labels), whatever its name: rkmon image ls finds
# the package's images by it.
TITLE = "rocke-monitor"
ROOT = pathlib.Path(__file__).resolve().parent.parent  # the package folder
# The bases the image builds on: (image, what it is, the family docker/pkgs.sh installs with); the first is the
# default, DEFAULT_BASE (the build schema's operating_system.base_image takes these, its default the first).
BASES = (
    (
        "ubuntu:26.04",
        "Ubuntu 26.04 (the default; its Python is the one the live monitor runs)",
        "deb",
    ),
    ("ubuntu:24.04", "Ubuntu 24.04", "deb"),
    (
        "rockylinux:9",
        "Rocky Linux 9, the RHEL 9 family (Python from AppStream; EPEL)",
        "rpm",
    ),
    (
        "almalinux:9",
        "AlmaLinux 9, the RHEL 9 family (as Rocky Linux 9)",
        "rpm",
    ),
)
DEFAULT_BASE = BASES[0][0]
BOM = chr(0xFEFF)
# The quick guides at the package's top, each for its reader: the operator's, the image builder's and the deployer's.
# `rk guide` prints them from the code that runs, in the container the image's (rkmon.py cmd_guide), so the build
# context takes them; tools/generated.py's docs index names them.
USER_GUIDE = "QUICK_USER_GUIDE.md"
BUILD_GUIDE = "QUICK_BUILD_GUIDE.md"
DEPLOY_GUIDE = "QUICK_DEPLOY_GUIDE.md"
# The build context of the package's code: the package's entries at its top that the image takes, every one but those
# of PACKAGE_ONLY. .dockerignore lets in these and nothing else (tools/generated.py writes it), and under them leaves
# out, wherever they are, the files and folders a secret or a private value may be in
# (secrets_registry.ignore_patterns); docker/Dockerfile copies what is left whole.
BUILD_CONTEXT = (
    ".dockerignore",
    ".gitignore",
    BUILD_GUIDE,
    DEPLOY_GUIDE,
    USER_GUIDE,
    "README.md",
    "VERSION",
    "agentic",
    "config.defaults.json",
    "conftest.py",
    "docker",
    "docs",
    "repository",
    "requirements",
    "rk",
    "rkmon.py",
    "rkmon_lib",
    "schema",
    "tests",
    "tools",
    "worker",
)
# The package's entries at its top that the image leaves out: config.json, a blank placeholder (a machine's
# configuration lives outside the package), and .claude, the setup skill that a deployer's own Claude Code session
# runs in the package folder on the host (/rkmon-setup), which the container has no use for.
PACKAGE_ONLY = ("config.json", ".claude")
BUILD_FILE = "your build settings file"  # the file in a fix's words, where no path of it is at hand


# --- the image's code: the build context as docker reads .dockerignore ------------------------------------------------


def dockerignore(root=ROOT) -> list:
    """The patterns of the .dockerignore in the package folder ROOT, in their order: its lines but blank lines and
    comments. An OSError when there is none."""
    lines = (
        (pathlib.Path(root) / ".dockerignore").read_text(encoding="utf-8").splitlines()
    )
    return [x.strip() for x in lines if x.strip() and not x.startswith("#")]


def _pattern(p: str) -> re.Pattern:
    """A .dockerignore pattern as a regular expression: `**` is any number of folders, `*` and `?` stay in one."""
    out, i = "", 0
    while i < len(p):
        if p.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif p.startswith("**", i):
            out, i = out + ".*", i + 2
        elif p[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif p[i] == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(p[i]), i + 1
    return re.compile(out)


def ignored(patterns: list, path: str) -> bool:
    """A .dockerignore's verdict on PATH (relative to the context): the last pattern that matches it or a folder above
    it wins; `!` takes a path back."""
    parts = path.split("/")
    prefixes = ["/".join(parts[: k + 1]) for k in range(len(parts))]
    verdict = False
    for raw in patterns:
        neg = raw.startswith("!")
        r = _pattern((raw[1:] if neg else raw).strip("/"))
        if any(r.fullmatch(x) for x in prefixes):
            verdict = not neg
    return verdict


def code_files(root=ROOT) -> list:
    """The files of the package folder ROOT that make its image's code, relative to it, sorted: each regular file its
    .dockerignore lets into the build context, which docker/Dockerfile copies whole. A link, or any other file that is
    not a regular one, is passed over; so is a folder the patterns leave out, unless a `!` pattern could take back
    something in it."""
    root = pathlib.Path(root)
    patterns = dockerignore(root)
    back = [p[1:].strip("/") for p in patterns if p.startswith("!")]

    def pruned(rel: str) -> bool:
        return ignored(patterns, rel) and not any(
            set("*?[") & set(b) or b.startswith(rel + "/") for b in back
        )

    out = []
    for folder, dirs, files in os.walk(root):
        here = pathlib.Path(folder).relative_to(root).as_posix()
        prefix = "" if here == "." else here + "/"
        dirs[:] = [
            d
            for d in dirs
            if not os.path.islink(os.path.join(folder, d)) and not pruned(prefix + d)
        ]
        for name in files:
            path = os.path.join(folder, name)
            if os.path.islink(path) or not os.path.isfile(path):
                continue
            if not ignored(patterns, prefix + name):
                out.append(prefix + name)
    return sorted(out)


def code_sha256(root=ROOT) -> str:
    """The SHA-256 of the code of the package folder ROOT, "sha256:…": see the module's docstring."""
    from . import code_volume

    return "sha256:" + code_volume.digest(root, code_files(root))


def family(base: str) -> str:
    """deb or rpm for a base image ("" for one the package does not know)."""
    for image, _what, fam in BASES:
        if base == image:
            return fam
    name = base.split("/")[-1].split(":")[0]
    if name in ("ubuntu", "debian"):
        return "deb"
    if name in ("rockylinux", "almalinux", "rhel", "ubi9", "centos"):
        return "rpm"
    return ""


# --- what an image carries, and the steps to another image ------------------------------------------------------------


def package_version(pkg=ROOT) -> str:
    """The package's VERSION (code_volume.version); "" when there is none."""
    from . import code_volume

    return code_volume.version(pkg)


def own_image_fix(program_name: str):
    """The fix for an image this package did not build, as the operator types it (PROGRAM_NAME: the command rkmon is
    run by, which rk doctor in the container is told): an image this package built, named to the deployment as any
    new image is (new_image_steps; rkmon_lib/checks.py Steps)."""
    from .checks import Steps

    return Steps(
        (
            "run an image built by this package (QUICK_BUILD_GUIDE.md):",
            *new_image_steps(program_name),
        )
    )


def new_image_steps(program_name: str) -> tuple:
    """The steps that move a deployment onto a new image once there is one, as the operator types
    them (PROGRAM_NAME, as own_image_fix's): image.tag set to its tag (tag_step), then the confirm, whose next line
    leads on (the image's load or pull while it is not here, state 6). The parts of a checks.Steps.
    """
    from .checks import Command

    return (tag_step(), ", then", Command((program_name, "confirm")))


def tag_step():
    """The step that names a new image to a deployment: image.tag set to its tag in your deploy
    settings file (rkmon_lib/checks.py Edit), the one home of its words for every fix that leads to a new image
    (new_image_steps; rkmon_lib/releases.py new_image_fix)."""
    from .checks import Described, Edit

    return Edit("image.tag", "your deploy settings file", Described("its tag"))


def describe(labels: dict) -> str:
    """One line for an image's labels: its package version, build date, base and ROCm."""
    if not labels or build_plan.PACKAGE_LABEL not in labels:
        return "no build labels"
    return (
        f"package {labels.get(build_plan.PACKAGE_LABEL) or '?'}, "
        f"built {labels.get(build_plan.DATE_LABEL) or '?'}, "
        f"{labels.get(build_plan.BASE_LABEL) or '?'}, "
        f"ROCm {labels.get(build_plan.ROCM_LABEL) or '?'}"
    )


# --- settings files -------------------------------------------------------------------------------------------------


def dotenv(path) -> dict:
    """The NAME=value lines of the docker/.env at PATH (dotenv_text); {} when there is none."""
    try:
        return dotenv_text(pathlib.Path(path).read_text(encoding="utf-8"))
    except OSError:
        return {}


def dotenv_text(text: str) -> dict:
    """The NAME=value lines of a docker/.env's TEXT, NAME -> value, as compose reads them: `#` comments, an optional
    `export `, optional quotes."""
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = v.strip()
        if len(v) >= 2 and v[0] == v[-1] and v[0] in "'\"":
            v = v[1:-1]
        out[k.strip()] = v
    return out


def load(path) -> dict:
    """A settings file's data (JSON, or the YAML subset rkmon_lib/answers.py reads); a mapping at the top level."""
    import sys

    sys.path.insert(0, str(ROOT))
    text = pathlib.Path(path).read_text(encoding="utf-8")
    if text.lstrip(BOM).lstrip().startswith("{"):
        data = json.loads(text)
    else:
        from rkmon_lib import answers

        data = answers.parse(text)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: not a mapping at the top level")
    return data
