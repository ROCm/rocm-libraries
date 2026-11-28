# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk doctor`'s checks: the machine's prerequisites (checked, never installed), the configuration and
its paths, the secrets' permissions, and the features.

Each check is a row of rkmon_lib/checks.py: PASS, FAIL, WARN (advice, or an optional tool), OFF (a feature that is off)
or SKIP (not needed with this configuration), with what was found (and, for a WARN or a FAIL, what it blocks) and, for
a WARN or a FAIL, its fix: a command, an edit, or steps of them. `report` prints
them as checks.render does, then the count of each status. A fix is as the operator types it on the host, through the
container's wrapper (`<wrapper> ...`): an rk command is `<wrapper> rk ...` (`rk_command`), a setting an edit of the
deploy settings file, then `<wrapper> confirm` and `<wrapper> apply` (`settings_parts`), a login or a secret its own
command; `section` names the fix of a part of the deployment: the start that sets it up, or the settings that hold it.
The prerequisites and what each blocks come from prerequisites.txt: each is a tool every image of this
package carries, so a missing one's fix is such an image (image_fix). A missing tool blocks only what needs it: a core
one fails (the first tick needs it), one review automation needs fails while it is enabled and is advice otherwise,
and an optional one is advice. Nothing here installs, starts, stops or changes anything.

Exit codes (rk doctor): 0 nothing fails; 3 a prerequisite is missing or too old (the report names
each one, what it blocks and its fix); 4 the prerequisites are there, but the configuration, its folders or its secrets
fail a check.

The first pass's check covers its chosen model only (features.review_automation.passes.first_review_pass.model): nothing
for the default (the verifier does it); for a model binding, its address, model and key are at hand (the live read-only
check of its endpoint is reach.gateways'); for the local worker, a read-only presence check (the endpoint's model list),
skipped while a review worker runs. Ollama is a prerequisite only while something uses the local worker on a local
Ollama (the first pass, the tick triage, the planner or the analyst).
"""

from __future__ import annotations

import dataclasses
import json
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass

from . import (
    build_plan,
    code_volume,
    coherence,
    config,
    container_layout,
    deploy_tree,
    features,
    paths,
    private_files,
    program,
    releases,
    repository_defaults,
)
from . import gh as gh_mod
from .checks import (
    Check,
    Command,
    Described,
    Edit,
    Status,
    Steps,
    check_to_json,
    render,
)

TABLE_FILE = pathlib.Path(__file__).with_name("prerequisites.txt")
EXIT_OK, EXIT_MISSING, EXIT_CONFIG = 0, 3, 4
DISK_WARN_GB, DISK_FAIL_GB = (
    60,
    15,
)  # in GiB: the clone, worktrees and builds of a busy week need tens of GiB
GIT_MIN = (2, 38)  # git merge-tree --write-tree
PY_MIN = (3, 10)


@dataclass(frozen=True)
class Prereq:
    key: str
    level: str  # core (the first tick needs it), review (needed while review automation is enabled), optional
    name: str
    look: str  # commands, or :llvm
    blocks: str  # what it blocks while it is missing


def table() -> dict[str, Prereq]:
    """prerequisites.txt: key -> Prereq."""
    out = {}
    for line in TABLE_FILE.read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            key, level, name, look, blocks = line.split("|", 4)
            out[key] = Prereq(key, level, name, look, blocks)
    return out


PREREQS = table()
PREREQ_NAMES = frozenset(
    p.name for p in PREREQS.values()
)  # the rows of prerequisites.txt: one that fails is exit 3


def run(cmd: list[str], timeout: int = 20, cwd: str | None = None) -> tuple[int, str]:
    """A command's exit code and its output; 127 when it is not installed. Never raises."""
    try:
        p = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd
        )
        return p.returncode, (p.stdout + p.stderr).strip()
    except FileNotFoundError:
        return 127, ""
    except (OSError, subprocess.SubprocessError) as exc:
        return 126, str(exc)


def _version(text: str) -> str:
    m = re.search(r"\d+\.\d+(?:\.\d+)?", text)
    return m.group(0) if m else (text.splitlines()[0][:60] if text else "")


def in_container(root: pathlib.Path = pathlib.Path("/"), environ=os.environ) -> bool:
    """Inside a container (Docker, Podman, a Kubernetes pod), whichever image it runs and whatever its kernel (Docker
    Desktop's is WSL's): the image test's tarball stage runs only in one (tests/container/stages.py).
    """
    if (
        (root / ".dockerenv").exists()
        or (root / "run" / ".containerenv").exists()
        or environ.get("container")
    ):
        return True
    try:
        cgroup = (root / "proc" / "1" / "cgroup").read_text()
    except OSError:
        return False
    return any(
        w in cgroup for w in ("docker", "containerd", "kubepods", "libpod", "/lxc")
    )


def deployment_image(environ=os.environ) -> bool:
    """Inside the package's own container image (docker/Dockerfile sets RKMON_CONTAINER=1), where the monitor runs:
    its volumes and its orchestrator. Outside it (the suite) the rows of what only the container has do not show.
    """
    return environ.get("RKMON_CONTAINER") == "1"


def wrapper_name(environ=os.environ) -> str:
    """The container's command on the host, as the operator types it: the name the package's wrapper of docker
    compose was run by, which it passes in (RKMON_WRAPPER), else the program's name (program.name()).
    """
    return (environ.get("RKMON_WRAPPER") or "").strip() or program.name()


def wrapper(environ=os.environ) -> str:
    """wrapper_name, quoted for a shell."""
    return shlex.quote(wrapper_name(environ))


# --- the fixes, as the operator types them on the host ----------------------------------------------------------------


def rk_command(*args: str, environ=os.environ) -> Command:
    """The rk command ARGS as the operator types it: through the container's wrapper from the host (`<wrapper> rk
    ARGS`), never rk alone."""
    return Command((wrapper_name(environ), "rk", *args))


# The fix of a part of the deployment that a row names (`section`), by the part's name: the stop and the start
# (steps.restart_parts), whose rk provision puts back what has gone missing of the work volume's set-up (its folders,
# the clone, the checks venv, the agentic files); or the sections of the deploy settings file that hold its settings,
# then `<wrapper> confirm` and `<wrapper> apply`.
SECTION_STARTS = ("deploy", "agentic")
SECTION_SETTINGS = {
    "reviewer": (
        "features.review_automation.passes.first_review_pass",
        "model_bindings",
    ),
    "model": ("model_bindings",),
    "bindings": ("model_bindings",),
    "jira": ("features.ticket_creation", "integrations.atlassian_cli"),
    "gpu": ("features.review_automation.remote_gpu_tests",),
    "publish": (coherence.TEAM,),
}


def section_parts(name: str, environ=os.environ) -> tuple:
    """The fix of the part NAME (SECTION_STARTS, SECTION_SETTINGS), as steps' parts, as the operator types them on the
    host."""
    from . import steps

    if name in SECTION_STARTS:
        return steps.restart_parts(wrapper_name(environ))
    where = " and ".join(SECTION_SETTINGS[name])
    return (
        f"edit {where} in {deploy_tree.FILE}, then",
        *apply_commands(environ),
    )


def section(name: str, environ=os.environ):
    """section_parts as a fix: the command alone, or the steps."""
    parts = section_parts(name, environ)
    return parts[0] if len(parts) == 1 else Steps(parts)


def remove_then_start(what: str, environ=os.environ) -> Steps:
    """The fix of a piece of the work volume's set-up that is there but not whole: WHAT removed in the container's
    shell, then the stop and the start, whose rk provision makes it again (it removes nothing once the work volume is
    set up: rkmon_lib/provision.py)."""
    from . import steps

    w = wrapper_name(environ)
    return Steps(
        ("in", Command((w, "shell")), f", remove {what}, then", *steps.restart_parts(w))
    )


def clone_fix(cfg: dict, environ=os.environ) -> Steps:
    """The fix of the monitor's clone when it is not the one the toolkit expects (deploy.clone_check): the stop and
    the start clone it while it is missing; one that is there is removed first, with its worktrees, all in the working
    folder (paths.worktree)."""
    p = paths.resolve(cfg)
    if not os.path.lexists(p["repo"]):
        return Steps((*section_parts("deploy", environ), "clones it"))
    return remove_then_start(
        f"the clone {p['repo']} and its worktrees ({p['working_dir']}/wt*)", environ
    )


def gh_login_fix(side: str, environ=os.environ) -> Command:
    """The command that logs in the GitHub login of SIDE (gh's READ or WRITE), as the operator types it on the host:
    that login's own (`<wrapper> login github_read`, `<wrapper> login github_write`: secrets_registry.set_command).
    The session and its agents read with the read login (gh.settings)."""
    from . import secrets_registry as sr

    (name,) = [n for n, s in sr.GH_SIDES.items() if s == side]
    return Command((wrapper_name(environ), *sr.set_command(sr.entry({}, name))))


def check_again(environ=os.environ) -> Command:
    """The command that runs these checks again: `<wrapper> doctor`."""
    return Command((wrapper_name(environ), "doctor"))


def apply_commands(environ=os.environ) -> tuple:
    """How an edit of the deploy settings file comes to run, as steps' parts typed on the host
    (QUICK_USER_GUIDE.md "Change a setting"): `<wrapper> confirm` and `<wrapper> apply`. Every settings fix ends with
    it, wherever it shows (settings_parts, section_parts, settings_file; runtime_config's where there is no
    configuration): a later start never applies, and before the first start apply's answer names the next step.
    apply_words says it too.
    """
    w = wrapper_name(environ)
    return (Command((w, "confirm")), "and", Command((w, "apply")))


def apply_words(environ=os.environ, edit: str = f"edit {deploy_tree.FILE}") -> str:
    """How a setting changes, in a message's words, as the operator types it on the host: EDIT, the edit of the deploy
    settings file in words, then apply_commands."""
    return str(Steps((f"{edit}, then", *apply_commands(environ))))


def settings_edit(internal: str, value, key: str = "") -> Edit:
    """The edit of the deploy settings file that gives the configuration key INTERNAL its VALUE (a Described one when
    only the operator knows it), at the key as that file names it: KEY where a key of each provider's binding feeds
    INTERNAL, else deploy_tree.name."""
    return Edit(key or deploy_tree.name(internal), deploy_tree.FILE, value)


def settings_parts(internal: str, value, environ=os.environ, key: str = "") -> tuple:
    """The steps that give the configuration key INTERNAL its VALUE: its edit (settings_edit), then `<wrapper> confirm`
    and `<wrapper> apply` (apply_commands)."""
    return (settings_edit(internal, value, key), ", then", *apply_commands(environ))


def url_key(binding: str, kind: str) -> str:
    """The deploy settings key that gives the model binding BINDING, of the provider KIND, its address: an Ollama's
    server_url, a claude binding's the Claude session's (bindings.claude_url: integrations.claude_code's), any other's
    base_url."""
    from . import deploy_tree

    if kind == "claude":
        return deploy_tree.name("claude.gateway")
    return f"model_bindings.{binding}." + (
        "server_url" if kind == "ollama" else "base_url"
    )


def worker_url_key(w: dict) -> str:
    """The deploy settings key of the local worker's address W["url"] (config.worker): url_key's, of its binding."""
    return url_key("worker", w["backend"])


def settings_file(environ=os.environ) -> tuple:
    """Where the operator fixes settings by hand, as steps' parts: the deploy settings file, then `<wrapper> confirm`
    and `<wrapper> apply` (apply_commands)."""
    return (f"{deploy_tree.FILE}, then", *apply_commands(environ))


def settings_fix(internal: str, value, environ=os.environ) -> Steps:
    """settings_parts as a fix."""
    return Steps(settings_parts(internal, value, environ))


def cannot_run_fix(environ=os.environ) -> Steps:
    """The fix of a configuration that cannot run: the settings fixed in the deploy settings file (settings_file)."""
    return Steps(("fix it in", *settings_file(environ)))


def image_fix(environ=os.environ) -> Steps:
    """The fix for what every image of this package carries: run one (its file loaded here, then the monitor on it),
    as the wrapper's name types it (image.own_image_fix)."""
    from . import image

    return image.own_image_fix(wrapper_name(environ))


# What the Claude session does at each tick, which waits while it lacks something: the session's row (container_checks)
# and the claude login's (rkmon_lib/reach.py claude_test) say so.
SESSION_BLOCKS = "Claude's part of each tick: verification, gate decisions, posting"


def claude_login_fix(cfg: dict | None):
    """The fix of a Claude session with no key, or one refused: the claude login's command
    (secrets_registry.set_fix)."""
    from . import secrets_registry as sr

    return sr.set_fix(sr.entry(cfg or {}, sr.CLAUDE_LOGIN))


def blocking(found: str, blocks: str) -> str:
    """A row's found text with what it blocks while it is not fixed."""
    return f"{found}; it blocks {blocks}" if blocks else found


def claude_code(cfg: dict | None = None) -> tuple[str, str]:
    """(version or "", login state: "logged in" | "not logged in" | "unknown") of the image's Claude Code. Only the
    claude login counts, the key the Claude session reads (orchestrator.read_key): the doctor's row of the
    key is the login's test call (rkmon_lib/reach.py claude_login), and the configuration table reads this answer.
    """
    rc, out = run(["claude", "--version"])
    if rc != 0:
        return "", "unknown"
    from . import orchestrator

    logged = bool(cfg) and bool(orchestrator.read_key(cfg))
    return f"{_version(out)} (here)", "logged in" if logged else "not logged in"


def gh_status(cfg: dict, side: str = "read") -> dict:
    """The gh the toolkit uses (gh.settings: paths.gh or PATH) and one login's folder (gh.login_dirs: the read login by
    default, `side` "write" for the write one): version, where, login, scopes. Never the token. A folder this user
    cannot read (the container's write login: it belongs to the write user) has `readable` False and no login read.
    """
    binary = gh_mod.settings(cfg)[0]
    folder = gh_mod.login_dirs(cfg)[side]
    env = {"GH_CONFIG_DIR": folder} if folder else {}
    where = shutil.which(binary) or (binary if os.path.exists(binary) else "")
    out = {
        "binary": where or binary,
        "found": bool(where),
        "version": "",
        "version_ok": False,
        "login": "",
        "scopes": [],
        "config_dir": folder or "gh's own (~/.config/gh)",
        "logged_in": False,
        "readable": not folder
        or not os.path.exists(folder)
        or os.access(folder, os.R_OK | os.X_OK),
    }
    if not where:
        return out
    rc, text = run([binary, "--version"])
    v = re.findall(r"\d+", _version(text))[:2]
    out["version"] = _version(text)
    out["version_ok"] = bool(v) and tuple(map(int, v)) >= gh_mod.MIN_VERSION
    if not out["readable"]:
        return out
    try:
        p = subprocess.run(
            [binary, "auth", "status"],
            capture_output=True,
            text=True,
            timeout=30,
            env=dict(os.environ, **env),
        )
        status = (p.stdout + p.stderr).strip()
        rc = p.returncode
    except (OSError, subprocess.SubprocessError) as exc:
        status, rc = str(exc), 1
    out["login"] = gh_mod.logged_in_as(status)[1]
    out["logged_in"] = rc == 0 and bool(out["login"])
    s = re.search(r"Token scopes:\s*(.*)", status)
    out["scopes"] = re.findall(r"[\w:]+", s.group(1)) if s else []
    return out


def gh_needed_scopes(cfg: dict) -> list[str]:
    """The token scopes the enabled features need: read:org always (code-owner teams); repo (or public_repo, for a
    public repository) once anything writes: comments, reviews, issues, auto-merge."""
    from . import secrets_registry as sr

    return ["read:org"] + (
        ["repo or public_repo"]
        if sr.needed_by(cfg, sr.entry(cfg, "github_write"))
        else []
    )


LLVM_TOOLS = "ROCm's LLVM tools"  # what paths.rocm_llvm_bin names, in a fix's words


def rocm_image(path="") -> dict:
    """The ROCm release the container image installed, as its ROCm record says (container_layout.rocm_record, at PATH
    or the image's): {version, core, llvm_bin, clang, gfx, os, components, hipcc, sha256, installed_at}, or {} on a
    host install or an image built without ROCm.
    """
    return container_layout.rocm_record(path)


def image_lacks_fix(what: str, environ=os.environ) -> Steps:
    """The fix in the container for WHAT (in words), a part every image of this package carries, which the
    configuration takes from the image (rkmon_lib/derived.py: ROCm's LLVM tools, the checks venv), when it is missing
    or broken: the image lacks it, and `<wrapper> image check` checks the image against this package. No setting
    gives it in the container."""
    return Steps(
        (
            Command((wrapper_name(environ), "image", "check")),
            f"(the image lacks {what})",
        )
    )


def clang_line(llvm_bin: str) -> str:
    """The first line of `clang --version` in a ROCm LLVM tools folder ("" when there is none)."""
    if not llvm_bin or not (pathlib.Path(llvm_bin) / "clang").exists():
        return ""
    rc, text = run([str(pathlib.Path(llvm_bin) / "clang"), "--version"])
    return text.splitlines()[0].strip() if rc == 0 and text else ""


def clang_parity(cfg: dict, line: str | None = None) -> Check:
    """This machine's clang against the one the gates were tuned on: PASS when its `clang --version` line names
    build_plan.ROCM_CLANG_COMMIT, the gate compiler's llvm-project commit, as the image's build checks; WARN when it
    names another, as G7's compile results can differ; SKIP when there is no clang here. LINE is the line to read;
    None reads this machine's clang (paths.rocm_llvm_bin). The WARN's fix is an image of that release:
    paths.rocm_llvm_bin is the image's (rkmon_lib/derived.py).
    """
    name = "clang parity"
    commit = build_plan.ROCM_CLANG_COMMIT
    line = clang_line(paths.resolve(cfg)["rocm_llvm_bin"]) if line is None else line
    if not line:
        return Check(
            name,
            Status.SKIP,
            f"no clang here to compare with the gate compiler (llvm-project {commit})",
        )
    if commit in line:
        return Check(name, Status.PASS, f"{line}: the clang the gates were tuned on")
    return Check(
        name,
        Status.WARN,
        f"this clang: {line}; the gates were tuned on the clang of llvm-project {commit}, ROCm "
        f"{build_plan.ROCM_RELEASE}'s; G7's compile results can differ from the reference machine's",
        Steps(
            (
                Edit(
                    "rocm.version",
                    "your build settings file",
                    Described(
                        f"{build_plan.ROCM_RELEASE}, the release the gates were tuned on"
                    ),
                ),
                ", then build that image and run it (QUICK_BUILD_GUIDE.md)",
            )
        ),
    )


def image_rocm_check(rec: dict | None = None) -> Check:
    """The image's own ROCm, as its build recorded it (container_layout.ROCM_RECORD): every image this package builds
    carries one, so in the container an image without the record was not built by it."""
    rec = rocm_image() if rec is None else rec
    if rec:
        return Check(
            "ROCm release (image)",
            "PASS",
            f"{rec.get('version')} (gfx {rec.get('gfx') or '-'}): {rec.get('clang')}; its "
            f"LLVM tools: {rec.get('llvm_bin')}",
        )
    return Check(
        "ROCm release (image)",
        "FAIL",
        f"this image records no ROCm ({container_layout.ROCM_RECORD}): it was not built by this package's Dockerfile",
        image_fix(),
    )


def worker_running(cfg: dict) -> bool:
    """A review worker is running now: no model call of any kind, not even a presence check."""
    try:
        import json

        st = json.loads((config.state_dir() / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    from .review import pipes

    return any(
        (p.get("worker") or {}).get("id")
        and not (config.state_dir() / "jobs" / p["worker"]["id"] / "rc").exists()
        for p in pipes(st)
    )


HIP_PROBE = "#include <hip/hip_runtime.h>\n__global__ void rkmon_probe() {}\n"


def hipcc_path(llvm_bin: str) -> str:
    """hipcc: on PATH, else beside ROCm's LLVM tools (<core>/bin/hipcc for <core>/lib/llvm/bin); "" when none."""
    found = shutil.which("hipcc")
    if found:
        return found
    if llvm_bin and len(pathlib.Path(llvm_bin).parents) > 2:
        cand = pathlib.Path(llvm_bin).parents[2] / "bin" / "hipcc"
        if cand.is_file() and os.access(cand, os.X_OK):
            return str(cand)
    return ""


def hip_check(llvm_bin: str, run_=None, hipcc: str | None = None) -> Check:
    """hipcc and the HIP headers: `hipcc --offload-arch=<target> -c` of `#include <hip/hip_runtime.h>` and a kernel,
    once for each of build_plan.HIP_CHECK_TARGETS (the image's build checks the same). Without them rocKE's tests
    that compile HIP skip here (the checks' pytest plugin turns a missing header into a skip), so a review's coverage
    drops with no failure to show it: a WARN, with the fix."""
    import tempfile

    from . import image

    hipcc = hipcc_path(llvm_bin) if hipcc is None else hipcc
    why = "rocKE's tests that compile HIP (they skip in the checks, and coverage drops silently)"
    fix = Steps(
        (
            "build the image again (QUICK_BUILD_GUIDE.md; docker/install_rocm.sh installs them), then",
            *image.new_image_steps(wrapper_name()),
        )
    )
    if not hipcc:
        return Check(
            "hipcc and the HIP headers",
            "WARN",
            blocking("no hipcc (ROCm not installed here, or not on PATH)", why),
            image_fix(),
        )
    with tempfile.TemporaryDirectory() as d:
        src = pathlib.Path(d) / "hip_probe.hip"
        src.write_text(HIP_PROBE)
        for target in build_plan.HIP_CHECK_TARGETS:
            obj = pathlib.Path(d) / f"hip_probe-{target}.o"
            rc, text = (run_ or run)(
                [hipcc, f"--offload-arch={target}", "-c", "-o", str(obj), str(src)],
                timeout=120,
            )
            if rc != 0 or not obj.is_file():
                return Check(
                    "hipcc and the HIP headers",
                    "WARN",
                    blocking(
                        f"{hipcc} cannot compile <hip/hip_runtime.h> for {target} (exit {rc}): "
                        f"{text.strip()[-160:]}",
                        why,
                    ),
                    fix,
                )
    return Check(
        "hipcc and the HIP headers",
        "PASS",
        f"{hipcc} compiles <hip/hip_runtime.h> for {', '.join(build_plan.HIP_CHECK_TARGETS)}",
    )


def venv_torch_check(venv: str, run_=None) -> Check:
    """The checks venv has no torch: CI's lanes run rocKE's tests without it, so a test that imports it must skip in
    the checks as it does in CI (a GPU PyTorch venv stays apart: requirements/README.md).
    """
    py = f"{venv}/bin/python" if venv and os.path.exists(f"{venv}/bin/python") else ""
    if not py:
        return Check("checks venv without torch", "SKIP", "no checks venv yet")
    rc, _text = (run_ or run)(
        [
            py,
            "-c",
            "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('torch') else 1)",
        ]
    )
    if rc == 1:
        return Check("checks venv without torch", "PASS", f"{py}: no torch, as in CI")
    if rc == 0:
        return Check(
            "checks venv without torch",
            "FAIL",
            blocking(
                f"{py} imports torch: CI's lanes run without it, so a test that must skip there runs here",
                "the checks' verdicts on tests that need torch (they pass here and skip in CI)",
            ),
            image_fix(),
        )
    return Check(
        "checks venv without torch",
        "WARN",
        f"{py} did not answer (exit {rc})",
        image_fix(),
    )


def python_dev_check(venv: str, run_=None) -> Check:
    """The Python development install (Python.h) of the interpreter the native build configures with (the checks
    venv's, else this one): CMake's find_package(Python3 ... Development) needs it, or the native build and ctest of
    every review never run."""
    py = f"{venv}/bin/python" if venv and os.path.exists(f"{venv}/bin/python") else ""
    py = py or sys.executable
    rc, text = (run_ or run)(
        [
            py,
            "-c",
            "import sysconfig, os; d = sysconfig.get_paths()['include']; "
            "print(d, os.path.isfile(os.path.join(d, 'Python.h')))",
        ]
    )
    inc, _, has = text.strip().rpartition(" ")
    if rc == 0 and has == "True":
        return Check("Python development headers", "PASS", f"{inc}/Python.h ({py})")
    return Check(
        "Python development headers",
        "FAIL",
        blocking(
            f"no Python.h for {py}" + (f" in {inc}" if inc else ""),
            "the native build and ctest of every review and gate (CMake cannot find Python3 Development)",
        ),
        image_fix(),
    )


def _pre(
    key: str,
    ok: bool,
    found: str,
    fix=None,
    review: bool = False,
    status: str = "",
) -> Check:
    """A prerequisite's check: its name, and what it blocks, from the table; FAIL or WARN by its level. Its fix, by
    default, is an image of this package, which carries every prerequisite (image_fix).
    """
    p = PREREQS[key]
    if not status:
        status = (
            "PASS"
            if ok
            else (
                "FAIL"
                if p.level == "core" or (p.level == "review" and review)
                else "WARN"
            )
        )
    if status in ("PASS", "SKIP"):
        return Check(p.name, status, found or "found")
    return Check(
        p.name, status, blocking(found or "missing", p.blocks), fix or image_fix()
    )


def prerequisites(cfg: dict | None) -> list[Check]:
    cfg = cfg or {}
    review = features.on(cfg, "review-drafts")
    out = []
    vi = sys.version_info
    rc, _ = run([sys.executable, "-c", "import venv, ensurepip"])
    out.append(
        _pre(
            "python",
            vi >= PY_MIN and rc == 0,
            f"{sys.executable} {vi.major}.{vi.minor}.{vi.micro}"
            + ("" if rc == 0 else ", venv/ensurepip missing"),
        )
    )
    rc, text = run(["git", "--version"])
    v = tuple(int(x) for x in re.findall(r"\d+", _version(text))[:2]) if rc == 0 else ()
    out.append(
        _pre(
            "git",
            bool(v) and v >= GIT_MIN,
            _version(text) if rc == 0 else "missing",
        )
    )
    g = gh_status(
        cfg
    )  # the read login: every read of the toolkit, the session's and the jobs'
    dirs = gh_mod.login_dirs(cfg or {})
    split = bool(dirs["read"]) and not dirs.get("unsplit")
    login = (gh_login_fix(gh_mod.READ if split else gh_mod.WRITE),)
    if not g["found"]:
        out.append(
            _pre(
                "gh",
                False,
                "missing",
                image_fix(),
            )
        )
    elif not g["version_ok"]:
        out.append(
            _pre(
                "gh",
                False,
                f"{g['binary']} {g['version']} (older than "
                f"{'.'.join(map(str, gh_mod.MIN_VERSION))})",
                image_fix(),
            )
        )
    else:
        me = config.operator_login(cfg)
        # The read login needs no write scope (its writes must be refused: gh_split_checks); without the split, gh's
        # own login does everything.
        need = [] if split else gh_needed_scopes(cfg)
        lacking = (
            [n for n in need if not any(s in g["scopes"] for s in n.split(" or "))]
            if g["scopes"]
            else []
        )
        ok = g["logged_in"] and (not me or g["login"] == me) and not lacking
        found = (
            f"{g['binary']} {g['version']}, "
            + (f"logged in as {g['login']}" if g["logged_in"] else "not logged in")
            + (f", scopes {', '.join(g['scopes'])}" if g["scopes"] else "")
            + f"; {'the read ' if split else ''}login kept in {g['config_dir']}"
        )
        fix = None
        if not g["logged_in"]:
            fix = Steps(login)
        elif me and g["login"] != me:
            found += f"; the configured login is {me}"
            fix = Steps(
                (
                    *login,
                    f"to log in as {me}; or",
                    *settings_parts("operator.github_login", g["login"]),
                )
            )
        elif lacking:
            found += f"; the enabled features need the scopes {', '.join(lacking)}"
            fix = Steps(
                (
                    gh_login_fix(gh_mod.READ),
                    "sets up the read login (a read-only token), which reads without those scopes",
                )
            )
        out.append(_pre("gh", ok, found, fix))
    version = claude_code(cfg)[0] if features.uses_claude(cfg) else ""
    if not features.uses_claude(cfg):  # no Claude session: Claude Code is not needed
        out.append(
            _pre(
                "claude",
                True,
                "not needed: no Claude session runs (features.review_automation is off)",
                status="SKIP",
            )
        )
    else:
        # the image's Claude Code: the claude login's row checks the key (reach.claude_login)
        out.append(_pre("claude", bool(version), version))
    w = config.worker(cfg)
    local = config.is_local_url(w["url"]) or not w["url"]
    users = worker_users(cfg) if cfg else ["the reviewer"]
    if not users:
        out.append(
            _pre(
                "ollama",
                True,
                "not needed: nothing uses the local review loop (reviewer "
                f"{bindings_mod().reviewer_choice(cfg)})",
                status="SKIP",
            )
        )
    elif w["backend"] == "ollama" and local:
        rc, text = run(["ollama", "--version"])
        out.append(
            _pre(
                "ollama",
                rc == 0,
                _version(text) if rc == 0 else "missing",
                Steps(
                    (
                        "no Ollama runs in the container: point the reviewer model at one it reaches,",
                        *settings_parts(
                            "worker.url",
                            Described("the address of that Ollama"),
                            key=worker_url_key(w),
                        ),
                    )
                ),
                review,
            )
        )
    else:
        out.append(
            _pre(
                "ollama",
                True,
                f"not needed: the reviewer model is at {config.safe_url(w['url'])}",
                status="SKIP",
            )
        )
    out.append(reviewer_check(cfg, w))
    out += role_checks(cfg, w) + first_pass_checks(cfg)
    rocminfo = next(
        (w for w in map(shutil.which, PREREQS["gpu"].look.split()) if w), ""
    )
    roots = (
        sorted(str(p) for p in pathlib.Path("/opt").glob("rocm*"))
        if pathlib.Path("/opt").is_dir()
        else []
    )
    out.append(
        _pre(
            "gpu",
            bool(rocminfo),
            (
                f"{rocminfo}; installs: {', '.join(roots) or 'none'}"
                if rocminfo
                else f"missing (ROCm installs: {', '.join(roots) or 'none'})"
            ),
        )
    )
    llvm = paths.resolve(cfg)["rocm_llvm_bin"]
    clang = pathlib.Path(llvm) / "clang" if llvm else None
    if clang and clang.exists():
        rc, text = run([str(clang), "--version"])
        out.append(_pre("llvm", True, f"{llvm}: clang {_version(text)}"))
    else:
        # The image's own (derived from its ROCm record): a missing or broken one is the image's.
        out.append(
            _pre(
                "llvm",
                False,
                (
                    f"no clang in {llvm} (paths.rocm_llvm_bin)"
                    if llvm
                    else "missing: the image records none (paths.rocm_llvm_bin is blank)"
                ),
                image_lacks_fix(LLVM_TOOLS),
                review,
            )
        )
    rec = rocm_image()
    if rec or deployment_image():
        out.append(image_rocm_check(rec))
    out.append(clang_parity(cfg))
    venv = paths.resolve(cfg)["venv"]
    for tool in ("cmake", "ninja"):
        found = shutil.which(tool) or (
            f"{venv}/bin/{tool}"
            if venv and os.path.exists(f"{venv}/bin/{tool}")
            else ""
        )
        out.append(
            _pre(
                tool,
                bool(found),
                found or "missing",
                image_fix(),
                review,
            )
        )
    cxx = next((c for c in ("c++", "g++", "clang++") if shutil.which(c)), "")
    rc, text = run([cxx, "--version"]) if cxx else (127, "")
    out.append(
        _pre(
            "cxx",
            bool(cxx),
            f"{cxx} {_version(text)}" if cxx else "missing",
            review=review,
        )
    )
    out.append(python_dev_check(venv))
    out.append(hip_check(llvm))
    out.append(venv_torch_check(venv))
    for tool in ("rsync", "ssh", "curl"):
        found = shutil.which(tool)
        out.append(_pre(tool, bool(found), found or "missing"))
    where = paths.resolve(cfg)["working_dir"] or str(pathlib.Path.home())
    probe = pathlib.Path(where)
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    free = shutil.disk_usage(probe).free / 2**30
    roomy = free >= DISK_WARN_GB
    out.append(
        Check(
            "free disk",
            "PASS" if roomy else "WARN" if free >= DISK_FAIL_GB else "FAIL",
            blocking(
                f"{free:.0f} GiB free at {probe}"
                + (
                    ""
                    if roomy
                    else f"; the clone, worktrees and builds need tens of GiB ({DISK_WARN_GB} GiB is comfortable)"
                ),
                "" if roomy else "the clone, the worktrees and the builds",
            ),
            (
                None
                if roomy
                else Steps(
                    (
                        "the working volume is on the disk the host keeps Docker's volumes on: free room there;",
                        Command((wrapper_name(), "shell")),
                        f"and du -h -d 1 {probe} show what takes the room in it",
                    )
                )
            ),
        )
    )
    return out


def bindings_mod():
    from . import bindings

    return bindings


def worker_users(cfg: dict) -> list[str]:
    """What uses the local worker (config `worker`): the reviewer ("worker"), the tick triage (models.tick resolving
    to it), the planner or the analyst (models.analytics and the rest, on the worker's endpoint).
    """
    from . import roles

    b = bindings_mod()
    out = []
    if b.reviewer_choice(cfg) == b.WORKER and not b.no_draft(cfg):
        out.append("the reviewer")
    if config.model_for(cfg, "tick")["by"] == "worker":
        out.append("the tick triage")
    out += [f"the {role}" for role in roles.ROLES if roles.wanted(cfg, role)]
    return out


def reviewer_check(cfg: dict, w: dict | None = None) -> Check:
    """The chosen reviewer (config `reviewer`) and nothing else: the default has nothing to check; a binding's address,
    model and key (no request: reach.gateways reads its endpoint), each fixed as review drafts fixes what it lacks
    (features.needs_fix); the local worker's model_check.
    """
    b = bindings_mod()
    r = b.reviewer(cfg)
    if r["kind"] == b.DEFAULT or b.no_draft(cfg):
        return Check(
            "reviewer",
            "PASS",
            f"default: the verifier ({b.verifier_name(cfg)}) reviews alone; nothing to "
            "check",
        )
    if r["kind"] == "missing":
        from . import deploy_tree

        return Check(
            "reviewer",
            "FAIL",
            f"{deploy_tree.name('reviewer')} is {r['choice']}, which model_bindings does not define",
            section("reviewer"),
        )
    if r["kind"] == "binding":
        from . import triage

        gap = triage.binding_gap(b.client_settings(cfg, "drafter"))
        return Check(
            "reviewer",
            "FAIL" if gap else "PASS",
            blocking(gap, "the first-pass drafts") if gap else b.reviewer_line(cfg),
            (features.needs_fix([gap], "review-drafts") if gap else None),
        )
    return dataclasses.replace(
        model_check(cfg, w), name="reviewer (the local review loop)"
    )


# The HTTP status in a probe's words, as reviewer_client.explain_status gives it ("... got HTTP 401 ...").
PROBE_STATUS = re.compile(r"\bHTTP (\d{3})\b")


def not_served_fix(w: dict, environ=os.environ) -> Steps:
    """The fix of the local worker's model W["model"] that its server does not serve, as the operator types it on the
    host: the server's side, the model pulled or served there, then `<wrapper> stop --drain`, then `<wrapper> start`,
    whose rk provision probes it again; or the binding's, a model the server serves, then `<wrapper> confirm` and
    `<wrapper> apply`. The doctor's reviewer row gives it when the server's model list does not name the model
    (model_check), and so does step 7's row when the start's probe found so (probe_fix: models.NOT_LISTED).
    """
    from . import steps

    return Steps(
        (
            f"pull or serve {w['model']} there, then",
            *steps.restart_parts(wrapper_name(environ)),
            "; or",
            *settings_parts("worker.model", Described("a model it serves"), environ),
        )
    )


def probe_fix(cfg: dict, w: dict, environ=os.environ) -> Steps:
    """The fix of the local worker's model W that its last probe refused (config.probe_failure reads that record, W's
    `probe`), by what refused it, as the operator types it on the host. reviewer_client.explain_status says what each
    HTTP status means.

    - HTTP 401 or 403, the key or the header that carries it: the key set again by its secret's command, then
      `<wrapper> stop --drain`, then `<wrapper> start`, since a start probes again after a failed probe and an apply
      with only a secret changed has nothing to deliver; or the binding's header setting in your deploy settings file,
      and how the key is sent while it goes as a bearer token alone (auth_mode "bearer", the binding's default, which
      sends no other header), then `<wrapper> confirm` and `<wrapper> apply`, which deliver them. With no secret named,
      the binding's key secret and that side's settings, then confirm and apply.
    - a server error (HTTP 5xx): the server put right, then the stop and the start.
    - a failed tool-call probe: the binding's model, one that can call tools, then confirm and apply.
    - a model the server does not list (models.NOT_LISTED): not_served_fix, the model served there, then the stop and
      the start, or a model it serves in the binding, then confirm and apply.
    - any other refusal of the test chat (a route the server does not find): the binding's edit, then confirm and
      apply.
    """
    from . import deploy_tree, models
    from . import secrets_registry as sr
    from . import steps

    rec = w.get("probe") if isinstance(w.get("probe"), dict) else {}
    chat = str(rec.get("chat") or "")
    m = PROBE_STATUS.search(chat if chat != "ok" else str(rec.get("tools") or ""))
    status = int(m.group(1)) if m else 0
    restart = steps.restart_parts(wrapper_name(environ))
    if status in (401, 403):
        # A key sent as a bearer token alone goes in no other header: the header's side sets how it is sent too.
        sent_as = ()
        if str(w.get("auth_mode") or "").lower() == "bearer":
            sent_as = (
                ", and",
                settings_edit("worker.auth_mode", "header"),
                '(or "header_and_bearer" to send it as a bearer token as well)',
            )
        header = (
            settings_edit(
                "worker.auth_header", Described("the header the endpoint reads")
            ),
            *sent_as,
            ", then",
            *apply_commands(environ),
        )
        name = str(w.get("api_key_secret") or "").strip()
        entry = sr.entry(cfg, name) if name else None
        if entry is None:
            secret = settings_edit(
                "worker.api_key_secret", Described("the secret the key is kept in")
            )
            return Steps((secret, ", and", *header))
        key = sr.set_fix(entry, environ)
        key_parts = key.parts if isinstance(key, Steps) else (key,)
        return Steps(
            (
                "set the key again:",
                *key_parts,
                ", then",
                *restart,
                "; or the header that carries it:",
                *header,
            )
        )
    if 500 <= status <= 599:
        return Steps(
            ("put the model server right (its log says what failed), then", *restart)
        )
    if chat == "ok":
        return Steps(
            (
                settings_edit("worker.model", Described("a model that can call tools")),
                ", then",
                *apply_commands(environ),
            )
        )
    if chat.startswith(models.NOT_LISTED):
        return not_served_fix(w, environ)
    return Steps(
        (
            f"edit model_bindings.{deploy_tree.WORKER} in {deploy_tree.FILE}, then",
            *apply_commands(environ),
        )
    )


def model_check(cfg: dict, w: dict | None = None) -> Check:
    """The reviewer model: read-only (the endpoint's model list), never while a review worker runs; a model that
    failed a start's test chat or tool-call probe (rk provision) is named, its fix what puts that failure right
    (probe_fix).
    """
    w = w or config.worker(cfg)
    if not w["url"] or not w["model"]:
        return Check(
            "reviewer model",
            "WARN",
            f"{worker_url_key(w)} or model_id is blank (needed while the first pass is the local review loop)",
            section("model"),
        )
    if worker_running(cfg):
        return Check(
            "reviewer model", "SKIP", "a review worker is running: not checked now"
        )
    sys.path.insert(0, str(paths.PACKAGE / "worker"))
    import reviewer_client

    p = _presence(reviewer_client, w)
    if not p["answers"]:
        from . import steps

        return Check(
            "reviewer model",
            "FAIL",
            f"{w['backend']} at {config.safe_url(w['url'])}: no answer ({p['error']})",
            Steps(
                (
                    "start the model server, then",
                    *steps.restart_parts(wrapper_name()),
                    "; or",
                    *settings_parts(
                        "worker.url",
                        Described("the model server's address"),
                        key=worker_url_key(w),
                    ),
                )
            ),
        )
    if not p["model_found"]:
        return Check(
            "reviewer model",
            "FAIL",
            f"{config.safe_url(w['url'])} answers but has no {w['model']}",
            not_served_fix(w),
        )
    why = config.probe_failure(w)
    if why:
        return Check(
            "reviewer model",
            "FAIL",
            f"{w['backend']} at {config.safe_url(w['url'])}: {w['model']} is there, but it "
            f"{why}",
            probe_fix(cfg, w),
        )
    return Check(
        "reviewer model",
        "PASS",
        f"{w['backend']} at {config.safe_url(w['url'])}: {w['model']} present",
    )


def role_checks(cfg: dict, w: dict | None = None) -> list[Check]:
    """The optional model roles beside the reviewer (rkmon_lib/roles.py): OFF, or their model on the reviewer's
    endpoint, read-only (its model list), never while a review worker runs. A role blocks nothing: what it lacks is a
    WARN, and the toolkit's own result stands meanwhile."""
    from . import roles

    w = w or config.worker(cfg)
    out = []
    for role in roles.ROLES:
        want, name, line = (
            roles.wanted(cfg, role),
            f"{role} model",
            roles.describe(cfg, role),
        )
        if not want:
            out.append(Check(name, "OFF", line))
        elif not w["url"]:
            out.append(
                Check(
                    name,
                    "WARN",
                    f"{want['model']}, but the reviewer has no endpoint",
                    section("model"),
                )
            )
        elif worker_running(cfg):
            out.append(
                Check(
                    name, "SKIP", f"{line}: a review worker is running: not checked now"
                )
            )
        else:
            sys.path.insert(0, str(paths.PACKAGE / "worker"))
            import reviewer_client

            p = _presence(reviewer_client, dict(w, model=want["model"]))
            ok = p["answers"] and p["model_found"]
            out.append(
                Check(
                    name,
                    "PASS" if ok else "WARN",
                    line
                    + (
                        ""
                        if ok
                        else (
                            "; not listed there"
                            if p["answers"]
                            else f"; no answer ({p['error']})"
                        )
                        + "; the toolkit's own result stands meanwhile"
                    ),
                    (
                        None
                        if ok
                        else Steps(
                            (
                                "choose a model the endpoint lists:",
                                *section_parts("model"),
                            )
                        )
                    ),
                )
            )
    return out


def first_pass_checks(cfg: dict) -> list[Check]:
    """The first passes (models.ci_triage, quality_draft, notice_draft: rkmon_lib/first_pass.py): OFF at the default,
    PASS with the model at hand, WARN when the model it names is not (its owner works without it meanwhile).
    """
    from . import first_pass

    out = []
    for role in first_pass.ROLES:
        line = first_pass.describe(cfg, role)
        if not first_pass.bound(cfg, role):
            out.append(Check(f"{role} first pass", "OFF", line))
            continue
        gap = first_pass.gap(cfg, role)
        out.append(
            Check(
                f"{role} first pass",
                "WARN" if gap else "PASS",
                blocking(
                    line,
                    (
                        f"its leads: {first_pass.OWNER[role]} works without them meanwhile"
                        if gap
                        else ""
                    ),
                ),
                section("model") if gap else None,
            )
        )
    return out


def _presence(reviewer_client, w: dict) -> dict:
    """The model list only (no prompt): does it answer, does it have the model."""
    out = {"answers": False, "model_found": False, "error": ""}
    try:
        if w["backend"] == "ollama":
            status, data = reviewer_client._request(w, "GET", "/api/tags", None, 15)
            names = (
                [m.get("name") or m.get("model") for m in (data.get("models") or [])]
                if isinstance(data, dict)
                else []
            )
        else:  # worker.models_url, else the places the start tries
            status, entries, _where = reviewer_client.list_models(w, 15)
            names = [m["id"] for m in entries]
        out["answers"] = status == 200
        out["model_found"] = w["model"] in names or f"{w['model']}:latest" in names
        out["error"] = "" if out["answers"] else f"HTTP {status}"
    except (
        Exception
    ) as exc:  # noqa: BLE001 - an endpoint that cannot be read is reported, never raised
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def secrets_checks(cfg: dict) -> list[Check]:
    """The secrets folder's rows: the folder, then each of its files by the private-file rules; none while the
    configuration names no secrets folder (the container's layout names its own volume: rkmon_lib/derived.py).
    """
    p = paths.resolve(cfg)
    if not p["secrets_dir"]:
        return []
    d = pathlib.Path(p["secrets_dir"])
    if not d.exists():
        return [
            Check(
                "secrets folder",
                "WARN",
                f"{d} does not exist yet",
                Command(("mkdir", "-m", "700", str(d))),
            )
        ]
    mode = d.stat().st_mode & 0o777
    fix = Command(("chmod", "700", str(d))) if mode & 0o077 else None
    out = [
        Check(
            "secrets folder",
            "FAIL" if fix else "PASS",
            f"{d} permissions {mode:o}",
            fix,
        )
    ]
    for f in sorted(x for x in d.iterdir() if x.is_symlink() or not x.is_dir()):
        out += private_files.problems(f, paths.PACKAGE, f"secret {f.name}")
    return out


def repository_defaults_checks(cfg: dict) -> list[Check]:
    """The repository defaults the configuration runs with, one row: PASS naming those that apply; WARN
    naming the keys that take the package's generic values instead, those the configuration gives no value of its own;
    PASS when it gives each its own. No row while the repository or its folder is blank.
    """
    repo, folder = cfg.get("repo"), paths.repository_folder(cfg)
    if not (repo and folder):
        return []
    name = "repository defaults"
    found = repository_defaults.find(repo, folder)
    if found is not None:
        where = found.path.relative_to(paths.PACKAGE)
        return [Check(name, "PASS", f"{where}, for {repo} and its folder {folder}")]
    generic = repository_defaults.generic(repo, folder, cfg)
    if not generic:
        return [
            Check(
                name,
                "PASS",
                f"none for {repo} and its folder {folder}, and the configuration gives each of their keys a value",
            )
        ]
    shows = rk_command("config", "show", "--source")
    return [
        Check(
            name,
            "WARN",
            repository_defaults.generic_found(repo, folder, generic),
            Steps(
                (
                    shows,
                    "shows each value the monitor runs with and where it comes from",
                )
            ),
        )
    ]


def configuration_checks(cfg: dict | None, why: str, fix=None) -> list[Check]:
    """The configuration's rows: CFG None is one that cannot be read, WHY saying why and FIX the fix its error
    names (config.ConfigError.fix: each error of the configuration's load names one)."""
    if cfg is None:
        return [Check("configuration", "FAIL", why, fix)]
    problems = config.problems(cfg)
    out = [
        Check(
            "configuration",
            "FAIL" if problems else "PASS",
            why,
            cannot_run_fix() if problems else None,
        )
    ]
    out += [Check("configuration", "FAIL", p, cannot_run_fix()) for p in problems]
    out += repository_defaults_checks(cfg)
    p = paths.resolve(cfg)
    for key in (
        "working_dir",
        "outputs_dir",
    ):  # the container's layout: its work volume
        if p[key]:
            there = os.path.isdir(p[key])
            out.append(
                Check(
                    f"paths.{key}",
                    "PASS" if there else "WARN",
                    p[key] + ("" if there else " (does not exist yet)"),
                    None if there else Steps((*section_parts("deploy"), "makes it")),
                )
            )
    if p["repo"]:
        from . import deploy

        bad = deploy.clone_check(cfg)
        out.append(
            Check(
                "the monitor's clone",
                "WARN" if bad else "PASS",
                bad or p["repo"],
                clone_fix(cfg) if bad else None,
            )
        )
    if p["worker_tools"]:
        from . import deploy

        # Content, not presence: the jobs run these copies, so one that is not the running code's is stale.
        stale = deploy.stale_worker_tools(cfg, paths.PACKAGE)
        missing = [n for n, why in stale if why == "missing"]
        differ = [n for n, why in stale if why == "differs"]
        found = p["worker_tools"]
        if missing:
            found += f": missing {', '.join(missing[:4])}"
        if differ:
            found += f"{';' if missing else ':'} not the running code's: {', '.join(differ[:4])}"
            found += f" and {len(differ) - 4} more" if len(differ) > 4 else ""
        out.append(
            Check(
                "worker tools",
                "WARN" if stale else "PASS",
                found,
                rk_command("worker-tools", "--install") if stale else None,
            )
        )
    if p["review_book"]:
        ok = os.path.exists(p["review_book"])
        out.append(
            Check(
                "review book",
                "PASS" if ok else "WARN",
                p["review_book"] + ("" if ok else " (not written yet)"),
                None if ok else Steps((*section_parts("agentic"), "writes it")),
            )
        )
    return out


def standby_checks(home=None, environ=os.environ) -> list[Check]:
    """While the monitor is in standby (features.standby): its WARN row, `standby: on since TIME (REASON):
    nothing is sent`, TIME as messages show a time and REASON why, from the internal state of the monitor's home HOME
    (config.home() by default; `standby.since`, `standby.reason`: each left out while blank), with what ends it
    (states.standby_fix), as the operator types it here; no row while standby is off. The admin dashboard's top line
    shows the same row."""
    from . import progress, records, state_store, states, steps

    if not features.standby(home):
        return []
    try:
        state = (
            state_store.read(config.home() if home is None else home).get("standby")
            or {}
        )
    except state_store.StateError:
        state = {}
    since = str(state.get("since") or "")
    when = records.parsed(since)
    words = "on" + (
        f" since {progress.shown_time(when) if when else since}" if since else ""
    )
    if state.get("reason"):
        words += f" ({state['reason']})"
    fix = states.standby_fix(
        lambda command: Command((wrapper_name(environ), *steps.typed(command)))
    )
    return [Check("standby", "WARN", f"{words}: nothing is sent", fix)]


def feature_checks(cfg: dict) -> list[Check]:
    """Each feature's switches; while the monitor is in standby, its row (standby_checks); then each capability: OFF
    with the setting that keeps it off, PASS, or FAIL with what it lacks and its fix (features.fix).
    """
    out = [Check("features", "PASS", features.headline(cfg)), *standby_checks()]
    for name, c in features.CAPABILITIES.items():
        if not features.on(cfg, name):
            out.append(
                Check(
                    features.lowered(c.title),
                    "OFF",
                    f"off: {features.why_off(cfg, name)}",
                )
            )
            continue
        gone = features.missing(cfg, name)
        v = features.value(cfg, name)
        out.append(
            Check(
                features.lowered(c.title),
                "FAIL" if gone else "PASS",
                ("on" if v is True else str(v))
                + (f", but needs {'; '.join(gone)}" if gone else ""),
                features.fix(cfg, name) if gone else None,
            )
        )
    return out


def container_checks(cfg: dict) -> list[Check]:
    """In the container: the orchestrator's Claude session (what it lacks; its model and effort) and its tick triage;
    which code runs and the code volume's check; the PR runners' slots wherever PR code runs in them
    (prcode.doctor_checks); and the atlassian CLI anywhere it is set up."""
    from . import atlassian, control, orchestrator, prcode, steps, triage

    out = []
    if deployment_image() and not features.uses_claude(cfg):
        out.append(
            Check(
                "Claude session (rk orchestrate)",
                "SKIP",
                "not needed: review automation is off, so no Claude session runs; each tick is run by fixed rules "
                "(rk rules-tick), and what needs judgement waits for a person (rk status)",
            )
        )
    elif deployment_image():
        # The session's key is the claude login's row (rkmon_lib/reach.py claude_login), with its test call. Each other
        # part names its own fix; the row's is the start, whose set-up installs the skill into the workspace.
        gone = orchestrator.check(cfg, with_key=False)
        out.append(
            Check(
                "Claude session (rk orchestrate)",
                "WARN" if gone else "PASS",
                (
                    blocking("; ".join(gone), SESSION_BLOCKS)
                    if gone
                    else f"{orchestrator.describe(cfg)}; model "
                    f"{config.orchestrator_model(cfg)}, effort {config.role_effort(cfg, 'orchestrator')[0]}"
                ),
                section("agentic") if gone else None,
            )
        )
        gap = orchestrator.settings_gap(cfg)
        writes = "rk orchestrate writes them, into a file that reads as a JSON object, as the session starts"
        # With no session running on them, its start writes them: there is nothing to start again.
        if gap and not control.session_pids(cfg):
            out.append(
                Check(
                    "Claude session settings",
                    "SKIP",
                    f"no Claude session runs here, and {writes} ({gap})",
                )
            )
        else:
            out.append(
                Check(
                    "Claude session settings",
                    "WARN" if gap else "PASS",
                    (
                        blocking(
                            gap,
                            "the session's auto-compaction (claude.auto_compact_window) and prompt-cache "
                            "lifetimes (claude.prompt_cache_ttl, claude.subagent_prompt_cache_ttl)",
                        )
                        if gap
                        else f"{orchestrator.settings_file(cfg)}: "
                        + ", ".join(
                            f"{k} {v}"
                            for k, v in orchestrator.session_settings(cfg).items()
                        )
                    ),
                    (
                        Steps((f"{writes}:", *steps.restart_parts(wrapper_name())))
                        if gap
                        else None
                    ),
                )
            )
        out += session_login_checks(cfg)
        why = triage.unavailable(cfg)
        named = config.model_for(cfg, "tick")["by"] in ("worker", "binding")
        out.append(
            Check(
                "tick triage",
                "WARN" if named and why else "PASS",
                f"{triage.describe(cfg)} reads each tick first"
                + (
                    f"; {blocking(why, 'the triage model: the rule filter decides meanwhile')}"
                    if named and why
                    else ""
                ),
                (
                    Steps(("give the triage model what it lacks in", *settings_file()))
                    if named and why
                    else None
                ),
            )
        )
    if deployment_image():
        out += [code_check(cfg), code_volume_check()]
    out += prcode.doctor_checks()
    out.append(atlassian.check_line(cfg))
    out += ticket_rows(cfg)
    return out


def ticket_rows(cfg: dict) -> list[Check]:
    """Check 6 with its read (ticket_fields.checked): the atlassian login's reads of JIRA's create metadata, through
    the writer in the container (atlassian's _run). A fix runs the check again as a setting's change runs it:
    `<wrapper> confirm` and `<wrapper> apply`."""
    from . import atlassian, ticket_fields as tf

    def run(words: list) -> tuple:
        return atlassian._run([atlassian.BINARY, *words], cfg, timeout=60)

    login = lambda: atlassian.status(cfg)["logged_in"]  # noqa: E731
    again = apply_commands()
    return tf.checked(cfg, deploy_tree.FILE, run, login, wrapper_name(), again=again)[0]


def session_login_checks(cfg: dict, run=None) -> list[Check]:
    """gh as the orchestrator's session runs it (its environment: orchestrator.session_env): the login it runs with,
    named (gh.logged_in_as), or the verifier's and the lane's own calls fail; a PASS only with a login. The session runs
    no atlassian CLI of its own: its JIRA calls are `rk atlassian`, and the atlassian CLI's row (atlassian.check_line)
    checks the login they use.
    """
    import shutil
    import subprocess
    from . import gh, orchestrator

    run = run or subprocess.run
    try:
        env = dict(os.environ, **orchestrator.session_env(cfg))
    except (ValueError, OSError) as exc:
        return [
            Check(
                "session login",
                "WARN",
                f"the session's environment: {exc}",
                cannot_run_fix(),
            )
        ]
    name = "session login: gh"
    cmd = [gh.settings(cfg)[0], "auth", "status"]
    where = f"GH_CONFIG_DIR {env.get('GH_CONFIG_DIR') or 'its default folder'}"
    fix = Steps((gh_login_fix(gh.READ), "(the read login the session runs with)"))
    blocks = "the verifier's and the code quality pass's GitHub reads"
    if not (shutil.which(cmd[0], path=env.get("PATH")) or os.path.isfile(cmd[0])):
        return [
            Check(
                name,
                "WARN",
                blocking(f"{cmd[0]} is not on the session's PATH", blocks),
                fix,
            )
        ]
    try:
        p = run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
            stdin=subprocess.DEVNULL,
        )
        rc, said = p.returncode, f"{p.stdout or ''}\n{p.stderr or ''}"
    except (OSError, subprocess.TimeoutExpired):
        rc, said = 1, ""
    login = gh.logged_in_as(said)[1]
    if rc == 0 and login:
        return [Check(name, "PASS", f"{login}, as the session runs it ({where})")]
    if rc == 0:
        return [
            Check(
                name,
                "WARN",
                f"gh answered, but named no login, as the session runs it ({where})",
                Steps((check_again(), "checks again")),
            )
        ]
    return [
        Check(
            name,
            "FAIL",
            blocking(f"not logged in as the session runs it ({where})", blocks),
            fix,
        )
    ]


def code_check(cfg: dict) -> Check:
    """Which code runs: the image's own, or the release root installed over it (rkmon_lib/releases.py), with the
    host's command that goes back to the code before it."""
    rel = releases.active_release(releases.code_dir())
    if rel is None:
        return Check(
            "code",
            "PASS",
            f"the image's own ({releases.image_id() or 'no digest recorded'})",
        )
    info = code_volume.release_info(rel)
    return Check(
        "code",
        "PASS",
        f"release {rel.name} over the image (version {info.get('version') or '?'}, installed {info.get('at')}); to "
        f"go back to the code before it: {wrapper()} rollback",
    )


def code_volume_check(environ=os.environ, lstat=os.lstat) -> Check:
    """The code volume's check, as the entry point's root step makes it at each start (rk_root.py Volume.problems,
    through releases.root_steps), what a change of the code leaves while it runs aside: PASS while the volume is root's
    and read-only all the way; else FAIL, which keeps the monitor from starting, with the fix the start's FAIL line
    gives (rk_root.py VOLUME_FIX). LSTAT is os.lstat (the tests give their own)."""
    root = releases.root_steps()
    folder = releases.code_dir(environ)
    volume = root.Volume(
        folder=str(folder), image=str(releases.IMAGE_CODE), lstat=lstat
    )
    bad = volume.problems(leftovers=False)
    if not bad:
        return Check(
            "code volume", "PASS", f"{folder} is root's and read-only all the way"
        )
    return Check(
        "code volume",
        "FAIL",
        blocking(
            f"{folder} is not root's and read-only all the way: {code_volume.listed(bad)}",
            "the monitor's start",
        ),
        Steps(
            (
                Command((wrapper_name(environ), "apply")),
                ", on the host,",
                root.VOLUME_FIX,
            )
        ),
    )


def _answers(host: str, port: int) -> bool:
    """Something listens at host:port (a TCP connection opens within a second)."""
    import socket

    try:
        with socket.create_connection((host, port), timeout=1):
            return True
    except OSError:
        return False


def _service_running() -> bool | None:
    """The admin dashboard's service holds state/web.lock while it runs (tools/rkweb.py reads it the same way);
    None when there is no lock file yet."""
    import fcntl

    path = config.state_dir() / "web.lock"
    if not path.exists():
        return None
    with open(path, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(f, fcntl.LOCK_UN)
        return False


def _port_check(
    name: str, host: str, port: int, running: bool | None, answers=_answers
) -> Check:
    probe = (
        container_layout.LOOPBACK
        if host in (container_layout.EVERY_ADDRESS, "")
        else host
    )
    up = answers(probe, port)
    if running:
        return Check(
            name,
            "PASS" if up else "WARN",
            f"{host}:{port}: "
            + ("answering" if up else "the service runs, but nothing answers there"),
            (
                None
                if up
                else Steps(
                    (
                        Command(("tools/rkweb.py", "stop")),
                        ", then",
                        Command(("tools/rkweb.py",)),
                        "(state/web.log says why)",
                    )
                )
            ),
        )
    if up:
        return Check(
            name,
            "WARN",
            blocking(
                f"{host}:{port}: held by another program, and the monitor service is not running",
                "the service: it cannot listen there",
            ),
            Steps(
                (
                    "free that port, or choose another: edit features.dashboards in",
                    *settings_file(),
                )
            ),
        )
    return Check(
        name,
        "PASS",
        f"{host}:{port}: free (the service is not running: tools/rkweb.py starts it)",
    )


def webhost_key_checks(cfg: dict) -> list[Check]:
    """The team dashboard pushed to the web host: the state records the web host's own key, the only key its ssh
    trusts (rkmon_lib/host_keys.py), or every push fails host key verification (ssh never asks). The row's fix is the
    command that shows the host's key and records it once you accept it (host_keys.record_fix). Only while a web host
    is named and its key is needed (secrets_registry.needed_by: the team dashboard published, and live, since in
    prototype nothing is pushed).
    """
    from . import host_keys, publish, secrets_registry as sr

    if not publish.configured(cfg) or not sr.needed_by(
        cfg, sr.entry(cfg, "web_host_ssh_key")
    ):
        return []
    name = "team dashboard's web host key"
    host = host_keys.target(cfg, host_keys.WEB)[0]
    keys = host_keys.recorded(cfg, host_keys.WEB)
    if keys:
        return [
            Check(
                name,
                "PASS",
                f"{host}: its own key is recorded ({len(keys)} line(s)), the only key ssh trusts it by",
            )
        ]
    return [
        Check(
            name,
            "FAIL",
            blocking(
                f"{host}: its own key is not recorded",
                "the team dashboard: ssh trusts the host by no other key, so every push fails host key verification",
            ),
            host_keys.record_fix(),
        )
    ]


def request_service_checks(
    cfg: dict, environ=os.environ, answering=None
) -> list[Check]:
    """The team dashboard published, live, with its priority box on: the request service on the web host
    (tools/webhost_server.py) answers at the team dashboard's address, asked from here, in the container from the
    container (check 7; web_host_install.answering, or ANSWERING in its place). Its fix installs the service
    (web_host_install.install_command), then runs these checks again. Nothing while the monitor serves the page itself,
    the box is off, or the team dashboard is in prototype, where nothing is pushed; asked each time, never cached.
    """
    from . import publish, web_host_install

    if not (
        publish.mode(cfg) == "publish"
        and features.live(cfg, "dashboards.team")
        and publish.takes_requests(cfg)
    ):
        return []
    url = str(((cfg.get("publish") or {}).get("html") or {}).get("url") or "").strip()
    if not url:
        return []
    why = (answering or web_host_install.answering)(url)
    if not why:
        return [
            Check(
                "team dashboard",
                "PASS",
                f"the request service on the web host answers at {url}",
            )
        ]
    return [
        Check(
            "team dashboard",
            "FAIL",
            f"the request service on the web host does not answer at the team dashboard's address, {url} ({why})",
            Steps(
                (
                    web_host_install.install_command(environ),
                    ", then",
                    check_again(environ),
                )
            ),
        )
    ]


def _access_refusal(cfg: dict, environ) -> tuple:
    """(why the admin dashboard's service does not start, its fix), apart, as its refusal holds them
    (webserve.access_refusal): the row gives the fix on a line of its own. (None, None) while the service may start.
    """
    from . import webserve

    refusal = webserve.access_refusal(cfg, environ)
    return (refusal.why, refusal.fix) if refusal else (None, None)


def web_checks(
    cfg: dict, environ=os.environ, answers=_answers, answering=None
) -> list[Check]:
    """The dashboards: where the admin page is reachable, and its access token, which every request carries; its port
    (answering, free, or held by another program); the team dashboard served here and its port; the request service on
    the team dashboard's web host (request_service_checks, ANSWERING its question), the admin dashboard on or off; in
    the container, whether docker/.env publishes what the settings name.
    """
    from . import webserve

    requests = request_service_checks(cfg, environ, answering)
    if not features.on(cfg, "admin-dashboard"):
        return [
            Check(
                "admin dashboard",
                "SKIP",
                "not needed: features.dashboards.admin is off, so no web server runs",
            ),
            *requests,
        ]
    where = webserve.exposure(cfg, environ)
    why, fix = _access_refusal(cfg, environ)
    found = (
        webserve.home_url(cfg, environ)
        + "; "
        + (
            f"reachable beyond this machine on {where}"
            if where
            else "this machine only (the loopback)"
        )
        + f", access token {'set' if webserve.access_token(cfg) else 'missing'}"
    )
    if why:
        found += "; " + blocking(
            why, "the admin dashboard's service: it does not start"
        )
    out = [Check("admin dashboard", "FAIL" if why else "PASS", found, fix)]
    running = _service_running()
    bind, _ = webserve.where(cfg, environ=environ)
    out.append(
        _port_check(
            "admin dashboard's port",
            bind,
            webserve.listen_port(cfg, environ),
            running,
            answers,
        )
    )
    if webserve.team_serving(cfg):
        tb, tp = webserve.team_where(cfg, environ)
        c = _port_check("team dashboard's port (served here)", tb, tp, running, answers)
        out.append(
            dataclasses.replace(
                c,
                found=f"{c.found}; {webserve.team_reach(cfg, environ)}",
            )
        )
    out += webhost_key_checks(cfg) + requests
    if deployment_image(environ):
        bad = webserve.publish_mismatch(cfg, environ)
        lines = " ".join(f"{k}={v}" for k, v, _d in webserve.publish_env(cfg))
        out.append(
            Check(
                "dashboards published (docker/.env)",
                "WARN" if bad else "PASS",
                ("; ".join(bad) + f"; the settings want: {lines}" if bad else lines),
                (
                    Steps(
                        (
                            "put the same values in your settings file, then",
                            *apply_commands(environ),
                        )
                    )
                    if bad
                    else None
                ),
            )
        )
    return out


def _split_found(r: dict, side: str = "") -> str:
    """One side's probe in words: its login and what it may do (ghlogins.may, on SIDE), or that its folder holds no
    login; never a token."""
    from . import ghlogins

    why = r.get("why") or ""
    if r.get("login"):
        said = f"{r['login']} in {r['dir']}: {ghlogins.may(r, side)}"
    elif r.get("can_write") is None:
        said = f"not known in {r['dir']}"
    else:
        said = f"no login in {r['dir']}"
        why = "" if why.startswith("no gh login") else why
    return (
        said
        + (f" ({why})" if why else "")
        + (" through the broker" if r.get("via") else "")
    )


def gh_split_checks(cfg: dict, check=None) -> list[Check]:
    """The GitHub read/write split, probed now on both sides (gh.probe: a comment with no body and a branch on a commit
    that does not exist, each refused either way, so nothing is made; ghlogins.py): the read login (and the session's,
    when it is another folder) must be refused both (403); the write login is judged by need (ghlogins.write_lacks): it
    must be able to comment, and to push while something that is on turns auto-merge on or off, and one that may
    comment but not push says what that leaves. A side with no login is never a PASS: the read side's and the session's
    reads, and the write side's writes when something that is on writes, need one. In the container the write login is
    probed through the broker, as the write user. `check`: ghlogins.check, for a test.
    """
    from . import ghlogins

    if not ghlogins.folders(cfg):
        one = gh_mod.login_dirs(cfg).get("unsplit")
        return [
            Check(
                "GitHub read/write split",
                "WARN",
                (
                    f"none yet: the login in {one} reads and writes (no read login)"
                    if one
                    else "none: this configuration reads and writes with gh's own login"
                ),
                Steps(
                    (
                        gh_login_fix(gh_mod.READ),
                        "sets up the read login (a read-only token)",
                    )
                ),
            )
        ]
    try:
        c, _ = (check or ghlogins.check)(cfg, None, True)
    except Exception as exc:  # noqa: BLE001 - a doctor line, never a traceback
        return [
            Check(
                "GitHub read/write split",
                "WARN",
                f"not checked: {exc}",
                Steps((rk_command("gh-logins"), "checks it once GitHub answers")),
            )
        ]
    out = []
    logins = {side: gh_login_fix(side) for side in gh_mod.BOTH}
    may = " and ".join(ghlogins.ABILITY[k] for k in ghlogins.write_needs(cfg))
    fixes = {
        "read": Steps((logins["read"], "with a read-only token (the repo read only)")),
        "session": Steps(
            (
                logins["read"],
                "with a read-only token (the session and its agents read with the read login)",
            )
        ),
        "write": Steps((logins["write"], f"with a token that may {may}")),
    }
    needs = {  # what a side with no login blocks, and the write side while its login may not comment
        "read": "every GitHub read",
        "session": "the session's GitHub reads",
        "write": "every GitHub write (rk post and the rest)",
    }
    from . import secrets_registry as sr

    writers = sr.needed_by(cfg, sr.entry(cfg, "github_write"))
    for side, r in c["sides"].items():
        name = f"GitHub {side} login"
        if side == "write" and not r.get("login") and not writers:
            out.append(
                Check(
                    name,
                    "OFF",
                    "none: nothing that is on writes to GitHub (the write login is needed once review automation or the "
                    f"merge queue goes live: {logins['write']})",
                )
            )
            continue
        found = _split_found(r, side) + (
            "; the session's gh runs with it"
            if side == "read" and "session" not in c["sides"] and r.get("login")
            else ""
        )
        if r.get("can_write") is not None and not r.get("login"):
            out.append(
                Check(
                    name,
                    "FAIL",
                    blocking(found, needs[side]),
                    fixes[side],
                )
            )
            continue
        if side == "write":
            refused, unknown = ghlogins.write_lacks(cfg, r)
        else:  # the read and the session login must make neither write
            refused = ["can_write"] if r.get("can_write") else []
            unknown = ["can_write"] if r.get("can_write") is None else []
        if refused:
            # A write login that may comment but not push says what that leaves in its own words (ghlogins.may).
            said = (
                found
                if refused == ["can_push"]
                else blocking(
                    found,
                    (
                        needs[side]
                        if side == "write"
                        else "the read/write split: GitHub writes are not limited to rk's guarded commands"
                    ),
                )
            )
            out.append(Check(name, "FAIL", said, fixes[side]))
        elif unknown:
            out.append(
                Check(
                    name,
                    "WARN",
                    found,
                    Steps((rk_command("gh-logins"), "checks it once GitHub answers")),
                )
            )
        else:
            out.append(Check(name, "PASS", found))
    return out


def state_checks(state: pathlib.Path) -> list[Check]:
    """The monitor's own files: a drain in force (WARN) or one whose process is gone or past its time (FAIL: it
    held new work back until the next tick clears it), a torn last ledger line, and an incidents file that does not
    read (each recovered by the next write, said here first)."""
    from . import store as store_mod

    out = []
    try:
        st = (
            json.loads((state / "state.json").read_text())
            if (state / "state.json").is_file()
            else {}
        )
    except (OSError, ValueError) as exc:
        return [
            Check(
                "monitor state",
                "FAIL",
                f"state.json does not read: {exc}",
                Steps(
                    (
                        Command(
                            ("python3", "-m", "json.tool", str(state / "state.json"))
                        ),
                        "shows where it stops reading",
                    )
                ),
            )
        ]
    d = store_mod.draining(st)
    if d:
        stale = store_mod.drain_stale(d)
        what = f"{d.get('why') or 'a drain'} since {d.get('since')}"
        out.append(
            Check(
                "drain",
                "FAIL" if stale else "WARN",
                (
                    f"{what}: {stale}; the next tick clears it"
                    if stale
                    else f"DRAINING: {what}: nothing new starts"
                ),
                (
                    Steps(
                        (
                            rk_command("tick"),
                            "clears it now (or",
                            rk_command("start"),
                            ")",
                        )
                    )
                    if stale
                    else Steps(
                        (
                            rk_command("start"),
                            "ends it now; else it ends once the work in flight finishes",
                        )
                    )
                ),
            )
        )
    ledger = state / "actions.jsonl"
    try:
        tail = ledger.read_bytes()[-65536:] if ledger.is_file() else b""
    except OSError:
        tail = b""
    if tail and not tail.endswith(b"\n"):
        out.append(
            Check(
                "ledger",
                "WARN",
                "the last line of actions.jsonl is torn (a write cut short): it is skipped, and the next record "
                "moves it to actions.torn",
                Steps((rk_command("tick"), "writes the next record")),
            )
        )
    inc = state / "incidents.json"
    if inc.is_file():
        try:
            json.loads(inc.read_text())
        except (OSError, ValueError):
            out.append(
                Check(
                    "incidents",
                    "WARN",
                    "incidents.json does not read (a write cut short): the next tick recovers it from "
                    "incidents.json.prev",
                    Steps((rk_command("tick"), "recovers it now")),
                )
            )
    return out


def model_guard_check(cfg: dict, state: pathlib.Path) -> Check:
    """The model guard (spend.guard): WARN while a call of the last spend.GUARD_DAYS days ran on another model family
    than its pass asked for, as the spend card's last refresh found (spend.read_card); OFF while spend is off, SKIP
    until a refresh has read the calls."""
    from . import spend

    name = "model guard"
    if not spend.on(cfg):
        return Check(
            name,
            "OFF",
            "off: the calls are read while spend is on (prices in the configuration)",
        )
    card = spend.read_card(state)
    if not card or "off_family" not in card:
        return Check(
            name,
            "SKIP",
            f"not read yet: the tick's next refresh of the spend card reads the calls ({rk_command('spend')} "
            "reads them now)",
        )
    if not card["off_family"]:
        return Check(
            name,
            "PASS",
            f"every call of the last {spend.GUARD_DAYS} days ran on the model family asked for "
            f"(read {card.get('at')})",
        )
    return Check(
        name,
        "WARN",
        f"{spend.guard_text(card['off_family'])} (read {card.get('at')})",
        Steps(
            (
                rk_command("spend"),
                "names them; then have the gateway answer each model id with the model it names",
            )
        ),
    )


def all_checks(
    cfg: dict | None,
    why: str,
    fresh: bool = False,
    notes: list[str] | None = None,
    fix=None,
) -> list[Check]:
    """Every check: the prerequisites, the configuration (`configuration_checks`, with WHY and FIX), the secrets, the
    features, the dashboards, the container; then the live, read-only checks of each configured service
    (rkmon_lib/reach.py: cached a few minutes unless `fresh`; `notes` gets the line that says which) and the
    RAM disk."""
    out = prerequisites(cfg) + configuration_checks(cfg, why, fix)
    if cfg is not None:
        from . import offload

        out += (
            secrets_checks(cfg)
            + feature_checks(cfg)
            + web_checks(cfg)
            + offload.doctor_checks(cfg)
            + container_checks(cfg)
        )
        from . import reach

        state = config.state_dir()
        remote, said = reach.remote_checks(
            cfg, state if state.is_dir() else None, fresh=fresh
        )
        out += remote + reach.ramdisk_check(cfg, state if state.is_dir() else None)
        out += state_checks(state) + [model_guard_check(cfg, state)]
        if cfg.get("repo"):
            gh_mod.configure(cfg)  # the gh binary the probes run
            out += gh_split_checks(cfg)
        if notes is not None:
            notes.append(said)
    return out


def report(checks: list[Check]) -> str:
    """The rows as checks.render prints them, then the count of each status."""
    n = {s: sum(1 for c in checks if c.status == s) for s in Status}
    return (
        render(checks)
        + f"\n\n{n[Status.PASS]} pass, {n[Status.FAIL]} fail, {n[Status.WARN]} warn, {n[Status.OFF]} off, "
        f"{n[Status.SKIP]} skipped\n"
    )


def configured_values(cfg: dict) -> str:
    """Every configured value, as the setup summary's Configuration section lists them (rk features), secrets only as
    set or missing with their path."""
    from . import control, summary

    lines = summary.configuration(
        cfg,
        claude_code(cfg),
        gh_status(cfg),
        control.state_summary(cfg),
    )
    return (
        "\n".join(["# rk doctor (this machine only: never published)", "", *lines])
        + "\n"
    )


def main(fresh: bool = False, fails: bool = False, as_json: bool = False) -> int:
    """`rk doctor [--fresh] [--fails] [--json]`: the report; with `fails`, only the names of the checks that FAIL, one
    per line; with `as_json`, only the rows, as one JSON list on one line (checks.check_to_json), which rkmon apply
    reads for each failure's fix. The exit code is the report's (exit_code) in each form.
    """
    where, how = config.home_and_why()
    fix = None
    try:
        cfg, why = config.load(check=False), f"{where / 'config.json'} ({how})"
    except (
        config.ConfigError
    ) as exc:  # no tick runs while it cannot be read: the row says how to fix it
        cfg, why, fix = None, str(exc), exc.fix(wrapper_name())
    notes: list[str] = []
    checks = all_checks(cfg, why, fresh=fresh, notes=notes, fix=fix)
    if as_json:
        print(json.dumps([check_to_json(c) for c in checks]))
        return exit_code(checks)
    if fails:
        for c in checks:
            if c.status == "FAIL":
                print(c.name)
        return exit_code(checks)
    if cfg is not None:
        from . import roles

        print(configured_values(cfg))
        print("## Roles: model and effort (rk roles)\n")
        print("\n".join(roles.table_lines(cfg)) + "\n")
        print("## Checks\n")
    print(report(checks), end="")
    for line in notes:
        print(line)
    return exit_code(checks)


def exit_code(checks: list[Check]) -> int:
    """0 nothing fails; 3 a prerequisite fails (a row of prerequisites.txt); 4 something else fails (the
    configuration, its folders, secrets)."""
    if any(c.status == "FAIL" and c.name in PREREQ_NAMES for c in checks):
        return EXIT_MISSING
    return EXIT_CONFIG if any(c.status == "FAIL" for c in checks) else EXIT_OK
