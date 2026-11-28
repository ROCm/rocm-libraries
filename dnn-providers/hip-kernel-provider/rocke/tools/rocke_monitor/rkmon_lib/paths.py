# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Every path the monitor uses, resolved from config.json in one place (`resolve`).

The keys live under `paths` in config.json; the registry (rkmon_lib/registry.py) describes each one.
`paths.working_dir` is the one folder the monitor owns, and a blank key means its default:
- under the working folder: toolkit (monitor/: the monitor's home, with its `rk` command, config.json and state/),
  repo, workers, tmp, probes, lessons, cache, scratch, worker_tools (tools/), venv, build, agentic (the agentic files
  filled in with this machine's settings), secrets_dir (.config/: mode 700, files 600) and run (the shared rate
  budget of the reviews that draft at once, and the registry of the processes the monitor started);
- the files written for the operator go to `paths.outputs_dir` (default: the working folder): the dashboard
  (rocke_merge_queue.md), the page (rocke_monitor.html), rocke_daily/, rocke_weekly/ (`weekly_report.dir`), the
  insights reports, the copies of posted reviews and the review spend files (rocke_spend/, `paths.spend_dir`);
- review_book, the local review loop's rubric: the filled-in copy under agentic/;
- posts: where `rk post --premises` keeps each verified post's copy (default <scratch>/post);
- site_lessons: the site lessons, one site's lessons in the skill's lessons.md format, which the skill's
  lessons_for.py reads: SITE_LESSONS in the monitor's state folder (STATE in `toolkit`, the monitor's home), so an
  update keeps them;
- verifier_team_file: the verifier's team member file `rk review-split` reads (default: the agentic install's copy,
  VERIFIER_FILE in <claude_workspace>/.claude/), and the code quality reviewer's (QUALITY_FILE) beside it;
- claude_projects, cmake, pyyaml: the Claude transcripts folder review spend reads, and the tools the checks' test runs
  get (each blank by default: transcripts.claude_projects, PATH, the venv);
- rocm_llvm_bin, ROCm's LLVM tools: no default; in the container, the image's (rkmon_lib/derived.py);
- ramdisk: the tmpfs folder of `ramdisk.path` while `ramdisk.enabled`, else ""; `tmp` defaults to <ramdisk>/tmp then
  (rkmon_lib/ramdisk.py: what else goes on it, and what never does);
- package: the toolkit's code (rk, rkmon_lib, tools/), wherever it runs from: read-only, never written at run time.

A configuration written before the package has `paths.root` and no `paths.working_dir`. Its working folder is
`paths.root`, its outputs folder is the folder of `paths.dashboard`, and the keys it lacks take the values the code
used before (LEGACY), so every path resolves as it did then.

Generated job scripts start with `shell_vars`: short shell variables for these paths. `repository_folder` is the
folder of the repository the monitor reviews (config rocke_path, the deploy settings' repository.folder), the one
function every reader asks for it; a check's --rocke-path asks it when none is given (prtree.folder_argument).
"""

import importlib
import os
import pathlib
import shlex
import sys

PACKAGE = pathlib.Path(__file__).resolve().parent.parent  # the code that is running


def worker_module(name: str):
    """The worker's module `name` (worker/<name>.py), imported when first asked. A module that docker/rkmon's deploy
    path imports reads worker/ only this way: that path runs from a package folder that holds no worker/
    (tests/test_deploy_path_folder.py)."""
    folder = str(PACKAGE / "worker")
    if folder not in sys.path:
        sys.path.insert(0, folder)
    return importlib.import_module(name)


def is_package(folder) -> bool:
    """Whether `folder` holds the monitor's code, a package: its rkmon.py and its rkmon_lib/."""
    f = pathlib.Path(folder)
    return (f / "rkmon.py").is_file() and (f / "rkmon_lib").is_dir()


# The monitor's state folder in its home (config.state_dir), and the site lessons' file in it (`resolve`).
STATE, SITE_LESSONS = "state", "site/lessons.md"
# The verifier's team member file's path in an agentic folder (the package's agentic/, a team's skills folder, the
# agentic install's .claude/), with forward slashes: verifier_file.FILE. Its one home is here, not in
# rkmon_lib/verifier_file.py: the image runs this file by its path (docker/rk_layout.py, rkmon_lib/gh_cap.py), where it
# can load nothing else of the package.
VERIFIER_FILE = "team_members/rocke_review_verifier.md"
# The code quality reviewer's team member file, in the same folder (a review in parts quotes its checks: split.py).
QUALITY_FILE = "team_members/rocke_code_quality_reviewer.md"
# The folders a developer's copy of the package makes at its top, which no package includes: .gitignore leaves them
# out (tools/generated.py), the image's build context lets in none of them (image.BUILD_CONTEXT), the package build
# and an install of the worker tools copy none of them (package_files.SKIP_PARTS), and the scan for stray private files
# does not look into them in a package folder in use (deployment.stray_private_files).
COPY_OWN_FOLDERS = (STATE, "dist", "sandbox")
# What a developer's copy of the package leaves wherever it runs, which no package includes: Python's caches and
# bytecode, and the copies a write keeps of a file it replaces (<name>.bak-<stamp>). Each is written as .gitignore
# takes it (a folder's name with a trailing /). .gitignore and .dockerignore leave each out wherever it is
# (tools/generated.py), the package build and an install of the worker tools copy none of them (package_files), and
# the scan for stray private files never looks into the folders among them (deployment.STRAY_SKIP).
COPY_OWN_LEFTOVERS = ("__pycache__/", "*.pyc", ".pytest_cache/", "*.bak-*")


# key -> its folder under the working folder
UNDER_WORKING = {
    "toolkit": "monitor",
    "repo": "repo",
    "workers": "workers",
    "tmp": "tmp",
    "probes": "probes",
    "lessons": "lessons",
    "cache": "cache",
    "scratch": "scratch",
    "worker_tools": "tools",
    "venv": "venv",
    "build": "build",
    "agentic": "agentic",
    "secrets_dir": ".config",
    "run": "run",
}
# The caches' folders in paths.cache, which the monitor writes and a PR's job reads: develop's
# results per develop commit, each PR's mutation sweeps, tools/lane_of.py's listings. docker/rk_layout.py makes them
# folders the PR runners read.
CACHE_DEVELOP, CACHE_MUTANTS, CACHE_LANE_OF = CACHES = ("develop", "mutants", "lane_of")
# key -> its name under the outputs folder ("" is the folder itself)
UNDER_OUTPUTS = {
    "dashboard": "rocke_merge_queue.md",
    "html": "rocke_monitor.html",
    "daily_dir": "rocke_daily",
    "insights_dir": "",
    "posted_dir": "",
    "spend_dir": "rocke_spend",
}
# What a configuration written before the package (`is_legacy`) gets for a key it lacks: the values the code used
# (the code ran from <root>/monitor, and its secrets lived in ~/.config/rkmon).
LEGACY = {
    "rocm_llvm_bin": "/opt/rocm/core-10.1/lib/llvm/bin",
    "review_book": "{outputs_dir}/review_book.md",
    "package": "{working_dir}/monitor",
    "secrets_dir": "~/.config/rkmon",
}
# The shell variable of each path in generated job scripts (`shell_vars`). Worktrees live directly in $D; $M is the
# toolkit's code (its tools/ and rk).
SHELL = (
    ("D", "working_dir"),
    ("R0", "repo"),
    ("M", "package"),
    ("T", "worker_tools"),
    ("WK", "workers"),
    ("BD", "build"),
    ("TP", "tmp"),
    ("PB", "probes"),
    ("CA", "cache"),
    ("SC", "scratch"),
)


class PathsError(ValueError):
    """A path the caller needs is not configured."""


def _clean(value) -> str:
    """A configured path: `~` expanded, no trailing slash; "" when blank."""
    if not value or not str(value).strip():
        return ""
    p = os.path.expanduser(str(value).strip())
    return p if p == "/" else p.rstrip("/")


def is_legacy(cfg: dict) -> bool:
    """A configuration from before the package: `paths.root`, no `paths.working_dir`."""
    p = cfg.get("paths") or {}
    return "working_dir" not in p and bool(p.get("root"))


def resolve(cfg: dict) -> dict:
    """Every path, absolute or "" (not configured). Idempotent: resolving a resolved `paths` gives the same."""
    p = cfg.get("paths") or {}
    legacy = is_legacy(cfg)
    wd = _clean(p.get("working_dir") or p.get("root"))
    # `root` is the working folder too: the drafter's modules read their folders under paths.root.
    out = {"working_dir": wd, "root": wd}
    for key, sub in UNDER_WORKING.items():
        out[key] = _clean(p.get(key)) or (f"{wd}/{sub}" if wd else "")
    # The RAM disk (rkmon_lib/ramdisk.py), while it is on: paths.tmp lives there unless the configuration names a
    # folder.
    r = cfg.get("ramdisk") or {}
    out["ramdisk"] = _clean(r.get("path")) if r.get("enabled") is True else ""
    if out["ramdisk"] and not _clean(p.get("tmp")):
        out["tmp"] = f"{out['ramdisk']}/tmp"
    dashboard = _clean(p.get("dashboard"))
    od = _clean(p.get("outputs_dir")) or (
        str(pathlib.PurePosixPath(dashboard).parent) if dashboard else wd
    )
    out["outputs_dir"] = od
    for key, name in UNDER_OUTPUTS.items():
        out[key] = _clean(p.get(key)) or (f"{od}/{name}".rstrip("/") if od else "")
    out["weekly_dir"] = _clean((cfg.get("weekly_report") or {}).get("dir")) or (
        f"{od}/rocke_weekly" if od else ""
    )
    out["posts"] = _clean(p.get("posts")) or (
        f"{out['scratch']}/post" if out["scratch"] else ""
    )
    out["site_lessons"] = _clean(p.get("site_lessons")) or (
        f"{out['toolkit']}/{STATE}/{SITE_LESSONS}" if out["toolkit"] else ""
    )
    out["rocm_llvm_bin"] = _clean(p.get("rocm_llvm_bin")) or (
        LEGACY["rocm_llvm_bin"] if legacy else ""
    )
    book = (
        LEGACY["review_book"].format(outputs_dir=od)
        if legacy
        else f"{out['agentic']}/review_book.md"
    )
    out["review_book"] = _clean(p.get("review_book")) or (book if wd else "")
    # The code's folder is not a setting; a resolved configuration (config.normalize) carries it.
    out["package"] = _clean(p.get("package")) or (
        LEGACY["package"].format(working_dir=wd) if legacy else str(PACKAGE)
    )
    out["gh"] = _clean(p.get("gh"))  # the gh binary: blank means gh on PATH
    # The verifier's team file: the agentic install's copy in the Claude workspace (agentic.refresh installs it).
    ws = _clean(cfg.get("claude_workspace"))
    out["verifier_team_file"] = _clean(p.get("verifier_team_file")) or (
        f"{ws}/.claude/{VERIFIER_FILE}" if ws else ""
    )
    for key in (
        "claude_projects",
        "cmake",
        "pyyaml",
        "torch_python",
    ):  # blank: transcripts.claude_projects, PATH, the venv, none
        out[key] = _clean(p.get(key))
    if legacy and not _clean(p.get("secrets_dir")):
        out["secrets_dir"] = _clean(LEGACY["secrets_dir"])
    return out


def get(cfg: dict, key: str) -> str:
    """One resolved path; PathsError when it is not configured (nothing is ever written relative to the current
    folder)."""
    value = resolve(cfg).get(key)
    if value is None:
        raise KeyError(key)
    if not value:
        raise PathsError(f"config.json: paths.{key} is not set (set paths.working_dir)")
    return value


def worktree(cfg: dict, name: str) -> str:
    """A worktree of the clone: `wt<N>`, `wt<N>m`, wtdev, wtbase, wtbatch, wtsanity, wtbisect, all in the working
    folder."""
    return f"{get(cfg, 'working_dir')}/{name}"


def repository_folder(cfg: dict) -> str:
    """The folder of the repository the monitor reviews, from the repository's root (config rocke_path, the deploy
    settings' repository.folder), with no slash at either end: blank when CFG names none.
    """
    return (cfg.get("rocke_path") or "").strip("/")


def sparse_dirs(cfg: dict) -> list[str]:
    """The repository folders a worktree checks out: rocKE's parent (the provider rocKE ships in) and .github."""
    k = repository_folder(cfg)
    return [k.rsplit("/", 1)[0] if "/" in k else k, ".github"] if k else [".github"]


def shell_vars(cfg: dict) -> str:
    """The first line of a generated job script: D working folder, R0 the clone, M the toolkit's code, T worker tools,
    WK workers, BD builds, TP tmp, PB probes, CA cache, SC scratch, VB the venv's bin, LB ROCm's LLVM tools.
    """
    p = resolve(cfg)
    for _var, key in SHELL:
        if not p[key]:
            raise PathsError(
                f"config.json: paths.{key} is not set (set paths.working_dir)"
            )
    pairs = [(var, p[key]) for var, key in SHELL] + [
        ("VB", f"{p['venv']}/bin"),
        ("LB", p["rocm_llvm_bin"]),
    ]
    return "; ".join(f"{var}={shlex.quote(value)}" for var, value in pairs)


def worker_tool(name: str) -> str:
    """A worker tool's path in a generated job script (after `shell_vars`): the installed copy in $T, or the package's
    own (worker/) while that install lacks the file. The container's start installs worker/ there, as
    `rk worker-tools --install` does; `rk doctor` names the files it lacks or has in another version.
    """
    return f"$([ -f $T/{name} ] && echo $T || echo $M/worker)/{name}"


def build_jobs() -> str:
    """The parallel jobs of a native build or its ctest, in a generated job script (after `shell_vars`): worker
    steps.py's count when the job runs (CPU affinity, the cgroup's CPU quota and memory, or $RKMON_JOBS), 1 when it
    cannot be read. Never the host's CPU count inside a limited container."""
    return f"$(python3 -B {worker_tool('steps.py')} jobs 2>/dev/null || echo 1)"
