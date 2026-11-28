# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The derived keys: one function per key, which computes the key's value at each load. Nobody sets a derived key;
the registry (rkmon_lib/registry.py) describes each one.

`Layout(container, given, image_venv, image_llvm_bin)`: what the functions read besides the keys before them.
CONTAINER: the monitor runs in the package's container, whose fixed layout is rkmon_lib/container_layout.py's. GIVEN: a
configuration file's own values for derived keys, by dotted key: config.json's, where a key it sets keeps that value;
the delivered deploy layer gives none. IMAGE_VENV: the image's checks venv when the image has one, else "".
IMAGE_LLVM_BIN: the folder of ROCm's LLVM tools that the image's ROCm record names (container_layout.rocm_llvm_bin),
else "".

`FUNCTIONS`: dotted key -> f(values, layout) -> the key's value, in the order runtime_config.resolve computes them,
each after the keys it reads. VALUES holds every key computed so far, by dotted key.

- The paths: `paths.working_dir` is the given folder (a configuration from before the package names it
  `paths.root`), else the container's /work; `claude_workspace` is the given folder, else the working folder's
  claude-workspace in the container; every other path is rkmon_lib/paths.py's `resolve` over the given paths, with the
  container's secrets folder and the image's venv where none is given. In the container, `paths.rocm_llvm_bin` is the
  image's LLVM tools wherever the image records them: a given folder is not kept, as the image's ROCm is the only one
  there.
- `ramdisk.path`: the given folder, else the container's RAM disk folder.
- `weekly_report.dir`: the resolved weekly reports' folder (`paths.weekly_dir`).
- `followup_scan.dir`: the given folder, else the follow-ups folder in the monitor's state folder (config.FOLLOWUPS in
  config.state_dir of `paths.toolkit`, the monitor's home).
- `codeowner_org`: the given organization, else the owner part of `repo`.
- `clone_url`: the given address (a mirror's, or a local path), else GitHub's address of `repo` (`clone_address`).
- `profile_dir`: the given folder, else the folder of the repository defaults for `repo` and `rocke_path`
  (rkmon_lib/repository_defaults.py), with the repository's profile for worker/drafter.py (drafter.json and its rules,
  drafter_rules.md); blank where none apply.
- The passes' keys, each by its reader's rule over the given value, after every other key: `worker.tag`
  (bindings.worker_tag: the given tag, else the leading letters of the reviewer model's id); `draft_stage`
  (draft_stage.choice: none with no first-pass drafter, else the given stage, where "auto" or a blank one is tool-loop
  with a drafter binding, else local-worker); `verifier_name` (bindings.verifier_name: the given name, else the audit
  model in words); `review_first_line_no_draft` (bindings.no_draft_line: the verifier's model at the verifier's effort);
  `review_first_line`, last (bindings.drafter_line: the drafter's). A given line 1 is not kept: line 1 gives this
  deployment's automated reviewer name (rkmon_lib/reviewer_name.py), by which the monitor knows its own reviews.
- `models.draft`: the given choice, a blank one included (nobody drafts), else the local worker (bindings.WORKER).
- The two dashboards' ports and addresses: the given ones, else the container's ports (container_layout) and, in the
  container, its listen address; elsewhere a blank address, which the reader takes as the loopback (webserve.HOST).
- `publish.html.dest`: the given page, else the page the site folder gives (`site_page`).
- `gpu.known_hosts`, `publish.html.known_hosts`: the known_hosts files of the GPU login host and of the web host
  in the working folder (`known_hosts_path`), which ssh's callers write the host's own keys the state records to
  (host_keys.known_hosts_file); a given file is not kept, as ssh trusts those keys alone.
- The places of the secrets and of gh's logins, by the secrets registry's rules (rkmon_lib/secrets_registry.py), ""
  while no secrets folder is known: `gpu.ssh_key`, `publish.html.key`, `admin_dashboard.access_token_file`,
  `teams.webhook_file`, `gh.read_config_dir` and `gh.write_config_dir` (`SECRET_PLACES`), each the own place of the
  entry whose config key it is, in its keeper's secrets folder (secrets_registry.home, by the layout alone: in the
  container, a writer's entry is in the writer's volume); then `worker.api_key_file`, the file of the secret
  `worker.api_key_secret` names (secrets_registry.binding_key_file, which reads the place of a secret with a key from
  the key), "" while it names none. A given place is not kept: the secrets are the monitor's to place, and an older
  host's place is none in the container. The readers of these keys read them as the load leaves them, and keep no rule
  of their own; config.normalize fills them in, by these functions, in a configuration read without the load.

GitHub's address of the repository REPO, <org>/<repo>: `github_address(repo)`, GITHUB/<org>/<repo>, "" for a blank
REPO. `clone_address(given, repo)`: where the monitor clones REPO from, GIVEN (a mirror's address or a local path,
file://...), else GitHub's address of REPO. The monitor's clone (deploy.clone_url), the GPU nodes' clone (gpu), the
repository `rkmon package update` fetches from (package_update.fetch) and update.repo (deploy_tree's rule) take them
from here.

The known_hosts file of a host ssh reaches: `known_hosts_path(working_dir, name)`, NAME in the working folder
WORKING_DIR, "" for a blank one: GPU_KNOWN_HOSTS, the GPU login host's, which the key gpu.known_hosts names, and
WEB_HOST_KNOWN_HOSTS, the web host's, which publish.html.known_hosts names. host_keys.known_hosts_file writes each.

The team dashboard's places on the web host, each from the site folder SITE (publish.html.site), relative to the login's
home, "" for a blank SITE: `site_page(site)`, the page, PAGE in SITE; `site_requests(site)`, other teams' priority
requests file, which holds the requesters' names and reasons: at the top of the login's home, never under SITE's first
folder, which a web server may publish whole (a per-user public_html), and named SITE with each `/` as FOLDER_SIGN, a
sign no plain name holds, so two sites never share one, then REQUESTS_SUFFIX. The deploy settings' rule for
publish.html.requests (rkmon_lib/deploy_tree.py) and the writer's check of the requests file it reads
(publish.target_refusal) take it from here.
"""

from __future__ import annotations

import os
from typing import NamedTuple

from . import container_layout

GITHUB = "https://github.com"  # GitHub's address: the repository <org>/<repo> is GITHUB/<org>/<repo>
PAGE = "index.html"  # the team dashboard's page in the site folder
REQUESTS_SUFFIX = (
    "-requests.jsonl"  # the requests file's name: the site folder's, then this
)
FOLDER_SIGN = "+"  # each `/` of the site folder in the requests file's name: a sign no plain name holds
GPU_KNOWN_HOSTS = "gpu_known_hosts"  # in the working folder: the GPU login host's keys the state records, for ssh
WEB_HOST_KNOWN_HOSTS = "web_host_known_hosts"  # in the working folder: the web host's keys the state records, for ssh


class Layout(NamedTuple):
    container: bool = False
    given: dict = {}
    image_venv: str = ""
    image_llvm_bin: str = ""


def _given(layout: Layout, key: str):
    """The given value of KEY, None when it is blank or not given."""
    v = (layout.given or {}).get(key)
    return None if v in (None, "") else v


def ramdisk_path(values: dict, layout: Layout) -> str:
    return _given(layout, "ramdisk.path") or container_layout.RAMDISK


def working_dir(values: dict, layout: Layout) -> str:
    from . import paths

    given = _given(layout, "paths.working_dir") or _given(layout, "paths.root")
    if given:
        return paths.resolve(_paths_view(values, layout))["working_dir"]
    return container_layout.WORK_DIR if layout.container else ""


def claude_workspace(values: dict, layout: Layout) -> str:
    given = _given(layout, "claude_workspace")
    if given:
        return given
    wd = values.get("paths.working_dir") or ""
    return f"{wd}/claude-workspace" if layout.container and wd else ""


def _paths_view(values: dict, layout: Layout) -> dict:
    """The configuration rkmon_lib/paths.py resolves the paths from: the given paths, the container's and the image's
    own where none is given, the image's LLVM tools wherever it records them, and the keys before them.
    """
    given = layout.given or {}
    p = {
        k.split(".", 1)[1]: v
        for k, v in given.items()
        if k.startswith("paths.") and v not in (None, "")
    }
    if layout.container:
        if not (p.get("working_dir") or p.get("root")):
            p["working_dir"] = container_layout.WORK_DIR
        p.setdefault("secrets_dir", container_layout.SECRETS_DIR)
        if layout.image_venv:
            p.setdefault("venv", layout.image_venv)
        if layout.image_llvm_bin:
            p["rocm_llvm_bin"] = layout.image_llvm_bin
    return {
        "paths": p,
        "ramdisk": {
            "path": values.get("ramdisk.path") or "",
            "enabled": values.get("ramdisk.enabled") is True,
        },
        "weekly_report": {"dir": _given(layout, "weekly_report.dir") or ""},
        "claude_workspace": values.get("claude_workspace") or "",
    }


def _path(name: str):
    """The function of paths.NAME: rkmon_lib/paths.py's `resolve` over the paths' view."""

    def f(values: dict, layout: Layout) -> str:
        from . import paths

        return paths.resolve(_paths_view(values, layout))[name]

    f.__name__ = f"path_{name}"
    return f


def weekly_report_dir(values: dict, layout: Layout) -> str:
    return values.get("paths.weekly_dir") or ""


def followup_scan_dir(values: dict, layout: Layout) -> str:
    from . import config

    given, toolkit = _given(layout, "followup_scan.dir"), values.get("paths.toolkit")
    if given or not toolkit:
        return given or ""
    return str(config.state_dir(toolkit) / config.FOLLOWUPS)


def codeowner_org(values: dict, layout: Layout) -> str:
    given = _given(layout, "codeowner_org")
    if given:
        return given
    repo = str(values.get("repo") or "")
    return repo.split("/", 1)[0] if "/" in repo else ""


def github_address(repo) -> str:
    """GitHub's address of the repository REPO: see the module's docstring."""
    repo = str(repo or "").strip()
    return f"{GITHUB}/{repo}" if repo else ""


def clone_address(given, repo) -> str:
    """Where the monitor clones the repository REPO from: see the module's docstring."""
    return str(given or "").strip() or github_address(repo)


def clone_url(values: dict, layout: Layout) -> str:
    return clone_address(_given(layout, "clone_url"), values.get("repo"))


def profile_dir(values: dict, layout: Layout) -> str:
    from . import paths, repository_defaults

    given = _given(layout, "profile_dir")
    if given:
        return given
    found = repository_defaults.find(
        values.get("repo"), paths.repository_folder(values)
    )
    return str(found.path.parent) if found else ""


def models_draft(values: dict, layout: Layout):
    from . import bindings

    given = layout.given or {}
    return given["models.draft"] if "models.draft" in given else bindings.WORKER


def admin_port(values: dict, layout: Layout):
    return _given(layout, "admin_dashboard.port") or container_layout.ADMIN_PORT


def team_port(values: dict, layout: Layout):
    return _given(layout, "team_dashboard.port") or container_layout.TEAM_PORT


def _bind(key: str):
    def f(values: dict, layout: Layout) -> str:
        given = _given(layout, key)
        if given:
            return given
        return container_layout.LISTEN_ADDRESS if layout.container else ""

    f.__name__ = f"bind_{key.split('.', 1)[0]}"
    return f


def _site(site) -> str:
    return str(site or "").strip().rstrip("/")


def site_page(site) -> str:
    """The team dashboard's page on the web host, from the site folder SITE: see the module's docstring."""
    site = _site(site)
    return f"{site}/{PAGE}" if site else ""


def site_requests(site) -> str:
    """Other teams' priority requests file on the web host, from the site folder SITE: see the module's docstring."""
    site = _site(site)
    return f"{site.replace('/', FOLDER_SIGN)}{REQUESTS_SUFFIX}" if site else ""


def publish_dest(values: dict, layout: Layout) -> str:
    return _given(layout, "publish.html.dest") or site_page(
        values.get("publish.html.site")
    )


def known_hosts_path(working_dir: str, name: str) -> str:
    """The known_hosts file of a host ssh reaches: see the module's docstring."""
    return os.path.join(working_dir, name) if working_dir else ""


def gpu_known_hosts(values: dict, layout: Layout) -> str:
    return known_hosts_path(values.get("paths.working_dir") or "", GPU_KNOWN_HOSTS)


def web_host_known_hosts(values: dict, layout: Layout) -> str:
    return known_hosts_path(values.get("paths.working_dir") or "", WEB_HOST_KNOWN_HOSTS)


def _environ(layout: Layout) -> dict:
    """The environment the secrets registry places an entry in: the container's own (docker/Dockerfile sets it, and
    doctor.deployment_image reads it) when LAYOUT is the container's, else none, whatever environment the process that
    loads the configuration has."""
    return {"RKMON_CONTAINER": "1"} if layout.container else {}


def _secret_place(key: str):
    """The function of KEY, the config key of a secret's or a login's entry (secrets_registry: the entry whose `key` it
    is): the entry's own place (secrets_registry.home), "" while no secrets folder is known.
    """

    def f(values: dict, layout: Layout) -> str:
        from . import paths, registry, secrets_registry

        cfg = registry.nest(values)
        entry = next(e for e in secrets_registry.entries(cfg) if e.key == key)
        try:
            return secrets_registry.home(cfg, entry, _environ(layout))
        except paths.PathsError:
            return ""

    f.__name__ = f"place_{key.replace('.', '_')}"
    return f


def worker_api_key_file(values: dict, layout: Layout) -> str:
    from . import registry, secrets_registry

    name = str(values.get("worker.api_key_secret") or "").strip()
    return secrets_registry.binding_key_file(registry.nest(values), name)


def _cfg(values: dict, layout: Layout, key: str) -> dict:
    """The configuration a reader's rule reads for KEY: the keys before it, nested, and KEY's given value, else ""."""
    from . import registry

    cfg = registry.nest(values)
    registry.put(cfg, key, _given(layout, key) or "")
    return cfg


def worker_tag(values: dict, layout: Layout) -> str:
    from . import bindings

    return bindings.worker_tag(_cfg(values, layout, "worker.tag"))


def draft_stage_name(values: dict, layout: Layout) -> str:
    from . import draft_stage as stages

    return stages.choice(_cfg(values, layout, "draft_stage"))


def verifier_name(values: dict, layout: Layout) -> str:
    from . import bindings

    return bindings.verifier_name(_cfg(values, layout, "verifier_name"))


def review_first_line_no_draft(values: dict, layout: Layout) -> str:
    from . import bindings

    return bindings.no_draft_line(_cfg(values, layout, "review_first_line_no_draft"))


def review_first_line(values: dict, layout: Layout) -> str:
    from . import bindings

    return bindings.drafter_line(_cfg(values, layout, "review_first_line"))


# The paths rkmon_lib/paths.py resolves after the working folder, in its order.
PATHS = (
    "root",
    "toolkit",
    "repo",
    "workers",
    "tmp",
    "probes",
    "lessons",
    "cache",
    "scratch",
    "worker_tools",
    "venv",
    "build",
    "agentic",
    "secrets_dir",
    "run",
    "ramdisk",
    "outputs_dir",
    "dashboard",
    "html",
    "daily_dir",
    "insights_dir",
    "posted_dir",
    "spend_dir",
    "weekly_dir",
    "posts",
    "site_lessons",
    "rocm_llvm_bin",
    "review_book",
    "package",
    "gh",
    "verifier_team_file",
    "claude_projects",
    "cmake",
    "pyyaml",
    "torch_python",
)
# The keys that name a secret's or a login's place: each is its entry's own place.
SECRET_PLACES = (
    "gpu.ssh_key",
    "publish.html.key",
    "admin_dashboard.access_token_file",
    "teams.webhook_file",
    "gh.read_config_dir",
    "gh.write_config_dir",
)

FUNCTIONS = {
    "ramdisk.path": ramdisk_path,
    "paths.working_dir": working_dir,
    "claude_workspace": claude_workspace,
    **{f"paths.{name}": _path(name) for name in PATHS},
    "weekly_report.dir": weekly_report_dir,
    "followup_scan.dir": followup_scan_dir,
    "codeowner_org": codeowner_org,
    "clone_url": clone_url,
    "profile_dir": profile_dir,
    "models.draft": models_draft,
    "admin_dashboard.port": admin_port,
    "admin_dashboard.bind": _bind("admin_dashboard.bind"),
    "team_dashboard.port": team_port,
    "team_dashboard.bind": _bind("team_dashboard.bind"),
    "publish.html.dest": publish_dest,
    "gpu.known_hosts": gpu_known_hosts,
    "publish.html.known_hosts": web_host_known_hosts,
    **{key: _secret_place(key) for key in SECRET_PLACES},
    "worker.api_key_file": worker_api_key_file,  # after them: the secret it names may be one of theirs
    "worker.tag": worker_tag,
    "draft_stage": draft_stage_name,
    "verifier_name": verifier_name,
    "review_first_line_no_draft": review_first_line_no_draft,
    "review_first_line": review_first_line,
}
