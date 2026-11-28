# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The agentic files: the Claude Code skill (/rocke-monitor, its procedures, rules and lessons), the reviewer team
members and agent types, their registry rows, and the local review loop's rubric (review_book.md). They are the
package's own files, in agentic/, changed there: templates in which every site-specific value is a placeholder
(PLACEHOLDERS, the shared vocabulary) or a role, and the operator is `<user_login>`.

Two steps:
- `render`: fills the placeholders with this machine's configuration into
  <WORKING_DIR>/agentic (outside the checkout). A blank value reads "(not set up)"; no placeholder is left. A blank
  `operator.github_login` is refused: `<user_login>` is the one value the skill cannot do without.
- `install`: copies the rendered skill, team members and agent types into <CLAUDE_WORKSPACE>/.claude (a changed file is
  backed up first as <name>.bak-<stamp>; shared policies already there are kept), removes the ones an earlier install
  wrote that the render no longer has (backed up first the same way when edited since; shared policies stay), and puts
  the registry rows in its dispatch table, in place of the rocKE rows there. The skill's lessons.md is also copied,
  once, to <LESSONS_FILE> (<paths.lessons>/lessons.md): the editable copy the skill reads and the weekly review edits,
  outside .claude/, which an unattended session may not write (the container's orchestrator runs with Claude Code's
  dontAsk permissions).
`refresh` takes both steps again when the installed render is older than its inputs. Where the settings are
delivered, every render lays a team's own review skills, delivered with them
(TEAM_SKILLS, `team_skills`), over the package's files: one with a package file's path replaces it, one with a new
path is added; its line says so (`laid_over`).
Paths that start at a placeholder are written with forward slashes, which every host reads (a Linux container
included); `<CONTAINER_NOTE>` says, in a container only, how the orchestrator drives the skill.
"""

from __future__ import annotations

import datetime as dt
import filecmp
import hashlib
import json
import os
import pathlib
import re
import shutil

from . import (
    config,
    container_layout,
    deploy_tree,
    paths,
    private_files,
    reviewer_name,
)

NOT_SET = "(not set up)"
SENTINEL_BEGIN = "<!-- BEGIN GENERATED REGISTRY"
SENTINEL_END = "<!-- END GENERATED REGISTRY -->"

# The skill's folder, and the file that holds the team members' rows for the workspace's dispatch table.
SKILL = "rocke-monitor"
REGISTRY_ROWS = "team_members/dispatch_rows.md"
# The workspace's shared policies: install adds one the workspace lacks, and never replaces or removes one.
SHARED = "skills/shared/"
# In the installed skill's folder: each file the install wrote, by digest (install: one edited since is backed up
# before it is replaced or removed).
INSTALLED = ".rkmon_installed.json"


# --- the placeholders: what each one is, and its value for a configuration ----------------------------------------------


def _p(cfg, key):
    return paths.resolve(cfg).get(key) or ""


def _admin(cfg):
    from . import webserve

    return webserve.admin(cfg)


def _admin_url(cfg) -> str:
    from . import webserve

    return webserve.home_url(cfg)


def _worker(cfg, key):
    from . import config

    return config.worker(cfg).get(key) or ""


def _orchestrator(cfg, words: bool = False) -> str:
    """The orchestrator model's id, as the session sends it, or its name in words."""
    from . import config

    return config.orchestrator_name(cfg) if words else config.claude_model_id(cfg)


def _claude(cfg, role: str, words: bool = False) -> str:
    """A Claude role's model (models.audit, models.deep), by the id its agent sends, or its name in words."""
    from . import config

    return config.claude_name(cfg, role) if words else config.claude_model_id(cfg, role)


# The names the Agent tool's `model` option takes (deploy_tree.FAMILIES), in words.
FAMILY_WORDS = ", ".join(deploy_tree.FAMILIES[:-1]) + " or " + deploy_tree.FAMILIES[-1]


def _alias(cfg, role: str) -> str:
    """A Claude role's model (models.audit, models.deep) by the name the Agent tool's `model` option takes: the family
    its id holds (deploy_tree.family); blank for a model of none, which the option cannot name.
    """
    from . import config

    return deploy_tree.family(config.claude_model_id(cfg, role))


def _effort(cfg, role: str) -> str:
    """A Claude role's resolved effort (config.role_effort: effort.<role>, else claude.effort, else the model's
    default: an agent file then has no `effort:` line, fill)."""
    from . import config

    return config.role_effort(cfg, role)[0] or config.MODEL_DEFAULT


def _offload(cfg, what: str) -> str:
    """The review offload's mode, or the auditor's spot-check sample (rkmon_lib/offload.py)."""
    from . import offload

    if what == "mode":
        return offload.mode(cfg)
    s = offload.settings(cfg)
    return (
        f"{float(s['spot_check']) * 100:.0f}% of them, at least {s['spot_check_min']}"
    )


def _checks(cfg) -> int:
    from . import config

    return config.checks_limit(cfg)


def _secret(cfg, name) -> str:
    """Where the secret NAME is kept (secrets_registry.path: in the container, the writer's volume for a secret the
    writer keeps); "" when no secrets folder is known."""
    from . import paths, secrets_registry as sr

    try:
        return sr.path(cfg, sr.entry(cfg, name))
    except paths.PathsError:
        return ""


def _team_chat(cfg) -> str:
    """The words for the team chat while the develop break alerts are on; "" (not set up) while they are off. The
    webhook decides the channel, and the message label only labels each message."""
    from . import features

    return "team chat" if features.on(cfg, "chat-alerts") else ""


def _user() -> str:
    import getpass

    try:
        return getpass.getuser()
    except (OSError, KeyError):
        return os.environ.get("USER", "")


def _in_container() -> bool:
    from . import doctor

    return doctor.deployment_image()


# The skill in the container (docker/): how the orchestrator drives it, said once in the skill's package section
# (rkmon_lib/orchestrator.py: the ticks, the tick triage, the review driver).
CONTAINER_NOTE = (
    "\n- **In the container.** `rk orchestrate` keeps this session and sends `/rocke-monitor resume` when it starts; the\n"
    "  ticks and the review driver are its own (Tick procedure). The tick triage reads each tick first: by default the\n"
    "  rule filter (`rk tick-filter`, no model), which keeps the lines that need you and drops the rest; or, an opt-in, a\n"
    "  triage model (`models.tick`) that also does the routine work with read-only `rk` commands. Each file the prompt\n"
    "  names has a tick's whole output, and with a triage model its notes (what it ran, such as a REVIEW_READY's\n"
    "  `rk review-prep` folder, and what it re-checked). Skip step 1 (the tick ran), read the files for context, and\n"
    "  handle the lines given as the tick procedure says; the rest needs nothing from you, and nothing the triage did\n"
    "  went outward. `CI_AFTER_POST #N <head commit>: ...` is the triage model's: a check failed on the head we last\n"
    "  posted on; post one CI update only if the failure is the PR's (L-209). When the prompt says the triage did not\n"
    "  answer, handle the whole tick in the file, from step 2. The plain `/rocke-monitor tick` means there is no\n"
    "  triage now: run the tick yourself. Everything outward goes through `rk` commands (posts, notices, putting a PR\n"
    "  on hold, issues, auto-merge; JIRA through `rk atlassian`, as you have no atlassian login, its writes going out\n"
    "  only while follow-up tickets are on and ticket creation is live): the session's guard refuses any other\n"
    "  GitHub write, a push to a git remote, a network write by `curl` or `wget`, and SSH. A background verification\n"
    "  reports in a turn of its own when it finishes."
)


def _container_note(cfg) -> str:
    return CONTAINER_NOTE if _in_container() else ""


def _lessons_file(cfg) -> str:
    d = _p(cfg, "lessons")
    return f"{d}/lessons.md" if d else ""


def _login(cfg) -> str:
    from . import config

    return config.operator_login(cfg)


def _issues_url(c: dict) -> str:
    """The develop issues' page on GitHub: their repository's (breakage.issue_repository), "" when it is blank."""
    from . import breakage, derived

    where = breakage.issue_repository(c)
    return f"{derived.github_address(where)}/issues" if where else ""


def _rocke_url(c: dict) -> str:
    """rocKE's folder on GitHub, on develop: "" while repo is blank."""
    from . import derived

    where = derived.github_address(c.get("repo"))
    return f"{where}/tree/develop/{paths.repository_folder(c)}" if where else ""


def _atlassian_site(c: dict) -> str:
    """The site the atlassian CLI's JIRA calls name (`--site`): rkmon_lib/atlassian.py, `jira_site`."""
    from . import atlassian

    return atlassian.jira_site(c)


def _jira_fields(c: dict) -> str:
    """The fields every new ticket must carry, as JSON (field id -> value): "{}" when there are none."""
    return json.dumps(
        (c.get("jira") or {}).get("extra_fields") or {}, ensure_ascii=False
    )


def _issue_type(c: dict) -> str:
    """The issue type a new ticket gets (rkmon_lib/ticket_fields.py, `issue_type`)."""
    from . import ticket_fields

    return ticket_fields.issue_type((c.get("jira") or {}).get("issue_type"))


# The operator's GitHub login in every shipped file: the deploy settings' integrations.github_cli.username
# (operator.github_login), which `render` fills in.
LOGIN = "<user_login>"
# Why the skill is not rendered without the operator's login; who says it names the fix (orchestrator.check, rk
# provision's agentic step).
LOGIN_MISSING = "integrations.github_cli.username is blank: the skill names the operator by it (<user_login>)"

PLACEHOLDERS = {
    LOGIN: (
        "the operator's GitHub login (integrations.github_cli.username): the account the monitor posts as, and whose "
        "instructions it follows",
        _login,
    ),
    "<AUTOMATED_REVIEWER_NAME>": (
        "this deployment's automated reviewer name, which line 1 of each review and the first line of each notice "
        "give (features.review_automation.automated_reviewer_name, else container.name)",
        reviewer_name.name,
    ),
    "<WORKING_DIR>": ("the folder the monitor owns", lambda c: _p(c, "working_dir")),
    "<OUTPUTS_DIR>": ("where the operator's files go", lambda c: _p(c, "outputs_dir")),
    "<TOOLKIT_DIR>": (
        "the monitor's home: rk and state/ (on a host, config.json too)",
        lambda c: _p(c, "toolkit"),
    ),
    "<PACKAGE_DIR>": ("the package's code (read-only)", lambda c: _p(c, "package")),
    "<RK>": (
        "the monitor's rk command",
        lambda c: f"{_p(c, 'toolkit')}/rk" if _p(c, "toolkit") else "",
    ),
    "<ROCKE_REPO>": (
        "rocKE's GitHub repository, as owner/name",
        lambda c: c.get("repo") or "",
    ),
    "<ROCKE_PATH>": (
        "rocKE's folder in it",
        paths.repository_folder,
    ),
    "<ROCKE_SRC_DIR>": ("the monitor's own clone", lambda c: _p(c, "repo")),
    "<ROCKE_URL>": ("rocKE's folder on GitHub", _rocke_url),
    "<ROCKE_GITHUB_ISSUES>": ("where develop issues are filed", _issues_url),
    "<SECRETS_DIR>": (
        "where this deployment's secrets live",
        lambda c: _p(c, "secrets_dir"),
    ),
    "<REVIEW_BOOK>": (
        "the local review loop's rubric (review_book.md)",
        lambda c: _p(c, "review_book"),
    ),
    "<REVIEW_CHECKLIST>": (
        "the shared rocKE review checklist",
        lambda c: c.get("review_checklist") or "",
    ),
    "<WORKER_MODEL>": (
        "the reviewer model (the first pass)",
        lambda c: _worker(c, "model"),
    ),
    "<WORKER_URL>": ("the reviewer model's endpoint", lambda c: _worker(c, "url")),
    "<ORCHESTRATOR_MODEL>": (
        "the orchestrator model's id, as its session sends it (features.review_automation.orchestrator.model: a "
        "built-in model's name gives its id); the verifier and code quality agents run "
        "features.review_automation.passes.verification_pass.model and "
        "features.review_automation.passes.code_quality_pass.model",
        lambda c: _orchestrator(c),
    ),
    "<ORCHESTRATOR_NAME>": (
        "the orchestrator model in words, for a review's footer",
        lambda c: _orchestrator(c, True),
    ),
    "<AUDIT_MODEL>": (
        "Claude's model for the verifier, which checks the first pass's review "
        "(features.review_automation.passes.verification_pass.model)",
        lambda c: _claude(c, "audit"),
    ),
    "<AUDIT_NAME>": (
        "the audit model in words, for a review's footer",
        lambda c: _claude(c, "audit", True),
    ),
    "<AUDIT_ALIAS>": (
        "the audit model by the name the Agent tool's `model` option takes, its family "
        f"({FAMILY_WORDS}); blank for a model of none",
        lambda c: _alias(c, "audit"),
    ),
    "<DEEP_MODEL>": (
        "Claude's model for the code quality pass and the in-depth work on a big inconsistency "
        "(features.review_automation.passes.code_quality_pass.model)",
        lambda c: _claude(c, "deep"),
    ),
    "<DEEP_NAME>": (
        "the deep review's model in words, for its footer",
        lambda c: _claude(c, "deep", True),
    ),
    "<DEEP_ALIAS>": (
        "the deep model (the code quality pass's, a deep review's) by the name the Agent tool's `model` option takes, "
        f"its family ({FAMILY_WORDS}); blank for a model of none",
        lambda c: _alias(c, "deep"),
    ),
    "<CLAUDE_EFFORT>": (
        "the orchestrator session's effort (features.review_automation.orchestrator.effort)",
        lambda c: _effort(c, "orchestrator"),
    ),
    "<VERIFY_EFFORT>": (
        "the verifier agent's effort (features.review_automation.passes.verification_pass.effort)",
        lambda c: _effort(c, "verify"),
    ),
    "<QUALITY_EFFORT>": (
        "the code quality agent's effort (features.review_automation.passes.code_quality_pass.effort)",
        lambda c: _effort(c, "quality"),
    ),
    "<AUDIT_EFFORT>": (
        "the offload audit's effort (the package defaults' effort.audit, else claude.effort)",
        lambda c: _effort(c, "audit"),
    ),
    "<DEEP_EFFORT>": (
        "a deep review's effort (the package defaults' effort.deep, else claude.effort)",
        lambda c: _effort(c, "deep"),
    ),
    "<OFFLOAD_MODE>": (
        "the review offload (the package defaults' offload.mode): off, shadow or on",
        lambda c: _offload(c, "mode"),
    ),
    "<OFFLOAD_SPOT_CHECK>": (
        "the auditor's sample of the 🟡/🟢 claims (the package defaults' offload.spot_check and "
        "offload.spot_check_min)",
        lambda c: _offload(c, "spot"),
    ),
    "<CHECKS_LIMIT>": (
        "the parallel builds asked for (monitor.parallel_builds)",
        lambda c: str(_checks(c)),
    ),
    "<RUN_DIR>": (
        "the shared rate budget and the process registry",
        lambda c: _p(c, "run"),
    ),
    "<WORKER_USER>": ("the Linux account that runs the monitor", lambda c: _user()),
    "<CLAUDE_WORKSPACE>": (
        "the Claude Code project with the monitor's skill",
        lambda c: c.get("claude_workspace") or "",
    ),
    "<LESSONS_FILE>": (
        "the editable lessons file the skill reads and the weekly review edits",
        _lessons_file,
    ),
    "<SITE_LESSONS_FILE>": (
        "the site lessons: one site's lessons, in the monitor's state (paths.site_lessons), which lessons_for.py adds "
        "to each role's extract",
        lambda c: _p(c, "site_lessons"),
    ),
    "<CONTAINER_NOTE>": (
        "in a container: how the orchestrator drives the skill (empty elsewhere)",
        _container_note,
    ),
    "<ATLASSIAN_SITE>": (
        "the JIRA site's name, which the atlassian CLI's --site takes (from integrations.atlassian_cli.jira_url)",
        _atlassian_site,
    ),
    "<ROCM_LLVM_BIN>": ("ROCm's LLVM tools", lambda c: _p(c, "rocm_llvm_bin")),
    "<JIRA_PROJECT>": (
        "the JIRA project",
        lambda c: (c.get("jira") or {}).get("project") or "",
    ),
    "<JIRA_URL>": (
        "the JIRA site",
        lambda c: ((c.get("jira") or {}).get("url") or "").rstrip("/"),
    ),
    "<JIRA_COMPONENT>": (
        "new tickets' component",
        lambda c: (c.get("jira") or {}).get("component") or "",
    ),
    "<JIRA_LABELS>": (
        "new tickets' labels",
        lambda c: ", ".join((c.get("jira") or {}).get("labels") or []),
    ),
    "<JIRA_ISSUE_TYPE>": (
        "new tickets' issue type (features.ticket_creation.issue_type)",
        _issue_type,
    ),
    "<JIRA_REQUIRED_FIELDS>": (
        "the fields every new ticket must carry, as JSON: field id -> value (features.ticket_creation.required_fields)",
        _jira_fields,
    ),
    "<TEAM_CHANNEL>": (
        "the team chat while the develop break alerts are on (its webhook decides the channel)",
        _team_chat,
    ),
    "<TEAMS_WEBHOOK_FILE>": (
        "the channel's webhook (a secret)",
        lambda c: _secret(c, "teams_webhook"),
    ),
    "<TEAM_WEB_HOST>": (
        "the team dashboard's web host",
        lambda c: ((c.get("publish") or {}).get("html") or {}).get("host") or "",
    ),
    "<TEAM_DASHBOARD_URL>": (
        "the team dashboard's address",
        lambda c: ((c.get("publish") or {}).get("html") or {}).get("url") or "",
    ),
    "<TEAM_DASHBOARD_DEST>": (
        "where each tick pushes it",
        lambda c: (
            lambda h: f"{h.get('host')}:{h.get('dest')}" if h.get("host") else ""
        )((c.get("publish") or {}).get("html") or {}),
    ),
    "<WEBHOST_KEY_FILE>": (
        "the web host's ssh key (a secret)",
        lambda c: _secret(c, "web_host_ssh_key"),
    ),
    "<ADMIN_PORT>": ("the admin dashboard's port", lambda c: str(_admin(c)["port"])),
    "<ADMIN_URL>": ("the admin dashboard", lambda c: _admin_url(c)),
}
PLACEHOLDER = re.compile(
    "|".join(map(re.escape, sorted(PLACEHOLDERS, key=len, reverse=True)))
)


def values(cfg: dict) -> dict[str, str]:
    """Placeholder -> its value for this (normalized) configuration; blank ones read "(not set up)", except
    <CONTAINER_NOTE>, which is naturally empty outside a container.
    """
    out = {}
    for name, (_what, get) in PLACEHOLDERS.items():
        try:
            v = str(get(cfg) or "")
        except (KeyError, TypeError, ValueError, paths.PathsError):
            v = ""
        out[name] = v if v or name == "<CONTAINER_NOTE>" else NOT_SET
    return out


def fill(text: str, vals: dict[str, str]) -> str:
    """The placeholders filled in; an agent file's `effort:` line that comes to the model's default is left out (the
    agent then runs at its model's own default effort)."""
    from . import config

    text = PLACEHOLDER.sub(lambda m: vals[m.group(0)], text)
    return text.replace(f"\neffort: {config.MODEL_DEFAULT}\n", "\n")


def left(text: str) -> list[str]:
    """The placeholders still in a text."""
    return sorted(set(PLACEHOLDER.findall(text)))


# A team's own review skills, as delivered beside the deploy settings while
# features.review_automation.skills_folder names them: a folder laid out as the package's agentic folder.
TEAM_SKILLS = (
    pathlib.Path(container_layout.SETTINGS_DIR) / container_layout.SKILLS_FOLDER
)


def team_skills() -> pathlib.Path | None:
    """The team's skills that a render of the package's folder takes: where the settings are delivered
    (runtime_config.delivered), those that go with the layer in place (deliver.skills_with_layer: TEAM_SKILLS, or what
    a delivery that stopped after its commit staged for it), None when there are none. None elsewhere, where the
    package's skills run alone.
    """
    # call-time imports: runtime_config and deliver bring in the deployment's modules, which only this lookup needs
    from . import deliver, runtime_config

    return (
        deliver.skills_with_layer(TEAM_SKILLS) if runtime_config.delivered() else None
    )


def laid_over(team: pathlib.Path | None) -> str:
    """The end of a render's line, which names TEAM's skills laid over the package's; "" with none."""
    return f", with the team's skills in {team} laid over the package's" if team else ""


def sources(
    bundle: pathlib.Path, team: pathlib.Path | None = None
) -> list[tuple[str, pathlib.Path]]:
    """Each file a render takes, by its path in the rendered folder (with /), in the order of those paths: every
    file of the bundle, and TEAM's (a folder laid out as the bundle) laid over them: one with a bundle file's path
    replaces it, one with a new path is added."""
    files = {}
    for top in (bundle, team) if team else (bundle,):
        files.update(
            (p.relative_to(top).as_posix(), p) for p in top.rglob("*") if p.is_file()
        )
    return sorted(files.items(), key=lambda f: f[0].split("/"))


def render(
    bundle: pathlib.Path,
    cfg: dict,
    out: pathlib.Path,
    team: pathlib.Path | None = None,
) -> list[pathlib.Path]:
    """Every file of the bundle, with TEAM's laid over them (sources), filled in, under `out` (replaced as a whole).
    Raises ValueError naming any file that still holds a placeholder, and refuses a configuration with no operator
    login (LOGIN_MISSING): no agent may guess who <user_login> is."""
    vals = values(cfg)
    if vals[LOGIN] == NOT_SET:
        raise ValueError(LOGIN_MISSING)
    tmp = out.with_name(out.name + ".new")
    if tmp.exists():
        shutil.rmtree(tmp)
    done, bad = [], []
    for rel, src in sources(bundle, team):
        dest = tmp / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        text = fill(src.read_text(encoding="utf-8"), vals)
        if left(text):
            bad.append(f"{rel}: {', '.join(left(text))}")
        dest.write_text(text, encoding="utf-8")
        shutil.copymode(src, dest)
        done.append(out / rel)
    if bad:
        shutil.rmtree(tmp)
        raise ValueError("placeholders left after rendering: " + "; ".join(bad))
    (tmp / RENDER_STAMP).write_text(
        json.dumps({"digest": render_digest(bundle, cfg, team)}), encoding="utf-8"
    )
    if out.exists():
        shutil.rmtree(out)
    os.replace(tmp, out)
    return done


# In a rendered folder: what it was rendered from (the bundle, the team's files and this configuration's values), by
# digest. `install` records it in the workspace, so `refresh` knows when the installed skill is older than they are.
RENDER_STAMP = ".rendered.json"


def render_digest(
    bundle: pathlib.Path, cfg: dict, team: pathlib.Path | None = None
) -> str:
    """The digest of a render's inputs: every file it takes (sources: the bundle's, with TEAM's laid over them), and
    every placeholder's value for this configuration."""
    h = hashlib.sha256(json.dumps(values(cfg), sort_keys=True).encode("utf-8"))
    for rel, src in sources(bundle, team):
        h.update(rel.encode("utf-8") + b"\0")
        h.update(src.read_bytes())
    return h.hexdigest()


def installed_digest(workspace: pathlib.Path) -> str:
    try:
        m = json.loads(
            (workspace / ".claude" / "skills" / SKILL / INSTALLED).read_text(
                encoding="utf-8"
            )
        )
    except (OSError, ValueError):
        return ""
    return str(m.get("_render") or "") if isinstance(m, dict) else ""


def refresh(
    cfg: dict,
    bundle: pathlib.Path | None = None,
    first: bool = False,
    team: pathlib.Path | None = None,
) -> list[str]:
    """Render and install again when the skill is installed and its render is older than this configuration, the
    bundle or the team's files laid over it (rk provision, `rk orchestrate`'s start, an update's restart): the
    installed skill always names the configuration's operator, and runs the team's skills as delivered. BUNDLE and
    TEAM: the folder rendered and the team's laid over it; with no BUNDLE, the package's agentic folder and, unless
    TEAM is given, the team's skills delivered here (team_skills). With FIRST (rk provision, which each start runs), a
    workspace without the skill gets its first render and install too. [] when the render the manifest records is
    current, whether or not a file of it is gone (rk provision puts it back: put_back), or not installed without
    FIRST; ValueError (LOGIN_MISSING) when the configuration has no operator login.
    """
    from . import config

    if bundle is None:
        bundle, team = paths.PACKAGE / "agentic", team or team_skills()
    ws = (cfg.get("claude_workspace") or "").strip()
    if not ws or not (bundle / "skills").is_dir():
        return []
    workspace = pathlib.Path(os.path.expanduser(ws))
    there = (workspace / ".claude" / "skills" / SKILL / "SKILL.md").is_file()
    if not there and not first:
        return []
    norm = config.normalize(cfg)
    installed = installed_digest(workspace)
    if installed == render_digest(bundle, norm, team):
        return []
    out = pathlib.Path(paths.get(cfg, "agentic"))
    render(bundle, norm, out, team)
    lessons = pathlib.Path(paths.get(cfg, "lessons")) / "lessons.md"
    if not there:
        said = f"rendered with this configuration into {out}"
    elif installed:
        said = f"the installed skill was older than this configuration: rendered again into {out}"
    else:
        said = (
            f"the installed skill has no record of its render ({INSTALLED} is gone, or an install was cut short): "
            f"rendered again into {out}"
        )
    return [said + laid_over(team), *install(out, workspace, lessons_file=lessons)]


def _stamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S")


def _write_manifest(mpath: pathlib.Path, manifest: dict) -> None:
    mpath.parent.mkdir(parents=True, exist_ok=True)
    mpath.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")


def _installs(rel: str) -> bool:
    """Whether install writes the render's file REL (its path there, with forward slashes, as the manifest names it)
    into the workspace: the skill, team members and agent types, not the registry rows, which go into the dispatch
    table. A manifest entry that names no such path (one edited by hand) is never removed.
    """
    parts = rel.split("/")
    return (
        parts[0] in ("skills", "agents", "team_members")
        and rel != REGISTRY_ROWS
        and ".." not in parts  # never one that climbs out of its folder
    )


def _backed_up(dest: pathlib.Path, installed: str | None, dry: bool) -> list[str]:
    """Back DEST up as <name>.bak-<stamp> before install replaces or removes it, unless it is as an install wrote it
    (INSTALLED, the digest given) or a backup holds it already. The line says so."""
    have = hashlib.sha256(dest.read_bytes()).hexdigest()
    if have == installed or any(
        b.is_file() and hashlib.sha256(b.read_bytes()).hexdigest() == have
        for b in dest.parent.glob(f"{dest.name}.bak-*")
    ):
        return []
    backup = dest.with_name(f"{dest.name}.bak-{_stamp()}")
    if not dry:
        shutil.copyfile(dest, backup)
    return [f"back up {dest} -> {backup.name}"]


def install(
    rendered: pathlib.Path,
    workspace: pathlib.Path,
    dry: bool = False,
    lessons_file: pathlib.Path | None = None,
) -> list[str]:
    """Copy the rendered skill, team members and agent types into <workspace>/.claude: a changed file is backed up
    first, a shared policy the workspace already has is kept, and the registry rows go between the dispatch table's
    sentinels, in place of the rocKE rows there (merge_rows). A file an earlier install wrote that the render no
    longer has (a skill the team dropped, or one a package update dropped) is removed, backed up first when edited
    since, and so is a folder it leaves empty; a shared policy stays. The skill's lessons.md also goes to
    `lessons_file` (<LESSONS_FILE>) when there is none yet: that copy is the one the skill reads and edits, and is never
    overwritten. The manifest (INSTALLED) records the files written, and the render last, once all of it is in place:
    an install cut short records none, so the next `refresh` installs again. The lines say what was (or would be) done.
    """
    claude = workspace / ".claude"
    out = []
    # The files an earlier install wrote, by digest: one still as installed is replaced or removed without a backup;
    # one edited since is backed up first, once.
    mpath = claude / "skills" / SKILL / INSTALLED
    try:
        manifest = json.loads(mpath.read_text(encoding="utf-8"))
        manifest = manifest if isinstance(manifest, dict) else {}
    except (OSError, ValueError):
        manifest = {}
    installing = set()
    for src in sorted(p for p in rendered.rglob("*") if p.is_file()):
        rel = src.relative_to(rendered).as_posix()
        if not _installs(rel):
            continue
        installing.add(rel)
        dest = claude / rel
        if rel.startswith(SHARED) and dest.exists():
            out.append(f"kept {dest} (the workspace's own)")
            continue
        new = hashlib.sha256(src.read_bytes()).hexdigest()
        if dest.exists() and filecmp.cmp(src, dest, shallow=False):
            manifest[rel] = new
            continue
        if dest.exists():
            out += _backed_up(dest, manifest.get(rel), dry)
        out.append(f"{'update' if dest.exists() else 'add'} {dest}")
        if not dry:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
        manifest[rel] = new
    # A file an earlier install wrote that this render no longer has (a skill the team dropped, or one a package
    # update dropped) goes, backed up first when edited since, and so does a folder it leaves empty. A shared policy
    # stays, the workspace's own from now on.
    for rel in sorted(r for r in manifest if r not in installing and _installs(r)):
        dest = claude / rel
        if not rel.startswith(SHARED) and dest.is_file():
            out += _backed_up(dest, manifest[rel], dry)
            out.append(
                f"remove {dest} (installed before; not in the package or the team's skills)"
            )
            if not dry:
                dest.unlink()
                top, folder = claude / rel.split("/")[0], dest.parent
                while folder != top and not any(folder.iterdir()):
                    folder.rmdir()
                    folder = folder.parent
        del manifest[rel]
    # The files written so far, and no render yet: an install cut short before its end leaves none recorded, so the
    # next refresh installs again (installed_digest).
    manifest.pop("_render", None)
    if not dry and manifest:
        _write_manifest(mpath, manifest)
    shipped = rendered / "skills" / SKILL / "lessons.md"
    if lessons_file and shipped.exists():
        if not lessons_file.exists():
            out.append(
                f"add {lessons_file} (the editable lessons file; never overwritten)"
            )
            if not dry:
                lessons_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(shipped, lessons_file)
        elif not filecmp.cmp(shipped, lessons_file, shallow=False):
            out.append(
                f"kept {lessons_file} (the editable lessons file; the shipped one is in the skill)"
            )
    rows = rendered / REGISTRY_ROWS
    if rows.exists():
        out += merge_rows(
            claude / "team_members" / "dispatch_table.md",
            rows.read_text(encoding="utf-8"),
            dry,
        )
    # Last: the render counts as installed once the files, the lessons file and the registry rows are in place.
    try:
        manifest["_render"] = json.loads(
            (rendered / RENDER_STAMP).read_text(encoding="utf-8")
        )["digest"]
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if not dry and "_render" in manifest:
        _write_manifest(mpath, manifest)
    top = private_files.git_top(workspace)
    if top:
        out.append(
            f"{workspace} is inside the git work tree {top}: to keep these files out of its commits, add "
            f".claude/skills/{SKILL}/, .claude/agents/rocke-*.md and .claude/team_members/rocke_*.md to "
            f"{top}/.git/info/exclude"
        )
    return out or ["the workspace already has these files"]


def put_back(rendered: pathlib.Path, workspace: pathlib.Path) -> list[str]:
    """The files install writes into <workspace>/.claude that are gone from there, copied again from RENDERED, the
    render installed there (rk provision, while refresh finds the render the manifest records current): SKILL.md and
    the skill's other files, a shared policy, the team members and the agent types, and the registry rows (merge_rows)
    when the dispatch table is gone or lacks one of them. A file that is there stays as it is, edited or not, and none
    is removed; a dispatch table with no generated registry section is the workspace's own. What was put back, by its
    path under .claude (with /); [] when nothing was gone."""
    claude = workspace / ".claude"
    back = []
    for src in sorted(p for p in rendered.rglob("*") if p.is_file()):
        rel = src.relative_to(rendered).as_posix()
        dest = claude / rel
        if _installs(rel) and not os.path.lexists(dest):
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dest)
            back.append(rel)
    rows = rendered / REGISTRY_ROWS
    table = claude / "team_members" / "dispatch_table.md"
    if rows.is_file():
        text = rows.read_text(encoding="utf-8")
        have = table.read_text(encoding="utf-8") if table.is_file() else ""
        lacks = {_row_file(r) for r in table_rows(text)} - {
            _row_file(r) for r in table_rows(have)
        }
        if lacks and (
            not table.is_file() or (SENTINEL_BEGIN in have and SENTINEL_END in have)
        ):
            merge_rows(table, text)
            back.append(table.relative_to(claude).as_posix())
    return back


def table_rows(text: str) -> list[str]:
    return [ln for ln in text.splitlines() if ln.startswith("|") and "`rocke_" in ln]


def _row_file(row: str) -> str:
    """The team member file a rocKE registry row names (table_rows), as it names it."""
    return re.search(r"`([^`]+)`", row).group(1)


def merge_rows(table: pathlib.Path, rows_text: str, dry: bool = False) -> list[str]:
    """Put the rocKE rows between the registry's sentinels, in place of every rocKE row there: a row for a team member
    the render no longer has (install removed its file) comes out, and the line says so.
    """
    rows = table_rows(rows_text)
    if not rows:
        return []
    if not table.exists():
        text = (
            "# Team Member Dispatch Reference\n\n## Team Member Registry\n\n| Generic Role | Team Member | File | "
            "Context | Domain |\n|---|---|---|---|---|\n"
            + SENTINEL_BEGIN
            + " — do not edit manually -->\n"
            + SENTINEL_END
            + "\n"
        )
    else:
        text = table.read_text(encoding="utf-8")
    if SENTINEL_BEGIN not in text or SENTINEL_END not in text:
        return [
            f"{table} has no generated registry section: add these rows yourself:",
            *rows,
        ]
    head, rest = text.split(SENTINEL_BEGIN, 1)
    mark, rest = rest.split("\n", 1)
    body, tail = rest.split(SENTINEL_END, 1)
    files = {_row_file(r) for r in rows}
    taken = sorted({_row_file(r) for r in table_rows(body)} - files)
    kept = [ln for ln in body.splitlines() if not table_rows(ln)]
    merged = sorted(
        [*(ln for ln in kept if ln.strip()), *rows],
        key=lambda ln: ln.split("|")[1].strip().lower(),
    )
    new = (
        head
        + SENTINEL_BEGIN
        + mark
        + "\n"
        + "\n".join(merged)
        + "\n"
        + SENTINEL_END
        + tail
    )
    if new == text:
        return []
    if not dry:
        table.parent.mkdir(parents=True, exist_ok=True)
        if table.exists():
            shutil.copy2(table, table.with_name(f"{table.name}.bak-{_stamp()}"))
        table.write_text(new, encoding="utf-8")
    return [
        f"registry rows for {', '.join(sorted(files))} in {table}"
        + (f"; the rows for {', '.join(taken)} taken out" if taken else "")
    ]
