# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The setup summary (VOCABULARY.md, "The setup summary"): what is configured and what is not, explicitly. `rk
features` prints it at any time.

1. Features: each feature's switches (prototype or live), what starts, then every capability, on or off and the
   setting that decides it; for one that is on but incomplete, what is missing and the command that fixes it; then
   "Configured and on" and "Not configured or off".
2. Configuration: every value the features use, grouped: where things are, GitHub, dashboards, JIRA, team chat,
   models (the reviewer model: its provider, endpoint, model and what the start's checks found; the orchestrator model;
   the routing), the parallel reviews (asked, in effect and why) and the checks limit, the monitor's stop state,
   secrets. A secret is shown only as "set" or "missing" with its file's path, never its value, or as "not used" and why
   (one a feature's sends need, while that feature is in prototype: its mode); an endpoint without any user, password or
   query. A row that names a setting names the deploy key that sets it: a package setting no deploy key sets has no row.
"""

from __future__ import annotations

import datetime as dt
import os
import pathlib

from . import breakage, config, deploy_tree, features, paths, publish, webserve
from .checks import PROBLEMS


def _secret_state(path: str | pathlib.Path | None) -> str:
    if not path:
        return "missing (no file configured)"
    p = pathlib.Path(path)
    try:
        ok = p.is_file() and bool(
            p.read_text(encoding="utf-8", errors="replace").strip()
        )
    except OSError:
        ok = False
    return f"{'set' if ok else 'missing'} ({p})"


def _found_state(cfg: dict, name: str) -> str:
    """The secret NAME's row, from the one that may look (logins.found: in the container the writer answers for the
    secrets it keeps): set or missing with its place, or not known with why."""
    from . import logins, secrets_registry as sr

    got = logins.found(cfg, sr.entry(cfg, name))
    return f"{logins.said(got)} ({got['where']})"


def _needed_state(cfg: dict, name: str, off: str) -> str:
    """The row of the secret NAME, which a feature's sends need: `_found_state`'s while it is needed
    (secrets_registry.needed_by); else not used, and why: the mode while it waits for a feature in prototype
    (secrets_registry.waiting_words), else OFF."""
    from . import secrets_registry as sr

    entry = sr.entry(cfg, name)
    if sr.needed_by(cfg, entry):
        return _found_state(cfg, name)
    return f"not used ({sr.waiting_words(cfg, entry) or off})"


def secrets(cfg: dict) -> list[tuple[str, str]]:
    """Each secret this configuration uses: (what, "set|missing (path)"). Never a value. The reviewer model's key is
    the secret its binding names (config.worker's api_key_file)."""
    w = config.worker(cfg)
    key = (
        _secret_state(w["api_key_file"])
        if w.get("api_key_file")
        else f"not used ({deploy_tree.name('worker.api_key_secret')} is blank)"
    )
    web = features.on(cfg, "admin-dashboard")
    return [
        (
            "admin dashboard access token",
            (
                _secret_state(webserve.admin(cfg)["access_token_file"])
                if web
                else "not used (features.dashboards.admin is off)"
            ),
        ),
        (
            "team chat webhook",
            _needed_state(cfg, "teams_webhook", "features.alerts is off"),
        ),
        (
            "team web host's ssh key",
            _needed_state(
                cfg, "web_host_ssh_key", "the team dashboard is not published"
            ),
        ),
        ("reviewer model's API key", key),
        ("GPU cluster's SSH key", _key_state(cfg)),
    ]


def _key_state(cfg: dict) -> str:
    """The remote GPU tests' SSH key: set or missing, with its path (a private key: never read here)."""
    from . import gpu

    if not features.on(cfg, gpu.FEATURE) and not (cfg.get("gpu") or {}).get(
        "login_host"
    ):
        return "not used (remote GPU tests are off)"
    key = gpu.key_file(cfg)
    return f"{'set' if key and os.path.isfile(key) else 'missing'} ({key or 'no file configured'})"


def features_table(cfg: dict) -> list[str]:
    """The features' switches, what starts, and every capability (rkmon_lib/features.py summary: the one account)."""
    return ["## Features", "", *features.summary(cfg).rstrip("\n").splitlines(), ""]


def _worker_rows(cfg: dict, w: dict, refused: str) -> list[tuple[str, str]]:
    """The local worker's rows of the Models table: shown only while it is the reviewer (reviewer worker)."""
    from . import models

    return [
        (
            f"Reviewer model: provider ({deploy_tree.name('worker.backend')})",
            {"ollama": "Ollama", "openai": "an OpenAI-compatible endpoint"}.get(
                w["backend"], w["backend"]
            ),
        ),
        (
            "Reviewer model: endpoint (<WORKER_URL>)",
            config.safe_url(w["url"])
            + (f"; chat route {w['chat_path']}" if w.get("chat_path") else "")
            + (
                f"; models listed at {config.safe_url(w['models_url'])}"
                if w.get("models_url")
                else ""
            ),
        ),
        (
            "Reviewer model: model (<WORKER_MODEL>)",
            w["model"]
            + (f" (shown as {w['model_name']})" if w.get("model_name") else "")
            + (f" — REFUSED for the reviewer role: it {refused}" if refused else ""),
        ),
        ("Reviewer model: key sent as", _key_sent(w)),
        (
            "Reviewer model: found by the start's checks",
            "; ".join(models.describe(w.get("probe") or {})),
        ),
        (
            "Reviewer model: limits in use",
            f"context {w['context']} tokens, max output {w['max_output_tokens']}, "
            + (
                f"{w['rpm']} requests a minute"
                if w.get("rpm")
                else "no request limit known"
            )
            + ", "
            + (
                f"{w['tpm']} tokens a minute"
                if w.get("tpm")
                else "no token limit known"
            ),
        ),
    ]


def _key_sent(w: dict) -> str:
    """How the reviewer endpoint gets its key, never the key: the headers it goes in."""
    from .models import client

    if not w.get("api_key_file"):
        return "no key"
    return " and ".join(
        f"`{k}: {v}`" for k, v in client().auth_headers(w, "<key>").items()
    )


def _pass_model(internal: str) -> str:
    """A Models row's name: the pass whose model the configuration key INTERNAL sets, and the deploy key that sets it
    (features.review_automation.passes.<pass>.model)."""
    key = deploy_tree.name(internal)
    return f"Model of the {key.split('.')[-2].replace('_', ' ')} ({key})"


def _warned(model: str) -> str:
    """A Claude model, with the warning for one not known to drive the monitor well (config.orchestrator_warning)."""
    warning = config.orchestrator_warning(model)
    return model + (f" (WARNING: {warning})" if warning else "")


def _table(title: str, rows: list[tuple[str, str]]) -> list[str]:
    return [
        f"### {title}",
        "",
        "| Item | Value |",
        "|---|---|",
        *(f"| {k} | {v or '(not set)'} |" for k, v in rows),
        "",
    ]


def _gh_row(cfg: dict, g: dict | None) -> str:
    """gh's version, where it lives, its login and scopes (never the token)."""
    if g is None:
        return "(not checked)"
    if not g["found"]:
        return f"missing ({g['binary']} not found)"
    return (
        f"{g['version']} at {g['binary']}; "
        + (f"logged in as {g['login']}" if g["logged_in"] else "not logged in")
        + (f"; scopes {', '.join(g['scopes'])}" if g["scopes"] else "")
        + f"; login kept in {g['config_dir']}"
    )


def configuration(
    cfg: dict,
    claude: tuple[str, str] | None = None,
    gh_info: dict | None = None,
    stop_state: str | None = None,
) -> list[str]:
    from . import breakage, derived, doctor

    p = paths.resolve(cfg)
    repo, k = cfg.get("repo") or "", paths.repository_folder(cfg)
    ii = cfg.get("incident_issue") or {}
    issues_repo = breakage.issue_repository(cfg)
    sc = config.scope(cfg)
    teams = cfg.get("codeowner_teams") or {}
    org = cfg.get("codeowner_org") or (repo.split("/")[0] if "/" in repo else "")
    pub = (cfg.get("publish") or {}).get("html") or {}
    j = cfg.get("jira") or {}
    w = config.worker(cfg)
    orch = config.orchestrator_model(cfg)
    routes = []
    for role in (r for r in config.ROLES if r not in config.MODEL_ROLES):
        m = config.model_for(cfg, role)
        routes.append(
            f"{role}: "
            + (
                {
                    "claude": (
                        f"Claude ({m['model']})"
                        if role in config.CLAUDE_MODELS
                        else f"the orchestrator ({orch})"
                    ),
                    "none": f"the orchestrator alone ({orch}, no first pass)",
                    "binding": f"the binding {m.get('binding')} ({m.get('name') or m.get('model')}, "
                    f"{m.get('provider')})",
                }.get(m["by"])
                or f"{m['backend']} {m['model'] or '(model not set)'}"
            )
        )
    where, how = config.home_and_why()
    out = ["## Configuration", ""]
    out += _table(
        "Where things are",
        [
            ("Working folder (<WORKING_DIR>)", p["working_dir"]),
            ("Outputs folder", p["outputs_dir"]),
            (
                "Monitor home: rk and state/, on a host config.json too (<TOOLKIT_DIR>)",
                f"{where} (found by: {how})",
            ),
            ("The monitor's clone (<ROCKE_SRC_DIR>)", p["repo"]),
            ("Secrets folder (<SECRETS_DIR>)", p["secrets_dir"]),
            ("The toolkit's code (read-only)", p["package"]),
            ("Worker tools", p["worker_tools"]),
            ("Review book", p["review_book"]),
        ],
    )
    out += _table(
        "GitHub",
        [
            ("Repository (<ROCKE_REPO>)", repo),
            (
                "rocKE (<ROCKE_URL>)",
                (
                    f"{derived.github_address(repo)}/tree/develop/{k}"
                    if repo and k
                    else ""
                ),
            ),
            (
                "Develop issues (<ROCKE_GITHUB_ISSUES>)",
                (
                    (
                        f"{derived.github_address(issues_repo)}/issues: labels "
                        f"{breakage.labels(cfg)[breakage.RED] or 'not set'} (CI red), "
                        f"{breakage.labels(cfg)[breakage.STABILITY] or 'not set'} (CI does not see "
                        "it)"
                    )
                    if issues_repo
                    else ""
                ),
            ),
            ("Operator (<user_login>)", config.operator_login(cfg) or ""),
            (
                "Scope labels",
                ", ".join(sc.labels)
                + (f"; shared labels: {sc.shared_text()}" if sc.shared else ""),
            ),
            (
                "Code-owner teams",
                "; ".join(
                    f"{area}: {', '.join(f'{org}/{t}' for t in ts) or '-'}"
                    for area, ts in teams.items()
                ),
            ),
            ("GitHub CLI", _gh_row(cfg, gh_info)),
        ],
    )
    where = webserve.exposure(cfg)
    team_on = features.on(cfg, "team-dashboard")
    serving = team_on and publish.mode(cfg) == "serve"
    out += _table(
        "Dashboards",
        [
            ("Admin dashboard (<ADMIN_URL>)", webserve.home_url(cfg)),
            (
                "Admin dashboard reachable from",
                (
                    f"beyond this machine, on {where}"
                    if where
                    else "this machine only (the loopback)"
                )
                + "; every request needs the access token ("
                + (
                    "set"
                    if webserve.access_token(cfg)
                    else "MISSING: the service does not start"
                )
                + ")",
            ),
            (
                "Team dashboard (<TEAM_DASHBOARD_URL>)",
                (
                    "off"
                    if not team_on
                    else (
                        f"served here at {webserve.team_url(cfg)} (read-only, with its request box)"
                        if serving
                        else pub.get("url")
                        or f"({deploy_tree.name('publish.html.url')} not set)"
                    )
                ),
            ),
            (
                "Pushed to (<TEAM_DASHBOARD_DEST>)",
                (
                    f"{pub.get('user') or '?'}@{pub['host']}:{pub.get('dest') or '?'}"
                    if pub.get("host") and team_on and not serving
                    else "off"
                ),
            ),
            ("Dashboard file", p["html"]),
            ("Markdown dashboard", p["dashboard"]),
        ]
        + (
            [
                (
                    "Published by docker/.env",
                    " ".join(f"{k}={v}" for k, v, _d in webserve.publish_env(cfg)),
                )
            ]
            if doctor.deployment_image()
            else []
        ),
    )
    from . import atlassian, orchestrator

    out += _table(
        "JIRA",
        [
            (
                "Project (<JIRA_URL>/<JIRA_PROJECT>)",
                (
                    f"{(j.get('url') or '(JIRA site not set)').rstrip('/')}/"
                    f"{j.get('project')}"
                    if j.get("project")
                    else "off"
                ),
            ),
            ("Component", j.get("component") or ""),
            ("Labels", ", ".join(j.get("labels") or [])),
            ("atlassian CLI (<ATLASSIAN_SITE>)", atlassian.summary_row(cfg)),
        ],
    )
    from . import post

    out += _table(
        "Team chat",
        [
            (
                f"Message label ({deploy_tree.name('teams.channel')})",
                post.message_label(cfg) or "none",
            ),
            ("Webhook", next(v for n, v in secrets(cfg) if n == "team chat webhook")),
        ],
    )
    version, login = claude or ("(not checked)", "(not checked)")
    from . import models

    refused = config.probe_failure(w)
    from . import bindings

    out += _table(
        "Models",
        [
            (
                _pass_model("reviewer"),
                bindings.reviewer_line(cfg)
                + "; line 1: "
                + (bindings.first_line(cfg) or "(not set yet)"),
            ),
            *(
                _worker_rows(cfg, w, refused)
                if bindings.reviewer_choice(cfg) == bindings.WORKER
                else []
            ),
            ("Orchestrator model (<ORCHESTRATOR_MODEL>)", _warned(orch)),
            ("Routing", "; ".join(routes)),
            *(
                (_pass_model(f"models.{r}"), _warned(config.claude_model(cfg, r)))
                for r in config.CLAUDE_MODELS
            ),
            (
                "Claude's effort (<CLAUDE_EFFORT>)",
                config.role_effort(cfg, "orchestrator")[0],
            ),
            ("Claude Code", f"{version or 'not found'}, {login}"),
            (
                "The monitor's Claude session (the container's orchestrator)",
                (
                    orchestrator.describe(cfg)
                    if doctor.deployment_image()
                    else "none here: a person's Claude Code session runs the skill"
                ),
            ),
        ],
    )
    rec = doctor.rocm_image()
    parity = doctor.clang_parity(cfg)
    out += _table(
        "ROCm",
        [
            (
                "LLVM tools (<ROCM_LLVM_BIN>)",
                (
                    (
                        p["rocm_llvm_bin"]
                        + (
                            f": {doctor.clang_line(p['rocm_llvm_bin'])}"
                            if doctor.clang_line(p["rocm_llvm_bin"])
                            else ""
                        )
                    )
                    if p["rocm_llvm_bin"]
                    else "not set: the develop guard and the compile gate wait"
                ),
            ),
            (
                "ROCm release (the image)",
                (
                    f"{rec.get('version')} (gfx {rec.get('gfx') or '-'})"
                    if rec
                    else "none (a host install, or an image built without ROCm)"
                ),
            ),
            (
                "Reference clang (the gate compiler)",
                f"{parity.status}: {parity.found}"
                + (f" — {parity.fix}" if parity.status in PROBLEMS else ""),
            ),
        ],
    )
    from . import ramdisk, scaler

    rd = ramdisk.settings(cfg)
    lo_p, hi_p = scaler.ranges()["pipes"]
    lo_h, hi_h = scaler.ranges()["heavy"]
    out += _table(
        "Parallel reviews and parallel builds",
        [
            (
                "Parallel reviews",
                config.pipes_line(cfg)
                + f"; the starting setting, then adjusted within "
                f"{lo_p}-{hi_p} to how busy this machine is",
            ),
            (
                "Parallel builds (<CHECKS_LIMIT>)",
                f"{config.checks_limit(cfg)} local build(s) with their tests at once to "
                f"start with, then adjusted within {lo_h}-{hi_h}; the parallel reviews "
                "never change it",
            ),
            (
                "RAM disk (host.resources.ramdisk_gib)",
                (
                    (
                        f"on: {rd['path'] or '(no RAM disk folder is set)'}, {rd['size_gb']:g} GiB: the PRs' "
                        "worktrees and builds and paths.tmp; rk doctor checks it is mounted"
                    )
                    if rd["enabled"]
                    else "off: the worktrees, builds and temporary files stay on the disk"
                ),
            ),
        ],
    )
    from . import gpu

    gs = gpu.settings(cfg)

    def key(name: str) -> str:
        # the deploy settings file's key that sets gpu.NAME
        return deploy_tree.name(f"gpu.{name}")

    if features.on(cfg, gpu.FEATURE) or gs["login_host"]:
        out += _table(
            "Remote GPU tests",
            [
                (
                    f"Login host ({key('login_host')})",
                    f"{gs['ssh_user'] or '?'}@{gs['login_host'] or '(not set)'}"
                    + (f", port {gs['ssh_port']}" if gs["ssh_port"] != 22 else ""),
                ),
                (
                    f"Partition; accounts in order ({key('partition')}; {key('accounts')})",
                    f"{gs['partition'] or '(not set)'}; {', '.join(gs['accounts']) or '(none)'}"
                    + (f"; constraint {gs['constraint']}" if gs["constraint"] else ""),
                ),
                (
                    "Targets and their GPU types",
                    ", ".join(f"{a} -> {t}" for a, t in gs["gres"].items()) or "(none)",
                ),
                ("Excluded nodes", ", ".join(gs["exclude"]) or "none"),
                (
                    f"Remote folder; Python ({key('remote_dir')}; {key('remote_python')})",
                    f"{gs['remote_dir']}; {gs['remote_python'] or '(not set)'}",
                ),
                (
                    "Limits",
                    f"{gs['time_limit']} a job; at most {gs['max_jobs_per_pr']} target(s) a review; reported after "
                    f"{gs['pending_report_min']} min in the queue; {gs['resubmits']} resend(s) without a node that "
                    "failed the probe",
                ),
            ],
        )
    out += _table("Monitor state", [("Stop state", stop_state or "(not checked)")])
    out += _table("Secrets", secrets(cfg))
    return out


def render(
    cfg: dict,
    claude: tuple[str, str] | None = None,
    gh_info: dict | None = None,
    stop_state: str | None = None,
) -> str:
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        "\n".join(
            [
                f"# rocKE monitor: setup summary ({stamp})",
                "",
                "What is configured and what is not. Secrets show only as set or missing, with their path.",
                "",
                *features_table(cfg),
                *configuration(cfg, claude, gh_info, stop_state),
            ]
        )
        + "\n"
    )
