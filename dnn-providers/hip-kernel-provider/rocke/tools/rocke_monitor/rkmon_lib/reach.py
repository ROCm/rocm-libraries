# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk doctor`'s live checks of each configured service: can the monitor reach it, and does it let us in. Each is
read-only and cheap, and none ever writes:
- GitHub: gh's account and server, whether its login holds (OK, or expired or invalid), and the API calls left this
  hour (`gh api rate_limit`, which costs none);
- JIRA: the site answers (its public server-info route), and the project reads through the atlassian CLI (each CLI
  call through `cli_run`: in the container, the writer runs it);
- the team chat: its webhook's host answers a TLS handshake; nothing is ever posted, and the webhook's address is a
  secret, never shown. In the container only the writer reads the webhook (rkmon_lib/writer.py), so the row says
  whether the writer has it, and `rk teams-test` is what reaches its host;
- the model gateways: each drafting binding's model list, with its key, never shown; the answer's status says whether
  the key is taken;
- in the container, the Claude session's key, the claude login: its test call (`claude_test`, which `rk claude-test`
  makes too), one message of one token sent as the session sends its requests, the key never shown; a FAIL while
  review automation is on, for the session cannot work without it;
- the GPU cluster, when it is set up: `rk gpu-check`'s read-only look at the login host;
- the RAM disk (local, never cached): mounted, and large enough for one review and one heavy job at the learned
  footprints, and in a memory-limited container, within the container's memory.
Each is a row of rkmon_lib/checks.py, its fix as the operator types it here (rkmon_lib/doctor.py's fixes). The remote
results are cached for CACHE_S seconds in state/doctor_cache.json (`--fresh` checks again), each for one
configuration, code and set of logins and secrets (setting, replacing or removing any entry of the secrets registry,
an atlassian token refresh among them, checks again: `marks`); the cache holds only what the report shows. The report
is for this machine only: no dashboard shows it, and it is never published.
"""

from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import pathlib
import re
import socket
import ssl
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request

from . import (
    atlassian,
    coherence,
    config,
    deployment,
    doctor,
    features,
    paths,
    ramdisk,
    whole_file,
    wrapping,
)
from .checks import (
    Check,
    Command,
    Described,
    Steps,
    check_from_json,
    check_to_json,
)

CACHE = "doctor_cache.json"
CACHE_S = 300
TIMEOUT_S = 10
TEST_CALL_TIMEOUT_S = (
    60  # the claude login's test call (claude_test): one message, answered in one token
)
RATE_LOW = 500  # fewer GitHub API calls left than this this hour: a warning
# an API error's type, as Anthropic's API names it: not_found_error
_ERROR_TYPE = re.compile(r"[a-z][a-z_]{0,39}")


# --- small helpers ----------------------------------------------------------------------------------------------------


def again(environ=os.environ) -> Command:
    """The command that runs the checks again now, the service checks past their cache: doctor.check_again with
    --fresh."""
    return Command((*doctor.check_again(environ).argv, "--fresh"))


def run_cmd(
    cmd: list[str], env: dict | None = None, timeout: int = 30
) -> tuple[int, str]:
    """A command's exit code and its output (127: not installed). Never raises."""
    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=dict(os.environ, **(env or {})),
        )
        return p.returncode, (p.stdout + p.stderr).strip()
    except FileNotFoundError:
        return 127, ""
    except (OSError, subprocess.SubprocessError) as exc:
        return 126, str(exc)


# What a request that nothing answered raises: the network's errors (urllib's URLError among them), a value that
# http.client refuses to send (ValueError: a line break in a header's value), and an answer that is no HTTP
# (http.client.HTTPException).
_NO_ANSWER = (OSError, ValueError, http.client.HTTPException)


def _no_answer(exc: BaseException) -> str:
    """Why nothing answered a request, from EXC, one of _NO_ANSWER: its type and the network's reason
    (`URLError: [Errno 111] Connection refused`); the type alone for a ValueError and an http.client.HTTPException,
    whose words may quote the request or what came back (a header's value: a key)."""
    if isinstance(exc, (ValueError, http.client.HTTPException)):
        return type(exc).__name__
    return f"{type(exc).__name__}: {getattr(exc, 'reason', exc)}"[:160]


def http_get(
    url: str, headers: dict | None = None, timeout: int = TIMEOUT_S
) -> tuple[int, str]:
    """(HTTP status, "") of a GET, or (0, the reason) when nothing answered (`_no_answer`). The body is read and
    dropped."""
    req = urllib.request.Request(
        url, headers=dict(headers or {}, **{"User-Agent": "rkmon-doctor"}), method="GET"
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read(65536)
            return r.status, ""
    except urllib.error.HTTPError as exc:
        return exc.code, ""
    except _NO_ANSWER as exc:
        return 0, _no_answer(exc)


class _Unredirected(urllib.request.HTTPRedirectHandler):
    """Follows no redirect: a request that carries a key goes to its own address alone, and a redirect is its
    answer."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def http_post(
    url: str, headers: dict, body: bytes, timeout: int = TEST_CALL_TIMEOUT_S
) -> tuple[int, str]:
    """(HTTP status, the error's type or "") of a POST that follows no redirect, or (0, the reason) when nothing
    answered (`_no_answer`). Of an answer only an error's type is kept (`_error_type`), never its message, which may
    quote the request."""
    req = urllib.request.Request(
        url,
        data=body,
        headers=dict(headers, **{"User-Agent": "rkmon-doctor"}),
        method="POST",
    )
    try:
        with urllib.request.build_opener(_Unredirected).open(req, timeout=timeout) as r:
            r.read(65536)
            return r.status, ""
    except urllib.error.HTTPError as exc:
        with exc:
            return exc.code, _error_type(exc)
    except _NO_ANSWER as exc:
        return 0, _no_answer(exc)


def _error_type(answer) -> str:
    """The type of the API error ANSWER holds, as Anthropic's API gives it (`{"error": {"type": "not_found_error",
    ...}}`); "" when it gives none so. Nothing else of the answer is kept."""
    try:
        data = json.loads(answer.read(65536) or b"{}")
    except (OSError, ValueError, http.client.HTTPException):
        return ""
    error = data.get("error") if isinstance(data, dict) else None
    kind = error.get("type") if isinstance(error, dict) else None
    return kind if isinstance(kind, str) and _ERROR_TYPE.fullmatch(kind) else ""


def tls_reach(host: str, port: int = 443, timeout: int = TIMEOUT_S) -> str:
    """ "" when `host` completes a TLS handshake on `port`, else why not."""
    try:
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with ssl.create_default_context().wrap_socket(sock, server_hostname=host):
                return ""
    except (OSError, ssl.SSLError) as exc:
        return f"{type(exc).__name__}: {exc}"[:160]


def _said(status: int, why: str) -> str:
    return f"HTTP {status}" if status else f"no answer ({why})"


# --- GitHub -----------------------------------------------------------------------------------------------------------


def github(cfg: dict, run=run_cmd) -> list[Check]:
    """gh's account, server and login, and the API calls left this hour."""
    from . import gh as gh_mod

    binary, env = gh_mod.settings(cfg)
    rc, text = run([binary, "auth", "status"], env)
    if rc == 127:
        return [
            Check(
                "GitHub",
                "FAIL",
                f"gh is not installed here ({binary})",
                doctor.image_fix(),
            )
        ]
    login_again = (
        Steps(
            (
                doctor.gh_login_fix(gh_mod.READ),
                "(the read login, a read-only token: the one gh uses here)",
            )
        )
        if doctor.deployment_image()
        else Command((binary, "auth", "login", "--scopes", "read:org,repo"))
    )
    server, login = gh_mod.logged_in_as(text)
    server = server or os.environ.get("GH_HOST") or "github.com"
    if rc != 0 or not login:
        bad = re.search(r"invalid|expired|revoked|bad credentials", text, re.I)
        return [
            Check(
                "GitHub",
                "FAIL",
                doctor.blocking(
                    f"{server}: "
                    + ("the login is expired or invalid" if bad else "not logged in"),
                    "every GitHub read and write",
                ),
                login_again,
            )
        ]
    me = config.operator_login(cfg) or ""
    rc, text = run([binary, "api", "rate_limit"], env)
    left = ""
    status, when = "PASS", "checks again"
    try:
        core = json.loads(text)["resources"]["core"]
        reset = time.strftime("%H:%MZ", time.gmtime(int(core["reset"])))
        left = (
            f"; rate limit {core['remaining']} of {core['limit']} left (resets {reset})"
        )
        if int(core["remaining"]) < RATE_LOW:
            status, when = "WARN", "checks again after the reset"
            left += (
                ": few API calls left this hour, so the tick reads less until the reset"
            )
    except (ValueError, KeyError, TypeError):
        left = "; rate limit not read" + (
            f" ({text.splitlines()[0][:80]})" if text else ""
        )
        status = "WARN"
    if me and login != me:
        return [
            Check(
                "GitHub",
                "WARN",
                f"{server}: account {login}, auth OK{left}; the configured login is {me}",
                Steps(
                    (
                        *(
                            login_again.parts
                            if isinstance(login_again, Steps)
                            else (login_again,)
                        ),
                        f"to log gh in as {me}; or",
                        *doctor.settings_parts("operator.github_login", login),
                    )
                ),
            )
        ]
    return [
        Check(
            "GitHub",
            status,
            f"{server}: account {login}, auth OK{left}",
            None if status == "PASS" else Steps((again(), when)),
        )
    ]


# --- JIRA -------------------------------------------------------------------------------------------------------------


def cli_run(cfg: dict):
    """The runner of these checks' atlassian CLI calls, (cmd, env) -> (exit code, output): atlassian._run, with the
    CLI's own environment (in the container, through the writer, which alone reads the login).
    """
    return lambda cmd, _env=None: atlassian._run(cmd, cfg)


def _cli_ready(cfg: dict, run) -> tuple:
    """(the CLI binary, "", None) when it is installed and logged in, else ("", why, what makes it ready:
    atlassian.cli_fix)."""
    s = atlassian.status(cfg, run=lambda cmd: run(cmd, atlassian.env(cfg)))
    fix = atlassian.cli_fix(cfg, s)
    if not s["found"]:
        return "", "the atlassian CLI is not installed here", fix
    if not s["logged_in"]:
        return "", "the atlassian CLI is not logged in", fix
    return s["binary"], "", None


def jira(cfg: dict, run=None, get=http_get) -> list[Check]:
    run = run or cli_run(cfg)
    j = cfg.get("jira") or {}
    project, url, site = (
        str(j.get("project") or ""),
        str(j.get("url") or "").rstrip("/"),
        atlassian.jira_site(cfg),
    )
    if not (project or url or site):
        # the deploy settings file's keys of the three, each once (one key gives the site and JIRA's address)
        keys = dict.fromkeys(
            deployment.N(k) for k in ("jira.project", "jira.url", "atlassian.site")
        )
        return [
            Check(
                "JIRA",
                "SKIP",
                f"not set up ({wrapping.joined(keys)} {'is' if len(keys) == 1 else 'are'} blank)",
            )
        ]
    out = []
    if url:
        status, why = get(f"{url}/rest/api/2/serverInfo")
        ok = status in (200, 401, 403)
        out.append(
            Check(
                "JIRA site",
                "PASS" if ok else "WARN",
                f"{config.safe_url(url)}: {_said(status, why)}",
                (
                    None
                    if ok
                    else Steps(
                        (
                            "fix the deployment's network to it, then",
                            again(),
                            "; or fix its address:",
                            *doctor.settings_parts(
                                "jira.url", Described("the JIRA site's address")
                            ),
                        )
                    )
                ),
            )
        )
    if site and project:
        binary, why, ready = _cli_ready(cfg, run)
        if not binary:
            out.append(
                Check(
                    "JIRA project",
                    "WARN",
                    doctor.blocking(
                        f"{project} on {site}: not checked: {why}",
                        "the ticket features",
                    ),
                    ready,
                )
            )
        else:
            rc, text = run(
                [
                    binary,
                    "--site",
                    site,
                    "--json",
                    "jira",
                    "project",
                    "versions",
                    project,
                ],
                atlassian.env(cfg),
            )
            out.append(
                Check(
                    "JIRA project",
                    "PASS" if rc == 0 else "WARN",
                    f"{project} on {site}: "
                    + (
                        "readable through the atlassian CLI"
                        if rc == 0
                        else f"not readable ({atlassian._first_line(text) or rc})"
                    ),
                    (
                        None
                        if rc == 0
                        else Steps(
                            (
                                f"fix {deployment.N('jira.project')}:",
                                *doctor.section_parts("jira"),
                                "; or the CLI's login (its site and scopes):",
                                atlassian.login_fix(),
                            )
                        )
                    ),
                )
            )
    elif project and not site:
        out.append(
            Check(
                "JIRA project",
                "WARN",
                f"{project}: no {deployment.N('atlassian.site')} to read it through",
                doctor.settings_fix(
                    "atlassian.site", Described("the Atlassian site's address")
                ),
            )
        )
    return out


# --- the team chat ----------------------------------------------------------------------------------------------------


def chat(cfg: dict, reach=tls_reach) -> list[Check]:
    """The webhook's host answers a TLS handshake: reachability only, never a post; its address is never shown. Where
    this process does not read the webhook (logins.here: the writer's, in the container), whether it is set, as the
    writer says (logins.found). No webhook is a warning only while it is needed (secrets_registry.needed_by: the alerts
    live, with a kind of alert on); in prototype the row is off and names the mode.
    """
    from . import logins, secrets_registry as sr

    entry = sr.entry(cfg, "teams_webhook")
    got = logins.found(cfg, entry)
    try:
        url = (
            pathlib.Path(got["where"]).read_text(encoding="utf-8").strip()
            if got["set"] and logins.here(entry)
            else ""
        )
    except OSError:
        url = ""
    if not got["set"] and not sr.needed_by(cfg, entry):
        # Nothing that is on posts to the channel (in prototype the alerts send nothing): no webhook is no warning.
        waiting = sr.waiting_words(cfg, entry)
        sets_it = sr.typed_set_command(entry)
        return [
            Check(
                "team chat",
                "OFF",
                (
                    f"{waiting}, so no webhook is needed yet ({sets_it} sets it)"
                    if waiting
                    else "the develop break alerts and the needs-you alerts are off (the webhook is needed once "
                    f"either is on: {sets_it})"
                ),
            )
        ]
    if got["set"] is None:
        return [
            Check(
                "team chat",
                "WARN",
                f"whether the webhook is set is not known ({got['where']})",
                again(),
            )
        ]
    if not got["set"]:
        return [
            Check(
                "team chat",
                "WARN",
                doctor.blocking(
                    f"the webhook is missing ({got['where']})",
                    "chat-alerts",
                ),
                sr.set_fix(entry),
            )
        ]
    if not logins.here(entry):
        return [
            Check(
                "team chat",
                "PASS",
                "the writer has the webhook and alone reads it, so its host is "
                "not reached from here (rk teams-test sends a test message through the writer)",
            )
        ]
    host = urllib.parse.urlsplit(url).hostname or ""
    why = reach(host) if host else "the webhook is not an address"
    return [
        Check(
            "team chat",
            "PASS" if not why else "WARN",
            (
                "the webhook's host answers (TLS); nothing was posted"
                if not why
                else doctor.blocking(
                    f"the webhook's host does not answer ({why})", "chat-alerts"
                )
            ),
            (
                None
                if not why
                else Steps(
                    (
                        "fix the deployment's network to the chat service, then",
                        again(),
                    )
                )
            ),
        )
    ]


# --- the model gateways -----------------------------------------------------------------------------------------------


def _key_said(status: int, why: str) -> tuple[str, str]:
    if status == 200:
        return "PASS", "HTTP 200: the key is taken"
    if status in (401, 403):
        return "FAIL", f"HTTP {status}: the key is refused"
    return "WARN", _said(status, why) + (" (the key not confirmed)" if status else "")


def gateways(cfg: dict, get=http_get) -> list[Check]:
    """Each binding the reviewer and the roles use: a cheap authenticated GET of the model list (the provider adapter's
    place and headers: /v1/models with anthropic-version for an anthropic binding). The Claude session's key is the
    claude login's row (claude_login)."""
    from . import bindings, first_pass

    llm_client = first_pass.llm_client()
    out, seen = [], set()
    for role in ("drafter", *(cfg.get("roles") or {})):
        name = bindings.binding_name(cfg, role)
        if not name or name.lower() in bindings.NO_DRAFT_NAMES or name in seen:
            continue
        seen.add(name)
        b = bindings.binding(cfg, role)
        if not b:
            where = deployment.N("reviewer") if role == "drafter" else f"roles.{role}"
            out.append(
                Check(
                    f"model gateway ({name})",
                    "WARN",
                    f"{where} names it, but model_bindings has no " "such binding",
                    doctor.section("reviewer" if role == "drafter" else "bindings"),
                )
            )
            continue
        ad = llm_client.adapter(b)
        base = llm_client.base_url(b)
        if not base:
            out.append(
                Check(
                    f"model gateway ({name})",
                    "WARN",
                    f"{doctor.url_key(name, b.get('kind') or '')} ({b['url']}) names a variable the monitor's "
                    "environment does not set",
                    Steps(
                        (
                            "give the binding its address:",
                            *doctor.section_parts(
                                "reviewer" if role == "drafter" else "bindings"
                            ),
                        )
                    ),
                )
            )
            continue
        auth = ad.headers(b, llm_client.api_key(b))
        status, why = 0, ""
        for url in ad.models_urls(b):
            status, why = get(url, auth)
            if status != 404:
                break
        s, said = _key_said(status, why)
        out.append(
            Check(
                f"model gateway ({name})",
                s,
                f"{config.safe_url(base)}: {said}",
                (
                    None
                    if s == "PASS"
                    else Steps(
                        (
                            doctor.rk_command("drafts"),
                            "shows the binding's url and its key's place; fix them:",
                            *doctor.section_parts(
                                "reviewer" if role == "drafter" else "bindings"
                            ),
                        )
                    )
                ),
            )
        )
    return out


# --- the Claude session's key -----------------------------------------------------------------------------------------


def claude_test(cfg: dict, again, post=http_post) -> Check:
    """The claude login's test call, as the login's row: the key the Claude session reads (orchestrator.read_key),
    sent in one message as the session sends its requests (orchestrator.test_request), never shown. PASS when it is
    answered. FAIL with no key, or when the key is refused (HTTP 401 or 403), its fix the login's command
    (doctor.claude_login_fix); FAIL when nothing answers, its fix the deployment's network or an edit of the address,
    and for another refusal, which says the address or the model is wrong, its fix an edit of either: each edit of the
    settings as doctor.settings_parts gives it (the deploy settings file, then `<wrapper> confirm` and
    `<wrapper> apply`); WARN for an answer that says nothing of the key (HTTP 429, or a server's error). AGAIN, a fix,
    makes the call again after one of the latter: the doctor's `again`, or the login's command, which makes it too.
    """
    from . import orchestrator
    from . import secrets_registry as sr

    def blocked(found: str) -> str:
        return doctor.blocking(found, doctor.SESSION_BLOCKS)

    name, key = sr.CLAUDE_LOGIN, orchestrator.read_key(cfg)
    if not key:
        return Check(
            name, "FAIL", blocked("no Claude key"), doctor.claude_login_fix(cfg)
        )
    where = config.safe_url(orchestrator.gateway(cfg)) or "Anthropic's own API"
    model = config.claude_model_id(cfg)
    status, why = post(*orchestrator.test_request(cfg, key))
    if 200 <= status < 300:
        return Check(
            name, "PASS", f"one test call answered, at {where} (model {model})"
        )
    said = f"HTTP {status}{f', {why}' if why else ''}" if status else why
    then = again.parts if isinstance(again, Steps) else (again,)
    if status in (401, 403):
        return Check(
            name,
            "FAIL",
            blocked(f"the key is refused, at {where} ({said})"),
            doctor.claude_login_fix(cfg),
        )
    if status == 429 or status >= 500:
        return Check(
            name,
            "WARN",
            f"the test call could not be answered, at {where} ({said}): the key is not confirmed",
            Steps(("try again in a few minutes:", *then)),
        )
    if not status:
        reached = Described(
            "an address the deployment reaches, or blank for Anthropic's own API"
        )
        return Check(
            name,
            "FAIL",
            blocked(f"the test call had no answer, at {where} ({said})"),
            Steps(
                (
                    "fix the deployment's network to it, or",
                    *doctor.settings_parts("claude.gateway", reached),
                    ", then",
                    *then,
                )
            ),
        )
    address, *confirm = doctor.settings_parts(
        "claude.gateway", Described("the address of a gateway that serves the model")
    )
    served = Described(f"a model the address serves (now {model})")
    other = doctor.settings_parts("orchestrator.model", served)[0]
    return Check(
        name,
        "FAIL",
        blocked(f"the test call was refused, at {where} ({said})"),
        Steps((address, ", or", other, *confirm, ", then", *then)),
    )


def claude_login(cfg: dict, post=http_post) -> list[Check]:
    """The Claude session's key, in the container: the claude login's test call (claude_test), one row,
    a FAIL while review automation is on and the call does not pass; OFF while it is off, for then no Claude session
    runs. None on a host: there the Claude session is a person's Claude Code, with its own login (rkmon_lib/doctor.py's
    Claude Code row)."""
    from . import secrets_registry as sr

    if not doctor.deployment_image():
        return []
    if not features.uses_claude(cfg):
        return [
            Check(
                sr.CLAUDE_LOGIN,
                "OFF",
                "review automation is off: no Claude session runs, so none reads the key",
            )
        ]
    return [claude_test(cfg, again(), post=post)]


# --- the GPU cluster --------------------------------------------------------------------------------------------------


def gpu_cluster(cfg: dict, check=None) -> list[Check]:
    from . import gpu

    if not features.on(cfg, gpu.FEATURE) and not gpu.settings(cfg)["login_host"]:
        return [
            Check(
                "GPU cluster",
                "SKIP",
                f"not set up ({deployment.N('gpu.enabled')} is false, {deployment.N('gpu.login_host')} is blank)",
            )
        ]
    lines = (check or gpu.check)(cfg)
    out = []
    for line in lines:
        status, _, rest = line.partition(" ")
        out.append(
            Check(
                "GPU cluster",
                status if status in ("PASS", "FAIL", "WARN") else "WARN",
                rest or line,
                (
                    None
                    if status == "PASS"
                    else Steps(
                        (
                            doctor.rk_command("gpu-check"),
                            "says more;",
                            *doctor.section_parts("gpu"),
                        )
                    )
                ),
            )
        )
    return out


# --- the RAM disk (local) ---------------------------------------------------------------------------------------------


def ramdisk_exec(path: str) -> str:
    """ "" when a program written to the RAM disk runs, else why not: a tiny script is written there and run (the PRs'
    tests write helpers into their temporary folder, which is on the RAM disk, and run them).
    """
    import tempfile

    d = pathlib.Path(path)
    if not d.is_dir() or not os.access(d, os.W_OK):
        return ""  # not this user's to test (the mount check above says what is wrong)
    fd, name = tempfile.mkstemp(prefix=".rkmon-exec-", suffix=".sh", dir=d)
    try:
        with os.fdopen(fd, "w") as f:
            f.write("#!/bin/sh\necho ok\n")
        os.chmod(name, 0o700)
        p = subprocess.run([name], capture_output=True, text=True, timeout=10)
        return (
            "" if p.stdout.strip() == "ok" else f"a script there exits {p.returncode}"
        )
    except PermissionError:
        return "a script written there cannot run: the mount is noexec"
    except (OSError, subprocess.SubprocessError) as exc:
        return f"a script written there cannot run: {exc}"
    finally:
        try:
            os.unlink(name)
        except OSError:
            pass


def ramdisk_check(
    cfg: dict,
    store_dir: pathlib.Path | None = None,
    usage=None,
    cgroup: str = "/sys/fs/cgroup",
    mem_gb: float | None = None,
    exec_check=None,
) -> list[Check]:
    """Mounted at ramdisk.path, as large as ramdisk.size_gb, room for one review and one heavy job at the learned
    footprints, and inside a memory-limited container, a limit that holds it besides the work.
    """
    s = ramdisk.settings(cfg)
    if not s["enabled"]:
        return [
            Check(
                "ramdisk",
                "OFF",
                "off (host.resources.ramdisk_gib 0): the worktrees, builds and temporary files stay on the "
                "disk",
            )
        ]
    u = usage or ramdisk.usage(cfg)
    if not u["mounted"]:
        return [
            Check(
                "ramdisk",
                "FAIL",
                doctor.blocking(
                    f"no tmpfs mounted at {s['path'] or '(ramdisk.path is blank)'}"
                    + (f" (it is on {u['fs']})" if u["fs"] else ""),
                    "every review and build: nothing that goes on the RAM disk starts",
                ),
                _mount_fix("mounts it"),
            )
        ]
    from . import scaler

    est = ramdisk.SEED_GB
    if store_dir is not None:
        try:
            est = scaler.estimates(
                json.loads((store_dir / scaler.STATE).read_text(encoding="utf-8"))
            )
        except (OSError, ValueError):
            pass
    found = (
        f"{u['path']}: tmpfs of {u['total_gb']:.1f} GiB, {u['used_gb']:.1f} GiB used"
    )
    out = []
    if s["size_gb"] and u["total_gb"] < s["size_gb"] * ramdisk.FULL_SIZE_SHARE:
        remount = _mount_fix("mounts it at that size")
        out.append(
            Check(
                "ramdisk",
                "WARN",
                found + f"; {deployment.N('ramdisk.size_gb')} is {s['size_gb']:g}",
                Steps(
                    (
                        *remount.parts,
                        "; or lower the setting:",
                        *doctor.settings_parts(
                            "ramdisk.size_gb",
                            Described("the size mounted, in GiB"),
                        ),
                    )
                ),
            )
        )
    elif u["total_gb"] < est["pipe"] + est["heavy"]:
        out.append(
            Check(
                "ramdisk",
                "WARN",
                doctor.blocking(
                    found + f"; one review ({est['pipe']:.1f} GiB) and one build "
                    f"({est['heavy']:.1f} GiB) do not fit",
                    "reviews: they wait for room",
                ),
                doctor.settings_fix(
                    "ramdisk.size_gb", Described("a larger size, in GiB")
                ),
            )
        )
    else:
        # The room left, not the size: what is on it already (reviews and jobs in flight, files left behind) holds
        # its share.
        free = u.get("free_gb")
        free = u["total_gb"] - (u.get("used_gb") or 0) if free is None else free
        fit_p = int(max(free, 0) // est["pipe"])
        out.append(
            Check(
                "ramdisk",
                "PASS" if fit_p else "WARN",
                found
                + f", {free:.1f} GiB free: room for about {fit_p} more reviews at {est['pipe']:.1f} GiB, or "
                f"{int(max(free, 0) // est['heavy'])} builds at {est['heavy']:.1f} GiB"
                + ("" if fit_p else "; new reviews wait until the work on it finishes"),
                (
                    None
                    if fit_p
                    else Steps((doctor.rk_command("status"), "shows the work on it"))
                ),
            )
        )
    why = (exec_check or ramdisk_exec)(u["path"])
    if why:
        out.append(
            Check(
                "ramdisk executes programs",
                "FAIL",
                doctor.blocking(
                    f"{u['path']}: {why}",
                    "the tests a review runs on it: any helper a test writes to its temporary folder fails",
                ),
                _mount_fix("mounts it with exec (docker's tmpfs default is noexec)"),
            )
        )
    try:
        limit = (pathlib.Path(cgroup) / "memory.max").read_text().strip()
    except OSError:
        limit = ""
    if limit.isdigit():
        lim_gb = int(limit) / 2**30
        if lim_gb < coherence.least(u["total_gb"]).memory:
            room = coherence.least(math.ceil(u["total_gb"])).memory
            need = Described(
                f"at least {room}: the RAM disk's size plus room for the work"
            )
            out.append(
                Check(
                    "ramdisk and the memory limit",
                    "WARN",
                    doctor.blocking(
                        f"the container may use {lim_gb:.1f} GiB of memory, and the RAM disk alone may hold "
                        f"{u['total_gb']:.1f} GiB",
                        "the work: the container is stopped for its memory when the RAM disk fills",
                    ),
                    doctor.settings_fix("resources.memory_gb", need),
                )
            )
    return out


def _mount_fix(what: str) -> Steps:
    """The fix of the RAM disk's mount, as the operator types it on the host: the deploy settings file's RAM disk size,
    which the compose override is made from (WHAT: what the override does), then `<wrapper> confirm` and `<wrapper>
    apply` (doctor.apply_commands).
    """
    from . import deploy_tree

    return Steps(
        (
            f"the compose override that {what} is made from {deploy_tree.name('ramdisk.size_gb')} in your deploy "
            "settings file: set it there, then",
            *doctor.apply_commands(),
        )
    )


# --- all of them, cached ----------------------------------------------------------------------------------------------

REMOTE = (
    "github",
    "jira",
    "chat",
    "gateways",
    "claude_login",
    "gpu_cluster",
)


def marks(cfg: dict) -> dict:
    """{name: when it last changed} for each entry of the secrets registry (logins.found's mark: in the container, the
    writer's answer for what the writer keeps), never a value or a size; "" for an entry not set, or one that cannot be
    looked at. Part of the cache key: setting, replacing or removing a login or a secret checks again.
    """
    from . import logins, secrets_registry as sr

    try:
        entries = sr.entries(cfg)
    except Exception:  # noqa: BLE001 - no registry: no marks
        return {}
    out = {}
    for e in entries:
        try:
            out[e.name] = logins.found(cfg, e)["mark"]
        except Exception:  # noqa: BLE001 - that entry alone has no mark
            out[e.name] = ""
    return out


def _digest(cfg: dict) -> str:
    keys = (
        "operator",
        "jira",
        "atlassian",
        "teams",
        "model_bindings",
        "roles",
        "reviewer",
        "claude",
        "orchestrator",
        "gpu",
        "features",
        "paths",
    )
    return hashlib.sha1(
        json.dumps(
            dict(
                {k: cfg.get(k) for k in keys},
                _code=code_id(),
                _marks=marks(cfg),
            ),
            sort_keys=True,
            default=str,
        ).encode()
    ).hexdigest()[:16]


def code_id(root: pathlib.Path | None = None) -> str:
    """Which code runs (its place, and each module's size and time): part of the cache key, so the checks another
    code ran (before an apply or a start put another release in, say) are never this code's result.
    """
    root = pathlib.Path(root or paths.PACKAGE)
    h = hashlib.sha1(str(root.resolve()).encode())
    for f in sorted([root / "rkmon.py", *(root / "rkmon_lib").glob("*.py")]):
        try:
            s = f.stat()
        except OSError:
            continue
        h.update(f"{f.name} {s.st_size} {s.st_mtime_ns}\n".encode())
    return h.hexdigest()[:16]


def remote_checks(
    cfg: dict,
    cache_dir: pathlib.Path | None,
    fresh: bool = False,
    clock=time.time,
    checks: dict | None = None,
) -> tuple[list[Check], str]:
    """The remote checks (REMOTE), from the cache when it is under CACHE_S seconds old and for this configuration,
    else run now and cached. Returns them and a line saying which."""
    fns = checks or {name: globals()[name] for name in REMOTE}
    path = cache_dir / CACHE if cache_dir else None
    key = _digest(cfg)
    if path and not fresh:
        try:
            c = json.loads(path.read_text(encoding="utf-8"))
            age = clock() - float(c["at"])
            if c.get("key") == key and 0 <= age < CACHE_S:
                # each row is checked again as it is read: a cache that does not read is run again
                return (
                    [check_from_json(x) for x in c["checks"]],
                    f"service checks from {age / 60:.0f} min ago (cached {CACHE_S // 60} min; {again()} "
                    "checks again)",
                )
        except (OSError, ValueError, KeyError, TypeError):
            pass
    out = []
    for name, fn in fns.items():
        try:
            out += fn(cfg)
        except (
            Exception
        ) as exc:  # noqa: BLE001 - one service's check never stops the report
            out.append(
                Check(
                    name,
                    "WARN",
                    f"the check failed: {type(exc).__name__}: {str(exc)[:160]}",
                    Steps((again(), "checks again")),
                )
            )
    if path:
        try:
            whole_file.write(
                path,
                json.dumps(
                    {
                        "at": clock(),
                        "key": key,
                        "checks": [check_to_json(c) for c in out],
                    }
                ),
                0o600,
            )
        except OSError:
            pass
    return out, "service checks run now (cached for the next few minutes)"
