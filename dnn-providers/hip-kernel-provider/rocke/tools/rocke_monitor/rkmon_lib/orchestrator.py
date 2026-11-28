# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`rk orchestrate`: the container's orchestrator. One long-lived Claude Code session (the Claude Agent SDK) runs the
monitor skill unattended, where a person's Claude Code session runs `/rocke-monitor resume` and its tick loop.

- It connects, resuming the last session when there is one (state/orchestrator.json), and sends `/rocke-monitor resume`.
  A recorded session that is not on this host (NOT_HERE: a Claude conversation stays on the host that made it) is
  dropped, and a new session starts at once; the log says so in one line. Any other error ends the orchestrator,
  which the entry point starts again.
- The ticks. Every `monitor.tick_interval_minutes` minutes (default 10) it runs `rk tick` itself, counted from the last
  tick on record (state.json `last_tick`), so after a (re)start a tick that is due runs at once; the tick triage reads
  the output first: by default the toolkit's rule filter (rkmon_lib/tickfilter.py, `rk tick-filter`, as the live tick
  loop uses), with no model; with a triage model (the package's `models.tick`, an opt-in: rkmon_lib/triage.py), the
  model also does the routine work with read-only rk commands. The session gets a turn only for the lines that need
  Claude, an audit (REVIEW_READY) and every outward action among them: `/rocke-monitor tick given=<the tick's file>` and
  those lines, verbatim (`act_prompt`). A tick runs whether or not the session is busy; its lines wait until the session
  is idle, and a line the session already got waits REPEAT_MIN minutes before it is handed again (an `!! ` line comes
  back with every tick). For triage.DOWN_MIN minutes after a triage model that failed (the session gets that tick
  whole), the session gets the plain `/rocke-monitor tick` while it is idle and runs `rk tick` itself (the container's
  entry point runs `rk tick` when the last tick is too old: `rk tick-due`). A triage model that is named but not at hand
  leaves the rule filter deciding. `rk tick-now` brings the next tick forward: its request (rkmon_lib/control.py
  TICK_NOW) is taken at the next look, every POLL_S seconds, once no tick of the orchestrator's runs.
- The review driver (`rk review-driver`) is kept running from here, never by the session: it starts after each tick
  while review automation is enabled and the monitor is on, and at once again after it exits with REVIEW_READY, is
  killed, or runs out of rounds with work in flight (so the parallel reviews never wait for the next tick). Its
  REVIEW_READY and REVIEW_MANUAL lines reach the session as a tick's do; TICK_STALE brings the next tick forward. A
  review that waits on a verifier (review.waiting_on_verifier) and that no subagent of the session holds is handed to it
  again, its REVIEW_READY line verbatim, once the session is idle after its start and after each tick (`rehand`): a
  REVIEW_READY lost with an orchestrator killed meanwhile, or printed by a driver an earlier orchestrator left running,
  is never stranded.
- One reader takes every message for the whole connection (the SDK needs exactly one). Background subagents and shell
  jobs report in turns the CLI starts by itself (task notifications). A notification with no turn after it within
  NUDGE_S gets a short nudge.
- Permissions: Claude Code's dontAsk mode, which refuses every tool call the allow list (`allowed_tools`) does not
  allow. The list holds the local tools, the skill, the two agent types and, for each folder the skill and its agents
  write files in (`write_folders`: the work folders, the weekly reports' folder and the site lessons' folder), one rule
  that lets the session and its agents create and edit files anywhere in it (`write_rule`): their file tools write
  nowhere else. The session has no MCP server (the atlassian CLI is the one way to JIRA). GUARD, a
  PreToolUse hook on every Bash call (the subagents' included), refuses outward actions that are not `rk` commands:
  GitHub writes through gh, pushes to a git remote, network writes by curl or wget, ssh, scp and Slurm commands, and
  the atlassian CLI's own writes unless follow-up tickets are on (an `rk atlassian` call is rk's to weigh, on each
  call: rkmon.py write_gate). Every outward step goes through an rk command, which carries its feature gate (the
  feature's mode: nothing goes out in prototype) and the ledger. It refuses the rk commands only the operator or rk
  itself runs, after rk, rkmon.py or the writer's entry point (rkmon_lib/outward.py's session column): a proposed ticket
  filed or declined, a raw team chat message, the team dashboard's push, the web host's requests read, the Teams test
  card and a writer's secret set. It also refuses what would change the state the review's agents and jobs share
  (rkmon_lib/workguard.py): a git config write, a PR ref fetched without `+`, a worktree outside a review directory's
  wt_v*/wt_q* paths or not made a tree with rk pr-tree first, and a write into the checks' shared build or its tree,
  also in a script a shell reads from a here-document. WebFetch and WebSearch are not offered.
- JIRA. The session's reads and writes are `rk atlassian`, the atlassian commands the writer lists
  (rkmon_lib/writer.py): rk runs the CLI with its login, in the container as the writer, which asks the outward gate
  before each write. The session has no atlassian login (logins_env).
- Auth comes from the `claude` login (LOGIN: the key alone, `rkmon login claude`; `read_key`) and
  integrations.claude_code's settings (claude.gateway, claude.api_key_sent_as, claude.api_key_header,
  claude.api_key_scheme): `session_variables` makes the session's variables from the two, into the session's
  environment only, never a log. No model name is among them: each call names its model by
  its id. CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC is always on (SESSION_ENV). `request_headers` gives the headers
  Claude Code sends for those variables, which a claude binding sends too (rkmon_lib/bindings.py). `test_request` is
  the claude login's test call, one message sent as the session sends its requests (rk claude-test makes it, and in
  the container the doctor: rkmon_lib/reach.py claude_test).
  `secret_path` is where an older package kept the session's settings, the claude login's older place, whose shape
  rkmon_lib/claude_env_shape.py reads. The session
  runs on `orchestrator.model`, sent by its id (config.claude_model_id: a built-in model's, the latest Claude model's),
  at its role's effort (`effort.orchestrator` over `claude.effort`: the SDK's effort option).
  Its turns and its subagents' keep the prompt cache for `claude.prompt_cache_ttl` (1h by default) and
  `claude.subagent_prompt_cache_ttl` (5m), which `rk orchestrate` writes into the session's settings file beside
  autoCompactWindow before the session starts (session_settings, write_settings).
- Usage, in tokens only (no prices and no cost here; rkmon_lib/usage.py reads them):
  each turn's Claude tokens (input, output, cache read and written) go to state/claude_usage.jsonl with what started
  the turn (resume, tick, act, task notification, nudge) and the PRs it audited; for each subagent, the PR and round its
  brief names (a PreToolUse hook on the Agent tool), its start, its total from its task notification, and its split by
  kind from its transcript (the SubagentStop hook). The triage's work goes to state/triage.jsonl, and the reviewer
  model's tokens to <RUN_DIR>/tokens.jsonl.
- Its output, the container's log, is lean: per tick the summary line, the `!! ` lines, what the user should know,
  which lines went to the session, and the board only when it changed.
- SIGTERM ends it: the tick and the review driver it runs are ended (each runs in a process group of its own, which
  is signalled whole: its gh and git calls end too), the session disconnects, and subagents still running end with it.
  The next start resumes the session, and `/rocke-monitor resume` re-checks what was in flight. An orchestrator killed
  outright (kill -9) cannot end its session's CLI: every process of the session carries ORCH_MARK, and the next start
  ends the ones whose orchestrator is gone (reap_stale_sessions).

`check` says what is missing before any of it runs (`rk orchestrate --check`, the container's entry point).
"""

from __future__ import annotations

import asyncio
import json
import os
import pathlib
import posixpath
import re
import shutil
import signal
import threading
import time

from . import (
    config,
    custom_headers,
    features,
    first_pass,
    outward,
    paths,
    registry,
    tickfilter,
    triage,
    whole_file,
    workguard,
)
from . import secrets_registry as sr
from .store import Store, now, stopped

LOGIN = sr.CLAUDE_LOGIN  # the login that holds the session's key
# The prompt cache's lifetime is a setting of the session's settings file (session_settings), not of its environment.
SESSION_ENV = {
    "DISABLE_AUTOUPDATER": "1",
    "DISABLE_TELEMETRY": "1",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
    "BASH_DEFAULT_TIMEOUT_MS": "600000",
}
POLL_S = 15
NUDGE_S = 300  # a task notification with no turn after it this long gets a nudge
NUDGE = (
    "A background task finished (its notification is above). Do what the skill says for it: verify, post, or "
    "record what is left."
)
BUFFER = (
    16 * 1024 * 1024
)  # one large tool result must not end the session (the SDK's default is 1 MiB)
STATE = "orchestrator.json"
# The CLI's error for a session to resume that is not on this host: a Claude conversation stays on the host that made
# it, so a session recorded on another host is not found here (not_here)
NOT_HERE = "No conversation found with session ID"
USAGE = "claude_usage.jsonl"
DRIVER_TIMEOUT_S = (
    3 * 3600
)  # a hard limit: the review driver exits by itself after 120 rounds (about two hours)
TRIAGE_TIMEOUT_S = 900
DRIVER_AGAIN = "DRIVER: still running after"  # the driver ran out of rounds with work in flight (rkmon.py)
REPEAT_MIN = 60  # a line the session got is handed again only after this long (an `!! ` line: every tick)
TICKS = "ticks"  # state/ticks: each tick's output with its triage, and each review driver's, for the session to read
TICKS_KEPT = 300
SEEN = "tick_seen.txt"  # the standing lines the rule filter already handed over (tickfilter.actionable's `seen`)
# The folders (paths.*) in which the session and its agents create and edit files (write_folders): the monitor's work
# folders, and the weekly reports' folder, where the weekly lessons review writes its drafts (the skill's weekly.md).
WRITE_FOLDERS = (
    "tmp",
    "scratch",
    "probes",
    "workers",
    "lessons",
    "posts",
    "cache",
    "weekly_dir",
)
_PROGRESS = re.compile(r"\d\d:\d\dZ ")  # the review driver's progress lines

# Outward actions that are not rk commands, refused with the reason (GUARD). Read-only GitHub calls (gh api GETs, gh pr
# view, gh run view) and local commands pass.
_GH_WRITE = re.compile(
    r"\bgh\s+(?:pr\s+(?:review|comment|merge|edit|close|reopen|ready|create|lock)|"
    r"issue\s+(?:create|comment|edit|close|reopen|delete|transfer|lock|pin)|release\s|repo\s+(?:create|delete|edit|"
    r"fork|rename|archive|sync)|workflow\s+(?:run|enable|disable)|run\s+(?:rerun|cancel|delete)|label\s+(?:create|"
    r"edit|delete|clone)|secret\s|variable\s|gist\s|cache\s+delete|ruleset\s|auth\s+(?:login|logout|refresh|token|"
    r"setup-git))"
)
_GH_API_WRITE = re.compile(
    r"\bgh\s+api\b(?=[^|;&]*(?:-X\s*(?:POST|PUT|PATCH|DELETE)|--method[\s=]+(?:POST|PUT|PATCH|"
    r"DELETE)|\s-[fF]\s|\s--(?:raw-)?field\b|\s--input\b))",
    re.I,
)
_GIT_PUSH = re.compile(r"\bgit\b[^|;&]*\s(?:push|send-email)\b")
_NET_WRITE = re.compile(
    r"\bcurl\b(?=[^|;&]*(?:\s-X\s*(?:POST|PUT|PATCH|DELETE)|--request[\s=]+(?:POST|PUT|PATCH|"
    r"DELETE)|\s-d\b|\s--data|\s-F\s|\s--form\b|\s--upload-file\b|\s-T\s|\s--json\b))|"
    r"\bwget\b(?=[^|;&]*(?:--post-data|--post-file|--method|--body-data|--body-file))",
    re.I,
)
_REMOTE = re.compile(
    r"(?:^|[\s;&|(])(?:ssh|scp|sftp|sbatch|srun|salloc|scancel)(?=\s|$)"
)
# The bare atlassian CLI's writes, by its words (an `rk atlassian` call is not weighed here: _RK_ATLASSIAN).
_ATLASSIAN_WRITE = re.compile(
    r"\batlassian\b[^|;&]*\s(?:create|update|delete|comment|transition|assign|move|"
    r"property-set|attach|remote-link|link|edit|add)\b"
)
# An `rk atlassian` call, to the end of its command (an rk line as outward.not_the_sessions reads one): rk weighs it on
# each call (rkmon.py write_gate: a write's feature, a mirror's refusal and a stop) and refuses one the writer does not
# list, so the word list never reads its JQL.
_RK_ATLASSIAN = re.compile(
    outward.RK_LINE + re.escape(outward.ATLASSIAN) + outward.NAME_END + r"[^|;&\n]*"
)


def guard(
    command: str, tickets: bool = False, shared=(), held: str = "", login: str = ""
) -> str | None:
    """Why the session may not run this Bash command (None when it may). `tickets`: the session may file follow-up
    tickets now (tickets_on), so the bare atlassian CLI's writes may go to JIRA; HELD, when it may not, why
    (tickets_held). An `rk atlassian` call passes, whatever it is: rk weighs it on each call (_RK_ATLASSIAN). A found
    issue's ticket is proposed, never filed, by the session.
    `shared`: the checks' shared builds and trees (workguard.shared_paths), which the agents reuse read-only. `login`:
    the operator's GitHub login (config.operator_login), which the refusal of a command only a person or rk itself runs
    names (outward.not_the_sessions_why).
    """
    c = command or ""
    for rx, why in (
        (_GH_WRITE, "a GitHub write through gh"),
        (_GH_API_WRITE, "a GitHub write through gh api"),
        (_GIT_PUSH, "a push to a git remote"),
        (_NET_WRITE, "a network write by curl or wget"),
        (_REMOTE, "ssh, scp or a Slurm command (the GPU runner's rk commands do that)"),
    ):
        if rx.search(c):
            return (
                f"refused: {why}. Every outward action goes through an rk command (its feature, which sends "
                "nothing in prototype, and the ledger); read-only calls are fine"
            )
    # The commands only the operator or rk itself runs (rkmon_lib/outward.py: the rows the session may not run).
    name = outward.not_the_sessions(c)
    if name:
        from .doctor import wrapper_name

        return (
            f"refused: {outward.not_the_sessions_why(name, login)}: a found issue's "
            'ticket is proposed with rk ticket-draft (the admin dashboard\'s "Tickets waiting for you" card files '
            "it), a develop break's team chat message is rk teams-breakage N, a secret is set on the host "
            f"({wrapper_name()} secret NAME), and an in-depth review waits for the operator's approval on the Needs "
            "you list"
        )
    if _ATLASSIAN_WRITE.search(_RK_ATLASSIAN.sub(" ", c)) and not tickets:
        return (
            f"refused: an atlassian CLI write ({held or TICKETS_OFF}): rk atlassian files a follow-up ticket, and keeps "
            "it as an unsent item in prototype or standby; a found issue's ticket is proposed with rk ticket-draft, "
            "and the admin dashboard's Tickets card files it; reads are fine"
        )
    return workguard.check(c, shared)


def tickets_on(cfg: dict) -> bool:
    """The session may write to JIRA past rk (the bare atlassian CLI's writes): the follow-up tickets of merged PRs,
    filed without asking, while they are on and go out (features.sends: ticket creation live, and the monitor not in
    standby, read at each call). Otherwise its follow-up tickets go through `rk atlassian`, whose gate weighs each one
    (rkmon.py write_gate) and keeps it as an unsent item while it does not go out. A found issue's ticket is not the
    session's to file: it proposes it with rk ticket-draft, and the operator's Approve on the admin dashboard files it.
    """
    name = "follow-up-tickets"
    return features.sends(cfg, features.CAPABILITIES[name].feature) and features.active(
        cfg, name
    )


TICKETS_OFF = "follow-up tickets are off or not set up"


def tickets_held(cfg: dict) -> str:
    """Why the session may not write to JIRA past rk now (tickets_on false), in words: the mode or standby
    (features.held_words), else TICKETS_OFF; "" when it may."""
    if tickets_on(cfg):
        return ""
    feature = features.CAPABILITIES["follow-up-tickets"].feature
    return features.held_words(cfg, feature) or TICKETS_OFF


# --- settings ---------------------------------------------------------------------------------------------------------


def settings(cfg: dict) -> dict:
    """`claude` in config: tick_min, auto_compact_window, and the workspace (claude_workspace)."""
    return {
        "tick_min": tick_min(cfg),
        "workspace": (cfg.get("claude_workspace") or "").strip(),
        "auto_compact_window": compact_window(cfg),
    }


def tick_min(cfg: dict) -> int:
    """Minutes between ticks: `claude.tick_min`, your deploy settings' monitor.tick_interval_minutes (its default
    when unset, or not a whole number above 0); the tick loop that runs without the orchestrator (docker/entrypoint.sh)
    reads it through `rk tick-min`.
    """
    v = registry.number(cfg, "claude.tick_min", int, registry.FALSY)
    return v if v > 0 else registry.default("claude.tick_min")


def compact_window(cfg: dict) -> int:
    """`claude.auto_compact_window`: the tokens of context at which the session compacts itself (its default when
    unset or out of COMPACT_RANGE; config.problems names a bad value)."""
    v = (cfg.get("claude") or {}).get("auto_compact_window")
    ok = (
        isinstance(v, int)
        and not isinstance(v, bool)
        and config.COMPACT_RANGE[0] <= v <= config.COMPACT_RANGE[1]
    )
    return v if ok else registry.default("claude.auto_compact_window")


def cache_ttl(cfg: dict, key: str) -> str:
    """`claude.prompt_cache_ttl` or `claude.subagent_prompt_cache_ttl`: one of config.CACHE_TTLS, its default when
    unset or not one of them (config.problems names a bad value).
    """
    v = str((cfg.get("claude") or {}).get(key) or "").strip()
    return v if v in config.CACHE_TTLS else registry.default(f"claude.{key}")


def session_settings(cfg: dict) -> dict:
    """The Claude Code settings the deployment owns in the session's settings file: autoCompactWindow, promptCacheTtl
    and subagentPromptCacheTtl (the keys a workspace's .claude/settings.local.json sets them with too).
    """
    return {
        "autoCompactWindow": compact_window(cfg),
        "promptCacheTtl": cache_ttl(cfg, "prompt_cache_ttl"),
        "subagentPromptCacheTtl": cache_ttl(cfg, "subagent_prompt_cache_ttl"),
    }


def settings_file(cfg: dict) -> pathlib.Path | None:
    """The settings the session loads (setting_sources=["project"]): <claude_workspace>/.claude/settings.json; None
    without a workspace."""
    ws = settings(cfg)["workspace"]
    return (
        pathlib.Path(os.path.expanduser(ws)) / ".claude" / "settings.json"
        if ws
        else None
    )


def settings_gap(cfg: dict) -> str:
    """What the session's settings file lacks of session_settings ("" when it carries them all)."""
    p = settings_file(cfg)
    if p is None:
        return "claude_workspace is blank"
    try:
        have = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else {}
    except (OSError, ValueError):
        return f"{p} is not readable JSON"
    have = have if isinstance(have, dict) else {}
    gone = [f"{k} {v}" for k, v in session_settings(cfg).items() if have.get(k) != v]
    return f"{p} does not set {', '.join(gone)}" if gone else ""


def write_settings(cfg: dict) -> list[str]:
    """Merge session_settings into the session's settings file (settings_file): its other keys are kept, and a file
    that changes is backed up first. A file that is not a JSON object is left alone, and the line says so.
    """
    p = settings_file(cfg)
    if p is None:
        return ["the session's settings: not written (claude_workspace is blank)"]
    want = session_settings(cfg)
    have = {}
    if p.exists():
        try:
            have = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            return [
                f"the session's settings: {p} is not JSON ({getattr(exc, 'msg', exc)}): not changed; fix it, and "
                "the session's next start writes them"
            ]
        if not isinstance(have, dict):
            return [f"the session's settings: {p} is not a JSON object: not changed"]
    said = ", ".join(f"{k} {v}" for k, v in want.items())
    if all(have.get(k) == v for k, v in want.items()):
        return [f"the session's settings: {p} already sets {said}"]
    out = []
    if p.exists():
        backup = p.with_name(
            f"{p.name}.bak-{time.strftime('%Y%m%d%H%M%S', time.gmtime())}"
        )
        shutil.copy2(p, backup)
        out.append(f"the session's settings: backed up {p} -> {backup.name}")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(dict(have, **want), indent=2) + "\n", encoding="utf-8")
    return out + [f"the session's settings: {p} sets {said}"]


def secret_path(cfg: dict) -> pathlib.Path:
    """Where an older package kept the session's settings, with the key among them: the claude login's older place that
    the one-time move copies the key from, in the monitor's secrets folder (secrets_registry.older_places). Its shape
    is rkmon_lib/claude_env_shape.py's to read.
    """
    return pathlib.Path(
        next(p for o, p in sr.older_places(cfg, sr.entry(cfg, LOGIN)) if o.copied)
    )


def gateway(cfg: dict) -> str:
    """The gateway Claude is reached through, as the deploy settings name it (integrations.claude_code.base_url,
    config claude.gateway); "" when they name none."""
    return str((cfg.get("claude") or {}).get("gateway") or "").strip()


def session_variables(cfg: dict, key: str) -> dict[str, str]:
    """The Claude session's variables that integrations.claude_code's settings and the key KEY make: a pure function,
    which reads no file, secret or environment, so a caller may give a marker for KEY.
    The settings are the configuration's claude.gateway, claude.api_key_sent_as, claude.api_key_header and
    claude.api_key_scheme.

    - ANTHROPIC_BASE_URL: the gateway (`gateway`), when there is one; with none, Anthropic's own API, it is not set.
    - How the key is sent (api_key_sent_as): "header", the default, ANTHROPIC_API_KEY; "bearer", ANTHROPIC_AUTH_TOKEN;
      "header_and_bearer", both, but ANTHROPIC_AUTH_TOKEN alone when a header is named.
    - A named header (api_key_header), sent as "header" or "header_and_bearer": ANTHROPIC_CUSTOM_HEADERS, that header
      with the key, after the word api_key_scheme gives when it gives one. A named header always comes with one of the
      two key variables. The word goes with a named header only: Anthropic's own header, x-api-key, takes the key alone.
    - A blank KEY: neither key variable, and no header.

    CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC is none of them: SESSION_ENV sets it to 1 for every session.
    """
    claude = cfg.get("claude") if isinstance(cfg.get("claude"), dict) else {}
    sent_as = str(claude.get("api_key_sent_as") or "").strip()
    header = str(claude.get("api_key_header") or "").strip()
    scheme = str(claude.get("api_key_scheme") or "").strip()
    out = {"ANTHROPIC_BASE_URL": gateway(cfg)} if gateway(cfg) else {}
    if not key:
        return out
    bearer = sent_as in ("bearer", "header_and_bearer")
    named = header if sent_as != "bearer" else ""
    if sent_as != "bearer" and not (bearer and named):
        out["ANTHROPIC_API_KEY"] = key
    if bearer:
        out["ANTHROPIC_AUTH_TOKEN"] = key
    if named:
        out["ANTHROPIC_CUSTOM_HEADERS"] = (
            f"{named}: {f'{scheme} ' if scheme else ''}{key}"
        )
    return out


def request_headers(variables: dict) -> dict[str, str]:
    """The headers Claude Code sends with each request for the session's VARIABLES (`session_variables`): the key in
    x-api-key for ANTHROPIC_API_KEY, `Authorization: Bearer <key>` for ANTHROPIC_AUTH_TOKEN, and each `Name: value`
    line of ANTHROPIC_CUSTOM_HEADERS (custom_headers.header_lines). Their values hold the key: never shown.
    """
    out = {}
    if variables.get("ANTHROPIC_API_KEY"):
        out["x-api-key"] = variables["ANTHROPIC_API_KEY"]
    if variables.get("ANTHROPIC_AUTH_TOKEN"):
        out["Authorization"] = f"Bearer {variables['ANTHROPIC_AUTH_TOKEN']}"
    for name, value in custom_headers.header_lines(
        variables.get("ANTHROPIC_CUSTOM_HEADERS") or ""
    ):
        if name and value:
            out[name] = value
    return out


TEST_PROMPT = (
    "Answer OK."  # the one message of the claude login's test call (test_request)
)


def test_request(cfg: dict, key: str) -> tuple[str, dict[str, str], bytes]:
    """The claude login's test call, sent as the Claude session sends its requests: one message
    (TEST_PROMPT) to the Messages API at the session's address (bindings.claude_url: integrations.claude_code.base_url,
    else Anthropic's own API), on the session's model by its id (config.claude_model_id), answered in one token, with
    the headers Claude Code sends for the session's variables and KEY (`session_variables`, `request_headers`).
    (url, headers, body): a pure function, which reads no secret and no environment. The headers hold KEY: never shown.
    """
    from . import bindings

    headers = {
        "anthropic-version": first_pass.llm_client().ANTHROPIC_VERSION,
        "content-type": "application/json",
        **request_headers(session_variables(cfg, key)),
    }
    body = {
        "model": config.claude_model_id(cfg),
        "max_tokens": 1,
        "messages": [{"role": "user", "content": TEST_PROMPT}],
    }
    url = f"{bindings.claude_url(cfg).rstrip('/')}/v1/messages"
    return url, headers, json.dumps(body).encode("utf-8")


def key_path(cfg: dict) -> pathlib.Path:
    """Where the claude login keeps the session's key: its place in the monitor's secrets folder
    (secrets_registry.path). paths.PathsError when no secrets folder is known."""
    return pathlib.Path(sr.path(cfg, sr.entry(cfg, LOGIN)))


def read_key(cfg: dict) -> str:
    """The session's key, the claude login's (`key_path`), blank space around it aside; "" when it is not set, cannot
    be read, or no secrets folder is known. Never shown."""
    try:
        return key_path(cfg).read_text(encoding="utf-8").strip()
    except (OSError, UnicodeDecodeError, paths.PathsError):
        return ""


def logins_env(cfg: dict) -> dict[str, str]:
    """Where gh's read login lives (GH_CONFIG_DIR), as rk gives its own children (config.env): the session's own gh
    calls, its subagents' too, run logged in, and only read. Nothing of the atlassian CLI's: the session's JIRA calls
    are `rk atlassian`, which runs the CLI with its login (in the container, as the writer).
    """
    from . import gh

    return dict(gh.settings(cfg)[1])


def session_env(cfg: dict) -> dict[str, str]:
    """The environment the session gets on top of the container's: SESSION_ENV, the read login's folder (logins_env),
    the variables integrations.claude_code's settings and the claude login's key make (`session_variables`,
    `read_key`), where Claude Code keeps its own state (CLAUDE_CONFIG_DIR, on the working volume), and the mark of the
    monitor's own writes (outward.own: the outward gate keeps a write of the session it stops as an unsent item, where
    it refuses one a person typed).
    """
    env = dict(SESSION_ENV, **logins_env(cfg), **session_variables(cfg, read_key(cfg)))
    env.setdefault(
        "CLAUDE_CONFIG_DIR",
        os.environ.get("CLAUDE_CONFIG_DIR")
        or f"{paths.get(cfg, 'working_dir')}/claude",
    )
    env[ORCH_MARK] = own_mark()
    env[outward.OWN_VAR] = "1"
    return env


# Every process of the session (the Claude Code CLI the SDK starts, and what it runs) carries this orchestrator's pid and
# start time, so the next orchestrator finds what one that died (kill -9, a crash) left behind (stale_sessions).
ORCH_MARK = "RKMON_ORCHESTRATOR"


def own_mark(pid: int | None = None) -> str:
    from . import procs

    pid = pid or os.getpid()
    return f"{pid}:{procs.start_time(pid)}"


def _pid_ns(pid: int) -> str:
    try:
        return os.readlink(f"/proc/{pid}/ns/pid")
    except OSError:
        return ""


def stale_sessions(
    table: dict | None = None, env=None, me: int | None = None, ns=_pid_ns
) -> list:
    """The processes of an earlier session whose orchestrator is gone: orphans (their parent is pid 1, the container's
    tini) that carry ORCH_MARK naming a pid that no longer runs with that start time. Never this orchestrator's own,
    never another user's, never one of another pid namespace (a container's, seen from its host).
    """
    from . import procs

    table = procs.table() if table is None else table
    env = env or procs.environ
    me = me or os.getpid()
    mine = own_mark(me)
    uid = os.getuid()
    here = ns(me)
    out = []
    for pid, s in table.items():
        if pid == me or s.get("uid") != uid or s.get("ppid") != 1:
            continue
        if ns(pid) != here:
            continue
        mark = env(pid).get(ORCH_MARK)
        if not mark or mark == mine:
            continue
        owner, _, start = mark.partition(":")
        live = table.get(int(owner)) if owner.isdigit() else None
        if not live or str(live.get("start")) != start:
            out.append({"pid": pid, "kind": "orphan", "what": mark, "cmd": ""})
    return out


def reap_stale_sessions(say=print, **kw) -> int:
    """End what an earlier session left running (stale_sessions): SIGTERM, then SIGKILL. How many."""
    from . import procs

    left = stale_sessions(**kw)
    if left:
        procs.terminate(left, grace=5.0)
        say(
            f"{now()} ORCH ended {len(left)} process(es) an earlier session left running "
            f"(pids {', '.join(str(e['pid']) for e in left[:10])})"
        )
    return len(left)


def check(cfg: dict, with_key: bool = True) -> list[str]:
    """What the orchestrator lacks (empty when it can run): review automation on (in the container the Claude session
    runs with it), the Claude workspace with the skill installed, and, WITH_KEY, the session's key (the claude login),
    each with the command that sets it as the operator types it here. The doctor's row of the session leaves the key
    out: the claude login's row checks it, with the login's test call (rkmon_lib/reach.py claude_login).
    """
    from . import doctor

    out = []
    if not config.operator_login(cfg):
        from . import agentic

        out.append(
            f"{agentic.LOGIN_MISSING} ({doctor.apply_words(edit='set it in your deploy settings file')})"
        )
    if not features.enabled(cfg, "review_automation"):
        out.append(
            "review automation is disabled (features.review_automation.enabled is false): the Claude session runs "
            "only while it is enabled"
        )
    ws = settings(cfg)["workspace"]
    if not ws:
        out.append(
            f"claude_workspace is blank (the container's layout sets it: {doctor.image_fix()})"
        )
    elif not (
        pathlib.Path(ws) / ".claude" / "skills" / "rocke-monitor" / "SKILL.md"
    ).is_file():
        out.append(
            f"the monitor's skill is not installed in {ws} ({doctor.section('agentic')} installs it)"
        )
    if with_key and not read_key(cfg):
        out.append(
            f"the Claude session has no key ({sr.how_to_set(sr.entry(cfg, LOGIN))})"
        )
    return out


def write_rule(folder: str, name: str = "the folder") -> str:
    """The allow rule that lets the session and its agents create and edit files anywhere in FOLDER, an absolute path
    (NAME: what the configuration calls it): an Edit rule, which covers Write too, whose path starts with `//`,
    Claude Code's mark of an absolute path (a path after one `/` starts at the settings file's folder). A folder that is
    not absolute is refused: no rule made from it allows what the monitor writes there.
    """
    if not folder.startswith("/"):
        raise ValueError(
            f"{name} is {folder!r}, not an absolute path: the session's file tools get no rule for it"
        )
    return f"Edit(//{folder.lstrip('/')}/**)"


def write_folders(cfg: dict) -> dict[str, str]:
    """Each folder the session and its agents create and edit files in, by what the configuration calls it: the
    folders of WRITE_FOLDERS, and the site lessons' folder (paths.site_lessons is the file the weekly lessons review
    edits, in the monitor's state)."""
    p = paths.resolve(cfg)
    out = {f"paths.{key}": p[key] for key in WRITE_FOLDERS if p.get(key)}
    if p.get("site_lessons"):
        out["paths.site_lessons' folder"] = posixpath.dirname(p["site_lessons"])
    return out


def allowed_tools(cfg: dict) -> list[str]:
    """The allow list of the dontAsk session (GUARD refuses outward Bash commands within it): the tools, and one
    write_rule for each folder the session writes files in (write_folders)."""
    out = [
        "Bash",
        "Read",
        "Glob",
        "Grep",
        "Agent",
        "TaskStop",
        "TaskOutput",
        "ToolSearch",
        "Skill(rocke-monitor)",
        "Agent(rocke-review-verifier)",
        "Agent(rocke-code-quality-reviewer)",
    ]
    return out + [
        write_rule(folder, name) for name, folder in write_folders(cfg).items()
    ]


def options(cfg: dict, sdk, resume: str | None, log=None, session=None) -> object:
    """The session's ClaudeAgentOptions: its guard (`guard`), and the hooks that attribute each subagent's tokens to its
    review. `guard` reads whether the session may write to JIRA (tickets_on) at each call, so that standby, begun while
    the session runs, stops those writes too.
    """

    async def bash_guard(input_data, tool_use_id, context):
        cmd = ((input_data or {}).get("tool_input") or {}).get("command", "")
        held = tickets_held(cfg)
        why = guard(
            cmd,
            not held,
            workguard.shared_paths(paths.get(cfg, "workers")),
            held,
            config.operator_login(cfg),
        )
        if not why:
            return {}
        record(
            cfg,
            {
                "kind": "refused",
                "tool_use_id": tool_use_id,
                "why": why.split(".")[0],
                "agent": (input_data or {}).get("agent_type") or "main",
            },
        )
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": why,
            }
        }

    async def agent_brief(input_data, tool_use_id, context):
        ti = (input_data or {}).get("tool_input") or {}
        pr, rnd = brief_pr(f"{ti.get('description') or ''}\n{ti.get('prompt') or ''}")
        record(
            cfg,
            {
                "kind": "agent",
                "tool_use_id": tool_use_id or "",
                "subagent_type": ti.get("subagent_type") or "",
                "pr": pr,
                "round": rnd,
                "by": (input_data or {}).get("agent_type") or "main",
            },
        )
        if session is not None and pr and tool_use_id:
            session.agent_started(tool_use_id, pr)
        return {}

    async def subagent_stop(input_data, tool_use_id, context):
        d = input_data or {}
        record(
            cfg,
            dict(
                {
                    "kind": "subagent",
                    "agent_id": d.get("agent_id") or "",
                    "tool_use_id": tool_use_id or "",
                    "agent_type": d.get("agent_type") or "",
                },
                **transcript_usage(d.get("agent_transcript_path")),
            ),
        )
        return {}

    kw = dict(
        cwd=settings(cfg)["workspace"],
        setting_sources=["project"],
        permission_mode="dontAsk",
        allowed_tools=allowed_tools(cfg),
        disallowed_tools=["WebFetch", "WebSearch"],
        hooks={
            "PreToolUse": [
                sdk.HookMatcher(matcher="Bash", hooks=[bash_guard], timeout=60),
                sdk.HookMatcher(matcher="Agent|Task", hooks=[agent_brief], timeout=30),
            ],
            "SubagentStop": [
                sdk.HookMatcher(matcher=None, hooks=[subagent_stop], timeout=60)
            ],
        },
        model=config.claude_model_id(cfg),
        env=session_env(cfg),
        resume=resume,
        system_prompt={"type": "preset", "preset": "claude_code"},
        max_buffer_size=BUFFER,
        stderr=log or (lambda line: None),
    )
    effort = config.role_effort(cfg, "orchestrator")[0]
    if not effort:  # neither effort.orchestrator nor claude.effort: the model's default
        return sdk.ClaudeAgentOptions(**kw)
    try:
        return sdk.ClaudeAgentOptions(effort=effort, **kw)
    except (
        TypeError
    ):  # an SDK from before the effort option: the model's default effort
        return sdk.ClaudeAgentOptions(**kw)


# A subagent's brief names its review: rk review-prep's folder (<tmp>/v<N><round>: prep.REVIEW_DIR), else `PR #N` or
# `#N` and the round in words.
_BRIEF_PR = re.compile(r"\bPR\s*#?(\d{1,7})\b|#(\d{1,7})\b")
_BRIEF_ROUND = re.compile(
    r"\b(fu\d+|fr)\b|\b(first[- ]round)\b|\bfollow[- ]up\b(?:\s*#?(\d+))?", re.I
)


def brief_pr(text: str) -> tuple[int | None, str]:
    """The PR and round a subagent's brief names ((None, "") when it names none): fr, fu<k>, or fu for a follow-up
    whose number it does not say."""
    from . import prep  # a call-time import: prep loads most of the package

    m = re.search(rf"\b{prep.REVIEW_DIR.pattern}\b", text or "")
    if m:
        return int(m.group(1)), m.group(2)
    m = _BRIEF_PR.search(text or "")
    if not m:
        return None, ""
    r = _BRIEF_ROUND.search(text)
    rnd = (
        ""
        if not r
        else (
            r.group(1).lower()
            if r.group(1)
            else "fr" if r.group(2) else f"fu{r.group(3) or ''}"
        )
    )
    return int(m.group(1) or m.group(2)), rnd


def transcript_usage(path) -> dict:
    """A subagent's tokens by kind from its transcript (one JSON line per message part; a message counts once, by its
    id, at its last part), and how many messages it had; {} when the transcript cannot be read.
    """
    by_id: dict = {}
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    msg = (json.loads(line) or {}).get("message")
                except (ValueError, AttributeError):
                    continue
                if (
                    isinstance(msg, dict)
                    and msg.get("role") == "assistant"
                    and isinstance(msg.get("usage"), dict)
                ):
                    by_id[msg.get("id") or len(by_id)] = msg["usage"]
    except (OSError, TypeError):
        return {}
    out = {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0}
    for u in by_id.values():
        for k, v in _usage(u).items():
            out[k] += v
    return dict(out, messages=len(by_id))


# --- state and usage --------------------------------------------------------------------------------------------------


def load_state(cfg: dict) -> dict:
    try:
        return json.loads(
            (config.toolkit_state_dir(cfg) / STATE).read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return {}


def save_state(cfg: dict, st: dict) -> None:
    """The session's state, STATE in the state folder, written whole at the state's shared mode (worker/perms.py)."""
    p = config.toolkit_state_dir(cfg) / STATE
    p.parent.mkdir(parents=True, exist_ok=True)
    whole_file.write(
        p,
        json.dumps(st, indent=1) + "\n",
        paths.worker_module("perms").SHARED_FILE_MODE,
    )


def not_here(error) -> bool:
    """Is ERROR the CLI's answer that the session to resume is not on this host (NOT_HERE, in its text or its
    stderr)?"""
    if error is None:
        return False
    return NOT_HERE in f"{error} {getattr(error, 'stderr', None) or ''}"


def record(cfg: dict, rec: dict) -> None:
    """One line of state/claude_usage.jsonl (never a prompt, a key or a header)."""
    p = config.toolkit_state_dir(cfg) / USAGE
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict({"ts": now()}, **rec), ensure_ascii=False) + "\n")


def _usage(u: dict | None) -> dict:
    u = u or {}
    return {
        "input": int(u.get("input_tokens") or 0),
        "output": int(u.get("output_tokens") or 0),
        "cache_read": int(u.get("cache_read_input_tokens") or 0),
        "cache_write": int(u.get("cache_creation_input_tokens") or 0),
    }


# --- the session ------------------------------------------------------------------------------------------------------


class Session:
    """The supervisor's view of the session: idle or busy, what started the turn in flight and the PRs it audits, the
    last tick, the task notifications still waiting for their turn, the lines waiting for the session (`pending`) and
    when each was last handed over (`handed`), and until when the tick triage counts as down.
    """

    def __init__(self, cfg: dict, clock=time.time, say=print):
        self.cfg, self.clock, self.say = cfg, clock, say
        self.busy, self.kind, self.last_tick, self.waiting, self.turns = (
            False,
            "",
            0.0,
            {},
            0,
        )
        self.session_id = load_state(cfg).get("session_id") or None
        self.pending: dict[str, dict] = {}
        self.handed: dict[str, float] = {}
        self.down_until = 0.0
        self.driver_wanted = (
            True  # the review driver starts with the orchestrator, then after each tick
        )
        self.turn_prs: list[int] = []
        self.task_prs: dict[str, int] = (
            {}
        )  # a subagent's tool_use_id -> the PR its brief names
        self.task_at: dict[str, float] = {}  # a subagent's tool_use_id -> its start
        self.task_bg: set[str] = set()  # the subagents that run in the background
        self.task_done: set[str] = set()  # the subagents that ended
        self.notified: set[int] = (
            set()
        )  # the PRs of the task notifications the next turn answers
        # The reviews awaiting verification that no verifier holds are handed again once the session is idle: after
        # the start (an earlier orchestrator's REVIEW_READY may never have reached its session) and after each tick.
        self.rehand_due = True

    def verifying(self, pr: int, since: float | None) -> bool:
        """A subagent of this session holds PR `pr`: one still running, or one started at `since` (when the review
        became ready) or later, which the session ran for it; what is left of that review is the session's call.
        """
        return any(
            p == pr
            and (
                tid not in self.task_done
                or (since is not None and self.task_at.get(tid, 0.0) >= since)
            )
            for tid, p in self.task_prs.items()
        )

    def agent_started(self, tool_use_id: str, pr: int) -> None:
        """The Agent tool's hook: a subagent for PR `pr` starts."""
        self.task_prs[tool_use_id] = pr
        self.task_at[tool_use_id] = self.clock()

    def on_message(self, sdk, m) -> None:
        """Every message the reader takes."""
        name = type(m).__name__
        if name == "ResultMessage":
            origin = (
                (getattr(m, "origin", None) or {}).get("kind") or self.kind or "turn"
            )
            self.busy, self.kind = False, ""
            self.turns += 1
            self.session_id = getattr(m, "session_id", None) or self.session_id
            save_state(self.cfg, {"session_id": self.session_id, "updated": now()})
            prs = (
                sorted(self.notified)
                if origin == "task-notification"
                else self.turn_prs
            )
            if origin == "task-notification":
                self.waiting.clear()
                self.notified.clear()
            else:
                self.turn_prs = []
            # A subagent run in the foreground ended with its turn (one in the background ends by its notification).
            self.task_done |= set(self.task_prs) - self.task_bg
            if int(getattr(m, "num_turns", 0) or 0) == 0:
                return  # a resumed session's report of a task that did not survive (no turn ran)
            record(
                self.cfg,
                dict(
                    {
                        "kind": origin,
                        "session": self.session_id,
                        "subtype": getattr(m, "subtype", ""),
                        "num_turns": getattr(m, "num_turns", 0),
                        "duration_ms": getattr(m, "duration_ms", 0),
                    },
                    **_usage(getattr(m, "usage", None)),
                    **({"prs": prs} if prs else {}),
                ),
            )
            self.say(f"{now()} ORCH turn done ({origin}, {getattr(m, 'subtype', '')})")
        elif name == "TaskStartedMessage":
            if getattr(m, "tool_use_id", ""):
                self.task_bg.add(m.tool_use_id)
            record(
                self.cfg,
                {
                    "kind": "task_start",
                    "task_id": getattr(m, "task_id", ""),
                    "tool_use_id": getattr(m, "tool_use_id", "") or "",
                },
            )
        elif name == "TaskNotificationMessage":
            u = getattr(m, "usage", None) or {}
            record(
                self.cfg,
                {
                    "kind": "task",
                    "task_id": getattr(m, "task_id", ""),
                    "status": getattr(m, "status", ""),
                    "tool_use_id": getattr(m, "tool_use_id", ""),
                    "total_tokens": int(
                        (
                            u.get("total_tokens")
                            if isinstance(u, dict)
                            else getattr(u, "total_tokens", 0)
                        )
                        or 0
                    ),
                    "summary": str(getattr(m, "summary", "") or "")[:200],
                },
            )
            self.waiting[getattr(m, "task_id", "")] = self.clock()
            if getattr(m, "tool_use_id", ""):
                self.task_done.add(m.tool_use_id)
            pr = self.task_prs.get(getattr(m, "tool_use_id", "") or "")
            if pr:
                self.notified.add(pr)
        elif name in ("AssistantMessage", "UserMessage", "StreamEvent"):
            self.busy = True

    def triage_on(self) -> bool:
        """The orchestrator runs the ticks and the triage reads them (the rule filter, or a triage model that is set
        up): always, but for triage.DOWN_MIN minutes after a triage model that did not answer.
        """
        return self.clock() >= self.down_until

    def tick_due(self) -> bool:
        """Time for the orchestrator's own `rk tick`."""
        return (
            self.triage_on()
            and self.clock() - self.last_tick >= settings(self.cfg)["tick_min"] * 60
        )

    def due(self) -> str | None:
        """What to send the session now, if anything: a nudge for a notification left without its turn, the lines
        waiting for it, or (with no triage) the plain tick."""
        if self.busy:
            return None
        t = self.clock()
        if self.waiting and min(self.waiting.values()) + NUDGE_S <= t:
            return "nudge"
        if self.pending:
            return "act"
        if (
            not self.triage_on()
            and t - self.last_tick >= settings(self.cfg)["tick_min"] * 60
        ):
            return "tick"
        return None

    def add(self, lines: list[str], src: str, how: str = "") -> None:
        """Lines for the session from one tick's or driver run's file (`how`: "rules" when the rule filter kept them);
        a line it got less than REPEAT_MIN minutes ago waits, unless it starts with `!! `.
        """
        t = self.clock()
        for line in lines:
            if (
                line.startswith("!! ")
                or t - self.handed.get(line, float("-inf")) >= REPEAT_MIN * 60
            ):
                self.pending[line] = {"src": src, **({"how": how} if how else {})}

    def add_whole(self, src: str) -> None:
        """A whole tick for the session: the triage did not answer."""
        self.pending[f"(the whole tick in {src})"] = {"src": src, "whole": True}


PROMPTS = {
    "resume": "/rocke-monitor resume",
    "tick": "/rocke-monitor tick",
    "nudge": NUDGE,
}


def act_prompt(pending: dict[str, dict]) -> str:
    """The session's prompt for the lines waiting for it: the skill's tick, given the files the orchestrator wrote
    (each tick's whole output with its triage), then the lines to act on, verbatim."""
    files = ",".join(dict.fromkeys(p["src"] for p in pending.values()))
    whole = [line for line, p in pending.items() if p.get("whole")]
    lines = [line for line, p in pending.items() if not p.get("whole")]
    ruled = lines and all(
        p.get("how") == "rules" for p in pending.values() if not p.get("whole")
    )
    out = [f"/rocke-monitor tick given={files}"]
    if whole:
        out.append(
            "The orchestrator ran rk tick and the tick triage did not answer: handle the whole tick in "
            + ("that file" if len(whole) == 1 else "each of those files")
            + " as the tick procedure says, from step 2."
        )
    if lines:
        out.append(
            (
                "Then handle these lines too"
                if whole
                else (
                    "The orchestrator ran rk tick and the rule filter (rk tick-filter) kept the lines that need you. "
                    "Handle these lines"
                    if ruled
                    else "The orchestrator ran rk tick and the tick triage did the rest. Handle these lines"
                )
            )
            + " as the tick procedure says:"
        )
        out += lines
    return "\n".join(out)


def recorded_tick(cfg: dict, now_s: float) -> float:
    """The last tick on record (state.json `last_tick`, whoever ran it) in seconds since the epoch, never later than
    `now_s`; 0.0 when there is none. The orchestrator's tick clock starts from it, so after a (re)start a tick that is
    due already runs at once."""
    try:
        st = json.loads(
            (config.toolkit_state_dir(cfg) / "state.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return 0.0
    at = _epoch(st.get("last_tick")) if isinstance(st, dict) else None
    return min(at, now_s) if at is not None else 0.0


async def send(client, s: Session, kind: str) -> None:
    s.busy, s.kind = True, kind
    if kind == "tick":
        s.last_tick = s.clock()
    if kind == "resume":
        s.last_tick = recorded_tick(s.cfg, s.clock())
    if kind == "tick":
        s.driver_wanted = True
    if kind == "nudge":
        s.waiting.clear()
    prompt = PROMPTS.get(kind, "")
    if kind == "act":
        prompt = act_prompt(s.pending)
        t = s.clock()
        s.handed = {
            line: at for line, at in s.handed.items() if t - at < REPEAT_MIN * 60
        }
        s.handed.update(
            {line: t for line, p in s.pending.items() if not p.get("whole")}
        )
        s.turn_prs = sorted(
            {
                int(m.group(1))
                for line in s.pending
                if line.startswith("REVIEW_READY")
                for m in [re.search(r"#(\d+)", line)]
                if m
            }
        )
        s.say(f"{now()} ORCH send {kind} ({len(s.pending)} line(s))")
        s.pending = {}
    else:
        s.say(f"{now()} ORCH send {kind}")
    await client.query(prompt)


# --- the ticks and the review driver ----------------------------------------------------------------------------------


async def rk_command(args: list[str], timeout: float) -> tuple[int, str]:
    """`rk <args>` as a child process in a session of its own: its exit code and its output (stdout and stderr). Past
    `timeout`, or when the orchestrator stops, the child's whole process group is ended (SIGTERM, then SIGKILL), so
    the gh and git it started go with it, and TimeoutError or the cancellation is raised.
    """
    proc = await asyncio.create_subprocess_exec(
        str(paths.PACKAGE / "rk"),
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        stdin=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except BaseException:
        _signal_group(proc.pid, signal.SIGTERM)
        try:
            await asyncio.wait_for(proc.wait(), 10)
        except asyncio.TimeoutError:
            pass
        _signal_group(proc.pid, signal.SIGKILL)  # what is left of the group
        raise
    return proc.returncode, out.decode("utf-8", "replace")


def _signal_group(pgid: int, sig: int) -> None:
    """Signal a process group (a child started with start_new_session leads its own); gone already: nothing."""
    try:
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError):
        pass


def new_file(cfg: dict, kind: str) -> pathlib.Path:
    """A new file in state/ticks for one tick's or one driver run's output; the oldest beyond TICKS_KEPT go."""
    d = config.toolkit_state_dir(cfg) / TICKS
    d.mkdir(parents=True, exist_ok=True)
    for old in sorted(d.iterdir())[:-TICKS_KEPT]:
        old.unlink(missing_ok=True)
    stamp = now().replace("-", "").replace(":", "")
    return d / f"{stamp}-{kind}.md"


def driver_ok(cfg: dict) -> bool:
    """The review driver may run: review automation on, and the monitor on and not stopped."""
    if not features.on(cfg, "review-drafts"):
        return False
    try:
        st = json.loads(
            (config.toolkit_state_dir(cfg) / "state.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return False
    return bool(st.get("enabled")) and not stopped(st)


def _epoch(at) -> float | None:
    """A state time (YYYY-MM-DDTHH:MM:SSZ) as seconds since the epoch; None when it is not one."""
    import datetime as dt

    try:
        return (
            dt.datetime.strptime(str(at), "%Y-%m-%dT%H:%M:%SZ")
            .replace(tzinfo=dt.timezone.utc)
            .timestamp()
        )
    except ValueError:
        return None


def unheld_ready(cfg: dict, s: Session) -> list[str]:
    """The REVIEW_READY line of every review that waits on a verifier (review.waiting_on_verifier) and that no verifier
    of the session holds (Session.verifying). A review becomes ready in the review driver, whose REVIEW_READY reaches
    the session once; an orchestrator killed meanwhile, or a driver left running by an earlier one, loses it.
    """
    from . import review

    try:
        st = json.loads(
            (config.toolkit_state_dir(cfg) / "state.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        return []
    return [
        review.ready_line_of(cfg, r)
        for r in review.waiting_on_verifier(st)
        if str(r.get("pr", "")).isdigit()
        and not s.verifying(int(r["pr"]), _epoch(r.get("at")))
    ]


def rehand(cfg: dict, s: Session) -> list[str]:
    """Once the session is idle after its start and after each tick (Session.rehand_due): the reviews awaiting
    verification that no verifier holds, handed to it again as the driver printed them (a line it got less than
    REPEAT_MIN minutes ago waits, as every line does)."""
    s.rehand_due = False
    if not driver_ok(cfg):
        return []
    t = s.clock()
    lines = [
        x
        for x in unheld_ready(cfg, s)
        if x not in s.pending and t - s.handed.get(x, float("-inf")) >= REPEAT_MIN * 60
    ]
    if lines:
        path = new_file(cfg, "ready")
        path.write_text(
            f"# awaiting verification, no verifier has them, {now()}\n\n"
            + "\n".join(lines)
            + "\n",
            encoding="utf-8",
        )
        s.add(lines, str(path))
        s.say(
            f"{now()} READY again for the session: "
            + ", ".join(_short(x) for x in lines)
        )
    return lines


def _short(line: str) -> str:
    return " ".join(line.split()[:2])[:60]


def in_thread(fn, *args) -> asyncio.Future:
    """fn(*args) in a daemon thread, as an awaitable: a stop never waits for a model's answer (the default executor's
    threads are joined when the loop closes)."""
    loop = asyncio.get_running_loop()
    fut = loop.create_future()

    def settle(ok: bool, value) -> None:
        if fut.done():  # given up on (a timeout, a stop)
            return
        if ok:
            fut.set_result(value)
        else:
            fut.set_exception(value)

    def work() -> None:
        try:
            ok, value = True, fn(*args)
        except Exception as exc:  # noqa: BLE001 - handed to the awaiting task as it is
            ok, value = False, exc
        try:
            loop.call_soon_threadsafe(settle, ok, value)
        except RuntimeError:  # the loop closed meanwhile: nobody waits any more
            pass

    threading.Thread(target=work, daemon=True).start()
    return fut


def tick_timeout_s(cfg: dict) -> float:
    """How long an rk tick may run before it is ended and the session hears of it: config tick_timeout_min, the tick
    loop's limit too (rkmon_lib/idle.py)."""
    from . import idle

    return idle.settings(cfg)["tick_timeout_min"] * 60


async def tick(cfg: dict, s: Session, rk, chat=None) -> None:
    """One tick: `rk tick`, its triage, its file, the lines for the session, the record and the lean output."""
    limit = tick_timeout_s(cfg)
    try:
        rc, out = await rk(["tick"], limit)
    except asyncio.TimeoutError:
        rc, out = (
            2,
            f"!! CRITICAL TOOL failure: rk tick ran over {limit / 60:g} min (tick_timeout_min) and was ended",
        )
    lines, board = triage.parse(out)
    s.rehand_due = True  # every tick: a review awaiting verification that no verifier holds goes to the session
    if (
        lines
        and lines[0].startswith("TICK ")
        and any(
            w in lines[0]
            for w in (
                " monitor OFF",
                " monitor STOPPED",
                tickfilter.TICK_RUNNING,
                "TICK idle:",
            )
        )
    ):
        s.say(
            f"{now()} {lines[0]}"
        )  # it did nothing (or nothing changed): nothing to triage, no review driver
        return
    classes = triage.classify_all(lines, features.on(cfg, "incident-issues"))
    verdict, down = None, ""
    ruled = (
        bool(triage.unavailable(cfg)) and chat is None
    )  # no triage model at hand: the rule filter decides
    if ruled:
        claude = tickfilter.actionable(out, rc, config.toolkit_state_dir(cfg) / SEEN)
    else:
        posts = triage.recent_posts(Store(config.toolkit_state_dir(cfg)))
        try:
            verdict = await asyncio.wait_for(
                in_thread(triage.run, cfg, rc, classes, board, posts, chat),
                TRIAGE_TIMEOUT_S,
            )
        except (triage.Down, asyncio.TimeoutError) as exc:
            down = str(exc) or f"no answer in {TRIAGE_TIMEOUT_S // 60} min"
            s.down_until = s.clock() + triage.DOWN_MIN * 60
        claude = [] if down else triage.for_claude(classes, verdict)
    user = triage.for_user(classes, verdict)
    path = new_file(cfg, "tick")
    path.write_text(tick_text(rc, out, claude, user, verdict, down), encoding="utf-8")
    if down:
        s.add_whole(str(path))
    else:
        s.add(claude, str(path), "rules" if ruled else "")
    counts: dict[str, int] = {}
    for _line, c in classes:
        counts[c] = counts.get(c, 0) + 1
    triage.record(
        cfg,
        {
            "rc": rc,
            "classes": counts,
            "claude": len(claude),
            "down": down,
            "file": path.name,
            "by": "rules" if ruled else "model",
            **(
                {k: verdict[k] for k in ("ran", "requests", "tokens")}
                if verdict
                else {}
            ),
        },
    )
    s.say(
        f"{now()} "
        + (
            (verdict or {}).get("summary")
            or (lines[0] if lines else f"TICK (rk tick exit {rc})")
        )
    )
    for line in [x for x in lines if x.startswith("!! ")] + [
        x for x in user if not x.startswith("!! ")
    ]:
        s.say(f"  {line}")
    if down:
        s.say(
            f"  the tick triage did not answer ({down}): the session gets the whole tick, then the plain tick for "
            f"{triage.DOWN_MIN} min"
        )
    elif claude:
        s.say(f"  for the session: {', '.join(_short(x) for x in claude)}")
    if board and any(x.startswith("BOARD (changed)") for x in lines):
        s.say(board)
    s.driver_wanted = True


def tick_text(
    rc: int,
    out: str,
    claude: list[str],
    user: list[str],
    verdict: dict | None,
    down: str,
) -> str:
    """A tick's file, for the session: the lines it acts on, the triage's notes, and the tick's whole output."""
    t = [
        f"# rk tick, {now()} (exit {rc})",
        "",
        "## For the session",
        *([f"- {x}" for x in claude] or ["- nothing"]),
        "",
        "## The tick triage",
    ]
    if down:
        t.append(f"It did not answer ({down}): handle the whole tick.")
    elif verdict is None:
        t.append(
            "The rule filter (rk tick-filter, no model) kept the lines above; nothing else ran."
        )
        t += [f"- for the user: {x}" for x in user]
    else:
        t += [
            f"Summary: {verdict['summary']}",
            *(f"- did: {x}" for x in verdict["done"]),
            *(f"- ran: {x}" for x in verdict["ran"]),
            *(f"- for the user: {x}" for x in user),
        ]
    return "\n".join(
        t + ["", "## The tick's output", "", "```", out.rstrip(), "```", ""]
    )


async def drive(cfg: dict, s: Session, rk) -> None:
    """One run of the review driver: the lines the session acts on, chosen as the live tick loop's driver filter
    chooses them (`rk tick-filter --driver`: REVIEW_READY, RECHECK_DONE, REVIEW_DROP, REVIEW_MANUAL and any line the
    rule table says acts on), for the session; and what decides the next run: at once after REVIEW_READY, else after
    the next tick; TICK_STALE brings that tick forward."""
    rc, out = await rk(["review-driver"], DRIVER_TIMEOUT_S)
    lines = [
        x.rstrip() for x in out.splitlines() if x.strip() and not _PROGRESS.match(x)
    ]
    if any(x.startswith("TICK_STALE") for x in lines):
        s.last_tick = 0.0
    body = [x for x in lines if not x.startswith(("DRIVER:", "TICK_STALE"))]
    if (
        "---" in body
    ):  # the driver printed its acts itself, after its last advance (rkmon.py _driver_acts)
        acts = body[len(body) - body[::-1].index("---") :]
    else:
        acts = tickfilter.driver_lines("\n".join(body))
    claude = [x for x in acts if not x.startswith("DRIVER_EXIT")]
    classes = [
        (x, triage.classify(x, features.on(cfg, "incident-issues")))
        for x in lines
        if not x.startswith("TICK_STALE") and x != "---"
    ]
    if claude:
        path = new_file(cfg, "driver")
        path.write_text(
            f"# rk review-driver, {now()} (exit {rc})\n\n```\n{out.rstrip()}\n```\n",
            encoding="utf-8",
        )
        s.add(claude, str(path))
        s.say(f"{now()} DRIVER for the session: {', '.join(_short(x) for x in claude)}")
    for x, c in classes:
        if c == triage.USER and x not in claude:
            s.say(f"{now()} DRIVER {x}")
    if any(x.startswith("REVIEW_READY") for x in lines):
        s.driver_wanted = True
    elif rc < 0 or any(x.startswith(DRIVER_AGAIN) for x in lines):
        # Killed (a signal, an OOM kill) or out of rounds with reviews still in flight: the pipes would wait for the next
        # tick, up to claude.tick_min minutes; it starts again at the next poll instead.
        s.driver_wanted = True
        s.say(
            f"{now()} DRIVER ended ({'signal ' + str(-rc) if rc < 0 else 'out of rounds'}): started again"
        )


async def run(
    cfg: dict,
    sdk=None,
    stop: asyncio.Event | None = None,
    poll_s: float = POLL_S,
    say=print,
    clock=time.time,
    rk=rk_command,
    chat=None,
) -> int:
    """The orchestrator until `stop` is set (SIGTERM, SIGINT) or the connection ends. 0 when stopped, 1 when the
    connection ended by itself (the entry point starts it again). `rk` runs an rk command (rk_command), `chat` is the
    tick triage's model (the reviewer endpoint by default)."""
    if sdk is None:
        import claude_agent_sdk as sdk  # the orchestrator's venv (docker/Dockerfile: /opt/rkmon/venv-claude)

        reap_stale_sessions(
            say
        )  # an earlier orchestrator killed outright left its session's CLI running
    stop = stop or asyncio.Event()
    s = Session(cfg, clock=clock, say=say)
    opts = options(cfg, sdk, s.session_id, log=lambda line: None, session=s)
    gone = triage.unavailable(cfg)
    say(
        f"{now()} ORCH start: workspace {settings(cfg)['workspace']}, model {config.claude_model_id(cfg)}, effort "
        f"{getattr(opts, 'effort', None) or 'the default'}, "
        f"{'resuming ' + s.session_id[:8] if s.session_id else 'a new session'}; the tick triage: "
        + triage.describe(cfg)
    )
    state = config.toolkit_state_dir(cfg)  # where rk tick-now's request waits
    # A call-time import: control loads the PR code's runner, which reads worker/ (rkmon_lib/prcode.py), and the
    # host's deploy side imports this module (rkmon_lib/bindings.py: a claude binding reached as the session is) with
    # no worker/ there.
    from . import control

    async def connected(opts) -> int:
        """The session OPTS give, until `stop` is set (0) or it ends by itself (1). A resume whose session is not on
        this host raises that error (not_here), wherever it shows: at the connect, or as the reader's.
        """
        jobs: dict[str, asyncio.Task] = {}
        async with sdk.ClaudeSDKClient(opts) as client:

            async def pump():
                async for m in client.receive_messages():
                    s.on_message(sdk, m)

            reader = asyncio.create_task(pump())
            await send(client, s, "resume")
            ended = 0
            try:
                while not stop.is_set():
                    try:
                        await asyncio.wait_for(stop.wait(), timeout=poll_s)
                    except asyncio.TimeoutError:
                        pass
                    if reader.done():
                        error = reader.exception()
                        if getattr(opts, "resume", None) and not_here(error):
                            raise error
                        say(f"{now()} ORCH the session ended: {error or 'no error'}")
                        ended = 1
                        break
                    for name, job in list(jobs.items()):
                        if job.done():
                            del jobs[name]
                            if not job.cancelled() and job.exception():
                                say(
                                    f"{now()} ORCH the {name} failed: {type(job.exception()).__name__}: "
                                    f"{str(job.exception())[:200]}"
                                )
                    if stop.is_set():
                        break
                    if "tick" not in jobs and control.take_tick(state):
                        s.last_tick = 0.0  # as TICK_STALE does: the next tick, on its own path, starts now
                        say(
                            f"{now()} ORCH rk tick-now asked for a tick: it is brought forward"
                        )
                    if "tick" not in jobs and s.tick_due():
                        s.last_tick = clock()
                        jobs["tick"] = asyncio.create_task(tick(cfg, s, rk, chat))
                    if "driver" not in jobs and s.driver_wanted:
                        s.driver_wanted = False
                        if driver_ok(cfg):
                            jobs["driver"] = asyncio.create_task(drive(cfg, s, rk))
                    if s.rehand_due and not s.busy:
                        rehand(cfg, s)
                    kind = s.due()
                    if kind:
                        await send(client, s, kind)
            finally:
                for job in jobs.values():
                    job.cancel()
                if jobs:
                    await asyncio.gather(*jobs.values(), return_exceptions=True)
                reader.cancel()
        return ended

    recorded = s.session_id
    try:
        ended = await connected(opts)
    except Exception as exc:  # noqa: BLE001 - only not_here's error is taken here
        if not (recorded and not_here(exc)):
            raise
        say(
            f"{now()} ORCH the recorded session {recorded[:8]} was not found on this host: a new session starts"
        )
        save_state(cfg, {"session_id": None, "updated": now()})
        # A new session, as at a start with none on record: nothing of the failed one carries over
        s = Session(cfg, clock=clock, say=say)
        ended = await connected(
            options(cfg, sdk, None, log=lambda line: None, session=s)
        )
    say(f"{now()} ORCH stopped after {s.turns} turn(s)")
    return ended


def main(cfg: dict, check_only: bool = False) -> int:
    """`rk orchestrate [--check]`. Before the session starts, the installed skill is rendered again when it is older
    than this configuration (agentic.refresh): it names the operator config.json names; and in the container the
    session's own settings go into its settings file (write_settings), which nothing else writes there. On a host the
    workspace is a person's, and its settings file stays theirs.
    """
    missing = check(cfg)
    if not missing and not check_only:
        from . import agentic, doctor

        try:
            lines = agentic.refresh(cfg)
            if doctor.deployment_image():  # on a host the workspace is a person's
                lines += write_settings(cfg)
            for line in lines:
                print(f"{now()} ORCH {line}", flush=True)
        except (ValueError, OSError) as exc:
            missing = [str(exc)]
    if missing:
        print("ORCHESTRATOR NOT SET UP:\n  - " + "\n  - ".join(missing))
        return 5
    if check_only:
        print(
            f"ORCHESTRATOR ready: workspace {settings(cfg)['workspace']}, model {config.claude_model_id(cfg)}, "
            f"effort {config.role_effort(cfg, 'orchestrator')[0]}, a tick every {settings(cfg)['tick_min']} min; the tick triage: "
            + triage.describe(cfg)
        )
        return 0
    stop = asyncio.Event()

    async def go():
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, stop.set)
        return await run(cfg, stop=stop, say=lambda line: print(line, flush=True))

    return asyncio.run(go())


def describe(cfg: dict) -> str:
    """The session's key in words, never its value: the claude login, set or not and where, and the variables the
    session gets (`session_variables`), by name. Its file's permissions are the doctor's secrets rows' (private_files).
    """
    try:
        where = key_path(cfg)
    except paths.PathsError as exc:
        return f"the claude login: not known ({exc})"
    if not read_key(cfg):
        return f"the claude login: not set ({where})"
    names = ", ".join(session_variables(cfg, "-"))
    return f"the claude login: set ({where}; not shown): the session gets {names}"
