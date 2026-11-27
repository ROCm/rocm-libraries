# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GitHub access through the `gh` CLI, under two logins: every gh call of the toolkit goes through this module.

- The read login reads the repo and is refused every write: every read here uses it (`rest`, `raw`, `graphql`,
  `run_read`, `exec_read`; tools/ghr.py is the shell tools' gh), and so do the Claude session, its agents and every job
  (`settings` gives them its folder as GH_CONFIG_DIR; `job_env` strips every other login from a job's environment).
- The write login is used by `write` alone, which only post.py's guarded write commands reach (post._gh_write), after
  their guards.

Where each login lives is the secrets registry's (rkmon_lib/secrets_registry.py, `path`): a folder a config key or an
environment variable names, else its own folder in <SECRETS_DIR>, where `rkmon login` keeps the login it makes.
`login_dirs` gives the folders gh uses: those, but for the read folder in the broker and before the split. A
configuration that names no folders at all keeps gh's own folder for both (no split). In the container the write
login's folder belongs to another user, the writer, so the session, its agents and the jobs cannot read it: the
GitHub write commands run as that user, with the image's code (docker/rk_write.py; rkmon_lib/writer.py routes them).

`configure(cfg)` applies the settings (rk does, once per command) and in a mirror (config mirror.enabled) turns every
gh write off. `_exec` is the one place that starts gh and `exec_read` the one that replaces a process with it
(tests/test_gh_split.py). In the container the gh they start is the GitHub request cap's counter (docker/rk_gh.py),
which counts each call (rkmon_lib/gh_cap.py); both wait first while the cap leaves no room, and say so on standard error
(gh_cap.wait_for_room), so that a time limit runs on gh's own time. `probe` tells the logins apart (ghlogins.py): two
writes GitHub refuses either way, a comment with no body and a branch with no name, each answered 422 for a login that
may make it and 403 for one that may not; a login that may make either can write. `login_with_token` logs a token in
to a folder (`rk read-login`: ghlogins.set_read).
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time

MIN_VERSION = (2, 40)  # gh auth status naming the account per host; tested with 2.46
BIN, READ_DIR, WRITE_DIR, WRITES_OFF, CONFIGURED, UNSPLIT = "gh", "", "", "", False, ""
# A token in one of these overrides GH_CONFIG_DIR: never passed on.
TOKEN_VARS = (
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "GH_ENTERPRISE_TOKEN",
    "GITHUB_ENTERPRISE_TOKEN",
)
# What a job never gets: a login folder other than the read one, a token, and the atlassian CLI's settings.
JOB_DROP = ("GH_CONFIG_DIR", *TOKEN_VARS, "ATLASSIAN_CONFIG_DIR", "ATLASSIAN_PROXY_URL")
# The model credentials, which a job that asks no model never gets either (job_env): the model gateways' and the Claude
# session's variables (their keys, headers and addresses, Claude Code's login folder), by these prefixes, and any name
# shaped like a secret's (worker/pr_env.py secret_shaped), since a model binding's key may have any name.
MODEL_CREDENTIALS = ("ANTHROPIC_", "CLAUDE_", "OPENAI_")
# Set in the environment of the rk the broker runs as the write user (docker/rk_write.py).
BROKERED = "RKMON_GH_BROKER"
HOSTS_FILE = "hosts.yml"  # the file in a login's folder that keeps gh's login


class GhError(RuntimeError):
    pass


def _expand(folder: str) -> str:
    return os.path.abspath(os.path.expanduser(folder)) if folder else ""


def _has_login(folder: str) -> bool:
    return os.access(os.path.join(folder, HOSTS_FILE), os.R_OK)


def login_dirs(cfg: dict, environ=None) -> dict:
    """{"read": folder, "write": folder}: the folders gh reads and writes with: each login's own (secrets_registry.path,
    in ENVIRON), "" for gh's own default folder (a configuration that names no folders). In the broker the read folder
    is the write one. With "unsplit": folder when there is no read login yet (the read folder is its home, the default
    one, and has no login) and the login made before the split (in the monitor's secrets folder, <SECRETS_DIR>/gh)
    is readable here: it reads, as it did, until `rkmon login github_read` adds the read login (ghlogins.py says
    so).
    """
    from . import paths, secrets_registry

    e = os.environ if environ is None else environ
    out, home = {}, {}
    for name, side in secrets_registry.GH_SIDES.items():
        entry = secrets_registry.entry(cfg, name)
        try:
            home[side] = secrets_registry.home(
                cfg, entry, e, keeper=secrets_registry.MONITOR
            )
        except paths.PathsError:
            home[side] = ""
        try:
            out[side] = secrets_registry.path(cfg, entry, e)
        except paths.PathsError:  # a configuration with no folders keeps gh's own
            out[side] = ""
    if e.get(BROKERED) == "1" and out["write"]:
        # The broker's user reads with its own login: the session's read folder is not its to read.
        out["read"] = out["write"]
    elif (
        home["read"]
        and out["read"] == home["read"]
        and not _has_login(out["read"])
        and _has_login(home["write"])
    ):
        out.update(read=home["write"], unsplit=home["write"])
    return out


def settings(cfg: dict) -> tuple[str, dict]:
    """(the gh binary, the environment the toolkit's children run gh with): the read login's folder, when one is set.
    The binary is paths.gh, else gh on PATH; in the package's image always gh on PATH, the GitHub request cap's counter
    (docker/rk_gh.py), so that every call there counts, the jobs' too (their RKMON_GH): config.problems refuses a set
    paths.gh there."""
    from . import doctor, paths

    p = paths.resolve(cfg)
    read = login_dirs(cfg)["read"]
    binary = "gh" if doctor.deployment_image() else p.get("gh") or "gh"
    return binary, ({"GH_CONFIG_DIR": read} if read else {})


def configure(cfg: dict) -> None:
    global BIN, READ_DIR, WRITE_DIR, WRITES_OFF, CONFIGURED, UNSPLIT
    from .mirror import (
        on as mirror_on,
    )  # a call-time import: mirror reads GitHub through this module

    # The broker's rk runs the image's gh on the broker's fixed PATH, never one the configuration names: the session
    # can edit the configuration, and that gh would run as the write user.
    BIN = "gh" if os.environ.get(BROKERED) == "1" else settings(cfg)[0]
    d = login_dirs(cfg)
    READ_DIR, WRITE_DIR, UNSPLIT = d["read"], d["write"], d.get("unsplit", "")
    WRITES_OFF = "a mirror (config mirror.enabled)" if mirror_on(cfg) else ""
    CONFIGURED = bool(cfg)


def _bin() -> str:
    """The gh binary: the configured one; in a tool a job runs (not configured), the job's RKMON_GH."""
    return BIN if CONFIGURED else os.environ.get("RKMON_GH") or "gh"


def _read_dir() -> str:
    """The read login's folder: the configured one; in a tool a job runs, the job's GH_CONFIG_DIR (the read folder:
    job_env)."""
    return READ_DIR if CONFIGURED else os.environ.get("GH_CONFIG_DIR", "")


def config_dir(side: str) -> str:
    """The configured folder of the read or the write login ("" for gh's own default)."""
    return _read_dir() if side == "read" else WRITE_DIR


def default_dir(environ=None) -> str:
    """The folder a gh that names none uses (gh's own rule): GH_CONFIG_DIR, else $XDG_CONFIG_HOME/gh, else
    ~/.config/gh."""
    e = os.environ if environ is None else environ
    if e.get("GH_CONFIG_DIR"):
        return _expand(e["GH_CONFIG_DIR"])
    if e.get("XDG_CONFIG_HOME"):
        return os.path.abspath(os.path.join(e["XDG_CONFIG_HOME"], "gh"))
    return _expand("~/.config/gh")


def _env(folder: str, base: dict | None = None) -> dict:
    """`base` (default: this process's environment) without a token variable, with `folder` as GH_CONFIG_DIR (gh's own
    default when it is "")."""
    e = {
        k: v
        for k, v in (os.environ if base is None else base).items()
        if k not in TOKEN_VARS
    }
    if folder:
        e["GH_CONFIG_DIR"] = folder
    else:
        e.pop("GH_CONFIG_DIR", None)
    return e


def environ() -> dict:
    """The environment the toolkit's own gh reads run with: the read login."""
    return _env(_read_dir())


def model_credential(name: str) -> bool:
    """Whether the variable `name` is one of MODEL_CREDENTIALS. What a secret's name looks like is worker/pr_env.py's,
    loaded here when asked (paths.worker_module), as paths is: the broker loads this module with the standard library
    only, and docker/rkmon's deploy path imports it."""
    from . import paths

    pr_env = paths.worker_module("pr_env")
    return name.startswith(MODEL_CREDENTIALS) or pr_env.secret_shaped(name)


def job_env(base: dict | None = None, model: bool = False) -> dict:
    """The environment for a job: `base` (default: this process's) without JOB_DROP, and, unless `model` (a job that
    asks a model: rkmon_lib/jobs.py MODEL_KINDS), without the model credentials; with the read login's folder as
    GH_CONFIG_DIR for what reads GitHub. The PR code a job runs gets worker/pr_env.py's list of it, nothing else.
    """
    e = {
        k: v
        for k, v in (os.environ if base is None else base).items()
        if k not in JOB_DROP and (model or not model_credential(k))
    }
    if _read_dir():
        e["GH_CONFIG_DIR"] = _read_dir()
    return e


def _exec(
    args: list[str],
    folder: str,
    timeout: float,
    errors: str = "strict",
    stdin_text: str | None = None,
) -> subprocess.CompletedProcess:
    """The one place the toolkit starts gh: under the login in `folder`, once the GitHub request cap has room (see the
    module's docstring); `stdin_text`, its standard input (a token goes there, never in an argument).
    """
    from . import gh_cap

    gh_cap.wait_for_room(args)
    return subprocess.run(
        [_bin(), *args],
        input=stdin_text,
        capture_output=True,
        text=True,
        timeout=timeout,
        errors=errors,
        env=_env(folder),
    )


# --- reads ------------------------------------------------------------------------------------------------------------

_WRITE_FLAGS = {"-X", "--method", "-f", "-F", "--field", "--raw-field", "--input"}
# gh subcommands that only read (`gh api` is checked by its flags).
READ_VERBS = {
    ("pr", "view"),
    ("pr", "diff"),
    ("pr", "checks"),
    ("pr", "list"),
    ("pr", "status"),
    ("issue", "view"),
    ("issue", "list"),
    ("run", "view"),
    ("run", "list"),
    ("repo", "view"),
    ("label", "list"),
    ("release", "view"),
    ("release", "list"),
    ("auth", "status"),
    ("search", "prs"),
    ("search", "issues"),
    ("search", "code"),
    ("search", "commits"),
    ("search", "repos"),
}
# gh api flags that take a value: the endpoint is the first other argument.
_API_VALUE_FLAGS = {
    "--jq",
    "-q",
    "-H",
    "--header",
    "-t",
    "--template",
    "--cache",
    "-p",
    "--preview",
    "--hostname",
}


def _only_get(args: list[str]) -> None:
    bad = [a for a in args if a in _WRITE_FLAGS or a.startswith("--method=")]
    if bad:
        raise GhError(f"refusing non-GET gh call: {bad}")


def _endpoint(args: list[str]) -> str:
    skip = False
    for a in args[1:]:
        if skip:
            skip = False
        elif a in _API_VALUE_FLAGS:
            skip = True
        elif not a.startswith("-"):
            return a
    return ""


def check_read(args: list[str]) -> None:
    """Refuse (GhError) a gh call that is not a read: a REST call with a method or body flag, a GraphQL mutation, a
    subcommand outside READ_VERBS."""
    if args[:1] == ["api"]:
        if _endpoint(args) == "graphql":
            if any(re.search(r"\bmutation\b", a) for a in args):
                raise GhError("refusing GraphQL mutation")
            return
        _only_get(args)
        return
    if tuple(args[:2]) not in READ_VERBS:
        raise GhError(f"refusing gh {' '.join(args[:2])}: not a read")


def run_read(
    args: list[str], timeout: float = 120, errors: str = "strict"
) -> subprocess.CompletedProcess:
    """One gh read under the read login, as it comes (no retry): check_read, then gh."""
    check_read(args)
    return _exec(args, _read_dir(), timeout, errors)


def exec_read(args: list[str]) -> None:
    """Replace this process with one gh read under the read login (tools/ghr.py): check_read, then exec, once the
    GitHub request cap has room (see the module's docstring)."""
    from . import gh_cap

    check_read(args)
    gh_cap.wait_for_room(args)
    os.execvpe(_bin(), [_bin(), *args], _env(_read_dir()))


# Every failure is retried but a JSON 4xx: GitHub answered and refused the request, and a retry gets the same answer.
# gh prints one as "gh: <message> (HTTP 4xx)", the message taken from the JSON body (401 Bad credentials, 404 Not
# Found, 422 No commit found); 429 (too many requests) passes in time, so it is retried. Everything else is retried,
# not only known patterns: each new kind of transient failure would otherwise fail a tick that passes when re-run (an
# HTTP/2 "stream error: stream ID 3; CANCEL; received from peer" mid-pagination; a non-JSON error page ("invalid
# character 'u' looking for beginning of value": "upstream connect error"); a truncated GraphQL response ("unexpected
# end of JSON input"); a DNS failure ("error connecting to api.github.com")) (lessons L-409, L-505).
_PERMANENT = re.compile(r"\(HTTP 4(?!29)\d\d\)|\bHTTP 4(?!29)\d\d: ")


def _run(
    args: list[str], timeout: int, attempts: int = 3, errors: str = "strict"
) -> str:
    """Run a gh read; retry every failure but a JSON 4xx (`transient`), with backoff. `errors` decodes its output."""
    err = ""
    for i in range(attempts):
        try:
            p = run_read(args, timeout, errors)
            if p.returncode == 0:
                return p.stdout
            err = p.stderr.strip()[:400] or f"exit {p.returncode} with no message"
        except subprocess.TimeoutExpired:
            err = f"timed out after {timeout}s"
        if not transient(err) or i == attempts - 1:
            break
        time.sleep(5 * (3**i))
    raise GhError(f"gh {' '.join(a for a in args[:2])}: {err}")


def _decode_stream(text: str) -> list:
    """`gh api --paginate` prints one JSON document per page; concatenate list pages."""
    out, dec, i = [], json.JSONDecoder(), 0
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        val, i = dec.raw_decode(text, i)
        out.extend(val if isinstance(val, list) else [val])
    return out


def rest(path: str, jq: str | None = None, paginate: bool = False, timeout: int = 120):
    """GET a REST path. With paginate, returns the concatenated list of every page."""
    args = ["api", path]
    if paginate:
        args.insert(1, "--paginate")
    if jq:
        args += ["--jq", jq]
    _only_get(args)
    try:
        text = _run(args, timeout)
    except GhError as exc:
        raise GhError(f"{exc} [{path[:160]}]") from None
    if paginate:
        return _decode_stream(text)
    text = text.strip()
    return json.loads(text) if text and text[0] in "[{" else text


def raw(path: str, timeout: int = 120) -> str:
    """GET a REST path's body as it comes, neither stripped nor parsed (a CI job's log: `rk job-log`); bytes that are
    not UTF-8 read as U+FFFD."""
    args = ["api", path]
    _only_get(args)
    try:
        return _run(args, timeout, errors="replace")
    except GhError as exc:
        raise GhError(f"{exc} [{path[:160]}]") from None


def transient(err: str) -> bool:
    """Whether a gh error is one a retry (or a smaller request) can get past: anything but a JSON 4xx other than 429.
    The PR search asks a smaller page again on one; the CI-time pass stops on one, and records a SHA GitHub does not
    know only on a JSON 4xx."""
    return not _PERMANENT.search(err)


def graphql(query: str, timeout: int = 120, attempts: int = 3, **variables) -> dict:
    """Run a GraphQL query (never a mutation). Variables are passed as typed fields."""
    if re.search(r"\bmutation\b", query):
        raise GhError("refusing GraphQL mutation")
    args = ["api", "graphql", "-f", f"query={query}"]
    for k, v in variables.items():
        if v is None:
            continue
        args += ["-F" if isinstance(v, (int, bool)) else "-f", f"{k}={v}"]
    data = json.loads(_run(args, timeout, attempts))
    if data.get("errors"):
        raise GhError(f"graphql: {json.dumps(data['errors'])[:400]}")
    return data["data"]


# --- writes -----------------------------------------------------------------------------------------------------------


def write(args: list[str], timeout: float = 120) -> subprocess.CompletedProcess:
    """Run one gh write under the write login, as it comes: post.py's guarded writes call it, and nothing else
    (tests/test_gh_split.py). GhError, before gh runs, in a mirror, and when the read and the write login are one
    configured folder (with no read login yet, `login_dirs` "unsplit", one login does both, as before the split): the
    read login must never be able to write."""
    if WRITES_OFF:
        raise GhError(
            f"refused: gh {' '.join(args[:2])}: {WRITES_OFF} writes nothing to GitHub"
        )
    w, r = WRITE_DIR, _read_dir()
    if (
        w
        and r
        and os.environ.get(BROKERED) != "1"
        and not UNSPLIT
        and os.path.realpath(w) == os.path.realpath(r)
    ):
        raise GhError(
            f"gh.write_config_dir and gh.read_config_dir are the same folder ({w}): the read login must not "
            "write; nothing was written"
        )
    return _exec(args, w, timeout)


# --- a login from a token ---------------------------------------------------------------------------------------------


def login_with_token(folder: str, token: str, timeout: float = 60) -> str:
    """Log TOKEN in to the gh folder FOLDER: `gh auth login --with-token`, the token on its standard input, never in an
    argument, and kept in the folder's own file (HOSTS_FILE; --insecure-storage: never in a system keyring, which keeps
    one entry for each account, whatever folder logs it in), its one copy. The folder ends 700 and its files 600. ""
    once gh took it, else gh's own words why not; OSError or subprocess.SubprocessError when gh did not run.
    """
    os.makedirs(folder, mode=0o700, exist_ok=True)
    p = _exec(
        [
            "auth",
            "login",
            "--hostname",
            "github.com",
            "--with-token",
            "--insecure-storage",
        ],
        folder,
        timeout,
        stdin_text=token.strip() + "\n",
    )
    os.chmod(folder, 0o700)
    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and not os.path.islink(path):
            os.chmod(path, 0o600)
    if p.returncode == 0:
        return ""
    return (p.stderr or p.stdout).strip()[:200] or f"gh exited {p.returncode}"


# --- which login a folder has -----------------------------------------------------------------------------------------

LOGGED_IN = re.compile(r"Logged in to (\S+) (?:account|as) ([A-Za-z0-9-]+)")


def logged_in_as(status: str) -> tuple[str, str]:
    """(server, login) that `gh auth status`'s answer STATUS says gh is logged in to, the first it names; ("", "")
    when it names none. The one reader of that answer (rk doctor's rows)."""
    m = LOGGED_IN.search(status or "")
    return (m.group(1), m.group(2)) if m else ("", "")


_HTTP = re.compile(r"\(HTTP (\d{3})\)|\bHTTP (\d{3})\b")


def probe_issue(cfg: dict) -> int:
    """`gh.probe_issue`: an issue or pull request of the repository the probe comments on (nothing is made)."""
    from . import registry

    return registry.number(cfg, "gh.probe_issue", int, registry.FALSY)


def _answer(p: subprocess.CompletedProcess) -> tuple:
    """(may, http, words): one of the probe's writes as gh answered it. `may`: True when GitHub would make it (422, or
    made: never expected), False when it refuses it (403, or no login), None when that is not known.
    """
    text = f"{p.stderr}\n{p.stdout}"
    m = _HTTP.search(text)
    http = int(m.group(1) or m.group(2)) if m else None
    if p.returncode == 0:
        return True, http, "accepted"
    if http == 422:
        return True, http, "HTTP 422 (validation): allowed"
    if http == 403:
        return False, http, "HTTP 403: refused"
    if (
        http == 401 or "auth login" in text
    ):  # gh asks for a login when the folder has none
        return False, http, "not logged in, or the token is expired or revoked"
    return (
        None,
        http,
        (p.stderr.strip() or p.stdout.strip() or f"exit {p.returncode}")[:200],
    )


def probe(folder: str, repo: str, issue: int, timeout: float = 60) -> dict:
    """The login in gh folder `folder` and whether it may write to `repo`. GitHub is asked for two writes it refuses
    either way, so nothing is made: a comment with no body on `issue`, and a branch with no name and no commit. It
    answers each with 422 (validation) when the login may make it, 403 when it may not.
    {"dir", "login", "http": the comment's HTTP status, "can_comment", "can_push", "can_write": whether it may make
    either, each True | False | None (not known), "why", "held": whether the folder has a login (its HOSTS_FILE), None
    when this user cannot read it}. A folder that has none is not probed.
    """
    out = {
        "dir": folder,
        "login": None,
        "http": None,
        "can_comment": None,
        "can_push": None,
        "can_write": None,
        "why": "",
        "held": True,
    }
    if not os.path.isfile(os.path.join(folder, HOSTS_FILE)):
        unreadable = os.path.exists(folder) and not os.access(folder, os.R_OK | os.X_OK)
        out.update(
            can_comment=False,
            can_push=False,
            can_write=False,
            held=None if unreadable else False,
            why=(
                f"{folder} is not readable by this user"
                if unreadable
                else f"no gh login in {folder}"
            ),
        )
        return out
    try:
        who = _exec(["api", "user", "--jq", ".login"], folder, timeout)
        out["login"] = (who.stdout.strip() or None) if who.returncode == 0 else None
        comment = _exec(
            [
                "api",
                "-X",
                "POST",
                f"repos/{repo}/issues/{issue}/comments",
                "-f",
                "body=",
            ],
            folder,
            timeout,
        )
        push = _exec(
            ["api", "-X", "POST", f"repos/{repo}/git/refs", "-f", "ref=", "-f", "sha="],
            folder,
            timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        out["why"] = f"gh did not run: {str(exc)[:160]}"
        return out
    comments, out["http"], said = _answer(comment)
    pushes, _http, pushed = _answer(push)
    out.update(can_comment=comments, can_push=pushes)
    if comments or pushes:
        out["can_write"] = True
    elif comments is False and pushes is False:
        out["can_write"] = False
    out["why"] = (
        f"a comment and a push: {said}"
        if said == pushed
        else f"a comment: {said}; a push: {pushed}"
    )
    return out


# --- every GitHub call, and the permissions it needs ------------------------------------------------------------------

READ, WRITE = "read", "write"  # the logins, by their side (secrets_registry.GH_SIDES)
BOTH = (READ, WRITE)
ANY_READ = "gh *"  # tools/ghr.py: the read its command line names
# The permissions a fine-grained token gives on the organization, by their names: a token's page lists them under
# Organization permissions. Every other permission CALLS names is the repository's (on <org>/<repo>).
ORGANIZATION_PERMISSIONS = ("Members",)
# Each GitHub call of the monitor and of the writer, once (tests/test_gh_calls.py finds each one in the code): the
# logins that make it, the call, what it is for, and the permissions it needs, as a fine-grained token names them (on
# <org>/<repo>, but ORGANIZATION_PERMISSIONS on the organization). A call is `METHOD path` ({...}: a value), `graphql
# MODULE.NAME` (a GraphQL query: a module's constant) or `gh` and its command's two words.
# - The read login makes the monitor's reads: the functions above, the jobs' own gh (worker/fetch_pr.py), and the
#   session's and its agents' reads through tools/ghr.py, which are given nothing the other reads do not need.
# - The write login makes post.py's guarded writes and, in the broker, the reads of the functions that write
#   (login_dirs gives the broker the write login for both). Both logins make the probe's calls and their own login.
# - GitHub takes Issues or Pull requests for an issue's or a PR's comments and labels: each call lists what the monitor
#   reads or writes them on.
# - A check run needs Checks, which a fine-grained token cannot be given: a token reads a public repository's check
#   runs without it, and no fine-grained token reads a private one's.
CALLS = (
    # pull requests
    (
        BOTH,
        "GET repos/{repo}/pulls/{n}",
        "a PR: its state, head, labels and merge",
        (("Pull requests", "read"),),
    ),
    (
        BOTH,
        "GET repos/{repo}/pulls/{n}/commits",
        "a PR's commits",
        (("Pull requests", "read"),),
    ),
    (
        BOTH,
        "GET repos/{repo}/pulls/{n}/reviews",
        "a PR's reviews",
        (("Pull requests", "read"),),
    ),
    (
        BOTH,
        "GET repos/{repo}/pulls/{n}/reviews/{id}",
        "one of a PR's reviews: our account's review that rk review-close took",
        (("Pull requests", "read"),),
    ),
    (
        (READ,),
        "GET repos/{repo}/pulls/{n}/comments",
        "a PR's review comments",
        (("Pull requests", "read"),),
    ),
    (
        (READ,),
        "GET repos/{repo}/pulls/{n}/requested_reviewers",
        "a PR's review requests, when the PR search cannot read them",
        (("Pull requests", "read"),),
    ),
    (
        (READ,),
        "GET repos/{repo}/issues/{n}/timeline",
        "when a PR's head was pushed",
        (("Pull requests", "read"),),
    ),
    (
        BOTH,
        "GET repos/{repo}/issues/{n}/comments",
        "a PR's or a develop issue's comments",
        (("Issues", "read"), ("Pull requests", "read")),
    ),
    (
        BOTH,
        "GET repos/{repo}/issues/comments/{id}",
        "one of a PR's comments: a post, the GPU tests' row",
        (("Pull requests", "read"),),
    ),
    (
        (READ,),
        "GET search/issues",
        "a label's merged PRs",
        (("Pull requests", "read"),),
    ),
    ((READ,), "gh pr view", "a PR, for a review's job", (("Pull requests", "read"),)),
    (
        (READ,),
        "gh pr diff",
        "a PR's diff, for a review's job",
        (("Pull requests", "read"),),
    ),
    (
        (READ,),
        "graphql snapshot._SEARCH",
        "the PR search: PRs, their review requests (a team's: Members), linked issues and checks",
        (
            ("Commit statuses", "read"),
            ("Issues", "read"),
            ("Members", "read"),
            ("Pull requests", "read"),
        ),
    ),
    (
        (READ,),
        "graphql snapshot._SLIM",
        "the short PR search: PRs, their labels and checks",
        (("Commit statuses", "read"), ("Pull requests", "read")),
    ),
    (
        (READ,),
        "graphql snapshot._THREADS",
        "a PR's review threads",
        (("Pull requests", "read"),),
    ),
    (
        (READ,),
        "graphql facts._THREADS",
        "a PR's review threads, for its facts",
        (("Pull requests", "read"),),
    ),
    (
        (READ,),
        "graphql facts._FORCE_PUSHES",
        "a PR's force pushes",
        (("Pull requests", "read"),),
    ),
    (
        (READ,),
        "graphql followups._THREADS",
        "a merged PR's review threads, for its follow-ups",
        (("Pull requests", "read"),),
    ),
    (
        (READ,),
        "graphql premises._THREADS",
        "a PR's open threads, for a post's premises",
        (("Pull requests", "read"),),
    ),
    (
        (READ,),
        "graphql stats._SEARCH",
        "the PRs of a period",
        (("Pull requests", "read"),),
    ),
    (
        (READ,),
        "graphql fetch_pr.query",
        "a PR's review threads, for a review's job",
        (("Pull requests", "read"),),
    ),
    # the develop issues and their labels
    (
        BOTH,
        "GET repos/{repo}/issues",
        "the develop issues: the open breakages, the issue a write made",
        (("Issues", "read"),),
    ),
    (
        BOTH,
        "GET repos/{repo}/issues/{n}",
        "a develop issue: its state, labels and assignees",
        (("Issues", "read"),),
    ),
    (
        BOTH,
        "GET repos/{repo}/labels/{name}",
        "a label, if it exists",
        (("Issues", "read"),),
    ),
    (
        (READ,),
        "graphql breakage._ISSUES",
        "the develop issues, searched",
        (("Issues", "read"),),
    ),
    # the code, its checks and CI
    ((READ,), "GET repos/{repo}/commits", "develop's commits", (("Contents", "read"),)),
    (
        (READ,),
        "GET repos/{repo}/commits/develop",
        "develop's head",
        (("Contents", "read"),),
    ),
    ((READ,), "GET repos/{repo}/commits/{sha}", "a commit", (("Contents", "read"),)),
    (
        (READ,),
        "GET repos/{repo}/compare/{base}...{head}",
        "what changed between two commits",
        (("Contents", "read"),),
    ),
    ((READ,), "graphql snapshot._BRANCH", "a branch's head", (("Contents", "read"),)),
    (
        (READ,),
        "GET repos/{repo}/rules/branches/develop",
        "develop's rules: its required checks",
        (),
    ),
    (
        (READ,),
        "GET repos/{repo}/commits/{sha}/status",
        "a commit's statuses",
        (("Commit statuses", "read"),),
    ),
    (
        (READ,),
        "GET repos/{repo}/commits/{sha}/check-runs",
        "a commit's check runs (Checks: above)",
        (),
    ),
    ((READ,), "GET repos/{repo}/actions/runs", "the CI runs", (("Actions", "read"),)),
    (
        (READ,),
        "GET repos/{repo}/actions/runs/{run}/jobs",
        "a CI run's jobs",
        (("Actions", "read"),),
    ),
    (
        (READ,),
        "GET repos/{repo}/actions/runs/{run}/attempts/{attempt}/jobs",
        "a CI run attempt's jobs",
        (("Actions", "read"),),
    ),
    (
        (READ,),
        "GET repos/{repo}/actions/jobs/{job}",
        "a CI job",
        (("Actions", "read"),),
    ),
    (
        (READ,),
        "GET repos/{repo}/actions/jobs/{job}/logs",
        "a CI job's log",
        (("Actions", "read"),),
    ),
    # the organization
    (
        (READ,),
        "GET orgs/{org}/teams/{team}/members",
        "a code-owner team's members",
        (("Members", "read"),),
    ),
    # the logins
    (BOTH, "GET user", "the probe: whose login a folder has", ()),
    (
        BOTH,
        "POST repos/{repo}/issues/{n}/comments",
        "the probe: an empty comment, refused either way: 422 when the login may comment, 403 when it may not",
        (),
    ),
    (
        BOTH,
        "POST repos/{repo}/git/refs",
        "the probe: a branch with no name, refused either way: 422 when the login may push, 403 when it may not",
        (),
    ),
    (BOTH, "gh auth login", "a login, made (rkmon login)", ()),
    (BOTH, "gh auth status", "a login, checked (rk doctor)", ()),
    ((READ,), ANY_READ, "the session's and its agents' reads", ()),
    # the writes
    (
        (WRITE,),
        "gh pr comment",
        "a review's post, a notice",
        (("Pull requests", "write"),),
    ),
    ((WRITE,), "gh pr review", "a change request", (("Pull requests", "write"),)),
    (
        (WRITE,),
        "gh pr merge",
        "auto-merge, turned on and off",
        (("Contents", "write"), ("Pull requests", "write")),
    ),
    (
        (WRITE,),
        "PATCH repos/{repo}/issues/comments/{id}",
        "a PR's comment, edited: the queue notice, the GPU tests' row",
        (("Pull requests", "write"),),
    ),
    ((WRITE,), "gh issue create", "a develop issue", (("Issues", "write"),)),
    ((WRITE,), "gh issue comment", "a develop issue's comment", (("Issues", "write"),)),
    (
        (WRITE,),
        "gh issue edit",
        "a develop issue's labels and assignees",
        (("Issues", "write"),),
    ),
    (
        (WRITE,),
        "gh label create",
        "a label the develop issues use",
        (("Issues", "write"),),
    ),
)


def permissions(side: str) -> tuple:
    """The permissions the login of SIDE (READ or WRITE) needs, as (permission, access) pairs in name order: Metadata
    to read, which every token has, and what its calls need (CALLS), write over read. The secrets registry's.
    """
    need = {"Metadata": "read"}
    for sides, _call, _what, perms in CALLS:
        if side in sides:
            for name, access in perms:
                if need.get(name) != "write":
                    need[name] = access
    return tuple(sorted(need.items()))
