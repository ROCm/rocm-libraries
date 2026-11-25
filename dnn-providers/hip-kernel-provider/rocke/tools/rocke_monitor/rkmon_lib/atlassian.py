# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The atlassian CLI: JIRA and Confluence from the command line, as the jira and confluence skills use it
(`atlassian --site <site> confluence page get ...`). Every container image carries it (build settings
atlassian_cli.package_url and atlassian_cli.package_sha256; docker/Dockerfile's atlassian stage).

- Its login lives in the atlassian login's folder of the secrets registry (`config_dir`: <SECRETS_DIR>/atlassian; in
  the container, the writer's volume, which only the writer reads), ATLASSIAN_CONFIG_DIR, where the CLI keeps its
  `credentials` and `sites` files, never in config.json: `env` names that folder as soon as a secrets folder is
  known, before any site is set (a login made before the site is set would go to a one-off container's home, which
  `--rm` throws away). The site (`atlassian.site`) comes from config. The configuration is the monitor's, which the
  session may edit, so the writer's CLI never takes a proxy from it. `env` is the environment the CLI runs with in
  this module's calls (`_run`, `listed`) and in rk doctor's checks of it (rkmon_lib/reach.py). The
  container's `atlassian` command, the writer's only, gives it its login's folder alone (docker/entrypoint.sh), and
  jobs and the worker tools get no atlassian setting (config.env). A login is never copied between machines:
  Atlassian rotates the refresh token, so two copies log each other out.
- Where this process does not read the login (`login_here`: in the container, every user but the writer), each call
  goes through the writer (`_run`: `rk atlassian`, one of the CLI's commands the writer lists, rkmon_lib/writer.py
  ATLASSIAN_COMMANDS), so the login never leaves the writer's volume. `listed` runs one of those commands with the
  login: the writer's half of `rk atlassian`, and the whole of it on a host.
- The sites the login reaches (`sites`: `atlassian site list --json`, else the CLI's `sites` file) are those the
  configured site is checked against (`wrong_sites`).
- Nothing here logs in (`rkmon login atlassian` runs the CLI's own login, as the writer: `login_fix`), and nothing
  reads the `credentials` file: `status` only reads the version, the login
  state and the active site. A write to JIRA comes from a caller of `_run` (tickets.py's filing) or of `listed` (the
  session's follow-up tickets, through `rk atlassian`, which asks the outward gate first: rkmon.py write_gate).
- The access token lasts an hour. The CLI refreshes it itself, through its refresh token, at its next real call (a
  read or a write), so an expired access token is still a login (`login_state`); nothing here refreshes it, and no
  token is ever printed.
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import shutil
import subprocess

from . import paths
from .checks import PROBLEMS, Check, Command, Described, Steps

BINARY = "atlassian"
IMAGE_RECORD = pathlib.Path(
    "/etc/rkmon/atlassian.json"
)  # what the image's build installed (docker/Dockerfile)
SITES_FILE = "sites"  # the CLI's list of the sites its login reaches, in its login folder (written at login)
# The CLI's login itself, in its login folder (never read here).
CREDENTIALS_FILE = "credentials"
# The features that need JIRA (and so the CLI's login): off while JIRA is not wanted.
JIRA_FEATURES = ("follow-up-tickets", "found-issue-tickets")


def settings(cfg: dict) -> dict:
    """`atlassian` in config: {"site"} (blank when not set)."""
    a = cfg.get("atlassian") or {}
    return {"site": (a.get("site") or "").strip()}


def jira_site(cfg: dict) -> str:
    """The site the CLI's JIRA calls name (`--site`): a filer's writes and the writer that keeps them to it
    (`outside`), check 6's reads (ticket_fields) and the CLI's status (`status`, `wrong_sites`). The CLI names a site
    by its name (site_name), which the deploy settings file gives by JIRA's address (its jira_url). "" when none is
    set."""
    return site_name(settings(cfg)["site"])


def site_name(address: str) -> str:
    """The name the CLI gives the site at ADDRESS: its host's first label, as `https://<site>.atlassian.net/` names
    `<site>`; a name is its own. The one rule for a site's name: JIRA's (jira_site), a listed site's that comes
    without one (`_site`), and the privacy check's (rkmon_lib/privacy.py)."""
    host = re.sub(r"^https?://", "", address.strip(), flags=re.IGNORECASE)
    return host.split("/", 1)[0].split(":", 1)[0].split(".", 1)[0]


def in_use(cfg: dict) -> bool:
    """The CLI is set up for this deployment once a site is named."""
    return bool(settings(cfg)["site"])


def site_key() -> str:
    """The deploy settings file's key that names the site (config atlassian.site), as messages name it."""
    from . import deploy_tree

    return deploy_tree.name("atlassian.site")


def missing(cfg: dict, login: bool = True) -> list[str]:
    """What a feature that files or reads through the CLI (the ticket features) lacks of it, each a features.Need: a
    site named (a setting), the CLI here, and, LOGIN, its login (its credentials in its folder, `config_dir`, as
    login_mark finds them; an expired access token is still a login, login_state), the last two set up as `cli_fix`
    says. No call to the CLI or the site.
    """
    from . import secrets_registry as sr
    from .features import Need

    if not in_use(cfg):
        return [
            Need.setting(
                "atlassian.site",
                "the JIRA site's address: the atlassian CLI files the tickets there",
            )
        ]
    if not shutil.which(BINARY):
        return [
            Need.fixed_by(
                "the atlassian CLI (every image built by this package carries it)",
                lambda environ: cli_fix(cfg, {"found": False}, environ),
            )
        ]
    if login and not login_mark(cfg):
        return [
            Need.secret(
                sr.entry(cfg, "atlassian"),
                "the atlassian CLI's login",
                lambda environ: cli_fix(
                    cfg, {"found": True, "logged_in": False}, environ
                ),
            )
        ]
    return []


def config_dir(cfg: dict) -> str:
    """Where the CLI keeps its login: the atlassian login's folder in the secrets registry (secrets_registry.path),
    <SECRETS_DIR>/atlassian, in the container the writer's volume; "" when no secrets folder is known.
    """
    from . import secrets_registry

    try:
        return secrets_registry.path(cfg, secrets_registry.entry(cfg, "atlassian"))
    except paths.PathsError:
        return ""


def env(cfg: dict) -> dict[str, str]:
    """The environment the CLI runs with here: its login folder whenever a secrets folder is known (whether or not a
    site is set yet: a login made first is kept there). The writer runs the CLI with its login, and the configuration
    is the monitor's: no proxy read from it ever carries the login.
    """
    d = config_dir(cfg)
    return {"ATLASSIAN_CONFIG_DIR": d} if d else {}


def login_mark(cfg: dict) -> str:
    """When the CLI's login last changed (a login, a refresh, a logout): its `credentials` file's time and inode, never
    its content or its size (logins.found: in the container, the writer's answer, as the writer keeps the login); ""
    when there is none. rk doctor's cached service checks are for one login state only.
    """
    from . import logins, secrets_registry

    return logins.found(cfg, secrets_registry.entry(cfg, "atlassian"))["mark"]


def login_here(cfg: dict) -> bool:
    """Whether this process reads the CLI's login itself (logins.here): on a host, and in the container the writer
    alone."""
    from . import logins, secrets_registry

    return logins.here(secrets_registry.entry(cfg, "atlassian"))


def _run(cmd: list[str], cfg: dict, timeout: int = 30) -> tuple[int, str]:
    """The CLI's command CMD (the CLI, then its words), within TIMEOUT seconds: (its exit code, its answer: stdout,
    then stderr). Here with `env` when this process reads the login (`login_here`), else through the writer, one of
    the commands it lists (`rk atlassian`; a command off its list is refused, exit 4). 127: no CLI here.
    """
    if not login_here(cfg):
        from . import writer

        p = writer.through(
            [writer.ATLASSIAN, f"{writer.TIMEOUT_OPTION}={timeout}", *cmd[1:]],
            timeout_s=timeout + writer.START_S,
        )
        return p.returncode, ((p.stdout or "") + (p.stderr or "")).strip()
    e = dict(os.environ, **env(cfg))
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=e)
        return p.returncode, (p.stdout + p.stderr).strip()
    except FileNotFoundError:
        return 127, ""
    except (OSError, subprocess.SubprocessError) as exc:
        return 126, str(exc)


# A line that may carry a token: never shown, and dropped from what the login's own commands answer through the writer.
TOKENISH = re.compile(r"[A-Za-z0-9_\-]{40,}")


def outside(cfg: dict, call) -> str:
    """Why the writer does not make CALL's write to JIRA (writer.held: a write goes to the configured site, `jira_site`,
    and project alone), in words; "" when it makes it, and for a read."""
    from . import tickets, writer

    return writer.held(
        call, jira_site(cfg), tickets.jira_project(cfg), tickets.epic(cfg)
    )


def listed(cfg: dict, words: list[str]) -> subprocess.CompletedProcess:
    """`rk atlassian WORDS`: one of the CLI's commands the writer lists (writer.atlassian_call: ValueError for any
    other, and for a write outside the configured site and project: `outside`), run here with `env` and the login,
    within its time limit; its stdout and stderr apart, as the CLI gave them, except that the login's own commands
    (`auth ...`) lose each line that looks like a token. 127: no CLI here; 124: no answer in time.
    """
    from . import writer

    call = writer.atlassian_call(words)
    why = outside(cfg, call)
    if why:
        raise ValueError(why)
    binary = shutil.which(BINARY)
    if not binary:
        return subprocess.CompletedProcess(
            call.argv, 127, "", "the atlassian CLI is not installed here"
        )
    try:
        p = subprocess.run(
            [binary, *call.argv],
            capture_output=True,
            text=True,
            timeout=call.timeout_s,
            env=dict(os.environ, **env(cfg)),
            stdin=subprocess.DEVNULL,
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(
            call.argv,
            124,
            "",
            f"the atlassian CLI gave no answer in {call.timeout_s} s",
        )
    except OSError as exc:
        return subprocess.CompletedProcess(
            call.argv, 126, "", f"{type(exc).__name__}: {exc}"
        )
    if call.command[:1] == ("auth",):

        def keep(text: str) -> str:
            return "".join(
                ln for ln in (text or "").splitlines(True) if not TOKENISH.search(ln)
            )

        p = subprocess.CompletedProcess(
            p.args, p.returncode, keep(p.stdout), keep(p.stderr)
        )
    return p


def _first_line(text: str) -> str:
    """The first line worth showing of the CLI's answer: never its update notice, never a line that looks like a
    token."""
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if (
            ln
            and not re.search(r"new version .* is available", ln, re.I)
            and not TOKENISH.search(ln)
        ):
            return ln[:120]
    return ""


_NOT_IN = re.compile(r"not (logged|authenticated)|no credentials", re.I)
_EXPIRED = re.compile(r"Token:\s*expired", re.I)
_LEFT = re.compile(r"Token:\s*valid\s*\((?:(\d+)h\s*)?(\d+)m remaining\)", re.I)


def login_state(rc: int, text: str) -> dict:
    """`atlassian auth status` read: {"logged_in": the CLI holds a login, "expired": its access token has expired,
    "minutes_left": the minutes that token has left (None: not said)}. An expired access token is still a login: the
    CLI refreshes it through its refresh token at its next real call, a read or a write (seen with 1.6.2: the read
    succeeds and the credentials are rewritten with an hour's token), while `auth status` itself exits 1 with
    "Token: expired". So only a missing or refused login is "not logged in"."""
    text = text or ""
    if _NOT_IN.search(text):
        return {"logged_in": False, "expired": False, "minutes_left": None}
    expired = bool(_EXPIRED.search(text))
    m = _LEFT.search(text)
    return {
        "logged_in": rc == 0
        or expired
        or bool(re.search(r"\blogged in\b", text, re.I)),
        "expired": expired,
        "minutes_left": (
            0
            if expired
            else (int(m.group(1) or 0) * 60 + int(m.group(2)) if m else None)
        ),
    }


def login_words(state: dict) -> str:
    """The login as the doctor's lines say it."""
    if not state["logged_in"]:
        return "not logged in"
    if state["expired"]:
        return "logged in; its access token has expired, and the CLI refreshes it at its next call"
    left = state["minutes_left"]
    return "logged in" + (
        f" (the access token has {left} min left)" if left is not None else ""
    )


def status(cfg: dict, run=None) -> dict:
    """What `rk doctor` and the logins' states (rkmon_lib/logins.py) show: {"found", "binary", "version",
    "logged_in", "expired", "minutes_left", "site_ok", "detail"} (login_state: an expired access token is still a
    login). Read-only: `atlassian version`, `atlassian auth status`, `atlassian --site <site> site active`, the site
    JIRA's (jira_site). No token is ever printed (the status lines are cut to their first line, and a line that looks
    like a token is dropped).
    """
    run = run or (lambda cmd: _run(cmd, cfg))
    binary = shutil.which(BINARY) or ""
    out = {
        "found": bool(binary),
        "binary": binary or BINARY,
        "version": "",
        "logged_in": False,
        "expired": False,
        "minutes_left": None,
        "site_ok": False,
        "detail": "",
    }
    if not binary:
        return out
    rc, text = run([binary, "version"])
    m = re.search(r"version\s+v?(\d+\.\d+(?:\.\d+)?)", text)
    out["version"] = m.group(1) if m else ""
    rc, text = run([binary, "auth", "status"])
    out.update(login_state(rc, text))
    first = next(
        (ln for ln in text.splitlines() if ln.strip() and not TOKENISH.search(ln)),
        "",
    )
    out["detail"] = first[:120]
    site = jira_site(cfg)
    if site and out["logged_in"]:
        rc, _text = run([binary, "--site", site, "site", "active"])
        out["site_ok"] = rc == 0
    return out


def _json_values(text: str):
    """Each JSON value the CLI's answer holds, one per line that starts one (its update notice and other lines
    aside)."""
    dec = json.JSONDecoder()
    for ln in (text or "").splitlines():
        ln = ln.strip()
        if ln[:1] in ("[", "{"):
            try:
                yield dec.raw_decode(ln)[0]
            except ValueError:
                continue


def _site(item) -> dict | None:
    """One site as the CLI lists it: {"name", "url", "id"}; None without a name or an address to take one from."""
    if not isinstance(item, dict):
        return None
    url = str(item.get("url") or "").strip()
    name = str(item.get("name") or "").strip()
    if not name and url:
        name = site_name(url)
    if not name:
        return None
    return {
        "name": name,
        "url": url,
        "id": str(item.get("id") or item.get("cloud_id") or ""),
    }


def parse_sites(text: str) -> list[dict]:
    """`atlassian site list --json`: a JSON list of {"id", "name", "url"} (or an object with "sites"); else its plain
    listing, one name per line (`* ` before the active one), up to its legend."""
    for v in _json_values(text):
        items = v.get("sites") if isinstance(v, dict) else v
        if isinstance(items, list):
            return [s for s in map(_site, items) if s]
    out = []
    for ln in (text or "").splitlines():
        if not ln.strip():
            continue  # blank lines anywhere
        m = re.fullmatch(r"\s*(?:\*\s+)?([A-Za-z0-9][\w.-]*)\s*", ln)
        if m:
            out.append({"name": m.group(1), "url": "", "id": ""})
        elif out:
            break  # the legend (`* = active site`), or anything after the list
    return out


def sites_file(cfg: dict) -> list[dict]:
    """The sites in the CLI's own `sites` file (its login folder; written at login): {profile: {"sites": [...]}}, the
    profile ATLASSIAN_PROFILE names, else DEFAULT; [] where this process does not read the login (`login_here`). The
    `credentials` file beside it is never read.
    """
    d = config_dir(cfg) if login_here(cfg) else ""
    if not d:
        return []
    try:
        data = json.loads((pathlib.Path(d) / SITES_FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    prof = (
        data.get(os.environ.get("ATLASSIAN_PROFILE") or "DEFAULT")
        if isinstance(data, dict)
        else None
    )
    items = prof.get("sites") if isinstance(prof, dict) else None
    return [s for s in map(_site, items if isinstance(items, list) else []) if s]


def sites(cfg: dict, run=None) -> tuple[list[dict], str]:
    """The sites the CLI's login reaches ([{"name", "url", "id"}], as the CLI lists them: `atlassian site list --json`,
    else its `sites` file), and why none could be read ("" when some were)."""
    run = run or (lambda cmd: _run(cmd, cfg))
    binary = shutil.which(BINARY)
    if not binary:
        why = "the atlassian CLI is not installed here"
    else:
        rc, text = run([binary, "site", "list", "--json"])
        found = parse_sites(text) if rc == 0 else []
        if found:
            return found, ""
        why = (
            f"atlassian site list: {_first_line(text) or f'exit {rc}'}"
            if rc
            else "atlassian site list names no site"
        )
    found = sites_file(cfg)
    return (found, "") if found else ([], why)


def wrong_sites(cfg: dict, run=None) -> list[str]:
    """The site the settings name (site_key, which Jira and Confluence both use) when the CLI's login does not reach
    it, in words with the sites it does reach; [] when it is among them (by name, site_name), or the login's sites
    cannot be read (the site check of `status` covers the site then)."""
    from . import deploy_tree

    found, _why = sites(cfg, run)
    if not found:
        return []
    names = {s["name"].lower() for s in found} | {
        site_name(s["url"]).lower() for s in found if s["url"]
    }
    site = jira_site(cfg).lower()
    if not site or site in names:
        return []
    return [
        f"{site_key()} {settings(cfg)['site']} is not a site your login reaches "
        f"({', '.join(sorted(s['name'] for s in found))}): fix it in {deploy_tree.FILE}, or log in to that site"
    ]


def image_record() -> dict:
    """The atlassian CLI the image's build installed ({} for an image without it)."""
    try:
        return json.loads(IMAGE_RECORD.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def login_fix(environ=os.environ, wrapper: str = "") -> Command:
    """The command that logs the CLI in, as the operator types it on the host: the container's login, made as the
    writer with the CLI's full scopes (`<wrapper> login atlassian`). WRAPPER: the wrapper's name, where the caller
    has it (check 6's, on the host); else the one ENVIRON gives (doctor.wrapper_name).
    """
    from . import doctor

    return Command((wrapper or doctor.wrapper_name(environ), "login", "atlassian"))


def cli_fix(cfg: dict, s: dict, environ=os.environ):
    """What makes the CLI ready, as the operator types it where ENVIRON says the fix is shown (a checks fix), by its
    status S (`status`'s "found" and "logged_in"): while it is not found, an image of this package, which carries it
    (doctor.image_fix); while it is not logged in, its login (login_fix); None once it is both. The CLI's row
    (check_line) and the row that reads through the CLI (rkmon_lib/reach.py: the JIRA project) fix by it.
    """
    from . import doctor

    if not s["found"]:
        return doctor.image_fix(environ)
    if not s["logged_in"]:
        return login_fix(environ)
    return None


def check_line(cfg: dict, s: dict | None = None) -> Check:
    """The CLI's row for rk doctor (a checks row named "atlassian CLI"): SKIP while no site is set; WARN when
    the CLI, its login or the site is missing (it blocks only the JIRA steps that use the CLI).
    """
    from . import doctor, writer

    name = "atlassian CLI"
    if not in_use(cfg):
        return Check(name, "SKIP", f"not set up ({site_key()} is blank)")

    def warn(found: str, fix) -> Check:
        return Check(
            name,
            "WARN",
            doctor.blocking(found, "the JIRA steps that use the CLI"),
            fix,
        )

    s = s or status(cfg)
    site = settings(cfg)["site"]
    if not s["found"]:
        return warn(
            "missing"
            + (" (the image's build did not install it)" if not image_record() else ""),
            cli_fix(cfg, s),
        )
    found = f"{s['binary']} {s['version'] or '(version unknown)'}; login kept in {config_dir(cfg) or '?'}"
    if not s["logged_in"]:
        return warn(found + "; not logged in", cli_fix(cfg, s))
    site_fix = Steps(
        (
            doctor.rk_command(writer.ATLASSIAN, "site", "list"),
            "names the sites your login reaches;",
            *doctor.settings_parts(
                "atlassian.site", Described("the address of one of them")
            ),
        )
    )
    if not s["site_ok"]:
        return warn(found + f"; logged in, but site {site} does not answer", site_fix)
    wrong = wrong_sites(cfg)
    if wrong:
        return warn(found + f"; {login_words(s)}; " + "; ".join(wrong), site_fix)
    return Check(name, "PASS", found + f"; {login_words(s)}; site {site}")


def summary_row(cfg: dict) -> str:
    """The configuration summary's line (rkmon_lib/summary.py; never a token)."""
    if not in_use(cfg):
        return f"off ({site_key()} is blank)"
    row = check_line(cfg)
    return row.found + (f" — {row.fix}" if row.status in PROBLEMS else "")
