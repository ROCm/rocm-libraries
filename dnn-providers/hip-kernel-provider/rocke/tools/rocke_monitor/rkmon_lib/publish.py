# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Push the HTML dashboard to the team's web host after every write (config `publish.html`; its `url` is the team
dashboard, and every update of the page is pushed there), and read back the priority requests other teams make there
(their clicks wait for the user's OK). Off while `publish.html.host` is blank.

The configuration's `publish.html` names the host, the login, the key (`key`: web_host_ssh_key's file in its keeper's
secrets folder, which the load derives), the site folder, the page's path (`dest`: the page in the site folder,
rkmon_lib/derived.py's site_page) and, while priority requests are on, `requests`, the requests file at the top of the
login's home, never under the site folder's first folder (derived.site_requests); each path relative to the login's
home. The host serves the site with tools/webhost_server.py, as rocke-monitor.service under the user's own account
(`rkmon web-host install` installs it; a plain `python3 -m http.server` would refuse every request with a 501). rsync
writes a temporary file and renames it, so a reader never sees half a page. A failed push or read never breaks a tick:
it returns a one-line reason for the caller to print. Where this process does not read the key (logins.here: in the
container, every user but the writer), the writer pushes the page and reads the requests file (`rk web-push FILE`,
`rk web-requests`: rkmon_lib/writer.py), whose halves are `push_page` and `read_requests`. The writer first checks the
login, host and path the configuration names (`target_refusal`). ssh trusts the web host by its own keys the state
records alone (rkmon_lib/host_keys.py), which `rkmon secret web_host_ssh_key FILE` shows and records once you accept
them; while none is recorded, a push or a read of the requests file is refused before rsync runs, its reason saying how
one is recorded (host_keys.unrecorded).

Or the monitor serves the team dashboard itself (the team dashboard either publishes to an external host and port or
is served locally on a chosen port): with `team_dashboard.mode` serve, `html` copies
the page to state/team_site/index.html, which `rk serve` serves with the web host's own server
(rkmon_lib/webserve.team_server), and `pull_requests` reads the requests it took, state/team_requests.jsonl.

While the team dashboard does not send (prototype, or the monitor in standby: features.held), the page is made after
every tick as live makes it, and the one outward gate (post.outward_gate) keeps it as the one unsent page item
(state_store.PAGE, replaced at each tick) instead of pushing it: nothing is written to the web host. Served here, the
page is served in either mode, and in standby its port shows a short notice alone (webserve.team_notice). In
prototype the web host's requests file is not read (priority requests are taken in where the page is served); in
standby it is read, and nothing is written there.
"""

from __future__ import annotations

import pathlib
import re
import shlex
import shutil
import subprocess
import tempfile

_TRAILER = re.compile(
    r"write error: Broken pipe|connection unexpectedly closed|^rsync error: .*\(code \d+\)"
)
RSYNC_TIMEOUT_S = 45  # the most one push or read of the web host takes
# How `rk web-push` and `rk web-requests` begin the line that says why they failed (html and pull_requests give the
# reason alone).
NOT_PUSHED, NOT_READ = "not pushed: ", "not read: "


SITE = "team_site"  # in state/: the served copy (index.html), when this monitor serves the team dashboard
REQUESTS = "team_requests.jsonl"  # in state/: its request box's requests, beside the site and never inside it


def mode(cfg: dict) -> str:
    """Where the team dashboard goes: "publish" (the team web host, publish.html) or "serve" (this monitor)."""
    from . import registry

    return str(registry.value(cfg, "team_dashboard.mode", registry.FALSY)).strip()


def site_dir(state_dir) -> pathlib.Path:
    return pathlib.Path(state_dir) / SITE


def requests_file(state_dir) -> pathlib.Path:
    return pathlib.Path(state_dir) / REQUESTS


def takes_requests(cfg: dict) -> bool:
    """The team dashboard takes priority requests: served here, or published with a requests file on the host."""
    if mode(cfg) == "serve":
        from . import features

        return features.on(cfg, "team-dashboard")
    return bool(((cfg.get("publish") or {}).get("html") or {}).get("requests"))


def key_file(cfg: dict) -> str:
    """The web host's ssh key, where the secrets registry places web_host_ssh_key (secrets_registry.path: its file in
    its keeper's secrets folder, `publish.html.key`, which the load derives and no config.json value replaces; in the
    container, the writer's volume); "" when no secrets folder is known."""
    from . import paths, secrets_registry as sr

    try:
        return sr.path(cfg, sr.entry(cfg, "web_host_ssh_key"))
    except paths.PathsError:
        return ""


def ssh_argv(cfg: dict, key: str | None = None) -> list[str]:
    """ssh to the web host, never interactive: the key only; the web host's own keys the state records only
    (host_keys.ssh_options: checked strictly, never from ssh's own known_hosts); a short connect timeout.
    """
    from . import host_keys

    return [
        "ssh",
        "-i",
        key if key is not None else key_file(cfg),
        "-o",
        "BatchMode=yes",
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "ConnectTimeout=10",
        *host_keys.ssh_options(cfg, host_keys.WEB),
    ]


def _rsync(
    cfg: dict, src: str, dest: str, run, *opts: str, missing_ok: bool = False
) -> str | None:
    """One rsync over the configured key: None when done (or, with `missing_ok`, when `src` does not exist), otherwise
    its last error line that says why. rsync's own trailer lines ("write error: Broken pipe", "(code 12)") follow the
    cause and only say the other end stopped (alone, every failed read would say only "Broken pipe"); one
    of them is kept in brackets. While no key of the web host is recorded, ssh would fail its host key check: rsync
    does not run, and the reason says how one is recorded (host_keys.unrecorded)."""
    from . import host_keys

    why = host_keys.unrecorded(cfg, host_keys.WEB)
    if why:
        return why
    ssh = shlex.join(ssh_argv(cfg))
    try:
        done = run(
            ["rsync", "-a", *opts, "-e", ssh, src, dest],
            capture_output=True,
            text=True,
            timeout=RSYNC_TIMEOUT_S,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return f"{type(exc).__name__}: {exc}"
    out = (done.stderr or done.stdout or "").strip()
    if done.returncode == 0 or (
        missing_ok
        and any(
            "link_stat" in x and "No such file or directory" in x
            for x in out.splitlines()
        )
    ):
        return None
    lines = [x.strip() for x in out.splitlines() if x.strip()]
    if not lines:
        return f"rsync exited {done.returncode}"
    why = [x for x in lines if not _TRAILER.search(x)]
    return (f"{why[-1]} ({lines[-1]})" if why and why[-1] != lines[-1] else lines[-1])[
        :300
    ]


def html(cfg: dict, path: str, run=subprocess.run) -> str | None:
    """Copy the written dashboard at `path` (state/published.html) to the configured web host, or, when this monitor
    serves the team dashboard, to state/team_site/index.html (written whole: whole_file). None when it was done
    or nothing is configured; otherwise a one-line reason."""
    if mode(cfg) == "serve":
        from . import features, whole_file

        if not features.active(cfg, "team-dashboard"):
            return None
        site = site_dir(pathlib.Path(path).parent)
        try:
            site.mkdir(parents=True, exist_ok=True)
            whole_file.write(
                site / "index.html", pathlib.Path(path).read_bytes(), 0o644
            )
        except OSError as exc:
            return f"{type(exc).__name__}: {exc}"
        return None
    pub = (cfg.get("publish") or {}).get("html")
    if not pub or not pub.get("host"):
        return None
    from . import features

    if not features.active(
        cfg, "team-dashboard"
    ):  # off, or not set up: the tick reports it once
        return None
    if not key_here(cfg):
        from . import writer

        p = writer.through(
            [writer.WEB_PUSH, str(path)], timeout_s=RSYNC_TIMEOUT_S + writer.START_S
        )
        return (
            None if p.returncode == 0 else writer.said(p).removeprefix(NOT_PUSHED)[:300]
        )
    return push_page(cfg, path, run)


def key_here(cfg: dict) -> bool:
    """Whether this process reads the web host's key itself (logins.here): on a host, and in the container the writer
    alone."""
    from . import logins, secrets_registry as sr

    return logins.here(sr.entry(cfg, "web_host_ssh_key"))


def _host(pub: dict, path: str) -> str:
    return f"{pub['user']}@{pub['host']}:{path}"


# A login or host name the writer takes from the configuration: letters, digits, `.`, `_` and `-`, never first a `-`
# (ssh would read `-oProxyCommand=...` as its option).
PLAIN_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9._-]*")


def target_refusal(cfg: dict, key: str) -> str:
    """Why the writer refuses to use the web host's key on `publish.html.KEY` ("dest": `rk web-push`; "requests": `rk
    web-requests`), "" when it may. In the container the configuration is the monitor's, which the session may edit:
    the writer takes from it a plain login and host name, and a path inside the login's home that starts with no `/`,
    `~` or `-` and has no `..`, dot-folder or dot-file part (the login's ssh keys, its shell's start files); the
    requests file is the one the page's folder gives (derived.site_requests), at the top of the login's home.
    """
    from . import derived

    pub = (cfg.get("publish") or {}).get("html") or {}
    for name in ("user", "host"):
        value = str(pub.get(name) or "")
        if value and not PLAIN_NAME.fullmatch(value):
            return f"publish.html.{name} is not a plain name"
    path = str(pub.get(key) or "")
    if path[:1] in ("/", "~", "-"):
        return f"publish.html.{key} starts with {path[0]}: the writer takes a path inside the web host login's home"
    if any(part.startswith(".") for part in path.split("/")):
        return f"publish.html.{key} has a .., dot-folder or dot-file part"
    if key == "requests" and path:
        folder = str(pub.get("dest") or "").rpartition("/")[0]
        if path != derived.site_requests(folder):
            return (
                "publish.html.requests is not the requests file named after the folder of publish.html.dest, at the "
                "top of the login's home"
            )
    return ""


def push_page(cfg: dict, path: str, run=subprocess.run) -> str | None:
    """`rk web-push FILE`: the page at PATH to the web host's destination (`publish.html.dest`), with the key read
    here (the writer's half of `html`; `rk web-push` asks `target_refusal` first): None when done, else the one-line
    reason."""
    pub = (cfg.get("publish") or {}).get("html") or {}
    if not (pub.get("host") and pub.get("user") and pub.get("dest")):
        return "publish.html names no web host, login and destination: the page has nowhere to go"
    return _rsync(cfg, path, _host(pub, pub["dest"]), run, "--chmod=F644")


def read_requests(cfg: dict, run=subprocess.run) -> tuple[str, str | None]:
    """`rk web-requests`: the web host's priority requests file (`publish.html.requests`), read with the key read here
    (the writer's half of `pull_requests`; `rk web-requests` asks `target_refusal` first): (its text, "" when there is
    none yet; None, or why it could not be read).
    """
    pub = (cfg.get("publish") or {}).get("html") or {}
    if not (pub.get("host") and pub.get("user") and pub.get("requests")):
        return "", "publish.html names no web host, login and requests file"
    with tempfile.TemporaryDirectory() as d:
        dest = pathlib.Path(d) / "requests.jsonl"
        why = _rsync(cfg, _host(pub, pub["requests"]), str(dest), run, missing_ok=True)
        if why:
            return "", why
        try:
            return dest.read_bytes().decode("utf-8", errors="replace"), None
        except FileNotFoundError:
            return "", None
        except OSError as exc:
            return "", f"{type(exc).__name__}: {exc}"


STATUS = "publish_status.txt"  # in state/: the last publish's outcome, so a change of it is journaled once


def configured(cfg: dict) -> bool:
    """Would `html` send the page anywhere: the team dashboard on, served here or with a web host named."""
    from . import features

    if not features.on(cfg, "team-dashboard"):
        return False
    if mode(cfg) == "serve":
        return True
    return bool(((cfg.get("publish") or {}).get("html") or {}).get("host"))


def blocked(cfg: dict, st: dict | None) -> str | None:
    """Why the team dashboard is not published now (None when it may be): on a mirror (it would overwrite the live
    monitor's team dashboard), or while an rk stop or rk kill is in force (nothing goes out until rk start).
    """
    from . import mirror
    from .store import stopped

    if mirror.on(cfg):
        return "this monitor is a mirror (config mirror.enabled): the team dashboard is not published"
    if st is not None and stopped(st):
        return "the monitor is stopped (rk start resumes): the team dashboard is not published"
    return None


PAGE_WHAT, PAGE_FOR = "the team dashboard's page", "the team dashboard's web host"


def held(cfg: dict, st: dict | None, store, path: str):
    """The one outward gate (post.outward_gate) for the push of the written page at PATH to the web host: None when it
    goes out; else the page kept as the unsent page item (state_store.PAGE, its file PATH) in STORE's state, and the
    gate's post.Held that says so. The page is the monitor's own, whoever wrote it. Served here, it is never held: it
    leaves no machine."""
    from . import post, state_store

    if mode(cfg) == "serve":
        return None
    try:
        post.outward_gate(
            cfg,
            st,
            PAGE_WHAT,
            ("team-dashboard",),
            post.unsent_item(state_store.PAGE, PAGE_FOR, file=str(path)),
            own=True,
            store=store,
        )
    except post.Held as kept:
        return kept
    return None


def team(cfg: dict, st: dict | None, store, path: str) -> str | None:
    """Publish the written page at `path` (`html`) unless `blocked`, or `held`: None when it was done, kept as the
    unsent page item, or nothing is configured; otherwise the one-line reason. Each change of the outcome (published,
    kept, blocked, failed) is one journal line, so the record says when publishing started, stopped and why, without a
    line per tick (a ledger line would make every next tick full). A page kept as a new unsent page item has the
    gate's line (post.keep), and no second one."""
    if not configured(cfg):
        return html(cfg, path)  # nothing to send it to: html says so, or does nothing
    why = blocked(cfg, st)
    kept = None if why else held(cfg, st, store, path)
    if not (why or kept):
        why = html(cfg, path)
    outcome = f"not published: {why or kept}" if (why or kept) else "published"
    f = pathlib.Path(store.dir) / STATUS
    try:
        last = f.read_text(encoding="utf-8").strip()
    except OSError:
        last = ""
    if last != outcome:
        if not (kept and kept.new):
            store.journal("publish", f"team dashboard {outcome}"[:300])
        from . import paths, whole_file

        try:
            whole_file.write(
                f, outcome + "\n", paths.worker_module("perms").SHARED_FILE_MODE
            )
        except OSError:
            pass
    return why


def pull_requests(cfg: dict, dest: str, run=subprocess.run) -> str | None:
    """Copy the web host's priority requests file to `dest` (rsync sends only what changed; through the writer, `dest`
    is written whole with the writer's answer, empty when the web host's file is empty or not there), or, when this
    monitor serves the team dashboard, the requests its server took (state/team_requests.jsonl, beside `dest`). None
    when it was copied, when there is none yet (nobody has asked), or when nothing is configured; otherwise a one-line
    reason. Published to the web host, the requests are read only while the team dashboard is live by its settings
    (features.live: in standby too); in prototype priority requests are taken in where the page is served.
    """
    if mode(cfg) == "serve":
        src = requests_file(pathlib.Path(dest).parent)
        try:
            if src.exists():
                shutil.copyfile(src, dest)
        except OSError as exc:
            return f"{type(exc).__name__}: {exc}"
        return None
    pub = (cfg.get("publish") or {}).get("html") or {}
    if not pub.get("host") or not pub.get("requests"):
        return None
    from . import features

    if not (
        features.live(cfg, "dashboards.team") and features.active(cfg, "team-dashboard")
    ):
        return None
    if not key_here(cfg):
        from . import writer

        p = writer.through(
            [writer.WEB_REQUESTS], timeout_s=RSYNC_TIMEOUT_S + writer.START_S
        )
        if p.returncode != 0:
            return writer.said(p).removeprefix(NOT_READ)[:300]
        try:
            pathlib.Path(dest).write_text(p.stdout or "", encoding="utf-8")
        except OSError as exc:
            return f"{type(exc).__name__}: {exc}"
        return None
    return _rsync(cfg, _host(pub, pub["requests"]), dest, run, missing_ok=True)
