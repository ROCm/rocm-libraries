# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The local service behind the HTML dashboard's "Request PR priority" box (a PR number and a button: the PR joins the
priority list, for its reviews and its merge, unless it is there already), behind its "Tickets waiting for you"
card (built like the priority-request card; rkmon_lib/tickets.py), and behind the Approve and Decline of an in-depth
review on its Needs you list.

The dashboard is written every tick in two variants from the same code (webdash.write): the local page, the admin
dashboard this service serves, and the published copy the team sees. Only the local page talks to this service; the
published copy never contacts 127.0.0.1 (Chrome's Local Network Access would prompt or block every viewer on
other teams) and only asks the web host (tools/webhost_server.py). `rk serve [--port P]`
(tools/rkweb.py starts it in the background) listens at the admin dashboard's address and port (`admin`); README.md,
"The admin dashboard", says where that is and who can reach it. Inside the package's container image it listens on
every address at INNER_PORT, which docker/compose.yaml publishes on the host, at the port the browser's Host and Origin
name. The served page pings it on load: where it answers, the box prioritizes directly, and the user approves or
declines the other teams' requests (their clicks wait for the user's OK), the proposed tickets and the in-depth
reviews. The local file opened from disk only links to the served page.

  GET  /                  the admin dashboard's page itself (webdash.local_path, as the last tick wrote it): served from
                          here, the page and the service share one origin.
  GET  /ping              -> {"ok": true, "me": <config operator.github_login>}
  GET  /ticket-draft?id=X a proposed ticket's full text (the card's link; tickets.page, every value escaped)
  GET  /unsent?file=NAME  an unsent item's file, or the team dashboard's page while it does not send (the "Not sent"
                          card's links; webdash.unsent_named): a review kept unposted, as text, or the page, in a
                          sandbox (no script runs, nothing loads)
  POST /priority          {"pr": N} -> {"ok": bool, "message": "..."}: what `rk prioritize N requested from the
                          dashboard` does (priority.request: refused for a PR the monitor does not track as open, or one
                          already in the priority queue)
  POST /priority-approve  {"pr": N}: approve PR N's pending request (priority.approve: the same prioritize, its reason
                          naming the requesters)
  POST /priority-decline  {"pr": N}: decline it (priority.decline: recorded, dropped from the list)
  POST /review-approve    {"pr": N}: approve PR N's in-depth review, as `rk review-approve N` does (review.approve)
  POST /review-decline    {"pr": N}: decline it, as `rk review-skip N` does (review.decline: refused, as the approval
                          is, for a PR with no in-depth review)
  POST /ticket-approve    {"id": X}: file the proposed ticket X through the atlassian CLI (tickets.approve: its key to
                          the ledger and the row; nothing while ticket creation is off or the monitor is stopped;
                          while it does not send, in prototype or standby, the ticket kept as an unsent item)
  POST /ticket-decline    {"id": X}: drop the proposed ticket X (tickets.decline: journaled)
Each POST runs under the store lock, so it waits for a running tick and never loses an update (a ticket's filing runs
between two such steps, under the write lock).

Security. Every request but a preflight and the icon carries the access token, from this machine too: a configured
secret (`admin_access_token`: never made here, never in a page), as HTTP Basic, any user name and the token as the
password (compared in constant time). The browser asks for it once, and sends it again with the page's
own requests. The service refuses to start without it, and says why (`access_refusal`: a Refusal, the sentence that
names the fix, with that fix apart, which rk doctor's row shows on a line of its own); it reads the token only
from a private file (rkmon_lib/private_files.py): a file that breaks a rule is not read. A request is answered only
when its Host is an address and port the service listens on (no DNS rebinding; checked before the token is asked for,
so a browser never asks for it under another name: its 403 names the Host values the page answers to and the port an
SSH tunnel to it uses, `host_refusal`), and when it comes from the page: its Origin is the service's own
(a same-origin GET carries no Origin, but Sec-Fetch-Site says same-origin; an Origin of `null`, which a file opened
from disk and a sandboxed frame on any site send, is refused), and it carries the X-Rkmon header, so the browser
preflights every cross-origin request (no simple cross-site request gets through). The page itself is never served to
a request another site makes (Sec-Fetch-Site) and never framed (X-Frame-Options, frame-ancestors). A preflight is
answered only for those origins, their methods and headers, with Access-Control-Allow-Private-Network (Chrome's private
network access to localhost). Anything else is refused without CORS headers, so a foreign page cannot read the answer.
Every POST and every refusal goes to the journal (origin, PR or proposed ticket, result; never a token); the pings
and preflights that pass go only to the service's own log (state/web.log), as an open page pings every 2 minutes. No
GitHub call: a priority and an in-depth review's approval or decline are local actions (state and journal, no ledger
entry); a ticket's filing is the one JIRA write, ledgered.

Beyond the loopback (`exposure`: a bind address other than the loopback; in the container, a publish address other than
the loopback, as the container's environment says), the page answers for the Host and origin of
`admin_dashboard.url` too (required when it listens on every address), and for that address's.

The team dashboard, served here (features.dashboards.team with its delivery serve: the team dashboard either
publishes to an external host and port or is served locally on a chosen port): `rk serve` runs the web host's
own server (tools/webhost_server.py) beside the admin page, on `team_dashboard.bind` and `.port` (in the container
0.0.0.0 at TEAM_INNER_PORT): the published copy (publish.html copies it to state/team_site/index.html) and its request
box, whose requests go to state/team_requests.jsonl, which the tick reads as it reads the web host's. It never serves
the admin page, a token or an Approve button. While the monitor is in standby, it answers every request
with a short page alone, which says so and that the team dashboard opens there once its operator ends standby
(`team_notice`), and takes in no request; it reads standby as each request comes, so nothing restarts when standby
ends. The admin dashboard still shows the team page (webdash.team_page).
"""

from __future__ import annotations

import base64
import contextlib
import fcntl
import hmac
import http.server
import importlib.util
import ipaddress
import json
import os
import pathlib
import re
import signal
import sys
import threading
import urllib.parse

from . import priority, tickets
from . import config, container_layout, paths, private_files, registry, wrapping
from .checks import FIX_LABEL, Described, Edit, Status, Steps
from .store import now

# The default address it listens on (admin_dashboard.bind): this machine only.
HOST = container_layout.LOOPBACK
# The admin page's and the team dashboard's ports inside the container (docker/compose.yaml publishes them), and
# elsewhere unless admin_dashboard.port names another, or team_dashboard.port when this monitor serves the team's.
INNER_PORT = DEFAULT_PORT = container_layout.ADMIN_PORT
TEAM_INNER_PORT = TEAM_PORT = container_layout.TEAM_PORT
HEADER = "X-Rkmon"
METHODS = ("GET", "POST")
HEADERS = ("content-type", "x-rkmon")  # what a preflight may ask for
ACCESS_SECRET = "admin_access_token"  # the access token's name in the secrets registry
MAX_BODY = 1024
PR_MAX = 10**7
TICKET_ROUTES = ("/ticket-approve", "/ticket-decline")
# An in-depth review's Approve and Decline, on its Needs you line (webdash.decide_buttons).
REVIEW_ROUTES = ("/review-approve", "/review-decline")
ROUTES = (
    "/priority",
    "/priority-approve",
    "/priority-decline",
    *REVIEW_ROUTES,
    *TICKET_ROUTES,
)
PAGE_ROUTES = ("/", "/index.html")
DRAFT_ROUTE = "/ticket-draft"
UNSENT_ROUTE = "/unsent"
PAGE_HEADERS = {
    "Content-Type": "text/html; charset=utf-8",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": "frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}
# An unsent item's file, as UNSENT_ROUTE serves it: a review as text; the team dashboard's page in a sandbox, where no
# script runs, it has an origin of its own and it loads nothing, so a page made from what the monitor read never acts
# here.
UNSENT_TEXT_HEADERS = dict(
    PAGE_HEADERS, **{"Content-Type": "text/plain; charset=utf-8"}
)
UNSENT_PAGE_HEADERS = dict(
    PAGE_HEADERS,
    **{
        "Content-Security-Policy": "sandbox allow-popups allow-popups-to-escape-sandbox; default-src 'none'; "
        "style-src 'unsafe-inline'; img-src data:; frame-ancestors 'none'"
    },
)
REALM = 'Basic realm="rkmon admin dashboard", charset="UTF-8"'
NEED_ACCESS = (
    "the access token is needed (HTTP Basic: any user name, the token as the password)"
)
# Before the first tick writes the page (the README's quick start opens it right after `rk resume`): a page that says so
# and reloads itself, never a JSON refusal.
NO_PAGE_YET = (
    "<!doctype html><html lang='en'><head><meta charset='utf-8'><meta http-equiv='refresh' content='30'>"
    "<title>rocKE monitor</title></head><body><p>No dashboard page yet: the next tick writes it. "
    "This page reloads itself every 30 seconds.</p></body></html>"
)
# What the team dashboard served here answers every request with while the monitor is in standby
# (team_notice), {} the command that ends standby, by the program's own name: the page is for other teams.
STANDBY_NOTICE = "This rocKE monitor is in standby: its team dashboard opens here once its operator runs {}."


def admin(cfg: dict) -> dict:
    """The admin dashboard's settings, read here and nowhere else: `admin_dashboard.port`, `.bind`, `.public_port`,
    `.url` and `.access_token_file` (the place the secrets registry sets for ACCESS_SECRET, which the load derives:
    rkmon_lib/derived.py; "" while no secrets folder is known). No setting names the host address it is published on:
    the host's 127.0.0.1 (HOST) alone.
    """
    a = cfg.get("admin_dashboard") or {}
    return {
        "port": int(a.get("port") or DEFAULT_PORT),
        "bind": (a.get("bind") or HOST).strip(),
        "public_port": int(a.get("public_port") or 0),
        "url": (a.get("url") or "").strip(),
        "access_token_file": str(a.get("access_token_file") or ""),
    }


def port(cfg: dict) -> int:
    """The admin dashboard's port (on this machine; inside the container, INNER_PORT is the one it listens on)."""
    return admin(cfg)["port"]


# --- the access token, and where the page is reachable ------------------------------------------------------------------


def loopback(addr: str) -> bool:
    """A loopback address (127.0.0.0/8, ::1, localhost): reachable from this machine only."""
    a = (addr or "").strip().strip("[]")
    if a.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(a).is_loopback
    except ValueError:
        return False


def exposure(cfg: dict, environ=os.environ, bind: str | None = None) -> str | None:
    """The address beyond this machine the admin page is reachable at, or None (the loopback only). Outside the
    container: its bind address (`bind`, what `rk serve --bind` gives, else the configured one). In the package's image
    the service always listens on every address inside, and what counts is where docker publishes it on the host:
    RKMON_ADMIN_HOST in the container's environment (compose passes it), when it is beyond the loopback. No setting
    names that address: it is the host's 127.0.0.1."""
    from . import doctor

    if doctor.deployment_image(environ):
        host = (environ.get("RKMON_ADMIN_HOST") or "").strip()
        return host if host and not loopback(host) else None
    bind = bind or admin(cfg)["bind"]
    return None if loopback(bind) else bind


def _token(cfg: dict) -> tuple:
    """(the access token, or None; the FAIL rows of its file). Its file (admin_dashboard.access_token_file) is a private
    file: one others may read is set to 600 first (private_files.tighten), and one that still breaks a rule of
    private_files.problems (a link, another owner, a place that carries it off) is not read, and its rows say why. No
    file there is no token, and no row.
    """
    f = admin(cfg)["access_token_file"]
    path = pathlib.Path(os.path.expanduser(f)) if f else None
    if path is None or not os.path.lexists(path):
        return None, []
    with contextlib.suppress(OSError):  # what stays wrong, problems() says
        private_files.tighten(path)
    rows = [
        r
        for r in private_files.problems(path, paths.PACKAGE)
        if r.status == Status.FAIL
    ]
    if rows:
        return None, rows
    try:  # a link put there since is not followed
        fd = os.open(path, os.O_RDONLY | private_files.NOFOLLOW)
        with open(fd, encoding="utf-8") as fh:
            return fh.read().strip() or None, []
    except OSError:
        return None, []


def access_token(cfg: dict) -> str | None:
    """The access token every request carries (a configured secret: never made here), or None when there is none,
    or when its file breaks a private file's rule (`access_refusal` says which, with the fix).
    """
    return _token(cfg)[0]


def home_url(cfg: dict, environ=os.environ) -> str:
    """The address a browser opens the admin page at: `admin_dashboard.url`, else the loopback (or the bind address it
    listens on) at the port the browser names (in the container, the host's published port).
    """
    from . import doctor

    a = admin(cfg)
    if a["url"]:
        return a["url"].rstrip("/") + "/"
    if doctor.deployment_image(environ):
        env_port = str(environ.get("RKMON_PORT") or "")
        return f"http://{HOST}:{int(env_port) if env_port.isdigit() else a['public_port'] or a['port']}/"
    host = (
        HOST
        if loopback(a["bind"]) or a["bind"] == container_layout.EVERY_ADDRESS
        else a["bind"]
    )
    return f"http://{host}:{a['port']}/"


def _in_order(fixes: list):
    """FIXES as one fix, in order: the one alone, else their steps joined by ", then"."""
    if len(fixes) == 1:
        return fixes[0]
    parts = []
    for f in fixes:
        parts += [", then"] if parts else []
        parts += list(f.parts) if isinstance(f, Steps) else [f]
    return Steps(tuple(parts))


class Refusal(str):
    """Why the service may not start (`access_refusal`): the sentence the service prints, which names the fix, and,
    apart, `why`, the sentence without the fix, and `fix`, the fix (rkmon_lib/checks.py), which rk doctor's row shows
    on a line of its own."""

    def __new__(cls, why: str, fix, shown: str = ""):
        self = super().__new__(cls, shown or why)
        self.why, self.fix = why, fix
        return self


def access_refusal(
    cfg: dict, environ=os.environ, bind: str | None = None
) -> Refusal | None:
    """Why the service may not start as configured (None when it may): it needs the access token, from a file that
    breaks no private file's rule, and on every address an address a browser opens. Its fix: that of each rule the
    token's file breaks, in order; else, with no token, the command that sets it (secrets_registry.set_fix, named as
    ENVIRON says the operator types it); else the edit that sets the address a browser opens.
    """
    from . import secrets_registry as sr

    token, rows = _token(cfg)
    if rows:
        lead = (
            "the admin dashboard needs its access token on every request, from this machine too, and reads it only "
            "from a private file: "
        )
        return Refusal(
            lead + "; ".join(r.found for r in rows),
            _in_order([r.fix for r in rows]),
            lead + "; ".join(f"{r.found} ({FIX_LABEL}{r.fix})" for r in rows),
        )
    if not token:
        why = (
            "the admin dashboard needs its access token on every request, from this machine too, and none is set "
            f"({admin(cfg)['access_token_file'] or 'the secret ' + ACCESS_SECRET})"
        )
        fix = sr.set_fix(sr.entry(cfg, ACCESS_SECRET), environ)
        return Refusal(why, fix, f"{why}: run {fix}")
    where = exposure(cfg, environ, bind)
    if where == container_layout.EVERY_ADDRESS and not admin(cfg)["url"]:
        return Refusal(
            f"the admin dashboard would listen on every address ({where}): set admin_dashboard.url, the address a "
            "browser opens; the service answers only requests that name it",
            Edit(
                "admin_dashboard.url",
                config.config_path(),
                Described("the address a browser opens"),
            ),
        )
    return None


def access_info(cfg: dict, ports, environ=os.environ, bind: str | None = None) -> dict:
    """{"where", "token", "hosts", "origins"}: the address beyond the loopback the page is reachable at (`exposure`,
    None on the loopback only), the token every request carries ("" when none is set: then every request is refused),
    and, beyond the loopback, the Host values and origins the page answers for besides the loopback's.
    """
    where = exposure(cfg, environ, bind)
    hosts, origins_ = [], []
    url = admin(cfg)["url"] if where else ""
    if url:
        u = urllib.parse.urlsplit(url)
        hosts.append(u.netloc.lower())
        origins_.append(f"{u.scheme.lower()}://{u.netloc.lower()}")
    if where and where != container_layout.EVERY_ADDRESS:
        for p in _ports(ports):
            hosts.append(f"{where}:{p}")
            origins_.append(f"http://{where}:{p}")
    return {
        "where": where,
        "token": access_token(cfg) or "",
        "hosts": tuple(dict.fromkeys(hosts)),
        "origins": tuple(dict.fromkeys(origins_)),
    }


def given_token(h: dict) -> str:
    """The token a request carries: the password of HTTP Basic (any user name); "" for anything else."""
    scheme, _, value = (h.get("authorization") or "").strip().partition(" ")
    if scheme.lower() == "basic":
        try:
            return (
                base64.b64decode(value.strip(), validate=True)
                .decode("utf-8")
                .partition(":")[2]
            )
        except (ValueError, UnicodeDecodeError):
            return ""
    return ""


def authorized(h: dict, access: dict | None) -> bool:
    want = (access or {}).get("token") or ""
    return bool(want) and hmac.compare_digest(
        given_token(h).encode("utf-8"), want.encode("utf-8")
    )


# --- the checks every request passes --------------------------------------------------------------------------------------


def _clean(s: str | None) -> str:
    """An untrusted header value for the journal: origin characters only, cut short."""
    return re.sub(r"[^\w:/.\-\[\]]", "?", s)[:80] if s else "none"


def _headers(raw) -> dict[str, str]:
    """Lower-cased header names; the first value of a repeated header, as http.client's get() reads it."""
    out: dict[str, str] = {}
    for k, v in raw.items():
        out.setdefault(k.lower(), v)
    return out


def _ports(listen_port) -> tuple[int, ...]:
    """The ports a browser on this machine names: the one the service listens on and, in a container, the host's port
    it is published on (a Server's `ports`)."""
    ports = (listen_port,) if isinstance(listen_port, int) else tuple(listen_port)
    return tuple(dict.fromkeys(p for p in ports if p))


def host_names(listen_port, access: dict | None = None) -> tuple[str, ...]:
    """The Host values the service answers to: the loopback's two names at each port a browser on this machine names
    (`_ports`) and, beyond the loopback, the ones ACCESS gives (access_info's "hosts").
    """
    return (
        *(f"{name}:{p}" for p in _ports(listen_port) for name in (HOST, "localhost")),
        *((access or {}).get("hosts") or ()),
    )


def _host_ok(h: dict, listen_port, access: dict | None = None) -> bool:
    return (h.get("host") or "").lower() in host_names(listen_port, access)


def host_refusal(listen_port, access: dict | None = None) -> str:
    """What a request with any other Host is told with its 403 (`answer`): the names the page answers to, and that an
    SSH tunnel to it uses the admin dashboard's port at both ends, the port a browser on this machine names (in the
    container, the one published on the host); README.md, "The admin dashboard", has the tunnel's command. Never the
    request's own Host."""
    port = _ports(listen_port)[-1]
    names = wrapping.joined(host_names(listen_port, access), "or")
    return (
        f"this page answers only at {names}; through an SSH tunnel, use the admin dashboard's port, {port}, at both "
        'ends (README.md, "The admin dashboard", shows the command)'
    )


def origins(listen_port, access: dict | None = None) -> tuple[str, ...]:
    """The origins that may call the service: the page the service serves (and, beyond the loopback, the page at
    admin_dashboard.url and at that address). Never `null`: a file opened from disk, or a sandboxed frame on any site.
    """
    return (
        *(
            f"http://{name}:{p}"
            for p in _ports(listen_port)
            for name in (HOST, "localhost")
        ),
        *((access or {}).get("origins") or ()),
    )


def page_refusal(h: dict, listen_port: int, access: dict | None = None) -> str | None:
    """Why a GET of the page is refused (None when it is not): a foreign Origin, or a foreign page that fetches or
    frames it, which Sec-Fetch-Site names. A navigation carries no Origin; every request passed its Host and carried
    the access token already (`answer`)."""
    if h.get("origin") not in (None, *origins(listen_port, access)):
        return "origin not allowed"
    if h.get("sec-fetch-site") not in (None, "none", "same-origin"):
        return "the page opens only on this machine"
    return None


def draft_refusal(h: dict, listen_port: int, access: dict | None = None) -> str | None:
    """Why a GET of a proposed ticket's text, or of an unsent item's file (UNSENT_ROUTE), is refused (None when it is
    not). Neither holds a secret, so a navigation from the file opened from disk (Sec-Fetch-Site cross-site) gets it
    too, with the access token as every request, its Host passed already (`answer`); never into a frame.
    """
    if h.get("origin") not in (None, *origins(listen_port, access)):
        return "origin not allowed"
    if h.get("sec-fetch-dest") in ("iframe", "frame", "embed", "object"):
        return "never in a frame"
    return None


def _file_bytes(p: pathlib.Path) -> bytes | None:
    """The bytes of the file P, never through a link (O_NOFOLLOW); None when it cannot be read."""
    try:
        fd = os.open(p, os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC)
        with os.fdopen(fd, "rb") as f:
            return f.read()
    except OSError:
        return None


def refusal(
    h: dict, listen_port: int, method: str, access: dict | None = None
) -> str | None:
    """Why a GET or POST is not the served page's (None when it is); every request passed its Host and carried the
    access token already (`answer`). A same-origin GET from the served page carries no Origin; Sec-Fetch-Site vouches
    for it.
    """
    origin = h.get("origin")
    same = (
        origin is None and method == "GET" and h.get("sec-fetch-site") == "same-origin"
    )
    if not same and origin not in origins(listen_port, access):
        return "origin not allowed"
    if h.get(HEADER.lower()) != "1":
        return f"no {HEADER} header"
    return None


def preflight(
    cfg: dict, h: dict, listen_port: int, access: dict | None = None
) -> tuple[int, dict]:
    """The answer to a CORS preflight: allowed only for the local page's origins, its methods and headers, and a
    Host the service listens on (a preflight is answered before `answer` checks the Host of other requests).
    """
    origin = h.get("origin")
    method = (h.get("access-control-request-method") or "").upper()
    asked = {
        x.strip().lower()
        for x in (h.get("access-control-request-headers") or "").split(",")
        if x.strip()
    }
    if (
        origin not in origins(listen_port, access)
        or not _host_ok(h, listen_port, access)
        or method not in METHODS
        or not asked <= set(HEADERS)
    ):
        return 403, {}
    return 204, {
        "Access-Control-Allow-Origin": origin,
        "Access-Control-Allow-Methods": ", ".join(METHODS),
        "Access-Control-Allow-Headers": f"Content-Type, {HEADER}",
        "Access-Control-Allow-Private-Network": "true",
        "Access-Control-Max-Age": "600",
        "Vary": "Origin",
    }


def pr_number(data) -> int | None:
    """The PR number of a request body: {"pr": 1234}, or "1234" / "#1234" as text."""
    v = data.get("pr") if isinstance(data, dict) else None
    if isinstance(v, str) and re.fullmatch(r"#?\d{1,7}", v.strip()):
        v = int(v.strip().lstrip("#"))
    return (
        v if isinstance(v, int) and not isinstance(v, bool) and 0 < v < PR_MAX else None
    )


def answer(
    cfg: dict,
    store,
    method: str,
    path: str,
    raw_headers,
    body: bytes,
    listen_port: int,
    access: dict | None = None,
) -> tuple[int, dict, dict | bytes | None]:
    """One request's answer: (status, headers, JSON payload, the page's bytes, or None). The journal gets every POST
    and every refusal. `access` (access_info): the token every request but a preflight and the icon carries, after its
    Host passed; without it, every such request is refused."""
    h = _headers(raw_headers)
    split = urllib.parse.urlsplit(path)
    route = split.path
    origin = h.get("origin")
    n = tid = None
    if (
        method == "POST"
    ):  # read for the journal too: a refused request still names its PR or proposed ticket
        try:
            data = json.loads(body.decode("utf-8") or "null")
        except (UnicodeDecodeError, ValueError):
            data = None
        n, tid = pr_number(data), tickets.draft_id(data)

    def refuse(
        status: int, why: str, extra: dict | None = None, message: str = ""
    ) -> tuple[int, dict, dict]:
        """The refusal STATUS: WHY in the journal, and in the answer too unless MESSAGE says more."""
        store.journal(
            "web",
            f"refused {method} {route[:40]}"
            + (f" #{n}" if n else "")
            + (f" {tid}" if tid else "")
            + f" from {_clean(origin)}: {why}",
        )
        return status, extra or {}, {"ok": False, "message": message or why}

    if (
        method == "GET" and route == "/favicon.ico"
    ):  # the browser asks with every page load: no icon, no journal line
        return 204, {}, None
    if method == "OPTIONS":
        status, hdrs = preflight(cfg, h, listen_port, access)
        if status != 204:
            store.journal(
                "web",
                f"refused preflight {route[:40]} from {_clean(origin)} "
                f"({_clean(h.get('access-control-request-method'))})",
            )
        return status, hdrs, None
    # Another Host is refused before the token is asked for: a browser never asks for it under another name.
    if not _host_ok(h, listen_port, access):
        return refuse(
            403, "host not allowed", message=host_refusal(listen_port, access)
        )
    if not authorized(h, access):
        return refuse(401, NEED_ACCESS, {"WWW-Authenticate": REALM})
    if method == "GET" and route in PAGE_ROUTES:
        why = page_refusal(h, listen_port, access)
        if why:
            return refuse(403, why)
        from . import webdash  # webdash imports this module

        try:
            page = webdash.local_path(cfg).read_bytes()
        except OSError:
            return 200, dict(PAGE_HEADERS), NO_PAGE_YET.encode("utf-8")
        return 200, dict(PAGE_HEADERS), page
    if method == "GET" and route == DRAFT_ROUTE:
        why = draft_refusal(h, listen_port, access)
        if why:
            return refuse(403, why)
        tid = tickets.clean_id(
            (urllib.parse.parse_qs(split.query).get("id") or [""])[0]
        )
        item = tickets.find(store.load(), tid) if tid else None
        if not item:
            return refuse(404, "no such proposed ticket")
        return 200, dict(PAGE_HEADERS), tickets.page(cfg, item).encode("utf-8")
    if method == "GET" and route == UNSENT_ROUTE:
        why = draft_refusal(h, listen_port, access)
        if why:
            return refuse(403, why)
        from . import state_store, webdash  # webdash imports this module

        found = webdash.unsent_named(
            cfg, store, (urllib.parse.parse_qs(split.query).get("file") or [""])[0]
        )
        data = _file_bytes(found[0]) if found else None
        if data is None:
            return refuse(404, "no such file of an unsent item")
        page = found[1] == state_store.PAGE
        return 200, dict(UNSENT_PAGE_HEADERS if page else UNSENT_TEXT_HEADERS), data
    why = refusal(h, listen_port, method, access)
    if why:
        return refuse(403, why)
    cors = {"Access-Control-Allow-Origin": origin, "Vary": "Origin"} if origin else {}
    if method == "GET" and route == "/ping":
        return 200, cors, {"ok": True, "me": config.operator_login(cfg)}
    if method != "POST" or route not in ROUTES:
        return refuse(404, "not found", cors)
    if (h.get("content-type") or "").split(";")[
        0
    ].strip().lower() != "application/json":
        return refuse(
            415,
            (
                'send JSON: {"pr": N}'
                if route not in TICKET_ROUTES
                else 'send JSON: {"id": ...}'
            ),
            cors,
        )
    if route in TICKET_ROUTES:
        if tid is None:
            return refuse(400, 'send a proposed ticket\'s id: {"id": "..."}', cors)
        me = config.operator_login(cfg) or "the operator"
        extra = {}
        if (
            route == "/ticket-approve"
        ):  # its own locks: the filing runs between two state changes, under the write lock
            ok, msg, extra = tickets.approve(cfg, store, tid, by=me, own=True)
        else:
            with store.locked(wait=True):
                st = store.load()
                ok, msg = tickets.decline(cfg, st, store, tid, by=me)
                store.save(st)
        store.journal(
            "web",
            f"POST {route} {tid} from {_clean(origin)}: {'done' if ok else 'refused'}: {msg}",
        )
        return (200 if ok else 409), cors, dict({"ok": ok, "message": msg}, **extra)
    if n is None:
        return refuse(400, 'send a PR number: {"pr": N}', cors)
    with store.locked(
        wait=True
    ):  # waits for a running tick; it loads the state the tick saved
        st = store.load()
        if route == "/priority":
            ok, msg = priority.request(st, store, n, by=config.operator_login(cfg))
        elif route == "/priority-approve":
            ok, msg = priority.approve(cfg, st, store, n)
        elif route == "/priority-decline":
            ok, msg = priority.decline(cfg, st, store, n)
        else:  # REVIEW_ROUTES
            from . import review

            decide = review.approve if route == "/review-approve" else review.decline
            ok, msg = decide(cfg, st, store, n)
        store.save(
            st
        )  # a refused approval may still answer the request (the PR closed meanwhile)
    store.journal(
        "web",
        f"POST {route} #{n} from {_clean(origin)}: {'done' if ok else 'refused'}: {msg}",
    )
    return (200 if ok else 409), cors, {"ok": ok, "message": msg}


class Handler(http.server.BaseHTTPRequestHandler):
    server_version = "rkmon-web"
    sys_version = ""
    timeout = 15  # a slow client cannot hold a thread

    def _respond(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_BODY:
            status, hdrs, payload = (
                413 if length > MAX_BODY else 400,
                {},
                {"ok": False, "message": "bad body length"},
            )
            self.server.store.journal(
                "web",
                f"refused {self.command} from {_clean(self.headers.get('Origin'))}: "
                f"body length {self.headers.get('Content-Length')!r:.20}",
            )
        else:
            body = self.rfile.read(length) if length else b""
            status, hdrs, payload = answer(
                self.server.cfg,
                self.server.store,
                self.command,
                self.path,
                self.headers,
                body,
                self.server.ports,
                getattr(self.server, "access", None),
            )
        raw = isinstance(payload, bytes)  # a page: its headers name its type
        data = (
            payload
            if raw
            else (
                json.dumps(payload, ensure_ascii=False).encode("utf-8")
                if payload is not None
                else b""
            )
        )
        self.send_response(status)
        for k, v in hdrs.items():
            self.send_header(k, v)
        if payload is not None and not raw:
            self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    do_GET = do_POST = do_OPTIONS = _respond

    def log_message(
        self, fmt: str, *args
    ) -> None:  # the service's own log: state/web.log via tools/rkweb.py
        headers = getattr(
            self, "headers", None
        )  # absent when the request line itself was bad
        sys.stderr.write(
            f"{now()} {fmt % args} origin={_clean(headers.get('Origin') if headers else None)}\n"
        )


class Server(http.server.ThreadingHTTPServer):
    """A pinging page is never kept waiting by a priority request that waits for a running tick."""

    allow_reuse_port = False  # never share the port with a second copy
    daemon_threads = True

    def __init__(
        self,
        cfg: dict,
        store,
        listen_port: int,
        bind: str = HOST,
        public_port: int = 0,
        access: dict | None = None,
    ):
        self.cfg, self.store, self.access = cfg, store, access
        super().__init__((bind, listen_port), Handler)
        self.ports = _ports((self.server_address[1], public_port))


def _ipv4(value: str) -> bool:
    try:
        ipaddress.IPv4Address(value)
        return True
    except ValueError:
        return False


def bind_refusal(bind: str) -> str | None:
    """Why the service may not listen on `bind` (None when it may): an IPv4 address of this machine, the loopback by
    default. The access token rule applies too (access_refusal)."""
    if loopback(bind) and _ipv4(bind):
        return None
    if not _ipv4(bind):
        return (
            f"admin_dashboard.bind is {bind!r}: an IPv4 address of this machine ({HOST} by default, "
            f"{container_layout.EVERY_ADDRESS} all)"
        )
    return None


def _stop(signum, frame):
    raise SystemExit(0)


def where(
    cfg: dict,
    bind: str | None = None,
    public_port: int | None = None,
    environ=os.environ,
) -> tuple[str, int]:
    """(the address to listen on, the host's published port): the arguments, else in the package's image every address
    and its RKMON_PORT (docker/compose.yaml publishes that port on the host), else the configuration.
    """
    from . import doctor

    image = doctor.deployment_image(environ)
    port_env = str(environ.get("RKMON_PORT") or "")
    return (
        bind or (container_layout.LISTEN_ADDRESS if image else admin(cfg)["bind"]),
        (
            public_port
            if public_port is not None
            else (
                int(port_env)
                if image and port_env.isdigit()
                else admin(cfg)["public_port"]
            )
        ),
    )


def listen_port(cfg: dict, environ=os.environ) -> int:
    """The port the admin page listens on: INNER_PORT in the package's image (the compose file publishes it), else
    admin_dashboard.port."""
    from . import doctor

    return INNER_PORT if doctor.deployment_image(environ) else port(cfg)


# --- the team dashboard, served here ----------------------------------------------------------------------------------


def team(cfg: dict) -> dict:
    """The team dashboard's settings (`team_dashboard`): mode (publish or serve), bind, port, publish_host,
    public_port, url."""
    t = cfg.get("team_dashboard") or {}
    return {
        "mode": str(registry.value(cfg, "team_dashboard.mode", registry.FALSY)).strip(),
        "bind": (t.get("bind") or HOST).strip(),
        "port": int(t.get("port") or TEAM_PORT),
        "publish_host": str(
            registry.value(cfg, "team_dashboard.publish_host", registry.FALSY)
        ).strip(),
        "public_port": int(t.get("public_port") or 0),
        "url": (t.get("url") or "").strip(),
    }


def team_serving(cfg: dict) -> bool:
    """This monitor serves the team dashboard: the feature is on and set up, in mode serve."""
    from . import features

    return team(cfg)["mode"] == "serve" and features.active(cfg, "team-dashboard")


def team_where(cfg: dict, environ=os.environ) -> tuple[str, int]:
    """(where the team dashboard listens, its port): team_dashboard.bind and .port; in the package's image every
    address at TEAM_INNER_PORT (the compose file publishes it)."""
    from . import doctor

    if doctor.deployment_image(environ):
        return container_layout.LISTEN_ADDRESS, TEAM_INNER_PORT
    t = team(cfg)
    return t["bind"], t["port"]


def team_url(cfg: dict, environ=os.environ) -> str:
    """The address other teams open the team dashboard at: team_dashboard.url, else where it listens (in the container,
    the host's published address and port)."""
    from . import doctor

    t = team(cfg)
    if t["url"]:
        return t["url"].rstrip("/") + "/"
    if doctor.deployment_image(environ):
        host = (environ.get("RKMON_TEAM_HOST") or t["publish_host"]).strip()
        env_port = str(environ.get("RKMON_TEAM_PORT") or "")
        p = int(env_port) if env_port.isdigit() else t["public_port"]
        return f"http://{HOST if host == container_layout.EVERY_ADDRESS else host}:{p or '?'}/"
    bind = t["bind"]
    return f"http://{HOST if bind == container_layout.EVERY_ADDRESS else bind}:{t['port']}/"


def team_reach(cfg: dict, environ=os.environ) -> str:
    """Who opens the team dashboard's page, in words: "other teams open" its address (`team_url`); in the container,
    while the settings publish its port on this host's loopback alone (features.team_host: in prototype, whatever
    publish_host says), "its page is on this host only"."""
    from . import doctor, features

    url = team_url(cfg, environ)
    if doctor.deployment_image(environ):
        published = features.team_host(cfg, team(cfg)["publish_host"])
        if published == container_layout.LOOPBACK:
            return f"its page is on this host only, at {url}"
    return f"other teams open {url}"


# What docker/compose.yaml gives a variable docker/.env leaves out: the team dashboard's address and port, which .env
# holds only while this monitor serves it. compose.yaml gives the others none: .env always holds them.
PUBLISH_DEFAULTS = {"RKMON_TEAM_HOST": HOST, "RKMON_TEAM_PORT": ""}


def publish_env(cfg: dict) -> list[tuple[str, str, str]]:
    """What docker/.env must say for docker/compose.yaml to publish the dashboards where the settings name them:
    (variable, its value, compose's default: "" for none). The team dashboard's only while this monitor serves it, at
    the address features.team_host gives: in prototype this host's own, whatever publish_host says.
    """
    from . import features

    t = team(cfg)
    port = registry.number(cfg, "admin_dashboard.public_port", int, registry.FALSY)
    out = [("RKMON_ADMIN_HOST", HOST, ""), ("RKMON_PORT", str(port), "")]
    if t["mode"] == "serve" and features.on(cfg, "team-dashboard"):
        team_port = registry.number(
            cfg, "team_dashboard.public_port", int, registry.FALSY
        )
        out += [
            (
                "RKMON_TEAM_HOST",
                features.team_host(cfg, t["publish_host"]),
                PUBLISH_DEFAULTS["RKMON_TEAM_HOST"],
            ),
            ("RKMON_TEAM_PORT", str(team_port), PUBLISH_DEFAULTS["RKMON_TEAM_PORT"]),
        ]
    return out


def publish_mismatch(cfg: dict, environ=os.environ) -> list[str]:
    """In the container: each docker/.env value compose used (the container's environment) that differs from the
    settings, in words."""
    out = []
    for var, want, default in publish_env(cfg):
        got = (environ.get(var) or "").strip() or default
        if got != want:
            out.append(
                f"{var} is {got or 'unset (a port docker picks)'} here; the settings say {want}"
            )
    return out


def _webhost():
    """The web host's own server, the one its install copies there (web_host_install.SERVER; standalone, it runs on
    the web host as it is)."""
    from . import web_host_install

    spec = importlib.util.spec_from_file_location(
        "rkmon_webhost_server", web_host_install.SERVER
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def team_notice() -> str:
    """While the monitor is in standby (features.standby, which reads the internal state again once it changed), the
    words the team dashboard served here answers every request with (STANDBY_NOTICE); "" otherwise. The served
    team dashboard's server asks it as each request comes (team_server)."""
    from . import features, program

    if not features.standby():
        return ""
    return STANDBY_NOTICE.format(f"{program.NAME} standby off")


def team_server(cfg: dict, store, environ=os.environ):
    """(the team dashboard's server, not started, or None; the line to print). It serves state/team_site (the published
    copy, publish.html writes it each tick) and takes the request box's requests into state/team_requests.jsonl;
    while the monitor is in standby, it answers every request with team_notice's words alone (the server's notice:
    tools/webhost_server.py), and takes in no request.
    """
    from . import publish

    if not team_serving(cfg):
        return None, ""
    bind, port_ = team_where(cfg, environ)
    site = publish.site_dir(store.dir)
    site.mkdir(parents=True, exist_ok=True)
    if not (site / "index.html").exists():
        (site / "index.html").write_text(NO_PAGE_YET, encoding="utf-8")
    url = team(cfg)["url"]
    origin = (
        [f"{urllib.parse.urlsplit(url).scheme}://{urllib.parse.urlsplit(url).netloc}"]
        if url
        else []
    )
    try:
        srv = _webhost().make_server(
            str(site), str(publish.requests_file(store.dir)), port_, bind, origin
        )
    except (OSError, ValueError) as exc:
        return None, f"RKWEB team dashboard not served on {bind}:{port_}: {exc}"
    srv.notice = team_notice
    return (
        srv,
        f"RKWEB serving the team dashboard on {bind}:{srv.server_address[1]} ({team_url(cfg, environ)}; read-only, with its request box)",
    )


def serve(
    cfg: dict,
    store,
    listen_port_: int | None = None,
    bind: str | None = None,
    public_port: int | None = None,
) -> int:
    """`rk serve`: run the service until stopped (SIGTERM, Ctrl-C). One copy per state directory: it holds
    state/web.lock while it runs, and a second copy says so and exits. It does not start without the access token
    (access_refusal). With the team dashboard served here, it serves that too.
    """
    from . import features

    if not features.on(cfg, "admin-dashboard"):
        print(
            "RKWEB not started: features.dashboards.admin is off, so no web server runs (no dashboard)"
        )
        return 0
    lock = open(store.dir / "web.lock", "a+")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        pid = (
            (store.dir / "web.pid").read_text().strip()
            if (store.dir / "web.pid").exists()
            else "?"
        )
        print(f"RKWEB already running (pid {pid})")
        lock.close()
        return 0
    p = listen_port_ or listen_port(cfg)
    bind, public_port = where(cfg, bind, public_port)
    why = bind_refusal(bind) or access_refusal(cfg, bind=bind)
    if why:
        print(f"RKWEB not started: {why}")
        lock.close()
        return 4
    try:
        httpd = Server(cfg, store, p, bind, public_port)
    except OSError as exc:
        print(f"RKWEB cannot listen on {bind}:{p}: {exc}")
        lock.close()
        return 4
    httpd.access = access_info(cfg, httpd.ports, bind=bind)
    team_srv, team_line = team_server(cfg, store)
    if team_line:
        print(f"{now()} {team_line}", flush=True)
        store.journal("web", team_line.replace("RKWEB ", "", 1))
    team_thread = None
    if team_srv:
        team_thread = threading.Thread(target=team_srv.serve_forever, daemon=True)
        team_thread.start()
    shown = httpd.ports[
        -1
    ]  # the port the browser uses: the published one in a container
    pidf = store.dir / "web.pid"
    pidf.write_text(f"{os.getpid()}\n")
    inside = (
        f" (inside a container, on {bind}:{httpd.server_address[1]})"
        if bind != HOST
        else ""
    )
    beyond = (
        f"; reachable beyond this machine on {httpd.access['where']}"
        if httpd.access["where"]
        else ""
    )
    store.journal(
        "web",
        f"service up on http://{HOST}:{shown}{inside} (pid {os.getpid()}) for the admin dashboard's page "
        f"(http://{HOST}:{shown}/; every request with the access token){beyond}",
    )
    print(
        f"{now()} RKWEB serving http://{HOST}:{shown}/{inside} (pid {os.getpid()}): the admin dashboard's page and its "
        f"requests (every request with the access token){beyond}",
        flush=True,
    )
    signal.signal(signal.SIGTERM, _stop)
    try:
        httpd.serve_forever()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        httpd.server_close()
        if team_srv:
            team_srv.shutdown()
            team_srv.server_close()
        if pidf.exists() and pidf.read_text().strip() == str(os.getpid()):
            pidf.unlink()
        store.journal("web", f"service on {bind}:{p} stopped")
        print(f"{now()} RKWEB stopped", flush=True)
        lock.close()
    return 0
