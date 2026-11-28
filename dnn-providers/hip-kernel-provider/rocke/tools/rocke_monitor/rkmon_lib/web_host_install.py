# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""What `rkmon web-host install` puts on the team's web host: the request service, which serves the
published team dashboard and takes the priority requests other teams make on it (tools/webhost_server.py), run as the
user service UNIT under the web host login's own account.

`plan(cfg) -> Plan`, from the running configuration CFG (rkmon_lib/runtime_config.py) by the registry's names: the
team dashboard's delivery (team_dashboard.mode), the web host (publish.html.host), its login (publish.html.user), the
site folder (publish.html.site), the page's public address (publish.html.url) and the requests file the site folder
gives while priority requests are on (publish.html.requests, a derived path), each path relative to the login's home.
It reads no configuration file, only the package's own files: SERVER and UNIT_TEMPLATE (OSError when one is missing).
ValueError says why there is no plan: `nothing_to_install`'s reason; a blank setting; a path the writer refuses
(publish.target_refusal), or one with a part that is not a plain name (publish.PLAIN_NAME; in the requests file's name,
each of the site folder's names that derived.FOLDER_SIGN joins), which the unit's command line and the remote half take
as it is; a public address the service does not serve the page at (service_address). Each reason names a setting by the
deploy settings file's key, and a path the site folder gives (the page, the requests file) by the site folder's.

`service_address(text) -> (port, origin)`: the public address TEXT's port and origin, as the plan takes them; Unserved,
a ValueError, for an address the service does not serve the page at: one that is not http:// with a port from
FIRST_PORT to 65535 (the service serves the page itself, over plain http, on a port the login may open); one that names
no host, or whose host is neither an IP address nor a name of letters, digits, "." and "-" (HOST_NAME, in its IDNA
form), which the unit's command line takes as it is; one with a path other than "/", for the service serves the site
folder at the root of its own port. Its `give` is the address to give in its place when one follows from TEXT (its
origin, then "/"), else "". The confirm's check 7 asks it too, while the web host needs the service
(rkmon_lib/coherence.py): GIVE says what to give, in a fix's words.

`Plan`:
- host, user: where the install logs in.
- base: the site folder's first folder, which holds the server and the unit's new copy.
- site, requests: the served folder and the requests file.
- port, bind, origin: the public address's port, the address the service listens on (every address the web host
  has: container_layout.EVERY_ADDRESS) and the page's origin, the one the service takes requests from: the address's
  origin as a browser sends it, which the service compares as it is (its host in lower case, a name in its IDNA form,
  an IPv6 address in brackets; then the port, never http's own).
- unit: the unit's text, UNIT_TEMPLATE with each @NAME@ filled in.
- copies: the files to copy, each a Copy(dest, text), DEST relative to the login's home: the server, as
  BASE/webhost_server.py, then the unit, as BASE/UNIT.new, which the remote half puts in place.
- `remote(install=True, replace=False)`: the command the install runs on the web host, with the remote half (REMOTE)
  on its standard input: `python3 - MODE REPLACE BASE SITE REQUESTS PORT`. MODE is --yes, which installs the service
  or restarts it, or --status, which only looks; REPLACE "yes" lets the remote half stop this login's own
  `http.server` on the port first.

`nothing_to_install(cfg) -> str`: why the web host needs nothing installed, "" when it needs the service: the monitor
serves the team dashboard and its priority requests itself (delivery "serve"), or priority requests are off, and the
web host needs the service only for them (publish.takes_requests).

`deploy_keys() -> tuple`: the deploy settings file's keys the plan is made from: the delivery, each of SETTINGS' and
the priority requests' switch, which gives the requests file. The service on the web host takes a change of one only
when the install runs again, after apply: the confirm says so under the change (rkmon_lib/confirm.py change_lines).

`unit_text(template, values) -> str`: TEMPLATE with each @NAME@ replaced by VALUES[NAME]; ValueError for a NAME that
VALUES lacks.

`install(cfg, run=subprocess.run, out=None) -> int`: `rk web-host-install`, the plan put on the web
host over ssh with the web host's key, which trusts the web host by its own keys the state records alone
(publish.ssh_argv), and says its errors alone (SSH_ERRORS_ONLY); in the container the writer runs it, as
`rkmon web-host install` asks from the host. Its steps, each failure naming its own: the look (REMOTE with --status: ssh
reaches the web host, its login runs python3, and that python3 runs the service), the copies (rsync, as the page's push
runs it: publish's own), then REMOTE with --yes, whose lines are printed before the outcome. The outcome
(rkmon_lib/outcome.py): INSTALLED; NOTHING TO DO for `nothing_to_install`'s reason; REFUSED where this process does not
read the web host's key (logins.here: the monitor's rk in the container), and in standby (features.standby), since the
monitor you moved from may still use the web host; FAILED, nothing changed, for a plan that cannot be made (its reason;
the fix, doctor.cannot_run_fix: the settings fixed in the deploy settings file), the key or the web host's own key not
set yet (exit 3: an earlier step's, with the command that sets it), a look that fails, or a look or a step REMOTE
refuses (REMOTE_REFUSED: its last line names the step); ERROR, part-way, for a copy that fails, a service that does not
come up (REMOTE has started the server it replaced again, as it was), or an ssh that fails during the install;
INTERRUPTED at a Ctrl-C, with what it leaves on the web host: nothing changed before the
copies; during them, some of the copies, and the unit as it was; once REMOTE installs, a unit that may not run. RUN runs
ssh and rsync; OUT takes what it prints.

`install_command(environ=os.environ)`: the command that installs the request service, as the operator types it on the
host: `<wrapper> web-host install` (the writer runs it in the container: `wrapper_command(environ)`).

`answering(url, timeout=ANSWER_TIMEOUT_S) -> str`: "" when the request service answers a GET of URL, the team
dashboard's address, from where it is asked (the doctor: in the container, from the container; check 7):
an answer whose Server header names the service (SERVER's own name), whatever its status: the confirm keeps the address
at the root of the service's port (service_address), where it serves the site folder, an empty one too; else why not, in
a few words: nothing answered (the network's reason), or another server did.

The web host runs SERVER and REMOTE on its own Python, 3.6 or newer: they import nothing from the package.
"""

from __future__ import annotations

import http.client
import ipaddress
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from typing import NamedTuple

from . import container_layout, derived, paths, publish, registry, wrapping
from .coherence import TEAM

TOOLS = paths.PACKAGE / "tools"
# The request service, copied to the web host; the user service that runs it, as its template with each @NAME@ to fill
# in; and the install's half on the web host, which runs there from its standard input.
SERVER = TOOLS / "webhost_server.py"
UNIT_TEMPLATE = TOOLS / "rocke-monitor.service"
REMOTE = TOOLS / "webhost_remote.py"
UNIT = UNIT_TEMPLATE.name  # the user service's name on the web host
# REMOTE's exit code for a step it refuses, which changed nothing (its REFUSED).
REMOTE_REFUSED = 4
SSH_FAILED = 255  # ssh's own exit code: it did not reach the host, or lost it
# ssh's errors, and none of its notices, in what the install prints: OpenSSH 10.1 and later warn at each connection to
# a server that offers no post-quantum key exchange, which says nothing of the install.
SSH_ERRORS_ONLY = ("-o", "LogLevel=ERROR")
NOT_FOUND = 127  # the web host's shell found no python3
# The look: ssh's connect, then a few quick commands on the web host.
LOOK_TIMEOUT_S = 60
# The install on the web host: the server it replaces stopped (10 s at most), the service started and asked.
INSTALL_TIMEOUT_S = 180
# The doctor's question to the service: a connect, then one small answer.
ANSWER_TIMEOUT_S = 10

SETTINGS = ("host", "user", "site", "url")  # the keys of publish.html the plan needs
# A setting a reason of publish.target_refusal names, by the registry's name.
_SETTING = re.compile(r"publish\.html\.(\w+)")
# The service has no TLS: it serves the page over plain http, at the root of a port of its own that the login may
# open, as in EXAMPLE_URL. Below FIRST_PORT, only root's services may open a port on a Linux host.
FIRST_PORT = 1024
EXAMPLE_URL = "http://dashboards.example.com:8080/"
GIVE = f'an http address with a port of {FIRST_PORT} or more and no path, such as "{EXAMPLE_URL}"'
# A host name in the origin, in lower case and its IDNA form: the unit's command line takes it as it is, where systemd
# would change a "%", a "$", a quote or a "\".
HOST_NAME = re.compile(r"[a-z0-9.-]+")
_FILLED = re.compile(r"@([A-Z_]+)@")  # a value of the unit's template to fill in


class Unserved(ValueError):
    """A public address the service does not serve the page at, and GIVE, the address to give in its place ("" when
    none follows from it): see the module's docstring."""

    def __init__(self, text: str, give: str = ""):
        super().__init__(text)
        self.give = give


class Copy(NamedTuple):
    dest: str
    text: str


class Plan(NamedTuple):
    host: str
    user: str
    base: str
    site: str
    requests: str
    port: str
    bind: str
    origin: str
    unit: str
    copies: tuple

    def remote(self, install: bool = True, replace: bool = False) -> list[str]:
        """The command the install runs on the web host: see the module's docstring."""
        mode = "--yes" if install else "--status"
        return [
            "python3",
            "-",
            mode,
            "yes" if replace else "no",
            self.base,
            self.site,
            self.requests,
            self.port,
        ]


def nothing_to_install(cfg: dict) -> str:
    """Why the web host needs nothing installed, "" when it needs the service: see the module's docstring."""
    if publish.mode(cfg) == "serve":
        return (
            "the monitor serves the team dashboard and its priority requests itself "
            f'({_named("team_dashboard.mode")} is "serve")'
        )
    if not publish.takes_requests(cfg):
        return (
            f"priority requests are off ({TEAM}.priority_requests.enabled), and the web host needs the request "
            "service only for them"
        )
    return ""


def deploy_keys() -> tuple:
    """The deploy settings file's keys the plan is made from: see the module's docstring."""
    return (
        _named("team_dashboard.mode"),
        *(_named(f"publish.html.{key}") for key in SETTINGS),
        f"{TEAM}.priority_requests.enabled",
    )


def _named(name: str) -> str:
    """The deploy settings file's key that sets NAME, else NAME."""
    file_keys = registry.key(name).file_keys
    return file_keys[0] if file_keys else name


def _setting(key: str) -> str:
    """publish.html.KEY in the deploy settings file's words: the key that sets it; for a path the site folder gives,
    which no key sets, by the site folder's: the page in it, and the requests file of it (named after it, at the top
    of the login's home).
    """
    site = _named("publish.html.site")
    if key == "requests":
        return f"the requests file of {site}"
    return f"the page in {site}" if key == "dest" else _named(f"publish.html.{key}")


def _html(cfg: dict) -> dict:
    return (cfg.get("publish") or {}).get("html") or {}


def _path(cfg: dict, key: str) -> str:
    """publish.html.KEY without a "/" at its end, a path the writer takes (publish.target_refusal, which checks the
    login's and the host's names too) and each part a plain name, and in the requests file's name each name
    derived.FOLDER_SIGN joins; ValueError otherwise, in the deploy settings file's words (_setting).
    """
    refusal = publish.target_refusal(cfg, key)
    if refusal:
        raise ValueError(_SETTING.sub(lambda m: _setting(m.group(1)), refusal))
    path = str(_html(cfg).get(key) or "").rstrip("/")
    names = path.replace(derived.FOLDER_SIGN, "/") if key == "requests" else path
    odd = [part for part in names.split("/") if not publish.PLAIN_NAME.fullmatch(part)]
    if odd:
        part = (
            f"the part {json.dumps(odd[0], ensure_ascii=False)}"
            if odd[0]
            else "an empty part"
        )
        raise ValueError(
            f"{_setting(key)} has {part}: the web host's service takes a path whose every part is a plain name "
            '(letters, digits, ".", "_" and "-")'
        )
    return path


def unit_text(template: str, values: dict) -> str:
    """TEMPLATE with each @NAME@ replaced by VALUES[NAME]; ValueError for a NAME that VALUES lacks."""
    left = sorted(set(_FILLED.findall(template)) - set(values))
    if left:
        raise ValueError(
            f"the unit's template {UNIT_TEMPLATE.name} has {wrapping.joined(f'@{n}@' for n in left)}, which the "
            "plan does not fill in"
        )
    return _FILLED.sub(lambda m: values[m.group(1)], template)


def service_address(text: str) -> tuple[int, str]:
    """The public address TEXT's port and origin; Unserved for one the service does not serve the page at: see the
    module's docstring.
    """
    key = _setting("url")
    not_a_host = f'the host of {key} is neither an IP address nor a name of letters, digits, "." and "-"'
    try:
        url = urllib.parse.urlsplit(text)
    except ValueError:  # brackets that hold no IPv6 address
        raise Unserved(not_a_host) from None
    try:
        port = url.port or 0
    except ValueError:  # not a number, or above 65535
        port = 0
    if url.scheme != "http" or not FIRST_PORT <= port <= 65535:
        raise Unserved(
            f"{key} is not an http:// address with a port from {FIRST_PORT} to 65535, such as {EXAMPLE_URL}: the web "
            "host's request service serves the page itself, over plain http, on a port its login may open"
        )
    host = url.hostname  # in lower case; an IPv6 address without its brackets
    if not host:
        raise Unserved(f"{key} names no host")
    if url.netloc.rpartition("@")[2].startswith("["):
        try:
            address = ipaddress.IPv6Address(host)
        except ValueError:
            address = None
        # A zone, such as %25eth0, is one no browser takes.
        if address is None or address.scope_id:
            raise Unserved(not_a_host)
        host = f"[{address.compressed}]"
    else:
        try:
            host = host.encode("idna").decode("ascii")
        except UnicodeError:  # an empty or overlong part
            raise Unserved(not_a_host) from None
        if not HOST_NAME.fullmatch(host):
            raise Unserved(not_a_host)
    origin = f"http://{host}:{port}"
    if url.path not in ("", "/"):
        raise Unserved(
            f"{key} has the path {json.dumps(url.path, ensure_ascii=False)}, and the web host's request service "
            f"serves the page at the root of its own port, {origin}/",
            give=f"{origin}/",
        )
    return port, origin


def plan(cfg: dict) -> Plan:
    """What the install puts on the web host: see the module's docstring."""
    why = nothing_to_install(cfg)
    if why:
        raise ValueError(why)
    pub = _html(cfg)
    blank = [_setting(key) for key in SETTINGS if not str(pub.get(key) or "").strip()]
    if blank:
        raise ValueError(f"{wrapping.joined(blank)} {wrapping.are(blank)} blank")
    site, requests = _path(cfg, "site"), _path(cfg, "requests")
    port, origin = service_address(str(pub["url"]).strip())
    values = {
        "BASE": site.split("/")[0],
        "SITE": site,
        "REQUESTS": requests,
        "PORT": str(port),
        "BIND": container_layout.EVERY_ADDRESS,
        "ORIGIN": origin,
    }
    unit = unit_text(UNIT_TEMPLATE.read_text(encoding="utf-8"), values)
    base = values["BASE"]
    return Plan(
        host=str(pub["host"]),
        user=str(pub["user"]),
        base=base,
        site=site,
        requests=requests,
        port=values["PORT"],
        bind=values["BIND"],
        origin=values["ORIGIN"],
        unit=unit,
        copies=(
            Copy(f"{base}/{SERVER.name}", SERVER.read_text(encoding="utf-8")),
            Copy(f"{base}/{UNIT}.new", unit),
        ),
    )


class _Left:
    """What the install leaves on the web host so far, which INTERRUPTED says at a Ctrl-C: see the module's
    docstring."""

    words = "nothing changed"


def install(cfg: dict, run=subprocess.run, out=None) -> int:
    """`rk web-host-install`: see the module's docstring."""
    from . import outcome

    out = sys.stdout if out is None else out

    def finish(word: str, text: str, **how) -> int:
        return outcome.finish(word, text, out=out, **how)

    left = _Left()
    try:
        return _install(cfg, run, out, finish, left)
    except KeyboardInterrupt:
        return finish("INTERRUPTED", left.words)


def _install(cfg: dict, run, out, finish, left: _Left) -> int:
    """install's steps, each outcome FINISH's, LEFT saying what they leave on the web host as they go."""
    from . import features, host_keys, logins, states, writer
    from . import secrets_registry as sr
    from .doctor import cannot_run_fix

    if not publish.key_here(cfg):
        return finish(
            "REFUSED", f"{writer.from_host([writer.WEB_HOST_INSTALL])}; nothing changed"
        )
    if features.standby():
        return finish(
            "REFUSED",
            "the monitor is in standby, and the monitor you moved from may still use the web host; nothing changed",
            fix=states.standby_fix(),
        )
    why = nothing_to_install(cfg)
    if why:
        return finish("NOTHING TO DO", why)
    try:
        p = plan(cfg)
        script = REMOTE.read_text(encoding="utf-8")
    except ValueError as exc:
        return finish("FAILED", f"{exc}; nothing changed", fix=cannot_run_fix())
    except OSError as exc:
        return finish(
            "ERROR", f"a file of the package cannot be read ({exc}); nothing changed"
        )
    entry = sr.entry(cfg, host_keys.WEB.secret)
    if not logins.look(cfg, entry)["set"]:
        return finish(
            "FAILED",
            f"{entry.name} is not set; nothing changed",
            fix=sr.set_fix(entry),
            earlier_step=True,
        )
    if not host_keys.recorded(cfg, host_keys.WEB):
        return finish(
            "FAILED",
            f"{host_keys.WEB.words}'s own key is not recorded, so ssh does not reach {p.host}; nothing changed",
            fix=_record_fix(cfg, entry),
            earlier_step=True,
        )
    target = f"{p.user}@{p.host}"
    ssh = [*publish.ssh_argv(cfg), *SSH_ERRORS_ONLY, target]
    try:
        look = _on_host(run, ssh, p.remote(install=False), script, LOOK_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return finish(
            "FAILED",
            f"the look at what serves port {p.port} on {p.host} took more than {LOOK_TIMEOUT_S} seconds; nothing "
            "changed",
        )
    except OSError as exc:
        return finish("ERROR", f"ssh did not start ({exc}); nothing changed")
    said = _last_line(look.stdout)
    if look.returncode == SSH_FAILED:
        return finish("FAILED", f"ssh did not reach {target} ({said}); nothing changed")
    if look.returncode == NOT_FOUND:
        return finish(
            "FAILED",
            f"{p.host} has no python3, which the request service runs on (3.6 or newer); nothing changed",
        )
    # REMOTE's refusal at the look, of a python3 too old: its last line says so, and that nothing changed
    if look.returncode == REMOTE_REFUSED:
        return finish("FAILED", f"{p.host}: {said}")
    if look.returncode != 0:
        return finish(
            "FAILED",
            f"python3 on {p.host} did not look at what serves port {p.port} ({said}); nothing changed",
        )
    left.words = f"some of the copies may be on {p.host}, and {UNIT} is as it was"
    with tempfile.TemporaryDirectory() as folder:
        for copy in p.copies:
            where, _, name = copy.dest.rpartition("/")
            src = os.path.join(folder, name)
            with open(src, "w", encoding="utf-8") as f:
                f.write(copy.text)
            # as the page's push copies it: written whole, then renamed; mode 644
            why = publish._rsync(cfg, src, f"{target}:{where}/", run, "--chmod=F644")
            if why:
                return finish("ERROR", f"{copy.dest} was not copied to {p.host}: {why}")
    left.words = f"the install on {p.host} stopped part-way, and {UNIT} may not run"
    try:
        done = _on_host(run, ssh, p.remote(), script, INSTALL_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        _print(out, exc.output)
        return finish(
            "ERROR",
            f"the install on {p.host} took more than {INSTALL_TIMEOUT_S} seconds, and {UNIT} may not run",
        )
    except OSError as exc:
        return finish(
            "ERROR",
            f"ssh did not start ({exc}): the copies are on {p.host}, and {UNIT} is as it was",
        )
    lines = (done.stdout or "").splitlines()
    if done.returncode == REMOTE_REFUSED and lines:
        _print(out, lines[:-1])
        return finish("FAILED", f"{p.host}: {lines[-1]}")
    _print(out, lines)
    if done.returncode == 0:
        return finish(
            "INSTALLED",
            f"the request service on {p.host}, {p.user}'s user service {UNIT}, answers on port {p.port}",
        )
    if done.returncode == SSH_FAILED:
        return finish(
            "ERROR", f"ssh lost {target} during the install ({_last_line(done.stdout)})"
        )
    if done.returncode == 1:
        return finish(
            "ERROR", f"{UNIT} did not come up on {p.host} (the lines above say why)"
        )
    return finish(
        "ERROR",
        f"the install on {p.host} exited {done.returncode} (the lines above say why)",
    )


def _record_fix(cfg: dict, entry):
    """How the web host's own key comes to be recorded: `login` while something that is on needs the key, as the
    logins' walk then records it (rkmon_lib/logins.py); else the key set again, which shows the web host's own key and
    records it once you accept it (docker/rkmon secret)."""
    from . import checks, host_keys, secrets_registry as sr
    from .doctor import wrapper_name

    if sr.needed_by(cfg, entry):
        return host_keys.record_fix()
    return checks.Steps(
        (
            checks.Command((wrapper_name(), "secret", entry.name, "FILE")),
            "sets the key again, shows the web host's own key, and records it once you accept it",
        )
    )


def _on_host(run, ssh: list, argv: list, script: str, timeout: int):
    """ARGV run on the web host by SSH (ssh's words, the login and host last), SCRIPT on its standard input: the
    finished process, what it printed and its errors as one text."""
    return run(
        [*ssh, shlex.join(argv)],
        input=script,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )


def _last_line(text) -> str:
    """The last line of TEXT that says anything, at most 200 characters; "no output" for none."""
    lines = [x.strip() for x in (text or "").splitlines() if x.strip()]
    return lines[-1][:200] if lines else "no output"


def _print(out, text) -> None:
    """TEXT, lines or what a process printed, on OUT, a line each."""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    lines = text.splitlines() if isinstance(text, str) else list(text or ())
    out.write("".join(f"{x}\n" for x in lines))
    out.flush()


def install_command(environ=os.environ):
    """The command that installs the request service, as the operator types it on the host: see the module's
    docstring."""
    return wrapper_command(environ)


def wrapper_command(environ=os.environ):
    """`<wrapper> web-host install`, the container's, which the operator types on the host: see the module's
    docstring."""
    from . import checks
    from .doctor import wrapper_name

    return checks.Command((wrapper_name(environ), "web-host", "install"))


def answering(url: str, timeout: float = ANSWER_TIMEOUT_S) -> str:
    """Whether the request service answers at URL, "" when it does: see the module's docstring."""
    from . import webserve

    name = webserve._webhost().Handler.server_version
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "rkmon-doctor"}, method="GET"
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            server = r.headers.get("Server") or ""
    except urllib.error.HTTPError as exc:  # an answer, of a status other than 2xx
        with exc:
            server = exc.headers.get("Server") or ""
    except OSError as exc:  # nothing answered (urllib's URLError among them)
        return f"{type(exc).__name__}: {getattr(exc, 'reason', exc)}"[:160]
    except (ValueError, http.client.HTTPException) as exc:  # no address, or no HTTP
        return f"{type(exc).__name__}: {exc}"[:160]
    if server.split()[:1] == [name]:
        return ""
    if not server:
        return "another server answers there, naming none"
    return f"another server answers there: {server}"[:160]
