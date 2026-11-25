# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""What `rkmon web-host install` puts on the team's web host (rkmon_lib/web_host_install.py): the plan
for each delivery and box state, from the running configuration alone; the settings it refuses, an address with a path
among them; the unit's text; the install's steps and outcomes over a stand-in for ssh and rsync, each failure
naming its step, a Ctrl-C what it leaves on the web host, and nothing run in standby, while the monitor serves the
page, with the box off, or where the web host's key is not read; the remote half (tools/webhost_remote.py, run here in
a stand-in web host's home): its look, which changes nothing, its refusals, which change nothing and exit
REMOTE_REFUSED, an older Python's at the look, and its question to the port, past any proxy; the install end to end on
that stand-in web host, the service answering; the doctor's row, which passes, then fails once the
service stops; and the one installer, the command the skill names, the tool it replaced gone.
"""

import functools
import http.server
import importlib.util
import io
import json
import os
import pathlib
import re
import shlex
import stat
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.dont_write_bytecode = True

import sandbox  # noqa: E402
from rendered import finished, one_line  # noqa: E402
from rkmon_lib import doctor, gh, publish, registry, state_store, webserve  # noqa: E402
from rkmon_lib import web_host_install as w  # noqa: E402
from rkmon_lib import writer  # noqa: E402

# The page's address: the root of the request service's port, where it serves the site folder (the unit's --directory).
URL = "http://web.example.com:8080/"
# The requests file the site folder rocke-monitor/site gives: at the top of the login's home, never in rocke-monitor,
# which a web server may publish.
REQUESTS = "rocke-monitor+site-requests.jsonl"
UNIT_LINE = (
    "ExecStart=/usr/bin/env python3 %h/rocke-monitor/webhost_server.py --directory %h/rocke-monitor/site --requests "
    "%h/rocke-monitor+site-requests.jsonl --port 8080 --bind 0.0.0.0 --origin http://web.example.com:8080"
)
SERVE = (
    "the monitor serves the team dashboard and its priority requests itself (features.dashboards.team.delivery is "
    '"serve")'
)
BOX_OFF = (
    "priority requests are off (features.dashboards.team.priority_requests.enabled), and the web host needs the "
    "request service only for them"
)
SITE = "features.dashboards.team.publish.site_folder"
PLAIN = 'the web host\'s service takes a path whose every part is a plain name (letters, digits, ".", "_" and "-")'
HTTP = (
    "features.dashboards.team.public_url is not an http:// address with a port from 1024 to 65535, such as "
    "http://dashboards.example.com:8080/: the web host's request service serves the page itself, over plain http, on a "
    "port its login may open"
)
# An address with a path, which the service does not serve the page at: the path, then the address to give.
PATH = (
    "features.dashboards.team.public_url has the path {}, and the web host's request service serves the page at the "
    "root of its own port, {}"
)
NOT_A_HOST = (
    'the host of features.dashboards.team.public_url is neither an IP address nor a name of letters, digits, "." '
    'and "-"'
)
# The package's own files the plan may read: its version and defaults, and the files of its schemas and its tools.
OWN = ("VERSION", "config.defaults.json")
OWN_FOLDERS = ("schema", "tools")
# A Python program that spies on each file it opens (open, io.open and os.open), then imports the module from the
# package in argv[1] and makes the plan for the running configuration in argv[2], and prints the files it opened.
SPY = """
import builtins, io, json, os, sys

opened = []
real_open, real_os_open = io.open, os.open


def spy(file, *args, **kwargs):
    if not isinstance(file, int):
        opened.append(os.fsdecode(file))
    return real_open(file, *args, **kwargs)


def spy_os(path, *args, **kwargs):
    opened.append(os.fsdecode(path))
    return real_os_open(path, *args, **kwargs)


builtins.open = io.open = spy
os.open = spy_os
sys.path.insert(0, sys.argv[1])
from rkmon_lib import web_host_install

web_host_install.plan(json.loads(sys.argv[2]))
print(json.dumps(opened))
"""


def cfg(delivery: str = "publish", box: bool = True, **html) -> dict:
    """A running configuration by the registry's names: the team dashboard on, its site folder rocke-monitor/site,
    and the page and the requests file where the site folder puts them (the requests file only while the box is on).
    """
    values = {
        "features.dashboards.team.enabled": True,
        "team_dashboard.mode": delivery,
        "publish.html.url": URL,
        "publish.html.host": "web.example.com",
        "publish.html.user": "op",
        "publish.html.site": "rocke-monitor/site",
        "publish.html.dest": "rocke-monitor/site/index.html",
        "publish.html.requests": REQUESTS if box else "",
    }
    values.update({f"publish.html.{k}": v for k, v in html.items()})
    return registry.nest(values)


class TestThePlan(unittest.TestCase):
    def refusal(self, c: dict) -> str:
        with self.assertRaises(ValueError) as caught:
            w.plan(c)
        return str(caught.exception)

    def test_the_plan_for_each_delivery_and_box_state(self):
        server = (ROOT / "tools" / "webhost_server.py").read_text(encoding="utf-8")
        unit = (ROOT / "tools" / "rocke-monitor.service").read_text(encoding="utf-8")
        for name, value in {
            "BASE": "rocke-monitor",
            "SITE": "rocke-monitor/site",
            "REQUESTS": REQUESTS,
            "PORT": "8080",
            "BIND": "0.0.0.0",
            "ORIGIN": "http://web.example.com:8080",
        }.items():
            unit = unit.replace(f"@{name}@", value)
        p = w.plan(cfg("publish", box=True))
        self.assertEqual(
            p,
            w.Plan(
                host="web.example.com",
                user="op",
                base="rocke-monitor",
                site="rocke-monitor/site",
                requests=REQUESTS,
                port="8080",
                bind="0.0.0.0",
                origin="http://web.example.com:8080",
                unit=unit,
                copies=(
                    w.Copy("rocke-monitor/webhost_server.py", server),
                    w.Copy("rocke-monitor/rocke-monitor.service.new", unit),
                ),
            ),
        )
        self.assertEqual(
            [ln for ln in p.unit.splitlines() if ln.startswith("ExecStart=")],
            [UNIT_LINE],
        )
        self.assertEqual(w.nothing_to_install(cfg("publish", box=True)), "")
        for delivery, box, why in (
            ("publish", False, BOX_OFF),
            ("serve", True, SERVE),
            ("serve", False, SERVE),
        ):
            with self.subTest(delivery=delivery, box=box):
                self.assertEqual(w.nothing_to_install(cfg(delivery, box)), why)
                self.assertEqual(self.refusal(cfg(delivery, box)), why)

    def test_the_deploy_keys_the_plan_is_made_from(self):
        """The deploy settings file's keys the plan is made from (deploy_keys), under whose change the confirm names
        the install: the delivery and the box's switch, which decide whether there is a plan, and the keys of the web
        host's settings, each of which changes the plan."""
        self.assertEqual(
            w.deploy_keys(),
            (
                "features.dashboards.team.delivery",
                "features.dashboards.team.publish.web_host",
                "features.dashboards.team.publish.username",
                SITE,
                "features.dashboards.team.public_url",
                "features.dashboards.team.priority_requests.enabled",
            ),
        )
        self.assertEqual(w.nothing_to_install(cfg()), "")
        self.assertEqual(
            [w.nothing_to_install(cfg("serve")), w.nothing_to_install(cfg(box=False))],
            [SERVE, BOX_OFF],
        )
        made = w.plan(cfg())
        for key, value in (
            ("host", "other.example.com"),
            ("user", "op2"),
            ("site", "rocke-monitor/other"),
            ("url", "http://web.example.com:8081/"),
        ):
            with self.subTest(key=key):
                self.assertNotEqual(w.plan(cfg(**{key: value})), made)

    def test_the_command_on_the_web_host(self):
        p = w.plan(cfg())
        tail = ["rocke-monitor", "rocke-monitor/site", REQUESTS, "8080"]
        self.assertEqual(p.remote(), ["python3", "-", "--yes", "no", *tail])
        self.assertEqual(
            p.remote(install=False, replace=True),
            ["python3", "-", "--status", "yes", *tail],
        )

    def test_the_port_and_the_origin_come_from_the_public_address(self):
        # The origin as a browser sends it, which the service compares as it is.
        for url, port, origin in (
            (URL, "8080", "http://web.example.com:8080"),
            ("http://web.example.com:1024/", "1024", "http://web.example.com:1024"),
            ("http://web.example.com:65535", "65535", "http://web.example.com:65535"),
            ("HTTP://Web.Example.COM:8080/", "8080", "http://web.example.com:8080"),
            (
                "http://Bücher.example.com:8080/",
                "8080",
                "http://xn--bcher-kva.example.com:8080",
            ),
            ("http://192.0.2.10:8080/", "8080", "http://192.0.2.10:8080"),
            ("http://[2001:DB8:0:0::1]:8080/", "8080", "http://[2001:db8::1]:8080"),
            (  # a login in the address is no part of the origin a browser sends
                "http://op:pw@web.example.com:8080/",
                "8080",
                "http://web.example.com:8080",
            ),
        ):
            with self.subTest(url=url):
                p = w.plan(cfg(url=url))
                self.assertEqual((p.port, p.origin), (port, origin))
                [line] = [
                    ln for ln in p.unit.splitlines() if ln.startswith("ExecStart=")
                ]
                self.assertTrue(
                    line.endswith(f" --port {port} --bind 0.0.0.0 --origin {origin}"),
                    line,
                )

    def test_the_settings_it_refuses(self):
        self.assertEqual(
            self.refusal(cfg(host="", user="", site="", url=" ")),
            "features.dashboards.team.publish.web_host, features.dashboards.team.publish.username, "
            "features.dashboards.team.publish.site_folder and features.dashboards.team.public_url are blank",
        )
        self.assertEqual(
            self.refusal(cfg(user="")),
            "features.dashboards.team.publish.username is blank",
        )
        # Each named by the deploy file's key, and the requests file, which no key sets, by the site folder.
        elsewhere = (
            f"the requests file of {SITE} is not the requests file named after the folder of the page in {SITE}, at "
            "the top of the login's home"
        )
        for html, why in (
            (
                dict(site="/srv/site"),
                f"{SITE} starts with /: the writer takes a path inside the web host login's home",
            ),
            (
                dict(site="rocke-monitor/../.ssh"),
                f"{SITE} has a .., dot-folder or dot-file part",
            ),
            (
                dict(user="-oProxyCommand=x"),
                "features.dashboards.team.publish.username is not a plain name",
            ),
            (
                dict(host="web host"),
                "features.dashboards.team.publish.web_host is not a plain name",
            ),
            (
                dict(site="rocke-monitor/my site"),
                f'{SITE} has the part "my site": {PLAIN}',
            ),
            (dict(site="rocke-monitor//site"), f"{SITE} has an empty part: {PLAIN}"),
            (dict(site="rocke-monitor/%u"), f'{SITE} has the part "%u": {PLAIN}'),
            (
                dict(requests="/srv/site-requests.jsonl"),
                f"the requests file of {SITE} starts with /: the writer takes a path inside the web host login's home",
            ),
            # Any place but the one the site folder gives: beside the site folder too, in a folder its web server
            # may publish.
            (dict(requests="srv/site-requests.jsonl"), elsewhere),
            (dict(requests="rocke-monitor/site-requests.jsonl"), elsewhere),
            (dict(requests="site-requests.jsonl"), elsewhere),
            (  # a page's folder with a part that is not a plain name gives a name the unit cannot take
                dict(
                    dest="rocke-monitor/my site/index.html",
                    requests="rocke-monitor+my site-requests.jsonl",
                ),
                f'the requests file of {SITE} has the part "my site-requests.jsonl": {PLAIN}',
            ),
            (  # no host: in two parts, as the privacy check reads the port as a host
                dict(url="http://" + ":8080/rocke-monitor/"),
                "features.dashboards.team.public_url names no host",
            ),
        ):
            with self.subTest(html=html):
                self.assertEqual(self.refusal(cfg(**html)), why)
        # The service serves the page itself, over plain http, on a port of its own its login may open.
        for url in (
            "https://dash.example.com/team/",
            "https://dash.example.com:8443/team/",
            "http://web.example.com/rocke-monitor/site/",  # no port: http's own, 80
            "http://web.example.com:/rocke-monitor/site/",
            "http://web.example.com:80/rocke-monitor/site/",
            "http://web.example.com:0/rocke-monitor/site/",
            "http://web.example.com:1023/rocke-monitor/site/",
            "http://web.example.com:65536/rocke-monitor/site/",
            "http://web.example.com:99999/rocke-monitor/site/",
            "web.example.com:8080/rocke-monitor/site/",  # no scheme
            "ftp://web.example.com:8080/rocke-monitor/site/",
        ):
            with self.subTest(url=url):
                self.assertEqual(self.refusal(cfg(url=url)), HTTP)
        # The unit's command line takes the origin as it is, where systemd would change a "%", a "$", a quote or a "\".
        for url in (
            "http://web.example.com%h:8080/",
            "http://web${HOME}.example.com:8080/",
            'http://"web".example.com:8080/',
            "http://web\\.example.com:8080/",
            "http://web_1.example.com:8080/",
            # A part longer than 63 letters, then an IPv6 address's zone, which no browser takes.
            "http://" + "a" * 64 + ".example.com:8080/",
            "http://[fe80::1%25eth0]:8080/",
            "http://[2001:db8::1:8080/",
            "http://[v1.x]:8080/",
        ):
            with self.subTest(url=url):
                self.assertEqual(self.refusal(cfg(url=url)), NOT_A_HOST)
        # The service serves the site folder at the root of its own port: at a path, the alerts would link to a page
        # it does not serve. The refusal names the address to give (Unserved.give); the others name none.
        for url, path in (
            (
                "http://web.example.com:8080/rocke-monitor/site/",
                '"/rocke-monitor/site/"',
            ),
            ("http://Web.Example.COM:8080/my-team", '"/my-team"'),
            ("http://web.example.com:8080/index.html", '"/index.html"'),
            ("http://web.example.com:8080//", '"//"'),
        ):
            with self.subTest(url=url):
                with self.assertRaises(w.Unserved) as caught:
                    w.plan(cfg(url=url))
                self.assertEqual(
                    (str(caught.exception), caught.exception.give),
                    (
                        PATH.format(path, "http://web.example.com:8080/"),
                        "http://web.example.com:8080/",
                    ),
                )
        for url in ("https://dash.example.com/team/", "http://web_1.example.com:8080/"):
            with self.subTest(url=url):
                with self.assertRaises(w.Unserved) as caught:
                    w.service_address(url)
                self.assertEqual(caught.exception.give, "")
        p = w.plan(
            cfg(site="rocke-monitor/site/")
        )  # a "/" at the end is no part of the folder
        self.assertEqual((p.site, p.base), ("rocke-monitor/site", "rocke-monitor"))
        # The requests file's name joins the site folder's names with "+", which no plain name holds.
        p = w.plan(
            cfg(site="a/b/c", dest="a/b/c/index.html", requests="a+b+c-requests.jsonl")
        )
        self.assertEqual(
            (p.site, p.base, p.requests), ("a/b/c", "a", "a+b+c-requests.jsonl")
        )

    def test_the_unit_s_text_fills_in_each_value_its_template_names(self):
        self.assertEqual(
            w.unit_text("a @X@ b @Y@ @X@", {"X": "1", "Y": "2"}), "a 1 b 2 1"
        )
        with self.assertRaises(ValueError) as caught:
            w.unit_text("--port @PORT@ --user @USER@ @NAME@", {"PORT": "8080"})
        self.assertEqual(
            str(caught.exception),
            "the unit's template rocke-monitor.service has @NAME@ and @USER@, which the plan does not fill in",
        )

    def test_no_configuration_file_is_read(self):
        """A fresh Python that spies on each file it opens, from its start, imports the module and makes the plan: it
        opens only the package's own files (OWN), the server and the unit's template among them.
        """
        with tempfile.TemporaryDirectory() as tmp:
            p = subprocess.run(
                [sys.executable, "-B", "-c", SPY, str(ROOT), json.dumps(cfg())],
                capture_output=True,
                text=True,
                env={"PATH": "/usr/bin:/bin", "HOME": tmp, "LANG": "C.UTF-8"},
                cwd=tmp,
                timeout=120,
            )
            self.assertEqual(p.returncode, 0, p.stderr)
            opened = [pathlib.Path(tmp, f) for f in json.loads(p.stdout)]

        def own(f: pathlib.Path) -> bool:
            if not f.is_relative_to(ROOT):
                return False
            name = f.relative_to(ROOT).as_posix()
            return name in OWN or name.startswith(tuple(f"{d}/" for d in OWN_FOLDERS))

        self.assertEqual([str(f) for f in opened if not own(f)], [])
        self.assertIn(w.SERVER, opened)
        self.assertIn(w.UNIT_TEMPLATE, opened)


# The web host's own key, as the state records it (publish.html.host_keys): the only key ssh trusts the web host by.
LINE = "web.example.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExamplePublicKeyNotReal0000000000000000"
TARGET = "op@web.example.com"
# The remote half's look on a web host where nothing serves the page's port, and the last lines of its install.
LOOK = "== web: port 8080\n(nothing listens on 8080)\n"
RUNNING = "== deploy\nrocke-monitor.service is running: answers: HTTP 200, Server rkmon-webhost Python/3.6.8\n"
INSTALLED = (
    "INSTALLED: the request service on web.example.com, op's user service rocke-monitor.service, answers on "
    "port 8080"
)
# rsync's line when the web host refuses a copy.
MKDIR = 'rsync: [Receiver] mkdir "rocke-monitor" failed: Permission denied (13)\n'
# The remote half's refusal of a python3 older than the service runs on: at the look, before the install copies
# anything.
OLD_PYTHON = (
    "python3 here is 3.5, and the request service needs 3.6 or newer; nothing changed"
)
HOST_ENV = {"RKMON_CONTAINER": ""}  # a host install
IMAGE_ENV = {
    "RKMON_CONTAINER": "1",
    "RKMON_WRAPPER": "rkmon",
}  # the container, its wrapper run as rkmon
# The variables that name a proxy, which would carry the doctor's question elsewhere: none here.
PROXIES = (
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "no_proxy",
    "NO_PROXY",
    "all_proxy",
    "ALL_PROXY",
)


class Host:
    """A stand-in for subprocess.run as the install runs ssh and rsync: each call, (its words, its keywords), in CALLS;
    each file an rsync copies, (its name, its text) as the copy reads it, in COPIED; and ANSWERS, given in turn: each
    (exit code, stdout, stderr), or an exception to raise. A call past the last answer is an IndexError.
    """

    def __init__(self, *answers):
        self.answers, self.calls, self.copied = list(answers), [], []

    def __call__(self, argv, **kw):
        self.calls.append((list(argv), kw))
        if argv[0] == "rsync":
            src = pathlib.Path(argv[-2])
            self.copied.append((src.name, src.read_text(encoding="utf-8")))
        answer = self.answers.pop(0)
        if isinstance(answer, BaseException):
            raise answer
        rc, out, err = answer
        return subprocess.CompletedProcess(argv, rc, out, err)


class InstallBase(unittest.TestCase):
    """A host install of the monitor (RKMON_CONTAINER empty), its home a temporary folder, no proxy and no writer to
    route through: the web host's key is a file this process reads (publish.html.key).
    """

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.d = pathlib.Path(tmp.name)
        self.home, self.work = self.d / "home", self.d / "work"
        self.home.mkdir()
        self.work.mkdir()
        self.key = self.d / "web_host_ssh_key"
        self.key.write_text("not a real key\n", encoding="utf-8")
        self.key.chmod(0o600)
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in (writer.BROKER_VAR, gh.BROKERED, *PROXIES)
        }
        env.update(RKMON_CONTAINER="", RKMON_HOME=str(self.home), RKMON_WRAPPER="rkmon")
        p = mock.patch.dict(os.environ, env, clear=True)
        p.start()
        self.addCleanup(p.stop)

    def config(self, mode="live", keys=(LINE,), delivery="publish", box=True, **html):
        """cfg's running configuration, the team dashboard in MODE, with the key's file and the web host's own KEYS
        recorded, and the working folder, where ssh's known_hosts file for the web host goes.
        """
        c = cfg(delivery, box, key=str(self.key), host_keys=list(keys), **html)
        c["features"]["dashboards"]["team"]["mode"] = mode
        c["paths"] = {"working_dir": str(self.work)}
        return c

    def install(self, c: dict, run) -> tuple[int, str]:
        """rk web-host-install with C and RUN: its exit code and what it printed."""
        out = io.StringIO()
        rc = w.install(c, run=run, out=out)
        return rc, out.getvalue()


class TestTheInstall(InstallBase):
    """rk web-host-install: its steps over ssh and rsync (stand-ins here: Host), and its outcome."""

    def test_installed_over_the_key_trusting_the_recorded_keys_alone(self):
        """The look, the two copies, then the install, each over the web host's key, ssh trusting the web host by the
        keys the state records alone; the remote half on ssh's standard input. The install's lines, then INSTALLED
        (exit 0); in prototype too, which readies the web host for live."""
        script = w.REMOTE.read_text(encoding="utf-8")
        on_host = dict(
            input=script, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        for mode in ("live", "prototype"):
            with self.subTest(mode=mode):
                c = self.config(mode)
                host = Host(
                    (0, LOOK, None), (0, "", ""), (0, "", ""), (0, RUNNING, None)
                )
                self.assertEqual(
                    self.install(c, host), (0, RUNNING + finished(INSTALLED))
                )
                p = w.plan(c)
                ssh = publish.ssh_argv(c)
                self.assertEqual(ssh[:3], ["ssh", "-i", str(self.key)])
                for option in (
                    "BatchMode=yes",
                    "StrictHostKeyChecking=yes",
                    f"UserKnownHostsFile={self.work}/web_host_known_hosts",
                    f"GlobalKnownHostsFile={os.devnull}",
                ):
                    self.assertIn(option, ssh)
                self.assertEqual(
                    (self.work / "web_host_known_hosts").read_text(), LINE + "\n"
                )
                (look, look_kw), *copies, (deploy, deploy_kw) = host.calls
                self.assertEqual(
                    (look, look_kw),
                    (
                        [
                            *ssh,
                            "-o",
                            "LogLevel=ERROR",
                            TARGET,
                            shlex.join(p.remote(install=False)),
                        ],
                        dict(on_host, timeout=w.LOOK_TIMEOUT_S),
                    ),
                )
                self.assertEqual(
                    look[-1],
                    "python3 - --status no rocke-monitor rocke-monitor/site rocke-monitor+site-requests.jsonl 8080",
                )
                self.assertEqual(
                    (deploy, deploy_kw),
                    (
                        [*ssh, "-o", "LogLevel=ERROR", TARGET, shlex.join(p.remote())],
                        dict(on_host, timeout=w.INSTALL_TIMEOUT_S),
                    ),
                )
                self.assertEqual(
                    deploy[-1],
                    "python3 - --yes no rocke-monitor rocke-monitor/site rocke-monitor+site-requests.jsonl 8080",
                )
                # Each copy as the page's push copies it: written whole, then renamed; mode 644; from a temporary
                # folder that is gone once the install ends.
                for (argv, kw), name in zip(
                    copies, ("webhost_server.py", "rocke-monitor.service.new")
                ):
                    self.assertEqual(
                        argv[:5],
                        ["rsync", "-a", "--chmod=F644", "-e", shlex.join(ssh)],
                    )
                    self.assertEqual(
                        (pathlib.Path(argv[5]).name, argv[6:]),
                        (name, [f"{TARGET}:rocke-monitor/"]),
                    )
                    self.assertEqual(
                        kw,
                        dict(
                            capture_output=True,
                            text=True,
                            timeout=publish.RSYNC_TIMEOUT_S,
                        ),
                    )
                    self.assertFalse(pathlib.Path(argv[5]).parent.exists())
                self.assertEqual(
                    host.copied,
                    [(pathlib.Path(x.dest).name, x.text) for x in p.copies],
                )

    def test_nothing_to_install_while_the_monitor_serves_the_page_or_the_box_is_off(
        self,
    ):
        for c, why in (
            (self.config(delivery="serve"), SERVE),
            (self.config(box=False), BOX_OFF),
        ):
            with self.subTest(why=why):
                host = Host()
                self.assertEqual(
                    self.install(c, host), (0, finished(f"NOTHING TO DO: {why}"))
                )
                self.assertEqual(host.calls, [])

    def test_refused_in_standby_and_nothing_runs(self):
        """The monitor you moved from may still use the web host."""
        state_store.set(self.home, "standby.on", True)
        host = Host()
        rc, out = self.install(self.config(), host)
        self.assertEqual(
            (rc, one_line(out)),
            (
                4,
                "REFUSED: the monitor is in standby, and the monitor you moved from may still use the web host; "
                "nothing changed fix: rkmon standby off, once rkmon status and the dashboards read right",
            ),
        )
        self.assertEqual(host.calls, [])

    def test_the_monitor_s_rk_in_the_container_refuses_it(self):
        """In the container the writer runs it, as the host asks (rkmon web-host install): the monitor's rk, which does
        not read the web host's key, refuses it, and nothing runs."""
        host = Host()
        with mock.patch.dict(
            os.environ,
            {
                writer.BROKER_VAR: "/opt/rkmon/docker/rk_write.py",
                "RKMON_CONTAINER": "1",
            },
        ):
            rc, out = self.install(self.config(), host)
        self.assertEqual(
            (rc, one_line(out)),
            (
                4,
                "REFUSED: the request service is installed on the web host from the host only: rkmon web-host "
                "install; nothing changed",
            ),
        )
        self.assertEqual(host.calls, [])

    def test_an_earlier_step_not_done_exits_3_and_nothing_runs(self):
        """The web host's own key not recorded, or a key of another host alone, which is none of its keys; the web
        host's key not set: FAILED, exit 3, its fix the command that does it. In live, the logins' walk records the
        host's key (rkmon login); in prototype, where nothing else needs the key yet, the key set again.
        """
        other = LINE.replace("web.example.com", "web2.example.com", 1)
        unrecorded = (
            "FAILED: the web host's own key is not recorded, so ssh does not reach web.example.com; nothing changed "
            "fix: "
        )
        login = unrecorded + "rkmon login shows it, and records it once you accept it"
        for c, said in (
            (self.config(keys=()), login),
            (self.config(keys=(other,)), login),
            (
                self.config("prototype", keys=()),
                unrecorded
                + "rkmon secret web_host_ssh_key FILE sets the key again, shows the web host's own key, and "
                "records it once you accept it",
            ),
        ):
            with self.subTest(said=said):
                host = Host()
                rc, out = self.install(c, host)
                self.assertEqual((rc, one_line(out)), (3, said))
                self.assertEqual(host.calls, [])
        self.key.unlink()
        host = Host()
        rc, out = self.install(self.config(), host)
        self.assertEqual(
            (rc, one_line(out)),
            (
                3,
                "FAILED: web_host_ssh_key is not set; nothing changed fix: rkmon secret web_host_ssh_key, and give "
                "an SSH private key: the whole key file, from its BEGIN line to its END line",
            ),
        )
        self.assertEqual(host.calls, [])

    def test_a_setting_it_cannot_use_fails_and_nothing_runs(self):
        """Its fix: the settings fixed in the deploy settings file, then confirmed and applied, as the operator types
        it on the host, wherever it runs (in the container, the writer runs it)."""
        c = self.config(url="https://dash.example.com/team/")
        fix = "fix it in your deploy settings file, then rkmon confirm and rkmon apply"
        for env in ({}, IMAGE_ENV):
            with self.subTest(env=env), mock.patch.dict(os.environ, env):
                host = Host()
                rc, out = self.install(c, host)
                self.assertEqual(
                    (rc, one_line(out)),
                    (4, f"FAILED: {HTTP}; nothing changed fix: {fix}"),
                )
                self.assertEqual(host.calls, [])

    def test_each_failure_names_its_step_and_nothing_runs_after_it(self):
        """FAILED (exit 4), nothing changed: ssh does not reach the web host; it has no python3, or one the remote
        half refuses at the look, as too old; the look fails, or takes too long; the remote half refuses a step, which
        its last line names, after its other lines. ERROR (exit 1), part-way: ssh does not start; a copy fails; the
        service does not come up (the remote half has started what it replaced again); ssh loses the web host; the
        install takes too long, or exits with another code.
        """
        look, copied = (0, LOOK, None), (0, "", "")
        no_ssh = FileNotFoundError(2, "No such file or directory", "ssh")
        refused = "port 8080 is held by PID 4242, which is not an http.server; nothing changed"
        down = (
            "rocke-monitor.service did not come up:\n* rocke-monitor.service - the request service\n"
            "started the old server again: python3 -m http.server 8080 (in rocke-monitor/site)\n"
        )
        lost = "Connection to web.example.com closed by remote host.\n"
        usage = "usage: python3 - --status|--yes REPLACE BASE SITE REQUESTS PORT < webhost_remote.py\n"
        for answers, rc, said in (
            (
                [(255, "op@web.example.com: Permission denied (publickey).\n", None)],
                4,
                finished(
                    "FAILED: ssh did not reach op@web.example.com (op@web.example.com: Permission denied "
                    "(publickey).); nothing changed"
                ),
            ),
            (
                [(127, "bash: line 1: python3: command not found\n", None)],
                4,
                finished(
                    "FAILED: web.example.com has no python3, which the request service runs on (3.6 or newer); "
                    "nothing changed"
                ),
            ),
            (
                [(4, f"{OLD_PYTHON}\n", None)],
                4,
                finished(f"FAILED: web.example.com: {OLD_PYTHON}"),
            ),
            (
                [
                    (
                        1,
                        "Traceback (most recent call last):\nOSError: [Errno 28] No space left\n",
                        None,
                    )
                ],
                4,
                finished(
                    "FAILED: python3 on web.example.com did not look at what serves port 8080 (OSError: [Errno 28] "
                    "No space left); nothing changed"
                ),
            ),
            (
                [subprocess.TimeoutExpired("ssh", w.LOOK_TIMEOUT_S)],
                4,
                finished(
                    "FAILED: the look at what serves port 8080 on web.example.com took more than 60 seconds; "
                    "nothing changed"
                ),
            ),
            (
                [no_ssh],
                1,
                finished(
                    "ERROR: ssh did not start ([Errno 2] No such file or directory: 'ssh'); nothing changed"
                ),
            ),
            (
                [look, (23, "", MKDIR)],
                1,
                finished(
                    f"ERROR: rocke-monitor/webhost_server.py was not copied to web.example.com: {MKDIR.strip()}"
                ),
            ),
            (
                [look, copied, (23, "", MKDIR)],
                1,
                finished(
                    "ERROR: rocke-monitor/rocke-monitor.service.new was not copied to web.example.com: "
                    + MKDIR.strip()
                ),
            ),
            (
                [look, copied, copied, (4, f"{LOOK}== deploy\n{refused}\n", None)],
                4,
                f"{LOOK}== deploy\n" + finished(f"FAILED: web.example.com: {refused}"),
            ),
            (
                [look, copied, copied, (1, f"{LOOK}== deploy\n{down}", None)],
                1,
                f"{LOOK}== deploy\n{down}"
                + finished(
                    "ERROR: rocke-monitor.service did not come up on web.example.com (the lines above say "
                    "why)"
                ),
            ),
            (
                [look, copied, copied, (255, lost, None)],
                1,
                lost
                + finished(
                    f"ERROR: ssh lost op@web.example.com during the install ({lost.strip()})"
                ),
            ),
            (
                [look, copied, copied, (2, usage, None)],
                1,
                usage
                + finished(
                    "ERROR: the install on web.example.com exited 2 (the lines above say why)"
                ),
            ),
            (
                [
                    look,
                    copied,
                    copied,
                    subprocess.TimeoutExpired(
                        "ssh", w.INSTALL_TIMEOUT_S, output="== deploy\n"
                    ),
                ],
                1,
                "== deploy\n"
                + finished(
                    "ERROR: the install on web.example.com took more than 180 seconds, and "
                    "rocke-monitor.service may not run"
                ),
            ),
            (
                [look, copied, copied, no_ssh],
                1,
                finished(
                    "ERROR: ssh did not start ([Errno 2] No such file or directory: 'ssh'): the copies are on "
                    "web.example.com, and rocke-monitor.service is as it was"
                ),
            ),
        ):
            with self.subTest(said=said):
                host = Host(*answers)
                self.assertEqual(self.install(self.config(), host), (rc, said))
                self.assertEqual(len(host.calls), len(answers))

    def test_ctrl_c_says_what_it_leaves_on_the_web_host(self):
        """A Ctrl-C: INTERRUPTED, exit 130, saying what is left on the web host. Before the look ends,
        nothing changed; during the copies, some of them may be there, and the unit is as it was; once the remote half
        installs, the unit may not run. Nothing runs after it."""
        look, copied = (0, LOOK, None), (0, "", "")
        copies = "some of the copies may be on web.example.com, and rocke-monitor.service is as it was"
        part_way = "the install on web.example.com stopped part-way, and rocke-monitor.service may not run"
        for before, left in (
            ([], "nothing changed"),
            ([look], copies),
            ([look, copied], copies),
            ([look, copied, copied], part_way),
        ):
            with self.subTest(left=left, before=len(before)):
                host = Host(*before, KeyboardInterrupt())
                self.assertEqual(
                    self.install(self.config(), host),
                    (130, finished(f"INTERRUPTED: {left}")),
                )
                self.assertEqual(len(host.calls), len(before) + 1)
        # Before it asks the web host anything: nothing runs.
        host = Host()
        with mock.patch.object(w, "plan", side_effect=KeyboardInterrupt):
            self.assertEqual(
                self.install(self.config(), host),
                (130, finished("INTERRUPTED: nothing changed")),
            )
        self.assertEqual(host.calls, [])


# The stand-in web host's tools for the remote half: nothing listens on the port (ss prints no line), no unit yet,
# lingering on, and a crontab with a line about the site and one that is not.
TOOLS = {
    "ss": "#!/bin/sh\nexit 0\n",
    "systemctl": '#!/bin/sh\nfor a; do [ "$a" = is-active ] && { echo inactive; exit 3; }; done\nexit 0\n',
    "loginctl": "#!/bin/sh\necho yes\n",
    "crontab": "#!/bin/sh\necho '30 4 * * * cd rocke-monitor/site && python3 -m http.server 8080'\n"
    "echo '0 * * * * unrelated'\n",
}
# A listener this login cannot see: no pid= in ss's line, whose last argument is "sport = :PORT".
HIDDEN = '#!/bin/sh\necho "LISTEN 0 5 0.0.0.0${2#sport = } 0.0.0.0:*"\n'
# User services that stop at logout, and a login that cannot turn lingering on.
LINGER_OFF = '#!/bin/sh\n[ "$1" = enable-linger ] && exit 1\necho no\n'
# python3 as 3.5 would run the remote half, which comes on its standard input.
AS_3_5 = (
    "import sys; sys.version_info = (3, 5, 2, 'final', 0); "
    "exec(compile(sys.stdin.read(), 'webhost_remote.py', 'exec'))"
)


def remote_half():
    """tools/webhost_remote.py as a module: loading it runs nothing, as it runs main only as a program."""
    spec = importlib.util.spec_from_file_location("webhost_remote", w.REMOTE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Answers(http.server.BaseHTTPRequestHandler):
    """Each GET answered with its server's `status` and no body, under server_version's name, logging nothing."""

    server_version = "stand-in/1.0"

    def do_GET(self):
        self.send_response(self.server.status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def log_message(self, *args):
        pass


def served(test: unittest.TestCase, srv) -> str:
    """SRV serving in a thread until TEST ends: its address."""
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    test.addCleanup(t.join, 5)
    test.addCleanup(srv.server_close)
    test.addCleanup(srv.shutdown)
    return f"http://127.0.0.1:{srv.server_address[1]}"


class TestTheRemoteHalf(unittest.TestCase):
    """tools/webhost_remote.py, run here in a stand-in web host's home, with stand-ins for its ss, systemctl, loginctl
    and crontab: its look changes nothing; its install takes the plan's command and finds the plan's copies; and each
    step it refuses exits REMOTE_REFUSED, its last line naming the step, and changes nothing.
    """

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        d = pathlib.Path(tmp.name)
        self.bin, self.home = d / "bin", d / "home"
        self.bin.mkdir()
        (self.home / "rocke-monitor" / "site").mkdir(parents=True)
        self.tools()
        (self.bin / "python3").symlink_to(
            sys.executable
        )  # the web host's python3: this one
        # A port nothing listens on: the look's question to a listener ss shows ends at once.
        self.port = sandbox.free_port()
        self.plan = w.plan(cfg(url=f"http://web.example.com:{self.port}/"))
        self.unit = self.home / ".config" / "systemd" / "user" / w.UNIT

    def tools(self, **more: str):
        """TOOLS, with MORE in their place, on the stand-in web host's PATH."""
        for name, text in dict(TOOLS, **more).items():
            (self.bin / name).write_text(text, encoding="utf-8")
            (self.bin / name).chmod(0o755)

    def listener(self, pid: int) -> str:
        """ss on the stand-in web host: a listener of PID's on the port, which this login sees."""
        users = f'users:(("x",pid={pid},fd=3))'
        return f"#!/bin/sh\necho 'LISTEN 0 5 0.0.0.0:{self.port} 0.0.0.0:* {users}'\n"

    def copies(self):
        """The plan's copies, where its rsync puts them."""
        for copy in self.plan.copies:
            (self.home / copy.dest).write_text(copy.text, encoding="utf-8")

    def remote(self, install: bool = True, python=()) -> tuple[int, list[str]]:
        """(exit code, lines) of the plan's command on the stand-in web host, its standard input the remote half;
        PYTHON: the words that run it in python3's place."""
        env = {
            "PATH": f"{self.bin}:/usr/bin:/bin",
            "HOME": str(self.home),
            "USER": "op",
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        argv = self.plan.remote(install=install)
        with open(w.REMOTE, "rb") as script:
            p = subprocess.run(
                [*(python or argv[:2]), *argv[2:]],
                stdin=script,
                capture_output=True,
                text=True,
                env=env,
                cwd=self.home,
                timeout=120,
            )
        return p.returncode, p.stdout.splitlines()

    def test_the_look_changes_nothing(self):
        rc, out = self.remote(install=False)
        self.assertEqual(rc, 0, out)
        self.assertTrue(
            out[0].startswith("== ") and out[0].endswith(f": port {self.port}"), out
        )
        self.assertEqual(
            out[1:-1],
            [
                f"(nothing listens on {self.port})",
                "== rocke-monitor.service: inactive; unit file: none",
                "== linger (user services keep running after logout): yes",
                "== crontab: 30 4 * * * cd rocke-monitor/site && python3 -m http.server 8080",
            ],
        )
        self.assertRegex(
            out[-1],
            rf"^== python3: Python 3\.\d+\.\d+; site: {re.escape(str(self.home))}/rocke-monitor/site$",
        )
        self.assertEqual(
            sorted(x.relative_to(self.home).as_posix() for x in self.home.rglob("*")),
            ["rocke-monitor", "rocke-monitor/site"],
        )
        # an older package's unit file for the page (webhost_remote.OLDER_UNITS): the look shows it too
        older = remote_half().OLDER_UNITS[0]
        file = self.home / ".config" / "systemd" / "user" / older
        file.parent.mkdir(parents=True)
        file.write_text("[Unit]\n", encoding="utf-8")
        rc, out = self.remote(install=False)
        self.assertEqual(rc, 0, out)
        self.assertIn(
            f"== {older}, an older package's: inactive; unit file: {file}", out
        )
        self.assertEqual(file.read_text(encoding="utf-8"), "[Unit]\n")

    def test_the_install_finds_the_plan_s_copies_where_the_plan_puts_them(self):
        """Without them it refuses; with them it goes on, to its check of the port, which stops it here: a listener
        this login cannot see or stop."""
        rc, out = self.remote()
        self.assertEqual(
            (rc, out[-2:]),
            (
                w.REMOTE_REFUSED,
                [
                    "== deploy",
                    "the new unit or webhost_server.py did not arrive; nothing changed",
                ],
            ),
        )
        self.copies()
        self.tools(ss=HIDDEN)
        rc, out = self.remote()
        self.assertEqual(
            (rc, out[-2:]),
            (
                w.REMOTE_REFUSED,
                [
                    "== deploy",
                    f"port {self.port} is held by a process this login cannot see or stop; nothing changed",
                ],
            ),
        )
        self.assertTrue(
            out[-3].endswith(f"; site: {self.home}/rocke-monitor/site"), out[-3]
        )
        self.assertFalse(self.unit.exists())

    def test_a_python_older_than_3_6_is_refused_at_the_look(self):
        """In either mode, before anything else: the install's look refuses it, before the install copies anything.
        Its one line says so, and nothing changed."""
        for install in (False, True):
            with self.subTest(install=install):
                rc, out = self.remote(install, python=(sys.executable, "-c", AS_3_5))
                self.assertEqual((rc, out), (w.REMOTE_REFUSED, [OLD_PYTHON]))
                self.assertEqual(
                    sorted(
                        x.relative_to(self.home).as_posix()
                        for x in self.home.rglob("*")
                    ),
                    ["rocke-monitor", "rocke-monitor/site"],
                )

    def test_its_question_to_the_port_goes_past_any_proxy_and_any_answer_counts(
        self,
    ):
        """answers(PORT), asked of 127.0.0.1 past the proxy the login's environment names (here one on a closed port,
        which answers nothing): any HTTP answer is the server's, a 404 too, as the site folder may hold no page yet.
        """
        remote = remote_half()
        proxy = f"http://127.0.0.1:{sandbox.closed_port(self)}"
        env = {k: v for k, v in os.environ.items() if k not in PROXIES}
        env.update(http_proxy=proxy, HTTP_PROXY=proxy)
        for status in (200, 404):
            with self.subTest(status=status):
                srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Answers)
                srv.status = status
                served(self, srv)
                with mock.patch.dict(os.environ, env, clear=True):
                    said = remote.answers(srv.server_address[1])
                self.assertEqual(
                    said,
                    f"answers: HTTP {status}, Server {Answers.server_version} {Answers.sys_version}",
                )

    def test_each_step_it_refuses_changes_nothing(self):
        """Another program on the port; this login's own http.server there (README.md's way by hand passes REPLACE
        yes, the install never does); user services that stop at logout, where this login cannot turn lingering on:
        refused, the step named in the last line, and nothing changed: the copies wait, no unit is installed, and what
        holds the port runs on."""
        self.copies()
        other = subprocess.Popen(["sleep", "60"])  # not an http.server
        own = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)", "http.server"]
        )
        for p in (other, own):
            self.addCleanup(p.wait)
            self.addCleanup(p.kill)
        for tools, last in (
            (
                {"ss": self.listener(other.pid)},
                f"port {self.port} is held by PID {other.pid}, which is not an http.server; nothing changed",
            ),
            (
                {"ss": self.listener(own.pid)},
                f"port {self.port} is served by this login's own http.server (PID {own.pid}, above): stop it and "
                "install again, and rocke-monitor.service takes its place, on the same folder and port; "
                "nothing changed",
            ),
            (
                {"loginctl": LINGER_OFF},
                "user services stop at logout here (Linger=no), and this login cannot turn lingering on, so "
                "rocke-monitor.service would stop when the ssh session ends: the host's admin runs "
                "`loginctl enable-linger op`; nothing changed",
            ),
        ):
            with self.subTest(last=last):
                self.tools(**tools)
                rc, out = self.remote()
                self.assertEqual((rc, out[-1]), (w.REMOTE_REFUSED, last), out)
                self.assertFalse(self.unit.exists())
                for copy in self.plan.copies:
                    self.assertTrue((self.home / copy.dest).is_file(), copy.dest)
        self.assertEqual((other.poll(), own.poll()), (None, None))


# The stand-in web host's own systemctl, loginctl and crontab, which the image test's web-host stage runs too.
WEB_HOST_BIN = ROOT / "tests" / "fakes" / "web_host" / "bin"
# ssh to the stand-in web host (TestOnAStandInWebHost): each call a JSON line of FAKE_LOG; the web host's own key
# checked as ssh checks it, by the known_hosts file its options name alone, else ssh's refusal (exit 255); then the
# command as the login's shell runs it there: in the web host's home FAKE_RHOME, on its PATH FAKE_RPATH, with ssh's
# standard input, and FAKE_BIND for the web host's systemctl (WEB_HOST_BIN).
FAKE_SSH = r"""#!/usr/bin/env python3
import json, os, subprocess, sys

args = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps(["ssh", *args]) + "\n")
options, i = {}, 0
while args[i].startswith("-"):
    if args[i] == "-o":
        name, _, value = args[i + 1].partition("=")
        options[name] = value
    i += 2
login, _, host = args[i].rpartition("@")
try:
    with open(options.get("UserKnownHostsFile", "")) as f:
        hosts = [x.split()[0] for x in f if x.strip()]
except OSError:
    hosts = []
if options.get("StrictHostKeyChecking") != "yes" or host not in hosts:
    sys.stderr.write("Host key verification failed.\n")
    sys.exit(255)
home = os.environ["FAKE_RHOME"]
env = {"HOME": home, "USER": login, "PATH": os.environ["FAKE_RPATH"], "LANG": "C.UTF-8"}
env["FAKE_BIND"] = os.environ.get("FAKE_BIND", "")
sys.exit(subprocess.run(["bash", "-c", " ".join(args[i + 1:])], env=env, cwd=home).returncode)
"""
# rsync to the stand-in web host: each call a JSON line of FAKE_LOG; its source copied into the folder its destination
# names in FAKE_RHOME, written whole, mode 644 (--chmod=F644).
FAKE_RSYNC = r"""#!/usr/bin/env python3
import json, os, shutil, sys

args = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as f:
    f.write(json.dumps(["rsync", *args]) + "\n")
src, dest = args[-2], args[-1].partition(":")[2]
folder = os.path.join(os.environ["FAKE_RHOME"], dest)
os.makedirs(folder, exist_ok=True)
to = os.path.join(folder, os.path.basename(src))
shutil.copyfile(src, to + ".part")
os.chmod(to + ".part", 0o644)
os.replace(to + ".part", to)
"""
# ss on the stand-in web host: the listener of each user service while it runs (the pids its systemctl keeps), else
# none.
FAKE_SS = r"""#!/usr/bin/env python3
import os, sys

port = sys.argv[-1].rpartition(":")[2]
home = os.environ["HOME"]
for name in sorted(os.listdir(home)):
    if not name.endswith(".service.pid"):
        continue
    try:
        with open(os.path.join(home, name)) as f:
            pid = int(f.read())
        with open("/proc/%d/stat" % pid) as f:
            ended = f.read().rsplit(")", 1)[1].split()[0] == "Z"
    except (OSError, ValueError, IndexError):
        ended = True
    if not ended:
        print('LISTEN 0 5 127.0.0.1:%s 0.0.0.0:* users:(("python3",pid=%d,fd=3))' % (port, pid))
"""


class TestOnAStandInWebHost(InstallBase):
    """The install end to end, its remote half and the request service run here: stand-ins for ssh and rsync reach a
    stand-in web host's home, whose systemctl, ss, loginctl and crontab are stand-ins too, and the service, started as
    the unit says, answers on 127.0.0.1 at the page's address. The doctor's row asks it there.
    """

    def setUp(self):
        super().setUp()
        self.rhome = self.d / "rhome"
        self.rhome.mkdir()
        here, self.there = self.d / "bin", self.d / "rbin"
        for folder, tools in (
            (here, {"ssh": FAKE_SSH, "rsync": FAKE_RSYNC}),
            (self.there, {"ss": FAKE_SS}),
        ):
            folder.mkdir()
            for name, text in tools.items():
                (folder / name).write_text(text, encoding="utf-8")
                (folder / name).chmod(0o755)
            # python3 here and on the web host, which the stand-ins and the service run on: this one
            (folder / "python3").symlink_to(sys.executable)
        self.log = self.d / "calls.log"
        self.port = sandbox.free_port()
        self.url = f"http://127.0.0.1:{self.port}/"
        p = mock.patch.dict(
            os.environ,
            {
                "PATH": f"{here}:/usr/bin:/bin",
                "FAKE_LOG": str(self.log),
                "FAKE_RHOME": str(self.rhome),
                "FAKE_RPATH": f"{self.there}:{WEB_HOST_BIN}:/usr/bin:/bin",
                "FAKE_BIND": "127.0.0.1",
            },
        )
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self.systemctl, "stop")

    def systemctl(self, *args: str, unit: str = w.UNIT) -> str:
        """What the stand-in web host's `systemctl --user ARGS UNIT` prints, as its login runs it."""
        return subprocess.run(
            [str(WEB_HOST_BIN / "systemctl"), "--user", *args, unit],
            env={
                "HOME": str(self.rhome),
                "PATH": f"{self.there}:{WEB_HOST_BIN}:/usr/bin:/bin",
                "LANG": "C.UTF-8",
                "FAKE_BIND": "127.0.0.1",
            },
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()

    def calls(self) -> list[list[str]]:
        return [json.loads(x) for x in self.log.read_text().splitlines()]

    def older_service(self, c: dict) -> str:
        """An older package's user service for the page (webhost_remote.OLDER_UNITS), as its install leaves it on the
        stand-in web host: its unit file, with the server and folders of the plan of C, the configuration, its server
        running on the page's port. Its name, once it answers there."""
        older = remote_half().OLDER_UNITS[0]
        p = w.plan(c)
        for copy in p.copies:
            if not copy.dest.endswith(".new"):
                (self.rhome / copy.dest).parent.mkdir(parents=True, exist_ok=True)
                (self.rhome / copy.dest).write_text(copy.text, encoding="utf-8")
        (self.rhome / "rocke-monitor" / "site").mkdir(parents=True, exist_ok=True)
        units = self.rhome / ".config" / "systemd" / "user"
        units.mkdir(parents=True)
        (units / older).write_text(p.unit, encoding="utf-8")
        self.addCleanup(self.systemctl, "stop", unit=older)
        self.systemctl("restart", unit=older)
        for _ in range(100):
            try:
                if self.get()[0] == 200:
                    return older
            except OSError:
                pass
            time.sleep(0.1)
        self.fail(f"{older} does not answer at {self.url}")

    def get(self) -> tuple[int, str, str]:
        """(status, Server header, body) of a GET of the page's address, past any proxy."""
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        try:
            with opener.open(self.url, timeout=10) as r:
                return r.status, r.headers.get("Server") or "", r.read().decode()
        except urllib.error.HTTPError as exc:
            with exc:
                return exc.code, exc.headers.get("Server") or "", ""

    def test_installed_it_answers_and_the_doctor_s_row_fails_once_it_stops(self):
        c = self.config(url=self.url)
        rc, out = self.install(c, subprocess.run)
        self.assertEqual(rc, 0, out)
        self.assertTrue(
            one_line(out).endswith(
                f"INSTALLED: the request service on web.example.com, op's user service "
                f"rocke-monitor.service, answers on port {self.port}"
            ),
            out,
        )
        self.assertEqual([x[0] for x in self.calls()], ["ssh", "rsync", "rsync", "ssh"])
        unit = self.rhome / ".config" / "systemd" / "user" / w.UNIT
        self.assertEqual(unit.read_text(encoding="utf-8"), w.plan(c).unit)
        server = self.rhome / "rocke-monitor" / w.SERVER.name
        self.assertEqual(
            server.read_text(encoding="utf-8"), w.SERVER.read_text(encoding="utf-8")
        )
        self.assertEqual(stat.S_IMODE(server.stat().st_mode), 0o644)
        self.assertFalse(
            (self.rhome / "rocke-monitor" / f"{w.UNIT}.new").exists()
        )  # in place
        # The site folder is made, as the page's first push needs it; the service answers there as itself.
        self.assertTrue((self.rhome / "rocke-monitor" / "site").is_dir())
        status, name, _ = self.get()
        self.assertEqual((status, name.split()[:1]), (200, ["rkmon-webhost"]))
        (self.rhome / "rocke-monitor" / "site" / "index.html").write_text(
            "<p>the page</p>"
        )
        self.assertEqual(self.get()[::2], (200, "<p>the page</p>"))
        (row,) = doctor.request_service_checks(c, IMAGE_ENV)
        self.assertEqual(
            (row.status, row.found),
            ("PASS", f"the request service on the web host answers at {self.url}"),
        )
        # Installed again, the service restarts in its own place.
        before = self.systemctl("show", "-p", "MainPID", "--value")
        rc, out = self.install(c, subprocess.run)
        self.assertEqual(rc, 0, out)
        self.assertNotIn(
            self.systemctl("show", "-p", "MainPID", "--value"), ("0", before)
        )
        self.assertEqual(self.get()[::2], (200, "<p>the page</p>"))
        # Once it stops, the row fails: its fix installs it, then runs the doctor again.
        self.systemctl("stop")
        (row,) = doctor.request_service_checks(c, IMAGE_ENV)
        self.assertEqual(row.status, "FAIL")
        self.assertTrue(
            row.found.startswith(
                "the request service on the web host does not answer at the team dashboard's address, "
                f"{self.url} (URLError: "
            ),
            row.found,
        )
        self.assertEqual(str(row.fix), "rkmon web-host install, then rkmon doctor")

    def test_an_older_package_s_service_on_the_port_gives_it_up(self):
        """An older package's user service for the page (webhost_remote.OLDER_UNITS), whose server holds the page's
        port: the look shows it, and the install stops and disables it, with a line that says so, and installs this
        package's service in its place, which answers there."""
        c = self.config(url=self.url)
        older = self.older_service(c)
        rc, out = self.install(c, subprocess.run)
        self.assertEqual(rc, 0, out)
        lines = out.splitlines()
        look = [
            x
            for x in lines
            if x.startswith(f"== {older}, an older package's: active; unit file: ")
        ]
        self.assertEqual(len(look), 1, out)
        self.assertIn(
            f"{older}, an older package's, serves port {self.port}: {w.UNIT} takes its place",
            lines,
        )
        self.assertTrue(
            one_line(out).endswith(
                f"INSTALLED: the request service on web.example.com, op's user service {w.UNIT}, answers on "
                f"port {self.port}"
            ),
            out,
        )
        self.assertEqual(self.systemctl("is-active", unit=older), "inactive")
        self.assertEqual(self.systemctl("is-active"), "active")
        status, name, _ = self.get()
        self.assertEqual((status, name.split()[:1]), (200, ["rkmon-webhost"]))

    def test_an_older_package_s_service_runs_again_when_this_one_does_not_come_up(self):
        """The install over an older package's service on the port (webhost_remote.OLDER_UNITS), whose new unit runs
        a program that serves nothing: this package's service is stopped, and the older one runs and answers again,
        with a line that says so."""
        c = self.config(url=self.url)
        older = self.older_service(c)
        silent = re.sub(
            r"(?m)^ExecStart=.*$",
            f'ExecStart={sys.executable} -c "import time; time.sleep(60)"',
            w.plan(c).unit,
        )
        with mock.patch.object(w, "unit_text", return_value=silent):
            rc, out = self.install(c, subprocess.run)
        self.assertNotEqual(rc, 0, out)
        lines = out.splitlines()
        self.assertIn(f"{w.UNIT} did not come up:", lines)
        self.assertIn(f"started {older} again, and {w.UNIT} is stopped", lines)
        self.assertEqual(self.systemctl("is-active"), "inactive")
        self.assertEqual(self.systemctl("is-active", unit=older), "active")
        for _ in range(100):
            try:
                if self.get()[0] == 200:
                    break
            except OSError:
                pass
            time.sleep(0.1)
        status, name, _ = self.get()
        self.assertEqual((status, name.split()[:1]), (200, ["rkmon-webhost"]))


class Quiet(http.server.SimpleHTTPRequestHandler):
    """http.server's own handler, logging nothing: another server than the request service."""

    def log_message(self, *args):
        pass


class TestTheDoctorsRow(InstallBase):
    """Check 7: with "publish", live and the priority box on, the doctor asks the request service at the
    team dashboard's address, from where it runs (in the container, from the container); no row otherwise.
    """

    def test_no_row_while_the_web_host_needs_no_service_or_nothing_is_pushed(self):
        asked = []
        off = self.config()
        off["features"]["dashboards"]["team"]["enabled"] = False
        for c in (
            self.config(delivery="serve"),
            self.config(box=False),
            self.config(mode="prototype"),
            self.config(url=" "),
            off,
        ):
            self.assertEqual(
                doctor.request_service_checks(c, IMAGE_ENV, answering=asked.append),
                [],
            )
        self.assertEqual(asked, [])

    def test_its_row_passes_while_the_service_answers_else_fails_with_the_install(self):
        c = self.config()
        asked = []
        (row,) = doctor.request_service_checks(
            c, IMAGE_ENV, answering=lambda url: asked.append(url) or ""
        )
        self.assertEqual(asked, [URL])
        self.assertEqual(
            (row.name, row.status, row.found, row.fix),
            (
                "team dashboard",
                "PASS",
                f"the request service on the web host answers at {URL}",
                None,
            ),
        )
        why = "URLError: [Errno 111] Connection refused"
        for env, fix in (
            (IMAGE_ENV, "rkmon web-host install, then rkmon doctor"),
            (HOST_ENV, "rkmon web-host install, then rkmon doctor"),
        ):
            with self.subTest(env=env):
                (row,) = doctor.request_service_checks(
                    c, env, answering=lambda url: why
                )
                self.assertEqual(
                    (row.name, row.status, row.found, str(row.fix)),
                    (
                        "team dashboard",
                        "FAIL",
                        "the request service on the web host does not answer at the team dashboard's address, "
                        f"{URL} ({why})",
                        fix,
                    ),
                )

    def test_the_dashboards_checks_ask_it_the_admin_dashboard_on_or_off(self):
        for admin in (False, True):
            with self.subTest(admin=admin):
                c = self.config()
                c["features"]["dashboards"]["admin"] = {"enabled": admin}
                rows = doctor.web_checks(
                    c, HOST_ENV, answers=lambda h, p: False, answering=lambda url: ""
                )
                self.assertIn(
                    ("team dashboard", "PASS"), [(r.name, r.status) for r in rows]
                )

    def test_it_asks_for_the_service_by_its_own_name(self):
        """web_host_install.answering, at the root of the service's port, where it serves the site folder (as the
        unit's --directory): "" for an answer whose Server header names the service; else why not: another server's
        answer, none, or an address it cannot ask.
        """
        site = self.d / "site"
        site.mkdir()
        (site / "index.html").write_text("<p>the page</p>")
        mod = webserve._webhost()
        mod.Handler.log_message = lambda *args: None
        service = served(
            self,
            mod.make_server(str(site), str(self.d / "requests.jsonl"), 0, "127.0.0.1"),
        )
        self.assertEqual(w.answering(f"{service}/"), "")
        plain = served(
            self,
            http.server.ThreadingHTTPServer(
                ("127.0.0.1", 0), functools.partial(Quiet, directory=str(site))
            ),
        )
        said = w.answering(f"{plain}/")
        self.assertTrue(
            said.startswith("another server answers there: SimpleHTTP/"), said
        )
        said = w.answering(f"http://127.0.0.1:{sandbox.closed_port(self)}/")
        self.assertTrue(said.startswith("URLError: "), said)
        self.assertIn("Connection refused", said)
        said = w.answering("not an address")
        self.assertTrue(said.startswith("ValueError: unknown url type"), said)


class TestTheOneInstaller(unittest.TestCase):
    def test_the_package_installs_the_request_service_one_way(self):
        """`rkmon web-host install` replaces the tool that read config.json: that tool is gone, and the
        skill's row names the command, which the operator runs from the host and the guard refuses to the session.
        """
        self.assertFalse((ROOT / "tools" / "deploy_webhost.py").exists())
        skill = ROOT / "agentic" / "skills" / "rocke-monitor" / "commands.md"
        (row,) = [
            line
            for line in skill.read_text(encoding="utf-8").splitlines()
            if line.startswith("| `web-host install` |")
        ]
        self.assertIn("runs it from a shell on the host: `rkmon web-host install`", row)
        self.assertIn(f"the guard refuses `rk {writer.WEB_HOST_INSTALL}` to you", row)
        self.assertNotIn("deploy_webhost", skill.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
