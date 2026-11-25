# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The dashboard's "Request PR priority" box and the local service behind it: every request carries the access token
(HTTP Basic, the loopback included), and only the served page (its own origin, never `null`), with the X-Rkmon header
and a Host the service listens on, gets an answer; a preflight is answered only for it; a request does what
`rk prioritize` does, once per PR; other teams' requests wait for the operator's OK. The page comes in two variants
from the same code: neither carries a secret, and only the local one contacts 127.0.0.1.
"""

import base64
import fcntl
import http.client
import json
import pathlib
import re
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
import guard_labels  # noqa: E402

from rkmon_lib import (  # noqa: E402
    config,
    file_types,
    priority,
    queue,
    report,
    settings_file,
    webdash,
    webserve,
)
from rkmon_lib.store import Store, empty_state, now  # noqa: E402

PUB = "http://web-host.example:8080"
PORT = 18613
OWN = f"http://127.0.0.1:{PORT}"  # the served page's origin
_SECRETS = (
    tempfile.TemporaryDirectory()
)  # stands in for the secrets folder: the tests never touch a real secret
TOKEN = "invented-" + "access-token-0123456789"  # an invented value, built at run time
_TOKEN_FILE = pathlib.Path(_SECRETS.name) / "admin_access_token"
_TOKEN_FILE.write_text(TOKEN + "\n", encoding="utf-8")
_TOKEN_FILE.chmod(0o600)
CFG = {
    "repo": "ROCm/r",
    "operator": {"github_login": "me"},
    "features": {
        "merge_queue": {"mode": "live", "auto_merge": "enabled"},
        "dashboards": {"team": {"enabled": True, "mode": "live"}},
        "ticket_creation": {
            "enabled": True,
            "triggers": {"review_follow_up": True, "out_of_scope_issue": True},
        },
    },
    "admin_dashboard": {
        "port": PORT,
        "bind": "127.0.0.1",
        "public_port": 0,
        "url": "",
        "access_token_file": str(_TOKEN_FILE),
    },
    "publish": {
        "html": {
            "host": "h",
            "user": "u",
            "key": "k",
            "dest": "d",
            "url": PUB + "/rocke-monitor/",
        }
    },
}
ACCESS = webserve.access_info(CFG, PORT, {})  # what `rk serve` gives every request
# What a request with another Host is told with its 403, the service listening at PORT on the loopback.
HOST_REFUSAL = (
    f"this page answers only at 127.0.0.1:{PORT} or localhost:{PORT}; through an SSH tunnel, use the admin "
    f'dashboard\'s port, {PORT}, at both ends (README.md, "The admin dashboard", shows the command)'
)


def admin_cfg(cfg: dict = CFG, **section) -> dict:
    """`cfg` with some of its admin_dashboard keys changed."""
    return dict(cfg, admin_dashboard=dict(cfg["admin_dashboard"], **section))


DROP = object()


def tearDownModule():
    _SECRETS.cleanup()


def _changed(h: dict, change: dict) -> dict:
    for k, v in change.items():
        k = k.replace("_", "-")
        h.pop(k, None) if v is DROP else h.__setitem__(k, v)
    return h


def basic(token: str = TOKEN, user: str = "anyone") -> str:
    """The Authorization a browser sends once the user typed the access token: HTTP Basic, any user name."""
    return "Basic " + base64.b64encode(f"{user}:{token}".encode()).decode()


def hdrs(**change) -> dict:
    """A request's headers as the served page sends them; `change` sets (DROP removes) one, e.g. Origin, X_Rkmon."""
    return _changed(
        {
            "Origin": OWN,
            "Host": f"127.0.0.1:{PORT}",
            "X-Rkmon": "1",
            "Authorization": basic(),
            "Content-Type": "application/json",
        },
        change,
    )


def preflight_hdrs(**change) -> dict:
    return _changed(
        {
            "Origin": OWN,
            "Host": f"127.0.0.1:{PORT}",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-rkmon",
            "Access-Control-Request-Private-Network": "true",
        },
        change,
    )


def pr_rec(n: int, **kw) -> dict:
    """A tracked PR as the state keeps it (no review of ours yet)."""
    r = {
        "n": n,
        "head": f"{n:040d}",
        "size": 100,
        "status": "open",
        "draft": False,
        "author": "a",
        "title": f"t{n}",
        "labels": [],
        "base": "develop",
        "head_ref": f"b{n}",
        "mergeable": "MERGEABLE",
        "ci_rollup": "SUCCESS",
        "review_decision": "REVIEW_REQUIRED",
        "reviews": [],
        "ours": None,
    }
    r.update(kw)
    return r


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(pathlib.Path(self.tmp.name))
        st = empty_state()
        st["last_tick"] = "2026-11-03T17:20:00Z"
        st["develop"]["sha"] = "d" * 40
        st["prs"] = {
            "7": pr_rec(7),
            "8": pr_rec(8, status="merged"),
            "9": pr_rec(9, draft=True),
            "10": pr_rec(10, closes=[1111]),
            "11": pr_rec(11, status="out-of-scope"),
        }
        st["post_merge"] = {
            "at": now(),
            "open": [
                {
                    "n": 1111,
                    "title": "x",
                    "kind": "stability",
                    "label": guard_labels.STABILITY,
                }
            ],
        }
        queue.refresh_develop_fixes(st, [])
        self.store.save(st)

    def tearDown(self):
        self.tmp.cleanup()

    def ask(self, method: str, path: str, headers: dict, body: bytes = b"") -> tuple:
        return webserve.answer(
            CFG, self.store, method, path, headers, body, PORT, ACCESS
        )

    def post(self, pr, **change) -> tuple:
        return self.ask(
            "POST", "/priority", hdrs(**change), json.dumps({"pr": pr}).encode()
        )

    def journal(self) -> str:
        return (
            self.store.journal_path.read_text()
            if self.store.journal_path.exists()
            else ""
        )


class TestConfigAndSecret(unittest.TestCase):
    def test_the_origins_port_and_host(self):
        # The served page's own origins; never null (a file opened from disk, or a sandboxed frame on any site).
        self.assertEqual(webserve.origins(PORT), (OWN, f"http://localhost:{PORT}"))
        self.assertEqual(webserve.port(CFG), PORT)
        self.assertEqual(webserve.port({}), webserve.DEFAULT_PORT)
        self.assertEqual(webserve.HOST, "127.0.0.1")

    def test_the_access_token_is_read_never_made(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "admin_access_token"
            cfg = {"admin_dashboard": {"access_token_file": str(path)}}
            self.assertIsNone(webserve.access_token(cfg))
            self.assertFalse(path.exists())  # a reader makes none
            path.write_text(TOKEN + "\n", encoding="utf-8")
            path.chmod(0o644)
            self.assertEqual(webserve.access_token(cfg), TOKEN)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)  # tightened when read
        self.assertEqual(
            config.load_defaults()["admin_dashboard"],
            {
                "port": "",
                "bind": "",
                "public_port": settings_file.schema_at(
                    file_types.DEPLOY.load(), "features.dashboards.admin.port"
                )["default"],
                "url": "",
                "access_token_file": "",
            },
        )  # the port and the address are derived: the container's (rkmon_lib/derived.py); the published port is the
        # deploy settings' (features.dashboards.admin.port), at its default
        # The access token file is the place the load derives, admin_access_token in the deployment's secrets folder
        # (<WORKING_DIR>/.config), which admin reads as the configuration holds it: a blank one is none, as admin
        # derives no place itself; nothing reads admin_dashboard.token_file.
        raw = {
            "admin_dashboard": {"port": 1, "token_file": "t", "access_token_file": ""},
            "paths": {"working_dir": "/w"},
        }
        self.assertEqual(webserve.admin(raw)["access_token_file"], "")
        a = webserve.admin(config.normalize(raw))
        self.assertEqual(a["access_token_file"], "/w/.config/admin_access_token")
        self.assertNotIn("token_file", a)
        # The package's keys alone: a configuration with no admin_dashboard section runs on the defaults, whatever
        # older names (web_port, web_token_file) a configuration file holds; no key names the host address.
        self.assertEqual(
            webserve.admin({"admin_dashboard": {"port": 1}, "web_port": 2})["port"], 1
        )
        self.assertEqual(
            webserve.admin({"web_port": 2, "web_token_file": "b"}),
            {
                "port": webserve.DEFAULT_PORT,
                "bind": "127.0.0.1",
                "public_port": 0,
                "url": "",
                "access_token_file": "",
            },
        )
        self.assertEqual(
            webserve.admin({"admin_dashboard": {"port": None}})["port"],
            webserve.DEFAULT_PORT,
        )

    def test_the_access_token_s_permissions_are_never_set_through_a_link(self):
        """A token file others may read is set to 600 as a private file is (private_files.tighten): a link is not
        followed, and the file it points to keeps its permissions. Nor is the token read through it, whatever the
        file it points to: a private file is never a link, and the refusal says so, with the fix.
        """
        with tempfile.TemporaryDirectory() as d:
            real = pathlib.Path(d).resolve() / "elsewhere"
            real.write_text(TOKEN + "\n", encoding="utf-8")
            real.chmod(0o644)
            link = real.with_name("admin_access_token")
            link.symlink_to(real)
            cfg = {"admin_dashboard": {"access_token_file": str(link)}}
            self.assertIsNone(webserve.access_token(cfg))
            self.assertEqual(real.stat().st_mode & 0o777, 0o644)
            self.assertEqual(
                webserve.access_refusal(cfg),
                "the admin dashboard needs its access token on every request, from this machine too, and reads it "
                f"only from a private file: {link} is a link, not a file (fix: rm {link}, "
                f"then cp {real} {link})",
            )
            # The file it points to is a private one now: the link is still none.
            real.chmod(0o600)
            self.assertIsNone(webserve.access_token(cfg))

    def test_the_access_token_s_file_keeps_every_private_file_rule(self):
        """Every rule of a private file holds for the token's file (private_files.problems), its place too: in a git
        work tree, the token is not read, and the refusal says why, with the fix."""
        with tempfile.TemporaryDirectory() as d:
            top = pathlib.Path(d).resolve()
            (top / ".git").mkdir()
            path = top / "admin_access_token"
            path.write_text(TOKEN + "\n", encoding="utf-8")
            path.chmod(0o600)
            cfg = {"admin_dashboard": {"access_token_file": str(path)}}
            self.assertIsNone(webserve.access_token(cfg))
            self.assertIn(
                f"only from a private file: {path} is inside the git work tree {top}: a commit there would publish "
                f"it (fix: mv {path} ~/",
                webserve.access_refusal(cfg),
            )
            (top / ".git").rmdir()
            self.assertEqual(webserve.access_token(cfg), TOKEN)
            self.assertIsNone(webserve.access_refusal(cfg))


class TestPreflight(Base):
    def test_answered_for_the_served_page(self):
        status, h, body = self.ask("OPTIONS", "/priority", preflight_hdrs())
        self.assertEqual((status, body), (204, None))
        self.assertEqual(h["Access-Control-Allow-Origin"], OWN)
        self.assertEqual(
            h["Access-Control-Allow-Private-Network"], "true"
        )  # Chrome's private network access
        self.assertEqual(h["Access-Control-Allow-Methods"], "GET, POST")
        self.assertEqual(h["Access-Control-Allow-Headers"], "Content-Type, X-Rkmon")
        self.assertEqual(h["Vary"], "Origin")
        self.assertEqual(
            self.journal(), ""
        )  # passing preflights stay in the service's own log

    def test_refused_for_anything_else(self):
        for change in (
            {"Origin": PUB},
            {"Origin": "http://evil.example"},
            {"Origin": "null"},
            {"Origin": DROP},
            {"Access-Control-Request-Method": "DELETE"},
            {"Access-Control-Request-Headers": "x-rkmon, authorization"},
            {"Access-Control-Request-Headers": "content-type,x-rkmon,x-rkmon-token"},
            {"Host": "evil.example:18613"},
            {"Host": "127.0.0.1:9999"},
        ):
            status, h, body = self.ask("OPTIONS", "/priority", preflight_hdrs(**change))
            self.assertEqual((status, h, body), (403, {}, None), change)
        self.assertIn(
            "refused preflight /priority from http://web-host.example:8080 (POST)",
            self.journal(),
        )


class TestRequests(Base):
    def test_ping_answers_the_served_page_only(self):
        status, h, body = self.ask("GET", "/ping?t=1", hdrs())
        self.assertEqual((status, body), (200, {"ok": True, "me": "me"}))
        self.assertEqual((h["Access-Control-Allow-Origin"], h["Vary"]), (OWN, "Origin"))
        self.assertEqual(
            self.journal(), ""
        )  # an open page pings every 2 minutes: not journaled
        self.assertEqual(
            self.ask("GET", "/ping", hdrs(Origin=PUB))[0::2],
            (403, {"ok": False, "message": "origin not allowed"}),
        )

    def test_other_origins_and_requests_without_the_header_are_refused_and_journaled(
        self,
    ):
        for change, why in (
            ({"Origin": PUB}, "origin not allowed"),
            ({"Origin": "http://evil.example"}, "origin not allowed"),
            # A file opened from disk, or a sandboxed frame on any site, with the browser's credentials.
            ({"Origin": "null"}, "origin not allowed"),
            ({"Origin": DROP}, "origin not allowed"),
            ({"X-Rkmon": DROP}, "no X-Rkmon header"),
            ({"X-Rkmon": "0"}, "no X-Rkmon header"),
            ({"Host": "rebound.example:18613"}, "host not allowed"),
        ):
            # The answer says what the journal says, but for another Host: the names the page answers to, and the fix.
            said = HOST_REFUSAL if "Host" in change else why
            for method, path in (("GET", "/ping"), ("POST", "/priority")):
                status, h, body = self.ask(
                    method,
                    path,
                    hdrs(**change),
                    b'{"pr": 7}' if method == "POST" else b"",
                )
                self.assertEqual(
                    (status, body), (403, {"ok": False, "message": said}), change
                )
                self.assertNotIn(
                    "Access-Control-Allow-Origin", h
                )  # a foreign page cannot even read the refusal
        self.assertNotIn("7", self.store.load().get("user_priority") or {})
        j = self.journal()
        self.assertIn(
            "refused POST /priority #7 from http://evil.example: origin not allowed", j
        )
        self.assertIn("refused POST /priority #7 from null: origin not allowed", j)
        self.assertIn("refused GET /ping from none: origin not allowed", j)
        self.assertIn(f"refused POST /priority #7 from {OWN}: no X-Rkmon header", j)
        self.assertIn(f"refused GET /ping from {OWN}: host not allowed", j)
        self.assertEqual(j.count(" web "), 14)  # one line per refused request
        # An untrusted origin is cleaned before it reaches the journal.
        odd = "http" + "://x"  # an untrusted origin, built at run time
        self.ask("GET", "/ping", hdrs(Origin=odd + "` **y** <b>"))
        self.assertIn(
            f"refused GET /ping from {odd}????y????b?: origin not allowed",
            self.journal(),
        )

    def test_a_host_at_another_port_is_told_the_port_and_the_tunnel_that_reaches_the_page(
        self,
    ):
        """A browser that names another port, as one does through an SSH tunnel whose two ends have different ports, is
        refused before the token is asked for, and told the names the page answers to and the port the tunnel uses at
        both ends, never its own Host; the journal says "host not allowed" as for any other Host.
        """
        other = PORT + 10
        for host in (f"127.0.0.1:{other}", f"localhost:{other}"):
            status, h, body = self.ask(
                "GET", "/", {"Host": host, "Sec-Fetch-Site": "none"}
            )
            self.assertEqual(
                (status, h, body), (403, {}, {"ok": False, "message": HOST_REFUSAL})
            )
            self.assertNotIn(str(other), body["message"])
        self.assertIn(f"port, {PORT}, at both ends", HOST_REFUSAL)
        self.assertEqual(
            self.journal().count("refused GET / from none: host not allowed"), 2
        )
        # Beyond the loopback, the names the access gives too (admin_dashboard.url's, and its address's).
        self.assertEqual(
            webserve.host_refusal(PORT, {"hosts": ("monitor.example.com",)}),
            HOST_REFUSAL.replace(
                f"127.0.0.1:{PORT} or localhost:{PORT}",
                f"127.0.0.1:{PORT}, localhost:{PORT} or monitor.example.com",
            ),
        )

    def test_every_request_needs_the_access_token_from_this_machine_too(self):
        """Every request but a preflight and the icon carries the access token, on the loopback too: HTTP Basic, any
        user name, the token as the password. Without it: the Basic challenge, and nothing else (no CORS header, no
        cookie)."""
        routes = (
            ("GET", "/"),
            ("GET", "/ping"),
            ("GET", "/ticket-draft?id=0000aaaa"),
            *(("POST", route) for route in webserve.ROUTES),
        )
        for auth in (
            DROP,
            "",
            basic(""),
            basic(TOKEN[:-1]),
            basic(TOKEN + "x"),
            basic("é" + TOKEN[1:]),
            "Basic !!!",
            f"Basic {TOKEN}",
            f"Bearer {TOKEN}",
            f"Token {TOKEN}",
        ):
            for method, route in routes:
                status, h, out = self.ask(
                    method,
                    route,
                    hdrs(Authorization=auth),
                    b'{"pr": 7}' if method == "POST" else b"",
                )
                self.assertEqual(
                    (status, h, out),
                    (
                        401,
                        {"WWW-Authenticate": webserve.REALM},
                        {"ok": False, "message": webserve.NEED_ACCESS},
                    ),
                    (auth, route),
                )
        self.assertNotIn("7", self.store.load().get("user_priority") or {})
        self.assertIn(
            f"refused POST /priority #7 from {OWN}: {webserve.NEED_ACCESS}",
            self.journal(),
        )
        self.assertNotIn(TOKEN[:12], self.journal())  # the token is never journaled
        for user in ("", "someone"):  # any user name
            self.assertEqual(
                self.ask("GET", "/ping", hdrs(Authorization=basic(user=user)))[0], 200
            )
        with mock.patch.object(
            webserve.hmac, "compare_digest", wraps=webserve.hmac.compare_digest
        ) as compare:
            self.assertEqual(self.post(7)[0], 200)
        compare.assert_called_once()  # compared in constant time
        # No access token on this machine: every request is refused, the right one too.
        access = webserve.access_info(
            admin_cfg(access_token_file=f"{self.tmp.name}/none"), PORT, {}
        )
        self.assertEqual(access["token"], "")
        for given in (access, None):
            self.assertEqual(
                webserve.answer(
                    CFG,
                    self.store,
                    "POST",
                    "/priority",
                    hdrs(),
                    b'{"pr": 10}',
                    PORT,
                    given,
                )[0:3:2],
                (401, {"ok": False, "message": webserve.NEED_ACCESS}),
            )
        self.assertNotIn("10", self.store.load()["user_priority"])

    def test_a_bad_request_is_refused_with_its_reason(self):
        for body in (
            b"",
            b"{",
            b'{"pr": "abc"}',
            b'{"pr": true}',
            b'{"pr": -3}',
            b'{"pr": 0}',
            b'{"pr": 1e3}',
            b'{"pr": 10000000}',
            b"[7]",
            b"\xff",
        ):
            status, h, out = self.ask("POST", "/priority", hdrs(), body)
            self.assertEqual((status, out["ok"]), (400, False), body)
            self.assertEqual(
                h["Access-Control-Allow-Origin"], OWN
            )  # the page shows why
        self.assertEqual(
            self.ask(
                "POST", "/priority", hdrs(Content_Type="text/plain"), b'{"pr": 7}'
            )[0],
            415,
        )
        self.assertEqual(self.ask("POST", "/elsewhere", hdrs(), b'{"pr": 7}')[0], 404)
        self.assertEqual(self.ask("GET", "/priority", hdrs())[0], 404)
        self.assertEqual(self.store.load().get("user_priority"), None)

    def test_only_a_tracked_open_pr_is_prioritized(self):
        for n, why in (
            (
                99,
                "#99 is not an open rocKE PR the monitor tracks (as of the last tick, 2026-11-03 17:20Z): "
                "nothing changed.",
            ),
            (8, "#8 is merged, not open: nothing changed."),
            (11, "#11 is out of scope, not open: nothing changed."),
        ):
            self.assertEqual(self.post(n)[0::2], (409, {"ok": False, "message": why}))
            self.assertRegex(
                self.journal(),
                rf"web +POST /priority #{n} from {re.escape(OWN)}: refused: {re.escape(why)}\n",
            )
        self.assertEqual(self.store.load().get("user_priority"), None)

    def test_applied_once_as_rk_prioritize_does(self):
        status, h, out = self.post(7)
        self.assertEqual(
            (status, out["ok"], h["Access-Control-Allow-Origin"]), (200, True, OWN)
        )
        self.assertEqual(
            out["message"],
            "#7 added to the priority queue: reviewed first and merged first. The next tick "
            "(within 10 min) makes it a priority in the merge plan.",
        )
        st = self.store.load()
        at = st["user_priority"]["7"]["at"]
        self.assertEqual(
            st["user_priority"], {"7": {"why": priority.DASHBOARD_WHY, "at": at}}
        )
        self.assertEqual(
            st["develop_fixes"]["7"], ["your priority: requested from the dashboard"]
        )  # review P0, tier 0
        j = self.journal()
        self.assertRegex(
            j,
            r"mode +prioritize #7 \(requested from the dashboard\): your priorities \[7\]\n",
        )
        self.assertRegex(
            j,
            rf"web +POST /priority #7 from {re.escape(OWN)}: done: #7 added to the priority queue",
        )
        # Entered again: refused, and nothing changes.
        status, h, out = self.post("#7")
        self.assertEqual(
            (status, out["ok"], h["Access-Control-Allow-Origin"]), (409, False, OWN)
        )
        self.assertEqual(
            out["message"],
            f"#7 is already in the priority queue since {at[:10]} {at[11:16]}Z (requested "
            "from the dashboard): nothing changed.",
        )
        self.assertEqual(
            self.store.load()["user_priority"],
            {"7": {"why": priority.DASHBOARD_WHY, "at": at}},
        )
        self.assertEqual(self.journal().count("prioritize #7"), 1)
        # A priority set by hand counts too.
        st = self.store.load()
        priority.set_priority(st, self.store, 10, "blocking work")
        self.store.save(st)
        self.assertIn("(blocking work): nothing changed.", self.post(10)[2]["message"])

    def test_a_develop_fix_and_a_draft_can_be_prioritized(self):
        """A develop fix is a top priority already; `rk prioritize` allows it, so the box does too."""
        status, _, out = self.post(10)
        self.assertEqual(status, 200)
        self.assertTrue(
            out["message"].endswith(
                " It was a top priority already (silent-break #1111); now it is "
                "yours too."
            ),
            out["message"],
        )
        self.assertEqual(
            self.store.load()["develop_fixes"]["10"],
            ["silent-break #1111", "your priority: requested from the dashboard"],
        )
        status, _, out = self.post(9)
        self.assertEqual(status, 200)
        self.assertTrue(
            out["message"].endswith(
                " It is a draft: it takes its place once it is ready for review."
            )
        )

    def test_a_request_waits_for_a_running_tick_and_loses_nothing(self):
        done = []
        with self.store.locked():  # a tick holds the lock
            t = threading.Thread(target=lambda: done.append(self.post(7)))
            t.start()
            t.join(0.5)
            self.assertTrue(t.is_alive())
            st = self.store.load()
            st["enabled"] = True  # the tick's own update, saved before it lets go
            self.store.save(st)
        t.join(10)
        self.assertEqual(done[0][0], 200)
        st = self.store.load()
        self.assertTrue(st["enabled"])
        self.assertIn("7", st["user_priority"])


class TestCli(Base):
    def run_rk(self, *argvs, cfg: dict | None = None, rc: int = 0) -> list[str]:
        import rkmon

        out = []
        with mock.patch.object(
            rkmon, "load_config", lambda: dict(cfg or CFG)
        ), mock.patch.object(rkmon, "Store", lambda: self.store), mock.patch(
            "builtins.print", lambda *a, **kw: out.append(" ".join(map(str, a)))
        ):
            for argv in argvs:
                self.assertEqual(rkmon.main(list(argv)), rc)
        return out

    def test_rk_prioritize_writes_what_it_wrote_before(self):
        out = self.run_rk(
            ["prioritize", "7", "blocking", "work"],
            ["unprioritize", "7"],
            ["prioritize", "10"],
        )
        j = self.journal()
        self.assertRegex(
            j, r"mode +prioritize #7 \(blocking work\): your priorities \[7\]\n"
        )
        self.assertRegex(j, r"mode +unprioritize #7: your priorities \[\]\n")
        self.assertRegex(j, r"mode +prioritize #10: your priorities \[10\]\n")
        self.assertTrue(out[-1].endswith("prioritized=[10]"), out)
        st = self.store.load()
        self.assertEqual(
            st["user_priority"],
            {"10": {"why": "", "at": st["user_priority"]["10"]["at"]}},
        )
        self.assertEqual(
            st["develop_fixes"],
            {"10": ["silent-break #1111", "your priority: floated to the top"]},
        )

    def test_rk_serve_runs_one_copy_per_state_directory(self):
        with open(self.store.dir / "web.lock", "a+") as f:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)  # a running copy
            (self.store.dir / "web.pid").write_text("4242\n")
            with mock.patch.object(webserve, "Server") as server:
                out = self.run_rk(["serve", "--port", "0"])
            server.assert_not_called()
        self.assertEqual(out, ["RKWEB already running (pid 4242)"])

    def test_rk_serve_needs_the_access_token_on_the_loopback_too(self):
        cfg = admin_cfg(access_token_file="")  # and no paths: no secrets folder either
        with mock.patch.object(webserve, "Server") as server:
            out = self.run_rk(["serve", "--port", "0"], cfg=cfg, rc=4)
        server.assert_not_called()
        self.assertEqual(
            out,
            [
                "RKWEB not started: the admin dashboard needs its access token on every request, from this machine "
                "too, and none is set (the secret admin_access_token): run rkmon secret admin_access_token --generate"
            ],
        )


class TestServer(Base):
    def test_it_listens_on_loopback_and_answers_over_http(self):
        srv = webserve.Server(CFG, self.store, 0)
        host, port = srv.server_address
        self.assertEqual(host, "127.0.0.1")
        srv.access = webserve.access_info(CFG, srv.ports, {})
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        own = f"http://127.0.0.1:{port}"
        sent = {"Host": f"127.0.0.1:{port}"}

        def call(method, path, headers, body=None):
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            try:
                c.request(method, path, body=body, headers=dict(headers, **sent))
                r = c.getresponse()
                data = r.read()
            finally:
                c.close()
            return (
                r.status,
                {k.lower(): v for k, v in r.getheaders()},
                json.loads(data) if data[:1] == b"{" else None,
            )

        try:
            with mock.patch.object(sys, "stderr"):  # the service's request log
                status, h, _ = call("OPTIONS", "/priority", preflight_hdrs(Origin=own))
                self.assertEqual(
                    (
                        status,
                        h["access-control-allow-origin"],
                        h["access-control-allow-private-network"],
                    ),
                    (204, own, "true"),
                )
                status, h, out = call(
                    "POST", "/priority", hdrs(Origin=own), json.dumps({"pr": 7})
                )
                self.assertEqual(
                    (status, out["ok"], h["access-control-allow-origin"]),
                    (200, True, own),
                )
                self.assertEqual(h["content-type"], "application/json; charset=utf-8")
                for change, refused in (
                    ({"Origin": PUB}, 403),
                    ({"Origin": "http://evil.example"}, 403),
                    ({"Origin": "null"}, 403),
                    ({"Origin": own, "Authorization": DROP}, 401),
                ):
                    status, h, out = call(
                        "POST", "/priority", hdrs(**change), json.dumps({"pr": 10})
                    )
                    self.assertEqual((status, out["ok"]), (refused, False), change)
                    self.assertNotIn("access-control-allow-origin", h)
                status, _, _ = call("PUT", "/priority", hdrs(Origin=own))
                self.assertEqual(status, 501)
                # An oversized body is refused before it is read.
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                c.putrequest("POST", "/priority", skip_host=True)
                for k, v in dict(hdrs(), **sent, **{"Content-Length": "5000"}).items():
                    c.putheader(k, v)
                c.endheaders()
                r = c.getresponse()
                r.read()
                c.close()
                self.assertEqual(r.status, 413)
        finally:
            srv.shutdown()
            srv.server_close()
            t.join(5)
        self.assertEqual(sorted(self.store.load()["user_priority"]), ["7"])


class TestContainer(Base):
    """In a container the service binds 0.0.0.0, so the published port reaches it; the compose file publishes it on the
    host's 127.0.0.1 only, at a port of its own (admin_dashboard.public_port), which the browser's Host and Origin
    name."""

    PUBLIC = 18614

    def test_where_it_may_listen(self):
        # Any IPv4 address of this machine; beyond the loopback the access
        # token rule decides (TestBeyondThisMachine in test_web_publishing.py).
        for bind in ("127.0.0.1", "0.0.0.0", "192.0.2.7"):
            self.assertIsNone(webserve.bind_refusal(bind), bind)
        self.assertIn(
            "an IPv4 address of this machine", webserve.bind_refusal("localhost")
        )
        cfg = admin_cfg(bind="0.0.0.0", public_port=self.PUBLIC)
        self.assertEqual(
            {k: webserve.admin(cfg)[k] for k in ("port", "bind", "public_port")},
            {"port": PORT, "bind": "0.0.0.0", "public_port": self.PUBLIC},
        )
        self.assertEqual(
            (webserve.admin(CFG)["bind"], webserve.admin(CFG)["public_port"]),
            ("127.0.0.1", 0),
        )

    def test_in_the_image_every_address_and_the_published_port(self):
        image = {"RKMON_CONTAINER": "1", "RKMON_PORT": "18999"}
        self.assertEqual(webserve.where(CFG, environ=image), ("0.0.0.0", 18999))
        self.assertEqual(
            webserve.listen_port(admin_cfg(port=1234), environ=image),
            webserve.INNER_PORT,
        )
        self.assertEqual(webserve.listen_port(admin_cfg(port=1234), environ={}), 1234)
        self.assertEqual(
            webserve.where(CFG, "127.0.0.1", 0, environ=image), ("127.0.0.1", 0)
        )  # arguments win
        self.assertEqual(
            webserve.where(CFG, environ={"RKMON_PORT": "18999"}), ("127.0.0.1", 0)
        )  # a host install

    def test_serve_refuses_to_start_without_the_access_token_wherever_it_listens(self):
        for bind in ("127.0.0.1", "0.0.0.0"):
            cfg = admin_cfg(
                port=0, bind=bind, access_token_file=f"{self.tmp.name}/none"
            )
            out = []
            with mock.patch.dict("os.environ", {"RKMON_CONTAINER": ""}), mock.patch(
                "builtins.print",
                side_effect=lambda *a, **k: out.append(" ".join(map(str, a))),
            ), mock.patch.object(webserve, "Server") as server:
                self.assertEqual(webserve.serve(cfg, self.store), 4)
            server.assert_not_called()
            self.assertEqual(
                out,
                [
                    "RKWEB not started: the admin dashboard needs its access token on every request, from this "
                    f"machine too, and none is set ({self.tmp.name}/none): run rkmon secret admin_access_token "
                    "--generate"
                ],
                bind,
            )

    def test_the_published_port_is_the_one_the_browser_names(self):
        ports = (PORT, self.PUBLIC)
        own = f"http://127.0.0.1:{self.PUBLIC}"
        for host in (
            f"127.0.0.1:{self.PUBLIC}",
            f"localhost:{self.PUBLIC}",
            f"127.0.0.1:{PORT}",
        ):
            status, _h, out = webserve.answer(
                CFG,
                self.store,
                "POST",
                "/priority",
                hdrs(Host=host, Origin=own),
                json.dumps({"pr": 7}).encode(),
                ports,
                ACCESS,
            )
            self.assertEqual((status, out["ok"]), (200, True), host)
            self.store.save(dict(self.store.load(), user_priority={}))
        status, _h, out = webserve.answer(
            CFG,
            self.store,
            "POST",
            "/priority",
            hdrs(Host="127.0.0.1:9999"),
            json.dumps({"pr": 7}).encode(),
            ports,
            ACCESS,
        )
        # Both ports are named; the tunnel takes the one published on the host, which a browser there names.
        public = self.PUBLIC
        self.assertEqual(
            (status, out["message"]),
            (
                403,
                f"this page answers only at 127.0.0.1:{PORT}, localhost:{PORT}, 127.0.0.1:{public} or "
                f"localhost:{public}; through an SSH tunnel, use the admin dashboard's port, {public}, at both "
                'ends (README.md, "The admin dashboard", shows the command)',
            ),
        )
        self.assertIn(own, webserve.origins(ports))
        self.assertEqual(
            webserve.origins(PORT), webserve.origins((PORT, 0))
        )  # 0: no published port of its own

    def test_a_server_bound_to_all_addresses_answers_for_its_published_port(self):
        srv = webserve.Server(CFG, self.store, 0, "0.0.0.0", self.PUBLIC)
        try:
            self.assertEqual(srv.server_address[0], "0.0.0.0")
            self.assertEqual(srv.ports, (srv.server_address[1], self.PUBLIC))
        finally:
            srv.server_close()


class TestPageVariants(unittest.TestCase):
    """The local page and the published copy come from the same code; only the local page contacts 127.0.0.1, and
    neither carries a secret."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = Store(pathlib.Path(self.tmp.name) / "state")
        self.cfg = dict(CFG, operator={"github_login": "oplogin"})
        self.st = empty_state()
        self.st["develop"]["sha"] = "d" * 40

    def tearDown(self):
        self.tmp.cleanup()

    def render(self, *variant) -> str:
        return webdash.render(
            self.cfg,
            self.st,
            self.store,
            [],
            [],
            report.board(self.cfg, self.st, self.store, [], []),
            *variant,
        )

    @staticmethod
    def section(page: str) -> str:
        s = page[page.index("<section id='plan'") :]
        return s[: s.index("</section>")]

    def test_the_local_file_talks_to_the_local_service(self):
        for mp in (
            None,
            {"unavailable": "no near-merge PR could be merged with develop here"},
        ):
            self.st["merge_plan"] = mp
            page = self.render(webdash.LOCAL)
            box = self.section(page)
            self.assertIn(
                "</h2><form id='prio' class='prio' data-port='18613' data-home='http://127.0.0.1:18613/' "
                "data-me='oplogin' autocomplete='off' aria-label='Request PR priority'>",
                box,
            )  # first in the merge plan box
            for absent in (TOKEN, "data-token", "X-Rkmon-Token", "dataset.token"):
                self.assertNotIn(absent, page)  # no secret in the page
            self.assertIn(
                "<input id='prio-n' type='number' min='1' max='9999999' step='1' placeholder='e.g. 101' "
                "required disabled>",
                box,
            )
            self.assertIn(
                "<button id='prio-go' type='submit' disabled>Request PR priority</button>",
                box,
            )
            self.assertNotIn("prio-who", box)  # the user prioritizes directly
            self.assertIn(
                "<p id='prio-status' class='sub' role='status' aria-live='polite'>Priority requests work here "
                "while oplogin's monitor service runs (tools/rkweb.py): the admin dashboard is "
                "http://127.0.0.1:18613/.</p>",
                box,
            )
            self.assertIn(
                "<h3>Priority requests <span class='c'>0 pending · oplogin approves or declines</span></h3>"
                "<p class='sub'>None right now.</p>",
                box,
            )
            self.assertEqual(page.count("<form id='prio'"), 1)
        # Served by the service, it pings its own origin with the custom header (so every cross-origin request is
        # preflighted), posts JSON there, and holds the 2-minute reload while a request is in flight; the browser
        # sends the access token again with each of those requests. Opened from disk, it only links the served page.
        self.assertIn("if(served){const stop=new AbortController();", page)
        self.assertIn("fetch(base+'/ping',{headers:{'X-Rkmon':'1'}", page)
        self.assertIn(
            "HOME=f.dataset.home||'http://127.0.0.1:'+port+'/',served=location.protocol!=='file:',\n"
            "base=location.origin,",
            page,
        )
        self.assertIn(
            "if(!served){const a=document.createElement('a');a.href=HOME;", page
        )
        self.assertIn(
            "open the admin dashboard at '),\na,document.createTextNode(' (it asks for the access token) for the "
            "box and the Approve and Decline buttons.')",
            page,
        )
        self.assertIn("post(base+'/priority',{pr:n})", page)
        self.assertIn(
            "post(base+'/'+b.dataset.route+'-'+b.dataset.act,{pr:+b.dataset.pr})", page
        )
        self.assertIn("if(window.rkmonBusy&&rkmonBusy()){left=15;return;}", page)

    def test_the_published_copy_never_contacts_the_service_nor_carries_the_secret(self):
        self.st["priority_requests"] = {
            "seen": {},
            "items": [
                {
                    "id": "abcdef01",
                    "pr": 7,
                    "who": "Jane",
                    "why": "x",
                    "at": now(),
                    "status": "pending",
                    "note": "",
                }
            ],
        }
        page = self.render()  # the default variant
        for absent in (
            "127.0.0.1",
            TOKEN,
            "X-Rkmon-Token",
            "data-token",
            "data-port",
            "/ping",
            "/priority-approve",
            "/priority-decline",
            "'/priority-'",
            "/review-",
            "data-route",
            "data-act",
            "data-local-only",
        ):
            self.assertNotIn(absent, page)
        box = self.section(page)
        self.assertIn(
            "</h2><form id='prio' class='prio' data-me='oplogin' autocomplete='off' "
            "aria-label='Request PR priority'>",
            box,
        )
        self.assertIn(
            "<span class='prio-ask' hidden><label for='prio-who'>Your name</label>", box
        )
        self.assertIn(
            "<p id='prio-status' class='sub' role='status' aria-live='polite'>Ask for a PR to be reviewed and "
            "merged first: oplogin approves or declines each request.</p>",
            box,
        )
        self.assertIn("<td>Jane</td><td>x</td>", box)  # the pending list, read-only
        self.assertIn(
            "post('priority-request',{pr:n,who:w,why:y})", page
        )  # the web host beside the page
        self.assertIn("'Requested: '+me+' will review it.", page)
        # A host that answers a POST with HTML (python3 -m http.server's 501) is named, not "try again later".
        self.assertIn(
            ".then(r=>r.json().catch(()=>({ok:false,status:r.status})))", page
        )
        self.assertIn(
            "'answered HTTP '+j.status+': it does not take requests yet. Tell '+me+'.'",
            page,
        )
        self.assertIn("'No answer from the web host: try again later.'", page)
        self.assertNotIn("did not go through", page)

    def test_an_in_depth_review_is_decided_on_its_needs_you_line(self):
        """On the admin dashboard, the line of an in-depth review waiting for approval has its cost, its Approve and
        Decline, shown once the page reaches the service, and a place for the service's answer; the published copy has
        the team's line alone: the size and the agents, with no cost and no command."""
        from rkmon_lib import split

        self.st["prs"] = {"8": pr_rec(8, size=80000)}
        self.st["review_first"] = [{"pr": 8, "again": False}]
        with mock.patch.dict(
            "os.environ", {"RKMON_WRAPPER": "rkmon"}
        ), mock.patch.object(split, "agent_cost", return_value=8.0):
            local, published = self.render(webdash.LOCAL), self.render()
        line = (
            "<li>#8 (a): an in-depth review of 80,000 changed lines needs your OK: about 11 agents, about $88. Approve "
            "it on the admin dashboard or with `rkmon rk review-approve 8`; decline it there or with `rkmon rk "
            "review-skip 8`."
        )
        # The buttons post to /review-approve and /review-decline (decide_buttons, as a priority request's).
        buttons = webdash.decide_buttons("review", 8)
        self.assertEqual(buttons.count("data-route='review'"), 2)
        needs = local[local.index("<section id='needs'") :]
        self.assertIn(
            f"{line} <span class='nw' data-local-only hidden>{buttons}</span> <span class='prio-st'></span></li>",
            needs[: needs.index("</section>")],
        )
        self.assertIn(
            "<li>#8: an in-depth review of 80,000 changed lines needs your OK: about 11 agents. Approve or decline it "
            "on the admin dashboard.</li>",
            published,
        )
        self.assertNotIn("$88", published)
        self.assertNotIn("review-approve", published)
        self.assertNotIn("data-route", published)

    def test_write_makes_both_and_pushes_only_the_published_copy(self):
        unread = pathlib.Path(self.tmp.name) / "home/.config/rkmon/admin_token"
        cfg = dict(
            admin_cfg(self.cfg, token_file=str(unread)),
            paths={
                "working_dir": self.tmp.name,
                "html": f"{self.tmp.name}/rocke_monitor.html",
            },
        )
        pushed = []
        with mock.patch.object(
            webdash.publish, "html", lambda c, path: pushed.append(path)
        ):
            path = webdash.write(
                cfg,
                self.st,
                self.store,
                [],
                [],
                report.board(cfg, self.st, self.store, [], []),
            )
        self.assertFalse(
            unread.exists()
        )  # nothing makes or reads admin_dashboard.token_file
        local = pathlib.Path(path).read_text(encoding="utf-8")
        published = (self.store.dir / webdash.PUBLISHED_FILE).read_text(
            encoding="utf-8"
        )
        self.assertEqual(
            (path, pushed),
            (cfg["paths"]["html"], [str(self.store.dir / webdash.PUBLISHED_FILE)]),
        )
        # The local page is yours only; the published copy is for others to read.
        self.assertEqual(
            (
                pathlib.Path(path).stat().st_mode & 0o777,
                (self.store.dir / webdash.PUBLISHED_FILE).stat().st_mode & 0o777,
            ),
            (0o600, 0o644),
        )
        for page in (local, published):
            self.assertNotIn("data-token", page)
            self.assertNotIn(TOKEN, page)
        self.assertIn("'http://127.0.0.1:'", local)
        self.assertNotIn("127.0.0.1", published)
        # The same page otherwise, but for the Tickets card (found-issue-tickets is on), the "Not sent" card (review
        # automation is in prototype) and the admin page's own cards (the scoreboard, review spend, today's log): the
        # admin dashboard's only.
        self.assertEqual(
            local.count("<section"),
            published.count("<section") + 2 + len(webdash.ADMIN_CARDS),
        )
        self.assertIn("<section id='not-sent' class='card'><h2>Not sent", local)
        self.assertNotIn("Not sent", published)
        self.assertIn(
            "<section id='tickets' class='card'><h2>Tickets waiting for you", local
        )
        self.assertNotIn("Tickets waiting for you", published)


def req(
    rid: str,
    pr,
    who: str = "Jane",
    why: str = "blocks the release",
    at: str | None = None,
) -> dict:
    """A line of the web host's requests file, as tools/webhost_server.py writes it."""
    return {
        "id": rid,
        "at": at or now(),
        "pr": pr,
        "who": who,
        "why": why,
        "addr": "10.0.0.1",
    }


class TestRequestQueue(Base):
    """Other teams' requests: their clicks are requests that wait for the operator's OK."""

    CFG = dict(
        CFG,
        publish={
            "html": dict(
                CFG["publish"]["html"],
                requests="rocke-monitor/requests/priority_requests.jsonl",
            )
        },
    )

    def pull(self, *records, why: str | None = None) -> list[str]:
        text = "".join(
            (json.dumps(r) if isinstance(r, dict) else r) + "\n" for r in records
        )

        def fetch(cfg, dest):
            if not why:
                pathlib.Path(dest).write_text(text)
            return why

        with self.store.locked():
            st = self.store.load()
            out = priority.pull(self.CFG, st, self.store, fetch)
            self.store.save(st)
        return out

    def item(self, n: int) -> dict:
        return [
            it
            for it in self.store.load()["priority_requests"]["items"]
            if it["pr"] == n
        ][-1]

    def test_pull_queues_dedupes_and_answers(self):
        st = self.store.load()
        st["prs"]["12"] = pr_rec(12)
        priority.set_priority(st, self.store, 12, "by hand")
        self.store.save(st)
        at12 = st["user_priority"]["12"]["at"]
        out = self.pull(
            req("aaaaaa01", 7),
            req("aaaaaa01", 7),
            req("aaaaaa02", 7, who="JANE"),
            req("aaaaaa03", 7, who="Bob", why="we wait\non it"),
            req("aaaaaa04", 8),
            req("aaaaaa05", 10),
            req("aaaaaa06", 12),
            req("aaaaaa07", 99),
            "not json",
            req("aaaaaa08", 7, who=""),
            dict(req("aaaaaa09", 7), pr="7"),
            req("bad id!", 7),
            req("aaaaaa10", 7, at="2026-01-01T00:00:00Z"),
        )
        self.assertEqual(
            out,
            [
                "PRIORITY_REQUEST #7 by Jane: blocks the release",
                "PRIORITY_REQUEST #7 by Bob: we wait on it",  # another requester: folded into #7's request
                "(priority request #8 by Jane answered: #8 is merged, not open)",
                "(priority request #10 by Jane answered: #10 is a top priority already: it fixes develop "
                "(silent-break #1111))",
                f"(priority request #12 by Jane answered: #12 is already in the priority queue since {at12[:10]} "
                f"{at12[11:16]}Z (by hand))",
                "(priority request #99 by Jane answered: #99 is not an open rocKE PR the monitor tracks (as of the last "
                "tick, 2026-11-03 17:20Z))",
            ],
        )
        self.assertEqual(
            self.pull(*[req(f"aaaaaa0{i}", 7) for i in range(1, 8)]), []
        )  # read already: no news
        st = self.store.load()
        shown = priority.shown(st)
        self.assertEqual([(it["pr"], it["status"]) for it in shown][0], (7, "pending"))
        # #8 merged: its row is cleared like every merged priority's.
        self.assertEqual(
            sorted((it["pr"], it["status"]) for it in shown[1:]),
            [(10, "answered"), (12, "answered"), (99, "answered")],
        )
        self.assertEqual(priority.requesters(shown[0]), "Jane, Bob")
        self.assertNotIn("7", st.get("user_priority") or {})  # waits for the user's OK
        # The board's Needs you, the md merge plan and both HTML boxes list it; untrusted text stays escaped.
        with mock.patch.dict("os.environ", {"RKMON_WRAPPER": "rkmon"}):
            needs = report.board(self.CFG, st, self.store, [], [])["needs"]
        self.assertIn(
            "Priority request for #7 by Jane, Bob: blocks the release. Approve it on the admin dashboard or with "
            "`rkmon rk prioritize 7`, or decline it there or with `rkmon rk priority-decline 7`.",
            needs,
        )
        self.assertEqual(sum(t.startswith("Priority request") for t in needs), 1)
        md = "\n".join(report.plan_md(self.CFG, st))
        self.assertIn("| #7 | Jane, Bob | blocks the release / we wait on it |", md)
        self.assertNotIn("#8 is merged", md)
        self.assertIn("| answered: #99 is not an open rocKE PR the monitor tracks", md)
        self.assertIn(
            "Not computed yet: the next tick computes it.", md
        )  # listed even before a plan exists
        box = webdash.priority_box(self.CFG, st, webdash.LOCAL)
        self.assertIn(
            "<td>Jane, Bob</td><td>blocks the release / we wait on it</td>", box
        )
        self.assertEqual(
            box.count("data-act='approve'"), 1
        )  # buttons for the pending request only
        self.assertIn(
            "<td class='nw' data-local-only hidden><button type='button' class='prio-act' data-route='priority' "
            "data-act='approve' data-pr='7'>Approve</button>",
            box,
        )
        self.assertIn("1 pending", box)
        published = webdash.priority_box(self.CFG, st)
        self.assertIn(
            "<td>Jane, Bob</td><td>blocks the release / we wait on it</td>", published
        )
        self.assertNotIn("data-act", published)  # read-only there

    def test_untrusted_text_is_one_short_line_and_escaped(self):
        self.pull(
            req(
                "bbbbbb01",
                7,
                who="<script>x</script>" + "w" * 60,
                why="a‮b\x00c\n" + "y" * 400,
            )
        )
        it = self.item(7)
        self.assertEqual(len(it["who"]), priority.WHO_MAX)
        self.assertEqual(
            it["why"][:5], "abc y"
        )  # the bidi override and the NUL go, the line break becomes a space
        self.assertEqual(len(it["why"]), priority.WHY_MAX)
        for box in (
            webdash.priority_box(self.CFG, self.store.load()),
            webdash.priority_box(self.CFG, self.store.load(), webdash.LOCAL),
        ):
            self.assertIn("&lt;script&gt;x&lt;/script&gt;", box)
            self.assertNotIn("<script>x", box)

    def test_a_failed_read_keeps_what_is_pending(self):
        self.pull(req("cccccc01", 7))
        self.assertEqual(
            self.pull(why="Permission denied (publickey)."),
            ["(priority requests not read: Permission denied (publickey).)"],
        )
        st = self.store.load()
        self.assertEqual(
            st["priority_requests"]["error"], "Permission denied (publickey)."
        )
        self.assertEqual(self.item(7)["status"], "pending")
        self.assertIn(
            "last read failed: Permission denied", priority.lines(self.CFG, st)[-1]
        )
        # Not configured: nothing is read and nothing is stored.
        st = empty_state()
        self.assertEqual(
            priority.pull(CFG, st, self.store, lambda *a: self.fail("no read")), []
        )
        self.assertNotIn("priority_requests", st)

    def test_approve_and_decline_from_the_served_page(self):
        self.pull(req("dddddd01", 7), req("dddddd02", 7, who="Bob"), req("dddddd03", 9))
        self.assertEqual(
            self.ask(
                "POST",
                "/priority-approve",
                hdrs(Authorization=DROP),
                b'{"pr": 7}',
            )[0],
            401,
        )
        status, h, out = self.ask("POST", "/priority-approve", hdrs(), b'{"pr": 7}')
        self.assertEqual(
            (status, out["ok"], h["Access-Control-Allow-Origin"]), (200, True, OWN)
        )
        st = self.store.load()
        why = "requested by Jane, Bob from the dashboard, approved by me"
        self.assertEqual(st["user_priority"]["7"]["why"], why)
        self.assertEqual(st["develop_fixes"]["7"], [f"your priority: {why}"])
        self.assertEqual(
            (self.item(7)["status"], self.item(7)["decided_by"]), ("approved", "me")
        )
        self.assertEqual(
            self.ask("POST", "/priority-approve", hdrs(), b'{"pr": 7}')[0::2],
            (
                409,
                {
                    "ok": False,
                    "message": "#7 has no pending priority request: nothing changed.",
                },
            ),
        )
        status, _, out = self.ask("POST", "/priority-decline", hdrs(), b'{"pr": 9}')
        self.assertEqual(
            (status, out["message"]),
            (200, "Declined the priority request for #9 (by Jane)."),
        )
        st = self.store.load()
        self.assertEqual(self.item(9)["status"], "declined")
        self.assertNotIn("9", st["user_priority"])
        self.assertEqual(
            [(it["pr"], it["status"]) for it in priority.shown(st)], [(7, "approved")]
        )  # declined: gone
        j = self.journal()
        self.assertRegex(
            j,
            r"mode +priority request #9 by Jane \(blocks the release\) declined by me\n",
        )
        self.assertRegex(
            j,
            rf"web +POST /priority-approve #7 from {re.escape(OWN)}: done: #7 approved",
        )
        # The same checks guard the new routes.
        self.assertEqual(
            self.ask(
                "POST",
                "/priority-decline",
                hdrs(Origin="http://evil.example"),
                b'{"pr": 9}',
            )[0],
            403,
        )
        self.assertEqual(
            self.ask("POST", "/priority-approve", hdrs(X_Rkmon=DROP), b'{"pr": 9}')[0],
            403,
        )

    def test_the_state_overtakes_a_pending_request(self):
        st = self.store.load()
        st["prs"]["13"] = pr_rec(13)
        self.store.save(st)
        self.pull(req("eeeeee01", 7), req("eeeeee02", 9), req("eeeeee03", 13))
        st = self.store.load()
        st["prs"]["9"]["status"] = "merged"
        self.store.save(st)
        self.pull()  # the next tick
        self.assertEqual(
            (self.item(9)["status"], self.item(9)["note"]),
            ("answered", "#9 is merged, not open any more"),
        )
        # The user's own direct priority (the box on the admin dashboard) approves the pending request too.
        self.assertEqual(self.ask("POST", "/priority", hdrs(), b'{"pr": 7}')[0], 200)
        self.assertEqual(
            (self.item(7)["status"], self.item(7)["decided_by"]), ("approved", "me")
        )
        self.assertEqual(
            self.store.load()["user_priority"]["7"]["why"], priority.DASHBOARD_WHY
        )
        # A PR closed before the next tick: its approval answers the request instead of prioritizing it.
        st = self.store.load()
        st["prs"]["13"]["status"] = "closed"
        self.store.save(st)
        self.assertEqual(
            self.ask("POST", "/priority-approve", hdrs(), b'{"pr": 13}')[0::2],
            (
                409,
                {
                    "ok": False,
                    "message": "#13 is closed, not open: the request is answered, nothing else "
                    "changed.",
                },
            ),
        )
        self.assertEqual(self.item(13)["status"], "answered")
        self.assertNotIn("13", self.store.load()["user_priority"])


class TestRequestCli(Base):
    CFG = TestRequestQueue.CFG
    pull = TestRequestQueue.pull
    item = TestRequestQueue.item
    run_rk = TestCli.run_rk

    def test_list_approve_and_decline_by_cli(self):
        self.pull(
            req("ffffff01", 7), req("ffffff02", 9, why="needed by the 7.2 branch")
        )
        out = "\n".join(self.run_rk(["priority-requests"])).splitlines()
        self.assertTrue(out[0].startswith("#7 pending · requested by Jane at "), out)
        self.assertTrue(out[1].endswith(": needed by the 7.2 branch"), out)
        self.assertTrue(out[-1].startswith("(2 pending; read from the web host "), out)
        self.run_rk(
            ["prioritize", "7"]
        )  # approving by CLI: the reason names the requester
        self.assertEqual(
            self.store.load()["user_priority"]["7"]["why"],
            "requested by Jane from the dashboard, approved by me",
        )
        self.assertEqual(self.item(7)["status"], "approved")
        self.assertEqual(
            self.run_rk(["priority-decline", "9"]),
            ["Declined the priority request for #9 (by Jane)."],
        )
        self.assertEqual(self.item(9)["status"], "declined")
        self.assertEqual(
            self.run_rk(["priority-decline", "9"], rc=4),
            ["REFUSED: #9 has no pending priority request: nothing changed."],
        )


class TestInDepthReview(Base):
    """An in-depth review's Approve and Decline on the served page's Needs you list do what `rk review-approve N` and
    `rk review-skip N` do, journalled; a PR with no in-depth review is refused, nothing changed.
    """

    def decide(self, act: str, n: int, **change) -> tuple:
        return self.ask(
            "POST", f"/review-{act}", hdrs(**change), json.dumps({"pr": n}).encode()
        )

    def test_the_served_page_approves_and_declines_as_the_commands_do(self):
        st = self.store.load()
        st["prs"]["12"] = pr_rec(12, size=80000)
        self.store.save(st)
        self.assertEqual(self.decide("approve", 12, Authorization=DROP)[0], 401)
        self.assertEqual(
            self.decide("decline", 12, Origin="http://evil.example")[0], 403
        )
        status, h, out = self.decide("approve", 12)
        self.assertEqual(
            (status, out["ok"], h["Access-Control-Allow-Origin"]), (200, True, OWN)
        )
        self.assertEqual(
            out["message"],
            "#12 approved: its in-depth review (80,000 changed lines: about 11 agents) starts at its turn.",
        )
        self.assertEqual([a["pr"] for a in self.store.load()["review_approved"]], [12])
        status, _, out = self.decide("decline", 12)
        self.assertEqual((status, out["ok"]), (200, True))
        st = self.store.load()
        self.assertEqual((st["review_skip"], st["review_approved"]), ([12], []))
        refused = (
            "#7 is 100 changed lines, within the 30,000 the autopilot reviews: its review needs no approval. "
            "Nothing changed."
        )
        for act in ("approve", "decline"):
            with self.subTest(act=act):
                self.assertEqual(
                    self.decide(act, 7)[0::2],
                    (409, {"ok": False, "message": refused}),
                )
        st = self.store.load()
        self.assertEqual((st["review_skip"], st["review_approved"]), ([12], []))
        j = self.journal()
        self.assertRegex(
            j,
            rf"web +POST /review-approve #12 from {re.escape(OWN)}: done: #12 approved",
        )
        self.assertRegex(
            j, rf"web +POST /review-decline #7 from {re.escape(OWN)}: refused: #7 is"
        )


class TestServedPage(Base):
    """The service serves the local page at /, so the page and its requests share one origin; every request carries
    the access token, a foreign page or Host never gets it, and it is never framed. A file opened from disk only links
    to the served page."""

    def setUp(self):
        super().setUp()
        self.html = pathlib.Path(self.tmp.name) / "rocke_monitor.html"
        self.cfg = dict(
            CFG, paths={"working_dir": str(self.html.parent), "html": str(self.html)}
        )

    def get(self, path: str = "/", **change) -> tuple:
        """A browser's navigation on this machine, with the access token it was given."""
        h = _changed(
            {
                "Host": f"127.0.0.1:{PORT}",
                "Sec-Fetch-Site": "none",
                "Sec-Fetch-Mode": "navigate",
                "Authorization": basic(),
            },
            change,
        )
        return webserve.answer(self.cfg, self.store, "GET", path, h, b"", PORT, ACCESS)

    def test_a_navigation_with_the_token_gets_the_page_never_framed(self):
        self.html.write_text("<html>the local page é</html>", encoding="utf-8")
        for path, change in (
            ("/", {}),
            ("/index.html", {}),
            ("/", {"Sec-Fetch-Site": "same-origin"}),
            ("/", {"Host": f"localhost:{PORT}"}),
            ("/", {"Origin": f"http://127.0.0.1:{PORT}"}),
        ):
            status, h, body = self.get(path, **change)
            self.assertEqual(
                (status, body),
                (200, "<html>the local page é</html>".encode("utf-8")),
                (path, change),
            )
            self.assertEqual(
                h,
                {
                    "Content-Type": "text/html; charset=utf-8",
                    "X-Frame-Options": "DENY",
                    "Content-Security-Policy": "frame-ancestors 'none'",
                    "X-Content-Type-Options": "nosniff",
                    "Referrer-Policy": "no-referrer",
                },
            )  # no CORS header (no other origin may read it), no cookie
        # The browser asks for an icon with every load (never journaled as a refusal): no icon, no line, no token.
        self.assertEqual(
            self.get("/favicon.ico", Sec_Fetch_Mode="no-cors", Authorization=DROP),
            (204, {}, None),
        )
        self.assertEqual(self.journal(), "")

    def test_without_the_token_no_page_and_no_cookie(self):
        """A browser's navigation (its Sec-Fetch headers) without the token gets the Basic challenge only: the
        browser asks for the token, and nothing names a secret."""
        self.html.write_text("<html>the local page</html>", encoding="utf-8")
        for change in ({"Authorization": DROP}, {"Authorization": basic("wrong")}):
            self.assertEqual(
                self.get(**change),
                (
                    401,
                    {"WWW-Authenticate": webserve.REALM},
                    {"ok": False, "message": webserve.NEED_ACCESS},
                ),
                change,
            )
        self.assertIn(
            f"refused GET / from none: {webserve.NEED_ACCESS}", self.journal()
        )

    def test_a_client_is_told_apart_by_the_token(self):
        """Any client on this machine without the token gets no page, whatever its Sec-Fetch headers say; with the
        token (the user's own curl, say) it gets it."""
        self.html.write_text("<html>the page</html>", encoding="utf-8")
        for change in (
            {"Sec-Fetch-Site": DROP, "Sec-Fetch-Mode": DROP},  # curl, urllib
            {"Sec-Fetch-Mode": DROP},
            {"Sec-Fetch-Mode": "cors"},
        ):
            self.assertEqual(self.get(Authorization=DROP, **change)[0], 401, change)
            self.assertEqual(
                self.get(**change)[0::2], (200, b"<html>the page</html>"), change
            )

    def test_a_foreign_page_or_host_does_not_get_it(self):
        self.html.write_text("<html>the admin page</html>", encoding="utf-8")
        for change, why in (
            ({"Host": f"rebound.example:{PORT}"}, "host not allowed"),
            ({"Origin": "http://evil.example"}, "origin not allowed"),
            ({"Origin": PUB}, "origin not allowed"),
            ({"Origin": "null"}, "origin not allowed"),
            ({"Sec-Fetch-Site": "cross-site"}, "the page opens only on this machine"),
            ({"Sec-Fetch-Site": "same-site"}, "the page opens only on this machine"),
        ):
            status, h, body = self.get(**change)
            said = HOST_REFUSAL if "Host" in change else why
            self.assertEqual(
                (status, body), (403, {"ok": False, "message": said}), change
            )
            self.assertNotIn("Access-Control-Allow-Origin", h)
        # Another Host is refused before the token is asked for: a browser never asks for it under another name.
        self.assertEqual(
            self.get(Host=f"rebound.example:{PORT}", Authorization=DROP)[0:2],
            (403, {}),
        )
        self.assertIn(
            "refused GET / from none: the page opens only on this machine",
            self.journal(),
        )

    def test_over_http(self):
        self.html.write_text("<html>over http</html>", encoding="utf-8")
        srv = webserve.Server(self.cfg, self.store, 0)
        port = srv.server_address[1]
        srv.access = webserve.access_info(self.cfg, srv.ports, {})
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            with mock.patch.object(sys, "stderr"):  # the service's request log
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                c.request(
                    "GET",
                    "/",
                    headers={
                        "Host": f"127.0.0.1:{port}",
                        "Sec-Fetch-Site": "none",
                        "Sec-Fetch-Mode": "navigate",
                        "Authorization": basic(),
                    },
                )
                r = c.getresponse()
                body = r.read()
                c.close()
        finally:
            srv.shutdown()
            srv.server_close()
            t.join(5)
        h = {k.lower(): v for k, v in r.getheaders()}
        self.assertEqual(
            (
                r.status,
                body,
                h["content-type"],
                h["x-frame-options"],
                h["cache-control"],
            ),
            (
                200,
                b"<html>over http</html>",
                "text/html; charset=utf-8",
                "DENY",
                "no-store",
            ),
        )

    def test_no_page_yet(self):
        # Before the first tick: a page that says so and reloads itself, unjournaled (it reloads every 30 s).
        status, h, body = self.get()
        self.assertEqual(
            (status, h["Content-Type"], h["X-Frame-Options"]),
            (200, "text/html; charset=utf-8", "DENY"),
        )
        self.assertIn(b"No dashboard page yet: the next tick writes it.", body)
        self.assertIn(b"<meta http-equiv='refresh' content='30'>", body)
        self.assertEqual(self.journal(), "")

    def test_the_served_page_calls_its_own_origin(self):
        own, alias = f"http://127.0.0.1:{PORT}", f"http://localhost:{PORT}"
        # A same-origin GET carries no Origin; Sec-Fetch-Site vouches for it, and no CORS header is needed.
        status, h, out = self.ask(
            "GET", "/ping", hdrs(Origin=DROP, Sec_Fetch_Site="same-origin")
        )
        self.assertEqual((status, out), (200, {"ok": True, "me": "me"}))
        self.assertNotIn("Access-Control-Allow-Origin", h)
        for site in ("cross-site", "same-site", "none", DROP):
            self.assertEqual(
                self.ask("GET", "/ping", hdrs(Origin=DROP, Sec_Fetch_Site=site))[0::2],
                (403, {"ok": False, "message": "origin not allowed"}),
                site,
            )
        # A POST always names its origin: the served page's, with the access token.
        status, h, out = self.post(7, Origin=own)
        self.assertEqual(
            (status, out["ok"], h["Access-Control-Allow-Origin"]), (200, True, own)
        )
        self.assertEqual(sorted(self.store.load()["user_priority"]), ["7"])
        self.assertEqual(
            self.post(10, Origin=DROP, Sec_Fetch_Site="same-origin")[0::2],
            (403, {"ok": False, "message": "origin not allowed"}),
        )  # a POST without an Origin is no page's
        self.assertEqual(
            self.post(10, Origin=own, Authorization=DROP)[0::2],
            (401, {"ok": False, "message": webserve.NEED_ACCESS}),
        )
        # Opened as localhost, it calls 127.0.0.1 cross-origin: preflighted and allowed.
        status, h = webserve.preflight(
            CFG, webserve._headers(preflight_hdrs(Origin=alias)), PORT
        )
        self.assertEqual((status, h["Access-Control-Allow-Origin"]), (204, alias))
        self.assertEqual(
            webserve.preflight(
                CFG,
                webserve._headers(preflight_hdrs(Origin="http://localhost:1")),
                PORT,
            ),
            (403, {}),
        )


if __name__ == "__main__":
    unittest.main()
