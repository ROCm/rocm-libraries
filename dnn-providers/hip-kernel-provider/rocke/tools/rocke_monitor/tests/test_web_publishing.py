# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Where the dashboards are published: the admin dashboard on the loopback by default, every request with a
configured access token, from this machine too (never in a page), and the service refuses to start without one;
the team dashboard published on the team web host or served by this monitor, read-only but for its
request box; in the container, compose publishes what the settings name and rk doctor says when docker/.env
differs."""

import base64
import http.client
import json
import os
import pathlib
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import sandbox  # noqa: E402
from rkmon_lib import (
    config,
    doctor,
    features,
    priority,
    publish,
    report,
    state_store,
    summary,
)  # noqa: E402
from rkmon_lib import web_host_install, webdash, webserve  # noqa: E402
from rkmon_lib.store import Store, empty_state, now  # noqa: E402

ACCESS = "access-" + "token-not-real-0123"  # built at run time
URL = "http://monitor.example.com:18613/"
HOST_ENV = {"RKMON_CONTAINER": ""}
IMAGE_ENV = {"RKMON_CONTAINER": "1"}


def basic(token: str, user: str = "anyone") -> str:
    return "Basic " + base64.b64encode(f"{user}:{token}".encode()).decode()


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = pathlib.Path(self.tmp.name)
        self.store = Store(self.d / "state")
        st = empty_state()
        st["develop"]["sha"] = "d" * 40
        st["prs"] = {
            "7": {
                "n": 7,
                "head": "a" * 40,
                "status": "open",
                "draft": False,
                "author": "a",
                "title": "t",
                "labels": [],
                "size": 1,
            }
        }
        self.store.save(st)
        (self.d / "secrets").mkdir(mode=0o700)
        self.access_file = self.d / "secrets" / "admin_access_token"

    def cfg(self, **admin) -> dict:
        return config.normalize(
            {
                "repo": "acme/widgets",
                "operator": {"github_login": "op"},
                "paths": {
                    "working_dir": str(self.d),
                    "secrets_dir": str(self.d / "secrets"),
                    "html": str(self.d / "rocke_monitor.html"),
                },
                "features": {},
                "admin_dashboard": dict({"port": 18613, "bind": "127.0.0.1"}, **admin),
            }
        )  # with the places the load derives

    def set_access(self, value: str = ACCESS) -> None:
        self.access_file.write_text(value + "\n", encoding="utf-8")
        self.access_file.chmod(0o600)


class TestBeyondThisMachine(Base):
    def test_where_the_page_is_reachable(self):
        from rkmon_lib import privacy

        self.assertTrue(
            privacy.LOCAL_HOSTS.match("192.0.2.7:18613")
        )  # a reserved documentation address
        self.assertFalse(privacy.LOCAL_HOSTS.match("192.0.3.7"))
        cfg = self.cfg()
        self.assertIsNone(webserve.exposure(cfg, HOST_ENV))
        for bind in ("0.0.0.0", "192.0.2.7"):
            self.assertEqual(webserve.exposure(self.cfg(bind=bind), HOST_ENV), bind)
        # In the image: where docker publishes it, as the container's environment says; no setting names it.
        self.assertIsNone(
            webserve.exposure(self.cfg(bind="0.0.0.0"), IMAGE_ENV)
        )  # inside: every address, by need
        self.assertEqual(
            webserve.exposure(cfg, dict(IMAGE_ENV, RKMON_ADMIN_HOST="0.0.0.0")),
            "0.0.0.0",
        )
        self.assertIsNone(
            webserve.exposure(self.cfg(publish_host="192.0.2.7"), IMAGE_ENV)
        )
        self.assertEqual(
            webserve.home_url(self.cfg(), HOST_ENV), "http://127.0.0.1:18613/"
        )
        self.assertEqual(
            webserve.home_url(self.cfg(bind="192.0.2.7"), HOST_ENV),
            "http://192.0.2.7:18613/",
        )
        self.assertEqual(webserve.home_url(self.cfg(url=URL[:-1]), HOST_ENV), URL)
        self.assertEqual(
            webserve.home_url(self.cfg(), dict(IMAGE_ENV, RKMON_PORT="18620")),
            "http://127.0.0.1:18620/",
        )

    def test_it_does_not_start_without_the_access_token_and_says_why(self):
        # Wherever it listens, the loopback included (rk serve --bind counts too), and the fix makes the token.
        for cfg, bind in (
            (self.cfg(), None),
            (self.cfg(), "0.0.0.0"),
            (self.cfg(bind="192.0.2.7"), None),
        ):
            self.assertEqual(
                webserve.access_refusal(cfg, HOST_ENV, bind),
                "the admin dashboard needs its access token on every request, from this machine too, and none is "
                f"set ({self.access_file}): run rkmon secret admin_access_token --generate",
                bind,
            )
        self.set_access()
        self.assertIsNone(webserve.access_refusal(self.cfg(), HOST_ENV))
        self.assertIsNone(webserve.access_refusal(self.cfg(bind="192.0.2.7"), HOST_ENV))
        for cfg, bind in ((self.cfg(bind="0.0.0.0"), None), (self.cfg(), "0.0.0.0")):
            self.assertIn(
                "set admin_dashboard.url", webserve.access_refusal(cfg, HOST_ENV, bind)
            )
        self.assertIsNone(
            webserve.access_refusal(self.cfg(bind="0.0.0.0", url=URL), HOST_ENV)
        )
        for admin in ({"bind": "192.0.2.8"}, {}):  # beyond this machine, or on it
            self.assertIn(
                "the access token, which every request carries (the secret admin_access_token: "
                "rkmon secret admin_access_token --generate)",
                "; ".join(
                    features.missing(
                        dict(
                            self.cfg(),
                            admin_dashboard=dict(
                                admin, access_token_file=str(self.d / "none")
                            ),
                        ),
                        "admin-dashboard",
                    )
                ),
                admin,
            )

    def test_the_refusal_holds_its_fix_apart_from_its_sentence(self):
        """The refusal is the sentence the service prints, which names the fix, and holds the two apart, as rk doctor's
        row shows them: the sentence without the fix, and the fix, for each way the service may not start.
        """
        from rkmon_lib import paths, private_files
        from rkmon_lib.checks import Edit

        target = self.d / "elsewhere"
        target.write_text(ACCESS + "\n", encoding="utf-8")
        self.access_file.symlink_to(target)
        (rule,) = [
            r
            for r in private_files.problems(self.access_file, paths.PACKAGE)
            if r.status == "FAIL"
        ]
        r = webserve.access_refusal(self.cfg(), HOST_ENV)
        self.assertEqual(r.fix, rule.fix)
        self.assertTrue(r.why.endswith(f"private file: {rule.found}"), r.why)
        self.assertEqual(r, f"{r.why} (fix: {rule.fix})")
        self.access_file.unlink()
        r = webserve.access_refusal(self.cfg(), HOST_ENV)
        self.assertEqual(str(r.fix), "rkmon secret admin_access_token --generate")
        self.assertEqual(r, f"{r.why}: run {r.fix}")
        self.set_access()
        r = webserve.access_refusal(self.cfg(bind="0.0.0.0"), HOST_ENV)
        self.assertIsInstance(r.fix, Edit)
        self.assertEqual((r.fix.key, r), ("admin_dashboard.url", r.why))

    def test_every_request_carries_it(self):
        self.set_access()
        cfg = self.cfg(bind="0.0.0.0", url=URL)
        (self.d / "rocke_monitor.html").write_text(
            "<html>the page</html>", encoding="utf-8"
        )
        access = webserve.access_info(cfg, (18613,), HOST_ENV)
        self.assertEqual(
            (access["hosts"], access["origins"]),
            (("monitor.example.com:18613",), ("http://monitor.example.com:18613",)),
        )

        def ask(method, path, auth=None, body=b"", **h):
            headers = {"Host": "monitor.example.com:18613", **h}
            if auth:
                headers["Authorization"] = auth
            return webserve.answer(
                cfg, self.store, method, path, headers, body, 18613, access
            )

        status, h, out = ask("GET", "/")
        self.assertEqual(
            (status, h["WWW-Authenticate"], out["message"]),
            (
                401,
                'Basic realm="rkmon admin dashboard", charset="UTF-8"',
                webserve.NEED_ACCESS,
            ),
        )
        for auth in (basic(ACCESS), basic(ACCESS, "")):
            self.assertEqual(
                ask("GET", "/", auth, Sec_Fetch_Site="none")[0:3:2],
                (200, b"<html>the page</html>"),
                auth,
            )
        for auth in (
            basic("wrong"),
            basic(ACCESS + "x"),
            "Basic !!!",
            f"Token {ACCESS}",
            f"Bearer {ACCESS}",  # HTTP Basic only
            ACCESS,
        ):
            self.assertEqual(ask("GET", "/", auth)[0], 401, auth)
        # Another Host is told the names the page answers to, admin_dashboard.url's among them, and the tunnel.
        self.assertEqual(
            ask("GET", "/", basic(ACCESS), Host="rebound.example:18613")[0:3:2],
            (
                403,
                {
                    "ok": False,
                    "message": "this page answers only at 127.0.0.1:18613, localhost:18613 or "
                    "monitor.example.com:18613; through an SSH tunnel, use the admin dashboard's port, 18613, at "
                    'both ends (README.md, "The admin dashboard", shows the command)',
                },
            ),
        )
        # The served page posts with the browser's credentials, its own origin and the header; no per-machine token.
        post = dict(
            Origin="http://monitor.example.com:18613",
            X_Rkmon="1",
            Content_Type="application/json",
        )
        post = {k.replace("_", "-"): v for k, v in post.items()}
        status, _h, out = ask("POST", "/priority", basic(ACCESS), b'{"pr": 7}', **post)
        self.assertEqual((status, out["ok"]), (200, True), out)
        self.assertEqual(ask("POST", "/priority", None, b'{"pr": 7}', **post)[0], 401)
        self.assertEqual(
            ask("GET", "/favicon.ico")[0], 204
        )  # the icon and preflights need no token
        self.assertNotIn(ACCESS, self.store.journal_path.read_text())
        self.assertIn(
            "refused GET / from none: the access token is needed",
            self.store.journal_path.read_text(),
        )

    def test_over_http(self):
        self.set_access()
        cfg = self.cfg(bind="192.0.2.7")
        srv = webserve.Server(cfg, self.store, 0)
        port = srv.server_address[1]
        srv.access = webserve.access_info(cfg, srv.ports, HOST_ENV)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            with mock.patch.object(sys, "stderr"):

                def get(headers):
                    c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                    c.request(
                        "GET",
                        "/ping",
                        headers=dict(
                            {
                                "Host": f"127.0.0.1:{port}",
                                "X-Rkmon": "1",
                                "Sec-Fetch-Site": "same-origin",
                            },
                            **headers,
                        ),
                    )
                    r = c.getresponse()
                    r.read()
                    c.close()
                    return r.status, {k.lower(): v for k, v in r.getheaders()}

                status, h = get({})
                self.assertEqual((status, h["www-authenticate"]), (401, webserve.REALM))
                self.assertEqual(get({"Authorization": basic(ACCESS)})[0], 200)
        finally:
            srv.shutdown()
            srv.server_close()
            t.join(5)

    def test_the_local_file_carries_no_secret_beyond_this_machine(self):
        self.set_access()
        cfg = dict(
            self.cfg(
                bind="0.0.0.0",
                url=URL,
                token_file=str(self.d / "secrets" / "admin_token"),
            )
        )
        st = self.store.load()
        with mock.patch.dict(os.environ, HOST_ENV), mock.patch.object(
            webdash.publish, "html", lambda c, p: None
        ):
            path = webdash.write(
                cfg, st, self.store, [], [], report.board(cfg, st, self.store, [], [])
            )
        page = pathlib.Path(path).read_text(encoding="utf-8")
        self.assertNotIn("data-token", page)
        self.assertNotIn(ACCESS, page)
        self.assertIn(f"data-home='{URL}'", page)
        self.assertIn("the admin dashboard is http://monitor.example.com:18613/.", page)
        self.assertFalse(
            (self.d / "secrets" / "admin_token").exists()
        )  # no per-machine secret made for it


class TestTeamDashboardServedHere(Base):
    def team_cfg(self, **team) -> dict:
        return dict(
            self.cfg(),
            features={"dashboards": {"team": {"enabled": True}}},
            team_dashboard=dict(
                {"mode": "serve", "bind": "127.0.0.1", "port": sandbox.free_port()},
                **team,
            ),
        )

    def test_served_read_only_with_its_request_box(self):
        cfg = self.team_cfg()
        self.assertTrue(features.active(cfg, "team-dashboard"))  # no web host needed
        self.assertTrue(webserve.team_serving(cfg))
        self.assertIn(
            "served here at http://127.0.0.1:18615/",
            features.configured_with(
                dict(cfg, team_dashboard={"mode": "serve"}), "team-dashboard"
            ),
        )
        pub = self.store.dir / webdash.PUBLISHED_FILE
        pub.write_text("<html>the team page</html>", encoding="utf-8")
        self.assertIsNone(publish.html(cfg, str(pub)))
        site = publish.site_dir(self.store.dir)
        self.assertEqual(
            (site / "index.html").read_text(encoding="utf-8"),
            "<html>the team page</html>",
        )
        srv, line = webserve.team_server(cfg, self.store, HOST_ENV)
        self.assertIn("RKWEB serving the team dashboard on 127.0.0.1:", line)
        port = srv.server_address[1]
        self.assertEqual(port, cfg["team_dashboard"]["port"])  # the port it is given
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        try:
            with mock.patch.object(sys, "stderr"):
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                c.request("GET", "/")
                r = c.getresponse()
                body = r.read()
                c.close()
                self.assertEqual((r.status, body), (200, b"<html>the team page</html>"))
                c = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                c.request(
                    "POST",
                    "/priority-request",
                    body=json.dumps({"pr": 7, "who": "Jane", "why": "blocks"}),
                    headers={"Content-Type": "application/json", "X-Rkmon": "1"},
                )
                r = c.getresponse()
                out = json.loads(r.read())
                c.close()
                self.assertEqual((r.status, out["ok"]), (200, True))
        finally:
            srv.shutdown()
            srv.server_close()
            t.join(5)
        self.assertEqual(
            json.loads(publish.requests_file(self.store.dir).read_text())["who"], "Jane"
        )
        # The tick reads them as it reads the web host's.
        st = self.store.load()
        lines = priority.pull(cfg, st, self.store, publish.pull_requests)
        self.assertEqual(lines, ["PRIORITY_REQUEST #7 by Jane: blocks"])
        # Off, or published on the web host: nothing served here.
        self.assertEqual(
            webserve.team_server(dict(cfg, features={}), self.store, HOST_ENV),
            (None, ""),
        )
        self.assertEqual(
            webserve.team_server(
                dict(cfg, team_dashboard={"mode": "publish"}), self.store, HOST_ENV
            ),
            (None, ""),
        )

    def test_it_serves_with_the_server_the_web_host_install_copies(self):
        """The monitor serves the team dashboard with the web host's own server, from its one home
        (web_host_install.SERVER)."""
        fake = self.d / "server.py"
        fake.write_text(
            "def make_server(site, requests, port, bind, origin):\n"
            "    raise ValueError('the server the install copies')\n",
            encoding="utf-8",
        )
        with mock.patch.object(web_host_install, "SERVER", fake):
            srv, line = webserve.team_server(self.team_cfg(), self.store, HOST_ENV)
        if srv is not None:  # a server from another file, made
            srv.server_close()
        self.assertIsNone(srv)
        self.assertTrue(line.endswith(": the server the install copies"), line)

    def test_in_the_image_every_address_inside(self):
        cfg = self.team_cfg(port=8080, publish_host="192.0.2.7", public_port=18615)
        self.assertEqual(
            webserve.team_where(cfg, IMAGE_ENV), ("0.0.0.0", webserve.TEAM_INNER_PORT)
        )
        self.assertEqual(webserve.team_where(cfg, HOST_ENV), ("127.0.0.1", 8080))
        self.assertEqual(webserve.team_url(cfg, IMAGE_ENV), "http://192.0.2.7:18615/")

    def test_in_standby_a_short_page_alone_until_standby_ends_with_no_restart(self):
        """While the monitor is in standby (standby.on in the internal state, read as each request comes),
        the team dashboard served here answers every request with a short page that says so and names what ends it,
        and takes in no priority request; once standby ends, the same server serves the page and takes requests.
        """
        home = mock.patch.dict(os.environ, {"RKMON_HOME": str(self.d)})
        home.start()
        self.addCleanup(home.stop)
        site = publish.site_dir(self.store.dir)
        site.mkdir(parents=True)
        (site / "index.html").write_text("<html>the team page</html>", encoding="utf-8")
        srv, _line = webserve.team_server(self.team_cfg(), self.store, HOST_ENV)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        self.addCleanup(t.join, 5)
        self.addCleanup(srv.server_close)
        self.addCleanup(srv.shutdown)
        quiet = mock.patch.object(sys, "stderr")  # the server's request log
        quiet.start()
        self.addCleanup(quiet.stop)

        def call(method: str, path: str, **asked) -> tuple[int, str]:
            c = http.client.HTTPConnection(
                "127.0.0.1", srv.server_address[1], timeout=10
            )
            try:
                headers = {"Content-Type": "application/json", "X-Rkmon": "1"}
                c.request(
                    method,
                    path,
                    body=json.dumps(asked) if asked else None,
                    headers=headers,
                )
                r = c.getresponse()
                return r.status, r.read().decode("utf-8")
            finally:
                c.close()

        def ask() -> tuple[int, dict]:
            status, text = call(
                "POST", "/priority-request", pr=7, who="Jane", why="blocks"
            )
            return status, json.loads(text)

        notice = (
            "This rocKE monitor is in standby: its team dashboard opens here once its operator runs rkmon "
            "standby off."
        )
        self.assertEqual(call("GET", "/"), (200, "<html>the team page</html>"))
        state_store.set(self.d, "standby.on", True)
        status, page = call("GET", "/")
        self.assertEqual(status, 503)
        self.assertIn(f"<body><p>{notice}</p></body>", page)
        self.assertNotIn("the team page", page)
        self.assertEqual(call("GET", "/index.html"), (status, page))
        self.assertEqual(ask(), (503, {"ok": False, "message": notice}))
        self.assertFalse(publish.requests_file(self.store.dir).exists())
        state_store.set(self.d, "standby.on", False)
        self.assertEqual(call("GET", "/"), (200, "<html>the team page</html>"))
        self.assertEqual(ask()[0], 200)
        self.assertEqual(
            json.loads(publish.requests_file(self.store.dir).read_text())["who"], "Jane"
        )
        self.assertTrue(t.is_alive())  # the one server all along


class TestContainerPublishing(Base):
    def test_the_env_lines_and_the_mismatch(self):
        cfg = dict(
            self.cfg(public_port=18620),
            features={"dashboards": {"team": {"enabled": True, "mode": "live"}}},
            team_dashboard={
                "mode": "serve",
                "publish_host": "0.0.0.0",
                "public_port": 18625,
            },
        )
        self.assertEqual(
            [f"{k}={v}" for k, v, _ in webserve.publish_env(cfg)],
            [
                "RKMON_ADMIN_HOST=127.0.0.1",
                "RKMON_PORT=18620",
                "RKMON_TEAM_HOST=0.0.0.0",
                "RKMON_TEAM_PORT=18625",
            ],
        )
        env = dict(
            IMAGE_ENV,
            RKMON_PORT="18614",
            RKMON_ADMIN_HOST="127.0.0.1",
            RKMON_TEAM_HOST="0.0.0.0",
        )
        self.assertEqual(
            webserve.publish_mismatch(cfg, env),
            [
                "RKMON_PORT is 18614 here; the settings say 18620",
                "RKMON_TEAM_PORT is unset (a port docker picks) here; the settings say 18625",
            ],
        )
        checks = doctor.web_checks(cfg, env, answers=lambda host, port: False)
        published = next(
            c for c in checks if c.name == "dashboards published (docker/.env)"
        )
        self.assertEqual(published.status, "WARN")
        self.assertIn(
            "; the settings want: RKMON_ADMIN_HOST=127.0.0.1 RKMON_PORT=18620 RKMON_TEAM_HOST=0.0.0.0 "
            "RKMON_TEAM_PORT=18625",
            published.found,
        )
        self.assertEqual(
            str(published.fix),
            "put the same values in your settings file, then rkmon confirm and rkmon apply",
        )
        env.update(RKMON_PORT="18620", RKMON_TEAM_PORT="18625")
        self.assertEqual(webserve.publish_mismatch(cfg, env), [])

    def test_in_prototype_the_team_page_is_on_this_host_only(self):
        """In prototype docker/.env publishes the team dashboard's port on this host's own address,
        whatever publish_host says (features.team_host), and the doctor wants the same: its "dashboards published" row
        passes, and the port's row says the page is on this host only. Live, the address is publish_host's.
        """
        for mode, host, reach in (
            (
                "prototype",
                "127.0.0.1",
                "its page is on this host only, at http://127.0.0.1:18625/",
            ),
            ("live", "192.0.2.7", "other teams open http://192.0.2.7:18625/"),
        ):
            with self.subTest(mode=mode):
                cfg = dict(
                    self.cfg(public_port=18620),
                    features={"dashboards": {"team": {"enabled": True, "mode": mode}}},
                    team_dashboard={
                        "mode": "serve",
                        "publish_host": "192.0.2.7",
                        "public_port": 18625,
                    },
                )
                env = dict(
                    IMAGE_ENV,
                    RKMON_ADMIN_HOST="127.0.0.1",
                    RKMON_PORT="18620",
                    RKMON_TEAM_HOST=host,
                    RKMON_TEAM_PORT="18625",
                )
                checks = doctor.web_checks(cfg, env, answers=lambda h, p: False)
                rows = {c.name: c for c in checks}
                published = rows["dashboards published (docker/.env)"]
                self.assertEqual(
                    (published.status, published.found),
                    (
                        "PASS",
                        f"RKMON_ADMIN_HOST=127.0.0.1 RKMON_PORT=18620 RKMON_TEAM_HOST={host} "
                        "RKMON_TEAM_PORT=18625",
                    ),
                )
                port = rows["team dashboard's port (served here)"]
                self.assertTrue(port.found.endswith(f"; {reach}"), port.found)


class TestDoctor(Base):
    def test_the_checks(self):
        names = lambda checks: {
            c.name: (c.status, c.found) for c in checks
        }  # noqa: E731
        # On the loopback too, the service does not start without the access token. The fix its refusal names (the
        # access token's own: secrets_registry.set_fix) is on a line of its own, not in the row's sentence.
        first = doctor.web_checks(self.cfg(), HOST_ENV, answers=lambda h, p: False)[0]
        fix = "rkmon secret admin_access_token --generate"
        why = webserve.access_refusal(self.cfg(), HOST_ENV)
        self.assertTrue(why.endswith(f": run {fix}"), why)
        self.assertEqual(
            (first.status, first.found, str(first.fix)),
            (
                "FAIL",
                "http://127.0.0.1:18613/; this machine only (the loopback), access token missing; "
                + doctor.blocking(
                    why.removesuffix(f": run {fix}"),
                    "the admin dashboard's service: it does not start",
                ),
                fix,
            ),
        )
        self.set_access()
        got = names(doctor.web_checks(self.cfg(), HOST_ENV, answers=lambda h, p: False))
        self.assertEqual(
            got["admin dashboard"],
            (
                "PASS",
                "http://127.0.0.1:18613/; this machine only (the loopback), access token set",
            ),
        )
        self.assertEqual(got["admin dashboard's port"][0], "PASS")
        got = names(
            doctor.web_checks(self.cfg(), HOST_ENV, answers=lambda h, p: True)
        )  # the service is not running
        self.assertEqual(got["admin dashboard's port"][0], "WARN")
        self.assertIn("held by another program", got["admin dashboard's port"][1])
        self.access_file.unlink()
        checks = doctor.web_checks(
            self.cfg(bind="192.0.2.7"), HOST_ENV, answers=lambda h, p: False
        )
        first = checks[0]
        self.assertEqual(first.status, "FAIL")
        self.assertIn(
            "; it blocks the admin dashboard's service: it does not start", first.found
        )
        self.assertIn(
            "reachable beyond this machine on 192.0.2.7, access token missing",
            first.found,
        )

    def test_the_fix_for_the_tokens_file_or_the_address_is_on_a_line_of_its_own(self):
        """The fix the service's refusal names is on the row's own line, never in its sentence: the fix of each private
        file rule the access token's file breaks, in order, or, for a service on every address, the edit that sets the
        address a browser opens, never the access token's own fix while the token is set.
        """
        from rkmon_lib import paths, private_files
        from rkmon_lib.checks import Check, Command, Edit, Steps

        def row(cfg):
            return doctor.web_checks(cfg, HOST_ENV, answers=lambda h, p: False)[0]

        target = self.d / "elsewhere"
        target.write_text(ACCESS + "\n", encoding="utf-8")
        self.access_file.symlink_to(target)
        (rule,) = [
            r
            for r in private_files.problems(self.access_file, paths.PACKAGE)
            if r.status == "FAIL"
        ]
        self.assertIn(
            f"{rule.found} (fix: {rule.fix})",
            webserve.access_refusal(self.cfg(), HOST_ENV),
        )
        first = row(self.cfg())
        self.assertEqual((first.status, first.fix), ("FAIL", rule.fix))
        self.assertIn(f"{rule.found}; it blocks", first.found)
        self.assertNotIn("fix:", first.found)
        rules = [
            Check("private file", "FAIL", "one rule", Command(("chmod", "600", "x"))),
            Check(
                "private file",
                "FAIL",
                "another",
                Steps((Command(("mv", "x", "y")), ", then name it by its new path")),
            ),
        ]
        with mock.patch.object(webserve, "_token", lambda cfg: (None, rules)):
            first = row(self.cfg())
        self.assertEqual(
            str(first.fix), "chmod 600 x, then mv x y, then name it by its new path"
        )
        self.assertTrue(
            first.found.endswith(
                "one rule; another; it blocks the admin dashboard's service: it does not start"
            )
        )
        self.access_file.unlink()
        self.set_access()
        first = row(self.cfg(bind="0.0.0.0"))
        self.assertEqual(first.status, "FAIL")
        self.assertIsInstance(first.fix, Edit)
        self.assertEqual(
            (first.fix.key, first.fix.file),
            ("admin_dashboard.url", str(config.config_path())),
        )


class TestSummary(Base):
    def test_the_summary(self):
        self.set_access()
        cfg = config.normalize(
            dict(
                self.cfg(bind="192.0.2.7"),
                features={"dashboards": {"team": {"enabled": True}}},
                team_dashboard={"mode": "serve", "bind": "0.0.0.0", "port": 18615},
            )
        )
        with mock.patch.dict(os.environ, HOST_ENV):
            page = summary.render(cfg)
        self.assertIn(
            "| Admin dashboard reachable from | beyond this machine, on 192.0.2.7; every request needs the "
            "access token (set) |",
            page,
        )
        self.assertIn(
            "| Team dashboard (<TEAM_DASHBOARD_URL>) | served here at http://127.0.0.1:18615/ (read-only, with "
            "its request box) |",
            page,
        )
        self.assertIn(
            f"| admin dashboard access token | set ({self.access_file}) |",
            page,
        )
        # On the loopback, the same token.
        self.access_file.unlink()
        with mock.patch.dict(os.environ, HOST_ENV):
            page = summary.render(config.normalize(self.cfg()))
        self.assertIn(
            "| Admin dashboard reachable from | this machine only (the loopback); every request needs the access "
            "token (MISSING: the service does not start) |",
            page,
        )
        self.assertIn(
            f"| admin dashboard access token | missing ({self.access_file}) |", page
        )


if __name__ == "__main__":
    unittest.main()
