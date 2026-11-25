# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The web host server (tools/webhost_server.py: other teams may request priority, and their clicks wait for the
operator's OK): it still serves the site as http.server did, and takes a priority request only
from the page, well-formed and within its size and rate limits, into a file outside the served directory, in the form
the monitor reads back."""

import http.client
import importlib.util
import json
import pathlib
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import priority  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "webhost_server", ROOT / "tools" / "webhost_server.py"
)
wh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wh)

ORIGIN = "http://web-host.example:8080"
PATH = "/rocke-monitor/priority-request"


class Clock:
    def __init__(self):
        self.t = 1_000_000.0

    def __call__(self) -> float:
        return self.t


class TestWebhost(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = pathlib.Path(self.tmp.name)
        (d / "website/rocke-monitor").mkdir(parents=True)
        (d / "website/rocke-monitor/index.html").write_text(
            "<!doctype html><title>dash</title>"
        )
        self.site, self.requests = (
            d / "website",
            d / "requests" / "priority_requests.jsonl",
        )
        self.clock = Clock()
        self.srv = wh.make_server(
            str(self.site),
            str(self.requests),
            0,
            "127.0.0.1",
            [ORIGIN],
            wh.Limits(clock=self.clock),
        )
        self.port = self.srv.server_address[1]
        self.thread = threading.Thread(target=self.srv.serve_forever, daemon=True)
        self.thread.start()
        self.quiet = mock.patch.object(sys, "stderr")  # the request log
        self.quiet.start()

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        self.thread.join(5)
        self.quiet.stop()
        self.tmp.cleanup()

    def call(self, method: str, path: str, body=None, **change) -> tuple[int, bytes]:
        h = {"Content-Type": "application/json", "X-Rkmon": "1", "Origin": ORIGIN}
        for k, v in change.items():
            k = k.replace("_", "-")
            h.pop(k, None) if v is None else h.__setitem__(k, v)
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            c.request(
                method,
                path,
                body=json.dumps(body) if isinstance(body, dict) else body,
                headers=h,
            )
            r = c.getresponse()
            return r.status, r.read()
        finally:
            c.close()

    def ask(self, body, **change) -> tuple[int, dict]:
        status, data = self.call("POST", PATH, body, **change)
        return status, json.loads(data)

    def lines(self) -> list[dict]:
        return (
            [json.loads(x) for x in self.requests.read_text().splitlines()]
            if self.requests.exists()
            else []
        )

    def test_the_site_is_served_as_before(self):
        for path in ("/rocke-monitor/", "/rocke-monitor/index.html"):
            status, data = self.call("GET", path)
            self.assertEqual(
                (status, data), (200, b"<!doctype html><title>dash</title>")
            )
        self.assertEqual(
            self.call("GET", PATH)[0], 404
        )  # no such file: the route takes POSTs only
        self.assertEqual(
            self.call("GET", "/../requests/priority_requests.jsonl")[0], 404
        )

    def test_a_request_is_appended_beside_the_site_in_the_form_the_monitor_reads(self):
        status, out = self.ask(
            {"pr": 10101, "who": "  Jane   Doe ", "why": "blocks\nthe 7.2 branch"}
        )
        self.assertEqual(
            (status, out["ok"], out["message"]),
            (200, True, "Requested: the monitor's owner will review it."),
        )
        self.assertEqual(
            self.ask({"pr": "10102", "who": "Bob", "why": "x"})[0], 200
        )  # digits as text too
        rec, rec2 = self.lines()
        self.assertEqual(
            {k: rec[k] for k in ("pr", "who", "why", "addr")},
            {
                "pr": 10101,
                "who": "Jane Doe",
                "why": "blocks the 7.2 branch",
                "addr": "127.0.0.1",
            },
        )
        self.assertEqual((rec["id"], rec2["pr"]), (out["id"], 10102))
        self.assertNotIn(str(self.site), str(self.requests.resolve()))
        # The monitor takes it as it is.
        self.assertEqual(
            priority.clean(rec), {k: rec[k] for k in ("id", "pr", "who", "why", "at")}
        )

    def test_only_the_page_may_post(self):
        for change, status in (
            ({"X_Rkmon": None}, 403),
            ({"Origin": "http://evil.example"}, 403),
            ({"Origin": "null"}, 403),
            ({"Content_Type": "text/plain"}, 415),
            ({"Content_Type": "application/x-www-form-urlencoded"}, 415),
        ):
            self.assertEqual(
                self.ask({"pr": 1, "who": "a", "why": "b"}, **change)[0], status, change
            )
        self.assertEqual(self.lines(), [])
        self.assertEqual(
            self.ask({"pr": 1, "who": "a", "why": "b"}, Origin=None)[0], 200
        )  # not a browser: no Origin
        self.assertEqual(
            self.call("POST", "/rocke-monitor/index.html", {"pr": 1})[0], 404
        )
        self.assertEqual(self.call("PUT", PATH)[0], 501)

    def test_every_field_is_checked(self):
        ok = {"pr": 7, "who": "Jane", "why": "blocks the release"}
        for pr in (
            "12a",
            "-1",
            -1,
            12345678,
            "12345678",
            True,
            0,
            "0",
            "",
            1.5,
            None,
            "#12",
            " 12",
        ):
            status, out = self.ask(dict(ok, pr=pr))
            self.assertEqual(
                (status, out["message"]), (400, "the PR number: digits only"), pr
            )
        for who in ("", "   ", "\x00\x01", "w" * 41, 7, None):
            self.assertEqual(self.ask(dict(ok, who=who))[0], 400, who)
        for why in ("", "y" * 301, ["x"]):
            self.assertEqual(self.ask(dict(ok, why=why))[0], 400, why)
        for body in ("not json", "[7]", "\xff"):
            self.assertEqual(
                self.call("POST", PATH, body.encode("latin-1"))[0], 400, body
            )
        self.assertEqual(self.lines(), [])
        self.assertEqual(
            self.ask(dict(ok, who="w" * 40, why="y" * 300))[0], 200
        )  # the limits themselves pass

    def test_size_and_rate_limits(self):
        c = http.client.HTTPConnection(
            "127.0.0.1", self.port, timeout=10
        )  # refused before the body is read
        c.putrequest("POST", PATH)
        for k, v in {
            "Content-Type": "application/json",
            "X-Rkmon": "1",
            "Content-Length": "5000",
        }.items():
            c.putheader(k, v)
        c.endheaders()
        self.assertEqual(c.getresponse().status, 413)
        c.close()
        ok = {"pr": 7, "who": "Jane", "why": "blocks the release"}
        self.assertEqual(
            [self.ask(ok)[0] for _ in range(6)], [200] * 5 + [429]
        )  # 5 an hour from one address
        self.assertEqual(
            self.ask(ok)[1]["message"],
            "too many requests from your address: try again within the hour",
        )
        self.clock.t += 3600
        self.assertEqual(self.ask(ok)[0], 200)
        self.assertEqual(len(self.lines()), 6)
        limits = wh.Limits(
            per_addr=100, per_day=3, clock=self.clock
        )  # and 100 a day in all
        self.assertEqual(
            [limits.take(f"10.0.0.{i}") for i in range(4)][-1],
            "too many requests today: try again tomorrow",
        )
        self.clock.t += 86400
        self.assertIsNone(limits.take("10.0.0.9"))

    def test_the_file_stops_growing(self):
        self.srv.max_file = 400
        ok = {"pr": 7, "who": "Jane", "why": "y" * 200}
        self.assertEqual(self.ask(ok)[0], 200)
        status, out = self.ask(ok)
        self.assertEqual(
            (status, out["message"]),
            (503, "the request list is full: tell the monitor's owner"),
        )
        self.assertEqual(len(self.lines()), 1)

    def test_the_requests_file_cannot_be_inside_the_site(self):
        with self.assertRaises(ValueError):
            wh.make_server(
                str(self.site),
                str(self.site / "rocke-monitor" / "requests.jsonl"),
                0,
                "127.0.0.1",
            )
        with self.assertRaises(SystemExit):
            wh.main(
                [
                    "--directory",
                    str(self.site),
                    "--requests",
                    str(self.site / "r.jsonl"),
                    "--port",
                    "0",
                ]
            )

    def test_with_no_address_given_it_listens_on_every_address(self):
        srv = wh.make_server(str(self.site), str(self.requests), 0)
        try:
            self.assertEqual(srv.server_address[0], "0.0.0.0")
        finally:
            srv.server_close()

    def test_while_its_notice_says_something_every_request_gets_that_alone(self):
        """The notice the monitor gives the server where it serves the page itself (Server.notice; the web host's has
        none) is asked as each request comes: while it says something, every request, whatever its method and path,
        is answered 503 with its words alone, escaped, as the page, or to a POST as the refusal's message, and
        nothing is taken in; once it says nothing, the same server serves the site and takes requests again.
        """
        self.assertIsNone(wh.Server.notice)
        words = []
        self.srv.notice = lambda: words[-1] if words else ""
        site = (200, b"<!doctype html><title>dash</title>")
        self.assertEqual(self.call("GET", "/rocke-monitor/"), site)
        words.append("In standby: <b>back</b> soon & then some.")
        page = (
            "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>rocKE monitor</title></head>"
            "<body><p>In standby: &lt;b&gt;back&lt;/b&gt; soon &amp; then some.</p></body></html>"
        ).encode("utf-8")
        for path in ("/rocke-monitor/", "/rocke-monitor/index.html", "/x"):
            with self.subTest(path=path):
                self.assertEqual(self.call("GET", path), (503, page))
        self.assertEqual(self.call("PUT", PATH, {"pr": 1}), (503, page))
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=10)
        try:
            c.request("HEAD", "/rocke-monitor/")
            r = c.getresponse()
            self.assertEqual((r.status, r.read()), (503, b""))
            self.assertEqual(
                [r.getheader(h) for h in ("Content-Type", "Content-Length")],
                ["text/html; charset=utf-8", str(len(page))],
            )
            self.assertEqual(r.getheader("Cache-Control"), "no-store")
        finally:
            c.close()
        asked = {"pr": 10101, "who": "Jane", "why": "blocks"}
        self.assertEqual(self.ask(asked), (503, {"ok": False, "message": words[-1]}))
        self.assertEqual(self.lines(), [])
        words.clear()
        self.assertEqual(self.call("GET", "/rocke-monitor/"), site)
        self.assertEqual(self.ask(asked)[0], 200)
        self.assertEqual([rec["pr"] for rec in self.lines()], [10101])


if __name__ == "__main__":
    unittest.main()
