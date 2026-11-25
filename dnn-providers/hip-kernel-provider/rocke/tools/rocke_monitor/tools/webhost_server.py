#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The rocKE monitor's web host server: other teams may request priority from the team dashboard; their clicks
are requests that wait for the monitor owner's OK.

It replaces `python3 -m http.server` in rocke-monitor.service on the web host: it serves the same directory the
same way (http.server's SimpleHTTPRequestHandler), and accepts one write, a priority request from the team dashboard,
which it appends as one JSON line to a requests file outside the served directory. The monitor
copies that file each tick over its publish key (rkmon_lib/publish.pull_requests) and keeps the requests pending until
the user approves or declines them. Nothing else changes: no other write, no other state.

    POST <the page's directory>/priority-request      (a relative URL from the page: same origin, no CORS)
         Content-Type: application/json, X-Rkmon: 1; an Origin, when the browser sends one, must be the page's
         {"pr": 101, "who": "Jane Doe", "why": "blocks the 7.2 branch"}
    -> 200 {"ok": true, "message": "...", "id": "..."}; 4xx/5xx {"ok": false, "message": "..."}

Limits: a body of at most 2 KB; the PR digits only (1 to 7 digits); `who` 1 to 40 and `why` 1 to 300 characters on one
printable line; 5 requests an hour from one address and 100 a day in all; the requests file stops growing at 1 MB.

Where the monitor serves the page itself (rkmon_lib/webserve.team_server), it gives the server a notice
(Server.notice), asked as each request comes: while the notice says something, every request is answered with it
alone, 503, as a short page (NOTICE_PAGE), or to a POST as the refusal's message; nothing is served and no request is
taken in. The web host's server has none.

Standard library only, Python 3.6 or later. It runs as tools/rocke-monitor.service on the web host.

Usage: webhost_server.py --directory DIR --requests FILE [--port 8080] [--bind 0.0.0.0] [--origin URL ...]
"""

import argparse
import collections
import functools
import html
import inspect
import json
import os
import re
import secrets
import socketserver
import sys
import threading
import time
from http import server

MAX_BODY = 2048
WHO_MAX, WHY_MAX = 40, 300
PER_ADDR_HOUR, PER_DAY = 5, 100
MAX_FILE = 1024 * 1024
ROUTE = "/priority-request"
DIGITS = re.compile(r"\d{1,7}")
# The page a notice answers with (Server.notice): its words alone, escaped.
NOTICE_PAGE = (
    "<!doctype html><html lang='en'><head><meta charset='utf-8'><title>rocKE monitor</title></head>"
    "<body><p>%s</p></body></html>"
)

ThreadingHTTPServer = getattr(server, "ThreadingHTTPServer", None)
if ThreadingHTTPServer is None:  # Python 3.6

    class ThreadingHTTPServer(socketserver.ThreadingMixIn, server.HTTPServer):
        daemon_threads = True


class Limits:
    """Accepted requests in memory: per client address in the last hour, and all of them in the last day."""

    def __init__(self, per_addr=PER_ADDR_HOUR, per_day=PER_DAY, clock=time.time):
        self.per_addr, self.per_day, self.clock = per_addr, per_day, clock
        self.addr = collections.defaultdict(collections.deque)
        self.day = collections.deque()
        self.lock = threading.Lock()

    def take(self, addr):
        """None when one more request from `addr` may go in (it is counted), otherwise why not."""
        with self.lock:
            t = self.clock()
            for q in list(self.addr.values()) + [self.day]:
                span = 86400 if q is self.day else 3600
                while q and t - q[0] >= span:
                    q.popleft()
            for a in [a for a, q in self.addr.items() if not q]:
                del self.addr[a]
            if len(self.addr[addr]) >= self.per_addr:
                return "too many requests from your address: try again within the hour"
            if len(self.day) >= self.per_day:
                return "too many requests today: try again tomorrow"
            self.addr[addr].append(t)
            self.day.append(t)
            return None


def line_text(v, most):
    """One line of printable text, 1 to `most` characters; "" when it is not."""
    if not isinstance(v, str):
        return ""
    s = "".join(ch for ch in " ".join(v.split()) if ch.isprintable())
    return s if 0 < len(s) <= most else ""


def parse(body):
    """A request body: ({"pr", "who", "why"}, None), or (None, why it is refused)."""
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return None, 'send JSON: {"pr": N, "who": ..., "why": ...}'
    if not isinstance(data, dict):
        return None, "send a JSON object"
    pr = data.get("pr")
    if (
        isinstance(pr, bool)
        or not isinstance(pr, (int, str))
        or not DIGITS.fullmatch(str(pr))
        or int(pr) == 0
    ):
        return None, "the PR number: digits only"
    who, why = line_text(data.get("who"), WHO_MAX), line_text(data.get("why"), WHY_MAX)
    if not who:
        return None, "your name: 1 to %d characters on one line" % WHO_MAX
    if not why:
        return None, "why: 1 to %d characters on one line" % WHY_MAX
    return {"pr": int(pr), "who": who, "why": why}, None


class Handler(server.SimpleHTTPRequestHandler):
    """http.server's static files, plus POST .../priority-request; while the server has a notice, the notice alone."""

    server_version = "rkmon-webhost"

    def parse_request(self):
        """http.server's reading of the request line and headers; then, while the server's notice says something
        (Server.notice, asked now), that as the answer, whatever the method and path (send_notice): False, as for a
        request http.server answers with an error, so nothing is served or taken in."""
        if not server.SimpleHTTPRequestHandler.parse_request(self):
            return False
        words = self.server.notice() if self.server.notice else ""
        if not words:
            return True
        self.send_notice(words)
        return False

    def send_notice(self, words):
        """The answer while the server has a notice: 503 with WORDS, as the page (NOTICE_PAGE; no body for a HEAD),
        or to a POST as the refusal's message, which the page's script shows. A body the request carries is not read,
        so the connection closes."""
        self.close_connection = True
        if self.command == "POST":
            return self.reply(503, words)
        body = (NOTICE_PAGE % html.escape(words)).encode("utf-8")
        self.send_response(503)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def reply(self, status, message, **extra):
        body = json.dumps(
            dict({"ok": status == 200, "message": message}, **extra)
        ).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        srv = self.server
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if not 0 < length <= MAX_BODY:  # refused before reading it
            self.close_connection = True
            return self.reply(
                413 if length > MAX_BODY else 400,
                "send a body of at most %d bytes" % MAX_BODY,
            )
        body = self.rfile.read(length)
        if not self.path.split("?", 1)[0].endswith(ROUTE):
            return self.reply(404, "not found")
        if (self.headers.get("Content-Type") or "").split(";")[
            0
        ].strip().lower() != "application/json":
            return self.reply(415, "send JSON")
        origin = self.headers.get("Origin")
        if self.headers.get("X-Rkmon") != "1" or (
            origin is not None and origin not in srv.origins_for(self.headers)
        ):
            return self.reply(
                403, "only the dashboard page can send a priority request"
            )
        req, why = parse(body)
        if why:
            return self.reply(400, why)
        why = srv.limits.take(self.client_address[0])
        if why:
            return self.reply(429, why)
        rec = dict(
            id=secrets.token_hex(6),
            at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            addr=self.client_address[0],
            **req
        )
        why = srv.append(rec)
        if why:
            return self.reply(503, why)
        self.log_message(
            "priority request #%d by %s (%s)", rec["pr"], rec["who"], rec["id"]
        )
        return self.reply(
            200, "Requested: the monitor's owner will review it.", id=rec["id"]
        )


class Server(ThreadingHTTPServer):
    daemon_threads = True
    # The notice (see the module's docstring): None, or a function, asked as each request comes, that gives "" to serve
    # as usual, else the words every request is answered with (Handler.send_notice).
    notice = None

    def __init__(
        self, address, handler, requests, origins=(), limits=None, max_file=MAX_FILE
    ):
        self.requests, self.origins = requests, set(origins)
        self.limits, self.max_file = limits or Limits(), max_file
        self.write_lock = threading.Lock()
        ThreadingHTTPServer.__init__(self, address, handler)

    def origins_for(self, headers):
        """The page's origins: the configured ones, or, with none configured, this request's own host."""
        if self.origins:
            return self.origins
        host = headers.get("Host") or ""
        return {"http://" + host, "https://" + host}

    def append(self, rec):
        """Append one request; None when written, otherwise why not (the file is full)."""
        line = json.dumps(rec, ensure_ascii=False) + "\n"
        with self.write_lock:
            size = (
                os.path.getsize(self.requests) if os.path.exists(self.requests) else 0
            )
            if size + len(line.encode("utf-8")) > self.max_file:
                return "the request list is full: tell the monitor's owner"
            with open(self.requests, "a", encoding="utf-8") as f:
                f.write(line)
        return None


def make_server(
    directory,
    requests,
    port,
    bind="",  # "": every address
    origins=(),
    limits=None,
    max_file=MAX_FILE,
):
    """The server, not started. The requests file must lie outside the served directory: the site is public."""
    directory, requests = os.path.realpath(directory), os.path.realpath(requests)
    if os.path.commonpath([directory, requests]) == directory:
        raise ValueError("the requests file must not be inside the served directory")
    os.makedirs(os.path.dirname(requests), exist_ok=True)
    if (
        "directory"
        in inspect.signature(server.SimpleHTTPRequestHandler.__init__).parameters
    ):
        handler = functools.partial(Handler, directory=directory)
    else:  # Python 3.6: serve the working directory
        os.chdir(directory)
        handler = Handler
    return Server((bind, port), handler, requests, origins, limits, max_file)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--directory",
        required=True,
        help="the served site (as http.server --directory)",
    )
    ap.add_argument(
        "--requests",
        required=True,
        help="the requests file, outside the served directory",
    )
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument(
        "--bind", default="", help="the address to listen on (default: every address)"
    )
    ap.add_argument(
        "--origin",
        action="append",
        default=[],
        help="the page's public origin, e.g. "
        "http://web-host.example:8080 (repeatable)",
    )
    a = ap.parse_args(argv)
    try:
        httpd = make_server(a.directory, a.requests, a.port, a.bind, a.origin)
    except (ValueError, OSError) as exc:
        sys.exit("webhost_server: %s" % exc)
    sys.stderr.write(
        "webhost_server: serving %s on %s:%d; priority requests to %s; origins %s\n"
        % (
            a.directory,
            a.bind or "every address",
            a.port,
            a.requests,
            ", ".join(a.origin) or "the request's host",
        )
    )
    sys.stderr.flush()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
