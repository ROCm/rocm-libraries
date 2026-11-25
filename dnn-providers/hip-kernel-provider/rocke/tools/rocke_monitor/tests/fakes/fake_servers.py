#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Fake GitHub, release, model and JIRA servers on 127.0.0.1, for the package's offline tests and the container smoke
test. Nothing here reaches the network.

- GitHub: the REST and GraphQL reads an observe-phase tick makes, for a repository with no open PR in scope (or the
  open PRs `prs` describes: the search, each PR's REST view and formal reviews, `gh pr view --json` and `gh pr diff`,
  no review threads), no open issue, and a develop whose CI has not run. The fake gh (fake_gh) sends its `api` calls
  here, with any --jq expression in X-Jq; answers are already in the shape that expression gives. Every request is
  logged to <state>/github.log.
- Release: the GitHub CLI's release layout (download/v<V>/gh_<V>_linux_<arch>.tar.gz, gh_<V>_checksums.txt, and a
  latest-release JSON), with the fake gh as the binary, for a gh install from it (deploy.gh_install).
- Model: Ollama's API (/api/tags, /api/show, /api/chat) and the OpenAI-compatible one (/v1/models,
  /v1/chat/completions), with one model that answers the model checks: a streamed answer, with a read_file tool
  call when the request offers tools (the tool-call probe, rkmon_lib/models.py).
- JIRA: /rest/api/2/serverInfo, the read-only check rk doctor makes (rkmon_lib/reach.py).

    python3 fake_servers.py --state-dir DIR [--model NAME] [--ports G,R,M,J]

writes DIR/servers.json with the four base URLs and the gh version, and serves until it is stopped. `start()` does the
same in a test's own process.
"""

import argparse
import hashlib
import http.server
import io
import json
import pathlib
import platform
import re
import tarfile
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
GH_RELEASE = "2.46.0"
DEVELOP_SHA = "0" * 40
NOTHING = object()  # a --jq answer that prints nothing
# A search's words that ask for what the fake GitHub has none of: issues, merged PRs and closed ones
NOT_SERVED = {"is:issue", "is:merged", "is:closed", "is:unmerged"}


def _arch() -> str:
    return {
        "x86_64": "amd64",
        "amd64": "amd64",
        "aarch64": "arm64",
        "arm64": "arm64",
    }.get(platform.machine().lower(), "amd64")


def release_files(version: str = GH_RELEASE) -> dict[str, bytes]:
    """Path -> body of a gh release with the fake gh inside (and a good checksum)."""
    name = f"gh_{version}_linux_{_arch()}"
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for member, data, mode in (
            (f"{name}/bin/gh", (HERE / "fake_gh").read_bytes(), 0o755),
            (f"{name}/LICENSE", b"fake\n", 0o644),
        ):
            info = tarfile.TarInfo(member)
            info.size, info.mode, info.mtime = len(data), mode, 0
            t.addfile(info, io.BytesIO(data))
    data = buf.getvalue()
    sums = f"{hashlib.sha256(data).hexdigest()}  {name}.tar.gz\n".encode()
    return {
        f"/cli/cli/releases/download/v{version}/{name}.tar.gz": data,
        f"/cli/cli/releases/download/v{version}/gh_{version}_checksums.txt": sums,
        "/latest": json.dumps({"tag_name": f"v{version}"}).encode(),
    }


class _Base(http.server.BaseHTTPRequestHandler):
    log_path: pathlib.Path | None = None

    def log_message(self, *a):
        pass

    def answer(self, status: int, body, ctype: str = "application/json") -> None:
        data = (
            body
            if isinstance(body, bytes)
            else (body if isinstance(body, str) else json.dumps(body)).encode()
        )
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def stream(self, chunks, fmt, ctype: str, tail: bytes = b"") -> None:
        """A streamed answer: each chunk formatted by `fmt`, then `tail`."""
        data = "".join(fmt(c) for c in chunks).encode() + tail
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return {}

    def record(self, **extra) -> None:
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps({"method": self.command, "path": self.path, **extra})
                    + "\n"
                )


class GitHub(_Base):
    """The reads an observe tick makes, for a repository with nothing in scope, or with the open PRs of `prs`: each a
    dict with number, title, author, labels, head (its SHA), head_ref, base_sha, committed_at, additions, deletions,
    files ([{path, additions, deletions}]) and diff, as tests/fakerepo.py pull_request makes them, and reviews, its
    formal reviews in GitHub's REST form (none when it has no such key). Its PRs are open: a search for issues, or for
    merged or closed PRs (NOT_SERVED), finds none.
    """

    prs: list = []

    def pr(self, n: str) -> dict | None:
        return next((p for p in self.prs if str(p["number"]) == str(n)), None)

    def do_GET(self):
        jq = self.headers.get("X-Jq") or ""
        path = self.path.split("?", 1)[0].strip("/")
        self.record(jq=jq)
        m = re.fullmatch(
            r"_fake/pr/(\d+)/(view|diff)", path
        )  # fake_gh's `pr view --json` and `pr diff`
        if m:
            p = self.pr(m.group(1))
            if p is None:
                self.answer(404, {"message": "Not Found"})
            elif m.group(2) == "diff":
                self.answer(200, p["diff"], "text/plain")
            else:
                self.answer(200, pr_view(p))
            return
        m = re.fullmatch(r"repos/[^/]+/[^/]+/pulls/(\d+)", path)
        if m:
            p = self.pr(m.group(1))
            if p is None:
                self.answer(404, {"message": "Not Found"})
            else:
                self.answer(200, shape(pr_rest(p), jq))
            return
        m = re.fullmatch(r"repos/[^/]+/[^/]+/pulls/(\d+)/reviews", path)
        if m:
            body = (self.pr(m.group(1)) or {}).get("reviews") or []
        elif re.fullmatch(r"repos/[^/]+/[^/]+/commits/[^/]+/check-runs", path):
            body = {"total_count": 0, "check_runs": []}
        elif re.fullmatch(r"repos/[^/]+/[^/]+/commits/[^/]+/status", path):
            body = {"state": "pending", "statuses": []}
        elif re.fullmatch(r"repos/[^/]+/[^/]+/commits/[^/]+", path):
            body = {"sha": DEVELOP_SHA, "commit": {"message": "develop"}}
        elif re.fullmatch(r"repos/[^/]+/[^/]+/actions/runs", path):
            body = {"total_count": 0, "workflow_runs": []}
        elif re.fullmatch(r"repos/[^/]+/[^/]+/rules/branches/[^/]+", path):
            body = []
        elif re.fullmatch(
            r"repos/[^/]+/[^/]+/(issues|pulls)(/\d+/(comments|reviews|timeline|commits))?",
            path,
        ):
            body = []
        elif re.fullmatch(r"orgs/[^/]+/teams/[^/]+/members", path):
            body = []
        elif re.fullmatch(r"repos/[^/]+/[^/]+", path):
            body = {
                "full_name": "/".join(path.split("/")[1:3]),
                "default_branch": "develop",
                "private": False,
            }
        elif path == "rate_limit":  # rk doctor's GitHub check (it costs no call)
            body = {
                "resources": {
                    "core": {
                        "limit": 5000,
                        "remaining": 4999,
                        "reset": 2000000000,
                        "used": 1,
                    }
                }
            }
        else:
            self.answer(404, {"message": "Not Found"})
            return
        out = shape(body, jq)
        self.answer(
            200,
            b"" if out is NOTHING else out,
            "text/plain" if out is NOTHING else "application/json",
        )

    def do_POST(self):
        if self.path.split("?", 1)[0].strip("/") != "graphql":
            self.record(refused=True)
            self.answer(405, {"message": "the fake GitHub takes no writes"})
            return
        req = self.body()
        q = req.get("query", "")
        self.record(query=q[:200])
        if "mutation" in q:
            self.answer(405, {"message": "the fake GitHub takes no writes"})
        elif "search(" in q:
            words = str((req.get("variables") or {}).get("q") or "").split()
            nodes = (
                [] if NOT_SERVED & set(words) else [search_node(p) for p in self.prs]
            )
            self.answer(
                200,
                {
                    "data": {
                        "search": {
                            "issueCount": len(nodes),
                            "pageInfo": {"hasNextPage": False, "endCursor": None},
                            "nodes": nodes,
                        }
                    }
                },
            )
        elif "reviewThreads" in q:  # no PR has review threads
            self.answer(
                200,
                {
                    "data": {
                        "repository": {
                            "pullRequest": {
                                "reviewThreads": {
                                    "pageInfo": {
                                        "hasNextPage": False,
                                        "endCursor": None,
                                    },
                                    "nodes": [],
                                }
                            }
                        }
                    }
                },
            )
        else:
            self.answer(200, {"data": {"repository": None}})


def search_node(p: dict) -> dict:
    """One PR as the monitor's search query (rkmon_lib/snapshot.py _SEARCH) reads it."""
    return {
        "number": p["number"],
        "title": p["title"],
        "isDraft": False,
        "createdAt": p["committed_at"],
        "updatedAt": p["committed_at"],
        "additions": p["additions"],
        "deletions": p["deletions"],
        "changedFiles": len(p["files"]),
        "author": {"login": p["author"]},
        "baseRefName": "develop",
        "headRefName": p["head_ref"],
        "headRefOid": p["head"],
        "mergeable": "MERGEABLE",
        "mergeStateStatus": "CLEAN",
        "reviewDecision": "REVIEW_REQUIRED",
        "autoMergeRequest": None,
        "reviewRequests": {"nodes": []},
        "closingIssuesReferences": {"nodes": []},
        "labels": {"nodes": [{"name": x} for x in p["labels"]]},
        "commits": {
            "nodes": [
                {
                    "commit": {
                        "oid": p["head"],
                        "committedDate": p["committed_at"],
                        "statusCheckRollup": None,
                    }
                }
            ]
        },
        "latestOpinionatedReviews": {"nodes": []},
    }


def pr_view(p: dict) -> dict:
    """`gh pr view N --json ...` for one PR: the fields the worker's fetch_pr.py asks for."""
    return {
        "number": p["number"],
        "title": p["title"],
        "body": p.get("body") or "A fake PR.",
        "author": {"login": p["author"]},
        "labels": [{"name": x} for x in p["labels"]],
        "state": "OPEN",
        "isDraft": False,
        "mergeable": "MERGEABLE",
        "headRefName": p["head_ref"],
        "headRefOid": p["head"],
        "baseRefName": "develop",
        "createdAt": p["committed_at"],
        "updatedAt": p["committed_at"],
        "additions": p["additions"],
        "deletions": p["deletions"],
        "commits": [
            {
                "oid": p["head"],
                "committedDate": p["committed_at"],
                "messageHeadline": p["title"],
            }
        ],
        "reviews": [],
        "comments": [],
        "files": p["files"],
    }


def pr_rest(p: dict) -> dict:
    """GET repos/O/R/pulls/N for one open PR."""
    return {
        "number": p["number"],
        "state": "open",
        "title": p["title"],
        "user": {"login": p["author"]},
        "merged_at": None,
        "merge_commit_sha": None,
        "merged_by": None,
        "labels": [{"name": x} for x in p["labels"]],
        "head": {"sha": p["head"], "ref": p["head_ref"]},
        "base": {"sha": p["base_sha"], "ref": "develop"},
    }


def _read(body, path: str):
    """`.a.b` read from body, None where a part is missing."""
    for part in path.strip(".").split("."):
        body = body.get(part) if isinstance(body, dict) else None
    return body


def shape(body, jq: str):
    """What gh's --jq would print for these bodies: lists stay empty, a field read on an object (`.a`, `.a.b`) is its
    value, and an object built of plain fields (`{state, head: .head.sha}`) is built; any other object is empty.
    """
    if not jq:
        return body
    jq = jq.strip()
    if jq.startswith("["):
        return []
    m = re.match(r"\.(\w+)\[\]", jq)
    if m and isinstance(body, dict) and body.get(m.group(1)) == []:
        return NOTHING  # each item of an empty list: gh prints nothing
    if re.fullmatch(r"(?:\.\w+)+", jq) and isinstance(body, dict):
        return _read(body, jq)
    if jq.startswith("{") and isinstance(body, dict):
        fields = [f.strip() for f in jq[1:-1].split(",")]
        built = {}
        for field in fields:
            m = re.fullmatch(r"(\w+)(?:\s*:\s*((?:\.\w+)+))?", field)
            if not m:
                return {}
            built[m.group(1)] = _read(body, m.group(2) or "." + m.group(1))
        return built
    return body


class Release(_Base):
    files: dict = {}

    def do_GET(self):
        self.record()
        body = self.files.get(self.path.split("?", 1)[0])
        if body is None:
            self.answer(404, {"message": "Not Found"})
        else:
            self.answer(200, body, "application/octet-stream")


class Model(_Base):
    model = "fake-model:1"

    def do_GET(self):
        self.record()
        if self.path == "/api/tags":
            self.answer(
                200,
                {
                    "models": [
                        {"name": self.model, "model": self.model, "size": 4 * 2**30}
                    ]
                },
            )
        elif self.path == "/v1/models":
            self.answer(
                200,
                {
                    "object": "list",
                    "data": [{"id": self.model, "max_model_len": 32768}],
                },
            )
        else:
            self.answer(404, {"error": "not found"})

    def do_POST(self):
        req = self.body()
        self.record(model=req.get("model"))
        if self.path == "/api/show":
            self.answer(200, {"model_info": {"fake.context_length": 32768}})
        elif self.path == "/api/chat" and req.get("stream"):
            call = (
                [
                    {
                        "function": {
                            "name": "read_file",
                            "arguments": {"path": "README.md"},
                        }
                    }
                ]
                if req.get("tools")
                else []
            )
            self.stream(
                [
                    {
                        "message": {
                            "role": "assistant",
                            "content": "" if call else "OK",
                            "tool_calls": call,
                        }
                    },
                    {"done": True},
                ],
                lambda c: json.dumps(c) + "\n",
                "application/x-ndjson",
            )
        elif self.path == "/api/chat":
            self.answer(
                200,
                {
                    "model": self.model,
                    "message": {"role": "assistant", "content": "OK"},
                    "done": True,
                },
            )
        elif self.path == "/v1/chat/completions" and req.get("stream"):
            delta = (
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": "c1",
                            "function": {
                                "name": "read_file",
                                "arguments": '{"path": "README.md"}',
                            },
                        }
                    ]
                }
                if req.get("tools")
                else {"content": "OK"}
            )
            self.stream(
                [{"choices": [{"index": 0, "delta": delta}]}],
                lambda c: f"data: {json.dumps(c)}\n\n",
                "text/event-stream",
                b"data: [DONE]\n\n",
            )
        elif self.path == "/v1/chat/completions":
            self.answer(
                200,
                {
                    "choices": [
                        {
                            "index": 0,
                            "message": {"role": "assistant", "content": "OK"},
                            "finish_reason": "stop",
                        }
                    ]
                },
            )
        else:
            self.answer(404, {"error": "not found"})


class Jira(_Base):
    def do_GET(self):
        self.record()
        if self.path.split("?", 1)[0] == "/rest/api/2/serverInfo":
            self.answer(200, {"version": "9.0.0", "serverTitle": "fake JIRA"})
        else:
            self.answer(404, {"errorMessages": ["not found"]})


def start(state_dir, model: str = "fake-model:1", ports=(0, 0, 0, 0), prs=None):
    """Start the four servers in this process: (their base URLs, a function that stops them). `prs`: the open PRs the
    fake GitHub has (GitHub's docstring), none by default."""
    state = pathlib.Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    servers, urls = [], {}
    for (name, handler), port in zip(
        (("github", GitHub), ("release", Release), ("model", Model), ("jira", Jira)),
        ports,
    ):
        cls = type(
            f"{handler.__name__}Handler",
            (handler,),
            {"log_path": state / f"{name}.log"},
        )
        if name == "github":
            cls.prs = list(prs or [])
        if name == "release":
            cls.files = release_files()
        if name == "model":
            cls.model = model
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), cls)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        servers.append(srv)
        urls[name] = f"http://127.0.0.1:{srv.server_address[1]}"
    urls["gh_version"] = GH_RELEASE
    urls["release_base"] = urls["release"] + "/cli/cli/releases"
    urls["release_latest"] = urls["release"] + "/latest"
    (state / "servers.json").write_text(json.dumps(urls, indent=1), encoding="utf-8")

    def stop():
        for s in servers:
            s.shutdown()
            s.server_close()

    return urls, stop


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--model", default="fake-model:1")
    ap.add_argument(
        "--ports",
        default="0,0,0,0",
        help="github,release,model,jira (0: any free port)",
    )
    a = ap.parse_args()
    ports = tuple(int(x) for x in a.ports.split(","))
    urls, _stop = start(a.state_dir, a.model, ports)
    print(json.dumps(urls), flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
