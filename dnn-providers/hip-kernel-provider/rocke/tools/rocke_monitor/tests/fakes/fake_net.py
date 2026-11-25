#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""GitHub and a Claude gateway at one HTTPS address, for the image test's deploy-path stage
(tests/container/stage_deploy_path.py). It runs in a container of the image under test on the deployment's network,
where docker gives it the names the deployment calls (network aliases): github.com, api.github.com and the gateway's
host. Its certificate names them all, and the image the deployment runs trusts the root certificate it makes. Nothing
here reaches the network.

- api.github.com: fake_servers.GitHub's reads (a repository with nothing in scope, or the open PRs CONFIG asks for),
  the base branch's head as the tick reads it (rkmon_lib/snapshot.py _BRANCH: develop's), and what gh's own login
  asks: the API's root (with no scopes header, as for a fine-grained token), /user and the viewer's login. Each call
  must carry a token the deployment logs in with, the read login's, or the write login's when CONFIG gives one: a
  call with another token, or none, is answered 401, as GitHub answers a token it does not know. Every write with the
  read login's token is refused as GitHub refuses a token that may only read (403), the login check's probes among
  them (PROBES): a comment with an empty body and a branch with no name and no commit, which GitHub never makes. The
  write login's probes are answered as GitHub answers a login that may write, which validates them and makes nothing
  (422); its comment on a pull request, as `gh pr comment` makes one, goes through: the pull request it looks up by
  its number, any number (its id and URL), then the comment, whose URL it answers. Any other write is refused (403).
- github.com: the repository's clone and fetches, through git's own http-backend, read-only: a push is refused.
- the gateway: the Anthropic Messages API's model list, messages (a short answer, streamed when asked) and token
  counts. Each call's key is checked against the one the stage gave, sent as the deploy settings say.
- any other name or path: 404.

Every request is logged to STATE/requests.log as a JSON line: its host, method, path and status, whether it was a
write and whether it was one of those probes, for api.github.com the token it carried (login for the read login's,
write for the write login's, other or none, never the token itself), and for the gateway the headers that carried the
key (their names only) and whether the key was right.

    python3 fake_net.py --config CONFIG --state STATE [--port 443]

CONFIG, a JSON file: {"github": {"login": the tokens' user, "token": the token the deployment's read login logs in
with, "write_token": the write login's (left out: no write login), "repository": "<org>/<repo>"}, "gateway": {"host":
its name, "key": the key, "header": the header that carries it
(blank: x-api-key), "scheme": the text before the key in it, "sent_as": "header", "bearer" or "header_and_bearer",
"models": the model ids it lists}, "prs": the open PRs, none when left out: each {"number", "lines", "at", "pushes",
"reviews"}, the PR fakerepo.pull_request opens with those LINES, AT (its commits' date) and PUSHES, and REVIEWS, its
formal reviews by the login's user: each {"push": the push it reviewed, from 1, "at": when it was submitted, "body"}}.
First it makes, in the folder STATE, empty or new: a root certificate (ca.crt) and the server's certificate for every
name it serves, and the repository (tests/fakerepo.py) with those PRs as git/<org>/<repo>.git. STATE/ready appears
once it serves.
"""

import argparse
import http.server
import json
import os
import pathlib
import re
import ssl
import subprocess
import sys
import threading
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))

import fake_servers  # noqa: E402

API, WEB, CODELOAD = "api.github.com", "github.com", "codeload.github.com"
# a GraphQL query of the viewer itself (requestedReviewer, in the search, is not one)
VIEWER = re.compile(r"(?<![A-Za-z])viewer\s*\{")
# and of the base branch's head (rkmon_lib/snapshot.py _BRANCH): the default branch's, or the one it names
BRANCH = re.compile(
    r"\{\s*(defaultBranchRef|ref)(?:\(qualifiedName:\s*\$r\))?\s*\{\s*name\s+target\s*\{\s*oid\s*\}"
)
# gh pr comment's look-up of the pull request it comments on (its variables' pr_number), and the comment it makes
PULL_REQUEST = re.compile(r"\bpullRequest\s*\(\s*number:")
ADD_COMMENT = re.compile(r"\baddComment\s*\(")
PR_ID = "PR_fake_"  # how a pull request's id begins, its number after it
# GitHub's answer to a write by a token that may only read
READ_ONLY = {
    "message": "Resource not accessible by personal access token",
    "documentation_url": "https://docs.github.com/rest",
    "status": "403",
}
# and to a comment with an empty body by a login that may write: it validates the body, and makes nothing
VALIDATION_FAILED = {
    "message": "Validation Failed",
    "errors": [{"resource": "IssueComment", "code": "missing_field", "field": "body"}],
    "documentation_url": "https://docs.github.com/rest",
    "status": "422",
}
# and to a branch with no name and no commit by a login that may push: it validates them, and makes nothing
REF_VALIDATION_FAILED = {
    "message": "Validation Failed",
    "documentation_url": "https://docs.github.com/rest",
    "status": "422",
}
# and to a call with a token it does not know, or none
BAD_CREDENTIALS = {
    "message": "Bad credentials",
    "documentation_url": "https://docs.github.com/rest",
    "status": "401",
}
# the login check's probes (rkmon_lib/gh.py probe), each by its path, the fields it leaves empty and the answer to a
# login that may make it
PROBES = (
    (
        re.compile(r"repos/[\w.-]+/[\w.-]+/issues/\d+/comments"),
        ("body",),
        VALIDATION_FAILED,
    ),
    (
        re.compile(r"repos/[\w.-]+/[\w.-]+/git/refs"),
        ("ref", "sha"),
        REF_VALIDATION_FAILED,
    ),
)
GIT_PATH = re.compile(
    r"/([\w.-]+)/([\w.-]+?)(?:\.git)?(/info/refs|/git-upload-pack|/git-receive-pack|/HEAD|/objects/.+)"
)
DAYS = "2"  # the certificates' life: the stage's run, with room


def names(config: dict) -> list:
    """Every name the fakes answer to: the network aliases the stage gives their container."""
    return [WEB, API, CODELOAD, config["gateway"]["host"]]


def probe_answer(path: str, body: dict) -> dict | None:
    """The answer to a login that may make it, when a POST of BODY to api.github.com's PATH (without its leading
    slash) is one of the login check's probes (PROBES); None when it is none of them."""
    for pattern, empty, answer in PROBES:
        if pattern.fullmatch(path) and all(body.get(k) == "" for k in empty):
            return answer
    return None


class Net(fake_servers.GitHub):
    config: dict = {}
    state: pathlib.Path = pathlib.Path(".")
    develop: str = fake_servers.DEVELOP_SHA
    lock = threading.Lock()

    def setup(self):
        super().setup()
        self.raw = None
        self.extra = {}

    # --- what every answer shares ------------------------------------------------------------------------------------
    def host(self) -> str:
        return (self.headers.get("Host") or "").split(":", 1)[0].lower()

    def record(self, **extra) -> None:
        self.extra.update(extra)

    def log(self, status: int) -> None:
        line = {
            "t": round(time.time(), 3),
            "host": self.host(),
            "method": self.command,
            "path": self.path,
            "status": status,
            **self.extra,
        }
        with self.lock, open(self.state / "requests.log", "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")

    def answer(self, status: int, body, ctype: str = "application/json") -> None:
        self.log(status)
        super().answer(status, body, ctype)

    def raw_body(self) -> bytes:
        """The request's body, read once, whether it came with a length or in chunks."""
        if self.raw is None:
            data = b""
            if "chunked" in (self.headers.get("Transfer-Encoding") or "").lower():
                while True:
                    size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                    if size == 0:
                        while self.rfile.readline() not in (b"\r\n", b"\n", b""):
                            pass
                        break
                    data += self.rfile.read(size)
                    self.rfile.readline()
            else:
                n = int(self.headers.get("Content-Length") or 0)
                data = self.rfile.read(n) if n else b""
            self.raw = data
        return self.raw

    def body(self) -> dict:
        try:
            value = json.loads(self.raw_body() or b"{}")
        except ValueError:
            return {}
        return value if isinstance(value, dict) else {}

    def signed(self) -> bool:
        """Whether an api.github.com call carries a token the deployment logs in with (gh sends `token TOKEN`): the
        read login's, or the write login's when CONFIG gives one; a call that does not is answered 401. Its log says
        which it carried: login (the read login's), write, other or none.
        """
        said = (self.headers.get("Authorization") or "").split(None, 1)
        github = self.config["github"]
        tokens = {github["token"]: "login"}
        if github.get("write_token"):
            tokens[github["write_token"]] = "write"
        token = tokens.get(said[-1], "other") if said else "none"
        self.record(token=token)
        if token not in ("login", "write"):
            self.raw_body()
            self.answer(401, BAD_CREDENTIALS)
            return False
        return True

    def writer(self) -> bool:
        """Whether this call, signed (signed), carries the write login's token."""
        return self.extra.get("token") == "write"

    # --- the methods -------------------------------------------------------------------------------------------------
    def do_GET(self):
        host = self.host()
        if host == API:
            if self.signed():
                self.github_get()
        elif host == WEB:
            self.git()
        elif host == self.config["gateway"]["host"]:
            self.claude()
        else:
            self.answer(404, {"message": "Not Found"})

    def do_POST(self):
        host = self.host()
        if host == API:
            if not self.signed():
                return
            query = self.body().get("query", "")
            path = self.path.split("?", 1)[0].strip("/")
            if path == "graphql" and "mutation" not in query and VIEWER.search(query):
                self.record(query=query[:200])
                login = self.config["github"]["login"]
                self.answer(200, {"data": {"viewer": {"login": login}}})
                return
            branch = BRANCH.search(query) if path == "graphql" else None
            if branch and "mutation" not in query:
                self.record(query=query[:200])
                head = {"name": "develop", "target": {"oid": self.develop}}
                self.answer(200, {"data": {"repository": {branch.group(1): head}}})
                return
            if path == "graphql" and self.writer() and self.pr_comment(query):
                return
            if path == "graphql" and "mutation" not in query:
                super().do_POST()
                return
            probed = probe_answer(path, self.body())
            self.record(write=True, probe=probed is not None)
            if probed is not None and self.writer():
                self.answer(422, probed)
            else:
                self.answer(403, READ_ONLY)
        elif host == WEB:
            self.git()
        elif host == self.config["gateway"]["host"]:
            self.claude()
        else:
            self.answer(404, {"message": "Not Found"})

    def do_HEAD(self):
        self.answer(404, b"")

    def refuse_write(self):
        self.raw_body()
        self.record(write=True, probe=False)
        if self.host() == API and not self.signed():
            return
        self.answer(403 if self.host() == API else 405, READ_ONLY)

    do_PUT = do_PATCH = do_DELETE = refuse_write

    # --- api.github.com ----------------------------------------------------------------------------------------------
    def pr_comment(self, query: str) -> bool:
        """The write login's comment on a pull request, as `gh pr comment` makes one: the pull request looked up by
        its number, any number (its id, number and URL), then the comment on it, whose URL it answers. Whether this
        GraphQL call was one of the two; any other is not answered here.
        """
        repo = self.config["github"]["repository"]
        variables = self.body().get("variables") or {}
        if "mutation" not in query and PULL_REQUEST.search(query):
            n = variables.get("pr_number")
            if not isinstance(n, int):
                return False
            self.record(query=query[:200])
            pr = {
                "id": f"{PR_ID}{n}",
                "number": n,
                "url": f"https://{WEB}/{repo}/pull/{n}",
            }
            self.answer(200, {"data": {"repository": {"pullRequest": pr}}})
            return True
        subject = str((variables.get("input") or {}).get("subjectId") or "")
        n = subject[len(PR_ID) :] if subject.startswith(PR_ID) else ""
        if "mutation" not in query or not ADD_COMMENT.search(query) or not n.isdigit():
            return False
        self.record(write=True, probe=False, query=query[:200])
        url = f"https://{WEB}/{repo}/pull/{n}#issuecomment-{time.time_ns() // 1000}"
        self.answer(
            200, {"data": {"addComment": {"commentEdge": {"node": {"url": url}}}}}
        )
        return True

    def github_get(self) -> None:
        """gh's login asks the API's root (its scopes header: none, as for a fine-grained token) and /user; develop's
        head is the served repository's; the rest is fake_servers.GitHub's."""
        path = self.path.split("?", 1)[0].strip("/")
        repo = self.config["github"]["repository"]
        if path == "":
            self.answer(200, {"current_user_url": f"https://{API}/user"})
        elif path == "user":
            login = self.config["github"]["login"]
            self.answer(200, {"login": login, "id": 1, "type": "User"})
        elif path in (
            f"repos/{repo}/commits/develop",
            f"repos/{repo}/branches/develop",
        ):
            body = {"sha": self.develop, "commit": {"message": "develop"}}
            if path.endswith("branches/develop"):
                body = {"name": "develop", "commit": body, "protected": False}
            self.answer(200, fake_servers.shape(body, self.headers.get("X-Jq") or ""))
        else:
            super().do_GET()

    # --- github.com: git's own http-backend ------------------------------------------------------------------------
    def git(self) -> None:
        path, _, query = self.path.partition("?")
        m = GIT_PATH.fullmatch(path)
        if not m:
            self.answer(404, b"", "text/plain")
            return
        if m.group(3) == "/git-receive-pack" or "git-receive-pack" in query:
            self.raw_body()
            self.record(write=True)
            self.answer(403, b"the fake GitHub takes no pushes\n", "text/plain")
            return
        data = self.raw_body() if self.command == "POST" else b""
        env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.state),
            "GIT_PROJECT_ROOT": str(self.state / "git"),
            "GIT_HTTP_EXPORT_ALL": "1",
            "PATH_INFO": f"/{m.group(1)}/{m.group(2)}.git{m.group(3)}",
            "REQUEST_METHOD": self.command,
            "QUERY_STRING": query,
            "CONTENT_TYPE": self.headers.get("Content-Type") or "",
            "CONTENT_LENGTH": str(len(data)),
            "REMOTE_ADDR": self.client_address[0],
            "GIT_PROTOCOL": self.headers.get("Git-Protocol") or "",
            "HTTP_CONTENT_ENCODING": self.headers.get("Content-Encoding") or "",
            # a clone's fetches ask for objects by filter, and later by id
            "GIT_CONFIG_COUNT": "3",
            "GIT_CONFIG_KEY_0": "safe.directory",
            "GIT_CONFIG_VALUE_0": "*",
            "GIT_CONFIG_KEY_1": "uploadpack.allowFilter",
            "GIT_CONFIG_VALUE_1": "true",
            "GIT_CONFIG_KEY_2": "uploadpack.allowAnySHA1InWant",
            "GIT_CONFIG_VALUE_2": "true",
        }
        p = subprocess.run(
            ["git", "http-backend"], input=data, env=env, capture_output=True
        )
        head, sep, rest = p.stdout.partition(b"\r\n\r\n")
        if not sep:
            head, sep, rest = p.stdout.partition(b"\n\n")
        status, headers = 200, []
        for line in head.decode("latin-1").splitlines():
            name, _, value = line.partition(":")
            if name.lower() == "status":
                status = int(value.split()[0])
            elif name:
                headers.append((name.strip(), value.strip()))
        if not sep:
            status, headers, rest = 500, [], p.stderr[-500:]
        self.log(status)
        self.send_response(status)
        for name, value in headers:
            if name.lower() != "content-length":
                self.send_header(name, value)
        self.send_header("Content-Length", str(len(rest)))
        self.end_headers()
        self.wfile.write(rest)

    # --- the Claude gateway ------------------------------------------------------------------------------------------
    def key_said(self) -> bool:
        """Whether the request carries the gateway's key as the settings send it; the headers that carried it are
        logged by name, never the key."""
        g = self.config["gateway"]
        key = g["key"]
        header = g.get("header") or "x-api-key"
        value = f"{g['scheme']} {key}" if g.get("scheme") else key
        carried = sorted({name.lower() for name, v in self.headers.items() if key in v})
        in_header = (self.headers.get(header) or "") == value
        in_bearer = (self.headers.get("Authorization") or "") == f"Bearer {key}"
        right = {
            "header": in_header,
            "bearer": in_bearer,
            "header_and_bearer": in_header and in_bearer,
        }.get(g.get("sent_as") or "header", False)
        self.record(key_headers=carried, key="right" if right else "wrong")
        return right

    def claude(self) -> None:
        path = self.path.split("?", 1)[0]
        req = self.body() if self.command == "POST" else {}
        models = self.config["gateway"]["models"]
        if not self.key_said():
            self.answer(
                401,
                {
                    "type": "error",
                    "error": {"type": "authentication_error", "message": "invalid key"},
                },
            )
        elif self.command == "GET" and path == "/v1/models":
            data = [
                {
                    "type": "model",
                    "id": m,
                    "display_name": m,
                    "created_at": "2026-01-01T00:00:00Z",
                }
                for m in models
            ]
            self.answer(
                200,
                {
                    "data": data,
                    "has_more": False,
                    "first_id": models[0],
                    "last_id": models[-1],
                },
            )
        elif self.command == "POST" and path == "/v1/messages/count_tokens":
            self.answer(200, {"input_tokens": 10})
        elif self.command == "POST" and path == "/v1/messages":
            model = str(req.get("model") or models[0])
            self.record(model=model, stream=bool(req.get("stream")))
            if req.get("stream"):
                self.log(200)
                self.stream(
                    sse_events(model),
                    lambda e: f"event: {e['type']}\ndata: {json.dumps(e)}\n\n",
                    "text/event-stream",
                )
            else:
                self.answer(200, message(model))
        else:
            self.answer(
                404,
                {
                    "type": "error",
                    "error": {"type": "not_found_error", "message": path},
                },
            )


def message(model: str) -> dict:
    """A finished one-word answer, as the Messages API gives it."""
    return {
        "id": "msg_fake",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": "OK"}],
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 10, "output_tokens": 1},
    }


def sse_events(model: str) -> list:
    """The same answer as the Messages API streams it."""
    start = dict(message(model), content=[], stop_reason=None)
    return [
        {"type": "message_start", "message": start},
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "OK"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 1},
        },
        {"type": "message_stop"},
    ]


# --- what the fakes make before they serve ---------------------------------------------------------------------------
def openssl(*args: str) -> None:
    subprocess.run(["openssl", *args], check=True, capture_output=True)


def certificates(state: pathlib.Path, served: list) -> None:
    """A root certificate (ca.crt), and the server's (server.crt, server.key) for each of SERVED, signed by it: the
    extensions every client checks, Python's strict verification included."""
    ca_ext = state / "ca.ext"
    ca_ext.write_text(
        "basicConstraints=critical,CA:TRUE\nkeyUsage=critical,keyCertSign,cRLSign\nsubjectKeyIdentifier=hash\n",
        encoding="utf-8",
    )
    server_ext = state / "server.ext"
    server_ext.write_text(
        "basicConstraints=critical,CA:FALSE\nkeyUsage=critical,digitalSignature\nextendedKeyUsage=serverAuth\n"
        "subjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid:always\n"
        f"subjectAltName={','.join('DNS:' + n for n in served)}\n",
        encoding="utf-8",
    )
    ec = ("-newkey", "ec", "-pkeyopt", "ec_paramgen_curve:P-256", "-nodes")
    s = str(state)
    openssl(
        "req",
        "-new",
        *ec,
        "-subj",
        "/CN=rkmon image test root",
        "-keyout",
        f"{s}/ca.key",
        "-out",
        f"{s}/ca.csr",
    )
    openssl(
        "x509",
        "-req",
        "-in",
        f"{s}/ca.csr",
        "-signkey",
        f"{s}/ca.key",
        "-days",
        DAYS,
        "-extfile",
        str(ca_ext),
        "-out",
        f"{s}/ca.crt",
    )
    openssl(
        "req",
        "-new",
        *ec,
        "-subj",
        f"/CN={served[0]}",
        "-keyout",
        f"{s}/server.key",
        "-out",
        f"{s}/server.csr",
    )
    openssl(
        "x509",
        "-req",
        "-in",
        f"{s}/server.csr",
        "-CA",
        f"{s}/ca.crt",
        "-CAkey",
        f"{s}/ca.key",
        "-CAcreateserial",
        "-days",
        DAYS,
        "-extfile",
        str(server_ext),
        "-out",
        f"{s}/server.crt",
    )


def repository(
    state: pathlib.Path, name: str, prs: list = (), login: str = ""
) -> tuple:
    """tests/fakerepo.py's repository, with develop one commit further on, made now (fakerepo.today), and the open PRs
    PRS asks for (the module's docstring: opened), as the bare repository git/NAME.git that github.com serves:
    (develop's head, GitHub's view of each PR). The monitor's clone starts from a recent date, and finds that commit.
    """
    import fakerepo

    origin = state / "work" / "origin"
    fakerepo.make(state / "work")
    fakerepo.today(origin)
    served = [opened(origin, name, spec, login) for spec in prs]
    bare = state / "git" / f"{name}.git"
    bare.parent.mkdir(parents=True)
    subprocess.run(
        ["git", "clone", "-q", "--mirror", str(origin), str(bare)],
        check=True,
        capture_output=True,
    )
    return fakerepo.git(bare, "rev-parse", "refs/heads/develop"), served


def opened(origin: pathlib.Path, name: str, spec: dict, login: str) -> dict:
    """The open PR SPEC asks for (the module's docstring), in ORIGIN, the repository NAME's work tree: GitHub's view
    of it (fakerepo.pull_request), with its formal reviews, each by LOGIN on the head of the push it names, in GitHub's
    REST form."""
    import fakerepo

    n = spec["number"]
    pr = fakerepo.pull_request(
        origin,
        n,
        lines=spec.get("lines", 3),
        at=spec.get("at") or fakerepo.ENV["GIT_COMMITTER_DATE"],
        pushes=spec.get("pushes", 1),
    )
    pr["reviews"] = [
        {
            "id": 1000 * n + k,
            "user": {"login": login},
            "body": review["body"],
            "state": "COMMENTED",
            "html_url": f"https://{WEB}/{name}/pull/{n}#pullrequestreview-{1000 * n + k}",
            "commit_id": pr["pushes"][review["push"] - 1],
            "submitted_at": review["at"],
        }
        for k, review in enumerate(spec.get("reviews") or (), 1)
    ]
    return pr


class TLSServer(http.server.ThreadingHTTPServer):
    """Each connection's TLS handshake in its own thread, so a slow client holds up no other."""

    daemon_threads = True

    def __init__(self, address, handler, context: ssl.SSLContext):
        super().__init__(address, handler)
        self.context = context

    def finish_request(self, request, client_address):
        try:
            request = self.context.wrap_socket(request, server_side=True)
        except (ssl.SSLError, OSError):
            return
        self.RequestHandlerClass(request, client_address, self)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--state", required=True)
    ap.add_argument("--port", type=int, default=443)
    a = ap.parse_args()
    config = json.loads(pathlib.Path(a.config).read_text(encoding="utf-8"))
    state = pathlib.Path(a.state)
    state.mkdir(mode=0o700, exist_ok=True)
    os.environ["HOME"] = str(state)  # git's and openssl's own files, if they write any
    certificates(state, names(config))
    github = config["github"]
    develop, prs = repository(
        state, github["repository"], config.get("prs") or [], github["login"]
    )
    handler = type(
        "NetHandler",
        (Net,),
        {
            "config": config,
            "state": state,
            "develop": develop,
            "prs": prs,
            "log_path": None,
        },
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(str(state / "server.crt"), str(state / "server.key"))
    server = TLSServer(("0.0.0.0", a.port), handler, context)
    (state / "ready").write_text(f"{a.port}\n", encoding="utf-8")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
