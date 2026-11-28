#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Fake reviewer-model endpoints for the container stages (tests/container/stage_models.py): an Ollama server and an
OpenAI-compatible gateway behind an API-management layer, on a loopback address, with nothing real behind them. Every
name and value here is invented.

- Ollama: /api/tags (a model that calls tools and one that only talks, each with its size), /api/show, /api/chat.
- Gateway: every route under a prefix (a gateway's route); the key in its own header, sent raw, and, with `bearer_too`,
  as a bearer token as well (anything less: 401); a model list at `models_paths` (inside the prefix, or the host's
  own /models outside it), each model by the ID requests take (a deployment ID) and, with `deployments`, a display
  name besides it; a LiteLLM-style model-info route under the prefix with the context length, the max output and the
  rate limits (unless `info` is off); chat at the `chat_routes` under the prefix, streamed or not, answered as vLLM
  does (a `reasoning` field beside the content, usage with reasoning tokens, stop_reason, token_ids, routed_experts,
  the model echoed in lower case); one model answers only in a simplified "unified" shape with no choices; HTTP 429
  with a Retry-After for the chat requests named in `fail429` (1 is the first chat request), or in `fail429_review`
  (1 is the first request of the review conversations, the worker's).
- Both answer as the review loop's model would: a request with tools and no tool result yet gets one read_file call
  (the model that only talks answers in text instead); once a tool result is in the conversation, the review itself.
  A chat takes `delay_s` seconds, so drafts in flight overlap. With `script` ({marker: [step, ...]}), a conversation
  whose system message has a marker follows its steps instead, one per answer (the number of assistant messages so
  far picks it; the last repeats): {"call": name, "args": {...}} is a tool call, {"text": "..."} the answer.

Every request is logged to <name>.log in the state folder, one JSON line: start and end times, method, path, status,
model and the PR its conversation reviews; never a header or a key, so a test can measure what ran at once and how
often. A client that goes away before its answer (a worker a stop ends mid-chat) is no error of the fake's: it prints
nothing for it.

    python3 fake_models.py --state-dir DIR --kind ollama|gateway [--name N] [--host H] [--port P] [--key-file F]
"""

import argparse
import hashlib
import http.server
import json
import pathlib
import re
import sys
import threading
import time

GOOD, TEXT_ONLY, UNIFIED = "Big.Model-v5.3-Flash", "Text.only-Model", "Unified-Model"
DEPLOY, SHOWN = (
    "example-chat-1-flash",
    "Example-Chat-1-Flash",
)  # a deployment ID and the name a list shows for it
OLLAMA_GOOD, OLLAMA_TEXT = "fake-model:1", "text-only:1"
PR_IN_PROMPT = re.compile(r"/pull/(\d+)/")


def _review(n) -> str:
    return (
        f"## PR #{n}: first-round review (from the fake reviewer model)\n\n| Item | Status |\n|---|---|\n"
        "| Our checks | not run |\n\nNet-new: none.\n\nUtility: a fake PR for the container test.\n"
    )


class _Model(http.server.BaseHTTPRequestHandler):
    """What both fakes share: the log, the answer the review loop expects, the 429 schedule."""

    name, log_path, delay_s = "model", None, 0.0
    script: dict | None = None
    fail429: frozenset = frozenset()
    fail429_review: frozenset = frozenset()
    retry_after = "3"
    lock = threading.Lock()
    chats = reviews = 0

    def log_message(self, *a):
        pass

    def _send(self, status: int, body, ctype: str = "application/json", headers=()):
        data = (
            body
            if isinstance(body, bytes)
            else (body if isinstance(body, str) else json.dumps(body)).encode()
        )
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self._status = status
        # The log line first, then the answer: a caller that has the answer finds its line (under the full suite's
        # load a test read the log between an answer and its line).
        self._record(getattr(self, "_t0", None), getattr(self, "_req", None))
        self.wfile.write(data)

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return {}

    @staticmethod
    def _pr(req: dict | None):
        msgs = (req or {}).get("messages") or []
        first = next(
            (m.get("content") or "" for m in msgs if m.get("role") == "user"), ""
        )
        m = PR_IN_PROMPT.search(first if isinstance(first, str) else "")
        return (int(m.group(1)) if m else None), first

    def _record(self, t0: float | None, req: dict | None = None) -> None:
        """The request's log line, once (its answer, or the end of a handler that sent none, writes it)."""
        if t0 is None or getattr(self, "_logged", False) or not self.log_path:
            return
        self._logged = True
        pr, first = self._pr(req)
        msgs = (req or {}).get("messages") or []
        rec = {
            "t0": round(t0, 3),
            "t1": round(time.time(), 3),
            "method": self.command,
            "path": self.path,
            "status": getattr(self, "_status", 0),
            "model": (req or {}).get("model"),
            "pr": pr,
            "tools": bool((req or {}).get("tools")),
            "turn": sum(1 for x in msgs if x.get("role") == "tool"),
            "conv": (
                hashlib.sha1(str(first).encode()).hexdigest()[:10] if first else None
            ),
        }
        with self.lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    def _chat_429(self, req: dict) -> bool:
        """Count this chat request; True when the schedule gives it a 429 (answered here)."""
        cls = type(self)
        with self.lock:
            cls.chats += 1
            n, review = cls.chats, None
            if self._pr(req)[0] is not None:
                cls.reviews += 1
                review = cls.reviews
        if n in self.fail429 or (review is not None and review in self.fail429_review):
            self._send(
                429,
                {"error": {"message": "rate limited (fake)", "type": "rate_limit"}},
                headers=[("Retry-After", self.retry_after)] if self.retry_after else [],
            )
            return True
        return False

    @classmethod
    def scripted(cls, req: dict) -> tuple[str, dict | None] | None:
        """The answer `script` gives this conversation, or None when no marker is in its system message."""
        msgs = req.get("messages") or []
        system = next(
            (m.get("content") or "" for m in msgs if m.get("role") == "system"), ""
        )
        steps = next(
            (v for k, v in (cls.script or {}).items() if k in str(system)), None
        )
        if not steps:
            return None
        step = steps[
            min(sum(1 for m in msgs if m.get("role") == "assistant"), len(steps) - 1)
        ]
        if "call" in step:
            return "", {"name": step["call"], "arguments": step.get("args") or {}}
        return step.get("text", ""), None

    @classmethod
    def answer_for(cls, req: dict, calls_tools: bool) -> tuple[str, dict | None]:
        """(text, tool call or None): a read_file call first, the review once a tool result came back."""
        got = cls.scripted(req)
        if got is not None:
            return got
        msgs = req.get("messages") or []
        if any(m.get("role") == "tool" for m in msgs):
            pr, _first = cls._pr(req)
            return _review(pr if pr is not None else "?"), None
        if req.get("tools") and calls_tools:
            return "", {"name": "read_file", "arguments": {"path": "README.md"}}
        if req.get("tools"):
            return "I would read README.md first, then look at the diff.", None
        return "OK", None


class Ollama(_Model):
    models = ((OLLAMA_GOOD, 4 * 2**30, True), (OLLAMA_TEXT, 2 * 2**30, False))

    def do_GET(self):
        t0 = self._t0 = time.time()
        if self.path == "/api/tags":
            self._send(
                200,
                {
                    "models": [
                        {
                            "name": n,
                            "model": n,
                            "size": s,
                            "details": {"family": "fake"},
                        }
                        for n, s, _ in self.models
                    ]
                },
            )
        else:
            self._send(404, {"error": "not found"})
        self._record(t0)

    def do_POST(self):
        t0 = self._t0 = time.time()
        req = self._req = self._body()
        tools_ok = {n: t for n, _, t in self.models}
        if self.path == "/api/show":
            self._send(
                200,
                (
                    {"model_info": {"fake.context_length": 32768}}
                    if req.get("model") in tools_ok
                    else {"error": "model not found"}
                ),
            )
        elif self.path == "/api/chat" and req.get("model") not in tools_ok:
            self._send(404, {"error": f"model '{req.get('model')}' not found"})
        elif self.path == "/api/chat":
            if not self._chat_429(req):
                time.sleep(self.delay_s)
                text, call = self.answer_for(req, tools_ok[req["model"]])
                msg = {"role": "assistant", "content": text}
                if call:
                    msg["tool_calls"] = [{"function": call}]
                if req.get("stream"):
                    chunks = [
                        {"message": msg, "done": False},
                        {"message": {"role": "assistant", "content": ""}, "done": True},
                    ]
                    self._send(
                        200,
                        "".join(json.dumps(c) + "\n" for c in chunks),
                        "application/x-ndjson",
                    )
                else:
                    self._send(
                        200, {"model": req["model"], "message": msg, "done": True}
                    )
        else:
            self._send(404, {"error": "not found"})
        self._record(t0, req)


class Gateway(_Model):
    prefix, header, key_file, bearer_too = "/gw/v2", "X-Example-Key", None, False
    chat_routes = ("/v1/chat/completions",)
    models_paths: tuple = ()  # absolute paths; () means <prefix>/v1/models
    deployments = (
        False  # list each model by a deployment ID with a display name besides it
    )
    info = True
    rpm, tpm, context, max_output = 4, 400000, 131072, 8192
    tools = {GOOD: True, TEXT_ONLY: False, UNIFIED: False, DEPLOY: True}

    def _listed(self) -> list:
        if self.deployments:
            return [
                {"deployment_id": DEPLOY, "display_name": SHOWN},
                {"deployment_id": UNIFIED.lower(), "display_name": UNIFIED},
            ]
        return [{"id": n, "object": "model"} for n in (GOOD, TEXT_ONLY, UNIFIED)]

    def _authorized(self) -> bool:
        key = (
            pathlib.Path(self.key_file).read_text(encoding="utf-8").strip()
            if self.key_file
            else ""
        )
        ok = bool(key) and self.headers.get(self.header) == key
        if ok and self.bearer_too:
            ok = self.headers.get("Authorization") == f"Bearer {key}"
        if not ok:
            self._send(
                401,
                {
                    "error": {
                        "message": "Access denied: a missing or invalid subscription key"
                    }
                },
            )
        return ok

    def do_GET(self):
        t0 = self._t0 = time.time()
        path = self.path.split("?", 1)[0]
        if path in (self.models_paths or (self.prefix + "/v1/models",)):
            if self._authorized():
                self._send(200, {"object": "list", "data": self._listed()})
        elif self.info and path in (
            self.prefix + "/v1/model/info",
            self.prefix + "/model/info",
        ):
            if self._authorized():
                self._send(
                    200,
                    {
                        "data": [
                            {
                                "model_name": GOOD,
                                "litellm_params": {"rpm": self.rpm, "tpm": self.tpm},
                                "model_info": {
                                    "max_input_tokens": self.context,
                                    "max_output_tokens": self.max_output,
                                    "supports_function_calling": True,
                                },
                            },
                            {
                                "model_name": TEXT_ONLY,
                                "model_info": {
                                    "max_input_tokens": 8192,
                                    "supports_function_calling": False,
                                },
                            },
                        ]
                    },
                )
        else:
            self._send(404, {"error": "no such route"})
        self._record(t0)

    def do_POST(self):
        t0 = self._t0 = time.time()
        req = self._req = self._body()
        path = self.path.split("?", 1)[0]
        model = req.get("model")
        known = {m.get("deployment_id") or m.get("id") for m in self._listed()} | (
            {DEPLOY} if self.deployments else set()
        )
        if path not in tuple(self.prefix + r for r in self.chat_routes):
            self._send(404, {"error": "no such route"})
        elif not self._authorized():
            pass
        elif model not in known:
            self._send(404, {"error": {"message": f"no model or deployment {model}"}})
        elif (
            model.lower() == UNIFIED.lower()
        ):  # a simplified shape with no choices, so no tool calls
            if not self._chat_429(req):
                answer = {
                    "model": model.lower(),
                    "deployment": "unified-1",
                    "response": {"role": "assistant", "content": "OK"},
                }
                if req.get("stream"):
                    self._send(
                        200,
                        f"data: {json.dumps(answer)}\n\ndata: [DONE]\n\n",
                        "text/event-stream",
                    )
                else:
                    self._send(200, answer)
        elif not self._chat_429(req):
            time.sleep(self.delay_s)
            text, call = self.answer_for(req, self.tools.get(model, False))
            calls = (
                [
                    {
                        "id": "chatcmpl-tool-1",
                        "type": "function",
                        "function": {
                            "name": call["name"],
                            "arguments": json.dumps(call["arguments"]),
                        },
                    }
                ]
                if call
                else []
            )
            usage = {
                "prompt_tokens": 1000,
                "completion_tokens": 300,
                "total_tokens": 1300,
                "completion_tokens_details": {"reasoning_tokens": 250},
            }
            if not req.get("stream"):
                self._send(
                    200,
                    {
                        "id": "chatcmpl-1",
                        "object": "chat.completion",
                        "model": model.lower(),
                        "choices": [
                            {
                                "index": 0,
                                "message": {
                                    "role": "assistant",
                                    "content": text or None,
                                    "reasoning": "Thinking it over.",
                                    "tool_calls": calls,
                                },
                                "logprobs": None,
                                "finish_reason": "tool_calls" if call else "stop",
                                "stop_reason": None,
                                "token_ids": None,
                            }
                        ],
                        "usage": usage,
                        "routed_experts": None,
                    },
                )
            else:
                delta = {"role": "assistant", "reasoning": "Thinking it over."}
                if calls:
                    delta["tool_calls"] = [dict(c, index=0) for c in calls]
                else:
                    delta["content"] = text
                chunks = [
                    {
                        "id": "chatcmpl-1",
                        "object": "chat.completion.chunk",
                        "model": model.lower(),
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": None,
                                "token_ids": None,
                            }
                        ],
                    },
                    {
                        "id": "chatcmpl-1",
                        "object": "chat.completion.chunk",
                        "model": model.lower(),
                        "choices": [
                            {
                                "index": 0,
                                "delta": {},
                                "finish_reason": "tool_calls" if call else "stop",
                                "stop_reason": None,
                            }
                        ],
                        "usage": usage,
                    },
                ]
                self._send(
                    200,
                    "".join(f"data: {json.dumps(c)}\n\n" for c in chunks)
                    + "data: [DONE]\n\n",
                    "text/event-stream",
                )
        self._record(t0, req)


class _Server(http.server.ThreadingHTTPServer):
    """The fakes' server, a thread for each request: a client gone before its answer is sent prints nothing; any other
    error prints its traceback, as the base server's do."""

    daemon_threads = True

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], ConnectionError):
            super().handle_error(request, client_address)


class Server:
    """One fake running in this process: its base URL, its handler class (settings may change while it runs) and
    `stop()`."""

    def __init__(self, srv, url: str, handler):
        self.srv, self.url, self.handler = srv, url, handler

    def stop(self) -> None:
        self.srv.shutdown()
        self.srv.server_close()


def start(
    state_dir,
    kind: str,
    name: str | None = None,
    host: str = "127.0.0.1",
    port: int = 0,
    **settings,
) -> Server:
    """Start one fake in this process. `settings` set the handler's attributes (prefix, header, key_file, bearer_too,
    chat_routes, models_paths, deployments, info, rpm, tpm, fail429, fail429_review, retry_after, delay_s, script).
    """
    state = pathlib.Path(state_dir)
    state.mkdir(parents=True, exist_ok=True)
    base = {"ollama": Ollama, "gateway": Gateway}[kind]
    name = name or kind
    attrs = dict(
        settings,
        log_path=state / f"{name}.log",
        name=name,
        chats=0,
        reviews=0,
        lock=threading.Lock(),
    )
    for key in ("fail429", "fail429_review"):
        if key in attrs:
            attrs[key] = frozenset(attrs[key])
    cls = type(f"{base.__name__}_{name}", (base,), attrs)
    srv = _Server((host, port), cls)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://{host}:{srv.server_address[1]}" + (
        cls.prefix if kind == "gateway" else ""
    )
    return Server(srv, url, cls)


def read_log(state_dir, name: str) -> list[dict]:
    p = pathlib.Path(state_dir) / f"{name}.log"
    try:
        return [
            json.loads(x)
            for x in p.read_text(encoding="utf-8").splitlines()
            if x.strip()
        ]
    except OSError:
        return []


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--state-dir", required=True)
    ap.add_argument("--kind", choices=("ollama", "gateway"), required=True)
    ap.add_argument("--name")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--key-file")
    a = ap.parse_args()
    srv = start(
        a.state_dir,
        a.kind,
        a.name,
        a.host,
        a.port,
        **({"key_file": a.key_file} if a.key_file else {}),
    )
    print(json.dumps({"url": srv.url}), flush=True)
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
