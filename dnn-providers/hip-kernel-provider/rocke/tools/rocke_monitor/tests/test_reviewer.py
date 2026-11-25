# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The reviewer, the first pass as one setting (config `reviewer`, rkmon_lib/bindings.py): each choice (the default, the
local worker, an OpenAI-compatible binding, an Anthropic binding) in the configuration's checks, line 1, rk doctor and
rk features; the tick triage, which never follows the reviewer (the rule filter by default, a model an opt-in); and the
Anthropic Messages adapter (worker/llm_client.py AnthropicMessages) against a fake Messages API on the loopback: one
request at a time, a tool loop, the triage's loop, and the tool-loop drafter's whole phased draft. The fake reads the
key and the headers only to check them; no test prints one."""

import http.server
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(ROOT / "tests"))
sys.dont_write_bytecode = True

import drafter  # noqa: E402
import drafter_fixture as fx  # noqa: E402
import llm_client as lc  # noqa: E402
from rkmon_lib import (  # noqa: E402
    bindings,
    config,
    container_layout,
    custom_headers,
    deploy_tree,
    doctor,
    features,
    first_pass,
    roles,
    triage,
)
from test_drafter import HUNT, PRIOR  # noqa: E402

KEY = "k-" + "anthropic-test-0042"  # built at run time: no real key is ever in a test
URL_VAR = "RKMON_TEST_ANTHROPIC_URL"
TOOL = {
    "type": "function",
    "function": {
        "name": "lookup",
        "description": "Look a word up.",
        "parameters": {
            "type": "object",
            "properties": {"word": {"type": "string"}},
            "required": ["word"],
        },
    },
}


# --- a fake Anthropic Messages API --------------------------------------------------------------------------------------


def neutral_view(body: dict) -> dict:
    """A Messages API request in the neutral (OpenAI-like) shape tests/drafter_fixture.FakeEndpoint reads: the system
    text first, then each message's text, its tool calls and its tool results in order.
    """
    msgs = [{"role": "system", "content": body.get("system") or ""}]
    for m in body["messages"]:
        blocks = (
            m["content"]
            if isinstance(m["content"], list)
            else [{"type": "text", "text": m["content"]}]
        )
        text = "".join(b.get("text") or "" for b in blocks if b.get("type") == "text")
        if m["role"] == "assistant":
            calls = [
                {
                    "id": b["id"],
                    "function": {"name": b["name"], "arguments": b["input"]},
                }
                for b in blocks
                if b.get("type") == "tool_use"
            ]
            msgs.append({"role": "assistant", "content": text, "tool_calls": calls})
            continue
        for b in blocks:
            if b.get("type") == "tool_result":
                msgs.append(
                    {
                        "role": "tool",
                        "tool_call_id": b["tool_use_id"],
                        "content": b.get("content") or "",
                    }
                )
        if text:
            msgs.append({"role": "user", "content": text})
    return {"messages": msgs}


def encode(neutral: dict, cached: int = 0) -> tuple[list[dict], str, dict]:
    """A neutral answer as the Messages API's blocks, stop reason and usage."""
    blocks = [{"type": "thinking", "thinking": "considered it", "signature": "sig-1"}]
    if neutral.get("content"):
        blocks.append({"type": "text", "text": neutral["content"]})
    for tc in neutral.get("tool_calls") or []:
        blocks.append(
            {
                "type": "tool_use",
                "id": tc["id"],
                "name": tc["function"]["name"],
                "input": tc["function"]["arguments"],
            }
        )
    usage = dict(neutral.get("usage") or {})
    prompt = int(usage.get("prompt_tokens") or 50)
    return (
        blocks,
        "tool_use" if neutral.get("tool_calls") else "end_turn",
        {
            "input_tokens": max(0, prompt - cached - 7),
            "cache_read_input_tokens": cached,
            "cache_creation_input_tokens": 7,
            "output_tokens": int(usage.get("completion_tokens") or 20),
        },
    )


def sse(blocks: list[dict], stop: str, usage: dict) -> bytes:
    """The answer as the Messages API streams it: message_start, each block in two deltas, message_delta, message_stop."""
    ev = [
        (
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "role": "assistant",
                    "content": [],
                    "usage": {k: usage[k] for k in usage if k != "output_tokens"},
                },
            },
        ),
        ("ping", {"type": "ping"}),
    ]
    for i, b in enumerate(blocks):
        if b["type"] == "tool_use":
            ev.append(
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": i,
                        "content_block": {
                            "type": "tool_use",
                            "id": b["id"],
                            "name": b["name"],
                            "input": {},
                        },
                    },
                )
            )
            js = json.dumps(b["input"])
            for part in (js[: len(js) // 2], js[len(js) // 2 :]):
                ev.append(
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": i,
                            "delta": {"type": "input_json_delta", "partial_json": part},
                        },
                    )
                )
        elif b["type"] == "thinking":
            ev.append(
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": i,
                        "content_block": {"type": "thinking", "thinking": ""},
                    },
                )
            )
            ev.append(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {"type": "thinking_delta", "thinking": b["thinking"]},
                    },
                )
            )
            ev.append(
                (
                    "content_block_delta",
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {
                            "type": "signature_delta",
                            "signature": b["signature"],
                        },
                    },
                )
            )
        else:
            ev.append(
                (
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": i,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            )
            for part in (
                b["text"][: len(b["text"]) // 2],
                b["text"][len(b["text"]) // 2 :],
            ):
                ev.append(
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": i,
                            "delta": {"type": "text_delta", "text": part},
                        },
                    )
                )
        ev.append(("content_block_stop", {"type": "content_block_stop", "index": i}))
    ev.append(
        (
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop},
                "usage": {"output_tokens": usage["output_tokens"]},
            },
        )
    )
    ev.append(("message_stop", {"type": "message_stop"}))
    return "".join(
        f"event: {name}\ndata: {json.dumps(d)}\n\n" for name, d in ev
    ).encode()


class FakeMessages:
    """The Messages API on 127.0.0.1: POST /v1/messages checks the key (x-api-key), anthropic-version and the custom
    header, records each request, and answers from `answer(body)` (a neutral answer), streamed or whole as the request
    asks. `refuse_429` answers that many requests with HTTP 429 first; `refuse_thinking` answers a request that sends
    thinking blocks back with an HTTP 400 that names them. A wrong key gets HTTP 401 quoting the key it got.
    """

    def __init__(self, answer, cached: int = 0):
        self.answer, self.cached = answer, cached
        self.bodies: list[dict] = []
        self.headers: list[dict] = []
        self.paths: list[str] = []
        self.refuse_429, self.refuse_thinking = 0, False
        self.lock = threading.Lock()
        owner = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def send(
                self,
                status: int,
                data: bytes,
                ctype: str = "application/json",
                extra: dict | None = None,
            ):
                self.send_response(status)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(data)))
                for k, v in (extra or {}).items():
                    self.send_header(k, v)
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                ok = self.headers.get("x-api-key") == KEY and self.headers.get(
                    "anthropic-version"
                )
                self.send(200 if ok else 401, b'{"data": [{"id": "model-x"}]}')

            def do_POST(self):
                body = json.loads(
                    self.rfile.read(int(self.headers.get("Content-Length") or 0))
                )
                with owner.lock:
                    owner.paths.append(self.path)
                    owner.headers.append(
                        {k.lower(): v for k, v in self.headers.items()}
                    )
                    owner.bodies.append(body)
                    limited = owner.refuse_429 > 0
                    owner.refuse_429 -= 1 if limited else 0
                if self.path != "/v1/messages" or not self.headers.get(
                    "anthropic-version"
                ):
                    return self.send(
                        404, b'{"type": "error", "error": {"type": "not_found_error"}}'
                    )
                if self.headers.get("x-api-key") != KEY:
                    msg = {
                        "type": "error",
                        "error": {
                            "type": "authentication_error",
                            "message": f"invalid x-api-key {self.headers.get('x-api-key')}",
                        },
                    }
                    return self.send(401, json.dumps(msg).encode())
                if limited:
                    return self.send(
                        429,
                        b'{"type": "error", "error": {"type": "rate_limit_error"}}',
                        extra={"retry-after": "0"},
                    )
                sent_thinking = any(
                    b.get("type") == "thinking"
                    for m in body["messages"]
                    if m["role"] == "assistant"
                    for b in m["content"]
                )
                if owner.refuse_thinking and sent_thinking:
                    return self.send(
                        400,
                        b'{"type": "error", "error": {"type": "invalid_request_error", '
                        b'"message": "thinking blocks in an edited history"}}',
                    )
                blocks, stop, usage = encode(owner.answer(body), owner.cached)
                if body.get("stream"):
                    return self.send(200, sse(blocks, stop, usage), "text/event-stream")
                return self.send(
                    200,
                    json.dumps(
                        {
                            "id": "msg_1",
                            "type": "message",
                            "role": "assistant",
                            "content": blocks,
                            "stop_reason": stop,
                            "usage": usage,
                        }
                    ).encode(),
                )

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def stop(self):
        self.server.shutdown()
        self.server.server_close()


def binding(**over) -> dict:
    """An anthropic binding that names its own address by reference (the variable's name, never its value), read from
    the environment the caller runs in."""
    return dict(
        {
            "provider": "anthropic",
            "url": f"${URL_VAR}",
            "model": "model-x",
            "name": "Model X",
            "effort": "medium",
            "rpm": 100,
            "tpm": 1000000,
        },
        **over,
    )


class _Fake(unittest.TestCase):
    """A fake Messages API, its address in the environment the caller runs in, under the variable the binding's
    reference names, and the binding's key in the file of the secret it names, in the secrets folder.
    """

    def answer(self, body):
        return {"content": "done"}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.tmp.name)
        self.secrets = self.base / "secrets"
        self.secrets.mkdir()
        self.key = (
            self.secrets / "b_api_key"
        )  # the file of the secret the binding names (self.binding)
        self.key.write_text(f"{KEY}\n", encoding="utf-8")
        self.srv = FakeMessages(self.answer)
        self.addCleanup(self.srv.stop)
        self.patch = mock.patch.dict(os.environ, {URL_VAR: self.srv.url})
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def tearDown(self):
        self.tmp.cleanup()

    def binding(self, **over) -> dict:
        """binding(), naming the secret that holds its key (its file: self.key, in the secrets folder)."""
        return binding(api_key_secret=self.key.name, **over)

    def settings(self, **over) -> dict:
        cfg = {
            "model_bindings": {"claude": self.binding(**over)},
            "reviewer": "claude",
            "drafter": {"limiter_dir": str(self.base / "limits")},
            "paths": {"root": str(self.base), "secrets_dir": str(self.secrets)},
        }
        return bindings.client_settings(cfg, "drafter")

    def limiter(self, s):
        return lc.Limiter(s, who="test", sleep=lambda _s: None)


# --- the configuration: each choice, the older keys, line 1 -------------------------------------------------------------


class TestTheChoice(unittest.TestCase):
    BINDINGS = {
        "gw": {
            "url": "https://gw.example.invalid/v1",
            "model": "gw-model",
            "name": "GW Model",
        },
        "claude": binding(),
    }

    def cfg(self, **over) -> dict:
        return dict(
            {
                "model_bindings": self.BINDINGS,
                "worker": {"model": "local-model:1", "url": "http://127.0.0.1:11434"},
            },
            **over,
        )

    def test_a_binding_on_this_host_is_local_by_config_s_one_list(self):
        """A binding with no `where` is local when its address names this host (config.is_local_url, over the one list
        of this host's names, container_layout.LOCAL_HOSTS), and on a gateway otherwise.
        """
        cfg = self.cfg(
            model_bindings={
                "near": {"url": "http://model-host.example.com:8000/v1", "model": "m"}
            }
        )
        self.assertEqual(bindings.binding_by_name(cfg, "near")["where"], "gateway")
        here = container_layout.LOCAL_HOSTS + ("model-host.example.com",)
        with mock.patch.object(container_layout, "LOCAL_HOSTS", here):
            self.assertEqual(bindings.binding_by_name(cfg, "near")["where"], "local")

    def test_an_address_that_cannot_be_read_is_refused_naming_its_key(self):
        """An address urlsplit cannot read (an IPv6 host with no closing bracket, brackets around no IPv6 address), one
        that names no host, or one whose port is not a number from 0 to 65535 is refused, naming its key: a binding's
        url (bindings.problems), the worker's url and models_url and a dashboard's url (config.problems). rk drafts
        (bindings.describe) shows the host config.safe_url shows, "(not an address)" for none, and the reference a
        `$NAME` address is."""
        keys = (
            "worker.url",
            "worker.models_url",
            deploy_tree.name("team_dashboard.url"),
        )
        for url, why, host in (
            ("http://[::1", "its host cannot be read", "(not an address)"),
            ("http://[zz]/v1", "its host cannot be read", "(not an address)"),
            # the next two in two parts: the privacy check reads what follows their colon as a host
            ("http://" + ":8080/v1", "it names no host", "(not an address)"),
            (
                "http://localhost" + ":x/v1",
                "its port is not a number from 0 to 65535",
                "localhost",
            ),
            (
                "http://localhost:99999/v1",
                "its port is not a number from 0 to 65535",
                "localhost",
            ),
            ("http://[::1]:11434/v1", "", "[::1]"),
        ):
            cfg = self.cfg(
                model_bindings={"gw": {"url": url, "model": "m"}},
                reviewer="gw",
                worker={"model": "local-model:1", "url": url, "models_url": url},
                team_dashboard={"url": url},
            )
            said = f" is {url!r}; {why}"
            with self.subTest(url=url):
                refused = [
                    p
                    for p in bindings.problems(cfg)
                    if p.startswith("model_bindings.gw.base_url")
                ]
                self.assertEqual(
                    refused, [f"model_bindings.gw.base_url{said}"] if why else []
                )
                problems = [p for p in config.problems(cfg) if p.startswith(keys)]
                self.assertEqual(
                    problems, [f"{key}{said}" for key in keys] if why else []
                )
                where = "local" if config.is_local_url(url) else "gateway"
                self.assertTrue(
                    bindings.describe(cfg)[1].startswith(
                        f"drafter: gw = m via openai-compatible at {host} ({where}), "
                    )
                )
        claude = self.cfg(reviewer="claude")
        self.assertTrue(
            bindings.describe(claude)[1].startswith(
                f"drafter: claude = Model X via claude at ${URL_VAR} "
            )
        )

    def test_each_choice_and_its_line_1(self):
        """default: the verifier reviews alone and line 1 names it; worker: the local worker's model; a binding: its
        name (an OpenAI-compatible one or an Anthropic one alike), at the first review pass's effort, which on its
        default is the verifier's for a Claude model and none for another. Line 1 follows a swap, and gives the
        automated reviewer name (here the container's name, rocke-monitor, as the name is blank).
        """
        rows = {
            "default": (
                "default",
                True,
                "Automated review by rocke-monitor: Model - Claude Opus, effort: high",
            ),
            "worker": (
                "worker",
                False,
                f"Automated review by rocke-monitor: Model - local-model:1, effort: {config.MODEL_DEFAULT}",
            ),
            "gw": (
                "binding",
                False,
                "Automated review by rocke-monitor: Model - GW Model, effort: the model's default",
            ),
            "claude": (
                "binding",
                False,
                "Automated review by rocke-monitor: Model - Model X, effort: high",
            ),
        }
        for choice, (kind, alone, line1) in rows.items():
            with self.subTest(choice=choice):
                cfg = config.normalize(self.cfg(reviewer=choice))
                self.assertEqual(
                    (
                        bindings.reviewer(cfg)["kind"],
                        bindings.no_draft(cfg),
                        bindings.first_line(cfg),
                    ),
                    (kind, alone, line1),
                )
                self.assertEqual(bindings.problems(cfg), [])
                self.assertEqual(
                    config.problems(
                        dict(
                            self.cfg(reviewer=choice),
                            repo="o/r",
                            rocke_path="p",
                            operator={"github_login": "op"},
                            scope={"labels": ["x"]},
                            paths={"working_dir": "/w"},
                        )
                    ),
                    [],
                )
        claude = bindings.reviewer(config.normalize(self.cfg(reviewer="claude")))
        self.assertEqual(
            (claude["provider"], claude["endpoint"], claude["model"]),
            ("claude", f"${URL_VAR}", "Model X"),
        )  # the variable's name, never its value
        self.assertNotIn(
            "none",
            bindings.reviewer_line(config.normalize(self.cfg(reviewer="default"))),
        )
        for backend, url in (("ollama", "server_url"), ("openai", "base_url")):
            blank = self.cfg(
                reviewer="worker", worker={"backend": backend, "model": "", "url": ""}
            )
            self.assertEqual(
                bindings.reviewer_line(blank),
                f"worker: the local review loop, {backend} model (model_bindings.worker.model_id not set) at "
                f"(model_bindings.worker.{url} not set)",
            )  # the deploy settings' keys, where the operator sets them

    def test_line_1_names_the_effort_the_drafter_runs_at(self):
        """Line 1 gives the effort the drafter's requests carry, the first review pass's (bindings.drafter_effort, its
        one home), and so does rk roles' first review pass row: on a Claude model the pass's own, else the
        verifier's, whatever the binding's own effort; on another model the pass's own, else none (line 1: the
        model's default); on the local worker worker.think: unset, as on the pass's default, none is sent (line 1:
        the model's default), false shows as none, and an OpenAI-compatible server gets only an effort its
        reasoning_effort takes.
        """
        local = self.cfg()["worker"]
        model = {"claude": "Model X", "gw": "GW Model", "worker": "local-model:1"}
        for choice, over, want in (
            ("claude", {}, "high"),
            ("claude", {"effort": {"verify": "xhigh"}}, "xhigh"),
            ("claude", {"effort": {"verify": "xhigh", "first_pass": "max"}}, "max"),
            ("gw", {}, ""),
            ("gw", {"effort": {"verify": "xhigh", "first_pass": "low"}}, "low"),
            ("worker", {}, ""),
            ("worker", {"worker": dict(local, think="medium")}, "medium"),
            ("worker", {"worker": dict(local, think=False)}, "none"),
            ("worker", {"worker": dict(local, backend="openai")}, ""),
            ("worker", {"worker": dict(local, backend="openai", think="high")}, "high"),
        ):
            with self.subTest(choice=choice, over=over):
                cfg = config.normalize(self.cfg(reviewer=choice, **over))
                self.assertEqual(
                    bindings.first_line(cfg),
                    "Automated review by rocke-monitor: Model - "
                    f"{model[choice]}, effort: {want or config.MODEL_DEFAULT}",
                )
                s = bindings.client_settings(cfg, "drafter")
                self.assertEqual(
                    None if s is None else s["effort"],
                    None if choice == "worker" else want,
                )  # the local worker has no binding: it sends worker.think
                row = {r[0]: r[3] for r in roles.pass_rows(cfg)}
                self.assertEqual(
                    row[roles.FIRST_REVIEW_PASS].split(" (")[0], want or "-"
                )

    def test_the_drafter_s_effort_reads_no_worker_module(self):
        """The host's rkmon works the drafter's effort out too (rkmon confirm, for the draft stage), and it reads no
        module of the worker folder: the efforts another model takes come from deploy_tree.OTHER_EFFORTS, never from
        the worker's client (first_pass.llm_client), so a run with no worker folder gives the same effort.
        """
        local = self.cfg()["worker"]
        gone = ModuleNotFoundError("No module named 'llm_client'")
        for choice, over, want in (
            ("gw", {"effort": {"verify": "xhigh", "first_pass": "low"}}, "low"),
            ("gw", {"effort": {"verify": "xhigh", "first_pass": "max"}}, ""),
            ("worker", {"worker": dict(local, backend="openai", think="high")}, "high"),
            ("worker", {"worker": dict(local, backend="openai", think="max")}, ""),
        ):
            with self.subTest(choice=choice, over=over), mock.patch.object(
                first_pass, "llm_client", side_effect=gone
            ):
                cfg = config.normalize(self.cfg(reviewer=choice, **over))
                self.assertEqual(
                    bindings.drafter_effort(cfg, bindings.binding(cfg, "drafter")),
                    want,
                )

    def test_the_older_keys_and_spellings(self):
        """worker_backend and roles.drafter are still read (their "none" is the default); `reviewer` decides when it is
        set, and setup rewrites the older keys onto it."""
        self.assertEqual(
            bindings.reviewer_choice({"worker_backend": "none"}), "default"
        )
        self.assertEqual(bindings.reviewer_choice({"reviewer": "none"}), "default")
        self.assertEqual(
            bindings.reviewer_choice(self.cfg(roles={"drafter": "gw"})), "gw"
        )
        self.assertEqual(
            bindings.reviewer_choice({"worker_backend": "qwen"}), "worker"
        )  # no such binding: the worker
        self.assertEqual(
            bindings.reviewer_choice({}), "worker"
        )  # a bare configuration in code
        # A configuration file with no reviewer key and none of the older keys: the default, unless a local worker is
        # set up (config.normalize resolves it).
        norm = lambda raw: bindings.reviewer_choice(config.normalize(raw))  # noqa: E731
        self.assertEqual(norm({}), "default")
        self.assertEqual(
            norm({"worker": {"backend": "ollama", "url": "", "model": ""}}), "default"
        )
        self.assertEqual(
            norm({"worker": {"url": "http://127.0.0.1:11434", "model": "m"}}), "worker"
        )
        self.assertEqual(norm({"worker_model": "qwen"}), "worker")
        self.assertEqual(norm({"ollama": {}}), "worker")
        self.assertEqual(norm({"worker_backend": "none"}), "default")
        self.assertEqual(norm({"reviewer": "worker"}), "worker")
        self.assertEqual(
            bindings.reviewer_choice(self.cfg(reviewer="claude", worker_backend="gw")),
            "claude",
        )
        text = "\n".join(
            bindings.problems(self.cfg(reviewer="claude", worker_backend="gw"))
        )
        self.assertIn(
            "worker_backend (gw) is an older name of reviewer (claude) and says otherwise",
            text,
        )
        new, said = bindings.migrate(
            self.cfg(worker_backend="none", models={"tick": "worker"})
        )
        self.assertEqual(
            (new["reviewer"], "worker_backend" in new, new["models"]["tick"]),
            ("default", False, "worker"),
        )
        self.assertIn("worker_backend (none) removed: reviewer decides", said)
        new, _ = bindings.migrate(
            {
                "roles": {"drafter": "gw", "quality": "gw"},
                "model_bindings": self.BINDINGS,
                "models": {"tick": "worker"},
            }
        )
        self.assertEqual(
            (new["reviewer"], new["roles"], new["models"]["tick"]),
            ("gw", {"quality": "gw"}, ""),
        )
        self.assertEqual(
            bindings.migrate({"reviewer": "none"}, full=False)[0]["reviewer"], "default"
        )
        self.assertNotIn(
            "reviewer",
            bindings.migrate({"deployment": {"project": "x"}}, full=False)[0],
        )  # a partial file: as given

    def test_the_first_pass_a_configuration_runs_is_the_one_normalize_reads(self):
        """config.first_pass_choice: the choice reviewer_choice reads in the normalized configuration, read from the
        configuration as it is."""
        for raw in (
            {},
            {"worker": {"backend": "ollama", "url": "", "model": ""}},
            {"worker": {"url": "http://127.0.0.1:11434", "model": "m"}},
            {"worker_model": "qwen"},
            {"ollama": {}},
            {"worker_backend": "none"},
            {"reviewer": "worker"},
            self.cfg(),
            self.cfg(reviewer="gw"),
            self.cfg(roles={"drafter": "gw"}),
            self.cfg(reviewer="default"),
        ):
            with self.subTest(raw=raw):
                self.assertEqual(
                    config.first_pass_choice(raw),
                    bindings.reviewer_choice(config.normalize(raw)),
                )
        self.assertEqual(
            (
                config.first_pass_choice({}),
                config.first_pass_choice(self.cfg()),
                config.first_pass_choice(self.cfg(reviewer="gw")),
            ),
            (bindings.DEFAULT, bindings.WORKER, "gw"),
        )

    def test_what_a_binding_has_for_each_key_it_leaves_out(self):
        """bindings.defaults: binding_defaults() under the defaults of the provider's adapter; an older provider name
        reads as its current one; binding_by_name fills in a binding's blanks from them.
        """
        claude = bindings.defaults("claude")
        self.assertEqual(
            (claude["auth_header"], claude["auth_scheme"], claude["chat_path"]),
            ("x-api-key", "", "/v1/messages"),
        )
        self.assertEqual(bindings.defaults("anthropic"), claude)
        for kind in ("openai-compatible", "openai", "ollama"):
            with self.subTest(kind=kind):
                self.assertEqual(bindings.defaults(kind), bindings.binding_defaults())
        self.assertIsNot(bindings.defaults("ollama"), bindings.binding_defaults())
        gw = bindings.binding_by_name(self.cfg(), "gw")
        self.assertEqual(
            (gw["auth_header"], gw["auth_scheme"], gw["stall_s"]),
            ("Authorization", "Bearer", bindings.binding_defaults()["stall_s"]),
        )

    def test_importing_the_bindings_reads_no_schema(self):
        """binding_defaults() reads the deploy schema at first use: a fresh interpreter that imports
        rkmon_lib.bindings has loaded neither the deploy tree nor the schemas' file types.
        """
        code = (
            "import sys; import rkmon_lib.bindings; "
            "print(sorted(m for m in ('rkmon_lib.deploy_tree', 'rkmon_lib.file_types') if m in sys.modules))"
        )
        run = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual((run.returncode, run.stdout.strip()), (0, "[]"), run.stderr)

    def test_what_the_checks_refuse(self):
        self.assertIn(
            "features.review_automation.passes.first_review_pass.model is 'nope', which model_bindings does not define",
            "\n".join(config.problems(self.cfg(reviewer="nope"))),
        )
        self.assertIn(
            "'worker' is a reviewer choice, not a binding's name",
            "\n".join(
                bindings.problems(
                    {"model_bindings": {"worker": {"url": "https://x.example.invalid"}}}
                )
            ),
        )
        bad = {
            "model_bindings": {
                "b": {
                    "provider": "bedrock",
                    "url": "gw.example.invalid",
                    "api_key_env": "A_KEY_VARIABLE",
                },
                "o": {"provider": "ollama", "model": "m"},
            }
        }
        text = "\n".join(bindings.problems(bad))
        for words in (
            "model_bindings.b.provider is 'bedrock'",
            "model_bindings.b.base_url is 'gw.example.invalid'",
            "model_bindings.o.server_url is not set",
            "model_bindings.b.api_key_env is not a binding setting",  # a key comes from a file alone
        ):
            self.assertIn(words, text)
        # The local worker's keys are needed only while it drafts the first pass.
        base = {
            "repo": "o/r",
            "rocke_path": "p",
            "operator": {"github_login": "op"},
            "scope": {"labels": ["x"]},
            "paths": {"working_dir": "/w"},
        }
        self.assertEqual(config.problems(dict(base, reviewer="default")), [])
        self.assertEqual(
            config.problems(base), []
        )  # a file that names no first pass and sets up no local worker: the default, as normalize reads it
        self.assertEqual(
            [
                p.split(" is blank")[0]
                for p in config.problems(dict(base, reviewer="worker"))
            ],
            ["model_bindings.worker.model_id", "model_bindings.worker"],
        )
        # A line 1 set in the configuration, which names another model than the reviewer's, is not read: line 1
        # follows the reviewer and gives the automated reviewer name, so there is no stale line to name.
        stale = config.normalize(
            self.cfg(
                reviewer="claude",
                review_first_line="Automated review: Model - local-model:1, effort: max",
            )
        )
        self.assertEqual(
            (stale["review_first_line"], bindings.first_line(stale)),
            ("Automated review by rocke-monitor: Model - Model X, effort: high",) * 2,
        )
        self.assertNotIn("review_first_line", "\n".join(bindings.problems(stale)))


# --- rk doctor and rk features: the chosen reviewer only ----------------------------------------------------------------


class TestDoctorAndFeatures(_Fake):
    def setUp(self):
        super().setUp()
        unset = mock.patch.dict(
            os.environ, {URL_VAR: ""}
        )  # the binding's address's variable is not set where the monitor runs, until a test sets it
        unset.start()
        self.addCleanup(unset.stop)

    def cfg(self, **over) -> dict:
        return dict(
            {
                "model_bindings": {"claude": self.binding()},
                "reviewer": "default",
                "repo": "o/r",
                "rocke_path": "p",
                "operator": {"github_login": "op"},
                "scope": {"labels": ["x"]},
                "paths": {
                    "working_dir": str(self.base / "w"),
                    "secrets_dir": str(self.secrets),
                },
                "features": {},
            },
            **over,
        )

    def test_the_reviewer_check_for_each_choice(self):
        c = doctor.reviewer_check(self.cfg())
        self.assertEqual(
            (c.status, c.found),
            (
                "PASS",
                "default: the verifier (Claude Opus) reviews alone; nothing to "
                "check",
            ),
        )
        c = doctor.reviewer_check(self.cfg(reviewer="claude"))
        self.assertEqual(c.status, "FAIL")  # its address's variable is not set here
        self.assertIn(
            f"integrations.claude_code.base_url (its address, which '${URL_VAR}' gives through a variable that is "
            "not set where the monitor runs)",
            c.found,
        )
        self.assertTrue(
            str(c.fix).startswith(
                "set integrations.claude_code.base_url in your deploy settings file to its address, which "
            ),
            c.fix,
        )
        with mock.patch.dict(os.environ, {URL_VAR: self.srv.url}):
            c = doctor.reviewer_check(self.cfg(reviewer="claude"))
            self.assertEqual(c.status, "PASS", c.found)
            self.assertNotIn(KEY, f"{c.found} {c.fix}")
        with mock.patch.object(
            doctor,
            "model_check",
            return_value=doctor.Check("reviewer model", "PASS", "m"),
        ):
            self.assertEqual(
                doctor.reviewer_check(self.cfg(reviewer="worker")).name,
                "reviewer (the local review loop)",
            )
        c = doctor.reviewer_check(self.cfg(reviewer="gone"))
        self.assertEqual(
            (c.status, c.found),
            (
                "FAIL",
                "features.review_automation.passes.first_review_pass.model is gone, which model_bindings does not define",
            ),
        )

    def test_ollama_is_a_prerequisite_only_while_the_local_worker_is_used(self):
        self.assertEqual(doctor.worker_users(self.cfg()), [])
        self.assertEqual(doctor.worker_users(self.cfg(reviewer="claude")), [])
        self.assertEqual(
            doctor.worker_users(self.cfg(reviewer="worker")), ["the reviewer"]
        )
        self.assertEqual(
            doctor.worker_users(self.cfg(models={"tick": "worker"})),
            ["the tick triage"],
        )
        gh = {"found": False, "config_dir": "gh's own", "binary": "gh"}
        with mock.patch.object(
            doctor, "run", return_value=(127, "")
        ), mock.patch.object(doctor, "gh_status", return_value=gh), mock.patch.object(
            doctor, "claude_code", return_value=("", "")
        ):
            checks = {c.name: c for c in doctor.prerequisites(self.cfg())}
        ollama = next(
            c for name, c in checks.items() if name.lower().startswith("ollama")
        )
        self.assertEqual(ollama.status, "SKIP")
        self.assertIn(
            "nothing uses the local review loop (reviewer default)", ollama.found
        )

    def test_rk_features_reports_the_reviewer(self):
        said = features.configured_with(self.cfg(), "review-drafts")
        self.assertTrue(
            said.startswith(
                "first pass default: the verifier (Claude Opus) reviews alone"
            ),
            said,
        )
        with mock.patch.dict(os.environ, {URL_VAR: self.srv.url}):
            cfg = self.cfg(reviewer="claude")
            self.assertEqual(features.missing(cfg, "review-drafts"), [])
            said = features.configured_with(cfg, "review-drafts")
        self.assertIn(f"first pass claude: Model X via claude at ${URL_VAR}", said)
        self.assertIn(
            "line 1: Automated review by rocke-monitor: Model - Model X, effort: high",
            said,
        )
        self.assertNotIn(KEY, said)


# --- the tick triage: the rule filter by default, a model an opt-in, never the reviewer ----------------------------------


class TestTickTriage(_Fake):
    def cfg(self, **over) -> dict:
        return dict(
            {
                "model_bindings": {
                    "claude": self.binding(),
                    "small": self.binding(model="small-1", name="Small"),
                },
                "reviewer": "claude",
                "paths": {
                    "working_dir": str(self.base / "w"),
                    "root": str(self.base),
                    "secrets_dir": str(self.secrets),
                },
                "drafter": {"limiter_dir": str(self.base / "limits")},
            },
            **over,
        )

    def test_the_rule_filter_unless_a_model_is_named(self):
        for reviewer in ("default", "worker", "claude"):
            with self.subTest(reviewer=reviewer):
                cfg = self.cfg(
                    reviewer=reviewer,
                    worker={"model": "m", "url": "http://127.0.0.1:9"},
                )
                self.assertEqual(
                    config.model_for(cfg, "tick")["by"], "none"
                )  # never the reviewer by itself
                self.assertEqual(triage.describe(cfg), triage.RULE_FILTER)
        self.assertEqual(
            config.model_for(self.cfg(models={"tick": "worker"}), "tick")["by"],
            "worker",
        )
        named = self.cfg(models={"tick": "small"})
        self.assertEqual(
            (
                config.model_for(named, "tick")["by"],
                config.model_for(named, "tick")["model"],
            ),
            ("binding", "small-1"),
        )
        self.assertIn(
            "the package's models.tick names 'gone', which model_bindings does not define",
            config.model_for(self.cfg(models={"tick": "gone"}), "tick")["why"],
        )
        self.assertIn(
            "the package's models.tick is 'gone'",
            "\n".join(config.problems(self.cfg(models={"tick": "gone"}))),
        )
        # A named model that is not at hand (its address's variable not set here): the rule filter decides meanwhile,
        # and says why.
        with mock.patch.dict(os.environ, {URL_VAR: ""}):
            self.assertIn(
                "the triage model waits: integrations.claude_code.base_url (its address, which",
                triage.describe(named),
            )
        self.assertEqual(triage.describe(named), "the binding small (Small, anthropic)")

    def test_a_triage_model_on_an_anthropic_binding(self):
        """models.tick naming an anthropic binding: its loop through the Messages API, the turn sent back with its own
        blocks (the tool_use id the tool result answers), its tokens; no Ollama anywhere.
        """
        verdict = {
            "summary": "rocke-monitor 10:20Z",
            "unsure": [],
            "claude": [],
            "user": [],
            "done": ["rk board"],
        }

        def answer(body):
            if not any(
                b.get("type") == "tool_result"
                for m in body["messages"]
                for b in m["content"]
            ):
                return {
                    "tool_calls": [
                        {
                            "id": "toolu_t1",
                            "function": {
                                "name": "rk",
                                "arguments": {"args": ["board"]},
                            },
                        }
                    ]
                }
            return {"content": json.dumps(verdict)}

        self.srv.answer = answer
        cfg = self.cfg(models={"tick": "small"})
        self.assertEqual(triage.unavailable(cfg), "")
        ran = []
        v = triage.run(
            cfg,
            0,
            [("TICK x", triage.QUIET)],
            "",
            [],
            rk=lambda args: ran.append(args) or "| ok |",
        )
        self.assertEqual(
            (v["summary"], ran, v["requests"]), ("rocke-monitor 10:20Z", [["board"]], 2)
        )
        self.assertGreater(v["tokens"][0], 0)
        second = self.srv.bodies[1]
        assistant = next(m for m in second["messages"] if m["role"] == "assistant")
        self.assertEqual(
            [b["type"] for b in assistant["content"]], ["thinking", "tool_use"]
        )  # its own blocks
        result = next(
            b
            for m in second["messages"]
            for b in m["content"]
            if b.get("type") == "tool_result"
        )
        self.assertEqual(result["tool_use_id"], "toolu_t1")
        self.assertEqual({h["x-api-key"] for h in self.srv.headers}, {KEY})


# --- the Anthropic Messages adapter --------------------------------------------------------------------------------------


class TestAnthropicMessages(_Fake):
    def test_a_tool_loop_streamed(self):
        def answer(body):
            if len(body["messages"]) == 1:
                return {
                    "content": "Looking.",
                    "tool_calls": [
                        {
                            "id": "toolu_1",
                            "function": {
                                "name": "lookup",
                                "arguments": {"word": "rocke"},
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 400},
                }
            return {"content": "It is a kernel provider."}

        self.srv.answer, self.srv.cached = answer, 300
        s = self.settings()
        lim = self.limiter(s)
        msgs = [
            {"role": "system", "content": "You answer briefly."},
            {"role": "user", "content": "What is rocke?"},
        ]
        a = lc.chat(s, msgs, tools=[TOOL], effort="low", limiter=lim)
        self.assertEqual(
            (a["content"], a["finish_reason"], a["reasoning"]),
            ("Looking.", "tool_calls", "considered it"),
        )
        self.assertEqual(
            a["tool_calls"],
            [
                {
                    "id": "toolu_1",
                    "function": {"name": "lookup", "arguments": {"word": "rocke"}},
                }
            ],
        )
        self.assertEqual(
            (a["usage"]["prompt"], a["usage"]["cached"], a["usage"]["completion"]),
            (400, 300, 20),
        )
        body, hdr = self.srv.bodies[0], self.srv.headers[0]
        self.assertEqual(
            (self.srv.paths[0], body["model"], body["system"], body["stream"]),
            ("/v1/messages", "model-x", "You answer briefly.", True),
        )
        self.assertEqual(
            (
                body["tools"][0]["name"],
                body["tools"][0]["input_schema"]["required"],
                body["tool_choice"],
                body["output_config"],
            ),
            ("lookup", ["word"], {"type": "auto"}, {"effort": "low"}),
        )
        self.assertNotIn(
            "temperature", body
        )  # current models refuse a sampling setting
        self.assertNotIn("thinking", body)  # the model's own default
        self.assertEqual(
            (hdr["x-api-key"], hdr["anthropic-version"]), (KEY, "2023-06-01")
        )
        self.assertNotIn("authorization", hdr)
        msgs += [
            {
                "role": "assistant",
                "content": a["content"],
                "tool_calls": a["tool_calls"],
            },
            {
                "role": "tool",
                "tool_name": "lookup",
                "content": "rocKE: a kernel provider",
            },
        ]
        b = lc.chat(s, msgs, tools=[TOOL], limiter=lim)
        self.assertEqual(
            (b["content"], b["finish_reason"], b["tool_calls"]),
            ("It is a kernel provider.", "stop", []),
        )
        sent = self.srv.bodies[1]["messages"]
        self.assertEqual([m["role"] for m in sent], ["user", "assistant", "user"])
        self.assertEqual(
            sent[2]["content"][0],
            {
                "type": "tool_result",
                "tool_use_id": "toolu_1",
                "content": "rocKE: a kernel provider",
            },
        )
        self.assertEqual(lim.summary()["requests"], 2)

    def test_a_whole_answer_a_bearer_gateway_and_the_url_from_the_environment(self):
        s = self.settings(stream=False)
        with mock.patch.dict(os.environ, {URL_VAR: self.srv.url}):
            self.assertEqual(
                lc.chat(
                    s, [{"role": "user", "content": "hi"}], limiter=self.limiter(s)
                )["content"],
                "done",
            )
        self.assertNotIn("stream", self.srv.bodies[0])

    def test_the_key_never_shows_and_a_429_waits(self):
        """The key is read from its file at each request, and never shows: a refused one, nor an empty file's."""
        s = self.settings()
        self.key.write_text("k-wrong-0001\n", encoding="utf-8")
        with self.assertRaises(lc.GatewayError) as e:
            lc.chat(
                s,
                [{"role": "user", "content": "hi"}],
                limiter=self.limiter(s),
                sleep=lambda _s: None,
            )
        self.assertEqual(e.exception.status, 401)
        self.assertIn("invalid x-api-key <key>", str(e.exception))
        self.assertNotIn("k-wrong-0001", str(e.exception))
        self.key.write_text(f"{KEY}\n", encoding="utf-8")
        self.srv.refuse_429 = 1
        lim = self.limiter(s)
        self.assertEqual(
            lc.chat(s, [{"role": "user", "content": "hi"}], limiter=lim)["content"],
            "done",
        )
        self.assertEqual(lim.summary()["rate_limited"], 1)
        self.key.write_text("", encoding="utf-8")
        with self.assertRaises(lc.GatewayError) as e:
            lc.chat(s, [{"role": "user", "content": "hi"}], limiter=self.limiter(s))
        self.assertIn(
            f"no key in the binding's api_key_file ({self.key})", str(e.exception)
        )

    def test_thinking_blocks_the_endpoint_refuses_go_once_more_without_them(self):
        self.srv.refuse_thinking = True
        s = self.settings()
        first = lc.chat(s, [{"role": "user", "content": "hi"}], limiter=self.limiter(s))
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "x", "blocks": first["blocks"]},
            {"role": "user", "content": "and then?"},
        ]
        self.assertEqual(lc.chat(s, msgs, limiter=self.limiter(s))["content"], "done")
        kinds = [
            [b["type"] for b in m["content"]]
            for m in self.srv.bodies[-1]["messages"]
            if m["role"] == "assistant"
        ]
        self.assertEqual(kinds, [["text"]])  # the resend: the thinking block left out

    def test_the_messages_shape(self):
        system, out = lc.to_anthropic(
            [
                {"role": "system", "content": "S"},
                {"role": "user", "content": "U"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "call.1",
                            "function": {"name": "a", "arguments": '{"x": 1}'},
                        },
                        {"id": "", "function": {"name": "b", "arguments": {}}},
                    ],
                },
                {"role": "tool", "tool_name": "a", "content": "ra"},
                {"role": "tool", "tool_name": "b", "content": "rb"},
                {"role": "user", "content": "next"},
            ]
        )
        self.assertEqual(system, "S")
        self.assertEqual([m["role"] for m in out], ["user", "assistant", "user"])
        ids = [b["id"] for b in out[1]["content"]]
        self.assertEqual(ids, ["toolu_rk0", "toolu_rk1"])  # ids the API takes
        self.assertEqual(
            [(b["type"], b.get("tool_use_id")) for b in out[2]["content"]],
            [
                ("tool_result", "toolu_rk0"),
                ("tool_result", "toolu_rk1"),
                ("text", None),
            ],
        )
        self.assertEqual(out[1]["content"][0]["input"], {"x": 1})


class TestAClaudeBindingAsTheSession(_Fake):
    """A claude binding that names no address or key of its own is reached as the Claude session is: at
    integrations.claude_code's address, with the claude login's key in the headers the session sends it in. Nothing
    of the session's own variables in the environment is read, and the key is in no setting.
    """

    def test_it_sends_the_key_as_the_session_does(self):
        from rkmon_lib import orchestrator

        secrets = self.secrets
        (secrets / "claude_key").write_text(f"{KEY}\n", encoding="utf-8")
        cfg = {
            "model_bindings": {"c": {"provider": "claude", "model": "model-x"}},
            "reviewer": "c",
            "drafter": {"limiter_dir": str(self.base / "limits")},
            "paths": {
                "root": str(self.base),
                "working_dir": str(self.base / "w"),
                "secrets_dir": str(secrets),
            },
            "claude": {"gateway": self.srv.url, "api_key_header": "X-Gateway-Key"},
        }
        with mock.patch.dict(
            os.environ,
            {
                "ANTHROPIC_BASE_URL": "http://127.0.0.1:9",
                "ANTHROPIC_API_KEY": "k-" + "session-env-not-read",
                "ANTHROPIC_CUSTOM_HEADERS": "X-Other: not-read",
            },
        ):
            s = bindings.client_settings(cfg, "drafter")
            self.assertNotIn(KEY, json.dumps(s))
            got = lc.chat(
                s, [{"role": "user", "content": "hi"}], limiter=self.limiter(s)
            )
        self.assertEqual(got["content"], "done")
        self.assertEqual(self.srv.paths, ["/v1/messages"])
        sent = self.srv.headers[0]
        want = orchestrator.request_headers(orchestrator.session_variables(cfg, KEY))
        self.assertEqual(
            want, {"x-api-key": KEY, "X-Gateway-Key": KEY}
        )  # the session's: its key variable's header and the named one
        self.assertEqual(
            {k.lower(): sent.get(k.lower()) for k in want},
            {k.lower(): v for k, v in want.items()},
        )
        self.assertNotIn("x-other", sent)

    def test_the_deploy_side_makes_it_with_no_worker_folder(self):
        """The deploy side runs on the host from a package folder that holds no worker/: the claude drafter's binding
        and line 1 come from the package's own code there, the key's headers with its mark in the key's place, and no
        folder named worker goes on the import path."""
        probe = (
            "import json, pathlib, sys\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "from rkmon_lib import bindings\n"
            "cfg = json.loads(sys.argv[2])\n"
            "b = bindings.binding(cfg, 'drafter')\n"
            "on_path = [p for p in sys.path if pathlib.Path(p).name == 'worker']\n"
            "print(json.dumps([b['key_headers'], bindings.drafter_line(cfg), on_path]))\n"
        )
        cfg = {
            "model_bindings": {"c": {"provider": "claude", "model": "model-x"}},
            "reviewer": "c",
            "claude": {
                "gateway": "https://gateway.example.com",
                "api_key_header": "X-Gateway-Key",
            },
        }
        with tempfile.TemporaryDirectory() as d:
            for folder in ("rkmon_lib", "schema"):
                shutil.copytree(
                    ROOT / folder,
                    pathlib.Path(d) / folder,
                    ignore=shutil.ignore_patterns("__pycache__"),
                )
            for name in ("config.defaults.json", "VERSION"):
                shutil.copy(ROOT / name, pathlib.Path(d) / name)
            p = subprocess.run(
                [sys.executable, "-I", "-B", "-c", probe, d, json.dumps(cfg)],
                cwd=d,
                capture_output=True,
                text=True,
                timeout=60,
            )
        self.assertEqual((p.returncode, p.stderr), (0, ""))
        headers, line, on_path = json.loads(p.stdout)
        mark = custom_headers.KEY_MARK
        self.assertEqual(headers, {"x-api-key": mark, "X-Gateway-Key": mark})
        self.assertIn("model-x", line)
        self.assertEqual(on_path, [])


class TestAnthropicDrafter(_Fake):
    """The tool-loop drafter (worker/drafter.py) on an Anthropic binding: the same phased loop, tools and evidence as on
    an OpenAI-compatible one, through the Messages API."""

    def test_a_whole_draft_through_the_messages_api(self):
        ep = fx.FakeEndpoint(dict(PRIOR, hunt=HUNT))
        self.srv.answer = lambda body: ep(None, neutral_view(body), "")
        self.srv.cached = 100
        f = fx.make(self.base / "pr")
        cfg = dict(
            f["cfg"],
            reviewer="cand",
            model_bindings={
                "cand": self.binding(model="Cand-1", name="Cand-1", effort="high")
            },
        )
        cfg.pop("worker_backend")
        cfg["paths"] = dict(cfg["paths"], secrets_dir=str(self.secrets))
        s = bindings.client_settings(cfg, "drafter")

        def chat(s_, msgs, tools, effort, limiter=None, max_tokens=None):
            return lc.chat(
                s_,
                msgs,
                tools,
                effort,
                limiter=limiter,
                max_tokens=max_tokens,
                sleep=lambda _s: None,
            )

        session = drafter.Session(f["ws"], 7, cfg, s, chat=chat)
        self.assertTrue(session.run_draft())
        text = (f["ws"] / "review_agent.md").read_text(encoding="utf-8")
        self.assertEqual(text.splitlines()[0], fx.LINE1)
        self.assertIn("| N1 | 🟠 | `x.py:2` | **answer returns 41, not 42**", text)
        self.assertIn(
            "_First-round review: first pass by Cand-1, verified by the orchestrator",
            text,
        )
        tot = json.loads((f["ws"] / "tokens_total.json").read_text(encoding="utf-8"))[
            "attempts"
        ][-1]
        self.assertEqual((tot["requests"], tot["cached_tokens"]), (10, 1000))
        self.assertEqual(set(self.srv.paths), {"/v1/messages"})
        self.assertEqual({h["x-api-key"] for h in self.srv.headers}, {KEY})
        for name in (
            "review_agent.md",
            "agent_log.txt",
            "draft_run.json",
            "tokens_total.json",
        ):
            self.assertNotIn(KEY, (f["ws"] / name).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
