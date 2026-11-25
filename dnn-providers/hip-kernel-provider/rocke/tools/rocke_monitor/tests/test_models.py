# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The two model roles: a reviewer model and an orchestrator model. The reviewer model is on an Ollama server or any
OpenAI-compatible endpoint, a gateway behind an API-management layer included (a subscription-key header, a route
prefix, model ids with capitals and dots); rk provision checks it (models.check) with a test chat and a tool-call probe
shaped like the review loop's requests, and its row names what fails. The orchestrator model is Claude's alone: no
setting routes verification, gates or posting to the reviewer model. Every endpoint here is a fake on 127.0.0.1; no
real model is called. Every fake value is built at run time.
"""

import http.server
import json
import os
import pathlib
import sys
import tempfile
import threading
import unittest
import urllib.parse
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(HERE / "fakes"))
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import fake_servers  # noqa: E402
import llm_client  # noqa: E402
import reviewer_client  # noqa: E402
from rkmon_lib import (
    config,
    deploy_tree,
    doctor,
    features,
    models,
    privacy,
    review,
    summary,
)  # noqa: E402

KEY = "k-" + "fake-subscription-0123"  # a planted key, built at run time
HEADER = "X-Fake-Subscription-Key"
GOOD, TEXT_ONLY = "Big.Model-v5.3-Flash", "Text.only-Model"
ORCH = "Other.Agent-Model-v2"  # an orchestrator model outside the Claude Opus and Sonnet families


class Gateway(http.server.BaseHTTPRequestHandler):
    """An OpenAI-compatible gateway behind an API-management layer: under a route prefix, the key only in its own
    header (a bearer token gets 401), a LiteLLM-style model-info route, one model that calls tools and one that only
    talks. `fail429` answers that many chat requests with 429 and a Retry-After first.
    """

    prefix, info, fail429, retry_after = "/route", True, 0, "1"
    seen: list = []

    def log_message(self, *a):
        pass

    def send(self, status, body, ctype="application/json", headers=()):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        for k, v in headers:
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        if self.headers.get(HEADER) == KEY:
            return True
        self.send(
            401,
            {
                "error": {
                    "message": "Access denied: missing or invalid subscription key"
                }
            },
        )
        return False

    def route(self):
        path = urllib.parse.urlsplit(self.path).path
        return path[len(self.prefix) :] if path.startswith(self.prefix + "/") else None

    def do_GET(self):
        Gateway.seen.append(("GET", self.path, dict(self.headers), None))
        r = self.route()
        if r is None:
            return self.send(404, {"error": "no such route"})
        if not self.authorized():
            return
        if r == "/v1/models":
            # vLLM's form: the context length in the model's list entry
            return self.send(
                200,
                {
                    "object": "list",
                    "data": [{"id": GOOD, "max_model_len": 131072}, {"id": TEXT_ONLY}],
                },
            )
        if r in ("/v1/model/info", "/model/info") and Gateway.info:
            return self.send(
                200,
                {
                    "data": [
                        {
                            "model_name": GOOD,
                            "litellm_params": {"rpm": 60, "tpm": 100000},
                            "model_info": {
                                "max_input_tokens": 131072,
                                "max_output_tokens": 8192,
                                "supports_function_calling": True,
                            },
                        }
                    ]
                },
            )
        self.send(404, {"error": "not found"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Gateway.seen.append(("POST", self.path, dict(self.headers), body))
        r = self.route()
        if r != "/v1/chat/completions":
            return self.send(404, {"error": "no such route"})
        if not self.authorized():
            return
        if Gateway.fail429 > 0:
            Gateway.fail429 -= 1
            return self.send(
                429,
                {"error": "rate limited"},
                headers=(
                    [("Retry-After", Gateway.retry_after)]
                    if Gateway.retry_after
                    else []
                ),
            )
        if not body.get("stream"):
            return self.send(200, {"choices": [{"message": {"content": "OK"}}]})
        if body.get("tools") and body["model"] == GOOD:
            delta = {
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
        else:
            delta = {"content": "I would read README.md first."}
        data = f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\ndata: [DONE]\n\n".encode()
        self.send(200, data, "text/event-stream")


def serve(handler):
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


class _Base(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv, cls.url = serve(Gateway)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        Gateway.seen.clear()
        Gateway.prefix, Gateway.info, Gateway.fail429, Gateway.retry_after = (
            "/route",
            True,
            0,
            "1",
        )
        self.tmp = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.tmp.name)
        self.keyfile = self.base / "worker_api_key"
        self.keyfile.write_text(KEY + "\n", encoding="utf-8")
        self.keyfile.chmod(0o600)

    def tearDown(self):
        self.tmp.cleanup()

    def s(self, **over) -> dict:
        return dict(
            dict(
                config.worker_defaults(),
                backend="openai",
                url=self.url + "/route",
                model=GOOD,
                api_key_file=str(self.keyfile),
                auth_header=HEADER,
                auth_scheme="",
            ),
            **over,
        )


class TestGateway(_Base):
    def test_a_route_prefix_is_kept_and_v1_is_never_doubled(self):
        u = urllib.parse.urlsplit
        self.assertEqual(
            reviewer_client._path(u("https://gw.example.com/route"), "/v1/models"),
            "/route/v1/models",
        )
        self.assertEqual(
            reviewer_client._path(u("https://gw.example.com/route/v1"), "/v1/models"),
            "/route/v1/models",
        )
        self.assertEqual(
            reviewer_client._path(
                u("https://gw.example.com/route/v1/"), "/v1/chat/completions"
            ),
            "/route/v1/chat/completions",
        )
        self.assertEqual(
            reviewer_client._path(u("https://gw.example.com"), "/v1/models"),
            "/v1/models",
        )
        self.assertEqual(
            reviewer_client._path(u("http://127.0.0.1:11434"), "/api/tags"), "/api/tags"
        )
        for base in (self.url + "/route", self.url + "/route/v1"):
            with self.subTest(base=base):
                listed, err, _where = models.openai_list(self.s(url=base))
                self.assertEqual(err, "")
                self.assertEqual(
                    [m["name"] for m in listed], [GOOD, TEXT_ONLY]
                )  # case and dots kept
        self.assertTrue(
            all(path.startswith("/route/v1/") for _m, path, _h, _b in Gateway.seen),
            Gateway.seen,
        )

    def test_the_key_goes_in_the_gateways_own_header_and_a_bearer_is_refused_clearly(
        self,
    ):
        listed, err, _where = models.openai_list(self.s())
        self.assertEqual(err, "")
        _m, _p, headers, _b = Gateway.seen[-1]
        self.assertEqual(headers.get(HEADER), KEY)
        self.assertNotIn("Authorization", headers)
        listed, err, _where = models.openai_list(
            self.s(auth_header="Authorization", auth_scheme="Bearer")
        )
        self.assertEqual(listed, [])
        self.assertIn("HTTP 401", err)
        self.assertIn("model_bindings.worker.api_key_header", err)
        self.assertIn("Authorization: Bearer <key>", err)
        self.assertNotIn(KEY, err)
        rec = models.check(self.s(auth_header="Authorization", auth_scheme="Bearer"))
        self.assertIn("HTTP 401", rec["chat"])
        self.assertTrue(
            config.probe_failure(dict(self.s(), probe=rec)).startswith(
                "failed the test chat"
            )
        )

    def test_the_key_comes_from_its_file_alone_and_no_other_header_is_sent(self):
        """api_key_env, headers_file and header_env are keys no settings file has (rkmon_lib/removed_keys.py): the key
        comes from its secret's file, never from a variable, and a binding sends no extra headers. A configuration that
        still names them sends neither, and is not refused for them, nor for worker.source.
        """
        hdr = self.base / "worker_headers"
        hdr.write_text(f"{HEADER}: {KEY}\nX-Route: blue\n", encoding="utf-8")
        old = {
            "api_key_file": "",
            "api_key_env": "FAKE_GW_KEY",
            "headers_file": str(hdr),
            "header_env": {"X-Route": "FAKE_GW_ROUTE"},
        }
        with mock.patch.dict(os.environ, {"FAKE_GW_KEY": KEY, "FAKE_GW_ROUTE": "blue"}):
            listed, err, _where = models.openai_list(self.s(**old))
        self.assertEqual(listed, [])
        self.assertIn("HTTP 401", err)
        _m, _p, headers, _b = Gateway.seen[-1]
        self.assertFalse({HEADER, "X-Route"} & set(headers), headers)
        self.assertEqual(models.openai_list(self.s())[1], "")  # the key file's key
        bad = dict(
            self.s(**old),
            api_key_env="a key",
            header_env={"X Route": "a value"},
            source="nope",
        )
        said = "\n".join(config.problems({"repo": "acme/widgets", "worker": bad}))
        for key in ("api_key_env", "header_env", "headers_file", "source"):
            self.assertNotIn(f"worker.{key}", said)

    def test_the_checks_send_the_settings_given_and_read_no_model_info_route(self):
        """The tool-call probe sends the settings given (the reviews' max output), in the review loop's tool shape;
        the record keeps the context the model list reports, and no model-info route is read, though the endpoint
        answers one."""
        rec = models.check(self.s(max_output_tokens=12000))
        self.assertEqual(
            (rec["chat"], rec["tools"], rec["context"]), ("ok", "ok", 131072)
        )
        self.assertFalse({"max_output", "rpm", "tpm"} & set(rec), rec)
        probe = [b for _m, p, _h, b in Gateway.seen if b and b.get("tools")][-1]
        self.assertEqual(probe["max_tokens"], 12000)  # the max output given
        self.assertEqual(
            probe["tools"], [reviewer_client.PROBE_TOOL]
        )  # the review loop's tool shape
        self.assertEqual(probe["tool_choice"], "auto")
        self.assertEqual(
            [p for _m, p, _h, _b in Gateway.seen if "/model/info" in p], []
        )

    def test_the_record_shows_an_address_without_a_user_or_a_password(self):
        """A model list that gets no answer, with no error to say why, is recorded as no answer from the address
        config.safe_url shows: never a user or a password in it."""
        probed = dict(answers=False, models=[], model_found=False, context=None)
        rec = models.probe_record(
            self.s(url="http://someone:pw-example@127.0.0.1:1/route"),
            dict(probed, latency_s=None, error="", models_url="", chat_path=""),
        )
        self.assertEqual(rec["chat"], "no answer from http://127.0.0.1:1/route")

    def test_a_model_that_cannot_call_tools_is_refused_for_the_reviewer_role(self):
        rec = models.check(self.s(model=TEXT_ONLY))
        self.assertEqual(rec["chat"], "ok")
        self.assertIn("answered in text, with no tool call", rec["tools"])
        cfg = {
            "worker": dict(self.s(model=TEXT_ONLY), probe=rec),
            "deployment": {"phase": "review"},
            "paths": {"working_dir": "/w"},
        }
        why = config.probe_failure(config.worker(cfg))
        self.assertTrue(why.startswith("failed the tool-call probe"), why)
        gone = features.missing(cfg, "review-drafts")
        self.assertTrue(
            any("passes its checks" in g and TEXT_ONLY in g for g in gone), gone
        )
        table = "\n".join(summary.features_table(cfg))
        self.assertRegex(
            table,
            r"\| Reviews \| on \(incomplete\) \| features.review_automation.enabled is true \|[^|]*\| needs "
            r"model_bindings.worker.model_id \(a model that passes its checks: ",
        )
        self.assertEqual(
            review._fill(cfg, {"prs": {}, "review_ready": []}, None, []), []
        )  # nothing starts with it
        with mock.patch.object(doctor, "worker_running", return_value=False):
            self.assertEqual(doctor.model_check(cfg).status, "FAIL")
        cfg["worker"][
            "model"
        ] = GOOD  # the record was of another model: not held against this one
        self.assertEqual(config.probe_failure(config.worker(cfg)), "")

    def test_a_429_during_the_start_s_checks_is_waited_out_or_left_unchecked_never_refused(
        self,
    ):
        """One HTTP 429 during the start's checks does not refuse the model nor leave review-drafts incomplete until
        they run again. A 429 is waited out (its Retry-After); one that persists, or asks for too long, leaves the
        model not checked, which refuses nothing, and says that the next start checks it again.
        """
        Gateway.fail429, Gateway.retry_after = (
            1,
            "1",
        )  # the test chat's first try: waited out
        rec = models.check(self.s())
        self.assertEqual((rec["chat"], rec["tools"]), ("ok", "ok"))
        Gateway.fail429, Gateway.retry_after = 99, "1"  # it persists past the waits
        rec = models.check(self.s())
        self.assertTrue(
            rec["chat"].startswith(
                "not checked: the endpoint answered the test prompt with HTTP 429"
            ),
            rec["chat"],
        )
        self.assertEqual(models.not_checked(rec), rec["chat"])
        self.assertTrue(
            rec["chat"].endswith("the next start checks it again"), rec["chat"]
        )
        cfg = {
            "worker": dict(self.s(), probe=rec),
            "deployment": {"phase": "review"},
            "paths": {"working_dir": "/w"},
        }
        self.assertEqual(config.probe_failure(config.worker(cfg)), "")  # not refused
        self.assertEqual(features.missing(cfg, "review-drafts"), [])
        import time

        Gateway.fail429, Gateway.retry_after = (
            99,
            "120",
        )  # asks for longer than the checks wait: no wait at all
        t0 = time.monotonic()
        rec = models.check(self.s())
        self.assertLess(time.monotonic() - t0, 10)
        self.assertIn("asked us to wait 120 s", rec["chat"])
        Gateway.fail429, Gateway.retry_after = (
            99,
            "1",
        )  # the tool-call probe alone: not checked either
        t = reviewer_client.probe_tools(self.s())
        self.assertIsNone(t["ok"])
        self.assertTrue(
            t["error"].startswith(
                "not checked: the endpoint answered the tool-call probe with HTTP 429"
            )
        )
        Gateway.fail429 = 0


class TestOllama(unittest.TestCase):
    def test_this_host_s_ollama_has_one_home(self):
        """This host's Ollama on Ollama's own port is models.OLLAMA_DEFAULT, built from models.OLLAMA_PORT: an older
        configuration's reviewer runs on it, and no other module writes the port."""
        there = "http://model-host.example.com:8000"
        with mock.patch.object(models, "OLLAMA_DEFAULT", there):
            self.assertEqual(
                config.worker_settings({"worker_model": "m"})["url"], there
            )
        # privacy.py lists the port among the numbers that are never a pull request, read from models.py too
        port = str(models.OLLAMA_PORT)
        self.assertIn(port, privacy.SAFE_NUMBERS)
        written = sorted(
            p.name
            for p in (ROOT / "rkmon_lib").glob("*.py")
            if p.name != "models.py" and port in p.read_text(encoding="utf-8")
        )
        self.assertEqual(written, [])

    def test_an_ollama_server_passes_the_probe(self):
        with tempfile.TemporaryDirectory() as d:
            urls, stop = fake_servers.start(pathlib.Path(d))
            try:
                s = dict(
                    config.worker_defaults(),
                    backend="ollama",
                    url=urls["model"],
                    model="fake-model:1",
                )
                with mock.patch.object(
                    reviewer_client, "journal_progress", lambda since: (None, None)
                ):
                    rec = models.check(s)
                self.assertEqual(
                    (rec["chat"], rec["tools"], rec["context"]), ("ok", "ok", 32768)
                )
            finally:
                stop()


# --- a gateway with its own routes (every value here is invented) ------------

DEPLOY, SHOWN = (
    "example-chat-1-flash",
    "Example-Chat-1-Flash",
)  # the ID requests send, and the name the list shows
MIXED = "Mixed.Case-Model-2"  # an ID with capitals: sent as listed, though answers echo it in lower case
UNIFIED = "Unified-Model"  # served only in a simplified, non-OpenAI answer shape


class RouteGateway(http.server.BaseHTTPRequestHandler):
    """A gateway route that is not /v1-shaped: chat at <host>/api/chat/completions only, models at the host's own
    /models (outside the /api prefix), each with a deployment ID and a display name; the key wanted twice (its own
    header, raw, and a bearer token). One model answers as vLLM does (an extra `reasoning` field, usage with reasoning
    tokens, stop_reason, token_ids, routed_experts; the model echoed in lower case), one only in a "unified" shape
    with no choices. `whole`: answer with one JSON body even when a stream is asked for.
    """

    seen: list = []
    whole = False

    def log_message(self, *a):
        pass

    def send(self, status, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def authorized(self):
        if (
            self.headers.get(HEADER) == KEY
            and self.headers.get("Authorization") == f"Bearer {KEY}"
        ):
            return True
        self.send(
            401,
            {
                "error": {
                    "message": "both the subscription key and the bearer token are needed"
                }
            },
        )
        return False

    def do_GET(self):
        RouteGateway.seen.append(("GET", self.path, dict(self.headers), None))
        if self.path != "/models":
            return self.send(404, {"error": "not found"})
        if self.authorized():
            self.send(
                200,
                {
                    "data": [
                        {"deployment_id": DEPLOY, "display_name": SHOWN},
                        {"id": MIXED, "object": "model"},
                        {"id": UNIFIED},
                    ]
                },
            )

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        RouteGateway.seen.append(("POST", self.path, dict(self.headers), body))
        if self.path != "/api/chat/completions":
            return self.send(404, {"error": "not found"})
        if not self.authorized():
            return
        if body["model"] not in (DEPLOY, MIXED, UNIFIED):
            return self.send(
                404, {"error": {"message": f"no deployment {body['model']}"}}
            )
        if body["model"] == UNIFIED:
            answer = {
                "model": UNIFIED.lower(),
                "deployment": "unified-1",
                "response": {"role": "assistant", "content": "OK"},
            }
            if body.get("stream") and not RouteGateway.whole:
                return self.send(
                    200,
                    f"data: {json.dumps(answer)}\n\ndata: [DONE]\n\n".encode(),
                    "text/event-stream",
                )
            return self.send(200, answer)
        tool = bool(body.get("tools")) and not any(
            m.get("role") == "tool" for m in body["messages"]
        )
        msg = {
            "role": "assistant",
            "content": None if tool else "OK",
            "reasoning": "First I think it over.",
            "tool_calls": (
                [
                    {
                        "id": "chatcmpl-tool-1",
                        "type": "function",
                        "function": {
                            "name": "read_file",
                            "arguments": '{"path": "README.md"}',
                        },
                    }
                ]
                if tool
                else []
            ),
        }
        usage = {
            "prompt_tokens": 12,
            "completion_tokens": 520,
            "total_tokens": 532,
            "completion_tokens_details": {"reasoning_tokens": 500},
        }
        if body.get("stream") and not RouteGateway.whole:
            first = {
                "id": "c",
                "object": "chat.completion.chunk",
                "model": body["model"].lower(),
                "choices": [
                    {
                        "index": 0,
                        "delta": {k: v for k, v in msg.items() if v not in (None, [])},
                        "finish_reason": None,
                        "token_ids": None,
                    }
                ],
            }
            last = {
                "id": "c",
                "object": "chat.completion.chunk",
                "model": body["model"].lower(),
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "tool_calls" if tool else "stop",
                        "stop_reason": None,
                    }
                ],
                "usage": usage,
            }
            data = (
                "".join(f"data: {json.dumps(c)}\n\n" for c in (first, last))
                + "data: [DONE]\n\n"
            )
            return self.send(200, data.encode(), "text/event-stream")
        self.send(
            200,
            {
                "id": "c",
                "object": "chat.completion",
                "model": body["model"].lower(),
                "choices": [
                    {
                        "index": 0,
                        "message": msg,
                        "logprobs": None,
                        "finish_reason": "tool_calls" if tool else "stop",
                        "stop_reason": None,
                        "token_ids": None,
                    }
                ],
                "usage": usage,
                "routed_experts": None,
            },
        )


class TestGatewayRoutes(unittest.TestCase):
    """Some gateways are not /v1-shaped: setup finds the chat route and the model list, records them, shows a model's
    display name and stores the ID requests send; the key may be needed twice; an answer in a simplified shape is
    refused with the reason; a vLLM answer's reasoning stays out of the draft and counts against the token budget.
    """

    @classmethod
    def setUpClass(cls):
        cls.srv, cls.host = serve(RouteGateway)

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        RouteGateway.seen.clear()
        RouteGateway.whole = False
        self.tmp = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.tmp.name)
        self.keyfile = self.base / "worker_api_key"
        self.keyfile.write_text(KEY + "\n", encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def s(self, **over) -> dict:
        return dict(
            dict(
                config.worker_defaults(),
                backend="openai",
                url=self.host + "/api",
                model=DEPLOY,
                api_key_file=str(self.keyfile),
                auth_header=HEADER,
                auth_scheme="",
                auth_mode="both",
            ),
            **over,
        )

    def test_setup_finds_the_chat_route_and_the_model_list_and_records_them(self):
        listed, err, where = models.openai_list(self.s())
        self.assertEqual(
            (err, where), ("", self.host + "/models")
        )  # tried <base>/v1/models and <base>/models first
        self.assertEqual(
            [(m["name"], m["display"]) for m in listed],
            [(DEPLOY, SHOWN), (MIXED, MIXED), (UNIFIED, UNIFIED)],
        )
        rec = models.check(self.s())
        self.assertEqual((rec["chat"], rec["tools"]), ("ok", "ok"), rec)
        self.assertEqual(
            (rec["chat_path"], rec["models_url"]),
            ("/chat/completions", self.host + "/models"),
        )
        found = models.apply_found(self.s(), rec)
        self.assertEqual(
            (found["chat_path"], found["models_url"]),
            ("/chat/completions", self.host + "/models"),
        )
        self.assertIn(
            "chat at /chat/completions, models listed at " + self.host + "/models",
            "; ".join(models.describe(rec)),
        )
        posts = [
            b
            for m, p, _h, b in RouteGateway.seen
            if m == "POST" and p == "/api/chat/completions"
        ]
        self.assertTrue(
            posts and all(b["model"] == DEPLOY for b in posts)
        )  # the deployment ID, never the name
        for (
            _m,
            _p,
            h,
            _b,
        ) in (
            RouteGateway.seen
        ):  # the key twice on every request: its own header and a bearer token
            if _p in ("/models", "/api/chat/completions"):
                self.assertEqual(
                    (h.get(HEADER), h.get("Authorization")), (KEY, f"Bearer {KEY}")
                )
        # A configuration with them recorded goes straight there; doctor's presence check reads the same list.
        RouteGateway.seen.clear()
        with mock.patch.object(doctor, "worker_running", return_value=False):
            check = doctor.model_check(
                {"worker": dict(found, probe=rec), "paths": {"working_dir": "/w"}}
            )
        self.assertEqual(check.status, "PASS", check)
        self.assertEqual([p for _m, p, _h, _b in RouteGateway.seen], ["/models"])
        msg = reviewer_client.chat(
            found,
            [{"role": "user", "content": "hi"}],
            budget=reviewer_client.Budget(found),
        )
        self.assertEqual(msg["content"], "OK")
        self.assertEqual(RouteGateway.seen[-1][1], "/api/chat/completions")

    def test_header_only_or_bearer_only_is_refused_by_a_gateway_that_wants_both(self):
        for mode in ("header", "bearer"):
            with self.subTest(mode=mode):
                listed, err, _where = models.openai_list(self.s(auth_mode=mode))
                self.assertEqual(listed, [])
                self.assertIn("HTTP 401", err)
                self.assertIn(
                    "model_bindings.worker.api_key_sent_as header_and_bearer", err
                )
                self.assertNotIn(KEY, err)
        self.assertEqual(
            reviewer_client.auth_headers(self.s(), "k"),
            {HEADER: "k", "Authorization": "Bearer k"},
        )
        self.assertEqual(
            reviewer_client.auth_headers(self.s(auth_mode="bearer"), "k"),
            {"Authorization": "Bearer k"},
        )
        self.assertEqual(
            reviewer_client.auth_headers(config.worker_defaults(), "k"),
            {"Authorization": "Bearer k"},
        )
        # the reviewer's client places the key as the drafter's does, by the one rule of llm_client
        self.assertIs(reviewer_client.auth_headers, llm_client.auth_headers)

    def test_a_vllm_answer_plain_and_with_a_tool_call(self):
        for whole in (
            False,
            True,
        ):  # streamed, and one JSON body although a stream was asked for
            with self.subTest(whole=whole):
                RouteGateway.whole = whole
                s = dict(
                    self.s(model=MIXED),
                    chat_path="/chat/completions",
                    models_url=self.host + "/models",
                    tpm=100000,
                )
                b = reviewer_client.Budget(s)
                msg = reviewer_client.chat(
                    s, [{"role": "user", "content": "Reply with OK."}], budget=b
                )
                self.assertEqual(
                    (msg["content"], msg["thinking"], msg["tool_calls"]),
                    ("OK", "First I think it over.", []),
                )  # the reasoning never joins the draft text
                self.assertNotIn("usage", msg)
                self.assertGreaterEqual(
                    sum(w[1] for w in b._mem["window"]), 520
                )  # reasoning tokens counted
                msg = reviewer_client.chat(
                    s,
                    [{"role": "user", "content": "read it"}],
                    tools=[reviewer_client.PROBE_TOOL],
                    budget=b,
                )
                self.assertEqual(
                    [
                        (c["function"]["name"], c["function"]["arguments"])
                        for c in msg["tool_calls"]
                    ],
                    [("read_file", {"path": "README.md"})],
                )
                self.assertEqual(msg["content"], "")
                sent = [bd for m, _p, _h, bd in RouteGateway.seen if m == "POST"]
                self.assertTrue(
                    all(bd["model"] == MIXED for bd in sent[-2:])
                )  # as listed, case kept
                self.assertTrue(reviewer_client.probe_tools(s)["ok"])

    def test_a_unified_answer_without_choices_is_refused_with_the_reason(self):
        for whole in (False, True):
            with self.subTest(whole=whole):
                RouteGateway.whole = whole
                rec = models.check(self.s(model=UNIFIED))
                self.assertEqual(rec["chat"], "ok")  # it answers ...
                self.assertTrue(
                    rec["tools"].startswith("the endpoint answers without tool calls"),
                    rec["tools"],
                )
                self.assertIn(
                    "no choices", rec["tools"]
                )  # ... but in a shape with no tool calls
                why = config.probe_failure(
                    config.worker(
                        {
                            "worker": dict(self.s(model=UNIFIED), probe=rec),
                            "paths": {"working_dir": "/w"},
                        }
                    )
                )
                self.assertTrue(
                    why.startswith(
                        "failed the tool-call probe: the endpoint answers without tool calls"
                    )
                )


class TestRoles(unittest.TestCase):
    """No setting routes verification, gates or posting to the reviewer model."""

    def cfg(self, **over):
        c = config.load_defaults()
        c.pop("_doc", None)
        c.update(
            repo="acme/widgets",
            rocke_path="a/b/rocke",
            operator={"github_login": "op-login"},
            scope={"labels": ["x"]},
            **over,
        )
        c["paths"] = dict(c["paths"], working_dir="/w")
        return c

    def test_verification_gates_and_posting_stay_with_the_orchestrator(self):
        orchestrators = [r for r in config.CLAUDE_ONLY if r not in config.CLAUDE_MODELS]
        self.assertEqual(orchestrators, ["verify", "quality", "post"])
        for role in orchestrators:
            for want in (
                "worker",
                "",
                {"backend": "openai", "url": "https://gw.example.com", "model": "m"},
                "qwen",
                "opus",
            ):
                with self.subTest(role=role, want=want):
                    c = self.cfg(
                        models=dict(config.load_defaults()["models"], **{role: want})
                    )
                    self.assertTrue(
                        any(
                            p.startswith(f"the package's models.{role} must be")
                            for p in config.problems(c)
                        )
                    )
                    self.assertEqual(
                        config.model_for(c, role), {"by": "claude", "model": "opus"}
                    )
        # Claude's audit and deep roles take a Claude model of their own, never the reviewer model; a message names
        # each by its pass's deploy key.
        key = {
            "audit": "features.review_automation.passes.verification_pass.model",
            "deep": "features.review_automation.passes.code_quality_pass.model",
        }
        for role in config.CLAUDE_MODELS:
            for want, refused, model in (
                ("worker", True, "opus"),
                ({"backend": "openai", "model": "m"}, True, "opus"),
                ("", False, "opus"),
                ("sonnet", False, "sonnet"),
                ("claude", False, "opus"),
            ):
                with self.subTest(role=role, want=want):
                    c = self.cfg(
                        models=dict(config.load_defaults()["models"], **{role: want})
                    )
                    self.assertEqual(
                        any(
                            p.startswith(f"{key[role]} must be a Claude model")
                            for p in config.problems(c)
                        ),
                        refused,
                    )
                    self.assertEqual(
                        config.model_for(c, role), {"by": "claude", "model": model}
                    )
        legacy = self.cfg(
            models={
                "verify": "claude-opus",
                "quality": "claude-opus",
                "post": "claude-opus",
                "draft": "worker",
            }
        )
        self.assertEqual(
            [p for p in config.problems(legacy) if p.startswith("models.")], []
        )
        self.assertEqual(config.model_for(legacy, "draft")["by"], "worker")
        self.assertEqual(
            config.model_for(self.cfg(orchestrator={"model": "sonnet"}), "post"),
            {"by": "claude", "model": "sonnet"},
        )
        for (
            role
        ) in (
            orchestrators
        ):  # a non-Claude orchestrator keeps them too; the reviewer model never gets them
            self.assertEqual(
                config.model_for(
                    self.cfg(orchestrator={"model": ORCH}, models={role: "worker"}),
                    role,
                ),
                {"by": "claude", "model": ORCH},
            )

    def test_a_model_named_with_a_context_window_is_known_and_named_as_its_base_id(
        self,
    ):
        """Claude Code's suffix in brackets (a context window) is cut by its one rule, deploy_tree.base_id, whatever
        the window: the model is known and named as the model without it."""
        for name in (
            "opus[1m]",
            "claude-sonnet-5-5[1m]",
            "opus[200k]",
            "claude-sonnet-5-5[2m]",
        ):
            base = deploy_tree.base_id(name)
            self.assertTrue(config.orchestrator_known(base), name)
            self.assertTrue(config.orchestrator_known(name), name)
            self.assertEqual(config.model_words(name), config.model_words(base), name)

    def test_the_orchestrator_may_be_any_model_with_a_warning_outside_the_known_ones(
        self,
    ):
        """Any orchestrator model may be chosen; one that uses the skills and docs effectively is best, and a less
        capable one gets a warning. Nothing is refused; the current Claude Opus and
        Sonnet families get no warning, any other model one."""
        for name in (
            "opus",
            "sonnet",
            "opusplan",
            "opus[1m]",
            "Sonnet[1m]",
            "claude-opus-5-5",
            "claude-sonnet-5-5",
            "claude-sonnet-4-5-20250929",
            "us.anthropic.claude-opus-5-5",
            "claude-opus-4-1@20250805",
        ):
            self.assertEqual(config.orchestrator_warning(name), "", name)
        for name in (
            "haiku",
            "fable",
            "claude-3-opus-20240229",
            "gpt-4o",
            "qwen3.6:35b",
            ORCH,
        ):
            self.assertTrue(
                config.orchestrator_warning(name).startswith(
                    f"{name} is not one of the models known"
                ),
                name,
            )
            self.assertIn("long and tool-heavy", config.orchestrator_warning(name))
            self.assertEqual(
                [
                    p
                    for p in config.problems(self.cfg(orchestrator={"model": name}))
                    if p.startswith("orchestrator")
                ],
                [],
                name,
            )
        self.assertEqual(
            [
                config.orchestrator_name({"orchestrator": {"model": m}})
                for m in (
                    "opus",
                    "claude-opus-5-5",
                    "claude-sonnet-4-5-20250929",
                    "fable",
                    "sonnet[1m]",
                    ORCH,
                )
            ],
            [
                "Claude Opus",
                "Claude Opus 5.5",
                "Claude Sonnet 4.5",
                "Claude Fable",
                "Claude Sonnet",
                ORCH,
            ],
        )
        cfg = config.normalize(self.cfg(orchestrator={"model": ORCH}))
        text = summary.render(cfg)
        self.assertIn(f"{ORCH} (WARNING: {ORCH} is not one of the models known", text)
        self.assertNotIn(
            "WARNING",
            summary.render(config.normalize(self.cfg(orchestrator={"model": "opus"})))
            .split("Orchestrator model", 1)[1]
            .split("\n", 1)[0],
        )


class TestTheStartChecksTheModel(unittest.TestCase):
    """What the start's checks found, in words (models.describe, the setup summary's row): only what a probe's record
    holds (models.probe_record), and while there is none, the start that makes one."""

    def record(self, s: dict) -> dict:
        answered = {
            "answers": True,
            "model_found": True,
            "models": [s["model"]],
            "context": 32768,
            "latency_s": 0.5,
            "error": "",
            "rate_limited": "",
            "models_url": "",
            "chat_path": "",
        }
        return models.probe_record(s, answered, now=lambda: "2000-01-01T00:00:00Z")

    def test_a_record_s_words_and_settings_are_what_it_holds(self):
        s = dict(
            config.worker_defaults(),
            backend="openai",
            url="https://gw.example.com/route",
            model="m",
            max_output_tokens=12000,
            rpm=0,
            tpm=0,
        )
        rec = self.record(s)
        more = dict(rec, max_output=1, rpm=2, tpm=3)  # no probe's record holds these
        self.assertEqual(models.describe(more), models.describe(rec))
        self.assertEqual(models.apply_found(s, more), models.apply_found(s, rec))
        self.assertIn("reported: context 32768", models.describe(rec))

    def test_with_no_record_the_start_that_makes_one(self):
        with mock.patch.dict(os.environ, {"RKMON_WRAPPER": "./rkmon"}):
            said = models.describe({})
            w = config.worker(
                {
                    "worker": {
                        "backend": "openai",
                        "url": "https://gw.example.com",
                        "model": "m",
                    }
                }
            )
            rows = dict(summary._worker_rows({}, w, ""))
        want = "not checked yet: the next start checks it (./rkmon stop --drain, then ./rkmon start)"
        self.assertEqual(said, [want])
        self.assertEqual(rows["Reviewer model: found by the start's checks"], want)


class TestRefusalsNameTheDeploySettings(unittest.TestCase):
    """An endpoint's refusal (reviewer_client.explain_status) and an answer in another shape (its shape note) name the
    deploy settings file's keys that set what was sent, each as rkmon_lib/deploy_tree.py names it, and no key of the
    configuration that the file does not have."""

    def test_the_keys_are_the_deploy_settings_file_s(self):
        for internal, key in reviewer_client.DEPLOY_KEYS.items():
            with self.subTest(internal=internal):
                self.assertEqual(deploy_tree.name(f"worker.{internal}"), key)
        for backend, key in reviewer_client.URL_KEYS.items():
            with self.subTest(backend=backend):
                self.assertEqual(doctor.url_key(deploy_tree.WORKER, backend), key)
        self.assertEqual(deploy_tree.AUTH_MODE["header_and_bearer"], "both")

    def test_each_refusal_names_them(self):
        s = dict(
            config.worker_defaults(),
            backend="openai",
            url="https://gw.example.com/route",
            model="m",
        )
        said = {
            "401": reviewer_client.explain_status(401, s, "the test prompt"),
            "404": reviewer_client.explain_status(404, s, "the model list"),
            "ollama 404": reviewer_client.explain_status(
                404, dict(s, backend="ollama"), "the model list"
            ),
            "shape": reviewer_client._shape_note({"response": {}}),
        }
        for key in reviewer_client.DEPLOY_KEYS.values():
            self.assertIn(key, said["401"])
        self.assertIn("api_key_sent_as header_and_bearer sends", said["401"])
        self.assertIn(reviewer_client.URL_KEYS["openai"], said["404"])
        self.assertIn(reviewer_client.URL_KEYS["ollama"], said["ollama 404"])
        self.assertIn(reviewer_client.URL_KEYS["openai"], said["shape"])
        for name, text in said.items():
            with self.subTest(name=name):
                self.assertNotRegex(text, r"(?<![\w.])worker\.")


class TestPipesInEffect(unittest.TestCase):
    def test_asked_in_effect_and_why(self):
        local = {
            "worker": {
                "backend": "ollama",
                "url": "http://127.0.0.1:11434",
                "model": "m:1",
            }
        }
        p = config.pipes(local)
        self.assertEqual((p["asked"], p["effective"]), (3, 1))
        self.assertIn("local Ollama, which keeps one model in memory", p["why"])
        local["worker"]["max_parallel"] = 2
        self.assertEqual(config.pipes(local)["effective"], 2)
        remote = {
            "worker": {
                "backend": "openai",
                "url": "https://gw.example.com/route",
                "model": "m",
                "rpm": 60,
            },
            "review": {"pipes": 9},
        }
        p = config.pipes(remote)
        self.assertEqual((p["asked"], p["effective"]), (5, 5))
        self.assertIn("share its limits of 60 requests a minute", p["why"])
        bare = {"worker": dict(remote["worker"], rpm=0, max_parallel=4)}
        self.assertIn(
            "allowed 4 at once (model_bindings.worker.limits.max_parallel_requests); no rate limits known "
            "(model_bindings.worker.limits.requests_per_minute, model_bindings.worker.limits.tokens_per_minute)",
            config.pipes(bare)["why"],
        )  # the deploy settings' keys, where the operator sets them
        self.assertTrue(
            config.pipes_line(remote).startswith(
                "asked 5, running 5, because the reviewer is an endpoint outside the container"
            )
        )
        # an Ollama the container reaches by a name, such as the host's: outside the container, never "no local memory"
        host = {"worker": dict(local["worker"], url="http://ollama.example:11434")}
        self.assertIn(
            "the reviewer is an endpoint outside the container",
            config.pipes(host)["why"],
        )
        self.assertEqual(
            config.checks_limit(remote), 1
        )  # the pipes never change the checks limit
        self.assertTrue(
            any(
                x.startswith("features.review_automation.parallel_reviews is 9")
                for x in config.problems(dict(remote, deployment={"phase": "observe"}))
            )
        )

    def test_the_most_is_the_deploy_schema_s_maximum_and_follows_it(self):
        """The most parallel reviews and builds are the deploy schema's maximums of the keys that set them
        (parallel_reviews, parallel_builds), their one home, read at the first use: review.pipes and checks.limit are
        held to them, an endpoint with no limit of its own is capped at the most reviews, and the scaler moves within
        them."""
        from rkmon_lib import file_types, scaler, settings_file

        most = (config.pipes_max, config.checks_max)
        for f in most:
            f.cache_clear()
            self.addCleanup(f.cache_clear)
        root = file_types.DEPLOY.load()
        for f, key in zip(most, ("review.pipes", "checks.limit")):
            with self.subTest(key=key):
                node = settings_file.schema_at(root, deploy_tree.file_key(key))
                self.assertEqual(f(), node["maximum"])
        pipes, checks = config.pipes_max() + 2, config.checks_max() + 2
        real = deploy_tree.node

        def raised(key):  # the schema with each maximum 2 higher
            path, node = real(key)
            if isinstance(node, dict) and "maximum" in node:
                node = dict(node, maximum=node["maximum"] + 2)
            return path, node

        for f in most:
            f.cache_clear()
        remote = {
            "worker": {
                "backend": "openai",
                "url": "https://gw.example.com/route",
                "model": "m",
            },
            "review": {"pipes": pipes},
            "checks": {"limit": checks},
        }
        with mock.patch.object(deploy_tree, "node", raised):
            self.assertEqual((config.pipes_max(), config.checks_max()), (pipes, checks))
            self.assertEqual(
                (config.pipes_asked(remote), config.checks_limit(remote)),
                (pipes, checks),
            )
            self.assertEqual(config.pipes(remote)["effective"], pipes)
            self.assertEqual(
                scaler.ranges(), {"pipes": (1, pipes), "heavy": (1, checks)}
            )
            found = config.problems(dict(remote, deployment={"phase": "observe"}))
            self.assertFalse(
                [x for x in found if x.startswith(("review.pipes", "checks.limit"))],
                found,
            )


if __name__ == "__main__":
    unittest.main()
