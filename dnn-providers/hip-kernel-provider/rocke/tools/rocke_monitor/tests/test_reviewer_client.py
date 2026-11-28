# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""worker/reviewer_client.py against a fake local server for both APIs (no test calls a real model): Ollama's /api/chat
(JSON lines) and an OpenAI-compatible /v1/chat/completions (server-sent events), tool calls included, and setup's
endpoint probe."""

import http.server
import json
import pathlib
import sys
import tempfile
import threading
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "worker"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import reviewer_client  # noqa: E402
import sandbox  # noqa: E402

SEEN = []


class _Fake(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, status, body, ctype="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body if isinstance(body, bytes) else json.dumps(body).encode())

    def do_GET(self):
        SEEN.append(("GET", self.path, dict(self.headers), None))
        if self.path == "/api/tags":
            return self._send(200, {"models": [{"name": "m:1"}]})
        if self.path == "/v1/models":
            return self._send(200, {"data": [{"id": "m:1", "max_model_len": 131072}]})
        self._send(404, {"error": "no"})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(("POST", self.path, dict(self.headers), body))
        if self.path == "/api/show":
            return self._send(200, {"model_info": {"qwen.context_length": 262144}})
        if self.path == "/api/chat" and not body.get("stream"):
            return self._send(200, {"message": {"content": "OK"}, "done": True})
        if self.path == "/v1/chat/completions" and not body.get("stream"):
            return self._send(200, {"choices": [{"message": {"content": "OK"}}]})
        if self.path == "/api/chat":
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for c in (
                {"message": {"thinking": "th"}},
                {"message": {"content": "hel"}},
                {
                    "message": {
                        "content": "lo",
                        "tool_calls": [
                            {
                                "function": {
                                    "name": "grep_code",
                                    "arguments": {"pattern": "x"},
                                }
                            }
                        ],
                    }
                },
                {"done": True},
            ):
                self.wfile.write((json.dumps(c) + "\n").encode())
            return
        if self.path == "/v1/chat/completions":
            if body["model"] == "fail":
                return self._send(500, {"error": "boom"})
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for c in (
                {"choices": [{"delta": {"reasoning_content": "th"}}]},
                {"choices": [{"delta": {"content": "hel"}}]},
                {
                    "choices": [
                        {
                            "delta": {
                                "content": "lo",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "c1",
                                        "function": {
                                            "name": "grep_code",
                                            "arguments": '{"pat',
                                        },
                                    }
                                ],
                            }
                        }
                    ]
                },
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "function": {"arguments": 'tern": "x"}'},
                                    }
                                ]
                            }
                        }
                    ]
                },
            ):
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self._send(404, {"error": "no"})


class TestClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Fake)
        cls.srv.daemon_threads = True
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        SEEN.clear()
        self.d = tempfile.TemporaryDirectory()
        self.key = pathlib.Path(self.d.name) / "key"
        self.key.write_text("k-test\n")

    def tearDown(self):
        self.d.cleanup()

    def s(self, backend, **over):
        import importlib

        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
        config = importlib.import_module("rkmon_lib.config")
        return dict(
            dict(config.worker_defaults(), backend=backend, url=self.url, model="m:1"),
            **over,
        )

    def test_ollama_streams_thinking_content_and_tool_calls(self):
        with mock.patch.object(
            reviewer_client, "journal_progress", lambda since: (None, None)
        ):
            msg = reviewer_client.chat(
                self.s("ollama"),
                [{"role": "user", "content": "hi"}],
                tools=[{"type": "function"}],
            )
        self.assertEqual((msg["thinking"], msg["content"]), ("th", "hello"))
        self.assertEqual(
            msg["tool_calls"][0]["function"],
            {"name": "grep_code", "arguments": {"pattern": "x"}},
        )
        _m, path, _h, body = SEEN[-1]
        self.assertEqual(path, "/api/chat")
        self.assertEqual(
            (body["model"], body["options"]["num_ctx"], body["tools"]),
            ("m:1", 262144, [{"type": "function"}]),
        )
        # worker.think unset: the request carries no effort, and the model takes its own default
        self.assertNotIn("think", body)
        for think in ("high", False):
            with mock.patch.object(
                reviewer_client, "journal_progress", lambda since: (None, None)
            ):
                reviewer_client.chat(
                    self.s("ollama", think=think), [{"role": "user", "content": "hi"}]
                )
            self.assertEqual(SEEN[-1][3]["think"], think)

    def test_openai_streams_and_joins_split_tool_arguments(self):
        history = [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {"path": "a"}}}
                ],
            },
            {"role": "tool", "tool_name": "read_file", "content": "text"},
        ]
        msg = reviewer_client.chat(
            self.s("openai", api_key_file=str(self.key), think="high"),
            history,
            tools=[{"t": 1}],
        )
        self.assertEqual((msg["thinking"], msg["content"]), ("th", "hello"))
        self.assertEqual(
            msg["tool_calls"],
            [
                {
                    "id": "c1",
                    "function": {"name": "grep_code", "arguments": {"pattern": "x"}},
                }
            ],
        )
        _m, path, headers, body = SEEN[-1]
        self.assertEqual(path, "/v1/chat/completions")
        self.assertEqual(headers.get("Authorization"), "Bearer k-test")
        self.assertEqual(
            (body["reasoning_effort"], body["tool_choice"], body["max_tokens"]),
            ("high", "auto", 16000),
        )
        sent = body["messages"]
        self.assertEqual(
            sent[1]["tool_calls"][0]["function"],
            {"name": "read_file", "arguments": '{"path": "a"}'},
        )
        self.assertEqual(
            sent[2],
            {
                "role": "tool",
                "tool_call_id": sent[1]["tool_calls"][0]["id"],
                "content": "text",
            },
        )

    def test_an_http_error_is_a_connection_error(self):
        with self.assertRaises(ConnectionError):
            reviewer_client.chat(
                self.s("openai", model="fail"), [{"role": "user", "content": "hi"}]
            )

    def test_probe_both_backends(self):
        for backend, context in (("ollama", 262144), ("openai", 131072)):
            with self.subTest(backend=backend):
                p = reviewer_client.probe(self.s(backend), timeout=10)
                self.assertTrue(p["answers"] and p["model_found"], p)
                self.assertEqual(p["context"], context)
                self.assertIsNotNone(p["latency_s"])
                self.assertEqual(p["error"], "")
        p = reviewer_client.probe(self.s("ollama", model="other:2"), timeout=10)
        self.assertEqual(
            (p["answers"], p["model_found"], p["latency_s"]), (True, False, None)
        )
        closed = f"http://127.0.0.1:{sandbox.closed_port(self)}"
        p = reviewer_client.probe(self.s("ollama", url=closed), timeout=2)
        self.assertFalse(p["answers"])
        self.assertTrue(p["error"])

    def test_only_a_local_ollama_is_unloaded(self):
        with mock.patch.object(reviewer_client.subprocess, "run") as run:
            reviewer_client.unload(self.s("openai"))
            reviewer_client.unload(self.s("ollama", url="http://models.example:11434"))
            run.assert_not_called()
            reviewer_client.unload(self.s("ollama"))
            self.assertEqual(run.call_args[0][0], ["ollama", "stop", "m:1"])

    def test_each_answered_request_goes_to_the_token_log(self):
        run = pathlib.Path(self.d.name) / "run"
        budget = reviewer_client.Budget(
            self.s("openai", api_key_file=str(self.key)),
            str(run),
            log=lambda text: None,
            who="pr12",
        )
        reviewer_client.chat(
            self.s("openai", api_key_file=str(self.key)),
            [{"role": "user", "content": "hi"}],
            budget=budget,
        )
        (rec,) = [
            json.loads(x)
            for x in (run / reviewer_client.TOKENS_LOG).read_text().splitlines()
        ]
        self.assertEqual((rec["who"], rec["model"]), ("pr12", "m:1"))
        self.assertEqual(budget.spent, [rec["prompt"], rec["answer"]])
        self.assertGreater(rec["prompt"], 0)
        # The endpoint's own prompt count wins over the estimate; the answer is the budget's, without that difference.
        msg = {
            "content": "x" * 400,
            "usage": {"prompt_tokens": 900, "completion_tokens": 50},
        }
        self.assertEqual(
            reviewer_client.request_tokens(msg, 100),
            (900, reviewer_client.answer_tokens(msg, 100) - 800),
        )
        self.assertEqual(
            reviewer_client.request_tokens({"content": "x" * 400}, 100), (100, 101)
        )
        # Without a run folder nothing is written; the total still counts.
        mem = reviewer_client.Budget(self.s("openai"), None)
        mem.count(5, 6)
        self.assertEqual(mem.spent, [5, 6])


if __name__ == "__main__":
    unittest.main()
