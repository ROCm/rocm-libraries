# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""worker/llm_client.py: the provider adapter's request and answers, the key's headers (never logged), usage with cached
tokens, the retries on 429 and failures, and the shared limiter (rolling minute, ranks, reserve, the requests in flight,
the run's caps and each thread's phase, the history the scaler reads). No socket reaches a model: each request goes to a
fake `post`, or to a port the test holds closed (sandbox.closed_port)."""

import json
import os
import pathlib
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.dont_write_bytecode = True

import llm_client as lc  # noqa: E402
import sandbox  # noqa: E402
from rkmon_lib import bindings, custom_headers, registry  # noqa: E402

KEY = "k-" + "planted-client-key-9917"


def settings(d: pathlib.Path, **over) -> dict:
    keyfile = d / "key"
    keyfile.write_text(KEY + "\n", encoding="utf-8")
    # a role's client settings as bindings.client_settings gives them: the package's binding defaults under their own
    # keys, and the drafter settings' reserve (bindings.drafter_settings)
    s = dict(
        bindings.binding_defaults(),
        reserve_pct=bindings.drafter_settings({})["reserve_pct"],
    )
    s |= {
        "provider": "openai",
        "url": "http://127.0.0.1:9/onprem",
        "chat_path": "/chat/completions",
        "api_key_file": str(keyfile),
        "auth_header": "Ocp-Apim-Subscription-Key",
        "auth_scheme": "",
        "auth_mode": "both",
        "model": "Model-X",
        "tpm": 1000,
        "rpm": 3,
        "max_output_tokens": 500,
        "limiter_dir": str(d / "limits"),
        "resend_wait_s": 0,
        "max_resends": 2,
        "stall_s": 5,
    }
    s.update(over)
    return s


def answer(content="", calls=(), usage=None):
    return {
        "role": "assistant",
        "content": content,
        "reasoning": "",
        "finish_reason": "stop",
        "tool_calls": [
            {"id": f"c{i}", "function": {"name": n, "arguments": a}}
            for i, (n, a) in enumerate(calls)
        ],
        "usage": usage,
    }


class Clock:
    def __init__(self, t=1000.0):
        self.t = t

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


class TestAdapter(unittest.TestCase):
    def test_the_header_lines(self):
        """`Name: value` lines, read as the Claude session sends them, by the one reader of that form
        (rkmon_lib/custom_headers.py), the home of the key's mark too, neither of them the client's: a line with no
        colon has no name.
        """
        self.assertFalse(hasattr(lc, "header_lines"))
        self.assertFalse(hasattr(lc, "KEY_MARK"))
        text = "A: b\n\nno colon\n C : d: e \nE:\n"
        self.assertEqual(
            custom_headers.header_lines(text),
            [("A", "b"), ("", "no colon"), ("C", "d: e"), ("E", "")],
        )
        self.assertEqual(custom_headers.header_lines(""), [])

    def test_a_variable_is_read_from_the_environment_alone(self):
        """A binding's `$NAME` url is read from the environment the caller runs in, and nowhere else: a file the
        settings name (the older secret claude_env) is never read. Its key and its headers come from no variable: a key
        from its file alone.
        """
        names = ("RKMON_TEST_URL", "RKMON_TEST_KEY", "RKMON_TEST_HEADERS")
        with tempfile.TemporaryDirectory() as d:
            older = pathlib.Path(d) / "claude_env"
            older.write_text(
                json.dumps(
                    dict(
                        zip(
                            names,
                            ("https://older.example.invalid", "k-older", "X-Older: v"),
                        )
                    )
                ),
                encoding="utf-8",
            )
            s = {
                "url": "$RKMON_TEST_URL",
                "api_key_env": "RKMON_TEST_KEY",
                "headers_env": "RKMON_TEST_HEADERS",
                "env_file": str(older),
            }
            with mock.patch.dict(os.environ, dict.fromkeys(names, "")):
                self.assertEqual((lc.base_url(s), lc.api_key(s)), ("", ""))
            given = ("https://gw.example.invalid/", "k-env", "X-Env: v")
            with mock.patch.dict(os.environ, dict(zip(names, given))):
                self.assertEqual(
                    (lc.base_url(s), lc.api_key(s)), ("https://gw.example.invalid", "")
                )
                for name, ad in lc.PROVIDERS.items():
                    with self.subTest(provider=name):
                        sent = ad.headers(s, "k-file")
                        self.assertNotIn("X-Env", sent)
                        self.assertNotIn("k-env", " ".join(sent.values()))

    def test_the_request_the_key_and_the_route(self):
        with tempfile.TemporaryDirectory() as d:
            s = settings(pathlib.Path(d))
            ad = lc.adapter(s)
            self.assertEqual(
                ad.route(s), ("http", "127.0.0.1", 9, "/onprem/chat/completions")
            )
            msgs = [
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}],
                },
                {"role": "tool", "tool_name": "f", "content": "out"},
            ]
            body = ad.body(s, msgs, [{"type": "function"}], "low", None)
            self.assertEqual(
                (
                    body["model"],
                    body["reasoning_effort"],
                    body["tool_choice"],
                    body["max_tokens"],
                ),
                ("Model-X", "low", "auto", 500),
            )
            self.assertEqual(
                (body["stream"], body["stream_options"]),
                (True, {"include_usage": True}),
            )
            sent = body["messages"]
            self.assertEqual(
                sent[1]["tool_calls"][0]["function"]["arguments"], '{"a": 1}'
            )
            self.assertEqual(sent[2]["tool_call_id"], sent[1]["tool_calls"][0]["id"])
            self.assertNotIn("reasoning_effort", ad.body(s, msgs, None, None, None))
            self.assertEqual(
                lc.auth_headers(s, KEY),
                {"Ocp-Apim-Subscription-Key": KEY, "Authorization": f"Bearer {KEY}"},
            )
            self.assertEqual(
                lc.auth_headers(dict(s, auth_mode=""), KEY),
                {"Ocp-Apim-Subscription-Key": KEY},
            )
            self.assertEqual(
                lc.auth_headers({}, KEY), {"Authorization": f"Bearer {KEY}"}
            )
            self.assertEqual(lc.scrub(f"echo {KEY} back", KEY), "echo <key> back")
            self.assertNotIn(KEY, lc.key_fingerprint(KEY))
            with self.assertRaises(lc.GatewayError):
                lc.adapter(dict(s, provider="smoke-signals"))

    def test_streamed_and_whole_answers(self):
        chunks = [
            {"choices": [{"delta": {"reasoning": "th"}}]},
            {"choices": [{"delta": {"reasoning_content": "ink", "content": "hel"}}]},
            {
                "choices": [
                    {
                        "delta": {
                            "content": "lo",
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "c1",
                                    "function": {"name": "grep", "arguments": '{"pat'},
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
                                {"index": 0, "function": {"arguments": 'tern": "x"}'}}
                            ]
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
            {
                "choices": [],
                "usage": {
                    "prompt_tokens": 50,
                    "completion_tokens": 7,
                    "prompt_tokens_details": {"cached_tokens": 30},
                    "completion_tokens_details": {"reasoning_tokens": 4},
                },
            },
        ]
        lines = [f"data: {json.dumps(c)}".encode() for c in chunks] + [b"data: [DONE]"]
        m = lc.OpenAICompatible.stream_answer(lines)
        self.assertEqual(
            (m["content"], m["reasoning"], m["finish_reason"]),
            ("hello", "think", "tool_calls"),
        )
        self.assertEqual(
            m["tool_calls"],
            [{"id": "c1", "function": {"name": "grep", "arguments": {"pattern": "x"}}}],
        )
        u = lc.usage_of(m, 999)
        self.assertEqual(
            (u["prompt"], u["cached"], u["completion"], u["reasoning"], u["reported"]),
            (50, 30, 7, 4, True),
        )
        self.assertIn("prompt_tokens_details.cached_tokens", u["usage_keys"])
        with self.assertRaises(lc.GatewayError):
            lc.OpenAICompatible.stream_answer([b'data: {"error": "boom"}'])
        w = lc.OpenAICompatible.whole_answer(
            json.dumps(
                {
                    "choices": [
                        {
                            "message": {
                                "content": "ok",
                                "reasoning_content": "r",
                                "tool_calls": [
                                    {
                                        "id": "z",
                                        "function": {
                                            "name": "f",
                                            "arguments": '{"a": 2}',
                                        },
                                    }
                                ],
                            },
                            "finish_reason": "stop",
                        }
                    ]
                }
            ).encode()
        )
        self.assertEqual(
            (w["content"], w["reasoning"], w["tool_calls"][0]["function"]["arguments"]),
            ("ok", "r", {"a": 2}),
        )
        est = lc.usage_of({"content": "x" * 70}, 12)
        self.assertEqual(
            (est["prompt"], est["reported"], est["cached"]), (12, False, 0)
        )

    def test_a_refused_key_is_named_by_where_the_binding_takes_it(self):
        """A 401 or a 403 names the settings the refused key and its headers come from: an OpenAI-compatible
        binding's own, and with the Claude session's key headers (key_headers, which rkmon_lib/bindings.py gives a
        claude binding that has no key of its own) the claude login's key, sent as integrations.claude_code's
        settings say. A request that carried no key (an ollama binding has no key settings) says so, and names no key
        setting."""

        class Refusing:
            """A connection whose request gets STATUS, with no text."""

            status = 401

            def __init__(self, *args, **kwargs):
                pass

            def request(self, *args, **kwargs):
                pass

            def getresponse(self):
                return self

            def read(self, *args):
                return b""

            def getheader(self, name, default=None):
                return default

            def getheaders(self):
                return []

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as d:
            gateway = settings(pathlib.Path(d))
            claude = dict(
                gateway,
                provider="anthropic",
                kind="claude",
                key_headers={"x-api-key": custom_headers.KEY_MARK},
            )
            ollama = dict(gateway, kind="ollama", api_key_file="")
            for status in (401, 403):
                Refusing.status = status
                for kind, s, key, start, unnamed in (
                    (
                        "openai",
                        gateway,
                        KEY,
                        "the endpoint refused the key (the binding's api_key_secret_name, sent as its api_key_sent_as",
                        ("claude",),
                    ),
                    (
                        "claude",
                        claude,
                        KEY,
                        "the endpoint refused the key (the claude login's, sent as integrations.claude_code's "
                        "api_key_sent_as",
                        ("api_key_secret_name",),
                    ),
                    (
                        "ollama",
                        ollama,
                        lc.api_key(ollama),
                        "the endpoint refused the request, which carried no key: the binding names none",
                        ("api_key_secret_name", "api_key_sent_as", "claude"),
                    ),
                ):
                    with self.subTest(status=status, kind=kind):
                        with mock.patch.object(
                            lc.http.client, "HTTPConnection", Refusing
                        ), self.assertRaises(lc.GatewayError) as cm:
                            lc.adapter(s).post(s, {}, key)
                        said = str(cm.exception)
                        self.assertTrue(
                            said.startswith(f"HTTP {status}: {start}"), said
                        )
                        for word in unnamed:
                            self.assertNotIn(word, said)
                        self.assertNotIn(KEY, said)


class TestChat(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)
        self.s = settings(self.d, tpm=0, rpm=0)

    def tearDown(self):
        self.tmp.cleanup()

    def test_a_429_blocks_every_caller_then_the_request_goes(self):
        seq = [
            lc.RateLimited("HTTP 429: busy", 0.01),
            answer("done", usage={"prompt_tokens": 9, "completion_tokens": 2}),
        ]
        seen = []

        def post(s, body, key):
            seen.append(key)
            r = seq.pop(0)
            if isinstance(r, Exception):
                raise r
            return r

        lim = lc.limiter_for(self.s, who="t", run=False)
        m = lc.chat(self.s, [{"role": "user", "content": "hi"}], limiter=lim, post=post)
        self.assertEqual(m["content"], "done")
        self.assertEqual(seen, [KEY, KEY])
        self.assertEqual(
            (lim.totals["rate_limited"], lim.totals["requests"], lim.used()), (1, 1, 11)
        )
        stats = lc.window_stats(self.s)
        self.assertEqual(
            (stats["rate_limited"], stats["requests"], stats["tokens"]), (1, 2, 11)
        )
        self.assertIn("HTTP 429", (self.d / "limits" / lc.LIMITER_LOG).read_text())
        for p in (self.d / "limits").iterdir():
            self.assertNotIn(KEY, p.read_text(encoding="utf-8"))

    def test_failures_resend_refusals_do_not(self):
        calls = []

        def flaky(s, body, key):
            calls.append(1)
            if len(calls) < 3:
                raise lc.GatewayError("HTTP 502: down", 502, transient=True)
            return answer("ok")

        self.assertEqual(
            lc.chat(self.s, [{"role": "user", "content": "x"}], post=flaky)["content"],
            "ok",
        )
        calls.clear()

        def refuse(s, body, key):
            calls.append(1)
            raise lc.GatewayError("HTTP 401: no", 401)

        with self.assertRaises(lc.GatewayError):
            lc.chat(self.s, [{"role": "user", "content": "x"}], post=refuse)
        self.assertEqual(len(calls), 1)

        def no_usage_option(s, body, key):
            if "stream_options" in body:
                raise lc.GatewayError(
                    "HTTP 400: the gateway refused the request: unknown field stream_options",
                    400,
                )
            return answer("plain")

        self.assertEqual(
            lc.chat(self.s, [{"role": "user", "content": "x"}], post=no_usage_option)[
                "content"
            ],
            "plain",
        )

    def test_the_closed_port_and_a_missing_key(self):
        closed = dict(
            self.s, url=f"http://127.0.0.1:{sandbox.closed_port(self)}/onprem"
        )
        with self.assertRaises(lc.GatewayError) as cm:
            lc.chat(
                dict(closed, max_resends=1),
                [{"role": "user", "content": "x"}],
                sleep=lambda s: None,
            )
        self.assertIn("no answer", str(cm.exception))
        with self.assertRaises(lc.GatewayError):
            lc.chat(
                dict(closed, api_key_file=str(self.d / "absent")),
                [{"role": "user", "content": "x"}],
            )


class TestLimiter(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)
        self.clock = Clock()

    def tearDown(self):
        self.tmp.cleanup()

    def lim(self, prio=False, budget=0, wall=0, **over):
        return lc.Limiter(
            settings(self.d, **over),
            who="t",
            budget_tokens=budget,
            wall_s=wall,
            prio=prio,
            clock=self.clock,
            sleep=self.clock.sleep,
        )

    def test_the_reserve_is_the_drafter_settings_one(self):
        """The share kept for the best-ranked caller is the client settings' reserve_pct, which the drafter settings
        give (bindings.drafter_settings, the package's under config drafter): the limiter has no value of its own.
        """
        package = registry.default("drafter")["reserve_pct"]
        self.assertEqual(self.lim().reserve, package / 100)
        self.assertEqual(self.lim(reserve_pct=35).reserve, 0.35)
        s = settings(self.d)
        del s["reserve_pct"]
        with self.assertRaises(KeyError):
            lc.Limiter(s, who="t", clock=self.clock, sleep=self.clock.sleep)

    def test_requests_and_tokens_per_rolling_minute(self):
        a = self.lim(reserve_pct=0)
        for _ in range(3):
            self.assertEqual(a.acquire(10), 0)
            a.settle(10)
        self.assertGreater(
            a.acquire(10), 55
        )  # the fourth request of a minute waits for the first to leave
        b = self.lim(rpm=0, reserve_pct=0)
        self.clock.t += 120
        b.acquire(900)
        b.settle(900)
        self.assertGreater(b.acquire(200), 55)  # 900 + 200 > 1000 tokens a minute
        self.assertEqual(
            b.acquire(5000) >= 0, True
        )  # larger than the limit: goes once the window is empty

    def test_buckets_by_provider_key_and_model(self):
        s = settings(self.d)
        self.assertEqual(
            lc.bucket(s), lc.bucket(dict(s, url="http://127.0.0.1:9/other"))
        )
        self.assertNotEqual(lc.bucket(s), lc.bucket(dict(s, model="Model-Y")))
        self.assertNotEqual(lc.bucket(s), lc.bucket(s, key="another"))
        self.assertTrue(lc.bucket_path(s).name.startswith("bucket-"))

    def test_a_priority_goes_first_and_the_reserve_is_kept_for_the_best_ranked(self):
        best = self.lim(prio=True, rpm=10, tpm=0)
        other = self.lim(rpm=10, tpm=0)
        best.acquire(1)  # the best-ranked caller is active
        best.settle(1)
        for _ in range(7):
            other.acquire(1)
            other.settle(1)
        waits = []
        other.sleep = lambda s: (waits.append(s), self.clock.sleep(s))
        other.acquire(1)  # 8 of 10 used: the last 20% is the best-ranked caller's
        self.assertTrue(waits and waits[0] <= 5.0, waits)
        self.assertIn(
            "kept for the best-ranked", (self.d / "limits" / lc.LIMITER_LOG).read_text()
        )
        self.clock.t += 120
        self.assertEqual(best.rank(), [0, 0])
        other.progress = 0.9
        self.assertEqual(other.rank(), [1, -0.9])

    def test_the_runs_caps(self):
        a = self.lim(budget=100, tpm=0, rpm=0)
        a.acquire(60)
        a.settle(60)
        a.totals["prompt"] = 60
        with self.assertRaises(lc.OverBudget) as cm:
            a.acquire(50)
        self.assertEqual(cm.exception.kind, "tokens")
        w = self.lim(wall=30, tpm=0, rpm=0)
        self.clock.t += 31
        with self.assertRaises(lc.OverBudget) as cm:
            w.acquire(1)
        self.assertEqual(cm.exception.kind, "wall")

    def test_the_history_the_scaler_reads(self):
        a = self.lim(tpm=1000, rpm=100, reserve_pct=0)
        for _ in range(4):
            a.acquire(100)
            a.settle(500)
        a.block(0.0, "HTTP 429 (retry 1 of 12)")
        st = lc.window_stats(
            settings(self.d, tpm=1000, rpm=100), 5, now=self.clock.t + 1
        )
        self.assertEqual(
            (st["tokens"], st["requests"], st["rate_limited"]), (2000, 4, 1)
        )
        self.assertEqual(st["share_tokens"], 0.4)
        self.assertEqual(st["share"], 0.4)
        self.assertEqual(
            lc.window_stats(settings(self.d), 5, now=self.clock.t + 400)["requests"], 0
        )
        a.close()
        self.assertEqual(
            lc.window_stats(settings(self.d), 5, now=self.clock.t)["active"], 0
        )

    def test_requests_in_flight_wait_for_a_free_place(self):
        a = self.lim(concurrency=1, tpm=0, rpm=0)
        b = self.lim(concurrency=1, tpm=0, rpm=0)
        a.acquire(1)  # in flight until it settles
        self.assertEqual(
            lc.window_stats(settings(self.d), 5, now=self.clock.t)["inflight"], 1
        )
        waits = []

        def sleep(s):
            waits.append(s)
            a.settle(1)  # the request in flight ends while b waits
            self.clock.sleep(s)

        b.sleep = sleep
        b.acquire(1)
        self.assertEqual(len(waits), 1)
        self.assertIn(
            "1 of 1 requests in flight",
            (self.d / "limits" / lc.LIMITER_LOG).read_text(),
        )
        b.settle(1)
        self.assertEqual(
            lc.window_stats(settings(self.d), 5, now=self.clock.t)["inflight"], 0
        )


class TestThreadsAndPhases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)
        self.clock = Clock()

    def tearDown(self):
        self.tmp.cleanup()

    def test_threads_share_the_run_with_their_own_tags_and_tickets(self):
        s = settings(self.d, tpm=0, rpm=0)
        lim = lc.limiter_for(s, who="t")

        def post(s_, body, key):
            time.sleep(0.01)
            return answer("ok", usage={"prompt_tokens": 10, "completion_tokens": 2})

        def work(name):
            lim.tags = {"pass": name, "step": 0}
            for _ in range(5):
                lc.chat(s, [{"role": "user", "content": name}], limiter=lim, post=post)

        threads = [threading.Thread(target=work, args=(n,)) for n in ("prior", "hunt")]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        summary = lim.summary()
        self.assertEqual((summary["requests"], summary["tokens"]), (10, 120))
        self.assertEqual(
            {k: v["requests"] for k, v in summary["passes"].items()},
            {"prior": 5, "hunt": 5},
        )
        lines = [
            json.loads(x)
            for x in (self.d / "limits" / lc.TOKENS_LOG).read_text().splitlines()
        ]
        self.assertEqual(sorted(x["pass"] for x in lines), ["hunt"] * 5 + ["prior"] * 5)
        st = json.loads(lc.bucket_path(s).read_text())
        self.assertEqual(len({w[2] for w in st["window"]}), 10)  # a ticket a request
        self.assertEqual({w[1] for w in st["window"]}, {12})  # each settled its own
        self.assertEqual(st["inflight"], {})

    def test_a_phase_has_its_own_wall_time_and_token_budget(self):
        lim = lc.Limiter(
            settings(self.d, tpm=0, rpm=0),
            who="t",
            clock=self.clock,
            sleep=self.clock.sleep,
        )
        lim.phase("prior", 30, tokens=100)
        self.assertEqual(lim.deadline_now(), self.clock.t + 30)
        lim.acquire(10)
        lim.settle(70)
        lim.tags = {"pass": "prior"}
        lim.count(
            {"prompt": 60, "completion": 10, "reasoning": 0, "reported": True}, 0, 0
        )
        with self.assertRaises(lc.OverBudget) as cm:
            lim.acquire(50)
        self.assertEqual((cm.exception.kind, cm.exception.scope), ("tokens", "phase"))
        self.assertIn(
            "the prior phase's token budget (100) would be passed: 70 used",
            str(cm.exception),
        )
        got = []
        t = threading.Thread(
            target=lambda: got.append(lim.acquire(50))
        )  # another thread: no phase holds it
        t.start()
        t.join()
        self.assertEqual(got, [0.0])
        self.clock.t += 31
        with self.assertRaises(lc.OverBudget) as cm:
            lim.acquire(1)
        self.assertEqual((cm.exception.kind, cm.exception.scope), ("wall", "phase"))
        self.assertIn("the prior phase's wall time (30 s) is spent", str(cm.exception))
        lim.phase()
        self.assertEqual(lim.acquire(1), 0.0)  # back to the run's caps
        run = lc.Limiter(
            settings(self.d, tpm=0, rpm=0),
            who="t",
            wall_s=60,
            clock=self.clock,
            sleep=self.clock.sleep,
        )
        run.phase("hunt", 600)
        self.assertEqual(run.deadline_now(), run.deadline)  # the run's cap comes first

    def test_a_request_ends_by_its_deadline(self):
        lines = [b'data: {"choices": [{"delta": {"content": "x"}}]}', b"data: [DONE]"]
        with self.assertRaises(lc.OverBudget):
            lc.OpenAICompatible.stream_answer(lines, deadline=time.time() - 1)
        self.assertEqual(
            lc.OpenAICompatible.stream_answer(lines, deadline=time.time() + 60)[
                "content"
            ],
            "x",
        )
        s = settings(self.d, tpm=0, rpm=0, concurrency=1)
        with self.assertRaises(lc.OverBudget):
            lc.OpenAICompatible().post(
                s, {}, "", deadline=time.time() - 1
            )  # no connection once it has passed
        lim = lc.limiter_for(s, who="t", run=False)
        lim.phase("hunt", 0.05)

        def cut(s_, body, key):
            time.sleep(0.1)
            raise lc.OverBudget(
                "wall", "the answer was still streaming at its deadline", scope="phase"
            )

        with self.assertRaises(lc.OverBudget) as cm:
            lc.chat(s, [{"role": "user", "content": "x"}], limiter=lim, post=cut)
        self.assertEqual(cm.exception.scope, "phase")
        self.assertIn("the hunt phase's wall time", str(cm.exception))
        self.assertEqual(
            lc.window_stats(s)["inflight"], 0
        )  # its place in flight is given back


if __name__ == "__main__":
    unittest.main()
