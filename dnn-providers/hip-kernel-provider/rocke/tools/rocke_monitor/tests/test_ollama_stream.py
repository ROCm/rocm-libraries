# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Tests for ollama_stream: a fake /api/chat server that answers, hangs, or answers late, and fake runner progress."""

import http.server
import json
import threading
import time
import unittest

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "worker"))
import ollama_stream as S  # noqa: E402

H = "runner-host ollama[208]:"


class _Handler(http.server.BaseHTTPRequestHandler):
    mode = "answer"
    delay = 0.0

    def log_message(self, *a):
        pass

    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        if self.mode == "hang":
            time.sleep(6)
            return
        time.sleep(self.delay)
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.end_headers()
        for c in (
            {"message": {"thinking": "th"}},
            {"message": {"content": "hel"}},
            {"message": {"content": "lo", "tool_calls": [{"function": {"name": "x"}}]}},
            {"done": True},
        ):
            self.wfile.write((json.dumps(c) + "\n").encode())
            self.wfile.flush()


class StreamChat(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.srv.daemon_threads = True
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.srv.server_address[1]}/api/chat"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def call(self, mode, progress, delay=0.0, stall_s=4, hang_s=1):
        _Handler.mode, _Handler.delay = mode, delay
        return S.stream_chat(
            self.url,
            b'{"stream": true}',
            stall_s,
            hang_s=hang_s,
            poll_s=0.2,
            progress=progress,
        )

    def test_a_normal_stream_is_joined_into_one_message(self):
        m = self.call("answer", lambda since: (None, None))
        self.assertEqual(
            (m["content"], m["thinking"], len(m["tool_calls"])), ("hello", "th", 1)
        )

    def test_a_prompt_with_no_progress_is_a_hung_runner_well_before_the_stall_timeout(
        self,
    ):
        t0 = time.time()
        with self.assertRaises(S.RunnerHung) as e:
            self.call("hang", lambda since: (since + 0.05, None))
        self.assertLess(time.time() - t0, 3)
        self.assertIn("first prompt batch", str(e.exception))

    def test_progress_that_stops_is_a_hung_runner(self):
        with self.assertRaises(S.RunnerHung) as e:
            self.call("hang", lambda since: (since + 0.05, since + 0.1))
        self.assertIn("after a prompt batch", str(e.exception))

    def test_without_a_prompt_line_only_the_stall_timeout_applies(self):
        t0 = time.time()
        with self.assertRaises(TimeoutError) as e:
            self.call("hang", lambda since: (None, None), stall_s=2)
        self.assertNotIsInstance(e.exception, S.RunnerHung)
        self.assertGreaterEqual(time.time() - t0, 1.9)

    def test_a_long_prefill_that_keeps_progressing_is_not_cut_off(self):
        m = self.call(
            "answer",
            lambda since: (since + 0.05, time.time() - 0.05),
            delay=2.5,
            hang_s=1,
        )
        self.assertEqual(m["content"], "hello")

    def test_a_prefill_longer_than_the_stall_timeout_is_kept_while_it_progresses(self):
        """A long prompt's prefill is kept while it progresses; the stall timeout does not count from the send."""
        m = self.call(
            "answer",
            lambda since: (since + 0.05, time.time() - 0.05),
            delay=3.0,
            stall_s=1.5,
            hang_s=1,
        )
        self.assertEqual(m["content"], "hello")


MS = "ms"  # a server log's unit, kept apart so no source line reads as a performance result


class ParseProgress(unittest.TestCase):
    def test_a_hung_first_batch_and_a_healthy_one(self):
        hung = "\n".join(
            [
                f"100.0 {H} slot   operator(): id  0 | task 0 | new prompt, n_ctx_slot = 262144, n_keep = 4, task.n_tokens = 40667",
                f"100.1 {H} slot   operator(): id  0 | task 0 | cached n_tokens = 0, memory_seq_rm [0, end)",
                f"100.2 {H} sched_reserve: reserve took 79.86 {MS}, sched copies = 1",
            ]
        )
        self.assertEqual(S.parse_progress(hung, 99.0), (100.0, None))
        ok = (
            hung
            + "\n"
            + "\n".join(
                [
                    f"100.6 {H} slot   operator(): id  0 | task 0 | cached n_tokens = 256, memory_seq_rm [256, end)",
                    f"100.9 {H} slot print_timing: id  0 | task 0 | prompt processing, n_tokens =    512, progress = 0.01",
                ]
            )
        )
        self.assertEqual(S.parse_progress(ok, 99.0), (100.0, 100.9))

    def test_lines_before_the_request_and_an_older_prompt_are_ignored(self):
        text = "\n".join(
            [
                f"50.0 {H} slot   operator(): id  0 | task 7 | new prompt, n_ctx_slot = 262144, n_keep = 4, task.n_tokens = 9",
                f"60.0 {H} slot   operator(): id  0 | task 7 | cached n_tokens = 256, memory_seq_rm [256, end)",
                "-- No entries --",
                f"200.0 {H} slot   operator(): id  0 | task 8 | new prompt, n_ctx_slot = 262144, n_keep = 4, task.n_tokens = 9",
            ]
        )
        self.assertEqual(S.parse_progress(text, 100.0), (200.0, None))
        self.assertEqual(S.parse_progress("", 100.0), (None, None))

    def test_a_restored_checkpoint_counts_as_progress(self):
        text = "\n".join(
            [
                f"300.0 {H} slot   operator(): id  0 | task 419 | new prompt, n_ctx_slot = 262144, n_keep = 4, task.n_tokens = 41543",
                f"300.1 {H} slot   operator(): id  0 | task 419 | restored context checkpoint (pos_min = 41462, n_tokens = 41463)",
            ]
        )
        self.assertEqual(S.parse_progress(text, 299.0), (300.0, 300.1))


if __name__ == "__main__":
    unittest.main()
