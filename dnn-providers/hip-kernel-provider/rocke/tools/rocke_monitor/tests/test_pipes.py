# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Parallel reviews (up to 5, three by default): up to the number in effect run at once, a local Ollama runs one unless
the operator confirmed more (the memory cap), every parallel review on a remote endpoint shares one file-locked rate
budget (the rate-limit cap, HTTP 429 with its Retry-After and backoff), and the checks limit on local builds and GPU
checks applies however many run. The review driver stays one process and prints REVIEW_READY for each finished draft.
Fake backends only: no model is called."""

import email.utils
import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import reviewer_client  # noqa: E402
import test_models  # noqa: E402  (the fake gateway)
from rkmon_lib import config, jobs, review  # noqa: E402
from rkmon_lib.store import Store, empty_state  # noqa: E402

OLD = "2000-01-01T00:00:00Z"
REMOTE = {
    "backend": "openai",
    "url": "https://models.example.com/route",
    "model": "Big.Model-v5.3-Flash",
}
LOCAL = {"backend": "ollama", "url": "http://127.0.0.1:11434", "model": "m:1"}


def a_cfg(work: pathlib.Path, worker: dict, pipes: int = 3, checks: int = 1) -> dict:
    c = config.load_defaults()
    c.pop("_doc", None)
    c.update(
        repo="acme/widgets",
        rocke_path="a/b/rocke",
        operator={"github_login": "op-login"},
        scope={"labels": ["x"]},
        worker=worker,
        review={"pipes": pipes},
        checks={"limit": checks},
        deployment=dict(c["deployment"], phase="review"),
        reviewer="worker",
    )
    c["paths"] = dict(c["paths"], working_dir=str(work))
    return config.normalize(c)


class Clock:
    """A fake clock: sleeping moves it on."""

    def __init__(self):
        self.t, self.slept = 1000.0, []

    def time(self):
        return self.t

    def sleep(self, s):
        self.slept.append(s)
        self.t += s


class _State(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.tmp.name)
        self.work = self.base / "work"
        self.work.mkdir()
        self.store = Store(self.base / "state")
        self.st = empty_state()
        self.st["prs"] = {
            str(n): {
                "n": n,
                "head": str(n) * 40,
                "status": "open",
                "draft": False,
                "size": 10,
            }
            for n in range(1, 7)
        }
        self.plan = [
            {"kind": "FIRST_ROUND", "pr": n, "text": f"first-round review of #{n}"}
            for n in range(1, 7)
        ]
        self.started = []

    def tearDown(self):
        self.tmp.cleanup()

    def fake_start(self, cfg, st, store, n, kind, again=False):
        """start_review without git or jobs: a pipe with a worker that has run for a long time."""
        self.started.append(n)
        review.pipes(st).append(
            {
                "pr": n,
                "kind": kind,
                "head": str(n) * 40,
                "since": None,
                "started": OLD,
                "stage": "worker",
                "worker": {"id": f"w{n}", "started": OLD},
                "checks": None,
            }
        )
        return [f"REVIEW_START #{n} ({kind})"]


class TestPipes(_State):
    def test_up_to_the_pipes_in_effect_run_and_each_draft_is_ready_on_its_own(self):
        cfg = a_cfg(self.work, REMOTE, pipes=3)
        status = {}
        with mock.patch.object(
            review, "start_review", self.fake_start
        ), mock.patch.object(
            review.jobs,
            "status",
            lambda store, job: (
                status.get((job or {}).get("id"), "running") if job else "none"
            ),
        ):
            lines = review._fill(cfg, self.st, self.store, self.plan)
            self.assertEqual(
                lines, [f"REVIEW_START #{n} (first-round)" for n in (1, 2, 3)]
            )
            self.assertEqual([p["pr"] for p in review.pipes(self.st)], [1, 2, 3])
            self.assertEqual(
                review._fill(cfg, self.st, self.store, self.plan), []
            )  # every pipe busy
            # #2's draft and checks are done: REVIEW_READY for it alone, and its pipe takes the next review
            review.pipe_of(self.st, 2)["checks"] = {"id": "c2", "started": OLD}
            status.update(w2="done", c2="done")
            with mock.patch.object(
                review, "has_draft", lambda cfg, n: n == 2
            ), mock.patch.object(
                review, "_record_checks_gates", lambda cfg, rec, n: None
            ), mock.patch.object(
                review, "worker_prefill", lambda cfg, st: True
            ):  # (no checks start here)
                lines = review.advance(cfg, self.st, self.store, self.plan)
        ready = [ln for ln in lines if ln.startswith("REVIEW_READY")]
        self.assertEqual(len(ready), 1)
        self.assertTrue(ready[0].startswith("REVIEW_READY #2 (first-round)"))
        self.assertIn("REVIEW_START #4 (first-round)", lines)
        self.assertEqual(sorted(p["pr"] for p in review.pipes(self.st)), [1, 3, 4])
        self.assertEqual([r["pr"] for r in self.st["review_ready"]], [2])

    def test_a_local_ollama_runs_one_draft_unless_more_were_confirmed(self):
        with mock.patch.object(review, "start_review", self.fake_start):
            review._fill(
                a_cfg(self.work, LOCAL, pipes=3), self.st, self.store, self.plan
            )
            self.assertEqual(self.started, [1])  # the memory cap
            self.st["review_pipes"] = []
            self.started.clear()
            review._fill(
                a_cfg(self.work, dict(LOCAL, max_parallel=2), pipes=3),
                self.st,
                self.store,
                self.plan,
            )
            self.assertEqual(self.started, [1, 2])
        p = config.pipes(a_cfg(self.work, LOCAL, pipes=3))
        self.assertEqual((p["asked"], p["effective"]), (3, 1))

    def test_nothing_new_starts_while_draining_or_stopped(self):
        cfg = a_cfg(self.work, REMOTE)
        with mock.patch.object(review, "start_review", self.fake_start):
            self.st["draining"] = {"since": OLD}
            self.assertEqual(review._fill(cfg, self.st, self.store, self.plan), [])
            del self.st["draining"]
            self.st["stopped"] = {"at": OLD, "how": "stop", "boot": ""}
            self.assertEqual(review._fill(cfg, self.st, self.store, self.plan), [])
        self.assertEqual(self.started, [])

    def test_the_checks_limit_holds_whatever_the_pipes(self):
        """Five pipes, one heavy slot: one checks job runs; the others wait for the slot."""
        started = []
        for n in range(1, 6):
            (self.work / f"wt{n}m").mkdir()
            review.pipes(self.st).append(
                {
                    "pr": n,
                    "kind": "first-round",
                    "head": str(n) * 40,
                    "since": None,
                    "started": OLD,
                    "stage": "worker",
                    "worker": {"id": f"w{n}", "started": OLD},
                    "checks": None,
                }
            )

        def start(store, kind, name, script, timeout_min):
            job = {"id": f"{kind}-{name}", "started": OLD}
            started.append(job["id"])
            return job

        for limit, want in ((1, 1), (2, 2)):
            with self.subTest(limit=limit):
                cfg = a_cfg(self.work, REMOTE, pipes=5, checks=limit)
                self.assertEqual(
                    config.checks_limit(cfg), limit
                )  # review.pipes 5 does not move it
                for p in review.pipes(self.st):
                    p["checks"] = None
                self.st["jobs"]["heavy"] = None
                started.clear()
                with mock.patch.object(review.jobs, "start", start), mock.patch.object(
                    review.jobs,
                    "status",
                    lambda store, job: "running" if job else "none",
                ), mock.patch.object(
                    review, "_changed_files", lambda cfg, rec: []
                ), mock.patch.object(
                    review, "pick_next", lambda *a, **k: None
                ):
                    lines = review.advance(cfg, self.st, self.store, self.plan)
                self.assertEqual(len(started), want, lines)
                self.assertEqual(len(jobs.heavy_holders(self.st)), want)
                waiting = [
                    ln
                    for ln in lines
                    if "checks wait: all parallel builds are busy" in ln
                ]
                self.assertEqual(len(waiting), 5 - want, lines)

    def test_the_worker_unloads_a_local_ollama_only_when_no_other_pipe_uses_it(self):
        cfg = a_cfg(self.work, LOCAL)
        alone = review.worker_script(cfg, 7, [], "first-round", unload=True)
        shared = review.worker_script(cfg, 7, [], "first-round", unload=False)
        self.assertIn("ollama stop", alone)
        self.assertNotIn("ollama stop", shared)
        self.assertIn("RKMON_PIPE=pr7", shared)


class TestPromptBudget(unittest.TestCase):
    def test_the_prompt_follows_the_models_context(self):
        self.assertEqual(
            reviewer_client.prompt_budget(262144), (650000, 150000)
        )  # the default: as before
        self.assertEqual(
            reviewer_client.prompt_budget(131072), (393216, 131072)
        )  # a gateway model's smaller context
        self.assertEqual(reviewer_client.prompt_budget(None), (650000, 150000))


class TestBudget(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.run = pathlib.Path(self.tmp.name) / "run"
        self.clock = Clock()
        self.said = []

    def tearDown(self):
        self.tmp.cleanup()

    def budget(self, **limits):
        return reviewer_client.Budget(
            dict(REMOTE, **limits),
            str(self.run),
            log=self.said.append,
            clock=self.clock.time,
            sleep=self.clock.sleep,
            who="pr7",
        )

    def test_every_pipe_shares_one_requests_budget_and_a_wait_is_logged(self):
        a, b = self.budget(rpm=2), self.budget(rpm=2)  # two pipes, one endpoint
        self.assertEqual(a.acquire(100), 0)
        self.assertEqual(b.acquire(100), 0)
        self.clock.t += 10
        waited = a.acquire(
            100
        )  # the third request of the minute waits for the first to leave the window
        self.assertAlmostEqual(waited, 50 + reviewer_client.SLACK_S, delta=0.01)
        self.assertTrue(
            self.said
            and "waiting" in self.said[0]
            and "2 of 2 requests" in self.said[0],
            self.said,
        )
        log = (self.run / "budget.log").read_text(encoding="utf-8")
        self.assertIn("pr7 budget: waiting", log)

    def test_the_tokens_budget_and_a_request_larger_than_it(self):
        t = self.budget(tpm=1000)
        self.assertEqual(t.acquire(600), 0)
        t.record(100)
        self.assertGreater(t.acquire(400), 0)  # 700 + 400 > 1000: it waits
        self.clock.t += 120
        self.assertEqual(
            t.acquire(5000), 0
        )  # larger than the whole budget: goes once the window is empty

    def test_a_request_leaves_the_window_a_slack_after_its_minute(self):
        a = self.budget(rpm=1)
        self.assertEqual(a.acquire(10), 0)
        self.clock.t += (
            60  # the endpoint's clock may still count the first request in its minute
        )
        self.assertAlmostEqual(a.acquire(10), reviewer_client.SLACK_S, delta=0.01)

    def test_each_answer_adds_its_tokens_to_its_own_request(self):
        a, b = self.budget(tpm=10000), self.budget(tpm=10000)  # two pipes, one endpoint
        a.acquire(600)
        self.clock.t += 5
        b.acquire(100)  # b asks while a's answer is still on its way
        a.record(50)
        b.record(7)
        window = json.loads(
            next(self.run.glob("budget-*.json")).read_text(encoding="utf-8")
        )["window"]
        self.assertEqual([w[1] for w in window], [650, 107])

    def state(self, **st) -> pathlib.Path:
        """The shared budget's file, as other pipes left it."""
        path = self.budget().path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(dict({"window": [], "blocked_until": 0.0, "queue": []}, **st)),
            encoding="utf-8",
        )
        return path

    def test_the_pipes_that_asked_first_go_first(self):
        # Under a limit of 4 requests a minute, no draft's second request takes the slot a third draft has waited
        # for since the first minute.
        t0 = self.clock.t
        path = self.state(
            window=[[t0, 10, "a-first"]], queue=[["c-first", t0 + 1, os.getpid()]]
        )
        self.clock.t += 5
        a = self.budget(rpm=2)
        waited = a.acquire(
            10
        )  # a's second request: one slot is free, but c asked first and keeps it
        self.assertAlmostEqual(
            waited,
            56 + reviewer_client.SLACK_S,
            delta=reviewer_client.QUEUE_POLL_S + 0.01,
        )
        st = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(
            [q[0] for q in st["queue"]], ["c-first"]
        )  # c keeps its place; a's is gone
        self.assertEqual(
            sum("asked first and go first" in s for s in self.said), 1, self.said
        )  # said once

    def test_a_place_is_dropped_with_its_process_or_when_its_pipe_stops_waiting(self):
        gone = subprocess.Popen([sys.executable, "-c", "pass"])
        gone.wait()
        path = self.state(
            window=[[self.clock.t, 10, "x"]], queue=[["dead", self.clock.t, gone.pid]]
        )
        self.assertEqual(
            self.budget(rpm=2).acquire(10), 0
        )  # the dead pipe's place holds nothing
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["queue"], [])

        def interrupted(_s):
            raise KeyboardInterrupt

        b = reviewer_client.Budget(
            dict(REMOTE, rpm=2),
            str(self.run),
            log=self.said.append,
            clock=self.clock.time,
            sleep=interrupted,
        )
        with self.assertRaises(KeyboardInterrupt):
            b.acquire(10)  # the minute is full: it waits, and is stopped while waiting
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["queue"], [])

    def test_a_model_role_keeps_within_its_share_and_never_holds_the_pipes_back(self):
        """The planner and the analyst (rkmon_lib/roles.py) take from the reviewer's own budget file, each within its
        share of the limits; a role over its share waits outside the queue, so the pipes never wait behind it.
        """
        pipe = self.budget(rpm=4, tpm=1000)
        queues = []

        def sleep(s):  # what the shared file holds while the role waits
            queues.append(json.loads(pipe.path.read_text(encoding="utf-8"))["queue"])
            self.clock.sleep(s)

        def role(**limits):
            return reviewer_client.Budget(
                dict(
                    REMOTE,
                    **limits,
                    model="Plan.Model-1",
                    budget_role="planner",
                    budget_share=0.25,
                    budget_label=f"{REMOTE['url']} {REMOTE['model']}",
                ),
                str(self.run),
                log=self.said.append,
                clock=self.clock.time,
                sleep=sleep,
                who="planner",
            )

        planner = role(rpm=4, tpm=1000)
        self.assertEqual(planner.path, pipe.path)
        self.assertEqual(planner.acquire(100), 0)  # its one request a minute (4 x 0.25)
        self.assertEqual(
            [pipe.acquire(100) for _ in range(3)], [0, 0, 0]
        )  # the pipes have the rest
        self.clock.t += 5
        self.assertAlmostEqual(
            planner.acquire(100), 55 + reviewer_client.SLACK_S, delta=0.01
        )
        self.assertEqual(queues, [[]])  # waiting, it held no place in the queue
        self.assertTrue(
            any(
                "the planner's share: 1 of 1 requests in the last minute" in s
                for s in self.said
            ),
            self.said,
        )
        self.clock.t += 120
        role, pipe = role(tpm=1000), self.budget(tpm=1000)  # a tokens limit only
        self.assertEqual((role.acquire(200), pipe.acquire(500)), (0, 0))
        self.assertGreater(
            role.acquire(100), 0
        )  # 200 + 100 > 250 tokens, its share of 1000; 800 would fit the whole
        self.assertTrue(
            any(
                "the planner's share: 200 + 100 of 250 tokens in the last minute" in s
                for s in self.said
            ),
            self.said,
        )

    def test_a_block_makes_every_pipe_wait(self):
        a, b = self.budget(), self.budget()
        a.block(30, "HTTP 429 from the endpoint")
        self.assertAlmostEqual(b.acquire(10), 30, delta=1)
        self.assertTrue(
            any("every review waits 30 s" in s for s in self.said), self.said
        )

    def test_the_rate_limit_headers(self):
        w = reviewer_client.wait_from_headers
        self.assertEqual(w([("Retry-After", "7")]), 7)
        self.assertEqual(w([("retry-after-ms", "1500")]), 1.5)
        base = email.utils.parsedate_to_datetime(
            "Thu, 01 Jan 2099 00:00:00 GMT"
        ).timestamp()
        self.assertAlmostEqual(
            w([("Retry-After", "Thu, 01 Jan 2099 00:00:10 GMT")], now=base), 10, delta=1
        )
        self.assertEqual(
            w(
                [
                    ("x-ratelimit-remaining-requests", "0"),
                    ("x-ratelimit-reset-requests", "6m0s"),
                ]
            ),
            360,
        )
        self.assertEqual(
            w(
                [
                    ("x-ratelimit-remaining-tokens", "0"),
                    ("x-ratelimit-reset-tokens", "1.5s"),
                ]
            ),
            1.5,
        )
        self.assertIsNone(w([("x-ratelimit-remaining-requests", "5")]))


class TestRateLimitedGateway(test_models._Base):
    """An HTTP 429 from the gateway: retried after its Retry-After (or the backoff), every pipe waiting with it."""

    def test_a_429_is_retried_after_its_retry_after_then_with_backoff(self):
        clock = Clock()
        run = self.base / "run"
        s = self.s()
        test_models.Gateway.fail429, test_models.Gateway.retry_after = 2, "3"
        budget = reviewer_client.Budget(
            s, str(run), log=lambda m: None, clock=clock.time, sleep=clock.sleep
        )
        msg = reviewer_client.chat(
            s,
            [{"role": "user", "content": "hi"}],
            tools=[reviewer_client.PROBE_TOOL],
            budget=budget,
        )
        self.assertEqual(msg["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(
            clock.slept, [3, 3]
        )  # the endpoint's own Retry-After, each time
        log = (run / "budget.log").read_text(encoding="utf-8")
        self.assertEqual(log.count("HTTP 429"), 2)
        test_models.Gateway.fail429, test_models.Gateway.retry_after = (
            2,
            "",
        )  # no Retry-After: the backoff
        clock.slept.clear()
        reviewer_client.chat(s, [{"role": "user", "content": "hi"}], budget=budget)
        self.assertEqual(
            clock.slept, [reviewer_client.BACKOFF_S, reviewer_client.BACKOFF_S * 2]
        )
        test_models.Gateway.fail429, test_models.Gateway.retry_after = 99, "1"
        with self.assertRaises(reviewer_client.RateLimited):
            reviewer_client.chat(
                s, [{"role": "user", "content": "hi"}], budget=budget, rate_waits=2
            )


class TestDriver(_State):
    def run_driver(self, rounds, store=None):
        import rkmon
        from rkmon_lib.store import now

        s = store or self.store
        st = s.load()
        if not st.get(
            "last_tick"
        ):  # a tick just ran: the state is current (else the driver says TICK_STALE)
            st["last_tick"] = now()
            s.save(st)
        out = []
        calls = iter(rounds)

        def advance(cfg, st, store, plan):
            return next(calls)

        with mock.patch.object(
            rkmon, "load_config", lambda: a_cfg(self.work, REMOTE)
        ), mock.patch.object(rkmon.review, "advance", advance), mock.patch.object(
            rkmon.daylog, "from_lines", lambda *a: None
        ):
            rc = rkmon.cmd_review_driver(
                a_cfg(self.work, REMOTE),
                store or self.store,
                sleep=lambda s: None,
                out=out.append,
            )
        return rc, out

    def test_one_driver_prints_each_ready_draft_and_exits(self):
        rc, out = self.run_driver(
            [
                [
                    "REVIEW #1: worker running, checks none",
                    "REVIEW #2: worker running, checks none",
                ],
                [
                    "REVIEW_READY #1 (first-round): ...",
                    "REVIEW_READY #2 (follow-up): ...",
                    "REVIEW_START #3 (first-round)",
                ],
            ]
        )
        self.assertEqual(rc, 0)
        self.assertIn(
            "REVIEW_READY #1 (first-round): ...\nREVIEW_READY #2 (follow-up): ...",
            out[-1],
        )
        self.assertTrue(out[0].endswith("REVIEW #1: worker running, checks none"))
        rc, out = self.run_driver([[]])
        self.assertIn("nothing to start now", out[-1])

    def test_a_second_driver_and_a_stopped_monitor(self):
        import fcntl

        with open(self.store.dir / "review_driver.lock", "a+") as held:
            fcntl.flock(held.fileno(), fcntl.LOCK_EX)
            rc, out = self.run_driver([["REVIEW #1: worker running"]])
        self.assertEqual(out, ["DRIVER: another review driver is already running"])
        st = self.store.load()
        st["stopped"] = {"at": OLD, "how": "stop", "boot": ""}
        self.store.save(st)
        rc, out = self.run_driver([["REVIEW #1: worker running"]])
        self.assertTrue(out[-1].startswith("DRIVER: the monitor is stopped"), out)

    def test_a_configuration_it_cannot_load_stops_it_naming_the_fix(self):
        """A configuration the driver cannot load stops it, exit 4, as rk refuses one (applied.cannot_run): why,
        then the fix the error names."""
        import rkmon
        from rkmon_lib import checks

        def refused():
            raise config.ConfigError("x.y is blank", checks.Command(("rkmon", "apply")))

        out = []
        with mock.patch.object(rkmon, "load_config", refused):
            rc = rkmon.cmd_review_driver(
                {}, self.store, sleep=lambda s: None, out=out.append
            )
        self.assertEqual(rc, 4)
        self.assertEqual(
            out,
            [
                "DRIVER: the configuration cannot run: x.y is blank",
                "fix: rkmon apply",
            ],
        )

    def test_each_round_advances_with_the_configuration_of_that_round(self):
        """Each round loads the configuration again: a setting that takes effect with no restart, changed while the
        driver runs (here the pipes, 3 then 2), is the one its next round's advance reads.
        """
        import rkmon
        from rkmon_lib.store import now

        st = self.store.load()
        st["last_tick"] = now()
        self.store.save(st)
        configs = iter([a_cfg(self.work, REMOTE, pipes=p) for p in (3, 2)])
        rounds = iter(
            [["REVIEW #1: worker running"], ["REVIEW_READY #1 (first-round): ..."]]
        )
        read = []

        def advance(cfg, st, store, plan):
            read.append(cfg["review"]["pipes"])
            return next(rounds)

        out = []
        with mock.patch.object(
            rkmon, "load_config", lambda: next(configs)
        ), mock.patch.object(rkmon.review, "advance", advance), mock.patch.object(
            rkmon.daylog, "from_lines", lambda *a: None
        ):
            rc = rkmon.cmd_review_driver(
                a_cfg(self.work, REMOTE),
                self.store,
                sleep=lambda s: None,
                out=out.append,
            )
        self.assertEqual(rc, 0)
        self.assertEqual(read, [3, 2])


if __name__ == "__main__":
    unittest.main()
