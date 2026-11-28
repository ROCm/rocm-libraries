# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The drain (rkmon_lib/drain.py; apply's step 2): `rk drain` holds a flag under which nothing new starts
until the work in flight has finished, then leaves it in place for the caller; at its deadline it lifts the flag and
fails; `rk undrain` lifts it, and a pause you set stays; a flag whose process is gone holds nothing; another drain
that holds refuses a second; a drain cut short lifts its flag; and rk stop leaves rk drain alone, which sees the stop
and ends, nothing being left to drain. Its outcome words, its limit and its looks are rk stop --drain's too
(drain.limits).
"""

import contextlib
import datetime as dt
import inspect
import io
import json
import os
import pathlib
import select
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.dont_write_bytecode = True

import rkmon  # noqa: E402
import sandbox  # noqa: E402
from rkmon_lib import (
    config,
    control,
    drain,
    registry,
    review,
    slots,
    store,
)  # noqa: E402

# One look at the work in flight a minute.
CFG = {"update": {"poll_s": 60, "drain_timeout_min": 60}}
STOP_DRAIN = "rk stop --drain"
# `rk drain` in a process of its own, as rk runs it (rkmon.py registers it, REGISTERED), the work in flight never
# finishing (argv: the package folder).
RK_DRAIN = """
import sys
from unittest import mock

sys.path.insert(0, sys.argv[1])
sys.dont_write_bytecode = True
import rkmon
from rkmon_lib import drain, review

with mock.patch.object(review, "advance", return_value=[]), mock.patch.object(
    drain, "in_flight", return_value=["review #7"]
):
    sys.exit(rkmon.main(["drain", "--timeout-minutes", "5"]))
"""


class Base(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.store = store.Store(pathlib.Path(tmp.name) / "state")
        self.said: list[str] = []
        self.minute = 0.0  # the drain's clock, in minutes; each sleep moves it on
        self.advance = mock.Mock(return_value=[])
        patch = mock.patch.object(review, "advance", self.advance)
        patch.start()
        self.addCleanup(patch.stop)

    def clock(self) -> float:
        return self.minute * 60

    def sleep(self, seconds: float) -> None:
        self.minute += seconds / 60

    def flying(self, *looks):
        """in_flight answering LOOKS in turn, the last one from then on; each a list, or a callable of the state."""
        looks = list(looks)

        def answer(st, _store):
            look = looks.pop(0) if len(looks) > 1 else looks[0]
            return look(st) if callable(look) else look

        return mock.patch.object(drain, "in_flight", answer)

    def drain(self, timeout_min=None, cfg=CFG) -> int:
        return drain.main(
            cfg,
            self.store,
            timeout_min=timeout_min,
            say=self.said.append,
            sleep=self.sleep,
            clock=self.clock,
        )

    def flag(self) -> dict | None:
        return self.store.load().get("draining")

    def put(self, **fields) -> None:
        with self.store.locked(wait=True):
            st = self.store.load()
            st.update(fields)
            self.store.save(st)


class TestDrain(Base):
    def test_the_flag_holds_while_the_work_in_flight_finishes_then_stays_for_the_caller(
        self,
    ):
        seen = []

        def look(st):
            d = st["draining"]
            seen.append((d["why"], d["pid"], d["until_min"], store.halted(st)[:33]))
            return ["review #7"]

        with self.flying(["review #7", "gate on #9"], look, []):
            self.assertEqual(self.drain(), 0)
        self.assertEqual(
            self.said,
            [
                "drain: nothing new starts; the work in flight finishes first (up to 60 minutes): review #7, gate "
                "on #9",
                "drain: still in flight: review #7",
                "DRAINED: nothing is in flight after 2 minutes",
            ],
        )
        self.assertEqual(
            seen, [("rk drain", os.getpid(), 60, "the monitor is draining (rk drain")]
        )
        # The parallel reviews advance at each look, so they finish without a tick.
        self.assertEqual(self.advance.call_count, 3)
        # Drained, the flag stays, no longer this process's: it holds until rk undrain, or DRAIN_MARGIN_MIN after.
        d = self.flag()
        self.assertEqual(
            {k: d.get(k) for k in ("why", "until_min", "pid", "pstart")},
            {"why": "rk drain", "until_min": 2.0, "pid": None, "pstart": None},
        )
        self.assertTrue(store.halted(self.store.load()))
        d["since"] = "2000-01-01T00:00:00Z"
        self.assertEqual(store.drain_stale(d), "it is past its 2 min")

    def test_nothing_in_flight_drains_at_once(self):
        with self.flying([]):
            self.assertEqual(self.drain(timeout_min=5), 0)
        self.assertEqual(
            self.said,
            [
                "drain: nothing new starts; nothing is in flight",
                "DRAINED: nothing is in flight",
            ],
        )
        self.assertEqual(self.flag()["until_min"], 0.0)

    def test_the_deadline_as_given_lifts_the_flag_and_fails(self):
        with self.flying(["gate on #9"]):
            self.assertEqual(self.drain(timeout_min=3), 1)
        self.assertEqual(
            self.said,
            [
                "drain: nothing new starts; the work in flight finishes first (up to 3 minutes): gate on #9",
                "DRAIN TIMEOUT after 3 minutes: gate on #9 still in flight; the drain is lifted",
            ],
        )
        self.assertEqual(self.minute, 3)
        self.assertIsNone(self.flag())
        self.assertEqual(store.halted(self.store.load()), "")

    def test_the_deadline_by_default_is_the_update_sections(self):
        with self.flying(["review #7"]):
            self.assertEqual(self.drain(cfg={"update": {"poll_s": 60}}), 1)
        self.assertEqual(self.minute, 60)
        self.assertEqual(
            self.said[-1],
            "DRAIN TIMEOUT after 60 minutes: review #7 still in flight; the drain is lifted",
        )

    def test_rk_drain_and_rk_stop_drain_wait_as_the_same_settings_say(self):
        """rk drain and rk stop --drain take their limit and their looks from one place (drain.limits): the
        configuration's update.drain_timeout_min and update.poll_s, the registry's defaults where it has none, and
        --timeout-minutes before the configuration's limit."""
        own = {"update": {"drain_timeout_min": 7, "poll_s": 3}}
        signature = inspect.signature(drain.wait)
        for cfg, minutes, limits in (
            (own, None, (7.0, 3.0)),
            ({}, None, (60.0, 20.0)),
            (own, 2.5, (2.5, 3.0)),
        ):
            asked = []

            def wait(*args, **kwargs):
                given = signature.bind(*args, **kwargs).arguments
                asked.append((given["timeout_min"], given["poll_s"]))
                raise drain.Busy(store.drain_record("rk drain", 60))

            with self.subTest(cfg=cfg, minutes=minutes), mock.patch.object(
                drain, "wait", wait
            ):
                self.assertEqual(self.drain(timeout_min=minutes, cfg=cfg), 4)
                self.assertEqual(
                    control.stop(
                        cfg,
                        self.store,
                        drain=True,
                        timeout_minutes=minutes,
                        out=self.said.append,
                        sleep=self.sleep,
                    ),
                    4,
                )
                self.assertEqual(asked, [limits, limits])
                self.assertEqual(drain.limits(cfg, minutes), limits)

    def test_the_registry_describes_the_drain_settings_for_rk_drain_and_rk_stop_drain(
        self,
    ):
        """update.drain_timeout_min and update.poll_s are rk drain's, which an apply and a rollback run, and rk stop
        --drain's: the description of each names the commands and both."""
        for key in (drain.TIMEOUT_KEY, drain.POLL_KEY):
            text = registry.descriptions()[key]
            self.assertIn("rk drain", text, key)
            self.assertIn("an apply and a rollback", text, key)
            self.assertIn(STOP_DRAIN, text, key)

    def test_rk_undrain_lifts_the_flag_and_a_pause_you_set_stays(self):
        out = []
        with self.flying([]):
            self.drain()
        since = self.flag()["since"]
        self.assertEqual(drain.undrain(self.store, out=out.append), 0)
        self.assertEqual(
            out[-1],
            f"UNDRAINED: the drain since {since} is lifted; new work starts again",
        )
        self.assertIsNone(self.flag())
        self.assertEqual(drain.undrain(self.store, out=out.append), 0)
        self.assertEqual(out[-1], "NOTHING TO DO: no drain of rk drain's is there")
        # Paused by you, drained, undrained: the pause stays.
        self.put(enabled=False, paused={"at": "2026-01-02T03:04:05Z"})
        with self.flying([]):
            self.drain()
        drain.undrain(self.store, out=out.append)
        self.assertTrue(
            out[-1].endswith(
                "is lifted; the pause you set at 2026-01-02T03:04:05Z stays (rk resume ends it)"
            ),
            out[-1],
        )
        st = self.store.load()
        self.assertEqual(
            (st.get("enabled"), st.get("paused"), st.get("draining")),
            (False, {"at": "2026-01-02T03:04:05Z"}, None),
        )

    def test_rk_undrain_leaves_another_commands_drain(self):
        self.put(draining=store.drain_record(STOP_DRAIN, 60))
        out = []
        self.assertEqual(drain.undrain(self.store, out=out.append), 0)
        self.assertEqual(out, ["NOTHING TO DO: no drain of rk drain's is there"])
        self.assertEqual(self.flag()["why"], STOP_DRAIN)

    def test_a_flag_whose_process_is_gone_holds_nothing_and_is_taken_over(self):
        gone = dict(store.drain_record(STOP_DRAIN, 60), pstart=-1)
        self.put(draining=gone)
        self.assertEqual(
            store.drain_stale(gone), f"the process that set it ({os.getpid()}) is gone"
        )
        self.assertEqual(store.halted(self.store.load()), "")
        with self.flying([]):
            self.assertEqual(self.drain(), 0)
        self.assertEqual(self.flag()["why"], "rk drain")

    def test_a_drain_that_holds_refuses_another_and_one_left_in_place_is_taken_over(
        self,
    ):
        for why in (STOP_DRAIN, "rk drain"):
            with self.subTest(why=why):
                held = store.drain_record(why, 60)  # this process: one that runs
                self.put(draining=held)
                self.said.clear()
                with self.flying(["review #7"]):
                    self.assertEqual(self.drain(), 4)
                self.assertEqual(
                    self.said,
                    [
                        f"REFUSED: a drain is in force already ({why}, since {held['since']}): it ends with that "
                        "command, or rk start"
                    ],
                )
                self.assertEqual(self.flag(), held)
        # One that rk drain left in place, drained: a second rk drain (apply run again) takes it over.
        left = {k: v for k, v in held.items() if k not in ("pid", "pstart")}
        self.put(draining=left)
        with self.flying([]):
            self.assertEqual(self.drain(), 0)

    def test_a_flag_lifted_or_replaced_meanwhile_ends_the_wait(self):
        # rk undrain lifts this drain's flag; another rk drain's may then take its place, drained and left in place.
        other = {"why": "rk drain", "since": "2026-01-02T03:04:05Z", "until_min": 1.0}
        for then in (None, other):
            with self.subTest(then=then):
                self.minute = 0.0

                def look(st, then=then):
                    drain.lift(self.store, "rk drain")
                    if then:
                        self.put(draining=then)
                    return ["review #7"]

                with self.flying(look):
                    self.assertEqual(self.drain(), 1)
                self.assertEqual(
                    self.said[-1],
                    "UNDRAINED: the flag was lifted meanwhile (rk undrain, or rk start)",
                )
                self.assertEqual(self.minute, 1)
                self.assertEqual(self.flag(), then)

    def test_a_stop_holds_more_than_a_drain_so_there_is_nothing_to_drain(self):
        stop = {"how": "stop", "at": "2026-01-02T03:04:05Z"}
        said = (
            "DRAINED: nothing is left to drain: the monitor is stopped (rk stop, 2026-01-02T03:04:05Z); rk start "
            "resumes it"
        )
        # Stopped before the drain: nothing to wait for, and the reviews do not advance under a stop.
        self.put(stopped=stop)
        with self.flying(["review #7"]):
            self.assertEqual(self.drain(), 0)
        self.assertEqual(self.said, [said])
        self.assertEqual((self.advance.call_count, self.minute), (0, 0))
        self.assertIsNone(self.flag())
        # Stopped while it waits.
        self.put(stopped=None)

        def look(st):
            self.put(stopped=stop)
            return ["review #7"]

        with self.flying(look):
            self.assertEqual(self.drain(), 0)
        self.assertEqual(self.said[-1], said)
        self.assertEqual((self.advance.call_count, self.minute), (1, 1))
        self.assertIsNone(self.flag())

    def test_a_drain_cut_short_lifts_its_flag(self):
        def interrupted(_seconds):
            raise KeyboardInterrupt

        with self.flying(["review #7"]), self.assertRaises(KeyboardInterrupt):
            drain.main(CFG, self.store, say=self.said.append, sleep=interrupted)
        self.assertIsNone(self.flag())

        def terminated(_seconds):
            os.kill(os.getpid(), signal.SIGTERM)

        with self.flying(["review #7"]), self.assertRaises(SystemExit) as end:
            drain.main(CFG, self.store, say=self.said.append, sleep=terminated)
        self.assertEqual(end.exception.code, 128 + signal.SIGTERM)
        self.assertIsNone(self.flag())
        # SIGTERM is Python's own again after the drain.
        self.assertEqual(signal.getsignal(signal.SIGTERM), signal.SIG_DFL)


class TestInFlight(Base):
    def test_in_flight_reads_the_state(self):
        st = {
            "review_pipes": [{"pr": 7}],
            "rechecks": {"8": {}},
            "queue_state": {},
            "guard": {},
            "jobs": {},
        }
        self.assertEqual(
            drain.in_flight(st, self.store), ["review #7", "checks re-run of #8"]
        )
        self.assertEqual(
            drain.in_flight({"review_pipeline": {"pr": 5}}, self.store), ["review #5"]
        )

    def job(self, jid: str) -> dict:
        """A job JID running now, in this process, with a limit of 240 minutes."""
        (self.store.dir / "jobs" / jid).mkdir(parents=True)
        started = dt.datetime.now(dt.timezone.utc)
        return {
            "id": jid,
            "pid": os.getpid(),
            "started": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "timeout_min": 240,
        }

    def test_in_flight_names_every_kind_of_work_and_not_work_done(self):
        st = {
            "review_pipes": {"7": {"pr": 7}},
            "queue_state": {
                "gate": {"pr": 9, "job": self.job("gate")},
                "batch_gate": {"job": self.job("batch")},
            },
            "guard": {
                "running": {"sha": "0123456789abcdef", "job": self.job("sanity")},
                "bisect": {"job": self.job("bisect")},
            },
            "draft_pool": {"11": {"job": self.job("pool")}},
            "shadow_pool": {"12": {"job": self.job("shadow")}},
            "roles": {"drafter": {"job": self.job("role")}},
            "gpu": {"jobs": [{"id": "x1", "stage_job": self.job("stage")}]},
            "jobs": {"heavy": [self.job("h1"), self.job("h2")], "drift": self.job("d")},
        }
        named = [
            "review #7",
            "gate on #9",
            "batch gate",
            "develop sanity on 0123456789",
            "bisect",
            "pool draft #11",
            "shadow draft #12",
            "role job drafter",
            "GPU staging x1",
            "build job h1",
            "build job h2",
            "drift job d",
        ]
        self.assertEqual(drain.in_flight(st, self.store), named)
        # A job done is no longer in flight; one whose record this code cannot read counts as running.
        (self.store.dir / "jobs" / "h2" / "rc").write_text("0")
        st["roles"]["drafter"]["job"] = {"id": "unread"}
        self.assertEqual(
            drain.in_flight(st, self.store), [n for n in named if n != "build job h2"]
        )


class TestCommands(Base):
    """`rk drain` and `rk undrain` as rk runs them."""

    def setUp(self):
        super().setUp()
        self.home = home = sandbox.own_home(self)
        (home / "work").mkdir()
        conf = {
            "repo": "acme/widgets",
            "rocke_path": "a/b/rocke",
            "operator": {"github_login": "<user_login>"},
            "scope": {"labels": ["team: widgets"]},
            "paths": {"working_dir": str(home / "work")},
            "update": {"poll_s": 2},  # one look every two seconds
        }
        (home / "config.json").write_text(json.dumps(conf), encoding="utf-8")

    def rk(self, *argv) -> tuple[int, str]:
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = rkmon.main(list(argv))
        return code, out.getvalue()

    def test_rk_drain_takes_its_deadline_and_rk_undrain_lifts_the_flag(self):
        with mock.patch.object(drain, "main", return_value=0) as main:
            self.assertEqual(self.rk("drain", "--timeout-minutes", "7")[0], 0)
            self.assertEqual(self.rk("drain")[0], 0)
        self.assertEqual(
            [c.kwargs["timeout_min"] for c in main.call_args_list], [7.0, None]
        )
        state = store.Store()
        with state.locked(wait=True):
            st = state.load()
            st["draining"] = {"why": "rk drain", "since": store.now(), "until_min": 1}
            state.save(st)
        code, out = self.rk("undrain")
        self.assertEqual(code, 0)
        self.assertTrue(out.startswith("UNDRAINED: the drain since "), out)
        self.assertNotIn("draining", state.load())

    def test_rk_stop_leaves_rk_drain_to_see_the_stop_and_end_drained(self):
        # No PR runner (prcode.stop_all), whatever this machine has, and a configuration folder of the test's own.
        env = {k: v for k, v in os.environ.items() if k != slots.RUNNER_VAR}
        env["XDG_CONFIG_HOME"] = str(self.home / "xdg")
        isolated = mock.patch.dict(os.environ, env, clear=True)
        isolated.start()
        self.addCleanup(isolated.stop)
        p = subprocess.Popen(
            [sys.executable, "-u", "-c", RK_DRAIN, str(ROOT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )

        def end():
            if p.poll() is None:
                p.kill()
            p.wait()
            p.stdout.close()

        self.addCleanup(end)
        # Its first look: the flag holds, the work in flight is named, and the next look is two seconds away.
        fd, said = p.stdout.fileno(), b""
        while b"drain: nothing new starts" not in said:
            more = os.read(fd, 4096) if select.select([fd], [], [], 60)[0] else b""
            if not more:
                self.fail(f"rk drain did not start its wait: {said!r}")
            said += more
        state = store.Store()
        out = []
        self.assertEqual(control.stop(config.load(), state, out=out.append), 0)
        said += p.communicate(timeout=60)[0]
        at = store.stopped(state.load())["at"]
        self.assertEqual(
            (p.returncode, said.decode().splitlines()[-1]),
            (
                0,
                f"DRAINED: nothing is left to drain: the monitor is stopped (rk stop, {at}); rk start resumes it",
            ),
        )
        self.assertTrue(out[-1].startswith("STOPPED: 0 process(es)"), out)
        self.assertIsNone(state.load().get("draining"))

    def test_the_help_gives_the_drains_margin_in_minutes(self):
        self.assertIn(
            f"it lapses {store.DRAIN_MARGIN_MIN} minutes after the drain",
            " ".join(rkmon.__doc__.split()),
        )


if __name__ == "__main__":
    unittest.main()
