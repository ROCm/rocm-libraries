# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rkmon adopt end to end: docker/rkmon run on a copy of the package against the stand-in docker, as
tests/deploy_path_fixture.py sets it up. adopt refuses while the container runs, shows the plan, and runs before the
first start, which then sets the work volume up; with the container stopped, it brings the earlier monitor's config.json
and its standby.
"""

import json
import pathlib
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import deploy_folder  # noqa: E402
import sandbox  # noqa: E402
from deploy_path_fixture import ELSEWHERE, GOOD, DeployPathCase  # noqa: E402
from rendered import finished, one_line, rewrapped  # noqa: E402
from rkmon_lib import container_layout, records  # noqa: E402


class TestAdopt(DeployPathCase):
    def test_adopt_runs_before_the_first_start_and_refuses_while_running(self):
        """rkmon adopt runs before the first start: with no image, state 6's refusal (exit 3); with it,
        --dry-run shows the plan, applies nothing, and ends with PASSED and the command without it. With no terminal
        and no --yes, a usage error, nothing applied; a no on a terminal, CANCELLED; --yes applies the plan: ADOPTED,
        then the deployment state's next line, the first start, which sets the work volume up (rk provision) after the
        move, then runs rk init and the container (tests/test_provision.py runs the two on a work volume). While the
        container runs, the deployment state's refusal with the need's own fix, the drain and the stop (exit 4).
        """
        old = self.d / "old"
        (old / "monitor" / "state").mkdir(parents=True)
        (old / "monitor" / "state" / "state.json").write_text(
            json.dumps({"prs": {"101": {"status": "open"}}})
        )
        self.confirmed()
        here = [
            [r"image inspect", GOOD, 0],
            *deploy_folder.secrets_volumes(True),
            [r"rk logins --json", "[]", 0],
        ]
        self.answer([[r"image inspect", "", 1]])
        rc, out, err = self.rkmon("adopt", "--from", str(old))
        self.assertEqual(rc, 3, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "REFUSED: the image rocke-monitor:rkmon-test is not on this host; nothing changed",
                    f"next: step 4, image: {self.w} image load IMAGE_FILE",
                )
            ),
            out,
        )
        self.assertFalse([c for c in self.calls() if "rk adopt" in c])
        self.answer(here)
        rc, out, err = self.rkmon("adopt", "--from", str(old), "--dry-run")
        self.assertEqual(rc, 0, out + err)
        self.assertIn("What moves (files):", out)
        self.assertTrue(
            one_line(out).endswith(
                f"PASSED: the plan can be applied next: {self.w} adopt --from {old}"
            ),
            out,
        )
        plan = [c for c in self.calls() if "rk adopt plan --stage /adopt" in c]
        self.assertEqual(len(plan), 1)
        self.assertIn(":/adopt:ro", plan[0])
        # the plan only reads: its one-off container's entry point leaves the volumes' files as they are
        leave = container_layout.LEAVE_VOLUMES_VARIABLE
        self.assertIn(f" -e {leave}=1 ", plan[0])
        rc, out, err = self.rkmon("adopt", "--from", str(old))
        self.assertEqual(rc, 2, out + err)
        self.assertIn(
            f"{self.w} adopt: no terminal to ask on: {self.w} adopt --from {old} --yes applies the plan shown above",
            one_line(err),
        )
        rc, out = sandbox.on_a_terminal(
            [self.w, "adopt", "--from", str(old)], self.home, self.env, typed=b"n\n"
        )
        self.assertEqual(rc, 1, out)
        self.assertIn(
            rewrapped(
                f"CANCELLED: nothing changed\nnext: step 6, start: {self.w} start\n"
            ),
            out.replace("\r\n", "\n"),
        )
        self.assertFalse([c for c in self.calls() if "rk adopt apply" in c])
        rc, out, err = self.rkmon("adopt", "--from", str(old), "--yes")
        self.assertEqual(rc, 0, out + err)
        (apply,) = [c for c in self.calls() if "rk adopt apply --stage /adopt" in c]
        self.assertNotIn(
            leave, apply
        )  # the apply writes: its one-off has the whole root phase
        self.assertTrue(
            out.endswith(
                finished(
                    "ADOPTED: the earlier monitor's state, its logins and secrets left behind; the deployment starts "
                    f"in standby: it sends nothing until you run {self.w} standby off",
                    f"next: step 6, start: {self.w} start",
                )
            ),
            out,
        )
        # The first start: delivered, then the work volume set up, then rk init and the container.
        stopped = self.d / "stopped"  # no container until the start's compose up
        stopped.touch()
        self.answer(
            [
                [r"image inspect", GOOD, 0],
                [r"--entrypoint python3", "", 0],
                [r"rk stopped$", "", 1],
            ]
        )
        rc, out, err = self.rkmon(
            "start", FAKE_DOCKER_RUNNING="1", FAKE_DOCKER_STOPPED=str(stopped)
        )
        self.assertEqual(rc, 0, out + err)
        self.assertIn("\nSTARTED: the monitor runs your confirmed settings\n", out)
        calls = [c.split("\t", 1)[-1] for c in self.calls()]
        order = [
            next(
                i for i, c in enumerate(calls) if "rk adopt apply --stage /adopt" in c
            ),
            next(i for i, c in enumerate(calls) if c.endswith(" rk provision")),
            next(i for i, c in enumerate(calls) if c.endswith("rk init --if-new --on")),
            calls.index("compose up -d"),
        ]
        self.assertEqual(order, sorted(order), calls)
        # While the container runs: refused, nothing moved.
        self.answer(
            self.delivered()
            + here
            + [
                [r"rk stopped", "", 1],
                [r"compose ps -q", "0123456789ab", 0],
                [
                    r"inspect --format \{\{\.State\.StartedAt\}\}",
                    "2026-11-15T09:00:00Z",
                    0,
                ],
            ]
        )
        self.log.unlink()
        rc, out, err = self.rkmon("adopt", "--from", str(old), FAKE_DOCKER_RUNNING="1")
        self.assertEqual(rc, 4, out + err)
        self.assertTrue(
            one_line(out).endswith(
                "REFUSED: the monitor's container is running, and its ticks would race the move; nothing changed "
                f"fix: {self.w} stop --drain "
                f"next: step 7, doctor: {self.w} doctor"
            ),
            out,
        )
        self.assertFalse([c for c in self.calls() if "rk adopt" in c])

    def test_adopt_with_the_container_stopped_brings_config_json_and_standby(self):
        """rkmon adopt with the container stopped (state 12), from DIR alone: --config is no option
        of it (exit 2, nothing run); it stages the earlier monitor's config.json with its state, its one-offs run as
        the host's command names itself (RKMON_WRAPPER), and ADOPTED says the deployment starts in standby, then state
        12's line. A copy whose internal state cannot be read is refused, nothing run, with that state's fix, then
        adopt again, and the state's line; a plan or an apply that rk refuses (exit 4), nothing changed, says why and
        the fix, then the command again, and leaves no record. An apply that stops part-way said what it did, why, and
        the fix (rk adopt's ERROR, exit 1): the command again, alone; its record leaves state 3, so that rkmon next
        names adopt again and start refuses, until adopt, run again, finishes it. The start after it, once the monitor
        says it runs in standby (rk applied --json), says so. Of the earlier monitor, adopt reads DIR alone, and
        neither reads another monitor on this host.
        """
        w = self.w
        old = self.d / "old"
        (old / "monitor" / "state").mkdir(parents=True)
        (old / "monitor" / "state" / "state.json").write_text(
            json.dumps({"prs": {"101": {"status": "open"}}})
        )
        (old / "monitor" / "config.json").write_text(
            json.dumps({"repo": "acme/widgets"})
        )
        self.confirmed()
        layer, applied = self.delivered()
        standby = [
            applied[0],
            json.dumps(dict(json.loads(applied[1]), standby=True)),
            0,
        ]
        rows = [
            layer,
            standby,
            [r"image inspect", GOOD, 0],
            *deploy_folder.secrets_volumes(True),
            [r"compose ps -a -q rkmon$", "0123456789ab", 0],
            [r"rk logins --json", "[]", 0],
            [r"rk stopped$", "", 1],
        ]
        self.answer([*rows, *ELSEWHERE])
        env = self.stopped_after_a_start()
        given = str(old / "monitor" / "config.json")
        rc, out, err = self.rkmon(
            "adopt", "--from", str(old), "--config", given, "--yes", **env
        )
        self.assertEqual(rc, 2, out + err)
        self.assertFalse([c for c in self.calls() if "rk adopt" in c])
        start = f"next: step 6, start: {w} start"
        # a copy whose internal state cannot be read: refused, with that state's fix, then adopt again
        bad = self.d / "bad"
        (bad / "monitor" / "state").mkdir(parents=True)
        (bad / "monitor" / "state" / "state.json").write_text(json.dumps({"prs": {}}))
        internal = bad / "monitor" / "state" / "internal_state.json"
        internal.write_text("{ not JSON")
        self.answer([*rows, *ELSEWHERE])
        rc, out, err = self.rkmon("adopt", "--from", str(bad), "--yes", **env)
        self.assertEqual(rc, 4, out + err)
        self.assertIn(
            f"REFUSED: the internal state {internal} is no JSON: ", one_line(out)
        )
        self.assertTrue(
            one_line(out).endswith(f", then run {w} adopt --from DIR again {start}"),
            out,
        )
        self.assertFalse([c for c in self.calls() if "rk adopt" in c])
        # rk refuses the plan, then the apply, nothing changed: what it said, then the command again; no record
        again = finished(f"next: {w} adopt --from {old} --yes")
        for step in ("plan", "apply"):
            with self.subTest(step=step):
                refusal = (
                    "REFUSED: a made-up refusal; nothing changed\nfix: a made-up fix"
                )
                self.answer([[rf" rk adopt {step} ", refusal, 4], *rows, *ELSEWHERE])
                rc, out, err = self.rkmon("adopt", "--from", str(old), "--yes", **env)
                self.assertEqual(rc, 4, out + err)
                self.assertTrue(out.endswith(f"{refusal}\n{again}"), out)
                self.assertIsNone(records.read(self.pkg).under_way)
        stopped = "ERROR: a made-up reason; the earlier monitor's state is in place, in standby\nfix: a made-up fix"
        self.answer([[r" rk adopt apply ", stopped, 1], *rows, *ELSEWHERE])
        rc, out, err = self.rkmon("adopt", "--from", str(old), "--yes", **env)
        self.assertEqual(rc, 1, out + err)
        self.assertTrue(out.endswith(f"{stopped}\n{again}"), out)
        # stopped part-way (state 3): rkmon next names adopt again, from the same copy, and start refuses
        self.assertEqual(records.read(self.pkg).under_way.command, "adopt")
        finish = (
            f"next: {w} adopt --from DIR (from the same copy, it finishes putting the earlier monitor's state in "
            "place)"
        )
        self.answer([*rows, *ELSEWHERE])
        rc, out, err = self.rkmon("next", **env)
        self.assertEqual((rc, err), (0, ""), out)
        self.assertTrue(one_line(out).endswith(f" {finish}"), out)
        rc, out, err = self.rkmon("start", **env)
        self.assertEqual(rc, 4, out + err)
        self.assertTrue(
            one_line(out).endswith(
                f"REFUSED: {w} adopt stopped part-way; nothing changed {finish}"
            ),
            out,
        )
        self.answer([*rows, *ELSEWHERE])
        self.log.unlink()
        rc, out, err = self.rkmon("adopt", "--from", str(old), "--yes", **env)
        self.assertEqual(rc, 0, out + err)
        self.assertIn(f"config.json: from {given} ", one_line(out))
        adopts = [c for c in self.calls() if " rk adopt " in c]
        self.assertEqual(len(adopts), 2, adopts)
        for call in adopts:
            self.assertIn(f" -e RKMON_WRAPPER={w} ", call)
        self.assert_reads_no_other_monitor()
        self.assertTrue(
            out.endswith(
                finished(
                    "ADOPTED: the earlier monitor's state, its logins and secrets left behind; the deployment starts "
                    f"in standby: it sends nothing until you run {w} standby off",
                    start,
                )
            ),
            out,
        )
        self.assertIsNone(records.read(self.pkg).under_way)
        self.log.unlink()
        rc, out, err = self.rkmon("start", **env)
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(
            out.endswith(
                finished(
                    "STARTED: the monitor runs your confirmed settings, in standby: it sends nothing until you run "
                    f"{w} standby off",
                    f"next: step 7, doctor: {w} doctor",
                )
            ),
            out,
        )
        self.assert_reads_no_other_monitor()


if __name__ == "__main__":
    unittest.main()
