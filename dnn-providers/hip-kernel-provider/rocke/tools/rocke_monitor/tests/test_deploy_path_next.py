# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""What each command ends with, and rkmon next, end to end: docker/rkmon run on a copy of the package against the
stand-in docker, as tests/deploy_path_fixture.py sets it up. Each command ends with one outcome line and one next or
done line, and exits with the outcome's code, but those NO_OUTCOME names. rkmon next shows each step DONE, TODO or REDO,
and the one command to run next, which names its step until a tick has run after a start; what it cannot read it says is
unknown. A line never breaks inside a command it gives.
"""

import json
import pathlib
import re
import shlex
import shutil
import sys
import unittest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

import deploy_folder  # noqa: E402
from deploy_path_fixture import (  # noqa: E402
    FRESH_ENV,
    GOOD,
    MAKES,
    NEXT_LINE,
    OUTCOME_LINE,
    DeployPathCase,
)
from rendered import finished, one_line  # noqa: E402
from rkmon_lib import checks, commands, outcome, records  # noqa: E402

# The commands that end with no outcome line, and why; every other command ends with one.
NO_OUTCOME = {
    "help": "it prints the help",
    "template section": "it prints the section",
    "next": "it prints the next line, and exits 0",
    "status": "it prints the deployment's state, and exits 0",
    "image ls": "it lists this package's images",
    "logs": "docker compose logs' code is its own",
    "shell": "the shell's code is its own",
    "rk": "rk's code is its own",
}
# The commands whose first example asks you before it does anything: with no terminal to ask on, they end with a
# usage error (exit 2) that names the command that answers yes.
ASKS = {"confirm"}


class TestNext(DeployPathCase):
    def test_each_command_ends_with_one_outcome(self):
        """Each command, its first example run in a fresh folder (~/ your home folder, as the shell reads
        it; the image file image load reads made), prints one outcome line, then one next or done line, which ends
        what it prints, and exits with the outcome word's code: 3 for FAILED or REFUSED when an earlier step is not
        done. NO_OUTCOME names the commands that end with no outcome line, and why; ASKS, those that ask you first,
        which with no terminal end with the usage error that names --yes."""
        names = {row.name for row in commands.TABLE}
        self.assertEqual(set(NO_OUTCOME) - names, set(), "NO_OUTCOME names no command")
        self.assertEqual(ASKS - names, set(), "ASKS names no command")
        for row in commands.TABLE:
            if row.name in NO_OUTCOME:
                continue
            with self.subTest(command=row.name):
                self.setUp()
                (self.home / "monitor.tgz").write_text("x")
                argv = [
                    f"{self.home}/{word[2:]}" if word.startswith("~/") else word
                    for word in shlex.split(row.examples[0][0])
                ]
                rc, out, err = self.rkmon(*argv, **FRESH_ENV.get(row.name, {}))
                said = f"exit {rc}\n{out}{err}"
                found = list(OUTCOME_LINE.finditer(out))
                nexts = list(NEXT_LINE.finditer(out))
                if row.name in ASKS:
                    self.assertEqual((rc, found, nexts), (2, [], []), said)
                    self.assertIn(
                        f": no terminal to ask on: {self.w} {row.name} --yes ", err
                    )
                    continue
                self.assertEqual(len(found), 1, said)
                self.assertEqual(len(nexts), 1, said)
                self.assertLess(found[0].start(), nexts[0].start(), said)
                last = " ".join(out[nexts[0].start() :].split())
                self.assertTrue(out.endswith(finished(last)), said)
                word = found[0].group(1) or found[0].group(2)
                codes = {outcome.code(word)}
                if word in ("FAILED", "REFUSED"):
                    codes.add(outcome.code(word, earlier_step=True))
                self.assertIn(rc, codes, said)

    def test_a_usage_error_from_rk_provision_check_says_nothing(self):
        """Whether the running container's work volume is set up is what `rk provision --check` in it says: an older
        package's rk, which lacks that command (a usage error that calls it an invalid choice: older_package), reads as
        set up by its own start; any other usage error says nothing, so rkmon next says what it leaves unknown, ERROR,
        exit 1."""
        self.confirmed()
        usage = "usage: rk provision [-h]\nrk: error: unrecognized arguments: --check"
        self.answer(
            [
                [r"rk provision --check$", usage, 2],
                [r"rk logins --json", "[]", 0],
                [r"rk stopped$", "", 1],
                [r"image inspect", GOOD, 0],
            ]
        )
        rc, out, err = self.rkmon("next", **self.running())
        self.assertEqual(rc, 1, out + err)
        self.assertIn(
            "ERROR: the running container did not say whether its work volume is set up (exit code 2: rk: error: "
            "unrecognized arguments: --check)",
            one_line(out),
        )

    def test_the_confirm_s_next_line_is_rkmon_next_s(self):
        """Step 3's next line is the state's, as rkmon next gives it, in each state a
        confirm can leave: with the image not here, step 4's command for these settings, image pull when they name a
        registry, else image load IMAGE_FILE (state 6); with the image here, the next step left: the logins while one
        is missing (state 7), else the start (state 8); after the first start, apply for the change it confirmed
        (state 11), and so on NOTHING TO DO when the file is what it confirmed. No next line names image check, which
        changes nothing."""

        def next_line(out: str) -> str:
            out = "\n" + out
            return one_line(out[out.rindex("\nnext:") :])

        here = [*self.image_code(), [r"image inspect", GOOD, 0]]
        absent = [[r"image inspect", "", 1]]
        # Before the first start, each command a deploy step's, with its step.
        for case, registry, answers, want in (
            (
                "no image, from a file",
                "",
                absent,
                "step 4, image: {} image load IMAGE_FILE",
            ),
            (
                "no image, from a registry",
                "registry.example.com/team",
                absent,
                "step 4, image: {} image pull",
            ),
            (
                "the image, a login missing",
                "",
                [*here, *deploy_folder.secrets_volumes(False)],
                "step 5, login: {} login",
            ),
            (
                "the image and the logins",
                "",
                [*here, *deploy_folder.secrets_volumes(True)],
                "step 6, start: {} start",
            ),
        ):
            with self.subTest(case=case):
                shutil.rmtree(
                    self.pkg / records.FOLDER, ignore_errors=True
                )  # no confirm recorded
                self.write_settings(registry=registry)
                self.answer(answers + [[r"rk logins --json", "[]", 0]])
                rc, out, err = self.rkmon(
                    "confirm", "--settings", str(self.settings), "--yes"
                )
                self.assertEqual(rc, 0, out + err)
                self.assertTrue(
                    out.startswith(
                        "== deploy step 3 of 8, confirm: check your deploy settings, show them, and confirm "
                        "them\n"
                    ),
                    out,
                )
                self.assertEqual(next_line(out), "next: " + want.format(self.w))
                self.assertNotIn("image check", out)
                rc, said, err = self.rkmon("next")
                self.assertEqual((rc, err), (0, ""), said)
                self.assertEqual(next_line(said), next_line(out))
        # After the first start, with the container running: a change confirmed waits for apply.
        shutil.rmtree(self.pkg / records.FOLDER, ignore_errors=True)
        self.write_settings()
        self.confirmed()
        running = self.delivered()
        self.write_settings(memory_gib=13)
        self.answer(
            [
                *running,
                [r"image inspect", GOOD, 0],
                [r"compose ps -a -q rkmon$", "0123456789ab", 0],
                [r"rk stopped$", "", 1],
                [r"rk logins --json", "[]", 0],
            ]
        )
        for word in ("CONFIRMED", "NOTHING TO DO"):
            with self.subTest(case=word):
                rc, out, err = self.rkmon("confirm", "--yes", **self.running())
                self.assertEqual(rc, 0, out + err)
                self.assertIn(f"\n{word}", out)
                self.assertEqual(next_line(out), f"next: {self.w} apply")
                rc, said, err = self.rkmon("next", **self.running())
                self.assertEqual((rc, err), (0, ""), said)
                self.assertEqual(next_line(said), next_line(out))

    def test_once_a_tick_has_run_since_the_start_each_line_is_the_command_alone(self):
        """Until deploy step 8 has passed after a start (a tick has run since then), a next line that names a deploy
        step's command carries the step's number and name; after it, the command alone, and for good, as this
        folder's records keep it: each case here starts without it, as after its volumes are removed. On a
        running deployment with a login missing (state 7), an rk stop left in it (state 12), or its settings file
        changed after the confirm (state 5), rkmon next and the command that ends with the state's line there (the
        confirm's NOTHING TO DO, the start's NOTHING TO DO or its refusal) give the same line: the command alone once a
        tick has run since the start, with its step while none has."""
        self.confirmed()
        delivered = self.delivered()
        missing = {
            "name": "github_read",
            "what": "gh's read login",
            "state": "MISSING",
            "found": "no login",
            "needed_by": [],
            "command": "login github_read",
        }
        container = [
            [r"image inspect", GOOD, 0],
            [r"compose ps -a -q rkmon$", "0123456789ab", 0],
            [r"compose ps -q rkmon$", "0123456789ab", 0],
            [r"inspect --format \{\{\.State\.StartedAt\}\}", "2026-11-15T09:00:00Z", 0],
        ]
        ticks = (
            ([r"rk tick-due", "last tick never", 1], "{words} {command}"),
            (
                [r"rk tick-due", "last tick 2026-11-15T09:05:00Z, 3 min ago", 0],
                "{command}",
            ),
        )
        runs = [r"rk stopped$", "", 1]
        for n, answers, words, command, ends in (
            (
                7,
                [[r"rk logins --json", json.dumps([missing]), 0], runs],
                "step 5, login:",
                "login",
                [(("confirm", "--yes"), 0), (("start",), 0)],
            ),
            (
                12,
                [[r"rk logins --json", "[]", 0], [r"rk stopped$", "", 0]],
                "step 6, start:",
                "start",
                [(("confirm", "--yes"), 0)],
            ),
            (
                5,
                [[r"rk logins --json", "[]", 0], runs],
                "step 3, confirm:",
                "confirm",
                [(("start",), 3)],
            ),
        ):
            if n == 5:
                self.write_settings(memory_gib=13)  # changed after the confirm
            for tick, form in ticks:
                want = "next: " + form.format(
                    words=words, command=f"{self.w} {command}"
                )
                with self.subTest(state=n, line=want):
                    records.clear_first_tick(self.pkg)
                    self.answer([*delivered, *container, tick, *answers])
                    rc, said, err = self.rkmon("next", **self.running())
                    self.assertEqual((rc, err), (0, ""), said)
                    self.assertTrue(said.endswith("\n" + finished(want)), said)
                    for typed, code in ends:
                        rc, out, err = self.rkmon(*typed, **self.running())
                        self.assertEqual(rc, code, out + err)
                        self.assertTrue(
                            one_line(out).endswith(one_line(f"\n{want}\n")), out
                        )
                    self.assertEqual(
                        records.read(self.pkg).first_tick is not None,
                        form == "{command}",
                    )

    def test_the_first_tick_a_command_sees_is_kept_until_the_volumes_are_removed(self):
        """The first command that sees a tick since a start keeps it in this folder's records, never under
        --dry-run, and a later tick changes nothing. From then on each next line names a deploy step's command without
        its step: after a later start, whose own line is one, and the line deploy step 1 ends with. rkmon remove
        --delete-data clears it, and each line names its step again."""
        w = self.w
        self.confirmed()
        delivered = self.delivered()
        stopped = self.d / "stopped"
        env = {**self.running(), "FAKE_DOCKER_STOPPED": str(stopped)}
        container = [
            [r"image inspect", GOOD, 0],
            [r"compose ps -a -q rkmon$", "0123456789ab", 0],
            [r"compose ps -q rkmon$", "0123456789ab", 0],
            [r"inspect --format \{\{\.State\.StartedAt\}\}", "2026-11-15T09:00:00Z", 0],
            [r"rk logins --json", "[]", 0],
            [r"rk stopped$", "", 1],
        ]

        def tick(when):
            said = f"last tick {when}, 3 min ago" if when else "last tick never"
            return [*delivered, *container, [r"rk tick-due", said, 0 if when else 1]]

        def ends(argv, want):
            rc, out, err = self.rkmon(*argv, **env)
            self.assertEqual(rc, 0, out + err)
            self.assertTrue(one_line(out).endswith(one_line(want)), out)

        self.answer(tick(""))
        ends(["next"], f"next: step 7, doctor: {w} doctor")
        self.answer(tick("2026-11-15T09:05:00Z"))
        ends(["--dry-run", "login"], f"next: {w} doctor")
        self.assertIsNone(records.read(self.pkg).first_tick)
        rc, out, err = self.rkmon("status", **env)
        self.assertEqual(rc, 0, out + err)
        self.assertIn(one_line(f"\nnext: {w} doctor\n"), one_line(out))
        first = records.FirstTick("2026-11-15T09:05:00Z")
        self.assertEqual(records.read(self.pkg).first_tick, first)
        self.answer(tick("2026-11-15T09:35:00Z"))
        ends(["next"], f"next: {w} doctor")
        self.assertEqual(records.read(self.pkg).first_tick, first)
        # A later start: no tick since it, and while the monitor does not say yet which settings it runs, the start's
        # own line is step 7's.
        stopped.touch()
        self.answer([delivered[0], *container])
        ends(["start"], f"next: {w} doctor")
        self.answer(tick(""))
        ends(["next"], f"next: {w} doctor")
        rc, out, err = self.rkmon("template", "deploy", str(self.home / "other.yaml"))
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(
            one_line(out).endswith(
                one_line(f"\nnext: {w} check --settings ~/other.yaml\n")
            ),
            out,
        )
        # Deploy step 2's line too, where the check passes: the host's own disk and memory decide two of its checks.
        small = self.home / "small.yaml"
        deploy_folder.write_settings(small, self.port, memory_gib=8, ramdisk_gib=0)
        host = [[r"^version", "29.0.0", 0], [r"compose version", "2.40.0", 0]]
        self.answer([*host, [r"DockerRootDir", str(self.d), 0]])
        rc, out, err = self.rkmon("check", "--settings", str(small))
        if rc == 0:
            self.assertTrue(
                one_line(out).endswith(
                    one_line(f"\nnext: {w} confirm --settings ~/small.yaml\n")
                ),
                out,
            )
        # The data removed: a first start again, and the logins, gone with their volumes, are the next step. Once the
        # removal is done (FAKE_DOCKER_DOWN), no container has its name, and the volumes are gone.
        removed = self.d / "removed.json"
        gone = [
            [r"image inspect", GOOD, 0],
            [r"^ps --all --filter ", "", 0],
            *deploy_folder.volumes(),
            *deploy_folder.secrets_volumes(False),
        ]
        removed.write_text(json.dumps(gone), encoding="utf-8")
        env["FAKE_DOCKER_DOWN"] = str(removed)
        self.answer(
            [
                *tick(""),
                *deploy_folder.volumes(deploy_folder.VOLUMES),
                *deploy_folder.secrets_volumes(True),
            ]
        )
        ends(["remove", "--delete-data", "--yes"], f"next: step 5, login: {w} login")
        self.assertIsNone(records.read(self.pkg).first_tick)

    def test_one_next_line_before_the_first_start(self):
        """Before the first start the logins and secrets come from the volumes that keep them. With none on this
        host yet, every one the settings need is missing: state 7. Once `rkmon login` made them, a one-off container of
        the image reads them, with no network and only those volumes, mounted read-only: state 7 while one holds
        nothing, else state 8. LOADED, next, status and the refusal of a command that needs the container end with the
        same next line, and reading makes no Docker object: no compose project's container, network or volume, and no
        one-off container but that reader; nor does the load, which sets nothing up.
        """
        self.confirmed()
        ref = self.confirm_record().image.reference
        image_file = self.home / "image.tar.gz"
        image_file.write_text("x")
        reader = re.compile(
            r"\trun --rm --network none .*--entrypoint python3 "
            + re.escape(ref)
            + " -I -c "
        )
        making = re.compile(
            r"\t(compose (run|up|create|start)|(volume|network|container) create|create|run) "
        )
        need = "github_read, claude and admin_access_token"
        # Before the first start each line carries its deploy step.
        login = f"next: step 5, login: {self.w} login"
        start = f"next: step 6, start: {self.w} start"
        unset = "your confirmed settings need {}, which are missing or wrong"
        for case, volumes, state, want in (
            (
                "no volume yet",
                deploy_folder.secrets_volumes(False),
                unset.format(need),
                login,
            ),
            (
                "a volume that has one of them",
                deploy_folder.secrets_volumes(True, ["claude", "admin_access_token"]),
                unset.format("claude and admin_access_token"),
                login,
            ),
            (
                "a volume that has them all",
                deploy_folder.secrets_volumes(True),
                "does not exist yet",
                start,
            ),
        ):
            with self.subTest(case=case):
                self.answer(
                    [
                        *volumes,
                        [r"^load -i", f"Loaded image: {ref}", 0],
                        [r"image inspect", GOOD, 0],
                        [r"--entrypoint python3", "TLS ok", 0],
                    ]
                )
                self.log.unlink(missing_ok=True)
                rc, out, err = self.rkmon("image", "load", str(image_file))
                self.assertEqual(rc, 0, out + err)
                self.assertTrue(
                    one_line(out).endswith(one_line(f"\nLOADED {ref}\n{want}\n")), out
                )
                loaded = self.calls()
                self.log.unlink(missing_ok=True)
                for args in (("next",), ("status",)):
                    rc, out, err = self.rkmon(*args)
                    self.assertEqual((rc, err), (0, ""), out)
                    self.assertIn(state, one_line(out))
                    self.assertTrue(one_line(out).endswith(one_line(want)), out)
                rc, out, err = self.rkmon("shell")
                self.assertEqual(rc, 3, out + err)
                self.assertTrue(one_line(out).endswith(one_line(want)), out)
                read = self.calls()
                self.assertEqual(
                    [c for c in read if making.search(c) and not reader.search(c)], []
                )
                # The load sets nothing up: no container, volume or network stays after it.
                self.assertEqual([c for c in loaded if MAKES.search(c)], [])
                for c in read + loaded:
                    self.assertNotIn(" rk logins", c)
                    # the load's reader of the image's own code (image_code) mounts nothing
                    if reader.search(c) and deploy_folder.IMAGE_CODE not in c:
                        mounts = re.findall(r"--mount (\S+)", c)
                        self.assertEqual(len(mounts), 2, c)
                        for m in mounts:
                            self.assertRegex(
                                m,
                                r"^type=volume,source=rkmon-test_\S+,target=/secrets(-write)?,readonly,volume-nocopy$",
                            )
                self.assertEqual(
                    any(reader.search(c) for c in read), volumes[1][1] != "", read
                )

    def test_a_long_path_to_rkmon_moves_each_command_whole_to_a_line_of_its_own(self):
        """A line that gives a command never breaks inside it (rkmon_lib/wrapping.py): where the line would pass a
        row's width, the command moves whole to a line of its own, so a line copied runs all of it. rkmon runs here by
        a path relative to the home folder, through a link there to the package's folder, so what it prints is the
        same wherever the temporary folder is: a path of 66 characters puts the second command of the check's fix at
        the end of its line, one of 90 puts step 3's next line past the width, and one of 78 the check's, each command
        short enough for a line of its own."""

        def run_by(length: int) -> str:
            """rkmon run by a path of LENGTH characters."""
            folder = "f" * (length - len("/docker/rkmon"))
            (self.home / folder).symlink_to(self.pkg)
            self.w = f"{folder}/docker/rkmon"
            return self.w

        w = run_by(66)
        self.answer([[r"^version", "29.0.0", 0], [r"compose version", "2.40.0", 0]])
        rc, out, err = self.rkmon("check")
        self.assertEqual(rc, 3, out + err)
        self.assertIn(
            "  FAIL  your settings file: none named\n"
            f"        fix: {w} check --settings FILE, FILE being\n"
            "             your deploy settings file\n"
            f"             ({w} template deploy FILE writes one to\n"
            "             start from)\n",
            out,
        )

        def wraps(line: str, w: str) -> None:
            """The line passes the width, and its command fits on a line."""
            self.assertGreater(len(line), checks.WIDTH)
            self.assertLessEqual(len(line) - line.index(w), checks.WIDTH)

        w = run_by(90)
        wraps(f"next: step 4, image: {w} image load IMAGE_FILE", w)
        self.answer([[r"image inspect", "", 1], [r"rk logins --json", "[]", 0]])
        rc, out, err = self.rkmon(
            "confirm", "--settings", "~/rkmon-deploy.yaml", "--yes"
        )
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(
            out.endswith(f"\nnext: step 4, image:\n{w} image load IMAGE_FILE\n"), out
        )
        w = run_by(78)
        wraps(f"next: {w} check --settings ~/rkmon-deploy.yaml", w)
        self.settings.chmod(0o644)
        self.answer([[r"^version", "29.0.0", 0], [r"compose version", "2.40.0", 0]])
        rc, out, err = self.rkmon("check", "--settings", "~/rkmon-deploy.yaml")
        self.assertEqual(rc, 4, out + err)
        self.assertTrue(
            out.endswith(f"\nnext:\n{w} check --settings ~/rkmon-deploy.yaml\n"), out
        )

    def test_next_says_where_this_deployment_is(self):
        """rkmon next: each step of the deploy path DONE, TODO or REDO, from this folder's records and its own
        container, then one next command: the state's, or, while the monitor runs, step 7 until the doctor has passed
        since its start, then step 8 until a tick has run since then; each with its step's number and name until a
        tick has run after a start, which this folder's records then keep; exit 0, but 2 for the command
        line. With --settings FILE, steps 1 and 2 are
        that file's; a build settings file shows the build path's five steps. rkmon doctor starts with the same rows,
        without the next line, then the state's row; when it passes, the state's next line follows, as next gives it,
        and next reads its result."""
        w = self.w
        rc, out, err = self.rkmon("next")
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(
            out.startswith(
                "The deploy path (QUICK_DEPLOY_GUIDE.md):\n"
                "  1. TODO  template: no deploy settings file named yet\n"
                "  2. TODO  check: after step 1\n"
            ),
            out,
        )
        self.assertTrue(
            out.endswith(
                "\n"
                + finished(
                    f"next: step 1, template: {w} template deploy ~/rkmon-deploy.yaml",
                    f"(have a file already? {w} next --settings FILE)",
                )
            ),
            out,
        )
        self.settings.chmod(0o644)
        rc, out, err = self.rkmon("next", "--settings", str(self.settings))
        self.assertEqual(rc, 0, out + err)
        self.assertIn("  1. DONE  template: ~/rkmon-deploy.yaml\n", out)
        self.assertRegex(out, r"(?m)^  2\. TODO  check: \d+ of \d+ checks fail$")
        self.assertTrue(
            out.endswith(
                "\n"
                + finished(
                    f"next: step 2, check: {w} check --settings ~/rkmon-deploy.yaml"
                )
            ),
            out,
        )
        self.settings.chmod(0o600)
        rc, out, err = self.rkmon("next", "extra")
        self.assertEqual(
            (rc, out, err), (2, "", f"{w} next: unrecognized arguments: extra\n")
        )
        # Confirmed, with no image here: the state's command.
        self.confirmed()
        self.answer([[r"image inspect", "", 1]])
        rc, out, err = self.rkmon("next")
        self.assertRegex(
            out,
            r"(?m)^  3\. DONE  confirm: ~/rkmon-deploy\.yaml and this package are what you confirmed at "
            r"\d{4}-\d\d-\d\d \d\d:\d\d UTC$",
        )
        self.assertIn(
            "  4. TODO  image: the image rocke-monitor:rkmon-test is not on this host\n",
            out,
        )
        self.assertTrue(
            out.endswith(
                "\n" + finished(f"next: step 4, image: {w} image load IMAGE_FILE")
            ),
            out,
        )
        # Running, the settings delivered and loaded: step 7 until the doctor has passed since the start.
        running = self.delivered() + [
            [r"image inspect", GOOD, 0],
            [r"rk logins --json", json.dumps(self.login_rows()), 0],
            [r"rk stopped", "", 1],
            [r"compose ps -q", "0123456789ab", 0],
            [r"inspect --format \{\{\.State\.StartedAt\}\}", "2026-11-15T09:00:00Z", 0],
        ]
        self.answer(running + [[r"rk tick-due", "last tick never", 1]])
        rc, out, err = self.rkmon("next", FAKE_DOCKER_RUNNING="1")
        self.assertIn("  6. DONE  start: running since 2026-11-15 09:00 UTC\n", out)
        self.assertIn("  8. TODO  first tick: none since the start\n", out)
        self.assertTrue(
            out.endswith("\n" + finished(f"next: step 7, doctor: {w} doctor")), out
        )
        tick = [r"rk tick-due", "last tick 2026-11-15T09:05:00Z, 3 min ago", 0]
        self.answer(running + [tick])
        rc, out, err = self.rkmon("next", FAKE_DOCKER_RUNNING="1")
        self.assertIn("  7. TODO  doctor: not run since the start\n", out)
        self.assertIn("  8. DONE  first tick: at 2026-11-15 09:05 UTC\n", out)
        # Step 8 has passed since the start: the doctor's line is the command alone.
        self.assertTrue(out.endswith("\n" + finished(f"next: {w} doctor")), out)
        report = checks.render(
            [
                checks.Check("docker", "PASS", "29.0.0"),
                checks.Check("GitHub read login", "PASS", "set"),
            ]
        )
        self.answer(running + [tick, [r"rk doctor", report, 0]])
        rc, out, err = self.rkmon("doctor", FAKE_DOCKER_RUNNING="1")
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(
            out.startswith(
                "== deploy step 7 of 8, doctor: check the running monitor and everything it reaches\n"
                "The deploy path (QUICK_DEPLOY_GUIDE.md):\n"
            ),
            out,
        )
        # PASSED, then the state's line as next gives it: a tick has run since the start, so nothing is left.
        self.assertTrue(
            out.endswith(
                "  8. DONE  first tick: at 2026-11-15 09:05 UTC\n"
                "  PASS  state: the monitor runs your confirmed settings\n"
                f"\n{report}\n\n"
                + finished(
                    "PASSED: 3 checks", "done: the monitor runs your confirmed settings"
                )
            ),
            out,
        )
        # The doctor passed since the start, and a tick has run: done.
        rc, out, err = self.rkmon("next", FAKE_DOCKER_RUNNING="1")
        self.assertRegex(
            out,
            r"(?m)^  7\. DONE  doctor: 3 checks passed at \d{4}-\d\d-\d\d \d\d:\d\d UTC$",
        )
        self.assertTrue(
            out.endswith("\ndone: the monitor runs your confirmed settings\n"), out
        )
        # Step 5's command, login, walks the logins: under --dry-run it offers none, and ends as the real run would,
        # with the state's line: a tick has run after a start, which this folder's records keep, so no step's words.
        rc, out, err = self.rkmon("--dry-run", "login")
        self.assertEqual(
            (rc, out.splitlines()[0]),
            (
                0,
                "== deploy step 5 of 8, login: set each login and secret your settings need (dry run: nothing "
                "changes)",
            ),
        )
        self.assertTrue(
            out.endswith(
                finished(
                    "PASSED: every login and secret your settings need is set",
                    f"next: {w} start",
                )
            ),
            out,
        )
        # A build settings file: the build path's five steps of that file.
        build = self.home / "rkmon-build.yaml"
        self.assertEqual(self.rkmon("template", "build", str(build))[0], 0)
        rc, out, err = self.rkmon("next", "--settings", str(build))
        self.assertEqual(rc, 0, out + err)
        self.assertTrue(
            out.startswith(
                "The build path (QUICK_BUILD_GUIDE.md):\n  1. DONE  template: ~/rkmon-build.yaml\n"
            ),
            out,
        )
        self.assertEqual(
            len(re.findall(r"(?m)^  \d\. (DONE|TODO|REDO)  ", out)), 5, out
        )
        self.assertTrue(
            out.endswith(
                "\n"
                + finished(
                    f"next: step 2, check: {w} check --settings ~/rkmon-build.yaml"
                )
            ),
            out,
        )


if __name__ == "__main__":
    unittest.main()
