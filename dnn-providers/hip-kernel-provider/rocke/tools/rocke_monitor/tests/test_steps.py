# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The two numbered paths (rkmon_lib/steps.py): the steps as `rkmon help` lists them, the
steps' banners and next lines, a command typed word by word and as a command line, and the done line.
"""

import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import checks, commands, program, steps  # noqa: E402
from rkmon_lib.steps import BUILD_STEPS, DEPLOY_STEPS  # noqa: E402

FILE = "~/rkmon-deploy.yaml"


class TestPaths(unittest.TestCase):
    def test_the_build_path(self):
        self.assertEqual(
            [(s.n, s.name, s.command, s.others) for s in BUILD_STEPS],
            [
                (1, "template", "template build FILE", ()),
                (2, "check", "check --settings FILE", ()),
                (3, "build", "build --settings FILE", ()),
                (4, "test", "image test --settings FILE", ()),
                (
                    5,
                    "ship",
                    "image save IMAGE_FILE --settings FILE",
                    ("image push --settings FILE",),
                ),
            ],
        )

    def test_the_deploy_path(self):
        self.assertEqual(
            [(s.n, s.name, s.command, s.others) for s in DEPLOY_STEPS],
            [
                (1, "template", "template deploy [--profile PROFILE] FILE", ()),
                (2, "check", "check --settings FILE", ()),
                (3, "confirm", "confirm --settings FILE", ()),
                (4, "image", "image load IMAGE_FILE", ("image pull",)),
                (5, "login", "login", ()),
                (6, "start", "start", ()),
                (7, "doctor", "doctor", ()),
                (8, "first tick", "first-tick", ()),
            ],
        )
        # Step 4 needs no command when the image is here, and `image check` checks its work: no next line
        # names it. No other step has either.
        self.assertEqual(
            [
                (s.n, s.skip, s.check)
                for s in (*BUILD_STEPS, *DEPLOY_STEPS)
                if s.skip or s.check
            ],
            [(4, "nothing when the image is here", "image check")],
        )
        self.assertEqual(DEPLOY_STEPS[3].path, "deploy")

    def test_each_step_knows_its_path(self):
        self.assertEqual(steps.PATHS, {"build": BUILD_STEPS, "deploy": DEPLOY_STEPS})
        for path, these in steps.PATHS.items():
            self.assertTrue(all(s.path == path and s.what for s in these))

    def test_as_rkmon_help_lists_them(self):
        # The two paths, as rkmon help lists them.
        paths = (
            commands.Path("build-path", "The build path, on any machine", BUILD_STEPS),
            commands.Path(
                "deploy-path",
                "The deploy path, on the machine that runs it",
                DEPLOY_STEPS,
                last="Where you are, and the next command: {program} next",
            ),
        )
        shown = [commands.section(p.key, (), paths) for p in paths]
        self.assertEqual(
            shown,
            [
                "The build path, on any machine (rkmon help build-path):\n"
                "  1 template    rkmon template build FILE\n"
                "  2 check       rkmon check --settings FILE\n"
                "  3 build       rkmon build --settings FILE\n"
                "  4 test        rkmon image test --settings FILE\n"
                "  5 ship        rkmon image save IMAGE_FILE --settings FILE\n"
                "                or: rkmon image push --settings FILE\n",
                "The deploy path, on the machine that runs it (rkmon help deploy-path):\n"
                "  1 template    rkmon template deploy [--profile PROFILE] FILE\n"
                "  2 check       rkmon check --settings FILE\n"
                "  3 confirm     rkmon confirm --settings FILE\n"
                "  4 image       rkmon image load IMAGE_FILE\n"
                "                or: rkmon image pull; nothing when the image is here\n"
                "                (rkmon image check checks it)\n"
                "  5 login       rkmon login\n"
                "  6 start       rkmon start\n"
                "  7 doctor      rkmon doctor\n"
                "  8 first tick  rkmon first-tick\n"
                "  Where you are, and the next command: rkmon next\n",
            ],
        )

    def test_the_lines_after_a_step_s_command(self):
        self.assertEqual(
            steps.aside(DEPLOY_STEPS[3]),
            (
                "or: rkmon image pull; nothing when the image is here",
                "(rkmon image check checks it)",
            ),
        )
        self.assertEqual(
            steps.aside(BUILD_STEPS[4]), ("or: rkmon image push --settings FILE", "")
        )
        self.assertEqual(steps.aside(DEPLOY_STEPS[4]), ("", ""))
        # each command as the text shows it: the README's table puts it in code
        self.assertEqual(
            steps.aside(DEPLOY_STEPS[3], lambda c: f"`./rkmon {c}`"),
            (
                "or: `./rkmon image pull`; nothing when the image is here",
                "(`./rkmon image check` checks it)",
            ),
        )
        with mock.patch.object(program, "name", return_value="./rkmon"):
            self.assertEqual(
                steps.aside(DEPLOY_STEPS[3])[1], "(./rkmon image check checks it)"
            )

    def test_after(self):
        self.assertIs(steps.after(DEPLOY_STEPS[2]), DEPLOY_STEPS[3])
        self.assertIs(steps.after(BUILD_STEPS[0]), BUILD_STEPS[1])
        self.assertIsNone(steps.after(BUILD_STEPS[-1]))
        self.assertIsNone(steps.after(DEPLOY_STEPS[-1]))


class TestBanners(unittest.TestCase):
    def test_the_samples(self):
        self.assertEqual(
            steps.banner(BUILD_STEPS[1]),
            "== build step 2 of 5, check: check this machine and your build settings; change nothing",
        )
        self.assertEqual(
            steps.banner(DEPLOY_STEPS[1]),
            "== deploy step 2 of 8, check: check this host and your deploy settings; change nothing",
        )
        self.assertEqual(
            steps.banner(DEPLOY_STEPS[2]),
            "== deploy step 3 of 8, confirm: check your deploy settings, show them, and confirm them",
        )
        self.assertEqual(
            steps.banner(DEPLOY_STEPS[4]),
            "== deploy step 5 of 8, login: set each login and secret your settings need",
        )

    def test_every_banner(self):
        self.assertEqual(
            [steps.banner(s) for s in BUILD_STEPS + DEPLOY_STEPS],
            [
                "== build step 1 of 5, template: write a build settings file for you to fill in",
                "== build step 2 of 5, check: check this machine and your build settings; change nothing",
                "== build step 3 of 5, build: build the image from your build settings",
                "== build step 4 of 5, test: test the image on this machine",
                "== build step 5 of 5, ship: save the image to a file, or push it to your registry",
                "== deploy step 1 of 8, template: write a deploy settings file for you to fill in",
                "== deploy step 2 of 8, check: check this host and your deploy settings; change nothing",
                "== deploy step 3 of 8, confirm: check your deploy settings, show them, and confirm them",
                "== deploy step 4 of 8, image: make sure the image is on this host and suits this package",
                "== deploy step 5 of 8, login: set each login and secret your settings need",
                "== deploy step 6 of 8, start: start the container with your confirmed settings",
                "== deploy step 7 of 8, doctor: check the running monitor and everything it reaches",
                "== deploy step 8 of 8, first tick: wait for the monitor's first tick",
            ],
        )

    def test_every_banner_fits_a_line(self):
        # With the dry run's ending too, no banner is longer than a check row's line.
        for s in BUILD_STEPS + DEPLOY_STEPS:
            for dry_run in (False, True):
                with self.subTest(step=(s.path, s.n), dry_run=dry_run):
                    self.assertLessEqual(
                        len(steps.banner(s, dry_run=dry_run)), checks.WIDTH
                    )

    def test_a_dry_run_says_so(self):
        self.assertTrue(
            steps.banner(BUILD_STEPS[2], dry_run=True).endswith(
                " (dry run: nothing changes)"
            )
        )
        self.assertEqual(
            steps.command_banner(
                "apply", "make the confirmed change on this deployment", dry_run=True
            ),
            "== apply: make the confirmed change on this deployment (dry run: nothing changes)",
        )

    def test_a_command_that_is_not_a_step(self):
        self.assertEqual(
            steps.command_banner("confirm", "what changes on this deployment"),
            "== confirm: what changes on this deployment",
        )

    def test_a_step_of_no_path(self):
        with self.assertRaises(ValueError):
            steps.banner(steps.Step(1, "x", "x", "x", "elsewhere"))


class TestNextLines(unittest.TestCase):
    def test_the_samples(self):
        self.assertEqual(
            steps.next_line(DEPLOY_STEPS[1], FILE=FILE),
            "next: step 2, check: rkmon check --settings ~/rkmon-deploy.yaml",
        )
        self.assertEqual(
            steps.next_line(DEPLOY_STEPS[3]),
            "next: step 4, image: rkmon image load IMAGE_FILE",
        )
        self.assertEqual(
            steps.next_line(DEPLOY_STEPS[5]),
            "next: step 6, start: rkmon start",
        )
        self.assertEqual(
            steps.next_line(DEPLOY_STEPS[0], FILE=FILE),
            "next: step 1, template: rkmon template deploy ~/rkmon-deploy.yaml",
        )

    def test_another_command_of_the_step(self):
        self.assertEqual(
            steps.next_line(DEPLOY_STEPS[3], "image pull"),
            "next: step 4, image: rkmon image pull",
        )
        with self.assertRaises(ValueError):
            steps.next_line(DEPLOY_STEPS[3], "image push")

    def test_the_name_rkmon_was_run_by(self):
        with mock.patch.object(program, "name", return_value="./rkmon"):
            self.assertEqual(
                steps.next_line(DEPLOY_STEPS[3], "image pull"),
                "next: step 4, image: ./rkmon image pull",
            )

    def test_each_word_as_the_shell_takes_it(self):
        # A word with a space is quoted, a leading ~/ left outside the quotes: the file, and the program's name.
        with mock.patch.object(program, "name", return_value="/opt/my tools/rkmon"):
            self.assertEqual(
                steps.next_line(DEPLOY_STEPS[1], FILE="~/my settings.yaml"),
                "next: step 2, check: '/opt/my tools/rkmon' check --settings ~/'my settings.yaml'",
            )
            self.assertEqual(
                steps.step_command(BUILD_STEPS[4], IMAGE_FILE="my image.tar"),
                "step 5, ship: '/opt/my tools/rkmon' image save 'my image.tar' --settings FILE",
            )

    def test_the_step_and_its_command(self):
        self.assertEqual(
            steps.step_command(DEPLOY_STEPS[0], FILE=FILE),
            "step 1, template: rkmon template deploy ~/rkmon-deploy.yaml",
        )
        self.assertEqual(
            steps.step_command(DEPLOY_STEPS[3], "image pull"),
            "step 4, image: rkmon image pull",
        )
        with self.assertRaises(ValueError):
            steps.step_command(BUILD_STEPS[4], "image pull")
        # the command that checks a step's work changes nothing: no next line names it
        with self.assertRaises(ValueError):
            steps.step_command(DEPLOY_STEPS[3], DEPLOY_STEPS[3].check)

    def test_the_step_and_its_command_as_a_fix(self):
        # As a fix: the step's words, then its command, which a wrap keeps whole; as text, step_command's line.
        fix = steps.step_fix(DEPLOY_STEPS[0], FILE=FILE)
        self.assertIsInstance(fix, checks.Steps)
        self.assertEqual(fix.commands, ("rkmon template deploy ~/rkmon-deploy.yaml",))
        self.assertEqual(str(fix), steps.step_command(DEPLOY_STEPS[0], FILE=FILE))
        with self.assertRaises(ValueError):
            steps.step_fix(BUILD_STEPS[4], "image pull")

    def test_the_command_alone(self):
        # Once deploy step 8 has passed after a start, a next line names a deploy step's command without its step.
        fix = steps.step_fix(DEPLOY_STEPS[0], FILE=FILE, plain=True)
        self.assertIsInstance(fix, checks.Steps)
        self.assertEqual(fix.commands, ("rkmon template deploy ~/rkmon-deploy.yaml",))
        self.assertEqual(str(fix), "rkmon template deploy ~/rkmon-deploy.yaml")
        self.assertEqual(
            steps.next_line(DEPLOY_STEPS[1], FILE=FILE, plain=True),
            "next: rkmon check --settings ~/rkmon-deploy.yaml",
        )
        self.assertEqual(
            steps.step_command(DEPLOY_STEPS[3], "image pull", plain=True),
            "rkmon image pull",
        )
        with self.assertRaises(ValueError):
            steps.next_line(DEPLOY_STEPS[3], "image push", plain=True)

    def test_command_name(self):
        self.assertEqual(
            [
                steps.command_name(c)
                for s in DEPLOY_STEPS
                for c in (s.command, *s.others)
            ],
            [
                "template deploy",
                "check",
                "confirm",
                "image load",
                "image pull",
                "login",
                "start",
                "doctor",
                "first-tick",
            ],
        )
        self.assertEqual(
            steps.command_name("image save IMAGE_FILE --settings FILE"), "image save"
        )

    def test_typed(self):
        self.assertEqual(
            steps.typed("template deploy [--profile PROFILE] FILE", FILE="f.yaml"),
            ("template", "deploy", "f.yaml"),
        )
        # An optional part is put in when a word is given for each placeholder in it.
        self.assertEqual(
            steps.typed(
                "template deploy [--profile PROFILE] FILE",
                FILE="-",
                PROFILE="review-only",
            ),
            ("template", "deploy", "--profile", "review-only", "-"),
        )
        self.assertEqual(
            steps.typed("image save IMAGE_FILE --settings FILE", FILE="b.yaml"),
            ("image", "save", "IMAGE_FILE", "--settings", "b.yaml"),
        )
        self.assertEqual(
            steps.typed("image save IMAGE_FILE --settings FILE", IMAGE_FILE="m.tar"),
            ("image", "save", "m.tar", "--settings", "FILE"),
        )
        # A word given stays one word, whatever it holds.
        self.assertEqual(
            steps.typed("check --settings FILE", FILE="~/my settings.yaml"),
            ("check", "--settings", "~/my settings.yaml"),
        )

    def test_the_command_line(self):
        line = steps.command_line(
            "template deploy [--profile PROFILE] FILE", FILE="~/my settings.yaml"
        )
        self.assertEqual(
            line,
            checks.Command(("rkmon", "template", "deploy", "~/my settings.yaml")),
        )
        self.assertEqual(str(line), "rkmon template deploy ~/'my settings.yaml'")
        with mock.patch.object(program, "name", return_value="./rkmon"):
            self.assertEqual(str(steps.command_line("login")), "./rkmon login")

    def test_the_done_line(self):
        self.assertEqual(steps.done_line("the monitor runs"), "done: the monitor runs")
        with self.assertRaises(ValueError):
            steps.done_line(" ")


if __name__ == "__main__":
    unittest.main()
