# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""One way to finish a command (rkmon_lib/outcome.py): each outcome word and its exit code, the
outcome line's form, the fix and next lines after it, a command that ends short, the tally of check rows, and the exit
codes as help gives them.
"""

import io
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import checks, outcome, wrapping  # noqa: E402
from rkmon_lib.checks import (  # noqa: E402
    DELETE,
    WIDTH,
    Check,
    Command,
    Described,
    Edit,
    Steps,
)

# A fix for the check rows that need one.
FIX = Command(("rkmon", "check", "--settings", "~/rkmon-build.yaml"))

# The outcome words and their exit codes.
TABLE = {
    "PASSED": 0,
    "FAILED": 4,
    "REFUSED": 4,
    "CANCELLED": 1,
    "ERROR": 1,
    "ROLLED BACK": 1,
    "INTERRUPTED": 130,
    "NOTHING TO DO": 0,
    **{
        w: 0
        for w in (
            "WROTE BUILT SAVED PUSHED LOADED PULLED CONFIRMED APPLIED RESTORED STARTED STOPPED PAUSED RESUMED SET "
            "REMOVED UPDATED MIGRATED ADOPTED RELEASED INSTALLED"
        ).split()
    },
}


def finished(*args, **kwargs):
    out = io.StringIO()
    rc = outcome.finish(*args, out=out, **kwargs)
    return rc, out.getvalue()


class TestWords(unittest.TestCase):
    def test_each_word_has_its_exit_code(self):
        self.assertEqual(outcome.WORDS, TABLE)
        for word, rc in TABLE.items():
            self.assertEqual(outcome.code(word), rc, word)

    def test_an_earlier_step_makes_failed_and_refused_exit_3(self):
        self.assertEqual(outcome.code("FAILED", earlier_step=True), 3)
        self.assertEqual(outcome.code("REFUSED", earlier_step=True), 3)
        for word in TABLE:
            if word not in ("FAILED", "REFUSED"):
                with self.assertRaises(ValueError):
                    outcome.code(word, earlier_step=True)

    def test_no_other_exit_code(self):
        codes = {n for n, _, _ in outcome.EXIT_CODES}
        self.assertEqual(codes, {0, 1, 2, 3, 4, 130})
        self.assertLessEqual(
            set(TABLE.values()) | {outcome.EARLIER_STEP, outcome.USAGE}, codes
        )

    def test_an_unknown_word_is_the_callers_mistake(self):
        for word in ("OK", "passed", "DONE", ""):
            with self.assertRaises(ValueError):
                outcome.line(word, "x")
            with self.assertRaises(ValueError):
                outcome.finish(word, "x", out=io.StringIO())


class TestLine(unittest.TestCase):
    def test_a_thing_made_or_moved_follows_its_word(self):
        self.assertEqual(
            outcome.line("BUILT", "rocke-monitor:2026.11.15"),
            "BUILT rocke-monitor:2026.11.15",
        )
        self.assertEqual(
            outcome.line(
                "WROTE",
                "~/rkmon-deploy.yaml (profile review-only; yours only, permissions 600)",
            ),
            "WROTE ~/rkmon-deploy.yaml (profile review-only; yours only, permissions 600)",
        )
        for word in ("SAVED", "PUSHED", "LOADED", "PULLED"):
            self.assertEqual(outcome.line(word, "x:1"), f"{word} x:1")

    def test_every_other_word_takes_a_colon(self):
        self.assertEqual(outcome.line("PASSED", "9 checks"), "PASSED: 9 checks")
        self.assertEqual(
            outcome.line("NOTHING TO DO", "nothing changed"),
            "NOTHING TO DO: nothing changed",
        )
        self.assertEqual(
            outcome.line("CONFIRMED", "4 changes wait"), "CONFIRMED: 4 changes wait"
        )

    def test_a_word_alone(self):
        self.assertEqual(outcome.line("CANCELLED"), "CANCELLED")
        self.assertEqual(outcome.line("PASSED", "  "), "PASSED")


class TestFinish(unittest.TestCase):
    def test_the_outcome_then_the_next_line(self):
        rc, text = finished(
            "CONFIRMED",
            "~/rkmon-deploy.yaml is this deployment's settings file",
            next="next: step 4, image: rkmon image load IMAGE_FILE",
        )
        self.assertEqual(rc, 0)
        self.assertEqual(
            text,
            "CONFIRMED: ~/rkmon-deploy.yaml is this deployment's settings file\n"
            "next: step 4, image: rkmon image load IMAGE_FILE\n",
        )

    def test_a_done_line(self):
        rc, text = finished(
            "APPLIED", "done", next="done: the monitor runs your confirmed settings"
        )
        self.assertEqual(
            (rc, text.splitlines()[-1]),
            (0, "done: the monitor runs your confirmed settings"),
        )

    def test_no_next_line(self):
        self.assertEqual(
            finished("CANCELLED", "nothing confirmed"),
            (1, "CANCELLED: nothing confirmed\n"),
        )

    def test_a_next_line_starts_with_next_or_done(self):
        for bad in ("rkmon apply", "next rkmon apply", "Next: rkmon apply"):
            with self.assertRaises(ValueError):
                finished("ERROR", "x", next=bad)

    def test_the_container_name_refusal_with_its_fix(self):
        rc, text = finished(
            "REFUSED",
            "the container name rocke-monitor is used by another deployment on this host; nothing changed",
            next="next: rkmon confirm --settings ~/rkmon-deploy.yaml",
            fix=Steps(
                (
                    Edit(
                        "container.name",
                        "~/rkmon-deploy.yaml",
                        Described("a name of its own, such as rocke-monitor-2"),
                    ),
                    "(",
                    Command(("docker", "compose", "ls", "--all")),
                    "lists the names in use)",
                )
            ),
        )
        self.assertEqual(rc, 4)
        self.assertEqual(
            text,
            "REFUSED: the container name rocke-monitor is used by another deployment on this host; nothing changed\n"
            "fix: set container.name in ~/rkmon-deploy.yaml to a name of its own, such as rocke-monitor-2 (docker compose "
            "ls --all\n"
            "     lists the names in use)\n"
            "next: rkmon confirm --settings ~/rkmon-deploy.yaml\n",
        )
        self.assertTrue(all(len(line) <= WIDTH for line in text.splitlines()))

    def test_a_long_outcome_line_and_its_fix_wrap(self):
        # The outcome line and the next line wrap at a check row's width, their further lines at column 0; the fix's
        # at column 5.
        rc, text = finished(
            "REFUSED",
            "~/rkmon-deploy.yaml has a secret_values section (3 entries), and no secret goes in a settings file; "
            "nothing changed",
            fix=Steps(
                (
                    "move each value into a file only you can read,",
                    Edit("secret_values", "~/rkmon-deploy.yaml", DELETE),
                    ", and run migrate again; after",
                    Command(("rkmon", "confirm")),
                    ",",
                    Command(("rkmon", "login")),
                    "lists each secret and installs it",
                )
            ),
            next="next: rkmon migrate --settings ~/rkmon-deploy.yaml",
        )
        self.assertEqual(rc, 4)
        self.assertEqual(
            text,
            "REFUSED: ~/rkmon-deploy.yaml has a secret_values section (3 entries), and no secret goes in a settings "
            "file; nothing\n"
            "changed\n"
            "fix: move each value into a file only you can read, delete secret_values from ~/rkmon-deploy.yaml, and "
            "run migrate\n"
            "     again; after rkmon confirm, rkmon login lists each secret and installs it\n"
            "next: rkmon migrate --settings ~/rkmon-deploy.yaml\n",
        )

    def test_a_long_next_line_wraps(self):
        self.assertEqual(
            finished(
                "REFUSED",
                "nothing written",
                next="next: step 1, template: rkmon template deploy ~/rkmon-deploy.yaml (with a file of your own: "
                "rkmon check --settings FILE)",
            ),
            (
                4,
                "REFUSED: nothing written\n"
                "next: step 1, template: rkmon template deploy ~/rkmon-deploy.yaml (with a file of your own:\n"
                "rkmon check --settings FILE)\n",
            ),
        )

    def test_a_refusal_for_an_earlier_step_exits_3(self):
        rc, _ = finished(
            "REFUSED", "no image", next="next: rkmon image pull", earlier_step=True
        )
        self.assertEqual(rc, 3)

    def test_a_fix_is_never_words_alone(self):
        for words in ("set it", " ", ""):
            with self.subTest(words=words), self.assertRaises(ValueError):
                finished("ERROR", "x", fix=words)
        self.assertEqual(
            finished("ERROR", "x", fix=FIX),
            (1, "ERROR: x\nfix: rkmon check --settings ~/rkmon-build.yaml\n"),
        )

    def test_the_fix_label_is_the_check_rows(self):
        lines = outcome.fix_lines(Steps([Command(("set", "it"))] * 30))
        self.assertGreater(len(lines), 1)
        self.assertTrue(lines[0].startswith(checks.FIX_LABEL + "set it"))
        indent = " " * len(checks.FIX_LABEL)
        self.assertTrue(
            all(line.startswith(indent + "set") for line in lines[1:]), lines
        )
        self.assertEqual(checks.FIX_COLUMN, checks.NAME_COLUMN + len(checks.FIX_LABEL))

    def test_it_writes_to_stdout_by_default(self):
        out = io.StringIO()
        saved, sys.stdout = sys.stdout, out
        try:
            rc = outcome.finish("PASSED", "1 check")
        finally:
            sys.stdout = saved
        self.assertEqual((rc, out.getvalue()), (0, "PASSED: 1 check\n"))


class TestStopped(unittest.TestCase):
    """A command that ends short: raised where it stops, finished where it is caught."""

    def test_its_rows_then_the_outcome_line_and_its_fix(self):
        rows = [
            Check("config.json", "PASS", "blank"),
            Check("private file", "FAIL", "docker/.env is in the new package", fix=FIX),
        ]
        text = (
            "the new package holds private files; nothing changed (fix each FAIL above)"
        )
        out = io.StringIO()
        rc = outcome.Stopped("REFUSED", text, fix=FIX, rows=rows).finish(out)
        self.assertEqual(
            (rc, out.getvalue()),
            (4, checks.render(rows) + "\n" + finished("REFUSED", text, fix=FIX)[1]),
        )

    def test_a_refusal_shown_its_own_way_then_the_outcome_line(self):
        """ABOVE, a refusal as it shows itself (a private file's: its rows, then the line that names the rules),
        after the rows, if any, and before the outcome line."""
        rows = [Check("private file", "FAIL", "~/s.yaml exists", fix=FIX)]
        above = checks.render(rows) + "\nWhere the rules are: README.md."
        text = "~/s.yaml is not written (the row above says why)"
        out = io.StringIO()
        rc = outcome.Stopped("REFUSED", text, above=above).finish(out)
        self.assertEqual(
            (rc, out.getvalue()), (4, above + "\n" + finished("REFUSED", text)[1])
        )

    def test_an_earlier_step_exits_3(self):
        out = io.StringIO()
        stop = outcome.Stopped(
            "REFUSED", "git is not on this host; nothing changed", FIX, True
        )
        self.assertEqual(
            (stop.finish(out), out.getvalue()),
            (
                3,
                "REFUSED: git is not on this host; nothing changed\n"
                "fix: rkmon check --settings ~/rkmon-build.yaml\n",
            ),
        )

    def test_the_next_line_the_caller_gives_ends_it(self):
        out = io.StringIO()
        stop = outcome.Stopped(
            "REFUSED", "git is not on this host; nothing changed", FIX, True
        )
        self.assertEqual(
            (stop.finish(out, next="next: rkmon package update"), out.getvalue()),
            (
                3,
                "REFUSED: git is not on this host; nothing changed\n"
                "fix: rkmon check --settings ~/rkmon-build.yaml\n"
                "next: rkmon package update\n",
            ),
        )

    def test_its_message_is_what_happened(self):
        stop = outcome.Stopped("ERROR", "the renames failed", rows=iter([]))
        self.assertEqual(
            (str(stop), stop.word, stop.fix, stop.earlier_step, stop.rows),
            ("the renames failed", "ERROR", None, False, ()),
        )

    def test_it_writes_to_stdout_by_default(self):
        out = io.StringIO()
        saved, sys.stdout = sys.stdout, out
        try:
            rc = outcome.Stopped("ERROR", "x").finish()
        finally:
            sys.stdout = saved
        self.assertEqual((rc, out.getvalue()), (1, "ERROR: x\n"))


class TestTally(unittest.TestCase):
    def test_failed_says_how_many_checks_failed(self):
        rows = [Check("docker", "PASS", "ok"), Check("settings file", "PASS", "ok")]
        rows += [
            Check(f"key{i}", "FAIL", "still the template's", fix=FIX) for i in range(3)
        ]
        rows += [
            Check("github_cli.version", "WARN", "not pinned", fix=FIX),
            Check("disk", "PASS", "541 GiB"),
        ]
        self.assertEqual(
            outcome.tally(rows),
            ("FAILED", "3 of 7 checks failed; nothing changed (fix each FAIL above)"),
        )

    def test_passed_with_its_warnings(self):
        rows = [
            Check("a", "PASS", "ok"),
            Check("b", "WARN", "look", fix=FIX),
            Check("c", "PASS", "ok"),
        ]
        self.assertEqual(outcome.tally(rows), ("PASSED", "3 checks (1 warning)"))
        rows.append(Check("d", "WARN", "look", fix=FIX))
        self.assertEqual(outcome.tally(rows), ("PASSED", "4 checks (2 warnings)"))
        self.assertEqual(outcome.tally(rows[:1]), ("PASSED", "1 check"))

    def test_off_and_skip_rows_are_not_counted(self):
        rows = [
            Check("a", "PASS", "ok"),
            Check("teams_webhook", "OFF", "alerts are off"),
            Check("image", "SKIP", "checked at step 4"),
            Check("b", "FAIL", "bad", fix=FIX),
        ]
        self.assertEqual(
            outcome.tally(rows)[1],
            "1 of 2 checks failed; nothing changed (fix each FAIL above)",
        )

    def test_what_a_failure_left_as_it_was(self):
        """A command that did part of its work before its checks (a confirm records your file first) says what the
        failure left as it was; a pass says only its counts."""
        rows = [Check("a", "PASS", "ok"), Check("b", "FAIL", "bad", fix=FIX)]
        self.assertEqual(
            outcome.tally(rows, "nothing confirmed"),
            ("FAILED", "1 of 2 checks failed; nothing confirmed (fix each FAIL above)"),
        )
        self.assertEqual(
            outcome.tally(rows[:1], "nothing confirmed"), ("PASSED", "1 check")
        )


class TestExitCodes(unittest.TestCase):
    def test_the_sentence_of_rkmon_help(self):
        # rkmon help's last lines, joined.
        self.assertEqual(
            outcome.exit_summary(),
            "0 done; 1 an error, a roll back, or you said no; 2 usage error; 3 an earlier step is not done; "
            "4 refused or a check failed (nothing changed); 130 interrupted.",
        )

    def test_held_each_code_stays_with_its_words(self):
        text = outcome.exit_summary(held=True)
        self.assertEqual(text.replace(wrapping.HOLD, " "), outcome.exit_summary())
        self.assertEqual(text.count("; "), len(outcome.EXIT_CODES) - 1)
        self.assertIn(wrapping.held("2 usage error"), text)

    def test_the_long_words_of_the_exit_codes(self):
        long = {n: words for n, _, words in outcome.EXIT_CODES}
        self.assertEqual(
            long[2], "a usage error, including a question with no terminal and no --yes"
        )
        self.assertIn(
            "(docker, buildx, a settings file, a confirm, an image, a login, a first start)",
            long[3],
        )


if __name__ == "__main__":
    unittest.main()
