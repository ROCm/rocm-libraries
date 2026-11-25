# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The one way to wrap plain English and to join words (rkmon_lib/wrapping.py): lines of at most a width, no word
broken, quoted phrases, commands and held items kept on one line, a line break kept with its own indent; items joined
in a sentence, and a number with its noun.
"""

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rkmon_lib import wrapping  # noqa: E402
from rkmon_lib.checks import Command, Steps  # noqa: E402
from rkmon_lib.wrapping import wrap  # noqa: E402


class TestWrap(unittest.TestCase):
    def test_lines_of_at_most_the_width_after_their_leads(self):
        text = "one two three four five six seven eight nine ten"
        self.assertEqual(
            wrap(text, 20, "# ", "#   "),
            ["# one two three four", "#   five six seven", "#   eight nine ten"],
        )

    def test_no_word_broken_and_no_break_at_a_hyphen(self):
        self.assertEqual(
            wrap("a https://example.com/a-very-long-address b", 10),
            ["a", "https://example.com/a-very-long-address", "b"],
        )
        self.assertEqual(wrap("rocke-monitor-2 x", 12), ["rocke-monitor-2", "x"])

    def test_a_quoted_phrase_that_fits_is_never_split(self):
        text = 'Change a deployment (rkmon help change; the steps: QUICK_USER_GUIDE.md, "Change a setting"):'
        self.assertEqual(
            wrap(text, 79),
            [
                "Change a deployment (rkmon help change; the steps: QUICK_USER_GUIDE.md,",
                '"Change a setting"):',
            ],
        )

    def test_a_quoted_phrase_longer_than_a_line_wraps_as_words(self):
        lines = wrap('x "one two three four five six" y', 12)
        self.assertTrue(all(len(line) <= 12 for line in lines), lines)
        self.assertEqual(" ".join(lines), 'x "one two three four five six" y')

    def test_a_command_that_fits_on_a_line_moves_whole_to_the_next_line(self):
        """A line copied runs the whole command: where the line would pass the width, the command moves whole to the
        next line. A text that names no commands, such as a next line, has the program's (program_commands).
        """
        self.assertEqual(
            wrap(
                "next: step 4, image: ~/a-long-folder/docker/rkmon image load IMAGE_FILE",
                50,
            ),
            [
                "next: step 4, image:",
                "~/a-long-folder/docker/rkmon image load IMAGE_FILE",
            ],
        )

    def test_a_command_longer_than_a_line_wraps_as_words(self):
        self.assertEqual(
            wrap("next: rkmon check --settings ~/rkmon-deploy.yaml --yes", 30),
            ["next: rkmon check --settings", "~/rkmon-deploy.yaml --yes"],
        )

    def test_the_commands_of_the_program_in_a_text(self):
        def commands(text):
            return [text[s:e] for s, e in wrapping.program_commands(text)]

        self.assertEqual(  # the program's name or a path to it, then its words up to a word that starts with (
            commands(
                "step 1: rkmon template deploy ~/rkmon-deploy.yaml (with a file of your own: ./rkmon check "
                "--settings FILE)"
            ),
            [
                "rkmon template deploy ~/rkmon-deploy.yaml",
                "./rkmon check --settings FILE",
            ],
        )
        self.assertEqual(  # a word that joins steps ends a command, and so does a mark after its last word
            commands(
                "run rkmon stop, then rkmon start and (rkmon status) or docker/rkmon logs."
            ),
            ["rkmon stop", "rkmon start", "rkmon status", "docker/rkmon logs"],
        )
        self.assertEqual(
            commands("rkmon_lib/wrapping.py and ~/rkmon-deploy.yaml name none"), []
        )

    def test_a_text_that_names_its_commands_keeps_those_whole(self):
        """A fix names its commands (rkmon_lib/checks.py): each that fits on a line stays whole, the program's or
        not, and the words after it wrap as words."""
        fix = Steps(
            [
                "make it yours only:",
                Command(["chmod", "600", "~/rkmon-build.yaml"]),
                "then",
                Command(["rkmon", "check"]),
            ]
        )
        self.assertEqual(
            wrap(fix, 40, "fix: ", "     "),
            [
                "fix: make it yours only:",
                "     chmod 600 ~/rkmon-build.yaml then",
                "     rkmon check",
            ],
        )
        fix = Steps(
            [
                Command(["rkmon", "logs", "--tail", "50"]),
                "and",
                Command(["rkmon", "rk", "status"]),
                "say why",
            ]
        )
        self.assertEqual(
            wrap(fix, 46, "fix: ", "     "),
            ["fix: rkmon logs --tail 50 and rkmon rk status", "     say why"],
        )

    def test_held_words_stay_on_one_line(self):
        text = f"a {wrapping.held('b c d')} e"
        self.assertEqual(wrap(text, 6), ["a", "b c d", "e"])
        self.assertNotIn(wrapping.HOLD, "".join(wrap(text, 6)))

    def test_marked_items_break_only_before_a_mark(self):
        items = ["a b", "c d", "e f"]
        self.assertEqual(wrap(" · ".join(items), 11), ["a b · c d ·", "e f"])
        self.assertEqual(wrap(wrapping.marked(items), 11), ["a b · c d", "· e f"])
        self.assertEqual(wrap(wrapping.marked(items), 6), ["a b", "· c d", "· e f"])
        self.assertEqual(wrapping.marked(["a"]), "a")

    def test_a_line_break_starts_a_line_and_keeps_its_indent(self):
        self.assertEqual(
            wrap("first line\n  second, indented", 40, "fix: ", "     "),
            ["fix: first line", "       second, indented"],
        )

    def test_an_empty_text_is_its_lead(self):
        self.assertEqual(wrap("", 40, "# "), ["#"])
        self.assertEqual(wrap("a\n\nb", 40, "# ", "# "), ["# a", "#", "# b"])


class TestWords(unittest.TestCase):
    def test_joined(self):
        self.assertEqual(wrapping.joined(["a"]), "a")
        self.assertEqual(wrapping.joined(("a", "b")), "a and b")
        self.assertEqual(wrapping.joined(["a", "b", "c"]), "a, b and c")
        self.assertEqual(wrapping.joined([1, 2, 3], "or"), "1, 2 or 3")
        with self.assertRaises(ValueError):
            wrapping.joined([])

    def test_are(self):
        self.assertEqual(wrapping.are(["a"]), "is")
        self.assertEqual(wrapping.are(("a", "b")), "are")

    def test_counted(self):
        self.assertEqual(wrapping.counted(1, "entry", "entries"), "1 entry")
        self.assertEqual(wrapping.counted(2, "entry", "entries"), "2 entries")
        self.assertEqual(wrapping.counted(0, "check"), "0 checks")
        self.assertEqual(wrapping.counted(1, "check"), "1 check")


if __name__ == "__main__":
    unittest.main()
