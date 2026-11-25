# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""docs/FAQ.md, the one home of the questions and answers: every `rk` and `rkmon` command it names exists with the
options it gives, each answer is five lines at most and ends in a link to its fact's home, every link it
makes stays inside the package, and the README and the user guide link to it. Every lint of tools/doc_lints.py reads
it too (tests/doc_lint_files.txt lists it): its keys, and that each link reaches a file and a heading, are checked
there.
"""

import contextlib
import io
import pathlib
import re
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.dont_write_bytecode = True

from rkmon_lib import commands as table  # noqa: E402
from rkmon_lib import secrets_registry as sr  # noqa: E402

FAQ = ROOT / "docs" / "FAQ.md"


def faq() -> str:
    return FAQ.read_text(encoding="utf-8")


def answers(text: str) -> dict[str, list[str]]:
    """Each question (a `### ` heading) and the lines of its answer, which end at the next heading."""
    out = {}
    for part in re.split(r"(?m)^(?=#{1,3} )", text):
        head, _, body = part.partition("\n")
        if head.startswith("### "):
            out[head[4:]] = [line for line in body.splitlines() if line.strip()]
    return out


def code(text: str) -> list[str]:
    """The inline code spans and the lines of the fenced blocks."""
    out = []
    for block in re.findall(r"(?ms)^```[^\n]*\n(.*?)^```", text):
        out += [line.strip() for line in block.splitlines() if line.strip()]
    out += re.findall(r"`([^`\n]+)`", re.sub(r"(?ms)^```.*?^```", "", text))
    return out


def rk_help(argv: list[str]) -> str:
    import rkmon

    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.suppress(SystemExit):
        rkmon.main(argv)
    return out.getvalue()


def wrapper_names() -> set[str]:
    """The first word of each rkmon command, from the command table (rkmon_lib/commands.py)."""
    return {r.words[0] for r in table.TABLE}


def wrapper_words() -> dict[str, set[str]]:
    """The rkmon commands that take a word of their own after them: each group's words from the command table (image:
    load, save, ...), and login's, the secrets registry's logins."""
    out = {}
    for r in table.TABLE:
        if len(r.words) > 1:
            out.setdefault(r.words[0], set()).add(r.words[1])
    out["login"] = {e.name for e in sr.entries({}) if e.shape == sr.LOGIN}
    return out


def wrapper_flags(words: list[str]) -> set[str]:
    """The options an rkmon command that starts with these words takes: its row's, and every command's."""
    row = table.find(words[:2], table.TABLE) or table.find(words[:1], table.TABLE)
    own = row.options if row is not None else ()
    return {f for o in (*own, *table.GLOBAL_OPTIONS, *table.LEADING) for f in o.flags}


def commands(spans: list[str]) -> list[tuple[str, list[str]]]:
    """(tool, words) for every command a span holds: ("rk", [...]) or ("rkmon", [...])."""
    out = []
    for s in spans:
        for m in re.finditer(r"(?:^|[\s(])((?:\./|docker/)?rkmon|rk)\s+([^|;`]+)", s):
            tool = "rkmon" if m.group(1).endswith("rkmon") else "rk"
            words = m.group(2).split()
            if tool == "rkmon" and words and words[0] == "rk":
                tool, words = "rk", words[1:]
            if words and re.match(r"[a-z]", words[0]):  # not a placeholder (<...>, ...)
                out.append((tool, words))
    return out


class TestFaq(unittest.TestCase):
    def test_it_starts_like_every_document_and_has_its_three_audiences(self):
        text = faq()
        self.assertTrue(text.startswith("<!--\nCopyright (c) Advanced Micro Devices"))
        for heading in (
            "## For people deploying and operating it",
            "## For PR authors receiving its reviews",
            "## For team leads",
        ):
            self.assertIn(heading, text)
        self.assertGreaterEqual(len(re.findall(r"^### .+\?$", text, re.M)), 20)

    def test_the_readme_and_the_user_guide_link_to_it(self):
        """The deploy guide names, as a whole doc, only the doc before it, the one after it and the reference
        (tests/test_docs.py)."""
        for name in (
            "README.md",
            "QUICK_USER_GUIDE.md",
        ):
            self.assertIn(
                "[docs/FAQ.md](docs/FAQ.md)",
                (ROOT / name).read_text(encoding="utf-8"),
                name,
            )

    def test_every_rk_command_and_option_exists(self):
        subs = re.search(r"\{([a-z0-9,-]+)\}", rk_help(["--help"])).group(1).split(",")
        found = 0
        for tool, words in commands(code(faq())):
            if tool != "rk":
                continue
            found += 1
            self.assertIn(words[0], subs, f"rk {' '.join(words)}")
            usage = rk_help([words[0], "--help"])
            for opt in (w for w in words[1:] if w.startswith("--")):
                self.assertIn(opt.split("=")[0], usage, f"rk {' '.join(words)}")
        self.assertGreater(found, 5)

    def test_every_rkmon_command_and_option_exists(self):
        names, after = wrapper_names(), wrapper_words()
        found = 0
        for tool, words in commands(code(faq())):
            if tool != "rkmon":
                continue
            found += 1
            self.assertIn(words[0], names, f"rkmon {' '.join(words)}")
            # a word of its own after the command (an option after it is checked below, with the others)
            if words[0] in after and len(words) > 1 and not words[1].startswith("-"):
                self.assertIn(words[1], after[words[0]], f"rkmon {' '.join(words)}")
            flags = wrapper_flags(words)
            for opt in (w for w in words[1:] if w.startswith("--")):
                self.assertIn(opt.split("=")[0], flags, f"rkmon {' '.join(words)}")
        self.assertGreater(found, 10)

    def test_your_own_model_for_a_pass(self):
        """The operators' answer on a model of their own for one pass: the binding, its key, the pass; which passes
        take any model; and the user guide's section it links, which says how to make the change.
        """
        text = faq()
        head = "### How do I use my own model for a specific pass?"
        self.assertIn(head, text)
        self.assertNotIn("Can I use a local model", text)
        part = text.split(head, 1)[1].split("\n### ", 1)[0]
        self.assertLess(
            text.index(head), text.index("## For PR authors receiving its reviews")
        )  # the operators' group
        for words in (
            "model_bindings",
            "provider",
            "install its key",
            "features.review_automation.passes.ci_triage_pass: { model: glm-flash, effort: default }",
            "first_review_pass",
            "ci_triage_pass",
            "first_code_quality_pass",
            "notice_draft_pass",
            "verification_pass",
            "code_quality_pass",
            "leads",
            "(../QUICK_USER_GUIDE.md#use-another-model-for-a-pass).",
        ):
            self.assertIn(words, part)

    def test_each_answer_is_five_lines_at_most_and_ends_in_a_link(self):
        """Each answer is five lines at most and ends in a link to its fact's home: its last words before
        its full stop are a markdown link, or one in parentheses. Every link stays inside the package; the links lint
        checks that each reaches a file and a heading there."""
        found = answers(faq())
        self.assertGreaterEqual(len(found), 20)
        for question, lines in found.items():
            self.assertLessEqual(len(lines), 5, question)
            self.assertRegex(" ".join(lines), r"\]\([^)\s]+\)\)?\.$", question)
        for link in re.findall(r"\]\(([^)\s]+)\)", faq()):
            self.assertNotRegex(
                link, r"^[a-z]+:", "the FAQ links inside the package only"
            )

    def test_the_check_catches_a_missing_command_and_option(self):
        self.assertEqual(commands(["./rkmon rk roles"]), [("rk", ["roles"])])
        self.assertEqual(
            commands(["./rkmon apply --timeout-minutes 30"]),
            [("rkmon", ["apply", "--timeout-minutes", "30"])],
        )
        self.assertNotIn("rebuild", wrapper_names())
        self.assertNotIn("swap", wrapper_names())
        self.assertIn("apply", wrapper_names())
        self.assertNotIn("--timeout", wrapper_flags(["apply"]))
        self.assertIn("--timeout-minutes", wrapper_flags(["apply"]))
        self.assertIn("--dry-run", wrapper_flags(["image", "load"]))


if __name__ == "__main__":
    unittest.main()
