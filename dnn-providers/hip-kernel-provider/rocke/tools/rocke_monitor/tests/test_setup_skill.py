# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The setup skill, /rkmon-setup (.claude/skills/rkmon-setup/SKILL.md): its front matter, with no allowed-tools line,
then rocKE's header; each rkmon command it names is one it runs or names for the user to run (tests/test_commands.py
parses each command a doc writes, the skill's among them); the commands it runs are the ones listed here, the look-ups
among them its read-only ones; it never runs what it names for the user; each key it names is a key of a
settings file; the package ships it (package_files.tracked), and a repository whose own ignore file leaves out every
.claude/ folder tracks it all the same, while a skill of the user's own and Claude Code's local files stay out of git
(.gitignore, which lets in each skill tools/generated.py's claude_skills names); the privacy check reads it as a skill,
and the image's build context leaves out all of .claude/ (image.PACKAGE_ONLY), so in the image alone the tests that
read the skill skip (build_context.left_out_here); and the docs' lints read it (tests/doc_lint_files.txt, which
tests/test_doc_lints.py lints).
"""

import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tools"), str(ROOT / "tests")]
sys.dont_write_bytecode = True

import build_context  # noqa: E402
import doc_lints  # noqa: E402
import generated  # noqa: E402
import make_package  # noqa: E402
import test_commands  # noqa: E402
from rkmon_lib import (  # noqa: E402
    build_plan,
    commands,
    image,
    package_files,
    privacy,
    secrets_registry,
)

SKILL = ".claude/skills/rkmon-setup/SKILL.md"
# A test that reads the skill skips in the image, which leaves .claude out (image.PACKAGE_ONLY): the skill serves the
# deployer's own Claude Code session on the host. In a checkout without the skill, the test fails.
NOT_HERE = build_context.left_out_here(SKILL)
READS_THE_SKILL = unittest.skipIf(bool(NOT_HERE), NOT_HERE)
# One of Claude Code's local files in the package folder, which may have private values in it.
LOCAL = ".claude/settings.local.json"
# A skill of the user's own in the package folder, which no package ships.
OWN = ".claude/skills/my-own-skill/SKILL.md"
# The rkmon commands the skill runs, each once the user says yes, by their rows' words in the command table:
# template deploy, template section, check and migrate (a file in an earlier layout).
RUNS = {("template", "deploy"), ("template", "section"), ("check",), ("migrate",)}
# The rkmon commands the skill names for the user to run, which it never runs: confirm and next, login
# (deploy step 5) and start.
THE_USER_S = {("confirm",), ("next",), ("login",), ("start",)}
# The skill's look-ups, each read-only.
LOOK_UPS = {
    "gh api user --jq .login",
    "gh label list -R <org>/<repo>",
    "nproc",
    "free -g",
    "docker version",
    "docker compose ls --all",
    "ss -ltn",
}
# The skill's sections of the commands it runs, and of what it never does.
RUN = "The commands you run"
NEVER = "What you never do"
CODE = re.compile(r"`([^`]+)`")


def text() -> str:
    return (ROOT / SKILL).read_text(encoding="utf-8")


def section(heading: str) -> str:
    """The skill's section under the heading `## HEADING`, up to the next heading of its level."""
    found = re.search(
        rf"^## {re.escape(heading)}\n(.*?)(?=^## |\Z)", text(), re.M | re.S
    )
    if not found:
        raise AssertionError(f"{SKILL} has no section {heading!r}")
    return found.group(1)


def code(part: str) -> list:
    """Each text in backticks in PART, a command among them."""
    return CODE.findall(part)


def rkmon(part: str) -> list:
    """Each rkmon command in backticks in PART."""
    return [c for c in code(part) if c.startswith("rkmon ")]


def row(command: str) -> tuple:
    """The words of the command table's row that COMMAND, an rkmon command as the skill writes it, names: the most of
    its words after the program's name that are a row's name."""
    words = shlex.split(command)[1:]
    for n in range(len(words), 0, -1):
        found = commands.find(words[:n], commands.TABLE)
        if found is not None:
            return found.words
    raise AssertionError(f"{command!r} names no command of the table")


class TestTheSkill(unittest.TestCase):
    @READS_THE_SKILL
    def test_its_front_matter_then_the_header_and_no_allowed_tools(self):
        """Claude Code reads the front matter at the file's very top: the name, which is its folder's and the slash
        command's, the description and the argument hint. With no allowed-tools line, each command asks first.
        """
        t = text()
        self.assertTrue(t.startswith("---\n"))
        front = t[4 : t.index("\n---\n", 4)].splitlines()
        self.assertEqual(
            [line.split(":", 1)[0] for line in front],
            ["name", "description", "argument-hint"],
        )
        values = dict(line.split(": ", 1) for line in front)
        self.assertEqual(values["name"], pathlib.PurePosixPath(SKILL).parent.name)
        self.assertEqual(values["argument-hint"], "[FILE]")
        self.assertIn("/rkmon-setup FILE", values["description"])
        self.assertEqual(make_package.header_problem(ROOT / SKILL), "")

    @READS_THE_SKILL
    def test_each_command_it_names_parses(self):
        """tests/test_commands.py parses each command a doc writes, in each of its forms: the skill is one of its
        docs."""
        docs = {d for pattern in test_commands.DOCS for d in ROOT.glob(pattern)}
        self.assertIn(ROOT / SKILL, docs)

    @READS_THE_SKILL
    def test_each_rkmon_command_it_names_it_runs_or_leaves_to_the_user(self):
        named = rkmon(text())
        self.assertTrue(named)
        self.assertEqual({row(c) for c in named} - RUNS - THE_USER_S, set())

    @READS_THE_SKILL
    def test_it_runs_its_commands_and_look_ups_and_no_others(self):
        part = section(RUN)
        self.assertEqual({row(c) for c in rkmon(part)}, RUNS)
        self.assertEqual(set(code(part)) - set(rkmon(part)), LOOK_UPS)

    @READS_THE_SKILL
    def test_it_never_runs_what_it_names_for_the_user(self):
        """Its section "What you never do": confirm, start and login are the user's."""
        self.assertEqual(
            {row(c) for c in rkmon(section(NEVER))},
            {("confirm",), ("start",), ("login",)},
        )
        self.assertEqual({row(c) for c in rkmon(section(RUN))} & THE_USER_S, set())

    def test_a_command_s_row_is_the_most_of_its_words_that_name_one(self):
        self.assertEqual(
            row("rkmon template deploy --profile PROFILE FILE"), ("template", "deploy")
        )
        self.assertEqual(row("rkmon template deploy"), ("template", "deploy"))
        self.assertEqual(row("rkmon login github_read"), ("login",))
        with self.assertRaisesRegex(AssertionError, "names no command"):
            row("rkmon no-such-command")

    @READS_THE_SKILL
    def test_each_key_it_names_is_a_key_of_a_settings_file(self):
        """The keys lint reads the user docs; the skill names keys too. Read as a user doc, it names no key that is
        not a settings file's, and no internal one; a key it misspelled would be named.
        """

        def as_user_doc(texts):
            return [t._replace(kind=doc_lints.USER) for t in texts]

        known = doc_lints.load_keys()
        found = as_user_doc(doc_lints.texts(ROOT, SKILL))
        self.assertTrue(found)
        self.assertEqual([str(p) for p in doc_lints.keys(found, known)], [])
        wrong = as_user_doc(
            doc_lints.read(SKILL, "The container's name (`container.nmae`).\n")
        )
        self.assertEqual(
            [p.what for p in doc_lints.keys(wrong, known)],
            [
                "container.nmae: no settings file has this key, and no internal key is named so"
            ],
        )


class TestWhereItIs(unittest.TestCase):
    @READS_THE_SKILL
    def test_the_package_ships_it(self):
        """The build copies package_files.tracked's files, and an unpacked package keeps it: the package's own ignore
        rules leave it in (this machine's git settings aside)."""
        shipped = [f.relative_to(ROOT).as_posix() for f in package_files.tracked(ROOT)]
        self.assertIn(SKILL, shipped)
        with tempfile.TemporaryDirectory() as d:
            unpacked = pathlib.Path(d) / "pkg"
            for rel in (".gitignore", SKILL):
                (unpacked / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(ROOT / rel, unpacked / rel)
            alone = dict(
                GIT_CEILING_DIRECTORIES=d,
                GIT_CONFIG_GLOBAL=os.devnull,
                GIT_CONFIG_NOSYSTEM="1",
            )
            with mock.patch.dict(os.environ, alone):
                kept = package_files.tracked(unpacked)
            self.assertEqual(
                [f.relative_to(unpacked).as_posix() for f in kept],
                sorted([".gitignore", SKILL]),
            )

    @READS_THE_SKILL
    def test_a_repository_that_leaves_out_claude_folders_tracks_it(self):
        """In a repository whose own ignore file leaves out every .claude/ folder, as the one the package lives in may,
        the package's .gitignore lets in again each skill the package ships (generated.claude_skills): git tracks the
        skill, and none of the rest of .claude/: a skill of the user's own (OWN), Claude Code's local files (LOCAL),
        or a file in the skill's folder that a secret's pattern names."""
        secret = next(
            p for p in secrets_registry.ignore_patterns() if not set(p) & set("*/")
        )
        skills = pathlib.PurePosixPath(SKILL).parent
        self.assertIn(skills.name, generated.claude_skills())
        self.assertNotIn(
            pathlib.PurePosixPath(OWN).parent.name, generated.claude_skills()
        )
        with tempfile.TemporaryDirectory() as d:
            repo = pathlib.Path(d) / "repo"
            pkg = repo / "tools" / "pkg"
            for rel, body in (
                (".gitignore", (ROOT / ".gitignore").read_text(encoding="utf-8")),
                (SKILL, text()),
                (OWN, "x\n"),
                (LOCAL, "{}\n"),
                (f"{skills}/{secret}", "x\n"),
            ):
                (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
                (pkg / rel).write_text(body, encoding="utf-8")
            (repo / ".gitignore").write_text(".claude/\n", encoding="utf-8")
            alone = dict(
                GIT_CEILING_DIRECTORIES=d,
                GIT_CONFIG_GLOBAL=os.devnull,
                GIT_CONFIG_NOSYSTEM="1",
            )
            with mock.patch.dict(os.environ, alone):
                try:
                    subprocess.run(
                        ["git", "init", "-q", str(repo)],
                        check=True,
                        capture_output=True,
                    )
                except (OSError, subprocess.CalledProcessError):
                    self.skipTest("no git")
                subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
                tracked = subprocess.run(
                    ["git", "-C", str(repo), "ls-files"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout.split()
        self.assertEqual(
            sorted(tracked),
            sorted([".gitignore", "tools/pkg/.gitignore", f"tools/pkg/{SKILL}"]),
        )

    def test_the_image_leaves_it_out(self):
        """docker/Dockerfile copies the build context whole: no part of .claude/ is in it, the skill or Claude Code's
        local files (LOCAL)."""
        self.assertIn(pathlib.PurePosixPath(SKILL).parts[0], image.PACKAGE_ONLY)
        for path in (SKILL, LOCAL):
            self.assertTrue(
                build_context.ignored(build_context.patterns(ROOT), path), path
            )

    def test_in_the_image_alone_the_tests_that_read_it_skip(self):
        """build_context.left_out_here: where the image's build record is, the tests that read the skill skip, and say
        why; where it is not, as in a checkout, they run, and a file missing there gives no reason to skip (OWN, which
        no package ships). A file the image keeps gives none either."""
        reason = "the image leaves .claude out (image.PACKAGE_ONLY)"
        with tempfile.TemporaryDirectory() as d:
            record = pathlib.Path(d) / build_plan.RECORD_FILE
            with mock.patch.object(build_plan, "RECORD_PATH", str(record)):
                for rel in (SKILL, OWN, "README.md"):
                    self.assertEqual(build_context.left_out_here(rel), "", rel)
                record.write_text("{}\n", encoding="utf-8")
                self.assertEqual(build_context.left_out_here(SKILL), reason)
                self.assertEqual(build_context.left_out_here("README.md"), "")
                self.assertEqual(
                    build_context.left_out_here("README.md", SKILL), reason
                )

    @READS_THE_SKILL
    def test_the_privacy_check_reads_it_as_a_skill(self):
        self.assertEqual(
            [name for name, inside in privacy.PARTS if inside(pathlib.Path(SKILL))],
            [privacy.S_SKILLS],
        )
        self.assertIn(ROOT / SKILL, privacy._files_under(ROOT))

    @READS_THE_SKILL
    def test_the_docs_lints_read_it(self):
        """As a skill: the words, old-commands, links and placeholders lints read it (tools/doc_lints.py)."""
        listed = (ROOT / "tests" / "doc_lint_files.txt").read_text(encoding="utf-8")
        self.assertIn(SKILL, listed.splitlines())
        self.assertEqual(
            {t.kind for t in doc_lints.texts(ROOT, SKILL)}, {doc_lints.SKILL}
        )


if __name__ == "__main__":
    unittest.main()
