# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The package's documents: every docs/ path the package mentions (its tests' fixtures aside) exists in it, and none
names the operator's old copies; the README covers what an operator needs before a first tick; its "Where things
live" names the places the code uses, and it and the glossary pass the docs' lints; its tables of the wrapper's two
paths name each step and its commands, and each command of rkmon a doc names is one rkmon takes, options included; the
maintainers' sections are docs/maintaining.md's alone, with each section the package cites; and `rk --help` names
every stop level.
"""

import ast
import pathlib
import posixpath
import re
import sys
import tempfile
import unittest
from typing import NamedTuple
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "tools"), str(ROOT), str(ROOT / "tests")]
sys.dont_write_bytecode = True

import build_context  # noqa: E402

DOC_REF = re.compile(r"(?<![\w.-])docs/([A-Za-z0-9_.-]+\.md)\b")
SKIP = {".git", "__pycache__", "dist", "state"}


class Home(NamedTuple):
    """A fact's home: its doc, its heading, and the link to them as the docs write it."""

    doc: str
    heading: str
    link: str


def stop_levels() -> Home:
    """The stop levels' home, as tests/doc_fact_homes.yaml names it (the fact stop_levels), the one place that
    names their heading."""
    import doc_lints

    [fact] = [f for f in doc_lints.load_homes(ROOT) if f.name == "stop_levels"]
    return Home(
        fact.home, fact.section, f"({fact.home}#{doc_lints.slug(fact.section)})"
    )


def package_texts():
    """(path in the package, text) of each text file of the package but its tests: their fixtures name the reviewed
    repository's docs (docs/a.md and the like), not the package's."""
    for f in sorted(ROOT.rglob("*")):
        rel = f.relative_to(ROOT)
        if not f.is_file() or set(rel.parts) & SKIP or rel.parts[0] == "tests":
            continue
        try:
            yield str(rel), f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue


def mentions() -> dict[str, set[str]]:
    """docs/<name>.md -> the package files that mention it."""
    out: dict[str, set[str]] = {}
    for rel, text in package_texts():
        for name in DOC_REF.findall(text):
            out.setdefault(name, set()).add(rel)
    return out


def readme_section(heading: str) -> str:
    """The text of README.md under HEADING, a whole heading line ("### On the host"), to the next heading of its level
    or above."""
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    level = len(heading) - len(heading.lstrip("#"))
    found = re.search(
        rf"^{re.escape(heading)}\n(.*?)(?=^#{{1,{level}}} |\Z)", readme, re.M | re.S
    )
    if found is None:
        raise AssertionError(f"README.md has no heading {heading!r}")
    return found.group(1)


def table(section: str) -> dict[str, str]:
    """The body rows of the tables in SECTION: each row's first cell (a lone `name` without its backticks) -> the
    row's other cells."""
    rows = {}
    lines = section.splitlines() + [""]
    for line, after in zip(lines, lines[1:]):
        if line.startswith("| ") and not after.startswith("|---"):
            first, _, rest = line[2:].partition(" | ")
            rows[re.sub(r"^`([^`]*)`$", r"\1", first)] = rest
    return rows


# An option of a command, as a doc names it once the brackets of an optional part are left out
OPTION = re.compile(r"-{1,2}[a-z][a-z-]*")


def rkmon_commands(text: str) -> list:
    """Each command of rkmon that TEXT, a code span or a line of a code block, names, read as the lints read one
    (wrapping.program_commands): (the command as TEXT has it, the row of rkmon's table its words start with, the
    longest, or None, and the options it names). The program's name with no word of a command's name after it names
    none (`docker/rkmon` alone, a folder `~/rkmon`, a placeholder); the options after `rkmon rk` are rk's, not named.
    """
    from rkmon_lib import commands, steps, wrapping

    out = []
    for start, end in wrapping.program_commands(text):
        words = text[start:end].split()[1:]
        options = []
        while words and OPTION.fullmatch(words[0]):
            options.append(words.pop(0))
        if not words or not steps.NAME_WORD.fullmatch(words[0]):
            continue
        found = [r for r in commands.TABLE if tuple(words[: len(r.words)]) == r.words]
        row = max(found, key=lambda r: len(r.words), default=None)
        if row is None or not any(a.rest for a in row.args):
            named = (w.strip("[.]").split("=", 1)[0] for w in words)
            options += [w for w in named if OPTION.fullmatch(w)]
        out.append((text[start:end], row, options))
    return out


def rk_parsers() -> dict:
    """rk's commands, each with its parser (argparse's), as rkmon.main builds them."""
    import argparse

    import rkmon

    built = []

    def parse(parser, *args, **kwargs):
        built.append(parser)
        raise SystemExit(0)

    with mock.patch.object(argparse.ArgumentParser, "parse_args", parse):
        try:
            rkmon.main(["status"])
        except SystemExit:
            pass
    return next(
        a for a in built[0]._actions if isinstance(a, argparse._SubParsersAction)
    ).choices


def rk_help_entries() -> dict[str, str]:
    """rk --help's entry of each command (rkmon.py's docstring): the line that names it, its name first after two
    spaces or after the " | " of a line that names two (review-skip N | review-unskip N), and the lines indented under
    that line."""
    import rkmon

    entries, names = {}, []
    for line in rkmon.__doc__.splitlines():
        if re.match(r"  \S", line):
            names = [
                part.split()[0] for part in line[2:].split("   ", 1)[0].split(" | ")
            ]
        elif not line.startswith("   "):
            names = []
        for name in names:
            entries[name] = entries.get(name, "") + line + "\n"
    return entries


def container_values() -> dict:
    """The derived keys' values in the package's container, with no value given: each function of
    rkmon_lib/derived.py's FUNCTIONS, in their order, after the keys it reads."""
    from rkmon_lib import derived

    layout, values = derived.Layout(container=True), {}
    for key, function in derived.FUNCTIONS.items():
        values[key] = function(values, layout)
    return values


def image_env() -> dict:
    """The environment docker/Dockerfile's ENV instructions give the image: name -> value."""
    text = (ROOT / "docker" / "Dockerfile").read_text(encoding="utf-8")
    return {
        name: value
        for line in text.replace("\\\n", " ").splitlines()
        if line.startswith("ENV ")
        for name, value in re.findall(r"(\w+)=(\S+)", line)
    }


class TestDocs(unittest.TestCase):
    def test_every_docs_path_the_package_mentions_exists(self):
        found = mentions()
        self.assertIn("restart.md", found)
        self.assertIn("review_pipeline.md", found)
        missing = {
            name: sorted(where)
            for name, where in found.items()
            if not (ROOT / "docs" / name).is_file()
        }
        self.assertEqual(missing, {}, "docs the package mentions but does not have")
        for gone in (
            "restart_runbook.md",
            "handoff_guide.md",
        ):  # the operator's own copies, which the package does not ship
            self.assertNotIn(gone, found)

    def test_the_package_ships_its_own_runbook_and_pipeline_guide(self):
        for name in ("restart.md", "review_pipeline.md"):
            text = (ROOT / "docs" / name).read_text(encoding="utf-8")
            self.assertTrue(
                text.startswith("<!--\nCopyright (c) Advanced Micro Devices"), name
            )
        guide = (ROOT / "docs" / "review_pipeline.md").read_text(encoding="utf-8")
        for words in (
            "first review pass",
            "orchestrator model",
            "### 3.3 Parallel reviews",
            "### 3.5 Local checks: the parallel builds",
            "| Review driver |",
            "| Scaler |",
            "Glossary",
            "docker/rkmon",
        ):
            self.assertIn(
                words, guide
            )  # each a word of its own: not a part of another, as "pipe" is of "pipeline"

    def test_the_pipeline_guide_quotes_the_first_pass_prompts_as_the_worker_sends_them(
        self,
    ):
        """docs/review_pipeline.md 5.4 quotes worker/review_agent.py's prompts word for word, in their order: the system
        prompt (up to the review book), the user prompt (up to the staged context, the repository and the PR's number
        as <ROCKE_REPO> and <N>) and the forced final turn, each line of a prompt a quoted paragraph.
        """
        section = (
            (ROOT / "docs" / "review_pipeline.md")
            .read_text(encoding="utf-8")
            .split("\n### 5.4 ", 1)[1]
            .split("\n### ", 1)[0]
        )
        quoted, paragraph = [], []
        for line in section.splitlines() + [""]:
            if line.startswith("> "):
                paragraph.append(line[2:])
            elif paragraph:
                quoted.append(" ".join(paragraph))
                paragraph = []
        tree = ast.parse(
            (ROOT / "worker" / "review_agent.py").read_text(encoding="utf-8")
        )
        named = {"REPO": "<ROCKE_REPO>", "N": "<N>"}

        def words(node):
            """A prompt's own text: what is added after it (the book, the context) left out."""
            if isinstance(node, ast.BinOp):
                return words(node.left)
            if isinstance(node, ast.JoinedStr):
                return "".join(
                    (
                        named[part.value.id]
                        if isinstance(part, ast.FormattedValue)
                        else part.value
                    )
                    for part in node.values
                )
            return node.value

        prompts = [
            words(n.value)
            for n in tree.body
            if isinstance(n, ast.Assign)
            and getattr(n.targets[0], "id", "") in ("system", "user")
        ] + [
            v.value
            for n in ast.walk(tree)
            if isinstance(n, ast.Dict)
            for v in n.values
            if isinstance(v, ast.Constant)
            and str(v.value).startswith("Stop investigating.")
        ]  # the forced final turn's message
        sent = [
            line
            for prompt in prompts
            for line in prompt.split("\n")
            if line and not line.startswith("=====")
        ]
        self.assertEqual(len(prompts), 3, prompts)
        self.assertEqual(quoted, sent)

    def test_the_readme_covers_what_an_operator_needs(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        for heading, inside in (
            (
                "## Quick start: to a first tick in prototype",
                (
                    "QUICK_BUILD_GUIDE.md",
                    "QUICK_DEPLOY_GUIDE.md",
                    "`docker/rkmon template deploy`",
                    "`docker/rkmon first-tick`",
                    "`docker/rkmon secret`",
                    "/rkmon-setup",
                    "docker/rkmon doctor",
                ),
            ),
            (
                "## Features",
                (
                    "features.review_automation",
                    "features.merge_queue",
                    "prototype",
                    "live",
                    "rk features",
                ),
            ),
            (
                "## The setup summary",
                ("rk features", "rkmon rk features", 'set" or "missing'),
            ),
            (
                "## Settings files",
                (
                    "No secret, key or login",
                    "rkmon confirm --settings FILE",
                    "tools/translate_production.py",
                    "(#your-private-files)",
                ),
            ),
            (
                "## The dashboards",
                (
                    "rkmon secret admin_access_token --generate",
                    "admin_access_token",
                    "HTTP Basic",
                    "features.dashboards.team.delivery",
                    "(#ports-volumes-and-dockerenv)",
                    "Tickets waiting for you",
                    "rk ticket-draft FILE",
                    "rk ticket-file ID",
                    "rk ticket-decline ID",
                    "One action per proposed ticket",
                    "Not sent",
                ),
            ),
            (
                "## Stopping the monitor",
                (
                    stop_levels().link,
                    "rkmon pause",
                    "rkmon stop",
                    "rkmon start",
                ),
            ),
            (
                "## The two model roles",
                (
                    "reviewer model",
                    "orchestrator model",
                    "model_bindings.worker",
                    "OpenAI-compatible",
                ),
            ),
            (
                "## Parallel reviews and parallel builds",
                (
                    "features.review_automation.parallel_reviews",
                    "monitor.parallel_builds",
                ),
            ),
        ):
            with self.subTest(heading=heading):
                self.assertIn(heading, readme)
                section = readme.split(heading, 1)[1].split("\n## ", 1)[0]
                for words in inside:
                    self.assertIn(words, section)

    def test_each_rkmon_command_a_doc_names_is_one_rkmon_takes(self):
        """Each command of rkmon that a doc of the package names, in a code span or a code block, is a command of
        rkmon's table (rkmon_lib/commands.py), with only the options that command or the program takes: a doc that
        names a command or an option rkmon does not have fails here. What a doc says a command does is the doc's to
        keep right; no test reads it."""
        import doc_lints
        from rkmon_lib import commands

        program = {
            f for o in commands.GLOBAL_OPTIONS + commands.LEADING for f in o.flags
        }
        wrong = []
        for rel in doc_lints.package_files(ROOT):
            if not rel.endswith(".md"):
                continue
            fenced = False
            lines = (ROOT / rel).read_text(encoding="utf-8").splitlines()
            for n, line in enumerate(lines, 1):
                if line.lstrip().startswith("```"):
                    fenced = not fenced
                    continue
                for text in [line] if fenced else re.findall(r"`([^`]+)`", line):
                    for shown, row, options in rkmon_commands(text):
                        if row is None:
                            wrong.append(
                                f"{rel}:{n}: {shown}: rkmon has no such command"
                            )
                            continue
                        takes = program | {f for o in row.options for f in o.flags}
                        wrong += [
                            f"{rel}:{n}: {shown}: rkmon {row.name} takes no {o}"
                            for o in options
                            if o not in takes
                        ]
        self.assertEqual(wrong, [])

    def test_the_stop_levels_name_each_command(self):
        """The stop levels, at their home (`stop_levels`: docs/restart.md), name each command that stops the monitor
        or brings it back: README's "Stopping the monitor" links there."""
        home = stop_levels()
        restart = (ROOT / home.doc).read_text(encoding="utf-8")
        heading = f"## {home.heading}"
        self.assertIn(heading, restart)
        section = restart.split(heading, 1)[1].split("\n## ", 1)[0]
        for words in (
            "rk pause",
            "rk stop",
            "rk kill",
            "rk start",
        ):
            self.assertIn(words, section)

    def test_rk_help_names_every_level(self):
        import rkmon

        for words in (
            "pause | resume",
            "stop [--drain]",
            "start ",
            "kill [--wait S]",
            "/rocke-monitor off",
        ):
            self.assertIn(words, rkmon.__doc__)

    def test_rk_help_has_a_line_for_each_command(self):
        """rk --help (rkmon.py's docstring) has a line for each command of rk's parser: the command's name first, after
        two spaces, or after the " | " of a line that names two (review-skip N | review-unskip N).
        """
        commands = rk_parsers()
        self.assertGreater(len(commands), 100)
        self.assertEqual(sorted(set(commands) - set(rk_help_entries())), [])

    def test_rk_help_names_each_option_of_each_command(self):
        """rk --help's entry of each command (rk_help_entries) names each option of the command's parser, by one of
        its spellings: an option the entry leaves out is one an operator never learns of from the help.
        """
        import argparse

        entries, missing = rk_help_entries(), {}
        for name, parser in sorted(rk_parsers().items()):
            for action in parser._actions:
                if isinstance(action, argparse._HelpAction):
                    continue
                spellings = action.option_strings
                if spellings and not any(
                    re.search(
                        rf"(?<![\w-]){re.escape(o)}(?![\w-])", entries.get(name, "")
                    )
                    for o in spellings
                ):
                    missing.setdefault(name, []).append("/".join(spellings))
        self.assertEqual(missing, {})

    def test_rk_help_names_each_key_by_its_name(self):
        """rk's help (its usage and its options' help) names a settings file's key (features.…) as the file names it,
        and no configuration key (`config NAME`) by a name the rename table retired."""
        import doc_lints
        from rkmon_lib import renames

        source = (ROOT / "rkmon.py").read_text(encoding="utf-8")
        text = "\n".join(
            n.value
            for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.Constant) and isinstance(n.value, str)
        )
        settings = doc_lints.load_keys().settings
        self.assertEqual(
            [
                m.group()
                for m in re.finditer(r"\bfeatures(?:\.[a-z_]+)+", text)
                if tuple(m.group().split(".")) not in settings
            ],
            [],
        )
        self.assertEqual(
            [
                old
                for old in renames.RENAMED
                if re.search(rf"\bconfig:? {re.escape(old)}\b", text)
            ],
            [],
        )

    def test_the_codes_messages_name_each_key_by_its_name(self):
        """Each message of the code (rk's help, the checks, setup's words, the placeholders' descriptions) names a
        settings file's key as the file names it (the keys lint): no key the settings files lack. The translation of
        a layout 1 configuration is left out: it names that layout's own keys (rkmon_lib/from_layout1.py,
        tools/translate_production.py)."""
        import doc_lints
        from rkmon_lib import package_files

        layout1 = {"rkmon_lib/from_layout1.py", "tools/translate_production.py"}
        files = []
        for path in package_files.tracked(ROOT):
            rel = (path.relative_to(ROOT) if path.is_absolute() else path).as_posix()
            if (
                rel.endswith(".py")
                and not rel.startswith("tests/")
                and rel not in layout1
            ):
                files.append(rel)
        self.assertIn("rkmon_lib/agentic.py", files)
        self.assertEqual(
            [str(p) for p in doc_lints.run(ROOT, sorted(files), ("keys",))], []
        )


class TestQuickGuide(unittest.TestCase):
    """QUICK_USER_GUIDE.md: the operator's guide at the package's top, which `rk guide` prints and the image carries."""

    def test_rk_guide_prints_the_guide_of_the_code_that_runs(self):
        import io
        import rkmon

        out = io.StringIO()
        self.assertEqual(rkmon.cmd_guide(ROOT, out), 0)
        self.assertEqual(
            out.getvalue(), (ROOT / "QUICK_USER_GUIDE.md").read_text(encoding="utf-8")
        )

    def test_an_install_without_the_guide_says_so(self):
        import io
        import tempfile
        import rkmon

        out = io.StringIO()
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(rkmon.cmd_guide(pathlib.Path(d), out), 1)
        self.assertIn("no guide in this install", out.getvalue())

    def test_the_guide_covers_what_an_operator_needs(self):
        # The operator's sections, in order, and nothing else.
        guide = (ROOT / "QUICK_USER_GUIDE.md").read_text(encoding="utf-8")
        self.assertTrue(guide.startswith("<!--\nCopyright (c) Advanced Micro Devices"))
        self.assertEqual(
            [line[3:] for line in guide.splitlines() if line.startswith("## ")],
            [
                "Your first day",
                "Check on it",
                "Change a setting",
                "Turn on a feature",
                "Go live",
                "Standby",
                "Pause, stop and restart",
                "Move to a new version",
                "Use another model for a pass",
                "Scale, and the RAM disk",
                "Logs, state and the outputs folder",
                "Common problems",
                "Remove a deployment",
            ],
        )
        for words in (
            "./rkmon rk guide",
            "./rkmon status",
            "./rkmon doctor",
            "./rkmon rk features",
            "./rkmon template section KEY",
            "./rkmon login github_write",
            "./rkmon rk sanity SHA --rehearsal",
            "./rkmon image load",
            "./rkmon confirm",
            "./rkmon apply",
            "./rkmon rollback",
            "./rkmon pause",
            "./rkmon rk review-skip N",
            "./rkmon remove --delete-data",
            stop_levels().link,
            "(README.md#where-things-live)",
        ):
            self.assertIn(words, guide)
        self.assertLessEqual(
            len(guide.splitlines()), 260, "the quick guide stays short"
        )

    def test_your_first_day_says_what_the_first_ticks_queue(self):
        """ "Your first day" says what the code does with the pull requests a deployment finds at its start: the search
        for those in scope has no date bound, so the first ticks queue every open one, however old, that is not a
        draft and has no review yet from this deployment, a review from its account whose first line gives its
        automated reviewer name (a fresh deployment's own reviews: none with no name, none of another deployment's
        under the same account); develop fixes and the operator's priorities first, then the others by number, lowest
        first. A change to the search, to which review is this deployment's, to which pull request waits for a first
        review, or to their order fails here, beside the guide's sentence it would make wrong.
        """
        from rkmon_lib import plan, queue, reviewer_name, snapshot, store

        guide = " ".join(
            (ROOT / "QUICK_USER_GUIDE.md").read_text(encoding="utf-8").split()
        )
        for words in (
            "The first ticks queue every open pull request in scope",
            "however old, that is not a draft and has no review yet from this deployment",
            "develop fixes and the pull requests you float to the top",
            "come first, then the others by number, lowest first",
            "This deployment's reviews are those from your GitHub account (`integrations.github_cli.username`) whose "
            "first line gives its automated reviewer name (`features.review_automation.automated_reviewer_name`)",
            "Do not post one by hand: the monitor records only what it posts",
        ):
            self.assertIn(words, guide)
        self.assertEqual(
            snapshot.label_query({"repo": "o/r"}, ["in-scope", "also"]),
            'repo:o/r is:pr is:open label:"in-scope","also"',
        )
        old = "2020-01-02T03:04:05Z"

        def pr(n, **kw):
            rec = {
                "n": n,
                "head": f"{n:040d}",
                "size": 100,
                "status": "open",
                "draft": False,
            }
            rec.update(
                author=f"dev{n}", ours=None, head_seen_at=old, head_committed_at=old
            )
            return dict(rec, **kw)

        st = store.empty_state()
        st["develop"]["sha"] = "d" * 40
        st["prs"] = {
            str(r["n"]): r
            for r in (
                pr(40, size=5),
                pr(10, size=900),
                pr(30, size=50),
                pr(20, size=10),
                pr(50, draft=True),
                pr(60, status="closed"),
                pr(70, ours={"head": f"{70:040d}", "at": old}),
            )
        }
        self.assertEqual(plan.review_kind(st["prs"]["10"]), "FIRST_ROUND")
        for n in ("50", "60", "70"):
            self.assertIsNone(plan.review_kind(st["prs"][n]), n)
        st["user_priority"] = {"40": {"why": "needed today"}}
        queue.refresh_develop_fixes(
            st, [{"status": "open", "fixed_by": 30, "id": "i1"}]
        )
        self.assertEqual(
            [
                i["pr"]
                for i in plan.make_plan({"settle_min": 30, "repo": "o/r"}, st, [], [])
                if i["kind"] == "FIRST_ROUND"
            ],
            [30, 40, 10, 20],
        )
        # This deployment's review is one from its account (snapshot.our_latest_review reads the account's posts
        # alone) whose line 1 gives its automated reviewer name; for a fresh deployment, another deployment's review
        # under the same account, and one whose line 1 gives no name, are not.
        cfg = {
            "operator": {"github_login": "op"},
            "features": {
                "review_automation": {"automated_reviewer_name": "Platform team"}
            },
        }
        line1 = "Automated review by {}: Model - m, effort: max"
        self.assertTrue(
            reviewer_name.own_review(cfg, line1.format("Platform team"), old)
        )
        self.assertFalse(reviewer_name.own_review(cfg, line1.format("Other team"), old))
        self.assertFalse(
            reviewer_name.own_review(
                cfg, "Automated review: Model - m, effort: max", old
            )
        )

    def test_the_readme_points_at_it_and_the_image_carries_both(self):
        top = "\n".join(
            (ROOT / "README.md").read_text(encoding="utf-8").splitlines()[:30]
        )
        self.assertIn("[QUICK_USER_GUIDE.md](QUICK_USER_GUIDE.md)", top)
        self.assertIn("[QUICK_BUILD_GUIDE.md](QUICK_BUILD_GUIDE.md)", top)
        self.assertIn("[QUICK_DEPLOY_GUIDE.md](QUICK_DEPLOY_GUIDE.md)", top)
        user = (ROOT / "QUICK_USER_GUIDE.md").read_text(encoding="utf-8")
        self.assertIn("[QUICK_BUILD_GUIDE.md](QUICK_BUILD_GUIDE.md)", user)
        self.assertIn("[QUICK_DEPLOY_GUIDE.md](QUICK_DEPLOY_GUIDE.md)", user)
        deploy = (ROOT / "QUICK_DEPLOY_GUIDE.md").read_text(encoding="utf-8")
        self.assertIn("[QUICK_USER_GUIDE.md](QUICK_USER_GUIDE.md)", deploy)
        self.assertIn("[README.md](README.md)", deploy)
        build = (ROOT / "QUICK_BUILD_GUIDE.md").read_text(encoding="utf-8")
        # A guide names the doc before it and the one after it, no other; another doc's part is linked
        # as that part.
        self.assertEqual(
            sorted(set(re.findall(r"\]\(([\w./-]+\.md)\)", build))),
            ["QUICK_DEPLOY_GUIDE.md", "README.md"],
        )
        self.assertIn(
            '[QUICK_USER_GUIDE.md, "Move to a new version"](QUICK_USER_GUIDE.md#move-to-a-new-version)',
            build,
        )
        self.assertIn(
            "[QUICK_BUILD_GUIDE.md](QUICK_BUILD_GUIDE.md)",
            (ROOT / "QUICK_DEPLOY_GUIDE.md").read_text(),
        )
        patterns = build_context.patterns(ROOT)
        for name in (
            "QUICK_USER_GUIDE.md",
            "QUICK_BUILD_GUIDE.md",
            "QUICK_DEPLOY_GUIDE.md",
            "README.md",
        ):
            self.assertFalse(build_context.ignored(patterns, name), name)

    def test_rk_guide_build_prints_the_build_guide(self):
        import io
        import rkmon

        for which, name in (
            ("build", "QUICK_BUILD_GUIDE.md"),
            ("deploy", "QUICK_DEPLOY_GUIDE.md"),
        ):
            out = io.StringIO()
            self.assertEqual(rkmon.cmd_guide(ROOT, out, which=which), 0)
            self.assertEqual(out.getvalue(), (ROOT / name).read_text(encoding="utf-8"))

    def test_the_build_guide_links_to_the_build_settings_and_the_labels(self):
        """The build settings and the image's labels have one home, README.md; the build guide links to
        them, and carries no copy of the template or its keys."""
        build = (ROOT / "QUICK_BUILD_GUIDE.md").read_text(encoding="utf-8")
        self.assertTrue(build.startswith("<!--\nCopyright (c) Advanced Micro Devices"))
        self.assertIn('[README.md, "Build settings"](README.md#build-settings)', build)
        self.assertIn('[README.md, "Image labels"](README.md#image-labels)', build)
        self.assertNotIn("```yaml", build)
        self.assertNotIn("schema_version", build)

    def test_the_image_labels_section_names_each_label(self):
        """README's "Image labels" is the labels' one home: it names each label `rkmon build` gives an
        image (build_plan.label_names), and no other."""
        from rkmon_lib import build_plan

        section = readme_section("## Image labels")
        named = set(
            re.findall(r"`(org\.(?:opencontainers|rocke)\.[a-z0-9_.]+)`", section)
        )
        self.assertEqual(named, set(build_plan.label_names()))

    def test_the_build_guide_has_its_sections_in_order_and_its_six_rows(self):
        """The build guide's sections, in order, a step's named as the step table names it; its six
        rows of what to have ready, each with where it is checked."""
        from rkmon_lib import steps

        build = (ROOT / "QUICK_BUILD_GUIDE.md").read_text(encoding="utf-8")
        ready = "Before you start: have these ready"
        self.assertEqual(
            re.findall(r"^## (.+)$", build, re.M),
            [ready]
            + [f"Step {s.n}: {s.name}" for s in steps.BUILD_STEPS]
            + [
                "Behind a network that inspects TLS",
                "One machine for both jobs",
                "Rebuilding the same package, with a tag of its own",
                "Next: QUICK_DEPLOY_GUIDE.md",
            ],
        )
        section = build.split(f"## {ready}\n", 1)[1].split("\n## ", 1)[0]
        rows = [
            [cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in section.splitlines()
            if re.match(r"\| \d+ \|", line)
        ]
        sizing = '[QUICK_DEPLOY_GUIDE.md, "Machine sizing"](QUICK_DEPLOY_GUIDE.md#machine-sizing)'
        self.assertEqual(
            [(r[0], r[3]) for r in rows],
            [
                ("1", "step 2"),
                ("2", "step 2"),
                ("3", "step 2 (with your own `rocm.package_url`: step 3)"),
                ("4", "step 2"),
                ("5", "step 2"),
                ("6", "step 2; the registry and its login: step 5"),
            ],
        )
        for row, have in zip(
            rows,
            (
                "The package, the version the deploy machine will run",
                f"Docker and buildx, and the Python and free disk that {sizing} states",
                "The ROCm release to install (`rocm.version`)",
                "The atlassian tool's Linux binary for x86-64",
                "its root certificate, as a PEM file",
                "a registry to push to",
            ),
        ):
            with self.subTest(row=row[0]):
                self.assertIn(have, row[1])

    def test_the_deploy_guide_links_to_the_deploy_settings_and_carries_no_copy(self):
        """The deploy settings and the model bindings have one home, README.md; the deploy guide links to
        them, and carries no copy of a template, its keys or the build's settings; and it names, as a whole doc, only
        the doc before it, the one after it and the reference."""
        deploy = (ROOT / "QUICK_DEPLOY_GUIDE.md").read_text(encoding="utf-8")
        self.assertTrue(deploy.startswith("<!--\nCopyright (c) Advanced Micro Devices"))
        self.assertIn(
            '[README.md, "Deploy settings"](README.md#deploy-settings)', deploy
        )
        self.assertIn('[README.md, "Model bindings"](README.md#model-bindings)', deploy)
        self.assertNotIn("```yaml", deploy)
        self.assertNotIn("schema_version", deploy)
        for key in (
            "rocm.package_url",
            "atlassian_cli.package_url",
        ):  # the image's settings are the build's only
            self.assertNotIn(key, deploy)
        self.assertEqual(
            sorted(set(re.findall(r"\]\(([\w./-]+\.md)\)", deploy))),
            ["QUICK_BUILD_GUIDE.md", "QUICK_USER_GUIDE.md", "README.md"],
        )

    def test_the_deploy_guide_has_its_sections_in_order_and_its_ten_rows(self):
        """The deploy guide's sections, in order, a step's named as the step table names it, and the four
        parts of "Before you start"; its ten rows of what to have ready, each with where it is checked.
        """
        from rkmon_lib import steps

        deploy = (ROOT / "QUICK_DEPLOY_GUIDE.md").read_text(encoding="utf-8")
        before = "Before you start"
        self.assertEqual(
            re.findall(r"^## (.+)$", deploy, re.M),
            [before, "With Claude Code", "The keys you must fill in"]
            + [f"Step {s.n}: {s.name}" for s in steps.DEPLOY_STEPS]
            + [
                "Several deployments",
                "Moving from an existing monitor",
                "Removing a deployment",
                "Next: QUICK_USER_GUIDE.md",
            ],
        )
        part = deploy.split(f"\n## {before}\n", 1)[1].split("\n## ", 1)[0]
        ready = "Have these ready"
        self.assertEqual(
            re.findall(r"^### (.+)$", part, re.M),
            ["Machine sizing", "The package and the image", "Pick a profile", ready],
        )
        section = part.split(f"\n### {ready}\n", 1)[1]
        rows = [
            [cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in section.splitlines()
            if re.match(r"\| \d+ \|", line)
        ]
        self.assertEqual(
            [(r[0], r[3]) for r in rows],
            [
                ("1", "step 4"),
                ("2", "step 4"),
                ("3", "step 2"),
                ("4", "step 2"),
                ("5", "step 5"),
                ("6", "steps 5 and 7"),
                ("7", "steps 2 and 5"),
                ("8", "step 5"),
                ("9", "steps 2 and 5"),
                ("10", "step 2"),
            ],
        )
        for row, have in zip(
            rows,
            (
                "The image: its file, or the registry it was pushed to",
                "The package, the same version as the image",
                "whether this one runs the merge queue (one per repository), an automated reviewer name of its own, "
                'and a GitHub account of its own ("Several deployments")',
                "The repository, `<org>/<repo>`, which must be public; the folder your project lives in there, "
                "from the repository's root (`repository.folder`, such as `projects/my-project`); and your team's "
                "scope labels",
                "two fine-grained tokens that `<org>` owns, made as [Make the GitHub tokens](#make-the-github-tokens) "
                "says",
                "an Anthropic API key, or your gateway's address, a token for it, and the header",
                "With tickets: the JIRA site's address, the project's key",
                "a Teams Workflows webhook for the channel",
                "a web host you can SSH to, a user there, the folder it serves, that user's SSH key, and the "
                "fingerprints of the web host's own keys",
                "The share of this host to give: CPUs, memory, a RAM disk",
            ),
        ):
            with self.subTest(row=row[0]):
                self.assertIn(have, row[1])
        self.assertIn(
            '[README.md, "The team dashboard"](README.md#the-team-dashboard)',
            rows[8][2],
        )
        self.assertIn("(`./rkmon secret admin_access_token --generate`)", section)

    def test_the_deploy_guide_names_rkmon_as_its_reader_runs_it(self):
        """The deploy guide's commands run from the package's docker/ folder as ./rkmon, and rkmon names itself in its
        shown lines as it was typed (rkmon_lib/program.py): so each command and each shown line the guide writes
        names it ./rkmon. Its generated sections name it rkmon, as their sources do."""
        from rkmon_lib import program

        deploy = (ROOT / "QUICK_DEPLOY_GUIDE.md").read_text(encoding="utf-8")
        own = re.sub(
            r"(?ms)^<!-- BEGIN GENERATED: .*?^<!-- END GENERATED: \S+ -->$", "", deploy
        )
        texts = re.findall(r"`([^`\n]+)`", own) + re.findall(
            r"(?ms)^```\w*\n(.*?)^```$", own
        )
        self.assertEqual(
            [t for t in texts if re.search(r"(?<![./\w-])rkmon (?=[a-z])", t)], []
        )
        self.assertIn(
            "`next: step 2, check: ./rkmon check --settings ~/rkmon-deploy.yaml`", own
        )
        self.assertEqual(program.name("./rkmon"), "./rkmon")

    def test_the_deploy_guide_s_generated_sections_are_in_their_places(self):
        """The keys to fill in, in "The keys you must fill in", and the logins with the GitHub tokens'
        permissions, in step 5, each between its markers (tests/test_generated.py keeps their text the generator's).
        """
        deploy = (ROOT / "QUICK_DEPLOY_GUIDE.md").read_text(encoding="utf-8")
        for heading, name in (
            ("The keys you must fill in", "required-keys"),
            ("Step 5: login", "logins"),
        ):
            with self.subTest(section=name):
                section = deploy.split(f"\n## {heading}\n", 1)[1].split("\n## ", 1)[0]
                self.assertIn(f"<!-- BEGIN GENERATED: {name}\n", section)
                self.assertIn(f"<!-- END GENERATED: {name} -->", section)

    def test_the_deploy_guide_makes_each_github_token_owner_first_with_the_permissions_its_calls_need(
        self,
    ):
        """The section "Make the GitHub tokens" says first that the repository must be public, then gives each field of
        GitHub's page a token needs, the resource owner first, as it cannot be changed once the token is made, and
        `<org>` for both; then each permission the login's calls need (gh.permissions) with the page's words for its
        access, Members among the organization permissions and every other among the repository ones; and what to do
        when a token expires, linked to the FAQ."""
        from rkmon_lib import gh

        deploy = (ROOT / "QUICK_DEPLOY_GUIDE.md").read_text(encoding="utf-8")
        part = deploy.split("\n#### Make the GitHub tokens\n", 1)[1].split("\n#", 1)[0]
        self.assertTrue(
            part.lstrip().startswith(
                "The monitor watches public repositories only, so `<org>/<repo>` must be public"
            )
        )
        rows = [
            [cell.strip() for cell in line.strip().strip("|").split("|")]
            for line in part.splitlines()
            if line.startswith("| ")
        ]
        self.assertEqual(
            [r[0] for r in rows],
            [
                "On the page",
                "Resource owner",
                "Token name",
                "Description",
                "Expiration",
                "Repository access",
                "Repository permissions",
                "Organization permissions",
            ],
        )
        self.assertIn(
            '[docs/FAQ.md, "What if a token or login expires?"](docs/FAQ.md#what-if-a-token-or-login-expires)',
            " ".join(part.split()),
        )
        access = {"Read-only": "read", "Read and write": "write"}
        for column, side in ((1, gh.READ), (2, gh.WRITE)):
            with self.subTest(login=rows[0][column]):
                self.assertTrue(rows[1][column].startswith("`<org>`"), rows[1][column])
                given = {}
                for row, organization in ((rows[6], False), (rows[7], True)):
                    if row[column] == "none":
                        continue
                    for group in row[column].split("; "):
                        names, how = group.split(": ", 1)
                        for name in names.split(", "):
                            self.assertEqual(name == "Members", organization, name)
                            given[name] = access[how.split(" (", 1)[0]]
                self.assertEqual(given, dict(gh.permissions(side)))

    def test_the_deploy_guide_s_sizing_numbers_are_the_code_s(self):
        """The section "Machine sizing" types its numbers, so each is pinned to the code's: the ranges the monitor
        moves the parallel reviews and builds in (scaler.ranges), which are the deploy schema's minimum and maximum of
        the keys that set them; the RAM disk's first estimates for each (deployment.PIPE_GB, HEAVY_GB); what all of
        them need at their most, rounded up, as the RAM disk's WARN counts it (deployment.sizing_checks); the memory
        over the RAM disk (deployment.WORK_GB); and an example that fits them all."""
        import math

        from rkmon_lib import deploy_tree, deployment, scaler

        deploy = (ROOT / "QUICK_DEPLOY_GUIDE.md").read_text(encoding="utf-8")
        part = " ".join(
            deploy.split("\n### Machine sizing\n", 1)[1].split("\n### ", 1)[0].split()
        )
        reviews = deploy_tree.node("features.review_automation.parallel_reviews")[1]
        builds = deploy_tree.node("monitor.parallel_builds")[1]
        self.assertEqual(
            scaler.ranges(),
            {
                "pipes": (reviews["minimum"], reviews["maximum"]),
                "heavy": (builds["minimum"], builds["maximum"]),
            },
        )
        self.assertIn(
            f"the parallel reviews, from {reviews['minimum']} to {reviews['maximum']}, and the parallel builds, "
            f"from {builds['minimum']} to {builds['maximum']},",
            part,
        )
        every = f"all {reviews['maximum']} parallel reviews and {builds['maximum']} parallel builds"
        self.assertEqual(part.count(every), 2)
        self.assertIn(
            f"about {deployment.PIPE_GB:g} GiB for each review and {deployment.HEAVY_GB:g} GiB for each build",
            part,
        )
        need = (
            reviews["maximum"] * deployment.PIPE_GB
            + builds["maximum"] * deployment.HEAVY_GB
        )
        self.assertIn(f"{every} need, about {math.ceil(need)} GiB, is a WARN", part)
        self.assertIn(f"at least `ramdisk_gib` plus {deployment.WORK_GB:g},", part)
        example = re.search(
            r"For example, (\d+) CPUs, a (\d+) GiB RAM disk and (\d+) GiB of memory leave room for "
            + re.escape(every),
            part,
        )
        self.assertIsNotNone(example, part)
        cpus, ramdisk, memory = map(int, example.groups())
        self.assertGreaterEqual(cpus, reviews["maximum"] + builds["maximum"])
        self.assertGreaterEqual(ramdisk, need)
        self.assertGreaterEqual(memory, ramdisk + deployment.WORK_GB)


class TestWhereThingsLive(unittest.TestCase):
    """README.md's "Where things live" and its glossary: each place they name is the one the code uses, and their
    words pass the docs' lints."""

    def test_the_container_folders_are_the_ones_the_code_uses(self):
        from rkmon_lib import (
            build_plan,
            code_volume,
            config,
            container_layout,
            deliver,
            one_time_move,
            releases,
            runtime_config,
            secrets_registry,
            state_store,
            store,
        )

        v = container_values()
        env = image_env()
        rows = table(readme_section("### In the container"))
        entrypoint = (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
        logs = re.search(r"^LOGS=\$\{RKMON_LOGS:-(\S+)\}", entrypoint, re.M).group(1)
        work, home = v["paths.working_dir"], v["paths.toolkit"]
        self.assertEqual(env["RKMON_HOME"], home)
        self.assertEqual(
            set(rows),
            {
                work,
                home,
                logs,
                str(releases.code_dir({})),
                env["CLAUDE_CONFIG_DIR"],
                v["claude_workspace"],
                str(releases.IMAGE_CODE),
                v["paths.secrets_dir"],
                container_layout.WRITE_SECRETS_DIR,
                container_layout.SETTINGS_DIR,
                build_plan.RECORD_PATH,  # the image's build record
                container_layout.ROCM_RECORD,  # the ROCm release the build installed
                v["ramdisk.path"],
            },
        )
        # The older package's code, which the one-time move's end removes; the site lessons, in the state.
        self.assertIn(f"`{one_time_move.OLDER_CODE}/`", rows[work])
        lessons = posixpath.relpath(v["paths.site_lessons"], home)
        self.assertIn(f"`{lessons}`", rows[home])
        # The delivered settings with the team's skills, and those they replaced; the monitor's record of those it
        # runs, and config.json.
        for name in (
            container_layout.LAYER_FILE,
            deliver.LAYER_KEPT,
            deliver.SKILLS_FOLDER + "/",
            deliver.SKILLS_KEPT + "/",
        ):
            self.assertIn(f"`{name}`", rows[container_layout.SETTINGS_DIR], name)
        for name in (runtime_config.APPLIED_FILE, runtime_config.CONFIG_JSON):
            self.assertIn(f"`{name}`", rows[home], name)
        for key in (
            "repo",
            "worker_tools",
            "workers",
            "agentic",
            "scratch",
            "cache",
            "run",
        ):
            folder = posixpath.relpath(v[f"paths.{key}"], work)
            self.assertIn(f"`{folder}/`", rows[work], key)
        with tempfile.TemporaryDirectory() as d:
            kept = store.Store(config.state_dir(d))
            for name in (
                f"{posixpath.relpath(str(config.state_dir(home)), home)}/",
                kept.state_path.name,
                kept.journal_path.name,
                kept.ledger_path.name,
                state_store.STATE_FILE,
            ):
                self.assertIn(f"`{name}`", rows[home], name)
            self.assertTrue((kept.dir / "jobs").is_dir())
        self.assertIn("`jobs/`", rows[home])
        names = sorted(set(re.findall(r'"\$LOGS/([\w-]+\.log)"', entrypoint)))
        self.assertGreater(len(names), 1)
        for name in names:
            self.assertIn(f"`{name}`", rows[logs], name)
        # Each piece the entry point keeps running, by the label its log lines give it, then its log.
        kept = re.findall(r'^keep "([^"]+)" "\$LOGS/([\w-]+\.log)"', entrypoint, re.M)
        self.assertGreater(len(kept), 1)
        for label, name in kept:
            self.assertRegex(rows[logs], rf"{re.escape(label)}[^`]*`{re.escape(name)}`")
        rotated = re.search(r'cp "\$f" "\$f(\.\w+)"', entrypoint).group(1)
        self.assertIn(f"`{rotated}`", rows[logs])
        self.assertIn(f"`{container_layout.SLOTS_DIR}`", rows[v["ramdisk.path"]])
        # What root makes in the code volume (docker/rk_root.py): the releases' folder and the two links.
        for name in (f"{code_volume.RELEASES}/", code_volume.CURRENT, code_volume.KEPT):
            self.assertIn(f"`{name}`", rows[str(releases.code_dir({}))], name)
        # Each login's folder, in the volume of the user who keeps it.
        cfg = {"paths": {"secrets_dir": v["paths.secrets_dir"]}}
        logins = [
            e
            for e in secrets_registry.entries(cfg)
            if e.shape == secrets_registry.LOGIN
        ]
        self.assertGreater(len(logins), 1)
        for entry in logins:
            place = secrets_registry.home(cfg, entry, environ=env)
            self.assertIn(f"`{place}`", rows[posixpath.dirname(place)], entry.name)
        # Each other secret, a file named as it is, in the volume of the user who keeps it, and only there.
        volumes = {v["paths.secrets_dir"], container_layout.WRITE_SECRETS_DIR}
        others = [e for e in secrets_registry.entries(cfg) if e not in logins]
        self.assertGreater(len(others), 1)
        for entry in others:
            folder = posixpath.dirname(secrets_registry.home(cfg, entry, environ=env))
            self.assertIn(folder, volumes, entry.name)
            for volume in volumes:
                named = f"`{entry.name}`" in rows[volume]
                self.assertEqual(named, volume == folder, (entry.name, volume))

    def test_the_outputs_folder_has_the_files_the_code_writes(self):
        from rkmon_lib import paths

        v = container_values()
        out, work = v["paths.outputs_dir"], v["paths.working_dir"]
        section = readme_section("### The outputs folder")
        self.assertIn(f"In the container the outputs folder is `{out}`", section)
        rows = table(section)
        for key in (*paths.UNDER_OUTPUTS, "weekly_dir"):
            value = v[f"paths.{key}"]
            if value != out:  # a file or a folder of its own in the outputs folder
                name = posixpath.relpath(value, out)
                self.assertIn(name + "/" if key.endswith("_dir") else name, rows, key)
        glossary = table(readme_section("## Glossary and placeholders"))
        self.assertIn(f"`{out}` in the container", glossary["Outputs folder"])
        self.assertIn(f"`{work}` in the container", glossary["Working folder"])

    def test_the_private_files_are_the_ones_rkmon_checks(self):
        from rkmon_lib import private_files

        host = readme_section("### On the host")
        section = readme_section("### Your private files")
        for rel in private_files.PACKAGE_FILES:
            self.assertIn(f"`{rel}`", host, rel)
            self.assertIn(f"`{rel}`", section, rel)
        flat = " ".join(section.split())
        clouds = re.search(r"a cloud-synced folder \(([^:)]+)\)", flat).group(1)
        self.assertGreater(len(clouds.split(", ")), 1)
        for name in clouds.split(", "):
            self.assertTrue(private_files.SYNCED.fullmatch(name), name)

    def test_the_settings_a_message_may_carry_are_private(self):
        from rkmon_lib import file_types, settings_file

        section = readme_section("### Where a private value may show")
        bullet = next(
            b for b in section.split("\n- ") if b.startswith("in the messages")
        )
        keys = re.findall(r"`([a-z_]+(?:\.[a-z_]+)+)`", bullet)
        self.assertGreater(len(keys), 1)
        schema = file_types.DEPLOY.load()
        for key in keys:
            node = settings_file.schema_at(schema, key) or {}
            self.assertEqual(node.get("x-class"), "private", key)

    def test_its_sections_pass_the_docs_lints(self):
        import doc_lints

        lines = (ROOT / "README.md").read_text(encoding="utf-8").split("\n")
        spans = []  # each section's lines, the first and the last, counted from 1
        for heading in ("## Where things live", "## Glossary and placeholders"):
            start = lines.index(heading)
            after = (
                i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")
            )
            spans.append((start + 1, next(after, len(lines))))
        found = [
            f"README.md:{p.line}: {p.lint}: {p.what}"
            for p in doc_lints.run(ROOT, ["README.md"])
            if any(first <= p.line <= last for first, last in spans)
        ]
        self.assertEqual(found, [])

    def test_the_glossary_has_the_settings_descriptions_placeholders(self):
        import doc_lints

        files = [
            "schema/deploy_settings.schema.json",
            "schema/build_settings.schema.json",
            "docker/compose.yaml",
        ]
        self.assertEqual(doc_lints.run(ROOT, files, ("placeholders",)), [])


class TestMaintaining(unittest.TestCase):
    """docs/maintaining.md, the maintainers' guide."""

    def test_each_section_the_package_cites_is_there(self):
        guide = (ROOT / "docs" / "maintaining.md").read_text(encoding="utf-8")
        headings = set(re.findall(r"^## (.+)$", guide, re.M))
        cited: dict[str, set[str]] = {}
        for rel, text in package_texts():
            for name in re.findall(r'docs/maintaining\.md, "([^"]+)"', text):
                cited.setdefault(" ".join(name.split()), set()).add(rel)
        self.assertIn("Publishing the package", cited)
        self.assertIn("Building the package and its tarball", cited)
        self.assertEqual(
            {name: sorted(w) for name, w in cited.items() if name not in headings}, {}
        )

    def test_the_package_folder_has_a_row_for_each_entry_of_the_package(self):
        """Each entry at the package's top: those the image takes (image.BUILD_CONTEXT) and those it leaves out
        (image.PACKAGE_ONLY), which tests/test_entrypoint.py checks against the package's own.
        """
        from rkmon_lib import image

        guide = (ROOT / "docs" / "maintaining.md").read_text(encoding="utf-8")
        part = guide.split("\n## The package folder\n", 1)[1].split("\n## ", 1)[0]
        named = {
            name.rstrip("/")
            for cell in re.findall(r"^\| ([^|]+) \|", part, re.M)
            for name in re.findall(r"`([^`]+)`", cell)
        }
        self.assertEqual(
            sorted({*image.BUILD_CONTEXT, *image.PACKAGE_ONLY} - named), []
        )

    def test_its_sections_are_its_alone(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        guide = (ROOT / "docs" / "maintaining.md").read_text(encoding="utf-8")
        headings = re.findall(r"^## (.+)$", guide, re.M)
        self.assertIn("Before you push a change", headings)
        for heading in headings:
            self.assertNotRegex(readme, rf"(?m)^#+ {re.escape(heading)}", heading)
        self.assertIn("docs/maintaining.md", readme_section("## Documentation"))


if __name__ == "__main__":
    unittest.main()
