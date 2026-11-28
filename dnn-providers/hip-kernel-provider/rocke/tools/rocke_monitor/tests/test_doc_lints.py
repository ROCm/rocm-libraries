# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The docs' lints (tools/doc_lints.py): what they read of a file; each lint on a sample that
passes and one that fails, whose problem names the home, key, word, command, quote, link or placeholder; what the lists
must give; the command line; and over the package: the files tests/doc_lint_files.txt lists pass every lint (a test the
image skips, as it leaves the setup skill out) and every Markdown file is one of them, every doc quotes only what the
code prints, no file of the package has an earlier name of a renamed thing, the lists have the facts written once and
the old commands, and each refusal's fix and next lines are the state table's.
"""

import contextlib
import io
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(ROOT / "tools"), str(ROOT), str(HERE)]
sys.dont_write_bytecode = True

import build_context  # noqa: E402
import doc_lints as dl  # noqa: E402
import test_states  # noqa: E402  (its facts for each state; its own tests run in their module)
from rkmon_lib import commands, outcome, registry, states, wrapping  # noqa: E402

FILES = HERE / "doc_lint_files.txt"
USER, MAINTAINER, SKILL, MESSAGE = dl.USER, dl.MAINTAINER, dl.SKILL, dl.MESSAGE
# The facts each written once, and the network the image test's stages run on, which its help says.
FACTS = {
    "change_a_setting",
    "turn_on_a_feature",
    "go_live",
    "first_day",
    "standby",
    "stop_levels",
    "machine_sizing",
    "what_goes_out",
    "admin_dashboard_tunnel",
    "several_deployments",
    "moving",
    "doc_map",
    "image_test_network",
}
# Each old command with a text it is found in: one for each command and option of rkmon_lib/commands.py's GONE, and
# those of tests/doc_old_commands.yaml, rk stop --wait among them.
OLD = {
    "rkmon preflight": "run `rkmon preflight` first",
    "rkmon settings --check": "rkmon settings --check FILE",
    "rkmon build --check": "rkmon build --settings f.yaml --check",
    "rkmon build --print": "rkmon build --settings f.yaml --print",
    "rkmon build --template": "rkmon build --template > build.yaml",
    "rkmon settings --template": "`settings --template --profile full`",
    "rkmon image import": "rkmon image import FILE",
    "rkmon image export": "rkmon image export",
    "rkmon guide": "./rkmon guide prints it",
    "rkmon first-tick --timeout": "rkmon first-tick --timeout 5",
    "rkmon setup": "rkmon setup --settings f.yaml --overwrite",
    "rkmon down": "./rkmon down --volumes --yes",
    "PREFLIGHT": "it printed PREFLIGHT PASSED",
    "rkmon settings": "rkmon settings --check FILE",
    "rkmon swap": "rkmon swap --tag 2026.11.15",
    "rkmon refresh": "rkmon refresh --from ../new",
    "setup --section": "run ./rkmon setup --section reviewer",
    "rkmon logins": "rkmon logins lists them",
    "rkmon logins --check": "rkmon logins --check shows them",
    "rk settings export": "rk settings export > f",
    "rk off": "`rk off` pauses",
    "rk on": "rk on resumes",
    "rk stop --wait": "rk stop --drain --wait 60",
    "rkmon update": "rkmon update --rollback",
    "rk update": "./rkmon rk update --dry-run",
    "rk setup": "./rkmon rk setup --section gh",
    "--timeout MIN": "rk stop --drain --timeout 30",
}
# Each word list of tests/doc_words.yaml with a text it finds a word in, and one in other words.
WORDS = {
    "rejected": ("It holds the key.", "It keeps the key."),
    "unwanted": ("The tile is no longer shown.", "The tile is not shown."),
    "closing_words": ("Blank: none.", "Leave it blank: none."),
    "parallel_reviews": ("one pipe(s) per review", "one parallel review"),
    "parallel_builds": ("a heavy slot", "a parallel build"),
    "merge_queue_tile": ("the Merge pipeline tile", "the Merge queue tile"),
    "dashboards": ("open the admin page", "open the admin dashboard"),
    "container_users": ("the files of rkwrite", "the files of the writer"),
    "mode": ("the quiet mode", "run it in live mode"),
}
LISTS = dl.load_words()
COMMANDS = dl.load_old_commands()


def problems(lint, rel, source, *args, **kwargs):
    """What the lint function LINT finds in SOURCE, the text of the package's file REL."""
    return [p.what for p in lint(dl.read(rel, source), *args, **kwargs)]


def earlier_names(rel, source):
    """The earlier names in SOURCE, the text of the file REL, as `every_file` finds them."""
    found = dl.read(rel, source)
    lists = [w for w in LISTS if w.every_file]
    commands = [c for c in COMMANDS.values() if c.every_file]
    return dl.words(found, lists, dl.KINDS, generated_too=True) + dl.old_commands(
        found, commands, generated_too=True
    )


def write(root: pathlib.Path, files: dict) -> None:
    """Write FILES (a path in the package -> its text) in the package at ROOT."""
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text, encoding="utf-8")


class Tree(unittest.TestCase):
    """A test in a package folder of its own (self.root) with FILES, removed when the test ends."""

    FILES: dict = {}

    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.root = pathlib.Path(d.name)
        write(self.root, self.FILES)


class TestWhatTheLintsRead(Tree):
    def test_a_markdown_file(self):
        found = dl.read(
            "README.md",
            "# Title\n\nA paragraph\non two lines.\n\n## Part `one`\n```sh\nrk pause\n```\n"
            "<!-- BEGIN GENERATED: rkmon-commands -->\nmade\n<!-- END GENERATED: rkmon-commands -->\nAfter it.\n",
        )
        self.assertEqual(
            [
                (t.line, t.text, t.sections, t.code, t.generated, t.heading)
                for t in found
            ],
            [
                (1, "Title", ("Title",), False, False, True),
                (3, "A paragraph\non two lines.", ("Title",), False, False, False),
                (6, "Part `one`", ("Title", "Part `one`"), False, False, True),
                (8, "rk pause", ("Title", "Part `one`"), True, False, False),
                (
                    10,
                    "<!-- BEGIN GENERATED: rkmon-commands -->\nmade\n<!-- END GENERATED: rkmon-commands -->",
                    ("Title", "Part `one`"),
                    False,
                    True,
                    False,
                ),
                (13, "After it.", ("Title", "Part `one`"), False, False, False),
            ],
        )
        self.assertEqual({t.kind for t in found}, {USER})
        self.assertEqual(dl.read("docs/notes.md", "x y\n")[0].kind, MAINTAINER)
        self.assertEqual(dl.read("agentic/skills/x/SKILL.md", "x y\n")[0].kind, SKILL)

    def test_the_strings_a_module_may_print(self):
        """No docstring (but the module's own, when it prints it), no bare name; an f-string whole, with VALUE for
        each of its values."""
        source = (
            '"""Help: the review pipes."""\n'
            "def f(key):\n"
            '    """A docstring about pipes."""\n'
            '    name = "pipes"\n'
            '    print("the heavy checks run", f"{key} heavy jobs {key:>{width}}")\n'
            '    return "the review " "pipes"\n'
        )
        self.assertEqual(
            [(t.line, t.text, t.kind) for t in dl.read("tools/x.py", source)],
            [
                (5, "the heavy checks run", MESSAGE),
                (5, "{…} heavy jobs {…}", MESSAGE),
                (6, "the review pipes", MESSAGE),
            ],
        )
        found = dl.read("tools/x.py", source + "print(__doc__)\n")
        self.assertEqual((found[0].line, found[0].text), (1, "Help: the review pipes."))
        script = dl.read("docker/x", "#!/usr/bin/env python3\nprint('a b')\n")
        self.assertEqual([(t.line, t.text) for t in script], [(2, "a b")])
        # {program} reads as the command's name, as the messages name it.
        named = dl.read(
            "rkmon_lib/x.py", 'print(program.named("run {program} help"))\n'
        )
        self.assertEqual([t.text for t in named], ["run rkmon help"])

    def test_a_json_files_strings_but_its_keys_and_bare_names(self):
        found = dl.read(
            "schema/x.json",
            '{"a": {"description": "the heavy checks", "enum": ["pipes"],\n"b": "c d"}}',
        )
        self.assertEqual(
            [(t.line, t.text, t.kind) for t in found],
            [(1, "the heavy checks", MESSAGE), (2, "c d", MESSAGE)],
        )

    def test_a_text_files_comments_and_its_code(self):
        found = dl.read(
            "docker/x.sh",
            "#!/bin/sh\n# Copyright\n#\n# What it does,\n#   on two lines.\nset -e\necho hi  # a note\n  # indented\n",
        )
        self.assertEqual(
            [(t.line, t.text, t.code) for t in found],
            [
                (1, "#!/bin/sh", True),
                (2, "Copyright", False),
                (4, "What it does,\n  on two lines.", False),
                (6, "set -e\necho hi  # a note", True),
                (8, "indented", False),
            ],
        )
        self.assertEqual(dl.read("docker/Dockerfile", "# a b\nFROM x\n")[0].text, "a b")

    def test_a_file_no_lint_reads(self):
        write(
            self.root,
            {
                "a.md": "x y\n",
                "image.png": "x y\n",
                "x/g.py": "print('a b')\n",
                "tests/b.md": "x y\n",
                "state/c.md": "x y\n",
                "dist/d.md": "x y\n",
                ".git/e": "x y\n",
                "x/__pycache__/f.md": "x y\n",
            },
        )
        (self.root / "bytes.txt").write_bytes(b"\xff\xfe# a b\n")
        self.assertEqual(
            dl.package_files(self.root), ["a.md", "bytes.txt", "image.png", "x/g.py"]
        )
        self.assertEqual(dl.texts(self.root, "image.png"), [])
        self.assertEqual(dl.texts(self.root, "bytes.txt"), [])
        self.assertEqual([t.text for t in dl.texts(self.root, "x/g.py")], ["a b"])

    def test_the_package_s_files_are_git_s_but_those_it_ignores(self):
        """package_files.tracked's files, but the tests: in a repository, its tracked files and the others it does not
        ignore (a compose override, a deny list kept in the folder); in a folder git ignores whole, every file.
        """
        write(
            self.root,
            {
                ".gitignore": "/docker/compose.override.yaml\nprivacy_deny.txt\n/dist/\n",
                "README.md": "x y\n",
                "docker/compose.override.yaml": "# the review pipes\n",
                "privacy_deny.txt": "x y\n",
                "tests/t.py": "print('a b')\n",
                "dist/pkg/README.md": "x y\n",
                "dist/pkg/tests/t.py": "print('a b')\n",
            },
        )
        subprocess.run(["git", "init", "-q", "-b", "main", str(self.root)], check=True)
        self.assertEqual(dl.package_files(self.root), [".gitignore", "README.md"])
        self.assertEqual(dl.package_files(self.root / "dist" / "pkg"), ["README.md"])

    def test_a_phrase_that_wraps_is_found_on_the_line_it_starts_on(self):
        self.assertEqual(
            [
                p.line
                for p in dl.words(
                    dl.read("README.md", "x\n\nOne line,\nthe admin\n  page.\n"), LISTS
                )
            ],
            [4],
        )


class TestHomes(unittest.TestCase):
    FACT = dl.Fact(
        "stop_levels",
        "docs/restart.md",
        "The stop levels",
        (re.compile(r"\| Level \| Command \|"),),
    )
    COPY = 'a copy of stop_levels ("| Level | Command |"): link to docs/restart.md, "The stop levels" instead'

    def test_a_copy_names_the_home_to_link(self):
        found = dl.read(
            "README.md", "## Stopping\n\n| Level | Command | What it does |\n"
        )
        self.assertEqual(
            dl.homes(found, [self.FACT]),
            [dl.Problem("README.md", 3, "homes", self.COPY)],
        )
        self.assertEqual(
            problems(
                dl.homes,
                "rkmon_lib/x.py",
                'print("| Level | Command |")\n',
                [self.FACT],
            ),
            [self.COPY],
        )

    def test_a_copy_of_a_help_home_names_the_command_to_type(self):
        fact = dl.Fact(
            "image_test_network",
            "rkmon_lib/commands.py",
            "",
            (re.compile(r"reaches only stand-ins"),),
            "image test",
        )
        self.assertEqual(
            problems(
                dl.homes,
                "QUICK_BUILD_GUIDE.md",
                "## Step 4\n\nIts network reaches only stand-ins on this host.\n",
                [fact],
            ),
            [
                'a copy of image_test_network ("reaches only stand-ins"): point to `rkmon image test --help` '
                "instead"
            ],
        )

    def test_a_help_home_names_a_command_of_the_table(self):
        entry = "x:\n  home: rkmon_lib/commands.py\n  help: {}\n  signatures:\n    - 'only stand-ins'\n"
        with tempfile.TemporaryDirectory() as tmp:
            homes = pathlib.Path(tmp) / dl.HOMES
            homes.parent.mkdir(parents=True)
            homes.write_text(entry.format("image test"), encoding="utf-8")
            self.assertEqual(dl.load_homes(tmp)[0].help, "image test")
            homes.write_text(entry.format("image tset"), encoding="utf-8")
            with self.assertRaisesRegex(
                ValueError, "help names no command of the command table: image tset"
            ):
                dl.load_homes(tmp)

    def test_the_home_and_a_generated_section_pass(self):
        home = "# Restart\n\n## The stop levels\n\n| Level | Command | What it does |\n"
        self.assertEqual(problems(dl.homes, "docs/restart.md", home, [self.FACT]), [])
        made = "<!-- BEGIN GENERATED: rkmon-commands -->\n| Level | Command |\n<!-- END GENERATED: rkmon-commands -->\n"
        self.assertEqual(problems(dl.homes, "README.md", made, [self.FACT]), [])

    def test_a_home_without_its_section_or_a_signature(self):
        self.assertEqual(
            dl.homes(dl.read("docs/restart.md", "# Restart\n"), [self.FACT]),
            [
                dl.Problem(
                    "docs/restart.md",
                    1,
                    "homes",
                    'no section "The stop levels", the home of stop_levels (tests/doc_fact_homes.yaml)',
                )
            ],
        )
        signature = self.FACT.signatures[0].pattern
        self.assertEqual(
            dl.homes(
                dl.read(
                    "docs/restart.md",
                    "# Restart\n\n## The stop levels\n\nrk pause\n",
                ),
                [self.FACT],
            ),
            [
                dl.Problem(
                    "docs/restart.md",
                    3,
                    "homes",
                    f"the home of stop_levels has no match of {signature!r}: write the fact here, or change its "
                    "signature in tests/doc_fact_homes.yaml",
                )
            ],
        )
        whole = dl.Fact(
            "doc_map", "README.md", "", (re.compile(r"\| Doc \| Reader \|"),)
        )
        self.assertEqual(
            problems(
                dl.homes, "README.md", "x\n\n| Doc | Reader | Its one job |\n", [whole]
            ),
            [],
        )
        self.assertEqual(len(problems(dl.homes, "README.md", "x\n", [whole])), 1)
        # A fact whose home is not among the texts linted.
        self.assertEqual(problems(dl.homes, "README.md", "x\n", [self.FACT]), [])


class TestKeys(unittest.TestCase):
    KNOWN = dl.Keys(
        frozenset(
            {
                ("monitor",),
                ("monitor", "parallel_builds"),
                ("features",),
                ("features", "*"),
                ("features", "*", "mode"),
            }
        ),
        frozenset({("retention",), ("retention", "job_days")}),
    )

    def test_a_key_of_a_settings_file_passes(self):
        for text in (
            "Set `monitor.parallel_builds`, or features.alerts.mode.",
            "Each features.<name>.mode, features.*.mode and monitor.<setting>.",
            "See docs/monitor.parallel_builds, monitor.md, monitor.example.com, monitor.tick() and services",
        ):
            with self.subTest(text=text):
                self.assertEqual(problems(dl.keys, "README.md", text, self.KNOWN), [])

    def test_a_key_no_settings_file_has(self):
        self.assertEqual(
            dl.keys(
                dl.read("README.md", "x\n\nSet `monitor.parallel_bilds`."), self.KNOWN
            ),
            [
                dl.Problem(
                    "README.md",
                    3,
                    "keys",
                    "monitor.parallel_bilds: no settings file has this key, and no internal key is named so",
                )
            ],
        )
        self.assertEqual(
            problems(
                dl.keys,
                "QUICK_USER_GUIDE.md",
                "```\nretention.job_days\n```\n",
                self.KNOWN,
            ),
            [
                "retention.job_days is an internal key: a user doc names only the settings files' keys"
            ],
        )

    def test_what_is_read_the_user_docs_the_compose_header_and_the_messages(self):
        text = "The monitor.parallel_bilds and retention.job_days keys.\n"
        self.assertEqual(problems(dl.keys, "docs/notes.md", text, self.KNOWN), [])
        # A message: an internal key passes there.
        self.assertEqual(
            problems(dl.keys, "rkmon_lib/x.py", f"print({text!r})\n", self.KNOWN),
            [
                "monitor.parallel_bilds: no settings file has this key, and no internal key is named so"
            ],
        )
        compose = "# Copyright\n#\n# Set retention.job_days.\nname: rkmon\n# retention.job_days\n"
        self.assertEqual(
            [
                (p.line, p.what)
                for p in dl.keys(dl.read("docker/compose.yaml", compose), self.KNOWN)
            ],
            [
                (
                    3,
                    "retention.job_days is an internal key: a user doc names only the settings files' keys",
                )
            ],
        )

    def test_a_message_names_a_settings_file_s_key_or_an_internal_key(self):
        """An earlier name of a key in a message is a problem; an internal key, a name a value of the code starts or
        ends, and a host's name are not."""
        known = self.KNOWN._replace(settings=self.KNOWN.settings | {("host",)})
        code = (
            'print("ramdisk: off (monitor.ramdisk_gb 0)")\n'
            'print("retention.job_days is 14")\n'
            'units = {f"{prefix}monitor.timer": 1, f"monitor.parallel_{k}": 2}\n'
            'print("a model server answers at host.docker.internal")\n'
        )
        self.assertEqual(
            [(p.line, p.what) for p in dl.keys(dl.read("rkmon_lib/x.py", code), known)],
            [
                (
                    1,
                    "monitor.ramdisk_gb: no settings file has this key, and no internal key is named so",
                )
            ],
        )

    def test_a_name_inside_an_internal_key_whose_value_is_a_mapping(self):
        known = self.KNOWN._replace(
            internal=self.KNOWN.internal | {("teams",), ("limits",)},
            inside=frozenset({("teams",)}),
        )
        self.assertEqual(
            problems(dl.keys, "README.md", "Set teams.<area> or teams.lib.", known),
            [
                "teams.<area> is an internal key: a user doc names only the settings files' keys",
                "teams.lib is an internal key: a user doc names only the settings files' keys",
            ],
        )
        self.assertEqual(
            problems(dl.keys, "README.md", "Set limits.max.", known),
            [
                "limits.max: no settings file has this key, and no internal key is named so"
            ],
        )

    def test_an_allowance_keeps_the_keys_of_its_place(self):
        """No key is read where an allowance (tests/doc_keys.yaml) keeps the whole text, and only the keys its keep
        does not match where it has one; the keys of a file it does not cover are read.
        """
        code = 'print("monitor.ramdisk_gb is now monitor.parallel_builds; set monitor.tick_mins too")\n'
        found = [
            "monitor.ramdisk_gb: no settings file has this key, and no internal key is named so",
            "monitor.tick_mins: no settings file has this key, and no internal key is named so",
        ]
        place = re.compile(r"rkmon_lib/layout1\.py")
        whole = self.KNOWN._replace(allowed=(dl.Allowance(place, (), "", None, "w"),))
        self.assertEqual(problems(dl.keys, "rkmon_lib/layout1.py", code, whole), [])
        self.assertEqual(problems(dl.keys, "rkmon_lib/x.py", code, whole), found)
        keep = re.compile(r"monitor\.ramdisk_gb", re.M)
        kept = self.KNOWN._replace(allowed=(dl.Allowance(place, (), "", keep, "w"),))
        self.assertEqual(
            problems(dl.keys, "rkmon_lib/layout1.py", code, kept), found[1:]
        )

    def test_the_package_s_keys(self):
        known = dl.load_keys()
        for words in (
            ("container", "name"),
            ("image", "tag"),
            ("features", "alerts", "mode"),
        ):
            self.assertIn(words, known.settings)
        self.assertIn(("retention", "job_days"), known.internal)
        self.assertNotIn(("retention", "job_days"), known.settings)

    def test_the_internal_keys_are_the_registry_s(self):
        """Each key of the registry, of every owner, is an internal key, a state key by its key in the state file too,
        and a name inside a key whose value is a mapping is one; a deploy key's file keys are the settings file's.
        """
        known = dl.load_keys()
        for k in registry.keys():
            with self.subTest(key=k.name):
                self.assertIn(tuple(k.name.split(".")), known.internal)
                if k.owner == registry.STATE:
                    for f in k.file_keys:
                        self.assertIn(tuple(f.split(".")), known.internal)
                self.assertEqual(
                    tuple(k.name.split(".")) in known.inside,
                    isinstance(k.default, dict),
                )
        deploy = registry.key("ramdisk.size_gb")
        self.assertEqual(deploy.owner, registry.DEPLOY)
        for f in deploy.file_keys:
            self.assertIn(tuple(f.split(".")), known.settings)
            self.assertNotIn(tuple(f.split(".")), known.internal)
        for name in ("admin_dashboard.port", "codeowner_teams"):
            self.assertNotEqual(registry.key(name).owner, registry.DEPLOY)
        self.assertEqual(
            problems(
                dl.keys,
                "README.md",
                "The admin_dashboard.port, codeowner_teams.<area> and hot_spots.<name> keys.",
                known,
            ),
            [
                f"{name} is an internal key: a user doc names only the settings files' keys"
                for name in (
                    "admin_dashboard.port",
                    "codeowner_teams.<area>",
                    "hot_spots.<name>",
                )
            ],
        )

    def test_the_package_s_allowances_layout_1_s_translators_and_schema(self):
        """tests/doc_keys.yaml keeps the keys that layout 1's translators and its schema name on purpose: the keys
        lint finds none there. Each allowance names files of the package, and why."""
        known = dl.load_keys()
        files = dl.package_files(ROOT)
        self.assertTrue(known.allowed)
        for a in known.allowed:
            with self.subTest(files=a.files.pattern):
                self.assertTrue(a.why)
                self.assertTrue(any(a.files.fullmatch(rel) for rel in files))
        layout1 = [
            "rkmon_lib/from_layout1.py",
            "tools/translate_production.py",
            "schema/layout1/deploy_settings.schema.json",
        ]
        self.assertEqual([str(p) for p in dl.run(ROOT, layout1, ("keys",))], [])


class TestWords(unittest.TestCase):
    def test_each_list_finds_its_words_and_passes_other_words(self):
        self.assertEqual(set(WORDS), {w.name for w in LISTS})
        for wl in LISTS:
            found, fine = WORDS[wl.name]
            with self.subTest(list=wl.name):
                what = problems(dl.words, "README.md", found, [wl])
                self.assertTrue(what)
                self.assertTrue(all(f"({wl.name}): {wl.say}" in w for w in what), what)
                self.assertEqual(problems(dl.words, "README.md", fine, LISTS), [])

    def test_a_problem_names_the_word_its_lists_and_what_to_write(self):
        self.assertEqual(
            dl.words(dl.read("README.md", "It holds the key.\n"), LISTS),
            [
                dl.Problem(
                    "README.md",
                    1,
                    "words",
                    '"holds" (rejected): write it in other words',
                )
            ],
        )
        self.assertEqual(
            problems(dl.words, "README.md", "the review pipes", LISTS),
            [
                '"pipes" (rejected, parallel_reviews): write it in other words; write the parallel reviews'
            ],
        )

    def test_text_in_backticks_is_read_for_a_list_that_says_code(self):
        for source in (
            "Run `admin page` and `rkwrite`.\n",
            "```\nadmin page rkwrite\n```\n",
        ):
            with self.subTest(source=source):
                what = problems(dl.words, "README.md", source, LISTS)
                self.assertEqual(len(what), 1, what)
                self.assertTrue(what[0].startswith('"rkwrite" (container_users)'), what)

    def test_the_texts_read_user_docs_skill_files_and_messages(self):
        self.assertEqual(
            problems(
                dl.words, "docs/notes.md", "the files of rkwrite, the scaler", LISTS
            ),
            [],
        )
        self.assertEqual(
            problems(dl.words, "agentic/skills/x/SKILL.md", "the scaler", LISTS), []
        )
        self.assertEqual(len(problems(dl.words, "README.md", "the scaler", LISTS)), 1)
        self.assertEqual(
            len(problems(dl.words, "rkmon_lib/x.py", 'print("the scaler")\n', LISTS)), 1
        )
        made = "<!-- BEGIN GENERATED: rkmon-commands -->\nthe scaler\n<!-- END GENERATED: rkmon-commands -->\n"
        self.assertEqual(problems(dl.words, "README.md", made, LISTS), [])
        self.assertEqual(
            len(problems(dl.words, "README.md", made, LISTS, generated_too=True)), 1
        )

    def test_an_allowance_keeps_its_own_words_only(self):
        self.assertEqual(
            problems(dl.words, "README.md", "Advanced Micro Devices, Inc.", LISTS), []
        )
        self.assertEqual(
            problems(dl.words, "README.md", "the advanced settings", LISTS),
            ['"advanced" (unwanted): write what is true today, in plain words'],
        )
        # "mode" beside its value: prototype mode, live mode, mode: prototype or live, the .mode key; a file's
        # permissions, by their octal number; and other programs' words: pytest's option, git's diff.
        for fine in (
            "features.alerts.mode",
            "mode: live",
            'mode: "prototype"',
            "in live mode",
            "the prototype mode",
            'Every feature starts in "prototype" mode.',
            "the folder is mode 700",
            "each file mode 0600, its folder mode 0700",
            "where modes 700 and 600 do not work",
            "a tmpfs at /tmp/x:size=4g,mode=1777,exec",
            "pytest --import-mode=importlib",
            "new file mode 100644",
            "deleted file mode 100755",
        ):
            with self.subTest(fine=fine):
                self.assertEqual(problems(dl.words, "README.md", fine, LISTS), [])
        for found in (
            "the quiet mode",
            "modes of the dashboards",
            "its mode. It is live",
            "set its mode to live",
            "Maintenance mode: the monitor waits.",
            "mode 800",
            "mode 60",
            "mode 12345",
            "mode=800",
            "the import mode",
            "a new mode",
        ):
            with self.subTest(found=found):
                self.assertEqual(len(problems(dl.words, "README.md", found, LISTS)), 1)

    def test_the_noun_holds_stays_where_it_names_the_merge_queue_s_holds(self):
        """The verb "holds" is rejected; the noun stays where it names the merge queue's holds of pull requests:
        the lessons' section and the tick summary's HOLDS line. Layout 1's schema keeps the words its package
        wrote."""
        for rel, fine in (
            (
                "rkmon_lib/lessons.py",
                'out += _section("Holds, releases and disarms", log(("hold",)))\n',
            ),
            (
                "rkmon_lib/tickfilter.py",
                'X = (r"HOLDS:", ROUTINE, "report.tick_summary", "the holds")\n',
            ),
            (
                "schema/layout1/deploy_settings.schema.json",
                '{"description": "The repository that holds rocKE."}\n',
            ),
        ):
            with self.subTest(rel=rel):
                self.assertEqual(problems(dl.words, rel, fine, LISTS), [])
        # The verb, there and in layout 2's schema.
        for rel, found in (
            ("rkmon_lib/tickfilter.py", 'print("another tick holds the lock")\n'),
            ("rkmon_lib/lessons.py", 'print("Holds, then releases")\n'),
            (
                "schema/deploy_settings.schema.json",
                '{"description": "The repository that holds rocKE."}\n',
            ),
        ):
            with self.subTest(found=found):
                self.assertEqual(len(problems(dl.words, rel, found, LISTS)), 1)

    def test_layout_1_s_key_advanced_stays_where_the_move_from_layout_1_names_it(self):
        """The word "advanced" is unwanted; the move from layout 1 (rkmon_lib/from_layout1.py) keeps it where it names
        layout 1's key and the keys under it, which it reads and names in its notes. The word stays found there, and
        the key in every other file."""
        rel = "rkmon_lib/from_layout1.py"
        for fine in (
            'self.note(f"advanced.jira.component is now {new}")\n',
            'self.note(f"advanced: {n} the same as the package\'s defaults: dropped")\n',
            'self.note(f"advanced is {shown}: not carried")\n',
        ):
            with self.subTest(fine=fine):
                self.assertEqual(problems(dl.words, rel, fine, LISTS), [])
        for path, found in (
            (rel, 'print("the advanced settings")\n'),
            ("rkmon_lib/config.py", 'print(f"advanced.jira.component is now {new}")\n'),
        ):
            with self.subTest(path=path, found=found):
                self.assertEqual(len(problems(dl.words, path, found, LISTS)), 1)

    def test_names_the_code_reads_as_they_are_written_stay_in_their_own_files(self):
        """Some listed words are names the code reads or prints as they are written, each kept in its own file only:
        the tick summary's HOLDS line and the board's Holds row and key in rkmon_lib/report.py, and the HOLDS line in
        tests/test_tick_filter.py; the SCALER line in rkmon_lib/tickfilter.py; the key {me} in rkmon_lib/dispatch.py;
        the dashboard's tk-act and data-act and a browser's Sec-Fetch-Mode header in tests/test_tickets.py; the GPU
        job's $J/mode file and MODE variable in rkmon_lib/gpu.py; a feature's mode beside the value the report of
        tools/translate_production.py fills in; and rsync's "Broken pipe" in rkmon_lib/publish.py and two tests. The
        words stay found in their other uses there, and the names in every other file.
        """
        other = "rkmon_lib/config.py"
        for rel, fine, found in (
            (
                "rkmon_lib/report.py",
                (
                    'lines.append("HOLDS: " + held)\n',
                    'rows.append(("Holds", n))\n',
                    'key.append("holds " + parts)\n',
                ),
                'print("the lock holds")\n',
            ),
            (
                "tests/test_tick_filter.py",
                ('line = "HOLDS: #5(triage)"\n',),
                'print("another tick holds the lock")\n',
            ),
            (
                "rkmon_lib/tickfilter.py",
                ('RULES = [(r"SCALER ", ROUTINE)]\n',),
                'print("the scaler moved")\n',
            ),
            (
                "rkmon_lib/dispatch.py",
                ('SECTIONS = [("{me}\'s hand-written", 30)]\n',),
                'print("tell me")\n',
            ),
            (
                "tests/test_tickets.py",
                (
                    "html = \"<button class='tk-act' data-act='approve'>\"\n",
                    "js = \"post(base+'/ticket-'+b.dataset.act)\"\n",
                    'nav = {"Sec-Fetch-Mode": "navigate"}\n',
                ),
                'print("act on it in the quiet mode")\n',
            ),
            (
                "rkmon_lib/gpu.py",
                (
                    'S = \'[ "$(cat "$J/mode")" = develop ] && B=-\'\n',
                    "S = f'D={d}; MODE={m}'\n",
                    "S = 'if [ $MODE = merge ]; then MODE=head; fi'\n",
                    "S = 'echo $MODE > $J/mode'\n",
                ),
                'print("the mode of the job")\n',
            ),
            (
                "tools/translate_production.py",
                (
                    'S = f", mode {shown(mode)}"\n',
                    'S = f"{key}: enabled {on}, mode {value}"\n',
                ),
                'print("in the quiet mode")\n',
            ),
            (
                "rkmon_lib/publish.py",
                ('LOST = r"write error: Broken pipe"\n',),
                'print("a pipe")\n',
            ),
            (
                "tests/test_publish.py",
                ('err = "write error: Broken pipe (32)"\n',),
                'print("a pipe")\n',
            ),
            (
                "tests/test_tick_filter.py",
                ('line = "(not read: Broken pipe)"\n',),
                'print("a pipe")\n',
            ),
        ):
            for text in fine:
                with self.subTest(rel=rel, fine=text):
                    self.assertEqual(problems(dl.words, rel, text, LISTS), [])
                with self.subTest(rel=other, found=text):
                    self.assertNotEqual(problems(dl.words, other, text, LISTS), [])
            with self.subTest(rel=rel, found=found):
                self.assertNotEqual(problems(dl.words, rel, found, LISTS), [])

    def test_a_word_is_found_whole_never_inside_a_longer_word(self):
        """A listed word at the end or the start of a longer word is not that word: "me" in "name", "acts" in
        "reacts", "act" in "actor"."""
        for fine in ("Give it a name and a theme.", "The actor reacts."):
            with self.subTest(fine=fine):
                self.assertEqual(problems(dl.words, "README.md", fine, LISTS), [])
        self.assertEqual(
            problems(dl.words, "README.md", "Ask me.", LISTS),
            ['"me" (rejected): write it in other words'],
        )

    def test_the_dashboards_earlier_names_are_found(self):
        """The monitor's two web pages are the admin dashboard and the team dashboard; their earlier names go."""
        for earlier in ("the local page", "the published copy", "the team page"):
            with self.subTest(earlier=earlier):
                self.assertEqual(
                    problems(dl.words, "README.md", f"Open {earlier}.", LISTS),
                    [
                        f'"{earlier.split(" ", 1)[1]}" (dashboards): write the admin dashboard, or the team '
                        "dashboard"
                    ],
                )

    def test_each_earlier_name_is_found_and_the_rest_passed_over(self):
        for sample in (
            "the review pipes in effect",
            "one pipe(s) per review",
            "Pipes stay at 1",
            "a heavy slot",
            "the heavy checks",
            "each heavy-job",
            "heavy local jobs",
            "the Merge pipeline tile",
            "up to <REVIEW_PIPES>",
            "`rk off` pauses",
            "rk on resumes",
            "rk stop --drain --wait 60",
            "rk stop [--drain] [--wait MIN]",
            "up to `--wait` minutes",
            "./rkmon preflight",
            "run `preflight` first",
            "rkmon settings --check f.yaml",
            "then run `rkmon settings`.",
            "rkmon build --settings f.yaml --check",
            "`settings --template --profile full`",
            "rkmon image import m.tgz",
            "then image export it",
            "docker/rkmon guide",
            "rkmon first-tick --timeout 5",
        ):
            with self.subTest(sample=sample):
                self.assertTrue(earlier_names("docs/notes.md", sample))
        # A replaced command in a message: after the program's name, or a field that holds it.
        for code in (
            'X = " preflight --settings f.yaml"\n',
            'X = " build --template"\n',
            'X = " guide build"\n',
            'X = " first-tick --timeout 10"\n',
            'X = f"{program.name()} guide build"\n',
            'X = program.named("run {program} image export")\n',
        ):
            with self.subTest(code=code):
                self.assertTrue(earlier_names("rkmon_lib/x.py", code))
        # The command table keeps each replaced command's words, so that the parser names the one that replaces it.
        gone = 'GONE = (Gone("image import", "image load IMAGE_FILE"),)\n'
        self.assertEqual(earlier_names("rkmon_lib/commands.py", gone), [])
        self.assertTrue(earlier_names("rkmon_lib/x.py", gone))
        self.assertTrue(
            earlier_names("rkmon_lib/commands.py", 'X = "rkmon image import m.tgz"\n')
        )
        # An allowance passes over its own words only, not an earlier name beside them.
        found = earlier_names(
            "rkmon_lib/publish.py",
            'X = "write error: Broken pipe, from the review pipes and heavy slots"\n',
        )
        self.assertEqual(
            [p.what.split('"')[1] for p in found], ["pipes", "heavy slots"]
        )
        # No allowance keeps a section named by the parallel reviews' earlier name: each word is found.
        found = earlier_names(
            "docs/internal_settings.md", "  pipes     the review pipes alone\n"
        )
        self.assertEqual([p.what.split('"')[1] for p in found], ["pipes", "pipes"])
        found = earlier_names("README.md", "Its settings are `--section pipes`.\n")
        self.assertEqual([p.what.split('"')[1] for p in found], ["pipes"])
        for fine in (
            "the parallel reviews and the parallel builds",
            "the review pipeline",
            "rk on the image's own code",
            "rk kill [--wait S]",
            "a bounded time (`--wait` seconds)",
            "rkmon check --settings FILE",
            "the browser's preflight request",
            "docker build --check",
            "gh pr create --template x",
            "rkmon first-tick --timeout-minutes 5",
            "rk guide",
            "./rkmon rk guide build",
            "the guide to the deploy path",
            "rkmon image load m.tgz",
            "`settings.py` reads the settings files",
        ):
            with self.subTest(fine=fine):
                self.assertEqual(earlier_names("README.md", fine), [])


class TestEveryFile(Tree):
    def test_every_file_is_read_its_code_and_its_generated_sections_too(self):
        """every_file over a package of its own, with the package's lists: a user doc's generated section, a
        maintainer doc and the code are each read for the earlier names."""
        lists = {
            rel: (ROOT / rel).read_text(encoding="utf-8")
            for rel in (dl.WORDS, dl.OLD_COMMANDS)
        }
        made = (
            "# Title\n\n<!-- BEGIN GENERATED: rkmon-commands -->\nthe review pipes\n"
            "<!-- END GENERATED: rkmon-commands -->\n"
        )
        files = {
            "README.md": made,
            "docs/notes.md": "Run rk off.\n",
            "tools/x.py": "X = 'a heavy slot'\n",
        }
        write(self.root, {**lists, **files})
        self.assertEqual(
            [(p.path, p.line, p.lint) for p in dl.every_file(self.root)],
            [
                ("README.md", 4, "words"),
                ("docs/notes.md", 1, "old-commands"),
                ("tools/x.py", 1, "words"),
            ],
        )


class TestOldCommands(unittest.TestCase):
    def test_each_old_command_is_found_with_what_to_run_instead(self):
        self.assertEqual(set(OLD), set(COMMANDS))
        for name, sample in OLD.items():
            with self.subTest(name=name):
                what = problems(dl.old_commands, "README.md", sample, [COMMANDS[name]])
                self.assertEqual(len(what), 1, what)
                self.assertIn(
                    f": {name} is an old command; instead: {COMMANDS[name].instead}",
                    what[0],
                )
        self.assertEqual(
            dl.old_commands(
                dl.read("README.md", "Run `rkmon preflight`."), COMMANDS.values()
            ),
            [
                dl.Problem(
                    "README.md",
                    1,
                    "old-commands",
                    '"rkmon preflight": rkmon preflight is an old command; instead: rkmon check --settings FILE',
                )
            ],
        )

    def test_each_replaced_command_of_the_table_is_one(self):
        """commands.GONE is the one home of the program's replaced commands and options: each row is an old command,
        named and replaced with the program's name before it and read in every file; the list holds none of them.
        """
        for g in commands.GONE:
            with self.subTest(words=g.words):
                c = COMMANDS[f"rkmon {g.words}"]
                # the command that replaces it, else the row's own words (setup's)
                instead = f"rkmon {g.instead}" if g.instead else g.replacement("rkmon")
                self.assertEqual((c.instead, c.every_file), (instead, True))
        listed = set(dl.answers.load(ROOT / dl.OLD_COMMANDS))
        self.assertEqual(
            listed, set(COMMANDS) - {f"rkmon {g.words}" for g in commands.GONE}
        )

    def test_every_text_is_read_its_code_too(self):
        for rel, source in (
            ("README.md", "```\nrkmon preflight\n```\n"),
            ("docs/notes.md", "rkmon preflight"),
            ("agentic/skills/x/SKILL.md", "rkmon preflight"),
            ("rkmon_lib/x.py", 'print("run rkmon preflight")\n'),
            (
                "rkmon_lib/y.py",
                'print(program.named("run {program} build --check FILE first"))\n',
            ),
        ):
            with self.subTest(rel=rel):
                self.assertEqual(
                    len(problems(dl.old_commands, rel, source, COMMANDS.values())), 1
                )

    def test_the_one_place_a_doc_may_name_rk_settings_export(self):
        """rk settings export in QUICK_DEPLOY_GUIDE's "Moving from an existing monitor", and nowhere else."""
        step = "1. (umask 077; ./rkmon rk settings export > ~/rkmon-deploy.yaml)\n"
        moving = "# Deploy\n\n## Moving from an existing monitor\n\n" + step
        self.assertEqual(
            problems(
                dl.old_commands, "QUICK_DEPLOY_GUIDE.md", moving, COMMANDS.values()
            ),
            [],
        )
        for rel, source in (
            ("QUICK_DEPLOY_GUIDE.md", "## Several deployments\n\n" + step),
            ("README.md", moving),
        ):
            with self.subTest(rel=rel, source=source):
                self.assertEqual(
                    len(problems(dl.old_commands, rel, source, COMMANDS.values())), 1
                )

    def test_the_commands_today_pass(self):
        for fine in (
            "rk pause, then rk resume",
            "rk stop --drain --timeout-minutes 60",
            "rkmon check --settings FILE, rkmon confirm --settings FILE",
            "rkmon image load FILE, rkmon image save",
            "rkmon package update --from NEW",
            "the quick guide, a preflight check, swap space, a refresh of the page",
            "rk guide prints it, as ./rkmon rk guide does",
            "rkmon help day-to-day, rkmon template build FILE",
        ):
            with self.subTest(fine=fine):
                self.assertEqual(
                    problems(dl.old_commands, "README.md", fine, COMMANDS.values()), []
                )


class TestQuotedOutput(Tree):
    FILES = {
        "rkmon_lib/sample.py": (
            "from rkmon_lib import outcome, program\n\n\n"
            "def report(path, n):\n"
            '    outcome.finish("REFUSED", f"the settings file {path} is missing; nothing changed")\n'
            '    print(f"CONFIRMED: {n} images match their digests")\n'
            '    print("ROLLBACK FAILED: the image before it is gone")\n'
            '    print(program.named("REFUSED: the file is not confirmed; run {program} confirm first"))\n'
        ),
    }

    def quoted(self, rel, source):
        return problems(dl.quoted_output, rel, source, dl.load_templates(self.root))

    def test_an_outcome_the_code_prints_passes(self):
        for quote in (
            "`REFUSED: the settings file <file> is missing; nothing changed`",
            "`REFUSED: the settings file … is missing; nothing changed`",
            "`CONFIRMED: 3 images match their digests`",
            "`ROLLBACK FAILED: the image before it is gone`",
            "`PASSED`",
            "`WROTE docker/.env`",
            "`rkmon check`",
            "`REFUSED: a b; nothing changed`",
            "`REFUSED: the file is not confirmed; run rkmon confirm first`",
            # A part of two words is too short to tell: it passes, printed or not.
            "`REFUSED: it broke; nothing changed`",
        ):
            with self.subTest(quote=quote):
                self.assertEqual(
                    self.quoted("QUICK_DEPLOY_GUIDE.md", f"It says {quote}.\n"), []
                )

    def test_an_outcome_the_code_does_not_print(self):
        self.assertEqual(
            self.quoted(
                "QUICK_DEPLOY_GUIDE.md",
                "It says `REFUSED: the settings file is gone; nothing changed`.\n",
            ),
            [
                '"REFUSED: the settings file is gone; nothing changed": the code prints no "the settings file is '
                'gone"; quote what it prints'
            ],
        )
        # Three words are enough to tell.
        self.assertEqual(
            self.quoted("QUICK_DEPLOY_GUIDE.md", "`REFUSED: the disk broke`\n"),
            [
                '"REFUSED: the disk broke": the code prints no "the disk broke"; quote what it prints'
            ],
        )
        self.assertEqual(
            self.quoted(
                "QUICK_DEPLOY_GUIDE.md",
                "It said `PREFLIGHT PASSED: every check ran`.\n",
            ),
            [
                '"PREFLIGHT PASSED: every check ran": the code prints no "PREFLIGHT PASSED: every check ran"; quote '
                "what it prints"
            ],
        )
        fenced = "x\n\n```\nrkmon confirm\nREFUSED: the settings file is gone; nothing changed\n```\n"
        found = dl.read("QUICK_DEPLOY_GUIDE.md", fenced)
        self.assertEqual(
            [p.line for p in dl.quoted_output(found, dl.load_templates(self.root))], [5]
        )

    def test_only_the_user_docs_are_read(self):
        bad = "It says `REFUSED: the settings file is gone; nothing changed`.\n"
        self.assertEqual(self.quoted("docs/notes.md", bad), [])
        made = f"<!-- BEGIN GENERATED: rkmon-commands -->\n{bad}<!-- END GENERATED: rkmon-commands -->\n"
        self.assertEqual(self.quoted("README.md", made), [])

    def test_the_code_s_texts_are_read_once(self):
        templates = dl.load_templates(self.root)
        self.assertIs(dl.load_templates(self.root), templates)
        self.assertIn("{…} images match their digests", {t.text for t in templates})


class TestQuotedRowsAndLines(Tree):
    """The check rows, fixes, next lines and notes a user doc quotes, against a sample of the code; the commands and
    the steps are the package's own (rkmon_lib/commands.py, rkmon_lib/steps.py)."""

    FILES = {
        "rkmon_lib/sample.py": (
            'AFTER = "{} (fix each FAIL above)"\n\n\n'
            "def report(given, who, key, file, value, why, path, n, m, what, reasons):\n"
            '    print(f"{given}: the id of {who}, one of the package\'s own users")\n'
            '    print(f"set {key} in {file} to {value}")\n'
            '    print("an id of 1000 or more")\n'
            '    print("the merge queue is off in these settings")\n'
            '    print("note: the image is older than the package")\n'
            '    print("the disk is nearly full")\n'
            '    print(", ".join(reasons))\n'
            "    print(AFTER.format(why))\n"
            '    print("the file " + path + " is outside the package folder")\n'
            '    print(f"the disk has {n} GiB free, less than the {m} GiB the image needs")\n'
            '    row = f"{n} logins are not set"\n'
            '    row += f" ({what})"\n'
            '    print(row, f"the {what} login is missing")\n'
            '    print("REFUSED: start it with docker/rkmon start first")\n'
        ),
        # A settings file's schema, whose key's x-source a fix's edit prints as the key's value.
        dl.SCHEMAS[-1]: json.dumps(
            {
                "type": "object",
                "properties": {
                    "tool": {
                        "type": "object",
                        "properties": {
                            "package_url": {
                                "type": "string",
                                "x-source": "the address of the tool's binary, from wherever your organization "
                                "keeps it",
                            }
                        },
                    }
                },
            }
        ),
    }

    def quoted(self, quote):
        source = f"It says `{quote}`.\n"
        templates = dl.load_templates(self.root)
        return problems(dl.quoted_output, "QUICK_DEPLOY_GUIDE.md", source, templates)

    def assert_printed(self, *quotes):
        for quote in quotes:
            with self.subTest(quote=quote):
                self.assertEqual(self.quoted(quote), [])

    def assert_not_printed(self, quote, part):
        with self.subTest(quote=quote):
            self.assertEqual(
                self.quoted(quote),
                [f'"{quote}": the code prints no "{part}"; quote what it prints'],
            )

    def test_a_check_row(self):
        self.assert_printed(
            "FAIL container ids: 1999 is set: the id of the GitHub writer, one of the package's own users",
            "OFF: the merge queue is off in these settings",
        )
        self.assert_not_printed(
            "FAIL container ids: the monitor's user and the writer share an id",
            "the monitor's user and the writer share an id",
        )
        self.assert_not_printed(
            "OFF: the queue is paused for now", "the queue is paused for now"
        )

    def test_a_fix_s_edits(self):
        self.assert_printed(
            "fix: set host.users.monitor_uid in ~/rkmon-deploy.yaml to an id of 1000 or more",
            "fix: set a.b in <file> to an id of 1000 or more, or delete a.b from <file>",
            # a file the code names in words, where no path of it is at hand
            "fix: set a.b in your deploy settings file to an id of 1000 or more",
            'fix: set a.b in your build settings file to false, and set c.d in <file> to "serve"',
        )
        # An edit's value in words must be a text of the code.
        self.assert_not_printed(
            "fix: set a.b in <file> to a free id of your own", "a free id of your own"
        )
        # In another order, an edit's words are a run of words, which the code does not print.
        self.assert_not_printed(
            "fix: set a.b to an id of 1000 or more in your deploy settings",
            "set a.b to an id of 1000 or more in your deploy settings",
        )
        # The edit's own form (checks.Edit's text) prints no run of words: an edit's FILE is a word, a word in angle
        # brackets or a file in words.
        self.assert_not_printed(
            "fix: set a.b in the wrong file to a value the code never prints",
            "set a.b in the wrong file to a value the code never prints",
        )

    def test_a_fix_s_edit_with_the_value_a_schema_says_where_to_find(self):
        # A key's x-source, which settings_file.source_of gives the edit as the key's value.
        self.assert_printed(
            "fix: set tool.package_url in your build settings file to the address of the tool's binary, from "
            "wherever your organization keeps it"
        )
        self.assert_not_printed(
            "fix: set tool.package_url in your build settings file to the address of the tool's source, from "
            "wherever you like",
            "the address of the tool's source, from wherever you like",
        )

    def test_a_command_of_a_fix_or_a_next_line(self):
        self.assert_printed(
            "fix: run ./rkmon confirm --settings <file>",
            "fix: run rkmon login, then rkmon start",
            "next: rkmon --image-code check --settings <file>",
            "next: docker/rkmon doctor",
        )
        self.assert_not_printed("fix: run ./rkmon launch --now", "rkmon launch --now")

    def test_a_next_line_s_step(self):
        self.assert_printed(
            # each path's steps, with their commands or one of their others
            "next: step 3, confirm: ./rkmon confirm --settings ~/rkmon-deploy.yaml",
            "next: step 3, build: rkmon build --settings <file>",
            "next: step 4, image: rkmon image pull",
            "next: step 5, ship: rkmon image push --settings <file>",
            "next: step 8, first tick: rkmon first-tick",
            "next: step 4, image",
        )
        self.assert_not_printed("next: step 9, launch", "step 9, launch")
        self.assert_not_printed("next: step 4, start: rkmon start", "step 4, start")
        self.assert_not_printed(
            "next: step 3, confirm: rkmon start", "step 3, confirm: rkmon start"
        )

    def test_a_note(self):
        self.assert_printed("note: the image is older than the package")
        self.assert_not_printed(
            "note: the image is newer than the package",
            "the image is newer than the package",
        )

    def test_how_the_code_s_texts_are_read(self):
        self.assert_printed(
            # the program's name alone, where a path to it is written
            "REFUSED: start it with ./rkmon start first",
            # a field str.format fills in
            "FAILED: 2 of 5 checks failed (fix each FAIL above)",
            # a text the code joins with +
            "FAIL settings file: the file /srv/x.yaml is outside the package folder",
        )

    def test_a_part_of_a_text_and_texts_put_together(self):
        self.assert_printed(
            # an ellipsis at either end for the rest of a text, with values of the quote's own
            "FAIL disk: the disk has 12 GiB free, less than ...",
            "FAIL disk: ... GiB free, less than the 30 GiB the image needs",
            # texts of the code put in another's value, each found whole
            "WARN image: the image is older than the package, the disk is nearly full",
            # a text in parentheses, read on its own
            "FAIL logins: 2 logins are not set (the read login is missing)",
        )
        self.assert_not_printed(
            "FAIL disk: the disk has 12 GiB free, more than ...",
            "the disk has 12 GiB free, more than ...",
        )
        self.assert_not_printed(
            "WARN image: the image is older than the package, the disk is broken",
            "the image is older than the package, the disk is broken",
        )
        self.assert_not_printed(
            "FAIL logins: 2 logins are not set (the read login was lost)",
            "2 logins are not set (the read login was lost)",
        )

    def test_a_path_of_the_quote_s_own_is_one_word(self):
        # A path and one word are too short to tell, as a value and one word are.
        self.assert_printed("FAIL private file: ~/rkmon-build.yaml exists")
        self.assert_not_printed(
            "FAIL private file: ~/rkmon-build.yaml is not one of yours",
            "~/rkmon-build.yaml is not one of yours",
        )


class TestLinks(Tree):
    FILES = {
        "README.md": (
            '# Top\n\n## The admin dashboard\n\n## Twice\n\n## Twice\n\n<a name="kept"></a>\n\n'
            "```\n## Not a heading\n```\n"
        ),
        "docs/restart.md": "## The stop levels\n",
    }

    def links(self, rel, source):
        write(self.root, {rel: source})
        return sorted(p.what for p in dl.links(dl.texts(self.root, rel), self.root))

    def test_a_link_and_a_doc_s_heading_that_reach_their_place(self):
        source = (
            "## A heading\n\n"
            "[a](../README.md#the-admin-dashboard) [b](../README.md#twice-1) [c](../README.md#kept) [d](#a-heading)\n"
            "[e](https://example.com/x#y) [f](restart.md) `[g](nope.md)`\n"
            'See README.md, "The admin dashboard", and restart.md, "The stop levels".\n'
        )
        self.assertEqual(self.links("docs/FAQ.md", source), [])

    def test_a_link_or_a_doc_s_heading_that_does_not(self):
        source = (
            "[a](../nope.md) [b](../README.md#no-such) [c](../../x.md) [d](../README.md#not-a-heading)\n"
            'See README.md, "No such part", and nope.md, "A part".\n'
        )
        self.assertEqual(
            self.links("docs/FAQ.md", source),
            sorted(
                [
                    "the link ../nope.md: the package has no nope.md",
                    "the link ../README.md#no-such: README.md has no heading whose anchor is #no-such",
                    "the link ../../x.md leaves the package",
                    "the link ../README.md#not-a-heading: README.md has no heading whose anchor is #not-a-heading",
                    'README.md, "No such part": README.md has no such heading',
                    'nope.md, "A part": the package has no nope.md',
                ]
            ),
        )
        self.assertEqual(
            self.links("rkmon_lib/x.py", "print('see README.md, \"No such part\"')\n"),
            ['README.md, "No such part": README.md has no such heading'],
        )

    def test_github_s_anchors(self):
        self.assertEqual(
            dl.slug("The container's wrapper, `docker/rkmon`"),
            "the-containers-wrapper-dockerrkmon",
        )
        self.assertEqual(
            dl.slug("4.9 Priorities and develop fixes"),
            "49-priorities-and-develop-fixes",
        )
        self.assertEqual(dl.slug("The stop-levels"), "the-stop-levels")
        self.assertEqual(
            dl.anchors(self.root / "README.md"),
            {"top", "the-admin-dashboard", "twice", "twice-1", "kept"},
        )


class TestPlaceholders(Tree):
    FILES = {
        "README.md": (
            "# Top\n\n## Glossary and placeholders\n\n| Placeholder | Meaning |\n|---|---|\n"
            "| `<deploy-host>` | the host you deploy on |\n\n## After\n\n`<later>`\n"
        ),
        "docker/Dockerfile": (
            "FROM base\n"
            "RUN --mount=type=secret,id=package_mirror \\\n"
            "    sh pkgs.sh install curl\n"
            "RUN --mount=type=secret,id=python_package_index pip install x\n"
        ),
    }

    def placeholders(self, rel, source):
        return problems(dl.placeholders, rel, source, dl.load_placeholders(self.root))

    def test_the_glossary_s_and_the_agentic_files(self):
        defined = dl.load_placeholders(self.root)
        self.assertIn("deploy-host", defined)
        self.assertIn("WORKING_DIR", defined)
        self.assertNotIn("later", defined)
        fine = "ssh <deploy-host> to <WORKING_DIR>, a<br>b, <details>x</details>, `<deploy-host>`\n"
        self.assertEqual(self.placeholders("QUICK_DEPLOY_GUIDE.md", fine), [])

    def test_a_build_secret_s_masked_name(self):
        # A masked line of a build tool shows a download location as its build secret's id, which the build file
        # mounts; a package without that file has none.
        self.assertEqual(
            dl.masked_names(self.root), {"package_mirror", "python_package_index"}
        )
        self.assertEqual(dl.masked_names(self.root / "docs"), frozenset())
        fine = (
            "`E: Failed to fetch <package_mirror>/dists/noble/InRelease`, then "
            "`pip: no index at <python_package_index>/simple/`\n"
        )
        self.assertEqual(self.placeholders("QUICK_BUILD_GUIDE.md", fine), [])
        self.assertEqual(
            self.placeholders("QUICK_BUILD_GUIDE.md", "`<rocm_package_url>`\n"),
            [
                '<rocm_package_url> is not in the glossary (README.md, "Glossary and placeholders")'
            ],
        )

    def test_a_placeholder_the_glossary_lacks(self):
        self.assertEqual(
            dl.placeholders(
                dl.read("QUICK_DEPLOY_GUIDE.md", "ssh <host>\n"),
                dl.load_placeholders(self.root),
            ),
            [
                dl.Problem(
                    "QUICK_DEPLOY_GUIDE.md",
                    1,
                    "placeholders",
                    '<host> is not in the glossary (README.md, "Glossary and placeholders")',
                )
            ],
        )
        for rel, source in (
            ("agentic/skills/x/SKILL.md", "<org>/<repo>"),
            ("rkmon_lib/x.py", 'print("<org>/<repo>")\n'),
        ):
            with self.subTest(rel=rel):
                self.assertEqual(len(self.placeholders(rel, source)), 2)
        self.assertEqual(self.placeholders("docs/notes.md", "<org>/<repo>"), [])

    def test_a_regular_expression_s_group_name_is_no_placeholder(self):
        source = (
            'ERROR = re.compile(r"^(?P<file>[^:]+):(?P<line>\\d+)")\nprint("<file>")\n'
        )
        self.assertEqual(
            self.placeholders("tools/x.py", source),
            ['<file> is not in the glossary (README.md, "Glossary and placeholders")'],
        )

    def test_an_allowance_keeps_the_names_it_matches_in_its_files_alone(self):
        write(
            self.root,
            {
                dl.PLACEHOLDERS: (
                    "allowed:\n  - files: 'agentic/.+\\.md'\n    keep: '<STATUS>'\n"
                    "    why: a form's slot\n"
                )
            },
        )
        allowed = dl.load_placeholder_allowances(self.root)
        defined = dl.load_placeholders(self.root)
        text = "`<STATUS> at <sha>`, then <STATUS>\n"
        for rel, want in (
            ("agentic/skills/x/SKILL.md", ["<sha>"]),
            ("QUICK_USER_GUIDE.md", ["<STATUS>", "<sha>", "<STATUS>"]),
        ):
            with self.subTest(rel=rel):
                found = problems(dl.placeholders, rel, text, defined, allowed)
                self.assertEqual([w.split(" ", 1)[0] for w in found], want)


class TestTheLists(Tree):
    def test_a_list_that_is_wrong_is_refused_naming_it(self):
        for name, text, load, want in (
            (
                dl.WORDS,
                "x:\n  words: [a]\n",
                dl.load_words,
                "tests/doc_words.yaml, x: say is required",
            ),
            (
                dl.WORDS,
                "x:\n  say: s\n  colour: red\n",
                dl.load_words,
                "tests/doc_words.yaml, x: no key colour",
            ),
            (
                dl.WORDS,
                "x:\n  say: s\n  allowed:\n    - keep: a\n",
                dl.load_words,
                "tests/doc_words.yaml, x: why is required",
            ),
            (
                dl.OLD_COMMANDS,
                "x:\n  instead: y\n  pattern: z\n  allowed:\n    - kind: [doc]\n      why: w\n",
                dl.load_old_commands,
                "tests/doc_old_commands.yaml, x: an allowance names no kind but user, maintainer, skill, message",
            ),
            (
                dl.OLD_COMMANDS,
                "rkmon guide:\n  instead: rkmon help\n  pattern: rkmon guide\n",
                dl.load_old_commands,
                "tests/doc_old_commands.yaml, rkmon guide: it is in rkmon_lib/commands.py's GONE",
            ),
            (
                dl.HOMES,
                "x:\n  home: README.md\n",
                dl.load_homes,
                "tests/doc_fact_homes.yaml, x: signatures is required",
            ),
            (
                dl.KEYS,
                "allowed:\n  - why: w\n",
                dl.load_key_allowances,
                "tests/doc_keys.yaml: files is required",
            ),
            (
                dl.KEYS,
                "allowed:\n  - files: a\n",
                dl.load_key_allowances,
                "tests/doc_keys.yaml: why is required",
            ),
            (
                dl.KEYS,
                "kept:\n  - files: a\n    why: w\n",
                dl.load_key_allowances,
                "tests/doc_keys.yaml: no key kept is known here",
            ),
            (
                dl.PLACEHOLDERS,
                "allowed:\n  - files: a\n    why: w\n",
                dl.load_placeholder_allowances,
                "tests/doc_placeholders.yaml: keep is required",
            ),
        ):
            with self.subTest(text=text):
                write(self.root, {name: text})
                with self.assertRaises(ValueError) as caught:
                    load(self.root)
                self.assertIn(want, str(caught.exception))


class TestCommandLine(Tree):
    FILES = {
        "QUICK_BUILD_GUIDE.md": "It holds one.\n",
        "QUICK_USER_GUIDE.md": "It keeps one.\n",
        "README.md": "x\n",
    }

    def main(self, argv, data=ROOT):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = dl.main(argv, root=self.root, data=data)
            except SystemExit as e:
                rc = e.code
        return rc, out.getvalue(), err.getvalue()

    def test_each_problem_on_a_line_and_the_exit_code(self):
        guide = str(self.root / "QUICK_BUILD_GUIDE.md")
        self.assertEqual(
            self.main(["--lint", "words", guide]),
            (
                1,
                'QUICK_BUILD_GUIDE.md:1: words: "holds" (rejected): write it in other words\n',
                "",
            ),
        )
        self.assertEqual(
            self.main(
                [
                    "--lint",
                    "words",
                    "--lint",
                    "keys",
                    str(self.root / "QUICK_USER_GUIDE.md"),
                ]
            ),
            (0, "", ""),
        )
        self.assertEqual(self.main(["--lint", "keys", guide]), (0, "", ""))

    def test_a_usage_error_or_a_file_it_cannot_read(self):
        rc, _out, err = self.main(["--lint", "spelling", str(self.root / "README.md")])
        self.assertEqual(rc, 2)
        self.assertIn("invalid choice: 'spelling'", err)
        rc, _out, err = self.main([str(ROOT / "README.md")])
        self.assertEqual(
            (rc, err.endswith("not a file of the package\n")), (2, True), err
        )
        rc, out, err = self.main(["--lint", "words", str(self.root / "nope.md")])
        self.assertEqual((rc, out), (2, ""))
        self.assertTrue(err.startswith("doc_lints.py: "), err)
        rc, out, err = self.main(
            [str(self.root / "README.md")], data=None
        )  # no lists in this package
        self.assertEqual((rc, out), (2, ""))
        self.assertIn("doc_fact_homes.yaml", err)
        with self.assertRaises(ValueError):
            dl.run(self.root, ["README.md"], ("spelling",), ROOT)


class TestThePackage(unittest.TestCase):
    def test_the_files_listed_pass_every_lint(self):
        lines = FILES.read_text(encoding="utf-8").splitlines()
        files = [ln for ln in lines if ln.strip() and not ln.startswith("#")]
        self.assertEqual(
            files, sorted(set(files)), "one path to a line, each once, in sorted order"
        )
        # The image leaves the setup skill out, so there the test skips; a checkout without a listed file fails it.
        not_here = build_context.left_out_here(*files)
        if not_here:
            self.skipTest(not_here)
        every = set(dl.package_files(ROOT))
        for rel in files:
            self.assertIn(rel, every, "a file of the package")
        self.assertEqual([str(p) for p in dl.run(ROOT, files)], [])

    def test_every_markdown_file_is_listed(self):
        """Every Markdown file of the package is on tests/doc_lint_files.txt, so the suite runs every lint over it."""
        listed = FILES.read_text(encoding="utf-8").splitlines()
        docs = [rel for rel in dl.package_files(ROOT) if rel.endswith(".md")]
        self.assertTrue(docs)
        self.assertEqual([rel for rel in docs if rel not in listed], [])

    def test_every_doc_quotes_what_the_code_prints(self):
        """The quoted-output lint over every doc of the package, listed or not: each outcome, check row, fix, next
        line and note a user doc quotes is one the code prints."""
        docs = [rel for rel in dl.package_files(ROOT) if rel.endswith(".md")]
        found = dl.run(ROOT, docs, ("quoted-output",))
        self.assertEqual([str(p) for p in found], [])

    def test_no_file_has_an_earlier_name_of_a_renamed_thing(self):
        self.assertEqual([str(p) for p in dl.every_file()], [])
        for rel in (
            "rkmon.py",
            "docker/rkmon",
            "rkmon_lib/scaler.py",
            "README.md",
            "docs/internal_settings.md",
            "agentic/skills/rocke-monitor/SKILL.md",
            "config.defaults.json",
            "schema/deploy_settings.schema.json",
            "docker/entrypoint.sh",
            "docker/compose.yaml",
            "docker/Dockerfile",
            "rk",
        ):
            with self.subTest(rel=rel):
                self.assertTrue(dl.texts(ROOT, rel), "a file every_file reads")

    def test_each_fact_has_its_home_in_the_package(self):
        facts = dl.load_homes()
        self.assertEqual({f.name for f in facts}, FACTS)
        for f in facts:
            with self.subTest(fact=f.name):
                self.assertTrue((ROOT / f.home).is_file(), f.home)
                self.assertTrue(f.signatures)


class TestRefusals(unittest.TestCase):
    def test_each_refusal_s_lines_are_the_state_table_s(self):
        """A refusal's outcome line is its state's sentence; its fix and next lines are the state's, and
        it has no line of its own. Before deploy step 8 has passed since the start and after it.
        """
        for n, case in test_states.CASES.items():
            for since in (test_states.RAN, states.NOT_STARTED):
                with self.subTest(state=n, since=since):
                    f = test_states.facts(**case)
                    state = states.current(f)
                    _rc, text = test_states.refused(state, f, since=since)
                    want = wrapping.wrap(
                        outcome.line(
                            "REFUSED", f"{states.sentence(state, f)}; nothing changed"
                        ),
                        outcome.WIDTH,
                    )
                    if state.fix and state.fix(f) is not None:
                        want += outcome.fix_lines(state.fix(f))
                    want += wrapping.wrap(
                        states.next_line(state, f, since), outcome.WIDTH
                    )
                    self.assertEqual(text.splitlines(), want)


if __name__ == "__main__":
    unittest.main()
