# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The settings templates the package gives you (rkmon_lib/settings_templates.py): every template, named by its file
type and profile, its text the one settings_file writes, which docker/rkmon prints and the privacy check reads; each
text the golden one (tests/settings_template_texts.py, a generated file: tools/generated.py).
And what the section command prints: each section, nested ones included, as the every-key text writes it, at its depth
in the file; pasted over a feature that a profile writes as its switch alone, and over a nested section that a missing
key's row names, it leaves a file
that passes its checks once its placeholders are filled in; with the feature then turned on, one that fails
only on the keys Required when it is on, until they are given. The section command's row in the command table, and
what docker/rkmon prints for it: a section alone, the list of sections, or a usage error that names the closest.
And the template commands' write: a new file, yours only, then WROTE and the next step (its command alone
once deploy step 8 has passed after a start); or a refusal, then
REFUSED and one next line: the deployment state's on the deploy side, your command again on the build side. Under
--dry-run, every guard and nothing written. And docker/rkmon's template commands (step 1 of each path): the
step's banner, then the write's lines; `-` prints the text alone; a command line it cannot read is a usage error.
"""

import io
import os
import pathlib
import re
import subprocess
import stat
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "tools"))
sys.dont_write_bytecode = True

import generated  # noqa: E402
import home_repository  # noqa: E402
import settings_template_texts as golden  # noqa: E402
import template_texts  # noqa: E402
import test_settings_schemas as schemas  # noqa: E402
import test_states  # noqa: E402  (its facts of each state)
from rendered import finished  # noqa: E402
from rkmon_lib import (  # noqa: E402
    coherence,
    commands,
    file_types,
    private_files,
    program,
    settings_file,
    settings_templates,
    states,
    steps,
)
from rkmon_lib.checks import Command, Status  # noqa: E402

T = settings_templates


def setUpModule():
    # The repository a new file fills in stands for the home repository: its repository defaults apply.
    home_repository.stand_in(
        schemas.FILLED["repository.name"], schemas.FILLED["repository.folder"]
    )


# The deploy file type, its templates' image.tag the golden texts' VERSION rather than this package's own.
DEPLOY = template_texts.pinned(file_types.DEPLOY, golden.VERSION)
RGT = schemas.RGT
CLUSTER = f"{RGT}.cluster"
# The one section the every-key text comments out after its comment: delivery "publish" does not choose it.
SERVE = f"{schemas.TEAM}.serve"
# Check 2 in a profile: a feature that, its section pasted and turned on, needs one the profile has off
# (coherence.NEEDS). Once nothing else fails, check 2's row alone says so; with the section of that feature pasted too,
# the one the profile writes as its switch alone, and that feature on, the file passes.
NEEDS_OFF = {
    ("review-only", "features.alerts.develop_breaks"): schemas.DG,
    ("merge-queue", "features.merge_queue.drift_notices"): schemas.RA,
    ("merge-queue", f"{schemas.TICKETS}.follow_up_tickets"): schemas.RA,
}
# The values the remote GPU tests need while they are on, as you would write them: invented.
GPU_TESTS_ON = {
    f"{CLUSTER}.login_host": "login.cluster.example.com",
    f"{CLUSTER}.ssh_user": "example-user",
    f"{CLUSTER}.partition": "gpu",
    f"{CLUSTER}.accounts": ["example-account"],
    f"{CLUSTER}.gpu_types": {"gfx942": "mi300x"},
    f"{RGT}.node.python": "~/venvs/tests/bin/python",
}
# The keys Required when a feature is on that fail once its section, pasted, is turned on: the section's own,
# left blank, and those of the section around it, which a switch alone leaves out. Each with a value, invented.
_ALERTS_ON = {"features.alerts.mode": "prototype"}
_TICKETS_ON = {
    f"{schemas.TICKETS}.mode": "prototype",
    f"{schemas.TICKETS}.project_key": "EXAMPLE",
    "integrations.atlassian_cli.jira_url": "https://jira.example.com",
}
REQUIRED_ON = {
    RGT: GPU_TESTS_ON,
    schemas.TEAM: {
        f"{schemas.TEAM}.public_url": "https://dashboards.example.com/team/",
        f"{schemas.TEAM}.publish.web_host": "web.example.com",
        f"{schemas.TEAM}.publish.username": "example-user",
        f"{schemas.TEAM}.publish.site_folder": "public_html/team",
    },
    "features.alerts.develop_breaks": _ALERTS_ON,
    "features.alerts.needs_you": _ALERTS_ON,
    f"{schemas.TICKETS}.follow_up_tickets": _TICKETS_ON,
    f"{schemas.TICKETS}.found_issue_tickets": _TICKETS_ON,
}
# What is left once they are given: spend on, a model the passes use with no price shows its cost as unknown.
WARNS_ON = {f"{schemas.RA}.spend": [("spend", Status.WARN)]}


def rkmon(*argv) -> subprocess.CompletedProcess:
    """docker/rkmon, run on this host with ARGV from its own folder, as rkmon: the name its messages give it
    (rkmon_lib/program.py)."""
    return subprocess.run(
        [sys.executable, "-B", "rkmon", *argv],
        cwd=ROOT / "docker",
        capture_output=True,
        text=True,
        timeout=60,
        stdin=subprocess.DEVNULL,
    )


def span(lines: list, key: str):
    """(first, end): the LINES of a file (each with its newline) that the section KEY takes, its comment first, then
    its own line and every line indented under it; None when the file does not have it.
    """
    first, end, at = 0, len(lines), None
    for depth, part in enumerate(key.split(".")):
        pad = " " * (2 * depth)
        own = re.compile(re.escape(pad + part) + r":( |\n)")
        at = next((i for i in range(first, end) if own.match(lines[i])), None)
        if at is None:
            return None
        first = at + 1
        end = next(
            (i for i in range(first, end) if not lines[i].startswith(pad + "  ")), end
        )
    pad = " " * (2 * key.count("."))
    first = at
    while first and lines[first - 1].startswith(pad + "# "):
        first -= 1
    return first, end


def switch_alone(lines: list, key: str) -> bool:
    """Whether LINES, the section KEY takes, are its comment, then its switch alone, off, with the switch's comment."""
    pad = " " * (2 * key.count("."))
    keys = [line for line in lines if not line.lstrip().startswith("#")]
    name = key.rsplit(".", 1)[-1]
    return keys == [f"{pad}{name}:\n", f"{pad}  enabled: false\n"]


def alone(text: str, key: str) -> bool:
    """Whether the deploy file TEXT writes the section KEY as its switch alone, off."""
    lines = text.splitlines(keepends=True)
    at = span(lines, key)
    return bool(at) and switch_alone(lines[at[0] : at[1]], key)


def paste(text: str, key: str) -> str:
    """The deploy file TEXT, with the section command's text for KEY pasted over the lines that KEY takes."""
    lines = text.splitlines(keepends=True)
    first, end = span(lines, key)
    return (
        "".join(lines[:first])
        + settings_file.section(DEPLOY, key)
        + "".join(lines[end:])
    )


def turned_on(text: str, key: str) -> str:
    """The deploy file TEXT with the feature KEY turned on, as you would: its switch, among the lines KEY takes, set
    to true."""
    lines = text.splitlines(keepends=True)
    first, end = span(lines, key)
    switch = " " * (2 * key.count(".") + 2) + "enabled: "
    at = [i for i in range(first, end) if lines[i].startswith(switch)]
    if len(at) != 1:
        raise ValueError(f"{key} has {len(at)} switches of its own")
    lines[at[0]] = switch + "true\n"
    return "".join(lines)


def checked(path: pathlib.Path, text: str, values=None) -> tuple:
    """The deploy file TEXT, written at PATH and read: (its rows that do not pass; those of the same file once each key
    you fill in (a placeholder in the template) that it has, the profile's and a pasted section's alike, holds the
    value you would give it (test_settings_schemas.FILLED), and each dotted key of VALUES its value, the key there or
    left out of the section around it).
    """
    path.write_text(text, encoding="utf-8")
    doc = settings_file.read(path, DEPLOY)
    before = [r for r in settings_file.validate(doc) if r.status != Status.PASS]
    for dotted in schemas.to_fill(DEPLOY):
        if schemas.found(doc.values, dotted):
            schemas.fill(doc.values, dotted, schemas.FILLED[dotted])
    for dotted, value in (values or {}).items():
        *around, last = dotted.split(".")
        schemas.value_at(doc.values, ".".join(around))[last] = value
    after = [r for r in settings_file.validate(doc) if r.status != Status.PASS]
    return before, after


def named(row) -> list:
    """The keys a row that fails names: the one it is named by, or, for the row of a section's missing keys (named by
    the section's last part), the keys it says are missing."""
    missing = re.fullmatch(r".+ (?:is|are) on, and (.+) (?:is|are) missing", row.found)
    if missing and "." not in row.name:
        return re.split(r", | and ", missing.group(1))
    return [row.name]


class TestSettingsTemplates(unittest.TestCase):
    maxDiff = None

    def test_every_template_is_named_by_its_file_type_and_profile(self):
        names = [p.name for p in settings_file.profiles(file_types.DEPLOY)]
        self.assertEqual(names, ["full", "review-only", "merge-queue"])
        self.assertEqual(
            [(t.file_type, t.profile, t.every_key, t.label) for t in T.templates()],
            [(file_types.BUILD, None, False, "build")]
            + [(file_types.DEPLOY, name, False, "deploy " + name) for name in names]
            + [(file_types.DEPLOY, None, True, "deploy every key")],
        )

    def test_a_template_s_text_is_the_one_settings_file_writes(self):
        for t in T.templates():
            with self.subTest(template=t.label):
                self.assertEqual(
                    t.text(),
                    settings_file.template(t.file_type, t.profile, t.every_key),
                )

    def test_each_template_names_the_command_that_prints_it(self):
        """The build template's command and each profile's, with - as the file, and for the every-key text the section
        command: what the privacy check names for a template it cannot render."""
        p = program.name()
        self.assertEqual(
            [str(t.command()) for t in T.templates()],
            [f"{p} template build -"]
            + [
                f"{p} template deploy --profile {name} -"
                for name in ("full", "review-only", "merge-queue")
            ]
            + [f"{p} template section"],
        )
        self.assertEqual(
            str(T.Template(file_types.DEPLOY).command()), f"{p} template deploy -"
        )

    def test_each_template_is_its_golden_text(self):
        """The build template's text, the three profiles' templates and the every-key text, with the golden texts'
        VERSION as the deploy file's image.tag, which a template gives the package's own version. A text out of step
        names the refresh, which writes the golden texts again from the writer."""
        written = template_texts.texts(golden.VERSION)
        self.assertEqual(list(written), [t.label for t in T.templates()])
        self.assertEqual(list(written), list(golden.TEXTS))
        for label, text in written.items():
            with self.subTest(template=label):
                self.assertEqual(
                    text,
                    golden.TEXTS[label],
                    f"the writer's text is not the golden one: {generated.REFRESH} writes the golden texts "
                    "again; review the change",
                )

    def test_docker_rkmon_prints_each_template_as_written(self):
        """`rkmon template build -` and `rkmon template deploy [--profile NAME] -` print the text settings_file
        writes, alone, so the privacy check reads what you are given; with no profile, the schema's first.
        """
        cases = [(["template", "build", "-"], T.Template(file_types.BUILD))]
        cases += [
            (["template", "deploy", "--profile", t.profile, "-"], t)
            for t in T.templates()
            if t.profile
        ]
        cases += [(["template", "deploy", "-"], T.Template(file_types.DEPLOY, "full"))]
        for argv, t in cases:
            with self.subTest(argv=argv):
                p = rkmon(*argv)
                self.assertEqual(p.returncode, 0, p.stderr)
                self.assertEqual(p.stdout, t.text())

    def test_a_profile_the_schema_does_not_have_is_a_usage_error(self):
        p = rkmon("template", "deploy", "--profile", "fast", "-")
        self.assertEqual((p.returncode, p.stdout), (2, ""))
        self.assertIn(
            "rkmon template deploy: no profile fast; the profiles are full, review-only, merge-queue",
            p.stderr,
        )


class TestSections(unittest.TestCase):
    """What `rkmon template section KEY` prints: any section, in full, as the every-key text writes it."""

    maxDiff = None

    def test_each_section_is_the_every_key_text_s_at_its_depth(self):
        """Every section the command lists, nested ones included: its comment, then every key with its comment,
        indented as in the file, and nothing of it left out, as in the every-key text. The serve block, which the
        every-key text comments out after its comment, is printed without the "# " in front of its lines, since you
        asked for it.
        """
        every = golden.TEXTS["deploy every key"]
        keys = [key for key, _ in settings_file.sections(DEPLOY)]
        self.assertIn(CLUSTER, keys)
        self.assertIn(SERVE, keys)
        for key in keys:
            text = settings_file.section(DEPLOY, key)
            pad = " " * (2 * key.count("."))
            with self.subTest(key=key):
                self.assertTrue(text.startswith(pad + "# "), text)
                if key == SERVE:
                    self.assertNotIn("\n" + text, every)
                    comment, *rest = text.splitlines(keepends=True)
                    text = comment + "".join(
                        f"{pad}# {line[len(pad):]}" for line in rest
                    )
                start = every.index("\n" + text) + 1
                after = every[start + len(text) :].split("\n", 1)[0]
                self.assertFalse(after.startswith(pad + " "), after)

    def needs_met(self, path, text, on, key, needed, values=None) -> str:
        """ON, the deploy file TEXT with the feature KEY's section pasted and KEY on, read at PATH once its
        placeholders are filled in and VALUES given: check 2's row alone, KEY needing NEEDED, which TEXT has off.
        Returned: ON with the section around NEEDED that TEXT writes as its switch alone pasted too, and NEEDED on, as
        the row's fix has you turn it on."""
        self.assertIn((key, needed), coherence.NEEDS)
        self.assertEqual(
            [(r.status, r.found) for r in checked(path, on, values)[1]],
            [(Status.FAIL, f"{key} is on, and it needs {needed}, which is off")],
        )
        parts = needed.split(".")
        around = [
            section
            for section in (".".join(parts[: i + 1]) for i in range(len(parts)))
            if alone(text, section)
        ]
        self.assertEqual(len(around), 1, needed)
        return turned_on(paste(on, around[0]), needed)

    def test_each_section_pasted_over_its_switch_alone_passes(self):
        """In each profile, each feature written as its switch alone, with the section's text pasted over it,
        its comment and all. The file fails only on the keys you fill in, once each, the checks across keys waiting
        for them; once they are filled in, the section's and the profile's alike, it passes its checks, those across
        keys included. Then the feature turned on (enabled: true), as you would: it fails only on the keys
        Required when it is on (REQUIRED_ON), each once, as their descriptions' closing sentences say; once they are
        given, the file passes, but for what WARNS_ON leaves. A feature that, on, needs one the profile has off
        (NEEDS_OFF) has check 2's row alone once nothing else fails; with the section of that feature pasted too, and
        that feature on, the rest holds as above."""
        closing = {
            dotted: settings_file.closing_of(node.get("description") or "")[1]
            for dotted, node, _ in settings_file.schema_keys(DEPLOY.load())
        }
        pasted, met = {}, set()
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "settings.yaml"
            for profile in settings_file.profiles(DEPLOY):
                text = settings_file.template(DEPLOY, profile.name)
                for key, _ in settings_file.sections(DEPLOY):
                    if not alone(text, key):
                        continue
                    pasted.setdefault(profile.name, []).append(key)
                    with self.subTest(profile=profile.name, key=key):
                        printed = paste(text, key)
                        before, after = checked(path, printed)
                        self.assertEqual(
                            schemas.open_rows(before),
                            (schemas.to_fill(DEPLOY), schemas.WAITING),
                            before,
                        )
                        on = turned_on(printed, key)
                        needed = NEEDS_OFF.get((profile.name, key))
                        # The section's text writes the feature on: check 2's row is there already.
                        if needed is not None and on == printed:
                            met_off = self.needs_met(path, text, printed, key, needed)
                            after = checked(path, met_off)[1]
                        self.assertEqual(after, [])
                        required = REQUIRED_ON.get(key, {})
                        if needed is not None:
                            met.add((profile.name, key))
                            on = self.needs_met(path, text, on, key, needed, required)
                        rows = checked(path, on)[1]
                        fails = [
                            k for r in rows if r.status == Status.FAIL for k in named(r)
                        ]
                        self.assertEqual(sorted(fails), sorted(required), rows)
                        for k in fails:
                            self.assertTrue(
                                closing[k].startswith("Required when "), (k, closing[k])
                            )
                        rows = checked(path, on, required)[1]
                        self.assertEqual(
                            [(r.name, r.status) for r in rows], WARNS_ON.get(key, [])
                        )
        self.assertEqual(met, set(NEEDS_OFF))
        both = [
            "features.dashboards.team",
            "features.alerts.develop_breaks",
            "features.alerts.needs_you",
            "features.ticket_creation.follow_up_tickets",
            "features.ticket_creation.found_issue_tickets",
            "features.weekly_report",
        ]
        reviews = [RGT, f"{schemas.RA}.spend"]
        self.assertEqual(
            pasted,
            {
                "full": reviews + ["features.merge_queue.auto_merge"] + both,
                "review-only": reviews + ["features.merge_queue"] + both,
                "merge-queue": [
                    schemas.RA,
                    "features.merge_queue.auto_merge",
                    "features.merge_queue.drift_notices",
                ]
                + both,
            },
        )
        everywhere = {key for keys in pasted.values() for key in keys}
        self.assertLessEqual(set(REQUIRED_ON) | set(WARNS_ON), everywhere)

    def test_a_nested_section_a_row_names_pasted_over_it_passes(self):
        """The remote GPU tests turned on, in each profile that writes them as their switch alone: their section pasted
        over it with its switch on, and its cluster block left empty. The missing keys' one row names the nested
        section, its fix the section command for it; its text, at its depth in the file, pasted over the empty block,
        leaves a file that passes its checks once the values the tests need while on are given.
        """
        on = settings_file.section(DEPLOY, RGT)
        self.assertEqual(on.count("\n      enabled: false\n"), 1)
        on = on.replace("\n      enabled: false\n", "\n      enabled: true\n")
        cluster = settings_file.section(DEPLOY, CLUSTER)
        self.assertTrue(cluster.startswith("      # Cluster: "))
        self.assertIn("\n      cluster:\n        # ", cluster)
        self.assertEqual(on.count(cluster), 1)
        empty = on.replace(cluster, "      cluster: {}\n")
        command = str(
            steps.command_line(file_types.DEPLOY.section_command, KEY=CLUSTER)
        )
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / "settings.yaml"
            for profile in ("full", "review-only"):
                with self.subTest(profile=profile):
                    lines = settings_file.template(DEPLOY, profile).splitlines(
                        keepends=True
                    )
                    first, end = span(lines, RGT)
                    self.assertTrue(switch_alone(lines[first:end], RGT))
                    text = "".join(lines[:first]) + empty + "".join(lines[end:])
                    rows = checked(path, text)[1]
                    self.assertEqual(
                        [(r.name, r.status) for r in rows],
                        [("cluster", Status.FAIL)] + schemas.WAITING,
                    )
                    self.assertIn(command + ", paste its text over", str(rows[0].fix))
                    self.assertEqual(text.count("\n      cluster: {}\n"), 1)
                    text = text.replace("\n      cluster: {}\n", "\n" + cluster)
                    # The checks across keys then ask for the values the tests need while they are on.
                    rows = checked(path, text)[1]
                    self.assertEqual(sorted(r.name for r in rows), sorted(GPU_TESTS_ON))
                    for row in rows:
                        self.assertEqual(row.status, Status.FAIL)
                        self.assertTrue(row.found.endswith(f", and {RGT} is on"), row)
                    self.assertEqual(checked(path, text, GPU_TESTS_ON)[1], [])


class TestSectionCommand(unittest.TestCase):
    """`rkmon template section [KEY]`: its row in the command table, and what docker/rkmon prints."""

    maxDiff = None

    def test_its_row_is_the_deploy_file_s_section_command(self):
        row = commands.find(("template", "section"), commands.TABLE)
        self.assertIs(row, T.SECTION)
        self.assertEqual(row.form, file_types.DEPLOY.section_command)
        self.assertEqual(row.group, "change")
        parse = commands.parser(commands.TABLE).parse_args
        ns = parse(["template", "section", CLUSTER])
        self.assertEqual((ns.row, ns.key), (row, CLUSTER))
        self.assertIsNone(parse(["template", "section"]).key)
        help_lines = commands.help_text(row).splitlines()
        self.assertIn("Usage:  rkmon template section [KEY]", help_lines)
        self.assertLessEqual(max(len(ln) for ln in help_lines), commands.WIDTH)

    def test_it_prints_a_section_alone(self):
        """No banner and no outcome line: the section's text as the every-key text writes it, a nested one at its
        depth."""
        for key in ("features.alerts", CLUSTER, SERVE, "model_bindings"):
            with self.subTest(key=key):
                p = rkmon("template", "section", key)
                self.assertEqual((p.returncode, p.stderr), (0, ""))
                self.assertEqual(
                    p.stdout, settings_file.section(file_types.DEPLOY, key)
                )

    def test_with_no_key_it_lists_each_section_s_key_and_title(self):
        p = rkmon("template", "section")
        self.assertEqual((p.returncode, p.stderr), (0, ""))
        found = settings_file.sections(file_types.DEPLOY)
        column = 2 + max(len(key) for key, _ in found)
        self.assertEqual(
            p.stdout.splitlines(),
            [f"{key.ljust(column)}{title}" for key, title in found],
        )
        self.assertEqual(p.stdout, T.section_text())
        self.assertIn(f"{CLUSTER}  ", p.stdout)

    def test_an_unknown_key_is_a_usage_error_that_names_the_closest_section(self):
        p = rkmon("template", "section", "features.alert")
        self.assertEqual((p.returncode, p.stdout), (2, ""))
        self.assertEqual(
            p.stderr,
            "rkmon template section: no section features.alert; did you mean features.alerts?\n",
        )
        # With none close to it: the command that lists them.
        out, err = io.StringIO(), io.StringIO()
        self.assertEqual(T.section_command("zzz", out, err), 2)
        self.assertEqual(out.getvalue(), "")
        self.assertEqual(
            err.getvalue(),
            "rkmon template section: no section zzz; rkmon template section lists the sections\n",
        )

    def test_its_help_and_its_usage_errors(self):
        p = rkmon("template", "section", "-h")
        self.assertEqual((p.returncode, p.stderr), (0, ""))
        self.assertEqual(p.stdout, commands.help_text(T.SECTION))
        p = rkmon("template", "section", "features.alerts", "features.merge_queue")
        self.assertEqual((p.returncode, p.stdout), (2, ""))
        self.assertEqual(
            p.stderr,
            "rkmon template section: unrecognized arguments: features.merge_queue\n",
        )


class TestWrite(unittest.TestCase):
    """`rkmon template build FILE` and `rkmon template deploy [--profile PROFILE] FILE` once parsed: the
    template as a new file, yours only, then the WROTE line and the next step's; or the refusal's rows, a blank line,
    REFUSED and one next line, with nothing written. Those lines show a file as settings_file.shown does, `~/…` in the
    home folder, where the test's folder is when TMPDIR is in there: the expectations name it so too. A command as you
    typed it names the file as you gave it."""

    maxDiff = None

    def setUp(self):
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        self.d = pathlib.Path(d.name)
        # A deployment with no settings confirmed (state 2), and its next line: no container
        # has started.
        self.facts = test_states.facts(**test_states.CASES[2])
        self.state_next = states.next_line(
            states.current(self.facts), self.facts, states.NOT_STARTED
        )

    def write(self, t: T.Template, file: str, package_dir=ROOT, **kwargs) -> tuple:
        """(exit code, what it printed) for template T written as FILE, by the command you would type for it (with
        --dry-run under a dry run); a deploy template's caller gives state 2's next line, unless another is given.
        """
        words = ["build"] if t.file_type is file_types.BUILD else ["deploy"]
        words += ["--profile", t.profile] if t.profile else []
        words += ["--dry-run"] if kwargs.get("dry_run") else []
        if t.file_type is file_types.DEPLOY:
            kwargs.setdefault("next_line", self.state_next)
        out = io.StringIO()
        argv = ["rkmon", "template", *words, file]
        rc = T.write(t, file, argv, package_dir, out, **kwargs)
        return rc, out.getvalue()

    def test_each_template_is_written_yours_only_with_the_values_to_fill_in(self):
        printed = []
        for t in [T.Template(file_types.DEPLOY)] + list(T.templates()):
            if t.every_key:
                continue
            with self.subTest(template=t.label, profile=t.profile):
                file = str(self.d / f"{t.label.replace(' ', '-')}-{t.profile}.yaml")
                rc, out = self.write(t, file)
                printed.append(out)
                self.assertEqual(rc, 0)
                text = pathlib.Path(file).read_text(encoding="utf-8")
                self.assertEqual(text, t.text())
                self.assertEqual(stat.S_IMODE(os.stat(file).st_mode), 0o600)
                marked = [
                    ln
                    for ln in text.splitlines()
                    if not ln.lstrip().startswith("#")
                    and settings_file.PLACEHOLDER.search(ln)
                ]
                deploy = t.file_type is file_types.DEPLOY
                profile = f"profile {t.profile or 'full'}; " if deploy else ""
                where = settings_file.shown(file)
                self.assertEqual(
                    out,
                    finished(
                        f"WROTE {where} ({profile}yours only, permissions 600): fill in the {len(marked)} values "
                        "marked <...>",
                        f"next: step 2, check: rkmon check --settings {where}",
                    ),
                )
        # The review-only file has 4 values to fill in.
        sample = "(profile review-only; yours only, permissions 600): fill in the 4 values marked <...> next: "
        self.assertEqual(sum(sample in " ".join(out.split()) for out in printed), 1)

    def test_the_next_steps_command_alone_once_deploy_step_8_has_passed(self):
        # Once this folder's records keep a first tick, which the caller reads, step 2's command alone.
        t = T.Template(file_types.DEPLOY, "review-only")
        file = str(self.d / "s.yaml")
        where = settings_file.shown(file)
        self.assertEqual(
            self.write(t, file, plain=True),
            (
                0,
                finished(
                    f"WROTE {where} (profile review-only; yours only, permissions 600): fill in the 4 values marked "
                    "<...>",
                    f"next: rkmon check --settings {where}",
                ),
            ),
        )

    def test_dash_prints_the_text_alone(self):
        for t in T.templates():
            if not t.every_key:
                with self.subTest(template=t.label):
                    self.assertEqual(self.write(t, "-"), (0, t.text()))
        self.assertEqual(list(self.d.iterdir()), [])

    def test_a_file_that_exists_is_refused_and_kept(self):
        """The row, a blank line, `REFUSED: nothing written`, then, on the deploy side, the deployment
        state's next line, state 2's here, which the caller gives."""
        self.assertEqual(states.current(self.facts).n, 2)
        self.assertEqual(
            self.state_next,
            "next: step 1, template: rkmon template deploy ~/rkmon-deploy.yaml (with a file of your own: rkmon check "
            "--settings FILE)",
        )
        file = self.d / "deploy.yaml"
        file.write_text("mine\n", encoding="utf-8")
        rc, out = self.write(T.Template(file_types.DEPLOY, "review-only"), str(file))
        self.assertEqual(rc, 4)
        self.assertEqual(file.read_text(encoding="utf-8"), "mine\n")
        row, after = out.split("\n\n")
        where = settings_file.shown(file)
        other = settings_file.shown(self.d / "deploy-2.yaml")
        self.assertTrue(row.startswith(f"  FAIL  private file: {where} exists\n"), out)
        self.assertEqual(  # the next line wrapped as the rows are
            after,
            "REFUSED: nothing written\n"
            "next: step 1, template: rkmon template deploy ~/rkmon-deploy.yaml (with a file of your own:\n"
            "rkmon check --settings FILE)\n",
        )
        self.assertIn(
            f"fix: rkmon template deploy --profile review-only {other}, or move that one aside first: "
            f"mv {where} {where}.old",
            " ".join(row.split()),
        )

    def test_a_refusal_shows_a_file_in_your_home_folder_as_its_own_lines_do(self):
        """The rows show a file in your home folder, and the paths beside it, as `~/…`, as the WROTE
        line and the next line do."""
        file = self.d / "rkmon-deploy.yaml"
        file.write_text("mine\n", encoding="utf-8")
        with mock.patch.dict(os.environ, {"HOME": str(self.d)}):
            rc, out = self.write(
                T.Template(file_types.DEPLOY, "review-only"), str(file)
            )
        self.assertEqual(rc, 4)
        self.assertEqual(file.read_text(encoding="utf-8"), "mine\n")
        row, after = out.split("\n\n")
        row, rules = row.rsplit("\n", 1)
        self.assertEqual(rules, private_files.RULES)  # the refusal's last line
        self.assertEqual(  # the row's words; where its lines break is the renderer's
            " ".join(row.split()),
            "FAIL private file: ~/rkmon-deploy.yaml exists fix: rkmon template deploy --profile review-only "
            "~/rkmon-deploy-2.yaml, or move that one aside first: mv ~/rkmon-deploy.yaml ~/rkmon-deploy.yaml.old",
        )
        self.assertEqual(
            after,
            "REFUSED: nothing written\n"
            "next: step 1, template: rkmon template deploy ~/rkmon-deploy.yaml (with a file of your own:\n"
            "rkmon check --settings FILE)\n",
        )

    def test_a_file_inside_the_package_folder_is_refused(self):
        """On the build side, a refusal's next line is your command again, as you typed it."""
        package = self.d / "package"
        package.mkdir()
        file = package / "rkmon-template-test-build.yaml"
        rc, out = self.write(T.Template(file_types.BUILD), str(file), package)
        self.assertEqual(rc, 4)
        self.assertFalse(file.exists())
        row, after = out.split("\n\n")
        flat = " ".join(row.split())
        where = settings_file.shown(file)
        self.assertIn(f"FAIL private file: {where} is inside the package folder", flat)
        self.assertIn("fix: rkmon template build ~/rkmon-template-test-build", flat)
        typed = Command(("rkmon", "template", "build", str(file)))
        self.assertEqual(after, finished("REFUSED: nothing written", f"next: {typed}"))

    def test_only_the_command_layer_gives_the_state_s_next_line(self):
        """A deploy template's write takes the state's next line from its caller, which gathers the state's facts; a
        build template's makes its own, your command again. Either the other way round is the caller's mistake, found
        before anything is written."""
        file = self.d / "s.yaml"
        for t, given in (
            (T.Template(file_types.DEPLOY, "full"), {}),
            (T.Template(file_types.BUILD), {"next_line": self.state_next}),
        ):
            with self.subTest(template=t.label):
                with self.assertRaises(TypeError):
                    T.write(
                        t,
                        str(file),
                        ["rkmon", "template", str(file)],
                        ROOT,
                        io.StringIO(),
                        **given,
                    )
                self.assertFalse(file.exists())
        # - prints the text alone, with no next line: none is needed.
        out = io.StringIO()
        deploy = T.Template(file_types.DEPLOY, "full")
        self.assertEqual(
            T.write(deploy, "-", ["rkmon", "template", "deploy", "-"], ROOT, out), 0
        )
        self.assertEqual(out.getvalue(), deploy.text())

    def test_a_dry_run_runs_every_guard_and_writes_nothing(self):
        """--dry-run: every guard runs, nothing is written, and it exits as the real run would. A file that can
        be written: PASSED, exit 0, then on the deploy side the state's next line, and on the build side step 1 to run
        for real. A file that cannot: the refusal, its retry the command as you typed it, --dry-run included.
        """
        deploy = T.Template(file_types.DEPLOY, "review-only")
        build = T.Template(file_types.BUILD)
        file = self.d / "new" / "s.yaml"
        where = settings_file.shown(file)
        kept = "yours only, permissions 600"
        self.assertEqual(
            self.write(deploy, str(file), dry_run=True),
            (
                0,
                finished(
                    f"PASSED: {where} can be written (profile review-only; {kept}); nothing written",
                    self.state_next,
                ),
            ),
        )
        self.assertEqual(
            self.write(build, str(file), dry_run=True),
            (
                0,
                finished(
                    f"PASSED: {where} can be written ({kept}); nothing written",
                    f"next: step 1, template: rkmon template build {where}",
                ),
            ),
        )
        self.assertEqual(list(self.d.iterdir()), [])
        mine = self.d / "mine.yaml"
        mine.write_text("mine\n", encoding="utf-8")
        taken = settings_file.shown(mine)
        typed = Command(("rkmon", "template", "build", "--dry-run", str(mine)))
        for t, again in ((deploy, self.state_next), (build, f"next: {typed}")):
            with self.subTest(template=t.label):
                rc, out = self.write(t, str(mine), dry_run=True)
                self.assertEqual(rc, 4)
                row, after = out.split("\n\n")
                self.assertTrue(
                    row.startswith(f"  FAIL  private file: {taken} exists\n"), out
                )
                self.assertIn(" --dry-run ", " ".join(row.split()))
                self.assertEqual(after, finished("REFUSED: nothing written", again))
        self.assertEqual(mine.read_text(encoding="utf-8"), "mine\n")
        self.assertEqual(list(self.d.iterdir()), [mine])

    def test_the_every_key_text_is_never_written(self):
        with self.assertRaises(ValueError):
            self.write(T.Template(file_types.DEPLOY, every_key=True), "-")


class TestTemplateCommand(unittest.TestCase):
    """docker/rkmon's `template build FILE` and `template deploy [--profile PROFILE] FILE`, step 1 of each path:
    the step's banner, then what settings_templates.write prints (TestWrite); a command line it cannot read is a
    usage error, said on stderr, with nothing written. On the deploy side a refusal's next line and a dry run's are the
    deployment state's, which docker is asked for: tests/test_deploy_path_confirm.py runs those, against its stand-in
    docker. The lines show a file as settings_file.shown does, `~/…` in the home folder, where the test's folder is
    when TMPDIR is in there: the expectations name it so too, in your command again as well.
    """

    maxDiff = None

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.d = pathlib.Path(tmp.name)

    def test_the_build_template_is_written_after_the_step_s_banner(self):
        """The banner, then the write's WROTE line and step 2's line, as write prints them for the same template."""
        t, f = T.Template(file_types.BUILD), self.d / "build.yaml"
        p = rkmon("template", "build", str(f))
        self.assertEqual((p.returncode, p.stderr), (0, ""), p.stdout)
        self.assertEqual(f.read_text(encoding="utf-8"), t.text())
        self.assertEqual(stat.S_IMODE(f.stat().st_mode), 0o600)
        other, printed = self.d / "other.yaml", io.StringIO()
        argv = ["rkmon", "template", "build", str(other)]
        self.assertEqual(T.write(t, str(other), argv, ROOT, printed), 0)
        self.assertEqual(
            p.stdout,
            steps.banner(steps.BUILD_STEPS[0])
            + "\n"
            + printed.getvalue().replace(
                settings_file.shown(other), settings_file.shown(f)
            ),
        )
        self.assertIn("\nnext: step 2, check: rkmon check --settings ", p.stdout)

    def test_a_file_that_exists_is_refused_and_left_as_it_is(self):
        """The banner, the private file's row, a blank line, `REFUSED: nothing written` and your command again as you
        typed it, --dry-run included, the file named as the row names it: exit 4.
        """
        f = self.d / "mine.yaml"
        f.write_text("mine\n", encoding="utf-8")
        where, other = settings_file.shown(f), settings_file.shown(
            self.d / "mine-2.yaml"
        )
        for dry in ([], ["--dry-run"]):
            with self.subTest(dry=dry):
                p = rkmon(*dry, "template", "build", str(f))
                self.assertEqual((p.returncode, p.stderr), (4, ""), p.stdout)
                self.assertEqual(f.read_text(encoding="utf-8"), "mine\n")
                banner, rest = p.stdout.split("\n", 1)
                self.assertEqual(banner, steps.banner(steps.BUILD_STEPS[0], bool(dry)))
                row, after = rest.split("\n\n")
                self.assertTrue(
                    row.startswith(f"  FAIL  private file: {where} exists\n"), p.stdout
                )
                typed = " ".join(["rkmon", *dry, "template", "build"])
                self.assertIn(
                    f"fix: {typed} {other}, or move that one aside first: mv {where} {where}.old",
                    " ".join(row.split()),
                )
                self.assertEqual(
                    after, f"REFUSED: nothing written\nnext: {typed} {where}\n"
                )
        self.assertEqual(os.listdir(self.d), ["mine.yaml"])

    def test_your_command_again_names_the_file_as_the_lines_do(self):
        """A file named from the folder you ran the command in, or in full in your home folder, as your shell writes
        ~/… out: the refusal's row, its fix and your command again on its next line each name it as
        settings_file.shown does, ~/… in your home folder."""
        home = self.d / "home"
        (home / "work").mkdir(parents=True)
        f = home / "work" / "mine.yaml"
        f.write_text("mine\n", encoding="utf-8")
        prog = str(ROOT / "docker" / "rkmon")
        name = program.name(prog)
        for typed, cwd in (("mine.yaml", home / "work"), (str(f), self.d)):
            with self.subTest(typed=typed):
                p = subprocess.run(
                    [sys.executable, "-B", prog, "template", "build", typed],
                    cwd=cwd,
                    env=dict(os.environ, HOME=str(home)),
                    capture_output=True,
                    text=True,
                    timeout=60,
                    stdin=subprocess.DEVNULL,
                )
                self.assertEqual((p.returncode, p.stderr), (4, ""), p.stdout)
                self.assertIn(
                    "\n  FAIL  private file: ~/work/mine.yaml exists\n", p.stdout
                )
                self.assertIn(
                    f"fix: {name} template build ~/work/mine-2.yaml, or move that one aside first: mv "
                    "~/work/mine.yaml ~/work/mine.yaml.old",
                    " ".join(p.stdout.split()),
                )
                self.assertTrue(
                    p.stdout.endswith(
                        f"REFUSED: nothing written\nnext: {name} template build ~/work/mine.yaml\n"
                    ),
                    p.stdout,
                )
        self.assertEqual(f.read_text(encoding="utf-8"), "mine\n")
        self.assertEqual(os.listdir(home / "work"), ["mine.yaml"])

    def test_a_dry_run_writes_nothing_and_names_the_real_run(self):
        f = self.d / "new" / "build.yaml"
        where = settings_file.shown(f)
        p = rkmon("template", "build", str(f), "--dry-run")
        self.assertEqual((p.returncode, p.stderr), (0, ""), p.stdout)
        self.assertEqual(os.listdir(self.d), [])
        self.assertEqual(
            p.stdout,
            steps.banner(steps.BUILD_STEPS[0], True)
            + "\n"
            + finished(
                f"PASSED: {where} can be written (yours only, permissions 600); nothing written",
                f"next: step 1, template: rkmon template build {where}",
            ),
        )

    def test_the_command_line(self):
        """One FILE (or `-`) after `build`, or after `deploy` and its optional --profile, a profile of the deploy
        schema's: anything else is a usage error (exit 2, said on stderr, nothing written).
        """
        f = str(self.d / "x.yaml")
        names = ", ".join(p.name for p in settings_file.profiles(file_types.DEPLOY))
        for argv, words in (
            (
                [],
                "rkmon template: name one of its commands: template build, template deploy, template section",
            ),
            (["deploy"], "the following arguments are required: FILE"),
            (["build", f, f + "2"], f"unrecognized arguments: {f}2"),
            (["build", "--tag", f], "unrecognized arguments: --tag"),
            (["image", f], 'unknown command "template image": rkmon help lists'),
            (["deploy", f, "--profile"], "argument --profile: expected one argument"),
            (
                ["build", "--profile", "full", f],
                "rkmon template build: unrecognized arguments: --profile",
            ),
            (
                ["deploy", "--profile", "nope", f],
                f"no profile nope; the profiles are {names}",
            ),
        ):
            with self.subTest(argv=argv):
                p = rkmon("template", *argv)
                self.assertEqual((p.returncode, p.stdout), (2, ""), p.stderr)
                self.assertTrue(p.stderr.startswith("rkmon template"), p.stderr)
                self.assertIn(words, p.stderr)
                self.assertEqual(os.listdir(self.d), [])
