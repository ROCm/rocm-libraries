# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The agentic files (agentic/: the package's own, changed where they are) and what the package does with them: the
committed files are templates that hold a placeholder for every site value and name the operator `<user_login>`, and
name only what the package has, each setting by its deploy settings key or its registry name; rendering them
(rkmon_lib/agentic.py) leaves no placeholder; installing backs up what it changes, removes what an earlier install
wrote that the render no longer has, keeps the workspace's own shared policies and merges the registry rows;
make_package builds the folder as it sits in the repository, with the agentic files as committed and a passing privacy
check report, and copies it only into a sandbox checkout (each build from a repository that holds the tests' copy of
the package, so the suite runs them in a copy that has no .git too).
"""

import json
import os
import pathlib
import re
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "tools")]
sys.dont_write_bytecode = True

import doc_lints  # noqa: E402
import sandbox  # noqa: E402
from rkmon_lib import (  # noqa: E402
    agentic,
    coherence,
    config,
    container_layout,
    deliver,
    deploy_tree,
    features,
    file_types,
    outward,
    paths,
    post,
    privacy,
    registry,
    settings_file,
    split,
    tickets,
    verifier_file,
)

BUNDLE = ROOT / "agentic"


def text(rel: str) -> str:
    """A committed agentic file's text."""
    return (BUNDLE / rel).read_text(encoding="utf-8")


def bundle_files() -> list[str]:
    """The committed agentic files, by their path under agentic/."""
    return sorted(
        f.relative_to(BUNDLE).as_posix() for f in BUNDLE.rglob("*") if f.is_file()
    )


def a_config(base: pathlib.Path, **over) -> dict:
    c = config.load_defaults()
    c.pop("_doc", None)
    c.update(
        repo="acme/widgets",
        rocke_path="a/b/rocke",
        operator={"github_login": "the-operator"},
        scope={"labels": ["x"]},
        **over,
    )
    c["paths"] = dict(c["paths"], working_dir=str(base / "work"))
    return config.normalize(c)


# A path that starts at a placeholder, written with backslashes: a Linux container reads only forward slashes.
PLACEHOLDER_PATH = re.compile(r"(<[A-Z_]+>)((?:\\[^\\\s`\"'|()<>\[\]*]+)+\\?)")
# A token, a word in capitals in angle brackets, is a placeholder the install fills in (agentic.PLACEHOLDERS) or one of
# the example words the files show in a command or a line's form, which the agent reads as it stands. A misspelt
# placeholder is neither: the install leaves it as written, and nothing is left for `agentic.left` to find.
TOKEN = re.compile(r"<[A-Z][A-Z0-9_]*>")
EXAMPLE_WORDS = frozenset(
    {
        "<N>",
        "<ID>",
        "<NAME>",
        "<NN>",
        "<R>",
        "<SAVE>",
        "<STATE>",
        "<STATUS>",
        "<TAB>",
        "<UTC>",
    }
)


def unknown_tokens(bundle: pathlib.Path) -> list[str]:
    """`file:line: token` for every token in the agentic files (`bundle`, the README with them) that is neither a
    placeholder nor one of EXAMPLE_WORDS."""
    return [
        f"{f.relative_to(bundle).as_posix()}:{i}: {m.group(0)}"
        for f in sorted(bundle.rglob("*"))
        if f.is_file()
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1)
        for m in TOKEN.finditer(line)
        if m.group(0) not in agentic.PLACEHOLDERS and m.group(0) not in EXAMPLE_WORDS
    ]


# --- what the agentic files may name ----------------------------------------------------------------------------------
# A live skill's own fill-in tokens (lower case: a live workspace's settings file fills them by hand) and that
# settings file, `.claude/project_config.md`, which a deployment does not have: the package keeps those settings in
# its settings layers, and `agentic.render` fills the placeholders. The agentic files hold none of them.
DEAD_POINTER = re.compile(
    r"<(?:paths\.[a-z_]+|wsl_[a-z_]+)>|(?<![\w/])\.claude/project_config\.md|\bproject_config\.md\b"
)
# The skill and the agent files tell the session which `rk` commands to run, which tick lines to act on and which
# `tools/` scripts exist. A name the package lacks would send the session to nothing.
CODE_SPAN = re.compile(r"`([^`\n]+)`")
RK_CMD = re.compile(r"(?:(?<![\w./<>-])rk|<RK>)\s+([a-z][a-z0-9]*(?:-[a-z0-9]+)*)\b")
ALERT_LINE = re.compile(r"!!\s+([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)\b")
LINE_WORD = re.compile(r"^(?:!!\s+)?([A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+)(?=\s|$)")
# A tools/ script of the package or the worker tools: bare, or under a placeholder; never a path inside rocKE's tree
# (`platform/tests/tools/...`). Its whole file name counts: `tools/x.sh` is not `tools/x.py`.
TOOL_FILE = re.compile(
    r"(?:(?<=<PACKAGE_DIR>/)|(?<=<TOOLKIT_DIR>/)|(?<=<WORKING_DIR>/)|(?<![\w./-]))tools/([\w-]+)\.(py|sh)\b"
)
# Names the agentic files give that the package does not hold, each with why.
EXEMPT_NAMES: dict[str, str] = {
    "ON_HOLD": "a PR label (merge_procedure.md's hold labels), not a tick line",
    "GOLDEN_GAP": "a line the verifier prints itself (lessons L-532), not a tick line",
    "PYTEST_ADDOPTS": "pytest's own environment variable (a lesson on Windows lanes), not a tick line",
}


def agentic_texts(bundle: pathlib.Path) -> list[tuple[str, str]]:
    """(path under `bundle`, text) of each agentic file the checks read: every file but the README, which describes
    them."""
    return [
        (f.relative_to(bundle).as_posix(), f.read_text(encoding="utf-8"))
        for f in sorted(bundle.rglob("*"))
        if f.is_file() and f.name != "README.md"
    ]


def dead_pointers(bundle: pathlib.Path) -> list[str]:
    """`file:line: token` for every live fill-in token or settings-file reference in the agentic files (`bundle`)."""
    return [
        f"{rel}:{i}: {m.group(0)}"
        for rel, body in agentic_texts(bundle)
        for i, line in enumerate(body.splitlines(), 1)
        for m in DEAD_POINTER.finditer(line)
    ]


def code_texts(text: str) -> list[str]:
    """The code a markdown text shows: its code spans and the lines of its fenced blocks."""
    out, fenced = [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            out.append(line.strip())
        else:
            out += CODE_SPAN.findall(line)
    return out


def names_in(text: str) -> set[tuple[str, str]]:
    """(kind, name) for every `rk` command, tick line and tools/ script a file's text names."""
    found = set()
    for code in code_texts(text):
        for m in RK_CMD.finditer(code):
            found.add(("rk command", m.group(1)))
        m = LINE_WORD.match(code)
        if m:
            found.add(("tick line", m.group(1)))
    for m in ALERT_LINE.finditer(text):
        found.add(("tick line", m.group(1)))
    for m in TOOL_FILE.finditer(text):
        found.add(("tools/ script", f"{m.group(1)}.{m.group(2)}"))
    return found


def package_names(package: pathlib.Path) -> dict:
    """What the package has: its `rk` subcommands (rk --help), its code's text (the tick lines' keywords are in it),
    and the file names of its tools/ and worker/ scripts."""
    p = subprocess.run(
        [sys.executable, "-B", str(package / "rkmon.py"), "--help"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    m = re.search(r"\{([a-z0-9,-]+)\}", p.stdout)
    commands = set(m.group(1).split(",")) if m else set()
    code = [package / "rkmon.py", package / "rk", package / "docker" / "entrypoint.sh"]
    for d in ("rkmon_lib", "tools", "worker"):
        code += sorted((package / d).rglob("*.py"))
    text = "\n".join(
        f.read_text(encoding="utf-8", errors="replace") for f in code if f.is_file()
    )
    scripts = {
        f.name
        for d in ("tools", "worker")
        for f in (package / d).glob("*")
        if f.suffix in (".py", ".sh")
    }
    return {"commands": commands, "code": text, "scripts": scripts}


def missing_names(bundle: pathlib.Path, have: dict) -> list[str]:
    """`file: kind name` for every `rk` command, tick line or tools/ script the agentic files (`bundle`) name that the
    package lacks (`have`: package_names) and EXEMPT_NAMES does not list."""
    out = []
    word = {}
    for rel, body in agentic_texts(bundle):
        for kind, name in sorted(names_in(body)):
            if name in EXEMPT_NAMES:
                continue
            if kind == "rk command":
                ok = name in have["commands"]
            elif kind == "tick line":
                if name not in word:
                    word[name] = bool(re.search(rf"\b{name}\b", have["code"]))
                ok = word[name]
            else:
                ok = name in have["scripts"]
            if not ok:
                out.append(f"{rel}: {kind} {name}")
    return out


def stale_exemptions(bundle: pathlib.Path) -> list[str]:
    """Each name EXEMPT_NAMES lists that no agentic file (`bundle`) gives: an exemption nothing needs."""
    given = {
        name for _rel, body in agentic_texts(bundle) for _kind, name in names_in(body)
    }
    return sorted(set(EXEMPT_NAMES) - given)


class TestCommittedFiles(unittest.TestCase):
    """The committed agentic files: the skill, the team members with their agent types and registry rows, the shared
    policies the skill reads and the review book, as templates."""

    def test_the_files_are_the_package_s_own(self):
        files = bundle_files()
        for rel in (
            "README.md",
            "review_book.md",
            "agents/rocke-code-quality-reviewer.md",
            "agents/rocke-review-verifier.md",
            "skills/rocke-monitor/SKILL.md",
            "skills/rocke-monitor/commands.md",
            "skills/rocke-monitor/incident_procedure.md",
            "skills/rocke-monitor/lane_commands.md",
            "skills/rocke-monitor/lessons.md",
            "skills/rocke-monitor/lessons_for.py",
            "skills/rocke-monitor/merge_procedure.md",
            "skills/rocke-monitor/offload.md",
            "skills/rocke-monitor/review_procedure.md",
            "skills/rocke-monitor/rules.md",
            "skills/rocke-monitor/weekly.md",
            "skills/shared/prerequisites.md",
            "skills/shared/temporary_file_policy.md",
            "team_members/rocke_code_quality_reviewer.md",
            "team_members/rocke_review_verifier.md",
            agentic.REGISTRY_ROWS,
        ):
            self.assertIn(rel, files)
        self.assertFalse([f for f in files if ".bak" in f], files)

    def test_the_agent_types_take_their_model_and_effort_from_the_settings(self):
        for rel, want in (
            (
                "agents/rocke-review-verifier.md",
                "model: <AUDIT_MODEL>\neffort: <VERIFY_EFFORT>\n",
            ),
            (
                "agents/rocke-code-quality-reviewer.md",
                "model: <DEEP_MODEL>\neffort: <QUALITY_EFFORT>\n",
            ),
        ):
            with self.subTest(rel=rel):
                agent = text(rel)
                self.assertIn(want, agent)
                self.assertEqual(agent.count("\neffort:"), 1, agent)

    def test_the_skill_runs_a_printed_rkmon_rk_command_as_its_own_rk(self):
        """The monitor's lines name the host's command, `rkmon rk CMD`, as the operator types it, but the session runs rk
        in the container: the CLI row says that such a line is `<RK> CMD` here, run only where the skill lets it, and
        that any other rkmon command is the operator's, on the host, to pass on."""
        (row,) = [
            line
            for line in text("skills/rocke-monitor/SKILL.md").splitlines()
            if line.startswith("| CLI |")
        ]
        self.assertIn("A printed `rkmon rk <command>`", row)
        self.assertIn(
            "is `<RK> <command>` here, run only where this skill lets you", row
        )
        self.assertIn(
            "any other `rkmon ...` command is <user_login>'s, on the host: pass it on",
            row,
        )

    def test_the_parallel_reviews_asked_for_are_read_from_rk_status(self):
        """No number of parallel reviews is rendered into the session's files: it would be the one asked for when the
        session started, while the setting can change under it and the scaler moves the number in effect. The lines
        name the setting, and `rk status` says how many run and why."""
        self.assertNotIn("<PARALLEL_REVIEWS>", agentic.PLACEHOLDERS)
        for rel in (
            "skills/rocke-monitor/SKILL.md",
            "skills/rocke-monitor/review_procedure.md",
        ):
            with self.subTest(rel=rel):
                body = text(rel)
                self.assertNotIn("<PARALLEL_REVIEWS>", body)
                self.assertIn("`features.review_automation.parallel_reviews`;", body)

    def test_placeholders_stand_for_the_site_values(self):
        skill = text("skills/rocke-monitor/SKILL.md")
        for want in (
            "| CLI | `<RK> <command>`",
            "JIRA tickets go to `<JIRA_PROJECT>`",
        ):
            self.assertIn(want, skill)
        self.assertIn(
            "component <JIRA_COMPONENT>; labels <JIRA_LABELS>",
            text("skills/rocke-monitor/rules.md"),
        )

    def test_no_file_names_the_install_without_a_container(self):
        """The monitor runs in its container, and an install on a host runs no services: no placeholder for a WSL
        distribution or a Linux prefix, and no agentic file names a host install's services, their switches, a user's
        systemd units, the WSL keep-alive, WSL itself, wsl.exe or the tail of the Linux prefix (`… -- bash FILE`).
        Ollama's own unit stays named: its rule says never to restart it.
        """
        self.assertEqual(
            sorted({"<LINUX_RUN>", "<DISTRO>"} & set(agentic.PLACEHOLDERS)), []
        )
        for rel in bundle_files():
            body = text(rel)
            for word in (
                "<LINUX_RUN>",
                "<DISTRO>",
                "rk service",
                "--services",
                "services.enabled",
                "services.claude_timer",
                "wsl_distro",
                "keep-alive",
                "systemctl --user",
                "wsl.exe",
                "WSL",
                "… -- bash",
            ):
                with self.subTest(rel=rel, word=word):
                    self.assertNotIn(word, body)

    def test_each_token_in_capitals_is_a_placeholder_or_an_example_word(self):
        self.assertEqual(unknown_tokens(BUNDLE), [])
        used = {m.group(0) for rel in bundle_files() for m in TOKEN.finditer(text(rel))}
        # Each example word is one the files use, and none is a placeholder: the install would fill it in.
        self.assertEqual(sorted(EXAMPLE_WORDS - used), [])
        self.assertEqual(sorted(EXAMPLE_WORDS & set(agentic.PLACEHOLDERS)), [])

    def test_the_token_check_finds_a_misspelt_placeholder(self):
        """A misspelt placeholder is no placeholder: `agentic.left` finds nothing left in it, and the check names it,
        in any agentic file, the README with them."""
        misspelt = "See <NO_SUCH_PLACEHOLDER>."
        self.assertEqual(agentic.left(misspelt), [])
        with tempfile.TemporaryDirectory() as d:
            bundle = pathlib.Path(d)
            (bundle / "skills").mkdir()
            (bundle / "skills" / "SKILL.md").write_text(
                f"Run `<RK> status <N>` in `<WORKING_DIR>/x`. {misspelt}\n"
                "Read `<WORKNG_DIR>/tools`.\n",
                encoding="utf-8",
            )
            (bundle / "README.md").write_text(
                "| `<WORKING_DIRS>` | the folder the monitor owns |\n", encoding="utf-8"
            )
            self.assertEqual(
                unknown_tokens(bundle),
                [
                    "README.md:1: <WORKING_DIRS>",
                    "skills/SKILL.md:1: <NO_SUCH_PLACEHOLDER>",
                    "skills/SKILL.md:2: <WORKNG_DIR>",
                ],
            )

    def test_the_skill_starts_with_its_front_matter(self):
        skill = text("skills/rocke-monitor/SKILL.md")
        self.assertTrue(skill.startswith("---\nname: rocke-monitor\n"))
        self.assertLess(
            skill.index("\n---\n"), skill.index("SPDX-License-Identifier: MIT")
        )
        self.assertIn("## The package: features and models", skill)

    def test_paths_after_a_placeholder_use_forward_slashes(self):
        for rel in bundle_files():
            with self.subTest(rel=rel):
                self.assertIsNone(PLACEHOLDER_PATH.search(text(rel)))
        # The check finds a path written with backslashes only where it starts at a placeholder.
        self.assertEqual(
            [
                m.group(0)
                for m in PLACEHOLDER_PATH.finditer(
                    "<WORKING_DIR>\\tmp\\v7fr and %USERPROFILE%\\x\\y"
                )
            ],
            ["<WORKING_DIR>\\tmp\\v7fr"],
        )


class TestRender(unittest.TestCase):
    """The committed files rendered with a configuration, and installed into a Claude workspace."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = pathlib.Path(self.tmp.name)

    def test_the_container_note_and_the_editable_lessons_file(self):
        cfg = a_config(self.base)
        out = self.base / "work" / "agentic"
        for inside in (False, True):
            with mock.patch.object(agentic, "_in_container", return_value=inside):
                agentic.render(BUNDLE, cfg, out)
            skill = (out / "skills" / "rocke-monitor" / "SKILL.md").read_text(
                encoding="utf-8"
            )
            self.assertEqual(
                "**In the container.** `rk orchestrate` keeps this session" in skill,
                inside,
            )
            self.assertEqual(agentic.left(skill), [])
        # How the ticks reach the session there: the orchestrator's own tick, read first by the tick triage (the rule
        # filter by default, a triage model as an opt-in), then only the lines that need Claude; the ticks and the
        # review driver are the orchestrator's (SKILL.md's "who runs the ticks" paragraph says so once).
        for words in (
            "the\n  ticks and the review driver are its own (Tick procedure)",
            "rule filter (`rk tick-filter`, no model)",
            "triage model (`models.tick`) that also does the routine work with read-only `rk` commands",
            "Skip step 1 (the tick ran)",
            "`CI_AFTER_POST #N <head commit>: ...` is the triage model's",
            "the triage did not\n  answer, handle the whole tick",
            "The plain `/rocke-monitor tick` means there is no\n  triage now",
        ):
            self.assertIn(words, skill)
        self.assertNotIn(
            "claude.tick_min", agentic.CONTAINER_NOTE
        )  # SKILL.md's paragraph says it
        vals = agentic.values(cfg)
        self.assertEqual(
            (vals["<AUDIT_MODEL>"], vals["<DEEP_MODEL>"], vals["<CLAUDE_EFFORT>"]),
            ("claude-opus-5-5", "claude-opus-5-5", "high"),
        )  # opus, sent by its id
        self.assertEqual(
            vals["<QUALITY_EFFORT>"], "medium"
        )  # the package default, as the skill dispatches the lane
        self.assertEqual(
            agentic.values(
                dict(cfg, claude={"effort": "xhigh"}, effort={"quality": ""})
            )["<QUALITY_EFFORT>"],
            "xhigh",
        )  # blank: claude.effort
        own = dict(
            cfg,
            claude={"effort": "high"},
            effort={
                "quality": "MAX",
                "verify": "low",
                "audit": "medium",
                "deep": "xhigh",
            },
        )
        self.assertEqual(
            tuple(
                agentic.values(own)[k]
                for k in (
                    "<QUALITY_EFFORT>",
                    "<CLAUDE_EFFORT>",
                    "<VERIFY_EFFORT>",
                    "<AUDIT_EFFORT>",
                    "<DEEP_EFFORT>",
                )
            ),
            ("max", "high", "low", "medium", "xhigh"),
        )  # each role's own over the base; the session (orchestrator) has none here: the base
        lessons = pathlib.Path(cfg["paths"]["lessons"]) / "lessons.md"
        self.assertEqual(agentic.values(cfg)["<LESSONS_FILE>"], str(lessons))
        target = self.base / "target"
        lines = agentic.install(out, target, lessons_file=lessons)
        self.assertIn(
            f"add {lessons} (the editable lessons file; never overwritten)", lines
        )
        self.assertEqual(
            lessons.read_text(encoding="utf-8"),
            (out / "skills" / "rocke-monitor" / "lessons.md").read_text(
                encoding="utf-8"
            ),
        )
        lessons.write_text(
            "# Lessons\n\n- L-1 · a lesson\n- L-2 · learned here\n", encoding="utf-8"
        )
        lines = agentic.install(out, target, lessons_file=lessons)
        self.assertIn(
            f"kept {lessons} (the editable lessons file; the shipped one is in the skill)",
            lines,
        )
        self.assertIn(
            "L-2 · learned here", lessons.read_text(encoding="utf-8")
        )  # never overwritten

    def test_the_site_lessons_are_found_through_the_registry(self):
        """The site lessons are the registry's paths.site_lessons, a file in the monitor's state, which the render
        names as <SITE_LESSONS_FILE>: the rendered lessons_for.py adds their sections to each role's extract, headed
        "(site lessons)", once the file is there, and reads no variable for it: RKMON_SITE_LESSONS, set to another
        file, changes nothing."""
        cfg = a_config(self.base)
        site = pathlib.Path(cfg["paths"]["site_lessons"])
        self.assertEqual(registry.key("paths.site_lessons").owner, registry.DERIVED)
        self.assertEqual(
            site, config.state_dir(cfg["paths"]["toolkit"]) / paths.SITE_LESSONS
        )
        self.assertEqual(agentic.values(cfg)["<SITE_LESSONS_FILE>"], str(site))
        out = self.base / "work" / "agentic"
        agentic.render(BUNDLE, cfg, out)
        script = out / "skills" / "rocke-monitor" / "lessons_for.py"
        self.assertIn(
            f'SITE = pathlib.Path("{site}")', script.read_text(encoding="utf-8")
        )
        self.assertNotIn("RKMON_SITE_LESSONS", script.read_text(encoding="utf-8"))
        elsewhere = self.base / "elsewhere.md"
        elsewhere.write_text(
            "# Site lessons\n\n## Worker\n\n- L-902 · a file the variable names\n",
            encoding="utf-8",
        )

        def worker() -> str:
            return subprocess.run(
                [sys.executable, "-B", str(script), "worker"],
                capture_output=True,
                text=True,
                timeout=60,
                check=True,
                env=dict(os.environ, RKMON_SITE_LESSONS=str(elsewhere)),
            ).stdout

        extract = worker()
        self.assertNotIn("L-902", extract)
        self.assertNotIn("(site lessons)", extract)  # none yet: lessons.md alone
        site.parent.mkdir(parents=True)
        site.write_text(
            "# Site lessons\n\n## Worker\n\n- L-901 · a fact of this site\n",
            encoding="utf-8",
        )
        extract = worker()
        self.assertIn("## Worker (site lessons)\n", extract)
        self.assertIn("- L-901 · a fact of this site\n", extract)
        self.assertNotIn("L-902", extract)

    def test_render_leaves_no_placeholder(self):
        out = self.base / "work" / "agentic"
        cfg = a_config(self.base, jira={"project": "DEMO"})
        files = agentic.render(BUNDLE, cfg, out)
        self.assertEqual(len(files), len(bundle_files()))
        for f in files:
            self.assertEqual(agentic.left(f.read_text(encoding="utf-8")), [], f)
        text = (out / "skills" / "rocke-monitor" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("only the-operator's explicit re-approval lifts it", text)
        self.assertIn("JIRA tickets go to `DEMO`", text)
        self.assertIn(
            f"| CLI | `{cfg['paths']['toolkit']}/rk <command>`. A printed `rkmon rk <command>`",
            text,
        )
        self.assertIn(
            agentic.NOT_SET,
            (out / "skills" / "rocke-monitor" / "rules.md").read_text(encoding="utf-8"),
        )  # the team channel is not set up
        agent = (out / "agents" / "rocke-review-verifier.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "model: claude-opus-5-5\neffort: high\n", agent
        )  # the audit model (models.audit: opus by default), by its id
        self.assertIn(
            "`features.review_automation.parallel_reviews`; `rk status` says how many run and why",
            text,
        )  # no number of parallel reviews is rendered
        sonnet = a_config(
            self.base, orchestrator={"model": "claude-sonnet-5-5"}, review={"pipes": 5}
        )
        agentic.render(BUNDLE, sonnet, out)
        # The verifier agent runs the audit model, Opus whatever the orchestrator runs on (models.audit).
        self.assertIn(
            "model: claude-opus-5-5\neffort: high\n",
            (out / "agents" / "rocke-review-verifier.md").read_text(encoding="utf-8"),
        )
        text = (out / "skills" / "rocke-monitor" / "SKILL.md").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "(`features.review_automation.orchestrator`: `claude-sonnet-5-5`", text
        )
        self.assertNotIn("5 asked", text)  # not even with the setting at 5
        own = a_config(
            self.base,
            models={"audit": "claude-opus-5-5", "deep": "opus"},
            effort={"verify": "max"},
            claude={
                "effort": "max"
            },  # the base of the roles with no effort of their own (the audit)
            offload={"mode": "shadow", "spot_check": 0.5, "spot_check_min": 3},
        )
        agentic.render(BUNDLE, own, out)
        rendered = (out / "agents" / "rocke-review-verifier.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("model: claude-opus-5-5\neffort: max\n", rendered)
        self.assertEqual(
            rendered.count("\neffort:"), 1, rendered
        )  # exactly one effort key in each agent file
        offload_md = (out / "skills" / "rocke-monitor" / "offload.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("`offload.mode` (shadow here)", offload_md)
        self.assertIn(
            "(its file runs `claude-opus-5-5`), effort `max`, in the background",
            " ".join(offload_md.split()),
        )
        self.assertEqual(agentic.left(offload_md), [])

    def test_install_backs_up_keeps_shared_policies_and_merges_rows(self):
        out = self.base / "work" / "agentic"
        agentic.render(BUNDLE, a_config(self.base), out)
        target = self.base / "target"
        (target / ".claude" / "skills" / "shared").mkdir(parents=True)
        (target / ".claude" / "skills" / "shared" / "prerequisites.md").write_text(
            "ours\n", encoding="utf-8"
        )
        lines = agentic.install(out, target)
        self.assertTrue(
            (target / ".claude" / "skills" / "rocke-monitor" / "SKILL.md").exists()
        )
        self.assertEqual(
            (target / ".claude" / "skills" / "shared" / "prerequisites.md").read_text(),
            "ours\n",
        )
        self.assertTrue(any(x.startswith("kept ") for x in lines))
        table = (target / ".claude" / "team_members" / "dispatch_table.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("`rocke_review_verifier.md`", table)
        self.assertLess(
            table.index("BEGIN GENERATED"), table.index("rocke_review_verifier")
        )
        self.assertLess(
            table.index("rocke_review_verifier"), table.index("END GENERATED")
        )
        skill = target / ".claude" / "skills" / "rocke-monitor" / "SKILL.md"
        skill.write_text("changed by hand\n", encoding="utf-8")
        again = agentic.install(out, target)
        self.assertTrue(any("back up" in x for x in again), again)
        self.assertEqual(len(list(skill.parent.glob("SKILL.md.bak-*"))), 1)
        third = agentic.install(out, target)
        self.assertFalse(
            [x for x in third if x.startswith(("add ", "update ", "back up "))], third
        )

    def test_an_install_cut_short_is_installed_again_by_the_next_refresh(self):
        """The render counts as installed only once its files, the lessons file and the registry rows are in place:
        an install cut short before the rows records no render, so the next refresh installs it again.
        """
        target = self.base / "target"
        cfg = a_config(self.base, claude_workspace=str(target))
        out = pathlib.Path(paths.get(cfg, "agentic"))
        agentic.render(BUNDLE, cfg, out)
        with mock.patch.object(agentic, "merge_rows", side_effect=OSError("cut")):
            with self.assertRaises(OSError):
                agentic.install(out, target)
        self.assertTrue((target / ".claude/skills/rocke-monitor/SKILL.md").is_file())
        self.assertEqual(agentic.installed_digest(target), "")
        self.assertTrue(agentic.refresh(cfg, BUNDLE))
        table = target / ".claude" / "team_members" / "dispatch_table.md"
        self.assertIn("`rocke_review_verifier.md`", table.read_text(encoding="utf-8"))
        self.assertEqual(agentic.refresh(cfg, BUNDLE), [])


# Why install's line removes a file, in its words.
GONE = "installed before; not in the package or the team's skills"


class TestTeamSkills(unittest.TestCase):
    """A team's own review skills, delivered beside the deploy settings, laid over the package's agentic
    folder before the render: a file with a package file's path replaces it, one with a new path is added, and the
    render's digest covers them, so a changed file renders the skill again. A file the team drops goes from the
    workspace at the next install, backed up first when edited there since; a shared policy stays.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = pathlib.Path(self.tmp.name)
        self.target = self.base / "target"
        self.cfg = a_config(self.base, claude_workspace=str(self.target))
        self.team = self.base / "skills"
        # The team's verifier is the package's with a rule of its own; the team adds an agent type too.
        self.write(
            "team_members/rocke_review_verifier.md",
            text("team_members/rocke_review_verifier.md")
            + "\n## The team's rule\n\nAsk <user_login> about tile sizes.\n",
        )
        self.write("agents/team-perf-reviewer.md", "# The team's perf reviewer\n")

    def write(self, rel: str, body: str) -> None:
        f = self.team / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(body, encoding="utf-8")

    def installed(self, rel: str) -> str:
        return (self.target / ".claude" / rel).read_text(encoding="utf-8")

    def manifest(self) -> dict:
        return json.loads(
            (
                self.target / ".claude/skills/rocke-monitor" / agentic.INSTALLED
            ).read_text(encoding="utf-8")
        )

    def test_the_team_s_files_replace_the_package_s_and_add_their_own(self):
        """The session renders the team's verifier, filled in, where rk review-split reads it, and adds the team's
        agent type; the package's other files stay."""
        out = pathlib.Path(paths.get(self.cfg, "agentic"))
        files = agentic.render(BUNDLE, self.cfg, out, self.team)
        self.assertEqual(len(files), len(bundle_files()) + 1)  # one replaced, one added
        agentic.install(out, self.target)
        verifier = pathlib.Path(paths.get(self.cfg, "verifier_team_file")).read_text(
            encoding="utf-8"
        )
        self.assertTrue(
            verifier.endswith(
                "\n## The team's rule\n\nAsk the-operator about tile sizes.\n"
            ),
            verifier[-200:],
        )
        self.assertEqual(agentic.left(verifier), [])
        self.assertEqual(
            self.installed("agents/team-perf-reviewer.md"),
            "# The team's perf reviewer\n",
        )
        self.assertIn(
            "only the-operator's explicit re-approval lifts it",
            self.installed("skills/rocke-monitor/SKILL.md"),
        )

    def test_check_18_passes_the_folder_and_the_session_renders_its_verifier(self):
        """Check 18: the team's verifier keeps each checklist lead, so the check passes its folder; the
        session renders that verifier, in which a review in parts finds each lead's bullet.
        """
        doc = settings_file.Document(
            file_types.DEPLOY,
            file_types.DEPLOY.version,
            {"features": {"review_automation": {"skills_folder": str(self.team)}}},
        )
        rows = coherence.skills(doc, ROOT)
        self.assertEqual(
            [(r.name, str(r.status)) for r in rows], [(coherence.SKILLS, "PASS")], rows
        )
        out = pathlib.Path(paths.get(self.cfg, "agentic"))
        agentic.render(BUNDLE, self.cfg, out, self.team)
        verifier = (out / verifier_file.FILE).read_text(encoding="utf-8")
        self.assertIn("\n## The team's rule\n", verifier)
        quoted = split.checklist(verifier, list(verifier_file.CHECKS))
        self.assertEqual([line for line in quoted if line.startswith("- !! ")], [])

    def test_a_changed_team_file_renders_the_skill_again(self):
        """A changed team file renders and installs the skill again; with no team's skills, the package's own come
        back."""
        self.assertTrue(agentic.refresh(self.cfg, BUNDLE, first=True, team=self.team))
        self.assertEqual(agentic.refresh(self.cfg, BUNDLE, team=self.team), [])
        self.write("agents/team-perf-reviewer.md", "# The team's perf reviewer, v2\n")
        lines = agentic.refresh(self.cfg, BUNDLE, team=self.team)
        self.assertTrue(lines and "rendered again" in lines[0], lines)
        self.assertIn(
            f"with the team's skills in {self.team} laid over the package's", lines[0]
        )
        self.assertEqual(
            self.installed("agents/team-perf-reviewer.md"),
            "# The team's perf reviewer, v2\n",
        )
        self.assertEqual(agentic.refresh(self.cfg, BUNDLE, team=self.team), [])
        # No team's skills any more (skills_folder blank): the package's verifier again.
        lines = agentic.refresh(self.cfg, BUNDLE)
        self.assertTrue(lines and "rendered again" in lines[0], lines)
        self.assertNotIn("team's skills", lines[0])
        self.assertNotIn(
            "The team's rule", self.installed("team_members/rocke_review_verifier.md")
        )

    def test_refresh_takes_the_team_s_skills_delivered_here(self):
        """With no folder given, refresh renders the package's agentic folder with the team's skills delivered here
        (team_skills: TEAM_SKILLS, where the settings are delivered and it is there), and the package's alone
        elsewhere."""
        delivered = "rkmon_lib.runtime_config.delivered"
        with mock.patch.object(agentic, "TEAM_SKILLS", self.team):
            with mock.patch(delivered, return_value=False):
                self.assertIsNone(agentic.team_skills())
                self.assertTrue(agentic.refresh(self.cfg, first=True))
            self.assertNotIn(
                "The team's rule",
                self.installed("team_members/rocke_review_verifier.md"),
            )
            with mock.patch(delivered, return_value=True):
                self.assertEqual(agentic.team_skills(), self.team)
                lines = agentic.refresh(self.cfg)
            self.assertIn(f"with the team's skills in {self.team} laid over", lines[0])
            self.assertIn(
                "The team's rule",
                self.installed("team_members/rocke_review_verifier.md"),
            )
        with mock.patch.object(agentic, "TEAM_SKILLS", self.base / "none"):
            with mock.patch(delivered, return_value=True):
                self.assertIsNone(agentic.team_skills())

    def test_a_delivery_that_stopped_after_its_commit_gives_the_skills_it_staged(self):
        """Between a delivery's commit and its finish, the team's skills that go with the layer in place are those it
        staged, or none when it marked them as none; before its commit, those in place (deliver.skills_with_layer,
        whose journal tests/test_deliver.py stops at each step)."""
        staged = self.base / (self.team.name + deliver.JOURNAL_NEW)
        staged.mkdir()
        layer_staged = self.base / (container_layout.LAYER_FILE + deliver.JOURNAL_NEW)
        delivered = "rkmon_lib.runtime_config.delivered"
        with mock.patch.object(agentic, "TEAM_SKILLS", self.team), mock.patch(
            delivered, return_value=True
        ):
            self.assertEqual(agentic.team_skills(), staged)
            layer_staged.write_text("{}", encoding="utf-8")
            self.assertEqual(agentic.team_skills(), self.team, "before the commit")
            layer_staged.unlink()
            staged.rmdir()
            (self.base / (staged.name + deliver.JOURNAL_NONE)).touch()
            self.assertIsNone(agentic.team_skills())

    def test_a_file_the_team_dropped_goes(self):
        """A skill the team dropped goes from the workspace at the next install, and its folder with it; as installed,
        it needs no backup. The team's other files and the package's stay."""
        self.write("skills/team-perf/SKILL.md", "# The team's perf skill\n")
        self.assertTrue(agentic.refresh(self.cfg, BUNDLE, first=True, team=self.team))
        skill = self.target / ".claude/skills/team-perf/SKILL.md"
        self.assertTrue(skill.is_file())
        (self.team / "skills/team-perf/SKILL.md").unlink()
        lines = agentic.refresh(self.cfg, BUNDLE, team=self.team)
        self.assertIn(f"remove {skill} ({GONE})", lines)
        self.assertFalse(skill.parent.exists())
        self.assertFalse([x for x in lines if x.startswith("back up ")], lines)
        self.assertNotIn("skills/team-perf/SKILL.md", self.manifest())
        self.assertEqual(
            self.installed("agents/team-perf-reviewer.md"),
            "# The team's perf reviewer\n",
        )
        self.assertTrue(
            (self.target / ".claude/skills/rocke-monitor/SKILL.md").is_file()
        )

    def test_a_dropped_file_edited_since_is_backed_up_then_goes(self):
        """A dropped file edited in the workspace since it was installed is backed up first, as an overwrite backs it
        up, and then goes."""
        self.assertTrue(agentic.refresh(self.cfg, BUNDLE, first=True, team=self.team))
        agent = self.target / ".claude/agents/team-perf-reviewer.md"
        agent.write_text("# The team's perf reviewer, edited here\n", encoding="utf-8")
        (self.team / "agents/team-perf-reviewer.md").unlink()
        lines = agentic.refresh(self.cfg, BUNDLE, team=self.team)
        backups = list(agent.parent.glob("team-perf-reviewer.md.bak-*"))
        self.assertEqual(len(backups), 1, backups)
        self.assertEqual(
            backups[0].read_text(encoding="utf-8"),
            "# The team's perf reviewer, edited here\n",
        )
        self.assertFalse(agent.exists())
        self.assertEqual(
            [x for x in lines if str(agent) in x],
            [f"back up {agent} -> {backups[0].name}", f"remove {agent} ({GONE})"],
        )

    def test_a_dropped_team_member_s_registry_row_comes_out_with_it(self):
        """A team member the render no longer has goes with its row of the workspace's dispatch table: every rocKE row
        there is the render's. The line says which rows came out."""
        row = "| Perf Reviewer | rocKE Perf Reviewer | `rocke_perf_reviewer.md` | Any | tile sizes |\n"
        self.write(agentic.REGISTRY_ROWS, text(agentic.REGISTRY_ROWS) + row)
        self.write("team_members/rocke_perf_reviewer.md", "# rocKE Perf Reviewer\n")
        self.assertTrue(agentic.refresh(self.cfg, BUNDLE, first=True, team=self.team))
        table = self.target / ".claude/team_members/dispatch_table.md"
        self.assertIn("`rocke_perf_reviewer.md`", table.read_text(encoding="utf-8"))
        (self.team / agentic.REGISTRY_ROWS).unlink()
        (self.team / "team_members/rocke_perf_reviewer.md").unlink()
        lines = agentic.refresh(self.cfg, BUNDLE, team=self.team)
        member = self.target / ".claude/team_members/rocke_perf_reviewer.md"
        self.assertIn(f"remove {member} ({GONE})", lines)
        self.assertNotIn("`rocke_perf_reviewer.md`", table.read_text(encoding="utf-8"))
        self.assertIn("`rocke_review_verifier.md`", table.read_text(encoding="utf-8"))
        self.assertIn(
            "registry rows for rocke_code_quality_reviewer.md, rocke_review_verifier.md in "
            f"{table}; the rows for rocke_perf_reviewer.md taken out",
            lines,
        )

    def test_a_shared_policy_the_render_no_longer_has_stays(self):
        """skills/shared/ is the workspace's: a shared policy an install added stays once the render no longer has
        it, and the install stops recording it."""
        self.write("skills/shared/team_policy.md", "# The team's policy\n")
        self.assertTrue(agentic.refresh(self.cfg, BUNDLE, first=True, team=self.team))
        self.assertIn("skills/shared/team_policy.md", self.manifest())
        (self.team / "skills/shared/team_policy.md").unlink()
        lines = agentic.refresh(self.cfg, BUNDLE, team=self.team)
        self.assertTrue(lines and "rendered again" in lines[0], lines)
        self.assertFalse([x for x in lines if x.startswith("remove ")], lines)
        self.assertEqual(
            self.installed("skills/shared/team_policy.md"), "# The team's policy\n"
        )
        self.assertNotIn("skills/shared/team_policy.md", self.manifest())

    def test_a_render_with_nothing_dropped_removes_nothing(self):
        """A render again that drops no file removes nothing: the changed file is updated and every other stays."""
        self.assertTrue(agentic.refresh(self.cfg, BUNDLE, first=True, team=self.team))
        claude = self.target / ".claude"
        before = sorted(p.relative_to(claude) for p in claude.rglob("*") if p.is_file())
        self.write("agents/team-perf-reviewer.md", "# The team's perf reviewer, v2\n")
        lines = agentic.refresh(self.cfg, BUNDLE, team=self.team)
        self.assertIn(f"update {claude / 'agents/team-perf-reviewer.md'}", lines)
        self.assertFalse(
            [x for x in lines if x.startswith(("remove ", "back up "))], lines
        )
        self.assertEqual(
            sorted(p.relative_to(claude) for p in claude.rglob("*") if p.is_file()),
            before,
        )

    def test_a_manifest_entry_for_a_file_install_never_writes_removes_nothing(self):
        """Only a file install writes can go: a manifest entry that names any other path (edited in by hand: outside
        the folders install fills, climbing out of one, the registry rows' file, an absolute path) removes nothing.
        """
        out = pathlib.Path(paths.get(self.cfg, "agentic"))
        agentic.render(BUNDLE, self.cfg, out)
        agentic.install(out, self.target)
        claude = self.target / ".claude"
        strays = {
            "stray.md": claude / "stray.md",
            "notes/stray.md": claude / "notes/stray.md",
            "skills/../stray.md": claude / "stray.md",
            agentic.REGISTRY_ROWS: claude / agentic.REGISTRY_ROWS,
            str(self.base / "outside.md"): self.base / "outside.md",
        }
        for f in strays.values():
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text("not the install's\n", encoding="utf-8")
        mpath = claude / "skills/rocke-monitor" / agentic.INSTALLED
        mpath.write_text(
            json.dumps({**self.manifest(), **{rel: "0" * 64 for rel in strays}}),
            encoding="utf-8",
        )
        lines = agentic.install(out, self.target)
        self.assertFalse(
            [x for x in lines if x.startswith(("remove ", "back up "))], lines
        )
        for f in strays.values():
            self.assertEqual(f.read_text(encoding="utf-8"), "not the install's\n", f)


class TestOffloadWording(unittest.TestCase):
    """The review offload in the committed files: the verifier's audit section and footer, the code quality pass's note,
    the procedure's pointer to offload.md and its two dispatches, and the agent types' model and effort; rendered.
    """

    def test_the_offload_wording_and_its_rendering(self):
        verifier = text("team_members/rocke_review_verifier.md")
        self.assertLess(
            verifier.index("### Offload audit (offload.mode on)"),
            verifier.index("### Output Format"),
        )
        self.assertIn("verified by <AUDIT_NAME>. Checks:", verifier)
        self.assertIn("(<OFFLOAD_SPOT_CHECK>)", verifier)
        quality = text("team_members/rocke_code_quality_reviewer.md")
        self.assertLess(
            quality.index("### With the review offload on"),
            quality.index("### Output Format"),
        )
        proc = text("skills/rocke-monitor/review_procedure.md")
        self.assertIn(
            "## On REVIEW_READY: verify, check, post\n\n**Review offload** (`offload.mode`",
            proc,
        )
        self.assertIn("as a subagent, in the background", proc)
        self.assertIn(
            "with no `model` option (its file in `.claude/agents/` runs `<AUDIT_MODEL>`",
            proc,
        )
        self.assertIn(
            "as a second background subagent (agent type `rocke-code-quality-reviewer` with no `model` option: its "
            "file runs `<DEEP_MODEL>`",
            proc,
        )
        with tempfile.TemporaryDirectory() as d:
            base = pathlib.Path(d)
            out = base / "work" / "agentic"
            agentic.render(BUNDLE, a_config(base, models={"deep": "sonnet"}), out)
            rendered = (out / "team_members" / "rocke_review_verifier.md").read_text(
                encoding="utf-8"
            )
            self.assertIn("verified by Claude Opus. Checks:", rendered)
            self.assertIn("(25% of them, at least 2)", rendered)
            lane = (out / "agents" / "rocke-code-quality-reviewer.md").read_text(
                encoding="utf-8"
            )
            self.assertIn(
                "model: claude-sonnet-5-5\neffort: medium\n", lane
            )  # sonnet, by its id; effort.quality: medium by default
            self.assertEqual(lane.count("\neffort:"), 1, lane)
            for f in out.rglob("*.md"):
                self.assertEqual(agentic.left(f.read_text(encoding="utf-8")), [], f)


# The names the Agent tool's `model` option takes, and the form in which the agentic files give it one.
AGENT_TOOL_NAMES = {"opus", "sonnet", "haiku", "fable"}
AGENT_TOOL_MODEL = re.compile(r"\bmodel `([^`]*)`")


class TestAgentToolModels(unittest.TestCase):
    """The models the rendered files give the Agent tool, whose `model` option takes a family's name only, and the
    agent files' front matter, which takes an id. Rendered with the orchestrator on opus[1m], the audit model on opus
    and the deep model on sonnet: the ids claude-opus-5-5[1m], claude-opus-5-5 and claude-sonnet-5-5.
    """

    IDS = ("claude-opus-5-5[1m]", "claude-opus-5-5", "claude-sonnet-5-5")

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        base = pathlib.Path(cls.tmp.name)
        cls.out = base / "work" / "agentic"
        cfg = a_config(
            base,
            orchestrator={"model": "opus[1m]"},
            models={"audit": "opus", "deep": "sonnet"},
        )
        agentic.render(BUNDLE, cfg, cls.out)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def flat(self, rel: str) -> str:
        """A rendered file's text, its lines joined by single spaces."""
        return " ".join((self.out / rel).read_text(encoding="utf-8").split())

    def instructions(self) -> list[str]:
        """The rendered files an agent reads (all but the folder's README), by their path under the folder."""
        return sorted(
            f.relative_to(self.out).as_posix()
            for f in self.out.rglob("*.md")
            if f.name != "README.md"
        )

    def test_the_agent_tool_is_given_a_family_s_name(self):
        given = {
            m.group(1)
            for rel in self.instructions()
            for m in AGENT_TOOL_MODEL.finditer(self.flat(rel))
        }
        self.assertLessEqual(given, AGENT_TOOL_NAMES)
        proc = self.flat("skills/rocke-monitor/review_procedure.md")
        self.assertIn(
            "fall back to `general-purpose` with model `opus` only when the type is not listed",
            proc,
        )  # the verifier's fallback: the audit model's family
        self.assertIn(
            "the same fallback, with model `sonnet`", proc
        )  # the code quality pass's: the deep model's
        self.assertIn(
            "The lane's agent file runs `claude-sonnet-5-5`, so the lane gets no `model` option; the verifier's runs "
            "`claude-opus-5-5`: when that is not `claude-sonnet-5-5`, dispatch the verifier with model `sonnet`.",
            self.flat("skills/rocke-monitor/offload.md"),
        )  # a deep review's verifier
        weekly = self.flat("skills/rocke-monitor/weekly.md")
        self.assertIn(
            "Dispatch one process analyst (no `model` option: it runs on the session's model) with:",
            weekly,
        )
        for model in self.IDS:
            self.assertNotIn(model, weekly)

    def test_a_pass_dispatched_by_its_agent_type_gets_no_model(self):
        """A pass dispatched by its agent type gets no `model` option, so its agent file's id holds: a family's name
        there would override it. Each place that dispatches one says so where it names the type, and no id is a
        subagent's model. (A deep review's verifier, whose file runs the audit model, is dispatched with the deep
        model's family when the two differ: test_the_agent_tool_is_given_a_family_s_name.)
        """
        found = []
        for rel in self.instructions():
            text = self.flat(rel)
            for m in re.finditer(r"agent type `rocke-[a-z-]+`|by its agent type", text):
                found.append(rel)
                self.assertTrue(
                    text[m.end() :].startswith(" with no `model` option"),
                    (rel, text[m.start() : m.end() + 60]),
                )
            for model in self.IDS:
                self.assertNotRegex(
                    text, re.escape(model) + "`? (?:background )?subagent", rel
                )
        self.assertEqual(
            found,
            [
                "skills/rocke-monitor/SKILL.md",
                "skills/rocke-monitor/offload.md",
                "skills/rocke-monitor/review_procedure.md",
                "skills/rocke-monitor/review_procedure.md",
            ],
        )  # the skill's word on the passes, the verifier in an audit, and the verifier and the lane in a review

    def test_the_agent_files_hold_the_ids(self):
        for rel, want in (
            ("agents/rocke-review-verifier.md", "claude-opus-5-5"),
            ("agents/rocke-code-quality-reviewer.md", "claude-sonnet-5-5"),
        ):
            front = (self.out / rel).read_text(encoding="utf-8").split("\n---\n")[0]
            self.assertIn(f"\nmodel: {want}\n", front + "\n", rel)


class TestBundleNames(unittest.TestCase):
    """The committed files name only what the package has, and no live fill-in token or settings file."""

    def test_the_committed_files_name_only_what_the_package_has(self):
        self.assertEqual(missing_names(BUNDLE, package_names(ROOT)), [])

    def test_the_committed_files_hold_no_live_fill_in_token_or_settings_file(self):
        self.assertEqual(dead_pointers(BUNDLE), [])

    def test_the_check_names_what_the_package_lacks(self):
        have = {
            "commands": {"status", "loop-status"},
            "code": 'print("REVIEW_READY #7")\nprint("!! DISARM_NEEDED #7")\n',
            "scripts": {"tick_loop.py", "mkwt.py"},
        }
        with tempfile.TemporaryDirectory() as d:
            bundle = pathlib.Path(d)
            (bundle / "skills").mkdir()
            (bundle / "skills" / "SKILL.md").write_text(
                "Run `rk status`, `<RK> loop-status` and `rk gh-logins`; never `rk <cmd>` as typed.\n"
                "On `REVIEW_READY #N` verify; `!! DISARM_NEEDED #N` disarms; on `!! GH_LOGIN ...` re-login; "
                "`GH_LOGINS` is routine; a PR labelled `ON_HOLD` waits.\n"
                "```\nrk merge-now 7\n```\n"
                "Keep `python3 -B <PACKAGE_DIR>/tools/tick_loop.py` and `<WORKING_DIR>/tools/mkwt.py`; read "
                "`<PACKAGE_DIR>/tools/ghr.py`; rocKE's own `platform/tests/tools/test_x.py` is no tool of ours.\n"
                "`bash <PACKAGE_DIR>/tools/tick_loop.sh` names a script the package lacks: its tick loop is a .py.\n",
                encoding="utf-8",
            )
            self.assertEqual(
                missing_names(bundle, have),
                [
                    "skills/SKILL.md: rk command gh-logins",
                    "skills/SKILL.md: rk command merge-now",
                    "skills/SKILL.md: tick line GH_LOGIN",
                    "skills/SKILL.md: tick line GH_LOGINS",
                    "skills/SKILL.md: tools/ script ghr.py",
                    "skills/SKILL.md: tools/ script tick_loop.sh",
                ],
            )

    def test_the_check_finds_a_live_fill_in_token_and_the_settings_file(self):
        """A live skill's own fill-in tokens and its settings file, which a deployment does not have, fail the check;
        the placeholders the install fills do not, nor does the README, which describes the files.
        """
        with tempfile.TemporaryDirectory() as d:
            bundle = pathlib.Path(d)
            (bundle / "skills").mkdir()
            (bundle / "skills" / "SKILL.md").write_text(
                "Open `<paths.root>/monitor` in `<wsl_distro>` as `<wsl_user>`; `<WORKING_DIR>` and `<RK>` are ours.\n"
                "The settings: `.claude/project_config.md` (project_config.md).\n",
                encoding="utf-8",
            )
            (bundle / "README.md").write_text(
                "A live skill's `<paths.root>` is no token of ours.\n", encoding="utf-8"
            )
            self.assertEqual(
                dead_pointers(bundle),
                [
                    "skills/SKILL.md:1: <paths.root>",
                    "skills/SKILL.md:1: <wsl_distro>",
                    "skills/SKILL.md:1: <wsl_user>",
                    "skills/SKILL.md:2: .claude/project_config.md",
                    "skills/SKILL.md:2: project_config.md",
                ],
            )

    def test_an_exempt_name_passes_only_while_it_is_exempt(self):
        have = {"commands": set(), "code": "", "scripts": set()}
        with tempfile.TemporaryDirectory() as d:
            bundle = pathlib.Path(d)
            (bundle / "merge.md").write_text(
                "A PR labelled `ON_HOLD` waits.\n", encoding="utf-8"
            )
            self.assertIn("ON_HOLD", EXEMPT_NAMES)
            self.assertEqual(missing_names(bundle, have), [])
            with mock.patch.dict(EXEMPT_NAMES, clear=True):
                self.assertEqual(
                    missing_names(bundle, have), ["merge.md: tick line ON_HOLD"]
                )

    def test_each_exemption_is_one_the_committed_files_need(self):
        """Each exempt name is one the committed files give: an exemption for a name no file gives goes."""
        self.assertEqual(stale_exemptions(BUNDLE), [])
        with tempfile.TemporaryDirectory() as d:
            bundle = pathlib.Path(d)
            (bundle / "merge.md").write_text(
                "A PR labelled `ON_HOLD` waits.\n", encoding="utf-8"
            )
            self.assertEqual(
                stale_exemptions(bundle), sorted(set(EXEMPT_NAMES) - {"ON_HOLD"})
            )

    def test_the_package_lists_come_from_the_package(self):
        have = package_names(ROOT)
        self.assertIn("scoreboard", have["commands"])
        self.assertIn("publish_package.py", have["scripts"])
        self.assertIn("mkwt.py", have["scripts"])  # a worker tool
        self.assertIn("REVIEW_READY", have["code"])
        self.assertTrue(set(EXEMPT_NAMES) and all(EXEMPT_NAMES.values()))


class TestGuardLines(unittest.TestCase):
    """The guards the package builds are named in the lanes' card and the subcommands, and the package has each one:
    none is exempt from the names check."""

    def test_the_card_and_the_subcommands_name_the_guards(self):
        card = text("skills/rocke-monitor/lane_commands.md")
        table = card.split("| Command | Prints |", 1)[1].split("\n\n", 1)[0]
        for row in (
            "rk lane-of",
            "rk probe-check",
            "rk probe-port",
            "rk shard-times",
            "rk werror",
            "rk job-log ID --errors",
        ):
            self.assertIn(
                f"| `{row}", table
            )  # rows of the table itself, before its blank line
        self.assertIn(
            "- `fixcheck: N<k> ran|read <how>`", card.split("It also re-checks")[0]
        )
        guards = card.split("**The checks' guard lines**")[1].split("**Recipes**")[0]
        for word in (
            "DEVTESTS_FAIL",
            "IMPORTERS_FAIL",
            "HALF RENUMBER",
            "ABI NUMBER CLASH",
            "FLAKE",
            "SURVIVED",
        ):
            self.assertIn(word, guards)
        cmds = text("skills/rocke-monitor/commands.md")
        rows = cmds.split("| `board`")[0]
        self.assertIn("| `pr-slot status` |", rows)
        self.assertIn(
            "| `pr-run [--slot NAME [--end]] [--env FILE] [-C DIR] [--out DIR] -- CMD` |",
            rows,
        )

    def test_the_guards_are_the_packages_and_none_is_exempt(self):
        have = package_names(ROOT)
        for cmd in (
            "lane-of",
            "probe-check",
            "probe-port",
            "shard-times",
            "werror",
            "job-log",
            "pr-slot",
            "pr-run",
        ):
            self.assertIn(cmd, have["commands"])
        for line in ("DEVTESTS_FAIL", "IMPORTERS_FAIL"):
            self.assertRegex(have["code"], rf"\b{line}\b")
        for name in ("probe-check", "lane-of", "shard-times", "DEVTESTS_FAIL"):
            self.assertNotIn(name, EXEMPT_NAMES)


class TestTeamChat(unittest.TestCase):
    def test_the_skill_names_the_team_chat_never_the_message_label(self):
        """<TEAM_CHANNEL> reads "team chat" while the alerts are on, whatever the message label, a blank one
        included (the webhook decides the channel), and "(not set up)" while they are off.
        """
        base = pathlib.Path("/w")
        for label in ("", "Chan"):
            on = a_config(
                base,
                features={
                    "alerts": {
                        "enabled": True,
                        "develop_breaks": True,
                        "needs_you": True,
                    }
                },
                teams={"channel": label},
            )
            self.assertEqual(agentic.values(on)["<TEAM_CHANNEL>"], "team chat")
        off = a_config(base, teams={"channel": "Chan"})
        self.assertEqual(agentic.values(off)["<TEAM_CHANNEL>"], agentic.NOT_SET)


class TestJiraPlaceholders(unittest.TestCase):
    def test_the_site_the_issue_type_and_the_required_fields_come_from_the_settings(
        self,
    ):
        """<ATLASSIAN_SITE> is the name the CLI's --site takes, from the JIRA address the settings give (a name is kept
        as it is); <JIRA_ISSUE_TYPE> is the issue type a ticket is filed with, Task while the setting is blank; and
        <JIRA_REQUIRED_FIELDS> gives the fields every ticket must carry as JSON, {} while there are none.
        """
        base = pathlib.Path("/w")
        for site in ("https://example-site.example.com", "example-site"):
            with self.subTest(site=site):
                cfg = a_config(base, atlassian={"site": site})
                self.assertEqual(
                    agentic.values(cfg)["<ATLASSIAN_SITE>"], "example-site"
                )
        blank = agentic.values(a_config(base))
        self.assertEqual(
            (blank["<JIRA_ISSUE_TYPE>"], blank["<JIRA_REQUIRED_FIELDS>"]),
            ("Task", "{}"),
        )
        fields = {"customfield_1": {"value": "Yes"}, "customfield_2": "x"}
        jira = dict(
            config.load_defaults()["jira"], issue_type="Bug", extra_fields=fields
        )
        mine = agentic.values(a_config(base, jira=jira))
        self.assertEqual(
            (mine["<JIRA_ISSUE_TYPE>"], json.loads(mine["<JIRA_REQUIRED_FIELDS>"])),
            ("Bug", fields),
        )

    def test_the_ticket_row_files_with_the_issue_type_and_the_required_fields(self):
        """The skill rendered with an issue type and a field every ticket must carry: its File a ticket row files the
        ticket with that type (--type "Bug") and gives the field as JSON."""
        fields = {"customfield_1": {"value": "Yes"}}
        jira = dict(
            config.load_defaults()["jira"], issue_type="Bug", extra_fields=fields
        )
        with tempfile.TemporaryDirectory() as tmp:
            base = pathlib.Path(tmp)
            out = base / "work" / "agentic"
            agentic.render(BUNDLE, a_config(base, jira=jira), out)
            text = (out / "skills" / "rocke-monitor" / "commands.md").read_text(
                encoding="utf-8"
            )
        rows = [
            line for line in text.splitlines() if line.startswith("| File a ticket |")
        ]
        self.assertEqual(len(rows), 1, rows)
        self.assertIn('--type "Bug"', rows[0])
        self.assertIn(json.dumps(fields, ensure_ascii=False), rows[0])


class TestServeRow(unittest.TestCase):
    """The subcommands' serve row in the committed commands.md: the access token is on every request, and where the
    page listens and who can reach it is the README's "The admin dashboard", the one place that says it; the row names
    no address, origin or token file of its own."""

    def test_the_row_points_to_the_readme(self):
        text = (
            ROOT / "agentic" / "skills" / "rocke-monitor" / "commands.md"
        ).read_text(encoding="utf-8")
        rows = [line for line in text.splitlines() if line.startswith("| `serve ")]
        self.assertEqual(len(rows), 1, rows)
        row = rows[0]
        self.assertIn(
            '"Request PR priority" box: it serves the admin dashboard, and every request carries the access token '
            '(where it listens and who can reach it: `<PACKAGE_DIR>/README.md`, "The admin dashboard"). '
            "On <user_login>'s machine",
            row,
        )
        for gone in ("127.0.0.1", "origin null", "per-machine", "token_file"):
            self.assertNotIn(gone, row)
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertRegex(
            readme, r"(?m)^#+ The admin dashboard$"
        )  # the section the row points to


class TestWhatPrototypeSends(unittest.TestCase):
    """What the skill says prototype and standby send, against the code that decides it: every feature
    that sends (features.SENDS) runs in a mode, in prototype what it would send is kept as an unsent item, and the lines
    a held write prints are the gate's own (post.outward_gate, the writer's for `rk atlassian`).
    """

    @staticmethod
    def words(rel: str) -> str:
        """A committed agentic file's text, its lines joined as they read."""
        return " ".join(text(rel).split())

    def test_each_feature_that_sends_runs_in_a_mode_and_prototype_keeps_what_it_would_send(
        self,
    ):
        skill = self.words("skills/rocke-monitor/SKILL.md")
        modes = skill.split("**Features and modes.**", 1)[1].split("**Off means", 1)[0]
        for feature in features.SENDS:
            self.assertIn(f"`{features.PREFIX}{feature}`", modes)
        for words in (
            "is kept as an unsent item instead",
            "In standby, which is the whole monitor's, no feature sends, in prototype or live",
        ):
            self.assertIn(words, modes)
        held = skill.split("**Prototype and standby keep what would go out.**", 1)[1]
        held = held.split(" - **", 1)[0]
        for line in (
            f"`{post.UNSENT}: <what> for <whom>, kept as an unsent item: <why>`",
            f"`{post.WAITING}: ...`",
            f"`{post.KEPT_KIND}: ...`",
        ):
            self.assertIn(line, held)

    def test_a_jira_write_kept_in_prototype_reads_as_the_writer_prints_it(self):
        kept = (
            f"{post.UNSENT}: {outward.row(outward.ATLASSIAN).what} for "
            f"{tickets.unsent_for({'jira': {'project': '<JIRA_PROJECT>'}})}, kept as an unsent item: ..."
        )
        self.assertIn(
            f"`{kept}`, exit 0", self.words("skills/rocke-monitor/commands.md")
        )


def a_clone(base: pathlib.Path) -> pathlib.Path:
    """A repository in BASE that tracks the tests' copy of the package (sandbox.copy_package: what the package holds,
    in this tree or in a copy that has no .git, as the image's has none), as a clone of the package's repository does:
    a build run by its tools/make_package.py takes the files git tracks there, wherever the suite runs.
    """
    clone = sandbox.copy_package(base / "clone")
    subprocess.run(["git", "init", "-q", str(clone)], check=True)
    subprocess.run(["git", "-C", str(clone), "add", "-A"], check=True)
    return clone


class TestMakePackage(unittest.TestCase):
    """Each build runs from a repository that holds the tests' copy of the package (`a_clone`), whose own make_package
    it runs."""

    @classmethod
    def setUpClass(cls):
        d = tempfile.TemporaryDirectory()
        cls.addClassCleanup(d.cleanup)
        cls.build = [
            sys.executable,
            "-B",
            str(a_clone(pathlib.Path(d.name)) / "tools" / "make_package.py"),
        ]

    def test_the_folder_as_in_the_repository_with_a_passing_privacy_check(self):
        with tempfile.TemporaryDirectory() as d:
            base = pathlib.Path(d)
            out = base / "dist"
            env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
            # The deny list, kept outside the package: here an invented one.
            deny = base / "private" / "privacy_deny.txt"
            deny.parent.mkdir()
            deny.write_text("invented" + "-deny-value-z\n", encoding="utf-8")
            # A monitor's configuration or state is no input of the build, nor is a Claude workspace, a review book or
            # a document from outside the package (its docs/ and agentic/ are its own); the deny list is a required
            # one.
            build = self.build
            for extra, why in (
                (
                    ["--deny", str(deny), "--config", str(base / "config.json")],
                    "unrecognized arguments: --config",
                ),
                (
                    ["--deny", str(deny), "--state", str(base / "state.json")],
                    "unrecognized arguments: --state",
                ),
                (
                    ["--deny", str(deny), "--doc", "design=" + str(base / "design.md")],
                    "unrecognized arguments: --doc",
                ),
                (
                    ["--deny", str(deny), "--workspace", str(base / "ws")],
                    "unrecognized arguments: --workspace",
                ),
                (
                    ["--deny", str(deny), "--review-book", str(base / "book.md")],
                    "unrecognized arguments: --review-book",
                ),
                (
                    ["--deny", str(deny), "--lenient-agentic"],
                    "unrecognized arguments: --lenient-agentic",
                ),
                ([], "the following arguments are required: --deny"),
            ):
                r = subprocess.run(
                    build + ["--out", str(out), *extra],
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=120,
                )
                self.assertEqual((r.returncode, why in r.stderr), (2, True), r.stderr)
                self.assertFalse(out.exists(), why)
            # A clone builds with --out and --deny alone.
            p = subprocess.run(
                build + ["--out", str(out), "--deny", str(deny), "--no-tests"],
                capture_output=True,
                text=True,
                env=env,
                timeout=600,
            )
            self.assertEqual(p.returncode, 0, p.stdout[-3000:] + p.stderr[-3000:])
            pkg = out / "rocke_monitor"
            for name in (
                "rk",
                "rkmon.py",
                "rkmon_lib",
                "tests",
                "tools",
                "worker",
                "requirements",
                "agentic",
                "docs",
                "README.md",
                "QUICK_USER_GUIDE.md",
                "QUICK_BUILD_GUIDE.md",
                "docs/internal_settings.md",
                "VERSION",
                ".gitignore",
                "config.json",
                "config.defaults.json",
                "docs/restart.md",
                "docs/review_pipeline.md",
                "docs/design.md",
                "docs/operations_guide.md",
            ):
                self.assertTrue((pkg / name).exists(), name)
            self.assertEqual(
                json.loads((pkg / "config.json").read_text()).keys(), {"_comment"}
            )
            report = (out / "privacy_report.txt").read_text(encoding="utf-8")
            self.assertIn("Result: PASS", report)
            # The check ran as the built folder's own code: each part read, each settings template rendered.
            for name, _inside in privacy.PARTS:
                self.assertRegex(
                    report, re.escape(f"  PASS  {name}: ") + r"\d+ file\(s\) read"
                )
            self.assertIn(f"  PASS  {privacy.S_TEMPLATES}: 5 rendered", report)
            # The documents and the agentic files are the package's own: the folder holds them as the tree does.
            for doc in ("design.md", "operations_guide.md"):
                self.assertEqual(
                    (pkg / "docs" / doc).read_bytes(),
                    (ROOT / "docs" / doc).read_bytes(),
                    doc,
                )
            built = sorted(
                f.relative_to(pkg / "agentic").as_posix()
                for f in (pkg / "agentic").rglob("*")
                if f.is_file()
            )
            self.assertEqual(built, bundle_files())
            for rel in built:
                self.assertEqual(
                    (pkg / "agentic" / rel).read_bytes(),
                    (BUNDLE / rel).read_bytes(),
                    rel,
                )
            self.assertFalse(list(pkg.rglob("__pycache__")))
            tars = list(out.glob("rocke_monitor-*.tar.gz"))
            self.assertTrue(tars)
            import tarfile

            with tarfile.open(
                tars[0]
            ) as t:  # nothing of the build machine in the headers: no owner names, ids 0
                members = t.getmembers()
            self.assertTrue(members)
            self.assertEqual(
                {(m.uname, m.gname, m.uid, m.gid) for m in members}, {("", "", 0, 0)}
            )
            self.assertFalse([m.name for m in members if m.mode & 0o022])
            self.assertEqual(
                sorted(m.name.split("/", 1)[1] for m in members if m.isfile()),
                sorted(str(p.relative_to(pkg)) for p in pkg.rglob("*") if p.is_file()),
            )
            # copied over the package in a sandbox checkout (no remote), below rocKE's ignore rules and its
            # provider's: git ignores none of its files
            co, _ = sandbox.fake_checkout(base)
            q = subprocess.run(
                build
                + [
                    "--out",
                    str(out),
                    "--deny",
                    str(deny),
                    "--no-tests",
                    "--install-into",
                    str(co),
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=600,
            )
            self.assertEqual(q.returncode, 0, q.stdout[-2000:] + q.stderr[-2000:])
            self.assertIn("0 ignored", q.stdout)
            subprocess.run(
                [
                    "git",
                    "-C",
                    str(co),
                    "remote",
                    "add",
                    "origin",
                    "https://github.com/ROCm/x.git",
                ],
                check=True,
            )
            r = subprocess.run(
                build
                + [
                    "--out",
                    str(out),
                    "--deny",
                    str(deny),
                    "--no-tests",
                    "--install-into",
                    str(co),
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=600,
            )
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("sandbox checkouts only", r.stdout + r.stderr)

    def test_a_failed_privacy_check_stops_the_build(self):
        """Step 5, the privacy check over the built folder, stops the build when it fails: here the deny list names
        one of the package's guides. No tarball follows."""
        with tempfile.TemporaryDirectory() as d:
            base = pathlib.Path(d)
            deny = base / "deny.txt"
            deny.write_text("QUICK_USER" + "_GUIDE\n", encoding="utf-8")
            out = base / "dist"
            p = subprocess.run(
                self.build + ["--out", str(out), "--deny", str(deny), "--no-tests"],
                capture_output=True,
                text=True,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                timeout=600,
            )
            self.assertEqual(p.returncode, 1, p.stdout[-2000:] + p.stderr[-2000:])
            self.assertIn("the privacy check failed: see privacy_report.txt", p.stderr)
            report = (out / "privacy_report.txt").read_text(encoding="utf-8")
            self.assertIn(f"  FAIL  {privacy.L_DENY}: 1 value(s) checked", report)
            self.assertIn("Result: FAIL", report)
            self.assertEqual(list(out.glob("*.tar.gz")), [])

    def test_a_deny_list_that_cannot_be_read_stops_the_build_at_step_2(self):
        """Step 2 reads the deny list (privacy.sources) and stops when it cannot be read: here it does not exist. Its
        row says so, and its fix is the build's command as run, naming another deny list; the privacy check never
        runs, and no tarball follows.
        """
        with tempfile.TemporaryDirectory() as d:
            base = pathlib.Path(d)
            out, missing = base / "dist", str(base / "no-such-deny.txt")
            p = subprocess.run(
                self.build + ["--out", str(out), "--deny", missing, "--no-tests"],
                capture_output=True,
                text=True,
                env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"),
                timeout=600,
            )
            self.assertEqual(p.returncode, 1, p.stdout[-2000:] + p.stderr[-2000:])
            self.assertIn(f"  FAIL  {privacy.S_DENY}: {privacy.NOT_READ}", p.stdout)
            # its fix, on one line (checks.render goes on at column 13)
            self.assertIn(
                f"        fix: python3 -B tools/make_package.py --out {out} --deny {privacy.FILE} --no-tests, "
                f"{privacy.FILE} being a deny list that can be read",
                re.sub(r"\n {13}", " ", p.stdout),
            )
            self.assertIn("the deny list cannot be read or gives no value", p.stderr)
            self.assertFalse((out / "privacy_report.txt").exists())
            self.assertEqual(list(out.glob("*.tar.gz")), [])


# Spellings of the operator's placeholder other than <user_login> (contributors are @<login>; a template's <reviewer>
# or <author> is anyone's).
OTHER_LOGIN_SPELLING = re.compile(
    r"<(?!user_login>)(?:(?:operator|me|my)(?:[_-]?(?:login|name|handle|account|id))?"
    r"|(?:reviewer|user|your)[_-]?(?:login|handle|account|id))>",
    re.I,
)
# The operator in a file an agent reads is <user_login>, which `render` fills in from the deploy settings'
# `integrations.github_cli.username`, so a post or a message names them by their login: never "the operator" or "the
# user" ("the user" in quotes stays: it is the wording a post must never use), and `<user_login>` is said to come from
# the settings once, at the top. The workspace's shared policies (skills/shared/) keep their words: there "the user" is
# whoever runs the session.
_SP = r"(?:[ \t]+|[ \t]*\n[ \t]*)"  # a space, or a line break in a wrapped paragraph
_WHO = r"\b[Tt]he" + _SP + "(?:operator|user)"
_CONFIG_LOGIN = (
    _SP
    + r"\(config"
    + _SP
    + r"`(?:integrations\.github_cli\.username|operator\.github_login)`\)"
)
OPERATOR_WORDS = tuple(
    re.compile(rx)
    for rx in (
        _WHO + r"(?:'s)?" + _CONFIG_LOGIN,
        _WHO + r"['’]s\b",
        r"(?<![\"“'‘])" + _WHO + r"\b(?![\w-]|[\"”'’])",
        re.escape("<user_login>") + r"(?:'s)?" + _CONFIG_LOGIN,
    )
)


def operator_words(text: str) -> list[str]:
    """The words in `text` that name the operator otherwise than <user_login> (OPERATOR_WORDS)."""
    return [m.group(0) for rx in OPERATOR_WORDS for m in rx.finditer(text)]


def operator_file(rel: str) -> bool:
    """A file an agent reads that names the operator <user_login>: every agentic file but the shared policies and
    the README."""
    return not rel.startswith("skills/shared/") and rel != "README.md"


class TestOperatorLogin(unittest.TestCase):
    """The operator is <user_login> in every file an agent reads; the toolkit fills it in from the deploy settings'
    integrations.github_cli.username; no agent guesses it, and no post carries the placeholder.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = pathlib.Path(self.tmp.name)

    def test_the_committed_files_name_the_operator_user_login(self):
        said = "`<user_login>` is the deploy settings' `integrations.github_cli.username`; the toolkit fills it in."
        self.assertEqual(text("skills/rocke-monitor/SKILL.md").count(said), 1)
        self.assertIn(said, text("review_book.md"))
        for rel in bundle_files():
            with self.subTest(rel=rel):
                body = text(rel)
                self.assertEqual(OTHER_LOGIN_SPELLING.findall(body), [])
                self.assertNotIn("config `me`", body)
                if operator_file(rel):
                    self.assertEqual(operator_words(body), [])

    def test_the_operator_words_check_finds_the_operator_named_otherwise(self):
        self.assertEqual(
            operator_words(
                "Tell the user in one line. The user's explicit re-approval lifts it; the operator (config\n"
                "`operator.github_login`) decides; ask <user_login> (config `integrations.github_cli.username`), "
                'never "the user"; the user-side API stays.\n'
                "Read <user_login>'s (config `integrations.github_cli.username`) comments in full.\n"
            ),
            [
                "the operator (config\n`operator.github_login`)",
                "The user's",
                "the user",
                "the operator",
                "<user_login> (config `integrations.github_cli.username`)",
                "<user_login>'s (config `integrations.github_cli.username`)",
            ],
        )
        self.assertTrue(operator_file("skills/rocke-monitor/SKILL.md"))
        self.assertFalse(operator_file("skills/shared/temporary_file_policy.md"))

    def test_render_fills_it_in_and_refuses_a_blank_login(self):
        out = self.base / "work" / "agentic"
        agentic.render(BUNDLE, a_config(self.base), out)
        skill = (out / "skills/rocke-monitor/SKILL.md").read_text(encoding="utf-8")
        self.assertIn("only the-operator's explicit re-approval lifts it", skill)
        self.assertNotIn("<user_login>", skill)
        blank = dict(a_config(self.base), operator={"github_login": ""})
        with self.assertRaises(ValueError) as cm:
            agentic.render(BUNDLE, blank, self.base / "work" / "other")
        self.assertEqual(str(cm.exception), agentic.LOGIN_MISSING)
        self.assertIn("integrations.github_cli.username is blank", str(cm.exception))
        self.assertFalse((self.base / "work" / "other").exists())

    def test_refresh_renders_again_when_the_login_changes(self):
        target = self.base / "target"
        cfg = a_config(self.base, claude_workspace=str(target))
        out = pathlib.Path(paths.get(cfg, "agentic"))
        agentic.render(BUNDLE, cfg, out)
        agentic.install(out, target)
        self.assertEqual(agentic.refresh(cfg, BUNDLE), [])  # current
        moved = dict(cfg, operator={"github_login": "new-op"})
        lines = agentic.refresh(moved, BUNDLE)
        self.assertTrue(lines and "rendered again" in lines[0], lines)
        skill = (target / ".claude/skills/rocke-monitor/SKILL.md").read_text(
            encoding="utf-8"
        )
        self.assertIn("only new-op's explicit re-approval lifts it", skill)
        self.assertEqual(agentic.refresh(moved, BUNDLE), [])
        with self.assertRaises(ValueError):
            agentic.refresh(dict(cfg, operator={"github_login": ""}), BUNDLE)


# An agent names a setting as the operator finds it: by its key in the deploy settings file, or by the registry's name
# for one the package keeps (a package default, internal state, a derived value, a name inside one of them). A
# configuration key the deploy settings file holds under a key of its own (`operator.github_login`, `repo`) is named by
# that key (deploy_tree.name). Markdown's bold (`**Scope.**`) ends no key.


def deploy_keys() -> frozenset:
    """Each key of the deploy settings file, every section above one included, as the tuple of its words."""
    schema = json.loads(
        pathlib.Path(file_types.DEPLOY.schema).read_text(encoding="utf-8")
    )
    return frozenset(
        tuple(path.split(".")[: n + 1])
        for path, _node, _required in settings_file.schema_keys(schema)
        for n in range(path.count(".") + 1)
    )


def misnamed_settings(bundle: pathlib.Path) -> list[str]:
    """`file:line: what` for each dotted key (doc_lints.KEY) in the agentic files (`bundle`, the README with them)
    that is no key of the deploy settings file and no registry name, or that the file holds under another key; and
    for each code span that is a one-word configuration key the file holds under another key.
    """
    known, deploy, held = doc_lints.load_keys(ROOT), deploy_keys(), deploy_tree.held()
    tops = {k[0] for k in known.settings | known.internal}
    out = []
    for f in sorted(bundle.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(bundle).as_posix()
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            for m in doc_lints.KEY.finditer(line.replace("**", "")):
                key, words = m.group(), tuple(m.group().split("."))
                if (
                    words[0] not in tops
                    or words[-1] in doc_lints.NOT_KEYS
                    or doc_lints._has(deploy, words)
                ):
                    continue
                named = deploy_tree.file_key(key)
                if named:
                    out.append(f"{rel}:{i}: {key}: the deploy settings name it {named}")
                elif not (
                    doc_lints._has(known.internal, words)
                    or any(
                        doc_lints._has(known.inside, words[:n])
                        for n in range(1, len(words))
                    )
                ):
                    out.append(f"{rel}:{i}: {key}: no deploy key and no registry name")
            for span in CODE_SPAN.findall(line):
                if "." not in span and span in held and (span,) not in deploy:
                    out.append(
                        f"{rel}:{i}: `{span}`: the deploy settings name it {held[span]}"
                    )
    return out


class TestSettingNames(unittest.TestCase):
    """The agentic files name each setting by its key in the deploy settings file, or by the registry's name for one
    the package keeps."""

    def test_each_setting_is_a_deploy_key_or_a_registry_name(self):
        self.assertEqual(misnamed_settings(BUNDLE), [])

    def test_the_check_finds_a_key_no_one_has_and_one_named_otherwise(self):
        with tempfile.TemporaryDirectory() as d:
            bundle = pathlib.Path(d)
            (bundle / "skills").mkdir()
            (bundle / "skills" / "rules.md").write_text(
                "- **Scope first.** `features.review_automation.parallel_reviews` and\n"
                "  `monitor.parallel_builds`; `offload.mode` and `roles.quality`.\n"
                "- Gap: `features.review_automation.schedule.follow_up_cooldown_min`;\n"
                "  issues in the repository `incident_issue.repo` names (blank: `repo`).\n",
                encoding="utf-8",
            )
            self.assertEqual(
                misnamed_settings(bundle),
                [
                    "skills/rules.md:3: features.review_automation.schedule."
                    "follow_up_cooldown_min: no deploy key and no registry name",
                    "skills/rules.md:4: incident_issue.repo: the deploy settings name it "
                    "features.merge_queue.develop_guard.issue_repository",
                    "skills/rules.md:4: `repo`: the deploy settings name it repository.name",
                ],
            )


class TestAutomatedReviewerName(unittest.TestCase):
    """The skill writes line 1 with <AUTOMATED_REVIEWER_NAME>, which the render fills in with this deployment's
    automated reviewer name; a change of the name renders the skill again.
    """

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = pathlib.Path(self.tmp.name)
        self.target = self.base / "target"

    def named(self, name: str) -> dict:
        cfg = a_config(self.base, claude_workspace=str(self.target))
        cfg["features"]["review_automation"]["automated_reviewer_name"] = name
        return cfg

    def installed(self, rel: str) -> str:
        return (self.target / ".claude/skills/rocke-monitor" / rel).read_text(
            encoding="utf-8"
        )

    def test_render_gives_the_name_and_a_rename_renders_again(self):
        cfg = self.named("team-a")
        out = pathlib.Path(paths.get(cfg, "agentic"))
        agentic.render(BUNDLE, cfg, out)
        agentic.install(out, self.target)
        line = "`Automated review by {}: Model - <drafter model>, effort: <effort>`"
        self.assertIn(line.format("team-a"), self.installed("rules.md"))
        self.assertNotIn("<AUTOMATED_REVIEWER_NAME>", self.installed("commands.md"))
        lines = agentic.refresh(self.named("team-b"), BUNDLE)
        self.assertTrue(lines and "rendered again" in lines[0], lines)
        self.assertIn(line.format("team-b"), self.installed("rules.md"))

    def test_each_notice_line_the_skill_gives_is_the_one_rk_writes(self):
        """The skill names each notice's first line as post.notice_line writes and checks it, so the orchestrator
        writes the line `rk` accepts."""
        from rkmon_lib import post

        cfg = self.named("team-a")
        out = pathlib.Path(paths.get(cfg, "agentic"))
        agentic.render(BUNDLE, cfg, out)
        agentic.install(out, self.target)
        skill = "".join(
            p.read_text(encoding="utf-8")
            for p in sorted((self.target / ".claude/skills/rocke-monitor").glob("*.md"))
        )
        for kind in post.NOTICES:
            with self.subTest(kind=kind):
                self.assertIn(f"`{post.notice_line(cfg, kind)}`", skill)

    def test_each_automated_line_the_skill_gives_is_one_rk_writes(self):
        """The converse: each `Automated ...` line the rendered skill, its agents and its team members give, each in
        backticks, is one rk writes: a notice's first line (post.notice_line), or a review's line 1 in
        reviewer_name.REVIEW_LINE's form with this deployment's name, whatever its model and effort.
        """
        from rkmon_lib import post, reviewer_name

        cfg = self.named("team-a")
        out = pathlib.Path(paths.get(cfg, "agentic"))
        agentic.render(BUNDLE, cfg, out)
        agentic.install(out, self.target)
        claude = self.target / ".claude"
        files = [
            *sorted((claude / "skills/rocke-monitor").glob("*.md")),
            *sorted((claude / "agents").glob("rocke-*.md")),
            *sorted((claude / "team_members").glob("rocke_*.md")),
        ]
        text = "".join(p.read_text(encoding="utf-8") for p in files)
        notices = {post.notice_line(cfg, kind) for kind in post.NOTICES}
        review = re.escape(reviewer_name.REVIEW_LINE)
        for field, rx in (
            ("name", re.escape(reviewer_name.name(cfg))),
            ("model", ".+"),
            ("effort", ".+"),
        ):
            review = review.replace(re.escape("{" + field + "}"), rx)
        lines = re.findall(r"`(Automated [^`]*)`", text)
        self.assertTrue(lines)
        self.assertEqual(
            len(lines), text.count("Automated "), "an Automated line outside backticks"
        )
        for line in lines:
            with self.subTest(line=line):
                self.assertTrue(line in notices or re.fullmatch(review, line), line)


class TestPostRefusesThePlaceholder(unittest.TestCase):
    def test_a_post_with_user_login_is_refused(self):
        from rkmon_lib import post

        self.assertTrue(
            [
                p
                for p in post.public("Thanks, <user_login> will re-approve.")
                if "<user_login>" in p
            ]
        )
        self.assertEqual(post.public("Thanks, op-login will re-approve."), [])


if __name__ == "__main__":
    unittest.main()
