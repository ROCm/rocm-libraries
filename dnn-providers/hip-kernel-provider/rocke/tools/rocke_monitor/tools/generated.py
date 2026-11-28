# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The package's generated parts: each section of a doc between its markers, and each generated file,
made from its one source through the functions the commands print with at run time, never from a copy of it.

- REFRESH: the one command that writes them again from their sources, `python3 -B tools/make_package.py
  --refresh-generated`, which calls `refresh`. RELEASE: the build of a release, the one command that writes VERSION.
- `Doc(path, reader, job, kind)`: a doc of the index, by its path in the package: who reads it, its one job, and
  its kind, USER, MAINTAINER or SKILL. DOCS: the index, every doc in its order, the one home of each doc's job
  and kind. A doc's row comes with the change that writes the doc: the doc map lists every row, whatever files the
  tree it is made in holds, and the docs' links lint (tools/doc_lints.py) fails a row whose doc the package lacks.
  `kind_of(path)`: a doc's kind: the index's, else SKILL under agentic/ and .claude/, else MAINTAINER.
- `claude_skills()`: the skills the package ships in CLAUDE_SKILLS, by their folders' names: the folder of each doc of
  the index there, a skill's SKILL.md. The one list of them: .gitignore lets each one in (LET_IN_GIT).
- `Section(name, source, places, render)`: a generated section: its name in its markers; its source, as its BEGIN
  marker names it; the docs it goes in; and `render(root, place) -> str`, its text, made from this package's
  sources: ROOT is the package folder whose docs are read, PLACE the doc's path in it. SECTIONS: each, by name.
  A source that is not written yet has no section here: its renderer is an addition of its own, a
  function and its row.
- `File(path, source, render)`: a generated file: its path, its source, and `render(root) -> str`, its whole text.
  FILES: each, in order. VERSION's render is None: its text is the package's version, the day of its release
  (make_package.version_text), which RELEASE writes and no source gives, so `drift` checks only its form, and
  `refresh` never writes it.
- A render, a section's or a file's, reads no file of ROOT that the image leaves out (its .dockerignore): the image's
  suite checks every generated part over the image's own files.
- The markers, each at the start of its line: `<!-- BEGIN GENERATED: NAME`, then its `source:` and `refresh:`
  lines, the last ending in `-->`; the section's text; `<!-- END GENERATED: NAME -->`. A BEGIN on one line,
  `<!-- BEGIN GENERATED: NAME -->`, is one too, and `refresh` writes it whole. A line that starts so inside fenced
  code, outside a section, is text.
- `blocks(text) -> (blocks, problems)`: the sections of a doc's text, each a `Block(name, first, last)`, the lines of
  its BEGIN and its END (from 0); and what is wrong with its markers, each `LINE: what` (from 1): a marker that is not
  one, a name SECTIONS lacks, a BEGIN with no END, a BEGIN inside a section, an END with no BEGIN or of another name.
- `section_text(section, root, place) -> str`: the section whole, from its BEGIN marker to its END.
- `drift(root, files=FILES, docs=None) -> list`: how the package at ROOT is out of step with its sources: for a doc,
  `PATH:LINE: what` (each marker problem; a section in a doc that is not its place; a section unlike its text, named
  with its source and REFRESH; one its source cannot make), then, once its markers are right, `PATH: what` for each
  section whose places name the doc and that it lacks, with the markers to add; for a file, `PATH: what` (missing,
  unlike its text, or VERSION not in its form, named with RELEASE). DOCS: the docs to read, by their paths in ROOT;
  every .md file of the package (make_package.tracked) when None. [] when the package is in step.
- `refresh(root, files=FILES, docs=None) -> list`: writes each doc and file whose text differs from what its sources
  make, and only those; returns their paths, sorted. A marker problem in any doc, a section in a doc that is not its
  place, or a text its source cannot make, is a ValueError naming each, and then nothing is written. A section a doc
  lacks is never added: where it goes is a person's choice, and `drift` names it.
- Markdown for a table: `cell(text, markdown=False)`, a cell, each line break a <br>: plain words with each character
  markdown reads as markup escaped, or (MARKDOWN) a text that is markdown already; `code(text, in_table=True)`, a code
  span (its | escaped in a table); `table(header, rows)`; `fenced(text)`, a fenced block of text.

Every text names the program rkmon (rkmon_lib/program.py), whatever runs the generator.
"""

from __future__ import annotations

import ast
import contextlib
import copy
import dataclasses
import datetime as dt
import io
import itertools
import json
import pathlib
import posixpath
import re
import sys
from typing import Callable, NamedTuple, Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _folder in (ROOT, ROOT / "tools"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))
sys.dont_write_bytecode = True

import make_package  # noqa: E402
import template_texts  # noqa: E402
from rkmon_lib import (  # noqa: E402
    agentic,
    answers,
    build_plan,
    commands,
    deploy_plan,
    deploy_tree,
    deployment,
    derived,
    features,
    file_types,
    gh,
    identity,
    image,
    outcome,
    outward,
    paths,
    post,
    program,
    registry,
    secrets_registry,
    settings_file,
    states,
    steps,
    wrapping,
)
from rkmon_lib.checks import Command, Described, Edit  # noqa: E402

REFRESH = "python3 -B tools/make_package.py --refresh-generated"
RELEASE = "python3 -B tools/make_package.py --out DIR --deny FILE --new-version"

# --- the docs index ---------------------------------------------------------------------------------------------------

USER, MAINTAINER, SKILL = "user", "maintainer", "skill"
KINDS = (USER, MAINTAINER, SKILL)


class Doc(NamedTuple):
    """A doc of the index."""

    path: str
    reader: str
    job: str
    kind: str


DOCS = (
    Doc("README.md", "everyone", "the reference", USER),
    Doc(
        image.BUILD_GUIDE,
        "the image builder",
        "from a build settings file to a shipped image",
        USER,
    ),
    Doc(
        image.DEPLOY_GUIDE,
        "the first-time deployer",
        "from an image to a first tick",
        USER,
    ),
    Doc(
        image.USER_GUIDE,
        "the operator",
        "day to day, after the first tick",
        USER,
    ),
    Doc(
        "docs/FAQ.md",
        "deployers, pull request authors, team leads",
        "short answers, each ending in a link to the fact's home",
        USER,
    ),
    Doc(
        "docs/restart.md",
        "the operator during a restart",
        "is it safe now; the stop levels; what a restart keeps; what to check after",
        USER,
    ),
    Doc(
        "docs/design.md",
        "maintainers, stakeholders",
        "why it is built this way",
        MAINTAINER,
    ),
    Doc(
        "docs/operations_guide.md",
        "stakeholders",
        "the overview: the problems it solves, what it produces, guardrails, benefits, limits, and the policies the "
        "operator keeps",
        MAINTAINER,
    ),
    Doc(
        "docs/review_pipeline.md",
        "maintainers, reviewers",
        "how a review is made; known issues",
        MAINTAINER,
    ),
    Doc(
        "docs/maintaining.md",
        "maintainers",
        "build, test, publish and regenerate the package, and make the package tarball",
        MAINTAINER,
    ),
    Doc(
        "docs/internal_settings.md",
        "maintainers",
        "the internal keys: package defaults, state and derived, with their sources and descriptions",
        MAINTAINER,
    ),
    Doc("agentic/README.md", "the Claude session", "the agents' guidance", SKILL),
    Doc(
        ".claude/skills/rkmon-setup/SKILL.md",
        "the deployer's own Claude Code session",
        "walk a deployer through deploy steps 1 and 2",
        SKILL,
    ),
)
SKILL_FOLDERS = ("agentic/", ".claude/")


def kind_of(path: str) -> str:
    """A doc's kind: the index's, else SKILL under agentic/ and .claude/, else MAINTAINER."""
    rel = pathlib.PurePosixPath(path).as_posix()
    for d in DOCS:
        if d.path == rel:
            return d.kind
    return SKILL if rel.startswith(SKILL_FOLDERS) else MAINTAINER


# The folder of the skills a Claude Code session in the package folder itself finds (the setup skill's).
CLAUDE_SKILLS = ".claude/skills/"


def claude_skills() -> tuple:
    """The skills the package ships in CLAUDE_SKILLS, by their folders' names: the folder of each doc of the index
    there, a skill's SKILL.md."""
    return tuple(
        pathlib.PurePosixPath(d.path).relative_to(CLAUDE_SKILLS).parts[0]
        for d in DOCS
        if d.path.startswith(CLAUDE_SKILLS)
    )


# --- markdown ---------------------------------------------------------------------------------------------------------

_ESCAPED = "\\`*~<|[$"
_ENTITY = re.compile(r"&(?:#[0-9]+|#[xX][0-9a-fA-F]+|[A-Za-z][A-Za-z0-9]*);")


def cell(text, markdown: bool = False) -> str:
    """TEXT in a table cell, each line break a <br>. As plain words (the default): each character markdown would read
    as markup escaped (an _ within a word aside), and an & that would start an entity written as &amp;. With MARKDOWN,
    TEXT is markdown already: only its | is escaped."""
    t = str(text).strip()
    if markdown:
        return t.replace("|", "\\|").replace("\n", "<br>")
    out = []
    for i, ch in enumerate(t):
        if ch in _ESCAPED:
            out.append("\\" + ch)
        elif ch == "_" and not (
            0 < i < len(t) - 1 and t[i - 1].isalnum() and t[i + 1].isalnum()
        ):
            out.append("\\_")
        elif ch == "&" and _ENTITY.match(t, i):
            out.append("&amp;")
        elif ch == "\n":
            out.append("<br>")
        else:
            out.append(ch)
    return "".join(out)


def code(text, in_table: bool = True) -> str:
    """TEXT as a code span on one line: its backticks fenced by a longer run, and in a table its | escaped."""
    t = " ".join(str(text).split("\n"))
    longest = max((len(run) for run in re.findall(r"`+", t)), default=0)
    fence = "`" * (longest + 1)
    pad = " " if t.startswith("`") or t.endswith("`") else ""
    if in_table:
        t = t.replace("|", "\\|")
    return f"{fence}{pad}{t}{pad}{fence}"


def table(header, rows) -> str:
    """A markdown table: the header, then each row, every cell given as markdown."""
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        if len(row) != len(header):
            raise ValueError(f"a row of {len(row)} cells under {len(header)} columns")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def fenced(text: str, info: str = "text") -> str:
    """TEXT as a fenced block, its fence longer than any fence in TEXT, each line's trailing blanks dropped."""
    lines = [line.rstrip() for line in str(text).strip("\n").split("\n")]
    longest = max(
        (len(m.group(1)) for line in lines for m in [re.match(r"\s*(`+)", line)] if m),
        default=0,
    )
    fence = "`" * max(3, longest + 1)
    return "\n".join([fence + info, *lines, fence])


def _listed(keys) -> str:
    return wrapping.joined([code(k, in_table=False) for k in keys])


def _under(key: str, prefix: str) -> bool:
    return key == prefix or key.startswith(prefix + ".")


def _of(module, *names: str) -> str:
    """A source as a marker names it: the module's file in the package, and the names in it."""
    rel = pathlib.Path(module.__file__).resolve().relative_to(ROOT).as_posix()
    return ", ".join((rel, *names))


def _schema_of(file_type, *names: str) -> str:
    rel = pathlib.Path(file_type.schema).resolve().relative_to(paths.PACKAGE.resolve())
    return ", ".join((rel.as_posix(), *names))


@contextlib.contextmanager
def _as_rkmon():
    """The program's name pinned to rkmon (rkmon_lib/program.py) while a text is made."""
    saved = list(sys.argv)
    sys.argv[:] = ["", *saved[1:]]
    try:
        yield
    finally:
        sys.argv[:] = saved


# --- the schemas ------------------------------------------------------------------------------------------------------


def _nodes(file_type) -> dict:
    """Each key of the file type's schema by its path, a map's entries as *, with its schema node."""
    return {path: node for path, node, _ in settings_file.schema_keys(file_type.load())}


def _shown(file_type) -> dict:
    """A key's path, a map's entries as *, -> the key as the every-key text names it (`model_bindings.<name>`)."""
    out = {}
    for c in settings_file.comments(file_type):
        out.setdefault(re.sub(r"(?<![^.])<[^<>.]+>(?![^.])", "*", c.key), c.key)
    return out


def _default(file_type, path: str):
    node = _nodes(file_type).get(path)
    if node is None or "default" not in node:
        raise ValueError(f"{path} has no default in {_schema_of(file_type)}")
    return node["default"]


def _blank(value) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _value(data, path: str):
    """The value at PATH in DATA, a template read back, or None where it has none (a map's entries, *, included)."""
    for word in path.split("."):
        if not isinstance(data, dict) or word not in data:
            return None
        data = data[word]
    return data


# --- the renderers ----------------------------------------------------------------------------------------------------


def _doc_map(root: pathlib.Path, place: str) -> str:
    here = posixpath.dirname(place) or "."
    rows = [
        (
            f"[{cell(d.path)}]({posixpath.relpath(d.path, here)})",
            cell(d.reader),
            cell(d.job),
        )
        for d in DOCS
    ]
    return table(("Doc", "Reader", "Its one job"), rows)


def _capabilities(root: pathlib.Path, place: str) -> str:
    rows = [
        (cell(c.title), cell(c.when), cell(c.what), cell(c.needs))
        for c in features.CAPABILITIES.values()
    ]
    return table(("Capability", "Runs while", "What it does", "What it needs"), rows)


def _profiles(root: pathlib.Path, place: str) -> str:
    deploy = file_types.DEPLOY
    rows = []
    for p in settings_file.profiles(deploy):
        values = "<br>".join(
            code(f"{key}: {json.dumps(value)}") for key, value in p.values.items()
        )
        write = steps.command_line(deploy.template_command, PROFILE=p.name)
        rows.append(
            (
                code(p.name),
                cell(p.what),
                values or cell("every key's default"),
                code(str(write)),
            )
        )
    return table(("Profile", "What it runs", "Its values", "Write it with"), rows)


def _path(path_steps) -> Callable:
    def render(root: pathlib.Path, place: str) -> str:
        prog = program.name()
        rows = []
        for s in path_steps:
            aside = steps.aside(s, lambda c: code(f"{prog} {c}"))
            command = code(f"{prog} {s.command}") + "".join(
                f"<br>{line}" for line in aside if line
            )
            rows.append((str(s.n), cell(s.name), command, cell(s.what)))
        return table(("#", "Step", "Command", "What it does"), rows)

    return render


# The facts of a deployment that runs its confirmed settings: each state's examples change some of them.
SAME = "sha256:" + "1" * 64
NEWER = "sha256:" + "2" * 64
OLDER = "sha256:" + "0" * 64


def _facts(**changes) -> states.Facts:
    """A deployment's facts, as a state's example gives them: the package's own names, the suggested file, an image
    whose tag is a version's form (YYYY.MM.DD), one change confirmed; CHANGES in place of any of them.
    """
    deploy = file_types.DEPLOY
    facts = dict(
        name_owner=identity.OURS,
        container_name=_default(deploy, "container.name"),
        settings_file=states.SUGGESTED_FILE,
        file_exists=True,
        under_way="",
        under_way_runs=False,
        first_apply=False,
        file_layout=deploy.version,
        package_layout=deploy.version,
        file_changed=False,
        package_changed=False,
        image=f"{_default(deploy, 'image.name')}:YYYY.MM.DD",
        image_here=True,
        registry=False,
        logins_missing=(),
        container=states.RUNNING,
        running_minutes=60.0,
        not_set_up=False,
        put_back="",
        put_back_fix=None,
        confirmed_fingerprint=SAME,
        confirmed_at="",
        waiting_changes=1,
        delivered_fingerprint=SAME,
        loaded_fingerprint=SAME,
        hand_edited=(),
        standby=False,
        paused=False,
        first_tick_recorded=False,
    )
    facts.update(changes)
    return states.Facts(**facts)


def _state_examples() -> dict:
    """Each state's number -> the changes to `_facts` of each way it reads (a state can read two ways)."""
    sr = secrets_registry
    login = next(
        e.name
        for e in sr.entries({})
        if e.shape == sr.LOGIN and sr.ALWAYS in e.needed_when
    )
    layout = file_types.DEPLOY.version
    model = Edit(
        "model_bindings.fast.base_url",
        states.SUGGESTED_FILE,
        Described("an address the container reaches"),
    )
    put_back = dict(delivered_fingerprint=OLDER, loaded_fingerprint=OLDER)
    waits = dict(delivered_fingerprint=NEWER, loaded_fingerprint=NEWER)
    stopped = dict(container=states.STOPPED, running_minutes=0.0)
    return {
        1: [dict(name_owner=identity.OTHER)],
        2: [
            dict(
                settings_file="",
                name_owner=identity.NONE,
                container=states.NO_CONTAINER,
            ),
            dict(file_exists=False),
        ],
        3: [dict(under_way="apply"), dict(under_way="rollback")],
        4: [dict(file_layout=layout - 1), dict(file_layout=layout + 1)],
        5: [dict(file_changed=True), dict(package_changed=True)],
        6: [dict(image_here=False), dict(image_here=False, registry=True)],
        7: [dict(logins_missing=(login,))],
        8: [
            dict(
                container=states.NO_CONTAINER,
                running_minutes=0.0,
                loaded_fingerprint="",
            ),
            dict(not_set_up=True, loaded_fingerprint=""),
        ],
        9: [dict(hand_edited=("docker/.env",))],
        10: [
            dict(
                put_back=f"the last change you confirmed was put back by {steps.command_line('rollback')}",
                **put_back,
            ),
            dict(
                put_back="the last change you confirmed was put back after a failed doctor check: model fast: no "
                "answer from the binding's base_url",
                put_back_fix=model,
                **put_back,
            ),
        ],
        11: [waits, dict(waits, **stopped)],
        12: [stopped],
        13: [
            dict(
                running_minutes=float(states.LOAD_WAIT_MINUTES),
                loaded_fingerprint=OLDER,
            )
        ],
        14: [dict(standby=True)],
        15: [dict(paused=True)],
        16: [{}],
    }


def _states(root: pathlib.Path, place: str) -> str:
    examples = _state_examples()
    # the table's next lines are a deployment's past deploy step 8: the doctor has passed and a tick has run since the
    # container's start
    ran = states.SinceStart(True, "2026-01-02 03:04 UTC")
    rows = []
    for s in states.TABLE:
        if s.n not in examples:
            raise ValueError(
                f"state {s.n}, {s.name}, has no example (tools/generated.py, _state_examples)"
            )
        for i, changes in enumerate(examples[s.n]):
            facts = _facts(**changes)
            if states.current(facts) != s:
                raise ValueError(
                    f"an example of state {s.n} is state {states.current(facts).n} "
                    "(tools/generated.py, _state_examples)"
                )
            rows.append(
                (
                    "" if i else str(s.n),
                    "" if i else cell(s.name),
                    cell(states.sentence(s, facts)),
                    code(states.next_line(s, facts, ran)),
                )
            )
    standby = _facts(standby=True)
    itself, last = states.current(standby), states.TABLE[-1]
    ending = states.sentence(last, standby)[len(last.words(standby)) :]
    first, login = states.TABLE[0], states.TABLE[7 - 1]
    # before step 8 has passed: state 7's line, and the walk of steps 7 and 8 (no tick since the start yet), as state
    # 16's line has it
    login_line = code(
        states.next_line(login, _facts(**examples[login.n][0]), states.NOT_STARTED),
        False,
    )
    walk = [
        code(
            states.next_line(
                last, _facts(**examples[last.n][0]), states.SinceStart(passed, "")
            ),
            False,
        )
        for passed in (False, True)
    ]
    refusal = io.StringIO()
    # this folder's own container cannot be asked while its name is another deployment's: nothing has passed
    states.refuse(
        first, _facts(**examples[first.n][0]), out=refusal, since=states.NOT_STARTED
    )
    earlier = wrapping.joined([str(n) for n in states.EARLIER_STEPS])
    walkers = wrapping.joined([str(n) for n in states.WALKS_7_AND_8])
    remove = code(steps.command_line("remove --delete-data"), False)
    return "\n\n".join(
        [
            table(("#", "State", "What it says", "Next line"), rows),
            cell(
                "Until deploy step 8 has passed after a start (a tick has run since a start: the first command that "
                f"sees one keeps that in this folder's records, until {remove} removes the deployment's volumes), a "
                f"next command that is a deploy step's carries that step's number and name: in state {login.n}, "
                f"{login_line}. While the container runs, the line of states {walkers} is {walk[0]} until the doctor "
                f"has passed since the start, then {walk[1]} until a tick has run.",
                markdown=True,
            ),
            cell(
                f"While standby is on, every state's sentence says so: state {itself.n}'s itself, and every other's "
                f'ends "{ending}".'
            ),
            cell(
                "A command that cannot run in the deployment's state refuses: it names the state's sentence, the "
                f"fix when there is one, and the state's next line. In state {first.n}:"
            ),
            fenced(refusal.getvalue()),
            cell(
                f"A refusal exits {outcome.EARLIER_STEP} in states {earlier}, where an earlier step is not done, and "
                f"{outcome.code('REFUSED')} in any other."
            ),
        ]
    )


def _exit_codes(root: pathlib.Path, place: str) -> str:
    rows = [(str(n), cell(long)) for n, _short, long in outcome.EXIT_CODES]
    return table(("Exit code", "What it means"), rows)


def _settings_table(comments) -> str:
    rows = []
    for c in comments:
        key = code(c.key)
        rows.append((f"**{key}**" if c.groups else key, cell(c.text)))
    return table(("Key", "What it is"), rows)


BINDINGS = "model_bindings"


def _deploy_settings(root: pathlib.Path, place: str) -> str:
    return _settings_table(
        c
        for c in settings_file.comments(file_types.DEPLOY)
        if not _under(c.key, BINDINGS)
    )


def _model_bindings(root: pathlib.Path, place: str) -> str:
    found = [
        c for c in settings_file.comments(file_types.DEPLOY) if _under(c.key, BINDINGS)
    ]
    if not found or found[0].key != BINDINGS:
        raise ValueError(f"the deploy schema's every-key text has no {BINDINGS}")
    parts, rows = [cell(found[0].text)], []
    for c in found[1:]:
        if c.groups and c.key.count(".") == 1:  # a form's heading
            if rows:
                parts.append(_settings_table(rows))
            parts.append(cell(c.text))
            rows = []
        else:
            rows.append(c)
    if rows:
        parts.append(_settings_table(rows))
    return "\n\n".join(parts)


def given_to_the_build() -> dict:
    """What `rkmon build` gives the build from each build settings key that gives it something
    (rkmon_lib/build_plan.py): a build argument of docker/Dockerfile, a build secret (a download location), or the root
    certificates' build context."""
    given = {a.key: f"build argument {code(a.name)}" for a in build_plan.ARGS if a.key}
    given.update({d.key: f"build secret {code(d.id)}" for d in build_plan.DOWNLOADS})
    given[deployment.CA_KEY] = f"build context {code(build_plan.CA_CONTEXT)}"
    return given


def _build_settings(root: pathlib.Path, place: str) -> str:
    """The build settings file's keys, each with what the build is given from it (— for a key that gives it nothing);
    a section's row in bold, with no such cell."""
    given, rows = given_to_the_build(), []
    for c in settings_file.comments(file_types.BUILD):
        key = code(c.key)
        if c.groups:
            rows.append((f"**{key}**", "", cell(c.text)))
        else:
            rows.append((key, given.get(c.key, "—"), cell(c.text)))
    return table(("Key", "Given to the build as", "What it is"), rows)


def _placeholder_keys(text: str) -> list:
    """The keys of a template's TEXT whose value is a placeholder (<...>), or a list with one, in the file's order."""
    found = []

    def walk(value, path: str) -> None:
        if isinstance(value, dict):
            for k, v in value.items():
                if not settings_file.PLACEHOLDER.fullmatch(str(k)):
                    walk(v, f"{path}.{k}" if path else str(k))
        elif isinstance(value, list):
            if any(
                isinstance(v, str) and settings_file.PLACEHOLDER.search(v)
                for v in value
            ):
                found.append(path)
        elif isinstance(value, str) and settings_file.PLACEHOLDER.search(value):
            found.append(path)

    walk(answers.loads(text), "")
    return found


def _keys_cell(keys) -> str:
    """Keys in a table cell; two or more under one section as the section, then each key's name in it."""
    parts = [k.split(".") for k in keys]
    common = []
    if len(parts) > 1:
        for words in zip(*(p[:-1] for p in parts)):
            if len(set(words)) > 1:
                break
            common.append(words[0])
    if not common:
        return ", ".join(code(k) for k in keys)
    rest = [".".join(p[len(common) :]) for p in parts]
    return f"in {code('.'.join(common))}: " + ", ".join(code(r) for r in rest)


def _required_keys(root: pathlib.Path, place: str) -> str:
    deploy = file_types.DEPLOY
    fill = {
        p.name: _placeholder_keys(settings_file.template(deploy, p.name))
        for p in settings_file.profiles(deploy)
    }
    placeholder = code("<...>", in_table=False)
    parts = []
    for name, keys in (
        fill.items()
        if len(set(map(tuple, fill.values()))) > 1
        else [("", next(iter(fill.values()), []))]
    ):
        whose = (
            f"the {code(name, in_table=False)} profile's template"
            if name
            else "each profile's template"
        )
        if keys:
            parts.append(
                f"In {whose}, these keys have a {placeholder} value for you to fill in: {_listed(keys)}."
            )
        else:
            parts.append(f"In {whose}, no key has a {placeholder} value to fill in.")
    # The every-key text: a key it leaves blank, or has no value for, is one you fill in when its condition is true.
    every = answers.loads(settings_file.template(deploy, None, True))
    shown, when = _shown(deploy), {}
    for path, node in _nodes(deploy).items():
        words = str(node.get("x-required-when") or "").strip()
        if words and _blank(_value(every, path)):
            when.setdefault(words, []).append(shown.get(path, path))
    if when:
        parts.append(
            "These keys are blank in the templates, and each needs a value when the condition beside it is true:"
        )
        parts.append(
            table(
                ("Required when", "Keys"),
                [(cell(words), _keys_cell(keys)) for words, keys in when.items()],
            )
        )
    return "\n\n".join(parts)


# Under the GitHub logins' permissions: why the monitor watches public repositories only (gh.CALLS: a check run needs
# Checks, which no fine-grained token is given).
PUBLIC_ONLY = (
    "The monitor watches public repositories: no fine-grained token can read a private repository's "
    "check runs."
)


def _access(permissions) -> str:
    """PERMISSIONS, (permission, access) pairs, as a cell, grouped by access ("Contents, Issues: write; Metadata:
    read"); "none" for none."""
    access = {}
    for permission, how in permissions:
        access.setdefault(how, []).append(permission)
    return cell(
        "; ".join(f"{', '.join(p)}: {how}" for how, p in access.items()) or "none"
    )


def _logins(root: pathlib.Path, place: str) -> str:
    sr = secrets_registry
    rows, made = [], []
    for e in sr.entries({}):
        needs = "; ".join(sr.condition_words(c) for c in e.needed_when)
        rows.append(
            (
                "login" if e.shape == sr.LOGIN else "secret",
                code(e.name),
                cell(f"the {e.keeper}"),
                code(str(Command((program.name(), *sr.set_command(e))))),
                cell(needs or "nothing: it is used when it is set"),
            )
        )
        if e.permissions:
            # the organization's permissions (gh.ORGANIZATION_PERMISSIONS) in a column of their own, as GitHub's
            # token page lists them
            on_org = [p for p in e.permissions if p[0] in gh.ORGANIZATION_PERMISSIONS]
            made.append(
                (
                    code(e.name),
                    cell(e.made_by),
                    _access([p for p in e.permissions if p not in on_org]),
                    _access(on_org),
                )
            )
    parts = [table(("Type", "Name", "Kept for", "Installed by", "Needed by"), rows)]
    if made:
        header = (
            "Login",
            "What to make",
            "Repository permissions",
            "Organization permissions",
        )
        parts += [table(header, made), PUBLIC_ONLY]
    return "\n\n".join(parts)


# The override's example: the RAM disk's size in GiB, a whole number and a part, so that it shows the size rounded up.
EXAMPLE_RAMDISK_GIB = 2.5


def _served_example() -> deploy_plan.DeployPlan:
    """What deploy_plan makes of the first profile's template with host.resources.ramdisk_gib at EXAMPLE_RAMDISK_GIB
    and the team dashboard served: by the first values of its switch keys (deploy_plan.TEAM_SWITCH; a boolean's
    true first, a choice's in the schema's order) that serve it (deploy_plan.served). A switch key that is neither a
    boolean nor a choice, or no values that serve it, is a ValueError."""
    deploy = file_types.DEPLOY
    nodes, choices = _nodes(deploy), []
    for key in deploy_plan.TEAM_SWITCH:
        node = nodes[key]
        if "enum" in node:
            choices.append(node["enum"])
        elif node.get("type") == "boolean":
            choices.append([True, False])
        else:
            raise ValueError(
                f"{key}, a switch key of the team dashboard, is neither a boolean nor a choice"
            )
    first = settings_file.profiles(deploy)[0].name
    base = settings_file.loads(settings_file.template(deploy, first), deploy)
    for chosen in itertools.product(*choices):
        values = copy.deepcopy(base.values)
        for key, value in [
            *zip(deploy_plan.TEAM_SWITCH, chosen),
            (deploy_plan.RAMDISK_GIB, EXAMPLE_RAMDISK_GIB),
        ]:
            deploy_tree.put(values, key, value)
        doc = dataclasses.replace(base, values=values)
        if deploy_plan.served(deploy_plan.settings(doc)):
            return deploy_plan.plan(doc, "", "", deploy_plan.Host(0, 0, ""))
    raise ValueError(
        "no values of the team dashboard's switch keys serve it (deploy_plan.served)"
    )


def _env_and_ports(root: pathlib.Path, place: str) -> str:
    """docker/.env's variables in their order, each with the deploy settings it is made from (deploy_plan.ENV_KEYS): a
    key's default with what it means (x-default-means); the switch keys (TEAM_SWITCH) as the condition the variable is
    there under; and for a variable made from none, the value the plan writes. Then the keys the override is made from
    (OVERRIDE_KEYS), and the override the plan makes while the team dashboard is served (`_served_example`), without
    its comment lines."""
    nodes, example = _nodes(file_types.DEPLOY), _served_example()
    switch, rows = deploy_plan.TEAM_SWITCH, []
    for name, keys in deploy_plan.ENV_KEYS.items():
        made = []
        for k in keys:
            if k in switch:
                continue
            means = str(nodes.get(k, {}).get("x-default-means") or "").strip()
            made.append(
                code(k) + (f" ({cell('at its default: ' + means)})" if means else "")
            )
        on = [k for k in keys if k in switch]
        if on:
            made.append(
                f"{cell('only while the container serves the team dashboard')} ({_keys_cell(on)})"
            )
        if not keys:
            if name not in example.variables:
                raise ValueError(
                    f"{name}: made from no deploy setting, and the plan writes no value of its own"
                )
            made.append(f"{cell('no setting: always')} {code(example.variables[name])}")
        rows.append((code(name), "; ".join(made)))
    shown = [x for x in example.override.split("\n") if x and not x.startswith("#")]
    ramdisk = code(deploy_plan.RAMDISK_GIB, in_table=False)
    return "\n\n".join(
        [
            "Each variable of docker/.env, and the deploy settings it is made from:",
            table(("Variable", "Made from"), rows),
            f"docker/compose.override.yaml is made from {_listed(deploy_plan.OVERRIDE_KEYS)}. While the container "
            f"serves the team dashboard, with {ramdisk} at {EXAMPLE_RAMDISK_GIB:g}, it adds this to "
            "docker/compose.yaml:",
            fenced("\n".join(shown), "yaml"),
        ]
    )


from rkmon_lib.change_plan import NEW_CONTAINER, NO_RESTART  # noqa: E402  (the classes)

# How a change of a key applies, by its class (x-applies): every key is DEFAULT_CLASS unless a test proves it
# another. NO_CLASS: the keys and sections that have none, and when a change of them applies.
CLASS_WORDS = {
    NO_RESTART: "at its next use; no restart",
    NEW_CONTAINER: "at a restart: {program} apply lets the work in flight finish, then makes the container again",
}
DEFAULT_CLASS = NEW_CONTAINER
NO_CLASS = {
    "container.name": "never on a deployment that exists: {program} confirm refuses it",
    "updates": "when you run {program} package update, which reads your file as it is then",
}


def _subtrees(classes: dict, cls: str) -> list:
    """The keys of class CLS, each as the highest section all of whose keys are of CLS, in the file's order."""
    out = []
    for key, c in classes.items():
        if c != cls:
            continue
        words = key.split(".")
        top = next(
            prefix
            for prefix in (".".join(words[:i]) for i in range(1, len(words) + 1))
            if all(v == cls for k, v in classes.items() if _under(k, prefix))
        )
        if top not in out:
            out.append(top)
    return out


def takes_effect_rows(classes: dict, shown: Optional[dict] = None) -> list:
    """The takes-effect table's rows from CLASSES, each key's path -> its x-applies (None for none), and SHOWN, a
    path -> the key as the every-key text names it: a row for each class but DEFAULT_CLASS that a key has, then
    "anything else", then each NO_CLASS row that names a key. A class CLASS_WORDS lacks, a key with no class that
    NO_CLASS does not name, or a NO_CLASS row that names a key with a class, is a ValueError.
    """
    shown = shown or {}
    unknown = sorted({c for c in classes.values() if c is not None} - set(CLASS_WORDS))
    if unknown:
        raise ValueError(
            f"x-applies {wrapping.joined(unknown)}: no such class ({wrapping.joined(list(CLASS_WORDS))})"
        )
    loose = [
        k
        for k, c in classes.items()
        if c is None and not any(_under(k, n) for n in NO_CLASS)
    ]
    if loose:
        raise ValueError(
            f"no x-applies for {wrapping.joined(loose)}: give each its class, or say when a change of it applies "
            "(tools/generated.py, NO_CLASS)"
        )
    rows = []
    for cls, words in CLASS_WORDS.items():
        if cls != DEFAULT_CLASS and cls in classes.values():
            keys = [code(shown.get(k, k)) for k in _subtrees(classes, cls)]
            rows.append((", ".join(keys), cell(program.named(words))))
    rows.append(
        (cell("anything else"), cell(program.named(CLASS_WORDS[DEFAULT_CLASS])))
    )
    for name, words in NO_CLASS.items():
        under = [c for k, c in classes.items() if _under(k, name)]
        if any(c is not None for c in under):
            raise ValueError(
                f"{name} has a class (x-applies), and tools/generated.py's NO_CLASS says it has none"
            )
        if under:
            rows.append((code(name), cell(program.named(words))))
    return rows


def _takes_effect(root: pathlib.Path, place: str) -> str:
    deploy = file_types.DEPLOY
    classes = {path: node.get("x-applies") for path, node in _nodes(deploy).items()}
    return table(
        ("You change", "It applies"), takes_effect_rows(classes, _shown(deploy))
    )


def _rkmon_commands(root: pathlib.Path, place: str) -> str:
    """The command table's commands, in its order: each in the form `rkmon help` lists it in (commands.form, its one
    home), and its summary, its help's first line."""
    prog = program.name()
    rows = [
        (code(f"{prog} {commands.form(row)}"), cell(program.named(row.summary)))
        for row in commands.TABLE
    ]
    see = code(f"{prog} help COMMAND", in_table=False)
    return f"Each command's help, with its options and examples: {see}.\n\n" + table(
        ("Command", "What it does"), rows
    )


def _rk_commands(root: pathlib.Path, place: str) -> str:
    """`rk --help`'s description, rkmon.py's docstring, as it prints it."""
    tree = ast.parse((ROOT / "rkmon.py").read_text(encoding="utf-8"))
    text = ast.get_docstring(tree, clean=False)
    if not text:
        raise ValueError("rkmon.py has no docstring")
    return fenced(text)


# The table of the commands that send (rkmon_lib/outward.py) in words: where each sends, and its gate; a gate at the
# send names what goes out (the row's `what`).
SENDS_TO = {
    outward.GITHUB: "GitHub",
    outward.TEAMS: "the team chat",
    outward.WEB: "the web host",
    outward.JIRA: "JIRA",
}
GATES = {
    outward.OWN: "the command's own, before each send",
    outward.GATED: "at the send, before {what} goes out, whoever asked for it",
}


def _picked(name: str) -> Optional[tuple]:
    """For rk NAME, a command that sends with no feature in the table: the option that picks its feature, and each of
    the option's choices with the features it sends for. None for any other command."""
    if name == "notice":
        return "--kind", {
            k: post.NOTICE_FEATURE.get(k, ()) for k in post.NOTICE_CHOICES
        }
    if name == outward.TEAMS_SEND:
        return "--type", dict(post.TEAMS_FEATURES)
    return None


def _titles(names, joiner: str) -> str:
    """The features NAMES by their titles (rkmon_lib/features.py), joined by JOINER."""
    unknown = [n for n in names if n not in features.CAPABILITIES]
    if unknown:
        raise ValueError(
            f"no feature {wrapping.joined(unknown)} in rkmon_lib/features.py"
        )
    return f" {joiner} ".join(features.CAPABILITIES[n].title for n in names)


def _sends_for(r: outward.Command) -> str:
    """The feature cell of R, a row that sends: its features, either one (an issue command's arguments picking one of
    the two); or, for a command with none in the table, each choice of the option that picks them (_picked) with its
    features.
    """
    if r.features:
        pick = (
            " (its arguments pick one)"
            if set(r.features) == set(outward.ISSUES)
            else ""
        )
        return cell(_titles(r.features, "or") + pick)
    picked = _picked(r.name)
    if picked is None:
        raise ValueError(
            f"rk {r.name} sends, and its row has no feature: give it one, or say what picks it "
            "(tools/generated.py, _picked)"
        )
    option, choices = picked
    lines = [f"by its {code(option)}:"]
    for choice, names in choices.items():
        if not names:
            raise ValueError(f"rk {r.name} {option} {choice} sends for no feature")
        lines.append(f"{code(choice)}: {cell(_titles(names, 'and'))}")
    return "<br>".join(lines)


def _outward_commands(root: pathlib.Path, place: str) -> str:
    rows = []
    for r in outward.TABLE:
        if r.dest == outward.NONE:
            continue
        if r.dest not in SENDS_TO or r.gate not in GATES:
            raise ValueError(
                f"rk {r.name}: no words for its destination {r.dest} or its gate {r.gate} "
                "(tools/generated.py, SENDS_TO and GATES)"
            )
        if not r.stops:
            stop = cell("no: it only takes auto-merge off")
        elif r.gate == outward.GATED:
            stop = cell("yes, at its gate")
        elif r.dry_run:
            stop = f"yes, but not with {code('--dry-run')}, which sends nothing"
        else:
            stop = cell("yes")
        rows.append(
            (
                code(f"rk {r.name}"),
                cell(SENDS_TO[r.dest]),
                cell(GATES[r.gate].format(what=r.what)),
                _sends_for(r),
                stop,
            )
        )
    return table(("Command", "Sends to", "Gate", "Feature", "A stop refuses it"), rows)


def _agentic_placeholders(root: pathlib.Path, place: str) -> str:
    rows = [
        (code(name), cell(what, markdown=True))
        for name, (what, _get) in agentic.PLACEHOLDERS.items()
    ]
    return table(("Placeholder", "What it is"), rows)


# The workspace dispatch table's rows of the package's team members, which rk provision puts there
# (agentic.merge_rows): one for each team member file of TEAM_MEMBERS, from its front matter (TEAM_KEYS: the table's
# columns but the file's).
DISPATCH_ROWS = f"agentic/{agentic.REGISTRY_ROWS}"
TEAM_MEMBERS = posixpath.dirname(DISPATCH_ROWS) + "/"
TEAM_KEYS = ("role", "name", "context", "domain")


def front_matter(text: str) -> dict:
    """The `key: value` lines of TEXT's front matter, between the `---` line it starts with and the next one; {} when
    it has none."""
    lines = text.split("\n")
    if lines[0].strip() != "---":
        return {}
    out = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return out
        key, sep, value = line.partition(":")
        if sep:
            out[key.strip()] = value.strip()
    return {}


def _dispatch_rows(root: pathlib.Path, place: str) -> str:
    """A row for each team member file of TEAM_MEMBERS but the rows' own, as the workspace's registry rebuild writes
    it: its role, its name, the file, its context and its domain, each as its front matter has it, sorted by role.
    """
    rows = []
    for p in sorted((root / TEAM_MEMBERS).glob("*.md")):
        if TEAM_MEMBERS + p.name == DISPATCH_ROWS:
            continue
        found = front_matter(p.read_text(encoding="utf-8"))
        lacks = [k for k in TEAM_KEYS if not found.get(k)]
        if lacks:
            raise ValueError(
                f"{TEAM_MEMBERS}{p.name}: its front matter has no {wrapping.joined(lacks)}"
            )
        role, name, context, domain = (cell(found[k], markdown=True) for k in TEAM_KEYS)
        rows.append((role, name, code(p.name), context, domain))
    return table(
        ("Generic Role", "Team Member", "File", "Context", "Domain"),
        sorted(rows, key=lambda r: r[0].lower()),
    )


class Section(NamedTuple):
    """A generated section."""

    name: str
    source: str
    places: tuple
    render: Callable


README, QUICK_DEPLOY, QUICK_USER = "README.md", image.DEPLOY_GUIDE, image.USER_GUIDE
SECTIONS = {
    s.name: s
    for s in (
        Section("doc-map", "tools/generated.py, DOCS", (README,), _doc_map),
        Section(
            "capabilities", _of(features, "CAPABILITIES"), (README,), _capabilities
        ),
        Section(
            "profiles",
            _schema_of(file_types.DEPLOY, "x-profiles"),
            (README,),
            _profiles,
        ),
        Section(
            "build-steps",
            _of(steps, "BUILD_STEPS"),
            (README,),
            _path(steps.BUILD_STEPS),
        ),
        Section(
            "deploy-steps",
            _of(steps, "DEPLOY_STEPS"),
            (README,),
            _path(steps.DEPLOY_STEPS),
        ),
        Section("states", _of(states, "TABLE"), (README,), _states),
        Section("rkmon-commands", _of(commands, "TABLE"), (README,), _rkmon_commands),
        Section("rk-commands", "rkmon.py, rk --help", (README,), _rk_commands),
        Section(
            "outward-commands",
            _of(outward, "TABLE"),
            (README,),
            _outward_commands,
        ),
        Section(
            "deploy-settings",
            _schema_of(file_types.DEPLOY),
            (README,),
            _deploy_settings,
        ),
        Section(
            "model-bindings",
            _schema_of(file_types.DEPLOY, BINDINGS),
            (README,),
            _model_bindings,
        ),
        Section(
            "build-settings",
            f"{_schema_of(file_types.BUILD)}, and "
            f"{_of(build_plan, 'ARGS', 'DOWNLOADS', 'CA_CONTEXT')}",
            (README,),
            _build_settings,
        ),
        Section(
            "required-keys",
            _schema_of(file_types.DEPLOY, "x-profiles and x-required-when"),
            (QUICK_DEPLOY,),
            _required_keys,
        ),
        Section("logins", _of(secrets_registry), (README, QUICK_DEPLOY), _logins),
        Section(
            "env-and-ports",
            _of(deploy_plan, "ENV_KEYS", "OVERRIDE_KEYS"),
            (README,),
            _env_and_ports,
        ),
        Section("exit-codes", _of(outcome, "EXIT_CODES"), (README,), _exit_codes),
        Section(
            "takes-effect",
            _schema_of(file_types.DEPLOY, "x-applies"),
            (QUICK_USER,),
            _takes_effect,
        ),
        Section(
            "agentic-placeholders",
            _of(agentic, "PLACEHOLDERS"),
            ("agentic/README.md",),
            _agentic_placeholders,
        ),
        Section(
            "dispatch-rows",
            f"{TEAM_MEMBERS}, each team member file's front matter",
            (DISPATCH_ROWS,),
            _dispatch_rows,
        ),
    )
}

# --- the generated files ----------------------------------------------------------------------------------------------


class File(NamedTuple):
    """A generated file."""

    path: str
    source: str
    render: Optional[Callable]


GOLDEN = "tests/settings_template_texts.py"


def _golden(root: pathlib.Path) -> str:
    """The settings templates' golden texts, their VERSION kept (tools/template_texts.py)."""
    found = template_texts.VERSION.search((root / GOLDEN).read_text(encoding="utf-8"))
    if not found:
        raise ValueError(f"{GOLDEN} has no VERSION line to keep")
    return template_texts.file_text(found.group(1))


# What a developer's copy makes, which the package never holds: .gitignore's first lines (its folders at the top,
# paths.COPY_OWN_FOLDERS, then what it leaves wherever it runs, paths.COPY_OWN_LEFTOVERS, as they are), and what
# .dockerignore leaves out under the entries it lets in (the leftovers after **/, a folder's trailing / dropped: docker
# reads a pattern as a file's or a folder's).
COPY_OWN_GIT = (
    *(f"/{folder}/" for folder in paths.COPY_OWN_FOLDERS),
    *paths.COPY_OWN_LEFTOVERS,
)
COPY_OWN_DOCKER = tuple(f"**/{p.rstrip('/')}" for p in paths.COPY_OWN_LEFTOVERS)
# .gitignore's lines about .claude/, after COPY_OWN_GIT: each skill the package ships (claude_skills) let in again, by
# its folder, which the repository the package lives in may leave out with every .claude/ folder it has; the rest of
# .claude/ kept out: a skill of your own, and Claude Code's local files, such as settings.local.json, which may have
# private values in them. The patterns that may match a secret's or a private value's file come after them, and so
# win. The image takes none of .claude/ (image.PACKAGE_ONLY).
LET_IN_GIT = (
    "!/.claude/",
    "/.claude/*",
    f"!/{CLAUDE_SKILLS}",
    f"/{CLAUDE_SKILLS}*",
    *(f"!/{CLAUDE_SKILLS}{name}/" for name in claude_skills()),
)
# rocKE's header (make_package.HEADER_LINES): as comments at the top of the ignore files, and in an HTML comment at
# the top of a Markdown file the refresh writes whole (with_header).
IGNORE_HEAD = "".join(f"# {line}\n" for line in make_package.HEADER_LINES) + "#\n"


def with_header(text: str) -> str:
    """TEXT, a Markdown file the refresh writes whole, with rocKE's header at its top."""
    return (
        "<!--\n"
        + "".join(f"{line}\n" for line in make_package.HEADER_LINES)
        + "-->\n\n"
        + text
    )


GITIGNORE_HEAD = f"""\
# {REFRESH} writes this file from rkmon_lib/paths.py
# (COPY_OWN_FOLDERS, COPY_OWN_LEFTOVERS), tools/generated.py (LET_IN_GIT, a line for each skill DOCS lists) and
# rkmon_lib/secrets_registry.py (ignore_patterns): change them, not this file.
#
# The package is read-only at run time: everything the monitor writes lives in <WORKING_DIR>, outside the repository.
# The first lines leave out what a developer's copy makes. The lines about .claude/ let each of the package's skills in
# again, since the repository the package lives in may leave out every .claude/ folder, and keep out the rest of
# .claude/: a skill of your own, and Claude Code's local files, such as settings.local.json, which may have private
# values in them. The others leave out, wherever they are, the files and folders that may have a secret or a private
# value in them. None of them leaves out a file of the package.
"""
DOCKERIGNORE_HEAD = f"""\
# {REFRESH} writes this file from rkmon_lib/image.py
# (BUILD_CONTEXT), rkmon_lib/paths.py (COPY_OWN_LEFTOVERS) and rkmon_lib/secrets_registry.py (ignore_patterns):
# change them, not this file.
#
# The build context is an allow-list: everything is left out, then only the package's own entries are let in. Under
# them, what a developer's copy makes and, wherever they are, the files and folders that may hold a secret or a
# private value are left out again: a file in the build context is in an image layer for good, even when a later step
# deletes it. Your network's root certificates reach the build as a build context of their own (docker/Dockerfile).
"""


def _gitignore(root: pathlib.Path) -> str:
    """.gitignore: what a developer's copy makes (COPY_OWN_GIT), the package's skills let in again (LET_IN_GIT), then
    each of secrets_registry.ignore_patterns as it is."""
    lines = (*COPY_OWN_GIT, *LET_IN_GIT, *secrets_registry.ignore_patterns())
    return IGNORE_HEAD + GITIGNORE_HEAD + "".join(f"{p}\n" for p in lines)


def _dockerignore(root: pathlib.Path) -> str:
    """.dockerignore, an allow-list: everything left out (*), then each entry of image.BUILD_CONTEXT let in; under
    them, what a developer's copy makes (COPY_OWN_DOCKER), then each of secrets_registry.ignore_patterns wherever it
    is (after **/, a folder's trailing / dropped: docker reads each pattern as a file's or a folder's).
    """
    lines = [
        "*",
        *(f"!{entry}" for entry in image.BUILD_CONTEXT),
        *COPY_OWN_DOCKER,
        *(f"**/{p.rstrip('/')}" for p in secrets_registry.ignore_patterns()),
    ]
    return IGNORE_HEAD + DOCKERIGNORE_HEAD + "".join(f"{x}\n" for x in lines)


INTERNAL_SETTINGS = "docs/internal_settings.md"
SHOWN_MAX = 60  # a default whose JSON is longer is cut: its home has it whole


def _shown_default(value) -> str:
    """A key's default in a table cell: its JSON, cut at SHOWN_MAX characters; nothing for a key with none."""
    if value is registry.MISSING:
        return ""
    text = json.dumps(value, ensure_ascii=False)
    return code(text if len(text) <= SHOWN_MAX else text[: SHOWN_MAX - 3] + "...")


def _internal_rows(
    owner: str, descriptions: dict, file_keys: bool = False, defaults: bool = False
) -> list:
    """The rows of OWNER's keys, in the registry's order: each key (with FILE_KEYS, its keys in its file; with DEFAULTS,
    its default), then each entry of its value the registry describes (registry.ENTRY), with DEFAULTS the entry's value
    in the key's default. A package or state key's description closes with its default, so its table has none beside.
    """
    rows = []
    for k in registry.keys():
        if k.owner != owner:
            continue
        files = ["<br>".join(code(f) for f in k.file_keys)] if file_keys else []
        shown = [_shown_default(k.default)] if defaults else []
        rows.append((code(k.name), *files, *shown, cell(k.description)))
        for name, text in descriptions.items():
            if name.startswith(k.name + ".") and registry.kind(name) == registry.ENTRY:
                inside = registry.get(k.default, name[len(k.name) + 1 :])
                shown = [_shown_default(inside)] if defaults else []
                rows.append((code(name), *[""] * len(files), *shown, cell(text)))
    return rows


def _internal_sections(descriptions: dict, names: set) -> list:
    """A row for each section the registry describes that has one of the keys NAMES in it, in the order of the
    sections' names."""
    return [
        (code(name), cell(text))
        for name, text in sorted(descriptions.items())
        if registry.kind(name) == registry.SECTION
        and any(n.startswith(name + ".") for n in names)
    ]


def _internal_settings() -> str:
    """docs/internal_settings.md before its header: the registry's package, state and derived keys, each owner's in a
    table of its own, in the registry's order, then the sections they are in."""
    descriptions = registry.descriptions()
    names = {k.name for k in registry.keys() if k.owner != registry.DEPLOY}

    def path(text: str) -> str:
        return code(text, in_table=False)

    return "\n".join(
        [
            "# Internal settings",
            "",
            "The configuration keys that are in no settings file: the package's defaults, what the monitor records in "
            "its internal state, and the values derived from others. Each key's description and default come from the "
            f"registry, {path(_of(registry))}, their one home: a package or state key's description closes with its "
            "default. This file is generated from it: change the registry, "
            f"then run {path(REFRESH)}. The settings files' keys are described in their schemas, "
            f"{path(_schema_of(file_types.DEPLOY))} and {path(_schema_of(file_types.BUILD))}.",
            "",
            "## Package defaults",
            "",
            "Values every deployment shares and no person sets, each in "
            f"{path(registry.PACKAGE_FILE.name)}.",
            "",
            table(
                ("Key", "What it is"),
                _internal_rows(registry.PACKAGE, descriptions),
            ),
            "",
            "## Internal state",
            "",
            "What the monitor found or recorded, in its internal state file, whose schema is "
            f"{path(_schema_of(file_types.INTERNAL_STATE))}. Beside each key, its key in that file.",
            "",
            table(
                ("Key", "In the state file", "What it is"),
                _internal_rows(registry.STATE, descriptions, file_keys=True),
            ),
            "",
            "## Derived",
            "",
            "Values computed at each load from the keys before them and the container's layout, one function each in "
            f"{path(_of(derived))}. A key's default is what a configuration file has when it leaves the key to its "
            "rule; the paths that are in no configuration file have none.",
            "",
            table(
                ("Key", "Default", "What it is"),
                _internal_rows(registry.DERIVED, descriptions, defaults=True),
            ),
            "",
            "## Sections",
            "",
            "The parts of the configuration the keys above are in.",
            "",
            table(("Section", "What it is"), _internal_sections(descriptions, names)),
        ]
    )


FILES = (
    File(
        ".gitignore",
        "rkmon_lib/paths.py (COPY_OWN_FOLDERS, COPY_OWN_LEFTOVERS) and rkmon_lib/secrets_registry.py "
        "(ignore_patterns)",
        _gitignore,
    ),
    File(
        ".dockerignore",
        "rkmon_lib/image.py (BUILD_CONTEXT), rkmon_lib/paths.py (COPY_OWN_LEFTOVERS) and "
        "rkmon_lib/secrets_registry.py (ignore_patterns)",
        _dockerignore,
    ),
    File(
        GOLDEN,
        "the settings templates, as rkmon_lib/settings_file.py writes them",
        _golden,
    ),
    File(
        INTERNAL_SETTINGS,
        _of(registry),
        lambda root: make_package.normalized(with_header(_internal_settings())),
    ),
    File("VERSION", "the day of the package's release", None),
)

# --- the markers ------------------------------------------------------------------------------------------------------

BEGIN = "<!-- BEGIN GENERATED:"
END = "<!-- END GENERATED:"
NAME = r"[a-z0-9]+(?:-[a-z0-9]+)*"
BEGIN_LINE = re.compile(rf"<!-- BEGIN GENERATED: ({NAME})(?=\s|$)")
END_LINE = re.compile(rf"<!-- END GENERATED: ({NAME}) -->")
FENCE = re.compile(r" {0,3}(`{3,}|~{3,})")


class Block(NamedTuple):
    """A section in a doc: its name, and the lines of its BEGIN and its END (from 0)."""

    name: str
    first: int
    last: int


def blocks(text: str) -> tuple:
    """The sections of a doc's TEXT, and what is wrong with its markers: see the module's docstring."""
    lines = text.split("\n")
    found, problems = [], []
    fence = None  # the fenced code that is open, outside any section: its character and length
    i = 0
    while i < len(lines):
        line = lines[i]
        opens = FENCE.match(line)
        if fence:
            run = opens.group(1) if opens else ""
            if (
                run[:1] == fence[0]
                and len(run) >= fence[1]
                and not line[opens.end() :].strip()
            ):
                fence = None
            i += 1
            continue
        if opens:
            fence = (opens.group(1)[0], len(opens.group(1)))
            i += 1
            continue
        if line.startswith(END):
            problems.append(f"{i + 1}: an END with no BEGIN: {line.strip()}")
            i += 1
            continue
        if not line.startswith(BEGIN):
            i += 1
            continue
        begin = BEGIN_LINE.match(line)
        if not begin:
            problems.append(
                f"{i + 1}: not a marker: a BEGIN is {BEGIN} NAME, NAME lowercase words joined by -"
            )
            i += 1
            continue
        name = begin.group(1)
        close = next((j for j in range(i, len(lines)) if "-->" in lines[j]), None)
        inside = [
            j
            for j in range(i + 1, len(lines) if close is None else close + 1)
            if lines[j].startswith((BEGIN, END))
        ]
        if close is None or inside:
            problems.append(
                f"{i + 1}: the BEGIN of {name} has no --> to end its comment"
            )
            i = inside[0] if inside else len(lines)
            continue
        j, last = close + 1, None
        while j < len(lines) and not lines[j].startswith((BEGIN, END)):
            j += 1
        if j == len(lines):
            problems.append(f"{i + 1}: the section {name} has no END")
        elif lines[j].startswith(BEGIN):
            problems.append(
                f"{j + 1}: a BEGIN inside the section {name}, which has no END before it"
            )
        else:
            end = END_LINE.fullmatch(lines[j].rstrip())
            if not end:
                problems.append(f"{j + 1}: not a marker: an END is {END} NAME -->")
            elif end.group(1) != name:
                problems.append(
                    f"{j + 1}: the END of {end.group(1)} inside the section {name}"
                )
            else:
                last = j
            j += 1
        if name not in SECTIONS:
            problems.append(
                f"{i + 1}: no section {name} in tools/generated.py: its sections are "
                f"{wrapping.joined(list(SECTIONS))}"
            )
        elif last is not None:
            found.append(Block(name, i, last))
        i = j
    return found, problems


def section_text(section: Section, root: pathlib.Path, place: str) -> str:
    """The section whole, from its BEGIN marker to its END."""
    with _as_rkmon():
        text = section.render(pathlib.Path(root), place).strip("\n")
    return "\n".join(
        [
            f"{BEGIN} {section.name}",
            f"     source: {section.source}",
            f"     refresh: {REFRESH} -->",
            "",
            *(line.rstrip() for line in text.split("\n")),
            "",
            f"{END} {section.name} -->",
        ]
    )


# --- drift and refresh ------------------------------------------------------------------------------------------------


def _docs(root: pathlib.Path, docs) -> list:
    if docs is None:
        return [f for f in make_package.tracked(root) if f.suffix == ".md"]
    return [root / d for d in docs]


def _version_in_form(path: pathlib.Path) -> bool:
    """Whether VERSION at PATH is the header, then a day as YYYY.MM.DD (make_package.version_text)."""
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    day = re.fullmatch(
        r"(\d{4})\.(\d\d)\.(\d\d)", text.rstrip("\n").rsplit("\n", 1)[-1]
    )
    try:
        return bool(day) and text == make_package.version_text(
            dt.date(*map(int, day.groups()))
        )
    except ValueError:  # no such day
        return False


def _made(root: pathlib.Path, f: File) -> str:
    try:
        return f.render(root)
    except (ValueError, OSError) as e:
        raise ValueError(
            f"{f.path}: cannot be made from its source ({f.source}): {e}"
        ) from None


def _where(rel: str, b: Block) -> str:
    """A section as a problem names it: the doc's path and the line of its BEGIN (from 1), and its name."""
    return f"{rel}:{b.first + 1}: the section {b.name}"


def _section(root: pathlib.Path, rel: str, b: Block) -> tuple:
    """The section of B in the doc at REL as its source makes it, and what is wrong with it: a doc that is not its
    place, or a source that cannot make it (the text then None)."""
    section = SECTIONS[b.name]
    wrong = []
    if rel not in section.places:
        wrong.append(
            f"{_where(rel, b)} goes in {wrapping.joined(list(section.places))}, not here"
        )
    try:
        return section_text(section, root, rel), wrong
    except ValueError as e:
        wrong.append(
            f"{_where(rel, b)} cannot be made from its source ({section.source}): {e}"
        )
        return None, wrong


def _missing(rel: str, found: list) -> list:
    """Each section whose places name the doc at REL that FOUND, its blocks, lacks: named with the markers to add."""
    have = {b.name for b in found}
    return [
        f"{rel}: the section {name} is missing: add {BEGIN} {name} --> and {END} {name} --> where it goes, "
        f"then run {REFRESH}"
        for name, section in SECTIONS.items()
        if rel in section.places and name not in have
    ]


def drift(root, files=FILES, docs=None) -> list:
    """How the package at ROOT is out of step with its sources: see the module's docstring."""
    root = pathlib.Path(root)
    out = []
    for doc in _docs(root, docs):
        rel = doc.relative_to(root).as_posix()
        text = doc.read_text(encoding="utf-8")
        found, problems = blocks(text)
        out += [f"{rel}:{p}" for p in problems]
        lines = text.split("\n")
        for b in found:
            want, wrong = _section(root, rel, b)
            out += wrong
            if want is not None and "\n".join(lines[b.first : b.last + 1]) != want:
                out.append(
                    f"{_where(rel, b)} is out of step with its source ({SECTIONS[b.name].source}): run {REFRESH}"
                )
        if not problems:
            out += _missing(rel, found)
    for f in files:
        path = root / f.path
        if f.render is None:
            if not _version_in_form(path):
                out.append(
                    f"{f.path}: not the header and a YYYY.MM.DD line: a release writes it: {RELEASE}"
                )
        elif not path.is_file():
            out.append(f"{f.path}: missing: run {REFRESH}")
        else:
            try:
                want = _made(root, f)
            except ValueError as e:
                out.append(str(e))
                continue
            if path.read_text(encoding="utf-8") != want:
                out.append(
                    f"{f.path}: out of step with its source ({f.source}): run {REFRESH}"
                )
    return out


def refresh(root, files=FILES, docs=None) -> list:
    """Write each doc and file whose text differs from what its sources make: see the module's docstring."""
    root = pathlib.Path(root)
    problems, writes = [], {}
    for doc in _docs(root, docs):
        rel = doc.relative_to(root).as_posix()
        text = doc.read_text(encoding="utf-8")
        found, bad = blocks(text)
        problems += [f"{rel}:{p}" for p in bad]
        if bad:
            continue
        made = [(b, *_section(root, rel, b)) for b in found]
        for _b, _want, wrong in made:
            problems += wrong
        lines = text.split("\n")
        for b, want, wrong in reversed(made):
            if not wrong:
                lines[b.first : b.last + 1] = want.split("\n")
        if "\n".join(lines) != text:
            writes[doc] = "\n".join(lines)
    for f in files:
        path = root / f.path
        # VERSION, whose render is None: a release writes it (RELEASE)
        if f.render is None:
            continue
        try:
            want = _made(root, f)
        except ValueError as e:
            problems.append(str(e))
            continue
        if not path.is_file() or path.read_text(encoding="utf-8") != want:
            writes[path] = want
    if problems:
        raise ValueError(
            "nothing written: the refresh stops on these\n  " + "\n  ".join(problems)
        )
    for path, text in writes.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return sorted(p.relative_to(root).as_posix() for p in writes)
