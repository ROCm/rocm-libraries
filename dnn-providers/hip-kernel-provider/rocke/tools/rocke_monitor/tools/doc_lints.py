# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The docs' lints: each reads the files it is given, and names each problem with its file,
its line and what to write instead.

Usage: python3 -B tools/doc_lints.py [--lint NAME]... FILE...
Each FILE is a file of the package. It prints each problem as `PATH:LINE: LINT: what`, and exits 0 when there is none,
1 when there is one, and 2 on a usage error or a file it cannot read. --lint, as often as needed, runs the lints it
names (LINTS); every lint runs by default.

What a lint reads of a file (`texts`, `read`), each a `Text`:
- markdown (.md): each heading, each paragraph (its lines between blank ones) and each fenced block, with the headings
  it is under, the doc's kind (tools/generated.py, `kind_of`: USER, MAINTAINER or SKILL) and whether it is in a
  generated section;
- Python (.py, or a file whose #! line runs python): each string it may print: a string constant that is no docstring
  (but the module's own, when the module prints it as its help) and no bare name (BARE); an f-string is one text, with
  VALUE in place of each of its values (`python_texts`: the texts as they are written, which tests/test_program.py
  reads for a program name written literally); and the lints read each `{program}` in a text as the command's name
  (program.NAME), as the messages name it (rkmon_lib/program.py). That is how the lints read the printed messages and
  the help: the command table, the state table's sentences, next lines and fixes, the steps and the checks' rows are
  string constants where they are written, read as they are written, never run;
- JSON: each string that is no key and no bare name: the schemas' descriptions, examples and comments;
- other text (TEXT_SUFFIXES, TEXT_NAMES): each paragraph of its comments, without their #, and each run of its other
  lines, as code.
A file that is not markdown is of the kind MESSAGE. A text is searched with each run of white space as one space, so
that a phrase may wrap; a problem names the line its match starts on.

The lints, by name (LINTS; `run` runs them over the files given):
- homes (HOMES): a signature of a fact outside its home is a copy, and its problem names the home to link. A generated
  section is no copy. A home among the files given has its section and each of its signatures.
- keys (KEYS): in a user doc and in the compose header (HEADERS), each dotted key whose first word starts a key of a
  settings file or an internal key is a key of a settings file (`*` stands for any word, and so does a word in angle
  brackets in a doc); an internal key is a problem there too: a name the registry knows (rkmon_lib/registry.py), or a
  name inside one of its keys whose value is a mapping (`load_keys`). In a message, each such key is a key of a
  settings file or an internal key, which an earlier name of a key is not. A name a value of the code starts or ends
  (`{…}.timer`, `monitor_{…}`) is no key the lint can read, and no key is read where an allowance of KEYS keeps
  it: layout 1's translators and its schema name layout 1's keys on purpose.
- words (WORDS): no word of a list in a user doc, a skill file or a message, but where an allowance of the list keeps
  it. Text in backticks and fenced blocks are read for a list that says code, and for no other.
- old-commands (`load_old_commands`): no old command in any text, but where an allowance keeps it: each command and
  option of the program's that another command replaced (commands.GONE, their one home, in rkmon_lib/commands.py:
  `gone_commands`), and the old commands OLD_COMMANDS lists, which GONE does not hold.
- quoted-output: each outcome, check row, fix, next line and note a user doc quotes, in backticks or in a fenced block,
  is printed by the package's code. The texts read are an outcome's after its word (`OUTCOME: TEXT`), a check row's
  after its status and name (`STATUS  NAME: FOUND`, or `STATUS: TEXT`: ROW), and each run of words of a fix, a next
  line or a note (LABELS) outside its commands and its edits. Each is printed whole, or each part of it between `; `
  that has three words or more is: one of the code's texts, or a part of one, each value of the code standing for any
  text, as do each word in angle brackets and each ellipsis of the quote (one at either end for the rest of a text:
  `_cut`); a text of the code found whole in a part reads as a value (`_texts_as_values`), and a text in parentheses
  is read on its own (ASIDE). The code's texts (`load_templates`) are those of its Python files (`_python`) and those
  it joins with + (`joined_texts`), each str.format field a value, and each settings key's `x-source`, which a fix
  prints as the key's value (settings_file.source_of), whole and each part between `; `. A quote and a
  text are read with the program's name alone where a path to it is written (`./rkmon`, `docker/rkmon`: BEFORE_NAME),
  and without the outcome word or the label they start with. In a fix, a next line or a note, each command of the
  program starts with one of the command table's (`_names_a_command`); each edit (EDIT: `set KEY in FILE to VALUE` or
  `delete KEY from FILE`, its FILE a word, a word in angle brackets or a file in words, FILES_IN_WORDS) has a value
  written as JSON or one the code prints, and an edit's own form prints no run of words; a next line's `step N, NAME`
  is a step of either path (rkmon_lib/steps.py), its command, if any, that step's. An outcome quoted with words in
  capitals before its outcome word is one the code printed once; it passes only when the code prints it whole.
- links: each markdown link into the package reaches its file, and its anchor a heading there; each doc and heading a
  text names (the doc's file name, a comma, then the heading in double quotes) is a doc of the package and one of its
  headings.
- placeholders (PLACEHOLDERS): each placeholder (a name in angle brackets that is no HTML tag, and no regular
  expression's group name, `(?P<name>...)`) in a user doc, a skill file or a message is in the README's glossary
  (GLOSSARY), in the agentic files' table (rkmon_lib/agentic.py, PLACEHOLDERS), or a build secret's id, which a masked
  line of a build tool shows as it is (`masked_names`); but where an allowance of PLACEHOLDERS keeps it: an example
  word of the agentic files, a slot of a form they show, which whoever writes the form fills.
A lint reads its data from the package at DATA (ROOT unless given): HOMES, KEYS, WORDS, OLD_COMMANDS, PLACEHOLDERS,
the settings files' schemas and the code; the internal keys are the running package's registry; the glossary, the
build file's secrets and the files a link reaches are the linted package's.

`every_file(root)`: the word lists and the old commands that say every_file (GONE's all do), over every file of the
package, of every kind, their code and their generated sections included: the earlier names of renamed things, which no
file keeps but where an allowance says why.
"""

from __future__ import annotations

import argparse
import ast
import functools
import json
import pathlib
import posixpath
import re
import sys
import urllib.parse
from typing import NamedTuple, Optional

ROOT = pathlib.Path(__file__).resolve().parent.parent
for _folder in (ROOT, ROOT / "tools"):
    if str(_folder) not in sys.path:
        sys.path.insert(0, str(_folder))
sys.dont_write_bytecode = True

import generated  # noqa: E402
import make_package  # noqa: E402
from rkmon_lib import (  # noqa: E402
    agentic,
    answers,
    checks,
    commands,
    deploy_tree,
    file_types,
    image,
    outcome,
    paths,
    program,
    progress,
    registry,
    settings_file,
    steps,
    wrapping,
)

USER, MAINTAINER, SKILL = generated.USER, generated.MAINTAINER, generated.SKILL
MESSAGE = "message"
KINDS = (USER, MAINTAINER, SKILL, MESSAGE)
LINTS = (
    "homes",
    "keys",
    "words",
    "old-commands",
    "quoted-output",
    "links",
    "placeholders",
)
HOMES = "tests/doc_fact_homes.yaml"
KEYS = "tests/doc_keys.yaml"
WORDS = "tests/doc_words.yaml"
OLD_COMMANDS = "tests/doc_old_commands.yaml"
PLACEHOLDERS = "tests/doc_placeholders.yaml"
GLOSSARY = ("README.md", "Glossary and placeholders")
# The files whose first paragraph, their header, a person reads as a user doc.
HEADERS = ("docker/compose.yaml",)
# The settings files' schemas, by their paths in a package (rkmon_lib/file_types.py: DEPLOY's and BUILD's).
SCHEMAS = tuple(
    pathlib.Path(t.schema).relative_to(paths.PACKAGE).as_posix()
    for t in (file_types.DEPLOY, file_types.BUILD)
)


class Problem(NamedTuple):
    """A problem a lint found: the file, the line (from 1), the lint's name and what is wrong."""

    path: str
    line: int
    lint: str
    what: str

    def __str__(self) -> str:
        return f"{self.path}:{self.line}: {self.lint}: {self.what}"


def _fields(entry, known: set, where: str, required: tuple = ()) -> dict:
    """ENTRY, a mapping with each key of REQUIRED and no key KNOWN lacks; a ValueError naming WHERE otherwise."""
    if not isinstance(entry, dict):
        raise ValueError(f"{where}: a mapping of keys to values is expected")
    unknown = sorted(set(entry) - known)
    if unknown:
        raise ValueError(f"{where}: no key {', '.join(unknown)} is known here")
    missing = [k for k in required if entry.get(k) in (None, "", [])]
    if missing:
        raise ValueError(f"{where}: {', '.join(missing)} is required")
    return entry


# --- what a lint reads -------------------------------------------------------------------------------------------------

TEXT_SUFFIXES = {".sh", ".yaml", ".yml", ".txt", ".service"}
TEXT_NAMES = {"Dockerfile", "rk"}
# The package's files that are not its own text: its tests (make_package.tracked leaves out its state at run time, a
# build's output and Python's caches).
SKIP = {"tests"}
# A string that is a name and nothing else: a key, a setting, a part of the state.
BARE = re.compile(r"[a-z_][a-z0-9_.-]*")
# An f-string's value, in its text.
VALUE = "{…}"
HEADING = re.compile(r" {0,3}(#{1,6})(?:[ \t]+(.*?))?(?:[ \t]+#+)?[ \t]*")
# Text in backticks, quoted as code.
CODE = re.compile(r"`[^`]*`")


class Text(NamedTuple):
    """A text of a file, as the lints read it: the file's path in the package; the line it starts on (from 1); the
    text as it is written; the kind of what it is in; the headings it is under, outermost first; and whether it is a
    fenced block, in a generated section, or a heading."""

    path: str
    line: int
    text: str
    kind: str
    sections: tuple = ()
    code: bool = False
    generated: bool = False
    heading: bool = False


def package_files(root) -> list:
    """The files of the package at ROOT (make_package.tracked), by their paths from it, but those in SKIP's folders;
    sorted."""
    root = pathlib.Path(root)
    rels = [f.relative_to(root) for f in make_package.tracked(root)]
    return sorted(rel.as_posix() for rel in rels if not set(rel.parts) & SKIP)


def _reader(rel: str, first_line: str = ""):
    """The reader of the file REL, whose first line is FIRST_LINE; None for a file no lint reads."""
    name = posixpath.basename(rel)
    suffix = posixpath.splitext(name)[1]
    if suffix == ".md":
        return _markdown
    if suffix in TEXT_SUFFIXES or name in TEXT_NAMES:
        return _paragraphs
    if suffix == ".json":
        return _json
    if suffix == ".py" or (not suffix and re.match(r"#!.*python", first_line)):
        return _python
    return None


def read(rel: str, source: str) -> list:
    """The texts of SOURCE, the text of the package's file REL (see the module's docstring); [] for a file no lint
    reads."""
    reader = _reader(rel, source.split("\n", 1)[0])
    return reader(rel, source) if reader else []


def texts(root, rel: str) -> list:
    """The texts of the file REL of the package at ROOT (`read`); [] for a file no lint reads, or one that is not
    text."""
    if posixpath.splitext(posixpath.basename(rel))[1] and _reader(rel) is None:
        return []
    try:
        source = (pathlib.Path(root) / rel).read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return []
    return read(rel, source)


def _fences(lines: list) -> dict:
    """Each fenced block of LINES (tools/generated.py, FENCE): its opening line -> its closing line, or len(LINES)
    when it has none; from 0."""
    out, i = {}, 0
    while i < len(lines):
        opens = generated.FENCE.match(lines[i])
        if not opens:
            i += 1
            continue
        char, size = opens.group(1)[0], len(opens.group(1))
        j = i + 1
        while j < len(lines):
            closes = generated.FENCE.match(lines[j])
            if (
                closes
                and closes.group(1)[0] == char
                and len(closes.group(1)) >= size
                and not lines[j][closes.end() :].strip()
            ):
                break
            j += 1
        out[i] = j
        i = j + 1
    return out


def _markdown(rel: str, source: str) -> list:
    kind = generated.kind_of(rel)
    # what is wrong with a marker is the drift test's
    found, _problems = generated.blocks(source)
    made = {i for b in found for i in range(b.first, b.last + 1)}
    lines = source.split("\n")
    fences = _fences(lines)
    out, para, stack = [], [], []

    def here() -> tuple:
        return tuple(title for _level, title in stack)

    def flush() -> None:
        if para:
            text = "\n".join(lines[i] for i in para)
            out.append(
                Text(rel, para[0] + 1, text, kind, here(), False, para[0] in made)
            )
            para.clear()

    i = 0
    while i < len(lines):
        if i in fences:
            flush()
            last = fences[i]
            text = "\n".join(lines[i + 1 : last])
            out.append(Text(rel, i + 2, text, kind, here(), True, i in made))
            i = last + 1
            continue
        heading = HEADING.fullmatch(lines[i])
        if (
            heading
            or not lines[i].strip()
            or (para and (i in made) != (para[0] in made))
        ):
            flush()
        if heading:
            level, title = len(heading.group(1)), (heading.group(2) or "").strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            out.append(Text(rel, i + 1, title, kind, here(), False, i in made, True))
        elif lines[i].strip():
            para.append(i)
        i += 1
    flush()
    return out


_WALKED = [None, []]  # the tree `_nodes` walked last, and its nodes


def _nodes(tree) -> list:
    """The nodes of TREE (ast.walk), walked once for the tree it was last asked about."""
    if _WALKED[0] is not tree:
        _WALKED[:] = [tree, list(ast.walk(tree))]
    return _WALKED[1]


def python_texts(rel: str, source: str, tree=None) -> list:
    """Each string the Python file REL, of SOURCE, may print, as a Text of the kind MESSAGE, as it is written:
    `{program}` stays as it is (the module's docstring says which strings, and how an f-string reads); sorted by line.
    TREE: SOURCE parsed (ast.parse), when the caller has it.
    """
    tree = ast.parse(source) if tree is None else tree
    nodes = _nodes(tree)
    quiet = {id(n.value) for n in nodes if isinstance(n, ast.Expr)}
    prints_its_help = any(isinstance(n, ast.Name) and n.id == "__doc__" for n in nodes)
    if prints_its_help and tree.body and isinstance(tree.body[0], ast.Expr):
        quiet.discard(id(tree.body[0].value))
    parts = set()  # an f-string's parts, which its text reads
    for n in nodes:
        if isinstance(n, ast.JoinedStr):
            parts.update(id(v) for v in n.values)
        elif isinstance(n, ast.FormattedValue) and n.format_spec is not None:
            parts.add(id(n.format_spec))
    out = []
    for n in nodes:
        if id(n) in quiet or id(n) in parts:
            continue
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            text = n.value
        elif isinstance(n, ast.JoinedStr):
            text = "".join(
                v.value if isinstance(v, ast.Constant) else VALUE for v in n.values
            )
        else:
            continue
        if text.strip() and not BARE.fullmatch(text):
            out.append(Text(rel, n.lineno, text, MESSAGE))
    return sorted(out, key=lambda t: t.line)


def _python(rel: str, source: str, tree=None) -> list:
    """python_texts, each `{program}` read as the command's name (program.NAME), as the messages name it."""
    return [
        t._replace(text=t.text.replace("{program}", program.NAME))
        for t in python_texts(rel, source, tree)
    ]


_JSON_STRING = re.compile(r'"(?:[^"\\]|\\.)*"')
_COLON = re.compile(r"\s*:")


def _json(rel: str, source: str) -> list:
    out, line, at = [], 1, 0
    for m in _JSON_STRING.finditer(source):
        if _COLON.match(source, m.end()):
            continue  # a key
        text = json.loads(m.group())
        if text.strip() and not BARE.fullmatch(text):
            line += source.count("\n", at, m.start())
            at = m.start()
            out.append(Text(rel, line, text, MESSAGE))
    return out


# A comment line's mark: a # that starts the line, but a #! line's, which runs the file.
COMMENT_LINE = re.compile(r"[ \t]*#(?!!) ?")


def _paragraphs(rel: str, source: str) -> list:
    """Each paragraph of a text file: a run of its comment lines, each without its #, between blank ones (a comment
    line with nothing after its # is a blank one); or a run of its other lines (a #! line among them), which is
    code."""
    out, para, first, code = [], [], 0, False

    def flush() -> None:
        if para:
            out.append(Text(rel, first + 1, "\n".join(para), MESSAGE, (), code))
            para.clear()

    for i, line in enumerate(source.split("\n")):
        mark = COMMENT_LINE.match(line)
        body = line[mark.end() :] if mark else line
        if not body.strip() or (para and code != (not mark)):
            flush()
        if body.strip():
            if not para:
                first, code = i, not mark
            para.append(body)
    flush()
    return out


_RUNS = re.compile(r"\s+|\S+")


def _flat(raw: str) -> tuple:
    """RAW with each run of white space as one space, and the offset in RAW of each of its characters."""
    text, at = [], []
    for m in _RUNS.finditer(raw):
        if m.group()[0].isspace():
            text.append(" ")
            at.append(m.start())
        else:
            text.append(m.group())
            at.extend(range(m.start(), m.end()))
    return "".join(text), at


def _matches(t: Text, pattern, raw: Optional[str] = None):
    """Each match of PATTERN in T's text, read with each run of white space as one space: the match, the line it starts
    on, and the offset in the text it starts at. RAW: the text as a lint marked it, with its lines and its offsets.
    """
    raw = t.text if raw is None else raw
    flat, at = _flat(raw)
    for m in pattern.finditer(flat):
        start = at[m.start()] if m.start() < len(at) else len(raw)
        yield m, t.line + raw.count("\n", 0, start), start


def _mark(m) -> str:
    """A match as marks that are neither a word nor a space, so that the words on either side of it stay apart."""
    return re.sub(r"[^\n]", "\0", m.group())


def _prose(text: str) -> str:
    """TEXT with its text in backticks marked out (`_mark`)."""
    return CODE.sub(_mark, text)


# --- allowances --------------------------------------------------------------------------------------------------------


class Allowance(NamedTuple):
    """A place that keeps a list's words or an old command: a file whose path FILES matches whole, of a kind KIND
    names, under the heading SECTION (each, when given); there, KEEP's matches stay (its ^ and $ at each line's ends),
    or the whole text when there is no KEEP; WHY says why."""

    files: Optional[re.Pattern]
    kind: tuple
    section: str
    keep: Optional[re.Pattern]
    why: str

    def covers(self, t: Text) -> bool:
        return (
            (self.files is None or bool(self.files.fullmatch(t.path)))
            and (not self.kind or t.kind in self.kind)
            and (not self.section or self.section in t.sections)
        )


def _allowances(
    entries, where: str, flags: int = 0, required: tuple = ("why",)
) -> tuple:
    """The allowances ENTRIES gives, in WHERE, each with the keys REQUIRED; a keep's case as FLAGS says."""
    out = []
    for a in entries or ():
        _fields(a, {"files", "kind", "section", "keep", "why"}, where, required)
        kind = tuple(a.get("kind") or ())
        if set(kind) - set(KINDS):
            raise ValueError(
                f"{where}: an allowance names no kind but {', '.join(KINDS)}"
            )
        out.append(
            Allowance(
                re.compile(a["files"]) if a.get("files") else None,
                kind,
                a.get("section") or "",
                re.compile(a["keep"], re.M | flags) if a.get("keep") else None,
                a["why"],
            )
        )
    return tuple(out)


def _unkept(t: Text, allowed, raw: str) -> Optional[str]:
    """RAW with what the allowances that cover T keep marked out (`_mark`); None when one of them keeps the whole
    text."""
    for a in allowed:
        if a.covers(t):
            if a.keep is None:
                return None
            raw = a.keep.sub(_mark, raw)
    return raw


# --- one home ----------------------------------------------------------------------------------------------------------


class Fact(NamedTuple):
    """A fact of HOMES: its name; its home, a doc and a heading of it ("" for the whole doc); its signatures; and, when
    the home is a command's help, that command's words after the program's name (`help`), as a person types them.
    """

    name: str
    home: str
    section: str
    signatures: tuple
    help: str = ""


def load_homes(root=ROOT) -> list:
    """The facts of HOMES in the package at ROOT. A `help` must name a command of the command table."""
    out = []
    for name, entry in answers.load(pathlib.Path(root) / HOMES).items():
        where = f"{HOMES}, {name}"
        _fields(
            entry,
            {"home", "section", "signatures", "help"},
            where,
            ("home", "signatures"),
        )
        signatures = tuple(re.compile(s) for s in entry["signatures"])
        help_ = entry.get("help") or ""
        if help_ and not commands.find(help_.split(), commands.TABLE):
            raise ValueError(
                f"{where}: help names no command of the command table: {help_}"
            )
        out.append(
            Fact(name, entry["home"], entry.get("section") or "", signatures, help_)
        )
    return out


def _home(fact: Fact) -> str:
    """The home as a copy's problem line names it: a command's help as a person types it, else the doc and its
    heading."""
    if fact.help:
        return f"`{program.NAME} {fact.help} --help`"
    return f'{fact.home}, "{fact.section}"' if fact.section else fact.home


def _at_home(t: Text, fact: Fact) -> bool:
    return t.path == fact.home and (not fact.section or fact.section in t.sections)


def homes(found, facts) -> list:
    """The copies of FACTS in the texts FOUND, and what each home among them lacks: its section or a signature."""
    out = []
    for t in found:
        if t.generated:
            continue
        for fact in facts:
            if _at_home(t, fact):
                continue
            for signature in fact.signatures:
                for m, line, _at in _matches(t, signature):
                    verb = "point to" if fact.help else "link to"
                    what = f'a copy of {fact.name} ("{m.group()}"): {verb} {_home(fact)} instead'
                    out.append(Problem(t.path, line, "homes", what))
    given = {t.path for t in found}
    for fact in facts:
        if fact.home not in given:
            continue
        home = [t for t in found if _at_home(t, fact)]
        if not home:
            what = f'no section "{fact.section}", the home of {fact.name} ({HOMES})'
            out.append(Problem(fact.home, 1, "homes", what))
            continue
        for signature in fact.signatures:
            if not any(True for t in home for _ in _matches(t, signature)):
                what = (
                    f"the home of {fact.name} has no match of {signature.pattern!r}: write the fact here, or change "
                    f"its signature in {HOMES}"
                )
                out.append(Problem(fact.home, home[0].line, "homes", what))
    return out


# --- keys --------------------------------------------------------------------------------------------------------------

# A dotted key: lowercase words joined by dots, each a word, * or a word in angle brackets; not in a path, an address
# or a call, nor in a name a value of the code starts or ends (VALUE).
KEY = re.compile(
    r"(?<![\w./<>}-])[a-z_][a-z0-9_]*(?:\.(?:[a-z0-9_]+|\*|<[A-Za-z0-9_-]+>))+(?![\w(/<{-]|\.[\w*<])"
)
# The last words of a file's name or a host's (`host.docker.internal`), which no key ends with.
NOT_KEYS = frozenset(
    "md py json yaml yml txt sh html jsonl toml cfg ini service log tar gz tgz zip env lock pem pub conf csv png svg "
    "css js com org net io ai dev internal".split()
)


class Keys(NamedTuple):
    """The keys of the settings files and the internal keys, each the tuple of its words (* for a map's entries),
    with every key above it; INSIDE, the internal keys whose value is a mapping, each name inside one an internal key
    too; and ALLOWED, the places that keep the keys the lint would find there (KEYS: `load_key_allowances`).
    """

    settings: frozenset
    internal: frozenset
    inside: frozenset = frozenset()
    allowed: tuple = ()


def _with_parents(words) -> set:
    return {tuple(words[: n + 1]) for n in range(len(words))}


def _schema_keys(root, rel: str) -> list:
    """(path, node) of each key of the schema REL in the package at ROOT (settings_file.schema_keys)."""
    schema = json.loads((pathlib.Path(root) / rel).read_text(encoding="utf-8"))
    return [(path, node) for path, node, _req in settings_file.schema_keys(schema)]


def load_key_allowances(root=ROOT) -> tuple:
    """The allowances of KEYS in the package at ROOT, its list `allowed`: each names its files and why."""
    data = answers.load(pathlib.Path(root) / KEYS)
    _fields(data, {"allowed"}, KEYS)
    return _allowances(data.get("allowed"), KEYS, required=("files", "why"))


def load_keys(root=ROOT) -> Keys:
    """The keys in the package at ROOT: the settings files' (SCHEMAS); and the internal keys, the registry's
    (rkmon_lib/registry.py, the running package's whatever ROOT is): each key's name (a deploy key's, the
    configuration's name for it), a state key's key in the state file, and each name the registry describes (a
    section, an entry), with every key above each; INSIDE, those of them whose value is a mapping (registry.kind's
    ENTRY: a name inside one); and ALLOWED, the allowances of KEYS (`load_key_allowances`).
    """
    settings = set()
    for rel in SCHEMAS:
        for path, _node in _schema_keys(root, rel):
            settings |= _with_parents(path.split("."))
    internal, inside = set(), set()
    for k in registry.keys():
        # a deploy key's file keys are the settings file's own
        for name in [k.name] + (
            [] if k.owner == registry.DEPLOY else list(k.file_keys)
        ):
            internal |= _with_parents(name.split("."))
            if isinstance(k.default, dict):
                inside.add(tuple(name.split(".")))
    for name in registry.descriptions():
        internal |= _with_parents(name.split("."))
    return Keys(
        frozenset(settings),
        frozenset(internal),
        frozenset(inside),
        load_key_allowances(root),
    )


def _has(keys: frozenset, words: tuple) -> bool:
    if words in keys:
        return True
    return any(
        len(k) == len(words)
        and all(a == b or "*" in (a, b) or b.startswith("<") for a, b in zip(k, words))
        for k in keys
    )


def _headers(found) -> set:
    """(path, line) of each text among FOUND that is in a header (HEADERS): a comment at its file's top, before its
    first code."""
    out, past = set(), set()
    for t in found:
        if t.path in HEADERS and t.path not in past:
            if t.code:
                past.add(t.path)
            else:
                out.add((t.path, t.line))
    return out


def keys(found, known: Keys) -> list:
    """Each dotted key among FOUND that is no key of a settings file: in the user docs and the headers (HEADERS), each
    one; in the other messages, each that is no internal key either; but where an allowance of KNOWN keeps it.
    """
    tops = {k[0] for k in known.settings | known.internal}
    header = _headers(found)
    out = []
    for t in found:
        user = t.kind == USER or (t.path, t.line) in header
        if t.generated or not (user or t.kind == MESSAGE):
            continue
        raw = _unkept(t, known.allowed, t.text)
        if raw is None:
            continue
        for m, line, _at in _matches(t, KEY, raw):
            words = tuple(m.group().split("."))
            if (
                words[0] not in tops
                or words[-1] in NOT_KEYS
                or _has(known.settings, words)
            ):
                continue
            if _has(known.internal, words) or any(
                _has(known.inside, words[:n]) for n in range(1, len(words))
            ):
                if not user:
                    continue
                what = f"{m.group()} is an internal key: a user doc names only the settings files' keys"
            else:
                what = f"{m.group()}: no settings file has this key, and no internal key is named so"
            out.append(Problem(t.path, line, "keys", what))
    return out


# --- words -------------------------------------------------------------------------------------------------------------


class WordList(NamedTuple):
    """A list of WORDS: its name; what finds its words (`word_pattern`); what to write instead; whether it reads code;
    whether every file of the package is read for it (`every_file`); and its allowances.
    """

    name: str
    pattern: re.Pattern
    say: str
    code: bool
    every_file: bool
    allowed: tuple


def word_pattern(entry: dict) -> re.Pattern:
    """What finds a list's words and patterns, as whole words: in any case, unless the list says match_case."""
    forms = [
        (r"(?<!\w)" if re.match(r"\w", word) else "")
        + re.escape(word)
        + (r"(?!\w)" if re.search(r"\w$", word) else "")
        for word in entry.get("words") or ()
    ]
    forms += [rf"\b(?:{pattern})\b" for pattern in entry.get("patterns") or ()]
    return re.compile("|".join(forms), 0 if entry.get("match_case") else re.I)


def load_words(root=ROOT) -> list:
    """The lists of WORDS in the package at ROOT."""
    out = []
    known = {"words", "patterns", "match_case", "say", "code", "every_file", "allowed"}
    for name, entry in answers.load(pathlib.Path(root) / WORDS).items():
        where = f"{WORDS}, {name}"
        _fields(entry, known, where, ("say",))
        flags = 0 if entry.get("match_case") else re.I
        out.append(
            WordList(
                name,
                word_pattern(entry),
                entry["say"],
                bool(entry.get("code")),
                bool(entry.get("every_file")),
                _allowances(entry.get("allowed"), where, flags),
            )
        )
    return out


def words(found, lists, kinds=(USER, SKILL, MESSAGE), generated_too=False) -> list:
    """Each word of LISTS in the texts FOUND of KINDS (in a generated section only with GENERATED_TOO) that no
    allowance keeps; a word two lists find is one problem, which names both."""
    out = []
    for t in found:
        if t.kind not in kinds or (t.generated and not generated_too):
            continue
        # the offset a match starts at -> [its line, its words, the lists that found it]
        hits = {}
        for wl in lists:
            if t.code and not wl.code:
                continue
            raw = _unkept(t, wl.allowed, t.text if wl.code else _prose(t.text))
            if raw is None:
                continue
            for m, line, at in _matches(t, wl.pattern, raw):
                hit = hits.setdefault(at, [line, m.group(), []])
                if len(m.group()) > len(hit[1]):
                    hit[1] = m.group()
                hit[2].append(wl)
        for at in sorted(hits):
            line, word, by = hits[at]
            what = f'"{word}" ({", ".join(w.name for w in by)}): {"; ".join(w.say for w in by)}'
            out.append(Problem(t.path, line, "words", what))
    return out


# --- old command names -------------------------------------------------------------------------------------------------


class OldCommand(NamedTuple):
    """An old command of OLD_COMMANDS: its name; what to run in its place; what finds it; whether every file of the
    package is read for it (`every_file`); and its allowances."""

    name: str
    instead: str
    pattern: re.Pattern
    every_file: bool
    allowed: tuple


# Where a text names a command of the program's that is one word: after the program's name (or a path that ends in it:
# ./rkmon, docker/rkmon), after an f-string's value (VALUE), which holds the name where the program prints it, at a code
# span's opening, or at the start of a text that goes on from the name (" guide build": `program.name() + " guide
# build"`).
ONE_WORD_COMMAND = rf"(?:^ |\b{re.escape(program.NAME)} |`|{re.escape(VALUE)} )"


def gone_commands(gone) -> list:
    """An old command for each row of GONE (commands.GONE, in rkmon_lib/commands.py: the one home of each command and
    option of the program's that another command replaced): its name and what to run instead (the command that
    replaces it, else the row's own words: Gone.replacement), each with the program's name, read in every file, and
    found by its command's words (a command of one word only where a text names a command: ONE_WORD_COMMAND), then
    the option that went, if any, before a code span ends; never as a file's name (`settings.py`).
    rkmon_lib/commands.py keeps each row's words.
    """
    keep = "^(?:" + "|".join(re.escape(g.words) for g in gone) + ")$"
    allowed = (
        Allowance(
            re.compile(r"rkmon_lib/commands\.py"),
            (),
            "",
            re.compile(keep, re.M),
            "the parser names each command and option another replaced with the command that replaces it",
        ),
    )
    out = []
    for g in gone:
        pattern = ONE_WORD_COMMAND if len(g.command) == 1 else r"\b"
        pattern += re.escape(" ".join(g.command))
        if g.option:
            pattern += r"\b[^`]*?" + re.escape(g.option)
        out.append(
            OldCommand(
                f"{program.NAME} {g.words}",
                (
                    f"{program.NAME} {g.instead}"
                    if g.instead
                    else g.replacement(program.NAME)
                ),
                re.compile(pattern + r"(?![\w-]|\.\w)"),
                True,
                allowed,
            )
        )
    return out


def load_old_commands(root=ROOT) -> dict:
    """The old commands, by name: those of commands.GONE (`gone_commands`), then those of OLD_COMMANDS in the package
    at ROOT, which has none of GONE's (a ValueError otherwise)."""
    out = {c.name: c for c in gone_commands(commands.GONE)}
    known = {"instead", "pattern", "every_file", "allowed"}
    for name, entry in answers.load(pathlib.Path(root) / OLD_COMMANDS).items():
        where = f"{OLD_COMMANDS}, {name}"
        if name in out:
            raise ValueError(f"{where}: it is in rkmon_lib/commands.py's GONE")
        _fields(entry, known, where, ("instead", "pattern"))
        out[name] = OldCommand(
            name,
            entry["instead"],
            re.compile(entry["pattern"]),
            bool(entry.get("every_file")),
            _allowances(entry.get("allowed"), where),
        )
    return out


def old_commands(found, commands, generated_too=False) -> list:
    """Each old command of COMMANDS in the texts FOUND (in a generated section only with GENERATED_TOO) that no
    allowance keeps."""
    out = []
    for t in found:
        if t.generated and not generated_too:
            continue
        for c in commands:
            raw = _unkept(t, c.allowed, t.text)
            if raw is None:
                continue
            for m, line, _at in _matches(t, c.pattern, raw):
                what = (
                    f'"{m.group()}": {c.name} is an old command; instead: {c.instead}'
                )
                out.append(Problem(t.path, line, "old-commands", what))
    return out


# --- quoted output -----------------------------------------------------------------------------------------------------

# A word in angle brackets, or an ellipsis, of a quote: any text.
PLACE = re.compile(r"<[^<>]+>|…|\.\.\.")
LETTERS = re.compile(r"[A-Za-z]{2,}")
# A path, a run of characters with a / in it (`~/rkmon-build.yaml`): one word, as a quote may give a path of its own
# where the code prints a value (`FAIL  private file: ~/rkmon-build.yaml exists`).
PATH_WORD = re.compile(r"\S*/\S*")
# A field of a text the code fills in with str.format: a value.
FIELD = re.compile(r"\{[^{}]*\}")
# What a quote, or a text of the code, writes before the program's name in a path to it (./rkmon, docker/rkmon): the
# lint reads the name alone (program.NAME), as the messages name the program.
BEFORE_NAME = re.compile(
    rf"(?<![\w./-])(?:\./|docker/)(?={re.escape(program.NAME)}(?![\w-]))"
)
# A check row as a quote gives it (rkmon_lib/checks.py, render): its status, its name, a colon and what was found; or
# its status, a colon and a text (`OFF: …`).
ROW = re.compile(
    rf"(?:{'|'.join(re.escape(str(s)) for s in checks.Status)})(?:: (.+)| ([^:]+?): (.+))"
)
# The labels a fix, a next line and a note start with (checks.FIX_LABEL, steps.NEXT_LABEL, progress.NOTE).
LABELS = (checks.FIX_LABEL, steps.NEXT_LABEL, progress.NOTE)
LABELLED = re.compile(rf"({'|'.join(re.escape(label) for label in LABELS)})(.+)")
# A step a next line names (steps.step_command): its number and its name, then its command, if any.
STEP = re.compile(r"step (\d+), ([^:;]+?)(?:: (.+))?")
# The files a fix names in words, where no path of one is at hand.
FILES_IN_WORDS = (deploy_tree.FILE, image.BUILD_FILE)
# The file of an edit: a file in words (FILES_IN_WORDS), a word in angle brackets, or one word.
EDITED_FILE = rf"(?:{'|'.join(re.escape(f) for f in FILES_IN_WORDS)}|<[^<>]+>|\S+)"
# An edit of a fix (checks.Edit): `set KEY in FILE to VALUE`, its value up to the end, a command (`\0`), `; `, or a
# word that joins steps (wrapping.JOINS) after a comma or before another step; or `delete KEY from FILE`.
EDIT = re.compile(
    rf"\bset (\S+) in ({EDITED_FILE}) to (.+?)"
    rf"(?=$|\0|; |, (?:{'|'.join(wrapping.JOINS)})\b| (?:{'|'.join(wrapping.JOINS)}) (?:set|delete|run)\b)"
    rf"|\bdelete (\S+) from ({EDITED_FILE})"
)
# An edit's value written as JSON (checks.Edit): a value of the code, never one of its texts.
JSON_VALUE = re.compile(r'"[^"]*"|-?\d+(?:\.\d+)?|true|false|null|\[.*\]')


class Template(NamedTuple):
    """A text of the code, with each run of white space as one space: the text and its longest part between values;
    its `regex` matches it whole (`_whole`)."""

    text: str
    longest: str

    @property
    def regex(self) -> re.Pattern:
        return _whole(self.text)


@functools.lru_cache(maxsize=None)
def _whole(text: str) -> re.Pattern:
    """What matches TEXT whole, each VALUE in it any text; compiled when it is first asked for."""
    return re.compile(".+?".join(re.escape(p) for p in text.split(VALUE)), re.S)


def _word_count(text: str) -> int:
    """The words of two letters or more of TEXT, outside its values, its words in angle brackets and its ellipses; a
    path with such words in it (PATH_WORD) is one word."""
    plain = PLACE.sub(" ", text.replace(VALUE, " "))
    paths = sum(1 for p in PATH_WORD.findall(plain) if LETTERS.search(p))
    return paths + len(LETTERS.findall(PATH_WORD.sub(" ", plain)))


def _template(text: str) -> Template:
    return Template(text, max(text.split(VALUE), key=len))


def _python_files(root) -> list:
    """The Python files of the package at ROOT (`_reader`), by their paths from it."""
    out = []
    for rel in package_files(root):
        first = ""
        if not posixpath.splitext(posixpath.basename(rel))[1]:
            try:
                with open(pathlib.Path(root) / rel, encoding="utf-8") as f:
                    first = f.readline()
            except (OSError, UnicodeDecodeError):
                continue
        if _reader(rel, first) is _python:
            out.append(rel)
    return out


def _term(node) -> str:
    """The text of a term of a sum: a string's, an f-string's (VALUE in place of each of its values), else VALUE."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        return "".join(
            v.value if isinstance(v, ast.Constant) else VALUE for v in node.values
        )
    return VALUE


def joined_texts(tree) -> list:
    """Each text the Python code TREE (ast.parse's) joins with +, whole: its terms' texts (`_term`) in order, for each
    sum that is no term of another and has a string or an f-string among its terms; each `{program}` read as the
    command's name (program.NAME), as `_python` reads it."""

    def is_sum(n) -> bool:
        return isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add)

    def terms(n) -> list:
        return terms(n.left) + terms(n.right) if is_sum(n) else [n]

    sums = [n for n in _nodes(tree) if is_sum(n)]
    inner = {id(side) for n in sums for side in (n.left, n.right) if is_sum(side)}
    out = []
    for n in sums:
        if id(n) in inner:
            continue
        found = terms(n)
        if any(
            isinstance(t, ast.JoinedStr)
            or (isinstance(t, ast.Constant) and isinstance(t.value, str))
            for t in found
        ):
            text = "".join(_term(t) for t in found)
            out.append(text.replace("{program}", program.NAME))
    return out


_TEMPLATES = {}


def load_templates(root=ROOT) -> tuple:
    """The texts of the code of the package at ROOT: its Python files' (each string it may print, `_python`, and each
    text it joins with +, `joined_texts`), each field str.format fills in a value; and each key's `x-source` in the
    settings files' schemas the package has (SCHEMAS), as a fix's edit prints it for the key's value
    (settings_file.source_of), with no field in it. Each is read whole and each part of it between `; `, that have two
    words or more, with the program's name read alone (BEFORE_NAME); a text that starts with its outcome word or a
    label (LABELS) is also read without it, as a quote's parts are. Read once for each package folder.
    """
    key = str(pathlib.Path(root).resolve())
    if key not in _TEMPLATES:
        quote, seen = _outcome(), set()

        def add(raw: str) -> None:
            flat = _flat(BEFORE_NAME.sub("", raw))[0].strip()
            for text in (flat, *flat.split("; ")):
                text = text.strip()
                m = quote.match(text)
                rest = m.group(3)[2:] if m and m.group(3).startswith(": ") else ""
                labelled = LABELLED.fullmatch(text)
                after = labelled.group(2) if labelled else ""
                seen.update(
                    each for each in (text, rest, after) if _word_count(each) >= 2
                )

        for rel in _python_files(root):
            try:
                source = (pathlib.Path(root) / rel).read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            tree = ast.parse(source)
            raws = [t.text for t in _python(rel, source, tree)] + joined_texts(tree)
            for raw in raws:
                add(FIELD.sub(VALUE, raw))
        for rel in SCHEMAS:
            if (pathlib.Path(root) / rel).is_file():
                for _path, node in _schema_keys(root, rel):
                    if node.get("x-source"):
                        add(str(settings_file.source_of(node)))
        _TEMPLATES[key] = tuple(_template(text) for text in sorted(seen))
    return _TEMPLATES[key]


def _outcome() -> re.Pattern:
    """An outcome as a quote gives it: the words in capitals before its word, if any; its word (rkmon_lib/outcome.py,
    WORDS); and the rest."""
    named = sorted(outcome.WORDS, key=len, reverse=True)
    return re.compile(
        rf"((?:[A-Z][A-Z0-9-]* )*?)({'|'.join(re.escape(w) for w in named)})(?=$|[: ])(.*)",
        re.S,
    )


def _quotes(t: Text):
    """(line, text) of each quote of T: each line of a fenced block, or each text in backticks."""
    if t.code:
        for i, line in enumerate(t.text.split("\n")):
            yield t.line + i, " ".join(line.split())
    else:
        for m in CODE.finditer(t.text):
            yield t.line + t.text.count("\n", 0, m.start()), " ".join(
                m.group()[1:-1].split()
            )


def _counts(tp: Template, edits: bool) -> bool:
    """Whether the template TP counts: with EDITS, each does; without, none of an edit's own form (EDIT: the text of
    checks.Edit), which a fix's edits are read by."""
    return edits or not EDIT.fullmatch(tp.text)


def _printed(part: str, templates, edits: bool = True) -> bool:
    """Whether the code prints PART: one of TEMPLATES (each that counts, `_counts`) matches it whole, or it is in one,
    each of its words in angle brackets and its ellipses any text; or one cut where PART's own ends are such a word
    matches the rest (`_cut`)."""
    pieces = PLACE.split(part)
    asked = re.compile(".+?".join(re.escape(p) for p in pieces), re.S)
    longest = max(pieces, key=len)
    return any(
        (
            (tp.longest in part and tp.regex.fullmatch(part))
            or (longest in tp.text and asked.search(tp.text))
        )
        and _counts(tp, edits)
        for tp in templates
    ) or _cut(part, templates, edits)


# What a quote's part has before its first word, and after its last: marks, and words in angle brackets.
LEADING = re.compile(r"(?:[^\w<]|<[^<>]+>)*")
TRAILING = re.compile(r"(?:[^\w>]|<[^<>]+>)*\Z")


def _cut(part: str, templates, edits: bool = True) -> bool:
    """Whether PART, which starts or ends with a word in angle brackets or an ellipsis that stands for the end of a
    text, with values of its own between, is a part of a text of the code: a template (each that counts) cut before
    the word PART starts with after it, or after the word PART ends with before it, matches the words between whole,
    each of the template's values any text."""
    lead = LEADING.match(part).group()
    trail = TRAILING.search(part, len(lead)).group()
    core = part[len(lead) : len(part) - len(trail)]
    cut_start, cut_end = bool(PLACE.search(lead)), bool(PLACE.search(trail))
    first, last = re.match(r"\w+", core), re.search(r"\w+\Z", core)
    if not (cut_start or cut_end) or not first or not last:
        return False
    first, last = first.group(), last.group()
    for tp in templates:
        if (cut_start and first not in tp.text) or (cut_end and last not in tp.text):
            continue
        if not _counts(tp, edits):
            continue
        starts = (
            [m.start() for m in re.finditer(rf"\b{re.escape(first)}\b", tp.text)]
            if cut_start
            else [0]
        )
        ends = (
            [m.end() for m in re.finditer(rf"\b{re.escape(last)}\b", tp.text)]
            if cut_end
            else [len(tp.text)]
        )
        for s in starts:
            for e in ends:
                cut = tp.text[s:e]
                if (
                    s < e
                    and all(p in core for p in cut.split(VALUE))
                    and _whole(cut).fullmatch(core)
                ):
                    return True
    return False


# id(templates) -> (templates, the texts `_texts_as_values` looks for, longest first)
_PLAIN_TEXTS = {}


def _texts_as_values(part: str, templates) -> str:
    """PART with each text of the code found whole in it as an ellipsis, as the code puts one text in another's value:
    each template of three words or more that has no value, from its first word to its last (the marks at its ends
    left out), between word boundaries in PART; the longest first."""
    key = id(templates)
    if key not in _PLAIN_TEXTS or _PLAIN_TEXTS[key][0] is not templates:
        plain = set()
        for tp in templates:
            inner = re.sub(r"^\W+|\W+$", "", tp.text)
            if VALUE not in tp.text and _word_count(inner) >= 3:
                plain.add(inner)
        _PLAIN_TEXTS[key] = (templates, sorted(plain, key=lambda w: (-len(w), w)))
    for inner in _PLAIN_TEXTS[key][1]:
        if inner in part:
            part = re.sub(rf"(?<!\w){re.escape(inner)}(?!\w)", "…", part)
    return part


# A text in parentheses, which the code often joins from another text of its own: `NAME (WHY)`.
ASIDE = re.compile(r"\(([^()]*)\)")


def _printed_part(
    part: str, least: int, templates, edits: bool = True, asides: bool = True
) -> bool:
    """Whether PART has fewer than LEAST words, or the code prints it (`_printed`): whole, once each text of the code
    found whole in it reads as any text (`_texts_as_values`), or (with ASIDES) with each text in parentheses (ASIDE)
    any text, each of those texts printed on its own."""
    if _word_count(part) < least or _printed(part, templates, edits):
        return True
    rest = _texts_as_values(part, templates)
    if rest != part and (_word_count(rest) < least or _printed(rest, templates, edits)):
        return True
    inner = ASIDE.findall(part)
    return (
        asides
        and bool(inner)
        and _printed_part(ASIDE.sub("(…)", part), least, templates, edits, False)
        and all(_printed_part(i, 3, templates, edits, False) for i in inner)
    )


def _unprinted_text(text: str, least: int, templates, edits: bool = True) -> str:
    """The first part of TEXT between `; ` that the code does not print (`_printed_part`); none ("") when it prints
    TEXT whole, or each of its parts."""
    text = text.strip()
    if _printed_part(text, least, templates, edits):
        return ""
    for part in text.split("; "):
        part = part.strip()
        if not _printed_part(part, least, templates, edits):
            return part
    return ""


class _Command(NamedTuple):
    """A command of the program in a text: where it starts and ends in the text, and its words after the name."""

    start: int
    end: int
    words: tuple


def _commands(text: str) -> list:
    """Each command of the program in TEXT, a fix, a next line or a note, as the wrap reads it and keeps it whole
    (wrapping.program_commands): where it starts and ends, and its words after the program's name.
    """
    return [
        _Command(start, end, tuple(text[start:end].split()[1:]))
        for start, end in wrapping.program_commands(text)
    ]


def _names_a_command(words) -> bool:
    """Whether WORDS, a command's words after the program's name, start with a command of the table (commands.TABLE)
    once the options before it are left out (commands.LEADING's with their values); words that name no command, or a
    placeholder for one, do too."""
    leading = {f: o for o in commands.LEADING for f in o.flags}
    words, i = list(words), 0
    while i < len(words) and words[i].startswith("-"):
        option = leading.get(words[i])
        i += 2 if option is not None and option.metavar else 1
    words = words[i:]
    if not words or PLACE.fullmatch(words[0]):
        return True
    return any(
        commands.find(words[:n], commands.TABLE) for n in range(len(words), 0, -1)
    )


def _pieces(text: str) -> list:
    """TEXT, a fix, a next line or a note, in its pieces, in order: ("command", WORDS) for each command of the program
    (`_commands`), ("edit", VALUE) for each edit (EDIT; "" for a delete), and ("words", TEXT) for each run of the other
    words between them."""
    found = _commands(text)
    marked, at = "", 0
    for c in found:
        marked += text[at : c.start] + "\0"
        at = c.end
    marked += text[at:]
    out, at, commands_found = [], 0, iter(found)
    for m in re.finditer(rf"\0|{EDIT.pattern}", marked):
        out.append(("words", marked[at : m.start()]))
        if m.group() == "\0":
            out.append(("command", next(commands_found).words))
        else:
            out.append(("edit", (m.group(3) or "").strip()))
        at = m.end()
    out.append(("words", marked[at:]))
    return out


def _unprinted_step(n: str, name: str, command: Optional[str]) -> str:
    """The part of step N, NAME and its COMMAND that is not printed: none ("") when it is a step of either path
    (rkmon_lib/steps.py) and COMMAND, if any, starts with the program's name and that step's command, or one of its
    others (steps.command_name)."""
    found = [
        s
        for s in steps.BUILD_STEPS + steps.DEPLOY_STEPS
        if str(s.n) == n and s.name == name
    ]
    if not found:
        return f"step {n}, {name}"
    if command is None:
        return ""
    first = next(iter(_commands(command)), None)
    names = [
        steps.command_name(c).split() for s in found for c in (s.command, *s.others)
    ]
    if first and not first.start:
        if any(list(first.words[: len(w)]) == w for w in names):
            return ""
    return f"step {n}, {name}: {command}"


def _unprinted_line(label: str, text: str, templates) -> str:
    """The first part of TEXT, a fix, a next line or a note after its LABEL, that the code (TEMPLATES) does not print;
    "" for none (the module's docstring says how it reads)."""
    step = STEP.fullmatch(text) if label == steps.NEXT_LABEL else None
    if step:
        wrong = _unprinted_step(*step.groups())
        if wrong or step.group(3) is None:
            return wrong
        text = step.group(3)
    for tag, piece in _pieces(text):
        if tag == "command":
            if not _names_a_command(piece):
                return " ".join((program.NAME, *piece))
        elif tag == "edit":
            if not JSON_VALUE.fullmatch(piece):
                wrong = _unprinted_text(piece, 3, templates)
                if wrong:
                    return wrong
        else:
            wrong = _unprinted_text(piece, 3, templates, edits=False)
            if wrong:
                return wrong
    return ""


def _unprinted(text: str, templates) -> str:
    """The first part of TEXT, a quote, that the code (TEMPLATES) does not print; "" when it prints each, or when the
    quote is none of an outcome, a check row, a fix, a next line and a note (the module's docstring says how it
    reads)."""
    m = _outcome().match(text)
    if m:
        before, _word, rest = m.groups()
        if before:
            # one the code printed once, unless it prints it still
            printed = _word_count(text) < 2 or _printed(text, templates)
            return "" if printed else text
        if rest.startswith(": "):
            return _unprinted_text(rest[2:], 3, templates)
        return ""  # the word alone, or an outcome that names a thing
    row = ROW.fullmatch(text)
    if row:
        return _unprinted_text(row.group(1) or row.group(3), 3, templates)
    labelled = LABELLED.fullmatch(text)
    if labelled:
        return _unprinted_line(labelled.group(1), labelled.group(2), templates)
    return ""


def quoted_output(found, templates) -> list:
    """Each outcome, check row, fix, next line and note a user doc among FOUND quotes that the code, TEMPLATES, does not
    print."""
    out = []
    for t in found:
        if t.generated or t.kind != USER:
            continue
        for line, text in _quotes(t):
            part = _unprinted(BEFORE_NAME.sub("", text), templates)
            if part:
                what = f'"{text}": the code prints no "{part}"; quote what it prints'
                out.append(Problem(t.path, line, "quoted-output", what))
    return out


# --- links and anchors -------------------------------------------------------------------------------------------------

LINK = re.compile(r"!?\[[^\]]*\]\(<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\)")
SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*:|//")
# A doc and a heading of it, as a text names them: the doc's file name, a comma, then the heading in double quotes.
SECTION = re.compile(r"(?<![\w./-])((?:[\w.-]+/)*[\w.-]+\.md), [\"“]([^\"”]+)[\"”]")
NAMED_ANCHOR = re.compile(r"<a\s+(?:name|id)=\"([^\"]+)\"")


def slug(heading: str) -> str:
    """GitHub's anchor for a heading."""
    s = heading.strip().lower().replace("`", "")
    s = re.sub(r"[^\w\- ]", "", s)
    return s.replace(" ", "-")


def anchors(path) -> set:
    """The anchors of the markdown file at PATH: its headings', as GitHub makes them (the second heading of the same
    words gets -1, the third -2), and each <a name> or <a id>."""
    source = pathlib.Path(path).read_text(encoding="utf-8")
    out, seen = set(NAMED_ANCHOR.findall(source)), {}
    for t in _markdown(pathlib.Path(path).name, source):
        if t.heading:
            s = slug(t.text)
            n = seen.get(s, 0)
            out.add(s if n == 0 else f"{s}-{n}")
            seen[s] = n + 1
    return out


def links(found, root) -> list:
    """Each link of the markdown texts among FOUND that does not reach its file or its anchor in the package at ROOT,
    and each doc and heading any text names that the package lacks."""
    root = pathlib.Path(root)
    known = {}

    def anchors_of(rel):
        if rel not in known:
            known[rel] = anchors(root / rel)
        return known[rel]

    out = []
    for t in found:
        if t.path.endswith(".md") and not t.code:
            for m, line, _at in _matches(t, LINK, _prose(t.text)):
                target = m.group(1)
                if SCHEME.match(target):
                    continue
                path, _, anchor = target.partition("#")
                rel = t.path
                if path:
                    rel = posixpath.normpath(
                        posixpath.join(
                            posixpath.dirname(t.path), urllib.parse.unquote(path)
                        )
                    )
                if rel == ".." or rel.startswith("../"):
                    what = f"the link {target} leaves the package"
                elif not (root / rel).exists():
                    what = f"the link {target}: the package has no {rel}"
                elif anchor and rel.endswith(".md") and anchor not in anchors_of(rel):
                    what = f"the link {target}: {rel} has no heading whose anchor is #{anchor}"
                else:
                    continue
                out.append(Problem(t.path, line, "links", what))
        for m, line, _at in _matches(t, SECTION):
            doc, heading = m.group(1), m.group(2)
            rel = doc
            if not (root / doc).is_file():
                rel = posixpath.normpath(posixpath.join(posixpath.dirname(t.path), doc))
            if not (root / rel).is_file():
                what = f'{doc}, "{heading}": the package has no {doc}'
            elif slug(heading) not in anchors_of(rel):
                what = f'{doc}, "{heading}": {doc} has no such heading'
            else:
                continue
            out.append(Problem(t.path, line, "links", what))
    return out


# --- placeholders ------------------------------------------------------------------------------------------------------

# A name in angle brackets, but a regular expression's group name: (?P<name>...)
PLACEHOLDER = re.compile(r"(?<!\(\?P)<([A-Za-z_][A-Za-z0-9_-]*)>")
CLOSING_TAG = re.compile(r"</([A-Za-z][A-Za-z0-9-]*)>")
# The HTML elements that have no closing tag. Any other tag is HTML where its file closes it.
VOID_TAGS = frozenset(
    "area br col embed hr img input link meta param track wbr".split()
)
# The image's build file, and a step's mount of a build secret in it, by the secret's id: a masked line of a build
# tool shows a download location as that id in angle brackets (docker/masked.sh; the glossary's "Masked name").
DOCKERFILE = "docker/Dockerfile"
SECRET_MOUNT = re.compile(r"--mount=type=secret,id=([a-z_]+)")


def masked_names(root=ROOT) -> frozenset:
    """The names a masked line of a build tool shows in angle brackets in place of a download location: the id of each
    build secret the build file of the package at ROOT mounts (DOCKERFILE, SECRET_MOUNT); none without that file.
    """
    path = pathlib.Path(root) / DOCKERFILE
    if not path.is_file():
        return frozenset()
    return frozenset(SECRET_MOUNT.findall(path.read_text(encoding="utf-8")))


def load_placeholders(root=ROOT) -> frozenset:
    """The placeholders, by name, that the glossary of the package at ROOT names (GLOSSARY), the agentic files'
    (rkmon_lib/agentic.py, PLACEHOLDERS), and the masked names of its build secrets (`masked_names`), which a quoted
    line of a build tool shows as they are."""
    doc, section = GLOSSARY
    found = texts(root, doc) if (pathlib.Path(root) / doc).is_file() else []
    names = {
        m.group(1)
        for t in found
        if section in t.sections
        for m in PLACEHOLDER.finditer(t.text)
    }
    names |= {
        m.group(1) for key in agentic.PLACEHOLDERS for m in PLACEHOLDER.finditer(key)
    }
    return frozenset(names | masked_names(root))


def load_placeholder_allowances(root=ROOT) -> tuple:
    """The allowances of PLACEHOLDERS in the package at ROOT, its list `allowed`: each names its files, the
    placeholders it keeps (keep) and why."""
    data = answers.load(pathlib.Path(root) / PLACEHOLDERS)
    _fields(data, {"allowed"}, PLACEHOLDERS)
    return _allowances(
        data.get("allowed"), PLACEHOLDERS, required=("files", "keep", "why")
    )


def placeholders(found, defined, allowed=()) -> list:
    """Each placeholder in the user docs, the skill files and the messages among FOUND that is not DEFINED, but where
    an allowance of ALLOWED keeps it."""
    closed = {}
    for t in found:
        closed.setdefault(t.path, set()).update(
            n.lower() for n in CLOSING_TAG.findall(t.text)
        )
    out = []
    for t in found:
        if t.generated or t.kind not in (USER, SKILL, MESSAGE):
            continue
        raw = _unkept(t, allowed, t.text)
        if raw is None:
            continue
        for m, line, _at in _matches(t, PLACEHOLDER, raw):
            name = m.group(1)
            if (
                name in defined
                or name.lower() in VOID_TAGS
                or name.lower() in closed[t.path]
            ):
                continue
            what = (
                f'{m.group()} is not in the glossary ({GLOSSARY[0]}, "{GLOSSARY[1]}")'
            )
            out.append(Problem(t.path, line, "placeholders", what))
    return out


# --- the lints together ------------------------------------------------------------------------------------------------


def run(root, files, names=LINTS, data=None) -> list:
    """The problems the lints NAMES find in FILES, by their paths in the package at ROOT, with the data of the package
    at DATA (ROOT when None); sorted."""
    unknown = sorted(set(names) - set(LINTS))
    if unknown:
        raise ValueError(
            f"no lint {', '.join(unknown)}: the lints are {', '.join(LINTS)}"
        )
    root = pathlib.Path(root)
    data = root if data is None else pathlib.Path(data)
    found = [t for rel in files for t in texts(root, rel)]
    if not found:
        return []  # no text, no problem: the lints' data is not read
    out = []
    if "homes" in names:
        out += homes(found, load_homes(data))
    if "keys" in names:
        out += keys(found, load_keys(data))
    if "words" in names:
        out += words(found, load_words(data))
    if "old-commands" in names:
        out += old_commands(found, load_old_commands(data).values())
    if "quoted-output" in names:
        out += quoted_output(found, load_templates(data))
    if "links" in names:
        out += links(found, root)
    if "placeholders" in names:
        out += placeholders(
            found, load_placeholders(root), load_placeholder_allowances(data)
        )
    return sorted(out)


def every_file(root=ROOT) -> list:
    """The words of the lists, and the old commands, that say every_file, in every file of the package at ROOT, of
    every kind, its code and its generated sections included, that no allowance keeps; sorted.
    """
    lists = [w for w in load_words(root) if w.every_file]
    commands = [c for c in load_old_commands(root).values() if c.every_file]
    out = []
    for rel in package_files(root):
        found = texts(root, rel)
        out += words(found, lists, KINDS, generated_too=True)
        out += old_commands(found, commands, generated_too=True)
    return sorted(out)


def main(argv=None, root=ROOT, data=None) -> int:
    parser = argparse.ArgumentParser(
        prog="doc_lints.py", description=__doc__.split("\n\n")[0]
    )
    parser.add_argument(
        "--lint",
        action="append",
        choices=LINTS,
        help="run this lint, as often as needed (every lint by default)",
    )
    parser.add_argument(
        "files", nargs="+", metavar="FILE", help="a file of the package"
    )
    args = parser.parse_args(argv)
    root = pathlib.Path(root).resolve()
    rels = []
    for f in args.files:
        try:
            rels.append(pathlib.Path(f).resolve().relative_to(root).as_posix())
        except ValueError:
            parser.error(f"{f}: not a file of the package")
    try:
        problems = run(root, rels, args.lint or LINTS, data)
    except (OSError, ValueError) as e:
        print(f"doc_lints.py: {e}", file=sys.stderr)
        return 2
    for p in problems:
        print(p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
