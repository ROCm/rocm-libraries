# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No file of the package refers to the sections or parts of its settings design, a document the package does not
hold: the package's text says each thing itself, or names where it lives in the package. The test fails on each line
that refers to the design, by file and line.

The files are those tests/test_headers.py walks (`package_files`), but for this one, whose examples are the forms it
looks for. A Python file is read as a person reads it (`prose`): its comments and the text of its strings, its code
blanked, so a float in code is none of the forms; a JSON file by its strings alone (`strings`), so a number in data is
none either. A form in NAMED is a reference wherever it stands; a form in BARE is one unless an allowance in ALLOWANCES
holds each section number in it. KEEP has an example of each thing the package keeps that looks like a section number,
which no form finds.
"""

import bisect
import io
import pathlib
import re
import sys
import tokenize
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.dont_write_bytecode = True

import test_headers  # noqa: E402  (package_files; its own tests run in their module)

ROOT = test_headers.ROOT
# This file: its examples are the forms it looks for.
OWN = pathlib.Path(__file__).resolve()

# Between two words of a reference: blanks, with at most one line break (a sentence wrapped there).
GAP = r"(?:[ \t]+(?:\n[ \t]*)?|\n[ \t]*)"
# A section's number, such as 6.7, 2.12 or 4.1a: never 0.5 or 2.0, nor part of a longer one (1.6.2, 127.0.0.1, 2.38+).
NUM = r"(?<![\w.])[1-9]\.[1-9]\d?[a-c]?(?![\w+%]|\.\d)"
NUMBER = re.compile(NUM)
# What a section number does in a sentence that names it: "as 6.6 says", "as 4.3 words it".
VERBS = (
    "says|shows?|words|writes|runs|leaves|compares|keeps|counts|copies|has|lists|names|holds|sets|"
    "defines|describes|asks|wants|adds|calls|reads|requires"
)
# The programs and models whose releases the package names: "Python 3.10", "cmake 3.16", "Claude Opus 5.5".
PROGRAMS = (
    "Python|python3?|pytest|git|gh|cmake|CMake|Ollama|ROCm|Docker|SDK|Ubuntu|"
    "Claude|Opus|Sonnet|Haiku"
)
# The units of a quantity: "1.5 GiB", "4.5 minutes", "1.5 x IQR".
UNITS = "[KMGT]i?B|ms|s|sec|seconds?|min|minutes?|hours?|days?|weeks?|times|x|×"

# The forms that name the design, by name: each match is a reference.
NAMED = {
    # the design and a section's or a part's number: "design 6.7", "(design 2.8)", "design's 4.1", "design Part 0"
    "design_number": re.compile(
        rf"\b[Dd]esign(?:'s)?[,:]?{GAP}"
        rf"(?:Parts?{GAP}|[Ss]ections?{GAP}|§)?\d+(?:\.\d+)*"
    ),
    # a row's note of the design's sections, read in the file as it is (code too): "design": "8.3", design="8.3"
    "design_note": re.compile(
        r"""\bdesign["']?[ \t]*[:=][ \t]*["'][^"'\n]*?"""
        rf"(?:{NUM}|\b(?:Parts?|[Dd]ecisions?)[ \t]+\d)"
    ),
    # one of the design's parts, capitalized as the design names them: "Part 5" (a split review's "part 2" is none)
    "part_number": re.compile(rf"\bParts?{GAP}\d+\b"),
    # one of the design's decisions: "decision 5", "decisions 5 and 13"
    "decision_number": re.compile(rf"\b[Dd]ecisions?{GAP}\d+\b"),
    # an item of one of the design's lists, alone in parentheses: "(item 9)", "(items 8 and 9: the caches)"
    "item_number": re.compile(rf"\(\s*[Ii]tems?{GAP}\d+"),
    # the design itself, by a name the package's own short design (docs/design.md) never needs: "the design's package"
    "the_design": re.compile(
        rf"\b[Tt]he{GAP}design's\b"
        rf"|\b(?:[Ss]ettings|[Pp]rivate){GAP}design\b"
        rf"|\bdesign{GAP}doc(?:ument)?s?\b"
    ),
}

# The forms in which a bare section number reads as a pointer, by name: each match is a reference unless an allowance
# holds each section number in it.
BARE = {
    # parentheses that open with section numbers (a call's are code): "(6.6)", "(2.6, 8.1)", "(6.1: it changes nothing)"
    "in_parentheses": re.compile(
        rf"(?<![\w.\])])\(\s*{NUM}(?=\s*[,;:)]|{GAP}(?:and|or|to){GAP}{NUM})"
    ),
    # a later item of a list, after , ; or : and a blank (a quoted key's value is data): "(states.row, 6.5: PASS)"
    "in_a_list": re.compile(rf"(?:[,;]|(?<![\"']):){GAP}{NUM}(?=[ \t]*[,;:)])"),
    # a section number as a name: "8.3's round trip", "as 6.6 says", "as 4.3 words it", "steps 1 to 9 of 6.5"
    "as_a_name": re.compile(
        rf"{NUM}(?:'s\b|{GAP}(?:{VERBS})\b)|\b(?:as|by|of|per|see|under){GAP}{NUM}"
    ),
    # a section number with one of its numbered parts: "4.3 check 15", "8.5 step 6", "2.12, item 9", "7.9 (item 5)"
    "with_its_part": re.compile(
        rf"{NUM},?{GAP}\(?(?:checks?|steps?|items?|rows?|rules?|tables?){GAP}\d+"
    ),
    # a section number that heads a line or a comment: "# 8.5: the first apply over an older package"
    "as_a_label": re.compile(rf"(?m)^[ \t]*{NUM}:"),
}

# What a section number may be instead, by name (the brief's Keep list): a BARE match each of whose section numbers
# lies inside a match of one of these is no reference.
ALLOWANCES = {
    # a numbered section of the package's own docs (review_pipeline.md, the review book, a team file's What to Check)
    "own_doc_section": re.compile(
        rf"§[ \t]*\d+(?:\.\d+)*"
        rf"|\b[Ss]ections?{GAP}\d+(?:\.\d+)*(?:{GAP}(?:to|and|or){GAP}\d+(?:\.\d+)*)?"
    ),
    # a release: "Python 3.10", "Claude Opus 5.5's", "this Python has it (3.12, ...)", "3.12 or newer", "before 3.13"
    "version": re.compile(
        rf"\b(?:{PROGRAMS})\b[^\n()]{{0,12}}?\(?[ \t]*{NUM}"
        rf"|{NUM}{GAP}(?:or{GAP}(?:newer|later|older)|on)\b"
        rf"|\b(?:before|since|than|from|until){GAP}{NUM}"
    ),
    # a quantity with its unit, a number that is no section either: "a RAM disk of 1.5 GiB", "as 1.5 times"
    "quantity": re.compile(rf"{NUM}[ \t]*(?:{UNITS})(?!\w)"),
    # a section of a public standard, named with it: "(RFC 8439, 2.3)", "RFC 8032, 7.1, TEST 1"
    "standard_section": re.compile(
        rf"\bRFC[ \t]*\d+,?{GAP}(?:[Ss]ection{GAP}|§)?{NUM}"
    ),
}

# The brief's Keep list, by name, each with an example (a file's name and its text) that no form finds.
KEEP = {
    # a numbered section of the package's own docs: its headings, its "section N.N", the review book's §N
    "own_doc_section": (
        "x.md",
        'docs/review_pipeline.md, "3.4 The review driver" (section 5.3); the review book\'s §11, '
        "its section 5 and §6.2's status words; What to Check §7",
    ),
    # a lessons id, as agentic/skills/rocke-monitor/lessons.md holds them: no form has a letter and a dash
    "lessons_id": ("x.md", "(L-541) and L-12's rule"),
    # a version number: no form takes one with a "+" or three parts, and ALLOWANCES' version holds the others
    "version": (
        "x.md",
        "git 2.38+, Python 3.10+, cmake 3.16, sparse worktrees need 2.25, gh 2.46.0, "
        "Claude Opus 5.5's prices, a Python of 3.12 or newer (3.6 or newer), "
        "this Python has it (3.12, and later releases)",
    ),
    # an IP address: a section number is never a piece of a longer number (NUM)
    "ip_address": ("x.md", "it listens on 127.0.0.1 (0.0.0.0: all of them)"),
    # a Debian package's version and its epoch: a list's colon has a blank after it
    "epoch_version": (
        "x.py",
        '"Setting up libfoo:amd64 (1:2.3) ... from setup.py (0.4.1)"\n',
    ),
    # a float in code: a Python file is read by its comments and strings (`prose`), and a call's parentheses are code
    "float_in_code": (
        "x.py",
        "time.sleep(1.5)\nlimits = (1.5, 2.25)\nRATE = {'a': 0.3}\n"
        "SCRIPT = 'time.sleep(1.5)'\n",
    ),
    # a float in data: a JSON file is read by its strings alone (`strings`)
    "float_in_data": (
        "x.json",
        '{"review_estimate_gib": 1.5, "sizes": [1, 2.5, 3], "a": "1.5 GiB"}\n',
    ),
    # JSON in a doc: a quoted key's value is data, not a list's section number
    "json_in_a_doc": (
        "x.md",
        '`{"sizes": {"small": 1.5, "large": 4.5}, '
        '"rate": {"input": 4.0, "cache_read": 0.2}}`',
    ),
    # a row's note that names the package's files: a digit in a file's name is no section's number
    "file_note": (
        "x.json",
        '{"design": "rkmon_lib/removed_keys.py (PACKAGE_SET), '
        'rkmon_lib/from_layout1.py"}\n',
    ),
    # docs/design.md, the package's own short design: "the design" names it, and its sections have no numbers
    "own_design": (
        "x.md",
        "The design, and why: docs/design.md. The design itself breaks no result.",
    ),
    # a split review's parts and the lessons extract's: lowercase, unlike the design's
    "split_review_part": (
        "x.md",
        "`lessons_for.py verifier --part 2`, for part 2 (`parts/part2.md`)",
    ),
    # a quantity, as ALLOWANCES' quantity holds it, and a fraction: "0.25" is never a section's number
    "quantity": (
        "x.md",
        "a RAM disk of 1.5 GiB, grown by 2.5 GiB as 1.5 times the reviews "
        "(0.25: a quarter)",
    ),
    # a package's version such as 1.0 or 2.0: a section's number never ends in 0
    "package_version": (
        "x.md",
        "the package folder (1.0, in use) and a newer package, ~/new (2.0)",
    ),
    # a section of a public standard, as ALLOWANCES' standard_section holds it
    "standard_section": (
        "x.py",
        "# 64 bytes of key stream (RFC 8439, 2.3)\n# RFC 7748, 5.2: the first vector\n",
    ),
}

# Each form the brief names, and each form above by its own example: each is found (a file's name and its text).
FOUND = (
    ("x.md", "design 6.7"),
    ("x.md", "a note (design 2.8)."),
    ("x.md", "a bare (6.6) or (8.1), (2.6, 8.1), and design Part 0, decision 5"),
    ("x.md", "the file as 4.1 writes it"),
    ("x.md", "8.3's round trip"),
    ("x.md", "Part 5 merges into the file's"),
    ("x.md", "the design's package"),
    ("x.md", "the key's home, design 5.2: none"),
    ("x.md", "(6.1: it changes nothing), (states.row, 6.5: PASS while it runs)"),
    (
        "x.md",
        "as 6.6 says; 4.3 check 15's ports; design 2.10, 8.5 step 6; 2.12, item 9; "
        "7.9 (item 5)",
    ),
    (
        "x.md",
        "steps 1 to 9 of 6.5; the caches (item 9: develop's results), "
        "(Wrapper.runs_older_code: 8.5)",
    ),
    ("x.py", "# 8.5: the first apply over an older package\n"),
    ("x.py", "# a note that wraps (design\n# 2.8) onto the next line\n"),
    ("x.py", "MESSAGE = ('the file, as the design '\n           '6.7 says')\n"),
    ("x.py", "ROW = Row(why='intended', design='8.3')\n"),
    ("x.json", '{"why": "intended", "design": "5.1, 8.1"}\n'),
    (
        "x.sh",
        "#!/bin/bash\n# a comment that wraps (design\n# 2.8) onto the next line\n"
        "true\n",
    ),
)


def prose(text: str) -> str:
    """TEXT, a Python file's, as a person reads it: its comments and the text of its strings at their places, and
    blanks for the rest, a comment's # and a string's quotes too, so that a sentence goes on across a wrapped comment
    or two strings side by side. The same lines and columns as TEXT; TEXT itself when tokenize cannot read it.
    """
    starts = [0]
    for line in text.split("\n"):
        starts.append(starts[-1] + len(line) + 1)
    kinds = {tokenize.COMMENT, tokenize.STRING}
    kinds |= {
        getattr(tokenize, k)
        for k in ("FSTRING_MIDDLE", "TSTRING_MIDDLE")
        if hasattr(tokenize, k)
    }
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(text).readline))
    except (tokenize.TokenError, SyntaxError):
        return text
    out = [c if c == "\n" else " " for c in text]
    for tok in tokens:
        if tok.type not in kinds:
            continue
        a = starts[tok.start[0] - 1] + tok.start[1]
        b = starts[tok.end[0] - 1] + tok.end[1]
        out[a:b] = text[a:b]
        if tok.type == tokenize.COMMENT:
            out[a] = " "
        elif tok.type == tokenize.STRING:
            q = re.match(r"[A-Za-z]*(\"\"\"|'''|\"|')", text[a:b])
            out[a : a + q.end()] = " " * q.end()
            out[b - len(q.group(1)) : b] = " " * len(q.group(1))
    return "".join(out)


def strings(text: str) -> str:
    """TEXT, a JSON file's, by its strings alone, at their places, and blanks for the rest (their quotes too): a
    number in data is no reference."""
    out = [c if c == "\n" else " " for c in text]
    for m in re.finditer(r'"((?:[^"\\\n]|\\.)*)"', text):
        out[m.start(1) : m.end(1)] = m.group(1)
    return "".join(out)


def findings_in(name: str, text: str, allowances=None) -> list:
    """(line, forms) for each line of TEXT, the file NAME's, that refers to the design, in order: the line a match
    starts on, and the names of the forms that match there. A Python file (by NAME, or by its #! line) is read in
    `prose`, a JSON file by its `strings`; any other file whole, a line's leading # blanked so a wrapped comment reads
    on. ALLOWANCES (by default the module's) hold the numbers of BARE's matches."""
    first = text.split("\n", 1)[0]
    if name.endswith(".py") or first.startswith("#!") and "python" in first:
        read = prose(text)
    elif name.endswith(".json"):
        read = strings(text)
    else:
        read = re.sub(
            r"(?m)^([ \t]*)(#+)", lambda m: m.group(1) + " " * len(m.group(2)), text
        )
    starts = [0]
    for line in read.split("\n"):
        starts.append(starts[-1] + len(line) + 1)
    found = {}
    for form, rx in NAMED.items():
        for m in rx.finditer(text if form == "design_note" else read):
            found.setdefault(bisect.bisect_right(starts, m.start()), set()).add(form)
    allowances = ALLOWANCES if allowances is None else allowances
    held = [m.span() for rx in allowances.values() for m in rx.finditer(read)]
    for form, rx in BARE.items():
        for m in rx.finditer(read):
            spans = [n.span() for n in NUMBER.finditer(read, m.start(), m.end())]
            if any(not any(a <= s and e <= b for a, b in held) for s, e in spans):
                line = bisect.bisect_right(starts, m.start())
                found.setdefault(line, set()).add(form)
    return [(n, sorted(forms)) for n, forms in sorted(found.items())]


def findings(root: pathlib.Path, files) -> list:
    """`path:line: forms: text` for each line of FILES (in the folder ROOT) that refers to the design, this file's
    aside; a file that is no UTF-8 text is none of them."""
    out = []
    for f in sorted(files):
        if f.resolve() == OWN:
            continue
        try:
            text = f.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        lines = text.split("\n")
        rel = f.relative_to(root).as_posix()
        for n, forms in findings_in(rel, text):
            out.append(f"{rel}:{n}: {', '.join(forms)}: {lines[n - 1].strip()[:160]}")
    return out


class TestDesignRefs(unittest.TestCase):
    def test_package(self):
        """No package file refers to the design's sections or parts: each line that does, by file and line."""
        found = findings(ROOT, test_headers.package_files())
        if found:
            self.fail(
                f"{len(found)} line(s) refer to the sections or parts of the settings "
                "design, which the package does not hold; say the thing itself, or "
                "name where it lives in the package:\n" + "\n".join(found)
            )

    def test_found(self):
        """Each form the brief names, and each form's own example, is found."""
        for name, text in FOUND:
            with self.subTest(text=text):
                self.assertTrue(findings_in(name, text))

    def test_each_form(self):
        """Each form of NAMED and BARE finds one of the examples."""
        found = set()
        for name, text in FOUND:
            for _line, forms in findings_in(name, text):
                found.update(forms)
        for form in [*NAMED, *BARE]:
            with self.subTest(form=form):
                self.assertIn(form, found)

    def test_keep(self):
        """The brief's Keep list: no form finds its examples."""
        for what, (name, text) in KEEP.items():
            with self.subTest(what=what):
                self.assertEqual(findings_in(name, text), [])

    def test_allowances(self):
        """Each allowance holds a number of KEEP's examples that a form finds without it."""
        for allowance in ALLOWANCES:
            others = {k: rx for k, rx in ALLOWANCES.items() if k != allowance}
            with self.subTest(allowance=allowance):
                self.assertTrue(
                    any(findings_in(name, text, others) for name, text in KEEP.values())
                )


if __name__ == "__main__":
    unittest.main()
