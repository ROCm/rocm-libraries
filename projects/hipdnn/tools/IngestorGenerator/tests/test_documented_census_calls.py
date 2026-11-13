# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Every documented ``hkp_register_census_tests`` call form pins EXPECTED_CASES.

Calls with arguments under ``DOCUMENTED_ROOTS`` must include ``EXPECTED_CASES``.
Without it, ``dnn-providers/hip-kernel-provider/src/tests/main.cpp`` skips case-name
comparison, the execution guard checks only registered cases, and ``HkpPackaging.cmake``
omits the ``-control-unregistered-case`` watcher. Discover call sites rather than
listing files.
"""

import re
from pathlib import Path

# ``tests/`` -> ``IngestorGenerator/`` -> ``tools/`` -> ``hipdnn/`` -> ``projects/``
# -> the repository root.
REPO_ROOT = Path(__file__).resolve().parents[5]

# The trees that publish a call form an author copies: the generator's own pages and
# templates, and the packaging module that defines the function. Two named roots rather
# than a repo-wide walk, which also sweeps generated build output -- CMake restates the
# call inside a configured ``CTestTestfile.cmake``. Within each root discovery is
# exhaustive.
DOCUMENTED_ROOTS = (
    REPO_ROOT / "projects" / "hipdnn" / "tools",
    REPO_ROOT / "dnn-providers" / "hip-kernel-provider" / "descriptor-packaging",
)

CALL = "hkp_register_census_tests("
PIN = "EXPECTED_CASES"

# Surfaces an author copies a call form out of: markdown prose, the Jinja templates
# spliced into a real CMakeLists.txt, and the ``.cmake`` module defining the function.
# Classifying a comment as documented is the safe direction.
DOCUMENTED_SUFFIXES = {".md", ".j2", ".cmake"}

# Python under these roots is the generator's and packager's own source and tests,
# where the call name appears as a matcher or substring (see test_fragment_contracts.py
# and test_packaged_dialect.py); ``test_unclassified_surfaces`` keeps that honest.
CODE_SUFFIXES = {".py"}

# Caches and history: a copy of the call nobody authors is not a form.
GENERATED_DIRS = {"__pycache__", ".git", ".pytest_cache"}

# Elision marks a documented signature uses to stand in for omitted arguments.
ELISION = "…."

# A runaway balance scan means the form is not delimited the way prose implies.
MAX_SPAN_LINES = 60


def _source_files():
    """Every hand-written file under the roots, unfiltered at the walk so a call form in
    an unclassified surface is still SEEN and can be reported."""
    for root in DOCUMENTED_ROOTS:
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            if GENERATED_DIRS.intersection(path.parts):
                continue
            yield path


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


class Occurrence:
    """One ``hkp_register_census_tests(`` in a documented surface."""

    def __init__(self, path: Path, text: str, start: int):
        self.path = path
        self.line = text.count("\n", 0, start) + 1
        self.rel = path.relative_to(REPO_ROOT).as_posix()

        open_paren = start + len(CALL) - 1
        self.args, self.span = _balanced_args(text, open_paren)

    @property
    def closed(self) -> bool:
        return self.args is not None

    @property
    def shows_arguments(self) -> bool:
        """THE RULE that separates a call FORM from a MENTION of the function.

        A form "shows arguments" when the text between its parentheses contains anything
        other than whitespace and elision marks. So::

            hkp_register_census_tests()                  <- names the function
            hkp_register_census_tests(...)               <- names the function
            hkp_register_census_tests(TARGET ... )       <- SHOWS ARGUMENTS
        """
        return bool(self.args.strip().strip(ELISION).strip())

    @property
    def pinned(self) -> bool:
        return PIN in self.args

    def __repr__(self) -> str:
        return f"{self.rel}:{self.line}"


def _balanced_args(text: str, open_paren: int):
    """Text between ``open_paren`` and its matching ``)``, plus the whole span. Balanced
    because the forms take three shapes -- a backticked signature, a ```cmake``` block,
    and a form wrapped across comment lines -- and only the parentheses are common to
    all."""
    depth = 0
    for index in range(open_paren, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                span = text[open_paren : index + 1]
                if span.count("\n") > MAX_SPAN_LINES:
                    return None, span
                return text[open_paren + 1 : index], span
    return None, text[open_paren:]


def _occurrence_from(text: str) -> Occurrence:
    """A call form written out here rather than discovered; the path is this file only
    so ``Occurrence`` has a location to report."""
    return Occurrence(Path(__file__).resolve(), text, text.index(CALL))


def _occurrences():
    found = []
    for path in _source_files():
        if path.suffix not in DOCUMENTED_SUFFIXES:
            continue
        text = _read(path)
        for match in re.finditer(re.escape(CALL), text):
            found.append(Occurrence(path, text, match.start()))
    return found


class TestEveryDocumentedCensusCallPinsItsCases:

    def test_the_walk_finds_the_documented_forms(self):
        """Anti-vacuity: if the root stops resolving, the guard below would go green
        over an empty set."""
        for root in DOCUMENTED_ROOTS:
            assert root.is_dir(), f"documented root did not resolve: {root}"
        occurrences = _occurrences()
        for root in DOCUMENTED_ROOTS:
            assert any(
                root in path.parents for path in (found.path for found in occurrences)
            ), (
                f"no {CALL!r} occurrence found anywhere under {root} -- either the "
                "documented call forms moved out of the classified file types, or "
                "this root is wrong. Either way this suite has stopped guarding that "
                "tree; fix the root rather than dropping it."
            )

    def test_every_form_that_shows_arguments_also_shows_expected_cases(self):
        unpinned = [
            occurrence
            for occurrence in _occurrences()
            if occurrence.closed
            and occurrence.shows_arguments
            and not occurrence.pinned
        ]
        assert not unpinned, "\n".join(
            [
                "documented hkp_register_census_tests() forms show arguments but omit "
                f"{PIN}:",
                "",
                *(
                    f"  {occurrence.rel}:{occurrence.line}\n"
                    f"      {' '.join(occurrence.span.split())[:140]}"
                    for occurrence in unpinned
                ),
                "",
                f"Add the {PIN} keyword to each form above. It is not optional "
                "documentation polish:",
                "  - without it hip-kernel-provider's src/tests/main.cpp returns "
                "before any case-name comparison happens;",
                "  - the execution guard then walks only the cases the suite itself "
                "registered, so DELETING a case deletes its own obligation and the "
                "census still reports complete;",
                "  - and HkpPackaging.cmake drops the -control-unregistered-case "
                "watcher that proves the comparison runs at all.",
                "",
                "An author copying an unpinned form ships a census that cannot "
                "notice a suite that shrank. If a form genuinely must elide its "
                "arguments, write it with empty or fully-elided parentheses -- "
                "hkp_register_census_tests() -- which this test treats as naming "
                "the function rather than showing a form to copy.",
            ]
        )

    def test_every_form_is_delimited(self):
        """A form whose parentheses never balance would be dropped by the
        ``occurrence.closed`` filter above and escape the pin requirement."""
        unclosed = [
            occurrence for occurrence in _occurrences() if not occurrence.closed
        ]
        assert not unclosed, "\n".join(
            [
                "hkp_register_census_tests( occurrences whose closing parenthesis "
                f"was never found within {MAX_SPAN_LINES} lines:",
                *(f"  {occurrence.rel}:{occurrence.line}" for occurrence in unclosed),
                "",
                "These are skipped by the EXPECTED_CASES check because they cannot be "
                "delimited, so they are reported here instead of passing quietly. "
                "Close the form, or reduce it to a bare "
                "hkp_register_census_tests() mention.",
            ]
        )

    def test_the_argument_rule_classifies_written_out_forms(self):
        """The walk only reports how the rule classified the tree, since every property
        derived from ``args`` agrees with itself by construction; these forms carry an
        expectation stated independently of the rule."""
        cases = (
            # text, shows_arguments, pinned
            (f"{CALL})", False, False),
            (f"{CALL}{ELISION})", False, False),
            (f"{CALL}\n    {ELISION}\n)", False, False),
            (f"{CALL}TARGET t {PIN} 4)", True, True),
            (f"{CALL}TARGET t {ELISION})", True, False),
            (f"{CALL}TARGET t {PIN} {ELISION})", True, True),
            (f"{CALL}TARGET t COST_HINT(4) {PIN} 4)", True, True),
            (f"{CALL}\n##  TARGET t\n##  {PIN} 4)", True, True),
            (f"{CALL}\n##  TARGET t\n##  {ELISION})", True, False),
        )
        for text, shows_arguments, pinned in cases:
            occurrence = _occurrence_from(text)
            assert occurrence.closed, f"{text!r} was not delimited"
            assert occurrence.shows_arguments is shows_arguments, (
                f"{text!r} must{'' if shows_arguments else ' not'} count as showing "
                "arguments"
            )
            assert (
                occurrence.pinned is pinned
            ), f"{text!r} must{'' if pinned else ' not'} count as pinned"

    def test_unclassified_surfaces(self):
        """``DOCUMENTED_SUFFIXES`` is a whitelist, the same silent under-coverage as a
        hard-coded file list, so a call form in an unclassified file type is
        reported."""
        stray = sorted(
            path.relative_to(REPO_ROOT).as_posix()
            for path in _source_files()
            if path.suffix not in DOCUMENTED_SUFFIXES
            and path.suffix not in CODE_SUFFIXES
            and CALL in _read(path)
        )
        assert not stray, "\n".join(
            [
                "hkp_register_census_tests( appears in files whose kind this suite "
                "has not classified:",
                *(f"  {name}" for name in stray),
                "",
                "Decide which it is and record the decision: add the suffix to "
                "DOCUMENTED_SUFFIXES if an author reads and copies a call form out "
                "of it, or to CODE_SUFFIXES if the name only appears there as a "
                "matcher or assertion substring.",
            ]
        )
