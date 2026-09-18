# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Every documented ``hkp_register_census_tests`` call form pins EXPECTED_CASES.

WHAT THIS PINS: each occurrence of ``hkp_register_census_tests(`` in the prose,
templates and CMake commentary under the roots in ``DOCUMENTED_ROOTS`` that SHOWS
ARGUMENTS must also carry the ``EXPECTED_CASES`` keyword. Deleting ``EXPECTED_CASES``
from any documented form -- the literal CMake blocks, the elided one-line signatures,
or the Jinja template sites -- must turn this suite RED.

WHY IT IS WORTH A TEST: the parameter is load-bearing three times over, and an
unpinned call is green in every one of them. Without it
``dnn-providers/hip-kernel-provider/src/tests/main.cpp`` returns before any
case-name comparison runs; the execution guard then walks only the suite's own
registered inventory, so a deleted case quietly takes its own obligation with it;
and ``HkpPackaging.cmake`` drops the ``-control-unregistered-case`` watcher that
proves the comparison is live at all. An author who copies an unpinned form out of
the documentation ships a census that stays green after losing a case.

Prose is not covered by the compiler or by any other gate in this tree, so the
correction that put ``EXPECTED_CASES`` into every documented form can drift straight
back out. This is the only thing holding it.

The sites are DISCOVERED, never listed: a hard-coded file list stops covering a page
someone adds later, which is exactly the class of defect being guarded against.
"""

import re
from pathlib import Path

# ``tests/`` -> ``IngestorGenerator/`` -> ``tools/`` -> ``hipdnn/`` -> ``projects/``
# -> the repository root.
REPO_ROOT = Path(__file__).resolve().parents[5]

# The trees that publish a call form an author copies: the generator's own pages and
# templates, and the packaging module that defines the function and documents it.
#
# Two named roots rather than one walk of the repository root. A repo-wide walk also
# sweeps generated build output -- CMake restates the call inside the
# ``_BACKTRACE_TRIPLES`` of a configured ``CTestTestfile.cmake`` -- and a form nobody
# edits by hand is not a form anyone copies, so policing it would only make the guard
# depend on whether a build tree happens to be present. Within each root discovery is
# exhaustive; the roots themselves are the only thing enumerated by hand.
DOCUMENTED_ROOTS = (
    REPO_ROOT / "projects" / "hipdnn" / "tools",
    REPO_ROOT / "dnn-providers" / "hip-kernel-provider" / "descriptor-packaging",
)

CALL = "hkp_register_census_tests("
PIN = "EXPECTED_CASES"

# Surfaces an author reads and copies a call form out of: markdown prose, the Jinja
# templates whose rendered fragments get spliced into a real CMakeLists.txt, and the
# ``.cmake`` module that defines the function -- its comments explain the call to the
# very authors who write one, so a form shown there is as copyable as one in a README.
# Classifying it as documented is also the safe direction: an illustrative form added
# to a comment is then held to the pin instead of quietly escaping it.
DOCUMENTED_SUFFIXES = {".md", ".j2", ".cmake"}

# Python under these roots is the generator's and packager's own source and test
# suites, where the call name legitimately appears as a matcher or a substring (see
# ``test_fragment_contracts.py`` and ``test_packaged_dialect.py``). Those are code
# about the call, not a form anyone copies, so they are excluded deliberately --
# not overlooked. ``test_unclassified_surfaces`` below keeps that decision honest.
CODE_SUFFIXES = {".py"}

# Caches and history, none of them edited by hand. Skipped for the same reason the
# roots stop short of a build tree: a copy of the call nobody authors is not a form.
GENERATED_DIRS = {"__pycache__", ".git", ".pytest_cache"}

# Elision marks a documented signature uses to stand in for omitted arguments.
ELISION = "…."

# A runaway balance scan means the form is not delimited the way prose implies;
# no real call form in this tree is anywhere near this long.
MAX_SPAN_LINES = 60


def _source_files():
    """Every hand-written file under the roots, regardless of extension.

    Deliberately unfiltered at the walk: the extension split happens afterwards so
    that a call form appearing in a surface nobody classified is still SEEN, and
    can be reported rather than silently skipped.
    """
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

        A form "shows arguments" when the text between its parentheses contains
        anything other than whitespace and elision marks. So::

            hkp_register_census_tests()                  <- names the function
            hkp_register_census_tests(...)               <- names the function
            hkp_register_census_tests(TARGET ... )       <- SHOWS ARGUMENTS

        Several sites are the first kind: prose reading "one
        ``hkp_register_census_tests()`` call per packed target", which is about the
        call's cardinality, and the packaging module's comments naming the function
        as the reader of a value they compute. Both show no arguments at all, and
        requiring a pin there would be requiring prose to stop being prose. Anything
        that names even one argument is holding itself out as a form to copy, and
        must be complete.
        """
        return bool(self.args.strip().strip(ELISION).strip())

    @property
    def pinned(self) -> bool:
        return PIN in self.args

    def __repr__(self) -> str:
        return f"{self.rel}:{self.line}"


def _balanced_args(text: str, open_paren: int):
    """Text between ``open_paren`` and its matching ``)``, plus the whole span.

    Balanced rather than line- or regex-delimited because the forms have three
    different shapes: a one-line signature inside backticks, a multi-line literal
    ```cmake``` block closing on a line of its own, and a form wrapped across
    ``##``- or `` * ``-prefixed comment lines in a template. Only the parentheses
    are common to all three.
    """
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
    """The documented forms, walked out of the tree rather than listed."""

    def test_the_walk_finds_the_documented_forms(self):
        """Anti-vacuity: a guard that discovers nothing passes for free.

        If the root ever stops resolving, or the pages move out from under this
        walk, the real assertion below would go green while checking an empty set.
        """
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
        """The guard itself.

        Removing ``EXPECTED_CASES`` from any documented form should turn this red.
        """
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
        """A form whose parentheses never balance was not really parsed.

        Without this, an unclosed form would be dropped from the checked set by the
        ``occurrence.closed`` filter above and silently escape the pin requirement.
        """
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

    def test_bare_mentions_are_recognised_and_exempt(self):
        """The control: the two classes really are distinguished.

        If this found nothing, ``shows_arguments`` could be a constant ``True`` and
        the guard above would still be green -- passing for the wrong reason. The
        exempt sites are prose about how MANY calls to make, not forms to copy.
        """
        mentions = [
            occurrence
            for occurrence in _occurrences()
            if occurrence.closed and not occurrence.shows_arguments
        ]
        assert mentions, (
            "no bare hkp_register_census_tests() mention found, so nothing proves "
            "this suite distinguishes a form that shows arguments from prose that "
            "merely names the function. If the last bare mention was genuinely "
            "rewritten, delete this test; do not weaken the rule to satisfy it."
        )
        for occurrence in mentions:
            assert not occurrence.pinned, (
                f"{occurrence.rel}:{occurrence.line} was classified as a bare mention "
                f"yet contains {PIN} -- the argument-detection rule is wrong."
            )

    def test_unclassified_surfaces(self):
        """The extension split cannot go stale unnoticed.

        ``DOCUMENTED_SUFFIXES`` is a whitelist, and a whitelist is the same silent
        under-coverage as a hard-coded file list one level up. If the call form
        turns up in a file type nobody has classified, say so rather than skipping
        it.
        """
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
