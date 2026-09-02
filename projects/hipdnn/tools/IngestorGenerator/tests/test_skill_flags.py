# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Every CLI flag the hipdnn-ingestor-engine skill cites must exist.

`test_skill_paths.py` makes "the paths resolve" permanent. This makes "the
commands run" permanent, and it is the check that was missing when `extend.md`
shipped `variant_reachability.py --score` (never a flag; the real ones are
`--score-field` / `--score-prefer`) and when RUNBOOK step 4a-3 documented a
`dispatch_parity.py` capability that did not exist until `--knobs` was written.

A stale flag fails as `argparse` exit 2 plus a usage string, which reads like
operator error rather than a lying document -- so it gets worked around, and the
document stays wrong. `tests/skill_flags.py` holds the conservative extractor;
this file is the assertions plus the fixtures proving the extractor is not the
kind of check that only ever sees good data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.skill_flags import discover_tools, extract_cited_flags, tool_flags

_REPO_ROOT = Path(__file__).resolve().parents[5]
_GEN = _REPO_ROOT / "projects/hipdnn/tools/IngestorGenerator"
_SKILL_DIR = _REPO_ROOT / "projects/hipdnn/tools/ai/skills/hipdnn-ingestor-engine"


@pytest.fixture(scope="module")
def tools():
    """`{basename: path}` for the pipeline tools plus `generate.py`."""
    return discover_tools(_GEN / "tools", _GEN / "generate.py")


@pytest.fixture(scope="module")
def declared(tools):
    return {name: tool_flags(path) for name, path in tools.items()}


@pytest.fixture(scope="module")
def cited(tools):
    found = []
    for markdown in sorted(_SKILL_DIR.glob("*.md")):
        found += extract_cited_flags(markdown.read_text(), markdown.name, set(tools))
    return found


def test_the_tools_are_discovered(tools):
    """A rename that empties the tool set would make every assertion below
    vacuous -- they iterate over citations bound to KNOWN tools."""
    assert "dispatch_parity.py" in tools
    assert "generate.py" in tools
    assert len(tools) >= 8, f"only found {sorted(tools)}"


def test_every_tool_declares_at_least_one_flag(declared):
    """Guards the `add_argument` pattern itself: a tool that switched to
    single-quoted or f-string flag names would silently declare nothing, and
    every citation against it would then read as dangling."""
    empty = sorted(name for name, flags in declared.items() if not flags)
    assert not empty, (
        f"no flags extracted from {empty} -- tests/skill_flags.py's "
        f"_ADD_ARGUMENT_RE no longer matches how these tools declare arguments"
    )


class TestEveryCitedFlagExists:
    def test_the_skill_cites_flags_at_all(self, cited):
        """The positive control for the whole file: zero citations means the
        extractor regressed, not that the skill stopped giving commands."""
        assert len(cited) >= 20, (
            f"only {len(cited)} tool-bound flag citations found across the "
            f"skill -- the extractor almost certainly regressed"
        )

    def test_no_cited_flag_is_undeclared(self, cited, declared):
        dangling = [c for c in cited if c.flag not in declared[c.tool]]
        assert not dangling, "flag(s) the skill cites that do not exist:\n" + "\n".join(
            f"  {c.file}:~{c.line}: `{c.tool} {c.flag}` "
            f"(declared: {', '.join(sorted(declared[c.tool])) or 'none'})\n"
            f"      in: {c.command}"
            for c in dangling
        )


class TestExtractorCatchesARealStaleFlag:
    """Negative tests, using the literal defect this file was written for.
    An assertion that only ever sees valid data is decoration."""

    def test_a_stale_flag_is_flagged(self, tools, declared):
        fixture = (
            "```bash\n"
            "$GEN/.venv/bin/python $GEN/tools/variant_reachability.py \\\n"
            "    --kdp x.kdp.json --shapes $SHAPES --score maxwaves\n"
            "```\n"
        )
        cited = extract_cited_flags(fixture, "fixture.md", set(tools))
        stale = [c for c in cited if c.flag not in declared[c.tool]]
        assert [c.flag for c in stale] == [
            "--score"
        ], f"expected --score to be caught as stale, got {[c.flag for c in cited]}"

    def test_the_real_flags_in_the_same_shape_are_not_flagged(self, tools, declared):
        """Companion control: the corrected command must come back clean, so the
        failure above is attributable to `--score` and not to the whole line."""
        fixture = (
            "```bash\n"
            "$GEN/.venv/bin/python $GEN/tools/variant_reachability.py \\\n"
            "    --kdp x.kdp.json --shapes $SHAPES --score-field waves \\\n"
            "    --score-prefer max\n"
            "```\n"
        )
        cited = extract_cited_flags(fixture, "fixture.md", set(tools))
        assert cited, "extractor found nothing in a command it should parse"
        assert not [c for c in cited if c.flag not in declared[c.tool]]

    def test_a_backslash_continuation_still_binds_to_its_tool(self, tools):
        """Every real command in the skill wraps. A flag on the second physical
        line must still be attributed, or the check misses most of the file."""
        fixture = (
            "```bash\n"
            "$GEN/.venv/bin/python $GEN/tools/dispatch_parity.py \\\n"
            "    --profile p.yaml \\\n"
            "    --shapes s.json\n"
            "```\n"
        )
        cited = extract_cited_flags(fixture, "fixture.md", set(tools))
        assert {c.flag for c in cited} == {"--profile", "--shapes"}
        assert {c.tool for c in cited} == {"dispatch_parity.py"}


class TestExtractionIsConservative:
    """A flag token means nothing on its own. Binding it to the wrong binary
    manufactures failures on a correct document, and a check that cries wolf
    gets deleted by the next person."""

    @pytest.mark.parametrize(
        "fixture",
        [
            # Another binary's flags, in a block that names no python tool.
            "```bash\n"
            "$BUILD/bin/hipdnn_validate_descriptors $TREE "
            "--expect-engine $ENGINE --json\n```\n",
            # Build systems and test runners.
            "```bash\ncmake -B $BUILD --preset default -DGPU_TARGETS=gfx950\n```\n",
            "```bash\nctest --test-dir $BUILD -L '^quick$' -N\n```\n",
            # A comment inside a block is documentation, not a command.
            "```bash\n# run it with --score later, once that exists\n```\n",
        ],
    )
    def test_flags_not_bound_to_a_python_tool_are_ignored(self, fixture, tools):
        assert extract_cited_flags(fixture, "fixture.md", set(tools)) == []

    def test_prose_flags_are_not_bound_to_a_nearby_tool(self, tools):
        """SKILL.md's CLI summary names `generate.py` and, two lines later,
        `hipdnn_validate_descriptors ... [--json]`. Proximity would bind --json
        to generate.py and fail a document that is correct."""
        fixture = (
            "- A's frozen CLI: `generate.py --config <yaml> --output-dir <dir>`.\n"
            "- B's frozen CLI and `--json` shape:\n"
            "  `hipdnn_validate_descriptors <root>... [--json]`,\n"
        )
        assert extract_cited_flags(fixture, "fixture.md", set(tools)) == []

    def test_the_last_tool_on_a_line_owns_the_flags(self, tools):
        """A piped command binds each flag to the tool it follows."""
        fixture = (
            "```bash\n"
            "python3 $GEN/tools/mine_shapes.py --arch gfx950 | "
            "python3 $GEN/tools/dispatch_parity.py --profile p.yaml\n"
            "```\n"
        )
        cited = extract_cited_flags(fixture, "fixture.md", set(tools))
        assert {(c.tool, c.flag) for c in cited} == {
            ("dispatch_parity.py", "--profile")
        }
