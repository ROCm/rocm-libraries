# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.tests.lane_mentions finds runner lane names, hand-written lane
declarations and skip reasons, and runner hardware needs, the same in worker
processes as in this one."""

from __future__ import annotations

from pathlib import Path

import pytest
from rocke.tests.host_cpus import available_cpus
from rocke.tests.lane_mentions import (
    LaneMention,
    LaneMentionKind,
    lane_mentions,
    lane_mentions_in_source,
)

needs_two_cpus = pytest.mark.skipif(
    available_cpus() < 2, reason="a pool of worker processes needs two CPUs"
)

DECLARED = "waits for runner-hardware: needs a gfx950 GPU"
UNKNOWN = "waits for no-such-lane: x"
SOURCES = {
    "declared.py": f"reason = {DECLARED!r}\n",
    "clean.py": 'reason = "needs a gfx950 GPU"\n',
    "named.py": "from rocke.tests.lanes import RUNNER_HARDWARE\nlane = RUNNER_HARDWARE\n",
    "unknown.py": f"reason = {UNKNOWN!r}\n",
}


def _write_sources(root: Path) -> list[str]:
    for name, source in SOURCES.items():
        (root / name).write_text(source, encoding="utf-8")
    return list(SOURCES)


def test_a_source_mentions_a_runner_lane_by_name_and_a_lane_by_declaration():
    assert lane_mentions_in_source(SOURCES["named.py"], "named.py") == [
        LaneMention("named.py", 2, LaneMentionKind.RUNNER_LANE_NAME, "RUNNER_HARDWARE")
    ]
    assert lane_mentions_in_source(SOURCES["declared.py"], "declared.py") == [
        LaneMention(
            "declared.py", 1, LaneMentionKind.HAND_WRITTEN_DECLARATION, DECLARED
        )
    ]
    assert lane_mentions_in_source(SOURCES["unknown.py"], "unknown.py") == [
        LaneMention("unknown.py", 1, LaneMentionKind.HAND_WRITTEN_DECLARATION, UNKNOWN)
    ]
    assert lane_mentions_in_source(SOURCES["clean.py"], "clean.py") == []


def _lines_of_kind(source: str, kind: LaneMentionKind) -> list[int]:
    return [
        mention.line
        for mention in lane_mentions_in_source(source, "t.py")
        if mention.kind is kind
    ]


def test_a_hand_written_lane_reason_is_found():
    source = (
        "import pytest, unittest\n"
        "pytest.skip('waits for a gfx950 runner in CI')\n"
        "pytest.mark.skipif(True, reason='no gfx950 runner')\n"
        "unittest.skipUnless(False, f'waiting on {x} in CI')\n"
        "pytest.skip('waits for ticket: torch')\n"
        "pytest.skip(msg='CI has no gfx1250')\n"
        "pytest.skip(skip_until_reason(TICKET, reason='waits for torch'))\n"
        "pytest.skip('needs ctest on PATH')\n"
        "pytest.mark.skipif(True, reason='the lane map is gfx942-only')\n"
        "pytest.importorskip('yaml', reason='needs PyYAML')\n"
    )
    assert _lines_of_kind(source, LaneMentionKind.HAND_WRITTEN_SKIP_REASON) == [
        2,
        3,
        4,
        5,
        6,
    ]


def test_runner_hardware_needs_are_found_from_literal_arches():
    source = (
        "ARCHES = ('gfx950',)\n"
        "BOTH = ('gfx942', 'gfx950')\n"
        "run_on_device(kernel, buffers, 64, ('gfx950',))\n"
        "run_on_device(kernel, buffers, 64, BOTH)\n"
        "gpu.current_device_arch_in(ARCHES, subject='x')\n"
        "skip_needing_arch(('gfx1250',), reason='x')\n"
        "supported_device_arch(kernel, arches=['gfx1151', 'gfx1201'])\n"
        "arch_skip_reason(computed_arches(), 'x')\n"
    )
    assert [
        (mention.line, mention.arches)
        for mention in lane_mentions_in_source(source, "t.py")
        if mention.kind is LaneMentionKind.RUNNER_HARDWARE_NEED
    ] == [
        (3, ("gfx950",)),
        (5, ("gfx950",)),
        (6, ("gfx1250",)),
        (7, ("gfx1151", "gfx1201")),
    ]


@needs_two_cpus
def test_the_pool_finds_what_each_file_holds_in_the_order_given(tmp_path):
    paths = _write_sources(tmp_path)
    assert lane_mentions(tmp_path, paths) == [
        mention
        for path in paths
        for mention in lane_mentions_in_source(SOURCES[path], path)
    ]


@needs_two_cpus
def test_the_pool_keeps_skip_reasons_and_runner_hardware_needs(tmp_path):
    source = (
        "pytest.skip('waits for a gfx950 runner in CI')\n"
        "skip_needing_arch(('gfx1250',), reason='x')\n"
    )
    paths = []
    for index in range(2):
        (tmp_path / f"needs{index}.py").write_text(source, encoding="utf-8")
        paths.append(f"needs{index}.py")
    found = {
        (mention.path, mention.kind, mention.arches)
        for mention in lane_mentions(tmp_path, paths)
    }
    for path in paths:
        assert (path, LaneMentionKind.HAND_WRITTEN_SKIP_REASON, ()) in found
        assert (path, LaneMentionKind.RUNNER_HARDWARE_NEED, ("gfx1250",)) in found


@needs_two_cpus
def test_a_file_that_does_not_parse_raises_from_the_pool(tmp_path):
    paths = _write_sources(tmp_path)
    (tmp_path / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    with pytest.raises(SyntaxError) as raised:
        lane_mentions(tmp_path, [*paths, "broken.py"])
    assert raised.value.filename == "broken.py"
