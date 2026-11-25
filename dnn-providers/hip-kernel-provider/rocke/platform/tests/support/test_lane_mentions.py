# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""rocke.tests.lane_mentions finds runner lane names and hand-written lane
declarations, the same in worker processes as in this one."""

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

DECLARED = "waits for gfx950-runner: needs a gfx950 GPU"
UNKNOWN = "waits for no-such-lane: x"
SOURCES = {
    "declared.py": f"reason = {DECLARED!r}\n",
    "clean.py": 'reason = "needs a gfx950 GPU"\n',
    "named.py": "from rocke.tests.lanes import GFX950_RUNNER\nlane = GFX950_RUNNER\n",
    "unknown.py": f"reason = {UNKNOWN!r}\n",
}


def _write_sources(root: Path) -> list[str]:
    for name, source in SOURCES.items():
        (root / name).write_text(source, encoding="utf-8")
    return list(SOURCES)


def test_a_source_mentions_a_runner_lane_by_name_and_a_lane_by_declaration():
    assert lane_mentions_in_source(SOURCES["named.py"], "named.py") == [
        LaneMention("named.py", 2, LaneMentionKind.RUNNER_LANE_NAME, "GFX950_RUNNER")
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


@needs_two_cpus
def test_the_pool_finds_what_each_file_holds_in_the_order_given(tmp_path):
    paths = _write_sources(tmp_path)
    assert lane_mentions(tmp_path, paths) == [
        mention
        for path in paths
        for mention in lane_mentions_in_source(SOURCES[path], path)
    ]


@needs_two_cpus
def test_a_file_that_does_not_parse_raises_from_the_pool(tmp_path):
    paths = _write_sources(tmp_path)
    (tmp_path / "broken.py").write_text("def broken(:\n", encoding="utf-8")
    with pytest.raises(SyntaxError) as raised:
        lane_mentions(tmp_path, [*paths, "broken.py"])
    assert raised.value.filename == "broken.py"
