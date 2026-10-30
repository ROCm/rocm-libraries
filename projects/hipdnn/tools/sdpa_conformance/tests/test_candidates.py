# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""``hipdnn_sdpa._candidates``: ``hipdnn_bench enumerate`` as a paged, cached candidate list.

The enumerator is a fake of the ``hipdnn_bench`` subprocess; it speaks the CLI of
``tools/bench/HipdnnBench.cpp`` (``enumerate`` first, ``--offset`` / ``--limit``, a page as
``pageJson`` writes it, a failure as ``Candidate enumeration failed: <message>`` on stderr, exit 1).
"""

import json
import subprocess
from pathlib import Path

import pytest

from hipdnn_sdpa import _candidates
from hipdnn_sdpa._candidates import Candidate, NotEnumerable

PREFIX = "/opt/hipdnn"
GRAPH = '{"graph": "problem-1"}'
ENGINE = "hipkernel:flydsl_sdpa_cdna"
NOT_SUPPORTED = "Engine does not support matched-catalog enumeration"
AMBIGUOUS = (
    "Ambiguous enrolled knob tuple for candidates 'k1' and 'k2'; expose enough KMD "
    "integer fields in the collection UED knobs"
)


def entry(kernel, **knobs):
    return {"id": kernel, "knob_settings": knobs, "kernel_features": {}}


class FakeBench:
    """``engines`` maps an engine name to its candidate entries, or to a failure message."""

    def __init__(self, engines):
        self.engines = engines
        self.calls = []
        self.graphs = []

    def __call__(self, cmd, **kwargs):
        self.calls.append(cmd)
        self.graphs.append(Path(cmd[cmd.index("--graph") + 1]).read_text())
        engine = cmd[cmd.index("--engine-name") + 1]
        offset = int(cmd[cmd.index("--offset") + 1]) if "--offset" in cmd else 0
        limit = int(cmd[cmd.index("--limit") + 1]) if "--limit" in cmd else 10000
        found = self.engines[engine]
        if isinstance(found, str):
            return subprocess.CompletedProcess(
                cmd, 1, "", f"Candidate enumeration failed: {found}\n"
            )
        end = offset + limit
        page = {
            "engine_name": engine,
            "candidates": found[offset:end],
            "total_count": len(found),
            "offset": offset,
            "next_offset": end if end < len(found) else None,
        }
        return subprocess.CompletedProcess(cmd, 0, json.dumps(page) + "\n", "")


@pytest.fixture
def bench(monkeypatch):
    def install(engines):
        fake = FakeBench(engines)
        monkeypatch.setattr(_candidates, "_run", fake)
        return fake

    monkeypatch.setenv(_candidates.contract.PREFIX_ENV, PREFIX)
    monkeypatch.setattr(_candidates, "_CACHE", {})
    return install


def test_candidates_come_back_in_catalog_order_with_their_kernel_and_knobs(bench):
    fake = bench(
        {
            ENGINE: [
                entry("k-a", block=64, mode="fast"),
                entry("k-b", block=128, mode="fast"),
                entry("k-c", block=128, mode="slow"),
            ]
        }
    )
    found = _candidates.list(GRAPH, ENGINE)
    assert found == [
        Candidate("k-a", (("block", 64), ("mode", "fast"))),
        Candidate("k-b", (("block", 128), ("mode", "fast"))),
        Candidate("k-c", (("block", 128), ("mode", "slow"))),
    ]
    (cmd,) = fake.calls
    assert cmd[:2] == [f"{PREFIX}/bin/hipdnn_bench", "enumerate"]
    assert cmd[cmd.index("--engine-name") + 1] == ENGINE
    assert cmd[cmd.index("--plugin-dir") + 1] == f"{PREFIX}/lib/hipdnn_plugins/engines"
    assert fake.graphs == [GRAPH]  # the graph file held the problem's JSON


def test_a_catalog_larger_than_a_page_is_read_to_the_end(bench, monkeypatch):
    monkeypatch.setattr(_candidates, "PAGE_LIMIT", 2)
    fake = bench({ENGINE: [entry(f"k{i}", n=i) for i in range(5)]})
    found = _candidates.list(GRAPH, ENGINE)
    assert [c.id for c in found] == ["k0", "k1", "k2", "k3", "k4"]
    offsets = [c[c.index("--offset") + 1] for c in fake.calls]
    assert offsets == ["0", "2", "4"]


@pytest.mark.parametrize("message", [NOT_SUPPORTED, AMBIGUOUS])
def test_an_engine_that_cannot_enumerate_is_a_record_carrying_the_message(
    bench, message
):
    bench({ENGINE: message})
    found = _candidates.list(GRAPH, ENGINE)
    assert isinstance(found, NotEnumerable)
    assert message in found.message


def test_an_empty_catalog_is_an_empty_list_not_a_not_enumerable_record(bench):
    bench({ENGINE: []})
    assert _candidates.list(GRAPH, ENGINE) == []


def test_results_are_cached_per_problem_and_engine_failures_included(bench):
    fake = bench({ENGINE: [entry("k-a")], "other": NOT_SUPPORTED})
    first = _candidates.list(GRAPH, ENGINE)
    assert _candidates.list(GRAPH, ENGINE) == first
    assert len(fake.calls) == 1
    assert isinstance(_candidates.list(GRAPH, "other"), NotEnumerable)
    assert isinstance(_candidates.list(GRAPH, "other"), NotEnumerable)
    assert len(fake.calls) == 2  # a new engine runs; the failure was cached too
    _candidates.list('{"graph": "problem-2"}', ENGINE)
    assert len(fake.calls) == 3  # a new problem runs again


def test_without_a_hipdnn_prefix_the_call_fails_loudly(bench, monkeypatch):
    bench({ENGINE: []})
    monkeypatch.delenv(_candidates.contract.PREFIX_ENV)
    with pytest.raises(RuntimeError, match=_candidates.contract.PREFIX_ENV):
        _candidates.list(GRAPH, ENGINE)
