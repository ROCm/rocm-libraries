# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""An engine-level (L1) measurement takes no workspace constraint.

`hipdnn_bench` refuses a workspace limit in its engine modes, because the limit turns the
query into a configuration-level one: the rows it produced would be rejected on import.
`generate` refuses it up front, before it measures anything.
"""

import pytest

# uhd_gen's command line imports its training stack.
pytest.importorskip("numpy")
pytest.importorskip("pandas")
from uhd_gen.__main__ import main  # noqa: E402


def _arguments(tmp_path, *extra):
    graphs = tmp_path / "graphs"
    graphs.mkdir(exist_ok=True)
    return [
        "generate",
        "--collect-only",
        "--graphs",
        str(graphs),
        "--descriptor-tree",
        str(tmp_path),
        "--engine-id",
        "7",
        "--role",
        "predict_engine",
        "--output-dir",
        str(tmp_path / "out"),
        *extra,
    ]


@pytest.mark.parametrize(
    "limit",
    [["--workspace-limit", "67108864"], ["--knob", "global.workspace_size_limit=64"]],
)
def test_a_workspace_limit_is_refused_before_anything_is_measured(
    tmp_path, monkeypatch, caplog, limit
):
    def measured(*args, **kwargs):
        raise AssertionError("a refused run must not start the bench")

    monkeypatch.setattr("uhd_gen.generate._run_json", measured)

    assert main(_arguments(tmp_path, *limit)) == 1
    assert "take no workspace constraint" in caplog.text
    assert not (tmp_path / "out").exists()
