# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""``hipdnn_sdpa.missing_bindings`` against a fake ``hipdnn_frontend`` module."""

import types

import pytest

import hipdnn_sdpa
from hipdnn_sdpa import _graph


def frontend_with(required):
    """A fake frontend module holding every name ``_graph._REQUIRED`` lists, minus none."""
    module = types.SimpleNamespace()
    for cls, attrs in required.items():
        setattr(module, cls, types.SimpleNamespace(**{a: object() for a in attrs}))
    return module


@pytest.fixture
def install(monkeypatch):
    def put(module):
        monkeypatch.setattr(_graph, "frontend", lambda: module)

    return put


def test_a_complete_frontend_has_no_missing_bindings(install):
    install(frontend_with(_graph._REQUIRED))
    assert hipdnn_sdpa.missing_bindings() == []


def test_a_frontend_without_ragged_offsets_or_sdpa_names_them_sorted(install):
    module = frontend_with(_graph._REQUIRED)
    del module.Tensor.set_ragged_offset
    del module.Graph.sdpa
    del module.Graph.sdpa_backward
    install(module)
    assert hipdnn_sdpa.missing_bindings() == [
        "Graph.sdpa",
        "Graph.sdpa_backward",
        "Tensor.set_ragged_offset",
    ]


def test_a_frontend_without_a_class_lists_its_names_and_a_bare_class(install):
    module = frontend_with(_graph._REQUIRED)
    del module.Handle
    del module.SdpaBackwardAttributes
    install(module)
    missing = hipdnn_sdpa.missing_bindings()
    assert "Handle" in missing
    assert "SdpaBackwardAttributes.set_dbias" in missing
    assert not any(m.startswith("SdpaAttributes") for m in missing)
