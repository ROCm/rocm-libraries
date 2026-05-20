# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for the ROCm tool binary resolver.

Why this matters: dnn-benchmarking's setup activates a venv whose
``rocm_sdk_core`` wheel installs a ``rocprofv3`` shim in ``.venv/bin``.
That shim points at the venv-bundled rocprofiler-sdk 1.18 / aqlprofile,
which SIGABRTs in ``HsaRsrcFactory::HsaRsrcFactory(bool)`` on any
torch-using workload. The system ``/opt/rocm/bin/rocprofv3`` does not.
The resolver exists to make sure we pick the system binary even when
PATH says otherwise.
"""

import os
import stat

import pytest

from dnn_benchmarking.metrics import _tool_resolver


@pytest.fixture
def fake_rocm_root(tmp_path, monkeypatch):
    """Build a fake $ROCM_PATH layout with an executable bin/<tool>."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    return bin_dir


def _make_executable(path):
    path.write_text("#!/bin/sh\necho hi\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


class TestResolveRocmTool:
    def test_prefers_rocm_path_over_path_resolution(self, fake_rocm_root, monkeypatch):
        """When both $ROCM_PATH/bin/<tool> and a PATH-resolved one exist,
        the resolver picks the ROCm-path one — that's the whole point of
        this helper."""
        rocm_tool = fake_rocm_root / "rocprofv3"
        _make_executable(rocm_tool)
        # Make shutil.which return a different path; we should ignore it.
        monkeypatch.setattr(
            _tool_resolver.shutil, "which", lambda _name: "/some/venv/bin/rocprofv3"
        )
        assert _tool_resolver.resolve_rocm_tool("rocprofv3") == str(rocm_tool)

    def test_falls_back_to_path_when_rocm_path_absent(self, tmp_path, monkeypatch):
        """If $ROCM_PATH/bin/<tool> doesn't exist, fall back to PATH so the
        tool is still usable on hosts where the layout differs."""
        monkeypatch.setenv("ROCM_PATH", str(tmp_path))  # tmp_path/bin doesn't exist
        monkeypatch.setattr(
            _tool_resolver.shutil, "which", lambda _name: "/somewhere/else/rocprofv3"
        )
        assert (
            _tool_resolver.resolve_rocm_tool("rocprofv3") == "/somewhere/else/rocprofv3"
        )

    def test_returns_none_when_neither_present(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ROCM_PATH", str(tmp_path))
        monkeypatch.setattr(_tool_resolver.shutil, "which", lambda _name: None)
        assert _tool_resolver.resolve_rocm_tool("rocprofv3") is None

    def test_skips_rocm_path_when_file_is_not_executable(
        self, fake_rocm_root, monkeypatch
    ):
        """A non-executable file at $ROCM_PATH/bin/<tool> shouldn't be picked.
        Otherwise a stray data file would mask a working PATH binary."""
        rocm_tool = fake_rocm_root / "rocprofv3"
        rocm_tool.write_text("not executable")  # no +x bit
        os.chmod(rocm_tool, 0o644)
        monkeypatch.setattr(
            _tool_resolver.shutil, "which", lambda _name: "/fallback/rocprofv3"
        )
        assert _tool_resolver.resolve_rocm_tool("rocprofv3") == "/fallback/rocprofv3"

    def test_default_rocm_root_is_opt_rocm(self, monkeypatch):
        """When ROCM_PATH isn't set, default to /opt/rocm — the ROCm install
        location on every standard host."""
        monkeypatch.delenv("ROCM_PATH", raising=False)
        assert _tool_resolver._preferred_rocm_root() == "/opt/rocm"

    def test_path_fallback_warns_loudly(self, tmp_path, monkeypatch, capsys):
        """The PATH-fallback case is exactly when the user is at risk
        of picking up the broken venv shim that this module's docstring
        warns about. Surface a warn_once so silent SIGABRT crashes
        downstream have a breadcrumb pointing back here."""
        # Force the diagnostic's seen-set to drop our message; otherwise
        # an earlier test in the same suite may have already fired it.
        from dnn_benchmarking.metrics._diagnostic import reset as reset_warn_once

        reset_warn_once()
        monkeypatch.setenv("ROCM_PATH", str(tmp_path))  # tmp_path/bin missing
        monkeypatch.setattr(
            _tool_resolver.shutil,
            "which",
            lambda _name: "/some/venv/bin/rocprofv3",
        )
        result = _tool_resolver.resolve_rocm_tool("rocprofv3")
        assert result == "/some/venv/bin/rocprofv3"
        captured = capsys.readouterr()
        assert "[metrics:tool_resolver]" in captured.err
        assert "PATH-resolved" in captured.err
        assert "/some/venv/bin/rocprofv3" in captured.err
