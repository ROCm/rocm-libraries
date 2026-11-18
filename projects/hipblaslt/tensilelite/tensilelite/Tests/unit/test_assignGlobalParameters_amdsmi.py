# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
Unit tests for the amd-smi discovery in
tensilelite.Common.GlobalParameters.assignGlobalParameters.

amd-smi is only needed at runtime to pin GPU clocks/fans during benchmarking
and tuning. It must NOT be required to build libraries or validate logic, so a
missing amd-smi has to be non-fatal (warn and leave AMDSMIPath unset) instead of
raising OSError. This guards against the CI regression where the build container
ships rocm-smi but not amd-smi.
"""

from pathlib import Path
from unittest.mock import patch

import pytest

# Importing GlobalParameters pulls in the TensileLite toolchain (rocisa bindings,
# etc.). If that chain is unavailable in the current environment, skip rather
# than error at collection time.
try:
    import tensilelite.Common.GlobalParameters as GP
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    GP = None
    _IMPORT_ERROR = exc

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        GP is None,
        reason=f"tensilelite.Common.GlobalParameters import unavailable: {_IMPORT_ERROR}",
    ),
]


def test_missing_amdsmi_is_non_fatal_and_warns():
    """A missing amd-smi must not raise; AMDSMIPath stays None and a warning is issued."""
    GP.globalParameters["AMDSMIPath"] = "stale"
    with patch.object(GP._runtime, "executable_search_paths", return_value=[]), \
         patch.object(GP, "printWarning") as mock_warn:
        # Must not raise, even on non-Windows.
        GP.assignGlobalParameters({}, {})

    assert GP.globalParameters["AMDSMIPath"] is None
    assert mock_warn.called
    assert any("amd-smi" in str(c.args[0]) for c in mock_warn.call_args_list)


def test_amdsmi_path_is_set_from_the_selected_runtime(tmp_path, monkeypatch):
    """amd-smi is selected from the frozen runtime, not an ambient prefix."""
    selected_bin = tmp_path / "selected" / "bin"
    ambient_bin = tmp_path / "ambient" / "bin"
    selected_bin.mkdir(parents=True)
    ambient_bin.mkdir(parents=True)
    selected_amdsmi = selected_bin / "amd-smi"
    selected_amdsmi.write_text("#!/bin/sh\n", encoding="utf-8")
    selected_amdsmi.chmod(0o755)
    (ambient_bin / "amd-smi").write_text("#!/bin/sh\n", encoding="utf-8")
    monkeypatch.setenv("ROCM_PATH", str(ambient_bin.parent))
    GP.globalParameters["AMDSMIPath"] = None
    with patch.object(
        GP._runtime, "executable_search_paths", return_value=[selected_bin]
    ):
        GP.assignGlobalParameters({}, {})

    assert GP.globalParameters["AMDSMIPath"] == str(selected_amdsmi)
