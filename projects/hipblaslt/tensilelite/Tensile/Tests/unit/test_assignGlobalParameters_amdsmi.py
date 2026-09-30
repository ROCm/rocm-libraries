# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""
Unit tests for the amd-smi discovery in
Tensile.Common.GlobalParameters.assignGlobalParameters.

amd-smi is only needed at runtime to pin GPU clocks/fans during benchmarking
and tuning. It must NOT be required to build libraries or validate logic, so a
missing amd-smi has to be non-fatal (warn and leave AMDSMIPath unset) instead of
raising OSError. This guards against the CI regression where the build container
ships rocm-smi but not amd-smi.
"""

from unittest.mock import patch

import pytest

# Importing GlobalParameters pulls in the Tensile toolchain (rocisa bindings,
# etc.). If that chain is unavailable in the current environment, skip rather
# than error at collection time.
try:
    import Tensile.Common.GlobalParameters as GP
    _IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - environment dependent
    GP = None
    _IMPORT_ERROR = exc

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        GP is None,
        reason=f"Tensile.Common.GlobalParameters import unavailable: {_IMPORT_ERROR}",
    ),
]


def test_missing_amdsmi_is_non_fatal_and_warns():
    """A missing amd-smi must not raise; AMDSMIPath stays None and a warning is issued."""
    GP.globalParameters["AMDSMIPath"] = "stale"
    with patch.object(GP, "locateExe", side_effect=OSError("Failed to locate amd-smi")), \
         patch.object(GP, "printWarning") as mock_warn:
        # Must not raise, even on non-Windows.
        GP.assignGlobalParameters({}, {})

    assert GP.globalParameters["AMDSMIPath"] is None
    assert mock_warn.called
    assert any("amd-smi" in str(c.args[0]) for c in mock_warn.call_args_list)


def test_amdsmi_path_is_set_when_found():
    """When amd-smi is located, AMDSMIPath is populated with its path."""
    GP.globalParameters["AMDSMIPath"] = None
    with patch.object(GP, "locateExe", return_value="/opt/rocm/bin/amd-smi"):
        GP.assignGlobalParameters({}, {})

    assert GP.globalParameters["AMDSMIPath"] == "/opt/rocm/bin/amd-smi"
    # amd-smi must be the binary that was looked up.
    with patch.object(GP, "locateExe", return_value="/opt/rocm/bin/amd-smi") as mock_locate:
        GP.assignGlobalParameters({}, {})
    assert mock_locate.call_args.args[1] == "amd-smi"
