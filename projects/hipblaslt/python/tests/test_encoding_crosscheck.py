# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Cross-check between hipBLASLt fp8 converters and ml_dtypes encodings.

Both hipBLASLt (_core.pack_fp8) and ml_dtypes encode float32 values to fp8
bytes.  These tests verify that both produce identical bit patterns over a
representative sweep of in-range values.

Known divergences (not tested here, documented for reference):
- Out-of-range values for FNUZ types (e.g. 448.0 > e4m3_fnuz max 240.0):
  hipBLASLt clamps to the representable max (240.0), while ml_dtypes converts
  to NaN (the FNUZ NaN sentinel 0x80).
- NaN payload bits in e5m2: both encoders produce NaN, but choose different
  bit patterns within the NaN payload (hipBLASLt: 0x7F, ml_dtypes: 0x7E).
  Both decode back to float NaN, so the semantic result is identical.
These divergences only appear outside the [-8, 8] test range or for NaN input;
they are not triggered by the linspace sweep below.
"""

import numpy as np
import ml_dtypes
import pytest
import hipblaslt

c = hipblaslt._core

# Four (hipBLASLt-fmt-string, ml_dtypes-dtype) pairs covering OCP and FNUZ types.
CASES = [
    ("e4m3", ml_dtypes.float8_e4m3fn),
    ("e5m2", ml_dtypes.float8_e5m2),
    ("e4m3_fnuz", ml_dtypes.float8_e4m3fnuz),
    ("e5m2_fnuz", ml_dtypes.float8_e5m2fnuz),
]


@pytest.mark.parametrize("fmt,mld_type", CASES, ids=[c[0] for c in CASES])
def test_ml_dtypes_matches_hipblaslt(fmt, mld_type):
    """Bit-for-bit agreement over linspace(-8, 8, 257).

    The range [-8, 8] is well within the representable range of all four fp8
    formats (e4m3fn max: 448, e5m2 max: inf, e4m3fnuz max: 240, e5m2fnuz max:
    57344), so no clamping or saturation artefacts occur.  The 257-point sweep
    probes midpoints between representable values and exercises round-to-nearest
    behaviour.
    """
    vals = np.linspace(-8.0, 8.0, 257, dtype=np.float32)

    hip_bytes = c.pack_fp8(vals, fmt)
    mld_bytes = vals.astype(mld_type).view(np.uint8)

    mismatches = np.nonzero(hip_bytes != mld_bytes)[0]
    assert mismatches.size == 0, (
        f"{fmt}: {mismatches.size} encoding divergence(s) at values "
        f"{vals[mismatches][:8]} — hipBLASLt bytes {hip_bytes[mismatches][:8]} "
        f"vs ml_dtypes bytes {mld_bytes[mismatches][:8]}"
    )
