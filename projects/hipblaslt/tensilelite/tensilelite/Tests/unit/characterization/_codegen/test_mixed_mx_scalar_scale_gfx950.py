# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Scalar A and B scales in mixed MXFP4 x MXFP8 kernels (gfx950).

With UseScaleAB: Scalar, the kernel loads one scale for A and one for B and
multiplies alpha by both before the output is written. A mixed problem type
that omits MacDataTypeA or MacDataTypeB used to treat the operand as converted
to the other operand's type: the kernel loaded that operand's scale for the
conversion, never applied it, and multiplied alpha by the other scale only.
Each case emits one kernel and checks that alpha is multiplied by both loaded
scales.
"""

import re

import pytest
import yaml

from config_harness import emit_kernels_from_config

pytestmark = pytest.mark.unit

_ARCH = "gfx950"

_SCALE_LOAD = re.compile(r"^s_load_dword ([^,\s]+), s\[sgprAddressScale([AB]):", re.M)
_ALPHA_UPDATE = re.compile(
    r"^v_mov_b32 (v\d+), s\[sgprAlpha\]\n(?P<body>(?:.*\n)*?)v_readfirstlane_b32 s\[sgprAlpha\], \1\b",
    re.M,
)
_MUL = re.compile(r"^v_mul_f32 (v\d+), \1, ([^,\s]+)", re.M)


def _config(dataTypes, glvwA, glvwB):
    problemType = {
        "OperationType": "GEMM",
        "DestDataType": "s",
        "ComputeDataType": "s",
        "HighPrecisionAccumulate": True,
        "MXBlockA": 32,
        "MXBlockB": 32,
        "TransposeA": True,
        "TransposeB": False,
        "UseBeta": True,
        "Batched": True,
        "UseScaleAB": "Scalar",
    }
    problemType.update(dataTypes)
    parameters = {
        "KernelLanguage": ["Assembly"],
        "MatrixInstruction": [[16, 16, 128, 1, 1, 4, 2, 2, 2]],
        "DepthU": [256],
        "PrefetchGlobalRead": [2],
        "PrefetchLocalRead": [1],
        "DirectToLds": [0],
        "GlobalReadVectorWidthA": [glvwA],
        "GlobalReadVectorWidthB": [glvwB],
        "LocalReadVectorWidth": [32],
        "VectorWidthA": [1],
        "VectorWidthB": [1],
        "ClusterLocalRead": [1],
        "1LDSBuffer": [0],
        "GlobalSplitU": [1],
        "ScheduleIterAlg": [3],
        "LdsPadA": [4],
        "LdsPadB": [4],
        "WorkGroupMapping": [64],
        "UseSubtileImpl": [True],
    }
    return {
        "GlobalParameters": {"MXScaleFormat": 1},
        "BenchmarkProblems": [[problemType, {
            "ForkParameters": [{key: values} for key, values in parameters.items()],
        }]],
    }


@pytest.mark.parametrize(
    "dataTypes, glvwA, glvwB",
    [
        ({"DataType": "F4", "DataTypeB": "F8"}, 32, 16),
        ({"DataType": "F4", "DataTypeA": "F8"}, 16, 32),
    ],
    ids=["F4xF8", "F8xF4"],
)
def test_alpha_takes_both_scalar_scales(tmp_path, dataTypes, glvwA, glvwB):
    configPath = tmp_path / "mixed_mx_scalar_scale.yaml"
    configPath.write_text(yaml.safe_dump(_config(dataTypes, glvwA, glvwB)))
    results = emit_kernels_from_config(configPath, arch=_ARCH, expected_fork_count=1)
    assert len(results) == 1
    _base, source, error = results[0]
    assert error == 0

    loaded = {}
    for register, operand in _SCALE_LOAD.findall(source):
        loaded.setdefault(operand, set()).add(register)
    assert set(loaded) == {"A", "B"}

    updates = list(_ALPHA_UPDATE.finditer(source))
    assert updates, "no alpha update found"
    for update in updates:
        factors = {m.group(2) for m in _MUL.finditer(update.group("body"))
                   if m.group(1) == update.group(1)}
        for operand in ("A", "B"):
            assert factors & loaded[operand], \
                "alpha is not multiplied by scale %s (loaded into %s; factors %s)" \
                % (operand, sorted(loaded[operand]), sorted(factors))
