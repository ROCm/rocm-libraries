# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The rocke_library_engine conv bindings forward every knob field of the spec dict.

No GPU needed. The parity emitters (tests/parity/*_emit.{py,c}) pin the C++
engine to the Python one config by config through the C API; this file pins
the Python-dict path of the ``rocke_library_engine`` binding (what
``kernels.conv.engine_bridge`` uses) to the same bytes: for a set of parity configs
that together set every knob field the emitters exercise, the binding lowers
the dataclass' dict to the Python engine's LLVM IR. A dict field the binding
dropped would build the default kernel under the knob's name and fail here.
"""

from __future__ import annotations

import dataclasses
import importlib
import os
import sys
import unittest

_PARITY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "parity")

# Parity config indices: a greedy cover of every non-default spec field the
# emitters set on configs that lower at llvm20 and llvm22 (fwd / dgrad /
# wgrad), so each field reaches the binding at least once. The gfx942 3-D
# configs sit past the emitters' own configs (fwd from 86, dgrad from 71,
# wgrad from 109).
_FWD = (
    3,
    9,
    10,
    19,
    30,
    92,
    118,
    131,
    136,
    140,
    142,
    147,
    151,
    155,
    157,
    165,
    167,
    168,
    170,
    175,
    176,
    178,
    180,
)
_DGRAD = (1, 8, 34, 50, 52, 87, 101, 116, 127, 128, 129, 130)
_WGRAD = (10, 17, 18, 20, 22, 33, 114, 117, 133, 139, 140, 141, 142)


def _engine():
    from rocke.core.backend import BackendError, import_engine

    from kernels.conv.engine_bridge import ENGINE_MODULE

    try:
        return import_engine(ENGINE_MODULE)
    except BackendError as e:
        raise unittest.SkipTest(f"C++ engine not importable: {e}") from None


def _emitter_spec(module: str, idx: int):
    if _PARITY not in sys.path:
        sys.path.insert(0, _PARITY)
    spec, arch = importlib.import_module(module)._spec(idx)
    return spec, arch


def _flat_dict(spec) -> dict:
    """A dgrad / wgrad dataclass as the flat dict the binding reads."""
    d = {
        f.name: getattr(spec, f.name)
        for f in dataclasses.fields(spec)
        if f.name not in ("problem", "data", "acc_epilogue", "lds_layout")
        and getattr(spec, f.name) is not None
    }
    d["problem"] = {
        k: v for k, v in dataclasses.asdict(spec.problem).items() if v is not None
    }
    d["dtype_a"] = spec.data.dtype_a
    d["dtype_b"] = spec.data.dtype_b
    d["dtype_d"] = spec.data.dtype_d
    return d


def _python_ll(build, spec, arch: str) -> str:
    from rocke.core.lower_llvm import _lower_kernel_to_llvm_python

    return _lower_kernel_to_llvm_python(build(spec, arch=arch), arch=arch)


class TestConvBindingForwardsKnobs(unittest.TestCase):
    def test_fwd(self):
        from kernels.common.conv_implicit_gemm import (
            build_implicit_gemm_conv,
            is_valid_spec,
        )
        from kernels.conv.engine_bridge import conv_implicit_gemm_spec_to_dict

        eng = _engine()
        for idx in _FWD:
            spec, arch = _emitter_spec("conv_implicit_gemm_emit", idx)
            with self.subTest(idx=idx):
                self.assertEqual(
                    tuple(
                        eng.conv_implicit_gemm_is_valid_spec(
                            conv_implicit_gemm_spec_to_dict(spec), arch
                        )
                    ),
                    is_valid_spec(spec, arch),
                )
                self.assertEqual(
                    eng.conv_implicit_gemm_lower_llvm(
                        conv_implicit_gemm_spec_to_dict(spec), arch=arch
                    ),
                    _python_ll(build_implicit_gemm_conv, spec, arch),
                )

    def test_dgrad(self):
        from kernels.common.conv_implicit_gemm_dgrad import (
            build_implicit_gemm_conv_dgrad,
            is_valid_dgrad_spec,
        )

        eng = _engine()
        for idx in _DGRAD:
            spec, arch = _emitter_spec("conv_implicit_gemm_dgrad_emit", idx)
            with self.subTest(idx=idx):
                d = _flat_dict(spec)
                self.assertEqual(
                    tuple(eng.conv_dgrad_is_valid_spec(d, arch)),
                    is_valid_dgrad_spec(spec, arch),
                )
                self.assertEqual(
                    eng.conv_dgrad_lower_llvm(d, arch),
                    _python_ll(build_implicit_gemm_conv_dgrad, spec, arch),
                )

    def test_wgrad(self):
        from kernels.common.conv_implicit_gemm_wgrad import (
            build_implicit_gemm_conv_wgrad,
        )

        eng = _engine()
        for idx in _WGRAD:
            spec, arch = _emitter_spec("conv_implicit_gemm_wgrad_emit", idx)
            with self.subTest(idx=idx):
                self.assertEqual(
                    eng.conv_wgrad_lower_llvm(_flat_dict(spec), arch),
                    _python_ll(build_implicit_gemm_conv_wgrad, spec, arch),
                )

    def test_3d_problem_without_is_3d_key(self):
        # dataclasses.asdict(ConvProblem) has no is_3d key: the depth fields
        # make the problem 3-D (the 3-D dgrad builds; a 2-D read would not).
        from kernels.common.conv_implicit_gemm_dgrad import (
            build_implicit_gemm_conv_dgrad,
        )

        eng = _engine()
        spec, arch = _emitter_spec("conv_implicit_gemm_dgrad_emit", 91)
        d = _flat_dict(spec)
        self.assertNotIn("is_3d", d["problem"])
        self.assertEqual(
            eng.conv_dgrad_lower_llvm(d, arch),
            _python_ll(build_implicit_gemm_conv_dgrad, spec, arch),
        )

    def test_refuses_a_knob_without_a_cpp_twin(self):
        from kernels.conv.engine_bridge import conv_implicit_gemm_spec_to_dict

        eng = _engine()
        spec, arch = _emitter_spec("conv_implicit_gemm_emit", 0)
        d = conv_implicit_gemm_spec_to_dict(spec)
        eng.conv_implicit_gemm_lower_llvm(dict(d, a_direct=0), arch=arch)
        with self.assertRaisesRegex(ValueError, "a_direct"):
            eng.conv_implicit_gemm_lower_llvm(dict(d, a_direct=1), arch=arch)
        dg, darch = _emitter_spec("conv_implicit_gemm_dgrad_emit", 91)
        with self.assertRaisesRegex(ValueError, "cshuffle_no_alias"):
            eng.conv_dgrad_lower_llvm(
                dict(_flat_dict(dg), cshuffle_no_alias=True), darch
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
