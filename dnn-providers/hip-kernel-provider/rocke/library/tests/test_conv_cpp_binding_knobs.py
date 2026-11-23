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

The dgrad / wgrad / Stage 2 bindings also refuse a key they do not read, so a
dataclass field added without a binding twin fails every case below rather
than building the default kernel. A field the C++ spec cannot carry (a fused
accumulator epilogue on wgrad; an explicit LDS layout object) is refused unless
it is the default.
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
_DGRAD = (1, 8, 34, 50, 52, 87, 101, 116, 127, 128, 129, 130, 147, 151, 152)
_WGRAD = (10, 17, 18, 20, 22, 33, 114, 117, 133, 139, 140, 141, 142, 143, 144)
# Stage 2 fold configs that build: the scalar folds and the vector / looped
# ones (vec, rezero, ws_bf16, fold_loop, slab_par); 16-21 both engines refuse.
_REDUCE = tuple(range(16))
_REDUCE_REFUSED = (16, 17, 18, 19, 20, 21)

# Non-default spec fields some emitter config sets but no config of the cover
# above does: they are set only by configs left out of the cover (configs
# that do not lower at both llvm20 and llvm22). A field missing from both the
# cover and this list fails test_the_cover_reaches_every_emitter_knob.
_NOT_IN_COVER = {
    "conv_implicit_gemm_emit": {
        "acc_epilogue.bias",
        "data.mma_dtype",
        "lds_k_pad",
        "split_atomic",
        "split_k_cluster",
    },
    "conv_implicit_gemm_dgrad_emit": {
        "async_dma",
        "chiplet_swizzle",
        "data.mma_dtype",
        "unroll_k",
    },
    "conv_implicit_gemm_wgrad_emit": {
        "data.mma_dtype",
        "prefetch_global_regs",
        "register_prefetch",
    },
}


def _set_fields(spec, prefix: str = "") -> set:
    """The non-default fields of a spec dataclass (nested ones dotted)."""
    out = set()
    for f in dataclasses.fields(spec):
        if f.name == "problem":
            continue
        v = getattr(spec, f.name)
        if dataclasses.is_dataclass(v) and not isinstance(v, type):
            out |= _set_fields(v, prefix + f.name + ".")
            continue
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:
            default = f.default_factory()
        else:
            continue
        if v != default:
            out.add(prefix + f.name)
    return out


def _emitter_configs(module: str):
    if _PARITY not in sys.path:
        sys.path.insert(0, _PARITY)
    mod = importlib.import_module(module)
    idx = 0
    while True:
        try:
            spec, _ = mod._spec(idx)
        except (SystemExit, IndexError, KeyError, ValueError):
            return
        yield idx, spec
        idx += 1


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
    from kernels.conv.engine_bridge import _conv_acc_epilogue_to_dict

    d = {
        f.name: getattr(spec, f.name)
        for f in dataclasses.fields(spec)
        if f.name not in ("problem", "data", "acc_epilogue", "lds_layout")
        and getattr(spec, f.name) is not None
    }
    d["acc_epilogue"] = _conv_acc_epilogue_to_dict(spec.acc_epilogue)
    d["problem"] = {
        k: v for k, v in dataclasses.asdict(spec.problem).items() if v is not None
    }
    d["dtype_a"] = spec.data.dtype_a
    d["dtype_b"] = spec.data.dtype_b
    d["dtype_d"] = spec.data.dtype_d
    return d


def _reduce_dict(spec) -> dict:
    """A WgradReduceSpec as the dict the conv_wgrad_reduce binding reads."""
    from kernels.common.conv_wgrad_workspace_reduce import _wg_M, _wg_N

    d = {
        f.name: getattr(spec, f.name)
        for f in dataclasses.fields(spec)
        if f.name != "problem"
    }
    d.update(
        wg_M=_wg_M(spec.problem),
        wg_N=_wg_N(spec.problem),
        problem_short=spec.problem.short(),
    )
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
            is_valid_wgrad_spec,
        )

        eng = _engine()
        for idx in _WGRAD:
            spec, arch = _emitter_spec("conv_implicit_gemm_wgrad_emit", idx)
            with self.subTest(idx=idx):
                d = _flat_dict(spec)
                self.assertEqual(
                    tuple(eng.conv_wgrad_is_valid_spec(d, arch)),
                    is_valid_wgrad_spec(spec, arch),
                )
                self.assertEqual(
                    eng.conv_wgrad_lower_llvm(d, arch),
                    _python_ll(build_implicit_gemm_conv_wgrad, spec, arch),
                )

    def test_wgrad_reduce(self):
        # The Stage 2 binding reads every fold knob: dropped, a vec=4 dict
        # would build the scalar fold under the vector fold's grid.
        from kernels.common.conv_wgrad_workspace_reduce import (
            build_conv_wgrad_workspace_reduce,
        )

        eng = _engine()
        for idx in _REDUCE:
            if idx in _REDUCE_REFUSED:
                continue
            spec, arch = _emitter_spec("conv_wgrad_workspace_reduce_emit", idx)
            with self.subTest(idx=idx):
                self.assertEqual(
                    eng.conv_wgrad_reduce_lower_llvm(_reduce_dict(spec), arch),
                    _python_ll(build_conv_wgrad_workspace_reduce, spec, arch),
                )
        for idx in _REDUCE_REFUSED:
            spec, arch = _emitter_spec("conv_wgrad_workspace_reduce_emit", idx)
            with self.subTest(idx=idx):
                with self.assertRaises(ValueError):
                    build_conv_wgrad_workspace_reduce(spec, arch=arch)
                with self.assertRaises(RuntimeError):
                    eng.conv_wgrad_reduce_lower_llvm(_reduce_dict(spec), arch)

    def test_refuses_a_key_it_does_not_read(self):
        eng = _engine()
        wg, warch = _emitter_spec("conv_implicit_gemm_wgrad_emit", 10)
        dg, darch = _emitter_spec("conv_implicit_gemm_dgrad_emit", 91)
        rd, rarch = _emitter_spec("conv_wgrad_workspace_reduce_emit", 9)
        for call, d, arch in (
            (eng.conv_wgrad_lower_llvm, _flat_dict(wg), warch),
            (eng.conv_wgrad_is_valid_spec, _flat_dict(wg), warch),
            (eng.conv_dgrad_lower_llvm, _flat_dict(dg), darch),
            (eng.conv_dgrad_is_valid_spec, _flat_dict(dg), darch),
            (eng.conv_wgrad_reduce_lower_llvm, _reduce_dict(rd), rarch),
        ):
            with self.subTest(call=call.__name__):
                call(d, arch)
                with self.assertRaisesRegex(ValueError, "no spec field 'bogus_knob'"):
                    call(dict(d, bogus_knob=1), arch)

    def test_refuses_a_fused_epilogue_it_cannot_carry(self):
        # wgrad has no C++ epilogue twin (dgrad has one, and test_dgrad covers
        # it). The identity epilogue (what _flat_dict passes for every case
        # above) is admitted.
        from kernels.common._conv_implicit_gemm_common import (
            ConvAccumulatorEpilogue,
        )
        from kernels.conv.engine_bridge import _conv_acc_epilogue_to_dict

        eng = _engine()
        relu = _conv_acc_epilogue_to_dict(ConvAccumulatorEpilogue(scale=2.0, relu=True))
        wg, warch = _emitter_spec("conv_implicit_gemm_wgrad_emit", 0)
        dg, darch = _emitter_spec("conv_implicit_gemm_dgrad_emit", 91)
        d = _flat_dict(wg)
        eng.conv_wgrad_lower_llvm(dict(d, acc_epilogue=None), warch)
        with self.assertRaisesRegex(ValueError, "acc_epilogue"):
            eng.conv_wgrad_lower_llvm(dict(d, acc_epilogue=relu), warch)
        for call, d, arch in (
            (eng.conv_wgrad_lower_llvm, _flat_dict(wg), warch),
            (eng.conv_dgrad_lower_llvm, _flat_dict(dg), darch),
        ):
            with self.subTest(call=call.__name__):
                with self.assertRaisesRegex(ValueError, "lds_layout"):
                    call(dict(d, lds_layout="xor_swizzled"), arch)

    def test_dgrad_acc_epilogue_reaches_the_kernel_name(self):
        # A fused epilogue builds Python's kernel under Python's name (the
        # epilogue tag), so it cannot share a symbol with the identity build.
        from kernels.common._conv_implicit_gemm_common import (
            ConvAccumulatorEpilogue,
        )
        from kernels.common.conv_implicit_gemm_dgrad import (
            build_implicit_gemm_conv_dgrad,
        )

        eng = _engine()
        spec, arch = _emitter_spec("conv_implicit_gemm_dgrad_emit", 91)
        epi = ConvAccumulatorEpilogue(bias=-0.5, relu=True, clamp_max=6.0)
        spec = dataclasses.replace(spec, acc_epilogue=epi)
        ll = eng.conv_dgrad_lower_llvm(_flat_dict(spec), arch)
        self.assertIn(epi.tag(), ll)
        self.assertEqual(ll, _python_ll(build_implicit_gemm_conv_dgrad, spec, arch))
        bad = ConvAccumulatorEpilogue(clamp_min=1.0, clamp_max=0.0)
        with self.assertRaisesRegex(RuntimeError, "clamp_min must be <= clamp_max"):
            eng.conv_dgrad_lower_llvm(
                _flat_dict(dataclasses.replace(spec, acc_epilogue=bad)), arch
            )
        with self.assertRaisesRegex(ValueError, "acc_epilogue' must be a dict"):
            eng.conv_dgrad_lower_llvm(dict(_flat_dict(spec), acc_epilogue=1), arch)

    def test_the_cover_reaches_every_emitter_knob(self):
        """The index tuples are a cover of the emitters' knob fields; an
        emitter config inserted before them shifts what they select. Every
        non-default field some config sets is set by a covered config, or is
        listed in _NOT_IN_COVER."""
        for module, cover in (
            ("conv_implicit_gemm_emit", _FWD),
            ("conv_implicit_gemm_dgrad_emit", _DGRAD),
            ("conv_implicit_gemm_wgrad_emit", _WGRAD),
        ):
            with self.subTest(module=module):
                every, covered = set(), set()
                for idx, spec in _emitter_configs(module):
                    fields = _set_fields(spec)
                    every |= fields
                    if idx in cover:
                        covered |= fields
                self.assertEqual(every - covered, _NOT_IN_COVER[module])

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
