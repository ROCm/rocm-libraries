# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Guard: every conv build_* in library/kernels must have arch as keyword-only.

Enforces the (spec, *, arch, ...) contract from PR #11237.
Any new conv builder added to library/kernels/__init__.py is automatically
covered because the test discovers builders by name prefix.
"""

from __future__ import annotations

import inspect

import kernels


_CONV_BUILDERS = [
    "build_implicit_gemm_conv",
    "build_implicit_gemm_conv_wgrad",
    "build_implicit_gemm_conv_dgrad",
    "build_direct_conv_16c",
    "build_direct_conv_4c",
    "build_deep_fused_conv_pool",
    "build_img2col",
]


def test_conv_builders_have_keyword_only_arch():
    for name in _CONV_BUILDERS:
        fn = getattr(kernels, name)
        params = list(inspect.signature(fn).parameters.values())
        kw_only = [p for p in params if p.kind == inspect.Parameter.KEYWORD_ONLY]
        assert kw_only, (
            f"{name}: no keyword-only parameters — missing '*' in signature"
        )
        assert kw_only[0].name == "arch", (
            f"{name}: first keyword-only parameter is '{kw_only[0].name}', expected 'arch'"
        )


def test_conv_builders_first_param_is_spec():
    for name in _CONV_BUILDERS:
        fn = getattr(kernels, name)
        params = list(inspect.signature(fn).parameters.values())
        assert params, f"{name}: has no parameters at all"
        assert params[0].kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ), f"{name}: first parameter '{params[0].name}' is not positional"
        assert params[0].name == "spec", (
            f"{name}: first parameter is '{params[0].name}', expected 'spec'"
        )
