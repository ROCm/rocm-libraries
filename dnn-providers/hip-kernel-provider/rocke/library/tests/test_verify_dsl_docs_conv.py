# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Verify the convolution (kernels.*) doc claims extracted from verify_dsl_docs.

This file carries the portion of ``platform/dsl_docs/development/verify_dsl_docs.py``
that exercised the ``kernels.*`` convolution builders documented in
``library/kernels/conv/docs/convolution.md``. It was removed from the platform
file, along with the doc, to keep ``platform/dsl_docs/`` free of ``kernels``
imports.

The cases mirror the original verifier's checks:

* ``ImplicitGemmConvSpec: build + print_ir + LLVM + HIP + CK Tile``
* ``DirectConv16cSpec + DirectConv4cSpec: build + LLVM``
* ``implicit-GEMM conv: build + LLVM + comgr`` (static build only, no launch)

plus the conv symbols the verifier's ``rocke.instances`` import check still
listed, now asserted against the modules the doc names. Host-only: the comgr
case needs the toolchain, not a GPU.
"""

from __future__ import annotations

import importlib
import unittest

from rocke import (
    compile_kernel,
    lower_kernel_to_hip,
    lower_kernel_to_llvm,
    print_ir,
)
from rocke.core.lower_cktile import (
    lower_implicit_gemm_conv_to_cktile,
    lower_spec_to_cktile,
)

from kernels.common.conv_direct_grouped import (
    DirectConv16cSpec,
    DirectConv4cSpec,
    DirectConvProblem,
    build_direct_conv_16c,
    build_direct_conv_4c,
)
from kernels.common.conv_implicit_gemm import (
    ConvProblem,
    ImplicitGemmConvSpec,
    build_implicit_gemm_conv,
)

# Module -> public symbols convolution.md documents from it.
_DOC_SYMBOLS = {
    "kernels.common.conv_implicit_gemm": (
        "ConvProblem",
        "ImplicitGemmConvSpec",
        "build_implicit_gemm_conv",
    ),
    "kernels.common.conv_direct_grouped": (
        "DirectConvProblem",
        "DirectConv16cSpec",
        "build_direct_conv_16c",
        "DirectConv4cSpec",
        "build_direct_conv_4c",
    ),
}


def _conv_spec() -> ImplicitGemmConvSpec:
    return ImplicitGemmConvSpec(
        problem=ConvProblem(
            N=8, Hi=56, Wi=56, C=64, K=64, Y=3, X=3, sH=1, sW=1, pH=1, pW=1, dH=1, dW=1
        ),
        tile_m=64,
        tile_n=64,
        tile_k=64,
        warp_m=2,
        warp_n=2,
        warp_tile_m=32,
        warp_tile_n=32,
        warp_tile_k=16,
        pipeline="mem",
        epilogue="cshuffle",
    )


class TestVerifyDslDocsConv(unittest.TestCase):
    """Convolution doc-example verification (originally in verify_dsl_docs.py)."""

    def test_documented_symbols_import(self) -> None:
        for mod, names in _DOC_SYMBOLS.items():
            m = importlib.import_module(mod)
            missing = [n for n in names if not hasattr(m, n)]
            self.assertEqual(missing, [], f"{mod} missing: {missing}")

    def test_implicit_gemm_conv_build_and_lower(self) -> None:
        spec = _conv_spec()
        kernel = build_implicit_gemm_conv(spec)
        for text in (
            print_ir(kernel),
            lower_kernel_to_llvm(kernel, arch="gfx950"),
            lower_kernel_to_hip(kernel),
            lower_spec_to_cktile(spec),
            lower_implicit_gemm_conv_to_cktile(spec),
        ):
            self.assertIsInstance(text, str)
            self.assertGreater(len(text), 0)

    def test_direct_conv_16c_4c_build_and_lower(self) -> None:
        # DirectConvProblem(N, H, W, groups, cpg, kpg, KH=3, KW=3, PAD=1, stride=1)
        # The 16c spec's default BLOCK_GROUPS=8; use groups=16 so divisibility holds.
        p16 = DirectConvProblem(
            N=8, H=56, W=56, groups=16, cpg=16, kpg=16, KH=3, KW=3, PAD=1, stride=1
        )
        ir16 = lower_kernel_to_llvm(
            build_direct_conv_16c(DirectConv16cSpec(problem=p16)), arch="gfx950"
        )
        self.assertGreater(len(ir16), 0)
        p4 = DirectConvProblem(
            N=8, H=56, W=56, groups=64, cpg=4, kpg=4, KH=3, KW=3, PAD=1, stride=1
        )
        ir4 = lower_kernel_to_llvm(
            build_direct_conv_4c(DirectConv4cSpec(problem=p4)), arch="gfx950"
        )
        self.assertGreater(len(ir4), 0)

    def test_implicit_gemm_conv_compiles(self) -> None:
        # The static build alone proves the IR + LLVM + comgr path; nothing is
        # allocated or launched.
        art = compile_kernel(build_implicit_gemm_conv(_conv_spec()))
        self.assertGreater(art.hsaco_bytes, 0)


if __name__ == "__main__":
    unittest.main()
