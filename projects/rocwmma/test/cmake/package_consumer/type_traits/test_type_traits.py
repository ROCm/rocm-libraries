# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compile actual public headers to check defaults, compatibility and diagnostics.

Requires a HIP Clang driver, but no GPU. Verifies the permanent owned-trait
behavior, the accepted =0 no-op and rejection of the removed legacy mode.
"""

import argparse
import subprocess
import unittest


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--compiler", required=True)
parser.add_argument("--arch", default="gfx1100")
parser.add_argument("--include", required=True)
args = parser.parse_args()

MODE = "ROCWMMA_LEGACY_STD_TRAIT_SPECIALIZATIONS"
SUPPRESS = "ROCWMMA_SUPPRESS_LEGACY_STD_TRAIT_DEPRECATION"
TYPES = (
    "rocwmma::float8_t", "rocwmma::bfloat8_t",
    "rocwmma::float8_fnuz_t", "rocwmma::bfloat8_fnuz_t", "rocwmma::xfloat32_t",
)
TRAITS = ("is_arithmetic", "is_floating_point", "is_signed")
PUBLIC = "#include <rocwmma/rocwmma.hpp>\n"


def compile_source(source, definitions=(), extra=()):
    return subprocess.run(
        [args.compiler, "-x", "hip", "-std=c++17", "-fsyntax-only", "-Werror",
         "-Wdeprecated-declarations", "-ferror-limit=0", "--offload-arch=" + args.arch,
         "-isystem", args.include,
         *["-D" + definition for definition in definitions], *extra, "-"],
        input=source, text=True, capture_output=True,
    )


def checks(namespace, expected):
    return "\n".join(
        f"static_assert({namespace}::{trait}<{typ}>::value == {str(expected).lower()});"
        for typ in TYPES for trait in TRAITS
    )


class TraitMigration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        baseline = compile_source("#include <type_traits>\n")
        if baseline.returncode:
            raise RuntimeError(baseline.stderr)

    def succeeds(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)

    def fails(self, result, message):
        self.assertNotEqual(result.returncode, 0, "Compilation unexpectedly succeeded")
        self.assertIn(message, result.stderr)

    def test_shipped_default(self):
        self.succeeds(compile_source(
            PUBLIC + f"static_assert({MODE} == 0);\n"
            + checks("rocwmma", True) + checks("std", False)
        ))

    def test_owned_traits_and_native_fallback(self):
        values = "\n".join(
            f"static_assert(rocwmma::{trait}_v<{typ}>);"
            for typ in TYPES for trait in TRAITS
        )
        controls = """
static_assert(rocwmma::is_signed_v<const int>);
static_assert(!rocwmma::is_signed_v<unsigned int>);
static_assert(rocwmma::is_floating_point_v<const double>);
static_assert(!rocwmma::is_floating_point_v<float&>);
static_assert(!rocwmma::is_arithmetic_v<void*>);
static_assert(!rocwmma::is_arithmetic_v<rocwmma::float8_t&>);
"""
        self.succeeds(compile_source(
            PUBLIC + checks("rocwmma", True) + values + controls, (MODE + "=0",)
        ))

    def test_no_standard_specializations(self):
        values = "\n".join(
            f"static_assert(!std::{trait}_v<{typ}>);"
            for typ in TYPES for trait in TRAITS
        )
        self.succeeds(compile_source(PUBLIC + checks("std", False) + values, (MODE + "=0",)))

    def test_hip_headers_first(self):
        source = ("#include <hip/hip_fp8.h>\n#include <hip/hip_fp16.h>\n"
                  + PUBLIC + checks("rocwmma", True))
        self.succeeds(compile_source(source, (MODE + "=0",)))

    def test_invalid_setting(self):
        self.fails(compile_source(PUBLIC, (MODE + "=2",)), "must be 0 or 1")

    def test_removed_legacy_rejected(self):
        for definitions in ((MODE + "=1",), (MODE + "=1", SUPPRESS + "=1")):
            with self.subTest(definitions=definitions):
                self.fails(compile_source(PUBLIC, definitions),
                           "legacy std trait specializations were removed")


if __name__ == "__main__":
    unittest.main(argv=[__file__], verbosity=2)
