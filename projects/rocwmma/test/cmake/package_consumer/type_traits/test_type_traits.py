# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

"""Compile actual public headers to check defaults, compatibility and diagnostics.

Requires a HIP Clang driver, but no GPU. The compiler/STL's handling of forbidden
specializations is measured independently of rocWMMA, so these checks do not
duplicate its toolchain-selection condition.
"""

import argparse
import pathlib
import subprocess
import tempfile
import unittest


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--compiler", required=True)
parser.add_argument("--arch", default="gfx1100")
parser.add_argument("--include", required=True)
parser.add_argument("--stage", type=int, choices=(1, 2, 3), required=True)
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
        probe = compile_source(
            "#include <type_traits>\nstruct marker {};\nnamespace std {\n"
            "template<> struct is_signed<marker> : true_type {};\n}\n"
        )
        cls.legacy_supported = probe.returncode == 0
        if not cls.legacy_supported and "specializ" not in probe.stderr:
            raise RuntimeError("Unexpected STL probe failure:\n" + probe.stderr)

    def succeeds(self, result):
        self.assertEqual(result.returncode, 0, result.stderr)

    def fails(self, result, message):
        self.assertNotEqual(result.returncode, 0, "Compilation unexpectedly succeeded")
        self.assertIn(message, result.stderr)

    def test_shipped_default(self):
        expected = int(args.stage == 1 and self.legacy_supported)
        self.succeeds(compile_source(
            PUBLIC + f"static_assert({MODE} == {expected});\n" + checks("rocwmma", True)
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
        if args.stage != 3 and self.legacy_supported:
            self.succeeds(compile_source(source, (MODE + "=1",)))

    def test_invalid_setting(self):
        self.fails(compile_source(PUBLIC, (MODE + "=2",)), "must be 0 or 1")

    def test_explicit_legacy(self):
        result = compile_source(PUBLIC + checks("std", True), (MODE + "=1", SUPPRESS + "=1"))
        if args.stage == 3:
            self.fails(result, "legacy std trait specializations were removed")
        elif self.legacy_supported:
            self.succeeds(result)
        else:
            self.fails(result, "specializ")

    def require_legacy(self):
        if args.stage == 3:
            self.skipTest("Legacy implementation has been removed")
        if not self.legacy_supported:
            self.skipTest("STL rejects legacy specializations; explicit override failure checked")

    def test_legacy_does_not_warn_on_owned_use(self):
        self.require_legacy()
        self.succeeds(compile_source(PUBLIC + checks("rocwmma", True), (MODE + "=1",)))

    def test_legacy_per_use_diagnostics_and_suppression(self):
        self.require_legacy()
        source = PUBLIC + checks("std", True)
        result = compile_source(source, (MODE + "=1",))
        self.fails(result, "deprecated")
        for trait in TRAITS:
            self.assertIn("use rocwmma::" + trait, result.stderr)
        self.succeeds(compile_source(source, (MODE + "=1", SUPPRESS + "=1")))

    def test_legacy_variable_templates(self):
        self.require_legacy()
        # _v may use compiler builtins or instantiate the class trait within a
        # system header, where deprecation is suppressed. It is not a portable
        # migration diagnostic. Require valid code in either implementation;
        # the class-use test above separately requires the actionable warning.
        source = PUBLIC + "\n".join(
            f"constexpr bool value_{i}_{trait} = std::{trait}_v<{typ}>;"
            for i, typ in enumerate(TYPES) for trait in TRAITS
        )
        result = compile_source(source, (MODE + "=1",),
                                ("-Wno-error=deprecated-declarations",))
        self.succeeds(result)
        if "deprecated" in result.stderr:
            self.assertIn("use rocwmma::", result.stderr)
        self.succeeds(compile_source(source, (MODE + "=1", SUPPRESS + "=1")))

    def test_legacy_system_header_instantiation(self):
        self.require_legacy()
        with tempfile.TemporaryDirectory() as directory:
            header = pathlib.Path(directory) / "trait_wrapper.hpp"
            header.write_text(
                "#include <type_traits>\n"
                "template<class T> constexpr bool wrapper() { return std::is_signed<T>::value; }\n"
            )
            source = PUBLIC + '#include <trait_wrapper.hpp>\nstatic_assert(wrapper<rocwmma::float8_t>());\n'
            self.fails(compile_source(source, (MODE + "=1",), ("-isystem", directory)),
                       "use rocwmma::is_signed")


if __name__ == "__main__":
    unittest.main(argv=[__file__], verbosity=2)
