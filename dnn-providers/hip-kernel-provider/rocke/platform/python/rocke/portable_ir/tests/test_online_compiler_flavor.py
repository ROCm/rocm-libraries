# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Owner-profile controls run under pytest and the portable unittest entrypoint."""
import ctypes
import os
import unittest
from unittest.mock import Mock, patch

from rocke.core import ir_export
from rocke.core.ir import IRBuilder
from rocke.portable_ir.examples import recipe_toy
from rocke.portable_ir.src import online, recipe_bundle
from rocke.runtime import comgr

ROUTES = ("recipe", "bundle", "json")


def convert(route, **kwargs):
    recipe = recipe_toy.make_recipe()
    specs = dict(arch="gfx950", ints={"D": 4}, strs={"dtype": "f32"})
    if route == "recipe":
        return online.recipe_cbor_to_llvm(
            recipe_bundle.cbor_encode(recipe), **specs, **kwargs
        )[0]
    if route == "bundle":
        bundle = recipe_bundle.build_bundle(
            [{"key": "toy", "arch": "gfx950", "recipe": recipe}], stamp_abi=False
        )
        return online.bundle_cbor_to_llvm(
            recipe_bundle.cbor_encode(bundle), "toy", **specs, **kwargs
        )[0]
    text = ir_export.export_kernel_ir_json(IRBuilder("owner_profile").kernel)
    return online.ir_json_to_llvm(text, arch="gfx950", **kwargs)[0]


class TestOnlineCompilerFlavor(unittest.TestCase):
    def setUp(self):
        environment = patch.dict(os.environ)
        environment.start()
        self.addCleanup(environment.stop)
        os.environ.pop("ROCKE_LLVM_FLAVOR", None)

    def require_engine(self):
        if online.resolve_library_path(build=False) is None:
            self.skipTest("no shared librocke; supply ROCKE_ONLINE_LIB")

    def test_auto_uses_owner_without_environment(self):
        self.require_engine()
        for major, flavor in (
            (20, "llvm20"),
            (22, "llvm22"),
            (23, "llvm23"),
            (24, "llvm24"),
        ):
            for route in ROUTES:
                with self.subTest(major=major, route=route):
                    info = comgr.CompilerInfo(
                        (major, 0, 0), "owner fixture", None, "selected", "selected"
                    )
                    query = Mock(return_value=info)
                    with patch.object(comgr, "_loaded_compiler_info", query):
                        expected = convert(route, llvm_flavor=flavor)
                        query.assert_not_called()
                        self.assertEqual(convert(route), expected)
                        query.assert_called_once_with(required=True)
                        self.assertNotIn("ROCKE_LLVM_FLAVOR", os.environ)

    def test_explicit_profile_is_compiler_free(self):
        self.require_engine()
        for route in ROUTES:
            with self.subTest(route=route):
                with patch.object(
                    comgr,
                    "_loaded_compiler_info",
                    Mock(side_effect=AssertionError("compiler query")),
                ):
                    self.assertTrue(convert(route, llvm_flavor="llvm24"))
                    with self.assertRaisesRegex(ValueError, "flavor"):
                        convert(route, llvm_flavor="invalid")
                self.assertNotIn("ROCKE_LLVM_FLAVOR", os.environ)

    def test_unavailable_owner_does_not_fall_back(self):
        for route in ROUTES:
            with self.subTest(route=route):
                # The unavailable owner must refuse before any native call;
                # an inert handle keeps this refusal test engine-independent.
                lib = Mock()
                with patch.object(
                    comgr, "_loaded_compiler_info", Mock(return_value=None)
                ), patch.object(online, "load", return_value=lib):
                    with self.assertRaisesRegex(
                        comgr.ComgrError, "Cannot determine LLVM flavor"
                    ):
                        convert(route)
                    self.assertEqual(lib.method_calls, [])

    def test_old_engine_missing_owner_exports_is_actionable(self):
        class LegacyOnly:
            def __getattr__(self, name):
                if name.endswith("_ex"):
                    raise AttributeError(name)
                return Mock()

        with self.assertRaisesRegex(RuntimeError, "owner-flavor exports"):
            online._bind(LegacyOnly())

    def test_legacy_json_export_still_consumes_native_environment(self):
        self.require_engine()
        os.environ["ROCKE_LLVM_FLAVOR"] = "llvm20"
        lib = online.load()
        text = ir_export.export_kernel_ir_json(IRBuilder("legacy_profile").kernel)
        out = ctypes.c_char_p()
        build, lower = ctypes.c_double(), ctypes.c_double()
        error = ctypes.create_string_buffer(4096)
        status = lib.rocke_online_ir_json_to_llvm(
            text.encode(),
            b"gfx950",
            ctypes.byref(out),
            ctypes.byref(build),
            ctypes.byref(lower),
            error,
            len(error),
        )
        self.assertEqual(status, 0, error.value)
        try:
            self.assertEqual(
                out.value.decode(),
                convert("json", llvm_flavor="llvm20").replace(
                    "owner_profile", "legacy_profile"
                ),
            )
        finally:
            lib.rocke_online_free(out)
        self.assertEqual(os.environ["ROCKE_LLVM_FLAVOR"], "llvm20")

    def test_legacy_exports_keep_signatures(self):
        self.require_engine()
        lib = online.load()
        for name, size in (("recipe", 12), ("bundle", 13), ("ir_json", 7)):
            stem = (
                "rocke_online_"
                + name
                + ("_cbor" if name != "ir_json" else "")
                + "_to_llvm"
            )
            old, explicit = getattr(lib, stem), getattr(lib, stem + "_ex")
            self.assertEqual(len(old.argtypes), size)
            self.assertEqual(len(explicit.argtypes), size + 1)
            self.assertIs(old.restype, ctypes.c_int)
            self.assertIs(explicit.restype, ctypes.c_int)


if __name__ == "__main__":
    unittest.main()
