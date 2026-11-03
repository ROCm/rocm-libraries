# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Portable Python calls pass the owner profile, never a process-env mutation."""
import ctypes
import os
from unittest.mock import Mock

import pytest
from rocke.core import ir_export
from rocke.core.ir import IRBuilder
from rocke.portable_ir.examples import recipe_toy
from rocke.portable_ir.src import online, recipe_bundle
from rocke.runtime import comgr


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


@pytest.mark.parametrize("route", ["recipe", "bundle", "json"])
@pytest.mark.parametrize(
    "major,flavor", [(20, "llvm20"), (22, "llvm22"), (23, "llvm23"), (24, "llvm24")]
)
def test_online_auto_uses_owner_without_environment(route, major, flavor, monkeypatch):
    monkeypatch.delenv("ROCKE_LLVM_FLAVOR", raising=False)
    info = comgr.CompilerInfo(
        (major, 0, 0), "owner fixture", None, "selected", "selected"
    )
    query = Mock(return_value=info)
    monkeypatch.setattr(comgr, "_loaded_compiler_info", query)
    expected = convert(route, llvm_flavor=flavor)
    query.assert_not_called()
    assert convert(route) == expected
    query.assert_called_once_with(required=True)
    assert "ROCKE_LLVM_FLAVOR" not in os.environ


@pytest.mark.parametrize("route", ["recipe", "bundle", "json"])
def test_online_explicit_profile_is_compiler_free(route, monkeypatch):
    monkeypatch.delenv("ROCKE_LLVM_FLAVOR", raising=False)
    monkeypatch.setattr(
        comgr,
        "_loaded_compiler_info",
        Mock(side_effect=AssertionError("compiler query")),
    )
    assert convert(route, llvm_flavor="llvm24")
    with pytest.raises(ValueError, match="flavor"):
        convert(route, llvm_flavor="invalid")
    assert "ROCKE_LLVM_FLAVOR" not in os.environ


@pytest.mark.parametrize("route", ["recipe", "bundle", "json"])
def test_online_unavailable_owner_does_not_fall_back(route, monkeypatch):
    monkeypatch.delenv("ROCKE_LLVM_FLAVOR", raising=False)
    monkeypatch.setattr(comgr, "_loaded_compiler_info", Mock(return_value=None))
    with pytest.raises(comgr.ComgrError, match="Cannot determine LLVM flavor"):
        convert(route)


def test_old_engine_missing_owner_exports_is_actionable():
    class LegacyOnly:
        def __getattr__(self, name):
            if name.endswith("_ex"):
                raise AttributeError(name)
            return Mock()

    with pytest.raises(RuntimeError, match="owner-flavor exports"):
        online._bind(LegacyOnly())


def test_legacy_json_export_still_consumes_native_environment(monkeypatch):
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", "llvm20")
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
    assert status == 0, error.value
    try:
        assert out.value.decode() == convert("json", llvm_flavor="llvm20").replace(
            "owner_profile", "legacy_profile"
        )
    finally:
        lib.rocke_online_free(out)
    assert os.environ["ROCKE_LLVM_FLAVOR"] == "llvm20"


def test_online_legacy_exports_keep_signatures():
    lib = online.load()
    for name, size in [("recipe", 12), ("bundle", 13), ("ir_json", 7)]:
        stem = (
            "rocke_online_" + name + ("_cbor" if name != "ir_json" else "") + "_to_llvm"
        )
        old, explicit = getattr(lib, stem), getattr(lib, stem + "_ex")
        assert len(old.argtypes) == size
        assert len(explicit.argtypes) == size + 1
        assert old.restype is explicit.restype is ctypes.c_int
