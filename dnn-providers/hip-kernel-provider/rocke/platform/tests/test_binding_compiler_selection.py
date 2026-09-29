# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Python-driven native lowering must use Python's compiler selection."""

import os
import shutil
import subprocess
import sys

import pytest


def build_library(root, text, name="libamd_comgr.so", extra=()):
    root.mkdir(parents=True, exist_ok=True)
    source = root / (name + ".c")
    source.write_text(text)
    output = root / name
    subprocess.run(
        ["cc", "-shared", "-fPIC", str(source), *extra, "-o", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )
    return output


BINDING_PROBE = r"""
import sys
from types import SimpleNamespace
from rocke.core.backend import lower_universal_gemm
from rocke.core.ir import IRBuilder
from rocke.core.ir_serialize import serialize
from rocke.core import lower_llvm
from rocke.instances import TileSpec, TraitSpec, UniversalGemmSpec
from rocke.runtime import comgr
import rocke_engine

# A bundled compiler and an independently discoverable native candidate differ.
sys.modules["torch"] = SimpleNamespace(
    __file__=sys.argv[1], version=SimpleNamespace(hip="99.0"))
assert lower_llvm._resolve_llvm_flavor() == "llvm23"
spec = UniversalGemmSpec(
    name="binding_compiler_probe",
    tile=TileSpec(tile_m=128, tile_n=128, tile_k=32,
                  warp_m=2, warp_n=2, warp_k=1,
                  warp_tile_m=32, warp_tile_n=32, warp_tile_k=16),
    trait=TraitSpec(pipeline="compv4", epilogue="cshuffle"))
result = lower_universal_gemm(spec, arch="gfx950", backend="both")
comgr._assert_ir_flavor_matches_lib(result.llvm_text)
assert lower_llvm._datalayout_for_flavor("llvm23") in result.llvm_text

# The direct serialized-IR binding also delegates AUTO, but explicit emission
# does not need to query or load a compiler.
b = IRBuilder("binding_offline_probe")
ir = serialize(b.kernel)
auto = rocke_engine.lower_serialized_ir(ir, arch="gfx950")
assert lower_llvm._datalayout_for_flavor("llvm23") in auto

def unexpected_query():
    raise AssertionError("explicit emission queried the compiler")

lower_llvm._resolve_llvm_flavor = unexpected_query
explicit = rocke_engine.lower_serialized_ir(ir, arch="gfx950", flavor="llvm20")
assert lower_llvm._datalayout_for_flavor("llvm20") in explicit
"""


def test_python_bindings_use_the_compiling_library(tmp_path):
    pytest.importorskip("rocke_engine")
    if sys.platform != "linux" or not shutil.which("cc"):
        pytest.skip("ELF loader fixtures require Linux and a C compiler")
    source = "void LLVMGetVersion(unsigned*a,unsigned*b,unsigned*c){*a=%d;*b=0;*c=0;}"
    build_library(tmp_path / "native" / "lib", source % 20)
    metadata = tmp_path / "native" / ".info"
    metadata.mkdir()
    (metadata / "version").write_text("7.1.0\n")
    build_library(tmp_path / "torch" / "lib", source % 23)
    env = dict(os.environ)
    for name in ("ROCKE_LLVM_FLAVOR", "ROCKE_COMGR_LIB", "ROCM_HOME"):
        env.pop(name, None)
    env.update(ROCM_PATH=str(tmp_path / "native"), ROCKE_BACKEND="python")
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    result = subprocess.run(
        [sys.executable, "-c", BINDING_PROBE, str(tmp_path / "torch" / "__init__.py")],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
