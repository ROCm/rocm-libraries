# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Python-driven native lowering uses one Python-selected compiler flavor."""

import os
import shutil
import subprocess
import sys
from unittest.mock import Mock

import pytest
from rocke.core import lower_llvm
from rocke.core.ir import IRBuilder
from rocke.core.ir_serialize import serialize
from rocke.runtime.comgr import ComgrError


@pytest.fixture
def engine():
    return pytest.importorskip("rocke_engine")


def test_bindings_use_python_compiler(engine, tmp_path):
    if sys.platform != "linux" or not shutil.which("cc"):
        pytest.skip("ELF compiler fixtures require Linux and a C compiler")
    for owner, major in (("native", 20), ("torch", 23)):
        root = tmp_path / owner / "lib"
        root.mkdir(parents=True)
        source = (
            "void LLVMGetVersion(unsigned*a,unsigned*b,unsigned*c){*a=%d;*b=0;*c=0;}"
            % major
        )
        subprocess.run(
            [
                "cc",
                "-shared",
                "-fPIC",
                "-x",
                "c",
                "-",
                "-o",
                str(root / "libamd_comgr.so"),
            ],
            input=source,
            capture_output=True,
            text=True,
            check=True,
        )
    env = dict(
        os.environ,
        ROCM_PATH=str(tmp_path / "native"),
        PYTHONPATH=os.pathsep.join(sys.path),
    )
    for key in ("ROCKE_LLVM_FLAVOR", "ROCKE_COMGR_LIB", "ROCM_HOME"):
        env.pop(key, None)
    probe = """
import sys
from types import SimpleNamespace
from rocke.core import lower_llvm
from rocke.core.ir import IRBuilder
from rocke.core.ir_serialize import serialize
import rocke_engine
sys.modules["torch"] = SimpleNamespace(__file__=sys.argv[1], version=SimpleNamespace(hip="99.0"))
assert lower_llvm._resolve_llvm_flavor() == "llvm23"
for text in (rocke_engine.layernorm2d_lower_llvm({}),
             rocke_engine.lower_serialized_ir(serialize(IRBuilder("compiler_probe").kernel))):
    assert lower_llvm._datalayout_for_flavor("llvm23") in text
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", probe, str(tmp_path / "torch/__init__.py")],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "endpoint",
    [
        "gemm",
        "layernorm2d",
        "fmha_mfma",
        "attention_unified",
        "fmha_varlen",
        "moe_gemm_fused",
        "sparse_attention",
        "gfx950_attention_tiled_3d",
        "fused_moe_e2e",
    ],
)
def test_resolver_failure_precedes_build(engine, monkeypatch, endpoint):
    resolve = Mock(side_effect=ComgrError("compiler unavailable"))
    monkeypatch.setattr(lower_llvm, "_resolve_llvm_flavor", resolve)
    with pytest.raises(ComgrError, match="compiler unavailable"):
        getattr(engine, endpoint + "_lower_llvm")({}, arch="gfx950")
    resolve.assert_called_once_with()


def test_dictionary_errors_precede_resolution(engine, monkeypatch):
    resolve = Mock(side_effect=AssertionError("unexpected resolution"))
    monkeypatch.setattr(lower_llvm, "_resolve_llvm_flavor", resolve)
    with pytest.raises((TypeError, RuntimeError), match="cast|convert|incompatible"):
        engine.layernorm2d_lower_llvm({"n_per_block": object()})
    resolve.assert_not_called()


@pytest.mark.parametrize(
    "endpoint,layout",
    [
        ("gfx942_attention_tiled_3d", "paged"),
        ("gfx942_attention_tiled_3d", "strided"),
        ("gfx950_attention_tiled_3d", "paged"),
        ("gfx950_attention_tiled_3d", "strided"),
        ("fmha_splitkv_decode", "paged"),
        ("fused_moe_e2e", "paged"),
    ],
)
def test_composite_resolves_once(engine, monkeypatch, endpoint, layout):
    resolve = Mock(side_effect=["llvm23", AssertionError("resolved twice")])
    monkeypatch.setattr(lower_llvm, "_resolve_llvm_flavor", resolve)
    spec = dict(
        head_size=64,
        block_size=16,
        num_query_heads=8,
        num_kv_heads=2,
        num_segments=2,
        num_seqs=1,
        dtype="fp16",
        kv_layout=layout,
        phase="both",
    )
    modules = 1 if endpoint.startswith("gfx942") else 2
    if endpoint == "fused_moe_e2e":
        spec = dict(
            tokens=1, experts=8, topk=2, hidden=4096, intermediate=7168, dtype="f16"
        )
        modules = 3
    text = getattr(engine, endpoint + "_lower_llvm")(
        spec, arch=endpoint[:6] if endpoint.startswith("gfx") else "gfx950"
    )
    assert text.count(lower_llvm._datalayout_for_flavor("llvm23")) == modules
    if "attention_tiled" in endpoint:
        assert ("k_stride_batch_bytes" in text) == (layout == "strided")
    resolve.assert_called_once_with()


def test_serialized_errors_and_explicit_emission(engine, monkeypatch):
    resolve = Mock(side_effect=ComgrError("compiler unavailable"))
    monkeypatch.setattr(lower_llvm, "_resolve_llvm_flavor", resolve)
    for text, message in (
        ("", "empty IR input"),
        ("not serialized IR", "parse failed"),
    ):
        with pytest.raises(RuntimeError, match=message):
            engine.lower_serialized_ir(text)
    with pytest.raises(RuntimeError, match="unknown LLVM flavor"):
        engine.lower_serialized_ir("not serialized IR", flavor="invalid")
    resolve.assert_not_called()
    ir = serialize(IRBuilder("offline_probe").kernel)
    with pytest.raises(ComgrError, match="compiler unavailable"):
        engine.lower_serialized_ir(ir)
    resolve.assert_called_once_with()
    resolve.reset_mock()
    assert engine.lower_serialized_ir(ir, flavor="llvm20")
    resolve.assert_not_called()
    monkeypatch.setattr(lower_llvm, "_resolve_llvm_flavor", lambda: "llvm20")
    with pytest.raises(RuntimeError, match="build failed"):
        engine.layernorm2d_lower_llvm({"dtype": "invalid"})
