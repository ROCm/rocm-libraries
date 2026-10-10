# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Python-driven native lowering must use Python's compiler selection."""

import os
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

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

# Spec bindings added alongside the existing families share the same selection.
conv_spec = {"problem": {"N": 1, "H": 16, "W": 32, "cpg": 16, "kpg": 128}}
conv = rocke_engine.conv_direct_nongrouped_lower_llvm(conv_spec, arch="gfx950")
assert lower_llvm._datalayout_for_flavor("llvm23") in conv

# Paged and strided attention bindings use the same compiler, including both
# modules emitted by the gfx950 segment/reduce path.
attention_spec = {"head_size": 64, "block_size": 16, "num_query_heads": 8,
                  "num_kv_heads": 2, "num_segments": 2, "num_seqs": 1,
                  "dtype": "fp16"}
for arch in ("gfx942", "gfx950"):
    for layout in ("paged", "strided"):
        text = getattr(rocke_engine, arch + "_attention_tiled_3d_lower_llvm")(
            dict(attention_spec, kv_layout=layout), arch=arch)
        expected_modules = 2 if arch == "gfx950" else 1
        assert text.count(lower_llvm._datalayout_for_flavor("llvm23")) == expected_modules
        assert ("k_stride_batch_bytes" in text) == (layout == "strided")

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


CLEANUP_PROBE = r"""
from rocke.core import lower_llvm
from rocke.core.ir import IRBuilder
from rocke.core.ir_serialize import serialize
from rocke.runtime.comgr import ComgrError
import rocke_engine

def unavailable():
    raise ComgrError("cleanup probe: compiler unavailable")

lower_llvm._resolve_llvm_flavor = unavailable
common = dict(head_size=64, num_query_heads=4, num_kv_heads=2,
              dtype="fp16", seqlen_q=16, seqlen_k=64, block_size=16,
              total_q=16, max_seqlen_q=16, max_seqlen_k=64, num_seqs=1)
attention = dict(common, block_size=16, num_segments=2, num_seqs=1)
moe = dict(dtype="f16", tile=dict(tile_m=32, tile_n=128, tile_k=64,
           warp_m=1, warp_n=2, warp_tile_m=32, warp_tile_n=32, warp_tile_k=16))
cases = [
    (rocke_engine.layernorm2d_lower_llvm, {}, "gfx950"),
    (rocke_engine.rmsnorm2d_lower_llvm, {}, "gfx950"),
    (rocke_engine.fmha_mfma_lower_llvm, dict(common, dtype="f16"), "gfx950"),
    (rocke_engine.attention_unified_lower_llvm, common, "gfx950"),
    (rocke_engine.fmha_bwd_lower_llvm, common, "gfx950"),
    (rocke_engine.fmha_head_grouping_lower_llvm, common, "gfx950"),
    (rocke_engine.fmha_appendkv_lower_llvm, common, "gfx950"),
    (rocke_engine.fmha_paged_prefill_lower_llvm, common, "gfx950"),
    (rocke_engine.fmha_varlen_lower_llvm,
     dict(common, max_seqlen_q=16, max_seqlen_k=64), "gfx950"),
    (rocke_engine.moe_gemm_fused_lower_llvm, moe, "gfx950"),
]
for kind in ("jenga", "vsa"):
    cases.append((rocke_engine.sparse_attention_lower_llvm,
                  dict(common, kind=kind), "gfx950"))
for arch in ("gfx942", "gfx950"):
    for layout in ("paged", "strided"):
        cases.append((getattr(rocke_engine, arch + "_attention_tiled_3d_lower_llvm"),
                      dict(attention, kv_layout=layout), arch))
for lower, spec, arch in cases:
    for _ in range(2):
        try:
            lower(spec, arch=arch)
        except ComgrError as exc:
            assert str(exc) == "cleanup probe: compiler unavailable"
        else:
            raise AssertionError(lower.__name__ + " did not propagate resolver failure")

# A failure in the second gfx950 stage must release its reduce builder too.
calls = 0
def fail_reduce():
    global calls
    calls += 1
    if calls == 1:
        return "llvm23"
    return unavailable()

lower_llvm._resolve_llvm_flavor = fail_reduce
try:
    rocke_engine.gfx950_attention_tiled_3d_lower_llvm(attention, arch="gfx950")
except ComgrError:
    assert calls == 2
else:
    raise AssertionError("reduce resolver failure did not propagate")

# Keep spec validation ahead of automatic compiler selection.
lower_llvm._resolve_llvm_flavor = unavailable
try:
    rocke_engine.layernorm2d_lower_llvm(dict(dtype="invalid"), arch="gfx950")
except ComgrError:
    raise AssertionError("compiler lookup preceded invalid-spec rejection")
except RuntimeError as exc:
    assert "build failed" in str(exc)
else:
    raise AssertionError("invalid spec was accepted")

# Dictionary conversion also retains precedence, including MFMA's early resolver.
for lower, spec in ((rocke_engine.layernorm2d_lower_llvm, dict(n_per_block=object())),
                    (rocke_engine.fmha_mfma_lower_llvm, dict(head_size=object()))):
    try:
        lower(spec, arch="gfx950")
    except ComgrError:
        raise AssertionError("compiler lookup preceded dictionary conversion")
    except (TypeError, RuntimeError):
        pass
    else:
        raise AssertionError("invalid dictionary value was accepted")

# MFMA has no returned builder owner, so resolution must precede its allocation.
try:
    rocke_engine.fmha_mfma_lower_llvm(dict(common, dtype="invalid"), arch="gfx950")
except ComgrError as exc:
    assert str(exc) == "cleanup probe: compiler unavailable"
else:
    raise AssertionError("MFMA compiler failure did not precede native build")

# Serialized input validation must not depend on compiler availability.
for text, message in (("", "empty IR input"), ("not serialized IR", "parse failed")):
    try:
        rocke_engine.lower_serialized_ir(text, arch="gfx950")
    except ComgrError:
        raise AssertionError("compiler lookup preceded serialized-input rejection")
    except RuntimeError as exc:
        assert message in str(exc)
    else:
        raise AssertionError("invalid serialized input was accepted")

try:
    rocke_engine.lower_serialized_ir("not serialized IR", flavor="invalid")
except RuntimeError as exc:
    assert "unknown LLVM flavor" in str(exc)
else:
    raise AssertionError("invalid explicit flavor was accepted")

ir = serialize(IRBuilder("cleanup_explicit_probe").kernel)
for _ in range(2):
    try:
        rocke_engine.lower_serialized_ir(ir, arch="gfx950")
    except ComgrError as exc:
        assert str(exc) == "cleanup probe: compiler unavailable"
    else:
        raise AssertionError("serialized-IR resolver failure did not propagate")

assert rocke_engine.lower_serialized_ir(ir, arch="gfx950", flavor="llvm20")
"""


def test_binding_resolver_failure_preserves_error_contract():
    pytest.importorskip("rocke_engine")
    result = subprocess.run(
        [sys.executable, "-B", "-c", CLEANUP_PROBE],
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr


def test_resolver_failure_releases_binding_builders(tmp_path):
    pytest.importorskip("rocke_engine")
    valgrind = shutil.which("valgrind")
    if sys.platform != "linux" or not valgrind:
        pytest.skip("builder leak detection requires Linux and Valgrind")
    report = tmp_path / "leaks.xml"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), TMPDIR=str(tmp_path))
    result = subprocess.run(
        [
            valgrind,
            "--vgdb=no",
            "--leak-check=full",
            "--show-leak-kinds=definite",
            "--xml=yes",
            "--xml-file=" + str(report),
            sys.executable,
            "-B",
            "-c",
            CLEANUP_PROBE,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    # Python and third-party shutdown reports are outside this ownership gate.
    # Require no lost arena allocation from any exercised rocKE binding.
    leaks = [
        error
        for error in ET.parse(report).findall("error")
        if error.findtext("kind") == "Leak_DefinitelyLost"
        and any(
            "rocke_arena" in (frame.findtext("fn") or "")
            for frame in error.findall("stack/frame")
        )
    ]
    assert not leaks, "\n".join(
        ET.tostring(error, encoding="unicode") for error in leaks
    )
