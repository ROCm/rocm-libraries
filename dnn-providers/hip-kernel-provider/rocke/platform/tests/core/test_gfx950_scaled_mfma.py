# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Complete scaled MFMA contracts, logical layouts and dual-engine emission."""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from itertools import product
from pathlib import Path

import pytest
from rocke.core.arch import ArchTarget
from rocke.core.arch.mfma_scale import matrix_layout
from rocke.core.ir import IRBuilder
from rocke.core.ir_serialize import serialize
from rocke.core.lower_hip import lower_kernel_to_hip
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.core.verify import verify
from rocke.instances.gfx950.block_scaled_gemm import (
    BlockScaledGemmSpec,
    build_block_scaled_gemm,
    is_valid_spec,
)

FORMATS = ("fp8", "bf8", "fp6", "bf6", "fp4")


class Arithmetic:
    def const_i32(self, x):
        return x

    def mod(self, a, b):
        return a % b

    def div(self, a, b):
        return a // b

    def mul(self, a, b):
        return a * b

    def add(self, a, b):
        return a + b


@pytest.mark.parametrize("a,b", list(product(FORMATS, repeat=2)))
def test_catalog_and_logical_layouts(a, b):
    atom = ArchTarget.from_gfx("gfx950").mma.op_for_shape(
        family="mfma_scaled",
        a_dtype=a,
        b_dtype=b,
        c_dtype="fp32",
        scales=("e8m0", "e8m0", 32),
        m=32,
        n=32,
        k=64,
    )
    assert (atom.a_frag_len, atom.b_frag_len, atom.c_frag_len, atom.wave_size) == (
        32,
        32,
        16,
        64,
    )
    cpu = Arithmetic()
    for layout, shape in (
        (atom.a_layout(), (32, 64)),
        (atom.b_layout(), (64, 32)),
        (atom.c_layout(), (32, 32)),
        (atom.a_scale_layout(), (32, 2)),
        (atom.b_scale_layout(), (2, 32)),
    ):
        points = [
            layout.coord(cpu, l, s) for l in range(64) for s in range(layout.frag_len)
        ]
        assert len(points) == len(set(points)) == shape[0] * shape[1]
        assert set(points) == set(product(range(shape[0]), range(shape[1])))
    for dtype in (atom.a_dtype, atom.b_dtype):
        packed = matrix_layout(dtype)
        assert packed.fragment.count == 32 and packed.fragment.carrier_count == 8
        assert packed.fragment.live_carriers in (4, 6, 8)
    for target in ("gfx942", "gfx1250", "gfx1201"):
        assert ArchTarget.from_gfx(target).mma.by_op_id(atom.op_id) is None


@pytest.mark.parametrize("idx", range(50))
@pytest.mark.parametrize("flavor", ["llvm20", "llvm22", "llvm23"])
def test_complete_kernel_golden_and_native_identity(idx, flavor, monkeypatch):

    spec = BlockScaledGemmSpec(
        "scaled_parity",
        32 if idx < 25 else 64,
        64 if idx < 25 else 96,
        128 if idx < 25 else 256,
        FORMATS[(idx % 25) // 5],
        FORMATS[idx % 5],
        "bf16" if idx < 25 else "fp16",
    )
    kernel = build_block_scaled_gemm(spec)
    assert not verify(kernel)
    llvm = _lower_kernel_to_llvm_python(kernel, arch="gfx950", llvm_flavor=flavor)
    calls = [
        line
        for line in llvm.splitlines()
        if "call <16 x float> @llvm.amdgcn.mfma.scale" in line
    ]
    assert len(calls) == spec.K // 64
    for line in calls:
        assert re.search(rf", i32 {(idx % 25) // 5}, i32 {idx % 5}, i32 0, i32 %", line)
    if flavor == "llvm23":
        golden = json.loads(
            Path(__file__).with_name("gfx950_scaled_mfma_llvm23.json").read_text()
        )
        assert hashlib.sha256(llvm.encode()).hexdigest() == golden[str(idx)]
    engine = pytest.importorskip("rocke_engine")
    assert engine.gfx950_block_scaled_gemm_serialize_ir(
        asdict(spec), "gfx950"
    ) == serialize(kernel)
    assert engine.lower_serialized_ir(serialize(kernel), "gfx950", flavor) == llvm
    monkeypatch.setenv("ROCKE_LLVM_FLAVOR", flavor)
    assert engine.gfx950_block_scaled_gemm_lower_llvm(asdict(spec), "gfx950") == llvm
    assert engine.gfx950_block_scaled_gemm_verify(asdict(spec), "gfx950") == []


@pytest.mark.parametrize("arch", ["gfx942", "gfx1201", "gfx1250"])
def test_wrong_target_rejection(arch):
    k = build_block_scaled_gemm(BlockScaledGemmSpec("reject", 32, 32, 64))
    with pytest.raises(NotImplementedError, match="requires gfx950"):
        _lower_kernel_to_llvm_python(k, arch=arch, llvm_flavor="llvm23")
    with pytest.raises(NotImplementedError, match="requires gfx950"):
        lower_kernel_to_hip(k, arch=arch)
    engine = pytest.importorskip("rocke_engine")
    with pytest.raises(RuntimeError, match="requires gfx950"):
        engine.lower_serialized_ir(serialize(k), arch, "llvm23")


def test_native_lowering_preserves_diagnostic(tmp_path):
    pytest.importorskip("rocke_engine")
    compiler = shutil.which("cc")
    if sys.platform != "linux" or not compiler:
        pytest.skip("ELF loader fixture requires Linux and a C compiler")
    source = tmp_path / "unqueryable.c"
    library = tmp_path / "unqueryable.so"
    source.write_text("void unqueryable_compiler(void) {}\n")
    subprocess.run(
        [compiler, "-shared", "-fPIC", str(source), "-o", str(library)],
        check=True,
        capture_output=True,
        text=True,
    )
    env = dict(os.environ)
    env.pop("ROCKE_LLVM_FLAVOR", None)
    env["ROCKE_COMGR_LIB"] = str(library)
    env["PYTHONPATH"] = os.pathsep.join(sys.path)
    # A loadable, unqueryable fixture prevents fallback to an installed COMGR.
    # A fresh process also avoids a previously cached compiler masking failure.
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            "import rocke_engine; "
            "rocke_engine.gfx950_block_scaled_gemm_lower_llvm("
            "{'name': 'diagnostic', 'M': 32, 'N': 32, 'K': 64}, 'gfx950')",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert child.returncode != 0
    assert "Cannot determine LLVM flavor" in child.stderr
    assert "unknown error" not in child.stderr


@pytest.mark.parametrize("name_bytes", [255, 256, 512])
def test_long_native_kernel_names(name_bytes):
    engine = pytest.importorskip("rocke_engine")
    empty = BlockScaledGemmSpec("", 32, 32, 64)
    spec = BlockScaledGemmSpec(
        "x" * (name_bytes - len(empty.kernel_name())), 32, 32, 64
    )
    assert len(spec.kernel_name().encode("ascii")) == name_bytes
    assert is_valid_spec(spec)[0]
    kernel = build_block_scaled_gemm(spec)
    native_ir = engine.gfx950_block_scaled_gemm_serialize_ir(asdict(spec), "gfx950")
    assert native_ir == serialize(kernel)
    assert engine.gfx950_block_scaled_gemm_verify(asdict(spec), "gfx950") == []
    for flavor in ("llvm20", "llvm22", "llvm23"):
        assert engine.lower_serialized_ir(native_ir, "gfx950", flavor) == (
            _lower_kernel_to_llvm_python(kernel, arch="gfx950", llvm_flavor=flavor)
        )


def test_builder_and_serialized_type_rejection():
    atom = next(
        x for x in ArchTarget.from_gfx("gfx950").mma.ops if x.family == "mfma_scaled"
    )
    ir = IRBuilder("bad")
    a = ir.zero_vec_f32(8)
    c = ir.zero_vec_f32(16)
    scale = ir.const_i32(127)
    with pytest.raises(ValueError, match="scaled MFMA requires"):
        ir.mma(atom, a, a, c, scale, scale)
    k = build_block_scaled_gemm(BlockScaledGemmSpec("bad", 32, 32, 64))
    op = next(x for x in k.body.ops if x.name == "tile.mma")
    op.operands[3] = c
    assert any("scaled MFMA requires" in str(d) for d in verify(k))
    with pytest.raises(ValueError, match="scaled MFMA requires"):
        _lower_kernel_to_llvm_python(k, arch="gfx950", llvm_flavor="llvm23")


@pytest.mark.parametrize(
    "M,N,K,block_k",
    [
        (16, 32, 64, 32),
        (32, 48, 64, 32),
        (32, 32, 96, 32),
        (32, 32, 64, 16),
        (2**30, 32, 64, 32),
    ],
)
def test_invalid_shape(M, N, K, block_k):
    s = BlockScaledGemmSpec("bad", M, N, K, block_k=block_k)
    assert not is_valid_spec(s)[0]
    with pytest.raises(ValueError):
        build_block_scaled_gemm(s)


@pytest.mark.parametrize("mutation", ["payload", "acc", "scale", "arity", "result"])
def test_serialized_native_contract_rejection(mutation):
    from rocke.core.ir import F32, VectorType

    k = build_block_scaled_gemm(BlockScaledGemmSpec("malformed", 32, 32, 64))
    op = next(x for x in k.body.ops if x.name == "tile.mma")
    if mutation == "arity":
        op.operands.pop()
    elif mutation == "result":
        op.results[0].type = VectorType(F32, 8)
    else:
        index = {"payload": 0, "acc": 2, "scale": 3}[mutation]
        op.operands[index] = op.operands[3] if mutation != "scale" else op.operands[2]
    assert any("scaled MFMA" in str(d) for d in verify(k))
    with pytest.raises(ValueError, match="scaled MFMA"):
        _lower_kernel_to_llvm_python(k, arch="gfx950", llvm_flavor="llvm23")
    engine = pytest.importorskip("rocke_engine")
    with pytest.raises(RuntimeError, match="scaled MFMA"):
        engine.lower_serialized_ir(serialize(k), "gfx950", "llvm23")


@pytest.mark.parametrize("dtype", ["unknown", "fp32", "FP6 ", " BF8", "fp4"])
def test_native_spec_format_validation(dtype):
    engine = pytest.importorskip("rocke_engine")
    spec = BlockScaledGemmSpec("formats", 32, 32, 64, dtype_a=dtype)
    if dtype.strip().lower() in ("unknown", "fp32"):
        with pytest.raises(RuntimeError, match="invalid gfx950"):
            engine.gfx950_block_scaled_gemm_serialize_ir(asdict(spec), "gfx950")
    else:
        assert engine.gfx950_block_scaled_gemm_serialize_ir(
            asdict(spec), "gfx950"
        ) == serialize(build_block_scaled_gemm(spec))
