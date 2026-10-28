# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""General rules for platform instance code (probed on batched_transpose):

- a call of a rocke.* helper with a C twin (sigmap external_helpers): a str
  result through a caller buffer (kernel_name_join), a None sequence argument
  with its companions NULL / 0, owned strings into a C string array;
- such a helper that returns a tuple through out parameters and its success
  flag (block_lds_reduce_with_index);
- b.param(name, t, noalias=..., align=...) into rocke_param_opts_t;
- any IR scalar type constant of rocke.core.ir with a C singleton (FP8E4M3);
- float arithmetic with an int operand, and true division;
- a platform struct's documented C getter / predicate for a member it has
  no field for (ArchTarget.max_threads_per_block, ArchTarget.fits_lds)."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys

import pytest
from trtest import ROCKE, TR, platform_probe, strip_holes

PROBE = """\
def probe_params(spec: BatchedTranspose2DSpec) -> KernelDef:
    b = IRBuilder(spec.kernel_name())
    b.param("X", PtrType(I32, "global"), readonly=True, align=8, noalias=False)
    b.param("n", I32)
    return b.kernel


def probe_fp8(spec: BatchedTranspose2DSpec) -> int:
    from ...core.ir import FP8E4M3

    t = FP8E4M3
    return 1 if t == FP8E4M3 else 0


def probe_float(spec: BatchedTranspose2DSpec) -> float:
    return 1.0 / spec.tile_m + spec.vec * 0.5 - 2


def probe_arch(spec: BatchedTranspose2DSpec, arch: str = "gfx950") -> int:
    from ...core.arch import ArchTarget

    target = ArchTarget.from_gfx(arch)
    if target.fits_lds(spec.tile_m * 2):
        return target.max_threads_per_block
    return 0
"""

# qwen3_sampler imports block_lds_reduce_with_index (the maps list a helper
# only when the package imports it)
PROBE_SAMPLER = """\
def probe_argmax(spec: Qwen3GreedySamplerSpec) -> KernelDef:
    b = IRBuilder("probe_argmax")
    tid = b.thread_id_x()
    lds_val = b.smem_alloc(F32, [64], name_hint="v")
    lds_idx = b.smem_alloc(F32, [64], name_hint="i")
    val, idx = block_lds_reduce_with_index(
        b, b.const_f32(1.0), tid, lds_val, lds_idx, tid, block_size=64, combine="argmin"
    )
    b.add(idx, tid)
    return b.kernel
"""


def _run(env, abi, out=None, m="batched_transpose"):
    """tr.abi, then tr.translate (--out when given, else --analyze)."""
    run = None
    for args in (
        ["tr.abi", m, "--header", str(abi)],
        ["tr.translate", m, "--abi", str(abi)]
        + (["--out", str(out)] if out else ["--analyze"]),
    ):
        run = subprocess.run(
            [sys.executable, "-m", *args, "--rocke", str(ROCKE)],
            cwd=TR,
            env=env,
            capture_output=True,
            text=True,
        )
    return run.stdout + run.stderr


@pytest.fixture(scope="module")
def probe(tmp_path_factory, clang_format):
    """(analyze output, source with whitespace folded, raw source, abi dir)."""
    root = tmp_path_factory.mktemp("platform_helpers")
    env = platform_probe(root, PROBE)
    abi = root / "abi"
    abi.mkdir()
    out = root / "batched_transpose.cpp"
    _run(env, abi, out)
    cpp = out.read_text()
    return _run(env, abi), re.sub(r"\s+", " ", cpp), cpp, abi


@pytest.fixture(scope="module")
def sampler(tmp_path_factory, clang_format):
    """(analyze output, source with whitespace folded) of the qwen3_sampler probe."""
    root = tmp_path_factory.mktemp("platform_helpers_sampler")
    m = "qwen3_sampler"
    env = platform_probe(root, PROBE_SAMPLER, package=m, arch="gfx1250")
    abi = root / "abi"
    abi.mkdir()
    out = root / f"{m}.cpp"
    _run(env, abi, out, m)
    return _run(env, abi, m=m), re.sub(r"\s+", " ", out.read_text())


def _body(cpp: str, fn: str) -> str:
    m = re.search(rf"\b{fn}\(.*?\{{(.*?)\}} /\* py:|\b{fn}\(.*?\{{(.*?)$", cpp)
    assert m, fn
    return m.group(1) or m.group(2)


@pytest.mark.parametrize(
    "name",
    [
        "BatchedTranspose2DSpec.kernel_name",
        "probe_params",
        "probe_fp8",
        "probe_float",
        "probe_arch",
    ],
)
def test_translates(probe, name):
    assert re.search(rf"^translated +{re.escape(name)}$", probe[0], re.M), probe[0]


def test_kernel_name_join_through_its_c_twin(probe):
    """kernel_name_join(self.name, self.dtype, f"..", ..): the parts as owned
    strings in Python's order, flags=None as NULL, NULL, 0, the str result
    through a caller buffer and its length not asked for."""
    body = _body(probe[1], "py_BatchedTranspose2DSpec_kernel_name")
    parts = re.findall(r"parts_\d+\.push_back\(([^;]*)\);", body)
    assert parts[0] == "std::string(self->dtype)", parts
    assert len(parts) == 4, parts
    assert re.search(
        r"rocke_kernel_name_join\(self->name, cstrs_\d+\.data\(\), "
        r"\(int\)py::len\(parts_\d+\), nullptr, nullptr, 0, buf_\d+, "
        r"sizeof\(buf_\d+\), nullptr\)",
        body,
    ), body
    assert 'throw ckc::ValueError("string result does not fit")' in body


def test_param_keywords_set_the_opts_fields(probe):
    body = _body(probe[1], "py_probe_params")
    sets = re.findall(r"opts_\d+\.(\w+) = ([^;]+);", body)
    assert sets == [
        ("readonly", "true"),
        ("readonly_set", "true"),
        ("align", "8"),
        ("align_set", "true"),
        ("noalias", "false"),
        ("noalias_set", "true"),
    ], sets
    assert re.search(r'rocke_b_param\(b, "X", .*, &opts_\d+\)', body), body


def test_out_parameter_helper_checks_its_flag(sampler):
    assert re.search(r"^translated +probe_argmax$", sampler[0], re.M), sampler[0]
    body = _body(sampler[1], "py_probe_argmax")
    assert re.search(
        r"if\(!rocke_block_lds_reduce_with_index\( ?b, \w+, tid, lds_val, lds_idx, "
        r"tid, 64, ROCKE_INDEX_ARGMIN, &val_\d+, &idx_\d+\)\) "
        r"py_adapt::raise_builder\(b\);",
        body,
    ), body


def test_ir_scalar_constant(probe):
    assert "rocke_fp8e4m3()" in _body(probe[1], "py_probe_fp8")


def test_float_arithmetic(probe):
    body = _body(probe[1], "py_probe_float")
    assert "py::truediv(1.0, (double)spec->tile_m)" in body, body
    assert "((double)spec->vec * 0.5)" in body, body


def test_documented_getter_and_predicate(probe):
    """target.max_threads_per_block (a property) and target.fits_lds(n) (a
    method annotated -> bool) of a platform struct with no such field: the C
    functions rocke_arch_target.h documents as ArchTarget.<name>, found
    through its "rocke_arch_target_t + rocke_arch_* () getters" row."""
    body = _body(probe[1], "py_probe_arch")
    assert re.search(r"rocke_arch_fits_lds\(target, \(long\)", body), body
    assert "return rocke_arch_max_threads_per_block(target);" in body, body


def test_source_compiles(probe, tmp_path):
    exe = shutil.which("g++")
    if exe is None:
        pytest.skip("no g++")
    cpp, abi = probe[2], probe[3]
    inc = tmp_path / "inc"
    shutil.copytree(abi, inc / "rocke")
    src = tmp_path / "src.cpp"
    src.write_text(strip_holes(cpp))
    cmd = [exe, "-std=c++20", "-fsyntax-only", "-Wall", "-Werror", "-I", str(inc)]
    cmd += ["-isystem", str(ROCKE / "platform" / "cpp" / "include")]
    cmd += ["-I", str(TR / "tr" / "prelude"), str(src)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr


def test_truediv_by_zero_raises(tmp_path):
    """py::truediv: Python's ZeroDivisionError text (raised as ValueError)."""
    exe = shutil.which("g++")
    if exe is None:
        pytest.skip("no g++")
    src = tmp_path / "t.cpp"
    src.write_text(
        '#include "rocke_py.hpp"\n#include <cstdio>\n'
        "int main() {\n"
        "  if (py::truediv(1.0, 4.0) != 0.25) return 1;\n"
        "  try { py::truediv(1.0, 0.0); } catch (const ckc::ValueError& e) {\n"
        "    std::puts(e.what()); return 0; }\n"
        "  return 2;\n}\n"
    )
    exe_out = tmp_path / "t"
    cmd = [exe, "-std=c++20", "-Wall", "-Werror", "-I", str(TR / "tr" / "prelude")]
    cmd += ["-I", str(ROCKE / "platform" / "cpp" / "include"), str(src), "-o"]
    cmd += [str(exe_out)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    r = subprocess.run([str(exe_out)], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip() == "float division by zero", r
