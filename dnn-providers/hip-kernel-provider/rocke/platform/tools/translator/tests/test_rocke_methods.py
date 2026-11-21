# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""General rules for the rocke.* classes a helper returns (probed on
batched_transpose):

- a method of a rocke.* class on its C struct (TensorView.tile,
  TileWindow.load_vec / store_vec / load_scalar): the C twin
  rocke_<stem>_<method> its header documents as <Class>.<method>, self passed
  by address wherever the C parameter list has it, the Python parameters
  paired with the C ones as a helper's are (by name; out, count and builder
  by role), so *local_indices becomes the array and its count;
- a Python argument the C twin has no parameter for (make_lds_view's
  exclusive): only its default translates."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys

import pytest
from trtest import ROCKE, TR, platform_probe, strip_holes

PROBE = """\
def probe_views(spec: BatchedTranspose2DSpec) -> KernelDef:
    io_ty = io_ir_type(spec.dtype)
    b = IRBuilder("probe_views")
    X = b.param("X", PtrType(io_ty, "global"))
    W = b.param("W", I32)
    v = make_global_view(X, shape=(2, spec.tile_m), dtype=io_ty, strides=(W, 1))
    p = make_global_view(
        X, shape=(2, spec.tile_m), dtype=io_ty, strides=(spec.tile_m, 1)
    )
    t = v.tile(lengths=(1, spec.tile_m), origin=(b.block_id_x(), b.const_i32(0)))
    pt = p.tile(lengths=(2, spec.tile_m), origin=(b.const_i32(0), b.const_i32(0)))
    lds = make_lds_view(b, dtype=io_ty, shape=(spec.tile_m,), name_hint="s")
    lt = lds.tile(lengths=(spec.tile_m,), origin=(b.const_i32(0),))
    x = t.load_vec(b, b.const_i32(0), b.thread_id_x(), n=2)
    lt.store_vec(b, b.thread_id_x(), value=x, n=2)
    y = pt.load_scalar(b, b.const_i32(1), b.thread_id_x())
    b.add(y, y)
    return b.kernel


def probe_exclusive(spec: BatchedTranspose2DSpec) -> KernelDef:
    b = IRBuilder("probe_exclusive")
    make_lds_view(b, dtype=io_ir_type(spec.dtype), shape=(4,), exclusive=True)
    return b.kernel


def probe_not_exclusive(spec: BatchedTranspose2DSpec) -> KernelDef:
    b = IRBuilder("probe_not_exclusive")
    make_lds_view(b, dtype=io_ir_type(spec.dtype), shape=(4,), exclusive=False)
    return b.kernel
"""


def _run(env, abi, out=None):
    run = None
    for args in (
        ["tr.abi", "batched_transpose", "--header", str(abi)],
        ["tr.translate", "batched_transpose", "--abi", str(abi)]
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
    root = tmp_path_factory.mktemp("rocke_methods")
    env = platform_probe(root, PROBE)
    abi = root / "abi"
    abi.mkdir()
    out = root / "batched_transpose.cpp"
    _run(env, abi, out)
    cpp = out.read_text()
    return _run(env, abi), re.sub(r"\s+", " ", cpp), cpp, abi


def _body(cpp: str, fn: str) -> str:
    m = re.search(rf"\b{fn}\(.*?\{{(.*?)\}} /\* py:|\b{fn}\(.*?\{{(.*?)$", cpp)
    assert m, fn
    return m.group(1) or m.group(2)


@pytest.mark.parametrize(
    "name", ["probe_views", "probe_not_exclusive", "build_batched_transpose2d"]
)
def test_translates(probe, name):
    assert re.search(rf"^translated +{re.escape(name)}$", probe[0], re.M), probe[0]


def test_methods_through_their_c_twins(probe):
    """v.tile(lengths=.., origin=..): out first, self second, rank the count;
    t.load_vec(b, *idx, n=..): builder first, self second, the indices in
    Python's order as the array and its count."""
    body = _body(probe[1], "py_probe_views")
    assert re.search(
        r"if\(rocke_tensor_view_tile\( ?&(\w+), &v, \w+\.data\(\), "
        r"std::array<rocke_value_t\*, 2>\{block_id_x_\d+, const_i32_\d+\}\.data\(\), "
        r"\(int\)py::len\(std::array<rocke_value_t\*, 2>\{block_id_x_\d+, "
        r"const_i32_\d+\}\)\) != ROCKE_OK\) "
        r'throw ckc::ValueError\("tile failed"\); rocke_tile_window_t t = \1;',
        body,
    ), body
    m = re.search(
        r"rocke_value_t\* (const_i32_\d+) = rocke_b_const_i32\(b, 0\); "
        r"rocke_value_t\* (thread_id_x_\d+) = rocke_b_thread_id_x\(b\); "
        r"std::vector<rocke_value_t\*> (\w+); \3\.push_back\(\1\); "
        r"\3\.push_back\(\2\); rocke_value_t\* x = rocke_tile_window_load_vec\( ?b, "
        r"&t, \3\.data\(\), \(int\)py::len\(\3\), 2\);",
        body,
    )
    assert m, body
    assert re.search(
        r"rocke_tile_window_store_vec\( ?b, &lt, (\w+)\.data\(\), "
        r"\(int\)py::len\(\1\), x, 2\);",
        body,
    ), body
    assert re.search(
        r"rocke_value_t\* y = rocke_tile_window_load_scalar\( ?b, &pt, (\w+)\.data\(\), "
        r"\(int\)py::len\(\1\)\);",
        body,
    ), body
    assert re.search(
        r'rocke_make_lds_view\(b, &\w+, io_ty, .*?\.data\(\), .*?, "s", nullptr\)',
        body,
    ), body


def test_argument_with_no_c_parameter_needs_its_default(probe):
    assert re.search(
        r"^hole +probe_exclusive .*no C parameter for make_lds_view\(exclusive=\)",
        probe[0],
        re.M,
    ), probe[0]


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
