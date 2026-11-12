# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""``cf.return`` under divergent control flow in the direct code-object backend.

A return under an ``scf.if`` whose condition, or an ``scf.for`` whose bounds,
differ between lanes takes only the lanes that reach it out of EXEC, for the
rest of the kernel: the others go on, and a wave with no lane left ends. A
return where every lane returns still just ends the wave.

The kernels are compiled for gfx950 (needs the direct library) and run on the
local device when it is a gfx950 (output compared with the host evaluation).
"""

from __future__ import annotations

import ctypes
import struct

import numpy as np
import pytest
from rocke.runtime import code_object

GFX950 = "amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-"
THREADS = 128  # two wave64s
UNTOUCHED = -7.0

_HEAD = """ckdsl.ir v1
kernel @early_return {
  attrs { max_workgroup_size = i:128 }
  params {
    %Y : ptr<f32,global>
    %n : i32
  }
  region @entry {
    %tid : i32 = gpu.thread_id (  ) { axis = s:"x" }
    %c0 : i32 = arith.constant (  ) { ity = s:"i32", value = i:0 }
    %c1 : i32 = arith.constant (  ) { ity = s:"i32", value = i:1 }
    %z : f32 = arith.constant (  ) { ity = s:"f32", value = f:0.0 }
"""
_STORE = '    memref.global_store_typed ( %Y, %tid, {v} ) {{ align = i:4, elem_type = s:"f32" }}\n'
_TAIL = "    cf.return (  )\n  }\n}\n"

_IF = (
    '    %c : i1 = arith.cmp ( %tid, %n ) { pred = s:"ge" }\n'
    "    scf.if ( %c )\n      region @then {\n        cf.return (  )\n      }\n"
    '    %one : f32 = arith.constant (  ) { ity = s:"f32", value = f:1.0 }\n'
) + _STORE.format(v="%one")

_IF_ELSE = (
    '    %c : i1 = arith.cmp ( %tid, %n ) { pred = s:"lt" }\n'
    "    scf.if_else ( %c )\n      region @then {\n        %t : i32 = arith.add ( %tid, %c1 )\n      }\n"
    "      region @else {\n        cf.return (  )\n      }\n"
    '    %two : f32 = arith.constant (  ) { ity = s:"f32", value = f:2.0 }\n'
) + _STORE.format(v="%two")

# Each lane counts to its own bound, %tid, and leaves for good once the count reaches %n.
_LOOP = (
    '    %r : f32 = scf.for ( %c0, %tid, %c1, %z ) { iter_args = l:[ { name = s:"%acc", type = s:"f32" } ], '
    'iv = s:"%i", iv_type = s:"i32", num_iter_args = i:1, unroll = b:false }\n'
    "      region @body {\n"
    '        %q : i1 = arith.cmp ( %i, %n ) { pred = s:"ge" }\n'
    "        scf.if ( %q )\n          region @then {\n            cf.return (  )\n          }\n"
    '        %one : f32 = arith.constant (  ) { ity = s:"f32", value = f:1.0 }\n'
    "        %a2 : f32 = arith.fadd ( %acc, %one )\n"
    "        scf.yield ( %a2 ) { num = i:1 }\n      }\n"
) + _STORE.format(v="%r")

# The shape of unified attention's sequence lookup: the trip count of a search loop comes from `n` through an
# int-to-float conversion (floor(log2(n)) + 1), the loop carries values read from memory, and the kernel
# returns when its block lies past the last sequence. Every value is wave-uniform in fact (block id and
# kernel arguments), so the return ends whole waves: it must not need the per-lane path.
_UNIFORM_HEAD = _HEAD.replace(
    "    %n : i32\n", "    %n : i32\n    %Q : ptr<i32,global>\n    %thr : i32\n"
)
_SEARCH = (
    "    %nf : f32 = arith.sitofp_f32 ( %n )\n"
    '    %nb : i32 = arith.bitcast ( %nf ) { target = s:"i32" }\n'
    '    %c23 : i32 = arith.constant (  ) { ity = s:"i32", value = i:23 }\n'
    '    %c126 : i32 = arith.constant (  ) { ity = s:"i32", value = i:126 }\n'
    "    %ex : i32 = arith.lshr ( %nb, %c23 )\n"
    "    %lg : i32 = arith.sub ( %ex, %c126 )\n"
    "    %iters : i32 = arith.smax ( %lg, %c1 )\n"
    '    %lo : i32 = scf.for ( %c0, %iters, %c1, %c0 ) { iter_args = l:[ { name = s:"%left", type = s:"i32" } ], '
    'iv = s:"%bs", iv_type = s:"i32", num_iter_args = i:1, unroll = b:false }\n'
    "      region @body {\n"
    '        %v : i32 = memref.global_load_typed ( %Q, %bs ) { align = i:4, elem_type = s:"i32" }\n'
    "        %l2 : i32 = arith.add ( %left, %v )\n"
    "        scf.yield ( %l2 ) { num = i:1 }\n      }\n"
    '    %past : i1 = arith.cmp ( %lo, %thr ) { pred = s:"ge" }\n'
    "    scf.if ( %past )\n      region @then {\n        cf.return (  )\n      }\n"
    '    %three : f32 = arith.constant (  ) { ity = s:"f32", value = f:3.0 }\n'
) + _STORE.format(v="%three")

CASES = {
    "if": (_IF, lambda t, n: 1.0 if t < n else UNTOUCHED),
    "if_else": (_IF_ELSE, lambda t, n: 2.0 if t < n else UNTOUCHED),
    "loop": (_LOOP, lambda t, n: float(t) if t <= n else UNTOUCHED),
}
# Lanes returning: none, part of the first wave, the whole first wave and part
# of the second, all but one, all.
NS = (THREADS, 37, 64, 70, 1, 0)


def _ir(body: str) -> str:
    return (_UNIFORM_HEAD if body is _SEARCH else _HEAD) + body + _TAIL


def _require_direct_library() -> None:
    if code_object._direct_library_path() is None:
        pytest.skip("build the Rust cdylib or set ROCKE_CODEOBJ_DIRECT_LIB")
    if "gfx950" not in code_object.direct_supported_targets():
        pytest.skip("the direct library predates gfx950")


def test_uniform_return_after_a_conversion_bound_search_compiles() -> None:
    _require_direct_library()
    hsaco = code_object.build_direct_rocke_ir_hsaco(_ir(_SEARCH), isa=GFX950)[0]
    assert hsaco.startswith(b"\x7fELF")


@pytest.mark.parametrize("name", CASES)
def test_divergent_return_compiles(name: str) -> None:
    _require_direct_library()
    hsaco, _timings, _info, diagnostics = code_object.build_direct_rocke_ir_hsaco(
        _ir(CASES[name][0]), isa=GFX950
    )
    assert hsaco.startswith(b"\x7fELF")
    assert diagnostics.startswith("instructions=")


def _run_search(rt, fn, n: int, thr: int, q: np.ndarray) -> np.ndarray:
    out = np.full(THREADS, UNTOUCHED, np.float32)
    dev_y, dev_q = rt.alloc(out.nbytes), rt.alloc(q.nbytes)
    ybuf = (ctypes.c_uint8 * out.nbytes).from_buffer(out)
    qbuf = (ctypes.c_uint8 * q.nbytes).from_buffer(q)
    rt.memcpy_h2d(dev_y, ybuf, out.nbytes)
    rt.memcpy_h2d(dev_q, qbuf, q.nbytes)
    rt.launch(
        fn, (1, 1, 1), (THREADS, 1, 1), struct.pack("<QI4xQi", dev_y, n, dev_q, thr)
    )
    rt.sync()
    rt.memcpy_d2h(ybuf, dev_y, out.nbytes)
    rt.free(dev_y)
    rt.free(dev_q)
    return out


def test_uniform_return_after_a_conversion_bound_search_on_device() -> None:
    _require_direct_library()
    from rocke.runtime.hip_module import Runtime, get_device_arch, get_device_count

    if get_device_count() == 0 or not get_device_arch(0).startswith("gfx950"):
        pytest.skip("needs a gfx950 device")
    rt = Runtime()
    mod = rt.load_module(
        code_object.build_direct_rocke_ir_hsaco(_ir(_SEARCH), isa=GFX950)[0]
    )
    try:
        fn = mod.get_function("early_return")
        q = np.arange(1, 33, dtype=np.int32)
        for n in (1, 2, 5, 8, 9, 31):
            iters = max(int(np.float32(n).view(np.int32)) >> 23, 0) - 126
            lo = int(q[: max(iters, 1)].sum())
            for thr in (lo - 1, lo, lo + 1):
                got = _run_search(rt, fn, n, thr, q)
                want = UNTOUCHED if lo >= thr else 3.0
                assert np.all(got == want), f"n={n} thr={thr}: {got[:4]} != {want}"
    finally:
        mod.unload()


@pytest.mark.parametrize("name", CASES)
def test_divergent_return_on_device(name: str) -> None:
    _require_direct_library()
    from rocke.runtime.hip_module import Runtime, get_device_arch, get_device_count

    if get_device_count() == 0 or not get_device_arch(0).startswith("gfx950"):
        pytest.skip("needs a gfx950 device")
    body, expect = CASES[name]
    hsaco = code_object.build_direct_rocke_ir_hsaco(_ir(body), isa=GFX950)[0]
    rt = Runtime()
    mod = rt.load_module(hsaco)
    try:
        fn = mod.get_function("early_return")
        for n in NS:
            out = np.full(THREADS, UNTOUCHED, np.float32)
            dev = rt.alloc(out.nbytes)
            buf = (ctypes.c_uint8 * out.nbytes).from_buffer(out)
            rt.memcpy_h2d(dev, buf, out.nbytes)
            rt.launch(fn, (1, 1, 1), (THREADS, 1, 1), struct.pack("<Qi", dev, n))
            rt.sync()
            rt.memcpy_d2h(buf, dev, out.nbytes)
            rt.free(dev)
            want = np.array([expect(t, n) for t in range(THREADS)], np.float32)
            assert np.array_equal(out, want), f"n={n}: {out} != {want}"
    finally:
        mod.unload()
