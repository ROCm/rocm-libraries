"""Python<->C++ FMHA featurizer round-trip test (the bridge guarantee).

The dispatcher's C++ featurizer MUST produce the exact same 69-feature vector as
the Python FmhaFeatureEngine.extract() the model trained on -- a silent mismatch
degrades kernel selection with no error. This test generates the C++ featurizer
(gen_fmha_featurizer), compiles it, and asserts BIT-IDENTICAL output vs. Python
over edge-case fixtures. Skipped if no C++ compiler.
"""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_HEUR = os.path.normpath(os.path.join(_HERE, "..", "..", "python", "rocke", "heuristics"))
if _HEUR not in sys.path:
    sys.path.insert(0, _HEUR)

import gen_fmha_featurizer as gen  # noqa: E402
from feature_engine import FmhaFeatureEngine  # noqa: E402

_CXX = shutil.which("g++") or shutil.which("c++") or shutil.which("clang++")
requires_cxx = pytest.mark.skipif(_CXX is None, reason="no C++ compiler")

# hw values used on both sides (device-derived at runtime; fixed here for parity).
_HW = dict(num_cus=256, simds_per_cu=4, shader_engines=32, max_clock_mhz=2400,
           wavefront_size=64, lds_capacity=65536, num_xcd=8)

# Edge-case fixtures: (problem dict, kernel/config dict). Chosen to exercise the
# formula boundaries: decode (sq=1), exact tile division (fmod=0), gqa vs mha,
# tn1=0 guard, each dtype, and a huge shape for float precision.
_FIXTURES = [
    # decode, mha, fp16
    (dict(batch=8, seqlen_q=1, seqlen_k=2048, nhead_q=32, nhead_k=32,
          hdim_q=128, hdim_v=128, dtype="fp16"),
     dict(pipeline=1, tile_m0=16, tile_n0=64, num_warps=1, paged=1)),
    # prefill, gqa, bf16, exact tile division (sq % tm0 == 0)
    (dict(batch=4, seqlen_q=256, seqlen_k=256, nhead_q=32, nhead_k=8,
          hdim_q=64, hdim_v=64, dtype="bf16"),
     dict(pipeline=1, tile_m0=32, tile_n0=32, num_warps=2, paged=1)),
    # tn0=0 -> tile_n0 zero (ntk/eff guards); num_warps=4
    (dict(batch=1, seqlen_q=512, seqlen_k=512, nhead_q=16, nhead_k=16,
          hdim_q=128, hdim_v=128, dtype="fp16"),
     dict(pipeline=1, tile_m0=16, tile_n0=0, num_warps=4, paged=1)),
    # huge shape (float precision on ops/mem)
    (dict(batch=64, seqlen_q=8192, seqlen_k=8192, nhead_q=64, nhead_k=8,
          hdim_q=256, hdim_v=256, dtype="bf16"),
     dict(pipeline=1, tile_m0=32, tile_n0=128, num_warps=2, paged=1)),
    # variant flags set (feature_count path: mask/bias/lse/sink)
    (dict(batch=2, seqlen_q=1024, seqlen_k=4096, nhead_q=40, nhead_k=8,
          hdim_q=128, hdim_v=128, dtype="fp16"),
     dict(pipeline=1, tile_m0=16, tile_n0=64, num_warps=1, paged=1,
          mask=1, bias=1, lse=1, sink=1)),
    # unknown dtype -> dt_enc default 0, bpe default 2
    (dict(batch=1, seqlen_q=128, seqlen_k=128, nhead_q=8, nhead_k=8,
          hdim_q=64, hdim_v=64, dtype="weird"),
     dict(pipeline=1, tile_m0=16, tile_n0=32, num_warps=1, paged=1)),
]


def _build_lib(tmp):
    """Generate + compile a tiny C++ shim exposing featurize -> double[69]."""
    disp = os.path.join(tmp, "dispatcher", "sdpa_fwd")
    gen.generate(__import__("pathlib").Path(disp))
    shim = os.path.join(tmp, "shim.cpp")
    with open(shim, "w") as f:
        f.write("""
#include "dispatcher/sdpa_fwd/FmhaFeaturizer.hpp"
#include <cstring>
using namespace rocke_client::dispatcher;
extern "C" void featurize_c(
    double batch,double sq,double sk,double hq,double hk,double dq,double dv,
    const char* dtype,
    double pip,double tm0,double tn0,double num_warps,
    double mask,double bias,double lse,double sink,double paged,
    double num_cus,double simds_per_cu,double total_simds,double shader_engines,
    double max_clock_mhz,double wavefront_size,double lds_capacity,double num_xcd,
    double* out)
{
    FmhaProblemInputs p; p.batch=batch;p.sq=sq;p.sk=sk;p.hq=hq;p.hk=hk;
    p.dq=dq;p.dv=dv;p.dtype=dtype;
    FmhaConfigInputs c; c.pip=pip;c.tm0=tm0;c.tn0=tn0;c.num_warps=num_warps;
    c.mask=mask;c.bias=bias;c.lse=lse;c.sink=sink;c.paged=paged;
    FmhaHwInputs hw; hw.num_cus=num_cus;hw.simds_per_cu=simds_per_cu;
    hw.total_simds=total_simds;hw.shader_engines=shader_engines;
    hw.max_clock_mhz=max_clock_mhz;hw.wavefront_size=wavefront_size;
    hw.lds_capacity=lds_capacity;hw.num_xcd=num_xcd;
    auto arr = fmha_featurize(p,c,hw).to_array();
    std::memcpy(out, arr.data(), arr.size()*sizeof(double));
}
""")
    so = os.path.join(tmp, "feat.so")
    subprocess.run([_CXX, "-std=c++17", "-O2", "-I", tmp, "-shared", "-fPIC",
                    "-o", so, shim], check=True, capture_output=True)
    lib = ctypes.CDLL(so)
    # shim signature: 7 problem doubles, dtype char*, 9 config doubles,
    # 8 hw doubles, out*.
    lib.featurize_c.argtypes = (
        [ctypes.c_double] * 7 + [ctypes.c_char_p] + [ctypes.c_double] * (9 + 8)
        + [ctypes.POINTER(ctypes.c_double)])
    return lib


@requires_cxx
@pytest.mark.parametrize("prob,cfg", _FIXTURES)
def test_roundtrip_bit_identical(prob, cfg, tmp_path):
    tmp = str(tmp_path)
    lib = _build_lib(tmp)

    # Python side
    eng = FmhaFeatureEngine(**_HW)
    py = eng.extract(prob, cfg)
    assert py.shape[0] == 69

    # C++ side
    out = (ctypes.c_double * 69)()
    lib.featurize_c(
        float(prob["batch"]), float(prob["seqlen_q"]), float(prob["seqlen_k"]),
        float(prob["nhead_q"]), float(prob["nhead_k"]),
        float(prob["hdim_q"]), float(prob["hdim_v"]),
        prob["dtype"].encode(),
        float(cfg.get("pipeline", 1)), float(cfg.get("tile_m0", 16)),
        float(cfg.get("tile_n0", 0)), float(cfg.get("num_warps", 1)),
        float(cfg.get("mask", 0)), float(cfg.get("bias", 0)),
        float(cfg.get("lse", 0)), float(cfg.get("sink", 0)),
        float(cfg.get("paged", 1)),
        float(_HW["num_cus"]), float(_HW["simds_per_cu"]),
        float(_HW["num_cus"] * _HW["simds_per_cu"]), float(_HW["shader_engines"]),
        float(_HW["max_clock_mhz"]), float(_HW["wavefront_size"]),
        float(_HW["lds_capacity"]), float(_HW["num_xcd"]),
        out)
    cpp = np.array([out[i] for i in range(69)], dtype=np.float64)

    # Bit-identical (same formulas, same op order) -- any diff is a real bug.
    names = eng.get_feature_names()
    for i in range(69):
        assert py[i] == cpp[i], (
            f"feature[{i}] {names[i]}: py={py[i]!r} cpp={cpp[i]!r}")


def test_generator_struct_matches_names(tmp_path):
    # The struct field order (to_array) must equal get_feature_names order.
    import pathlib
    names = gen.generate(pathlib.Path(tmp_path))
    struct = (tmp_path / "FmhaFeatures.hpp").read_text()
    # every feature name appears as a field, in order
    idx = [struct.index(f"double {n} ") for n in names]
    assert idx == sorted(idx), "struct field order diverges from get_feature_names"
