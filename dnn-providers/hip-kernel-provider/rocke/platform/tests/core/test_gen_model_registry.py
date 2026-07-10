"""Tests for the C model-registry generator (gen_model_registry).

Verifies the generated registry: discovers meta sidecars, keys by
(op, arch, dtype), carries num_features (the drift guard), and compiles +
looks up correctly as real C. Skipped if no C compiler.
"""

from __future__ import annotations

import ctypes
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_HEUR = os.path.normpath(os.path.join(_HERE, "..", "..", "python", "rocke", "heuristics"))
if _HEUR not in sys.path:
    sys.path.insert(0, _HEUR)

import gen_model_registry as gmr  # noqa: E402

_CC = shutil.which("cc") or shutil.which("gcc")
requires_cc = pytest.mark.skipif(_CC is None, reason="no C compiler")


def _model(dir_: Path, name: str, meta: dict):
    (dir_ / name).mkdir(parents=True, exist_ok=True)
    (dir_ / name / "model_tflops.meta.json").write_text(json.dumps(meta))


def test_discovery_sorted_and_keyed(tmp_path):
    models = tmp_path / "models"
    _model(models, "b", {"symbol": "rocke_score_gemm_universal_fp8_gfx942_tflops",
                         "op": "gemm_universal", "arch": "gfx942", "dtype": "fp8",
                         "num_features": 72})
    _model(models, "a", {"symbol": "rocke_score_fmha_fp16_gfx950_tflops",
                         "op": "fmha", "arch": "gfx950", "dtype": "fp16",
                         "num_features": 69})
    entries = gmr.generate(models, tmp_path / "out")
    # sorted by (op, arch, dtype): fmha before gemm_universal
    assert [e["op"] for e in entries] == ["fmha", "gemm_universal"]
    assert entries[0]["num_features"] == 69 and entries[1]["num_features"] == 72


def test_missing_key_raises(tmp_path):
    models = tmp_path / "models"
    _model(models, "bad", {"op": "fmha", "arch": "gfx950", "dtype": "fp16"})  # no symbol/num_features
    with pytest.raises(ValueError):
        gmr.generate(models, tmp_path / "out")


def test_empty_registry_is_valid(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    entries = gmr.generate(models, tmp_path / "out")
    assert entries == []
    # source still emits a valid (empty) table + lookup.
    src = (tmp_path / "out" / "rocke_model_registry.c").read_text()
    assert "kModels[]" in src and "rocke_lookup_model" in src


@requires_cc
def test_generated_registry_compiles_and_looks_up(tmp_path):
    models = tmp_path / "models"
    _model(models, "fmha", {"symbol": "rocke_score_fmha_fp16_gfx950_tflops",
                            "op": "fmha", "arch": "gfx950", "dtype": "fp16",
                            "num_features": 69})
    out = tmp_path / "out"
    gmr.generate(models, out)

    (out / "stub.c").write_text(
        "double rocke_score_fmha_fp16_gfx950_tflops(const double* f)"
        "{ return f[0] + f[68]; }\n")
    (out / "t.c").write_text('''
#include "rocke_model_registry.h"
int hit_num_features(void){
    const RockeModelEntry* m = rocke_lookup_model("fmha","gfx950","fp16");
    return m ? m->num_features : -1;
}
double hit_score(const double* f){
    const RockeModelEntry* m = rocke_lookup_model("fmha","gfx950","fp16");
    return m ? m->score(f) : -1.0;
}
int miss_is_null(void){
    return rocke_lookup_model("fmha","gfx942","fp16") == 0 ? 1 : 0;
}
int count(void){ return rocke_model_count(); }
''')
    so = out / "reg.so"
    subprocess.run(
        [_CC, "-I", str(out), "-O2", "-shared", "-fPIC", "-o", str(so),
         str(out / "t.c"), str(out / "stub.c"), str(out / "rocke_model_registry.c")],
        check=True, capture_output=True)
    lib = ctypes.CDLL(str(so))
    assert lib.hit_num_features() == 69
    assert lib.miss_is_null() == 1
    assert lib.count() == 1
    lib.hit_score.restype = ctypes.c_double
    lib.hit_score.argtypes = [ctypes.POINTER(ctypes.c_double)]
    f = (ctypes.c_double * 69)()
    f[0] = 2.0
    f[68] = 3.0
    assert lib.hit_score(f) == 5.0
