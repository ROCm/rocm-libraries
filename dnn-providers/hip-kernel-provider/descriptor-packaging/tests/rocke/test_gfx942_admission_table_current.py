"""Drift guard for hipDNN's gfx942 unified 3D admission table.

The kernel ingestor engine ``hipkernel:Gfx942AttentionUnified3d`` restates, in
C++, which graphs rocKE's strided split-KV decode candidate admits
(``dispatch.attention.strided_decode.make_candidate().admits``), and
``TestGfx942AttentionUnified3dAdmission.cpp`` checks it against
``Gfx942AttentionUnified3dAdmissionTable.inc``, a table generated from the rocKE
library. This test re-runs the generator in memory and requires the committed
table byte-for-byte, so a change to the candidate's admission fails CI even
when nobody regenerates the table -- but only where the change alters a
SAMPLED row.

Coverage is the generator's sampled grid only: batch {1, 3, 64} x (Hq, Hkv)
{(8, 1), (32, 8), (12, 4), (32, 1), (16, 16)} x seqlen_q {1, 2} x seqlen_k
{1, 64, 256, 512, 8192} x D {64, 96, 128, 256} x dtype {fp16, bf16} x mask
{none, top-left causal, bottom-right causal, window 128} x kv_block_size
{16, 32, 64}, all at 304 CUs. A change that only moves admission at another
shape (for example only at batch 1024 or D 192) passes this test.

CI lane: the ``hip-kernel-provider-hkp-pack-quick`` CTest, which runs
``pytest descriptor-packaging/tests -m quick`` from the source tree; it is
registered only on builds with ``HIPKERNELPROVIDER_ENABLE_ROCKE=ON``.
``conftest.py`` puts the rocKE library on ``sys.path`` for the generator's
imports.

Fix a failure by regenerating the table (command in the generator docstring)
and updating the engine's graph match in ``Gfx942AttentionUnified3dNative.cpp``.
"""

import contextlib
import importlib.util
import io
from pathlib import Path

import pytest

# <provider>/descriptor-packaging/tests/rocke/<this>
_PROVIDER = Path(__file__).resolve().parents[3]
_PACKS = _PROVIDER / "src" / "tests" / "engines" / "kernel_ingestor_engine" / "packs"
_GENERATOR = _PACKS / "gen_gfx942_attention_unified_3d_admission.py"
_TABLE = _PACKS / "Gfx942AttentionUnified3dAdmissionTable.inc"


@pytest.mark.quick
def test_gfx942_admission_table_is_current():
    assert _GENERATOR.is_file(), f"missing generator: {_GENERATOR}"
    assert _TABLE.is_file(), f"missing table: {_TABLE}"

    spec = importlib.util.spec_from_file_location("_gen_gfx942_admission", _GENERATOR)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        gen.main()

    assert out.getvalue() == _TABLE.read_text(), (
        f"{_TABLE.name} is stale against rocKE's strided decode admission: regenerate "
        f"it with the command in {_GENERATOR.name} and update the engine's graph match"
    )
