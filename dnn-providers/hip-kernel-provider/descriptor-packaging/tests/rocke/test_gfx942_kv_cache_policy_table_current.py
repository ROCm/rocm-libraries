"""Drift guard for hipDNN's gfx942 3D decode K/V cache-policy table.

The kernel ingestor engine restates rocKE's ``_kv_cache_policy_3d`` in C++
(``gfx942KvCachePolicy`` in ``Gfx942AttentionUnified3dGeometry.hpp``, which the
scorer uses to break ties between kernels that differ only in
``kv_cache_policy``) and checks it against
``Gfx942AttentionUnified3dKvCachePolicyTable.inc``, a table generated from the
rocKE library. This test re-runs the generator in memory and requires the
committed table byte-for-byte, so a change to the rule fails CI even when
nobody regenerates the table -- but only where the change alters a SAMPLED row.

Coverage is the generator's grid only: batch {1, 4, 8, 9, 16, 32, 64, 256} x
Hkv {1, 8} x D {64, 128, 256} x seqlen_k {1, 512, 8191, 8192, 32768, 131072}.
It holds K/V of exactly 256 MiB (B 8, Skv 8192, Hkv 8, D 128) and the next
batch up, so moving the threshold or turning ">" into ">=" changes a row.

CI lane: the ``hip-kernel-provider-hkp-pack-quick`` CTest, which runs
``pytest descriptor-packaging/tests -m quick`` from the source tree; it is
registered only on builds with ``HIPKERNELPROVIDER_ENABLE_ROCKE=ON``.
``conftest.py`` puts the rocKE library on ``sys.path`` for the generator's
imports.

Fix a failure by regenerating the table (command in the generator docstring)
and updating ``gfx942KvCachePolicy`` to match.
"""

import contextlib
import importlib.util
import io
from pathlib import Path

import pytest

# <provider>/descriptor-packaging/tests/rocke/<this>
_PROVIDER = Path(__file__).resolve().parents[3]
_PACKS = _PROVIDER / "src" / "tests" / "engines" / "kernel_ingestor_engine" / "packs"
_GENERATOR = _PACKS / "gen_gfx942_attention_unified_3d_kv_cache_policy.py"
_TABLE = _PACKS / "Gfx942AttentionUnified3dKvCachePolicyTable.inc"


@pytest.mark.quick
def test_gfx942_kv_cache_policy_table_is_current():
    assert _GENERATOR.is_file(), f"missing generator: {_GENERATOR}"
    assert _TABLE.is_file(), f"missing table: {_TABLE}"

    spec = importlib.util.spec_from_file_location(
        "_gen_gfx942_kv_cache_policy", _GENERATOR
    )
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        gen.main()

    assert out.getvalue() == _TABLE.read_text(), (
        f"{_TABLE.name} is stale against rocKE's _kv_cache_policy_3d: regenerate it "
        f"with the command in {_GENERATOR.name} and update gfx942KvCachePolicy to match"
    )
