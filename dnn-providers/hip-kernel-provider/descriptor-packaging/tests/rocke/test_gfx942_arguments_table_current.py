"""Drift guard for hipDNN's gfx942 3D decode segment-argument table.

The kernel ingestor engine restates, in C++, the ordered argument list rocKE's
strided decode passes the segment kernel (``_3d_signature(dtype,
strided_kv=True)`` with ``StridedKvCacheLayout.arguments()`` for the K/V
strides) and checks it against ``Gfx942AttentionUnified3dArgumentsTable.inc``,
a table generated from the rocKE library. This test re-runs the generator in
memory and requires the committed table byte-for-byte, so a change to the
signature order or the stride packing fails CI even when nobody regenerates
the table.

CI lane: the ``hip-kernel-provider-hkp-pack-quick`` CTest, which runs
``pytest descriptor-packaging/tests -m quick`` from the source tree; it is
registered only on builds with ``HIPKERNELPROVIDER_ENABLE_ROCKE=ON``.
``conftest.py`` puts the rocKE library on ``sys.path`` for the generator's
imports.

Fix a failure by regenerating the table (command in the generator docstring)
and updating ``Gfx942AttentionUnified3dArguments.hpp`` to match.
"""

import contextlib
import importlib.util
import io
from pathlib import Path

import pytest

# <provider>/descriptor-packaging/tests/rocke/<this>
_PROVIDER = Path(__file__).resolve().parents[3]
_PACKS = _PROVIDER / "src" / "tests" / "engines" / "kernel_ingestor_engine" / "packs"
_GENERATOR = _PACKS / "gen_gfx942_attention_unified_3d_arguments.py"
_TABLE = _PACKS / "Gfx942AttentionUnified3dArgumentsTable.inc"


@pytest.mark.quick
def test_gfx942_arguments_table_is_current():
    assert _GENERATOR.is_file(), f"missing generator: {_GENERATOR}"
    assert _TABLE.is_file(), f"missing table: {_TABLE}"

    spec = importlib.util.spec_from_file_location("_gen_gfx942_arguments", _GENERATOR)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        gen.main()

    assert out.getvalue() == _TABLE.read_text(), (
        f"{_TABLE.name} is stale against rocKE's segment signature / KvTensorLayout: "
        f"regenerate it with the command in {_GENERATOR.name} and update the C++ "
        f"argument builder to match"
    )
