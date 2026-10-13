# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The shipped dense-attention catalog, reproduced from the rocKE registry.

``catalog_diff.py`` can say two catalogs are the same; this says the one that
shipped is what the dispatcher registry still enumerates. If rocKE adds, drops
or re-defaults a dense candidate or a tile, the shipped KDP and the registry
disagree and this fails -- the drift is found here, not by a user's dispatch.

The slice rule (what "the shipped catalog" means) is explicit:

  * candidate  ``attention_dense_grid`` / spec ``gfx950_dense_grid``,
  * one request per shipped ``(dtype, head_size, num_query_heads,
    num_kv_heads, causal)`` at the canonical B1 / Sq512 / Skv512, ragged 0,
  * from the ``production`` sweep keep specs that differ from the
    candidate's own default spec ONLY in ``block_m`` / ``block_n`` (the tile is
    a knob axis; every other knob axis is crossed by the sweep but not shipped).

The slice itself lives in ``tools/registry_enumerator.py``; this test drives it
from the engine manifest, so it checks the tool, not a copy of its logic.

The registry spec dataclass carries more fields than the shipped KDP spec; the
extra ones must sit at their dataclass defaults, which is asserted here.

Skipped when rocKE is not importable (a plain IngestorGenerator checkout).
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

import catalog_diff  # noqa: E402

_REPO = Path(__file__).resolve().parents[5]
_PROVIDER = _REPO / "dnn-providers" / "hip-kernel-provider"
_SHIPPED_KDP = (
    _PROVIDER
    / "src/engines/kernel_ingestor_engine/descriptors/rocKE"
    / "gfx950_attention_dense/gfx950_attention_dense.kdp.json"
)
_ROCKE_PATHS = (
    _PROVIDER / "rocke" / "library",
    _PROVIDER / "rocke" / "platform" / "python",
)

_MANIFEST = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "gfx950_attention_dense.engine.yaml"
)


@pytest.fixture(scope="module")
def shipped() -> dict:
    if not _SHIPPED_KDP.is_file():
        pytest.skip(f"shipped KDP not in this checkout: {_SHIPPED_KDP}")
    return json.loads(_SHIPPED_KDP.read_text())


@pytest.fixture(scope="module")
def registry():
    missing = [str(p) for p in _ROCKE_PATHS if not p.is_dir()]
    if missing:
        pytest.skip(f"rocKE not in this checkout: {missing}")
    added = [str(p) for p in _ROCKE_PATHS if str(p) not in sys.path]
    sys.path[:0] = added
    try:
        yield from _registry_enumerator()
    finally:
        for p in added:
            if p in sys.path:
                sys.path.remove(p)


def _registry_enumerator():
    import registry_enumerator as enumerator

    manifest = enumerator.load_manifest(str(_MANIFEST))
    try:
        enumerator.bind_provider(manifest.get("provider_root"))
        slice_ = enumerator._attention_slice(manifest)
    except enumerator.EnumerationError as exc:
        pytest.skip(f"rocKE dispatcher registry not importable: {exc}")
    yield slice_


def _heads(shipped: dict) -> set:
    return {
        (
            e["kernel_source"]["spec"]["dtype"],
            e["kernel_source"]["spec"]["head_size"],
            e["kernel_source"]["spec"]["num_query_heads"],
            e["kernel_source"]["spec"]["num_kv_heads"],
            bool(e["kernel_source"]["spec"]["causal"]),
        )
        for e in shipped["kernelDescriptors"]
    }


def _entry(shipped_entry: dict, ks) -> dict:
    """A catalog entry for registry spec ``ks``, in the shipped entry's shape."""
    shipped_keys = shipped_entry["kernel_source"]["spec"].keys()
    spec = {k: getattr(ks, k) for k in shipped_keys}
    dtype = spec["dtype"]
    name = (
        f"attention_dense.{dtype}_d{spec['head_size']}"
        f"_hq{spec['num_query_heads']}_kv{spec['num_kv_heads']}"
        f"_c{int(spec['causal'])}_bm{spec['block_m']}_bn{spec['block_n']}.gfx950"
    )
    meta = {
        "dtype": dtype.upper(),
        "head_size": spec["head_size"],
        "num_query_heads": spec["num_query_heads"],
        "num_kv_heads": spec["num_kv_heads"],
        "causal": int(spec["causal"]),
        "ragged": int(spec["ragged"]),
        "sliding_window": spec["sliding_window"],
        "batch": spec["batch"],
        "seqlen_q": spec["seqlen_q"],
        "seqlen_kv": spec["seqlen_kv"],
        "block_m": spec["block_m"],
        "block_n": spec["block_n"],
    }
    return {
        "version": shipped_entry["version"],
        "id": "regenerated",
        "name": name,
        "kernel_source": {**shipped_entry["kernel_source"], "spec": spec},
        "metadata": meta,
        "priority": shipped_entry["priority"],
    }


def _field_default(f: dataclasses.Field):
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return dataclasses.MISSING


def test_registry_reproduces_the_shipped_dense_catalog(shipped, registry):
    template = shipped["kernelDescriptors"][0]
    shipped_keys = set(template["kernel_source"]["spec"])
    pairs = [
        (_entry(template, ks), ks)
        for dtype, d, hq, hk, causal in sorted(_heads(shipped))
        for ks in registry(dtype, d, hq, hk, causal)
    ]
    entries = [e for e, _ in pairs]
    regenerated = {**shipped, "kernelDescriptors": entries}
    # Every shipped field matches value for value (the default gate keys on them).
    problems = catalog_diff.compare_catalog(shipped, regenerated)
    assert not problems, "\n".join(problems)
    assert len(entries) == len(shipped["kernelDescriptors"])

    # The registry spec is wider than the shipped one. The shipped catalog is
    # only the same kernel set if every field it does not carry sits at its
    # dataclass default -- hkp_pack hydrates Spec(**fields), so that is what the
    # packer compiles. A registry candidate that moves one off-default is a
    # different binary the shipped KDP cannot express.
    off_default = []
    for _, ks in pairs:
        for f in dataclasses.fields(ks):
            if f.name in shipped_keys:
                continue
            default = _field_default(f)
            if default is dataclasses.MISSING or getattr(ks, f.name) != default:
                off_default.append((f.name, getattr(ks, f.name), default))
    assert not off_default, sorted({repr(o) for o in off_default})[:10]


def test_block_n_256_exists_at_d64_and_not_at_d128(shipped, registry):
    """The tile axis is LDS-bounded: D128 needs 1184 B per block_n row, so
    256x256 (303,104 B) exceeds the 163,840 B budget and is pruned, while D64
    (528 B per row) keeps it. An earlier plan draft counted six D64 tiles."""
    by_d: dict[int, set] = {}
    for dtype, d, hq, hk, causal in sorted(_heads(shipped)):
        for ks in registry(dtype, d, hq, hk, causal):
            by_d.setdefault(ks.head_size, set()).add((ks.block_m, ks.block_n))
    assert (256, 256) in by_d[64]
    assert (256, 256) not in by_d[128]
    assert len(by_d[64]) == 7
    assert len(by_d[128]) == 6
    shipped_tiles = {
        d: {
            (e["metadata"]["block_m"], e["metadata"]["block_n"])
            for e in shipped["kernelDescriptors"]
            if e["metadata"]["head_size"] == d
        }
        for d in (64, 128)
    }
    assert shipped_tiles == by_d
