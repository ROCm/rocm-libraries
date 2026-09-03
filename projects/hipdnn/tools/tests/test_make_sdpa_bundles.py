# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generated SDPA bundles must have the same shape as the shipped ones.

A generated bundle is fed to the same loader as a shipped one, and that loader is
unforgiving in two ways that produce no useful signal:

  - A missing attribute key fails `json::to<Graph>`, and the bundle is logged and
    skipped rather than registered. No test exists, so nothing fails: the sweep
    reports a clean run over the cases that did register.
  - A dtype spelled with a name the FlatBuffer enum does not use converts anyway,
    and is refused later by `from_binary` with "Failed to create backend graph
    descriptor from serialized data" -- which names neither the field nor the value.

Both shipped, both silently, and between them they cost 242 of 377 problems in a
gfx942 sweep that looked like it had run everything. The shipped bundles are the
only executable statement of the shape, so these compare against them directly
rather than against a second copy of the rules.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[4]
TOOL = REPO / "projects/hipdnn/tools/make_sdpa_bundles.py"
KDP = (
    REPO
    / "dnn-providers/hip-kernel-provider/descriptor-packaging/examples/descriptors"
    / "rocKE/gfx942_attention_dense/gfx942_attention_dense.kdp.json"
)
SHIPPED = REPO / "dnn-providers/integration-tests/integration-test-bundles/quick/SdpaFwd/bshd"

pytestmark = pytest.mark.skipif(
    not (TOOL.is_file() and KDP.is_file() and SHIPPED.is_dir()),
    reason="generator, KDP or shipped bundle tree not present",
)


def _tool():
    spec = importlib.util.spec_from_file_location("make_sdpa_bundles", TOOL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _graphs(root: Path):
    """Every bundle graph under `root`, keyed by (dtype dir, causal)."""
    found = {}
    for path in sorted(root.rglob("*.json")):
        if path.name.endswith(".meta.json"):
            continue
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not doc.get("nodes"):
            continue
        attrs = doc["nodes"][0].get("attributes", {})
        # Causality is the window, not the flag -- the shipped bundles leave
        # causal_mask False and carry left_bound/right_bound instead.
        causal = attrs.get("right_bound") is not None
        dtype_dir = next((p for p in path.parts if p in ("bf16", "fp16")), None)
        if dtype_dir:
            found.setdefault((dtype_dir, causal), (path, doc))
    return found


@pytest.fixture(scope="module")
def generated(tmp_path_factory, request):
    out = tmp_path_factory.mktemp("bundles")
    tool = _tool()
    # Small cap: this asserts shape, and one bundle per (dtype, causal) is enough.
    argv = [
        "make_sdpa_bundles",
        "--kdp", str(KDP),
        "--out", str(out),
        "--limit", "40",
        "--min-candidates", "3",
        "--max-bytes", str(16 * 1024 * 1024),
    ]
    monkeypatch = pytest.MonkeyPatch()
    request.addfinalizer(monkeypatch.undo)
    monkeypatch.setattr(sys, "argv", argv)
    tool.main()
    return _graphs(out)


@pytest.fixture(scope="module")
def shipped():
    return _graphs(SHIPPED)


def test_the_generator_produces_both_dtypes_and_both_mask_modes(generated):
    """The corpus is worthless if a whole quadrant silently never appears."""
    assert set(generated) == {
        ("bf16", True), ("bf16", False), ("fp16", True), ("fp16", False)
    }, f"missing quadrants: {sorted(generated)}"


def test_every_generated_bundle_declares_the_shipped_attribute_keys(generated, shipped):
    """Key presence, not just value.

    `left_bound: null` and an absent `left_bound` are different documents to the
    reader: one is an explicit "no window", the other fails conversion outright.
    """
    for key, (gen_path, gen_doc) in sorted(generated.items()):
        if key not in shipped:
            continue
        _, ship_doc = shipped[key]
        gen_attrs = set(gen_doc["nodes"][0]["attributes"])
        ship_attrs = set(ship_doc["nodes"][0]["attributes"])
        assert gen_attrs == ship_attrs, (
            f"{key} bundle {gen_path.name} attribute keys differ from the shipped bundle; "
            f"missing={sorted(ship_attrs - gen_attrs)} extra={sorted(gen_attrs - ship_attrs)}"
        )


def test_generated_dtypes_use_the_flatbuffer_enum_spellings(generated, shipped):
    """`half`, not `float16`.

    A wrong-but-plausible spelling converts and is refused at from_binary, so this
    compares against the shipped bundle rather than against a hand-written table
    that could drift the same way.
    """
    for key, (gen_path, gen_doc) in sorted(generated.items()):
        if key not in shipped:
            continue
        _, ship_doc = shipped[key]
        for field in ("io_data_type", "compute_data_type", "intermediate_data_type"):
            assert gen_doc.get(field) == ship_doc.get(field), (
                f"{key} bundle {gen_path.name} {field}={gen_doc.get(field)!r}, "
                f"shipped says {ship_doc.get(field)!r}"
            )
        gen_tensor_dtypes = {t["data_type"] for t in gen_doc["tensors"]}
        ship_tensor_dtypes = {t["data_type"] for t in ship_doc["tensors"]}
        assert gen_tensor_dtypes == ship_tensor_dtypes, (
            f"{key} bundle {gen_path.name} tensor dtypes {sorted(gen_tensor_dtypes)} "
            f"differ from shipped {sorted(ship_tensor_dtypes)}"
        )
