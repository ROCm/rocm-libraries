# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""CI-facing static checks for rocke.

This harness intentionally lives under ``python/test`` so CI entry points do not
reach into example directories directly. It samples shipped example problem
descriptions and checks that the static IR parity suite covers every supported
architecture family without needing a GPU.

Run:
  PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=Python:python/test \
    python tests/test_rocke_ci_static.py
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

# The representative-IR parity harness lives in the instances test layer; make
# it importable from this root-level CI test (run via pytest or directly).
sys.path.insert(0, str(Path(__file__).resolve().parents[0] / "instances"))

from rocke.benchmark.gemm.fp16_rcr_sweep import (
    GemmSweepConfig,
    GemmSweepShape,
    expand_sweep,
)
from rocke.core.arch import known_arches
from rocke_ir_parity_harness import (  # noqa: E402 -- after sys.path shim
    cases,
    check_golden,
    current_flavor,
)

_PY_ROOT = Path(__file__).resolve().parents[1] / "Python"
_EXAMPLES = _PY_ROOT / "rocke" / "examples"
_GOLDEN = (
    Path(__file__).resolve().parents[0]
    / "golden"
    / "rocke_representative_ir_sha256.json"
)


def _read_json(path: Path):
    return json.loads(path.read_text())


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
        if limit is not None and len(rows) >= limit:
            break
    return rows


class TestExampleProblemSamples(unittest.TestCase):
    """Keep non-GPU CI anchored to shapes shipped with examples."""

    def test_gemm_example_shape_expands_dispatch_plan(self):
        data = _read_json(
            _EXAMPLES / "gfx1151" / "gemm" / "data" / "01_f16_verify.json"
        )
        shape = data["shape"]
        sweep_shape = GemmSweepShape(
            int(shape["M"]),
            int(shape["N"]),
            int(shape["K"]),
            label="gfx1151-example-f16-verify",
            verify=True,
        )

        plan = expand_sweep(GemmSweepConfig(arch="gfx950", shapes=(sweep_shape,)))

        self.assertGreater(len(plan.variants), 0)
        self.assertEqual(plan.config.shapes[0].as_tuple(), (128, 128, 128))
        self.assertTrue(plan.config.shapes[0].verify)

    def test_gfx950_attention_trace_samples_decode_and_prefill(self):
        rows = _read_jsonl(_EXAMPLES / "gfx950" / "attention" / "aiter_ua_shapes.json")
        kinds = {row["kind"] for row in rows}
        head_sizes = {int(row["head_size"]) for row in rows}
        max_q = {int(row["max_seqlen_q"]) for row in rows}

        self.assertIn("2d", kinds)
        self.assertIn(64, head_sizes)
        self.assertTrue(any(q == 1 for q in max_q), "decode sample missing")
        self.assertTrue(any(q > 1 for q in max_q), "prefill sample missing")

    def test_qwen_a3b_example_has_gpu_smoke_targets(self):
        data = _read_json(_EXAMPLES / "data" / "dsl_a3b_full_model.json")
        layer_names = {row["name"] for row in data["decode_layers"]}

        self.assertIn("qkv_proj", layer_names)
        self.assertIn("decode_attn", layer_names)
        self.assertIn("moe_e2e", layer_names)
        self.assertEqual(data["model"], "Qwen3-30B-A3B")


class TestIrParityCoverage(unittest.TestCase):
    """Validate that static parity samples span architectures and families."""

    def test_ir_cases_cover_all_static_architectures(self):
        arch_counts: dict[str, int] = {}
        family_by_arch: dict[str, set[str]] = {}

        for case in cases():
            arch = case["arch"]
            family = case["family"]
            arch_counts[arch] = arch_counts.get(arch, 0) + 1
            family_by_arch.setdefault(arch, set()).add(family)

        expected_arches = set(known_arches()) | {"gfx1201"}
        self.assertLessEqual(expected_arches, set(arch_counts))
        for arch in expected_arches:
            self.assertGreater(arch_counts[arch], 0, f"{arch} has no IR cases")
        for arch in ("gfx942", "gfx950", "gfx1151", "gfx1201"):
            self.assertIn("gemm", family_by_arch[arch])

        self.assertIn("unified_attention", family_by_arch["gfx950"])
        self.assertIn("moe", family_by_arch["gfx950"])
        self.assertIn("deep_fused_conv", family_by_arch["gfx950"])

    def test_ir_case_ids_are_unique(self):
        case_ids = [case["case_id"] for case in cases()]
        self.assertEqual(len(case_ids), len(set(case_ids)))

    def test_ir_cases_match_golden_sha256(self):
        """Byte-stability gate: every case's lowered-IR sha256 (and the set of
        expected failures) must match the committed golden for this host's llvm
        flavor. A drift here means emitted IR changed -- if intended & reviewed,
        re-bless with:
          python tests/instances/rocke_ir_parity_harness.py \\
            --write tests/golden/rocke_representative_ir_sha256.json
        """
        self.assertTrue(
            _GOLDEN.exists(),
            f"golden missing: {_GOLDEN} (bless with harness --write)",
        )
        drift = check_golden(_GOLDEN, current_flavor())
        self.assertEqual(drift, [], "IR drift vs golden:\n  " + "\n  ".join(drift))


if __name__ == "__main__":
    unittest.main(verbosity=2)
