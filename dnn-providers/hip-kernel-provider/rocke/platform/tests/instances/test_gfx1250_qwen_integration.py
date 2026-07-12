# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""No-GPU tests for the gfx1250 Qwen3-30B-A3B day-0 integration scaffolding:
gate routing, graph-capture-safe custom ops, and the one-layer reference
harness. Arch facts and per-operator contracts live in the
``test_gfx1250_{foundation,attention,gemm,moe}`` suites.
"""

from __future__ import annotations

import dataclasses
import unittest

from rocke.examples.gfx1250.qwen3_30b_a3b.qwen3_30b_a3b_integration import (
    GATE_ENVS,
    MASTER_ENV,
    GateConfig,
    OperatorAvailability,
    QwenOneLayerHarness,
    custom_op_specs,
    plan_qwen_layer,
    register_torch_custom_ops,
    select_route,
)
from rocke.examples.gfx1250.qwen3_30b_a3b.qwen3_30b_a3b_shapes import (
    ARCH,
    QWEN3_30B_A3B_CONFIG,
    Qwen3A3BConfig,
    layer_shapes,
)


class TestGfx1250QwenLayerTable(unittest.TestCase):
    def test_qwen_shape_table_routes_expected_decode_shapes(self):
        shapes = {shape.name: shape for shape in layer_shapes()}
        self.assertEqual(shapes["qkv_proj"].dims["M"], 2)
        self.assertEqual(shapes["qkv_proj"].dims["N"], 2560)
        self.assertEqual(shapes["qkv_proj"].dims["K"], 2048)
        self.assertEqual(shapes["decode_attention"].dims["nhead_q"], 32)
        self.assertEqual(shapes["decode_attention"].dims["nhead_k"], 4)
        self.assertEqual(shapes["router_topk"].dims["experts"], 128)
        self.assertEqual(shapes["moe_e2e"].dims["topk"], 8)

    def test_default_config_is_full_qwen_shape(self):
        self.assertEqual(QWEN3_30B_A3B_CONFIG.hidden, 2048)
        self.assertEqual(QWEN3_30B_A3B_CONFIG.num_experts, 128)
        self.assertEqual(QWEN3_30B_A3B_CONFIG.arch, ARCH)


class TestGfx1250QwenGateRouting(unittest.TestCase):
    def test_default_env_keeps_everything_on_fallback(self):
        plan = plan_qwen_layer(gates=GateConfig.from_env({}))
        self.assertTrue(plan)
        self.assertTrue(all(not decision.enabled for decision in plan))
        self.assertTrue(all(not decision.accelerated for decision in plan))
        self.assertTrue(all("master gate" in decision.reason for decision in plan))

    def test_independent_gemm_gate_selects_only_available_gemm_shapes(self):
        gates = GateConfig.from_env(
            {
                MASTER_ENV: "1",
                GATE_ENVS["gemm"]: "1",
                GATE_ENVS["attention"]: "0",
                GATE_ENVS["norm"]: "0",
                GATE_ENVS["routing"]: "0",
                GATE_ENVS["moe"]: "0",
            }
        )
        availability = OperatorAvailability.from_mapping({"gemm": True})
        plan = {
            decision.op_name: decision
            for decision in plan_qwen_layer(gates=gates, availability=availability)
        }
        self.assertTrue(plan["qkv_proj"].accelerated)
        self.assertTrue(plan["o_proj"].accelerated)
        self.assertEqual(
            plan["qkv_proj"].backend, "rocke.gfx1250.qwen3_30b_a3b.dense_gemm"
        )
        self.assertFalse(plan["decode_attention"].accelerated)
        self.assertIn(GATE_ENVS["attention"], plan["decode_attention"].reason)
        self.assertFalse(plan["router_topk"].accelerated)

    def test_enabled_gate_still_falls_back_without_operator_branch(self):
        gates = GateConfig.from_env({MASTER_ENV: "1", GATE_ENVS["attention"]: "1"})
        decision = {item.op_name: item for item in plan_qwen_layer(gates=gates)}[
            "decode_attention"
        ]
        self.assertTrue(decision.enabled)
        self.assertFalse(decision.accelerated)
        self.assertIn("operator branch unavailable", decision.reason)
        self.assertEqual(decision.backend, decision.fallback_backend)

    def test_shape_routing_rejects_non_qwen_shape(self):
        qkv = {shape.name: shape for shape in layer_shapes()}["qkv_proj"]
        bad_qkv = dataclasses.replace(qkv, dims={**qkv.dims, "N": 4096})
        gates = GateConfig.from_env({MASTER_ENV: "1", GATE_ENVS["gemm"]: "1"})
        decision = select_route(
            bad_qkv,
            gates=gates,
            availability=OperatorAvailability.from_mapping({"gemm": True}),
        )
        self.assertFalse(decision.enabled)
        self.assertFalse(decision.accelerated)
        self.assertIn("not in the Qwen3-30B-A3B gfx1250 routing table", decision.reason)

    def test_all_gates_can_be_enabled_independently_for_e2e_scaffold(self):
        gates = GateConfig.from_env(
            {
                MASTER_ENV: "1",
                GATE_ENVS["gemm"]: "1",
                GATE_ENVS["attention"]: "1",
                GATE_ENVS["norm"]: "1",
                GATE_ENVS["routing"]: "1",
                GATE_ENVS["moe"]: "1",
            }
        )
        availability = OperatorAvailability.from_mapping(
            {
                "gemm": True,
                "attention": True,
                "norm": True,
                "routing": True,
                "moe": True,
            }
        )
        plan = plan_qwen_layer(gates=gates, availability=availability)
        self.assertTrue(all(decision.enabled for decision in plan))
        self.assertTrue(all(decision.accelerated for decision in plan))
        self.assertTrue(
            {decision.gate for decision in plan}
            >= {"gemm", "attention", "norm", "routing", "moe"}
        )


class TestGfx1250QwenCustomOps(unittest.TestCase):
    def test_custom_op_specs_are_graph_capture_compatible_out_abis(self):
        specs = {spec.name: spec for spec in custom_op_specs()}
        self.assertIn("rmsnorm_add_out", specs)
        self.assertIn("router_topk_out", specs)
        for spec in specs.values():
            self.assertTrue(spec.graph_capture_safe)
            self.assertIn("Tensor(a!)", spec.schema)
            self.assertTrue(spec.schema.endswith("-> ()"))

    def test_custom_op_cpu_reference_impls_when_torch_is_available(self):
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch unavailable: {exc}")

        self.assertTrue(register_torch_custom_ops())
        x = torch.randn(2, 4)
        residual = torch.randn(2, 4) * 0.01
        weight = torch.ones(4)
        norm_out = torch.empty_like(x)
        residual_out = torch.empty_like(x)
        torch.ops.rocke_gfx1250_qwen.rmsnorm_add_out(
            x, residual, weight, norm_out, residual_out, 1e-6
        )
        self.assertTrue(torch.isfinite(norm_out).all().item())
        self.assertTrue(torch.allclose(residual_out, x + residual))

        logits = torch.randn(2, 6)
        weights_out = torch.empty(2, 3)
        ids_out = torch.empty(2, 3, dtype=torch.int32)
        torch.ops.rocke_gfx1250_qwen.router_topk_out(logits, weights_out, ids_out, 3)
        self.assertTrue(torch.allclose(weights_out.sum(dim=-1), torch.ones(2)))
        self.assertEqual(ids_out.dtype, torch.int32)


class TestGfx1250QwenOneLayerHarness(unittest.TestCase):
    def test_tiny_one_layer_reference_runs_with_independent_gates(self):
        try:
            import torch
        except Exception as exc:
            self.skipTest(f"torch unavailable: {exc}")

        config = Qwen3A3BConfig(
            batch=2,
            hidden=4,
            moe_intermediate=3,
            num_experts=4,
            topk=2,
            num_query_heads=1,
            num_kv_heads=1,
            head_dim=4,
        )
        gates = GateConfig.from_env({MASTER_ENV: "1", GATE_ENVS["routing"]: "1"})
        harness = QwenOneLayerHarness(
            config=config,
            gates=gates,
            availability=OperatorAvailability.from_mapping({"routing": True}),
        )
        out = harness.run_reference_torch(torch)
        self.assertEqual(out["hidden_states"].shape, (2, 4))
        self.assertEqual(out["router_weights"].shape, (2, 2))
        self.assertEqual(out["router_ids"].shape, (2, 2))
        self.assertEqual(out["moe_out"].shape, (2, 4))
        self.assertTrue(torch.isfinite(out["moe_out"]).all().item())
        plan = {decision.op_name: decision for decision in out["plan"]}
        self.assertTrue(plan["router_topk"].accelerated)
        self.assertFalse(plan["qkv_proj"].accelerated)


if __name__ == "__main__":
    unittest.main(verbosity=2)
