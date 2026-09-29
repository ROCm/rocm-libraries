# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Executable attention candidates own a complete launch contract."""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest import mock

from dispatch.attention import (
    ATTENTION_EXECUTION_REGISTRY,
    ATTENTION_ROUTE_REGISTRY,
    AttentionRequest,
    attention_execution_candidates,
)
from dispatch.attention import bindings as attention_bindings
from dispatch.attention.common import AttentionTuningSpec, _problem
from kernels.common.attention_dense_spec import AttentionDenseSpec


def _req(arch="gfx950", **kw) -> AttentionRequest:
    base = dict(
        batch=1,
        nhead_q=32,
        nhead_k=8,
        seqlen_q=1024,
        seqlen_k=1024,
        hdim_q=128,
        hdim_v=128,
        arch=arch,
        dtype="bf16",
        mask_type=1,
    )
    base.update(kw)
    return AttentionRequest(**base)


class _Tensor:
    def __init__(
        self,
        shape,
        dtype,
        values=None,
        *,
        device="cuda:0",
        contiguous=True,
        strides=None,
    ):
        self.shape = tuple(shape)
        self.dtype = dtype
        self.device = device
        self._values = values
        self._contiguous = contiguous
        self._strides = strides

    def is_contiguous(self):
        return self._contiguous

    def detach(self):
        return self

    def stride(self, dim):
        if self._strides is not None:
            return self._strides[dim]
        stride = 1
        values = [0] * len(self.shape)
        for index in range(len(self.shape) - 1, -1, -1):
            values[index] = stride
            stride *= self.shape[index]
        return values[dim]

    def cpu(self):
        return self

    def tolist(self):
        if self._values is None:
            raise AssertionError("no host values")
        return self._values


def _paged_tensors(*, num_blocks=64):
    return {
        "q": _Tensor((1024, 32, 128), "torch.bfloat16"),
        "out": _Tensor((1024, 32, 128), "torch.bfloat16"),
        "k": _Tensor((num_blocks, 16, 8, 128), "torch.bfloat16"),
        "v": _Tensor((num_blocks, 16, 8, 128), "torch.bfloat16"),
        "cu_seqlens_q": _Tensor((2,), "torch.int32", [0, 1024]),
        "seqused_k": _Tensor((1,), "torch.int32", [1024]),
        "block_table": _Tensor((1, 64), "torch.int32", [list(range(64))]),
    }


class TestRegistrySplit(unittest.TestCase):
    def test_routing_labels_are_not_on_the_execution_registry(self):
        exec_names = {c.name for c in attention_execution_candidates()}
        route_names = {c.name for c in ATTENTION_ROUTE_REGISTRY.candidates()}
        for name in (
            "attention_unified_2d",
            "attention_unified_3d",
            "attention_d256_decode",
            "attention_gfx942_dense_pipe",
            "attention_gfx950_d256",
        ):
            self.assertIn(name, route_names)
            self.assertNotIn(name, exec_names)

    def test_route_only_candidates_are_not_executable(self):
        for name in (
            "attention_unified_2d",
            "attention_unified_3d",
            "attention_d256_decode",
            "attention_gfx942_dense_pipe",
            "attention_gfx950_d256",
        ):
            with self.subTest(name=name):
                candidate = ATTENTION_ROUTE_REGISTRY.get(name)
                self.assertIsNone(candidate.build)
                self.assertIsNone(candidate.bind_torch)

    def test_every_execution_candidate_is_buildable_and_torch_bindable(self):
        self.assertTrue(ATTENTION_EXECUTION_REGISTRY.require_build)
        self.assertTrue(ATTENTION_EXECUTION_REGISTRY.require_torch_binding)
        for candidate in attention_execution_candidates():
            with self.subTest(candidate=candidate.name):
                self.assertIsNotNone(candidate.build)
                self.assertIsNotNone(candidate.bind_torch)

    def test_execution_registry_rejects_a_candidate_without_build(self):
        from dispatch.attention.common import (
            ATTENTION_ABI_VERSION,
            ATTENTION_DIM_VOCABULARY,
            FAMILY,
        )
        from rocke.dispatch.core import (
            Capability,
            CandidateRegistry,
            KernelCandidate,
        )

        registry = CandidateRegistry(
            FAMILY,
            dim_vocabulary=ATTENTION_DIM_VOCABULARY,
            require_build=True,
            require_torch_binding=True,
        )
        candidate = KernelCandidate(
            name="attention_missing_build",
            family=FAMILY,
            algorithm="probe",
            spec_id="probe",
            abi_version=ATTENTION_ABI_VERSION,
            priority=100,
            capability=Capability(arches=("gfx950",), dtypes=("bf16",)),
            _supports=lambda _req: (True, "ok"),
            select_spec=lambda _req: None,
            signature=lambda _spec: (),
            grid=lambda _spec, _req: (1, 1, 1),
            block=lambda _spec: (1, 1, 1),
            sweep_space=lambda _req: (),
            build=None,
            bind_torch=lambda *_args, **_kw: None,
        )
        with self.assertRaisesRegex(ValueError, "declares no build"):
            registry.register(candidate)

    def test_dense_select_spec_returns_the_concrete_dense_spec(self):
        candidate = ATTENTION_EXECUTION_REGISTRY.get("attention_gfx950_dense")
        req = _req(algorithm="attention_dense")
        spec = candidate.select_spec(req)
        self.assertIsInstance(spec, AttentionDenseSpec)
        self.assertNotEqual(candidate.grid(spec, req), (0, 0, 0))
        self.assertNotEqual(candidate.block(spec), (0, 0, 0))
        self.assertTrue(candidate.signature(spec))

    def test_tuning_select_spec_returns_an_attention_tuning_spec(self):
        candidate = next(
            c
            for c in attention_execution_candidates()
            if c.name.startswith("attention_gfx950_u2d_narrow_nw2_mw16_t4xb_llvm")
        )
        req = _req(algorithm=candidate.algorithm, spec_id=candidate.spec_id)
        spec = candidate.select_spec(req)
        self.assertIsInstance(spec, AttentionTuningSpec)
        self.assertIn("@", spec.tuning_id)
        self.assertNotEqual(candidate.grid(spec, req), (0, 0, 0))
        built = candidate.built(spec, "gfx950")
        self.assertTrue(getattr(built, "name", None) or built)

    def test_gfx942_dense_bind_torch_omits_paged_kwargs(self):
        candidate = ATTENTION_EXECUTION_REGISTRY.get("attention_gfx942_dense")
        req = _req(arch="gfx942", dtype="fp16", algorithm="attention_dense")
        spec = candidate.select_spec(req)
        tensors = {"q": object(), "k": object(), "v": object(), "out": object()}
        captured = {}

        def fake_run(
            *,
            spec,
            q,
            k,
            v,
            out,
            scale,
            stream=0,
            arch="gfx942",
            cu_seqlens_q=None,
            cu_seqlens_kv=None,
        ):
            captured.update(
                spec=spec,
                q=q,
                k=k,
                v=v,
                out=out,
                scale=scale,
                stream=stream,
                arch=arch,
            )
            return "ok"

        with mock.patch.object(
            attention_bindings,
            "_dense_runner",
            return_value=(fake_run, lambda _s: (1, 1, 1), lambda _s: (64, 1, 1)),
        ):
            binding = attention_bindings.bind_dense_attention_torch(req, spec, tensors)
            binding.launch()
        self.assertEqual(captured["q"], tensors["q"])
        self.assertEqual(captured["arch"], "gfx942")
        self.assertNotIn("block_tables", captured)
        self.assertNotIn("kv_lens", captured)
        self.assertNotIn("sinks", captured)

    def test_explicit_binding_validates_every_paged_input_class(self):
        problem = _problem(_req())
        valid = _paged_tensors()
        self.assertEqual(
            attention_bindings.validate_tuning_attention_tensors(problem, valid),
            64,
        )
        row_padded = dict(valid)
        row_padded["block_table"] = _Tensor(
            (1, 64),
            "torch.int32",
            [list(range(64))],
            contiguous=False,
            strides=(80, 1),
        )
        self.assertEqual(
            attention_bindings.validate_tuning_attention_tensors(problem, row_padded),
            64,
        )
        cases = {
            "q_shape": ("q", _Tensor((1023, 32, 128), "torch.bfloat16")),
            "k_dtype": ("k", _Tensor((64, 16, 8, 128), "torch.float16")),
            "v_shape": ("v", _Tensor((63, 16, 8, 128), "torch.bfloat16")),
            "cu_dtype": (
                "cu_seqlens_q",
                _Tensor((2,), "torch.float16", [0, 1024]),
            ),
            "seqused_range": (
                "seqused_k",
                _Tensor((1,), "torch.int32", [1025]),
            ),
            "block_table_shape": (
                "block_table",
                _Tensor((2, 32), "torch.int32", [list(range(32))] * 2),
            ),
            "block_table_value": (
                "block_table",
                _Tensor(
                    (1, 64),
                    "torch.int32",
                    [[64] + list(range(1, 64))],
                ),
            ),
            "block_table_stride": (
                "block_table",
                _Tensor(
                    (1, 64),
                    "torch.int32",
                    [list(range(64))],
                    strides=(128, 2),
                ),
            ),
        }
        for label, (name, replacement) in cases.items():
            with self.subTest(name=label), self.assertRaises(ValueError):
                tensors = dict(valid)
                tensors[name] = replacement
                attention_bindings.validate_tuning_attention_tensors(problem, tensors)
        tensors = dict(valid)
        tensors["k"] = _Tensor((64, 16, 8, 128), "torch.bfloat16", contiguous=False)
        with self.assertRaisesRegex(ValueError, "contiguous"):
            attention_bindings.validate_tuning_attention_tensors(problem, tensors)

    def test_explicit_binding_can_skip_only_metadata_content_sync(self):
        problem = _problem(_req())
        tensors = _paged_tensors()
        tensors["block_table"] = _Tensor(
            (1, 64), "torch.int32", [[999] + list(range(1, 64))]
        )
        self.assertEqual(
            attention_bindings.validate_tuning_attention_tensors(
                problem, tensors, validate_contents=False
            ),
            64,
        )

    def test_explicit_binding_ignores_invalid_unused_table_columns(self):
        problem = _problem(_req())
        tensors = _paged_tensors()
        tensors["seqused_k"] = _Tensor((1,), "torch.int32", [16])
        tensors["block_table"] = _Tensor(
            (1, 64),
            "torch.int32",
            [[0] + [999] * 63],
        )
        self.assertEqual(
            attention_bindings.validate_tuning_attention_tensors(problem, tensors),
            64,
        )

    def test_explicit_binding_rejects_cpu_and_semantic_mismatches(self):
        request = _req()
        problem = _problem(request)
        tensors = _paged_tensors()
        for tensor in tensors.values():
            tensor.device = "cpu"
        with self.assertRaisesRegex(ValueError, "HIP/CUDA"):
            attention_bindings.validate_tuning_attention_tensors(problem, tensors)

        candidate = next(
            c
            for c in attention_execution_candidates()
            if c.name.startswith("attention_gfx950_u2d_narrow_nw2_mw16_t4xb_llvm")
        )
        spec = candidate.select_spec(
            replace(request, algorithm=candidate.algorithm, spec_id=candidate.spec_id)
        )
        with self.assertRaisesRegex(ValueError, "num_query_heads"):
            attention_bindings.validate_tuning_attention_contract(
                request,
                replace(problem, num_query_heads=16),
                spec,
            )

    def test_explicit_binding_refreshes_large_cache_addressing(self):
        candidate = next(
            c
            for c in attention_execution_candidates()
            if c.name.startswith("attention_gfx950_u2d_narrow_nw2_mw16_t4xb_llvm")
        )
        req = _req(algorithm=candidate.algorithm, spec_id=candidate.spec_id)
        spec = candidate.select_spec(req)
        tensors = _paged_tensors(num_blocks=65537)
        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return "ok"

        with mock.patch(
            "kernels.common.attention_unified.run_unified_attention_torch",
            side_effect=fake_run,
        ):
            binding = candidate.bound_torch(req, spec, tensors)
            binding.launch()
        runtime_spec = captured["tuning_spec"]
        self.assertTrue(runtime_spec.kernel_spec.use_i64_kv_addr)
        self.assertEqual(runtime_spec.num_kv_blocks, 65537)
        self.assertNotEqual(runtime_spec.tuning_id, spec.tuning_id)


if __name__ == "__main__":
    unittest.main()
