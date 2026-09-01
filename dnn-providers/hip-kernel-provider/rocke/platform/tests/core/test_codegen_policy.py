# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only contracts for typed per-kernel code-generation policy."""

from __future__ import annotations

import inspect
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rocke.core.codegen_policy import (
    CodegenPolicy,
    SchedulerStrategy,
    apply_codegen_policy,
    codegen_policy_for_kernel,
)
from rocke.core.ir import IRBuilder
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.dispatch.core import KernelId
from rocke.helpers.autotune import AutotuneConfig, Autotuner
from rocke.helpers.compile import (
    _comgr_options_for_kernel,
    compile_kernel,
    compile_kernel_via_hipcc,
)
from rocke.helpers.manifest import make_simple_op_manifest


def _kernel():
    builder = IRBuilder("codegen_policy_test")
    builder.kernel.attrs["max_workgroup_size"] = 64
    builder.ret()
    return builder.kernel


def _kernel_id() -> KernelId:
    return KernelId(
        op="gemm",
        family="gemm_fp16_rcr",
        candidate="default",
        algorithm="cshuffle",
        spec_id="default",
        arch="gfx950",
        abi_version="test/v1",
        request_hash="1111111111111111",
        spec_hash="2222222222222222",
    )


class TestCodegenPolicy(unittest.TestCase):
    def test_every_supported_scheduler_strategy_is_canonical(self):
        expected = {
            "max-ilp",
            "max-memory-clause",
            "iterative-ilp",
            "iterative-minreg",
            "iterative-maxocc",
        }
        self.assertEqual({strategy.value for strategy in SchedulerStrategy}, expected)
        for strategy in SchedulerStrategy:
            with self.subTest(strategy=strategy.value):
                policy = CodegenPolicy(scheduler_strategy=strategy)
                self.assertEqual(policy.scheduler_strategy, strategy.value)
                self.assertEqual(
                    CodegenPolicy(scheduler_strategy=strategy.value).scheduler_strategy,
                    strategy.value,
                )
                kernel = _kernel()
                apply_codegen_policy(kernel, policy)
                self.assertIn(
                    f'"amdgpu-sched-strategy"="{strategy.value}"',
                    _lower_kernel_to_llvm_python(kernel, arch="gfx950"),
                )

    def test_invalid_scheduler_strategies_are_rejected(self):
        for value in ("", "default", "ITERATIVE-ILP", 1, True, object()):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    CodegenPolicy(scheduler_strategy=value)

    def test_apply_and_remove_policy(self):
        kernel = _kernel()
        apply_codegen_policy(
            kernel, CodegenPolicy(scheduler_strategy="iterative-minreg")
        )
        self.assertEqual(
            codegen_policy_for_kernel(kernel),
            CodegenPolicy(scheduler_strategy="iterative-minreg"),
        )
        apply_codegen_policy(kernel, CodegenPolicy())
        self.assertNotIn("scheduler_strategy", kernel.attrs)

    def test_apply_rejects_untyped_policy(self):
        with self.assertRaises(TypeError):
            apply_codegen_policy(_kernel(), {"scheduler_strategy": "max-ilp"})

    def test_policy_survives_serialization_round_trip(self):
        kernel = _kernel()
        apply_codegen_policy(kernel, CodegenPolicy(scheduler_strategy="max-ilp"))
        reparsed = parse(serialize(kernel))
        self.assertEqual(
            codegen_policy_for_kernel(reparsed), codegen_policy_for_kernel(kernel)
        )
        self.assertEqual(serialize(reparsed), serialize(kernel))

    def test_default_policy_does_not_change_lowered_llvm(self):
        kernel = _kernel()
        before = _lower_kernel_to_llvm_python(kernel, arch="gfx950")
        apply_codegen_policy(kernel, CodegenPolicy())
        after = _lower_kernel_to_llvm_python(kernel, arch="gfx950")
        self.assertEqual(after, before)
        self.assertNotIn("amdgpu-sched-strategy", after)

    def test_scheduler_attribute_has_stable_order(self):
        kernel = _kernel()
        kernel.attrs["waves_per_eu"] = 2
        apply_codegen_policy(
            kernel, CodegenPolicy(scheduler_strategy="iterative-maxocc")
        )
        llvm = _lower_kernel_to_llvm_python(kernel, arch="gfx950")
        attrs = next(
            line for line in llvm.splitlines() if line.startswith("attributes #0")
        )
        self.assertIn(
            '"amdgpu-flat-work-group-size"="64,64" '
            '"amdgpu-sched-strategy"="iterative-maxocc" '
            '"amdgpu-waves-per-eu"="2,2"',
            attrs,
        )

    def test_scheduler_policy_is_not_forwarded_as_a_raw_comgr_flag(self):
        kernel = _kernel()
        apply_codegen_policy(kernel, CodegenPolicy(scheduler_strategy="max-ilp"))
        self.assertEqual(_comgr_options_for_kernel(kernel), ["-O3"])

    def test_compile_artifact_and_manifest_record_policy(self):
        kernel = _kernel()
        policy = CodegenPolicy(scheduler_strategy="max-memory-clause")
        apply_codegen_policy(kernel, policy)
        timings = SimpleNamespace(bc=0.0, relocatable=0.0, executable=0.0)
        with patch(
            "rocke.helpers.compile._lower_llvm_via_backend", return_value="llvm"
        ), patch(
            "rocke.helpers.compile.build_hsaco_from_llvm_ir",
            return_value=(b"hsaco", timings),
        ):
            artifact = compile_kernel(kernel)
        self.assertEqual(artifact.codegen_policy, policy)
        manifest = make_simple_op_manifest(
            artifact=artifact,
            kind="elementwise_fp16",
            op="copy",
            dtype="f16",
            threads_per_block=64,
            default_shape=(64,),
            args_signature=[],
        )
        self.assertEqual(manifest["codegen_policy"], policy.as_dict())
        self.assertEqual(manifest["codegen_policy_key"], policy.cache_key)
        manifest = make_simple_op_manifest(
            artifact=artifact,
            kind="elementwise_fp16",
            op="copy",
            dtype="f16",
            threads_per_block=64,
            default_shape=(64,),
            args_signature=[],
            extra={"custom_field": "kept"},
        )
        self.assertEqual(manifest["custom_field"], "kept")
        for reserved in ("codegen_policy", "codegen_policy_key"):
            with self.subTest(reserved=reserved):
                with self.assertRaisesRegex(ValueError, reserved):
                    make_simple_op_manifest(
                        artifact=artifact,
                        kind="elementwise_fp16",
                        op="copy",
                        dtype="f16",
                        threads_per_block=64,
                        default_shape=(64,),
                        args_signature=[],
                        extra={reserved: "incorrect"},
                    )

    def test_hipcc_rejects_a_policy_it_cannot_honor(self):
        kernel = _kernel()
        apply_codegen_policy(kernel, CodegenPolicy(scheduler_strategy="max-ilp"))
        with self.assertRaisesRegex(ValueError, "does not support scheduler_strategy"):
            compile_kernel_via_hipcc(kernel)


class TestCodegenPolicyIdentity(unittest.TestCase):
    def test_default_kernel_id_is_byte_for_byte_unchanged(self):
        kernel_id = _kernel_id()
        self.assertEqual(
            kernel_id.compile_key,
            "gfx950:test/v1:2222222222222222",
        )
        self.assertNotIn("codegen_policy_key", kernel_id.as_dict())

    def test_kernel_id_distinguishes_explicit_policies(self):
        base = _kernel_id()
        ilp = base.with_codegen_policy(CodegenPolicy(scheduler_strategy="max-ilp"))
        occ = base.with_codegen_policy(
            CodegenPolicy(scheduler_strategy="iterative-maxocc")
        )
        self.assertNotEqual(ilp.compile_key, occ.compile_key)
        self.assertNotEqual(ilp.selection_key, occ.selection_key)
        self.assertIn("codegen_policy_key", ilp.as_dict())

    def test_autotuner_allows_same_name_with_distinct_policies(self):
        configs = [
            AutotuneConfig(
                spec=object(),
                name="same",
                extra={"codegen_policy": CodegenPolicy(scheduler_strategy="max-ilp")},
            ),
            AutotuneConfig(
                spec=object(),
                name="same",
                extra={
                    "codegen_policy": CodegenPolicy(
                        scheduler_strategy="iterative-maxocc"
                    )
                },
            ),
        ]
        tuner = Autotuner(
            configs=configs,
            key_fn=lambda **_: (),
            bench_fn=lambda cfg, **_: 1.0 if "maxocc" in cfg.identity else 2.0,
            launch_fn=lambda *_args, **_kwargs: None,
            cache_path=None,
        )
        self.assertEqual(tuner.select().codegen_policy, configs[1].codegen_policy)

    def test_autotune_policy_extra_defaults_and_rejects_untyped_values(self):
        self.assertEqual(
            AutotuneConfig(spec=object(), name="default").codegen_policy,
            CodegenPolicy(),
        )
        with self.assertRaisesRegex(TypeError, "must be CodegenPolicy"):
            AutotuneConfig(
                spec=object(),
                name="invalid",
                extra={"codegen_policy": {"scheduler_strategy": "max-ilp"}},
            )

    def test_autotune_config_constructor_signature_is_unchanged(self):
        self.assertEqual(
            tuple(inspect.signature(AutotuneConfig).parameters),
            ("spec", "name", "extra"),
        )


if __name__ == "__main__":
    unittest.main()
