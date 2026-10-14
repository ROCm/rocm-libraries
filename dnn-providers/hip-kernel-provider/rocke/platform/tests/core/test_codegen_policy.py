# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only contracts for typed per-kernel code-generation policy."""

from __future__ import annotations

import inspect
import unittest

from rocke.core.codegen_policy import (
    CodegenPolicy,
    SchedulerStrategy,
    apply_codegen_policy,
    codegen_policy_for_kernel,
)
from rocke.core.ir import IRBuilder
from rocke.core.ir_serialize import parse, serialize
from rocke.core.lower_llvm import _lower_kernel_to_llvm_python
from rocke.helpers.autotune import AutotuneConfig
from rocke.helpers.compile import (
    _comgr_options_for_kernel,
    compile_kernel_via_hipcc,
)


def _kernel():
    builder = IRBuilder("codegen_policy_test")
    builder.kernel.attrs["max_workgroup_size"] = 64
    builder.ret()
    return builder.kernel


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

    def test_hipcc_rejects_a_policy_it_cannot_honor(self):
        kernel = _kernel()
        apply_codegen_policy(kernel, CodegenPolicy(scheduler_strategy="max-ilp"))
        with self.assertRaisesRegex(ValueError, "does not support scheduler_strategy"):
            compile_kernel_via_hipcc(kernel)

    def test_autotune_extra_carries_policy_without_api_change(self):
        policy = CodegenPolicy(scheduler_strategy="iterative-maxocc")
        config = AutotuneConfig(
            spec=object(),
            name="iterative-maxocc",
            extra={"codegen_policy": policy},
        )
        self.assertEqual(
            tuple(inspect.signature(AutotuneConfig).parameters),
            ("spec", "name", "extra"),
        )
        self.assertIs(config.extra["codegen_policy"], policy)


if __name__ == "__main__":
    unittest.main()
