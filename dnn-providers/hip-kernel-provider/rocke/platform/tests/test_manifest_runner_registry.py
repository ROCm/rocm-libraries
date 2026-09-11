# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The manifest runner's kind registry.

``run_manifest`` used to route on a hand-maintained ``if kind == ...`` chain,
which meant a family whose buffer knowledge lives outside this package could
not be run without editing the shipped wheel. These tests pin the replacement:
the same kinds are still served, and a new one can be added from outside.
"""

from __future__ import annotations

import unittest

from rocke.run_manifest import register_manifest_runner, registered_manifest_kinds

# Every kind the pre-registry branch chain handled. Losing one of these is a
# silent loss of the ability to run an already-emitted manifest, so the list is
# spelled out rather than derived.
_CHAIN_KINDS = frozenset(
    {
        "batched_gemm_fp16",
        "conv_bf16",
        "conv_fp16",
        "conv_fp32",
        "deep_fused_conv_pool_fp16",
        "deep_fused_conv_pool_i8i4",
        "elementwise_fp16",
        "gemm_fp16",
        "gemm_iu8",
        "layernorm_fp16",
        "matmul_nbits_fp16",
        "reduce_fp16",
        "rmsnorm_fp16",
        "transpose_fp16",
    }
)


class TestRunnerRegistry(unittest.TestCase):
    def test_serves_every_kind_the_branch_chain_did(self):
        self.assertEqual(frozenset(registered_manifest_kinds()), _CHAIN_KINDS)

    def test_kinds_are_reported_sorted_for_stable_error_messages(self):
        kinds = registered_manifest_kinds()
        self.assertEqual(list(kinds), sorted(kinds))

    def test_a_family_outside_this_package_can_register(self):
        def _builder(manifest, shape, verify):
            raise AssertionError("not called")

        register_manifest_runner("test_only_kind", _builder)
        try:
            self.assertIn("test_only_kind", registered_manifest_kinds())
        finally:
            from rocke import run_manifest as rm

            rm._RUNNERS.pop("test_only_kind", None)

    def test_conflicting_registration_is_refused(self):
        # Two modules claiming one kind is a real ambiguity: whichever imported
        # last would silently win. Name the incumbent so the clash is diagnosable.
        with self.assertRaises(ValueError) as ctx:
            register_manifest_runner("gemm_fp16", lambda m, s, v: None)
        self.assertIn("gemm_fp16", str(ctx.exception))
        self.assertIn("run_gemm_manifest_problem", str(ctx.exception))

    def test_re_registering_the_same_builder_is_idempotent(self):
        from rocke.instances.common.manifest_runner.gemm import (
            run_gemm_manifest_problem,
        )

        register_manifest_runner("gemm_fp16", run_gemm_manifest_problem)
        self.assertIn("gemm_fp16", registered_manifest_kinds())

    def test_running_a_manifest_does_not_require_the_dispatcher(self):
        # Most manifest workflows never touch dispatch: ten of the fourteen
        # registered kinds have no candidate at all (gemm_iu8, matmul_nbits,
        # the deep-fused conv pools, the simple ops...), and the examples that
        # emit them build a spec, compile, and shell out to this module. If
        # `bind` ever became a precondition rather than one more registerable
        # adapter, those all break. Importing dispatch here is the first step
        # down that road, so fail on it directly.
        import os
        import subprocess
        import sys

        probe = (
            "import rocke.run_manifest, sys; "
            "leaked = sorted(m for m in sys.modules if m.startswith('rocke.dispatch')); "
            "print(leaked)"
        )
        # A fresh interpreter inherits the environment but not the sys.path this
        # suite arranges, so hand that down. Without it `import rocke` fails in a
        # checkout and the probe reports an import error rather than a leak.
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
        out = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=True,
            env=env,
        )
        self.assertEqual(
            out.stdout.strip(), "[]", "run_manifest must not pull in dispatch"
        )

    def test_every_builder_takes_the_dispatch_free_contract(self):
        # A registered builder is called as builder(manifest, shape, verify).
        # Nothing may demand a DispatchResult: a hand-written manifest has none.
        import inspect

        from rocke import run_manifest as rm

        for kind, builder in rm._RUNNERS.items():
            with self.subTest(kind=kind):
                params = list(inspect.signature(builder).parameters.values())
                required = [
                    p
                    for p in params
                    if p.default is inspect.Parameter.empty
                    and p.kind
                    in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
                ]
                self.assertEqual(len(required), 3, f"{kind} takes {required}")

    def test_unknown_kind_error_lists_what_is_available(self):
        import json
        import tempfile
        from pathlib import Path

        from rocke.run_manifest import run_manifest

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "k.hsaco").write_bytes(b"\x00")
            (root / "manifest.json").write_text(
                json.dumps(
                    {"kind": "no_such_kind", "kernel_name": "k", "hsaco": "k.hsaco"}
                )
            )
            with self.assertRaises(ValueError) as ctx:
                run_manifest(root / "manifest.json")
        message = str(ctx.exception)
        self.assertIn("no_such_kind", message)
        self.assertIn("gemm_fp16", message)
        self.assertIn("runner_module", message)

    def test_runner_module_is_imported_before_kind_lookup(self):
        import json
        import sys
        import tempfile
        from pathlib import Path

        from rocke.run_manifest import resolve_manifest_runner
        from rocke import run_manifest as rm

        kind = "kda_test_only_kind"
        self.assertNotIn(kind, rm.registered_manifest_kinds())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "ck_test_kda_runner.py").write_text(
                "from rocke.run_manifest import register_manifest_runner\n"
                "def _builder(manifest, shape, verify):\n"
                "    raise AssertionError('not called')\n"
                "register_manifest_runner('kda_test_only_kind', _builder)\n"
            )
            sys.path.insert(0, tmp)
            try:
                builder = resolve_manifest_runner(
                    {
                        "kind": kind,
                        "runner_module": "ck_test_kda_runner",
                    }
                )
                self.assertEqual(builder.__name__, "_builder")
                self.assertIn(kind, rm.registered_manifest_kinds())
            finally:
                sys.path.remove(tmp)
                rm._RUNNERS.pop(kind, None)
                sys.modules.pop("ck_test_kda_runner", None)

    def test_missing_runner_module_is_an_import_error(self):
        from rocke.run_manifest import resolve_manifest_runner

        with self.assertRaises(ImportError):
            resolve_manifest_runner(
                {
                    "kind": "no_such_kind",
                    "runner_module": "ck_test_no_such_runner_module",
                }
            )


if __name__ == "__main__":
    unittest.main()
