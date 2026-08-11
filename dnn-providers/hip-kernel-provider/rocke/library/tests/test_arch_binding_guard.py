# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Guard: the spec builder must reach ``_resolve_attention_arch`` lazily.

``builders.common.attention_spec_builder`` used to bind the resolver with
``from kernels.common.attention_unified import _resolve_attention_arch``. A bound
import freezes the reference at import time, so ``mock.patch.object(au,
"_resolve_attention_arch", ...)`` -- which rebinds the attribute on the *module*
-- never reached the builder. The builder then resolved the real device arch,
and the gfx950-only spec overrides raised ``TypeError`` against a gfx942 spec
class. The binding dates to #9057 and turned load-bearing in #9233.

The rule is now: reach the resolver through the ``_kau`` module handle. Nothing
in the language enforces that, so these tests do -- one statically, one through
observable behaviour.

Both tests are host-independent. The behavioural one pins *two* archs, because
the resolver falls back to ``"gfx950"`` with no device and ``_tiled_2d_impl``
maps every arch except gfx942/gfx1250 onto the gfx950 impl -- so a single pin
can pass by coincidence on most hosts. Both builder entry points are covered:
``_tiled_spec_from_problem`` and ``_tiled_3d_spec_from_problem`` each hold their
own ``_kau._resolve_attention_arch()`` call site, and both were served by the
same bound import before the fix.

gfx1250 is deliberately NOT pinned: it needs its own problem shape (2D rejects
``block_size=16``, 3D rejects ``head_size=128``), so it would couple this guard
to the evolving gfx1250 support matrix while adding no discriminating power --
the gfx942 pin alone already fails on the unfixed builder.
"""

from __future__ import annotations

import unittest
from unittest import mock

import builders.common.attention_spec_builder as asb
import kernels.common.attention_unified as au
from kernels import UnifiedAttentionProblem

# Plain bf16 D128 prefill: no sinks / sliding-window / softcap / alibi / bias, so
# it routes onto the ordinary 2D tiled path on every arch under test and is not
# diverted by any arch-specific fast route.
_PROBLEM = UnifiedAttentionProblem(
    total_q=1024,
    num_seqs=2,
    num_query_heads=16,
    num_kv_heads=2,
    head_size=128,
    block_size=16,
    max_seqlen_q=512,
    max_seqlen_k=4096,
    dtype="bf16",
)

# Decode-shaped variant of the same plain bf16 problem, for the 3D entry point.
_PROBLEM_3D = UnifiedAttentionProblem(
    total_q=2,
    num_seqs=2,
    num_query_heads=16,
    num_kv_heads=2,
    head_size=128,
    block_size=16,
    max_seqlen_q=1,
    max_seqlen_k=4096,
    dtype="bf16",
)

_BOUND_IMPORT_TRAP = (
    "attention_spec_builder must reach _resolve_attention_arch through its "
    "attention_unified module handle -- a bound import freezes the reference at "
    "import time and silently defeats mock.patch.object on the module"
)


class TestArchResolverBinding(unittest.TestCase):
    def test_builder_does_not_bind_the_resolver(self):
        """Static form: the name must not be an attribute of the builder."""
        # assertFalse(hasattr(...)), not assertNotIn(..., vars(asb)) -- the
        # latter prints the builder's entire module dict on failure.
        self.assertFalse(hasattr(asb, "_resolve_attention_arch"), _BOUND_IMPORT_TRAP)

    def test_patching_the_module_steers_the_builder(self):
        """Behavioural form: the pin must reach the spec class the builder picks."""
        entries = (
            ("2d", asb._tiled_spec_from_problem, _PROBLEM),
            ("3d", asb._tiled_3d_spec_from_problem, _PROBLEM_3D),
        )
        for arch in ("gfx942", "gfx950"):
            for label, build, problem in entries:
                with self.subTest(arch=arch, entry=label):
                    with mock.patch.object(
                        au, "_resolve_attention_arch", return_value=arch
                    ):
                        spec = build(problem)
                    self.assertTrue(
                        type(spec).__module__.startswith(f"kernels.{arch}."),
                        f"{_BOUND_IMPORT_TRAP}; {label} pinned {arch} but got "
                        f"{type(spec).__module__}",
                    )


if __name__ == "__main__":
    unittest.main()
