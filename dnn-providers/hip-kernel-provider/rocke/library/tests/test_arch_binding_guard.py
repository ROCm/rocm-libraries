# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Guard: the spec builder selects on its ``arch`` ARGUMENT, not on the host.

History. ``builders.common.attention_spec_builder`` used to bind the resolver
with ``from kernels.common.attention_unified import _resolve_attention_arch``. A
bound import freezes the reference at import time, so ``mock.patch.object(au,
"_resolve_attention_arch", ...)`` -- which rebinds the attribute on the *module*
-- never reached the builder. The builder then resolved the real device arch,
and the gfx950-only spec overrides raised ``TypeError`` against a gfx942 spec
class. The binding dates to #9057 and turned load-bearing in #9233.

That whole failure mode is now unreachable by construction: the spec builders
take ``arch`` as a required parameter and never consult the resolver, so there
is no reference left to freeze. This guard therefore asserts the *stronger*
property that replaced it -- the explicit argument decides, and the host cannot
influence the answer:

1. statically, the builder still must not bind the resolver's name; and
2. behaviourally, ``build(problem, arch)`` picks the spec class for ``arch``
   even while ``au._resolve_attention_arch`` is patched to report a DIFFERENT
   arch. Under the old implicit builder the patched resolver would win and the
   assertion would fail, so this is a strict tightening of the old test rather
   than a weaker restatement of it.

Both builder entry points are covered: ``_tiled_spec_from_problem`` and
``_tiled_3d_spec_from_problem`` each used to hold their own
``_kau._resolve_attention_arch()`` call site, and both were served by the same
bound import before the fix.

gfx1250 is deliberately NOT pinned: it needs its own problem shape (2D rejects
``block_size=16``, 3D rejects ``head_size=128``), so it would couple this guard
to the evolving gfx1250 support matrix while adding no discriminating power.
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
    "attention_spec_builder must not bind _resolve_attention_arch -- the spec "
    "builders take arch explicitly, so any reference to the resolver is a "
    "re-introduced host dependency"
)

_HOST_INDEPENDENCE_TRAP = (
    "attention_spec_builder must select on its arch ARGUMENT -- reading the "
    "host resolver instead makes the selected spec depend on the box the "
    "dispatcher happens to run on, which AOT dispatch forbids"
)


class TestArchResolverBinding(unittest.TestCase):
    def test_builder_does_not_bind_the_resolver(self):
        """Static form: the name must not be an attribute of the builder."""
        # assertFalse(hasattr(...)), not assertNotIn(..., vars(asb)) -- the
        # latter prints the builder's entire module dict on failure.
        self.assertFalse(hasattr(asb, "_resolve_attention_arch"), _BOUND_IMPORT_TRAP)

    def test_the_arch_argument_beats_the_host(self):
        """Behavioural form: the argument decides, even against a hostile host.

        Each case patches the module resolver to the *opposite* arch. A builder
        that still consulted the host would return that arch's spec class and
        fail here, so the patch is the discriminator -- not scenery.
        """
        entries = (
            ("2d", asb._tiled_spec_from_problem, _PROBLEM),
            ("3d", asb._tiled_3d_spec_from_problem, _PROBLEM_3D),
        )
        for arch, host_arch in (("gfx942", "gfx950"), ("gfx950", "gfx942")):
            for label, build, problem in entries:
                with self.subTest(arch=arch, host=host_arch, entry=label):
                    with mock.patch.object(
                        au, "_resolve_attention_arch", return_value=host_arch
                    ):
                        spec = build(problem, arch)
                    self.assertTrue(
                        type(spec).__module__.startswith(f"kernels.{arch}."),
                        f"{_HOST_INDEPENDENCE_TRAP}; {label} asked for {arch} "
                        f"on a host reporting {host_arch} but got "
                        f"{type(spec).__module__}",
                    )


if __name__ == "__main__":
    unittest.main()
