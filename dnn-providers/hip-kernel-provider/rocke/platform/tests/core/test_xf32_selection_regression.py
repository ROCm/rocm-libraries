# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""No pre-existing MMA selection moved when XF32 arrived (XF32 plan, step 7).

Adding rows to a catalog is only safe if the rows are unreachable from every
query that predates them, and if the preference rule introduced alongside them
is inert on the atoms that were already there. Both are load-bearing: every
GEMM, attention and convolution instance in the tree picks its atom out of this
catalog, so a perturbed answer is a silently different kernel.

The check here is a differential one. A *stripped* catalog is built per arch by
dropping the emulated rows, reconstructing the catalog exactly as it stood
before step 4 -- then every selection entry point is asked the same question of
both catalogs, for every query the shipped atoms can pose, and the answers must
be equal. Unlike an expected-value table, this cannot rot: adding a native atom
updates both sides at once, while a leak updates only one.

Two narrower angles that a differential test alone cannot see are pinned too:
that no non-xf32 query returns an emulated atom on any arch, and that the
comparator never fires on an all-native catalog.
"""

from __future__ import annotations

import unittest
from typing import Any

from rocke.core.arch import ArchTarget, known_arches
from rocke.core.arch.target import MmaCatalog, MmaOp

# Every entry point that resolves a query into atoms. They all funnel through
# enumerate(), which is the point -- but "they all funnel through enumerate"
# is an implementation fact, and this test is meant to survive its changing.
_QUERY_METHODS = ("enumerate", "has_shape", "select_largest_k", "op_for_shape")


def _answer(catalog: MmaCatalog, method: str, **kwargs: Any) -> Any:
    """The catalog's answer, with a raised ambiguity counting as an answer.

    ``select_largest_k`` / ``op_for_shape`` raise on an ambiguous query rather
    than picking arbitrarily. A new row turning a formerly unique query
    ambiguous is exactly the regression being hunted, so the exception is
    compared like any other result instead of being swallowed.
    """
    try:
        got = getattr(catalog, method)(**kwargs)
    except ValueError as exc:
        return ("raised", type(exc).__name__, str(exc))
    if isinstance(got, list):
        return [op.op_id for op in got]
    if isinstance(got, MmaOp):
        return got.op_id
    return got


def _queries(op: MmaOp) -> dict[str, Any]:
    """The full operand contract of one shipped atom, as query kwargs."""
    return {
        "family": op.family,
        "a_dtype": op.a_dtype,
        "b_dtype": op.b_dtype,
        "c_dtype": op.c_dtype,
        "scales": (op.a_scale_dtype, op.b_scale_dtype, op.scale_block_k),
    }


def _kwargs_for(method: str, op: MmaOp) -> dict[str, Any]:
    kwargs = _queries(op)
    if method == "enumerate":
        return kwargs
    kwargs["m"], kwargs["n"] = op.m, op.n
    if method in ("has_shape", "op_for_shape"):
        kwargs["k"] = op.k
    return kwargs


def _pair(gfx: str) -> tuple[MmaCatalog, MmaCatalog, tuple[MmaOp, ...]]:
    """The live catalog, the pre-XF32 catalog, and the rows they share."""
    live = ArchTarget.from_gfx(gfx).mma
    stripped = MmaCatalog([op for op in live.ops if op.emulation is None])
    return live, stripped, stripped.ops


class TestSelectionIsUnchanged(unittest.TestCase):
    """Differential: live catalog vs the same catalog minus the emulated rows."""

    def test_every_entry_point_answers_identically(self):
        for gfx in known_arches():
            live, stripped, pre_existing = _pair(gfx)
            for op in pre_existing:
                for method in _QUERY_METHODS:
                    kwargs = _kwargs_for(method, op)
                    with self.subTest(gfx=gfx, op_id=op.op_id, method=method):
                        self.assertEqual(
                            _answer(live, method, **kwargs),
                            _answer(stripped, method, **kwargs),
                        )

    def test_unscaled_queries_answer_identically(self):
        """The spelling instances actually use: no explicit scale contract.

        A scale-free query is the widest net a caller can cast, so it is the
        one most likely to catch a stray row.
        """
        for gfx in known_arches():
            live, stripped, pre_existing = _pair(gfx)
            for op in pre_existing:
                kwargs = {
                    "family": op.family,
                    "a_dtype": op.a_dtype,
                    "b_dtype": op.b_dtype,
                    "c_dtype": op.c_dtype,
                }
                with self.subTest(gfx=gfx, op_id=op.op_id):
                    self.assertEqual(
                        _answer(live, "enumerate", **kwargs),
                        _answer(stripped, "enumerate", **kwargs),
                    )

    def test_supports_dtype_combo_is_unchanged(self):
        """The arch-level predicate gating whole instances."""
        for gfx in known_arches():
            target = ArchTarget.from_gfx(gfx)
            _, stripped, pre_existing = _pair(gfx)
            for op in pre_existing:
                with self.subTest(gfx=gfx, op_id=op.op_id):
                    self.assertEqual(
                        target.supports_dtype_combo(
                            op.a_dtype, op.b_dtype, op.c_dtype, family=op.family
                        ),
                        bool(
                            stripped.enumerate(
                                family=op.family,
                                a_dtype=op.a_dtype,
                                b_dtype=op.b_dtype,
                                c_dtype=op.c_dtype,
                            )
                        ),
                    )

    def test_by_op_id_still_resolves_every_pre_existing_atom(self):
        """Lookup by handle is order-sensitive; the new rows must not shadow."""
        for gfx in known_arches():
            live, _, pre_existing = _pair(gfx)
            for op in pre_existing:
                with self.subTest(gfx=gfx, op_id=op.op_id):
                    self.assertIs(live.by_op_id(op.op_id), op)


class TestEmulatedRowsAreUnreachableFromOldQueries(unittest.TestCase):
    def test_no_pre_existing_query_can_return_an_emulated_atom(self):
        """Stated directly, not just as a consequence of the differential."""
        for gfx in known_arches():
            catalog = ArchTarget.from_gfx(gfx).mma
            for op in catalog.ops:
                if op.emulation is not None:
                    continue
                got = catalog.enumerate(
                    family=op.family,
                    a_dtype=op.a_dtype,
                    b_dtype=op.b_dtype,
                    c_dtype=op.c_dtype,
                )
                with self.subTest(gfx=gfx, op_id=op.op_id):
                    self.assertTrue(
                        all(cand.is_native for cand in got),
                        "an emulated atom reached a query that predates it",
                    )

    def test_fp32_never_reaches_an_xf32_atom(self):
        """The one confusion the emulation invites: fp32 in, xf32 selected.

        XF32 is opt-in. A caller asking for fp32 gets the native fp32 MFMA or
        nothing -- never a silently lower-precision emulation.
        """
        for gfx in known_arches():
            got = ArchTarget.from_gfx(gfx).mma.enumerate(
                family="mma", a_dtype="fp32", b_dtype="fp32", c_dtype="fp32"
            )
            with self.subTest(gfx=gfx):
                self.assertTrue(all(op.is_native for op in got))
                self.assertTrue(all(op.a_dtype == "fp32" for op in got))

    def test_the_comparator_is_inert_on_an_all_native_catalog(self):
        """Why no arch without emulated rows could have moved at all."""
        for gfx in known_arches():
            _, stripped, _ = _pair(gfx)
            with self.subTest(gfx=gfx):
                self.assertTrue(all(op.is_native for op in stripped.ops))
                for op in stripped.ops:
                    self.assertEqual(
                        stripped.enumerate(**_queries(op)),
                        [
                            cand
                            for cand in stripped.ops
                            if (cand.family, cand.a_dtype, cand.b_dtype, cand.c_dtype)
                            == (op.family, op.a_dtype, op.b_dtype, op.c_dtype)
                            and (
                                cand.a_scale_dtype,
                                cand.b_scale_dtype,
                                cand.scale_block_k,
                            )
                            == (op.a_scale_dtype, op.b_scale_dtype, op.scale_block_k)
                        ],
                        "the preference rule dropped or reordered a native atom",
                    )


if __name__ == "__main__":
    unittest.main()
