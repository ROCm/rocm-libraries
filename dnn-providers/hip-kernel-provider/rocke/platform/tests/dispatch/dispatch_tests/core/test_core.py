# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Operator-agnostic dispatcher contracts in :mod:`rocke.dispatch.core`.

Covers the two identity keys and the registry lookup primitives. Family-specific
behavior is tested beside its family (see ``dispatch_tests/gemm``).
"""

from __future__ import annotations

import json
import unittest
from dataclasses import dataclass, replace

from rocke.core.arch import known_arches
from rocke.dispatch.core import (
    Capability,
    CandidateRegistry,
    DimRelation,
    KernelCandidate,
    KernelId,
    OperatorRequest,
    ShapeRange,
)


def _kernel_id(**overrides) -> KernelId:
    base = {
        "op": "gemm",
        "family": "gemm_fp16_rcr",
        "candidate": "cdna_cshuffle_default",
        "algorithm": "cshuffle",
        "spec_id": "cdna_cshuffle_default",
        "arch": "gfx950",
        "abi_version": "rocke-gemm-fp16-rcr/v1",
        "request_hash": "1111111111111111",
        "spec_hash": "2222222222222222",
    }
    base.update(overrides)
    return KernelId(**base)


# Registration requires a capability, so tests about registry mechanics rather
# than about coverage get one that constrains nothing but the arch list.
_ANY_ARCH = Capability(arches=known_arches())


def _candidate(
    name: str,
    *,
    family: str = "dummy",
    priority: int = 0,
    abi="dummy/v1",
    capability=_ANY_ARCH,
    _supports=lambda _req: (True, "ok"),
):
    return KernelCandidate(
        name=name,
        family=family,
        algorithm="dummy_algorithm",
        spec_id=f"{name}_spec",
        abi_version=abi,
        priority=priority,
        _supports=_supports,
        select_spec=lambda _req: object(),
        signature=lambda _spec: (),
        grid=lambda _spec, _req: (1, 1, 1),
        block=lambda _spec: (1, 1, 1),
        sweep_space=lambda _req: (),
        capability=capability,
    )


@dataclass(frozen=True)
class _Request(OperatorRequest):
    """A minimal family request: arch, dtype, two dims, and a feature set."""

    arch: str = "gfx950"
    dtype: str = "fp16"
    layout: str = "RCR"
    rows: int = 128
    cols: int = 256
    causal: bool = False

    def normalized(self) -> dict:
        return {"arch": self.arch, "dtype": self.dtype, "layout": self.layout}

    def dims(self):
        return {"rows": self.rows, "cols": self.cols, "total": self.rows * self.cols}

    def features(self) -> frozenset[str]:
        return frozenset({"causal"}) if self.causal else frozenset()


_VOCAB = ("rows", "cols", "total")


class TestKernelIdKeys(unittest.TestCase):
    def test_compile_key_is_problem_independent(self):
        """Two problems selecting the same spec must share one compile."""
        a = _kernel_id(request_hash="aaaaaaaaaaaaaaaa")
        b = _kernel_id(request_hash="bbbbbbbbbbbbbbbb")
        self.assertEqual(a.compile_key, b.compile_key)
        self.assertNotEqual(a.selection_key, b.selection_key)

    def test_compile_key_tracks_arch_abi_and_spec(self):
        base = _kernel_id()
        for field, value in (
            ("arch", "gfx942"),
            ("abi_version", "rocke-gemm-fp16-rcr/v2"),
            ("spec_hash", "3333333333333333"),
        ):
            with self.subTest(field=field):
                self.assertNotEqual(
                    base.compile_key, replace(base, **{field: value}).compile_key
                )

    def test_selection_key_distinguishes_every_field(self):
        base = _kernel_id()
        for field, value in (
            ("op", "batched_gemm"),
            ("family", "gemm_bf16_rcr"),
            ("candidate", "cdna_mem_64x128"),
            ("algorithm", "mem"),
            ("spec_id", "cdna_mem_64x128"),
            ("arch", "gfx942"),
            ("abi_version", "rocke-gemm-fp16-rcr/v2"),
            ("request_hash", "4444444444444444"),
            ("spec_hash", "5555555555555555"),
        ):
            with self.subTest(field=field):
                self.assertNotEqual(
                    base.selection_key, replace(base, **{field: value}).selection_key
                )

    def test_cache_key_alias_value_is_unchanged_by_the_split(self):
        """Pinned literal: benchmark records and manifests already carry it."""
        self.assertEqual(
            _kernel_id().cache_key,
            "gemm:gemm_fp16_rcr:cdna_cshuffle_default:gfx950:cshuffle:"
            "cdna_cshuffle_default:rocke-gemm-fp16-rcr/v1:"
            "1111111111111111:2222222222222222",
        )

    def test_cache_key_is_the_selection_key(self):
        kid = _kernel_id()
        self.assertEqual(kid.cache_key, kid.selection_key)


class TestRegistryLookup(unittest.TestCase):
    def _registry(self) -> CandidateRegistry:
        registry = CandidateRegistry("dummy")
        registry.register(_candidate("fast", priority=10))
        registry.register(_candidate("generic", priority=50))
        return registry

    def test_get_returns_the_registered_candidate(self):
        self.assertEqual(self._registry().get("fast").name, "fast")

    def test_get_unknown_name_lists_what_is_registered(self):
        with self.assertRaises(ValueError) as ctx:
            self._registry().get("typo")
        message = str(ctx.exception)
        self.assertIn("unknown candidate 'typo'", message)
        self.assertIn("fast", message)
        self.assertIn("generic", message)

    def test_resolve_returns_the_candidate_named_by_the_id(self):
        registry = self._registry()
        kid = _kernel_id(candidate="fast", abi_version="dummy/v1")
        self.assertIs(registry.resolve(kid), registry.get("fast"))

    def test_resolve_rejects_a_stale_abi_version(self):
        """A replayed id from an older build must fail loudly, not rebind."""
        kid = _kernel_id(candidate="fast", abi_version="dummy/v0")
        with self.assertRaisesRegex(ValueError, "ABI mismatch for 'fast'"):
            self._registry().resolve(kid)

    def test_resolve_rejects_an_unknown_candidate(self):
        with self.assertRaisesRegex(ValueError, "unknown candidate"):
            self._registry().resolve(_kernel_id(candidate="retired"))


class TestRegistryCoverage(unittest.TestCase):
    def test_coverage_is_json_serializable(self):
        registry = CandidateRegistry("dummy")
        registry.register(_candidate("fast", priority=10))
        json.dumps(registry.coverage())

    def test_coverage_reports_every_candidate_in_priority_order(self):
        registry = CandidateRegistry("dummy")
        registry.register(_candidate("generic", priority=50))
        registry.register(_candidate("fast", priority=10))
        manifest = registry.coverage()

        self.assertEqual(manifest["family"], "dummy")
        self.assertEqual(
            [c["name"] for c in manifest["candidates"]], ["fast", "generic"]
        )
        self.assertEqual(
            manifest["candidates"][0],
            {
                "name": "fast",
                "algorithm": "dummy_algorithm",
                "spec_id": "fast_spec",
                "abi_version": "dummy/v1",
                "priority": 10,
                # Both False here because this fixture declares neither, and
                # "dummy" requires neither. Reported either way so that "can I
                # compile and launch this?" stays a lookup rather than a call
                # that might raise.
                "buildable": False,
                "bindable": False,
                # Never None: registration requires a declared capability, so
                # the manifest cannot have a hole in it.
                "capability": _ANY_ARCH.as_dict(),
            },
        )

    def test_coverage_serializes_a_declared_capability(self):
        registry = CandidateRegistry("dummy", dim_vocabulary=_VOCAB)
        registry.register(
            _candidate(
                "fast",
                capability=Capability(
                    arches=("gfx942", "gfx950"),
                    dtypes=("fp16",),
                    shapes=(ShapeRange(frozenset({"rows", "cols"}), multiple_of=64),),
                    relations=(DimRelation("cols", ">=", "rows"),),
                    supports_features=frozenset({"causal"}),
                ),
            )
        )
        capability = registry.coverage()["candidates"][0]["capability"]

        json.dumps(capability)
        self.assertEqual(capability["arches"], ["gfx942", "gfx950"])
        self.assertEqual(
            capability["shapes"], [{"dims": ["cols", "rows"], "multiple_of": 64}]
        )
        self.assertEqual(
            capability["relations"], [{"lhs": "cols", "op": ">=", "rhs": "rows"}]
        )
        self.assertEqual(capability["supports_features"], ["causal"])

    def test_coverage_of_an_empty_registry_is_empty(self):
        self.assertEqual(
            CandidateRegistry("dummy").coverage(),
            {
                "family": "dummy",
                "requires_build": False,
                "requires_binding": False,
                "candidates": [],
            },
        )

    def test_coverage_reports_a_familys_build_and_binding_stance(self):
        coverage = CandidateRegistry(
            "dummy", require_build=True, require_binding=True
        ).coverage()
        self.assertTrue(coverage["requires_build"])
        self.assertTrue(coverage["requires_binding"])


class TestShapeRange(unittest.TestCase):
    def test_a_string_dims_is_the_singleton_case(self):
        self.assertEqual(ShapeRange("rows", min=32).names(), ("rows",))

    def test_a_scalar_bound_broadcasts_over_a_set(self):
        rng = ShapeRange(frozenset({"rows", "cols"}), multiple_of=64)
        self.assertTrue(rng.check({"rows": 128, "cols": 256})[0])
        self.assertFalse(rng.check({"rows": 128, "cols": 100})[0])

    def test_names_are_sorted_so_messages_are_reproducible(self):
        """Set iteration order varies per process; golden tests cannot."""
        for _ in range(64):
            rng = ShapeRange(frozenset({"Wi", "Hi", "C"}), min=1)
            self.assertEqual(rng.names(), ("C", "Hi", "Wi"))

    def test_a_missing_dim_names_what_the_family_did_provide(self):
        ok, why = ShapeRange("hdim_k", min=1).check({"rows": 8})
        self.assertFalse(ok)
        self.assertIn("dim 'hdim_k' not provided", why)
        self.assertIn("['rows']", why)

    def test_each_bound_rejects_and_explains(self):
        cases = (
            (ShapeRange("rows", allowed=(64, 128)), {"rows": 96}, "not in (64, 128)"),
            (ShapeRange("rows", min=128), {"rows": 64}, "< min 128"),
            (ShapeRange("rows", max=128), {"rows": 256}, "> max 128"),
            (ShapeRange("rows", multiple_of=64), {"rows": 100}, "not a multiple of 64"),
        )
        for rng, dims, expected in cases:
            with self.subTest(expected=expected):
                ok, why = rng.check(dims)
                self.assertFalse(ok)
                self.assertIn(expected, why)

    def test_an_unconstrained_range_accepts_any_present_dim(self):
        self.assertTrue(ShapeRange("rows").check({"rows": 7})[0])


class TestDimRelation(unittest.TestCase):
    def test_every_operator_evaluates(self):
        dims = {"a": 8, "b": 4}
        cases = (
            ("==", "a", False),
            ("!=", "a", True),
            ("<", "a", False),
            ("<=", "a", False),
            (">", "a", True),
            (">=", "a", True),
            ("multiple_of", "a", True),
        )
        for op, lhs, expected in cases:
            with self.subTest(op=op):
                self.assertEqual(DimRelation(lhs, op, "b").check(dims)[0], expected)

    def test_rhs_may_be_a_literal(self):
        self.assertTrue(DimRelation("a", "multiple_of", 4).check({"a": 8})[0])

    def test_multiple_of_zero_is_a_rejection_not_a_crash(self):
        ok, _ = DimRelation("a", "multiple_of", "b").check({"a": 8, "b": 0})
        self.assertFalse(ok)

    def test_an_unknown_operator_fails_at_construction(self):
        """A typo must not sit dormant until some request reaches it."""
        with self.assertRaisesRegex(ValueError, "unknown DimRelation op"):
            DimRelation("a", "=<", "b")

    def test_a_missing_dim_on_either_side_is_reported(self):
        for lhs, rhs in (("a", "missing"), ("missing", "a")):
            with self.subTest(lhs=lhs, rhs=rhs):
                ok, why = DimRelation(lhs, "==", rhs).check({"a": 1})
                self.assertFalse(ok)
                self.assertIn("not provided", why)

    def test_violation_message_names_both_sides(self):
        ok, why = DimRelation("hdim_q", "==", "hdim_v").check(
            {"hdim_q": 128, "hdim_v": 64}
        )
        self.assertFalse(ok)
        self.assertIn("hdim_q=128", why)
        self.assertIn("hdim_v=64", why)


class TestCapability(unittest.TestCase):
    def test_arch_fails_closed_when_none_are_declared(self):
        """A capability that declares no arch must match nothing, not everything."""
        ok, why = Capability().check(_Request())
        self.assertFalse(ok)
        self.assertIn("arch 'gfx950' not in ()", why)

    def test_an_undeclared_arch_is_rejected(self):
        cap = Capability(arches=("gfx942",))
        self.assertFalse(cap.check(_Request(arch="gfx950"))[0])
        self.assertTrue(cap.check(_Request(arch="gfx942"))[0])

    def test_empty_tuples_other_than_arches_mean_unconstrained(self):
        self.assertTrue(Capability(arches=("gfx950",)).check(_Request())[0])

    def test_dtype_and_layout_are_gated_case_insensitively(self):
        cap = Capability(arches=("gfx950",), dtypes=("fp16",), layouts=("RCR",))
        self.assertTrue(cap.check(_Request(dtype="FP16", layout="rcr"))[0])
        self.assertFalse(cap.check(_Request(dtype="bf16"))[0])
        self.assertFalse(cap.check(_Request(layout="RRR"))[0])

    def test_shape_and_relation_constraints_are_applied(self):
        cap = Capability(
            arches=("gfx950",),
            shapes=(ShapeRange("rows", max=256),),
            relations=(DimRelation("cols", ">=", "rows"),),
        )
        self.assertTrue(cap.check(_Request(rows=128, cols=256))[0])
        self.assertFalse(cap.check(_Request(rows=512, cols=1024))[0])
        self.assertFalse(cap.check(_Request(rows=256, cols=128))[0])

    def test_a_derived_dim_is_constrainable(self):
        """``total`` is computed by dims(), not stored on the request."""
        cap = Capability(arches=("gfx950",), shapes=(ShapeRange("total", max=1024),))
        self.assertFalse(cap.check(_Request(rows=128, cols=256))[0])
        self.assertTrue(cap.check(_Request(rows=8, cols=8))[0])

    def test_an_undeclared_feature_is_rejected(self):
        """Forgetting to declare a feature must reject, not silently ignore it."""
        cap = Capability(arches=("gfx950",))
        ok, why = cap.check(_Request(causal=True))
        self.assertFalse(ok)
        self.assertIn("cannot serve features ['causal']", why)

    def test_a_declared_feature_is_accepted(self):
        cap = Capability(arches=("gfx950",), supports_features=frozenset({"causal"}))
        self.assertTrue(cap.check(_Request(causal=True))[0])

    def test_a_required_feature_must_be_present(self):
        cap = Capability(
            arches=("gfx950",),
            supports_features=frozenset({"causal"}),
            requires_features=frozenset({"causal"}),
        )
        self.assertTrue(cap.check(_Request(causal=True))[0])
        ok, why = cap.check(_Request(causal=False))
        self.assertFalse(ok)
        self.assertIn("requires features ['causal']", why)

    def test_dim_names_collects_from_shapes_and_relations(self):
        cap = Capability(
            arches=("gfx950",),
            shapes=(ShapeRange(frozenset({"rows", "cols"})),),
            relations=(DimRelation("total", ">=", "rows"), DimRelation("cols", ">", 1)),
        )
        self.assertEqual(cap.dim_names(), frozenset({"rows", "cols", "total"}))


class TestCapabilityRegistration(unittest.TestCase):
    def test_an_undeclared_capability_is_rejected(self):
        """Coverage is mandatory: an undeclared candidate is invisible to
        for_arch and coverage, which would make them lie by omission."""
        registry = CandidateRegistry("dummy")
        with self.assertRaisesRegex(ValueError, "declares no capability"):
            registry.register(_candidate("legacy", capability=None))

    def test_the_rejection_names_the_offending_candidate(self):
        registry = CandidateRegistry("dummy")
        with self.assertRaises(ValueError) as ctx:
            registry.register(_candidate("forgot_to_declare", capability=None))
        self.assertIn("'forgot_to_declare'", str(ctx.exception))

    def test_a_declared_capability_must_name_an_arch(self):
        registry = CandidateRegistry("dummy")
        with self.assertRaisesRegex(ValueError, "declares no arch coverage"):
            registry.register(_candidate("ungated", capability=Capability()))

    def test_an_unknown_arch_is_rejected_at_registration(self):
        registry = CandidateRegistry("dummy")
        with self.assertRaisesRegex(ValueError, r"unknown arches \['gfx9999'\]"):
            registry.register(
                _candidate("typo", capability=Capability(arches=("gfx9999",)))
            )

    def test_a_misspelled_dim_is_rejected_at_registration(self):
        """The hazard named dimensions introduce: a typo that never fires."""
        registry = CandidateRegistry("dummy", dim_vocabulary=_VOCAB)
        capability = Capability(arches=("gfx950",), shapes=(ShapeRange("rowz", min=1),))
        with self.assertRaisesRegex(ValueError, r"unknown dims \['rowz'\]"):
            registry.register(_candidate("typo", capability=capability))

    def test_dims_are_unchecked_when_the_family_declares_no_vocabulary(self):
        registry = CandidateRegistry("dummy")
        registry.register(
            _candidate(
                "loose",
                capability=Capability(
                    arches=("gfx950",), shapes=(ShapeRange("anything", min=1),)
                ),
            )
        )
        self.assertEqual(len(registry.candidates()), 1)


class TestCapabilityPrefilter(unittest.TestCase):
    def _registry(self) -> CandidateRegistry:
        registry = CandidateRegistry("dummy", dim_vocabulary=_VOCAB)
        registry.register(
            _candidate(
                "gfx942_only",
                priority=10,
                capability=Capability(arches=("gfx942",)),
            )
        )
        registry.register(
            _candidate(
                "gfx950_only",
                priority=20,
                capability=Capability(arches=("gfx950",)),
            )
        )
        return registry

    def test_capability_filters_before_the_predicate_runs(self):
        supported = self._registry().supported(_Request(arch="gfx942"))
        self.assertEqual([c.name for c in supported], ["gfx942_only"])

    def test_the_predicate_still_narrows_what_capability_admits(self):
        registry = CandidateRegistry("dummy", dim_vocabulary=_VOCAB)
        registry.register(
            _candidate(
                "picky",
                capability=Capability(arches=("gfx950",)),
                _supports=lambda req: (req.rows >= 256, f"rows={req.rows} below 256"),
            )
        )
        self.assertEqual(registry.supported(_Request(rows=512)), registry.candidates())
        self.assertEqual(registry.supported(_Request(rows=64)), ())

    def test_the_predicate_is_not_consulted_once_capability_rejects(self):
        calls = []

        def _supports(req):
            calls.append(req)
            return True, "ok"

        registry = CandidateRegistry("dummy")
        registry.register(
            _candidate(
                "gated",
                capability=Capability(arches=("gfx942",)),
                _supports=_supports,
            )
        )
        registry.supported(_Request(arch="gfx950"))
        self.assertEqual(calls, [])

    def test_a_capability_rejection_is_labelled_in_the_aggregated_error(self):
        with self.assertRaises(ValueError) as ctx:
            self._registry().select(_Request(arch="gfx1250"))
        message = str(ctx.exception)
        self.assertIn("gfx942_only: capability: arch 'gfx1250' not in", message)
        self.assertIn("gfx950_only: capability:", message)

    def test_the_prefilter_runs_before_the_predicate(self):
        """A predicate that would crash must never be reached once capability
        has already ruled the request out."""

        def explode(_req):
            raise AssertionError("predicate ran despite a capability rejection")

        candidate = _candidate(
            "gfx942_only",
            capability=Capability(arches=("gfx942",)),
            _supports=explode,
        )
        ok, why = candidate.admits(_Request(arch="gfx950"))
        self.assertFalse(ok)
        self.assertIn("capability: arch 'gfx950' not in", why)


class TestForArch(unittest.TestCase):
    def _registry(self) -> CandidateRegistry:
        registry = CandidateRegistry("dummy")
        registry.register(
            _candidate(
                "generic",
                priority=50,
                capability=Capability(arches=("gfx942", "gfx950")),
            )
        )
        registry.register(
            _candidate(
                "specialized",
                priority=10,
                capability=Capability(arches=("gfx950",)),
            )
        )
        return registry

    def test_for_arch_returns_declaring_candidates_in_priority_order(self):
        names = [c.name for c in self._registry().for_arch("gfx950")]
        self.assertEqual(names, ["specialized", "generic"])

    def test_for_arch_excludes_candidates_that_did_not_declare_the_arch(self):
        self.assertEqual(
            [c.name for c in self._registry().for_arch("gfx942")], ["generic"]
        )

    def test_an_unserved_arch_returns_nothing(self):
        self.assertEqual(self._registry().for_arch("gfx1250"), ())


if __name__ == "__main__":
    unittest.main()
