# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Tests for recipe admission guards (src/guard.py + the C evaluator in
# recipe_vm.cpp).
#
# A guard's whole value is that "admitted" can be trusted, so the tests are
# weighted accordingly. Soundness -- never accept what the gate rejects -- is
# checked directly, on the fallback paths as well as the happy one, and against
# a gate deliberately built so a naive derivation would get it wrong. Coverage
# (not rejecting more than necessary) is checked too, but a failure there is a
# weaker claim and the tests say so.
#
# The other half is that the Python and C evaluators agree. They are separate
# implementations reading the same guard, and a disagreement means the check a
# caller ran is not the check the VM will run -- so the parity test drives both
# over the same points, including the negative and zero values that are exactly
# where their `mod` semantics differ.
#
#   python3 -m unittest rocke.portable_ir.tests.test_guard

import unittest

from rocke.portable_ir.src.guard import (
    GuardDerivationError,
    attach_guard,
    axis_rules,
    derive_guard,
    guard_accepts,
    verify_guard,
)
from rocke.portable_ir.utils.recipe_expand import (
    GUARD_SCHEMA,
    GuardRejected,
    check_guard,
    expand_recipe,
)

# A gate with one per-axis constraint and one coupling, which is the shape of
# the real ones: head_size runs 16..256 by 16, block_size is one of three
# values, and the block has to divide the head. The coupling is the part a
# per-axis derivation cannot see, so it is what the tests lean on.
CANDS = {
    "head_size": [8 * i for i in range(1, 40)],  # deliberately wider than legal
    "block_size": [16, 32, 64, 96],
}


def gate(p):
    h, b = p["head_size"], p["block_size"]
    return 16 <= h <= 256 and h % 16 == 0 and b in (16, 32, 64) and h % b == 0


class TestAxisRules(unittest.TestCase):
    def _accepts(self, rules, axis, v):
        g = {"schema": GUARD_SCHEMA, "free": [axis], "rules": rules}
        ok, _ = check_guard(g, {axis: v}, {})
        return ok

    def test_stride_form_is_exact(self):
        """A run compresses to two rules that accept the run and nothing else."""
        legal = list(range(16, 257, 16))
        rules = axis_rules("head_size", legal)
        self.assertEqual(len(rules), 2)
        for v in range(0, 300):
            self.assertEqual(
                self._accepts(rules, "head_size", v),
                v in legal,
                f"head_size={v}",
            )

    def test_enumeration_when_not_a_run(self):
        legal = [16, 32, 64, 128]  # doubling, not an arithmetic progression
        rules = axis_rules("block_size", legal)
        for v in range(0, 200):
            self.assertEqual(self._accepts(rules, "block_size", v), v in legal)

    def test_bounds_precede_divisibility(self):
        """Bounds first, so a negative never reaches the `mod`.

        Python's `mod` floors and C's truncates, so they differ on a negative
        left operand. Today's rules ask only whether a remainder is zero, which
        is sign-independent, but a rule that did anything else would depend on
        this ordering -- so it is pinned rather than left to chance."""
        rules = axis_rules("head_size", list(range(16, 257, 16)))
        self.assertIn("must be in", rules[0]["reason"])
        self.assertIn("multiple of", rules[1]["reason"])
        for v in (-256, -16, -1, 0):
            self.assertFalse(self._accepts(rules, "head_size", v), f"head_size={v}")

    def test_string_axis(self):
        rules = axis_rules("dtype", ["bf16", "f16"])
        for v, want in (("bf16", True), ("f16", True), ("f32", False)):
            g = {"schema": GUARD_SCHEMA, "free": ["dtype"], "rules": rules}
            ok, _ = check_guard(g, {}, {"dtype": v})
            self.assertEqual(ok, want)


class TestDerivation(unittest.TestCase):
    def test_recovers_the_gate_exactly(self):
        """On a gate whose structure the candidate rules can express, derivation
        should reproduce it -- not merely stay sound. Sound-but-useless is the
        easy failure mode here, so this pins agreement in both directions."""
        g = derive_guard(gate, CANDS, gate_name="test", arch="gfx950")
        self.assertEqual(g["derivation"]["method"], "coupled")
        report = verify_guard(g, gate, CANDS, samples=2000, seed=99)
        self.assertEqual(report.unsound, [])
        self.assertEqual(report.strict, [], "derivation lost coverage it could keep")

    def test_finds_the_coupling_rule(self):
        g = derive_guard(gate, CANDS, gate_name="test")
        self.assertTrue(
            any("divide" in r["reason"] for r in g["rules"]),
            f"no divisibility rule in {[r['reason'] for r in g['rules']]}",
        )

    def test_blocklists_holes_no_rule_can_explain(self):
        """A gate with an arbitrary illegal set has no rule in the candidate
        library that fits it. Derivation must fall back rather than approximate:
        an approximation here is a false accept, the one error the guard exists
        to prevent.

        The holes are chosen to be points the base gate really does accept, and
        whose axis values stay legal in other combinations -- otherwise the
        per-axis measurement removes them on its own and the fallback is never
        reached."""
        holes = {(64, 16), (96, 16), (128, 32), (192, 64)}

        def lumpy(p):
            return gate(p) and (p["head_size"], p["block_size"]) not in holes

        g = derive_guard(lumpy, CANDS, gate_name="lumpy")
        self.assertEqual(g["derivation"]["method"], "blocklist")
        for h, b in holes:
            self.assertFalse(
                guard_accepts(g, {"head_size": h, "block_size": b}),
                f"guard accepted the hole ({h}, {b})",
            )
        self.assertEqual(verify_guard(g, lumpy, CANDS, samples=2000).unsound, [])

    def test_allowlists_when_the_space_was_only_sampled(self):
        """A blocklist is only sound when every point inside the per-axis rules
        was tested; naming the bad ones is otherwise a claim about points nobody
        looked at. With the cross product merely sampled, derivation has to
        degrade the other way -- accept only what it confirmed."""
        holes = {(64, 16), (96, 16), (128, 32), (192, 64)}

        def lumpy(p):
            return gate(p) and (p["head_size"], p["block_size"]) not in holes

        g = derive_guard(lumpy, CANDS, gate_name="lumpy", max_cross=12)
        self.assertEqual(g["derivation"]["method"], "allowlist")
        self.assertFalse(g["derivation"]["exhaustive"])
        self.assertEqual(verify_guard(g, lumpy, CANDS, samples=2000).unsound, [])

    def test_single_axis_needs_no_coupling(self):
        g = derive_guard(
            lambda p: p["head_size"] in (16, 32, 64),
            {"head_size": CANDS["head_size"]},
        )
        self.assertEqual(g["derivation"]["method"], "factored")
        self.assertTrue(guard_accepts(g, {"head_size": 32}))
        self.assertFalse(guard_accepts(g, {"head_size": 48}))

    def test_refuses_when_nothing_is_legal(self):
        with self.assertRaises(GuardDerivationError):
            derive_guard(lambda p: False, CANDS)


class TestOracle(unittest.TestCase):
    def test_catches_an_unsound_guard(self):
        """The oracle is the reason a machine-fitted predicate is shippable at
        all, so it has to actually catch a guard that is too permissive. Here is
        one that forgot the coupling."""
        loose = {
            "schema": GUARD_SCHEMA,
            "free": ["block_size", "head_size"],
            "rules": axis_rules("head_size", list(range(16, 257, 16)))
            + axis_rules("block_size", [16, 32, 64]),
        }
        report = verify_guard(loose, gate, CANDS, samples=2000)
        self.assertFalse(report.sound)
        self.assertTrue(all(not gate(p) for p in report.unsound))

    def test_separates_strict_from_unsound(self):
        """Over-strict is a coverage loss, not a bug, and must not be reported
        as one -- a build that fails on strictness would block a family whose
        gate simply has no closed form."""
        tight = {
            "schema": GUARD_SCHEMA,
            "free": ["block_size", "head_size"],
            "rules": [
                {
                    "reason": "only 64/16",
                    "pred": {
                        "mul": [
                            {"eq": [{"spec": "head_size"}, 64]},
                            {"eq": [{"spec": "block_size"}, 16]},
                        ]
                    },
                }
            ],
        }
        report = verify_guard(tight, gate, CANDS, samples=2000)
        self.assertTrue(report.sound)
        self.assertTrue(report.strict)


class TestCheckGuard(unittest.TestCase):
    def setUp(self):
        self.g = derive_guard(gate, CANDS, gate_name="test")

    def test_no_guard_accepts(self):
        """Guards are additive: recipes recorded before they existed still run."""
        self.assertEqual(check_guard(None, {"anything": 7}, {}), (True, ""))

    def test_unbound_axis_is_refused_by_name(self):
        ok, why = check_guard(self.g, {"head_size": 64}, {})
        self.assertFalse(ok)
        self.assertIn("block_size", why)

    def test_reason_names_the_failing_constraint(self):
        ok, why = check_guard(self.g, {"head_size": 48, "block_size": 32}, {})
        self.assertFalse(ok)
        self.assertIn("divide", why)

    def test_unknown_schema_is_refused_not_accepted(self):
        """An engine older than the bundle cannot know what a newer guard would
        have rejected, so the only safe reading of one is 'no'."""
        ok, why = check_guard(
            {**self.g, "schema": "rocke.guard/v2"},
            {"head_size": 64, "block_size": 16},
            {},
        )
        self.assertFalse(ok)
        self.assertIn("schema", why)

    def test_require_verified_narrows_to_built_points(self):
        """The strict policy gives up the rolled interior for the strongest
        evidence: only points the generator actually built and compared."""
        built = [
            {"head_size": 64, "block_size": 16},
            {"head_size": 128, "block_size": 32},
        ]
        g = derive_guard(gate, CANDS, gate_name="test", verified=built)
        for pt in built:
            self.assertTrue(check_guard(g, pt, {}, require_verified=True)[0])

        other = {"head_size": 96, "block_size": 16}
        self.assertTrue(gate(other), "test needs a legal point outside the built set")
        self.assertTrue(check_guard(g, other, {})[0], "should pass the ordinary rules")
        ok, why = check_guard(g, other, {}, require_verified=True)
        self.assertFalse(ok)
        self.assertIn("verified", why)


class TestRecipeIntegration(unittest.TestCase):
    """The guard has to be enforced by the replay path, not just available to
    callers who remember to ask."""

    def _recipe(self):
        from rocke.portable_ir.examples import recipe_toy

        r = recipe_toy.make_recipe()
        # The toy takes D and dtype; guard D to a small legal set.
        g = derive_guard(
            lambda p: p["D"] in (2, 4, 8) and p["dtype"] == "f32",
            {"D": [1, 2, 3, 4, 6, 8, 12], "dtype": ["f32", "f16"]},
            gate_name="toy",
        )
        return attach_guard(r, g)

    def test_attach_checks_axes_match(self):
        from rocke.portable_ir.examples import recipe_toy

        with self.assertRaises(GuardDerivationError):
            attach_guard(
                recipe_toy.make_recipe(),
                {"schema": GUARD_SCHEMA, "free": ["D"], "rules": []},
            )

    def test_expand_enforces(self):
        r = self._recipe()
        out = expand_recipe(r, {"D": 4, "dtype": "f32"})
        self.assertTrue(out["program"])
        with self.assertRaises(GuardRejected):
            expand_recipe(r, {"D": 3, "dtype": "f32"})
        with self.assertRaises(GuardRejected):
            expand_recipe(r, {"D": 4, "dtype": "f16"})

    def test_expand_can_opt_out(self):
        """The generator has to replay points before it knows what the guard
        should say, so the enforcement has an explicit off switch."""
        r = self._recipe()
        self.assertTrue(expand_recipe(r, {"D": 3, "dtype": "f32"}, enforce_guard=False))


def _online_lib():
    """Path to a prebuilt shared librocke, or None. Same discovery the other
    portable-IR tests use; building one here would make a unit test take
    minutes."""
    import os
    import tempfile

    here = os.path.dirname(os.path.abspath(__file__))
    platform = os.path.normpath(os.path.join(here, "..", "..", "..", "..", ".."))
    for cand in (
        os.environ.get("ROCKE_ONLINE_LIB"),
        os.path.join(tempfile.gettempdir(), "rocke_online", "librocke.so"),
        os.path.join(platform, "build", "librocke.so"),
    ):
        if cand and os.path.exists(cand):
            return cand
    return None


@unittest.skipIf(
    _online_lib() is None,
    "no shared librocke; build one with online.build_lib() or set ROCKE_ONLINE_LIB",
)
class TestCEvaluatorParity(unittest.TestCase):
    """The Python and C guard evaluators must return the same verdict.

    They are separate implementations reading the same guard, and this is the
    property the whole design rests on: a caller asks in C++, the generator
    fitted and verified in Python, and the VM enforces in C. If those three can
    disagree, a guard that was proved sound against the gate in Python says
    nothing about what the engine will admit.

    Driven over ordinary values AND over zero, negatives, off-stride and
    out-of-range ones -- the inputs a JIT caller can actually produce, and the
    ones where the two evaluators' integer semantics differ."""

    POINTS = [
        {"D": d, "dtype": t}
        for d in (-8, -3, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 1000)
        for t in ("f32", "f16")
    ]

    @classmethod
    def setUpClass(cls):
        import os

        os.environ["ROCKE_ONLINE_LIB"] = _online_lib()
        from rocke.portable_ir.src import online, recipe_bundle
        from rocke.portable_ir.examples import recipe_toy

        cls.online, cls.recipe_bundle = online, recipe_bundle
        cls.guard = derive_guard(
            lambda p: p["D"] in (2, 4, 6, 8) and p["dtype"] == "f32",
            {"D": [1, 2, 3, 4, 5, 6, 7, 8, 9, 12], "dtype": ["f32", "f16"]},
            gate_name="toy",
        )
        cls.recipe = attach_guard(recipe_toy.make_recipe(), cls.guard)
        cls.cbor = recipe_bundle.cbor_encode(cls.recipe)

    def test_verdicts_match_python(self):
        for pt in self.POINTS:
            ints = {"D": pt["D"]}
            strs = {"dtype": pt["dtype"]}
            py_ok, py_why = check_guard(self.guard, ints, strs)
            verdict, c_why = self.online.check_recipe_guard(
                self.cbor, ints=ints, strs=strs
            )
            self.assertEqual(
                verdict,
                "admitted" if py_ok else "refused",
                f"{pt}: python said {py_ok!r} ({py_why}), C said {verdict!r} ({c_why})",
            )
            if not py_ok:
                self.assertEqual(py_why, c_why, f"{pt}: reasons differ")

    def test_stride_rule_agrees_on_negatives(self):
        """Pins the specific hazard: the emitter's divisibility rule asks only
        whether a remainder is zero, so floored and truncated `%` cannot reach
        different verdicts. A rule that compared a remainder against anything
        else would break here first."""
        rules = axis_rules("D", [2, 4, 6, 8])
        g = {"schema": GUARD_SCHEMA, "free": ["D"], "rules": rules}
        recipe = attach_guard(
            {**self.recipe, "spec": [{"name": "D", "kind": "int"}]}, g
        )
        cbor = self.recipe_bundle.cbor_encode(recipe)
        for d in range(-20, 21):
            py_ok, _ = check_guard(g, {"D": d}, {})
            verdict, why = self.online.check_recipe_guard(cbor, ints={"D": d})
            self.assertEqual(
                verdict, "admitted" if py_ok else "refused", f"D={d} ({why})"
            )

    def test_absent_guard_reported_not_assumed(self):
        from rocke.portable_ir.examples import recipe_toy

        cbor = self.recipe_bundle.cbor_encode(recipe_toy.make_recipe())
        verdict, _ = self.online.check_recipe_guard(
            cbor, ints={"D": 4}, strs={"dtype": "f32"}
        )
        self.assertEqual(verdict, "absent")

    def test_vm_refuses_to_lower_a_guarded_out_point(self):
        """The check is enforced by the replay path, so a caller who never asks
        still cannot compile an unsupported configuration."""
        ll, _ = self.online.recipe_cbor_to_llvm(
            self.cbor, ints={"D": 4}, strs={"dtype": "f32"}
        )
        self.assertIn("define", ll)
        with self.assertRaises(RuntimeError) as cm:
            self.online.recipe_cbor_to_llvm(
                self.cbor, ints={"D": 3}, strs={"dtype": "f32"}
            )
        self.assertIn("guard refused", str(cm.exception))

    def test_bundle_lookup_and_check(self):
        bundle = {
            "schema": "rocke.bundle/v1",
            "entries": [
                {"key": "toy", "arch": "gfx950", "recipe": self.recipe},
            ],
        }
        cbor = self.recipe_bundle.cbor_encode(bundle)
        self.assertTrue(self.online.bundle_contains(cbor, "toy", arch="gfx950"))
        self.assertFalse(self.online.bundle_contains(cbor, "toy", arch="gfx942"))
        self.assertFalse(self.online.bundle_contains(cbor, "nope", arch="gfx950"))

        verdict, _ = self.online.check_bundle_guard(
            cbor, "toy", arch="gfx950", ints={"D": 4}, strs={"dtype": "f32"}
        )
        self.assertEqual(verdict, "admitted")
        verdict, why = self.online.check_bundle_guard(
            cbor, "toy", arch="gfx950", ints={"D": 5}, strs={"dtype": "f32"}
        )
        self.assertEqual(verdict, "refused")
        self.assertTrue(why)

        # Absence is a distinct answer from refusal: for a pruned bundle it is
        # the rejection for concrete recipes.
        with self.assertRaises(KeyError):
            self.online.check_bundle_guard(
                cbor, "toy", arch="gfx942", ints={"D": 4}, strs={"dtype": "f32"}
            )

    def test_require_verified_flag_reaches_c(self):
        pt = dict(self.guard["verified"][0])
        ints = {k: v for k, v in pt.items() if not isinstance(v, str)}
        strs = {k: v for k, v in pt.items() if isinstance(v, str)}
        verdict, _ = self.online.check_recipe_guard(
            self.cbor, ints=ints, strs=strs, require_verified=True
        )
        self.assertEqual(verdict, "admitted")

        outside = [
            p
            for p in ({"D": d, "dtype": "f32"} for d in (2, 4, 6, 8))
            if p not in self.guard["verified"]
        ]
        if outside:
            ints = {"D": outside[0]["D"]}
            verdict, why = self.online.check_recipe_guard(
                self.cbor, ints=ints, strs={"dtype": "f32"}, require_verified=True
            )
            self.assertEqual(verdict, "refused")
            self.assertIn("verified", why)


if __name__ == "__main__":
    unittest.main()
