# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Tests for the two compatibility contracts (src/abi.py, cpp/include/rocke/abi.h).
#
# What these are really defending is a deployment property that no single-process
# test can observe directly: a bundle written today will be read by an engine
# built at some other time, in either direction. So the tests are written around
# the two questions that actually decide whether that goes wrong.
#
#   Does an engine refuse what it would misread?  A future artifact must be
#   refused by BOTH engines, with an error that says "too old", and it must be
#   refused before any output is produced.
#
#   Does it refuse anything MORE than that?  This is the expensive failure and
#   the reason min_reader is derived from content rather than stamped from the
#   generator's own version: an artifact using only old constructs must stay
#   readable no matter how new the generator was.
#
#   python3 -m unittest rocke.portable_ir.tests.test_abi

import ctypes
import os
import tempfile
import unittest

from rocke.portable_ir.examples import recipe_toy
from rocke.portable_ir.src import abi
from rocke.portable_ir.src.recipe_bundle import cbor_encode
from rocke.portable_ir.utils.recipe_expand import ExpandError, expand_recipe

SPEC = {"D": 4, "dtype": "f32"}


def from_the_future(artifact, bump=1):
    """An artifact as a NEWER generator would have written it.

    Written by hand rather than through abi.stamp because stamp refuses to
    over-claim: a level-N generator cannot honestly require level N+1, since it
    does not know any N+1 construct to have used. Forging the block here is the
    point -- it is the only way to stand in for the deployment this whole scheme
    exists for, an engine meeting a bundle from a build it has never seen."""
    out = dict(artifact)
    out["abi"] = {
        "min_reader": abi.RECIPE_ABI + bump,
        "writer": abi.RECIPE_ABI + bump,
        "engine": "99.0.0+future",
        "build_id": "f" * 12,
    }
    return out


class TestMinReaderDerivation(unittest.TestCase):
    def test_ordinary_recipe_is_level_one(self):
        """The property that keeps a generator upgrade from being a flag day:
        a recipe using only long-standing constructs declares the floor, so
        every engine ever built can still read it."""
        self.assertEqual(abi.recipe_min_reader(recipe_toy.make_recipe()), 1)

    def test_stamp_is_derived_not_asserted(self):
        stamped = abi.stamp(recipe_toy.make_recipe(), engine="test")
        self.assertEqual(stamped["abi"]["min_reader"], 1)
        self.assertEqual(stamped["abi"]["writer"], abi.RECIPE_ABI)

    def test_unregistered_op_refuses_to_stamp(self):
        """A generator that cannot describe what it just emitted must not make
        compatibility claims about it. This is the one part of registry drift a
        machine can catch."""
        r = recipe_toy.make_recipe()
        r["program"].append({"op": "tensor.teleport", "bind": "x"})
        with self.assertRaises(abi.AbiError) as cm:
            abi.stamp(r)
        self.assertIn("tensor.teleport", str(cm.exception))
        # ...and inspection of a foreign recipe stays possible.
        self.assertEqual(abi.recipe_min_reader(r, strict=False), 1)

    def test_stamp_refuses_to_over_claim(self):
        """A generator cannot honestly require a reader newer than itself: it
        knows no construct that would justify it, so the claim could only be a
        mistake, and it would strand the artifact on every engine in existence."""
        with self.assertRaises(abi.AbiError):
            abi.stamp(recipe_toy.make_recipe(), min_reader=abi.RECIPE_ABI + 1)

    def test_finds_constructs_nested_in_compile_time_bodies(self):
        """static_for bodies are where most of a rolled recipe lives, so a walk
        that only looked at the top level would report the floor for almost
        everything and the gate would be decorative."""
        r = recipe_toy.make_recipe()
        body = next(i for i in r["program"] if i["op"] == "static_for")["body"]
        body.append({"op": "tensor.teleport"})
        with self.assertRaises(abi.AbiError):
            abi.stamp(r)


class TestRegistryIsComplete(unittest.TestCase):
    def test_registry_lists_exactly_the_ops_the_expander_implements(self):
        """Reads the expander's source on purpose.

        The registry is only load-bearing if it keeps up with the engines, and
        the failure mode is silent: an op added to both VMs but not registered
        still works, so nothing complains until the day someone needs the level
        it should have carried. Recovering the dispatch table from source is
        crude, but it fails the moment the two fall out of step, which no test
        written against behaviour would."""
        import inspect
        import re

        from rocke.portable_ir.utils import recipe_expand

        src = inspect.getsource(recipe_expand._Expander._instr)
        implemented = set(re.findall(r'op == "([a-z0-9_]+)"', src))
        self.assertEqual(
            implemented,
            set(abi.INSTR_OPS),
            "abi.INSTR_OPS and the expander's dispatch have diverged; add the "
            "op to the registry in the change that adds it to the VMs",
        )


class TestReaderCheck(unittest.TestCase):
    def test_absent_block_means_level_one(self):
        """Recipes recorded before the block existed must stay readable."""
        abi.check(recipe_toy.make_recipe())
        self.assertTrue(expand_recipe(recipe_toy.make_recipe(), SPEC)["program"])

    def test_future_artifact_refused(self):
        future = from_the_future(recipe_toy.make_recipe())
        with self.assertRaises(abi.AbiError) as cm:
            abi.check(future)
        self.assertIn(f">= {abi.RECIPE_ABI + 1}", str(cm.exception))

    def test_expander_refuses_a_future_recipe(self):
        future = from_the_future(recipe_toy.make_recipe())
        with self.assertRaises(abi.AbiError):
            expand_recipe(future, SPEC)

    def test_expander_now_checks_schema(self):
        """The C VM always checked this and the Python mirror did not, so the
        oracle would replay recipes the engine it mirrors refuses."""
        r = dict(recipe_toy.make_recipe(), schema="rocke.recipe/v2")
        with self.assertRaises(ExpandError) as cm:
            expand_recipe(r, SPEC)
        self.assertIn("schema", str(cm.exception))

    def test_provenance_is_not_compared(self):
        """A newer writer with old content stays readable. If this ever fails,
        someone has started comparing `writer` or `engine`, and every deployed
        engine is about to reject every new bundle."""
        r = abi.stamp(recipe_toy.make_recipe(), engine="99.0.0+future")
        r["abi"]["writer"] = abi.RECIPE_ABI + 5
        abi.check(r)
        self.assertTrue(expand_recipe(r, SPEC)["program"])


def _online_lib():
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


@unittest.skipIf(_online_lib() is None, "no shared librocke; run online.build_lib()")
class TestCEngineAgrees(unittest.TestCase):
    """The C engine has to make the same admission decisions as the Python one.

    A version scheme that the two engines interpret differently is worse than
    none: it would license exactly the mixed-build deployment it exists to stop."""

    @classmethod
    def setUpClass(cls):
        os.environ["ROCKE_ONLINE_LIB"] = _online_lib()
        from rocke.portable_ir.src import online

        cls.online = online
        cls.lib = online.load()

    def test_levels_match_the_python_mirror(self):
        """src/abi.py hard-codes both numbers; cpp/include/rocke/abi.h defines
        them. Nothing but this test stops the two from drifting apart, and a
        Python generator that thought it was newer than the engine would stamp
        artifacts the engine then refuses."""
        self.assertEqual(self.lib.rocke_abi_version(), abi.BINARY_ABI)
        self.assertEqual(self.lib.rocke_recipe_abi_level(), abi.RECIPE_ABI)

    def test_non_int_returns_have_an_explicit_restype(self):
        """Every entry point whose return is not an int must say so.

        ctypes defaults an unset restype to c_int and reads four bytes back. For
        a one-byte `bool` that means three bytes of whatever was left in the
        register: an -O0 build tends to zero them and the binding looks correct,
        while an optimized build does not and `bool(...)` reports true for a
        function that returned false. Exactly that shipped here for a while,
        passing every local run and failing only under the gates' release build,
        so the restypes are pinned rather than trusted."""
        want = {
            "rocke_bundle_contains": ctypes.c_bool,
            "rocke_launch_plan_geometry": ctypes.c_bool,
            "rocke_launch_plan_kernel_name": ctypes.c_char_p,
            "rocke_launch_plan_kernarg_size": ctypes.c_uint,
            "rocke_engine_version": ctypes.c_char_p,
            "rocke_build_id": ctypes.c_char_p,
        }
        for name, restype in want.items():
            self.assertIs(getattr(self.lib, name).restype, restype, name)
        self.assertIsNone(self.lib.rocke_online_free.restype)

    def test_bundles_record_real_provenance(self):
        """Provenance is obtained through a soft path, so its failure mode is an
        empty string rather than an error -- which is how it first shipped
        blank, the getters having never been given a ctypes restype. A bundle
        built where an engine exists must name it, or the field is decoration
        and the first artifact that needs tracing down cannot be."""
        from rocke.portable_ir.src.recipe_bundle import build_bundle

        block = build_bundle(
            [{"key": "toy", "arch": "gfx950", "recipe": recipe_toy.make_recipe()}]
        )["abi"]
        self.assertTrue(block["engine"], "engine version stamped empty")
        self.assertTrue(block["build_id"], "build id stamped empty")
        self.assertEqual(block["min_reader"], 1)

    def test_c_vm_refuses_a_future_recipe(self):
        future = from_the_future(recipe_toy.make_recipe())
        with self.assertRaises(RuntimeError) as cm:
            self.online.recipe_cbor_to_llvm(
                cbor_encode(future), ints={"D": 4}, strs={"dtype": "f32"}
            )
        self.assertIn("recipe reader >=", str(cm.exception))

    def test_c_vm_accepts_a_stamped_current_recipe(self):
        """Stamping must be free: the block is metadata, so an artifact that
        carries it lowers exactly as one that does not."""
        plain = recipe_toy.make_recipe()
        stamped = abi.stamp(plain, engine="test")
        a, _ = self.online.recipe_cbor_to_llvm(
            cbor_encode(plain), ints={"D": 4}, strs={"dtype": "f32"}
        )
        b, _ = self.online.recipe_cbor_to_llvm(
            cbor_encode(stamped), ints={"D": 4}, strs={"dtype": "f32"}
        )
        self.assertEqual(a, b)

    def test_future_bundle_refused_before_lookup(self):
        bundle = from_the_future(
            {
                "schema": "rocke.bundle/v1",
                "entries": [
                    {"key": "toy", "arch": "gfx950", "recipe": recipe_toy.make_recipe()}
                ],
            }
        )
        blob = cbor_encode(bundle)
        with self.assertRaises(RuntimeError) as cm:
            self.online.bundle_cbor_to_llvm(
                blob, "toy", arch="gfx950", ints={"D": 4}, strs={"dtype": "f32"}
            )
        self.assertIn("recipe reader >=", str(cm.exception))

    def test_guard_check_reports_too_old_as_error_not_refusal(self):
        """A caller must be able to tell a deployment problem from an
        unsupported shape: the first is fixed by shipping matched artifacts, the
        second by routing elsewhere. Answering 'refused' here would file the
        former under the latter and nobody would ever look."""
        bundle = {
            "schema": "rocke.bundle/v1",
            "entries": [
                {
                    "key": "toy",
                    "arch": "gfx950",
                    "recipe": from_the_future(recipe_toy.make_recipe()),
                }
            ],
        }
        blob = cbor_encode(bundle)
        with self.assertRaises(RuntimeError) as cm:
            self.online.check_bundle_guard(
                blob, "toy", arch="gfx950", ints={"D": 4}, strs={"dtype": "f32"}
            )
        self.assertIn("recipe reader >=", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
