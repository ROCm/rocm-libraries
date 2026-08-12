# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
#
# Tests for launch geometry and the launch-plan API (src/launch.py,
# cpp/include/rocke/recipe_launch.h).
#
# What these defend is the step that used to be impossible without an
# interpreter: taking a shipped CBOR bundle all the way to a launch. Two
# properties carry the weight.
#
#   The two engines agree. A C client and a Python client must compute the same
#   grid and the same kernarg offsets for the same shape, or a kernel validated
#   through one path is launched wrongly through the other.
#
#   Offsets follow natural alignment. This only becomes observable when a
#   signature mixes widths, so the interesting case is tested explicitly rather
#   than relying on the all-pointer recipes that happen to be lying around --
#   those are correct under back-to-back packing too, and would pass whatever
#   the code did.
#
#   python3 -m unittest rocke.portable_ir.tests.test_launch

import os
import tempfile
import unittest

from rocke.portable_ir.examples import recipe_toy
from rocke.portable_ir.src import launch
from rocke.portable_ir.src.recipe_bundle import build_bundle, cbor_encode
from rocke.portable_ir.utils.recipe_expand import ExpandError

# grid.x = ceil(D / 64), the shape of a real elementwise dispatch.
CEIL_DIV_D_64 = {"div": [{"add": [{"spec": "D"}, 63]}, 64]}


def toy(**kw):
    return launch.attach_launch(
        recipe_toy.make_recipe(),
        grid=kw.pop("grid", [CEIL_DIV_D_64, 1, 1]),
        block=kw.pop("block", [64, 1, 1]),
        **kw,
    )


def mixed_signature_recipe():
    """A kernel whose args mix 8- and 4-byte widths.

    The recipes in the tree are all-pointer, and every packing rule agrees on
    those. (ptr, i32, ptr) is the smallest signature where natural alignment and
    back-to-back packing disagree: the trailing pointer belongs at 16, not 12."""
    ptr = {"kind": "ptr", "pointee": "f32", "space": "global"}
    return {
        "schema": "rocke.recipe/v1",
        "kernel_name_fmt": "mixed_{D}",
        "spec": [],
        "attrs": {},
        "program": [
            {"op": "param", "name": "A", "type": ptr, "bind": "A"},
            {"op": "param", "name": "n", "type": "i32", "bind": "n"},
            {"op": "param", "name": "B", "type": ptr, "bind": "B"},
            {"op": "ret"},
        ],
    }


class TestGeometry(unittest.TestCase):
    def test_grid_is_a_function_of_the_shape(self):
        r = toy()
        for d, want in ((64, 1), (65, 2), (256, 4), (1000, 16)):
            g = launch.eval_launch(r, {"D": d}, {"dtype": "f32"})
            self.assertEqual(g["grid"], (want, 1, 1), f"D={d}")
            self.assertEqual(g["block"], (64, 1, 1))

    def test_absent_geometry_is_none_not_a_default(self):
        """A recipe with no launch block is not a recipe that wants one
        workgroup. Defaulting would turn missing metadata into a wrong launch,
        which is undetectable at the call site."""
        self.assertIsNone(
            launch.eval_launch(recipe_toy.make_recipe(), {"D": 4}, {"dtype": "f32"})
        )

    def test_nonpositive_extent_is_refused(self):
        r = launch.attach_launch(
            recipe_toy.make_recipe(),
            grid=[{"sub": [{"spec": "D"}, 4]}, 1, 1],
            block=[64, 1, 1],
        )
        with self.assertRaises(ExpandError) as cm:
            launch.eval_launch(r, {"D": 4}, {"dtype": "f32"})
        self.assertIn(">= 1", str(cm.exception))

    def test_malformed_block_is_refused(self):
        r = dict(recipe_toy.make_recipe(), launch={"grid": [1, 1], "block": [64, 1, 1]})
        with self.assertRaises(ExpandError):
            launch.eval_launch(r, {"D": 4}, {"dtype": "f32"})

    def test_attach_validates_shape_at_authoring_time(self):
        """Caught in the generator, where the fix is, rather than inside a JIT
        on a machine that cannot trace it back."""
        with self.assertRaises(ExpandError):
            launch.attach_launch(
                recipe_toy.make_recipe(), grid=[1, 1], block=[64, 1, 1]
            )


class TestSignature(unittest.TestCase):
    def test_natural_alignment_on_a_mixed_signature(self):
        args = launch.signature(mixed_signature_recipe(), {"D": 1}, {})
        self.assertEqual(
            [(a["name"], a["offset"], a["size"]) for a in args],
            [("A", 0, 8), ("n", 8, 4), ("B", 16, 8)],
            "the trailing pointer must be padded to 8-byte alignment",
        )
        self.assertEqual(launch.kernarg_size(args), 24)

    def test_kernarg_size_matches_what_python_actually_packs(self):
        """Pins this against runtime/packing.py, the third implementation of the
        same rule and the one already in production. If they disagree, one of
        the two is writing outside a buffer the other sized."""
        from rocke.runtime.packing import pack_args

        args = launch.signature(mixed_signature_recipe(), {"D": 1}, {})
        blob = pack_args(
            [{"name": a["name"], "type": a["type"]} for a in args],
            {"A": 0x1000, "n": 7, "B": 0x2000},
        )
        self.assertEqual(len(blob), launch.kernarg_size(args))

    def test_unsupported_arg_type_refuses_the_plan(self):
        """A guessed width does not fail, it shifts every following argument."""
        r = mixed_signature_recipe()
        r["program"][1]["type"] = "f16"
        with self.assertRaises(ExpandError) as cm:
            launch.signature(r, {"D": 1}, {})
        self.assertIn("f16", str(cm.exception))


def _online_lib():
    for cand in (
        os.environ.get("ROCKE_ONLINE_LIB"),
        os.path.join(tempfile.gettempdir(), "rocke_online", "librocke.so"),
    ):
        if cand and os.path.exists(cand):
            return cand
    return None


@unittest.skipIf(_online_lib() is None, "no shared librocke; run online.build_lib()")
class TestCEngineAgrees(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["ROCKE_ONLINE_LIB"] = _online_lib()
        from rocke.portable_ir.src import online

        cls.online = online
        online.load()

    def test_plans_match_across_shapes(self):
        """The whole point: a C client and a Python client launch identically."""
        r = toy()
        for d in (64, 65, 256, 1000, 4096):
            si, ss = {"D": d}, {"dtype": "f32"}
            self.assertEqual(
                self.online.plan_launch(cbor_encode(r), ints=si, strs=ss),
                launch.plan(r, si, ss),
                f"engines disagree at D={d}",
            )

    def test_mixed_signature_offsets_match(self):
        r = mixed_signature_recipe()
        c = self.online.plan_launch(cbor_encode(r), ints={"D": 1})
        self.assertEqual(
            [(a["name"], a["offset"], a["size"]) for a in c["args"]],
            [("A", 0, 8), ("n", 8, 4), ("B", 16, 8)],
        )
        self.assertEqual(c["kernarg_size"], 24)
        self.assertEqual(c["args"], launch.signature(r, {"D": 1}, {}))

    def test_absent_geometry_reported_as_absent(self):
        c = self.online.plan_launch(
            cbor_encode(recipe_toy.make_recipe()), ints={"D": 4}, strs={"dtype": "f32"}
        )
        self.assertIsNone(c["geometry"])
        self.assertEqual(c["kernel_name"], "rocke_recipe_toy_d4_f32")

    def test_plan_from_a_bundle(self):
        bundle = cbor_encode(
            build_bundle([{"key": "toy", "arch": "gfx950", "recipe": toy()}])
        )
        c = self.online.plan_launch(
            bundle, "toy", arch="gfx950", ints={"D": 256}, strs={"dtype": "f32"}
        )
        self.assertEqual(c["geometry"]["grid"], (4, 1, 1))
        with self.assertRaises(KeyError):
            self.online.plan_launch(
                bundle, "nope", arch="gfx950", ints={"D": 256}, strs={"dtype": "f32"}
            )

    def test_nonpositive_extent_refused_by_c_too(self):
        r = launch.attach_launch(
            recipe_toy.make_recipe(),
            grid=[{"sub": [{"spec": "D"}, 4]}, 1, 1],
            block=[64, 1, 1],
        )
        with self.assertRaises(RuntimeError) as cm:
            self.online.plan_launch(
                cbor_encode(r), ints={"D": 4}, strs={"dtype": "f32"}
            )
        self.assertIn(">= 1", str(cm.exception))

    def test_guard_refusal_blocks_planning(self):
        """Planning a launch for a shape the kernel will not serve is not a
        meaningful question, and answering it would hand back a grid for a
        kernel that is never going to be built."""
        from rocke.portable_ir.utils.recipe_expand import GUARD_SCHEMA

        r = toy()
        r["guard"] = {
            "schema": GUARD_SCHEMA,
            "free": ["D"],
            "rules": [
                {"axis": "D", "kind": "bounds", "pred": {"ge": [{"spec": "D"}, 128]}}
            ],
        }
        blob = cbor_encode(r)
        self.assertEqual(
            self.online.plan_launch(blob, ints={"D": 256}, strs={"dtype": "f32"})[
                "geometry"
            ]["grid"],
            (4, 1, 1),
        )
        with self.assertRaises(RuntimeError) as cm:
            self.online.plan_launch(blob, ints={"D": 64}, strs={"dtype": "f32"})
        self.assertIn("guard", str(cm.exception).lower())


if __name__ == "__main__":
    unittest.main()
