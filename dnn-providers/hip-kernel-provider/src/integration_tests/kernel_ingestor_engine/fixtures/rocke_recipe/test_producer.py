# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Offline bundle selection/regeneration tests; run with ROCKE_BACKEND=python."""

import json
from pathlib import Path
import tempfile
import unittest

from produce_sdpa import KEY, produce, selected_arches
from rocke.portable_ir.src.recipe_bundle import bundle_lookup, read_bundle


class ProducerTests(unittest.TestCase):
    def test_target_selection(self):
        self.assertEqual(
            selected_arches(["gfx950:xnack-", "gfx942", "gfx950"]),
            ["gfx942", "gfx950"],
        )
        for targets in ([], ["gfx90a"], ["gfx950", "gfx1100"], ["gfx9XX"]):
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                selected_arches(targets)

    def test_dual_then_single_target_regeneration(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            manifest = produce(output, ["gfx950", "gfx942"], "llvm23")
            bundle = read_bundle(str(output / "sdpa_dense.cbor"))
            self.assertEqual(len(bundle["entries"]), 4)
            self.assertEqual(
                json.loads((output / "sdpa.kdp.json").read_text())["arch"],
                ["gfx942", "gfx950"],
            )
            for arch in ("gfx942", "gfx950"):
                self.assertIsNotNone(bundle_lookup(bundle, KEY, arch))
                self.assertIsNotNone(bundle_lookup(bundle, KEY + "_short", arch))
                refs = manifest["targets"][arch]["references"]
                self.assertEqual(set(refs), {"512", "768", "1024"})
                for sequence, reference in refs.items():
                    self.assertTrue(
                        (output / arch / f"reference-{sequence}.ll").is_file()
                    )
                    names = {arg["name"] for arg in reference["launch"]["args"]}
                    self.assertTrue(
                        {"q_ptr", "k_ptr", "v_ptr", "o_ptr", "scale"} <= names
                    )
                    if arch == "gfx950":
                        self.assertTrue({"batch", "seqlen_q", "seqlen_kv"} <= names)
            self.assertIsNone(bundle_lookup(bundle, KEY, "gfx90a"))

            # Reusing a build directory must remove a formerly selected target
            # from both runtime inputs, even though reference evidence can remain.
            manifest = produce(output, ["gfx942"], "llvm23")
            bundle = read_bundle(str(output / "sdpa_dense.cbor"))
            self.assertEqual(len(bundle["entries"]), 2)
            self.assertEqual(set(manifest["targets"]), {"gfx942"})
            self.assertIsNone(bundle_lookup(bundle, KEY, "gfx950"))
            self.assertEqual(
                json.loads((output / "sdpa.kdp.json").read_text())["arch"], ["gfx942"]
            )
            before = (output / "sdpa_dense.cbor").read_bytes()
            with self.assertRaises(ValueError):
                produce(output, ["gfx942", "gfx90a"], "llvm23")
            self.assertEqual((output / "sdpa_dense.cbor").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
