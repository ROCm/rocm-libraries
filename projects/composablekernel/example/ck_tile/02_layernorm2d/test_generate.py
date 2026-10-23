# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CPU-only regression checks: python3 -m unittest discover -s . -p test_generate.py."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

spec = importlib.util.spec_from_file_location(
    "layernorm_generate", Path(__file__).with_name("generate.py")
)
generate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(generate)


class InstanceFilterTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(disable_16b_welford=False)
        self.all_blobs = generate.layernorm_fwd_codegen(".", None).get_blobs(self.args)

    def test_empty_filter_preserves_all_instances(self):
        self.assertTrue(self.all_blobs)
        for pattern in ("", "*"):
            with self.subTest(pattern=pattern):
                codegen = generate.layernorm_fwd_codegen(".", pattern)
                self.assertEqual(codegen.get_blobs(self.args), self.all_blobs)

    def test_exact_filter_keeps_original_instance_content(self):
        name = "layernorm2d_fwd_fp16_n256"
        selected = generate.layernorm_fwd_codegen(".", name).get_blobs(self.args)
        self.assertEqual([blob.name for blob in selected], [name])
        original = next(blob for blob in self.all_blobs if blob.name == name)
        self.assertEqual(selected[0].content, original.content)

    def test_wildcard_selects_multiple_instances(self):
        blobs = generate.layernorm_fwd_codegen(
            ".", "layernorm2d_fwd_fp16_n*"
        ).get_blobs(self.args)
        self.assertGreater(len(blobs), 1)
        self.assertLess(len(blobs), len(self.all_blobs))
        self.assertTrue(
            all(blob.name.startswith("layernorm2d_fwd_fp16_n") for blob in blobs)
        )

    def test_unmatched_filter_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "matched no kernels"):
            generate.layernorm_fwd_codegen(".", "nonexistent").get_blobs(self.args)

    def test_big_only_filter_keeps_unbounded_dispatch(self):
        for pattern in (
            "layernorm2d_fwd_fp16_nbig",
            "layernorm2d_fwd_bf16_nbig",
            "*_nbig*",
        ):
            with self.subTest(pattern=pattern):
                codegen = generate.layernorm_fwd_codegen(".", pattern)
                api = codegen.content_api(self.args)
                self.assertNotRegex(api, r"\bif\s*\{")
                self.assertNotIn("a.n <=", api)
                for blob in codegen.get_blobs(self.args):
                    for instance in blob.instance_list:
                        self.assertIn(instance.call_name, api)

    def test_list_generation_and_dispatch_select_same_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            codegen = generate.layernorm_fwd_codegen(
                directory, "layernorm2d_fwd_fp16_n256"
            )
            codegen.list_blobs(self.args)
            codegen.gen_blobs(self.args)
            listed = (
                Path(directory, "layernorm2d_fwd_blobs.txt").read_text().splitlines()
            )
            self.assertEqual(len(listed), 3)
            self.assertEqual(
                set(listed),
                {
                    str(p)
                    for p in Path(directory).iterdir()
                    if p.suffix in (".cpp", ".hpp")
                },
            )
            api = Path(directory, "layernorm2d_fwd_api.cpp").read_text()
            selected = codegen.get_blobs(self.args)[0]
            for instance in selected.instance_list:
                self.assertIn(instance.call_name, api)
            excluded = generate.layernorm_fwd_codegen(
                ".", "layernorm2d_fwd_bf16_n256"
            ).get_blobs(self.args)[0]
            for instance in excluded.instance_list:
                self.assertNotIn(instance.call_name, api)


if __name__ == "__main__":
    unittest.main()
