#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import importlib.util
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "generate_sdpa_fwd_golden.py"


@unittest.skipIf(
    importlib.util.find_spec("torch") is None, "the generator requires PyTorch"
)
class TestGenerateSdpaFwdGoldenScale(unittest.TestCase):
    def generate(self, *extra_args: str) -> tuple[dict, dict]:
        with TemporaryDirectory() as tmp:
            base = Path(tmp) / "bundle"
            command = [
                sys.executable,
                str(SCRIPT_PATH),
                "--base-filename",
                str(base),
                "--q-dims",
                "1",
                "1",
                "4",
                "16",
                "--v-dims",
                "1",
                "1",
                "4",
                "16",
                *extra_args,
            ]
            subprocess.run(command, capture_output=True, text=True, check=True)
            graph = json.loads(base.with_suffix(".json").read_text())
            meta = json.loads(Path(f"{base}.meta.json").read_text())
            return graph, meta

    def test_no_attn_scale_writes_one(self):
        # An unset attn_scale_value means no scaling in hipDNN, as in cuDNN.
        graph, meta = self.generate()
        self.assertEqual(graph["nodes"][0]["attributes"]["attn_scale_value"], 1.0)
        self.assertEqual(meta["config"]["scale"], 1.0)

    def test_explicit_attn_scale_is_kept(self):
        graph, meta = self.generate("--attn-scale", "0.25")
        self.assertEqual(graph["nodes"][0]["attributes"]["attn_scale_value"], 0.25)
        self.assertEqual(meta["config"]["scale"], 0.25)


if __name__ == "__main__":
    unittest.main()
