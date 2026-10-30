#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Self-test for pytorch_compile_probe.py's manifest and preflight rules.

Needs no PyTorch tree or hipDNN install: python test_pytorch_compile_probe.py
"""

import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import pytorch_compile_probe as probe  # noqa: E402


class _Stop(Exception):
    pass


class ProbeTest(unittest.TestCase):
    def test_sdpa_off_excludes_before_hipify_preflight(self):
        seen = {}

        def hipify(_root, _out, entries):
            seen.update({e["id"]: e["tier"] for e in entries})
            raise _Stop()

        argv = ["probe", "--pytorch-root", ".", "--pytorch-build", "."]
        argv += ["--hipdnn-prefix", ".", "--out", ".", "--only", "mha"]
        config = {"cudnn_compatibility": True, "sdpa": False, "skip_json_lib": False}
        with mock.patch.multiple(
            probe,
            preflight_env=mock.DEFAULT,
            preflight_tree=mock.DEFAULT,
            preflight_prefix=mock.Mock(return_value=config),
            tree_status=mock.Mock(return_value=[]),
            preflight_hipify=hipify,
        ), mock.patch.object(sys, "argv", argv), redirect_stdout(StringIO()):
            with self.assertRaises(_Stop):
                probe.main()
        self.assertEqual(seen, {"mha": "EXCLUDED"})

    def test_compile_tiers_require_a_canary(self):
        for tier in ("COMPILE", "COMPILE_ASIS"):
            entry = {"id": "x", "source": "a.cpp", "compiled": "b.cpp", "tier": tier}
            with tempfile.TemporaryDirectory() as tmp:
                manifest = Path(tmp) / "manifest.json"
                manifest.write_text(json.dumps({"entries": [entry]}))
                with mock.patch.object(probe, "MANIFEST", manifest):
                    with self.assertRaises(probe.ProbeError, msg=tier):
                        probe.load_manifest()

    def test_shipped_manifest_loads(self):
        with redirect_stderr(StringIO()):
            manifest = probe.load_manifest()
        for entry in manifest["entries"]:
            if entry["tier"] in ("COMPILE", "COMPILE_ASIS"):
                self.assertTrue(entry.get("canary"), entry["id"])


if __name__ == "__main__":
    unittest.main()
