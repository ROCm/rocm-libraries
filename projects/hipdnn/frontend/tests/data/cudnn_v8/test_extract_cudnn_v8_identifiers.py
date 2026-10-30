#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT
"""Self-test for extract_cudnn_v8_identifiers.py's input pins; runs as a ctest case.

Uses stand-in input trees, so it needs no network, upstream checkout or GPU.
"""

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_cudnn_v8_identifiers as extractor  # noqa: E402

FRONTEND_FILES = {
    "include/cudnn_frontend.h": '#include "cudnn_frontend_Tensor.h"\n',
    "include/cudnn_frontend_version.h": (
        "#define CUDNN_FRONTEND_MAJOR_VERSION 1\n"
        "#define CUDNN_FRONTEND_MINOR_VERSION 24\n"
        "#define CUDNN_FRONTEND_PATCH_VERSION 0\n"
    ),
    "include/cudnn_frontend_Tensor.h": "cudnnBackendDescriptor_t desc;\n",
    "include/cudnn_backend_base.h": "cudnnStatus_t status;\n",
    "include/cudnn_frontend/knobs.h": "CUDNN_KNOB_TYPE_COUNTS;\n",
    "samples/legacy_samples/conv_sample.cpp": "cudnnHandle_t handle;\n",
}
PYTORCH_FILE = "aten/src/ATen/native/cudnn/Conv_v8.cpp"
PYTORCH_FILES = {PYTORCH_FILE: "cudnnBackendExecute(handle, plan, pack);\n"}
FRONTEND_VERSION = 12400


class _ReachedScan(Exception):
    pass


def sha256(text: bytes) -> str:
    return hashlib.sha256(text.replace(b"\r\n", b"\n")).hexdigest()


def write_tree(root: Path, files: dict) -> None:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(text.encode("utf-8"))


class CheckInputsTest(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.root = Path(self._dir.name)
        self.frontend = self.root / "cudnn-frontend"
        self.pytorch = self.root / "pytorch"
        write_tree(self.frontend, FRONTEND_FILES)
        write_tree(self.pytorch, PYTORCH_FILES)
        # Recomputed here rather than taken from the extractor, so a change to the
        # digest format also fails, as it would against the committed pins.
        tree = hashlib.sha256()
        for relative in sorted(FRONTEND_FILES):
            digest = sha256(FRONTEND_FILES[relative].encode("utf-8"))
            tree.update(f"{relative}\0{digest}\n".encode("utf-8"))
        self.pins_file = self.root / "pins.json"
        self.pins = {
            "cudnn_frontend": {"inputs_sha256": tree.hexdigest()},
            "pytorch": {
                "repo": "pytorch",
                "tag": "v0",
                "commit": "1" * 40,
                "files": {
                    relative: sha256(text.encode("utf-8"))
                    for relative, text in PYTORCH_FILES.items()
                },
            },
        }
        self.pins_file.write_text(json.dumps(self.pins), encoding="utf-8")
        self.set_frontend_commit("0" * 40)

    def tearDown(self):
        self._dir.cleanup()

    def set_frontend_commit(self, commit: str) -> None:
        self.manifest = self.root / "manifest.json"
        corpus = {
            "repo": "cudnn-frontend",
            "tag": "v1.24.0",
            "commit": commit,
            "frontend_version": FRONTEND_VERSION,
        }
        self.manifest.write_text(json.dumps({"corpus": corpus}), encoding="utf-8")

    def run_extractor(self):
        """Exit code and stderr of --check; scanning means the pin checks passed."""
        argv = ["extract", "--check", "--output", str(self.root / "out")]
        argv += ["--cudnn-frontend", str(self.frontend), "--pytorch", str(self.pytorch)]
        stderr = StringIO()
        with mock.patch.multiple(
            extractor,
            PINS_FILE=self.pins_file,
            CORPUS_MANIFEST=self.manifest,
            scan=mock.Mock(side_effect=_ReachedScan),
        ), mock.patch.object(sys, "argv", argv), redirect_stderr(
            stderr
        ), redirect_stdout(
            StringIO()
        ):
            code = extractor.main()
        return code, stderr.getvalue()

    def assert_rejected(self, message: str) -> None:
        code, stderr = self.run_extractor()
        self.assertEqual(code, 2, stderr)
        self.assertIn("inputs do not match their pins", stderr)
        self.assertIn(message, stderr)

    def test_pinned_inputs_pass(self):
        with self.assertRaises(_ReachedScan):
            self.run_extractor()

    def test_crlf_copy_of_pinned_inputs_passes(self):
        path = self.frontend / "include" / "cudnn_frontend_Tensor.h"
        path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
        with self.assertRaises(_ReachedScan):
            self.run_extractor()

    def test_changed_frontend_file_is_rejected(self):
        path = self.frontend / "include" / "cudnn_frontend_Tensor.h"
        path.write_bytes(path.read_bytes() + b"cudnnTensorDescriptor_t extra;\n")
        self.assert_rejected("cudnn-frontend inputs sha256")

    def test_added_frontend_file_is_rejected(self):
        write_tree(self.frontend, {"samples/legacy_samples/extra.h": "\n"})
        self.assert_rejected("cudnn-frontend inputs sha256")

    def test_changed_pytorch_file_is_rejected(self):
        path = self.pytorch / PYTORCH_FILE
        path.write_bytes(path.read_bytes() + b"\n")
        self.assert_rejected(f"PyTorch {PYTORCH_FILE} sha256")

    @unittest.skipUnless(shutil.which("git"), "git is not on PATH")
    def test_checkout_at_another_commit_is_rejected(self):
        git = ["git", "-C", str(self.frontend), "-c", "core.autocrlf=false"]
        git += ["-c", "user.name=t", "-c", "user.email=t@t"]
        # Signing and hooks from the user's global config must not fail the commit.
        git += ["-c", "commit.gpgsign=false"]
        subprocess.run(git + ["init", "-q"], check=True)
        subprocess.run(git + ["add", "-A"], check=True)
        subprocess.run(git + ["commit", "-q", "--no-verify", "-m", "x"], check=True)
        head = subprocess.run(
            git + ["rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()

        self.assert_rejected(f"cudnn-frontend checkout is at {head}")

        self.set_frontend_commit(head)
        with self.assertRaises(_ReachedScan):
            self.run_extractor()


if __name__ == "__main__":
    unittest.main()
