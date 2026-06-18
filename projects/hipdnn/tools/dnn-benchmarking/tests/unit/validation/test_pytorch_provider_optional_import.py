# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests that the PyTorch provider keeps torch optional at import time."""

import os
import subprocess
import sys


def test_cli_and_provider_import_without_torch() -> None:
    """Importing the CLI must not require torch until the provider is used."""
    code = """
import builtins
real_import = builtins.__import__

def import_without_torch(name, *args, **kwargs):
    if name == 'torch' or name.startswith('torch.'):
        raise ImportError('blocked torch')
    return real_import(name, *args, **kwargs)

builtins.__import__ = import_without_torch
from dnn_benchmarking.cli.parser import create_parser
from dnn_benchmarking.validation.providers.pytorch_provider import PyTorchReferenceProvider
assert create_parser() is not None
assert PyTorchReferenceProvider().is_available() is False
"""
    env = os.environ.copy()
    env.setdefault("DNN_BENCH_WORKSPACE", "/tmp/dnn-bench-test-cache")
    result = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
