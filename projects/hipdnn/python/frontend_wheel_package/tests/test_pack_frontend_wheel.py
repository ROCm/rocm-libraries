# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import importlib.util
from pathlib import Path

import pytest


PACKER_PATH = Path(__file__).resolve().parents[1] / "pack_frontend_wheel.py"
PACKER_SPEC = importlib.util.spec_from_file_location("pack_frontend_wheel", PACKER_PATH)
assert PACKER_SPEC is not None
assert PACKER_SPEC.loader is not None
pack_frontend_wheel = importlib.util.module_from_spec(PACKER_SPEC)
PACKER_SPEC.loader.exec_module(pack_frontend_wheel)


def test_find_native_extension_rejects_multiple_preferred_matches(tmp_path):
    first = tmp_path / "hipdnn_frontend_python.abi3.so"
    second = tmp_path / "libhipdnn_frontend_python.pyd"
    first.write_bytes(b"")
    second.write_bytes(b"")

    with pytest.raises(SystemExit) as exc_info:
        pack_frontend_wheel.find_native_extension(tmp_path)

    message = str(exc_info.value)
    assert (
        "Multiple native extensions found; pass --extension to select one:" in message
    )
    assert str(first) in message
    assert str(second) in message
