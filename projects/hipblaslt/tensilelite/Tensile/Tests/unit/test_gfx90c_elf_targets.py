# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

import shutil
import subprocess

import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    ("target", "expected_feature"),
    [
        ("gfx90c", "EF_AMDGPU_FEATURE_XNACK_ANY_V4"),
        ("gfx90c:xnack+", "EF_AMDGPU_FEATURE_XNACK_ON_V4"),
        ("gfx90c:xnack-", "EF_AMDGPU_FEATURE_XNACK_OFF_V4"),
    ],
)
def test_assembled_gfx90c_elf_target_features(tmp_path, target, expected_feature):
    """Inspect the artifact, not merely the assembler command line."""
    assembler = shutil.which("amdclang++") or shutil.which("clang++")
    readobj = shutil.which("llvm-readobj")
    if assembler is None or readobj is None:
        pytest.skip("an AMDGPU-capable clang++ and llvm-readobj are required")

    source = tmp_path / "target.s"
    obj = tmp_path / "target.o"
    source.write_text(
        ".text\n"
        f'.amdgcn_target "amdgcn-amd-amdhsa--{target}"\n'
        ".globl gfx90c_target_test\n"
        ".p2align 8\n"
        ".type gfx90c_target_test,@function\n"
        "gfx90c_target_test:\n"
        "  s_endpgm\n",
        encoding="utf-8",
    )
    assembled = subprocess.run(
        [
            assembler,
            "--target=amdgcn-amd-amdhsa",
            f"-mcpu={target}",
            "-mcode-object-version=4",
            "-x",
            "assembler",
            "-c",
            str(source),
            "-o",
            str(obj),
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    if assembled.returncode != 0:
        pytest.skip(
            f"{assembler} cannot assemble gfx90c target IDs: "
            f"{assembled.stderr.strip()}"
        )

    inspected = subprocess.run(
        [readobj, "--file-headers", str(obj)],
        text=True,
        capture_output=True,
        check=True,
    ).stdout
    assert "EF_AMDGPU_MACH_AMDGCN_GFX90C" in inspected
    # The feature constant's V4 suffix verifies the code-object encoding while
    # avoiding llvm-readobj ABI-field formatting differences across releases.
    assert expected_feature in inspected
