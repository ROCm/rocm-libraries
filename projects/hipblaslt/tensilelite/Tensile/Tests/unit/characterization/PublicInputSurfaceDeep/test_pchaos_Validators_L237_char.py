################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

# Characterization of Tensile/Toolchain/Validators.py:237 [guard-return]
#   branch_id: 09380ac263b6fa5b5ce1fae3b659c59c5e2178a5
#   function: _validateExecutable
#   predicate: _exeExists(Path(file))   (== os.access(Path(file), os.X_OK))
#     true_branch  -> return file   (absolute path validated)
#     false_branch -> raise FileNotFoundError
#
# Classification: runtime-dependent. The predicate is a live filesystem probe
# (os.access X_OK); its truth value is determined entirely by external state
# (path existence + execute permission), not by pure boolean/integer logic.
# These tests pin ACTUAL behavior by constructing the filesystem state
# deterministically (CPU-only, no GPU, no network).

import os
import stat
from pathlib import Path

import pytest

from Tensile.Toolchain.Validators import _exeExists

pytestmark = pytest.mark.unit


def test_exeexists_true_executable_file(tmp_path):
    # exists AND has execute bit -> True (the true_branch of the guard)
    p = tmp_path / "an_exe"
    p.write_text("#!/bin/sh\n")
    p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    assert _exeExists(Path(p)) is True


def test_exeexists_false_nonexistent(tmp_path):
    # path does not exist -> False (false_branch -> FileNotFoundError)
    p = tmp_path / "nonexistent" / "path" / "to" / "compiler"
    assert os.path.exists(p) is False
    assert _exeExists(Path(p)) is False


def test_exeexists_false_exists_not_executable(tmp_path):
    # exists but NO execute bit -> False (false_branch -> FileNotFoundError)
    p = tmp_path / "not_executable_file"
    p.write_text("not executable\n")
    p.chmod(stat.S_IRUSR | stat.S_IWUSR)  # rw------- , no X
    assert os.path.exists(p) is True
    assert _exeExists(Path(p)) is False
