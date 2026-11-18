# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Windows DLL-directory selection (Review A, Decision 5), tested as a pure function.

No real directories and no os.add_dll_directory, so it runs on Linux too, like rocISA's
test_dll_dirs.py.
"""

import os

from hipblaslt._loader import windows_dirs_to_register, windows_dll_dirs

_J = os.path.join


def test_library_dir_first_then_wheel_runtime_then_hip_dirs():
    lib = _J("sp", "_rocm_sdk_libraries", "bin", "libhipblaslt.dll")
    dirs = windows_dll_dirs(lib, {}, [_J("sp", "_rocm_sdk_core", "bin")], "")
    assert dirs == [
        _J("sp", "_rocm_sdk_libraries", "bin"),
        _J("sp", "_rocm_sdk_core", "bin"),
    ]


def test_override_gets_its_own_hip_runtime_dir():
    # Spike 3: an override's HIP runtime isn't next to libhipblaslt.dll, and it isn't the one
    # rocm_sdk reports for the active environment.
    lib = _J("old", "_rocm_sdk_libraries", "bin", "libhipblaslt.dll")
    dirs = windows_dll_dirs(lib, {}, [_J("new", "_rocm_sdk_core", "bin")], "")
    assert dirs[:2] == [
        _J("old", "_rocm_sdk_libraries", "bin"),
        _J("old", "_rocm_sdk_core", "bin"),
    ]
    assert dirs.index(_J("old", "_rocm_sdk_core", "bin")) < dirs.index(
        _J("new", "_rocm_sdk_core", "bin")
    )


def test_environment_roots_then_build_time_bin():
    lib = _J("rocm", "bin", "libhipblaslt.dll")
    env = {"ROCM_PATH": "rp", "HIP_PATH": "hp", "ROCM_HOME": "rh"}
    dirs = windows_dll_dirs(lib, env, [], _J("build", "bin"))
    assert dirs == [
        _J("rocm", "bin"),
        _J("rp", "bin"),
        _J("hp", "bin"),
        _J("rh", "bin"),
        _J("build", "bin"),
    ]


def test_no_wheel_sibling_outside_the_wheel_layout():
    dirs = windows_dll_dirs(_J("opt", "rocm", "bin", "libhipblaslt.dll"), {}, [], "")
    assert dirs == [_J("opt", "rocm", "bin")]


def test_duplicates_keep_their_first_position():
    lib = _J("rocm", "bin", "libhipblaslt.dll")
    dirs = windows_dll_dirs(
        lib, {"ROCM_PATH": "rocm"}, [_J("rocm", "bin")], _J("rocm", "bin")
    )
    assert dirs.count(_J("rocm", "bin")) == 1
    assert dirs[0] == _J("rocm", "bin")


def test_empty_entries_are_skipped():
    lib = _J("rocm", "bin", "libhipblaslt.dll")
    assert "" not in windows_dll_dirs(lib, {"ROCM_PATH": ""}, [""], "")


# --- which candidates actually get registered --------------------------------------------------


def test_only_the_first_hip_runtime_is_registered():
    # The bug this guards against: an override whose own runtime directory was registered, but
    # the active environment's runtime (also registered) loaded instead.
    cands = ["old_lib", "old_core", "new_core", "build_bin"]
    has = {"old_core", "new_core", "build_bin"}
    assert windows_dirs_to_register(cands, lambda d: d in has) == [
        "old_lib",
        "old_core",
    ]


def test_runtime_next_to_the_library_registers_one_directory():
    assert windows_dirs_to_register(["rocm_bin", "other"], lambda d: True) == [
        "rocm_bin"
    ]


def test_no_runtime_anywhere_registers_every_candidate():
    assert windows_dirs_to_register(["a", "b"], lambda d: False) == ["a", "b"]


def test_no_candidates_registers_nothing():
    assert windows_dirs_to_register([], lambda d: True) == []
