# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The lookup order and the four load checks, as pure functions.

Each check gets a case that must fire and a case that must stay quiet. No GPU and no second ROCm
needed.
"""

import os
import sys

import pytest

from hipblaslt import _loader as L

_J = os.path.join


# --- lookup order ------------------------------------------------------------------------------


def test_lookup_order_on_linux():
    env = {
        L.LIBRARY_ENV: "/o/libhipblaslt.so.1",
        "ROCM_PATH": "/rp",
        "ROCM_HOME": "/rh",
    }
    c = L.candidate_libraries(env, "linux", ["/sdk/libhipblaslt.so.1"], "/build/lib")
    assert [x.source for x in c] == [
        L.LIBRARY_ENV,
        "rocm-sdk",
        "ROCM_PATH",
        "ROCM_HOME",
        "build",
    ]
    assert [x.path for x in c] == [
        "/o/libhipblaslt.so.1",
        "/sdk/libhipblaslt.so.1",
        _J("/rp", "lib", "libhipblaslt.so.1"),
        _J("/rh", "lib", "libhipblaslt.so.1"),
        _J("/build/lib", "libhipblaslt.so.1"),
    ]


def test_windows_looks_in_bin_for_the_dll():
    c = L.candidate_libraries({"ROCM_PATH": "R"}, "win32", [], "")
    assert c == [L.Candidate(_J("R", "bin", "libhipblaslt.dll"), "ROCM_PATH")]


def test_first_existing_candidate_wins():
    c = L.candidate_libraries(
        {"ROCM_PATH": "/rp"}, "linux", ["/sdk/libhipblaslt.so.1"], "/b"
    )
    chosen = L.choose_library(c, lambda p: p.startswith("/rp"))
    assert chosen.source == "ROCM_PATH"


def test_missing_override_is_an_error_not_a_fallthrough():
    c = L.candidate_libraries(
        {L.LIBRARY_ENV: "/nope.so"}, "linux", ["/sdk/libhipblaslt.so.1"], ""
    )
    with pytest.raises(L.HipblasLtLoadError, match="doesn't exist"):
        L.choose_library(c, lambda p: p != "/nope.so")


def test_nothing_found_lists_every_place_tried():
    c = L.candidate_libraries({"ROCM_PATH": "/rp"}, "linux", [], "/b")
    with pytest.raises(L.HipblasLtLoadError) as e:
        L.choose_library(c, lambda p: False)
    assert "/rp" in str(e.value) and "/b" in str(e.value)


# --- check 1: the override can't take effect ----------------------------------------------------


def test_override_raises_when_another_hipblaslt_is_loaded():
    loaded = ["/torch/lib/libhipblaslt.so.1", "/torch/lib/libamdhip64.so.7"]
    with pytest.raises(
        L.HipblasLtLoadError, match="can't take effect.*/torch/lib/libhipblaslt"
    ):
        L.check_override_can_take_effect("/mine/libhipblaslt.so.1", loaded, "linux")


def test_override_is_fine_when_nothing_or_itself_is_loaded():
    L.check_override_can_take_effect(
        "/mine/libhipblaslt.so.1", ["/usr/lib/libc.so.6"], "linux"
    )
    L.check_override_can_take_effect(
        "/mine/libhipblaslt.so.1", ["/mine/libhipblaslt.so.1"], "linux"
    )


def test_windows_module_names_are_matched_case_insensitively():
    loaded = [r"C:\torch\lib\LIBHIPBLASLT.DLL"]
    with pytest.raises(L.HipblasLtLoadError):
        L.check_override_can_take_effect(r"C:\mine\libhipblaslt.dll", loaded, "win32")


# --- check 2: the override is active ------------------------------------------------------------


def test_override_warning_names_the_library_and_the_framework_risk():
    msg = L.override_warning("/o/libhipblaslt.so.1")
    assert "/o/libhipblaslt.so.1" in msg
    assert "torch" in msg


# --- check 3: companion libraries from another build --------------------------------------------


def test_companions_from_another_directory_raise():
    loaded = [
        "/new/lib/libhipblaslt.so.1",
        "/old/lib/libtensilelite-host.so.1",
        "/new/lib/liborigami.so.1",
    ]
    with pytest.raises(L.HipblasLtLoadError, match="/old/lib/libtensilelite-host"):
        L.check_companions("/new/lib/libhipblaslt.so.1", loaded, "linux")


def test_companions_next_to_hipblaslt_are_fine():
    loaded = [
        "/new/lib/libhipblaslt.so.1",
        "/new/lib/libtensilelite-host.so.1",
        "/new/lib/liborigami.so.1",
        "/new/lib/librocroller.so.1",
        "/elsewhere/libamdhip64.so.7",
    ]
    L.check_companions("/new/lib/libhipblaslt.so.1", loaded, "linux")


def test_windows_companions_have_no_rocroller():
    loaded = [
        r"C:\new\bin\libhipblaslt.dll",
        r"C:\new\bin\origami.dll",
        r"C:\x\rocroller.dll",
    ]
    L.check_companions(r"C:\new\bin\libhipblaslt.dll", loaded, "win32")
    with pytest.raises(L.HipblasLtLoadError):
        L.check_companions(
            r"C:\new\bin\libhipblaslt.dll", [r"C:\old\bin\origami.dll"], "win32"
        )


# --- check 4: a different build than the one the bindings were built against -------------------


def test_same_build_is_quiet():
    assert (
        L.version_mismatch_warning((100500, "25f28906"), (100500, "25f28906"), "/p")
        is None
    )


def test_no_device_means_no_check():
    assert L.version_mismatch_warning((100500, "25f28906"), None, "/p") is None


def test_same_version_different_revision_warns():
    # Two builds can share a version; only the revision tells them apart.
    msg = L.version_mismatch_warning((100500, "25f28906"), (100500, "565c4c74"), "/p")
    assert (
        msg and "1.5.0 (565c4c74)" in msg and "1.5.0 (25f28906)" in msg and "/p" in msg
    )


def test_different_version_warns():
    msg = L.version_mismatch_warning((100500, "a"), (100401, "a"), "/p")
    assert msg and "1.4.1" in msg


def test_the_loaded_version_is_checked_once(monkeypatch, recwarn):
    monkeypatch.setattr(
        L, "_library", L.LoadInfo("/p/libhipblaslt.so.1", "rocm-sdk", True)
    )
    monkeypatch.setattr(L, "_built_version", (100500, "25f28906"))
    monkeypatch.setattr(L, "_version_checked", False)
    monkeypatch.delenv(L.LIBRARY_ENV, raising=False)
    L.check_loaded_version((100500, "565c4c74"))
    L.check_loaded_version((100500, "565c4c74"))
    found = [w for w in recwarn if issubclass(w.category, L.HipblasLtLoadWarning)]
    assert len(found) == 1 and "565c4c74" in str(found[0].message)


def test_no_version_warning_with_an_override(monkeypatch, recwarn):
    # The override already warned at import; a different build is what the caller asked for.
    monkeypatch.setattr(
        L, "_library", L.LoadInfo("/o/libhipblaslt.so.1", L.LIBRARY_ENV, True)
    )
    monkeypatch.setattr(L, "_built_version", (100500, "25f28906"))
    monkeypatch.setattr(L, "_version_checked", False)
    monkeypatch.setenv(L.LIBRARY_ENV, "/o/libhipblaslt.so.1")
    L.check_loaded_version((100500, "565c4c74"))
    assert not [w for w in recwarn if issubclass(w.category, L.HipblasLtLoadWarning)]


def test_import_does_not_check_the_loaded_version():
    # The check needs a handle, and creating one at import allocates GPU memory (and exits the
    # process if that fails), so import leaves it for the first Handle(). A fresh process, since
    # other tests here create handles.
    import subprocess

    code = "import warnings; warnings.simplefilter('ignore'); import hipblaslt; print(hipblaslt._loader._version_checked)"
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=300
    )
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().splitlines()[-1] == "False"


# --- the loaded-library scan --------------------------------------------------------------------


def test_loaded_hipblaslt_ignores_lookalikes_and_duplicates():
    loaded = [
        "/a/libhipblaslt.so.1",
        "/a/libhipblaslt.so.1",
        "/a/libhipblaslt-ext-helper.so",
        "/a/libhipblas.so.3",
    ]
    assert L.loaded_hipblaslt(loaded, "linux") == ["/a/libhipblaslt.so.1"]


def test_this_process_reports_its_libhipblaslt():
    import hipblaslt

    found = L.loaded_hipblaslt(L.loaded_libraries(), sys.platform)
    assert len(found) == 1
    assert os.path.realpath(found[0]) == os.path.realpath(hipblaslt.library.path)
