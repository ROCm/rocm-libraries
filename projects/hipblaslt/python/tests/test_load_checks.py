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
    # The build-time library comes before ROCM_PATH and ROCM_HOME: they usually name a base ROCm,
    # not the hipBLASLt the bindings were built against.
    assert [x.source for x in c] == [
        L.LIBRARY_ENV,
        "rocm-sdk",
        "build",
        "ROCM_PATH",
        "ROCM_HOME",
    ]
    assert [x.path for x in c] == [
        "/o/libhipblaslt.so.1",
        "/sdk/libhipblaslt.so.1",
        _J("/build/lib", "libhipblaslt.so.1"),
        _J("/rp", "lib", "libhipblaslt.so.1"),
        _J("/rh", "lib", "libhipblaslt.so.1"),
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


def test_an_exported_rocm_path_doesnt_beat_the_build_time_library():
    c = L.candidate_libraries({"ROCM_PATH": "/opt/rocm"}, "linux", [], "/install/lib")
    assert L.choose_library(c, lambda p: True).source == "build"
    assert L.choose_library(c, lambda p: p.startswith("/opt")).source == "ROCM_PATH"


# --- import didn't get the build-time library ---------------------------------------------------


def test_the_build_time_library_is_quiet():
    b = "/install/lib/libhipblaslt.so.1"
    assert L.not_the_build_warning(b, "build", b, True, "linux") is None


def test_the_override_is_quiet():
    # The override warns separately, and is what the caller asked for.
    msg = L.not_the_build_warning(
        "/o/libhipblaslt.so.1",
        L.LIBRARY_ENV,
        "/install/lib/libhipblaslt.so.1",
        True,
        "linux",
    )
    assert msg is None


@pytest.mark.parametrize(
    "source,why",
    [
        ("rocm-sdk", "rocm-sdk wheels"),
        ("already loaded", "already loaded"),
        ("ROCM_PATH", "through ROCM_PATH"),
    ],
)
def test_another_library_warns_and_names_both(source, why):
    msg = L.not_the_build_warning(
        "/other/libhipblaslt.so.1",
        source,
        "/install/lib/libhipblaslt.so.1",
        True,
        "linux",
    )
    assert msg and why in msg and "/other/" in msg and "/install/lib/" in msg
    assert L.LIBRARY_ENV in msg


def test_a_missing_build_time_library_says_so():
    msg = L.not_the_build_warning(
        "/opt/rocm/lib/libhipblaslt.so.1",
        "ROCM_PATH",
        "/gone/libhipblaslt.so.1",
        False,
        "linux",
    )
    assert msg and "no longer exists" in msg


def test_the_same_file_under_another_name_is_quiet(tmp_path):
    # The rocm-sdk wheels hard-link one libhipblaslt into two packages.
    name = L.library_file_name(sys.platform)
    (tmp_path / "devel").mkdir()
    (tmp_path / "libraries").mkdir()
    built = tmp_path / "devel" / name
    built.write_bytes(b"library")
    os.link(built, tmp_path / "libraries" / name)
    msg = L.not_the_build_warning(
        str(tmp_path / "libraries" / name), "rocm-sdk", str(built), True, sys.platform
    )
    assert msg is None


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


def test_hardlinked_library_names_are_the_same_loaded_file(tmp_path):
    name = L.library_file_name(sys.platform)
    loaded = tmp_path / name
    loaded.write_bytes(b"library")
    alias_dir = tmp_path / "alias"
    alias_dir.mkdir()
    alias = alias_dir / name
    os.link(loaded, alias)
    paths = [str(loaded), str(alias)]

    L.check_override_can_take_effect(str(alias), [str(loaded)], sys.platform)
    L.check_single_hipblaslt(str(alias), paths, sys.platform)
    assert L.loaded_hipblaslt(paths, sys.platform) == [str(loaded)]


def test_windows_module_names_are_matched_case_insensitively():
    loaded = [r"C:\torch\lib\LIBHIPBLASLT.DLL"]
    with pytest.raises(L.HipblasLtLoadError):
        L.check_override_can_take_effect(r"C:\mine\libhipblaslt.dll", loaded, "win32")


# --- check 2: the override is active ------------------------------------------------------------


def test_override_warning_names_the_library_and_the_framework_risk():
    msg = L.override_warning("/o/libhipblaslt.so.1", ["/elsewhere/libamdhip64.so.7"])
    assert "/o/libhipblaslt.so.1" in msg
    assert "torch" in msg
    # The runtime that actually loaded, which needn't be next to the override.
    assert "/elsewhere/libamdhip64.so.7" in msg


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


# --- which already-loaded library _core can bind to ------------------------------------------------


def test_versioned_file_names_match_the_soname():
    assert L.matches_lib_name(
        "/r/lib/libhipblaslt.so.1.5.0", "libhipblaslt.so.1", "linux"
    )
    assert L.matches_lib_name("/r/lib/libhipblaslt.so.1", "libhipblaslt.so.1", "linux")
    assert not L.matches_lib_name(
        "/r/lib/libhipblaslt.so.0", "libhipblaslt.so.1", "linux"
    )
    assert not L.matches_lib_name(
        "/r/lib/libhipblaslt.so.10", "libhipblaslt.so.1", "linux"
    )
    assert L.matches_lib_name(r"C:\r\LIBHIPBLASLT.DLL", "libhipblaslt.dll", "win32")


def test_an_already_loaded_copy_with_the_same_soname_is_used():
    loaded = ["/torch/lib/libhipblaslt.so.1"]
    assert (
        L.compatible_already_loaded(loaded, "linux", "libhipblaslt.so.1") == loaded[0]
    )
    assert (
        L.compatible_already_loaded(
            ["/usr/lib/libc.so.6"], "linux", "libhipblaslt.so.1"
        )
        is None
    )


def test_an_already_loaded_copy_with_another_soname_raises():
    # _core's NEEDED libhipblaslt.so.1 wouldn't bind to it, so a second hipBLASLt would load.
    with pytest.raises(L.HipblasLtLoadError, match="need libhipblaslt.so.1"):
        L.compatible_already_loaded(
            ["/old/libhipblaslt.so.0"], "linux", "libhipblaslt.so.1"
        )


def test_only_the_selected_hipblaslt_may_be_loaded_after_import():
    L.check_single_hipblaslt("/a/libhipblaslt.so.1", ["/a/libhipblaslt.so.1"], "linux")
    with pytest.raises(L.HipblasLtLoadError, match="/b/libhipblaslt.so.1"):
        L.check_single_hipblaslt(
            "/a/libhipblaslt.so.1",
            ["/a/libhipblaslt.so.1", "/b/libhipblaslt.so.1"],
            "linux",
        )
    with pytest.raises(L.HipblasLtLoadError, match="nowhere"):
        L.check_single_hipblaslt("/a/libhipblaslt.so.1", [], "linux")


# --- the build-time HIP runtime -----------------------------------------------------------------


# --- dependencies loaded from hipBLASLt's own directory -----------------------------------------

_NEEDED = {
    "/r/lib/libhipblaslt.so.1": [
        "libtensilelite-host.so.1",
        "librocroller.so.1",
        "libc.so.6",
    ],
    "/r/lib/libtensilelite-host.so.1": ["liborigami.so.1", "libamdhip64.so.7"],
    "/r/lib/librocroller.so.1": ["libamdhip64.so.7", "libhipblaslt.so.1"],
    "/r/lib/liborigami.so.1": [],
    "/r/lib/libamdhip64.so.7": [],
}


def _plan(loaded=()):
    return L.dependencies_to_preload(
        "/r/lib/libhipblaslt.so.1",
        lambda p: _NEEDED.get(p, []),
        _NEEDED.__contains__,
        loaded,
        "linux",
    )


def test_dependencies_come_from_hipblaslts_directory_before_their_dependents():
    order = _plan()
    assert sorted(order) == sorted(set(_NEEDED) - {"/r/lib/libhipblaslt.so.1"})
    for lib, needs in _NEEDED.items():
        for name in needs:
            dep = "/r/lib/" + name
            if lib in order and dep in order:
                assert order.index(dep) < order.index(lib), (dep, lib)


def test_an_already_loaded_dependency_isnt_loaded_again():
    order = _plan(loaded=["/opt/rocm/lib/libamdhip64.so.7.2.0"])
    assert "/r/lib/libamdhip64.so.7" not in order
    assert "/r/lib/librocroller.so.1" in order


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads an ELF file")
def test_the_extensions_dynamic_section_names_hipblaslt():
    from hipblaslt import _core

    dynamic = L.elf_dynamic(_core.__file__)
    assert any(n.startswith("libhipblaslt.so") for n in dynamic.needed)
    assert isinstance(dynamic.rpath_governs, bool)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads an ELF file")
def test_the_loaded_library_reports_its_soname():
    import hipblaslt

    assert L.elf_dynamic(hipblaslt.library.path).soname.startswith("libhipblaslt.so.")


# --- the chosen file's SONAME, before it's loaded ------------------------------------------------


def test_a_matching_or_unknown_soname_passes():
    L.check_soname("/o/libhipblaslt.so.1", "libhipblaslt.so.1", "libhipblaslt.so.1")
    L.check_soname("/o/libhipblaslt.so.1", "", "libhipblaslt.so.1")
    L.check_soname("/o/libhipblaslt.so.1", "libhipblaslt.so.0", "")


def test_a_different_soname_raises_before_loading():
    with pytest.raises(
        L.HipblasLtLoadError, match="is libhipblaslt.so.0.*need libhipblaslt.so.1"
    ):
        L.check_soname("/o/libhipblaslt.so.1", "libhipblaslt.so.0", "libhipblaslt.so.1")


# --- one HIP runtime -----------------------------------------------------------------------------


def test_two_hip_runtimes_raise():
    loaded = [
        "/old/lib/libamdhip64.so.5",
        "/new/lib/libamdhip64.so.7.2.0",
        "/x/libc.so.6",
    ]
    assert L.loaded_hip_runtimes(loaded, "linux") == loaded[:2]
    with pytest.raises(L.HipblasLtLoadError, match="more than one HIP runtime"):
        L.check_single_hip_runtime(loaded, "linux")


def test_one_hip_runtime_passes_even_under_two_names(tmp_path):
    L.check_single_hip_runtime(["/r/lib/libamdhip64.so.7", "/r/lib/libc.so.6"], "linux")
    real = tmp_path / "libamdhip64.so.7"
    real.write_bytes(b"runtime")
    (tmp_path / "other").mkdir()
    os.link(real, tmp_path / "other" / "libamdhip64.so.7")
    L.check_single_hip_runtime(
        [str(real), str(tmp_path / "other" / "libamdhip64.so.7")], sys.platform
    )


def test_windows_hip_runtimes_are_amdhip64_dlls():
    loaded = [r"C:\r\bin\amdhip64_6.dll", r"C:\s\bin\AMDHIP64_7.DLL"]
    with pytest.raises(L.HipblasLtLoadError):
        L.check_single_hip_runtime(loaded, "win32")


def test_the_build_on_another_runtime_warns():
    msg = L.build_runtime_warning(
        ["/other/libamdhip64.so.7"], "build", "/rocm/lib/libamdhip64.so.7", "linux"
    )
    assert msg and "/other/" in msg and "/rocm/lib/" in msg


@pytest.mark.parametrize(
    "runtimes,source",
    [
        (["/rocm/lib/libamdhip64.so.7"], "build"),
        (["/other/libamdhip64.so.7"], "rocm-sdk"),
        ([], "build"),
    ],
    ids=["its-own-runtime", "not-the-build", "no-runtime-yet"],
)
def test_no_runtime_warning_otherwise(runtimes, source):
    assert (
        L.build_runtime_warning(runtimes, source, "/rocm/lib/libamdhip64.so.7", "linux")
        is None
    )


def test_a_file_that_isnt_elf_has_no_dynamic_section(tmp_path):
    f = tmp_path / "not-elf.so"
    f.write_bytes(b"#!/bin/sh\n" + b"\0" * 64)
    assert L.elf_dynamic(str(f)) is None
    assert L.elf_dynamic(str(tmp_path / "missing.so")) is None


def test_a_failed_preload_names_ld_library_path(tmp_path):
    with pytest.raises(L.HipblasLtLoadError, match="LD_LIBRARY_PATH is set"):
        L._preload(
            str(tmp_path / "libmissing.so.1"), {"LD_LIBRARY_PATH": "/other/rocm/lib"}
        )
    with pytest.raises(L.HipblasLtLoadError) as excinfo:
        L._preload(str(tmp_path / "libmissing.so.1"), {})
    assert "LD_LIBRARY_PATH" not in str(excinfo.value)


def test_a_build_time_library_gets_its_hip_runtime_first():
    build = L.Candidate("/install/lib/libhipblaslt.so.1", "build")
    runtime = "/custom/rocm/lib/libamdhip64.so.7"
    assert L.hip_runtime_to_preload(build, runtime, [], "linux") == runtime


@pytest.mark.parametrize(
    "candidate,loaded,platform",
    [
        (L.Candidate("/sdk/libhipblaslt.so.1", "rocm-sdk"), [], "linux"),
        (
            L.Candidate("/install/lib/libhipblaslt.so.1", "build"),
            ["/x/libamdhip64.so.7"],
            "linux",
        ),
        (L.Candidate(r"C:\b\libhipblaslt.dll", "build"), [], "win32"),
    ],
    ids=["not-build-time", "runtime-already-loaded", "windows"],
)
def test_no_hip_runtime_preload_otherwise(candidate, loaded, platform):
    assert (
        L.hip_runtime_to_preload(
            candidate, "/custom/libamdhip64.so.7", loaded, platform
        )
        is None
    )


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
    monkeypatch.delenv(L.LIBRARY_ENV, raising=False)
    L.check_loaded_version((100500, "565c4c74"))
    assert not [w for w in recwarn if issubclass(w.category, L.HipblasLtLoadWarning)]


def test_setting_the_override_after_import_doesnt_silence_the_warning(
    monkeypatch, recwarn
):
    # Setting it later (say, for a child process) doesn't change the library already loaded.
    monkeypatch.setattr(
        L, "_library", L.LoadInfo("/sdk/libhipblaslt.so.1", "rocm-sdk", True)
    )
    monkeypatch.setattr(L, "_built_version", (100500, "25f28906"))
    monkeypatch.setattr(L, "_version_checked", False)
    monkeypatch.setenv(L.LIBRARY_ENV, "/elsewhere/libhipblaslt.so.1")
    L.check_loaded_version((100500, "565c4c74"))
    assert [w for w in recwarn if issubclass(w.category, L.HipblasLtLoadWarning)]


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
