# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Consumer-side regression: the imported Config template
(hipdnn_flatbuffers_sdkConfig_imported.cmake.in) wires the shared version
helper correctly. We don't try to run the full Config (it would require
real find_dependency() entries); instead we lift the helper-invocation
block out of the template, run it through CMake's @VAR@ substitution
ourselves, and exercise it with simulated flatbuffers state so that:

  * the include() path resolves to the helper that is shipped alongside
    the installed Config, and
  * an ABI-incompatible flatbuffers (mismatched version) is rejected at
    consumer find_package() time.

If the template ever drops the `include(... hipdnn_flatbuffers_version_check
.cmake)` line or the call to `hipdnn_check_flatbuffers_version`, this test
fails fast.
"""

import os
import re
import shutil
import subprocess

import pytest


_HIPDNN_DIR = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
_HELPER = os.path.join(
    _HIPDNN_DIR,
    "flatbuffers_sdk",
    "cmake",
    "hipdnn_flatbuffers_version_check.cmake",
)
_IMPORTED_TEMPLATE = os.path.join(
    _HIPDNN_DIR,
    "flatbuffers_sdk",
    "cmake",
    "hipdnn_flatbuffers_sdkConfig_imported.cmake.in",
)


def _cmake_or_skip():
    cmake = shutil.which("cmake")
    if cmake is None:
        pytest.skip("cmake not on PATH")
    return cmake


def _extract_guard_block(template_text):
    """Pull the include(...) + hipdnn_check_flatbuffers_version(...) calls
    from the template. Both must be present; otherwise the consumer-side
    guard isn't wired and the test fails."""
    include_match = re.search(
        r'include\(\s*"\$\{CMAKE_CURRENT_LIST_DIR\}/hipdnn_flatbuffers_version_check\.cmake"\s*\)',
        template_text,
    )
    call_match = re.search(
        r"hipdnn_check_flatbuffers_version\s*\([^)]*\)",
        template_text,
        re.DOTALL,
    )
    assert (
        include_match
    ), "Imported Config template must include hipdnn_flatbuffers_version_check.cmake"
    assert (
        call_match
    ), "Imported Config template must call hipdnn_check_flatbuffers_version()"
    # Use the literal helper path so this synthetic driver doesn't depend
    # on installing the package first.
    rewritten_include = f'include("{_HELPER}")'
    return rewritten_include + "\n" + call_match.group(0)


def _substitute_baked_version(snippet, baked_version):
    """Mimic configure_package_config_file's @VAR@ substitution."""
    return snippet.replace("@HIPDNN_FLATBUFFERS_VERSION@", baked_version)


def _run(tmp_path, snippet):
    cmake = _cmake_or_skip()
    script = tmp_path / "consumer_driver.cmake"
    script.write_text(snippet)
    proc = subprocess.run(
        [cmake, "-P", str(script)],
        capture_output=True,
        text=True,
    )
    return proc.returncode, proc.stdout + proc.stderr


@pytest.fixture
def template_text():
    with open(_IMPORTED_TEMPLATE, encoding="utf-8") as f:
        return f.read()


def test_imported_template_invokes_shared_helper(template_text):
    """Sanity-check the wiring; failure here is the most likely regression
    if someone edits the template without re-running the shared check."""
    _extract_guard_block(template_text)


def test_consumer_with_matching_version_passes(template_text, tmp_path):
    guard = _extract_guard_block(template_text)
    snippet = _substitute_baked_version(guard, "25.9.23")
    driver = (
        'set(flatbuffers_VERSION "25.9.23")\n' "set(flatbuffers_FOUND TRUE)\n" + snippet
    )
    rc, output = _run(tmp_path, driver)
    assert rc == 0, output
    assert "FATAL" not in output
    assert "CMake Warning" not in output


def test_consumer_with_mismatched_version_is_rejected(template_text, tmp_path):
    """The Config must reject ABI-incompatible FlatBuffers at find_package
    time so the consumer doesn't link against headers baked for a
    different on-the-wire schema."""
    guard = _extract_guard_block(template_text)
    snippet = _substitute_baked_version(guard, "25.9.23")
    driver = (
        'set(flatbuffers_VERSION "24.12.23")\n'
        "set(flatbuffers_FOUND TRUE)\n"
        'set(flatbuffers_DIR "/fake/consumer/path")\n' + snippet
    )
    rc, output = _run(tmp_path, driver)
    assert rc != 0, output
    assert "FlatBuffers version mismatch" in output
    assert "24.12.23" in output
    assert "25.9.23" in output


def test_consumer_with_unset_version_warns(template_text, tmp_path):
    guard = _extract_guard_block(template_text)
    snippet = _substitute_baked_version(guard, "25.9.23")
    driver = "unset(flatbuffers_VERSION)\n" "set(flatbuffers_FOUND TRUE)\n" + snippet
    rc, output = _run(tmp_path, driver)
    assert rc == 0, output
    assert "CMake Warning" in output
    assert "could not validate FlatBuffers version" in output
