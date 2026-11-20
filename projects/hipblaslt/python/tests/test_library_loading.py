# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Which libhipblaslt a fresh `import hipblaslt` really loads, read from /proc/self/maps (Linux).

Each case imports in a subprocess with a controlled environment, then checks the process has one
libhipblaslt and one HIP runtime, and that the libhipblaslt is the one the lookup chose.

Two extra cases run only when a second ROCm is available, because CI has one:
- HIPBLASLT_PY_TEST_ALT_LIBRARY: a libhipblaslt.so.1 from a different build, used as the override;
- HIPBLASLT_PY_TEST_OTHER_ROCM_LIB: another ROCm's lib directory, put on LD_LIBRARY_PATH.
"""

import json
import os
import subprocess
import sys
import textwrap

import pytest

import hipblaslt

pytestmark = pytest.mark.skipif(
    not sys.platform.startswith("linux"), reason="reads /proc/self/maps"
)

_CHILD = textwrap.dedent(
    """
    import json, os, warnings
    warnings.simplefilter("ignore")
    import hipblaslt
    from hipblaslt import _loader
    maps = _loader.loaded_libraries()
    print(json.dumps({
        "path": hipblaslt.library.path,
        "source": hipblaslt.library.source,
        "hipblaslt": [os.path.realpath(p) for p in _loader.loaded_hipblaslt(maps, "linux")],
        "amdhip": sorted({os.path.realpath(p) for p in maps
                          if os.path.basename(p).startswith("libamdhip64.so")}),
    }))
    """
)

_DROP = (
    "LD_LIBRARY_PATH",
    "ROCM_PATH",
    "ROCM_HOME",
    "HIPBLASLT_PY_LIBRARY",
    "LD_PRELOAD",
)


def _import_in_child(**env_overrides: str) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in _DROP}
    env.update(env_overrides)
    out = subprocess.run(
        [sys.executable, "-c", _CHILD],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def _import_in_child_without_rocm_sdk() -> dict:
    env = {k: v for k, v in os.environ.items() if k not in _DROP}
    code = "import sys; sys.modules['rocm_sdk'] = None\n" + _CHILD
    out = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-4000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def _assert_one_of_each(r: dict) -> None:
    assert r["hipblaslt"] == [os.path.realpath(r["path"])], r
    assert len(r["amdhip"]) == 1, f"two HIP runtimes in one process: {r['amdhip']}"


def _default_choice() -> dict:
    return _import_in_child()


def test_without_ld_library_path():
    r = _default_choice()
    _assert_one_of_each(r)
    assert r["source"] in {"rocm-sdk", "build"}, r


def test_with_only_ld_library_path():
    expected = _default_choice()
    r = _import_in_child(LD_LIBRARY_PATH=os.path.dirname(expected["path"]))
    _assert_one_of_each(r)
    assert r["hipblaslt"] == expected["hipblaslt"]


def test_build_time_fallback():
    # Without rocm_sdk or ROCM_PATH, the build-time library loads, with its HIP runtime first.
    r = _import_in_child_without_rocm_sdk()
    _assert_one_of_each(r)
    assert r["source"] == "build", r


def test_override_wins():
    expected = _default_choice()
    r = _import_in_child(HIPBLASLT_PY_LIBRARY=expected["path"])
    _assert_one_of_each(r)
    assert r["source"] == "HIPBLASLT_PY_LIBRARY"
    assert r["hipblaslt"] == expected["hipblaslt"]


@pytest.mark.skipif(
    not os.environ.get("HIPBLASLT_PY_TEST_ALT_LIBRARY"),
    reason="needs a second hipBLASLt build",
)
def test_override_to_another_build():
    alt = os.environ["HIPBLASLT_PY_TEST_ALT_LIBRARY"]
    r = _import_in_child(HIPBLASLT_PY_LIBRARY=alt)
    _assert_one_of_each(r)
    assert r["hipblaslt"] == [os.path.realpath(alt)]


@pytest.mark.skipif(
    not os.environ.get("HIPBLASLT_PY_TEST_OTHER_ROCM_LIB"),
    reason="needs a second ROCm install",
)
def test_another_rocm_on_ld_library_path_doesnt_leak_in():
    expected = _default_choice()
    r = _import_in_child(LD_LIBRARY_PATH=os.environ["HIPBLASLT_PY_TEST_OTHER_ROCM_LIB"])
    _assert_one_of_each(r)
    assert r["hipblaslt"] == expected["hipblaslt"]


def test_this_process_matches_the_lookup():
    # The test session itself was imported the same way.
    found = [
        os.path.realpath(p)
        for p in hipblaslt._loader.loaded_hipblaslt(
            hipblaslt._loader.loaded_libraries(), "linux"
        )
    ]
    assert found == [os.path.realpath(hipblaslt.library.path)]
