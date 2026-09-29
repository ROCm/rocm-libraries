"""The CMake wiring of a build without rocKE, driven by real sub-configures.

A synthetic consumer includes `HkpPackaging.cmake` and calls its functions, as
`tests/rocke/test_hkp_python_environment.py` does for the rocKE wheel lifecycle.
Here no rocKE wheel, comgr or private import directory exists, and the supplied
interpreter has no pip: a pack wired with `ENABLE_ROCKE OFF` must need none of
them.
"""

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from cmake_harness import PKG, SuppliedPython, consumer_preamble

pytestmark = pytest.mark.quick

# Stands in for hkp_pack.py: records how the pack step launched it.
CONSUMER = """
import json
import os
import sys
from pathlib import Path

argv = sys.argv[1:]
out_root = Path(argv[argv.index('--out-root') + 1])
out_root.mkdir(parents=True, exist_ok=True)
(out_root / 'invocation.json').write_text(json.dumps({
    'python': sys.executable,
    'argv': argv,
    'env': {key: os.environ.get(key) for key in
            ('ROCKE_BACKEND', 'ROCKE_CPP_STRICT', 'ROCKE_COMGR_LIB',
             'HKP_PACK_JOBS', 'PYTHONPATH')},
}), encoding='utf-8')
"""


class _Consumer(SuppliedPython):
    def __init__(self, root, *, cmake, make_program):
        self.cmake = cmake
        self.make_program = make_program
        self.source = root / "source with spaces"
        self.build_dir = root / "build with spaces"
        self.source.mkdir(parents=True)
        super().__init__(root, pip=False)
        (self.source / "consumer.py").write_text(CONSUMER, encoding="utf-8")
        (self.source / "authored").mkdir()
        (self.source / "kpack with spaces" / "rocm_kpack").mkdir(parents=True)

    def wire(self, rocke_keywords):
        """One root, wired with `rocke_keywords` spliced into the call."""
        self._write(
            """set(HKP_TOOL "${CMAKE_CURRENT_SOURCE_DIR}/consumer.py")
hkp_wire_pack_target(
    NAME off
    SOURCE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/authored"
    OUT_ROOT "${CMAKE_CURRENT_BINARY_DIR}/off"
    ARCHES gfx942 HIPCC unused
    ROCM_KPACK_DIR "${CMAKE_CURRENT_SOURCE_DIR}/kpack with spaces"
    """
            + rocke_keywords
            + ")\n"
        )

    def register_tests(self, enable_rocke):
        self._write(
            f"""enable_testing()
set(HIPKERNELPROVIDER_ENABLE_TESTS ON)
set(HIPKERNELPROVIDER_ENABLE_ROCKE {enable_rocke})
hkp_register_tests("" unused "")
"""
        )

    def _write(self, body):
        (self.source / "CMakeLists.txt").write_text(
            consumer_preamble("HkpPackWiring") + body, encoding="utf-8"
        )

    def configure(self, *, python=None, success=True):
        proc = self.run(
            self.cmake,
            "-S",
            self.source,
            "-B",
            self.build_dir,
            "-G",
            "Ninja",
            f"-DCMAKE_MAKE_PROGRAM={self.make_program}",
            f"-DPython3_EXECUTABLE={python or self.python}",
            success=success,
        )
        return " ".join((proc.stdout + proc.stderr).split())

    def build(self):
        self.run(self.cmake, "--build", self.build_dir, "--target", "hkp_packaging_off")

    def invocation(self):
        return json.loads(
            (self.build_dir / "off" / "invocation.json").read_text(encoding="utf-8")
        )

    def registered_commands(self):
        ctest = Path(self.cmake).with_name("ctest" + Path(self.cmake).suffix)
        proc = self.run(ctest, "--show-only=json-v1", "--test-dir", self.build_dir)
        return {t["name"]: t["command"] for t in json.loads(proc.stdout)["tests"]}


@pytest.fixture
def consumer(tmp_path, cmake, cmake_make_program):
    return _Consumer(tmp_path, cmake=cmake, make_program=cmake_make_program)


def test_a_root_without_rocke_packs_under_a_base_interpreter_without_pip(consumer):
    """The pack runs under the supplied interpreter with no rocKE environment, no
    private import directory on PYTHONPATH, and `--no-rocke` in place of the wheel
    stamp. pip is absent, so any rocKE wheel step would fail configure or build."""
    consumer.run(consumer.python, "-m", "pip", "--version", success=False)
    consumer.wire("ENABLE_ROCKE OFF PACK_JOBS 2")
    consumer.configure()
    consumer.build()

    record = consumer.invocation()
    assert Path(record["python"]) == consumer.python
    assert "--no-rocke" in record["argv"]
    assert "--rocke-wheel-stamp" not in record["argv"]
    # The harness strips PYTHONPATH from the build's environment, so any value
    # the pack sees is a prepend the command made.
    assert record["env"] == {
        "ROCKE_BACKEND": None,
        "ROCKE_CPP_STRICT": None,
        "ROCKE_COMGR_LIB": None,
        "HKP_PACK_JOBS": "2",
        "PYTHONPATH": None,
    }
    assert "path_list_prepend" not in (consumer.build_dir / "build.ninja").read_text(
        encoding="utf-8"
    )


@pytest.mark.parametrize(
    "rocke_keywords",
    [
        'ROCKE_INTERP "x"',
        'ROCKE_INTERP ""',
        "ROCKE_INTERP PACK_JOBS 2",
    ],
    ids=["valued", "empty-value", "no-value"],
)
def test_a_rocke_keyword_on_a_root_without_rocke_fails_configure(
    consumer, rocke_keywords
):
    """Each ROCKE_* keyword names a toolchain the build does not have, however it
    is spelled: with a value, with an empty one, or with none."""
    consumer.wire(f"ENABLE_ROCKE OFF {rocke_keywords}")
    diagnostic = consumer.configure(success=False)
    assert "disables rocKE but was wired with ROCKE_INTERP" in diagnostic


def test_a_root_wired_without_enable_rocke_fails_configure(consumer):
    """Neither mode is reachable by omission."""
    consumer.wire("PACK_JOBS 2")
    diagnostic = consumer.configure(success=False)
    assert "was wired without ENABLE_ROCKE" in diagnostic


@pytest.mark.parametrize("enable_rocke", ["ON", "OFF"])
def test_the_registered_suites_collect_tests_rocke_only_with_rocke(
    consumer, enable_rocke
):
    """Without rocKE, both pytest entries ignore tests/rocke/ and nothing else;
    with it, neither ignores anything. Configured under this interpreter, which
    can import pytest as hkp_register_tests requires."""
    consumer.register_tests(enable_rocke)
    consumer.configure(python=sys.executable)
    commands = consumer.registered_commands()
    assert set(commands) == {
        "hip-kernel-provider-hkp-pack-quick",
        "hip-kernel-provider-hkp-pack",
    }

    rocke_tests = os.path.normpath(PKG / "tests" / "rocke")
    for name, command in commands.items():
        ignored = [
            os.path.normpath(arg.removeprefix("--ignore="))
            for arg in command
            if arg.startswith("--ignore")
        ]
        assert ignored == ([rocke_tests] if enable_rocke == "OFF" else []), name


@pytest.mark.parametrize(("arch", "verdict"), [("gfx950", "TRUE"), ("gfx942", "FALSE")])
def test_the_rocke_only_probe_answers_under_a_base_interpreter_without_pip(
    consumer, rocke_fixture, arch, verdict
):
    """The probe that leaves a rocKE-only default root dormant runs the packer's
    selection under the pip-less interpreter, through a path with spaces. The
    fixture's rocKE kernel is scoped to gfx950: selected there, pruned for gfx942.
    A probe that could not run would answer FALSE for both."""
    shutil.copytree(rocke_fixture, consumer.source / "authored" / "rocKE" / "attention")
    consumer._write(
        f"""_hkp_root_selects_only_rocke(_only "${{CMAKE_CURRENT_SOURCE_DIR}}/authored" {arch})
message(STATUS "rocke-only=${{_only}}")
"""
    )
    assert f"rocke-only={verdict}" in consumer.configure()
