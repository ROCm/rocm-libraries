# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""embed_kernel_sources() on a target nothing registered kernels on, configured for real.

The plugin target calls embed_kernel_sources() unconditionally, but only the kernel
ingestor's HIP MLOps engines register kernels on it. With HIPDNN_ENABLE_KERNEL_INGESTOR OFF
the target holds no KERNELEMBEDDING_KERNEL_FILES, and the module once read the property's
-NOTFOUND value as a file name and failed the configure. No CI configuration builds without
the ingestor, so a synthetic NONE-language consumer here includes the production module and
its templates and configures an empty target beside one that embeds a kernel. Nothing is
compiled.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

pytestmark = pytest.mark.quick

HKP_SRC = Path(__file__).resolve().parents[2] / "src"
MODULE = HKP_SRC / "cmake" / "KernelEmbedding.cmake"
TEMPLATES = HKP_SRC / "cmake" / "templates"

OUTPUTS = (
    "kernel_sources.cpp",
    "kernel_sources.hpp",
    "kernel_includes.cpp",
    "kernel_includes.hpp",
)

CONSUMER = """cmake_minimum_required(VERSION 3.25.2)
project(KernelEmbeddingConsumer NONE)
include("{module}")

add_custom_target(no_kernels)
add_custom_target(with_kernels)
add_kernels_for_embedding(TARGET with_kernels FILES kernels/Embedded.cpp
                          KEYS kernels/Embedded.cpp)

foreach(_target IN ITEMS no_kernels with_kernels)
    set(_out "${{CMAKE_BINARY_DIR}}/${{_target}}")
    embed_kernel_sources(TARGET ${{_target}}
                         OUTPUT_SRCS_CPP "${{_out}}/kernel_sources.cpp"
                         OUTPUT_SRCS_HPP "${{_out}}/kernel_sources.hpp"
                         OUTPUT_INCS_CPP "${{_out}}/kernel_includes.cpp"
                         OUTPUT_INCS_HPP "${{_out}}/kernel_includes.hpp")
    get_target_property(_manifest ${{_target}} KERNELEMBEDDING_KEY_MANIFEST)
    file(WRITE "${{_out}}/manifest_path.txt" "${{_manifest}}")
endforeach()

get_property(_depends DIRECTORY PROPERTY CMAKE_CONFIGURE_DEPENDS)
file(WRITE "${{CMAKE_BINARY_DIR}}/configure_depends.txt" "${{_depends}}")
"""


@pytest.fixture
def configured(tmp_path, cmake, cmake_make_program):
    """Configure the consumer once and return its build directory."""
    source = tmp_path / "source"
    build = tmp_path / "build"
    # configure_file() reads the templates from ${PROJECT_SOURCE_DIR}/src/cmake/templates.
    shutil.copytree(TEMPLATES, source / "src" / "cmake" / "templates")
    (source / "kernels").mkdir()
    (source / "kernels" / "Embedded.cpp").write_text(
        'extern "C" __global__ void Embedded() {}\n', encoding="utf-8"
    )
    (source / "CMakeLists.txt").write_text(
        CONSUMER.format(module=MODULE.as_posix()), encoding="utf-8"
    )
    proc = subprocess.run(
        [
            cmake,
            "-S",
            str(source),
            "-B",
            str(build),
            "-G",
            "Ninja",
            f"-DCMAKE_MAKE_PROGRAM={cmake_make_program}",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return build


def _read(path):
    return path.read_text(encoding="utf-8")


def test_a_target_with_no_registered_kernels_gets_empty_tables(configured):
    out = configured / "no_kernels"
    for name in OUTPUTS:
        assert (out / name).is_file(), name
    # A map entry is spelled {"<key>", ...}; an empty table has none.
    assert '{"' not in _read(out / "kernel_sources.cpp")
    assert '{"' not in _read(out / "kernel_includes.cpp")
    assert "NOTFOUND" not in "".join(_read(out / name) for name in OUTPUTS)

    manifest = Path(_read(out / "manifest_path.txt"))
    assert manifest == configured / "no_kernels_kernel_keys.txt"
    assert _read(manifest) == ""


def test_a_registered_kernel_is_embedded_beside_the_empty_target(configured):
    """The control: the consumer drives the real module, so a kernel registered on the
    other target lands in its tables, its manifest and the configure dependencies."""
    out = configured / "with_kernels"
    assert '{"kernels/Embedded.cpp", ' in _read(out / "kernel_sources.cpp")
    assert "void Embedded()" in _read(out / "kernel_sources.cpp")

    manifest = Path(_read(out / "manifest_path.txt"))
    assert _read(manifest).startswith("kernels/Embedded.cpp\t")

    depends = _read(configured / "configure_depends.txt").split(";")
    assert [Path(d).name for d in depends] == ["Embedded.cpp"]
