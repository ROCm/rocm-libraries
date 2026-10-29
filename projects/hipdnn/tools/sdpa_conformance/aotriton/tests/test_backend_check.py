# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""The plugin's check that the hipDNN backend mapped into the process is --hipdnn-prefix's."""

import os

from hipdnn_aotriton_plugin import _mapped_backends, backend_problem


def test_selected_backend_passes(tmp_path):
    lib = tmp_path / "sdk" / "lib" / "libhipdnn_backend.so"
    assert backend_problem([str(lib)], str(tmp_path / "sdk")) is None


def test_backend_outside_rocm_path_fails(tmp_path):
    devel = tmp_path / "devel" / "lib" / "libhipdnn_backend.so"
    problem = backend_problem([str(devel)], str(tmp_path / "sdk"))
    assert problem and "not under" in problem


def test_sibling_prefix_is_not_a_match(tmp_path):
    # "sdk-devel" starts with "sdk" as a string but is a different prefix.
    lib = tmp_path / "sdk-devel" / "lib" / "libhipdnn_backend.so"
    assert backend_problem([str(lib)], str(tmp_path / "sdk")) is not None


def test_two_mapped_backends_fail(tmp_path):
    a = tmp_path / "sdk" / "lib" / "libhipdnn_backend.so"
    b = tmp_path / "devel" / "lib" / "libhipdnn_backend.so"
    problem = backend_problem([str(a), str(b)], str(tmp_path / "sdk"))
    assert problem and "found 2" in problem


def test_symlinked_prefix_resolves(tmp_path):
    real = tmp_path / "real"
    (real / "lib").mkdir(parents=True)
    lib = real / "lib" / "libhipdnn_backend.so"
    lib.touch()
    link = tmp_path / "link"
    os.symlink(real, link)
    assert backend_problem([str(lib)], str(link)) is None


def test_mapped_backends_reads_maps(tmp_path):
    maps = tmp_path / "maps"
    maps.write_text(
        "7f00-7f01 r--p 00000000 08:01 1 /opt/sdk/lib/libhipdnn_backend.so\n"
        "7f01-7f02 r-xp 00001000 08:01 1 /opt/sdk/lib/libhipdnn_backend.so\n"
        "7f02-7f03 r--p 00000000 08:01 2 /opt/sdk/lib/libamdhip64.so\n"
    )
    assert _mapped_backends(str(maps)) == ["/opt/sdk/lib/libhipdnn_backend.so"]
