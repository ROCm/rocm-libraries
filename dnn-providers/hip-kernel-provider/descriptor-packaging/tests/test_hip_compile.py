"""Unit tests for hkp_pack.hip_compile._hip_compile_command."""

import pytest

from hkp_pack.hip_compile import _hip_compile_command


_COMPILER = "/usr/bin/clang++"
_SOURCE = "/src/kernel.hip"
_ARCH = "gfx942"
_OUT = "/out/kernel.co"


def _cmd(build=None, hip_path=None, device_lib_path=None):
    return _hip_compile_command(
        _COMPILER, _SOURCE, _ARCH, build or {}, _OUT, hip_path, device_lib_path
    )


class TestDefaultOptimisation:
    def test_default_O3_when_no_opt_flag(self):
        cmd = _cmd()
        assert "-O3" in cmd

    def test_authored_O2_suppresses_default(self):
        cmd = _cmd({"flags": ["-O2"]})
        assert "-O2" in cmd
        assert "-O3" not in cmd

    def test_authored_O0_suppresses_default(self):
        cmd = _cmd({"flags": ["-O0"]})
        assert "-O0" in cmd
        assert "-O3" not in cmd


class TestPathFlags:
    def test_hip_path_present_when_given(self):
        cmd = _cmd(hip_path="/opt/rocm")
        assert "--hip-path=/opt/rocm" in cmd

    def test_hip_path_absent_when_none(self):
        cmd = _cmd(hip_path=None)
        assert not any("--hip-path" in c for c in cmd)

    def test_device_lib_path_present_when_given(self):
        cmd = _cmd(device_lib_path="/opt/rocm/amdgcn/bitcode")
        assert "--hip-device-lib-path=/opt/rocm/amdgcn/bitcode" in cmd

    def test_device_lib_path_absent_when_none(self):
        cmd = _cmd(device_lib_path=None)
        assert not any("--hip-device-lib-path" in c for c in cmd)


class TestDefines:
    def test_string_define(self):
        cmd = _cmd({"defines": {"FOO": "bar"}})
        assert "-DFOO=bar" in cmd

    def test_bool_true_define(self):
        cmd = _cmd({"defines": {"ENABLE": True}})
        assert "-DENABLE=1" in cmd

    def test_bool_false_define(self):
        cmd = _cmd({"defines": {"DISABLE": False}})
        assert "-DDISABLE=0" in cmd


class TestCommandStructure:
    def test_compiler_is_first(self):
        cmd = _cmd()
        assert cmd[0] == _COMPILER

    def test_offload_device_only_and_x_hip(self):
        cmd = _cmd()
        assert "--offload-device-only" in cmd
        assert "-x" in cmd
        idx = cmd.index("-x")
        assert cmd[idx + 1] == "hip"

    def test_offload_arch(self):
        cmd = _cmd()
        assert f"--offload-arch={_ARCH}" in cmd

    def test_cuid_none_pinned(self):
        cmd = _cmd()
        assert "-fuse-cuid=none" in cmd

    def test_source_and_output_at_end(self):
        cmd = _cmd()
        assert cmd[-3:] == [_SOURCE, "-o", _OUT]
