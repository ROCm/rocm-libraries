# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""ISA census is diagnostic evidence, not a hardware correctness or speed gate."""

import errno
import importlib.util
from types import SimpleNamespace

import pytest
from rocke.assets import platform_tools_dir


def _tool():
    path = platform_tools_dir() / "compare_fragment_transport_isa.py"
    spec = importlib.util.spec_from_file_location("fragment_transport_isa", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_opcode_counts_ignore_comments_labels_and_directives():
    tool = _tool()
    assembly = """
// v_mov_b32 is mentioned, not executed
v_label:
    .amdhsa_next_free_vgpr 20
    v_mov_b32 v0, 0
    global_load_dwordx3 v[1:3], v[4:5], off
    v_mfma_scale_f32_32x32x64_f8f6f4 a[0:15], v[0:7], v[8:15], a[0:15], v16, v17
    s_endpgm
"""
    assert tool.opcode_counts(assembly) == {
        "global_load_dwordx3": 1,
        "s_endpgm": 1,
        "v_mfma_scale_f32_32x32x64_f8f6f4": 1,
        "v_mov_b32": 1,
    }
    assert tool.resource_metadata(assembly) == {"next_free_vgpr": 20}


def test_added_padding_move_requires_review():
    tool = _tool()
    baseline = "    v_wmma_f32_16x16x128_f8f6f4 v0, v1, v2, v0\n"
    candidate = "    v_mov_b32 v9, 0\n" + baseline
    report = tool.compare_assembly(baseline, candidate)
    assert report["status"] == "review-required"
    assert report["opcode_delta"] == {"v_mov_b32": 1}


def test_changed_matrix_instruction_or_empty_evidence_fails():
    tool = _tool()
    baseline = "    v_mfma_f32_16x16x8_xf32 a0, v0, v1, a0\n"
    candidate = "    v_mfma_f32_16x16x4f32 a0, v0, v1, a0\n"
    assert tool.compare_assembly(baseline, candidate)["status"] == "matrix-mismatch"
    assert tool.compare_assembly("", "")["status"] == "matrix-mismatch"


def test_missing_resource_metadata_is_not_treated_as_zero():
    tool = _tool()
    baseline = "    v_mfma_f32_16x16x8_xf32 a0, v0, v1, a0\n"
    candidate = baseline + "    .amdhsa_private_segment_fixed_size 0\n"
    assert tool.compare_assembly(baseline, candidate)["status"] == "review-required"


def test_compiler_multicall_symlink_keeps_its_driver_name(tmp_path):
    target = tmp_path / "amdllvm"
    target.write_text("multicall driver")
    driver = tmp_path / "amdclang++"
    try:
        driver.symlink_to(target)
    except NotImplementedError as exc:
        pytest.skip(f"cannot create symlinks: {exc}")
    except OSError as exc:
        # Windows may report ERROR_NOT_SUPPORTED or ERROR_PRIVILEGE_NOT_HELD.
        if getattr(exc, "winerror", None) in (50, 1314) or exc.errno in {
            errno.EACCES,
            errno.EPERM,
            errno.ENOSYS,
            errno.ENOTSUP,
            errno.EOPNOTSUPP,
        }:
            pytest.skip(f"cannot create symlinks: {exc}")
        raise
    assert _tool().compiler_driver_path(str(driver)) == str(driver)


def test_compile_arm_preserves_diagnostics_and_refuses_failed_assembly(
    tmp_path, monkeypatch
):
    tool = _tool()
    out = tmp_path / "candidate"

    def run(command, **kwargs):
        if kwargs["check"]:
            assert command[1] == "-c"
            return SimpleNamespace(stdout="emitted IR\n")
        assert command[0] == "synthetic-compiler"
        assert "-mcpu=gfx950" in command
        return SimpleNamespace(returncode=17, stdout="tool banner\n", stderr="bad IR\n")

    monkeypatch.setattr(tool, "subprocess", SimpleNamespace(run=run))
    with pytest.raises(RuntimeError, match=r"compiler failed \(17\): .*compiler\.log"):
        tool.compile_arm(
            tmp_path, {"target": "gfx950"}, "llvm23", "synthetic-compiler", out
        )
    assert (out / "compiler.log").read_text(encoding="utf-8") == "tool banner\nbad IR\n"
    assert not (out / "kernel.s").exists()


def test_case_inventory_covers_widths_mixed_formats_and_preparations():
    table = _tool().cases()
    assert {case["target"] for case in table} == {"gfx942", "gfx950", "gfx1250"}
    assert {case["mode"] for case in table if case["kind"] == "gfx942"} == {
        "raw",
        "carrier",
        "rne",
        "prepacked",
        "fp32",
    }
    assert any(case.get("tensor_scale") for case in table)
    assert any(case.get("a") != case.get("b") for case in table if "a" in case)
    assert any(case.get("path") == "wmma" for case in table)
    assert len(table) == 69
