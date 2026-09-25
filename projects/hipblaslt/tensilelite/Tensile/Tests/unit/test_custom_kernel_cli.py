# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""CLI-entrypoint coverage for the custom-kernel tooling: the ``main()``
functions of ``Tensile.AddCustomConfig`` and ``Tensile.ValidateMetadata``.

Both are driven by argparse over ``sys.argv`` and terminate via ``sys.exit``;
these tests monkeypatch ``sys.argv`` and assert exit codes + stdout/stderr,
using real .s files written into ``tmp_path``.
"""

import sys
from textwrap import dedent, indent

import pytest

import Tensile.AddCustomConfig as acc
import Tensile.ValidateMetadata as vm

pytestmark = pytest.mark.unit


META_EMPTY = dedent("""\
    .amdgpu_metadata
    ---
    amdhsa.kernels: []
    ...
    """)


def _write_kernel_s(path, config_block):
    """Write a minimal custom-kernel .s file with an embedded custom.config."""
    body = indent(dedent(config_block).strip(), "  ")
    path.write_text(
        ".amdgpu_metadata\n"
        "---\n"
        "custom.config:\n"
        f"{body}\n"
        "amdhsa.kernels:\n"
        f"  - .name: {path.stem}\n"
        "    .max_flat_workgroup_size: 64\n"
        "    .args:\n"
        "      - .name: D\n"
        "        .size: 8\n"
        "        .offset: 0\n"
        "        .value_kind: global_buffer\n"
        "...\n"
    )


# --------------------------------------------------------------------------- #
# AddCustomConfig.main()
# --------------------------------------------------------------------------- #


def test_addcustomconfig_main_file_not_found(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["prog", "/no/such/file.s"])

    with pytest.raises(SystemExit) as e:
        acc.main()

    assert e.value.code == 1
    assert "File not found" in capsys.readouterr().err


def test_addcustomconfig_main_provenance_only_injects(tmp_path, monkeypatch, capsys):
    p = tmp_path / "k.s"
    p.write_text(META_EMPTY)
    monkeypatch.setattr(sys, "argv", ["prog", str(p)])

    acc.main()  # returns normally (no sys.exit on success)

    assert "custom.config:" in p.read_text()
    assert "Injected custom.config" in capsys.readouterr().out


def test_addcustomconfig_main_dry_run_non_s_warns(tmp_path, monkeypatch, capsys):
    p = tmp_path / "k.txt"
    p.write_text(META_EMPTY)
    monkeypatch.setattr(sys, "argv", ["prog", str(p), "--dry-run"])

    acc.main()

    out = capsys.readouterr()
    assert "does not end with .s" in out.err
    assert "custom.config block that would be inserted" in out.out
    assert p.read_text() == META_EMPTY  # dry-run leaves the file untouched


def test_addcustomconfig_main_already_has_config(tmp_path, monkeypatch, capsys):
    p = tmp_path / "k.s"
    _write_kernel_s(p, """\
        InternalSupportParams:
          KernArgsVersion: 0
        """)
    monkeypatch.setattr(sys, "argv", ["prog", str(p)])

    with pytest.raises(SystemExit) as e:
        acc.main()

    assert e.value.code == 1
    assert "already has a custom.config" in capsys.readouterr().err


def test_addcustomconfig_main_yaml_parse_error(tmp_path, monkeypatch, capsys):
    p = tmp_path / "k.s"
    p.write_text(META_EMPTY)
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("BenchmarkProblems: [\n")
    monkeypatch.setattr(sys, "argv", ["prog", str(p), "--yaml", str(bad_yaml)])

    with pytest.raises(SystemExit) as e:
        acc.main()

    assert e.value.code == 1
    assert "ERROR" in capsys.readouterr().err


def test_addcustomconfig_main_read_error_exits(tmp_path, monkeypatch, capsys):
    p = tmp_path / "k.s"
    p.write_text(dedent("""\
        .amdgpu_metadata
        ---
        custom.config: [
        ...
        """))
    monkeypatch.setattr(sys, "argv", ["prog", str(p)])

    with pytest.raises(SystemExit) as e:
        acc.main()

    assert e.value.code == 1


def test_addcustomconfig_main_with_yaml_injects_interface(tmp_path, monkeypatch):
    # main() derives the kernel name from the .s basename, so it must match a
    # CustomKernel name declared in the YAML.
    p = tmp_path / "mykernel.s"
    p.write_text(META_EMPTY)
    y = tmp_path / "t.yaml"
    y.write_text(dedent("""\
        BenchmarkProblems:
          -
            - OperationType: GEMM
            - ForkParameters:
              - CustomKernel:
                - name: mykernel
                  args: []
                  macrotile: [256, 256, 64]
                  threads: [256, 1, 1]
                  grid: [TilesX, TilesY, One]
              - MatrixInstruction:
                - [16, 16, 16, 1]
        """))
    monkeypatch.setattr(sys, "argv", ["prog", str(p), "--yaml", str(y)])

    acc.main()

    text = p.read_text()
    assert "custom.config:" in text
    assert "ProblemType:" in text
    assert "CustomKernel:" in text


_META_WITH_DETECTED = dedent("""\
    .amdgpu_metadata
    ---
    amdhsa.kernels:
      - .name: mykernel
        .reqd_workgroup_size: [128, 1, 1]
        .wavefront_size: 64
        .args:
          - .name: D
            .size: 8
            .offset: 0
            .value_kind: global_buffer
    ...
    """)


def test_addcustomconfig_main_autodetects_threads_and_wavefront(tmp_path, monkeypatch, capsys):
    p = tmp_path / "k.s"
    p.write_text(_META_WITH_DETECTED)
    monkeypatch.setattr(sys, "argv", ["prog", str(p)])

    acc.main()

    out = capsys.readouterr().out
    assert "Auto-detected" in out
    assert "wavefront_size=64" in out
    assert "threads=[128, 1, 1]" in out
    assert "custom.config:" in p.read_text()


def test_addcustomconfig_main_merges_detected_into_yaml_config(tmp_path, monkeypatch):
    # The YAML CustomKernel omits threads/WavefrontSize; main() fills them from
    # the .s .reqd_workgroup_size / .wavefront_size auto-detection.
    p = tmp_path / "mykernel.s"
    p.write_text(_META_WITH_DETECTED)
    y = tmp_path / "t.yaml"
    y.write_text(dedent("""\
        BenchmarkProblems:
          -
            - OperationType: GEMM
            - ForkParameters:
              - CustomKernel:
                - name: mykernel
                  args: []
                  macrotile: [256, 256, 64]
                  grid: [TilesX, TilesY, One]
              - MatrixInstruction:
                - [16, 16, 16, 1]
        """))
    monkeypatch.setattr(sys, "argv", ["prog", str(p), "--yaml", str(y)])

    acc.main()

    text = p.read_text()
    assert "threads: [128, 1, 1]" in text
    assert "WavefrontSize: 64" in text


def test_addcustomconfig_main_inject_failure_exits(tmp_path, monkeypatch, capsys):
    # A .s with no .amdgpu_metadata section has no insert point, so injection
    # fails and main() exits non-zero.
    p = tmp_path / "k.s"
    p.write_text("s_nop 0\ns_endpgm\n")
    monkeypatch.setattr(sys, "argv", ["prog", str(p)])

    with pytest.raises(SystemExit) as e:
        acc.main()

    assert e.value.code == 1
    assert "No .amdgpu_metadata" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# ValidateMetadata.main()
# --------------------------------------------------------------------------- #


def test_validatemetadata_main_root_not_found(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["prog", "--custom-kernels-root", "/no/such/dir"])

    with pytest.raises(SystemExit) as e:
        vm.main()

    assert e.value.code == 1
    assert "not found" in capsys.readouterr().err


def test_validatemetadata_main_clean_exits_zero(tmp_path, monkeypatch, capsys):
    _write_kernel_s(tmp_path / "good.s", """\
        InternalSupportParams:
          KernArgsVersion: 0
        """)
    monkeypatch.setattr(sys, "argv", ["prog", "--custom-kernels-root", str(tmp_path)])

    with pytest.raises(SystemExit) as e:
        vm.main()

    assert e.value.code == 0
    assert "0 error(s)" in capsys.readouterr().out


def test_validatemetadata_main_strict_bad_exits_one(tmp_path, monkeypatch, capsys):
    _write_kernel_s(tmp_path / "bad.s", """\
        InternalSupportParams: {}
        """)
    monkeypatch.setattr(sys, "argv",
                        ["prog", "--custom-kernels-root", str(tmp_path), "--strict"])

    with pytest.raises(SystemExit) as e:
        vm.main()

    assert e.value.code == 1
    assert "1 error(s)" in capsys.readouterr().out


def test_validatemetadata_main_non_strict_bad_exits_zero(tmp_path, monkeypatch, capsys):
    _write_kernel_s(tmp_path / "bad.s", """\
        InternalSupportParams: {}
        """)
    monkeypatch.setattr(sys, "argv", ["prog", "--custom-kernels-root", str(tmp_path)])

    with pytest.raises(SystemExit) as e:
        vm.main()

    assert e.value.code == 0
    assert "1 warning(s)" in capsys.readouterr().out


def test_validatemetadata_main_default_root_uses_shipped_kernels(monkeypatch, capsys):
    # With no --custom-kernels-root, main() auto-detects the shipped
    # Tensile/CustomKernels directory and validates the real kernels.
    monkeypatch.setattr(sys, "argv", ["prog"])

    with pytest.raises(SystemExit) as e:
        vm.main()

    assert e.value.code == 0
    assert "Summary:" in capsys.readouterr().out
