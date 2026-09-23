# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for generate.py.

Tests cover _preview_files() correctness across the supported modes,
mode enum file inclusion, and CLI behavior.
"""

import subprocess
from pathlib import Path

import pytest

from generate import (
    MODE_BACKEND,
    MODE_FRONTEND,
    MODE_FULL,
    _preview_files,
)
from codegen.config_loader import load_config
from codegen.models import EnumDef, EnumValue
from tests.helpers import make_data_field, make_minimal_config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config_with_mode_fields():
    """Config with a generatable mode field (non-shared, with enum_def)."""
    enum_def = EnumDef(
        backend_header="HipdnnTestMode.h",
        backend_prefix="HIPDNN_TEST_",
        values=[
            EnumValue(name="FOO", value=1),
            EnumValue(name="BAR", value=2),
        ],
    )
    return make_minimal_config(
        data_fields=[
            make_data_field(
                name="test_mode",
                type="mode",
                test_enum_value="FOO",
                test_backend_value="HIPDNN_TEST_FOO",
                backend_setter="setTestMode",
                backend_getter="getTestMode",
                backend_type_name="HIPDNN_TYPE_TEST_MODE",
                frontend_inverse_converter="fromTestMode",
                test_alt_enum_value="BAR",
                shared=False,
                enum_def=enum_def,
            )
        ]
    )


@pytest.fixture
def config_no_mode_fields():
    """Config without any mode fields (matmul-like)."""
    return make_minimal_config(data_fields=[])


# ---------------------------------------------------------------------------
# Task 2B.9: _preview_files() correctness
# ---------------------------------------------------------------------------


class TestPreviewFilesBackend:
    """Verify backend mode produces the expected file list."""

    def test_backend_file_count(self, convolution_fwd_config):
        files = _preview_files(convolution_fwd_config, MODE_BACKEND)
        # 9 file templates + 12 fragment templates + 4 per generatable mode field
        n_mode_files = 4 * len(convolution_fwd_config.generatable_mode_fields)
        expected = 9 + 12 + n_mode_files
        assert len(files) == expected

    def test_backend_contains_descriptor_header(self, convolution_fwd_config):
        files = _preview_files(convolution_fwd_config, MODE_BACKEND)
        assert any("ConvolutionFwdOperationDescriptor.hpp" in f for f in files)

    def test_backend_contains_descriptor_source(self, convolution_fwd_config):
        files = _preview_files(convolution_fwd_config, MODE_BACKEND)
        assert any("ConvolutionFwdOperationDescriptor.cpp" in f for f in files)

    def test_backend_contains_packer(self, convolution_fwd_config):
        files = _preview_files(convolution_fwd_config, MODE_BACKEND)
        assert any("Packer.hpp" in f for f in files)

    def test_backend_contains_unpacker(self, convolution_fwd_config):
        files = _preview_files(convolution_fwd_config, MODE_BACKEND)
        assert any("Unpacker.hpp" in f for f in files)

    def test_backend_contains_test_descriptor(self, convolution_fwd_config):
        files = _preview_files(convolution_fwd_config, MODE_BACKEND)
        assert any("TestConvolutionFwdOperationDescriptor.cpp" in f for f in files)

    def test_backend_contains_test_graph(self, convolution_fwd_config):
        files = _preview_files(convolution_fwd_config, MODE_BACKEND)
        assert any("TestGraphDescriptorConvolutionFwd.cpp" in f for f in files)

    def test_backend_contains_test_from_node(self, convolution_fwd_config):
        files = _preview_files(convolution_fwd_config, MODE_BACKEND)
        assert any("TestConvolutionFwdOperationFromNode.cpp" in f for f in files)

    def test_backend_contains_integration_lowering(self, convolution_fwd_config):
        files = _preview_files(convolution_fwd_config, MODE_BACKEND)
        assert any(
            "IntegrationConvolutionFpropDescriptorLowering.cpp" in f for f in files
        )

    def test_backend_contains_integration_lifting(self, convolution_fwd_config):
        files = _preview_files(convolution_fwd_config, MODE_BACKEND)
        assert any(
            "IntegrationConvolutionFpropDescriptorLifting.cpp" in f for f in files
        )

    def test_backend_contains_all_fragments(self, convolution_fwd_config):
        files = _preview_files(convolution_fwd_config, MODE_BACKEND)
        expected_fragments = [
            "fragments/attribute_enum_block.txt",
            "fragments/descriptor_type_enum.txt",
            "fragments/string_utils_block.txt",
            "fragments/factory_case.txt",
            "fragments/cmake_entries.txt",
            "fragments/node_factory_case.txt",
            "fragments/operation_unpacker_case.txt",
            "fragments/operation_unpacker_test.txt",
            "fragments/operation_type_enum.txt",
            "fragments/node_unpack_override.txt",
            "fragments/packer_name_addition.txt",
            "fragments/descriptor_lifting_additions.txt",
        ]
        for fragment in expected_fragments:
            assert fragment in files, f"Missing fragment: {fragment}"


class TestPreviewFilesFrontend:
    """Verify frontend mode produces the expected file list."""

    def test_frontend_file_count(self, convolution_fwd_config):
        files = _preview_files(convolution_fwd_config, MODE_FRONTEND)
        # 2 files + 3 tests + 4 fragments = 9
        assert len(files) == 9

    def test_frontend_contains_attributes_header(self, convolution_fwd_config):
        files = _preview_files(convolution_fwd_config, MODE_FRONTEND)
        assert any("Attributes.hpp" in f for f in files)

    def test_frontend_contains_node_header(self, convolution_fwd_config):
        files = _preview_files(convolution_fwd_config, MODE_FRONTEND)
        assert any("Node.hpp" in f for f in files)

    def test_frontend_contains_test_attributes(self, convolution_fwd_config):
        files = _preview_files(convolution_fwd_config, MODE_FRONTEND)
        assert any("TestConvolutionFpropAttributes.cpp" in f for f in files)

    def test_frontend_contains_test_node(self, convolution_fwd_config):
        files = _preview_files(convolution_fwd_config, MODE_FRONTEND)
        assert any("TestConvolutionFpropNode.cpp" in f for f in files)

    def test_frontend_contains_test_graph(self, convolution_fwd_config):
        files = _preview_files(convolution_fwd_config, MODE_FRONTEND)
        assert any("TestGraphConvolutionFprop.cpp" in f for f in files)

    def test_frontend_contains_all_fragments(self, convolution_fwd_config):
        files = _preview_files(convolution_fwd_config, MODE_FRONTEND)
        expected_fragments = [
            "fragments/graph_method.txt",
            "fragments/graph_includes.txt",
            "fragments/frontend_cmake_entries.txt",
        ]
        for fragment in expected_fragments:
            assert fragment in files, f"Missing fragment: {fragment}"

    def test_frontend_does_not_contain_backend_files(self, convolution_fwd_config):
        files = _preview_files(convolution_fwd_config, MODE_FRONTEND)
        assert not any("descriptors/" in f for f in files)
        assert not any("factory_case.txt" in f for f in files)


class TestPreviewFilesFull:
    """Verify full mode combines backend + frontend."""

    def test_full_contains_backend_and_frontend(self, convolution_fwd_config):
        files = _preview_files(convolution_fwd_config, MODE_FULL)
        # Must contain both backend and frontend files
        assert any("ConvolutionFwdOperationDescriptor.hpp" in f for f in files)
        assert any("ConvolutionFpropAttributes.hpp" in f for f in files)
        assert any("factory_case.txt" in f for f in files)
        assert any("graph_method.txt" in f for f in files)

    def test_full_file_count(self, convolution_fwd_config):
        backend = _preview_files(convolution_fwd_config, MODE_BACKEND)
        frontend = _preview_files(convolution_fwd_config, MODE_FRONTEND)
        full = _preview_files(convolution_fwd_config, MODE_FULL)
        # Full = backend + frontend (mode enum files are included in both)
        # Mode enum files are only added for backend and full, not frontend
        # The full count should be backend + frontend (mode enums counted once in full)
        assert len(full) == len(backend) + len(frontend)


# ---------------------------------------------------------------------------
# Mode enum files -- present/absent based on config
# ---------------------------------------------------------------------------


class TestPreviewModeEnumFiles:
    """Verify mode enum files are included only when generatable_mode_fields exist."""

    def test_backend_includes_mode_enum_files(self, config_with_mode_fields):
        files = _preview_files(config_with_mode_fields, MODE_BACKEND)
        assert any("HipdnnTestMode.h" in f for f in files)
        assert any("mode_backend_plumbing_test_mode.txt" in f for f in files)
        assert any("mode_frontend_plumbing_test_mode.txt" in f for f in files)
        assert any("mode_frontend_tests_test_mode.txt" in f for f in files)

    def test_full_includes_mode_enum_files(self, config_with_mode_fields):
        files = _preview_files(config_with_mode_fields, MODE_FULL)
        assert any("HipdnnTestMode.h" in f for f in files)

    def test_no_mode_fields_no_mode_enum_files(self, config_no_mode_fields):
        files = _preview_files(config_no_mode_fields, MODE_BACKEND)
        assert not any("mode_backend_plumbing" in f for f in files)
        assert not any("mode_frontend_plumbing" in f for f in files)

    def test_frontend_mode_no_mode_enum_files(self, config_with_mode_fields):
        """Frontend mode does not include mode enum files."""
        files = _preview_files(config_with_mode_fields, MODE_FRONTEND)
        assert not any("HipdnnTestMode.h" in f for f in files)
        assert not any("mode_backend_plumbing" in f for f in files)

    def test_shared_mode_field_not_generatable(self):
        """Shared mode fields are excluded from generatable_mode_fields."""
        enum_def = EnumDef(
            backend_header="HipdnnShared.h",
            backend_prefix="HIPDNN_SHARED_",
            values=[EnumValue(name="A", value=1)],
        )
        config = make_minimal_config(
            data_fields=[
                make_data_field(
                    name="shared_mode",
                    type="mode",
                    shared=True,
                    test_enum_value="A",
                    test_backend_value="HIPDNN_SHARED_A",
                    backend_setter="setShared",
                    backend_getter="getShared",
                    backend_type_name="HIPDNN_TYPE_SHARED",
                    frontend_inverse_converter="fromShared",
                    test_alt_enum_value="B",
                    enum_def=enum_def,
                )
            ]
        )
        files = _preview_files(config, MODE_BACKEND)
        assert not any("HipdnnShared.h" in f for f in files)


class TestPreviewFilesWithRealConfigs:
    """Cross-check _preview_files with real configs (convolution_fwd vs matmul)."""

    def test_convolution_has_mode_enum_files(self, convolution_fwd_config):
        """ConvFwd has a non-shared mode field with enum_def."""
        files = _preview_files(convolution_fwd_config, MODE_BACKEND)
        assert any("HipdnnConvolutionMode.h" in f for f in files)

    def test_matmul_has_no_mode_enum_files(self, matmul_config):
        """Matmul has no data fields, so no mode enum files."""
        files = _preview_files(matmul_config, MODE_BACKEND)
        assert not any("mode_backend_plumbing" in f for f in files)
        assert not any("mode_frontend_plumbing" in f for f in files)

    def test_matmul_backend_exact_count(self, matmul_config):
        """Matmul backend: 9 files + 12 fragments = 21, no mode enums.

        matmul_config has constants_include set, so no constants file is generated.
        """
        files = _preview_files(matmul_config, MODE_BACKEND)
        assert len(files) == 21


class TestPreviewFilesConstants:
    """Verify constants file presence in preview based on constants_include."""

    def test_no_constants_when_constants_include_set(self, convolution_fwd_config):
        """ConvFwd has constants_include set, no constants file in preview."""
        files = _preview_files(convolution_fwd_config, MODE_BACKEND)
        constants_files = [f for f in files if "constants/" in f]
        assert len(constants_files) == 0

    def test_constants_in_backend_when_not_set(self, load_test_config):
        """Config without constants_include gets a constants file in backend preview."""
        config = load_test_config("batchnorm_backward.yaml")
        files = _preview_files(config, MODE_BACKEND)
        constants_files = [f for f in files if "constants/" in f]
        assert len(constants_files) == 1
        assert "BatchnormBackwardConstants.hpp" in constants_files[0]

    def test_no_constants_in_frontend_mode(self, convolution_fwd_config):
        """Frontend mode never generates constants file."""
        files = _preview_files(convolution_fwd_config, MODE_FRONTEND)
        constants_files = [f for f in files if "constants/" in f]
        assert len(constants_files) == 0


# ---------------------------------------------------------------------------
# Task 2B.10: CLI behavior tests
# ---------------------------------------------------------------------------


class TestCLIBehavior:
    """Test CLI argument handling and edge cases via subprocess."""

    @pytest.fixture
    def generate_script(self):
        """Path to the generate.py script."""
        return Path(__file__).parent.parent / "generate.py"

    @pytest.fixture
    def python_exe(self):
        """Path to the .venv Python executable."""
        exe = Path(__file__).parent.parent / ".venv" / "bin" / "python"
        if not exe.exists():
            pytest.skip("No .venv/bin/python found; CLI tests require a local venv")
        return exe

    def test_missing_config_file_exits(self, generate_script, python_exe, tmp_path):
        """--config pointing to a nonexistent file exits with code 1."""
        result = subprocess.run(
            [
                str(python_exe),
                str(generate_script),
                "--config",
                str(tmp_path / "nonexistent.yaml"),
                "--output-dir",
                str(tmp_path / "out"),
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 1
        assert "not found" in result.stderr

    def test_lift_only_flag_removed(self, generate_script, python_exe, tmp_path):
        """The deprecated --lift-only flag is no longer accepted."""
        config_file = Path(__file__).parent.parent / "configs" / "matmul.yaml"
        result = subprocess.run(
            [
                str(python_exe),
                str(generate_script),
                "--config",
                str(config_file),
                "--output-dir",
                str(tmp_path / "out"),
                "--lift-only",
            ],
            capture_output=True,
            text=True,
        )
        # argparse exits with code 2 on unknown arguments
        assert result.returncode == 2
        assert (
            "unrecognized arguments" in result.stderr or "--lift-only" in result.stderr
        )

    def test_mode_lift_only_value_rejected(self, generate_script, python_exe, tmp_path):
        """--mode lift-only is rejected by argparse choices after the flag was removed."""
        config_file = Path(__file__).parent.parent / "configs" / "matmul.yaml"
        result = subprocess.run(
            [
                str(python_exe),
                str(generate_script),
                "--config",
                str(config_file),
                "--output-dir",
                str(tmp_path / "out"),
                "--mode",
                "lift-only",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2, (
            f"Expected argparse rejection (rc=2), got {result.returncode}: "
            f"stderr={result.stderr!r}"
        )
        assert "invalid choice" in result.stderr or "lift-only" in result.stderr

    def test_mode_lift_only_constant_removed(self):
        """MODE_LIFT_ONLY symbol no longer exists; the constant was folded into MODE_BACKEND."""
        import generate

        assert not hasattr(
            generate, "MODE_LIFT_ONLY"
        ), "MODE_LIFT_ONLY should have been folded into MODE_BACKEND when the lift-only mode was removed"
        # The valid modes tuple should not contain a 'lift-only' entry.
        assert "lift-only" not in generate.VALID_MODES

    def test_dry_run_no_files_written(self, generate_script, python_exe, tmp_path):
        """--dry-run does not create any files."""
        config_file = Path(__file__).parent.parent / "configs" / "matmul.yaml"
        output_dir = tmp_path / "dry_run_output"
        result = subprocess.run(
            [
                str(python_exe),
                str(generate_script),
                "--config",
                str(config_file),
                "--output-dir",
                str(output_dir),
                "--mode",
                "backend",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Dry run" in result.stdout
        assert "No files were written" in result.stdout
        # Output dir should not exist (dry run does not create it)
        assert not output_dir.exists()

    def test_valid_backend_run(self, generate_script, python_exe, tmp_path):
        """Valid backend run creates files and reports success."""
        config_file = Path(__file__).parent.parent / "configs" / "matmul.yaml"
        output_dir = tmp_path / "backend_output"
        result = subprocess.run(
            [
                str(python_exe),
                str(generate_script),
                "--config",
                str(config_file),
                "--output-dir",
                str(output_dir),
                "--mode",
                "backend",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Generated" in result.stdout
        assert "Done" in result.stdout
        # Verify some files were actually created
        assert output_dir.exists()
        generated_files = list(output_dir.rglob("*"))
        assert len(generated_files) > 0

    def test_dry_run_file_list_matches_expected(
        self, generate_script, python_exe, tmp_path
    ):
        """Dry run prints the same file count as _preview_files returns."""
        config_file = Path(__file__).parent.parent / "configs" / "matmul.yaml"
        result = subprocess.run(
            [
                str(python_exe),
                str(generate_script),
                "--config",
                str(config_file),
                "--output-dir",
                str(tmp_path / "out"),
                "--mode",
                "backend",
                "--dry-run",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # Count lines with file paths (indented with two spaces, contain / or .)
        file_lines = [
            line
            for line in result.stdout.splitlines()
            if line.startswith("  ") and ("/" in line or "." in line)
        ]
        matmul_config = load_config(config_file)
        expected = _preview_files(matmul_config, MODE_BACKEND)
        assert len(file_lines) == len(expected)
