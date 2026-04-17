# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Integration tests for the descriptor codegen pipeline.

Tests exercise the full pipeline from config loading through template rendering,
covering generator.py render methods and generate.py's _preview_files function.
"""

import pytest
from pathlib import Path

from codegen.config_loader import load_config, validate_for_mode
from codegen.generator import DescriptorGenerator
from generate import _preview_files


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

from tests.helpers import ALL_CONFIG_NAMES

# Configs with complete frontend fields (inputs, outputs, node_type_enum)
FRONTEND_CAPABLE_CONFIGS = [
    "convolution_fwd.yaml",
    "pointwise.yaml",
    "batchnorm_inference.yaml",
    "sdpa.yaml",
]

# Configs that generate mode enum files (has enum_def, not shared)
MODE_ENUM_CONFIGS = [
    "convolution_fwd.yaml",
    "pointwise.yaml",
]

# Copyright header present in all generated C++ files
COPYRIGHT_MARKER = "Copyright"


def _applicable_modes(config_name: str) -> list[str]:
    """Return the list of render modes applicable to a given config."""
    modes = ["backend", "lift-only"]
    if config_name in FRONTEND_CAPABLE_CONFIGS:
        modes.extend(["frontend", "full"])
    return modes


# ---------------------------------------------------------------------------
# Task 3.1.1: Full pipeline per config x mode (parameterized)
# ---------------------------------------------------------------------------


def _build_config_mode_params():
    """Build parameterized (config_name, mode) tuples for all applicable combos."""
    params = []
    for config_name in ALL_CONFIG_NAMES:
        for mode in _applicable_modes(config_name):
            params.append((config_name, mode))
    return params


class TestFullPipeline:
    """Full pipeline tests across configs and modes."""

    @pytest.mark.parametrize(
        "config_name, mode",
        _build_config_mode_params(),
        ids=[f"{c}-{m}" for c, m in _build_config_mode_params()],
    )
    def test_render_produces_files(
        self, config_name, mode, load_test_config, generator, tmp_path
    ):
        """Rendering produces a non-empty list of files that all exist on disk."""
        config = load_test_config(config_name)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        written = generator.render(config, output_dir, mode)

        assert len(written) > 0, f"No files generated for {config_name} in {mode} mode"

        for rel_path in written:
            full_path = output_dir / rel_path
            assert full_path.exists(), f"Generated file does not exist: {rel_path}"
            content = full_path.read_text()
            assert len(content) > 0, f"Generated file is empty: {rel_path}"

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_backend_files_have_copyright(
        self, config_name, load_test_config, generator, tmp_path
    ):
        """Generated C++ files in backend mode contain a copyright header."""
        config = load_test_config(config_name)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        written = generator.render(config, output_dir, "backend")

        cpp_files = [f for f in written if f.endswith((".hpp", ".cpp"))]
        assert len(cpp_files) > 0, "Expected at least one C++ file in backend output"

        for rel_path in cpp_files:
            content = (output_dir / rel_path).read_text()
            assert COPYRIGHT_MARKER in content, f"Missing copyright in {rel_path}"

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_lift_only_files_have_copyright(
        self, config_name, load_test_config, generator, tmp_path
    ):
        """Generated C++ files in lift-only mode contain a copyright header."""
        config = load_test_config(config_name)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        written = generator.render(config, output_dir, "lift-only")

        cpp_files = [f for f in written if f.endswith((".hpp", ".cpp"))]
        assert len(cpp_files) > 0, "Expected at least one C++ file in lift-only output"

        for rel_path in cpp_files:
            content = (output_dir / rel_path).read_text()
            assert COPYRIGHT_MARKER in content, f"Missing copyright in {rel_path}"

    def test_convolution_fwd_all_four_modes(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """convolution_fwd renders successfully in all four modes."""
        config = convolution_fwd_config
        for mode in ["backend", "frontend", "full", "lift-only"]:
            output_dir = tmp_path / f"output_{mode}"
            output_dir.mkdir()
            written = generator.render(config, output_dir, mode)
            assert len(written) > 0, f"No files for mode={mode}"
            for rel_path in written:
                assert (output_dir / rel_path).exists()

    def test_full_mode_produces_superset_of_backend_and_frontend(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Full mode output is the union of backend and frontend outputs."""
        config = convolution_fwd_config

        backend_dir = tmp_path / "backend"
        backend_dir.mkdir()
        backend_files = set(generator.render(config, backend_dir, "backend"))

        frontend_dir = tmp_path / "frontend"
        frontend_dir.mkdir()
        frontend_files = set(generator.render(config, frontend_dir, "frontend"))

        full_dir = tmp_path / "full"
        full_dir.mkdir()
        full_files = set(generator.render(config, full_dir, "full"))

        # Full mode contains all backend files and all frontend files
        assert backend_files.issubset(
            full_files
        ), f"Backend files missing from full: {backend_files - full_files}"
        assert frontend_files.issubset(
            full_files
        ), f"Frontend files missing from full: {frontend_files - full_files}"


# ---------------------------------------------------------------------------
# Task 3.1.2: Dry-run parity
# ---------------------------------------------------------------------------


class TestDryRunParity:
    """Verify _preview_files matches actual generation."""

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_backend_parity(self, config_name, load_test_config, generator, tmp_path):
        """Preview file list matches actual backend render output."""
        config = load_test_config(config_name)
        expected = set(_preview_files(config, "backend"))

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        actual = set(generator.render(config, output_dir, "backend"))

        assert expected == actual, (
            f"Dry-run parity mismatch for {config_name} backend.\n"
            f"  Only in preview: {expected - actual}\n"
            f"  Only in actual: {actual - expected}"
        )

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_lift_only_parity(self, config_name, load_test_config, generator, tmp_path):
        """Preview file list matches actual lift-only render output."""
        config = load_test_config(config_name)
        expected = set(_preview_files(config, "lift-only"))

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        actual = set(generator.render(config, output_dir, "lift-only"))

        assert expected == actual, (
            f"Dry-run parity mismatch for {config_name} lift-only.\n"
            f"  Only in preview: {expected - actual}\n"
            f"  Only in actual: {actual - expected}"
        )

    @pytest.mark.parametrize("config_name", FRONTEND_CAPABLE_CONFIGS)
    def test_frontend_parity(self, config_name, load_test_config, generator, tmp_path):
        """Preview file list matches actual frontend render output exactly."""
        config = load_test_config(config_name)
        expected = set(_preview_files(config, "frontend"))

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        actual = set(generator.render(config, output_dir, "frontend"))

        assert expected == actual, (
            f"Parity mismatch for {config_name} frontend.\n"
            f"  Only in preview: {expected - actual}\n"
            f"  Only in render:  {actual - expected}"
        )

    @pytest.mark.parametrize("config_name", FRONTEND_CAPABLE_CONFIGS)
    def test_full_parity(self, config_name, load_test_config, generator, tmp_path):
        """Preview file list matches actual full render output exactly."""
        config = load_test_config(config_name)
        expected = set(_preview_files(config, "full"))

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        actual = set(generator.render(config, output_dir, "full"))

        assert expected == actual, (
            f"Parity mismatch for {config_name} full.\n"
            f"  Only in preview: {expected - actual}\n"
            f"  Only in render:  {actual - expected}"
        )


# ---------------------------------------------------------------------------
# Task 3.1.3: Descriptor lifting additions content
# ---------------------------------------------------------------------------


class TestDescriptorLiftingAdditions:
    """Test _render_descriptor_lifting_additions content."""

    def test_hpp_section_markers(self, convolution_fwd_config, generator):
        """Output contains HPP section markers with class name."""
        output = generator._render_descriptor_lifting_additions(convolution_fwd_config)
        cn = convolution_fwd_config.class_name
        assert f"{cn}.hpp" in output
        assert "Add these to the class declaration" in output

    def test_from_node_declaration(self, convolution_fwd_config, generator):
        """Output contains fromNode static method declaration."""
        output = generator._render_descriptor_lifting_additions(convolution_fwd_config)
        cn = convolution_fwd_config.class_name
        assert f"static std::shared_ptr<{cn}>" in output
        assert "fromNode" in output
        assert "const std::unordered_map<int64_t" in output
        assert "std::shared_ptr<TensorDescriptor>" in output

    def test_name_member_declaration(self, convolution_fwd_config, generator):
        """Output contains _name member declaration."""
        output = generator._render_descriptor_lifting_additions(convolution_fwd_config)
        assert "std::string _name;" in output

    def test_operation_name_ext_handling(self, convolution_fwd_config, generator):
        """Output contains HIPDNN_ATTR_OPERATION_NAME_EXT in setAttribute/getAttribute."""
        output = generator._render_descriptor_lifting_additions(convolution_fwd_config)
        assert "HIPDNN_ATTR_OPERATION_NAME_EXT" in output
        assert "setString(_name" in output
        assert "getString(_name" in output

    def test_from_node_implementation_body(self, convolution_fwd_config, generator):
        """Output contains extracted fromNode() implementation."""
        output = generator._render_descriptor_lifting_additions(convolution_fwd_config)
        cn = convolution_fwd_config.class_name
        assert f"std::shared_ptr<{cn}> {cn}::fromNode(" in output

    def test_packer_update_reminder(self, convolution_fwd_config, generator):
        """Output contains packer update reminder."""
        output = generator._render_descriptor_lifting_additions(convolution_fwd_config)
        assert "Packer Update Required" in output
        assert "packer_name_addition.txt" in output
        assert "finalizeDescriptor()" in output

    def test_graph_descriptor_test_reminder(self, convolution_fwd_config, generator):
        """Output contains graph descriptor name tests note."""
        output = generator._render_descriptor_lifting_additions(convolution_fwd_config)
        assert "Graph Descriptor Name Tests" in output
        assert "OperationNamePreservedInSerialization" in output
        assert "OperationNameRoundTripThroughLifting" in output

    def test_operation_type_enum_present_when_set(
        self, convolution_fwd_config, generator
    ):
        """When operation_type_enum is set, HIPDNN_ATTR_OPERATION_TYPE_EXT is in output."""
        assert (
            convolution_fwd_config.operation_type_enum
        ), "convolution_fwd should have operation_type_enum set"
        output = generator._render_descriptor_lifting_additions(convolution_fwd_config)
        assert "HIPDNN_ATTR_OPERATION_TYPE_EXT" in output
        assert convolution_fwd_config.operation_type_enum in output

    def test_operation_type_enum_absent_when_unset(self, generator):
        """When operation_type_enum is not set, HIPDNN_ATTR_OPERATION_TYPE_EXT is absent."""
        from tests.helpers import make_minimal_config

        config = make_minimal_config(operation_type_enum="")
        output = generator._render_descriptor_lifting_additions(config)
        assert "HIPDNN_ATTR_OPERATION_TYPE_EXT" not in output

    def test_cpp_section_markers(self, convolution_fwd_config, generator):
        """Output contains CPP section markers."""
        output = generator._render_descriptor_lifting_additions(convolution_fwd_config)
        cn = convolution_fwd_config.class_name
        assert f"{cn}.cpp" in output
        assert "Add these changes" in output

    def test_build_node_name_assignment(self, convolution_fwd_config, generator):
        """Output contains node->name = _name assignment."""
        output = generator._render_descriptor_lifting_additions(convolution_fwd_config)
        assert "node->name = _name;" in output

    def test_to_string_name_output(self, convolution_fwd_config, generator):
        """Output contains toString name output."""
        output = generator._render_descriptor_lifting_additions(convolution_fwd_config)
        assert '"name=" + _name' in output

    def test_unordered_map_include(self, convolution_fwd_config, generator):
        """Output contains #include <unordered_map>."""
        output = generator._render_descriptor_lifting_additions(convolution_fwd_config)
        assert "#include <unordered_map>" in output

    def test_hipdnn_operation_type_include(self, convolution_fwd_config, generator):
        """Output contains HipdnnOperationType.h include."""
        output = generator._render_descriptor_lifting_additions(convolution_fwd_config)
        assert '#include "HipdnnOperationType.h"' in output


# ---------------------------------------------------------------------------
# Task 3.1.4: Mode enum conditional generation
# ---------------------------------------------------------------------------


class TestModeEnumGeneration:
    """Test conditional mode enum file generation."""

    def test_convolution_fwd_generates_mode_enum_files(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Config with generatable_mode_fields produces mode enum files."""
        assert len(convolution_fwd_config.generatable_mode_fields) > 0
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        written = generator.render(convolution_fwd_config, output_dir, "backend")

        # Find mode enum files in the output
        mode_files = [f for f in written if "mode_" in f]
        assert len(mode_files) > 0, "Expected mode enum files in backend output"

        # Verify backend header exists
        backend_headers = [f for f in written if f.startswith("backend/include/")]
        assert len(backend_headers) > 0, "Expected backend include header for mode enum"

    def test_convolution_fwd_backend_header_contains_typedef(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Mode enum backend header contains the enum typedef."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        written = generator.render(convolution_fwd_config, output_dir, "backend")

        # Find the mode enum backend header
        backend_headers = [f for f in written if f.startswith("backend/include/")]
        assert len(backend_headers) > 0

        for header_path in backend_headers:
            content = (output_dir / header_path).read_text()
            assert (
                "typedef" in content or "enum" in content
            ), f"Backend header {header_path} missing enum typedef"

    def test_pointwise_generates_mode_enum_files(
        self, pointwise_config, generator, tmp_path
    ):
        """Pointwise config with enum_def also produces mode enum files."""
        assert len(pointwise_config.generatable_mode_fields) > 0
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        written = generator.render(pointwise_config, output_dir, "backend")

        mode_files = [f for f in written if "mode_" in f]
        assert len(mode_files) > 0

    def test_matmul_no_mode_enum_files(self, matmul_config, generator, tmp_path):
        """Config without mode fields produces no mode enum files."""
        assert len(matmul_config.generatable_mode_fields) == 0
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        written = generator.render(matmul_config, output_dir, "backend")

        mode_files = [f for f in written if "mode_" in f]
        assert (
            len(mode_files) == 0
        ), f"Unexpected mode enum files for matmul: {mode_files}"

    def test_shared_mode_field_no_enum_files(
        self, load_test_config, generator, tmp_path
    ):
        """Config where mode fields are shared (convolution_bwd) produces no mode enum files."""
        config = load_test_config("convolution_bwd.yaml")
        assert (
            len(config.generatable_mode_fields) == 0
        ), "convolution_bwd mode fields should all be shared"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        written = generator.render(config, output_dir, "backend")

        mode_files = [f for f in written if "mode_" in f]
        assert len(mode_files) == 0

    def test_mode_enum_per_field_files(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Each generatable mode field produces 4 output files."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        written = generator.render(convolution_fwd_config, output_dir, "backend")

        for df in convolution_fwd_config.generatable_mode_fields:
            # Backend header
            expected_header = f"backend/include/{df.enum_def.backend_header}"
            assert (
                expected_header in written
            ), f"Missing backend header for mode field {df.name}"

            # Backend plumbing fragment
            expected_backend_frag = f"fragments/mode_backend_plumbing_{df.name}.txt"
            assert expected_backend_frag in written

            # Frontend plumbing fragment
            expected_frontend_frag = f"fragments/mode_frontend_plumbing_{df.name}.txt"
            assert expected_frontend_frag in written

            # Frontend test fragment
            expected_test_frag = f"fragments/mode_frontend_tests_{df.name}.txt"
            assert expected_test_frag in written


# ---------------------------------------------------------------------------
# Task 3.1.5: Generator error handling
# ---------------------------------------------------------------------------


class TestGeneratorErrorHandling:
    """Test error paths in generator."""

    def test_invalid_mode_raises_value_error(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """render() with an unknown mode raises ValueError."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        with pytest.raises(ValueError, match="Unknown render mode"):
            generator.render(convolution_fwd_config, output_dir, "invalid-mode")

    def test_invalid_mode_error_message_lists_valid_modes(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """ValueError message includes the list of valid modes."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        with pytest.raises(ValueError) as exc_info:
            generator.render(convolution_fwd_config, output_dir, "bogus")
        error_msg = str(exc_info.value)
        assert "backend" in error_msg
        assert "frontend" in error_msg
        assert "full" in error_msg
        assert "lift-only" in error_msg

    def test_empty_string_mode_raises_value_error(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """render() with empty string mode raises ValueError."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        with pytest.raises(ValueError, match="Unknown render mode"):
            generator.render(convolution_fwd_config, output_dir, "")

    def test_template_rendering_error_is_runtime_error(self, generator, template_dir):
        """_render_template wraps Jinja2 errors in RuntimeError."""
        # Use a non-existent template name to trigger an error
        from tests.helpers import make_minimal_config

        config = make_minimal_config()
        with pytest.raises(RuntimeError, match="Failed to render template"):
            generator._render_template("nonexistent_template_xyz.j2", config)


# ---------------------------------------------------------------------------
# Task 3.1.6: Template output content verification
# ---------------------------------------------------------------------------


class TestTemplateOutputContent:
    """Verify generated files contain expected content."""

    def test_backend_descriptor_hpp_content(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Backend descriptor .hpp contains class name, pragma once, generated include."""
        config = convolution_fwd_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        hpp_path = (
            output_dir / "backend" / "src" / "descriptors" / config.header_filename
        )
        content = hpp_path.read_text()

        assert "#pragma once" in content
        assert config.class_name in content
        assert config.fbs_generated_header in content

    def test_backend_descriptor_cpp_content(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Backend descriptor .cpp contains class name and implementation markers."""
        config = convolution_fwd_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        cpp_path = (
            output_dir / "backend" / "src" / "descriptors" / config.source_filename
        )
        content = cpp_path.read_text()

        assert config.class_name in content
        assert "setAttribute" in content
        assert "getAttribute" in content

    def test_packer_hpp_content(self, convolution_fwd_config, generator, tmp_path):
        """Packer .hpp contains packer function name."""
        config = convolution_fwd_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        packer_path = (
            output_dir
            / "frontend"
            / "include"
            / "hipdnn_frontend"
            / "detail"
            / config.packer_filename
        )
        content = packer_path.read_text()

        assert config.frontend.packer_function in content
        assert "#pragma once" in content

    def test_unpacker_hpp_content(self, convolution_fwd_config, generator, tmp_path):
        """Unpacker .hpp contains unpacker function name."""
        config = convolution_fwd_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        unpacker_path = (
            output_dir
            / "frontend"
            / "include"
            / "hipdnn_frontend"
            / "detail"
            / config.unpacker_filename
        )
        content = unpacker_path.read_text()

        assert config.frontend.unpacker_function in content
        assert "#pragma once" in content

    def test_test_descriptor_cpp_content(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Test descriptor .cpp contains TEST_F and class name."""
        config = convolution_fwd_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        test_path = (
            output_dir
            / "backend"
            / "tests"
            / "descriptors"
            / config.test_descriptor_filename
        )
        content = test_path.read_text()

        assert "TEST_F" in content
        assert config.class_name in content

    def test_test_graph_ops_content(self, convolution_fwd_config, generator, tmp_path):
        """Graph operations test contains TEST_F and operation verification."""
        config = convolution_fwd_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        test_path = (
            output_dir
            / "backend"
            / "tests"
            / "descriptors"
            / config.test_graph_filename
        )
        content = test_path.read_text()

        assert "TEST_F" in content

    def test_test_from_node_content(self, convolution_fwd_config, generator, tmp_path):
        """fromNode test contains TEST_F and fixture class."""
        config = convolution_fwd_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        test_path = (
            output_dir
            / "backend"
            / "tests"
            / "descriptors"
            / config.test_from_node_filename
        )
        content = test_path.read_text()

        assert "TEST_F" in content
        assert "fromNode" in content

    def test_integration_lowering_content(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Integration lowering test contains fixture class and TEST_F."""
        config = convolution_fwd_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        test_path = output_dir / "tests" / "frontend" / config.test_integration_filename
        content = test_path.read_text()

        assert "TEST_F" in content
        assert "RoundTrip" in content

    def test_integration_lifting_content(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Integration lifting test contains TestableGraph and fromBackendDescriptor."""
        config = convolution_fwd_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        test_path = (
            output_dir / "tests" / "frontend" / config.test_integration_lifting_filename
        )
        content = test_path.read_text()

        assert "TEST_F" in content
        assert "fromBackendDescriptor" in content or "fromNode" in content

    def test_frontend_attributes_hpp_content(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Frontend attributes header contains attributes class name."""
        config = convolution_fwd_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "frontend")

        attr_path = (
            output_dir
            / "frontend"
            / "include"
            / "hipdnn_frontend"
            / "attributes"
            / config.attributes_header_filename
        )
        content = attr_path.read_text()

        assert "#pragma once" in content
        assert config.frontend.attributes_class in content

    def test_frontend_node_hpp_content(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Frontend node header contains node class name."""
        config = convolution_fwd_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "frontend")

        node_path = (
            output_dir
            / "frontend"
            / "include"
            / "hipdnn_frontend"
            / "node"
            / config.node_header_filename
        )
        content = node_path.read_text()

        assert "#pragma once" in content
        assert config.frontend.node_class in content

    def test_frontend_test_attributes_content(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Frontend attributes test contains gtest macros and attributes class."""
        config = convolution_fwd_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "frontend")

        test_path = output_dir / "frontend" / "tests" / config.test_attributes_filename
        content = test_path.read_text()

        # Templates may use TEST() or TEST_F() depending on the test pattern
        assert "TEST" in content
        assert config.frontend.attributes_class in content

    def test_frontend_test_node_content(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Frontend node test contains gtest macros and node class."""
        config = convolution_fwd_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "frontend")

        test_path = output_dir / "frontend" / "tests" / config.test_node_filename
        content = test_path.read_text()

        # Templates may use TEST() or TEST_F() depending on the test pattern
        assert "TEST" in content
        assert config.frontend.node_class in content

    def test_fragment_cmake_entries_content(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """CMake entries fragment contains source file names."""
        config = convolution_fwd_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        cmake_path = output_dir / "fragments" / "cmake_entries.txt"
        content = cmake_path.read_text()

        # Should reference at least some of the generated files
        assert config.source_filename in content or config.class_name in content

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_all_configs_backend_descriptor_has_class_name(
        self, config_name, load_test_config, generator, tmp_path
    ):
        """Every config's backend descriptor header contains its class name."""
        config = load_test_config(config_name)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        hpp_path = (
            output_dir / "backend" / "src" / "descriptors" / config.header_filename
        )
        content = hpp_path.read_text()
        assert config.class_name in content

    @pytest.mark.parametrize("config_name", ALL_CONFIG_NAMES)
    def test_all_configs_packer_has_function_name(
        self, config_name, load_test_config, generator, tmp_path
    ):
        """Every config's packer header contains the packer function name."""
        config = load_test_config(config_name)
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        packer_path = (
            output_dir
            / "frontend"
            / "include"
            / "hipdnn_frontend"
            / "detail"
            / config.packer_filename
        )
        content = packer_path.read_text()
        assert config.frontend.packer_function in content


# ---------------------------------------------------------------------------
# Additional: render_mode_enums and _render_mode_template coverage
# ---------------------------------------------------------------------------


class TestRenderModeEnums:
    """Directly test render_mode_enums method."""

    def test_render_mode_enums_returns_files(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """render_mode_enums produces 4 files per generatable mode field."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        written = generator.render_mode_enums(convolution_fwd_config, output_dir)

        expected_count = len(convolution_fwd_config.generatable_mode_fields) * 4
        assert len(written) == expected_count

    def test_render_mode_enums_empty_for_matmul(
        self, matmul_config, generator, tmp_path
    ):
        """render_mode_enums returns empty list when no generatable fields."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        written = generator.render_mode_enums(matmul_config, output_dir)
        assert written == []


# ---------------------------------------------------------------------------
# Additional: DescriptorGenerator initialization
# ---------------------------------------------------------------------------


class TestGeneratorInit:
    """Test DescriptorGenerator.__init__() Jinja2 environment setup."""

    def test_generator_creates_jinja_environment(self, template_dir):
        """Generator initializes a Jinja2 Environment with the template directory."""
        gen = DescriptorGenerator(template_dir)
        assert gen.env is not None

    def test_generator_env_keeps_trailing_newline(self, template_dir):
        """Jinja2 environment is configured to keep trailing newlines."""
        gen = DescriptorGenerator(template_dir)
        assert gen.env.keep_trailing_newline is True

    def test_generator_env_trims_blocks(self, template_dir):
        """Jinja2 environment is configured to trim blocks."""
        gen = DescriptorGenerator(template_dir)
        assert gen.env.trim_blocks is True

    def test_generator_env_lstrips_blocks(self, template_dir):
        """Jinja2 environment is configured to left-strip blocks."""
        gen = DescriptorGenerator(template_dir)
        assert gen.env.lstrip_blocks is True


# ---------------------------------------------------------------------------
# Additional: Direct render method tests for maximum coverage
# ---------------------------------------------------------------------------


class TestDirectRenderMethods:
    """Test individual render methods directly for coverage."""

    def test_render_backend_returns_list(self, matmul_config, generator, tmp_path):
        """render_backend returns a list of file paths."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = generator.render_backend(matmul_config, output_dir)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_render_frontend_returns_list(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """render_frontend returns a list of file paths."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = generator.render_frontend(convolution_fwd_config, output_dir)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_render_full_returns_list(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """render_full returns a list of file paths."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = generator.render_full(convolution_fwd_config, output_dir)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_render_lift_only_returns_list(self, matmul_config, generator, tmp_path):
        """render_lift_only returns a list of file paths."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = generator.render_lift_only(matmul_config, output_dir)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_render_backend_file_count(self, matmul_config, generator, tmp_path):
        """render_backend for matmul (no mode enums) produces exactly 19 files."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = generator.render_backend(matmul_config, output_dir)
        # 9 file templates + 10 fragment templates = 19
        assert len(result) == 19

    def test_render_lift_only_file_count(self, matmul_config, generator, tmp_path):
        """render_lift_only produces exactly 11 outputs (3 files + 7 fragments + 1 additions)."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = generator.render_lift_only(matmul_config, output_dir)
        # 3 lift templates + 6 lift fragments + 1 descriptor_lifting_additions = 10
        assert len(result) == 10

    def test_render_frontend_file_count(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """render_frontend produces exactly 9 outputs (2 files + 3 tests + 4 fragments)."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = generator.render_frontend(convolution_fwd_config, output_dir)
        # 2 file templates + 3 test templates + 4 fragment templates = 9
        assert len(result) == 9

    def test_render_dispatches_correctly(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """render() dispatches to the correct method for each mode."""
        for mode in ["backend", "frontend", "full", "lift-only"]:
            output_dir = tmp_path / f"output_{mode}"
            output_dir.mkdir()
            result = generator.render(convolution_fwd_config, output_dir, mode)
            assert isinstance(result, list)
            assert len(result) > 0


# ---------------------------------------------------------------------------
# Additional: _render_template direct coverage
# ---------------------------------------------------------------------------


class TestRenderTemplate:
    """Test _render_template directly."""

    def test_render_template_returns_string(self, convolution_fwd_config, generator):
        """_render_template returns rendered string content."""
        result = generator._render_template("descriptor.hpp.j2", convolution_fwd_config)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_render_template_includes_config_data(
        self, convolution_fwd_config, generator
    ):
        """Rendered template includes data from config."""
        result = generator._render_template("descriptor.hpp.j2", convolution_fwd_config)
        assert convolution_fwd_config.class_name in result

    def test_render_mode_template_returns_string(
        self, convolution_fwd_config, generator
    ):
        """_render_mode_template returns rendered string content."""
        mode_fields = convolution_fwd_config.generatable_mode_fields
        if mode_fields:
            result = generator._render_mode_template(
                "mode_backend_header.j2", convolution_fwd_config, mode_fields[0]
            )
            assert isinstance(result, str)
            assert len(result) > 0

    def test_render_mode_template_error_wraps_in_runtime_error(
        self, convolution_fwd_config, generator
    ):
        """_render_mode_template wraps errors in RuntimeError with field info."""
        from tests.helpers import make_data_field

        df = make_data_field(name="test_field")
        with pytest.raises(RuntimeError, match="mode field 'test_field'"):
            generator._render_mode_template(
                "nonexistent_mode_template.j2", convolution_fwd_config, df
            )


# ---------------------------------------------------------------------------
# Optional tensor and shared utility template output verification
# ---------------------------------------------------------------------------


class TestOptionalTensorTemplateOutput:
    """Verify template changes for optional tensor support and shared utilities."""

    # --- Packer tests ---

    def test_packer_optional_tensors_use_optional_ref(
        self, pointwise_config, generator, tmp_path
    ):
        """Packer uses ensureAndSetOptionalTensorRef for optional tensors."""
        config = pointwise_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        packer_path = (
            output_dir
            / "frontend"
            / "include"
            / "hipdnn_frontend"
            / "detail"
            / config.packer_filename
        )
        content = packer_path.read_text()
        assert "ensureAndSetOptionalTensorRef" in content

    def test_packer_required_tensors_use_required_ref(
        self, pointwise_config, generator, tmp_path
    ):
        """Packer uses ensureAndSetTensorRef (not Optional) for required tensors."""
        config = pointwise_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        packer_path = (
            output_dir
            / "frontend"
            / "include"
            / "hipdnn_frontend"
            / "detail"
            / config.packer_filename
        )
        content = packer_path.read_text()
        # Find lines with ensureAndSetTensorRef that are NOT ensureAndSetOptionalTensorRef
        lines = content.split("\n")
        has_required_ref = any(
            "ensureAndSetTensorRef" in line
            and "ensureAndSetOptionalTensorRef" not in line
            for line in lines
        )
        assert (
            has_required_ref
        ), "Expected ensureAndSetTensorRef for required tensors (in_0, out_0)"

    def test_packer_all_required_no_optional_ref(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Config with all required tensors does NOT contain ensureAndSetOptionalTensorRef."""
        config = convolution_fwd_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        packer_path = (
            output_dir
            / "frontend"
            / "include"
            / "hipdnn_frontend"
            / "detail"
            / config.packer_filename
        )
        content = packer_path.read_text()
        assert "ensureAndSetOptionalTensorRef" not in content

    # --- Descriptor setAttribute tests ---

    def test_descriptor_set_optional_tensor(
        self, pointwise_config, generator, tmp_path
    ):
        """Descriptor.cpp uses setOptionalTensorDescriptor for optional tensors."""
        config = pointwise_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        cpp_path = (
            output_dir / "backend" / "src" / "descriptors" / config.source_filename
        )
        content = cpp_path.read_text()
        assert "setOptionalTensorDescriptor" in content

    def test_descriptor_set_required_tensor(
        self, pointwise_config, generator, tmp_path
    ):
        """Descriptor.cpp uses setTensorDescriptor (not Optional) for required tensors."""
        config = pointwise_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        cpp_path = (
            output_dir / "backend" / "src" / "descriptors" / config.source_filename
        )
        content = cpp_path.read_text()
        lines = content.split("\n")
        has_required_set = any(
            "setTensorDescriptor" in line
            and "setOptionalTensorDescriptor" not in line
            and "setTensorDescriptorArray" not in line
            for line in lines
        )
        assert (
            has_required_set
        ), "Expected setTensorDescriptor for required tensors (in_0, out_0)"

    # --- Descriptor getAttribute tests ---

    def test_descriptor_get_optional_tensor(
        self, pointwise_config, generator, tmp_path
    ):
        """Descriptor.cpp uses getOptionalTensorDescriptor for optional tensors."""
        config = pointwise_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        cpp_path = (
            output_dir / "backend" / "src" / "descriptors" / config.source_filename
        )
        content = cpp_path.read_text()
        assert "getOptionalTensorDescriptor" in content

    # --- getTensorDescriptors() tests ---

    def test_descriptor_get_tensors_guards_optional(
        self, pointwise_config, generator, tmp_path
    ):
        """getTensorDescriptors() guards optional tensors with null checks."""
        config = pointwise_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        cpp_path = (
            output_dir / "backend" / "src" / "descriptors" / config.source_filename
        )
        content = cpp_path.read_text()
        assert "if(_in1Desc)" in content

    # --- toString() tests ---

    def test_descriptor_to_string_optional_uid(
        self, pointwise_config, generator, tmp_path
    ):
        """toString() uses nullopt pattern for optional tensor UIDs."""
        config = pointwise_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        cpp_path = (
            output_dir / "backend" / "src" / "descriptors" / config.source_filename
        )
        content = cpp_path.read_text()
        assert "nullopt" in content

    def test_descriptor_to_string_optional_scalar(
        self, pointwise_config, generator, tmp_path
    ):
        """toString() uses nullopt pattern for optional scalar fields."""
        config = pointwise_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        cpp_path = (
            output_dir / "backend" / "src" / "descriptors" / config.source_filename
        )
        content = cpp_path.read_text()
        # Pointwise has optional scalars like relu_lower_clip
        # The toString should have both the nullopt pattern and std::to_string for them
        # Count occurrences of the optional scalar pattern
        optional_scalar_count = content.count("? std::to_string(*_data.")
        # Pointwise has 7 optional scalar fields (relu_lower_clip, relu_upper_clip,
        # relu_lower_clip_slope, swish_beta, elu_alpha, softplus_beta, axis)
        assert (
            optional_scalar_count >= 1
        ), "Expected at least one optional scalar with nullopt pattern in toString()"

    # --- Vector field (setScalarVector / getScalarVector) tests ---

    def test_descriptor_vectors_use_scalar_vector(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Descriptor.cpp uses setScalarVector<int64_t> and getScalarVector<int64_t>."""
        config = convolution_fwd_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        cpp_path = (
            output_dir / "backend" / "src" / "descriptors" / config.source_filename
        )
        content = cpp_path.read_text()
        assert "setScalarVector<int64_t>" in content
        assert "getScalarVector<int64_t>" in content

    def test_descriptor_vectors_no_int64_vector(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Descriptor.cpp does NOT contain setInt64Vector or getInt64Vector."""
        config = convolution_fwd_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        cpp_path = (
            output_dir / "backend" / "src" / "descriptors" / config.source_filename
        )
        content = cpp_path.read_text()
        assert "setInt64Vector" not in content
        assert "getInt64Vector" not in content

    # --- Tensor array shared utility tests ---

    def test_descriptor_tensor_array_uses_shared_utils(
        self, batchnorm_config, generator, tmp_path
    ):
        """Descriptor.cpp uses setTensorDescriptorArray and getTensorDescriptorArray."""
        config = batchnorm_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        cpp_path = (
            output_dir / "backend" / "src" / "descriptors" / config.source_filename
        )
        content = cpp_path.read_text()
        assert "setTensorDescriptorArray" in content
        assert "getTensorDescriptorArray" in content

    # --- Optional scalar field (setOptionalScalar / getOptionalScalar) tests ---

    def test_descriptor_optional_scalars_use_set_optional_scalar(
        self, pointwise_config, generator, tmp_path
    ):
        """setAttribute uses setOptionalScalar for optional scalar fields."""
        config = pointwise_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        cpp_path = (
            output_dir / "backend" / "src" / "descriptors" / config.source_filename
        )
        content = cpp_path.read_text()
        assert "setOptionalScalar<" in content

    def test_descriptor_optional_scalars_use_get_optional_scalar(
        self, pointwise_config, generator, tmp_path
    ):
        """getAttribute uses getOptionalScalar for optional scalar fields."""
        config = pointwise_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        cpp_path = (
            output_dir / "backend" / "src" / "descriptors" / config.source_filename
        )
        content = cpp_path.read_text()
        assert "getOptionalScalar<" in content

    def test_descriptor_required_scalars_no_optional_scalar(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Config with no optional scalars does NOT contain setOptionalScalar."""
        config = convolution_fwd_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        cpp_path = (
            output_dir / "backend" / "src" / "descriptors" / config.source_filename
        )
        content = cpp_path.read_text()
        assert "setOptionalScalar" not in content
        assert "getOptionalScalar" not in content

    # --- fromNode() dereference tests ---

    def test_descriptor_from_node_dereferences_optional_uid(
        self, pointwise_config, generator, tmp_path
    ):
        """fromNode() dereferences optional tensor UIDs with *."""
        config = pointwise_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        cpp_path = (
            output_dir / "backend" / "src" / "descriptors" / config.source_filename
        )
        content = cpp_path.read_text()
        # Optional tensor UIDs must be dereferenced with * in findTensorInMap calls
        assert "*attrs->in_1_tensor_uid" in content
        assert "*attrs->in_2_tensor_uid" in content


# ---------------------------------------------------------------------------
# Constants file generation
# ---------------------------------------------------------------------------


class TestConstantsGeneration:
    """Test that the constants header file is generated correctly."""

    def test_constants_generated_when_not_set(
        self, load_test_config, generator, tmp_path
    ):
        """Constants file IS generated for configs without constants_include."""
        config = load_test_config("batchnorm_backward.yaml")
        assert not config.test_data.constants_include

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        written = generator.render(config, output_dir, "backend")

        constants_path = (
            f"test_sdk/include/hipdnn_test_sdk/constants/"
            f"{config.effective_constants_include}.hpp"
        )
        assert constants_path in written

        # Verify the file exists and has correct content
        full_path = output_dir / constants_path
        assert full_path.exists()
        content = full_path.read_text()
        assert "#pragma once" in content
        assert "namespace hipdnn_tests::constants" in content
        prefix = config.tensor_const_prefix
        assert f"{prefix}TENSOR_DY_UID" in content
        assert f"{prefix}TENSOR_X_UID" in content

    def test_constants_not_generated_when_set(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Constants file is NOT generated for configs with constants_include."""
        assert (
            convolution_fwd_config.test_data.constants_include == "ConvFpropConstants"
        )

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        written = generator.render(convolution_fwd_config, output_dir, "backend")

        constants_files = [f for f in written if "constants/" in f]
        assert len(constants_files) == 0

    def test_constants_generated_in_lift_only_mode(
        self, load_test_config, generator, tmp_path
    ):
        """Constants file IS generated in lift-only mode when not set."""
        config = load_test_config("batchnorm_backward.yaml")
        assert not config.test_data.constants_include

        output_dir = tmp_path / "output"
        output_dir.mkdir()
        written = generator.render(config, output_dir, "lift-only")

        constants_path = (
            f"test_sdk/include/hipdnn_test_sdk/constants/"
            f"{config.effective_constants_include}.hpp"
        )
        assert constants_path in written

    def test_constants_not_generated_in_lift_only_when_set(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Constants file is NOT generated in lift-only mode when set."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        written = generator.render(convolution_fwd_config, output_dir, "lift-only")

        constants_files = [f for f in written if "constants/" in f]
        assert len(constants_files) == 0

    def test_constants_has_tensor_uids(self, load_test_config, generator, tmp_path):
        """Generated constants include UID constants for all tensors."""
        config = load_test_config("batchnorm_inference.yaml")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        constants_path = (
            output_dir
            / "test_sdk/include/hipdnn_test_sdk/constants"
            / f"{config.effective_constants_include}.hpp"
        )
        content = constants_path.read_text()

        prefix = config.tensor_const_prefix
        for tf in config.tensor_fields:
            uid = config.test_data.tensor_uids.get(tf.name, 0)
            assert f"{prefix}TENSOR_{tf.name.upper()}_UID = {uid}" in content

    def test_constants_has_dims_and_strides(
        self, load_test_config, generator, tmp_path
    ):
        """Generated constants include dims/strides for tensors with configs."""
        config = load_test_config("batchnorm_inference.yaml")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        constants_path = (
            output_dir
            / "test_sdk/include/hipdnn_test_sdk/constants"
            / f"{config.effective_constants_include}.hpp"
        )
        content = constants_path.read_text()

        prefix = config.tensor_const_prefix
        assert f"{prefix}TENSOR_X_DIMS" in content
        assert f"{prefix}TENSOR_X_STRIDES" in content

    def test_all_test_files_reference_constants_header(
        self, load_test_config, generator, tmp_path
    ):
        """All generated test files include the constants header."""
        config = load_test_config("batchnorm_inference.yaml")
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(config, output_dir, "backend")

        expected_include = config.effective_constants_include

        test_files = [
            output_dir / "backend/tests/descriptors" / config.test_descriptor_filename,
            output_dir / "backend/tests/descriptors" / config.test_graph_filename,
            output_dir / "backend/tests/descriptors" / config.test_from_node_filename,
            output_dir / "tests/frontend" / config.test_integration_filename,
            output_dir / "tests/frontend" / config.test_integration_lifting_filename,
        ]

        for test_file in test_files:
            content = test_file.read_text()
            assert (
                expected_include in content
            ), f"{test_file.name} does not reference {expected_include}"


class TestLiftingTemplateImprovements:
    """Test ASSERT_EQ and auto-UIDs in lifting template."""

    def test_lifting_uses_assert_eq_not_ge(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Lifting test uses ASSERT_EQ for tensorMap.size(), not ASSERT_GE."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(convolution_fwd_config, output_dir, "backend")

        lifting_path = (
            output_dir
            / "tests/frontend"
            / convolution_fwd_config.test_integration_lifting_filename
        )
        content = lifting_path.read_text()

        assert "ASSERT_GE(tensorMap.size()" not in content
        assert "ASSERT_EQ(tensorMap.size()" in content

    def test_lifting_has_auto_uid_test(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Lifting test includes AutoAssignedUidsPreservedInLiftingRoundTrip."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(convolution_fwd_config, output_dir, "backend")

        lifting_path = (
            output_dir
            / "tests/frontend"
            / convolution_fwd_config.test_integration_lifting_filename
        )
        content = lifting_path.read_text()

        assert "AutoAssignedUidsPreservedInLiftingRoundTrip" in content
        assert "std::adjacent_find" in content
        assert "std::set<int64_t>" in content

    def test_lifting_includes_algorithm_and_set(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Lifting test includes <algorithm> and <set> headers."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(convolution_fwd_config, output_dir, "backend")

        lifting_path = (
            output_dir
            / "tests/frontend"
            / convolution_fwd_config.test_integration_lifting_filename
        )
        content = lifting_path.read_text()

        assert "#include <algorithm>" in content
        assert "#include <set>" in content

    def test_lifting_uses_constants_not_inline(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Lifting test uses K_TENSOR_ constants, not K_TEST_ inline constants."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(convolution_fwd_config, output_dir, "backend")

        lifting_path = (
            output_dir
            / "tests/frontend"
            / convolution_fwd_config.test_integration_lifting_filename
        )
        content = lifting_path.read_text()

        assert "K_TEST_" not in content
        prefix = convolution_fwd_config.tensor_const_prefix
        assert f"{prefix}TENSOR_X_UID" in content
        assert "using namespace hipdnn_tests::constants;" in content

    def test_lowering_uses_constants_not_inline(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Lowering test uses K_TENSOR_ constants, not K_TEST_ inline constants."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(convolution_fwd_config, output_dir, "backend")

        lowering_path = (
            output_dir
            / "tests/frontend"
            / convolution_fwd_config.test_integration_filename
        )
        content = lowering_path.read_text()

        assert "K_TEST_" not in content
        prefix = convolution_fwd_config.tensor_const_prefix
        assert f"{prefix}TENSOR_X_UID" in content
        assert "using namespace hipdnn_tests::constants;" in content

    def test_lowering_uses_assert_eq_not_ge(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Lowering test uses ASSERT_EQ for tensor count, not ASSERT_GE."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(convolution_fwd_config, output_dir, "backend")

        lowering_path = (
            output_dir
            / "tests/frontend"
            / convolution_fwd_config.test_integration_filename
        )
        content = lowering_path.read_text()

        assert "ASSERT_GE(graphT.tensors.size()" not in content
        assert "ASSERT_EQ(graphT.tensors.size()" in content
