# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Integration tests for the descriptor codegen pipeline.

Tests exercise the full pipeline from config loading through template rendering,
covering generator.py render methods and generate.py's _preview_files function.
"""

import pytest
from pathlib import Path

from codegen.config_loader import load_config, validate_for_mode
from codegen.generator import BACKEND_FRAGMENT_FILENAMES, DescriptorGenerator
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
    modes = ["backend"]
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

    def test_convolution_fwd_all_modes(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """convolution_fwd renders successfully in all supported modes."""
        config = convolution_fwd_config
        for mode in ["backend", "frontend", "full"]:
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

    # File templates emitted by render_backend (paths are relative to output_dir).
    # Pinned by name so a future template addition/removal is forced through this
    # list instead of a magic count drifting silently.
    _BACKEND_FILE_TEMPLATE_BASENAMES = (
        "descriptor.hpp",
        "descriptor.cpp",
        "packer.hpp",
        "test_descriptor.cpp",
        "test_graph_ops.cpp",
        "test_from_node.cpp",
        "test_integration.cpp",
        "test_integration_lifting.cpp",
        "unpacker.hpp",
    )

    # File templates emitted by render_frontend (basenames, before per-config
    # filename interpolation).
    _FRONTEND_FILE_TEMPLATE_BASENAMES = (
        "attributes.hpp",
        "node.hpp",
        "test_attributes.cpp",
        "test_node.cpp",
        "test_frontend_graph.cpp",
    )

    _FRONTEND_FRAGMENT_FILENAMES = (
        "graph_method.txt",
        "graph_includes.txt",
        "frontend_cmake_entries.txt",
        "node_type_enum.txt",
    )

    def test_render_backend_file_set(self, matmul_config, generator, tmp_path):
        """render_backend for matmul (no mode enums) produces exactly the
        expected set of files: one per file-template, one per backend
        fragment, and the descriptor_lifting_additions text. Compared by
        name-set so a template add/remove fails the test descriptively."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = generator.render_backend(matmul_config, output_dir)

        result_basenames = {Path(p).name for p in result}
        expected_file_basenames = {
            Path(getattr(matmul_config, attr)).name
            for attr in (
                "header_filename",
                "source_filename",
                "packer_filename",
                "test_descriptor_filename",
                "test_graph_filename",
                "test_from_node_filename",
                "test_integration_filename",
                "test_integration_lifting_filename",
                "unpacker_filename",
            )
        }
        expected = expected_file_basenames | set(BACKEND_FRAGMENT_FILENAMES)

        assert result_basenames == expected
        # Sanity: file-template count matches the canonical basename list so
        # edits to render_backend's file_templates dict can't silently drop one.
        assert len(self._BACKEND_FILE_TEMPLATE_BASENAMES) == len(
            expected_file_basenames
        )

    def test_render_frontend_file_set(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """render_frontend produces exactly the expected file + fragment set,
        compared by name."""
        config = convolution_fwd_config
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = generator.render_frontend(config, output_dir)

        result_basenames = {Path(p).name for p in result}
        expected_file_basenames = {
            Path(getattr(config, attr)).name
            for attr in (
                "attributes_header_filename",
                "node_header_filename",
                "test_attributes_filename",
                "test_node_filename",
                "test_frontend_graph_filename",
            )
        }
        expected = expected_file_basenames | set(self._FRONTEND_FRAGMENT_FILENAMES)

        assert result_basenames == expected
        assert len(self._FRONTEND_FILE_TEMPLATE_BASENAMES) == len(
            expected_file_basenames
        )

    def test_render_frontend_honors_attributes_filename_override(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Setting frontend.attributes_filename routes the generated Attributes
        header and the matching test file basenames through the override
        instead of the attributes_class-derived default. Mirrors the model-
        level coverage in test_models.py at the render layer so a regression
        in the basename-plumbing wiring fails here too."""
        config = convolution_fwd_config
        # convolution_fwd ships with attributes_filename='ConvolutionFprop'
        # already set; override to a sentinel value to prove plumbing.
        config.frontend.attributes_filename = "OverrideBaseName"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = generator.render_frontend(config, output_dir)

        basenames = {Path(p).name for p in result}
        assert "OverrideBaseNameAttributes.hpp" in basenames
        assert "OverrideBaseNameNode.hpp" in basenames
        assert "TestOverrideBaseNameAttributes.cpp" in basenames
        assert "TestOverrideBaseNameNode.cpp" in basenames
        assert "TestGraphOverrideBaseName.cpp" in basenames

    def test_render_dispatches_correctly(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """render() dispatches to the correct method for each mode."""
        for mode in ["backend", "frontend", "full"]:
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

    def test_finalize_sentinel_check_uses_sdk_name(self, pointwise_config, generator):
        """The descriptor.cpp finalize() body must compare against the SDK
        sentinel symbol (``sdk_name``), not the bare backend ``name``.

        Pointwise's PointwiseMode enum has ``NOT_SET`` (sentinel:true) with
        ``sdk_name: UNSET``. The rendered finalize check therefore needs to
        read ``PointwiseMode::UNSET`` so it matches what the SDK enum
        actually exports — emitting ``::NOT_SET`` would compile against the
        SDK header but compare a never-equal value, masking real
        finalize() failures.
        """
        rendered = generator._render_template("descriptor.cpp.j2", pointwise_config)
        assert "PointwiseMode::UNSET" in rendered
        assert "PointwiseMode::NOT_SET" not in rendered


# ---------------------------------------------------------------------------
# tensor_fields[].frontend_getter override — render-level verification
# ---------------------------------------------------------------------------


class TestTensorFieldFrontendGetterOverride:
    """Verify that the per-tensor ``frontend_getter`` override resolves
    correctly when packer/unpacker templates render, for the three
    historically-tripping configs (batchnorm, batchnorm_backward, sdpa).

    The original bug emitted ``attributes.()`` (empty getter) in the
    packer and ``attributes.set_(`` (empty setter) in the unpacker
    whenever a ``tensor_fields[]`` entry had no matching ``frontend.inputs``
    or ``frontend.outputs`` name. With the override re-honored, every
    tensor field now resolves either via name-match, abbreviation
    fallback, or explicit ``frontend_getter``, and the rendered output
    must never contain those broken literals.
    """

    @pytest.mark.parametrize(
        "config_name",
        ["batchnorm.yaml", "batchnorm_backward.yaml", "sdpa.yaml"],
    )
    def test_packer_has_no_empty_attribute_call(
        self, config_name, load_test_config, generator
    ):
        """The packer must never render ``attributes.()`` — that literal
        is the symptom of an unresolved tensor_field accessor.
        """
        config = load_test_config(config_name)
        rendered = generator._render_template("packer.hpp.j2", config)
        assert "attributes.()" not in rendered, (
            f"{config_name}: packer rendered an empty attribute getter "
            f"call — at least one tensor_field failed to resolve to a "
            f"frontend accessor."
        )

    @pytest.mark.parametrize(
        "config_name",
        ["batchnorm.yaml", "batchnorm_backward.yaml", "sdpa.yaml"],
    )
    def test_unpacker_has_no_empty_setter_call(
        self, config_name, load_test_config, generator
    ):
        """The unpacker must never render ``attributes.set_(`` — that
        literal is the symptom of an unresolved tensor_field with a
        getter that does not start with ``get_`` so the setter could
        not be derived.
        """
        config = load_test_config(config_name)
        rendered = generator._render_template("unpacker.hpp.j2", config)
        assert "attributes.set_(" not in rendered, (
            f"{config_name}: unpacker rendered an empty setter call — "
            f"at least one tensor_field failed to derive a setter name."
        )

    def test_sdpa_unpacker_uses_set_bias_for_attn_mask(
        self, load_test_config, generator
    ):
        """SDPA's ``attn_mask`` tensor_field intentionally maps to the
        hand-written ``get_bias()``/``set_bias()`` accessors on
        ``SdpaAttributes``. The override must produce a real
        ``attributes.set_bias(`` call in the unpacker, not an empty
        ``attributes.set_(`` or a ``set_attn_mask(`` call against a
        non-existent setter.
        """
        config = load_test_config("sdpa.yaml")
        rendered = generator._render_template("unpacker.hpp.j2", config)
        assert "attributes.set_bias(" in rendered, (
            "sdpa.yaml: unpacker should emit attributes.set_bias(...) "
            "for the divergent attn_mask → get_bias() override."
        )
        assert "attributes.set_attn_mask(" not in rendered, (
            "sdpa.yaml: unpacker must NOT call set_attn_mask — "
            "SdpaAttributes exposes that tensor via the bias accessor."
        )

    def test_sdpa_packer_emits_max_seq_len_kv_as_int32(
        self, load_test_config, generator
    ):
        """SDPA's ``max_seq_len_kv`` is ``scalar_int32``. The packer must
        emit the field with ``HIPDNN_TYPE_INT32`` and use the attribute
        identifier — never silently drop it because the type is unknown.
        Regression guard: prior to scalar_int32 support, this field
        produced no code in the generated packer.
        """
        config = load_test_config("sdpa.yaml")
        rendered = generator._render_template("packer.hpp.j2", config)
        assert (
            "HIPDNN_ATTR_SDPA_FWD_MAX_SEQ_LEN_KV_EXT" in rendered
        ), "sdpa.yaml: packer must emit max_seq_len_kv attribute"
        assert "HIPDNN_TYPE_INT32" in rendered, (
            "sdpa.yaml: packer must emit HIPDNN_TYPE_INT32 for the "
            "scalar_int32 field"
        )

    def test_sdpa_unpacker_emits_max_seq_len_kv_as_int32(
        self, load_test_config, generator
    ):
        """SDPA's ``max_seq_len_kv`` (scalar_int32, fbs-optional) must
        unpack into a ``std::optional<int32_t>`` and write
        ``HIPDNN_TYPE_INT32`` into the getDescriptorAttrOptionalScalar
        call.
        """
        config = load_test_config("sdpa.yaml")
        rendered = generator._render_template("unpacker.hpp.j2", config)
        assert "std::optional<int32_t>" in rendered, (
            "sdpa.yaml: unpacker must declare an int32_t optional local "
            "for max_seq_len_kv"
        )
        assert "HIPDNN_TYPE_INT32" in rendered, (
            "sdpa.yaml: unpacker must pass HIPDNN_TYPE_INT32 for the "
            "scalar_int32 field"
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


class TestIntegrationFixtureNameHonorsAttributesFilename:
    """Verify Integration{...}Descriptor{Lowering,Lifting} fixture and TEST_F
    suite names use ``frontend.attributes_filename`` when set, falling back to
    ``op.name`` otherwise. The in-tree files for conv use the
    ``IntegrationConvFprop*`` basename rather than ``IntegrationConvolutionFwd*``,
    so the generator must match.
    """

    def test_lowering_fixture_uses_attributes_filename(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """convolution_fwd has attributes_filename='ConvolutionFprop'; the
        lowering fixture must be named after it, not after op.name."""
        assert convolution_fwd_config.frontend.attributes_filename == "ConvolutionFprop"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(convolution_fwd_config, output_dir, "backend")

        lowering_path = (
            output_dir
            / "tests/frontend"
            / convolution_fwd_config.test_integration_filename
        )
        content = lowering_path.read_text()

        assert "class IntegrationConvolutionFpropDescriptorLowering" in content
        assert "TEST_F(IntegrationConvolutionFpropDescriptorLowering," in content
        assert "IntegrationConvolutionFwdDescriptorLowering" not in content

    def test_lifting_fixture_uses_attributes_filename(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """convolution_fwd lifting fixture must be named after attributes_filename."""
        assert convolution_fwd_config.frontend.attributes_filename == "ConvolutionFprop"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(convolution_fwd_config, output_dir, "backend")

        lifting_path = (
            output_dir
            / "tests/frontend"
            / convolution_fwd_config.test_integration_lifting_filename
        )
        content = lifting_path.read_text()

        assert "class IntegrationConvolutionFpropDescriptorLifting" in content
        assert "TEST_F(IntegrationConvolutionFpropDescriptorLifting," in content
        assert "IntegrationConvolutionFwdDescriptorLifting" not in content

    def test_lowering_fixture_falls_back_to_op_name(
        self, pointwise_config, generator, tmp_path
    ):
        """pointwise has no attributes_filename; fixture must use op.name."""
        assert pointwise_config.frontend.attributes_filename is None
        assert pointwise_config.name == "Pointwise"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(pointwise_config, output_dir, "backend")

        lowering_path = (
            output_dir / "tests/frontend" / pointwise_config.test_integration_filename
        )
        content = lowering_path.read_text()

        assert "class IntegrationPointwiseDescriptorLowering" in content
        assert "TEST_F(IntegrationPointwiseDescriptorLowering," in content

    def test_lifting_fixture_falls_back_to_op_name(
        self, pointwise_config, generator, tmp_path
    ):
        """pointwise lifting fixture must use op.name when no override is set."""
        assert pointwise_config.frontend.attributes_filename is None
        assert pointwise_config.name == "Pointwise"
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(pointwise_config, output_dir, "backend")

        lifting_path = (
            output_dir
            / "tests/frontend"
            / pointwise_config.test_integration_lifting_filename
        )
        content = lifting_path.read_text()

        assert "class IntegrationPointwiseDescriptorLifting" in content
        assert "TEST_F(IntegrationPointwiseDescriptorLifting," in content


# ---------------------------------------------------------------------------
# Task 2.1: tensor-array unpacking emit
# ---------------------------------------------------------------------------


class TestUnpackerTensorArray:
    """Verify the unpacker emits a tensor-array block for each tensor_array_field."""

    def test_unpacker_calls_unpack_tensor_array_helper(
        self, batchnorm_config, generator, tmp_path
    ):
        """The generated unpacker invokes unpackAndRegisterTensorArray for peer_stats."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(batchnorm_config, output_dir, "backend")

        unpacker_path = (
            output_dir
            / "frontend/include/hipdnn_frontend/detail"
            / batchnorm_config.unpacker_filename
        )
        content = unpacker_path.read_text()
        assert "unpackAndRegisterTensorArray" in content
        assert "HIPDNN_ATTR_OPERATION_BATCHNORM_PEER_STATS_EXT" in content

    def test_unpacker_calls_frontend_setter_for_tensor_array(
        self, batchnorm_config, generator, tmp_path
    ):
        """The generated unpacker calls attributes.set_peer_stats(...)."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(batchnorm_config, output_dir, "backend")

        unpacker_path = (
            output_dir
            / "frontend/include/hipdnn_frontend/detail"
            / batchnorm_config.unpacker_filename
        )
        content = unpacker_path.read_text()
        assert "attributes.set_peer_stats(" in content

    def test_unpacker_no_tensor_array_block_when_absent(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """Configs without tensor_array_fields should not emit the array block."""
        assert convolution_fwd_config.tensor_array_fields == []
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(convolution_fwd_config, output_dir, "backend")

        unpacker_path = (
            output_dir
            / "frontend/include/hipdnn_frontend/detail"
            / convolution_fwd_config.unpacker_filename
        )
        content = unpacker_path.read_text()
        assert "unpackAndRegisterTensorArray" not in content


# ---------------------------------------------------------------------------
# Task 2.2: camelCase normalization for tensor_array names in packer
# ---------------------------------------------------------------------------


class TestPackerTensorArrayCamelCase:
    """Verify tensor_array_field local variables use camelCase, identifiers stay snake."""

    def test_packer_uses_camel_locals_for_peer_stats(
        self, batchnorm_config, generator, tmp_path
    ):
        """Packer locals (Attrs/Ptrs vars) use peerStats not peer_stats."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(batchnorm_config, output_dir, "backend")

        packer_path = (
            output_dir
            / "frontend/include/hipdnn_frontend/detail"
            / batchnorm_config.packer_filename
        )
        content = packer_path.read_text()
        assert "peerStatsAttrs" in content
        assert "peerStatsPtrs" in content
        # The snake_case local variable names must NOT appear
        assert "peer_statsAttrs" not in content
        assert "peer_statsPtrs" not in content

    def test_packer_keeps_snake_for_attribute_identifiers(
        self, batchnorm_config, generator, tmp_path
    ):
        """attr_name and frontend_getter identifiers stay in original snake form."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(batchnorm_config, output_dir, "backend")

        packer_path = (
            output_dir
            / "frontend/include/hipdnn_frontend/detail"
            / batchnorm_config.packer_filename
        )
        content = packer_path.read_text()
        # The attr enum and the frontend getter both keep snake_case
        assert "HIPDNN_ATTR_OPERATION_BATCHNORM_PEER_STATS_EXT" in content
        assert "attributes.get_peer_stats()" in content
        # Error label uses the snake-case taf.name
        assert "batchnorm peer_stats" in content


# ---------------------------------------------------------------------------
# Task 2.3: backend C-API mode-enum header includes in packer
# ---------------------------------------------------------------------------


class TestPackerModeBackendHeader:
    """Verify the packer emits #include for mode-field backend headers."""

    def test_convolution_fwd_packer_includes_mode_backend_header(
        self, convolution_fwd_config, generator, tmp_path
    ):
        """ConvFwd packer must #include HipdnnConvolutionMode.h."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(convolution_fwd_config, output_dir, "backend")

        packer_path = (
            output_dir
            / "frontend/include/hipdnn_frontend/detail"
            / convolution_fwd_config.packer_filename
        )
        content = packer_path.read_text()
        assert '#include "HipdnnConvolutionMode.h"' in content

    def test_batchnorm_packer_no_mode_header(
        self, batchnorm_config, generator, tmp_path
    ):
        """Batchnorm has no mode fields, so no mode backend header include."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(batchnorm_config, output_dir, "backend")

        packer_path = (
            output_dir
            / "frontend/include/hipdnn_frontend/detail"
            / batchnorm_config.packer_filename
        )
        content = packer_path.read_text()
        assert "HipdnnConvolutionMode.h" not in content
        assert "HipdnnPointwiseMode.h" not in content

    def test_mode_field_backend_headers_unique(self, convolution_fwd_config):
        """The derived list deduplicates headers across mode fields."""
        headers = convolution_fwd_config.mode_field_backend_headers
        assert headers == list(dict.fromkeys(headers))
        assert "HipdnnConvolutionMode.h" in headers


class TestTensorArrayFieldCamelCase:
    """Verify tensor_array_field C++ identifiers use camelCase, not snake_case.

    The TensorArrayField.camel_name property (added in Phase 2.2) is used by
    packer.hpp.j2 and unpacker.hpp.j2 for local variable names. These tests
    pin the same convention into the descriptor (.hpp/.cpp), descriptor
    lifting fragment, test descriptor, test graph ops, and test from_node
    templates so that all C++ identifiers derived from a tensor_array_field
    follow the in-tree convention (e.g., `_peerStatsDescs`, not
    `_peer_statsDescs`).

    Snake-case is preserved for non-identifier surfaces: attribute enum
    constants (HIPDNN_ATTR_OPERATION_BATCHNORM_PEER_STATS), error labels
    ("peer_stats"), comments ("// Tensor array: peer_stats"), the FBS
    underlying field (`_data.peer_stats_tensor_uid`), and the toString
    label ("peer_stats_uids").
    """

    def test_descriptor_hpp_member_uses_camel_case(
        self, batchnorm_config, generator, tmp_path
    ):
        """Descriptor private member is `_peerStatsDescs`, not `_peer_statsDescs`."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(batchnorm_config, output_dir, "backend")

        hpp_path = (
            output_dir
            / "backend"
            / "src"
            / "descriptors"
            / batchnorm_config.header_filename
        )
        content = hpp_path.read_text()

        assert (
            "std::vector<std::shared_ptr<TensorDescriptor>> _peerStatsDescs;" in content
        ), "Descriptor .hpp must declare _peerStatsDescs (camelCase) member"
        assert (
            "_peer_statsDescs" not in content
        ), "Descriptor .hpp must not emit snake-case _peer_statsDescs identifier"

    def test_descriptor_cpp_member_uses_camel_case(
        self, batchnorm_config, generator, tmp_path
    ):
        """Descriptor .cpp references _peerStatsDescs in setter, getter, fromNode, etc."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(batchnorm_config, output_dir, "backend")

        cpp_path = (
            output_dir
            / "backend"
            / "src"
            / "descriptors"
            / batchnorm_config.source_filename
        )
        content = cpp_path.read_text()

        assert "_peerStatsDescs" in content
        assert "_peer_statsDescs" not in content
        # Snake-case must remain for FBS field, error label, attribute enum.
        assert "_data.peer_stats_tensor_uid" in content
        assert "HIPDNN_ATTR_OPERATION_BATCHNORM_PEER_STATS" in content
        assert '"BatchnormOperationDescriptor::fromNode: peer_stats"' in content

    def test_lifting_additions_fragment_uses_camel_case(
        self, batchnorm_config, generator, tmp_path
    ):
        """The descriptor_lifting_additions fragment must reference _peerStatsDescs."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(batchnorm_config, output_dir, "backend")

        fragment_path = output_dir / "fragments" / "descriptor_lifting_additions.txt"
        content = fragment_path.read_text()

        assert (
            "desc->_peerStatsDescs.push_back(" in content
        ), "Lifting fragment must use camelCase _peerStatsDescs in fromNode body"
        assert "desc->_peer_statsDescs" not in content

    def test_test_graph_ops_local_var_uses_camel_case(
        self, batchnorm_config, generator, tmp_path
    ):
        """Graph descriptor test fixture local var is peerStatsDescs (no underscore prefix)."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(batchnorm_config, output_dir, "backend")

        test_path = (
            output_dir
            / "backend"
            / "tests"
            / "descriptors"
            / batchnorm_config.test_graph_filename
        )
        content = test_path.read_text()

        # Local function param / local vec is camelCase, no leading underscore.
        assert "peerStatsDescs" in content
        assert "peer_statsDescs" not in content

    def test_test_descriptor_fixture_member_uses_camel_case(
        self, batchnorm_config, generator, tmp_path
    ):
        """Per-element test fixture members are _peerStatsDesc0/1 (camelCase)."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(batchnorm_config, output_dir, "backend")

        test_path = (
            output_dir
            / "backend"
            / "tests"
            / "descriptors"
            / batchnorm_config.test_descriptor_filename
        )
        content = test_path.read_text()

        assert "_peerStatsDesc0" in content
        assert "_peer_statsDesc0" not in content

    def test_test_from_node_local_var_uses_camel_case(
        self, batchnorm_config, generator, tmp_path
    ):
        """fromNode test local TensorAttributesT vars use camelCase."""
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        generator.render(batchnorm_config, output_dir, "backend")

        test_path = (
            output_dir
            / "backend"
            / "tests"
            / "descriptors"
            / batchnorm_config.test_from_node_filename
        )
        content = test_path.read_text()

        assert "peerStatsAttrs0" in content
        assert "peer_statsAttrs0" not in content


# ---------------------------------------------------------------------------
# extra_data_type_fields template branches
# ---------------------------------------------------------------------------


class TestExtraDataTypeFieldsUnpacker:
    """Verify the ``extra_data_type_fields`` block in ``unpacker.hpp.j2`` emits
    correctly-propagating error handling for both the sentinel-gated and
    non-sentinel branches. Both branches now share the same code path: they
    call ``unpackGraphDataType`` and assign its result directly. The backend
    reports count=0 when storage is at the sentinel, in which case
    ``unpackGraphDataType`` returns ``DataType::NOT_SET`` — which is the
    frontend default the assignment preserves.

    Regression guard for a reviewer-flagged bug: an earlier sentinel branch
    swallowed *all* errors from the helper (not just "attribute absent"),
    masking real backend failures.
    """

    def test_sentinel_branch_uses_unpack_graph_data_type(self, sdpa_config, generator):
        """SDPA's ``mma_core_mode`` (sentinel: ``DataType::NOT_SET``) must
        render ``unpackGraphDataType``, propagate real errors, and assign the
        result directly to the attributes member.
        """
        rendered = generator._render_template("unpacker.hpp.j2", sdpa_config)

        # The unified helper is invoked for the sentinel branch
        assert (
            "unpackGraphDataType(" in rendered
        ), "sentinel branch must use unpackGraphDataType"
        # The removed optional helper must not be referenced
        assert (
            "unpackOptionalGraphDataType(" not in rendered
        ), "unpackOptionalGraphDataType has been removed and must not appear"
        # Targets the correct attribute
        assert "HIPDNN_ATTR_SDPA_FWD_MMA_CORE_MODE_EXT" in rendered

        # Real backend errors propagate (not swallowed)
        assert (
            "if(mmaCoreModeErr.is_bad())" in rendered
        ), "sentinel branch must check the error and return on real failure"
        assert (
            "return mmaCoreModeErr;" in rendered
        ), "sentinel branch must return the propagated error"

        # Member-access assignment (sdpa.yaml uses bare member name)
        assert (
            "attributes.mma_core_mode = mmaCoreMode" in rendered
        ), "sentinel branch must assign the unpacked value directly"

        # Regression guard: the old swallowing pattern is gone
        assert "if(!mmaCoreModeErr.is_bad())" not in rendered, (
            "sentinel branch must not silently swallow errors via the "
            "old `if(!Err.is_bad())` pattern"
        )

    def test_non_sentinel_branch_propagates_errors(self, generator):
        """A synthetic config with an ``ExtraDataTypeField`` whose ``sentinel``
        is empty must render ``unpackGraphDataType`` and the standard
        ``if(...Err.is_bad()) { return ...Err; }`` chain.

        No in-tree config exercises this branch, so the test builds the
        OperationConfig directly.
        """
        from tests.helpers import make_minimal_config
        from codegen.models import ExtraDataTypeField

        config = make_minimal_config(
            extra_data_type_fields=[
                ExtraDataTypeField(
                    name="extra_dt",
                    attr_name="HIPDNN_ATTR_OP_EXTRA_DT",
                    frontend_getter="get_extra_dt()",
                    sentinel="",
                    error_label="extra_dt",
                ),
            ],
        )
        rendered = generator._render_template("unpacker.hpp.j2", config)

        # The unified helper is used for the non-sentinel branch
        assert (
            "unpackGraphDataType(" in rendered
        ), "non-sentinel branch must use unpackGraphDataType"
        # The removed optional helper must not be referenced
        assert (
            "unpackOptionalGraphDataType(" not in rendered
        ), "unpackOptionalGraphDataType has been removed and must not appear"
        # Targets the correct attribute
        assert "HIPDNN_ATTR_OP_EXTRA_DT" in rendered

        # Standard error propagation
        assert (
            "if(extraDtErr.is_bad())" in rendered
        ), "non-sentinel branch must check the error explicitly"
        assert (
            "return extraDtErr;" in rendered
        ), "non-sentinel branch must propagate the error"

        # Setter-based assignment (synthetic field uses get_extra_dt() →
        # set_extra_dt)
        assert "attributes.set_extra_dt(extraDt)" in rendered, (
            "non-sentinel branch must call the derived setter with the "
            "unwrapped value"
        )


class TestExtraDataTypeFieldsPacker:
    """Packer-side coverage for sentinel-gated DataType fields.

    The reviewer asked for symmetric verification: the packer must omit
    the attribute when the field equals the sentinel, and emit the
    standard ``setDescriptorAttrDataType`` call inside the gate.
    """

    def test_sdpa_packer_emits_sentinel_gated_mma_core_mode(
        self, sdpa_config, generator
    ):
        """SDPA's ``mma_core_mode`` packer block must guard
        ``setDescriptorAttrDataType`` behind a ``DataType::NOT_SET`` check.
        """
        rendered = generator._render_template("packer.hpp.j2", sdpa_config)

        assert "if(attributes.mma_core_mode != DataType::NOT_SET)" in rendered, (
            "packer must gate mma_core_mode on the sentinel before calling "
            "setDescriptorAttrDataType"
        )
        assert (
            "setDescriptorAttrDataType(" in rendered
        ), "packer must invoke setDescriptorAttrDataType for the gated value"
        assert (
            "HIPDNN_ATTR_SDPA_FWD_MMA_CORE_MODE_EXT" in rendered
        ), "packer must reference the correct attribute identifier"
