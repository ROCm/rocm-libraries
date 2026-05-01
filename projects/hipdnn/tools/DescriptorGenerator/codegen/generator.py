# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Template rendering orchestrator."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .models import DataField, OperationConfig


class DescriptorGenerator:
    """Renders all templates for a given OperationConfig."""

    def __init__(self, template_dir: Path):
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render_lift_only(self, config: OperationConfig, output_dir: Path) -> list[str]:
        """Render only lifting-related templates. Returns list of written files."""
        written = []

        # Lifting file templates
        lift_templates = {
            "unpacker.hpp.j2": Path("frontend/include/hipdnn_frontend/detail")
            / config.unpacker_filename,
            "test_from_node.cpp.j2": Path("backend/tests/descriptors")
            / config.test_from_node_filename,
            "test_integration_lifting.cpp.j2": Path("tests/frontend")
            / config.test_integration_lifting_filename,
        }

        for template_name, rel_path in lift_templates.items():
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            content = self._render_template(template_name, config)
            out_path.write_text(content)
            written.append(str(rel_path))

        # Lifting fragment templates
        lift_fragments = {
            "fragments/node_factory_case.j2": "node_factory_case.txt",
            "fragments/operation_unpacker_case.j2": "operation_unpacker_case.txt",
            "fragments/operation_unpacker_test.j2": "operation_unpacker_test.txt",
            "fragments/operation_type_enum.j2": "operation_type_enum.txt",
            "fragments/node_unpack_override.j2": "node_unpack_override.txt",
            "fragments/packer_name_addition.j2": "packer_name_addition.txt",
        }

        fragments_dir = output_dir / "fragments"
        fragments_dir.mkdir(parents=True, exist_ok=True)

        for template_name, filename in lift_fragments.items():
            out_path = fragments_dir / filename
            content = self._render_template(template_name, config)
            out_path.write_text(content)
            written.append(f"fragments/{filename}")

        # Generate descriptor lifting additions (manual insertion guide)
        additions = self._render_descriptor_lifting_additions(config)
        additions_path = fragments_dir / "descriptor_lifting_additions.txt"
        additions_path.write_text(additions)
        written.append("fragments/descriptor_lifting_additions.txt")

        # Generate constants file (when constants_include is not set — no pre-existing header)
        if not config.test_data.constants_include:
            written += self._render_constants(config, output_dir)

        return written

    def _render_descriptor_lifting_additions(self, config: OperationConfig) -> str:
        """Generate a text file showing what to add to existing descriptor files."""
        lines = []
        cn = config.class_name

        lines.append(f"# Descriptor Lifting Additions for {cn}")
        lines.append(f"# Add these changes to the existing {cn}.hpp/.cpp files.")
        lines.append("")

        # --- HPP additions ---
        lines.append("=" * 72)
        lines.append(f"# {cn}.hpp — Add these to the class declaration")
        lines.append("=" * 72)
        lines.append("")
        lines.append("# 1. Add include (at top of file):")
        lines.append("#include <unordered_map>")
        lines.append("")
        lines.append("# 2. Add public static method (after buildNode declaration):")
        lines.append(f"    static std::shared_ptr<{cn}>")
        lines.append(f"        fromNode(const {config.fbs_namespace}::NodeT& nodeT,")
        lines.append(
            "                 const std::unordered_map<int64_t, "
            "std::shared_ptr<TensorDescriptor>>& tensorMap);"
        )
        lines.append("")
        lines.append("# 3. Add private member (after _data):")
        lines.append("    std::string _name;")
        lines.append("")

        # --- CPP additions ---
        lines.append("=" * 72)
        lines.append(f"# {cn}.cpp — Add these changes")
        lines.append("=" * 72)
        lines.append("")
        lines.append("# 1. Add include:")
        lines.append('#include "HipdnnOperationType.h"')
        lines.append("")

        lines.append("# 2. In setAttribute switch, add before default:")
        lines.append("    case HIPDNN_ATTR_OPERATION_NAME_EXT:")
        lines.append("        setString(_name,")
        lines.append("                  attributeType,")
        lines.append("                  elementCount,")
        lines.append("                  arrayOfElements,")
        lines.append(f'                  "{cn}::setAttribute()");')
        lines.append("        break;")
        lines.append("")

        lines.append("# 3. In getAttribute switch, add before default:")
        lines.append("    case HIPDNN_ATTR_OPERATION_NAME_EXT:")
        lines.append("        getString(_name,")
        lines.append("                  attributeType,")
        lines.append("                  requestedElementCount,")
        lines.append("                  elementCount,")
        lines.append("                  arrayOfElements,")
        lines.append(f'                  "{cn}::getAttribute()");')
        lines.append("        break;")

        if config.operation_type_enum:
            lines.append("    case HIPDNN_ATTR_OPERATION_TYPE_EXT:")
            lines.append(f"        getOperationType({config.operation_type_enum},")
            lines.append("                         attributeType,")
            lines.append("                         requestedElementCount,")
            lines.append("                         elementCount,")
            lines.append("                         arrayOfElements,")
            lines.append(f'                         "{cn}::getAttribute()");')
            lines.append("        break;")
        lines.append("")

        lines.append("# 4. In buildNode(), add before compute_data_type:")
        lines.append("    node->name = _name;")
        lines.append("")

        lines.append("# 5. In toString(), prepend name to output:")
        lines.append('    str += "name=" + _name;')
        lines.append("")

        lines.append("# 6. Add fromNode() implementation (at end of file):")
        content = self._render_template("descriptor.cpp.j2", config)
        # Extract fromNode from the rendered output
        from_node_start = content.find(f"std::shared_ptr<{cn}> {cn}::fromNode(")
        if from_node_start >= 0:
            # Find the closing brace
            brace_depth = 0
            i = content.index("{", from_node_start)
            for j in range(i, len(content)):
                if content[j] == "{":
                    brace_depth += 1
                elif content[j] == "}":
                    brace_depth -= 1
                    if brace_depth == 0:
                        from_node_end = j + 1
                        break
            lines.append(content[from_node_start:from_node_end])
        lines.append("")

        lines.append("=" * 72)
        lines.append("# IMPORTANT: Packer Update Required")
        lines.append("=" * 72)
        lines.append("")
        lines.append("# The existing packer must also pack the operation name.")
        lines.append(f"# See fragments/packer_name_addition.txt for the code to add")
        lines.append(
            f"# to {config.packer_filename} before the finalizeDescriptor() call."
        )
        lines.append("")
        lines.append("=" * 72)
        lines.append("# NOTE: Graph Descriptor Name Tests (Auto-Generated)")
        lines.append("=" * 72)
        lines.append("")
        lines.append(
            "# The graph descriptor test template now auto-generates name tests:"
        )
        lines.append("#   - OperationNamePreservedInSerialization")
        lines.append("#   - OperationNameRoundTripThroughLifting")
        lines.append("# These are generated unconditionally — no manual work needed.")
        lines.append(
            "# If the graph test file was generated fresh (backend or full mode),"
        )
        lines.append("# the name tests are already included.")

        return "\n".join(lines) + "\n"

    def render(
        self, config: OperationConfig, output_dir: Path, mode: str = "backend"
    ) -> list[str]:
        """Render templates based on the specified mode.

        Args:
            config: The operation configuration.
            output_dir: Root directory for generated output.
            mode: One of 'backend', 'frontend', 'full', or 'lift-only'.

        Returns:
            List of relative paths for all written files.
        """
        dispatch = {
            "backend": self.render_backend,
            "frontend": self.render_frontend,
            "full": self.render_full,
            "lift-only": self.render_lift_only,
        }
        renderer = dispatch.get(mode)
        if renderer is None:
            raise ValueError(
                f"Unknown render mode '{mode}'. "
                f"Valid modes: {', '.join(dispatch.keys())}"
            )
        return renderer(config, output_dir)

    def render_backend(self, config: OperationConfig, output_dir: Path) -> list[str]:
        """Render backend templates and write to output_dir. Returns list of written files."""
        written = []

        # Template -> output path mapping
        file_templates = {
            "descriptor.hpp.j2": Path("backend/src/descriptors")
            / config.header_filename,
            "descriptor.cpp.j2": Path("backend/src/descriptors")
            / config.source_filename,
            "packer.hpp.j2": Path("frontend/include/hipdnn_frontend/detail")
            / config.packer_filename,
            "test_descriptor.cpp.j2": Path("backend/tests/descriptors")
            / config.test_descriptor_filename,
            "test_graph_ops.cpp.j2": Path("backend/tests/descriptors")
            / config.test_graph_filename,
            "test_from_node.cpp.j2": Path("backend/tests/descriptors")
            / config.test_from_node_filename,
            "test_integration.cpp.j2": Path("tests/frontend")
            / config.test_integration_filename,
            "test_integration_lifting.cpp.j2": Path("tests/frontend")
            / config.test_integration_lifting_filename,
            "unpacker.hpp.j2": Path("frontend/include/hipdnn_frontend/detail")
            / config.unpacker_filename,
        }

        for template_name, rel_path in file_templates.items():
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            content = self._render_template(template_name, config)
            out_path.write_text(content)
            written.append(str(rel_path))

        # Fragment templates
        fragment_templates = {
            "fragments/attribute_enum_block.j2": "attribute_enum_block.txt",
            "fragments/descriptor_type_enum.j2": "descriptor_type_enum.txt",
            "fragments/string_utils_block.j2": "string_utils_block.txt",
            "fragments/factory_case.j2": "factory_case.txt",
            "fragments/cmake_entries.j2": "cmake_entries.txt",
            "fragments/node_factory_case.j2": "node_factory_case.txt",
            "fragments/operation_unpacker_case.j2": "operation_unpacker_case.txt",
            "fragments/operation_unpacker_test.j2": "operation_unpacker_test.txt",
            "fragments/operation_type_enum.j2": "operation_type_enum.txt",
            "fragments/node_unpack_override.j2": "node_unpack_override.txt",
        }

        fragments_dir = output_dir / "fragments"
        fragments_dir.mkdir(parents=True, exist_ok=True)

        for template_name, filename in fragment_templates.items():
            out_path = fragments_dir / filename
            content = self._render_template(template_name, config)
            out_path.write_text(content)
            written.append(f"fragments/{filename}")

        # Mode enum plumbing (only for fields with enum_def)
        if config.generatable_mode_fields:
            written += self.render_mode_enums(config, output_dir)

        # Generate constants file (when constants_include is not set — no pre-existing header)
        if not config.test_data.constants_include:
            written += self._render_constants(config, output_dir)

        return written

    def render_frontend(self, config: OperationConfig, output_dir: Path) -> list[str]:
        """Render frontend templates and write to output_dir. Returns list of written files."""
        written = []

        # Frontend file templates
        file_templates = {
            "attributes.hpp.j2": Path("frontend/include/hipdnn_frontend/attributes")
            / config.attributes_header_filename,
            "node.hpp.j2": Path("frontend/include/hipdnn_frontend/node")
            / config.node_header_filename,
        }

        for template_name, rel_path in file_templates.items():
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            content = self._render_template(template_name, config)
            out_path.write_text(content)
            written.append(str(rel_path))

        # Frontend test templates
        test_templates = {
            "test_attributes.cpp.j2": Path("frontend/tests")
            / config.test_attributes_filename,
            "test_node.cpp.j2": Path("frontend/tests") / config.test_node_filename,
            "test_frontend_graph.cpp.j2": Path("frontend/tests")
            / config.test_frontend_graph_filename,
        }

        for template_name, rel_path in test_templates.items():
            out_path = output_dir / rel_path
            out_path.parent.mkdir(parents=True, exist_ok=True)
            content = self._render_template(template_name, config)
            out_path.write_text(content)
            written.append(str(rel_path))

        # Frontend fragment templates
        fragment_templates = {
            "fragments/graph_method.j2": "graph_method.txt",
            "fragments/graph_includes.j2": "graph_includes.txt",
            "fragments/frontend_cmake_entries.j2": "frontend_cmake_entries.txt",
            "fragments/node_type_enum.j2": "node_type_enum.txt",
        }

        fragments_dir = output_dir / "fragments"
        fragments_dir.mkdir(parents=True, exist_ok=True)

        for template_name, filename in fragment_templates.items():
            out_path = fragments_dir / filename
            content = self._render_template(template_name, config)
            out_path.write_text(content)
            written.append(f"fragments/{filename}")

        return written

    def render_full(self, config: OperationConfig, output_dir: Path) -> list[str]:
        """Render all backend and frontend templates. Returns list of written files."""
        written = self.render_backend(config, output_dir)
        written += self.render_frontend(config, output_dir)
        return written

    def render_mode_enums(self, config: OperationConfig, output_dir: Path) -> list[str]:
        """Render mode enum templates for fields with enum_def. Returns list of written files."""
        written = []

        for df in config.generatable_mode_fields:
            # Backend header (new file)
            header_path = (
                output_dir / "backend" / "include" / df.enum_def.backend_header
            )
            header_path.parent.mkdir(parents=True, exist_ok=True)
            content = self._render_mode_template("mode_backend_header.j2", config, df)
            header_path.write_text(content)
            written.append(f"backend/include/{df.enum_def.backend_header}")

            # Backend plumbing fragment
            fragments_dir = output_dir / "fragments"
            fragments_dir.mkdir(parents=True, exist_ok=True)

            backend_frag = fragments_dir / f"mode_backend_plumbing_{df.name}.txt"
            content = self._render_mode_template(
                "fragments/mode_backend_plumbing.j2", config, df
            )
            backend_frag.write_text(content)
            written.append(f"fragments/mode_backend_plumbing_{df.name}.txt")

            # Frontend plumbing fragment
            frontend_frag = fragments_dir / f"mode_frontend_plumbing_{df.name}.txt"
            content = self._render_mode_template(
                "fragments/mode_frontend_plumbing.j2", config, df
            )
            frontend_frag.write_text(content)
            written.append(f"fragments/mode_frontend_plumbing_{df.name}.txt")

            # Frontend converter test fragment
            frontend_test_frag = fragments_dir / f"mode_frontend_tests_{df.name}.txt"
            content = self._render_mode_template(
                "fragments/mode_frontend_tests.j2", config, df
            )
            frontend_test_frag.write_text(content)
            written.append(f"fragments/mode_frontend_tests_{df.name}.txt")

        return written

    def _render_constants(self, config: OperationConfig, output_dir: Path) -> list[str]:
        """Render the shared constants header. Returns list of written files."""
        rel_path = (
            Path("test_sdk/include/hipdnn_test_sdk/constants")
            / f"{config.effective_constants_include}.hpp"
        )
        out_path = output_dir / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        content = self._render_template("constants.hpp.j2", config)
        out_path.write_text(content)
        return [str(rel_path)]

    def _render_template(self, template_name: str, config: OperationConfig) -> str:
        try:
            template = self.env.get_template(template_name)
            return template.render(op=config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to render template '{template_name}' for "
                f"operation '{config.name}': {e}"
            ) from e

    def _render_mode_template(
        self, template_name: str, config: OperationConfig, df: DataField
    ) -> str:
        try:
            template = self.env.get_template(template_name)
            return template.render(op=config, df=df)
        except Exception as e:
            raise RuntimeError(
                f"Failed to render template '{template_name}' for "
                f"mode field '{df.name}' in operation '{config.name}': {e}"
            ) from e
