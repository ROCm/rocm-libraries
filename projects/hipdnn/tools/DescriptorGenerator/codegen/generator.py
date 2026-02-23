# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Template rendering orchestrator."""

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .models import OperationConfig


class DescriptorGenerator:
    """Renders all templates for a given OperationConfig."""

    def __init__(self, template_dir: Path):
        self.env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            keep_trailing_newline=True,
            trim_blocks=True,
            lstrip_blocks=True,
        )

    def render(self, config: OperationConfig, output_dir: Path) -> list[str]:
        """Render all templates and write to output_dir. Returns list of written files."""
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
            "test_integration.cpp.j2": Path("tests/frontend")
            / config.test_integration_filename,
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
        }

        fragments_dir = output_dir / "fragments"
        fragments_dir.mkdir(parents=True, exist_ok=True)

        for template_name, filename in fragment_templates.items():
            out_path = fragments_dir / filename
            content = self._render_template(template_name, config)
            out_path.write_text(content)
            written.append(f"fragments/{filename}")

        return written

    def _render_template(self, template_name: str, config: OperationConfig) -> str:
        try:
            template = self.env.get_template(template_name)
            return template.render(op=config)
        except Exception as e:
            raise RuntimeError(
                f"Failed to render template '{template_name}' for "
                f"operation '{config.name}': {e}"
            ) from e
