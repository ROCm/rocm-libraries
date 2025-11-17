#!/usr/bin/env python3
"""
YAML patching and modification logic
"""

import subprocess
import shutil
from pathlib import Path
from typing import Optional


class YAMLPatcher:
    """Handles patching YAML files for different test types and modifiers"""

    def __init__(self, script_dir: str):
        self.script_dir = Path(script_dir)
        self.patch_script = self.script_dir / "patch_yaml.py"

    def patch_yaml(self, source_yaml: str, output_dir: str, test_type: str,
                  active_modifier: str, all_modifiers: str) -> str:
        """
        Patch and copy YAML file based on test type and modifiers

        Args:
            source_yaml: Source YAML file path
            output_dir: Output directory for patched YAML
            test_type: 'exe_time', 'codegen_time', or 'dbg_verify'
            active_modifier: The modifier being applied (e.g., '+sia3')
            all_modifiers: All modifiers in section (for context)

        Returns:
            Path to patched YAML file
        """
        source_path = Path(source_yaml)
        yaml_basename = source_path.stem

        # Determine output filename based on active modifier
        suffix = ""
        if active_modifier == "+sia3":
            suffix = "-sia3"
        elif active_modifier == "+sparse":
            suffix = "-sparse"
        elif active_modifier == "+sia5":
            suffix = "-sia5"

        output_yaml = Path(output_dir) / f"{yaml_basename}{suffix}.yaml"

        # Copy base YAML
        shutil.copy(source_yaml, output_yaml)

        # Apply test-type-specific patches
        if test_type == "codegen_time":
            subprocess.run(
                ['python3', str(self.patch_script),
                 '--input', str(output_yaml),
                 '--output', str(output_yaml),
                 '--patch-codegen'],
                check=True
            )

        # Apply modifier-specific patches
        if active_modifier == "+sia3":
            subprocess.run(
                ['python3', str(self.patch_script),
                 '--input', str(output_yaml),
                 '--output', str(output_yaml),
                 '--schedule-iter-alg', '3'],
                check=True
            )
        elif active_modifier == "+sparse":
            subprocess.run(
                ['python3', str(self.patch_script),
                 '--input', str(output_yaml),
                 '--output', str(output_yaml),
                 '--enable-sparse'],
                check=True
            )

        return str(output_yaml)

