#!/usr/bin/env python3
"""
Test list parsing and filtering
"""

import re
from pathlib import Path
from typing import List, Tuple
from .utils import log_error, get_yaml_variant_name


class TestListParser:
    """Parses test list files and applies patterns/filters"""

    def __init__(self, test_dir: str):
        self.test_dir = Path(test_dir)
        self.yaml_dir = self.test_dir / "yaml"
        self.test_lists_dir = self.test_dir / "test_lists"

    def get_test_list(self, test_type: str, gpu_arch: str, pattern: str = ".*") -> List[Tuple[str, str, str]]:
        """
        Get test list for a GPU architecture

        Args:
            test_type: 'exe-time', 'codegen-time', or 'dbg-verify'
            gpu_arch: GPU architecture (e.g., 'gfx950')
            pattern: Regex pattern to filter test cases

        Returns:
            List of tuples: (yaml_path, active_modifier, all_modifiers)
        """
        test_list_file = self.test_lists_dir / f"{test_type}.txt"

        if not test_list_file.exists():
            log_error(f"Test list not found: {test_list_file}")
            return []

        # Compile pattern
        try:
            pattern_re = re.compile(pattern)
        except re.error as e:
            log_error(f"Invalid regex pattern '{pattern}': {e}")
            return []

        # Current section settings
        section_arches = []
        section_modifiers = ""

        # Track seen combinations to avoid duplicates
        seen_combinations = set()
        results = []

        with open(test_list_file, 'r') as f:
            for line in f:
                line = line.strip()

                # Skip comments and empty lines
                if not line or line.startswith('#'):
                    continue

                # Check if this is a section header: [gfx942][gfx950][+sia3]
                if line.startswith('[') and line.endswith(']'):
                    # Parse section header
                    section_arches = []
                    section_modifiers_list = []

                    # Extract all bracketed items
                    bracket_items = re.findall(r'\[([^\]]+)\]', line)
                    for item in bracket_items:
                        if item.startswith('+'):
                            # This is a modifier
                            section_modifiers_list.append(item)
                        else:
                            # This is an architecture
                            section_arches.append(item)

                    section_modifiers = ' '.join(section_modifiers_list)
                    continue

                # This is a test case line
                yaml_file = line.strip()

                # Check if current GPU is in section architectures
                if gpu_arch not in section_arches:
                    continue

                # Check if yaml_file contains glob patterns (* or ?)
                yaml_files_to_process = []
                if '*' in yaml_file or '?' in yaml_file:
                    # It's a glob pattern - expand it
                    pattern_path = self.yaml_dir / yaml_file
                    matched_files = list(self.yaml_dir.glob(yaml_file))

                    if not matched_files:
                        log_error(f"No YAML files found matching pattern: {yaml_file}")
                        continue

                    yaml_files_to_process.extend(matched_files)
                else:
                    # It's a literal filename
                    yaml_path = self.yaml_dir / yaml_file
                    if not yaml_path.exists():
                        log_error(f"YAML file not found: {yaml_path}")
                        continue
                    yaml_files_to_process.append(yaml_path)

                # Process each YAML file (either single file or multiple from glob)
                for yaml_path in yaml_files_to_process:
                    # Generate base variant (no modifier applied)
                    combo_key = (str(yaml_path), '', '')
                    if combo_key not in seen_combinations:
                        # Check if base variant name matches pattern
                        base_yaml_name = f"{yaml_path.stem}.yaml"
                        if pattern_re.search(base_yaml_name):
                            results.append((str(yaml_path), '', section_modifiers))
                            seen_combinations.add(combo_key)

                    # Generate additional variants for each modifier
                    if section_modifiers:
                        for modifier in section_modifiers.split():
                            combo_key = (str(yaml_path), modifier, '')
                            if combo_key not in seen_combinations:
                                # Generate variant name with modifier suffix
                                variant_name = get_yaml_variant_name(yaml_path.stem, modifier)
                                variant_name_with_ext = f"{variant_name}.yaml"

                                # Check if variant name matches pattern
                                if pattern_re.search(variant_name_with_ext):
                                    results.append((str(yaml_path), modifier, section_modifiers))
                                    seen_combinations.add(combo_key)

        return results

