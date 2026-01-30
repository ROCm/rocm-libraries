#!/usr/bin/env python3

# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic, Union
import re
import warnings

"""
A configuration parser for rocPRIM config header files that extracts and manages algorithm-specific 
configurations across different architectures and data types. This module provides functionality 
to parse C++ template specializations from header files containing default configurations for 
various AMD GPU architectures. It supports both simple parameter extraction using predefined 
patterns and custom extraction logic through user-provided functions.
"""

T = TypeVar("T")  # Type variable for config parameters


@dataclass
class ConfigInfo(Generic[T]):
    """Generic configuration info class."""

    arch: str
    key_type: str
    value_type: str
    params: T


ParamType = Union[Dict[str, List[int]], Any]
ExtractorType = Callable[[str, str, str, str], Optional[ParamType]]


class ConfigParser(Generic[T]):
    """Parser for device configuration header files."""

    def __init__(
        self,
        algo_name: str,
        param_names: Optional[List[str]] = None,
        extract_params: Optional[ExtractorType] = None,
    ):
        self.algo_name = algo_name
        self.param_names = param_names
        self.custom_extractor = extract_params
        self.configs: List[ConfigInfo[T]] = []

        # Initialize regex patterns
        self.arch_pattern = r"target_arch::(\w+)"
        self.key_type_pattern = r"key_type\s*=\s*([^\s,]+)"
        self.value_type_pattern = r"value_type\s*=\s*([^\s,]+)"

        # Build default config pattern if using simple parameter extraction
        if param_names and not extract_params:
            config_params_pattern = r"\s*,\s*".join([r"(\d+)"] * len(param_names))
            self.config_pattern = rf"{algo_name}_config\s*<{config_params_pattern}>"

        # Base pattern for matching configurations
        self.base_pattern = (
            r"//\s*Based on key_type\s*=\s*[^\s,]+,\s*value_type\s*=\s*[^\n]+\n"
            r"\s*template\s*<[^>]+>\s*"
            r"struct\s+default_{algo_name}_config<[^\{{]+>>[^\{{]*"
            r":[^\{{]+\n*\{{[^\}}]*\}};"
        )

        self.specialization_pattern = self.base_pattern.format(algo_name=algo_name)

    def _extract_simple_params(
        self, config_text: str, arch: str, key_type: str, value_type: str
    ) -> Optional[Dict[str, List[int]]]:
        """Default parameter extraction for simple configurations."""
        if not self.param_names:
            return None

        config_match = re.search(self.config_pattern, config_text)
        if not config_match:
            return None

        params = {}
        try:
            for i, param_name in enumerate(self.param_names, 1):
                params[param_name] = [int(config_match.group(i))]
            return params
        except (IndexError, ValueError) as e:
            warnings.warn(
                f"Failed to parse parameters for {arch}/{key_type}/{value_type}: {e}"
            )
            return None

    def parse_header(self, header_content: str) -> List[ConfigInfo[T]]:
        """
        Parse the header content and extract configurations.

        Args:
            header_content: Content of the header file.

        Returns:
            List of ConfigInfo objects containing the parsed configurations.
        """
        self.configs = []
        specializations = re.finditer(
            self.specialization_pattern, header_content, re.DOTALL
        )

        for spec in specializations:
            self._process_specialization(spec.group(0))

        if not self.configs:
            warnings.warn(
                f"No configuration patterns found for algorithm '{self.algo_name}'"
            )

        return self.configs

    def _process_specialization(self, spec_text: str) -> None:
        """
        Process a single specialization and extract its configuration.

        Args:
            spec_text: Text of the specialization to process.
        """
        arch_match = re.search(self.arch_pattern, spec_text)
        key_match = re.search(self.key_type_pattern, spec_text)
        value_match = re.search(self.value_type_pattern, spec_text)

        if not all((arch_match, key_match, value_match)):
            warnings.warn("Failed to extract basic fields from specialization.")
            return

        arch = arch_match.group(1)
        key_type = key_match.group(1).strip()
        value_type = value_match.group(1).strip()

        # Use custom extractor if provided, otherwise use simple extraction
        if self.custom_extractor:
            params = self.custom_extractor(spec_text, arch, key_type, value_type)
        else:
            params = self._extract_simple_params(spec_text, arch, key_type, value_type)

        if params is not None:
            self.configs.append(
                ConfigInfo(
                    arch=arch, key_type=key_type, value_type=value_type, params=params
                )
            )

    def get_default_config(
        self, arch: str, key_type: str, value_type: str = ""
    ) -> Optional[T]:
        """
        Get the default configuration for a specific architecture and types.

        Args:
            arch: Target architecture (e.g., 'gfx1030').
            key_type: Type of keys (e.g., 'int', 'float').
            value_type: Type of values (e.g., 'int', 'empty_type').

        Returns:
            Configuration parameters, or None if not found.
        """
        value_type = "empty_type" if not value_type else value_type

        # It could very much be that the header parsing failed and
        # self.configs is empty. In that case we cannot return
        # a default config.
        if self.configs is None:
            return None

        for config in self.configs:
            if (
                config.arch == arch
                and config.key_type == key_type
                and config.value_type == value_type
            ):
                return config.params

        for config in self.configs:
            if (
                config.arch == "unknown"
                and config.key_type == key_type
                and config.value_type == value_type
            ):
                return config.params

        warnings.warn(
            f"No configuration found for:\n"
            f"  Architecture: {arch}\n"
            f"  Key type: {key_type}\n"
            f"  Value type: {value_type}\n"
            f"Available architectures: {sorted(set(c.arch for c in self.configs))}"
        )
        return None
