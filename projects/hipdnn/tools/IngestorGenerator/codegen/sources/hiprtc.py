# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Scans ``.cpp``/``.hip`` files for ``__global__`` entry points and candidate
KMD field names.

Text-based extraction, not a preprocessor or parser: an unrecognized source
shape yields no candidates rather than a wrong guess.
"""

import re
from pathlib import Path

from .base import CandidateKernel, SourceAdapterResult

#: ``extern "C" __global__ void Name(...)``, the shape used by the HIP kernel
#: fixtures under ``kernel_ingestor_engine/test_descriptors/*/*/kernels/``.
_ENTRY_POINT_PATTERN = re.compile(
    r'extern\s+"C"\s+__global__\s+void\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\('
)

#: A HIP_PLUGIN_* define; one this file never sets is supplied by the compile
#: command.
_DEFINE_USE_PATTERN = re.compile(r"\b(HIP_PLUGIN_[A-Z0-9_]+)\b")
_DEFINE_SET_PATTERN = re.compile(r"#\s*define\s+(HIP_PLUGIN_[A-Z0-9_]+)")

#: Entry point template parameter list, e.g. ``template <int BlockSize>``.
_TEMPLATE_PATTERN = re.compile(r"template\s*<([^>]*)>")
_TEMPLATE_PARAM_NAME_PATTERN = re.compile(
    r"(?:int|typename|class)\s+([A-Za-z_][A-Za-z0-9_]*)"
)


def _candidate_fields(text: str) -> list[str]:
    """HIP_PLUGIN_* defines referenced but not set here, plus template
    parameter names from any preceding ``template<...>`` line."""
    used = set(_DEFINE_USE_PATTERN.findall(text))
    set_locally = set(_DEFINE_SET_PATTERN.findall(text))
    externally_supplied = sorted(used - set_locally)

    template_params: list[str] = []
    for match in _TEMPLATE_PATTERN.finditer(text):
        template_params.extend(_TEMPLATE_PARAM_NAME_PATTERN.findall(match.group(1)))

    return externally_supplied + template_params


class HiprtcAdapter:
    """Scans ``.cpp``/``.hip`` sources for ``extern "C" __global__`` entry
    points."""

    def infer(self, *sources: Path) -> SourceAdapterResult:
        kernels: list[CandidateKernel] = []
        for source in sources:
            text = source.read_text()
            fields = _candidate_fields(text)
            for match in _ENTRY_POINT_PATTERN.finditer(text):
                kernels.append(
                    CandidateKernel(
                        entry_point=match.group("name"),
                        source_file=source.name,
                        template_params=fields,
                    )
                )

        # One pack per source file: multiple entry points in one file are
        # instantiations of the same operation.
        distinct_files = {k.source_file for k in kernels}
        return SourceAdapterResult(
            kernels=kernels,
            suggested_pack_count=max(len(distinct_files), 1),
        )
