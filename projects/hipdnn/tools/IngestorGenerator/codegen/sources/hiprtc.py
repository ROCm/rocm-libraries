# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Scans one or more ``.cpp``/``.hip`` files for ``__global__`` entry points
and candidate KMD field names.

This is text-based extraction, not a real preprocessor or parser, and is
deliberately conservative: an unrecognized source shape yields no candidates for
that function rather than a wrong guess.
"""

import re
from pathlib import Path

from .base import CandidateKernel, SourceAdapterResult

#: ``extern "C" __global__ void Name(...)`` -- the shape the HIP kernel
#: fixtures under ``kernel_ingestor_engine/test_descriptors/*/*/kernels/``
#: use (ConvFwd.cpp, PointwiseAdd.cpp).
_ENTRY_POINT_PATTERN = re.compile(
    r'extern\s+"C"\s+__global__\s+void\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*\('
)

#: A compile-time #define this file itself does not set -- a strong signal it is
#: expected to arrive from the compile command.
_DEFINE_USE_PATTERN = re.compile(r"\b(HIP_PLUGIN_[A-Z0-9_]+)\b")
_DEFINE_SET_PATTERN = re.compile(r"#\s*define\s+(HIP_PLUGIN_[A-Z0-9_]+)")

#: A template parameter list on the entry point itself, e.g.
#: ``template <int BlockSize, typename T>``.
_TEMPLATE_PATTERN = re.compile(r"template\s*<([^>]*)>")
_TEMPLATE_PARAM_NAME_PATTERN = re.compile(
    r"(?:int|typename|class)\s+([A-Za-z_][A-Za-z0-9_]*)"
)


def _candidate_fields(text: str) -> list[str]:
    """Best-effort field-name candidates: externally-supplied HIP_PLUGIN_*
    defines this file references but never sets itself, plus any template
    parameter names on a preceding ``template<...>`` line."""
    used = set(_DEFINE_USE_PATTERN.findall(text))
    set_locally = set(_DEFINE_SET_PATTERN.findall(text))
    externally_supplied = sorted(used - set_locally)

    template_params: list[str] = []
    for match in _TEMPLATE_PATTERN.finditer(text):
        template_params.extend(_TEMPLATE_PARAM_NAME_PATTERN.findall(match.group(1)))

    return externally_supplied + template_params


class HiprtcAdapter:
    """Scans HIP/HIPRTC-style ``.cpp``/``.hip`` sources for
    ``extern "C" __global__`` entry points."""

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

        # One pack per distinct source file: several entry points in one file are
        # the same operation's instantiations.
        distinct_files = {k.source_file for k in kernels}
        return SourceAdapterResult(
            kernels=kernels,
            suggested_pack_count=max(len(distinct_files), 1),
        )
