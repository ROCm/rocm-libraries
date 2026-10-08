# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Generate deterministic GEMM candidates for a packaged solution index.

The functions in this package read metadata and construct client inputs. They
do not execute kernels or claim that native support checks have passed.
"""

from .candidates import GemmCase, GemmProfile, GemmShape, candidate_gemms
from .clients import bench_yaml, tensile_ini
from .metadata import LibraryMetadata, load_library_metadata

__all__ = [
    "GemmCase",
    "GemmProfile",
    "GemmShape",
    "LibraryMetadata",
    "bench_yaml",
    "candidate_gemms",
    "load_library_metadata",
    "tensile_ini",
]
