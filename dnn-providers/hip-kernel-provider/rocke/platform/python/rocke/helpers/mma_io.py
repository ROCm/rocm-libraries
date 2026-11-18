# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Compatibility exports for the former matrix-I/O helper module.

New tensor transport uses ``rocke.helpers.tiling.load_fragment`` and
``store_fragment``. Storage typing belongs to core, encoded-pattern packing
has its own helper, and the raw-layout adapter is retained for existing callers
without owning a separate load implementation.
"""

from ..core.ir_storage import storage_element_type, storage_ir_type, storage_vector_type
from ..core.mma_operand import mma_operand_ir_type
from .packing import pack_fragment_bits
from .tiling.compat import load_matrix_fragment

__all__ = [
    "storage_ir_type",
    "storage_element_type",
    "storage_vector_type",
    "mma_operand_ir_type",
    "load_matrix_fragment",
    "pack_fragment_bits",
]
