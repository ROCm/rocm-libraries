# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Rule-based configuration for direct-convolution kernel instances.

Background
----------
On the feature branch the direct-conv kernel instances lived inside the
dispatcher JSON config tree
(``codegen/configs/grouped_conv/<variant>/<subset>/nhwgc_{fp16,bf16}.json``,
tagged ``"kind": "direct_conv"``). The upstream ``develop`` branch has replaced
the whole JSON mechanism with Python *rule sets* under ``codegen/grouped_conv/``
and deletes the JSON tree. To survive that merge the direct-conv instances are
re-expressed here as a rule module that won't collide with develop's files.

Five rule sets are provided, all reachable through :func:`get_configs` and
following develop's naming convention:

``profiler``
    Replays the exact instances that the JSON tree carried, family by family.
    The data tables below (``_FAITHFUL_TABLES``) are the union of the curated
    JSON instance sets. The per-instance ``direction`` field is not stored in
    the tables (it is constant within a row) and is stamped per requested
    variant.

``full``
    Produces the instances programmatically via cartesian-product generators
    with light pruning. It is constructed so that its output is a *superset* of
    the profiler set (the coverage test only requires ``>=``).

``tests``
    A ~20% stratified slice of the ``profiler`` set (subset of ``profiler``).

``full-tests``
    A ~20% stratified slice of the ``full`` set, using develop's selection
    convention (subset of ``full``).

``tiny``
    One instance per channel family, taken from the ``full`` set (subset of
    ``full``).

All rule sets assign deterministic, gap-free ids (the derived subsets renumber
after selection). Original JSON ids are intentionally NOT preserved: the build
registers kernels by filename prefix glob and the regression harness matches
runtime ``direct_tile_conv_*`` names, neither of which depends on the codegen
id.

Entry point
-----------
``get_configs(arch, variants, ndims, datatypes, subset="profiler",
rule_set="profiler") -> List[DirectConvKernelConfig]``
"""

from typing import List

from unified_grouped_conv_codegen import (
    DirectConvKernelConfig,
    GroupedConvVariant,
    direct_conv_supported_on_arch,
)

# ---------------------------------------------------------------------------
# Domain
# ---------------------------------------------------------------------------

# Direct-conv kernels are NHWGC, 2D, fp16/bf16, forward + bwd_data only.
_SUPPORTED_LAYOUT = "nhwgc"
_SUPPORTED_NDIM = 2
_SUPPORTED_DATATYPES = ("fp16", "bf16")
_SUPPORTED_VARIANTS = (
    GroupedConvVariant.FORWARD,
    GroupedConvVariant.BACKWARD_DATA,
)

# variant -> direct_conv Direction enum token stamped into the config payload.
_VARIANT_DIRECTION = {
    GroupedConvVariant.FORWARD: "Fprop",
    GroupedConvVariant.BACKWARD_DATA: "Dgrad",
}

# Config key that the variant-dependent 'direction' field must precede so that
# the emitted C++ designated initializers keep their canonical order.
_DIRECTION_BEFORE_KEY = "swizzle_type"


def _stamp_direction(row, direction):
    """Return an ordered config dict with the variant 'direction' inserted at its
    canonical position (immediately before ``swizzle_type``). The table rows do
    not store 'direction' (it is constant within a row), so we splice it in
    here to preserve designated-initializer field order."""
    out = {}
    for key, value in row.items():
        if key == _DIRECTION_BEFORE_KEY:
            out["direction"] = direction
        out[key] = value
    return out


# ---------------------------------------------------------------------------
# Profiler configs: union of the curated instance sets used in CK Profiler.
#
# Keyed by (channel_family, impl, version). Each row is an ordered dict of the
# family-specific Config fields.
# The order matters: the codegen emits C++ designated initializers, whose order must
# match the Config struct field order. The variant-dependent 'direction' field
# is NOT stored here (it is constant within a row); it is inserted at its
# canonical position -- immediately before 'swizzle_type' -- by _stamp_direction
# when a concrete variant is requested.
# ---------------------------------------------------------------------------

_PROFILER_TABLES = {
    (4, 'tile_grouped', 'v3'): [
        {'waves_c64': 1, 'waves_q4': 1, 'swizzle_type': 'None', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_c64': 1, 'waves_q4': 1, 'swizzle_type': 'None', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_c64': 1, 'waves_q4': 1, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_c64': 1, 'waves_q4': 1, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_c64': 2, 'waves_q4': 1, 'swizzle_type': 'None', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_c64': 2, 'waves_q4': 1, 'swizzle_type': 'None', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_c64': 2, 'waves_q4': 1, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_c64': 2, 'waves_q4': 1, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_c64': 2, 'waves_q4': 2, 'swizzle_type': 'None', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_c64': 2, 'waves_q4': 2, 'swizzle_type': 'None', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_c64': 2, 'waves_q4': 2, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_c64': 2, 'waves_q4': 2, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_c64': 2, 'waves_q4': 4, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 1},
        {'waves_c64': 2, 'waves_q4': 4, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 2},
        {'waves_c64': 2, 'waves_q4': 4, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_c64': 2, 'waves_q4': 4, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_c64': 2, 'waves_q4': 8, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 1},
        {'waves_c64': 2, 'waves_q4': 8, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 2},
        {'waves_c64': 2, 'waves_q4': 8, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_c64': 2, 'waves_q4': 8, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
    ],
    (8, 'tile_grouped', 'v2'): [
        {'waves_per_wg': 1, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 1, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 16, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 16, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 2, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 2, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 3, 'swizzle_type': 'None', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 3, 'swizzle_type': 'None', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 3, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 3, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 4, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 4, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 5, 'swizzle_type': 'None', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 5, 'swizzle_type': 'None', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 5, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 5, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 6, 'swizzle_type': 'None', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 6, 'swizzle_type': 'None', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 6, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 6, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 7, 'swizzle_type': 'None', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 7, 'swizzle_type': 'None', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 7, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 7, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 8, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 1},
        {'waves_per_wg': 8, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 2},
        {'waves_per_wg': 8, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 4},
        {'waves_per_wg': 8, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 8, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
    ],
    (16, 'tile_grouped', 'v2'): [
        {'waves_per_wg': 1, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 1, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 16, 'swizzle_type': 'None', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 16, 'swizzle_type': 'None', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 16, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 16, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 2, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 2, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 3, 'swizzle_type': 'None', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 3, 'swizzle_type': 'None', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 3, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 3, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 4, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 4, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 5, 'swizzle_type': 'None', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 5, 'swizzle_type': 'None', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 5, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 5, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 6, 'swizzle_type': 'None', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 6, 'swizzle_type': 'None', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 6, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 6, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 7, 'swizzle_type': 'None', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 7, 'swizzle_type': 'None', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 7, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 7, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 8, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 8, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 1},
        {'waves_per_wg': 8, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 2},
        {'waves_per_wg': 8, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 4},
        {'waves_per_wg': 8, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 8, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
    ],
    (32, 'tile_dense', 'v3'): [
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 1, 'c_slices_per_wave': 1, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 1, 'c_slices_per_wave': 1, 'swizzle_type': 'None', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 1, 'c_slices_per_wave': 2, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 1, 'c_slices_per_wave': 3, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 1, 'c_slices_per_wave': 4, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 1, 'c_slices_per_wave': 6, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 16, 'c_slices_per_wave': 2, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 16, 'c_slices_per_wave': 2, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 16, 'c_slices_per_wave': 2, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 16, 'c_slices_per_wave': 2, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 16, 'c_slices_per_wave': 3, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 16, 'c_slices_per_wave': 3, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 16, 'c_slices_per_wave': 3, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 16, 'c_slices_per_wave': 3, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 16, 'c_slices_per_wave': 4, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 16, 'c_slices_per_wave': 4, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 16, 'c_slices_per_wave': 4, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 16, 'c_slices_per_wave': 4, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 2, 'c_slices_per_wave': 1, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 2, 'c_slices_per_wave': 1, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 2, 'c_slices_per_wave': 1, 'swizzle_type': 'None', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 2, 'c_slices_per_wave': 1, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 2, 'c_slices_per_wave': 1, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 2, 'c_slices_per_wave': 2, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 2, 'c_slices_per_wave': 2, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 2, 'c_slices_per_wave': 2, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 2, 'c_slices_per_wave': 2, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 2, 'c_slices_per_wave': 3, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 2, 'c_slices_per_wave': 3, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 2, 'c_slices_per_wave': 3, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 2, 'c_slices_per_wave': 3, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 2, 'c_slices_per_wave': 4, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 2, 'c_slices_per_wave': 4, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 2, 'c_slices_per_wave': 4, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 2, 'c_slices_per_wave': 4, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 3, 'c_slices_per_wave': 1, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 3, 'c_slices_per_wave': 1, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 3, 'c_slices_per_wave': 2, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 3, 'c_slices_per_wave': 2, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 3, 'c_slices_per_wave': 3, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 4, 'c_slices_per_wave': 1, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 4, 'c_slices_per_wave': 1, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 4, 'c_slices_per_wave': 1, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 4, 'c_slices_per_wave': 1, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 4, 'c_slices_per_wave': 2, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 4, 'c_slices_per_wave': 2, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 4, 'c_slices_per_wave': 2, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 4, 'c_slices_per_wave': 2, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 4, 'c_slices_per_wave': 3, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 4, 'c_slices_per_wave': 3, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 4, 'c_slices_per_wave': 3, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 4, 'c_slices_per_wave': 3, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 4, 'c_slices_per_wave': 4, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 4, 'c_slices_per_wave': 4, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 4, 'c_slices_per_wave': 4, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 4, 'c_slices_per_wave': 4, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 5, 'c_slices_per_wave': 1, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 5, 'c_slices_per_wave': 1, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 5, 'c_slices_per_wave': 2, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 6, 'c_slices_per_wave': 1, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 6, 'c_slices_per_wave': 1, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 7, 'c_slices_per_wave': 1, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 7, 'c_slices_per_wave': 1, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 8, 'c_slices_per_wave': 1, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 8, 'c_slices_per_wave': 1, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 8, 'c_slices_per_wave': 1, 'swizzle_type': 'None', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 8, 'c_slices_per_wave': 1, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 8, 'c_slices_per_wave': 1, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 8, 'c_slices_per_wave': 2, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 8, 'c_slices_per_wave': 2, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 8, 'c_slices_per_wave': 2, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 8, 'c_slices_per_wave': 2, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 8, 'c_slices_per_wave': 3, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 8, 'c_slices_per_wave': 3, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 8, 'c_slices_per_wave': 3, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 8, 'c_slices_per_wave': 3, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 8, 'c_slices_per_wave': 4, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 8, 'c_slices_per_wave': 4, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 8, 'c_slices_per_wave': 4, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'mfma_shape': 'M16N16K32', 'waves_per_wg': 8, 'c_slices_per_wave': 4, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
    ],
    (32, 'tile_grouped', 'v2'): [
        {'waves_per_wg': 10, 'swizzle_type': 'None', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 10, 'swizzle_type': 'None', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 10, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 10, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 12, 'swizzle_type': 'None', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 12, 'swizzle_type': 'None', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 12, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 12, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 14, 'swizzle_type': 'None', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 14, 'swizzle_type': 'None', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 14, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 14, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 16, 'swizzle_type': 'None', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 16, 'swizzle_type': 'None', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 16, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 16, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 2, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 2, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 4, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 4, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 4, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 4, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 6, 'swizzle_type': 'None', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 6, 'swizzle_type': 'None', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 6, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 6, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 8, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 8, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 1},
        {'waves_per_wg': 8, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 16},
        {'waves_per_wg': 8, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 2},
        {'waves_per_wg': 8, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 4},
        {'waves_per_wg': 8, 'swizzle_type': 'CyclicShift', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 8, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8},
        {'waves_per_wg': 8, 'swizzle_type': 'XOR', 'epilogue': 'RegistersToLdsToGlobalMemory', 'vector_size': 8},
    ],
}


# ---------------------------------------------------------------------------
# Full rule set
# ---------------------------------------------------------------------------
#
# Each family is generated as a cartesian product over its parameter axes, then
# lightly pruned to keep only kernel-valid combinations. The axes are chosen as
# supersets of the profiler tables so that the generative output covers (>=) the
# profiler set. The codegen key ORDER per family must match the profiler tables
# (designated-initializer field order).
#
# Swizzle policy: XOR and CyclicShift are the preferred swizzles and are emitted
# programmatically. The "None" swizzle is an exception reserved for the
# parameter combinations where neither XOR nor CyclicShift is applicable. Rather
# than re-deriving that applicability formula, the generators take the required
# None instances straight from the profiler tables (see
# _none_exception_rows): the None entries there are exactly the cases that
# needed the fallback.

# Common epilogue / vector-size axes.
_EPILOGUES = ("RegistersToGlobalMemory", "RegistersToLdsToGlobalMemory")


def _emit(config_keys_order, fields):
    """Build an ordered config dict from (key, value) fields, enforcing the
    canonical key order used by the faithful tables. The variant-dependent
    'direction' field is NOT included here; it is spliced in later by
    _stamp_direction."""
    d = {}
    for k in config_keys_order:
        d[k] = fields[k]
    return d


def _none_exception_rows(family_key):
    """Return the None-swizzle rows the JSON config actually requires for a
    family. These are the fallback cases where XOR/CyclicShift do not apply; the
    faithful tables (the JSON union) carry exactly those entries."""
    return [
        dict(row)
        for row in _PROFILER_TABLES.get(family_key, [])
        if row.get('swizzle_type') == 'None'
    ]


def _dedupe_rows(rows):
    """Drop duplicate config rows (order-independent), preserving first order."""
    seen = set()
    out = []
    for row in rows:
        key = frozenset(row.items())
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _gen_4c():
    """4c tile_grouped v3: (waves_c64, waves_q4) grids + swizzle/epilogue/vec."""
    order = ['waves_c64', 'waves_q4', 'swizzle_type', 'epilogue', 'vector_size']
    rows = []
    grids = [(1, 1), (2, 1), (2, 2), (2, 4), (2, 8)]
    for wc, wq in grids:
        # Preferred swizzle: XOR vec8, both epilogues.
        for ep in _EPILOGUES:
            rows.append(_emit(order, {
                'waves_c64': wc, 'waves_q4': wq,
                'swizzle_type': 'XOR', 'epilogue': ep, 'vector_size': 8}))
        # CyclicShift small-vector fallbacks (DRAM epilogue) for wide-q grids.
        if wq >= 4:
            for vec in (1, 2):
                rows.append(_emit(order, {
                    'waves_c64': wc, 'waves_q4': wq,
                    'swizzle_type': 'CyclicShift',
                    'epilogue': 'RegistersToGlobalMemory', 'vector_size': vec}))
    return rows


def _gen_grouped_8_16(channel_family):
    """8c/16c tile_grouped v2: waves_per_wg 1..8,16 over swizzle/epilogue."""
    order = ['waves_per_wg', 'swizzle_type', 'epilogue', 'vector_size']
    rows = []
    for w in (1, 2, 3, 4, 5, 6, 7, 8, 16):
        # Preferred swizzle: XOR vec8, both epilogues.
        for ep in _EPILOGUES:
            rows.append(_emit(order, {
                'waves_per_wg': w,
                'swizzle_type': 'XOR', 'epilogue': ep, 'vector_size': 8}))
        # CyclicShift small-vector fallbacks at the largest power-of-two grid.
        if w == 8:
            for vec in (1, 2, 4):
                rows.append(_emit(order, {
                    'waves_per_wg': w,
                    'swizzle_type': 'CyclicShift',
                    'epilogue': 'RegistersToLdsToGlobalMemory',
                    'vector_size': vec}))
            rows.append(_emit(order, {
                'waves_per_wg': w,
                'swizzle_type': 'CyclicShift',
                'epilogue': 'RegistersToGlobalMemory', 'vector_size': 8}))
    return rows


def _gen_32c_grouped():
    """32c tile_grouped v2: waves_per_wg even 2..16 over swizzle/epilogue."""
    order = ['waves_per_wg', 'swizzle_type', 'epilogue', 'vector_size']
    rows = []
    for w in (2, 4, 6, 8, 10, 12, 14, 16):
        # Preferred swizzle: XOR vec8, both epilogues.
        for ep in _EPILOGUES:
            rows.append(_emit(order, {
                'waves_per_wg': w,
                'swizzle_type': 'XOR', 'epilogue': ep, 'vector_size': 8}))
        if w in (4, 8):
            for ep in _EPILOGUES:
                rows.append(_emit(order, {
                    'waves_per_wg': w,
                    'swizzle_type': 'CyclicShift', 'epilogue': ep,
                    'vector_size': 8}))
        if w == 8:
            for vec in (1, 2, 4, 16):
                rows.append(_emit(order, {
                    'waves_per_wg': w,
                    'swizzle_type': 'CyclicShift',
                    'epilogue': 'RegistersToLdsToGlobalMemory',
                    'vector_size': vec}))
    return rows


def _gen_32c_dense():
    """32c tile_dense v3: (waves_per_wg, c_slices_per_wave) grid over swizzle/
    epilogue, M16N16K32 only."""
    order = ['mfma_shape', 'waves_per_wg', 'c_slices_per_wave',
             'swizzle_type', 'epilogue', 'vector_size']
    rows = []
    for w in (1, 2, 3, 4, 5, 6, 7, 8, 16):
        for cspw in (1, 2, 3, 4, 6):
            # Preferred swizzles: CyclicShift (any waves) and XOR (power-of-two
            # waves). None is added as an exception via _none_exception_rows.
            for sw in ('CyclicShift', 'XOR'):
                if sw == 'XOR' and (w & (w - 1)) != 0:
                    continue
                for ep in _EPILOGUES:
                    rows.append(_emit(order, {
                        'mfma_shape': 'M16N16K32', 'waves_per_wg': w,
                        'c_slices_per_wave': cspw,
                        'swizzle_type': sw,
                        'epilogue': ep, 'vector_size': 8}))
    return rows


def _full_tables():
    """Assemble the full table set keyed like _PROFILER_TABLES.

    Each family's programmatic XOR/CyclicShift rows are augmented with the
    None-swizzle EXCEPTION rows, then de-duplicated.
    """
    generated = {
        (4, 'tile_grouped', 'v3'): _gen_4c(),
        (8, 'tile_grouped', 'v2'): _gen_grouped_8_16(8),
        (16, 'tile_grouped', 'v2'): _gen_grouped_8_16(16),
        (32, 'tile_dense', 'v3'): _gen_32c_dense(),
        (32, 'tile_grouped', 'v2'): _gen_32c_grouped(),
    }
    return {
        family_key: _dedupe_rows(rows + _none_exception_rows(family_key))
        for family_key, rows in generated.items()
    }


# ---------------------------------------------------------------------------
# Config materialisation from tables
# ---------------------------------------------------------------------------


def _build_from_tables(tables, arch, variants, ndims, datatypes):
    """Expand a family->rows table into a flat list of DirectConvKernelConfig,
    with deterministic, gap-free ids. Shared by the "profiler" and "full" rule
    sets (and, indirectly, by the derived subsets which renumber afterwards)."""
    configs: List[DirectConvKernelConfig] = []
    next_id = 0

    # Deterministic iteration order: ndim, family key, variant, datatype, row.
    for ndim in ndims:
        if ndim != _SUPPORTED_NDIM:
            continue
        for (channel_family, impl, version), rows in tables.items():
            for variant in variants:
                if variant not in _SUPPORTED_VARIANTS:
                    continue
                direction = _VARIANT_DIRECTION[variant]
                for datatype in datatypes:
                    if datatype not in _SUPPORTED_DATATYPES:
                        continue
                    for row in rows:
                        config = _stamp_direction(row, direction)
                        inst = {
                            "channel_family": channel_family,
                            "impl": impl,
                            "version": version,
                            "config": config,
                        }
                        if not direct_conv_supported_on_arch(
                            inst, variant, arch
                        ):
                            continue
                        configs.append(
                            DirectConvKernelConfig(
                                channel_family=channel_family,
                                impl=impl,
                                id=next_id,
                                config=config,
                                version=version,
                                variant=variant,
                                ndim_spatial=ndim,
                                layout=_SUPPORTED_LAYOUT,
                                datatype=datatype,
                                arch=arch,
                            )
                        )
                        next_id += 1

    return configs


def _renumber(configs):
    """Reassign deterministic, gap-free ids 0..N-1 after sub-selection."""
    for new_id, cfg in enumerate(configs):
        cfg.id = new_id
    return configs


# ---------------------------------------------------------------------------
# Stratified sub-selection 
# ---------------------------------------------------------------------------

def _classify_config(cfg) -> str:
    """Feature category for stratified selection.
    The category folds the kernel-defining axes (channel family, impl,
    variant) together with the datatype -- guaranteeing the subset keeps at
    least one config per (family, impl, variant, datatype)."""
    variant = getattr(cfg.variant, "value", cfg.variant)
    dt = getattr(cfg, "datatype", None) or "fp16"
    return f"{cfg.channel_family}c:{cfg.impl}:{variant}:{dt}"


def _select_test_configs(configs):
    """Select ~20% of configs with stratified sampling (develop convention):
    every 5th config (ranks 4, 9, 14, ...) within each feature category, with a
    minimum of one config per category."""
    from collections import defaultdict

    configs = list(configs)
    categories = defaultdict(list)  # category -> list of original indices
    for idx, cfg in enumerate(configs):
        categories[_classify_config(cfg)].append(idx)

    selected = set()
    for cat_indices in categories.values():
        picked = False
        for rank, idx in enumerate(cat_indices):
            if (rank + 1) % 5 == 0:
                selected.add(idx)
                picked = True
        if not picked and cat_indices:
            selected.add(cat_indices[0])

    return [configs[i] for i in sorted(selected)]


def _select_tiny_configs(configs):
    """One config per channel family, taken from the full set."""
    from collections import OrderedDict

    by_family = OrderedDict()
    for cfg in configs:
        by_family.setdefault(cfg.channel_family, cfg)
    return list(by_family.values())


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

# Table-backed base rule sets: the full instance universe ("full") and its
# curated profiler subset ("profiler").
_RULE_TABLE_BUILDERS = {
    "profiler": lambda: _PROFILER_TABLES,
    "full": _full_tables,
}

# Derived rule sets: each builds one of the base sets, then sub-selects.
#   tests       -> 20% stratified slice of the profiler set
#   full-tests  -> 20% stratified slice of the full set
#   tiny        -> one instance per channel family from the full set
_DERIVED_RULE_SETS = {
    "tests": ("profiler", _select_test_configs),
    "full-tests": ("full", _select_test_configs),
    "tiny": ("full", _select_tiny_configs),
}


def get_configs(
    arch: str = "gfx950",
    variants=None,
    ndims=None,
    datatypes=None,
    subset: str = "profiler",
    rule_set: str = "profiler",
) -> List[DirectConvKernelConfig]:
    """Generate direct-conv kernel configs for a requested domain.

    Args:
        arch:      target GPU arch (e.g. "gfx950", "gfx942"). Instances not
                   supported on the arch are filtered out.
        variants:  iterable of GroupedConvVariant (forward / bwd_data). None ->
                   all supported variants.
        ndims:     iterable of spatial dims. Direct-conv is 2D-only; any value
                   other than 2 yields no instances.
        datatypes: iterable of "fp16"/"bf16". None -> all supported.
        subset:    accepted for signature parity with develop's rule modules;
                   does not prune instances (the rule_set selects the set).
        rule_set:  one of "profiler", "full", "tests", "full-tests", "tiny".

    Returns:
        List[DirectConvKernelConfig] with deterministic, gap-free ids.
    """
    if variants is None:
        variants = list(_SUPPORTED_VARIANTS)
    if ndims is None:
        ndims = [_SUPPORTED_NDIM]
    if datatypes is None:
        datatypes = list(_SUPPORTED_DATATYPES)

    if rule_set in _RULE_TABLE_BUILDERS:
        tables = _RULE_TABLE_BUILDERS[rule_set]()
        return _build_from_tables(tables, arch, variants, ndims, datatypes)

    if rule_set in _DERIVED_RULE_SETS:
        base_name, selector = _DERIVED_RULE_SETS[rule_set]
        base_tables = _RULE_TABLE_BUILDERS[base_name]()
        base_configs = _build_from_tables(
            base_tables, arch, variants, ndims, datatypes
        )
        return _renumber(selector(base_configs))

    raise ValueError(
        f"unknown direct_conv rule_set {rule_set!r}; expected one of "
        f"{sorted(list(_RULE_TABLE_BUILDERS) + list(_DERIVED_RULE_SETS))}"
    )


# ---------------------------------------------------------------------------
# Uniform rule-set entry points
# ---------------------------------------------------------------------------
#
# These mirror the per-rule-set entry points of develop's grouped_conv rule
# modules so that the unified codegen can call direct-conv alongside the
# implicit-GEMM rules with a single ``(arch, variants, ndims, datatypes)``
# signature. Each simply selects the corresponding direct-conv rule_set. There
# is intentionally no direct-conv entry point for the "default" rule set.


def get_full_configs(arch, variants, ndims, datatypes):
    """Direct-conv instances for the "full" rule set."""
    return get_configs(arch, variants, ndims, datatypes, rule_set="full")


def get_full_test_configs(arch, variants, ndims, datatypes):
    """Direct-conv instances for the "full-tests" rule set."""
    return get_configs(arch, variants, ndims, datatypes, rule_set="full-tests")


def get_profiler_configs(arch, variants, ndims, datatypes):
    """Direct-conv instances for the "profiler" rule set."""
    return get_configs(arch, variants, ndims, datatypes, rule_set="profiler")


def get_test_configs(arch, variants, ndims, datatypes):
    """Direct-conv instances for the "tests" rule set."""
    return get_configs(arch, variants, ndims, datatypes, rule_set="tests")


def get_tiny_configs(arch, variants, ndims, datatypes):
    """Direct-conv instances for the "tiny" rule set."""
    return get_configs(arch, variants, ndims, datatypes, rule_set="tiny")
