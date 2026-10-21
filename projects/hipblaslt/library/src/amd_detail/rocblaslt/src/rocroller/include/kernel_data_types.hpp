// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * The element formats and scaling arrangements hipBLASLt uses to describe a
 * checked-in assembly kernel.
 *
 * A KernelType (kernel_type.hpp) says what a kernel's operands look like, and
 * the tile ranker reads it to decide which kernel shapes suit a problem. Those
 * descriptions are hipBLASLt's own: they are matched against kernels that ship
 * in this repository as assembly, and the ranker runs in every configuration,
 * including one built with no kernel generator present at all.
 *
 * The values are the ones rocRoller gives the same formats. rocRoller is the
 * generator hipBLASLt currently drives for these kernels, and a KernelType the
 * ranker fills in is handed to it; matching values make that hand-off a cast
 * (rocroller_interop.hpp) instead of a lookup table. Nothing here depends on
 * rocRoller being present, and these are the declarations every translation
 * unit uses, whether or not it is built against rocRoller's headers.
 */

namespace hipblaslt
{
    /**
     * @brief The format of one element of a kernel operand.
     *
     * Only the formats the ranker and the kernels it ranks actually name are
     * listed. The gaps are formats rocRoller knows and hipBLASLt does not use;
     * leaving the numbering alone is what keeps the hand-off a cast.
     */
    enum class DataType : int
    {
        Float    = 0,
        Double   = 1,
        Half     = 4,
        BFloat16 = 6,
        FP8      = 8,
        BF8      = 10,
        FP4      = 12,
        FP6      = 14,
        BF6      = 16,
        Int32    = 21,
        E8M0     = 32,
        None     = 38, //< Any, unknown, or not yet decided.
    };

    /**
     * @brief How a kernel operand's block scales are supplied.
     *
     * Separate means the scales arrive in their own tensor, which is what every
     * checked-in MX kernel expects. A generator may offer further arrangements;
     * those are named at the boundary where they are needed, not here.
     */
    enum class ScaleMode : int
    {
        None     = 0, //< Operand is not scaled.
        Separate = 2, //< Scales are a tensor of their own.
    };
} // namespace hipblaslt
