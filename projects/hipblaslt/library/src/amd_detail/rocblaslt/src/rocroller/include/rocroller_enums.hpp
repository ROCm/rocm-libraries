// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * Declarations of the two rocRoller enumerations the tile ranker names, for
 * translation units that are built without a rocRoller checkout.
 *
 * The ranker decides which kernel shapes suit a problem. It describes a kernel
 * with a KernelType, whose fields are rocRoller enumerators, but it calls no
 * rocRoller function: the dependency is these two enumerations and nothing
 * else. Taking them from rocRoller's own headers brought in 80 headers and a
 * libfmt clone, and made the ranker unbuildable in a checkout that leaves the
 * rocRoller sources out.
 *
 * Only translation units compiled with HIPBLASLT_USE_ROCROLLER see rocRoller's
 * declarations instead; the two must agree, because a KernelType built by the
 * ranker is read by the rocRoller call path.
 *
 * When you add an enumerator here, add the matching static_assert to
 * rocroller_enum_conformance.cpp, which is what checks these values against
 * rocRoller's headers.
 *
 * Do not include this header directly; kernel_type.hpp and analytical_utils.hpp
 * choose between it and rocRoller's headers.
 */

// Supplied to the ranker by rocRoller's data-type header before this one
// replaced it, and still used by the types declared in kernel_type.hpp.
#include <compare>
#include <cstddef>
#include <functional>
#include <vector>

namespace rocRoller
{
    // rocRoller fixes this enumeration's underlying type at int. Spelt the same
    // way here so that a KernelType has one layout across the whole library.
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
        None     = 38,
    };

    namespace Operations
    {
        // rocRoller leaves this one's underlying type unspecified, which for a
        // scoped enumeration makes it int. Spelt the same way for the same reason.
        enum class ScaleMode
        {
            None     = 0,
            Separate = 2,
        };
    } // namespace Operations
} // namespace rocRoller
