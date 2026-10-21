// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * Declarations of the two rocRoller enumerations the tile ranker names, for
 * translation units that are built without a rocRoller checkout.
 *
 * The ranker decides which kernel shapes suit a problem. It describes a kernel
 * with a KernelType, whose fields are rocRoller enumerators, but it calls no
 * rocRoller function: the dependency is a pair of enumerations and nothing
 * else. Pulling them from rocRoller's own headers brought in eighty headers
 * and a libfmt clone, and made the ranker unbuildable in a checkout that
 * leaves the rocRoller sources out.
 *
 * Only translation units that are compiled with HIPBLASLT_HAVE_ROCROLLER_HEADERS
 * see rocRoller's declarations instead. The underlying types and values match,
 * checked at build time by rocroller_enum_conformance.cpp.
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

#include "rocroller_enum_list.hpp"

namespace rocRoller
{
#define HIPBLASLT_ROCROLLER_ENUMERATOR(name, value) name = value,

    // Spelt with the same underlying type rocRoller fixes on the enumeration,
    // so that a KernelType has one layout across the whole library.
    enum class DataType : int
    {
        HIPBLASLT_ROCROLLER_DATA_TYPES(HIPBLASLT_ROCROLLER_ENUMERATOR)
    };

    namespace Operations
    {
        // rocRoller leaves this one's underlying type unspecified, which makes
        // it int; spelt the same way here for the same reason.
        enum class ScaleMode
        {
            HIPBLASLT_ROCROLLER_SCALE_MODES(HIPBLASLT_ROCROLLER_ENUMERATOR)
        };
    } // namespace Operations

#undef HIPBLASLT_ROCROLLER_ENUMERATOR
} // namespace rocRoller
