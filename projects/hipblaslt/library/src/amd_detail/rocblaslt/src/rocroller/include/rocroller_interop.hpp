// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * Hands hipBLASLt's kernel descriptions to rocRoller.
 *
 * hipBLASLt describes a kernel's operands with its own enumerations
 * (kernel_data_types.hpp), which the tile ranker uses in every configuration.
 * rocRoller, the generator hipBLASLt drives when it is enabled, has
 * enumerations of its own for the same formats and gives them the same values,
 * so a conversion is a cast. The static assertions below are what entitles the
 * cast: if rocRoller renumbers either enumeration, they fail here rather than
 * letting a kernel be generated for the wrong format.
 *
 * Only sources built against rocRoller's headers include this. Everything that
 * reaches the ranker stops at kernel_data_types.hpp.
 *
 * A failure here means rocRoller renumbered an enumeration. Take the new value
 * from rocRoller/DataTypes/DataTypes.hpp or rocRoller/Operations/BlockScale_fwd.hpp
 * and update kernel_data_types.hpp.
 */

#include "kernel_data_types.hpp"

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Operations/BlockScale_fwd.hpp>

#include <ostream>
#include <string>
#include <type_traits>

namespace hipblaslt
{
    static_assert(std::is_same_v<std::underlying_type_t<rocRoller::DataType>,
                                 std::underlying_type_t<DataType>>,
                  "rocRoller::DataType changed its underlying type; update kernel_data_types.hpp.");
    static_assert(std::is_same_v<std::underlying_type_t<rocRoller::Operations::ScaleMode>,
                                 std::underlying_type_t<ScaleMode>>,
                  "rocRoller::Operations::ScaleMode changed its underlying type; update "
                  "kernel_data_types.hpp.");

#define HIPBLASLT_ASSERT_SAME_DATA_TYPE(name)                                                      \
    static_assert(static_cast<int>(DataType::name) == static_cast<int>(rocRoller::DataType::name), \
                  "rocRoller::DataType::" #name " changed value; update kernel_data_types.hpp.")

    HIPBLASLT_ASSERT_SAME_DATA_TYPE(Float);
    HIPBLASLT_ASSERT_SAME_DATA_TYPE(Double);
    HIPBLASLT_ASSERT_SAME_DATA_TYPE(Half);
    HIPBLASLT_ASSERT_SAME_DATA_TYPE(BFloat16);
    HIPBLASLT_ASSERT_SAME_DATA_TYPE(FP8);
    HIPBLASLT_ASSERT_SAME_DATA_TYPE(BF8);
    HIPBLASLT_ASSERT_SAME_DATA_TYPE(FP4);
    HIPBLASLT_ASSERT_SAME_DATA_TYPE(FP6);
    HIPBLASLT_ASSERT_SAME_DATA_TYPE(BF6);
    HIPBLASLT_ASSERT_SAME_DATA_TYPE(Int32);
    HIPBLASLT_ASSERT_SAME_DATA_TYPE(E8M0);
    HIPBLASLT_ASSERT_SAME_DATA_TYPE(None);

#undef HIPBLASLT_ASSERT_SAME_DATA_TYPE

    static_assert(static_cast<int>(ScaleMode::None)
                      == static_cast<int>(rocRoller::Operations::ScaleMode::None),
                  "rocRoller::Operations::ScaleMode::None changed value; update "
                  "kernel_data_types.hpp.");
    static_assert(static_cast<int>(ScaleMode::Separate)
                      == static_cast<int>(rocRoller::Operations::ScaleMode::Separate),
                  "rocRoller::Operations::ScaleMode::Separate changed value; update "
                  "kernel_data_types.hpp.");

    /**
     * @brief The rocRoller format that stands for this hipBLASLt one.
     */
    inline constexpr rocRoller::DataType toRocRoller(DataType type)
    {
        return static_cast<rocRoller::DataType>(type);
    }

    /**
     * @brief The rocRoller scaling arrangement that stands for this hipBLASLt one.
     */
    inline constexpr rocRoller::Operations::ScaleMode toRocRoller(ScaleMode mode)
    {
        return static_cast<rocRoller::Operations::ScaleMode>(mode);
    }

    // The spellings rocRoller gives these formats. Generated kernel names are
    // built from them, so they have to read exactly as rocRoller's do; deferring
    // keeps that true without a second table to maintain. Found by argument
    // lookup, which is why callers that already said toString(x) or stream x
    // need no change.
    inline std::string toString(DataType type)
    {
        return rocRoller::toString(toRocRoller(type));
    }

    inline std::string toString(ScaleMode mode)
    {
        return rocRoller::Operations::toString(toRocRoller(mode));
    }

    inline std::ostream& operator<<(std::ostream& stream, DataType type)
    {
        return stream << toRocRoller(type);
    }

    inline std::ostream& operator<<(std::ostream& stream, ScaleMode mode)
    {
        return stream << toRocRoller(mode);
    }
} // namespace hipblaslt
