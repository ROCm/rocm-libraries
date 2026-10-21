// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Checks the enumerations rocroller_enums.hpp declares against rocRoller's own.
//
// The tile ranker is built without a rocRoller checkout, so it declares the two
// rocRoller enumerations it names itself. A KernelType it fills in is read by
// the rocRoller call path, which is built against rocRoller's headers, so the
// two declarations have to agree on every value. This translation unit is the
// only one that sees both, and it is compiled only when rocRoller is enabled --
// which is also the only configuration in which disagreeing would matter.
//
// A failure here means rocRoller renumbered an enumeration. Read the new value
// out of the header named in the message and update rocroller_enum_list.hpp.
//
// The file emits no code; the static assertions are the whole point.

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Operations/Command.hpp>

#include <type_traits>

#include "rocroller_enum_list.hpp"

static_assert(std::is_same_v<std::underlying_type_t<rocRoller::DataType>, int>,
              "rocRoller::DataType changed its underlying type; update the hipBLASLt "
              "declaration in rocroller_enums.hpp.");

static_assert(std::is_same_v<std::underlying_type_t<rocRoller::Operations::ScaleMode>, int>,
              "rocRoller::Operations::ScaleMode changed its underlying type; update the "
              "hipBLASLt declaration in rocroller_enums.hpp.");

#define HIPBLASLT_CHECK_DATA_TYPE(name, value)                            \
    static_assert(static_cast<int>(rocRoller::DataType::name) == (value), \
                  "rocRoller::DataType::" #name " changed value; update " \
                  "HIPBLASLT_ROCROLLER_DATA_TYPES in rocroller_enum_list.hpp.");
HIPBLASLT_ROCROLLER_DATA_TYPES(HIPBLASLT_CHECK_DATA_TYPE)
#undef HIPBLASLT_CHECK_DATA_TYPE

#define HIPBLASLT_CHECK_SCALE_MODE(name, value)                                        \
    static_assert(static_cast<int>(rocRoller::Operations::ScaleMode::name) == (value), \
                  "rocRoller::Operations::ScaleMode::" #name " changed value; update " \
                  "HIPBLASLT_ROCROLLER_SCALE_MODES in rocroller_enum_list.hpp.");
HIPBLASLT_ROCROLLER_SCALE_MODES(HIPBLASLT_CHECK_SCALE_MODE)
#undef HIPBLASLT_CHECK_SCALE_MODE
