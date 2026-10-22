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
// A failure here means rocRoller renumbered an enumeration. Take the new value
// from rocRoller/DataTypes/DataTypes.hpp or rocRoller/Operations/BlockScale_fwd.hpp
// and update rocroller_enums.hpp.
//
// There is one assertion per enumerator declared in rocroller_enums.hpp. When
// you add an enumerator there, add its assertion here.
//
// The file emits no code; the static assertions are the whole point.

#include <rocRoller/DataTypes/DataTypes.hpp>
#include <rocRoller/Operations/Command.hpp>

#include <type_traits>

static_assert(std::is_same_v<std::underlying_type_t<rocRoller::DataType>, int>,
              "rocRoller::DataType changed its underlying type; update rocroller_enums.hpp.");
static_assert(
    std::is_same_v<std::underlying_type_t<rocRoller::Operations::ScaleMode>, int>,
    "rocRoller::Operations::ScaleMode changed its underlying type; update rocroller_enums.hpp.");

static_assert(static_cast<int>(rocRoller::DataType::Float) == 0,
              "rocRoller::DataType::Float changed value; update rocroller_enums.hpp.");
static_assert(static_cast<int>(rocRoller::DataType::Double) == 1,
              "rocRoller::DataType::Double changed value; update rocroller_enums.hpp.");
static_assert(static_cast<int>(rocRoller::DataType::Half) == 4,
              "rocRoller::DataType::Half changed value; update rocroller_enums.hpp.");
static_assert(static_cast<int>(rocRoller::DataType::BFloat16) == 6,
              "rocRoller::DataType::BFloat16 changed value; update rocroller_enums.hpp.");
static_assert(static_cast<int>(rocRoller::DataType::FP8) == 8,
              "rocRoller::DataType::FP8 changed value; update rocroller_enums.hpp.");
static_assert(static_cast<int>(rocRoller::DataType::BF8) == 10,
              "rocRoller::DataType::BF8 changed value; update rocroller_enums.hpp.");
static_assert(static_cast<int>(rocRoller::DataType::FP4) == 12,
              "rocRoller::DataType::FP4 changed value; update rocroller_enums.hpp.");
static_assert(static_cast<int>(rocRoller::DataType::FP6) == 14,
              "rocRoller::DataType::FP6 changed value; update rocroller_enums.hpp.");
static_assert(static_cast<int>(rocRoller::DataType::BF6) == 16,
              "rocRoller::DataType::BF6 changed value; update rocroller_enums.hpp.");
static_assert(static_cast<int>(rocRoller::DataType::Int32) == 21,
              "rocRoller::DataType::Int32 changed value; update rocroller_enums.hpp.");
static_assert(static_cast<int>(rocRoller::DataType::E8M0) == 32,
              "rocRoller::DataType::E8M0 changed value; update rocroller_enums.hpp.");
static_assert(static_cast<int>(rocRoller::DataType::None) == 38,
              "rocRoller::DataType::None changed value; update rocroller_enums.hpp.");

static_assert(static_cast<int>(rocRoller::Operations::ScaleMode::None) == 0,
              "rocRoller::Operations::ScaleMode::None changed value; update rocroller_enums.hpp.");
static_assert(
    static_cast<int>(rocRoller::Operations::ScaleMode::Separate) == 2,
    "rocRoller::Operations::ScaleMode::Separate changed value; update rocroller_enums.hpp.");
