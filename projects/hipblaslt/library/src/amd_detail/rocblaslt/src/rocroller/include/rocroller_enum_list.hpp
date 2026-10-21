// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

/**
 * The rocRoller enumerators the tile ranker names, together with the numeric
 * value rocRoller assigns each one.
 *
 * Two places need this list and must agree on it: rocroller_enums.hpp declares
 * the enumerations for translation units built without a rocRoller checkout,
 * and rocroller_enum_conformance.cpp checks those values against rocRoller's
 * own headers in builds that have them. Both expand the lists below, so the
 * declarations and the check cannot drift apart.
 *
 * The values are part of an interface, not an implementation detail: a
 * KernelType is filled in by the ranker and read by the rocRoller call path,
 * and in a build with rocRoller enabled those two are compiled against
 * different declarations of the same enumeration.
 *
 * Add an entry when the ranker starts using another enumerator. Take the value
 * from rocRoller's own header; the conformance check will reject a wrong one.
 */

// rocRoller::DataType, declared in rocRoller/DataTypes/DataTypes.hpp.
#define HIPBLASLT_ROCROLLER_DATA_TYPES(X) \
    X(Float, 0)                           \
    X(Double, 1)                          \
    X(Half, 4)                            \
    X(BFloat16, 6)                        \
    X(FP8, 8)                             \
    X(BF8, 10)                            \
    X(FP4, 12)                            \
    X(FP6, 14)                            \
    X(BF6, 16)                            \
    X(Int32, 21)                          \
    X(E8M0, 32)                           \
    X(None, 38)

// rocRoller::Operations::ScaleMode, declared in
// rocRoller/Operations/BlockScale_fwd.hpp and reached through
// rocRoller/Operations/Command.hpp.
#define HIPBLASLT_ROCROLLER_SCALE_MODES(X) \
    X(None, 0)                             \
    X(Separate, 2)
