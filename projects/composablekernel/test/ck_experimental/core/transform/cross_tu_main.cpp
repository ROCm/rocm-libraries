// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/// @file
/// @brief gtest entry for V4 cross-TU NTTP equality of SPLICED graphs.
///
/// Two assertions:
///   1. SAME spliced builder text in two TUs -> SAME mangled NTTP -> SAME
///      address. Proves spliceInto produces bit-identical TransformGraph
///      values regardless of TU; pool merge + edge-id remap + base_offset
///      rewrite are deterministic and free of padding-byte drift.
///   2. DIFFERENT spliced builder text in a third TU -> DIFFERENT mangled
///      NTTP -> DIFFERENT address. Proves the test mechanism has
///      discriminating power and isn't trivially passing via lld ICF on
///      structurally-similar canaries.

#include <gtest/gtest.h>

#include "cross_tu_graph.hpp"

TEST(V4CrossTU, SameSplicedBuilderProducesEqualNTTP)
{
    EXPECT_EQ(cross_tu_v4::tuACanary(), cross_tu_v4::tuBCanary())
        << "Two TUs running the SAME spliced builder produced different NTTP "
           "specializations -- splice canonicalization is broken across "
           "compilation units (likely padding-byte drift in TransformGraph<V> "
           "or non-deterministic spliceInto pool/edge layout).";
}

TEST(V4CrossTU, DifferentSplicedBuilderProducesDistinctNTTP)
{
    EXPECT_NE(cross_tu_v4::tuACanary(), cross_tu_v4::tuCCanary())
        << "Two TUs running structurally-different spliced builders produced "
           "the SAME NTTP specialization -- either operator== is too "
           "permissive or lld ICF folded distinct kCanaryFor<G> "
           "instantiations. Pass -Wl,--icf=none if the latter.";
}
