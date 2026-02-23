// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <stdio.h>
#include "ck_tile/core/arch/mma/utility.hpp"

using namespace ck_tile;

int main()
{
    // Define distribution encoding
    // Example RDNA3 V_WMMA_F32_16X16X16_F16 A Matrix (M, K)
    // L{RM} V{K}
    using Encoding =
        tile_distribution_encoding<sequence<2>, // R (= Repeat) Lanes 0-15 are duplicated at 16-31
                                   tuple<sequence<16>, sequence<16>>, // H (= Hidden dims = unmerged
                                                                      // dims) for M, K dimension
                                   tuple<sequence<0, 1>>, // P major (= Parallelism = lanes)
                                   tuple<sequence<0, 0>>, // P minor
                                   sequence<2>,           // Y major (= Yield = Vector items)
                                   sequence<0>            // Y minor
                                   >;

    TileDistrEncRegMap<Encoding>::print();

    // Example RDNA3 V_WMMA_F32_16X16X16_F16 C Matrix (M, N)
    // M{2, 1} L{M1N} V{M2M0} (dummy unmerge to be more similar to other layouts)
    using Encoding2 =
        tile_distribution_encoding<sequence<>,                             // R (= Repeat)
                                   tuple<sequence<8, 2, 1>, sequence<16>>, // H (= Hidden dims =
                                                                           // unmerged dims) for M,
                                                                           // N dimension
                                   tuple<sequence<1, 2>>, // P major (= Parallelism = lanes)
                                   tuple<sequence<1, 0>>, // P minor
                                   sequence<1, 1>,        // Y major (= Yield = Vector items)
                                   sequence<0, 2>         // Y minor
                                   >;

    TileDistrEncRegMap<Encoding2>::print();

    return 0;
}
