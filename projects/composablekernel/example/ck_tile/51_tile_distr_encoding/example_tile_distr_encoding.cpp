// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <stdio.h>
#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"

namespace ck_tile {
template <typename TileDistrEnc,
          index_t num_repeat,
          index_t mat_major_size,
          index_t mat_minor_size,
          index_t num_lanes,
          index_t num_vector_items>
struct NiceTileDistrEncWrapper
{
    // TODO: Add some static asserts to make sure this is a proper Tile Descr encoding for Lane
    // Vector mapping.

    static constexpr auto distr = make_static_tile_distribution(TileDistrEnc{});

    static auto calc_matrix_indices_from_lane_vector(index_t lane_inx, index_t vector_inx)
    {
        using arr1           = array<index_t, 1>;
        const auto ps_ys_idx = container_concat(arr1{lane_inx}, arr1{vector_inx});
        const auto window_adaptor_thread_coord_tmp =
            make_tensor_adaptor_coordinate(distr.get_ps_ys_to_xs_adaptor(), ps_ys_idx);
        return window_adaptor_thread_coord_tmp.get_bottom_index();
    }

    // TODO: Ugly inverse mapping implementation mat_major, mat_minor, rep -> LV;
    // Static storage duration variables are zero-initialized by default.
    static inline index_t inverse_map_valid[64][128][4];
    static inline index_t inverse_map_lane[64][128][4];
    static inline index_t inverse_map_vec[64][128][4];

    static void calc_inverse_map()
    {
        for(index_t l = 0; l < num_lanes; l++)
        {
            for(index_t v = 0; v < num_vector_items; v++)
            {
                auto res = calc_matrix_indices_from_lane_vector(l, v); // Matrix major, minor inx;

                // We assume that repeated matrix elements appear at increasing L and V indices.
                for(index_t r = 0; r < num_repeat; r++)
                {
                    if(!inverse_map_valid[res[0]][res[1]][r])
                    {
                        inverse_map_valid[res[0]][res[1]][r] = 1;
                        inverse_map_lane[res[0]][res[1]][r]  = l;
                        inverse_map_vec[res[0]][res[1]][r]   = v;
                    }
                }
            }
        }
    }

    static array<index_t, 2>
    calc_lane_vector_from_matrix_indices(ck_tile::index_t mat_major_inx, // M or N
                                         ck_tile::index_t mat_minor_inx, // K or M
                                         ck_tile::index_t repeat_inx = 0)
    {
        if(inverse_map_valid[mat_major_inx][mat_minor_inx][repeat_inx])
        {
            return {inverse_map_lane[mat_major_inx][mat_minor_inx][repeat_inx],
                    inverse_map_vec[mat_major_inx][mat_minor_inx][repeat_inx]};
        }
        return {-1, -1};
    }

    static void print_mapping()
    {
        printf("(lane, vector) item to matrix element\n L | ");
        for(index_t v = 0; v < num_vector_items; v++)
        {
            printf("vec%2d | ", v);
        }
        printf("\n");

        for(index_t l = 0; l < num_lanes; l++)
        {
            printf("%2d | ", l);
            for(index_t v = 0; v < num_vector_items; v++)
            {
                auto res = calc_matrix_indices_from_lane_vector(l, v);
                printf("%2d %2d | ", res[0], res[1]);
            }
            printf("\n");
        }
    }

    static void print_inverse_mapping()
    {
        printf("Matrix element to (lane, vector) item. Elements are replicated an additional %d "
               "time(s) in higher lanes. \n",
               num_repeat - 1);
        printf("Mat| ");
        for(index_t k = 0; k < mat_minor_size; k++)
        {
            printf("   %2d | ", k);
        }
        printf("\n");

        for(index_t m = 0; m < mat_major_size; m++)
        {
            printf("%2d | ", m);
            for(index_t k = 0; k < mat_minor_size; k++)
            {
                auto res = calc_lane_vector_from_matrix_indices(m, k, 0);
                printf("%2d %2d | ", res[0], res[1]);
            }
            printf("\n");
        }
    }
};
} // namespace ck_tile

int main()
{
    using namespace ck_tile;

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

    // constexpr index_t mat_major_size =
    //     container_reduce(typename Encoding::HsLengthss{}[number<0>{}], multiplies<>{}, 1);
    // constexpr index_t mat_minor_size =
    //     container_reduce(typename Encoding::HsLengthss{}[number<0>{}], multiplies<>{}, 1);
    // constexpr index_t repeat_size = Encoding::RsLengths{}[number<0>{}];
    // auto lengths = distr.get_lengths();
    // printf("Got lengths %d %d\n", lengths.get(number<0>{}).value,
    // lengths.get(number<1>{}).value);

    // TODO: We should be able to get all these lengths from the tile distr itself (as tried above)
    // but it's a bit awkard.
    constexpr index_t num_repeat       = 2;
    constexpr index_t mat_major_size   = 16; // M or N
    constexpr index_t mat_minor_size   = 16; // K or M
    constexpr index_t num_lanes        = 32;
    constexpr index_t num_vector_items = 16;

    using NTDEW = NiceTileDistrEncWrapper<Encoding,
                                          num_repeat,
                                          mat_major_size,
                                          mat_minor_size,
                                          num_lanes,
                                          num_vector_items>;

    NTDEW::print_mapping();
    NTDEW::calc_inverse_map();
    NTDEW::print_inverse_mapping();

    return 0;
}
