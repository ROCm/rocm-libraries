// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "ck_tile/core.hpp"
#include "ck_tile/host.hpp"
#include "../mx_gemm.hpp"

template <typename GemmConfig>
struct MXGemmArchTraits
{
    using Config = GemmConfig;

    // Preshuffle weights: port from mx flatmm arch traits
    template <typename dtype>
    static auto preShuffleWeight(ck_tile::HostTensor<dtype>& src)
    {
        constexpr ck_tile::index_t NLane = Config::N_Warp_Tile;
        auto src_lengths                 = src.get_lengths();
        const int K                      = src_lengths[0];
        const int N                      = src_lengths[1];
        constexpr int packed_size        = ck_tile::numeric_traits<dtype>::PackedSize;
        const int KPack = std::is_same_v<dtype, ck_tile::pk_fp6x16_t> ? 32 : 16 * packed_size;

        const int KLane = ck_tile::get_warp_size() / NLane;
        const int K0    = K / (KLane * KPack);

        ck_tile::HostTensor<dtype> shuffled(ck_tile::HostTensorDescriptor(
            {static_cast<std::size_t>(N * K)}, {static_cast<std::size_t>(1)}));

        for(int n = 0; n < N; ++n)
        {
            for(int k = 0; k < K; k += packed_size)
            {
                const int n0 = n / NLane;
                const int n1 = n % NLane;

                const int k0    = k / (KLane * KPack);
                const int tempk = k % (KLane * KPack);
                const int k1    = tempk / KPack;
                const int k2    = tempk % KPack;

                const int outputIndex = n0 * KPack * NLane * KLane * K0 +
                                        k0 * KPack * NLane * KLane + k1 * KPack * NLane +
                                        n1 * KPack + k2;

                shuffled(outputIndex) = src(k, n);
            }
        }
        return shuffled;
    }

    // Preshuffle scale: port from mx flatmm arch traits
    template <bool KLast, typename dtype>
    static auto preShuffleScale(ck_tile::HostTensor<dtype>& src)
    {
        auto src_lengths = src.get_lengths();
        const auto MN    = KLast ? src_lengths[0] : src_lengths[1];
        const auto K     = KLast ? src_lengths[1] : src_lengths[0];

        constexpr std::size_t MNXdlPack   = 2;
        constexpr std::size_t KXdlPack    = 2;
        constexpr std::size_t XdlMNThread = Config::N_Warp_Tile;
        constexpr std::size_t XdlKThread  = ck_tile::get_warp_size() / XdlMNThread;

        const auto MNPadded = ck_tile::integer_least_multiple(MN, XdlMNThread * MNXdlPack);
        ck_tile::HostTensor<dtype> shuffled(ck_tile::HostTensorDescriptor(
            {static_cast<std::size_t>(MNPadded * K)}, {static_cast<std::size_t>(1)}));

        const std::size_t K0 = K / KXdlPack / XdlKThread;

        for(std::size_t n = 0; n < static_cast<std::size_t>(MNPadded); ++n)
        {
            for(std::size_t k = 0; k < static_cast<std::size_t>(K); ++k)
            {
                const auto n0    = n / (XdlMNThread * MNXdlPack);
                const auto tempn = n % (XdlMNThread * MNXdlPack);
                const auto n1    = tempn % XdlMNThread;
                const auto n2    = tempn / XdlMNThread;

                const auto k0    = k / (XdlKThread * KXdlPack);
                const auto tempk = k % (XdlKThread * KXdlPack);
                const auto k1    = tempk % XdlKThread;
                const auto k2    = tempk / XdlKThread;

                const auto outputIndex = n0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread * K0 +
                                         k0 * MNXdlPack * KXdlPack * XdlMNThread * XdlKThread +
                                         k1 * MNXdlPack * KXdlPack * XdlMNThread +
                                         n1 * MNXdlPack * KXdlPack + k2 * MNXdlPack + n2;

                if constexpr(KLast)
                    shuffled(outputIndex) = n < static_cast<std::size_t>(MN) ? src(n, k) : dtype{};
                else
                    shuffled(outputIndex) = n < static_cast<std::size_t>(MN) ? src(k, n) : dtype{};
            }
        }

        return shuffled;
    }
};
