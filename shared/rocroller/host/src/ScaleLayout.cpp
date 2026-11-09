// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <rocRoller/hostnumerics/ScaleLayout.hpp>

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>

namespace rocRoller::hostnumerics
{
    namespace
    {
        constexpr size_t lanesPerSimd = 16;
        constexpr size_t simdsPerWave = 4;

        // A swizzle tile split into the factors that the kernel's VGPRs, SIMDs and
        // lanes index, in source order along reduction then non-reduction.
        struct SwizzleFactors
        {
            size_t vgprIndex      = 0;
            size_t vgprBlock      = 0;
            size_t simdBlock      = 0;
            size_t simdIndexIndex = 0;
            size_t simdIndexBlock = 0;
        };

        SwizzleFactors GetSwizzleFactors(const ScaleSwizzle& swizzle)
        {
            if(swizzle.tileNonReduction != 64 && swizzle.tileNonReduction != 32)
                throw std::runtime_error("Scale swizzle tileNonReduction must be 32 or 64.");
            if(swizzle.tileReduction == 0 || swizzle.tileReduction % 4 != 0)
                throw std::runtime_error(
                    "Scale swizzle tileReduction must be a nonzero multiple of 4.");
            if(swizzle.subTileReduction == 0)
                throw std::runtime_error("Scale swizzle subTileReduction must be nonzero.");
            if(swizzle.tileNonReduction == 32 && swizzle.subTileReduction != 2
               && swizzle.subTileReduction != 4)
                throw std::runtime_error(
                    "Scale swizzle with tileNonReduction 32 requires subTileReduction 2 or 4.");

            const size_t   simdIndex = swizzle.tileNonReduction / lanesPerSimd;
            SwizzleFactors factors;
            factors.simdBlock      = simdsPerWave / simdIndex;
            factors.vgprIndex      = std::min(simdIndex, swizzle.subTileReduction);
            factors.vgprBlock      = swizzle.tileReduction / factors.simdBlock / factors.vgprIndex;
            factors.simdIndexBlock = factors.vgprIndex;
            factors.simdIndexIndex = simdIndex / factors.simdIndexBlock;
            if(factors.vgprIndex * factors.vgprBlock * factors.simdBlock != swizzle.tileReduction)
                throw std::runtime_error(
                    "Scale swizzle tileReduction does not divide into its VGPR blocks.");
            if(lanesPerSimd * factors.simdIndexIndex * factors.simdIndexBlock
               != swizzle.tileNonReduction)
                throw std::runtime_error(
                    "Scale swizzle tileNonReduction does not divide into its SIMD lanes.");
            return factors;
        }

        void RequireDivisible(size_t extent, size_t tile, const char* message)
        {
            if(extent % tile != 0)
                throw std::runtime_error(message);
        }

    }

    roc::hostnumerics::ReshapePermutation
        ScaleStoragePermutation(const roc::hostnumerics::Shape&                natural,
                                const std::optional<ScaleSwizzle>&             swizzle,
                                const std::optional<roc::hostnumerics::Shape>& tile)
    {
        using roc::hostnumerics::Permutation;
        using roc::hostnumerics::ReshapePermutation;
        using roc::hostnumerics::Shape;
        if(natural.rank() != 2 || (tile && tile->rank() != 2))
            throw std::runtime_error("rocRoller scale layouts require rank-two shapes.");
        if(tile && ((*tile)[0] == 0 || (*tile)[1] == 0))
            throw std::runtime_error("Scale tile extents must be nonzero.");
        if(!swizzle)
            return tile ? ReshapePermutation::tiled(natural, *tile)
                        : ReshapePermutation::identity(natural);
        const SwizzleFactors factors = GetSwizzleFactors(*swizzle);
        if(!tile)
        {
            RequireDivisible(natural[1],
                             swizzle->tileReduction,
                             "Scale reduction extent must be divisible by the swizzle tile.");
            RequireDivisible(natural[0],
                             swizzle->tileNonReduction,
                             "Scale non-reduction extent must be divisible by the swizzle tile.");
            const Shape         factorsShape{natural[0] / swizzle->tileNonReduction,
                                     factors.simdIndexBlock,
                                     factors.simdIndexIndex,
                                     lanesPerSimd,
                                     natural[1] / swizzle->tileReduction,
                                     factors.simdBlock,
                                     factors.vgprBlock,
                                     factors.vgprIndex};
            std::vector<size_t> order;
            if(swizzle->tileNonReduction == 64)
                order = {0, 7, 2, 3, 4, 5, 6, 1};
            else if(swizzle->subTileReduction == 4)
                order = {0, 7, 2, 3, 4, 6, 5, 1};
            else
                order = {0, 1, 2, 3, 4, 7, 5, 6};
            return {natural, natural, factorsShape, Permutation(std::move(order))};
        }

        RequireDivisible((*tile)[1],
                         swizzle->tileReduction,
                         "Scale tile reduction extent must be divisible by the swizzle tile.");
        RequireDivisible((*tile)[0],
                         swizzle->tileNonReduction,
                         "Scale tile non-reduction extent must be divisible by the swizzle tile.");
        RequireDivisible(
            natural[1], (*tile)[1], "Scale reduction extent must be divisible by the tile.");
        RequireDivisible(
            natural[0], (*tile)[0], "Scale non-reduction extent must be divisible by the tile.");
        const Shape         factorsShape{natural[0] / (*tile)[0],
                                 (*tile)[0] / swizzle->tileNonReduction,
                                 factors.simdIndexBlock,
                                 factors.simdIndexIndex,
                                 lanesPerSimd,
                                 natural[1] / (*tile)[1],
                                 (*tile)[1] / swizzle->tileReduction,
                                 factors.simdBlock,
                                 factors.vgprBlock,
                                 factors.vgprIndex};
        std::vector<size_t> order;
        if(swizzle->tileNonReduction == 64)
            order = {0, 5, 1, 9, 3, 4, 6, 7, 8, 2};
        else if(swizzle->subTileReduction == 4)
            order = {0, 5, 1, 9, 3, 4, 6, 8, 7, 2};
        else
            order = {0, 5, 1, 2, 3, 4, 6, 9, 7, 8};
        return {natural, natural, factorsShape, Permutation(std::move(order))};
    }
}
