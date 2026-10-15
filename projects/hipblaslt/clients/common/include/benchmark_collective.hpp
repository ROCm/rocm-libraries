// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>

namespace hipblaslt_bench
{
    struct MaxOp
    {
        template <typename T>
        T operator()(T a, T b) const
        {
            return std::max(a, b);
        }
    };

    struct MinOp
    {
        template <typename T>
        T operator()(T a, T b) const
        {
            return std::min(a, b);
        }
    };

    // Reduces a per-rank quantity to the one value the whole group sees. A
    // default-constructed instance is the single-rank case: agree returns its
    // argument.
    struct CollectiveAgreement
    {
        std::function<bool(const void*, void*, size_t)> allgather;
        uint32_t                                        world = 1;

        template <typename T, typename Op>
        T agree(T mine, Op reduce) const
        {
            if(!allgather)
                return mine;

            const std::unique_ptr<T[]> all = std::make_unique<T[]>(world);
            if(!allgather(&mine, all.get(), sizeof(T)))
                throw std::runtime_error("collective allgather failed");

            T acc = all[0];
            for(uint32_t j = 1; j < world; ++j)
                acc = reduce(acc, all[j]);
            return acc;
        }
    };
} // namespace hipblaslt_bench
