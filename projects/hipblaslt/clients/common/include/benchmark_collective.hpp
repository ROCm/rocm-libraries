// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>

namespace hipblaslt_bench
{
    enum class AgreeOp
    {
        Max,
        Min,
        All,
        Any
    };

    // Reduces a per-rank quantity to one every rank in the group sees. A
    // default-constructed instance is the single-rank case: both callbacks empty.
    struct CollectiveAgreement
    {
        std::function<double(double, AgreeOp)> value;
        std::function<bool(bool, AgreeOp)>     flag;
    };

    inline double
        agree_value(const CollectiveAgreement& agreement, double mine, AgreeOp op)
    {
        return agreement.value ? agreement.value(mine, op) : mine;
    }

    inline bool agree_flag(const CollectiveAgreement& agreement, bool mine, AgreeOp op)
    {
        return agreement.flag ? agreement.flag(mine, op) : mine;
    }
} // namespace hipblaslt_bench
