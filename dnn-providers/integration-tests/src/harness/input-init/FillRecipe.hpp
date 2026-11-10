// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace hipdnn_integration_tests
{

struct FillRecipe
{
    enum class Kind
    {
        FREE,
        FIXED,
        VALUES,
    };

    // The numbers of a VALUES recipe. The list holds exact 64-bit integers when every
    // number is an integer, else doubles.
    using ValueList = std::variant<std::vector<int64_t>, std::vector<double>>;

    static constexpr float K_DEFAULT_LO = -1.0f;
    static constexpr float K_DEFAULT_HI = 1.0f;

    Kind kind = Kind::FREE;
    float lo = K_DEFAULT_LO;
    float hi = K_DEFAULT_HI;
    float value = 0.0f;

    // VALUES only. Element i in logical index order gets list[i % n]. With repeat
    // false, n must equal the element count. The list is immutable, so copies share it.
    std::shared_ptr<const ValueList> list;
    bool repeat = true;

    static FillRecipe free(float lo, float hi)
    {
        FillRecipe f;
        f.kind = Kind::FREE;
        f.lo = lo;
        f.hi = hi;
        return f;
    }
    static FillRecipe fixed(float v)
    {
        FillRecipe f;
        f.kind = Kind::FIXED;
        f.value = v;
        return f;
    }
    static FillRecipe values(std::vector<int64_t> numbers, bool repeat = true)
    {
        return valuesOf(ValueList(std::move(numbers)), repeat);
    }
    static FillRecipe values(std::vector<double> numbers, bool repeat = true)
    {
        return valuesOf(ValueList(std::move(numbers)), repeat);
    }

    /// Largest magnitude a draw from this recipe can have.
    float maxMagnitude() const
    {
        if(kind == Kind::FIXED)
        {
            return std::fabs(value);
        }
        if(kind == Kind::VALUES && list != nullptr)
        {
            return std::visit(
                [](const auto& numbers) {
                    float magnitude = 0.0f;
                    for(const auto number : numbers)
                    {
                        magnitude = std::max(magnitude, std::fabs(static_cast<float>(number)));
                    }
                    return magnitude;
                },
                *list);
        }
        return std::max(std::fabs(lo), std::fabs(hi));
    }

    friend bool operator==(const FillRecipe& a, const FillRecipe& b)
    {
        return a.kind == b.kind && a.lo == b.lo && a.hi == b.hi && a.value == b.value
               && a.repeat == b.repeat && sameList(a.list, b.list);
    }
    friend bool operator!=(const FillRecipe& a, const FillRecipe& b)
    {
        return !(a == b);
    }

private:
    static FillRecipe valuesOf(ValueList numbers, bool repeat)
    {
        const auto count = std::visit([](const auto& list) { return list.size(); }, numbers);
        if(count == 0)
        {
            throw std::invalid_argument("FillRecipe values needs a non-empty list");
        }

        FillRecipe f;
        f.kind = Kind::VALUES;
        f.list = std::make_shared<const ValueList>(std::move(numbers));
        f.repeat = repeat;
        return f;
    }

    static bool sameList(const std::shared_ptr<const ValueList>& a,
                         const std::shared_ptr<const ValueList>& b)
    {
        if(a == nullptr || b == nullptr)
        {
            return a == b;
        }
        return *a == *b;
    }
};

} // namespace hipdnn_integration_tests
