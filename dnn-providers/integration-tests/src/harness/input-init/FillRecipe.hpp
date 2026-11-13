// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <utility>
#include <variant>
#include <vector>

namespace hipdnn_integration_tests
{

/// A fill recipe in a case's `metadata.inputs` is malformed (parseInputFillSpec,
/// InputFillRecipes::loadFromJson). An authoring error; the metadata parser re-raises it as a
/// BundleMetadataError.
class FillRecipeError : public std::invalid_argument
{
public:
    using std::invalid_argument::invalid_argument;
};

// How one input tensor is filled. Which fields a kind reads:
//
//   FREE       lo, hi                    uniform draw over [lo, hi]
//   FIXED      value                     every element is `value`
//   NONZERO    lo, hi                    FREE, but no element is zero in the tensor's data type
//   ZEROS      (none)                    every element is zero
//   LARGE      scale                     magnitudes in [scale / 2, scale] (see fillTyped), never
//                                        above the data type's largest finite value
//   SUBNORMAL  (none)                    every element is subnormal in the tensor's data type
//   SPECIAL    special, fraction,        exactly round(fraction * elements) elements, at least
//              lo, hi                    one when fraction > 0, are the non-finite `special`;
//                                        the rest are FREE over [lo, hi]
//   VALUES     list, repeat              element i in logical index order is list[i % n]; with
//                                        repeat false, n must equal the element count
//
// Whether a data type can hold a kind is decided where the tensor is filled, not here:
// the recipe does not know the data type.
struct FillRecipe
{
    enum class Kind
    {
        FREE,
        FIXED,
        NONZERO,
        ZEROS,
        LARGE,
        SUBNORMAL,
        SPECIAL,
        VALUES,
    };

    /// The non-finite value a SPECIAL recipe plants.
    enum class SpecialValue
    {
        QUIET_NAN,
        POS_INF,
        NEG_INF,
    };

    // The numbers of a VALUES recipe. The list holds exact 64-bit integers when every
    // number is an integer, else doubles.
    using ValueList = std::variant<std::vector<int64_t>, std::vector<double>>;

    static constexpr float K_DEFAULT_LO = -1.0f;
    static constexpr float K_DEFAULT_HI = 1.0f;
    static constexpr float K_DEFAULT_SCALE = 1.0f;

    Kind kind = Kind::FREE;
    float lo = K_DEFAULT_LO;
    float hi = K_DEFAULT_HI;
    float value = 0.0f;
    float scale = K_DEFAULT_SCALE;
    SpecialValue special = SpecialValue::QUIET_NAN;
    float fraction = 0.0f;

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
    static FillRecipe nonzero(float lo, float hi)
    {
        FillRecipe f;
        f.kind = Kind::NONZERO;
        f.lo = lo;
        f.hi = hi;
        return f;
    }
    static FillRecipe zeros()
    {
        FillRecipe f;
        f.kind = Kind::ZEROS;
        return f;
    }
    static FillRecipe large(float scale)
    {
        FillRecipe f;
        f.kind = Kind::LARGE;
        f.scale = scale;
        return f;
    }
    static FillRecipe subnormal()
    {
        FillRecipe f;
        f.kind = Kind::SUBNORMAL;
        return f;
    }
    // Named for what it plants rather than `special`, which is the field's name.
    static FillRecipe nonFinite(SpecialValue special,
                                float fraction,
                                float lo = K_DEFAULT_LO,
                                float hi = K_DEFAULT_HI)
    {
        FillRecipe f;
        f.kind = Kind::SPECIAL;
        f.special = special;
        f.fraction = fraction;
        f.lo = lo;
        f.hi = hi;
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

    friend bool operator==(const FillRecipe& a, const FillRecipe& b)
    {
        return a.kind == b.kind && a.lo == b.lo && a.hi == b.hi && a.value == b.value
               && a.scale == b.scale && a.special == b.special && a.fraction == b.fraction
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
