// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>

namespace hipdnn_integration_tests
{

/// A fill recipe in a case's `metadata.inputs` is malformed (InputFillRecipes::loadFromJson).
/// An authoring error; the metadata parser re-raises it as a BundleMetadataError.
class FillRecipeError : public std::runtime_error
{
public:
    using std::runtime_error::runtime_error;
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
    };

    /// The non-finite value a SPECIAL recipe plants.
    enum class SpecialValue
    {
        QUIET_NAN,
        POS_INF,
        NEG_INF,
    };

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

    friend bool operator==(const FillRecipe& a, const FillRecipe& b)
    {
        return a.kind == b.kind && a.lo == b.lo && a.hi == b.hi && a.value == b.value
               && a.scale == b.scale && a.special == b.special && a.fraction == b.fraction;
    }
    friend bool operator!=(const FillRecipe& a, const FillRecipe& b)
    {
        return !(a == b);
    }
};

} // namespace hipdnn_integration_tests
