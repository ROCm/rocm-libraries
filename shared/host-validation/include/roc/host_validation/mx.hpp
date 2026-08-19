// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <roc/host_validation/tensor.hpp>
#include <variant>

namespace roc::host_validation {
namespace detail {
struct MxDataRecipeAccess;
}

struct MxBoundedDataParameters {
    double lower = -1.0;
    double upper = 1.0;

    friend bool operator==(const MxBoundedDataParameters&,
                           const MxBoundedDataParameters&) = default;
};

struct MxAlternatingSignDataParameters {
    double maximumMagnitude = 1.0;

    friend bool operator==(const MxAlternatingSignDataParameters&,
                           const MxAlternatingSignDataParameters&) = default;
};

struct MxNormalDataParameters {
    double mean = 0.0;
    double standardDeviation = 1.0;

    friend bool operator==(const MxNormalDataParameters&, const MxNormalDataParameters&) = default;
};

struct MxUniformIntegerDataParameters {
    int lower = 0;
    int upper = 1;

    friend bool operator==(const MxUniformIntegerDataParameters&,
                           const MxUniformIntegerDataParameters&) = default;
};

// Immutable source-value recipe for packed MX data. Scale generation is an
// independent MxScaleGenerationMode policy on MxGenerationProblem.
class MxDataRecipe {
   public:
    [[nodiscard]] static MxDataRecipe bounded(MxBoundedDataParameters parameters = {});
    [[nodiscard]] static MxDataRecipe boundedAlternatingSign(
        MxAlternatingSignDataParameters parameters = {});
    [[nodiscard]] static MxDataRecipe unbounded();
    [[nodiscard]] static MxDataRecipe identity();
    [[nodiscard]] static MxDataRecipe constant(double value);
    [[nodiscard]] static MxDataRecipe sequential();
    [[nodiscard]] static MxDataRecipe rowIndex();
    [[nodiscard]] static MxDataRecipe columnIndex();
    [[nodiscard]] static MxDataRecipe checkerboard();
    [[nodiscard]] static MxDataRecipe scaledDiagonal();
    [[nodiscard]] static MxDataRecipe typeMaximum();
    [[nodiscard]] static MxDataRecipe typeDenormalMinimum();
    [[nodiscard]] static MxDataRecipe typeDenormalMaximum();
    [[nodiscard]] static MxDataRecipe typeNaN();
    [[nodiscard]] static MxDataRecipe typeInfinity();
    [[nodiscard]] static MxDataRecipe trigonometric();
    [[nodiscard]] static MxDataRecipe normal(MxNormalDataParameters parameters = {});
    [[nodiscard]] static MxDataRecipe uniformInteger(MxUniformIntegerDataParameters parameters);

    friend bool operator==(const MxDataRecipe&, const MxDataRecipe&) = default;

   private:
    enum class Kind {
        Bounded,
        BoundedAlternatingSign,
        Unbounded,
        Identity,
        Constant,
        Sequential,
        RowIndex,
        ColumnIndex,
        Checkerboard,
        ScaledDiagonal,
        TypeMaximum,
        TypeDenormalMinimum,
        TypeDenormalMaximum,
        TypeNaN,
        TypeInfinity,
        Trigonometric,
        Normal,
        UniformInteger,
    };

    using Parameters = std::variant<std::monostate, double, MxBoundedDataParameters,
                                    MxAlternatingSignDataParameters, MxNormalDataParameters,
                                    MxUniformIntegerDataParameters>;

    explicit MxDataRecipe(Kind kind, Parameters parameters = {});

    Kind kind_;
    Parameters parameters_;

    friend struct detail::MxDataRecipeAccess;
};

// Selects how block scales are produced, independently of the data recipe.
// Derived chooses a scale from each block's generated source values. Every
// other mode writes one scale-specific constant to all blocks and does not
// change the generated data values.
enum class MxScaleGenerationMode {
    Derived,
    // Smallest encoded finite scale: numerical zero where the scale format has
    // one, otherwise its minimum positive value (for example E8M0 raw zero).
    Minimum,
    One,
    Two,
    Maximum,
    NaN,
};

// Describes a rank-two, block-scaled MX tensor in its natural host layout.
// Data generation and scale selection are independent: data controls source
// values, while scale controls only how each block scale is selected.
struct MxGenerationProblem {
    MxGenerationProblem() : data(MxDataRecipe::bounded()) {}

    // Packed data encoding and per-block scale encoding.
    ScalarType dataType = ScalarType::Float4E2M1;
    ScalarType scaleType = ScalarType::E8M0;

    // Logical data shape. A zero leading dimension selects shape[0], otherwise
    // data uses strides {1, leadingDimension}.
    Shape shape;
    ptrdiff_t leadingDimension = 0;

    // Scales group blockSize consecutive elements along blockAxis. The other
    // axis is the free coordinate identifying independent block sequences.
    size_t blockAxis = 0;
    size_t blockSize = 32;

    // Source values to quantize into dataType.
    MxDataRecipe data;

    // Scale-selection policy. Derived computes scales from data; constant modes
    // use the same encoded scale for every block.
    MxScaleGenerationMode scale = MxScaleGenerationMode::Derived;

    // Seed used by stochastic data and scale choices.
    uint64_t seed = 0;
};

struct MxGenerationResult {
    // Packed data with the requested leading dimension.
    Tensor data;

    // One-dimensional scales in natural block order.
    Tensor scales;

    // UInt32 tensor mapping every logical data element to its entry in scales.
    Tensor scaleIndices;

    // Contiguous Float32 tensor equal to decoded data * selected scale.
    Tensor reference;
};

MxGenerationResult generateMx(const MxGenerationProblem& problem);
MxGenerationResult generateMx(const MxGenerationProblem& problem,
                              const TensorStorageAllocator& allocator);
}  // namespace roc::host_validation
