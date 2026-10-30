// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/stl/optional.h>
#include <nanobind/stl/vector.h>

#include <cstdint>
#include <optional>
#include <roc/hostnumerics/generation.hpp>
#include <roc/hostnumerics/permutation.hpp>
#include <utility>
#include <vector>

#include "bindings.hpp"

namespace nb = nanobind;
using namespace nb::literals;

namespace roc::hostnumerics::python_bindings {
namespace {
Tensor generateTensor(ScalarType type, std::vector<size_t> dimensions,
                      const GenerationRecipe& recipe) {
    return generate(type, Shape(std::move(dimensions)), recipe);
}
}  // namespace

void registerGenerationBindings(nb::module_& module) {
    nb::enum_<IntegerRounding>(module, "IntegerRounding")
        .value("TowardZero", IntegerRounding::TowardZero)
        .value("NearestEven", IntegerRounding::NearestEven);
    nb::enum_<IntegerOverflow>(module, "IntegerOverflow")
        .value("Identity", IntegerOverflow::Identity)
        .value("Saturate", IntegerOverflow::Saturate)
        .value("ModuloWrap", IntegerOverflow::ModuloWrap);
    nb::enum_<BFloat16Rounding>(module, "BFloat16Rounding")
        .value("NearestEven", BFloat16Rounding::NearestEven)
        .value("Truncate", BFloat16Rounding::Truncate);
    nb::enum_<FloatingSourcePrecision>(module, "FloatingSourcePrecision")
        .value("Float32", FloatingSourcePrecision::Float32)
        .value("Direct", FloatingSourcePrecision::Direct);
    nb::enum_<FloatingOverflow>(module, "FloatingOverflow")
        .value("FormatDefault", FloatingOverflow::FormatDefault)
        .value("SaturateFinite", FloatingOverflow::SaturateFinite);
    nb::enum_<E8M0Underflow>(module, "E8M0Underflow")
        .value("Nearest", E8M0Underflow::Nearest)
        .value("FlushBelowFloat32Normal", E8M0Underflow::FlushBelowFloat32Normal);
    nb::class_<ScalarConversionOptions>(module, "ScalarConversionOptions")
        .def(nb::init<IntegerRounding, IntegerOverflow, BFloat16Rounding, FloatingSourcePrecision,
                      FloatingOverflow, E8M0Underflow>(),
             "integer_rounding"_a, "integer_overflow"_a,
             "bfloat16_rounding"_a = BFloat16Rounding::NearestEven,
             "floating_source_precision"_a = FloatingSourcePrecision::Float32,
             "floating_overflow"_a = FloatingOverflow::FormatDefault,
             "e8m0_underflow"_a = E8M0Underflow::Nearest)
        .def_rw("floating_source_precision", &ScalarConversionOptions::floatingSourcePrecision)
        .def_rw("floating_overflow", &ScalarConversionOptions::floatingOverflow)
        .def_rw("bfloat16_rounding", &ScalarConversionOptions::bfloat16Rounding)
        .def_rw("e8m0_underflow", &ScalarConversionOptions::e8m0Underflow)
        .def_rw("integer_rounding", &ScalarConversionOptions::integerRounding)
        .def_rw("integer_overflow", &ScalarConversionOptions::integerOverflow);
    nb::class_<BinomialHalfGenerationParameters>(module, "BinomialHalfGenerationParameters")
        .def(nb::init<uint32_t>(), "trials"_a = 1)
        .def_rw("trials", &BinomialHalfGenerationParameters::trials);
    nb::class_<GenerationRecipeSettings>(
        module, "GenerationRecipeSettings",
        "Seed and logical index order shared by one immutable recipe.")
        .def(nb::init<uint64_t, Permutation>(), "seed"_a, "index_order"_a)
        .def_rw("seed", &GenerationRecipeSettings::seed)
        .def_rw("index_order", &GenerationRecipeSettings::indexOrder)
        .def_rw("index_offset", &GenerationRecipeSettings::indexOffset);

    nb::class_<ConstantGenerationParameters>(module, "ConstantGenerationParameters",
                                             "One numerical value for constant generation.")
        .def(nb::init<double>(), "value"_a)
        .def_rw("value", &ConstantGenerationParameters::value);

    nb::class_<ChoiceGenerationParameters>(
        module, "ChoiceGenerationParameters",
        "Candidate values; repeated entries increase selection frequency.")
        .def(nb::init<std::vector<double>>(), "values"_a)
        .def_rw("values", &ChoiceGenerationParameters::values);

    nb::class_<UniformIntegerGenerationParameters>(module, "UniformIntegerGenerationParameters",
                                                   "Inclusive integer generation bounds.")
        .def(nb::init<int, int>(), "lower"_a, "upper"_a)
        .def_rw("lower", &UniformIntegerGenerationParameters::lower)
        .def_rw("upper", &UniformIntegerGenerationParameters::upper);

    nb::class_<UniformRealGenerationParameters>(module, "UniformRealGenerationParameters",
                                                "Numerical bounds for uniform real generation.")
        .def(nb::init<double, double>(), "lower"_a, "upper"_a)
        .def_rw("lower", &UniformRealGenerationParameters::lower)
        .def_rw("upper", &UniformRealGenerationParameters::upper);

    nb::class_<NormalGenerationParameters>(module, "NormalGenerationParameters",
                                           "Mean and standard deviation for normal generation.")
        .def(nb::init<double, double>(), "mean"_a, "standard_deviation"_a)
        .def_rw("mean", &NormalGenerationParameters::mean)
        .def_rw("standard_deviation", &NormalGenerationParameters::standardDeviation);

    nb::class_<DimensionGenerationParameters>(module, "DimensionGenerationParameters",
                                              "Zero-based tensor dimension.")
        .def(nb::init<size_t>(), "dimension"_a)
        .def_rw("dimension", &DimensionGenerationParameters::dimension);

    nb::class_<AffineIndexRemainderGenerationParameters>(
        module, "AffineIndexRemainderGenerationParameters",
        "Signed affine coordinate expression and positive remainder divisor.")
        .def(nb::init<std::vector<int64_t>, int64_t, int64_t>(), "dimension_coefficients"_a,
             "offset"_a, "positive_divisor"_a)
        .def_rw("dimension_coefficients",
                &AffineIndexRemainderGenerationParameters::dimensionCoefficients)
        .def_rw("offset", &AffineIndexRemainderGenerationParameters::offset)
        .def_rw("positive_divisor", &AffineIndexRemainderGenerationParameters::positiveDivisor);

    nb::class_<RandomEncodedExponentGenerationParameters>(
        module, "RandomEncodedExponentGenerationParameters",
        "Inclusive unbiased exponent bounds and optional source encoding.")
        .def(nb::init<int, int, std::optional<ScalarType>>(), "lower_unbiased_exponent"_a,
             "upper_unbiased_exponent"_a, "source_type"_a)
        .def_rw("lower_unbiased_exponent",
                &RandomEncodedExponentGenerationParameters::lowerUnbiasedExponent)
        .def_rw("upper_unbiased_exponent",
                &RandomEncodedExponentGenerationParameters::upperUnbiasedExponent)
        .def_rw("source_type", &RandomEncodedExponentGenerationParameters::sourceType);

    nb::class_<RawConstantGenerationParameters>(module, "RawConstantGenerationParameters",
                                                "Low destination storage bits to write.")
        .def(nb::init<uint64_t>(), "bits"_a)
        .def_rw("bits", &RawConstantGenerationParameters::bits);

    nb::class_<GenerationAffineValueParameters>(
        module, "GenerationAffineValueParameters",
        "Numerical postprocessing as generated_value * scale + offset.")
        .def(nb::init<double, double>(), "scale"_a, "offset"_a)
        .def_rw("scale", &GenerationAffineValueParameters::scale)
        .def_rw("offset", &GenerationAffineValueParameters::offset);

    nb::class_<AlternatingSignGenerationParameters>(
        module, "AlternatingSignGenerationParameters",
        "Dimensions whose coordinate parity controls the generated sign.")
        .def(nb::init<std::vector<size_t>, bool>(), "dimensions"_a, "negative_when_odd"_a)
        .def_rw("dimensions", &AlternatingSignGenerationParameters::dimensions)
        .def_rw("negative_when_odd", &AlternatingSignGenerationParameters::negativeWhenOdd);

    auto recipe = nb::class_<GenerationRecipe>(
        module, "GenerationRecipe",
        "Immutable typed recipe. Construct it with real_only, replicated, or cartesian.");

    nb::class_<GenerationRecipe::Component>(
        recipe, "Component", "Immutable scalar generator; modifiers return a new component.")
        .def("with_absolute_transform", &GenerationRecipe::Component::withAbsoluteTransform)
        .def("with_sine_transform", &GenerationRecipe::Component::withSineTransform)
        .def("with_cosine_transform", &GenerationRecipe::Component::withCosineTransform)
        .def("with_exp2_transform", &GenerationRecipe::Component::withExp2Transform)
        .def("with_affine_value_mapping", &GenerationRecipe::Component::withAffineValueMapping,
             "parameters"_a)
        .def("with_alternating_sign", &GenerationRecipe::Component::withAlternatingSign,
             "parameters"_a)
        .def("with_zero_outside_main_diagonal",
             &GenerationRecipe::Component::withZeroOutsideMainDiagonal);

    recipe.def_static("zero", &GenerationRecipe::zero)
        .def_static("constant", &GenerationRecipe::constant, "parameters"_a)
        .def_static("choice", &GenerationRecipe::choice, "parameters"_a)
        .def_static("uniform_integer", &GenerationRecipe::uniformInteger, "parameters"_a)
        .def_static("absolute_uniform_integer", &GenerationRecipe::absoluteUniformInteger,
                    "parameters"_a)
        .def_static("uniform_real", &GenerationRecipe::uniformReal, "parameters"_a)
        .def_static("normal", &GenerationRecipe::normal, "parameters"_a)
        .def_static("binomial_half", &GenerationRecipe::binomialHalf, "parameters"_a)
        .def_static("sine", &GenerationRecipe::sine)
        .def_static("cosine", &GenerationRecipe::cosine)
        .def_static("absolute_sine", &GenerationRecipe::absoluteSine)
        .def_static("absolute_cosine", &GenerationRecipe::absoluteCosine)
        .def_static("serial_index", &GenerationRecipe::serialIndex)
        .def_static("serial_dimension", &GenerationRecipe::serialDimension, "parameters"_a)
        .def_static("affine_index_remainder", &GenerationRecipe::affineIndexRemainder,
                    "parameters"_a)
        .def_static("identity", &GenerationRecipe::identity)
        .def_static("checkerboard_uniform_integer", &GenerationRecipe::checkerboardUniformInteger,
                    "parameters"_a)
        .def_static("type_maximum", &GenerationRecipe::typeMaximum)
        .def_static("type_lowest", &GenerationRecipe::typeLowest)
        .def_static("type_denormal_minimum", &GenerationRecipe::typeDenormalMinimum)
        .def_static("type_denormal_maximum", &GenerationRecipe::typeDenormalMaximum)
        .def_static("type_nan", &GenerationRecipe::typeNaN)
        .def_static("type_infinity", &GenerationRecipe::typeInfinity)
        .def_static("type_negative_infinity", &GenerationRecipe::typeNegativeInfinity)
        .def_static("type_negative_zero", &GenerationRecipe::typeNegativeZero)
        .def_static("uniform_type_range", &GenerationRecipe::uniformTypeRange)
        .def_static("random_encoded_exponent", &GenerationRecipe::randomEncodedExponent,
                    "parameters"_a)
        .def_static("raw_constant", &GenerationRecipe::rawConstant, "parameters"_a)
        .def_static("uniform_raw_integer", &GenerationRecipe::uniformRawInteger, "parameters"_a)
        .def_static("uniform_finite_encoded_value", &GenerationRecipe::uniformFiniteEncodedValue)
        .def_static("random_raw_bits", &GenerationRecipe::randomRawBits)
        .def_static("raw_serial_dimension", &GenerationRecipe::rawSerialDimension, "parameters"_a)
        .def_static("real_only", &GenerationRecipe::realOnly, "component"_a, "settings"_a)
        .def_static("replicated", &GenerationRecipe::replicated, "component"_a, "settings"_a)
        .def_static("cartesian", &GenerationRecipe::cartesian, "real"_a, "imaginary"_a,
                    "settings"_a)
        .def_prop_ro("seed", &GenerationRecipe::seed)
        .def_prop_ro("index_offset", &GenerationRecipe::indexOffset)
        .def("with_index_offset", &GenerationRecipe::withIndexOffset, "offset"_a)
        .def("index_order", &GenerationRecipe::indexOrder, "rank"_a)
        .def_static("real_only_last_dimension_fastest",
                    &GenerationRecipe::realOnlyLastDimensionFastest, "component"_a, "seed"_a)
        .def_static("replicated_last_dimension_fastest",
                    &GenerationRecipe::replicatedLastDimensionFastest, "component"_a, "seed"_a)
        .def_static("cartesian_last_dimension_fastest",
                    &GenerationRecipe::cartesianLastDimensionFastest, "real"_a, "imaginary"_a,
                    "seed"_a)
        .def("with_last_dimension_fastest_index_order",
             &GenerationRecipe::withLastDimensionFastestIndexOrder)
        .def("with_seed", &GenerationRecipe::withSeed, "seed"_a)
        .def(
            "with_index_order",
            [](const GenerationRecipe& recipe, const Permutation& order) {
                return recipe.withIndexOrder(order);
            },
            "index_order"_a);

    module.def("generate_tensor",
               static_cast<Tensor (*)(ScalarType, Shape, const GenerationRecipe&)>(&generate),
               "type"_a, "shape"_a, "recipe"_a,
               "Allocate and fill a contiguous tensor from a typed recipe.");
    module.def("generate_tensor",
               static_cast<Tensor (*)(ScalarType, Layout, const GenerationRecipe&)>(&generate),
               "type"_a, "layout"_a, "recipe"_a,
               "Allocate and fill a tensor with the requested layout.");
    module.def("generate_tensor",
               static_cast<Tensor (*)(ScalarType, std::vector<size_t>, const GenerationRecipe&)>(
                   &generateTensor),
               "type"_a, "shape"_a, "recipe"_a,
               "Allocate and fill a contiguous tensor from Python dimensions.");
    module.def(
        "generate_at",
        [](Tensor& tensor, size_t logicalIndex, const GenerationRecipe& recipe) -> Tensor& {
            generateAt(tensor, logicalIndex, recipe);
            return tensor;
        },
        "tensor"_a, "logical_index"_a, "recipe"_a, nb::rv_policy::reference,
        "Fill one logical tensor element from a typed recipe.");
    module.def(
        "generate_tensor",
        [](ScalarType type, std::vector<size_t> shape, const GenerationRecipe& recipe,
           const ScalarConversionOptions& options) {
            return generate(type, Shape(std::move(shape)), recipe, options);
        },
        "type"_a, "shape"_a, "recipe"_a, "options"_a);
    module.def("generate_tensor",
               static_cast<Tensor (*)(ScalarType, Layout, const GenerationRecipe&,
                                      const ScalarConversionOptions&)>(&generate),
               "type"_a, "layout"_a, "recipe"_a, "options"_a);
    module.def(
        "generate_tensor",
        [](Tensor& tensor, const GenerationRecipe& recipe,
           const ScalarConversionOptions& options) -> Tensor& {
            generate(tensor, recipe, options);
            return tensor;
        },
        "tensor"_a, "recipe"_a, "options"_a, nb::rv_policy::reference);
    module.def(
        "generate_at",
        [](Tensor& tensor, size_t index, const GenerationRecipe& recipe,
           const ScalarConversionOptions& options) -> Tensor& {
            generateAt(tensor, index, recipe, options);
            return tensor;
        },
        "tensor"_a, "logical_index"_a, "recipe"_a, "options"_a, nb::rv_policy::reference);
}
}  // namespace roc::hostnumerics::python_bindings
