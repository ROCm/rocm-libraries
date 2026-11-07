// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <nanobind/nanobind.h>
#include <nanobind/stl/complex.h>

#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <roc/hostnumerics/scalar_storage.hpp>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace nb = nanobind;
using namespace nb::literals;
using namespace roc::hostnumerics;

namespace {
// The C++ type that decoding returns to Python for each scalar category. Each
// represents every value of its category exactly.
template <typename Tag>
constexpr auto pythonValueIdentity() {
    constexpr ScalarCategory category = scalarTypeInfo(Tag::type).category;
    if constexpr (category == ScalarCategory::Boolean)
        return std::type_identity<bool>{};
    else if constexpr (category == ScalarCategory::SignedInteger)
        return std::type_identity<int64_t>{};
    else if constexpr (category == ScalarCategory::UnsignedInteger)
        return std::type_identity<uint64_t>{};
    else if constexpr (category == ScalarCategory::FloatingPoint)
        return std::type_identity<double>{};
    else
        return std::type_identity<std::complex<double>>{};
}

template <typename Tag>
using PythonValue = typename decltype(pythonValueIdentity<Tag>())::type;

template <typename Tag>
constexpr const char* pythonTypeName() {
    using Value = PythonValue<Tag>;
    if constexpr (std::is_same_v<Value, bool>)
        return "bool";
    else if constexpr (std::is_integral_v<Value>)
        return "int";
    else if constexpr (std::is_same_v<Value, double>)
        return "float";
    else
        return "complex";
}

// Encodes a Python int as an integer type. Identity overflow requires the value
// to fit, which C++ only asserts, so check it here.
template <typename Tag, typename Integer>
void encodeInteger(std::span<std::byte> storage, size_t index, Integer value,
                   const ScalarConversionOptions& options) {
    constexpr bool isInt4 = Tag::type == ScalarType::Int4;
    using Range = std::conditional_t<isInt4, int8_t, typename Tag::Storage>;
    const bool fits = isInt4 ? std::cmp_greater_equal(value, -8) && std::cmp_less_equal(value, 7)
                             : std::in_range<Range>(value);
    if (options.integerOverflow == IntegerOverflow::Identity && !fits)
        throw nb::value_error(("Element " + std::to_string(index) + " is outside the range of " +
                               std::string(scalarTypeName(Tag::type)) +
                               "; IntegerOverflow.Identity requires values that fit.")
                                  .c_str());
    encodeScalar(Tag::type, storage, index, value, options);
}

nb::bytes encodeScalars(ScalarType type, const nb::sequence& values,
                        const ScalarConversionOptions& options) {
    const size_t count = nb::len(values);
    return visitScalarType(type, [&]<typename Tag>() {
        std::vector<std::byte> storage((count * Tag::storageBits + 7) / 8);
        for (size_t index = 0; index < count; ++index) {
            const nb::object item = values[index];
            if constexpr (std::is_integral_v<PythonValue<Tag>> &&
                          !std::is_same_v<PythonValue<Tag>, bool>) {
                int64_t signedValue = 0;
                uint64_t unsignedValue = 0;
                if (nb::isinstance<nb::int_>(item) && nb::try_cast(item, signedValue)) {
                    encodeInteger<Tag>(storage, index, signedValue, options);
                    continue;
                }
                if (nb::isinstance<nb::int_>(item) && nb::try_cast(item, unsignedValue)) {
                    encodeInteger<Tag>(storage, index, unsignedValue, options);
                    continue;
                }
            } else {
                PythonValue<Tag> value{};
                if (nb::try_cast(item, value)) {
                    // C++ only asserts this precondition, so check it here.
                    constexpr bool isUnsignedScale = Tag::type == ScalarType::E4M3 ||
                                                     Tag::type == ScalarType::E5M3 ||
                                                     Tag::type == ScalarType::E8M0;
                    if constexpr (isUnsignedScale)
                        if (value < 0)
                            throw nb::value_error(
                                ("Element " + std::to_string(index) + " is negative, which " +
                                 std::string(scalarTypeName(type)) + " cannot encode.")
                                    .c_str());
                    encodeScalar(type, storage, index, value, options);
                    continue;
                }
            }
            throw nb::type_error(("Element " + std::to_string(index) + " is not a " +
                                  pythonTypeName<Tag>() + " value for " +
                                  std::string(scalarTypeName(type)) + ".")
                                     .c_str());
        }
        return nb::bytes(storage.data(), storage.size());
    });
}

nb::list decodeScalars(ScalarType type, const nb::bytes& storage, size_t count) {
    const std::span<const std::byte> bytes(reinterpret_cast<const std::byte*>(storage.c_str()),
                                           storage.size());
    // Python values represent every decoded value exactly, so no option applies.
    const ScalarConversionOptions exact{IntegerRounding::TowardZero, IntegerOverflow::Identity};
    return visitScalarType(type, [&]<typename Tag>() {
        nb::list result;
        for (size_t index = 0; index < count; ++index)
            result.append(decodeScalar<PythonValue<Tag>>(type, bytes, index, exact));
        return result;
    });
}
}  // namespace

namespace roc::hostnumerics::python_bindings {
void registerScalarStorageBindings(nb::module_& module) {
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

    nb::class_<ScalarConversionOptions>(module, "ScalarConversionOptions", R"doc(
Choices for converting values to and from encoded scalars, as in the C++
ScalarConversionOptions. The integer rounding and overflow are required; the
floating choices are keyword-only and default to the C++ defaults.
)doc")
        .def(nb::init<IntegerRounding, IntegerOverflow, BFloat16Rounding, FloatingSourcePrecision,
                      FloatingOverflow, E8M0Underflow>(),
             "integer_rounding"_a, "integer_overflow"_a, nb::kw_only(),
             "bfloat16_rounding"_a = BFloat16Rounding::NearestEven,
             "floating_source_precision"_a = FloatingSourcePrecision::Float32,
             "floating_overflow"_a = FloatingOverflow::FormatDefault,
             "e8m0_underflow"_a = E8M0Underflow::Nearest)
        .def_rw("integer_rounding", &ScalarConversionOptions::integerRounding)
        .def_rw("integer_overflow", &ScalarConversionOptions::integerOverflow)
        .def_rw("bfloat16_rounding", &ScalarConversionOptions::bfloat16Rounding)
        .def_rw("floating_source_precision", &ScalarConversionOptions::floatingSourcePrecision)
        .def_rw("floating_overflow", &ScalarConversionOptions::floatingOverflow)
        .def_rw("e8m0_underflow", &ScalarConversionOptions::e8m0Underflow);

    module.def("encode_scalars", &encodeScalars, "scalar_type"_a, "values"_a, "options"_a,
               R"doc(
Encode values as consecutive elements of scalar_type and return the bytes.

Values are bool for Boolean, int for integer types, float for floating types
and complex for complex types. Elements are laid out as in the C++
encodeScalar; unused bits of the last byte are zero. Raises TypeError for a
value of another type, and ValueError for a negative value as E4M3, E5M3 or
E8M0 or for an int outside an integer type's range when
options.integer_overflow is IntegerOverflow.Identity.

Example:
    >>> options = ScalarConversionOptions(IntegerRounding.NearestEven,
    ...                                   IntegerOverflow.Saturate)
    >>> encode_scalars(ScalarType.Float4E2M1, [0.5, 3.0, -1.5], options)
    b'Q\x0b'
)doc");
    module.def("decode_scalars", &decodeScalars, "scalar_type"_a, "storage"_a, "count"_a,
               R"doc(
Decode the first count elements of scalar_type from storage into a list.

Values are returned exactly, with the Python types that encode_scalars
accepts. Raises RuntimeError if storage holds fewer than count elements.
)doc");
}
}  // namespace roc::hostnumerics::python_bindings
