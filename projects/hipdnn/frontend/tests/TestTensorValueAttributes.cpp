// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <optional>
#include <variant>

using hipdnn_data_sdk::types::bfloat16;
using hipdnn_data_sdk::types::half;
using hipdnn_frontend::graph::Graph;
using hipdnn_frontend::graph::ScalarType;
using hipdnn_frontend::graph::TensorAttributes;

namespace
{
constexpr float PI_FLOAT = 3.14159265358979323846F;

// Assert the full RFC-0016 §4.2 getter matrix for a float-valued tensor.
// The two non-template variant getters must mirror the templated optionals:
// a value is visible through get_pass_by_value() iff it is visible through
// get_pass_by_value<T>(), and likewise for the compile-time-constant getter.
void expectFloatState(const TensorAttributes& tensor,
                      const bool isPassByValue,
                      const std::optional<float> passByValue,
                      const std::optional<float> compileTimeConstant,
                      const bool hasCompileTimeConstant,
                      const bool isRuntimePassByValue)
{
    EXPECT_EQ(tensor.get_is_pass_by_value(), isPassByValue);
    EXPECT_EQ(tensor.get_is_runtime_pass_by_value(), isRuntimePassByValue);
    EXPECT_EQ(tensor.get_has_compile_time_constant(), hasCompileTimeConstant);

    const std::optional<float> pbv = tensor.get_pass_by_value<float>();
    EXPECT_EQ(pbv.has_value(), passByValue.has_value());
    if(passByValue.has_value() && pbv.has_value())
    {
        EXPECT_FLOAT_EQ(pbv.value(), passByValue.value());
    }

    const std::optional<float> ctc = tensor.get_compile_time_constant<float>();
    EXPECT_EQ(ctc.has_value(), compileTimeConstant.has_value());
    if(compileTimeConstant.has_value() && ctc.has_value())
    {
        EXPECT_FLOAT_EQ(ctc.value(), compileTimeConstant.value());
    }

    const TensorAttributes::ValueVariant pbvVar = tensor.get_pass_by_value();
    EXPECT_EQ(std::holds_alternative<std::monostate>(pbvVar), !passByValue.has_value());

    const TensorAttributes::ValueVariant ctcVar = tensor.get_compile_time_constant();
    EXPECT_EQ(std::holds_alternative<std::monostate>(ctcVar), !compileTimeConstant.has_value());
}

// Round-trip a runtime-with-default scalar of an arbitrary supported type.
template <typename T>
void expectRuntimeRoundTrip(const T value)
{
    const TensorAttributes tensor(value, ScalarType::RUNTIME_PARAM);
    EXPECT_TRUE(tensor.get_is_pass_by_value());
    EXPECT_TRUE(tensor.get_is_runtime_pass_by_value());
    EXPECT_FALSE(tensor.get_has_compile_time_constant());

    const std::optional<T> opt = tensor.get_pass_by_value<T>();
    ASSERT_TRUE(opt.has_value());
    EXPECT_EQ(opt.value(), value);

    // A runtime tensor never answers the compile-time getter, even for the right type.
    EXPECT_FALSE(tensor.get_compile_time_constant<T>().has_value());
}
} // namespace

// --- Runtime-with-default default paths (plain ctor / set_value; flag true) ---
// Per RFC-0016 §4.3, the plain constructor and set_value bake a default AND set
// the runtime flag (cuDNN parity), so they are runtime-with-default, not
// compile-time constants.

TEST(TestTensorValueAttributes, PlainConstructorIsRuntimeWithDefault)
{
    const TensorAttributes tensor(PI_FLOAT);
    expectFloatState(tensor,
                     /*isPassByValue*/ true,
                     /*passByValue*/ PI_FLOAT,
                     /*compileTimeConstant*/ std::nullopt,
                     /*hasCompileTimeConstant*/ false,
                     /*isRuntimePassByValue*/ true);
}

TEST(TestTensorValueAttributes, SetValueIsRuntimeWithDefault)
{
    TensorAttributes tensor;
    tensor.set_value(PI_FLOAT);
    expectFloatState(tensor, true, PI_FLOAT, std::nullopt, false, true);
}

// --- Compile-time-constant creation paths (flag false, value present) --------

TEST(TestTensorValueAttributes, SetCompileTimeConstantIsCompileTimeConstant)
{
    TensorAttributes tensor;
    tensor.set_compile_time_constant(PI_FLOAT);
    expectFloatState(tensor, true, std::nullopt, PI_FLOAT, true, false);
}

TEST(TestTensorValueAttributes, ConstructorCompileTimeConstMode)
{
    const TensorAttributes tensor(PI_FLOAT, ScalarType::COMPILE_TIME_CONST);
    expectFloatState(tensor, true, std::nullopt, PI_FLOAT, true, false);
}

TEST(TestTensorValueAttributes, GraphTensorCompileTimeConstMode)
{
    const std::shared_ptr<TensorAttributes> tensor
        = Graph::tensor(PI_FLOAT, ScalarType::COMPILE_TIME_CONST);
    ASSERT_NE(tensor, nullptr);
    expectFloatState(*tensor, true, std::nullopt, PI_FLOAT, true, false);
}

// --- Runtime-with-default creation paths (flag true, value present) ----------

TEST(TestTensorValueAttributes, ConstructorRuntimeParamMode)
{
    const TensorAttributes tensor(PI_FLOAT, ScalarType::RUNTIME_PARAM);
    expectFloatState(tensor, true, PI_FLOAT, std::nullopt, false, true);
}

TEST(TestTensorValueAttributes, GraphTensorRuntimeParamMode)
{
    const std::shared_ptr<TensorAttributes> tensor
        = Graph::tensor(PI_FLOAT, ScalarType::RUNTIME_PARAM);
    ASSERT_NE(tensor, nullptr);
    expectFloatState(*tensor, true, PI_FLOAT, std::nullopt, false, true);
}

TEST(TestTensorValueAttributes, SetIsPassByValueAfterSetValueKeepsValue)
{
    TensorAttributes tensor;
    tensor.set_value(PI_FLOAT);
    tensor.set_is_pass_by_value(true);
    // set_is_pass_by_value flips only the flag: value survives as a default.
    expectFloatState(tensor, true, PI_FLOAT, std::nullopt, false, true);
}

// --- Runtime user-supplied creation path (flag true, value cleared) ----------

TEST(TestTensorValueAttributes, SetAsRuntimeParameterClearsValue)
{
    TensorAttributes tensor(PI_FLOAT); // start as compile-time constant
    tensor.set_as_runtime_parameter();
    expectFloatState(tensor, true, std::nullopt, std::nullopt, false, true);
}

TEST(TestTensorValueAttributes, ClearValueOnRuntimeDefaultBecomesUserSupplied)
{
    TensorAttributes tensor(PI_FLOAT, ScalarType::RUNTIME_PARAM);
    tensor.clear_value();
    // Flag stays true, value gone: this is the pure user-supplied state.
    expectFloatState(tensor, true, std::nullopt, std::nullopt, false, true);
}

// --- set_value sets the runtime flag regardless of prior flag state ----------

TEST(TestTensorValueAttributes, SetValueAfterSetIsPassByValueIsRuntimeWithDefault)
{
    TensorAttributes tensor;
    tensor.set_is_pass_by_value(true);
    tensor.set_value(PI_FLOAT); // set_value bakes a default and sets the runtime flag
    expectFloatState(tensor, true, PI_FLOAT, std::nullopt, false, true);
}

// --- Ordinary (not by-value) -------------------------------------------------

TEST(TestTensorValueAttributes, DefaultConstructedIsOrdinary)
{
    const TensorAttributes tensor;
    expectFloatState(tensor, false, std::nullopt, std::nullopt, false, false);
}

// --- Type dispatch: wrong-type queries return nullopt ------------------------

TEST(TestTensorValueAttributes, WrongTypeCompileTimeConstantReturnsNullopt)
{
    TensorAttributes tensor;
    tensor.set_compile_time_constant(42.0F); // compile-time constant float

    const std::optional<float> match = tensor.get_compile_time_constant<float>();
    ASSERT_TRUE(match.has_value());
    EXPECT_FLOAT_EQ(match.value(), 42.0F);

    EXPECT_FALSE(tensor.get_compile_time_constant<half>().has_value());
    EXPECT_FALSE(tensor.get_compile_time_constant<bfloat16>().has_value());
    EXPECT_FALSE(tensor.get_compile_time_constant<uint8_t>().has_value());
    EXPECT_FALSE(tensor.get_compile_time_constant<int32_t>().has_value());
    EXPECT_FALSE(tensor.get_compile_time_constant<int64_t>().has_value());
    EXPECT_FALSE(tensor.get_compile_time_constant<double>().has_value());
    EXPECT_FALSE(tensor.get_compile_time_constant<bool>().has_value());

    // Compile-time constant never answers the runtime getter, not even for float.
    EXPECT_FALSE(tensor.get_pass_by_value<float>().has_value());
}

TEST(TestTensorValueAttributes, WrongTypeRuntimeWithDefaultReturnsNullopt)
{
    const TensorAttributes tensor(int32_t{123}, ScalarType::RUNTIME_PARAM);

    const std::optional<int32_t> match = tensor.get_pass_by_value<int32_t>();
    ASSERT_TRUE(match.has_value());
    EXPECT_EQ(match.value(), 123);

    EXPECT_FALSE(tensor.get_pass_by_value<float>().has_value());
    EXPECT_FALSE(tensor.get_pass_by_value<double>().has_value());
    EXPECT_FALSE(tensor.get_pass_by_value<half>().has_value());
    EXPECT_FALSE(tensor.get_pass_by_value<bfloat16>().has_value());
    EXPECT_FALSE(tensor.get_pass_by_value<uint8_t>().has_value());
    EXPECT_FALSE(tensor.get_pass_by_value<int64_t>().has_value());
    EXPECT_FALSE(tensor.get_pass_by_value<bool>().has_value());

    // Runtime tensor never answers the compile-time getter, not even for int32.
    EXPECT_FALSE(tensor.get_compile_time_constant<int32_t>().has_value());
}

TEST(TestTensorValueAttributes, RuntimeWithDefaultRoundTripsAllTypes)
{
    expectRuntimeRoundTrip<float>(PI_FLOAT);
    expectRuntimeRoundTrip<double>(2.718281828459045);
    expectRuntimeRoundTrip<uint8_t>(200);
    expectRuntimeRoundTrip<int32_t>(-12345);
    expectRuntimeRoundTrip<int64_t>(456789012345LL);
    expectRuntimeRoundTrip<bool>(true);
    expectRuntimeRoundTrip<bool>(false);
}
