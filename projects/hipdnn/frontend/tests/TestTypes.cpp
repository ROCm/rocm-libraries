// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "hipdnn_frontend/Types.hpp"
#include <gtest/gtest.h>
#include <sstream>

TEST(TestTypes, HeuristicModeConversion)
{
    using namespace hipdnn_frontend;

    EXPECT_EQ(toBackendType(HeuristicMode::FALLBACK),
              hipdnnBackendHeurMode_t::HIPDNN_HEUR_MODE_FALLBACK);
    EXPECT_EQ(toBackendType(HeuristicMode::A), hipdnnBackendHeurMode_t::HIPDNN_HEUR_MODE_FALLBACK);
    EXPECT_EQ(toBackendType(HeuristicMode::B), hipdnnBackendHeurMode_t::HIPDNN_HEUR_MODE_FALLBACK);
    EXPECT_EQ(toBackendType(HeuristicMode::OPENSOURCE),
              hipdnnBackendHeurMode_t::HIPDNN_HEUR_MODE_FALLBACK);
}

TEST(TestTypes, HeuristicModeToString)
{
    using namespace hipdnn_frontend;

    EXPECT_STREQ(to_string(HeuristicMode::FALLBACK), "FALLBACK");
    EXPECT_STREQ(to_string(HeuristicMode::A), "A");
    EXPECT_STREQ(to_string(HeuristicMode::B), "B");
    EXPECT_STREQ(to_string(HeuristicMode::OPENSOURCE), "OPENSOURCE");
}

TEST(TestTypes, BehaviorNoteFromBackend)
{
    using namespace hipdnn_frontend;

    EXPECT_EQ(fromHipdnnBehaviorNote(HIPDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION),
              BehaviorNote::RUNTIME_COMPILATION);
    EXPECT_EQ(fromHipdnnBehaviorNote(HIPDNN_BEHAVIOR_NOTE_REQUIRES_LAYOUT_TRANSFORM),
              BehaviorNote::REQUIRES_LAYOUT_TRANSFORM);
    EXPECT_EQ(fromHipdnnBehaviorNote(HIPDNN_BEHAVIOR_NOTE_SUPPORTS_GRAPH_CAPTURE),
              BehaviorNote::SUPPORTS_GRAPH_CAPTURE);
    EXPECT_EQ(fromHipdnnBehaviorNote(HIPDNN_BEHAVIOR_NOTE_EXTERNAL_LIBRARY_DEPENDENCY),
              BehaviorNote::EXTERNAL_LIBRARY_DEPENDENCY);
    EXPECT_EQ(fromHipdnnBehaviorNote(HIPDNN_BEHAVIOR_NOTE_SUPPORTS_EXECUTION_PLAN_SERIALIZATION),
              BehaviorNote::SUPPORTS_EXECUTION_PLAN_SERIALIZATION);

    constexpr hipdnnBackendBehaviorNote_t UNKNOWN_NOTE = HIPDNN_BEHAVIOR_NOTE_TYPE_COUNT + 1;
    EXPECT_EQ(fromHipdnnBehaviorNote(UNKNOWN_NOTE), static_cast<BehaviorNote>(UNKNOWN_NOTE));
}

TEST(TestTypes, IsKnownBehaviorNote)
{
    using namespace hipdnn_frontend;

    EXPECT_TRUE(isKnownBehaviorNote(BehaviorNote::RUNTIME_COMPILATION));
    EXPECT_TRUE(isKnownBehaviorNote(BehaviorNote::REQUIRES_LAYOUT_TRANSFORM));
    EXPECT_TRUE(isKnownBehaviorNote(BehaviorNote::SUPPORTS_GRAPH_CAPTURE));
    EXPECT_TRUE(isKnownBehaviorNote(BehaviorNote::EXTERNAL_LIBRARY_DEPENDENCY));
    EXPECT_TRUE(isKnownBehaviorNote(BehaviorNote::SUPPORTS_EXECUTION_PLAN_SERIALIZATION));
    EXPECT_FALSE(isKnownBehaviorNote(static_cast<BehaviorNote>(HIPDNN_BEHAVIOR_NOTE_TYPE_COUNT)));
}

TEST(TestTypes, BehaviorNoteToString)
{
    using namespace hipdnn_frontend;

    EXPECT_STREQ(to_string(BehaviorNote::RUNTIME_COMPILATION), "RUNTIME_COMPILATION");
    EXPECT_STREQ(to_string(BehaviorNote::REQUIRES_LAYOUT_TRANSFORM), "REQUIRES_LAYOUT_TRANSFORM");
    EXPECT_STREQ(to_string(BehaviorNote::SUPPORTS_GRAPH_CAPTURE), "SUPPORTS_GRAPH_CAPTURE");
    EXPECT_STREQ(to_string(BehaviorNote::EXTERNAL_LIBRARY_DEPENDENCY),
                 "EXTERNAL_LIBRARY_DEPENDENCY");
    EXPECT_STREQ(to_string(BehaviorNote::SUPPORTS_EXECUTION_PLAN_SERIALIZATION),
                 "SUPPORTS_EXECUTION_PLAN_SERIALIZATION");
    EXPECT_STREQ(to_string(static_cast<BehaviorNote>(-1)), "unknown");

    std::ostringstream oss;
    oss << BehaviorNote::SUPPORTS_GRAPH_CAPTURE;
    EXPECT_EQ(oss.str(), "SUPPORTS_GRAPH_CAPTURE");
}

TEST(TestTypes, GetDataTypeEnumFromType)
{
    using namespace hipdnn_frontend;

    EXPECT_EQ(getDataTypeEnumFromType<float>(), DataType::FLOAT);
    EXPECT_EQ(getDataTypeEnumFromType<half>(), DataType::HALF);
    EXPECT_EQ(getDataTypeEnumFromType<bfloat16>(), DataType::BFLOAT16);
    EXPECT_EQ(getDataTypeEnumFromType<double>(), DataType::DOUBLE);
    EXPECT_EQ(getDataTypeEnumFromType<uint8_t>(), DataType::UINT8);
    EXPECT_EQ(getDataTypeEnumFromType<int32_t>(), DataType::INT32);
    EXPECT_EQ(getDataTypeEnumFromType<int8_t>(), DataType::INT8);
    EXPECT_EQ(getDataTypeEnumFromType<fp8_e4m3>(), DataType::FP8_E4M3);
    EXPECT_EQ(getDataTypeEnumFromType<fp8_e5m2>(), DataType::FP8_E5M2);
    EXPECT_EQ(getDataTypeEnumFromType<int64_t>(), DataType::INT64);
    EXPECT_EQ(getDataTypeEnumFromType<bool>(), DataType::BOOLEAN);

    EXPECT_EQ(getDataTypeEnumFromType<float*>(), DataType::NOT_SET);
    EXPECT_EQ(getDataTypeEnumFromType<char>(), DataType::NOT_SET);
}

TEST(TestTypes, DataTypeToString)
{
    using namespace hipdnn_frontend;

    EXPECT_STREQ(to_string(DataType::FLOAT), "fp32");
    EXPECT_STREQ(to_string(DataType::HALF), "fp16");
    EXPECT_STREQ(to_string(DataType::BFLOAT16), "bf16");
    EXPECT_STREQ(to_string(DataType::DOUBLE), "fp64");
    EXPECT_STREQ(to_string(DataType::UINT8), "uint8");
    EXPECT_STREQ(to_string(DataType::INT32), "int32");
    EXPECT_STREQ(to_string(DataType::INT8), "int8");
    EXPECT_STREQ(to_string(DataType::FP8_E4M3), "fp8_e4m3");
    EXPECT_STREQ(to_string(DataType::FP8_E5M2), "fp8_e5m2");
    EXPECT_STREQ(to_string(DataType::FP8_E8M0), "fp8_e8m0");
    EXPECT_STREQ(to_string(DataType::FP4_E2M1), "fp4_e2m1");
    EXPECT_STREQ(to_string(DataType::INT4), "int4");
    EXPECT_STREQ(to_string(DataType::FP6_E2M3), "fp6_e2m3");
    EXPECT_STREQ(to_string(DataType::FP6_E3M2), "fp6_e3m2");
    EXPECT_STREQ(to_string(DataType::INT64), "int64");
    EXPECT_STREQ(to_string(DataType::BOOLEAN), "boolean");
    EXPECT_STREQ(to_string(DataType::INT8x4), "int8x4");
    EXPECT_STREQ(to_string(DataType::UINT8x4), "uint8x4");
    EXPECT_STREQ(to_string(DataType::INT8x32), "int8x32");
    EXPECT_STREQ(to_string(DataType::FAST_FLOAT_FOR_FP8), "fast_float_for_fp8");
    EXPECT_STREQ(to_string(DataType::COMPLEX_FP32), "complex_fp32");
    EXPECT_STREQ(to_string(DataType::COMPLEX_FP64), "complex_fp64");
    EXPECT_STREQ(to_string(DataType::NOT_SET), "unknown");
}

TEST(TestTypes, DataTypeCudnnCompatHasNoBackendMapping)
{
    using namespace hipdnn_frontend;

    for(auto dt : {DataType::INT8x4,
                   DataType::UINT8x4,
                   DataType::INT8x32,
                   DataType::FAST_FLOAT_FOR_FP8,
                   DataType::COMPLEX_FP32,
                   DataType::COMPLEX_FP64})
    {
        EXPECT_EQ(toHipdnnDataType(dt), std::nullopt)
            << "Unexpected backend mapping for " << to_string(dt);
    }
}

TEST(TestTypes, PointwiseModeToString)
{
    using namespace hipdnn_frontend;

    EXPECT_STREQ(to_string(PointwiseMode::NOT_SET), "NOT_SET");
    EXPECT_STREQ(to_string(PointwiseMode::RELU_FWD), "RELU_FWD");
    EXPECT_STREQ(to_string(PointwiseMode::ADD), "ADD");
    EXPECT_STREQ(to_string(PointwiseMode::BINARY_SELECT), "BINARY_SELECT");
    EXPECT_STREQ(to_string(PointwiseMode::MOD), "MOD");
    EXPECT_STREQ(to_string(PointwiseMode::POW), "POW");
    EXPECT_STREQ(to_string(PointwiseMode::COS), "COS");
    EXPECT_STREQ(to_string(PointwiseMode::COUNT), "UNKNOWN");

    // Verify all valid modes produce a non-UNKNOWN string
    for(auto mode : {PointwiseMode::NOT_SET,
                     PointwiseMode::ABS,
                     PointwiseMode::ADD,
                     PointwiseMode::ADD_SQUARE,
                     PointwiseMode::BINARY_SELECT,
                     PointwiseMode::CEIL,
                     PointwiseMode::CMP_EQ,
                     PointwiseMode::CMP_GE,
                     PointwiseMode::CMP_GT,
                     PointwiseMode::CMP_LE,
                     PointwiseMode::CMP_LT,
                     PointwiseMode::CMP_NEQ,
                     PointwiseMode::DIV,
                     PointwiseMode::ELU_BWD,
                     PointwiseMode::ELU_FWD,
                     PointwiseMode::ERF,
                     PointwiseMode::EXP,
                     PointwiseMode::FLOOR,
                     PointwiseMode::GELU_APPROX_TANH_BWD,
                     PointwiseMode::GELU_APPROX_TANH_FWD,
                     PointwiseMode::GELU_BWD,
                     PointwiseMode::GELU_FWD,
                     PointwiseMode::GEN_INDEX,
                     PointwiseMode::IDENTITY,
                     PointwiseMode::LOG,
                     PointwiseMode::LOGICAL_AND,
                     PointwiseMode::LOGICAL_NOT,
                     PointwiseMode::LOGICAL_OR,
                     PointwiseMode::MAX,
                     PointwiseMode::MIN,
                     PointwiseMode::MUL,
                     PointwiseMode::NEG,
                     PointwiseMode::RECIPROCAL,
                     PointwiseMode::RELU_BWD,
                     PointwiseMode::RELU_FWD,
                     PointwiseMode::RSQRT,
                     PointwiseMode::SIGMOID_BWD,
                     PointwiseMode::SIGMOID_FWD,
                     PointwiseMode::SIN,
                     PointwiseMode::SOFTPLUS_BWD,
                     PointwiseMode::SOFTPLUS_FWD,
                     PointwiseMode::SQRT,
                     PointwiseMode::SUB,
                     PointwiseMode::SWISH_BWD,
                     PointwiseMode::SWISH_FWD,
                     PointwiseMode::TAN,
                     PointwiseMode::TANH_BWD,
                     PointwiseMode::TANH_FWD,
                     PointwiseMode::MOD,
                     PointwiseMode::POW,
                     PointwiseMode::COS})
    {
        EXPECT_STRNE(to_string(mode), "UNKNOWN")
            << "to_string returned UNKNOWN for PointwiseMode " << static_cast<int>(mode);
        EXPECT_STRNE(to_string(mode), "")
            << "to_string returned empty for PointwiseMode " << static_cast<int>(mode);
    }
}

TEST(TestTypes, PointwiseModeCudnnCompatClassificationAndMapping)
{
    using namespace hipdnn_frontend;

    EXPECT_TRUE(isUnaryPointwiseMode(PointwiseMode::COS));
    EXPECT_TRUE(isBinaryPointwiseMode(PointwiseMode::MOD));
    EXPECT_TRUE(isBinaryPointwiseMode(PointwiseMode::POW));

    EXPECT_FALSE(isBinaryPointwiseMode(PointwiseMode::COS));
    EXPECT_FALSE(isUnaryPointwiseMode(PointwiseMode::MOD));
    EXPECT_FALSE(isUnaryPointwiseMode(PointwiseMode::POW));

    for(auto mode : {PointwiseMode::MOD, PointwiseMode::POW, PointwiseMode::COS})
    {
        EXPECT_EQ(toBackendPointwiseMode(mode), std::nullopt)
            << "Unexpected backend mapping for " << to_string(mode);
    }
}

TEST(TestTypes, ResampleAndPaddingCudnnCompatHaveNoBackendMapping)
{
    using namespace hipdnn_frontend;

    EXPECT_EQ(toBackendResampleMode(ResampleMode::BILINEAR), std::nullopt);
    EXPECT_EQ(toBackendResampleMode(ResampleMode::NEAREST), std::nullopt);
    EXPECT_EQ(toBackendPaddingMode(PaddingMode::EDGE_VAL_PAD), std::nullopt);
}

TEST(TestTypes, DataTypeStreamOperator)
{
    using namespace hipdnn_frontend;

    std::ostringstream oss;

    oss << DataType::FLOAT;
    EXPECT_EQ(oss.str(), "fp32");
    oss.str("");

    oss << DataType::HALF;
    EXPECT_EQ(oss.str(), "fp16");
    oss.str("");

    oss << DataType::BFLOAT16;
    EXPECT_EQ(oss.str(), "bf16");
    oss.str("");

    oss << DataType::DOUBLE;
    EXPECT_EQ(oss.str(), "fp64");
    oss.str("");

    oss << DataType::UINT8;
    EXPECT_EQ(oss.str(), "uint8");
    oss.str("");

    oss << DataType::INT32;
    EXPECT_EQ(oss.str(), "int32");
    oss.str("");

    oss << DataType::INT8;
    EXPECT_EQ(oss.str(), "int8");
    oss.str("");

    oss << DataType::FP8_E4M3;
    EXPECT_EQ(oss.str(), "fp8_e4m3");
    oss.str("");

    oss << DataType::FP8_E5M2;
    EXPECT_EQ(oss.str(), "fp8_e5m2");
    oss.str("");

    oss << DataType::FP8_E8M0;
    EXPECT_EQ(oss.str(), "fp8_e8m0");
    oss.str("");

    oss << DataType::FP4_E2M1;
    EXPECT_EQ(oss.str(), "fp4_e2m1");
    oss.str("");

    oss << DataType::INT4;
    EXPECT_EQ(oss.str(), "int4");
    oss.str("");

    oss << DataType::FP6_E2M3;
    EXPECT_EQ(oss.str(), "fp6_e2m3");
    oss.str("");

    oss << DataType::FP6_E3M2;
    EXPECT_EQ(oss.str(), "fp6_e3m2");
    oss.str("");

    oss << DataType::INT64;
    EXPECT_EQ(oss.str(), "int64");
    oss.str("");

    oss << DataType::BOOLEAN;
    EXPECT_EQ(oss.str(), "boolean");
    oss.str("");

    oss << DataType::NOT_SET;
    EXPECT_EQ(oss.str(), "unknown");
}

TEST(TestTypes, KnobValueTypeToString)
{
    using namespace hipdnn_frontend;

    EXPECT_STREQ(to_string(KnobValueType::INT64), "int64");
    EXPECT_STREQ(to_string(KnobValueType::FLOAT64), "float64");
    EXPECT_STREQ(to_string(KnobValueType::STRING), "string");
}

TEST(TestTypes, KnobValueTypeStreamOperator)
{
    using namespace hipdnn_frontend;

    std::ostringstream oss;

    oss << KnobValueType::INT64;
    EXPECT_EQ(oss.str(), "int64");
    oss.str("");

    oss << KnobValueType::FLOAT64;
    EXPECT_EQ(oss.str(), "float64");
    oss.str("");

    oss << KnobValueType::STRING;
    EXPECT_EQ(oss.str(), "string");
}

TEST(TestTypes, GetKnobValueTypeFromVariantInt64)
{
    using namespace hipdnn_frontend;

    const std::variant<int64_t, double, std::string> value = static_cast<int64_t>(42);
    EXPECT_EQ(getKnobValueTypeFromVariant(value), KnobValueType::INT64);
}

TEST(TestTypes, GetKnobValueTypeFromVariantFloat64)
{
    using namespace hipdnn_frontend;

    const std::variant<int64_t, double, std::string> value = 3.14;
    EXPECT_EQ(getKnobValueTypeFromVariant(value), KnobValueType::FLOAT64);
}

TEST(TestTypes, GetKnobValueTypeFromVariantString)
{
    using namespace hipdnn_frontend;

    const std::variant<int64_t, double, std::string> value = std::string("test");
    EXPECT_EQ(getKnobValueTypeFromVariant(value), KnobValueType::STRING);
}

TEST(TestTypes, ToHipdnnDataType)
{
    using namespace hipdnn_frontend;

    EXPECT_EQ(toHipdnnDataType(DataType::FLOAT), HIPDNN_DATA_FLOAT);
    EXPECT_EQ(toHipdnnDataType(DataType::DOUBLE), HIPDNN_DATA_DOUBLE);
    EXPECT_EQ(toHipdnnDataType(DataType::HALF), HIPDNN_DATA_HALF);
    EXPECT_EQ(toHipdnnDataType(DataType::INT8), HIPDNN_DATA_INT8);
    EXPECT_EQ(toHipdnnDataType(DataType::INT32), HIPDNN_DATA_INT32);
    EXPECT_EQ(toHipdnnDataType(DataType::UINT8), HIPDNN_DATA_UINT8);
    EXPECT_EQ(toHipdnnDataType(DataType::BFLOAT16), HIPDNN_DATA_BFLOAT16);
    EXPECT_EQ(toHipdnnDataType(DataType::FP8_E4M3), HIPDNN_DATA_FP8_E4M3);
    EXPECT_EQ(toHipdnnDataType(DataType::FP8_E5M2), HIPDNN_DATA_FP8_E5M2);
    EXPECT_EQ(toHipdnnDataType(DataType::FP8_E8M0), HIPDNN_DATA_FP8_E8M0);
    EXPECT_EQ(toHipdnnDataType(DataType::FP4_E2M1), HIPDNN_DATA_FP4_E2M1);
    EXPECT_EQ(toHipdnnDataType(DataType::INT4), HIPDNN_DATA_INT4);
    EXPECT_EQ(toHipdnnDataType(DataType::FP6_E2M3), HIPDNN_DATA_FP6_E2M3_EXT);
    EXPECT_EQ(toHipdnnDataType(DataType::FP6_E3M2), HIPDNN_DATA_FP6_E3M2_EXT);
    EXPECT_EQ(toHipdnnDataType(DataType::INT64), HIPDNN_DATA_INT64);
    EXPECT_EQ(toHipdnnDataType(DataType::BOOLEAN), HIPDNN_DATA_BOOLEAN);
    EXPECT_EQ(toHipdnnDataType(DataType::NOT_SET), std::nullopt);
}

TEST(TestTypes, FromHipdnnDataTypeAllValidTypes)
{
    using namespace hipdnn_frontend;

    auto check = [](hipdnnDataType_t hipdnnType, DataType expected) {
        auto [dt, err] = fromHipdnnDataType(hipdnnType);
        EXPECT_TRUE(err.is_good())
            << "Error for " << static_cast<int>(hipdnnType) << ": " << err.get_message();
        EXPECT_EQ(dt, expected);
    };

    check(HIPDNN_DATA_FLOAT, DataType::FLOAT);
    check(HIPDNN_DATA_DOUBLE, DataType::DOUBLE);
    check(HIPDNN_DATA_HALF, DataType::HALF);
    check(HIPDNN_DATA_INT8, DataType::INT8);
    check(HIPDNN_DATA_INT32, DataType::INT32);
    check(HIPDNN_DATA_UINT8, DataType::UINT8);
    check(HIPDNN_DATA_BFLOAT16, DataType::BFLOAT16);
    check(HIPDNN_DATA_FP8_E4M3, DataType::FP8_E4M3);
    check(HIPDNN_DATA_FP8_E5M2, DataType::FP8_E5M2);
    check(HIPDNN_DATA_FP8_E8M0, DataType::FP8_E8M0);
    check(HIPDNN_DATA_FP4_E2M1, DataType::FP4_E2M1);
    check(HIPDNN_DATA_INT4, DataType::INT4);
    check(HIPDNN_DATA_FP6_E2M3_EXT, DataType::FP6_E2M3);
    check(HIPDNN_DATA_FP6_E3M2_EXT, DataType::FP6_E3M2);
    check(HIPDNN_DATA_INT64, DataType::INT64);
    check(HIPDNN_DATA_BOOLEAN, DataType::BOOLEAN);
}

TEST(TestTypes, FromHipdnnDataTypeUnknownReturnsError)
{
    using namespace hipdnn_frontend;

    auto unknownType = static_cast<hipdnnDataType_t>(9999);
    auto [dt, err] = fromHipdnnDataType(unknownType);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_EQ(dt, DataType::NOT_SET);
    EXPECT_TRUE(err.get_message().find("Unknown") != std::string::npos);
}

TEST(TestTypes, FromHipdnnDataTypeRoundTrip)
{
    using namespace hipdnn_frontend;

    for(auto dt : {DataType::FLOAT,
                   DataType::DOUBLE,
                   DataType::HALF,
                   DataType::INT8,
                   DataType::INT32,
                   DataType::UINT8,
                   DataType::BFLOAT16,
                   DataType::FP8_E4M3,
                   DataType::FP8_E5M2,
                   DataType::FP8_E8M0,
                   DataType::FP4_E2M1,
                   DataType::INT4,
                   DataType::FP6_E2M3,
                   DataType::FP6_E3M2,
                   DataType::INT64,
                   DataType::BOOLEAN})
    {
        auto hipdnnOpt = toHipdnnDataType(dt);
        ASSERT_TRUE(hipdnnOpt.has_value()) << "toHipdnnDataType failed for " << to_string(dt);
        auto [roundTripped, err] = fromHipdnnDataType(hipdnnOpt.value());
        EXPECT_TRUE(err.is_good()) << "fromHipdnnDataType failed for " << to_string(dt);
        EXPECT_EQ(roundTripped, dt) << "Round-trip mismatch for " << to_string(dt);
    }
}

TEST(TestTypes, FromHipdnnConvModeValidModes)
{
    using namespace hipdnn_frontend;

    auto [xcorr, xcorrErr] = fromHipdnnConvMode(HIPDNN_CROSS_CORRELATION);
    EXPECT_TRUE(xcorrErr.is_good());
    EXPECT_EQ(xcorr, ConvolutionMode::CROSS_CORRELATION);

    auto [conv, convErr] = fromHipdnnConvMode(HIPDNN_CONVOLUTION);
    EXPECT_TRUE(convErr.is_good());
    EXPECT_EQ(conv, ConvolutionMode::CONVOLUTION);
}

TEST(TestTypes, FromHipdnnConvModeUnknownReturnsError)
{
    using namespace hipdnn_frontend;

    auto unknownMode = static_cast<hipdnnConvolutionMode_t>(9999);
    auto [mode, err] = fromHipdnnConvMode(unknownMode);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_EQ(mode, ConvolutionMode::NOT_SET);
    EXPECT_TRUE(err.get_message().find("Unknown") != std::string::npos);
}

TEST(TestTypes, FromHipdnnConvModeRoundTrip)
{
    using namespace hipdnn_frontend;

    for(auto mode : {ConvolutionMode::CROSS_CORRELATION, ConvolutionMode::CONVOLUTION})
    {
        auto hipdnnOpt = toBackendConvMode(mode);
        ASSERT_TRUE(hipdnnOpt.has_value());
        auto [roundTripped, err] = fromHipdnnConvMode(hipdnnOpt.value());
        EXPECT_TRUE(err.is_good());
        EXPECT_EQ(roundTripped, mode);
    }
}

TEST(TestTypes, FromHipdnnPointwiseModeAllValidModes)
{
    using namespace hipdnn_frontend;

    const std::vector<std::pair<hipdnnPointwiseMode_t, PointwiseMode>> validModes = {
        {HIPDNN_POINTWISE_ABS, PointwiseMode::ABS},
        {HIPDNN_POINTWISE_ADD, PointwiseMode::ADD},
        {HIPDNN_POINTWISE_ADD_SQUARE, PointwiseMode::ADD_SQUARE},
        {HIPDNN_POINTWISE_BINARY_SELECT, PointwiseMode::BINARY_SELECT},
        {HIPDNN_POINTWISE_CEIL, PointwiseMode::CEIL},
        {HIPDNN_POINTWISE_CMP_EQ, PointwiseMode::CMP_EQ},
        {HIPDNN_POINTWISE_CMP_GE, PointwiseMode::CMP_GE},
        {HIPDNN_POINTWISE_CMP_GT, PointwiseMode::CMP_GT},
        {HIPDNN_POINTWISE_CMP_LE, PointwiseMode::CMP_LE},
        {HIPDNN_POINTWISE_CMP_LT, PointwiseMode::CMP_LT},
        {HIPDNN_POINTWISE_CMP_NEQ, PointwiseMode::CMP_NEQ},
        {HIPDNN_POINTWISE_DIV, PointwiseMode::DIV},
        {HIPDNN_POINTWISE_ELU_BWD, PointwiseMode::ELU_BWD},
        {HIPDNN_POINTWISE_ELU_FWD, PointwiseMode::ELU_FWD},
        {HIPDNN_POINTWISE_ERF, PointwiseMode::ERF},
        {HIPDNN_POINTWISE_EXP, PointwiseMode::EXP},
        {HIPDNN_POINTWISE_FLOOR, PointwiseMode::FLOOR},
        {HIPDNN_POINTWISE_GELU_APPROX_TANH_BWD, PointwiseMode::GELU_APPROX_TANH_BWD},
        {HIPDNN_POINTWISE_GELU_APPROX_TANH_FWD, PointwiseMode::GELU_APPROX_TANH_FWD},
        {HIPDNN_POINTWISE_GELU_BWD, PointwiseMode::GELU_BWD},
        {HIPDNN_POINTWISE_GELU_FWD, PointwiseMode::GELU_FWD},
        {HIPDNN_POINTWISE_GEN_INDEX, PointwiseMode::GEN_INDEX},
        {HIPDNN_POINTWISE_IDENTITY, PointwiseMode::IDENTITY},
        {HIPDNN_POINTWISE_LOG, PointwiseMode::LOG},
        {HIPDNN_POINTWISE_LOGICAL_AND, PointwiseMode::LOGICAL_AND},
        {HIPDNN_POINTWISE_LOGICAL_NOT, PointwiseMode::LOGICAL_NOT},
        {HIPDNN_POINTWISE_LOGICAL_OR, PointwiseMode::LOGICAL_OR},
        {HIPDNN_POINTWISE_MAX, PointwiseMode::MAX},
        {HIPDNN_POINTWISE_MIN, PointwiseMode::MIN},
        {HIPDNN_POINTWISE_MUL, PointwiseMode::MUL},
        {HIPDNN_POINTWISE_NEG, PointwiseMode::NEG},
        {HIPDNN_POINTWISE_RECIPROCAL, PointwiseMode::RECIPROCAL},
        {HIPDNN_POINTWISE_RELU_BWD, PointwiseMode::RELU_BWD},
        {HIPDNN_POINTWISE_RELU_FWD, PointwiseMode::RELU_FWD},
        {HIPDNN_POINTWISE_RSQRT, PointwiseMode::RSQRT},
        {HIPDNN_POINTWISE_SIGMOID_BWD, PointwiseMode::SIGMOID_BWD},
        {HIPDNN_POINTWISE_SIGMOID_FWD, PointwiseMode::SIGMOID_FWD},
        {HIPDNN_POINTWISE_SIN, PointwiseMode::SIN},
        {HIPDNN_POINTWISE_SOFTPLUS_BWD, PointwiseMode::SOFTPLUS_BWD},
        {HIPDNN_POINTWISE_SOFTPLUS_FWD, PointwiseMode::SOFTPLUS_FWD},
        {HIPDNN_POINTWISE_SQRT, PointwiseMode::SQRT},
        {HIPDNN_POINTWISE_SUB, PointwiseMode::SUB},
        {HIPDNN_POINTWISE_SWISH_BWD, PointwiseMode::SWISH_BWD},
        {HIPDNN_POINTWISE_SWISH_FWD, PointwiseMode::SWISH_FWD},
        {HIPDNN_POINTWISE_TAN, PointwiseMode::TAN},
        {HIPDNN_POINTWISE_TANH_BWD, PointwiseMode::TANH_BWD},
        {HIPDNN_POINTWISE_TANH_FWD, PointwiseMode::TANH_FWD},
    };

    for(const auto& [hipdnnMode, expectedMode] : validModes)
    {
        auto [mode, err] = fromHipdnnPointwiseMode(hipdnnMode);
        EXPECT_TRUE(err.is_good())
            << "fromHipdnnPointwiseMode failed for mode value " << static_cast<int>(hipdnnMode);
        EXPECT_EQ(mode, expectedMode) << "Mismatch for mode value " << static_cast<int>(hipdnnMode);
    }
}

TEST(TestTypes, FromHipdnnPointwiseModeUnknownReturnsError)
{
    using namespace hipdnn_frontend;

    auto unknownMode = static_cast<hipdnnPointwiseMode_t>(9999);
    auto [mode, err] = fromHipdnnPointwiseMode(unknownMode);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_EQ(mode, PointwiseMode::NOT_SET);
    EXPECT_TRUE(err.get_message().find("Unknown") != std::string::npos);
}

TEST(TestTypes, FromHipdnnNormFwdPhaseValidPhases)
{
    using namespace hipdnn_frontend;

    auto [inference, inferenceErr] = fromHipdnnNormFwdPhase(HIPDNN_NORM_FWD_INFERENCE);
    EXPECT_TRUE(inferenceErr.is_good());
    EXPECT_EQ(inference, NormFwdPhase::INFERENCE);

    auto [training, trainingErr] = fromHipdnnNormFwdPhase(HIPDNN_NORM_FWD_TRAINING);
    EXPECT_TRUE(trainingErr.is_good());
    EXPECT_EQ(training, NormFwdPhase::TRAINING);
}

TEST(TestTypes, FromHipdnnNormFwdPhaseUnknownReturnsError)
{
    using namespace hipdnn_frontend;

    auto unknownPhase = static_cast<hipdnnNormFwdPhase_t>(9999);
    auto [phase, err] = fromHipdnnNormFwdPhase(unknownPhase);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_EQ(phase, NormFwdPhase::NOT_SET);
    EXPECT_TRUE(err.get_message().find("Unknown") != std::string::npos);
}

TEST(TestTypes, FromHipdnnNormFwdPhaseRoundTrip)
{
    using namespace hipdnn_frontend;

    for(auto phase : {NormFwdPhase::INFERENCE, NormFwdPhase::TRAINING})
    {
        auto hipdnnOpt = toBackendNormFwdPhase(phase);
        ASSERT_TRUE(hipdnnOpt.has_value())
            << "toBackendNormFwdPhase failed for phase " << static_cast<int>(phase);
        auto [roundTripped, err] = fromHipdnnNormFwdPhase(hipdnnOpt.value());
        EXPECT_TRUE(err.is_good())
            << "fromHipdnnNormFwdPhase failed for phase " << static_cast<int>(phase);
        EXPECT_EQ(roundTripped, phase)
            << "Round-trip mismatch for phase " << static_cast<int>(phase);
    }
}

TEST(TestTypes, FromHipdnnDiagonalAlignmentValidValues)
{
    using namespace hipdnn_frontend;

    auto [topLeft, topLeftErr]
        = fromHipdnnDiagonalAlignment(HIPDNN_DIAGONAL_ALIGNMENT_TOP_LEFT_EXT);
    EXPECT_TRUE(topLeftErr.is_good());
    EXPECT_EQ(topLeft, DiagonalAlignment::TOP_LEFT);

    auto [bottomRight, bottomRightErr]
        = fromHipdnnDiagonalAlignment(HIPDNN_DIAGONAL_ALIGNMENT_BOTTOM_RIGHT_EXT);
    EXPECT_TRUE(bottomRightErr.is_good());
    EXPECT_EQ(bottomRight, DiagonalAlignment::BOTTOM_RIGHT);
}

TEST(TestTypes, FromHipdnnDiagonalAlignmentUnknownReturnsError)
{
    using namespace hipdnn_frontend;

    auto unknownVal = static_cast<hipdnnDiagonalAlignment_t>(9999);
    auto [alignment, err] = fromHipdnnDiagonalAlignment(unknownVal);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_EQ(alignment, DiagonalAlignment::TOP_LEFT);
    EXPECT_TRUE(err.get_message().find("Unknown") != std::string::npos);
}

TEST(TestTypes, FromHipdnnDiagonalAlignmentRoundTrip)
{
    using namespace hipdnn_frontend;

    for(auto alignment : {DiagonalAlignment::TOP_LEFT, DiagonalAlignment::BOTTOM_RIGHT})
    {
        auto backend = toBackendDiagonalAlignment(alignment);
        auto [roundTripped, err] = fromHipdnnDiagonalAlignment(backend);
        EXPECT_TRUE(err.is_good());
        EXPECT_EQ(roundTripped, alignment);
    }
}

TEST(TestTypes, FromHipdnnAttentionImplementationValidValues)
{
    using namespace hipdnn_frontend;

    auto [autoVal, autoErr]
        = fromHipdnnAttentionImplementation(HIPDNN_ATTENTION_IMPLEMENTATION_AUTO_EXT);
    EXPECT_TRUE(autoErr.is_good());
    EXPECT_EQ(autoVal, AttentionImplementation::AUTO);

    auto [composite, compositeErr]
        = fromHipdnnAttentionImplementation(HIPDNN_ATTENTION_IMPLEMENTATION_COMPOSITE_EXT);
    EXPECT_TRUE(compositeErr.is_good());
    EXPECT_EQ(composite, AttentionImplementation::COMPOSITE);

    auto [unified, unifiedErr]
        = fromHipdnnAttentionImplementation(HIPDNN_ATTENTION_IMPLEMENTATION_UNIFIED_EXT);
    EXPECT_TRUE(unifiedErr.is_good());
    EXPECT_EQ(unified, AttentionImplementation::UNIFIED);
}

TEST(TestTypes, FromHipdnnAttentionImplementationUnknownReturnsError)
{
    using namespace hipdnn_frontend;

    auto unknownVal = static_cast<hipdnnAttentionImplementation_t>(9999);
    auto [impl, err] = fromHipdnnAttentionImplementation(unknownVal);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_EQ(impl, AttentionImplementation::AUTO);
    EXPECT_TRUE(err.get_message().find("Unknown") != std::string::npos);
}

TEST(TestTypes, FromHipdnnAttentionImplementationRoundTrip)
{
    using namespace hipdnn_frontend;

    for(auto impl : {AttentionImplementation::AUTO,
                     AttentionImplementation::COMPOSITE,
                     AttentionImplementation::UNIFIED})
    {
        auto backend = toBackendAttentionImplementation(impl);
        auto [roundTripped, err] = fromHipdnnAttentionImplementation(backend);
        EXPECT_TRUE(err.is_good());
        EXPECT_EQ(roundTripped, impl);
    }
}

TEST(TestTypes, FromHipdnnPointwiseModeRoundTrip)
{
    using namespace hipdnn_frontend;

    for(auto mode : {PointwiseMode::ABS,
                     PointwiseMode::ADD,
                     PointwiseMode::ADD_SQUARE,
                     PointwiseMode::BINARY_SELECT,
                     PointwiseMode::CEIL,
                     PointwiseMode::CMP_EQ,
                     PointwiseMode::CMP_GE,
                     PointwiseMode::CMP_GT,
                     PointwiseMode::CMP_LE,
                     PointwiseMode::CMP_LT,
                     PointwiseMode::CMP_NEQ,
                     PointwiseMode::DIV,
                     PointwiseMode::ELU_BWD,
                     PointwiseMode::ELU_FWD,
                     PointwiseMode::ERF,
                     PointwiseMode::EXP,
                     PointwiseMode::FLOOR,
                     PointwiseMode::GELU_APPROX_TANH_BWD,
                     PointwiseMode::GELU_APPROX_TANH_FWD,
                     PointwiseMode::GELU_BWD,
                     PointwiseMode::GELU_FWD,
                     PointwiseMode::GEN_INDEX,
                     PointwiseMode::IDENTITY,
                     PointwiseMode::LOG,
                     PointwiseMode::LOGICAL_AND,
                     PointwiseMode::LOGICAL_NOT,
                     PointwiseMode::LOGICAL_OR,
                     PointwiseMode::MAX,
                     PointwiseMode::MIN,
                     PointwiseMode::MUL,
                     PointwiseMode::NEG,
                     PointwiseMode::RECIPROCAL,
                     PointwiseMode::RELU_BWD,
                     PointwiseMode::RELU_FWD,
                     PointwiseMode::RSQRT,
                     PointwiseMode::SIGMOID_BWD,
                     PointwiseMode::SIGMOID_FWD,
                     PointwiseMode::SIN,
                     PointwiseMode::SOFTPLUS_BWD,
                     PointwiseMode::SOFTPLUS_FWD,
                     PointwiseMode::SQRT,
                     PointwiseMode::SUB,
                     PointwiseMode::SWISH_BWD,
                     PointwiseMode::SWISH_FWD,
                     PointwiseMode::TAN,
                     PointwiseMode::TANH_BWD,
                     PointwiseMode::TANH_FWD})
    {
        auto hipdnnOpt = toBackendPointwiseMode(mode);
        ASSERT_TRUE(hipdnnOpt.has_value())
            << "toBackendPointwiseMode failed for mode " << static_cast<int>(mode);
        auto [roundTripped, err] = fromHipdnnPointwiseMode(hipdnnOpt.value());
        EXPECT_TRUE(err.is_good())
            << "fromHipdnnPointwiseMode failed for mode " << static_cast<int>(mode);
        EXPECT_EQ(roundTripped, mode) << "Round-trip mismatch for mode " << static_cast<int>(mode);
    }
}

TEST(TestTypes, FromHipdnnReductionModeAllValidModes)
{
    using namespace hipdnn_frontend;

    const std::vector<std::pair<hipdnnReduceTensorOp_t, ReductionMode>> validModes = {
        {HIPDNN_REDUCE_TENSOR_ADD, ReductionMode::ADD},
        {HIPDNN_REDUCE_TENSOR_MUL, ReductionMode::MUL},
        {HIPDNN_REDUCE_TENSOR_MIN, ReductionMode::MIN},
        {HIPDNN_REDUCE_TENSOR_MAX, ReductionMode::MAX},
        {HIPDNN_REDUCE_TENSOR_AMAX, ReductionMode::AMAX},
        {HIPDNN_REDUCE_TENSOR_AVG, ReductionMode::AVG},
        {HIPDNN_REDUCE_TENSOR_NORM1, ReductionMode::NORM1},
        {HIPDNN_REDUCE_TENSOR_NORM2, ReductionMode::NORM2},
        {HIPDNN_REDUCE_TENSOR_MUL_NO_ZEROS, ReductionMode::MUL_NO_ZEROS},
    };

    for(const auto& [hipdnnMode, expectedMode] : validModes)
    {
        auto [mode, err] = fromHipdnnReduceTensorOp(hipdnnMode);
        EXPECT_TRUE(err.is_good())
            << "fromHipdnnReduceTensorOp failed for mode value " << static_cast<int>(hipdnnMode);
        EXPECT_EQ(mode, expectedMode) << "Mismatch for mode value " << static_cast<int>(hipdnnMode);
    }
}

TEST(TestTypes, FromHipdnnReductionModeUnknownReturnsError)
{
    using namespace hipdnn_frontend;

    auto unknownMode = static_cast<hipdnnReduceTensorOp_t>(9999);
    auto [mode, err] = fromHipdnnReduceTensorOp(unknownMode);
    EXPECT_TRUE(err.is_bad());
    EXPECT_EQ(err.code, ErrorCode::HIPDNN_BACKEND_ERROR);
    EXPECT_EQ(mode, ReductionMode::NOT_SET);
    EXPECT_TRUE(err.get_message().find("Unknown") != std::string::npos);
}

TEST(TestTypes, FromHipdnnReductionModeRoundTrip)
{
    using namespace hipdnn_frontend;

    for(auto mode : {
            ReductionMode::ADD,
            ReductionMode::MUL,
            ReductionMode::MIN,
            ReductionMode::MAX,
            ReductionMode::AMAX,
            ReductionMode::AVG,
            ReductionMode::NORM1,
            ReductionMode::NORM2,
            ReductionMode::MUL_NO_ZEROS,
        })
    {
        auto hipdnnOpt = toBackendReductionMode(mode);
        ASSERT_TRUE(hipdnnOpt.has_value())
            << "toBackendReductionMode failed for mode " << static_cast<int>(mode);
        auto [roundTripped, err] = fromHipdnnReduceTensorOp(hipdnnOpt.value());
        EXPECT_TRUE(err.is_good())
            << "fromHipdnnReduceTensorOp failed for mode " << static_cast<int>(mode);
        EXPECT_EQ(roundTripped, mode) << "Round-trip mismatch for mode " << static_cast<int>(mode);
    }
}

TEST(TestTypes, ToBackendReductionModeNotSetReturnsNullopt)
{
    using namespace hipdnn_frontend;

    EXPECT_EQ(toBackendReductionMode(ReductionMode::NOT_SET), std::nullopt);
}
