// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// MIOPEN_CONVOLUTION_ATTRIB_MATH_TYPE semantics: a descriptor takes the global default from
// MIOPEN_TF32_OVERRIDE when it is created; afterwards only the descriptor decides.

#include <miopen/conv/problem_description.hpp>
#include <miopen/convolution.hpp>
#include <miopen/db_record.hpp>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>

#include <nlohmann/json.hpp>

#include <gtest/gtest.h>

#include "gtest_common.hpp"

namespace {

const lib_env::LibEnvVar kTf32Override{"MIOPEN_TF32_OVERRIDE"};

int GetMathType(miopenConvolutionDescriptor_t desc)
{
    int value = -1;
    EXPECT_EQ(miopenGetConvolutionAttribute(desc, MIOPEN_CONVOLUTION_ATTRIB_MATH_TYPE, &value),
              miopenStatusSuccess);
    return value;
}

struct ConvDesc
{
    ConvDesc() { EXPECT_EQ(miopenCreateConvolutionDescriptor(&desc), miopenStatusSuccess); }
    ~ConvDesc() { miopenDestroyConvolutionDescriptor(desc); }
    ConvDesc(const ConvDesc&)            = delete;
    ConvDesc& operator=(const ConvDesc&) = delete;
    miopenConvolutionDescriptor_t desc   = nullptr;
};

miopen::conv::ProblemDescription MakeProblem(const miopen::ConvolutionDescriptor& conv)
{
    const auto x = miopen::TensorDescriptor{miopenFloat, {1, 8, 16, 16}};
    const auto w = miopen::TensorDescriptor{miopenFloat, {8, 8, 3, 3}};
    const auto y = conv.GetForwardOutputTensor(x, w, miopenFloat);
    return {x, w, y, conv, miopen::conv::Direction::Forward};
}

} // namespace

TEST(CPU_ConvMathType_NONE, DefaultsToPedanticWhenEnvUnset)
{
    ScopedEnvironment<bool> env(kTf32Override);
    ConvDesc d;
    EXPECT_EQ(GetMathType(d.desc), miopenMathPedantic);
}

TEST(CPU_ConvMathType_NONE, EnvZeroSelectsPedantic)
{
    ScopedEnvironment<bool> env(kTf32Override, false);
    ConvDesc d;
    EXPECT_EQ(GetMathType(d.desc), miopenMathPedantic);
}

TEST(CPU_ConvMathType_NONE, EnvOneSelectsTf32)
{
    ScopedEnvironment<bool> env(kTf32Override, true);
    ConvDesc d;
    EXPECT_EQ(GetMathType(d.desc), miopenMathDefault);
}

TEST(CPU_ConvMathType_NONE, SetterOverridesGlobalDefaultBothWays)
{
    {
        ScopedEnvironment<bool> env(kTf32Override, true);
        ConvDesc d;
        ASSERT_EQ(miopenSetConvolutionAttribute(
                      d.desc, MIOPEN_CONVOLUTION_ATTRIB_MATH_TYPE, miopenMathPedantic),
                  miopenStatusSuccess);
        EXPECT_EQ(GetMathType(d.desc), miopenMathPedantic);
        EXPECT_FALSE(miopen::deref(d.desc).EnableTF32());
    }
    {
        ScopedEnvironment<bool> env(kTf32Override, false);
        ConvDesc d;
        ASSERT_EQ(miopenSetConvolutionAttribute(
                      d.desc, MIOPEN_CONVOLUTION_ATTRIB_MATH_TYPE, miopenMathDefault),
                  miopenStatusSuccess);
        EXPECT_EQ(GetMathType(d.desc), miopenMathDefault);
        EXPECT_TRUE(miopen::deref(d.desc).EnableTF32());
    }
}

TEST(CPU_ConvMathType_NONE, GlobalDefaultIsNotConsultedAfterCreation)
{
    ScopedEnvironment<bool> env(kTf32Override, false);
    ConvDesc d;
    {
        ScopedEnvironment<bool> later(kTf32Override, true);
        EXPECT_EQ(GetMathType(d.desc), miopenMathPedantic);
        EXPECT_FALSE(miopen::deref(d.desc).EnableTF32());
    }
}

TEST(CPU_ConvMathType_NONE, InitPreservesMathType)
{
    ScopedEnvironment<bool> env(kTf32Override);
    ConvDesc d;
    ASSERT_EQ(miopenSetConvolutionAttribute(
                  d.desc, MIOPEN_CONVOLUTION_ATTRIB_MATH_TYPE, miopenMathDefault),
              miopenStatusSuccess);
    ASSERT_EQ(miopenInitConvolutionDescriptor(d.desc, miopenConvolution, 1, 1, 1, 1, 1, 1),
              miopenStatusSuccess);
    EXPECT_EQ(GetMathType(d.desc), miopenMathDefault);

    const int pads[3] = {1, 1, 1}, strides[3] = {1, 1, 1}, dilations[3] = {1, 1, 1};
    ASSERT_EQ(
        miopenInitConvolutionNdDescriptor(d.desc, 3, pads, strides, dilations, miopenConvolution),
        miopenStatusSuccess);
    EXPECT_EQ(GetMathType(d.desc), miopenMathDefault);
}

TEST(CPU_ConvMathType_NONE, JsonRoundTripKeepsMathType)
{
    ScopedEnvironment<bool> env(kTf32Override);
    miopen::ConvolutionDescriptor conv{{1, 1}, {1, 1}, {1, 1}};
    conv.attribute.Set(MIOPEN_CONVOLUTION_ATTRIB_MATH_TYPE, miopenMathDefault);

    const nlohmann::json json = conv;
    miopen::ConvolutionDescriptor loaded{};
    json.get_to(loaded);
    EXPECT_TRUE(loaded.EnableTF32());

    // A record written before the math type was serialized keeps the global default.
    auto old_json = json;
    old_json["attribute"].erase("mathType");
    {
        ScopedEnvironment<bool> tf32(kTf32Override, true);
        miopen::ConvolutionDescriptor old_loaded{};
        old_json.get_to(old_loaded);
        EXPECT_TRUE(old_loaded.EnableTF32());
    }
}

TEST(CPU_ConvMathType_NONE, KeysFollowTheDescriptorOnly)
{
    ScopedEnvironment<bool> env(kTf32Override);
    miopen::ConvolutionDescriptor fp32{{1, 1}, {1, 1}, {1, 1}};
    miopen::ConvolutionDescriptor tf32{{1, 1}, {1, 1}, {1, 1}};
    tf32.attribute.Set(MIOPEN_CONVOLUTION_ATTRIB_MATH_TYPE, miopenMathDefault);

    const auto p_fp32 = MakeProblem(fp32);
    const auto p_tf32 = MakeProblem(tf32);
    EXPECT_FALSE(p_fp32.UseTF32());
    EXPECT_TRUE(p_tf32.UseTF32());

    const auto find_fp32 = miopen::DbRecord(miopen::DbKinds::FindDb, p_fp32).GetKey();
    const auto find_tf32 = miopen::DbRecord(miopen::DbKinds::FindDb, p_tf32).GetKey();
    const auto perf_fp32 = miopen::DbRecord(miopen::DbKinds::PerfDb, p_fp32).GetKey();
    const auto perf_tf32 = miopen::DbRecord(miopen::DbKinds::PerfDb, p_tf32).GetKey();
    EXPECT_EQ(find_tf32, find_fp32 + "_cxTF32");
    EXPECT_EQ(perf_tf32, perf_fp32 + "xTF32");
    EXPECT_NE(p_fp32.MakeNetworkConfig().ToString(), p_tf32.MakeNetworkConfig().ToString());

    // Creating the descriptor under the global default gives the same keys as the setter.
    ScopedEnvironment<bool> tf32_default(kTf32Override, true);
    const miopen::ConvolutionDescriptor from_env{{1, 1}, {1, 1}, {1, 1}};
    EXPECT_EQ(miopen::DbRecord(miopen::DbKinds::FindDb, MakeProblem(from_env)).GetKey(), find_tf32);
}
