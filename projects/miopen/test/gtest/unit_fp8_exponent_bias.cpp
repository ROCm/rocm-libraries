/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
#include <gtest/gtest.h>

#include <miopen/datatype.hpp>

#include "conv_test_base.hpp"
#include "unit_conv_solver.hpp"

#include <algorithm>
#include <string>

namespace {

std::string BiasDefine(miopenDataType_t type, const std::string& device_name)
{
    return miopen::GetDataTypeKBP(type, device_name).GenerateFor(miopen::kbp::HIP{});
}

void ExpectKernelBias(const std::string& device_name, int expected_bias)
{
    const auto expected = "-DMIOPEN_FP8_IEEE_EXPONENT_BIAS=" + std::to_string(expected_bias);
    for(const auto type : {miopenFloat8_fnuz, miopenBFloat8_fnuz})
    {
        EXPECT_NE(BiasDefine(type, device_name).find(expected), std::string::npos)
            << device_name << " type " << type << " expected " << expected;
        EXPECT_NE(miopen::GetDataTypeKernelParams(type, device_name).find(expected),
                  std::string::npos)
            << device_name << " type " << type << " expected " << expected;
    }
}

// Restores the process-wide host bias so other tests are unaffected.
struct HostBiasGuard
{
    bool saved = miopen_f8::get_hip_f8_ieee_exponent_bias();
    ~HostBiasGuard() { miopen_f8::set_hip_f8_ieee_exponent_bias(saved); }
};

// Compile options that the naive convolution solver passes to its kernel on a mock device.
void GetNaiveConvOptions(const std::string& device_name,
                         miopenDataType_t type,
                         std::string& options)
{
    const auto& devs = GetAllKnownDevices();
    const auto it    = std::find_if(
        devs.begin(), devs.end(), [&](const auto& dev) { return dev.second.name == device_name; });
    ASSERT_NE(it, devs.end()) << "no mock device " << device_name;

    auto handle    = MockHandle{it->second, false};
    const auto ctx = miopen::ExecutionContext{&handle};
    const auto problem =
        miopen::unit_tests::ConvTestCase{
            {1, 16, 14, 14}, {16, 16, 3, 3}, {1, 1}, {1, 1}, {1, 1}, type, type, type}
            .GetProblemDescription(miopen::conv::Direction::Forward);

    const auto solution = miopen::solver::conv::ConvDirectNaiveConvFwd{}.GetSolution(ctx, problem);
    ASSERT_EQ(solution.construction_params.size(), 1u);
    options = solution.construction_params[0].comp_options;
}

} // namespace

TEST(CPU_Fp8ExponentBias_NONE, KernelParamsUseDeviceBias)
{
    ExpectKernelBias("gfx942", 0);
    ExpectKernelBias("gfx950", 1);
    ExpectKernelBias("gfx1200", 1);
}

TEST(CPU_Fp8ExponentBias_NONE, HostBiasSwitchesInOneProcess)
{
    HostBiasGuard guard;

    // 1.0f encodes as 0x40 with the FNUZ bias and as 0x38 (FP8) or 0x3C (BF8) with the IEEE bias.
    const auto check = [](bool ieee) {
        EXPECT_EQ(miopen_f8::get_hip_f8_ieee_exponent_bias(), ieee);
        EXPECT_EQ(float8_fnuz(1.0f).data, ieee ? 0x38 : 0x40);
        EXPECT_EQ(bfloat8_fnuz(1.0f).data, ieee ? 0x3C : 0x40);
    };

    SetHostFp8ExponentBias<float8_fnuz>("gfx942");
    check(false);
    SetHostFp8ExponentBias<float8_fnuz>("gfx950");
    check(true);
    SetHostFp8ExponentBias<bfloat8_fnuz>("gfx942");
    check(false);
    SetHostFp8ExponentBias<bfloat8_fnuz>("gfx1200");
    check(true);
    SetHostFp8ExponentBias<float8_fnuz>("gfx942");
    check(false);
}

TEST(CPU_Fp8ExponentBias_NONE, NaiveConvCompileOptionsUseDeviceBias)
{
    const auto bias_define = [](int bias) {
        return "-DMIOPEN_FP8_IEEE_EXPONENT_BIAS=" + std::to_string(bias);
    };

    for(const auto type : {miopenFloat8_fnuz, miopenBFloat8_fnuz})
    {
        std::string options;
        GetNaiveConvOptions("gfx942", type, options);
        EXPECT_NE(options.find(bias_define(0)), std::string::npos)
            << "gfx942 type " << type << ": " << options;

        GetNaiveConvOptions("gfx950", type, options);
        EXPECT_NE(options.find(bias_define(1)), std::string::npos)
            << "gfx950 type " << type << ": " << options;
    }

    // Only FP8 problems carry the define.
    std::string options;
    GetNaiveConvOptions("gfx950", miopenHalf, options);
    EXPECT_EQ(options.find("MIOPEN_FP8_IEEE_EXPONENT_BIAS"), std::string::npos) << options;
}
