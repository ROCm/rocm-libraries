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
