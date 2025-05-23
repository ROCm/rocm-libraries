/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#include <tuple>
#include <miopen/miopen.h>
#include <gtest/gtest.h>
#include <gtest/gtest_common.hpp>
#include "../conv2d.hpp"
#include "get_handle.hpp"
#include "lib_env_var.hpp"

namespace conv_igemm_dynamic_xdlops_nhwc_bf16 {

void GetArgs(const std::string& param, std::vector<std::string>& tokens)
{
    std::stringstream ss(param);
    std::istream_iterator<std::string> begin(ss);
    std::istream_iterator<std::string> end;
    while(begin != end)
        tokens.push_back(*begin++);
}

class GPU_Conv2d_BFP16 : public testing::TestWithParam<std::vector<std::string>>
{
};

void Run2dDriver(miopenDataType_t prec)
{

    std::vector<std::string> params;
    switch(prec)
    {
    case miopenBFloat16: params = GPU_Conv2d_BFP16::GetParam(); break;
    case miopenFloat:
    case miopenHalf:
    case miopenInt8:
    case miopenInt32:
    case miopenInt64:
    case miopenDouble:
    case miopenFloat8_fnuz:
    case miopenBFloat8_fnuz:
        FAIL() << "miopenFloat, miopenHalf, miopenInt8, miopenInt32, "
                  "miopenDouble, miopenFloat8_fnuz, miopenBFloat8_fnuz "
                  "data type not supported by conv_igemm_dynamic_xdlops_nhwc_bf16 test";

    default: params = GPU_Conv2d_BFP16::GetParam();
    }

    ScopedEnvironment<std::string> find_mode_env1(MIOPEN_FIND_MODE, "normal");
    ScopedEnvironment<std::string> find_only_solver_env(
        MIOPEN_DEBUG_FIND_ONLY_SOLVER,
        "ConvAsmImplicitGemmGTCDynamicFwdXdlopsNHWC;ConvAsmImplicitGemmGTCDynamicBwdXdlopsNHWC;"
        "ConvAsmImplicitGemmGTCDynamicWrwXdlopsNHWC");

    for(const auto& test_value : params)
    {
        std::vector<std::string> tokens;
        GetArgs(test_value, tokens);
        std::vector<const char*> ptrs;

        std::transform(tokens.begin(),
                       tokens.end(),
                       std::back_inserter(ptrs),
                       [](const std::string& str) { return str.data(); });

        testing::internal::CaptureStderr();
        test_drive<conv2d_driver>(ptrs.size(), ptrs.data());
        auto capture = testing::internal::GetCapturedStderr();
        std::cout << capture;
    }
};

bool IsTestSupportedForDevice(const miopen::Handle& handle)
{
    const auto& target  = handle.GetTargetProperties();
    std::string devName = handle.GetDeviceName();
    if(target.Xnack() && *target.Xnack())
        return false;

    if(devName == "gfx90a" || devName == "gfx942")
        return true;
    else
        return false;
}

std::vector<std::string> GetTestCases(const std::string& precision)
{
    const std::string flags         = "test_conv2d " + precision + " --verbose ";
    const std::string dis_bk_data   = " --disable-backward-data";
    const std::string dis_bk_wei    = " --disable-backward-weights";
    const std::string dis_fwd       = " --disable-forward";
    const std::string dis_vali      = " --disable-validation";
    const std::string in_nhwc       = " --in_layout NHWC";
    const std::string fil_nhwc      = " --fil_layout NHWC";
    const std::string out_nhwc      = " --out_layout NHWC";
    const std::string args_nhwc_fwd = dis_bk_data + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc;
    const std::string args_nhwc_bwd = dis_fwd + dis_bk_wei + in_nhwc + fil_nhwc + out_nhwc;
    const std::string args_nhwc_wrw = dis_fwd + dis_bk_data + in_nhwc + fil_nhwc + out_nhwc;

    const std::vector<std::string> test_cases = {
        // clang-format off
    //fwd
    {flags + " --input  64 256  7  7 --weights 128 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + args_nhwc_fwd},
    {flags + " --input  32 160 73 73 --weights  64 160 1 1 --pads_strides_dilations 0 0 1 1 1 1" + args_nhwc_fwd},
    {flags + " --input  16  64 56 56 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + args_nhwc_fwd},
    {flags + " --input   2 256 40 52 --weights 256 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + args_nhwc_fwd},
    {flags + " --input   2  64 59 57 --weights  12  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + args_nhwc_fwd},
    {flags + " --input  32 128 14 14 --weights  64 128 1 1 --pads_strides_dilations 0 0 2 2 1 1" + args_nhwc_fwd},
    {flags + " --input  64  64 17 17 --weights 192  64 1 7 --pads_strides_dilations 0 3 1 1 1 1" + args_nhwc_fwd},
    {flags + " --input  64  64 17 17 --weights 192  64 7 1 --pads_strides_dilations 3 0 1 1 1 1" + args_nhwc_fwd},
    {flags + " --input   4 128 28 28 --weights 128 128 2 2 --pads_strides_dilations 0 0 2 2 1 1" + args_nhwc_fwd},
    {flags + " --input  32 128  8  8 --weights 192 128 3 1 --pads_strides_dilations 1 0 1 1 1 1" + args_nhwc_fwd},
    {flags + " --input  64 192 17 17 --weights 160 192 3 3 --pads_strides_dilations 0 0 2 2 1 1" + args_nhwc_fwd},
    {flags + " --input  64  32 73 73 --weights  64  32 3 3 --pads_strides_dilations 1 1 1 1 1 1" + args_nhwc_fwd},
    {flags + " --input  16  64 56 56 --weights  64  64 3 3 --pads_strides_dilations 1 1 1 1 1 1" + args_nhwc_fwd},
    {flags + " --input  64   3 78 78 --weights  64   3 7 7 --pads_strides_dilations 0 0 2 2 1 1" + args_nhwc_fwd},
    {flags + " --input  16 192 17 17 --weights 224 192 1 7 --pads_strides_dilations 0 3 1 1 1 1" + args_nhwc_fwd},
    {flags + " --input  16   3 17 17 --weights  64   3 1 1 --pads_strides_dilations 0 0 1 1 1 1" + args_nhwc_fwd},

    //nhwc_bwd
    {flags + " --input  64 256  7  7 --weights 128 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + args_nhwc_bwd},
    {flags + " --input  32 160 73 73 --weights  64 160 1 1 --pads_strides_dilations 0 0 1 1 1 1" + args_nhwc_bwd},
    {flags + " --input  16  64 56 56 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + args_nhwc_bwd},
    {flags + " --input   2 256 40 52 --weights 256 256 1 1 --pads_strides_dilations 0 0 1 1 1 1" + args_nhwc_bwd},
    {flags + " --input   2  64 32 28 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1" + args_nhwc_bwd},
    {flags + " --input  32 128 14 14 --weights  64 128 1 1 --pads_strides_dilations 0 0 2 2 1 1" + args_nhwc_bwd},
    {flags + " --input  64  64 17 17 --weights 192  64 1 7 --pads_strides_dilations 0 3 1 1 1 1" + args_nhwc_bwd},
    {flags + " --input  64  64 17 17 --weights 192  64 7 1 --pads_strides_dilations 3 0 1 1 1 1" + args_nhwc_bwd},
    {flags + " --input   4 128 28 28 --weights 128 128 2 2 --pads_strides_dilations 0 0 2 2 1 1" + args_nhwc_bwd},
    {flags + " --input  32 128  8  8 --weights 192 128 3 1 --pads_strides_dilations 1 0 1 1 1 1" + args_nhwc_bwd},
    {flags + " --input  64 192 17 17 --weights 160 192 3 3 --pads_strides_dilations 0 0 2 2 1 1" + args_nhwc_bwd},
    {flags + " --input  64  32 73 73 --weights  64  32 3 3 --pads_strides_dilations 1 1 1 1 1 1" + args_nhwc_bwd},
    {flags + " --input  16  64 56 56 --weights  64  64 3 3 --pads_strides_dilations 1 1 1 1 1 1" + args_nhwc_bwd},
    {flags + " --input  16  16 25 25 --weights  64  16 3 3 --pads_strides_dilations 0 0 1 1 1 1" + args_nhwc_bwd},
    {flags + " --input  15 256 1  1  --weights 340 256 3 3 --pads_strides_dilations 1 1 1 1 1 1" + args_nhwc_bwd},
    {flags + " --input  15 128 10 10 --weights 340 128 3 3 --pads_strides_dilations 1 1 1 1 1 1" + args_nhwc_bwd},

    //nhwc_wrw
    {flags + " --input  64 256  7  7 --weights 128 256 1 1 --pads_strides_dilations 0 0 1 1 1 1 " + args_nhwc_wrw},
    {flags + " --input  32 160 73 73 --weights  64 160 1 1 --pads_strides_dilations 0 0 1 1 1 1 " + args_nhwc_wrw},
    {flags + " --input  16  64 56 56 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1 " + args_nhwc_wrw},
    {flags + " --input   2 256 40 52 --weights 256 256 1 1 --pads_strides_dilations 0 0 1 1 1 1 " + args_nhwc_wrw},
    {flags + " --input   2  64 32 28 --weights  64  64 1 1 --pads_strides_dilations 0 0 1 1 1 1 " + args_nhwc_wrw},
    {flags + " --input  32 128 14 14 --weights  64 128 1 1 --pads_strides_dilations 0 0 2 2 1 1 " + args_nhwc_wrw},
    {flags + " --input  64  64 17 17 --weights 192  64 1 7 --pads_strides_dilations 0 3 1 1 1 1 " + args_nhwc_wrw},
    {flags + " --input  64  64 17 17 --weights 192  64 7 1 --pads_strides_dilations 3 0 1 1 1 1 " + args_nhwc_wrw},
    {flags + " --input   4 128 28 28 --weights 128 128 2 2 --pads_strides_dilations 0 0 2 2 1 1 " + args_nhwc_wrw},
    {flags + " --input  32 128  8  8 --weights 192 128 3 1 --pads_strides_dilations 1 0 1 1 1 1 " + args_nhwc_wrw},
    {flags + " --input  64 192 17 17 --weights 160 192 3 3 --pads_strides_dilations 0 0 2 2 1 1 " + args_nhwc_wrw},
    {flags + " --input  64  32 73 73 --weights  64  32 3 3 --pads_strides_dilations 1 1 1 1 1 1 " + args_nhwc_wrw},
    {flags + " --input  16  64 56 56 --weights  64  64 3 3 --pads_strides_dilations 1 1 1 1 1 1 " + args_nhwc_wrw},
    {flags + " --input  16  16 25 25 --weights  64  16 3 3 --pads_strides_dilations 0 0 1 1 1 1 " + args_nhwc_wrw},
    {flags + " --input  4 32 79 141 --weights 64 32 5 10 --pads_strides_dilations 0 0 2 2 1 1 " + args_nhwc_wrw},
    {flags + " --input  400  256 7 7 --weights 1024  256  7 7 --pads_strides_dilations 0 0 1 1 1 1 " + args_nhwc_wrw},
    {flags + " --input  400  256 1 1 --weights 1024  256  1 1 --pads_strides_dilations 0 0 1 1 1 1 " + args_nhwc_wrw},
    {flags + " --input  1 3 32 32 --weights 1 3 11 11 --pads_strides_dilations 1 1 2 2 2 1 " + args_nhwc_wrw},
    {flags + " --input  1 3 224 224 --weights 1 3 3 3 --pads_strides_dilations 0 0 1 1 2 2 " + args_nhwc_wrw},
    {flags + " --input  1 1 8 8 --weights 1 1 2 2 --pads_strides_dilations 0 0 1 1 2 2 " + args_nhwc_wrw},
    {flags + " --input  1 128 56 56 --weights 1 128 5 5 --pads_strides_dilations 0 0 2 2 1 1 " + args_nhwc_wrw}
        // clang-format on
    };
    return test_cases;
}

} // namespace conv_igemm_dynamic_xdlops_nhwc_bf16
using namespace conv_igemm_dynamic_xdlops_nhwc_bf16;

TEST_P(GPU_Conv2d_BFP16, Bf16Test_conv_igemm_dynamic_xdlops_nhwc_bf16)
{
    const auto& handle = get_handle();
    if(IsTestSupportedForDevice(handle))
    {
        Run2dDriver(miopenBFloat16);
    }
    else
    {
        GTEST_SKIP();
    }
};

INSTANTIATE_TEST_SUITE_P(Full, GPU_Conv2d_BFP16, testing::Values(GetTestCases("--bfloat16")));
