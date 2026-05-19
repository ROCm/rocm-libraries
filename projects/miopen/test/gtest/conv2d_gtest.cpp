// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "conv_common_gtest.hpp"
#include "gtest/gtest_common.hpp"
#include <gtest/gtest.h>
#include "conv2d_gtest.hpp"

namespace {

using TestCase = Conv2DBaseTestCase<>;

template <typename T>
auto GenCases(bool smoke_test)
{
    const bool cmdline_smoke_test = !conv2d_test_base<T>::has_commandline_arg("--all");

    if(cmdline_smoke_test != smoke_test)
    {
        MIOPEN_FRIENDLY_SKIP("Skipping " << (smoke_test ? "smoke" : "full")
                                         << " tests as per command line argument.");
    }

    const int limit_set =
        conv2d_test_base<T>::template get_commandline_arg_as_value<int>("--limit", 2);

    Conv2DBaseTestParameters<T> params(smoke_test, limit_set);

    if(conv2d_test_base<T>::has_commandline_arg("--disable-forward"))
    {
        params.base_params.do_forward = {false};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--disable-backward-data"))
    {
        params.base_params.do_backward_data = {false};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--disable-backward-weights"))
    {
        params.base_params.do_backward_weights = {false};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--enable-fdb"))
    {
        params.base_params.enable_fdb = {true};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--group-count"))
    {
        params.base_params.groupCount = {
            conv2d_test_base<T>::template get_commandline_arg_as_value<int>("--group-count", 1)};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--search"))
    {
        params.base_params.search = {
            conv2d_test_base<T>::template get_commandline_arg_as_value<int>(
                "--search", params.base_params.search[0])};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--generate-float"))
    {
        params.base_params.gen_float = {true};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--cmode"))
    {
        params.base_params.conv_mode = {
            conv2d_test_base<T>::template get_commandline_arg_as_value<std::string>("--cmode")};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--pmode"))
    {
        params.base_params.pad_mode = {
            conv2d_test_base<T>::template get_commandline_arg_as_value<std::string>("--pmode")};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--preallocate"))
    {
        params.base_params.preallocate = {true};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--deterministic"))
    {
        params.deterministic = {true};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--tensor_vect"))
    {
        params.tensor_vect = {conv2d_test_base<T>::template get_commandline_arg_as_value<size_t>(
            "--tensor_vect", params.tensor_vect[0])};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--vector_length"))
    {
        params.vector_length = {conv2d_test_base<T>::template get_commandline_arg_as_value<size_t>(
            "--vector_length", params.vector_length[0])};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--output_type"))
    {
        params.output_type = {
            conv2d_test_base<T>::template get_commandline_arg_as_value<std::string>(
                "--output_type", params.output_type[0])};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--int8_vectorize"))
    {
        params.int8_vectorize = {conv2d_test_base<T>::template get_commandline_arg_as_value<bool>(
            "--int8_vectorize", params.int8_vectorize[0])};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--in_layout"))
    {
        params.in_layout = {conv2d_test_base<T>::template get_commandline_arg_as_value<std::string>(
            "--in_layout", params.in_layout[0])};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--fil_layout"))
    {
        params.fil_layout = {
            conv2d_test_base<T>::template get_commandline_arg_as_value<std::string>(
                "--fil_layout", params.fil_layout[0])};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--out_layout"))
    {
        params.out_layout = {
            conv2d_test_base<T>::template get_commandline_arg_as_value<std::string>(
                "--out_layout", params.out_layout[0])};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--batch_size"))
    {
        params.batch_size = {conv2d_test_base<T>::template get_commandline_arg_as_value<size_t>(
            "--batch_size", params.batch_size[0])};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--input_channels"))
    {
        params.input_channels = {conv2d_test_base<T>::template get_commandline_arg_as_value<size_t>(
            "--input_channels", params.input_channels[0])};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--output_channels"))
    {
        params.output_channels = {
            conv2d_test_base<T>::template get_commandline_arg_as_value<size_t>(
                "--output_channels", params.output_channels[0])};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--spatial_dim_elements"))
    {
        params.spatial_dim_elements = {
            conv2d_test_base<T>::template get_commandline_arg_as_vector<size_t>(
                "--spatial_dim_elements", params.spatial_dim_elements[0])};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--filter_dims"))
    {
        params.filter_dims = {conv2d_test_base<T>::template get_commandline_arg_as_vector<size_t>(
            "--filter_dims", params.filter_dims[0])};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--pads_strides_dilations"))
    {
        params.pads_strides_dilations = {
            conv2d_test_base<T>::template get_commandline_arg_as_vector<int>(
                "--pads_strides_dilations", params.pads_strides_dilations[0])};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--trans_output_pads"))
    {
        params.trans_output_pads = {
            conv2d_test_base<T>::template get_commandline_arg_as_vector<int>(
                "--trans_output_pads", params.trans_output_pads[0])};
    }

    if(conv2d_test_base<T>::has_commandline_arg("--input"))
    {
        params.input_dims = {
            conv2d_test_base<T>::template get_commandline_arg_as_vector<size_t>("--input", {})};

        params.use_input_dims = true;
    }

    if(conv2d_test_base<T>::has_commandline_arg("--weights"))
    {
        params.weight_tensor_dims = {
            conv2d_test_base<T>::template get_commandline_arg_as_vector<size_t>("--weights", {})};

        params.use_weight_tensor_dims = true;
    }

    if(conv2d_test_base<T>::has_commandline_arg("--half"))
    {
        params.input_data_type = miopenHalf;
    }
    else if(conv2d_test_base<T>::has_commandline_arg("--float"))
    {
        params.input_data_type = miopenFloat;
    }
    else if(conv2d_test_base<T>::has_commandline_arg("--double"))
    {
        params.input_data_type = miopenDouble;
    }
    else if(conv2d_test_base<T>::has_commandline_arg("--int8"))
    {
        params.input_data_type = miopenInt8;
    }
    else if(conv2d_test_base<T>::has_commandline_arg("--bfloat16"))
    {
        params.input_data_type = miopenBFloat16;
    }

    if(conv2d_test_base<T>::has_commandline_arg("--tolerance"))
    {
        params.tolerance = conv2d_test_base<T>::template get_commandline_arg_as_value<double>(
            "--tolerance", params.tolerance);
    }

    return conv2d_test_base<T>::GenTestParams(params);
}

template <typename T>
auto GetCasesFull()
{
    static const auto cases = GenCases<T>(false);
    return cases;
}

template <typename T>
auto GetCasesSmoke()
{
    static const auto cases = GenCases<T>(true);
    return cases;
}

} // namespace

using GPU_Conv2d_FP16  = conv2d_test_base<half_float::half>;
using GPU_Conv2d_FP32  = conv2d_test_base<float>;
using GPU_Conv2d_FP64  = conv2d_test_base<double>;
using GPU_Conv2d_I8    = conv2d_test_base<int8_t>;
using GPU_Conv2d_BFP16 = conv2d_test_base<bfloat16>;

TEST_P(GPU_Conv2d_FP16, TestFloat16)
{
    GetTestParams();

    if(this->input_data_type != miopenHalf)
    {
        MIOPEN_FRIENDLY_SKIP("Test for half data type requested. Skipping current test.");
    }

    run();
}

TEST_P(GPU_Conv2d_FP32, TestFloat)
{
    GetTestParams();

    if(this->input_data_type != miopenFloat)
    {
        MIOPEN_FRIENDLY_SKIP("Test for float data type requested. Skipping current test.");
    }
    run();
}

TEST_P(GPU_Conv2d_FP64, TestFloat64)
{
    GetTestParams();

    if(this->input_data_type != miopenDouble)
    {
        MIOPEN_FRIENDLY_SKIP("Test for double data type requested. Skipping current test.");
    }
    run();
}

TEST_P(GPU_Conv2d_I8, TestInt8)
{
    GetTestParams();

    if(this->input_data_type != miopenInt8)
    {
        MIOPEN_FRIENDLY_SKIP("Test for int8 data type requested. Skipping current test.");
    }
    run();
}

TEST_P(GPU_Conv2d_BFP16, TestBFloat16)
{
    GetTestParams();

    if(this->input_data_type != miopenBFloat16)
    {
        MIOPEN_FRIENDLY_SKIP("Test for bfloat16 data type requested. Skipping current test.");
    }
    run();
}

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv2d_FP16,
                         GetCasesSmoke<half_float::half>(),
                         DefaultTestNameGenerator<TestCase>{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Conv2d_FP16,
                         GetCasesFull<half_float::half>(),
                         DefaultTestNameGenerator<TestCase>{});

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv2d_FP32,
                         GetCasesSmoke<float>(),
                         DefaultTestNameGenerator<TestCase>{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Conv2d_FP32,
                         GetCasesFull<float>(),
                         DefaultTestNameGenerator<TestCase>{});

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv2d_FP64,
                         GetCasesSmoke<double>(),
                         DefaultTestNameGenerator<TestCase>{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Conv2d_FP64,
                         GetCasesFull<double>(),
                         DefaultTestNameGenerator<TestCase>{});

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv2d_I8,
                         GetCasesSmoke<int8_t>(),
                         DefaultTestNameGenerator<TestCase>{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Conv2d_I8,
                         GetCasesFull<int8_t>(),
                         DefaultTestNameGenerator<TestCase>{});

INSTANTIATE_TEST_SUITE_P(Smoke,
                         GPU_Conv2d_BFP16,
                         GetCasesSmoke<bfloat16>(),
                         DefaultTestNameGenerator<TestCase>{});
INSTANTIATE_TEST_SUITE_P(Full,
                         GPU_Conv2d_BFP16,
                         GetCasesFull<bfloat16>(),
                         DefaultTestNameGenerator<TestCase>{});
