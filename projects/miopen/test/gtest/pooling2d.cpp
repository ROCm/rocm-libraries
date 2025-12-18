// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "../pooling_common.hpp"
#include <half/half.hpp>
#include <vector>
#include <limits>

namespace {

struct Pooling2dTestCase
{
    std::vector<int> input_dims; // [N, C, H, W]
    std::vector<int> lens;       // [H, W]
    std::vector<int> pads;       // [H, W]
    std::vector<int> strides;    // [H, W]
    miopenIndexType_t index_type;
    miopenPoolingMode_t mode;
    int wsidx;

    friend std::ostream& operator<<(std::ostream& os, const Pooling2dTestCase& tc)
    {
        return os << "input_dims: [" << tc.input_dims[0] << "," << tc.input_dims[1] << ","
                  << tc.input_dims[2] << "," << tc.input_dims[3] << "] lens: [" << tc.lens[0]
                  << "," << tc.lens[1] << "] pads: [" << tc.pads[0] << "," << tc.pads[1]
                  << "] strides: [" << tc.strides[0] << "," << tc.strides[1]
                  << "] index_type: " << tc.index_type << ", mode: " << tc.mode
                  << ", wsidx: " << tc.wsidx;
    }
};

std::vector<Pooling2dTestCase> GetPooling2dTestCases()
{
    return {
        // Dataset 0: Default dataset (various tensor sizes)
        // input_dims, lens, pads, strides, index_type, mode, wsidx
        {{5, 32, 8, 8}, {2, 2}, {0, 0}, {2, 2}, miopenIndexUint8, miopenPoolingMax, 1},
        {{5, 32, 8, 8}, {3, 3}, {1, 1}, {1, 1}, miopenIndexUint8, miopenPoolingAverage, 1},
        {{10, 3, 32, 32}, {2, 2}, {0, 0}, {2, 2}, miopenIndexUint8, miopenPoolingMax, 0},
        {{10, 3, 32, 32}, {3, 3}, {1, 1}, {1, 1}, miopenIndexUint8, miopenPoolingAverage, 1},
        {{2, 64, 112, 112}, {2, 2}, {0, 0}, {2, 2}, miopenIndexUint8, miopenPoolingMax, 1},
        {{4, 3, 224, 224}, {3, 3}, {1, 1}, {1, 1}, miopenIndexUint8, miopenPoolingAverage, 0},

        // Dataset 1: Minimal dataset (asymmetric configs, small tensors)
        {{1, 4, 4, 4}, {2, 2}, {0, 0}, {1, 1}, miopenIndexUint8, miopenPoolingMax, 0},
        {{1, 4, 4, 4}, {2, 2}, {0, 0}, {2, 2}, miopenIndexUint8, miopenPoolingAverage, 1},
        {{1, 4, 4, 4}, {1, 2}, {0, 0}, {1, 1}, miopenIndexUint8, miopenPoolingMax, 1},
        {{1, 4, 4, 4}, {2, 1}, {0, 0}, {2, 1}, miopenIndexUint8, miopenPoolingAverage, 0},
        {{1, 4, 4, 4}, {2, 2}, {0, 0}, {1, 2}, miopenIndexUint8, miopenPoolingMax, 1},
        {{1, 4, 4, 4}, {2, 2}, {0, 0}, {2, 1}, miopenIndexUint8, miopenPoolingAverage, 0},

        // Additional coverage: different index types and modes
        {{5, 32, 8, 8}, {2, 2}, {0, 0}, {2, 2}, miopenIndexUint32, miopenPoolingMax, 1},
        {{5, 32, 8, 8},
         {3, 3},
         {1, 1},
         {1, 1},
         miopenIndexUint32,
         miopenPoolingAverageInclusive,
         0},
    };
}

template <typename T, typename Index>
void RunPooling2dTestWithIndexType(const Pooling2dTestCase& test_case)
{
    // Create input tensor
    tensor<T> input{static_cast<size_t>(test_case.input_dims[0]),
                    static_cast<size_t>(test_case.input_dims[1]),
                    static_cast<size_t>(test_case.input_dims[2]),
                    static_cast<size_t>(test_case.input_dims[3])};
    input.generate(tensor_elem_gen_integer{miopen_type<T>{} == miopenHalf ? 5 : 17});

    // Setup pooling descriptor
    miopen::PoolingDescriptor filter{
        test_case.mode, miopenPaddingDefault, test_case.lens, test_case.strides, test_case.pads};
    filter.SetIndexType(test_case.index_type);
    filter.SetWorkspaceIndexMode(miopenPoolingWorkspaceIndexMode_t(test_case.wsidx));

    // Validate dimensions
    auto input_desc = miopen::TensorDescriptor(miopen_type<T>{}, test_case.input_dims);
    int spt_dim     = static_cast<int>(test_case.input_dims.size()) - 2;

    if(spt_dim != 2)
    {
        GTEST_SKIP() << "Only 2D pooling is supported (spt_dim == 2)";
    }

    for(int i = 0; i < spt_dim; i++)
    {
        if(test_case.lens[i] >
           (input_desc.GetLengths()[i + 2] + static_cast<uint64_t>(2) * test_case.pads[i]))
        {
            GTEST_SKIP() << "Invalid config: lens[" << i << "] > (input_dims[" << i + 2
                         << "] + 2 * pads[" << i << "])";
        }
    }

    // Run forward pooling
    std::vector<Index> indices;
    verify_forward_pooling<2> forward_verifier;
    auto forward_result     = forward_verifier.cpu(input, filter, indices);
    auto forward_gpu_result = forward_verifier.gpu(input, filter, indices);

    // Compare forward results
    EXPECT_EQ(miopen::range_distance(forward_result), miopen::range_distance(forward_gpu_result));

    using value_type               = T;
    const double tolerance         = 80.0;
    const double threshold         = std::numeric_limits<value_type>::epsilon() * tolerance;
    const double forward_rms_error = miopen::rms_range(forward_result, forward_gpu_result);

    EXPECT_LE(forward_rms_error, threshold)
        << "Forward RMS error: " << forward_rms_error << " exceeds threshold: " << threshold;

    // Run backward pooling
    auto dout = forward_result;
    dout.generate(tensor_elem_gen_integer{2503});

    verify_backward_pooling<2> backward_verifier;
    auto backward_result = backward_verifier.cpu(
        input, dout, forward_result, filter, indices, test_case.wsidx != 0, true);
    auto backward_gpu_result = backward_verifier.gpu(
        input, dout, forward_result, filter, indices, test_case.wsidx != 0, true);

    // Compare backward results
    EXPECT_EQ(miopen::range_distance(backward_result), miopen::range_distance(backward_gpu_result));

    const double backward_rms_error = miopen::rms_range(backward_result, backward_gpu_result);

    EXPECT_LE(backward_rms_error, threshold)
        << "Backward RMS error: " << backward_rms_error << " exceeds threshold: " << threshold;
}

template <typename T>
void RunPooling2dTest(const Pooling2dTestCase& test_case)
{
    // Dispatch to the appropriate index type template
    switch(test_case.index_type)
    {
    case miopenIndexUint8: {
        RunPooling2dTestWithIndexType<T, uint8_t>(test_case);
        break;
    }
    case miopenIndexUint16: {
        RunPooling2dTestWithIndexType<T, uint16_t>(test_case);
        break;
    }
    case miopenIndexUint32: {
        RunPooling2dTestWithIndexType<T, uint32_t>(test_case);
        break;
    }
    case miopenIndexUint64: {
        RunPooling2dTestWithIndexType<T, uint64_t>(test_case);
        break;
    }
    default: {
        GTEST_FAIL() << "Unsupported index type: " << test_case.index_type;
        break;
    }
    }
}

} // namespace

class GPU_Pooling2d_FP32 : public testing::TestWithParam<Pooling2dTestCase>
{
    void SetUp() override { prng::reset_seed(); }
};

class GPU_Pooling2d_FP16 : public testing::TestWithParam<Pooling2dTestCase>
{
    void SetUp() override { prng::reset_seed(); }
};

TEST_P(GPU_Pooling2d_FP32, FloatTest_pooling2d) { RunPooling2dTest<float>(GetParam()); }

TEST_P(GPU_Pooling2d_FP16, HalfTest_pooling2d) { RunPooling2dTest<half_float::half>(GetParam()); }

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Pooling2d_FP32, testing::ValuesIn(GetPooling2dTestCases()));

INSTANTIATE_TEST_SUITE_P(Smoke, GPU_Pooling2d_FP16, testing::ValuesIn(GetPooling2dTestCases()));
