// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include "get_handle.hpp"
#include "gtest_common.hpp"
#include "pooling3d.hpp"
#include "../pooling_common.hpp"
#include <half/half.hpp>
#include <vector>
#include <limits>

namespace {

struct Pooling3dTestCase
{
    std::vector<int> in_shape;        // [N, C, D, H, W] for 3D
    std::vector<int> lens;            // [D, H, W] kernel sizes
    std::vector<int> strides;         // [D, H, W] strides
    std::vector<int> pads;            // [D, H, W] padding
    std::string mode;                 // "miopenPoolingMax", "miopenPoolingAverage", "miopenPoolingAverageInclusive"
    std::string index_type;           // "miopenIndexUint8", "miopenIndexUint16", "miopenIndexUint32", "miopenIndexUint64"
    int wsidx;                        // workspace index
    int verify_indices;               // whether to verify indices

    friend std::ostream& operator<<(std::ostream& os, const Pooling3dTestCase& tc)
    {
        return os << "in_shape: [" << tc.in_shape[0] << "," << tc.in_shape[1] << ","
                  << tc.in_shape[2] << "," << tc.in_shape[3] << "," << tc.in_shape[4]
                  << "] lens: [" << tc.lens[0] << "," << tc.lens[1] << "," << tc.lens[2]
                  << "] strides: [" << tc.strides[0] << "," << tc.strides[1] << "," << tc.strides[2]
                  << "] mode:" << tc.mode << " index_type:" << tc.index_type << " wsidx:" << tc.wsidx;
    }
};

template <typename T>
std::vector<Pooling3dTestCase> GetPooling3dTestCases()
{
    return {
        // in_shape, lens, strides, pads, mode, index_type, wsidx, verify_indices
        {{16, 64, 3, 4, 4}, {2, 2, 2}, {2, 2, 2}, {0, 0, 0}, "miopenPoolingMax", "miopenIndexUint8", 1, 1},
        {{16, 32, 4, 9, 9}, {3, 3, 3}, {1, 1, 1}, {1, 1, 1}, "miopenPoolingMax", "miopenIndexUint8", 1, 1},
    };
}

template <typename T>
void RunPooling3dTest(const Pooling3dTestCase& test_case)
{
    uint64_t max_value = miopen_type<T>{} == miopenHalf ? 5 : 17;

    // Create input tensor
    tensor<T> input{static_cast<size_t>(test_case.in_shape[0]),
                    static_cast<size_t>(test_case.in_shape[1]),
                    static_cast<size_t>(test_case.in_shape[2]),
                    static_cast<size_t>(test_case.in_shape[3]),
                    static_cast<size_t>(test_case.in_shape[4])};
    input.generate(tensor_elem_gen_integer{max_value});

    // Create pooling descriptor
    std::unordered_map<std::string, miopenPoolingMode_t> mode_lookup = {
        {"MAX", miopenPoolingMax},
        {"MIOPENPOOLINGMAX", miopenPoolingMax},
        {"AVERAGE", miopenPoolingAverage},
        {"MIOPENPOOLINGAVERAGE", miopenPoolingAverage},
        {"AVERAGEINCLUSIVE", miopenPoolingAverageInclusive},
        {"MIOPENPOOLINGAVERAGEINCLUSIVE", miopenPoolingAverageInclusive},
    };

    std::unordered_map<std::string, miopenIndexType_t> index_type_lookup = {
        {miopen::ToUpper("miopenIndexUint8"), miopenIndexUint8},
        {miopen::ToUpper("miopenIndexUint16"), miopenIndexUint16},
        {miopen::ToUpper("miopenIndexUint32"), miopenIndexUint32},
        {miopen::ToUpper("miopenIndexUint64"), miopenIndexUint64},
    };

    miopen::PoolingDescriptor filter{
        mode_lookup.at(miopen::ToUpper(test_case.mode)),
        miopenPaddingDefault,
        test_case.lens,
        test_case.strides,
        test_case.pads};

    auto idx_typ = index_type_lookup.at(miopen::ToUpper(test_case.index_type));
    filter.SetIndexType(idx_typ);
    filter.SetWorkspaceIndexMode(miopenPoolingWorkspaceIndexMode_t(test_case.wsidx));

    // Validate dimensions
    int spt_dim = test_case.in_shape.size() - 2;
    if(spt_dim != 3)
    {
        GTEST_SKIP() << "Invalid spatial dimensions, expected 3D pooling";
    }

    auto input_desc = miopen::TensorDescriptor(miopen_type<T>{}, test_case.in_shape);
    for(int i = 0; i < spt_dim; i++)
    {
        if(test_case.lens[i] > (input_desc.GetLengths()[i + 2] + static_cast<uint64_t>(2) * test_case.pads[i]))
        {
            GTEST_SKIP() << "Invalid kernel size for input dimensions";
        }
    }

    // Run forward pooling test
    std::vector<uint8_t> indices{};
    verify_forward_pooling<3> forward_verifier{};
    auto cpu_out = forward_verifier.cpu(input, filter, indices);
    std::vector<uint8_t> gpu_indices{};
    auto gpu_out = forward_verifier.gpu(input, filter, gpu_indices);

    // Compare forward results
    EXPECT_EQ(miopen::range_distance(cpu_out), miopen::range_distance(gpu_out));

    using value_type = T;
    const double tolerance = 80.0;
    const double threshold = std::numeric_limits<value_type>::epsilon() * tolerance;
    const double rms_error = miopen::rms_range(cpu_out, gpu_out);

    EXPECT_LE(rms_error, threshold)
        << "Forward pooling RMS error: " << rms_error << " exceeds threshold: " << threshold;

    if(rms_error > threshold)
    {
        const auto mxdiff = miopen::max_diff(cpu_out, gpu_out);
        std::cout << "Forward pooling max diff: " << mxdiff << std::endl;
    }

    // Run backward test
    auto dout = cpu_out;
    dout.generate(tensor_elem_gen_integer{2503});

    verify_backward_pooling<3> backward_verifier{};
    auto cpu_backward = backward_verifier.cpu(input, dout, cpu_out, filter, gpu_indices, test_case.wsidx != 0, static_cast<bool>(test_case.verify_indices));
    auto gpu_backward = backward_verifier.gpu(input, dout, cpu_out, filter, gpu_indices, test_case.wsidx != 0, static_cast<bool>(test_case.verify_indices));

    // Compare backward results
    EXPECT_EQ(miopen::range_distance(cpu_backward), miopen::range_distance(gpu_backward));

    const double backward_rms_error = miopen::rms_range(cpu_backward, gpu_backward);
    EXPECT_LE(backward_rms_error, threshold)
        << "Backward pooling RMS error: " << backward_rms_error << " exceeds threshold: " << threshold;

    if(backward_rms_error > threshold)
    {
        const auto mxdiff = miopen::max_diff(cpu_backward, gpu_backward);
        std::cout << "Backward pooling max diff: " << mxdiff << std::endl;
    }
}

} // namespace

class GPU_Pooling3d_FP32 : public testing::TestWithParam<Pooling3dTestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        if(!IsTestSupportedByDevice(Gpu::All))
        {
            GTEST_SKIP();
        }
    }
};

class GPU_Pooling3d_FP16 : public testing::TestWithParam<Pooling3dTestCase>
{
    void SetUp() override
    {
        prng::reset_seed();
        if(!IsTestSupportedByDevice(Gpu::All))
        {
            GTEST_SKIP();
        }
    }
};

TEST_P(GPU_Pooling3d_FP32, FloatTest_pooling3d)
{
    RunPooling3dTest<float>(GetParam());
}

TEST_P(GPU_Pooling3d_FP16, HalfTest_pooling3d)
{
    RunPooling3dTest<half_float::half>(GetParam());
}

INSTANTIATE_TEST_SUITE_P(Full, GPU_Pooling3d_FP32, testing::ValuesIn(GetPooling3dTestCases<float>()));

INSTANTIATE_TEST_SUITE_P(Full, GPU_Pooling3d_FP16, testing::ValuesIn(GetPooling3dTestCases<half_float::half>()));
