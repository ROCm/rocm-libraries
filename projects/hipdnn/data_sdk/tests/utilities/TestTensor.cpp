// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cmath>
#include <gtest/gtest.h>

#include <hip/hip_runtime.h>
#include <hipdnn_data_sdk/types.hpp>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <hipdnn_test_sdk/utilities/TestUtilities.hpp>
#include <numeric>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

using namespace hipdnn_data_sdk::utilities;
using namespace hipdnn_data_sdk::types;

TEST(TestTensor, Swap)
{
    Tensor<float> tensorA({2, 2});
    Tensor<float> tensorB{{2}};
    tensorA.fillWithValue(1.f);
    tensorB.fillWithValue(2.f);

    std::swap(tensorA, tensorB);

    ASSERT_EQ(tensorA.dims().size(), 1);
    ASSERT_EQ(tensorB.dims().size(), 2);
    ASSERT_EQ(tensorA.memory().hostData()[0], 2.f);
    ASSERT_EQ(tensorB.memory().hostData()[0], 1.f);
}

TEST(TestTensor, BasicRowMajorUsage)
{
    Tensor<float> tensor({1, 2, 3, 4});

    // NCHW (row-major) strides with dims {N=1, C=2, H=3, W=4}:
    // N stride = C*H*W = 2*3*4 = 24
    // C stride = H*W = 3*4 = 12
    // H stride = W = 4
    // W stride = 1 (innermost dimension)
    EXPECT_EQ(tensor.memory().count(), 24);
    EXPECT_EQ(tensor.strides()[0], 24);
    EXPECT_EQ(tensor.strides()[1], 12);
    EXPECT_EQ(tensor.strides()[2], 4);
    EXPECT_EQ(tensor.strides()[3], 1);
}

TEST(TestTensor, FillWithValuesPacked)
{
    Tensor<float> tensor({1, 2, 3, 4});

    tensor.fillWithValue(1.0f);
    auto buffer = tensor.memory().hostData();

    for(size_t i = 0; i < tensor.memory().count(); i++)
    {
        EXPECT_FLOAT_EQ(1.0f, buffer[i]);
    }
}

TEST(TestTensor, FillWithValuesNonPacked)
{
    Tensor<float> tensor({1, 2, 3, 4}, {30, 15, 5, 1});

    tensor.fillWithValue(1.0f);

    for(auto it{tensor.cbegin()}; it != tensor.cend(); ++it)
    {
        EXPECT_FLOAT_EQ(1.0f, (*reinterpret_cast<const float*>((*it))));
    }
}

TEST(TestTensor, FillWithRandomValuesPacked)
{
    Tensor<float> tensor({1, 2, 3, 4});

    tensor.fillWithRandomValues(1.0f, 3.0f);
    auto buffer = tensor.memory().hostData();

    for(size_t i = 0; i < tensor.memory().count(); i++)
    {
        EXPECT_GE(buffer[i], 1.0f);
        EXPECT_LE(buffer[i], 3.0f);
    }
}

TEST(TestTensor, FillWithRandomValuesNonPacked)
{
    Tensor<float> tensor({1, 2, 3, 4}, {30, 15, 5, 1});

    tensor.fillWithRandomValues(1.0f, 3.0f);

    for(auto it{tensor.cbegin()}; it != tensor.cend(); ++it)
    {
        auto val{(*reinterpret_cast<const float*>((*it)))};
        EXPECT_GE(val, 1.0f);
        EXPECT_LE(val, 3.0f);
    }
}

TEST(TestTensor, FillWithValuesHostGeneratorPacked)
{
    Tensor<float> tensor({1, 2, 3, 4});
    struct UniformCpuGenerator
    {
        explicit UniformCpuGenerator(float min, float max, unsigned int seed)
            : _min(min)
            , _max(max)
            , _seed(seed)
        {
        }

        void operator()(float* data, size_t count) const
        {
            std::mt19937 rng(_seed);
            std::uniform_real_distribution<float> dist(_min, _max);

            for(size_t i = 0; i < count; ++i)
            {
                data[i] = static_cast<float>(dist(rng));
            }
        }

    private:
        float _min;
        float _max;
        unsigned int _seed;
    };

    const float min = 7.0f;
    const float max = 10.0f;
    tensor.fillWithValues(UniformCpuGenerator(min, max, std::random_device{}()), true);

    for(auto it{tensor.cbegin()}; it != tensor.cend(); ++it)
    {
        auto val{(*static_cast<const float*>((*it)))};
        EXPECT_GE(val, min);
        EXPECT_LE(val, max);
    }
}

TEST(TestTensor, FillWithValuesHostGeneratorNonPacked)
{
    Tensor<float> tensor({1, 2, 3, 4}, {30, 15, 5, 1});
    struct UniformCpuGenerator
    {
        explicit UniformCpuGenerator(float min, float max, unsigned int seed)
            : _min(min)
            , _max(max)
            , _seed(seed)
        {
        }

        void operator()(float* data, size_t count) const
        {
            std::mt19937 rng(_seed);
            std::uniform_real_distribution<float> dist(_min, _max);

            for(size_t i = 0; i < count; ++i)
            {
                data[i] = static_cast<float>(dist(rng));
            }
        }

    private:
        float _min;
        float _max;
        unsigned int _seed;
    };

    const float min = -10.0f;
    const float max = -7.0f;
    tensor.fillWithValues(UniformCpuGenerator(min, max, std::random_device{}()), true);

    for(auto it{tensor.cbegin()}; it != tensor.cend(); ++it)
    {
        auto val{(*static_cast<const float*>((*it)))};
        EXPECT_GE(val, min);
        EXPECT_LE(val, max);
    }
}

TEST(TestTensor, FillWithValuesDeviceGeneratorPacked)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> tensor({1, 2, 3, 4});

    struct DeviceGpuGenerator
    {
        void operator()(float* data, size_t count) const
        {
            std::vector<float> writeData(count);
            std::iota(writeData.begin(), writeData.end(), 0.0f);

            auto err = hipMemcpyWithStream(
                data, writeData.data(), count * sizeof(float), hipMemcpyHostToDevice, nullptr);
            if(err != hipSuccess)
            {
                throw std::runtime_error("hipMemcpyWithStream failed");
            }
        }
    };

    tensor.fillWithValues(DeviceGpuGenerator(), false);

    auto hostData = static_cast<float*>(tensor.rawHostData());
    for(size_t i = 0; i < tensor.elementCount(); i++)
    {
        EXPECT_EQ(hostData[i], static_cast<float>(i));
    }
}

TEST(TestTensor, FillWithValuesDeviceGeneratorNonPacked)
{
    SKIP_IF_NO_DEVICES();

    Tensor<float> tensor({1, 2, 3, 4}, {30, 15, 5, 1});

    struct DeviceGpuGenerator
    {
        void operator()(float* data, size_t count) const
        {
            std::vector<float> writeData(count);
            std::iota(writeData.begin(), writeData.end(), 0.0f);

            auto err = hipMemcpyWithStream(
                data, writeData.data(), count * sizeof(float), hipMemcpyHostToDevice, nullptr);
            if(err != hipSuccess)
            {
                throw std::runtime_error("hipMemcpyWithStream failed");
            }
        }
    };

    tensor.fillWithValues(DeviceGpuGenerator(), false);

    auto hostData = static_cast<float*>(tensor.rawHostData());
    for(size_t i = 0; i < tensor.elementCount(); i++)
    {
        EXPECT_EQ(hostData[i], static_cast<float>(i));
    }
}

TEST(TestTensor, FillWithValuesInvalidGeneratorThrows)
{
    Tensor<float> tensor({1, 2, 3, 4});
    const ValueGenerator<float> invalidGenerator;

    EXPECT_THROW(tensor.fillWithValues(invalidGenerator, false), std::invalid_argument);
    EXPECT_THROW(tensor.fillWithValues(invalidGenerator, true), std::invalid_argument);
}

TEST(TestTensor, BasicNclUsage)
{
    Tensor<float> tensor({1, 2, 3}, TensorLayout::NCL);

    // NCL (row-major/channel-first) strides with dims {N=1, C=2, L=3}:
    // N stride = C*L = 2*3 = 6
    // C stride = L = 3
    // L stride = 1 (innermost dimension)
    EXPECT_EQ(tensor.memory().count(), 6);
    EXPECT_EQ(tensor.strides()[0], 6);
    EXPECT_EQ(tensor.strides()[1], 3);
    EXPECT_EQ(tensor.strides()[2], 1);
}

TEST(TestTensor, BasicNlcUsage)
{
    Tensor<float> tensor({1, 2, 3}, TensorLayout::NLC);

    // NLC (channel-last) strides with dims {N=1, C=2, L=3}:
    // N stride = L*C = 3*2 = 6
    // C stride = 1 (innermost dimension)
    // L stride = C = 2
    EXPECT_EQ(tensor.memory().count(), 6);
    EXPECT_EQ(tensor.strides()[0], 6);
    EXPECT_EQ(tensor.strides()[1], 1);
    EXPECT_EQ(tensor.strides()[2], 2);
}

TEST(TestTensor, BasicNhwcUsage)
{
    Tensor<float> tensor({1, 2, 3, 4}, TensorLayout::NHWC);

    EXPECT_EQ(tensor.memory().count(), 24);
    // NHWC strides with dims {N=1, C=2, H=3, W=4}:
    // N stride = H*W*C = 3*4*2 = 24
    // C stride = 1 (innermost dimension)
    // H stride = W*C = 4*2 = 8
    // W stride = C = 2
    EXPECT_EQ(tensor.strides()[0], 24);
    EXPECT_EQ(tensor.strides()[1], 1);
    EXPECT_EQ(tensor.strides()[2], 8);
    EXPECT_EQ(tensor.strides()[3], 2);
}

TEST(TestTensor, GetAndSetHostValueNcl)
{
    Tensor<float> tensor({2, 3, 4}, TensorLayout::NCL);
    tensor.fillWithValue(0.0f);
    tensor.setHostValue(99.0f, 1, 2, 3);

    EXPECT_FLOAT_EQ(tensor.getHostValue(1, 2, 3), 99.0f);
}

TEST(TestTensor, GetAndSetHostValueNlc)
{
    Tensor<float> tensor({2, 3, 4}, TensorLayout::NLC);
    tensor.fillWithValue(0.0f);
    tensor.setHostValue(99.0f, 1, 2, 3);

    EXPECT_FLOAT_EQ(tensor.getHostValue(1, 2, 3), 99.0f);
}

TEST(TestTensor, GetAndSetHostValueNchw)
{
    Tensor<float> tensor({1, 2, 3, 4});
    tensor.fillWithValue(0.0f);
    tensor.setHostValue(99.0f, 0, 1, 1, 2);

    EXPECT_FLOAT_EQ(tensor.getHostValue(0, 1, 1, 2), 99.0f);
}

TEST(TestTensor, GetAndSetHostValueNhwc)
{
    Tensor<float> tensor({1, 2, 3, 4}, TensorLayout::NHWC);
    tensor.fillWithValue(0.0f);
    tensor.setHostValue(99.0f, 0, 1, 1, 2);

    EXPECT_FLOAT_EQ(tensor.getHostValue(0, 1, 1, 2), 99.0f);
}

TEST(TestTensor, ExceedDimensions)
{
    Tensor<float> tensor({1, 2, 3, 4});
    tensor.fillWithValue(0.0f);

    EXPECT_THROW(tensor.setHostValue(99.0f, 0, 1, 1, 2, 3), std::invalid_argument);
}

TEST(TestTensor, GetIndex)
{
    //Strides {60, 20, 5, 1}
    const Tensor<float> tensor({2, 3, 4, 5});

    // 1*60 + 2*20 + 3*5 + 4*1 = 119
    EXPECT_EQ(tensor.getIndex(1, 2, 3, 4), 119);
    const std::vector<int64_t> indices1 = {1, 2, 3, 4};
    EXPECT_EQ(tensor.getIndex(indices1), 119);

    // Test first element
    EXPECT_EQ(tensor.getIndex(0, 0, 0, 0), 0);
    const std::vector<int64_t> indices2 = {0, 0, 0, 0};
    EXPECT_EQ(tensor.getIndex(indices2), 0);

    // Test partial indices
    EXPECT_EQ(tensor.getIndex(1, 0), 60);
    const std::vector<int64_t> indices3 = {1, 0};
    EXPECT_EQ(tensor.getIndex(indices3), 60);

    EXPECT_EQ(tensor.getIndex(0, 1), 20);
    const std::vector<int64_t> indices4 = {0, 1};
    EXPECT_EQ(tensor.getIndex(indices4), 20);
}

TEST(TestTensor, GetIndexNcl)
{
    // Strides {12, 4, 1}
    const Tensor<float> tensor({2, 3, 4}, TensorLayout::NCL);

    // 1*12 + 2*4 + 3*1 = 23
    EXPECT_EQ(tensor.getIndex(1, 2, 3), 23);
    const std::vector<int64_t> indices = {1, 2, 3};
    EXPECT_EQ(tensor.getIndex(indices), 23);
}

TEST(TestTensor, GetIndexNlc)
{
    // Strides {12, 1, 3}
    const Tensor<float> tensor({2, 3, 4}, TensorLayout::NLC);

    // 1*12 + 2*1 + 3*3 = 23
    EXPECT_EQ(tensor.getIndex(1, 2, 3), 23);
    const std::vector<int64_t> indices = {1, 2, 3};
    EXPECT_EQ(tensor.getIndex(indices), 23);
}

TEST(TestTensor, GetIndexNhwc)
{
    //Strides {60, 1, 15, 3}
    const Tensor<float> tensor({2, 3, 4, 5}, TensorLayout::NHWC);

    // 1*60 + 2*1 + 3*15 + 4*3 = 119
    EXPECT_EQ(tensor.getIndex(1, 2, 3, 4), 119);
    const std::vector<int64_t> indices = {1, 2, 3, 4};
    EXPECT_EQ(tensor.getIndex(indices), 119);
}

TEST(TestTensor, GetIndexEdgeCases)
{
    const Tensor<float> tensor({10, 20});

    // Last element
    EXPECT_EQ(tensor.getIndex(9, 19), 199);

    std::vector<int64_t> indices = {9, 19};
    EXPECT_EQ(tensor.getIndex(indices), 199);

    // Single dim
    EXPECT_EQ(tensor.getIndex(5), 100);
    indices = {5};
    EXPECT_EQ(tensor.getIndex(indices), 100);

    // Empty dim
    const std::vector<int64_t> emptyIndices;
    EXPECT_EQ(tensor.getIndex(emptyIndices), 0);
}

TEST(TestTensor, GetIndexInvalidCases)
{
    const Tensor<float> tensor({2, 3, 4});

    EXPECT_THROW(tensor.getIndex(0, 1, 2, 3), std::invalid_argument);

    const std::vector<int64_t> tooManyIndices = {0, 1, 2, 3};
    EXPECT_THROW(tensor.getIndex(tooManyIndices), std::invalid_argument);
}

TEST(TestTensor, GetHostValueWithVector)
{
    Tensor<float> tensor({2, 3, 4});
    tensor.fillWithValue(0.0f);

    tensor.setHostValue(42.0f, 1, 2, 3);
    const std::vector<int64_t> indices2 = {1, 2, 3};
    EXPECT_FLOAT_EQ(tensor.getHostValue(indices2), 42.0f);
}

TEST(TestTensor, SetHostValueWithVector)
{
    Tensor<float> tensor({2, 3, 4});
    tensor.fillWithValue(0.0f);

    const std::vector<int64_t> indices1 = {0, 1, 2};
    tensor.setHostValue(10.0f, indices1);

    const std::vector<int64_t> indices2 = {1, 0, 3};
    tensor.setHostValue(20.0f, indices2);

    const std::vector<int64_t> indices3 = {1, 2, 1};
    tensor.setHostValue(30.0f, indices3);

    EXPECT_FLOAT_EQ(tensor.getHostValue(0, 1, 2), 10.0f);
    EXPECT_FLOAT_EQ(tensor.getHostValue(1, 0, 3), 20.0f);
    EXPECT_FLOAT_EQ(tensor.getHostValue(1, 2, 1), 30.0f);
}

TEST(TestTensor, FillWithData)
{
    std::vector<int> data{0, 1, 2, 3};
    Tensor<int> tensor({2, 2});
    tensor.fillWithData(data.data(), data.size() * sizeof(int));

    for(size_t i = 0; i < data.size(); i++)
    {
        EXPECT_EQ(data[i], tensor.memory().hostData()[i]);
    }
}

TEST(TestTensor, MixedIndexingMethods)
{
    Tensor<float> tensor({3, 4, 5});
    tensor.fillWithValue(0.0f);

    tensor.setHostValue(100.0f, 2, 3, 4);
    std::vector<int64_t> indices = {2, 3, 4};
    EXPECT_FLOAT_EQ(tensor.getHostValue(indices), 100.0f);

    indices = {1, 2, 3};
    tensor.setHostValue(200.0f, indices);
    EXPECT_FLOAT_EQ(tensor.getHostValue(1, 2, 3), 200.0f);
}

TEST(TestTensor, GetIndexConsistency)
{
    const Tensor<float> tensor({5, 6, 7});

    for(int i = 0; i < 5; ++i)
    {
        for(int j = 0; j < 6; ++j)
        {
            for(int k = 0; k < 7; ++k)
            {
                auto variadicIndex = tensor.getIndex(i, j, k);

                const std::vector<int64_t> vecIndices = {i, j, k};
                auto vectorIndex = tensor.getIndex(vecIndices);

                EXPECT_EQ(variadicIndex, vectorIndex);
            }
        }
    }
}

TEST(TestTensor, DefaultPackedStrides1D)
{
    Tensor<float> tensor({10});

    EXPECT_EQ(tensor.dims(), (std::vector<int64_t>{10}));
    EXPECT_EQ(tensor.strides(), (std::vector<int64_t>{1}));
    EXPECT_EQ(tensor.memory().count(), 10);
}

TEST(TestTensor, DefaultPackedStrides2D)
{
    Tensor<float> tensor({5, 8});

    EXPECT_EQ(tensor.dims(), (std::vector<int64_t>{5, 8}));
    // Row-major: outer dim stride = 8, inner dim stride = 1
    EXPECT_EQ(tensor.strides(), (std::vector<int64_t>{8, 1}));
    EXPECT_EQ(tensor.memory().count(), 40);
}

TEST(TestTensor, DefaultPackedStrides3D)
{
    Tensor<float> tensor({4, 5, 6});

    EXPECT_EQ(tensor.dims(), (std::vector<int64_t>{4, 5, 6}));
    // Row-major: strides = {5*6, 6, 1} = {30, 6, 1}
    EXPECT_EQ(tensor.strides(), (std::vector<int64_t>{30, 6, 1}));
    EXPECT_EQ(tensor.memory().count(), 120);
}

TEST(TestTensor, DefaultPackedStrides4D)
{
    Tensor<float> tensor({2, 3, 4, 5});

    EXPECT_EQ(tensor.dims(), (std::vector<int64_t>{2, 3, 4, 5}));
    // Row-major: strides = {3*4*5, 4*5, 5, 1} = {60, 20, 5, 1}
    EXPECT_EQ(tensor.strides(), (std::vector<int64_t>{60, 20, 5, 1}));
    EXPECT_EQ(tensor.memory().count(), 120);
}

TEST(TestTensor, DefaultPackedStrides5D)
{
    Tensor<float> tensor({2, 3, 4, 5, 6});

    EXPECT_EQ(tensor.dims(), (std::vector<int64_t>{2, 3, 4, 5, 6}));
    // Row-major: strides = {3*4*5*6, 4*5*6, 5*6, 6, 1} = {360, 120, 30, 6, 1}
    EXPECT_EQ(tensor.strides(), (std::vector<int64_t>{360, 120, 30, 6, 1}));
    EXPECT_EQ(tensor.memory().count(), 720);
}

TEST(TestTensor, DefaultPackedStrides6D)
{
    Tensor<float> tensor({1, 2, 3, 4, 5, 6});

    EXPECT_EQ(tensor.dims(), (std::vector<int64_t>{1, 2, 3, 4, 5, 6}));
    // Row-major: strides = {2*3*4*5*6, 3*4*5*6, 4*5*6, 5*6, 6, 1} = {720, 360, 120, 30, 6, 1}
    EXPECT_EQ(tensor.strides(), (std::vector<int64_t>{720, 360, 120, 30, 6, 1}));
    EXPECT_EQ(tensor.memory().count(), 720);
}

TEST(TestTensor, DefaultPackedStridesIndexing)
{
    Tensor<float> tensor({3, 4, 5});
    tensor.fillWithValue(0.0f);

    // Test setting and getting values with default packed strides
    tensor.setHostValue(1.0f, 0, 0, 0);
    tensor.setHostValue(2.0f, 1, 2, 3);
    tensor.setHostValue(3.0f, 2, 3, 4);

    EXPECT_FLOAT_EQ(tensor.getHostValue(0, 0, 0), 1.0f);
    EXPECT_FLOAT_EQ(tensor.getHostValue(1, 2, 3), 2.0f);
    EXPECT_FLOAT_EQ(tensor.getHostValue(2, 3, 4), 3.0f);

    // Verify index calculation
    // For (1, 2, 3) with strides {20, 5, 1}: 1*20 + 2*5 + 3*1 = 33
    EXPECT_EQ(tensor.getIndex(1, 2, 3), 33);
}

TEST(TestTensor, DefaultPackedStridesCompatibleWithNcl)
{
    // Default packed strides should be equivalent to NCL for 3D tensors
    const std::vector<int64_t> dims = {2, 3, 4};
    const Tensor<float> tensorDefault(dims);
    const Tensor<float> tensorNcl(dims, TensorLayout::NCL);

    EXPECT_EQ(tensorDefault.strides(), tensorNcl.strides());
}

TEST(TestTensor, DefaultPackedStridesCompatibleWithNchw)
{
    // Default packed strides should be equivalent to NCHW for 4D tensors
    const std::vector<int64_t> dims = {2, 3, 4, 5};
    const Tensor<float> tensorDefault(dims);
    const Tensor<float> tensorNchw(dims, TensorLayout::NCHW);

    EXPECT_EQ(tensorDefault.strides(), tensorNchw.strides());
}

TEST(TestTensor, DefaultPackedStridesCompatibleWithNcdhw)
{
    // Default packed strides should be equivalent to NCDHW for 5D tensors
    const std::vector<int64_t> dims = {2, 3, 4, 5, 6};
    const Tensor<float> tensorDefault(dims);
    const Tensor<float> tensorNcdhw(dims, TensorLayout::NCDHW);

    EXPECT_EQ(tensorDefault.strides(), tensorNcdhw.strides());
}

TEST(TestTensor, BasicNcdhwUsage)
{
    Tensor<float> tensor({1, 2, 3, 4, 5}, TensorLayout::NCDHW);

    // NCDHW (row-major) strides with dims {N=1, C=2, D=3, H=4, W=5}:
    // N stride = C*D*H*W = 2*3*4*5 = 120
    // C stride = D*H*W = 3*4*5 = 60
    // D stride = H*W = 4*5 = 20
    // H stride = W = 5
    // W stride = 1 (innermost dimension)
    EXPECT_EQ(tensor.memory().count(), 120);
    EXPECT_EQ(tensor.strides()[0], 120);
    EXPECT_EQ(tensor.strides()[1], 60);
    EXPECT_EQ(tensor.strides()[2], 20);
    EXPECT_EQ(tensor.strides()[3], 5);
    EXPECT_EQ(tensor.strides()[4], 1);
}

TEST(TestTensor, BasicNdhwcUsage)
{
    Tensor<float> tensor({1, 2, 3, 4, 5}, TensorLayout::NDHWC);

    EXPECT_EQ(tensor.memory().count(), 120);
    // NDHWC strides with dims {N=1, C=2, D=3, H=4, W=5}:
    // N stride = D*H*W*C = 3*4*5*2 = 120
    // C stride = 1 (innermost dimension)
    // D stride = H*W*C = 4*5*2 = 40
    // H stride = W*C = 5*2 = 10
    // W stride = C = 2
    EXPECT_EQ(tensor.strides()[0], 120);
    EXPECT_EQ(tensor.strides()[1], 1);
    EXPECT_EQ(tensor.strides()[2], 40);
    EXPECT_EQ(tensor.strides()[3], 10);
    EXPECT_EQ(tensor.strides()[4], 2);
}

TEST(TestTensor, GetAndSetHostValueNcdhw)
{
    Tensor<float> tensor({1, 2, 3, 4, 5}, TensorLayout::NCDHW);
    tensor.fillWithValue(0.0f);
    tensor.setHostValue(99.0f, 0, 1, 2, 3, 4);

    EXPECT_FLOAT_EQ(tensor.getHostValue(0, 1, 2, 3, 4), 99.0f);
}

TEST(TestTensor, GetAndSetHostValueNdhwc)
{
    Tensor<float> tensor({1, 2, 3, 4, 5}, TensorLayout::NDHWC);
    tensor.fillWithValue(0.0f);
    tensor.setHostValue(99.0f, 0, 1, 2, 3, 4);

    EXPECT_FLOAT_EQ(tensor.getHostValue(0, 1, 2, 3, 4), 99.0f);
}

TEST(TestTensor, ElementAccessOperatorNcl)
{
    Tensor<float> tensor({2, 3, 4}, TensorLayout::NCL);
    tensor.fillWithValue(0.0f);
    tensor(1, 2, 3) = 79.0f;

    EXPECT_FLOAT_EQ(tensor(1, 2, 3), 79.0f);
}

TEST(TestTensor, ElementAccessOperatorNlc)
{
    Tensor<float> tensor({2, 3, 4}, TensorLayout::NLC);
    tensor.fillWithValue(0.0f);
    tensor(1, 2, 3) = 79.0f;

    EXPECT_FLOAT_EQ(tensor(1, 2, 3), 79.0f);
}

TEST(TestTensor, ElementAccessOperatorNchw)
{
    Tensor<float> tensor({1, 2, 3, 4});
    tensor.fillWithValue(0.0f);
    tensor(0, 1, 2, 2) = 79.0f;

    EXPECT_FLOAT_EQ(tensor(0, 1, 2, 2), 79.0f);
}

TEST(TestTensor, ElementAccessOperatorNhwc)
{
    Tensor<float> tensor({1, 2, 3, 4}, TensorLayout::NHWC);
    tensor.fillWithValue(0.0f);
    tensor(0, 1, 2, 2) = 79.0f;

    EXPECT_FLOAT_EQ(tensor(0, 1, 2, 2), 79.0f);
}

TEST(TestTensor, ElementAccessOperatorNcdhw)
{
    Tensor<float> tensor({1, 2, 3, 4, 5}, TensorLayout::NCDHW);
    tensor.fillWithValue(0.0f);
    tensor(0, 1, 2, 3, 4) = 79.0f;

    EXPECT_FLOAT_EQ(tensor(0, 1, 2, 3, 4), 79.0f);
}

TEST(TestTensor, ElementAccessOperatorNdhwc)
{
    Tensor<float> tensor({1, 2, 3, 4, 5}, TensorLayout::NDHWC);
    tensor.fillWithValue(0.0f);
    tensor(0, 1, 2, 3, 4) = 79.0f;

    EXPECT_FLOAT_EQ(tensor(0, 1, 2, 3, 4), 79.0f);
}

TEST(TestTensor, GetIndex5D)
{
    //Strides {120, 60, 20, 5, 1}
    const Tensor<float> tensor({2, 2, 3, 4, 5}, TensorLayout::NCDHW);

    // 1*120 + 1*60 + 2*20 + 3*5 + 4*1 = 239
    EXPECT_EQ(tensor.getIndex(1, 1, 2, 3, 4), 239);
    const std::vector<int64_t> indices1 = {1, 1, 2, 3, 4};
    EXPECT_EQ(tensor.getIndex(indices1), 239);

    // Test first element
    EXPECT_EQ(tensor.getIndex(0, 0, 0, 0, 0), 0);
    const std::vector<int64_t> indices2 = {0, 0, 0, 0, 0};
    EXPECT_EQ(tensor.getIndex(indices2), 0);

    // Test partial indices
    EXPECT_EQ(tensor.getIndex(1, 0), 120);
    const std::vector<int64_t> indices3 = {1, 0};
    EXPECT_EQ(tensor.getIndex(indices3), 120);

    EXPECT_EQ(tensor.getIndex(0, 1), 60);
    const std::vector<int64_t> indices4 = {0, 1};
    EXPECT_EQ(tensor.getIndex(indices4), 60);
}

TEST(TestTensor, GetIndex5DNdhwc)
{
    //Strides {120, 1, 40, 10, 2}
    const Tensor<float> tensor({2, 2, 3, 4, 5}, TensorLayout::NDHWC);

    // 1*120 + 1*1 + 2*40 + 3*10 + 4*2 = 239
    EXPECT_EQ(tensor.getIndex(1, 1, 2, 3, 4), 239);
    const std::vector<int64_t> indices = {1, 1, 2, 3, 4};
    EXPECT_EQ(tensor.getIndex(indices), 239);
}

TEST(TestTensor, ConstructorThrowsOnMismatchedDimsAndStrides)
{
    const std::vector<int64_t> dims = {2, 3, 4};
    const std::vector<int64_t> strides = {12, 4}; // Only 2 strides for 3 dims

    EXPECT_THROW(const Tensor<float> tensor(dims, strides), std::invalid_argument);
}

TEST(TestTensor, ConstructorThrowsOnNegativeDimension)
{
    const std::vector<int64_t> dims = {2, -3, 4};
    const std::vector<int64_t> strides = {12, 4, 1};

    EXPECT_THROW(const Tensor<float> tensor(dims, strides), std::invalid_argument);
}

TEST(TestTensor, ConstructorThrowsOnZeroDimension)
{
    const std::vector<int64_t> dims = {2, 0, 4};
    const std::vector<int64_t> strides = {12, 4, 1};

    EXPECT_THROW(const Tensor<float> tensor(dims, strides), std::invalid_argument);
}

TEST(TestTensor, ConstructorThrowsOnNegativeStride)
{
    const std::vector<int64_t> dims = {2, 3, 4};
    const std::vector<int64_t> strides = {12, -4, 1};

    EXPECT_THROW(const Tensor<float> tensor(dims, strides), std::invalid_argument);
}

TEST(TestTensor, ConstructorThrowsOnZeroStride)
{
    const std::vector<int64_t> dims = {2, 3, 4};
    const std::vector<int64_t> strides = {12, 0, 1};

    EXPECT_THROW(const Tensor<float> tensor(dims, strides), std::invalid_argument);
}

TEST(TestTensor, ConstructorThrowsOnMultipleInvalidDimensions)
{
    const std::vector<int64_t> dims = {-2, 1, -4};
    const std::vector<int64_t> strides = {12, 4, 1};

    EXPECT_THROW(const Tensor<float> tensor(dims, strides), std::invalid_argument);
}

TEST(TestTensor, ConstructorThrowsOnMultipleInvalidStrides)
{
    const std::vector<int64_t> dims = {2, 3, 4};
    const std::vector<int64_t> strides = {-12, 1, -1};

    EXPECT_THROW(const Tensor<float> tensor(dims, strides), std::invalid_argument);
}

// Sparse (strided) tensor test
TEST(TestTensor, SparseTensorCreationAndUsage)
{
    // Create a sparse tensor with dims {2,2,2,2} and strides {2,4,8,16}
    // This represents a non-packed layout with gaps in memory
    const std::vector<int64_t> dims = {2, 2, 2, 2};
    const std::vector<int64_t> strides = {2, 4, 8, 16};

    Tensor<float> tensor(dims, strides);

    // Verify properties
    EXPECT_EQ(tensor.dims(), dims);
    EXPECT_EQ(tensor.strides(), strides);
    EXPECT_EQ(tensor.elementCount(), 16); // Logical elements: 2*2*2*2

    // But calculateElementSpace returns sum of (dim-1)*stride
    // = (2-1)*2 + (2-1)*4 + (2-1)*8 + (2-1)*16 = 2+4+8+16 = 30
    // Add the init value of 1 and you get 31
    EXPECT_EQ(tensor.elementSpace(), 31);

    // Verify it's not packed
    EXPECT_FALSE(tensor.isPacked());

    // Test setting and getting values at different indices
    tensor.fillWithValue(0.0f);

    // Set values at specific logical indices
    tensor.setHostValue(10.0f, 0, 0, 0, 0); // Offset: 0
    tensor.setHostValue(20.0f, 1, 0, 0, 0); // Offset: 1*2 = 2
    tensor.setHostValue(30.0f, 0, 1, 0, 0); // Offset: 1*4 = 4
    tensor.setHostValue(40.0f, 0, 0, 1, 0); // Offset: 1*8 = 8
    tensor.setHostValue(50.0f, 0, 0, 0, 1); // Offset: 1*16 = 16
    tensor.setHostValue(99.0f, 1, 1, 1, 1); // Offset: 2+4+8+16 = 30

    // Verify values
    EXPECT_FLOAT_EQ(tensor.getHostValue(0, 0, 0, 0), 10.0f);
    EXPECT_FLOAT_EQ(tensor.getHostValue(1, 0, 0, 0), 20.0f);
    EXPECT_FLOAT_EQ(tensor.getHostValue(0, 1, 0, 0), 30.0f);
    EXPECT_FLOAT_EQ(tensor.getHostValue(0, 0, 1, 0), 40.0f);
    EXPECT_FLOAT_EQ(tensor.getHostValue(0, 0, 0, 1), 50.0f);
    EXPECT_FLOAT_EQ(tensor.getHostValue(1, 1, 1, 1), 99.0f);

    // Verify index calculations
    EXPECT_EQ(tensor.getIndex(0, 0, 0, 0), 0);
    EXPECT_EQ(tensor.getIndex(1, 0, 0, 0), 2);
    EXPECT_EQ(tensor.getIndex(0, 1, 0, 0), 4);
    EXPECT_EQ(tensor.getIndex(0, 0, 1, 0), 8);
    EXPECT_EQ(tensor.getIndex(0, 0, 0, 1), 16);
    EXPECT_EQ(tensor.getIndex(1, 1, 1, 1), 30);
}

/* ======== fillWithSentinelValue tests ======== */

template <typename T>
class TensorSentinel : public ::testing::Test
{
};

using SentinelTypes = ::testing::Types<float,
                                       double,
                                       half,
                                       bfloat16,
                                       hipdnn_data_sdk::types::fp8_e4m3,
                                       hipdnn_data_sdk::types::fp8_e5m2,
                                       int8_t,
                                       uint8_t,
                                       int32_t>;

TYPED_TEST_SUITE(TensorSentinel, SentinelTypes, );

TYPED_TEST(TensorSentinel, PackedTensorFilled)
{
    using hipdnn_data_sdk::types::isnan;
    Tensor<TypeParam> tensor({2, 3});
    tensor.fillWithSentinelValue();

    for(auto valuePtr : tensor)
    {
        auto value = *static_cast<TypeParam*>(valuePtr);
        if constexpr(std::numeric_limits<TypeParam>::has_quiet_NaN)
        {
            EXPECT_TRUE(isnan(value));
        }
        else
        {
            EXPECT_EQ(value, std::numeric_limits<TypeParam>::max());
        }
    }
}

TYPED_TEST(TensorSentinel, StridedTensorFilled)
{
    using hipdnn_data_sdk::types::isnan;
    const std::vector<int64_t> dims = {2, 2, 2};
    const std::vector<int64_t> strides = {2, 4, 8};

    Tensor<TypeParam> tensor(dims, strides);
    tensor.fillWithSentinelValue();

    for(auto valuePtr : tensor)
    {
        auto value = *static_cast<TypeParam*>(valuePtr);
        if constexpr(std::numeric_limits<TypeParam>::has_quiet_NaN)
        {
            EXPECT_TRUE(isnan(value));
        }
        else
        {
            EXPECT_EQ(value, std::numeric_limits<TypeParam>::max());
        }
    }
}

/// A stride PERMUTATION must be iterated in index order, not linearly.
///
/// Regression guard for visitsInIndexOrder. dims {2,3,4} with strides {12,1,3} makes
/// axis 1 the fastest-varying in memory, and elementCount == elementSpace == 24, so
/// isPacked() is true and the old gate took the linear path. Writes then landed on
/// coordinates the stride-aware reads did not fetch them by, with no error anywhere.
TEST(TestTensor, IteratesAPermutedStrideTensorInIndexOrder)
{
    const std::vector<int64_t> dims{2, 3, 4};
    const std::vector<int64_t> strides{12, 1, 3};

    // NOLINTNEXTLINE(misc-const-correctness) mutated through the iterator; begin() is not const
    hipdnn_data_sdk::utilities::Tensor<float> tensor(dims, strides);

    EXPECT_TRUE(tensor.isPacked())
        << "a permutation spans the same memory, so isPacked() must still be true here "
           "-- if it is not, this test no longer covers the case it was written for";

    EXPECT_NO_THROW(std::ignore
                    = std::get<ITensorIterator<false>::CompositeIndex>(tensor.begin().index()))
        << "a stride permutation must be demoted to the stride-aware walk";

    // Write a distinct value per position in iteration order, the way a random fill does.
    float next = 1.0F;
    for(auto valuePtr : tensor)
    {
        *static_cast<float*>(valuePtr) = next;
        next += 1.0F;
    }

    // Read back by coordinate: the n-th value written must land on the n-th coordinate.
    float expected = 1.0F;
    for(int64_t i0 = 0; i0 < dims[0]; ++i0)
    {
        for(int64_t i1 = 0; i1 < dims[1]; ++i1)
        {
            for(int64_t i2 = 0; i2 < dims[2]; ++i2)
            {
                EXPECT_EQ(tensor.getHostValue(i0, i1, i2), expected)
                    << "value at (" << i0 << "," << i1 << "," << i2
                    << ") does not match its position in iteration order. The iterator "
                       "wrote linearly while the read honoured the declared strides.";
                expected += 1.0F;
            }
        }
    }
}

/// Row-major tensors must keep the linear fast path.
///
/// The gate is only correct if it does not demote every packed tensor to the slower
/// walk. Both strategies produce the right values, so assert the chosen one too.
TEST(TestTensor, IteratesARowMajorTensorInIndexOrder)
{
    const std::vector<int64_t> dims{2, 3, 4};
    const std::vector<int64_t> strides{12, 4, 1};

    // NOLINTNEXTLINE(misc-const-correctness) mutated through the iterator; begin() is not const
    hipdnn_data_sdk::utilities::Tensor<float> tensor(dims, strides);
    EXPECT_TRUE(tensor.isPacked())
        << "these strides are the packed row-major strides, so isPacked() must be true "
           "here -- if it is not, this test no longer guards the fast path it was "
           "written for";

    EXPECT_NO_THROW(std::ignore
                    = std::get<ITensorIterator<false>::LinearIndex>(tensor.begin().index()))
        << "packed row-major strides must keep the linear fast path";

    float next = 1.0F;
    for(auto valuePtr : tensor)
    {
        *static_cast<float*>(valuePtr) = next;
        next += 1.0F;
    }

    float expected = 1.0F;
    for(int64_t i0 = 0; i0 < dims[0]; ++i0)
    {
        for(int64_t i1 = 0; i1 < dims[1]; ++i1)
        {
            for(int64_t i2 = 0; i2 < dims[2]; ++i2)
            {
                EXPECT_EQ(tensor.getHostValue(i0, i1, i2), expected);
                expected += 1.0F;
            }
        }
    }
}

/// An axis of extent 1 must not disqualify the fast path.
///
/// NHWC activations with 1x1 spatial extent carry dims {N,C,1,1} with strides
/// {C,1,C,C}, because the H and W strides come from the layout rather than from the
/// unit extents. Index 0 on such an axis contributes 0 whatever the stride says, so
/// the tensor still visits memory in index order. Both index strategies satisfy that,
/// so the value check alone cannot tell whether the exemption survives; assert the
/// chosen strategy as well, the way TestTensorIterator does.
TEST(TestTensor, IteratesAUnitExtentTensorInIndexOrder)
{
    const std::vector<int64_t> dims{2, 3, 1, 1};
    const std::vector<int64_t> strides{3, 1, 3, 3};

    // NOLINTNEXTLINE(misc-const-correctness) mutated through the iterator; begin() is not const
    hipdnn_data_sdk::utilities::Tensor<float> tensor(dims, strides);
    EXPECT_TRUE(tensor.isPacked())
        << "the unused unit-axis strides do not extend the span, so isPacked() must be "
           "true here -- if it is not, this test no longer covers the case it was "
           "written for";

    EXPECT_NO_THROW(std::ignore
                    = std::get<ITensorIterator<false>::LinearIndex>(tensor.begin().index()))
        << "unit-extent axes must not push a row-major tensor onto the strided walk";

    float next = 1.0F;
    for(auto valuePtr : tensor)
    {
        *static_cast<float*>(valuePtr) = next;
        next += 1.0F;
    }

    float expected = 1.0F;
    for(int64_t i0 = 0; i0 < dims[0]; ++i0)
    {
        for(int64_t i1 = 0; i1 < dims[1]; ++i1)
        {
            EXPECT_EQ(tensor.getHostValue(i0, i1, 0, 0), expected)
                << "value at (" << i0 << "," << i1
                << ",0,0) does not match its position in iteration order";
            expected += 1.0F;
        }
    }
}
