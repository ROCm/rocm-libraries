// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

using namespace hipdnn_sdk::utilities;

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

TEST(TestTensor, FillWithValues)
{
    Tensor<float> tensor({1, 2, 3, 4});

    tensor.fillWithValue(1.0f);
    auto buffer = tensor.memory().hostData();

    for(size_t i = 0; i < tensor.memory().count(); i++)
    {
        EXPECT_FLOAT_EQ(1.0f, buffer[i]);
    }
}

TEST(TestTensor, FillWithRandomValues)
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
    Tensor<float> tensor({2, 3, 4, 5});

    // 1*60 + 2*20 + 3*5 + 4*1 = 119
    EXPECT_EQ(tensor.getIndex(1, 2, 3, 4), 119);
    std::vector<int64_t> indices1 = {1, 2, 3, 4};
    EXPECT_EQ(tensor.getIndex(indices1), 119);

    // Test first element
    EXPECT_EQ(tensor.getIndex(0, 0, 0, 0), 0);
    std::vector<int> indices2 = {0, 0, 0, 0};
    EXPECT_EQ(tensor.getIndex(indices2), 0);

    // Test partial indices
    EXPECT_EQ(tensor.getIndex(1, 0), 60);
    std::vector<size_t> indices3 = {1, 0};
    EXPECT_EQ(tensor.getIndex(indices3), 60);

    EXPECT_EQ(tensor.getIndex(0, 1), 20);
    std::vector<size_t> indices4 = {0, 1};
    EXPECT_EQ(tensor.getIndex(indices4), 20);
}

TEST(TestTensor, GetIndexNhwc)
{
    //Strides {60, 1, 15, 3}
    Tensor<float> tensor({2, 3, 4, 5}, TensorLayout::NHWC);

    // 1*60 + 2*1 + 3*15 + 4*3 = 119
    EXPECT_EQ(tensor.getIndex(1, 2, 3, 4), 119);
    std::vector<int64_t> indices = {1, 2, 3, 4};
    EXPECT_EQ(tensor.getIndex(indices), 119);
}

TEST(TestTensor, GetIndexEdgeCases)
{
    Tensor<float> tensor({10, 20});

    // Last element
    EXPECT_EQ(tensor.getIndex(9, 19), 199);

    std::vector<int64_t> indices = {9, 19};
    EXPECT_EQ(tensor.getIndex(indices), 199);

    // Single dim
    EXPECT_EQ(tensor.getIndex(5), 100);
    indices = {5};
    EXPECT_EQ(tensor.getIndex(indices), 100);

    // Empty dim
    std::vector<int64_t> emptyIndices;
    EXPECT_EQ(tensor.getIndex(emptyIndices), 0);
}

TEST(TestTensor, GetIndexInvalidCases)
{
    Tensor<float> tensor({2, 3, 4});

    EXPECT_THROW(tensor.getIndex(0, 1, 2, 3), std::invalid_argument);

    std::vector<int64_t> tooManyIndices = {0, 1, 2, 3};
    EXPECT_THROW(tensor.getIndex(tooManyIndices), std::invalid_argument);
}

TEST(TestTensor, GetHostValueWithVector)
{
    Tensor<float> tensor({2, 3, 4});
    tensor.fillWithValue(0.0f);

    tensor.setHostValue(42.0f, 1, 2, 3);
    std::vector<size_t> indices2 = {1, 2, 3};
    EXPECT_FLOAT_EQ(tensor.getHostValue(indices2), 42.0f);
}

TEST(TestTensor, SetHostValueWithVector)
{
    Tensor<float> tensor({2, 3, 4});
    tensor.fillWithValue(0.0f);

    std::vector<int64_t> indices1 = {0, 1, 2};
    tensor.setHostValue(10.0f, indices1);

    std::vector<size_t> indices2 = {1, 0, 3};
    tensor.setHostValue(20.0f, indices2);

    std::vector<int> indices3 = {1, 2, 1};
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
    Tensor<float> tensor({5, 6, 7});

    for(int i = 0; i < 5; ++i)
    {
        for(int j = 0; j < 6; ++j)
        {
            for(int k = 0; k < 7; ++k)
            {
                auto variadicIndex = tensor.getIndex(i, j, k);

                std::vector<int64_t> vecIndices = {i, j, k};
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

TEST(TestTensor, DefaultPackedStridesCompatibleWithNchw)
{
    // Default packed strides should be equivalent to NCHW for 4D tensors
    std::vector<int64_t> dims = {2, 3, 4, 5};
    Tensor<float> tensorDefault(dims);
    Tensor<float> tensorNchw(dims, TensorLayout::NCHW);

    EXPECT_EQ(tensorDefault.strides(), tensorNchw.strides());
}

TEST(TestTensor, DefaultPackedStridesCompatibleWithNcdhw)
{
    // Default packed strides should be equivalent to NCDHW for 5D tensors
    std::vector<int64_t> dims = {2, 3, 4, 5, 6};
    Tensor<float> tensorDefault(dims);
    Tensor<float> tensorNcdhw(dims, TensorLayout::NCDHW);

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

TEST(TestTensor, GetIndex5D)
{
    //Strides {120, 60, 20, 5, 1}
    Tensor<float> tensor({2, 2, 3, 4, 5}, TensorLayout::NCDHW);

    // 1*120 + 1*60 + 2*20 + 3*5 + 4*1 = 239
    EXPECT_EQ(tensor.getIndex(1, 1, 2, 3, 4), 239);
    std::vector<int64_t> indices1 = {1, 1, 2, 3, 4};
    EXPECT_EQ(tensor.getIndex(indices1), 239);

    // Test first element
    EXPECT_EQ(tensor.getIndex(0, 0, 0, 0, 0), 0);
    std::vector<int> indices2 = {0, 0, 0, 0, 0};
    EXPECT_EQ(tensor.getIndex(indices2), 0);

    // Test partial indices
    EXPECT_EQ(tensor.getIndex(1, 0), 120);
    std::vector<size_t> indices3 = {1, 0};
    EXPECT_EQ(tensor.getIndex(indices3), 120);

    EXPECT_EQ(tensor.getIndex(0, 1), 60);
    std::vector<size_t> indices4 = {0, 1};
    EXPECT_EQ(tensor.getIndex(indices4), 60);
}

TEST(TestTensor, GetIndex5DNdhwc)
{
    //Strides {120, 1, 40, 10, 2}
    Tensor<float> tensor({2, 2, 3, 4, 5}, TensorLayout::NDHWC);

    // 1*120 + 1*1 + 2*40 + 3*10 + 4*2 = 239
    EXPECT_EQ(tensor.getIndex(1, 1, 2, 3, 4), 239);
    std::vector<int64_t> indices = {1, 1, 2, 3, 4};
    EXPECT_EQ(tensor.getIndex(indices), 239);
}

TEST(TestTensor, ConstructorThrowsOnMismatchedDimsAndStrides)
{
    std::vector<int64_t> dims = {2, 3, 4};
    std::vector<int64_t> strides = {12, 4}; // Only 2 strides for 3 dims

    EXPECT_THROW(Tensor<float> tensor(dims, strides), std::invalid_argument);
}

TEST(TestTensor, ConstructorThrowsOnNegativeDimension)
{
    std::vector<int64_t> dims = {2, -3, 4};
    std::vector<int64_t> strides = {12, 4, 1};

    EXPECT_THROW(Tensor<float> tensor(dims, strides), std::invalid_argument);
}

TEST(TestTensor, ConstructorThrowsOnZeroDimension)
{
    std::vector<int64_t> dims = {2, 0, 4};
    std::vector<int64_t> strides = {12, 4, 1};

    EXPECT_THROW(Tensor<float> tensor(dims, strides), std::invalid_argument);
}

TEST(TestTensor, ConstructorThrowsOnNegativeStride)
{
    std::vector<int64_t> dims = {2, 3, 4};
    std::vector<int64_t> strides = {12, -4, 1};

    EXPECT_THROW(Tensor<float> tensor(dims, strides), std::invalid_argument);
}

TEST(TestTensor, ConstructorThrowsOnZeroStride)
{
    std::vector<int64_t> dims = {2, 3, 4};
    std::vector<int64_t> strides = {12, 0, 1};

    EXPECT_THROW(Tensor<float> tensor(dims, strides), std::invalid_argument);
}

TEST(TestTensor, ConstructorThrowsOnMultipleInvalidDimensions)
{
    std::vector<int64_t> dims = {-2, 1, -4};
    std::vector<int64_t> strides = {12, 4, 1};

    EXPECT_THROW(Tensor<float> tensor(dims, strides), std::invalid_argument);
}

TEST(TestTensor, ConstructorThrowsOnMultipleInvalidStrides)
{
    std::vector<int64_t> dims = {2, 3, 4};
    std::vector<int64_t> strides = {-12, 1, -1};

    EXPECT_THROW(Tensor<float> tensor(dims, strides), std::invalid_argument);
}

// Sparse (strided) tensor test
TEST(TestTensor, SparseTensorCreationAndUsage)
{
    // Create a sparse tensor with dims {2,2,2,2} and strides {2,4,8,16}
    // This represents a non-packed layout with gaps in memory
    std::vector<int64_t> dims = {2, 2, 2, 2};
    std::vector<int64_t> strides = {2, 4, 8, 16};

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
