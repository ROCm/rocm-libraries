// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>
#include <hipdnn_data_sdk/utilities/StringUtil.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_frontend::graph;

TEST(TestTensorAttributes, DefaultConstructor)
{
    const TensorAttributes tensor;
    EXPECT_EQ(tensor.get_uid(), 0);
    EXPECT_EQ(tensor.get_name(), "");
    EXPECT_EQ(tensor.get_data_type(), DataType::NOT_SET);
    EXPECT_TRUE(tensor.get_stride().empty());
    EXPECT_TRUE(tensor.get_dim().empty());
    EXPECT_EQ(tensor.get_volume(), 1);
    EXPECT_FALSE(tensor.get_is_virtual());
    EXPECT_FALSE(tensor.has_uid());
}

TEST(TestTensorAttributes, SetAndGetUid)
{
    TensorAttributes tensor;
    tensor.set_uid(42);
    EXPECT_EQ(tensor.get_uid(), 42);
    EXPECT_TRUE(tensor.has_uid());

    tensor.clear_uid();
    EXPECT_EQ(tensor.get_uid(), 0);
    EXPECT_FALSE(tensor.has_uid());
}

TEST(TestTensorAttributes, SetAndGetName)
{
    TensorAttributes tensor;
    tensor.set_name("TestTensor");
    EXPECT_EQ(tensor.get_name(), "TestTensor");
}

TEST(TestTensorAttributes, SetAndGetDataType)
{
    TensorAttributes tensor;
    tensor.set_data_type(DataType::FLOAT);
    EXPECT_EQ(tensor.get_data_type(), DataType::FLOAT);
}

TEST(TestTensorAttributes, SetAndGetStride)
{
    TensorAttributes tensor;
    tensor.set_stride({1, 2, 3});
    EXPECT_EQ(tensor.get_stride(), std::vector<int64_t>({1, 2, 3}));
}

TEST(TestTensorAttributes, SetAndGetDim)
{
    TensorAttributes tensor;
    tensor.set_dim({4, 5, 6});
    EXPECT_EQ(tensor.get_dim(), std::vector<int64_t>({4, 5, 6}));
    EXPECT_EQ(tensor.get_volume(), 4 * 5 * 6);
}

TEST(TestTensorAttributes, SetAndGetIsVirtual)
{
    TensorAttributes tensor;
    tensor.set_is_virtual(true);
    EXPECT_TRUE(tensor.get_is_virtual());

    tensor.set_is_virtual(false);
    EXPECT_FALSE(tensor.get_is_virtual());
}

TEST(TestTensorAttributes, SetOutput)
{
    TensorAttributes tensor;
    tensor.set_output(true);
    EXPECT_FALSE(tensor.get_is_virtual());

    tensor.set_output(false);
    EXPECT_TRUE(tensor.get_is_virtual());
}

TEST(TestTensorAttributes, SetFromGraphAttributes)
{
    GraphAttributes graphAttributes;
    graphAttributes.set_io_data_type(DataType::FLOAT);
    graphAttributes.set_intermediate_data_type(DataType::HALF);

    TensorAttributes tensor;
    tensor.set_is_virtual(false).fill_from_context(graphAttributes);
    EXPECT_EQ(tensor.get_data_type(), DataType::FLOAT);

    tensor.set_data_type(DataType::NOT_SET);
    tensor.set_is_virtual(true).fill_from_context(graphAttributes);
    EXPECT_EQ(tensor.get_data_type(), DataType::HALF);
}

TEST(TestTensorAttributes, ValidateSucceedsOnValueTensor)
{
    const TensorAttributes tensor(1.f);
    EXPECT_EQ(tensor.validate(), Error(ErrorCode::OK, ""));
}

TEST(TestTensorAttributes, ValidateSucceedsOnVirtualTensor)
{
    TensorAttributes tensor;
    tensor.set_dim({1});
    tensor.set_stride({1});
    tensor.set_is_virtual(true);
    tensor.set_data_type(DataType::FLOAT);

    EXPECT_EQ(tensor.validate(), Error(ErrorCode::OK, ""));
}

TEST(TestTensorAttributes, ValidateFailsOnVirtualValueTensor)
{
    TensorAttributes tensor(1.f);
    tensor.set_dim({1});
    tensor.set_stride({1});
    tensor.set_is_virtual(true);

    EXPECT_EQ(tensor.validate(),
              Error(ErrorCode::INVALID_VALUE, "Tensor  cannot be virtual and pass by value"));
}

TEST(TestTensorAttributes, ValidateFailsOnDifferentDimAndStrideSize)
{
    TensorAttributes tensor;
    tensor.set_dim({1});
    tensor.set_stride({1, 2});
    tensor.set_data_type(DataType::FLOAT);

    EXPECT_EQ(tensor.validate(),
              Error(ErrorCode::INVALID_VALUE, "Tensor  dims and strides have different sizes"));
}

TEST(TestTensorAttributes, ValidateFailsOnEmptyDims)
{
    TensorAttributes tensor;
    tensor.set_data_type(DataType::FLOAT);

    EXPECT_EQ(tensor.validate(),
              Error(ErrorCode::ATTRIBUTE_NOT_SET, "Tensor  dims must be non-empty"));
}

TEST(TestTensorAttributes, ValidateFailsOnNonPositiveDimension)
{
    const std::vector<std::vector<int64_t>> testDims
        = {{0, 1}, {1, 0, 1}, {-1, 1, 1}, {1, 1, 1, -1}};

    for(const auto& dim : testDims)
    {
        TensorAttributes tensor;
        tensor.set_dim(dim);
        tensor.set_stride(dim);
        tensor.set_data_type(DataType::FLOAT);

        EXPECT_EQ(tensor.validate(),
                  Error(ErrorCode::INVALID_VALUE, "Tensor  must have only positive dimensions"))
            << "Dims: " << hipdnn_data_sdk::utilities::vecToString(dim);
    }
}

TEST(TestTensorAttributes, ValidateDataType)
{
    TensorAttributes tensor;
    tensor.set_dim({4, 5, 6});
    tensor.set_stride({0, 1, 2});

    const std::vector<std::pair<DataType, ErrorCode>> expectedResults
        = {{DataType::NOT_SET, ErrorCode::ATTRIBUTE_NOT_SET},
           {DataType::FLOAT, ErrorCode::OK},
           {DataType::HALF, ErrorCode::OK},
           {DataType::BFLOAT16, ErrorCode::OK},
           {DataType::DOUBLE, ErrorCode::OK},
           {DataType::UINT8, ErrorCode::OK},
           {DataType::INT32, ErrorCode::OK},
           {DataType::INT8, ErrorCode::OK},
           {DataType::FP8_E4M3, ErrorCode::OK},
           {DataType::FP8_E5M2, ErrorCode::OK},
           {DataType::FP8_E8M0, ErrorCode::OK},
           {DataType::FP4_E2M1, ErrorCode::OK},
           {DataType::INT4, ErrorCode::OK},
           {DataType::FP6_E2M3, ErrorCode::OK},
           {DataType::FP6_E3M2, ErrorCode::OK},
           {DataType::INT64, ErrorCode::OK}};

    for(auto [dataType, errorCode] : expectedResults)
    {
        tensor.set_data_type(dataType);
        auto result = tensor.validate();
        EXPECT_EQ(result.code, errorCode) << "For " + std::string(to_string(dataType));
    }
}
