// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file TestDeclaredAlignment.cpp
 * @brief The arithmetic every engine's applicability check shares for a tensor's declared
 *        alignment: a declaration of A bytes guarantees an N-byte access exactly when N
 *        divides A, and a declaration or need that is not positive guarantees nothing.
 */

#include <hip_kernel_provider_common/DeclaredAlignment.hpp>

#include <cstdint>
#include <vector>

#include <flatbuffers/flatbuffers.h>
#include <gtest/gtest.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>

namespace hip_kernel_provider_common
{
namespace
{

namespace data_objects = hipdnn_flatbuffers_sdk::data_objects;

/// A serialized tensor declaring @p alignment bytes. Owns the buffer the view points into.
class DeclaringTensor
{
public:
    explicit DeclaringTensor(int64_t alignment)
    {
        const std::vector<int64_t> dims{1, 2, 3, 4};
        const std::vector<int64_t> strides{24, 12, 4, 1};
        _builder.Finish(
            data_objects::CreateTensorAttributesDirect(_builder,
                                                       /*uid=*/1,
                                                       /*name=*/"t",
                                                       data_objects::DataType::FLOAT,
                                                       &strides,
                                                       &dims,
                                                       /*virtual_=*/false,
                                                       data_objects::TensorValue::NONE,
                                                       /*value=*/0,
                                                       /*is_runtime_pass_by_value=*/false,
                                                       ::flatbuffers::nullopt,
                                                       alignment));
    }

    const data_objects::TensorAttributes& get() const
    {
        return *::flatbuffers::GetRoot<data_objects::TensorAttributes>(_builder.GetBufferPointer());
    }

private:
    ::flatbuffers::FlatBufferBuilder _builder;
};

// Bug caught: a missing declaration (0) reported as covering every need.
TEST(TestDeclaredAlignment, ZeroDeclaredCoversNothing)
{
    EXPECT_FALSE(declaredAlignmentCovers(int64_t{0}, 1));
    EXPECT_FALSE(declaredAlignmentCovers(int64_t{0}, 4));
    EXPECT_FALSE(declaredAlignmentCovers(int64_t{0}, 16));
}

// Bug caught: a negative declaration passing the modulo test (-16 % 4 == 0 in C++).
TEST(TestDeclaredAlignment, NegativeDeclaredCoversNothing)
{
    EXPECT_FALSE(declaredAlignmentCovers(int64_t{-16}, 4));
    EXPECT_FALSE(declaredAlignmentCovers(int64_t{-16}, 16));
    EXPECT_FALSE(declaredAlignmentCovers(int64_t{-1}, 1));
}

// Bug caught: an empty need reported as covered (and a zero divisor reaching the modulo).
TEST(TestDeclaredAlignment, ZeroNeedIsNeverCovered)
{
    EXPECT_FALSE(declaredAlignmentCovers(int64_t{16}, 0));
    EXPECT_FALSE(declaredAlignmentCovers(int64_t{0}, 0));
}

// Bug caught: a negative need divided into the declaration and accepted.
TEST(TestDeclaredAlignment, NegativeNeedIsNeverCovered)
{
    EXPECT_FALSE(declaredAlignmentCovers(int64_t{16}, -4));
    EXPECT_FALSE(declaredAlignmentCovers(int64_t{16}, -16));
}

// Bug caught: a declaration below the need accepted, so a 4-byte promise serves a 16-byte access.
TEST(TestDeclaredAlignment, NeedAboveDeclaredIsNotCovered)
{
    EXPECT_FALSE(declaredAlignmentCovers(int64_t{8}, 16));
    EXPECT_FALSE(declaredAlignmentCovers(int64_t{4}, 8));
    EXPECT_FALSE(declaredAlignmentCovers(int64_t{1}, 2));
}

// Bug caught: alignment treated as a minimum (declared >= need) rather than a multiple
// guarantee, so a 24-byte declaration passes for a 16-byte access although address 24 is not a
// multiple of 16.
TEST(TestDeclaredAlignment, NonDivisibleDeclaredIsNotCovered)
{
    EXPECT_FALSE(declaredAlignmentCovers(int64_t{24}, 16));
    EXPECT_FALSE(declaredAlignmentCovers(int64_t{6}, 4));
    EXPECT_FALSE(declaredAlignmentCovers(int64_t{20}, 8));
}

// Bug caught: the boundary declined, so a kernel needing exactly what the graph declares is lost.
TEST(TestDeclaredAlignment, ExactMatchIsCovered)
{
    EXPECT_TRUE(declaredAlignmentCovers(int64_t{16}, 16));
    EXPECT_TRUE(declaredAlignmentCovers(int64_t{4}, 4));
    EXPECT_TRUE(declaredAlignmentCovers(int64_t{1}, 1));
}

// Bug caught: only an exact match accepted, so a graph declaring 16 loses every kernel that
// needs 8, 4, 2 or 1; and a declaration that is a multiple (not a power of two) mishandled.
TEST(TestDeclaredAlignment, MultipleOfNeedIsCovered)
{
    EXPECT_TRUE(declaredAlignmentCovers(int64_t{16}, 8));
    EXPECT_TRUE(declaredAlignmentCovers(int64_t{16}, 4));
    EXPECT_TRUE(declaredAlignmentCovers(int64_t{16}, 1));
    EXPECT_TRUE(declaredAlignmentCovers(int64_t{24}, 8));
    EXPECT_TRUE(declaredAlignmentCovers(int64_t{64}, 16));
}

// Bug caught: the tensor overload reading the wrong field, or not using the multiple rule.
TEST(TestDeclaredAlignment, TensorOverloadReadsTheDeclaredAlignment)
{
    const DeclaringTensor sixteen(16);
    EXPECT_TRUE(declaredAlignmentCovers(sixteen.get(), 16));
    EXPECT_TRUE(declaredAlignmentCovers(sixteen.get(), 4));
    EXPECT_FALSE(declaredAlignmentCovers(sixteen.get(), 32));

    const DeclaringTensor twentyFour(24);
    EXPECT_TRUE(declaredAlignmentCovers(twentyFour.get(), 8));
    EXPECT_FALSE(declaredAlignmentCovers(twentyFour.get(), 16));
}

// Bug caught: a tensor that declares nothing (or a bogus negative) served by the overload.
TEST(TestDeclaredAlignment, TensorOverloadDeclinesANonPositiveDeclaration)
{
    const DeclaringTensor zero(0);
    EXPECT_FALSE(declaredAlignmentCovers(zero.get(), 1));
    EXPECT_FALSE(declaredAlignmentCovers(zero.get(), 16));

    const DeclaringTensor negative(-16);
    EXPECT_FALSE(declaredAlignmentCovers(negative.get(), 4));

    const DeclaringTensor sixteen(16);
    EXPECT_FALSE(declaredAlignmentCovers(sixteen.get(), 0));
    EXPECT_FALSE(declaredAlignmentCovers(sixteen.get(), -4));
}

} // namespace
} // namespace hip_kernel_provider_common
