// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// AC for the shim Tensor_attributes (RFC 0012 §4.4.1): a cuDNN-shaped
// `graph::Tensor_attributes` that hipified v9 graph-construction code declares
// and configures against the shim. The shim aliases hipDNN's native
// `graph::TensorAttributes` 1:1 (no wrapper, no shim-side UID allocator or
// identity map), so these assertions both exercise the setter surface and pin
// that the alias is exact. Gated behind HIPDNN_ENABLE_CUDNN_COMPATIBILITY in
// the frontend tests CMakeLists.
#include <hipdnn_compatibility/cudnn/cudnn_frontend.h>

#include <gtest/gtest.h>

#include <type_traits>
#include <vector>

namespace cfe = hipdnn_frontend::compatibility::cudnn_frontend;

// The shim attribute type must be the *same* type as hipDNN's, not a parallel
// re-declaration (RFC 0012 §4.4.1 — alias, do not wrap). Both the class name
// and cuDNN's `Tensor_attributes` spelling resolve to the one hipDNN type.
static_assert(
    std::is_same_v<cfe::graph::Tensor_attributes, hipdnn_frontend::graph::TensorAttributes>,
    "cudnn_frontend::graph::Tensor_attributes must alias the hipDNN type");
static_assert(std::is_same_v<cfe::graph::Tensor_attributes, cfe::graph::TensorAttributes>,
              "the Tensor_attributes and TensorAttributes spellings must be the same type");

namespace
{
// Mirror a hipified consumer (RFC 0012 §4.2 workflow 1): resolve graph symbols
// through the shim namespace.
namespace fe = hipdnn_frontend::compatibility::cudnn_frontend;

TEST(TestCudnnShimTensor, DefaultConstructs)
{
    const fe::graph::Tensor_attributes tensor;
    EXPECT_EQ(tensor.get_uid(), 0);
    EXPECT_FALSE(tensor.has_uid());
    EXPECT_EQ(tensor.get_name(), "");
    EXPECT_EQ(tensor.get_data_type(), fe::DataType_t::NOT_SET);
    EXPECT_TRUE(tensor.get_dim().empty());
    EXPECT_TRUE(tensor.get_stride().empty());
    EXPECT_FALSE(tensor.get_is_virtual());
}

// Every setter named in the acceptance criteria that hipDNN publishes, exercised
// through the shim alias. The setters chain (return *this), exactly as cuDNN FE.
TEST(TestCudnnShimTensor, AllSettersChainAndApply)
{
    fe::graph::Tensor_attributes tensor;
    tensor.set_dim({1, 64, 28, 28})
        .set_stride({50176, 784, 28, 1})
        .set_data_type(fe::DataType_t::HALF)
        .set_uid(7)
        .set_is_virtual(false)
        .set_name("input_x");

    EXPECT_EQ(tensor.get_dim(), std::vector<int64_t>({1, 64, 28, 28}));
    EXPECT_EQ(tensor.get_stride(), std::vector<int64_t>({50176, 784, 28, 1}));
    EXPECT_EQ(tensor.get_data_type(), fe::DataType_t::HALF);
    EXPECT_EQ(tensor.get_uid(), 7);
    EXPECT_TRUE(tensor.has_uid());
    EXPECT_FALSE(tensor.get_is_virtual());
    EXPECT_EQ(tensor.get_name(), "input_x");
}

// set_output is the cuDNN convenience inverse of set_is_virtual.
TEST(TestCudnnShimTensor, SetOutputIsInverseOfVirtual)
{
    fe::graph::Tensor_attributes tensor;
    tensor.set_output(true);
    EXPECT_FALSE(tensor.get_is_virtual());
    tensor.set_output(false);
    EXPECT_TRUE(tensor.get_is_virtual());
}

// UID assignment / clearing forwards to hipDNN's native UID handling; the shim
// keeps no identity map of its own (RFC 0012 §4.4.1, §5.3).
TEST(TestCudnnShimTensor, UidAssignmentForwardsToHipdnn)
{
    fe::graph::Tensor_attributes tensor;
    EXPECT_FALSE(tensor.has_uid());

    tensor.set_uid(42);
    EXPECT_EQ(tensor.get_uid(), 42);
    EXPECT_TRUE(tensor.has_uid());

    tensor.clear_uid();
    EXPECT_EQ(tensor.get_uid(), 0);
    EXPECT_FALSE(tensor.has_uid());
}

// Copying carries the full configuration, including UID identity — there is no
// shim-side per-tensor state that a copy could drop or alias.
TEST(TestCudnnShimTensor, IdentityPreservedAcrossCopies)
{
    fe::graph::Tensor_attributes original;
    original.set_dim({2, 3})
        .set_stride({3, 1})
        .set_data_type(fe::DataType_t::FLOAT)
        .set_uid(99)
        .set_name("t");

    const fe::graph::Tensor_attributes copyConstructed(original);
    EXPECT_EQ(copyConstructed.get_uid(), 99);
    EXPECT_TRUE(copyConstructed.has_uid());
    EXPECT_EQ(copyConstructed.get_name(), "t");
    EXPECT_EQ(copyConstructed.get_dim(), std::vector<int64_t>({2, 3}));
    EXPECT_EQ(copyConstructed.get_data_type(), fe::DataType_t::FLOAT);

    fe::graph::Tensor_attributes copyAssigned;
    copyAssigned = original;
    EXPECT_EQ(copyAssigned.get_uid(), 99);
    EXPECT_TRUE(copyAssigned.has_uid());
    EXPECT_EQ(copyAssigned.get_name(), "t");

    // A reference to the shim type binds a hipDNN value with no conversion,
    // because they are the same type.
    const hipdnn_frontend::graph::TensorAttributes& asNative = original;
    EXPECT_EQ(asNative.get_uid(), 99);
}

} // namespace
