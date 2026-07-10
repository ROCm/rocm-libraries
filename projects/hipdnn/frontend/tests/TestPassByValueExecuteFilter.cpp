// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Unit tests for detail::validatePassByValueVariantPack (RFC-0016 execute-time
// filter). The only hard requirement is that a PURE runtime pass-by-value tensor
// (flag set, no baked value/default) MUST have a host-supplied scalar in the
// variant pack:
//   - pure runtime (flag set, no value) -> INVALID_VALUE if its uid is MISSING
//     (the host must supply the scalar).
//   - carries a value (compile-time const OR runtime-with-default) -> OK whether
//     or not its uid is in the pack; callers routinely include every uid and the
//     backend ignores the pointer for by-value tensors (baked value/default wins).
//   - ordinary (no flag, no value) -> ignored.
//   - all constraints satisfied -> OK.

#include <gtest/gtest.h>

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_frontend/detail/DescriptorHelpers.hpp>

using hipdnn_frontend::Error;
using hipdnn_frontend::ErrorCode;
using hipdnn_frontend::detail::validatePassByValueVariantPack;
using hipdnn_frontend::graph::ScalarType;
using hipdnn_frontend::graph::TensorAttributes;

namespace
{

using TensorSet = std::unordered_set<std::shared_ptr<TensorAttributes>>;
using VariantPack = std::unordered_map<int64_t, void*>;

// A dummy, non-null host pointer standing in for a real variant-pack device
// address. The filter only inspects key presence, never dereferences it.
void* const K_DUMMY_PTR = reinterpret_cast<void*>(0x1);

// Compile-time constant: value baked in, runtime flag clear.
std::shared_ptr<TensorAttributes> makeConstTensor(int64_t uid)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_value(3.14F);
    tensor->set_uid(uid);
    return tensor;
}

// Runtime-with-default: value present AND runtime flag set.
std::shared_ptr<TensorAttributes> makeRuntimeWithDefaultTensor(int64_t uid)
{
    auto tensor = std::make_shared<TensorAttributes>(2.71F, ScalarType::RUNTIME_PARAM);
    tensor->set_uid(uid);
    return tensor;
}

// Pure runtime pass-by-value: flag set, no value.
std::shared_ptr<TensorAttributes> makeRuntimeTensor(int64_t uid)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_as_runtime_parameter();
    tensor->set_uid(uid);
    return tensor;
}

// Ordinary tensor: no value, no runtime flag.
std::shared_ptr<TensorAttributes> makeOrdinaryTensor(int64_t uid)
{
    auto tensor = std::make_shared<TensorAttributes>();
    tensor->set_uid(uid);
    return tensor;
}

} // namespace

// A compile-time constant whose uid is present in the pack is accepted: the
// baked value wins and the extra variant-pack entry is harmlessly ignored.
TEST(PassByValueExecuteFilter, CompileTimeConstantInPackIsOk)
{
    const TensorSet tensors{makeConstTensor(1)};
    const VariantPack pack{{1, K_DUMMY_PTR}};

    const Error result = validatePassByValueVariantPack(tensors, pack);

    EXPECT_FALSE(result.is_bad());
    EXPECT_EQ(result.get_code(), ErrorCode::OK);
}

// A runtime-with-default tensor carries a baked default; supplying its uid in the
// pack is likewise accepted (the default wins today — RFC §4.9 limitation).
TEST(PassByValueExecuteFilter, RuntimeWithDefaultInPackIsOk)
{
    const TensorSet tensors{makeRuntimeWithDefaultTensor(2)};
    const VariantPack pack{{2, K_DUMMY_PTR}};

    const Error result = validatePassByValueVariantPack(tensors, pack);

    EXPECT_FALSE(result.is_bad());
    EXPECT_EQ(result.get_code(), ErrorCode::OK);
}

// A pure runtime pass-by-value tensor requires a host scalar; a missing uid is
// rejected.
TEST(PassByValueExecuteFilter, PureRuntimeMissingFromPackIsInvalid)
{
    const TensorSet tensors{makeRuntimeTensor(3)};
    const VariantPack pack{}; // uid 3 absent

    const Error result = validatePassByValueVariantPack(tensors, pack);

    EXPECT_TRUE(result.is_bad());
    EXPECT_EQ(result.get_code(), ErrorCode::INVALID_VALUE);
}

// A pure runtime pass-by-value tensor with its uid present is accepted.
TEST(PassByValueExecuteFilter, PureRuntimePresentInPackIsOk)
{
    const TensorSet tensors{makeRuntimeTensor(4)};
    const VariantPack pack{{4, K_DUMMY_PTR}};

    const Error result = validatePassByValueVariantPack(tensors, pack);

    EXPECT_FALSE(result.is_bad());
    EXPECT_EQ(result.get_code(), ErrorCode::OK);
}

// Ordinary tensors carry no by-value state and are ignored regardless of whether
// their uid appears in the pack.
TEST(PassByValueExecuteFilter, OrdinaryTensorPresentIsIgnored)
{
    const TensorSet tensors{makeOrdinaryTensor(5)};
    const VariantPack pack{{5, K_DUMMY_PTR}};

    const Error result = validatePassByValueVariantPack(tensors, pack);

    EXPECT_FALSE(result.is_bad());
    EXPECT_EQ(result.get_code(), ErrorCode::OK);
}

TEST(PassByValueExecuteFilter, OrdinaryTensorAbsentIsIgnored)
{
    const TensorSet tensors{makeOrdinaryTensor(6)};
    const VariantPack pack{}; // uid 6 absent

    const Error result = validatePassByValueVariantPack(tensors, pack);

    EXPECT_FALSE(result.is_bad());
    EXPECT_EQ(result.get_code(), ErrorCode::OK);
}

// A mixed set with every constraint satisfied: pure runtime present, value-bearing
// tensors may appear in the pack (harmlessly ignored), ordinary irrelevant.
TEST(PassByValueExecuteFilter, MixedAllSatisfiedIsOk)
{
    const TensorSet tensors{
        makeConstTensor(10), // baked; pack entry harmless
        makeRuntimeWithDefaultTensor(11), // baked default; pack entry harmless
        makeRuntimeTensor(12), // pure runtime, must be present
        makeOrdinaryTensor(13), // ignored
    };
    const VariantPack pack{
        {10, K_DUMMY_PTR}, // baked value present — allowed (ignored by backend)
        {11, K_DUMMY_PTR}, // baked default present — allowed
        {12, K_DUMMY_PTR}, // satisfies the pure runtime requirement
        {13, K_DUMMY_PTR}, // ordinary presence is irrelevant
    };

    const Error result = validatePassByValueVariantPack(tensors, pack);

    EXPECT_FALSE(result.is_bad());
    EXPECT_EQ(result.get_code(), ErrorCode::OK);
}

// A mixed set where a single tensor violates its constraint (the pure runtime
// tensor's uid is missing) is rejected even though the rest are satisfied.
TEST(PassByValueExecuteFilter, MixedSingleViolationIsBad)
{
    const TensorSet tensors{
        makeConstTensor(20), // baked; irrelevant to the filter
        makeRuntimeWithDefaultTensor(21), // baked default; irrelevant
        makeRuntimeTensor(22), // pure runtime, but uid 22 is MISSING
        makeOrdinaryTensor(23), // ignored
    };
    const VariantPack pack{
        {23, K_DUMMY_PTR}, // only the ordinary uid is present
    };

    const Error result = validatePassByValueVariantPack(tensors, pack);

    EXPECT_TRUE(result.is_bad());
    EXPECT_EQ(result.get_code(), ErrorCode::INVALID_VALUE);
}
