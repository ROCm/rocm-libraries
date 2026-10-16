// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Checks the cuDNN v8 shim attribute table against the identifier and
// attribute-usage fixtures extracted from upstream cudnn-frontend.
#include <hipdnn_compatibility/cudnn/cudnn_graph.h>

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace
{

namespace backend = hipdnn_frontend::compatibility::cudnn_frontend::detail::backend;

template <typename Wanted, typename T>
void addIfType(std::vector<std::pair<Wanted, std::string>>& out, T value, const char* name)
{
    if constexpr(std::is_same_v<Wanted, T>)
    {
        out.emplace_back(value, name);
    }
}

template <typename Wanted>
std::vector<std::pair<Wanted, std::string>> fixtureEnumerators()
{
    std::vector<std::pair<Wanted, std::string>> result;
#define HIPDNN_CUDNN_SHIM_ENUMERATOR(T, NAME) addIfType<Wanted>(result, NAME, #NAME);
#define HIPDNN_CUDNN_SHIM_TYPE(...)
#define HIPDNN_CUDNN_SHIM_FUNCTION(...)
#define HIPDNN_CUDNN_SHIM_MACRO(...)
#define HIPDNN_CUDNN_SHIM_ALREADY_DECLARED(...)
#include "data/cudnn_v8/cudnn_v8_identifiers.inc"
#undef HIPDNN_CUDNN_SHIM_ENUMERATOR
#undef HIPDNN_CUDNN_SHIM_TYPE
#undef HIPDNN_CUDNN_SHIM_FUNCTION
#undef HIPDNN_CUDNN_SHIM_MACRO
#undef HIPDNN_CUDNN_SHIM_ALREADY_DECLARED
    return result;
}

struct UsagePair
{
    cudnnBackendAttributeName_t attribute;
    cudnnBackendAttributeType_t type;
    const char* attributeName;
    const char* typeName;
};

std::vector<UsagePair> fixtureUsage()
{
    return {
#define HIPDNN_CUDNN_SHIM_ATTR_USAGE(ATTR, TYPE) {ATTR, TYPE, #ATTR, #TYPE},
#include "data/cudnn_v8/cudnn_v8_attribute_usage.inc"
#undef HIPDNN_CUDNN_SHIM_ATTR_USAGE
    };
}

TEST(TestCudnnShimV8AttributeTable, EveryFixtureAttributeHasRow)
{
    const auto attributes = fixtureEnumerators<cudnnBackendAttributeName_t>();
    ASSERT_FALSE(attributes.empty());
    for(const auto& [attribute, name] : attributes)
    {
        const std::optional<backend::AttributeInfo> info = backend::attributeInfo(attribute);
        ASSERT_TRUE(info.has_value()) << name << " has no attribute-table row";
        EXPECT_EQ(std::string(info->name), name);
        EXPECT_NE(std::string(backend::descriptorTypeName(info->owner)),
                  "unknown cudnnBackendDescriptorType_t")
            << name;
        EXPECT_GE(info->typeCount, 1);
        EXPECT_LE(info->typeCount, 2);
    }
}

TEST(TestCudnnShimV8AttributeTable, EveryUsagePairAccepted)
{
    const auto usage = fixtureUsage();
    ASSERT_FALSE(usage.empty());
    for(const UsagePair& pair : usage)
    {
        const std::optional<backend::AttributeInfo> info = backend::attributeInfo(pair.attribute);
        ASSERT_TRUE(info.has_value()) << pair.attributeName;
        EXPECT_TRUE(info->accepts(pair.type)) << pair.attributeName << " rejects " << pair.typeName;
    }
}

TEST(TestCudnnShimV8AttributeTable, DescriptorValuedRowsHaveReferentRule)
{
    for(const auto& [attribute, name] : fixtureEnumerators<cudnnBackendAttributeName_t>())
    {
        const backend::AttributeInfo info = *backend::attributeInfo(attribute);
        const bool descriptorValued = info.accepts(CUDNN_TYPE_BACKEND_DESCRIPTOR);
        EXPECT_EQ(descriptorValued, info.referentKind != backend::ReferentKind::NONE) << name;
        if(info.referentKind == backend::ReferentKind::EXACT)
        {
            EXPECT_NE(std::string(backend::descriptorTypeName(info.referent)),
                      "unknown cudnnBackendDescriptorType_t")
                << name;
        }
    }
}

TEST(TestCudnnShimV8AttributeTable, CountBoundsAreConsistent)
{
    for(const auto& [attribute, name] : fixtureEnumerators<cudnnBackendAttributeName_t>())
    {
        const backend::AttributeInfo info = *backend::attributeInfo(attribute);
        if(info.countRule == backend::CountRule::SCALAR)
        {
            EXPECT_EQ(info.maxCount, 1) << name;
        }
        else
        {
            EXPECT_GE(info.maxCount, 1) << name;
        }
        if(info.countRule == backend::CountRule::CHAR_STRING)
        {
            EXPECT_TRUE(info.accepts(CUDNN_TYPE_CHAR)) << name;
        }
    }
}

TEST(TestCudnnShimV8AttributeTable, DescriptorTypeNamesMatchFixture)
{
    for(const auto& [type, name] : fixtureEnumerators<cudnnBackendDescriptorType_t>())
    {
        EXPECT_EQ(std::string(backend::descriptorTypeName(type)), name);
    }
}

TEST(TestCudnnShimV8AttributeTable, AttributeTypeNamesMatchFixture)
{
    for(const auto& [type, name] : fixtureEnumerators<cudnnBackendAttributeType_t>())
    {
        EXPECT_EQ(std::string(backend::attributeTypeName(type)), name);
        EXPECT_GT(backend::elementSize(type), 0U) << name;
    }
}

TEST(TestCudnnShimV8AttributeTable, OperationClassificationMatchesPrefix)
{
    for(const auto& [type, name] : fixtureEnumerators<cudnnBackendDescriptorType_t>())
    {
        const bool expected = name.rfind("CUDNN_BACKEND_OPERATION_", 0) == 0;
        EXPECT_EQ(backend::isOperationDescriptorType(type), expected) << name;
    }
}

} // namespace
