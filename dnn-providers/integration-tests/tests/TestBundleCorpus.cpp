// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <gtest/gtest.h>

#include <filesystem>
#include <set>
#include <string>
#include <variant>

#include <hipdnn_flatbuffers_sdk/data_objects/graph_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_backward_attributes_generated.h>
#include <hipdnn_test_sdk/utilities/cpu_graph_executor/detail/PlanUtils.hpp>

#include "harness/bundle/BundleDiscovery.hpp"
#include "harness/bundle/IntegrationTestBundle.hpp"

using namespace hipdnn_integration_tests::bundle;

namespace
{

constexpr size_t MIN_SDPA_SUITES = 66;

std::filesystem::path bundleRoot()
{
    return std::filesystem::path(__FILE__).parent_path() / ".." / "integration-test-bundles";
}

} // namespace

// The harness drops a bundle that fails to load without registering a test, and only checks
// the SDPA mask when the reference runs, so neither failure shows up in a green bundle run.
TEST(TestSdpaBundleCorpus, EverySdpaBundleLoadsWithAValidMask)
{
    std::set<std::string> sdpaSuites;
    for(const auto& discovered : discoverBundles(bundleRoot()))
    {
        if(discovered.suiteName.find("Sdpa") == std::string::npos)
        {
            continue;
        }
        sdpaSuites.insert(discovered.suiteName);
        SCOPED_TRACE(discovered.diagnosticPath().string());

        const auto loaded = loadIntegrationTestBundle(discovered);
        if(const auto* error = std::get_if<LoadError>(&loaded))
        {
            ADD_FAILURE() << "bundle failed to load: " << toString(*error);
            continue;
        }

        const auto wrapper = std::get<IntegrationTestBundle>(loaded).graphWrapper();
        for(uint32_t nodeIndex = 0; nodeIndex < wrapper.nodeCount(); ++nodeIndex)
        {
            const auto& node = wrapper.getNode(nodeIndex);
            if(const auto* fwdAttrs = node.attributes_as_SdpaAttributes())
            {
                EXPECT_NO_THROW(extractDiagonalBandParams(*fwdAttrs, "TestSdpaBundleCorpus"));
            }
            if(const auto* bwdAttrs = node.attributes_as_SdpaBackwardAttributes())
            {
                EXPECT_NO_THROW(extractDiagonalBandParams(*bwdAttrs, "TestSdpaBundleCorpus"));
            }
        }
    }

    EXPECT_GE(sdpaSuites.size(), MIN_SDPA_SUITES);
}
