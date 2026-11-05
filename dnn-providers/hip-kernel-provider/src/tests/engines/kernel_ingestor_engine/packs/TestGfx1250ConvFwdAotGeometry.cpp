// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>

#include <gtest/gtest.h>

#include "engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdGeometry.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdAotTestCatalog.hpp"
#include "tests/engines/kernel_ingestor_engine/packs/Gfx1250ConvFwdAotTestVectors.hpp"

/**
 * @file TestGfx1250ConvFwdAotGeometry.cpp
 * @brief hipkernel:Gfx1250ConvFwdAot's port of rocKE's dispatcher, on every catalog
 *        instance: each admits and plans exactly what the dispatcher does.
 *
 * The engine decides per graph which kernels serve it and how each launches, through its
 * C++ port of rocKE's dispatcher (Gfx1250ConvFwdGeometry.hpp) over each kernel's KMD
 * fields. Gfx1250ConvFwdAotTestVectors.hpp, generated through the packaging adapter
 * (make_gfx1250_conv_fwd_aot_test_vectors.py --header), holds every instance's KMD fields,
 * a problem set and, per instance, how many problems it admits and an FNV-1a digest of
 * every admitted problem's launch plan (grid, block, every kernel argument, workspaces).
 * This suite decodes each instance as the engine's prepare step does and recomputes both,
 * so a port that drifts from the committed vectors fails here rather than launching a wrong
 * grid. It reads only that header: a list or dispatcher change that leaves the header stale
 * is caught by the tooling test
 * test_make_gfx1250_conv_fwd_aot_test_vectors.py::test_the_golden_header_regenerates_from_the_in_tree_dispatcher.
 */
namespace
{

using namespace hip_kernel_provider::kernel_ingestor_engine::testing::gfx1250_conv_fwd_aot;

constexpr uint64_t FNV_OFFSET = 0xCBF29CE484222325ULL;
constexpr uint64_t FNV_PRIME = 0x100000001B3ULL;

uint64_t fnv1a(uint64_t digest, const std::string& text)
{
    for(const auto byte : text)
    {
        digest = (digest ^ static_cast<uint8_t>(byte)) * FNV_PRIME;
    }
    return digest;
}
/// The plan as the generator's canonical_plan() spells it: per step
/// `kernel|gx,gy,gz|bx|args;`, each argument `name=i:<value>`, `name=p:<buffer>+<offset>`
/// or `name=b:<buffer>`, then `ws:` and the workspaces in name order.
std::string canonicalPlan(const conv::Plan& plan)
{
    std::string out;
    for(const auto& step : plan.steps)
    {
        out += step.kernel + "|" + std::to_string(step.grid[0]) + "," + std::to_string(step.grid[1])
               + "," + std::to_string(step.grid[2]) + "|" + std::to_string(step.block[0]) + "|";
        for(size_t i = 0; i < step.args.size(); ++i)
        {
            const auto& arg = step.args[i];
            out += (i == 0 ? "" : ",") + arg.name + "=";
            if(arg.kind == conv::Arg::Kind::POINTER)
            {
                out += "p:" + arg.buffer + "+" + std::to_string(arg.value);
            }
            else if(arg.kind == conv::Arg::Kind::BYTES)
            {
                out += "b:" + arg.buffer;
            }
            else
            {
                out += "i:" + std::to_string(arg.value);
            }
        }
        out += ';';
    }
    out += "ws:";
    bool first = true;
    for(const auto& [name, bytes] : plan.workspaces)
    {
        out += (first ? "" : ",") + name + "=" + std::to_string(bytes);
        first = false;
    }
    return out;
}

TEST(TestGfx1250ConvFwdAotGeometry, EveryInstancePlansWhatTheDispatcherPlans)
{
    ASSERT_FALSE(vectors::KERNELS.empty());
    for(const auto& expected : vectors::KERNELS)
    {
        const auto geometry = instanceGeometry(expected);
        const auto kernel = conv::CatalogKernel::parse(geometry);
        uint64_t digest = FNV_OFFSET;
        int64_t admitted = 0;
        for(size_t index = 0; index < vectors::PROBLEMS.size(); ++index)
        {
            const auto problem = problemAt(index);
            if(problem.dtype != geometry.dtype() || kernel.refuse(problem).has_value())
            {
                continue;
            }
            digest = fnv1a(digest,
                           std::to_string(index) + ":" + canonicalPlan(kernel.plan(problem)) + ";");
            ++admitted;
        }
        EXPECT_EQ(admitted, expected.admitted) << "instance " << expected.instanceId
                                               << " admits another problem set than the dispatcher";
        EXPECT_EQ(digest, expected.digest)
            << "instance " << expected.instanceId << " plans another launch than the dispatcher";
    }
}

TEST(TestGfx1250ConvFwdAotGeometry, EveryInstanceIsOneOfItsOwn)
{
    std::set<std::string> ids;
    for(const auto& expected : vectors::KERNELS)
    {
        EXPECT_EQ(instanceGeometry(expected).label(), expected.instanceId);
        EXPECT_TRUE(ids.insert(expected.instanceId).second)
            << expected.instanceId << " is listed twice";
    }
}

} // namespace
