// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include "ClientProblemFactory.hpp"

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/FusedA2AKernArg.hpp>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace TensileLite;

namespace
{
    ContractionProblemGemm fusedProblem(
        size_t M, size_t N, size_t ldd, uint32_t world, int64_t extent, size_t batch = 1)
    {
        constexpr size_t K = 2048;
        auto             p = ContractionProblemGemm::GEMM_Strides(true,
                                                      false,
                                                      rocisa::DataType::BFloat16,
                                                      rocisa::DataType::BFloat16,
                                                      rocisa::DataType::BFloat16,
                                                      rocisa::DataType::BFloat16,
                                                      M,
                                                      N,
                                                      K,
                                                      batch,
                                                      K,
                                                      K * M,
                                                      K,
                                                      K * N,
                                                      ldd,
                                                      ldd * N,
                                                      ldd,
                                                      ldd * N,
                                                      0.0);
        p.setFusedGemmA2A(true);
        p.setWorld(world);
        p.setFusedA2AExtent(extent);
        return p;
    }
}

TEST(FusedA2AProblem, ShapePredicate)
{
    Predicates::Contraction::FusedA2AShape mt256(256);
    EXPECT_TRUE(mt256(fusedProblem(3072, 512, 3072, 4, 2048)));
    EXPECT_TRUE(mt256(fusedProblem(3072, 512, 3072, 8, 2048)));
    // AM % (MT0 * W) != 0
    EXPECT_FALSE(mt256(fusedProblem(3072, 512, 3072, 8, 3072)));
    // AM > M
    EXPECT_FALSE(mt256(fusedProblem(3072, 512, 3072, 2, 4096)));
    // M % MT0 != 0
    EXPECT_FALSE(mt256(fusedProblem(3000, 512, 3000, 2, 2048)));
    // ldd % 8 != 0
    EXPECT_FALSE(mt256(fusedProblem(3072, 512, 3076, 2, 2048)));
    // (ldd >> 3) == 2^19
    EXPECT_FALSE(mt256(fusedProblem(3072, 512, size_t(1) << 22, 2, 2048)));
    // (nShard >> 3) == 16352 and 16384
    EXPECT_TRUE(mt256(fusedProblem(261632, 16, 261632, 2, 261632)));
    EXPECT_FALSE(mt256(fusedProblem(262144, 16, 262144, 2, 262144)));
    EXPECT_FALSE(mt256(fusedProblem(3072, 512, 3072, 2, 0)));
    EXPECT_FALSE(mt256(fusedProblem(3072, 512, 3072, 2, -2048)));
    EXPECT_FALSE(mt256(fusedProblem(3072, 512, 3072, 2, 2048, 2)));

    auto plain
        = ContractionProblemGemm::GEMM(false, true, 3000, 500, 100, 3000, 500, 3000, 0.0, false, 1);
    EXPECT_TRUE(mt256(plain));
}

namespace
{
    template <typename T>
    void setOption(Client::po::variables_map& args, std::string const& name, T value)
    {
        args[name].value() = std::move(value);
    }

    Client::po::variables_map factoryArgs(bool fused, size_t problemCount)
    {
        Client::po::variables_map args;
        setOption(args, "problem-identifier", std::string("Contraction_l_Alik_Bljk_Cijk_Dijk"));
        setOption(args,
                  "problem-size",
                  std::vector<std::vector<size_t>>(problemCount, {3072, 512, 1, 2048}));
        setOption(args, "strided-batched", true);
        setOption(args, "batch-mode", 0);
        setOption(args, "grouped-gemm", false);
        setOption(args, "sparse", 0);
        setOption(args, "high-precision-accumulate", false);
        setOption(args, "kernel-language", KernelLanguage::Any);
        setOption(args, "performance-metric", PerformanceMetric::DeviceEfficiency);
        setOption(args, "deterministic-mode", false);
        setOption(args, "c-equal-d", false);
        setOption(args, "type", rocisa::DataType::BFloat16);
        setOption(args, "mx-a-block", 0);
        setOption(args, "mx-b-block", 0);
        setOption(args, "mx-a-type", rocisa::DataType::E8);
        setOption(args, "mx-b-type", rocisa::DataType::E8);
        setOption(args, "fused-gemm-a2a", fused);
        setOption(args, "metadata-layout", 0);
        setOption(args, "a-ops", TensorOps{});
        setOption(args, "b-ops", TensorOps{});
        setOption(args, "c-ops", TensorOps{});
        setOption(args, "d-ops", TensorOps{});
        setOption(args, "use-bias", 0);
        setOption(args, "bias-source", static_cast<int>(ContractionProblemGemm::TENSOR::D));
        setOption(args, "use-scaleAlphaVec", 0);
        return args;
    }

    ContractionProblemGemm const& gemmAt(Client::ClientProblemFactory const& factory, size_t i)
    {
        return *std::dynamic_pointer_cast<ContractionProblemGemm>(factory.problems().at(i));
    }
}

TEST(FusedA2AClientProblemFactory, PerProblemWorldAndExtent)
{
    auto args = factoryArgs(true, 2);
    setOption(args, "world", std::vector<int>{2, 4});
    setOption(args, "fused-a2a-extent", std::vector<int>{2048, 2048});
    Client::ClientProblemFactory factory(args);
    ASSERT_EQ(factory.problems().size(), 2u);
    uint32_t const worlds[] = {2, 4};
    for(size_t i = 0; i < 2; i++)
    {
        auto const& p = gemmAt(factory, i);
        EXPECT_TRUE(p.fusedGemmA2A());
        EXPECT_EQ(p.world(), worlds[i]);
        EXPECT_EQ(p.fusedA2AExtent(), 2048);
        size_t const nShard = 2048 / worlds[i];
        auto const&  recv   = p.tensors()[ContractionProblemGemm::TENSOR::A2A_RECV];
        EXPECT_EQ(recv.sizes(), (std::vector<size_t>{nShard, 512, worlds[i]}));
        EXPECT_EQ(recv.strides(), (std::vector<size_t>{1, nShard, nShard * 512}));
    }
}

TEST(FusedA2AClientProblemFactory, InvalidWorldOrExtentThrows)
{
    struct Case
    {
        bool             fused;
        std::vector<int> worlds;
        std::vector<int> extents;
    };
    for(auto const& c : std::vector<Case>{{true, {2, 4}, {2048}},
                                          {true, {0}, {2048}},
                                          {true, {FUSED_A2A_MAX_RANKS + 1}, {2048}},
                                          {true, {2}, {0}},
                                          {false, {2}, {}}})
    {
        auto args = factoryArgs(c.fused, c.worlds.size());
        setOption(args, "world", c.worlds);
        setOption(args, "fused-a2a-extent", c.extents);
        EXPECT_THROW(Client::ClientProblemFactory{args}, std::invalid_argument)
            << c.fused << " " << c.worlds.at(0) << " " << c.extents.size();
    }
}

TEST(FusedA2AClientProblemFactory, NonFusedKeepsDefaultWorld)
{
    Client::ClientProblemFactory factory(factoryArgs(false, 1));
    ASSERT_EQ(factory.problems().size(), 1u);
    EXPECT_EQ(gemmAt(factory, 0).world(), 1u);
}
