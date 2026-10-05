/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// CPU wiring/regression test for the precomputed config-tower embedding tables (SILOTIGER-1145):
// confirm each shipped gfx950 {solver}_kernel_config_embeddings.bin has the expected format and that
// TryEncodeKernelConfigsFromTable returns valid 64-d embeddings for known configs. The deep fp16-vs-
// fp32 ranking parity is validated offline against real FillValidKernels; this guards the ship/
// install/load/lookup path in CI without a GPU.

#include <gtest/gtest.h>
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/conv/heuristics/ai_conv_nd_kernel_tuning_utils.hpp>
#include <miopen/db_path.hpp>
#include <miopen/filesystem.hpp>

#include <cmath>
#include <cstdint>
#include <fstream>
#include <numeric>
#include <string>
#include <vector>

#if MIOPEN_ENABLE_AI_KERNEL_TUNING

namespace {

namespace cs = miopen::ai::tuning::candidate_selection;

struct TableCase
{
    const char* arch;
    const char* solver;
    std::vector<std::string> configs; // known kernel strings (no split_k) present in the table
};

const std::vector<TableCase>& Cases()
{
    static const std::string fwd =
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 16, Default, 32, 32, 4, 2, 4, "
        "4, 4, 1, 1, 1>";
    static const std::string fwd_pad0 =
        "DeviceGroupedConvFwdMultipleABD_Xdl_CShuffle<128, 128, 128, 16, Filter1x1Pad0, 32, 32, 4, "
        "2, 4, 4, 4, 1, 1, 1>";
    static const std::string bwd3d =
        "DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffleV3_Large_Tensor<256, 128, 128, 32, 8, 8, "
        "Default, 32, 32, 2, 2, 1, 1, 1, 1>";
    static const std::vector<TableCase> cases = {
        {"gfx950", "ConvHipImplicitGemmGroupFwdXdlops", {fwd, fwd_pad0}},
        {"gfx950", "ConvHipImplicitGemm3DGroupFwdXdlops", {fwd}},
        {"gfx950", "ConvHipImplicitGemm3DGroupBwdXdlops", {bwd3d}},
        {"gfx942", "ConvHipImplicitGemmGroupFwdXdlops", {fwd, fwd_pad0}},
        {"gfx942", "ConvHipImplicitGemm3DGroupBwdXdlops", {bwd3d}},
    };
    return cases;
}

std::string TablePath(const std::string& arch, const std::string& solver)
{
    return (miopen::GetSystemDbPath() / (arch + "_" + solver + "_kernel_config_embeddings.bin"))
        .string();
}

} // namespace

class CPU_ConfigEmbeddingTable : public ::testing::TestWithParam<TableCase>
{
};

TEST_P(CPU_ConfigEmbeddingTable, ShipsLoadsAndHits)
{
    const auto& tc         = GetParam();
    const std::string arch = tc.arch;
    const auto path        = TablePath(arch, tc.solver);
    if(!miopen::fs::exists(path))
        GTEST_SKIP() << "table not installed: " << path;

    // Format header: "MICE" | u32 version=1 | key_dim | emb_dim=64 | rows | key_dtype=0 | emb_dtype=1
    {
        std::ifstream is(path, std::ios::binary);
        char magic[4] = {};
        is.read(magic, 4);
        std::uint32_t ver = 0, key_dim = 0, emb_dim = 0, rows = 0, kdt = 9, edt = 9;
        auto g = [&](std::uint32_t& v) { is.read(reinterpret_cast<char*>(&v), sizeof(v)); };
        g(ver);
        g(key_dim);
        g(emb_dim);
        g(rows);
        g(kdt);
        g(edt);
        ASSERT_TRUE(is.good());
        EXPECT_EQ(std::string(magic, 4), "MICE");
        EXPECT_EQ(ver, 1u);
        EXPECT_EQ(emb_dim, 64u);
        EXPECT_EQ(kdt, 0u); // fp32 keys
        EXPECT_EQ(edt, 1u); // fp16 embeddings
        EXPECT_GT(rows, 0u);
    }

    // Functional: known configs must hit the table and yield finite 64-d embeddings.
    const auto& model = cs::GetCandidateSelectionModel(arch, tc.solver);
    std::vector<int> indexes;
    std::vector<std::vector<std::string>> heuristic_kernels;
    miopen::solver::conv::FillHeuristicKernels(tc.configs, indexes, heuristic_kernels);
    ASSERT_FALSE(heuristic_kernels.empty());

    const auto encoded = cs::EncodeKernelParams(heuristic_kernels, model.metadata(), false);
    ASSERT_EQ(encoded.size(), tc.configs.size());

    auto embeddings = cs::TryEncodeKernelConfigsFromTable(encoded, arch, tc.solver);
    ASSERT_TRUE(embeddings.has_value()) << "known configs did not hit the shipped table";
    ASSERT_EQ(embeddings->size(), encoded.size());
    for(const auto& e : *embeddings)
    {
        EXPECT_EQ(e.size(), 64u);
        for(float v : e)
            EXPECT_TRUE(std::isfinite(v)) << "known-config embedding has a non-finite value";
    }
}

INSTANTIATE_TEST_SUITE_P(Full, CPU_ConfigEmbeddingTable, ::testing::ValuesIn(Cases()));

#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
