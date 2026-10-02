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

// Offline generator for the precomputed config-tower embedding tables (see the
// precompute-config-tower design, JIRA SILOTIGER-1145).
//
// The config tower is a pure function of (arch, solver, kernel-config): its output does not
// depend on the convolution problem, and the set of configs a solver can emit is finite. This
// generator enumerates that full universe with the *same* runtime helpers the heuristic uses
// (CkImplLibLoader::GetAllKernelTypeStrings -> FillHeuristicKernels -> [ExpandKernelParamsWithSplitK]
// -> EncodeKernelParams -> CandidateSelectionModel::EncodeKernelConfigs), so the emitted
// embeddings are byte-identical to what runs today, and writes one table per solver:
//
//     {arch}_{solver}_kernel_config_embeddings.bin
//
// It only runs when MIOPEN_GEN_CONFIG_EMB_DIR points at an output directory, so it is a no-op
// in normal CI. gfx950 only for now.
//
// Run (inside a build with the gfx950 models + CK impl lib installed):
//     MIOPEN_GEN_CONFIG_EMB_DIR=/tmp/emb ./bin/test_conv_ai_config_embedding_gen

#include <gtest/gtest.h>

#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/conv/heuristics/ai_conv_nd_kernel_tuning_utils.hpp>
#include <miopen/solver/ck_impl_lib_loader.hpp>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <set>
#include <string>
#include <vector>

#if MIOPEN_ENABLE_AI_KERNEL_TUNING

namespace {

using miopen::ai::tuning::candidate_selection::CandidateSelectionModel;
using miopen::ai::tuning::candidate_selection::EncodeKernelParams;
using miopen::ai::tuning::candidate_selection::ExpandKernelParamsWithSplitK;
using miopen::ai::tuning::candidate_selection::GetCandidateSelectionModel;
using miopen::solver::CKSolverType;
using miopen::solver::CkImplLibLoader;

// The six gfx950 candidate-selection solvers. solver_name must match the shipped model filenames
// ({arch}_{solver}_*.tn.model); uses_split_k mirrors each solver's SolverHeuristicConfig.
struct GenSpec
{
    CKSolverType ck;
    const char* solver_name;
    bool uses_split_k;
};

const std::vector<GenSpec>& Specs()
{
    static const std::vector<GenSpec> specs = {
        {CKSolverType::GrpConvFwd, "ConvHipImplicitGemmGroupFwdXdlops", false},
        {CKSolverType::GrpConvBwd, "ConvHipImplicitGemmGroupBwdXdlops", true},
        {CKSolverType::GrpConvWrw, "ConvHipImplicitGemmGroupWrwXdlops", true},
        {CKSolverType::GrpConv3dFwd, "ConvHipImplicitGemm3DGroupFwdXdlops", false},
        {CKSolverType::GrpConv3dBwd, "ConvHipImplicitGemm3DGroupBwdXdlops", false},
        {CKSolverType::GrpConv3dWrw, "ConvHipImplicitGemm3DGroupWrwXdlops", true},
    };
    return specs;
}

// IEEE-754 binary32 -> binary16, round-to-nearest-even. Embeddings are only consumed by a ranking
// dot product, so half precision is ample; this keeps the shipped table ~2x smaller than fp32.
std::uint16_t FloatToHalf(float f)
{
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    const std::int32_t exp   = static_cast<std::int32_t>((x >> 23) & 0xFF) - 127 + 15;
    const std::uint32_t mant = x & 0x7FFFFFu;

    if(((x >> 23) & 0xFF) == 0xFF) // Inf / NaN
        return static_cast<std::uint16_t>(sign | 0x7C00u | (mant != 0 ? 0x200u : 0u));
    if(exp >= 0x1F) // overflow -> Inf
        return static_cast<std::uint16_t>(sign | 0x7C00u);
    if(exp <= 0) // subnormal / underflow
    {
        if(exp < -10)
            return static_cast<std::uint16_t>(sign);
        std::uint32_t m = (mant | 0x800000u) >> static_cast<std::uint32_t>(1 - exp);
        if((m & 0x1000u) != 0) // round to nearest even
            m += 0x2000u;
        return static_cast<std::uint16_t>(sign | (m >> 13));
    }
    std::uint32_t half = sign | (static_cast<std::uint32_t>(exp) << 10) | (mant >> 13);
    if((mant & 0x1000u) != 0) // round to nearest even
        half += 1;
    return static_cast<std::uint16_t>(half);
}

template <typename T>
void Put(std::ofstream& os, T v)
{
    os.write(reinterpret_cast<const char*>(&v), sizeof(T));
}

// Table layout (little-endian):
//   magic 'M','I','C','E' | u32 version | u32 key_dim | u32 emb_dim | u32 num_rows
//   | u32 key_dtype(0=fp32) | u32 emb_dtype(1=fp16)
//   then num_rows x [ key_dim x fp32 (the EncodeKernelParams vector) | emb_dim x fp16 ]
// Keyed by the exact encoded-param vector so lookups are collision-free; the loader builds a
// hash map from it at load time.
void WriteTable(const std::string& path,
                const std::vector<std::vector<float>>& keys,
                const std::vector<std::vector<float>>& embeddings)
{
    ASSERT_EQ(keys.size(), embeddings.size());
    ASSERT_FALSE(keys.empty());
    const std::uint32_t key_dim = static_cast<std::uint32_t>(keys.front().size());
    const std::uint32_t emb_dim = static_cast<std::uint32_t>(embeddings.front().size());

    std::ofstream os(path, std::ios::binary | std::ios::trunc);
    ASSERT_TRUE(os.good()) << "cannot open " << path;
    os.write("MICE", 4);
    Put<std::uint32_t>(os, 1);
    Put<std::uint32_t>(os, key_dim);
    Put<std::uint32_t>(os, emb_dim);
    Put<std::uint32_t>(os, static_cast<std::uint32_t>(keys.size()));
    Put<std::uint32_t>(os, 0); // key dtype fp32
    Put<std::uint32_t>(os, 1); // emb dtype fp16
    for(std::size_t i = 0; i < keys.size(); ++i)
    {
        ASSERT_EQ(keys[i].size(), key_dim);
        ASSERT_EQ(embeddings[i].size(), emb_dim);
        for(float k : keys[i])
            Put<float>(os, k);
        for(float e : embeddings[i])
            Put<std::uint16_t>(os, FloatToHalf(e));
    }
}

// Enumerate the full config universe for one solver and encode it through the config tower,
// deduplicating identical encoded rows. Returns (keys, embeddings).
void BuildTableForSolver(const std::string& arch,
                         const GenSpec& spec,
                         std::vector<std::vector<float>>& out_keys,
                         std::vector<std::vector<float>>& out_emb)
{
    const auto& loader       = CkImplLibLoader::Get(arch);
    const auto all_kernels   = loader.GetAllKernelTypeStrings(spec.ck);
    ASSERT_FALSE(all_kernels.empty()) << spec.solver_name << ": no kernels enumerated";

    std::vector<int> indexes;
    std::vector<std::vector<std::string>> heuristic_kernels;
    miopen::solver::conv::FillHeuristicKernels(all_kernels, indexes, heuristic_kernels);
    ASSERT_FALSE(heuristic_kernels.empty());

    const auto& model = GetCandidateSelectionModel(arch, spec.solver_name);

    std::vector<std::vector<std::string>> params;
    if(spec.uses_split_k)
    {
        const auto& split_ks = model.metadata().GetSplitKValues();
        std::vector<int> hidx(heuristic_kernels.size());
        std::iota(hidx.begin(), hidx.end(), 0);
        auto always_valid        = [](int, int) { return true; };
        auto [expanded, mapping] = ExpandKernelParamsWithSplitK(
            heuristic_kernels, hidx, split_ks, std::move(always_valid));
        params = std::move(expanded);
    }
    else
    {
        params = heuristic_kernels;
    }

    const auto encoded = EncodeKernelParams(params, model.metadata(), spec.uses_split_k);
    const auto emb     = model.EncodeKernelConfigs(encoded);
    ASSERT_EQ(encoded.size(), emb.size());

    // Dedup identical encoded rows (distinct kernel strings can encode identically).
    std::set<std::vector<float>> seen;
    for(std::size_t i = 0; i < encoded.size(); ++i)
    {
        if(seen.insert(encoded[i]).second)
        {
            out_keys.push_back(encoded[i]);
            out_emb.push_back(emb[i]);
        }
    }
}

} // namespace

TEST(GPU_ConfigEmbeddingGen_FP32, Gfx950)
{
    const char* dir = std::getenv("MIOPEN_GEN_CONFIG_EMB_DIR");
    if(dir == nullptr)
        GTEST_SKIP() << "set MIOPEN_GEN_CONFIG_EMB_DIR to emit tables";

    const std::string arch = "gfx950";
    for(const auto& spec : Specs())
    {
        std::vector<std::vector<float>> keys, emb;
        BuildTableForSolver(arch, spec, keys, emb);
        const std::string path =
            std::string(dir) + "/" + arch + "_" + spec.solver_name + "_kernel_config_embeddings.bin";
        WriteTable(path, keys, emb);

        const std::size_t key_dim = keys.front().size();
        const std::size_t emb_dim = emb.front().size();
        const std::size_t bytes   = 28 + keys.size() * (key_dim * 4 + emb_dim * 2);
        std::cout << spec.solver_name << ": rows=" << keys.size() << " key_dim=" << key_dim
                  << " emb_dim=" << emb_dim << " table=" << (bytes / 1024) << " KiB -> " << path
                  << std::endl;
    }
}

#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
