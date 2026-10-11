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

// CPU tests for the precomputed config-tower embedding tables (SILOTIGER-1145). No GPU: these lock
// the ship/install/load/lookup path and the behavior-preserving fallbacks that the reviewer
// flagged.
//   - every shipped {arch}_{solver}_kernel_config_embeddings.bin is present (fail, not skip),
//     well-formed (MICE v1, fp32 keys + fp32 embeddings, emb_dim 64, size matches the header), and
//     a real key read back out of the file looks up to the exact stored embedding (no
//     quantization);
//   - the all-NaN sentinel row returns a NaN embedding so unknown kernels rank last;
//   - a miss (dimension drift or an un-enumerated config) returns nullopt -> whole-batch fdeep;
//   - MIOPEN_DEBUG_AI_DISABLE_PRECOMPUTED_CONFIG_EMB forces the fdeep fallback;
//   - a truncated / size-mismatched file is rejected by the header check (not loaded).
// The table keys are the exact fp32 bytes of EncodeKernelParams output, so a key read from the file
// is a valid lookup input -- no kernel-config strings are hardcoded here.

#include <gtest/gtest.h>
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/db_path.hpp>
#include <miopen/filesystem.hpp>

#include "gtest_common.hpp" // ScopedEnvironment, MIOPEN_LIB_ENV_VAR

#include <unistd.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#if MIOPEN_ENABLE_AI_KERNEL_TUNING

MIOPEN_LIB_ENV_VAR(MIOPEN_SYSTEM_DB_PATH)
MIOPEN_LIB_ENV_VAR(MIOPEN_DEBUG_AI_DISABLE_PRECOMPUTED_CONFIG_EMB)

namespace {

namespace cs = miopen::ai::tuning::candidate_selection;

struct ArchSolver
{
    const char* arch;
    const char* solver;
};

const std::vector<ArchSolver>& AllTables()
{
    static const std::vector<ArchSolver> t = {
        {"gfx950", "ConvHipImplicitGemmGroupFwdXdlops"},
        {"gfx950", "ConvHipImplicitGemmGroupBwdXdlops"},
        {"gfx950", "ConvHipImplicitGemmGroupWrwXdlops"},
        {"gfx950", "ConvHipImplicitGemm3DGroupFwdXdlops"},
        {"gfx950", "ConvHipImplicitGemm3DGroupBwdXdlops"},
        {"gfx950", "ConvHipImplicitGemm3DGroupWrwXdlops"},
        {"gfx942", "ConvHipImplicitGemmGroupFwdXdlops"},
        {"gfx942", "ConvHipImplicitGemmGroupBwdXdlops"},
        {"gfx942", "ConvHipImplicitGemmGroupWrwXdlops"},
        {"gfx942", "ConvHipImplicitGemm3DGroupFwdXdlops"},
        {"gfx942", "ConvHipImplicitGemm3DGroupBwdXdlops"},
        {"gfx942", "ConvHipImplicitGemm3DGroupWrwXdlops"},
    };
    return t;
}

std::string TablePath(const std::string& arch, const std::string& solver)
{
    return (miopen::GetSystemDbPath() / (arch + "_" + solver + "_kernel_config_embeddings.bin"))
        .string();
}

struct ParsedTable
{
    bool ok               = false;
    std::uint32_t key_dim = 0, emb_dim = 0, num_rows = 0;
    std::vector<std::vector<float>> keys, embs; // fp32
};

// Mirror of the loader's header parse; also used to verify file size matches the header exactly.
ParsedTable ParseTable(const std::string& path)
{
    ParsedTable p;
    std::ifstream is(path, std::ios::binary);
    if(!is.good())
        return p;
    char magic[4] = {};
    is.read(magic, 4);
    std::uint32_t ver = 0, kd = 0, ed = 0, nr = 0, kdt = 9, edt = 9;
    auto g = [&](std::uint32_t& v) { is.read(reinterpret_cast<char*>(&v), sizeof(v)); };
    g(ver);
    g(kd);
    g(ed);
    g(nr);
    g(kdt);
    g(edt);
    if(!is.good() || std::memcmp(magic, "MICE", 4) != 0 || ver != 1 || kdt != 0 || edt != 0 ||
       kd == 0 || ed == 0)
        return p;
    p.key_dim  = kd;
    p.emb_dim  = ed;
    p.num_rows = nr;
    for(std::uint32_t r = 0; r < nr; ++r)
    {
        std::vector<float> key(kd), emb(ed);
        is.read(reinterpret_cast<char*>(key.data()), static_cast<std::streamsize>(kd * 4));
        is.read(reinterpret_cast<char*>(emb.data()), static_cast<std::streamsize>(ed * 4));
        if(!is.good())
            return p;
        p.keys.push_back(std::move(key));
        p.embs.push_back(std::move(emb));
    }
    // There must be no trailing bytes: file size == header + rows.
    const auto trailing = is.get();
    p.ok                = (p.keys.size() == nr) && (trailing == std::char_traits<char>::eof());
    return p;
}

bool AllNaN(const std::vector<float>& v)
{
    for(float x : v)
        if(!std::isnan(x))
            return false;
    return !v.empty();
}

// First non-sentinel (not all-NaN) key in the table; returns index or npos.
std::size_t FirstRealRow(const ParsedTable& pt)
{
    for(std::size_t i = 0; i < pt.keys.size(); ++i)
        if(!AllNaN(pt.keys[i]))
            return i;
    return static_cast<std::size_t>(-1);
}

} // namespace

class CPU_PrecomputeTable_NONE : public ::testing::TestWithParam<ArchSolver>
{
};

// Every shipped table is present, well-formed, and round-trips its own keys to the exact embedding.
TEST_P(CPU_PrecomputeTable_NONE, FormatAndLookup)
{
    const auto& t   = GetParam();
    const auto path = TablePath(t.arch, t.solver);
    // Fail (do not skip): with AI kernel tuning enabled the install must ship these tables.
    ASSERT_TRUE(miopen::fs::exists(path)) << "shipped embedding table missing: " << path;

    const auto pt = ParseTable(path);
    ASSERT_TRUE(pt.ok) << "malformed table or size mismatch: " << path;
    EXPECT_EQ(pt.emb_dim, 64u);
    EXPECT_GT(pt.num_rows, 0u);

    // Feed a handful of real keys from the file back through the loader; expect the exact stored
    // fp32 embedding (the keys are the raw EncodeKernelParams bytes, so this is a true lookup).
    std::vector<std::vector<float>> probe;
    std::vector<std::size_t> idx;
    for(std::size_t i = 0; i < pt.keys.size() && probe.size() < 5; ++i)
        if(!AllNaN(pt.keys[i]))
        {
            probe.push_back(pt.keys[i]);
            idx.push_back(i);
        }
    ASSERT_FALSE(probe.empty());

    auto emb = cs::TryEncodeKernelConfigsFromTable(probe, t.arch, t.solver);
    ASSERT_TRUE(emb.has_value()) << "known keys did not hit the table";
    ASSERT_EQ(emb->size(), probe.size());
    for(std::size_t j = 0; j < probe.size(); ++j)
        EXPECT_EQ((*emb)[j], pt.embs[idx[j]]);
}

INSTANTIATE_TEST_SUITE_P(Full, CPU_PrecomputeTable_NONE, ::testing::ValuesIn(AllTables()));

namespace {
// One representative, always-shipped table drives the behavior tests.
constexpr const char* kArch   = "gfx950";
constexpr const char* kSolver = "ConvHipImplicitGemm3DGroupFwdXdlops";
} // namespace

// The all-NaN sentinel row makes an unknown kernel (all-NaN encoding) hit and sort last.
TEST(CPU_PrecomputeTableBehaviors_NONE, SentinelReturnsNaN)
{
    const auto pt = ParseTable(TablePath(kArch, kSolver));
    ASSERT_TRUE(pt.ok);
    const std::vector<std::vector<float>> nan_cand = {
        std::vector<float>(pt.key_dim, std::numeric_limits<float>::quiet_NaN())};
    auto emb = cs::TryEncodeKernelConfigsFromTable(nan_cand, kArch, kSolver);
    ASSERT_TRUE(emb.has_value()) << "all-NaN sentinel key did not hit the table";
    ASSERT_EQ(emb->size(), 1u);
    EXPECT_TRUE(AllNaN(emb->front()));
}

// A dimension drift or an un-enumerated (even known-dimension) config misses -> nullopt.
TEST(CPU_PrecomputeTableBehaviors_NONE, MissFallsBack)
{
    const auto pt = ParseTable(TablePath(kArch, kSolver));
    ASSERT_TRUE(pt.ok);
    const auto real = FirstRealRow(pt);
    ASSERT_NE(real, static_cast<std::size_t>(-1));

    // Wrong key dimension.
    const std::vector<float> wrong_dim(pt.key_dim + 1, 0.0f);
    EXPECT_FALSE(cs::TryEncodeKernelConfigsFromTable({wrong_dim}, kArch, kSolver).has_value());

    // Right dimension, perturbed value -> not a stored key.
    std::vector<float> perturbed = pt.keys[real];
    perturbed[0] += 12345.0f;
    EXPECT_FALSE(cs::TryEncodeKernelConfigsFromTable({perturbed}, kArch, kSolver).has_value());

    // One miss aborts the whole batch, even mixed with a real hit (whole-batch fallback policy).
    EXPECT_FALSE(cs::TryEncodeKernelConfigsFromTable({pt.keys[real], perturbed}, kArch, kSolver)
                     .has_value());
}

// The disable env var forces the fdeep path (nullopt) even for a known key.
TEST(CPU_PrecomputeTableBehaviors_NONE, DisableEnvForcesFallback)
{
    const auto pt = ParseTable(TablePath(kArch, kSolver));
    ASSERT_TRUE(pt.ok);
    const auto real = FirstRealRow(pt);
    ASSERT_NE(real, static_cast<std::size_t>(-1));

    ScopedEnvironment<bool> disable(MIOPEN_DEBUG_AI_DISABLE_PRECOMPUTED_CONFIG_EMB, true);
    EXPECT_FALSE(cs::TryEncodeKernelConfigsFromTable({pt.keys[real]}, kArch, kSolver).has_value());
}

// A truncated / size-mismatched .bin is rejected by the header check -> not loaded -> nullopt.
// Uses a fresh fake solver name (so it dodges the loader's per-(arch,solver) table cache) and a
// temp system-db dir pointed at by MIOPEN_SYSTEM_DB_PATH.
TEST(CPU_PrecomputeTableBehaviors_NONE, TruncatedFileRejected)
{
    namespace fs   = miopen::fs;
    const auto dir = fs::temp_directory_path() /
                     ("miopen_precompute_test_" + std::to_string(static_cast<long>(::getpid())));
    fs::create_directories(dir);
    const char* fake = "FakeTruncatedSolver";
    const auto path =
        (dir / (std::string(kArch) + "_" + fake + "_kernel_config_embeddings.bin")).string();
    {
        // Header claims 100 rows of a 16-key / 64-emb fp32 table but the body is nearly empty.
        std::ofstream os(path, std::ios::binary | std::ios::trunc);
        os.write("MICE", 4);
        const std::uint32_t hdr[] = {1u, 16u, 64u, 100u, 0u, 0u};
        os.write(reinterpret_cast<const char*>(hdr), sizeof(hdr));
        const float junk[8] = {};
        os.write(reinterpret_cast<const char*>(junk), sizeof(junk));
    }
    {
        ScopedEnvironment<std::string> sys_db(MIOPEN_SYSTEM_DB_PATH, dir.string());
        const std::vector<std::vector<float>> cand = {std::vector<float>(16, 1.0f)};
        EXPECT_FALSE(cs::TryEncodeKernelConfigsFromTable(cand, kArch, fake).has_value());
    }
    std::error_code ec;
    fs::remove_all(dir, ec);
}

#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
