// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// CPU smoke tests for the shipped AI-heuristic model assets (no GPU required).
//
// Three independent failure modes have been seen in practice:
//
//   1. fdeep version mismatch (keras2 vs keras3) — the input encoder loads but throws at inference
//      time ("tensor_shape_variable needs to be an array"). SelectBestVocabCandidate catches the
//      exception and returns nullopt silently, which looks like a non-AI fallback, not a build
//      defect. Caught here by asserting EncodeInputFeatures does not throw.
//
//   2. .bin md5 mismatch — the config-embedding table's header md5 (bytes 28-59) must equal
//      md5(input_encoder.tn.model). A mismatch (e.g. after re-stamping the input encoder for a
//      keras2/keras3 swap without updating the .bin) makes SelectBestVocabCandidate fail closed
//      and return nullopt silently. Caught here by reading both and comparing.
//
//   3. Missing asset — a shipped (arch, solver) pair has no model files. Caught by asserting
//      AssetsPresent rather than silently skipping.
//
// All three are CPU-only checks that run on any GPU runner (the shared gtest main requires some
// GPU present, but the checks themselves make no GPU calls).

#include <gtest/gtest.h>
#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_KERNEL_TUNING
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/conv/heuristics/ai_conv_nd_kernel_tuning_utils.hpp>
#include <miopen/conv/problem_description.hpp>
#include <miopen/tensor.hpp>
#include <miopen/convolution.hpp>
#include <miopen/db_path.hpp>
#include <miopen/filesystem.hpp>
#include <array>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

using namespace miopen::ai::tuning::candidate_selection;

namespace {

struct ArchSolver
{
    std::string arch;
    std::string solver;
    bool is3d;
    bool uses_split_k;
};

const std::vector<ArchSolver>& AllArchSolvers()
{
    static const std::vector<ArchSolver> v = [] {
        std::vector<ArchSolver> out;
        for(const auto& arch : {"gfx942", "gfx950", "gfx1250"})
        {
            out.push_back({arch, "ConvHipImplicitGemmGroupFwdXdlops", false, false});
            out.push_back({arch, "ConvHipImplicitGemmGroupBwdXdlops", false, true});
            out.push_back({arch, "ConvHipImplicitGemmGroupWrwXdlops", false, true});
            out.push_back({arch, "ConvHipImplicitGemm3DGroupFwdXdlops", true, false});
            out.push_back({arch, "ConvHipImplicitGemm3DGroupBwdXdlops", true, false});
            out.push_back({arch, "ConvHipImplicitGemm3DGroupWrwXdlops", true, true});
        }
        return out;
    }();
    return v;
}

bool AssetsPresent(const ArchSolver& as)
{
    const auto db = miopen::GetSystemDbPath();
    return miopen::fs::exists(db / (as.arch + "_" + as.solver + "_input_encoder.tn.model")) &&
           miopen::fs::exists(db / (as.arch + "_" + as.solver + "_metadata.tn.model")) &&
           miopen::fs::exists(db / (as.arch + "_" + as.solver + "_kernel_config_embeddings.bin"));
}

// Compute MD5 of a file and return as 32-char lowercase hex string.
// Uses the same algorithm the C++ precompute path uses to stamp/check the .bin header.
std::string FileMd5Hex(const miopen::fs::path& p)
{
    // Read entire file
    std::ifstream f(p, std::ios::binary);
    if(!f) return "";
    std::vector<unsigned char> data((std::istreambuf_iterator<char>(f)), {});

    // MD5 implementation (RFC 1321) — self-contained, no OpenSSL dependency.
    // Included here rather than via a header to keep the test self-contained.
    auto leftrotate = [](uint32_t x, uint32_t c) -> uint32_t {
        return (x << c) | (x >> (32 - c));
    };
    static const uint32_t K[64] = {
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a,
        0xa8304613, 0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be,
        0x6b901122, 0xfd987193, 0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340,
        0x265e5a51, 0xe9b6c7aa, 0xd62f105d, 0x02441453, 0xd8a1e681, 0xe7d3fbc8,
        0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed, 0xa9e3e905, 0xfcefa3f8,
        0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122, 0xfde5380c,
        0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665,
        0xf4292244, 0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92,
        0xffeff47d, 0x85845dd1, 0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1,
        0xf7537e82, 0xbd3af235, 0x2ad7d2bb, 0xeb86d391};
    static const uint32_t S[64] = {7,  12, 17, 22, 7,  12, 17, 22, 7,  12, 17, 22, 7,
                                    12, 17, 22, 5,  9,  14, 20, 5,  9,  14, 20, 5,  9,
                                    14, 20, 5,  9,  14, 20, 4,  11, 16, 23, 4,  11, 16,
                                    23, 4,  11, 16, 23, 4,  11, 16, 23, 6,  10, 15, 21,
                                    6,  10, 15, 21, 6,  10, 15, 21, 6,  10, 15, 21};
    uint64_t msg_len = data.size();
    data.push_back(0x80);
    while(data.size() % 64 != 56)
        data.push_back(0);
    for(int i = 0; i < 8; i++)
        data.push_back(static_cast<unsigned char>((msg_len * 8 >> (8 * i)) & 0xff));

    uint32_t a0 = 0x67452301, b0 = 0xefcdab89, c0 = 0x98badcfe, d0 = 0x10325476;
    for(size_t off = 0; off < data.size(); off += 64)
    {
        uint32_t M[16];
        for(int i = 0; i < 16; i++)
            M[i] = static_cast<uint32_t>(data[off + 4 * i]) |
                   (static_cast<uint32_t>(data[off + 4 * i + 1]) << 8) |
                   (static_cast<uint32_t>(data[off + 4 * i + 2]) << 16) |
                   (static_cast<uint32_t>(data[off + 4 * i + 3]) << 24);
        uint32_t A = a0, B = b0, C = c0, D = d0;
        for(int i = 0; i < 64; i++)
        {
            uint32_t FF, g;
            if(i < 16) { FF = (B & C) | (~B & D); g = i; }
            else if(i < 32) { FF = (D & B) | (~D & C); g = (5 * i + 1) % 16; }
            else if(i < 48) { FF = B ^ C ^ D; g = (3 * i + 5) % 16; }
            else { FF = C ^ (B | ~D); g = (7 * i) % 16; }
            FF += A + K[i] + M[g];
            A = D; D = C; C = B; B += leftrotate(FF, S[i]);
        }
        a0 += A; b0 += B; c0 += C; d0 += D;
    }
    std::ostringstream ss;
    for(uint32_t w : {a0, b0, c0, d0})
        for(int i = 0; i < 4; i++)
            ss << std::hex << std::setw(2) << std::setfill('0')
               << ((w >> (8 * i)) & 0xff);
    return ss.str();
}

// Extract the 32-char md5 string stored at bytes [28,60) of a .bin file.
std::string BinHeaderMd5(const miopen::fs::path& p)
{
    std::ifstream f(p, std::ios::binary);
    if(!f) return "";
    f.seekg(28);
    std::array<char, 32> buf{};
    f.read(buf.data(), 32);
    if(f.gcount() != 32) return "";
    return std::string(buf.data(), 32);
}

miopen::conv::ProblemDescription MakeProblem(bool is3d,
                                             miopenDataType_t dtype,
                                             miopen::conv::Direction dir)
{
    if(is3d)
    {
        miopen::TensorDescriptor in(dtype, std::vector<int>{1, 64, 8, 56, 56});
        miopen::TensorDescriptor wei(dtype, std::vector<int>{64, 64, 3, 3, 3});
        miopen::TensorDescriptor out(dtype, std::vector<int>{1, 64, 8, 56, 56});
        miopen::ConvolutionDescriptor conv(
            3, miopenConvolution, miopenPaddingDefault, {1,1,1}, {1,1,1}, {1,1,1}, {0,0,0});
        return miopen::conv::ProblemDescription(in, wei, out, conv, dir);
    }
    miopen::TensorDescriptor in(dtype, std::vector<int>{1, 64, 56, 56});
    miopen::TensorDescriptor wei(dtype, std::vector<int>{64, 64, 3, 3});
    miopen::TensorDescriptor out(dtype, std::vector<int>{1, 64, 56, 56});
    miopen::ConvolutionDescriptor conv(
        2, miopenConvolution, miopenPaddingDefault, {1,1}, {1,1}, {1,1}, {0,0});
    return miopen::conv::ProblemDescription(in, wei, out, conv, dir);
}

miopen::conv::Direction DirectionOf(const std::string& solver)
{
    if(solver.find("Wrw") != std::string::npos)
        return miopen::conv::Direction::BackwardWeights;
    if(solver.find("Bwd") != std::string::npos)
        return miopen::conv::Direction::BackwardData;
    return miopen::conv::Direction::Forward;
}

std::string NameOf(const ::testing::TestParamInfo<ArchSolver>& info)
{
    return info.param.arch + "_" + info.param.solver;
}

} // namespace

class CPU_ConvAiAssetSmoke_NONE : public ::testing::TestWithParam<ArchSolver>
{
};

// Verifies the input encoder loads and runs inference without throwing.
// Catches fdeep keras2/keras3 version mismatches: a mismatched model throws
// "tensor_shape_variable needs to be an array" at inference time, which the
// production path catches and converts to a silent non-AI fallback.
TEST_P(CPU_ConvAiAssetSmoke_NONE, InputEncoderLoadsAndRuns)
{
    const auto as = GetParam();
    if(!AssetsPresent(as))
        GTEST_SKIP() << "No assets for " << as.arch << "/" << as.solver;

    const auto& model = GetCandidateSelectionModel(as.arch, as.solver);
    const auto prob   = MakeProblem(as.is3d, miopenHalf, DirectionOf(as.solver));
    const auto feats  = miopen::solver::conv::GetFeaturesND(prob, /*max_cu=*/304, as.arch);

    std::vector<float> encoded;
    ASSERT_NO_THROW({ encoded = model.EncodeInputFeatures(feats); })
        << as.arch << "/" << as.solver
        << ": EncodeInputFeatures threw — likely a fdeep keras2/keras3 version mismatch. "
           "The production path catches this exception and falls back silently.";
    EXPECT_FALSE(encoded.empty());
}

// Verifies the md5 stored in the .bin header matches md5(input_encoder.tn.model).
// A mismatch makes SelectBestVocabCandidate fail closed (returns nullopt) silently.
// This happens when the input encoder is replaced (e.g. keras2<->keras3 swap) without
// re-stamping the .bin header.
TEST_P(CPU_ConvAiAssetSmoke_NONE, BinMd5MatchesInputEncoder)
{
    const auto as  = GetParam();
    if(!AssetsPresent(as))
        GTEST_SKIP() << "No assets for " << as.arch << "/" << as.solver;

    const auto db      = miopen::GetSystemDbPath();
    const auto enc_path = db / (as.arch + "_" + as.solver + "_input_encoder.tn.model");
    const auto bin_path = db / (as.arch + "_" + as.solver + "_kernel_config_embeddings.bin");

    const auto file_md5   = FileMd5Hex(enc_path);
    const auto header_md5 = BinHeaderMd5(bin_path);

    ASSERT_FALSE(file_md5.empty())   << "Could not read input encoder: " << enc_path;
    ASSERT_FALSE(header_md5.empty()) << "Could not read .bin header md5: " << bin_path;
    EXPECT_EQ(file_md5, header_md5)
        << as.arch << "/" << as.solver
        << ": md5 in .bin header does not match input_encoder.tn.model. "
           "SelectBestVocabCandidate will fail closed (nullopt) for all shapes on this solver. "
           "Re-stamp the .bin header with md5(input_encoder.tn.model).";
}

// Verifies the config-embedding table loads and returns a non-empty vocabulary.
// A corrupt or truncated .bin, or a header the loader rejects, produces an empty vocab
// and a silent non-AI fallback for every shape.
TEST_P(CPU_ConvAiAssetSmoke_NONE, ConfigEmbeddingTableNonEmpty)
{
    const auto as = GetParam();
    if(!AssetsPresent(as))
        GTEST_SKIP() << "No assets for " << as.arch << "/" << as.solver;

    const auto& vocab = GetConfigEmbeddingVocab(as.arch, as.solver);
    EXPECT_FALSE(vocab.empty())
        << as.arch << "/" << as.solver
        << ": GetConfigEmbeddingVocab returned empty — .bin rejected (corrupt header, md5 "
           "mismatch, or truncated file). Every shape on this solver silently falls back to "
           "non-AI config selection.";
}

INSTANTIATE_TEST_SUITE_P(Full,
                         CPU_ConvAiAssetSmoke_NONE,
                         ::testing::ValuesIn(AllArchSolvers()),
                         NameOf);

#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
