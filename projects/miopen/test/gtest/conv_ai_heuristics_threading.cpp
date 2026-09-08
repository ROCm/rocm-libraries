// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

// Thread-safety coverage for the AI-heuristic model caches (ROCM-27477).
//
// GetFdeepModel / GetModel in ai_heuristics.cpp cache loaded models in process-global
// static maps. When convolutions run concurrently from multiple threads on the same
// device (e.g. per-thread MIOpen handles) those callers race on the same cache key
// during first population; before the mutex fix this was a use-after-free that
// segfaulted deep inside fdeep predict.
//
// The concurrent burst below is host-side work (fdeep load/predict runs on the CPU, no
// kernels), but the test is gated to Instinct/CDNA devices: the candidate-selection
// models only ship for gfx942/gfx950, and we do not want it executing on consumer/APU
// parts (gfx1151, gfx1103, ...). Catching the race needs a cold cache when the burst
// starts. The gfx950 parameter is reliably cold: no other test uses gfx950 candidate-
// selection keys. The gfx942 key can be pre-warmed by conv_ai_candidate_selection_model
// (same keys) when the whole miopen_gtest binary runs in one process; a Standard-only or
// otherwise isolated/filtered run keeps it cold too. Without the fix this crashes
// (SIGSEGV/SIGABRT) or trips ThreadSanitizer; with it, all threads return identical,
// non-empty encodings. (ThreadSanitizer would be the order-independent guard.)

#include <gtest/gtest.h>
#include "gtest_common.hpp" // Gpu, IsTestSupportedByDevice
#include <miopen/conv/heuristics/ai_candidate_selection.hpp>
#include <miopen/db_path.hpp>
#include <miopen/filesystem.hpp>

#include <latch>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if MIOPEN_ENABLE_AI_KERNEL_TUNING
using namespace miopen::ai::tuning::candidate_selection;

namespace {

// The 2D group-fwd xdlops solver whose HeuristicInit surfaced the ROCM-27477 crash.
constexpr auto kSolver = "ConvHipImplicitGemmGroupFwdXdlops";

// Instinct/CDNA only. Candidate-selection (KTN) models ship for gfx942/gfx950; the
// rest of the mask is future-proofing. This excludes consumer/APU parts
// (gfx103X/gfx110X/gfx115X/gfx120X/gfx125X, e.g. Strix Halo gfx1151 and gfx1103).
// Note: Gpu::operator| is not constexpr, so this is built at call time.
Gpu InstinctDevices() { return Gpu::gfx908 | Gpu::gfx90A | Gpu::gfx94X | Gpu::gfx950; }

std::vector<miopen::fs::path> GetModelFiles(const std::string& arch)
{
    const auto db = miopen::GetSystemDbPath();
    return {db / (arch + "_" + kSolver + "_input_encoder.tn.model"),
            db / (arch + "_" + kSolver + "_kernel_config_encoder.tn.model"),
            db / (arch + "_" + kSolver + "_metadata.tn.model")};
}

std::vector<miopen::fs::path> GetMissingModelFiles(const std::string& arch)
{
    std::vector<miopen::fs::path> missing;
    for(const auto& f : GetModelFiles(arch))
        if(!miopen::fs::exists(f))
            missing.push_back(f);
    return missing;
}

std::string FormatMissing(const std::vector<miopen::fs::path>& missing)
{
    std::ostringstream os;
    for(const auto& f : missing)
        os << "\n  " << f.string();
    return os.str();
}

// Parameterized by candidate-selection model arch. fdeep inference is host-side, so on
// an Instinct host any shipped model set exercises the cache regardless of that host's
// exact arch.
class GPU_AIHeuristicsThreadSafety_FP32 : public ::testing::TestWithParam<std::string>
{
protected:
    void SetUp() override
    {
        if(!IsTestSupportedByDevice(InstinctDevices()))
            GTEST_SKIP() << "Not an Instinct/CDNA device; skipping AI-heuristic cache test.";

        const auto missing = GetMissingModelFiles(GetParam());
        if(!missing.empty())
            GTEST_SKIP() << "Candidate-selection model files not shipped for arch " << GetParam()
                         << ":" << FormatMissing(missing);
    }
};

} // namespace

// Hammer GetFdeepModel from many threads on a cold cache. Each thread owns its own
// CandidateSelectionModel (mirroring per-thread MIOpen handles) and repeatedly encodes
// the same inputs; a latch lines the threads up so the first-population race is likely.
// Success criteria: no crash, no thrown exception, and every thread produces the same
// non-empty encoding (a corrupted/half-built model would diverge or throw).
TEST_P(GPU_AIHeuristicsThreadSafety_FP32, ConcurrentGetFdeepModel)
{
    const std::string arch    = GetParam();
    constexpr int num_threads = 32;
    constexpr int iters       = 8;

    CandidateSelectionMetadata meta(arch, kSolver);

    std::map<std::string, float> features;
    for(const auto& name : meta.input_params())
        features[name] = 1.0f;

    const size_t cfg_size = meta.output_params().size() - meta.GetConstantOutputIndices().size();
    const std::vector<std::vector<float>> candidates(16, std::vector<float>(cfg_size, 0.0f));

    std::latch start_line(num_threads);
    std::vector<std::string> errors(num_threads);           // per-thread failure message ("" == ok)
    std::vector<std::vector<float>> input_out(num_threads); // per-thread input-encoder result

    auto worker = [&](int tid) {
        try
        {
            CandidateSelectionModel model(arch, kSolver);
            start_line.arrive_and_wait(); // maximize overlap on the cold cache
            for(int i = 0; i < iters; ++i)
            {
                auto enc = model.EncodeInputFeatures(features);
                auto cfg = model.EncodeKernelConfigs(candidates);
                if(enc.empty())
                    throw std::runtime_error("EncodeInputFeatures returned empty");
                if(cfg.empty() || cfg.front().empty())
                    throw std::runtime_error("EncodeKernelConfigs returned empty");
                if(i == 0)
                    input_out[tid] = std::move(enc);
            }
        }
        catch(const std::exception& e)
        {
            errors[tid] = e.what();
        }
        catch(...)
        {
            errors[tid] = "unknown exception";
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for(int t = 0; t < num_threads; ++t)
        threads.emplace_back(worker, t);
    for(auto& th : threads)
        th.join();

    for(int t = 0; t < num_threads; ++t)
        ASSERT_TRUE(errors[t].empty()) << "thread " << t << " failed: " << errors[t];

    // All threads encoded identical inputs -> identical, deterministic outputs. A model
    // object corrupted by a race would diverge here even if it did not crash outright.
    for(int t = 1; t < num_threads; ++t)
    {
        ASSERT_EQ(input_out[t].size(), input_out[0].size())
            << "thread " << t << " produced a different-sized encoding";
        EXPECT_EQ(input_out[t], input_out[0])
            << "thread " << t << " produced a different encoding than thread 0";
    }
}

INSTANTIATE_TEST_SUITE_P(Standard,
                         GPU_AIHeuristicsThreadSafety_FP32,
                         testing::Values(std::string{"gfx942"}, std::string{"gfx950"}),
                         [](const testing::TestParamInfo<std::string>& info) {
                             return info.param;
                         });
#endif // MIOPEN_ENABLE_AI_KERNEL_TUNING
