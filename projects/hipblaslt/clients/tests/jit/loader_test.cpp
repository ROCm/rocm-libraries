// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-loader.hpp"
#include "test_helpers.hpp"

#include <Tensile/hip/HipHardware.hpp>
#include <algorithm>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>

// Builds the plain and plain-pair bundles of the current device's architecture
// with comgr for the device, loads them through the TensileLite loader and checks
// which solution their libraries select for each GEMM: plain's one solution for
// the FP16 GEMM it was generated for, plain-pair's first solution when K is a
// multiple of 512 and its second otherwise, and nothing for a transposed A.
// Also checks that the loader rejects entries whose solutions and kernels do
// not match. Launches nothing.
namespace
{
    namespace fs = std::filesystem;
    namespace hj = hipblaslt_jit;
    using hipblaslt_jit_test::require;

    // FP16 A, B, C and D with FP32 alpha, beta and accumulation, as the bundles' entries expect.
    TensileLite::ContractionProblemGemm gemm(bool transA, size_t K)
    {
        constexpr size_t M = 256, N = 128;
        const auto       half = rocisa::DataType::Half;
        const size_t     lda  = transA ? K : M;
        auto problem = TensileLite::ContractionProblemGemm::GEMM_Strides(
            transA, false, half, half, half, half, M, N, K, 1, lda, lda * (transA ? M : K), K,
            K * N, M, M * N, M, M * N, 0.5);
        problem.setComputeInputTypeA(half);
        problem.setComputeInputTypeB(half);
        problem.setAlphaType(rocisa::DataType::Float);
        problem.setBetaType(rocisa::DataType::Float);
        problem.setHighPrecisionAccumulate(true);
        problem.setStridedBatched(true);
        problem.setUseDeviceUserArguments(false);
        problem.setAlphaRestriction(TensileLite::toScalarValueEnum(1.25));
        problem.setBetaRestriction(TensileLite::toScalarValueEnum(0.5));
        problem.setCEqualsD(false);
        return problem;
    }

    std::string whyNot(const hj::TensileBundle&                   bundle,
                       int                                        index,
                       const TensileLite::ContractionProblemGemm& problem)
    {
        std::ostringstream reason;
        const auto&        solution = *bundle.library->solutions.at(index);
        solution.hardwarePredicate->debugEval(*bundle.hardware, reason);
        solution.problemPredicate->debugEval(problem, reason);
        return reason.str();
    }

    // Requires that the library selects local solution index, or nothing when index is -1.
    void select(const hj::TensileBundle& bundle, bool transA, size_t K, int index)
    {
        const auto problem  = gemm(transA, K);
        const auto selected = bundle.library->findBestSolution(problem, *bundle.hardware);
        const auto what     = std::string(transA ? "a transposed A" : "the GEMM") + " with K="
                          + std::to_string(K);
        if(index < 0)
            require(selected == nullptr, "The library selects a solution for " + what);
        else
            require(selected == bundle.library->solutions.at(index),
                    "The library does not select solution " + std::to_string(index) + " for "
                        + what + ": " + whyNot(bundle, index, problem));
    }

    // The entry with every index 1 encoded as index 2, so that its solutions are 0 and 2.
    std::vector<uint8_t> skipIndexOne(std::vector<uint8_t> entry)
    {
        const std::vector<uint8_t> from{0xa5, 'i', 'n', 'd', 'e', 'x', 1}; // MsgPack "index": 1
        auto                       at = entry.begin();
        while((at = std::search(at, entry.end(), from.begin(), from.end())) != entry.end())
            *(at += from.size() - 1) = 2;
        return entry;
    }
}

int main(int argc, char** argv)
try
{
    require(argc == 3, "Usage: hipblaslt-jit-loader-test BUNDLES SCRATCH");
    const auto scratch = fs::u8path(argv[2]);
    fs::remove_all(scratch);
    fs::create_directories(scratch);

    int             device = 0;
    hipDeviceProp_t properties{};
    require(hipGetDevice(&device) == hipSuccess
                && hipGetDeviceProperties(&properties, device) == hipSuccess,
            "Cannot query the current HIP device");
    const auto hardware = TensileLite::hip::GetDevice(properties, device);
    const auto bundles  = hipblaslt_jit_test::deviceBundles(fs::u8path(argv[1]),
                                                           properties.gcnArchName);

    const auto load = [&](const std::string& name, hj::BuiltSolution& built) {
        const auto read   = hj::readTensileSourceBundle(bundles / name);
        const auto status = hj::makeComgrBuilder()->build(
            read.solution, {properties.gcnArchName, hj::jitCodeObjectVersion, scratch}, built);
        require(status.ok(), std::string("The build for ") + properties.gcnArchName
                                 + " failed: " + status.message);
        auto bundle = hj::parseTensileBundle(built, hardware);
        hj::loadTensileBundle(*bundle, built);
        std::cout << "PASS loaded " << name << " with " << bundle->library->solutions.size()
                  << " solutions and " << bundle->kernels.size() << " kernel on "
                  << properties.gcnArchName << '\n';
        return bundle;
    };

    hj::BuiltSolution built;
    const auto        plain = load("plain", built);
    select(*plain, false, 512, 0);
    select(*plain, true, 512, -1);
    std::cout << "PASS plain selects its solution for the GEMM it was generated for and "
                 "nothing for a transposed A\n";

    const auto pair = load("plain-pair", built);
    require(pair->library->solutions.size() == 2 && pair->kernels.size() == 1,
            "plain-pair is not two solutions of one kernel");
    select(*pair, false, 512, 0);
    select(*pair, false, 256, 1);
    select(*pair, true, 512, -1);
    std::cout << "PASS plain-pair selects solution 0 for K=512, 1 for K=256 and nothing for "
                 "a transposed A\n";

    const auto reject = [&](hj::BuiltSolution damaged, const std::string& what) {
        hipblaslt_jit_test::reject([&] { hj::parseTensileBundle(damaged, hardware); },
                                   "The loader accepted " + what);
    };
    auto gap            = built;
    gap.generated.entry = skipIndexOne(gap.generated.entry);
    reject(gap, "solutions 0 and 2");
    auto unnamed                  = built;
    unnamed.generated.kernelNames = {};
    reject(unnamed, "solutions whose kernel the build does not define");
    auto unused = built;
    unused.generated.kernelNames.push_back("hipblaslt_jit_loader_test_unused");
    reject(unused, "a built kernel no solution names");
    std::cout << "PASS the loader rejects solutions 0 and 2, an undefined kernel and an "
                 "unused kernel\n";
    return 0;
}
catch(const std::exception& error)
{
    std::cerr << "FAIL: " << error.what() << '\n';
    return 1;
}
