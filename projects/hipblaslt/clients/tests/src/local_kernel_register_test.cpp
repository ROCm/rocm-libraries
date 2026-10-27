// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// End-to-end tests of the local kernel registration C API, using shipped bf16 TN
// shards as payloads. A build with the option ON always produces such shards,
// so a missing payload is a failure, not a skip.

#include <gtest/gtest.h>

#include "local_kernel_test_common.hpp"

#include <Tensile/ContractionProblem.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/MasterSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>

#include <atomic>
#include <cstdio>
#include <thread>

namespace
{
    using namespace local_kernel_test;

    struct ShardKernel
    {
        int         shippedIndex;
        std::string name;
    };

    // The shard's solutions in the order registration assigns their indices.
    std::vector<ShardKernel> shardKernels(std::string const& dat)
    {
        using Problem  = TensileLite::ContractionProblemGemm;
        using Solution = TensileLite::ContractionSolution;
        auto shard     = std::dynamic_pointer_cast<TensileLite::MasterSolutionLibrary<Problem>>(
            TensileLite::LoadLibraryFile<Problem, Solution>(dat));
        if(!shard)
            return {};
        shard->materializeAllSolutions();
        std::vector<ShardKernel> kernels;
        for(auto const& entry : shard->solutions)
            if(entry.second)
                kernels.push_back({entry.first, entry.second->KernelName()});
        return kernels;
    }

    std::vector<std::string> kernelNames(std::vector<ShardKernel> const& kernels)
    {
        std::vector<std::string> names;
        for(auto const& kernel : kernels)
            if(std::find(names.begin(), names.end(), kernel.name) == names.end())
                names.push_back(kernel.name);
        return names;
    }

    std::vector<int>
        registerPayload(hipblasLtHandle_t handle, std::string const& dat, std::string const& co)
    {
        std::vector<int> indices(4096);
        int              numIndices = 0;
        if(hipblasLtLocalKernelRegister(handle,
                                        dat.c_str(),
                                        co.c_str(),
                                        indices.data(),
                                        static_cast<int>(indices.size()),
                                        &numIndices)
           != HIPBLAS_STATUS_SUCCESS)
            return {};
        indices.resize(std::min<size_t>(numIndices, indices.size()));
        return indices;
    }

    bool resolve(hipblasLtHandle_t                 handle,
                 Problem const&                    p,
                 int                               index,
                 hipblasLtMatmulHeuristicResult_t& out,
                 size_t&                           workspace)
    {
        return findUsableKernel(handle, p, {index}, out, workspace) == index;
    }

    // Two devices of one architecture, so a payload for one serves both.
    bool haveTwoMatchingDevices()
    {
        int count = 0;
        if(hipGetDeviceCount(&count) != hipSuccess || count < 2)
            return false;
        hipDeviceProp_t a, b;
        if(hipGetDeviceProperties(&a, 0) != hipSuccess
           || hipGetDeviceProperties(&b, 1) != hipSuccess)
            return false;
        return std::strcmp(a.gcnArchName, b.gcnArchName) == 0;
    }

    void expectNothingRegistered(hipblasLtHandle_t handle)
    {
        size_t registered = 123, selectable = 123;
        ASSERT_EQ(hipblasLtLocalKernelGetCounts(handle, &registered, &selectable),
                  HIPBLAS_STATUS_SUCCESS);
        ASSERT_EQ(registered, 0u) << "nothing registered at start";
        ASSERT_EQ(selectable, 0u) << "nothing selectable at start";
    }
} // namespace

// Checks that:
//
//   1. A registered kernel can be run by index immediately.
//   2. Registering does not change heuristic selection.
//   3. Mapping an exact shape puts the kernel first, ahead of the shipped
//      library's choices in their original order, even though that shape was
//      already queried.
//   4. Other shapes are unaffected.
TEST(LocalKernelRegister, DISABLED_Worker)
{
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    expectNothingRegistered(handle);

    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);

    const std::vector<int> shipped = topIndices(handle, p, 8);
    ASSERT_FALSE(shipped.empty());
    for(int idx : shipped)
        ASSERT_LT(idx, LOCAL_INDEX_BASE) << "only shipped kernels before registration";

    int  isLocal      = -1;
    auto shippedFirst = topResults(handle, p, 1);
    ASSERT_EQ(shippedFirst.size(), 1u);
    ASSERT_EQ(hipblasLtMatmulAlgoIsLocalKernel(&shippedFirst[0].algo, &isLocal),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(isLocal, 0) << "a shipped kernel is not reported as a local kernel";

    Registered r = registerUsable(handle, p, candidates);
    ASSERT_GE(r.index, 0) << "a registered kernel resolves by index and supports the problem";
    std::printf("using a kernel from %s\n",
                std::filesystem::path(r.stem).filename().string().c_str());

    size_t registered = 0, selectable = 0;
    ASSERT_EQ(hipblasLtLocalKernelGetCounts(handle, &registered, &selectable),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_GT(registered, 0u);
    EXPECT_EQ(selectable, 0u) << "registering alone makes nothing selectable";
    EXPECT_GE(r.index, LOCAL_INDEX_BASE);

    EXPECT_EQ(topIndices(handle, p, 8), shipped) << "registration should not change selection";

    ASSERT_EQ(hipblasLtMatmulAlgoIsLocalKernel(&r.result.algo, &isLocal), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(isLocal, 1) << "the registered kernel is reported as a local kernel";
    EXPECT_EQ(algoIndex(r.result.algo), r.index) << "the algo round-trips its assigned index";
    runAndCheck(handle, p, r.result.algo, r.workspace);

    ASSERT_EQ(hipblasLtLocalKernelSetExactMatch(handle, r.index, M, N, 1, K),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtLocalKernelGetCounts(handle, &registered, &selectable),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(selectable, 1u);

    const std::vector<int> mapped = topIndices(handle, p, 8);
    ASSERT_GE(mapped.size(), std::min<size_t>(2, shipped.size() + 1));
    EXPECT_EQ(mapped[0], r.index) << "the mapped kernel should come first";
    EXPECT_EQ(std::count(mapped.begin(), mapped.end(), r.index), 1)
        << "the mapped kernel should appear once";
    EXPECT_TRUE(std::equal(mapped.begin() + 1, mapped.end(), shipped.begin()))
        << "the shipped choices should follow in their original order";

    auto best = topResults(handle, p, 1);
    ASSERT_EQ(best.size(), 1u);
    EXPECT_EQ(algoIndex(best[0].algo), r.index);
    runAndCheck(handle, p, best[0].algo, best[0].workspaceSize);

    Problem neighbour;
    ASSERT_EQ(buildProblem(neighbour, 32), HIPBLAS_STATUS_SUCCESS);
    const int other = topIndex(handle, neighbour);
    EXPECT_GE(other, 0) << "a neighbouring shape still resolves";
    EXPECT_LT(other, LOCAL_INDEX_BASE) << "an unmapped shape (M=32) should get a shipped kernel";

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelRegister, EndToEnd)
{
    EXPECT_EQ(runWorker("LocalKernelRegister.DISABLED_Worker"), 0)
        << "the registration worker failed; its output is above";
}

// Kernels from two payloads, mapped to two shapes, each selected for its own.
TEST(LocalKernelRegister, DISABLED_TwoShapesWorker)
{
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    expectNothingRegistered(handle);

    Problem small, large;
    ASSERT_EQ(buildProblem(small, 16), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(buildProblem(large, 64), HIPBLAS_STATUS_SUCCESS);
    ASSERT_GE(topIndex(handle, small), 0);
    ASSERT_GE(topIndex(handle, large), 0);

    Registered first = registerUsable(handle, small, candidates);
    ASSERT_GE(first.index, 0);
    Registered second = registerUsable(handle, large, candidates, first.stem);
    ASSERT_GE(second.index, 0);
    std::printf("payloads: %s and %s\n",
                std::filesystem::path(first.stem).filename().string().c_str(),
                std::filesystem::path(second.stem).filename().string().c_str());

    ASSERT_EQ(hipblasLtLocalKernelSetExactMatch(handle, first.index, small.m, N, 1, K),
              HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipblasLtLocalKernelSetExactMatch(handle, second.index, large.m, N, 1, K),
              HIPBLAS_STATUS_SUCCESS);

    size_t registered = 0, selectable = 0;
    ASSERT_EQ(hipblasLtLocalKernelGetCounts(handle, &registered, &selectable),
              HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(selectable, 2u);

    for(auto const* entry : {&small, &large})
    {
        const int expected = entry == &small ? first.index : second.index;
        auto      best     = topResults(handle, *entry, 1);
        ASSERT_EQ(best.size(), 1u);
        EXPECT_EQ(algoIndex(best[0].algo), expected) << "M=" << entry->m;
        runAndCheck(handle, *entry, best[0].algo, best[0].workspaceSize);
    }

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelRegister, TwoShapes)
{
    EXPECT_EQ(runWorker("LocalKernelRegister.DISABLED_TwoShapesWorker"), 0)
        << "the worker failed; its output is above";
}

// Registered and mapped through a handle on device 0, then selected and run
// through a handle on device 1, which has never loaded the code object.
TEST(LocalKernelRegister, DISABLED_TwoDevicesWorker)
{
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t first;
    ASSERT_EQ(hipblasLtCreate(&first), HIPBLAS_STATUS_SUCCESS);
    expectNothingRegistered(first);

    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);
    Registered r = registerUsable(first, p, candidates);
    ASSERT_GE(r.index, 0);
    ASSERT_EQ(hipblasLtLocalKernelSetExactMatch(first, r.index, M, N, 1, K),
              HIPBLAS_STATUS_SUCCESS);

    ASSERT_EQ(hipSetDevice(1), hipSuccess);
    hipblasLtHandle_t second;
    ASSERT_EQ(hipblasLtCreate(&second), HIPBLAS_STATUS_SUCCESS);

    auto best = topResults(second, p, 1);
    ASSERT_EQ(best.size(), 1u);
    EXPECT_EQ(algoIndex(best[0].algo), r.index) << "selection is shared by every device";
    runAndCheck(second, p, best[0].algo, best[0].workspaceSize);

    EXPECT_EQ(hipblasLtDestroy(second), HIPBLAS_STATUS_SUCCESS);
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    EXPECT_EQ(hipblasLtDestroy(first), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelRegister, TwoDevices)
{
    if(!haveTwoMatchingDevices())
        GTEST_SKIP() << "needs two devices of the same architecture";

    EXPECT_EQ(runWorker("LocalKernelRegister.DISABLED_TwoDevicesWorker"), 0)
        << "the worker failed; its output is above";
}

// Registered through a handle on device 0 while device 1 is current. The code
// object is loaded for the handle's device, so the kernel runs there, and the
// caller's device is left current.
TEST(LocalKernelRegister, DISABLED_OtherDeviceCurrentWorker)
{
    ASSERT_EQ(hipSetDevice(0), hipSuccess);
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    expectNothingRegistered(handle);

    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);

    Registered r;
    for(auto const& stem : candidates)
    {
        ASSERT_EQ(hipSetDevice(1), hipSuccess);
        const auto indices = registerPayload(handle, stem + ".dat", stem + ".co");
        int        current = -1;
        ASSERT_EQ(hipGetDevice(&current), hipSuccess);
        EXPECT_EQ(current, 1) << "registration leaves the caller's device current";
        ASSERT_EQ(hipSetDevice(0), hipSuccess);
        ASSERT_FALSE(indices.empty()) << stem;

        r.index = findUsableKernel(handle, p, indices, r.result, r.workspace);
        if(r.index >= 0)
            break;
    }
    ASSERT_GE(r.index, 0) << "no payload holds a kernel for this problem";
    runAndCheck(handle, p, r.result.algo, r.workspace);

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelRegister, RegistersOnTheHandlesDevice)
{
    if(!haveTwoMatchingDevices())
        GTEST_SKIP() << "needs two devices of the same architecture";

    EXPECT_EQ(runWorker("LocalKernelRegister.DISABLED_OtherDeviceCurrentWorker"), 0)
        << "the worker failed; its output is above";
}

TEST(LocalKernelRegister, DISABLED_ForeignProcessorMetadataWorker)
{
    using namespace TensileLite;

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    expectNothingRegistered(handle);

    int             device = 0;
    hipDeviceProp_t prop;
    ASSERT_EQ(hipGetDevice(&device), hipSuccess);
    ASSERT_EQ(hipGetDeviceProperties(&prop, device), hipSuccess);
    auto foreign = std::string(prop.gcnArchName).rfind("gfx950", 0) == 0
                       ? AMDGPU::Processor::gfx942
                       : AMDGPU::Processor::gfx950;

    // Name lookup needs only metadata, so this registers a solution for a
    // different processor without requiring another device or code object.
    auto solution               = std::make_shared<ContractionSolution>();
    solution->index             = -1;
    solution->kernelName        = "ForeignKernel";
    solution->solutionName      = "ForeignSolution";
    solution->hardwarePredicate = std::make_shared<Predicates::True<Hardware>>();
    ASSERT_TRUE(pinToProcessor(*solution, foreign));
    auto indices = globalLocalKernelLibrary<ContractionProblemGemm>()->addSolutions({solution});
    ASSERT_EQ(indices.size(), 1u);

    hipblasLtMatmulAlgo_t algo{};
    std::memcpy(algo.data, &indices[0], sizeof(indices[0]));
    EXPECT_TRUE(hipblaslt_ext::getKernelNameFromAlgo(handle, algo).empty());
    EXPECT_TRUE(hipblaslt_ext::getSolutionNameFromAlgo(handle, algo).empty());
    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelRegister, ForeignProcessorMetadata)
{
    EXPECT_EQ(runWorker("LocalKernelRegister.DISABLED_ForeignProcessorMetadataWorker"), 0)
        << "the worker failed; its output is above";
}

// Payloads that reach the code object load and fail there are refused cleanly:
// nothing is registered and the shipped kernel already in use still runs.
TEST(LocalKernelRegister, DISABLED_BadPayloadsWorker)
{
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    expectNothingRegistered(handle);

    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);
    auto shipped = topResults(handle, p, 1);
    ASSERT_EQ(shipped.size(), 1u);
    runAndCheck(handle, p, shipped[0].algo, shipped[0].workspaceSize);

    const std::string dat   = candidates[0] + ".dat";
    const auto        names = kernelNames(shardKernels(dat));
    ASSERT_FALSE(names.empty());

    ScratchDir        scratch;
    const std::string truncated = scratch.file("truncated.co");
    ASSERT_TRUE(writeTruncatedCopy(candidates[0] + ".co", truncated));
    const std::string wrongArch = scratch.file("wrong_arch.co");
    const std::string unrelated = scratch.file("unrelated.co");
    ASSERT_TRUE(compileCodeObject(standInKernels(names), otherArch(), wrongArch));
    ASSERT_TRUE(
        compileCodeObject(standInKernels({"NotAKernelOfThisShard"}), deviceArch(), unrelated));
    std::vector<std::pair<char const*, std::string>> cases = {
        {"a truncated code object", truncated},
        {"a code object for another architecture", wrongArch},
        {"a code object without the shard's kernels", unrelated},
    };

    for(auto const& [what, co] : cases)
    {
        int numIndices = -1;
        EXPECT_EQ(
            hipblasLtLocalKernelRegister(handle, dat.c_str(), co.c_str(), nullptr, 0, &numIndices),
            HIPBLAS_STATUS_INVALID_VALUE)
            << what;
        EXPECT_EQ(numIndices, 0) << what;

        size_t registered = 123;
        ASSERT_EQ(hipblasLtLocalKernelGetCounts(handle, &registered, nullptr),
                  HIPBLAS_STATUS_SUCCESS);
        EXPECT_EQ(registered, 0u) << what;

        SCOPED_TRACE(std::string("the shipped kernel after ") + what);
        runAndCheck(handle, p, shipped[0].algo, shipped[0].workspaceSize);
    }

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelRegister, RejectsBadPayloads)
{
    EXPECT_EQ(runWorker("LocalKernelRegister.DISABLED_BadPayloadsWorker"), 0)
        << "the worker failed; its output is above";
}

// A kernel runs from the code object it was registered with, even when a
// shipped kernel and an earlier registered kernel have the same name. The
// replacement here is an empty kernel, so it leaves D untouched.
TEST(LocalKernelRegister, DISABLED_OwnCodeObjectWorker)
{
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    expectNothingRegistered(handle);

    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);

    std::string                      stem;
    std::vector<int>                 original;
    size_t                           position = 0;
    hipblasLtMatmulHeuristicResult_t originalResult{};
    size_t                           originalWs = 0;
    for(auto const& candidate : candidates)
    {
        original = registerPayload(handle, candidate + ".dat", candidate + ".co");
        ASSERT_FALSE(original.empty()) << candidate;
        int idx = findUsableKernel(handle, p, original, originalResult, originalWs);
        if(idx >= 0)
        {
            stem     = candidate;
            position = std::find(original.begin(), original.end(), idx) - original.begin();
            break;
        }
    }
    ASSERT_FALSE(stem.empty()) << "no payload holds a kernel for this problem";

    const auto kernels = shardKernels(stem + ".dat");
    ASSERT_EQ(kernels.size(), original.size());
    const std::string name = kernels[position].name;
    EXPECT_EQ(hipblaslt_ext::getKernelNameFromAlgo(handle, originalResult.algo), name);

    hipblasLtMatmulHeuristicResult_t shippedResult{};
    size_t                           shippedWs = 0;
    ASSERT_TRUE(resolve(handle, p, kernels[position].shippedIndex, shippedResult, shippedWs))
        << "the shipped kernel of the same name";
    runAndCheck(handle, p, shippedResult.algo, shippedWs);
    runAndCheck(handle, p, originalResult.algo, originalWs);

    ScratchDir        scratch;
    const std::string standIn = scratch.file("stand_in.co");
    ASSERT_TRUE(compileCodeObject(standInKernels(kernelNames(kernels)), deviceArch(), standIn));
    const auto replaced = registerPayload(handle, stem + ".dat", standIn);
    ASSERT_EQ(replaced.size(), original.size());

    hipblasLtMatmulHeuristicResult_t replacedResult{};
    size_t                           replacedWs = 0;
    ASSERT_TRUE(resolve(handle, p, replaced[position], replacedResult, replacedWs));
    EXPECT_EQ(hipblaslt_ext::getKernelNameFromAlgo(handle, replacedResult.algo), name);

    std::vector<uint16_t> d;
    runMatmul(handle, p, replacedResult.algo, replacedWs, d);
    ASSERT_FALSE(d.empty());
    EXPECT_FALSE(allEqualK(d)) << "the replacement ran another module's kernel of the same name";

    runAndCheck(handle, p, originalResult.algo, originalWs);
    runAndCheck(handle, p, shippedResult.algo, shippedWs);

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelRegister, KernelsRunFromTheirOwnCodeObject)
{
    EXPECT_EQ(runWorker("LocalKernelRegister.DISABLED_OwnCodeObjectWorker"), 0)
        << "the worker failed; its output is above";
}

// One thread keeps running a shipped kernel while another registers bad and
// good payloads.
TEST(LocalKernelRegister, DISABLED_RegisterWhileLaunchingWorker)
{
    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";

    int device = 0;
    ASSERT_EQ(hipGetDevice(&device), hipSuccess);
    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);
    expectNothingRegistered(handle);

    Problem p;
    ASSERT_EQ(buildProblem(p), HIPBLAS_STATUS_SUCCESS);
    auto shipped = topResults(handle, p, 1);
    ASSERT_EQ(shipped.size(), 1u);

    const std::string dat = candidates[0] + ".dat";
    ScratchDir        scratch;
    const std::string truncated = scratch.file("truncated.co");
    ASSERT_TRUE(writeTruncatedCopy(candidates[0] + ".co", truncated));
    const std::string wrongArch = scratch.file("wrong_arch.co");
    ASSERT_TRUE(
        compileCodeObject(standInKernels(kernelNames(shardKernels(dat))), otherArch(), wrongArch));
    std::vector<std::string> bad = {truncated, wrongArch};

    std::atomic<bool> done{false};
    std::atomic<int>  launches{0};
    std::thread       launcher([&]() {
        ASSERT_EQ(hipSetDevice(device), hipSuccess);
        hipblasLtHandle_t own;
        ASSERT_EQ(hipblasLtCreate(&own), HIPBLAS_STATUS_SUCCESS);
        while(!done.load() || launches.load() < 10)
        {
            runAndCheck(own, p, shipped[0].algo, shipped[0].workspaceSize);
            if(::testing::Test::HasFailure())
                break;
            ++launches;
        }
        EXPECT_EQ(hipblasLtDestroy(own), HIPBLAS_STATUS_SUCCESS);
    });
    struct StopLauncher
    {
        std::atomic<bool>& done;
        std::thread&       launcher;

        ~StopLauncher()
        {
            done.store(true);
            if(launcher.joinable())
                launcher.join();
        }
    } stopLauncher{done, launcher};

    for(int i = 0; i < 20; i++)
    {
        for(auto const& co : bad)
        {
            int numIndices = -1;
            EXPECT_EQ(hipblasLtLocalKernelRegister(
                          handle, dat.c_str(), co.c_str(), nullptr, 0, &numIndices),
                      HIPBLAS_STATUS_INVALID_VALUE)
                << co;
        }
    }
    Registered r = registerUsable(handle, p, candidates);
    ASSERT_GE(r.index, 0);
    runAndCheck(handle, p, r.result.algo, r.workspace);

    done.store(true);
    launcher.join();
    EXPECT_GE(launches.load(), 10);

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}

TEST(LocalKernelRegister, RegisterWhileLaunching)
{
    EXPECT_EQ(runWorker("LocalKernelRegister.DISABLED_RegisterWhileLaunchingWorker"), 0)
        << "the worker failed; its output is above";
}

// None of these register anything, so they run in this process.
TEST(LocalKernelRegister, RejectsInvalidArguments)
{
    hipblasLtHandle_t handle;
    ASSERT_EQ(hipblasLtCreate(&handle), HIPBLAS_STATUS_SUCCESS);

    const std::string missing    = "/nonexistent/local_kernel_payload";
    int               numIndices = -1;
    EXPECT_EQ(
        hipblasLtLocalKernelRegister(
            handle, (missing + ".dat").c_str(), (missing + ".co").c_str(), nullptr, 0, &numIndices),
        HIPBLAS_STATUS_INVALID_VALUE);
    EXPECT_EQ(numIndices, 0);

    const std::vector<std::string> candidates = candidatePayloads();
    ASSERT_FALSE(candidates.empty()) << "no payload shard in this build's library";
    EXPECT_EQ(hipblasLtLocalKernelRegister(handle,
                                           (candidates[0] + ".dat").c_str(),
                                           (missing + ".co").c_str(),
                                           nullptr,
                                           0,
                                           &numIndices),
              HIPBLAS_STATUS_INVALID_VALUE)
        << "a missing code object is refused before anything is registered";

    EXPECT_EQ(hipblasLtLocalKernelSetExactMatch(handle, 0, M, N, 1, K),
              HIPBLAS_STATUS_INVALID_VALUE)
        << "a shipped index cannot be mapped";
    EXPECT_EQ(hipblasLtLocalKernelSetExactMatch(handle, INT32_MAX, M, N, 1, K),
              HIPBLAS_STATUS_INVALID_VALUE)
        << "an unregistered local kernel index cannot be mapped";

    int isLocal = 0;
    EXPECT_EQ(hipblasLtMatmulAlgoIsLocalKernel(nullptr, &isLocal), HIPBLAS_STATUS_INVALID_VALUE);

    size_t registered = 123;
    ASSERT_EQ(hipblasLtLocalKernelGetCounts(handle, &registered, nullptr), HIPBLAS_STATUS_SUCCESS);
    EXPECT_EQ(registered, 0u) << "a refused registration leaves nothing behind";

    EXPECT_EQ(hipblasLtDestroy(handle), HIPBLAS_STATUS_SUCCESS);
}
