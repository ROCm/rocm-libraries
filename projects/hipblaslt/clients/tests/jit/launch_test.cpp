// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "hipblaslt-jit-fs.hpp"
#include "library_fixture.hpp"
#include "solution_entry.hpp"
#include <Tensile/DataTypes_Half.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipHardware.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <catch2/catch_test_macros.hpp>
#include <hip/hip_runtime_api.h>

#include <chrono>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

// Compiles this device's committed assembly, publishes the code object into
// the JIT solution library, retrieves it through the solution library,
// launches it the way the TensileLite client does and checks D. Also checks
// that the library stores a helper relocatable once, and that a held file lock
// makes another acquire time out.
namespace
{
    namespace fs = std::filesystem;
    namespace hj = hipblaslt_jit;

    using Half = TensileLite::Half;
    using hipblaslt_jit_test::Device;

    // rows x columns of value(row, column), column-major with leading dimension rows.
    template <class F>
    std::vector<Half> matrix(size_t rows, size_t columns, F&& value)
    {
        std::vector<Half> result(rows * columns);
        for(size_t column = 0; column < columns; ++column)
            for(size_t row = 0; row < rows; ++row)
                result[row + column * rows] = static_cast<Half>(value(row, column));
        return result;
    }

    bool upload(Device& memory, const std::vector<Half>& values)
    {
        return hipMemcpy(memory.pointer,
                         values.data(),
                         values.size() * sizeof(Half),
                         hipMemcpyHostToDevice)
               == hipSuccess;
    }
}

TEST_CASE("a published JIT solution launches", "[jit-gpu]")
{
    const auto target = hipblaslt_jit_test::currentDeviceTarget();

    const auto libraryRoot = fs::u8path(HIPBLASLT_JIT_LAUNCH_LIBRARY);
    const auto scratch     = fs::u8path(HIPBLASLT_JIT_LAUNCH_SCRATCH);
    fs::remove_all(libraryRoot);
    fs::remove_all(scratch);
    fs::create_directories(scratch);

    const auto prepared
        = hipblaslt_jit_test::prepareSolutions(fs::u8path(HIPBLASLT_JIT_DATA), target, scratch);
    const auto& plain  = prepared.front();
    const auto  kernel = plain.source.solutions.front()->kernelName;
    REQUIRE((plain.name == "plain" && plain.ks.size() == 1));

    hj::BackendInfo backend;
    backend.id      = "pregenerated";
    backend.version = 1;
    const auto key  = hj::CacheKey::make(target, backend, hj::jitCodeObjectVersion);

    hj::Artifact executable;
    auto builtStatus
        = hj::makeComgrBuilder()->build(plain.source, hj::BuildKind::Executable, executable);
    hj::keepBuildLog(builtStatus, scratch);
    JIT_REQUIRE("The build for " + target.targetId() + " failed: " + builtStatus.message,
                builtStatus.ok());

    hj::JitLibrary       library(libraryRoot);
    const auto           problem   = hipblaslt_jit_test::fp16Gemm(false, plain.ks.front());
    constexpr auto       workspace = hipblaslt_jit_test::workspaceLimit;
    const auto           solutions = hj::supportingSolutions(
        plain.source.solutions, problem, *target.hardware, workspace);
    std::vector<int32_t> published;
    const auto           publishedStatus
        = library.publish(key, target.device, problem, executable, solutions, published);
    JIT_REQUIRE("publish: " + publishedStatus.message, publishedStatus.ok());
    JIT_REQUIRE("publish did not return one JIT solution index",
                published.size() == 1 && TensileLite::isJitKernelIndex(published.front()));

    library.beginQuery(target, backend, workspace, {});
    const auto found   = library.findTopSolutions(problem, *target.hardware, 1);
    const auto byIndex = library.getSolutionByIndex(problem, *target.hardware, published.front());
    const auto query   = library.queryStatus();
    library.endQuery();
    JIT_REQUIRE("retrieval: " + query.message, query.ok());
    JIT_REQUIRE("findTopSolutions did not return the published kernel",
                found.size() == 1 && found.front() && found.front()->KernelName() == kernel);
    JIT_REQUIRE("getSolutionByIndex did not return the published kernel",
                byIndex && byIndex->KernelName() == kernel);

    hj::Status why;
    const auto view = library.resolve(target.device, published.front(), why);
    JIT_REQUIRE("resolve: " + why.message, view.adapter != nullptr);

    // fp16Gemm's problem: column-major, A not transposed, lda = M, ldb = K, ldc = ldd = M.
    constexpr size_t M = 256;
    constexpr size_t N = 128;
    const size_t     K = plain.ks.front();
    constexpr double alpha = 1.25, beta = 0.5;
    const auto       a     = matrix(M, K, [](size_t m, size_t k) {
        return double(int((m + 2 * k) % 5) - 2) / 4;
    });
    const auto       b     = matrix(K, N, [](size_t k, size_t n) {
        return double(int((3 * k + n) % 7) - 3) / 4;
    });
    const auto       c     = matrix(M, N, [](size_t m, size_t n) {
        return double(int((m + n) % 9) - 4) / 8;
    });
    Device           A(a.size() * sizeof(Half));
    Device           B(b.size() * sizeof(Half));
    Device           C(c.size() * sizeof(Half));
    Device           D(c.size() * sizeof(Half));
    JIT_REQUIRE("Cannot initialize the GEMM tensors",
                upload(A, a) && upload(B, b) && upload(C, c)
                    && hipMemset(D.pointer, 0xff, c.size() * sizeof(Half)) == hipSuccess);
    const auto   workspaceBytes = found.front()->requiredWorkspaceSize(problem, *target.hardware);
    Device       work(workspaceBytes);
    if(workspaceBytes != 0)
    {
        JIT_REQUIRE("Cannot allocate the kernel workspace", work.pointer != nullptr);
    }

    TensileLite::ContractionInputs inputs;
    inputs.a     = A.pointer;
    inputs.b     = B.pointer;
    inputs.c     = C.pointer;
    inputs.d     = D.pointer;
    inputs.alpha = static_cast<float>(alpha);
    inputs.beta  = static_cast<float>(beta);
    inputs.ws    = work.pointer;

    std::vector<TensileLite::KernelInvocation> kernels;
    try
    {
        kernels = found.front()->solve(problem, inputs, *target.hardware);
    }
    catch(const std::exception& error)
    {
        JIT_REQUIRE(std::string("solve: ") + error.what(), false);
    }
    JIT_REQUIRE("solve returned no kernel invocation", !kernels.empty());
    JIT_REQUIRE("launchKernels failed", view.adapter->launchKernels(kernels) == hipSuccess);
    JIT_REQUIRE("the launched kernel did not complete", hipDeviceSynchronize() == hipSuccess);

    std::vector<Half> d(c.size());
    JIT_REQUIRE("Cannot copy D back",
                hipMemcpy(d.data(), D.pointer, d.size() * sizeof(Half), hipMemcpyDeviceToHost)
                    == hipSuccess);
    size_t      wrong = 0;
    std::string first;
    for(size_t n = 0; n < N; ++n)
        for(size_t m = 0; m < M; ++m)
        {
            double product = 0;
            for(size_t k = 0; k < K; ++k)
                product += double(a[m + k * M]) * double(b[k + n * K]);
            const double expected = alpha * product + beta * double(c[m + n * M]);
            const double actual   = double(d[m + n * M]);
            // FP16 rounding of D, plus FP32 accumulation slack.
            if(!(std::abs(actual - expected) <= 1e-2 + std::ldexp(std::abs(expected), -10))
               && wrong++ == 0)
                first = "D(" + std::to_string(m) + "," + std::to_string(n) + ") = "
                        + std::to_string(actual) + ", expected " + std::to_string(expected);
        }
    JIT_REQUIRE(std::to_string(wrong) + " of " + std::to_string(M * N)
                    + " elements of D differ from the host reference; first " + first,
                wrong == 0);

    // Another hipBLASLt build owns a sibling revision directory and never reads this one.
    JIT_REQUIRE("The library is not under <root>/v1/<git revision>",
                !hj::JitLibrary::buildRevision().empty()
                    && library.buildDirectory().parent_path() == libraryRoot / "v1"
                    && fs::is_directory(library.directory(key)));
    const auto other = library.buildDirectory().parent_path() / "another-build";
    fs::rename(library.buildDirectory(), other);
    hj::JitLibrary fresh(libraryRoot);
    fresh.beginQuery(target, backend, workspace, {});
    const auto unseen = fresh.findTopSolutions(problem, *target.hardware, 1);
    fresh.endQuery();
    JIT_REQUIRE("A library published by another build was read", unseen.empty());
}

TEST_CASE("a stored helper relocatable is reused and linked", "[jit-cpu]")
{
    const auto root = fs::u8path(HIPBLASLT_JIT_HELPER_LIBRARY);
    fs::remove_all(root);
    const auto target = hj::Target::fromTargetId("gfx950");
    REQUIRE((target.has_value()));
    hj::CacheKey key;
    key.target         = *target;
    key.wavefrontSize  = 64;
    key.backendId      = "helper-store";
    key.backendVersion = 1;

    const std::string helperKernel = "hipblaslt_jit_builder_test_scale";
    const auto        text         = hipblaslt_jit_test::readFile(fs::u8path(HIPBLASLT_JIT_KERNEL));
    int               compiled     = 0;
    const auto        compile      = [&](hj::Artifact& out) {
        ++compiled;
        hj::SourceCodeBundle helper;
        helper.mainKernel = {"builder_test_kernel.hip",
                             hj::Artifact::Kind::Hip,
                             std::vector<uint8_t>(text.begin(), text.end())};
        helper.target     = *target;
        return hj::makeComgrBuilder()->build(helper, hj::BuildKind::Relocatable, out);
    };

    hj::Artifact stored;
    {
        hj::JitLibrary library(root);
        hj::Artifact   first, second;
        const auto     missed = library.acquireHelper(key, 0, helperKernel, compile, first);
        JIT_REQUIRE("A missing helper was not compiled once and stored: " + missed.message,
                    missed.ok() && compiled == 1 && first.kind == hj::Artifact::Kind::Relocatable
                        && first.name == helperKernel && !first.bytes.empty()
                        && fs::is_regular_file(library.directory(key) / "helpers"
                                               / fs::u8path(helperKernel + ".o")));
        const auto hit = library.acquireHelper(key, 0, helperKernel, compile, second);
        JIT_REQUIRE("A stored helper was compiled again or changed: " + hit.message,
                    hit.ok() && compiled == 1 && second.bytes == first.bytes);
        hj::Artifact unused;
        const auto   misnamed = library.acquireHelper(key, 0, "not_a_kernel", compile, unused);
        JIT_REQUIRE("A helper that does not define its kernel name was stored",
                    !misnamed.ok() && compiled == 2
                        && !fs::exists(library.directory(key) / "helpers" / "not_a_kernel.o"));
        stored = first;
    }
    hj::JitLibrary reopened(root);
    hj::Artifact   helper;
    {
        const auto hit = reopened.acquireHelper(key, 0, helperKernel, compile, helper);
        INFO(("Another library instance did not read the stored helper: " + hit.message));
        REQUIRE((hit.ok() && compiled == 2 && helper.bytes == stored.bytes));
    }
}

TEST_CASE("a held file lock times out and is acquired once released", "[jit-cpu]")
{
    const auto scratch = fs::u8path(HIPBLASLT_JIT_LOCK_SCRATCH);
    fs::remove_all(scratch);
    fs::create_directories(scratch);
    const auto file = scratch / "lock";

    hj::files::FileLock held;
    const auto          taken = hj::files::FileLock::acquire(file, std::chrono::seconds(1), held);
    JIT_REQUIRE("A free lock was not acquired: " + taken.message, taken.ok());
    hj::files::FileLock second;
    const auto          start   = std::chrono::steady_clock::now();
    const auto          blocked = hj::files::FileLock::acquire(
        file, std::chrono::milliseconds(200), second);
    const auto          waited  = std::chrono::steady_clock::now() - start;
    JIT_REQUIRE(
        "A held lock did not time out after 200 ms: " + blocked.message,
        !blocked.ok() && blocked.message.find("timed out waiting for the lock") != std::string::npos
            && waited >= std::chrono::milliseconds(200) && waited < std::chrono::seconds(5));
    held.release();
    const auto freed = hj::files::FileLock::acquire(file, std::chrono::seconds(1), second);
    JIT_REQUIRE("A released lock was not acquired: " + freed.message, freed.ok());
}
