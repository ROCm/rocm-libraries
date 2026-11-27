// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// An older hipBLASLt revision directory is part of the same solution library.
// KernelsFromAnywhere keeps argument semantics on the CustomKernel record.
// KernelArguments finds each one by name, so argument order does not
// invalidate a revision. A stored content hash of that semantic-name set,
// the code object's target and its code-object version still have to agree.
// The directory name is not that check.
#include "hipblaslt-jit-component.hpp"
#include "library_fixture.hpp"
#include "solution_entry.hpp"
#include <Tensile/CachingLibrary.hpp>
#include <Tensile/ExactLogicLibrary.hpp>
#include <Tensile/SingleSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <Tensile/msgpack/Writing.hpp>
#include <catch2/catch_test_macros.hpp>
#include <hip/hip_runtime_api.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <iterator>
#include <set>
#include <string>

namespace
{
    namespace fs = std::filesystem;
    namespace hj = hipblaslt_jit;

    using Gemm     = TensileLite::ContractionProblemGemm;
    using Solution = TensileLite::ContractionSolution;

    struct Published
    {
        hj::DeviceTarget target;
        hj::CacheKey     key;
        Gemm             problem;
        std::string      kernel;
        int32_t          index = 0;
    };

    std::string fileText(const fs::path& path)
    {
        std::ifstream input(path, std::ios::binary);
        return {std::istreambuf_iterator<char>(input), {}};
    }

    void publishPlain(hj::JitLibrary& library, const fs::path& scratch, Published& published)
    {
        published.target    = hipblaslt_jit_test::currentDeviceTarget();
        const auto prepared = hipblaslt_jit_test::prepareSolutions(
            fs::u8path(HIPBLASLT_JIT_DATA), published.target, scratch);
        const auto& plain = prepared.front();
        published.kernel  = plain.source.solutions.front()->kernelName;
        JIT_REQUIRE("The committed bundle is not the plain kernel", plain.name == "plain");

        hj::BackendInfo backend;
        backend.id        = "pregenerated";
        backend.version   = 1;
        published.key     = hj::CacheKey::make(published.target, backend, hj::jitCodeObjectVersion);
        published.problem = hipblaslt_jit_test::fp16Gemm(false, plain.ks.front());

        hj::Artifact executable;
        auto         built
            = hj::makeComgrBuilder()->build(plain.source, hj::BuildKind::Executable, executable);
        hj::keepBuildLog(built, scratch);
        JIT_REQUIRE("The build for " + published.target.targetId() + " failed: " + built.message,
                    built.ok());
        const auto           solutions = hj::supportingSolutions(plain.source.solutions,
                                                       published.problem,
                                                       *published.target.hardware,
                                                       hipblaslt_jit_test::workspaceLimit);
        std::vector<int32_t> indices;
        const auto           status = library.publish(published.key,
                                            published.target.device,
                                            published.problem,
                                            executable,
                                            solutions,
                                            indices);
        JIT_REQUIRE("publish: " + status.message,
                    status.ok() && indices.size() == 1
                        && TensileLite::isJitKernelIndex(indices.front()));
        published.index = indices.front();
    }

    std::vector<int32_t> lookupIndices(hj::JitLibrary& library, const Published& published)
    {
        std::vector<int32_t> indices;
        const auto           status = library.lookup(published.key,
                                           published.target.device,
                                           published.problem,
                                           *published.target.hardware,
                                           4,
                                                     {},
                                           indices);
        JIT_REQUIRE("lookup: " + status.message, status.ok());
        return indices;
    }

    std::vector<fs::path> matchingFiles(const fs::path& root, bool codeObject)
    {
        std::vector<fs::path> files;
        std::error_code       error;
        for(fs::recursive_directory_iterator next(root, error), end; !error && next != end;
            next.increment(error))
        {
            if(!next->is_regular_file())
                continue;
            const auto path = next->path();
            if(codeObject)
            {
                if(path.extension() == ".co")
                    files.push_back(path);
                continue;
            }
            const auto name = path.filename().string();
            if(path.extension() != ".dat" || name.rfind("TensileLibrary_lazy", 0) == 0
               || name.find("Mapping") != std::string::npos || name == "allocator.dat")
                continue;
            if(path.generic_string().find("/k/") == std::string::npos)
                continue;
            files.push_back(path);
        }
        return files;
    }

    // Edits every solution object the entry file holds, once. The master map
    // and the single-solution row can share one object.
    void mutateEntries(const fs::path&                                               root,
                       const std::function<void(TensileLite::ContractionSolution&)>& mutate)
    {
        using Rows       = TensileLite::ProblemSelectionLibrary<Gemm, Solution>;
        using Single     = TensileLite::SingleSolutionLibrary<Gemm, Solution>;
        using Caching    = TensileLite::CachingLibrary<Gemm, Solution>;
        const auto files = matchingFiles(root, false);
        JIT_REQUIRE("The published library has no solution entry", !files.empty());
        for(const auto& file : files)
        {
            std::shared_ptr<TensileLite::SolutionLibrary<Gemm, Solution>> loaded;
            try
            {
                loaded = TensileLite::LoadLibraryFile<Gemm>(file.string());
            }
            catch(const std::exception& error)
            {
                JIT_REQUIRE(std::string("Cannot load ") + file.string() + ": " + error.what(),
                            false);
            }
            auto master = std::dynamic_pointer_cast<hj::GemmMaster>(loaded);
            JIT_REQUIRE("The entry is not a solution library: " + file.string(), master != nullptr);
            std::set<Solution*> seen;
            auto                once = [&](const std::shared_ptr<Solution>& solution) {
                if(!solution || !seen.insert(solution.get()).second)
                    return;
                mutate(*solution);
            };
            for(auto& item : master->solutions)
                once(item.second);
            auto library = master->library;
            if(auto cache = std::dynamic_pointer_cast<Caching>(library))
                library = cache->library();
            if(auto rows = std::dynamic_pointer_cast<Rows>(library))
                for(auto& row : rows->rows)
                    if(auto single = std::dynamic_pointer_cast<Single>(row.second))
                        once(single->solution);
            JIT_REQUIRE("The entry has no solution to edit: " + file.string(), !seen.empty());
            const auto    bytes = TensileLite::MessagePackWriteLibraryData(*master);
            std::ofstream out(file, std::ios::binary | std::ios::trunc);
            out.write(reinterpret_cast<const char*>(bytes.data()),
                      static_cast<std::streamsize>(bytes.size()));
            JIT_REQUIRE("Cannot rewrite " + file.string(), static_cast<bool>(out));
        }
    }

    hj::Artifact hipExecutable(const hj::Target& target, const fs::path& scratch)
    {
        const auto           text = hipblaslt_jit_test::readFile(fs::u8path(HIPBLASLT_JIT_KERNEL));
        hj::SourceCodeBundle bundle;
        bundle.mainKernel = {"builder_test_kernel.hip",
                             hj::Artifact::Kind::Hip,
                             std::vector<uint8_t>(text.begin(), text.end())};
        bundle.target     = target;
        bundle.kernelNames.push_back("hipblaslt_jit_builder_test_scale");
        hj::Artifact executable;
        auto built = hj::makeComgrBuilder()->build(bundle, hj::BuildKind::Executable, executable);
        hj::keepBuildLog(built, scratch);
        JIT_REQUIRE("The HIP executable was not built: " + built.message, built.ok());
        return executable;
    }
}

TEST_CASE("an older hipBLASLt revision is reused when semantic names match", "[jit-gpu]")
{
    const auto root = fs::u8path(HIPBLASLT_JIT_LAUNCH_SCRATCH) / "revision-reuse" / "names";
    fs::remove_all(root);
    const auto scratch = root / "scratch";
    fs::create_directories(scratch);

    hj::JitLibrary library(root);
    Published      published;
    publishPlain(library, scratch, published);
    const auto revision = library.buildDirectory();
    // Sorts before a hex tweak, so a search that ignores "current first" hits it first.
    const auto older = revision.parent_path() / "0older";
    fs::rename(revision, older);
    mutateEntries(older, [](Solution& solution) {
        JIT_REQUIRE("The stored kernel has fewer than two semantic arguments",
                    solution.customKernel.args.size() >= 2);
        std::reverse(solution.customKernel.args.begin(), solution.customKernel.args.end());
    });

    hj::JitLibrary fresh(root);
    const auto     found = lookupIndices(fresh, published);
    JIT_REQUIRE("An older revision whose semantic names match was not found",
                found.size() == 1 && found.front() == published.index);
    JIT_REQUIRE("The older revision directory was deleted", fs::is_directory(older));
}

TEST_CASE("an older revision is rejected when a required semantic name is gone", "[jit-gpu]")
{
    const auto root = fs::u8path(HIPBLASLT_JIT_LAUNCH_SCRATCH) / "revision-reuse" / "semantic";
    fs::remove_all(root);
    const auto scratch = root / "scratch";
    fs::create_directories(scratch);

    hj::JitLibrary library(root);
    Published      published;
    publishPlain(library, scratch, published);
    const auto revision = library.buildDirectory();
    const auto older    = revision.parent_path() / "0older";
    fs::rename(revision, older);

    {
        hj::JitLibrary before(root);
        const auto     found = lookupIndices(before, published);
        JIT_REQUIRE("The older revision was not found before a semantic name was removed",
                    found.size() == 1 && found.front() == published.index);
    }
    const auto objects = matchingFiles(older, true);
    JIT_REQUIRE("The older revision has no code object", objects.size() == 1);
    JIT_REQUIRE("The code object does not declare AddressD, so removing it cannot be observed",
                fileText(objects.front()).find("AddressD") != std::string::npos);
    mutateEntries(older, [](Solution& solution) {
        auto&      args     = solution.customKernel.args;
        const auto addressD = std::find_if(args.begin(), args.end(), [](const auto& arg) {
            return arg.semantic == TensileLite::CustomArgSemantic::AddressD;
        });
        JIT_REQUIRE("The stored record has no AddressD semantic", addressD != args.end());
        args.erase(addressD);
    });

    hj::JitLibrary after(root);
    const auto     rejected = lookupIndices(after, published);
    JIT_REQUIRE("An older entry was reused after AddressD was removed from its record",
                rejected.empty());
    JIT_REQUIRE("The older revision directory was deleted", fs::is_directory(older));
}

TEST_CASE("the current revision wins when an older revision has the same index", "[jit-gpu]")
{
    const auto root = fs::u8path(HIPBLASLT_JIT_LAUNCH_SCRATCH) / "revision-reuse" / "collision";
    fs::remove_all(root);
    const auto scratch = root / "scratch";
    fs::create_directories(scratch);

    hj::JitLibrary library(root);
    Published      published;
    publishPlain(library, scratch, published);
    const auto revision = library.buildDirectory();
    const auto older    = revision.parent_path() / "0older";
    fs::copy(revision, older, fs::copy_options::recursive);

    const auto currentObjects = matchingFiles(revision, true);
    const auto olderObjects   = matchingFiles(older, true);
    JIT_REQUIRE("The published library has no code object",
                currentObjects.size() == 1 && olderObjects.size() == 1);
    const auto hip = hipExecutable(published.target, scratch);
    {
        std::ofstream out(currentObjects.front(), std::ios::binary | std::ios::trunc);
        out.write(reinterpret_cast<const char*>(hip.bytes.data()),
                  static_cast<std::streamsize>(hip.bytes.size()));
        JIT_REQUIRE("Cannot overwrite the current code object", static_cast<bool>(out));
    }
    JIT_REQUIRE("The two revisions do not hold different code objects",
                fileText(currentObjects.front()) != fileText(olderObjects.front()));
    JIT_REQUIRE("allocator.dat was merged or a revision lost its own",
                fs::is_regular_file(revision / "allocator.dat")
                    && fs::is_regular_file(older / "allocator.dat")
                    && !fs::exists(revision.parent_path() / "allocator.dat"));

    hj::Status why;
    const auto view = library.resolve(published.target.device, published.index, why);
    JIT_REQUIRE("resolve: " + why.message, why.ok() && view.adapter != nullptr);
    const auto helperStatus = view.adapter->initKernel("hipblaslt_jit_builder_test_scale");
    const auto plainStatus  = view.adapter->initKernel(published.kernel);
    JIT_REQUIRE("resolve returned the older revision's code object",
                helperStatus == hipSuccess && plainStatus != hipSuccess);
    JIT_REQUIRE("The older revision directory was deleted", fs::is_directory(older));
}

TEST_CASE("a directory whose name matches but whose content hash does not is rejected", "[jit-gpu]")
{
    const auto root = fs::u8path(HIPBLASLT_JIT_LAUNCH_SCRATCH) / "revision-reuse" / "hash";
    fs::remove_all(root);
    const auto scratch = root / "scratch";
    fs::create_directories(scratch);

    hj::JitLibrary library(root);
    Published      published;
    publishPlain(library, scratch, published);
    const auto revision = library.buildDirectory();
    const auto older    = revision.parent_path() / "0older";
    fs::rename(revision, older);

    std::vector<fs::path> hashFiles;
    std::error_code       error;
    for(fs::recursive_directory_iterator next(older, error), end; !error && next != end;
        next.increment(error))
        if(next->is_regular_file() && next->path().extension() == ".hash")
            hashFiles.push_back(next->path());
    JIT_REQUIRE("The older revision has no semantic-name hash", hashFiles.size() == 1);
    JIT_REQUIRE("The device directory was renamed",
                fs::is_directory(older / published.key.directoryName()));
    {
        std::ofstream out(hashFiles.front(), std::ios::binary | std::ios::trunc);
        out << published.index << " 0000000000000000\n";
        JIT_REQUIRE("Cannot rewrite the content hash", static_cast<bool>(out));
    }

    hj::JitLibrary fresh(root);
    const auto     rejected = lookupIndices(fresh, published);
    JIT_REQUIRE("A directory with a matching name and a different content hash was reused",
                rejected.empty());
    JIT_REQUIRE("The older revision directory was deleted", fs::is_directory(older));
}
