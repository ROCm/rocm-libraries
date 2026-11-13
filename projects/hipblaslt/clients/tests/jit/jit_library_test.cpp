// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Exercises the JIT solution library with the solution and executable built
// from the committed plain assembly of the current device: cache keys, directory
// checks, the stock TensileLite loader reading what the library writes,
// exact-size lookup, index allocation, a library whose publish stopped early,
// concurrent publishers in separate processes, and the rejection of fused GEMM
// and all-to-all.

#include "hipblaslt-jit-fs.hpp"
#include "hipblaslt-jit-library.hpp"
#include "hipblaslt-jit-msgpack.hpp"
#include "hipblaslt-jit-source-bundle.hpp"
#include "library_fixture.hpp"
#include "solution_entry.hpp"
#include <Tensile/AMDGPU.hpp>
#include <Tensile/CachingLibrary.hpp>
#include <Tensile/ExactLogicLibrary.hpp>
#include <Tensile/PlaceholderLibrary.hpp>
#include <Tensile/SingleSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/msgpack/Writing.hpp>

#define CATCH_CONFIG_RUNNER
#include <catch2/catch_session.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <thread>

#include <limits.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

extern char** environ;

namespace hj = hipblaslt_jit;
namespace fs = std::filesystem;
using TensileLite::ContractionProblemGemm;
using TensileLite::ContractionSolution;

namespace
{
    constexpr int32_t base = TensileLite::JitKernelIndexBase;

    using Rows = TensileLite::ProblemSelectionLibrary<ContractionProblemGemm, ContractionSolution>;
    using Single = TensileLite::SingleSolutionLibrary<ContractionProblemGemm, ContractionSolution>;

    void ok(const hj::Status& status, const std::string& what)
    {
        JIT_REQUIRE(what + ": " + status.message, status.ok());
    }
    template <class T>
    std::string text(const std::vector<T>& values)
    {
        std::ostringstream out;
        for(const auto& value : values)
            out << (out.tellp() ? "," : "") << value;
        return "[" + out.str() + "]";
    }

    std::vector<uint8_t> read(const fs::path& path)
    {
        std::vector<uint8_t> bytes;
        ok(hj::files::readFile(path, size_t{1} << 30, bytes), "read");
        return bytes;
    }
    std::string readText(const fs::path& path)
    {
        const auto bytes = read(path);
        return std::string(bytes.begin(), bytes.end());
    }

    // The solution a kernel's assembly records, and the executable built from it.
    struct Entry
    {
        hipblaslt_jit_test::Solution solution;
        hj::Artifact                 executable;
        std::string                  kernel;
    };

    // Assembles source's main kernel, reads its solution and links its executable.
    Entry build(const hj::SourceCodeBundle& source, const fs::path& scratch)
    {
        const auto              builder = hj::makeComgrBuilder();
        hj::TensileliteAssembly assembly;
        hj::Artifact            relocatable;
        auto status = hj::TensileliteAssembly::make(*builder, source, assembly, relocatable);
        hj::keepBuildLog(status, scratch);
        ok(status, "assemble " + source.mainKernel.name);
        JIT_REQUIRE("The assembly does not record one solution", assembly.solutions.size() == 1);
        auto linked       = source;
        linked.mainKernel = std::move(relocatable);
        linked.solutions  = assembly.solutions;
        Entry entry;
        status = builder->build(linked, hj::BuildKind::Executable, entry.executable);
        hj::keepBuildLog(status, scratch);
        ok(status, "link " + source.mainKernel.name);
        entry.solution = assembly.solutions.front();
        entry.kernel   = entry.solution->kernelName;
        return entry;
    }

    // An entry file holding a copy of solution as index, written by TensileLite.
    std::vector<uint8_t> entryAs(const hipblaslt_jit_test::Solution& solution, int32_t index)
    {
        auto copy   = hj::copySolution(solution);
        copy->index = index;
        auto rows   = std::make_shared<Rows>();
        rows->rows.emplace_back(copy->problemPredicate, std::make_shared<Single>(copy));
        hj::GemmMaster entry;
        entry.solutions.emplace(index, copy);
        entry.library = rows;
        return TensileLite::MessagePackWriteLibraryData(entry);
    }

    // The lazy master and mapping the library writes for an ISA.
    std::string lazyMaster(const std::string& arch)
    {
        return "TensileLibrary_lazy_" + arch + ".dat";
    }
    std::string lazyMapping(const std::string& arch)
    {
        return "TensileLiteLibrary_lazy_" + arch + "_Mapping.dat";
    }

    // The plain bundle's ProblemType, with room for any workspace.
    ContractionProblemGemm gemm(size_t m, size_t n = 128, size_t k = 512, size_t batch = 1)
    {
        auto problem = hipblaslt_jit_test::fp16Gemm(false, k, m, n, batch);
        problem.setWorkspaceSize(size_t{1} << 30);
        return problem;
    }

    // Loads a key directory with the calls TensileHost makes and no JIT code.
    std::shared_ptr<hj::GemmMaster> stock(const fs::path& directory, const std::string& arch)
    {
        const auto path   = (directory / lazyMaster(arch)).string();
        auto       master = std::dynamic_pointer_cast<hj::GemmMaster>(
            TensileLite::LoadLibraryFilePreload<ContractionProblemGemm>(path, {}));
        JIT_REQUIRE("The stock loader cannot read " + path,
                    master && master->initLibraryMapping(path));
        return master;
    }

    // The entry file prefix each master row selects, in order.
    std::vector<std::string> rowPrefixes(const hj::GemmMaster& master)
    {
        using Caching = TensileLite::CachingLibrary<ContractionProblemGemm, ContractionSolution>;
        using Placeholder
            = TensileLite::PlaceholderLibrary<ContractionProblemGemm, ContractionSolution>;
        auto library = master.library;
        if(const auto cache = std::dynamic_pointer_cast<Caching>(library))
            library = cache->library();
        const auto* rows = dynamic_cast<const Rows*>(library.get());
        JIT_REQUIRE("The master does not select its rows by problem", rows != nullptr);
        std::vector<std::string> prefixes;
        for(const auto& row : rows->rows)
        {
            const auto* placeholder = dynamic_cast<const Placeholder*>(row.second.get());
            JIT_REQUIRE("A master row is not an entry file", placeholder != nullptr);
            prefixes.push_back(placeholder->filePrefix);
        }
        return prefixes;
    }

    // Every mapped index loads with its code object, and every master row is
    // of a mapped solution.
    void consistentSlice(const fs::path&              directory,
                         const TensileLite::Hardware& hardware,
                         const std::string&           arch)
    {
        const auto                    master = stock(directory, arch);
        std::map<std::string, size_t> rows, mapped;
        for(const auto& prefix : rowPrefixes(*master))
            ++rows[prefix];
        for(const auto& [index, prefix] : master->libraryMapping)
        {
            JIT_REQUIRE("Mapped index outside the JIT range", TensileLite::isJitKernelIndex(index));
            JIT_REQUIRE("Missing " + prefix + ".co",
                        fs::is_regular_file(directory / (prefix + ".co")));
            const auto solution = master->getSolutionByIndex(hardware, index);
            JIT_REQUIRE("Mapped index " + std::to_string(index) + " does not load",
                        solution && solution->index == index
                            && solution->codeObjectFilename.load() == prefix + ".co");
            ++mapped[prefix];
        }
        for(const auto& [prefix, count] : rows)
        {
            JIT_REQUIRE("Master rows of " + prefix + " outnumber its mapped solutions",
                        mapped[prefix] >= count);
        }
    }

    // The key directory, and each slice under it. A query's master lives in one slice.
    void consistent(const fs::path&              directory,
                    const TensileLite::Hardware& hardware,
                    const std::string&           arch)
    {
        if(fs::is_regular_file(directory / lazyMaster(arch)))
            consistentSlice(directory, hardware, arch);
        std::error_code error;
        if(!fs::exists(directory / "slices", error))
            return;
        for(const auto& entry : fs::recursive_directory_iterator(directory / "slices"))
            if(entry.is_directory() && fs::is_regular_file(entry.path() / lazyMaster(arch)))
                consistentSlice(entry.path(), hardware, arch);
    }

    // Names, modes, sizes and modification times of everything under root.
    std::string tree(const fs::path& root)
    {
        std::vector<std::string> lines;
        for(const auto& entry : fs::recursive_directory_iterator(root))
        {
            struct stat info
            {
            };
            JIT_REQUIRE("lstat failed", ::lstat(entry.path().c_str(), &info) == 0);
            lines.push_back(entry.path().lexically_relative(root).string() + " "
                            + std::to_string(info.st_mode) + " " + std::to_string(info.st_size)
                            + " " + std::to_string(info.st_mtim.tv_sec) + "."
                            + std::to_string(info.st_mtim.tv_nsec));
        }
        std::sort(lines.begin(), lines.end());
        std::string result;
        for(const auto& line : lines)
            result += line + "\n";
        return result;
    }

    unsigned mode(const fs::path& path)
    {
        struct stat info
        {
        };
        JIT_REQUIRE("lstat " + path.string(), ::lstat(path.c_str(), &info) == 0);
        return info.st_mode & 07777;
    }
    std::string octal(unsigned value)
    {
        std::ostringstream out;
        out << std::oct << value;
        return out.str();
    }

    struct Context
    {
        fs::path             scratch;
        hj::SourceCodeBundle source; // the plain assembly, for device 0
        Entry                entry; // built from source
        std::string          arch; // processor name of device 0, such as "gfx942"
        TensileLite::AMDGPU  hardware{};

        fs::path fresh(const std::string& name) const
        {
            auto path = scratch / name;
            fs::remove_all(path);
            return path;
        }

        // The plain kernel renamed <kernel><suffix>, so one size can have
        // solutions of several kernels.
        Entry renamed(const std::string& suffix) const
        {
            const auto& kernel = entry.kernel;
            std::string text(source.mainKernel.bytes.begin(), source.mainKernel.bytes.end());
            for(auto at = text.find(kernel); at != std::string::npos;
                at      = text.find(kernel, at + kernel.size() + suffix.size()))
                text.insert(at + kernel.size(), suffix);
            auto copy = source;
            copy.mainKernel.bytes.assign(text.begin(), text.end());
            return build(copy, scratch);
        }
        std::vector<int32_t> find(hj::JitLibrary&                 library,
                                  const hj::CacheKey&             key,
                                  const ContractionProblemGemm&   problem,
                                  size_t                          count   = 4,
                                  const std::vector<std::string>& exclude = {}) const
        {
            std::vector<int32_t> indices;
            ok(library.lookup(key, 0, problem, hardware, count, exclude, indices), "lookup");
            return indices;
        }
    };

    std::vector<int32_t> publish(hj::JitLibrary&               library,
                                 const hj::CacheKey&           key,
                                 const ContractionProblemGemm& problem,
                                 const std::vector<Entry>&     entries)
    {
        std::vector<int32_t> indices;
        for(const auto& entry : entries)
        {
            std::vector<int32_t> published;
            ok(library.publish(key, 0, problem, entry.executable, {entry.solution}, published),
               "publish");
            indices.insert(indices.end(), published.begin(), published.end());
        }
        JIT_REQUIRE("Publish returned " + text(indices), indices.size() == entries.size());
        return indices;
    }

    fs::path builtUnder(const fs::path& root)
    {
        return hj::JitLibrary(root).buildDirectory();
    }

    // gfx942 and gfx950, and the committed plain bundles, are wave64.
    // The target id is the device processor with the sramecc+/xnack- feature spelling.
    hj::CacheKey testKey(const Context& ctx, uint32_t backendVersion = 1)
    {
        hj::CacheKey key;
        key.target.processor = ctx.hardware.processor;
        key.target.sramecc   = hj::TargetFeature::On;
        key.target.xnack     = hj::TargetFeature::Off;
        key.wavefrontSize    = 64;
        key.backendId        = "test";
        key.backendVersion   = backendVersion;
        return key;
    }

    void keys(const Context& ctx)
    {
        const auto key  = testKey(ctx);
        const auto name = key.directoryName();
        JIT_REQUIRE(
            "Unexpected directory " + name,
            name.find(ctx.arch) != std::string::npos && name.find("test") != std::string::npos
                && name.find("_wf64_") != std::string::npos && name.find("=3a") != std::string::npos
                && name.find('/') == std::string::npos);
        // A processor different from the device, so this field still changes the key.
        const auto otherProcessor
            = ctx.arch == "gfx942" ? hj::Processor::gfx950 : hj::Processor::gfx942;
        const std::vector<std::function<void(hj::CacheKey&)>> identity{
            [](auto& k) { k.target.sramecc = hj::TargetFeature::Off; },
            [&](auto& k) { k.target.processor = otherProcessor; },
            [](auto& k) { k.wavefrontSize = 32; },
            [](auto& k) { k.backendId = "other"; },
            [](auto& k) { k.backendVersion = 2; },
        };
        for(const auto& change : identity)
        {
            auto changed = key;
            change(changed);
            INFO(("A device or backend field did not change the directory"));
            REQUIRE((changed.directoryName() != name));
        }
        {
            auto changed              = key;
            changed.codeObjectVersion = 5;
            INFO(("The code object version changed the directory"));
            REQUIRE((changed.directoryName() == name));
        }

        const auto problem = gemm(256);
        JIT_REQUIRE("Unexpected problem sizes " + text(hj::problemSizes(problem)),
                    hj::problemSizes(problem) == std::vector<size_t>{256, 128, 1, 512});
        const auto type = hj::problemTypeKey(problem);
        JIT_REQUIRE("Sizes changed the ProblemType key",
                    type == hj::problemTypeKey(gemm(512, 64, 128, 3)));
        auto accumulate = gemm(256);
        accumulate.setHighPrecisionAccumulate(false);
        JIT_REQUIRE("HPA did not change the ProblemType key",
                    type != hj::problemTypeKey(accumulate));

        const std::vector<size_t> sizes{256, 128, 1, 512};
        const auto                prefix = hj::entryPrefix(type, {"kernel"}, sizes);
        JIT_REQUIRE("Unexpected entry prefix " + prefix,
                    prefix.find("k/") == 0 && prefix.find("/kernel/") != std::string::npos
                        && prefix.find("256x128x1x512") != std::string::npos);
        JIT_REQUIRE("Entry prefixes are not a function of their content",
                    prefix == hj::entryPrefix(type, {"kernel", "kernel"}, sizes)
                        && prefix != hj::entryPrefix(type + " ", {"kernel"}, sizes)
                        && prefix != hj::entryPrefix(type, {"kernel2"}, sizes)
                        && prefix != hj::entryPrefix(type, {"kernel", "kernel2"}, sizes)
                        && prefix != hj::entryPrefix(type, {"kernel"}, {256, 128, 1, 1024}));
    }

    void directories(const Context& ctx)
    {
        const auto base = ctx.fresh("directories");
        fs::create_directories(base);
        const auto attempt = [&](const fs::path& root) {
            hj::JitLibrary       library(root);
            std::vector<int32_t> indices;
            return library.lookup(testKey(ctx), 0, gemm(256), ctx.hardware, 1, {}, indices);
        };
        for(unsigned rejected : {0777u, 0770u, 0720u, 0702u})
        {
            const auto root = base / ("mode-" + octal(rejected));
            fs::create_directory(root);
            JIT_REQUIRE("chmod failed", ::chmod(root.c_str(), rejected) == 0);
            const auto status = attempt(root);
            JIT_REQUIRE("A writable root was accepted: " + status.message,
                        !status.ok() && status.stage == hj::Stage::Lookup
                            && status.message.find("JIT solution library disabled") == 0);
            JIT_REQUIRE("A rejected root was written to", fs::is_empty(root));
        }
        for(unsigned accepted : {0700u, 0750u, 0755u})
        {
            const auto root = base / ("mode-" + octal(accepted));
            fs::create_directory(root);
            JIT_REQUIRE("chmod failed", ::chmod(root.c_str(), accepted) == 0);
            ok(attempt(root), "private root");
            JIT_REQUIRE("Library directories are not private",
                        mode(root / "v1") == 0700 && mode(builtUnder(root)) == 0700
                            && mode(builtUnder(root) / testKey(ctx).directoryName()) == 0700);
        }
        const auto missing = base / "missing" / "nested";
        ok(attempt(missing), "missing root");
        JIT_REQUIRE("A created root is not private", mode(missing) == 0700);

        fs::create_directory_symlink(base / "mode-700", base / "link");
        JIT_REQUIRE("A symbolic link root was accepted", !attempt(base / "link").ok());
        std::ofstream(base / "file") << "not a directory";
        JIT_REQUIRE("A file root was accepted", !attempt(base / "file").ok());

        const auto shared = builtUnder(base / "mode-700") / testKey(ctx).directoryName();
        JIT_REQUIRE("chmod failed", ::chmod(shared.c_str(), 0770) == 0);
        JIT_REQUIRE("A group-writable key directory was accepted",
                    !attempt(base / "mode-700").ok());
        JIT_REQUIRE("chmod failed",
                    ::chmod(shared.c_str(), 0700) == 0
                        && ::chmod(builtUnder(base / "mode-700").c_str(), 0707) == 0);
        JIT_REQUIRE("An other-writable build directory was accepted",
                    !attempt(base / "mode-700").ok());
    }

    void roundTrip(const Context& ctx)
    {
        const auto     root = ctx.fresh("round-trip");
        hj::JitLibrary library(root);
        const auto     key     = testKey(ctx);
        const auto     problem = gemm(256);
        JIT_REQUIRE("An empty library returned solutions", ctx.find(library, key, problem).empty());
        const auto indices = publish(library, key, problem, {ctx.entry});
        JIT_REQUIRE("First index " + text(indices), indices == std::vector<int32_t>{base});

        const auto directory = library.directory(key);
        const auto prefix    = hj::entryPrefix(
            hj::problemTypeKey(problem), {ctx.entry.kernel}, hj::problemSizes(problem));
        const auto slice = directory / "slices" / hj::sliceKey(problem);
        JIT_REQUIRE("Unexpected key directory contents",
                    fs::is_regular_file(slice / lazyMaster(ctx.arch))
                        && fs::is_regular_file(slice / lazyMapping(ctx.arch))
                        && fs::is_regular_file(directory / "cache-key.msgpack")
                        && fs::is_regular_file(directory / "indices.msgpack")
                        && fs::is_directory(directory / "staging")
                        && fs::is_regular_file(slice / (prefix + ".co"))
                        && fs::is_regular_file(slice / (prefix + ".dat"))
                        && !fs::exists(directory / "cache-key.json")
                        && !fs::exists(directory / lazyMaster(ctx.arch)));
        std::set<std::string> names;
        for(const auto& entry : fs::directory_iterator(builtUnder(root)))
            names.insert(entry.path().filename().string());
        JIT_REQUIRE("Unexpected build directory contents",
                    names == std::set<std::string>{"allocator.dat", "lock", key.directoryName()});
        JIT_REQUIRE("Publishing left temporary files", fs::is_empty(directory / "staging"));
        {
            hj::msgpack_io::CacheKeyDocument document;
            ok(hj::msgpack_io::readCacheKey(read(directory / "cache-key.msgpack"), document),
               "cache key");
            INFO(("cache-key.msgpack does not hold the key"));
            REQUIRE((document.backendId == key.backendId
                     && document.backendVersion == key.backendVersion
                     && document.targetId == key.target.targetId()
                     && document.wavefrontSize == key.wavefrontSize));
        }
        JIT_REQUIRE("The code object was not stored unchanged",
                    read(slice / (prefix + ".co")) == ctx.entry.executable.bytes);
        int64_t next = 0;
        ok(hj::msgpack_io::readAllocator(read(builtUnder(root) / "allocator.dat"), next),
           "allocator");
        JIT_REQUIRE("The allocator did not advance", next == base + 1);

        const auto master = stock(slice, ctx.arch);
        JIT_REQUIRE("Unexpected index mapping",
                    master->libraryMapping == std::map<int, std::string>{{base, prefix}});
        const auto best = master->findBestSolution(problem, ctx.hardware);
        JIT_REQUIRE("The stock loader did not find the published solution",
                    best && best->index == base && best->kernelName == ctx.entry.kernel
                        && best->codeObjectFilename.load() == prefix + ".co");
        const auto byIndex = master->getSolutionByIndex(ctx.hardware, base);
        JIT_REQUIRE("The stock loader did not resolve the published index", byIndex == best);

        // The solution itself accepts these problems; only the row's sizes reject them.
        for(const auto& other :
            {gemm(264), gemm(256, 136), gemm(256, 128, 1024), gemm(256, 128, 512, 2)})
        {
            JIT_REQUIRE("The solution rejects a nearby size", (*best->problemPredicate)(other));
            JIT_REQUIRE("An entry matched a size it was not published for",
                        !master->findBestSolution(other, ctx.hardware)
                            && ctx.find(library, key, other).empty());
        }
        // Each hit still runs the solution's own predicates.
        auto accumulate = gemm(256);
        accumulate.setHighPrecisionAccumulate(false);
        JIT_REQUIRE("A hit skipped the solution predicates",
                    ctx.find(library, key, accumulate).empty());
        JIT_REQUIRE("Lookup missed the published entry",
                    ctx.find(library, key, problem) == indices);

        hj::Status why;
        const auto view = library.resolve(0, base, why);
        JIT_REQUIRE("resolve: " + why.message, view.master && view.adapter);
        const auto solution = library.solutionByIndex(0, ctx.hardware, base, why);
        JIT_REQUIRE("solutionByIndex: " + why.message,
                    solution && solution->codeObjectFilename.load() == prefix + ".co");
        JIT_REQUIRE("An index resolved on a device that never used the library",
                    !library.resolve(1, base, why).master && !why.ok());
        JIT_REQUIRE("An unpublished index resolved",
                    !library.resolve(0, base + 1, why).master && !why.ok());
    }

    void dedupeAndOrder(const Context& ctx)
    {
        const auto     root = ctx.fresh("order");
        hj::JitLibrary library(root);
        const auto     key     = testKey(ctx);
        const auto     problem = gemm(256);
        const auto     a = ctx.renamed("_A"), b = ctx.renamed("_B"), c = ctx.renamed("_C");
        const auto     first = publish(library, key, problem, {a, b, a});
        JIT_REQUIRE("Batch indices " + text(first),
                    first == std::vector<int32_t>{base, base + 1, base});
        const auto before = tree(root);
        JIT_REQUIRE("Republishing changed indices",
                    publish(library, key, problem, {b, a}) == std::vector<int32_t>{base + 1, base});
        JIT_REQUIRE("Republishing published entries wrote files", tree(root) == before);
        JIT_REQUIRE("A new entry did not take the next index",
                    publish(library, key, problem, {c}) == std::vector<int32_t>{base + 2});
        JIT_REQUIRE("Lookup is not in publication order",
                    ctx.find(library, key, problem)
                        == std::vector<int32_t>{base, base + 1, base + 2});
        JIT_REQUIRE("Lookup ignored the count",
                    ctx.find(library, key, problem, 2) == std::vector<int32_t>{base, base + 1});
        JIT_REQUIRE("Lookup returned an excluded kernel",
                    ctx.find(library, key, problem, 4, {a.kernel})
                        == std::vector<int32_t>{base + 1, base + 2});
        JIT_REQUIRE("Exclusions reduced the count",
                    ctx.find(library, key, problem, 1, {a.kernel, b.kernel})
                        == std::vector<int32_t>{base + 2});
        JIT_REQUIRE("Another size did not get its own entry",
                    publish(library, key, gemm(512), {a}) == std::vector<int32_t>{base + 3}
                        && ctx.find(library, key, gemm(512)) == std::vector<int32_t>{base + 3}
                        && ctx.find(library, key, problem).size() == 3);

        std::vector<int32_t>     seen(4, -1);
        std::vector<std::string> errors(4);
        std::vector<std::thread> threads;
        for(size_t i = 0; i < seen.size(); ++i)
            threads.emplace_back([&, i] {
                try
                {
                    seen[i] = publish(library, key, gemm(1024), {c})[0];
                }
                catch(const std::exception& error)
                {
                    errors[i] = error.what();
                }
            });
        for(auto& thread : threads)
            thread.join();
        for(size_t i = 0; i < seen.size(); ++i)
        {
            JIT_REQUIRE("Concurrent thread " + std::to_string(i) + ": " + errors[i],
                        seen[i] == base + 4);
        }
        consistent(library.directory(key), ctx.hardware, ctx.arch);

        hj::JitLibrary reopened(root);
        JIT_REQUIRE("A reopened library allocated another index for a published entry",
                    publish(reopened, key, problem, {a}) == std::vector<int32_t>{base});
    }

    void mismatch(const Context& ctx)
    {
        const auto root = ctx.fresh("mismatch");
        {
            hj::JitLibrary library(root);
            publish(library, testKey(ctx, 1), gemm(256), {ctx.entry});
        }
        const auto directory = hj::JitLibrary(root).directory(testKey(ctx, 1));
        const auto snapshot  = tree(directory);
        {
            hj::JitLibrary library(root);
            JIT_REQUIRE("Another backend version reused an entry",
                        ctx.find(library, testKey(ctx, 2), gemm(256)).empty());
            JIT_REQUIRE("Indices are not unique across keys",
                        publish(library, testKey(ctx, 2), gemm(256), {ctx.entry})
                            == std::vector<int32_t>{base + 1});
            {
                auto changed              = testKey(ctx, 1);
                changed.codeObjectVersion = 5;
                JIT_REQUIRE("The code object version hid a published entry",
                            ctx.find(library, changed, gemm(256))
                                == ctx.find(library, testKey(ctx, 1), gemm(256)));
            }
            {
                auto changed           = testKey(ctx, 1);
                changed.target.sramecc = hj::TargetFeature::Off;
                INFO(("A different target reused an entry"));
                REQUIRE((ctx.find(library, changed, gemm(256)).empty()));
            }
        }
        JIT_REQUIRE("Another key changed this key's directory", tree(directory) == snapshot);

        {
            const auto                       key = testKey(ctx, 1);
            hj::msgpack_io::CacheKeyDocument document;
            document.backendId      = key.backendId;
            document.backendVersion = key.backendVersion + 1;
            document.targetId       = key.target.targetId();
            document.wavefrontSize  = key.wavefrontSize;
            std::vector<uint8_t> bytes;
            ok(hj::msgpack_io::writeCacheKey(document, bytes), "cache key");
            std::ofstream(directory / "cache-key.msgpack", std::ios::binary | std::ios::trunc)
                .write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
        }
        const auto mismatched = tree(root);
        {
            hj::JitLibrary       library(root);
            std::vector<int32_t> indices;
            auto                 status
                = library.lookup(testKey(ctx, 1), 0, gemm(256), ctx.hardware, 1, {}, indices);
            JIT_REQUIRE("A cache key of another backend version was accepted: " + status.message,
                        !status.ok() && indices.empty()
                            && status.message.find("records a different device or backend")
                                   != std::string::npos);
            status = library.publish(
                testKey(ctx, 1), 0, gemm(256), ctx.entry.executable, {ctx.entry.solution}, indices);
            JIT_REQUIRE("Publishing under a cache key of another backend version was accepted",
                        !status.ok()
                            && status.message.find("records a different device or backend")
                                   != std::string::npos);
            JIT_REQUIRE("A mismatched cache key affected another key",
                        ctx.find(library, testKey(ctx, 2), gemm(256))
                            == std::vector<int32_t>{base + 1});
        }
        JIT_REQUIRE("A directory with a mismatched cache key was modified",
                    tree(root) == mismatched);

        std::ofstream(directory / "cache-key.msgpack", std::ios::trunc) << "{}";
        fs::create_directories(root / "v2" / (ctx.arch + "-0000000000000000"));
        std::ofstream(root / "v2" / "allocator.dat") << "not a library";
        const auto tampered = tree(root);
        {
            hj::JitLibrary       library(root);
            std::vector<int32_t> indices;
            auto                 status
                = library.lookup(testKey(ctx, 1), 0, gemm(256), ctx.hardware, 1, {}, indices);
            JIT_REQUIRE("A tampered cache key was accepted: " + status.message,
                        !status.ok() && indices.empty() && status.stage == hj::Stage::Lookup
                            && status.message.find("cache key") != std::string::npos);
            status = library.publish(
                testKey(ctx, 1), 0, gemm(256), ctx.entry.executable, {ctx.entry.solution}, indices);
            JIT_REQUIRE("Publishing into a tampered directory was accepted",
                        !status.ok() && status.stage == hj::Stage::Publish);
            JIT_REQUIRE("A tampered directory affected another key",
                        ctx.find(library, testKey(ctx, 2), gemm(256))
                            == std::vector<int32_t>{base + 1});
        }
        JIT_REQUIRE("A mismatched directory was modified or deleted", tree(root) == tampered);
    }

    // A prebuilt library disables the JIT library only for a JIT kernel index,
    // not for one user kernel registration assigns.
    void prebuiltIndices()
    {
        using Mapping       = std::map<int, std::string>;
        using Solutions     = std::map<int, hipblaslt_jit_test::Solution>;
        constexpr int local = TensileLite::LocalKernelIndexBase;
        JIT_REQUIRE("A prebuilt index outside the JIT range counted as a JIT index",
                    !hj::holdsJitIndex(Mapping{})
                        && !hj::holdsJitIndex(
                            Mapping{{0, "a"}, {base - 1, "a"}, {local, "b"}, {INT32_MAX, "b"}}));
        JIT_REQUIRE("A prebuilt JIT kernel index was missed",
                    hj::holdsJitIndex(Mapping{{0, "a"}, {base, "a"}})
                        && hj::holdsJitIndex(Mapping{{local - 1, "a"}, {local, "b"}})
                        && hj::holdsJitIndex(Solutions{{base, nullptr}}));
    }

    void allocator(const Context& ctx)
    {
        const auto     root = ctx.fresh("allocator");
        hj::JitLibrary library(root);
        const auto     key = testKey(ctx);
        publish(library, key, gemm(256), {ctx.entry});
        const auto           directory = library.directory(key);
        const auto           allocator = builtUnder(root) / "allocator.dat";
        std::vector<uint8_t> bytes;
        std::vector<int32_t> indices;
        ok(hj::msgpack_io::writeAllocator(int64_t{base} - 1, bytes), "allocator");
        ok(hj::files::writeAtomically(directory / "staging", allocator, bytes), "allocator");
        const auto rejected = library.publish(
            key, 0, gemm(272), ctx.entry.executable, {ctx.entry.solution}, indices);
        INFO(("An allocator below JitKernelIndexBase was used"));
        REQUIRE((!rejected.ok()
                 && rejected.message.find("outside the JIT range") != std::string::npos));

        // An edited entry and mapping can name any index; lookup returns reserved ones only.
        const auto prefix = hj::entryPrefix(
            hj::problemTypeKey(gemm(256)), {ctx.entry.kernel}, hj::problemSizes(gemm(256)));
        const auto slice   = directory / "slices" / hj::sliceKey(gemm(256));
        const auto mapping = slice / lazyMapping(ctx.arch);
        for(const int32_t outside :
            {int32_t{7}, int32_t{TensileLite::LocalKernelIndexBase}, int32_t{INT32_MAX}})
        {
            auto mapped     = TensileLite::LoadLibraryMapping(mapping.string());
            mapped[outside] = prefix;
            ok(hj::files::writeAtomically(directory / "staging",
                                          mapping,
                                          TensileLite::MessagePackWriteLibraryMapping(mapped)),
               "tamper");
            ok(hj::files::writeAtomically(directory / "staging",
                                          slice / (prefix + ".dat"),
                                          entryAs(ctx.entry.solution, outside)),
               "tamper");
            hj::JitLibrary reopened(root);
            INFO(("Lookup returned " + std::to_string(outside) + ", outside the reserved range"));
            REQUIRE((ctx.find(reopened, key, gemm(256)).empty()));
        }
    }

    // Puts bytes back with the same atomic replace publish uses.
    void restore(const fs::path& staging, const fs::path& target, const std::vector<uint8_t>& bytes)
    {
        ok(hj::files::writeAtomically(staging, target, bytes),
           "restore " + target.filename().string());
    }

    // A fresh process. The parent has already called HIP, and HIP is not fork-safe.
    struct ChildCommand
    {
        enum class Role
        {
            None,
            Lock,
            Writer,
            Reader
        } role
            = Role::None;
        fs::path scratch;
        fs::path root;
        fs::path lockFile;
        fs::path done;
        int      writer    = 0;
        int      perWriter = 0;
        int      startFd   = -1;
    };
    ChildCommand g_child;

    std::string executablePath()
    {
        std::string path(PATH_MAX, '\0');
        const auto  length = ::readlink("/proc/self/exe", path.data(), path.size() - 1);
        if(length <= 0)
            throw std::runtime_error("Cannot read /proc/self/exe");
        path.resize(static_cast<size_t>(length));
        return path;
    }

    pid_t spawnChild(const std::vector<std::string>& roleArgs)
    {
        const auto               exe = executablePath();
        std::vector<std::string> storage{exe, "--child"};
        storage.insert(storage.end(), roleArgs.begin(), roleArgs.end());
        std::vector<char*> argv;
        argv.reserve(storage.size() + 1);
        for(auto& word : storage)
            argv.push_back(word.data());
        argv.push_back(nullptr);
        pid_t      pid = -1;
        const auto rc  = ::posix_spawn(&pid, exe.c_str(), nullptr, nullptr, argv.data(), environ);
        JIT_REQUIRE("posix_spawn failed: " + std::to_string(rc), rc == 0 && pid > 0);
        return pid;
    }

    bool childArguments(int argc, char** argv)
    {
        if(argc < 3 || std::string(argv[1]) != "--child")
            return false;
        const std::string role = argv[2];
        const auto        need = [&](int count) {
            if(argc < count)
                throw std::runtime_error("JIT library child is missing arguments");
        };
        if(role == "lock")
        {
            need(4);
            g_child.role     = ChildCommand::Role::Lock;
            g_child.lockFile = argv[3];
        }
        else if(role == "writer")
        {
            need(8);
            g_child.role      = ChildCommand::Role::Writer;
            g_child.scratch   = argv[3];
            g_child.root      = argv[4];
            g_child.writer    = std::stoi(argv[5]);
            g_child.perWriter = std::stoi(argv[6]);
            g_child.startFd   = std::stoi(argv[7]);
        }
        else if(role == "reader")
        {
            need(8);
            g_child.role      = ChildCommand::Role::Reader;
            g_child.scratch   = argv[3];
            g_child.root      = argv[4];
            g_child.perWriter = std::stoi(argv[5]);
            g_child.startFd   = std::stoi(argv[6]);
            g_child.done      = argv[7];
        }
        else
            throw std::runtime_error("Unknown JIT library child role " + role);
        return true;
    }

    // A publish that stopped is the files it had already replaced: the allocator,
    // then the mapping, then the master. Lookup reads the master. A later publish
    // reuses an index the mapping already names, and otherwise takes the next
    // allocator value, leaving a gap when the allocator moved first.
    void interrupted(const Context& ctx)
    {
        enum class Stop
        {
            BeforeStore,
            AfterAllocator,
            AfterMapping,
            AfterMaster,
        };
        const std::pair<Stop, const char*> stops[] = {{Stop::BeforeStore, "before-store"},
                                                      {Stop::AfterAllocator, "after-allocator"},
                                                      {Stop::AfterMapping, "after-mapping"},
                                                      {Stop::AfterMaster, "after-master"}};
        const auto                         key     = testKey(ctx);
        for(const auto& [stop, name] : stops)
        {
            const auto     root = ctx.fresh(std::string("interrupt-") + name);
            hj::JitLibrary library(root);
            JIT_REQUIRE(std::string(name) + " lost the first solution",
                        publish(library, key, gemm(256), {ctx.entry})
                            == std::vector<int32_t>{base});
            const auto directory      = library.directory(key);
            const auto slice          = directory / "slices" / hj::sliceKey(gemm(256));
            const auto staging        = directory / "staging";
            const auto master         = slice / lazyMaster(ctx.arch);
            const auto mapping        = slice / lazyMapping(ctx.arch);
            const auto indices        = directory / "indices.msgpack";
            const auto allocator      = directory.parent_path() / "allocator.dat";
            const auto savedMaster    = read(master);
            const auto savedMapping   = read(mapping);
            const auto savedIndices   = read(indices);
            const auto savedAllocator = read(allocator);
            if(stop != Stop::BeforeStore)
            {
                JIT_REQUIRE(std::string(name) + " did not publish the second solution",
                            publish(library, key, gemm(512), {ctx.entry})
                                == std::vector<int32_t>{base + 1});
            }
            if(stop != Stop::AfterMaster)
                restore(staging, master, savedMaster);
            if(stop == Stop::BeforeStore || stop == Stop::AfterAllocator)
            {
                restore(staging, mapping, savedMapping);
                restore(staging, indices, savedIndices);
            }
            if(stop == Stop::BeforeStore)
                restore(staging, allocator, savedAllocator);

            consistent(directory, ctx.hardware, ctx.arch);
            JIT_REQUIRE(std::string(name) + " lost a solution that was already in the master",
                        ctx.find(library, key, gemm(256)) == std::vector<int32_t>{base});
            JIT_REQUIRE(std::string(name) + " showed a solution the master does not name",
                        ctx.find(library, key, gemm(512)).size()
                            == (stop == Stop::AfterMaster ? 1u : 0u));
            const int32_t expected = stop == Stop::AfterAllocator ? base + 2 : base + 1;
            const auto    again    = publish(library, key, gemm(512), {ctx.entry});
            JIT_REQUIRE(std::string(name) + " republished as " + text(again),
                        again == std::vector<int32_t>{expected});
            JIT_REQUIRE(std::string(name) + " republish is not found",
                        ctx.find(library, key, gemm(512)) == again);
            JIT_REQUIRE(std::string(name) + " did not accept another solution",
                        publish(library, key, gemm(1024), {ctx.entry})[0] == expected + 1);
            consistent(directory, ctx.hardware, ctx.arch);
        }

        const auto lockRoot = ctx.fresh("dead-lock");
        const auto lockKey  = testKey(ctx);
        {
            hj::JitLibrary library(lockRoot);
            publish(library, lockKey, gemm(256), {ctx.entry});
        }
        const auto lockFile = hj::JitLibrary(lockRoot).directory(lockKey).parent_path() / "lock";
        std::cout.flush();
        const auto child  = spawnChild({"lock", lockFile.string()});
        int        status = 0;
        JIT_REQUIRE("The lock holder did not exit",
                    ::waitpid(child, &status, 0) == child && WIFEXITED(status)
                        && WEXITSTATUS(status) == 0);
        {
            hj::JitLibrary library(lockRoot);
            INFO(("A lock held by an exited process blocked publication"));
            REQUIRE((publish(library, lockKey, gemm(512), {ctx.entry})[0] == base + 1));
        }
    }

    void refresh(const Context& ctx)
    {
        const auto     root = ctx.fresh("refresh");
        const auto     key  = testKey(ctx);
        hj::JitLibrary reader(root), writer(root);
        JIT_REQUIRE("An empty library returned solutions",
                    ctx.find(reader, key, gemm(256)).empty());
        const auto first = publish(writer, key, gemm(256), {ctx.entry});
        JIT_REQUIRE("A reader did not see another instance's entry",
                    ctx.find(reader, key, gemm(256)) == first);
        hj::Status why;
        const auto old = reader.resolve(0, first[0], why);
        JIT_REQUIRE("resolve: " + why.message, old.master != nullptr);
        const auto second = publish(writer, key, gemm(512), {ctx.entry});
        const auto view   = reader.resolve(0, second[0], why);
        JIT_REQUIRE("resolve did not reload for a newer index: " + why.message,
                    view.master && view.master != old.master && view.adapter == old.adapter);
        JIT_REQUIRE("A superseded snapshot lost its solution",
                    old.master->getSolutionByIndex(ctx.hardware, first[0]) != nullptr);

        hj::JitLibrary discovering(root, [&ctx](int, hj::CacheKey& k) {
            k = testKey(ctx);
            return hj::Status{};
        });
        JIT_REQUIRE("An index was not found without a lookup: " + why.message,
                    discovering.solutionByIndex(0, ctx.hardware, second[0], why) != nullptr);
    }

    void fusedA2A(const Context& ctx)
    {
        const auto root  = ctx.fresh("fused-a2a");
        const auto key   = testKey(ctx);
        auto       fused = gemm(256);
        fused.setFusedGemmA2A(true);
        fused.setFusedA2AExtent(256);
        fused.setWorld(2);
        const auto rejected = [](const hj::Status& status, hj::Stage stage) {
            return status.code == hj::Status::Code::NotSupported && status.stage == stage
                   && status.message.find("fused GEMM and all-to-all") != std::string::npos;
        };
        std::vector<int32_t> indices;
        {
            hj::JitLibrary library(root);
            auto           status = library.lookup(key, 0, fused, ctx.hardware, 1, {}, indices);
            JIT_REQUIRE("A fused all-to-all lookup was not rejected: " + status.message,
                        rejected(status, hj::Stage::Lookup) && indices.empty());
            status = library.publish(
                key, 0, fused, ctx.entry.executable, {ctx.entry.solution}, indices);
            JIT_REQUIRE("A fused all-to-all publication was not rejected: " + status.message,
                        rejected(status, hj::Stage::Publish) && indices.empty());
        }
        JIT_REQUIRE("Rejecting fused all-to-all touched the library", !fs::exists(root));
        std::string reason;
        try
        {
            hj::problemTypeKey(fused);
        }
        catch(const std::runtime_error& e)
        {
            reason = e.what();
        }
        JIT_REQUIRE("A fused all-to-all problem has a JIT ProblemType: " + reason,
                    reason.find("fused GEMM and all-to-all") != std::string::npos);

        hj::JitLibrary library(root);
        const auto     plain  = publish(library, key, gemm(256), {ctx.entry});
        const auto     status = library.lookup(key, 0, fused, ctx.hardware, 1, {}, indices);
        JIT_REQUIRE("A plain solution of the same sizes served fused all-to-all",
                    rejected(status, hj::Stage::Lookup) && indices.empty());
        JIT_REQUIRE("The plain problem lost its solution",
                    ctx.find(library, key, gemm(256)) == plain);
    }

    ContractionProblemGemm sharedProblem(int j)
    {
        return gemm(256 + 8 * j);
    }
    ContractionProblemGemm ownProblem(int writer, int j, int perWriter)
    {
        return gemm(4096 + 8 * (writer * perWriter + j));
    }

    void writeEntries(const Context&      ctx,
                      const fs::path&     root,
                      const hj::CacheKey& key,
                      const fs::path&     output,
                      int                 writer,
                      int                 perWriter)
    {
        hj::JitLibrary library(root);
        std::ofstream  out(output / ("writer-" + std::to_string(writer) + ".txt"));
        for(int j = 0; j < perWriter; ++j)
        {
            out << "shared " << j << ' ' << publish(library, key, sharedProblem(j), {ctx.entry})[0]
                << '\n';
            out << "own " << j << ' '
                << publish(library, key, ownProblem(writer, j, perWriter), {ctx.entry})[0] << '\n';
        }
    }

    void readEntries(const Context&      ctx,
                     const fs::path&     root,
                     const hj::CacheKey& key,
                     int                 perWriter,
                     const fs::path&     done)
    {
        hj::JitLibrary       library(root);
        std::vector<int32_t> seen(perWriter, -1);
        const auto           deadline = std::chrono::steady_clock::now() + std::chrono::minutes(2);
        for(bool last = false; !last;)
        {
            last = fs::exists(done);
            JIT_REQUIRE("The writers did not finish", std::chrono::steady_clock::now() < deadline);
            for(int j = 0; j < perWriter; ++j)
            {
                const auto found = ctx.find(library, key, sharedProblem(j), 1);
                JIT_REQUIRE("A reader saw an entry disappear or change",
                            found.size() <= 1 && (seen[j] < 0 || found == std::vector{seen[j]}));
                if(found.empty())
                    continue;
                seen[j] = found[0];
                hj::Status why;
                JIT_REQUIRE("A reader could not load a found entry: " + why.message,
                            library.solutionByIndex(0, ctx.hardware, found[0], why) != nullptr);
            }
            if(fs::exists(builtUnder(root) / key.directoryName() / lazyMaster(ctx.arch)))
                consistent(builtUnder(root) / key.directoryName(), ctx.hardware, ctx.arch);
        }
        JIT_REQUIRE("The reader never saw every shared entry",
                    std::find(seen.begin(), seen.end(), -1) == seen.end());
    }

    void concurrency(const Context& ctx, int writers, int perWriter)
    {
        const auto root = ctx.fresh("concurrency");
        const auto key  = testKey(ctx);
        const auto done = ctx.scratch / "concurrency-writers-done";
        fs::remove(done);
        int start[2];
        JIT_REQUIRE("pipe failed", ::pipe(start) == 0);
        std::vector<pid_t> children;
        for(int w = 0; w < writers; ++w)
            children.push_back(spawnChild({"writer",
                                           ctx.scratch.string(),
                                           root.string(),
                                           std::to_string(w),
                                           std::to_string(perWriter),
                                           std::to_string(start[0])}));
        children.push_back(spawnChild({"reader",
                                       ctx.scratch.string(),
                                       root.string(),
                                       std::to_string(perWriter),
                                       std::to_string(start[0]),
                                       done.string()}));
        ::close(start[0]);
        const std::string go(children.size(), 'g');
        JIT_REQUIRE("Cannot start the children",
                    ::write(start[1], go.data(), go.size()) == static_cast<ssize_t>(go.size()));
        ::close(start[1]);
        const auto wait = [](pid_t child) {
            int status = 0;
            return ::waitpid(child, &status, 0) == child && WIFEXITED(status)
                   && WEXITSTATUS(status) == 0;
        };
        bool passed = true;
        for(int w = 0; w < writers; ++w)
            passed = wait(children[w]) && passed;
        std::ofstream(done) << "done";
        passed = wait(children.back()) && passed;
        JIT_REQUIRE("A concurrent publisher or reader failed", passed);

        std::map<int, std::set<int32_t>> sharedIndices;
        std::set<int32_t>                all;
        size_t                           owned = 0;
        for(int w = 0; w < writers; ++w)
        {
            std::ifstream in(ctx.scratch / ("writer-" + std::to_string(w) + ".txt"));
            std::string   kind;
            int           j     = 0;
            int32_t       index = 0;
            while(in >> kind >> j >> index)
            {
                if(kind == "shared")
                    sharedIndices[j].insert(index);
                else
                {
                    JIT_REQUIRE("Two writers got the same index", !all.count(index));
                    ++owned;
                }
                all.insert(index);
            }
        }
        JIT_REQUIRE("A writer did not record every result",
                    owned == size_t(writers) * perWriter
                        && sharedIndices.size() == size_t(perWriter));
        for(const auto& [j, indices] : sharedIndices)
        {
            JIT_REQUIRE("Writers got different indices for one entry", indices.size() == 1);
        }
        const auto        directory = builtUnder(root) / key.directoryName();
        const auto        master = stock(directory / "slices" / hj::sliceKey(gemm(256)), ctx.arch);
        std::set<int32_t> mapped;
        for(const auto& [index, prefix] : master->libraryMapping)
            mapped.insert(index);
        JIT_REQUIRE("The mapping does not hold exactly the returned indices", mapped == all);
        int64_t next = 0;
        ok(hj::msgpack_io::readAllocator(read(builtUnder(root) / "allocator.dat"), next),
           "allocator");
        JIT_REQUIRE("Deduplicated entries consumed indices",
                    next == base + static_cast<int64_t>(all.size()) && *all.rbegin() == next - 1);
        JIT_REQUIRE("Publishers left temporary files", fs::is_empty(directory / "staging"));
        consistent(directory, ctx.hardware, ctx.arch);
        hj::JitLibrary library(root);
        for(int j = 0; j < perWriter; ++j)
        {
            JIT_REQUIRE("A shared entry has more than one row",
                        ctx.find(library, key, sharedProblem(j))
                            == std::vector<int32_t>{*sharedIndices[j].begin()});
            for(int w = 0; w < writers; ++w)
            {
                JIT_REQUIRE("A writer's entry is lost",
                            ctx.find(library, key, ownProblem(w, j, perWriter)).size() == 1);
            }
        }
    }
}

Context openContext(const fs::path& scratch)
{
    Context ctx;
    ctx.scratch = scratch;
    fs::create_directories(ctx.scratch);

    const auto        target      = hipblaslt_jit_test::currentDeviceTarget();
    const std::string gcnArchName = target.properties.gcnArchName;
    ctx.arch                      = gcnArchName.substr(0, gcnArchName.find(':'));
    const auto processor          = TensileLite::AMDGPU::toProcessor(ctx.arch);
    JIT_REQUIRE("No TensileLite processor for " + ctx.arch,
                TensileLite::AMDGPU::toString(processor) == ctx.arch);
    ctx.hardware = TensileLite::AMDGPU(processor, 256, ctx.arch);
    ctx.source
        = hj::source_bundle::readSourceBundle(fs::u8path(HIPBLASLT_JIT_DATA) / ctx.arch / "plain");
    ctx.source.target            = target;
    ctx.source.codeObjectVersion = hj::jitCodeObjectVersion;
    ctx.entry                    = build(ctx.source, ctx.scratch);
    const auto& replayed         = *ctx.entry.solution;
    if(!(*replayed.problemPredicate)(gemm(256)))
    {
        replayed.problemPredicate->debugEval(gemm(256), std::cerr);
        throw std::runtime_error("The test problem does not match the plain solution");
    }
    if(!(*replayed.hardwarePredicate)(ctx.hardware))
    {
        replayed.hardwarePredicate->debugEval(ctx.hardware, std::cerr);
        throw std::runtime_error("The " + ctx.arch + " plain solution rejects this device");
    }
    return ctx;
}

void runLibrary(const fs::path& scratch, int writers, int perWriter)
{
    fs::remove_all(scratch);
    Context ctx = openContext(scratch);
    if(writers)
        concurrency(ctx, writers, perWriter);
    else
    {
        keys(ctx);
        directories(ctx);
        roundTrip(ctx);
        dedupeAndOrder(ctx);
        mismatch(ctx);
        prebuiltIndices();
        allocator(ctx);
        interrupted(ctx);
        refresh(ctx);
        fusedA2A(ctx);
    }
}

TEST_CASE("the JIT solution library", "[jit-gpu][library-checks]")
{
    hipblaslt_jit_test::ScratchDir scratch("library");
    runLibrary(scratch.path, 0, 0);
}

TEST_CASE("concurrent JIT solution library publishers", "[jit-gpu][library-concurrency]")
{
    hipblaslt_jit_test::ScratchDir scratch("library-concurrency");
    runLibrary(scratch.path, 8, 4);
}

// Started with --child. The parent has already initialized HIP, so this is a
// new process rather than a fork of that one.
TEST_CASE("JIT library child", "[jit-library-child]")
{
    if(g_child.role == ChildCommand::Role::None)
        return;
    if(g_child.role == ChildCommand::Role::Lock)
    {
        hj::files::FileLock held;
        if(!hj::files::FileLock::acquire(g_child.lockFile, std::chrono::seconds(5), held).ok())
            ::_exit(4);
        ::_exit(0);
    }
    const auto work = g_child.scratch
                      / (g_child.role == ChildCommand::Role::Writer
                             ? "work-writer-" + std::to_string(g_child.writer)
                             : "work-reader");
    auto       ctx = openContext(work);
    const auto key = testKey(ctx);
    if(g_child.startFd >= 0)
    {
        char go = 0;
        if(::read(g_child.startFd, &go, 1) != 1)
            throw std::runtime_error("the parent did not start this child");
    }
    if(g_child.role == ChildCommand::Role::Writer)
        writeEntries(ctx, g_child.root, key, g_child.scratch, g_child.writer, g_child.perWriter);
    else
        readEntries(ctx, g_child.root, key, g_child.perWriter, g_child.done);
}

int main(int argc, char** argv)
{
    try
    {
        if(childArguments(argc, argv))
        {
            char* catchArgv[] = {argv[0], const_cast<char*>("[jit-library-child]")};
            return Catch::Session().run(2, catchArgv);
        }
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAIL " << error.what() << '\n';
        return 2;
    }
    return Catch::Session().run(argc, argv);
}
