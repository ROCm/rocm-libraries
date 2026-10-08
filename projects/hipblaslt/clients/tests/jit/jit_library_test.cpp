// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Exercises the JIT solution library: cache keys, directory
// checks, the stock TensileLite loader reading what the library writes,
// exact-size lookup, index allocation, a library whose publish stopped early,
// concurrent publishers in separate processes, and the rejection of fused GEMM
// and all-to-all.

#include "solution_entry.hpp"
#include "hipblaslt-jit-fs.hpp"
#include "hipblaslt-jit-library.hpp"
#include "hipblaslt-jit-msgpack.hpp"
#include <Tensile/AMDGPU.hpp>
#include <Tensile/Tensile.hpp>
#include <msgpack.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <sstream>
#include <thread>

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace hj = hipblaslt_jit;
namespace fs = std::filesystem;
using TensileLite::ContractionProblemGemm;

namespace
{
    constexpr int32_t base = hj::jitIndexBase;


    void ok(const hj::Status& status, const std::string& what)
    {
        {
            INFO((what + ": " + status.message));
            REQUIRE((status.ok()));
        }
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

    // A one-solution entry and its main kernel.
    struct Entry
    {
        std::vector<uint8_t> bytes;
        std::string          kernel;
    };

    // The entry with its kernel renamed, so one size can have several solutions.
    Entry renamed(const Entry& entry, const std::string& suffix)
    {
        const auto handle
            = msgpack::unpack(reinterpret_cast<const char*>(entry.bytes.data()), entry.bytes.size());
        const auto field = [](const msgpack::object& map, const std::string& key) {
            {
                INFO(("Expected a map"));
                REQUIRE((map.type == msgpack::type::MAP));
            }
            for(uint32_t i = 0; i < map.via.map.size; ++i)
            {
                auto& member = map.via.map.ptr[i];
                if(member.key.type == msgpack::type::STR && member.key.as<std::string>() == key)
                    return &member.val;
            }
            throw std::runtime_error("Missing " + key);
        };
        const auto* solutions = field(handle.get(), "solutions");
        {
            INFO(("Expected one solution"));
            REQUIRE((solutions->type == msgpack::type::ARRAY && solutions->via.array.size == 1));
        }
        auto*      name   = field(solutions->via.array.ptr[0], "kernelName");
        const auto kernel = entry.kernel + suffix;
        name->via.str.ptr  = kernel.data();
        name->via.str.size = static_cast<uint32_t>(kernel.size());
        msgpack::sbuffer buffer;
        msgpack::pack(buffer, handle.get());
        return {std::vector<uint8_t>(buffer.data(), buffer.data() + buffer.size()), kernel};
    }

    // The entry built, with its solution 0 supporting the request.
    std::pair<hj::BuiltSolution, std::vector<int>> built(const Entry& entry)
    {
        hj::BuiltSolution result;
        result.generated.entry       = entry.bytes;
        result.generated.kernelNames = {entry.kernel};
        const std::string object     = "code object of " + entry.kernel;
        result.object.bytes.assign(object.begin(), object.end());
        return {std::move(result), {0}};
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

    // The plain bundle's ProblemType: FP16 NN with FP32 compute and HPA.
    ContractionProblemGemm gemm(size_t m, size_t n = 128, size_t k = 512, size_t batch = 1)
    {
        using rocisa::DataType;
        auto problem = ContractionProblemGemm::GEMM_Strides(false,
                                                            false,
                                                            DataType::Half,
                                                            DataType::Half,
                                                            DataType::Half,
                                                            DataType::Half,
                                                            m,
                                                            n,
                                                            k,
                                                            batch,
                                                            m,
                                                            m * k,
                                                            k,
                                                            k * n,
                                                            m,
                                                            m * n,
                                                            m,
                                                            m * n,
                                                            0.5);
        problem.setComputeInputTypeA(DataType::Half);
        problem.setComputeInputTypeB(DataType::Half);
        problem.setAlphaType(DataType::Float);
        problem.setBetaType(DataType::Float);
        problem.setHighPrecisionAccumulate(true);
        problem.setActivationComputeType(DataType::Float);
        problem.setF32XdlMathOp(DataType::Float);
        problem.setWorkspaceSize(size_t{1} << 30);
        return problem;
    }

    // Loads a key directory with the calls TensileHost makes and no JIT code.
    std::shared_ptr<hj::GemmMaster> stock(const fs::path& directory, const std::string& arch)
    {
        const auto path   = (directory / lazyMaster(arch)).string();
        auto       master = std::dynamic_pointer_cast<hj::GemmMaster>(
            TensileLite::LoadLibraryFilePreload<ContractionProblemGemm>(path, {}));
        {
            INFO(("The stock loader cannot read " + path));
            REQUIRE((master && master->initLibraryMapping(path)));
        }
        return master;
    }

    // Every master row is mapped, and every mapped index loads with its code object.
    void consistentSlice(const fs::path&              directory,
                         const TensileLite::Hardware& hardware,
                         const std::string&           arch)
    {
        std::vector<std::string> rows;
        ok(hj::msgpack_io::readMasterPrefixes(read(directory / lazyMaster(arch)), rows),
           "master rows");
        const auto               master = stock(directory, arch);
        std::set<std::string>    mapped;
        for(const auto& [index, prefix] : master->libraryMapping)
        {
            {
                INFO(("Mapped index outside the JIT range"));
                REQUIRE((hj::isJitIndex(index)));
            }
            {
                INFO(("Two indices map " + prefix));
                REQUIRE((mapped.insert(prefix).second));
            }
            {
                INFO(("Missing " + prefix + ".co"));
                REQUIRE((fs::is_regular_file(directory / (prefix + ".co"))));
            }
            const auto solution = master->getSolutionByIndex(hardware, index);
            {
                INFO(("Mapped index " + std::to_string(index) + " does not load"));
                REQUIRE((solution && solution->index == index
                    && solution->codeObjectFilename.load() == prefix + ".co"));
            }
        }
        {
            INFO(("Duplicate master rows"));
            REQUIRE((std::set<std::string>(rows.begin(), rows.end()).size() == rows.size()));
        }
        for(const auto& row : rows)
        {
            INFO(("Master row " + row + " is not mapped"));
            REQUIRE((mapped.count(row)));
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
            struct stat info{};
            {
                INFO(("lstat failed"));
                REQUIRE((::lstat(entry.path().c_str(), &info) == 0));
            }
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
        struct stat info{};
        {
            INFO(("lstat " + path.string()));
            REQUIRE((::lstat(path.c_str(), &info) == 0));
        }
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
        fs::path            scratch;
        Entry               entry;
        std::string         arch; // processor name of device 0, such as "gfx942"
        TensileLite::AMDGPU hardware{};

        fs::path fresh(const std::string& name) const
        {
            auto path = scratch / name;
            fs::remove_all(path);
            return path;
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
        hj::SupportedSolutions solutions;
        for(const auto& entry : entries)
            solutions.push_back(built(entry));
        std::vector<int32_t> indices;
        ok(library.publish(key, 0, problem, solutions, indices), "publish");
        {
            INFO(("Publish returned " + text(indices)));
            REQUIRE((indices.size() == entries.size()));
        }
        return indices;
    }

    // gfx90a, gfx942 and gfx950, and the committed plain bundles, are wave64.
    // The target id is the device processor with the sramecc+/xnack- feature spelling.
    hj::CacheKey testKey(const Context& ctx, const std::string& backendVersion = "1")
    {
        hj::CacheKey key;
        key.targetId       = ctx.arch + ":sramecc+:xnack-";
        key.isa            = ctx.arch;
        key.libraryArch    = ctx.arch;
        key.wavefrontSize  = 64;
        key.backendId      = "test";
        key.backendVersion = backendVersion;
        key.comgr          = "3.0:/opt/rocm/lib/libamd_comgr.so:1:2";
        key.rocmPath       = "/opt/rocm";
        key.environment    = {{"LLVM_PATH", "/opt/rocm/llvm"}};
        return key;
    }

    void keys(const Context& ctx)
    {
        const auto key = testKey(ctx);
        const auto name = key.directoryName();
        {
            INFO(("Unexpected directory " + name));
            REQUIRE((name.find(ctx.arch) != std::string::npos && name.find("test") != std::string::npos
                && name.find("_wf64_") != std::string::npos && name.find("=3a") != std::string::npos
                && name.find('/') == std::string::npos));
        }
        // An ISA different from the device, so this field still changes the key.
        const auto otherIsa = ctx.arch == "gfx942" ? "gfx950" : "gfx942";
        const std::vector<std::function<void(hj::CacheKey&)>> identity{
            [&](auto& k) { k.targetId = ctx.arch + ":sramecc-:xnack-"; },
            [&](auto& k) { k.isa = otherIsa; },
            [](auto& k) { k.libraryArch += "v1"; },
            [](auto& k) { k.wavefrontSize = 32; },
            [](auto& k) { k.backendId = "other"; },
            [](auto& k) { k.backendVersion = "2"; },
        };
        const std::vector<std::function<void(hj::CacheKey&)>> toolchain{
            [](auto& k) { k.codeObjectVersion = 5; },
            [](auto& k) { k.comgr = "3.1"; },
            [](auto& k) { k.rocmPath = "/opt/rocm-other"; },
            [](auto& k) { k.environment["AMD_COMGR_DRIVER_OPTIONS_APPEND"] = "-O0"; },
        };
        for(const auto& change : identity)
        {
            auto changed = key;
            change(changed);
            INFO(("A device or backend field did not change the directory"));
            REQUIRE((changed.directoryName() != name));
        }
        for(const auto& change : toolchain)
        {
            auto changed = key;
            change(changed);
            INFO(("A toolchain field changed the directory"));
            REQUIRE((changed.directoryName() == name));
        }

        const char* environment[] = {"HIP_PATH=/hip",
                                     "LLVM_PATH=/llvm",
                                     "AMD_COMGR_DRIVER_OPTIONS_APPEND=-O3",
                                     "AMD_COMGR_HOTSWAP_PLUGIN=/plugin.so",
                                     "AMD_COMGR_CACHE=0",
                                     "AMD_COMGR_CACHE_DIR=/cache",
                                     "AMD_COMGR_CACHE_POLICY=prune",
                                     "AMD_COMGR_SAVE_TEMPS=1",
                                     "AMD_COMGR_SAVE_LLVM_TEMPS=1",
                                     "AMD_COMGR_REDIRECT_LOGS=stderr",
                                     "AMD_COMGR_EMIT_VERBOSE_LOGS=1",
                                     "AMD_COMGR_LOG_LEVEL=3",
                                     "AMD_COMGR_TIME_STATISTICS=1",
                                     "AMD_COMGR_TIME_STATISTICS_GRANULARITY=1",
                                     "AMD_COMGR_USE_VFS=0",
                                     "ROCM_PATH=/rocm",
                                     "PATH=/bin",
                                     "MALFORMED",
                                     nullptr};
        const std::map<std::string, std::string> expected{
            {"AMD_COMGR_DRIVER_OPTIONS_APPEND", "-O3"},
            {"AMD_COMGR_HOTSWAP_PLUGIN", "/plugin.so"},
            {"HIP_PATH", "/hip"},
            {"LLVM_PATH", "/llvm"}};
        {
            INFO(("compilerEnvironment kept the wrong variables"));
            REQUIRE((hj::compilerEnvironment(environment) == expected));
        }

        const auto problem = gemm(256);
        {
            INFO(("Unexpected problem sizes " + text(hj::problemSizes(problem))));
            REQUIRE((hj::problemSizes(problem) == std::vector<size_t>{256, 128, 1, 512}));
        }
        const auto type = hj::problemTypeKey(problem);
        {
            INFO(("Sizes changed the ProblemType key"));
            REQUIRE((type == hj::problemTypeKey(gemm(512, 64, 128, 3))));
        }
        auto accumulate = gemm(256);
        accumulate.setHighPrecisionAccumulate(false);
        {
            INFO(("HPA did not change the ProblemType key"));
            REQUIRE((type != hj::problemTypeKey(accumulate)));
        }

        const std::vector<size_t> sizes{256, 128, 1, 512};
        const auto                prefix = hj::entryPrefix(type, "kernel", sizes);
        {
            INFO(("Unexpected entry prefix " + prefix));
            REQUIRE((prefix.find("k/") == 0 && prefix.find("/kernel/") != std::string::npos
                && prefix.find("256x128x1x512") != std::string::npos));
        }
        {
            INFO(("Entry prefixes are not a function of their content"));
            REQUIRE((prefix == hj::entryPrefix(type, "kernel", sizes)
                && prefix != hj::entryPrefix(type + " ", "kernel", sizes)
                && prefix != hj::entryPrefix(type, "kernel2", sizes)
                && prefix != hj::entryPrefix(type, "kernel", {256, 128, 1, 1024})));
        }
        std::cout << "PASS cache keys, compiler environment, ProblemType keys and entry names\n";
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
            {
                INFO(("chmod failed"));
                REQUIRE((::chmod(root.c_str(), rejected) == 0));
            }
            const auto status = attempt(root);
            {
                INFO(("A writable root was accepted: " + status.message));
                REQUIRE((!status.ok() && status.stage == hj::Stage::Lookup
                    && status.message.find("JIT solution library disabled") == 0));
            }
            {
                INFO(("A rejected root was written to"));
                REQUIRE((fs::is_empty(root)));
            }
        }
        for(unsigned accepted : {0700u, 0750u, 0755u})
        {
            const auto root = base / ("mode-" + octal(accepted));
            fs::create_directory(root);
            {
                INFO(("chmod failed"));
                REQUIRE((::chmod(root.c_str(), accepted) == 0));
            }
            ok(attempt(root), "private root");
            {
                INFO(("Library directories are not private"));
                REQUIRE((mode(root / "v1") == 0700
                    && mode(root / "v1" / testKey(ctx).directoryName()) == 0700));
            }
        }
        const auto missing = base / "missing" / "nested";
        ok(attempt(missing), "missing root");
        {
            INFO(("A created root is not private"));
            REQUIRE((mode(missing) == 0700));
        }

        fs::create_directory_symlink(base / "mode-700", base / "link");
        {
            INFO(("A symbolic link root was accepted"));
            REQUIRE((!attempt(base / "link").ok()));
        }
        std::ofstream(base / "file") << "not a directory";
        {
            INFO(("A file root was accepted"));
            REQUIRE((!attempt(base / "file").ok()));
        }

        const auto shared = base / "mode-700" / "v1" / testKey(ctx).directoryName();
        {
            INFO(("chmod failed"));
            REQUIRE((::chmod(shared.c_str(), 0770) == 0));
        }
        {
            INFO(("A group-writable key directory was accepted"));
            REQUIRE((!attempt(base / "mode-700").ok()));
        }
        {
            INFO(("chmod failed"));
            REQUIRE((::chmod(shared.c_str(), 0700) == 0
                && ::chmod((base / "mode-700" / "v1").c_str(), 0707) == 0));
        }
        {
            INFO(("An other-writable schema directory was accepted"));
            REQUIRE((!attempt(base / "mode-700").ok()));
        }
        std::cout << "PASS group- and other-writable, linked and non-directory roots are rejected; "
                     "created directories are 0700\n";
    }

    void roundTrip(const Context& ctx)
    {
        const auto     root = ctx.fresh("round-trip");
        hj::JitLibrary library(root);
        const auto     key     = testKey(ctx);
        const auto     problem = gemm(256);
        {
            INFO(("An empty library returned solutions"));
            REQUIRE((ctx.find(library, key, problem).empty()));
        }
        const auto indices = publish(library, key, problem, {ctx.entry});
        {
            INFO(("First index " + text(indices)));
            REQUIRE((indices == std::vector<int32_t>{base}));
        }

        const auto directory = library.directory(key);
        const auto prefix    = hj::entryPrefix(
            hj::problemTypeKey(problem), ctx.entry.kernel, hj::problemSizes(problem));
        const auto slice = directory / "slices" / hj::sliceKey(problem);
        {
            INFO(("Unexpected key directory contents"));
            REQUIRE((fs::is_regular_file(slice / lazyMaster(ctx.arch))
                && fs::is_regular_file(slice / lazyMapping(ctx.arch))
                && fs::is_regular_file(directory / "cache-key.msgpack")
                && fs::is_regular_file(directory / "indices.msgpack")
                && fs::is_directory(directory / "staging")
                && fs::is_regular_file(slice / (prefix + ".co"))
                && fs::is_regular_file(slice / (prefix + ".dat"))
                && !fs::exists(directory / "cache-key.json")
                && !fs::exists(directory / lazyMaster(ctx.arch))));
        }
        std::set<std::string> names;
        for(const auto& entry : fs::directory_iterator(root / "v1"))
            names.insert(entry.path().filename().string());
        {
            INFO(("Unexpected schema directory contents"));
            REQUIRE((names == std::set<std::string>{"allocator.dat", "lock", key.directoryName()}));
        }
        {
            INFO(("Publishing left temporary files"));
            REQUIRE((fs::is_empty(directory / "staging")));
        }
        {
            hj::msgpack_io::CacheKeyDocument document;
            ok(hj::msgpack_io::readCacheKey(read(directory / "cache-key.msgpack"), document),
               "cache key");
            INFO(("cache-key.msgpack does not hold the key"));
            REQUIRE((document.backendId == key.backendId && document.backendVersion == key.backendVersion
                && document.isa == key.isa && document.targetId == key.targetId
                && document.libraryArch == key.libraryArch && document.wavefrontSize == key.wavefrontSize
                && document.comgr == key.comgr && document.rocmPath == key.rocmPath
                && document.codeObjectVersion == key.codeObjectVersion));
        }
        {
            INFO(("The code object was not stored unchanged"));
            REQUIRE((read(slice / (prefix + ".co")) == built(ctx.entry).first.object.bytes));
        }
        int64_t next = 0;
        ok(hj::msgpack_io::readAllocator(read(root / "v1" / "allocator.dat"), next), "allocator");
        {
            INFO(("The allocator did not advance"));
            REQUIRE((next == base + 1));
        }

        const auto master = stock(slice, ctx.arch);
        {
            INFO(("Unexpected index mapping"));
            REQUIRE((master->libraryMapping == std::map<int, std::string>{{base, prefix}}));
        }
        const auto best = master->findBestSolution(problem, ctx.hardware);
        {
            INFO(("The stock loader did not find the published solution"));
            REQUIRE((best && best->index == base && best->kernelName == ctx.entry.kernel
                && best->codeObjectFilename.load() == prefix + ".co"));
        }
        const auto byIndex = master->getSolutionByIndex(ctx.hardware, base);
        {
            INFO(("The stock loader did not resolve the published index"));
            REQUIRE((byIndex == best));
        }

        // The solution itself accepts these problems; only the row's sizes reject them.
        for(const auto& other : {gemm(264), gemm(256, 136), gemm(256, 128, 1024), gemm(256, 128, 512, 2)})
        {
            {
                INFO(("The solution rejects a nearby size"));
                REQUIRE(((*best->problemPredicate)(other)));
            }
            {
                INFO(("An entry matched a size it was not published for"));
                REQUIRE((!master->findBestSolution(other, ctx.hardware)
                    && ctx.find(library, key, other).empty()));
            }
        }
        // Each hit still runs the solution's own predicates.
        auto accumulate = gemm(256);
        accumulate.setHighPrecisionAccumulate(false);
        {
            INFO(("A hit skipped the solution predicates"));
            REQUIRE((ctx.find(library, key, accumulate).empty()));
        }
        {
            INFO(("Lookup missed the published entry"));
            REQUIRE((ctx.find(library, key, problem) == indices));
        }

        hj::Status why;
        const auto view = library.resolve(0, base, why);
        {
            INFO(("resolve: " + why.message));
            REQUIRE((view.master && view.adapter));
        }
        const auto solution = library.solutionByIndex(0, ctx.hardware, base, why);
        {
            INFO(("solutionByIndex: " + why.message));
            REQUIRE((solution && solution->codeObjectFilename.load() == prefix + ".co"));
        }
        {
            INFO(("An index resolved on a device that never used the library"));
            REQUIRE((!library.resolve(1, base, why).master && !why.ok()));
        }
        {
            INFO(("A prebuilt index resolved in the JIT library"));
            REQUIRE((!library.resolve(0, 5, why).master
                && why.message.find("outside the reserved JIT range") != std::string::npos));
        }
        {
            INFO(("An unpublished index resolved"));
            REQUIRE((!library.resolve(0, base + 1, why).master && !why.ok()));
        }
        std::cout << "PASS the stock loader reads the published library; exact sizes only, and "
                     "solution predicates still run\n";
    }

    void dedupeAndOrder(const Context& ctx)
    {
        const auto     root = ctx.fresh("order");
        hj::JitLibrary library(root);
        const auto     key     = testKey(ctx);
        const auto     problem = gemm(256);
        const auto     a = renamed(ctx.entry, "_A"), b = renamed(ctx.entry, "_B"),
                   c    = renamed(ctx.entry, "_C");
        const auto first = publish(library, key, problem, {a, b, a});
        {
            INFO(("Batch indices " + text(first)));
            REQUIRE((first == std::vector<int32_t>{base, base + 1, base}));
        }
        const auto before = tree(root);
        {
            INFO(("Republishing changed indices"));
            REQUIRE((
                publish(library, key, problem, {b, a}) == std::vector<int32_t>{base + 1, base}));
        }
        {
            INFO(("Republishing published entries wrote files"));
            REQUIRE((tree(root) == before));
        }
        {
            INFO(("A new entry did not take the next index"));
            REQUIRE((publish(library, key, problem, {c}) == std::vector<int32_t>{base + 2}));
        }
        {
            INFO(("Lookup is not in publication order"));
            REQUIRE((
                ctx.find(library, key, problem) == std::vector<int32_t>{base, base + 1, base + 2}));
        }
        {
            INFO(("Lookup ignored the count"));
            REQUIRE((ctx.find(library, key, problem, 2) == std::vector<int32_t>{base, base + 1}));
        }
        {
            INFO(("Lookup returned an excluded kernel"));
            REQUIRE((ctx.find(library, key, problem, 4, {a.kernel})
                == std::vector<int32_t>{base + 1, base + 2}));
        }
        {
            INFO(("Exclusions reduced the count"));
            REQUIRE((ctx.find(library, key, problem, 1, {a.kernel, b.kernel})
                == std::vector<int32_t>{base + 2}));
        }
        {
            INFO(("Another size did not get its own entry"));
            REQUIRE((publish(library, key, gemm(512), {a}) == std::vector<int32_t>{base + 3}
                && ctx.find(library, key, gemm(512)) == std::vector<int32_t>{base + 3}
                && ctx.find(library, key, problem).size() == 3));
        }

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
            INFO(("Concurrent thread " + std::to_string(i) + ": " + errors[i]));
            REQUIRE((seen[i] == base + 4));
        }
        consistent(library.directory(key), ctx.hardware, ctx.arch);

        // The same kernel, problem type and sizes name one entry. Rewriting that
        // file does not allocate another index.
        const auto directory = library.directory(key);
        const auto prefix
            = hj::entryPrefix(hj::problemTypeKey(gemm(2048)), a.kernel, hj::problemSizes(gemm(2048)));
        const auto collided = publish(library, key, gemm(2048), {a})[0];
        std::vector<uint8_t> other;
        ok(hj::msgpack_io::rewriteEntryIndex(b.bytes, 0, collided, other), "rewrite");
        const auto slice = directory / "slices" / hj::sliceKey(gemm(2048));
        ok(hj::files::writeAtomically(directory / "staging", slice / (prefix + ".dat"), other),
           "replace entry");
        hj::JitLibrary reopened(root);
        const auto     renamedIndex = publish(reopened, key, gemm(2048), {a})[0];
        {
            INFO(("The same identity allocated another index"));
            REQUIRE((renamedIndex == collided && !fs::exists(slice / (prefix + "_1.dat"))));
        }
        consistent(directory, ctx.hardware, ctx.arch);
        std::cout << "PASS deduplication, collisions, publication order, top-N and exclusions\n";
    }

    void mismatch(const Context& ctx)
    {
        const auto root = ctx.fresh("mismatch");
        {
            hj::JitLibrary library(root);
            publish(library, testKey(ctx, "1"), gemm(256), {ctx.entry});
        }
        const auto directory = hj::JitLibrary(root).directory(testKey(ctx, "1"));
        const auto snapshot  = tree(directory);
        {
            hj::JitLibrary library(root);
            {
                INFO(("Another backend version reused an entry"));
                REQUIRE((ctx.find(library, testKey(ctx, "2"), gemm(256)).empty()));
            }
            {
                INFO(("Indices are not unique across keys"));
                REQUIRE((publish(library, testKey(ctx, "2"), gemm(256), {ctx.entry})
                    == std::vector<int32_t>{base + 1}));
            }
            const auto published = ctx.find(library, testKey(ctx, "1"), gemm(256));
            const std::vector<std::function<void(hj::CacheKey&)>> toolchain{
                [](auto& k) { k.comgr = "3.1"; },
                [](auto& k) { k.rocmPath = "/opt/rocm-other"; },
                [](auto& k) { k.environment["AMD_COMGR_DRIVER_OPTIONS_APPEND"] = "-O0"; },
                [](auto& k) { k.codeObjectVersion = 5; },
            };
            for(const auto& change : toolchain)
            {
                auto changed = testKey(ctx, "1");
                change(changed);
                {
                    INFO(("A toolchain field hid a published entry"));
                    REQUIRE((ctx.find(library, changed, gemm(256)) == published));
                }
            }
            {
                auto changed = testKey(ctx, "1");
                changed.targetId = ctx.arch + ":sramecc-:xnack-";
                INFO(("A different target reused an entry"));
                REQUIRE((ctx.find(library, changed, gemm(256)).empty()));
            }
        }
        {
            INFO(("Another key changed this key's directory"));
            REQUIRE((tree(directory) == snapshot));
        }

        std::ofstream(directory / "cache-key.msgpack", std::ios::trunc) << "{}";
        fs::create_directories(root / "v2" / (ctx.arch + "-0000000000000000"));
        std::ofstream(root / "v2" / "allocator.dat") << "not a library";
        const auto tampered = tree(root);
        {
            hj::JitLibrary       library(root);
            std::vector<int32_t> indices;
            auto status = library.lookup(testKey(ctx, "1"), 0, gemm(256), ctx.hardware, 1, {}, indices);
            {
                INFO(("A tampered cache key was accepted: " + status.message));
                REQUIRE((!status.ok() && indices.empty() && status.stage == hj::Stage::Lookup
                    && status.message.find("cache key") != std::string::npos));
            }
            status = library.publish(testKey(ctx, "1"), 0, gemm(256), {built(ctx.entry)}, indices);
            {
                INFO(("Publishing into a tampered directory was accepted"));
                REQUIRE((!status.ok() && status.stage == hj::Stage::Publish));
            }
            {
                INFO(("A tampered directory affected another key"));
                REQUIRE((
                    ctx.find(library, testKey(ctx, "2"), gemm(256)) == std::vector<int32_t>{base + 1}));
            }
        }
        {
            INFO(("A mismatched directory was modified or deleted"));
            REQUIRE((tree(root) == tampered));
        }
        std::cout << "PASS mismatched and tampered keys are ignored and left untouched; other "
                     "schemas are never read\n";
    }

    void allocator(const Context& ctx)
    {
        const auto     root = ctx.fresh("allocator");
        hj::JitLibrary library(root);
        const auto     key = testKey(ctx);
        publish(library, key, gemm(256), {ctx.entry});
        const auto           directory = library.directory(key);
        std::vector<uint8_t> bytes;
        ok(hj::msgpack_io::writeAllocator(INT32_MAX, bytes), "allocator");
        ok(hj::files::writeAtomically(directory / "staging", root / "v1" / "allocator.dat", bytes),
           "allocator");
        {
            INFO(("The last reserved index was not allocated"));
            REQUIRE((
                publish(library, key, gemm(264), {ctx.entry}) == std::vector<int32_t>{INT32_MAX}));
        }
        std::vector<int32_t> indices;
        const auto status = library.publish(key, 0, gemm(272), {built(ctx.entry)}, indices);
        {
            INFO(("An exhausted range allocated an index: " + status.message));
            REQUIRE((!status.ok() && status.message.find("exhausted") != std::string::npos));
        }
        {
            INFO(("Published entries stopped resolving when the range was exhausted"));
            REQUIRE((publish(library, key, gemm(264), {ctx.entry}) == std::vector<int32_t>{INT32_MAX}
                && ctx.find(library, key, gemm(264)) == std::vector<int32_t>{INT32_MAX}));
        }

        ok(hj::msgpack_io::writeAllocator(base - 1, bytes), "allocator");
        ok(hj::files::writeAtomically(directory / "staging", root / "v1" / "allocator.dat", bytes),
           "allocator");
        {
            INFO(("An allocator below the reserved range was used"));
            REQUIRE((!library.publish(key, 0, gemm(272), {built(ctx.entry)}, indices).ok()));
        }

        const auto prefix = hj::entryPrefix(
            hj::problemTypeKey(gemm(256)), ctx.entry.kernel, hj::problemSizes(gemm(256)));
        ok(hj::msgpack_io::rewriteEntryIndex(ctx.entry.bytes, 0, 7, bytes), "rewrite");
        const auto slice = directory / "slices" / hj::sliceKey(gemm(256));
        ok(hj::files::writeAtomically(directory / "staging", slice / (prefix + ".dat"), bytes),
           "tamper");
        hj::JitLibrary reopened(root);
        {
            INFO(("Lookup returned an index outside the reserved range"));
            REQUIRE((ctx.find(reopened, key, gemm(256)).empty()));
        }
        std::cout << "PASS index allocation up to INT32_MAX, exhaustion, and out-of-range entries "
                     "dropped\n";
    }

    // Puts bytes back with the same atomic replace publish uses.
    void restore(const fs::path& staging, const fs::path& target, const std::vector<uint8_t>& bytes)
    {
        ok(hj::files::writeAtomically(staging, target, bytes), "restore " + target.filename().string());
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
        const std::pair<Stop, const char*> stops[]
            = {{Stop::BeforeStore, "before-store"},
               {Stop::AfterAllocator, "after-allocator"},
               {Stop::AfterMapping, "after-mapping"},
               {Stop::AfterMaster, "after-master"}};
        const auto key = testKey(ctx);
        for(const auto& [stop, name] : stops)
        {
            const auto     root = ctx.fresh(std::string("interrupt-") + name);
            hj::JitLibrary library(root);
            {
                INFO((std::string(name) + " lost the first solution"));
                REQUIRE((publish(library, key, gemm(256), {ctx.entry}) == std::vector<int32_t>{base}));
            }
            const auto directory = library.directory(key);
            const auto slice     = directory / "slices" / hj::sliceKey(gemm(256));
            const auto staging   = directory / "staging";
            const auto master    = slice / lazyMaster(ctx.arch);
            const auto mapping   = slice / lazyMapping(ctx.arch);
            const auto indices   = directory / "indices.msgpack";
            const auto allocator = directory.parent_path() / "allocator.dat";
            const auto savedMaster    = read(master);
            const auto savedMapping   = read(mapping);
            const auto savedIndices   = read(indices);
            const auto savedAllocator = read(allocator);
            if(stop != Stop::BeforeStore)
            {
                INFO((std::string(name) + " did not publish the second solution"));
                REQUIRE((publish(library, key, gemm(512), {ctx.entry})
                    == std::vector<int32_t>{base + 1}));
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
            {
                INFO((std::string(name) + " lost a solution that was already in the master"));
                REQUIRE((ctx.find(library, key, gemm(256)) == std::vector<int32_t>{base}));
            }
            {
                INFO((std::string(name) + " showed a solution the master does not name"));
                REQUIRE((ctx.find(library, key, gemm(512)).size()
                    == (stop == Stop::AfterMaster ? 1u : 0u)));
            }
            const int32_t expected = stop == Stop::AfterAllocator ? base + 2 : base + 1;
            const auto    again    = publish(library, key, gemm(512), {ctx.entry});
            {
                INFO((std::string(name) + " republished as " + text(again)));
                REQUIRE((again == std::vector<int32_t>{expected}));
            }
            {
                INFO((std::string(name) + " republish is not found"));
                REQUIRE((ctx.find(library, key, gemm(512)) == again));
            }
            {
                INFO((std::string(name) + " did not accept another solution"));
                REQUIRE((publish(library, key, gemm(1024), {ctx.entry})[0] == expected + 1));
            }
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
        const auto child = ::fork();
        {
            INFO(("fork failed"));
            REQUIRE((child >= 0));
        }
        if(child == 0)
        {
            hj::files::FileLock held;
            if(!hj::files::FileLock::acquire(lockFile, std::chrono::seconds(5), held).ok())
                ::_exit(4);
            ::_exit(3);
        }
        int status = 0;
        {
            INFO(("The lock holder did not exit"));
            REQUIRE((::waitpid(child, &status, 0) == child && WIFEXITED(status)
                && WEXITSTATUS(status) == 3));
        }
        {
            hj::JitLibrary library(lockRoot);
            INFO(("A lock held by an exited process blocked publication"));
            REQUIRE((publish(library, lockKey, gemm(512), {ctx.entry})[0] == base + 1));
        }
        std::cout << "PASS a partial library stays consistent, and a lock dies with its process\n";
    }

    void refresh(const Context& ctx)
    {
        const auto     root = ctx.fresh("refresh");
        const auto     key  = testKey(ctx);
        hj::JitLibrary reader(root), writer(root);
        {
            INFO(("An empty library returned solutions"));
            REQUIRE((ctx.find(reader, key, gemm(256)).empty()));
        }
        const auto first = publish(writer, key, gemm(256), {ctx.entry});
        {
            INFO(("A reader did not see another instance's entry"));
            REQUIRE((ctx.find(reader, key, gemm(256)) == first));
        }
        hj::Status why;
        const auto old = reader.resolve(0, first[0], why);
        {
            INFO(("resolve: " + why.message));
            REQUIRE((old.master != nullptr));
        }
        const auto second = publish(writer, key, gemm(512), {ctx.entry});
        const auto view   = reader.resolve(0, second[0], why);
        {
            INFO(("resolve did not reload for a newer index: " + why.message));
            REQUIRE((view.master && view.master != old.master && view.adapter == old.adapter));
        }
        {
            INFO(("A superseded snapshot lost its solution"));
            REQUIRE((old.master->getSolutionByIndex(ctx.hardware, first[0]) != nullptr));
        }

        const auto deviceKey = [&ctx](std::string comgr) {
            return [&ctx, comgr](int, hj::CacheKey& k) {
                k       = testKey(ctx, "");
                k.comgr = comgr.empty() ? k.comgr : comgr;
                return hj::Status{};
            };
        };
        hj::JitLibrary discovering(root, deviceKey(""));
        {
            INFO(("An index was not found without a lookup: " + why.message));
            REQUIRE((discovering.solutionByIndex(0, ctx.hardware, second[0], why) != nullptr));
        }
        hj::JitLibrary other(root, deviceKey("3.1"));
        {
            INFO(("A different comgr path hid a published index: " + why.message));
            REQUIRE((other.resolve(0, second[0], why).master != nullptr));
        }
        std::cout << "PASS readers reload on change, keep superseded snapshots, and find indices "
                     "in any backend's directory\n";
    }

    void fusedA2A(const Context& ctx)
    {
        const auto root  = ctx.fresh("fused-a2a");
        const auto key   = testKey(ctx);
        auto       fused = gemm(256);
        fused.setFusedGemmA2A(true);
        fused.setFusedA2AExtent(256);
        fused.setFusedA2AWorld(2);
        const auto rejected = [](const hj::Status& status, hj::Stage stage) {
            return status.code == hj::Status::Code::NotSupported && status.stage == stage
                   && status.message.find("fused GEMM and all-to-all") != std::string::npos;
        };
        std::vector<int32_t> indices;
        {
            hj::JitLibrary library(root);
            auto status = library.lookup(key, 0, fused, ctx.hardware, 1, {}, indices);
            {
                INFO(("A fused all-to-all lookup was not rejected: " + status.message));
                REQUIRE((rejected(status, hj::Stage::Lookup) && indices.empty()));
            }
            status = library.publish(key, 0, fused, {built(ctx.entry)}, indices);
            {
                INFO(("A fused all-to-all publication was not rejected: " + status.message));
                REQUIRE((rejected(status, hj::Stage::Publish) && indices.empty()));
            }
        }
        {
            INFO(("Rejecting fused all-to-all touched the library"));
            REQUIRE((!fs::exists(root)));
        }
        std::string reason;
        try
        {
            hj::problemTypeKey(fused);
        }
        catch(const std::runtime_error& e)
        {
            reason = e.what();
        }
        {
            INFO(("A fused all-to-all problem has a JIT ProblemType: " + reason));
            REQUIRE((reason.find("fused GEMM and all-to-all") != std::string::npos));
        }

        hj::JitLibrary library(root);
        const auto     plain  = publish(library, key, gemm(256), {ctx.entry});
        const auto     status = library.lookup(key, 0, fused, ctx.hardware, 1, {}, indices);
        {
            INFO(("A plain solution of the same sizes served fused all-to-all"));
            REQUIRE((rejected(status, hj::Stage::Lookup) && indices.empty()));
        }
        {
            INFO(("The plain problem lost its solution"));
            REQUIRE((ctx.find(library, key, gemm(256)) == plain));
        }
        std::cout << "PASS fused GEMM and all-to-all is neither served nor stored, even beside a "
                     "plain solution of the same sizes\n";
    }

    void concurrency(const Context& ctx, int writers, int perWriter)
    {
        const auto root = ctx.fresh("concurrency");
        const auto key  = testKey(ctx);
        const auto done = ctx.scratch / "concurrency-writers-done";
        fs::remove(done);
        int start[2];
        {
            INFO(("pipe failed"));
            REQUIRE((::pipe(start) == 0));
        }
        std::vector<pid_t> children;
        const auto         spawn = [&](const std::function<void()>& body) {
            std::cout.flush();
            const auto child = ::fork();
            {
                INFO(("fork failed"));
                REQUIRE((child >= 0));
            }
            if(child == 0)
            {
                ::close(start[1]);
                char go = 0;
                int  code = ::read(start[0], &go, 1) == 1 ? 0 : 5;
                try
                {
                    if(!code)
                        body();
                }
                catch(const std::exception& error)
                {
                    std::cerr << "FAIL child " << ::getpid() << ": " << error.what() << '\n';
                    code = 1;
                }
                catch(...)
                {
                    std::cerr << "FAIL child " << ::getpid() << '\n';
                    code = 1;
                }
                ::_exit(code);
            }
            children.push_back(child);
        };
        const auto shared = [](int j) { return gemm(256 + 8 * j); };
        const auto own    = [&](int w, int j) { return gemm(4096 + 8 * (w * perWriter + j)); };
        for(int w = 0; w < writers; ++w)
            spawn([&, w] {
                hj::JitLibrary library(root);
                std::ofstream  out(ctx.scratch / ("writer-" + std::to_string(w) + ".txt"));
                for(int j = 0; j < perWriter; ++j)
                {
                    out << "shared " << j << ' ' << publish(library, key, shared(j), {ctx.entry})[0]
                        << '\n';
                    out << "own " << j << ' ' << publish(library, key, own(w, j), {ctx.entry})[0]
                        << '\n';
                }
            });
        spawn([&] {
            hj::JitLibrary       library(root);
            std::vector<int32_t> seen(perWriter, -1);
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::minutes(2);
            for(bool last = false; !last;)
            {
                last = fs::exists(done);
                {
                    INFO(("The writers did not finish"));
                    REQUIRE((std::chrono::steady_clock::now() < deadline));
                }
                for(int j = 0; j < perWriter; ++j)
                {
                    const auto found = ctx.find(library, key, shared(j), 1);
                    {
                        INFO(("A reader saw an entry disappear or change"));
                        REQUIRE((
                            found.size() <= 1 && (seen[j] < 0 || found == std::vector{seen[j]})));
                    }
                    if(found.empty())
                        continue;
                    seen[j] = found[0];
                    hj::Status why;
                    {
                        INFO(("A reader could not load a found entry: " + why.message));
                        REQUIRE((
                            library.solutionByIndex(0, ctx.hardware, found[0], why) != nullptr));
                    }
                }
                if(fs::exists(root / "v1" / key.directoryName() / lazyMaster(ctx.arch)))
                    consistent(root / "v1" / key.directoryName(), ctx.hardware, ctx.arch);
            }
            {
                INFO(("The reader never saw every shared entry"));
                REQUIRE((std::find(seen.begin(), seen.end(), -1) == seen.end()));
            }
        });
        ::close(start[0]);
        const std::string go(children.size(), 'g');
        {
            INFO(("Cannot start the children"));
            REQUIRE((::write(start[1], go.data(), go.size()) == static_cast<ssize_t>(go.size())));
        }
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
        {
            INFO(("A concurrent publisher or reader failed"));
            REQUIRE((passed));
        }

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
                    {
                        INFO(("Two writers got the same index"));
                        REQUIRE((!all.count(index)));
                    }
                    ++owned;
                }
                all.insert(index);
            }
        }
        {
            INFO(("A writer did not record every result"));
            REQUIRE((
                owned == size_t(writers) * perWriter && sharedIndices.size() == size_t(perWriter)));
        }
        for(const auto& [j, indices] : sharedIndices)
        {
            INFO(("Writers got different indices for one entry"));
            REQUIRE((indices.size() == 1));
        }
        const auto directory = root / "v1" / key.directoryName();
        const auto master    = stock(directory / "slices" / hj::sliceKey(gemm(256)), ctx.arch);
        std::set<int32_t> mapped;
        for(const auto& [index, prefix] : master->libraryMapping)
            mapped.insert(index);
        {
            INFO(("The mapping does not hold exactly the returned indices"));
            REQUIRE((mapped == all));
        }
        int64_t next = 0;
        ok(hj::msgpack_io::readAllocator(read(root / "v1" / "allocator.dat"), next), "allocator");
        {
            INFO(("Deduplicated entries consumed indices"));
            REQUIRE((next == base + static_cast<int64_t>(all.size()) && *all.rbegin() == next - 1));
        }
        {
            INFO(("Publishers left temporary files"));
            REQUIRE((fs::is_empty(directory / "staging")));
        }
        consistent(directory, ctx.hardware, ctx.arch);
        hj::JitLibrary library(root);
        for(int j = 0; j < perWriter; ++j)
        {
            {
                INFO(("A shared entry has more than one row"));
                REQUIRE((ctx.find(library, key, shared(j))
                    == std::vector<int32_t>{*sharedIndices[j].begin()}));
            }
            for(int w = 0; w < writers; ++w)
            {
                INFO(("A writer's entry is lost"));
                REQUIRE((ctx.find(library, key, own(w, j)).size() == 1));
            }
        }
        std::cout << "PASS " << writers << " processes published " << perWriter
                  << " shared and " << perWriter
                  << " own entries each while a reader looked them up: " << all.size()
                  << " entries, no duplicates, no gaps\n";
    }
}

void runLibrary(const fs::path& scratch, int writers, int perWriter)
{
    Context ctx;
    ctx.scratch = scratch;
    fs::remove_all(ctx.scratch);
    fs::create_directories(ctx.scratch);

    int             device = 0;
    hipDeviceProp_t properties{};
    {
        INFO(("Cannot query the current HIP device"));
        REQUIRE((hipGetDevice(&device) == hipSuccess
            && hipGetDeviceProperties(&properties, device) == hipSuccess));
    }
    const std::string gcnArchName = properties.gcnArchName;
    ctx.arch                      = gcnArchName.substr(0, gcnArchName.find(':'));
    const auto processor          = TensileLite::AMDGPU::toProcessor(ctx.arch);
    {
        INFO(("No TensileLite processor for " + ctx.arch));
        REQUIRE((TensileLite::AMDGPU::toString(processor) == ctx.arch));
    }
    ctx.hardware = TensileLite::AMDGPU(processor, 256, ctx.arch);
    const auto prepared = hipblaslt_jit_test::prepareSolutions(fs::u8path(HIPBLASLT_JIT_DATA));
    const auto plain    = std::find_if(prepared.begin(), prepared.end(), [&](const auto& item) {
        return item.arch == ctx.arch && item.name == "plain";
    });
    {
        INFO(("No plain entry for " + ctx.arch));
        REQUIRE((plain != prepared.end()));
    }
    ctx.entry.bytes = plain->solution.entry;
    const auto library = std::dynamic_pointer_cast<hj::GemmMaster>(
        TensileLite::LoadLibraryData<ContractionProblemGemm>(ctx.entry.bytes));
    {
        INFO(("The replay has no solution 0"));
        REQUIRE((library && library->solutions.count(0)));
    }
    const auto& replayed = *library->solutions.at(0);
    ctx.entry.kernel     = replayed.kernelName;
    if(!(*replayed.problemPredicate)(gemm(256)))
    {
        replayed.problemPredicate->debugEval(gemm(256), std::cerr);
        throw std::runtime_error("The test problem does not match the replayed solution");
    }
    if(!(*replayed.hardwarePredicate)(ctx.hardware))
    {
        replayed.hardwarePredicate->debugEval(ctx.hardware, std::cerr);
        throw std::runtime_error("The " + ctx.arch + " plain solution rejects this device");
    }
    std::cout << "PASS " << ctx.arch << " plain entry matches the test problem\n";
    if(writers)
        concurrency(ctx, writers, perWriter);
    else
    {
        keys(ctx);
        directories(ctx);
        roundTrip(ctx);
        dedupeAndOrder(ctx);
        mismatch(ctx);
        allocator(ctx);
        interrupted(ctx);
        refresh(ctx);
        fusedA2A(ctx);
    }
    std::cout << "ALL JIT LIBRARY CHECKS PASSED\n";
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
