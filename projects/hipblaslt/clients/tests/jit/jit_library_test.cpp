// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Exercises the JIT solution library: cache keys, directory
// checks, the stock TensileLite loader reading what the library writes,
// exact-size lookup, index allocation, a crash after every publication step,
// concurrent publishers in separate processes, and the rejection of fused GEMM
// and all-to-all.

#include "test_helpers.hpp"
#include "hipblaslt-jit-fs.hpp"
#include "hipblaslt-jit-library.hpp"
#include "hipblaslt-jit-msgpack.hpp"
#include "hipblaslt-jit-source-bundle.hpp"
#include <Tensile/AMDGPU.hpp>
#include <Tensile/Tensile.hpp>
#include <msgpack.hpp>

#include <algorithm>
#include <chrono>
#include <climits>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <sstream>
#include <thread>

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

namespace hj        = hipblaslt_jit;
namespace fs        = std::filesystem;
namespace artifacts = hipblaslt_jit::source_bundle;
using TensileLite::ContractionProblemGemm;

namespace
{
    constexpr int32_t base = hj::jitIndexBase;

    using hipblaslt_jit_test::require;

    void ok(const hj::Status& status, const std::string& what)
    {
        require(status.ok(), what + ": " + status.message);
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
            require(map.type == msgpack::type::MAP, "Expected a map");
            for(uint32_t i = 0; i < map.via.map.size; ++i)
            {
                auto& member = map.via.map.ptr[i];
                if(member.key.type == msgpack::type::STR && member.key.as<std::string>() == key)
                    return &member.val;
            }
            throw std::runtime_error("Missing " + key);
        };
        const auto* solutions = field(handle.get(), "solutions");
        require(solutions->type == msgpack::type::ARRAY && solutions->via.array.size == 1,
                "Expected one solution");
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
        require(master && master->initLibraryMapping(path), "The stock loader cannot read " + path);
        return master;
    }

    // Every master row is mapped, and every mapped index loads with its code object.
    void consistent(const fs::path&              directory,
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
            require(hj::isJitIndex(index), "Mapped index outside the JIT range");
            require(mapped.insert(prefix).second, "Two indices map " + prefix);
            require(fs::is_regular_file(directory / (prefix + ".co")), "Missing " + prefix + ".co");
            const auto solution = master->getSolutionByIndex(hardware, index);
            require(solution && solution->index == index
                        && solution->codeObjectFilename.load() == prefix + ".co",
                    "Mapped index " + std::to_string(index) + " does not load");
        }
        require(std::set<std::string>(rows.begin(), rows.end()).size() == rows.size(),
                "Duplicate master rows");
        for(const auto& row : rows)
            require(mapped.count(row), "Master row " + row + " is not mapped");
    }

    // Names, modes, sizes and modification times of everything under root.
    std::string tree(const fs::path& root)
    {
        std::vector<std::string> lines;
        for(const auto& entry : fs::recursive_directory_iterator(root))
        {
            struct stat info{};
            require(::lstat(entry.path().c_str(), &info) == 0, "lstat failed");
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
        require(::lstat(path.c_str(), &info) == 0, "lstat " + path.string());
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
        require(indices.size() == entries.size(), "Publish returned " + text(indices));
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
        require(key.canonicalJson()
                    == std::string(R"({"backend":{"id":"test","version":"1"},"code_object_version":4,)")
                           + R"("comgr":"3.0:/opt/rocm/lib/libamd_comgr.so:1:2",)"
                           + R"("compiler_environment":{"LLVM_PATH":"/opt/rocm/llvm"},)"
                           + R"("rocm_path":"/opt/rocm","schema":1,"target":{"isa":")" + ctx.arch
                           + R"(","library_arch":")" + ctx.arch + R"(","target_id":")" + ctx.arch
                           + R"(:sramecc+:xnack-","wavefront_size":64}})",
                "Unexpected cache key " + key.canonicalJson());
        const auto name = key.directoryName();
        require(name.size() == ctx.arch.size() + 17 && name.rfind(ctx.arch + "-", 0) == 0,
                "Unexpected directory " + name);
        // An ISA different from the device, so this field still changes the key.
        const auto otherIsa = ctx.arch == "gfx942" ? "gfx950" : "gfx942";
        const std::vector<std::function<void(hj::CacheKey&)>> changes{
            [&](auto& k) { k.targetId = ctx.arch + ":sramecc-:xnack-"; },
            [&](auto& k) { k.isa = otherIsa; },
            [](auto& k) { k.libraryArch += "v1"; },
            [](auto& k) { k.wavefrontSize = 32; },
            [](auto& k) { k.backendId = "other"; },
            [](auto& k) { k.backendVersion = "2"; },
            [](auto& k) { k.codeObjectVersion = 5; },
            [](auto& k) { k.comgr = "3.1"; },
            [](auto& k) { k.rocmPath = "/opt/rocm-other"; },
            [](auto& k) { k.environment["AMD_COMGR_DRIVER_OPTIONS_APPEND"] = "-O0"; },
        };
        std::set<std::string> names{name};
        for(const auto& change : changes)
        {
            auto changed = key;
            change(changed);
            names.insert(changed.directoryName());
        }
        require(names.size() == changes.size() + 1, "A cache key field does not change the key");

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
        require(hj::compilerEnvironment(environment) == expected,
                "compilerEnvironment kept the wrong variables");

        const auto problem = gemm(256);
        require(hj::problemSizes(problem) == std::vector<size_t>{256, 128, 1, 512},
                "Unexpected problem sizes " + text(hj::problemSizes(problem)));
        const auto type = hj::problemTypeKey(problem);
        require(type == hj::problemTypeKey(gemm(512, 64, 128, 3)),
                "Sizes changed the ProblemType key");
        auto accumulate = gemm(256);
        accumulate.setHighPrecisionAccumulate(false);
        require(type != hj::problemTypeKey(accumulate), "HPA did not change the ProblemType key");

        const std::vector<size_t> sizes{256, 128, 1, 512};
        const auto                prefix = hj::entryPrefix(type, "kernel", sizes);
        require(prefix.size() == 52 && prefix.rfind("TensileLibrary_JIT_", 0) == 0,
                "Unexpected entry prefix " + prefix);
        require(prefix == hj::entryPrefix(type, "kernel", sizes)
                    && prefix != hj::entryPrefix(type + " ", "kernel", sizes)
                    && prefix != hj::entryPrefix(type, "kernel2", sizes)
                    && prefix != hj::entryPrefix(type, "kernel", {256, 128, 1, 1024})
                    && hj::entryPrefix(type, "kernel", sizes, 2) == prefix + "_2",
                "Entry prefixes are not a function of their content");
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
            require(::chmod(root.c_str(), rejected) == 0, "chmod failed");
            const auto status = attempt(root);
            require(!status.ok() && status.stage == hj::Stage::Lookup
                        && status.message.find("JIT solution library disabled") == 0,
                    "A writable root was accepted: " + status.message);
            require(fs::is_empty(root), "A rejected root was written to");
        }
        for(unsigned accepted : {0700u, 0750u, 0755u})
        {
            const auto root = base / ("mode-" + octal(accepted));
            fs::create_directory(root);
            require(::chmod(root.c_str(), accepted) == 0, "chmod failed");
            ok(attempt(root), "private root");
            require(mode(root / "v1") == 0700
                        && mode(root / "v1" / testKey(ctx).directoryName()) == 0700,
                    "Library directories are not private");
        }
        const auto missing = base / "missing" / "nested";
        ok(attempt(missing), "missing root");
        require(mode(missing) == 0700, "A created root is not private");

        fs::create_directory_symlink(base / "mode-700", base / "link");
        require(!attempt(base / "link").ok(), "A symbolic link root was accepted");
        std::ofstream(base / "file") << "not a directory";
        require(!attempt(base / "file").ok(), "A file root was accepted");

        const auto shared = base / "mode-700" / "v1" / testKey(ctx).directoryName();
        require(::chmod(shared.c_str(), 0770) == 0, "chmod failed");
        require(!attempt(base / "mode-700").ok(), "A group-writable key directory was accepted");
        require(::chmod(shared.c_str(), 0700) == 0
                    && ::chmod((base / "mode-700" / "v1").c_str(), 0707) == 0,
                "chmod failed");
        require(!attempt(base / "mode-700").ok(), "An other-writable schema directory was accepted");
        std::cout << "PASS group- and other-writable, linked and non-directory roots are rejected; "
                     "created directories are 0700\n";
    }

    void roundTrip(const Context& ctx)
    {
        const auto     root = ctx.fresh("round-trip");
        hj::JitLibrary library(root);
        const auto     key     = testKey(ctx);
        const auto     problem = gemm(256);
        require(ctx.find(library, key, problem).empty(), "An empty library returned solutions");
        const auto indices = publish(library, key, problem, {ctx.entry});
        require(indices == std::vector<int32_t>{base}, "First index " + text(indices));

        const auto directory = library.directory(key);
        const auto prefix    = hj::entryPrefix(
            hj::problemTypeKey(problem), ctx.entry.kernel, hj::problemSizes(problem));
        std::set<std::string> names;
        for(const auto& entry : fs::directory_iterator(directory))
            names.insert(entry.path().filename().string());
        require(names
                    == std::set<std::string>{lazyMaster(ctx.arch),
                                             lazyMapping(ctx.arch),
                                             "cache-key.json",
                                             "staging",
                                             prefix + ".co",
                                             prefix + ".dat"},
                "Unexpected key directory contents");
        names.clear();
        for(const auto& entry : fs::directory_iterator(root / "v1"))
            names.insert(entry.path().filename().string());
        require(names == std::set<std::string>{"allocator.dat", "lock", key.directoryName()},
                "Unexpected schema directory contents");
        require(fs::is_empty(directory / "staging"), "Publishing left temporary files");
        require(readText(directory / "cache-key.json") == key.canonicalJson(),
                "cache-key.json does not hold the key");
        require(read(directory / (prefix + ".co")) == built(ctx.entry).first.object.bytes,
                "The code object was not stored unchanged");
        int64_t next = 0;
        ok(hj::msgpack_io::readAllocator(read(root / "v1" / "allocator.dat"), next), "allocator");
        require(next == base + 1, "The allocator did not advance");

        const auto master = stock(directory, ctx.arch);
        require(master->libraryMapping == std::map<int, std::string>{{base, prefix}},
                "Unexpected index mapping");
        const auto best = master->findBestSolution(problem, ctx.hardware);
        require(best && best->index == base && best->kernelName == ctx.entry.kernel
                    && best->codeObjectFilename.load() == prefix + ".co",
                "The stock loader did not find the published solution");
        const auto byIndex = master->getSolutionByIndex(ctx.hardware, base);
        require(byIndex == best, "The stock loader did not resolve the published index");

        // The solution itself accepts these problems; only the row's sizes reject them.
        for(const auto& other : {gemm(264), gemm(256, 136), gemm(256, 128, 1024), gemm(256, 128, 512, 2)})
        {
            require((*best->problemPredicate)(other), "The solution rejects a nearby size");
            require(!master->findBestSolution(other, ctx.hardware)
                        && ctx.find(library, key, other).empty(),
                    "An entry matched a size it was not published for");
        }
        // Each hit still runs the solution's own predicates.
        auto accumulate = gemm(256);
        accumulate.setHighPrecisionAccumulate(false);
        require(ctx.find(library, key, accumulate).empty(), "A hit skipped the solution predicates");
        require(ctx.find(library, key, problem) == indices, "Lookup missed the published entry");

        hj::Status why;
        const auto view = library.resolve(0, base, why);
        require(view.master && view.adapter, "resolve: " + why.message);
        const auto solution = library.solutionByIndex(0, ctx.hardware, base, why);
        require(solution && solution->codeObjectFilename.load() == prefix + ".co",
                "solutionByIndex: " + why.message);
        require(!library.resolve(1, base, why).master && !why.ok(),
                "An index resolved on a device that never used the library");
        require(!library.resolve(0, 5, why).master
                    && why.message.find("outside the reserved JIT range") != std::string::npos,
                "A prebuilt index resolved in the JIT library");
        require(!library.resolve(0, base + 1, why).master && !why.ok(),
                "An unpublished index resolved");
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
        require(first == std::vector<int32_t>{base, base + 1, base}, "Batch indices " + text(first));
        const auto before = tree(root);
        require(publish(library, key, problem, {b, a}) == std::vector<int32_t>{base + 1, base},
                "Republishing changed indices");
        require(tree(root) == before, "Republishing published entries wrote files");
        require(publish(library, key, problem, {c}) == std::vector<int32_t>{base + 2},
                "A new entry did not take the next index");
        require(ctx.find(library, key, problem) == std::vector<int32_t>{base, base + 1, base + 2},
                "Lookup is not in publication order");
        require(ctx.find(library, key, problem, 2) == std::vector<int32_t>{base, base + 1},
                "Lookup ignored the count");
        require(ctx.find(library, key, problem, 4, {a.kernel})
                    == std::vector<int32_t>{base + 1, base + 2},
                "Lookup returned an excluded kernel");
        require(ctx.find(library, key, problem, 1, {a.kernel, b.kernel})
                    == std::vector<int32_t>{base + 2},
                "Exclusions reduced the count");
        require(publish(library, key, gemm(512), {a}) == std::vector<int32_t>{base + 3}
                    && ctx.find(library, key, gemm(512)) == std::vector<int32_t>{base + 3}
                    && ctx.find(library, key, problem).size() == 3,
                "Another size did not get its own entry");

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
            require(seen[i] == base + 4, "Concurrent thread " + std::to_string(i) + ": " + errors[i]);
        consistent(library.directory(key), ctx.hardware, ctx.arch);

        // A published name whose entry holds another kernel is a hash collision.
        const auto directory = library.directory(key);
        const auto prefix
            = hj::entryPrefix(hj::problemTypeKey(gemm(2048)), a.kernel, hj::problemSizes(gemm(2048)));
        const auto collided = publish(library, key, gemm(2048), {a})[0];
        std::vector<uint8_t> other;
        ok(hj::msgpack_io::rewriteEntryIndex(b.bytes, 0, collided, other), "rewrite");
        ok(hj::files::writeAtomically(directory / "staging", directory / (prefix + ".dat"), other),
           "replace entry");
        hj::JitLibrary reopened(root);
        const auto     renamedIndex = publish(reopened, key, gemm(2048), {a})[0];
        require(renamedIndex == base + 6 && fs::exists(directory / (prefix + "_1.dat")),
                "A hash collision reused another kernel's entry");
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
            require(ctx.find(library, testKey(ctx, "2"), gemm(256)).empty(),
                    "Another backend version reused an entry");
            require(publish(library, testKey(ctx, "2"), gemm(256), {ctx.entry})
                        == std::vector<int32_t>{base + 1},
                    "Indices are not unique across keys");
            const std::vector<std::function<void(hj::CacheKey&)>> changes{
                [](auto& k) { k.comgr = "3.1"; },
                [](auto& k) { k.rocmPath = "/opt/rocm-other"; },
                [](auto& k) { k.environment["AMD_COMGR_DRIVER_OPTIONS_APPEND"] = "-O0"; },
                [&](auto& k) { k.targetId = ctx.arch + ":sramecc-:xnack-"; },
                [](auto& k) { k.codeObjectVersion = 5; },
            };
            for(const auto& change : changes)
            {
                auto changed = testKey(ctx, "1");
                change(changed);
                require(ctx.find(library, changed, gemm(256)).empty(),
                        "A mismatched key reused an entry");
            }
        }
        require(tree(directory) == snapshot, "Another key changed this key's directory");

        std::ofstream(directory / "cache-key.json", std::ios::trunc) << "{}";
        fs::create_directories(root / "v2" / (ctx.arch + "-0000000000000000"));
        std::ofstream(root / "v2" / "allocator.dat") << "not a library";
        const auto tampered = tree(root);
        {
            hj::JitLibrary       library(root);
            std::vector<int32_t> indices;
            auto status = library.lookup(testKey(ctx, "1"), 0, gemm(256), ctx.hardware, 1, {}, indices);
            require(!status.ok() && indices.empty() && status.stage == hj::Stage::Lookup
                        && status.message.find("cache key") != std::string::npos,
                    "A tampered cache key was accepted: " + status.message);
            status = library.publish(testKey(ctx, "1"), 0, gemm(256), {built(ctx.entry)}, indices);
            require(!status.ok() && status.stage == hj::Stage::Publish,
                    "Publishing into a tampered directory was accepted");
            require(ctx.find(library, testKey(ctx, "2"), gemm(256)) == std::vector<int32_t>{base + 1},
                    "A tampered directory affected another key");
        }
        require(tree(root) == tampered, "A mismatched directory was modified or deleted");
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
        require(publish(library, key, gemm(264), {ctx.entry}) == std::vector<int32_t>{INT32_MAX},
                "The last reserved index was not allocated");
        std::vector<int32_t> indices;
        const auto status = library.publish(key, 0, gemm(272), {built(ctx.entry)}, indices);
        require(!status.ok() && status.message.find("exhausted") != std::string::npos,
                "An exhausted range allocated an index: " + status.message);
        require(publish(library, key, gemm(264), {ctx.entry}) == std::vector<int32_t>{INT32_MAX}
                    && ctx.find(library, key, gemm(264)) == std::vector<int32_t>{INT32_MAX},
                "Published entries stopped resolving when the range was exhausted");

        ok(hj::msgpack_io::writeAllocator(base - 1, bytes), "allocator");
        ok(hj::files::writeAtomically(directory / "staging", root / "v1" / "allocator.dat", bytes),
           "allocator");
        require(!library.publish(key, 0, gemm(272), {built(ctx.entry)}, indices).ok(),
                "An allocator below the reserved range was used");

        const auto prefix = hj::entryPrefix(
            hj::problemTypeKey(gemm(256)), ctx.entry.kernel, hj::problemSizes(gemm(256)));
        ok(hj::msgpack_io::rewriteEntryIndex(ctx.entry.bytes, 0, 7, bytes), "rewrite");
        ok(hj::files::writeAtomically(directory / "staging", directory / (prefix + ".dat"), bytes),
           "tamper");
        hj::JitLibrary reopened(root);
        require(ctx.find(reopened, key, gemm(256)).empty(),
                "Lookup returned an index outside the reserved range");
        std::cout << "PASS index allocation up to INT32_MAX, exhaustion, and out-of-range entries "
                     "dropped\n";
    }

    hj::PublishStep crashStep;

    void crashes(const Context& ctx)
    {
        using Step = hj::PublishStep;
        const std::pair<Step, const char*> steps[]
            = {{Step::Staged, "staged"},
               {Step::Locked, "locked"},
               {Step::Allocated, "allocated"},
               {Step::CodeObjects, "code objects"},
               {Step::Entries, "entries"},
               {Step::Mapping, "mapping"},
               {Step::Master, "master"},
               {Step::Unlocked, "unlocked"}};
        const auto key = testKey(ctx);
        for(const auto& [step, name] : steps)
        {
            const auto root = ctx.fresh(std::string("crash-") + name);
            {
                hj::JitLibrary library(root);
                publish(library, key, gemm(256), {ctx.entry});
            }
            std::cout.flush();
            const auto child = ::fork();
            require(child >= 0, "fork failed");
            if(child == 0)
            {
                crashStep          = step;
                hj::publishStepHook = [](Step reached) {
                    if(reached == crashStep)
                        ::_exit(3);
                };
                hj::JitLibrary       library(root);
                std::vector<int32_t> indices;
                static_cast<void>(library.publish(key, 0, gemm(512), {built(ctx.entry)}, indices));
                ::_exit(4);
            }
            int status = 0;
            require(::waitpid(child, &status, 0) == child && WIFEXITED(status)
                        && WEXITSTATUS(status) == 3,
                    std::string("The publisher did not stop after ") + name);

            const auto directory = hj::JitLibrary(root).directory(key);
            consistent(directory, ctx.hardware, ctx.arch);
            hj::JitLibrary library(root);
            require(ctx.find(library, key, gemm(256)) == std::vector<int32_t>{base},
                    std::string("A crash after ") + name + " lost a published entry");
            const bool visible = step >= Step::Master;
            require(ctx.find(library, key, gemm(512)).size() == (visible ? 1u : 0u),
                    std::string("A crash after ") + name + " left the wrong lookup result");
            const int32_t expected = step < Step::Allocated ? base + 1
                                     : step < Step::Mapping ? base + 2
                                                            : base + 1;
            const auto again = publish(library, key, gemm(512), {ctx.entry});
            require(again == std::vector<int32_t>{expected},
                    std::string("Republishing after ") + name + " returned " + text(again));
            require(ctx.find(library, key, gemm(512)) == again,
                    std::string("Republishing after ") + name + " is not found");
            require(publish(library, key, gemm(1024), {ctx.entry})[0] == expected + 1,
                    std::string("The lock was not released after ") + name);
            consistent(directory, ctx.hardware, ctx.arch);
        }
        std::cout << "PASS a publisher killed after every step leaves a consistent library that "
                     "the next publisher completes\n";
    }

    void refresh(const Context& ctx)
    {
        const auto     root = ctx.fresh("refresh");
        const auto     key  = testKey(ctx);
        hj::JitLibrary reader(root), writer(root);
        require(ctx.find(reader, key, gemm(256)).empty(), "An empty library returned solutions");
        const auto first = publish(writer, key, gemm(256), {ctx.entry});
        require(ctx.find(reader, key, gemm(256)) == first,
                "A reader did not see another instance's entry");
        hj::Status why;
        const auto old = reader.resolve(0, first[0], why);
        require(old.master != nullptr, "resolve: " + why.message);
        const auto second = publish(writer, key, gemm(512), {ctx.entry});
        const auto view   = reader.resolve(0, second[0], why);
        require(view.master && view.master != old.master && view.adapter == old.adapter,
                "resolve did not reload for a newer index: " + why.message);
        require(old.master->getSolutionByIndex(ctx.hardware, first[0]) != nullptr,
                "A superseded snapshot lost its solution");

        const auto deviceKey = [&ctx](std::string comgr) {
            return [&ctx, comgr](int, hj::CacheKey& k) {
                k       = testKey(ctx, "");
                k.comgr = comgr.empty() ? k.comgr : comgr;
                return hj::Status{};
            };
        };
        hj::JitLibrary discovering(root, deviceKey(""));
        require(discovering.solutionByIndex(0, ctx.hardware, second[0], why) != nullptr,
                "An index was not found without a lookup: " + why.message);
        hj::JitLibrary other(root, deviceKey("3.1"));
        require(!other.resolve(0, second[0], why).master && !why.ok(),
                "An index resolved from a directory for another toolchain");
        std::cout << "PASS readers reload on change, keep superseded snapshots, and find indices "
                     "in any backend's directory for their toolchain\n";
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
            require(rejected(status, hj::Stage::Lookup) && indices.empty(),
                    "A fused all-to-all lookup was not rejected: " + status.message);
            status = library.publish(key, 0, fused, {built(ctx.entry)}, indices);
            require(rejected(status, hj::Stage::Publish) && indices.empty(),
                    "A fused all-to-all publication was not rejected: " + status.message);
        }
        require(!fs::exists(root), "Rejecting fused all-to-all touched the library");
        std::string reason;
        try
        {
            hj::problemTypeKey(fused);
        }
        catch(const std::runtime_error& e)
        {
            reason = e.what();
        }
        require(reason.find("fused GEMM and all-to-all") != std::string::npos,
                "A fused all-to-all problem has a JIT ProblemType: " + reason);

        hj::JitLibrary library(root);
        const auto     plain  = publish(library, key, gemm(256), {ctx.entry});
        const auto     status = library.lookup(key, 0, fused, ctx.hardware, 1, {}, indices);
        require(rejected(status, hj::Stage::Lookup) && indices.empty(),
                "A plain solution of the same sizes served fused all-to-all");
        require(ctx.find(library, key, gemm(256)) == plain, "The plain problem lost its solution");
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
        require(::pipe(start) == 0, "pipe failed");
        std::vector<pid_t> children;
        const auto         spawn = [&](const std::function<void()>& body) {
            std::cout.flush();
            const auto child = ::fork();
            require(child >= 0, "fork failed");
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
                require(std::chrono::steady_clock::now() < deadline, "The writers did not finish");
                for(int j = 0; j < perWriter; ++j)
                {
                    const auto found = ctx.find(library, key, shared(j), 1);
                    require(found.size() <= 1 && (seen[j] < 0 || found == std::vector{seen[j]}),
                            "A reader saw an entry disappear or change");
                    if(found.empty())
                        continue;
                    seen[j] = found[0];
                    hj::Status why;
                    require(library.solutionByIndex(0, ctx.hardware, found[0], why) != nullptr,
                            "A reader could not load a found entry: " + why.message);
                }
                if(fs::exists(root / "v1" / key.directoryName() / lazyMaster(ctx.arch)))
                    consistent(root / "v1" / key.directoryName(), ctx.hardware, ctx.arch);
            }
            require(std::find(seen.begin(), seen.end(), -1) == seen.end(),
                    "The reader never saw every shared entry");
        });
        ::close(start[0]);
        const std::string go(children.size(), 'g');
        require(::write(start[1], go.data(), go.size()) == static_cast<ssize_t>(go.size()),
                "Cannot start the children");
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
        require(passed, "A concurrent publisher or reader failed");

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
                    require(!all.count(index), "Two writers got the same index");
                    ++owned;
                }
                all.insert(index);
            }
        }
        require(owned == size_t(writers) * perWriter && sharedIndices.size() == size_t(perWriter),
                "A writer did not record every result");
        for(const auto& [j, indices] : sharedIndices)
            require(indices.size() == 1, "Writers got different indices for one entry");
        const auto directory = root / "v1" / key.directoryName();
        const auto master    = stock(directory, ctx.arch);
        std::set<int32_t> mapped;
        for(const auto& [index, prefix] : master->libraryMapping)
            mapped.insert(index);
        require(mapped == all, "The mapping does not hold exactly the returned indices");
        int64_t next = 0;
        ok(hj::msgpack_io::readAllocator(read(root / "v1" / "allocator.dat"), next), "allocator");
        require(next == base + static_cast<int64_t>(all.size()) && *all.rbegin() == next - 1,
                "Deduplicated entries consumed indices");
        require(fs::is_empty(directory / "staging"), "Publishers left temporary files");
        consistent(directory, ctx.hardware, ctx.arch);
        hj::JitLibrary library(root);
        for(int j = 0; j < perWriter; ++j)
        {
            require(ctx.find(library, key, shared(j))
                        == std::vector<int32_t>{*sharedIndices[j].begin()},
                    "A shared entry has more than one row");
            for(int w = 0; w < writers; ++w)
                require(ctx.find(library, key, own(w, j)).size() == 1, "A writer's entry is lost");
        }
        std::cout << "PASS " << writers << " processes published " << perWriter
                  << " shared and " << perWriter
                  << " own entries each while a reader looked them up: " << all.size()
                  << " entries, no duplicates, no gaps\n";
    }
}

int main(int argc, char** argv)
{
    int writers = 0, perWriter = 0;
    if(argc == 7 && std::string(argv[3]) == "--writers" && std::string(argv[5]) == "--per-writer")
    {
        writers   = std::atoi(argv[4]);
        perWriter = std::atoi(argv[6]);
    }
    if((argc != 3 && argc != 7) || (argc == 7 && (writers < 1 || perWriter < 1)))
    {
        std::cerr << "Usage: " << argv[0]
                  << " BUNDLES SCRATCH [--writers N --per-writer M]\n";
        return 2;
    }
    try
    {
        Context ctx;
        ctx.scratch = fs::absolute(argv[2]);
        fs::remove_all(ctx.scratch);
        fs::create_directories(ctx.scratch);

        int             device = 0;
        hipDeviceProp_t properties{};
        require(hipGetDevice(&device) == hipSuccess
                    && hipGetDeviceProperties(&properties, device) == hipSuccess,
                "Cannot query the current HIP device");
        const std::string gcnArchName = properties.gcnArchName;
        ctx.arch                      = gcnArchName.substr(0, gcnArchName.find(':'));
        const auto processor          = TensileLite::AMDGPU::toProcessor(ctx.arch);
        require(TensileLite::AMDGPU::toString(processor) == ctx.arch,
                "No TensileLite processor for " + ctx.arch);
        ctx.hardware = TensileLite::AMDGPU(processor, 256, ctx.arch);
        const auto plain
            = hipblaslt_jit_test::deviceBundles(fs::u8path(argv[1]), gcnArchName) / "plain";
        require(fs::is_directory(plain), "No plain bundle for " + ctx.arch);
        ctx.entry.bytes = artifacts::readSourceBundle(plain).library;
        const auto library = std::dynamic_pointer_cast<hj::GemmMaster>(
            TensileLite::LoadLibraryData<ContractionProblemGemm>(ctx.entry.bytes));
        require(library && library->solutions.count(0), "The replay has no solution 0");
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
        std::cout << "PASS " << ctx.arch << " plain bundle matches the test problem\n";
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
            crashes(ctx);
            refresh(ctx);
            fusedA2A(ctx);
        }
        std::cout << "ALL JIT LIBRARY CHECKS PASSED\n";
    }
    catch(const std::exception& error)
    {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
