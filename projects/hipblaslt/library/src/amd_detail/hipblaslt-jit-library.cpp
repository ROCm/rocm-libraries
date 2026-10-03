// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-library.hpp"
#include "hipblaslt-jit-code-object.hpp"
#include "hipblaslt-jit-fs.hpp"
#include "hipblaslt-jit-hash.hpp"
#include "hipblaslt-jit-json.hpp"
#include "hipblaslt-jit-msgpack.hpp"
#include "hipblaslt-jit-problem-type.hpp"
#include "rocblaslt_secure_env.hpp"
#include <Tensile/CachingLibrary.hpp>
#include <Tensile/ExactLogicLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#include <algorithm>
#include <charconv>
#include <cstring>
#include <new>
#include <stdexcept>
#include <string_view>
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
extern char** environ;
#endif

namespace hipblaslt_jit
{
    namespace
    {
        namespace fs = std::filesystem;

        constexpr size_t fileLimit = size_t{1} << 30;

        using GemmRows = TensileLite::ProblemSelectionLibrary<TensileLite::ContractionProblemGemm,
                                                              TensileLite::ContractionSolution>;

        const GemmRows* rowsOf(const GemmMaster& master)
        {
            auto library = master.library;
            if(auto cache = std::dynamic_pointer_cast<
                   TensileLite::CachingLibrary<TensileLite::ContractionProblemGemm,
                                               TensileLite::ContractionSolution>>(library))
                library = cache->library();
            return dynamic_cast<const GemmRows*>(library.get());
        }

        Status failed(Stage stage, std::string message)
        {
            return {Status::Code::Failed, stage, std::move(message)};
        }

        template <class F>
        Status staged(Stage stage, F&& f)
        {
            Status status;
            try
            {
                status = f();
            }
            catch(const std::bad_alloc&)
            {
                status = failed(stage, "Out of memory");
            }
            catch(const std::exception& e)
            {
                status = failed(stage, e.what());
            }
            catch(...)
            {
                status = failed(stage, "Unknown exception");
            }
            if(!status.ok())
                status.stage = stage;
            return status;
        }

        void reached([[maybe_unused]] PublishStep step)
        {
#ifdef HIPBLASLT_JIT_LIBRARY_TESTING
            if(publishStepHook)
                publishStepHook(step);
#endif
        }

        bool startsWith(std::string_view text, std::string_view prefix)
        {
            return text.substr(0, prefix.size()) == prefix;
        }

        // Reads a string that json::quote wrote at text[at] and moves past it.
        bool unquote(std::string_view text, size_t& at, std::string& value)
        {
            if(at >= text.size() || text[at] != '"')
                return false;
            value.clear();
            for(++at; at < text.size(); ++at)
            {
                if(text[at] == '"')
                {
                    ++at;
                    return true;
                }
                if(text[at] != '\\')
                    value += text[at];
                else if(++at < text.size() && text[at] == 'u')
                {
                    unsigned code = 0;
                    const auto digits = text.substr(at + 1, 4);
                    if(digits.size() != 4
                       || std::from_chars(digits.data(), digits.data() + 4, code, 16).ptr
                              != digits.data() + 4)
                        return false;
                    value += static_cast<char>(code);
                    at += 4;
                }
                else if(at < text.size())
                    value += text[at];
            }
            return false;
        }

        // The backend fields of a cache-key.json; the rest must match another key.
        bool backendOf(std::string_view json, std::string& id, std::string& version)
        {
            constexpr std::string_view head = R"({"backend":{"id":)", middle = R"(,"version":)";
            size_t                     at   = head.size();
            if(!startsWith(json, head) || !unquote(json, at, id)
               || json.substr(at, middle.size()) != middle)
                return false;
            at += middle.size();
            return unquote(json, at, version);
        }

        template <class T>
        void addOnce(std::vector<T*>& items, T* item)
        {
            if(std::find(items.begin(), items.end(), item) == items.end())
                items.push_back(item);
        }

        bool validEntry(const std::vector<uint8_t>& entry, int32_t index, const std::string& kernel)
        {
            const auto library = std::dynamic_pointer_cast<GemmMaster>(
                TensileLite::LoadLibraryData<TensileLite::ContractionProblemGemm>(entry));
            return library && library->solutions.size() == 1 && library->solutions.count(index)
                   && library->solutions.at(index)
                   && library->solutions.at(index)->kernelName == kernel;
        }
    }

#ifdef HIPBLASLT_JIT_LIBRARY_TESTING
    void (*publishStepHook)(PublishStep) = nullptr;
#endif

    std::map<std::string, std::string> compilerEnvironment(const char* const* environment)
    {
        static constexpr std::string_view ignored[]
            = {"AMD_COMGR_CACHE",
               "AMD_COMGR_CACHE_DIR",
               "AMD_COMGR_CACHE_POLICY",
               "AMD_COMGR_SAVE_TEMPS",
               "AMD_COMGR_SAVE_LLVM_TEMPS",
               "AMD_COMGR_REDIRECT_LOGS",
               "AMD_COMGR_EMIT_VERBOSE_LOGS",
               "AMD_COMGR_LOG_LEVEL",
               "AMD_COMGR_TIME_STATISTICS",
               "AMD_COMGR_TIME_STATISTICS_GRANULARITY",
               "AMD_COMGR_USE_VFS"};
        std::map<std::string, std::string> result;
        for(auto variable = environment; variable && *variable; ++variable)
        {
            const std::string_view entry(*variable);
            const auto             equals = entry.find('=');
            if(equals == std::string_view::npos)
                continue;
            const auto name = entry.substr(0, equals);
            if(name == "HIP_PATH" || name == "LLVM_PATH"
               || (startsWith(name, "AMD_COMGR_")
                   && std::find(std::begin(ignored), std::end(ignored), name) == std::end(ignored)))
                result.emplace(name, entry.substr(equals + 1));
        }
        return result;
    }

    CacheKey CacheKey::make(const DeviceTarget& target, const BackendInfo& backend, int codeObjectVersion)
    {
        struct Toolchain
        {
            std::string                        comgr, rocmPath;
            std::map<std::string, std::string> environment;
        };
        static const Toolchain toolchain{code_object::comgrIdentity(),
                                         code_object::rocmPath(),
#ifdef _WIN32
                                         compilerEnvironment(_environ)};
#else
                                         compilerEnvironment(environ)};
#endif
        CacheKey key;
        key.targetId          = target.targetId;
        key.isa               = target.isa;
        key.libraryArch       = target.libraryArch;
        key.wavefrontSize     = target.wavefrontSize;
        key.backendId         = backend.id;
        key.backendVersion    = backend.version;
        key.codeObjectVersion = codeObjectVersion;
        key.comgr             = toolchain.comgr;
        key.rocmPath          = toolchain.rocmPath;
        key.environment       = toolchain.environment;
        return key;
    }

    std::string CacheKey::canonicalJson() const
    {
        using json::literal;
        using json::object;
        using json::quote;
        json::Members environmentMembers;
        for(const auto& [name, value] : environment)
            environmentMembers.emplace_back(name, quote(value));
        return object({
            {"backend", object({{"id", quote(backendId)}, {"version", quote(backendVersion)}})},
            {"code_object_version", literal(codeObjectVersion)},
            {"comgr", quote(comgr)},
            {"compiler_environment", object(environmentMembers)},
            {"rocm_path", quote(rocmPath)},
            {"schema", literal(jitLibrarySchema)},
            {"target",
             object({{"isa", quote(isa)},
                     {"library_arch", quote(libraryArch)},
                     {"target_id", quote(targetId)},
                     {"wavefront_size", literal(wavefrontSize)}})},
        });
    }

    std::string CacheKey::directoryName() const
    {
        return isa + "-" + Fnv1a().add(canonicalJson()).hex();
    }

    std::string problemTypeKey(const TensileLite::ContractionProblemGemm& problem)
    {
        auto fields = problemTypeFields(problem);
        fields.emplace_back("OperationIdentifier", json::quote(problem.operationIdentifier()));
        fields.emplace_back("GroupedGemm", json::literal(problem.groupedGemm()));
        return json::object(fields);
    }

    std::vector<size_t> problemSizes(const TensileLite::ContractionProblemGemm& problem)
    {
        std::vector<size_t> sizes;
        const auto          count = problem.c().dimensions() + problem.boundIndices().size();
        for(size_t index = 0; index < count; ++index)
            sizes.push_back(problem.size(index));
        return sizes;
    }

    std::string entryPrefix(const std::string&         problemTypeKey,
                            const std::string&         kernelName,
                            const std::vector<size_t>& sizes,
                            unsigned                   collision)
    {
        Fnv1a content;
        content.add(kernelName);
        for(auto size : sizes)
            content.add(std::to_string(size));
        auto prefix = "TensileLibrary_JIT_" + Fnv1a().add(problemTypeKey).hex() + "_" + content.hex();
        return collision ? prefix + "_" + std::to_string(collision) : prefix;
    }

    struct JitLibrary::Directory
    {
        fs::path    path;
        std::string isa, key;
        Status      failure; // a cache-key.json that does not match this key

        std::mutex                                        mutex;
        bool                                              loaded = false;
        std::shared_ptr<GemmMaster>                       master;
        files::FileIdentity                               masterFile, mappingFile;
        std::map<int, TensileLite::hip::SolutionAdapter*> adapters;

        fs::path masterPath() const
        {
            return path / ("TensileLibrary_lazy_" + isa + ".dat");
        }
        fs::path mappingPath() const
        {
            return path / ("TensileLiteLibrary_lazy_" + isa + "_Mapping.dat");
        }
        fs::path keyPath() const
        {
            return path / "cache-key.json";
        }
        fs::path staging() const
        {
            return path / "staging";
        }

        // Creates cache-key.json once; an existing one must hold this key.
        Status checkKey() const
        {
            const auto           file = keyPath();
            std::vector<uint8_t> bytes;
            std::error_code      error;
            if(!fs::exists(file, error))
            {
                bytes.assign(key.begin(), key.end());
                if(auto status = files::writeAtomically(staging(), file, bytes); !status.ok())
                    return status;
            }
            if(auto status = files::readFile(file, fileLimit, bytes); !status.ok())
                return status;
            if(std::string(bytes.begin(), bytes.end()) != key)
                return failed(Stage::Configure,
                              "JIT solution library disabled: " + file.u8string()
                                  + " does not hold this process's cache key");
            return {};
        }

        // Loads the master and mapping again when either file changed. Callers hold mutex.
        Status refresh()
        {
            const auto masterNow  = files::identify(masterPath());
            const auto mappingNow = files::identify(mappingPath());
            if(loaded && masterNow == masterFile && mappingNow == mappingFile)
                return {};
            std::shared_ptr<GemmMaster> library;
            if(masterNow.exists)
            {
                const auto file = masterPath().string();
                library         = std::dynamic_pointer_cast<GemmMaster>(
                    TensileLite::LoadLibraryFilePreload<TensileLite::ContractionProblemGemm>(file,
                                                                                             {}));
                if(!library || !rowsOf(*library))
                    return failed(Stage::Lookup, "Cannot load the JIT solution library " + file);
                if(!library->initLibraryMapping(file))
                    return failed(Stage::Lookup, "Cannot load the index mapping of " + file);
            }
            master      = std::move(library);
            masterFile  = masterNow;
            mappingFile = mappingNow;
            loaded      = true;
            return {};
        }

        TensileLite::hip::SolutionAdapter* adapter(int device)
        {
            auto& adapter = adapters[device];
            if(!adapter)
            {
                // Leaked like TensileHost's adapters: copied algorithms and
                // captured graphs can launch after any owner is destroyed.
                adapter = new TensileLite::hip::SolutionAdapter;
                adapter->codeObjectDir(path.string());
            }
            return adapter;
        }
    };

    JitLibrary::JitLibrary(fs::path root, DeviceKey deviceKey)
        : m_root(root.lexically_normal())
        , m_deviceKey(std::move(deviceKey))
    {
        if(!m_root.has_filename() && m_root.has_parent_path())
            m_root = m_root.parent_path();
    }

    JitLibrary::~JitLibrary() = default;

    fs::path JitLibrary::defaultRoot()
    {
        if(const char* path = rocblaslt_secure_getenv("HIPBLASLT_JIT_LIBRARY_PATH"); path && *path)
        {
            std::error_code error;
            auto            absolute = fs::absolute(fs::u8path(path), error);
            return error ? fs::u8path(path) : absolute;
        }
#ifdef _WIN32
        std::error_code error;
        const auto      temporary = fs::temp_directory_path(error);
        const char*     user      = std::getenv("USERNAME");
        return temporary / ("hipblaslt-jit-" + std::string(user && *user ? user : "user"));
#else
        return fs::path("/tmp") / ("hipblaslt-jit-" + std::to_string(geteuid()));
#endif
    }

    const fs::path& JitLibrary::root() const noexcept
    {
        return m_root;
    }

    fs::path JitLibrary::schemaDirectory() const
    {
        return m_root / ("v" + std::to_string(jitLibrarySchema));
    }

    fs::path JitLibrary::directory(const CacheKey& key) const
    {
        return schemaDirectory() / key.directoryName();
    }

    void JitLibrary::disable(const std::string& reason)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if(m_disabled.empty())
            m_disabled = reason;
    }

    Status JitLibrary::checkRoot()
    {
        if(m_disabled.empty() && !m_checked)
        {
            if(rocblaslt_process_is_privileged())
                m_disabled = "the process runs with elevated privileges";
#ifndef TENSILE_MSGPACK
            m_disabled = "hipBLASLt was built to read YAML libraries";
#endif
            for(const auto& path : {m_root, schemaDirectory()})
                if(m_disabled.empty())
                    if(auto status = files::privateDirectory(path); !status.ok())
                        m_disabled = status.message;
            m_checked = true;
        }
        if(!m_disabled.empty())
            return failed(Stage::Configure, "JIT solution library disabled: " + m_disabled);
        return {};
    }

    JitLibrary::Directory* JitLibrary::open(const std::string& name,
                                            const std::string& isa,
                                            const std::string& key,
                                            Status&            why)
    {
        auto& slot = m_directories[name];
        if(!slot)
        {
            auto opened  = std::make_unique<Directory>();
            opened->path = schemaDirectory() / name;
            opened->isa  = isa;
            opened->key  = key;
            for(const auto& path : {opened->path, opened->staging()})
                if(auto status = files::privateDirectory(path); !status.ok())
                {
                    m_directories.erase(name);
                    why = failed(Stage::Configure,
                                 "JIT solution library disabled: " + status.message);
                    return nullptr;
                }
            opened->failure = opened->checkKey();
            slot            = std::move(opened);
        }
        why = slot->failure;
        return why.ok() ? slot.get() : nullptr;
    }

    Status JitLibrary::attach(const CacheKey& key, int device, Directory*& attached)
    {
        attached = nullptr;
        std::lock_guard<std::mutex> lock(m_mutex);
        if(auto status = checkRoot(); !status.ok())
            return status;
        Status why;
        attached = open(key.directoryName(), key.isa, key.canonicalJson(), why);
        if(attached)
            addOnce(m_routes[device], attached);
        return why;
    }

    void JitLibrary::discover(int device)
    {
        CacheKey key;
        if(!m_deviceKey || !checkRoot().ok() || !m_deviceKey(device, key).ok())
            return;
        std::error_code error;
        for(fs::directory_iterator next(schemaDirectory(), error), end; !error && next != end;
            next.increment(error))
        {
            const auto name = next->path().filename().string();
            if(!startsWith(name, key.isa + "-"))
                continue;
            std::vector<uint8_t> bytes;
            if(!files::readFile(next->path() / "cache-key.json", fileLimit, bytes).ok())
                continue;
            const std::string json(bytes.begin(), bytes.end());
            auto              candidate = key;
            if(!backendOf(json, candidate.backendId, candidate.backendVersion)
               || candidate.canonicalJson() != json || candidate.directoryName() != name)
                continue;
            Status why;
            if(auto* directory = open(name, key.isa, json, why))
                addOnce(m_routes[device], directory);
        }
    }

    Status JitLibrary::lookup(const CacheKey&                           key,
                              int                                       device,
                              const TensileLite::ContractionProblemGemm& problem,
                              const TensileLite::Hardware&              hardware,
                              size_t                                    count,
                              const std::vector<std::string>&           excludeKernels,
                              std::vector<int32_t>&                     indices)
    {
        indices.clear();
        return staged(Stage::Lookup, [&]() -> Status {
            if(const auto reason = notImplemented(problem))
                return {Status::Code::NotSupported, Stage::Lookup, reason};
            Directory* directory = nullptr;
            if(auto status = attach(key, device, directory); !status.ok())
                return status;
            std::shared_ptr<GemmMaster> master;
            {
                std::lock_guard<std::mutex> lock(directory->mutex);
                if(auto status = directory->refresh(); !status.ok())
                    return status;
                master = directory->master;
            }
            if(!master || !count)
                return {};
            // The stock search cannot list every hit: single solution libraries lack
            // findTopSolutions, and findAllSolutions skips the row predicates.
            std::vector<int32_t> found;
            for(const auto& row : rowsOf(*master)->rows)
            {
                if(!row.first(problem, hardware))
                    continue;
                const auto solution = row.second->findBestSolution(problem, hardware);
                // Only a hand-edited file can hold an index the mapping cannot resolve.
                if(!solution || !isJitIndex(solution->index)
                   || !master->libraryMapping.count(solution->index)
                   || std::find(excludeKernels.begin(), excludeKernels.end(), solution->kernelName)
                          != excludeKernels.end())
                    continue;
                found.push_back(solution->index);
            }
            std::sort(found.begin(), found.end());
            found.erase(std::unique(found.begin(), found.end()), found.end());
            if(found.size() > count)
                found.resize(count);
            indices = std::move(found);
            return {};
        });
    }

    Status JitLibrary::publish(const CacheKey&                           key,
                               int                                       device,
                               const TensileLite::ContractionProblemGemm& problem,
                               const std::vector<BuiltSolution>&         solutions,
                               std::vector<int32_t>&                     indices)
    {
        indices.clear();
        return staged(Stage::Publish, [&]() -> Status {
            if(const auto reason = notImplemented(problem))
                return {Status::Code::NotSupported, Stage::Publish, reason};
            Directory* directory = nullptr;
            if(auto status = attach(key, device, directory); !status.ok())
                return status;
            if(solutions.empty())
                return {};
            for(const auto& built : solutions)
            {
                if(!built.helpers.empty())
                    return {Status::Code::NotSupported,
                            Stage::Publish,
                            "The JIT solution library stores one code object per solution"};
                if(built.object.bytes.empty() || built.generated.kernelName.empty())
                    return failed(Stage::Publish, "A built solution has no code object or kernel");
            }
            const auto type  = problemTypeKey(problem);
            const auto sizes = problemSizes(problem);
            reached(PublishStep::Staged);

            files::FileLock lock;
            const auto      lockFile = schemaDirectory() / "lock";
            if(auto status = files::FileLock::acquire(lockFile, std::chrono::seconds(120), lock);
               !status.ok())
                return status;
            reached(PublishStep::Locked);
            if(auto status = directory->checkKey(); !status.ok())
                return status;

            const auto           staging   = directory->staging();
            const auto           allocator = schemaDirectory() / "allocator.dat";
            std::vector<uint8_t> bytes, master;
            std::error_code      error;
            int64_t              next = jitIndexBase;
            if(fs::exists(allocator, error))
            {
                if(auto status = files::readFile(allocator, fileLimit, bytes); !status.ok())
                    return status;
                if(auto status = msgpack_io::readAllocator(bytes, next); !status.ok())
                    return status;
                if(next < jitIndexBase)
                    return failed(Stage::Publish, allocator.u8string() + " is below the JIT range");
            }
            std::map<int32_t, std::string> mapping;
            if(fs::exists(directory->mappingPath(), error))
            {
                if(auto status = files::readFile(directory->mappingPath(), fileLimit, bytes);
                   !status.ok())
                    return status;
                if(auto status = msgpack_io::readMapping(bytes, mapping); !status.ok())
                    return status;
            }
            std::vector<std::string> rowPrefixes;
            if(fs::exists(directory->masterPath(), error))
            {
                if(auto status = files::readFile(directory->masterPath(), fileLimit, master);
                   !status.ok())
                    return status;
                if(auto status = msgpack_io::readMasterPrefixes(master, rowPrefixes); !status.ok())
                    return status;
            }
            std::map<std::string, int32_t> published;
            for(const auto& [index, prefix] : mapping)
                published.emplace(prefix, index);

            struct Placement
            {
                std::string          prefix, kernel;
                size_t               source = 0;
                int32_t              index  = -1;
                bool                 fresh  = false;
                bool                 row    = false;
                std::vector<uint8_t> entry, predicate;
            };
            std::vector<Placement> placements;
            std::vector<size_t>    placementOf;
            for(size_t source = 0; source < solutions.size(); ++source)
            {
                const auto& kernel = solutions[source].generated.kernelName;
                for(unsigned collision = 0;; ++collision)
                {
                    auto       prefix = entryPrefix(type, kernel, sizes, collision);
                    const auto same   = std::find_if(placements.begin(),
                                                   placements.end(),
                                                   [&](const auto& p) { return p.prefix == prefix; });
                    if(same != placements.end())
                    {
                        if(same->kernel != kernel)
                            continue;
                        placementOf.push_back(static_cast<size_t>(same - placements.begin()));
                        break;
                    }
                    Placement placement{prefix, kernel, source};
                    const auto existing = published.find(prefix);
                    if(existing == published.end())
                        placement.fresh = true;
                    else
                    {
                        // A published name may only be reused for the same kernel;
                        // anything else is a hash collision and takes the next name.
                        msgpack_io::EntryFields fields;
                        if(!files::readFile(directory->path / (prefix + ".dat"), fileLimit, bytes)
                                .ok()
                           || !msgpack_io::readEntry(bytes, fields).ok() || fields.kernelName != kernel
                           || fields.index != existing->second)
                            continue;
                        placement.index     = existing->second;
                        placement.predicate = std::move(fields.predicate);
                        placement.row       = std::find(rowPrefixes.begin(), rowPrefixes.end(), prefix)
                                        == rowPrefixes.end();
                    }
                    placementOf.push_back(placements.size());
                    placements.push_back(std::move(placement));
                    break;
                }
            }

            const auto fresh = std::count_if(
                placements.begin(), placements.end(), [](const auto& p) { return p.fresh; });
            if(fresh)
            {
                if(next + fresh - 1 > INT32_MAX)
                    return failed(Stage::Publish, "The reserved JIT index range is exhausted");
                for(auto& placement : placements)
                    if(placement.fresh)
                        placement.index = static_cast<int32_t>(next++);
                // Allocate first: a crash after this leaves a gap, never a reused index.
                if(auto status = msgpack_io::writeAllocator(next, bytes); !status.ok())
                    return status;
                if(auto status = files::writeAtomically(staging, allocator, bytes); !status.ok())
                    return status;
            }
            reached(PublishStep::Allocated);

            // Code objects, then entries, then the mapping, then the master: each
            // file is complete before any file that refers to it.
            for(const auto& placement : placements)
                if(placement.fresh)
                    if(auto status = files::writeAtomically(
                           staging,
                           directory->path / (placement.prefix + ".co"),
                           solutions[placement.source].object.bytes);
                       !status.ok())
                        return status;
            reached(PublishStep::CodeObjects);

            for(auto& placement : placements)
            {
                if(!placement.fresh)
                    continue;
                if(auto status = msgpack_io::rewriteEntryIndex(
                       solutions[placement.source].generated.entry, placement.index, placement.entry);
                   !status.ok())
                    return status;
                if(!validEntry(placement.entry, placement.index, placement.kernel))
                    return failed(Stage::Publish,
                                  "The entry for " + placement.kernel + " does not load as solution "
                                      + std::to_string(placement.index));
                msgpack_io::EntryFields fields;
                if(auto status = msgpack_io::readEntry(placement.entry, fields); !status.ok())
                    return status;
                placement.predicate = std::move(fields.predicate);
                placement.row       = true;
                if(auto status = files::writeAtomically(
                       staging, directory->path / (placement.prefix + ".dat"), placement.entry);
                   !status.ok())
                    return status;
            }
            reached(PublishStep::Entries);

            if(fresh)
            {
                for(const auto& placement : placements)
                    if(placement.fresh)
                        mapping.emplace(placement.index, placement.prefix);
                if(auto status = msgpack_io::writeMapping(mapping, bytes); !status.ok())
                    return status;
                if(auto status = files::writeAtomically(staging, directory->mappingPath(), bytes);
                   !status.ok())
                    return status;
            }
            reached(PublishStep::Mapping);

            std::vector<msgpack_io::MasterRow> rows;
            for(const auto& placement : placements)
                if(placement.row)
                    rows.push_back({sizes, placement.predicate, placement.prefix});
            if(!rows.empty())
            {
                if(auto status = msgpack_io::appendMasterRows(master, rows, bytes); !status.ok())
                    return status;
                if(auto status = files::writeAtomically(staging, directory->masterPath(), bytes);
                   !status.ok())
                    return status;
            }
            reached(PublishStep::Master);
            lock.release();
            reached(PublishStep::Unlocked);

            {
                std::lock_guard<std::mutex> guard(directory->mutex);
                if(auto status = directory->refresh(); !status.ok())
                    return status;
            }
            for(auto placement : placementOf)
                indices.push_back(placements[placement].index);
            return {};
        });
    }

    JitLibrary::View JitLibrary::resolve(int device, int32_t index, Status& why)
    {
        View view;
        why = staged(Stage::Lookup, [&]() -> Status {
            if(!isJitIndex(index))
                return failed(Stage::Lookup,
                              "Solution index " + std::to_string(index)
                                  + " is outside the reserved JIT range");
            Status reloadFailure;
            for(const bool reload : {false, true})
            {
                std::vector<Directory*> routes;
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    if(!m_disabled.empty())
                        return failed(Stage::Lookup,
                                      "JIT solution library disabled: " + m_disabled);
                    if(reload)
                        discover(device);
                    routes = m_routes[device];
                }
                for(auto* directory : routes)
                {
                    std::lock_guard<std::mutex> lock(directory->mutex);
                    if(reload)
                        if(auto status = directory->refresh(); !status.ok())
                        {
                            reloadFailure = std::move(status);
                            continue;
                        }
                    if(directory->master && directory->master->libraryMapping.count(index))
                    {
                        view = {directory->master, directory->adapter(device)};
                        return {};
                    }
                }
            }
            if(!reloadFailure.ok())
                return reloadFailure;
            return failed(Stage::Lookup,
                          "JIT solution index " + std::to_string(index)
                              + " is not in a JIT solution library for device "
                              + std::to_string(device));
        });
        return view;
    }

    std::shared_ptr<TensileLite::ContractionSolution> JitLibrary::solutionByIndex(
        int device, const TensileLite::Hardware& hardware, int32_t index, Status& why)
    {
        std::shared_ptr<TensileLite::ContractionSolution> solution;
        const auto                                        view = resolve(device, index, why);
        if(!view.master)
            return solution;
        why = staged(Stage::Lookup, [&]() -> Status {
            solution = view.master->getSolutionByIndex(hardware, index);
            if(!solution)
                return failed(Stage::Lookup,
                              "Cannot load JIT solution index " + std::to_string(index));
            return {};
        });
        return solution;
    }
}
