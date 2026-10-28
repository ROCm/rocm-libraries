// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-library.hpp"
#include "hipblaslt-jit-code-object.hpp"
#include "hipblaslt-jit-fs.hpp"
#include "hipblaslt-jit-msgpack.hpp"
#include "hipblaslt-jit-problem-type.hpp"
#include "rocblaslt_secure_env.hpp"
#include <Tensile/CachingLibrary.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/DataTypes.hpp>
#include <Tensile/ExactLogicLibrary.hpp>
#include <Tensile/PlaceholderLibrary.hpp>
#include <Tensile/SingleSolutionLibrary.hpp>
#include <Tensile/Tensile.hpp>
#include <Tensile/hip/HipSolutionAdapter.hpp>
#ifdef TENSILE_MSGPACK
#include <Tensile/msgpack/Writing.hpp>
#endif
#include <algorithm>
#include <cctype>
#include <climits>
#include <cstring>
#include <fstream>
#include <new>
#include <set>
#include <sstream>
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
        using GemmSolution = std::shared_ptr<TensileLite::ContractionSolution>;

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

        bool startsWith(std::string_view text, std::string_view prefix)
        {
            return text.substr(0, prefix.size()) == prefix;
        }

        // Escapes characters that are not safe in one path component. '=' introduces
        // a two-digit hex byte, so the encoding does not collide with a literal.
        std::string filesystemLiteral(std::string_view text)
        {
            std::string out;
            out.reserve(text.size());
            for(unsigned char c : text)
            {
                if(std::isalnum(c) || c == '_' || c == '-' || c == '.' || c == '+')
                    out += static_cast<char>(c);
                else
                {
                    constexpr char digits[] = "0123456789abcdef";
                    out += '=';
                    out += digits[c >> 4];
                    out += digits[c & 0xf];
                }
            }
            return out.empty() ? std::string("x") : out;
        }

        std::string pathComponent(std::string part)
        {
            if(part.empty() || part == "." || part == "..")
                return "x" + part;
            constexpr size_t width = 120;
            if(part.size() <= width)
                return part;
            std::string nested;
            for(size_t i = 0; i < part.size(); i += width)
            {
                if(!nested.empty())
                    nested += '/';
                auto piece = part.substr(i, width);
                if(piece == "." || piece == "..")
                    piece.insert(piece.begin(), 'x');
                nested += piece;
            }
            return nested;
        }

        msgpack_io::CacheKeyDocument documentOf(const CacheKey& key)
        {
            msgpack_io::CacheKeyDocument document;
            document.backendId         = key.backendId;
            document.backendVersion    = key.backendVersion;
            document.comgr             = key.comgr;
            document.rocmPath          = key.rocmPath;
            document.targetId          = key.target.targetId();
            document.isa               = key.target.processorName();
            document.libraryArch       = key.target.libraryArch();
            document.codeObjectVersion = key.codeObjectVersion;
            document.schema            = jitLibrarySchema;
            document.wavefrontSize     = key.wavefrontSize;
            document.environment       = key.environment;
            return document;
        }

        // A stored field that is absent keeps this build's value. comgr, the
        // ROCm path, the schema and the code-object version never reject a
        // directory: only this build writes under its revision directory.
        bool directoryAgrees(const msgpack_io::CacheKeyDocument& stored, const CacheKey& key)
        {
            if(stored.hasIsa && stored.isa != key.target.processorName())
                return false;
            if(stored.hasTargetId && stored.targetId != key.target.targetId())
                return false;
            if(stored.hasLibraryArch && stored.libraryArch != key.target.libraryArch())
                return false;
            if(stored.hasWavefront && stored.wavefrontSize != key.wavefrontSize)
                return false;
            if(stored.hasBackendId && stored.backendId != key.backendId)
                return false;
            if(stored.hasBackendVersion && stored.backendVersion != key.backendVersion)
                return false;
            return true;
        }

        Status ensureEntryParents(const fs::path& root, const fs::path& file)
        {
            const auto relative = file.parent_path().lexically_relative(root);
            if(relative.empty() || relative == ".")
                return {};
            fs::path current = root;
            for(const auto& part : relative)
            {
                if(part == "." || part == "..")
                    return failed(Stage::Publish, "An entry path escapes the library directory");
                current /= part;
                if(auto status = files::privateDirectory(current); !status.ok())
                    return status;
            }
            return {};
        }

        template <class T>
        void addOnce(std::vector<T*>& items, T* item)
        {
            if(std::find(items.begin(), items.end(), item) == items.end())
                items.push_back(item);
        }

        // An absent file is an empty mapping. TensileLite reads a damaged
        // mapping as an empty one, and this library never writes one empty.
        Status readMapping(Stage stage, const fs::path& file, std::map<int, std::string>& mapping)
        {
            mapping.clear();
            std::error_code error;
            if(!fs::exists(file, error))
                return {};
            mapping = TensileLite::LoadLibraryMapping(file.string());
            if(mapping.empty())
                return failed(stage, "Cannot read the index mapping " + file.u8string());
            return {};
        }

        std::vector<uint8_t> encodeLibrary(const GemmMaster& library)
        {
#ifdef TENSILE_MSGPACK
            return TensileLite::MessagePackWriteLibraryData(library);
#else
            throw std::runtime_error("Writing a library needs a MessagePack build of TensileLite");
#endif
        }

        std::vector<uint8_t> encodeMapping(const std::map<int, std::string>& mapping)
        {
#ifdef TENSILE_MSGPACK
            return TensileLite::MessagePackWriteLibraryMapping(mapping);
#else
            throw std::runtime_error("Writing a mapping needs a MessagePack build of TensileLite");
#endif
        }

        // The file prefix a master row selects, or nullptr for a row that is
        // not a placeholder.
        const std::string* placeholderPrefix(const GemmRows::Row& row)
        {
            const auto placeholder = dynamic_cast<const TensileLite::PlaceholderLibrary<
                TensileLite::ContractionProblemGemm,
                TensileLite::ContractionSolution>*>(row.second.get());
            return placeholder ? &placeholder->filePrefix : nullptr;
        }

        // A master row: exactly sizes and the solution's own problem predicate.
        TensileLite::ProblemPredicate<TensileLite::ContractionProblemGemm>
            rowPredicate(const std::vector<size_t>& sizes, const GemmSolution& solution)
        {
            using Gemm      = TensileLite::ContractionProblemGemm;
            using Predicate = TensileLite::Predicates::Predicate<Gemm>;
            using And       = TensileLite::Predicates::And<Gemm>;
            std::vector<std::shared_ptr<Predicate>> terms;
            for(size_t index = 0; index < sizes.size(); ++index)
                terms.push_back(std::make_shared<TensileLite::Predicates::Contraction::SizeEqual>(
                    index, sizes[index]));
            if(const auto all = std::dynamic_pointer_cast<And>(solution->problemPredicate))
                terms.insert(terms.end(), all->value.begin(), all->value.end());
            else if(solution->problemPredicate)
                terms.push_back(solution->problemPredicate);
            return {std::make_shared<And>(terms)};
        }

        // An entry: every solution of one executable, each selected by its own
        // problem predicate.
        std::shared_ptr<GemmMaster> entryOf(const std::vector<GemmSolution>& solutions)
        {
            using Single = TensileLite::SingleSolutionLibrary<TensileLite::ContractionProblemGemm,
                                                              TensileLite::ContractionSolution>;
            auto rows  = std::make_shared<GemmRows>();
            auto entry = std::make_shared<GemmMaster>();
            for(const auto& solution : solutions)
            {
                entry->solutions.emplace(solution->index, solution);
                rows->rows.emplace_back(solution->problemPredicate,
                                        std::make_shared<Single>(solution));
            }
            entry->library = rows;
            return entry;
        }
    }

    std::map<std::string, std::string> compilerEnvironment(const char* const* environment)
    {
        // Logging, temporary files and time statistics do not change code-object bytes.
        static constexpr std::string_view ignored[]
            = {"AMD_COMGR_SAVE_TEMPS",
               "AMD_COMGR_SAVE_LLVM_TEMPS",
               "AMD_COMGR_REDIRECT_LOGS",
               "AMD_COMGR_EMIT_VERBOSE_LOGS",
               "AMD_COMGR_LOG_LEVEL",
               "AMD_COMGR_TIME_STATISTICS",
               "AMD_COMGR_TIME_STATISTICS_GRANULARITY"};
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
        key.target            = target;
        key.wavefrontSize     = target.wavefrontSize;
        key.backendId         = backend.id;
        key.backendVersion    = backend.version;
        key.codeObjectVersion = codeObjectVersion;
        key.comgr             = toolchain.comgr;
        key.rocmPath          = toolchain.rocmPath;
        key.environment       = toolchain.environment;
        return key;
    }

    std::string CacheKey::directoryName() const
    {
        // One path component. Toolchain identity (comgr, ROCm path, compiler
        // environment) is recorded in the cache key and is not part of the name.
        auto name = filesystemLiteral(target.processorName()) + "_"
                    + filesystemLiteral(target.targetId()) + "_"
                    + filesystemLiteral(target.libraryArch()) + "_wf"
                    + std::to_string(wavefrontSize) + "_" + filesystemLiteral(backendId) + "_"
                    + std::to_string(backendVersion);
        if(name == "." || name == "..")
            name.insert(name.begin(), 'x');
        return name;
    }

    std::string problemTypeKey(const TensileLite::ContractionProblemGemm& problem)
    {
        const auto fields = problemTypeFields(problem);
        std::string key;
        const auto add = [&](const std::string& token) {
            if(!key.empty())
                key += '.';
            key += token;
        };
        const auto flag = [&](const char* name, bool value) {
            add(std::string(name) + (value ? "1" : "0"));
        };
        const auto number = [&](const char* name, long long value) {
            add(std::string(name) + std::to_string(value));
        };
        const auto dataType = [&](const char* name, rocisa::DataType value) {
            number(name, static_cast<int>(value));
        };
        add("GEMM");
        flag("sb", fields.stridedBatched);
        flag("ta", fields.transA);
        flag("tb", fields.transB);
        dataType("dta", fields.aType);
        dataType("dtb", fields.bType);
        dataType("dtc", fields.cType);
        dataType("dtd", fields.dType);
        dataType("maca", fields.computeInputTypeA);
        dataType("macb", fields.computeInputTypeB);
        dataType("compute", fields.computeType);
        dataType("f32", fields.f32XdlMathOp);
        flag("hpa", fields.highPrecisionAccumulate);
        number("bias", fields.useBias);
        flag("e", fields.useE);
        flag("grad", fields.useGradient);
        add(std::string("scale") + filesystemLiteral(fields.useScaleAB.empty() ? "none" : fields.useScaleAB));
        flag("scd", fields.useScaleCD);
        number("sav", fields.useScaleAlphaVec);
        flag("amax", fields.outputAmaxD);
        number("mxa", fields.mxBlockA);
        number("mxb", fields.mxBlockB);
        number("sparse", fields.sparse);
        flag("swa", fields.swizzleTensorA);
        flag("swb", fields.swizzleTensorB);
        flag("gate", fields.useGateResidual);
        if(fields.mxBlockA)
            dataType("mxsa", fields.mxTypeA);
        if(fields.mxBlockB)
            dataType("mxsb", fields.mxTypeB);
        if(!fields.biasDataTypeWhiteList.empty())
            dataType("biasdt", fields.biasDataTypeWhiteList.front());
        if(!fields.biasSrcWhiteList.empty())
            number("biassrc", fields.biasSrcWhiteList.front());
        if(!fields.gateResidualDataTypeWhiteList.empty())
            dataType("gatedt", fields.gateResidualDataTypeWhiteList.front());
        if(fields.useE)
            dataType("dte", fields.eType);
        if(fields.activationType != TensileLite::ActivationType::None)
            number("act", static_cast<unsigned>(fields.activationType));
        if(fields.activationType != TensileLite::ActivationType::None)
            dataType("actc", fields.activationComputeDataType);
        add(std::string("id") + filesystemLiteral(fields.operationIdentifier));
        flag("gg", fields.groupedGemm);
        return key;
    }

    std::vector<size_t> problemSizes(const TensileLite::ContractionProblemGemm& problem)
    {
        std::vector<size_t> sizes;
        const auto          count = problem.c().dimensions() + problem.boundIndices().size();
        for(size_t index = 0; index < count; ++index)
            sizes.push_back(problem.size(index));
        return sizes;
    }

    std::string sliceKey(const TensileLite::ContractionProblemGemm& problem)
    {
        const auto abbrev = [](rocisa::DataType type) { return rocisa::TypeAbbrev(type); };
        const auto single = [](rocisa::DataType type) {
            return type == rocisa::DataType::Float;
        };
        std::string compute = abbrev(problem.computeInputTypeA());
        if(problem.computeInputTypeA() != problem.computeInputTypeB())
            compute += abbrev(problem.computeInputTypeB());
        // Same tokens as TensileLite's lazy placeholder: operation identifier,
        // then selection()'s datatype, scale and epilogue suffix.
        std::string key = problem.operationIdentifier();
        key += "_" + abbrev(problem.a().dataType()) + abbrev(problem.b().dataType());
        key += "_" + abbrev(problem.c().dataType()) + compute;
        if(problem.activationType() != TensileLite::ActivationType::None)
        {
            auto name = TensileLite::ToString(problem.activationType());
            for(auto& character : name)
                character = static_cast<char>(std::toupper(static_cast<unsigned char>(character)));
            if(name == "ALL")
                key += "_A";
            else if(name == "HIPBLASLT_ALL")
                key += "_HA";
            else
                key += "_" + name;
        }
        if(problem.mxBlockA())
            key += "_MXA" + abbrev(problem.mxTypeA()) + "B" + std::to_string(problem.mxBlockA());
        if(problem.mxBlockB())
            key += "_MXB" + abbrev(problem.mxTypeB()) + "B" + std::to_string(problem.mxBlockB());
        if(problem.swizzleTensorA())
            key += "_STA";
        if(problem.swizzleTensorB())
            key += "_STB";
        if(problem.useBias())
            key += "_Bias";
        if(problem.useE())
            key += problem.useGradient() ? "_Grad" : "_Aux";
        if(problem.groupedGemm())
            key += "_GG";
        else if(!problem.stridedBatched())
            key += "_GB";
        if(problem.useScaleAB() == "Scalar")
            key += "_SAB";
        else if(problem.useScaleAB() == "Vector")
            key += "_SABV";
        if(problem.useScaleCD())
            key += "_SCD";
        if(problem.useScaleAlphaVec())
            key += "_SAV";
        if(problem.sparse())
        {
            key += problem.sparse() == 2 ? "_SPB" : "_SPA";
            key += "ML" + std::to_string(problem.metadataLayout());
        }
        if(!single(problem.f32XdlMathOp()) && single(problem.computeInputTypeA())
           && single(problem.computeInputTypeB()))
            key += "_M" + abbrev(problem.f32XdlMathOp());
        if(problem.getUseDeviceUserArguments())
            key += "_UA";
        if(problem.useGateResidual())
            key += "_GateRes";
        return pathComponent(filesystemLiteral(key));
    }

    std::string entryPrefix(const std::string&              problemTypeKey,
                            const std::vector<std::string>& kernelNames,
                            const std::vector<size_t>&      sizes)
    {
        std::string kernels;
        for(const auto& kernel : std::set<std::string>(kernelNames.begin(), kernelNames.end()))
        {
            if(!kernels.empty())
                kernels += '+';
            kernels += kernel;
        }
        std::string sizesText;
        for(auto size : sizes)
        {
            if(!sizesText.empty())
                sizesText += 'x';
            sizesText += std::to_string(size);
        }
        return "k/" + pathComponent(filesystemLiteral(kernels)) + "/t/"
               + pathComponent(filesystemLiteral(problemTypeKey)) + "/s/"
               + pathComponent(filesystemLiteral(sizesText));
    }

    std::shared_ptr<TensileLite::ContractionSolution>
        copySolution(const std::shared_ptr<TensileLite::ContractionSolution>& solution)
    {
        if(!solution)
            return {};
        const auto copied = std::dynamic_pointer_cast<GemmMaster>(
            TensileLite::LoadLibraryData<TensileLite::ContractionProblemGemm>(
                encodeLibrary(*entryOf({solution}))));
        if(!copied || copied->solutions.size() != 1)
            throw std::runtime_error("Solution " + solution->solutionName
                                     + " does not load from TensileLite's encoding");
        return copied->solutions.begin()->second;
    }

    std::vector<std::shared_ptr<TensileLite::ContractionSolution>> supportingSolutions(
        const std::vector<std::shared_ptr<TensileLite::ContractionSolution>>& solutions,
        const TensileLite::ContractionProblemGemm&                            problem,
        const TensileLite::Hardware&                                          hardware,
        size_t                                                                workspaceLimit)
    {
        auto limited = problem;
        limited.setWorkspaceSize(workspaceLimit);
        std::vector<std::shared_ptr<TensileLite::ContractionSolution>> result;
        for(const auto& solution : solutions)
        {
            if(!solution)
                continue;
            TensileLite::Task task(hardware, limited, *solution);
            if(TensileLite::selectionPredicate(task, hardware, *solution, limited)
               && solution->requiredWorkspaceSize(limited, hardware) <= workspaceLimit)
                result.push_back(solution);
        }
        return result;
    }

    struct JitLibrary::Directory
    {
        struct Slice
        {
            fs::path                                          path;
            std::mutex                                        mutex;
            bool                                              loaded = false;
            std::shared_ptr<GemmMaster>                       master;
            files::FileIdentity                               masterFile, mappingFile;
            std::map<int, TensileLite::hip::SolutionAdapter*> adapters;
            // Code-object prefixes already passed to hipModuleLoad on this device.
            // A prefix is written once and names one executable, so this survives
            // refresh(), which publish calls after every new entry in the slice.
            std::map<int, std::set<std::string>>              loadedCodeObjects;

            fs::path masterPath(const Target& target) const
            {
                return path / ("TensileLibrary_lazy_" + target.processorName() + ".dat");
            }
            fs::path mappingPath(const Target& target) const
            {
                return path / ("TensileLiteLibrary_lazy_" + target.processorName() + "_Mapping.dat");
            }

            // Loads this slice's master and mapping only. Callers hold mutex.
            Status refresh(const Target& target)
            {
                const auto masterNow  = files::identify(masterPath(target));
                const auto mappingNow = files::identify(mappingPath(target));
                if(loaded && masterNow == masterFile && mappingNow == mappingFile)
                    return {};
                std::shared_ptr<GemmMaster> library;
                if(masterNow.exists)
                {
                    const auto file = masterPath(target).string();
                    library         = std::dynamic_pointer_cast<GemmMaster>(
                        TensileLite::LoadLibraryFilePreload<TensileLite::ContractionProblemGemm>(
                            file, {}));
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
                auto& slot = adapters[device];
                if(!slot)
                {
                    // Leaked like TensileHost's adapters: copied algorithms and
                    // captured graphs can launch after any owner is destroyed.
                    slot = new TensileLite::hip::SolutionAdapter;
                    slot->codeObjectDir(path.string());
                }
                return slot;
            }
        };

        fs::path    path;
        CacheKey    identity;
        Status      failure; // a cache-key.msgpack this directory cannot use

        std::mutex                                  sliceMutex;
        std::map<std::string, std::unique_ptr<Slice>> slices;
        std::mutex                                    indexMutex;
        files::FileIdentity                           indexIdentity;
        bool                                          indexKnown = false;
        std::map<int, std::string>                    indexCache;

        // Reads indices.msgpack again only when the file changes, then returns
        // that one index's slice. name is empty when the index is absent.
        Status sliceName(int32_t index, std::string& name)
        {
            std::lock_guard<std::mutex> lock(indexMutex);
            const auto                  now = files::identify(indexMapPath());
            if(!indexKnown || now != indexIdentity)
            {
                std::map<int, std::string> fresh;
                if(now.exists)
                {
                    if(auto status = readMapping(Stage::Lookup, indexMapPath(), fresh); !status.ok())
                        return status;
                }
                indexCache    = std::move(fresh);
                indexIdentity = now;
                indexKnown    = true;
            }
            name.clear();
            const auto located = indexCache.find(index);
            if(located != indexCache.end())
                name = located->second;
            return {};
        }

        fs::path keyPath() const
        {
            return path / "cache-key.msgpack";
        }
        fs::path staging() const
        {
            return path / "staging";
        }
        fs::path indexMapPath() const
        {
            return path / "indices.msgpack";
        }
        fs::path helperDirectory() const
        {
            return path / "helpers";
        }
        fs::path helperPath(const std::string& kernelName) const
        {
            return helperDirectory() / (pathComponent(filesystemLiteral(kernelName)) + ".o");
        }
        fs::path slicePath(const std::string& key) const
        {
            return path / "slices" / key;
        }

        // Does not create the directory. A query that misses reads nothing.
        Slice* existingSlice(const std::string& key)
        {
            std::lock_guard<std::mutex> lock(sliceMutex);
            const auto                  found = slices.find(key);
            if(found != slices.end())
                return found->second.get();
            std::error_code error;
            const auto      location = slicePath(key);
            if(!fs::is_directory(location, error))
                return nullptr;
            auto opened  = std::make_unique<Slice>();
            opened->path = location;
            auto* raw    = opened.get();
            slices.emplace(key, std::move(opened));
            return raw;
        }

        Status makeSlice(const std::string& key, Slice*& out)
        {
            out = nullptr;
            if(auto status = files::privateDirectory(path / "slices"); !status.ok())
                return status;
            const auto location = slicePath(key);
            if(auto status = ensureEntryParents(path / "slices", location / "keep"); !status.ok())
                return status;
            if(auto status = files::privateDirectory(location); !status.ok())
                return status;
            std::lock_guard<std::mutex> lock(sliceMutex);
            auto&                       slot = slices[key];
            if(!slot)
            {
                slot       = std::make_unique<Slice>();
                slot->path = location;
            }
            out = slot.get();
            return {};
        }

        // Creates cache-key.msgpack once. An existing file may differ in comgr,
        // the ROCm path, the schema integer or the code-object version; those do
        // not make the directory unusable. A present device or backend field
        // that disagrees does.
        Status checkKey() const
        {
            const auto           file = keyPath();
            std::vector<uint8_t> bytes;
            std::error_code      error;
            if(!fs::exists(file, error))
            {
                if(auto status = msgpack_io::writeCacheKey(documentOf(identity), bytes); !status.ok())
                    return status;
                if(auto status = files::writeAtomically(staging(), file, bytes); !status.ok())
                    return status;
            }
            if(auto status = files::readFile(file, fileLimit, bytes); !status.ok())
                return status;
            msgpack_io::CacheKeyDocument stored;
            if(auto status = msgpack_io::readCacheKey(bytes, stored); !status.ok())
                return status;
            if(!directoryAgrees(stored, identity))
                return failed(Stage::Configure,
                              file.u8string() + " records a different device or backend");
            return {};
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

    std::string JitLibrary::buildRevision()
    {
#define HIPBLASLT_JIT_TO_STR2(x) #x
#define HIPBLASLT_JIT_TO_STR(x) HIPBLASLT_JIT_TO_STR2(x)
        return HIPBLASLT_JIT_TO_STR(HIPBLASLT_VERSION_TWEAK);
#undef HIPBLASLT_JIT_TO_STR
#undef HIPBLASLT_JIT_TO_STR2
    }

    const fs::path& JitLibrary::root() const noexcept
    {
        return m_root;
    }

    fs::path JitLibrary::buildDirectory() const
    {
        return m_root / ("v" + std::to_string(jitLibrarySchema))
               / filesystemLiteral(buildRevision());
    }

    fs::path JitLibrary::directory(const CacheKey& key) const
    {
        return buildDirectory() / key.directoryName();
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
            if(m_disabled.empty() && buildRevision().empty())
                m_disabled = "this hipBLASLt build has no git revision";
            for(const auto& path : {m_root, buildDirectory().parent_path(), buildDirectory()})
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
                                            const CacheKey&    key,
                                            Status&            why)
    {
        auto& slot = m_directories[name];
        if(!slot)
        {
            auto opened      = std::make_unique<Directory>();
            opened->path     = buildDirectory() / name;
            opened->identity = key;
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
        attached = open(key.directoryName(), key, why);
        if(attached)
            addOnce(m_routes[device], attached);
        return why;
    }

    Status JitLibrary::acquireHelper(const CacheKey&                         key,
                                     int                                     device,
                                     const std::string&                      kernelName,
                                     const std::function<Status(Artifact&)>& compile,
                                     Artifact&                               out)
    {
        out = {};
        if(kernelName.empty())
            return failed(Stage::Build, "A helper has no kernel name");
        Directory* directory = nullptr;
        if(auto status = attach(key, device, directory); !status.ok())
            return status;

        const auto file   = directory->helperPath(kernelName);
        auto       stored = [&](bool& present) -> Status {
            std::error_code error;
            present = fs::is_regular_file(file, error);
            if(!present)
                return {};
            std::vector<uint8_t> bytes;
            if(auto status = files::readFile(file, fileLimit, bytes); !status.ok())
                return status;
            out.name  = kernelName;
            out.kind  = Artifact::Kind::Relocatable;
            out.bytes = std::move(bytes);
            return {};
        };

        bool present = false;
        if(auto status = stored(present); !status.ok() || present)
            return status;

        Artifact compiled;
        if(auto status = compile(compiled); !status.ok())
            return status;
        if(compiled.kind != Artifact::Kind::Relocatable || compiled.bytes.empty())
            return failed(Stage::Build, "Helper " + kernelName + " produced no relocatable");
        const auto defined = code_object::inspect(compiled.bytes);
        const auto& names  = defined.kernelNames;
        if(!defined.ok() || std::find(names.begin(), names.end(), kernelName) == names.end())
            return failed(Stage::Build, "The relocatable built for helper " + kernelName
                                            + " does not define it");
        compiled.name = kernelName;

        files::FileLock lock;
        const auto      lockFile = buildDirectory() / "lock";
        if(auto status = files::FileLock::acquire(lockFile, std::chrono::seconds(120), lock);
           !status.ok())
            return status;
        if(auto status = stored(present); !status.ok() || present)
            return status;
        if(auto status = files::privateDirectory(directory->helperDirectory()); !status.ok())
            return status;
        if(auto status = ensureEntryParents(directory->helperDirectory(), file); !status.ok())
            return status;
        if(auto status = files::writeAtomically(directory->staging(), file, compiled.bytes);
           !status.ok())
            return status;
        out = std::move(compiled);
        return {};
    }

    void JitLibrary::discover(int device)
    {
        CacheKey key;
        if(!m_deviceKey || !checkRoot().ok() || !m_deviceKey(device, key).ok())
            return;
        std::error_code error;
        for(fs::directory_iterator next(buildDirectory(), error), end; !error && next != end;
            next.increment(error))
        {
            if(!next->is_directory())
                continue;
            const auto           name = next->path().filename().string();
            std::vector<uint8_t> bytes;
            if(!files::readFile(next->path() / "cache-key.msgpack", fileLimit, bytes).ok())
                continue;
            msgpack_io::CacheKeyDocument stored;
            if(!msgpack_io::readCacheKey(bytes, stored).ok())
                continue;
            auto candidate = key;
            if(stored.hasBackendId)
                candidate.backendId = stored.backendId;
            if(stored.hasBackendVersion)
                candidate.backendVersion = stored.backendVersion;
            if(candidate.directoryName() != name || !directoryAgrees(stored, candidate))
                continue;
            Status why;
            if(auto* directory = open(name, candidate, why))
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
            const auto sliceName = sliceKey(problem);
            auto*      slice     = directory->existingSlice(sliceName);
            std::shared_ptr<GemmMaster> master;
            if(slice)
            {
                std::lock_guard<std::mutex> lock(slice->mutex);
                if(auto status = slice->refresh(directory->identity.target); !status.ok())
                    return status;
                master = slice->master;
            }
            if(!master || !count)
                return {};
            // The stock search cannot list every hit: single solution libraries lack
            // findTopSolutions, and findAllSolutions skips the row predicates. A
            // matching row's entry lists each of its solutions whose own
            // predicates hold.
            std::vector<int32_t> found;
            for(const auto& row : rowsOf(*master)->rows)
            {
                if(!row.first(problem, hardware))
                    continue;
                TensileLite::SolutionSet<TensileLite::ContractionSolution> solutions;
                try
                {
                    solutions = row.second->findAllSolutions(problem, hardware);
                }
                catch(const std::exception& error)
                {
                    // This build wrote the entry, so a failure is a damaged library.
                    return failed(Stage::Lookup,
                                  "Cannot read a JIT solution library entry in "
                                      + slice->path.u8string() + ": " + error.what());
                }
                for(const auto& solution : solutions)
                    // Only a hand-edited file can hold an index the mapping cannot resolve.
                    if(solution && isJitIndex(solution->index)
                       && master->libraryMapping.count(solution->index)
                       && !excludedKernel(excludeKernels, solution->kernelName))
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

    Status JitLibrary::publish(
        const CacheKey&                                                       key,
        int                                                                   device,
        const TensileLite::ContractionProblemGemm&                            problem,
        const Artifact&                                                       executable,
        const std::vector<std::shared_ptr<TensileLite::ContractionSolution>>& solutions,
        std::vector<int32_t>&                                                 indices)
    {
        indices.clear();
        return staged(Stage::Publish, [&]() -> Status {
            if(const auto reason = notImplemented(problem))
                return {Status::Code::NotSupported, Stage::Publish, reason};
            Directory* directory = nullptr;
            if(auto status = attach(key, device, directory); !status.ok())
                return status;
            const auto& target = directory->identity.target;
            if(executable.bytes.empty() || executable.kind != Artifact::Kind::Executable)
                return failed(Stage::Publish, "A built solution has no executable");
            if(solutions.empty())
                return {};
            const auto defined = code_object::inspect(executable.bytes);
            if(!defined.ok())
                return failed(Stage::Publish, "Cannot read the kernels of a built executable");
            std::vector<GemmSolution> named; // one per solution name, in order
            std::vector<size_t>       nameOf; // per solution, its position in named
            std::vector<std::string>  kernels;
            for(const auto& solution : solutions)
            {
                if(!solution || solution->kernelName.empty() || solution->solutionName.empty())
                    return failed(Stage::Publish, "A built solution has no kernel or name");
                if(std::find(defined.kernelNames.begin(),
                             defined.kernelNames.end(),
                             solution->kernelName)
                   == defined.kernelNames.end())
                    return failed(Stage::Publish,
                                  "The executable does not define kernel " + solution->kernelName);
                const auto same
                    = std::find_if(named.begin(), named.end(), [&](const GemmSolution& other) {
                          return other->solutionName == solution->solutionName;
                      });
                nameOf.push_back(static_cast<size_t>(same - named.begin()));
                if(same == named.end())
                    named.push_back(solution);
                kernels.push_back(solution->kernelName);
            }
            const auto sizes  = problemSizes(problem);
            const auto prefix = entryPrefix(problemTypeKey(problem), kernels, sizes);

            files::FileLock lock;
            const auto      lockFile = buildDirectory() / "lock";
            if(auto status = files::FileLock::acquire(lockFile, std::chrono::seconds(120), lock);
               !status.ok())
                return status;
            if(auto status = directory->checkKey(); !status.ok())
                return status;
            const auto sliceName = sliceKey(problem);
            Directory::Slice* slice = nullptr;
            if(auto status = directory->makeSlice(sliceName, slice); !status.ok())
                return status;

            const auto           staging   = directory->staging();
            const auto           allocator = buildDirectory() / "allocator.dat";
            std::vector<uint8_t> bytes;
            std::error_code      error;
            int64_t              next = jitIndexBase;
            if(fs::exists(allocator, error))
            {
                if(auto status = files::readFile(allocator, fileLimit, bytes); !status.ok())
                    return status;
                if(auto status = msgpack_io::readAllocator(bytes, next); !status.ok())
                    return status;
                if(next < jitIndexBase || next > jitIndexEnd)
                    return failed(Stage::Publish,
                                  allocator.u8string() + " is outside the JIT range");
            }
            std::map<int, std::string> mapping;
            if(auto status = readMapping(Stage::Publish, slice->mappingPath(target), mapping);
               !status.ok())
                return status;
            std::shared_ptr<GemmMaster> master;
            if(fs::exists(slice->masterPath(target), error))
            {
                const auto file = slice->masterPath(target).string();
                master          = std::dynamic_pointer_cast<GemmMaster>(
                    TensileLite::LoadLibraryFilePreload<TensileLite::ContractionProblemGemm>(
                        file, {}));
                if(!master || !rowsOf(*master))
                    return failed(Stage::Publish, "Cannot load the JIT solution library " + file);
            }

            // The same kernels, problem type and sizes reuse the entry, and a
            // solution already in it keeps its index. entry is in index order.
            const auto entryFile = slice->path / (prefix + ".dat");
            const bool present
                = std::any_of(mapping.begin(), mapping.end(), [&](const auto& mapped) {
                      return mapped.second == prefix;
                  });
            std::vector<GemmSolution> entry;
            if(present)
            {
                const auto stored = std::dynamic_pointer_cast<GemmMaster>(
                    TensileLite::LoadLibraryFile<TensileLite::ContractionProblemGemm>(
                        entryFile.string()));
                if(!stored)
                    return failed(Stage::Publish,
                                  "Cannot load the JIT solution library entry "
                                      + entryFile.u8string());
                for(const auto& [index, solution] : stored->solutions)
                    entry.push_back(solution);
            }
            std::vector<int32_t>      namedIndex;
            std::vector<GemmSolution> added;
            for(const auto& solution : named)
            {
                const auto same
                    = std::find_if(entry.begin(), entry.end(), [&](const GemmSolution& stored) {
                          return stored->solutionName == solution->solutionName;
                      });
                if(same != entry.end())
                {
                    namedIndex.push_back((*same)->index);
                    continue;
                }
                if(next >= jitIndexEnd)
                    return failed(Stage::Publish, "The reserved JIT index range is exhausted");
                auto copy   = copySolution(solution);
                copy->index = static_cast<int32_t>(next++);
                namedIndex.push_back(copy->index);
                added.push_back(std::move(copy));
            }
            entry.insert(entry.end(), added.begin(), added.end());

            if(!added.empty())
            {
                // Allocate first: a crash after this leaves a gap, never a reused index.
                if(auto status = msgpack_io::writeAllocator(next, bytes); !status.ok())
                    return status;
                if(auto status = files::writeAtomically(staging, allocator, bytes); !status.ok())
                    return status;
            }

            // The executable is already linked with any helpers.
            if(!present)
            {
                const auto object = slice->path / (prefix + ".co");
                if(auto status = ensureEntryParents(slice->path, object); !status.ok())
                    return status;
                if(auto status = files::writeAtomically(staging, object, executable.bytes);
                   !status.ok())
                    return status;
            }

            if(!added.empty())
            {
                bytes            = encodeLibrary(*entryOf(entry));
                const auto check = std::dynamic_pointer_cast<GemmMaster>(
                    TensileLite::LoadLibraryData<TensileLite::ContractionProblemGemm>(bytes));
                const auto loads = [&](const GemmSolution& solution) {
                    if(!check)
                        return false;
                    const auto found = check->solutions.find(solution->index);
                    return found != check->solutions.end()
                           && found->second->kernelName == solution->kernelName;
                };
                for(const auto& solution : entry)
                    if(!loads(solution))
                        return failed(Stage::Publish,
                                      "The entry for " + solution->solutionName
                                          + " does not load as solution "
                                          + std::to_string(solution->index));
                if(auto status = ensureEntryParents(slice->path, entryFile); !status.ok())
                    return status;
                if(auto status = files::writeAtomically(staging, entryFile, bytes); !status.ok())
                    return status;
            }

            // A crash can leave entry solutions the mappings do not name yet.
            std::map<int, std::string> indexMap;
            if(auto status = readMapping(Stage::Publish, directory->indexMapPath(), indexMap);
               !status.ok())
                return status;
            bool mappingChanged = false, indexMapChanged = false;
            for(const auto& solution : entry)
            {
                auto& mapped = mapping[solution->index];
                mappingChanged |= mapped != prefix;
                mapped = prefix;
                auto& sliceOf = indexMap[solution->index];
                indexMapChanged |= sliceOf != sliceName;
                sliceOf = sliceName;
            }
            if(mappingChanged)
                if(auto status = files::writeAtomically(
                       staging, slice->mappingPath(target), encodeMapping(mapping));
                   !status.ok())
                    return status;
            if(indexMapChanged)
                if(auto status = files::writeAtomically(
                       staging, directory->indexMapPath(), encodeMapping(indexMap));
                   !status.ok())
                    return status;

            // One master row per entry solution, where the entry's rows were.
            auto placeholder = std::make_shared<TensileLite::PlaceholderLibrary<
                TensileLite::ContractionProblemGemm,
                TensileLite::ContractionSolution>>();
            placeholder->filePrefix = prefix;
            std::vector<GemmRows::Row> ours, rows;
            for(const auto& solution : entry)
                ours.emplace_back(rowPredicate(sizes, solution), placeholder);
            size_t stale = 0;
            if(master)
                for(const auto& row : rowsOf(*master)->rows)
                {
                    const auto* rowPrefix = placeholderPrefix(row);
                    if(!rowPrefix || *rowPrefix != prefix)
                        rows.push_back(row);
                    else if(stale++ == 0)
                        rows.insert(rows.end(), ours.begin(), ours.end());
                }
            if(stale == 0)
                rows.insert(rows.end(), ours.begin(), ours.end());
            if(!added.empty() || stale != entry.size())
            {
                auto updated     = std::make_shared<GemmMaster>();
                updated->library = std::make_shared<GemmRows>(rows);
                if(master)
                    updated->version = master->version;
                if(auto status = files::writeAtomically(
                       staging, slice->masterPath(target), encodeLibrary(*updated));
                   !status.ok())
                    return status;
            }

            lock.release();

            {
                std::lock_guard<std::mutex> guard(slice->mutex);
                if(auto status = slice->refresh(target); !status.ok())
                    return status;
            }
            for(const auto position : nameOf)
                indices.push_back(namedIndex[position]);
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
                    std::string sliceOf;
                    if(auto status = directory->sliceName(index, sliceOf); !status.ok())
                    {
                        reloadFailure = std::move(status);
                        continue;
                    }
                    Directory::Slice* slice = nullptr;
                    if(!sliceOf.empty())
                        slice = directory->existingSlice(sliceOf);
                    if(!slice)
                        continue;
                    std::lock_guard<std::mutex> lock(slice->mutex);
                    if(reload)
                        if(auto status = slice->refresh(directory->identity.target); !status.ok())
                        {
                            reloadFailure = std::move(status);
                            continue;
                        }
                    if(slice->master && slice->master->libraryMapping.count(index))
                    {
                        auto*      adapter = slice->adapter(device);
                        const auto prefix  = slice->master->libraryMapping.at(index);
                        if(slice->loadedCodeObjects[device].insert(prefix).second)
                            static_cast<void>(adapter->loadCodeObjectFile(
                                (slice->path / (prefix + ".co")).string()));
                        view = {slice->master, adapter};
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

    namespace
    {
        struct ActiveQuery
        {
            bool                     engaged = false;
            bool                     allowGeneration = false;
            DeviceTarget             target;
            BackendInfo              backend;
            size_t                   workspace = 0;
            std::vector<std::string> exclude;
            JitLibrary::GenerateMissing generate;
            Status                   failure;
        };

        ActiveQuery& activeQuery()
        {
            thread_local ActiveQuery query;
            return query;
        }

        // Empty when the problem cannot be named. That problem is not generated
        // and does not suppress a later one.
        std::string generationAttemptKey(const ActiveQuery&                       query,
                                         const CacheKey&                          key,
                                         const TensileLite::ContractionProblemGemm& problem,
                                         int                                      numSolutions)
        {
            try
            {
                std::ostringstream attempt;
                attempt << query.target.device << '|' << key.directoryName() << '|'
                        << key.codeObjectVersion << '|' << query.workspace << '|' << numSolutions
                        << '|' << problemTypeKey(problem);
                for(auto size : problemSizes(problem))
                    attempt << 'x' << size;
                for(const auto& name : query.exclude)
                    attempt << "|x" << name;
                return attempt.str();
            }
            catch(const std::exception&)
            {
                return {};
            }
        }
    }

    TensileLite::SolutionVector<TensileLite::ContractionSolution>
        JitLibrary::solutionsFor(const TensileLite::ContractionProblemGemm& problem,
                                 const TensileLite::Hardware&               hardware,
                                 int                                        numSolutions,
                                 bool                                       generate)
    {
        using Solution = TensileLite::ContractionSolution;
        TensileLite::SolutionVector<Solution> result;
        auto&                                 query = activeQuery();
        if(!query.engaged || numSolutions <= 0 || problem.groupedGemm())
            return result;
        auto problemCopy = problem;
        problemCopy.setWorkspaceSize(query.workspace);
        const auto key          = CacheKey::make(query.target, query.backend, jitCodeObjectVersion);
        const auto attemptKey   = generationAttemptKey(query, key, problemCopy, numSolutions);
        bool       alreadyGenerated = false;
        if(!attemptKey.empty())
        {
            std::lock_guard<std::mutex> guard(m_attemptMutex);
            alreadyGenerated = m_attempted.count(attemptKey) != 0;
        }
        std::vector<int32_t> indices;
        query.failure = lookup(key,
                               query.target.device,
                               problemCopy,
                               hardware,
                               static_cast<size_t>(numSolutions),
                               query.exclude,
                               indices);
        const bool miss = indices.size() < static_cast<size_t>(numSolutions);
        if(generate && query.allowGeneration && miss && !alreadyGenerated && !attemptKey.empty()
           && query.generate && query.failure.ok())
        {
            {
                std::lock_guard<std::mutex> guard(m_attemptMutex);
                m_attempted.insert(attemptKey);
            }
            auto exclude = query.exclude;
            for(auto index : indices)
            {
                Status why;
                if(auto solution = solutionByIndex(query.target.device, hardware, index, why))
                    if(!excludedKernel(exclude, solution->kernelName))
                        exclude.push_back(solution->kernelName);
            }
            std::vector<int32_t> extra;
            query.generate(static_cast<size_t>(numSolutions) - indices.size(),
                           exclude,
                           extra,
                           query.failure);
            for(auto index : extra)
                if(std::find(indices.begin(), indices.end(), index) == indices.end())
                    indices.push_back(index);
            if(!indices.empty())
                query.failure = {};
        }
        if(indices.size() > static_cast<size_t>(numSolutions))
            indices.resize(static_cast<size_t>(numSolutions));
        for(auto index : indices)
        {
            Status why;
            if(auto solution = solutionByIndex(query.target.device, hardware, index, why))
                result.push_back(std::move(solution));
        }
        return result;
    }

    void JitLibrary::beginQuery(bool                     allowGeneration,
                                const DeviceTarget&      target,
                                const BackendInfo&       backend,
                                size_t                   workspace,
                                std::vector<std::string> excludeKernels,
                                GenerateMissing          generate)
    {
        auto& query           = activeQuery();
        query                 = {};
        query.engaged         = true;
        query.allowGeneration = allowGeneration;
        query.target          = target;
        query.backend         = backend;
        query.workspace       = workspace;
        query.exclude         = std::move(excludeKernels);
        query.generate        = std::move(generate);
    }

    void JitLibrary::endQuery()
    {
        activeQuery() = {};
    }

    Status JitLibrary::queryStatus() const
    {
        return activeQuery().failure;
    }

    std::shared_ptr<TensileLite::ContractionSolution>
        JitLibrary::getSolutionByIndex(TensileLite::ContractionProblemGemm const&,
                                       TensileLite::Hardware const& hardware,
                                       const int                    index) const
    {
        const auto& query = activeQuery();
        if(!query.engaged)
            return {};
        Status why;
        return const_cast<JitLibrary*>(this)->solutionByIndex(query.target.device, hardware, index, why);
    }

    std::shared_ptr<TensileLite::ContractionSolution>
        JitLibrary::findBestSolution(TensileLite::ContractionProblemGemm const& problem,
                                     TensileLite::Hardware const&               hardware,
                                     double*                                    fitness) const
    {
        auto found = findTopSolutions(problem, hardware, 1);
        if(fitness)
            *fitness = found.empty() ? 0.0 : 1.0;
        return found.empty() ? nullptr : found.front();
    }

    TensileLite::SolutionSet<TensileLite::ContractionSolution>
        JitLibrary::findAllSolutions(TensileLite::ContractionProblemGemm const& problem,
                                     TensileLite::Hardware const&               hardware,
                                     TensileLite::SolutionLibrarySearchType) const
    {
        auto found = const_cast<JitLibrary*>(this)->solutionsFor(problem, hardware, INT32_MAX, false);
        TensileLite::SolutionSet<TensileLite::ContractionSolution> result;
        result.insert(found.begin(), found.end());
        return result;
    }

    TensileLite::SolutionSet<TensileLite::ContractionSolution> JitLibrary::findAllSolutionsGroupedGemm(
        std::vector<TensileLite::ContractionProblemGemm> const&,
        TensileLite::Hardware const&,
        TensileLite::SolutionLibrarySearchType) const
    {
        return {};
    }

    std::string JitLibrary::type() const
    {
        return "Jit";
    }

    std::string JitLibrary::description() const
    {
        return "JIT solution library";
    }

    TensileLite::SolutionVector<TensileLite::ContractionSolution>
        JitLibrary::findTopSolutions(TensileLite::ContractionProblemGemm const& problem,
                                     TensileLite::Hardware const&               hardware,
                                     int                                        numSolutions) const
    {
        return const_cast<JitLibrary*>(this)->solutionsFor(problem, hardware, numSolutions, true);
    }

    TensileLite::SolutionVector<TensileLite::ContractionSolution> JitLibrary::findTopSolutionsGroupedGemm(
        std::vector<TensileLite::ContractionProblemGemm> const&,
        TensileLite::Hardware const&,
        int) const
    {
        return {};
    }
}
