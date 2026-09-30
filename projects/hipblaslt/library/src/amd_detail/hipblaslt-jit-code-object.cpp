// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-code-object.hpp"
#include "hipblaslt-jit-mode.hpp"
#include "rocblaslt_secure_env.hpp"

#include <amd_comgr/amd_comgr.h>
#include <hip/hip_runtime_api.h>

#include <atomic>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <new>
#include <set>
#include <sstream>
#include <system_error>
#include <utility>
#ifndef _WIN32
#include <dlfcn.h>
#include <sys/stat.h>
#endif

namespace hipblaslt_jit::code_object
{
    namespace
    {
        struct Failure
        {
            Status      status;
            std::string message;
        };

        [[noreturn]] void fail(Status status, std::string message)
        {
            throw Failure{status, std::move(message)};
        }

        std::string statusString(amd_comgr_status_t status)
        {
            const char* text = nullptr;
            if(amd_comgr_status_string(status, &text) == AMD_COMGR_STATUS_SUCCESS && text)
                return text;
            return "comgr status " + std::to_string(static_cast<int>(status));
        }

        void check(amd_comgr_status_t status,
                   const char*        what,
                   Status             onError = Status::ComgrError)
        {
            if(status != AMD_COMGR_STATUS_SUCCESS)
                fail(status == AMD_COMGR_STATUS_ERROR_INVALID_ARGUMENT ? Status::InvalidArgument
                                                                       : onError,
                     std::string(what) + ": " + statusString(status));
        }

        template <typename H, amd_comgr_status_t (*Destroy)(H)>
        class Handle
        {
        public:
            Handle() = default;
            ~Handle()
            {
                reset();
            }
            Handle(const Handle&)            = delete;
            Handle& operator=(const Handle&) = delete;
            Handle(Handle&& other) noexcept
                : value(std::exchange(other.value, H{}))
            {
            }
            Handle& operator=(Handle&& other) noexcept
            {
                if(this != &other)
                {
                    reset();
                    value = std::exchange(other.value, H{});
                }
                return *this;
            }
            H* out() noexcept
            {
                reset();
                return &value;
            }
            H get() const noexcept
            {
                return value;
            }

        private:
            void reset() noexcept
            {
                if(value.handle != 0)
                    Destroy(value);
                value = H{};
            }
            H value{};
        };

        using Data       = Handle<amd_comgr_data_t, amd_comgr_release_data>;
        using DataSet    = Handle<amd_comgr_data_set_t, amd_comgr_destroy_data_set>;
        using ActionInfo = Handle<amd_comgr_action_info_t, amd_comgr_destroy_action_info>;
        using Metadata   = Handle<amd_comgr_metadata_node_t, amd_comgr_destroy_metadata>;

        DataSet makeSet()
        {
            DataSet set;
            check(amd_comgr_create_data_set(set.out()), "amd_comgr_create_data_set");
            return set;
        }

        void add(DataSet&              set,
                 amd_comgr_data_kind_t kind,
                 const std::string&    name,
                 const char*           bytes,
                 size_t                size)
        {
            Data data;
            check(amd_comgr_create_data(kind, data.out()), "amd_comgr_create_data");
            check(amd_comgr_set_data(data.get(), size, bytes), "amd_comgr_set_data");
            check(amd_comgr_set_data_name(data.get(), name.c_str()), "amd_comgr_set_data_name");
            check(amd_comgr_data_set_add(set.get(), data.get()), "amd_comgr_data_set_add");
        }

        std::vector<char> bytesOf(amd_comgr_data_t data)
        {
            size_t size = 0;
            check(amd_comgr_get_data(data, &size, nullptr), "amd_comgr_get_data");
            std::vector<char> bytes(size);
            if(size)
                check(amd_comgr_get_data(data, &size, bytes.data()), "amd_comgr_get_data");
            bytes.resize(size);
            return bytes;
        }

        std::string nameOf(amd_comgr_data_t data)
        {
            size_t size = 0;
            check(amd_comgr_get_data_name(data, &size, nullptr), "amd_comgr_get_data_name");
            std::string name(size, '\0');
            if(size)
                check(amd_comgr_get_data_name(data, &size, name.data()),
                      "amd_comgr_get_data_name");
            while(!name.empty() && name.back() == '\0')
                name.pop_back();
            return name;
        }

        std::vector<std::pair<std::string, std::vector<char>>>
            collect(const DataSet& set, amd_comgr_data_kind_t kind)
        {
            size_t count = 0;
            check(amd_comgr_action_data_count(set.get(), kind, &count),
                  "amd_comgr_action_data_count");
            std::vector<std::pair<std::string, std::vector<char>>> items;
            for(size_t i = 0; i < count; ++i)
            {
                Data data;
                check(amd_comgr_action_data_get_data(set.get(), kind, i, data.out()),
                      "amd_comgr_action_data_get_data");
                items.emplace_back(nameOf(data.get()), bytesOf(data.get()));
            }
            return items;
        }

        ActionInfo makeAction(const std::string&              isaName,
                              const std::vector<std::string>& options,
                              bool                            hip = false)
        {
            ActionInfo info;
            check(amd_comgr_create_action_info(info.out()), "amd_comgr_create_action_info");
            check(amd_comgr_action_info_set_isa_name(info.get(), isaName.c_str()),
                  ("unsupported ISA name '" + isaName + "'").c_str());
            check(amd_comgr_action_info_set_logging(info.get(), true),
                  "amd_comgr_action_info_set_logging");
            if(hip)
            {
                check(amd_comgr_action_info_set_language(info.get(), AMD_COMGR_LANGUAGE_HIP),
                      "amd_comgr_action_info_set_language");
                check(amd_comgr_action_info_set_device_lib_linking(info.get(), true),
                      "amd_comgr_action_info_set_device_lib_linking");
            }
            std::vector<const char*> argv;
            argv.reserve(options.size());
            for(const auto& option : options)
                argv.push_back(option.c_str());
            check(amd_comgr_action_info_set_option_list(info.get(), argv.data(), argv.size()),
                  "amd_comgr_action_info_set_option_list");
            return info;
        }

        std::string join(const std::vector<std::string>& options)
        {
            std::string text;
            for(const auto& option : options)
                text += (text.empty() ? "" : " ") + option;
            return text;
        }

        // Runs one comgr action, appending its log/diagnostics whether or not it succeeds.
        DataSet run(amd_comgr_action_kind_t         kind,
                    const char*                     stage,
                    const ActionInfo&               info,
                    const DataSet&                  input,
                    const std::vector<std::string>& options,
                    std::string&                    log)
        {
            DataSet    output = makeSet();
            const auto status = amd_comgr_do_action(kind, info.get(), input.get(), output.get());
            std::string text;
            for(auto dataKind : {AMD_COMGR_DATA_KIND_LOG, AMD_COMGR_DATA_KIND_DIAGNOSTIC})
            {
                size_t count = 0;
                if(amd_comgr_action_data_count(output.get(), dataKind, &count)
                   != AMD_COMGR_STATUS_SUCCESS)
                    continue;
                for(const auto& item : collect(output, dataKind))
                    text.append(item.second.begin(), item.second.end());
            }
            while(!text.empty() && (text.back() == '\n' || text.back() == '\0'))
                text.pop_back();
            if(status != AMD_COMGR_STATUS_SUCCESS || !text.empty())
            {
                log += std::string("[") + stage + "] options: " + join(options) + "\n";
                if(!text.empty())
                    log += text + "\n";
            }
            if(status != AMD_COMGR_STATUS_SUCCESS)
                fail(Status::ComgrError,
                     std::string(stage) + " failed: " + statusString(status)
                         + (text.empty() ? " (comgr produced no log)" : ""));
            return output;
        }

        bool startsWith(const std::string& text, const char* prefix)
        {
            return text.rfind(prefix, 0) == 0;
        }

        void validateTarget(const Target& target)
        {
            if(target.gfx.size() < 4 || !startsWith(target.gfx, "gfx"))
                fail(Status::InvalidArgument, "target processor must look like gfxNNN, got '"
                                                  + target.gfx + "'");
            for(size_t i = 3; i < target.gfx.size(); ++i)
                if(!std::isalnum(static_cast<unsigned char>(target.gfx[i])))
                    fail(Status::InvalidArgument, "invalid target processor '" + target.gfx + "'");
            for(const auto& feature : target.features)
                if(feature.size() < 2 || (feature.back() != '+' && feature.back() != '-'))
                    fail(Status::InvalidArgument,
                         "target feature must end in '+' or '-', got '" + feature + "'");
            if(target.wavefrontSize != 0 && target.wavefrontSize != 32
               && target.wavefrontSize != 64)
                fail(Status::InvalidArgument, "wavefront size must be 0, 32 or 64");
        }

        void validateOptions(const Options& options)
        {
            if(options.codeObjectVersion < 4 || options.codeObjectVersion > 6)
                fail(Status::InvalidArgument,
                     "code object version must be 4, 5 or 6, got "
                         + std::to_string(options.codeObjectVersion));
            std::set<std::string> names;
            for(const auto& include : options.includes)
            {
                if(include.name.empty() || include.name.front() == '/'
                   || include.name.find("..") != std::string::npos)
                    fail(Status::InvalidArgument,
                         "include name must be a relative path without '..': '" + include.name
                             + "'");
                if(!names.insert(include.name).second)
                    fail(Status::InvalidArgument, "duplicate include name '" + include.name + "'");
            }
        }

        // comgr materializes data objects as files named after the data; use
        // index-prefixed, file-system-safe names and keep the caller's names separately.
        std::string dataName(size_t index, const std::string& name, const char* extension)
        {
            std::string safe;
            for(char c : name)
                safe += (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-')
                            ? c
                            : '_';
            if(safe.size() > 96)
                safe.resize(96);
            return std::to_string(index) + "-" + safe + extension;
        }

        template <typename Source>
        void validateSources(const std::vector<Source>& sources)
        {
            if(sources.empty())
                fail(Status::InvalidArgument, "no sources supplied");
            std::set<std::string> names;
            for(const auto& source : sources)
            {
                if(source.name.empty())
                    fail(Status::InvalidArgument, "source name must not be empty");
                if(!names.insert(source.name).second)
                    fail(Status::InvalidArgument, "duplicate source name '" + source.name + "'");
            }
        }

        // Deterministic per-source compilation unit id (FNV-1a over name and text).
        std::string contentId(const HipSource& source)
        {
            std::uint64_t hash = 14695981039346656037ull;
            auto          mix  = [&](const std::string& bytes) {
                for(unsigned char c : bytes)
                    hash = (hash ^ c) * 1099511628211ull;
                hash = (hash ^ 0xff) * 1099511628211ull;
            };
            mix(source.name);
            mix(source.text);
            char text[17];
            std::snprintf(text, sizeof(text), "%016llx", static_cast<unsigned long long>(hash));
            return text;
        }

        std::string waveOption(const Target& target)
        {
            return target.resolvedWavefrontSize() == 64 ? "-mwavefrontsize64"
                                                        : "-mno-wavefrontsize64";
        }

        void addIncludes(DataSet& set, const Options& options)
        {
            for(const auto& include : options.includes)
                add(set,
                    AMD_COMGR_DATA_KIND_INCLUDE,
                    include.name,
                    include.text.data(),
                    include.text.size());
        }

        std::vector<Relocatable> relocatablesFrom(const DataSet&                  output,
                                                  const std::vector<std::string>& names)
        {
            auto items = collect(output, AMD_COMGR_DATA_KIND_RELOCATABLE);
            if(items.size() != names.size())
                fail(Status::NoOutput,
                     "expected " + std::to_string(names.size()) + " relocatable objects, comgr produced "
                         + std::to_string(items.size()));
            std::vector<Relocatable> objects;
            for(size_t i = 0; i < items.size(); ++i)
            {
                if(items[i].second.empty())
                    fail(Status::NoOutput, "empty relocatable for '" + names[i] + "'");
                objects.push_back({names[i], std::move(items[i].second)});
            }
            return objects;
        }

        // Replaces the target ID after "amdgcn-amd-amdhsa--" in the target directive
        // and metadata, up to the closing quote or the end of the token.
        std::string retarget(std::string text, const std::string& targetId)
        {
            const std::string triple = "amdgcn-amd-amdhsa--";
            for(const char* anchor : {".amdgcn_target", "amdhsa.target:"})
            {
                for(size_t at = text.find(anchor); at != std::string::npos;
                    at        = text.find(anchor, at + 1))
                {
                    const auto lineEnd = text.find('\n', at);
                    const auto start   = text.find(triple + "gfx", at);
                    if(start == std::string::npos || start > lineEnd)
                        continue;
                    const auto begin = start + triple.size();
                    auto       end   = begin;
                    while(end < text.size() && text[end] != '"' && text[end] != '\''
                          && !std::isspace(static_cast<unsigned char>(text[end])))
                        ++end;
                    text.replace(begin, end - begin, targetId);
                }
            }
            return text;
        }

        RelocatableResult assembleImpl(const std::vector<AssemblySource>& sources,
                                       const Target&                      target,
                                       const Options&                     options,
                                       std::string&                       log)
        {
            validateTarget(target);
            validateOptions(options);
            validateSources(sources);

            std::vector<std::string> flags
                = {"-mcode-object-version=" + std::to_string(options.codeObjectVersion),
                   waveOption(target),
                   "-Wno-unused-command-line-argument"};
            if(options.architectureDefaults
               && (startsWith(target.gfx, "gfx11") || startsWith(target.gfx, "gfx12")))
                flags.insert(flags.end(),
                             {"-Xclangas", "-target-feature", "-Xclangas", "+real-true16"});
            for(const auto& dir : options.includeDirectories)
                flags.push_back("-I" + dir);
            flags.insert(flags.end(), options.assemblerFlags.begin(), options.assemblerFlags.end());

            auto                     input = makeSet();
            std::vector<std::string> names;
            for(size_t i = 0; i < sources.size(); ++i)
            {
                const auto& text = options.retargetAssembly
                                       ? retarget(sources[i].text, target.targetId())
                                       : sources[i].text;
                add(input,
                    AMD_COMGR_DATA_KIND_SOURCE,
                    dataName(i, sources[i].name, ".s"),
                    text.data(),
                    text.size());
                names.push_back(sources[i].name);
            }
            addIncludes(input, options);

            const auto info   = makeAction(target.isaName(), flags);
            const auto output = run(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE,
                                    "assemble",
                                    info,
                                    input,
                                    flags,
                                    log);
            return {Status::Success, {}, relocatablesFrom(output, names)};
        }

        RelocatableResult compileHipImpl(const std::vector<HipSource>& sources,
                                         const Target&                 target,
                                         const Options&                options,
                                         std::string&                  log)
        {
            validateTarget(target);
            validateOptions(options);
            validateSources(sources);

            const auto               version = std::to_string(options.codeObjectVersion);
            std::vector<std::string> compile
                = {"-O3", "-std=c++17", "-mcode-object-version=" + version, waveOption(target)};
            if(!options.rocmPath.empty())
                compile.push_back("--rocm-path=" + options.rocmPath);
            if(options.includeHipRuntimeWrapper)
                compile.insert(compile.end(), {"-include", "__clang_hip_runtime_wrapper.h"});
            for(const auto& dir : options.includeDirectories)
                compile.push_back("-I" + dir);

            std::vector<std::string> codegen = {"-O3",
                                                "-Xclang",
                                                "-disable-llvm-optzns",
                                                "-mcode-object-version=" + version,
                                                waveOption(target)};
            codegen.insert(codegen.end(), options.codegenFlags.begin(), options.codegenFlags.end());

            const auto        codegenInfo = makeAction(target.isaName(), codegen);
            RelocatableResult result{Status::Success, {}, {}};
            // One action per source: COMPILE_SOURCE_TO_RELOCATABLE accepts a single
            // source, and every translation unit needs its own -cuid or the
            // __hip_cuid_<id> symbols collide when the objects are linked together.
            for(size_t i = 0; i < sources.size(); ++i)
            {
                auto options_i = compile;
                options_i.push_back("-cuid=" + contentId(sources[i]));
                options_i.insert(
                    options_i.end(), options.compilerFlags.begin(), options.compilerFlags.end());
                const auto info  = makeAction(target.isaName(), options_i, true);
                auto       input = makeSet();
                add(input,
                    AMD_COMGR_DATA_KIND_SOURCE,
                    dataName(i, sources[i].name, ".hip"),
                    sources[i].text.data(),
                    sources[i].text.size());
                addIncludes(input, options);
                if(options.hipPipeline == HipPipeline::SourceToRelocatable)
                {
                    const auto output = run(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE,
                                            "compile HIP to relocatable",
                                            info,
                                            input,
                                            options_i,
                                            log);
                    result.objects.push_back(
                        std::move(relocatablesFrom(output, {sources[i].name}).front()));
                    continue;
                }
                const auto bitcode = run(AMD_COMGR_ACTION_COMPILE_SOURCE_WITH_DEVICE_LIBS_TO_BC,
                                         "compile HIP to bitcode",
                                         info,
                                         input,
                                         options_i,
                                         log);
                const auto output  = run(AMD_COMGR_ACTION_CODEGEN_BC_TO_RELOCATABLE,
                                        "codegen bitcode to relocatable",
                                        codegenInfo,
                                        bitcode,
                                        codegen,
                                        log);
                result.objects.push_back(
                    std::move(relocatablesFrom(output, {sources[i].name}).front()));
            }
            return result;
        }

        Result linkImpl(const std::vector<Relocatable>& objects,
                        const Target&                   target,
                        const Options&                  options,
                        std::string&                    log)
        {
            validateTarget(target);
            validateOptions(options);
            if(objects.empty())
                fail(Status::InvalidArgument, "no relocatable objects to link");

            auto input = makeSet();
            for(size_t i = 0; i < objects.size(); ++i)
            {
                if(objects[i].bytes.empty())
                    fail(Status::InvalidArgument,
                         "relocatable '" + objects[i].name + "' is empty");
                add(input,
                    AMD_COMGR_DATA_KIND_RELOCATABLE,
                    dataName(i, objects[i].name, ".o"),
                    objects[i].bytes.data(),
                    objects[i].bytes.size());
            }
            const auto info   = makeAction(target.isaName(), options.linkerFlags);
            const auto output = run(AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE,
                                    "link",
                                    info,
                                    input,
                                    options.linkerFlags,
                                    log);
            auto executables = collect(output, AMD_COMGR_DATA_KIND_EXECUTABLE);
            if(executables.size() != 1 || executables.front().second.empty())
                fail(Status::NoOutput,
                     "link produced " + std::to_string(executables.size())
                         + " executables; expected one non-empty executable");
            return {Status::Success, {}, std::move(executables.front().second)};
        }

        template <typename R, typename F>
        R guarded(F&& body) noexcept
        {
            std::string log;
            try
            {
                R result   = body(log);
                result.log = std::move(log) + result.log;
                return result;
            }
            catch(Failure& failure)
            {
                R result;
                result.status = failure.status;
                try
                {
                    result.log = std::move(log) + "error: " + failure.message + "\n";
                }
                catch(...)
                {
                }
                return result;
            }
            catch(const std::bad_alloc&)
            {
                R result;
                result.status = Status::InternalError;
                return result;
            }
            catch(const std::exception& error)
            {
                R result;
                result.status = Status::InternalError;
                try
                {
                    result.log = std::move(log) + "internal error: " + error.what() + "\n";
                }
                catch(...)
                {
                }
                return result;
            }
            catch(...)
            {
                R result;
                result.status = Status::InternalError;
                return result;
            }
        }

        std::string metadataString(amd_comgr_metadata_node_t node, const char* key)
        {
            amd_comgr_metadata_kind_t kind;
            check(amd_comgr_get_metadata_kind(node, &kind), "amd_comgr_get_metadata_kind",
                  Status::MetadataError);
            if(kind != AMD_COMGR_METADATA_KIND_STRING)
                fail(Status::MetadataError, std::string("metadata '") + key + "' is not a scalar");
            size_t size = 0;
            check(amd_comgr_get_metadata_string(node, &size, nullptr),
                  "amd_comgr_get_metadata_string", Status::MetadataError);
            std::string value(size, '\0');
            check(amd_comgr_get_metadata_string(node, &size, value.data()),
                  "amd_comgr_get_metadata_string", Status::MetadataError);
            while(!value.empty() && value.back() == '\0')
                value.pop_back();
            return value;
        }

        bool lookup(amd_comgr_metadata_node_t map, const char* key, Metadata& value)
        {
            return amd_comgr_metadata_lookup(map, key, value.out()) == AMD_COMGR_STATUS_SUCCESS;
        }

        std::string stringField(amd_comgr_metadata_node_t map, const char* key, bool required)
        {
            Metadata value;
            if(!lookup(map, key, value))
            {
                if(required)
                    fail(Status::MetadataError, std::string("missing metadata key '") + key + "'");
                return {};
            }
            return metadataString(value.get(), key);
        }

        std::uint64_t numberField(amd_comgr_metadata_node_t map, const char* key, bool required)
        {
            const auto text = stringField(map, key, required);
            if(text.empty())
                return 0;
            errno     = 0;
            char* end = nullptr;
            const auto value = std::strtoull(text.c_str(), &end, 0);
            if(errno != 0 || end == text.c_str() || *end != '\0')
                fail(Status::MetadataError,
                     std::string("metadata '") + key + "' is not an unsigned integer: '" + text
                         + "'");
            return value;
        }

        template <typename F>
        void forEach(amd_comgr_metadata_node_t list, const char* key, F&& visit)
        {
            size_t count = 0;
            check(amd_comgr_get_metadata_list_size(list, &count),
                  (std::string("metadata '") + key + "' is not a list").c_str(),
                  Status::MetadataError);
            for(size_t i = 0; i < count; ++i)
            {
                Metadata element;
                check(amd_comgr_index_list_metadata(list, i, element.out()),
                      "amd_comgr_index_list_metadata", Status::MetadataError);
                visit(element.get());
            }
        }

        std::uint16_t le16(const unsigned char* p)
        {
            return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
        }

        int elfCodeObjectVersion(const unsigned char* elf, size_t size)
        {
            static const char bundleMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
            if(size >= 4 && std::memcmp(elf, "CCOB", 4) == 0)
                fail(Status::MetadataError,
                     "input is a compressed offload bundle; extract the code object first");
            if(size >= sizeof(bundleMagic) - 1
               && std::memcmp(elf, bundleMagic, sizeof(bundleMagic) - 1) == 0)
                fail(Status::MetadataError,
                     "input is a clang offload bundle; extract the code object first");
            if(size < 64 || std::memcmp(elf, "\x7f" "ELF", 4) != 0)
                fail(Status::MetadataError, "input is not an ELF file");
            if(elf[4] != 2 || elf[5] != 1)
                fail(Status::MetadataError, "input is not a little-endian ELF64 file");
            constexpr std::uint16_t amdgpuMachine = 224; // EM_AMDGPU
            if(le16(elf + 18) != amdgpuMachine)
                fail(Status::MetadataError, "input is not an AMDGPU ELF file");
            constexpr unsigned char amdhsaAbi = 64; // ELFOSABI_AMDGPU_HSA
            if(elf[7] != amdhsaAbi)
                fail(Status::MetadataError, "input is not an AMDHSA code object");
            return elf[8] + 2;
        }

        MetadataResult readMetadataImpl(const void* bytes, size_t size, std::string&)
        {
            if(!bytes)
                fail(Status::InvalidArgument, "null code object");
            MetadataResult result;
            auto&          md = result.metadata;
            md.codeObjectVersion
                = elfCodeObjectVersion(static_cast<const unsigned char*>(bytes), size);
            const bool relocatable = le16(static_cast<const unsigned char*>(bytes) + 16) == 1;

            Data data;
            check(amd_comgr_create_data(relocatable ? AMD_COMGR_DATA_KIND_RELOCATABLE
                                                    : AMD_COMGR_DATA_KIND_EXECUTABLE,
                                        data.out()),
                  "amd_comgr_create_data");
            check(amd_comgr_set_data(data.get(), size, static_cast<const char*>(bytes)),
                  "amd_comgr_set_data");
            check(amd_comgr_set_data_name(data.get(), "code-object"), "amd_comgr_set_data_name");

            size_t isaSize = 0;
            check(amd_comgr_get_data_isa_name(data.get(), &isaSize, nullptr),
                  "amd_comgr_get_data_isa_name", Status::MetadataError);
            md.isaName.assign(isaSize, '\0');
            check(amd_comgr_get_data_isa_name(data.get(), &isaSize, md.isaName.data()),
                  "amd_comgr_get_data_isa_name", Status::MetadataError);
            while(!md.isaName.empty() && md.isaName.back() == '\0')
                md.isaName.pop_back();

            Metadata root;
            check(amd_comgr_get_data_metadata(data.get(), root.out()),
                  "amd_comgr_get_data_metadata", Status::MetadataError);
            md.target = stringField(root.get(), "amdhsa.target", false);

            Metadata kernels;
            if(!lookup(root.get(), "amdhsa.kernels", kernels))
                fail(Status::MetadataError, "missing metadata key 'amdhsa.kernels'");
            forEach(kernels.get(), "amdhsa.kernels", [&](amd_comgr_metadata_node_t node) {
                KernelMetadata k;
                k.name                    = stringField(node, ".name", true);
                k.symbol                  = stringField(node, ".symbol", true);
                k.kernargSegmentSize      = numberField(node, ".kernarg_segment_size", true);
                k.kernargSegmentAlign     = numberField(node, ".kernarg_segment_align", false);
                k.groupSegmentFixedSize   = numberField(node, ".group_segment_fixed_size", true);
                k.privateSegmentFixedSize = numberField(node, ".private_segment_fixed_size", true);
                k.vgprCount      = static_cast<std::uint32_t>(numberField(node, ".vgpr_count", true));
                k.agprCount      = static_cast<std::uint32_t>(numberField(node, ".agpr_count", false));
                k.sgprCount      = static_cast<std::uint32_t>(numberField(node, ".sgpr_count", true));
                k.vgprSpillCount
                    = static_cast<std::uint32_t>(numberField(node, ".vgpr_spill_count", false));
                k.sgprSpillCount
                    = static_cast<std::uint32_t>(numberField(node, ".sgpr_spill_count", false));
                k.wavefrontSize
                    = static_cast<std::uint32_t>(numberField(node, ".wavefront_size", true));
                k.maxFlatWorkgroupSize = static_cast<std::uint32_t>(
                    numberField(node, ".max_flat_workgroup_size", false));
                Metadata args;
                if(lookup(node, ".args", args))
                    forEach(args.get(), ".args", [&](amd_comgr_metadata_node_t arg) {
                        KernelArgument a;
                        a.name      = stringField(arg, ".name", false);
                        a.valueKind = stringField(arg, ".value_kind", true);
                        a.offset    = numberField(arg, ".offset", true);
                        a.size      = numberField(arg, ".size", true);
                        k.arguments.push_back(std::move(a));
                    });
                md.kernels.push_back(std::move(k));
            });
            result.status = Status::Success;
            return result;
        }

#ifndef _WIN32
        std::filesystem::path libraryOf(const void* symbol)
        {
            Dl_info info{};
            if(dladdr(symbol, &info) == 0 || !info.dli_fname || !*info.dli_fname)
                return {};
            std::error_code error;
            auto            path = std::filesystem::canonical(info.dli_fname, error);
            return error ? std::filesystem::path(info.dli_fname) : path;
        }
#endif
    }

    const char* toString(Status status) noexcept
    {
        switch(status)
        {
        case Status::Success:
            return "success";
        case Status::InvalidArgument:
            return "invalid argument";
        case Status::ComgrError:
            return "comgr error";
        case Status::NoOutput:
            return "no output";
        case Status::MetadataError:
            return "metadata error";
        case Status::InternalError:
            return "internal error";
        }
        return "unknown";
    }

    Target Target::fromTargetId(const std::string& targetId, int wavefrontSize)
    {
        Target             target;
        std::istringstream parts(targetId);
        std::string        part;
        std::getline(parts, target.gfx, ':');
        while(std::getline(parts, part, ':'))
            if(!part.empty())
                target.features.push_back(part);
        target.wavefrontSize = wavefrontSize;
        return target;
    }

    std::string Target::targetId() const
    {
        std::string id = gfx;
        for(const auto& feature : features)
            id += ":" + feature;
        return id;
    }

    std::string Target::isaName() const
    {
        return "amdgcn-amd-amdhsa--" + targetId();
    }

    int Target::resolvedWavefrontSize() const
    {
        if(wavefrontSize != 0)
            return wavefrontSize;
        return (gfx.size() == 6 && gfx.rfind("gfx9", 0) == 0) ? 64 : 32;
    }

    RelocatableResult assembleRelocatables(const std::vector<AssemblySource>& sources,
                                           const Target&                      target,
                                           const Options&                     options) noexcept
    {
        return guarded<RelocatableResult>(
            [&](std::string& log) { return assembleImpl(sources, target, options, log); });
    }

    RelocatableResult compileHipRelocatables(const std::vector<HipSource>& sources,
                                             const Target&                 target,
                                             const Options&                options) noexcept
    {
        return guarded<RelocatableResult>(
            [&](std::string& log) { return compileHipImpl(sources, target, options, log); });
    }

    Result link(const std::vector<Relocatable>& objects,
                const Target&                   target,
                const Options&                  options) noexcept
    {
        return guarded<Result>(
            [&](std::string& log) { return linkImpl(objects, target, options, log); });
    }

    Result assemble(const std::vector<AssemblySource>& sources,
                    const Target&                      target,
                    const Options&                     options) noexcept
    {
        return guarded<Result>([&](std::string& log) {
            auto objects = assembleImpl(sources, target, options, log);
            return linkImpl(objects.objects, target, options, log);
        });
    }

    Result compileHip(const std::vector<HipSource>& sources,
                      const Target&                 target,
                      const Options&                options) noexcept
    {
        return guarded<Result>([&](std::string& log) {
            auto objects = compileHipImpl(sources, target, options, log);
            return linkImpl(objects.objects, target, options, log);
        });
    }

    MetadataResult readMetadata(const void* elf, std::size_t size) noexcept
    {
        return guarded<MetadataResult>(
            [&](std::string& log) { return readMetadataImpl(elf, size, log); });
    }

    MetadataResult readMetadata(const std::vector<char>& elf) noexcept
    {
        return readMetadata(elf.data(), elf.size());
    }

    std::string comgrVersion() noexcept
    {
        try
        {
            size_t major = 0, minor = 0;
            amd_comgr_get_version(&major, &minor);
            return std::to_string(major) + "." + std::to_string(minor);
        }
        catch(...)
        {
            return {};
        }
    }

    std::string comgrIdentity() noexcept
    {
        try
        {
            auto identity = comgrVersion();
#ifndef _WIN32
            const auto  path = libraryOf(reinterpret_cast<const void*>(&amd_comgr_get_version));
            struct stat info{};
            if(!path.empty() && ::stat(path.c_str(), &info) == 0)
                identity += ":" + path.string() + ":" + std::to_string(info.st_size) + ":"
                            + std::to_string(static_cast<long long>(info.st_mtim.tv_sec)
                                                 * 1000000000LL
                                             + info.st_mtim.tv_nsec);
#endif
            return identity;
        }
        catch(...)
        {
            return {};
        }
    }

    std::string rocmPath() noexcept
    {
        try
        {
            if(const char* hip = rocblaslt_secure_getenv("HIP_PATH"); hip && *hip)
                return hip;
#ifndef _WIN32
            const auto runtime = libraryOf(reinterpret_cast<const void*>(&hipModuleLoadData));
            if(runtime.has_parent_path() && runtime.parent_path().has_parent_path())
                return runtime.parent_path().parent_path().string();
#endif
            return {};
        }
        catch(...)
        {
            return {};
        }
    }

    const char* toString(ComgrCache setting) noexcept
    {
        switch(setting)
        {
        case ComgrCache::Default:
            return "comgr's default";
        case ComgrCache::UserValue:
            return "the user's AMD_COMGR_CACHE";
        case ComgrCache::Disabled:
            return "disabled at library load";
        case ComgrCache::DisabledLate:
            return "disabled after library load";
        }
        return "unknown";
    }

    ComgrCache comgrCachePolicy(const char* jitMode, const char* comgrCache) noexcept
    {
        Mode mode;
        if(!parseMode(jitMode, mode) || mode == Mode::Off)
            return ComgrCache::Default;
        return comgrCache ? ComgrCache::UserValue : ComgrCache::Disabled;
    }

    namespace
    {
        std::atomic<ComgrCache> setByHipblaslt{ComgrCache::Default};

        ComgrCache applyComgrCachePolicy(ComgrCache disabled) noexcept
        {
            if(const auto set = setByHipblaslt.load(); set != ComgrCache::Default)
                return set;
            const auto policy = comgrCachePolicy(rocblaslt_secure_getenv("HIPBLASLT_JIT"),
                                                 std::getenv("AMD_COMGR_CACHE"));
            if(policy != ComgrCache::Disabled)
                return policy;
#ifdef _WIN32
            if(_putenv_s("AMD_COMGR_CACHE", "0") != 0)
#else
            if(setenv("AMD_COMGR_CACHE", "0", 0) != 0)
#endif
                return ComgrCache::Default;
            setByHipblaslt = disabled;
            return disabled;
        }

        [[maybe_unused]] const ComgrCache atLoad = applyComgrCachePolicy(ComgrCache::Disabled);
    }

    ComgrCache prepareProcessEnvironment() noexcept
    {
        return applyComgrCachePolicy(ComgrCache::DisabledLate);
    }
}
