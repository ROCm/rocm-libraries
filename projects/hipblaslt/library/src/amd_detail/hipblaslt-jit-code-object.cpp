// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-code-object.hpp"
#include "rocblaslt_secure_env.hpp"

#include <amd_comgr/amd_comgr.h>
#include <hip/hip_runtime_api.h>

#include <cctype>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <new>
#include <set>
#include <system_error>
#include <utility>
#ifndef _WIN32
#include <dlfcn.h>
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
                 const void*           bytes,
                 size_t                size)
        {
            Data data;
            check(amd_comgr_create_data(kind, data.out()), "amd_comgr_create_data");
            check(amd_comgr_set_data(data.get(), size, static_cast<const char*>(bytes)),
                  "amd_comgr_set_data");
            check(amd_comgr_set_data_name(data.get(), name.c_str()), "amd_comgr_set_data_name");
            check(amd_comgr_data_set_add(set.get(), data.get()), "amd_comgr_data_set_add");
        }

        std::vector<uint8_t> bytesOf(amd_comgr_data_t data)
        {
            size_t size = 0;
            check(amd_comgr_get_data(data, &size, nullptr), "amd_comgr_get_data");
            std::vector<uint8_t> bytes(size);
            if(size)
                check(amd_comgr_get_data(data, &size, reinterpret_cast<char*>(bytes.data())),
                      "amd_comgr_get_data");
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

        std::vector<std::pair<std::string, std::vector<uint8_t>>>
            collect(const DataSet& set, amd_comgr_data_kind_t kind)
        {
            size_t count = 0;
            check(amd_comgr_action_data_count(set.get(), kind, &count),
                  "amd_comgr_action_data_count");
            std::vector<std::pair<std::string, std::vector<uint8_t>>> items;
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

        void validate(const Options& options, const std::vector<hipblaslt_jit::Artifact>& includes)
        {
            if(options.codeObjectVersion < 4 || options.codeObjectVersion > 6)
                fail(Status::InvalidArgument,
                     "code object version must be 4, 5 or 6, got "
                         + std::to_string(options.codeObjectVersion));
            std::set<std::string> names;
            for(const auto& include : includes)
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

        // A -cuid unique within one link, whose sources have distinct names.
        std::string contentId(const std::string& name)
        {
            std::string id;
            for(unsigned char c : name)
                id += (std::isalnum(c) || c == '_') ? static_cast<char>(c) : '_';
            return id.empty() ? std::string("unit") : id;
        }

        // wavefrontSize, or the processor default when it is 0.
        std::string waveOption(const Target& target, int wavefrontSize)
        {
            if(wavefrontSize == 0)
                wavefrontSize = target.processor < Processor::gfx1010 ? 64 : 32;
            return wavefrontSize == 64 ? "-mwavefrontsize64" : "-mno-wavefrontsize64";
        }

        void addIncludes(DataSet& set, const std::vector<hipblaslt_jit::Artifact>& includes)
        {
            for(const auto& include : includes)
                add(set,
                    AMD_COMGR_DATA_KIND_INCLUDE,
                    include.name,
                    reinterpret_cast<const char*>(include.bytes.data()),
                    include.bytes.size());
        }

        std::vector<uint8_t> relocatableFrom(const DataSet& output, const std::string& name)
        {
            auto items = collect(output, AMD_COMGR_DATA_KIND_RELOCATABLE);
            if(items.size() != 1 || items.front().second.empty())
                fail(Status::NoOutput, "comgr produced no relocatable for '" + name + "'");
            return std::move(items.front().second);
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

        std::vector<uint8_t> assemble(const hipblaslt_jit::SourceCodeBundle& bundle,
                                      const Target&                          target,
                                      const Options&                         options,
                                      std::string&                           log)
        {
            const auto&                    source = bundle.mainKernel;
            const std::vector<std::string> flags
                = {"-mcode-object-version=" + std::to_string(options.codeObjectVersion),
                   waveOption(target, options.wavefrontSize),
                   "-Wno-unused-command-line-argument"};
            const auto text = retarget(std::string(source.bytes.begin(), source.bytes.end()),
                                       target.targetId());

            auto input = makeSet();
            add(input,
                AMD_COMGR_DATA_KIND_SOURCE,
                dataName(0, source.name, ".s"),
                text.data(),
                text.size());
            addIncludes(input, bundle.includes);
            const auto info   = makeAction(target.isaName(), flags);
            const auto output = run(AMD_COMGR_ACTION_ASSEMBLE_SOURCE_TO_RELOCATABLE,
                                    "assemble",
                                    info,
                                    input,
                                    flags,
                                    log);
            return relocatableFrom(output, source.name);
        }

        std::vector<uint8_t> compileHip(const hipblaslt_jit::SourceCodeBundle& bundle,
                                        const Target&                          target,
                                        const Options&                         options,
                                        std::string&                           log)
        {
            const auto&              source  = bundle.mainKernel;
            const auto               version = std::to_string(options.codeObjectVersion);
            std::vector<std::string> flags
                = {"-O3", "-std=c++17", "-mcode-object-version=" + version, waveOption(target, 0)};
            // Without --rocm-path comgr finds the HIP headers only through HIP_PATH.
            if(const auto rocm = rocmPath(); !rocm.empty())
                flags.push_back("--rocm-path=" + rocm);
            // comgr omits the HIP runtime wrapper that the clang driver normally injects.
            flags.insert(flags.end(), {"-include", "__clang_hip_runtime_wrapper.h"});
            // Every translation unit of a link needs its own -cuid, or the
            // __hip_cuid_<id> symbols collide.
            flags.push_back("-cuid=" + contentId(source.name));
            flags.insert(flags.end(), bundle.hipFlags.begin(), bundle.hipFlags.end());

            const auto        info  = makeAction(target.isaName(), flags, true);
            auto              input = makeSet();
            const std::string text(source.bytes.begin(), source.bytes.end());
            add(input,
                AMD_COMGR_DATA_KIND_SOURCE,
                dataName(0, source.name, ".hip"),
                text.data(),
                text.size());
            addIncludes(input, bundle.includes);
            const auto output = run(AMD_COMGR_ACTION_COMPILE_SOURCE_TO_RELOCATABLE,
                                    "compile HIP to relocatable",
                                    info,
                                    input,
                                    flags,
                                    log);
            return relocatableFrom(output, source.name);
        }

        template <typename R, typename F>
        R guarded(F&& body) noexcept
        {
            R           result;
            std::string log;
            try
            {
                try
                {
                    result     = body(log);
                    result.log = std::move(log) + result.log;
                }
                catch(Failure& failure)
                {
                    result.status = failure.status;
                    result.log    = std::move(log) + "error: " + failure.message + "\n";
                }
                catch(const std::exception& error)
                {
                    result.status = Status::InternalError;
                    result.log    = std::move(log) + "internal error: " + error.what() + "\n";
                }
            }
            catch(...)
            {
                result.status = Status::InternalError;
            }
            return result;
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

        std::string stringField(amd_comgr_metadata_node_t map, const char* key)
        {
            Metadata value;
            if(!lookup(map, key, value))
                fail(Status::MetadataError, std::string("missing metadata key '") + key + "'");
            return metadataString(value.get(), key);
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

        // A little-endian field of the size of T at offset, which must lie in the object.
        template <typename T>
        T field(const std::vector<uint8_t>& elf, std::uint64_t offset)
        {
            if(offset > elf.size() || elf.size() - offset < sizeof(T))
                fail(Status::MetadataError, "the code object is truncated");
            T value = 0;
            for(size_t i = 0; i < sizeof(T); ++i)
                value |= static_cast<T>(static_cast<T>(elf[offset + i]) << (8 * i));
            return value;
        }

        int elfCodeObjectVersion(const unsigned char* elf, size_t size)
        {
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

        ObjectCheck readMetadataImpl(const std::vector<uint8_t>& elf, std::string&)
        {
            ObjectCheck result;
            result.codeObjectVersion = elfCodeObjectVersion(elf.data(), elf.size());
            const bool relocatable   = le16(elf.data() + 16) == 1;

            Data data;
            check(amd_comgr_create_data(relocatable ? AMD_COMGR_DATA_KIND_RELOCATABLE
                                                    : AMD_COMGR_DATA_KIND_EXECUTABLE,
                                        data.out()),
                  "amd_comgr_create_data");
            check(amd_comgr_set_data(
                      data.get(), elf.size(), reinterpret_cast<const char*>(elf.data())),
                  "amd_comgr_set_data");
            check(amd_comgr_set_data_name(data.get(), "code-object"), "amd_comgr_set_data_name");

            size_t isaSize = 0;
            check(amd_comgr_get_data_isa_name(data.get(), &isaSize, nullptr),
                  "amd_comgr_get_data_isa_name", Status::MetadataError);
            result.isaName.assign(isaSize, '\0');
            check(amd_comgr_get_data_isa_name(data.get(), &isaSize, result.isaName.data()),
                  "amd_comgr_get_data_isa_name", Status::MetadataError);
            while(!result.isaName.empty() && result.isaName.back() == '\0')
                result.isaName.pop_back();

            Metadata root;
            check(amd_comgr_get_data_metadata(data.get(), root.out()),
                  "amd_comgr_get_data_metadata", Status::MetadataError);

            Metadata kernels;
            if(!lookup(root.get(), "amdhsa.kernels", kernels))
                fail(Status::MetadataError, "missing metadata key 'amdhsa.kernels'");
            forEach(kernels.get(), "amdhsa.kernels", [&](amd_comgr_metadata_node_t node) {
                result.kernelNames.push_back(stringField(node, ".name"));
            });
            result.status = Status::Success;
            return result;
        }

        MetadataNote metadataNoteImpl(const std::vector<uint8_t>& elf, std::string&)
        {
            elfCodeObjectVersion(elf.data(), elf.size());
            constexpr std::uint32_t noteSection = 7; // SHT_NOTE
            constexpr std::uint32_t amdgpuNote  = 32; // NT_AMDGPU_METADATA
            const auto              headers     = field<std::uint64_t>(elf, 0x28);
            const auto              headerSize  = field<std::uint16_t>(elf, 0x3a);
            const auto              count       = field<std::uint16_t>(elf, 0x3c);
            const auto              padded = [](std::uint64_t size) { return (size + 3) & ~3ull; };
            for(std::uint16_t i = 0; i < count; ++i)
            {
                const auto header = headers + std::uint64_t{i} * headerSize;
                if(field<std::uint32_t>(elf, header + 4) != noteSection)
                    continue;
                auto       at  = field<std::uint64_t>(elf, header + 0x18);
                const auto end = at + field<std::uint64_t>(elf, header + 0x20);
                if(end < at || end > elf.size())
                    fail(Status::MetadataError, "a note section lies outside the code object");
                while(end - at >= 12)
                {
                    const auto nameSize = field<std::uint32_t>(elf, at);
                    const auto descSize = field<std::uint32_t>(elf, at + 4);
                    const auto type     = field<std::uint32_t>(elf, at + 8);
                    const auto name     = at + 12;
                    const auto desc     = name + padded(nameSize);
                    if(desc > end || end - desc < descSize)
                        fail(Status::MetadataError, "a note lies outside its section");
                    if(type == amdgpuNote && nameSize == 7
                       && std::memcmp(elf.data() + name, "AMDGPU", 7) == 0)
                    {
                        MetadataNote result;
                        result.bytes.assign(elf.begin() + desc, elf.begin() + desc + descSize);
                        result.status = Status::Success;
                        return result;
                    }
                    at = desc + padded(descSize);
                }
            }
            fail(Status::MetadataError, "the code object has no AMDGPU metadata note");
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

    Result compileRelocatable(const hipblaslt_jit::SourceCodeBundle& bundle,
                              const Target&                          target,
                              const Options&                         options) noexcept
    {
        return guarded<Result>([&](std::string& log) {
            using Kind         = hipblaslt_jit::Artifact::Kind;
            const auto& kernel = bundle.mainKernel;
            if(kernel.name.empty() || kernel.bytes.empty())
                fail(Status::InvalidArgument, "no sources supplied");
            validate(options, bundle.includes);
            Result result{Status::Success, {}, {kernel.name, Kind::Relocatable, {}}};
            if(kernel.kind == Kind::Assembly)
                result.artifact.bytes = assemble(bundle, target, options, log);
            else if(kernel.kind == Kind::Hip)
                result.artifact.bytes = compileHip(bundle, target, options, log);
            else
                fail(Status::InvalidArgument, "the main kernel is not assembly or HIP source");
            return result;
        });
    }

    Result linkRelocatables(const std::vector<hipblaslt_jit::Artifact>& objects,
                            const Target&                               target) noexcept
    {
        return guarded<Result>([&](std::string& log) {
            if(objects.empty())
                fail(Status::InvalidArgument, "no relocatable objects to link");
            auto input = makeSet();
            for(size_t i = 0; i < objects.size(); ++i)
            {
                const auto& object = objects[i];
                if(object.kind != hipblaslt_jit::Artifact::Kind::Relocatable)
                    fail(Status::InvalidArgument, "'" + object.name + "' is not a relocatable");
                if(object.bytes.empty())
                    fail(Status::InvalidArgument, "relocatable '" + object.name + "' is empty");
                add(input,
                    AMD_COMGR_DATA_KIND_RELOCATABLE,
                    dataName(i, object.name, ".o"),
                    object.bytes.data(),
                    object.bytes.size());
            }
            // comgr links through its clang driver, so lld options need -Xlinker.
            const std::vector<std::string> flags  = {"-Xlinker", "--build-id=sha1"};
            const auto                     info   = makeAction(target.isaName(), flags);
            const auto                     output = run(
                AMD_COMGR_ACTION_LINK_RELOCATABLE_TO_EXECUTABLE, "link", info, input, flags, log);
            auto executables = collect(output, AMD_COMGR_DATA_KIND_EXECUTABLE);
            if(executables.size() != 1 || executables.front().second.empty())
                fail(Status::NoOutput,
                     "link produced " + std::to_string(executables.size())
                         + " executables; expected one non-empty executable");
            return Result{Status::Success,
                          {},
                          {{},
                           hipblaslt_jit::Artifact::Kind::Executable,
                           std::move(executables.front().second)}};
        });
    }

    ObjectCheck inspect(const std::vector<uint8_t>& elf) noexcept
    {
        return guarded<ObjectCheck>([&](std::string& log) { return readMetadataImpl(elf, log); });
    }

    MetadataNote metadataNote(const std::vector<uint8_t>& elf) noexcept
    {
        return guarded<MetadataNote>(
            [&](std::string& log) { return metadataNoteImpl(elf, log); });
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
}

namespace hipblaslt_jit
{
    std::optional<Target> Target::fromTargetId(const std::string& targetId)
    {
        std::vector<std::string> parts;
        for(size_t begin = 0;;)
        {
            const auto end = targetId.find(':', begin);
            parts.push_back(targetId.substr(begin, end - begin));
            if(end == std::string::npos)
                break;
            begin = end + 1;
        }
        Target target;
        target.processor = TensileLite::AMDGPU::toProcessor(parts.front());
        if(target.processor == Processor::gfx000 || target.processorName() != parts.front())
            return std::nullopt;
        for(size_t i = 1; i < parts.size(); ++i)
        {
            const auto& part    = parts[i];
            const auto  name    = part.substr(0, part.empty() ? 0 : part.size() - 1);
            auto*       feature = name == "sramecc" ? &target.sramecc
                                  : name == "xnack" ? &target.xnack
                                                    : nullptr;
            if(!feature || *feature != TargetFeature::Any
               || (part.back() != '+' && part.back() != '-'))
                return std::nullopt;
            *feature = part.back() == '+' ? TargetFeature::On : TargetFeature::Off;
        }
        return target;
    }

    std::string Target::processorName() const
    {
        return TensileLite::AMDGPU::toString(processor);
    }

    std::string Target::targetId() const
    {
        const auto feature = [](const char* name, TargetFeature setting) {
            if(setting == TargetFeature::Any)
                return std::string();
            return std::string(":") + name + (setting == TargetFeature::On ? "+" : "-");
        };
        return processorName() + feature("sramecc", sramecc) + feature("xnack", xnack);
    }

    std::string Target::isaName() const
    {
        return "amdgcn-amd-amdhsa--" + targetId();
    }
}
