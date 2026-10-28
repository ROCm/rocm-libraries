// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-msgpack.hpp"
#ifdef TENSILE_MSGPACK
#include <msgpack.hpp>
#include <new>
#include <stdexcept>
#include <string>
#include <string_view>
#endif

namespace hipblaslt_jit::msgpack_io
{
#ifdef TENSILE_MSGPACK
    namespace
    {
        using Packer = msgpack::packer<msgpack::sbuffer>;

        [[noreturn]] void malformed(const std::string& message)
        {
            throw std::runtime_error(message);
        }

        template <class F>
        Status guarded(const char* what, F&& f)
        {
            try
            {
                f();
                return {};
            }
            catch(const std::bad_alloc&)
            {
                throw;
            }
            catch(const std::exception& e)
            {
                return {Status::Code::Failed, Stage::Configure, std::string(what) + ": " + e.what()};
            }
        }

        msgpack::object_handle parse(const std::vector<uint8_t>& bytes, const char* what)
        {
            size_t offset = 0;
            auto   handle
                = msgpack::unpack(reinterpret_cast<const char*>(bytes.data()), bytes.size(), offset);
            if(offset != bytes.size())
                malformed(std::string(what) + " has trailing data");
            return handle;
        }

        std::vector<uint8_t> contents(const msgpack::sbuffer& buffer)
        {
            const auto* data = reinterpret_cast<const uint8_t*>(buffer.data());
            return {data, data + buffer.size()};
        }

        std::string_view text(const msgpack::object& object, const char* what)
        {
            if(object.type != msgpack::type::STR)
                malformed(std::string(what) + " is not a string");
            return {object.via.str.ptr, object.via.str.size};
        }

        bool named(const msgpack::object& key, std::string_view name)
        {
            return key.type == msgpack::type::STR
                   && std::string_view(key.via.str.ptr, key.via.str.size) == name;
        }

        const msgpack::object* find(const msgpack::object& object, std::string_view name)
        {
            if(object.type != msgpack::type::MAP)
                return nullptr;
            for(uint32_t i = 0; i < object.via.map.size; ++i)
                if(named(object.via.map.ptr[i].key, name))
                    return &object.via.map.ptr[i].val;
            return nullptr;
        }

        const msgpack::object& member(const msgpack::object& object, std::string_view name, const char* what)
        {
            if(object.type != msgpack::type::MAP)
                malformed(std::string(what) + " is not a map");
            if(const auto* value = find(object, name))
                return *value;
            malformed(std::string(what) + " has no \"" + std::string(name) + "\"");
        }

        void string(Packer& packer, std::string_view value)
        {
            packer.pack_str(static_cast<uint32_t>(value.size()));
            packer.pack_str_body(value.data(), static_cast<uint32_t>(value.size()));
        }
    }

    Status writeAllocator(int64_t next, std::vector<uint8_t>& bytes)
    {
        bytes.clear();
        return guarded("Invalid JIT index allocator", [&] {
            msgpack::sbuffer buffer;
            Packer           packer(buffer);
            packer.pack_map(1);
            string(packer, "next");
            packer.pack(next);
            bytes = contents(buffer);
        });
    }

    Status readAllocator(const std::vector<uint8_t>& bytes, int64_t& next)
    {
        return guarded("Invalid JIT index allocator", [&] {
            const auto handle = parse(bytes, "The allocator");
            next              = member(handle.get(), "next", "The allocator").as<int64_t>();
        });
    }

    namespace
    {
        std::string optionalString(const msgpack::object& root, const char* name, bool& present)
        {
            present = false;
            const auto* value = find(root, name);
            if(!value || value->type != msgpack::type::STR)
                return {};
            present = true;
            return std::string(text(*value, name));
        }

        int optionalInt(const msgpack::object& root, const char* name, bool& present, int fallback)
        {
            present           = false;
            const auto* value = find(root, name);
            if(!value
               || (value->type != msgpack::type::POSITIVE_INTEGER
                   && value->type != msgpack::type::NEGATIVE_INTEGER))
                return fallback;
            present = true;
            return static_cast<int>(value->as<int64_t>());
        }
    }

    Status writeCacheKey(const CacheKeyDocument& document, std::vector<uint8_t>& bytes)
    {
        bytes.clear();
        return guarded("Invalid JIT cache key", [&] {
            msgpack::sbuffer buffer;
            Packer           packer(buffer);
            packer.pack_map(11);
            string(packer, "backend_id");
            string(packer, document.backendId);
            string(packer, "backend_version");
            packer.pack(document.backendVersion);
            string(packer, "code_object_version");
            packer.pack(document.codeObjectVersion);
            string(packer, "comgr");
            string(packer, document.comgr);
            string(packer, "compiler_environment");
            packer.pack_map(static_cast<uint32_t>(document.environment.size()));
            for(const auto& [name, value] : document.environment)
            {
                string(packer, name);
                string(packer, value);
            }
            string(packer, "isa");
            string(packer, document.isa);
            string(packer, "library_arch");
            string(packer, document.libraryArch);
            string(packer, "rocm_path");
            string(packer, document.rocmPath);
            string(packer, "schema");
            packer.pack(document.schema);
            string(packer, "target_id");
            string(packer, document.targetId);
            string(packer, "wavefront_size");
            packer.pack(document.wavefrontSize);
            bytes = contents(buffer);
        });
    }

    Status readCacheKey(const std::vector<uint8_t>& bytes, CacheKeyDocument& document)
    {
        document = {};
        return guarded("Invalid JIT cache key", [&] {
            const auto handle = parse(bytes, "The cache key");
            const auto& root  = handle.get();
            if(root.type != msgpack::type::MAP)
                malformed("The cache key is not a map");
            bool ignored = false;
            document.backendId = optionalString(root, "backend_id", document.hasBackendId);
            const int version   = optionalInt(root, "backend_version", document.hasBackendVersion, 0);
            if(document.hasBackendVersion && version >= 0)
                document.backendVersion = static_cast<uint32_t>(version);
            else
                document.hasBackendVersion = false;
            document.codeObjectVersion
                = optionalInt(root, "code_object_version", ignored, document.codeObjectVersion);
            document.comgr       = optionalString(root, "comgr", ignored);
            document.isa         = optionalString(root, "isa", document.hasIsa);
            document.libraryArch = optionalString(root, "library_arch", document.hasLibraryArch);
            document.rocmPath    = optionalString(root, "rocm_path", ignored);
            document.schema      = optionalInt(root, "schema", ignored, document.schema);
            document.targetId    = optionalString(root, "target_id", document.hasTargetId);
            document.wavefrontSize
                = optionalInt(root, "wavefront_size", document.hasWavefront, document.wavefrontSize);
            if(const auto* environment = find(root, "compiler_environment");
               environment && environment->type == msgpack::type::MAP)
            {
                for(uint32_t i = 0; i < environment->via.map.size; ++i)
                {
                    const auto& field = environment->via.map.ptr[i];
                    if(field.key.type != msgpack::type::STR || field.val.type != msgpack::type::STR)
                        continue;
                    document.environment.emplace(text(field.key, "An environment name"),
                                                 text(field.val, "An environment value"));
                }
            }
        });
    }
#else
    namespace
    {
        Status unavailable()
        {
            return {Status::Code::Failed,
                    Stage::Configure,
                    "The JIT solution library requires a MessagePack build"};
        }
    }

    Status writeAllocator(int64_t, std::vector<uint8_t>&)
    {
        return unavailable();
    }
    Status readAllocator(const std::vector<uint8_t>&, int64_t&)
    {
        return unavailable();
    }
    Status writeCacheKey(const CacheKeyDocument&, std::vector<uint8_t>&)
    {
        return unavailable();
    }
    Status readCacheKey(const std::vector<uint8_t>&, CacheKeyDocument&)
    {
        return unavailable();
    }
#endif
}
