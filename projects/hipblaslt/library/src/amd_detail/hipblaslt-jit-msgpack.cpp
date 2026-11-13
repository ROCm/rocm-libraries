// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "hipblaslt-jit-msgpack.hpp"
#ifdef TENSILE_MSGPACK
#include <Tensile/msgpack/MessagePack.hpp>
#include <new>
#include <stdexcept>
#include <string>

namespace hipblaslt_jit::msgpack_io
{
    namespace
    {
        struct Allocator
        {
            int64_t next = 0;
        };
    }
}

namespace TensileLite::Serialization
{
    template <typename IO>
    struct MappingTraits<hipblaslt_jit::msgpack_io::Allocator, IO>
    {
        using iot = IOTraits<IO>;
        static void mapping(IO& io, hipblaslt_jit::msgpack_io::Allocator& allocator)
        {
            iot::mapRequired(io, "next", allocator.next);
        }
        const static bool flow = false;
    };

    template <typename IO>
    struct MappingTraits<hipblaslt_jit::msgpack_io::CacheKeyDocument, IO>
    {
        using iot = IOTraits<IO>;
        static void mapping(IO& io, hipblaslt_jit::msgpack_io::CacheKeyDocument& key)
        {
            if(!iot::outputting(io))
            {
                key.hasBackendId      = iot::hasKey(io, "backend_id");
                key.hasBackendVersion = iot::hasKey(io, "backend_version");
                key.hasTargetId       = iot::hasKey(io, "target_id");
                key.hasWavefront      = iot::hasKey(io, "wavefront_size");
            }
            iot::mapOptional(io, "backend_id", key.backendId);
            iot::mapOptional(io, "backend_version", key.backendVersion);
            iot::mapOptional(io, "target_id", key.targetId);
            iot::mapOptional(io, "wavefront_size", key.wavefrontSize);
        }
        const static bool flow = false;
    };
}
#endif

namespace hipblaslt_jit::msgpack_io
{
#ifdef TENSILE_MSGPACK
    namespace
    {
        template <class Document>
        void write(Document document, std::vector<uint8_t>& bytes)
        {
            TensileLite::Serialization::MessagePackOutput out;
            out.output(document);
            const auto* data = reinterpret_cast<const uint8_t*>(out.buffer.data());
            bytes.assign(data, data + out.buffer.size());
        }

        template <class Document>
        Status read(const std::vector<uint8_t>& bytes, const char* what, Document& document)
        {
            const auto invalid = [what](const std::string& why) {
                return Status{Status::Code::Failed, Stage::Configure, what + (": " + why)};
            };
            try
            {
                size_t     offset = 0;
                const auto handle = msgpack::unpack(
                    reinterpret_cast<const char*>(bytes.data()), bytes.size(), offset);
                if(offset != bytes.size())
                    return invalid("trailing data");
                TensileLite::Serialization::MessagePackInput in(handle.get());
                in.input(document);
                if(!in.error.empty())
                    return invalid(in.error.front());
                return {};
            }
            catch(const std::bad_alloc&)
            {
                throw;
            }
            catch(const std::exception& error)
            {
                return invalid(error.what());
            }
        }
    }

    Status writeAllocator(int64_t next, std::vector<uint8_t>& bytes)
    {
        write(Allocator{next}, bytes);
        return {};
    }

    Status readAllocator(const std::vector<uint8_t>& bytes, int64_t& next)
    {
        Allocator allocator;
        auto      status = read(bytes, "Invalid JIT index allocator", allocator);
        if(status.ok())
            next = allocator.next;
        return status;
    }

    Status writeCacheKey(const CacheKeyDocument& document, std::vector<uint8_t>& bytes)
    {
        write(document, bytes);
        return {};
    }

    Status readCacheKey(const std::vector<uint8_t>& bytes, CacheKeyDocument& document)
    {
        document = {};
        return read(bytes, "Invalid JIT cache key", document);
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
