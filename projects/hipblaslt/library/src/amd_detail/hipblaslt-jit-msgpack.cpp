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
}
#endif

namespace hipblaslt_jit::msgpack_io
{
#ifdef TENSILE_MSGPACK
    namespace
    {
        Status invalid(const std::string& why)
        {
            return {Status::Code::Failed, Stage::Configure, "Invalid JIT index allocator: " + why};
        }
    }

    Status writeAllocator(int64_t next, std::vector<uint8_t>& bytes)
    {
        Allocator                                     allocator{next};
        TensileLite::Serialization::MessagePackOutput out;
        out.output(allocator);
        const auto* data = reinterpret_cast<const uint8_t*>(out.buffer.data());
        bytes.assign(data, data + out.buffer.size());
        return {};
    }

    Status readAllocator(const std::vector<uint8_t>& bytes, int64_t& next)
    {
        try
        {
            size_t     offset = 0;
            const auto handle = msgpack::unpack(
                reinterpret_cast<const char*>(bytes.data()), bytes.size(), offset);
            if(offset != bytes.size())
                return invalid("trailing data");
            Allocator                                    allocator;
            TensileLite::Serialization::MessagePackInput in(handle.get());
            in.input(allocator);
            if(!in.error.empty())
                return invalid(in.error.front());
            next = allocator.next;
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
#endif
}
