#pragma once

#include <GPUArchitecture/GPUArchitecture.hpp>
#include <GPUArchitecture/GPUArchitectureTarget.hpp>
#include <GPUArchitecture/GPUCapability.hpp>
#include <GPUArchitecture/GPUInstructionInfo.hpp>

#include <msgpack.hpp>

namespace msgpack
{
    MSGPACK_API_VERSION_NAMESPACE(MSGPACK_DEFAULT_API_NS)
    {
        namespace adaptor
        {
            template <>
            struct convert<rocRoller::GPUArchitecture>
            {
                msgpack::object const& operator()(msgpack::object const&      o,
                                                  rocRoller::GPUArchitecture& v) const
                {
                    if(o.type != msgpack::type::ARRAY)
                    {
                        throw msgpack::type_error();
                    }
                    if(o.via.array.size != 3)
                    {
                        throw msgpack::type_error();
                    }
                    v = rocRoller::GPUArchitecture(
                        rocRoller::GPUArchitectureTarget::fromString(
                            o.via.array.ptr[0].as<std::string>()),
                        o.via.array.ptr[1].as<std::map<rocRoller::GPUCapability, int>>(),
                        o.via.array.ptr[2]
                            .as<std::map<std::string, rocRoller::GPUInstructionInfo>>());
                    return o;
                }
            };

            template <>
            struct pack<rocRoller::GPUArchitecture>
            {
                template <typename Stream>
                packer<Stream>& operator()(msgpack::packer<Stream>&          o,
                                           rocRoller::GPUArchitecture const& v) const
                {
                    o.pack_array(3);
                    o.pack(v.target().toString());
                    o.pack(v.getAllCapabilities());
                    o.pack(v.getAllIntructionInfo());
                    return o;
                }
            };

            template <>
            struct convert<rocRoller::GPUArchitectureTarget>
            {
                msgpack::object const& operator()(msgpack::object const&            o,
                                                  rocRoller::GPUArchitectureTarget& v) const
                {
                    if(o.type != msgpack::type::ARRAY)
                    {
                        throw msgpack::type_error();
                    }
                    if(o.via.array.size != 1)
                    {
                        throw msgpack::type_error();
                    }
                    v = rocRoller::GPUArchitectureTarget::fromString(
                        o.via.array.ptr[0].as<std::string>());
                    return o;
                }
            };

            template <>
            struct pack<rocRoller::GPUArchitectureTarget>
            {
                template <typename Stream>
                packer<Stream>& operator()(msgpack::packer<Stream>&                o,
                                           rocRoller::GPUArchitectureTarget const& v) const
                {
                    o.pack_array(1);
                    o.pack(v.toString());
                    return o;
                }
            };

            template <>
            struct convert<rocRoller::GPUArchitectureGFX>
            {
                msgpack::object const& operator()(msgpack::object const&         o,
                                                  rocRoller::GPUArchitectureGFX& v) const
                {
                    if(o.type != msgpack::type::ARRAY)
                    {
                        throw msgpack::type_error();
                    }
                    if(o.via.array.size != 1)
                    {
                        throw msgpack::type_error();
                    }
                    v = rocRoller::GPUArchitectureGFX(o.via.array.ptr[0].as<int32_t>());
                    return o;
                }
            };

            template <>
            struct pack<rocRoller::GPUArchitectureGFX>
            {
                template <typename Stream>
                packer<Stream>& operator()(msgpack::packer<Stream>&             o,
                                           rocRoller::GPUArchitectureGFX const& v) const
                {
                    o.pack_array(1);
                    o.pack((int32_t)v);
                    return o;
                }
            };

            template <>
            struct convert<rocRoller::GPUCapability>
            {
                msgpack::object const& operator()(msgpack::object const&    o,
                                                  rocRoller::GPUCapability& v) const
                {
                    if(o.type != msgpack::type::ARRAY)
                    {
                        throw msgpack::type_error();
                    }
                    if(o.via.array.size != 1)
                    {
                        throw msgpack::type_error();
                    }
                    v = rocRoller::GPUCapability(o.via.array.ptr[0].as<uint8_t>());
                    return o;
                }
            };

            template <>
            struct pack<rocRoller::GPUCapability>
            {
                template <typename Stream>
                packer<Stream>& operator()(msgpack::packer<Stream>&        o,
                                           rocRoller::GPUCapability const& v) const
                {
                    o.pack_array(1);
                    o.pack((uint8_t)v);
                    return o;
                }
            };

            template <>
            struct convert<rocRoller::GPUInstructionInfo>
            {
                msgpack::object const& operator()(msgpack::object const&         o,
                                                  rocRoller::GPUInstructionInfo& v) const
                {
                    if(o.type != msgpack::type::ARRAY)
                    {
                        throw msgpack::type_error();
                    }
                    if(o.via.array.size != 7)
                    {
                        throw msgpack::type_error();
                    }
                    v = rocRoller::GPUInstructionInfo(
                        o.via.array.ptr[0].as<std::string>(),
                        o.via.array.ptr[1].as<int>(),
                        o.via.array.ptr[2].as<std::vector<rocRoller::GPUWaitQueueType>>(),
                        o.via.array.ptr[3].as<int>(),
                        o.via.array.ptr[4].as<bool>(),
                        o.via.array.ptr[5].as<bool>(),
                        o.via.array.ptr[6].as<unsigned int>());
                    return o;
                }
            };

            template <>
            struct pack<rocRoller::GPUInstructionInfo>
            {
                template <typename Stream>
                packer<Stream>& operator()(msgpack::packer<Stream>&             o,
                                           rocRoller::GPUInstructionInfo const& v) const
                {
                    o.pack_array(7);
                    o.pack(v.getInstruction());
                    o.pack(v.getWaitCount());
                    o.pack(v.getWaitQueues());
                    o.pack(v.getLatency());
                    o.pack(v.hasImplicitAccess());
                    o.pack(v.isBranch());
                    o.pack(v.maxLiteralValue());
                    return o;
                }
            };

            template <>
            struct convert<rocRoller::GPUWaitQueueType>
            {
                msgpack::object const& operator()(msgpack::object const&       o,
                                                  rocRoller::GPUWaitQueueType& v) const
                {
                    if(o.type != msgpack::type::ARRAY)
                    {
                        throw msgpack::type_error();
                    }
                    if(o.via.array.size != 1)
                    {
                        throw msgpack::type_error();
                    }
                    v = rocRoller::GPUWaitQueueType(o.via.array.ptr[0].as<uint8_t>());
                    return o;
                }
            };

            template <>
            struct pack<rocRoller::GPUWaitQueueType>
            {
                template <typename Stream>
                packer<Stream>& operator()(msgpack::packer<Stream>&           o,
                                           rocRoller::GPUWaitQueueType const& v) const
                {
                    o.pack_array(1);
                    o.pack((uint8_t)v);
                    return o;
                }
            };
        }
    }
}
