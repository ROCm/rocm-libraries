// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <rocRoller/Serialization/Base.hpp>
#include <rocRoller/Serialization/Containers.hpp>
#include <rocRoller/Serialization/HasTraits.hpp>
#include <rocRoller/Utilities/Error.hpp>

#include <amd_comgr/amd_comgr.h>

namespace rocRoller
{
    namespace Serialization
    {
        struct ComgrMetadataNode
        {
            ComgrMetadataNode() = default;

            ~ComgrMetadataNode()
            {
                reset();
            }

            // Non-copyable
            ComgrMetadataNode(ComgrMetadataNode const&) = delete;
            ComgrMetadataNode& operator=(ComgrMetadataNode const&) = delete;

            // Movable
            ComgrMetadataNode(ComgrMetadataNode&& rhs) noexcept
                : handle(rhs.handle)
            {
                rhs.handle = {};
            }

            ComgrMetadataNode& operator=(ComgrMetadataNode&& rhs) noexcept
            {
                if(this != &rhs)
                {
                    reset();
                    handle     = rhs.handle;
                    rhs.handle = {};
                }
                return *this;
            }

            [[nodiscard]] amd_comgr_metadata_node_t* addressof() noexcept
            {
                return &handle;
            }

        private:
            amd_comgr_metadata_node_t handle{};

            void reset() noexcept
            {
                // A handle value of 0 indicates an uninitialized/null metadata node
                if(handle.handle != 0)
                {
                    amd_comgr_destroy_metadata(handle);
                    handle = {};
                }
            }
        };

        struct ComgrNodeInput
        {
            amd_comgr_metadata_node_t node;
            void*                     context;

            ComgrNodeInput(amd_comgr_metadata_node_t n, void* c = nullptr)
                : node(n)
                , context(c)
            {
            }

            template <typename T>
            void mapRequired(const char* key, T& obj)
            {
                ComgrMetadataNode value;
                auto              status = amd_comgr_metadata_lookup(node, key, value.addressof());
                AssertFatal(status == AMD_COMGR_STATUS_SUCCESS,
                            "Key ",
                            ShowValue(key),
                            " not found in comgr metadata");
                input(*value.addressof(), obj);
            }

            template <typename T>
            void mapOptional(const char* key, T& obj)
            {
                ComgrMetadataNode value;
                auto              status = amd_comgr_metadata_lookup(node, key, value.addressof());
                if(status == AMD_COMGR_STATUS_SUCCESS)
                {
                    input(*value.addressof(), obj);
                }
            }

            template <typename T>
            requires(CMappedType<T, ComgrNodeInput> || EmptyMappedType<T, ComgrNodeInput>) void input(
                amd_comgr_metadata_node_t& n, T& obj)
            {
                ComgrNodeInput subInput(n, context);
                EmptyContext   ctx;
                MappingTraits<T, ComgrNodeInput>::mapping(subInput, obj, ctx);
            }

            template <SequenceType<ComgrNodeInput> T>
            void input(amd_comgr_metadata_node_t& n, T& obj)
            {
                size_t count;
                auto   status = amd_comgr_get_metadata_list_size(n, &count);
                AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to get list size");

                for(size_t i = 0; i < count; i++)
                {
                    ComgrMetadataNode elNode;
                    status = amd_comgr_index_list_metadata(n, i, elNode.addressof());
                    AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to index list");

                    auto& value = SequenceTraits<T, ComgrNodeInput>::element(*this, obj, i);
                    input(*elNode.addressof(), value);
                }
            }

            template <typename T>
            void input(amd_comgr_metadata_node_t& n, T& obj)
            {
                comgrNodeInputHelper(n, obj);
            }

            template <CustomMappingType<ComgrNodeInput> T>
            void input(amd_comgr_metadata_node_t& n, T& obj)
            {
                auto callback = [](amd_comgr_metadata_node_t key,
                                   amd_comgr_metadata_node_t value,
                                   void*                     user_data) -> amd_comgr_status_t {
                    auto*  pair = static_cast<std::pair<ComgrNodeInput*, T*>*>(user_data);
                    size_t size;
                    amd_comgr_get_metadata_string(key, &size, nullptr); // Get size
                    std::string keyStr(size - 1, '\0');
                    amd_comgr_get_metadata_string(key, &size, keyStr.data()); // Get string

                    CustomMappingTraits<T, ComgrNodeInput>::inputOne(
                        *pair->first, keyStr, *pair->second);
                    return AMD_COMGR_STATUS_SUCCESS;
                };

                ComgrNodeInput                 subInput(n, context);
                std::pair<ComgrNodeInput*, T*> userData(&subInput, &obj);
                amd_comgr_iterate_map_metadata(n, callback, &userData);
            }

            template <CHasScalarTraits T>
            void input(amd_comgr_metadata_node_t& n, T& obj)
            {
                std::string stringVal;
                input(n, stringVal);
                ScalarTraits<T>::input(stringVal, obj);
            }

            constexpr bool outputting() const
            {
                return false;
            }

        private:
            template <typename T>
            void comgrNodeInputHelper(amd_comgr_metadata_node_t& n, T& obj)
            {
                amd_comgr_metadata_kind_t kind;
                amd_comgr_get_metadata_kind(n, &kind);
                if(kind == AMD_COMGR_METADATA_KIND_STRING)
                {
                    size_t      size;
                    auto        status = amd_comgr_get_metadata_string(n, &size, nullptr);
                    std::string str(size - 1, '\0');
                    status = amd_comgr_get_metadata_string(n, &size, str.data());
                }
            }
        };

        template <>
        inline void ComgrNodeInput::comgrNodeInputHelper(amd_comgr_metadata_node_t& n,
                                                         std::string&               val)
        {
            size_t size;
            auto   status = amd_comgr_get_metadata_string(n, &size, nullptr);
            AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to get string size");

            val.resize(size - 1);
            status = amd_comgr_get_metadata_string(n, &size, val.data());
            AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to get string");
        }

        template <>
        inline void ComgrNodeInput::comgrNodeInputHelper(amd_comgr_metadata_node_t& n, int& val)
        {
            std::string str;
            comgrNodeInputHelper(n, str);
            val = std::stoi(str);
        }

        template <>
        inline void ComgrNodeInput::comgrNodeInputHelper(amd_comgr_metadata_node_t& n,
                                                         unsigned int&              val)
        {
            std::string str;
            comgrNodeInputHelper(n, str);
            val = static_cast<unsigned int>(std::stoul(str));
        }

        template <>
        inline void ComgrNodeInput::comgrNodeInputHelper(amd_comgr_metadata_node_t& n, uint8_t& val)
        {
            std::string str;
            comgrNodeInputHelper(n, str);
            val = static_cast<uint8_t>(std::stoul(str));
        }

        template <>
        inline void ComgrNodeInput::comgrNodeInputHelper(amd_comgr_metadata_node_t& n, bool& val)
        {
            amd_comgr_metadata_kind_t kind;
            amd_comgr_get_metadata_kind(n, &kind);

            if(kind == AMD_COMGR_METADATA_KIND_STRING)
            {
                std::string str;
                comgrNodeInputHelper(n, str);
                if(str == "true" || str == "1")
                    val = true;
                else if(str == "false" || str == "0")
                    val = false;
                else
                    AssertFatal(false, "Invalid boolean string: ", str);
            }
            else
            {
                AssertFatal(false, "Unsupported boolean metadata");
            }
        }

        template <>
        inline void ComgrNodeInput::comgrNodeInputHelper(amd_comgr_metadata_node_t& n, Half& val)
        {
            float floatVal;
            comgrNodeInputHelper(n, floatVal);
            val = floatVal;
        }

        template <>
        inline void ComgrNodeInput::comgrNodeInputHelper(amd_comgr_metadata_node_t& n,
                                                         BFloat16&                  val)
        {
            float floatVal;
            comgrNodeInputHelper(n, floatVal);
            val.data = floatVal;
        }

        template <>
        inline void ComgrNodeInput::comgrNodeInputHelper(amd_comgr_metadata_node_t& n, FP8& val)
        {
            std::string str;
            comgrNodeInputHelper(n, str);
            val.data = static_cast<uint8_t>(std::strtoul(str.c_str(), nullptr, 10));
        }

        template <>
        inline void ComgrNodeInput::comgrNodeInputHelper(amd_comgr_metadata_node_t& n, BF8& val)
        {
            std::string str;
            comgrNodeInputHelper(n, str);
            val.data = static_cast<uint8_t>(std::strtoul(str.c_str(), nullptr, 10));
        }

        template <>
        inline void ComgrNodeInput::comgrNodeInputHelper(amd_comgr_metadata_node_t& n, FP6& val)
        {
            std::string str;
            comgrNodeInputHelper(n, str);
            val.data = static_cast<uint8_t>(std::strtoul(str.c_str(), nullptr, 10));
        }

        template <>
        inline void ComgrNodeInput::comgrNodeInputHelper(amd_comgr_metadata_node_t& n, BF6& val)
        {
            std::string str;
            comgrNodeInputHelper(n, str);
            val.data = static_cast<uint8_t>(std::strtoul(str.c_str(), nullptr, 10));
        }

        template <>
        inline void ComgrNodeInput::comgrNodeInputHelper(amd_comgr_metadata_node_t& n, FP4& val)
        {
            std::string str;
            comgrNodeInputHelper(n, str);
            val.data = static_cast<uint8_t>(std::strtoul(str.c_str(), nullptr, 10));
        }

        template <>
        inline void ComgrNodeInput::comgrNodeInputHelper(amd_comgr_metadata_node_t& n, E8M0& val)
        {
            std::string str;
            comgrNodeInputHelper(n, str);
            val.scale = static_cast<uint8_t>(std::strtoul(str.c_str(), nullptr, 10));
        }

        template <>
        struct IOTraits<ComgrNodeInput>
        {
            using IO = ComgrNodeInput;

            template <typename T>
            static void mapRequired(IO& io, const char* key, T& obj)
            {
                io.mapRequired(key, obj);
            }

            template <typename T, typename Context>
            static void mapRequired(IO& io, const char* key, T& obj, Context& ctx)
            {
                io.mapRequired(key, obj);
            }

            template <typename T>
            static void mapOptional(IO& io, const char* key, T& obj)
            {
                io.mapOptional(key, obj);
            }

            template <typename T, typename Context>
            static void mapOptional(IO& io, const char* key, T& obj, Context& ctx)
            {
                io.mapOptional(key, obj);
            }

            static constexpr bool outputting(IO& io)
            {
                return io.outputting();
            }

            static void setError(IO& io, std::string const& msg)
            {
                throw std::runtime_error(msg);
            }

            static void setContext(IO& io, void* ctx)
            {
                io.context = ctx;
            }

            static void* getContext(IO& io)
            {
                return io.context;
            }

            template <typename T>
            static void enumCase(IO& io, T& member, const char* key, T value)
            {
                size_t size;
                auto   status = amd_comgr_get_metadata_string(io.node, &size, nullptr);
                if(status == AMD_COMGR_STATUS_SUCCESS)
                {
                    std::string str(size - 1, '\0');
                    amd_comgr_get_metadata_string(io.node, &size, str.data());
                    if(str == key)
                    {
                        member = value;
                    }
                }
            }
        };

    } // namespace Serialization
} // namespace rocRoller
