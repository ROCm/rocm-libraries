/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/


#pragma once

#include <rocRoller/Serialization/Base.hpp>
#include <rocRoller/Serialization/Containers.hpp>
#include <rocRoller/Serialization/HasTraits.hpp>
#include <rocRoller/Utilities/Error.hpp>

#include <amd_comgr/amd_comgr.h>

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_map>

namespace rocRoller
{
    namespace Serialization
    {
        struct ComgrNodeInput
        {
            amd_comgr_metadata_node_t node;
            void* context;

            ComgrNodeInput(amd_comgr_metadata_node_t n, void* c = nullptr)
                : node(n)
                , context(c)
            {
            }

            template <typename T>
            void mapRequired(const char* key, T& obj)
            {
                amd_comgr_metadata_node_t value;
                auto status = amd_comgr_metadata_lookup(node, key, &value);
                AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, 
                    "Key ", ShowValue(key), " not found in comgr metadata");
                input(value, obj);
                amd_comgr_destroy_metadata(value);
            }

            template <typename T>
            void mapOptional(const char* key, T& obj)
            {
                amd_comgr_metadata_node_t value;
                auto status = amd_comgr_metadata_lookup(node, key, &value);
                if(status == AMD_COMGR_STATUS_SUCCESS)
                {
                    input(value, obj);
                    amd_comgr_destroy_metadata(value);
                }
            }

            template <typename T>
            requires(CMappedType<T, ComgrNodeInput> || EmptyMappedType<T, ComgrNodeInput>)
            void input(amd_comgr_metadata_node_t& n, T& obj)
            {
                ComgrNodeInput subInput(n, context);
                EmptyContext ctx;
                MappingTraits<T, ComgrNodeInput>::mapping(subInput, obj, ctx);
            }

            template <typename T>
            void input(amd_comgr_metadata_node_t& n, T& obj)
            {
                comgrNodeInputHelper(n, obj);
            }

            template <SequenceType<ComgrNodeInput> T>
            void input(amd_comgr_metadata_node_t& n, T& obj)
            {
                size_t count;
                auto status = amd_comgr_get_metadata_list_size(n, &count);
                AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to get list size");

                for(size_t i = 0; i < count; i++)
                {
                    amd_comgr_metadata_node_t elNode;
                    status = amd_comgr_index_list_metadata(n, i, &elNode);
                    AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to index list");
                    
                    auto& value = SequenceTraits<T, ComgrNodeInput>::element(*this, obj, i);
                    input(elNode, value);
                    amd_comgr_destroy_metadata(elNode);
                }
            }

            template <CustomMappingType<ComgrNodeInput> T>
            void input(amd_comgr_metadata_node_t& n, T& obj)
            {   
                auto callback = [] (amd_comgr_metadata_node_t key, 
                                    amd_comgr_metadata_node_t value, 
                                    void* user_data
                                )-> amd_comgr_status_t
                {
                    auto* pair = static_cast<std::pair<ComgrNodeInput*, T*>*>(user_data);
                    size_t size;
                    amd_comgr_get_metadata_string(key, &size, nullptr); // Get size
                    std::string keyStr(size - 1, '\0');
                    amd_comgr_get_metadata_string(key, &size, keyStr.data()); // Get string
                    
                    CustomMappingTraits<T, ComgrNodeInput>::inputOne(*pair->first, keyStr, *pair->second);
                    return AMD_COMGR_STATUS_SUCCESS;
                };

                ComgrNodeInput subInput(n, context);
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
                size_t size;
                auto status = amd_comgr_get_metadata_string(n, &size, nullptr);
                if(status == AMD_COMGR_STATUS_SUCCESS)
                {
                    std::string str(size - 1, '\0');
                    amd_comgr_get_metadata_string(n, &size, str.data());
                    
                    // Parse the string based on type
                    if constexpr (std::is_integral_v<T>)
                    {
                        if constexpr (std::is_signed_v<T>)
                        {
                            obj = static_cast<T>(std::strtoll(str.c_str(), nullptr, 10));
                        }
                        else
                        {
                            obj = static_cast<T>(std::strtoull(str.c_str(), nullptr, 10));
                        }
                    }
                    else if constexpr (std::is_floating_point_v<T>)
                    {
                        if constexpr (std::is_same_v<T, float>)
                        {
                            obj = std::strtof(str.c_str(), nullptr);
                        }
                        else if constexpr (std::is_same_v<T, double>)
                        {
                            obj = std::strtod(str.c_str(), nullptr);
                        }
                        else
                        {
                            obj = static_cast<T>(std::strtold(str.c_str(), nullptr));
                        }
                    }
                    else if constexpr (std::is_same_v<T, bool>)
                    {
                        obj = (str == "true" || str == "1" || str == "True" || str == "TRUE");
                    }
                }
                else
                {
                    AssertFatal(false, "Failed to read comgr metadata value");
                }
            }
        };

        template <>
        inline void ComgrNodeInput::comgrNodeInputHelper(amd_comgr_metadata_node_t& n, std::string& val)
        {
            size_t size;
            auto status = amd_comgr_get_metadata_string(n, &size, nullptr);
            AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to get string size");
            
            val.resize(size - 1);
            status = amd_comgr_get_metadata_string(n, &size, val.data());
            AssertFatal(status == AMD_COMGR_STATUS_SUCCESS, "Failed to get string");
        }

        template <>
        inline void ComgrNodeInput::comgrNodeInputHelper(amd_comgr_metadata_node_t& n, Half& val)
        {
            float floatVal;
            comgrNodeInputHelper(n, floatVal);
            val = floatVal;
        }

        template <>
        inline void ComgrNodeInput::comgrNodeInputHelper(amd_comgr_metadata_node_t& n, BFloat16& val)
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
            // Assuming FP8 has a constructor or assignment from string
            // You may need to adjust this based on FP8's actual interface
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
                auto status = amd_comgr_get_metadata_string(io.node, &size, nullptr);
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