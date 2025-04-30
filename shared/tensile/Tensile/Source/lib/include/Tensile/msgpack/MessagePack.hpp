/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

#include <type_traits>

#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/Serialization.hpp>

#include <msgpack.hpp>

namespace Tensile
{
    namespace Serialization
    {
        template <typename T>
        struct is_Convertible
        {
            static const bool value = false;
        };

        template <>
        struct is_Convertible<std::string>
        {
            static const bool value = true;
        };
        template <>
        struct is_Convertible<int8_t>
        {
            static const bool value = true;
        };
        template <>
        struct is_Convertible<int16_t>
        {
            static const bool value = true;
        };
        template <>
        struct is_Convertible<int32_t>
        {
            static const bool value = true;
        };
        template <>
        struct is_Convertible<int64_t>
        {
            static const bool value = true;
        };
        template <>
        struct is_Convertible<size_t>
        {
            static const bool value = true;
        };
        template <>
        struct is_Convertible<float>
        {
            static const bool value = true;
        };
        template <>
        struct is_Convertible<double>
        {
            static const bool value = true;
        };
        template <>
        struct is_Convertible<bool>
        {
            static const bool value = true;
        };

        void objectToMap(msgpack::object&                                  object,
                         std::unordered_map<std::string, msgpack::object>& map);

        struct MessagePackInput
        {
            msgpack::object          object;
            std::vector<std::string> error;

            std::unordered_map<std::string, msgpack::object> objectMap;
            std::unordered_set<std::string>                  usedKeys;

            int enumFound = 0;

            void* context = nullptr;

            MessagePackInput(msgpack::object object, void* context = nullptr)
                : object(object)
                , context(context)
            {
            }

            MessagePackInput createSubRef(msgpack::object otherObject)
            {
                return MessagePackInput(otherObject, context);
            }

            void addError(std::string const& msg)
            {
                error.push_back(msg);
            }

            template <typename T>
            void mapRequired(const char* key, T& obj)
            {
                if(objectMap.empty())
                    objectToMap(object, objectMap);

                auto iterator = objectMap.find(key);
                if(iterator != objectMap.end())
                {
                    auto&            value  = iterator->second;
                    MessagePackInput subRef = createSubRef(value);
                    subRef.input(obj);
                    error.insert(error.end(), subRef.error.begin(), subRef.error.end());
                    if(Tensile::Debug::Instance().printDataInit())
                        usedKeys.insert(key);
                }
                else
                {
                    std::string msg = "Unknown key ";
                    msg += key;
                    msg += " (keys: ";
                    bool first = true;
                    for(auto const& pair : objectMap)
                    {
                        if(!first)
                            msg += ", ";
                        msg += pair.first;
                        first = false;
                    }
                    msg += ")";
                    addError(msg);
                }
            }

            template <typename T>
            void mapOptional(const char* key, T& obj)
            {
                if(objectMap.empty())
                    objectToMap(object, objectMap);

                auto iterator = objectMap.find(key);
                if(iterator != objectMap.end())
                {
                    auto& value = iterator->second;
                    createSubRef(value).input(obj);
                    if(Tensile::Debug::Instance().printDataInit())
                        usedKeys.insert(key);
                }
            }

            template <typename T>
            void input(T& obj)
            {
                EmptyContext ctx;
                input(obj, ctx);
            }

            void checkUsedKeys()
            {
                std::unordered_set<std::string> fileKeys;
                for(auto const& pair : objectMap)
                    fileKeys.insert(pair.first);

                if(usedKeys != fileKeys)
                {
                    for(auto const& k : fileKeys)
                        if(usedKeys.find(k) == usedKeys.end())
                            addError(concatenate("Error: Unused key ", k));
                }
            }

            template <typename T, typename Context>
            typename std::enable_if<has_MappingTraits<T, MessagePackInput, Context>::value,
                                    void>::type
                input(T& obj, Context& ctx)
            {
                MappingTraits<T, MessagePackInput, Context>::mapping(*this, obj, ctx);

                if(Tensile::Debug::Instance().printDataInit())
                    checkUsedKeys();
            }

            template <typename T, typename Context>
            typename std::enable_if<has_EmptyMappingTraits<T, MessagePackInput, Context>::value,
                                    void>::type
                input(T& obj, Context& ctx)
            {
                MappingTraits<T, MessagePackInput, Context>::mapping(*this, obj);

                if(Tensile::Debug::Instance().printDataInit())
                    checkUsedKeys();
            }

            template <typename T, typename Context>
            typename std::enable_if<is_Convertible<T>::value, void>::type input(T&       obj,
                                                                                Context& ctx)
            {
                object.convert(obj);
            }

            template <typename T, typename Context>
            typename std::enable_if<has_EnumTraits<T, MessagePackInput>::value, void>::type
                input(T& obj, Context& ctx)
            {
                enumFound = 0;
                EnumTraits<T, MessagePackInput>::enumeration(*this, obj);

                if(enumFound != 1)
                    addError(concatenate("Enum not found!", obj));
            }

            template <typename T, typename Context>
            typename std::enable_if<has_SequenceTraits<T, MessagePackInput>::value, void>::type
                input(T& obj, Context& ctx)
            {
                assert(object.type == msgpack::type::object_type::ARRAY);
                auto result = object.as<std::vector<msgpack::object>>();

                for(size_t i = 0; i < result.size(); i++)
                {
                    MessagePackInput subRef = createSubRef(result[i]);
                    auto& value = SequenceTraits<T, MessagePackInput>::element(*this, obj, i);
                    subRef.input(value);

                    if(!subRef.error.empty())
                    {
                        error.insert(error.end(), subRef.error.begin(), subRef.error.end());
                        return;
                    }
                }
            }

            template <typename T, typename Context>
            typename std::enable_if<has_CustomMappingTraits<T, MessagePackInput>::value, void>::type
                input(T& obj, Context& ctx)
            {
                objectToMap(object, objectMap);

                for(auto& element : objectMap)
                {
                    CustomMappingTraits<T, MessagePackInput>::inputOne(*this, element.first, obj);
                }
            }

            template <typename T>
            void enumCase(T& member, const char* key, T value)
            {
                assert(object.type == msgpack::type::object_type::STR);
                std::string result;
                object.convert(result);

                if(result == key)
                {
                    enumFound++;
                    member = value;
                }
            }
        };

        template <>
        struct IOTraits<MessagePackInput>
        {
            template <typename T>
            static void mapRequired(MessagePackInput& io, const char* key, T& obj)
            {
                io.mapRequired(key, obj);
            }

            template <typename T>
            static void mapOptional(MessagePackInput& io, const char* key, T& obj)
            {
                io.mapOptional(key, obj);
            }

            static bool outputting(MessagePackInput& io)
            {
                return false;
            }

            static void setError(MessagePackInput& io, std::string const& msg)
            {
                io.error.push_back(msg);
            }

            static void setContext(MessagePackInput& io, void* ctx)
            {
                io.context = ctx;
            }

            static void* getContext(MessagePackInput& io)
            {
                return io.context;
            }

            template <typename T>
            static void enumCase(MessagePackInput& io, T& member, const char* key, T value)
            {
                io.enumCase(member, key, value);
            }
        };

        template <typename T, typename IO>
        struct SequenceTraits<std::vector<T>, IO>
        {
            static size_t size(MessagePackInput& io, std::vector<T>& v)
            {
                return v.size();
            }
            static T& element(MessagePackInput& io, std::vector<T>& v, size_t index)
            {
                if(index >= v.size())
                {
                    size_t n = index - v.size() + 1;
                    v.insert(v.end(), n, T());
                }

                return v[index];
            }
        };

        template <typename IO>
        struct MappingTraits<std::shared_ptr<Tensile::MasterContractionLibrary>, IO>
        {
            using obj = MasterContractionLibrary;

            static void mapping(IO& io, std::shared_ptr<obj>& o)
            {
                PointerMappingTraits<obj, IO>::mapping(io, o);
            }
        };
    }
}
