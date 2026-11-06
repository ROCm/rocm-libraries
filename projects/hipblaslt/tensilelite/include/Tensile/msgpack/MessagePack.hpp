/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2022-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include <algorithm>
#include <cstring>
#include <limits>
#include <type_traits>

#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/Debug.hpp>
#include <Tensile/Serialization.hpp>

#include <msgpack.hpp>

#include <tensilelitehost/export.h>

namespace TensileLite
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

        TENSILELITEHOST_EXPORT void objectToMap(const msgpack::object&                            object,
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
                    if(TensileLite::Debug::Instance().printDataInit())
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
                    if(TensileLite::Debug::Instance().printDataInit())
                        usedKeys.insert(key);
                }
            }

            /// Absent means defaultValue, so that what the writer omitted and
            /// what the reader produces are the same statement.
            template <typename T>
            void mapOptional(const char* key, T& obj, T const& defaultValue)
            {
                obj = defaultValue;
                mapOptional(key, obj);
            }

            // Hands back a view of a BIN value without converting it to a C++
            // type. Used for the indexed library format's solution blob, which
            // is copied wholesale and deliberately left unparsed.
            //
            // This cannot go through input()/mapRequired(): SequenceTraits is
            // specialized for every std::vector<T> below, so a byte vector
            // would be read as a msgpack ARRAY rather than a BIN.
            bool mapRawBytes(const char* key, const uint8_t*& ptr, size_t& size)
            {
                if(objectMap.empty())
                    objectToMap(object, objectMap);

                auto iterator = objectMap.find(key);
                if(iterator == objectMap.end())
                    return false;

                auto const& value = iterator->second;
                if(value.type != msgpack::type::object_type::BIN)
                    return false;

                ptr  = reinterpret_cast<const uint8_t*>(value.via.bin.ptr);
                size = value.via.bin.size;

                if(TensileLite::Debug::Instance().printDataInit())
                    usedKeys.insert(key);

                return true;
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

                if(TensileLite::Debug::Instance().printDataInit())
                    checkUsedKeys();
            }

            template <typename T, typename Context>
            typename std::enable_if<has_EmptyMappingTraits<T, MessagePackInput, Context>::value,
                                    void>::type
                input(T& obj, Context& ctx)
            {
                MappingTraits<T, MessagePackInput, Context>::mapping(*this, obj);

                if(TensileLite::Debug::Instance().printDataInit())
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
                {
                    std::string s;
                    object.convert(s);
                    addError(concatenate("Enum not found! ", s));
                }
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
                bool match = false;
                if(object.type == msgpack::type::object_type::STR)
                {
                    std::string result;
                    object.convert(result);

                    if(result == key)
                        match = true;
                }
                else if(object.type == msgpack::type::POSITIVE_INTEGER ||
                        object.type == msgpack::type::NEGATIVE_INTEGER)
                {
                    int64_t intValue;
                    object.convert(intValue);

                    if(static_cast<int64_t>(value) == intValue)
                        match = true;
                }
                else
                {
                    addError(concatenate("Unexpected msgpack type for enum: ", (int)object.type,
                                       " (expected STRING or INTEGER)"));
                }
                    
                if(match)
                {
                    enumFound++;
                    member = value;
                }
            }
        };

        template <>
        struct IOTraits<MessagePackInput>
        {
            static bool hasKey(MessagePackInput& io, const char* key)
            {
                if(io.objectMap.empty())
                    objectToMap(io.object, io.objectMap);
                return io.objectMap.find(key) != io.objectMap.end();
            }

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

            template <typename T>
            static void mapOptional(MessagePackInput& io,
                                    const char*       key,
                                    T&                obj,
                                    T const&          defaultValue)
            {
                io.mapOptional(key, obj, defaultValue);
            }

            static bool
                mapRawBytes(MessagePackInput& io, const char* key, const uint8_t*& ptr, size_t& size)
            {
                return io.mapRawBytes(key, ptr, size);
            }

            // Builds the callable a SolutionBlobCache uses to parse one solution
            // out of the blob, long after this load has returned.
            //
            // The sub-input is given a null context on purpose. Only library
            // node traits consult the context, and the LibraryIOContext driving
            // this load lives on the loader's stack -- capturing it would
            // dangle the first time a query materializes a solution.
            template <typename MySolution>
            static std::function<std::shared_ptr<MySolution>(const uint8_t*, size_t)>
                solutionDeserializer(MessagePackInput& io)
            {
                return [](const uint8_t* data, size_t size) -> std::shared_ptr<MySolution> {
                    try
                    {
                        auto handle
                            = msgpack::unpack(reinterpret_cast<const char*>(data), size);

                        MessagePackInput            sub(handle.get(), nullptr);
                        std::shared_ptr<MySolution> solution;
                        PointerMappingTraits<MySolution, MessagePackInput>::mapping(sub, solution);

                        if(!sub.error.empty())
                            return nullptr;

                        return solution;
                    }
                    catch(std::exception const&)
                    {
                        return nullptr;
                    }
                };
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

        /// Writes the documents MessagePackInput reads, driven by the same
        /// MappingTraits. A map is written into a child output first because
        /// the msgpack map header carries the number of entries.
        struct MessagePackOutput
        {
            msgpack::sbuffer         buffer;
            std::vector<std::string> error;

            // appendMap only needs how many entries this map holds, and path()
            // only needs the key being written right now, so neither key is
            // copied into a container.
            //
            // lastKey is borrowed, never owned. A key may be a temporary -- see
            // DefaultCustomMappingTraits in Serialization/Containers.hpp -- so
            // it is only set for the duration of the mapRequired call that
            // supplied it, which is the only window in which path() reads it.
            size_t      keyCount = 0;
            const char* lastKey  = nullptr;

            std::string enumKey;
            int         enumFound = 0;

            void* context = nullptr;

            MessagePackOutput const* parent  = nullptr;
            bool                     inValue = false;
            std::vector<size_t>      positions;

            explicit MessagePackOutput(void*                    context = nullptr,
                                       MessagePackOutput const* parent  = nullptr)
                : context(context)
                , parent(parent)
            {
            }

            MessagePackOutput(MessagePackOutput const&)            = delete;
            MessagePackOutput& operator=(MessagePackOutput const&) = delete;

            msgpack::packer<msgpack::sbuffer> packer()
            {
                return msgpack::packer<msgpack::sbuffer>(buffer);
            }

            std::string path() const
            {
                std::string rv = parent ? parent->path() : std::string();
                if(inValue)
                {
                    if(!rv.empty())
                        rv += ".";
                    rv += lastKey ? lastKey : "";
                }
                for(size_t position : positions)
                    rv += concatenate("[", position, "]");
                return rv;
            }

            void addError(std::string const& msg)
            {
                auto where = path();
                error.push_back(where.empty() ? msg : concatenate(where, ": ", msg));
            }

            uint32_t length(size_t size)
            {
                if(size > std::numeric_limits<uint32_t>::max())
                    addError(concatenate("Too many elements for msgpack: ", size));
                return static_cast<uint32_t>(size);
            }

            /// An output has no document to inspect: the keys it has written
            /// are not retained. Every caller must guard the question with
            /// !outputting(io) and decide from the object instead.
            bool hasKey(const char*) const
            {
                return false;
            }

            /// Packs a key and counts it. The key is only read here: the bytes
            /// are copied into the buffer and the pointer is not retained.
            void packKey(const char* key)
            {
                auto   pk  = packer();
                size_t len = std::strlen(key);
                pk.pack_str(length(len));
                pk.pack_str_body(key, static_cast<uint32_t>(len));
                keyCount++;
            }

            template <typename T>
            void mapRequired(const char* key, T& obj)
            {
                packKey(key);

                // The caller owns key and must keep it alive across this call;
                // nothing borrows it past the clear below.
                lastKey = key;
                inValue = true;
                output(obj);
                inValue = false;
                lastKey = nullptr;
            }

            /// With no value meaning "absent", an output has nothing to decide
            /// from and writes the key with whatever the member holds -- which
            /// for a key the source document omitted is its default. Prefer the
            /// overload below wherever a default can be named: it is the only
            /// form that can reproduce an absent key.
            template <typename T>
            void mapOptional(const char* key, T& obj)
            {
                mapRequired(key, obj);
            }

            /// Omits the key when the value is the one an absent key would
            /// produce, so a document that did not carry it does not grow one.
            template <typename T>
            void mapOptional(const char* key, T& obj, T const& defaultValue)
            {
                if(obj == defaultValue)
                    return;
                mapRequired(key, obj);
            }

            bool mapRawBytes(const char* key, const uint8_t*& ptr, size_t& size)
            {
                packKey(key);
                auto pk = packer();
                pk.pack_bin(length(size));
                pk.pack_bin_body(reinterpret_cast<const char*>(ptr), size);
                return true;
            }

            template <typename T>
            void output(T& obj)
            {
                EmptyContext ctx;
                output(obj, ctx);
            }

            template <typename T, typename Context>
            typename std::enable_if<has_MappingTraits<T, MessagePackOutput, Context>::value,
                                    void>::type
                output(T& obj, Context& ctx)
            {
                if(rejectNull(obj))
                    return;

                MessagePackOutput sub(context, this);
                MappingTraits<T, MessagePackOutput, Context>::mapping(sub, obj, ctx);
                appendMap(sub);
            }

            template <typename T, typename Context>
            typename std::enable_if<has_EmptyMappingTraits<T, MessagePackOutput, Context>::value,
                                    void>::type
                output(T& obj, Context& ctx)
            {
                if(rejectNull(obj))
                    return;

                MessagePackOutput sub(context, this);
                MappingTraits<T, MessagePackOutput, Context>::mapping(sub, obj);
                appendMap(sub);
            }

            template <typename T, typename Context>
            typename std::enable_if<is_Convertible<T>::value, void>::type output(T&       obj,
                                                                                 Context& ctx)
            {
                if constexpr(std::is_floating_point<T>::value)
                    packFloat(obj);
                else
                    packer().pack(obj);
            }

            template <typename T, typename Context>
            typename std::enable_if<has_EnumTraits<T, MessagePackOutput>::value, void>::type
                output(T& obj, Context& ctx)
            {
                enumFound = 0;
                EnumTraits<T, MessagePackOutput>::enumeration(*this, obj);

                if(enumFound == 0)
                {
                    addError(concatenate("Enum value has no name: ", static_cast<int64_t>(obj)));
                    packer().pack_nil();
                    return;
                }

                packer().pack(enumKey);
            }

            template <typename T, typename Context>
            typename std::enable_if<has_SequenceTraits<T, MessagePackOutput>::value, void>::type
                output(T& obj, Context& ctx)
            {
                size_t size = SequenceTraits<T, MessagePackOutput>::size(*this, obj);
                packer().pack_array(length(size));

                positions.push_back(0);
                for(size_t i = 0; i < size; i++)
                {
                    positions.back() = i;
                    output(SequenceTraits<T, MessagePackOutput>::element(*this, obj, i));
                }
                positions.pop_back();
            }

            template <typename T, typename Context>
            typename std::enable_if<has_CustomMappingTraits<T, MessagePackOutput>::value,
                                    void>::type
                output(T& obj, Context& ctx)
            {
                MessagePackOutput sub(context, this);
                CustomMappingTraits<T, MessagePackOutput>::output(sub, obj);
                appendMap(sub);
            }

            template <typename T>
            void enumCase(T& member, const char* key, T value)
            {
                if(member == value && enumFound++ == 0)
                    enumKey = key;
            }

        private:
            template <typename T>
            bool rejectNull(T const& obj)
            {
                return false;
            }

            template <typename T>
            bool rejectNull(std::shared_ptr<T> const& obj)
            {
                if(obj)
                    return false;

                addError("Null pointer");
                packer().pack_nil();
                return true;
            }

            void appendMap(MessagePackOutput const& sub)
            {
                packer().pack_map(length(sub.keyCount));
                buffer.write(sub.buffer.data(), sub.buffer.size());
                error.insert(error.end(), sub.error.begin(), sub.error.end());
            }

            // msgpack-c packs a float holding an integral value as an integer
            // (dropping the sign of -0.0); writing the float formats directly
            // keeps the encoding independent of the msgpack-c version.
            void packFloat(float value)
            {
                uint32_t bits;
                std::memcpy(&bits, &value, sizeof(bits));
                char bytes[] = {char(0xca),
                                char(bits >> 24),
                                char(bits >> 16),
                                char(bits >> 8),
                                char(bits)};
                buffer.write(bytes, sizeof(bytes));
            }

            void packFloat(double value)
            {
                uint64_t bits;
                std::memcpy(&bits, &value, sizeof(bits));
                char bytes[9] = {char(0xcb)};
                for(int i = 0; i < 8; i++)
                    bytes[1 + i] = char(bits >> (56 - 8 * i));
                buffer.write(bytes, sizeof(bytes));
            }
        };

        template <>
        struct IOTraits<MessagePackOutput>
        {
            static bool hasKey(MessagePackOutput& io, const char* key)
            {
                return io.hasKey(key);
            }

            template <typename T>
            static void mapRequired(MessagePackOutput& io, const char* key, T& obj)
            {
                io.mapRequired(key, obj);
            }

            template <typename T>
            static void mapOptional(MessagePackOutput& io, const char* key, T& obj)
            {
                io.mapOptional(key, obj);
            }

            template <typename T>
            static void mapOptional(MessagePackOutput& io,
                                    const char*        key,
                                    T&                 obj,
                                    T const&           defaultValue)
            {
                io.mapOptional(key, obj, defaultValue);
            }

            static bool mapRawBytes(MessagePackOutput& io,
                                    const char*        key,
                                    const uint8_t*&    ptr,
                                    size_t&            size)
            {
                return io.mapRawBytes(key, ptr, size);
            }

            template <typename MySolution>
            static std::function<std::shared_ptr<MySolution>(const uint8_t*, size_t)>
                solutionDeserializer(MessagePackOutput& io)
            {
                return {};
            }

            static bool outputting(MessagePackOutput& io)
            {
                return true;
            }

            static void setError(MessagePackOutput& io, std::string const& msg)
            {
                io.addError(msg);
            }

            static void setContext(MessagePackOutput& io, void* ctx)
            {
                io.context = ctx;
            }

            static void* getContext(MessagePackOutput& io)
            {
                return io.context;
            }

            template <typename T>
            static void enumCase(MessagePackOutput& io, T& member, const char* key, T value)
            {
                io.enumCase(member, key, value);
            }
        };

        template <typename T, typename IO>
        struct SequenceTraits<std::vector<T>, IO>
        {
            static size_t size(IO& io, std::vector<T>& v)
            {
                return v.size();
            }
            static T& element(IO& io, std::vector<T>& v, size_t index)
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
        struct MappingTraits<std::shared_ptr<TensileLite::MasterContractionLibrary>, IO>
        {
            using obj = MasterContractionLibrary;

            static void mapping(IO& io, std::shared_ptr<obj>& o)
            {
                PointerMappingTraits<obj, IO>::mapping(io, o);
            }
        };
    }
}
