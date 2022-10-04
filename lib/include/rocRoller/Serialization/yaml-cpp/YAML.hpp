/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2022 Advanced Micro Devices, Inc.
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

#include <yaml-cpp/yaml.h>

#include <cstddef>

namespace rocRoller
{
    namespace Serialization
    {
        struct EmitterOutput
        {
            YAML::Emitter* emitter;
            void*          context;

            void mapRequired();

            EmitterOutput(YAML::Emitter* e, void* c = nullptr)
                : emitter(e)
                , context(c)
            {
            }

            ~EmitterOutput() {}

            template <typename T>
            void mapRequired(const char* key, T& obj)
            {
                *emitter << YAML::Key << key << YAML::Value;
                output(obj);
            }

            template <typename T>
            void mapOptional(const char* key, T& obj)
            {
                mapRequired(key, obj);
            }

            template <typename T>
            void outputDoc(T& obj)
            {
                *emitter << YAML::BeginDoc;
                output(obj);
                *emitter << YAML::EndDoc;
            }

            template <CMappedType<EmitterOutput> T>
            void output(T& obj)
            {
                *emitter << YAML::BeginMap;
                EmptyContext ctx;
                MappingTraits<T, EmitterOutput>::mapping(*this, obj, ctx);
                *emitter << YAML::EndMap;
            }

            template <EmptyMappedType<EmitterOutput> T>
            void output(T& obj)
            {
                *emitter << YAML::BeginMap;
                EmptyContext ctx;
                MappingTraits<T, EmitterOutput>::mapping(*this, obj, ctx);
                *emitter << YAML::EndMap;
            }

            template <ValueType<EmitterOutput> T>
            void output(T& obj)
            {
                *emitter << obj;
            }

            template <SequenceType<EmitterOutput> T>
            void output(T& obj)
            {
                using ST = SequenceTraits<T, EmitterOutput>;

                auto count = ST::size(*this, obj);
                *emitter << YAML::BeginSeq;

                for(size_t i = 0; i < count; i++)
                {
                    auto& value = ST::element(*this, obj, i);
                    output(value);
                }

                *emitter << YAML::EndSeq;
            }

            template <CustomMappingType<EmitterOutput> T>
            void output(T& obj)
            {
                *emitter << YAML::BeginMap;
                CustomMappingTraits<T, EmitterOutput>::output(*this, obj, nullptr);
                *emitter << YAML::EndMap;
            }

            template <EnumType<EmitterOutput> T>
            void output(T& obj)
            {
                *emitter << ToString(obj);
            }

            constexpr bool outputting() const
            {
                return true;
            }
        };

        template <>
        struct IOTraits<EmitterOutput>
        {
            using IO = EmitterOutput;

            template <typename T>
            static void mapRequired(IO& io, const char* key, T& obj)
            {
                io.mapRequired(key, obj);
            }

            template <typename T>
            static void mapRequired(IO& io, const char* key, T const& obj)
            {
                AssertFatal(outputting(io));
                T tmp = obj;
                io.mapRequired(key, tmp);
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

            static bool outputting(IO& io)
            {
                return io.outputting();
            }

            static void setError(IO& io, std::string const& msg)
            {
                throw std::runtime_error(msg);
            }

            static void setContext(IO& io, void* ctx)
            {
                // Serialization context is primarily for input and not supported yet.
            }

            static void* getContext(IO& io)
            {
                return nullptr;
            }

            template <typename T>
            static void enumCase(IO& io, T& member, const char* key, T value)
            {
                // Enumerations are handled more directly when outputting.
            }
        };

        struct NodeInput
        {
            YAML::Node* m_node;
            void*       context;

            void mapRequired();

            NodeInput(YAML::Node* n, void* c = nullptr)
                : m_node(n)
                , context(c)
            {
            }

            template <typename T>
            void mapRequired(const char* key, T& obj)
            {
                auto subnode = (*m_node)[key];
                AssertFatal(subnode, "Key ", ShowValue(key), " not found: ", YAML::Dump(*m_node));
                input(subnode, obj);
            }

            template <typename T>
            void mapOptional(const char* key, T& obj)
            {
                auto subnode = (*m_node)[key];
                if(subnode)
                    input(subnode, obj);
            }

            template <typename T>
            requires(CMappedType<T, NodeInput> || EmptyMappedType<T, NodeInput>) void input(
                YAML::Node& node, T& obj)
            {
                NodeInput    subInput(&node, context);
                EmptyContext ctx;
                MappingTraits<T, NodeInput>::mapping(subInput, obj, ctx); //, context);
            }

            template <typename T>
            void input(YAML::Node& node, T& obj)
            {
                obj = node.as<T>();
            }

            template <>
            void input(YAML::Node& node, Half& val)
            {
                float floatVal;
                input(node, floatVal);
                val = floatVal;
            }

            template <SequenceType<NodeInput> T>
            void input(YAML::Node& node, T& obj)
            {
                auto count = node.size();

                for(size_t i = 0; i < count; i++)
                {
                    auto& value  = SequenceTraits<T, NodeInput>::element(*this, obj, i);
                    auto  elNode = node[i];
                    input(elNode, value);
                }
            }

            template <CustomMappingType<NodeInput> T>
            void input(YAML::Node& node, T& obj)
            {
                NodeInput subInput(&node, context);
                for(auto pair : node)
                {
                    CustomMappingTraits<T, NodeInput>::inputOne(subInput, pair.first.Scalar(), obj);
                }
            }

            template <EnumType<NodeInput> T>
            void input(YAML::Node& node, T& obj)
            {
                NodeInput subInput(&node, context);
                EnumTraits<T, NodeInput>::enumeration(subInput, obj);
            }

            constexpr bool outputting() const
            {
                return false;
            }
        };

        template <>
        struct IOTraits<NodeInput>
        {
            using IO = NodeInput;

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
                // Serialization context is primarily for input and not supported yet.
            }

            static void* getContext(IO& io)
            {
                return nullptr;
            }

            template <typename T>
            static void enumCase(IO& io, T& member, const char* key, T value)
            {
                if(io.m_node->Scalar() == key)
                {
                    member = value;
                }
            }
        };

    } // namespace Serialization
} // namespace rocRoller
