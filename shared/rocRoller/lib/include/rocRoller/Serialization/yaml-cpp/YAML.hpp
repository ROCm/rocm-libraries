/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2019-2025 AMD ROCm(TM) Software
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
                using MyTraits = MappingTraits<T, EmitterOutput>;

                if constexpr(CHasFlow<MyTraits>)
                    emitter->SetMapFormat(YAML::Flow);
                else
                    emitter->SetMapFormat(YAML::Block);

                *emitter << YAML::BeginMap;
                EmptyContext ctx;
                MyTraits::mapping(*this, obj, ctx);
                *emitter << YAML::EndMap;
            }

            template <EmptyMappedType<EmitterOutput> T>
            void output(T& obj)
            {
                using MyTraits = MappingTraits<T, EmitterOutput>;
                if constexpr(CHasFlow<MyTraits>)
                    emitter->SetMapFormat(YAML::Flow);
                else
                    emitter->SetMapFormat(YAML::Block);
                *emitter << YAML::BeginMap;
                EmptyContext ctx;
                MyTraits::mapping(*this, obj, ctx);
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
                if constexpr(CHasFlow<ST>)
                    emitter->SetSeqFormat(YAML::Flow);
                else
                    emitter->SetSeqFormat(YAML::Block);

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
                using MyTraits = CustomMappingTraits<T, EmitterOutput>;

                if constexpr(CHasFlow<MyTraits>)
                    emitter->SetMapFormat(YAML::Flow);
                else
                    emitter->SetMapFormat(YAML::Block);
                *emitter << YAML::BeginMap;
                MyTraits::output(*this, obj, nullptr);
                *emitter << YAML::EndMap;
            }

            template <CHasScalarTraits T>
            void output(T& obj)
            {
                *emitter << ScalarTraits<T>::output(obj);
            }

            constexpr bool outputting() const
            {
                return true;
            }
        };

        template <>
        inline void EmitterOutput::output(FP8& obj)
        {
            std::stringstream ss;
            ss << obj;
            *emitter << ss.str();
        }

        template <>
        inline void EmitterOutput::output(BF8& obj)
        {
            std::stringstream ss;
            ss << obj;
            *emitter << ss.str();
        }

        template <>
        inline void EmitterOutput::output(FP6& obj)
        {
            std::stringstream ss;
            ss << obj;
            *emitter << ss.str();
        }

        template <>
        inline void EmitterOutput::output(BF6& obj)
        {
            std::stringstream ss;
            ss << obj;
            *emitter << ss.str();
        }

        template <>
        inline void EmitterOutput::output(FP4& obj)
        {
            std::stringstream ss;
            ss << obj;
            *emitter << ss.str();
        }

        template <>
        inline void EmitterOutput::output(BFloat16& obj)
        {
            std::stringstream ss;
            ss << obj;
            *emitter << ss.str();
        }

        template <>
        inline void EmitterOutput::output(E8M0& obj)
        {
            std::stringstream ss;
            ss << obj;
            *emitter << ss.str();
        }

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

        template <typename T>
        inline void nodeInputHelper(YAML::Node& n, T& obj)
        {
            obj = n.as<T>();
        }

        template <>
        inline void nodeInputHelper(YAML::Node& n, Half& val)
        {
            float floatVal;
            nodeInputHelper(n, floatVal);
            val = floatVal;
        }

        template <>
        inline void nodeInputHelper(YAML::Node& n, BFloat16& val)
        {
            float floatVal;
            nodeInputHelper(n, floatVal);
            val.data = floatVal;
        }

        struct NodeInput
        {
            YAML::Node* node;
            void*       context;

            NodeInput(YAML::Node* n, void* c = nullptr)
                : node(n)
                , context(c)
            {
            }

            template <typename T>
            void mapRequired(const char* key, T& obj)
            {
                auto subnode = (*node)[key];
                AssertFatal(subnode, "Key ", ShowValue(key), " not found: ", YAML::Dump(*node));
                input(subnode, obj);
            }

            template <typename T>
            void mapOptional(const char* key, T& obj)
            {
                auto subnode = (*node)[key];
                if(subnode)
                    input(subnode, obj);
            }

            template <typename T>
            requires(CMappedType<T, NodeInput> || EmptyMappedType<T, NodeInput>) void input(
                YAML::Node& n, T& obj)
            {
                NodeInput    subInput(&n, context);
                EmptyContext ctx;
                MappingTraits<T, NodeInput>::mapping(subInput, obj, ctx); //, context);
            }

            template <typename T>
            void input(YAML::Node& n, T& obj)
            {
                nodeInputHelper(n, obj);
            }

            template <SequenceType<NodeInput> T>
            void input(YAML::Node& n, T& obj)
            {
                auto count = n.size();

                for(size_t i = 0; i < count; i++)
                {
                    auto& value  = SequenceTraits<T, NodeInput>::element(*this, obj, i);
                    auto  elNode = n[i];
                    input(elNode, value);
                }
            }

            template <CustomMappingType<NodeInput> T>
            void input(YAML::Node& n, T& obj)
            {
                NodeInput subInput(&n, context);
                for(auto pair : n)
                {
                    CustomMappingTraits<T, NodeInput>::inputOne(subInput, pair.first.Scalar(), obj);
                }
            }

            template <CHasScalarTraits T>
            void input(YAML::Node& n, T& obj)
            {
                std::string stringVal;
                input(n, stringVal);
                ScalarTraits<T>::input(stringVal, obj);
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
                if(io.node->Scalar() == key)
                {
                    member = value;
                }
            }
        };

    } // namespace Serialization
} // namespace rocRoller

namespace YAML
{
    template <>
    struct convert<rocRoller::BFloat16>
    {
        static Node encode(const rocRoller::BFloat16& rhs)
        {
            Node node;
            node.push_back(rhs.data);
            return node;
        }

        static bool decode(const Node& node, rocRoller::BFloat16& rhs)
        {
            if(!node.IsSequence() || node.size() != 1)
            {
                return false;
            }

            rhs.data = node[0].as<decltype(rhs.data)>();
            return true;
        }
    };

    template <>
    struct convert<rocRoller::FP8>
    {
        static Node encode(const rocRoller::FP8& rhs)
        {
            Node node;
            node.push_back(rhs.data);
            return node;
        }

        static bool decode(const Node& node, rocRoller::FP8& rhs)
        {
            if(!node.IsSequence() || node.size() != 1)
            {
                return false;
            }

            rhs.data = node[0].as<decltype(rhs.data)>();
            return true;
        }
    };

    template <>
    struct convert<rocRoller::BF8>
    {
        static Node encode(const rocRoller::BF8& rhs)
        {
            Node node;
            node.push_back(rhs.data);
            return node;
        }

        static bool decode(const Node& node, rocRoller::BF8& rhs)
        {
            if(!node.IsSequence() || node.size() != 1)
            {
                return false;
            }

            rhs.data = node[0].as<decltype(rhs.data)>();
            return true;
        }
    };

    template <>
    struct convert<rocRoller::FP6>
    {
        static Node encode(const rocRoller::FP6& rhs)
        {
            Node node;
            node.push_back(rhs.data);
            return node;
        }

        static bool decode(const Node& node, rocRoller::FP6& rhs)
        {
            if(!node.IsSequence() || node.size() != 1)
            {
                return false;
            }

            rhs.data = node[0].as<decltype(rhs.data)>();
            return true;
        }
    };

    template <>
    struct convert<rocRoller::BF6>
    {
        static Node encode(const rocRoller::BF6& rhs)
        {
            Node node;
            node.push_back(rhs.data);
            return node;
        }

        static bool decode(const Node& node, rocRoller::BF6& rhs)
        {
            if(!node.IsSequence() || node.size() != 1)
            {
                return false;
            }

            rhs.data = node[0].as<decltype(rhs.data)>();
            return true;
        }
    };

    template <>
    struct convert<rocRoller::FP4>
    {
        static Node encode(const rocRoller::FP4& rhs)
        {
            Node node;
            node.push_back(rhs.data);
            return node;
        }

        static bool decode(const Node& node, rocRoller::FP4& rhs)
        {
            if(!node.IsSequence() || node.size() != 1)
            {
                return false;
            }

            rhs.data = node[0].as<decltype(rhs.data)>();
            return true;
        }
    };

    template <>
    struct convert<rocRoller::E8M0>
    {
        static Node encode(const rocRoller::E8M0& rhs)
        {
            Node node;
            node.push_back(rhs.scale);
            return node;
        }

        static bool decode(const Node& node, rocRoller::E8M0& rhs)
        {
            if(!node.IsSequence() || node.size() != 1)
            {
                return false;
            }

            rhs.scale = node[0].as<decltype(rhs.scale)>();
            return true;
        }
    };

} // namespace YAML
