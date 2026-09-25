// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <Tensile/ContractionLibrary.hpp>
#include <Tensile/Serialization.hpp>

#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace TensileLite
{
    namespace Serialization
    {
        template <typename T>
        struct is_YamlCppConvertible : std::false_type
        {
        };

        template <typename T>
        struct is_StdVector : std::false_type
        {
        };

        template <typename T, typename Allocator>
        struct is_StdVector<std::vector<T, Allocator>> : std::true_type
        {
        };

#define TENSILE_YAML_CPP_CONVERTIBLE(T)              \
    template <>                                      \
    struct is_YamlCppConvertible<T> : std::true_type \
    {                                                \
    }

        TENSILE_YAML_CPP_CONVERTIBLE(std::string);
        TENSILE_YAML_CPP_CONVERTIBLE(bool);
        TENSILE_YAML_CPP_CONVERTIBLE(int8_t);
        TENSILE_YAML_CPP_CONVERTIBLE(int16_t);
        TENSILE_YAML_CPP_CONVERTIBLE(int32_t);
        TENSILE_YAML_CPP_CONVERTIBLE(int64_t);
        TENSILE_YAML_CPP_CONVERTIBLE(uint8_t);
        TENSILE_YAML_CPP_CONVERTIBLE(uint16_t);
        TENSILE_YAML_CPP_CONVERTIBLE(uint32_t);
        TENSILE_YAML_CPP_CONVERTIBLE(uint64_t);
        TENSILE_YAML_CPP_CONVERTIBLE(float);
        TENSILE_YAML_CPP_CONVERTIBLE(double);

#undef TENSILE_YAML_CPP_CONVERTIBLE

        template <typename T, typename IO>
        struct SequenceTraits<std::vector<T>, IO>
        {
            static size_t size(IO&, std::vector<T>& value)
            {
                return value.size();
            }

            static T& element(IO&, std::vector<T>& value, size_t index)
            {
                if(index >= value.size())
                    value.resize(index + 1);
                return value[index];
            }
        };

        struct YamlCppInput
        {
            YAML::Node                      node;
            std::vector<std::string>        error;
            std::unordered_set<std::string> usedKeys;
            int                             enumFound = 0;
            void*                           context   = nullptr;

            explicit YamlCppInput(YAML::Node node, void* context = nullptr)
                : node(std::move(node))
                , context(context)
            {
            }

            YamlCppInput createSubRef(YAML::Node const& child) const
            {
                return YamlCppInput(child, context);
            }

            void addError(std::string const& message, YAML::Node const& where = YAML::Node())
            {
                std::ostringstream msg;
                if(where.IsDefined() && !where.Mark().is_null())
                    msg << where.Mark().line + 1 << ':' << where.Mark().column + 1 << ": ";
                msg << message;
                error.push_back(msg.str());
            }

            void appendErrors(YamlCppInput const& child)
            {
                error.insert(error.end(), child.error.begin(), child.error.end());
            }

            template <typename T>
            void mapRequired(const char* key, T& value)
            {
                if(!node.IsMap())
                {
                    addError("expected a mapping", node);
                    return;
                }

                auto child = node[key];
                if(!child.IsDefined())
                {
                    addError(concatenate("missing required key '", key, "'"), node);
                    return;
                }

                usedKeys.insert(key);
                auto input = createSubRef(child);
                input.input(value);
                appendErrors(input);
            }

            template <typename T, typename Context>
            void mapRequired(const char* key, T& value, Context& context)
            {
                if(!node.IsMap())
                {
                    addError("expected a mapping", node);
                    return;
                }

                auto child = node[key];
                if(!child.IsDefined())
                {
                    addError(concatenate("missing required key '", key, "'"), node);
                    return;
                }

                usedKeys.insert(key);
                auto input = createSubRef(child);
                input.input(value, context);
                appendErrors(input);
            }

            template <typename T>
            void mapOptional(const char* key, T& value)
            {
                if(!node.IsMap())
                {
                    addError("expected a mapping", node);
                    return;
                }

                auto child = node[key];
                if(child.IsDefined())
                {
                    usedKeys.insert(key);
                    auto input = createSubRef(child);
                    input.input(value);
                    appendErrors(input);
                }
            }

            template <typename T, typename Context>
            void mapOptional(const char* key, T& value, Context& context)
            {
                if(!node.IsMap())
                {
                    addError("expected a mapping", node);
                    return;
                }

                auto child = node[key];
                if(child.IsDefined())
                {
                    usedKeys.insert(key);
                    auto input = createSubRef(child);
                    input.input(value, context);
                    appendErrors(input);
                }
            }

            template <typename T>
            void input(T& value)
            {
                EmptyContext context;
                input(value, context);
            }

            void checkUsedKeys()
            {
                if(!node.IsMap())
                    return;

                for(auto const& entry : node)
                {
                    auto key = entry.first.as<std::string>();
                    if(usedKeys.count(key) == 0)
                        addError(concatenate("unknown key '", key, "'"), entry.first);
                }
            }

            template <typename T, typename Context>
            typename std::enable_if<has_MappingTraits<T, YamlCppInput, Context>::value, void>::type
                input(T& value, Context& context)
            {
                MappingTraits<T, YamlCppInput, Context>::mapping(*this, value, context);
                checkUsedKeys();
            }

            template <typename T, typename Context>
            typename std::enable_if<has_EmptyMappingTraits<T, YamlCppInput, Context>::value,
                                    void>::type
                input(T& value, Context&)
            {
                MappingTraits<T, YamlCppInput, Context>::mapping(*this, value);
                checkUsedKeys();
            }

            template <typename T, typename Context>
            typename std::enable_if<is_YamlCppConvertible<T>::value, void>::type input(T& value,
                                                                                       Context&)
            {
                try
                {
                    value = node.as<T>();
                }
                catch(YAML::Exception const& exception)
                {
                    addError(exception.msg, node);
                }
            }

            template <typename T, typename Context>
            typename std::enable_if<has_EnumTraits<T, YamlCppInput>::value, void>::type
                input(T& value, Context&)
            {
                enumFound = 0;
                EnumTraits<T, YamlCppInput>::enumeration(*this, value);
                if(enumFound != 1)
                    addError(concatenate("unknown enumerated scalar '", node.Scalar(), "'"), node);
            }

            template <typename T, typename Context>
            typename std::enable_if<has_SequenceTraits<T, YamlCppInput>::value, void>::type
                input(T& value, Context&)
            {
                if(!node.IsSequence())
                {
                    addError("expected a sequence", node);
                    return;
                }

                if(!is_StdVector<T>::value
                   && node.size() > SequenceTraits<T, YamlCppInput>::size(*this, value))
                {
                    addError(concatenate("unexpected sequence length ", node.size()), node);
                    return;
                }

                for(size_t index = 0; index < node.size(); ++index)
                {
                    auto  child   = createSubRef(node[index]);
                    auto& element = SequenceTraits<T, YamlCppInput>::element(*this, value, index);
                    child.input(element);
                    appendErrors(child);
                    if(!child.error.empty())
                        return;
                }
            }

            template <typename T, typename Context>
            typename std::enable_if<has_CustomMappingTraits<T, YamlCppInput>::value, void>::type
                input(T& value, Context&)
            {
                if(!node.IsMap())
                {
                    addError("expected a mapping", node);
                    return;
                }

                for(auto const& entry : node)
                {
                    auto key = entry.first.as<std::string>();
                    CustomMappingTraits<T, YamlCppInput>::inputOne(*this, key, value);
                }
            }

            template <typename T>
            void enumCase(T& member, const char* key, T value)
            {
                if(node.IsScalar() && node.Scalar() == key)
                {
                    ++enumFound;
                    member = value;
                }
            }
        };

        template <>
        struct IOTraits<YamlCppInput>
        {
            template <typename T>
            static void mapRequired(YamlCppInput& io, const char* key, T& value)
            {
                io.mapRequired(key, value);
            }

            template <typename T, typename Context>
            static void mapRequired(YamlCppInput& io, const char* key, T& value, Context& context)
            {
                io.mapRequired(key, value, context);
            }

            template <typename T>
            static void mapOptional(YamlCppInput& io, const char* key, T& value)
            {
                io.mapOptional(key, value);
            }

            template <typename T, typename Context>
            static void mapOptional(YamlCppInput& io, const char* key, T& value, Context& context)
            {
                io.mapOptional(key, value, context);
            }

            static bool mapRawBytes(YamlCppInput&, const char*, const uint8_t*&, size_t&)
            {
                return false;
            }

            template <typename MySolution>
            static std::function<std::shared_ptr<MySolution>(const uint8_t*, size_t)>
                solutionDeserializer(YamlCppInput&)
            {
                return {};
            }

            static bool outputting(YamlCppInput&)
            {
                return false;
            }

            static void setError(YamlCppInput& io, std::string const& message)
            {
                io.addError(message, io.node);
            }

            static void setContext(YamlCppInput& io, void* context)
            {
                io.context = context;
            }

            static void* getContext(YamlCppInput& io)
            {
                return io.context;
            }

            template <typename T>
            static void enumCase(YamlCppInput& io, T& member, const char* key, T value)
            {
                io.enumCase(member, key, value);
            }
        };

        template <>
        struct MappingTraits<std::shared_ptr<TensileLite::MasterContractionLibrary>, YamlCppInput>
        {
            using Object = MasterContractionLibrary;

            static void mapping(YamlCppInput& io, std::shared_ptr<Object>& value)
            {
                PointerMappingTraits<Object, YamlCppInput>::mapping(io, value);
            }
        };
    } // namespace Serialization
} // namespace TensileLite
