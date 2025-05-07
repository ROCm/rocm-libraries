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

#include <Tensile/Serialization/Base.hpp>

#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ExactLogicLibrary.hpp>
#include <Tensile/FreeSizeLibrary.hpp>
#include <Tensile/GranularitySelectionLibrary.hpp>
#include <Tensile/MLPClassificationLibrary.hpp>
#include <Tensile/PropertyMatching.hpp>
#include <Tensile/ContractionTaskPredicates.hpp>

#include <cstddef>
#include <map>

namespace TensileLite
{
    namespace Serialization
    {
        template <typename Key>
        struct KeyConversion
        {
            static std::string toString(Key const& value)
            {
                return concatenate(value);
            }

            static Key fromString(std::string const& value)
            {
                std::istringstream stream(value);
                Key                rv;
                stream >> rv;

                return rv;
            }
        };

        template <>
        struct KeyConversion<std::string>
        {
            static std::string const& toString(std::string const& value)
            {
                return value;
            }

            static std::string const& fromString(std::string const& value)
            {
                return value;
            }
        };

        template <typename Map, typename IO, bool Sort, bool Flow>
        struct DefaultCustomMappingTraits
        {
            using iot         = IOTraits<IO>;
            using key_type    = typename Map::key_type;
            using mapped_type = typename Map::mapped_type;

            static void inputOne(IO& io, std::string const& keyStr, Map& value)
            {
                iot::mapRequired(
                    io, keyStr.c_str(), value[KeyConversion<key_type>::fromString(keyStr)]);
            }

            static void output(IO& io, Map& value)
            {
                if(Sort)
                {
                    std::vector<key_type> keys;
                    keys.reserve(value.size());
                    for(auto const& pair : value)
                        keys.push_back(pair.first);
                    std::sort(keys.begin(), keys.end());

                    for(auto const& key : keys)
                    {
                        auto keyStr = KeyConversion<key_type>::toString(key);
                        iot::mapRequired(io, keyStr.c_str(), value.find(key)->second);
                    }
                }
                else
                {
                    for(auto& pair : value)
                    {
                        auto keyStr = KeyConversion<key_type>::toString(pair.first);
                        iot::mapRequired(io, keyStr.c_str(), pair.second);
                    }
                }
            }

            static const bool flow = Flow;
        };

        template <typename IO>
        struct CustomMappingTraits<std::map<std::string, std::string>, IO>
            : public DefaultCustomMappingTraits<std::map<std::string, std::string>, IO, false, true>
        {
        };

        template <typename IO>
        struct CustomMappingTraits<std::map<std::string, double>, IO>
            : public DefaultCustomMappingTraits<std::map<std::string, double>, IO, false, true>
        {
        };

        template <typename IO>
        struct CustomMappingTraits<std::map<int, double>, IO>
            : public DefaultCustomMappingTraits<std::map<int, double>, IO, false, true>
        {
        };

        template <typename Seq, typename IO, bool Flow>
        struct DefaultSequenceTraits
        {
            using Value = typename Seq::value_type;

            static size_t size(IO& io, Seq& s)
            {
                return s.size();
            }
            static Value& element(IO& io, Seq& s, size_t index)
            {
                if(index >= s.size())
                {
                    size_t n = index - s.size() + 1;
                    s.insert(s.end(), n, Value());
                }

                return s[index];
            }

            const static bool flow = Flow;
        };

#define TENSILE_SERIALIZE_VECTOR(flow, ...)                                \
    template <typename IO>                                                 \
    struct SequenceTraits<std::vector<__VA_ARGS__>, IO>                    \
        : public DefaultSequenceTraits<std::vector<__VA_ARGS__>, IO, flow> \
    {                                                                      \
    }

        TENSILE_SERIALIZE_VECTOR(true, ExactSelectionTableEntry);

        // TENSILE_SERIALIZE_VECTOR(true, float);
        TENSILE_SERIALIZE_VECTOR(false, float);
        TENSILE_SERIALIZE_VECTOR(false, TensileLite::Half);

        TENSILE_SERIALIZE_VECTOR(true,
                                 TensileLite::ExactLogicLibrary<TensileLite::ContractionProblemGemm,
                                                            TensileLite::ContractionSolution,
                                                            TensileLite::HardwarePredicate>::Row);

        TENSILE_SERIALIZE_VECTOR(
            true,
            TensileLite::ExactLogicLibrary<TensileLite::ContractionProblemGemm,
                                       TensileLite::ContractionSolution,
                                       TensileLite::ProblemPredicate<ContractionProblemGemm>>::Row);

        TENSILE_SERIALIZE_VECTOR(
            true, std::shared_ptr<TensileLite::Predicates::Predicate<TensileLite::ContractionProblemGemm>>);
        TENSILE_SERIALIZE_VECTOR(
            true, std::shared_ptr<TensileLite::Predicates::Predicate<TensileLite::Hardware>>);
        TENSILE_SERIALIZE_VECTOR(true,
                                 std::shared_ptr<TensileLite::Predicates::Predicate<TensileLite::AMDGPU>>);
        TENSILE_SERIALIZE_VECTOR(true,
                                 std::shared_ptr<TensileLite::Predicates::Predicate<TensileLite::Task>>);
        TENSILE_SERIALIZE_VECTOR(
            true, std::shared_ptr<TensileLite::Property<TensileLite::ContractionProblemGemm>>);
        TENSILE_SERIALIZE_VECTOR(true, std::shared_ptr<TensileLite::Property<TensileLite::Task>>);
        TENSILE_SERIALIZE_VECTOR(
            true, std::shared_ptr<TensileLite::MLFeatures::MLFeature<TensileLite::ContractionProblemGemm>>);

        TENSILE_SERIALIZE_VECTOR(false, std::shared_ptr<TensileLite::ContractionSolution>);

        TENSILE_SERIALIZE_VECTOR(true, TensileLite::MLPClassification::ResBlock);

        template <typename Value, typename IO>
        struct SequenceTraits<std::vector<TensileLite::FreeSizeEntry<Value>>, IO>
            : public DefaultSequenceTraits<std::vector<TensileLite::FreeSizeEntry<Value>>, IO, false>
        {
        };

        template <typename Key, typename Value, typename IO>
        struct SequenceTraits<std::vector<TensileLite::Matching::MatchingTableEntry<Key, Value>>, IO>
            : public DefaultSequenceTraits<
                  std::vector<TensileLite::Matching::MatchingTableEntry<Key, Value>>,
                  IO,
                  false>
        {
        };

        template <typename T, size_t N, typename IO>
        struct SequenceTraits<std::array<T, N>, IO>
        {
            using Value = T;
            static size_t size(IO& io, std::array<T, N>& v)
            {
                return N;
            }
            static T& element(IO& io, std::array<T, N>& v, size_t index)
            {
                if(index >= N)
                {
                    IOTraits<IO>::setError(io,
                                           concatenate("invalid array<T, ", N, "> index ", index));
                }

                return v[index];
            }

            const static bool flow = (N < 10);
        };
    } // namespace Serialization
} // namespace TensileLite
