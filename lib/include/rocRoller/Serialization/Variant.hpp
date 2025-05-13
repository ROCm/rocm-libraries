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

#include <variant>

#include <rocRoller/Serialization/Base.hpp>

namespace rocRoller
{
    namespace Serialization
    {
        template <typename T>
        concept CVariant = requires
        {
            // clang-format off
             { std::variant_size<T>::value } -> std::convertible_to<size_t>;
            // clang-format on
        };

        static_assert(CVariant<std::variant<int, float>>);
        static_assert(!CVariant<std::string>);

        template <typename T>
        concept CNamedVariant = requires(T const& t)
        {
            // clang-format off
            requires CVariant<T>;
            { name(t) } -> std::convertible_to<std::string>;
            // clang-format on
        };

        template <typename T>
        struct VariantTypeKeySpecifier
        {
            static std::string TypeKey()
            {
                return "type";
            }
        };

        /**
         * For a variant whose alternatives are not variant, returns name(v).
         *
         * If the alternative held in `v` is also a variant, returns a '.'-separated
         * path of the name of each alternative in the hierarchy.
         */
        template <CNamedVariant Var>
        std::string typePath(Var const& v);

        /**
         * Splits `input` into two pieces:
         *  - The first is `input` up until the first occurrence of `delim`,
         *    or the entire input if `delim` is not in `input`.
         *  - The second is everything after the first occurrence of `delim`,
         *    or an empty string if `delim` is not in `input`.
         */
        inline std::tuple<std::string, std::string> splitFirst(std::string const& delim,
                                                               std::string const& input)
        {
            auto delimPos = input.find(delim);
            if(delimPos == std::string::npos)
                return {input, ""};

            auto delimEnd = delimPos + delim.size();

            return {input.substr(0, delimPos), input.substr(delimEnd)};
        }

        /**
         * Serializes an instantiation of std::variant with a 'type' field that redirects to each alternative of the
         * variant.
         *
         * In addition to the CNamedVariant requirements, each alternative must be serializable.
         *
         * If the type field should be named something other than 'type', or if some alternatives are not serializable,
         * the subclass can reimplement `mapping` as appropriate, or specialize VariantTypeKeySpecifier.
         *
         * For a hierarchy of variants, this will automatically group the keys using typePath.  This allows only one
         * key which contains the types of every variant alternative in path (eg. Edge.DataFlow.Offset).
         *
         * This mechanism uses the context mechanism of the serialization framework.  The object passed is a std::string.
         * Each level of the hierarchy will consume the first token of the string (everthing up until the first `.`)
         * and pass the rest on to the next level.  Once the string is empty, it will revert to EmptyContext. This also
         * allows the Adhoc dimension to use this for its `m_name` field.
         */
        template <CNamedVariant T, typename IO, typename Context = EmptyContext>
        struct DefaultVariantMappingTraits
        {
            using iot            = IOTraits<IO>;
            using AlternativeFn  = std::function<T(void)>;
            using AlternativeMap = std::map<std::string, AlternativeFn>;

            static constexpr AlternativeMap GetAlternatives()
            {
                AlternativeMap rv;
                AddAlternatives<0>(rv);
                return rv;
            }

            const static AlternativeMap alternatives;

            template <int AltIdx>
            static constexpr auto GetAlternative()
            {
                using AltType = std::variant_alternative_t<AltIdx, T>;
                auto typeName = name(DefaultConstruct<T, AltType>::call());
                return std::make_pair(typeName, DefaultConstruct<T, AltType>::call);
            }

            template <int AltIdx = 0>
            static void AddAlternatives(AlternativeMap& alts)
            {
                if constexpr(AltIdx < std::variant_size_v<T>)
                {
                    auto alt = GetAlternative<AltIdx>();
                    AssertFatal(
                        alts.count(alt.first) == 0, "Duplicate alternative names: ", alt.first);

                    alts.insert(alt);
                    AddAlternatives<AltIdx + 1>(alts);
                }
            }

            static void mapping(IO& io, T& exp, Context& ctx)
            {
                // Only Empty or String is supported.
                static_assert(
                    std::same_as<Context, EmptyContext> || std::same_as<Context, std::string>);
                std::string myTypePath;
                // std::string typeName;
                std::string typeKey = VariantTypeKeySpecifier<T>::TypeKey();

                std::optional<std::string> remainingCtx;

                bool mapTypeName = true;

                if(iot::outputting(io))
                {
                    myTypePath = typePath(exp);
                }

                if constexpr(std::same_as<Context, std::string>)
                {
                    if(iot::outputting(io))
                        AssertFatal(ctx == myTypePath);
                    else
                        myTypePath = ctx;
                }
                else
                {
                    iot::mapRequired(io, typeKey.c_str(), myTypePath);
                }

                auto [typeName, rest] = splitFirst(".", myTypePath);

                if(!iot::outputting(io))
                {
                    exp = alternatives.at(typeName)();
                }

                if(!rest.empty())
                    remainingCtx = rest;

                std::visit(
                    [&io, &remainingCtx](auto& theExp) {
                        using U = std::decay_t<decltype(theExp)>;
                        if(remainingCtx)
                        {
                            MappingTraits<U, IO, std::string>::mapping(io, theExp, *remainingCtx);
                        }
                        else
                        {
                            MappingTraits<U, IO, EmptyContext>::mapping(io, theExp);
                        }
                    },
                    exp);
            }

            static void mapping(IO& io, T& exp)
            {
                AssertFatal((std::same_as<Context, EmptyContext>));

                Context ctx;
                return mapping(io, exp, ctx);
            }
        };

        template <CNamedVariant T, typename IO, typename Context>
        const typename DefaultVariantMappingTraits<T, IO, Context>::AlternativeMap
            DefaultVariantMappingTraits<T, IO, Context>::alternatives
            = DefaultVariantMappingTraits<T, IO, Context>::GetAlternatives();

        struct RemainingTypePathVisitor
        {
            template <CNamedVariant Var>
            std::string operator()(Var const& v)
            {
                return typePath(v);
            }

            template <typename T>
            requires(!CVariant<T>) std::string operator()(T const&)
            {
                return "";
            }
        };

        template <CNamedVariant Var>
        std::string typePath(Var const& v)
        {
            auto myName = name(v);

            auto rest = std::visit(RemainingTypePathVisitor{}, v);

            if(rest.empty())
                return myName;

            return myName + "." + rest;
        }

    }
}
