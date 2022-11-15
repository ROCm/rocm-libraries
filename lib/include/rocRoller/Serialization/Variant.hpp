#pragma once

#include <variant>

#include "Base.hpp"

// delete this when graph rearch complete
#include <rocRoller/Expression.hpp>

namespace rocRoller
{
    namespace Serialization
    {
        template <typename T>
        concept CNamedVariant = requires(T const& t)
        {
            // clang-format off
            requires (std::variant_size_v<T> > 0);

            { name(t) } -> std::convertible_to<std::string>;
            // clang-format on
        };

        /**
         * Serializes an instantiation of std::variant with a 'type' field that redirects to each alternative of the
         * variant.
         *
         * In addition to the CNamedVariant requirements, each alternative must be serializable.
         *
         * If the type field should be named something other than 'type', or if some alternatives are not
         * serializable, the subclass can reimplement `mapping` as appropriate.
         */
        template <CNamedVariant T, typename IO, typename Context = EmptyContext>
        struct DefaultVariantMappingTraits
        {
            using iot            = IOTraits<IO>;
            using AlternativeFn  = std::function<T(void)>;
            using AlternativeMap = std::map<std::string, AlternativeFn>;

            template <typename U>
            static T defaultConstruct()
            {
                return U();
            }

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
                auto typeName = name(defaultConstruct<AltType>());
                // delete this when graph rearch complete
                if(typeid(AltType) == typeid(rocRoller::Expression::WaveTilePtr2))
                {
                    return std::make_pair(typeName + "2", defaultConstruct<AltType>);
                }
                return std::make_pair(typeName, defaultConstruct<AltType>);
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
                std::string typeName;

                if(iot::outputting(io))
                {
                    typeName = name(exp);
                }

                iot::mapRequired(io, "type", typeName);

                if(!iot::outputting(io))
                {
                    exp = alternatives.at(typeName)();
                }

                std::visit(
                    [&io, &ctx](auto& theExp) {
                        using U = std::decay_t<decltype(theExp)>;
                        MappingTraits<U, IO, Context>::mapping(io, theExp, ctx);
                    },
                    exp);
            }

            static void mapping(IO& io, T& exp)
            {
                Context ctx;
                return mapping(io, exp, ctx);
            }
        };
    }
}
