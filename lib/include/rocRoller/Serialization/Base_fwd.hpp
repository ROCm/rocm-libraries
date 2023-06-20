
#pragma once

namespace rocRoller
{
    namespace Serialization
    {
        template <typename IO>
        struct IOTraits;

        template <typename T, typename IO, typename Context>
        struct MappingTraits;

        template <typename T, typename IO>
        struct EnumTraits;

        template <typename T, typename IO>
        struct CustomMappingTraits;

        template <typename T, typename IO>
        struct SequenceTraits;

        template <typename T, typename IO>
        struct EnumTraits;

        template <typename Object, typename IO, typename Context>
        struct EmptyMappingTraits;
    }

}
