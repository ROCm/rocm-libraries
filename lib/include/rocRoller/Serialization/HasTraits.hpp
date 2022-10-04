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

#include <cstddef>

namespace rocRoller
{
    namespace Serialization
    {
        template <typename T, T>
        struct SameType;

        template <typename T, typename IO, typename Context = EmptyContext>
        concept CMappedType = requires(T& obj, IO& io, Context& ctx)
        {
            {MappingTraits<T, IO, Context>::mapping(io, obj, ctx)};
        };

        template <typename T, typename IO, typename Context = EmptyContext>
        concept CEmptyMappedType = requires(T& obj, IO& io)
        {
            {MappingTraits<T, IO, Context>::mapping(io, obj)};
        };

        template <typename T, typename IO, typename Context = EmptyContext>
        struct has_MappingTraits
        {
            static const bool value = CMappedType<T, IO, Context>;
        };

        template <typename T, typename IO, typename Context = EmptyContext>
        class has_EmptyMappingTraits
        {
            using mapping = void (*)(IO&, T&);

            template <typename U>
            static uint8_t test(SameType<mapping, &U::mapping>*);

            template <typename U>
            static uint32_t test(...);

        public:
            static const bool value = sizeof(test<MappingTraits<T, IO, Context>>(nullptr)) == 1;
        };

        template <typename T, typename IO, typename Context = EmptyContext>
        concept EmptyMappedType = has_EmptyMappingTraits<T, IO, Context>::
            value && !has_MappingTraits<T, IO, Context>::value;

        template <typename T, typename IO>
        class has_EnumTraits
        {
            using enumeration = void (*)(IO&, T&);

            template <typename U>
            static uint8_t test(SameType<enumeration, &U::enumeration>*);

            template <typename U>
            static uint32_t test(...);

        public:
            static const bool value = sizeof(test<EnumTraits<T, IO>>(nullptr)) == 1;
        };

        template <typename T, typename IO>
        concept EnumType = has_EnumTraits<T, IO>::value;

        template <typename T, typename IO>
        class has_SequenceTraits
        {
            using size = size_t (*)(IO&, T&);

            template <typename U>
            static uint8_t test(SameType<size, &U::size>*);

            template <typename u>
            static uint32_t test(...);

        public:
            static const bool value = sizeof(test<SequenceTraits<T, IO>>(nullptr)) == 1;
        };

        template <typename T, typename IO>
        concept SequenceType = has_SequenceTraits<T, IO>::value;

        template <typename T, typename IO>
        class has_CustomMappingTraits
        {
            using inputOne = void (*)(IO&, std::string const&, T&);
            using output   = void (*)(IO&, T&);

            template <typename U>
            static uint8_t test(SameType<inputOne, &U::inputOne>*, SameType<output, &U::output>*);

            template <typename u>
            static uint32_t test(...);

        public:
            static const bool value
                = sizeof(test<CustomMappingTraits<T, IO>>(nullptr, nullptr)) == 1;
        };

        template <typename T, typename IO>
        concept CustomMappingType = has_CustomMappingTraits<T, IO>::value;

        template <typename T, typename IO>
        struct has_SerializationTraits
        {
            static const bool value0
                = has_EmptyMappingTraits<T, IO>::value || has_MappingTraits<T, IO>::value;
            static const bool value1 = has_EnumTraits<T, IO>::value;
            static const bool value2 = has_SequenceTraits<T, IO>::value;
            static const bool value3 = has_CustomMappingTraits<T, IO>::value;

            static const int count = value0 + value1 + value2 + value3;

            static_assert(count == 0 || count == 1, "Ambiguous serialization!");

            static const bool value = value0 || value1 || value2 || value3;
        };

        template <typename T, typename IO, typename Context = EmptyContext>
        concept SerializableType = has_SerializationTraits<T, IO>::value;

        template <typename T, typename IO, typename Context = EmptyContext>
        concept ValueType = !has_SerializationTraits<T, IO>::value;

    } // namespace Serialization
} // namespace rocRoller
