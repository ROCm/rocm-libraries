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
#include <rocRoller/Serialization/HasTraits.hpp>

#include <rocRoller/Utilities/Concepts.hpp>
#include <rocRoller/Utilities/Error.hpp>

#include <llvm/ObjectYAML/YAML.h>

#include <cstddef>

namespace llvm
{
    namespace yaml
    {
        /**
         * Map the `rocRoller::Serialization` traits classes into `llvm::yaml` traits classes.
         */

        namespace sn = rocRoller::Serialization;

        /**
         * The LLVM documentation tells you to put a member:
         *
         * static const bool flow = true;
         *
         * in order to designate a type to be serialized with the flow style (i.e. to use brackets instead of whitespace)
         *
         * What it doesn't clarify is that the value of the bool is not important, so:
         *
         * static const bool flow = false;
         *
         * will *also* mark a type as using the flow style.  The only way to not use flow style is to not have a
         * member named `flow`.  This presents a challenge when trying to specify a traits class that may or may not
         * use flow style based on a template parameter.
         *
         * In the rocRoller::Serialization traits classes, having a `flow` member with a value of `false` will work to
         * specify the non-flow style.
         *
         * The following `FlowBase` can be used as a base class for `llvm::yaml` traits classes and contains a flow
         * member only for types that evaluate to `true`. This uses a series of concepts declared in HasTraits.hpp
         */

        template <typename T>
        struct FlowBase
        {
        };

        template <sn::CHasFlow T>
        struct FlowBase<T>
        {
            static const bool flow = true;
        };

        template <typename T>
        requires(sn::has_SequenceTraits<T, IO>::value) struct SequenceTraits<T>
            : public FlowBase<sn::SequenceTraits<T, IO>>
        {
            static size_t size(IO& io, T& seq)
            {
                return sn::SequenceTraits<T, IO>::size(io, seq);
            }

            static auto& element(IO& io, T& seq, size_t index)
            {
                return sn::SequenceTraits<T, IO>::element(io, seq, index);
            }
        };

        template <typename T>
        requires(sn::has_EnumTraits<T, IO>::value) struct ScalarEnumerationTraits<T>
        {
            static void enumeration(IO& io, T& value)
            {
                sn::EnumTraits<T, IO>::enumeration(io, value);
            }
        };

        template <typename T, typename Context>
        requires(sn::has_MappingTraits<T, IO>::value) struct MappingContextTraits<T, Context>
        {
            static void mapping(IO& io, T& obj, Context& ctx)
            {
                sn::MappingTraits<T, IO, Context>::mapping(io, obj, ctx);
            }
        };

        template <typename T, typename Context>
        concept HasContextMappingTraits = requires(IO& io, T& obj, Context& ctx)
        {
            {MappingContextTraits<T, Context>::mapping(io, obj, ctx)};
        };

        template <typename T>
        requires(sn::has_EmptyMappingTraits<T, IO>::value) struct MappingTraits<T>
            : public FlowBase<sn::MappingTraits<T, IO, EmptyContext>>
        {
            static void mapping(IO& io, T& obj)
            {
                sn::MappingTraits<T, IO, EmptyContext>::mapping(io, obj);
            }
        };

        template <typename T>
        concept HasEmptyMappingTraits = requires(IO& io, T& obj)
        {
            {MappingTraits<T>::mapping(io, obj)};
        };

        template <typename T>
        requires(sn::has_CustomMappingTraits<T, IO>::value) struct CustomMappingTraits<T>
        {
            using Impl = sn::CustomMappingTraits<T, IO>;

            static void inputOne(IO& io, StringRef key, T& value)
            {
                Impl::inputOne(io, key.str(), value);
            }

            static void output(IO& io, T& value)
            {
                Impl::output(io, value);
            }
        };

        template <typename T>
        struct Hide
        {
            T& value;

            Hide(T& value)
                : value(value)
            {
            }

            T& operator*()
            {
                return value;
            }
        };
    } // namespace yaml
} // namespace llvm

namespace rocRoller
{
    namespace Serialization
    {
        template <>
        struct IOTraits<llvm::yaml::IO>
        {
            using IO = llvm::yaml::IO;

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
                io.mapRequired(key, obj, ctx);
            }

            template <typename T>
            static void mapOptional(IO& io, const char* key, T& obj)
            {
                io.mapOptional(key, obj);
            }

            template <typename T, typename Context>
            static void mapOptional(IO& io, const char* key, T& obj, Context& ctx)
            {
                io.mapOptional(key, obj, ctx);
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
                io.setContext(ctx);
            }

            static void* getContext(IO& io)
            {
                return io.getContext();
            }

            template <typename T>
            static void enumCase(IO& io, T& member, const char* key, T value)
            {
                io.enumCase(member, key, value);
            }
        };

    } // namespace Serialization
} // namespace rocRoller

namespace llvm
{
    namespace yaml
    {
        LLVM_YAML_STRONG_TYPEDEF(size_t, FooType);

        using mysize_t
            = std::conditional<std::is_same<size_t, uint64_t>::value, FooType, size_t>::type;

        template <>
        struct ScalarTraits<mysize_t>
        {
            static void output(const mysize_t& value, void* ctx, raw_ostream& stream)
            {
                uint64_t tmp = value;
                ScalarTraits<uint64_t>::output(tmp, ctx, stream);
            }

            static StringRef input(StringRef str, void* ctx, mysize_t& value)
            {
                uint64_t tmp;
                auto     rv = ScalarTraits<uint64_t>::input(str, ctx, tmp);
                value       = tmp;
                return rv;
            }

            static bool mustQuote(StringRef)
            {
                return false;
            }
        };

        template <typename T>
        struct MappingTraits<Hide<T>>
        {
            static void mapping(IO& io, Hide<T>& value)
            {
                sn::MappingTraits<T, IO>::mapping(io, *value);
            }

            static const bool flow = sn::MappingTraits<T, IO>::flow;
        };

        template <typename T>
        struct SequenceTraits<Hide<T>>
        {
            using Impl  = sn::SequenceTraits<T, IO>;
            using Value = typename Impl::Value;

            static size_t size(IO& io, Hide<T>& t)
            {
                return Impl::size(io, *t);
            }
            static Value& element(IO& io, Hide<T>& t, size_t index)
            {
                return Impl::element(io, *t, index);
            }

            static const bool flow = Impl::flow;
        };

        template <typename T>
        struct ScalarEnumerationTraits<Hide<T>>
        {
            static void enumeration(IO& io, Hide<T>& value)
            {
                sn::EnumTraits<T, IO>::enumeration(io, *value);
            }
        };

        template <typename T>
        struct CustomMappingTraits<Hide<T>>
        {
            using Impl = sn::CustomMappingTraits<T, IO>;

            static void inputOne(IO& io, StringRef key, Hide<T>& value)
            {
                Impl::inputOne(io, key.str(), *value);
            }

            static void output(IO& io, Hide<T>& value)
            {
                Impl::output(io, *value);
            }
        };

        /**
         * Add serialization for small floating point types. Defer to built-in fp32 serialization with conversion.
         */
        template <typename T>
        requires(rocRoller::CIsAnyOf<T,
                                     rocRoller::Half,
                                     rocRoller::BFloat16,
                                     rocRoller::FP8,
                                     rocRoller::BF8,
                                     rocRoller::FP6,
                                     rocRoller::BF6,
                                     rocRoller::FP4,
                                     rocRoller::E8M0>) struct ScalarTraits<T>
        {
            static void output(const T& value, void* ctx, llvm::raw_ostream& out)
            {
                float floatVal = value;
                ScalarTraits<float>::output(value, ctx, out);
            }

            static StringRef input(StringRef scalar, void* ctx, T& value)
            {
                float floatVal = 0.0f;

                auto rv = ScalarTraits<float>::input(scalar, ctx, floatVal);

                value = T(floatVal);

                return rv;
            }

            static QuotingType mustQuote(StringRef ref)
            {
                return ScalarTraits<float>::mustQuote(ref);
            }
        };

        template <rocRoller::Serialization::CHasScalarTraits Scalar>
        struct ScalarTraits<Scalar>
        {
            using rrTraits = rocRoller::Serialization::ScalarTraits<Scalar>;

            static void output(const Scalar& value, void* ctx, llvm::raw_ostream& out)
            {
                auto string = rrTraits::output(value);
                ScalarTraits<std::string>::output(string, ctx, out);
            }

            static StringRef input(StringRef scalar, void* ctx, Scalar& value)
            {
                std::string string;
                auto        rv = ScalarTraits<std::string>::input(scalar, ctx, string);

                rrTraits::input(string, value);

                return rv;
            }

            static QuotingType mustQuote(StringRef ref)
            {
                return ScalarTraits<std::string>::mustQuote(ref);
            }
        };

    } // namespace yaml
} // namespace llvm
