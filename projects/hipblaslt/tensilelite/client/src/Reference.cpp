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

#include "Reference.hpp"
#include "DataInitialization.hpp"
#include "Tensile/TensorDescriptor_fwd.hpp"
#include "Tensile/Utils.hpp"
#include "TimingInstrumentation.hpp"
#include "TypedId.hpp"

#include <cstddef>
#include <iostream>
#include <omp.h>

#define MAX_OMP_THREADS 64

namespace TensileLite
{

    namespace
    {

        // Helper to load data from various source types into a float buffer.
        template <typename SrcType>
        std::vector<float> loadToFloat(void const* src, size_t N)
        {
            std::vector<float> buffer(N);
            const SrcType*     sPtr = static_cast<const SrcType*>(src);
            for(size_t i = 0; i < N; ++i)
            {
                buffer[i] = static_cast<float>(sPtr[i]);
            }
            return buffer;
        }

        // Helper to store data from a float buffer into various destination types.
        template <typename DstType>
        void storeFromFloat(void* dst, const std::vector<float>& buffer, size_t N)
        {
            DstType* dPtr = static_cast<DstType*>(dst);
            for(size_t i = 0; i < N; ++i)
            {
                dPtr[i] = static_cast<DstType>(buffer[i]);
            }
        }

        // Helper class that wraps a shadow copy of input buffers in float format.
        // It quietly manages the indirection between directly using the input pointer
        // (for float) and a shadow copy (for half / bfloat16).
        class ShadowBuffer
        {
            std::vector<float> m_storage;
            const float*       m_ptr = nullptr;

        public:
            ShadowBuffer() = default;
            ShadowBuffer(void const* ptr, rocisa::DataType type, size_t N)
            {
                if(ptr == nullptr)
                {
                    assert(N == 0 && "Null pointer with non-zero size");
                }
                else if(type == rocisa::DataType::Float)
                {
                    m_ptr = static_cast<const float*>(ptr);
                }
                else if(type == rocisa::DataType::Half)
                {
                    m_storage = loadToFloat<TensileLite::Half>(ptr, N);
                    m_ptr     = m_storage.data();
                }
                else if(type == rocisa::DataType::BFloat16)
                {
                    m_storage = loadToFloat<TensileLite::BFloat16>(ptr, N);
                    m_ptr     = m_storage.data();
                }
                else
                {
                    throw std::runtime_error("Unsupported type for ShadowBuffer");
                }
            }

            const float* data() const
            {
                return m_ptr;
            }

            explicit operator bool() const
            {
                return m_ptr != nullptr;
            }

            // Array access convenience
            float operator[](size_t idx) const
            {
                return m_ptr[idx];
            }
        };
    }

    namespace Client
    {
        template <typename T>
        T abs(T val)
        {
            return (val > T(0))? val : -val;
        }

        template <typename T>
        struct Transform
        {
            inline static T Input(T const& val, bool conj)
            {
                return val;
            }
        };

        template <typename T>
        struct Transform<std::complex<T>>
        {
            inline static std::complex<T> Input(std::complex<T> const& val, bool conj)
            {
                if(conj)
                    return std::conj(val);

                return val;
            }
        };

        void throwException(const std::string& msg)
        {
            throw std::runtime_error(msg.c_str());
        }

        template <typename Accumulator,
                  typename MathOpAccum = Accumulator,
                  typename TypeL,
                  typename TypeR>
        inline Accumulator multiply(TypeL l, TypeR r)
        {
            /* Transform the data type from TypeL/TypeR to Accumulator if TypeL!=ACC or TypeR!=ACC, but filter out cases, I8/I32/I32 and I8x4/I32/I32
             *
             * There are three cases of doing multiplication and their conditions to do transform or not are as below.
             * 1. AxB : (A!=ACC or B!=ACC) and A!=I8 and A!=I8x4
             * 2. Alpha x rC :  (Alpha!=ACC or rC!=ACC)
             * 3. Beta x C : (Beta!=ACC or C!=ACC)
            */
            constexpr bool needAccumCast
                = !(std::is_same<TypeL, Accumulator>() && std::is_same<TypeR, Accumulator>())
                  && !std::is_same<TypeL,
                                   Int8>() //case I8/I32/I32, I8 implicitly cast to int.
                  && !std::is_same<TypeL,
                                   Int8x4>(); //case I8x4/I32/I32, I8x4 overloading the op*.

            using LMultT = std::conditional_t<needAccumCast, Accumulator, TypeL>;
            using RMultT = std::conditional_t<needAccumCast, Accumulator, TypeR>;

            constexpr bool needMathOpAccumCast = !std::is_same<Accumulator, MathOpAccum>();
            using LMathOpMultT = std::conditional_t<needMathOpAccumCast, MathOpAccum, LMultT>;
            using RMathOpMultT = std::conditional_t<needMathOpAccumCast, MathOpAccum, RMultT>;
            return static_cast<Accumulator>(static_cast<LMultT>(static_cast<LMathOpMultT>(l))
                                            * static_cast<RMultT>(static_cast<RMathOpMultT>(r)));
        }

        template <typename Accumulator,
                  typename MathOpAccum = Accumulator,
                  typename TypeL,
                  typename TypeR>
        inline Accumulator div(TypeL l, TypeR r)
        {
            /* Transform the data type from TypeL/TypeR to Accumulator if TypeL!=ACC or TypeR!=ACC, but filter out cases, I8/I32/I32 and I8x4/I32/I32
             *
             * There are three cases of doing multiplication and their conditions to do transform or not are as below.
             * 1. AxB : (A!=ACC or B!=ACC) and A!=I8 and A!=I8x4
             * 2. Alpha x rC :  (Alpha!=ACC or rC!=ACC)
             * 3. Beta x C : (Beta!=ACC or C!=ACC)
            */
            constexpr bool needAccumCast
                = !(std::is_same<TypeL, Accumulator>() && std::is_same<TypeR, Accumulator>())
                  && !std::is_same<TypeL,
                                   Int8>() //case I8/I32/I32, I8 be implicitly cast to int.
                  && !std::is_same<TypeL,
                                   Int8x4>(); //case I8x4/I32/I32, I8x4 overloading the op*.

            using LMultT = std::conditional_t<needAccumCast, Accumulator, TypeL>;
            using RMultT = std::conditional_t<needAccumCast, Accumulator, TypeR>;

            constexpr bool needMathOpAccumCast = !std::is_same<Accumulator, MathOpAccum>();
            using LMathOpMultT = std::conditional_t<needMathOpAccumCast, MathOpAccum, LMultT>;
            using RMathOpMultT = std::conditional_t<needMathOpAccumCast, MathOpAccum, RMultT>;

            return static_cast<Accumulator>(static_cast<LMultT>(static_cast<LMathOpMultT>(l))
                                            / static_cast<RMultT>(static_cast<RMathOpMultT>(r)));
        }

        template <typename Accumulator, typename Type>
        inline Accumulator cast(Type val)
        {
            /* Transform the data type from TypeL/TypeR to Accumulator if TypeL!=ACC or TypeR!=ACC, but filter out cases, I8/I32/I32 and I8x4/I32/I32
             *
             * There are three cases of doing multiplication and their conditions to do transform or not are as below.
             * 1. AxB : (A!=ACC or B!=ACC) and A!=I8 and A!=I8x4
             * 2. Alpha x rC :  (Alpha!=ACC or rC!=ACC)
             * 3. Beta x C : (Beta!=ACC or C!=ACC)
            */
            constexpr bool needAccumCast
                = !std::is_same<Type, Accumulator>()
                  && !std::is_same<Type,
                                   Int8>() //case I8/I32/I32, I8 be implicitly cast to int.
                  && !std::is_same<Type,
                                   Int8x4>(); //case I8x4/I32/I32, I8x4 overloading the op*.

            using MultT = std::conditional_t<needAccumCast, Accumulator, Type>;
            return static_cast<MultT>(val);
        }

        template <typename T, typename Accumulator>
        typename std::enable_if<std::is_same<int8_t, T>::value, T>::type
            SaturateCast(Accumulator val)
        {
            if constexpr(std::is_same<Accumulator, Half>::value
                         || std::is_same<Accumulator, BFloat16>::value)
            {
                float tmp = std::nearbyint((float)val); //round to even
                if(tmp > static_cast<float>(127))
                    tmp = static_cast<float>(127);
                else if(tmp < static_cast<float>(-128))
                    tmp = static_cast<float>(-128);
                return static_cast<T>(tmp);
            }
            else
            {
                if constexpr(std::is_same<Accumulator, float>::value
                             || std::is_same<Accumulator, double>::value)
                    val = std::nearbyint(val); //round to even
                if(val > static_cast<Accumulator>(127))
                    val = static_cast<Accumulator>(127);
                else if(val < static_cast<Accumulator>(-128))
                    val = static_cast<Accumulator>(-128);
                return static_cast<T>(val);
            }
        }

        template <typename T, typename Accumulator>
        typename std::enable_if<
            std::is_same<Float8, T>::value || std::is_same<Float8_fnuz, T>::value
                || std::is_same<BFloat8, T>::value || std::is_same<BFloat8_fnuz, T>::value,
            T>::type
            SaturateCast(Accumulator val)
        {
            if constexpr(std::is_same<Accumulator, BFloat16>::value)
                return static_cast<T>(static_cast<float>(val));
            else
                return static_cast<T>(val);
        }

        template <typename T, typename Accumulator>
        typename std::enable_if<!std::is_same<int8_t, T>::value && !std::is_same<Float8, T>::value
                                    && !std::is_same<Float8_fnuz, T>::value
                                    && !std::is_same<BFloat8, T>::value
                                    && !std::is_same<BFloat8_fnuz, T>::value,
                                T>::type
            SaturateCast(Accumulator val)
        {
            return static_cast<T>(val);
        }

        template <typename Accumulator>
        typename std::enable_if<std::is_same<Half, Accumulator>::value
                                    || std::is_same<float, Accumulator>::value
                                    || std::is_same<double, Accumulator>::value
                                    || std::is_same<BFloat16, Accumulator>::value
                                    || std::is_same<int32_t, Accumulator>::value
                                    || std::is_same<int64_t, Accumulator>::value
                                    || std::is_same<int8_t, Accumulator>::value,
                                Accumulator>::type
            GetValue(rocisa::DataType dataType, void const* voidPtr, int pos, bool aConjugate)
        {
            switch(dataType)
            {
            case rocisa::DataType::Float:
            {
                auto typedPtr = static_cast<float const*>(voidPtr);
                return cast<Accumulator>(Transform<float>::Input(typedPtr[pos], aConjugate));
            }
            break;
            case rocisa::DataType::Double:
            {
                auto typedPtr = static_cast<double const*>(voidPtr);
                return cast<Accumulator>(Transform<double>::Input(typedPtr[pos], aConjugate));
            }
            break;
            case rocisa::DataType::Half:
            {
                auto typedPtr = static_cast<Half const*>(voidPtr);
                return cast<Accumulator>(Transform<Half>::Input(typedPtr[pos], aConjugate));
            }
            break;
            case rocisa::DataType::Int32:
            {
                auto typedPtr = static_cast<int32_t const*>(voidPtr);
                return cast<Accumulator>(Transform<int32_t>::Input(typedPtr[pos], aConjugate));
            }
            break;
            case rocisa::DataType::Int64:
            {
                auto typedPtr = static_cast<int64_t const*>(voidPtr);
                return cast<Accumulator>(Transform<int64_t>::Input(typedPtr[pos], aConjugate));
            }
            break;
            case rocisa::DataType::BFloat16:
            {
                auto typedPtr = static_cast<BFloat16 const*>(voidPtr);
                return cast<Accumulator>(Transform<BFloat16>::Input(typedPtr[pos], aConjugate));
            }
            break;
            case rocisa::DataType::Int8:
            {
                auto typedPtr = static_cast<int8_t const*>(voidPtr);
                return cast<Accumulator>(Transform<int8_t>::Input(typedPtr[pos], aConjugate));
            }
            break;
            case rocisa::DataType::Float8:
            {
                auto typedPtr = static_cast<Float8 const*>(voidPtr);
                return cast<Accumulator>(Transform<Float8>::Input(typedPtr[pos], aConjugate));
            }
            break;
            case rocisa::DataType::BFloat8:
            {
                auto typedPtr = static_cast<BFloat8 const*>(voidPtr);
                return cast<Accumulator>(Transform<BFloat8>::Input(typedPtr[pos], aConjugate));
            }
            case rocisa::DataType::Float8_fnuz:
            {
                auto typedPtr = static_cast<Float8_fnuz const*>(voidPtr);
                return cast<Accumulator>(Transform<Float8_fnuz>::Input(typedPtr[pos], aConjugate));
            }
            break;
            case rocisa::DataType::BFloat8_fnuz:
            {
                auto typedPtr = static_cast<BFloat8_fnuz const*>(voidPtr);
                return cast<Accumulator>(Transform<BFloat8_fnuz>::Input(typedPtr[pos], aConjugate));
            }
            break;
            case rocisa::DataType::E8:
            {
                auto typedPtr = static_cast<E8 const*>(voidPtr);
                return cast<Accumulator>(Transform<E8>::Input(typedPtr[pos], aConjugate));
            }
            case rocisa::DataType::E5M3:
            {
                auto typedPtr = static_cast<E5M3 const*>(voidPtr);
                return cast<Accumulator>(Transform<E5M3>::Input(typedPtr[pos], aConjugate));
            }
            break;
                break;
            case rocisa::DataType::XFloat32:
            case rocisa::DataType::ComplexFloat:
            case rocisa::DataType::ComplexDouble:
            case rocisa::DataType::Int8x4:
            case rocisa::DataType::Count:
            case rocisa::DataType::Float8BFloat8:
            case rocisa::DataType::BFloat8Float8:
            case rocisa::DataType::Float8BFloat8_fnuz:
            case rocisa::DataType::BFloat8Float8_fnuz:
            case rocisa::DataType::Float6:
            case rocisa::DataType::BFloat6:
            case rocisa::DataType::Float4:
                ;
            }
            return DataInitialization::getValue<Accumulator, InitMode::Zero>();
        }

        template <typename Accumulator>
        typename std::enable_if<!std::is_same<Half, Accumulator>::value
                                    && !std::is_same<float, Accumulator>::value
                                    && !std::is_same<double, Accumulator>::value
                                    && !std::is_same<BFloat16, Accumulator>::value
                                    && !std::is_same<int32_t, Accumulator>::value
                                    && !std::is_same<int64_t, Accumulator>::value
                                    && !std::is_same<int8_t, Accumulator>::value
                                    && !std::is_same<Float8, Accumulator>::value
                                    && !std::is_same<BFloat8, Accumulator>::value
                                    && !std::is_same<Float8_fnuz, Accumulator>::value
                                    && !std::is_same<BFloat8_fnuz, Accumulator>::value,
                                Accumulator>::type
            GetValue(rocisa::DataType biasType, void const* biasptr, int pos, bool aConjugate)
        {
            return DataInitialization::getValue<Accumulator, InitMode::Zero>();
        }

        template <typename Accumulator,
                  std::enable_if_t<std::is_same<Half, Accumulator>::value
                                       || std::is_same<float, Accumulator>::value
                                       || std::is_same<double, Accumulator>::value
                                       || std::is_same<BFloat16, Accumulator>::value
                                       || std::is_same<Float8, Accumulator>::value
                                       || std::is_same<BFloat8, Accumulator>::value
                                       || std::is_same<Float8_fnuz, Accumulator>::value
                                       || std::is_same<BFloat8_fnuz, Accumulator>::value
                                       || std::is_same<int32_t, Accumulator>::value
                                       || std::is_same<int64_t, Accumulator>::value
                                       || std::is_same<int8_t, Accumulator>::value,
                                   bool>
                  = true>
        void SetValue(rocisa::DataType dataType, Accumulator& src, void* dstPtr, size_t pos)
        {
            switch(dataType)
            {
            case rocisa::DataType::Float:
            {
                auto typedPtr = static_cast<float*>(dstPtr);
                typedPtr[pos] = SaturateCast<float>(src);
            }
            break;
            case rocisa::DataType::Double:
            {
                auto typedPtr = static_cast<double*>(dstPtr);
                typedPtr[pos] = SaturateCast<double>(src);
            }
            break;
            case rocisa::DataType::Half:
            {
                auto typedPtr = static_cast<Half*>(dstPtr);
                typedPtr[pos] = SaturateCast<Half>(src);
            }
            break;
            case rocisa::DataType::Int32:
            {
                auto typedPtr = static_cast<int32_t*>(dstPtr);
                typedPtr[pos] = SaturateCast<int32_t>(src);
            }
            break;
            case rocisa::DataType::BFloat16:
            {
                auto typedPtr = static_cast<BFloat16*>(dstPtr);
                typedPtr[pos] = SaturateCast<BFloat16>(src);
            }
            break;
            case rocisa::DataType::Int8:
            {
                auto typedPtr = static_cast<int8_t*>(dstPtr);
                typedPtr[pos] = SaturateCast<int8_t>(src);
            }
            break;
            case rocisa::DataType::Float8:
            {
                auto typedPtr = static_cast<Float8*>(dstPtr);
                typedPtr[pos] = SaturateCast<Float8>(src);
            }
            break;
            case rocisa::DataType::BFloat8:
            {
                auto typedPtr = static_cast<BFloat8*>(dstPtr);
                typedPtr[pos] = SaturateCast<BFloat8>(src);
            }
            break;
            case rocisa::DataType::Float8_fnuz:
            {
                auto typedPtr = static_cast<Float8_fnuz*>(dstPtr);
                typedPtr[pos] = SaturateCast<Float8_fnuz>(src);
            }
            break;
            case rocisa::DataType::BFloat8_fnuz:
            {
                auto typedPtr = static_cast<BFloat8_fnuz*>(dstPtr);
                typedPtr[pos] = SaturateCast<BFloat8_fnuz>(src);
            }
            break;
            case rocisa::DataType::XFloat32:
            case rocisa::DataType::ComplexFloat:
            case rocisa::DataType::ComplexDouble:
            case rocisa::DataType::Int8x4:
            case rocisa::DataType::Int64:
            case rocisa::DataType::Count:
            case rocisa::DataType::Float8BFloat8:
            case rocisa::DataType::BFloat8Float8:
            case rocisa::DataType::Float8BFloat8_fnuz:
            case rocisa::DataType::BFloat8Float8_fnuz:
            case rocisa::DataType::Float6:
            case rocisa::DataType::BFloat6:
            case rocisa::DataType::Float4:
            case rocisa::DataType::E8:
            case rocisa::DataType::E5M3:
                ;
            }
        }

        template <typename Accumulator,
                  std::enable_if_t<!std::is_same<Half, Accumulator>::value
                                       && !std::is_same<float, Accumulator>::value
                                       && !std::is_same<double, Accumulator>::value
                                       && !std::is_same<BFloat16, Accumulator>::value
                                       && !std::is_same<int32_t, Accumulator>::value
                                       && !std::is_same<int64_t, Accumulator>::value
                                       && !std::is_same<int8_t, Accumulator>::value
                                       && !std::is_same<Float8, Accumulator>::value
                                       && !std::is_same<BFloat8, Accumulator>::value
                                       && !std::is_same<Float8_fnuz, Accumulator>::value
                                       && !std::is_same<BFloat8_fnuz, Accumulator>::value,
                                   bool>
                  = true>
        void SetValue(rocisa::DataType dataType, Accumulator& src, void* dstPtr, size_t pos)
        {
            switch(dataType)
            {
            case rocisa::DataType::Float:
            case rocisa::DataType::Double:
            case rocisa::DataType::Half:
            case rocisa::DataType::Int32:
            case rocisa::DataType::Int64:
            case rocisa::DataType::BFloat16:
            case rocisa::DataType::Int8:
            case rocisa::DataType::Float8:
            case rocisa::DataType::BFloat8:
            case rocisa::DataType::Float8_fnuz:
            case rocisa::DataType::BFloat8_fnuz:
            case rocisa::DataType::XFloat32:
                break;
            case rocisa::DataType::ComplexFloat:
            case rocisa::DataType::ComplexDouble:
                break;
            case rocisa::DataType::Int8x4:
            {
                throw std::runtime_error("Not supported yet.");
            }
            break;
            case rocisa::DataType::Count:
            case rocisa::DataType::Float8BFloat8:
            case rocisa::DataType::BFloat8Float8:
            case rocisa::DataType::Float8BFloat8_fnuz:
            case rocisa::DataType::BFloat8Float8_fnuz:
            case rocisa::DataType::Float6:
            case rocisa::DataType::BFloat6:
            case rocisa::DataType::Float4:
            case rocisa::DataType::E8:
            case rocisa::DataType::E5M3:
                ;
            }
        }

        template <typename T>
        typename std::enable_if<std::is_same<float, T>::value || std::is_same<Half, T>::value
                                    || std::is_same<BFloat16, T>::value,
                                T>::type
            Activation(ActivationType activationType,
                       T              val,
                       ActivationType activationType2,
                       std::vector<T> args)
        {
            // Only cast to float in BFloat16
            constexpr bool needCast = std::is_same<BFloat16, T>();
            using castT             = std::conditional_t<needCast, float, T>;
            const auto isForAll     = activationType == ActivationType::All
                                  || activationType == ActivationType::Hipblaslt_all;
            auto new_type = isForAll ? activationType2 : activationType;
            if(new_type == ActivationType::Abs)
            {
                return static_cast<T>(std::max(static_cast<castT>(val), -static_cast<castT>(val)));
            }
            else if(new_type == ActivationType::Clippedrelu)
            {
                if(val > args[0])
                    return static_cast<T>(
                        std::min(static_cast<castT>(val), static_cast<castT>(args[1])));
                return static_cast<T>(
                    std::min(static_cast<castT>(0.0), static_cast<castT>(args[1])));
            }
            else if(new_type == ActivationType::Clamp)
            {
                return static_cast<T>(
                    std::max(static_cast<castT>(args[0]),
                             std::min(static_cast<castT>(val), static_cast<castT>(args[1]))));
            }
            else if(new_type == ActivationType::Exp)
            {
                return static_cast<T>(exp(static_cast<castT>(val)));
            }
            else if(new_type == ActivationType::Gelu || new_type == ActivationType::Geluscaling)
            {
                auto castedVal = static_cast<castT>(val);
                auto k0        = static_cast<castT>(0.7978845608028654);
                auto k1        = static_cast<castT>(0.044715);
                // float(0.5 * x * (1 + tanh(k0 * x * (1 + k1 * x * x))));
                auto tmp = (static_cast<castT>(1)
                            + multiply<castT>(k1, multiply<castT>(castedVal, castedVal)));
                tmp      = multiply<castT>(k0, multiply<castT>(castedVal, tmp));
                tmp      = static_cast<castT>(1) + static_cast<castT>(tanh(tmp));
                tmp = multiply<castT>(static_cast<castT>(0.5f), multiply<castT>(castedVal, tmp));
                if(new_type == ActivationType::Geluscaling)
                    tmp = multiply<castT>(tmp, static_cast<castT>(args[0]));
                return static_cast<T>(tmp);
            }
            else if(new_type == ActivationType::Leakyrelu)
            {
                assert((args.size() == getAdditionalArgNum(activationType)));
                auto tmp = static_cast<castT>(val);
                tmp      = tmp > static_cast<castT>(0.f) ? tmp : multiply<castT>(tmp, args[0]);
                return (T)(tmp);
            }
            else if(new_type == ActivationType::Relu)
            {
                return (T)(std::max(0.f, static_cast<float>(val)));
            }
            else if(new_type == ActivationType::Sigmoid)
            {
                return static_cast<T>(1.f
                                      / (1.f + static_cast<castT>(exp(-static_cast<castT>(val)))));
            }
            else if(new_type == ActivationType::Tanh)
            {
                return multiply<T>(
                    tanh(multiply<castT>(static_cast<castT>(val), static_cast<castT>(args[0]))),
                    static_cast<castT>(args[1]));
            }
            else if(new_type == ActivationType::DGelu)
            {
                auto castedVal = static_cast<float>(val);
                auto k0        = 0.0535161f;
                auto k1        = 0.398942f;
                auto k2        = 0.0356774f;
                auto k3        = 0.797885f;
                // Original: x1 * x2 = (0.0535161x3 + 0.398942x) x cosh-2(0.0356774x3 + 0.797885x)
                // 0.5 * tanh(0.0356774x3 + 0.797885x) + (0.0535161x3 + 0.398942x) * (4/pow(exp(-(0.0356774 * pow(x, 3)+ 0.797885 * x)) + exp((0.0356774 * pow(x, 3)+ 0.797885 * x)), 2)) + 0.5
                // x1 = (0.0535161 * pow(x, 3) + 0.398942 * x)
                // xx = 0.0356774 * pow(x, 3)+ 0.797885 * x
                // x2 = 4/pow(math.exp(-xx) + math.exp(xx),2)
                // 0.5 * math.tanh(xx) + x1 * x2 + 0.5
                float pow3 = castedVal * castedVal * castedVal;
                float x1   = k0 * pow3 + k1 * castedVal;
                float xx   = k2 * pow3 + k3 * castedVal;
                float x2   = 4 / pow(exp(-xx) + exp(xx), 2);
                float tmp  = 0.5 * tanh(xx) + x1 * x2 + 0.5;
                return static_cast<T>(tmp);
            }
            else if(new_type == ActivationType::Silu)
            {
                auto castedVal = static_cast<castT>(val);
                return static_cast<T>(castedVal / (1.f + static_cast<castT>(exp(-castedVal))));
            }
            else if(new_type == ActivationType::Swish)
            {
                auto castedVal = static_cast<castT>(val);
                return static_cast<T>(castedVal
                                      / (1.f
                                         + static_cast<castT>(exp(-multiply<castT>(
                                             castedVal, static_cast<castT>(args[0]))))));
            }
            return val;
        }

        template <typename T>
        typename std::enable_if<std::is_same<double, T>::value || std::is_same<int32_t, T>::value,
                                T>::type
            Activation(ActivationType activationType,
                       T              val,
                       ActivationType activationType2,
                       std::vector<T> args)
        {
            const auto isForAll = activationType == ActivationType::All
                                  || activationType == ActivationType::Hipblaslt_all;
            auto new_type = isForAll ? activationType2 : activationType;
            if(new_type == ActivationType::Abs)
            {
                return static_cast<T>(abs(val));
            }
            else if(new_type == ActivationType::Clippedrelu)
            {
                if(val > args[0])
                    return static_cast<T>(std::min(val, args[1]));
                return static_cast<T>(std::min(static_cast<T>(0.0), args[1]));
            }
            else if(new_type == ActivationType::Relu)
            {
                return static_cast<T>(std::max(static_cast<T>(0.0), val));
            }
            else if(new_type == ActivationType::Leakyrelu)
            {
                assert((args.size() == getAdditionalArgNum(activationType)));
                val = val > 0 ? val : val * args[0];
                return val;
            }
            else if(new_type == ActivationType::DGelu)
            {
                throw std::runtime_error("Unsupported type dgelu.");
            }
            else if(new_type == ActivationType::DRelu)
            {
                throw std::runtime_error("Unsupported type drelu.");
            }
            return val;
        }

        template <typename T>
        typename std::enable_if<!std::is_same<Half, T>::value && !std::is_same<float, T>::value
                                    && !std::is_same<double, T>::value
                                    && !std::is_same<BFloat16, T>::value
                                    && !std::is_same<int32_t, T>::value,
                                T>::type
            Activation(ActivationType activationType,
                       T              val,
                       ActivationType activationType2,
                       std::vector<T> args)
        {
            return val;
        }

        template <
            typename Input,
            typename Accumulator,
            std::enable_if_t<
                std::is_same<Half, Input>::value || std::is_same<float, Input>::value
                    || std::is_same<double, Input>::value || std::is_same<BFloat16, Input>::value
                    || std::is_same<int32_t, Input>::value || std::is_same<int8_t, Input>::value
                    || std::is_same<Float8, Input>::value || std::is_same<BFloat8, Input>::value
                    || std::is_same<Float8_fnuz, Input>::value
                    || std::is_same<BFloat8_fnuz, Input>::value,
                bool>
            = true>
        std::string ReductionCPU(TensorDescriptor const&  biasTensor,
                                 TensorDescriptor const&  tensor,
                                 void const*              src,
                                 ContractionInputs const& inputs,
                                 const size_t&            elementsToValidate,
                                 size_t                   reducIdx)
        {
            size_t validationStride = 1;
            if(elementsToValidate > 0 && elementsToValidate < biasTensor.totalLogicalElements())
                validationStride
                    = NextPrime(biasTensor.totalAllocatedElements() / elementsToValidate);

            Input const* srcTyped = (Input const*)src;
            // For 2D bias reduction, d batch = 1
            if((tensor.dimensions() == 3 && tensor.sizes()[2] == 1) || tensor.dimensions() == 2)
            {
                omp_set_num_threads(MAX_OMP_THREADS);
#pragma omp parallel for
                for(size_t bNum = 0; bNum < biasTensor.totalLogicalElements();
                    bNum += validationStride)
                {
                    std::vector<int64_t> coord(tensor.dimensions());
                    size_t               sumLength = 0;
                    size_t               idx       = 0;
                    if(reducIdx == 1)
                    {
                        sumLength = tensor.sizes()[1];
                        coord[0]  = bNum;
                        idx       = 1;
                    }
                    else
                    {
                        sumLength = tensor.sizes()[0];
                        coord[1]  = bNum;
                        idx       = 0;
                    }
                    Accumulator sum = static_cast<Accumulator>(0);
                    for(size_t i = 0; i < sumLength; i++)
                    {
                        coord[idx] = i;
                        auto index = tensor.index(coord);
                        sum += static_cast<Accumulator>(srcTyped[index]);
                    }
                    auto biasPtr = (void*)inputs.bias;
                    SetValue<Accumulator>(biasTensor.dataType(), sum, biasPtr, bNum);
                }
            }
            else
            {
                std::string msg = "Unsupported reduction dimension "
                                  + std::to_string(tensor.dimensions()) + ".";
                return msg;
            }
            return "";
        }

        template <
            typename Input,
            typename Accumulator,
            std::enable_if_t<
                !std::is_same<Half, Input>::value && !std::is_same<float, Input>::value
                    && !std::is_same<double, Input>::value && !std::is_same<BFloat16, Input>::value
                    && !std::is_same<int32_t, Input>::value && !std::is_same<int8_t, Input>::value
                    && !std::is_same<Float8, Input>::value && !std::is_same<BFloat8, Input>::value
                    && !std::is_same<Float8_fnuz, Input>::value
                    && !std::is_same<BFloat8_fnuz, Input>::value,
                bool>
            = true>
        std::string ReductionCPU(TensorDescriptor const&  biasTensor,
                                 TensorDescriptor const&  tensor,
                                 void const*              src,
                                 ContractionInputs const& inputs,
                                 const size_t&            elementsToValidate,
                                 size_t                   reducIdx)
        {
            throw std::runtime_error("Unsupported input type.");
        }

        bool isFastPathEligible(ContractionProblemGemm const& problem)
        {

            // For more precise numerical correctness with XFloat32, skip this fast path.
            // If we knew at this point that the data was initialized as whole number floats,
            // we could continue down this fast path, because there would be no rounding
            // errors incurred by f32 accumulation. But we do not.
            auto rejectFast = [](const char* reason) {
                if (false) {  // Re-enable when testing to find reason.
                    std::clog << "FAST_PATH_REJECT: " << reason << std::endl;
                }
                return false;
            };

            if(problem.f32XdlMathOp() == rocisa::DataType::XFloat32)
            {
                return rejectFast("XFloat32");
            }

            auto isSupportedType = [](rocisa::DataType t) {
                return t == rocisa::DataType::Float || t == rocisa::DataType::Half
                       || t == rocisa::DataType::BFloat16;
            };

            if(!isSupportedType(problem.a().dataType()) || !isSupportedType(problem.b().dataType())
               || !isSupportedType(problem.c().dataType())
               || !isSupportedType(problem.d().dataType()))
            {
                std::string detail = "unsupported_type"
                    " A=" + TensileLite::ToString(problem.a().dataType())
                    + " B=" + TensileLite::ToString(problem.b().dataType())
                    + " C=" + TensileLite::ToString(problem.c().dataType())
                    + " D=" + TensileLite::ToString(problem.d().dataType());
                return rejectFast(detail.c_str());
            }

            if(problem.batchIndices().empty())
            {
                return rejectFast("no_batch_indices");
            }

            if(problem.useGradient())
            {
                return rejectFast("gradient");
            }

            if(problem.outputAmaxD())
            {
                return rejectFast("amaxD");
            }

            if(problem.useE())
            {
                return rejectFast("useE");
            }

            if(problem.useScaleCD())
            {
                return rejectFast("scaleCD");
            }

            if(problem.boundIndices().size() != 1 || problem.freeIndicesA().size() != 1
               || problem.freeIndicesB().size() != 1)
            {
                return rejectFast("multi_index");
            }

            // Layout validation — index accesses are safe because the
            // index-structure check above verified exactly 1 element in each.
            size_t indexMA = problem.freeIndicesA()[0].i;
            size_t indexKA = problem.boundIndices()[0].a;
            size_t indexNB = problem.freeIndicesB()[0].i;
            size_t indexKB = problem.boundIndices()[0].b;
            size_t indexMD = problem.freeIndices()[0].d;

            size_t strideMA = problem.a().strides()[indexMA];
            size_t strideKA = problem.a().strides()[indexKA];
            size_t strideNB = problem.b().strides()[indexNB];
            size_t strideKB = problem.b().strides()[indexKB];

            bool isPackedA = (strideMA == 1 || strideKA == 1);
            bool isPackedB = (strideNB == 1 || strideKB == 1);
            bool isPackedD = (problem.d().strides()[indexMD] == 1);
            if(!isPackedA || !isPackedB || !isPackedD)
            {
                return rejectFast("layout");
            }

            return true;
        }

        // Solve combinations of f16, bf16, f32 gemm problems using efficient CPU code.
        // This function assumes the problem is eligible for the fast path — callers
        // must check isFastPathEligible() first.
        void solveCPUFastInF32(ContractionProblemGemm const& problem,
                               ContractionInputs const&      inputs)
        {
            if(!isFastPathEligible(problem))
            {
                throw std::runtime_error(
                    "solveCPUFastInF32 called on an ineligible problem. "
                    "Callers must check isFastPathEligible() first.");
            }

            bool               doActivation = false;
            std::vector<float> actArgs;
            if(problem.activationType() != ActivationType::None)
            {
                doActivation = true;
                for(int i = 0; i < inputs.activationArgs.size(); i++)
                {
                    actArgs.push_back(constVariantCast<float>(inputs.activationArgs[i]));
                }
            }

            size_t indexMA = problem.freeIndicesA()[0].i;
            size_t indexKA = problem.boundIndices()[0].a;
            size_t indexNB = problem.freeIndicesB()[0].i;
            size_t indexKB = problem.boundIndices()[0].b;
            size_t indexMD = problem.freeIndices()[0].d;

            size_t strideMA = problem.a().strides()[indexMA];
            size_t strideKA = problem.a().strides()[indexKA];
            size_t strideNB = problem.b().strides()[indexNB];
            size_t strideKB = problem.b().strides()[indexKB];

            size_t indexND  = problem.freeIndices()[1].d;
            size_t strideND = problem.d().strides()[indexND];
            size_t strideNC = problem.c().strides()[indexND];

            size_t strideBatchA = problem.a().strides()[problem.batchIndices()[0].a];
            size_t strideBatchB = problem.b().strides()[problem.batchIndices()[0].b];
            size_t strideBatchC = problem.c().strides()[problem.batchIndices()[0].d];
            size_t strideBatchD = problem.d().strides()[problem.batchIndices()[0].d];

            // 4. Shadow copies in f32.
            ShadowBuffer shadowA(
                inputs.a, problem.a().dataType(), problem.a().totalAllocatedElements());
            ShadowBuffer shadowB(
                inputs.b, problem.b().dataType(), problem.b().totalAllocatedElements());
            ShadowBuffer shadowC(
                inputs.c, problem.c().dataType(), problem.c().totalAllocatedElements());

            std::vector<float> shadowD;
            float*             ptrD = nullptr;
            if(problem.d().dataType() == rocisa::DataType::Float)
            {
                ptrD = static_cast<float*>(inputs.d);
            }
            else
            {
                shadowD.resize(problem.d().totalAllocatedElements());
                ptrD = shadowD.data();
            }

            bool useScaleAlphaVec = problem.useScaleAlphaVec();
            int  factorDim        = problem.getParams().factorDim(); // 0 = Row(M), 1 = Col(N)

            ShadowBuffer shadowAlphaVec;
            if(problem.useScaleAlphaVec())
            {
                size_t vecLen  = (factorDim == 0) ? problem.freeSizeA(0) : problem.freeSizeB(0);
                shadowAlphaVec = ShadowBuffer(inputs.scaleAlphaVec, problem.alphaType(), vecLen);
            }

            size_t sizeBatch = problem.batchSize(0);
            size_t sizeK     = problem.boundSize(0);
            size_t sizeM     = problem.freeSizeA(0);
            size_t sizeN     = problem.freeSizeB(0);

            enum class ScaleABMode { None, Scalar, Vector };
            ScaleABMode  scaleABMode = ScaleABMode::None;
            ShadowBuffer shadowScaleA, shadowScaleB;
            float        scaleABScalar = 1.0f; // pre-multiplied scalar for Scalar mode
            {
                std::string useScaleAB = problem.useScaleAB();
                if(useScaleAB == "Vector")
                {
                    scaleABMode  = ScaleABMode::Vector;
                    shadowScaleA = ShadowBuffer(inputs.scaleA, problem.alphaType(), sizeM);
                    shadowScaleB = ShadowBuffer(inputs.scaleB, problem.alphaType(), sizeN);
                }
                else if(useScaleAB == "Scalar")
                {
                    scaleABMode = ScaleABMode::Scalar;
                    ShadowBuffer tmpA(inputs.scaleA, problem.alphaType(), 1);
                    ShadowBuffer tmpB(inputs.scaleB, problem.alphaType(), 1);
                    scaleABScalar = tmpA[0] * tmpB[0];
                }
            }

            constexpr size_t BLOCK_M = 32;
            constexpr size_t BLOCK_N = 32;
            constexpr size_t BLOCK_K = 8;

            auto nTiles = (sizeN / BLOCK_N + (sizeN % BLOCK_N != 0));
            auto mTiles = (sizeM / BLOCK_M + (sizeM % BLOCK_M != 0));
            auto kTiles = (sizeK / BLOCK_K + (sizeK % BLOCK_K != 0));

// Perform the contraction.
// Parallelize over the 3 non-reduction dimensions: batch, M, and N.
// Each thread computes a BLOCK_M x BLOCK_N tile.
#pragma omp parallel for collapse(3)
            for(size_t b = 0; b < sizeBatch; ++b)
            {
                const float* curBatchA = shadowA.data() + (b * strideBatchA);
                const float* curBatchB = shadowB.data() + (b * strideBatchB);
                const float* curBatchC = shadowC.data() + (b * strideBatchC);
                float*       curBatchD = ptrD + (b * strideBatchD);
                for(size_t m = 0; m < mTiles; ++m)
                {
                    auto m0 = m * BLOCK_M;
                    for(size_t n = 0; n < nTiles; ++n)
                    {
                        auto n0 = n * BLOCK_N;

                        std::array<float, BLOCK_M * BLOCK_K> aReg = {0};
                        std::array<float, BLOCK_K * BLOCK_N> bReg = {0};
                        std::array<float, BLOCK_M * BLOCK_N> cReg = {0};
                        for(size_t k = 0; k < kTiles; ++k)
                        {
                            auto k0 = k * BLOCK_K;

                            // Populate A 'registers':
                            for(size_t km = 0; km < BLOCK_K; ++km)
                            {
                                for(size_t mm = 0; mm < BLOCK_M; ++mm)
                                {
                                    size_t global_k = k0 + km;
                                    size_t global_m = m0 + mm;
                                    if(global_k < sizeK && global_m < sizeM)
                                    {
                                        auto offset = global_m * strideMA + global_k * strideKA;
                                        aReg[km * BLOCK_M + mm] = curBatchA[offset];
                                    }
                                    else
                                    {
                                        aReg[km * BLOCK_M + mm] = 0.0f;
                                    }
                                }
                            }

                            // Populate B 'registers':
                            for(size_t kn = 0; kn < BLOCK_K; ++kn)
                            {
                                for(size_t nn = 0; nn < BLOCK_N; ++nn)
                                {
                                    size_t global_k = k0 + kn;
                                    size_t global_n = n0 + nn;
                                    if(global_k < sizeK && global_n < sizeN)
                                    {
                                        bReg[kn * BLOCK_N + nn]
                                            = curBatchB[global_n * strideNB + global_k * strideKB];
                                    }
                                    else
                                    {
                                        bReg[kn * BLOCK_N + nn] = 0.0f;
                                    }
                                }
                            }

                            // Perform matrix multiplication accumulation with k as inner-most (fastest)
                            // dimension for both A and B. A, B, and C of sizes defined by BLOCK_M, BLOCK_N, BLOCK_K.
                            // Store result in row-major order.
                            auto innerReduction = [BLOCK_M, BLOCK_N, BLOCK_K](
                                                      const float* A, const float* B, float* C) {
                                for(size_t k_i = 0; k_i < BLOCK_K; ++k_i)
                                {
                                    for(size_t m_i = 0; m_i < BLOCK_M; ++m_i)
                                    {
                                        for(size_t n_i = 0; n_i < BLOCK_N; ++n_i)
                                        {
                                            auto  b_index = k_i * BLOCK_N + n_i;
                                            auto  a_index = k_i * BLOCK_M + m_i;
                                            auto  c_index = m_i * BLOCK_N + n_i;
                                            float valB    = B[b_index];
                                            float valA    = A[a_index];
                                            C[c_index] += valA * valB;
                                        }
                                    }
                                }
                            };
                            innerReduction(aReg.data(), bReg.data(), cReg.data());
                        }

                        // Copy from cReg back.
                        for(size_t nn = 0; nn < BLOCK_N; ++nn)
                        {
                            for(size_t mm = 0; mm < BLOCK_M; ++mm)
                            {
                                size_t global_n = n0 + nn;
                                size_t global_m = m0 + mm;
                                if(global_n < sizeN && global_m < sizeM)
                                {
                                    size_t idxD     = global_n * strideND + global_m;
                                    curBatchD[idxD] = cReg[mm * BLOCK_N + nn];
                                }
                            }
                        }

                        // Perform all the post-reduction stuff.
                        const float originalAlpha = std::get<float>(inputs.alpha);
                        const float beta          = std::get<float>(inputs.beta);
                        for(size_t nn = 0; nn < BLOCK_N; ++nn)
                        {
                            for(size_t mm = 0; mm < BLOCK_M; ++mm)
                            {
                                size_t global_n = n0 + nn;
                                size_t global_m = m0 + mm;
                                if(global_n < sizeN && global_m < sizeM)
                                {
                                    size_t idxD      = global_m + (global_n * strideND);
                                    size_t idxC      = global_m + (global_n * strideNC);
                                    auto   startingC = curBatchC[idxC];
                                    auto   current   = curBatchD[idxD];
                                    float  alpha     = originalAlpha;
                                    if(scaleABMode == ScaleABMode::Vector)
                                    {
                                        alpha *= shadowScaleA[global_m];
                                        alpha *= shadowScaleB[global_n];
                                    }
                                    else if(scaleABMode == ScaleABMode::Scalar)
                                    {
                                        alpha *= scaleABScalar;
                                    }
                                    if(useScaleAlphaVec)
                                    {
                                        if(factorDim == 1)
                                        {
                                            alpha *= shadowAlphaVec[global_n];
                                        }
                                        else
                                        {
                                            alpha *= shadowAlphaVec[global_m];
                                        }
                                    }
                                    if(beta != 0.0f)
                                    {
                                        current = (alpha * current) + (beta * startingC);
                                    }
                                    else
                                    {
                                        current = (alpha * current);
                                    }

                                    if(problem.useBias() && inputs.bias)
                                    {

                                        assert(!problem.useGradient()
                                               && "Bias gradient not supported on this path.");

                                        size_t dNum
                                            = global_m + (global_n * sizeM) + (b * sizeM * sizeN);
                                        auto const&          d = problem.d();
                                        std::vector<int64_t> dCoord(d.dimensions());
                                        CoordNumbered(dNum,
                                                      dCoord.begin(),
                                                      dCoord.end(),
                                                      d.sizes().begin(),
                                                      d.sizes().end());

                                        auto const&          bias = problem.bias();
                                        std::vector<int64_t> biasCoord(bias.dimensions());
                                        for(size_t i = 0; i < problem.batchIndices().size(); i++)
                                        {
                                            auto const& idx   = problem.batchIndices()[i];
                                            size_t      coord = dCoord[idx.d];
                                            if(biasCoord.size() > 2)
                                            {
                                                biasCoord[2] = coord;
                                            }
                                        }
                                        auto biasIndex = problem.bias().index(biasCoord);

                                        int pos = 0;
                                        if(problem.getParams().factorDim())
                                            pos = int(int(dNum / problem.d().sizes()[0])
                                                      % problem.d().sizes()[1])
                                                  + biasIndex;
                                        else
                                            pos = int(dNum % problem.d().sizes()[0]) + biasIndex;

                                        bool aConjugate   = false;
                                        using Accumulator = float;
                                        Accumulator biasVal
                                            = GetValue<Accumulator>(problem.bias().dataType(),
                                                                    inputs.bias,
                                                                    pos,
                                                                    aConjugate);

                                        current += biasVal;
                                    }

                                    if(doActivation)
                                    {
                                        current = Activation(problem.activationType(),
                                                             current,
                                                             problem.getParams().activationEnum(),
                                                             actArgs);
                                    }

                                    curBatchD[idxD] = current;
                                }
                            }
                        }
                    }
                }
            }

            // 6. Write Back
            if(problem.d().dataType() == rocisa::DataType::Half)
            {
                storeFromFloat<TensileLite::Half>(
                    inputs.d, shadowD, problem.d().totalAllocatedElements());
            }
            else if(problem.d().dataType() == rocisa::DataType::BFloat16)
            {
                storeFromFloat<TensileLite::BFloat16>(
                    inputs.d, shadowD, problem.d().totalAllocatedElements());
            }

        }

        template <typename Accumulator,
                  typename MathOpAccum,
                  typename Type,
                  typename ComputeInputType
                  , std::enable_if_t<true
#ifdef TENSILE_USE_FP6
                                       && !std::is_same<Float6x16, Type>::value
#endif // #ifdef TENSILE_USE_FP6
#ifdef TENSILE_USE_BF6
                                       && !std::is_same<BFloat6x16, Type>::value
#endif // #ifdef TENSILE_USE_BF6
#ifdef TENSILE_USE_FP4
                                       && !std::is_same<Float4x2, Type>::value
#endif // #ifdef TENSILE_USE_FP4
                , bool>
            = true>
        inline Accumulator getElement(
            ContractionProblemGemm const& problem,
            Type const* ptr,
            const size_t idx,
            void const* scalePtr,
            const bool conjugate)
        {
            // case I8/I32/I32, I8 be implicitly cast to int.
            constexpr bool needAccumCast = !std::is_same<Type, Accumulator>() && !std::is_same<Type, Int8>();
            using MultT = std::conditional_t<needAccumCast, Accumulator, Type>;

            constexpr bool needMathOpAccumCast = !std::is_same<Accumulator, MathOpAccum>();
            using MathOpMultT = std::conditional_t<needMathOpAccumCast, MathOpAccum, MultT>;

            Type val = Transform<Type>::Input(ptr[idx], conjugate);

            if constexpr(sizeof(Type) > sizeof(ComputeInputType))
            {
                ComputeInputType valCast;
                if(problem.useScaleAB() == "Scalar")
                {
                    Accumulator scale = GetValue<Accumulator>(problem.alphaType(), scalePtr, 0, conjugate);
                    auto tmp = multiply<Accumulator>(val, scale);
                    valCast = static_cast<ComputeInputType>(tmp);
                }
                else
                {
                    valCast = static_cast<ComputeInputType>(val);
                }
                return static_cast<Accumulator>(static_cast<MultT>(static_cast<MathOpMultT>(valCast)));
            }

            return static_cast<Accumulator>(static_cast<MultT>(static_cast<MathOpMultT>(val)));
        }

#if defined(TENSILE_USE_FP6) || defined(TENSILE_USE_BF6) || defined(TENSILE_USE_FP4)
        template <typename Accumulator,
                  typename MathOpAccum,
                  typename Type,
                  typename ComputeInputType
            , std::enable_if_t<false
#ifdef TENSILE_USE_FP6
                                       || std::is_same<Float6x16, Type>::value
#endif // #ifdef TENSILE_USE_FP6
#ifdef TENSILE_USE_BF6
                                       || std::is_same<BFloat6x16, Type>::value
#endif // #ifdef TENSILE_USE_BF6
#ifdef TENSILE_USE_FP4
                                       || std::is_same<Float4x2, Type>::value
#endif // #ifdef TENSILE_USE_FP4
                , bool>
            = true>
        inline Accumulator getElement(
            ContractionProblemGemm const& problem,
            Type const* ptr,
            const size_t idx,
            void const* scalePtr,
            const bool conjugate)
        {
            size_t packIdx = idx / TypeInfo<Type>::Packing;
            size_t elemIdx = idx % TypeInfo<Type>::Packing;

            return static_cast<Accumulator>(ptr[packIdx].getElement(elemIdx));
        }
#endif

        template<typename Inputs, typename Accumulator, typename MathOpAccum, typename AType, typename BType, typename ComputeInputTypeA, typename ComputeInputTypeB
            , std::enable_if_t<(!std::is_same<Int8x4, AType>::value && !std::is_same<Int8x4, BType>::value), bool> = true
        >
        Accumulator multiply(
            ContractionProblemGemm const& problem,
                             ContractionInputs const&      inputs,
                             AType const*                  aPtr,
                             BType const*                  bPtr,
                             const size_t                  aIdx,
                             const size_t                  bIdx,
                             const bool                    aConjugate,
                             const bool                    bConjugate)
        {

            auto aVal = getElement<Accumulator, MathOpAccum, AType, ComputeInputTypeA>(
                problem, aPtr, aIdx, inputs.scaleA, aConjugate);
            auto bVal = getElement<Accumulator, MathOpAccum, BType, ComputeInputTypeB>(
                problem, bPtr, bIdx, inputs.scaleB, bConjugate);

            return multiply<Accumulator>(aVal, bVal);
        }

        template<typename Inputs, typename Accumulator, typename MathOpAccum, typename AType, typename BType, typename ComputeInputTypeA, typename ComputeInputTypeB
            , std::enable_if_t<std::is_same<Int8x4, AType>::value && std::is_same<Int8x4, BType>::value, bool> = true
        >
        Accumulator multiply(
            ContractionProblemGemm const& problem,
                             ContractionInputs const&      inputs,
                             AType const*                  aPtr,
                             BType const*                  bPtr,
                             const size_t                  aIdx,
                             const size_t                  bIdx,
                             const bool                    aConjugate,
                             const bool                    bConjugate)
        {
            AType aVal = Transform<AType>::Input(aPtr[aIdx], aConjugate);
            BType bVal = Transform<BType>::Input(bPtr[bIdx], bConjugate);
            return  multiply<Accumulator, MathOpAccum>(aVal, bVal);
        }

        template <typename Inputs, typename Accumulator, typename MathOpAccum>
        void ReferenceSolution<Inputs, Accumulator, MathOpAccum>::SolveCPU(
            ContractionProblemGemm const& problem,
            ContractionInputs const&      inputs,
            size_t                        elementsToValidate)
        {
            Accumulator* ws                   = nullptr;
            size_t       validationStrideGemm = 1;
            if(problem.useGradient() && problem.useBias()
               && (problem.biasSrc() == ContractionProblemGemm::D))
            {
                validationStrideGemm = 1;
                ws                   = (Accumulator*)malloc(problem.d().totalAllocatedElements()
                                          * sizeof(Accumulator));
            }
            else
            {
                if(elementsToValidate > 0
                   && elementsToValidate < problem.d().totalLogicalElements())
                    validationStrideGemm
                        = NextPrime(problem.d().totalAllocatedElements() / elementsToValidate);
            }

            // Convert void* to pointers
            typename Inputs::AType const* aPtr = (typename Inputs::AType const*)inputs.a;
            typename Inputs::BType const* bPtr = (typename Inputs::BType const*)inputs.b;
            typename Inputs::CType const* cPtr = (typename Inputs::CType const*)inputs.c;
            typename Inputs::DType*       dPtr = (typename Inputs::DType*)inputs.d;

            auto const& freeIndicesA = problem.freeIndicesA();
            auto const& freeIndicesB = problem.freeIndicesB();
            auto const& batchIndices = problem.batchIndices();
            auto const& boundIndices = problem.boundIndices();

            auto const& a    = problem.a();
            auto const& b    = problem.b();
            auto const& c    = problem.c();
            auto const& d    = problem.d();
            auto const& bias = problem.bias();
            auto const& mxsa = problem.mxsa();
            auto const& mxsb = problem.mxsb();

            bool aConjugate = false;
            bool bConjugate = false;

            for(auto const& op : problem.aOps())
                if(op.type == TensorOp::Type::ComplexConjugate)
                    aConjugate = true;

            for(auto const& op : problem.bOps())
                if(op.type == TensorOp::Type::ComplexConjugate)
                    bConjugate = true;

            std::vector<size_t> freeASize(freeIndicesA.size());
            std::vector<size_t> freeBSize(freeIndicesB.size());
            std::vector<size_t> batchSize(batchIndices.size());
            std::vector<size_t> boundSize(boundIndices.size());

            for(int i = 0; i < freeASize.size(); i++)
                freeASize[i] = problem.freeSizeA(i);
            for(int i = 0; i < freeBSize.size(); i++)
                freeBSize[i] = problem.freeSizeB(i);
            for(int i = 0; i < batchSize.size(); i++)
                batchSize[i] = problem.batchSize(i);
            for(int i = 0; i < boundSize.size(); i++)
                boundSize[i] = problem.boundSize(i);

            auto boundCount = CoordCount(boundSize.begin() + 1, boundSize.end());

            if(std::get<typename Inputs::AlphaType>(inputs.alpha)
               != static_cast<typename Inputs::AlphaType>(0))
            {
                if(inputs.a == nullptr || inputs.b == nullptr)
                {
                    std::ostringstream msg;
                    msg << "Unsupported nullptr for";
                    if(!inputs.a)
                        msg << " A";
                    if(!inputs.b)
                        msg << " B";
                    msg << " when Alpha !=0";

                    throw std::runtime_error(msg.str());
                }
            }

            Accumulator    amaxD(0);
            Accumulator    negOne(-1);
            constexpr bool notCmplxAmaxD = !std::is_same<Accumulator, std::complex<double>>()
                                           && !std::is_same<Accumulator, std::complex<float>>();

            // gemm
            omp_set_num_threads(MAX_OMP_THREADS);
#pragma omp parallel for
            for(size_t dNum = 0; dNum < d.totalLogicalElements(); dNum += validationStrideGemm)
            {
                std::vector<int64_t> aCoord(a.dimensions());
                std::vector<int64_t> bCoord(b.dimensions());
                std::vector<int64_t> cCoord(c.dimensions());
                std::vector<int64_t> dCoord(d.dimensions());
                std::vector<int64_t> biasCoord(bias.dimensions());
                std::vector<int64_t> mxsaCoord(mxsa.dimensions());
                std::vector<int64_t> mxsbCoord(mxsb.dimensions());
                CoordNumbered(
                    dNum, dCoord.begin(), dCoord.end(), d.sizes().begin(), d.sizes().end());

                for(size_t i = 0; i < problem.batchIndices().size(); i++)
                {
                    auto const& idx   = problem.batchIndices()[i];
                    size_t      coord = dCoord[idx.d];

                    aCoord[idx.a] = coord;
                    bCoord[idx.b] = coord;
                    cCoord[idx.c] = coord;
                    if(biasCoord.size() > 2)
                        biasCoord[2] = coord;
                    if(mxsaCoord.size())
                        mxsaCoord[idx.a] = coord;
                    if(mxsbCoord.size())
                        mxsbCoord[idx.b] = coord;
                }

                for(size_t i = 0; i < problem.freeIndices().size(); i++)
                {
                    auto const& idx   = problem.freeIndices()[i];
                    size_t      coord = dCoord[idx.d];

                    cCoord[idx.c] = coord;

                    if(idx.isA)
                    {
                        aCoord[idx.i] = coord;
                        if(mxsaCoord.size())
                            mxsaCoord[idx.i] = coord;
                    }
                    else
                    {
                        bCoord[idx.i] = coord;
                        if(mxsbCoord.size())
                            mxsbCoord[idx.i] = coord;
                    }
                }

                Accumulator value(0);

                // Check short-circuit for alpha = 0
                if(std::get<typename Inputs::AlphaType>(inputs.alpha)
                   != static_cast<typename Inputs::AlphaType>(0))
                {
                    for(size_t boundNum = 0; boundNum < boundCount; boundNum++)
                    {
                        std::vector<int64_t> bound(problem.boundIndices().size());
                        CoordNumbered(boundNum,
                                      bound.begin() + 1,
                                      bound.end(),
                                      boundSize.begin() + 1,
                                      boundSize.end());

                        for(int i = 1; i < bound.size(); i++)
                        {
                            aCoord[boundIndices[i].a] = bound[i];
                            bCoord[boundIndices[i].b] = bound[i];

                            if(problem.boundIndices()[i].aMirror)
                                aCoord[boundIndices[i].a]
                                    = boundSize[i] - aCoord[boundIndices[i].a] - 1;
                            if(problem.boundIndices()[i].bMirror)
                                bCoord[boundIndices[i].b]
                                    = boundSize[i] - bCoord[boundIndices[i].b] - 1;

                            if(problem.mxBlockA())
                               mxsaCoord[boundIndices[i].a] = aCoord[boundIndices[i].a] / problem.mxBlockA();
                            if(problem.mxBlockB())
                               mxsbCoord[boundIndices[i].b] = bCoord[boundIndices[i].b] / problem.mxBlockB();
                        }

                        size_t aIndex = a.index(aCoord);
                        size_t bIndex = b.index(bCoord);
                        size_t mxsaIndex = problem.mxBlockA() ? mxsa.index(mxsaCoord) : 0;
                        size_t mxsbIndex = problem.mxBlockB() ? mxsb.index(mxsbCoord) : 0;

                        auto aStride = problem.a().strides()[boundIndices[0].a];
                        auto bStride = problem.b().strides()[boundIndices[0].b];
                        auto mxsaStride = problem.mxBlockA() ? mxsa.strides()[boundIndices[0].a] : 0;
                        auto mxsbStride = problem.mxBlockB() ? mxsb.strides()[boundIndices[0].b] : 0;

                        // innermost bound calculation:
                        size_t innerMXLoop = std::max<size_t>(std::max<size_t>(problem.mxBlockA(), problem.mxBlockB()), 1);
                        for(size_t i = 0; i < boundSize[0]; i+=innerMXLoop)
                        {
                            Accumulator val(0);
                            for(size_t j = 0; j<innerMXLoop ; j++)
                            {
                                size_t idx = i + j;
                                size_t aI = boundIndices[0].aMirror ? (boundSize[0] - idx - 1) : idx;
                                size_t bI = boundIndices[0].bMirror ? (boundSize[0] - idx - 1) : idx;

                                size_t aIdx = aIndex + (aI * aStride);
                                size_t bIdx = bIndex + (bI * bStride);
                                val += multiply<Inputs, Accumulator, MathOpAccum,
                                                  typename Inputs::AType,
                                                  typename Inputs::BType,
                                                  typename Inputs::ComputeInputTypeA,
                                                  typename Inputs::ComputeInputTypeB>(
                                    problem, inputs, aPtr, bPtr, aIdx, bIdx, aConjugate, bConjugate);
                            }

                            Accumulator mxScale(1);
                            if (problem.mxBlockA())
                            {
                                size_t mxsaI = (boundIndices[0].aMirror ? (boundSize[0] - i - 1) : i) / problem.mxBlockA();
                                size_t mxsaIdx = mxsaIndex + (mxsaI * mxsaStride);
                                Accumulator sa = abs(GetValue<Accumulator>(mxsa.dataType(), inputs.mxsa, mxsaIdx, 0));
                                mxScale = multiply<Accumulator>(mxScale, sa);
                            }

                            if (problem.mxBlockB())
                            {
                                size_t mxsbI = (boundIndices[0].bMirror ? (boundSize[0] - i - 1) : i) / problem.mxBlockB();
                                size_t mxsbIdx = mxsbIndex + (mxsbI * mxsbStride);
                                Accumulator sb = abs(GetValue<Accumulator>(mxsb.dataType(), inputs.mxsb, mxsbIdx, 0));
                                mxScale = multiply<Accumulator>(mxScale, sb);
                            }
                            value += multiply<Accumulator>(val, mxScale);
                        }
                    }
                }

                auto cIndex = c.index(cCoord);
                auto dIndex = d.index(dCoord);

                // Ensure zero*nan returns zero
                Accumulator alpha = constVariantCast<Accumulator>(inputs.alpha);
                Accumulator beta  = constVariantCast<Accumulator>(inputs.beta);
                auto        zero  = static_cast<Accumulator>(0);

                if(problem.useScaleAB() == "Scalar")
                {
                    Accumulator scaleA
                        = GetValue<Accumulator>(problem.alphaType(), inputs.scaleA, 0, aConjugate);
                    Accumulator scaleB
                        = GetValue<Accumulator>(problem.alphaType(), inputs.scaleB, 0, aConjugate);
                    if constexpr(sizeof(typename Inputs::AType)
                                 <= sizeof(typename Inputs::ComputeInputTypeA))
                        alpha *= scaleA;

                    if constexpr(sizeof(typename Inputs::BType)
                                 <= sizeof(typename Inputs::ComputeInputTypeB))
                        alpha *= scaleB;
                }
                else if(problem.useScaleAB() == "Vector")
                {
                    auto posB = int(int(dNum / problem.d().sizes()[0]) % problem.d().sizes()[1]);
                    auto posA = int(dNum % problem.d().sizes()[0]);
                    Accumulator scaleA = GetValue<Accumulator>(
                        problem.alphaType(), inputs.scaleA, posA, aConjugate);
                    Accumulator scaleB = GetValue<Accumulator>(
                        problem.alphaType(), inputs.scaleB, posB, aConjugate);
                    if constexpr(sizeof(typename Inputs::AType)
                                 <= sizeof(typename Inputs::ComputeInputTypeA))
                        alpha *= scaleA;

                    if constexpr(sizeof(typename Inputs::BType)
                                 <= sizeof(typename Inputs::ComputeInputTypeB))
                        alpha *= scaleB;
                }

                auto resultD = multiply<Accumulator>(alpha, value);

                if(problem.useScaleAlphaVec())
                {
                    int pos = 0;
                    if(problem.getParams().factorDim())
                        pos = int(int(dNum / problem.d().sizes()[0]) % problem.d().sizes()[1]);
                    else
                        pos = int(dNum % problem.d().sizes()[0]);
                    Accumulator scaleAlphaVec = GetValue<Accumulator>(
                        problem.alphaType(), inputs.scaleAlphaVec, pos, aConjugate);
                    resultD *= scaleAlphaVec;
                }

                if(beta != zero)
                {
                    Accumulator cValue = multiply<Accumulator>(beta, cPtr[cIndex]);
                    if(problem.useScaleCD())
                    {
                        Accumulator scaleC = GetValue<Accumulator>(
                            problem.betaType(), inputs.scaleC, 0, aConjugate);
                        cValue *= scaleC;
                    }

                    resultD += cValue;
                }

                // bias
                if(problem.useBias() && inputs.bias && !problem.useGradient())
                {
                    auto biasIndex = problem.bias().index(biasCoord);
                    int  pos       = 0;
                    if(problem.getParams().factorDim())
                        pos = int(int(dNum / problem.d().sizes()[0]) % problem.d().sizes()[1])
                              + biasIndex;
                    else
                        pos = int(dNum % problem.d().sizes()[0]) + biasIndex;

                    Accumulator bias = GetValue<Accumulator>(
                        problem.bias().dataType(), inputs.bias, pos, aConjugate);

                    resultD += bias;
                }
                // E
                if(problem.useE() && !problem.useGradient())
                {
                    auto eIndex
                        = problem.tensors()[ContractionProblemGemm::TENSOR::E].index(dCoord);
                    SetValue<Accumulator>(
                        problem.tensors()[ContractionProblemGemm::TENSOR::E].dataType(),
                        resultD,
                        inputs.e,
                        eIndex);
                }
                // Activation adds here
                std::vector<Accumulator> actArgs;
                for(int i = 0; i < inputs.activationArgs.size(); i++)
                    actArgs.push_back(constVariantCast<Accumulator>(inputs.activationArgs[i]));
                if(problem.useGradient() && problem.activationType() != ActivationType::None
                   && problem.getParams().activationEnum() != ActivationType::None)
                {
                    Accumulator dataE = static_cast<Accumulator>(0);
                    if(problem.useE())
                    {
                        auto eIndex
                            = problem.tensors()[ContractionProblemGemm::TENSOR::E].index(
                                dCoord);
                        dataE = GetValue<Accumulator>(
                            problem.tensors()[ContractionProblemGemm::TENSOR::E].dataType(),
                            inputs.e,
                            eIndex,
                            aConjugate);
                    }
                    dataE = Activation(problem.activationType(),
                                       dataE,
                                       problem.getParams().activationEnum(),
                                       actArgs);
                    resultD *= dataE;
                }
                else
                {
                    resultD = Activation(problem.activationType(),
                                         resultD,
                                         problem.getParams().activationEnum(),
                                         actArgs);
                }

                omp_set_num_threads(MAX_OMP_THREADS);
#pragma omp critical
                {
                    if constexpr(notCmplxAmaxD)
                    {
                        if(problem.outputAmaxD())
                        {
                            Accumulator absResultD
                                = (resultD > zero) ? resultD : resultD * negOne;
                            if(absResultD > amaxD)
                                amaxD = absResultD;
                        }
                    }
                }

                if(problem.useScaleCD())
                {
                    Accumulator scaleD = GetValue<Accumulator>(
                        problem.betaType(), inputs.scaleD, 0, aConjugate);
                    resultD *= scaleD;
                }
                if(problem.useBias() && problem.useGradient()
                   && (problem.biasSrc() == ContractionProblemGemm::D))
                {
                    ws[dIndex] = resultD;
                }
                dPtr[dIndex] = SaturateCast<typename Inputs::DType>(resultD);
            }

            if(problem.outputAmaxD())
            {
                SetValue<Accumulator>(
                    problem.tensors()[ContractionProblemGemm::TENSOR::AMAXD].dataType(),
                    amaxD,
                    inputs.amaxD,
                    0);
            }

            if(problem.useGradient() && problem.useBias())
            {
                auto& biasTensor = problem.tensor(ContractionProblemGemm::TENSOR::BIAS);
                if(problem.biasSrc() == ContractionProblemGemm::D)
                {
                    auto msg = ReductionCPU<Accumulator, Accumulator>(
                        biasTensor, d, ws, inputs, elementsToValidate, 1);
                    if(!msg.empty())
                    {
                        free(ws);
                        std::runtime_error(msg.c_str());
                    }
                }
                else if(problem.biasSrc() == ContractionProblemGemm::A)
                {
                    auto reducIdx = problem.transA() ? 0 : 1;
                    auto msg      = ReductionCPU<typename Inputs::AType, Accumulator>(
                        biasTensor, a, inputs.a, inputs, elementsToValidate, reducIdx);
                    if(!msg.empty())
                    {
                        std::runtime_error(msg.c_str());
                    }
                }
                else if(problem.biasSrc() == ContractionProblemGemm::B)
                {
                    auto reducIdx = problem.transB() ? 1 : 0;
                    auto msg      = ReductionCPU<typename Inputs::BType, Accumulator>(
                        biasTensor, b, inputs.b, inputs, elementsToValidate, reducIdx);
                    if(!msg.empty())
                    {
                        std::runtime_error(msg.c_str());
                    }
                }
                else
                {
                    std::string msg = "Unsupported bias reduction source "
                                      + std::to_string(problem.biasSrc()) + ".";
                    throw std::runtime_error(msg.c_str());
                }
                free(ws);
            }
        }

        template <typename Inputs, typename Accumulator, typename MathOpAccum>
        void ReferenceSolution<Inputs, Accumulator, MathOpAccum>::SolveCPU(
            ContractionProblemGroupedGemm const& problem,
            ContractionGroupedInputs const&      inputs,
            size_t                               elementsToValidate)
        {
            for(int idx = 0; idx < problem.gemms.size(); idx++)
            {
                ReferenceSolution<Inputs, Accumulator, MathOpAccum>::SolveCPU(
                    problem.gemms[idx], inputs.grouped[idx], elementsToValidate);
            }
        }

        uint64_t getInputContractionInputsTypeId(ContractionProblemGemm const& problem)
        {
            // retreive alpha/beta type set via setAlpha/BetaType()
            auto alphaType = problem.alphaType();
            auto betaType  = problem.betaType();

            // Backward-compatible: when setAlpha/BetaType() wasn't called, use the old way
            // Could remove after rocBLAS is updated
            if(alphaType == rocisa::DataType::None)
            {
                alphaType = problem.a().dataType() == rocisa::DataType::BFloat16
                                ? rocisa::DataType::Float
                                : problem.d().dataType();
            }
            if(betaType == rocisa::DataType::None)
            {
                betaType = alphaType;
            }

            if(problem.useE())
            {
                if(alphaType != betaType)
                {
                    throw std::runtime_error("Alpha type and beta type must be the same.");
                }
            }

            return TensileLite::GemmTypeId(problem.a().dataType(),
                                           problem.b().dataType(),
                                           problem.c().dataType(),
                                           problem.d().dataType(),
                                           alphaType,
                                           betaType,
                                           problem.computeInputTypeA(),
                                           problem.computeInputTypeB());
        }

        template <typename Problem, typename Inputs>
        void SolveCPUTemplates(uint64_t const& contractionInputsTypeId,
                               Problem const&  problem,
                               Inputs const&   inputs,
                               size_t          elementsToValidate)
        {
            bool isHPA = false;
            if constexpr(std::is_same<ContractionProblemGemm, Problem>::value)
            {
                isHPA = problem.highPrecisionAccumulate();
            }
            else if constexpr(std::is_same<ContractionProblemGroupedGemm, Problem>::value)
            {
                isHPA = problem.gemms[0].highPrecisionAccumulate();
            }

            switch(contractionInputsTypeId)
            {
            case TypedGemm_S_S_S::TypeId():
            {
                if(problem.f32XdlMathOp() == rocisa::DataType::XFloat32)
                    return ReferenceSolution<TypedGemm_S_S_S, float, XFloat32>::SolveCPU(
                        problem, inputs, elementsToValidate);
                else
                    return ReferenceSolution<TypedGemm_S_S_S>::SolveCPU(
                        problem, inputs, elementsToValidate);
            }
            case TypedGemm_D_D_D::TypeId():
            {
                return ReferenceSolution<TypedGemm_D_D_D>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_C_C_C::TypeId():
            {
                return ReferenceSolution<TypedGemm_C_C_C>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_Z_Z_Z::TypeId():
            {
                return ReferenceSolution<TypedGemm_Z_Z_Z>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
#ifdef TENSILE_USE_HALF
            case TypedGemm_H_H_H::TypeId():
            {
                if(isHPA)
                {
                    return ReferenceSolution<TypedGemm_H_H_H, float>::SolveCPU(
                        problem, inputs, elementsToValidate);
                }
                else
                {
                    return ReferenceSolution<TypedGemm_H_H_H>::SolveCPU(
                        problem, inputs, elementsToValidate);
                }
            }
            case TypedGemm_H_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_H_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_H_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_H_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_SH_H_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_SH_H_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_HS_H_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_HS_H_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
#endif // TENSILE_USE_HALF
            case TypedGemm_I8x4_I32_I32::TypeId():
            {
                return ReferenceSolution<TypedGemm_I8x4_I32_I32>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_I32_I32_I32::TypeId():
            {
                return ReferenceSolution<TypedGemm_I32_I32_I32>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_I8_I8_I32::TypeId():
            {
                return ReferenceSolution<TypedGemm_I8_I8_I32, int32_t>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_I8_I32_I32::TypeId():
            {
                return ReferenceSolution<TypedGemm_I8_I32_I32>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_I8_I32_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_I8_I32_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_I8_I8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_I8_I8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_I8_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_I8_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
#ifdef TENSILE_USE_BF16
            case TypedGemm_B_B_S::TypeId():
            {
                if(isHPA)
                {
                    return ReferenceSolution<TypedGemm_B_B_S, float>::SolveCPU(
                        problem, inputs, elementsToValidate);
                }
                else
                {
                    return ReferenceSolution<TypedGemm_B_B_S>::SolveCPU(
                        problem, inputs, elementsToValidate);
                }
            }
            case TypedGemm_S_B_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_S_B_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_H_B_H_S::TypeId():
            {
                if(isHPA)
                {
                    return ReferenceSolution<TypedGemm_H_B_H_S, float>::SolveCPU(
                        problem, inputs, elementsToValidate);
                }
                else
                {
                    return ReferenceSolution<TypedGemm_H_B_H_S>::SolveCPU(
                        problem, inputs, elementsToValidate);
                }
            }
            case TypedGemm_I8_B_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_I8_B_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
#endif // TENSILE_USE_BF16
#ifdef TENSILE_USE_FP8_BF8
            case TypedGemm_F8_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8_S_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8_B_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8_B_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8_F8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8_F8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8_B8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8_B8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8_S_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8_B_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8_B_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8_F8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8_F8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8_B8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8_B8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            // hybrid
            case TypedGemm_F8B8_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8B8_S_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8B8_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8B8_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8B8_B_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8B8_B_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8B8_F8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8B8_F8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8B8_B8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8B8_B8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8F8_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8F8_S_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8F8_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8F8_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8F8_B_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8F8_B_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8F8_F8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8F8_F8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8F8_B8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8F8_B8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_H_F8B8_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_H_F8B8_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_H_B8F8_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_H_B8F8_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }

            // F8 NANOO
            case TypedGemm_F8N_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8N_S_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8N_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8N_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8N_B_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8N_B_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8N_F8N_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8N_F8N_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8N_B8N_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8N_B8N_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8N_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8N_S_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8N_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8N_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8N_B_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8N_B_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8N_F8N_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8N_F8N_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8N_B8N_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8N_B8N_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            // hybrid - NANOO
            case TypedGemm_F8B8N_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8B8N_S_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8B8N_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8B8N_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8B8N_B_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8B8N_B_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8B8N_F8N_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8B8N_F8N_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8B8N_B8N_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8B8N_B8N_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8F8N_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8F8N_S_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8F8N_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8F8N_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8F8N_B_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8F8N_B_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8F8N_F8N_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8F8N_F8N_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8F8N_B8N_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8F8N_B8N_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_H_F8B8N_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_H_F8B8N_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_H_B8F8N_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_H_B8F8N_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
#ifdef TENSILE_USE_HALF
            case TypedGemm_H_F8_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_H_F8_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_H_B8_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_H_B8_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_HF8_H_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_HF8_H_S_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8H_H_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8H_H_S_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_HF8_H_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_HF8_H_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8H_H_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8H_H_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_HF8_H_FP8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_HF8_H_FP8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8H_H_FP8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8H_H_FP8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_HF8_FP8_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_HF8_FP8_S_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8H_FP8_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8H_FP8_S_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_HF8_FP8_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_HF8_FP8_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8H_FP8_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8H_FP8_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_HF8_FP8_FP8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_HF8_FP8_FP8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8H_FP8_FP8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8H_FP8_FP8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }

            // F8 NANOO
            case TypedGemm_H_F8N_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_H_F8N_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_H_B8N_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_H_B8N_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_HF8N_H_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_HF8N_H_S_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8NH_H_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8NH_H_S_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_HF8N_H_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_HF8N_H_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8NH_H_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8NH_H_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            // TODO:; why FP8, not F8... need to change it to FP8N???
            case TypedGemm_HF8N_H_FP8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_HF8N_H_FP8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8NH_H_FP8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8NH_H_FP8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_HF8N_FP8_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_HF8N_FP8_S_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8NH_FP8_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8NH_FP8_S_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_HF8N_FP8_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_HF8N_FP8_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8NH_FP8_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8NH_FP8_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_HF8N_FP8_FP8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_HF8N_FP8_FP8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F8NH_FP8_FP8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8NH_FP8_FP8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
#endif // TENSILE_USE_HALF
#endif // TENSILE_USE_FP8_BF8

#ifdef TENSILE_USE_FP6
            case TypedGemm_F6_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F6_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
#endif //TENSILE_USE_FP6
#ifdef TENSILE_USE_BF6
            case TypedGemm_BF6_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_BF6_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
#endif //TENSILE_USE_BF6
#if defined(TENSILE_USE_FP6) && defined(TENSILE_USE_BF6)
            case TypedGemm_F6B6_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F6B6_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B6F6_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B6F6_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
#endif // defined(TENSILE_USE_FP6) && defined(TENSILE_USE_BF6)
#ifdef TENSILE_USE_FP4
            case TypedGemm_F4_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F4_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
#endif //TENSILE_USE_FP4
#if defined(TENSILE_USE_FP4) && defined(TENSILE_USE_FP6)
            case TypedGemm_F4F6_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F4F6_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F6F4_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F6F4_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
#endif // defined(TENSILE_USE_FP4) && defined(TENSILE_USE_FP6)
#if defined(TENSILE_USE_FP4) && defined(TENSILE_USE_BF6)
            case TypedGemm_F4B6_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F4B6_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B6F4_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B6F4_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
#endif // defined(TENSILE_USE_FP4) && defined(TENSILE_USE_BF6)
#if defined(TENSILE_USE_FP8_BF8) && defined(TENSILE_USE_FP4)
            // DestDataType: S
            case TypedGemm_F8F4_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8F4_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F4F8_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F4F8_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8F4_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8F4_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F4B8_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F4B8_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            // DestDataType: F8
            case TypedGemm_F8F4_F8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8F4_F8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F4F8_F8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F4F8_F8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8F4_F8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8F4_F8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F4B8_F8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F4B8_F8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            // DestDataType: B8
            case TypedGemm_F8F4_B8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8F4_B8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F4F8_B8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F4F8_B8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8F4_B8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8F4_B8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F4B8_B8_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F4B8_B8_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
#endif // defined(TENSILE_USE_FP8_BF8) && defined(TENSILE_USE_FP4)
#if defined(TENSILE_USE_FP8_BF8) && defined(TENSILE_USE_FP4) && defined(TENSILE_USE_HALF)
            // DestDataType: H
            case TypedGemm_F8F4_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8F4_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F4F8_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F4F8_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8F4_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8F4_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F4B8_H_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F4B8_H_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
#endif // defined(TENSILE_USE_FP8_BF8) && defined(TENSILE_USE_FP4) && defined(TENSILE_USE_HALF)
#if defined(TENSILE_USE_FP8_BF8) && defined(TENSILE_USE_FP4) && defined(TENSILE_USE_BF16)
            // DestDataType: B
            case TypedGemm_F8F4_B_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8F4_B_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F4F8_B_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F4F8_B_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8F4_B_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8F4_B_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F4B8_B_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F4B8_B_S, float>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
#endif // defined(TENSILE_USE_FP8_BF8) && defined(TENSILE_USE_FP4) && defined(TENSILE_USE_BF16)
#if defined(TENSILE_USE_FP8_BF8) && defined(TENSILE_USE_FP6)
            case TypedGemm_F8F6_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8F6_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F6F8_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F6F8_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8F6_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8F6_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_F6B8_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F6B8_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
#endif // defined(TENSILE_USE_FP8_BF8) && defined(TENSILE_USE_FP6)
#if defined(TENSILE_USE_FP8_BF8) && defined(TENSILE_USE_BF6)
            case TypedGemm_F8B6_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_F8B6_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B6F8_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B6F8_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B8B6_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B8B6_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
            case TypedGemm_B6B8_S_S::TypeId():
            {
                return ReferenceSolution<TypedGemm_B6B8_S_S>::SolveCPU(
                    problem, inputs, elementsToValidate);
            }
#endif // defined(TENSILE_USE_FP8_BF8) && defined(TENSILE_USE_BF6)
            default:;
            }

            throw std::runtime_error("Data type not implemented.");
        }

        void SolveGemmCPU(ContractionProblemGemm const& problem,
                          ContractionInputs const&      inputs,
                          size_t                        elementsToValidate,
                          bool                          tryFastPath)
        {

            // The fast solver computes all elements. If the number of elements to validate
            // is in [0, sparsityThreshold * totalElements), skip this solver, falling through to another
            // solver that handles the partial validation sparsity efficiently.
            double sparsityThreshold        = 0.2;
            bool   isDenseEnoughForFastPath = true;
            if(elementsToValidate >= 0
               && elementsToValidate < sparsityThreshold * problem.d().totalLogicalElements())
            {
                isDenseEnoughForFastPath = false;
            }

            if(tryFastPath && isDenseEnoughForFastPath && isFastPathEligible(problem))
            {
                ScopedTimer timer("solve_cpu_fast");
                solveCPUFastInF32(problem, inputs);
                return;
            }

            {
                ScopedTimer timer("solve_cpu_slow");
                auto contractionInputsTypeId = getInputContractionInputsTypeId(problem);
                SolveCPUTemplates(contractionInputsTypeId, problem, inputs, elementsToValidate);
            }
        }

        void SolveCPU(ContractionProblem const* problem,
                      ProblemInputs const*      inputs,
                      size_t                    elementsToValidate)
        {

            if(auto groupedProblem = dynamic_cast<ContractionProblemGroupedGemm const*>(problem))
            {
                auto refInput = dynamic_cast<ContractionGroupedInputs const*>(inputs);
                if(!refInput)
                {
                    throw std::runtime_error("Unable to cast input to ContractionGroupedInputs.");
                }
                if(groupedProblem->gemms.size() != refInput->grouped.size())
                {
                    throw std::runtime_error("Mismatched number of grouped problems and inputs.");
                }
                for(uint64_t i = 0; i < groupedProblem->gemms.size(); ++i)
                {
                    ContractionProblemGemm problem = groupedProblem->gemms[i];
                    ContractionInputs      input   = refInput->grouped[i];
                    SolveGemmCPU(problem, input, elementsToValidate);
                }
                return;
            }

            else if(auto gemmProblem = dynamic_cast<ContractionProblemGemm const*>(problem))
            {
                auto refInput = dynamic_cast<ContractionInputs const*>(inputs);
                if(!refInput)
                {
                    throw std::runtime_error("Unable to cast input to ContractionInputs.");
                }
                SolveGemmCPU(*gemmProblem, *refInput, elementsToValidate);
            }

            else
            {
                throw std::runtime_error(
                    "[Reference] Failed to cast to any ContractionProblem");
            }
        }
    } // namespace Client
} // namespace TensileLite
