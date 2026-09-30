/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef ROCWMMA_KERNEL_GENERATOR_HPP
#define ROCWMMA_KERNEL_GENERATOR_HPP

#include <functional>
#include <tuple>
#include <type_traits>
#include <vector>

#include "hip_device.hpp"
#include <rocwmma/internal/types.hpp>
#include <rocwmma/internal/utility/sequence.hpp>

namespace rocwmma
{

    ///
    /// TestParams: nested tuple of kernel parameters to build
    /// a set of test kernels. E.g.
    ///
    /// tuple< tuple<KernelParams0...>, tuple<KernelParams1...>, ...>;
    ///
    /// KernelParams: tuple of Params to build a SINGLE kernel. E.g.
    ///
    /// tuple<KernelParams...>;
    ///

    /// Workflow:
    ///
    /// 1. Build a set of testing kernels with TestParams.
    ///
    /// 2. Map KernelParams to the kernel's template arguments.
    /// This is the responsibility of the kernel generator
    /// implementation (KernelGeneratorImpl).
    ///
    /// 3. Instantiate each set of KernelParams with the impl,
    /// and return a vector of kernels.
    ///
    /// The following utilities are used to build (1).
    /// - Concat<>
    /// - CombineOne<>
    /// - CombineMany<>
    /// - CombineLists<>
    ///
    /// for 2) and 3) KernelGenerator class takes the
    /// KernelGeneratorImpl and instantiates one kernel
    /// per tuple of KernelParams from TestParams (1). It
    /// returns the kernels as a vector<KernelI>.

    /// There are several classes that provide functionality
    /// to build combinations of basic types.
    ///
    /// Concat: Concatenation of types together.
    /// E.g.
    /// Concat( A ) = tuple<A>
    /// Concat( A, B ) = tuple<A, B>
    /// Concat( tuple<A>, B ) = tuple<A, B>
    /// Concat( A, tuple<B> ) = tuple<A, B>
    /// Concat( tuple<A>, tuple<B> ) = tuple<A, B>
    /// Concat( A, B, C, ...) = tuple<A, B, C, ...>
    namespace detail
    {
        // C++17 equivalent of make_tuple's decay/reference_wrapper unwrapping,
        // without instantiating tuple constructors just to compute a type.
        template <typename T>
        struct KernelParamDecay
        {
            using type = T;
        };

        template <typename T>
        struct KernelParamDecay<std::reference_wrapper<T>>
        {
            using type = T&;
        };

        template <typename T>
        using KernelParamDecayT = typename KernelParamDecay<std::decay_t<T>>::type;
    } // namespace detail

    template <typename... Args>
    struct Concat;

    template <typename Arg>
    struct Concat<Arg>
    {
        using Result = std::tuple<detail::KernelParamDecayT<Arg>>;
    };

    template <typename... Args>
    struct Concat<std::tuple<Args...>>
    {
        using Result = std::tuple<Args...>;
    };

    template <typename Lhs, typename Rhs, typename... Rest>
    struct Concat<Lhs, Rhs, Rest...>
    {
        using Result = typename Concat<typename Concat<Lhs, Rhs>::Result, Rest...>::Result;
    };

    template <typename Lhs, typename Rhs>
    struct Concat<Lhs, Rhs>
    {
        using Result = std::tuple<detail::KernelParamDecayT<Lhs>, detail::KernelParamDecayT<Rhs>>;
    };

    template <typename Lhs, typename... Rhs>
    struct Concat<Lhs, std::tuple<Rhs...>>
    {
        using Result = std::tuple<detail::KernelParamDecayT<Lhs>, Rhs...>;
    };

    template <typename... Lhs, typename... Rhs>
    struct Concat<std::tuple<Lhs...>, std::tuple<Rhs...>>
    {
        using Result = std::tuple<Lhs..., Rhs...>;
    };

    template <typename... Lhs, typename Rhs>
    struct Concat<std::tuple<Lhs...>, Rhs>
    {
        using Result = std::tuple<Lhs..., detail::KernelParamDecayT<Rhs>>;
    };

    /// CombineOne: Creates combinatorial pairs of LHS
    /// with EACH type of RHS if RHS is a tuple.
    /// NOTE: First level of tuples are collapsed into
    /// nested tuples, as required by the generator.
    ///
    /// E.g.
    /// CombineOne( A, B ) = tuple< tuple<A, B> >
    /// CombineOne( A, tuple<B> ) = tuple< tuple<A, B> >
    /// CombineOne( A, tuple<B, C, D> ) = tuple< tuple<A, B>,
    ///                                          tuple<A, C>,
    ///                                          tuple<A, D>>
    /// CombineOne( tuple<A, B>, tuple<C, D> ) = tuple< tuple<A, B, C>,
    ///                                          tuple<A, B, D>>
    ///

    template <typename Lhs, typename Rhs>
    struct CombineOne
    {
        using Result = std::tuple<typename Concat<Lhs, Rhs>::Result>;
    };

    template <typename Lhs, typename Rhs0, typename... Rhs>
    struct CombineOne<Lhs, std::tuple<Rhs0, Rhs...>>
    {
        using Result
            = std::tuple<typename Concat<Lhs, Rhs0>::Result, typename Concat<Lhs, Rhs>::Result...>;
    };

    /// CombineMany: Creates combinatorial pairs two lists:
    /// EACH type of LHS with EACH type of RHS.
    /// NOTE: First level of tuples are collapsed into
    /// nested tuples, as required by the generator.
    ///
    /// E.g.
    /// CombineMany( A, B ) = tuple< tuple<A, B> >
    /// CombineMany( A, tuple<B> ) = tuple< tuple<A, B> >
    /// CombineMany( A, tuple<B, C, D> ) = tuple< tuple<A, B>,
    ///                                           tuple<A, C>,
    ///                                           tuple<A, D>>
    /// CombineMany( tuple<A, B>, tuple<C, D> ) = tuple< tuple<A, C>,
    ///                                                 tuple<A, D>,
    ///                                                 tuple<B, C>,
    ///                                                 tuple<B, D>>
    template <typename Lhs, typename Rhs>
    struct CombineMany
    {
        using Result = typename CombineOne<Lhs, Rhs>::Result;
    };

    template <typename Lhs, typename Rhs>
    struct CombineMany<std::tuple<Lhs>, Rhs>
    {
        using Result = typename CombineOne<Lhs, Rhs>::Result;
    };

    template <typename Lhs0, typename... Lhs, typename Rhs>
    struct CombineMany<std::tuple<Lhs0, Lhs...>, Rhs>
    {
        using Mine   = typename CombineOne<Lhs0, Rhs>::Result;
        using Next   = CombineMany<std::tuple<Lhs...>, Rhs>;
        using Result = typename Concat<Mine, typename Next::Result>::Result;
    };

    /// CombineLists: Creates combinatorial sets from multiple lists.
    ///
    /// E.g:
    /// CombineLists( tuple< tuple<A, B> >, tuple< tuple<C, D> > ) =
    ///     tuple< tuple<A, B, C, D> >
    /// CombineLists( tuple< tuple<A, B>, tuple<C, D> >, tuple< tuple<E, F>, tuple<G, H> > ) =
    ///     tuple< tuple<A, B, E, F>, tuple<A, B, G, H>, tuple<C, D, E, F>, tuple<C, D, G, H> >

    template <typename List0, typename... Lists>
    struct CombineLists
    {
        using Result = typename CombineMany<List0, typename CombineLists<Lists...>::Result>::Result;
    };

    template <typename List>
    struct CombineLists<List>
    {
        using Result = List;
    };

    // Override wrapper to apply to tuples

    namespace detail
    {
        template <typename DataT, typename... TupleTs>
        struct contains_type<DataT, std::tuple<TupleTs...>> : contains_type<DataT, TupleTs...>
        {
        };

    } // namespace detail

    /// Kernel Generator
    /// Requires two inputs:
    /// TestParams: nested tuple of KernelParams
    /// GeneratorImpl: a generator class that
    /// maps KernelParams to instantiation of the
    /// actual kernel.
    ///
    /// NOTE: The GeneratorImpl class decides the final
    /// generated kernel instantiated type. This class
    /// simply returns a vector of generated kernels of
    /// this type.
    template <typename TestParams, class GeneratorImpl>
    struct KernelGenerator
    {
        template <typename... Ts>
        ROCWMMA_HOST static void generate(Ts...)
        {
        }
    };

    template <typename First, typename... Params, class GeneratorImpl>
    struct KernelGenerator<std::tuple<First, Params...>, GeneratorImpl>
    {
        using ResultT = std::vector<typename GeneratorImpl::ResultT>;
        ROCWMMA_HOST static ResultT generate()
        {
            auto result = ResultT();
            generate(result);
            return result;
        }

        ROCWMMA_HOST static void generate(ResultT& kernels)
        {
            // Initializer-list evaluation preserves order without instantiating
            // every suffix of the list or exceeding the compiler's fold depth.
            using Expand = int[];
            (void)Expand{0, (generateOne<First>(kernels), 0), (generateOne<Params>(kernels), 0)...};
        }

    private:
        template <typename KernelParams>
        ROCWMMA_HOST static void generateOne(ResultT& kernels)
        {
            auto gen_kernel
                = [](ResultT& k) { k.push_back(GeneratorImpl::generate(KernelParams())); };

            if constexpr(contains_type_v<float8_t, KernelParams>
                         || contains_type_v<bfloat8_t, KernelParams>)
            {
                if constexpr(!(bool)ROCWMMA_FP8)
                {
                    // Current KernelParams have f8: skip kernel on unsupported arch.
                    return;
                }

                // Quirk: Here, the host code supports F8, but doesn't know
                // if the runtime target supports it. Make sure the runtime
                // target can support this type, otherwise don't generate the kernel.
                if constexpr((bool)ROCWMMA_ARCH_HOST)
                {
                    // Only gfx950 and gfx12 devices support f8
                    using DeviceInfo = HipDevice;
                    auto arch        = DeviceInfo::instance()->getGcnArch();
                    if(arch != DeviceInfo::hipGcnArch_t::GFX950
                       && arch != DeviceInfo::hipGcnArch_t::GFX1200
                       && arch != DeviceInfo::hipGcnArch_t::GFX1201
                       && arch != DeviceInfo::hipGcnArch_t::GFX1250)
                    {
                        // Current KernelParams have f8: skip kernel on host.
                        return;
                    }
                }

                // Generate kernel
                gen_kernel(kernels);
            }
            else if constexpr(contains_type_v<float8_fnuz_t, KernelParams>
                              || contains_type_v<bfloat8_fnuz_t, KernelParams>)
            {
                if constexpr(!(bool)ROCWMMA_FP8_FNUZ)
                {
                    // Current KernelParams have f8_fnuz: skip kernel on unsupported arch.
                    return;
                }

                // Quirk: Here, the host code supports F8_fnuz, but doesn't know
                // if the runtime target supports it. Make sure the runtime
                // target can support this type, otherwise don't generate the kernel.
                if constexpr((bool)ROCWMMA_ARCH_HOST)
                {
                    // Only gfx94* devices support f8_fnuz
                    using DeviceInfo = HipDevice;
                    auto arch        = DeviceInfo::instance()->getGcnArch();
                    if(arch != DeviceInfo::hipGcnArch_t::GFX942)
                    {
                        // Current KernelParams have f8_fnuz: skip kernel on host.
                        return;
                    }
                }

                // Generate kernel
                gen_kernel(kernels);
            }
            else
            {
                // Generate kernel
                gen_kernel(kernels);
            }
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_KERNEL_GENERATOR_HPP
