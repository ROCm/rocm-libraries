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

#include <tuple>
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
    template <typename... Args>
    struct Concat;

    template <typename Arg>
    struct Concat<Arg>
    {
        using Result = decltype(std::make_tuple(Arg()));
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
        using Result = decltype(std::make_tuple(Lhs(), Rhs()));
    };

    template <typename Lhs, typename... Rhs>
    struct Concat<Lhs, std::tuple<Rhs...>>
    {
        using Result = decltype(std::tuple_cat(std::make_tuple(Lhs()), std::tuple<Rhs...>()));
    };

    template <typename... Lhs, typename... Rhs>
    struct Concat<std::tuple<Lhs...>, std::tuple<Rhs...>>
    {
        using Result = decltype(std::tuple_cat(std::tuple<Lhs...>(), std::tuple<Rhs...>()));
    };

    template <typename... Lhs, typename Rhs>
    struct Concat<std::tuple<Lhs...>, Rhs>
    {
        using Result = decltype(std::tuple_cat(std::tuple<Lhs...>(), std::make_tuple(Rhs())));
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
        using Result = decltype(std::make_tuple(typename Concat<Lhs, Rhs>::Result()));
    };

    template <typename Lhs, typename Rhs>
    struct CombineOne<Lhs, std::tuple<Rhs>>
    {
        using Result = decltype(std::make_tuple(typename Concat<Lhs, Rhs>::Result()));
    };

    template <typename Lhs, typename Rhs0, typename... Rhs>
    struct CombineOne<Lhs, std::tuple<Rhs0, Rhs...>>
    {
        using Mine   = typename Concat<Lhs, Rhs0>::Result;
        using Next   = CombineOne<Lhs, std::tuple<Rhs...>>;
        using Result = decltype(std::tuple_cat(std::make_tuple(Mine()), typename Next::Result()));
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
        using Result = decltype(std::tuple_cat(Mine(), typename Next::Result()));
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

    template <typename KernelParams, typename... Next, class GeneratorImpl>
    struct KernelGenerator<std::tuple<KernelParams, Next...>, GeneratorImpl>
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
            // Generates the kernel for the current set of KernelParams
            auto gen_kernel
                = [](ResultT& k) { k.push_back(GeneratorImpl::generate(KernelParams())); };

            // Advances to the next set of KernelParams
            auto next_kernel = [](ResultT& k) {
                KernelGenerator<std::tuple<Next...>, GeneratorImpl>::generate(k);
            };

            if constexpr(contains_type_v<float8_t,
                                         KernelParams> || contains_type_v<bfloat8_t, KernelParams>)
            {
                if constexpr(!(bool)ROCWMMA_FP8)
                {
                    // Current KernelParams have f8: skip kernel on unsupported arch.
                    next_kernel(kernels);
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
                    if(arch != DeviceInfo::hipGcnArch_t::GFX1200
                       && arch != DeviceInfo::hipGcnArch_t::GFX1201
                       && arch != DeviceInfo::hipGcnArch_t::GFX950)
                    {
                        // Current KernelParams have f8: skip kernel on host.
                        next_kernel(kernels);
                        return;
                    }
                }

                // Generate kernel
                gen_kernel(kernels);
                next_kernel(kernels);
            }
            else if constexpr(contains_type_v<
                                  float8_fnuz_t,
                                  KernelParams> || contains_type_v<bfloat8_fnuz_t, KernelParams>)
            {
                if constexpr(!(bool)ROCWMMA_FP8_FNUZ)
                {
                    // Current KernelParams have f8_fnuz: skip kernel on unsupported arch.
                    next_kernel(kernels);
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
                        next_kernel(kernels);
                        return;
                    }
                }

                // Generate kernel
                gen_kernel(kernels);
                next_kernel(kernels);
            }
            else
            {
                // Generate kernel
                gen_kernel(kernels);
                next_kernel(kernels);
            }
        }
    };

} // namespace rocwmma

#endif // ROCWMMA_KERNEL_GENERATOR_HPP
