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
#ifndef ROCWMMA_REGISTER_LAYOUT_TRANSFORMS_IMPL_HPP
#define ROCWMMA_REGISTER_LAYOUT_TRANSFORMS_IMPL_HPP

#include "layout.hpp"
#include "layout_traits.hpp"
#include "transforms/transforms.hpp"

namespace rocwmma
{
    namespace LayoutTransforms_impl
    {
        using RegisterLayout::Format;

        // Specific transform from one format to another
        template <RegisterLayout::Format Src, RegisterLayout::Format Dst, typename... Traits>
        struct register_layout_transform_impl;

        // Non-interleaved formats
        // SOA <-> AOS
        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::SOA, Format::AOS, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::SoaToAos<LayoutTraits::BlockDim,
                                            LayoutTraits::MaxVectorWidth>::exec(forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::AOS, Format::SOA, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::AosToSoa<LayoutTraits::BlockDim,
                                            LayoutTraits::MaxVectorWidth>::exec(forward<VecT>(v));
            }
        };

        // Non-interleaved gfx11 formats
        // SOA, AOS <-> WMMA input
        // SOA, AOS <-> WMMA acc
        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::SOA, Format::WMMA_INPUT_GFX11, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::to_wmma_input_gfx11(forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::AOS, Format::WMMA_INPUT_GFX11, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                using xform0
                    = register_layout_transform_impl<Format::AOS, Format::SOA, LayoutTraits>;
                using xform1 = register_layout_transform_impl<Format::SOA,
                                                              Format::WMMA_INPUT_GFX11,
                                                              LayoutTraits>;
                return xform1::exec(xform0(forward<VecT>(v)));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::WMMA_INPUT_GFX11, Format::SOA, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::from_wmma_input_gfx11(forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::WMMA_INPUT_GFX11, Format::AOS, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                using xform0 = register_layout_transform_impl<Format::WMMA_INPUT_GFX11,
                                                              Format::SOA,
                                                              LayoutTraits>;
                using xform1
                    = register_layout_transform_impl<Format::SOA, Format::AOS, LayoutTraits>;
                return xform1::exec(xform0(forward<VecT>(v)));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::SOA, Format::WMMA_ACC_GFX11, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::to_wmma_acc_gfx11(forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::AOS, Format::WMMA_ACC_GFX11, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                using xform0
                    = register_layout_transform_impl<Format::AOS, Format::SOA, LayoutTraits>;
                using xform1 = register_layout_transform_impl<Format::SOA,
                                                              Format::WMMA_ACC_GFX11,
                                                              LayoutTraits>;
                return xform1::exec(xform0(forward<VecT>(v)));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::WMMA_ACC_GFX11, Format::SOA, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::from_wmma_acc_gfx11(forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::WMMA_ACC_GFX11, Format::AOS, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                using xform0 = register_layout_transform_impl<Format::WMMA_ACC_GFX11,
                                                              Format::SOA,
                                                              LayoutTraits>;
                using xform1
                    = register_layout_transform_impl<Format::SOA, Format::AOS, LayoutTraits>;
                return xform1::exec(xform0(forward<VecT>(v)));
            }
        };

        // Interleaved formats
        // SOA_INT <-> AOS_INT
        // SOA_INT, AOS_INT <-> A-major acc fmt
        // SOA_INT, AOS_INT <-> B-major acc fmt
        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::SOA_INT, Format::AOS_INT, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::soa_int_to_aos_int<LayoutTraits::KPerThread>(forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::AOS_INT, Format::SOA_INT, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::aos_int_to_soa_int<LayoutTraits::DimPerThread>(forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::SOA_INT,
                                              Format::ACC_INT_A_MAJOR,
                                              LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                // Vector size per acc block
                constexpr uint32_t AccVecSize
                    = LayoutTraits::MmaDim * LayoutTraits::MmaDim / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t MmaBlocksA = LayoutTraits::KDim / LayoutTraits::MmaDim;

                // Note: Retrieve VW of native accumulator
                constexpr uint32_t AccMaxVW = detail::MaxVWSelector<accumulator,
                                                                    LayoutTraits::BlockDim,
                                                                    LayoutTraits::KDim,
                                                                    typename LayoutTraits::DataT,
                                                                    void>::Result;

                return Transforms::soa_int_to_mma_acc_int_a_major<AccVecSize, MmaBlocksA, AccMaxVW>(
                    forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::AOS_INT,
                                              Format::ACC_INT_A_MAJOR,
                                              LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                // Vector size per acc block
                constexpr uint32_t AccVecSize
                    = LayoutTraits::MmaDim * LayoutTraits::MmaDim / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t MmaBlocksA = LayoutTraits::KDim / LayoutTraits::MmaDim;
                constexpr uint32_t MmaBlocksB = LayoutTraits::BlockDim / LayoutTraits::MmaDim;

                // Note: Retrieve VW of native accumulator
                constexpr uint32_t AccMaxVW = detail::MaxVWSelector<accumulator,
                                                                    LayoutTraits::BlockDim,
                                                                    LayoutTraits::KDim,
                                                                    typename LayoutTraits::DataT,
                                                                    void>::Result;

                return Transforms::
                    aos_int_to_mma_acc_int_a_major<AccVecSize, MmaBlocksA, MmaBlocksB, AccMaxVW>(
                        forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::ACC_INT_A_MAJOR,
                                              Format::SOA_INT,
                                              LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                // Vector size per acc block
                constexpr uint32_t AccVecSize
                    = LayoutTraits::MmaDim * LayoutTraits::MmaDim / Constants::AMDGCN_WAVE_SIZE;
                constexpr uint32_t MmaBlocksB = LayoutTraits::BlockDim / LayoutTraits::MmaDim;

                // Note: Retrieve VW of native accumulator
                constexpr uint32_t AccMaxVW = detail::MaxVWSelector<accumulator,
                                                                    LayoutTraits::BlockDim,
                                                                    LayoutTraits::KDim,
                                                                    typename LayoutTraits::DataT,
                                                                    void>::Result;

                return Transforms::mma_acc_int_a_major_to_soa_int<AccVecSize, MmaBlocksB, AccMaxVW>(
                    forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::ACC_INT_A_MAJOR,
                                              Format::AOS_INT,
                                              LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                // Vector size per acc block
                constexpr uint32_t AccVecSize
                    = LayoutTraits::MmaDim * LayoutTraits::MmaDim / Constants::AMDGCN_WAVE_SIZE;

                // Note: Retrieve VW of native accumulator
                constexpr uint32_t AccMaxVW = detail::MaxVWSelector<accumulator,
                                                                    LayoutTraits::BlockDim,
                                                                    LayoutTraits::KDim,
                                                                    typename LayoutTraits::DataT,
                                                                    void>::Result;

                return Transforms::mma_acc_int_a_major_to_aos_int<AccVecSize, AccMaxVW>(
                    forward<VecT>(v));
            }
        };

        // Interleaved gfx11 formats
        // SOA_INT, AOS_INT <-> WMMA input
        // SOA_INT, AOS_INT <-> WMMA acc
        // A-major acc fmt <-> WMMA acc
        // B-major acc fmt <-> WMMA acc
        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::SOA_INT,
                                              Format::WMMA_INPUT_GFX11,
                                              LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::to_wmma_input_gfx11(forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::AOS_INT,
                                              Format::WMMA_INPUT_GFX11,
                                              LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                using xform0 = register_layout_transform_impl<Format::AOS_INT,
                                                              Format::SOA_INT,
                                                              LayoutTraits>;
                using xform1 = register_layout_transform_impl<Format::SOA_INT,
                                                              Format::WMMA_INPUT_GFX11,
                                                              LayoutTraits>;
                return xform1::exec(xform0(forward<VecT>(v)));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::WMMA_INPUT_GFX11,
                                              Format::SOA_INT,
                                              LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::from_wmma_input_gfx11(forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::WMMA_INPUT_GFX11,
                                              Format::AOS_INT,
                                              LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                using xform0 = register_layout_transform_impl<Format::WMMA_INPUT_GFX11,
                                                              Format::SOA_INT,
                                                              LayoutTraits>;
                using xform1 = register_layout_transform_impl<Format::SOA_INT,
                                                              Format::AOS_INT,
                                                              LayoutTraits>;
                return xform1::exec(xform0(forward<VecT>(v)));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::SOA_INT, Format::WMMA_ACC_GFX11, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                using xform0 = register_layout_transform_impl<Format::SOA_INT,
                                                              Format::ACC_INT_A_MAJOR,
                                                              LayoutTraits>;
                using xform1 = register_layout_transform_impl<Format::ACC_INT_A_MAJOR,
                                                              Format::WMMA_ACC_GFX11,
                                                              LayoutTraits>;
                return xform1::exec(xform0(forward<VecT>(v)));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::AOS_INT, Format::WMMA_ACC_GFX11, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                using xform0 = register_layout_transform_impl<Format::AOS_INT,
                                                              Format::ACC_INT_A_MAJOR,
                                                              LayoutTraits>;
                using xform1 = register_layout_transform_impl<Format::ACC_INT_A_MAJOR,
                                                              Format::WMMA_ACC_GFX11,
                                                              LayoutTraits>;
                return xform1::exec(xform0(forward<VecT>(v)));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::WMMA_ACC_GFX11, Format::SOA_INT, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                using xform0 = register_layout_transform_impl<Format::WMMA_ACC_GFX11,
                                                              Format::ACC_INT_A_MAJOR,
                                                              LayoutTraits>;
                using xform1 = register_layout_transform_impl<Format::ACC_INT_A_MAJOR,
                                                              Format::SOA_INT,
                                                              LayoutTraits>;
                return xform1::exec(xform0(forward<VecT>(v)));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::WMMA_ACC_GFX11, Format::AOS_INT, LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                using xform0 = register_layout_transform_impl<Format::WMMA_ACC_GFX11,
                                                              Format::ACC_INT_A_MAJOR,
                                                              LayoutTraits>;
                using xform1 = register_layout_transform_impl<Format::ACC_INT_A_MAJOR,
                                                              Format::AOS_INT,
                                                              LayoutTraits>;
                return xform1::exec(xform0(forward<VecT>(v)));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::ACC_INT_A_MAJOR,
                                              Format::WMMA_ACC_GFX11,
                                              LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::to_wmma_acc_gfx11(forward<VecT>(v));
            }
        };

        template <typename LayoutTraits>
        struct register_layout_transform_impl<Format::WMMA_ACC_GFX11,
                                              Format::ACC_INT_A_MAJOR,
                                              LayoutTraits>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                return Transforms::from_wmma_acc_gfx11(forward<VecT>(v));
            }
        };

        // Keeps things a bit more tidy. Quick access to register layout traits.
        using LayoutTraits_impl::register_layout_traits;
#define traits_lhs register_layout_traits<RegisterLayoutLhs>
#define traits_rhs register_layout_traits<RegisterLayoutRhs>

        // Note: If you arrive at an undefined register_transform error, it is likely
        // the layout transformation is not currently supported. Need to either implement
        // the transform or ensure your layout transform mapping is correct.

        // Interface to transform from one register layout to another.
        template <typename RegisterLayoutSrc, typename RegisterLayoutDst, uint32_t WaveCount, typename Enabler = void>
        struct register_layout_transform;

        // Passthrough transform (NOP):
        // - Layouts are the same
        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs, uint32_t WaveCount>
        struct register_layout_transform<
            RegisterLayoutLhs,
            RegisterLayoutRhs,
            WaveCount,
            enable_if_t<is_layout_same_v<RegisterLayoutLhs, RegisterLayoutRhs>>>
        {
            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                // NOP
                return v;
            }
        };

        // Unsupported transform:
        // - Invalid RegisterLayouts
        // - Non-orthogonal (no transform path)
        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs, uint32_t WaveCount>
        struct register_layout_transform<
            RegisterLayoutLhs,
            RegisterLayoutRhs,
            WaveCount,
            enable_if_t<!is_layout_same_v<RegisterLayoutLhs, RegisterLayoutRhs>
                        && (!traits_lhs::is_register_layout || !traits_rhs::is_register_layout
                            || !is_layout_orthogonal_v<RegisterLayoutLhs, RegisterLayoutRhs>)>>
        {
            template <typename VecT>
            ROCWMMA_UNSUPPORTED_IMPL("Register layout transform is not supported")
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                // TODO: remove impl to catch unsupported transforms
                return forward<VecT>(v);
            }
        };

        // Valid transform:
        // - RegisterLayouts are valid
        // - Orthogonal (transform path exists)
        template <typename RegisterLayoutLhs, typename RegisterLayoutRhs, uint32_t WaveCount>
        struct register_layout_transform<
            RegisterLayoutLhs,
            RegisterLayoutRhs,
            WaveCount,
            enable_if_t<(traits_lhs::is_register_layout && traits_rhs::is_register_layout)
                        && is_layout_orthogonal_v<RegisterLayoutLhs, RegisterLayoutRhs>>>
        {

            template <typename VecT>
            ROCWMMA_DEVICE constexpr static inline decltype(auto) exec(VecT&& v)
            {
                // Extract traits for functional implementation
                using RegisterLayout::Format;
                using storage_traits
                    = conditional_t<traits_lhs::is_storage, traits_lhs, traits_rhs>;
                using transform_impl = register_layout_transform_impl<traits_lhs::Format,
                                                                      traits_rhs::Format,
                                                                      storage_traits>;

                if constexpr( WaveCount > 1 && storage_traits::is_interleaved)
                {
                    // Functionally, cooperative fragments have a reduced register footprint
                    auto res = decay_t<VecT>{};
                    using VecTraits = VecTraits<decay_t<VecT>>;
                    using VecFwdT = typename VecTraits::template VecT<typename VecTraits::DataT, VecTraits::size() / WaveCount>;

                    (VecFwdT&)res = transform_impl::exec((VecFwdT&)v);
                    // Forward to functional implementation
                    return res;
                }
                else
                {
                    return transform_impl::exec(forward<VecT>(v));
                }
            }
        };

#undef traits_lhs
#undef traits_rhs

    } // namespace LayoutTransforms_impl

} // namespace rocWMMA

#endif // ROCWMMA_REGISTER_LAYOUT_TRANSFORMS_HPP
