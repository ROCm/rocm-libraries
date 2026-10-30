/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
#ifndef ROCWMMA_TEST_REGRESSION_WMMA_CHARACTERIZATION_HPP
#define ROCWMMA_TEST_REGRESSION_WMMA_CHARACTERIZATION_HPP

// Characterization of the WMMA backend (detail::amdgcn_wmma, MmaTraits and
// MmaSelector) as a set of compile-time expectations.
//
// The expectations below were generated once from the implementation as it
// behaved before the wmma_impl.hpp restructuring, and are frozen: they are
// never regenerated from new code. A deliberate behaviour change must edit the
// affected lines by hand and be reviewed as such.
//
// All expectations are literal data. The helpers at the top only compare an
// implementation query against the recorded answer; they never derive an
// expectation from the implementation. A failing static_assert names the
// helper and the (target, types, M, N, K) in its instantiation note, and the
// "requested here" note points at the offending line of data.
//
// The expectations hold only in the host pass (see "Context"). Include this
// header from host-only code.

#if defined(__HIP_DEVICE_COMPILE__) && __HIP_DEVICE_COMPILE__
#error "wmma_characterization.hpp holds host-pass expectations only"
#endif

#include <rocwmma/rocwmma.hpp>

#include "wmma_target_backend.hpp"

namespace rocwmma
{
    namespace wmma_characterization
    {
        // Lists of K values. k_set is a membership set, k_seq an ordered sequence,
        // target_list a list of target ids.
        template <uint32_t... Ks>
        struct k_list
        {
        };

        template <uint32_t... Ks>
        struct k_set
        {
            template <uint32_t K>
            static constexpr bool contains = ((K == Ks) || ...);
        };

        template <uint32_t... Ks>
        struct k_seq
        {
        };

        template <uint32_t... Targets>
        struct target_list
        {
        };

        // K values probed for support, and BlockKTest values probed for selection.
        using probe_ks
            = k_list<1u, 2u, 3u, 4u, 6u, 8u, 12u, 16u, 24u, 32u, 48u, 64u, 96u, 128u, 192u, 256u>;
        using selector_block_k_tests = k_list<1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u>;

        template <typename A,
                  typename B,
                  typename C,
                  uint32_t M,
                  uint32_t N,
                  uint32_t K,
                  uint32_t Target>
        using op_t = detail::amdgcn_wmma<A, B, C, M, N, K, Target>;

        // ---- Section 2 helpers ------------------------------------------------
        template <uint32_t Target,
                  typename A,
                  typename B,
                  typename C,
                  uint32_t M,
                  uint32_t N,
                  uint32_t K,
                  typename Supported>
        constexpr void check_supported_k()
        {
            static_assert(MmaTraits<op_t<A, B, C, M, N, K, Target>>::is_supported
                              == Supported::template contains<K>,
                          "is_supported differs from the recorded supported-K set");
        }

        template <uint32_t Target,
                  typename A,
                  typename B,
                  typename C,
                  uint32_t M,
                  uint32_t N,
                  typename Supported,
                  uint32_t... Ks>
        constexpr void check_supported_ks(k_list<Ks...>)
        {
            (check_supported_k<Target, A, B, C, M, N, Ks, Supported>(), ...);
        }

        template <uint32_t Target,
                  typename A,
                  typename B,
                  typename C,
                  uint32_t M,
                  uint32_t N,
                  typename Supported>
        constexpr void expect_supported_k()
        {
            check_supported_ks<Target, A, B, C, M, N, Supported>(probe_ks{});
        }

        // ---- Section 3 helpers ------------------------------------------------
        template <uint32_t Target,
                  typename A,
                  typename B,
                  typename C,
                  uint32_t M,
                  uint32_t N,
                  uint32_t K,
                  typename ARegs,
                  typename BRegs,
                  typename CRegs,
                  typename DRegs>
        constexpr void expect_metadata()
        {
            using Op     = op_t<A, B, C, M, N, K, Target>;
            using Traits = MmaTraits<Op>;
            static_assert(Traits::is_supported, "recorded as supported");
            static_assert(Traits::is_wmma, "is_wmma");
            static_assert(!Traits::is_mfma, "is_mfma");
            static_assert(Traits::BlockM == M, "BlockM");
            static_assert(Traits::BlockN == N, "BlockN");
            static_assert(Traits::BlockK == K, "BlockK");
            static_assert(is_same_v<typename Traits::ARegsT, ARegs>, "MmaTraits::ARegsT");
            static_assert(is_same_v<typename Traits::BRegsT, BRegs>, "MmaTraits::BRegsT");
            static_assert(is_same_v<typename Traits::CRegsT, CRegs>, "MmaTraits::CRegsT");
            static_assert(is_same_v<typename Traits::DRegsT, DRegs>, "MmaTraits::DRegsT");
            static_assert(is_same_v<typename Op::ARegsT, ARegs>, "Op::ARegsT");
            static_assert(is_same_v<typename Op::BRegsT, BRegs>, "Op::BRegsT");
            static_assert(is_same_v<typename Op::CRegsT, CRegs>, "Op::CRegsT");
            static_assert(is_same_v<typename Op::DRegsT, DRegs>, "Op::DRegsT");
        }

        // ---- Section 4 helpers ------------------------------------------------
        template <uint32_t Target,
                  typename A,
                  typename B,
                  typename C,
                  uint32_t M,
                  uint32_t N,
                  uint32_t K,
                  typename ARegs,
                  typename BRegs,
                  typename CRegs,
                  typename DRegs>
        constexpr void check_unsupported_regs()
        {
            using Op     = op_t<A, B, C, M, N, K, Target>;
            using Traits = MmaTraits<Op>;

            // Only targets for which this key is unsupported fall back to the primary
            // template; supported targets are covered by section 3.
            if constexpr(!Traits::is_supported)
            {
                static_assert(is_same_v<typename Op::Unsupported, detail::Unsupported>,
                              "Unsupported tag");
                static_assert(is_same_v<typename Traits::ARegsT, ARegs>, "MmaTraits::ARegsT");
                static_assert(is_same_v<typename Traits::BRegsT, BRegs>, "MmaTraits::BRegsT");
                static_assert(is_same_v<typename Traits::CRegsT, CRegs>, "MmaTraits::CRegsT");
                static_assert(is_same_v<typename Traits::DRegsT, DRegs>, "MmaTraits::DRegsT");
            }
        }

        template <typename A,
                  typename B,
                  typename C,
                  uint32_t M,
                  uint32_t N,
                  uint32_t K,
                  typename ARegs,
                  typename BRegs,
                  typename CRegs,
                  typename DRegs,
                  uint32_t... Targets>
        constexpr void check_unsupported_regs_all(target_list<Targets...>)
        {
            (check_unsupported_regs<Targets, A, B, C, M, N, K, ARegs, BRegs, CRegs, DRegs>(), ...);
        }

        template <typename Targets,
                  typename A,
                  typename B,
                  typename C,
                  uint32_t M,
                  uint32_t N,
                  uint32_t K,
                  typename ARegs,
                  typename BRegs,
                  typename CRegs,
                  typename DRegs>
        constexpr void expect_unsupported_regs()
        {
            check_unsupported_regs_all<A, B, C, M, N, K, ARegs, BRegs, CRegs, DRegs>(Targets{});
        }

        // ---- Section 5 helpers ------------------------------------------------
        template <uint32_t Target,
                  typename A,
                  typename B,
                  typename C,
                  uint32_t M,
                  uint32_t N,
                  uint32_t BlockKTest,
                  uint32_t SelectedK>
        constexpr void check_selected_k()
        {
            static_assert(is_same_v<SelectedWmmaFor<Target, A, B, C, M, N, BlockKTest>,
                                    op_t<A, B, C, M, N, SelectedK, Target>>,
                          "selected op differs from the recorded one for this BlockKTest");
        }

        template <uint32_t Target,
                  typename A,
                  typename B,
                  typename C,
                  uint32_t M,
                  uint32_t N,
                  uint32_t... BlockKTests,
                  uint32_t... SelectedKs>
        constexpr void check_selected_ks(k_list<BlockKTests...>, k_seq<SelectedKs...>)
        {
            static_assert(sizeof...(BlockKTests) == sizeof...(SelectedKs),
                          "one selected K per BlockKTest");
            (check_selected_k<Target, A, B, C, M, N, BlockKTests, SelectedKs>(), ...);
        }

        template <uint32_t Target,
                  typename A,
                  typename B,
                  typename C,
                  uint32_t M,
                  uint32_t N,
                  typename SelectedKs>
        constexpr void expect_selected_k()
        {
            check_selected_ks<Target, A, B, C, M, N>(selector_block_k_tests{}, SelectedKs{});
        }

        // ---- Section 6 helpers ------------------------------------------------
        template <uint32_t Target,
                  typename A,
                  typename B,
                  typename C,
                  uint32_t M,
                  uint32_t N,
                  uint32_t K,
                  typename ARegs,
                  typename BRegs,
                  typename CRegs,
                  typename DRegs>
        constexpr void expect_enabler()
        {
            using Defaulted = detail::amdgcn_wmma<A, B, C, M, N, K, Target>;
            using WithVoid  = detail::amdgcn_wmma<A, B, C, M, N, K, Target, void>;
            using WithInt   = detail::amdgcn_wmma<A, B, C, M, N, K, Target, int>;
            using IntTraits = MmaTraits<WithInt>;

            static_assert(is_same_v<Defaulted, WithVoid>, "Enabler defaults to void");
            static_assert(MmaTraits<WithVoid>::is_supported, "Enabler = void is supported");
            static_assert(!IntTraits::is_supported, "Enabler = int is unsupported");
            static_assert(is_same_v<typename WithInt::Unsupported, detail::Unsupported>,
                          "Enabler = int has the Unsupported tag");
            static_assert(is_same_v<typename IntTraits::ARegsT, ARegs>, "ARegsT");
            static_assert(is_same_v<typename IntTraits::BRegsT, BRegs>, "BRegsT");
            static_assert(is_same_v<typename IntTraits::CRegsT, CRegs>, "CRegsT");
            static_assert(is_same_v<typename IntTraits::DRegsT, DRegs>, "DRegsT");
        }

        // =======================================================================
        // Generated expectations
        // =======================================================================

        // ---------------------------------------------------------------------------
        // 1. Context
        //
        // The compilation context the expectations were recorded in. The unsupported
        // register types (section 4) depend on the current wave size, not on the
        // explicit target, so they only hold in this context.
        // ---------------------------------------------------------------------------
        static_assert(ROCWMMA_DEVICE_COMPILE == 0, "expectations are recorded in the host pass");
        static_assert(ROCWMMA_ARCH_HOST == 1, "host pass");
        static_assert(ROCWMMA_ARCH_GFX9 == 0, "no device arch family on the host");
        static_assert(ROCWMMA_ARCH_GFX11 == 0, "no device arch family on the host");
        static_assert(ROCWMMA_ARCH_GFX12 == 0, "no device arch family on the host");
        static_assert(ROCWMMA_WAVE64_MODE == 0, "wave mode macros are unset on the host");
        static_assert(ROCWMMA_WAVE32_MODE == 0, "wave mode macros are unset on the host");
        static_assert(ROCWMMA_BLOCK_DIM_16_SUPPORTED == 1, "host supports block dim 16");
        static_assert(ROCWMMA_BLOCK_DIM_32_SUPPORTED == 1, "host supports block dim 32");
        static_assert(Constants::AMDGCN_WAVE_SIZE == 64, "host defaults to wave size 64");
        static_assert(Constants::AMDGCN_CURRENT_ARCH_ID == Constants::AMDGCN_ARCH_ID_NONE,
                      "no current target on the host");

        // FP8 and half-precision macros that decide what float8_t, bfloat8_t and hfloat16_t
        // are on the host. float8_t and bfloat8_t are always the OCP types (aliases of
        // forward-declared HIP structs); ROCWMMA_FP8 additionally gates the VecT
        // registrations for them (vector.hpp), and ROCWMMA_NO_HALF removes hfloat16_t
        // (types.hpp). HIP_NO_HALF is not buildable today (see README), so
        // ROCWMMA_NO_HALF == 0 is the only recorded configuration.
        static_assert(ROCWMMA_FP8 == 1, "OCP fp8 is visible on the host");
        static_assert(ROCWMMA_NO_HALF == 0, "hfloat16_t exists");
        static_assert(is_same_v<float8_t, hip_fp8_e4m3> && is_same_v<bfloat8_t, hip_fp8_e5m2>,
                      "float8_t / bfloat8_t are the OCP fp8 types");
        static_assert(is_same_v<hfloat16_t, __half>, "hfloat16_t is __half");

        // Builtin availability: in the host pass __has_builtin reports none of the
        // __builtin_amdgcn_wmma_* builtins (checked with ROCm 7.2), so no expectation
        // below depends on builtin availability. Device-pass context is covered in
        // wmma_characterization.cpp.

        // ---------------------------------------------------------------------------
        // 2. Supported-K sets
        //
        // Per (target, A/B/C types, M, N): exactly which K of the probe set
        // {1,2,3,4,6,8,12,16,24,32,48,64,96,128,192,256} satisfy
        // MmaTraits<amdgcn_wmma<A, B, C, M, N, K, target>>::is_supported.
        // ---------------------------------------------------------------------------
        // clang-format off
        inline void supported_k_sets()
        {
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float16_t, float16_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float16_t, float16_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float16_t, float16_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float16_t, float16_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float16_t, float16_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float16_t, float16_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float16_t, float16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float16_t, float16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float16_t, float16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float16_t, float16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float32_t, float32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float32_t, float32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float32_t, float32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float32_t, float32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float32_t, float32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, int8_t, int8_t, int32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, int8_t, int8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, int8_t, int8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, int8_t, int8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, int8_t, int8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, uint8_t, uint8_t, int32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, uint8_t, uint8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, uint8_t, uint8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, uint8_t, uint8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, uint8_t, uint8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, uint8_t, uint8_t, uint32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, uint8_t, uint8_t, uint32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, uint8_t, uint8_t, uint32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, uint8_t, uint8_t, uint32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, uint8_t, uint8_t, uint32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float64_t, float64_t, float64_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float64_t, float64_t, float64_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float64_t, float64_t, float64_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float64_t, float64_t, float64_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, float64_t, float64_t, float64_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX908, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float16_t, float16_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float16_t, float16_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float16_t, float16_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float16_t, float16_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float16_t, float16_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float16_t, float16_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float16_t, float16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float16_t, float16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float16_t, float16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float16_t, float16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float32_t, float32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float32_t, float32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float32_t, float32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float32_t, float32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float32_t, float32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, int8_t, int8_t, int32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, int8_t, int8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, int8_t, int8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, int8_t, int8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, int8_t, int8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, uint8_t, uint8_t, int32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, uint8_t, uint8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, uint8_t, uint8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, uint8_t, uint8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, uint8_t, uint8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, uint8_t, uint8_t, uint32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, uint8_t, uint8_t, uint32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, uint8_t, uint8_t, uint32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, uint8_t, uint8_t, uint32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, uint8_t, uint8_t, uint32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float64_t, float64_t, float64_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float64_t, float64_t, float64_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float64_t, float64_t, float64_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float64_t, float64_t, float64_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, float64_t, float64_t, float64_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX90A, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float16_t, float16_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float16_t, float16_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float16_t, float16_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float16_t, float16_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float16_t, float16_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float16_t, float16_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float16_t, float16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float16_t, float16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float16_t, float16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float16_t, float16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float32_t, float32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float32_t, float32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float32_t, float32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float32_t, float32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float32_t, float32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, int8_t, int8_t, int32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, int8_t, int8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, int8_t, int8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, int8_t, int8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, int8_t, int8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, uint8_t, uint8_t, int32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, uint8_t, uint8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, uint8_t, uint8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, uint8_t, uint8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, uint8_t, uint8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, uint8_t, uint8_t, uint32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, uint8_t, uint8_t, uint32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, uint8_t, uint8_t, uint32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, uint8_t, uint8_t, uint32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, uint8_t, uint8_t, uint32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float64_t, float64_t, float64_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float64_t, float64_t, float64_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float64_t, float64_t, float64_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float64_t, float64_t, float64_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, float64_t, float64_t, float64_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX942, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float16_t, float16_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float16_t, float16_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float16_t, float16_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float16_t, float16_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float16_t, float16_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float16_t, float16_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float16_t, float16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float16_t, float16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float16_t, float16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float16_t, float16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float32_t, float32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float32_t, float32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float32_t, float32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float32_t, float32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float32_t, float32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, int8_t, int8_t, int32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, int8_t, int8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, int8_t, int8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, int8_t, int8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, int8_t, int8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, uint8_t, uint8_t, int32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, uint8_t, uint8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, uint8_t, uint8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, uint8_t, uint8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, uint8_t, uint8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, uint8_t, uint8_t, uint32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, uint8_t, uint8_t, uint32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, uint8_t, uint8_t, uint32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, uint8_t, uint8_t, uint32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, uint8_t, uint8_t, uint32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float64_t, float64_t, float64_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float64_t, float64_t, float64_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float64_t, float64_t, float64_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float64_t, float64_t, float64_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, float64_t, float64_t, float64_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX950, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float32_t, float32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float32_t, float32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float32_t, float32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float32_t, float32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float32_t, float32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, int8_t, int8_t, int32_t, 16, 16, k_set<8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, int8_t, int8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, int8_t, int8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, int8_t, int8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, int8_t, int8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, uint8_t, uint8_t, int32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, uint8_t, uint8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, uint8_t, uint8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, uint8_t, uint8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, uint8_t, uint8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, uint8_t, uint8_t, uint32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, uint8_t, uint8_t, uint32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, uint8_t, uint8_t, uint32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, uint8_t, uint8_t, uint32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, uint8_t, uint8_t, uint32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float64_t, float64_t, float64_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float64_t, float64_t, float64_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float64_t, float64_t, float64_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float64_t, float64_t, float64_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, float64_t, float64_t, float64_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1100, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float32_t, float32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float32_t, float32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float32_t, float32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float32_t, float32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float32_t, float32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, int8_t, int8_t, int32_t, 16, 16, k_set<8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, int8_t, int8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, int8_t, int8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, int8_t, int8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, int8_t, int8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, uint8_t, uint8_t, int32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, uint8_t, uint8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, uint8_t, uint8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, uint8_t, uint8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, uint8_t, uint8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, uint8_t, uint8_t, uint32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, uint8_t, uint8_t, uint32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, uint8_t, uint8_t, uint32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, uint8_t, uint8_t, uint32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, uint8_t, uint8_t, uint32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float64_t, float64_t, float64_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float64_t, float64_t, float64_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float64_t, float64_t, float64_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float64_t, float64_t, float64_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, float64_t, float64_t, float64_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1101, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float32_t, float32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float32_t, float32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float32_t, float32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float32_t, float32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float32_t, float32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, int8_t, int8_t, int32_t, 16, 16, k_set<8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, int8_t, int8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, int8_t, int8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, int8_t, int8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, int8_t, int8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, uint8_t, uint8_t, int32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, uint8_t, uint8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, uint8_t, uint8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, uint8_t, uint8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, uint8_t, uint8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, uint8_t, uint8_t, uint32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, uint8_t, uint8_t, uint32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, uint8_t, uint8_t, uint32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, uint8_t, uint8_t, uint32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, uint8_t, uint8_t, uint32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float64_t, float64_t, float64_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float64_t, float64_t, float64_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float64_t, float64_t, float64_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float64_t, float64_t, float64_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, float64_t, float64_t, float64_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1102, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float32_t, float32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float32_t, float32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float32_t, float32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float32_t, float32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float32_t, float32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, int8_t, int8_t, int32_t, 16, 16, k_set<8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, int8_t, int8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, int8_t, int8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, int8_t, int8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, int8_t, int8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, uint8_t, uint8_t, int32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, uint8_t, uint8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, uint8_t, uint8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, uint8_t, uint8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, uint8_t, uint8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, uint8_t, uint8_t, uint32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, uint8_t, uint8_t, uint32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, uint8_t, uint8_t, uint32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, uint8_t, uint8_t, uint32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, uint8_t, uint8_t, uint32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float64_t, float64_t, float64_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float64_t, float64_t, float64_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float64_t, float64_t, float64_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float64_t, float64_t, float64_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, float64_t, float64_t, float64_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1103, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float32_t, float32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float32_t, float32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float32_t, float32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float32_t, float32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float32_t, float32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, int8_t, int8_t, int32_t, 16, 16, k_set<8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, int8_t, int8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, int8_t, int8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, int8_t, int8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, int8_t, int8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, uint8_t, uint8_t, int32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, uint8_t, uint8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, uint8_t, uint8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, uint8_t, uint8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, uint8_t, uint8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, uint8_t, uint8_t, uint32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, uint8_t, uint8_t, uint32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, uint8_t, uint8_t, uint32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, uint8_t, uint8_t, uint32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, uint8_t, uint8_t, uint32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float64_t, float64_t, float64_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float64_t, float64_t, float64_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float64_t, float64_t, float64_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float64_t, float64_t, float64_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, float64_t, float64_t, float64_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1150, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float32_t, float32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float32_t, float32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float32_t, float32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float32_t, float32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float32_t, float32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, int8_t, int8_t, int32_t, 16, 16, k_set<8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, int8_t, int8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, int8_t, int8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, int8_t, int8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, int8_t, int8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, uint8_t, uint8_t, int32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, uint8_t, uint8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, uint8_t, uint8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, uint8_t, uint8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, uint8_t, uint8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, uint8_t, uint8_t, uint32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, uint8_t, uint8_t, uint32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, uint8_t, uint8_t, uint32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, uint8_t, uint8_t, uint32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, uint8_t, uint8_t, uint32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float64_t, float64_t, float64_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float64_t, float64_t, float64_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float64_t, float64_t, float64_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float64_t, float64_t, float64_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, float64_t, float64_t, float64_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1151, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float32_t, float32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float32_t, float32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float32_t, float32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float32_t, float32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float32_t, float32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, int8_t, int8_t, int32_t, 16, 16, k_set<8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, int8_t, int8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, int8_t, int8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, int8_t, int8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, int8_t, int8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, uint8_t, uint8_t, int32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, uint8_t, uint8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, uint8_t, uint8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, uint8_t, uint8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, uint8_t, uint8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, uint8_t, uint8_t, uint32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, uint8_t, uint8_t, uint32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, uint8_t, uint8_t, uint32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, uint8_t, uint8_t, uint32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, uint8_t, uint8_t, uint32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float64_t, float64_t, float64_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float64_t, float64_t, float64_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float64_t, float64_t, float64_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float64_t, float64_t, float64_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, float64_t, float64_t, float64_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1152, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float32_t, float32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float32_t, float32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float32_t, float32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float32_t, float32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float32_t, float32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, int8_t, int8_t, int32_t, 16, 16, k_set<8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, int8_t, int8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, int8_t, int8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, int8_t, int8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, int8_t, int8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, uint8_t, uint8_t, int32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, uint8_t, uint8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, uint8_t, uint8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, uint8_t, uint8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, uint8_t, uint8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, uint8_t, uint8_t, uint32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, uint8_t, uint8_t, uint32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, uint8_t, uint8_t, uint32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, uint8_t, uint8_t, uint32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, uint8_t, uint8_t, uint32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float64_t, float64_t, float64_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float64_t, float64_t, float64_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float64_t, float64_t, float64_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float64_t, float64_t, float64_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, float64_t, float64_t, float64_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1153, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_set<8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, float8_t, float32_t, 16, 16, k_set<16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float32_t, float32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float32_t, float32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float32_t, float32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float32_t, float32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float32_t, float32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, bfloat8_t, float32_t, 16, 16, k_set<16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, float8_t, float32_t, 16, 16, k_set<8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, int8_t, int8_t, int32_t, 16, 16, k_set<8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, int8_t, int8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, int8_t, int8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, int8_t, int8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, int8_t, int8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, uint8_t, uint8_t, int32_t, 16, 16, k_set<8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, uint8_t, uint8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, uint8_t, uint8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, uint8_t, uint8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, uint8_t, uint8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, uint8_t, uint8_t, uint32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, uint8_t, uint8_t, uint32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, uint8_t, uint8_t, uint32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, uint8_t, uint8_t, uint32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, uint8_t, uint8_t, uint32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float64_t, float64_t, float64_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float64_t, float64_t, float64_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float64_t, float64_t, float64_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float64_t, float64_t, float64_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, float64_t, float64_t, float64_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1170, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_set<8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, float8_t, float32_t, 16, 16, k_set<16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float32_t, float32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float32_t, float32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float32_t, float32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float32_t, float32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float32_t, float32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, bfloat8_t, float32_t, 16, 16, k_set<16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, float8_t, float32_t, 16, 16, k_set<8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, int8_t, int8_t, int32_t, 16, 16, k_set<8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, int8_t, int8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, int8_t, int8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, int8_t, int8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, int8_t, int8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, uint8_t, uint8_t, int32_t, 16, 16, k_set<8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, uint8_t, uint8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, uint8_t, uint8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, uint8_t, uint8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, uint8_t, uint8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, uint8_t, uint8_t, uint32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, uint8_t, uint8_t, uint32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, uint8_t, uint8_t, uint32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, uint8_t, uint8_t, uint32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, uint8_t, uint8_t, uint32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float64_t, float64_t, float64_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float64_t, float64_t, float64_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float64_t, float64_t, float64_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float64_t, float64_t, float64_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, float64_t, float64_t, float64_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1171, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_set<8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, float8_t, float32_t, 16, 16, k_set<16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float32_t, float32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float32_t, float32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float32_t, float32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float32_t, float32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float32_t, float32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, bfloat8_t, float32_t, 16, 16, k_set<16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, float8_t, float32_t, 16, 16, k_set<8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, int8_t, int8_t, int32_t, 16, 16, k_set<8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, int8_t, int8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, int8_t, int8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, int8_t, int8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, int8_t, int8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, uint8_t, uint8_t, int32_t, 16, 16, k_set<8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, uint8_t, uint8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, uint8_t, uint8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, uint8_t, uint8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, uint8_t, uint8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, uint8_t, uint8_t, uint32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, uint8_t, uint8_t, uint32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, uint8_t, uint8_t, uint32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, uint8_t, uint8_t, uint32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, uint8_t, uint8_t, uint32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float64_t, float64_t, float64_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float64_t, float64_t, float64_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float64_t, float64_t, float64_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float64_t, float64_t, float64_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, float64_t, float64_t, float64_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1200, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_set<8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, float8_t, float32_t, 16, 16, k_set<16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float32_t, float32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float32_t, float32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float32_t, float32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float32_t, float32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float32_t, float32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, bfloat8_t, float32_t, 16, 16, k_set<16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, float8_t, float32_t, 16, 16, k_set<8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_set<4, 8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, int8_t, int8_t, int32_t, 16, 16, k_set<8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, int8_t, int8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, int8_t, int8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, int8_t, int8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, int8_t, int8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, uint8_t, uint8_t, int32_t, 16, 16, k_set<8, 16>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, uint8_t, uint8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, uint8_t, uint8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, uint8_t, uint8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, uint8_t, uint8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, uint8_t, uint8_t, uint32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, uint8_t, uint8_t, uint32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, uint8_t, uint8_t, uint32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, uint8_t, uint8_t, uint32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, uint8_t, uint8_t, uint32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float64_t, float64_t, float64_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float64_t, float64_t, float64_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float64_t, float64_t, float64_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float64_t, float64_t, float64_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, float64_t, float64_t, float64_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1201, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_set<4, 8, 16, 32>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_set<4, 8, 16, 32>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_set<16, 32, 64, 128>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_set<8, 16, 32, 64, 128>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float16_t, 16, 16, k_set<16, 32, 64, 128>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float32_t, 16, 16, k_set<16, 32, 64, 128>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float16_t, 16, 16, k_set<4, 8, 16, 32>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float32_t, 16, 16, k_set<4, 8, 16, 32>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float32_t, float32_t, float32_t, 16, 16, k_set<2, 4>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float32_t, float32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float32_t, float32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float32_t, float32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float32_t, float32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float16_t, 16, 16, k_set<16, 32, 64, 128>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float32_t, 16, 16, k_set<16, 32, 64, 128>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float16_t, 16, 16, k_set<16, 32, 64, 128>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float32_t, 16, 16, k_set<8, 16, 32, 64, 128>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_set<4, 8, 16, 32>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_set<4, 8, 16, 32>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, int8_t, int8_t, int32_t, 16, 16, k_set<8, 16, 32, 64>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, int8_t, int8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, int8_t, int8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, int8_t, int8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, int8_t, int8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, uint8_t, uint8_t, int32_t, 16, 16, k_set<8, 16, 32, 64>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, uint8_t, uint8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, uint8_t, uint8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, uint8_t, uint8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, uint8_t, uint8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, uint8_t, uint8_t, uint32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, uint8_t, uint8_t, uint32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, uint8_t, uint8_t, uint32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, uint8_t, uint8_t, uint32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, uint8_t, uint8_t, uint32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float64_t, float64_t, float64_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float64_t, float64_t, float64_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float64_t, float64_t, float64_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float64_t, float64_t, float64_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, float64_t, float64_t, float64_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_GFX1250, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float16_t, float16_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float16_t, float16_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float16_t, float16_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float16_t, float16_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float16_t, float16_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float16_t, float16_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float16_t, float16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float16_t, float16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float16_t, float16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float16_t, float16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float32_t, float32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float32_t, float32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float32_t, float32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float32_t, float32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float32_t, float32_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, bfloat8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, bfloat8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, bfloat8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, bfloat8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, bfloat8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, bfloat8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, bfloat8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, bfloat8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, bfloat8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, bfloat8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, float8_t, float16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, float8_t, float16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, float8_t, float16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, float8_t, float16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, float8_t, float16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, float8_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, float8_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, float8_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, float8_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, float8_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, int8_t, int8_t, int32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, int8_t, int8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, int8_t, int8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, int8_t, int8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, int8_t, int8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, uint8_t, uint8_t, int32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, uint8_t, uint8_t, int32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, uint8_t, uint8_t, int32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, uint8_t, uint8_t, int32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, uint8_t, uint8_t, int32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, uint8_t, uint8_t, uint32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, uint8_t, uint8_t, uint32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, uint8_t, uint8_t, uint32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, uint8_t, uint8_t, uint32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, uint8_t, uint8_t, uint32_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float64_t, float64_t, float64_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float64_t, float64_t, float64_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float64_t, float64_t, float64_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float64_t, float64_t, float64_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, float64_t, float64_t, float64_t, 16, 8, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_set<>>();
            expect_supported_k<Constants::AMDGCN_ARCH_ID_NONE, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_set<>>();
        }
        // clang-format on

        // ---------------------------------------------------------------------------
        // 3. Supported metadata
        //
        // Per supported (target, types, M, N, K): the exact register types (as
        // canonical VecT<elem, N> expressions, compared with is_same_v), and
        // BlockK == K, BlockM == M, BlockN == N, is_wmma, !is_mfma.
        // ---------------------------------------------------------------------------
        // clang-format off
        inline void supported_metadata()
        {
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float16_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float16_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float16_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1100, int8_t, int8_t, int32_t, 16, 16, 8,
                            VecT<int32_t, 2>, VecT<int32_t, 2>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1100, int8_t, int8_t, int32_t, 16, 16, 16,
                            VecT<int32_t, 4>, VecT<int32_t, 4>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float16_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float16_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float16_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1101, int8_t, int8_t, int32_t, 16, 16, 8,
                            VecT<int32_t, 2>, VecT<int32_t, 2>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1101, int8_t, int8_t, int32_t, 16, 16, 16,
                            VecT<int32_t, 4>, VecT<int32_t, 4>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float16_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float16_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float16_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1102, int8_t, int8_t, int32_t, 16, 16, 8,
                            VecT<int32_t, 2>, VecT<int32_t, 2>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1102, int8_t, int8_t, int32_t, 16, 16, 16,
                            VecT<int32_t, 4>, VecT<int32_t, 4>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float16_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float16_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float16_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1103, int8_t, int8_t, int32_t, 16, 16, 8,
                            VecT<int32_t, 2>, VecT<int32_t, 2>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1103, int8_t, int8_t, int32_t, 16, 16, 16,
                            VecT<int32_t, 4>, VecT<int32_t, 4>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float16_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float16_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float16_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1150, int8_t, int8_t, int32_t, 16, 16, 8,
                            VecT<int32_t, 2>, VecT<int32_t, 2>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1150, int8_t, int8_t, int32_t, 16, 16, 16,
                            VecT<int32_t, 4>, VecT<int32_t, 4>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float16_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float16_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float16_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1151, int8_t, int8_t, int32_t, 16, 16, 8,
                            VecT<int32_t, 2>, VecT<int32_t, 2>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1151, int8_t, int8_t, int32_t, 16, 16, 16,
                            VecT<int32_t, 4>, VecT<int32_t, 4>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float16_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float16_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float16_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1152, int8_t, int8_t, int32_t, 16, 16, 8,
                            VecT<int32_t, 2>, VecT<int32_t, 2>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1152, int8_t, int8_t, int32_t, 16, 16, 16,
                            VecT<int32_t, 4>, VecT<int32_t, 4>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float16_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float16_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float16_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 8,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 16,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1153, int8_t, int8_t, int32_t, 16, 16, 8,
                            VecT<int32_t, 2>, VecT<int32_t, 2>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1153, int8_t, int8_t, int32_t, 16, 16, 16,
                            VecT<int32_t, 4>, VecT<int32_t, 4>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, bfloat8_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, bfloat8_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, float8_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float16_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float16_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float16_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, bfloat8_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, float8_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, float8_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, int8_t, int8_t, int32_t, 16, 16, 8,
                            VecT<int32_t, 1>, VecT<int32_t, 1>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, int8_t, int8_t, int32_t, 16, 16, 16,
                            VecT<int32_t, 2>, VecT<int32_t, 2>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, uint8_t, uint8_t, int32_t, 16, 16, 8,
                            VecT<uint32_t, 1>, VecT<uint32_t, 1>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1170, uint8_t, uint8_t, int32_t, 16, 16, 16,
                            VecT<uint32_t, 2>, VecT<uint32_t, 2>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, bfloat8_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, bfloat8_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, float8_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float16_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float16_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float16_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, bfloat8_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, float8_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, float8_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, int8_t, int8_t, int32_t, 16, 16, 8,
                            VecT<int32_t, 1>, VecT<int32_t, 1>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, int8_t, int8_t, int32_t, 16, 16, 16,
                            VecT<int32_t, 2>, VecT<int32_t, 2>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, uint8_t, uint8_t, int32_t, 16, 16, 8,
                            VecT<uint32_t, 1>, VecT<uint32_t, 1>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1171, uint8_t, uint8_t, int32_t, 16, 16, 16,
                            VecT<uint32_t, 2>, VecT<uint32_t, 2>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, bfloat8_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, bfloat8_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, float8_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float16_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float16_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float16_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, bfloat8_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, float8_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, float8_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, int8_t, int8_t, int32_t, 16, 16, 8,
                            VecT<int32_t, 1>, VecT<int32_t, 1>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, int8_t, int8_t, int32_t, 16, 16, 16,
                            VecT<int32_t, 2>, VecT<int32_t, 2>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, uint8_t, uint8_t, int32_t, 16, 16, 8,
                            VecT<uint32_t, 1>, VecT<uint32_t, 1>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1200, uint8_t, uint8_t, int32_t, 16, 16, 16,
                            VecT<uint32_t, 2>, VecT<uint32_t, 2>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, bfloat8_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, bfloat8_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, float8_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float16_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float16_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float16_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, bfloat8_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, float8_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, float8_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, int8_t, int8_t, int32_t, 16, 16, 8,
                            VecT<int32_t, 1>, VecT<int32_t, 1>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, int8_t, int8_t, int32_t, 16, 16, 16,
                            VecT<int32_t, 2>, VecT<int32_t, 2>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, uint8_t, uint8_t, int32_t, 16, 16, 8,
                            VecT<uint32_t, 1>, VecT<uint32_t, 1>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1201, uint8_t, uint8_t, int32_t, 16, 16, 16,
                            VecT<uint32_t, 2>, VecT<uint32_t, 2>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 32,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, float32_t, 16, 16, 32,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float16_t, 16, 16, 16,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float16_t, 16, 16, 32,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float16_t, 16, 16, 64,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float16_t, 16, 16, 128,
                            VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float32_t, 16, 16, 32,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float32_t, 16, 16, 64,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float32_t, 16, 16, 128,
                            VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float16_t, 16, 16, 16,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float16_t, 16, 16, 32,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float16_t, 16, 16, 64,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float16_t, 16, 16, 128,
                            VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float32_t, 16, 16, 32,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float32_t, 16, 16, 64,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float32_t, 16, 16, 128,
                            VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float16_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float16_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float16_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float16_t, 16, 16, 32,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float32_t, 16, 16, 32,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float32_t, float32_t, float32_t, 16, 16, 2,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float32_t, float32_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float16_t, 16, 16, 16,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float16_t, 16, 16, 32,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float16_t, 16, 16, 64,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float16_t, 16, 16, 128,
                            VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float32_t, 16, 16, 32,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float32_t, 16, 16, 64,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float32_t, 16, 16, 128,
                            VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float16_t, 16, 16, 16,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float16_t, 16, 16, 32,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float16_t, 16, 16, 64,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float16_t, 16, 16, 128,
                            VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float32_t, 16, 16, 32,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float32_t, 16, 16, 64,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float32_t, 16, 16, 128,
                            VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, float32_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, float32_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, float32_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, float32_t, 16, 16, 32,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 4,
                            VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 8,
                            VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 16,
                            VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 32,
                            VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, int8_t, int8_t, int32_t, 16, 16, 8,
                            VecT<int32_t, 1>, VecT<int32_t, 1>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, int8_t, int8_t, int32_t, 16, 16, 16,
                            VecT<int32_t, 2>, VecT<int32_t, 2>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, int8_t, int8_t, int32_t, 16, 16, 32,
                            VecT<int32_t, 4>, VecT<int32_t, 4>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, int8_t, int8_t, int32_t, 16, 16, 64,
                            VecT<int32_t, 8>, VecT<int32_t, 8>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, uint8_t, uint8_t, int32_t, 16, 16, 8,
                            VecT<uint32_t, 1>, VecT<uint32_t, 1>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, uint8_t, uint8_t, int32_t, 16, 16, 16,
                            VecT<uint32_t, 2>, VecT<uint32_t, 2>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, uint8_t, uint8_t, int32_t, 16, 16, 32,
                            VecT<uint32_t, 4>, VecT<uint32_t, 4>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
            expect_metadata<Constants::AMDGCN_ARCH_ID_GFX1250, uint8_t, uint8_t, int32_t, 16, 16, 64,
                            VecT<uint32_t, 8>, VecT<uint32_t, 8>, VecT<int32_t, 8>, VecT<int32_t, 8>>();
        }
        // clang-format on

        // ---------------------------------------------------------------------------
        // 4. Unsupported register types
        //
        // The primary (unsupported) template sizes registers from the *current*
        // Constants::AMDGCN_WAVE_SIZE (64 on the host), not from the explicit target,
        // and sizes the accumulator as BlockM * BlockM (rectangular shapes expose it).
        // Per (types, M, N, K): the registers every target reports when the key is
        // unsupported for it; the check covers all targets below for which the key
        // is unsupported, and also asserts the Unsupported tag.
        // ---------------------------------------------------------------------------
        // clang-format off
        using all_targets = target_list<
                                        Constants::AMDGCN_ARCH_ID_GFX908,
                                        Constants::AMDGCN_ARCH_ID_GFX90A,
                                        Constants::AMDGCN_ARCH_ID_GFX942,
                                        Constants::AMDGCN_ARCH_ID_GFX950,
                                        Constants::AMDGCN_ARCH_ID_GFX1100,
                                        Constants::AMDGCN_ARCH_ID_GFX1101,
                                        Constants::AMDGCN_ARCH_ID_GFX1102,
                                        Constants::AMDGCN_ARCH_ID_GFX1103,
                                        Constants::AMDGCN_ARCH_ID_GFX1150,
                                        Constants::AMDGCN_ARCH_ID_GFX1151,
                                        Constants::AMDGCN_ARCH_ID_GFX1152,
                                        Constants::AMDGCN_ARCH_ID_GFX1153,
                                        Constants::AMDGCN_ARCH_ID_GFX1170,
                                        Constants::AMDGCN_ARCH_ID_GFX1171,
                                        Constants::AMDGCN_ARCH_ID_GFX1200,
                                        Constants::AMDGCN_ARCH_ID_GFX1201,
                                        Constants::AMDGCN_ARCH_ID_GFX1250,
                                        Constants::AMDGCN_ARCH_ID_NONE>;

        inline void unsupported_registers()
        {
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 24>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 32>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 3>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 6>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 12>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 24>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 32>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 48>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 64>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, 4,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, 6,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, 8,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, 12,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, 16,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, 24,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, 32,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, 48,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, 64,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, 96,
                                    VecT<float32_t, 24>, VecT<float32_t, 12>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, 128,
                                    VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, 192,
                                    VecT<float32_t, 48>, VecT<float32_t, 24>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, 256,
                                    VecT<float32_t, 64>, VecT<float32_t, 32>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, 4,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, 6,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, 8,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, 12,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, 16,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, 24,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, 32,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, 48,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, 64,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, 96,
                                    VecT<float32_t, 24>, VecT<float32_t, 24>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, 128,
                                    VecT<float32_t, 32>, VecT<float32_t, 32>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, 192,
                                    VecT<float32_t, 48>, VecT<float32_t, 48>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, 256,
                                    VecT<float32_t, 64>, VecT<float32_t, 64>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 12>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 16, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 16, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 16, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 16, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 16, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 16, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 16, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 16, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 16, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 16, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 16, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 16, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 24>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 16, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 32>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 32, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 32, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 32, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 32, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 32, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 32, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 32, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 32, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 12>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 32, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 32, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 24>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 32, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 32>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 32, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 48>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 32, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 64>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 16, 4,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 16, 6,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 16, 8,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 16, 12,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 16, 16,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 16, 24,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 16, 32,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 16, 48,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 16, 64,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 16, 96,
                                    VecT<float32_t, 24>, VecT<float32_t, 12>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 16, 128,
                                    VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 16, 192,
                                    VecT<float32_t, 48>, VecT<float32_t, 24>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 16, 256,
                                    VecT<float32_t, 64>, VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 32, 4,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 32, 6,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 32, 8,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 32, 12,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 32, 16,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 32, 24,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 32, 32,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 32, 48,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 32, 64,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 32, 96,
                                    VecT<float32_t, 24>, VecT<float32_t, 24>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 32, 128,
                                    VecT<float32_t, 32>, VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 32, 192,
                                    VecT<float32_t, 48>, VecT<float32_t, 48>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 32, 32, 256,
                                    VecT<float32_t, 64>, VecT<float32_t, 64>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 8, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 8, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 8, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 8, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 8, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 8, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 8, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 8, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 8, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 8, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 8, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 8, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 8, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 8, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 8, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 12>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat16_t, bfloat16_t, float32_t, 16, 8, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 16, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 16, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 16, 8,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 16, 12,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 16, 16,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 16, 24,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 16, 32,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 16, 48,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 16, 64,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 16, 96,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 16, 128,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 16, 192,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 16, 256,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 32, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 32, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 32, 8,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 32, 12,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 32, 16,
                                    VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 32, 24,
                                    VecT<float32_t, 1>, VecT<float32_t, 3>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 32, 32,
                                    VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 32, 48,
                                    VecT<float32_t, 3>, VecT<float32_t, 6>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 32, 64,
                                    VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 32, 96,
                                    VecT<float32_t, 6>, VecT<float32_t, 12>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 32, 128,
                                    VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 32, 192,
                                    VecT<float32_t, 12>, VecT<float32_t, 24>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 32, 256,
                                    VecT<float32_t, 16>, VecT<float32_t, 32>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 16, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 16, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 16, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 16, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 16, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 16, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 16, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 16, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 16, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 16, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 16, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 16, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 12>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 16, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 32, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 32, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 32, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 32, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 32, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 32, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 32, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 32, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 32, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 32, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 32, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 32, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 24>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 32, 32, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 32>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 8, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 8, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 8, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 8, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 8, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 8, 8,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 8, 12,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 8, 16,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 8, 24,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 8, 32,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 8, 48,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 8, 64,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 8, 96,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 8, 128,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 8, 192,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float16_t, 16, 8, 256,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 16, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 16, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 16, 8,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 16, 12,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 16, 16,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 16, 24,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 16, 32,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 16, 48,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 16, 64,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 16, 96,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 16, 128,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 16, 192,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 16, 256,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 32, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 32, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 32, 8,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 32, 12,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 32, 16,
                                    VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 32, 24,
                                    VecT<float32_t, 1>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 32, 32,
                                    VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 32, 48,
                                    VecT<float32_t, 3>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 32, 64,
                                    VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 32, 96,
                                    VecT<float32_t, 6>, VecT<float32_t, 12>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 32, 128,
                                    VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 32, 192,
                                    VecT<float32_t, 12>, VecT<float32_t, 24>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 32, 256,
                                    VecT<float32_t, 16>, VecT<float32_t, 32>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 16, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 16, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 16, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 16, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 16, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 16, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 16, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 16, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 16, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 16, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 16, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 16, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 12>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 16, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 32, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 32, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 32, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 32, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 32, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 32, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 32, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 32, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 32, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 32, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 32, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 32, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 24>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 32, 32, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 8, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 8, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 8, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 8, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 8, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 8, 8,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 8, 12,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 8, 16,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 8, 24,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 8, 32,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 8, 48,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 8, 64,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 8, 96,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 8, 128,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 8, 192,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, bfloat8_t, float32_t, 16, 8, 256,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 16, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 16, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 16, 8,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 16, 12,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 16, 16,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 16, 24,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 16, 32,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 16, 48,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 16, 64,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 16, 96,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 16, 128,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 16, 192,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 16, 256,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 32, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 32, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 32, 8,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 32, 12,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 32, 16,
                                    VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 32, 24,
                                    VecT<float32_t, 1>, VecT<float32_t, 3>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 32, 32,
                                    VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 32, 48,
                                    VecT<float32_t, 3>, VecT<float32_t, 6>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 32, 64,
                                    VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 32, 96,
                                    VecT<float32_t, 6>, VecT<float32_t, 12>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 32, 128,
                                    VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 32, 192,
                                    VecT<float32_t, 12>, VecT<float32_t, 24>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 32, 256,
                                    VecT<float32_t, 16>, VecT<float32_t, 32>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 16, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 16, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 16, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 16, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 16, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 16, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 16, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 16, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 16, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 16, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 16, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 16, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 12>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 16, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 32, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 32, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 32, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 32, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 32, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 32, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 32, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 32, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 32, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 32, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 32, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 32, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 24>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 32, 32, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 32>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 8, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 8, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 8, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 8, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 8, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 8, 8,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 8, 12,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 8, 16,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 8, 24,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 8, 32,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 8, 48,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 8, 64,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 8, 96,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 8, 128,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 8, 192,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float16_t, 16, 8, 256,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 16, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 16, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 16, 8,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 16, 12,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 16, 16,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 16, 24,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 16, 32,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 16, 48,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 16, 64,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 16, 96,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 16, 128,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 16, 192,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 16, 256,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 32, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 32, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 32, 8,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 32, 12,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 32, 16,
                                    VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 32, 24,
                                    VecT<float32_t, 1>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 32, 32,
                                    VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 32, 48,
                                    VecT<float32_t, 3>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 32, 64,
                                    VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 32, 96,
                                    VecT<float32_t, 6>, VecT<float32_t, 12>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 32, 128,
                                    VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 32, 192,
                                    VecT<float32_t, 12>, VecT<float32_t, 24>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 32, 256,
                                    VecT<float32_t, 16>, VecT<float32_t, 32>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 16, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 16, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 16, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 16, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 16, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 16, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 16, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 16, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 16, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 16, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 16, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 16, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 12>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 16, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 32, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 32, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 32, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 32, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 32, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 32, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 32, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 32, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 32, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 32, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 32, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 32, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 24>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 32, 32, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 8, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 8, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 8, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 8, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 8, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 8, 8,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 8, 12,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 8, 16,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 8, 24,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 8, 32,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 8, 48,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 8, 64,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 8, 96,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 8, 128,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 8, 192,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, bfloat8_t, float8_t, float32_t, 16, 8, 256,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 16, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 16, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 16, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 16, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 16, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 16, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 16, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 16, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 16, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 16, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 16, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 16, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 24>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 16, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 32>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 32, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 32, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 32, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 32, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 3>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 32, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 32, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 6>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 32, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 32, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 12>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 32, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 32, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 24>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 32, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 32>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 32, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 48>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 32, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 64>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 16, 4,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 16, 6,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 16, 8,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 16, 12,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 16, 16,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 16, 24,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 16, 32,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 16, 48,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 16, 64,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 16, 96,
                                    VecT<float32_t, 24>, VecT<float32_t, 12>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 16, 128,
                                    VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 16, 192,
                                    VecT<float32_t, 48>, VecT<float32_t, 24>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 16, 256,
                                    VecT<float32_t, 64>, VecT<float32_t, 32>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 32, 4,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 32, 6,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 32, 8,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 32, 12,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 32, 16,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 32, 24,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 32, 32,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 32, 48,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 32, 64,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 32, 96,
                                    VecT<float32_t, 24>, VecT<float32_t, 24>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 32, 128,
                                    VecT<float32_t, 32>, VecT<float32_t, 32>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 32, 192,
                                    VecT<float32_t, 48>, VecT<float32_t, 48>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 32, 32, 256,
                                    VecT<float32_t, 64>, VecT<float32_t, 64>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 8, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 8, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 8, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 8, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 8, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 8, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 8, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 8, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 8, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 8, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 8, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 8, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 8, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 8, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 8, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 12>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float16_t, 16, 8, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 16, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 16, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 16, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 16, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 16, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 16, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 16, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 16, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 16, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 16, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 16, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 16, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 24>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 16, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 32>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 32, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 32, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 32, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 32, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 32, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 32, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 32, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 32, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 12>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 32, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 32, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 24>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 32, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 32>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 32, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 48>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 32, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 64>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 16, 4,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 16, 6,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 16, 8,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 16, 12,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 16, 16,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 16, 24,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 16, 32,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 16, 48,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 16, 64,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 16, 96,
                                    VecT<float32_t, 24>, VecT<float32_t, 12>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 16, 128,
                                    VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 16, 192,
                                    VecT<float32_t, 48>, VecT<float32_t, 24>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 16, 256,
                                    VecT<float32_t, 64>, VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 32, 4,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 32, 6,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 32, 8,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 32, 12,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 32, 16,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 32, 24,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 32, 32,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 32, 48,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 32, 64,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 32, 96,
                                    VecT<float32_t, 24>, VecT<float32_t, 24>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 32, 128,
                                    VecT<float32_t, 32>, VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 32, 192,
                                    VecT<float32_t, 48>, VecT<float32_t, 48>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 32, 32, 256,
                                    VecT<float32_t, 64>, VecT<float32_t, 64>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 8, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 8, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 8, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 8, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 8, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 8, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 8, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 8, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 8, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 8, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 8, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 8, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 8, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 8, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 8, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 12>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float16_t, float16_t, float32_t, 16, 8, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 16, 4,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 16, 6,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 16, 8,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 16, 12,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 16, 16,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 16, 24,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 16, 32,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 16, 48,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 16, 64,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 16, 96,
                                    VecT<float32_t, 24>, VecT<float32_t, 24>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 16, 128,
                                    VecT<float32_t, 32>, VecT<float32_t, 32>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 16, 192,
                                    VecT<float32_t, 48>, VecT<float32_t, 48>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 16, 256,
                                    VecT<float32_t, 64>, VecT<float32_t, 64>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 32, 4,
                                    VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 32, 6,
                                    VecT<float32_t, 1>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 32, 8,
                                    VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 32, 12,
                                    VecT<float32_t, 3>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 32, 16,
                                    VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 32, 24,
                                    VecT<float32_t, 6>, VecT<float32_t, 12>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 32, 32,
                                    VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 32, 48,
                                    VecT<float32_t, 12>, VecT<float32_t, 24>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 32, 64,
                                    VecT<float32_t, 16>, VecT<float32_t, 32>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 32, 96,
                                    VecT<float32_t, 24>, VecT<float32_t, 48>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 32, 128,
                                    VecT<float32_t, 32>, VecT<float32_t, 64>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 32, 192,
                                    VecT<float32_t, 48>, VecT<float32_t, 96>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 32, 256,
                                    VecT<float32_t, 64>, VecT<float32_t, 128>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 16, 2,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 16, 3,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 16, 4,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 16, 6,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 16, 8,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 16, 12,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 16, 16,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 16, 24,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 16, 32,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 16, 48,
                                    VecT<float32_t, 24>, VecT<float32_t, 12>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 16, 64,
                                    VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 16, 96,
                                    VecT<float32_t, 48>, VecT<float32_t, 24>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 16, 128,
                                    VecT<float32_t, 64>, VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 16, 192,
                                    VecT<float32_t, 96>, VecT<float32_t, 48>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 16, 256,
                                    VecT<float32_t, 128>, VecT<float32_t, 64>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 32, 2,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 32, 3,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 32, 4,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 32, 6,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 32, 8,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 32, 12,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 32, 16,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 32, 24,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 32, 32,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 32, 48,
                                    VecT<float32_t, 24>, VecT<float32_t, 24>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 32, 64,
                                    VecT<float32_t, 32>, VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 32, 96,
                                    VecT<float32_t, 48>, VecT<float32_t, 48>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 32, 128,
                                    VecT<float32_t, 64>, VecT<float32_t, 64>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 32, 192,
                                    VecT<float32_t, 96>, VecT<float32_t, 96>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 32, 32, 256,
                                    VecT<float32_t, 128>, VecT<float32_t, 128>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 8, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 8, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 8, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 8, 4,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 8, 6,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 8, 8,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 8, 12,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 8, 16,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 8, 24,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 8, 32,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 8, 48,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 8, 64,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 8, 96,
                                    VecT<float32_t, 24>, VecT<float32_t, 12>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 8, 128,
                                    VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 8, 192,
                                    VecT<float32_t, 48>, VecT<float32_t, 24>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float32_t, float32_t, float32_t, 16, 8, 256,
                                    VecT<float32_t, 64>, VecT<float32_t, 32>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 16, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 16, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 16, 8,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 16, 12,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 16, 16,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 16, 24,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 16, 32,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 16, 48,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 16, 64,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 16, 96,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 16, 128,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 16, 192,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 16, 256,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 32, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 32, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 32, 8,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 32, 12,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 32, 16,
                                    VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 32, 24,
                                    VecT<float32_t, 1>, VecT<float32_t, 3>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 32, 32,
                                    VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 32, 48,
                                    VecT<float32_t, 3>, VecT<float32_t, 6>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 32, 64,
                                    VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 32, 96,
                                    VecT<float32_t, 6>, VecT<float32_t, 12>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 32, 128,
                                    VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 32, 192,
                                    VecT<float32_t, 12>, VecT<float32_t, 24>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 32, 256,
                                    VecT<float32_t, 16>, VecT<float32_t, 32>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 16, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 16, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 16, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 16, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 16, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 16, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 16, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 16, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 16, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 16, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 16, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 16, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 12>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 16, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 32, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 32, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 32, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 32, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 32, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 32, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 32, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 32, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 32, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 32, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 32, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 32, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 24>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 32, 32, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 32>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 8, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 8, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 8, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 8, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 8, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 8, 8,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 8, 12,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 8, 16,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 8, 24,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 8, 32,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 8, 48,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 8, 64,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 8, 96,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 8, 128,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 8, 192,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float16_t, 16, 8, 256,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 16, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 16, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 16, 8,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 16, 12,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 16, 16,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 16, 24,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 16, 32,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 16, 48,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 16, 64,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 16, 96,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 16, 128,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 16, 192,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 16, 256,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 32, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 32, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 32, 8,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 32, 12,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 32, 16,
                                    VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 32, 24,
                                    VecT<float32_t, 1>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 32, 32,
                                    VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 32, 48,
                                    VecT<float32_t, 3>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 32, 64,
                                    VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 32, 96,
                                    VecT<float32_t, 6>, VecT<float32_t, 12>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 32, 128,
                                    VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 32, 192,
                                    VecT<float32_t, 12>, VecT<float32_t, 24>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 32, 256,
                                    VecT<float32_t, 16>, VecT<float32_t, 32>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 16, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 16, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 16, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 16, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 16, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 16, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 16, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 16, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 16, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 16, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 16, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 16, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 12>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 16, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 32, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 32, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 32, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 32, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 32, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 32, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 32, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 32, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 32, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 32, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 32, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 32, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 24>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 32, 32, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 8, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 8, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 8, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 8, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 8, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 8, 8,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 8, 12,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 8, 16,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 8, 24,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 8, 32,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 8, 48,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 8, 64,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 8, 96,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 8, 128,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 8, 192,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, bfloat8_t, float32_t, 16, 8, 256,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 16, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 16, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 16, 8,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 16, 12,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 16, 16,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 16, 24,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 16, 32,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 16, 48,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 16, 64,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 16, 96,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 16, 128,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 16, 192,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 16, 256,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 32, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 32, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 32, 8,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 32, 12,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 32, 16,
                                    VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 32, 24,
                                    VecT<float32_t, 1>, VecT<float32_t, 3>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 32, 32,
                                    VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 32, 48,
                                    VecT<float32_t, 3>, VecT<float32_t, 6>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 32, 64,
                                    VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 32, 96,
                                    VecT<float32_t, 6>, VecT<float32_t, 12>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 32, 128,
                                    VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 32, 192,
                                    VecT<float32_t, 12>, VecT<float32_t, 24>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 32, 256,
                                    VecT<float32_t, 16>, VecT<float32_t, 32>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 16, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 16, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 16, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 16, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 16, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 16, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 16, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 16, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 16, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 16, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 16, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 16, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 12>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 16, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 32, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 32, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 32, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 32, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 32, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 32, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 32, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 32, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 32, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 32, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 32, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 32, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 24>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 32, 32, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 32>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 8, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 8, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 8, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 8, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 8, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 8, 8,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 8, 12,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 8, 16,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 8, 24,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 8, 32,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 8, 48,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 8, 64,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 8, 96,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 8, 128,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 8, 192,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float16_t, 16, 8, 256,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 16, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 16, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 16, 8,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 16, 12,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 16, 16,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 16, 24,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 16, 32,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 16, 48,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 16, 64,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 16, 96,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 16, 128,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 16, 192,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 16, 256,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 32, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 32, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 32, 8,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 32, 12,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 32, 16,
                                    VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 32, 24,
                                    VecT<float32_t, 1>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 32, 32,
                                    VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 32, 48,
                                    VecT<float32_t, 3>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 32, 64,
                                    VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 32, 96,
                                    VecT<float32_t, 6>, VecT<float32_t, 12>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 32, 128,
                                    VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 32, 192,
                                    VecT<float32_t, 12>, VecT<float32_t, 24>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 32, 256,
                                    VecT<float32_t, 16>, VecT<float32_t, 32>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 16, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 16, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 16, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 16, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 16, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 16, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 16, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 16, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 16, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 16, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 16, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 16, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 12>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 16, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 32, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 32, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 32, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 32, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 32, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 32, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 32, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 32, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 32, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 32, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 32, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 32, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 24>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 32, 32, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 8, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 8, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 8, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 8, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 8, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 8, 8,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 8, 12,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 8, 16,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 8, 24,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 8, 32,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 8, 48,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 8, 64,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 8, 96,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 8, 128,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 8, 192,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, float8_t, float8_t, float32_t, 16, 8, 256,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 16, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 16, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 16, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 16, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 16, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 16, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 16, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 16, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 16, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 16, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 16, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 16, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 24>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 16, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 32>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 32, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 32, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 32, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 32, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 32, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 32, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 32, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 32, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 12>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 32, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 32, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 24>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 32, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 32>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 32, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 48>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 32, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 64>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 16, 4,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 16, 6,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 16, 8,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 16, 12,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 16, 16,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 16, 24,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 16, 32,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 16, 48,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 16, 64,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 16, 96,
                                    VecT<float32_t, 24>, VecT<float32_t, 12>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 16, 128,
                                    VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 16, 192,
                                    VecT<float32_t, 48>, VecT<float32_t, 24>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 16, 256,
                                    VecT<float32_t, 64>, VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 32, 4,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 32, 6,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 32, 8,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 32, 12,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 32, 16,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 32, 24,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 32, 32,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 32, 48,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 32, 64,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 32, 96,
                                    VecT<float32_t, 24>, VecT<float32_t, 24>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 32, 128,
                                    VecT<float32_t, 32>, VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 32, 192,
                                    VecT<float32_t, 48>, VecT<float32_t, 48>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 32, 32, 256,
                                    VecT<float32_t, 64>, VecT<float32_t, 64>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 8, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 8, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 8, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 8, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 8, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 8, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 8, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 8, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 8, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 8, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 8, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 8, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 8, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 8, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 8, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 12>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, float32_t, 16, 8, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 24>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 32>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 3>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 6>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 12>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 24>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 32>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 48>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 64>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, 4,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, 6,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, 8,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, 12,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, 16,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, 24,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, 32,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, 48,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, 64,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, 96,
                                    VecT<float32_t, 24>, VecT<float32_t, 12>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, 128,
                                    VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, 192,
                                    VecT<float32_t, 48>, VecT<float32_t, 24>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, 256,
                                    VecT<float32_t, 64>, VecT<float32_t, 32>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, 4,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, 6,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, 8,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, 12,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, 16,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, 24,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, 32,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, 48,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, 64,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, 96,
                                    VecT<float32_t, 24>, VecT<float32_t, 24>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, 128,
                                    VecT<float32_t, 32>, VecT<float32_t, 32>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, 192,
                                    VecT<float32_t, 48>, VecT<float32_t, 48>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, 256,
                                    VecT<float32_t, 64>, VecT<float32_t, 64>, VecT<float32_t, 8>, VecT<float32_t, 8>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, 4,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, 6,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, 8,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, 12,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, 16,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, 24,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, 32,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, 48,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, 64,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, 96,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, 128,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, 192,
                                    VecT<float32_t, 24>, VecT<float32_t, 12>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, 256,
                                    VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 16, 1,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 16, 2,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 16, 3,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 16, 4,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 16, 6,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 16, 8,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 16, 12,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 16, 16,
                                    VecT<int32_t, 1>, VecT<int32_t, 1>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 16, 24,
                                    VecT<int32_t, 1>, VecT<int32_t, 1>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 16, 32,
                                    VecT<int32_t, 2>, VecT<int32_t, 2>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 16, 48,
                                    VecT<int32_t, 3>, VecT<int32_t, 3>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 16, 64,
                                    VecT<int32_t, 4>, VecT<int32_t, 4>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 16, 96,
                                    VecT<int32_t, 6>, VecT<int32_t, 6>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 16, 128,
                                    VecT<int32_t, 8>, VecT<int32_t, 8>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 16, 192,
                                    VecT<int32_t, 12>, VecT<int32_t, 12>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 16, 256,
                                    VecT<int32_t, 16>, VecT<int32_t, 16>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 32, 1,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 32, 2,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 32, 3,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 32, 4,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 32, 6,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 32, 8,
                                    VecT<int32_t, 0>, VecT<int32_t, 1>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 32, 12,
                                    VecT<int32_t, 0>, VecT<int32_t, 1>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 32, 16,
                                    VecT<int32_t, 1>, VecT<int32_t, 2>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 32, 24,
                                    VecT<int32_t, 1>, VecT<int32_t, 3>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 32, 32,
                                    VecT<int32_t, 2>, VecT<int32_t, 4>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 32, 48,
                                    VecT<int32_t, 3>, VecT<int32_t, 6>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 32, 64,
                                    VecT<int32_t, 4>, VecT<int32_t, 8>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 32, 96,
                                    VecT<int32_t, 6>, VecT<int32_t, 12>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 32, 128,
                                    VecT<int32_t, 8>, VecT<int32_t, 16>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 32, 192,
                                    VecT<int32_t, 12>, VecT<int32_t, 24>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 32, 256,
                                    VecT<int32_t, 16>, VecT<int32_t, 32>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 16, 1,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 16, 2,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 16, 3,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 16, 4,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 16, 6,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 16, 8,
                                    VecT<int32_t, 1>, VecT<int32_t, 0>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 16, 12,
                                    VecT<int32_t, 1>, VecT<int32_t, 0>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 16, 16,
                                    VecT<int32_t, 2>, VecT<int32_t, 1>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 16, 24,
                                    VecT<int32_t, 3>, VecT<int32_t, 1>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 16, 32,
                                    VecT<int32_t, 4>, VecT<int32_t, 2>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 16, 48,
                                    VecT<int32_t, 6>, VecT<int32_t, 3>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 16, 64,
                                    VecT<int32_t, 8>, VecT<int32_t, 4>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 16, 96,
                                    VecT<int32_t, 12>, VecT<int32_t, 6>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 16, 128,
                                    VecT<int32_t, 16>, VecT<int32_t, 8>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 16, 192,
                                    VecT<int32_t, 24>, VecT<int32_t, 12>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 16, 256,
                                    VecT<int32_t, 32>, VecT<int32_t, 16>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 32, 1,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 32, 2,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 32, 3,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 32, 4,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 32, 6,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 32, 8,
                                    VecT<int32_t, 1>, VecT<int32_t, 1>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 32, 12,
                                    VecT<int32_t, 1>, VecT<int32_t, 1>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 32, 16,
                                    VecT<int32_t, 2>, VecT<int32_t, 2>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 32, 24,
                                    VecT<int32_t, 3>, VecT<int32_t, 3>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 32, 32,
                                    VecT<int32_t, 4>, VecT<int32_t, 4>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 32, 48,
                                    VecT<int32_t, 6>, VecT<int32_t, 6>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 32, 64,
                                    VecT<int32_t, 8>, VecT<int32_t, 8>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 32, 96,
                                    VecT<int32_t, 12>, VecT<int32_t, 12>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 32, 128,
                                    VecT<int32_t, 16>, VecT<int32_t, 16>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 32, 192,
                                    VecT<int32_t, 24>, VecT<int32_t, 24>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 32, 32, 256,
                                    VecT<int32_t, 32>, VecT<int32_t, 32>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 8, 1,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 8, 2,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 8, 3,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 8, 4,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 8, 6,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 8, 8,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 8, 12,
                                    VecT<int32_t, 0>, VecT<int32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 8, 16,
                                    VecT<int32_t, 1>, VecT<int32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 8, 24,
                                    VecT<int32_t, 1>, VecT<int32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 8, 32,
                                    VecT<int32_t, 2>, VecT<int32_t, 1>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 8, 48,
                                    VecT<int32_t, 3>, VecT<int32_t, 1>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 8, 64,
                                    VecT<int32_t, 4>, VecT<int32_t, 2>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 8, 96,
                                    VecT<int32_t, 6>, VecT<int32_t, 3>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 8, 128,
                                    VecT<int32_t, 8>, VecT<int32_t, 4>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 8, 192,
                                    VecT<int32_t, 12>, VecT<int32_t, 6>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, int8_t, int8_t, int32_t, 16, 8, 256,
                                    VecT<int32_t, 16>, VecT<int32_t, 8>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 16, 1,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 16, 2,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 16, 3,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 16, 4,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 16, 6,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 16, 8,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 16, 12,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 16, 16,
                                    VecT<uint32_t, 1>, VecT<uint32_t, 1>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 16, 24,
                                    VecT<uint32_t, 1>, VecT<uint32_t, 1>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 16, 32,
                                    VecT<uint32_t, 2>, VecT<uint32_t, 2>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 16, 48,
                                    VecT<uint32_t, 3>, VecT<uint32_t, 3>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 16, 64,
                                    VecT<uint32_t, 4>, VecT<uint32_t, 4>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 16, 96,
                                    VecT<uint32_t, 6>, VecT<uint32_t, 6>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 16, 128,
                                    VecT<uint32_t, 8>, VecT<uint32_t, 8>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 16, 192,
                                    VecT<uint32_t, 12>, VecT<uint32_t, 12>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 16, 256,
                                    VecT<uint32_t, 16>, VecT<uint32_t, 16>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 32, 1,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 32, 2,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 32, 3,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 32, 4,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 32, 6,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 32, 8,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 1>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 32, 12,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 1>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 32, 16,
                                    VecT<uint32_t, 1>, VecT<uint32_t, 2>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 32, 24,
                                    VecT<uint32_t, 1>, VecT<uint32_t, 3>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 32, 32,
                                    VecT<uint32_t, 2>, VecT<uint32_t, 4>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 32, 48,
                                    VecT<uint32_t, 3>, VecT<uint32_t, 6>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 32, 64,
                                    VecT<uint32_t, 4>, VecT<uint32_t, 8>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 32, 96,
                                    VecT<uint32_t, 6>, VecT<uint32_t, 12>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 32, 128,
                                    VecT<uint32_t, 8>, VecT<uint32_t, 16>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 32, 192,
                                    VecT<uint32_t, 12>, VecT<uint32_t, 24>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 32, 256,
                                    VecT<uint32_t, 16>, VecT<uint32_t, 32>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 16, 1,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 16, 2,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 16, 3,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 16, 4,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 16, 6,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 16, 8,
                                    VecT<uint32_t, 1>, VecT<uint32_t, 0>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 16, 12,
                                    VecT<uint32_t, 1>, VecT<uint32_t, 0>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 16, 16,
                                    VecT<uint32_t, 2>, VecT<uint32_t, 1>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 16, 24,
                                    VecT<uint32_t, 3>, VecT<uint32_t, 1>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 16, 32,
                                    VecT<uint32_t, 4>, VecT<uint32_t, 2>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 16, 48,
                                    VecT<uint32_t, 6>, VecT<uint32_t, 3>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 16, 64,
                                    VecT<uint32_t, 8>, VecT<uint32_t, 4>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 16, 96,
                                    VecT<uint32_t, 12>, VecT<uint32_t, 6>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 16, 128,
                                    VecT<uint32_t, 16>, VecT<uint32_t, 8>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 16, 192,
                                    VecT<uint32_t, 24>, VecT<uint32_t, 12>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 16, 256,
                                    VecT<uint32_t, 32>, VecT<uint32_t, 16>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 32, 1,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 32, 2,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 32, 3,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 32, 4,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 32, 6,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 32, 8,
                                    VecT<uint32_t, 1>, VecT<uint32_t, 1>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 32, 12,
                                    VecT<uint32_t, 1>, VecT<uint32_t, 1>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 32, 16,
                                    VecT<uint32_t, 2>, VecT<uint32_t, 2>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 32, 24,
                                    VecT<uint32_t, 3>, VecT<uint32_t, 3>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 32, 32,
                                    VecT<uint32_t, 4>, VecT<uint32_t, 4>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 32, 48,
                                    VecT<uint32_t, 6>, VecT<uint32_t, 6>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 32, 64,
                                    VecT<uint32_t, 8>, VecT<uint32_t, 8>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 32, 96,
                                    VecT<uint32_t, 12>, VecT<uint32_t, 12>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 32, 128,
                                    VecT<uint32_t, 16>, VecT<uint32_t, 16>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 32, 192,
                                    VecT<uint32_t, 24>, VecT<uint32_t, 24>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 32, 32, 256,
                                    VecT<uint32_t, 32>, VecT<uint32_t, 32>, VecT<int32_t, 16>, VecT<int32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 8, 1,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 8, 2,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 8, 3,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 8, 4,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 8, 6,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 8, 8,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 8, 12,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 8, 16,
                                    VecT<uint32_t, 1>, VecT<uint32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 8, 24,
                                    VecT<uint32_t, 1>, VecT<uint32_t, 0>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 8, 32,
                                    VecT<uint32_t, 2>, VecT<uint32_t, 1>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 8, 48,
                                    VecT<uint32_t, 3>, VecT<uint32_t, 1>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 8, 64,
                                    VecT<uint32_t, 4>, VecT<uint32_t, 2>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 8, 96,
                                    VecT<uint32_t, 6>, VecT<uint32_t, 3>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 8, 128,
                                    VecT<uint32_t, 8>, VecT<uint32_t, 4>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 8, 192,
                                    VecT<uint32_t, 12>, VecT<uint32_t, 6>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, int32_t, 16, 8, 256,
                                    VecT<uint32_t, 16>, VecT<uint32_t, 8>, VecT<int32_t, 4>, VecT<int32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 16, 1,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 16, 2,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 16, 3,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 16, 4,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 16, 6,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 16, 8,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 16, 12,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 16, 16,
                                    VecT<uint32_t, 1>, VecT<uint32_t, 1>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 16, 24,
                                    VecT<uint32_t, 1>, VecT<uint32_t, 1>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 16, 32,
                                    VecT<uint32_t, 2>, VecT<uint32_t, 2>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 16, 48,
                                    VecT<uint32_t, 3>, VecT<uint32_t, 3>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 16, 64,
                                    VecT<uint32_t, 4>, VecT<uint32_t, 4>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 16, 96,
                                    VecT<uint32_t, 6>, VecT<uint32_t, 6>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 16, 128,
                                    VecT<uint32_t, 8>, VecT<uint32_t, 8>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 16, 192,
                                    VecT<uint32_t, 12>, VecT<uint32_t, 12>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 16, 256,
                                    VecT<uint32_t, 16>, VecT<uint32_t, 16>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 32, 1,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 32, 2,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 32, 3,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 32, 4,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 32, 6,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 32, 8,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 1>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 32, 12,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 1>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 32, 16,
                                    VecT<uint32_t, 1>, VecT<uint32_t, 2>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 32, 24,
                                    VecT<uint32_t, 1>, VecT<uint32_t, 3>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 32, 32,
                                    VecT<uint32_t, 2>, VecT<uint32_t, 4>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 32, 48,
                                    VecT<uint32_t, 3>, VecT<uint32_t, 6>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 32, 64,
                                    VecT<uint32_t, 4>, VecT<uint32_t, 8>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 32, 96,
                                    VecT<uint32_t, 6>, VecT<uint32_t, 12>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 32, 128,
                                    VecT<uint32_t, 8>, VecT<uint32_t, 16>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 32, 192,
                                    VecT<uint32_t, 12>, VecT<uint32_t, 24>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 32, 256,
                                    VecT<uint32_t, 16>, VecT<uint32_t, 32>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 16, 1,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 16, 2,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 16, 3,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 16, 4,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 16, 6,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 16, 8,
                                    VecT<uint32_t, 1>, VecT<uint32_t, 0>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 16, 12,
                                    VecT<uint32_t, 1>, VecT<uint32_t, 0>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 16, 16,
                                    VecT<uint32_t, 2>, VecT<uint32_t, 1>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 16, 24,
                                    VecT<uint32_t, 3>, VecT<uint32_t, 1>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 16, 32,
                                    VecT<uint32_t, 4>, VecT<uint32_t, 2>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 16, 48,
                                    VecT<uint32_t, 6>, VecT<uint32_t, 3>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 16, 64,
                                    VecT<uint32_t, 8>, VecT<uint32_t, 4>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 16, 96,
                                    VecT<uint32_t, 12>, VecT<uint32_t, 6>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 16, 128,
                                    VecT<uint32_t, 16>, VecT<uint32_t, 8>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 16, 192,
                                    VecT<uint32_t, 24>, VecT<uint32_t, 12>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 16, 256,
                                    VecT<uint32_t, 32>, VecT<uint32_t, 16>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 32, 1,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 32, 2,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 32, 3,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 32, 4,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 32, 6,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 32, 8,
                                    VecT<uint32_t, 1>, VecT<uint32_t, 1>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 32, 12,
                                    VecT<uint32_t, 1>, VecT<uint32_t, 1>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 32, 16,
                                    VecT<uint32_t, 2>, VecT<uint32_t, 2>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 32, 24,
                                    VecT<uint32_t, 3>, VecT<uint32_t, 3>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 32, 32,
                                    VecT<uint32_t, 4>, VecT<uint32_t, 4>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 32, 48,
                                    VecT<uint32_t, 6>, VecT<uint32_t, 6>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 32, 64,
                                    VecT<uint32_t, 8>, VecT<uint32_t, 8>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 32, 96,
                                    VecT<uint32_t, 12>, VecT<uint32_t, 12>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 32, 128,
                                    VecT<uint32_t, 16>, VecT<uint32_t, 16>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 32, 192,
                                    VecT<uint32_t, 24>, VecT<uint32_t, 24>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 32, 32, 256,
                                    VecT<uint32_t, 32>, VecT<uint32_t, 32>, VecT<uint32_t, 16>, VecT<uint32_t, 16>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 8, 1,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 8, 2,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 8, 3,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 8, 4,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 8, 6,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 8, 8,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 8, 12,
                                    VecT<uint32_t, 0>, VecT<uint32_t, 0>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 8, 16,
                                    VecT<uint32_t, 1>, VecT<uint32_t, 0>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 8, 24,
                                    VecT<uint32_t, 1>, VecT<uint32_t, 0>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 8, 32,
                                    VecT<uint32_t, 2>, VecT<uint32_t, 1>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 8, 48,
                                    VecT<uint32_t, 3>, VecT<uint32_t, 1>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 8, 64,
                                    VecT<uint32_t, 4>, VecT<uint32_t, 2>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 8, 96,
                                    VecT<uint32_t, 6>, VecT<uint32_t, 3>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 8, 128,
                                    VecT<uint32_t, 8>, VecT<uint32_t, 4>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 8, 192,
                                    VecT<uint32_t, 12>, VecT<uint32_t, 6>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, uint8_t, uint8_t, uint32_t, 16, 8, 256,
                                    VecT<uint32_t, 16>, VecT<uint32_t, 8>, VecT<uint32_t, 4>, VecT<uint32_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 16, 1,
                                    VecT<float64_t, 0>, VecT<float64_t, 0>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 16, 2,
                                    VecT<float64_t, 0>, VecT<float64_t, 0>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 16, 3,
                                    VecT<float64_t, 0>, VecT<float64_t, 0>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 16, 4,
                                    VecT<float64_t, 1>, VecT<float64_t, 1>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 16, 6,
                                    VecT<float64_t, 1>, VecT<float64_t, 1>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 16, 8,
                                    VecT<float64_t, 2>, VecT<float64_t, 2>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 16, 12,
                                    VecT<float64_t, 3>, VecT<float64_t, 3>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 16, 16,
                                    VecT<float64_t, 4>, VecT<float64_t, 4>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 16, 24,
                                    VecT<float64_t, 6>, VecT<float64_t, 6>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 16, 32,
                                    VecT<float64_t, 8>, VecT<float64_t, 8>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 16, 48,
                                    VecT<float64_t, 12>, VecT<float64_t, 12>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 16, 64,
                                    VecT<float64_t, 16>, VecT<float64_t, 16>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 16, 96,
                                    VecT<float64_t, 24>, VecT<float64_t, 24>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 16, 128,
                                    VecT<float64_t, 32>, VecT<float64_t, 32>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 16, 192,
                                    VecT<float64_t, 48>, VecT<float64_t, 48>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 16, 256,
                                    VecT<float64_t, 64>, VecT<float64_t, 64>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 32, 1,
                                    VecT<float64_t, 0>, VecT<float64_t, 0>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 32, 2,
                                    VecT<float64_t, 0>, VecT<float64_t, 1>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 32, 3,
                                    VecT<float64_t, 0>, VecT<float64_t, 1>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 32, 4,
                                    VecT<float64_t, 1>, VecT<float64_t, 2>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 32, 6,
                                    VecT<float64_t, 1>, VecT<float64_t, 3>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 32, 8,
                                    VecT<float64_t, 2>, VecT<float64_t, 4>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 32, 12,
                                    VecT<float64_t, 3>, VecT<float64_t, 6>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 32, 16,
                                    VecT<float64_t, 4>, VecT<float64_t, 8>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 32, 24,
                                    VecT<float64_t, 6>, VecT<float64_t, 12>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 32, 32,
                                    VecT<float64_t, 8>, VecT<float64_t, 16>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 32, 48,
                                    VecT<float64_t, 12>, VecT<float64_t, 24>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 32, 64,
                                    VecT<float64_t, 16>, VecT<float64_t, 32>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 32, 96,
                                    VecT<float64_t, 24>, VecT<float64_t, 48>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 32, 128,
                                    VecT<float64_t, 32>, VecT<float64_t, 64>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 32, 192,
                                    VecT<float64_t, 48>, VecT<float64_t, 96>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 32, 256,
                                    VecT<float64_t, 64>, VecT<float64_t, 128>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 16, 1,
                                    VecT<float64_t, 0>, VecT<float64_t, 0>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 16, 2,
                                    VecT<float64_t, 1>, VecT<float64_t, 0>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 16, 3,
                                    VecT<float64_t, 1>, VecT<float64_t, 0>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 16, 4,
                                    VecT<float64_t, 2>, VecT<float64_t, 1>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 16, 6,
                                    VecT<float64_t, 3>, VecT<float64_t, 1>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 16, 8,
                                    VecT<float64_t, 4>, VecT<float64_t, 2>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 16, 12,
                                    VecT<float64_t, 6>, VecT<float64_t, 3>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 16, 16,
                                    VecT<float64_t, 8>, VecT<float64_t, 4>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 16, 24,
                                    VecT<float64_t, 12>, VecT<float64_t, 6>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 16, 32,
                                    VecT<float64_t, 16>, VecT<float64_t, 8>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 16, 48,
                                    VecT<float64_t, 24>, VecT<float64_t, 12>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 16, 64,
                                    VecT<float64_t, 32>, VecT<float64_t, 16>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 16, 96,
                                    VecT<float64_t, 48>, VecT<float64_t, 24>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 16, 128,
                                    VecT<float64_t, 64>, VecT<float64_t, 32>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 16, 192,
                                    VecT<float64_t, 96>, VecT<float64_t, 48>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 16, 256,
                                    VecT<float64_t, 128>, VecT<float64_t, 64>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 32, 1,
                                    VecT<float64_t, 0>, VecT<float64_t, 0>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 32, 2,
                                    VecT<float64_t, 1>, VecT<float64_t, 1>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 32, 3,
                                    VecT<float64_t, 1>, VecT<float64_t, 1>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 32, 4,
                                    VecT<float64_t, 2>, VecT<float64_t, 2>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 32, 6,
                                    VecT<float64_t, 3>, VecT<float64_t, 3>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 32, 8,
                                    VecT<float64_t, 4>, VecT<float64_t, 4>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 32, 12,
                                    VecT<float64_t, 6>, VecT<float64_t, 6>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 32, 16,
                                    VecT<float64_t, 8>, VecT<float64_t, 8>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 32, 24,
                                    VecT<float64_t, 12>, VecT<float64_t, 12>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 32, 32,
                                    VecT<float64_t, 16>, VecT<float64_t, 16>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 32, 48,
                                    VecT<float64_t, 24>, VecT<float64_t, 24>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 32, 64,
                                    VecT<float64_t, 32>, VecT<float64_t, 32>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 32, 96,
                                    VecT<float64_t, 48>, VecT<float64_t, 48>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 32, 128,
                                    VecT<float64_t, 64>, VecT<float64_t, 64>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 32, 192,
                                    VecT<float64_t, 96>, VecT<float64_t, 96>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 32, 32, 256,
                                    VecT<float64_t, 128>, VecT<float64_t, 128>, VecT<float64_t, 16>, VecT<float64_t, 16>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 8, 1,
                                    VecT<float64_t, 0>, VecT<float64_t, 0>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 8, 2,
                                    VecT<float64_t, 0>, VecT<float64_t, 0>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 8, 3,
                                    VecT<float64_t, 0>, VecT<float64_t, 0>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 8, 4,
                                    VecT<float64_t, 1>, VecT<float64_t, 0>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 8, 6,
                                    VecT<float64_t, 1>, VecT<float64_t, 0>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 8, 8,
                                    VecT<float64_t, 2>, VecT<float64_t, 1>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 8, 12,
                                    VecT<float64_t, 3>, VecT<float64_t, 1>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 8, 16,
                                    VecT<float64_t, 4>, VecT<float64_t, 2>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 8, 24,
                                    VecT<float64_t, 6>, VecT<float64_t, 3>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 8, 32,
                                    VecT<float64_t, 8>, VecT<float64_t, 4>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 8, 48,
                                    VecT<float64_t, 12>, VecT<float64_t, 6>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 8, 64,
                                    VecT<float64_t, 16>, VecT<float64_t, 8>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 8, 96,
                                    VecT<float64_t, 24>, VecT<float64_t, 12>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 8, 128,
                                    VecT<float64_t, 32>, VecT<float64_t, 16>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 8, 192,
                                    VecT<float64_t, 48>, VecT<float64_t, 24>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, float64_t, float64_t, float64_t, 16, 8, 256,
                                    VecT<float64_t, 64>, VecT<float64_t, 32>, VecT<float64_t, 4>, VecT<float64_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 16, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 16, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 16, 4,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 16, 6,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 16, 8,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 16, 12,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 16, 16,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 16, 24,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 16, 32,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 16, 48,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 16, 64,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 16, 96,
                                    VecT<float32_t, 24>, VecT<float32_t, 24>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 16, 128,
                                    VecT<float32_t, 32>, VecT<float32_t, 32>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 16, 192,
                                    VecT<float32_t, 48>, VecT<float32_t, 48>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 16, 256,
                                    VecT<float32_t, 64>, VecT<float32_t, 64>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 32, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 32, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 32, 4,
                                    VecT<float32_t, 1>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 32, 6,
                                    VecT<float32_t, 1>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 32, 8,
                                    VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 32, 12,
                                    VecT<float32_t, 3>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 32, 16,
                                    VecT<float32_t, 4>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 32, 24,
                                    VecT<float32_t, 6>, VecT<float32_t, 12>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 32, 32,
                                    VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 32, 48,
                                    VecT<float32_t, 12>, VecT<float32_t, 24>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 32, 64,
                                    VecT<float32_t, 16>, VecT<float32_t, 32>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 32, 96,
                                    VecT<float32_t, 24>, VecT<float32_t, 48>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 32, 128,
                                    VecT<float32_t, 32>, VecT<float32_t, 64>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 32, 192,
                                    VecT<float32_t, 48>, VecT<float32_t, 96>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 32, 256,
                                    VecT<float32_t, 64>, VecT<float32_t, 128>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 16, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 16, 2,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 16, 3,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 16, 4,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 16, 6,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 16, 8,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 16, 12,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 16, 16,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 16, 24,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 16, 32,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 16, 48,
                                    VecT<float32_t, 24>, VecT<float32_t, 12>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 16, 64,
                                    VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 16, 96,
                                    VecT<float32_t, 48>, VecT<float32_t, 24>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 16, 128,
                                    VecT<float32_t, 64>, VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 16, 192,
                                    VecT<float32_t, 96>, VecT<float32_t, 48>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 16, 256,
                                    VecT<float32_t, 128>, VecT<float32_t, 64>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 32, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 32, 2,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 32, 3,
                                    VecT<float32_t, 1>, VecT<float32_t, 1>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 32, 4,
                                    VecT<float32_t, 2>, VecT<float32_t, 2>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 32, 6,
                                    VecT<float32_t, 3>, VecT<float32_t, 3>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 32, 8,
                                    VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 32, 12,
                                    VecT<float32_t, 6>, VecT<float32_t, 6>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 32, 16,
                                    VecT<float32_t, 8>, VecT<float32_t, 8>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 32, 24,
                                    VecT<float32_t, 12>, VecT<float32_t, 12>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 32, 32,
                                    VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 32, 48,
                                    VecT<float32_t, 24>, VecT<float32_t, 24>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 32, 64,
                                    VecT<float32_t, 32>, VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 32, 96,
                                    VecT<float32_t, 48>, VecT<float32_t, 48>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 32, 128,
                                    VecT<float32_t, 64>, VecT<float32_t, 64>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 32, 192,
                                    VecT<float32_t, 96>, VecT<float32_t, 96>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 32, 32, 256,
                                    VecT<float32_t, 128>, VecT<float32_t, 128>, VecT<float32_t, 16>, VecT<float32_t, 16>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 8, 1,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 8, 2,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 8, 3,
                                    VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 8, 4,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 8, 6,
                                    VecT<float32_t, 1>, VecT<float32_t, 0>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 8, 8,
                                    VecT<float32_t, 2>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 8, 12,
                                    VecT<float32_t, 3>, VecT<float32_t, 1>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 8, 16,
                                    VecT<float32_t, 4>, VecT<float32_t, 2>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 8, 24,
                                    VecT<float32_t, 6>, VecT<float32_t, 3>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 8, 32,
                                    VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 8, 48,
                                    VecT<float32_t, 12>, VecT<float32_t, 6>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 8, 64,
                                    VecT<float32_t, 16>, VecT<float32_t, 8>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 8, 96,
                                    VecT<float32_t, 24>, VecT<float32_t, 12>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 8, 128,
                                    VecT<float32_t, 32>, VecT<float32_t, 16>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 8, 192,
                                    VecT<float32_t, 48>, VecT<float32_t, 24>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
            expect_unsupported_regs<all_targets, xfloat32_t, xfloat32_t, float32_t, 16, 8, 256,
                                    VecT<float32_t, 64>, VecT<float32_t, 32>, VecT<float32_t, 4>, VecT<float32_t, 4>>();
        }
        // clang-format on

        // ---------------------------------------------------------------------------
        // 5. Selector
        //
        // Per (target, types, M, N): the K chosen by the production MmaSelector for
        // BlockKTest = 1, 2, 4, ..., 256 (in that order). Types, M and N are never
        // changed by the selector. Support of the selected op follows from sections 2/3.
        // ---------------------------------------------------------------------------
        // clang-format off
        inline void selector()
        {
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, bfloat8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float16_t, float16_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float16_t, float16_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float16_t, float16_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float16_t, float16_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float16_t, float16_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float16_t, float16_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float16_t, float16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float16_t, float16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float16_t, float16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float16_t, float16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float32_t, float32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float32_t, float32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float32_t, float32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float32_t, float32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float32_t, float32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, int8_t, int8_t, int32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, int8_t, int8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, int8_t, int8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, int8_t, int8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, int8_t, int8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, uint8_t, uint8_t, int32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, uint8_t, uint8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, uint8_t, uint8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, uint8_t, uint8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, uint8_t, uint8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, uint8_t, uint8_t, uint32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, uint8_t, uint8_t, uint32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, uint8_t, uint8_t, uint32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, uint8_t, uint8_t, uint32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, uint8_t, uint8_t, uint32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float64_t, float64_t, float64_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float64_t, float64_t, float64_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float64_t, float64_t, float64_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float64_t, float64_t, float64_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, float64_t, float64_t, float64_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX908, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, bfloat8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float16_t, float16_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float16_t, float16_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float16_t, float16_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float16_t, float16_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float16_t, float16_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float16_t, float16_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float16_t, float16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float16_t, float16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float16_t, float16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float16_t, float16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float32_t, float32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float32_t, float32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float32_t, float32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float32_t, float32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float32_t, float32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, int8_t, int8_t, int32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, int8_t, int8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, int8_t, int8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, int8_t, int8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, int8_t, int8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, uint8_t, uint8_t, int32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, uint8_t, uint8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, uint8_t, uint8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, uint8_t, uint8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, uint8_t, uint8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, uint8_t, uint8_t, uint32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, uint8_t, uint8_t, uint32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, uint8_t, uint8_t, uint32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, uint8_t, uint8_t, uint32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, uint8_t, uint8_t, uint32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float64_t, float64_t, float64_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float64_t, float64_t, float64_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float64_t, float64_t, float64_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float64_t, float64_t, float64_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, float64_t, float64_t, float64_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX90A, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, bfloat8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float16_t, float16_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float16_t, float16_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float16_t, float16_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float16_t, float16_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float16_t, float16_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float16_t, float16_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float16_t, float16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float16_t, float16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float16_t, float16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float16_t, float16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float32_t, float32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float32_t, float32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float32_t, float32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float32_t, float32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float32_t, float32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, int8_t, int8_t, int32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, int8_t, int8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, int8_t, int8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, int8_t, int8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, int8_t, int8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, uint8_t, uint8_t, int32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, uint8_t, uint8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, uint8_t, uint8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, uint8_t, uint8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, uint8_t, uint8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, uint8_t, uint8_t, uint32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, uint8_t, uint8_t, uint32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, uint8_t, uint8_t, uint32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, uint8_t, uint8_t, uint32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, uint8_t, uint8_t, uint32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float64_t, float64_t, float64_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float64_t, float64_t, float64_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float64_t, float64_t, float64_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float64_t, float64_t, float64_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, float64_t, float64_t, float64_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX942, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, bfloat8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float16_t, float16_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float16_t, float16_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float16_t, float16_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float16_t, float16_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float16_t, float16_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float16_t, float16_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float16_t, float16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float16_t, float16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float16_t, float16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float16_t, float16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float32_t, float32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float32_t, float32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float32_t, float32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float32_t, float32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float32_t, float32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, int8_t, int8_t, int32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, int8_t, int8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, int8_t, int8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, int8_t, int8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, int8_t, int8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, uint8_t, uint8_t, int32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, uint8_t, uint8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, uint8_t, uint8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, uint8_t, uint8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, uint8_t, uint8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, uint8_t, uint8_t, uint32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, uint8_t, uint8_t, uint32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, uint8_t, uint8_t, uint32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, uint8_t, uint8_t, uint32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, uint8_t, uint8_t, uint32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float64_t, float64_t, float64_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float64_t, float64_t, float64_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float64_t, float64_t, float64_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float64_t, float64_t, float64_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, float64_t, float64_t, float64_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX950, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float16_t, float16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float32_t, float32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float32_t, float32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float32_t, float32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float32_t, float32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float32_t, float32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, int8_t, int8_t, int32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, int8_t, int8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, int8_t, int8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, int8_t, int8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, int8_t, int8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, uint8_t, uint8_t, int32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, uint8_t, uint8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, uint8_t, uint8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, uint8_t, uint8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, uint8_t, uint8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, uint8_t, uint8_t, uint32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, uint8_t, uint8_t, uint32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, uint8_t, uint8_t, uint32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, uint8_t, uint8_t, uint32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, uint8_t, uint8_t, uint32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float64_t, float64_t, float64_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float64_t, float64_t, float64_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float64_t, float64_t, float64_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float64_t, float64_t, float64_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, float64_t, float64_t, float64_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1100, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float16_t, float16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float32_t, float32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float32_t, float32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float32_t, float32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float32_t, float32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float32_t, float32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, int8_t, int8_t, int32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, int8_t, int8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, int8_t, int8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, int8_t, int8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, int8_t, int8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, uint8_t, uint8_t, int32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, uint8_t, uint8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, uint8_t, uint8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, uint8_t, uint8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, uint8_t, uint8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, uint8_t, uint8_t, uint32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, uint8_t, uint8_t, uint32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, uint8_t, uint8_t, uint32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, uint8_t, uint8_t, uint32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, uint8_t, uint8_t, uint32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float64_t, float64_t, float64_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float64_t, float64_t, float64_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float64_t, float64_t, float64_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float64_t, float64_t, float64_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, float64_t, float64_t, float64_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1101, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float16_t, float16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float32_t, float32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float32_t, float32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float32_t, float32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float32_t, float32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float32_t, float32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, int8_t, int8_t, int32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, int8_t, int8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, int8_t, int8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, int8_t, int8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, int8_t, int8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, uint8_t, uint8_t, int32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, uint8_t, uint8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, uint8_t, uint8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, uint8_t, uint8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, uint8_t, uint8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, uint8_t, uint8_t, uint32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, uint8_t, uint8_t, uint32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, uint8_t, uint8_t, uint32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, uint8_t, uint8_t, uint32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, uint8_t, uint8_t, uint32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float64_t, float64_t, float64_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float64_t, float64_t, float64_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float64_t, float64_t, float64_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float64_t, float64_t, float64_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, float64_t, float64_t, float64_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1102, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float16_t, float16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float32_t, float32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float32_t, float32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float32_t, float32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float32_t, float32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float32_t, float32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, int8_t, int8_t, int32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, int8_t, int8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, int8_t, int8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, int8_t, int8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, int8_t, int8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, uint8_t, uint8_t, int32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, uint8_t, uint8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, uint8_t, uint8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, uint8_t, uint8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, uint8_t, uint8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, uint8_t, uint8_t, uint32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, uint8_t, uint8_t, uint32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, uint8_t, uint8_t, uint32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, uint8_t, uint8_t, uint32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, uint8_t, uint8_t, uint32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float64_t, float64_t, float64_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float64_t, float64_t, float64_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float64_t, float64_t, float64_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float64_t, float64_t, float64_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, float64_t, float64_t, float64_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1103, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float16_t, float16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float32_t, float32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float32_t, float32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float32_t, float32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float32_t, float32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float32_t, float32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, int8_t, int8_t, int32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, int8_t, int8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, int8_t, int8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, int8_t, int8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, int8_t, int8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, uint8_t, uint8_t, int32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, uint8_t, uint8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, uint8_t, uint8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, uint8_t, uint8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, uint8_t, uint8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, uint8_t, uint8_t, uint32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, uint8_t, uint8_t, uint32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, uint8_t, uint8_t, uint32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, uint8_t, uint8_t, uint32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, uint8_t, uint8_t, uint32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float64_t, float64_t, float64_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float64_t, float64_t, float64_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float64_t, float64_t, float64_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float64_t, float64_t, float64_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, float64_t, float64_t, float64_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1150, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float16_t, float16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float32_t, float32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float32_t, float32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float32_t, float32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float32_t, float32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float32_t, float32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, int8_t, int8_t, int32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, int8_t, int8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, int8_t, int8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, int8_t, int8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, int8_t, int8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, uint8_t, uint8_t, int32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, uint8_t, uint8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, uint8_t, uint8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, uint8_t, uint8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, uint8_t, uint8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, uint8_t, uint8_t, uint32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, uint8_t, uint8_t, uint32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, uint8_t, uint8_t, uint32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, uint8_t, uint8_t, uint32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, uint8_t, uint8_t, uint32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float64_t, float64_t, float64_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float64_t, float64_t, float64_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float64_t, float64_t, float64_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float64_t, float64_t, float64_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, float64_t, float64_t, float64_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1151, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float16_t, float16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float32_t, float32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float32_t, float32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float32_t, float32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float32_t, float32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float32_t, float32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, int8_t, int8_t, int32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, int8_t, int8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, int8_t, int8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, int8_t, int8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, int8_t, int8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, uint8_t, uint8_t, int32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, uint8_t, uint8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, uint8_t, uint8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, uint8_t, uint8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, uint8_t, uint8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, uint8_t, uint8_t, uint32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, uint8_t, uint8_t, uint32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, uint8_t, uint8_t, uint32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, uint8_t, uint8_t, uint32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, uint8_t, uint8_t, uint32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float64_t, float64_t, float64_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float64_t, float64_t, float64_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float64_t, float64_t, float64_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float64_t, float64_t, float64_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, float64_t, float64_t, float64_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1152, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float16_t, float16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float32_t, float32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float32_t, float32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float32_t, float32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float32_t, float32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float32_t, float32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, int8_t, int8_t, int32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, int8_t, int8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, int8_t, int8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, int8_t, int8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, int8_t, int8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, uint8_t, uint8_t, int32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, uint8_t, uint8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, uint8_t, uint8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, uint8_t, uint8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, uint8_t, uint8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, uint8_t, uint8_t, uint32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, uint8_t, uint8_t, uint32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, uint8_t, uint8_t, uint32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, uint8_t, uint8_t, uint32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, uint8_t, uint8_t, uint32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float64_t, float64_t, float64_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float64_t, float64_t, float64_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float64_t, float64_t, float64_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float64_t, float64_t, float64_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, float64_t, float64_t, float64_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1153, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float16_t, float16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float32_t, float32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float32_t, float32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float32_t, float32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float32_t, float32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float32_t, float32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, int8_t, int8_t, int32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, int8_t, int8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, int8_t, int8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, int8_t, int8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, int8_t, int8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, uint8_t, uint8_t, int32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, uint8_t, uint8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, uint8_t, uint8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, uint8_t, uint8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, uint8_t, uint8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, uint8_t, uint8_t, uint32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, uint8_t, uint8_t, uint32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, uint8_t, uint8_t, uint32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, uint8_t, uint8_t, uint32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, uint8_t, uint8_t, uint32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float64_t, float64_t, float64_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float64_t, float64_t, float64_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float64_t, float64_t, float64_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float64_t, float64_t, float64_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, float64_t, float64_t, float64_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1170, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float16_t, float16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float32_t, float32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float32_t, float32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float32_t, float32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float32_t, float32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float32_t, float32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, int8_t, int8_t, int32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, int8_t, int8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, int8_t, int8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, int8_t, int8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, int8_t, int8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, uint8_t, uint8_t, int32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, uint8_t, uint8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, uint8_t, uint8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, uint8_t, uint8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, uint8_t, uint8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, uint8_t, uint8_t, uint32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, uint8_t, uint8_t, uint32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, uint8_t, uint8_t, uint32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, uint8_t, uint8_t, uint32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, uint8_t, uint8_t, uint32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float64_t, float64_t, float64_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float64_t, float64_t, float64_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float64_t, float64_t, float64_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float64_t, float64_t, float64_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, float64_t, float64_t, float64_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1171, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float16_t, float16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float32_t, float32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float32_t, float32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float32_t, float32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float32_t, float32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float32_t, float32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, int8_t, int8_t, int32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, int8_t, int8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, int8_t, int8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, int8_t, int8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, int8_t, int8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, uint8_t, uint8_t, int32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, uint8_t, uint8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, uint8_t, uint8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, uint8_t, uint8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, uint8_t, uint8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, uint8_t, uint8_t, uint32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, uint8_t, uint8_t, uint32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, uint8_t, uint8_t, uint32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, uint8_t, uint8_t, uint32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, uint8_t, uint8_t, uint32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float64_t, float64_t, float64_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float64_t, float64_t, float64_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float64_t, float64_t, float64_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float64_t, float64_t, float64_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, float64_t, float64_t, float64_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1200, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float16_t, float16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float32_t, float32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float32_t, float32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float32_t, float32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float32_t, float32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float32_t, float32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, int8_t, int8_t, int32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, int8_t, int8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, int8_t, int8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, int8_t, int8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, int8_t, int8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, uint8_t, uint8_t, int32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 16, 16, 16, 16>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, uint8_t, uint8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, uint8_t, uint8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, uint8_t, uint8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, uint8_t, uint8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, uint8_t, uint8_t, uint32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, uint8_t, uint8_t, uint32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, uint8_t, uint8_t, uint32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, uint8_t, uint8_t, uint32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, uint8_t, uint8_t, uint32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float64_t, float64_t, float64_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float64_t, float64_t, float64_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float64_t, float64_t, float64_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float64_t, float64_t, float64_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, float64_t, float64_t, float64_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1201, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 32, 32, 32, 32>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 32, 32, 32, 32>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 16, 32, 64, 128, 128>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 32, 64, 128, 128>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 16, 32, 64, 128, 128>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 16, 32, 64, 128, 128>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 32, 32, 32, 32>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 32, 32, 32, 32>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float16_t, float16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float32_t, float32_t, float32_t, 16, 16, k_seq<1, 2, 4, 4, 4, 4, 4, 4, 4>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float32_t, float32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float32_t, float32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float32_t, float32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float32_t, float32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 16, 32, 64, 128, 128>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 16, 32, 64, 128, 128>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 16, 32, 64, 128, 128>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 32, 64, 128, 128>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_seq<1, 1, 4, 8, 16, 32, 32, 32, 32>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_seq<1, 1, 4, 8, 16, 32, 32, 32, 32>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, int8_t, int8_t, int32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 32, 64, 64, 64>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, int8_t, int8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, int8_t, int8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, int8_t, int8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, int8_t, int8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, uint8_t, uint8_t, int32_t, 16, 16, k_seq<1, 1, 1, 8, 16, 32, 64, 64, 64>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, uint8_t, uint8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, uint8_t, uint8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, uint8_t, uint8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, uint8_t, uint8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, uint8_t, uint8_t, uint32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, uint8_t, uint8_t, uint32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, uint8_t, uint8_t, uint32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, uint8_t, uint8_t, uint32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, uint8_t, uint8_t, uint32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float64_t, float64_t, float64_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float64_t, float64_t, float64_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float64_t, float64_t, float64_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float64_t, float64_t, float64_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, float64_t, float64_t, float64_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_GFX1250, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat16_t, bfloat16_t, bfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat16_t, bfloat16_t, bfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat16_t, bfloat16_t, bfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat16_t, bfloat16_t, bfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat16_t, bfloat16_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat16_t, bfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat16_t, bfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat16_t, bfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat16_t, bfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, bfloat8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float16_t, float16_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float16_t, float16_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float16_t, float16_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float16_t, float16_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float16_t, float16_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float16_t, float16_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float16_t, float16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float16_t, float16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float16_t, float16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float16_t, float16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float32_t, float32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float32_t, float32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float32_t, float32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float32_t, float32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float32_t, float32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, bfloat8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, bfloat8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, bfloat8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, bfloat8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, bfloat8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, bfloat8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, bfloat8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, bfloat8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, bfloat8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, bfloat8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, float8_t, float16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, float8_t, float16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, float8_t, float16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, float8_t, float16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, float8_t, float16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, float8_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, float8_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, float8_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, float8_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float8_t, float8_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, hfloat16_t, hfloat16_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, hfloat16_t, hfloat16_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, hfloat16_t, hfloat16_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, hfloat16_t, hfloat16_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, hfloat16_t, hfloat16_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, hfloat16_t, hfloat16_t, hfloat16_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, hfloat16_t, hfloat16_t, hfloat16_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, hfloat16_t, hfloat16_t, hfloat16_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, hfloat16_t, hfloat16_t, hfloat16_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, hfloat16_t, hfloat16_t, hfloat16_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, int8_t, int8_t, int32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, int8_t, int8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, int8_t, int8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, int8_t, int8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, int8_t, int8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, uint8_t, uint8_t, int32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, uint8_t, uint8_t, int32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, uint8_t, uint8_t, int32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, uint8_t, uint8_t, int32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, uint8_t, uint8_t, int32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, uint8_t, uint8_t, uint32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, uint8_t, uint8_t, uint32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, uint8_t, uint8_t, uint32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, uint8_t, uint8_t, uint32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, uint8_t, uint8_t, uint32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float64_t, float64_t, float64_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float64_t, float64_t, float64_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float64_t, float64_t, float64_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float64_t, float64_t, float64_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, float64_t, float64_t, float64_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, xfloat32_t, xfloat32_t, float32_t, 16, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, xfloat32_t, xfloat32_t, float32_t, 16, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, xfloat32_t, xfloat32_t, float32_t, 32, 16, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, xfloat32_t, xfloat32_t, float32_t, 32, 32, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
            expect_selected_k<Constants::AMDGCN_ARCH_ID_NONE, xfloat32_t, xfloat32_t, float32_t, 16, 8, k_seq<1, 1, 1, 1, 1, 1, 1, 1, 1>>();
        }
        // clang-format on

        // ---------------------------------------------------------------------------
        // 6. Enabler
        //
        // Per target with any supported key, for one representative supported key:
        // the Enabler = void form is the defaulted form and is supported; any other
        // Enabler (int) selects the unsupported primary template and its registers.
        // ---------------------------------------------------------------------------
        // clang-format off
        inline void enabler()
        {
            expect_enabler<Constants::AMDGCN_ARCH_ID_GFX1100, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_enabler<Constants::AMDGCN_ARCH_ID_GFX1101, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_enabler<Constants::AMDGCN_ARCH_ID_GFX1102, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_enabler<Constants::AMDGCN_ARCH_ID_GFX1103, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_enabler<Constants::AMDGCN_ARCH_ID_GFX1150, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_enabler<Constants::AMDGCN_ARCH_ID_GFX1151, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_enabler<Constants::AMDGCN_ARCH_ID_GFX1152, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_enabler<Constants::AMDGCN_ARCH_ID_GFX1153, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_enabler<Constants::AMDGCN_ARCH_ID_GFX1170, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_enabler<Constants::AMDGCN_ARCH_ID_GFX1171, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_enabler<Constants::AMDGCN_ARCH_ID_GFX1200, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_enabler<Constants::AMDGCN_ARCH_ID_GFX1201, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
            expect_enabler<Constants::AMDGCN_ARCH_ID_GFX1250, bfloat16_t, bfloat16_t, bfloat16_t, 16, 16, 4,
                            VecT<float32_t, 0>, VecT<float32_t, 0>, VecT<float32_t, 2>, VecT<float32_t, 2>>();
        }
        // clang-format on

        // ---------------------------------------------------------------------------
        // 7. Ill-formed probes
        // ---------------------------------------------------------------------------
        // No probe above is excluded: every (target, tuple, shape, K) and every
        // selector chain compiled when the expectations were recorded.
        // The following register types named in the expectations are incomplete
        // (VecT<T, N> with N == 0 or an unregistered N). They can be named and
        // compared but not instantiated; the expectations only compare types.
        //   VecT<float32_t, 0>  (named in 850 register positions)
        //   VecT<float32_t, 12>  (named in 152 register positions)
        //   VecT<float32_t, 24>  (named in 106 register positions)
        //   VecT<float32_t, 48>  (named in 42 register positions)
        //   VecT<float32_t, 6>  (named in 160 register positions)
        //   VecT<float32_t, 96>  (named in 8 register positions)
        //   VecT<float64_t, 0>  (named in 24 register positions)
        //   VecT<float64_t, 12>  (named in 10 register positions)
        //   VecT<float64_t, 24>  (named in 10 register positions)
        //   VecT<float64_t, 48>  (named in 9 register positions)
        //   VecT<float64_t, 6>  (named in 10 register positions)
        //   VecT<float64_t, 96>  (named in 4 register positions)
        //   VecT<int32_t, 0>  (named in 64 register positions)
        //   VecT<int32_t, 12>  (named in 9 register positions)
        //   VecT<int32_t, 24>  (named in 4 register positions)
        //   VecT<int32_t, 6>  (named in 10 register positions)
        //   VecT<uint32_t, 0>  (named in 128 register positions)
        //   VecT<uint32_t, 12>  (named in 18 register positions)
        //   VecT<uint32_t, 24>  (named in 8 register positions)
        //   VecT<uint32_t, 6>  (named in 20 register positions)

    } // namespace wmma_characterization
} // namespace rocwmma

#endif // ROCWMMA_TEST_REGRESSION_WMMA_CHARACTERIZATION_HPP
