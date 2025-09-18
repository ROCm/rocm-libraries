// Copyright (C) 2022 - 2023 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <functional>
#include <iostream>
#include <thread>

using namespace std::placeholders;

#include "../../shared/concurrency.h"
#include "../../shared/environment.h"
#include "../../shared/work_queue.h"
#include "function_pool.h"
#include "rtc_cache.h"
#include "rtc_realcomplex_gen.h"
#include "rtc_stockham_gen.h"
#include "rtc_twiddle_gen.h"
#include "solution_map.h"

#include "device/kernel-generator-embed.h"

#if __has_include(<filesystem>)
#include <filesystem>
#else
#include <experimental/filesystem>
namespace std
{
    namespace filesystem = experimental::filesystem;
}
#endif
namespace fs = std::filesystem;

struct WorkItem
{
    std::string      kernel_name;
    kernel_src_gen_t generate_src;
    std::string      sol_arch_name;
};
typedef WorkQueue<WorkItem> CompileQueue;

// call supplied function with exploded out combinations of
// direction, placement, array types, unitstride-ness, callbacks
void stockham_combo(ComputeScheme                     scheme,
                    FFTKernel                         kernel,
                    std::function<void(int,
                                       rocfft_result_placement,
                                       rocfft_array_type,
                                       rocfft_array_type,
                                       EmbeddedType,
                                       SBRC_TRANSPOSE_TYPE,
                                       DirectRegType,
                                       IntrinsicAccessType,
                                       int,
                                       int,
                                       bool,
                                       CallbackType)> func)
{
    // unit stride is the common case, default to that
    std::vector<bool>                    unitstride_range = {true};
    std::vector<rocfft_result_placement> placements       = {rocfft_placement_notinplace};
    std::vector<EmbeddedType>            ebtypes          = {EmbeddedType::NONE};
    std::vector<SBRC_TRANSPOSE_TYPE>     sbrc_trans_types = {SBRC_TRANSPOSE_TYPE::NONE};
    std::vector<DirectRegType>           dir_reg_types    = {FORCE_OFF_OR_NOT_SUPPORT};
    std::vector<IntrinsicAccessType>     intrinsic_modes  = {DISABLE_BOTH};
    // SBCC can be used with or without large twd.  Large twd may be
    // base 4, 5, 6, 8.  Base 4 is unused since it's only useful up
    // to 4k lengths, which we already have single kernels for.  Base
    // 8 can be 2 or 3 steps; other bases are always 3 step.
    std::vector<std::array<int, 2>> base_steps = {{0, 0}, {5, 3}, {6, 3}, {8, 2}, {8, 3}};

    switch(scheme)
    {
    case CS_KERNEL_STOCKHAM_BLOCK_CC:
    {
        placements.push_back(rocfft_placement_inplace);
        // SBCC is never unit stride
        unitstride_range = {false};

        // if no dir-to-reg support, then we don't have intrinsic buffer RW,
        // and only force_off_or_not_support for dir2reg. Else, we have all possibilities
        // (even force_off_or_not_support is still included)
        if(kernel.direct_to_from_reg)
        {
            dir_reg_types.push_back(DirectRegType::TRY_ENABLE_IF_SUPPORT);
            intrinsic_modes.push_back(IntrinsicAccessType::ENABLE_BOTH);
            intrinsic_modes.push_back(IntrinsicAccessType::ENABLE_LOAD_ONLY);
        }
        break;
    }
    case CS_KERNEL_STOCKHAM_BLOCK_CR:
    {
        base_steps.resize(1);
        // SBCR is never unit stride
        unitstride_range = {false};
        ebtypes.push_back(EmbeddedType::C2Real_PRE);

        // if no dir-to-reg support, then we don't have intrinsic
        // buffer RW, and only force_off_or_not_support for
        // dir2reg. Else, SBCR only allows "both" or "none" for
        // intrinsic.
        if(kernel.direct_to_from_reg)
        {
            dir_reg_types.push_back(DirectRegType::TRY_ENABLE_IF_SUPPORT);
            intrinsic_modes.push_back(IntrinsicAccessType::ENABLE_BOTH);
        }
        break;
    }
    case CS_KERNEL_STOCKHAM_BLOCK_RC:
    case CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z:
    case CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY:
    case CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY:
    {
        // SBRC allows direct to/from reg but does not support
        // intrinsic
        if(kernel.direct_to_from_reg)
            dir_reg_types.push_back(DirectRegType::TRY_ENABLE_IF_SUPPORT);

        // SBRC is never unit stride
        unitstride_range = {false};

        base_steps.resize(1);
        // All SBRCs have ALIGNED and UNALIGNED, but no NONE
        sbrc_trans_types.erase(sbrc_trans_types.begin());
        sbrc_trans_types.push_back(SBRC_TRANSPOSE_TYPE::TILE_ALIGNED);
        sbrc_trans_types.push_back(SBRC_TRANSPOSE_TYPE::TILE_UNALIGNED);
        // Finish SBRC-2D and SBRC-3D-ERC without DIAGONAL
        if(scheme == CS_KERNEL_STOCKHAM_BLOCK_RC
           || scheme == CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY)
            break;
        // DIAGONAL Transpose
        sbrc_trans_types.push_back(SBRC_TRANSPOSE_TYPE::DIAGONAL);

        break;
    }
    case CS_KERNEL_STOCKHAM:
    {
        base_steps.resize(1);
        unitstride_range = {true, false};
        placements.push_back(rocfft_placement_inplace);
        ebtypes.push_back(EmbeddedType::Real2C_POST);
        ebtypes.push_back(EmbeddedType::C2Real_PRE);

        // SBRR always uses TRY_ENABLE_IF_SUPPORT, and does not
        // support intrinsic parameter.
        dir_reg_types = {DirectRegType::TRY_ENABLE_IF_SUPPORT};
        break;
    }
    default:
        // throw std::runtime_error("unsupported scheme in stockham_combo aot_rtc");
        // since it is not possible that we are here,
        // so directly return is fine which means do nothing
        return;
    }

    for(auto direction : {-1, 1})
    {
        for(auto placement : placements)
        {
            for(auto inArrayType : {rocfft_array_type_complex_interleaved})
            {
                for(auto outArrayType : {rocfft_array_type_complex_interleaved})
                {
                    // inplace requires same array types
                    if(placement == rocfft_placement_inplace && inArrayType != outArrayType)
                        continue;
                    for(auto unitstride : unitstride_range)
                    {
                        for(auto base_step : base_steps)
                        {
                            for(auto ebtype : ebtypes)
                            {
                                for(auto sbrc_trans_type : sbrc_trans_types)
                                {
                                    for(auto dir_reg_type : dir_reg_types)
                                    {
                                        for(auto intrinsic : intrinsic_modes)
                                        {
                                            for(auto cbtype :
                                                {CallbackType::NONE, CallbackType::USER_LOAD_STORE})
                                            {
                                                func(direction,
                                                     placement,
                                                     inArrayType,
                                                     outArrayType,
                                                     ebtype,
                                                     sbrc_trans_type,
                                                     dir_reg_type,
                                                     intrinsic,
                                                     base_step[0],
                                                     base_step[1],
                                                     unitstride,
                                                     cbtype);
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void build_stockham_function_pool(CompileQueue& queue)
{
    // build everything in the 64k LDS function pool
    function_pool fp(65536);

    // fused Bluestein and partial-pass kernels are always built at runtime
    auto fuseBlue = BluesteinFuseType::BFT_NONE;
    auto ppType   = PartialPassType::PPT_NONE;
    auto ppParams = StockhamPartialPassParams();

    for(const auto& i : fp.get_map())
    {
        // we only want to compile kernels explicitly marked for AOT RTC
        if(!i.second.aot_rtc)
            continue;

        auto length1D = i.first.lengths[0];
        // auto length2D            = i.first.lengths[1];
        auto                      precision = i.first.precision;
        auto                      scheme    = i.first.scheme;
        std::vector<unsigned int> factors;
        std::copy(i.second.factors.begin(), i.second.factors.end(), std::back_inserter(factors));

        StockhamGeneratorSpecs specs{factors,
                                     {},
                                     {static_cast<unsigned int>(precision)},
                                     static_cast<unsigned int>(i.second.workgroup_size),
                                     PrintScheme(scheme)};
        specs.threads_per_transform = i.second.threads_per_transform[0];
        specs.half_lds              = i.second.half_lds;
        specs.direct_to_from_reg    = i.second.direct_to_from_reg;

        stockham_combo(
            scheme,
            i.second,
            [=, &queue, &specs](int                     direction,
                                rocfft_result_placement placement,
                                rocfft_array_type       inArrayType,
                                rocfft_array_type       outArrayType,
                                EmbeddedType            ebtype,
                                SBRC_TRANSPOSE_TYPE     sbrc_trans_type,
                                DirectRegType           dir_reg_type,
                                IntrinsicAccessType     intrinsic,
                                int                     ltwd_base,
                                int                     ltwd_step,
                                bool                    unitstride,
                                CallbackType            cbtype) {
                // intrinsic mode require non-callback and enable dir_reg
                if((cbtype != CallbackType::NONE || dir_reg_type == FORCE_OFF_OR_NOT_SUPPORT)
                   && (intrinsic != IntrinsicAccessType::DISABLE_BOTH))
                    return;

                // for SBRR kernels, only do embedded pre/post processing for
                // unit-stride, since that's all we'd use it for.
                //
                // we currently also only do this processing for even lengths.
                if(ebtype != EmbeddedType::NONE)
                {
                    if((scheme == CS_KERNEL_STOCKHAM && !unitstride) || length1D % 2 != 0)
                        return;
                }
                specs.ebtype = ebtype;

                auto kernel_name = stockham_rtc_kernel_name(specs,
                                                            specs,
                                                            scheme,
                                                            direction,
                                                            precision,
                                                            placement,
                                                            inArrayType,
                                                            outArrayType,
                                                            unitstride,
                                                            ltwd_base,
                                                            ltwd_step,
                                                            false,
                                                            dir_reg_type,
                                                            intrinsic,
                                                            sbrc_trans_type,
                                                            cbtype,
                                                            fuseBlue,
                                                            ppType,
                                                            ppParams,
                                                            {},
                                                            {});
                std::function<std::string(const std::string&)> generate_src
                    = [=](const std::string& kernel_name) -> std::string {
                    return stockham_rtc(specs,
                                        specs,
                                        ppParams,
                                        nullptr,
                                        kernel_name,
                                        scheme,
                                        direction,
                                        precision,
                                        placement,
                                        inArrayType,
                                        outArrayType,
                                        unitstride,
                                        ltwd_base,
                                        ltwd_step,
                                        false,
                                        dir_reg_type,
                                        intrinsic,
                                        sbrc_trans_type,
                                        cbtype,
                                        fuseBlue,
                                        ppType,
                                        {},
                                        {});
                };
                queue.push({kernel_name, generate_src, ""});
            });
    }
}

void build_realcomplex(CompileQueue& queue)
{
    for(auto precision : {rocfft_precision_single, rocfft_precision_double})
    {
        for(bool planar : {true, false})
        {
            // build even-length kernels, as they're commonly used
            for(auto scheme : {CS_KERNEL_R_TO_CMPLX, CS_KERNEL_CMPLX_TO_R})
            {
                for(size_t dim : {1, 2, 3})
                {
                    for(bool Ndiv4 : {true, false})
                    {
                        // standalone even-length kernels may be
                        // first/last in the plan, so allow for
                        // callbacks
                        for(auto cbtype : {CallbackType::NONE, CallbackType::USER_LOAD_STORE})
                        {
                            // r2c may have planar output, c2r may have planar input
                            auto inArrayType  = (scheme == CS_KERNEL_CMPLX_TO_R && planar)
                                                    ? rocfft_array_type_complex_planar
                                                    : rocfft_array_type_complex_interleaved;
                            auto outArrayType = (scheme == CS_KERNEL_R_TO_CMPLX && planar)
                                                    ? rocfft_array_type_complex_planar
                                                    : rocfft_array_type_complex_interleaved;

                            RealComplexEvenSpecs specs{
                                {scheme, dim, precision, inArrayType, outArrayType, cbtype, {}, {}},
                                Ndiv4};
                            auto kernel_name = realcomplex_even_rtc_kernel_name(specs);
                            std::function<std::string(const std::string&)> generate_src
                                = [=](const std::string& kernel_name) -> std::string {
                                return realcomplex_even_rtc(kernel_name, specs);
                            };
                            queue.push({kernel_name, generate_src});
                        }
                    }
                }
            }
            for(auto scheme : {CS_KERNEL_R_TO_CMPLX_TRANSPOSE, CS_KERNEL_TRANSPOSE_CMPLX_TO_R})
            {
                // r2c may have planar output, c2r may have planar input
                auto inArrayType  = (scheme == CS_KERNEL_TRANSPOSE_CMPLX_TO_R && planar)
                                        ? rocfft_array_type_complex_planar
                                        : rocfft_array_type_complex_interleaved;
                auto outArrayType = (scheme == CS_KERNEL_R_TO_CMPLX_TRANSPOSE && planar)
                                        ? rocfft_array_type_complex_planar
                                        : rocfft_array_type_complex_interleaved;

                RealComplexEvenTransposeSpecs specs{{scheme,
                                                     static_cast<size_t>(1),
                                                     precision,
                                                     inArrayType,
                                                     outArrayType,
                                                     CallbackType::NONE,
                                                     {},
                                                     {}}};
                auto kernel_name = realcomplex_even_transpose_rtc_kernel_name(specs);
                std::function<std::string(const std::string&)> generate_src
                    = [=](const std::string& kernel_name) -> std::string {
                    return realcomplex_even_transpose_rtc(kernel_name, specs);
                };
                queue.push({kernel_name, generate_src, ""});
            }
        }
    }
}

void build_twiddle(CompileQueue& queue)
{
    const auto twiddle_kernel_types = {
        TwiddleTableType::RADICES,
        TwiddleTableType::LENGTH_N,
        TwiddleTableType::HALF_N,
        TwiddleTableType::LARGE,
    };
    for(auto precision : {rocfft_precision_single, rocfft_precision_double})
    {
        for(auto type : twiddle_kernel_types)
        {
            auto kernel_name = twiddle_rtc_kernel_name(type, precision);
            std::function<std::string(const std::string&)> generate_src
                = [=](const std::string& kernel_name) -> std::string {
                return twiddle_rtc(kernel_name, type, precision);
            };
            queue.push({kernel_name, generate_src, ""});
        }
    }
}

void solution_kernel_combo(FMKey                             kernel_key,
                           std::function<void(int,
                                              rocfft_result_placement,
                                              rocfft_array_type,
                                              rocfft_array_type,
                                              EmbeddedType,
                                              SBRC_TRANSPOSE_TYPE,
                                              DirectRegType,
                                              IntrinsicAccessType,
                                              int,
                                              int,
                                              int,
                                              bool,
                                              CallbackType)> func)
{
    std::vector<bool> unitstride_range;

    // we pre-build the kernels with the exact settings if possible
    KernelConfig&     config     = kernel_key.kernel_config;
    EmbeddedType      ebtype     = config.ebType;
    PlacementCode     placement  = config.placement;
    rocfft_array_type iAryType   = config.iAryType;
    rocfft_array_type oAryType   = config.oAryType;
    int               direction  = config.direction;
    int               static_dim = config.static_dim;

    DirectRegType dir_reg_type
        = config.direct_to_from_reg ? TRY_ENABLE_IF_SUPPORT : FORCE_OFF_OR_NOT_SUPPORT;
    IntrinsicAccessType intrinsic_mode = config.intrinsic_buffer_inst ? ENABLE_BOTH : DISABLE_BOTH;

    // SBCC can be used with or without large twd.  Large twd may be
    // base 4, 5, 6, 8.  Base 4 is unused since it's only useful up
    // to 4k lengths, which we already have single kernels for.
    // for use_3steps = FALSE: base is always 8, steps can be 2 or 3
    // for            = TRUE:  base can be 5, 6, steps is always 3
    std::vector<std::array<int, 2>> base_steps = {{0, 0}, {5, 3}, {6, 3}, {8, 2}, {8, 3}};

    // The static_dim value in solution map should be tuned for a known dim.
    // To avoid unused kernels, we should build for the exact one.
    // But if it is 0, which is from old format version,
    // we expand it to all support values (basically, 1,2,3, block compute > 1)
    std::vector<int>                     static_dims_range = {static_dim};
    std::vector<int>                     dir_range         = {direction};
    std::vector<rocfft_result_placement> placement_range;
    if(placement != PC_UNSET)
    {
        placement_range
            = {(placement == PC_IP) ? rocfft_placement_inplace : rocfft_placement_notinplace};
    }

    // a C2C kernel, we bwd and fwd can share the same solution
    if(iAryType != rocfft_array_type_real && oAryType != rocfft_array_type_real)
        dir_range = {-1, 1};

    switch(kernel_key.scheme)
    {
    case CS_KERNEL_STOCKHAM_BLOCK_CC:
    {
        // SBCC is never unit stride
        unitstride_range = {false};
        // placement
        if(placement == PC_UNSET)
        {
            placement_range = {rocfft_placement_inplace, rocfft_placement_notinplace};
        }
        // sbcc can be used in 2D, 3D, for L1D, it's still psuedo-2D
        if(static_dim == 0)
        {
            static_dims_range = {2, 3};
        }
        // depends on use_3steps flag
        if(config.use_3steps_large_twd)
            base_steps = {{5, 3}, {6, 3}};
        else
            base_steps = {{8, 2}, {8, 3}};
        break;
    }
    case CS_KERNEL_STOCKHAM_BLOCK_CR:
    {
        base_steps.resize(1);
        // SBCR is never unit stride
        unitstride_range = {false};
        // placement
        if(placement == PC_UNSET)
        {
            placement_range = {rocfft_placement_notinplace};
        }
        // sbcr now is used in 3D only
        if(static_dim == 0)
        {
            static_dims_range = {3};
        }
        break;
    }
    case CS_KERNEL_STOCKHAM_BLOCK_RC:
    case CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z:
    case CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY:
    {
        base_steps.resize(1);
        unitstride_range = {true, false};
        // placement
        if(placement == PC_UNSET)
        {
            placement_range = {rocfft_placement_notinplace};
        }
        // SBRC can be used in 2D, 3D, but SBRC-with-Transpose are 3D
        if(static_dim == 0)
        {
            if(kernel_key.scheme == CS_KERNEL_STOCKHAM_BLOCK_RC)
                static_dims_range = {2, 3};
            else
                static_dims_range = {3};
        }
        break;
    }
    case CS_KERNEL_STOCKHAM:
    {
        base_steps.resize(1);
        unitstride_range = {true, false};
        // placement
        if(placement == PC_UNSET)
        {
            placement_range = {rocfft_placement_inplace, rocfft_placement_notinplace};
        }
        // Stockham can use in 1D/2D/3D
        if(static_dim == 0)
        {
            static_dims_range = {1, 2, 3};
        }
        break;
    }
    default:
        // throw std::runtime_error("unsupported scheme in stockham_combo aot_rtc");
        // since it is not possible that we are here,
        // so directly return is fine which means do nothing
        return;
    }

    for(auto direction : dir_range)
    {
        for(auto placement : placement_range)
        {
            for(auto inArrayType : {iAryType})
            {
                for(auto outArrayType : {oAryType})
                {
                    if(placement == rocfft_placement_inplace && inArrayType != outArrayType)
                        continue;
                    for(auto unitstride : unitstride_range)
                    {
                        for(auto static_dim : static_dims_range)
                        {
                            for(auto base_step : base_steps)
                            {
                                for(auto callback :
                                    {CallbackType::NONE, CallbackType::USER_LOAD_STORE})
                                {
                                    func(direction,
                                         placement,
                                         inArrayType,
                                         outArrayType,
                                         ebtype,
                                         kernel_key.sbrcTrans,
                                         dir_reg_type,
                                         intrinsic_mode,
                                         static_dim,
                                         base_step[0],
                                         base_step[1],
                                         unitstride,
                                         callback);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void build_solution_kernels(CompileQueue& queue)
{
    // build every kernel in the solution map
    solution_map& solmap = solution_map::get_solution_map();

    std::vector<SolutionNode> kernel_nodes;
    solmap.get_all_kernels(kernel_nodes, true);

    // fused Bluestein and partial-pass kernels are always built at runtime
    auto fuseBlue = BluesteinFuseType::BFT_NONE;
    auto ppType   = PartialPassType::PPT_NONE;
    auto ppParams = StockhamPartialPassParams();

    for(const SolutionNode& kernel_sol : kernel_nodes)
    {
        const std::string&  arch_name  = kernel_sol.arch_name;
        const FMKey&        kernel_key = kernel_sol.kernel_key;
        const KernelConfig& config     = kernel_key.kernel_config;

        // auto length1D = kernel_key.lengths[0];
        // auto length2D = kernel_key.lengths[1];
        auto                      precision = kernel_key.precision;
        auto                      scheme    = kernel_key.scheme;
        std::vector<unsigned int> factors;
        std::copy(config.factors.begin(), config.factors.end(), std::back_inserter(factors));

        solution_kernel_combo(
            kernel_key,
            [=, &queue](int                     direction,
                        rocfft_result_placement placement,
                        rocfft_array_type       inArrayType,
                        rocfft_array_type       outArrayType,
                        EmbeddedType            ebtype,
                        SBRC_TRANSPOSE_TYPE     sbrc_trans_type,
                        DirectRegType           dir_reg_type,
                        IntrinsicAccessType     intrinsic,
                        int                     static_dim,
                        int                     ltwd_base,
                        int                     ltwd_step,
                        bool                    unitstride,
                        CallbackType            cbtype) {
                // callbacks need to disable intrisinc mode
                // so if we are pre-compiling a callbacks + intrinsic, runtime-compilation
                // eventually goes to a callback + non-intrinsic (see stockham_rtc_kernel_name)
                if(cbtype != CallbackType::NONE && (intrinsic != IntrinsicAccessType::DISABLE_BOTH))
                    intrinsic = IntrinsicAccessType::DISABLE_BOTH;

                // same rejection as aot function_pool, but no need to test (length % 2)
                // since the kernel is tuned from real nodes.
                if(ebtype != EmbeddedType::NONE)
                {
                    if((scheme == CS_KERNEL_STOCKHAM && !unitstride)
                       || cbtype != CallbackType::NONE)
                        return;
                }

                StockhamGeneratorSpecs specs{factors,
                                             {},
                                             {static_cast<unsigned int>(precision)},
                                             static_cast<unsigned int>(config.workgroup_size),
                                             PrintScheme(scheme)};
                specs.threads_per_transform = config.threads_per_transform[0];
                specs.half_lds              = config.half_lds;
                specs.direct_to_from_reg    = config.direct_to_from_reg;
                specs.wgs_is_derived        = true;
                // kernel_sol should specify the static_dim, need to set here,
                // so move specs to local instead of captured (need mutable if captured)
                specs.static_dim = static_dim;
                specs.ebtype     = ebtype;

                auto kernel_name = stockham_rtc_kernel_name(specs,
                                                            specs,
                                                            scheme,
                                                            direction,
                                                            precision,
                                                            placement,
                                                            inArrayType,
                                                            outArrayType,
                                                            unitstride,
                                                            ltwd_base,
                                                            ltwd_step,
                                                            false,
                                                            dir_reg_type,
                                                            intrinsic,
                                                            sbrc_trans_type,
                                                            cbtype,
                                                            fuseBlue,
                                                            ppType,
                                                            ppParams,
                                                            {},
                                                            {});

                std::function<std::string(const std::string&)> generate_src
                    = [=](const std::string& kernel_name) -> std::string {
                    return stockham_rtc(specs,
                                        specs,
                                        ppParams,
                                        nullptr,
                                        kernel_name,
                                        scheme,
                                        direction,
                                        precision,
                                        placement,
                                        inArrayType,
                                        outArrayType,
                                        unitstride,
                                        ltwd_base,
                                        ltwd_step,
                                        false,
                                        dir_reg_type,
                                        intrinsic,
                                        sbrc_trans_type,
                                        cbtype,
                                        fuseBlue,
                                        ppType,
                                        {},
                                        {});
                };
                queue.push({kernel_name, generate_src, arch_name});
            });
    }
}

int main(int argc, char** argv)
{
    if(argc < 4)
    {
        puts("Usage: rocfft_aot_helper temp_cachefile.db output_cachefile.db "
             "path/to/rocfft_rtc_helper gfx000 gfx001 ...");
        return 1;
    }

    std::string              temp_cache_file   = argv[1];
    std::string              output_cache_file = argv[2];
    std::string              rtc_helper        = argv[3];
    std::vector<std::string> gpu_archs;
    for(int i = 4; i < argc; ++i)
        gpu_archs.push_back(argv[i]);

    // Default to using a persistent file in the current dir if no
    // cache file was given
    if(temp_cache_file.empty())
    {
        temp_cache_file = "rocfft_temp_cache.db";
    }

    // force RTC to use the temporary cache file
    rocfft_setenv("ROCFFT_RTC_CACHE_PATH", temp_cache_file.c_str());

    // disable system cache since we want to compile everything - use
    // an in-memory DB which will always be empty
    rocfft_setenv("ROCFFT_RTC_SYS_CACHE_PATH", ":memory:");

    // tell RTC where the compile helper is
    rocfft_setenv("ROCFFT_RTC_PROCESS_HELPER", rtc_helper.c_str());

    RTCCache::single = std::make_unique<RTCCache>();

    RTCCache::single->enable_write_mostly();

    CompileQueue queue;

    static const size_t      NUM_THREADS = rocfft_concurrency();
    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);
    for(size_t i = 0; i < NUM_THREADS; ++i)
    {
        threads.emplace_back([&queue, &gpu_archs]() {
            while(true)
            {
                auto item = queue.pop();
                if(item.kernel_name.empty())
                    break;

                for(const auto& gpu_arch : gpu_archs)
                {
                    if(item.sol_arch_name.empty())
                    {
                        RTCCache::cached_compile(
                            item.kernel_name, gpu_arch, item.generate_src, generator_sum());
                    }
                    else if(gpu_arch.find(item.sol_arch_name) != std::string::npos)
                    {
                        // std::cout << "arch: " << gpu_arch
                        //           << ", solution-kernel: " << item.kernel_name << std::endl;
                        RTCCache::cached_compile(
                            item.kernel_name, gpu_arch, item.generate_src, generator_sum());
                    }
                }
            }
        });
    }

    build_stockham_function_pool(queue);
    build_realcomplex(queue);
    build_twiddle(queue);
    build_solution_kernels(queue);

    // signal end of results with empty work items
    for(size_t i = 0; i < NUM_THREADS; ++i)
        queue.push({});
    for(size_t i = 0; i < NUM_THREADS; ++i)
        threads[i].join();

    // write the output file using what we collected in the temporary
    // cache
    RTCCache::single->write_aot_cache(output_cache_file, generator_sum(), gpu_archs);

    // try to shrink the temp cache file to 10 GiB
    try
    {
        RTCCache::single->cleanup_cache(static_cast<sqlite3_int64>(10) * 1024 * 1024 * 1024);
    }
    // the build should still succeed even if we fail to shrink the temp cache
    catch(std::exception&)
    {
        std::cerr << "warning: failed to shrink temp cache" << std::endl;
    }

    RTCCache::single.reset();

    return 0;
}
