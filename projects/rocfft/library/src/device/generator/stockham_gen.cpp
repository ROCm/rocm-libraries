// Copyright (C) 2021 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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
using namespace std::placeholders;

#include "../../../../shared/arithmetic.h"
#include "../../../../shared/device_properties.h"
#include "../../../../shared/precision_type.h"
#include "generator.h"
#include "stockham_gen.h"
#include <algorithm>
#include <array>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "stockham_gen_cc.h"
#include "stockham_gen_cr.h"
#include "stockham_gen_rc.h"
#include "stockham_gen_rr.h"
#include "stockham_pp_gen_cc.h"
#include "stockham_pp_gen_rr.h"

#include "stockham_gen_2d.h"

// this rolls up all the information about the generated launchers,
// enough to genernate the function pool entry
struct GeneratedLauncher
{
    GeneratedLauncher(StockhamKernel&                  kernel,
                      const std::string&               scheme,
                      const std::string&               pp_child_scheme,
                      const unsigned int&              pp_threads_per_transform,
                      const std::vector<unsigned int>& pp_factors_curr,
                      const std::vector<unsigned int>& pp_factors_other,
                      const unsigned int&              pp_current_dim,
                      const unsigned int&              pp_off_dim,
                      const unsigned int&              precision_type,
                      const unsigned int&              transform_type,
                      const std::string&               gcn_arch_name,
                      const std::string&               sbrc_type,
                      const std::string&               sbrc_transpose_type)
        : scheme(scheme)
        , pp_child_scheme(pp_child_scheme)
        , pp_threads_per_transform(pp_threads_per_transform)
        , pp_factors_curr(pp_factors_curr)
        , pp_factors_other(pp_factors_other)
        , pp_current_dim(pp_current_dim)
        , pp_off_dim(pp_off_dim)
        , lengths(kernel.launcher_lengths())
        , factors(kernel.launcher_factors())
        , transforms_per_block(kernel.launcher_transforms_per_block())
        , workgroup_size(kernel.launcher_workgroup_size())
        , half_lds(kernel.half_lds)
        , direct_to_from_reg(kernel.direct_to_from_reg)
        , sbrc_type(sbrc_type)
        , sbrc_transpose_type(sbrc_transpose_type)
        , precision_type(precision_type)
        , transform_type(transform_type)
        , gcn_arch_name(gcn_arch_name)

    {
    }

    std::string               scheme;
    std::string               pp_child_scheme;
    unsigned int              pp_threads_per_transform;
    std::vector<unsigned int> pp_factors_curr;
    std::vector<unsigned int> pp_factors_other;
    unsigned int              pp_current_dim;
    unsigned int              pp_off_dim;
    std::vector<unsigned int> lengths;
    std::vector<unsigned int> factors;

    unsigned int transforms_per_block;
    unsigned int workgroup_size;
    bool         half_lds;
    bool         direct_to_from_reg;

    // SBRC transpose type
    std::string sbrc_type;
    std::string sbrc_transpose_type;

    unsigned int precision_type;
    unsigned int transform_type;

    std::string gcn_arch_name;

    // output a json object that the python generator can parse to know
    // how to build the function pool
    std::string to_string() const
    {
        std::string output = "{";

        const char* OBJ_DELIM = "";
        const char* COMMA     = ",";

        auto quote_str  = [](const std::string& s) { return "\"" + s + "\""; };
        auto add_member = [&](const std::string& key, const std::string& value) {
            output += OBJ_DELIM;
            output += quote_str(key) + " : " + value;
            OBJ_DELIM = COMMA;
        };
        auto vec_to_list = [&](const std::vector<unsigned int>& vec) {
            const char* LIST_DELIM = "";
            std::string list_str   = "[";
            for(auto i : vec)
            {
                list_str += LIST_DELIM;
                list_str += std::to_string(i);
                LIST_DELIM = COMMA;
            }
            list_str += "]";
            return list_str;
        };

        add_member("scheme", quote_str(scheme));
        add_member("factors", vec_to_list(factors));
        add_member("lengths", vec_to_list(lengths));
        add_member("transforms_per_block", std::to_string(transforms_per_block));
        add_member("workgroup_size", std::to_string(workgroup_size));
        add_member("half_lds", half_lds ? "true" : "false");
        add_member("direct_to_from_reg", direct_to_from_reg ? "true" : "false");
        add_member("sbrc_type", quote_str(sbrc_type));
        add_member("sbrc_transpose_type", quote_str(sbrc_transpose_type));
        add_member("precision_type", std::to_string(precision_type));
        add_member("transform_type", std::to_string(transform_type));
        add_member("gcn_arch_name", quote_str(gcn_arch_name));
        add_member("pp_child_scheme", quote_str(pp_child_scheme));
        add_member("pp_threads_per_transform", std::to_string(pp_threads_per_transform));
        add_member("pp_factors_curr", vec_to_list(pp_factors_curr));
        add_member("pp_factors_other", vec_to_list(pp_factors_other));
        add_member("pp_current_dim", std::to_string(pp_current_dim));
        add_member("pp_off_dim", std::to_string(pp_off_dim));

        output += "}";
        return output;
    }
};

struct LaunchSuffix
{
    std::string function_suffix;
    std::string scheme;
    std::string sbrc_type;
    std::string sbrc_transpose_type;
};

void make_launcher(const unsigned int&              precision_type,
                   const unsigned int&              transform_type,
                   const std::vector<LaunchSuffix>& launcher_suffixes,
                   StockhamKernel&                  kernel,
                   const std::string&               gcn_arch_name,
                   const std::string&               pp_child_scheme,
                   const unsigned int&              pp_threads_per_transform,
                   const std::vector<unsigned int>& pp_factors_curr,
                   const std::vector<unsigned int>& pp_factors_other,
                   const unsigned int&              pp_current_dim,
                   const unsigned int&              pp_off_dim,
                   std::vector<GeneratedLauncher>&  generated_launchers)
{

    for(auto&& launcher : launcher_suffixes)
    {
        generated_launchers.emplace_back(kernel,
                                         launcher.scheme,
                                         pp_child_scheme,
                                         pp_threads_per_transform,
                                         pp_factors_curr,
                                         pp_factors_other,
                                         pp_current_dim,
                                         pp_off_dim,
                                         precision_type,
                                         transform_type,
                                         gcn_arch_name,
                                         launcher.sbrc_type,
                                         launcher.sbrc_transpose_type);
    }
}

// parse comma-separated string booleans
std::vector<bool> parse_bool_csv(const std::string& arg)
{
    std::vector<bool> bools;

    size_t prev_pos = 0;
    for(;;)
    {
        auto pos = arg.find(',', prev_pos);
        if(pos == std::string::npos)
        {
            bools.push_back(arg.substr(prev_pos) == "1");
            break;
        }
        bools.push_back(arg.substr(prev_pos, pos - prev_pos) == "1");
        prev_pos = pos + 1;
    }
    return bools;
}

// parse comma-separated string uints
std::vector<unsigned int> parse_uints_csv(const std::string& arg)
{
    std::vector<unsigned int> uints;

    size_t prev_pos = 0;
    for(;;)
    {
        auto pos = arg.find(',', prev_pos);
        if(pos == std::string::npos)
        {
            uints.push_back(std::stoi(arg.substr(prev_pos)));
            break;
        }
        uints.push_back(std::stoi(arg.substr(prev_pos, pos - prev_pos)));
        prev_pos = pos + 1;
    }
    return uints;
}

const char* COMMA = ",";

// output json (via stdout) describing the launchers that were generated, so
// kernel-generator can generate the function pool
void output_json(const std::vector<GeneratedLauncher>& launchers,
                 const std::string&                    kernel_name,
                 std::ostream&                         output)
{
    const char* LIST_DELIM = "";

    // store all variants of one kernel in a list, and store with kernel name as key
    output << "\"" << kernel_name << "\" : ";
    output << "[";
    for(auto& launcher : launchers)
    {
        output << LIST_DELIM;
        output << launcher.to_string() << "\n";
        LIST_DELIM = COMMA;
    }
    output << "]";
}

// Render stockham partial-pass kernel generated launchers in JSON format.
void stockham_partial_pass_variants(const std::string&               kernel_name,
                                    const StockhamGeneratorSpecs&    specs1,
                                    const StockhamGeneratorSpecs&    specs2,
                                    const StockhamPartialPassParams& params_1,
                                    const StockhamPartialPassParams& params_2,
                                    std::ostream&                    output)
{
    std::vector<GeneratedLauncher> launchers;

    if(specs1.scheme == "CS_3D_PP" && specs2.scheme == "CS_3D_PP")
    {
        // SBRR_PP + SBCC_PP
        if(params_1.current_dim == 0 && params_2.current_dim == 2)
        {
            StockhamPartialPassKernelRR kernelRR(specs1, params_1);
            make_launcher(specs1.precision,
                          *specs1.transform_type,
                          {{"pp_stoc", specs1.scheme, "", ""}},
                          kernelRR,
                          specs1.gcn_arch_name,
                          "CS_KERNEL_STOCKHAM_PP",
                          params_1.pp_threads_per_transform,
                          params_1.pp_factors_curr,
                          params_1.pp_factors_other,
                          params_1.current_dim,
                          params_1.off_dim,
                          launchers);

            StockhamPartialPassKernelCC kernelCC(specs2, params_2, false);
            make_launcher(specs2.precision,
                          *specs2.transform_type,
                          {{"pp_sbcc", specs2.scheme, "", ""}},
                          kernelCC,
                          specs2.gcn_arch_name,
                          "CS_KERNEL_STOCKHAM_PP_BLOCK_CC",
                          params_2.pp_threads_per_transform,
                          params_2.pp_factors_curr,
                          params_2.pp_factors_other,
                          params_2.current_dim,
                          params_2.off_dim,
                          launchers);
        }
        else if(params_1.current_dim == 2 && params_2.current_dim == 0)
        {
            StockhamPartialPassKernelRR kernelCC(specs1, params_1);
            make_launcher(specs1.precision,
                          *specs1.transform_type,
                          {{"pp_sbcc", specs1.scheme, "", ""}},
                          kernelCC,
                          specs1.gcn_arch_name,
                          "CS_KERNEL_STOCKHAM_PP_BLOCK_CC",
                          params_1.pp_threads_per_transform,
                          params_1.pp_factors_curr,
                          params_1.pp_factors_other,
                          params_1.current_dim,
                          params_1.off_dim,
                          launchers);

            StockhamPartialPassKernelCC kernelRR(specs2, params_2, false);
            make_launcher(specs2.precision,
                          *specs2.transform_type,
                          {{"pp_stoc", specs2.scheme, "", ""}},
                          kernelRR,
                          specs2.gcn_arch_name,
                          "CS_KERNEL_STOCKHAM_PP",
                          params_2.pp_threads_per_transform,
                          params_2.pp_factors_curr,
                          params_2.pp_factors_other,
                          params_2.current_dim,
                          params_2.off_dim,
                          launchers);
        }
        // SBCC_PP + SBCC_PP
        else if((params_1.current_dim == 1 && params_2.current_dim == 2)
                || (params_1.current_dim == 2 && params_2.current_dim == 1))
        {
            throw std::runtime_error("CS_KERNEL_STOCKHAM_PP_BLOCK_CC + "
                                     "CS_KERNEL_STOCKHAM_PP_BLOCK_CC with x as off-dimension not "
                                     "yet implemented for CS_3D_PP");
        }
        // SBRR_PP + SBCC_PP
        else if((params_1.current_dim == 0 && params_2.current_dim == 1)
                || (params_1.current_dim == 1 && params_2.current_dim == 0))
        {
            throw std::runtime_error("CS_KERNEL_STOCKHAM_PP + CS_KERNEL_STOCKHAM_PP_BLOCK_CC with "
                                     "with z as off-dimension not yet "
                                     "implemented for CS_3D_PP");
        }
        else
        {
            throw std::runtime_error("invalid dimensions for CS_3D_PP");
        }
    }
    else if(specs1.scheme == "CS_REAL_3D_PP" && specs2.scheme == "CS_REAL_3D_PP")
    {
        if(params_1.current_dim == 0 && params_2.current_dim == 2)
        {
            StockhamPartialPassKernelRR kernelRR(specs1, params_1);
            make_launcher(specs1.precision,
                          *specs1.transform_type,
                          {{"pp_stoc", specs1.scheme, "", ""}},
                          kernelRR,
                          specs1.gcn_arch_name,
                          "CS_KERNEL_STOCKHAM_PP",
                          params_1.pp_threads_per_transform,
                          params_1.pp_factors_curr,
                          params_1.pp_factors_other,
                          params_1.current_dim,
                          params_1.off_dim,
                          launchers);

            StockhamPartialPassKernelCC kernelCC(specs2, params_2, false);
            make_launcher(specs2.precision,
                          *specs2.transform_type,
                          {{"pp_sbcc", specs2.scheme, "", ""}},
                          kernelCC,
                          specs2.gcn_arch_name,
                          "CS_KERNEL_STOCKHAM_PP_BLOCK_CC",
                          params_2.pp_threads_per_transform,
                          params_2.pp_factors_curr,
                          params_2.pp_factors_other,
                          params_2.current_dim,
                          params_2.off_dim,
                          launchers);
        }
    }
    else
    {
        throw std::runtime_error("unhandled scheme");
    }

    output_json(launchers, kernel_name, output);
}

// Render stockham kernel generated launchers in JSON format.
void stockham_variants(const std::string&            kernel_name,
                       const StockhamGeneratorSpecs& specs,
                       const StockhamGeneratorSpecs& specs2d,
                       std::ostream&                 output)
{

    std::vector<GeneratedLauncher> launchers;

    if(specs.scheme == "CS_KERNEL_STOCKHAM")
    {
        StockhamKernelRR kernel(specs);
        make_launcher(specs.precision,
                      *specs.transform_type,
                      {{"stoc", specs.scheme, "", ""}},
                      kernel,
                      specs.gcn_arch_name,
                      "CS_NONE",
                      0,
                      std::vector<unsigned int>(),
                      std::vector<unsigned int>(),
                      0,
                      0,
                      launchers);
    }
    else if(specs.scheme == "CS_KERNEL_STOCKHAM_BLOCK_CC")
    {
        StockhamKernelCC kernel(specs, false, false);
        make_launcher(specs.precision,
                      *specs.transform_type,
                      {{"sbcc", specs.scheme, "", ""}},
                      kernel,
                      specs.gcn_arch_name,
                      "CS_NONE",
                      0,
                      std::vector<unsigned int>(),
                      std::vector<unsigned int>(),
                      0,
                      0,
                      launchers);
    }
    else if(specs.scheme == "CS_KERNEL_STOCKHAM_BLOCK_RC")
    {
        StockhamKernelRC kernel(specs, false);

        std::vector<LaunchSuffix> suffixes;
        suffixes.push_back({"sbrc", "CS_KERNEL_STOCKHAM_BLOCK_RC", "SBRC_2D", "TILE_ALIGNED"});
        suffixes.push_back(
            {"sbrc_unaligned", "CS_KERNEL_STOCKHAM_BLOCK_RC", "SBRC_2D", "TILE_UNALIGNED"});
        suffixes.push_back({"sbrc3d_fft_trans_xy_z_tile_aligned",
                            "CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z",
                            "SBRC_3D_FFT_TRANS_XY_Z",
                            "TILE_ALIGNED"});
        suffixes.push_back({"sbrc3d_fft_trans_xy_z_tile_unaligned",
                            "CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z",
                            "SBRC_3D_FFT_TRANS_XY_Z",
                            "TILE_UNALIGNED"});
        suffixes.push_back({"sbrc3d_fft_trans_xy_z_diagonal",
                            "CS_KERNEL_STOCKHAM_TRANSPOSE_XY_Z",
                            "SBRC_3D_FFT_TRANS_XY_Z",
                            "DIAGONAL"});
        suffixes.push_back({"sbrc3d_fft_trans_z_xy_tile_aligned",
                            "CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY",
                            "SBRC_3D_FFT_TRANS_Z_XY",
                            "TILE_ALIGNED"});
        suffixes.push_back({"sbrc3d_fft_trans_z_xy_tile_unaligned",
                            "CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY",
                            "SBRC_3D_FFT_TRANS_Z_XY",
                            "TILE_UNALIGNED"});
        suffixes.push_back({"sbrc3d_fft_trans_z_xy_diagonal",
                            "CS_KERNEL_STOCKHAM_TRANSPOSE_Z_XY",
                            "SBRC_3D_FFT_TRANS_Z_XY",
                            "DIAGONAL"});
        suffixes.push_back({"sbrc3d_fft_erc_trans_z_xy_tile_aligned",
                            "CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY",
                            "SBRC_3D_FFT_ERC_TRANS_Z_XY",
                            "TILE_ALIGNED"});
        suffixes.push_back({"sbrc3d_fft_erc_trans_z_xy_tile_unaligned",
                            "CS_KERNEL_STOCKHAM_R_TO_CMPLX_TRANSPOSE_Z_XY",
                            "SBRC_3D_FFT_ERC_TRANS_Z_XY",
                            "TILE_UNALIGNED"});

        make_launcher(specs.precision,
                      *specs.transform_type,
                      suffixes,
                      kernel,
                      specs.gcn_arch_name,
                      "CS_NONE",
                      0,
                      std::vector<unsigned int>(),
                      std::vector<unsigned int>(),
                      0,
                      0,
                      launchers);
    }
    else if(specs.scheme == "CS_KERNEL_STOCKHAM_BLOCK_CR")
    {
        StockhamKernelCR kernel(specs);

        make_launcher(specs.precision,
                      *specs.transform_type,
                      {{"sbcr", specs.scheme, "", ""}},
                      kernel,
                      specs.gcn_arch_name,
                      "CS_NONE",
                      0,
                      std::vector<unsigned int>(),
                      std::vector<unsigned int>(),
                      0,
                      0,
                      launchers);
    }
    else if(specs.scheme == "CS_KERNEL_2D_SINGLE")
    {
        StockhamKernelFused2D fused2d(specs, specs2d);

        launchers.emplace_back(fused2d,
                               specs.scheme,
                               "CS_NONE",
                               0,
                               std::vector<unsigned int>(),
                               std::vector<unsigned int>(),
                               0,
                               0,
                               specs.precision,
                               *specs.transform_type,
                               specs.gcn_arch_name,
                               "",
                               "");
    }
    else
        throw std::runtime_error("unhandled scheme");

    output_json(launchers, kernel_name, output);
}

// =========================================================
// Partial pass parameters row-major ordering helpers.
// Kernel configuration parameters for CS_3D_PP in
// kernel-generator.py are in column-major ordering.
// =========================================================

// Gets partial-pass dimension in row-major ordering.
size_t pp_dim_rm(const size_t dim)
{
    if(dim == 0)
        return 2;

    if(dim == 2)
        return 0;

    return dim;
}
// Gets partial pass parameters in row-major ordering.
void pp_params_rm(std::vector<unsigned int>& parent_length,
                  std::vector<unsigned int>& dims,
                  std::vector<unsigned int>& factors1,
                  std::vector<unsigned int>& factors2,
                  std::vector<unsigned int>& pp_factors1,
                  std::vector<unsigned int>& pp_factors2,
                  std::vector<unsigned int>& workgroup_size,
                  std::vector<unsigned int>& threads_per_transform,
                  std::vector<unsigned int>& threads_per_transform_pp,
                  std::vector<bool>&         direct_to_from_reg)
{
    std::reverse(parent_length.begin(), parent_length.end());

    auto dims_rm = dims;
    for(auto& dim : dims_rm)
        dim = pp_dim_rm(dim);

    if(dims_rm != dims)
    {
        factors1.swap(factors2);
        pp_factors1.swap(pp_factors2);
        std::reverse(workgroup_size.begin(), workgroup_size.end());
        std::reverse(threads_per_transform.begin(), threads_per_transform.end());
        std::reverse(threads_per_transform_pp.begin(), threads_per_transform_pp.end());
        std::reverse(direct_to_from_reg.begin(), direct_to_from_reg.end());
    }
}
// Gets partial-pass off-dimension in row-major ordering.
unsigned int get_pp_off_dim(const std::vector<unsigned int>& dims)
{
    if(dims.size() != 2)
        throw std::runtime_error("CS_3D_PP requires two dimensions configuration");

    unsigned int dims_sum = 0, off_dim = 0;
    for(const auto& dim : dims)
    {
        if(dim < 0 || dim > 2)
            throw std::runtime_error("Invalid dimensions configuration for CS_3D_PP");

        dims_sum += dim;
    }
    switch(dims_sum)
    {
    case 1:
        off_dim = 2;
        break;
    case 2:
        off_dim = 1;
        break;
    case 3:
        off_dim = 0;
        break;
    default:
        throw std::runtime_error("Invalid dimensions configuration for CS_3D_PP");
    }

    return off_dim;
}

// =========================================================
// CS_3D_PP validation helpers.
// =========================================================

// Validates that the current_dim is the correct index
// in parent_length for the given factors.
void validate_pp_length(const std::string&               scheme,
                        const StockhamPartialPassParams& pp_params,
                        const std::vector<unsigned int>& factors)

{
    unsigned int length_curr = product(factors.begin(), factors.end());

    if(scheme == "CS_REAL_3D_PP" && pp_params.current_dim == 0)
    {
        // For real 3D FFTs, the z-dimension length is halved in the
        // complex domain. So we need to multiply by 2 here to get
        // the correct length to compare against parent_length.
        length_curr *= 2;
    }

    auto curr_dim = pp_params.current_dim;
    if(length_curr != pp_params.parent_length[curr_dim])
        throw std::runtime_error("Invalid partial-pass kernel length configuration");
}

// Validates that the off-dim factors are valid and match
// the off-dimension length in parent_length.
void validate_pp_off_dim_length(const StockhamPartialPassParams& pp_params_1,
                                const StockhamPartialPassParams& pp_params_2)
{
    auto off_factors_all = pp_params_1.pp_factors_curr;
    off_factors_all.insert(off_factors_all.end(),
                           pp_params_2.pp_factors_curr.begin(),
                           pp_params_2.pp_factors_curr.end());

    unsigned int length_off_dim = product(off_factors_all.begin(), off_factors_all.end());

    if(pp_params_1.parent_length[pp_params_1.off_dim]
       != pp_params_2.parent_length[pp_params_2.off_dim])
        throw std::runtime_error("Invalid partial-pass kernel off-dimension length");

    if(length_off_dim != pp_params_1.parent_length[pp_params_1.off_dim])
        throw std::runtime_error("Invalid partial-pass kernel off-dimension length");
}
// Validate grid parameters for partial pass kernels.
void validate_pp_grid_params(const StockhamPartialPassParams& params_1,
                             const StockhamPartialPassParams& params_2,
                             const StockhamGeneratorSpecs&    specs_1,
                             const StockhamGeneratorSpecs&    specs_2)
{
    if((specs_1.scheme == "CS_3D_PP" && specs_2.scheme == "CS_3D_PP")
       || (specs_1.scheme == "CS_REAL_3D_PP" && specs_2.scheme == "CS_REAL_3D_PP"))
    {
        // In kernel config file, kernels can be configured in either order,
        // so we need to check both possibilities for which kernel is the SBRR_PP
        // and which is the SBCC_PP (params_1.current_dim vs. params_2.current_dim).

        // SBRR_PP + SBCC_PP
        if((params_1.current_dim == 0 && params_2.current_dim == 2)
           || (params_1.current_dim == 2 && params_2.current_dim == 0))
        {
            // SBRR needs tpb to be prod(pp_factors),
            // so that it has the required off-dim data in LDS
            // to perform partial passes
            auto tpb_sbrr = (params_1.current_dim == 0 && params_2.current_dim == 2)
                                ? specs_1.workgroup_size / specs_1.threads_per_transform
                                : specs_2.workgroup_size / specs_2.threads_per_transform;

            auto prod_factors_off_dim
                = (params_1.current_dim == 0 && params_2.current_dim == 2)
                      ? product(params_1.pp_factors_curr.begin(), params_1.pp_factors_curr.end())
                      : product(params_2.pp_factors_curr.begin(), params_2.pp_factors_curr.end());
            if(tpb_sbrr != prod_factors_off_dim)
            {
                throw std::runtime_error("CS_KERNEL_STOCKHAM_PP requires transform-per-block "
                                         "to be prod(pp_factors)");
            }
        }
        // SBCC_PP + SBCC_PP
        else if((params_1.current_dim == 1 && params_2.current_dim == 2)
                || (params_1.current_dim == 2 && params_2.current_dim == 1))
        {
            throw std::runtime_error("CS_KERNEL_STOCKHAM_PP_BLOCK_CC + "
                                     "CS_KERNEL_STOCKHAM_PP_BLOCK_CC with x as off-dimension not "
                                     "yet implemented for CS_3D_PP");
        }
        // SBRR_PP + SBCC_PP
        else if((params_1.current_dim == 0 && params_2.current_dim == 1)
                || (params_1.current_dim == 1 && params_2.current_dim == 0))
        {
            throw std::runtime_error("CS_KERNEL_STOCKHAM_PP + CS_KERNEL_STOCKHAM_PP_BLOCK_CC with "
                                     "with z as off-dimension not yet "
                                     "implemented for CS_3D_PP");
        }
        else
        {
            throw std::runtime_error("invalid dimensions for CS_3D_PP");
        }

        // Validate pp_threads_per_transform against threads_per_transform
        if(params_1.pp_threads_per_transform > specs_1.threads_per_transform
           || params_2.pp_threads_per_transform > specs_2.threads_per_transform)
        {
            throw std::runtime_error(
                "CS_KERNEL_STOCKHAM_PP requires threads_per_transform_pp to be "
                "equal or less than threads_per_transform");
        }

        // Validate pp_threads_per_transform against pp_factors_curr
        if((params_1.pp_factors_curr.size() == 1 && params_1.pp_threads_per_transform > 1)
           || (params_2.pp_factors_curr.size() == 1 && params_2.pp_threads_per_transform > 1))
        {
            throw std::runtime_error("CS_KERNEL_STOCKHAM_PP and CS_KERNEL_STOCKHAM_PP_BLOCK_CC "
                                     "require threads_per_transform_pp to be 1 when "
                                     "pp_factors has only one factor");
        }
    }
    else
    {
        throw std::runtime_error("unhandled scheme");
    }
}

int main()
{
    std::string line;

    std::string  kernel_name;
    std::string  gcn_arch_name;
    std::string  scheme;
    unsigned int transform_type;
    bool         half_lds;
    unsigned int lds_size_bytes;
    unsigned int bytes_per_element;

    const char* DELIM = "";
    std::cout << "{";

    // Loop until calling process signals EOF
    while(std::getline(std::cin, line))
    {
        // fetch input from stdin - newline delimits kernel, ws delimits kernel args
        std::stringstream        ss(line);
        std::string              tmp;
        std::vector<std::string> tokens;
        while(getline(ss, tmp, ' '))
        {
            tokens.push_back(tmp);
        }

        // work backwards from the end
        auto arg = tokens.rbegin();

        lds_size_bytes = std::stoul(*arg);
        ++arg;

        bytes_per_element = std::stoul(*arg);
        ++arg;

        std::string kernel_name = *arg;

        ++arg;
        scheme = *arg;

        std::vector<unsigned int> parent_length, dims, threads_per_transform_pp, pp_factors1,
            pp_factors2;
        if(scheme == "CS_3D_PP" || scheme == "CS_REAL_3D_PP")
        {
            ++arg;
            parent_length = parse_uints_csv(*arg);

            ++arg;
            pp_factors2 = parse_uints_csv(*arg);

            ++arg;
            pp_factors1 = parse_uints_csv(*arg);

            ++arg;
            threads_per_transform_pp = parse_uints_csv(*arg);

            ++arg;
            dims = parse_uints_csv(*arg);
        }

        ++arg;
        std::vector<bool> direct_to_from_reg;
        direct_to_from_reg = parse_bool_csv(*arg);

        ++arg;
        half_lds = *arg == "1";

        ++arg;
        std::vector<unsigned int> workgroup_size;
        workgroup_size = parse_uints_csv(*arg);

        ++arg;
        std::vector<unsigned int> threads_per_transform;
        threads_per_transform = parse_uints_csv(*arg);

        ++arg;
        gcn_arch_name = *arg;

        ++arg;
        transform_type = std::stoul(*arg);

        ++arg;
        unsigned int precision;
        precision = std::stoul(*arg);

        // create spec and pass to stockham_variants, writes partial output to stdout
        std::cout << DELIM;

        if(scheme == "CS_3D_PP" || scheme == "CS_REAL_3D_PP")
        {
            std::vector<unsigned int> factors1, factors2;

            ++arg;
            factors2 = parse_uints_csv(*arg);

            ++arg;
            factors1 = parse_uints_csv(*arg);

            auto off_dim = get_pp_off_dim(dims);

            pp_params_rm(parent_length,
                         dims,
                         factors1,
                         factors2,
                         pp_factors1,
                         pp_factors2,
                         workgroup_size,
                         threads_per_transform,
                         threads_per_transform_pp,
                         direct_to_from_reg);

            if(threads_per_transform.size() != 2)
                throw std::runtime_error(
                    "CS_3D_PP requires two threads_per_transform configuration");

            if(threads_per_transform_pp.size() != 2)
                throw std::runtime_error(
                    "CS_3D_PP requires two threads_per_transform_pp configuration");

            if(direct_to_from_reg.size() != 2)
                throw std::runtime_error("CS_3D_PP requires two direct_to_from_reg configuration");

            StockhamGeneratorSpecs specs1(
                factors1, {}, precision, gcn_arch_name, workgroup_size[0], scheme, transform_type);
            specs1.direct_to_from_reg    = direct_to_from_reg[0];
            specs1.threads_per_transform = threads_per_transform[0];
            specs1.wgs_is_derived        = true;

            StockhamGeneratorSpecs specs2(
                factors2, {}, precision, gcn_arch_name, workgroup_size[1], scheme, transform_type);
            specs2.direct_to_from_reg    = direct_to_from_reg[1];
            specs2.threads_per_transform = threads_per_transform[1];
            specs2.wgs_is_derived        = true;

            StockhamPartialPassParams pp_params_1(parent_length,
                                                  threads_per_transform_pp[0],
                                                  dims[0],
                                                  off_dim,
                                                  pp_factors1,
                                                  pp_factors2);
            StockhamPartialPassParams pp_params_2(parent_length,
                                                  threads_per_transform_pp[1],
                                                  dims[1],
                                                  off_dim,
                                                  pp_factors2,
                                                  pp_factors1);

            validate_pp_length(scheme, pp_params_1, factors1);
            validate_pp_length(scheme, pp_params_2, factors2);
            validate_pp_off_dim_length(pp_params_1, pp_params_2);
            validate_pp_grid_params(pp_params_1, pp_params_2, specs1, specs2);

            stockham_partial_pass_variants(
                kernel_name, specs1, specs2, pp_params_1, pp_params_2, std::cout);
        }
        else
        {
            std::vector<unsigned int> factors;
            std::vector<unsigned int> factors2d;
            if(scheme == "CS_KERNEL_2D_SINGLE")
            {
                ++arg;
                factors2d = parse_uints_csv(*arg);
            }

            ++arg;
            factors = parse_uints_csv(*arg);

            StockhamGeneratorSpecs specs(factors,
                                         factors2d,
                                         precision,
                                         gcn_arch_name,
                                         workgroup_size[0],
                                         scheme,
                                         transform_type);
            specs.half_lds           = half_lds;
            specs.direct_to_from_reg = direct_to_from_reg[0];

            specs.bytes_per_element = bytes_per_element;

            specs.threads_per_transform = threads_per_transform.front();

            // second dimension for 2D_SINGLE
            StockhamGeneratorSpecs specs2d(factors2d,
                                           factors,
                                           precision,
                                           gcn_arch_name,
                                           workgroup_size[0],
                                           scheme,
                                           transform_type);

            if(!threads_per_transform.empty())
                specs2d.threads_per_transform = threads_per_transform.back();

            // 2D_SINGLE kernels use the specified workgroup size
            // directly.
            // Kernels with an architecture other than generic will
            // also use the workgroup size directly, as the calculations
            // for wgs_is_derived=false assume an LDS size of 64KiB.
            if(scheme == "CS_KERNEL_2D_SINGLE" || gcn_arch_name != generic_gcn_arch_name)
            {
                specs.wgs_is_derived   = true;
                specs2d.wgs_is_derived = true;
            }

            // aim for occupancy-2 by default
            specs.lds_byte_limit   = lds_size_bytes / 2;
            specs2d.lds_byte_limit = lds_size_bytes / 2;

            stockham_variants(kernel_name, specs, specs2d, std::cout);
        }

        DELIM = COMMA;
        std::cout << std::flush;
    }
    std::cout << "}" << std::endl;

    return EXIT_SUCCESS;
}
