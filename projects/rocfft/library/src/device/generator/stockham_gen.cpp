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

#include "../../../../shared/precision_type.h"
#include "generator.h"
#include "stockham_gen.h"
#include <array>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "stockham_gen_cc.h"
#include "stockham_gen_cr.h"
#include "stockham_gen_rc.h"
#include "stockham_gen_rr.h"

#include "stockham_gen_2d.h"

// this rolls up all the information about the generated launchers,
// enough to genernate the function pool entry
struct GeneratedLauncher
{
    GeneratedLauncher(StockhamKernel&    kernel,
                      const std::string& scheme,
                      bool               double_precision,
                      const std::string& sbrc_type,
                      const std::string& sbrc_transpose_type)
        : scheme(scheme)
        , lengths(kernel.launcher_lengths())
        , factors(kernel.launcher_factors())
        , transforms_per_block(kernel.transforms_per_block)
        , workgroup_size(kernel.workgroup_size)
        , half_lds(kernel.half_lds)
        , direct_to_from_reg(kernel.direct_to_from_reg)
        , sbrc_type(sbrc_type)
        , sbrc_transpose_type(sbrc_transpose_type)
        , double_precision(double_precision)
    {
    }

    std::string               scheme;
    std::vector<unsigned int> lengths;
    std::vector<unsigned int> factors;

    unsigned int transforms_per_block;
    unsigned int workgroup_size;
    bool         half_lds;
    bool         direct_to_from_reg;

    // SBRC transpose type
    std::string sbrc_type;
    std::string sbrc_transpose_type;
    bool        double_precision;

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
        add_member("double_precision", double_precision ? "true" : "false");

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

void make_launcher(const std::vector<unsigned int>& precision_types,
                   const std::vector<LaunchSuffix>& launcher_suffixes,
                   StockhamKernel&                  kernel,
                   std::vector<GeneratedLauncher>&  generated_launchers)
{
    for(auto precision_type : precision_types)
    {
        for(auto&& launcher : launcher_suffixes)
        {
            generated_launchers.emplace_back(kernel,
                                             launcher.scheme,
                                             precision_type == rocfft_precision_double,
                                             launcher.sbrc_type,
                                             launcher.sbrc_transpose_type);
        }
    }
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

void stockham_variants(const std::string&            kernel_name,
                       const StockhamGeneratorSpecs& specs,
                       const StockhamGeneratorSpecs& specs2d,
                       std::ostream&                 output)
{

    std::vector<GeneratedLauncher> launchers;

    if(specs.scheme == "CS_KERNEL_STOCKHAM")
    {
        StockhamKernelRR kernel(specs);
        make_launcher(specs.precisions, {{"stoc", specs.scheme, "", ""}}, kernel, launchers);
    }
    else if(specs.scheme == "CS_KERNEL_STOCKHAM_BLOCK_CC")
    {
        StockhamKernelCC kernel(specs, false, false);
        make_launcher(specs.precisions, {{"sbcc", specs.scheme, "", ""}}, kernel, launchers);
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

        make_launcher(specs.precisions, suffixes, kernel, launchers);
    }
    else if(specs.scheme == "CS_KERNEL_STOCKHAM_BLOCK_CR")
    {
        StockhamKernelCR kernel(specs);

        make_launcher(specs.precisions, {{"sbcr", specs.scheme, "", ""}}, kernel, launchers);
    }
    else if(specs.scheme == "CS_KERNEL_2D_SINGLE")
    {
        StockhamKernelFused2D fused2d(specs, specs2d);

        // output 2D launchers
        for(auto prec_type : specs.precisions)
        {
            launchers.emplace_back(
                fused2d, specs.scheme, (prec_type == rocfft_precision_double), "", "");
        }
    }
    else
        throw std::runtime_error("unhandled scheme");

    // output json (via stdout) describing the launchers that were generated, so
    // kernel-generator can generate the function pool

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

static size_t max_bytes_per_element(const std::vector<unsigned int>& precisions)
{
    // generate for the maximum element size in the available
    // precisions
    size_t element_size = 0;
    for(auto p : precisions)
    {
        element_size = std::max(element_size, complex_type_size(static_cast<rocfft_precision>(p)));
    }
    return element_size;
}

int main()
{
    std::string line;

    std::string  kernel_name;
    std::string  scheme;
    bool         direct_to_from_reg;
    bool         half_lds;
    unsigned int workgroup_size;
    unsigned int lds_size_bytes;

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

        std::string kernel_name = *arg;

        ++arg;
        scheme = *arg;

        ++arg;
        direct_to_from_reg = *arg == "1";

        ++arg;
        half_lds = *arg == "1";

        ++arg;
        workgroup_size = std::stoul(*arg);

        ++arg;
        std::vector<unsigned int> threads_per_transform;
        threads_per_transform = parse_uints_csv(*arg);

        ++arg;
        std::vector<unsigned int> precisions;
        precisions = parse_uints_csv(*arg);

        std::vector<unsigned int> factors;
        std::vector<unsigned int> factors2d;
        if(scheme == "CS_KERNEL_2D_SINGLE")
        {
            ++arg;
            factors2d = parse_uints_csv(*arg);
        }

        ++arg;
        factors = parse_uints_csv(*arg);

        StockhamGeneratorSpecs specs(factors, factors2d, precisions, workgroup_size, scheme);
        specs.half_lds           = half_lds;
        specs.direct_to_from_reg = direct_to_from_reg;

        specs.bytes_per_element = max_bytes_per_element(precisions);

        specs.threads_per_transform = threads_per_transform.front();

        // second dimension for 2D_SINGLE
        StockhamGeneratorSpecs specs2d(factors2d, factors, precisions, workgroup_size, scheme);
        if(!threads_per_transform.empty())
            specs2d.threads_per_transform = threads_per_transform.back();

        // aim for occupancy-2 by default
        specs.lds_byte_limit   = lds_size_bytes / 2;
        specs2d.lds_byte_limit = lds_size_bytes / 2;

        // create spec and pass to stockham_variants, writes partial output to stdout
        std::cout << DELIM;
        stockham_variants(kernel_name, specs, specs2d, std::cout);
        DELIM = COMMA;
        std::cout << std::flush;
    }
    std::cout << "}" << std::endl;

    return EXIT_SUCCESS;
}
