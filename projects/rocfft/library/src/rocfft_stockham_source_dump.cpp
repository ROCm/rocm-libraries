#include "../../shared/device_properties.h"
#include "compute_scheme.h"
#include "device/generator/stockham_gen.h"
#include "enum_printer.h"
#include "rtc_stockham_gen.h"

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{

std::vector<unsigned int> parse_factors(const std::string& csv)
{
    std::vector<unsigned int> factors;
    std::stringstream         stream(csv);
    std::string               token;

    while(std::getline(stream, token, ','))
    {
        if(token.empty())
            continue;
        const auto factor = static_cast<unsigned int>(std::stoul(token));
        if(factor == 0)
            throw std::runtime_error("factorization values must be non-zero");
        factors.push_back(factor);
    }

    if(factors.empty())
        throw std::runtime_error("factorization must contain at least one factor");

    return factors;
}

unsigned long long product_of_factors(const std::vector<unsigned int>& factors)
{
    unsigned long long product = 1;
    for(const auto factor : factors)
        product *= factor;
    return product;
}

int parse_direction(const std::string& direction)
{
    if(direction == "forward")
        return -1;
    if(direction == "inverse")
        return 1;

    throw std::runtime_error("direction must be 'forward' or 'inverse'");
}

rocfft_precision parse_precision(const std::string& precision)
{
    if(precision == "single")
        return rocfft_precision_single;
    if(precision == "double")
        return rocfft_precision_double;

    throw std::runtime_error("precision must be 'single' or 'double'");
}

bool parse_bool_flag(const std::string& value)
{
    if(value == "0" || value == "false")
        return false;
    if(value == "1" || value == "true")
        return true;

    throw std::runtime_error("boolean flags must be one of: 0, 1, false, true");
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        if(argc < 8 || argc > 9)
        {
            std::cerr
                << "usage: rocfft_stockham_source_dump <forward|inverse> <single|double> <f0,f1,...> <wgs> <tpt> <half_lds> <direct_reg> [arch]\n";
            return 1;
        }

        const std::string direction_arg         = argv[1];
        const auto        precision             = parse_precision(argv[2]);
        const auto        factors               = parse_factors(argv[3]);
        const auto        workgroup_size        = static_cast<unsigned int>(std::stoul(argv[4]));
        const auto        threads_per_transform = static_cast<unsigned int>(std::stoul(argv[5]));
        const bool        half_lds              = parse_bool_flag(argv[6]);
        const bool        direct_to_from_reg    = parse_bool_flag(argv[7]);
        const std::string arch                  = argc == 9 ? argv[8] : get_curr_gcn_arch_name();

        const auto scheme    = CS_KERNEL_STOCKHAM;
        const auto direction = parse_direction(direction_arg);
        const auto length    = product_of_factors(factors);

        StockhamGeneratorSpecs specs{factors,
                                     {},
                                     static_cast<unsigned int>(precision),
                                     arch,
                                     workgroup_size,
                                     PrintScheme(scheme)};
        specs.threads_per_transform = threads_per_transform;
        specs.half_lds              = half_lds;
        specs.direct_to_from_reg    = direct_to_from_reg;
        specs.wgs_is_derived        = true;
        specs.static_dim            = 1;

        const auto partial_pass = PartialPassType::PPT_NONE;
        const auto pp_params    = StockhamPartialPassParams{};
        const auto dir_reg_type = direct_to_from_reg ? DirectRegType::TRY_ENABLE_IF_SUPPORT
                                                     : DirectRegType::FORCE_OFF_OR_NOT_SUPPORT;

        auto kernel_name = stockham_rtc_kernel_name(specs,
                                                    specs,
                                                    scheme,
                                                    direction,
                                                    precision,
                                                    rocfft_placement_inplace,
                                                    rocfft_array_type_complex_interleaved,
                                                    rocfft_array_type_complex_interleaved,
                                                    true,
                                                    0,
                                                    0,
                                                    false,
                                                    dir_reg_type,
                                                    IntrinsicAccessType::DISABLE_BOTH,
                                                    SBRC_TRANSPOSE_TYPE::NONE,
                                                    CallbackType::NONE,
                                                    BluesteinFuseType::BFT_NONE,
                                                    partial_pass,
                                                    pp_params,
                                                    {},
                                                    {});

        unsigned int transforms_per_block = 0;
        auto         source               = stockham_rtc(specs,
                                           specs,
                                           pp_params,
                                           &transforms_per_block,
                                           kernel_name,
                                           scheme,
                                           direction,
                                           precision,
                                           rocfft_placement_inplace,
                                           rocfft_array_type_complex_interleaved,
                                           rocfft_array_type_complex_interleaved,
                                           true,
                                           0,
                                           0,
                                           false,
                                           dir_reg_type,
                                           IntrinsicAccessType::DISABLE_BOTH,
                                           SBRC_TRANSPOSE_TYPE::NONE,
                                           CallbackType::NONE,
                                           BluesteinFuseType::BFT_NONE,
                                           partial_pass,
                                           {},
                                           {});

        if(transforms_per_block != 1)
        {
            throw std::runtime_error("rocfft_stockham_source_dump only supports single-transform kernels, got "
                                     "transforms_per_block="
                                     + std::to_string(transforms_per_block));
        }

        std::cout << "// kernel_name=" << kernel_name << "\n";
        std::cout << "// length=" << length << "\n";
        std::cout << "// workgroup_size=" << workgroup_size << "\n";
        std::cout << "// threads_per_transform=" << threads_per_transform << "\n";
        std::cout << "// transforms_per_block=" << transforms_per_block << "\n";
        std::cout << source;
        return 0;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << "\n";
        return 1;
    }
}
