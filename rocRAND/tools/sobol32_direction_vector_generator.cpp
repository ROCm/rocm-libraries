// Copyright (c) 2019-2024 Advanced Micro Devices, Inc. All rights reserved.
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

#include "sobol_utils.hpp"
#include "utils.hpp"

#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>

#include <stdint.h>

int main(int argc, char const* argv[])
{
    using namespace std::literals::string_view_literals;
    if(argc != 5 || argv[1] == "--help"sv)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "  ./sobol32_direction_vector_generator "
                     "<sets of direction numbers file> "
                     "rocrand_h_sobol32_direction_vectors "
                     "../../library/include/rocrand/rocrand_sobol32_precomputed.h "
                     "../../library/src/rocrand_sobol32_precomputed.bin"
                  << std::endl;
        return -1;
    }
    const std::filesystem::path vector_file(argv[1]);

    std::vector<rocrand_tools::sobol_set> inputs(rocrand_tools::SOBOL_DIM);
    std::vector<uint32_t>                 directions_32(rocrand_tools::SOBOL32_N);

    if(!rocrand_tools::read_sobol_set(inputs.data(), rocrand_tools::SOBOL_DIM, vector_file))
    {
        return -1;
    }

    rocrand_tools::init_direction_vectors<uint32_t>(inputs.data(),
                                                    directions_32.data(),
                                                    32,
                                                    rocrand_tools::SOBOL_DIM);

    const std::string_view      symbol{argv[2]};
    const std::filesystem::path header_path(argv[3]);
    const std::filesystem::path binary_path(argv[4]);

    std::ofstream header_out(header_path, std::ios_base::out | std::ios_base::trunc);
    std::ofstream binary_out(binary_path,
                             std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);

    rocrand_tools::write_preamble(header_out, "sobol32_direction_vector_generator")
        << "#ifndef ROCRAND_SOBOL32_PRECOMPUTED_H_\n"
        << "#define ROCRAND_SOBOL32_PRECOMPUTED_H_\n"
        << '\n'
        << "#include \"rocrand/rocrandapi.h\"\n"
        << "\n"
        << "#ifndef SOBOL_DIM\n"
        << "    #define SOBOL_DIM " << rocrand_tools::SOBOL_DIM << "\n"
        << "#endif // SOBOL_DIM\n"
        << "#define SOBOL32_N " << rocrand_tools::SOBOL32_N << "\n"
        << "\n"
        << "extern \"C\" ROCRANDAPI const unsigned int " << symbol << "[SOBOL32_N];\n"
        << "\n"
        << "#endif // ROCRAND_SOBOL32_PRECOMPUTED_H_\n";

    binary_out.write(reinterpret_cast<const char*>(directions_32.data()),
                     rocrand_tools::SOBOL32_N * sizeof(directions_32[0]));

    return 0;
}
