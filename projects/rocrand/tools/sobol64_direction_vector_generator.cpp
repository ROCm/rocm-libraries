// Copyright (c) 2019-2023 Advanced Micro Devices, Inc. All rights reserved.
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
    using namespace std::string_view_literals;
    if(argc != 5 || argv[1] == "--help"sv)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "  ./sobol64_direction_vector_generator "
                     "<sets of direction numbers file> "
                     "rocrand_h_sobol64_direction_vectors "
                     "../../library/include/rocrand/rocrand_sobol64_precomputed.h "
                     "../../library/src/rocrand_sobol64_precomputed.bin"
                  << std::endl;
        return -1;
    }

    const std::filesystem::path vector_file(argv[1]);

    std::vector<rocrand_tools::sobol_set> inputs(rocrand_tools::SOBOL_DIM);
    std::vector<uint64_t>                 directions_64(rocrand_tools::SOBOL64_N);

    if(!rocrand_tools::read_sobol_set(inputs.data(), rocrand_tools::SOBOL_DIM, vector_file))
    {
        return -1;
    }

    rocrand_tools::init_direction_vectors<uint64_t>(inputs.data(),
                                                    directions_64.data(),
                                                    64,
                                                    rocrand_tools::SOBOL_DIM);

    const std::string_view      symbol{argv[2]};
    const std::filesystem::path header_path(argv[3]);
    const std::filesystem::path binary_path(argv[4]);

    std::ofstream header_out(header_path, std::ios_base::out | std::ios_base::trunc);
    std::ofstream binary_out(binary_path,
                             std::ios_base::out | std::ios_base::trunc | std::ios_base::binary);

    rocrand_tools::write_preamble(header_out, "sobol64_direction_vector_generator")
        << "#ifndef ROCRAND_SOBOL64_PRECOMPUTED_H_\n"
        << "#define ROCRAND_SOBOL64_PRECOMPUTED_H_\n"
        << '\n'
        << "#include \"rocrand/rocrandapi.h\"\n"
        << "\n"
        << "#ifndef SOBOL_DIM\n"
        << "    #define SOBOL_DIM " << rocrand_tools::SOBOL_DIM << "\n"
        << "#endif // SOBOL_DIM\n"
        << "#define SOBOL64_N " << rocrand_tools::SOBOL64_N << "\n"
        << "\n"
        << "extern \"C\" ROCRANDAPI const unsigned long long " << symbol << "[SOBOL64_N];\n"
        << "\n"
        << "#endif // ROCRAND_SOBOL64_PRECOMPUTED_H_\n";

    binary_out.write(reinterpret_cast<const char*>(directions_64.data()),
                     rocrand_tools::SOBOL64_N * sizeof(directions_64[0]));

    return 0;
}
