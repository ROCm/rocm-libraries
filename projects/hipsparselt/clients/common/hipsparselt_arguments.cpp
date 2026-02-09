/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2024 Advanced Micro Devices, Inc.
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

#include "hipsparselt_arguments.hpp"
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <istream>
#include <ostream>
#include <utility>

void Arguments::init()
{
    // match python in hipsparselt_common.py

    function[0] = 0;
    strcpy(name, "hipsparselt-bench");
    category[0]            = 0;
    known_bug_platforms[0] = 0;

    alpha = 1.0;
    beta  = 0.0;

    stride_a = 0;
    stride_b = 0;
    stride_c = 0;
    stride_d = 0;

    user_allocated_workspace = 0;

    M = 128;
    N = 128;
    K = 128;

    lda = 0;
    ldb = 0;
    ldc = 0;
    ldd = 0;

    batch_count = 1;

    iters      = 10;
    cold_iters = 2;

    algo           = 0;
    solution_index = 0;

    a_type       = HIP_R_16F;
    b_type       = HIP_R_16F;
    c_type       = HIP_R_16F;
    d_type       = HIP_R_16F;
    compute_type = HIPSPARSELT_COMPUTE_32F;
    bias_type    = HIP_R_32F;

    prune_algo = HIPSPARSELT_PRUNE_SPMMA_STRIP;

    initialization = hipsparselt_initialization::hpl;

    // memory padding for testing write out of bounds
    pad = 4096;

    // 16 bit
    threads = 0;
    streams = 0;

    // bytes
    devices = 0;

    norm_check = 0;
    unit_check = 1;
    norm_check_assert = true;
    timing     = 0;

    transA = '*';
    transB = '*';

    activation_type = hipsparselt_activation_type::none;
    activation_arg1 = 0.0f;
    activation_arg2 = std::numeric_limits<float>::infinity();
    c_noalias_d     = false;
    HMM             = false;
    search          = false;
    search_iters    = 10;

    inEqualOut      = false;
    logging         = -1;
}

// Function to print Arguments out to stream in YAML format
hipsparselt_internal_ostream& operator<<(hipsparselt_internal_ostream& os, const Arguments& arg)
{
    // delim starts as "{ " and becomes ", " afterwards
    auto print_pair = [&, delim = "{ "](const char* name, const auto& value) mutable {
        os << delim << std::make_pair(name, value);
        delim = ", ";
    };

    // Print each (name, value) tuple pair
#define NAME_VALUE_PAIR(NAME) print_pair(#NAME, arg.NAME)
    // cppcheck-suppress unknownMacro
    FOR_EACH_ARGUMENT(NAME_VALUE_PAIR, ;);

    // Closing brace
    return os << " }\n";
}

// Google Tests uses this automatically with std::ostream to dump parameters
std::ostream& operator<<(std::ostream& os, const Arguments& arg)
{
    hipsparselt_internal_ostream oss;
    // Print to hipsparselt_internal_ostream, then transfer to std::ostream
    return os << (oss << arg);
}

// Function to read Structures data from stream
std::istream& operator>>(std::istream& is, Arguments& arg)
{
    is.read(reinterpret_cast<char*>(&arg), sizeof(arg));
    return is;
}

// Error message about incompatible binary file format
static void validation_error [[noreturn]] (const char* name)
{
    hipsparselt_cerr << "Arguments field \"" << name
                     << "\" does not match format.\n\n"
                        "Fatal error: Binary test data does match input format.\n"
                        "Ensure that hipsparselt_arguments.hpp and hipsparselt_common.yaml\n"
                        "define exactly the same Arguments, that hipsparselt_gentest.py\n"
                        "generates the data correctly, and that endianness is the same."
                     << std::endl;
    hipsparselt_abort();
}

// hipsparselt_gentest.py is expected to conform to this format.
// hipsparselt_gentest.py uses hipsparselt_common.yaml to generate this format.
void Arguments::validate(std::istream& ifs)
{
    char      header[12]{}, trailer[12]{};
    Arguments arg{};

    ifs.read(header, sizeof(header));
    ifs >> arg;
    ifs.read(trailer, sizeof(trailer));
    if(strcmp(header, "hipSPARSELt"))
        validation_error("header");

    if(strcmp(trailer, "HIPsparselT"))
        validation_error("trailer");

    auto check_func = [sig = 0u](const char* name, const auto& value) mutable {
        static_assert(sizeof(value) <= 256,
                      "Fatal error: Arguments field is too large (greater than 256 bytes).");
        for(size_t i = 0; i < sizeof(value); ++i)
        {
            if(reinterpret_cast<const unsigned char*>(&value)[i] ^ sig ^ i)
                validation_error(name);
        }
        sig = (sig + 89) % 256;
    };

    // Apply check_func to each pair (name, value) of Arguments as a tuple
#define CHECK_FUNC(NAME) check_func(#NAME, arg.NAME)
    FOR_EACH_ARGUMENT(CHECK_FUNC, ;);
}
