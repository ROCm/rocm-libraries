/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2022-2025 Advanced Micro Devices, Inc.
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
#include "hipsparselt_data.hpp"
#include "hipsparselt_datatype2string.hpp"
#include "hipsparselt_test.hpp"
#include "spmm/testing_compress.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{

    // ----------------------------------------------------------------------------
    // compress
    // ----------------------------------------------------------------------------

    // In the general case of <Ti, To, Tc>, these tests do not apply, and if this
    // functor is called, an internal error message is generated. When converted
    // to bool, this functor returns false.
    template <typename Ti, typename To = Ti, typename Tc = To, typename TBias = Ti, typename = void>
    struct compress_testing : hipsparselt_test_invalid
    {
    };

    // When Ti = To = Tc != void, this test applies.
    // When converted to bool, this functor returns true.
    template <typename Ti, typename To, typename Tc, typename TBias>
    struct compress_testing<
        Ti,
        To,
        Tc,
        TBias,
        std::enable_if_t<std::is_same<Ti, __half>{} || std::is_same<Ti, hip_bfloat16>{}
                         || std::is_same<Ti, int8_t>{}
#ifdef HIPSPARSELT_CLIENT_ENABLE_FP8_OCP
			 || std::is_same<Ti, hipsparselt_fp8_e4m3>{}
                         || std::is_same<Ti, hipsparselt_fp8_e5m2>{}
#endif
			 >> : hipsparselt_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "compress"))
                testing_compress<Ti, To, Tc>(arg);
            else if(!strcmp(arg.function, "compress_batched"))
                testing_compress<Ti, To, Tc, hipsparselt_batch_type::batched>(arg);
            else if(!strcmp(arg.function, "compress_strided_batched"))
                testing_compress<Ti, To, Tc, hipsparselt_batch_type::strided_batched>(arg);
            else if(!strcmp(arg.function, "compress_bad_arg"))
                testing_compress_bad_arg<Ti, To, Tc>(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct compress_test : RocSparseLt_Test<compress_test, compress_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return hipsparselt_spmm_dispatch<type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "compress") || !strcmp(arg.function, "compress_batched")
                   || !strcmp(arg.function, "compress_strided_batched")
                   || !strcmp(arg.function, "compress_bad_arg");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocSparseLt_TestName<compress_test> name(arg.name);
            name << hip_datatype_to_string(arg.a_type);

            if(strstr(arg.function, "_bad_arg") != nullptr)
            {
                name << "_bad_arg";
            }
            else
            {
                name << '_' << (arg.sparse_b ? "SB" : "SA");

                name << '_' << (char)std::toupper(arg.transA) << (char)std::toupper(arg.transB);

                name << '_' << arg.M << '_' << arg.N << '_' << arg.K << '_'
                     << (arg.sparse_b ? arg.ldb : arg.lda);

                name << '_' << (char)std::toupper(arg.orderA) << (char)std::toupper(arg.orderB)
                     << (char)std::toupper(arg.orderC) << (char)std::toupper(arg.orderD);

                if(strstr(arg.function, "_batched") != nullptr)
                    name << '_' << arg.batch_count;

                if(strstr(arg.function, "_strided_batched") != nullptr)
                    name << '_' << (arg.sparse_b ? arg.stride_b : arg.stride_a);
            }
            if(arg.func_version > 1)
                name << "_v" << arg.func_version;
            return std::move(name);
        }
    };

    TEST_P(compress_test, conversion)
    {
        RUN_TEST_ON_THREADS_STREAMS(hipsparselt_spmm_dispatch<compress_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(compress_test);

} // namespace
