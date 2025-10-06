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
#include "testing_auxiliary.hpp"
#include "type_dispatch.hpp"
#include <cctype>
#include <cstring>
#include <type_traits>

namespace
{

    // ----------------------------------------------------------------------------
    // aux
    // ----------------------------------------------------------------------------

    // In the general case of <Ti, To, Tc>, these tests do not apply, and if this
    // functor is called, an internal error message is generated. When converted
    // to bool, this functor returns false.
    template <typename Ti, typename To = Ti, typename Tc = To, typename TBias = Ti, typename = void>
    struct aux_testing : hipsparselt_test_invalid
    {
    };

    // When Ti = To = Tc != void, this test applies.
    // When converted to bool, this functor returns true.
    template <typename Ti, typename To, typename Tc, typename TBias>
    struct aux_testing<
        Ti,
        To,
        Tc,
        TBias,
        std::enable_if_t<std::is_same<Ti, __half>{} || std::is_same<Ti, hip_bfloat16>{}
                         || std::is_same<Ti, int8_t>{}>> : hipsparselt_test_valid
    {
        void operator()(const Arguments& arg)
        {
            if(!strcmp(arg.function, "aux_get_version"))
                testing_aux_get_version(arg);
            else if(!strcmp(arg.function, "aux_handle_init_bad_arg"))
                testing_aux_handle_init_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_handle_destroy_bad_arg"))
                testing_aux_handle_destroy_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_handle"))
                testing_aux_handle(arg);
            else if(!strcmp(arg.function, "aux_mat_init_dense_bad_arg"))
                testing_aux_mat_init_dense_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_mat_init_structured_bad_arg"))
                testing_aux_mat_init_structured_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_mat_dense_init_arg"))
                testing_aux_mat_dense_init(arg);
            else if(!strcmp(arg.function, "aux_mat_structured_init"))
                testing_aux_mat_structured_init(arg);
            else if(!strcmp(arg.function, "aux_mat_assign"))
                testing_aux_mat_assign(arg);
            else if(!strcmp(arg.function, "aux_mat_destroy_bad_arg"))
                testing_aux_mat_destroy_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_mat_set_attr_bad_arg"))
                testing_aux_mat_set_attr_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_mat_get_attr_bad_arg"))
                testing_aux_mat_get_attr_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_mat_set_get_attr"))
                testing_aux_mat_set_get_attr(arg);
            else if(!strcmp(arg.function, "aux_matmul_init_bad_arg"))
                testing_aux_matmul_init_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_matmul_init"))
                testing_aux_matmul_init(arg);
            else if(!strcmp(arg.function, "aux_matmul_assign"))
                testing_aux_matmul_assign(arg);
            else if(!strcmp(arg.function, "aux_matmul_set_attr_bad_arg"))
                testing_aux_matmul_set_attr_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_matmul_get_attr_bad_arg"))
                testing_aux_matmul_get_attr_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_matmul_set_get_attr"))
                testing_aux_matmul_set_get_attr(arg);
            else if(!strcmp(arg.function, "aux_matmul_set_get_bias_vector"))
                testing_aux_matmul_set_get_bias_vector(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_init_bad_arg"))
                testing_aux_matmul_alg_init_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_init"))
                testing_aux_matmul_alg_init(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_assign"))
                testing_aux_matmul_alg_assign(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg"))
                testing_aux_matmul_alg_set_attr_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_matmul_alg_get_attr_bad_arg"))
                testing_aux_matmul_alg_get_attr_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_matmul_plan_init_bad_arg"))
                testing_aux_matmul_plan_init_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_matmul_plan_init"))
                testing_aux_matmul_plan_init(arg);
            else if(!strcmp(arg.function, "aux_matmul_plan_destroy_bad_arg"))
                testing_aux_matmul_plan_destroy_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_get_workspace_size_bad_arg"))
                testing_aux_get_workspace_size_bad_arg(arg);
            else if(!strcmp(arg.function, "aux_get_workspace_size"))
                testing_aux_get_workspace_size(arg);
            else
                FAIL() << "Internal error: Test called with unknown function: " << arg.function;
        }
    };

    struct aux_test : RocSparseLt_Test<aux_test, aux_testing>
    {
        // Filter for which types apply to this suite
        static bool type_filter(const Arguments& arg)
        {
            return hipsparselt_spmm_dispatch<type_filter_functor>(arg);
        }

        // Filter for which functions apply to this suite
        static bool function_filter(const Arguments& arg)
        {
            return !strcmp(arg.function, "aux_get_version")
                   || !strcmp(arg.function, "aux_handle_init_bad_arg")
                   || !strcmp(arg.function, "aux_handle_destroy_bad_arg")
                   || !strcmp(arg.function, "aux_handle")
                   || !strcmp(arg.function, "aux_mat_init_dense_bad_arg")
                   || !strcmp(arg.function, "aux_mat_init_structured_bad_arg")
                   || !strcmp(arg.function, "aux_mat_dense_init_arg")
                   || !strcmp(arg.function, "aux_mat_structured_init")
                   || !strcmp(arg.function, "aux_mat_assign")
                   || !strcmp(arg.function, "aux_mat_destroy_bad_arg")
                   || !strcmp(arg.function, "aux_mat_set_attr_bad_arg")
                   || !strcmp(arg.function, "aux_mat_get_attr_bad_arg")
                   || !strcmp(arg.function, "aux_mat_set_get_attr")
                   || !strcmp(arg.function, "aux_matmul_init_bad_arg")
                   || !strcmp(arg.function, "aux_matmul_init")
                   || !strcmp(arg.function, "aux_matmul_assign")
                   || !strcmp(arg.function, "aux_matmul_set_attr_bad_arg")
                   || !strcmp(arg.function, "aux_matmul_get_attr_bad_arg")
                   || !strcmp(arg.function, "aux_matmul_set_get_attr")
                   || !strcmp(arg.function, "aux_matmul_set_get_bias_vector")
                   || !strcmp(arg.function, "aux_matmul_alg_init_bad_arg")
                   || !strcmp(arg.function, "aux_matmul_alg_init")
                   || !strcmp(arg.function, "aux_matmul_alg_assign")
                   || !strcmp(arg.function, "aux_matmul_alg_set_attr_bad_arg")
                   || !strcmp(arg.function, "aux_matmul_alg_get_attr_bad_arg")
                   || !strcmp(arg.function, "aux_matmul_plan_init_bad_arg")
                   || !strcmp(arg.function, "aux_matmul_plan_init")
                   || !strcmp(arg.function, "aux_matmul_plan_destroy_bad_arg")
                   || !strcmp(arg.function, "aux_get_workspace_size_bad_arg")
                   || !strcmp(arg.function, "aux_get_workspace_size");
        }

        // Google Test name suffix based on parameters
        static std::string name_suffix(const Arguments& arg)
        {
            RocSparseLt_TestName<aux_test> name(arg.name);

            name << hip_datatype_to_string(arg.a_type) << hip_datatype_to_string(arg.b_type)
                 << hip_datatype_to_string(arg.c_type) << hip_datatype_to_string(arg.d_type);

            return std::move(name);
        }
    };

    TEST_P(aux_test, conversion)
    {
        RUN_TEST_ON_THREADS_STREAMS(hipsparselt_spmm_dispatch<aux_testing>(GetParam()));
    }
    INSTANTIATE_TEST_CATEGORIES(aux_test);

} // namespace
