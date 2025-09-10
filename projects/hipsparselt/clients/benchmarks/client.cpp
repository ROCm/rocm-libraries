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

#include "program_options.hpp"

#include "hipsparselt_data.hpp"
#include "hipsparselt_datatype2string.hpp"
#include "hipsparselt_parse_data.hpp"
#include "type_dispatch.hpp"
#include "utility.hpp"
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstring>
#include <hipsparselt/hipsparselt.h>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "testing_compress.hpp"
#include "testing_prune.hpp"
#include "testing_spmm.hpp"

#include "type_dispatch.hpp"
#include "utility.hpp"
#include <algorithm>
#undef I

using namespace roc; // For emulated program_options
using namespace std::literals; // For std::string literals of form "str"s

struct str_less
{
    bool operator()(const char* a, const char* b) const
    {
        return strcmp(a, b) < 0;
    }
};

// Map from const char* to function taking const Arguments& using comparison above
using func_map = std::map<const char*, void (*)(const Arguments&), str_less>;

// Run a function by using map to map arg.function to function
void run_function(const func_map& map, const Arguments& arg, const std::string& msg = "")
{
    auto match = map.find(arg.function);
    if(match == map.end())
        throw std::invalid_argument("Invalid combination --function "s + arg.function
                                    + " --a_type "s + hip_datatype_to_string(arg.a_type) + msg);
    match->second(arg);
}

// Template to dispatch testing_gemm_strided_batched_ex for performance tests
// the test is marked invalid when (Ti, To, Tc) not in (H/H/S, B/B/S, I8/I8/32, I8/H/I32)
template <typename Ti, typename To = Ti, typename Tc = To, typename TBias = Ti, typename = void>
struct perf_sparse : hipsparselt_test_invalid
{
};

template <typename Ti, typename To, typename Tc, typename TBias>
struct perf_sparse<
    Ti,
    To,
    Tc,
    TBias,
    std::enable_if_t<
        (std::is_same<Ti, To>{} && (std::is_same<Ti, __half>{} || std::is_same<Ti, hip_bfloat16>{})
         && std::is_same<Tc, float>{})
        || (std::is_same<Ti, To>{} && (std::is_same<Ti, int8_t>{}) && std::is_same<Tc, int32_t>{})
        || (std::is_same<Ti, int8_t>{} && (std::is_same<To, __half>{})
            && std::is_same<Tc, int32_t>{})
        || (std::is_same<Ti, int8_t>{} && (std::is_same<To, hip_bfloat16>{})
            && std::is_same<Tc, int32_t>{})
#ifdef HIPSPARSELT_CLIENT_ENABLE_FP8_OCP
        || (std::is_same<Ti, hipsparselt_fp8_e4m3>{} && (std::is_same<To, float>{})
            && std::is_same<Tc, float>{})
        || (std::is_same<Ti, hipsparselt_fp8_e5m2>{} && (std::is_same<To, float>{})
            && std::is_same<Tc, float>{})
#ifdef __HIP_PLATFORM_NVIDIA__
        || (std::is_same<Ti, hipsparselt_fp8_e4m3>{} && (std::is_same<To, __half>{})
            && std::is_same<Tc, float>{})
        || (std::is_same<Ti, hipsparselt_fp8_e4m3>{} && (std::is_same<To, hip_bfloat16>{})
            && std::is_same<Tc, float>{})
        || (std::is_same<Ti, hipsparselt_fp8_e5m2>{} && (std::is_same<To, __half>{})
            && std::is_same<Tc, float>{})
        || (std::is_same<Ti, hipsparselt_fp8_e5m2>{} && (std::is_same<To, hip_bfloat16>{})
            && std::is_same<Tc, float>{})
#endif
#endif
	>> : hipsparselt_test_valid
{
    void operator()(const Arguments& arg)
    {
        static const func_map map = {
            {"prune", testing_prune<Ti, To, Tc>},
            {"prune_batched", testing_prune<Ti, To, Tc, hipsparselt_batch_type::batched>},
            {"prune_strided_batched",
             testing_prune<Ti, To, Tc, hipsparselt_batch_type::strided_batched>},
            {"compress", testing_compress<Ti, To, Tc>},
            {"compress_batched", testing_compress<Ti, To, Tc, hipsparselt_batch_type::batched>},
            {"compress_strided_batched",
             testing_compress<Ti, To, Tc, hipsparselt_batch_type::strided_batched>},
            {"spmm", testing_spmm<Ti, To, Tc, TBias>},
            {"spmm_batched", testing_spmm<Ti, To, Tc, TBias, hipsparselt_batch_type::batched>},
            {"spmm_strided_batched",
             testing_spmm<Ti, To, Tc, TBias, hipsparselt_batch_type::strided_batched>},
        };
        run_function(map, arg);
    }
};

int run_bench_test(Arguments& arg, const std::string& filter, bool any_stride, bool yaml = false)
{
    static int runOnce = (hipsparseLtInitialize(), 0); // Initialize hipSPARSELt

    hipsparselt_cout << std::setiosflags(std::ios::fixed)
                     << std::setprecision(7); // Set precision to 7 digits

    // disable unit_check in client benchmark, it is only used in gtest unit test
    arg.unit_check = 0;

    // enable timing check,otherwise no performance data collected
    arg.timing = 1;

    // One stream and one thread (0 indicates to use default behavior)
    arg.streams = 0;
    arg.threads = 0;

    // Skip past any testing_ prefix in function
    static constexpr char prefix[] = "testing_";
    const char*           function = arg.function;
    if(!strncmp(function, prefix, sizeof(prefix) - 1))
        function += sizeof(prefix) - 1;

    if(yaml && strstr(function, "_bad_arg"))
        return 0;
    if(!filter.empty())
    {
        if(!strstr(function, filter.c_str()))
            return 0;
    }

    // adjust dimension for GEMM routines
    int64_t min_lda = arg.transA == 'N' ? (arg.orderA == 'C' ? arg.M : arg.K)
                                        : (arg.orderA == 'C' ? arg.K : arg.M);
    int64_t min_ldb = arg.transB == 'N' ? (arg.orderB == 'C' ? arg.K : arg.N)
                                        : (arg.orderB == 'C' ? arg.N : arg.K);
    int64_t min_ldc = arg.orderC == 'C' ? arg.M : arg.N;
    int64_t min_ldd = arg.orderD == 'C' ? arg.M : arg.N;
    if(arg.lda < min_lda)
    {
        hipsparselt_cout << "hipsparselt-bench INFO: lda < min_lda, set lda = " << min_lda
                         << std::endl;
        arg.lda = min_lda;
    }
    if(arg.ldb < min_ldb)
    {
        hipsparselt_cout << "hipsparselt-bench INFO: ldb < min_ldb, set ldb = " << min_ldb
                         << std::endl;
        arg.ldb = min_ldb;
    }
    if(arg.ldc < min_ldc)
    {
        hipsparselt_cout << "hipsparselt-bench INFO: ldc < min_ldc, set ldc = " << min_ldc
                         << std::endl;
        arg.ldc = min_ldc;
    }
    if(arg.ldd < min_ldd)
    {
        hipsparselt_cout << "hipsparselt-bench INFO: ldd < min_ldd, set ldd = " << min_ldc
                         << std::endl;
        arg.ldd = min_ldd;
    }

    int64_t min_stride_a = arg.lda
                           * (arg.transA == 'N' ? (arg.orderA == 'C' ? arg.K : arg.M)
                                                : (arg.orderA == 'C' ? arg.M : arg.K));
    int64_t min_stride_b = arg.ldb
                           * (arg.transB == 'N' ? (arg.orderB == 'C' ? arg.N : arg.K)
                                                : (arg.orderB == 'C' ? arg.K : arg.N));
    int64_t min_stride_c    = arg.ldc * (arg.orderC == 'C' ? arg.N : arg.M);
    int64_t min_stride_d    = arg.ldd * (arg.orderD == 'C' ? arg.N : arg.M);
    int64_t min_bias_stride = arg.M;

    // force using min_stride when stride is -1
    // stride_a is zero means the boradcast mode.
    if(arg.stride_a == -1 || (!any_stride && arg.stride_a < min_stride_a && arg.stride_a != 0))
    {
        hipsparselt_cout
            << "hipsparselt-bench INFO: stride_a < min_stride_a and stride_a != 0, set stride_a = "
            << min_stride_a << std::endl;
        arg.stride_a = min_stride_a;
    }
    if(arg.stride_b == -1 || (!any_stride && arg.stride_b < min_stride_b && arg.stride_b != 0))
    {
        hipsparselt_cout << "hipsparselt-bench INFO: stride_b < min_stride_b, set stride_b = "
                         << min_stride_b << std::endl;
        arg.stride_b = min_stride_b;
    }
    if(arg.stride_c == -1 || (!any_stride && arg.stride_c < min_stride_c))
    {
        hipsparselt_cout << "hipsparselt-bench INFO: stride_c < min_stride_c, set stride_c = "
                         << min_stride_c << std::endl;
        arg.stride_c = min_stride_c;
    }
    if(arg.stride_d == -1 || (!any_stride && arg.stride_d < min_stride_d))
    {
        hipsparselt_cout << "hipsparselt-bench INFO: stride_d < min_stride_d, set stride_d = "
                         << min_stride_d << std::endl;
        arg.stride_d = min_stride_d;
    }
    if(arg.bias_stride == -1
       || (!any_stride && arg.bias_stride < min_bias_stride && arg.bias_stride != 0))
    {
        hipsparselt_cout << "hipsparselt-bench INFO: bias_stride < min_bias_stride and bias_stride "
                            "!=0, set bias_stride = "
                         << min_bias_stride << std::endl;
        arg.bias_stride = min_bias_stride;
    }
    arg.norm_check_assert = false;
    hipsparselt_spmm_dispatch<perf_sparse>(arg);
    return 0;
}

int hipsparselt_bench_datafile(const std::string& filter, bool any_stride)
{
    int ret = 0;
    for(Arguments arg : HipSparseLt_TestData())
        ret |= run_bench_test(arg, filter, any_stride, true);
    test_cleanup::cleanup();
    return ret;
}

// Replace --batch with --batch_count for backward compatibility
void fix_batch(int argc, char* argv[])
{
    static char b_c[] = "--batch_count";
    for(int i = 1; i < argc; ++i)
        if(!strcmp(argv[i], "--batch"))
        {
            static int once
                = (hipsparselt_cerr << argv[0]
                                    << " warning: --batch is deprecated, and --batch_count "
                                       "should be used instead."
                                    << std::endl,
                   0);
            argv[i] = b_c;
        }
}

int main(int argc, char* argv[])
try
{
    fix_batch(argc, argv);
    Arguments   arg;
    std::string function;
    std::string precision;
    std::string a_type;
    std::string b_type;
    std::string c_type;
    std::string d_type;
    std::string compute_type;
    std::string bias_type;
    std::string initialization;
    std::string filter;
    std::string activation_type;
    char        order, order_a, order_b, order_c, order_d;
    int         device_id;
    int         flags             = 0;
    bool        datafile          = hipsparselt_parse_data(argc, argv);
    bool        log_function_name = false;
    bool        any_stride        = false;

    arg.init(); // set all defaults

    options_description desc("hipsparselt-bench command line options");
    desc.add_options()
        // clang-format off
        ("sizem,m",
         value<int64_t>(&arg.M)->default_value(128),
         "Specific matrix size: the number of rows or columns in matrix.")

        ("sizen,n",
         value<int64_t>(&arg.N)->default_value(128),
         "Specific matrix the number of rows or columns in matrix")

        ("sizek,k",
         value<int64_t>(&arg.K)->default_value(128),
         "Specific matrix size: the number of columns in A and rows in B.")

        ("lda",
         value<int64_t>(&arg.lda)->default_value(-1),
         "Leading dimension of matrix A.")

        ("ldb",
         value<int64_t>(&arg.ldb)->default_value(-1),
         "Leading dimension of matrix B.")

        ("ldc",
         value<int64_t>(&arg.ldc)->default_value(-1),
         "Leading dimension of matrix C.")

        ("ldd",
         value<int64_t>(&arg.ldd)->default_value(-1),
         "Leading dimension of matrix D.")

        ("any_stride",
         value<bool>(&any_stride)->default_value(false),
         "Do not modify input strides based on leading dimensions")

        ("stride_a",
         value<int64_t>(&arg.stride_a)->default_value(-1),
         "Specific stride of strided_batched matrix A, second dimension * leading dimension.")

        ("stride_b",
         value<int64_t>(&arg.stride_b)->default_value(-1),
         "Specific stride of strided_batched matrix B, second dimension * leading dimension.")

        ("stride_c",
         value<int64_t>(&arg.stride_c)->default_value(-1),
         "Specific stride of strided_batched matrix C, second dimension * leading dimension.")

        ("stride_d",
         value<int64_t>(&arg.stride_d)->default_value(-1),
         "Specific stride of strided_batched matrix D, second dimension * leading dimension.")

        ("alpha",
          value<float>(&arg.alpha)->default_value(1.0), "specifies the scalar alpha")

        ("beta",
         value<float>(&arg.beta)->default_value(0.0), "specifies the scalar beta")

        ("function,f",
         value<std::string>(&function),
         "BLAS function to test.")

        ("precision,r",
         value<std::string>(&precision)->default_value("f16_r"), "Precision. "
         "Options: h,f16_r,bf16_r,i8_r")

        ("a_type",
         value<std::string>(&a_type), "Precision of matrix A. "
        "Options: h,f16_r,bf16_r,i8_r")

        ("b_type",
         value<std::string>(&b_type), "Precision of matrix B. "
        "Options: h,f16_r,bf16_r,i8_r")

        ("c_type",
         value<std::string>(&c_type), "Precision of matrix C. "
         "Options: h,f16_r,bf16_r,i8_r")

        ("d_type",
         value<std::string>(&d_type), "Precision of matrix D. "
        "Options: h,f16_r,bf16_r,i8_r")

        ("compute_type",
         value<std::string>(&compute_type), "Precision of computation. "
         "Options: s,f32_r,i32_r")

        ("initialization",
         value<std::string>(&initialization)->default_value("hpl"),
         "Intialize with random integers, trig functions sin and cos, or hpl-like input. "
         "Options: rand_int, trig_float, hpl")

        ("transposeA",
         value<char>(&arg.transA)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose")

        ("transposeB",
         value<char>(&arg.transB)->default_value('N'),
         "N = no transpose, T = transpose, C = conjugate transpose")

        ("batch_count",
         value<int32_t>(&arg.batch_count)->default_value(1),
         "Number of matrices. Only applicable to batched and strided_batched routines")

        ("HMM",
         value<bool>(&arg.HMM)->default_value(false),
         "Parameter requesting the use of HipManagedMemory")

        ("verify,v",
         value<int8_t>(&arg.norm_check)->default_value(0),
         "Validate GPU results with CPU? 0 = No, 1 = Yes (default: No)")

        ("iters,i",
         value<int32_t>(&arg.iters)->default_value(10),
         "Iterations to run inside timing loop")

        ("cold_iters,j",
         value<int32_t>(&arg.cold_iters)->default_value(2),
         "Cold Iterations to run before entering the timing loop")

        ("algo",
         value<uint32_t>(&arg.algo)->default_value(0),
         "extended precision spmm algorithm")

        ("solution_index",
         value<int32_t>(&arg.solution_index)->default_value(0),
         "extended precision spmm solution index")

        ("prune_algo",
         value<uint32_t>(&arg.prune_algo)->default_value(1),
         "prune algo, 0: tile algo, 1: (default) stip algo")

        ("activation_type",
         value<std::string>(&activation_type)->default_value("none"),
         "Options: None, clippedrelu, gelu, relu")

        ("activation_arg1",
         value<float>(&arg.activation_arg1)->default_value(std::numeric_limits<float>::quiet_NaN()),
         "activation argument #1, when activation_type is clippedrelu, this argument used to be the threshold(default=0). when type is gelu, this argument used to be the gelu scaling. (default=1)")

        ("activation_arg2",
         value<float>(&arg.activation_arg2)->default_value(std::numeric_limits<float>::infinity()),
         "activation argument #2, when activation_type is clippedrelu, this argument used to be the upperbound.")

        ("bias_vector",
         bool_switch(&arg.bias_vector)->default_value(false),
         "Apply bias vector")

        ("bias_stride",
         value<int64_t>(&arg.bias_stride)->default_value(0),
         "Specific stride of strided_batched vector bias. (defalut:0, broadcast)")

        ("bias_type",
         value<std::string>(&bias_type), "Precision of bias. (default: f16_r - when input precision is f16_r, bf16_r - when input precision is bf16_r, f32_r - when input precision is i8_r ) "
         "Options: s,f32_r,h,f16_r,b,bf16_r")

        ("device",
         value<int>(&device_id)->default_value(0),
         "Set default device to be used for subsequent program runs")

        ("c_noalias_d",
         bool_switch(&arg.c_noalias_d)->default_value(false),
         "C and D are stored in separate memory")

        ("workspace",
         value<size_t>(&arg.user_allocated_workspace)->default_value(0),
         "Set fixed workspace memory size instead of using hipsparselt managed memory")

        ("search",
         bool_switch(&arg.search)->default_value(false),
         "Evaluates all available algorithms and find the fastest one.")

        ("search_iters",
         value<int32_t>(&arg.search_iters)->default_value(10),
         "Iterations to run inside timing loop of each algorithms when search is on. (default: 10)")

        ("sparse_b",
         bool_switch(&arg.sparse_b)->default_value(false),
         "Structurted Sparsity Matrix B (A is Dense Matrix)")

        ("func_version",
         value<int32_t>(&arg.func_version)->default_value(1),
         "Specify the API version of Prune/Compress's function")

        ("log_function_name",
         bool_switch(&log_function_name)->default_value(false),
         "Function name precedes other itmes.")

        ("function_filter",
         value<std::string>(&filter),
         "Simple strstr filter on function name only without wildcards")

        ("order",
         value<char>(&order)->default_value('C'),
         "C = Column Major, R = Row Major")

        ("order_a",
         value<char>(&order_a)->default_value('N'),
         "C = Column Major, R = Row Major")

        ("order_b",
         value<char>(&order_b)->default_value('N'),
         "C = Column Major, R = Row Major")

        ("order_c",
         value<char>(&order_c)->default_value('N'),
         "C = Column Major, R = Row Major")

        ("order_d",
         value<char>(&order_d)->default_value('N'),
         "C = Column Major, R = Row Major")

        ("alpha_vector_scaling",
         bool_switch(&arg.alpha_vector_scaling)->default_value(false),
         "Apply alpha vector scaling")

        ("help,h", "produces this help message")

        ("version", "Prints the version number");
    // clang-format on

    // parse command line into arg structure and stack variables using desc
    variables_map vm;
    store(parse_command_line(argc, argv, desc), vm);
    notify(vm);

    if((argc <= 1 && !datafile) || vm.count("help"))
    {
        hipsparselt_cout << desc << std::endl;
        return 0;
    }

    if(vm.find("version") != vm.end())
    {
        int                      version;
        hipsparselt_local_handle handle;
        hipsparseLtGetVersion(handle, &version);

        char rev[128];
        hipsparseLtGetGitRevision(handle, &rev[0]);
        hipsparselt_cout << "hipSPARSELt version: " << version << " revision: " << rev << "\n" << std::endl;
        return 0;
    }

    // transfer local variable state
    ArgumentModel_set_log_function_name(log_function_name);

    // Device Query
    int64_t device_count = query_device_property();

    hipsparselt_cout << std::endl;
    if(device_count <= device_id)
        throw std::invalid_argument("Invalid Device ID");
    set_device(device_id);

    if(datafile)
        return hipsparselt_bench_datafile(filter, any_stride);

    // single bench run

    // validate arguments

    std::transform(precision.begin(), precision.end(), precision.begin(), ::tolower);
    auto prec = string_to_hip_datatype(precision);
    if(prec == static_cast<hipDataType>(-1))
        throw std::invalid_argument("Invalid value for --precision " + precision);

    arg.a_type = a_type == "" ? prec : string_to_hip_datatype(a_type);
    if(arg.a_type == static_cast<hipDataType>(-1))
        throw std::invalid_argument("Invalid value for --a_type " + a_type);

    arg.b_type = b_type == "" ? prec : string_to_hip_datatype(b_type);
    if(arg.b_type == static_cast<hipDataType>(-1))
        throw std::invalid_argument("Invalid value for --b_type " + b_type);

    arg.c_type = c_type == "" ? prec : string_to_hip_datatype(c_type);
    if(arg.c_type == static_cast<hipDataType>(-1))
        throw std::invalid_argument("Invalid value for --c_type " + c_type);

    arg.d_type = d_type == "" ? prec : string_to_hip_datatype(d_type);
    if(arg.d_type == static_cast<hipDataType>(-1))
        throw std::invalid_argument("Invalid value for --d_type " + d_type);

    bool is_f16      = arg.a_type == HIP_R_16F || arg.a_type == HIP_R_16BF;
    bool is_f32      = arg.a_type == HIP_R_32F;
    bool is_f8       = arg.a_type == HIP_R_8F_E4M3 || arg.a_type == HIP_R_8F_E5M2;
    arg.compute_type = compute_type == ""
#ifdef __HIP_PLATFORM_AMD__
                           ? (is_f16 || is_f8 ? HIPSPARSELT_COMPUTE_32F : HIPSPARSELT_COMPUTE_32I)
#else
                           ? (is_f16   ? HIPSPARSELT_COMPUTE_16F
                              : is_f32 ? HIPSPARSELT_COMPUTE_TF32
                              : is_f8  ? HIPSPARSELT_COMPUTE_32F
                                       : HIPSPARSELT_COMPUTE_32I)
#endif
                           : string_to_hipsparselt_computetype(compute_type);
    if(arg.compute_type == static_cast<hipsparseLtComputetype_t>(-1))
        throw std::invalid_argument("Invalid value for --compute_type " + compute_type);

    if(bias_type == "")
    {
#ifdef __HIP_PLATFORM_AMD__
        switch(arg.a_type)
        {
        case HIP_R_16BF:
        case HIP_R_16F:
            arg.bias_type = arg.a_type;
            break;
        default:
            arg.bias_type = HIP_R_32F;
            break;
        }
#endif
#ifdef __HIP_PLATFORM_NVIDIA__
        arg.bias_type
            = (arg.a_type == HIP_R_8I ? HIP_R_32F : arg.c_type);
#endif
    }
    else
    {
        arg.bias_type = string_to_hip_datatype(bias_type);

        bool valid = false;
#ifdef __HIP_PLATFORM_AMD__
        switch(arg.a_type)
        {
        case HIP_R_16BF:
        case HIP_R_16F:
            valid = arg.bias_type == arg.a_type || arg.bias_type == HIP_R_32F;
            break;
        default:
            valid = arg.bias_type == HIP_R_32F;
            break;
        }
#endif
#ifdef __HIP_PLATFORM_NVIDIA__
        valid = (arg.a_type == HIP_R_8I && arg.bias_type == HIP_R_32F) ||
                (arg.a_type != HIP_R_8I && arg.bias_type == arg.c_type);
#endif
        if(!valid)
            throw std::invalid_argument("Invalid value for --bias_type " + bias_type);
    }

    arg.initialization = string2hipsparselt_initialization(initialization);
    if(arg.initialization == static_cast<hipsparselt_initialization>(-1))
        throw std::invalid_argument("Invalid value for --initialization " + initialization);

    arg.activation_type = string_to_hipsparselt_activation_type(activation_type);
    if(arg.activation_type == static_cast<hipsparselt_activation_type>(-1))
        throw std::invalid_argument("Invalid value for --activation_type " + activation_type);
    if(std::isnan(arg.activation_arg1))
    {
        if(arg.activation_type == hipsparselt_activation_type::gelu)
            arg.activation_arg1 = 1.f;
        else
            arg.activation_arg1 = 0.f;
    }

    if(arg.M < 0)
        throw std::invalid_argument("Invalid value for -m " + std::to_string(arg.M));
    if(arg.N < 0)
        throw std::invalid_argument("Invalid value for -n " + std::to_string(arg.N));
    if(arg.K < 0)
        throw std::invalid_argument("Invalid value for -k " + std::to_string(arg.K));

    int copied = snprintf(arg.function, sizeof(arg.function), "%s", function.c_str());
    if(copied <= 0 || copied >= sizeof(arg.function))
        throw std::invalid_argument("Invalid value for --function");

    if(order != 'C' && order != 'R')
        throw std::invalid_argument("Invalid value for --order " + std::to_string(order));
    if(order_a != 'C' && order_a != 'R')
    {
        if(order_a == 'N')
            order_a = order;
        else
            throw std::invalid_argument("Invalid value for --order_a " + std::to_string(order_a));
    }
    if(order_b != 'C' && order_b != 'R')
    {
        if(order_b == 'N')
            order_b = order;
        else
            throw std::invalid_argument("Invalid value for --order_b " + std::to_string(order_b));
    }
    if(order_c != 'C' && order_c != 'R')
    {
        if(order_c == 'N')
            order_c = order;
        else
            throw std::invalid_argument("Invalid value for --order_c " + std::to_string(order_c));
    }
    if(order_d != 'C' && order_d != 'R')
    {
        if(order_d == 'N')
            order_d = order;
        else
            throw std::invalid_argument("Invalid value for --order_d " + std::to_string(order_d));
    }

    arg.orderA = order != order_a ? order_a : order;
    arg.orderB = order != order_b ? order_b : order;
    arg.orderC = order != order_c ? order_c : order;
    arg.orderD = order != order_d ? order_d : order;

    return run_bench_test(arg, filter, any_stride);
}
catch(const std::invalid_argument& exp)
{
    hipsparselt_cerr << exp.what() << std::endl;
    return -1;
}
