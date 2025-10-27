// Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "../hipfftw_helper.h"

#include "../../shared/environment.h"
#include "../../shared/gpubuf.h"
#include "../../shared/hostbuf.h"
#include "../../shared/params_gen.h"
#include "../../shared/test_params.h"

#include <cstdint>
#include <cstring>
#include <fftw3.h>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <numeric>

#ifdef _OPENMP
#include <omp.h>
#endif

extern size_t max_length_for_hipfftw_test;
extern size_t max_io_gb_for_hipfftw_test;

// test details
namespace
{
    //
    //---------------------------------------------------------------------------------------------
    //                                  COMMONS AND HELPERS
    //---------------------------------------------------------------------------------------------
    //

    size_t max_byte_size_for_hipfftw_tests()
    {
        auto get_io_byte_size_limit = []() {
            size_t tmp = vramgb * ONE_GiB;
            if(tmp == 0)
            {
                size_t free = 0, total = 0;
                if(hipMemGetInfo(&free, &total) == hipSuccess)
                    tmp = total / 8;
            }
            tmp = std::min(tmp, ramgb * ONE_GiB);
            tmp = std::min(tmp, max_io_gb_for_hipfftw_test * ONE_GiB);
            if(verbose > 0)
            {
                std::cout << "Limit for the size of I/O data used in hipfftw tests: ";
                if(tmp >= ONE_GiB)
                    std::cout << float(tmp) / ONE_GiB << " GiB." << std ::endl;
                else if(tmp >= ONE_MiB)
                    std::cout << float(tmp) / ONE_MiB << " MiB." << std ::endl;
                else
                    std::cout << float(tmp) / ONE_KiB << " KiB." << std ::endl;
            }
            return tmp;
        };
        static const size_t io_byte_size_limit = get_io_byte_size_limit();
        return io_byte_size_limit;
    }

    std::ranlux24_base& get_pseudo_rng()
    {
        static std::ranlux24_base gen(random_seed);
        return gen;
    }

    // NOTE: this function makes use of comparison operator < and != which must be defined for the
    // specialization type T
    template <typename T>
    void insert_into_unique_sorted_params(std::vector<T>& unique_sorted_params,
                                          const T&        param_to_insert)
    {
        auto it = std::lower_bound(
            unique_sorted_params.begin(), unique_sorted_params.end(), param_to_insert);
        if(it != unique_sorted_params.end() && *it == param_to_insert)
            return; // it's already in there generated
        unique_sorted_params.insert(it, param_to_insert);
    }

    enum class hipfftw_internal_exception
    {
        none,
        flow_redirection,
        invalid_args,
        unsupported_args,
        ill_defined
    };
    // for well-defined internal exceptions, we may expect a specific report thereof
    // in the hipfftw exception log (if logging is activated)
    template <hipfftw_internal_exception>
    constexpr std::string_view hipfftw_expected_log_instance;
    template <>
    constexpr std::string_view hipfftw_expected_log_instance<
        hipfftw_internal_exception::invalid_args> = R"(Invalid argument reported)";
    template <>
    constexpr std::string_view hipfftw_expected_log_instance<
        hipfftw_internal_exception::unsupported_args> = R"(Unsupported usage reported)";
    template <>
    constexpr std::string_view hipfftw_expected_log_instance<
        hipfftw_internal_exception::flow_redirection> = R"(Redirecting execution flow)";

    // randomizers
    // Note: albeit not supported, ranks > 3 are "valid" rank argument
    // --> limiting rank value to max of 10 by default to avoid ridiculously long
    // lengths possibly created in automated parameter generations;
    template <bool validity_flag,
              int  min_rank          = validity_flag ? 1 : std::numeric_limits<int>::lowest(),
              int  max_rank          = validity_flag ? 10 : 0,
              std::enable_if_t<(min_rank <= max_rank) && (!validity_flag || min_rank > 0)
                                   && (validity_flag || max_rank <= 0),
                               bool> = true>
    int get_random_rank()
    {
        static std::uniform_int_distribution<int> rank_rng(min_rank, max_rank);

        auto ret = rank_rng(get_pseudo_rng());
        if(rank_is_valid_for_hipfftw(ret) != validity_flag)
        {
            throw std::runtime_error(
                "failed to generate a rank value of desired validity randomly");
        }
        return ret;
    }
    template <bool validity_flag, typename type_to_consider_for_validity>
    std::vector<ptrdiff_t>
        get_random_lengths(int       desired_rank,
                           ptrdiff_t max_abs_len
                           = std::numeric_limits<type_to_consider_for_validity>::max(),
                           ptrdiff_t min_abs_len = 0)
    {
        std::vector<ptrdiff_t> ret;
        // cannot generate lengths for invalid ranks --> return empty lengths in that case
        if(!rank_is_valid_for_hipfftw(desired_rank))
            return ret;
        if(min_abs_len < 0 || min_abs_len > max_abs_len)
            throw std::invalid_argument("invalid bounds used for get_random_lengths");
        // generate values that are all representable as integers
        auto&                                    pseudo_rng = get_pseudo_rng();
        std::uniform_int_distribution<ptrdiff_t> length_rng(min_abs_len, max_abs_len);
        // setter lambda
        auto set_random_len = [&]() {
            for(auto& l : ret)
            {
                const ptrdiff_t val = length_rng(pseudo_rng);
                if constexpr(validity_flag)
                    l = val;
                else
                {
                    if(pseudo_rng() % 2)
                        l = -val;
                    else
                        l = val;
                }
            }
        };

        ret.resize(desired_rank);
        set_random_len();
        while(lengths_are_valid_for_hipfftw_as<type_to_consider_for_validity>(ret, desired_rank)
              != validity_flag)
            set_random_len();

        return ret;
    }
    template <bool validity_flag>
    int get_random_sign(fft_transform_type intended_dft_kind)
    {
        if(!validity_flag && is_real(intended_dft_kind))
            throw std::invalid_argument("An invalid sign cannot be generated for real transforms "
                                        "(sign is ignored in that case)");
        std::uniform_int_distribution<int> sign_rng(std::numeric_limits<int>::lowest(),
                                                    std::numeric_limits<int>::max());

        int tmp = validity_flag && is_complex(intended_dft_kind)
                      ? (is_fwd(intended_dft_kind) ? FFTW_FORWARD : FFTW_BACKWARD)
                      : sign_rng(get_pseudo_rng());

        while(sign_is_valid_for_hipfftw(tmp, intended_dft_kind) != validity_flag)
            tmp = sign_rng(get_pseudo_rng());
        return tmp;
    }
    template <bool validity_flag>
    unsigned get_random_flags()
    {
        std::uniform_int_distribution<unsigned> flags_rng(std::numeric_limits<unsigned>::lowest(),
                                                          std::numeric_limits<unsigned>::max());

        auto tmp = flags_rng(get_pseudo_rng());
        if constexpr(validity_flag)
        {
            tmp &= hipfftw_valid_flags_mask;
            if(!flags_are_valid_for_hipfftw(tmp))
                throw std::runtime_error("failed to create random valid flags");
            return tmp;
        }
        while(flags_are_valid_for_hipfftw(tmp))
            tmp = flags_rng(get_pseudo_rng());
        return tmp;
    }
    size_t get_random_idx(size_t upper_bound)
    {
        if(upper_bound == 0)
            throw std::invalid_argument("upper_bound must be strictly positive for get_random_idx");
        std::uniform_int_distribution<size_t> idx_rng(0, upper_bound - 1);
        return idx_rng(get_pseudo_rng());
    }

    // calculates the threshold value X such that the number of relevant elements
    // is no greater (resp. larger) than num_elems, if all elements of lengths are
    // all no greater (resp. larger) than X, and lengths.size() == rank [using bisection]
    ptrdiff_t get_len_threshold(size_t num_elems, int rank, bool is_real_inplace)
    {
        if(rank < 1)
            throw std::invalid_argument("invalid rank used in get_len_threshold");
        if(num_elems == 0)
            return 1;
        constexpr ptrdiff_t X_max = std::numeric_limits<ptrdiff_t>::max();
        // we need to find X in [0, X_max] s.t.
        // largest_idx(X) <= num_elems && largest_idx(X + 1) > num_elems
        auto largest_idx = [&](ptrdiff_t X) {
            size_t ret = rank > 1 && is_real_inplace ? 2 * (X / 2 + 1) : X;
            for(auto i = 1; i < rank; i++)
                ret *= X;
            return ret;
        };
        // initialization
        ptrdiff_t X_down
            = rank == 1 ? static_cast<ptrdiff_t>(num_elems)
                        : static_cast<ptrdiff_t>(std::floor(std::pow(num_elems, 1.0 / rank)));
        ptrdiff_t diff = 1;
        ptrdiff_t X_up = X_down;
        while(largest_idx(X_up) <= num_elems && X_up < X_max)
        {
            X_down = X_up;
            X_up   = X_up <= X_max - diff ? X_up + diff : X_max;
            diff *= 2;
        }
        diff = 1;
        while(largest_idx(X_down) > num_elems && X_down > 0)
        {
            X_up   = X_down;
            X_down = X_down >= diff ? X_down - diff : 0;
            diff *= 2;
        }
        // bisection
        while(X_up - X_down > 1)
        {
            const auto tmp = (X_up + X_down) / 2;
            if(largest_idx(tmp) <= num_elems)
                X_down = tmp;
            else
                X_up = tmp;
        }
        return X_down;
    }

    template <fft_precision prec>
    size_t max_num_elems_for_data_size(size_t data_byte_size, fft_transform_type dft_kind)
    {
        return data_byte_size
               / (is_real(dft_kind) ? sizeof(hipfftw_real_t<prec>)
                                    : sizeof(hipfftw_complex_t<prec>));
    }

    // exception for hip runtime error(s) specifically
    struct hip_runtime_error : public std::runtime_error
    {
        const hipError_t hip_error;
        hip_runtime_error(const std::string& info, hipError_t hip_status)
            : std::runtime_error::runtime_error(info)
            , hip_error(hip_status)

        {
            ++n_hip_failures;
        }
    };
    int get_current_device_id()
    {
        int        ret        = hipInvalidDeviceId;
        const auto hip_status = hipGetDevice(&ret);
        if(hip_status != hipSuccess)
            throw hip_runtime_error("hipGetDevice failed.", hip_status);
        return ret;
    }

    //
    //---------------------------------------------------------------------------------------------
    //                             EXISTENCE OF UTILITY FUNCTIONS
    //---------------------------------------------------------------------------------------------
    //

    template <fft_precision prec>
    void test_existence_of_utility_functions()
    {
        try
        {
            // call utility functions - they need to exist but don't need to work
            const auto& hipfftw_ = hipfftw_funcs<prec>::get_instance();
            hipfftw_.print_plan(nullptr);
            hipfftw_.set_timelimit(0.0);
            hipfftw_.cost(nullptr);
            hipfftw_.flops(nullptr, nullptr, nullptr, nullptr);
            hipfftw_.cleanup();
        }
        catch(const hipfftw_undefined_function_ptr& e)
        {
            GTEST_FAIL() << "Undefined function pointers detected. Error info: " << e.what();
        }
        catch(...)
        {
            GTEST_FAIL() << "Unexpected failure";
        }
    }

    //
    //---------------------------------------------------------------------------------------------
    //                                   ALLOCATION AND FREE
    //---------------------------------------------------------------------------------------------
    //

    enum class hipfftw_alloc_func_type
    {
        unspecified,
        real,
        complex
    };
    bool hipfftw_alloc_func_type_is_valid(hipfftw_alloc_func_type func)
    {
        return func == hipfftw_alloc_func_type::unspecified || func == hipfftw_alloc_func_type::real
               || func == hipfftw_alloc_func_type::complex;
    }
    // bit mask to enable/disable allocation kind(s), listed by increasing "rank"
    // to enable meaningful comparison operators (implicitly defined based on the underlying type)
    enum hipfftw_alloc_memkind : unsigned
    {
        none          = 0x0,
        pageable_host = 0x1 << 0,
        pinned_host   = 0x1 << 1,
        any           = pageable_host | pinned_host
    };

    const std::vector<hipfftw_alloc_memkind> hipfftw_possible_memkinds
        = {pinned_host, pageable_host};

    bool hipfftw_alloc_kind_is_valid(hipfftw_alloc_memkind kind)
    {
        return kind == (kind & hipfftw_alloc_memkind::any);
    }
    std::string hipfftw_alloc_kind_to_string(hipfftw_alloc_memkind kind)
    {
        if(!hipfftw_alloc_kind_is_valid(kind))
            throw std::invalid_argument("alloc_kind_to_string: invalid kind");
        if(kind == hipfftw_alloc_memkind::none)
            return "none";
        if(std::find(hipfftw_possible_memkinds.begin(), hipfftw_possible_memkinds.end(), kind)
           == hipfftw_possible_memkinds.end())
        {
            // several values enabled
            std::string ret;
            for(auto to_consider : hipfftw_possible_memkinds)
            {
                if(!(kind & to_consider))
                    continue;
                if(!ret.empty())
                    ret += "_or_";
                ret += hipfftw_alloc_kind_to_string(to_consider);
            }
            return ret;
        }
        // kind is a well-defined value in hipfftw_possible_memkinds
        switch(kind)
        {
        case hipfftw_alloc_memkind::pinned_host:
            return "pinned_host";
            break;
        case hipfftw_alloc_memkind::pageable_host:
            return "pageable_host";
            break;
        default:
            throw std::runtime_error("alloc_kind_to_string: internal error encountered "
                                     "(unexpected value for kind)");
            break;
        }
        // unreachable
    }

    template <fft_precision prec>
    struct hipfftw_malloc_params
    {
        size_t                  alloc_arg;
        hipfftw_alloc_func_type alloc_func;
        hipfftw_alloc_memkind   alloc_kind;

        size_t get_byte_size() const
        {
            return alloc_arg
                   * (alloc_func == hipfftw_alloc_func_type::unspecified
                          ? sizeof(char)
                          : (alloc_func == hipfftw_alloc_func_type::real
                                 ? sizeof(hipfftw_real_t<prec>)
                                 : sizeof(hipfftw_complex_t<prec>)));
        }

        std::string to_string() const
        {
            if(!hipfftw_alloc_func_type_is_valid(alloc_func))
                throw std::runtime_error("invalid type of allocation function");
            if(!hipfftw_alloc_kind_is_valid(alloc_kind))
                throw std::runtime_error("invalid allocation kind(s)");

            std::string ret;
            if constexpr(prec == fft_precision_single)
                ret += "fftwf_";
            else
                ret += "fftw_";

            if(alloc_func == hipfftw_alloc_func_type::unspecified)
                ret += "malloc_";
            else if(alloc_func == hipfftw_alloc_func_type::real)
                ret += "alloc_real_";
            else
                ret += "alloc_complex_";
            ret += std::to_string(alloc_arg);
            ret += "_alloc_kind_" + hipfftw_alloc_kind_to_string(alloc_kind);
            return ret;
        }
        // for using with insert_into_unique_sorted_params
        bool operator<(const hipfftw_malloc_params& other) const
        {
            return to_string() < other.to_string();
        }
        bool operator==(const hipfftw_malloc_params& other) const
        {
            return to_string() == other.to_string();
        }
    };

    template <fft_precision prec>
    std::vector<hipfftw_malloc_params<prec>> params_for_testing_hipfftw_malloc()
    {
        std::vector<hipfftw_malloc_params<prec>> ret;
        // testing argument value 0 and a randomly chosen one (max 64MiB in byte size, arbitrarily chosen)
        constexpr size_t max_test_alloc_size = 1ULL << 26;

        const std::vector<hipfftw_alloc_func_type> func_range
            = {hipfftw_alloc_func_type::unspecified,
               hipfftw_alloc_func_type::real,
               hipfftw_alloc_func_type::complex};
        std::vector<hipfftw_alloc_memkind> memkind_range;
        // add all possible combinations of memory kinds:
        for(auto kind = static_cast<std::underlying_type<hipfftw_alloc_memkind>::type>(
                hipfftw_alloc_memkind::none);
            kind <= static_cast<std::underlying_type<hipfftw_alloc_memkind>::type>(
                hipfftw_alloc_memkind::any);
            kind++)
        {
            memkind_range.push_back(static_cast<hipfftw_alloc_memkind>(kind));
        }

        hipfftw_malloc_params<prec> to_add;
        for(auto func : func_range)
        {
            size_t max_arg = max_test_alloc_size;
            if(func == hipfftw_alloc_func_type::real)
                max_arg /= sizeof(hipfftw_real_t<prec>);
            else if(func == hipfftw_alloc_func_type::complex)
                max_arg /= sizeof(hipfftw_complex_t<prec>);
            std::uniform_int_distribution<size_t> arg_rng(1, max_arg);
            for(auto kind : memkind_range)
            {
                for(auto arg : {size_t(0), arg_rng(get_pseudo_rng())})
                {
                    to_add.alloc_arg  = arg;
                    to_add.alloc_func = func;
                    to_add.alloc_kind = kind;
                    insert_into_unique_sorted_params(ret, to_add);
                }
            }
        }
        return ret;
    }

    template <fft_precision prec>
    class hipfftw_allocation_test : public ::testing::TestWithParam<hipfftw_malloc_params<prec>>
    {
    protected:
        void* test_allocation = nullptr;
        bool  expect_no_allocation;
        std::map<hipfftw_alloc_memkind, std::unique_ptr<EnvironmentSetTemp>> temp_alloc_limit_env;

        void SetUp() override
        {
            if(test_allocation)
                GTEST_FAIL() << "Starting from an unclean slate (test_allocation is not nullptr)";
            const hipfftw_malloc_params<prec>& params = this->GetParam();
            // check validity of params
            if(!hipfftw_alloc_kind_is_valid(params.alloc_kind))
                GTEST_FAIL() << "invalid value for allocation kind";
            if(!hipfftw_alloc_func_type_is_valid(params.alloc_func))
                GTEST_FAIL() << "unknown allocation function";

            size_t limit_for_alloc_kind = 0;
            for(auto alloc_kind_candidate : hipfftw_possible_memkinds)
            {
                if(alloc_kind_candidate != hipfftw_alloc_memkind::pinned_host
                   && alloc_kind_candidate != hipfftw_alloc_memkind::pageable_host)
                {
                    throw std::runtime_error("unexpected memory allocation kind "
                                             + hipfftw_alloc_kind_to_string(alloc_kind_candidate));
                }
                const std::string control_env_var
                    = alloc_kind_candidate == hipfftw_alloc_memkind::pinned_host
                          ? "HIPFFTW_BYTE_SIZE_LIMIT_PINNED_HOST_ALLOC"
                          : "HIPFFTW_BYTE_SIZE_LIMIT_PAGEABLE_HOST_ALLOC";

                if(alloc_kind_candidate & params.alloc_kind)
                {
                    const auto test_user_limit = rocfft_getenv(control_env_var.c_str());
                    limit_for_alloc_kind
                        = std::max(limit_for_alloc_kind,
                                   test_user_limit.empty() ? std::numeric_limits<size_t>::max()
                                                           : size_t(std::stoull(test_user_limit)));
                }
                else
                {
                    // disable the other possible allocation kind(s) by temporarily
                    // setting the corresponding byte size limit to 0
                    temp_alloc_limit_env[alloc_kind_candidate]
                        = std::make_unique<EnvironmentSetTemp>(control_env_var.c_str(), "0");
                    // skip if temporary limit(s) was(were) not successfully set
                    const auto tmp_limit = rocfft_getenv(control_env_var.c_str());
                    if(tmp_limit.empty() || std::stoull(tmp_limit) != 0)
                    {
                        GTEST_SKIP() << "failed to set environment variable disabling "
                                     << hipfftw_alloc_kind_to_string(alloc_kind_candidate)
                                     << " allocation(s) by hipFFTW";
                    }
                }
            }
            const size_t req_byte_size = params.get_byte_size();
            expect_no_allocation       = params.alloc_kind == hipfftw_alloc_memkind::none
                                   || req_byte_size == 0 || req_byte_size > limit_for_alloc_kind;
        }
        void TearDown() override
        {
            temp_alloc_limit_env.clear();
            const hipfftw_funcs<prec>& hipfftw_impl = hipfftw_funcs<prec>::get_instance();
            if(test_allocation && !hipfftw_impl.free.may_be_used())
                GTEST_FAIL() << "An allocation was created but it can't be freed";
            // note: free should be stable even with nullptr
            if(hipfftw_impl.free.may_be_used())
                hipfftw_impl.free(test_allocation);
        }

        void test_malloc_write_and_read()
        {
            const hipfftw_malloc_params<prec>& params       = this->GetParam();
            const hipfftw_funcs<prec>&         hipfftw_impl = hipfftw_funcs<prec>::get_instance();
            hipfftw_exception_logger           exception_logger;

            struct allocation_test_to_be_skipped : std::runtime_error
            {
                using std::runtime_error::runtime_error;
            };
            struct allocation_test_failed : std::runtime_error
            {
                using std::runtime_error::runtime_error;
            };
            struct allocation_test_success
            {
            }; // used to cut test execution short when applicable

            try
            {
                // The test
                // - fills values in the allocated arrays as 0, 1, ..., max_elem, 0, 1, ..., max_elem, 0, 1, etc.
                //   (max_elem, max_elem-1, ..., 1, 0, max_elem, max_elem-1, ..., for imaginary values)
                // - reads the values thereafter and accumulates them as a sum of doubles
                // - checks the result.
                // --> the expected_result's must be exactly representable as double values
                const size_t max_elem = static_cast<size_t>(std::numeric_limits<uint8_t>::max());
                const size_t cycle_sz = max_elem + 1;
                const size_t max_representable_result
                    = (1ULL << std::numeric_limits<double>::digits) - 1;
                auto sum_of_integers = [](size_t to, size_t from = 0) {
                    if(from > to)
                        throw std::invalid_argument("invalid argument for sum_of_integers lambda");
                    return (to + from) * (to - from + 1) / 2;
                };
                const size_t sum_of_cycles
                    = (params.alloc_arg / cycle_sz) * sum_of_integers(max_elem);
                const size_t tail_sz = params.alloc_arg % cycle_sz;
                const size_t expected_result_r
                    = sum_of_cycles + (tail_sz > 0 ? sum_of_integers(tail_sz - 1) : 0);
                const size_t expected_result_i
                    = sum_of_cycles
                      + (tail_sz > 0 ? sum_of_integers(max_elem, max_elem + 1 - tail_sz) : 0);
                if(expected_result_r > max_representable_result
                   || (params.alloc_func == hipfftw_alloc_func_type::complex
                       && expected_result_i > max_representable_result))
                    throw allocation_test_to_be_skipped("Test cannot reliably check for argument "
                                                        + std::to_string(params.alloc_arg));

                if(params.alloc_func == hipfftw_alloc_func_type::unspecified)
                    test_allocation = hipfftw_impl.malloc(params.alloc_arg);
                else if(params.alloc_func == hipfftw_alloc_func_type::real)
                    test_allocation = hipfftw_impl.alloc_real(params.alloc_arg);
                else
                    test_allocation = hipfftw_impl.alloc_complex(params.alloc_arg);

                // check that the allocation behaved as expected
                if(expect_no_allocation)
                {
                    if(!test_allocation)
                        throw allocation_test_success();
                    else
                        // no allocation should have happened, nullptr should have been returned
                        throw allocation_test_failed(
                            "An allocation was unexpectedly produced for this test");
                }
                if(!test_allocation)
                {
                    throw allocation_test_failed("allocation failed");
                }
                // check that the allocation is of the expected type
                hipPointerAttribute_t attributes;
                auto hip_status = hipPointerGetAttributes(&attributes, test_allocation);
                if(hip_status != hipSuccess)
                    throw hip_runtime_error("hipPointerGetAttributes failed.", hip_status);
                switch(attributes.type)
                {
                case hipMemoryType::hipMemoryTypeHost:
                    EXPECT_NE(params.alloc_kind & hipfftw_alloc_memkind::pinned_host, 0);
                    break;
                case hipMemoryType::hipMemoryTypeUnregistered:
                    EXPECT_NE(params.alloc_kind & hipfftw_alloc_memkind::pageable_host, 0);
                    break;
                default:
                    GTEST_FAIL() << "Unexpected kind of memory created: attributes.type = "
                                 << attributes.type;
                    break;
                }

                // check that the host can write to the entire allocation
#ifdef _OPENMP
#pragma omp parallel for
#endif
                for(size_t idx = 0; idx < params.alloc_arg; idx++)
                {
                    uint8_t val = static_cast<uint8_t>(idx % cycle_sz);
                    if(params.alloc_func
                       == hipfftw_alloc_func_type::unspecified) // write as uint8_t
                        static_cast<uint8_t*>(test_allocation)[idx] = val;
                    else if(params.alloc_func
                            == hipfftw_alloc_func_type::real) // write as float/double
                        static_cast<hipfftw_real_t<prec>*>(test_allocation)[idx] = val;
                    else // write as complex value of req. precision
                    {
                        static_cast<hipfftw_complex_t<prec>*>(test_allocation)[idx][0] = val;
                        static_cast<hipfftw_complex_t<prec>*>(test_allocation)[idx][1]
                            = max_elem - val;
                    }
                }
                // check that the host can read from the entire allocation
                double result[2] = {0, 0};
#ifdef _OPENMP
#pragma omp parallel for reduction(+ : result)
#endif
                for(size_t idx = 0; idx < params.alloc_arg; idx++)
                {
                    if(params.alloc_func == hipfftw_alloc_func_type::unspecified)
                    {
                        // read as uint8_t, accumulate as double
                        result[0] += static_cast<uint8_t*>(test_allocation)[idx];
                    }
                    else if(params.alloc_func == hipfftw_alloc_func_type::real)
                    {
                        // read as float/double, accumulate as double
                        result[0] += static_cast<hipfftw_real_t<prec>*>(test_allocation)[idx];
                    }
                    else // write as complex value of req. precision
                    {
                        // read as complex value of req. precision, accumulate as doubles
                        result[0] += static_cast<hipfftw_complex_t<prec>*>(test_allocation)[idx][0];
                        result[1] += static_cast<hipfftw_complex_t<prec>*>(test_allocation)[idx][1];
                    }
                }
                // validity checks
                if(result[0] != expected_result_r)
                    throw allocation_test_failed("incorrect result for accumulated real parts");
                if(params.alloc_func == hipfftw_alloc_func_type::complex
                   && result[1] != expected_result_i)
                    throw allocation_test_failed(
                        "incorrect result for accumulated imaginary parts");
            }
            catch(const allocation_test_success&)
            {
                // so far so good
            }
            catch(const hipfftw_undefined_function_ptr& e)
            {
                GTEST_FAIL() << "undefined function pointers detected. Error info: " << e.what();
            }
            catch(const hip_runtime_error& e)
            {
                if(skip_runtime_fails)
                    GTEST_SKIP() << e.what() << "\nError code: " << e.hip_error << ".";
                else
                    GTEST_FAIL() << e.what() << "\nError code: " << e.hip_error << ".";
            }
            catch(const allocation_test_to_be_skipped& e)
            {
                GTEST_SKIP() << e.what();
            }
            catch(const allocation_test_failed& e)
            {
                std::ostringstream gtest_info;
                gtest_info << e.what();
                const auto log_content = exception_logger.get_log();
                if(!log_content.empty())
                    gtest_info << "\nContent of error log :\n " << log_content;
                GTEST_FAIL() << gtest_info.str();
            }
            catch(...)
            {
                std::ostringstream gtest_info;
                gtest_info << "unidentified exception caught during test.";
                const auto log_content = exception_logger.get_log();
                if(!log_content.empty())
                    gtest_info << "\nContent of error log :\n " << log_content;
                GTEST_FAIL() << gtest_info.str();
            }

            // pinned host allocation is the first-ranked choice of hipfftw
            // check that the execution flow was redirected if disabled
            for(auto possible_memkind : hipfftw_possible_memkinds)
            {
                if(possible_memkind <= params.alloc_kind)
                    continue;
                // possible_memkind is higher ranked than the test's targeted kind
                // flow re-direction must have happened
                if(test_allocation && exception_logger.is_active())
                {
                    const auto log_content = exception_logger.get_log();
                    if(log_content.find(hipfftw_expected_log_instance<
                                        hipfftw_internal_exception::flow_redirection>)
                       == std::string::npos)
                    {
                        GTEST_FAIL() << "No instance of \""
                                     << hipfftw_expected_log_instance<
                                            hipfftw_internal_exception::
                                                flow_redirection> << "\" in log despite "
                                     << hipfftw_alloc_kind_to_string(possible_memkind)
                                     << " allocation kind supposedly "
                                        "disabled via environment variable.\nContent of log:\n"
                                     << log_content;
                    }
                    else
                        break;
                }
            }
        }

    public:
        static std::string TestName(
            const testing::TestParamInfo<typename hipfftw_allocation_test::ParamType>& info)
        {
            return info.param.to_string();
        }
    };

    using allocation_sp = hipfftw_allocation_test<fft_precision_single>;
    TEST_P(allocation_sp, malloc_write_and_read)
    {
        test_malloc_write_and_read();
    }
    using allocation_dp = hipfftw_allocation_test<fft_precision_double>;
    TEST_P(allocation_dp, malloc_write_and_read)
    {
        test_malloc_write_and_read();
    }

    //
    //---------------------------------------------------------------------------------------------
    //                    INPUT VALIDATION FOR PLAN CREATION AND EXECUTION
    //---------------------------------------------------------------------------------------------
    //

    // bit-flagging enum used for configuring tests's execution I/O
    enum hipfftw_execution_io_args : unsigned
    {
        use_creation_io       = 0x0,
        non_null_new_in       = 0x1 << 0,
        non_null_new_out      = 0x1 << 1,
        new_io_same_placement = 0x1 << 2,
        // all flags must be up to be generally clean with new I/O
        clean_new_io = non_null_new_in | non_null_new_out | new_io_same_placement
    };

    static bool hipfftw_execution_io_args_are_well_defined(hipfftw_execution_io_args args)
    {
        return args == (args & hipfftw_execution_io_args::clean_new_io);
    }

    enum class hipfftw_step
    {
        plan_creation,
        plan_execution
    };

    template <fft_precision prec>
    struct hipfftw_input_validation_params
    {
        hipfftw_plan_creation_func  creation_options;
        hipfftw_plan_execution_func execution_option;
        std::pair<bool, bool>       creation_io_is_null;
        hipfftw_execution_io_args   execution_io;
        hipfftw_helper<prec>        plan_helper;

        // NOTE: placement is to be respected by tests' choice of I/O at plan creation!
        fft_result_placement creation_placement() const
        {
            return plan_helper.get_placement();
        }

        fft_result_placement execution_placement() const
        {
            auto ret = creation_placement();
            if(!use_creation_io_at_execution()
               && !(execution_io & hipfftw_execution_io_args::new_io_same_placement))
            {
                if(ret == fft_placement_inplace)
                    ret = fft_placement_notinplace;
                else
                    ret = fft_placement_inplace;
            }
            return ret;
        }

        bool use_creation_io_at_execution() const
        {
            return execution_io == hipfftw_execution_io_args::use_creation_io;
        }

        bool is_execution_arg_null(fft_io io_label) const
        {
            if(io_label != fft_io::fft_io_in && io_label != fft_io::fft_io_out)
                throw std::invalid_argument(
                    "invalid io_label for hipfftw_input_validation_params::is_execution_io_null");
            if(use_creation_io_at_execution())
                return io_label == fft_io::fft_io_in ? creation_io_is_null.first
                                                     : creation_io_is_null.second;
            const auto non_null_io_mask = io_label == fft_io::fft_io_in
                                              ? hipfftw_execution_io_args::non_null_new_in
                                              : hipfftw_execution_io_args::non_null_new_out;
            return !(execution_io & non_null_io_mask);
        }

        // checks consistency between values for test parameters that may have
        // overlapping scopes/meaning, in some specific cases
        bool can_be_tested(bool io_allocation_is_allowed = true) const
        {
            if(!hipfftw_execution_io_args_are_well_defined(execution_io))
                return false;
            if(!plan_helper.can_use_creation_options(creation_options))
                return false;
            if(creation_placement() == fft_placement_inplace)
            {
                if(creation_io_is_null.first != creation_io_is_null.second)
                    return false;
            }
            else
            {
                if(creation_io_is_null.first && creation_io_is_null.second)
                    return false;
            }
            if(execution_option != hipfftw_plan_execution_func::DEFAULT
               && use_creation_io_at_execution()
                      != (execution_option == hipfftw_plan_execution_func::EXECUTE))
            {
                return false;
            }
            if(execution_placement() == fft_placement_inplace)
            {
                if(is_execution_arg_null(fft_io::fft_io_in)
                   != is_execution_arg_null(fft_io::fft_io_out))
                    return false; // would be out-of-place at execution
            }
            else
            {
                if(is_execution_arg_null(fft_io::fft_io_in)
                   && is_execution_arg_null(fft_io::fft_io_out))
                    return false; // would be in-place at execution
            }
            if(!io_allocation_is_allowed)
            {
                // do not tolerate allow SetUp to allocate
                bool ret = creation_io_is_null.first
                           && (creation_placement() == fft_placement_inplace
                               || creation_io_is_null.second);
                if(!use_creation_io_at_execution() && ret)
                {
                    ret = is_execution_arg_null(fft_io::fft_io_in)
                          && (execution_placement() == fft_placement_inplace
                              || is_execution_arg_null(fft_io::fft_io_out));
                }
                return ret;
            }
            return true;
        }

        bool has_valid_io_for(hipfftw_step step) const
        {
            if(step != hipfftw_step::plan_creation && step != hipfftw_step::plan_execution)
                throw std::invalid_argument("Invalid step for has_valid_io_for");
            if(step == hipfftw_step::plan_creation)
            {
                // anything goes if using FFTW_ESTIMATE or FFTW_WISDOM_ONLY in flags
                const auto flags = plan_helper.get_flags();
                if(flags & FFTW_ESTIMATE || flags & FFTW_WISDOM_ONLY)
                    return true;
                return !creation_io_is_null.first
                       && (creation_placement() == fft_placement_inplace
                           || !creation_io_is_null.second);
            }
            if(!hipfftw_execution_io_args_are_well_defined(execution_io))
                throw std::runtime_error("invalid plan execution args");
            return !is_execution_arg_null(fft_io::fft_io_in)
                   && (creation_placement() == fft_placement_inplace
                       || !is_execution_arg_null(fft_io::fft_io_out));
        }

        // helper to determine if an internal exception may reliably be expected
        // for the given step (creation/execution) and, if yes, which kind of
        // internal exception
        hipfftw_internal_exception expected_internal_exception_for(hipfftw_step step) const
        {
            if(step != hipfftw_step::plan_creation && step != hipfftw_step::plan_execution)
                throw std::invalid_argument("Invalid step in expected_internal_exception_for");
            if(!can_be_tested())
                throw std::runtime_error(
                    hipfftw_creation_options_to_string(
                        creation_options, plan_helper.get_dft_kind(), plan_helper.get_rank())
                    + " cannot be tested for these parameters");

            const bool valid_args_for_plan_creation
                = plan_helper.is_valid_for_creation_with(creation_options)
                  && has_valid_io_for(hipfftw_step::plan_creation);
            const bool plan_can_be_created = valid_args_for_plan_creation
                                             && plan_helper.can_create_plan_with(creation_options);
            if(step == hipfftw_step::plan_creation)
            {
                if(plan_can_be_created)
                    return hipfftw_internal_exception::none;
                // plan cannot be created
                if(valid_args_for_plan_creation)
                    return hipfftw_internal_exception::unsupported_args;
                // plan cannot be created and arguments were invalid...
                // We may however have a mixed bag of some invalid and other unsupported args.
                // In such cases, the specific exception to expect would be ill-defined
                if(!plan_helper.has_unsupported_args_for(creation_options))
                    return hipfftw_internal_exception::invalid_args;
                else
                    return hipfftw_internal_exception::ill_defined;
            }
            else
            {
                if(!plan_can_be_created || !has_valid_io_for(hipfftw_step::plan_execution)
                   || !plan_helper.can_use_execution_option(execution_option))
                {
                    return hipfftw_internal_exception::invalid_args;
                }
                return hipfftw_internal_exception::none;
            }
        }

        std::string to_string() const
        {
            std::ostringstream ret;
            ret << plan_helper.token();
            ret << "_creation_func_"
                << hipfftw_creation_options_to_string(
                       creation_options, plan_helper.get_dft_kind(), plan_helper.get_rank());
            ret << "_creation_in_ptr" << (creation_io_is_null.first ? "_" : "_not_") << "nullptr";
            ret << "_creation_out_ptr" << (creation_io_is_null.second ? "_" : "_not_") << "nullptr";
            if(!hipfftw_execution_io_args_are_well_defined(execution_io))
                throw std::runtime_error("invalid plan execution args");
            if(!use_creation_io_at_execution())
            {
                ret << "_execution_new_in_ptr"
                    << (is_execution_arg_null(fft_io::fft_io_in) ? "_" : "_not_") << "nullptr";
                ret << "_execution_out_ptr"
                    << (is_execution_arg_null(fft_io::fft_io_out) ? "_" : "_not_") << "nullptr";
                ret << "_execution_placement"
                    << ((execution_io & hipfftw_execution_io_args::new_io_same_placement)
                            ? "_same_as_"
                            : "_different_than_")
                    << "creation_placement";
            }
            if(execution_option != hipfftw_plan_execution_func::DEFAULT)
            {
                ret << "_executed_via_" << hipfftw_execution_option_to_string(execution_option);
            }
            return ret.str();
        }

        // for using with insert_into_unique_sorted_params
        bool operator<(const hipfftw_input_validation_params& other) const
        {
            return to_string() < other.to_string();
        }
        bool operator==(const hipfftw_input_validation_params& other) const
        {
            return to_string() == other.to_string();
        }
    };

    template <fft_precision prec>
    std::vector<hipfftw_input_validation_params<prec>> params_for_testing_input_validation_params()
    {
        // constexpr used for readability of template specialization values below
        constexpr bool valid_value          = true;
        constexpr int  min_unsupported_rank = 4;
        // broad scope of hipfftw_helper structs configured with (zero or possibly
        // many) invalid/unsupported parameter value(s)
        std::vector<hipfftw_helper<prec>> helper_scope;
        for(auto dft_kind : trans_type_range_full)
        {
            std::vector<int> rank_range = {1, 2, 3};
            rank_range.push_back(get_random_rank<!valid_value>());
            rank_range.push_back(get_random_rank<valid_value, min_unsupported_rank>());
            for(auto rank : rank_range)
            {
                for(auto placement : place_range)
                {
                    std::vector<std::vector<ptrdiff_t>> range_of_lengths;
                    // most creation funcs take lengths as pointers
                    // --> test for empty lengths (re-interpreted as a nullptr
                    // arg by hipfftw_helper)
                    range_of_lengths.emplace_back(std::vector<ptrdiff_t>());
                    if(rank > 0)
                    {
                        const bool is_real_inplace
                            = is_real(dft_kind) && placement == fft_placement_inplace;
                        const ptrdiff_t allocatable_len_threshold = std::min(
                            get_len_threshold(max_num_elems_for_data_size<prec>(
                                                  max_byte_size_for_hipfftw_tests(), dft_kind),
                                              rank,
                                              is_real_inplace),
                            static_cast<ptrdiff_t>(max_length_for_hipfftw_test));
                        const auto valid_int_lengths
                            = get_random_lengths<valid_value, int>(rank, allocatable_len_threshold);
                        // always add valid integer lengths for valid ranks
                        range_of_lengths.emplace_back(valid_int_lengths);
                        // invalid integer lengths (most likely nonzero)
                        const auto invalid_int_lengths = get_random_lengths<!valid_value, int>(
                            rank, allocatable_len_threshold);
                        range_of_lengths.emplace_back(invalid_int_lengths);
                        // invalid integer lengths (some zero)
                        auto invalid_int_lengths_due_to_some_zero = valid_int_lengths;
                        invalid_int_lengths_due_to_some_zero[get_random_idx(rank)] = 0;
                        range_of_lengths.emplace_back(invalid_int_lengths_due_to_some_zero);
                        if(rank > 1 && rank < min_unsupported_rank)
                        {
                            // no support for layouts that trigger an int overflow for any relevant
                            // element index (unless GURU64 creation functions are used)
                            const auto min_overflowing_len = get_len_threshold(
                                std::numeric_limits<int>::max(), rank, is_real_inplace);
                            const auto unsupported_int_lengths
                                = get_random_lengths<valid_value, int>(
                                    rank, std::numeric_limits<int>::max(), min_overflowing_len + 1);
                            range_of_lengths.emplace_back(unsupported_int_lengths);
                        }
                    }
                    for(const auto& lengths : range_of_lengths)
                    {
                        std::vector<int> sign_range = {get_random_sign<valid_value>(dft_kind)};
                        if(is_complex(dft_kind))
                            sign_range.push_back(get_random_sign<!valid_value>(dft_kind));
                        for(auto sign : sign_range)
                        {
                            // FFTW_ESTIMATE is always supported
                            std::vector<unsigned> flags_range = {FFTW_ESTIMATE};
                            // some invalid flags
                            flags_range.push_back(get_random_flags<!valid_value>());
                            // unsupported FFTW_WISDOM_ONLY
                            flags_range.push_back(FFTW_WISDOM_ONLY
                                                  | get_random_flags<valid_value>());
                            // unsupported FFTW_PRESERVE_INPUT for multi-dimensional c2r
                            flags_range.push_back(FFTW_PRESERVE_INPUT
                                                  | get_random_flags<valid_value>());
                            for(auto flags : flags_range)
                            {
                                hipfftw_helper<prec> helper_to_add;
                                helper_to_add.set_creation_args(
                                    dft_kind, rank, lengths, placement, sign, flags);
                                helper_scope.emplace_back(helper_to_add);
                            }
                        }
                    }
                }
            }
        }
        // create a full-scope map containing all the generated test parameters; the map keys
        // capture the hipfftw's function name that the tests would target
        // --> ease for guaranteeing coverage even with low test probability in the end
        std::map<std::string, std::vector<hipfftw_input_validation_params<prec>>> full_scope_tests;
        hipfftw_input_validation_params<prec>                                     test_to_add;

        for(const auto& helper : helper_scope)
        {
            // do not allocate for the lengths designed to trigger an overflow
            // (allocation sizes would be ridiculously large)
            const bool test_may_allocate = helper.has_valid_rank() && helper.has_valid_lengths()
                                           && helper.get_data_byte_size(fft_io::fft_io_in)
                                                  <= max_byte_size_for_hipfftw_tests()
                                           && helper.get_data_byte_size(fft_io::fft_io_out)
                                                  <= max_byte_size_for_hipfftw_tests();
            for(auto creation : hipfftw_plan_creation_func_candidates)
            {
                for(auto execution : {hipfftw_plan_execution_func::EXECUTE,
                                      hipfftw_plan_execution_func::EXECUTE_DFT,
                                      hipfftw_plan_execution_func::EXECUTE_DFT_R2C,
                                      hipfftw_plan_execution_func::EXECUTE_DFT_C2R})
                {
                    // full range considered for creation_io_is_null and execution_io
                    // parameters: some might be ruled out later on because they can't
                    // be tested (e.g., "not_inplace" required yet using nullptr for
                    // creation input and output would be nonsensical)
                    const std::vector<std::pair<bool, bool>> creation_io_is_null_range
                        = {{false, false}, {true, false}, {false, true}, {true, true}};
                    for(auto set_creation_io_as_null : creation_io_is_null_range)
                    {
                        for(std::underlying_type_t<hipfftw_execution_io_args> exec_io
                            = hipfftw_execution_io_args::use_creation_io;
                            exec_io <= hipfftw_execution_io_args::clean_new_io;
                            exec_io++)
                        {
                            test_to_add.creation_options    = creation;
                            test_to_add.execution_option    = execution;
                            test_to_add.creation_io_is_null = set_creation_io_as_null;
                            test_to_add.execution_io
                                = static_cast<hipfftw_execution_io_args>(exec_io);
                            test_to_add.plan_helper = helper;
                            // skip params if they can't/shouldn't be tested anyways
                            if(!test_to_add.can_be_tested(test_may_allocate))
                                continue;
                            // tests expect a failure at execution at least
                            if(test_to_add.expected_internal_exception_for(
                                   hipfftw_step::plan_execution)
                               == hipfftw_internal_exception::none)
                                continue;
                            if(test_to_add.expected_internal_exception_for(
                                   hipfftw_step::plan_creation)
                                   == hipfftw_internal_exception::invalid_args
                               || test_to_add.expected_internal_exception_for(
                                      hipfftw_step::plan_creation)
                                      == hipfftw_internal_exception::unsupported_args)
                            {
                                insert_into_unique_sorted_params(
                                    full_scope_tests[hipfftw_creation_options_to_string(
                                        creation, helper.get_dft_kind(), helper.get_rank())],
                                    test_to_add);
                            }
                            else
                            {
                                insert_into_unique_sorted_params(
                                    full_scope_tests[hipfftw_execution_option_to_string(execution)],
                                    test_to_add);
                            }
                        }
                    }
                }
            }
        }
        std::vector<hipfftw_input_validation_params<prec>> ret;
        for(auto pair : full_scope_tests)
        {
            const auto& targeted_func  = pair.first;
            auto&       targeted_tests = pair.second;
            if(targeted_tests.empty())
            {
                throw std::runtime_error("params_for_testing_input_validation_params: empty list "
                                         "of (supposedly broad-spectrum) tests for "
                                         + targeted_func);
            }
            // add one randomly-chosen one to guarantee coverage for the targeted function
            const auto forced_coverage_idx = get_random_idx(targeted_tests.size());
            ret.emplace_back(targeted_tests[forced_coverage_idx]);
            targeted_tests.erase(targeted_tests.begin() + forced_coverage_idx);
            // consider all other probabilistically
            for(const auto& test : targeted_tests)
            {
                const double roll = hash_prob(random_seed, test.to_string());
                // not distinguishing between real/complex for this list generation
                if(roll > test_prob)
                {
                    if(verbose > 4)
                    {
                        std::cout << "Test skipped: (roll=" << roll << " > " << test_prob << ")\n";
                    }
                    continue;
                }
                ret.emplace_back(test);
            }
        }
        return ret;
    }

    template <fft_precision prec>
    class hipfftw_argument_validation
        : public ::testing::TestWithParam<hipfftw_input_validation_params<prec>>
    {
    protected:
        void SetUp() override
        {
            const hipfftw_input_validation_params<prec>& params = this->GetParam();

            if(!params.can_be_tested())
                GTEST_FAIL() << "invalid parameters which cannot be tested";

            // get_data_byte_size requires valid ranks and lengths to be calculated (of course)
            // --> make sure the I/O data sizes are not zero for test consistency w.r.t. testing
            // for nullptr data args I/O
            const size_t input_data_size
                = params.plan_helper.has_valid_rank() && params.plan_helper.has_valid_lengths()
                      ? params.plan_helper.get_data_byte_size(fft_io_in)
                      : sizeof(hipfftw_complex_t<prec>);
            const size_t output_data_size
                = params.plan_helper.has_valid_rank() && params.plan_helper.has_valid_lengths()
                      ? params.plan_helper.get_data_byte_size(fft_io_out)
                      : sizeof(hipfftw_complex_t<prec>);

            if(params.creation_io_is_null.first)
                plan_creation_input.free();
            else
                plan_creation_input.alloc(input_data_size);

            if(params.creation_placement() == fft_placement_inplace
               || params.creation_io_is_null.second)
                plan_creation_output.free();
            else
                plan_creation_output.alloc(output_data_size);

            if(params.use_creation_io_at_execution())
            {
                plan_execution_input.free();
                plan_execution_output.free();
            }
            else
            {
                if(!params.is_execution_arg_null(fft_io::fft_io_in))
                    plan_execution_input.alloc(input_data_size);
                else
                    plan_execution_input.free();

                if(params.execution_placement() == fft_placement_inplace
                   || params.is_execution_arg_null(fft_io::fft_io_out))
                    plan_execution_output.free();
                else
                    plan_execution_output.alloc(output_data_size);
            }
        }
        void TearDown() override
        {
            plan_creation_input.free();
            plan_creation_output.free();
            plan_execution_input.free();
            plan_execution_output.free();
            this->GetParam().plan_helper.release_plan();
        }
        hostbuf plan_creation_input;
        hostbuf plan_creation_output;
        hostbuf plan_execution_input;
        hostbuf plan_execution_output;

        void expect_failure(hipfftw_step step_target)
        {
            const hipfftw_input_validation_params<prec>& params = this->GetParam();
            std::unique_ptr<hipfftw_exception_logger>    exception_logger;
            bool                                         check_log_content = false;
            std::string                                  log_content;
            const auto expected_exception = params.expected_internal_exception_for(step_target);
            if(expected_exception != hipfftw_internal_exception::invalid_args
               && expected_exception != hipfftw_internal_exception::unsupported_args)
                GTEST_FAIL() << "invalid expected_exception to be tested for: only invalid or "
                                "unsupported arguments may be tested";
            const auto expected_log_instance
                = expected_exception == hipfftw_internal_exception::invalid_args
                      ? hipfftw_expected_log_instance<hipfftw_internal_exception::invalid_args>
                      : hipfftw_expected_log_instance<hipfftw_internal_exception::unsupported_args>;
            try
            {
                if(step_target == hipfftw_step::plan_creation)
                {
                    exception_logger  = std::make_unique<hipfftw_exception_logger>();
                    check_log_content = exception_logger->is_active();
                }
                params.plan_helper.create_plan(plan_creation_input.data(),
                                               params.creation_placement() == fft_placement_inplace
                                                   ? plan_creation_input.data()
                                                   : plan_creation_output.data(),
                                               params.creation_options);
                if(step_target == hipfftw_step::plan_creation)
                {
                    log_content = exception_logger->get_log();
                    exception_logger.reset();
                    const std::shared_ptr<hipfftw_plan_bundle_t<prec>> plan_bundle
                        = params.plan_helper.get_plan_bundle();
                    if(!plan_bundle)
                        throw std::runtime_error(
                            "the plan bundle could not be retrieved from the parameters");
                    if(plan_bundle->plan)
                        throw std::runtime_error(
                            hipfftw_creation_options_to_string(plan_bundle->creation_func,
                                                               params.plan_helper.get_dft_kind(),
                                                               params.plan_helper.get_rank())
                            + " actually created a plan for these parameters");
                }
                else
                {
                    exception_logger  = std::make_unique<hipfftw_exception_logger>();
                    check_log_content = exception_logger->is_active();
                    void* exec_in     = params.use_creation_io_at_execution()
                                            ? plan_creation_input.data()
                                            : plan_execution_input.data();
                    void* exec_out    = params.use_creation_io_at_execution()
                                            ? (params.creation_placement() == fft_placement_inplace
                                                   ? plan_creation_input.data()
                                                   : plan_creation_output.data())
                                            : (params.execution_placement() == fft_placement_inplace
                                                   ? plan_execution_input.data()
                                                   : plan_execution_output.data());
                    // intentionally do not check that hipfftw_test_plan != nullptr as that's
                    // kind of the point of this test: even if it doesn't report error codes,
                    // execution must not misbehave (e.g. must not segfault) with invalid argument
                    // (if hipfftw's exception handler is made verbose, it should print failure
                    //  info to the log, and that's verified in the end)
                    params.plan_helper.execute(exec_in, exec_out, params.execution_option);
                    log_content = exception_logger->get_log();
                    exception_logger.reset();
                }
            }
            catch(const hipfftw_undefined_function_ptr& e)
            {
                GTEST_FAIL() << "undefined function pointers detected. Error info: " << e.what();
            }
            catch(const std::runtime_error e)
            {
                if(log_content.empty() && exception_logger)
                    log_content = exception_logger->get_log();
                std::ostringstream gtest_info;
                gtest_info << e.what();
                if(!log_content.empty())
                    gtest_info << "\nContent of error log:\n" << log_content;
                GTEST_FAIL() << gtest_info.str();
            }
            catch(...)
            {
                if(log_content.empty() && exception_logger)
                    log_content = exception_logger->get_log();
                std::ostringstream gtest_info;
                gtest_info << "unidentified exception caught during test.";
                if(!log_content.empty())
                    gtest_info << "\nContent of error log:\n" << log_content;
                GTEST_FAIL() << gtest_info.str();
            }

            if(log_content.empty() && exception_logger)
                log_content = exception_logger->get_log();
            if(check_log_content && log_content.find(expected_log_instance) == std::string::npos)
            {
                GTEST_FAIL() << "No instance of \"" << expected_log_instance
                             << "\" detected in error log when testing for plan "
                             << (step_target == hipfftw_step::plan_creation ? "creation."
                                                                            : "execution.")
                             << "\nContent of error log:\n"
                             << log_content;
            }
        }

        void input_validation_test()
        {
            const auto& param = this->GetParam();
            if(param.expected_internal_exception_for(hipfftw_step::plan_execution)
               == hipfftw_internal_exception::none)
            {
                GTEST_FAIL() << "Invalid parameters for testing input validation (no internal "
                                "exception expected up to execution)";
            }
            // only one well-defined kind of exception should be expected at plan creation
            // for reliable testing: if plan creation arguments are a mixed bags of invalid
            // and unsupported values, implementation details may trigger one kind of
            // exception or the other (internally)
            const auto expected_plan_creation_exception
                = param.expected_internal_exception_for(hipfftw_step::plan_creation);
            if(expected_plan_creation_exception == hipfftw_internal_exception::invalid_args
               || expected_plan_creation_exception == hipfftw_internal_exception::unsupported_args)
            {
                expect_failure(hipfftw_step::plan_creation);
            }
            // always test for execution
            expect_failure(hipfftw_step::plan_execution);
        }

    public:
        static std::string TestName(
            const testing::TestParamInfo<typename hipfftw_argument_validation::ParamType>& info)
        {
            return info.param.to_string();
        }
    };

    //
    //---------------------------------------------------------------------------------------------
    //                                 FUNCTIONAL VALIDATION
    //---------------------------------------------------------------------------------------------
    //

    enum class hipfftw_data_memory_type
    {
        pageable_host,
        pinned_host,
#ifndef WIN32
        // linux-only
        managed,
#endif
        device
    };

    const std::vector<hipfftw_data_memory_type>& get_possible_data_mem_types()
    {
        auto create_possible_cases = []() {
            // always testable
            std::vector<hipfftw_data_memory_type> ret = {hipfftw_data_memory_type::pageable_host,
                                                         hipfftw_data_memory_type::pinned_host,
                                                         hipfftw_data_memory_type::device};
#ifndef WIN32
            // "managed" may or may not be supported
            hipDeviceProp_t props;
            if(hipGetDeviceProperties(&props, get_current_device_id()) == hipSuccess)
            {
                // explicitly ruling out gfx908 (MI100)
                if(std::strstr(props.gcnArchName, "gfx908") == nullptr && props.managedMemory == 1)
                    ret.push_back(hipfftw_data_memory_type::managed);
            }
#endif
            return ret;
        };
        const static std::vector<hipfftw_data_memory_type> possible_cases = create_possible_cases();
        return possible_cases;
    };

    std::string hipfftw_data_mem_type_to_string(hipfftw_data_memory_type mem_type)
    {
        switch(mem_type)
        {
        case hipfftw_data_memory_type::pageable_host:
            return "pageable_host";
            break;
        case hipfftw_data_memory_type::pinned_host:
            return "pinned_host";
            break;
#ifndef WIN32
        case hipfftw_data_memory_type::managed:
            return "managed";
            break;
#endif
        case hipfftw_data_memory_type::device:
            return "device";
            break;
        default:
            throw std::runtime_error("internal error: unexpected value of mem_tye in "
                                     "hipfftw_data_mem_type_to_string");
            break;
        }
    };

    template <fft_precision prec>
    struct hipfftw_functional_validation_params
    {
        // define type of I/O argument memory to be tested at a given step (creation/execution)
        // by a map: mem_type[{step_label, io_label}] represents the test's target memory type to consider
        // for the "io_label" I/O argument at step "step_label"
        std::map<std::pair<hipfftw_step, fft_io>, hipfftw_data_memory_type> mem_type;
        hipfftw_execution_io_args                                           execution_io;
        hipfftw_helper<prec>                                                plan_helper;

        fft_transform_type get_dft_kind() const
        {
            return plan_helper.get_dft_kind();
        }
        fft_result_placement get_placement() const
        {
            return plan_helper.get_placement();
        }
        bool use_creation_io_at_execution() const
        {
            return execution_io == hipfftw_execution_io_args::use_creation_io;
        }
        fft_array_type get_array_type(fft_io io) const
        {
            if(io != fft_io::fft_io_in && io != fft_io::fft_io_out)
                throw std::invalid_argument("invalid io argument for "
                                            "hipfftw_functional_validation_params::get_array_type");
            const auto dft_kind = plan_helper.get_dft_kind();
            if(is_complex(dft_kind))
                return fft_array_type_complex_interleaved;
            else if(is_fwd(dft_kind) == (io == fft_io::fft_io_in))
                return fft_array_type_real;
            else
                return fft_array_type_hermitian_interleaved;
        }
        std::vector<size_t> get_lengths() const
        {
            return plan_helper.template get_length_as<size_t>();
        }
        std::vector<size_t> get_ilengths() const
        {
            auto ilengths = get_lengths();
            if(plan_helper.get_dft_kind() == fft_transform_type_real_inverse)
                ilengths.back() = ilengths.back() / 2 + 1;
            return ilengths;
        }
        int get_rank() const
        {
            return plan_helper.get_rank();
        }
        std::vector<size_t> get_istride() const
        {
            return plan_helper.template get_strides_as<size_t>(fft_io::fft_io_in);
        }
        std::vector<size_t> get_ostride() const
        {
            return plan_helper.template get_strides_as<size_t>(fft_io::fft_io_out);
        }
        size_t get_idist() const
        {
            return plan_helper.template get_dist_as<size_t>(fft_io::fft_io_in);
        }
        size_t get_odist() const
        {
            return plan_helper.template get_dist_as<size_t>(fft_io::fft_io_out);
        }
        size_t get_nbatch() const
        {
            return plan_helper.template get_nbatch_as<size_t>(fft_io::fft_io_in);
        }
        std::vector<size_t> get_contiguous_istride() const
        {
            // equivalent to plan's strides for now, to be reconsidered once more general
            // configurations are enabled
            return plan_helper.template get_strides_as<size_t>(fft_io::fft_io_in);
        }
        size_t get_contiguous_idist() const
        {
            // equivalent to plan's strides for now, to be reconsidered once more general
            // configurations are enabled
            return plan_helper.template get_dist_as<size_t>(fft_io::fft_io_in);
        }

        bool can_be_tested() const
        {
            for(auto step : {hipfftw_step::plan_creation, hipfftw_step::plan_execution})
            {
                for(auto io : {fft_io::fft_io_in, fft_io::fft_io_out})
                {
                    const std::pair<hipfftw_step, fft_io> key = {step, io};
                    if(mem_type.find(key) == mem_type.end())
                    {
                        // incomplete mem_type map
                        return false;
                    }
                }
            }
            // if using new I/O at execution, they must be clean
            if(execution_io != hipfftw_execution_io_args::use_creation_io
               && execution_io != hipfftw_execution_io_args::clean_new_io)
            {
                return false;
            }
            if(!plan_helper.can_create_plan())
                return false;
            const auto placement = plan_helper.get_placement();
            if(placement == fft_placement_inplace)
            {
                if(mem_type.at({hipfftw_step::plan_creation, fft_io::fft_io_in})
                   != mem_type.at({hipfftw_step::plan_creation, fft_io::fft_io_out}))
                    return false;
                if(mem_type.at({hipfftw_step::plan_execution, fft_io::fft_io_in})
                   != mem_type.at({hipfftw_step::plan_execution, fft_io::fft_io_out}))
                    return false;
            }
            if(execution_io == hipfftw_execution_io_args::use_creation_io)
            {
                if(mem_type.at({hipfftw_step::plan_creation, fft_io::fft_io_in})
                       != mem_type.at({hipfftw_step::plan_execution, fft_io::fft_io_in})
                   || mem_type.at({hipfftw_step::plan_creation, fft_io::fft_io_out})
                          != mem_type.at({hipfftw_step::plan_execution, fft_io::fft_io_out}))
                    return false;
            }
            return true;
        }

        std::string to_string() const
        {
            for(auto step : {hipfftw_step::plan_creation, hipfftw_step::plan_execution})
            {
                for(auto io : {fft_io::fft_io_in, fft_io::fft_io_out})
                {
                    const std::pair<hipfftw_step, fft_io> key = {step, io};
                    if(mem_type.find(key) == mem_type.end())
                        throw std::runtime_error("incomplete mem_type map");
                }
            }
            std::ostringstream ret;
            ret << plan_helper.token();
            ret << "_creation_input_mem_type_"
                << hipfftw_data_mem_type_to_string(
                       mem_type.at({hipfftw_step::plan_creation, fft_io::fft_io_in}));
            if(plan_helper.get_placement() == fft_placement_notinplace)
            {
                ret << "_creation_output_mem_type_"
                    << hipfftw_data_mem_type_to_string(
                           mem_type.at({hipfftw_step::plan_creation, fft_io::fft_io_out}));
            }
            if(execution_io == hipfftw_execution_io_args::clean_new_io)
            {
                ret << "_execution_input_mem_type_"
                    << hipfftw_data_mem_type_to_string(
                           mem_type.at({hipfftw_step::plan_execution, fft_io::fft_io_in}));
                if(plan_helper.get_placement() == fft_placement_notinplace)
                {
                    ret << "_execution_output_mem_type_"
                        << hipfftw_data_mem_type_to_string(
                               mem_type.at({hipfftw_step::plan_execution, fft_io::fft_io_out}));
                }
            }
            return ret.str();
        }

        // for using with insert_into_unique_sorted_params
        bool operator<(const hipfftw_functional_validation_params& other) const
        {
            return to_string() < other.to_string();
        }
        bool operator==(const hipfftw_functional_validation_params& other) const
        {
            return to_string() == other.to_string();
        }
    };

    template <fft_precision prec>
    class hipfftw_functional_validation
        : public ::testing::TestWithParam<hipfftw_functional_validation_params<prec>>
    {
    protected:
        void SetUp() override
        {
            try
            {
                const hipfftw_functional_validation_params<prec>& params = this->GetParam();

                if(!params.can_be_tested())
                    GTEST_FAIL() << "invalid parameters, cannot be tested";
                if(reference_plan)
                    GTEST_FAIL()
                        << "Starting from an unclean slate (reference plan is not nullptr)";

                execution_results_on_host.resize(1);
                execution_results_on_host[0].alloc(
                    params.plan_helper.get_data_byte_size(fft_io::fft_io_out));

                std::vector<fft_io> io_range = {fft_io::fft_io_in};
                if(params.get_placement() == fft_placement_notinplace)
                    io_range.push_back(fft_io::fft_io_out);
                std::vector<hipfftw_step> step_range = {hipfftw_step::plan_creation};
                if(!params.use_creation_io_at_execution())
                    step_range.push_back(hipfftw_step::plan_execution);
                for(auto io : io_range)
                {
                    const size_t data_size = params.plan_helper.get_data_byte_size(io);
                    auto&        io_verification_vec
                        = io == fft_io::fft_io_in ? verification_input : verification_output;
                    io_verification_vec.resize(1);
                    io_verification_vec[0].alloc(data_size);
                    for(auto step : step_range)
                    {
                        const std::pair<hipfftw_step, fft_io> map_key = {step, io};
                        const auto mem_type                           = params.mem_type.at(map_key);
                        if(mem_type == hipfftw_data_memory_type::pageable_host
                           || mem_type == hipfftw_data_memory_type::pinned_host)
                        {
                            host_io_buffer[map_key].alloc(
                                data_size, mem_type == hipfftw_data_memory_type::pinned_host);
                        }
                        else
                        {
#ifndef WIN32
                            const auto hip_status = gpu_io_buffer[map_key].alloc(
                                data_size, mem_type == hipfftw_data_memory_type::managed);
#else
                            const auto hip_status = gpu_io_buffer[map_key].alloc(data_size);
#endif
                            if(hip_status != hipSuccess)
                            {
                                std::ostringstream gtest_info;
                                gtest_info << "failed to allocate a buffer of type "
                                           << hipfftw_data_mem_type_to_string(mem_type)
                                           << " and byte size " << std::to_string(data_size)
                                           << ". Current device ID is " << get_current_device_id();
                                throw hip_runtime_error(gtest_info.str(), hip_status);
                            }
                        }
                    }
                }
                if(verification_input.size() != 1
                   || (params.get_placement() != fft_placement_inplace
                       && verification_output.size() != 1))
                    GTEST_FAIL() << "Verification IO buffer incorrectly initialized";
                // generate input data
                const std::vector<size_t> field_lower(params.get_rank(), 0);
                const auto                ilength = params.get_ilengths();
                std::vector<size_t>       contiguous_istride(params.get_rank());
                size_t                    val = 1;
                for(int dim = params.get_rank() - 1; dim >= 0; dim--)
                {
                    contiguous_istride[dim] = val;
                    val *= ilength[dim];
                }
                const auto contiguous_idist = val;
                set_input<hostbuf, hipfftw_real_t<prec>>(verification_input,
                                                         fft_input_random_generator_host,
                                                         params.get_array_type(fft_io::fft_io_in),
                                                         params.get_lengths(),
                                                         ilength,
                                                         params.get_istride(),
                                                         params.get_idist(),
                                                         params.get_nbatch(),
                                                         get_curr_device_prop(),
                                                         field_lower,
                                                         0 /* field_lower_batch */,
                                                         contiguous_istride,
                                                         contiguous_idist);
                // create the reference plan (systematically using the most general guru64 creation)
                reference_plan = params.plan_helper.get_reference_plan(
                    verification_input[0].data(),
                    params.get_placement() == fft_placement_inplace
                        ? verification_input[0].data()
                        : verification_output[0].data());

                if(!reference_plan)
                {
                    GTEST_FAIL() << "could not create a reference plan";
                }
            }
            catch(const hip_runtime_error& e)
            {
                if(skip_runtime_fails)
                    GTEST_SKIP() << e.what() << "\nError code: " << e.hip_error << ".";
                else
                    GTEST_FAIL() << e.what() << "\nError code: " << e.hip_error << ".";
            }
        }
        void TearDown() override
        {
            verification_input.clear();
            verification_output.clear();
            execution_results_on_host.clear();
            host_io_buffer.clear();
            gpu_io_buffer.clear();
            if constexpr(prec == fft_precision_single)
                fftwf_destroy_plan(reference_plan);
            else
                fftw_destroy_plan(reference_plan);
            this->GetParam().plan_helper.release_plan();
        }
        // verification buffers (set_input and other common routines require std::vector's of size 1 for these)
        std::vector<hostbuf> verification_input;
        std::vector<hostbuf> verification_output;
        std::vector<hostbuf> execution_results_on_host;
        // possible host buffers (pageable or pinned host allocation)
        std::map<std::pair<hipfftw_step, fft_io>, hostbuf> host_io_buffer;
        // possible nonhost buffers (may be current/other device or runtime-managed)
        std::map<std::pair<hipfftw_step, fft_io>, gpubuf> gpu_io_buffer;
        // reference plan
        hipfftw_plan_t<prec> reference_plan = nullptr;

        void functional_test() const
        {
            const hipfftw_functional_validation_params<prec>& params = this->GetParam();
            try
            {
                std::ostringstream gtest_info;
                if(verification_input.size() != 1
                   || (params.get_placement() != fft_placement_inplace
                       && verification_output.size() != 1))
                    GTEST_FAIL() << "The verification I/O buffer(s) were not initialized as "
                                    "needed; host buffer(s) are required";
                if(execution_results_on_host.size() != 1)
                    GTEST_FAIL() << "Improper test initialization: no host buffer to copy the "
                                    "execution results";
                // get/define raw pointers as needed
                std::map<std::pair<hipfftw_step, fft_io>, void*>                    test_io_ptr;
                std::map<std::pair<hipfftw_step, fft_io>, hipfftw_data_memory_type> test_io_type;
                for(auto step : {hipfftw_step::plan_creation, hipfftw_step::plan_execution})
                {
                    if(step == hipfftw_step::plan_execution
                       && params.use_creation_io_at_execution())
                    {
                        for(auto io : {fft_io::fft_io_in, fft_io::fft_io_out})
                        {
                            const std::pair<hipfftw_step, fft_io> map_key = {step, io};
                            const std::pair<hipfftw_step, fft_io> creation_key
                                = {hipfftw_step::plan_creation, io};
                            test_io_ptr[map_key]  = test_io_ptr[creation_key];
                            test_io_type[map_key] = test_io_type[creation_key];
                        }
                        continue;
                    }
                    for(auto io : {fft_io::fft_io_in, fft_io::fft_io_out})
                    {
                        const std::pair<hipfftw_step, fft_io> map_key = {step, io};
                        if(io == fft_io::fft_io_out
                           && params.get_placement() == fft_placement_inplace)
                        {
                            const std::pair<hipfftw_step, fft_io> input_key
                                = {step, fft_io::fft_io_in};
                            test_io_ptr[map_key]  = test_io_ptr[input_key];
                            test_io_type[map_key] = test_io_type[input_key];
                            continue;
                        }
                        if(params.mem_type.find(map_key) == params.mem_type.end())
                            GTEST_FAIL() << "incomplete mem_type map in test parameters";
                        test_io_type[map_key] = params.mem_type.at(map_key);
                        if(test_io_type[map_key] == hipfftw_data_memory_type::pageable_host
                           || test_io_type[map_key] == hipfftw_data_memory_type::pinned_host)
                        {
                            const auto host_buf_it = host_io_buffer.find(map_key);
                            if(host_buf_it == host_io_buffer.end()
                               || host_buf_it->second.size()
                                      < params.plan_helper.get_data_byte_size(io))
                            {
                                GTEST_FAIL()
                                    << "The test " << (io == fft_io::fft_io_in ? "input" : "output")
                                    << " buffer was not initialized (host buffer required)";
                            }
                            else
                            {
                                test_io_ptr[map_key] = host_buf_it->second.data();
                            }
                        }
                        else
                        {
                            const auto gpu_buf_it = gpu_io_buffer.find(map_key);
                            if(gpu_buf_it == gpu_io_buffer.end()
                               || gpu_buf_it->second.size()
                                      < params.plan_helper.get_data_byte_size(io))
                            {
                                GTEST_FAIL()
                                    << "The test " << (io == fft_io::fft_io_in ? "input" : "output")
                                    << " buffer was not initialized (GPU buffer required)";
                            }
                            else
                            {
                                test_io_ptr[map_key] = gpu_buf_it->second.data();
                            }
                        }
                    }
                }
                // copy input data to hipfftw's input
                const std::pair<hipfftw_step, fft_io> exec_in_key
                    = {hipfftw_step::plan_execution, fft_io::fft_io_in};
                if(test_io_type.at(exec_in_key) == hipfftw_data_memory_type::device)
                {
                    // an explicit host-to-device copy is needed
                    const auto hip_status
                        = hipMemcpyAsync(test_io_ptr.at(exec_in_key),
                                         verification_input[0].data(),
                                         params.plan_helper.get_data_byte_size(fft_io::fft_io_in),
                                         hipMemcpyHostToDevice);
                    if(hip_status != hipSuccess)
                        throw hip_runtime_error("hipMemcpyAsync failed.", hip_status);
                }
                else
                {
                    std::memcpy(test_io_ptr.at(exec_in_key),
                                verification_input[0].data(),
                                params.plan_helper.get_data_byte_size(fft_io::fft_io_in));
                }

                std::shared_future<void> reference_cpu_dft = std::async(std::launch::async, [&]() {
                    if constexpr(prec == fft_precision_single)
                        fftwf_execute(reference_plan);
                    else
                        fftw_execute(reference_plan);
                });
                hipfftw_exception_logger exception_logger;

                params.plan_helper.create_plan(
                    test_io_ptr.at({hipfftw_step::plan_creation, fft_io::fft_io_in}),
                    test_io_ptr.at({hipfftw_step::plan_creation, fft_io::fft_io_out}));
                params.plan_helper.execute(
                    test_io_ptr.at({hipfftw_step::plan_execution, fft_io::fft_io_in}),
                    test_io_ptr.at({hipfftw_step::plan_execution, fft_io::fft_io_out}));

                if(exception_logger.is_active())
                {
                    const auto log_content = exception_logger.get_log();
                    if(!log_content.empty())
                    {
                        GTEST_FAIL() << "Non-empty log content detected:\n" << log_content;
                    }
                }

                // copy hipfftw results back into the execution_results_on_host[0] buffer
                // for verification purposes
                const std::pair<hipfftw_step, fft_io> exec_out_key
                    = {hipfftw_step::plan_execution, fft_io::fft_io_out};
                if(test_io_type.at(exec_out_key) == hipfftw_data_memory_type::device)
                {
                    // making this copy synchronous as the next step is verifying the results
                    const auto hip_status
                        = hipMemcpy(execution_results_on_host[0].data(),
                                    test_io_ptr.at(exec_out_key),
                                    params.plan_helper.get_data_byte_size(fft_io::fft_io_out),
                                    hipMemcpyDeviceToHost);
                    if(hip_status != hipSuccess)
                        throw hip_runtime_error("hipMemcpy failed (copying output results).",
                                                hip_status);
                }
                else
                {
                    std::memcpy(execution_results_on_host[0].data(),
                                test_io_ptr.at(exec_out_key),
                                params.plan_helper.get_data_byte_size(fft_io::fft_io_out));
                }
                // compare results
                if(reference_cpu_dft.valid())
                    reference_cpu_dft.get();
                const auto test_lengths  = params.get_lengths();
                auto       test_olengths = test_lengths;
                if(params.get_dft_kind() == fft_transform_type_real_forward)
                    test_olengths.back() = test_olengths.back() / 2 + 1;
                const auto  total_length     = product(test_lengths.begin(), test_lengths.end());
                const auto& reference_output = params.get_placement() == fft_placement_inplace
                                                   ? verification_input
                                                   : verification_output;

                const auto   ref_norm = norm(reference_output,
                                           test_olengths,
                                           params.get_nbatch(),
                                           prec,
                                           params.get_array_type(fft_io::fft_io_out),
                                           params.get_ostride(),
                                           params.get_odist(),
                                           {0} /* offset */);
                const double test_epsilon
                    = prec == fft_precision_single ? single_epsilon : double_epsilon;
                const double linf_cutoff = test_epsilon * ref_norm.l_inf * log(total_length);
                // compare results
                const auto diff = distance(reference_output,
                                           execution_results_on_host,
                                           test_olengths,
                                           params.get_nbatch(),
                                           prec,
                                           params.get_array_type(fft_io::fft_io_out),
                                           params.get_ostride(),
                                           params.get_odist(),
                                           params.get_array_type(fft_io::fft_io_out),
                                           params.get_ostride(),
                                           params.get_odist(),
                                           nullptr,
                                           linf_cutoff,
                                           {0} /* offset */,
                                           {0} /* offset */);
                EXPECT_LE(diff.l_inf, linf_cutoff);
                EXPECT_LE(diff.l_2 / ref_norm.l_2, sqrt(log2(total_length)) * test_epsilon);
                if constexpr(prec == fft_precision_single)
                {
                    max_linf_eps_single = std::max(max_linf_eps_single,
                                                   diff.l_inf / ref_norm.l_inf / log(total_length));
                    max_l2_eps_single   = std::max(
                        max_l2_eps_single, diff.l_2 / ref_norm.l_2 * sqrt(log2(total_length)));
                }
                else
                {
                    max_linf_eps_double = std::max(max_linf_eps_double,
                                                   diff.l_inf / ref_norm.l_inf / log(total_length));
                    max_l2_eps_double   = std::max(
                        max_l2_eps_double, diff.l_2 / ref_norm.l_2 * sqrt(log2(total_length)));
                }
            }
            catch(const hipfftw_undefined_function_ptr& e)
            {
                GTEST_FAIL() << "undefined function pointers detected. Error info: " << e.what();
            }
            catch(const hip_runtime_error e)
            {
                if(skip_runtime_fails)
                    GTEST_SKIP() << e.what() << "\nError code: " << e.hip_error << ".";
                else
                    GTEST_FAIL() << e.what() << "\nError code: " << e.hip_error << ".";
            }
            catch(const std::runtime_error e)
            {
                GTEST_FAIL() << e.what();
            }
            catch(...)
            {
                GTEST_FAIL() << "unidentified exception caught during test.";
            }
        }

    public:
        static std::string TestName(
            const testing::TestParamInfo<typename hipfftw_functional_validation::ParamType>& info)
        {
            return info.param.to_string();
        }
    };

    template <fft_precision prec>
    std::vector<hipfftw_functional_validation_params<prec>>
        params_for_functional_tests(size_t desired_full_suite_size)
    {
        std::vector<hipfftw_functional_validation_params<prec>> full_list;
        hipfftw_functional_validation_params<prec>              to_add;
        std::uniform_int_distribution<int>                      coin_toss(0, 1);
        // for readability of template specialization values below
        constexpr bool valid_value        = true;
        const auto&    possible_mem_types = get_possible_data_mem_types();
        while(full_list.size() < desired_full_suite_size)
        {
            to_add.execution_io = coin_toss(get_pseudo_rng()) == 0
                                      ? hipfftw_execution_io_args::use_creation_io
                                      : hipfftw_execution_io_args::clean_new_io;
            const auto dft_kind
                = trans_type_range_full[get_random_idx(trans_type_range_full.size())];
            const auto rank            = get_random_rank<valid_value, 1, 3>();
            const auto placement       = place_range[get_random_idx(place_range.size())];
            const bool is_real_inplace = is_real(dft_kind) && placement == fft_placement_inplace;
            const ptrdiff_t len_threshold
                = std::min(get_len_threshold(max_num_elems_for_data_size<prec>(
                                                 max_byte_size_for_hipfftw_tests(), dft_kind),
                                             rank,
                                             is_real_inplace),
                           static_cast<ptrdiff_t>(max_length_for_hipfftw_test));
            to_add.plan_helper.set_creation_args(
                dft_kind,
                rank,
                get_random_lengths<valid_value, int>(rank, len_threshold),
                placement,
                is_fwd(dft_kind) ? FFTW_FORWARD : FFTW_BACKWARD,
                FFTW_ESTIMATE);
            for(auto step : {hipfftw_step::plan_creation, hipfftw_step::plan_execution})
            {
                if(step == hipfftw_step::plan_execution
                   && to_add.execution_io == hipfftw_execution_io_args::use_creation_io)
                {
                    for(auto io : {fft_io::fft_io_in, fft_io::fft_io_out})
                    {
                        const std::pair<hipfftw_step, fft_io> key = {step, io};
                        const std::pair<hipfftw_step, fft_io> creation_key
                            = {hipfftw_step::plan_creation, io};
                        to_add.mem_type[key] = to_add.mem_type[creation_key];
                    }
                    continue;
                }
                for(auto io : {fft_io::fft_io_in, fft_io::fft_io_out})
                {
                    const std::pair<hipfftw_step, fft_io> key = {step, io};
                    if(placement == fft_placement_inplace && io == fft_io::fft_io_out)
                    {
                        auto input_key       = key;
                        input_key.second     = fft_io::fft_io_in;
                        to_add.mem_type[key] = to_add.mem_type[input_key];
                    }
                    else
                    {
                        to_add.mem_type[key]
                            = possible_mem_types[get_random_idx(possible_mem_types.size())];
                    }
                }
            }
            // skip params if they can't be tested for some reason
            if(!to_add.can_be_tested())
                continue;
            insert_into_unique_sorted_params(full_list, to_add);
        }
        if(test_prob == 1.0 && real_prob_factor == 1.0)
            return full_list;
        std::vector<hipfftw_functional_validation_params<prec>> ret;
        for(const auto& test : full_list)
        {
            const double roll = hash_prob(random_seed, test.to_string());
            const double run_prob
                = test_prob * (is_real(test.plan_helper.get_dft_kind()) ? real_prob_factor : 1.0);

            if(roll > run_prob)
            {
                if(verbose > 4)
                {
                    std::cout << "Test skipped: (roll=" << roll << " > " << run_prob << ")\n";
                }
                continue;
            }
            ret.emplace_back(test);
        }
        return ret;
    }

} // hipfftw test details' anonymous namespace

//
//---------------------------------------------------------------------------------------------
//                                    INSTANTIATION OF TESTS
//---------------------------------------------------------------------------------------------
//

TEST(hipfftw_test, utility_functions)
{
    test_existence_of_utility_functions<fft_precision_single>();
    test_existence_of_utility_functions<fft_precision_double>();
}

INSTANTIATE_TEST_SUITE_P(
#ifdef __HIP_PLATFORM_AMD__
    hipfftw_test,
#else
    DISABLED_hipfftw_test,
#endif
    allocation_sp,
    ::testing::ValuesIn(params_for_testing_hipfftw_malloc<fft_precision_single>()),
    allocation_sp::TestName);

INSTANTIATE_TEST_SUITE_P(
#ifdef __HIP_PLATFORM_AMD__
    hipfftw_test,
#else
    DISABLED_hipfftw_test,
#endif
    allocation_dp,
    ::testing::ValuesIn(params_for_testing_hipfftw_malloc<fft_precision_double>()),
    allocation_dp::TestName);

using argument_validation_sp = hipfftw_argument_validation<fft_precision_single>;
TEST_P(argument_validation_sp, creation_and_execution)
{
    input_validation_test();
}
using argument_validation_dp = hipfftw_argument_validation<fft_precision_double>;
TEST_P(argument_validation_dp, creation_and_execution)
{
    input_validation_test();
}

INSTANTIATE_TEST_SUITE_P(
#ifdef __HIP_PLATFORM_AMD__
    hipfftw_test,
#else
    DISABLED_hipfftw_test,
#endif
    argument_validation_sp,
    ::testing::ValuesIn(params_for_testing_input_validation_params<fft_precision_single>()),
    argument_validation_sp::TestName);

INSTANTIATE_TEST_SUITE_P(
#ifdef __HIP_PLATFORM_AMD__
    hipfftw_test,
#else
    DISABLED_hipfftw_test,
#endif
    argument_validation_dp,
    ::testing::ValuesIn(params_for_testing_input_validation_params<fft_precision_double>()),
    argument_validation_dp::TestName);

using hipfftw_functional_validation_sp = hipfftw_functional_validation<fft_precision_single>;
TEST_P(hipfftw_functional_validation_sp, accuracy_vs_fftw)
{
    functional_test();
}

using hipfftw_functional_validation_dp = hipfftw_functional_validation<fft_precision_double>;
TEST_P(hipfftw_functional_validation_dp, accuracy_vs_fftw)
{
    functional_test();
}

static constexpr size_t full_suite_size = 1024; // per precision
INSTANTIATE_TEST_SUITE_P(
    hipfftw_test,
    hipfftw_functional_validation_sp,
    ::testing::ValuesIn(params_for_functional_tests<fft_precision_single>(full_suite_size)),
    hipfftw_functional_validation_sp::TestName);
INSTANTIATE_TEST_SUITE_P(
    hipfftw_test,
    hipfftw_functional_validation_dp,
    ::testing::ValuesIn(params_for_functional_tests<fft_precision_double>(full_suite_size)),
    hipfftw_functional_validation_dp::TestName);

// params_for_functional_tests may return empty vectors for low test probabilities.
// The following ensures such cases do not make gtest report an error due to uninstantiated
// hipfftw_functional_validation_{sp,dp}.
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(hipfftw_functional_validation_sp);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(hipfftw_functional_validation_dp);
