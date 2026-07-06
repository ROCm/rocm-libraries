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
#include "../../shared/fftw_transform.h"
#include "../../shared/gpubuf.h"
#include "../../shared/hostbuf.h"
#include "../../shared/params_gen.h"
#include "../../shared/reference_fft_data.h"
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

extern size_t      max_length_for_hipfftw_test;
extern size_t      max_nbatch_for_hipfftw_test;
extern size_t      max_io_gb_for_hipfftw_test;
extern size_t      max_num_arg_validation_tests_per_hipfftw_plan_type;
extern size_t      max_elementary_stride_for_hipfftw_test;
extern std::string hipfftw_token_for_functional_test;

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
            size_t tmp = system_memory::singleton().get_limit_bytes();
            for(size_t dev_id = 0; dev_id < device_memory_accountant::singleton().num_devices();
                dev_id++)
            {
                tmp = std::min(
                    tmp, device_memory_accountant::singleton().get_limit_bytes_on_device(dev_id));
            }
            tmp = std::min(tmp, max_io_gb_for_hipfftw_test * ONE_GiB);
            if(verbose > 0)
            {
                std::cout << "Limit for the size of I/O data used in hipfftw tests: "
                          << byte_size_to_str(tmp) << std ::endl;
            }
            return tmp;
        };
        static const size_t io_byte_size_limit = get_io_byte_size_limit();
        return io_byte_size_limit;
    }

    // Random value generators are defined and used herein for the generation of
    // argument-validation and/or functional test parameters. In the latter
    // case, only valid (*and* supported) parameter values must be generated
    // whereas invalid values need to be (knowingly) generated and thrown in
    // the mix in the former case. To that end, several get_random* functions
    // defined below are templated with a ``bool`` parameter reflecting
    // whether a valid ('true' specialization) or invalid ('false' specialization)
    // random parameter value is to be returned.
    // The following constexpr is a self-explanatory placeholder introduced to
    // improve code readability w.r.t. template specializations used in parameter
    // generations here below, e.g., "get_random_rank<!valid_value>()" is
    // a lot more intuitive to understand than "get_random_rank<false>()"
    constexpr bool valid_value = true;

    std::ranlux24_base& get_pseudo_rng()
    {
        // The fully-randomized data layouts being tested via hipFFTW (generated in
        // this unit) explore a wider scope of plan configurations than the rocFFT
        // unit tests. That creates a significant risk for false-positives in hipFFTW
        // tests, i.e., hipFFTW test failures actually triggered by a (unknown-yet)
        // rocFFT defect under the hood.
        // Restricting the possible test-generation seeds to a set of pre-verified (on
        // gfx90a) values mitigates that risk w/o sacrificing hipFFTW code and feature
        // coverage.
        const std::vector<size_t> verified_seeds = {1884086845,
                                                    3282587354,
                                                    2802468591,
                                                    2531043007,
                                                    2913365592,
                                                    334387271,
                                                    3196785377,
                                                    3518924498,
                                                    3455844329,
                                                    3390661914,
                                                    2632016286,
                                                    1664906633,
                                                    3892674020};
        static const auto         test_generation_seed
            = verified_seeds[random_seed % verified_seeds.size()];
        static std::ranlux24_base gen(test_generation_seed);
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
    // Note: albeit not supported, ranks > 3 are "valid" arguments
    // --> limiting rank value to max of 5 by default to avoid ridiculously long
    // lengths/batches possibly created in automated parameter generations;
    // (can make calls to array_valid pretty time-consuming)
    template <bool validity_flag,
              int  min_rank          = validity_flag ? 1 : std::numeric_limits<int>::lowest(),
              int  max_rank          = validity_flag ? 5 : 0,
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

    template <bool strictly_positive_values, typename type_to_consider_for_validity>
    std::vector<ptrdiff_t>
        get_random_vector(int       desired_rank,
                          ptrdiff_t max_abs_val
                          = std::numeric_limits<type_to_consider_for_validity>::max(),
                          ptrdiff_t min_abs_val = 0)
    {
        std::vector<ptrdiff_t> ret;
        // cannot generate vectors for invalid ranks --> return an empty vector in that case
        if(!rank_is_valid_for_hipfftw(desired_rank))
            return ret;
        if(min_abs_val < 0 || min_abs_val > max_abs_val)
            throw std::invalid_argument("invalid bounds used for get_random_vector");
        if constexpr(strictly_positive_values)
        {
            if(max_abs_val < 1)
                throw std::runtime_error(
                    "invalid bounds for generating a vector of strictly positive values");
        }
        // generate values that are all representable as integers
        auto&                                    pseudo_rng = get_pseudo_rng();
        std::uniform_int_distribution<ptrdiff_t> val_rng(min_abs_val, max_abs_val);
        // setter lambda
        auto set_random_values = [&]() {
            for(auto& l : ret)
            {
                const ptrdiff_t val = val_rng(pseudo_rng);
                if constexpr(strictly_positive_values)
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
        set_random_values();
        constexpr type_to_consider_for_validity min_valid_val = 1;
        while(vector_has_valid_values_as<type_to_consider_for_validity>(
                  ret, desired_rank, min_valid_val)
              != strictly_positive_values)
        {
            set_random_values();
        }

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
    template <typename T>
    T get_random_element_in(const std::vector<T>& element_list)
    {
        return element_list[get_random_idx(element_list.size())];
    }
    template <typename T>
    T get_random_element_in(const std::set<T>& element_set)
    {
        return *std::next(element_set.begin(), get_random_idx(element_set.size()));
    }

    struct valid_values_cannot_be_created : std::runtime_error
    {
        valid_values_cannot_be_created()
            : std::runtime_error::runtime_error("")
        {
        }
    };

    template <bool validity_flag>
    std::vector<ptrdiff_t>
        get_random_lengths_from_fwd_domain_nembed(const std::vector<ptrdiff_t>& fwd_domain_nembed,
                                                  bool                          is_real_inplace)
    {
        std::vector<ptrdiff_t> ret(fwd_domain_nembed.size());
        if(fwd_domain_nembed.empty())
            return ret;
        if constexpr(validity_flag)
        {
            if(!vector_has_valid_values_as<int>(fwd_domain_nembed, fwd_domain_nembed.size(), 1))
                throw valid_values_cannot_be_created();
            if(static_cast<size_t>(
                   std::count(fwd_domain_nembed.begin(), fwd_domain_nembed.end() - 1, 1))
               == fwd_domain_nembed.size() - 1)
            {
                if(fwd_domain_nembed.back() <= (is_real_inplace ? 2 : 1))
                {
                    // only valid lengths are 1x1x...x1
                    throw valid_values_cannot_be_created();
                }
            }
            if(is_real_inplace && fwd_domain_nembed.back() % 2 == 1)
                throw valid_values_cannot_be_created();
        }

        auto make_lengths = [&]() {
            if constexpr(validity_flag)
            {
                for(size_t dim_idx = 0; dim_idx < ret.size(); dim_idx++)
                {
                    auto max_len = fwd_domain_nembed[dim_idx];
                    if(is_real_inplace && dim_idx == ret.size() - 1)
                        max_len--;
                    std::uniform_int_distribution<ptrdiff_t> len_rng(1, max_len);
                    ret[dim_idx] = len_rng(get_pseudo_rng());
                }
            }
            else
            {
                static std::uniform_int_distribution<ptrdiff_t> len_rng(
                    -max_length_for_hipfftw_test, max_length_for_hipfftw_test);
                for(auto& tmp : ret)
                    tmp = len_rng(get_pseudo_rng());
            }
        };
        auto lengths_are_valid = [&]() {
            auto check = true;
            for(size_t dim = 0; dim < ret.size() && check; dim++)
            {
                if(is_real_inplace && dim == ret.size() - 1)
                    check = fwd_domain_nembed[dim] >= 2 * (ret[dim] / 2 + 1) && ret[dim] > 0;
                else
                    check = fwd_domain_nembed[dim] >= ret[dim] && ret[dim] > 0;
            }
            // rule out testing for all length values of 1 as it triggers very questionable
            // corner-case usage where all strides become irrelevant on paper
            return check
                   && static_cast<size_t>(std::count(ret.begin(), ret.end(), 1)) != ret.size();
        };
        make_lengths();
        while(lengths_are_valid() != validity_flag)
        {
            make_lengths(); // try again
        }
        return ret;
    }

    template <bool validity_flag>
    std::vector<ptrdiff_t>
        get_random_bwd_domain_nembed(ptrdiff_t                     max_nembed_fwd_domain,
                                     const std::vector<ptrdiff_t>& fwd_domain_nembed,
                                     const std::vector<ptrdiff_t>& lengths,
                                     fft_transform_type            dft_type,
                                     fft_result_placement          placement)
    {
        if(!lengths.empty() && fwd_domain_nembed.size() != lengths.size())
            throw std::invalid_argument("fwd_domain_nembed.size() == lengths.size() required by "
                                        "get_random_bwd_domain_nembed for non-empty lengths");
        if(max_nembed_fwd_domain <= 0)
            throw std::invalid_argument(
                "max_nembed_fwd_domain > 0 required by get_random_bwd_domain_nembed");
        if constexpr(validity_flag)
        {
            if(lengths.empty())
                throw valid_values_cannot_be_created();
            if(!vector_has_valid_values_as<int>(fwd_domain_nembed, fwd_domain_nembed.size(), 1)
               || !vector_has_valid_values_as<int>(lengths, lengths.size(), 1))
                throw valid_values_cannot_be_created();
        }

        std::vector<ptrdiff_t> ret(fwd_domain_nembed.size());
        if(fwd_domain_nembed.empty())
            return ret;

        auto make_bwd_domain_nembed = [&]() {
            if constexpr(validity_flag)
            {
                for(size_t dim_idx = 0; dim_idx < ret.size(); dim_idx++)
                {
                    ptrdiff_t max_nembed_for_dim = is_real(dft_type) && dim_idx == ret.size() - 1
                                                       ? max_nembed_fwd_domain / 2 + 1
                                                       : max_nembed_fwd_domain;
                    auto      min_nembed         = lengths[dim_idx];
                    if(is_real(dft_type) && dim_idx == ret.size() - 1)
                        min_nembed = min_nembed / 2 + 1;
                    if(min_nembed > max_nembed_for_dim)
                        throw valid_values_cannot_be_created();
                    if(placement == fft_placement_inplace)
                    {
                        ret[dim_idx] = fwd_domain_nembed[dim_idx];
                        if(is_real(dft_type) && dim_idx == ret.size() - 1)
                        {
                            if(fwd_domain_nembed[dim_idx] % 2 == 1)
                                throw valid_values_cannot_be_created();
                            ret[dim_idx] = ret[dim_idx] / 2;
                        }
                        if(ret[dim_idx] < min_nembed)
                            throw valid_values_cannot_be_created();
                    }
                    else
                    {
                        std::uniform_int_distribution<ptrdiff_t> nembed_rng(min_nembed,
                                                                            max_nembed_for_dim);
                        ret[dim_idx] = nembed_rng(get_pseudo_rng());
                    }
                }
            }
            else
            {
                std::uniform_int_distribution<ptrdiff_t> nembed_rng(-max_nembed_fwd_domain,
                                                                    max_nembed_fwd_domain);
                for(size_t dim_idx = 0; dim_idx < ret.size(); dim_idx++)
                {
                    ret[dim_idx] = nembed_rng(get_pseudo_rng());
                    if(is_real(dft_type) && dim_idx == ret.size() - 1)
                        ret[dim_idx] = ret[dim_idx] / 2 + 1;
                }
            }
        };
        auto bwd_domain_nembed_are_valid = [&]() {
            auto       check       = true;
            const bool is_real_dft = is_real(dft_type);
            for(size_t dim = 0; dim < ret.size() && check; dim++)
            {
                const auto min_nembed
                    = lengths.empty() ? 0
                                      : (dim == ret.size() - 1 && is_real_dft ? lengths[dim] / 2 + 1
                                                                              : lengths[dim]);
                check = ret[dim] >= min_nembed && ret[dim] > 0;
                if(placement == fft_placement_inplace && check)
                {
                    check = ret[dim]
                            == (dim == ret.size() - 1 && is_real_dft ? fwd_domain_nembed[dim] / 2
                                                                     : fwd_domain_nembed[dim]);
                }
            }
            return check;
        };

        make_bwd_domain_nembed();
        while(bwd_domain_nembed_are_valid() != validity_flag)
        {
            make_bwd_domain_nembed(); // try again
        }
        return ret;
    }

    // calculates the threshold value X such that the byte size of I/O default-layout data
    // sets is no greater than max_byte_size, if all elements of lengths are no greater than X,
    // and if lengths.size() == rank [using bisection]
    template <fft_precision prec>
    ptrdiff_t find_threshold_length_for_byte_size(size_t max_byte_size, int rank, bool real_dft)
    {
        if(rank < 1)
            throw std::invalid_argument("invalid rank used in find_threshold_length_for_byte_size");
        if(max_byte_size == 0)
            return 0;
        constexpr ptrdiff_t X_max = std::numeric_limits<ptrdiff_t>::max();
        // we need to find X in [0, X_max] s.t.
        // byte_size(X) <= max_byte_size && byte_size(X + 1) > max_byte_size
        auto byte_size = [&](ptrdiff_t X) {
            // for real DFT, default layout's largest memory footprint are in the complex domain
            size_t ret = real_dft ? (X / 2 + 1) : X;
            for(auto i = 1; i < rank; i++)
                ret *= X;
            ret *= sizeof(hipfftw_complex_t<prec>);
            return ret;
        };
        // initialization
        ptrdiff_t X_down = static_cast<ptrdiff_t>(std::floor(std::pow(
            max_byte_size
                / (real_dft ? sizeof(hipfftw_real_t<prec>) : sizeof(hipfftw_complex_t<prec>)),
            1.0 / rank)));
        ptrdiff_t diff   = 1;
        ptrdiff_t X_up   = X_down;
        while(byte_size(X_up) <= max_byte_size && X_up < X_max)
        {
            X_down = X_up;
            X_up   = X_up <= X_max - diff ? X_up + diff : X_max;
            diff *= 2;
        }
        diff = 1;
        while(byte_size(X_down) > max_byte_size && X_down > 0)
        {
            X_up   = X_down;
            X_down = X_down >= diff ? X_down - diff : 0;
            diff *= 2;
        }
        // bisection
        while(X_up - X_down > 1)
        {
            const auto tmp = (X_up + X_down) / 2;
            if(byte_size(tmp) <= max_byte_size)
                X_down = tmp;
            else
                X_up = tmp;
        }
        return X_down;
    }

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
        friend std::ostream& operator<<(std::ostream& stream, const hipfftw_malloc_params& params)
        {
            stream << params.to_string();
            return stream;
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
            catch(const std::bad_alloc&)
            {
                GTEST_SKIP() << "host memory allocation failure";
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
        bool can_be_tested() const
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
            if(flags_are_valid_for_hipfftw(plan_helper.get_flags())
               && !(plan_helper.get_flags() & FFTW_ESTIMATE))
            {
                // I/O data pointers may be touched at creation. In that case,
                // the I/O allocations must make sense and be large enough
                // --> cannot test if I/O allocation sizes cannot be reliably calculated
                try
                {
                    (void)plan_helper.get_data_byte_size(fft_io::fft_io_in);
                    (void)plan_helper.get_data_byte_size(fft_io::fft_io_out);
                }
                catch(const typename hipfftw_helper<prec>::num_elements_calc_exception& e)
                {
                    return false;
                }
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
                if(!plan_helper.has_unsupported_args())
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
        friend std::ostream& operator<<(std::ostream&                          stream,
                                        const hipfftw_input_validation_params& params)
        {
            stream << params.to_string();
            return stream;
        }
    };

    template <int min_unsupported = 4, std::enable_if_t<(min_unsupported > 1), bool> = true>
    std::vector<int> arg_validation_runtime_rank_range()
    {
        std::vector<int> ret = {
            get_random_rank<!valid_value>(), // invalid
            get_random_rank<valid_value, min_unsupported>(), // valid but unsupported
            get_random_rank<valid_value, 1, min_unsupported - 1>() // valid and supported
        };
        return ret;
    }

    template <typename validation_t = int,
              std::enable_if_t<
                  std::is_same_v<validation_t, int> || std::is_same_v<validation_t, ptrdiff_t>,
                  bool> = true>
    std::vector<std::vector<ptrdiff_t>>
        arg_validation_strictly_positive_vec_range(int vec_size, ptrdiff_t max_value)
    {
        std::vector<std::vector<ptrdiff_t>> ret;
        if(vec_size < 1)
            return ret;
        if(max_value < 1)
            throw std::invalid_argument("invalid max_val in arg_validation_strictly_positive_vec");
        // always add valid values
        ret.emplace_back(get_random_vector<valid_value, validation_t>(vec_size, max_value));
        // invalid integer values (most likely nonzero)
        ret.emplace_back(get_random_vector<!valid_value, validation_t>(vec_size, max_value));
        // invalid integer values due to some zero(es)
        auto tmp = get_random_vector<valid_value, validation_t>(vec_size, max_value);
        tmp[get_random_idx(tmp.size())] = 0;
        ret.emplace_back(tmp);
        return ret;
    }

    std::vector<std::vector<ptrdiff_t>>
        arg_validation_lengths_range_many_dft(const std::vector<ptrdiff_t>& fwd_domain_nembed,
                                              bool                          is_real_inplace)
    {
        std::vector<std::vector<ptrdiff_t>> ret;
        if(fwd_domain_nembed.empty())
        {
            ret.emplace_back(std::vector<ptrdiff_t>());
            return ret;
        }

        ret.emplace_back(get_random_lengths_from_fwd_domain_nembed<!valid_value>(fwd_domain_nembed,
                                                                                 is_real_inplace));
        try
        {
            ret.emplace_back(get_random_lengths_from_fwd_domain_nembed<valid_value>(
                fwd_domain_nembed, is_real_inplace));
        }
        catch(const valid_values_cannot_be_created& e)
        {
        }
        return ret;
    }

    std::vector<std::vector<ptrdiff_t>> arg_validation_bwd_domain_nembed_range_many_dft(
        ptrdiff_t                     max_nembed_fwd_domain,
        const std::vector<ptrdiff_t>& fwd_domain_nembed,
        const std::vector<ptrdiff_t>& lengths,
        fft_transform_type            dft_kind,
        fft_result_placement          placement)
    {
        std::vector<std::vector<ptrdiff_t>> ret;

        ret.emplace_back(get_random_bwd_domain_nembed<!valid_value>(
            max_nembed_fwd_domain, fwd_domain_nembed, lengths, dft_kind, placement));
        try
        {
            ret.emplace_back(get_random_bwd_domain_nembed<valid_value>(
                max_nembed_fwd_domain, fwd_domain_nembed, lengths, dft_kind, placement));
        }
        catch(const valid_values_cannot_be_created& e)
        {
        }

        return ret;
    }

    std::vector<ptrdiff_t> arg_validation_dist_range_many_dft(ptrdiff_t                 nbatch,
                                                              const hipfftw_ionembed_t& ionembed,
                                                              fft_io                    io,
                                                              ptrdiff_t (*random_dist_func)())
    {
        std::vector<ptrdiff_t> ret;
        ptrdiff_t              to_add;
        const int*             nembed = ionembed.get_nembed(io);
        const int              stride = ionembed.get_elementary_stride(io);
        if(nembed)
        {
            // add a value that will always be valid for valid nembed and stride
            to_add = product(nembed, nembed + ionembed.get_nembed_size(io)) * stride;
        }
        else
        {
            to_add = random_dist_func();
        }
        ret.push_back(to_add);
        // 0-distances are invalid
        if(nbatch > 1)
            ret.push_back(0);
        // aliasing layout on output is invalid
        if(nbatch > stride)
        {
            ret.push_back(1);
        }
        return ret;
    }

    std::vector<int> arg_validation_sign_range(fft_transform_type test_dft_type)
    {
        std::vector<int> ret = {get_random_sign<valid_value>(test_dft_type)};
        if(is_complex(test_dft_type))
            ret.push_back(get_random_sign<!valid_value>(test_dft_type));
        return ret;
    }

    std::vector<unsigned> arg_validation_flags_range(fft_transform_type test_dft_type,
                                                     int                test_rank)
    {
        // FFTW_ESTIMATE is always supported
        std::vector<unsigned> ret = {FFTW_ESTIMATE};
        // some invalid flags
        ret.push_back(get_random_flags<!valid_value>());
        // unsupported FFTW_WISDOM_ONLY
        ret.push_back(FFTW_WISDOM_ONLY | get_random_flags<valid_value>());
        if(test_dft_type == fft_transform_type_real_inverse && test_rank > 1)
        {
            // unsupported FFTW_PRESERVE_INPUT for multi-dimensional c2r
            ret.push_back(FFTW_PRESERVE_INPUT | get_random_flags<valid_value>());
        }
        return ret;
    }

    template <fft_precision prec>
    std::vector<hipfftw_helper<prec>> test_scope_for_arg_validation_of_plan_dft_nd()
    {
        std::vector<hipfftw_helper<prec>> ret;
        while(ret.size() < max_num_arg_validation_tests_per_hipfftw_plan_type)
        {
            const auto dft_kind = get_random_element_in(trans_type_range_full);
            // only ranks 1-3 for plan_dft_nd functions
            const auto      rank      = get_random_rank<valid_value, 1, 3>();
            const auto      placement = get_random_element_in(place_range);
            const ptrdiff_t max_len
                = std::min(static_cast<ptrdiff_t>(max_length_for_hipfftw_test),
                           find_threshold_length_for_byte_size<prec>(
                               max_byte_size_for_hipfftw_tests(), rank, is_real(dft_kind)));
            const auto lengths
                = get_random_element_in(arg_validation_strictly_positive_vec_range(rank, max_len));
            const auto sign  = get_random_element_in(arg_validation_sign_range(dft_kind));
            const auto flags = get_random_element_in(arg_validation_flags_range(dft_kind, rank));
            hipfftw_helper<prec> helper_to_add;
            helper_to_add.set_creation_args(dft_kind, rank, lengths, placement, sign, flags);
            ret.push_back(helper_to_add);
        }
        return ret;
    }

    template <fft_precision prec>
    std::vector<hipfftw_helper<prec>> test_scope_for_arg_validation_of_plan_dft()
    {
        std::vector<hipfftw_helper<prec>> ret;

        while(ret.size() < max_num_arg_validation_tests_per_hipfftw_plan_type)
        {
            const auto dft_kind  = get_random_element_in(trans_type_range_full);
            const auto rank      = get_random_element_in(arg_validation_runtime_rank_range());
            const auto placement = get_random_element_in(place_range);
            ptrdiff_t  max_len   = max_length_for_hipfftw_test;
            if(rank_is_valid_for_hipfftw(rank))
            {
                max_len = std::min(max_len,
                                   find_threshold_length_for_byte_size<prec>(
                                       max_byte_size_for_hipfftw_tests(), rank, is_real(dft_kind)));
            }
            std::vector<std::vector<ptrdiff_t>> range_of_lengths
                = arg_validation_strictly_positive_vec_range(rank, max_len);
            // --> test for empty lengths, too (re-interpreted as a nullptr argument by hipfftw_helper)
            range_of_lengths.push_back(std::vector<ptrdiff_t>());
            const auto lengths = get_random_element_in(range_of_lengths);
            const auto sign    = get_random_element_in(arg_validation_sign_range(dft_kind));
            const auto flags   = get_random_element_in(arg_validation_flags_range(dft_kind, rank));
            hipfftw_helper<prec> helper_to_add;
            helper_to_add.set_creation_args(dft_kind, rank, lengths, placement, sign, flags);
            ret.push_back(helper_to_add);
        }
        return ret;
    }

    template <fft_precision prec>
    std::vector<hipfftw_helper<prec>> test_scope_for_arg_validation_of_plan_many_dft()
    {
        std::vector<hipfftw_helper<prec>> ret;
        constexpr int batch_rank = 1; // nothing else possible via the *_many_dft* apis
        auto          get_random_elementary_stride = [](bool valid_and_supported_stride) {
            static std::uniform_int_distribution<ptrdiff_t> stride_rng(
                -max_elementary_stride_for_hipfftw_test, max_elementary_stride_for_hipfftw_test);
            auto ret = stride_rng(get_pseudo_rng());
            while((ret > 0) == valid_and_supported_stride)
                ret = stride_rng(get_pseudo_rng());
            return ret;
        };
        auto get_random_dist = []() {
            static std::uniform_int_distribution<ptrdiff_t> dist_rng(
                -max_byte_size_for_hipfftw_tests()
                    / (max_nbatch_for_hipfftw_test * sizeof(hipfftw_complex_t<prec>)),
                max_byte_size_for_hipfftw_tests()
                    / (max_nbatch_for_hipfftw_test * sizeof(hipfftw_complex_t<prec>)));
            return dist_rng(get_pseudo_rng());
        };

        while(ret.size() < max_num_arg_validation_tests_per_hipfftw_plan_type)
        {

            const auto dft_kind = get_random_element_in(trans_type_range_full);
            const auto rank     = get_random_element_in(arg_validation_runtime_rank_range());
            const auto nbatch   = get_random_element_in(arg_validation_strictly_positive_vec_range(
                                                          batch_rank, max_nbatch_for_hipfftw_test))
                                    .at(0);
            const auto placement  = get_random_element_in(place_range);
            const bool is_real_ip = is_real(dft_kind) && placement == fft_placement_inplace;
            // working by fwd/bwd domain instead of I/O for ease of generalization
            // fwd domain := input (resp. output) domain of forward (resp. inverse) transform
            ptrdiff_t max_nembed_fwd_domain = max_length_for_hipfftw_test;
            if(rank_is_valid_for_hipfftw(rank) && nbatch > 0)
            {
                max_nembed_fwd_domain
                    = std::min(max_nembed_fwd_domain,
                               find_threshold_length_for_byte_size<prec>(
                                   max_byte_size_for_hipfftw_tests()
                                       / (max_elementary_stride_for_hipfftw_test * nbatch),
                                   rank,
                                   is_real(dft_kind)));
            }
            auto fwd_domain_nembed_range
                = arg_validation_strictly_positive_vec_range(rank, max_nembed_fwd_domain);
            if(is_real_ip && rank_is_valid_for_hipfftw(rank))
            {
                // make the last entry of the valid fwd_domain_nembed in range even
                // so that bwd_domain_nembed can be calculated
                for(auto& tmp : fwd_domain_nembed_range)
                {
                    if(vector_has_valid_values_as<int>(tmp, rank, 1) && tmp.back() % 2 == 1)
                        tmp.back()--;
                }
            }
            if(fwd_domain_nembed_range.empty())
            {
                // e.g., negative or zero rank
                fwd_domain_nembed_range.push_back(std::vector<ptrdiff_t>());
            }
            const auto fwd_domain_nembed = get_random_element_in(fwd_domain_nembed_range);
            std::vector<std::vector<ptrdiff_t>> range_of_lengths
                = arg_validation_lengths_range_many_dft(fwd_domain_nembed, is_real_ip);
            // --> test for empty lengths, too (re-interpreted as a nullptr argument by hipfftw_helper)
            range_of_lengths.push_back(std::vector<ptrdiff_t>());
            const auto lengths = get_random_element_in(range_of_lengths);
            const auto bwd_domain_nembed
                = get_random_element_in(arg_validation_bwd_domain_nembed_range_many_dft(
                    max_nembed_fwd_domain, fwd_domain_nembed, lengths, dft_kind, placement));
            std::set<ptrdiff_t> istride_range = {get_random_elementary_stride(!valid_value),
                                                 get_random_elementary_stride(valid_value)};
            if(is_real_ip)
                istride_range.insert(1);
            const auto istride = get_random_element_in(istride_range);

            std::set<ptrdiff_t> ostride_range = {get_random_elementary_stride(!valid_value),
                                                 get_random_elementary_stride(valid_value)};
            if(placement == fft_placement_inplace && istride > 0)
                ostride_range.insert(istride);
            const auto  ostride     = get_random_element_in(ostride_range);
            const auto& inembed_vec = is_fwd(dft_kind) ? fwd_domain_nembed : bwd_domain_nembed;
            const auto& onembed_vec = is_fwd(dft_kind) ? bwd_domain_nembed : fwd_domain_nembed;
            hipfftw_ionembed_t ionembed(istride, inembed_vec, ostride, onembed_vec);
            const auto         idist = get_random_element_in(arg_validation_dist_range_many_dft(
                nbatch, ionembed, fft_io::fft_io_in, get_random_dist));
            const auto         odist = get_random_element_in(arg_validation_dist_range_many_dft(
                nbatch, ionembed, fft_io::fft_io_out, get_random_dist));

            const auto sign  = get_random_element_in(arg_validation_sign_range(dft_kind));
            const auto flags = get_random_element_in(arg_validation_flags_range(dft_kind, rank));

            hipfftw_helper<prec> helper_to_add;
            helper_to_add.set_creation_args(
                dft_kind, rank, lengths, placement, sign, flags, ionembed, nbatch, idist, odist);
            ret.emplace_back(helper_to_add);
        }
        return ret;
    }

    void shuffle_vectors(std::vector<ptrdiff_t>& len,
                         std::vector<ptrdiff_t>& is,
                         std::vector<ptrdiff_t>& os,
                         bool                    all_but_last_dim = false)
    {
        if(len.size() != is.size() || len.size() != os.size())
            throw std::invalid_argument("shuffle_vectors: inconsistent vector sizes.");
        if(len.size() <= 1 || (len.size() == 2 && all_but_last_dim))
            return;
        std::vector<fftw_iodim64> layout(len.size());
        for(auto idx = len.size(); idx-- > 0;)
        {
            layout[idx].n  = len[idx];
            layout[idx].is = is[idx];
            layout[idx].os = os[idx];
        }
        std::shuffle(layout.begin(), layout.end() - (all_but_last_dim ? 1 : 0), get_pseudo_rng());
        for(auto idx = layout.size(); idx-- > 0;)
        {
            len[idx] = layout[idx].n;
            is[idx]  = layout[idx].is;
            os[idx]  = layout[idx].os;
        }
    }

    template <fft_precision prec, bool use_guru64>
    std::vector<hipfftw_helper<prec>> test_scope_for_arg_validation_of_plan_guru_dft()
    {
        constexpr int       min_unsupported_batch_rank = 2;
        constexpr ptrdiff_t max_guru_val
            = use_guru64 ? static_cast<ptrdiff_t>(std::numeric_limits<int>::max())
                         : std::numeric_limits<ptrdiff_t>::max();

        auto range_of_strides = [](const std::vector<ptrdiff_t>& lengths,
                                   fft_transform_type            dft_kind,
                                   fft_result_placement          placement,
                                   fft_io                        io) {
            std::vector<std::vector<ptrdiff_t>> ret;
            if(lengths.empty())
            {
                ret.emplace_back(std::vector<ptrdiff_t>());
                return ret;
            }
            std::vector<ptrdiff_t> abs_len = lengths;
            std::for_each(abs_len.begin(), abs_len.end(), [](ptrdiff_t& x) { x = std::abs(x); });
            auto strides_to_add = default_strides(dft_kind, placement, io, abs_len);
            ret.push_back(strides_to_add);
            // add some invalid strides (for nontrivial corresponding lengths)
            decltype(strides_to_add)::value_type tmp      = 0;
            auto                                 rand_idx = get_random_idx(strides_to_add.size());
            std::swap(strides_to_add[rand_idx], tmp);
            ret.push_back(strides_to_add);
            std::swap(strides_to_add[rand_idx], tmp);
            // add some unsupported strides
            rand_idx = get_random_idx(strides_to_add.size());
            strides_to_add[rand_idx] *= -1;
            ret.push_back(strides_to_add);
            return ret;
        };

        auto range_of_distances = [](const std::vector<ptrdiff_t>& batches,
                                     const std::vector<ptrdiff_t>& lengths,
                                     fft_transform_type            dft_kind,
                                     fft_result_placement          placement,
                                     fft_io                        io) {
            std::vector<std::vector<ptrdiff_t>> ret;
            if(batches.empty())
            {
                ret.emplace_back(std::vector<ptrdiff_t>());
                return ret;
            }
            std::vector<ptrdiff_t> abs_batches = batches;
            std::vector<ptrdiff_t> abs_lengths = lengths;
            std::for_each(
                abs_lengths.begin(), abs_lengths.end(), [](ptrdiff_t& x) { x = std::abs(x); });
            std::for_each(
                abs_batches.begin(), abs_batches.end(), [](ptrdiff_t& x) { x = std::abs(x); });
            auto distances_to_add
                = default_distances(dft_kind, placement, io, abs_lengths, abs_batches);
            ret.push_back(distances_to_add);
            // add some invalid distances (for nontrivial corresponding batch sizes)
            decltype(distances_to_add)::value_type tmp = 0;
            auto rand_idx                              = get_random_idx(distances_to_add.size());
            std::swap(distances_to_add[rand_idx], tmp);
            ret.push_back(distances_to_add);
            std::swap(distances_to_add[rand_idx], tmp);
            // add some unsupported distances
            rand_idx = get_random_idx(distances_to_add.size());
            distances_to_add[rand_idx] *= -1;
            ret.push_back(distances_to_add);
            return ret;
        };

        std::vector<hipfftw_helper<prec>> ret;

        while(ret.size() < max_num_arg_validation_tests_per_hipfftw_plan_type)
        {
            const auto dft_kind   = get_random_element_in(trans_type_range_full);
            const auto rank       = get_random_element_in(arg_validation_runtime_rank_range());
            const auto batch_rank = get_random_element_in(
                arg_validation_runtime_rank_range<min_unsupported_batch_rank>());
            // rough limit for product(batches.begin(), batches.end()) <= max_nbatch_for_hipfftw_test
            const auto batch_max_val = std::min(
                max_guru_val,
                std::max(static_cast<ptrdiff_t>(1),
                         static_cast<ptrdiff_t>(std::pow(max_nbatch_for_hipfftw_test,
                                                         1.0 / std::max(batch_rank, 1)))));
            auto batches_range
                = arg_validation_strictly_positive_vec_range(batch_rank, batch_max_val);
            // --> test for empty batches/distances, too
            batches_range.emplace_back(std::vector<ptrdiff_t>());
            auto       batches = get_random_element_in(batches_range);
            const auto nbatch_bound
                = std::accumulate(batches.begin(),
                                  batches.end(),
                                  static_cast<ptrdiff_t>(1),
                                  [](ptrdiff_t acc, ptrdiff_t x) {
                                      return acc * std::max(ptrdiff_t(1), std::abs(x));
                                  });
            const auto placement = get_random_element_in(place_range);
            ptrdiff_t  max_len   = std::min(max_guru_val / nbatch_bound,
                                         static_cast<ptrdiff_t>(max_length_for_hipfftw_test));
            if(rank_is_valid_for_hipfftw(rank))
            {
                max_len = std::min(
                    max_len,
                    find_threshold_length_for_byte_size<prec>(
                        max_byte_size_for_hipfftw_tests() / nbatch_bound, rank, is_real(dft_kind)));
            }
            auto len_range = arg_validation_strictly_positive_vec_range(rank, max_len);
            // --> test for empty lengths/strides, too
            len_range.emplace_back(std::vector<ptrdiff_t>());
            auto lengths  = get_random_element_in(len_range);
            auto istrides = get_random_element_in(
                range_of_strides(lengths, dft_kind, placement, fft_io::fft_io_in));
            auto ostrides = get_random_element_in(
                range_of_strides(lengths, dft_kind, placement, fft_io::fft_io_out));
            auto idist = get_random_element_in(
                range_of_distances(batches, lengths, dft_kind, placement, fft_io::fft_io_in));
            auto odist = get_random_element_in(
                range_of_distances(batches, lengths, dft_kind, placement, fft_io::fft_io_out));
            const auto sign  = get_random_element_in(arg_validation_sign_range(dft_kind));
            const auto flags = get_random_element_in(arg_validation_flags_range(dft_kind, rank));

            shuffle_vectors(lengths, istrides, ostrides, is_real(dft_kind));
            shuffle_vectors(batches, idist, odist);
            hipfftw_helper<prec> helper_to_add;
            helper_to_add.set_creation_args(dft_kind,
                                            rank,
                                            lengths,
                                            placement,
                                            sign,
                                            flags,
                                            istrides,
                                            ostrides,
                                            batch_rank,
                                            batches,
                                            idist,
                                            odist);
            ret.emplace_back(helper_to_add);
        }
        return ret;
    }

    // broad scope of hipfftw_helpers structs compatible with using a specific
    // plan creation function and configured with (zero or possibly many)
    // invalid/unsupported parameter value(s)
    template <fft_precision prec>
    std::vector<hipfftw_helper<prec>>
        test_scope_for_arg_validation_of(hipfftw_plan_creation_func creation_func)
    {
        constexpr bool use_guru64 = true; // for readability
        switch(creation_func)
        {
        case hipfftw_plan_creation_func::PLAN_DFT_ND:
            return test_scope_for_arg_validation_of_plan_dft_nd<prec>();
        case hipfftw_plan_creation_func::PLAN_DFT:
            return test_scope_for_arg_validation_of_plan_dft<prec>();
        case hipfftw_plan_creation_func::PLAN_MANY:
            return test_scope_for_arg_validation_of_plan_many_dft<prec>();
        case hipfftw_plan_creation_func::PLAN_GURU:
            return test_scope_for_arg_validation_of_plan_guru_dft<prec, !use_guru64>();
        case hipfftw_plan_creation_func::PLAN_GURU64:
            return test_scope_for_arg_validation_of_plan_guru_dft<prec, use_guru64>();
        default:
            throw std::invalid_argument(
                "creation_func unknown to test_scope_for_arg_validation_of");
        }
    }

    template <fft_precision prec>
    std::vector<hipfftw_plan_execution_func>
        arg_validation_exec_func_range(const hipfftw_helper<prec>& test_helper)
    {
        const static std::vector<hipfftw_plan_execution_func> possible_exec_functions
            = {hipfftw_plan_execution_func::EXECUTE,
               hipfftw_plan_execution_func::EXECUTE_DFT,
               hipfftw_plan_execution_func::EXECUTE_DFT_R2C,
               hipfftw_plan_execution_func::EXECUTE_DFT_C2R};
        std::vector<hipfftw_plan_execution_func> ret;
        for(auto can_be_used : {true, false})
        {
            auto to_add = get_random_element_in(possible_exec_functions);
            while(test_helper.can_use_execution_option(to_add) != can_be_used)
            {
                to_add = get_random_element_in(possible_exec_functions);
            }
            ret.push_back(to_add);
        }
        return ret;
    }

    template <fft_precision prec>
    std::vector<hipfftw_execution_io_args>
        arg_validation_exec_io_flags_range(const hipfftw_helper<prec>& test_helper)
    {
        auto create_possible_io_flags = []() {
            std::vector<hipfftw_execution_io_args> ret;
            for(std::underlying_type_t<hipfftw_execution_io_args> tmp
                = hipfftw_execution_io_args::use_creation_io;
                tmp < hipfftw_execution_io_args::clean_new_io;
                tmp++)
            {
                ret.push_back(static_cast<hipfftw_execution_io_args>(tmp));
            }
            return ret;
        };
        const static auto possible_exec_io_flags = create_possible_io_flags();
        // always add clean new io in range, to guarantee one testable combo
        return {hipfftw_execution_io_args::clean_new_io,
                get_random_element_in(possible_exec_io_flags)};
    }

    template <fft_precision prec>
    std::vector<hipfftw_input_validation_params<prec>> params_for_testing_input_validation_params()
    {
        // create a full-scope map containing all the generated test parameters; the map keys
        // capture the hipfftw's function name that the test targets
        // --> ease for guaranteeing coverage even with low test probability in the end
        std::map<std::string, std::vector<hipfftw_input_validation_params<prec>>> full_scope_tests;
        hipfftw_input_validation_params<prec>                                     test_to_add;

        const std::vector<std::pair<bool, bool>> possible_creation_io_is_null_inplace
            = {{false, false}, {true, true}};
        const std::vector<std::pair<bool, bool>> possible_creation_io_is_null_notinplace
            = {{false, false}, {true, false}, {false, true}};

        for(auto creation_func : {hipfftw_plan_creation_func::PLAN_DFT_ND,
                                  hipfftw_plan_creation_func::PLAN_DFT,
                                  hipfftw_plan_creation_func::PLAN_MANY,
                                  hipfftw_plan_creation_func::PLAN_GURU,
                                  hipfftw_plan_creation_func::PLAN_GURU64})
        {
            for(const auto& helper : test_scope_for_arg_validation_of<prec>(creation_func))
            {
                const std::vector<std::pair<bool, bool>>& possible_creation_io_is_null
                    = helper.get_placement() == fft_placement_inplace
                          ? possible_creation_io_is_null_inplace
                          : possible_creation_io_is_null_notinplace;
                for(auto exec_func : arg_validation_exec_func_range(helper))
                {
                    for(auto exec_io_flags : arg_validation_exec_io_flags_range(helper))
                    {
                        test_to_add.creation_options = creation_func;
                        test_to_add.execution_option = exec_func;
                        test_to_add.creation_io_is_null
                            = get_random_element_in(possible_creation_io_is_null);
                        test_to_add.execution_io = exec_io_flags;
                        test_to_add.plan_helper  = helper;
                        // skip params if they can't/shouldn't be tested anyways
                        if(!test_to_add.can_be_tested())
                            continue;

                        // tests expect a failure at execution at least
                        if(test_to_add.expected_internal_exception_for(hipfftw_step::plan_execution)
                           == hipfftw_internal_exception::none)
                            continue;
                        std::string map_key;
                        if(test_to_add.expected_internal_exception_for(hipfftw_step::plan_creation)
                               == hipfftw_internal_exception::invalid_args
                           || test_to_add.expected_internal_exception_for(
                                  hipfftw_step::plan_creation)
                                  == hipfftw_internal_exception::unsupported_args)
                        {
                            map_key = hipfftw_creation_options_to_string(
                                creation_func, helper.get_dft_kind(), helper.get_rank());
                        }
                        else
                        {
                            map_key = hipfftw_execution_option_to_string(exec_func);
                        }
                        insert_into_unique_sorted_params(full_scope_tests[map_key], test_to_add);
                    }
                }
            }
        }
        std::vector<hipfftw_input_validation_params<prec>> ret;
        for(auto& pair : full_scope_tests)
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
            try
            {
                const hipfftw_input_validation_params<prec>& params = this->GetParam();

                if(!params.can_be_tested())
                    GTEST_FAIL() << "invalid parameters which cannot be tested";
                safe_to_touch_nonnull_io = true;

                // get_data_byte_size requires valid ranks and lengths to be calculated (of course)
                // --> make sure the I/O data sizes are not zero for test consistency w.r.t. testing
                // for nullptr data args I/O
                auto compute_io_size = [&](fft_io io) {
                    try
                    {
                        return params.plan_helper.get_data_byte_size(io);
                    }
                    catch(const typename hipfftw_helper<prec>::num_elements_calc_exception& e)
                    {
                        safe_to_touch_nonnull_io = false;
                        return sizeof(hipfftw_complex_t<prec>); // some nonzero value
                    }
                };
                const size_t input_data_size  = compute_io_size(fft_io::fft_io_in);
                const size_t output_data_size = compute_io_size(fft_io::fft_io_out);
                if(input_data_size > max_byte_size_for_hipfftw_tests()
                   || output_data_size > max_byte_size_for_hipfftw_tests())
                {
                    GTEST_SKIP()
                        << "Skipping test due to excessive I/O byte size (max of I/O byte size: "
                        << std::max(input_data_size, output_data_size)
                        << " vs limit: " << max_byte_size_for_hipfftw_tests() << ")";
                }

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
            ROCFFT_CATCH_TEST_EXCEPTIONS;
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
        bool    safe_to_touch_nonnull_io;

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
                void* creation_in  = plan_creation_input.data();
                void* creation_out = params.creation_placement() == fft_placement_inplace
                                         ? plan_creation_input.data()
                                         : plan_creation_output.data();
                if((creation_in || creation_out)
                   && flags_are_valid_for_hipfftw(params.plan_helper.get_flags())
                   && !(params.plan_helper.get_flags() & FFTW_ESTIMATE)
                   && !safe_to_touch_nonnull_io)
                {
                    // I/O may be touched during plan creation, yet their size couldn't be clearly determined
                    GTEST_SKIP() << "unsafe to test for these parameters (plan creation)";
                }
                params.plan_helper.create_plan(creation_in, creation_out, params.creation_options);
                if(step_target == hipfftw_step::plan_creation)
                {
                    log_content = exception_logger->get_log();
                    exception_logger.reset();
                    if(params.plan_helper.get_plan())
                        throw std::runtime_error(
                            hipfftw_creation_options_to_string(
                                params.plan_helper.get_plan_creation_function(),
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

                    if(params.plan_helper.get_plan() && (creation_in || creation_out)
                       && !safe_to_touch_nonnull_io)
                    {
                        // A plan was successfully created and I/O may be touched during plan execution,
                        // yet their size(s) couldn't be clearly determined
                        GTEST_SKIP() << "unsafe to test for these parameters (plan execution)";
                    }
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
            catch(const std::runtime_error& e)
            {
                if(log_content.empty() && exception_logger)
                    log_content = exception_logger->get_log();
                std::ostringstream gtest_info;
                gtest_info << e.what();
                if(!log_content.empty())
                    gtest_info << "\nContent of error log:\n" << log_content;
                GTEST_FAIL() << gtest_info.str();
            }
            catch(const std::bad_alloc&)
            {
                GTEST_SKIP() << "host memory allocation failure";
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
#ifndef _WIN32
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
#ifndef _WIN32
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
#ifndef _WIN32
        case hipfftw_data_memory_type::managed:
            return "managed";
            break;
#endif
        case hipfftw_data_memory_type::device:
            return "device";
            break;
        default:
            throw std::runtime_error(
                "internal error: unexpected value of mem_type in hipfftw_data_mem_type_to_string");
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

        hipfftw_functional_validation_params()
            : manually_created(false){};

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
            return plan_helper.template get_lengths_as<size_t>();
        }
        std::vector<size_t> get_ilengths() const
        {
            auto ilengths = get_lengths();
            if(plan_helper.get_dft_kind() == fft_transform_type_real_inverse)
                ilengths.back() = ilengths.back() / 2 + 1;
            return ilengths;
        }
        std::vector<size_t> get_olengths() const
        {
            auto olengths = get_lengths();
            if(plan_helper.get_dft_kind() == fft_transform_type_real_forward)
                olengths.back() = olengths.back() / 2 + 1;
            return olengths;
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
            return plan_helper.get_distance(fft_io::fft_io_in);
        }
        size_t get_odist() const
        {
            return plan_helper.get_distance(fft_io::fft_io_out);
        }
        size_t get_nbatch() const
        {
            return plan_helper.get_nbatch();
        }
        std::vector<size_t> get_contiguous_istrides() const
        {
            std::vector<size_t> contiguous_istrides(get_rank());
            const auto          ilen   = get_ilengths();
            contiguous_istrides.back() = 1;
            for(auto dim = get_rank() - 1; dim-- > 0;)
            {
                contiguous_istrides[dim] = contiguous_istrides[dim + 1] * ilen[dim + 1];
            }
            return contiguous_istrides;
        }
        size_t get_contiguous_idist() const
        {
            const auto cont_istrides = get_contiguous_istrides();
            const auto ilen          = get_ilengths();
            return ilen.front() * cont_istrides.front();
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
            if(manually_created)
                ret << "manual_";
            ret << plan_helper.token();
            ret << "_" << creation_input_mem_type_label << "_"
                << hipfftw_data_mem_type_to_string(
                       mem_type.at({hipfftw_step::plan_creation, fft_io::fft_io_in}));
            if(plan_helper.get_placement() == fft_placement_notinplace)
            {
                ret << "_" << creation_output_mem_type_label << "_"
                    << hipfftw_data_mem_type_to_string(
                           mem_type.at({hipfftw_step::plan_creation, fft_io::fft_io_out}));
            }
            if(execution_io == hipfftw_execution_io_args::clean_new_io)
            {
                ret << "_" << execution_input_mem_type_label << "_"
                    << hipfftw_data_mem_type_to_string(
                           mem_type.at({hipfftw_step::plan_execution, fft_io::fft_io_in}));
                if(plan_helper.get_placement() == fft_placement_notinplace)
                {
                    ret << "_" << execution_output_mem_type_label << "_"
                        << hipfftw_data_mem_type_to_string(
                               mem_type.at({hipfftw_step::plan_execution, fft_io::fft_io_out}));
                }
            }
            return ret.str();
        }

        // constructor from token
        hipfftw_functional_validation_params(const std::string& manual_token)
            : manually_created(true)
        {
            plan_helper.from_token(manual_token);

            auto get_mem_type_from_str = [&](const std::string_view& which_io_label) {
                std::ostringstream failure_info;
                auto               pos = manual_token.find(which_io_label);
                if(pos == std::string::npos)
                {
                    failure_info << which_io_label << " absent from manual token (" << manual_token
                                 << ")";
                    throw std::runtime_error(failure_info.str());
                }
                pos += which_io_label.size() + 1; // +1 for the '_' delimiter
                for(auto tmp : get_possible_data_mem_types())
                {
                    const auto match = hipfftw_data_mem_type_to_string(tmp);
                    if(manual_token.find(match, pos) == pos)
                        return tmp;
                }

                failure_info
                    << "A type of memory allocation testable by hipfftw cannot be determined from "
                    << manual_token
                    << ": the (partial) token might be invalid or the targeted type of memory is "
                       "not accessible on this platform.";
                throw std::runtime_error(failure_info.str());
            };

            execution_io = manual_token.find(execution_input_mem_type_label) != std::string::npos
                               ? hipfftw_execution_io_args::clean_new_io
                               : hipfftw_execution_io_args::use_creation_io;

            mem_type[{hipfftw_step::plan_creation, fft_io::fft_io_in}]
                = get_mem_type_from_str(creation_input_mem_type_label);
            if(plan_helper.get_placement() == fft_placement_notinplace)
                mem_type[{hipfftw_step::plan_creation, fft_io::fft_io_out}]
                    = get_mem_type_from_str(creation_output_mem_type_label);
            else
                mem_type[{hipfftw_step::plan_creation, fft_io::fft_io_out}]
                    = mem_type[{hipfftw_step::plan_creation, fft_io::fft_io_in}];
            if(execution_io == hipfftw_execution_io_args::clean_new_io)
            {
                mem_type[{hipfftw_step::plan_execution, fft_io::fft_io_in}]
                    = get_mem_type_from_str(execution_input_mem_type_label);
                if(plan_helper.get_placement() == fft_placement_notinplace)
                    mem_type[{hipfftw_step::plan_execution, fft_io::fft_io_out}]
                        = get_mem_type_from_str(execution_output_mem_type_label);
                else
                    mem_type[{hipfftw_step::plan_execution, fft_io::fft_io_out}]
                        = mem_type[{hipfftw_step::plan_execution, fft_io::fft_io_in}];
            }
            else
            {
                for(auto io : {fft_io::fft_io_in, fft_io::fft_io_out})
                    mem_type[{hipfftw_step::plan_execution, io}]
                        = mem_type[{hipfftw_step::plan_creation, io}];
            }
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
        friend std::ostream& operator<<(std::ostream&                               stream,
                                        const hipfftw_functional_validation_params& params)
        {
            stream << params.to_string();
            return stream;
        }

        bool is_manual_test() const
        {
            return manually_created;
        }

    private:
        static constexpr std::string_view creation_input_mem_type_label = "creation_input_mem_type";
        static constexpr std::string_view creation_output_mem_type_label
            = "creation_output_mem_type";
        static constexpr std::string_view execution_input_mem_type_label
            = "execution_input_mem_type";
        static constexpr std::string_view execution_output_mem_type_label
            = "execution_output_mem_type";
        bool manually_created;
    };

#ifdef __HIP_PLATFORM_AMD__
    template <fft_precision prec>
    bool is_known_defect(const hipfftw_helper<prec>& helper)
    {
        bool       ret      = false;
        const auto dft_kind = helper.get_dft_kind();
        const auto len      = helper.get_lengths();
        if(helper.get_rank() == 3 && is_complex(dft_kind) && !helper.is_using_default_strides())
        {
            // rocfft can't create some plans with non-default strides for lengths
            // AxBxC wherein the two lengths of smallest input or output strides are in
            const std::vector<std::vector<ptrdiff_t>> symptomatic_sub_len
                = {{16, 4},  {4, 16},  {16, 16}, {27, 4},  {4, 27},  {25, 4},  {4, 25},
                   {16, 25}, {25, 16}, {25, 25}, {8, 9},   {9, 8},   {8, 4},   {4, 8},
                   {8, 8},   {4, 9},   {9, 4},   {4, 4},   {20, 10}, {10, 20}, {27, 27},
                   {32, 9},  {9, 32},  {32, 4},  {4, 32},  {32, 8},  {8, 32},  {27, 9},
                   {9, 27},  {16, 9},  {9, 16},  {25, 32}, {32, 25}, {9, 9},   {25, 9},
                   {9, 25},  {16, 8},  {8, 16},  {16, 32}, {32, 16}, {25, 8},  {8, 25}};
            // (Note: failing lengths usually have a value of length involving a prime factor > 17
            // along the slowest dimension.  See adhoc tokens in rocfft's disabled suite of adhoc
            // accuracy tests)
            for(auto io : {fft_io::fft_io_in, fft_io::fft_io_out})
            {
                const auto& strides = helper.get_strides(io);
                // get the sub-lengths corresponding of the fastest-varying dimensions
                auto sub_len = len;
                sub_len.erase(sub_len.begin()
                              + std::distance(strides.begin(),
                                              std::max_element(strides.begin(), strides.end())));
                ret = ret
                      || std::find(symptomatic_sub_len.begin(), symptomatic_sub_len.end(), sub_len)
                             != symptomatic_sub_len.end();
            }
        }
        if(helper.get_rank() > 1 && is_real(dft_kind) && !helper.is_using_default_strides())
        {
            // rocfft can't create some plans with non-default strides for real transforms of
            // lengths in the following sets. See adhoc tokens in rocfft's disabled suite of
            // adhoc accuracy tests.
            // - 3D size Ax1xB
            ret = ret || (helper.get_rank() == 3 && len[1] == 1);
            // - 2D sizes in the following set
            const std::vector<std::vector<ptrdiff_t>> symptomatic_real_lengths
                = {{25, 8},  {9, 54},  {81, 18}, {25, 16}, {8, 18}, {64, 8},  {64, 16},
                   {27, 16}, {25, 32}, {9, 16},  {8, 8},   {9, 32}, {20, 20}, {16, 8},
                   {32, 8},  {81, 64}, {16, 16}, {4, 18},  {9, 8},  {32, 18}};

            ret = ret
                  || std::any_of(symptomatic_real_lengths.begin(),
                                 symptomatic_real_lengths.end(),
                                 [&](const std::vector<ptrdiff_t>& len_to_skip) {
                                     return len_to_skip == len;
                                 });
        }
        if(is_real(dft_kind) && len.back() % 2 == 0
           && (helper.get_nbatch() > 1 || product(len.begin(), len.end() - 1) > 1)
           && helper.get_strides(fft_io::fft_io_in).back() == 1
           && helper.get_strides(fft_io::fft_io_out).back() == 1)
        {
            // rocfft can't handle odd values of (non-elementary) strides/distances
            // in real domain, with even values of lengths.back() for real transforms.
            // Incorrect results are generated
            auto tmp
                = helper.get_strides(is_fwd(dft_kind) ? fft_io::fft_io_in : fft_io::fft_io_out);
            tmp.erase(tmp.end() - 1); // remove elementary strides from considerations
            tmp.push_back(
                helper.get_distance(is_fwd(dft_kind) ? fft_io::fft_io_in : fft_io::fft_io_out));

            ret = ret
                  || std::any_of(
                      tmp.begin(), tmp.end(), [](const typename decltype(tmp)::value_type& val) {
                          return val % 2 == 1;
                      });
        }
        // incorrect results may be produced by rocfft for some of the following cases
        if(is_complex(dft_kind))
        {
            // see adhoc tokens in rocfft's disabled suite of adhoc accuracy tests
            if(helper.is_using_default_strides())
            {
                ret = ret
                      || (helper.get_rank() == 2 && len[0] == 81 && len[1] == 74
                          && !helper.is_using_default_distances());
            }
            else
            {
                ret = ret || (helper.get_rank() == 2 && len[0] == 50 && len[1] == 129);
            }
        }
        else
        {
            if(helper.get_rank() == 1 && helper.get_nbatch() > 1
               && helper.get_distance(fft_io::fft_io_in) == 1
               && helper.get_distance(fft_io::fft_io_out) == 1)
            {
                // inner batch 1D: see adhoc token in rocfft's disabled suite of adhoc accuracy tests
                ret = ret || len[0] == 486;
            }
            if(helper.get_rank() > 1 && !helper.is_using_default_strides())
            {
                if(helper.get_rank() == 2)
                {
                    ret = ret || (len[0] == 26 && len[1] == 52);
                }
                if(helper.get_rank() == 3)
                {
                    // may fail if the last length is a multiple of a prime larger than 16. Examples
                    // of failing lengths: 15x12x76, 7x14x76, 11x12x38, 20x8x38, 5x12x38, 6x24x38,
                    // 6x25x38, 26x26x134, 2x40x62, 36x13x46, 2x5x46, 6x12x46, 2x14x46 (see adhoc
                    // tokens in rocfft's disabled suite of adhoc accuracy tests)
                    auto tmp = helper.get_lengths().back();
                    if(tmp % 2 == 0)
                    {
                        auto largest_prime_factor = 2;
                        while(tmp != 1)
                        {
                            while(tmp % largest_prime_factor == 0)
                            {
                                tmp /= largest_prime_factor;
                            }
                            if(tmp != 1)
                                largest_prime_factor++;
                        }
                        ret = ret || largest_prime_factor > 16;
                    }
                }
            }
        }
        return ret;
    }
#endif

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
#ifdef __HIP_PLATFORM_AMD__
                if(!params.is_manual_test() && is_known_defect(params.plan_helper))
                    GTEST_SKIP()
                        << "skipped automatically-generated test due to known rocFFT defect";
#endif
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
#ifndef _WIN32
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
                const std::vector<size_t> ioffset = {0};
                const std::vector<size_t> field_lower(params.get_rank(), 0);
                set_input<hostbuf, hipfftw_real_t<prec>>(verification_input,
                                                         fft_input_random_generator_host,
                                                         params.get_array_type(fft_io::fft_io_in),
                                                         params.get_lengths(),
                                                         params.get_ilengths(),
                                                         params.get_istride(),
                                                         ioffset,
                                                         params.get_idist(),
                                                         params.get_nbatch(),
                                                         get_curr_device_prop(),
                                                         field_lower,
                                                         0 /* field_lower_batch */,
                                                         params.get_contiguous_istrides(),
                                                         params.get_contiguous_idist());
                // create the reference plan (systematically using the most general guru64 creation)
                reference_plan = fftw_trait<hipfftw_real_t<prec>>::make_wrapper(
                    params.plan_helper.get_reference_plan(verification_input[0].data(),
                                                          params.get_placement()
                                                                  == fft_placement_inplace
                                                              ? verification_input[0].data()
                                                              : verification_output[0].data()));

                if(!reference_plan)
                {
                    GTEST_FAIL() << "could not create a reference plan";
                }
            }
            ROCFFT_CATCH_TEST_EXCEPTIONS;
        }
        void TearDown() override
        {
            verification_input.clear();
            verification_output.clear();
            execution_results_on_host.clear();
            host_io_buffer.clear();
            gpu_io_buffer.clear();
            reference_plan.reset();
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
        fftw_plan_wrapper_t<hipfftw_real_t<prec>> reference_plan
            = fftw_trait<hipfftw_real_t<prec>>::make_wrapper(nullptr);

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
                    fftw_execute_type<hipfftw_real_t<prec>>(reference_plan);
                });
                hipfftw_exception_logger exception_logger;

                params.plan_helper.create_plan(
                    test_io_ptr.at({hipfftw_step::plan_creation, fft_io::fft_io_in}),
                    test_io_ptr.at({hipfftw_step::plan_creation, fft_io::fft_io_out}));
                if(!params.plan_helper.get_plan())
                {
                    std::ostringstream gtest_info;
                    gtest_info << "Plan creation failed";
                    if(exception_logger.is_active())
                    {
                        const auto log_content = exception_logger.get_log();
                        if(!log_content.empty())
                        {
                            gtest_info << "\nNon-empty log content detected:\n" << log_content;
                        }
                    }
                    GTEST_FAIL() << gtest_info.str();
                }
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
                const auto  test_lengths     = params.get_lengths();
                const auto  total_length     = product(test_lengths.begin(), test_lengths.end());
                const auto& reference_output = params.get_placement() == fft_placement_inplace
                                                   ? verification_input
                                                   : verification_output;

                const auto   ref_norm = norm(reference_output,
                                           params.get_olengths(),
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
                                           params.get_olengths(),
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
            ROCFFT_CATCH_TEST_EXCEPTIONS;
        }

    public:
        static std::string TestName(
            const testing::TestParamInfo<typename hipfftw_functional_validation::ParamType>& info)
        {
            return info.param.to_string();
        }
    };

    template <fft_precision prec>
    void setup_random_default_plan(hipfftw_helper<prec>& helper, bool unbatched)
    {
        const auto    dft_kind   = get_random_element_in(trans_type_range_full);
        const auto    rank       = get_random_rank<valid_value, 1, 3>();
        const auto    placement  = get_random_element_in(place_range);
        constexpr int batch_rank = 1;
        const auto    batches    = get_random_vector<valid_value, int>(
            batch_rank, unbatched ? 1 : max_nbatch_for_hipfftw_test, 1);
        const size_t    max_data_size_per_batch = max_byte_size_for_hipfftw_tests() / batches[0];
        const ptrdiff_t max_len = std::min(static_cast<ptrdiff_t>(max_length_for_hipfftw_test),
                                           find_threshold_length_for_byte_size<prec>(
                                               max_data_size_per_batch, rank, is_real(dft_kind)));
        const auto      lengths = get_random_vector<valid_value, int>(rank, max_len);

        helper.set_creation_args(
            dft_kind,
            rank,
            lengths,
            placement,
            is_fwd(dft_kind) ? FFTW_FORWARD : FFTW_BACKWARD,
            FFTW_ESTIMATE,
            default_strides(dft_kind, placement, fft_io::fft_io_in, lengths),
            default_strides(dft_kind, placement, fft_io::fft_io_out, lengths),
            batch_rank,
            batches,
            default_distances(dft_kind, placement, fft_io::fft_io_in, lengths, batches),
            default_distances(dft_kind, placement, fft_io::fft_io_out, lengths, batches));
    }

    template <fft_precision prec>
    void setup_random_nembed_compliant_plan(hipfftw_helper<prec>& helper)
    {
        std::uniform_int_distribution<ptrdiff_t> stride_rng(1,
                                                            max_elementary_stride_for_hipfftw_test);
        const auto    dft_kind   = get_random_element_in(trans_type_range_full);
        const auto    rank       = get_random_rank<valid_value, 1, 3>();
        const auto    placement  = get_random_element_in(place_range);
        constexpr int batch_rank = 1;
        const auto    batches
            = get_random_vector<valid_value, int>(batch_rank, max_nbatch_for_hipfftw_test, 1);
        // working by fwd/bwd domain instead of I/O for ease of generalization
        // fwd domain := input (resp. output) domain of forward (resp. inverse) transform
        const auto   elementary_fwd_stride = is_real(dft_kind) && placement == fft_placement_inplace
                                                 ? 1
                                                 : stride_rng(get_pseudo_rng());
        const auto   elementary_bwd_stride = placement == fft_placement_inplace
                                                 ? elementary_fwd_stride
                                                 : stride_rng(get_pseudo_rng());
        const size_t max_data_size_per_batch
            = max_byte_size_for_hipfftw_tests()
              / (std::max(elementary_bwd_stride, elementary_bwd_stride) * batches[0]);
        const ptrdiff_t max_nembed_fwd_domain
            = std::min(static_cast<ptrdiff_t>(max_length_for_hipfftw_test),
                       find_threshold_length_for_byte_size<prec>(
                           max_data_size_per_batch, rank, is_real(dft_kind)));
        // The generation of random, constraints-abiding-yet-nembed-compliant data
        // layouts may sometimes run into deadends (catch a valid_values_cannot_be_created
        // exception) --> try again until a valid (random) configuration is found, if so
        bool found_one = false;
        while(!found_one)
        {
            try
            {
                auto fwd_nembed
                    = get_random_vector<valid_value, int>(rank, max_nembed_fwd_domain, 1);
                if(is_real(dft_kind) && placement == fft_placement_inplace)
                {
                    // fwd_nembed.back() * elementary_fwd_stride must be even
                    if((fwd_nembed.back() * elementary_fwd_stride) % 2 == 1)
                        fwd_nembed.back()--;
                }
                auto lengths = get_random_lengths_from_fwd_domain_nembed<valid_value>(
                    fwd_nembed, is_real(dft_kind) && placement == fft_placement_inplace);
                const auto bwd_nembed = get_random_bwd_domain_nembed<valid_value>(
                    max_nembed_fwd_domain, fwd_nembed, lengths, dft_kind, placement);
                hipfftw_ionembed_t ionembed(
                    is_fwd(dft_kind) ? elementary_fwd_stride : elementary_bwd_stride,
                    is_fwd(dft_kind) ? fwd_nembed : bwd_nembed,
                    is_fwd(dft_kind) ? elementary_bwd_stride : elementary_fwd_stride,
                    is_fwd(dft_kind) ? bwd_nembed : fwd_nembed);
                const auto idist = product(ionembed.get_nembed(fft_io::fft_io_in),
                                           ionembed.get_nembed(fft_io::fft_io_in) + rank)
                                   * ionembed.get_elementary_stride(fft_io::fft_io_in);
                const auto odist = product(ionembed.get_nembed(fft_io::fft_io_out),
                                           ionembed.get_nembed(fft_io::fft_io_out) + rank)
                                   * ionembed.get_elementary_stride(fft_io::fft_io_out);

                helper.set_creation_args(dft_kind,
                                         rank,
                                         lengths,
                                         placement,
                                         is_fwd(dft_kind) ? FFTW_FORWARD : FFTW_BACKWARD,
                                         FFTW_ESTIMATE,
                                         ionembed,
                                         batches[0],
                                         idist,
                                         odist);
                found_one = true;
            }
            catch(const valid_values_cannot_be_created& e)
            {
                // try again
                found_one = false;
            }
        }
    }

    // always nembed-compatible
    template <fft_precision prec>
    void setup_inner_batched_plan(hipfftw_helper<prec>& helper)
    {
        const auto dft_kind = get_random_element_in(trans_type_range_full);
        // TODO: re-enable 1D for complex DFTs as well once rocfft can reliably handle those
        const auto rank = is_complex(dft_kind) ? get_random_rank<valid_value, 2, 3>()
                                               : get_random_rank<valid_value, 1, 3>();
        // real in-place requires unit strides, hence out-of-place only for inner-batched, real transforms
        const auto placement
            = is_real(dft_kind) ? fft_placement_notinplace : get_random_element_in(place_range);
        constexpr int batch_rank = 1;
        const auto    batches
            = get_random_vector<valid_value, int>(batch_rank, max_nbatch_for_hipfftw_test, 1);
        constexpr ptrdiff_t dist = 1; // same in fwd and bwd domains
        const size_t        max_data_size_per_batch
            = max_byte_size_for_hipfftw_tests() / (dist * batches[0]);
        const ptrdiff_t max_len    = std::min(static_cast<ptrdiff_t>(max_length_for_hipfftw_test),
                                           find_threshold_length_for_byte_size<prec>(
                                               max_data_size_per_batch, rank, is_real(dft_kind)));
        const auto      lengths    = get_random_vector<valid_value, int>(rank, max_len, 1);
        auto            fwd_nembed = lengths;
        auto            bwd_nembed = lengths;
        if(is_real(dft_kind))
            bwd_nembed.back() = bwd_nembed.back() / 2 + 1;

        hipfftw_ionembed_t ionembed(batches[0] * dist,
                                    is_fwd(dft_kind) ? fwd_nembed : bwd_nembed,
                                    batches[0] * dist,
                                    is_fwd(dft_kind) ? bwd_nembed : fwd_nembed);

        helper.set_creation_args(dft_kind,
                                 rank,
                                 lengths,
                                 placement,
                                 is_fwd(dft_kind) ? FFTW_FORWARD : FFTW_BACKWARD,
                                 FFTW_ESTIMATE,
                                 ionembed,
                                 batches[0],
                                 dist,
                                 dist);
    }

    // guru-/guru64-compatible plan (only).
    // plan configurations with packed layouts but shuffling the natural
    // dimension ordering, e.g., using column-major layouts
    template <fft_precision prec>
    void setup_nondefault_dim_ordering_plan(hipfftw_helper<prec>& helper)
    {
        constexpr int batch_rank = 1; // nothing else is supported

        auto get_random_shuffling_indices = [](int rank, bool is_real_in_place) {
            if(rank < 1)
                throw std::invalid_argument("setup_nondefault_dim_ordering_plan::get_random_"
                                            "shuffling_indices: unexpected rank.");
            // shuffle the natural ordering:
            std::vector<size_t> dimension_ordering(rank);
            std::iota(dimension_ordering.begin(), dimension_ordering.end(), 0);
            std::shuffle(dimension_ordering.begin(),
                         dimension_ordering.end() - (is_real_in_place ? 1 : 0),
                         get_pseudo_rng());
            return dimension_ordering;
        };

        // 1D cases are never guru-only cases
        const int  rank     = get_random_rank<valid_value, 2, 3>();
        const auto dft_kind = get_random_element_in(trans_type_range_full);
        // real in-place requires unit elementary strides, so rule it out for 2D real cases
        const auto placement  = rank == 2 && is_real(dft_kind) ? fft_placement_notinplace
                                                               : get_random_element_in(place_range);
        const auto is_real_ip = is_real(dft_kind) && placement == fft_placement_inplace;
        const auto batches
            = get_random_vector<valid_value, ptrdiff_t>(batch_rank, max_nbatch_for_hipfftw_test, 1);
        const size_t    max_data_size_per_batch = max_byte_size_for_hipfftw_tests() / batches[0];
        const ptrdiff_t max_len = std::min(static_cast<ptrdiff_t>(max_length_for_hipfftw_test),
                                           find_threshold_length_for_byte_size<prec>(
                                               max_data_size_per_batch, rank, is_real(dft_kind)));
        const auto      lengths = get_random_vector<valid_value, ptrdiff_t>(rank, max_len, 1);
        auto            shuffling_indices = get_random_shuffling_indices(rank, is_real_ip);
        const auto      istrides
            = default_strides(dft_kind, placement, fft_io::fft_io_in, lengths, shuffling_indices);
        const auto idist = default_distances(
            dft_kind, placement, fft_io::fft_io_in, lengths, batches, shuffling_indices);
        if(placement == fft_placement_notinplace)
        {
            // can use different shuffling on output
            shuffling_indices = get_random_shuffling_indices(rank, is_real_ip);
        }
        const auto ostrides
            = default_strides(dft_kind, placement, fft_io::fft_io_out, lengths, shuffling_indices);
        const auto odist = default_distances(
            dft_kind, placement, fft_io::fft_io_out, lengths, batches, shuffling_indices);

        helper.set_creation_args(dft_kind,
                                 rank,
                                 lengths,
                                 placement,
                                 is_fwd(dft_kind) ? FFTW_FORWARD : FFTW_BACKWARD,
                                 FFTW_ESTIMATE,
                                 istrides,
                                 ostrides,
                                 batch_rank,
                                 batches,
                                 idist,
                                 odist);
    }

    // guru-/guru64-compatible plan (only).
    // plan configurations with not-quite-default 3D layouts: slowest-dimension's stride is
    // added some arbitrary values making the layout nembed-incompatible (3D only since lesser
    // similar rank cases are always nembed-compatible)
    template <fft_precision prec>
    void setup_tweaked_default_batched_plan(hipfftw_helper<prec>& helper)
    {
        constexpr int rank       = 3;
        constexpr int batch_rank = 1;
        const auto    dft_kind   = get_random_element_in(trans_type_range_full);
        const auto    placement  = get_random_element_in(place_range);
        const bool    is_real_ip = is_real(dft_kind) && placement == fft_placement_inplace;
        // min original batch value of 2 (halved afterwards)
        auto batches
            = get_random_vector<valid_value, int>(batch_rank, max_nbatch_for_hipfftw_test, 2);
        const size_t    max_data_size_per_batch = max_byte_size_for_hipfftw_tests() / batches[0];
        const ptrdiff_t max_len = std::min(static_cast<ptrdiff_t>(max_length_for_hipfftw_test),
                                           find_threshold_length_for_byte_size<prec>(
                                               max_data_size_per_batch, rank, is_real(dft_kind)));
        auto            lengths = get_random_vector<valid_value, int>(rank, max_len);
        while(lengths[rank - 1] == 1)
        {
            // AxBx1 cannot be made nembed-incompatible by simply
            // tweaking the slowest dimension's (A's) stride
            lengths = get_random_vector<valid_value, int>(rank, max_len);
        }
        // parameters to tweak (initialized to default values)
        auto istrides = default_strides(dft_kind, placement, fft_io::fft_io_in, lengths);
        auto ostrides = default_strides(dft_kind, placement, fft_io::fft_io_out, lengths);
        auto idist    = default_distances(dft_kind, placement, fft_io::fft_io_in, lengths, batches);
        auto odist = default_distances(dft_kind, placement, fft_io::fft_io_out, lengths, batches);
        // buy some wiggle room: double distances and halve the batch size
        idist[0] *= 2;
        odist[0] *= 2;
        batches[0] /= 2;
        // tweak slowest dimension's stride
        const auto min_slow_dim_istride
            = istrides.front() + (is_real_ip && is_fwd(dft_kind) ? 2 : 1);
        const auto min_slow_dim_ostride
            = ostrides.front() + (is_real_ip && is_bwd(dft_kind) ? 2 : 1);
        const auto                    max_slow_dim_istride = 2 * istrides.front();
        const auto                    max_slow_dim_ostride = 2 * ostrides.front();
        std::uniform_int_distribution i_stride_gen(min_slow_dim_istride, max_slow_dim_istride);
        std::uniform_int_distribution o_stride_gen(min_slow_dim_ostride, max_slow_dim_ostride);
        while(istrides[0] % istrides[1] == 0 && ostrides[0] % ostrides[1] == 0)
        {
            istrides[0] = i_stride_gen(get_pseudo_rng());
            if(placement == fft_placement_inplace)
            {
                if(is_real(dft_kind))
                {
                    if(is_fwd(dft_kind))
                    {
                        if(istrides[0] % 2 == 1)
                            istrides[0]--;
                        ostrides[0] = istrides[0] / 2;
                    }
                    else
                        ostrides[0] = 2 * istrides[0];
                }
                else
                    ostrides[0] = istrides[0];
            }
            else
                ostrides[0] = o_stride_gen(get_pseudo_rng());
        }

        helper.set_creation_args(dft_kind,
                                 rank,
                                 lengths,
                                 placement,
                                 is_fwd(dft_kind) ? FFTW_FORWARD : FFTW_BACKWARD,
                                 FFTW_ESTIMATE,
                                 istrides,
                                 ostrides,
                                 batch_rank,
                                 batches,
                                 idist,
                                 odist);
    }

    template <fft_precision prec>
    std::vector<hipfftw_functional_validation_params<prec>>
        params_for_functional_tests(size_t desired_full_suite_size, const std::string& manual_token)
    {
        std::vector<hipfftw_functional_validation_params<prec>> full_list;
        hipfftw_functional_validation_params<prec>              to_add;

        enum class test_layout
        {
            default_unbatched,
            default_batched,
            random_nembed_compatible,
            inner_batched,
            nondefault_dim_ordering,
            tweaked_default_batched
        };
        const std::vector<test_layout> possible_test_layouts
            = {test_layout::default_unbatched,
               test_layout::default_batched,
               test_layout::random_nembed_compatible,
               test_layout::inner_batched,
               test_layout::nondefault_dim_ordering,
               test_layout::tweaked_default_batched};
        std::uniform_int_distribution<int> coin_toss(0, 1);
        const auto&                        possible_mem_types = get_possible_data_mem_types();
        while(full_list.size() < desired_full_suite_size)
        {
            to_add.execution_io    = coin_toss(get_pseudo_rng()) == 0
                                         ? hipfftw_execution_io_args::use_creation_io
                                         : hipfftw_execution_io_args::clean_new_io;
            const auto data_layout = get_random_element_in(possible_test_layouts);
            switch(data_layout)
            {
            case test_layout::default_unbatched:
                [[fallthrough]];
            case test_layout::default_batched:
                setup_random_default_plan(to_add.plan_helper,
                                          data_layout == test_layout::default_unbatched);
                break;
            case test_layout::random_nembed_compatible:
                setup_random_nembed_compliant_plan(to_add.plan_helper);
                break;
            case test_layout::inner_batched:
                setup_inner_batched_plan(to_add.plan_helper);
                break;
            case test_layout::nondefault_dim_ordering:
                setup_nondefault_dim_ordering_plan(to_add.plan_helper);
                break;
            case test_layout::tweaked_default_batched:
                setup_tweaked_default_batched_plan(to_add.plan_helper);
                break;
            default:
                throw std::runtime_error(
                    "unexpected value of data_layout in params_for_functional_tests");
                break;
            }
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
                    if(to_add.plan_helper.get_placement() == fft_placement_inplace
                       && io == fft_io::fft_io_out)
                    {
                        auto input_key       = key;
                        input_key.second     = fft_io::fft_io_in;
                        to_add.mem_type[key] = to_add.mem_type[input_key];
                    }
                    else
                    {
                        to_add.mem_type[key] = get_random_element_in(possible_mem_types);
                    }
                }
            }
            // skip params if they can't be tested for some reason
            if(!to_add.can_be_tested())
            {
                if(verbose)
                {
                    // Likely some bug/mistake in one of the above setup_* functions.
                    // That should be avoided as it can slow down test generation if
                    // the failure to generate testable parameters happens frequently
                    std::cout << to_add.to_string()
                              << " rejected as a functional test (cannot be tested)." << std::endl;
                }
                continue;
            }
            insert_into_unique_sorted_params(full_list, to_add);
        }
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
        // always add the manually-provided test, if matching target test's precision
        if(!manual_token.empty()
           && manual_token.find(hipfftw_trait<prec>::prec_label) != std::string::npos)
        {
            insert_into_unique_sorted_params(
                ret, hipfftw_functional_validation_params<prec>(manual_token));
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
INSTANTIATE_TEST_SUITE_P(hipfftw_test,
                         hipfftw_functional_validation_sp,
                         ::testing::ValuesIn(params_for_functional_tests<fft_precision_single>(
                             full_suite_size, hipfftw_token_for_functional_test)),
                         hipfftw_functional_validation_sp::TestName);
INSTANTIATE_TEST_SUITE_P(hipfftw_test,
                         hipfftw_functional_validation_dp,
                         ::testing::ValuesIn(params_for_functional_tests<fft_precision_double>(
                             full_suite_size, hipfftw_token_for_functional_test)),
                         hipfftw_functional_validation_dp::TestName);

// params_for_functional_tests may return empty vectors for low test probabilities.
// The following ensures such cases do not make gtest report an error due to uninstantiated
// hipfftw_functional_validation_{sp,dp}.
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(hipfftw_functional_validation_sp);
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(hipfftw_functional_validation_dp);
