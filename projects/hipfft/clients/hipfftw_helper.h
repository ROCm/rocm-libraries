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

#ifndef HIPFFTW_HELPER_H
#define HIPFFTW_HELPER_H

#include "../shared/environment.h"
#include "../shared/fft_params.h"
#include <algorithm>
#include <fftw3.h>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#ifdef WIN32
#include <windows.h>
// psapi.h requires windows.h to be included first
#include <psapi.h>
typedef HMODULE LIB_HANDLE_T;
#else
#include <dlfcn.h>
#include <link.h>
typedef void* LIB_HANDLE_T;
#endif

template <fft_precision prec>
struct hipfftw_trait;
template <>
struct hipfftw_trait<fft_precision_single>
{
    using plan_t    = fftwf_plan;
    using complex_t = fftwf_complex;
    using real_t    = float;
};
template <>
struct hipfftw_trait<fft_precision_double>
{
    using plan_t    = fftw_plan;
    using complex_t = fftw_complex;
    using real_t    = double;
};

template <fft_precision prec>
using hipfftw_real_t = typename hipfftw_trait<prec>::real_t;
template <fft_precision prec>
using hipfftw_complex_t = typename hipfftw_trait<prec>::complex_t;
template <fft_precision prec>
using hipfftw_plan_t = typename hipfftw_trait<prec>::plan_t;

// singleton class encapsulating the dynamically-loaded hipfftw library
class dynamically_loaded_hipfftw
{
private:
    LIB_HANDLE_T       lib_handle;
    std::ostringstream load_error_info;

    dynamically_loaded_hipfftw()
    {
#ifdef __HIP_PLATFORM_AMD__
        const std::string lib_basename = "hipfftw";
#else
        const std::string lib_basename = "cufftw";
#endif
#ifdef WIN32
        const std::string lib_fullame = lib_basename + ".dll";
        lib_handle                    = LoadLibraryA(lib_fullame.c_str());
#else
        const std::string lib_fullame  = "lib" + lib_basename + ".so";
        lib_handle                     = dlopen(lib_fullame.c_str(), RTLD_LAZY);
#endif
        load_error_info.clear();
        if(!lib_handle)
        {
            load_error_info << "failed to open library " << lib_fullame;
#ifdef WIN32
            load_error_info << ". System's error code = " << GetLastError();
#else
            load_error_info << ". System's error message = " << dlerror();
#endif
            // do not throw from here to ease exception handling
        }
    }
    /* disable copies and moves */
    dynamically_loaded_hipfftw(const dynamically_loaded_hipfftw&) = delete;
    dynamically_loaded_hipfftw(dynamically_loaded_hipfftw&&)      = delete;
    dynamically_loaded_hipfftw& operator=(const dynamically_loaded_hipfftw&) = delete;
    dynamically_loaded_hipfftw& operator=(dynamically_loaded_hipfftw&&) = delete;

    static const dynamically_loaded_hipfftw& get_instance()
    {
        static dynamically_loaded_hipfftw singleton_instance;
        return singleton_instance;
    }

public:
    static LIB_HANDLE_T get_lib()
    {
        return get_instance().lib_handle;
    }
    static std::string get_load_error_info()
    {
        return get_instance().load_error_info.str();
    }
    ~dynamically_loaded_hipfftw()
    {
        if(lib_handle)
        {
#ifdef WIN32
            (void)FreeLibrary(lib_handle);
#else
            (void)dlclose(lib_handle);
#endif
        }
        lib_handle = nullptr;
    }
};

// exception specific to issues when loading hipfftw and/or when fetching
// the address of the supposedly-available functions therefrom
struct hipfftw_undefined_function_ptr : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

// helper struct for retrieving a function's return type
template <class T>
struct func_ret;
template <typename R, class... Args>
struct func_ret<R(Args...)>
{
    using type = R;
};
template <class T>
using func_ret_t = typename func_ret<T>::type;

template <typename func_type, std::enable_if_t<std::is_function_v<func_type>, bool> = true>
struct dynamically_loaded_function_t
{
private:
    // address of the desired function, to be fetched from a dynamically loaded shared library
    func_type* func_ptr;
    // address of the reference function (from linked fftw3)
    func_type* const reference_func_ptr;
    // symbol of said function
    std::string func_symbol;

public:
    dynamically_loaded_function_t(const char* symbol, func_type* ref_func_address)
        : func_ptr(nullptr)
        , reference_func_ptr(ref_func_address)
        , func_symbol(symbol)
    {
    }

    // forwarding functional calls
    template <typename... Args>
    func_ret_t<func_type> operator()(Args... args) const
    {
        if(!may_be_used())
            throw hipfftw_undefined_function_ptr(dynamically_loaded_hipfftw::get_load_error_info());
        return func_ptr(args...);
    }
    template <bool call_reference, typename... Args>
    func_ret_t<func_type> call(Args... args) const
    {
        if constexpr(!call_reference)
        {
            return this->operator()(args...);
        }
        else
        {
            if(!reference_func_ptr)
                throw hipfftw_undefined_function_ptr(
                    "Ill-defined reference function pointer for symbol " + func_symbol);
            return reference_func_ptr(args...);
        }
        // unreachable
    }
    void load_implementation()
    {
        const auto hipfftw_lib = dynamically_loaded_hipfftw::get_lib();
        if(!hipfftw_lib)
        {
            // make func_ptr unambiguously unset to force the dedicated exception
            // to be thrown at forwarded functional call(s)
            func_ptr = nullptr;
            return;
        }
#ifdef WIN32
        func_ptr = reinterpret_cast<func_type*>(GetProcAddress(hipfftw_lib, func_symbol.c_str()));
#else
        func_ptr = reinterpret_cast<func_type*>(dlsym(hipfftw_lib, func_symbol.c_str()));
#endif
    }
    bool may_be_used() const
    {
        return func_ptr != nullptr;
    }
    std::string get_symbol() const
    {
        return func_symbol;
    }
};

template <typename T, typename... Args>
static void load_implementations(dynamically_loaded_function_t<T>& first, Args&... others)
{
    first.load_implementation();
    if constexpr(sizeof...(others) > 0)
        load_implementations(others...);
}

// define singleton structures encapsulating all the hipfftw function
// pointers (one specialization per supported precision)
template <fft_precision prec>
struct hipfftw_funcs;

#define HIPFFTW_STRINGIFY(x) #x
#define HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, func)                        \
    dynamically_loaded_function_t<decltype(prefix##func)> func                                   \
        = dynamically_loaded_function_t<decltype(prefix##func)>(HIPFFTW_STRINGIFY(prefix##func), \
                                                                &(prefix##func));

#define HIPFFTW_FUNCS_SPECIALIZATION(prefix, specialization)                           \
    template <>                                                                        \
    struct hipfftw_funcs<specialization>                                               \
    {                                                                                  \
    private:                                                                           \
        hipfftw_funcs()                                                                \
        {                                                                              \
            load_implementations(malloc,                                               \
                                 alloc_real,                                           \
                                 alloc_complex,                                        \
                                 free,                                                 \
                                 destroy_plan,                                         \
                                 cleanup,                                              \
                                 execute,                                              \
                                 plan_dft_1d,                                          \
                                 plan_dft_2d,                                          \
                                 plan_dft_3d,                                          \
                                 plan_dft,                                             \
                                 plan_dft_r2c_1d,                                      \
                                 plan_dft_r2c_2d,                                      \
                                 plan_dft_r2c_3d,                                      \
                                 plan_dft_r2c,                                         \
                                 plan_dft_c2r_1d,                                      \
                                 plan_dft_c2r_2d,                                      \
                                 plan_dft_c2r_3d,                                      \
                                 plan_dft_c2r,                                         \
                                 print_plan,                                           \
                                 set_timelimit,                                        \
                                 cost,                                                 \
                                 flops,                                                \
                                 execute_dft,                                          \
                                 execute_dft_r2c,                                      \
                                 execute_dft_c2r,                                      \
                                 plan_many_dft,                                        \
                                 plan_many_dft_r2c,                                    \
                                 plan_many_dft_c2r);                                   \
        }                                                                              \
        /* disable copies and moves */                                                 \
        hipfftw_funcs(const hipfftw_funcs&) = delete;                                  \
        hipfftw_funcs& operator=(const hipfftw_funcs&) = delete;                       \
        hipfftw_funcs(hipfftw_funcs&&)                 = delete;                       \
        hipfftw_funcs& operator=(hipfftw_funcs&&) = delete;                            \
                                                                                       \
    public:                                                                            \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, malloc)            \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, alloc_real)        \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, alloc_complex)     \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, free)              \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, destroy_plan)      \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, cleanup)           \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, execute)           \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_1d)       \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_2d)       \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_3d)       \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft)          \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_r2c_1d)   \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_r2c_2d)   \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_r2c_3d)   \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_r2c)      \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_c2r_1d)   \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_c2r_2d)   \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_c2r_3d)   \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_c2r)      \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, print_plan)        \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, set_timelimit)     \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, cost)              \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, flops)             \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, execute_dft)       \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, execute_dft_r2c)   \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, execute_dft_c2r)   \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_many_dft)     \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_many_dft_r2c) \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_many_dft_c2r) \
        static const hipfftw_funcs& get_instance()                                     \
        {                                                                              \
            static const hipfftw_funcs instance;                                       \
            return instance;                                                           \
        }                                                                              \
    }

HIPFFTW_FUNCS_SPECIALIZATION(fftwf_, fft_precision_single);
HIPFFTW_FUNCS_SPECIALIZATION(fftw_, fft_precision_double);

// structure enabling verbosity for hipfftw's exception handler and redirecting std::cerr
// to a runtime buffer throughout its lifetime (unless it was already enabled prior/externally)
struct hipfftw_exception_logger
{
    bool                  active;
    std::stringstream     buffer;
    std::streambuf* const original_cerr_rdbuf = nullptr;

    std::unique_ptr<EnvironmentSetTemp> hipfftw_temp_logger_env;

public:
    hipfftw_exception_logger()
        : active(false)
        , original_cerr_rdbuf(std::cerr.rdbuf())
    {
#ifdef __HIP_PLATFORM_AMD__
        const auto env_val = rocfft_getenv("HIPFFTW_LOG_EXCEPTIONS");
        // activate temporary redirection only if not already used otherwise
        // (e.g., in test user's environment )
        if(env_val.empty() || std::stoull(env_val) == 0)
        {
            hipfftw_temp_logger_env
                = std::make_unique<EnvironmentSetTemp>("HIPFFTW_LOG_EXCEPTIONS", "1");
            const auto temp_env_val = rocfft_getenv("HIPFFTW_LOG_EXCEPTIONS");
            active                  = !temp_env_val.empty() && std::stoull(temp_env_val) != 0;
        }
#endif
        if(active)
            std::cerr.rdbuf(buffer.rdbuf());
    }
    hipfftw_exception_logger(const hipfftw_exception_logger&) = delete;
    hipfftw_exception_logger(hipfftw_exception_logger&&)      = delete;
    hipfftw_exception_logger& operator=(const hipfftw_exception_logger&) = delete;
    hipfftw_exception_logger& operator=(hipfftw_exception_logger&&) = delete;
    ~hipfftw_exception_logger()
    {
        if(active)
        {
            // restore cerr to its original state
            std::cerr.rdbuf(original_cerr_rdbuf);
        }
    }
    bool is_active() const
    {
        return active;
    }
    std::string get_log() const
    {
        return buffer.str();
    }
};

// bit-flagging enum used for representing (combinations of) plan creation
// function(s) to consider
enum hipfftw_plan_creation_func : unsigned
{
    NONE        = 0x0, // not to be used (exceptfor validating values)
    PLAN_DFT_ND = 0x1 << 0,
    PLAN_DFT    = 0x1 << 1,
    PLAN_MANY   = 0x1 << 2,
    PLAN_GURU   = 0x1 << 3,
    PLAN_GURU64 = 0x1 << 4,
    ANY         = PLAN_DFT_ND | PLAN_DFT | PLAN_MANY | PLAN_GURU | PLAN_GURU64
};
static const std::vector<hipfftw_plan_creation_func> hipfftw_plan_creation_func_candidates
    = {hipfftw_plan_creation_func::PLAN_DFT_ND,
       hipfftw_plan_creation_func::PLAN_DFT,
       hipfftw_plan_creation_func::PLAN_MANY,
       hipfftw_plan_creation_func::PLAN_GURU,
       hipfftw_plan_creation_func::PLAN_GURU64};

static bool hipfftw_creation_options_are_well_defined(hipfftw_plan_creation_func creation_options)
{
    return creation_options == (creation_options & hipfftw_plan_creation_func::ANY);
}

static std::string hipfftw_creation_options_to_string(hipfftw_plan_creation_func creation_options,
                                                      fft_transform_type         dft_type,
                                                      int                        intended_rank)
{
    if(!hipfftw_creation_options_are_well_defined(creation_options))
        throw std::invalid_argument(
            "invalid creation_options for hipfftw_creation_options_to_string");
    if(creation_options == hipfftw_plan_creation_func::NONE)
        return "none";
    if(creation_options == hipfftw_plan_creation_func::ANY)
        return "any";
    if(std::find(hipfftw_plan_creation_func_candidates.begin(),
                 hipfftw_plan_creation_func_candidates.end(),
                 creation_options)
       == hipfftw_plan_creation_func_candidates.end())
    {
        // 2 or more qualifying candidates flagged in creation_options
        std::string ret;
        for(auto candidate : hipfftw_plan_creation_func_candidates)
        {
            if(creation_options & candidate)
            {
                if(!ret.empty())
                    ret += "_or_";
                ret += hipfftw_creation_options_to_string(candidate, dft_type, intended_rank);
            }
        }
        return ret;
    }
    // creation_options is one unique qualifying candidate
    std::ostringstream ret;
    const std::string  real_or_empty_qualifier
        = is_real(dft_type) ? (is_fwd(dft_type) ? "_r2c" : "_c2r") : "";
    switch(creation_options)
    {
    case hipfftw_plan_creation_func::PLAN_DFT_ND:
        ret << "plan_dft" << real_or_empty_qualifier << "_" << (intended_rank < 0 ? "negative" : "")
            << std::abs(intended_rank) << "d";
        break;
    case hipfftw_plan_creation_func::PLAN_DFT:
        ret << "plan_dft" << real_or_empty_qualifier;
        break;
    case hipfftw_plan_creation_func::PLAN_MANY:
        ret << "plan_many_dft" << real_or_empty_qualifier;
        break;
    case hipfftw_plan_creation_func::PLAN_GURU:
        ret << "plan_guru_dft" << real_or_empty_qualifier;
        break;
    case hipfftw_plan_creation_func::PLAN_GURU64:
        ret << "plan_guru64_dft" << real_or_empty_qualifier;
        break;
    default:
        throw std::runtime_error("hipfftw_creation_options_to_string: internal error encountered "
                                 "(unexpected value for creation_options)");
        break;
    }
    return ret.str();
}

enum class hipfftw_plan_execution_func
{
    EXECUTE,
    EXECUTE_DFT,
    EXECUTE_DFT_R2C,
    EXECUTE_DFT_C2R,
    DEFAULT
};

static std::string hipfftw_execution_option_to_string(hipfftw_plan_execution_func execution_option)
{
    switch(execution_option)
    {
    case hipfftw_plan_execution_func::EXECUTE:
        return "execute";
        break;
    case hipfftw_plan_execution_func::EXECUTE_DFT:
        return "execute_dft";
        break;
    case hipfftw_plan_execution_func::EXECUTE_DFT_R2C:
        return "execute_dft_r2c";
        break;
    case hipfftw_plan_execution_func::EXECUTE_DFT_C2R:
        return "execute_dft_c2r";
        break;
    case hipfftw_plan_execution_func::DEFAULT:
        return "default_execution";
        break;
    default:
        throw std::runtime_error("hipfftw_execution_option_to_string: internal error encountered "
                                 "(unexpected value for execution_option)");
    }
}

template <
    fft_precision prec,
    std::enable_if_t<prec == fft_precision_single || prec == fft_precision_double, bool> = true>
struct hipfftw_plan_bundle_t
{
private:
    const decltype(hipfftw_funcs<prec>::destroy_plan)& plan_destructor;

public:
    hipfftw_plan_t<prec>       plan;
    std::pair<void*, void*>    creation_io; // not owned
    hipfftw_plan_creation_func creation_func;
    std::string                plan_token; // <-- plan details, except for creation io data pointers
    hipfftw_plan_bundle_t(decltype(plan_destructor) plan_destructor_func)
        : plan_destructor(plan_destructor_func)
        , plan(nullptr)
        , creation_io({nullptr, nullptr})
        , creation_func(hipfftw_plan_creation_func::NONE)
        , plan_token("")
    {
    }
    ~hipfftw_plan_bundle_t()
    {
        // make sure the plan destructor may be used to avoid
        // throwing from the hipfftw_plan_bundle_t destructor
        if(plan_destructor.may_be_used())
        {
            // should be stable even if plan == nullptr;
            plan_destructor(plan);
        }
        else if(plan)
        {
            std::cerr << "WARNING: A " << (prec == fft_precision_single ? "single" : "double")
                      << "-precision plan was seemingly created but its destructor cannot be used "
                      << std::endl;
        }
    }
    // disable copies and moves
    hipfftw_plan_bundle_t(const hipfftw_plan_bundle_t&) = delete;
    hipfftw_plan_bundle_t& operator=(const hipfftw_plan_bundle_t&) = delete;
    hipfftw_plan_bundle_t(hipfftw_plan_bundle_t&&)                 = delete;
    hipfftw_plan_bundle_t& operator=(hipfftw_plan_bundle_t&&) = delete;
};

static bool rank_is_valid_for_hipfftw(int r)
{
    return r > 0;
}
template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
static bool vector_has_valid_values_as(const std::vector<ptrdiff_t>& vals,
                                       int                           intended_rank,
                                       T min_valid_val = std::numeric_limits<T>::lowest(),
                                       T max_valid_val = std::numeric_limits<T>::max())
{
    if(!rank_is_valid_for_hipfftw(intended_rank))
        return false; // impossible to validate values for an invalid intended_rank
    // check that vals are all in [min_valid_val, max_valid_val]
    return vals.size() == intended_rank
           && std::all_of(vals.begin(), vals.end(), [&](const ptrdiff_t& val) {
                  return val >= min_valid_val && val <= max_valid_val;
              });
}
static bool sign_is_valid_for_hipfftw(int s, const fft_transform_type& dft_kind)
{
    if(is_real(dft_kind))
        return true; // sign is irrelevant for real transforms
    return s == (is_fwd(dft_kind) ? FFTW_FORWARD : FFTW_BACKWARD);
}
static constexpr unsigned hipfftw_valid_flags_mask
    = FFTW_WISDOM_ONLY | FFTW_MEASURE | FFTW_DESTROY_INPUT | FFTW_UNALIGNED | FFTW_CONSERVE_MEMORY
      | FFTW_EXHAUSTIVE | FFTW_PRESERVE_INPUT | FFTW_PATIENT | FFTW_ESTIMATE;
static bool flags_are_valid_for_hipfftw(unsigned f)
{
    return (f & hipfftw_valid_flags_mask) == f;
}

static std::vector<ptrdiff_t> default_strides(fft_transform_type            dft_type,
                                              fft_result_placement          placement,
                                              fft_io                        io,
                                              const std::vector<ptrdiff_t>& lengths)
{
    std::vector<ptrdiff_t> ret(lengths.size());
    ptrdiff_t              def_stride = 1;
    for(auto dim_idx = lengths.size(); dim_idx-- > 0;)
    {
        ret[dim_idx] = def_stride;
        if(dim_idx == lengths.size() - 1 && is_real(dft_type))
        {
            if((io == fft_io_out) == is_fwd(dft_type))
                def_stride *= (lengths[dim_idx] / 2 + 1);

            else
            {
                if(placement == fft_placement_inplace)
                    def_stride *= 2 * (lengths[dim_idx] / 2 + 1);
                else
                    def_stride *= lengths[dim_idx];
            }
        }
        else
            def_stride *= lengths[dim_idx];
    }
    return ret;
}

static std::vector<ptrdiff_t> default_distances(fft_transform_type            dft_type,
                                                fft_result_placement          placement,
                                                fft_io                        io,
                                                const std::vector<ptrdiff_t>& lengths,
                                                const std::vector<ptrdiff_t>& batches)
{
    std::vector<ptrdiff_t> ret(batches.size());
    if(batches.empty() || lengths.empty())
        return ret;
    auto temp_lengths = lengths;
    temp_lengths.insert(temp_lengths.begin(), 1);
    auto def_dist = default_strides(dft_type, placement, io, temp_lengths).front();
    for(auto batch_dim = 0; batch_dim < batches.size(); batch_dim++)
    {
        ret[batch_dim] = def_dist;
        def_dist *= batches[batch_dim];
    }
    return ret;
}

struct hipfftw_helper_num_elements_calc_exception : public std::runtime_error
{
    using std::runtime_error::runtime_error;
};

template <
    fft_precision prec,
    std::enable_if_t<prec == fft_precision_single || prec == fft_precision_double, bool> = true>
struct hipfftw_helper
{
private:
    // plan_bundle stores information about the latest plan possibly created by this
    // object. A shard_ptr is used to make hipfftw_helper safe w.r.t. shallow
    // copies (as required by gtest for parameterized tests).
    // This member is also made mutable so we can release/create it even from a
    // const-qualified objects (e.g., to release owned resources upon test completion,
    // or to re-create the plan at execution if needed or found necessary)
    mutable std::shared_ptr<hipfftw_plan_bundle_t<prec>> plan_bundle;

    fft_transform_type     dft_kind;
    int                    rank       = 0;
    int                    batch_rank = 0;
    std::vector<ptrdiff_t> lengths;
    std::vector<ptrdiff_t> istrides;
    std::vector<ptrdiff_t> ostrides;
    std::vector<ptrdiff_t> batches;
    std::vector<ptrdiff_t> idist;
    std::vector<ptrdiff_t> odist;
    fft_result_placement   plan_placement;
    int                    sign  = 0;
    unsigned               flags = std::numeric_limits<unsigned>::max();

    template <typename T>
    void reset_member_value(T& member, const T& new_value)
    {
        if(new_value != member)
        {
            member = new_value;
            plan_bundle.reset();
        }
    }

    hipfftw_plan_creation_func get_creation_func(hipfftw_plan_creation_func creation_options) const
    {
        if(!hipfftw_creation_options_are_well_defined(creation_options))
            throw std::invalid_argument("invalid creation_options for get_creation_func");
        if(!can_use_creation_options(creation_options))
        {
            // e.g., rank < 0 with creation_options == hipfftw_plan_creation_func::PLAN_DFT_ND
            throw std::invalid_argument(
                "The plan creation options "
                + hipfftw_creation_options_to_string(creation_options, dft_kind, rank)
                + " cannot be used with this object");
        }
        std::vector<hipfftw_plan_creation_func> valid_candidates;
        for(auto candidate : hipfftw_plan_creation_func_candidates)
        {
            if(!(creation_options & candidate))
                continue; // candidate is not in given creation_options
            if(can_use_creation_options(candidate))
            {
                // If creation_options != candidate for all candidates, creation_optionsactually
                // combines 2 or more candidates --> only the candidates actually supporting plan
                // creation will be considered "valid". If there exists one (usable) candidate s.t.
                // creation_options == candidate however, this choice is considered "enforced"
                // (e.g. for function-specific argument validation testing purposes)
                if(creation_options == candidate || can_create_plan_with(candidate))
                    valid_candidates.push_back(candidate);
            }
        }
        if(valid_candidates.empty())
            return hipfftw_plan_creation_func::NONE;
        // "randomly" (yet reproducibly) choose
        return valid_candidates[std::hash<std::string>()(token()) % valid_candidates.size()];
    }

    template <bool make_reference_plan = false>
    hipfftw_plan_t<prec>
        make_plan(void* in, void* out, hipfftw_plan_creation_func chosen_creation) const
    {
        if(std::find(hipfftw_plan_creation_func_candidates.begin(),
                     hipfftw_plan_creation_func_candidates.end(),
                     chosen_creation)
           == hipfftw_plan_creation_func_candidates.end())
        {
            throw std::invalid_argument("Invalid chosen_creation for hipfftw_helper::make_plan");
        }
        const auto effective_placement
            = in == out ? fft_placement_inplace : fft_placement_notinplace;
        if(effective_placement != plan_placement)
            throw std::invalid_argument(
                "Invalid I/O argument for plan creation w.r.t. expected plan placement.");

        // fetch/infer plan creation function arguments
        const auto& hipfftw_impl = hipfftw_funcs<prec>::get_instance();
        const auto  int_len      = get_length_as<int>();
        const int*  int_len_ptr  = int_len.empty() ? nullptr : int_len.data();

        switch(chosen_creation)
        {
        case hipfftw_plan_creation_func::PLAN_DFT_ND:
        {
            if(!can_use_creation_options(hipfftw_plan_creation_func::PLAN_DFT_ND))
                throw std::runtime_error("hipfftw_plan_creation_func::PLAN_DFT_ND cannot be used.");
            if(rank == 1)
            {
                if(dft_kind == fft_transform_type_real_forward)
                {
                    return hipfftw_impl.plan_dft_r2c_1d.template call<make_reference_plan>(
                        int_len_ptr[0],
                        static_cast<hipfftw_real_t<prec>*>(in),
                        static_cast<hipfftw_complex_t<prec>*>(out),
                        flags);
                }
                else if(dft_kind == fft_transform_type_real_inverse)
                {
                    return hipfftw_impl.plan_dft_c2r_1d.template call<make_reference_plan>(
                        int_len_ptr[0],
                        static_cast<hipfftw_complex_t<prec>*>(in),
                        static_cast<hipfftw_real_t<prec>*>(out),
                        flags);
                }
                else
                {

                    return hipfftw_impl.plan_dft_1d.template call<make_reference_plan>(
                        int_len_ptr[0],
                        static_cast<hipfftw_complex_t<prec>*>(in),
                        static_cast<hipfftw_complex_t<prec>*>(out),
                        sign,
                        flags);
                }
            }
            else if(rank == 2)
            {
                if(dft_kind == fft_transform_type_real_forward)
                {
                    return hipfftw_impl.plan_dft_r2c_2d.template call<make_reference_plan>(
                        int_len_ptr[0],
                        int_len_ptr[1],
                        static_cast<hipfftw_real_t<prec>*>(in),
                        static_cast<hipfftw_complex_t<prec>*>(out),
                        flags);
                }
                else if(dft_kind == fft_transform_type_real_inverse)
                {

                    return hipfftw_impl.plan_dft_c2r_2d.template call<make_reference_plan>(
                        int_len_ptr[0],
                        int_len_ptr[1],
                        static_cast<hipfftw_complex_t<prec>*>(in),
                        static_cast<hipfftw_real_t<prec>*>(out),
                        flags);
                }
                else
                {
                    return hipfftw_impl.plan_dft_2d.template call<make_reference_plan>(
                        int_len_ptr[0],
                        int_len_ptr[1],
                        static_cast<hipfftw_complex_t<prec>*>(in),
                        static_cast<hipfftw_complex_t<prec>*>(out),
                        sign,
                        flags);
                }
            }
            else
            {
                if(dft_kind == fft_transform_type_real_forward)
                {
                    return hipfftw_impl.plan_dft_r2c_3d.template call<make_reference_plan>(
                        int_len_ptr[0],
                        int_len_ptr[1],
                        int_len_ptr[2],
                        static_cast<hipfftw_real_t<prec>*>(in),
                        static_cast<hipfftw_complex_t<prec>*>(out),
                        flags);
                }
                else if(dft_kind == fft_transform_type_real_inverse)
                {
                    return hipfftw_impl.plan_dft_c2r_3d.template call<make_reference_plan>(
                        int_len_ptr[0],
                        int_len_ptr[1],
                        int_len_ptr[2],
                        static_cast<hipfftw_complex_t<prec>*>(in),
                        static_cast<hipfftw_real_t<prec>*>(out),
                        flags);
                }
                else
                {
                    return hipfftw_impl.plan_dft_3d.template call<make_reference_plan>(
                        int_len_ptr[0],
                        int_len_ptr[1],
                        int_len_ptr[2],
                        static_cast<hipfftw_complex_t<prec>*>(in),
                        static_cast<hipfftw_complex_t<prec>*>(out),
                        sign,
                        flags);
                }
            }
        }
        break;
        case hipfftw_plan_creation_func::PLAN_DFT:
        {
            if(!can_use_creation_options(hipfftw_plan_creation_func::PLAN_DFT))
                throw std::runtime_error("hipfftw_plan_creation_func::PLAN_DFT cannot be used.");

            if(dft_kind == fft_transform_type_real_forward)
            {
                return hipfftw_impl.plan_dft_r2c.template call<make_reference_plan>(
                    rank,
                    int_len_ptr,
                    static_cast<hipfftw_real_t<prec>*>(in),
                    static_cast<hipfftw_complex_t<prec>*>(out),
                    flags);
            }
            else if(dft_kind == fft_transform_type_real_inverse)
            {
                return hipfftw_impl.plan_dft_c2r.template call<make_reference_plan>(
                    rank,
                    int_len_ptr,
                    static_cast<hipfftw_complex_t<prec>*>(in),
                    static_cast<hipfftw_real_t<prec>*>(out),
                    flags);
            }
            else
            {
                return hipfftw_impl.plan_dft.template call<make_reference_plan>(
                    rank,
                    int_len_ptr,
                    static_cast<hipfftw_complex_t<prec>*>(in),
                    static_cast<hipfftw_complex_t<prec>*>(out),
                    sign,
                    flags);
            }
        }
        break;
        case hipfftw_plan_creation_func::PLAN_MANY:
        {
            if(!can_use_creation_options(hipfftw_plan_creation_func::PLAN_MANY))
                throw std::runtime_error("hipfftw_plan_creation_func::PLAN_MANY cannot be used.");
            const int* inembed = get_nembed_from_strides(fft_io::fft_io_in);
            const int* onembed = get_nembed_from_strides(fft_io::fft_io_out);
            // set elementary strides to 0 if corresponding strides are empty (invalid case anyways)
            const int istride = istrides.empty() ? 0 : istrides.back();
            const int ostride = ostrides.empty() ? 0 : ostrides.back();

            if(dft_kind == fft_transform_type_real_forward)
            {
                return hipfftw_impl.plan_many_dft_r2c.template call<make_reference_plan>(
                    rank,
                    int_len_ptr,
                    static_cast<int>(batches[0]),
                    static_cast<hipfftw_real_t<prec>*>(in),
                    inembed,
                    istride,
                    idist[0],
                    static_cast<hipfftw_complex_t<prec>*>(out),
                    onembed,
                    ostride,
                    odist[0],
                    flags);
            }
            else if(dft_kind == fft_transform_type_real_inverse)
            {
                return hipfftw_impl.plan_many_dft_c2r.template call<make_reference_plan>(
                    rank,
                    int_len_ptr,
                    static_cast<int>(batches[0]),
                    static_cast<hipfftw_complex_t<prec>*>(in),
                    inembed,
                    istride,
                    idist[0],
                    static_cast<hipfftw_real_t<prec>*>(out),
                    onembed,
                    ostride,
                    odist[0],
                    flags);
            }
            else
            {
                return hipfftw_impl.plan_many_dft.template call<make_reference_plan>(
                    rank,
                    int_len_ptr,
                    static_cast<int>(batches[0]),
                    static_cast<hipfftw_complex_t<prec>*>(in),
                    inembed,
                    istride,
                    idist[0],
                    static_cast<hipfftw_complex_t<prec>*>(out),
                    onembed,
                    ostride,
                    odist[0],
                    sign,
                    flags);
            }
        }
        case hipfftw_plan_creation_func::PLAN_GURU:
            [[fallthrough]];
        case hipfftw_plan_creation_func::PLAN_GURU64:
            throw std::runtime_error("Enforced plan creation is not implemented yet");
            break;
        default:
            throw std::runtime_error("Unknown kind of plan creation");
            break;
        }
        // unreachable
    }

    // converts vec to an std::vector<T> if it can be done so safely
    // (an empty vector is returned otherwise)
    template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
    static std::vector<T> convert_vector_to(const std::vector<ptrdiff_t> vec)
    {
        if constexpr(std::is_same_v<T, typename decltype(vec)::value_type>)
            return vec;
        std::vector<T> ret;
        if(std::any_of(vec.begin(), vec.end(), [](const typename decltype(vec)::value_type& val) {
               return val < std::numeric_limits<T>::lowest() || val > std::numeric_limits<T>::max();
           }))
        {
            // not a safe conversion, return empty lengths
            return ret;
        }
        ret.assign(vec.begin(), vec.end());
        return ret;
    }

    struct strides_are_incompatible_with_nembed : std::runtime_error
    {
        strides_are_incompatible_with_nembed()
            : std::runtime_error::runtime_error(
                "stride value(s) are not compatible with nembed convention(s)")
        {
        }
    };
    const int* get_nembed_from_strides(fft_io io) const
    {
        if(io != fft_io::fft_io_in && io != fft_io::fft_io_out)
            throw std::invalid_argument("Invalid io");
        const auto&            strides = io == fft_io::fft_io_in ? istrides : ostrides;
        std::hash<std::string> hasher;
        // use nullptr as nembed for default strides half of the time,
        // to guarantee testing thereof
        if(strides == default_strides(dft_kind, plan_placement, io, lengths)
           && (hasher(token()) & 1))
            return nullptr;
        static std::map<fft_io, std::vector<int>> map_of_nembed_vecs;

        auto& nembed_vec = map_of_nembed_vecs[io];
        nembed_vec.resize(lengths.size());
        for(auto nembed_dim = nembed_vec.size(); nembed_dim-- > 0;)
        {
            if(nembed_dim > 0)
            {
                if(strides[nembed_dim] == 0 || strides[nembed_dim - 1] % strides[nembed_dim] != 0)
                    throw strides_are_incompatible_with_nembed();
                nembed_vec[nembed_dim] = strides[nembed_dim - 1] / strides[nembed_dim];
            }
            else
            {
                // actually irrelevant/unused value...
                // set it to the minimum value documented to be valid by FFTW3
                // to avoid triggering (reference) plan creation failures
                if(is_real(dft_kind) && lengths.size() == 1)
                {
                    const int cmplx_stride = lengths[nembed_dim] / 2 + 1;
                    if((io == fft_io_in) == is_bwd(dft_kind))
                        nembed_vec[nembed_dim] = cmplx_stride;
                    else
                    {
                        if(plan_placement == fft_placement_inplace)
                            nembed_vec[nembed_dim] = 2 * cmplx_stride;
                        else
                            nembed_vec[nembed_dim] = lengths[nembed_dim];
                    }
                }
                else
                    nembed_vec[nembed_dim] = lengths[nembed_dim];
            }
        }
        return nembed_vec.data();
    }

    // (private) validity checks
    bool has_valid_rank(hipfftw_plan_creation_func creation_options
                        = hipfftw_plan_creation_func::ANY) const
    {
        if(!hipfftw_creation_options_are_well_defined(creation_options))
            throw std::invalid_argument("ill-defined creation_options used in has_valid_rank");
        // check if valid for any of the possible plan creation functions
        bool ret = false;
        for(auto creation_func : hipfftw_plan_creation_func_candidates)
        {
            if(ret)
                break;
            if(!(creation_options & creation_func))
                continue;
            switch(creation_func)
            {
            case hipfftw_plan_creation_func::PLAN_DFT_ND:
                ret = rank == 1 || rank == 2 || rank == 3;
                break;
            case hipfftw_plan_creation_func::PLAN_DFT:
                [[fallthrough]];
            case hipfftw_plan_creation_func::PLAN_MANY:
                [[fallthrough]];
            case hipfftw_plan_creation_func::PLAN_GURU64:
                [[fallthrough]];
            case hipfftw_plan_creation_func::PLAN_GURU:
                ret = rank_is_valid_for_hipfftw(rank);
                break;
            default:
                throw std::runtime_error("hipfftw_helper: internal error encountered (unexpected "
                                         "value for creation_func)");
            }
        }
        return ret;
    }
    bool has_valid_lengths() const
    {
        constexpr ptrdiff_t min_len = 1;
        return vector_has_valid_values_as<ptrdiff_t>(lengths, rank, min_len);
    }
    bool has_valid_sign() const
    {
        return sign_is_valid_for_hipfftw(sign, dft_kind);
    }
    bool has_valid_flags() const
    {
        return flags_are_valid_for_hipfftw(flags);
    }
    bool has_valid_batch_rank(hipfftw_plan_creation_func creation_options
                              = hipfftw_plan_creation_func::ANY) const
    {
        if(!hipfftw_creation_options_are_well_defined(creation_options))
            throw std::invalid_argument(
                "ill-defined creation_options used in has_valid_batch_rank");
        if(!rank_is_valid_for_hipfftw(batch_rank))
            return false;
        // check if valid for any of the possible plan creation functions
        bool ret = false;
        for(auto creation_func : hipfftw_plan_creation_func_candidates)
        {
            if(ret)
                break;
            if(!(creation_options & creation_func))
                continue;
            switch(creation_func)
            {
            case hipfftw_plan_creation_func::PLAN_DFT_ND:
                [[fallthrough]];
            case hipfftw_plan_creation_func::PLAN_DFT:
                [[fallthrough]];
            case hipfftw_plan_creation_func::PLAN_MANY:
                ret = batch_rank == 1;
                break;
            case hipfftw_plan_creation_func::PLAN_GURU64:
                [[fallthrough]];
            case hipfftw_plan_creation_func::PLAN_GURU:
                ret = rank_is_valid_for_hipfftw(batch_rank);
                break;
            default:
                throw std::runtime_error("hipfftw_helper: internal error encountered (unexpected "
                                         "value for creation_func)");
            }
        }
        return ret;
    }
    bool has_valid_batches() const
    {
        constexpr ptrdiff_t min_batch = 1;
        return vector_has_valid_values_as<ptrdiff_t>(batches, batch_rank, min_batch);
    }
    bool has_valid_strides(fft_io io, hipfftw_plan_creation_func creation_options) const
    {
        if(io != fft_io::fft_io_in && io != fft_io::fft_io_out)
            throw std::invalid_argument("invalid io");

        if(!hipfftw_creation_options_are_well_defined(creation_options))
            throw std::invalid_argument("ill-defined creation_options used in has_valid_strides");
        const auto& strides = io == fft_io::fft_io_in ? istrides : ostrides;
        bool        ret     = vector_has_valid_values_as<ptrdiff_t>(strides, rank);
        // 0 stride values are invalid for any nontrivial length
        for(auto dim = 0; ret && dim < rank; dim++)
        {
            if(lengths[dim] != 1 && strides[dim] == 0)
                ret = false;
        }
        if(ret && plan_placement == fft_placement_inplace)
        {
            // for in-place operations, input and output strides cannot
            // be considered independent of each other
            if(!vector_has_valid_values_as<ptrdiff_t>(istrides, rank)
               || !vector_has_valid_values_as<ptrdiff_t>(ostrides, rank))
            {
                ret = false;
            }
            const auto ifact = dft_kind == fft_transform_type_real_forward ? 1 : 2;
            const auto ofact = dft_kind == fft_transform_type_real_inverse ? 1 : 2;
            for(size_t dim = 0; ret && dim < rank - 1; dim++)
                ret = ifact * istrides[dim] == ofact * ostrides[dim];
        }
        if(!ret)
            return ret;
        // check if valid for any of the possible plan creation functions
        ret = false;
        for(auto creation_func : hipfftw_plan_creation_func_candidates)
        {
            if(ret)
                break;
            if(!(creation_options & creation_func))
                continue;
            switch(creation_func)
            {
            case hipfftw_plan_creation_func::PLAN_DFT_ND:
                [[fallthrough]];
            case hipfftw_plan_creation_func::PLAN_DFT:
                ret = strides == default_strides(dft_kind, plan_placement, io, lengths);
                break;
            case hipfftw_plan_creation_func::PLAN_MANY:
            {
                const int* nembed = nullptr;
                try
                {
                    nembed = get_nembed_from_strides(io);
                    if(!nembed)
                        ret = true; // nullptr nembed implictly means "default values", hence valid
                    else
                    {
                        for(size_t dim = 0; dim < lengths.size(); dim++)
                        {
                            auto min_nembed = lengths[dim];
                            if(is_real(dft_kind) && dim == lengths.size() - 1)
                            {
                                const auto min_cmplx_nembed = min_nembed / 2 + 1;

                                if(is_fwd(dft_kind) == (io == fft_io::fft_io_out))
                                    min_nembed = min_cmplx_nembed;
                                else if(plan_placement == fft_placement_inplace)
                                    min_nembed = 2 * min_cmplx_nembed;
                            }
                            ret = nembed[dim] >= min_nembed;
                        }
                    }
                }
                catch(...)
                {
                    ret = false;
                }
                break;
            }
            case hipfftw_plan_creation_func::PLAN_GURU64:
                [[fallthrough]];
            case hipfftw_plan_creation_func::PLAN_GURU:
                ret = false; // to be defined when guru apis are enabled
                break;
            default:
                throw std::runtime_error("hipfftw_helper: internal error encountered (unexpected "
                                         "value for creation_func)");
            }
        }
        return ret;
    }
    bool has_valid_distances(fft_io io) const
    {
        if(io != fft_io::fft_io_in && io != fft_io::fft_io_out)
            throw std::invalid_argument("invalid io");

        const auto& distances = io == fft_io::fft_io_in ? idist : odist;
        bool        ret       = vector_has_valid_values_as<ptrdiff_t>(distances, batch_rank);
        // 0 distance values are invalid for any nontrivial batch
        for(auto batch_dim = 0; ret && batch_dim < batch_rank; batch_dim++)
        {
            if(batches[batch_dim] != 1 && distances[batch_dim] == 0)
                ret = false;
        }
        if(ret && plan_placement == fft_placement_inplace)
        {
            // for in-place operations, input and output distances cannot
            // be considered independent of each other
            if(!vector_has_valid_values_as<ptrdiff_t>(idist, batch_rank)
               || !vector_has_valid_values_as<ptrdiff_t>(odist, batch_rank))
            {
                ret = false;
            }
            const auto ifact = dft_kind == fft_transform_type_real_forward ? 1 : 2;
            const auto ofact = dft_kind == fft_transform_type_real_inverse ? 1 : 2;
            for(size_t batch_dim = 0; ret && batch_dim < batch_rank; batch_dim++)
                ret = ifact * idist[batch_dim] == ofact * odist[batch_dim];
        }
        return ret;
    }

public:
    hipfftw_helper()                       = default;
    ~hipfftw_helper()                      = default;
    hipfftw_helper(hipfftw_helper&& other) = default;
    hipfftw_helper& operator=(hipfftw_helper&& other) = default;
    hipfftw_helper(const hipfftw_helper& other)       = default;
    hipfftw_helper& operator=(const hipfftw_helper& rhs) = default;

    void set_creation_args(fft_transform_type            dft_kind_to_set,
                           int                           rank_to_set,
                           const std::vector<ptrdiff_t>& lengths_to_set,
                           fft_result_placement          placement_to_set,
                           int                           sign_to_set,
                           unsigned                      flags_to_set)
    {
        set_creation_args(
            dft_kind_to_set,
            rank_to_set,
            lengths_to_set,
            placement_to_set,
            sign_to_set,
            flags_to_set,
            /* default strides + unit batch + zero distances */
            default_strides(dft_kind_to_set, placement_to_set, fft_io::fft_io_in, lengths_to_set),
            default_strides(dft_kind_to_set, placement_to_set, fft_io::fft_io_out, lengths_to_set),
            1,
            std::vector<ptrdiff_t>(1, 1),
            std::vector<ptrdiff_t>(1, 0),
            std::vector<ptrdiff_t>(1, 0));
    }

    void set_creation_args(fft_transform_type            dft_kind_to_set,
                           int                           rank_to_set,
                           const std::vector<ptrdiff_t>& lengths_to_set,
                           fft_result_placement          placement_to_set,
                           int                           sign_to_set,
                           unsigned                      flags_to_set,
                           const std::vector<ptrdiff_t>& istrides_to_set,
                           const std::vector<ptrdiff_t>& ostrides_to_set,
                           int                           batch_rank_to_set,
                           const std::vector<ptrdiff_t>& batches_to_set,
                           const std::vector<ptrdiff_t>& idist_to_set,
                           const std::vector<ptrdiff_t>& odist_to_set)
    {
        if(lengths_to_set.size() != istrides_to_set.size()
           || lengths_to_set.size() != ostrides_to_set.size())
            throw std::invalid_argument("Lengths and i/o strides must have the same size.");
        if(rank_is_valid_for_hipfftw(rank_to_set) && !lengths_to_set.empty()
           && rank_to_set != lengths_to_set.size())
            throw std::invalid_argument(
                "Inconsistent size for (non-empty) lengths_to_set w.r.t. (valid) rank_to_set.");
        if(batches_to_set.size() != idist_to_set.size()
           || batches_to_set.size() != odist_to_set.size())
            throw std::invalid_argument("Batches and i/o distances must have the same size.");
        if(rank_is_valid_for_hipfftw(batch_rank_to_set) && !batches_to_set.empty()
           && batch_rank_to_set != batches_to_set.size())
            throw std::invalid_argument("Inconsistent size for (non-empty) batches_to_set w.r.t. "
                                        "(valid) batch_rank_to_set.");

        reset_member_value(dft_kind, dft_kind_to_set);
        reset_member_value(rank, rank_to_set);
        reset_member_value(lengths, lengths_to_set);
        reset_member_value(plan_placement, placement_to_set);
        reset_member_value(sign, sign_to_set);
        reset_member_value(flags, flags_to_set);
        reset_member_value(istrides, istrides_to_set);
        reset_member_value(ostrides, ostrides_to_set);
        reset_member_value(batch_rank, batch_rank_to_set);
        reset_member_value(batches, batches_to_set);
        reset_member_value(idist, idist_to_set);
        reset_member_value(odist, odist_to_set);
    }

    // getters
    fft_transform_type get_dft_kind() const
    {
        return dft_kind;
    }
    int get_rank() const
    {
        return rank;
    }
    // returns the lengths as an std::vector<T> if they may all be safely converted to T
    // (the returned vector is empty otherwise)
    template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
    std::vector<T> get_length_as() const
    {
        return convert_vector_to<T>(lengths);
    }
    fft_result_placement get_placement() const
    {
        return plan_placement;
    }
    int get_sign() const
    {
        return sign;
    }
    unsigned get_flags() const
    {
        return flags;
    }
    // returns the strides as an std::vector<T> if they may all be safely converted to T
    // (the returned vector is empty otherwise)
    template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
    std::vector<T> get_strides_as(fft_io io) const
    {
        if(io != fft_io::fft_io_in && io != fft_io::fft_io_out)
            throw std::invalid_argument("invalid io");
        const std::vector<ptrdiff_t>& strides = io == fft_io::fft_io_in ? istrides : ostrides;
        return convert_vector_to<T>(strides);
    }
    int get_batch_rank() const
    {
        return batch_rank;
    }
    // returns the batches as an std::vector<T> if they may all be safely converted to T
    // (the returned vector is empty otherwise)
    template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
    std::vector<T> get_batches_as() const
    {
        return convert_vector_to<T>(batches);
    }
    // returns the distances as an std::vector<T> if they may all be safely converted to T
    // (the returned vector is empty otherwise)
    template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
    std::vector<T> get_distances_as(fft_io io) const
    {
        if(io != fft_io::fft_io_in && io != fft_io::fft_io_out)
            throw std::invalid_argument("invalid io");
        const std::vector<ptrdiff_t>& dist = io == fft_io::fft_io_in ? idist : odist;
        return convert_vector_to<T>(dist);
    }

    template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
    T get_dist_as(fft_io io) const
    {
        if(io != fft_io::fft_io_in && io != fft_io::fft_io_out)
            throw std::invalid_argument("invalid io");
        auto const tmp = get_distances_as<T>(io);
        if(tmp.empty())
            throw std::runtime_error("distance(s) cannot be safely converted to the desired type");
        if(tmp.size() != 1)
            throw std::runtime_error(
                "a single distance value cannot be queried for multi-batched cases");
        return tmp[0];
    }
    template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
    T get_nbatch_as() const
    {
        auto const tmp = get_batches_as<T>();
        if(tmp.size() != 1)
            throw std::runtime_error(
                "a single batch size cannot be queried for multi-batched cases");
        return tmp[0];
    }

    std::shared_ptr<hipfftw_plan_bundle_t<prec>> get_plan_bundle() const
    {
        return plan_bundle;
    }
    // checks if the current parameters can be used with (any of) the given option(s) of
    // plan creation, given the corresponding plan creation's signature (NOT whether they're
    // valid or not). For instance, one cannot possibly communicate rank > 3 with
    // hipfftw_plan_creation_func::PLAN_DFT_ND, or communicate non-default strides with
    // hipfftw_plan_creation_func::PLAN_DFT_ND or hipfftw_plan_creation_func::PLAN_DFT...
    bool can_use_creation_options(hipfftw_plan_creation_func creation_options) const
    {
        if(!hipfftw_creation_options_are_well_defined(creation_options))
            throw std::invalid_argument(
                "ill-defined creation_options used in can_use_creation_options");
        if(creation_options == hipfftw_plan_creation_func::NONE)
            return false;
        if(std::find(hipfftw_plan_creation_func_candidates.begin(),
                     hipfftw_plan_creation_func_candidates.end(),
                     creation_options)
           == hipfftw_plan_creation_func_candidates.end())
        {
            // creation_options combines several candidates in hipfftw_plan_creation_func_candidates
            // --> parse them individually and find out if any applicable can be used
            return std::any_of(hipfftw_plan_creation_func_candidates.begin(),
                               hipfftw_plan_creation_func_candidates.end(),
                               [=](const hipfftw_plan_creation_func& candidate) {
                                   return (creation_options & candidate)
                                          && can_use_creation_options(candidate);
                               });
        }
        // "creation_options" actually is an individual value in hipfftw_plan_creation_func_candidates
        switch(creation_options)
        {
        case hipfftw_plan_creation_func::PLAN_DFT_ND:
            // only unbatched cases (making distances irrelevant)
            if(batch_rank != 1 || batches.size() != 1 || batches[0] != 1)
                return false;
            // only default strides
            if(istrides != default_strides(dft_kind, plan_placement, fft_io::fft_io_in, lengths)
               || ostrides
                      != default_strides(dft_kind, plan_placement, fft_io::fft_io_out, lengths))
                return false;
            // rank is not passed as an argument but dictated by the called function,
            // (must be 1, 2, or 3), and as many lengths must be passed as individual
            // integer values
            return (rank == 1 || rank == 2 || rank == 3) && get_length_as<int>().size() == rank;
            break;
        case hipfftw_plan_creation_func::PLAN_DFT:
            // only unbatched cases (making distances irrelevant)
            if(batch_rank != 1 || batches.size() != 1 || batches[0] != 1)
                return false;
            // only default strides if lengths are not empty
            if(!lengths.empty())
            {
                if(istrides != default_strides(dft_kind, plan_placement, fft_io::fft_io_in, lengths)
                   || ostrides
                          != default_strides(dft_kind, plan_placement, fft_io::fft_io_out, lengths))
                    return false;
            }
            // the lengths must be representable as integers, if not empty (supposedly
            // intentionally, e.g., for input validation testing purposes)
            return lengths.empty() || get_length_as<int>().size() == rank;
            break;
        case hipfftw_plan_creation_func::PLAN_MANY:
        {
            // batch_rank == 1 only
            if(batch_rank != 1 || batches.size() != 1)
                return false;
            // only strides that may be represented via inembed/onembed
            for(auto io : {fft_io::fft_io_in, fft_io::fft_io_out})
            {
                try
                {
                    (void)get_nembed_from_strides(io);
                }
                catch(const strides_are_incompatible_with_nembed&)
                {
                    return false;
                }
            }
            // the lengths must be representable as integers, if not empty (supposedly
            // intentionally, e.g., for input validation testing purposes)
            return lengths.empty() || get_length_as<int>().size() == rank;
            break;
        }
        case hipfftw_plan_creation_func::PLAN_GURU:
            [[fallthrough]];
        case hipfftw_plan_creation_func::PLAN_GURU64:
            return false;
            break;
        default:
            throw std::runtime_error("hipfftw_helper: internal error encountered (unexpected value "
                                     "for creation_options)");
            break;
        }
        // unreachable
    }

    // checks validity of configuration parameters and whether creation can be
    // attempted via (any of) the given option(s)
    bool is_valid_for_creation_with(hipfftw_plan_creation_func creation_options) const
    {
        if(!hipfftw_creation_options_are_well_defined(creation_options))
            throw std::invalid_argument("invalid creation_options for is_valid_for_creation_with");

        return has_valid_rank(creation_options) && has_valid_lengths() && has_valid_sign()
               && has_valid_flags() && has_valid_batch_rank(creation_options) && has_valid_batches()
               && has_valid_strides(fft_io::fft_io_in, creation_options)
               && has_valid_strides(fft_io::fft_io_out, creation_options)
               && has_valid_distances(fft_io::fft_io_in) && has_valid_distances(fft_io::fft_io_out)
               && can_use_creation_options(creation_options);
    }
    bool is_valid_for_creation() const
    {
        return is_valid_for_creation_with(hipfftw_plan_creation_func::ANY);
    }
    // check expected support by (any of) the given option(s)
    bool has_unsupported_args_for(hipfftw_plan_creation_func creation_options) const
    {
        // extra conditions for valid configurations that are not supported by hipfftw:
        if(rank > 3)
            return true;
        if(batch_rank > 1)
            return true;
        if(flags & FFTW_WISDOM_ONLY)
            return true;
        if(dft_kind == fft_transform_type_real_inverse && rank > 1 && (flags & FFTW_PRESERVE_INPUT))
            return true;
        // negative strides and distances are not supported
        for(const std::vector<ptrdiff_t>& tmp : {istrides, ostrides, idist, odist})
        {
            if(std::any_of(tmp.begin(), tmp.end(), [](const ptrdiff_t& val) { return val < 0; }))
                return true;
        }
        return false;
    }
    bool can_create_plan_with(hipfftw_plan_creation_func creation_options) const
    {
        if(!hipfftw_creation_options_are_well_defined(creation_options))
            throw std::invalid_argument("invalid creation_option for can_create_plan_with");

        if(!is_valid_for_creation_with(creation_options))
            return false;
        if(has_unsupported_args_for(creation_options))
            return false;
        return true;
    }
    bool can_create_plan() const
    {
        return can_create_plan_with(hipfftw_plan_creation_func::ANY);
    }
    bool can_use_execution_option(hipfftw_plan_execution_func exec_option) const
    {
        if(exec_option == hipfftw_plan_execution_func::DEFAULT
           || exec_option == hipfftw_plan_execution_func::EXECUTE)
            return true;
        if(is_complex(dft_kind))
            return exec_option == hipfftw_plan_execution_func::EXECUTE_DFT;
        else if(dft_kind == fft_transform_type_real_forward)
            return exec_option == hipfftw_plan_execution_func::EXECUTE_DFT_R2C;
        else
            return exec_option == hipfftw_plan_execution_func::EXECUTE_DFT_C2R;
    }
    // create a token consistent with other tests to enable kernel precompilation
    // for valid cases, and/or capturing all required details about members otherwise
    std::string token() const
    {
        std::ostringstream ret;
        switch(dft_kind)
        {
        case fft_transform_type_complex_forward:
            ret << "complex_forward";
            break;
        case fft_transform_type_complex_inverse:
            ret << "complex_inverse";
            break;
        case fft_transform_type_real_forward:
            ret << "real_forward";
            break;
        case fft_transform_type_real_inverse:
            ret << "real_inverse";
            break;
        default:
            throw std::runtime_error("unknown type of transform");
        }

        auto append_vec = [&](const std::string& vec_name, const std::vector<ptrdiff_t>& vec) {
            ret << "_" << vec_name;
            if(vec.empty())
                ret << "_none";
            else
            {
                for(const auto& elem : vec)
                    ret << (elem < 0 ? "_negative_" : "_") << std::abs(elem);
            }
        };

        // report rank if invalid and/or empty lengths (distinguishing tokens resulting in failing creation)
        if(!rank_is_valid_for_hipfftw(rank) || lengths.empty())
            ret << "_rank" << (rank < 0 ? "_negative_" : "_") << std::abs(rank);
        append_vec("len", lengths);
        if constexpr(prec == fft_precision_single)
            ret << "_single";
        else
            ret << "_double";
        ret << (plan_placement == fft_placement_inplace ? "_ip" : "_op");
        if(batch_rank != 1)
            ret << "_batch_rank_" << (batch_rank < 0 ? "_negative_" : "_") << std::abs(batch_rank);
        append_vec("batch", batches);
        append_vec("istride", istrides);
        if(!is_real(dft_kind))
            ret << "_CI";
        else if(dft_kind == fft_transform_type_real_forward)
            ret << "_R";
        else
            ret << "_HI";
        append_vec("ostride", ostrides);
        if(!is_real(dft_kind))
            ret << "_CI";
        else if(dft_kind == fft_transform_type_real_forward)
            ret << "_HI";
        else
            ret << "_R";
        append_vec("idist", idist);
        append_vec("odist", odist);

        // for simple copy-pasting of token for testing the same DFT via hipFFT/rocFFT
        ret << "_ioffset_0_0_ooffset_0_0";

        if(!has_valid_sign())
            ret << "_invalid_sign" << (sign < 0 ? "_negative_" : "_") << std::abs(sign);
        ret << "_flags_" << flags;
        return ret.str();
    }
    // create_plan invokes an hipfftw plan creation function for the object's configuration
    // parameters, the corresponding plan pointer returned by hipfftw is stored internally.
    // IMPORTANT NOTE: if one wants to target a specific creation function (as represented
    // by any value in hipfftw_plan_creation_func_candidates), setting the creation_options
    // argument to that specific value effectively bypasses the verification that the
    // object's configuration is actually (expected to be) supported and attempts the plan
    // creation anyways (unless it simply cannot be done, e.g., attempting
    // creation_options = hipfftw_plan_creation_func::PLAN_DFT_ND herein on an object
    // holding a value for rank > 3 simply cannot be done)
    void create_plan(void*                      in,
                     void*                      out,
                     hipfftw_plan_creation_func creation_options
                     = hipfftw_plan_creation_func::ANY) const
    {
        const auto&                      hipfftw_impl  = hipfftw_funcs<prec>::get_instance();
        const hipfftw_plan_creation_func chosen_option = get_creation_func(creation_options);
        if(chosen_option == hipfftw_plan_creation_func::NONE)
        {
            plan_bundle = std::make_shared<hipfftw_plan_bundle_t<prec>>(hipfftw_impl.destroy_plan);
            plan_bundle->creation_io   = {in, out};
            plan_bundle->plan          = nullptr;
            plan_bundle->creation_func = chosen_option;
            plan_bundle->plan_token    = "";
            return;
        }
        // early return if there is no need to (re)build
        if(plan_bundle && plan_bundle->plan_token == token() && plan_bundle->creation_io.first == in
           && plan_bundle->creation_io.second == out && plan_bundle->creation_func == chosen_option)
            return;

        // create the desired plan
        plan_bundle = std::make_shared<hipfftw_plan_bundle_t<prec>>(hipfftw_impl.destroy_plan);
        plan_bundle->plan          = make_plan(in, out, chosen_option);
        plan_bundle->creation_io   = {in, out};
        plan_bundle->creation_func = chosen_option;
        plan_bundle->plan_token    = token();
    }

    // returns a reference FFTW plan for the current configuration
    // The returned plan is NOT owned by this object!
    hipfftw_plan_t<prec> get_reference_plan(void*                      in,
                                            void*                      out,
                                            hipfftw_plan_creation_func creation_options
                                            = hipfftw_plan_creation_func::ANY) const
    {
        const hipfftw_plan_creation_func chosen_option = get_creation_func(creation_options);
        if(chosen_option == hipfftw_plan_creation_func::NONE)
        {
            return nullptr;
        }
        constexpr bool make_reference_plan = true;
        return make_plan<make_reference_plan>(in, out, chosen_option);
    }

    void execute(void*                       execute_in,
                 void*                       execute_out,
                 hipfftw_plan_execution_func exec_option
                 = hipfftw_plan_execution_func::DEFAULT) const
    {
        if(exec_option != hipfftw_plan_execution_func::EXECUTE
           && exec_option != hipfftw_plan_execution_func::EXECUTE_DFT
           && exec_option != hipfftw_plan_execution_func::EXECUTE_DFT_R2C
           && exec_option != hipfftw_plan_execution_func::EXECUTE_DFT_C2R
           && exec_option != hipfftw_plan_execution_func::DEFAULT)
            throw std::runtime_error("ill-defined execution option");

        if(!plan_bundle || plan_bundle->plan_token != token())
        {
            // plan is not created or possibly not up-to-date
            create_plan(execute_in, execute_out);
        }

        const auto& hipfftw_impl = hipfftw_funcs<prec>::get_instance();
        if(exec_option == hipfftw_plan_execution_func::EXECUTE
           || (execute_in == plan_bundle->creation_io.first
               && execute_out == plan_bundle->creation_io.second
               && exec_option == hipfftw_plan_execution_func::DEFAULT))
        {
            hipfftw_impl.execute(plan_bundle->plan);
        }
        else
        {
            if(exec_option == hipfftw_plan_execution_func::EXECUTE_DFT
               || (is_complex(dft_kind) && exec_option == hipfftw_plan_execution_func::DEFAULT))
                hipfftw_impl.execute_dft(plan_bundle->plan,
                                         static_cast<hipfftw_complex_t<prec>*>(execute_in),
                                         static_cast<hipfftw_complex_t<prec>*>(execute_out));
            else if(exec_option == hipfftw_plan_execution_func::EXECUTE_DFT_R2C
                    || (dft_kind == fft_transform_type_real_forward
                        && exec_option == hipfftw_plan_execution_func::DEFAULT))
                hipfftw_impl.execute_dft_r2c(plan_bundle->plan,
                                             static_cast<hipfftw_real_t<prec>*>(execute_in),
                                             static_cast<hipfftw_complex_t<prec>*>(execute_out));
            else if(exec_option == hipfftw_plan_execution_func::EXECUTE_DFT_C2R
                    || (dft_kind == fft_transform_type_real_inverse
                        && exec_option == hipfftw_plan_execution_func::DEFAULT))
                hipfftw_impl.execute_dft_c2r(plan_bundle->plan,
                                             static_cast<hipfftw_complex_t<prec>*>(execute_in),
                                             static_cast<hipfftw_real_t<prec>*>(execute_out));
        }
    }

    size_t get_num_elements_in(fft_io in_or_out) const
    {
        if(in_or_out != fft_io_in && in_or_out != fft_io_out)
            throw std::invalid_argument("invalid in_or_out for get_num_elements_in");
        if(!has_valid_rank() || !has_valid_lengths() || !has_valid_batch_rank()
           || !has_valid_batches())
            throw hipfftw_helper_num_elements_calc_exception(
                "get_num_elements_in requires valid rank, batch_rank, lengths, and batches");
        const auto& strides   = in_or_out == fft_io::fft_io_in ? istrides : ostrides;
        const auto& distances = in_or_out == fft_io::fft_io_in ? idist : odist;
        if(!vector_has_valid_values_as<ptrdiff_t>(strides, rank, 0)
           || !vector_has_valid_values_as<ptrdiff_t>(distances, batch_rank, 0))
            throw hipfftw_helper_num_elements_calc_exception(
                "get_num_elements_in assumes non-negative strides and distances.");
        size_t last_elem_idx = 0;
        for(auto len_dim = lengths.size(); len_dim-- > 0;)
            last_elem_idx += (lengths[len_dim] - 1) * strides[len_dim];
        for(auto batch_dim = batches.size(); batch_dim-- > 0;)
            last_elem_idx += (batches[batch_dim] - 1) * distances[batch_dim];
        return last_elem_idx + 1;
    }

    size_t get_data_byte_size(fft_io in_or_out) const
    {
        if(in_or_out != fft_io_in && in_or_out != fft_io_out)
            throw std::invalid_argument("invalid in_or_out for get_data_byte_size");
        // for in-place, input and output data sizes are enforced equal
        std::vector<fft_io> io_range_to_consider = {in_or_out};
        if(plan_placement == fft_placement_inplace)
            io_range_to_consider.push_back(in_or_out == fft_io::fft_io_in ? fft_io::fft_io_out
                                                                          : fft_io::fft_io_in);

        size_t ret = 0;
        for(auto io : io_range_to_consider)
        {
            const size_t num_elems = get_num_elements_in(io);
            if(is_complex(dft_kind) || (is_fwd(dft_kind) == (io == fft_io_out)))
                ret = std::max(ret, num_elems * sizeof(hipfftw_complex_t<prec>));
            else
                ret = std::max(ret, num_elems * sizeof(hipfftw_real_t<prec>));
        }
        return ret;
    }
    void release_plan() const
    {
        plan_bundle.reset();
    }
};

#endif
