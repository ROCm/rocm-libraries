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

#define HIPFFTW_FUNCS_SPECIALIZATION(prefix, specialization)                         \
    template <>                                                                      \
    struct hipfftw_funcs<specialization>                                             \
    {                                                                                \
    private:                                                                         \
        hipfftw_funcs()                                                              \
        {                                                                            \
            load_implementations(malloc,                                             \
                                 alloc_real,                                         \
                                 alloc_complex,                                      \
                                 free,                                               \
                                 destroy_plan,                                       \
                                 cleanup,                                            \
                                 execute,                                            \
                                 plan_dft_1d,                                        \
                                 plan_dft_2d,                                        \
                                 plan_dft_3d,                                        \
                                 plan_dft,                                           \
                                 plan_dft_r2c_1d,                                    \
                                 plan_dft_r2c_2d,                                    \
                                 plan_dft_r2c_3d,                                    \
                                 plan_dft_r2c,                                       \
                                 plan_dft_c2r_1d,                                    \
                                 plan_dft_c2r_2d,                                    \
                                 plan_dft_c2r_3d,                                    \
                                 plan_dft_c2r,                                       \
                                 print_plan,                                         \
                                 set_timelimit,                                      \
                                 cost,                                               \
                                 flops,                                              \
                                 execute_dft,                                        \
                                 execute_dft_r2c,                                    \
                                 execute_dft_c2r);                                   \
        }                                                                            \
        /* disable copies and moves */                                               \
        hipfftw_funcs(const hipfftw_funcs&) = delete;                                \
        hipfftw_funcs& operator=(const hipfftw_funcs&) = delete;                     \
        hipfftw_funcs(hipfftw_funcs&&)                 = delete;                     \
        hipfftw_funcs& operator=(hipfftw_funcs&&) = delete;                          \
                                                                                     \
    public:                                                                          \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, malloc)          \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, alloc_real)      \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, alloc_complex)   \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, free)            \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, destroy_plan)    \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, cleanup)         \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, execute)         \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_1d)     \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_2d)     \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_3d)     \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft)        \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_r2c_1d) \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_r2c_2d) \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_r2c_3d) \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_r2c)    \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_c2r_1d) \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_c2r_2d) \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_c2r_3d) \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_c2r)    \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, print_plan)      \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, set_timelimit)   \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, cost)            \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, flops)           \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, execute_dft)     \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, execute_dft_r2c) \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, execute_dft_c2r) \
        static const hipfftw_funcs& get_instance()                                   \
        {                                                                            \
            static const hipfftw_funcs instance;                                     \
            return instance;                                                         \
        }                                                                            \
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

static const std::vector<hipfftw_plan_execution_func> hipfftw_plan_execution_func_candidates
    = {hipfftw_plan_execution_func::EXECUTE,
       hipfftw_plan_execution_func::EXECUTE_DFT,
       hipfftw_plan_execution_func::EXECUTE_DFT_R2C,
       hipfftw_plan_execution_func::EXECUTE_DFT_C2R,
       hipfftw_plan_execution_func::DEFAULT};

static bool hipfftw_execution_func_is_well_defined(hipfftw_plan_execution_func exec_func)
{
    return std::find(hipfftw_plan_execution_func_candidates.begin(),
                     hipfftw_plan_execution_func_candidates.end(),
                     exec_func)
           != hipfftw_plan_execution_func_candidates.end();
}

static std::string hipfftw_execution_option_to_string(hipfftw_plan_execution_func execution_option)
{
    if(!hipfftw_execution_func_is_well_defined(execution_option))
        throw std::invalid_argument(
            "invalid execution_option for hipfftw_execution_option_to_string");

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
static bool lengths_are_valid_for_hipfftw_as(const std::vector<ptrdiff_t> len, int intended_rank)
{
    if(!rank_is_valid_for_hipfftw(intended_rank))
        return false; // impossible to validate lengths for an invalid rank
    // check that lengths are all strictly positive and representable with
    // type T without data loss
    return len.size() == intended_rank
           && std::all_of(len.begin(), len.end(), [](const decltype(len)::value_type& val) {
                  return val > 0 && val <= std::numeric_limits<T>::max();
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
    int                    rank = 0;
    std::vector<ptrdiff_t> lengths;
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
            [[fallthrough]];
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
        reset_member_value(dft_kind, dft_kind_to_set);
        reset_member_value(rank, rank_to_set);
        reset_member_value(lengths, lengths_to_set);
        reset_member_value(plan_placement, placement_to_set);
        reset_member_value(sign, sign_to_set);
        reset_member_value(flags, flags_to_set);
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
        if constexpr(std::is_same_v<T, typename decltype(lengths)::value_type>)
            return lengths;
        std::vector<T> ret;
        if(std::any_of(lengths.begin(),
                       lengths.end(),
                       [](const typename decltype(lengths)::value_type& val) {
                           return val < std::numeric_limits<T>::lowest()
                                  || val > std::numeric_limits<T>::max();
                       }))
        {
            // not a safe conversion, return empty lengths
            return ret;
        }
        ret.assign(lengths.begin(), lengths.end());
        return ret;
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
    std::shared_ptr<hipfftw_plan_bundle_t<prec>> get_plan_bundle() const
    {
        return plan_bundle;
    }
    template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
    std::vector<T> get_strides_as(fft_io io) const
    {
        if(!rank_is_valid_for_hipfftw(rank) || !has_valid_lengths())
            throw std::runtime_error(
                "cannot calculate default strides with invalid rank or invalid lengths");
        // only default strides for now
        std::vector<ptrdiff_t> strides(rank, 1);
        if(rank > 1)
        {
            if(is_complex(dft_kind))
                strides[rank - 2] = lengths.back();
            else
            {
                if(is_fwd(dft_kind) == (io == fft_io::fft_io_out))
                    strides[rank - 2] = lengths.back() / 2 + 1;
                else
                {
                    if(plan_placement == fft_placement_inplace)
                        strides[rank - 2] = 2 * (lengths.back() / 2 + 1);
                    else
                        strides[rank - 2] = lengths.back();
                }
            }
        }
        for(auto dim = rank - 3; dim >= 0; dim--)
            strides[dim] = strides[dim + 1] * lengths[dim + 1];

        std::vector<T> ret;
        if(std::any_of(strides.begin(),
                       strides.end(),
                       [](const typename decltype(strides)::value_type& val) {
                           return val < std::numeric_limits<T>::lowest()
                                  || val > std::numeric_limits<T>::max();
                       }))
        {
            // not a safe conversion, return empty lengths
            return ret;
        }
        ret.assign(strides.begin(), strides.end());
        return ret;
    }
    template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
    T get_dist_as(fft_io io) const
    {
        if(!rank_is_valid_for_hipfftw(rank) || !has_valid_lengths())
            throw std::runtime_error(
                "cannot calculate default distance(s) with invalid rank or invalid lengths");
        // only default distances for now
        ptrdiff_t dist = 0;
        if(rank == 1)
        {
            if(is_complex(dft_kind))
                dist = lengths.back();
            else
            {
                if(is_fwd(dft_kind) == (io == fft_io::fft_io_out))
                    dist = lengths.back() / 2 + 1;
                else
                {
                    if(plan_placement == fft_placement_inplace)
                        dist = 2 * (lengths.back() / 2 + 1);
                    else
                        dist = lengths.back();
                }
            }
        }
        else
        {
            const auto strides = get_strides_as<ptrdiff_t>(io);
            dist               = strides.front() * lengths.front();
        }
        if(dist < std::numeric_limits<T>::lowest() || dist > std::numeric_limits<T>::max())
            throw std::runtime_error("distance cannot be safely converted to the desired type");
        return static_cast<T>(dist);
    }
    template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
    T get_nbatch_as(fft_io io) const
    {
        // only unbatched for now
        T ret = 1;
        return ret;
    }

    // validity checks
    bool has_valid_rank() const
    {
        return rank_is_valid_for_hipfftw(rank);
    }
    bool has_valid_lengths() const
    {
        return lengths_are_valid_for_hipfftw_as<ptrdiff_t>(lengths, rank);
    }
    bool has_valid_sign() const
    {
        return sign_is_valid_for_hipfftw(sign, dft_kind);
    }
    bool has_valid_flags() const
    {
        return flags_are_valid_for_hipfftw(flags);
    }
    // checks if the current parameters can be used with (any of) the given option(s) of
    // plan creation (NOT whether they're valid or not). For instance, one cannot possibly
    // communicate rank > 3 with hipfftw_plan_creation_func::PLAN_DFT_ND, or communicate
    // non-default strides with hipfftw_plan_creation_func::PLAN_DFT_ND or
    // hipfftw_plan_creation_func::PLAN_DFT...
    // TODO: expand logic when extra configuration parameters are added (e.g. batch sizes,
    // strides, etc.)
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
            // rank is not passed as an argument but dictated by the called function,
            // (must be 1, 2, or 3), and as many lengths must be passed as individual
            // integer values
            return (rank == 1 || rank == 2 || rank == 3) && get_length_as<int>().size() == rank;
            break;
        case hipfftw_plan_creation_func::PLAN_DFT:
            // the lengths must be representable as integers, if not empty (supposedly
            // intentionally, e.g., for input validation testing purposes)
            return lengths.empty() || get_length_as<int>().size() == rank;
            break;
        case hipfftw_plan_creation_func::PLAN_MANY:
            [[fallthrough]];
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

        // TODO: expand the global validity checks below when this struct is
        // expanded to cover more configurations (e.g., batching, srides, etc.)
        return has_valid_rank() && has_valid_lengths() && has_valid_sign() && has_valid_flags()
               && can_use_creation_options(creation_options);
    }
    bool is_valid_for_creation() const
    {
        return is_valid_for_creation_with(hipfftw_plan_creation_func::ANY);
    }
    // check expected support by (any of) the given option(s)
    bool has_unsupported_args_for(hipfftw_plan_creation_func creation_options) const
    {
        // extra conditions for configurations supported by hipfftw:
        if(rank > 3)
            return true;
        if(flags & FFTW_WISDOM_ONLY)
            return true;
        if(dft_kind == fft_transform_type_real_inverse && rank > 1 && (flags & FFTW_PRESERVE_INPUT))
            return true;
        if(!(creation_options & hipfftw_plan_creation_func::PLAN_GURU64) && has_valid_rank()
           && has_valid_lengths())
        {
            // cannot handle data sizes involving more elements than the
            // largest representable int value
            if(get_num_elements_in(fft_io_in) > std::numeric_limits<int>::max()
               || get_num_elements_in(fft_io_out) > std::numeric_limits<int>::max())
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
        if(!hipfftw_execution_func_is_well_defined(exec_option))
            throw std::invalid_argument(
                "invalid exec_option for hipfftw_helper::can_use_execution_option");

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

        // report rank if invalid
        if(!has_valid_rank() || lengths.empty())
            ret << "_invalid_rank" << (rank < 0 ? "_negative_" : "_") << std::abs(rank);
        ret << "_len";
        if(lengths.empty())
            ret << "_none";
        else
        {
            for(const auto& len : lengths)
                ret << (len < 0 ? "_negative_" : "_") << std::abs(len);
        }
        if constexpr(prec == fft_precision_single)
            ret << "_single";
        else
            ret << "_double";
        ret << (plan_placement == fft_placement_inplace ? "_ip" : "_op");
        // only supporting unbatched cases as of now
        ret << "_batch_1";
        if(has_valid_rank() && has_valid_lengths())
        {
            ret << "_istride";
            for(const auto& stride : get_strides_as<size_t>(fft_io::fft_io_in))
                ret << "_" << stride;
            if(!is_real(dft_kind))
                ret << "_CI";
            else if(dft_kind == fft_transform_type_real_forward)
                ret << "_R";
            else
                ret << "_HI";
            ret << "_ostride";
            for(const auto& stride : get_strides_as<size_t>(fft_io::fft_io_out))
                ret << "_" << stride;
            if(!is_real(dft_kind))
                ret << "_CI";
            else if(dft_kind == fft_transform_type_real_forward)
                ret << "_HI";
            else
                ret << "_R";
            ret << "_idist_" << get_dist_as<size_t>(fft_io::fft_io_in);
            ret << "_odist_" << get_dist_as<size_t>(fft_io::fft_io_out);
            ret << "_ioffset_0_0_ooffset_0_0";
        }
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
        if(!hipfftw_execution_func_is_well_defined(exec_option))
            throw std::invalid_argument("invalid exec_option for hipfftw_helper::execute");

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

    // TODO: revise/expand logic below when the structure is expanded for more cases (batches,
    // non-default strides, etc.)
    size_t get_num_elements_in(fft_io in_or_out) const
    {
        if(in_or_out != fft_io_in && in_or_out != fft_io_out)
            throw std::invalid_argument("invalid in_or_out for get_num_elements_in");
        if(!has_valid_rank() || !has_valid_lengths())
            throw std::runtime_error("get_num_elements_in requires valid rank and lengths");
        const auto tmp = get_length_as<size_t>();
        if(tmp.empty() || tmp.size() != rank)
        {
            throw std::runtime_error(
                "get_num_elements_in failed to correctly convert lengths to size_t values");
        }
        size_t num_elems = 1;
        if(is_complex(dft_kind))
        {
            num_elems *= tmp[rank - 1];
        }
        else
        {
            const size_t cmplx_len = tmp[rank - 1] / 2 + 1;
            if(is_fwd(dft_kind) == (in_or_out == fft_io_out))
                num_elems *= cmplx_len;
            else
                num_elems
                    *= plan_placement == fft_placement_inplace ? 2 * cmplx_len : tmp[rank - 1];
        }
        num_elems *= product(tmp.begin(), tmp.begin() + rank - 1);
        return num_elems;
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
