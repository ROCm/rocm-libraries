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

#include "../shared/array_validator.h"
#include "../shared/client_data_layout_helpers.h"
#include "../shared/environment.h"
#include "../shared/fft_params.h"
#include <algorithm>
#include <fftw3.h>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>

#ifdef _WIN32
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
    using plan_t                                 = fftwf_plan;
    using complex_t                              = fftwf_complex;
    using real_t                                 = float;
    static constexpr std::string_view prec_label = R"(single)";
};
template <>
struct hipfftw_trait<fft_precision_double>
{
    using plan_t                                 = fftw_plan;
    using complex_t                              = fftw_complex;
    using real_t                                 = double;
    static constexpr std::string_view prec_label = R"(double)";
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
#ifdef _WIN32
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
#ifdef _WIN32
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
#ifdef _WIN32
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
#ifdef _WIN32
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

#define HIPFFTW_FUNCS_SPECIALIZATION(prefix, specialization)                             \
    template <>                                                                          \
    struct hipfftw_funcs<specialization>                                                 \
    {                                                                                    \
    private:                                                                             \
        hipfftw_funcs()                                                                  \
        {                                                                                \
            load_implementations(malloc,                                                 \
                                 alloc_real,                                             \
                                 alloc_complex,                                          \
                                 free,                                                   \
                                 destroy_plan,                                           \
                                 cleanup,                                                \
                                 execute,                                                \
                                 plan_dft_1d,                                            \
                                 plan_dft_2d,                                            \
                                 plan_dft_3d,                                            \
                                 plan_dft,                                               \
                                 plan_dft_r2c_1d,                                        \
                                 plan_dft_r2c_2d,                                        \
                                 plan_dft_r2c_3d,                                        \
                                 plan_dft_r2c,                                           \
                                 plan_dft_c2r_1d,                                        \
                                 plan_dft_c2r_2d,                                        \
                                 plan_dft_c2r_3d,                                        \
                                 plan_dft_c2r,                                           \
                                 print_plan,                                             \
                                 set_timelimit,                                          \
                                 cost,                                                   \
                                 flops,                                                  \
                                 execute_dft,                                            \
                                 execute_dft_r2c,                                        \
                                 execute_dft_c2r,                                        \
                                 plan_many_dft,                                          \
                                 plan_many_dft_r2c,                                      \
                                 plan_many_dft_c2r,                                      \
                                 plan_guru_dft,                                          \
                                 plan_guru_dft_r2c,                                      \
                                 plan_guru_dft_c2r,                                      \
                                 plan_guru64_dft,                                        \
                                 plan_guru64_dft_r2c,                                    \
                                 plan_guru64_dft_c2r);                                   \
        }                                                                                \
        /* disable copies and moves */                                                   \
        hipfftw_funcs(const hipfftw_funcs&) = delete;                                    \
        hipfftw_funcs& operator=(const hipfftw_funcs&) = delete;                         \
        hipfftw_funcs(hipfftw_funcs&&)                 = delete;                         \
        hipfftw_funcs& operator=(hipfftw_funcs&&) = delete;                              \
                                                                                         \
    public:                                                                              \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, malloc)              \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, alloc_real)          \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, alloc_complex)       \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, free)                \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, destroy_plan)        \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, cleanup)             \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, execute)             \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_1d)         \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_2d)         \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_3d)         \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft)            \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_r2c_1d)     \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_r2c_2d)     \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_r2c_3d)     \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_r2c)        \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_c2r_1d)     \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_c2r_2d)     \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_c2r_3d)     \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_dft_c2r)        \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, print_plan)          \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, set_timelimit)       \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, cost)                \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, flops)               \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, execute_dft)         \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, execute_dft_r2c)     \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, execute_dft_c2r)     \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_many_dft)       \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_many_dft_r2c)   \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_many_dft_c2r)   \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_guru_dft)       \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_guru_dft_r2c)   \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_guru_dft_c2r)   \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_guru64_dft)     \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_guru64_dft_r2c) \
        HIPFFTW_DECLARE_DYNAMICALLY_LOADED_FUNCTION_POINTER(prefix, plan_guru64_dft_c2r) \
        static const hipfftw_funcs& get_instance()                                       \
        {                                                                                \
            static const hipfftw_funcs instance;                                         \
            return instance;                                                             \
        }                                                                                \
        template <bool use_guru64>                                                       \
        const auto& get_plan_guru_dft_r2c() const                                        \
        {                                                                                \
            if constexpr(use_guru64)                                                     \
                return plan_guru64_dft_r2c;                                              \
            else                                                                         \
                return plan_guru_dft_r2c;                                                \
        }                                                                                \
        template <bool use_guru64>                                                       \
        const auto& get_plan_guru_dft_c2r() const                                        \
        {                                                                                \
            if constexpr(use_guru64)                                                     \
                return plan_guru64_dft_c2r;                                              \
            else                                                                         \
                return plan_guru_dft_c2r;                                                \
        }                                                                                \
        template <bool use_guru64>                                                       \
        const auto& get_plan_guru_dft() const                                            \
        {                                                                                \
            if constexpr(use_guru64)                                                     \
                return plan_guru64_dft;                                                  \
            else                                                                         \
                return plan_guru_dft;                                                    \
        }                                                                                \
    }

HIPFFTW_FUNCS_SPECIALIZATION(fftwf_, fft_precision_single);
HIPFFTW_FUNCS_SPECIALIZATION(fftw_, fft_precision_double);

// structure enabling verbosity for hipfftw's exception handler and redirecting std::cerr
// to a runtime buffer throughout its lifetime (unless it was already enabled prior/externally)
struct hipfftw_exception_logger
{
private:
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
    NONE        = 0x0, // not to be used (except for validating values)
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
            std::cerr << "WARNING: A " << hipfftw_trait<prec>::prec_label
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
static constexpr ptrdiff_t min_ptrdiff_t_representable_as()
{
    if constexpr(std::is_unsigned_v<T> || sizeof(T) <= sizeof(ptrdiff_t))
        return std::numeric_limits<T>::lowest();
    else
        return std::numeric_limits<ptrdiff_t>::lowest();
}

template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
static constexpr ptrdiff_t max_ptrdiff_t_representable_as()
{
    if constexpr(sizeof(T) < sizeof(ptrdiff_t)
                 || (sizeof(T) == sizeof(ptrdiff_t) && std::is_signed_v<T>))
        return std::numeric_limits<T>::max();
    else
        return std::numeric_limits<ptrdiff_t>::max();
}

template <typename T>
static bool
    vector_has_valid_values_as(const std::vector<ptrdiff_t>& vals,
                               int                           intended_rank,
                               ptrdiff_t min_valid_val = min_ptrdiff_t_representable_as<T>(),
                               ptrdiff_t max_valid_val = max_ptrdiff_t_representable_as<T>())
{
    if(!rank_is_valid_for_hipfftw(intended_rank))
        return false; // impossible to validate values for an invalid intended_rank
    // check that vals are all in [min_valid_val, max_valid_val]
    return vals.size() == static_cast<size_t>(intended_rank)
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

    fft_transform_type                dft_kind;
    int                               rank       = 0;
    int                               batch_rank = 0;
    std::vector<ptrdiff_t>            lengths;
    std::vector<ptrdiff_t>            istrides;
    std::vector<ptrdiff_t>            ostrides;
    std::optional<hipfftw_ionembed_t> ionembed;
    std::vector<ptrdiff_t>            batches;
    std::vector<ptrdiff_t>            idist;
    std::vector<ptrdiff_t>            odist;
    fft_result_placement              plan_placement;
    int                               sign  = 0;
    unsigned                          flags = std::numeric_limits<unsigned>::max();

    template <typename T>
    void reset_member_value(T& member, const T& new_value)
    {
        if(new_value != member)
        {
            member = new_value;
            plan_bundle.reset();
        }
    }
    template <typename T>
    void reset_member_value(std::optional<T>& member, const T& new_value)
    {
        if(!member || new_value != *member)
        {
            member = new_value;
            plan_bundle.reset();
        }
    }

    hipfftw_plan_creation_func get_creation_func(hipfftw_plan_creation_func creation_options) const
    {
        if(!hipfftw_creation_options_are_well_defined(creation_options))
            throw std::invalid_argument(
                "invalid creation_options for hipfftw_helper::get_creation_func");
        if(!can_use_creation_options(creation_options))
        {
            // e.g., rank < 0 with creation_options == hipfftw_plan_creation_func::PLAN_DFT_ND
            throw std::invalid_argument(
                "hipfftw_helper::get_creation_func: the plan creation options "
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
            throw std::invalid_argument("hipfftw_helper::make_plan: invalid I/O argument for plan "
                                        "creation w.r.t. expected plan placement.");

        // fetch/infer plan creation function arguments
        const auto& hipfftw_impl = hipfftw_funcs<prec>::get_instance();
        const auto  int_len      = get_lengths_as<int>();
        const int*  int_len_ptr  = int_len.empty() ? nullptr : int_len.data();

        switch(chosen_creation)
        {
        case hipfftw_plan_creation_func::PLAN_DFT_ND:
        {
            if(!can_use_creation_options(hipfftw_plan_creation_func::PLAN_DFT_ND))
                throw std::runtime_error("hipfftw_helper::make_plan: "
                                         "hipfftw_plan_creation_func::PLAN_DFT_ND cannot be used.");
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
                throw std::runtime_error("hipfftw_helper::make_plan: "
                                         "hipfftw_plan_creation_func::PLAN_DFT cannot be used.");

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
                throw std::runtime_error("hipfftw_helper::make_plan: "
                                         "hipfftw_plan_creation_func::PLAN_MANY cannot be used.");
            if(!ionembed)
                throw std::logic_error(
                    "hipfftw_helper::make_plan: hipfftw_plan_creation_func::PLAN_MANY seemingly "
                    "usable but io_nembed has no value/was not set.");

            const int* inembed = ionembed->get_nembed(fft_io::fft_io_in);
            const int* onembed = ionembed->get_nembed(fft_io::fft_io_out);

            if(dft_kind == fft_transform_type_real_forward)
            {
                return hipfftw_impl.plan_many_dft_r2c.template call<make_reference_plan>(
                    rank,
                    int_len_ptr,
                    static_cast<int>(batches[0]),
                    static_cast<hipfftw_real_t<prec>*>(in),
                    inembed,
                    ionembed->get_elementary_stride(fft_io::fft_io_in),
                    static_cast<int>(idist[0]),
                    static_cast<hipfftw_complex_t<prec>*>(out),
                    onembed,
                    ionembed->get_elementary_stride(fft_io::fft_io_out),
                    static_cast<int>(odist[0]),
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
                    ionembed->get_elementary_stride(fft_io::fft_io_in),
                    static_cast<int>(idist[0]),
                    static_cast<hipfftw_real_t<prec>*>(out),
                    onembed,
                    ionembed->get_elementary_stride(fft_io::fft_io_out),
                    static_cast<int>(odist[0]),
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
                    ionembed->get_elementary_stride(fft_io::fft_io_in),
                    static_cast<int>(idist[0]),
                    static_cast<hipfftw_complex_t<prec>*>(out),
                    onembed,
                    ionembed->get_elementary_stride(fft_io::fft_io_out),
                    static_cast<int>(odist[0]),
                    sign,
                    flags);
            }
        }
        break;
        case hipfftw_plan_creation_func::PLAN_GURU:
            return create_guru_plan<make_reference_plan, /* use_guru64 = */ false>(in, out);
            break;
        case hipfftw_plan_creation_func::PLAN_GURU64:
            return create_guru_plan<make_reference_plan, /* use_guru64 = */ true>(in, out);
            break;
        default:
            throw std::runtime_error("hipfftw_helper::make_plan: unknown kind of plan creation");
            break;
        }
        // unreachable
    }

    template <bool make_reference_plan, bool use_guru64>
    hipfftw_plan_t<prec> create_guru_plan(void* in, void* out) const
    {
        if constexpr(use_guru64)
        {
            if(!can_use_creation_options(hipfftw_plan_creation_func::PLAN_GURU64))
                throw std::runtime_error("hipfftw_helper::create_guru_plan: "
                                         "hipfftw_plan_creation_func::PLAN_GURU64 cannot be used.");
        }
        else
        {
            if(!can_use_creation_options(hipfftw_plan_creation_func::PLAN_GURU))
                throw std::runtime_error("hipfftw_helper::create_guru_plan: "
                                         "hipfftw_plan_creation_func::PLAN_GURU cannot be used.");
        }
        // NOTE: fftw_iodim (resp. fftw_iodim64) and fftwf_iodim (resp. fftwf_iodim64) are actually
        // the same types
        static_assert(
            std::is_same_v<fftw_iodim64, fftwf_iodim64> && std::is_same_v<fftw_iodim, fftwf_iodim>);
        using io_dim_t           = std::conditional_t<use_guru64, fftw_iodim64, fftw_iodim>;
        const auto& hipfftw_impl = hipfftw_funcs<prec>::get_instance();
        const auto* dims         = get_guru_dims<io_dim_t>();
        const auto* howmany_dims = get_guru_howmany_dims<io_dim_t>();

        if(dft_kind == fft_transform_type_real_forward)
        {
            const auto& plan_creation_functor
                = hipfftw_impl.template get_plan_guru_dft_r2c<use_guru64>();
            return plan_creation_functor.template call<make_reference_plan>(
                rank,
                dims,
                batch_rank,
                howmany_dims,
                static_cast<hipfftw_real_t<prec>*>(in),
                static_cast<hipfftw_complex_t<prec>*>(out),
                flags);
        }
        else if(dft_kind == fft_transform_type_real_inverse)
        {
            const auto& plan_creation_functor
                = hipfftw_impl.template get_plan_guru_dft_c2r<use_guru64>();
            return plan_creation_functor.template call<make_reference_plan>(
                rank,
                dims,
                batch_rank,
                howmany_dims,
                static_cast<hipfftw_complex_t<prec>*>(in),
                static_cast<hipfftw_real_t<prec>*>(out),
                flags);
        }
        else
        {
            const auto& plan_creation_functor
                = hipfftw_impl.template get_plan_guru_dft<use_guru64>();
            return plan_creation_functor.template call<make_reference_plan>(
                rank,
                dims,
                batch_rank,
                howmany_dims,
                static_cast<hipfftw_complex_t<prec>*>(in),
                static_cast<hipfftw_complex_t<prec>*>(out),
                sign,
                flags);
        }
    }

    // converts vec to an std::vector<T> if it can be done so without underflow/overflow
    // [throws a type_conversion_exception if not]
    template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
    static std::vector<T> convert_vector_to(const std::vector<ptrdiff_t>& vec)
    {
        if constexpr(std::is_same_v<T, ptrdiff_t>)
            return vec;
        std::vector<T> ret;
        if(vec.empty())
            return ret;
        if(!vector_has_valid_values_as<T>(vec, vec.size()))
        {
            throw type_conversion_exception(
                "a hipfftw_helper object could not convert one of its std::vector<ptrdiff> members "
                "to a vector of another integral type (value(s) out of representable range).");
        }
        ret.assign(vec.begin(), vec.end());
        return ret;
    }

    void reset_io_nembed_from_strides()
    {
        try
        {
#ifdef __HIP_PLATFORM_AMD__
            // use nullptr as ionembed for default layouts ~half of the time in order to
            // guarantee testing thereof
            std::hash<std::string> hasher;
            const bool             use_nullptr_for_default_inembed = hasher(token() + "_in") & 1;
            const bool             use_nullptr_for_default_onembed = hasher(token() + "_out") & 1;
#else
            // NOTE: cuFFTW does not accept nullptr for default ionembed
            constexpr bool use_nullptr_for_default_inembed = false;
            constexpr bool use_nullptr_for_default_onembed = false;
#endif
            hipfftw_ionembed_t tmp(istrides,
                                   ostrides,
                                   lengths,
                                   dft_kind,
                                   plan_placement,
                                   use_nullptr_for_default_inembed,
                                   use_nullptr_for_default_onembed);
            reset_member_value(ionembed, tmp);
        }
        catch(const ionembed_exception& e)
        {
            // couldnt' construct the hipfftw_ionembed_t object
            if(ionembed)
                plan_bundle.reset();
            // cannot use ionembed
            ionembed.reset();
        }
    }

    struct cannot_use_guru : std::runtime_error
    {
        cannot_use_guru()
            : std::runtime_error(
                "guru plan creation functions cannot be used (integral value out of range)."){};
    };

    template <typename iodim>
    const iodim* get_guru_dims() const
    {
        static_assert(std::is_same_v<iodim, fftw_iodim> || std::is_same_v<iodim, fftw_iodim64>);
        // cannot generate sensible "dims" from empty lengths and/or with invalid rank
        if(lengths.empty() || !rank_is_valid_for_hipfftw(rank))
            return nullptr;
        if(lengths.size() != static_cast<size_t>(rank)
           || istrides.size() != static_cast<size_t>(rank)
           || ostrides.size() != static_cast<size_t>(rank))
            throw std::logic_error("hipfftw_helper::get_guru_dims(): size mismatch between "
                                   "lengths, istrides, and/or ostrides.");
        if constexpr(std::is_same_v<iodim, fftw_iodim>)
        {
            for(const auto& vec : {lengths, istrides, ostrides})
                if(!vector_has_valid_values_as<int>(vec, vec.size()))
                {
                    throw cannot_use_guru();
                }
        }
        static std::vector<iodim> dims_vec;
        dims_vec.resize(rank);
        for(auto dim = 0; dim < rank; dim++)
        {
            dims_vec[dim].n  = lengths[dim];
            dims_vec[dim].is = istrides[dim];
            dims_vec[dim].os = ostrides[dim];
        }
        return dims_vec.data();
    }

    template <typename iodim>
    const iodim* get_guru_howmany_dims() const
    {
        static_assert(std::is_same_v<iodim, fftw_iodim> || std::is_same_v<iodim, fftw_iodim64>);
        // cannot generate sensible "howmany_dims" from empty batches and/or with invalid batch_rank
        if(batches.empty() || !rank_is_valid_for_hipfftw(batch_rank))
            return nullptr;
        if(batches.size() != static_cast<size_t>(batch_rank)
           || idist.size() != static_cast<size_t>(batch_rank)
           || odist.size() != static_cast<size_t>(batch_rank))
            throw std::logic_error("hipfftw_helper::get_guru_dims(): size mismatch between "
                                   "batches, idist, and/or odist.");
        if constexpr(std::is_same_v<iodim, fftw_iodim>)
        {
            for(const auto& vec : {batches, idist, odist})
                if(!vector_has_valid_values_as<int>(vec, vec.size()))
                {
                    throw cannot_use_guru();
                }
        }
        static std::vector<iodim> howmany_dims_vec;
        howmany_dims_vec.resize(batch_rank);
        for(auto batch_dim = 0; batch_dim < batch_rank; batch_dim++)
        {
            howmany_dims_vec[batch_dim].n  = batches[batch_dim];
            howmany_dims_vec[batch_dim].is = idist[batch_dim];
            howmany_dims_vec[batch_dim].os = odist[batch_dim];
        }
        return howmany_dims_vec.data();
    }

    // (private) validity checks
    bool has_valid_rank(hipfftw_plan_creation_func creation_options
                        = hipfftw_plan_creation_func::ANY) const
    {
        if(!hipfftw_creation_options_are_well_defined(creation_options))
            throw std::invalid_argument(
                "ill-defined creation_options passed to hipfftw_helper::has_valid_rank");
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
                throw std::runtime_error("hipfftw_helper::has_valid_rank: internal error "
                                         "encountered (unexpected value for creation_func)");
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
                "ill-defined creation_options passed to hipfftw_helper::has_valid_batch_rank");
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
                throw std::runtime_error("hipfftw_helper::has_valid_batch_rank internal error "
                                         "encountered (unexpected value for creation_func)");
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
            throw std::invalid_argument("invalid io passed to hipfftw_helper::has_valid_strides");

        if(!hipfftw_creation_options_are_well_defined(creation_options))
            throw std::invalid_argument(
                "ill-defined creation_options passed to hipfftw_helper::has_valid_strides");
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
            for(int dim = 0; ret && dim < rank - 1; dim++)
            {
                if(lengths[dim] == 1)
                    continue;
                ret = ifact * istrides[dim] == ofact * ostrides[dim];
            }
            if(ret)
            {
                if(is_complex(dft_kind))
                    ret = istrides.back() == ostrides.back();
                else // unit elementary strides only for in-place real transforms
                    ret = istrides.back() == 1 && ostrides.back() == 1;
            }
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
                bool valid_io_nembed = ionembed.has_value();
                if(valid_io_nembed && !lengths.empty())
                {
                    if(std::any_of(
                           lengths.begin(), lengths.end(), [](ptrdiff_t len) { return len != 1; })
                       && (ionembed->get_elementary_stride(fft_io::fft_io_in) == 0
                           || ionembed->get_elementary_stride(fft_io::fft_io_out) == 0))
                        valid_io_nembed = false;
                    for(auto io : {fft_io::fft_io_in, fft_io::fft_io_out})
                    {
                        const int* nembed = ionembed->get_nembed(io);
                        if(!nembed) // <-- default, always valid
                            continue;
                        for(auto dim = lengths.size(); dim-- > 0;)
                        {
                            auto min_nembed = lengths[dim];
                            if(is_real(dft_kind) && dim == lengths.size() - 1)
                            {
                                if(is_fwd(dft_kind) == (io == fft_io::fft_io_out))
                                    min_nembed = lengths[dim] / 2 + 1;
                                else if(plan_placement == fft_placement_inplace)
                                    min_nembed = 2 * (lengths[dim] / 2 + 1);
                            }
                            if(nembed[dim] < min_nembed)
                                valid_io_nembed = false;
                        }
                    }
                }
                ret = valid_io_nembed;
            }
            break;
            case hipfftw_plan_creation_func::PLAN_GURU:
            {
                ret = vector_has_valid_values_as<int>(strides, rank);
            }
            break;
            case hipfftw_plan_creation_func::PLAN_GURU64:
            {
                ret = vector_has_valid_values_as<ptrdiff_t>(strides, rank);
            }
            break;
            default:
                throw std::runtime_error("hipfftw_helper::has_valid_strides: internal error "
                                         "encountered (unexpected value for creation_func)");
            }
        }
        return ret;
    }
    bool has_valid_distances(fft_io io) const
    {
        if(io != fft_io::fft_io_in && io != fft_io::fft_io_out)
            throw std::invalid_argument("invalid io passed to hipfftw_helper::has_valid_distances");

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
            for(int batch_dim = 0; ret && batch_dim < batch_rank; batch_dim++)
                ret = batches[batch_dim] == 1
                      || ifact * idist[batch_dim] == ofact * odist[batch_dim];
        }
        return ret;
    }

    static void create_temp_plan_from_token(const std::string& token)
    {
        hipfftw_helper<prec> temp;
        temp.from_token(token);
        hostbuf input;
        input.alloc(temp.get_data_byte_size(fft_io::fft_io_in));
        if(temp.plan_placement == fft_placement_inplace)
            temp.create_plan(input.data(), input.data());
        else
        {
            hostbuf output;
            output.alloc(temp.get_data_byte_size(fft_io::fft_io_out));
            temp.create_plan(input.data(), output.data());
        }
    }

public:
    hipfftw_helper()                       = default;
    ~hipfftw_helper()                      = default;
    hipfftw_helper(hipfftw_helper&& other) = default;
    hipfftw_helper& operator=(hipfftw_helper&& other) = default;
    hipfftw_helper(const hipfftw_helper& other)       = default;
    hipfftw_helper& operator=(const hipfftw_helper& rhs) = default;

    // overload for default, unbatched configurations (compatible with plan_dft* funcs)
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

    // overload for advanced configurations (compatible with plan_many* funcs)
    void set_creation_args(fft_transform_type            dft_kind_to_set,
                           int                           rank_to_set,
                           const std::vector<ptrdiff_t>& lengths_to_set,
                           fft_result_placement          placement_to_set,
                           int                           sign_to_set,
                           unsigned                      flags_to_set,
                           const hipfftw_ionembed_t&     ionembed_to_set,
                           ptrdiff_t                     batch_to_set,
                           ptrdiff_t                     idist_to_set,
                           ptrdiff_t                     odist_to_set)
    {
        if(rank_is_valid_for_hipfftw(rank_to_set))
        {
            if(!lengths_to_set.empty() && lengths_to_set.size() != static_cast<size_t>(rank_to_set))
                throw std::invalid_argument(
                    "Inconsistent size for non-empty lengths given to hipfftw::set_creation_args.");
            for(auto io : {fft_io::fft_io_in, fft_io::fft_io_out})
            {
                if(ionembed_to_set.get_nembed(io)
                   && ionembed_to_set.get_nembed_size(io) != static_cast<size_t>(rank_to_set))
                    throw std::invalid_argument("Inconsistent size for non-empty inembed or "
                                                "onembed given to hipfftw::set_creation_args.");
            }
        }

        reset_member_value(dft_kind, dft_kind_to_set);
        reset_member_value(rank, rank_to_set);
        reset_member_value(lengths, lengths_to_set);
        reset_member_value(plan_placement, placement_to_set);
        reset_member_value(sign, sign_to_set);
        reset_member_value(flags, flags_to_set);
        reset_member_value(
            istrides,
            ionembed_to_set.as_generalized_strides(
                fft_io::fft_io_in, dft_kind_to_set, placement_to_set, lengths_to_set));
        reset_member_value(
            ostrides,
            ionembed_to_set.as_generalized_strides(
                fft_io::fft_io_out, dft_kind_to_set, placement_to_set, lengths_to_set));
        reset_member_value(ionembed, ionembed_to_set);
        reset_member_value(batch_rank, 1 /* implicit */);
        reset_member_value(batches, std::vector<ptrdiff_t>(1, batch_to_set));
        reset_member_value(idist, std::vector<ptrdiff_t>(1, idist_to_set));
        reset_member_value(odist, std::vector<ptrdiff_t>(1, odist_to_set));
    }

    // most general overload
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
        if(rank_is_valid_for_hipfftw(rank_to_set))
        {
            for(const auto& vec : {lengths_to_set, istrides_to_set, ostrides_to_set})
            {
                if(!vec.empty() && vec.size() != static_cast<size_t>(rank_to_set))
                {
                    throw std::invalid_argument(
                        "Inconsistent size for non-empty lengths, istrides, or ostrides given to "
                        "hipfftw::set_creation_args.");
                }
            }
        }
        if(rank_is_valid_for_hipfftw(batch_rank_to_set))
        {
            for(const auto& vec : {batches_to_set, idist_to_set, odist_to_set})
            {
                if(!vec.empty() && vec.size() != static_cast<size_t>(batch_rank_to_set))
                {
                    throw std::invalid_argument("Inconsistent size for non-empty batches, idist, "
                                                "or odist given to hipfftw::set_creation_args.");
                }
            }
        }

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
        reset_io_nembed_from_strides();
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
    // (a type_conversion_exception is thrown otherwise)
    template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
    std::vector<T> get_lengths_as() const
    {
        return convert_vector_to<T>(lengths);
    }
    const decltype(lengths)& get_lengths() const
    {
        return lengths;
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
    // (a type_conversion_exception is thrown otherwise)
    template <typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>
    std::vector<T> get_strides_as(fft_io io) const
    {
        if(io != fft_io::fft_io_in && io != fft_io::fft_io_out)
            throw std::invalid_argument("invalid io passed to hipfftw_helper::get_strides_as");
        const std::vector<ptrdiff_t>& strides = io == fft_io::fft_io_in ? istrides : ostrides;
        return convert_vector_to<T>(strides);
    }
    const decltype(istrides)& get_strides(fft_io io) const
    {
        static_assert(std::is_same_v<decltype(istrides), decltype(ostrides)>);
        if(io != fft_io::fft_io_in && io != fft_io::fft_io_out)
            throw std::invalid_argument("invalid io passed to hipfftw_helper::get_strides");
        return io == fft_io::fft_io_in ? istrides : ostrides;
    }

    int get_batch_rank() const
    {
        return batch_rank;
    }

    // interfacing routines to be used only for valid and supported cases
    // (i.e., strictly positive values, batch_rank = 1, etc.)
    size_t get_distance(fft_io io) const
    {
        if(io != fft_io::fft_io_in && io != fft_io::fft_io_out)
            throw std::invalid_argument("invalid io passed to hipfftw_helper::get_distance");
        const auto& distances = io == fft_io::fft_io_in ? idist : odist;
        if(batch_rank != 1 || batch_rank < 0 || distances.size() != static_cast<size_t>(batch_rank))
            throw std::runtime_error("hipfftw_helper::get_distance: a single distance value cannot "
                                     "be queried for this configuration");
        const auto tmp = convert_vector_to<size_t>(distances);
        return tmp[0];
    }
    size_t get_nbatch() const
    {
        if(batch_rank != 1 || batch_rank < 0 || batches.size() != static_cast<size_t>(batch_rank))
            throw std::runtime_error("hipfftw_helper::get_nbatch: a single batch size cannot be "
                                     "queried for this configuration");
        const auto tmp = convert_vector_to<size_t>(batches);
        return tmp[0];
    }

    hipfftw_plan_creation_func get_plan_creation_function() const
    {
        if(!plan_bundle)
            return hipfftw_plan_creation_func::NONE;
        return plan_bundle->creation_func;
    }
    hipfftw_plan_t<prec> get_plan() const
    {
        if(!plan_bundle)
            return nullptr;
        return plan_bundle->plan;
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
                "ill-defined creation_options passed to hipfftw_helper::can_use_creation_options");
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
                               [=, this](const hipfftw_plan_creation_func& candidate) {
                                   return (creation_options & candidate)
                                          && can_use_creation_options(candidate);
                               });
        }
        // "creation_options" actually is an individual value in hipfftw_plan_creation_func_candidates
        switch(creation_options)
        {
        case hipfftw_plan_creation_func::PLAN_DFT_ND:
            [[fallthrough]];
        case hipfftw_plan_creation_func::PLAN_DFT:
        {
            // only unbatched cases (making distances irrelevant)
            if(batch_rank != 1 || batches.size() != 1 || batches[0] != 1)
                return false;
            // no empty lengths (~> nullptr lengths) for PLAN_DFT_ND
            if(creation_options == hipfftw_plan_creation_func::PLAN_DFT_ND && lengths.empty())
                return false;
            // only default strides (this check is relevant only for non-empty lengths)
            if(!lengths.empty())
            {
                if(istrides != default_strides(dft_kind, plan_placement, fft_io::fft_io_in, lengths)
                   || ostrides
                          != default_strides(dft_kind, plan_placement, fft_io::fft_io_out, lengths))
                    return false;
            }
            // lengths must be representable as int
            try
            {
                auto tmp = get_lengths_as<int>();
            }
            catch(const type_conversion_exception& e)
            {
                return false;
            }
            // rank is not passed as an argument but dictated by the called function
            // (must be 1, 2, or 3) if not using PLAN_DFT
            return creation_options == hipfftw_plan_creation_func::PLAN_DFT
                   || (rank >= 1 && rank <= 3);
        }
        case hipfftw_plan_creation_func::PLAN_MANY:
        {
            // batch_rank == 1 only
            if(batch_rank != 1 || batches.size() != 1 || idist.size() != 1 || odist.size() != 1)
                return false;
            // only strides that may be represented via inembed/onembed
            if(!ionembed)
                return false;
            // lengths, batches, and distances must be representable as int
            try
            {
                auto tmp = get_lengths_as<int>();
                tmp      = convert_vector_to<int>(batches);
                tmp      = convert_vector_to<int>(idist);
                tmp      = convert_vector_to<int>(odist);
            }
            catch(const type_conversion_exception& e)
            {
                return false;
            }
            return true;
        }
        case hipfftw_plan_creation_func::PLAN_GURU:
        {
            // anything goes provided it can be represented as regular integer(s)
            try
            {
                auto tmp = get_guru_dims<fftw_iodim>();
                tmp      = get_guru_howmany_dims<fftw_iodim>();
            }
            catch(const cannot_use_guru& e)
            {
                return false;
            }
            return true;
        }
        case hipfftw_plan_creation_func::PLAN_GURU64:
            // anything goes!
            return true;
        default:
            throw std::runtime_error("hipfftw_helper::can_use_creation_options: internal error "
                                     "encountered (unexpected value for creation_options)");
        }
        // unreachable
    }

    // checks validity of configuration parameters and whether creation can be
    // attempted via (any of) the given option(s)
    bool is_valid_for_creation_with(hipfftw_plan_creation_func creation_options) const
    {
        if(!hipfftw_creation_options_are_well_defined(creation_options))
            throw std::invalid_argument(
                "invalid creation_options passed to hipfftw_helper::is_valid_for_creation_with");

        auto ret = has_valid_rank(creation_options) && has_valid_lengths() && has_valid_sign()
                   && has_valid_flags() && has_valid_batch_rank(creation_options)
                   && has_valid_batches() && has_valid_strides(fft_io::fft_io_in, creation_options)
                   && has_valid_strides(fft_io::fft_io_out, creation_options)
                   && has_valid_distances(fft_io::fft_io_in)
                   && has_valid_distances(fft_io::fft_io_out)
                   && can_use_creation_options(creation_options);

        // If output data layout is entirely defined by non-negative values (hence not "unsupported"),
        // also check that the output data layout is not self-aliasing
        if(ret && vector_has_valid_values_as<ptrdiff_t>(lengths, rank, 1)
           && vector_has_valid_values_as<ptrdiff_t>(ostrides, rank, 1)
           && vector_has_valid_values_as<ptrdiff_t>(batches, batch_rank, 1)
           && vector_has_valid_values_as<ptrdiff_t>(odist, batch_rank, 1))
        {
            const size_t        gen_size = lengths.size() + batches.size();
            std::vector<size_t> gen_len(gen_size), gen_strides(gen_size);
            for(auto dim = lengths.size(); dim-- > 0;)
            {
                gen_len[dim]
                    = dft_kind == fft_transform_type_real_forward && dim == lengths.size() - 1
                          ? lengths[dim] / 2 + 1
                          : lengths[dim];
                gen_strides[dim] = ostrides[dim];
            }
            for(auto batch_dim = batches.size(); batch_dim-- > 0;)
            {
                gen_len[lengths.size() + batch_dim]     = batches[batch_dim];
                gen_strides[lengths.size() + batch_dim] = odist[batch_dim];
            }
            ret = array_valid(gen_len, gen_strides);
        }
        return ret;
    }
    bool is_valid_for_creation() const
    {
        return is_valid_for_creation_with(hipfftw_plan_creation_func::ANY);
    }
    // check expected support
    bool has_unsupported_args() const
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
            throw std::invalid_argument(
                "invalid creation_option for hipfftw_helper::can_create_plan_with");

        if(!is_valid_for_creation_with(creation_options))
            return false;
        if(has_unsupported_args())
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
            throw std::runtime_error("hipfftw_helper::token: unknown type of transform");
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
        ret << "_" << hipfftw_trait<prec>::prec_label;
        ret << (plan_placement == fft_placement_inplace ? "_ip" : "_op");
        if(batch_rank != 1)
            ret << "_batch_rank" << (batch_rank < 0 ? "_negative_" : "_") << std::abs(batch_rank);
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

    // NOTE: from_token only supports configurations that fft_params itself supports
    void from_token(const std::string& token)
    {
        fft_params tmp;
        tmp.from_token(token);
        if(tmp.precision != prec)
            throw std::invalid_argument("hipfftw_helper::from_token: precision mismatch between "
                                        "token and (specialization of) hipfftw_helper object");
        if(tmp.istride.size() != tmp.ostride.size() || tmp.istride.size() != tmp.length.size())
            throw std::runtime_error(
                "unexpected mismatch of vector sizes in hipfftw_helper::from_token");
        dft_kind   = tmp.transform_type;
        rank       = tmp.length.size();
        batch_rank = 1;
        lengths.resize(rank);
        istrides.resize(rank);
        ostrides.resize(rank);
        for(auto dim = 0; dim < rank; dim++)
        {
            lengths[dim]  = tmp.length[dim];
            istrides[dim] = tmp.istride[dim];
            ostrides[dim] = tmp.ostride[dim];
        }

        batches        = std::vector<ptrdiff_t>(1, tmp.nbatch);
        idist          = std::vector<ptrdiff_t>(1, tmp.idist);
        odist          = std::vector<ptrdiff_t>(1, tmp.odist);
        plan_placement = tmp.placement;
        sign           = is_fwd(dft_kind) ? FFTW_FORWARD : FFTW_BACKWARD;
        flags          = FFTW_ESTIMATE;

        const std::string flags_label = "flags";
        auto              pos         = token.find(flags_label);
        if(pos != std::string::npos)
        {
            pos += flags_label.size() + 1;
            flags = std::stoull(token.substr(pos, token.find("_", pos)));
        }
        reset_io_nembed_from_strides();
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

    size_t get_num_elements_in(fft_io in_or_out) const
    {
        if(in_or_out != fft_io_in && in_or_out != fft_io_out)
            throw std::invalid_argument(
                "invalid in_or_out passed to hipfftw_helper::get_num_elements_in");
        if(!has_valid_rank() || !has_valid_lengths() || !has_valid_batch_rank()
           || !has_valid_batches())
            throw num_elements_calc_exception("hipfftw_helper::get_num_elements_in requires valid "
                                              "rank, batch_rank, lengths, and batches");
        const auto& strides   = in_or_out == fft_io::fft_io_in ? istrides : ostrides;
        const auto& distances = in_or_out == fft_io::fft_io_in ? idist : odist;
        if(!vector_has_valid_values_as<ptrdiff_t>(strides, rank, 0)
           || !vector_has_valid_values_as<ptrdiff_t>(distances, batch_rank, 0))
            throw num_elements_calc_exception(
                "hipfftw_helper::get_num_elements_in assumes non-negative strides and distances.");
        size_t elem_count = 1;
        for(auto len_dim = lengths.size(); len_dim-- > 0;)
        {
            if(((dft_kind == fft_transform_type_real_forward && in_or_out == fft_io::fft_io_out)
                || (dft_kind == fft_transform_type_real_inverse && in_or_out == fft_io::fft_io_in))
               && len_dim == lengths.size() - 1)
            {
                elem_count += (lengths[len_dim] / 2) * strides[len_dim];
            }
            else
                elem_count += (lengths[len_dim] - 1) * strides[len_dim];
        }
        for(auto batch_dim = batches.size(); batch_dim-- > 0;)
            elem_count += (batches[batch_dim] - 1) * distances[batch_dim];
        return elem_count;
    }

    size_t get_data_byte_size(fft_io in_or_out) const
    {
        if(in_or_out != fft_io_in && in_or_out != fft_io_out)
            throw std::invalid_argument(
                "invalid in_or_out passed to hipfftw_helper::get_data_byte_size");
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
    bool is_using_default_strides() const
    {
        return istrides == default_strides(dft_kind, plan_placement, fft_io::fft_io_in, lengths)
               && ostrides
                      == default_strides(dft_kind, plan_placement, fft_io::fft_io_out, lengths);
    }
    bool is_using_default_distances() const
    {
        return idist
                   == default_distances(
                       dft_kind, plan_placement, fft_io::fft_io_in, lengths, batches)
               && odist
                      == default_distances(
                          dft_kind, plan_placement, fft_io::fft_io_out, lengths, batches);
    }

    operator fft_params() const
    {
        fft_params ret;
        ret.length         = convert_vector_to<decltype(ret.length)::value_type>(lengths);
        ret.precision      = prec;
        ret.placement      = plan_placement;
        ret.transform_type = dft_kind;
        if(batch_rank != 1 || batches[0] < 0)
            throw std::runtime_error(
                "Conversion to fft_params impossible for batch_rank != 1 or negative batch sizes");
        ret.nbatch        = batches[0];
        ret.run_callbacks = false;
        ret.scale_factor  = 1.0;
        ret.istride       = convert_vector_to<decltype(ret.istride)::value_type>(istrides);
        ret.ostride       = convert_vector_to<decltype(ret.ostride)::value_type>(ostrides);
        if(idist[0] < 0 || odist[0] < 0)
            throw std::runtime_error("Conversion to fft_params impossible for negative distances");
        ret.idist = idist[0];
        ret.odist = odist[0];
        ret.validate(); // sets itype, otype, isize, osize, etc. from the above
        // other ret's members should be irrelevant/fine with default values
        return ret;
    }

    fft_params make_params_for_reference_cpu() const
    {
        fft_params converted = *this;
        return converted.make_params_for_reference_cpu();
    }

    // public ad-hoc exceptions, specific to hipfftw_helper
    struct type_conversion_exception : std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };

    struct num_elements_calc_exception : std::runtime_error
    {
        using std::runtime_error::runtime_error;
    };
    friend void create_hipfftw_plan_from_token_using_temp_io(const std::string&, bool);
};

// trigger a plan creation via hipfftw, without user-defined I/O
static void create_hipfftw_plan_from_token_using_temp_io(const std::string& token, bool verbose)
{
    if(token.find(hipfftw_trait<fft_precision_single>::prec_label) != std::string::npos)
        hipfftw_helper<fft_precision_single>::create_temp_plan_from_token(token);
    else if(token.find(hipfftw_trait<fft_precision_double>::prec_label) != std::string::npos)
        hipfftw_helper<fft_precision_double>::create_temp_plan_from_token(token);
    else if(verbose)
        std::cout << "Unknown/unexpected precision from " << token << std::endl;
}

#endif
