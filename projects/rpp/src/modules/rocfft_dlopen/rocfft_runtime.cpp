/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "rocfft_runtime.hpp"

#include <mutex>
#include <string>

#include "load_function.hpp"

namespace rpp {
namespace rocfft_dlopen {

namespace {

// Populated on the first initialize(); read-only afterwards.
bool g_available = false;
std::string& failure_reason() {
    static std::string reason;
    return reason;
}

#ifdef _WIN32
constexpr const char* kRocfftLibraryName = "rocfft.dll";
#else
// Pin the SONAME rather than the unversioned development symlink: the runtime package
// ships only the former, and a major-version bump should fail to load rather than bind
// against an incompatible ABI.
constexpr const char* kRocfftLibraryName = "librocfft.so.0";
#endif

}  // namespace

fp_rocfft_setup g_rocfft_setup = nullptr;
fp_rocfft_cleanup g_rocfft_cleanup = nullptr;
fp_rocfft_plan_create g_rocfft_plan_create = nullptr;
fp_rocfft_plan_destroy g_rocfft_plan_destroy = nullptr;
fp_rocfft_plan_get_work_buffer_size g_rocfft_plan_get_work_buffer_size = nullptr;
fp_rocfft_plan_description_create g_rocfft_plan_description_create = nullptr;
fp_rocfft_plan_description_destroy g_rocfft_plan_description_destroy = nullptr;
fp_rocfft_plan_description_set_data_layout g_rocfft_plan_description_set_data_layout = nullptr;
fp_rocfft_execution_info_create g_rocfft_execution_info_create = nullptr;
fp_rocfft_execution_info_destroy g_rocfft_execution_info_destroy = nullptr;
fp_rocfft_execution_info_set_stream g_rocfft_execution_info_set_stream = nullptr;
fp_rocfft_execution_info_set_work_buffer g_rocfft_execution_info_set_work_buffer = nullptr;
fp_rocfft_execute g_rocfft_execute = nullptr;

namespace {

// Resolve every entry point RPP uses. Partial resolution is treated as failure so that a
// mismatched librocfft cannot leave a null pointer to be called later.
bool load_rocfft() {
#ifdef _WIN32
    void* handle = static_cast<void*>(LoadLibraryA(kRocfftLibraryName));
    if (!handle) {
        failure_reason() = std::string("could not load ") + kRocfftLibraryName;
        return false;
    }
#else
    // RTLD_LOCAL keeps rocFFT's symbols out of the global namespace; RTLD_NOW surfaces a
    // broken library here rather than at the first call. The handle is deliberately never
    // dlclose()d: rocFFT holds HIP state whose teardown must not race RPP's own.
    void* handle = dlopen(kRocfftLibraryName, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        const char* err = dlerror();
        failure_reason() = std::string("could not load ") + kRocfftLibraryName + ": " +
                           (err ? err : "unknown error");
        return false;
    }
    (void)dlerror();  // clear error state before the symbol lookups
#endif

// Stringifying/pasting the argument suppresses macro expansion, so the rocFFT spellings
// below resolve to the symbol name and the g_-prefixed pointer rather than to the
// call-site redirection macros in rocfft_runtime.hpp.
#define RPP_BIND_ROCFFT_SYMBOL(sym)                                          \
    do {                                                                     \
        if (!load_function(handle, #sym, g_##sym)) {                         \
            failure_reason() = std::string("loaded ") + kRocfftLibraryName + \
                               " but could not resolve symbol " #sym;        \
            return false;                                                    \
        }                                                                    \
    } while (0)

    RPP_BIND_ROCFFT_SYMBOL(rocfft_setup);
    RPP_BIND_ROCFFT_SYMBOL(rocfft_cleanup);
    RPP_BIND_ROCFFT_SYMBOL(rocfft_plan_create);
    RPP_BIND_ROCFFT_SYMBOL(rocfft_plan_destroy);
    RPP_BIND_ROCFFT_SYMBOL(rocfft_plan_get_work_buffer_size);
    RPP_BIND_ROCFFT_SYMBOL(rocfft_plan_description_create);
    RPP_BIND_ROCFFT_SYMBOL(rocfft_plan_description_destroy);
    RPP_BIND_ROCFFT_SYMBOL(rocfft_plan_description_set_data_layout);
    RPP_BIND_ROCFFT_SYMBOL(rocfft_execution_info_create);
    RPP_BIND_ROCFFT_SYMBOL(rocfft_execution_info_destroy);
    RPP_BIND_ROCFFT_SYMBOL(rocfft_execution_info_set_stream);
    RPP_BIND_ROCFFT_SYMBOL(rocfft_execution_info_set_work_buffer);
    RPP_BIND_ROCFFT_SYMBOL(rocfft_execute);

#undef RPP_BIND_ROCFFT_SYMBOL

    return true;
}

}  // namespace

bool initialize() {
    static std::once_flag once;
    std::call_once(once, []() { g_available = load_rocfft(); });
    return g_available;
}

const char* error_message() {
    return failure_reason().c_str();
}

}  // namespace rocfft_dlopen
}  // namespace rpp
