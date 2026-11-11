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

/*! \file rocfft_runtime.hpp
 *  \brief Runtime (dlopen) binding to rocFFT for the RPP HIP audio path.
 *
 *  RPP needs rocFFT only for the spectrogram kernel. Linking it would make rocFFT a
 *  build-time dependency of RPP, which in a staged ROCm build forces RPP's build stage to
 *  wait on the FFT stage. Instead the library is resolved at runtime and the few
 *  declarations RPP needs are reproduced below, so neither the rocFFT headers nor
 *  librocfft are required to compile or link RPP.
 *
 *  This changes only *how* rocFFT is bound, not whether it is used: every transform is still
 *  performed by the real rocFFT at the same speed, and librocfft must be installed to run
 *  audio on the HIP backend. Nothing here emulates rocFFT and there is no software fallback -
 *  when the library cannot be loaded the HIP spectrogram kernel reports a hard failure.
 *
 *  This mirrors the pattern hipSOLVER uses to make rocSPARSE optional, see
 *  projects/hipsolver/library/src/amd_detail/dlopen/.
 *
 *  Call sites are unchanged: the macros below redirect the usual rocFFT spellings to the
 *  resolved function pointers.
 *
 *  What RPP copies from rocFFT is checked against the real headers by
 *  utilities/test_suite/HIP/rocfft_abi_guard.cpp: the enumerator values and opaque handles in
 *  rocfft_vendored_types.h, and the parameter lists in rocfft_vendored_prototypes.h.
 */

#ifndef RPP_ROCFFT_DLOPEN_ROCFFT_RUNTIME_HPP
#define RPP_ROCFFT_DLOPEN_ROCFFT_RUNTIME_HPP

#include <stddef.h>

// The enums and opaque handles RPP reproduces from <rocfft/rocfft.h>. Kept in a separate,
// include-free header so the ABI guard can pull in these exact declarations next to the real
// ones; see rocfft_vendored_types.h for the drift warning.
#include "rocfft_vendored_types.h"

namespace rpp {
namespace rocfft_dlopen {

// The entry points RPP calls, as function pointer types built from the vendored types above.
// Included rather than written out here so that the ABI guard can rebuild these same
// signatures from the real rocFFT types and compare; see rocfft_vendored_prototypes.h.
#include "rocfft_vendored_prototypes.h"

// Bound by rocfft_runtime.cpp on first use.

extern fp_rocfft_setup g_rocfft_setup;
extern fp_rocfft_cleanup g_rocfft_cleanup;
extern fp_rocfft_plan_create g_rocfft_plan_create;
extern fp_rocfft_plan_destroy g_rocfft_plan_destroy;
extern fp_rocfft_plan_get_work_buffer_size g_rocfft_plan_get_work_buffer_size;
extern fp_rocfft_plan_description_create g_rocfft_plan_description_create;
extern fp_rocfft_plan_description_destroy g_rocfft_plan_description_destroy;
extern fp_rocfft_plan_description_set_data_layout g_rocfft_plan_description_set_data_layout;
extern fp_rocfft_execution_info_create g_rocfft_execution_info_create;
extern fp_rocfft_execution_info_destroy g_rocfft_execution_info_destroy;
extern fp_rocfft_execution_info_set_stream g_rocfft_execution_info_set_stream;
extern fp_rocfft_execution_info_set_work_buffer g_rocfft_execution_info_set_work_buffer;
extern fp_rocfft_execute g_rocfft_execute;

}  // namespace rocfft_dlopen
}  // namespace rpp

#define rocfft_setup ::rpp::rocfft_dlopen::g_rocfft_setup
#define rocfft_cleanup ::rpp::rocfft_dlopen::g_rocfft_cleanup
#define rocfft_plan_create ::rpp::rocfft_dlopen::g_rocfft_plan_create
#define rocfft_plan_destroy ::rpp::rocfft_dlopen::g_rocfft_plan_destroy
#define rocfft_plan_get_work_buffer_size ::rpp::rocfft_dlopen::g_rocfft_plan_get_work_buffer_size
#define rocfft_plan_description_create ::rpp::rocfft_dlopen::g_rocfft_plan_description_create
#define rocfft_plan_description_destroy ::rpp::rocfft_dlopen::g_rocfft_plan_description_destroy
#define rocfft_plan_description_set_data_layout \
    ::rpp::rocfft_dlopen::g_rocfft_plan_description_set_data_layout
#define rocfft_execution_info_create ::rpp::rocfft_dlopen::g_rocfft_execution_info_create
#define rocfft_execution_info_destroy ::rpp::rocfft_dlopen::g_rocfft_execution_info_destroy
#define rocfft_execution_info_set_stream ::rpp::rocfft_dlopen::g_rocfft_execution_info_set_stream
#define rocfft_execution_info_set_work_buffer \
    ::rpp::rocfft_dlopen::g_rocfft_execution_info_set_work_buffer
#define rocfft_execute ::rpp::rocfft_dlopen::g_rocfft_execute

namespace rpp {
namespace rocfft_dlopen {

/*! \brief Resolve librocfft once per process.
 *
 *  Idempotent and thread-safe; only the first call does work.
 *
 *  \return true if rocFFT is usable.
 */
bool initialize();

/*! \brief Why rocFFT is unavailable.
 *  \return A diagnostic naming the library and the failure, or an empty string when rocFFT
 *          is available. Never nullptr.
 */
const char* error_message();

}  // namespace rocfft_dlopen
}  // namespace rpp

#endif  // RPP_ROCFFT_DLOPEN_ROCFFT_RUNTIME_HPP
