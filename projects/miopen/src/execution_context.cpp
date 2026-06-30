/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2017 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
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

#include <miopen/execution_context.hpp>

#include <miopen/datatype.hpp>
#include <miopen/env.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/hip_build_utils.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/version.h>

MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_GCN_ASM_KERNELS)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_HIP_KERNELS)
MIOPEN_DECLARE_ENV_VAR_UINT64(MIOPEN_DEBUG_AMD_ROCM_METADATA_ENFORCE)
MIOPEN_DECLARE_ENV_VAR_BOOL(MIOPEN_DEBUG_AMD_ROCM_METADATA_PREFER_OLDER)

namespace miopen {
namespace debug {

// NOLINTNEXTLINE (cppcoreguidelines-avoid-non-const-global-variables)
MIOPEN_EXPORT bool IsWarmupOngoing = false;

} // namespace debug
} // namespace miopen

static std::ostream& operator<<(std::ostream& os, const rocm_meta_version& rmv)
{
    switch(rmv.getValue())
    {
    case rocm_meta_version::Unknown: return os << "Unknown";
    case rocm_meta_version::AMDHSA_COv2: return os << "AMDHSA_COv2";
    case rocm_meta_version::AMDHSA_COv2_COv3: return os << "AMDHSA_COv2_COv3";
    case rocm_meta_version::AMDHSA_COv3: return os << "AMDHSA_COv3";
    default: break;
    }
    return os << "<Error>";
}

/// This is intended to use only in Asm Solvers which support both CO v2 and CO v3.
/// It says which code object format shall be selected during the build process.
///
/// If ROCm supports only v2 or v3, the answer is trivial. When Solver supports
/// single CO version, the logic is trivial as well.
///
/// However, when both ROCm and Solver are able to support both code object formats,
/// these is no objective criterion for making a decision. The following behavior
/// is implemented:
/// * By default, the newer format is used (CO v3).
/// * If MIOPEN_DEBUG_AMD_ROCM_METADATA_PREFER_OLDER is set to 1, then
///   the behavior is reversed and CO v2 is selected.
///
/// \todo Dismiss MIOPEN_DEBUG_AMD_ROCM_METADATA_PREFER_OLDER (and, possibly,
/// rocm_meta_version::AMDHSA_COv2_COv3) as soon as MIOpen drops support for the
/// ROCm runtimes that can load and run both v2 and v3 Code Objects.
///
/// \todo Move this out of the rocm_meta_version class.
bool rocm_meta_version::UseV3() const
{
    if(val == AMDHSA_COv2_COv3)
        return !miopen::env::enabled(MIOPEN_DEBUG_AMD_ROCM_METADATA_PREFER_OLDER);
    return (val == AMDHSA_COv3);
}

namespace miopen {

static rocm_meta_version AmdRocmMetadataVersionGetEnv()
{
    rocm_meta_version val{env::value(MIOPEN_DEBUG_AMD_ROCM_METADATA_ENFORCE)};
    if(!val.IsValid())
    {
        MIOPEN_LOG_W("Incorrect MIOPEN_DEBUG_AMD_ROCM_ENFORCE_MDVERSION = " << val.getValue()
                                                                            << ", using default.");
        val = rocm_meta_version::Unknown;
    }
    return val;
}

static rocm_meta_version AmdRocmMetadataVersionDetect(const miopen::ExecutionContext& context)
{
    rocm_meta_version rmv = AmdRocmMetadataVersionGetEnv();
    if(rmv.IsUnknown())
    {
        (void)context;
        rmv = rocm_meta_version::Default;
    }
    MIOPEN_LOG_NQI(
        "ROCm MD version "
        << rmv
        << ", HIP version " MIOPEN_STRINGIZE(HIP_PACKAGE_VERSION_MAJOR) "." MIOPEN_STRINGIZE(
               HIP_PACKAGE_VERSION_MINOR) "." MIOPEN_STRINGIZE(HIP_PACKAGE_VERSION_PATCH)
        << ", MIOpen version " MIOPEN_STRINGIZE(MIOPEN_VERSION_MAJOR) "." MIOPEN_STRINGIZE(
               MIOPEN_VERSION_MINOR) "." MIOPEN_STRINGIZE(MIOPEN_VERSION_PATCH) "." MIOPEN_STRINGIZE(MIOPEN_VERSION_TWEAK));
    return rmv;
}

static bool DetectAmdRocmMetadata(miopen::ExecutionContext& context)
{
    static const bool ret_bool = true;
    // cppcheck-suppress knownConditionTrueFalse
    if(ret_bool)
    {
        static const rocm_meta_version ret_rmv = AmdRocmMetadataVersionDetect(context);
        context.rmv                            = ret_rmv;
    }
    return ret_bool;
}

bool IsHipKernelsEnabled()
{
#if MIOPEN_USE_HIP_KERNELS
    return !env::disabled(MIOPEN_DEBUG_HIP_KERNELS);
#else
    return env::enabled(MIOPEN_DEBUG_HIP_KERNELS);
#endif
}

void ExecutionContext::DetectRocm()
{
    use_asm_kernels = false;
    use_hip_kernels = IsHipKernelsEnabled();
    rmv             = rocm_meta_version::Default;
    if(DetectAmdRocmMetadata(*this))
    {
        use_asm_kernels = !env::disabled(MIOPEN_DEBUG_GCN_ASM_KERNELS) && ValidateGcnAssembler();
    }
}

} // namespace miopen
