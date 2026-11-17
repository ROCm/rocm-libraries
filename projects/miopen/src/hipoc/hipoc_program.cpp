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
#include <miopen/config.h>

#include <miopen/errors.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/hip_build_utils.hpp>
#include <miopen/hipoc_program.hpp>
#include <miopen/kernel.hpp>
#include <miopen/kernel_warnings.hpp>
#include <miopen/logger.hpp>
#include <miopen/mlir_build.hpp>
#include <miopen/stringutils.hpp>
#include <miopen/target_properties.hpp>
#include <miopen/temp_file.hpp>
#include <miopen/write_file.hpp>
#include <miopen/env.hpp>
#include <miopen/comgr.hpp>

#include <cstring>
#include <mutex>
#include <optional>
#include <sstream>

#if defined(__linux__)
#include <unistd.h>
#endif

/// 0 or undef or wrong - auto-detect
MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_DEVICE_ARCH)

#if MIOPEN_USE_COMGR
#define MIOPEN_WORKAROUND_ROCM_COMPILER_SUPPORT_ISSUE_27 1
#endif

// Temporarily disable warnings as errors for kernel builds to see real breaks with compiler changes
#define MIOPEN_WORKAROUND_COMPILER_CHANGE 1

namespace miopen {

static hipModulePtr CreateModule(const fs::path& hsaco_file)
{
    hipModule_t raw_m;
    auto status = hipModuleLoad(&raw_m, hsaco_file.string().c_str());
    hipModulePtr m{raw_m};
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Failed creating module from file " + hsaco_file);
    return m;
}

template <typename T> /// intended for std::string and std::vector<char>
hipModulePtr CreateModuleInMem(const T& blob)
{
    hipModule_t raw_m;
    auto status = hipModuleLoadData(&raw_m, reinterpret_cast<const void*>(blob.data()));
    hipModulePtr m{raw_m};
    if(status != hipSuccess)
        MIOPEN_THROW_HIP_STATUS(status, "Failed loading module");
    return m;
}

HIPOCProgramImpl::HIPOCProgramImpl(const fs::path& program_name, const fs::path& filespec)
    : program(program_name), hsaco_file(filespec)
{
    module = CreateModule(hsaco_file);
}

HIPOCProgramImpl::HIPOCProgramImpl(const fs::path& program_name, const std::vector<char>& blob)
    : program(program_name), binary(blob) // Store the binary data to prevent use-after-free
{
    const auto& arch = env::value(MIOPEN_DEVICE_ARCH);
    if(!arch.empty())
        return;
    module = CreateModuleInMem(binary); // Use stored binary instead of parameter
}

HIPOCProgramImpl::HIPOCProgramImpl(const fs::path& program_name, const std::vector<uint8_t>& blob)
    : program(program_name),
      binary(blob.begin(), blob.end()) // Store the binary data to prevent use-after-free
{
    const auto& arch = env::value(MIOPEN_DEVICE_ARCH);
    if(!arch.empty())
        return;
    module = CreateModuleInMem(binary); // Use stored binary instead of parameter
}

HIPOCProgramImpl::HIPOCProgramImpl(const fs::path& program_name,
                                   std::string params,
                                   const TargetProperties& target_,
                                   const std::string& kernel_src)
    : program(program_name), target(target_)
{
    BuildCodeObject(params, kernel_src);
    if(!binary.empty())
    {
        module = CreateModuleInMem(binary);
    }
    else
    {
        const auto& arch = env::value(MIOPEN_DEVICE_ARCH);
        if(arch.empty())
        {
            module = CreateModule(hsaco_file);
        }
    }
}

#if !MIOPEN_USE_COMGR
void HIPOCProgramImpl::BuildCodeObjectInFile(std::string& params,
                                             std::string_view src,
                                             const fs::path& filename)
{
    dir.emplace(filename.filename().string());
    hsaco_file = make_object_file_name(dir.value() / filename);

    if(filename.extension() == dynamic_library_postfix) // ".so" or ".dll"
    {
        WriteFile(src, hsaco_file);
    }
    else if(filename.extension() == ".s")
    {
        const auto assembled = AmdgcnAssemble(src, params, target);
        WriteFile(assembled, hsaco_file);
    }
    else if(filename.extension() == ".cpp")
    {
        hsaco_file = HipBuild(dir.value(), filename, src, params, target);
    }
#if MIOPEN_USE_MLIR
    else if(filename.extension() == ".mlir")
    {
        std::vector<char> buffer;
        MiirGenBin(params, buffer);
        WriteFile(buffer, hsaco_file);
    }
#endif
    else
    {
        MIOPEN_THROW("Unsupported file extension: " + filename.extension().string());
    }
    if(!fs::exists(hsaco_file))
        MIOPEN_THROW("Cant find file: " + hsaco_file);
}

#else // MIOPEN_USE_COMGR
void HIPOCProgramImpl::BuildCodeObjectInMemory(const std::string& params,
                                               const std::string_view src,
                                               const fs::path& filename)
{
    if(filename.extension() == dynamic_library_postfix) // ".so" or ".dll"
    {
        binary.resize(src.size());
        std::memcpy(&binary[0], src.data(), src.size());
    }
    else
    {
#if MIOPEN_WORKAROUND_ROCM_COMPILER_SUPPORT_ISSUE_27
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
#endif
        if(filename.extension() == ".cpp")
        {
            hiprtc::BuildHip(filename.string(), src, params, target, binary);
        }
        else if(filename.extension() == ".s")
        {
            comgr::BuildAsm(filename.string(), src, params, target, binary);
        }
#if MIOPEN_USE_MLIR
        else if(filename.extension() == ".mlir")
        {
            MiirGenBin(params, binary);
        }
#endif
        else
        {
            MIOPEN_THROW("Unsupported file extension: " + filename.extension().string());
        }
    }
    if(binary.empty())
        MIOPEN_THROW("Code object build failed. Source: " + filename);
}
#endif // MIOPEN_USE_COMGR

void HIPOCProgramImpl::BuildCodeObject(std::string params, const std::string& kernel_src)
{
    const auto src = [&]() -> std::string_view {
        if(program.extension() == ".mlir")
            return {}; // MLIR solutions do not use source code.
        if(!kernel_src.empty())
            return kernel_src;
        return GetKernelSrc(program);
    }();

#if MIOPEN_BUILD_DEV && !MIOPEN_WORKAROUND_COMPILER_CHANGE
    if(program.extension() == ".cpp")
    {
        params += " -Werror" + HipKernelWarningsString();
    }
#else
    if(program.extension() == ".cpp")
        params += " -Wno-everything";
#endif

#if MIOPEN_USE_COMGR /// \todo Refactor when functionality stabilize.
    BuildCodeObjectInMemory(params, src, program);
#else
    BuildCodeObjectInFile(params, src, program);
#endif
}

HIPOCProgram::HIPOCProgram() {}
HIPOCProgram::HIPOCProgram(const fs::path& program_name,
                           std::string params,
                           const TargetProperties& target,
                           const std::string& kernel_src)
    : impl(std::make_shared<HIPOCProgramImpl>(program_name, params, target, kernel_src))
{
}

HIPOCProgram::HIPOCProgram(const fs::path& program_name, const fs::path& hsaco)
    : impl(std::make_shared<HIPOCProgramImpl>(program_name, hsaco))
{
}

HIPOCProgram::HIPOCProgram(const fs::path& program_name, const std::vector<char>& hsaco)
    : impl(std::make_shared<HIPOCProgramImpl>(program_name, hsaco))
{
}

HIPOCProgram::HIPOCProgram(const fs::path& program_name, const std::vector<uint8_t>& hsaco)
    : impl(std::make_shared<HIPOCProgramImpl>(program_name, hsaco))
{
}

hipModule_t HIPOCProgram::GetModule() const { return impl->module.get(); }

fs::path HIPOCProgram::GetCodeObjectPathname() const
{
    if(!impl->hsaco_file.empty())
    {
        return impl->hsaco_file;
    }
    else
    {
        MIOPEN_THROW(miopenStatusInternalError, "Empty code object path.");
    }
}

const std::vector<char>& HIPOCProgram::GetCodeObjectBlob() const { return impl->binary; }

void HIPOCProgram::FreeCodeObjectFileStorage()
{
    if(impl->dir.has_value())
    {
        impl->dir.reset();
        impl->hsaco_file.clear();
    }
}

bool HIPOCProgram::IsCodeObjectInMemory() const { return !impl->binary.empty(); };

bool HIPOCProgram::IsCodeObjectInFile() const { return !impl->hsaco_file.empty(); }

bool HIPOCProgram::IsCodeObjectInTempFile() const { return impl->dir.has_value(); }

void HIPOCProgram::AttachBinary(std::vector<char> binary) { impl->binary = std::move(binary); }

void HIPOCProgram::AttachBinary(fs::path binary)
{
    if(impl->hsaco_file != binary)
        impl->dir.reset();
    impl->hsaco_file = std::move(binary);
}

} // namespace miopen
