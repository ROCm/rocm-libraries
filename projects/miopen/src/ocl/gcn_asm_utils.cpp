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
#include <miopen/env.hpp>
#include <miopen/gcn_asm_utils.hpp>
#include <miopen/config.h>

#if MIOPEN_USE_COMGR

bool ValidateGcnAssembler() { return true; }

#else // !MIOPEN_USE_COMGR

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <miopen/filesystem.hpp>
#include <miopen/errors.hpp>
#include <miopen/manage_ptr.hpp>
#include <miopen/write_file.hpp>
#include <miopen/kernel.hpp>
#include <miopen/logger.hpp>
#include <miopen/exec_utils.hpp>
#include <miopen/temp_file.hpp>
#include <sstream>

#ifdef __linux__
#include <paths.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#endif // __linux__

#include <boost/filesystem/operations.hpp>
namespace fs = miopen::fs;

/// SWDEV-233338: hip-clang reports unknown target instead of amdgpu.
/// \todo Try to assemble AMD GCN source?
#define WORKAROUND_SWDEV_233338 1

MIOPEN_DECLARE_ENV_VAR_STR(MIOPEN_EXPERIMENTAL_GCN_ASM_PATH)

static const char option_no_co_v3[] = "-mno-code-object-v3";

static bool GcnAssemblerHasBug34765();
static bool GcnAssemblerSupportsNoCOv3();
static std::string CleanupPath(const char* p);

std::string GetGcnAssemblerPathImpl()
{
    const auto& asm_path_env_p = miopen::env::value(MIOPEN_EXPERIMENTAL_GCN_ASM_PATH);
    if(!asm_path_env_p.empty())
    {
        return CleanupPath(asm_path_env_p.c_str());
    }
#ifdef MIOPEN_AMDGCN_ASSEMBLER // string literal generated by CMake
    return CleanupPath(MIOPEN_AMDGCN_ASSEMBLER);
#else
    return "";
#endif
}

std::string GetGcnAssemblerPath()
{
    static const auto result = GetGcnAssemblerPathImpl();
    return result;
}

bool ValidateGcnAssemblerImpl()
{
    const auto path = GetGcnAssemblerPath();
    if(path.empty())
    {
        MIOPEN_LOG_NQE("Path to assembler is not provided. Expect performance degradation.");
        return false;
    }
    if(!std::ifstream(path).good())
    {
        MIOPEN_LOG_NQE("Wrong path to assembler: '" << path
                                                    << "'. Expect performance degradation.");
        return false;
    }

    std::stringstream clang_stdout;
    MIOPEN_LOG_NQI2("Running: " << '\'' << path << " --version" << '\'');
    auto clang_rc = miopen::exec::Run(path + " --version", nullptr, &clang_stdout);

    if(clang_rc != 0)
    {
        return false;
    }

    std::string clang_result_line;
    std::getline(clang_stdout, clang_result_line);
    MIOPEN_LOG_NQI2(clang_result_line);

    if(clang_result_line.find("clang") != std::string::npos)
    {
        while(!clang_stdout.eof())
        {
            std::getline(clang_stdout, clang_result_line);
            MIOPEN_LOG_NQI2(clang_result_line);
            if(clang_result_line.find("Target: ") != std::string::npos &&
               clang_result_line.find("amdgcn") != std::string::npos)
                return true;
        }
#if WORKAROUND_SWDEV_233338
        return true;
#endif
    }
    MIOPEN_LOG_NQE("Specified assembler does not support AMDGPU. Expect performance degradation.");
    return false;
}

bool ValidateGcnAssembler()
{
    static const bool result = ValidateGcnAssemblerImpl();
    return result;
}

static std::string CleanupPath(const char* p)
{
    std::string path(p);
    static const char bad[] = "!#$*;<>?@\\^`{|}";
    for(char* c = &path[0]; c < (&path[0] + path.length()); ++c)
    {
        if(std::iscntrl(*c) != 0)
        {
            *c = '_';
            continue;
        }
        for(const char* b = &bad[0]; b < (&bad[0] + sizeof(bad) - 1); ++b)
        {
            if(*b == *c)
            {
                *c = '_';
                break;
            }
        }
    }
    return path;
}

/*
 * Temporary function which emulates online assembly feature of OpenCL-on-ROCm being developed.
 * Not intended to be used in production code, so error handling is very straghtforward,
 * just catch whatever possible and throw an exception.
 */
std::string AmdgcnAssemble(std::string_view source,
                           std::string_view params,
                           const miopen::TargetProperties& target)
{
    miopen::TempFile outfile("assembly");

    std::ostringstream options;
    options << " -x assembler -target amdgcn--amdhsa";
#if WORKAROUND_ISSUE_3001
    if(target.Xnack() && !*target.Xnack())
        options << " -mno-xnack";
#endif
    /// \todo Hacky way to find out which CO version we need to assemble for.
    if(params.find("ROCM_METADATA_VERSION=5", 0) == std::string::npos) // Assume that !COv3 == COv2.
        if(GcnAssemblerSupportsNoCOv3()) // If assembling for COv2, then disable COv3.
            options << ' ' << option_no_co_v3;

    options << ' ' << params;
    if(GcnAssemblerHasBug34765())
        GenerateClangDefsym(options, "WORKAROUND_BUG_34765", 1);

    options << " - -o " << outfile.Path();
    MIOPEN_LOG_I2("'" << options.str() << "'");

    std::istringstream clang_stdin(source.data());
    const auto clang_path = GetGcnAssemblerPath();
    const auto clang_rc =
        miopen::exec::Run(clang_path + " " + options.str(), &clang_stdin, nullptr);
    if(clang_rc != 0)
    {
        MIOPEN_LOG_W(options.str());
        MIOPEN_THROW("Assembly error(" + std::to_string(clang_rc) + ")");
    }

    std::string out;
    std::ifstream file(outfile.Path(), std::ios::binary | std::ios::ate);
    bool outfile_read_failed = false;
    do
    {
        const auto size = file.tellg();
        if(size == -1)
        {
            outfile_read_failed = true;
            break;
        }
        out.resize(size, '\0');
        file.seekg(std::ios::beg);
        if(file.fail())
        {
            outfile_read_failed = true;
            break;
        }
        if(file.rdbuf()->sgetn(&out[0], size) != size)
        {
            outfile_read_failed = true;
            break;
        }
    } while(false);
    file.close();
    if(outfile_read_failed)
    {
        MIOPEN_THROW("Error: X-AMDGCN-ASM: outfile_read_failed");
    }
    return out;
}

static void AmdgcnAssembleQuiet(std::string_view source, std::string_view params)
{
    std::stringstream clang_stdout_unused;
    const auto clang_path = GetGcnAssemblerPath();
    const auto args       = std::string(" -x assembler -target amdgcn--amdhsa") //
                      + " " + params                                            //
                      + " " + source                                            //
                      + " -o /dev/null" + // We do not need output file
                      " 2>&1";            // Keep console clean from error messages.
    MIOPEN_LOG_NQI2(clang_path << " " << args);
    const int clang_rc = miopen::exec::Run(clang_path + " " + args, nullptr, &clang_stdout_unused);
    if(clang_rc != 0)
        MIOPEN_THROW("Assembly error(" + std::to_string(clang_rc) + ")");
}

static bool GcnAssemblerHasBug34765Impl()
{
    auto p = fs::temp_directory_path() / boost::filesystem::unique_path().string();
    miopen::WriteFile(miopen::GetKernelSrc("bugzilla_34765_detect.s"), p);
    const auto& src = p.string();
    try
    {
        AmdgcnAssembleQuiet(src, "-mcpu=gfx900");
        return false;
    }
    catch(...)
    {
        MIOPEN_LOG_NQI("Detected");
        return true;
    }
}

static bool GcnAssemblerHasBug34765()
{
    const static bool b = GcnAssemblerHasBug34765Impl();
    return b;
}

static bool GcnAssemblerSupportsOption(const std::string& option)
{
    auto p = fs::temp_directory_path() / boost::filesystem::unique_path().string();
    miopen::WriteFile(miopen::GetKernelSrc("dummy_kernel.s"), p);
    const auto& src = p.string();
    try
    {
        AmdgcnAssembleQuiet(src, "-mcpu=gfx900 " + option);
        MIOPEN_LOG_NQI("Supported: '" << option << '\'');
        return true;
    }
    catch(...)
    {
        MIOPEN_LOG_NQI("Not supported: '" << option << '\'');
        return false;
    }
}

static bool GcnAssemblerSupportsNoCOv3()
{
    const static bool b = GcnAssemblerSupportsOption(option_no_co_v3);
    return b;
}

#endif // !MIOPEN_USE_COMGR

template <>
void GenerateClangDefsym<const std::string&>(std::ostream& stream,
                                             const std::string& name,
                                             const std::string& value)
{
    stream << " -Wa,-defsym," << name << "=" << value;
}
