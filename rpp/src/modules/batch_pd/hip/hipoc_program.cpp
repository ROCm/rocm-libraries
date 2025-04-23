/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

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

#include <sstream>
#include <optional>
#include <unistd.h>
#include <hip/hiprtc.h>

#include "hipoc_program.hpp"
#include "errors.hpp"
#include "gcn_asm_utils.hpp"
#include "hip_build_utils.hpp"
#include "kernel.hpp"
#include "kernel_warnings.hpp"
#include "stringutils.hpp"
#include "tmp_dir.hpp"
#include "write_file.hpp"

namespace rpp {

std::string remove_extension(const std::string& filename) {
    size_t lastdot = filename.find_last_of(".");
    if (lastdot == std::string::npos) return filename;
    return filename.substr(0, lastdot);
}

hipModulePtr CreateModule(const fs::path& hsaco_file)
{
    hipModule_t raw_m;
    // std::cout<<"attempting to run hipModule Load of hsacofile:\n";// <<hsaco_file.string().c_str()<<std::endl;
    auto status = hipModuleLoad(&raw_m, hsaco_file.string().c_str());
    hipModulePtr m{raw_m};
    if(status != hipSuccess)
        RPP_THROW_HIP_STATUS(status, "Failed creating module");
    return m;
}
hipModulePtr CreateModuleRTC(char** codeData)
{
    hipModule_t raw_m;
    // std::cout<<"attempting to run hipModule Load data of codeData:\n";// <<hsaco_file.string().c_str()<<std::endl;
    auto status = hipModuleLoadData(&raw_m, *codeData);
    hipModulePtr m{raw_m};
    if(status != hipSuccess)
        RPP_THROW_HIP_STATUS(status, "Failed creating module");
    return m;
}
struct HIPOCProgramImpl
{
    HIPOCProgramImpl(const std::string& program_name, const fs::path& hsaco)
        : name(program_name), hsaco_file(hsaco)
    {
        this->module = CreateModule(this->hsaco_file);
    }
    HIPOCProgramImpl(const std::string& program_name,
                     std::string params,
                     bool is_kernel_str,
                     std::string _dev_name,
                     const std::string& kernel_src)
        : name(program_name), dev_name(_dev_name)
    {
#if defined(HIPRTC)
        // std::cout<<"Using HIPRTC API to do run time compilation\n";
        this->module = this->BuildAndCreateModuleRTC(program_name, params, is_kernel_str, kernel_src);
#elif defined(HSACOO)
        // std::cout<<"Using HSACOO MODE OF RUNTIME COMPILATION\n";
        this->BuildModule(program_name, params, is_kernel_str, kernel_src);
        this->module = CreateModule(this->hsaco_file);
#else
        std::cout<<"Need to build the kernels statically and call the hipLaunch of desired kernel"<<std::endl;
#endif

    }
    std::string name;
    std::string dev_name;
    fs::path hsaco_file;
    hipModulePtr module;
    std::optional<TmpDir> dir;
    hipModulePtr BuildAndCreateModuleRTC(const std::string& program_name,
                     std::string params,
                     bool is_kernel_str,
                     const std::string& kernel_src) {
        std::string filename =
            is_kernel_str ? "tinygemm.cl" : program_name; // jn : don't know what this is
        dir.emplace(filename);
        hiprtcProgram prog;
	std::string buffer = GetKernelSrc(program_name);
	hiprtcCreateProgram(&prog,      // prog
                        buffer.c_str(),      // buffer
			filename.c_str(), // name
			0,          // numHeaders
			nullptr,    // headers
			nullptr);   // includeNames
    	hipDeviceProp_t props;
	int device = 0;
	hipGetDeviceProperties(&props, device);
      	std::string archName(props.gcnArchName);
    	std::string sarg = "--gpu-architecture=" + archName;
	const char* options[] = {
        	sarg.c_str()
    	};
    	hiprtcResult compileResult{hiprtcCompileProgram(prog, 1, options)};
	size_t logSize;
	hiprtcGetProgramLogSize(prog, &logSize);
	if (logSize) {
        	std::string log(logSize, '\0');
		hiprtcGetProgramLog(prog, &log[0]);
        	// std::cout << log << '\n';
    	}
	if (compileResult != HIPRTC_SUCCESS) { std::cout<<"Compilation failed."<<compileResult<<std::endl; exit(1);}
	size_t codeSize;
	hiprtcGetCodeSize(prog, &codeSize);
    	std::vector<char> code(codeSize);
	hiprtcGetCode(prog, code.data());
	hiprtcDestroyProgram(&prog);
	hipModule_t raw_m;
    	hipFunction_t kernel;
	auto status = hipModuleLoadData(&raw_m, code.data());
	if(status != hipSuccess)
        	RPP_THROW_HIP_STATUS(status, "Failed creating module");
        hipModulePtr m{raw_m};
        return m;
    }

    void BuildModule(const std::string& program_name,
                     std::string params,
                     bool is_kernel_str,
                     const std::string& kernel_src)
    {
        // std::cout<<"Build Module invoked"<<std::endl;
        std::string filename =
            is_kernel_str ? "tinygemm.cl" : program_name; // jn : don't know what this is
        dir.emplace(filename);

        // std::cout<<"filename"<<filename<<std::endl;
        this->hsaco_file = dir->path / (filename + ".o");
        // std::cout<<"hsaco file path"<<hsaco_file<<std::endl;
        // std::cout<<"isKernelStr"<<is_kernel_str<<std::endl;
        std::string src;
        if(kernel_src.empty()) {
            src = is_kernel_str ? program_name : GetKernelSrc(program_name);
            // std::cout<<"Done with GetKErnelSrc call\n";
        }
        else
            src = kernel_src;
        if(!is_kernel_str && rpp::EndsWith(program_name, ".so"))
        {
            WriteFile(src, hsaco_file);
        }
        else if(!is_kernel_str && rpp::EndsWith(program_name, ".s"))
        {
            //AmdgcnAssemble(src, params);
            WriteFile(src, hsaco_file);
        }
        else if(!is_kernel_str && rpp::EndsWith(program_name, ".cpp"))
        {
#if RPP_BUILD_DEV
            params += " -Werror" + HipKernelWarningsString();
#else
            params += " -Wno-everything";
#endif
            // std::cout<<"Hip Build being invoked\n";
            hsaco_file = HipBuild(dir, filename, src, params, dev_name);
        }
        else
        {
	    std::cout<<"Else WriteFile Case::::::::::::::::::\n";

            WriteFile(src, dir->path / filename);

#if RPP_BUILD_DEV
            params += " -Werror" + OclKernelWarningsString();
#else
            params += " -Wno-everything";
#endif
            dir->Execute("/usr/local/bin/clang-ocl", params + " " + filename + " -o " + hsaco_file.string());
        }
        // std::cout<<"Done Building Module to get hsaco_file "<<hsaco_file.string().c_str()<<std::flush;
        //if(!fs::exists(hsaco_file))
          //  RPP_THROW("Cant find file: " + hsaco_file.string());

    ;
    }

};

HIPOCProgram::HIPOCProgram() {}
HIPOCProgram::HIPOCProgram(const std::string& program_name,
                           std::string params,
                           bool is_kernel_str,
                           std::string dev_name,
                           const std::string& kernel_src)
    : impl(std::make_shared<HIPOCProgramImpl>(
          program_name, params, is_kernel_str, dev_name, kernel_src))
{
}

HIPOCProgram::HIPOCProgram(const std::string& program_name, const fs::path& hsaco)
    : impl(std::make_shared<HIPOCProgramImpl>(program_name, hsaco))
{
}

hipModule_t HIPOCProgram::GetModule() const { return this->impl->module.get(); }

fs::path HIPOCProgram::GetBinary() const { return this->impl->hsaco_file; }

} // namespace rpp
