// Copyright (C) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "rtc_test_harness.h"
#include "../../shared/environment.h"
#include "device/generator/generator.h"
#include "device/kernel-generator-embed.h"

#if __has_include(<filesystem>)
#include <filesystem>
#else
#include <experimental/filesystem>
namespace std
{
    namespace filesystem = experimental::filesystem;
}
#endif

namespace fs = std::filesystem;

#include <atomic>
#include <fstream>

static std::atomic_uint file_index;

// test if the argument name is a callback pointer
static bool is_cb_ptr(const std::string& arg_name)
{
    return arg_name == "load_cb_fn" || arg_name == "load_cb_data" || arg_name == "store_cb_fn"
           || arg_name == "store_cb_data";
}

// test if the argument name is the callback LDS size value
static bool is_cb_lds_bytes(const std::string& arg_name)
{
    return arg_name == "load_cb_lds_bytes";
}

// test if the argument name is for callbacks
static bool is_cb_arg(const std::string& arg_name)
{
    return is_cb_ptr(arg_name) || is_cb_lds_bytes(arg_name);
}

// remove "const" qualifier, real_type_t from type
std::string cleanup_type(std::string type, const std::string& function_name)
{
    if(type.compare(0, 6, "const ") == 0)
        type.erase(0, 6);
    if(type == "real_type_t<scalar_type>")
    {
        if(function_name.find("_dp") != std::string::npos)
            type = "double";
        if(function_name.find("_sp") != std::string::npos)
            type = "float";
        if(function_name.find("_half") != std::string::npos)
            type = "rocfft_fp16";
    }
    return type;
}

// generate source code for "init_kernel" test harness function
std::string test_harness_init(const Function& f)
{
    const std::string& name = f.name;

    Function init{"init_kernel"};

    init.body += CommentLines{"edit this function to set the inputs to the kernel"};

    Variable gridDim{"gridDim", "dim3"};
    Variable blockDim{"blockDim", "dim3"};
    Variable lds_bytes{"lds_bytes", "unsigned int"};

    std::string scalar_type = "typedef rocfft_complex<";
    // look at the function name to figure out precision
    if(name.find("_dp") != std::string::npos)
        scalar_type += "double";
    else if(name.find("_sp") != std::string::npos)
        scalar_type += "float";
    else if(name.find("_half") != std::string::npos)
        scalar_type += "rocfft_fp16";
    scalar_type += "> scalar_type;\n";

    StatementList globals;
    globals += CommentLines{"declare globals for kernel " + name};
    for(const auto& arg : f.arguments.arguments)
    {
        // callback args are hardcoded to null/0
        if(is_cb_arg(arg.name))
            continue;

        std::string varName = arg.name;

        // functions can declare const parameters - we'll be
        // declaring a gpubuf with a non-const template parameter
        auto actual_type = cleanup_type(arg.type, name);

        // pointers need to point to some device memory
        if(arg.pointer)
        {
            Variable v{varName, "gpubuf_t<" + actual_type + ">"};
            globals += Declaration{v};
            if(actual_type == "scalar_type")
                init.body += Assign{v, "random_complex_device<scalar_type>(0)"};
            else
                init.body += Assign{v, "host_vec_to_dev<" + actual_type + ">({})"};
        }
        else
        {
            // otherwise just pass by value
            Variable v{varName, actual_type};
            globals += Declaration{v};
            init.body += Assign{v, "0"};
        }
    }
    globals += Declaration{gridDim};
    globals += Declaration{blockDim};
    globals += Declaration{lds_bytes};

    init.body += Assign{gridDim, ComplexLiteral{1, 1, 1}};
    init.body += Assign{blockDim, ComplexLiteral{1, 1, 1}};
    init.body += Assign{lds_bytes, 0};

    Variable kargs{"kargs", "RTCKernelArgs"};

    return scalar_type + globals.render() + init.render();
}

// generate source code for "launch_kernel" test harness function
std::string test_harness_launch(const Function& f)
{
    const std::string& name = f.name;

    Variable kargs{"kargs", "RTCKernelArgs"};
    Variable rtckernel{"rtckernel", "std::unique_ptr<RTCKernel>&"};
    Variable deviceProp{"deviceProp", "const hipDeviceProp_t&"};

    Function launch{"launch_kernel"};
    launch.arguments.append(rtckernel);
    launch.arguments.append(deviceProp);

    launch.body += Declaration{kargs};

    for(const auto& arg : f.arguments.arguments)
    {
        // ignore const when looking at argument types
        auto actual_type = cleanup_type(arg.type, name);

        if(is_cb_ptr(arg.name))
            launch.body += Call{"kargs.append_ptr", {"nullptr"}};
        else if(is_cb_lds_bytes(arg.name))
            launch.body += Call{"kargs.append_ptr", {0}};
        else if(arg.pointer)
            launch.body += Call{"kargs.append_ptr", {arg.name + ".data()"}};
        else if(actual_type == "size_t")
            launch.body += Call{"kargs.append_size_t", {arg.name}};
        else if(actual_type == "int")
            launch.body += Call{"kargs.append_int", {arg.name}};
        else if(actual_type == "unsigned int")
            launch.body += Call{"kargs.append_unsigned_int", {arg.name}};
        else if(actual_type == "double")
            launch.body += Call{"kargs.append_double", {"1.0"}};
        else if(actual_type == "float")
            launch.body += Call{"kargs.append_float", {"1.0"}};
        else if(actual_type == "rocfft_fp16")
            launch.body += Call{"kargs.append_half", {"1.0"}};
        else
        {
            throw std::runtime_error("unsupported kernel arg type generating test harness");
        }
    }
    launch.body
        += Call{"rtckernel->launch", {kargs, "gridDim", "blockDim", "lds_bytes", deviceProp}};
    return launch.render();
}

// generate source code for "main" test harness function
std::string test_harness_main(const Function& f)
{
    const std::string& name = f.name;

    Function main{"main"};
    main.return_type = "int";
    Variable rtckernel{"rtc_kernel", "std::unique_ptr<RTCKernel>"};

    main.body += CommentLines{"open kernel source file and read it to a string"};
    Variable kernel_file{"kernel_file", "std::ifstream"};
    Variable kernel_src{"kernel_src", "std::string"};
    main.body += Declaration{kernel_file};
    main.body += Declaration{kernel_src};
    main.body += Call{"kernel_file.open", {"\"" + name + ".h\""}};
    main.body += If{
        Not{CallExpr{"kernel_file.is_open", {}}},
        {Call{"throw std::runtime_error", {"\"" + name + ".h not found in current directory\""}}}};
    main.body += Call{"std::getline", {kernel_file, kernel_src, "static_cast<char>(0)"}};

    main.body += CommentLines{"compile the kernel"};
    main.body += Declaration{rtckernel, CallExpr{"compile", {"\"" + name + "\"", kernel_src}}};

    main.body += CommentLines{"initialize arguments, grid"};
    main.body += Call{"init_kernel", {}};

    Variable deviceProp{"deviceProp", "hipDeviceProp_t"};
    main.body += Declaration{deviceProp, CallExpr{"get_curr_device_prop", {}}};

    Variable trial{"trial", "unsigned int"};
    Variable num_trials{"num_trials", "unsigned int"};
    Variable start{"start", "hipEvent_t"};
    Variable stop{"stop", "hipEvent_t"};
    Variable samples{"samples", "std::vector<float>"};

    main.body += Declaration{num_trials, 1};
    main.body += Declaration{start};
    main.body += Declaration{stop};
    main.body += Declaration{samples};
    main.body += Call{"samples.resize", {num_trials}};

    main.body += If{CallExpr{"hipEventCreate", {start.address()}} != "hipSuccess",
                    {Call{"throw std::runtime_error", {"\"hipEventCreate failed\""}}}};
    main.body += If{CallExpr{"hipEventCreate", {stop.address()}} != "hipSuccess",
                    {Call{"throw std::runtime_error", {"\"hipEventCreate failed\""}}}};

    For launch_loop{trial, 0, trial < num_trials, 1, {}};
    launch_loop.body += If{CallExpr{"hipEventRecord", {start}} != "hipSuccess",
                           {Call{"throw std::runtime_error", {"\"hipEventRecord failed\""}}}};
    launch_loop.body += Call{"launch_kernel", {rtckernel, deviceProp}};
    launch_loop.body += If{CallExpr{"hipEventRecord", {stop}} != "hipSuccess",
                           {Call{"throw std::runtime_error", {"\"hipEventRecord failed\""}}}};
    launch_loop.body += If{CallExpr{"hipEventSynchronize", {stop}} != "hipSuccess",
                           {Call{"throw std::runtime_error", {"\"hipEventRecord failed\""}}}};
    launch_loop.body += If{CallExpr{"hipEventElapsedTime", {"samples.data() + trial", start, stop}}
                               != "hipSuccess",
                           {Call{"throw std::runtime_error", {"\"hipEventElapsedTime failed\""}}}};

    main.body += launch_loop;

    For print_loop{trial, 0, trial < num_trials, 1, {}};
    print_loop.body += Call{"printf", {"\"%f ms\\n\"", "static_cast<double>(samples[trial])"}};
    main.body += print_loop;
    main.body += Call{"std::sort", {"samples.begin()", "samples.end()"}};

    main.body += Call{"printf",
                      {"\"median: %f ms\\n\"",
                       Ternary{num_trials % 2,
                               "samples[num_trials / 2]",
                               "(samples[num_trials / 2] + samples[num_trials / 2 + 1]) / 2"}}};

    main.body += ReturnExpr{0};
    return main.render();
}

// given AST of a kernel and its source code, write a standalone test
// harness for it
void write_standalone_test_harness(const Function& f, const std::string& src)
{
    if(rocfft_getenv("ROCFFT_DEBUG_GENERATE_KERNEL_HARNESS") != "1")
        return;

    const std::string& name = f.name;

    // give each kernel its own numbered source file - plans contain
    // multiple kernels, and kernels are compiled in parallel, so we
    // need to make sure they all have unique names
    auto cur_file_index = file_index.fetch_add(1);

    auto env_user_path = rocfft_getenv("ROCFFT_DEBUG_KERNEL_HARNESS_PATH");
    auto user_path     = env_user_path.empty() ? fs::current_path() : fs::path(env_user_path);

    auto file_name = ("rocfft_kernel_harness_" + std::to_string(cur_file_index) + ".cpp");
    auto file_path = user_path / file_name;

    std::ofstream main_file(file_path.string());

    main_file << "// standalone test harness for kernel " << name << ".\n";
    main_file << "// edit init_kernel to set args + grid.\n\n";

    // Embedded source files have #includes stripped out because hipRTC
    // normally won't be able to find them.  Add includes for test
    // harness files
    main_file << "#include <hip/hip_runtime_api.h>\n";
    main_file << "#include <hip/hiprtc.h>\n";
    main_file << "#include <fstream>\n";
    main_file << "#include <functional>\n";
    main_file << "#include <future>\n";
    main_file << "#include <memory>\n";
    main_file << "#include <random>\n";
    main_file << "#include <string>\n";
    main_file << "#include <vector>\n";

    main_file << "#define ROCFFT_DEBUG_GENERATE_KERNEL_HARNESS\n";
    main_file << rocfft_hip_h;
    main_file << rocfft_complex_h;
    main_file << gpubuf_h;
    main_file << device_properties_h;
    main_file << rtc_kernel_h;
    main_file << rtc_kernel_cpp;
    main_file << rtc_test_harness_helper_cpp;

    main_file << test_harness_init(f);
    main_file << "\n\n";
    main_file << test_harness_launch(f);
    main_file << "\n\n";
    main_file << test_harness_main(f);

    // write kernel source to its own file, so it can be formatted
    // easily if desired
    std::ofstream kernel_file(user_path / (name + ".h"));
    kernel_file << src;
}
