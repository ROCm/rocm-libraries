// Copyright (C) 2023 - 2024 Advanced Micro Devices, Inc. All rights reserved.
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

//
// A simple standalone interface to tune rocFFT single kernel
//
// Usage examples:
//
//     -- brute-force tuning
//        rocfft_kernel_config_search brute-force -l 64
//     -- manual tuning
//        rocfft_kernel_config_search manual -l 64 -b 1 -f 8 8 -w 64 --tpt 2 --half-lds 1 --direct-reg 1
//

#include "../../shared/CLI11.hpp"
#include "../../shared/arithmetic.h"
#include "../../shared/gpubuf.h"
#include "../../shared/hip_object_wrapper.h"
#include "device/generator/stockham_gen.h"
#include "rtc_compile.h"
#include "rtc_stockham_gen.h"
#include "rtc_stockham_kernel.h"

#include <iostream>
#include <iterator>
#include <map>
#include <random>
#include <set>

static const std::vector<unsigned int> supported_factors
    = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 16, 17};
static const std::vector<unsigned int> supported_wgs{64, 128, 256};

// recursively find all unique factorizations of given length.  each
// factorization is a vector of ints, sorted so they're uniquified in
// a set.
std::set<std::vector<unsigned int>> factorize(unsigned int length)
{
    std::set<std::vector<unsigned int>> ret;
    for(auto factor : supported_factors)
    {
        if(length % factor == 0)
        {
            unsigned int remain = length / factor;
            if(remain == 1)
                ret.insert({factor});
            else
            {
                // recurse into remainder
                auto remain_factorization = factorize(remain);
                for(auto& remain_factors : remain_factorization)
                {
                    std::vector<unsigned int> factors{factor};
                    std::copy(
                        remain_factors.begin(), remain_factors.end(), std::back_inserter(factors));
                    std::sort(factors.begin(), factors.end());
                    ret.insert(factors);
                }
            }
        }
    }
    return ret;
}

// recursively return power set of a range of ints
std::set<std::vector<unsigned int>> power_set(std::vector<unsigned int>::const_iterator begin,
                                              std::vector<unsigned int>::const_iterator end)
{
    std::set<std::vector<unsigned int>> ret;
    // either include the front element in the output, or don't
    if(std::distance(begin, end) == 1)
    {
        ret.insert({*begin});
        ret.insert({});
    }
    else
    {
        // recurse into the remainder
        auto remain = power_set(begin + 1, end);
        for(auto r : remain)
        {
            ret.insert(r);
            r.push_back(*begin);
            ret.insert(r);
        }
    }
    return ret;
}

std::set<unsigned int, std::greater<unsigned int>>
    supported_threads_per_transform(const std::vector<unsigned int>& factorization)
{
    std::set<unsigned int, std::greater<unsigned int>> tpts;
    auto tpt_candidates = power_set(factorization.begin(), factorization.end());
    for(auto tpt : tpt_candidates)
    {
        if(tpt.empty())
            continue;
        tpts.insert(product(tpt.begin(), tpt.end()));
    }
    return tpts;
}

std::string test_kernel_name(unsigned int                     length,
                             const std::vector<unsigned int>& factorization,
                             unsigned int                     wgs,
                             unsigned int                     tpt,
                             bool                             half_lds,
                             bool                             direct_to_from_reg)
{
    std::string ret = "fft_test_len_";
    ret += std::to_string(length);
    ret += "_factors";
    for(auto f : factorization)
    {
        ret += "_";
        ret += std::to_string(f);
    }
    ret += "_wgs_";
    ret += std::to_string(wgs);
    ret += "_tpt_";
    ret += std::to_string(tpt);
    if(half_lds)
        ret += "_halfLds";
    if(direct_to_from_reg)
        ret += "_dirReg";

    return ret;
}

std::string test_kernel_src(const std::string&               kernel_name,
                            hipDeviceProp_t                  device_prop,
                            unsigned int&                    transforms_per_block,
                            unsigned int                     length,
                            ComputeScheme                    compute_scheme,
                            rocfft_precision                 precision,
                            const std::vector<unsigned int>& factorization,
                            unsigned int                     wgs,
                            unsigned int                     tpt,
                            bool                             half_lds,
                            bool                             direct_to_from_reg)
{
    StockhamGeneratorSpecs specs{factorization,
                                 {},
                                 {static_cast<unsigned int>(rocfft_precision_single)},
                                 wgs,
                                 PrintScheme(compute_scheme)};

    auto ppParams = StockhamPartialPassParams();

    specs.threads_per_transform = tpt;
    specs.half_lds              = half_lds;
    specs.direct_to_from_reg    = direct_to_from_reg;
    // aim for occupancy-2
    specs.lds_byte_limit = device_prop.sharedMemPerBlock / 2;

    return stockham_rtc(specs,
                        specs,
                        ppParams,
                        &transforms_per_block,
                        kernel_name,
                        compute_scheme,
                        -1,
                        precision,
                        rocfft_placement_notinplace,
                        rocfft_array_type_complex_interleaved,
                        rocfft_array_type_complex_interleaved,
                        true,
                        0,
                        0,
                        false,
                        direct_to_from_reg ? DirectRegType::TRY_ENABLE_IF_SUPPORT
                                           : DirectRegType::FORCE_OFF_OR_NOT_SUPPORT,
                        IntrinsicAccessType::DISABLE_BOTH,
                        SBRC_TRANSPOSE_TYPE::NONE,
                        CallbackType::NONE,
                        BluesteinFuseType::BFT_NONE,
                        PartialPassType::PPT_NONE,
                        {},
                        {});
}

// things that we need to remember between kernel launches
struct device_data_t
{
    std::vector<rocfft_complex<float>> host_input_buf;
    gpubuf_t<rocfft_complex<float>>    fake_twiddles;
    gpubuf_t<rocfft_complex<float>>    input_buf;
    gpubuf_t<rocfft_complex<float>>    output_buf;
    gpubuf_t<size_t>                   lengths;
    gpubuf_t<size_t>                   stride_in;
    gpubuf_t<size_t>                   stride_out;
    size_t                             batch;
    hipEvent_wrapper_t                 start;
    hipEvent_wrapper_t                 stop;

    device_data_t()
    {
        start.alloc();
        stop.alloc();
    }
    ~device_data_t() = default;
};

// run the kernel, returning the median execution time
float launch_kernel(RTCKernel&             kernel,
                    unsigned int           blocks,
                    unsigned int           wgs,
                    unsigned int           lds_bytes,
                    unsigned int           ntrial,
                    const hipDeviceProp_t& prop,
                    device_data_t&         data)
{
    RTCKernelArgs kargs;
    kargs.append_ptr(data.fake_twiddles.data());
    kargs.append_size_t(1);
    kargs.append_ptr(data.lengths.data());
    kargs.append_ptr(data.stride_in.data());
    kargs.append_ptr(data.stride_out.data());
    kargs.append_size_t(data.batch);
    kargs.append_ptr(nullptr);
    kargs.append_ptr(nullptr);
    kargs.append_unsigned_int(0);
    kargs.append_ptr(nullptr);
    kargs.append_ptr(nullptr);
    kargs.append_ptr(data.input_buf.data());
    kargs.append_ptr(data.output_buf.data());
    std::vector<float> times;
    for(unsigned int i = 0; i < ntrial; ++i)
    {
        // simulate rocfft-bench behaviour - memcpy input to device
        // before each execution
        if(hipMemcpy(data.input_buf.data(),
                     data.host_input_buf.data(),
                     data.host_input_buf.size() * sizeof(rocfft_complex<float>),
                     hipMemcpyHostToDevice)
           != hipSuccess)
            throw std::runtime_error("failed to hipMemcpy");

        if(hipEventRecord(data.start) != hipSuccess)
            throw std::runtime_error("hipEventRecord start failed");
        kernel.launch(kargs, {blocks}, {wgs}, lds_bytes, prop);
        if(hipEventRecord(data.stop) != hipSuccess)
            throw std::runtime_error("hipEventRecord stop failed");
        if(hipEventSynchronize(data.stop) != hipSuccess)
            throw std::runtime_error("hipEventSynchronize failed");
        float time;
        if(hipEventElapsedTime(&time, data.start, data.stop) != hipSuccess)
            throw std::runtime_error("hipEventElapsedTime failed");
        times.push_back(time);
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

unsigned int get_lds_bytes(unsigned int length, unsigned int transforms_per_block, bool half_lds)
{
    // assume single precision complex
    return length * transforms_per_block * sizeof(rocfft_complex<float>) / (half_lds ? 2 : 1);
}

size_t batch_size(unsigned int length)
{
    // target 2 GiB memory usage (2^31), assume single precision so
    // each element is 2^3 bytes
    size_t target_elems = 1U << 28;
    return target_elems / length;
}

std::vector<rocfft_complex<float>> create_input_buf(unsigned int length, size_t batch)
{
    auto                               elems = length * batch;
    std::vector<rocfft_complex<float>> buf;
    buf.reserve(elems);
    std::mt19937 gen;
    for(unsigned int i = 0; i < elems; ++i)
    {
        float x = static_cast<float>(gen()) / static_cast<float>(gen.max());
        float y = static_cast<float>(gen()) / static_cast<float>(gen.max());
        buf.push_back({x, y});
    }
    return buf;
}

gpubuf_t<rocfft_complex<float>> create_device_buf(unsigned int length, size_t batch)
{
    auto                            elems = length * batch;
    gpubuf_t<rocfft_complex<float>> device_buf;
    if(device_buf.alloc(elems * sizeof(rocfft_complex<float>)) != hipSuccess)
        throw std::runtime_error("failed to hipMalloc");
    if(hipMemset(device_buf.data(), 0, elems * sizeof(rocfft_complex<float>)) != hipSuccess)
        throw std::runtime_error("failed to hipMemset");

    return device_buf;
}

gpubuf_t<size_t> create_lengths(unsigned int length)
{
    gpubuf_t<size_t> device_buf;
    if(device_buf.alloc(sizeof(size_t)) != hipSuccess)
        throw std::runtime_error("failed to hipMalloc");

    if(hipMemcpy(device_buf.data(), &length, sizeof(size_t), hipMemcpyHostToDevice) != hipSuccess)
        throw std::runtime_error("failed to hipMemcpy");
    return device_buf;
}

gpubuf_t<size_t> create_strides(unsigned int length)
{
    std::array<size_t, 2> strides{1, length};
    gpubuf_t<size_t>      device_buf;
    if(device_buf.alloc(sizeof(size_t) * 2) != hipSuccess)
        throw std::runtime_error("failed to hipMalloc");
    if(hipMemcpy(device_buf.data(), strides.data(), 2 * sizeof(size_t), hipMemcpyHostToDevice)
       != hipSuccess)
        throw std::runtime_error("failed to hipMemcpy");
    return device_buf;
}

int main(int argc, char** argv)
{
    unsigned int  length         = 0;
    unsigned int  ntrial         = 0;
    unsigned int  nbatch         = 1;
    ComputeScheme compute_scheme = CS_KERNEL_STOCKHAM;

    rocfft_precision                        precision = rocfft_precision_single;
    std::map<std::string, rocfft_precision> precision_map{
        {"single", rocfft_precision::rocfft_precision_single},
        {"double", rocfft_precision::rocfft_precision_double},
        {"half", rocfft_precision::rocfft_precision_half}};

    CLI::App app{"rocfft kernel config search"};

    auto brute_force = app.add_subcommand(
        "brute-force", "brute force tuning kernel config with build-in combinations");

    brute_force->add_option("-l, --length", length, "Select a 1D FFT problem size")->default_val(8);
    brute_force->add_option("-N, --ntrial", ntrial, "Trial size for tuning the problem")
        ->default_val(10);

    auto manual_tuning = app.add_subcommand("manual", "manual tuning kernel config");

    std::string               kernel_type;
    std::vector<unsigned int> factorization;
    unsigned int              wgs;
    unsigned int              tpt;
    bool                      half_lds           = true;
    bool                      direct_to_from_reg = true;

    manual_tuning
        ->add_option("--kernel-type", kernel_type, "The valid types are: sbrr/sbcc/sbrc/sbcr")
        ->default_val("sbrr");
    manual_tuning->add_option("-l, --length", length, "Select a 1D FFT problem size")
        ->default_val(8);
    manual_tuning->add_option("-N, --ntrial", ntrial, "Trial size for tuning the problem")
        ->default_val(10);
    manual_tuning
        ->add_option(
            "--precision", precision, "Transform precision: single (default), double, half")
        ->transform(CLI::CheckedTransformer(precision_map, CLI::ignore_case));
    manual_tuning->add_option("-b, --batchSize", nbatch, "Batch size of FFT")->default_val(1);
    manual_tuning->add_option(
        "-f, --factorization", factorization, "Factorization for a given FFT problem");
    manual_tuning->add_option("-w, --wgs", wgs, "Work group size")->default_val(64);
    manual_tuning->add_option("--tpt", tpt, "Thread per transform")->default_val(1);
    manual_tuning->add_option("--half-lds", half_lds, "Use half LDS or not")->default_val(true);
    manual_tuning->add_option("--direct-reg", direct_to_from_reg, "Direct load to/from reg")
        ->default_val(true);

    app.require_subcommand(0, 1);

    // Parse args and catch any errors here
    try
    {
        app.parse(argc, argv);
    }
    catch(const CLI::ParseError& e)
    {
        return app.exit(e);
    }

    if(hipInit(0) != hipSuccess)
        throw std::runtime_error("hipInit failure");

    hipDeviceProp_t device_prop;
    // Todo: support device id
    if(hipGetDeviceProperties(&device_prop, 0) != hipSuccess)
        throw std::runtime_error("hipGetDeviceProperties failure");

    if(brute_force->parsed())
    {
        // init device data
        device_data_t data;
        data.batch = batch_size(length);
        // construct random input on host side, allocate input/output
        // buffers on GPU.  input will be copied to GPU at launch time
        data.host_input_buf = create_input_buf(length, data.batch);
        data.input_buf      = create_device_buf(length, data.batch);
        data.output_buf     = create_device_buf(length, data.batch);
        // create twiddles table same length as FFT.  this isn't exactly
        // what rocFFT would do but is close enough.
        auto host_twiddles = create_input_buf(length, 1);
        data.fake_twiddles = create_device_buf(length, 1);
        if(hipMemcpy(data.fake_twiddles.data(),
                     host_twiddles.data(),
                     host_twiddles.size() * sizeof(rocfft_complex<float>),
                     hipMemcpyHostToDevice)
           != hipSuccess)
            throw std::runtime_error("failed to hipMemcpy");
        data.lengths    = create_lengths(length);
        data.stride_in  = create_strides(length);
        data.stride_out = create_strides(length);
        std::cout << "length " << length << ", batch " << data.batch << std::endl;

        const auto factorizations = factorize(length);

        // remember the best configuration observed so far
        float                     best_time               = std::numeric_limits<float>::max();
        unsigned int              best_wgs                = 0;
        unsigned int              best_tpt                = 0;
        bool                      best_half_lds           = true;
        bool                      best_direct_to_from_reg = true;
        std::vector<unsigned int> best_factorization;
        std::string               best_kernel_src;

        size_t count = 0;
        for(auto factorization : factorizations)
        {
            ++count;
            std::cout << "factorization " << count << " of " << factorizations.size() << std::endl;

            auto tpts = supported_threads_per_transform(factorization);

            // go through all permutations of the factors
            do
            {
                for(auto wgs : supported_wgs)
                {
                    for(auto tpt : tpts)
                    {
                        // tpt must fit into wgs
                        if(tpt > wgs)
                            continue;

                        // There's no point in testing a half-full
                        // wgs since so many threads will be doing
                        // nothing.  And at larger wgs you could just
                        // drop to a smaller one anyway.
                        if(tpt <= wgs / 2)
                            continue;

                        for(bool half_lds : {true, false})
                        {
                            for(bool direct_to_from_reg : {true, false})
                            {
                                // half lds currently requires direct to/from reg
                                if(half_lds && !direct_to_from_reg)
                                    continue;
                                auto kernel_name = test_kernel_name(
                                    length, factorization, wgs, tpt, half_lds, direct_to_from_reg);
                                unsigned int transforms_per_block = 0;
                                auto         kernel_src           = test_kernel_src(kernel_name,
                                                                  device_prop,
                                                                  transforms_per_block,
                                                                  length,
                                                                  compute_scheme,
                                                                  precision,
                                                                  factorization,
                                                                  wgs,
                                                                  tpt,
                                                                  half_lds,
                                                                  direct_to_from_reg);

                                auto code = compile_inprocess(kernel_src, device_prop.gcnArchName);
                                RTCKernelStockham kernel(kernel_name, code);

                                float time = launch_kernel(
                                    kernel,
                                    DivRoundingUp<unsigned int>(data.batch, transforms_per_block),
                                    tpt * transforms_per_block,
                                    get_lds_bytes(length, transforms_per_block, half_lds),
                                    ntrial,
                                    device_prop,
                                    data);

                                // print median time for this length
                                // in a format that can be easily
                                // grepped for and shoved into a
                                // database if desired
                                std::cout << length << ", " << kernel_name << ", "
                                          << std::setprecision(3) << static_cast<double>(time)
                                          << "ms " << std::endl;

                                if(time < best_time)
                                {
                                    best_time               = time;
                                    best_wgs                = wgs;
                                    best_tpt                = tpt;
                                    best_half_lds           = half_lds;
                                    best_direct_to_from_reg = direct_to_from_reg;
                                    best_factorization      = factorization;
                                    best_kernel_src         = std::move(kernel_src);
                                }
                            }
                        }
                        // found the largest number of threads
                        // that'll fit into wgs, don't bother
                        // testing fewer as that's just worse
                        // utilization of the workgroup
                        break;
                    }
                }
            } while(std::next_permutation(factorization.begin(), factorization.end()));
        }

        // print a line with the best config, to go into kernel-generator.py
        std::cout << "  NS(length= " << length << ", workgroup_size= " << best_wgs
                  << ", threads_per_transform=" << best_tpt << ", factors=(";
        bool first_factor = true;
        for(auto f : best_factorization)
        {
            if(!first_factor)
                std::cout << ", ";
            first_factor = false;
            std::cout << f;
        }
        std::cout << ")";
        if(!best_half_lds)
            std::cout << ", half_lds=False";
        if(!best_direct_to_from_reg)
            std::cout << ", direct_to_from_reg=False";
        std::cout << " best_time," << best_time << std::endl;
    }
    else if(manual_tuning->parsed())
    {
        // init device data
        device_data_t data;
        data.batch = nbatch;
        // construct random input on host side, allocate input/output
        // buffers on GPU.  input will be copied to GPU at launch time
        data.host_input_buf = create_input_buf(length, data.batch);
        data.input_buf      = create_device_buf(length, data.batch);
        data.output_buf     = create_device_buf(length, data.batch);
        // create twiddles table same length as FFT.  this isn't exactly
        // what rocFFT would do but is close enough.
        auto host_twiddles = create_input_buf(length, 1);
        data.fake_twiddles = create_device_buf(length, 1);
        if(hipMemcpy(data.fake_twiddles.data(),
                     host_twiddles.data(),
                     host_twiddles.size() * sizeof(rocfft_complex<float>),
                     hipMemcpyHostToDevice)
           != hipSuccess)
            throw std::runtime_error("failed to hipMemcpy");
        data.lengths    = create_lengths(length);
        data.stride_in  = create_strides(length);
        data.stride_out = create_strides(length);

        // Fixme: support other kernel types properly
        if(kernel_type == "sbcc")
            compute_scheme = CS_KERNEL_STOCKHAM_BLOCK_CC;
        else if(kernel_type == "sbrc")
            compute_scheme = CS_KERNEL_STOCKHAM_BLOCK_RC;
        else if(kernel_type == "sbcr")
            compute_scheme = CS_KERNEL_STOCKHAM_BLOCK_CR;

        // Todo: verify factorization
        // const auto factorizations = factorize(length);

        // Todo: verify tpt
        // auto tpts = supported_threads_per_transform(factorization);
        if(tpt < wgs)
        {
            // Fixme:
            // half lds currently requires direct to/from reg

            auto kernel_name
                = test_kernel_name(length, factorization, wgs, tpt, half_lds, direct_to_from_reg);
            unsigned int transforms_per_block = 0;

            auto kernel_src = test_kernel_src(kernel_name,
                                              device_prop,
                                              transforms_per_block,
                                              length,
                                              compute_scheme,
                                              precision,
                                              factorization,
                                              wgs,
                                              tpt,
                                              half_lds,
                                              direct_to_from_reg);

            auto              code = compile_inprocess(kernel_src, device_prop.gcnArchName);
            RTCKernelStockham kernel(kernel_name, code);

            float time
                = launch_kernel(kernel,
                                DivRoundingUp<unsigned int>(data.batch, transforms_per_block),
                                tpt * transforms_per_block,
                                get_lds_bytes(length, transforms_per_block, half_lds),
                                ntrial,
                                device_prop,
                                data);

            // print median time for this length in a format that can be easily
            // grepped for and shoved into a database if desired
            std::cout << kernel_name << ", " << std::setprecision(3) << static_cast<double>(time)
                      << std::endl;
        }
    }

    return 0;
}
