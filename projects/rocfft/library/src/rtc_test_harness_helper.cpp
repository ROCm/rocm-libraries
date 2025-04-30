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

// utility code to embed into generated test harnesses, to simplify
// allocating and initializing device memory

// copy a host vector to the device
template <typename T>
gpubuf_t<T> host_vec_to_dev(const std::vector<T>& hvec)
{
    gpubuf_t<T> ret;
    if(ret.alloc(sizeof(T) * hvec.size()) != hipSuccess)
        throw std::runtime_error("failed to hipMalloc");
    if(hipMemcpy(ret.data(), hvec.data(), sizeof(T) * hvec.size(), hipMemcpyHostToDevice)
       != hipSuccess)
        throw std::runtime_error("failed to memcpy");
    return ret;
}

template <typename T1, typename T2>
T1 ceildiv(T1 a, T2 b)
{
    return (a + b - 1) / b;
}

// generate random complex input
template <typename Tcomplex>
gpubuf_t<Tcomplex> random_complex_device(unsigned int count)
{
    std::vector<Tcomplex> hostBuf(count);

    auto partitions     = std::max<size_t>(std::thread::hardware_concurrency(), 32);
    auto partition_size = ceildiv(count, partitions);

#pragma omp parallel for
    for(unsigned int partition = 0; partition < partitions; ++partition)
    {
        std::mt19937                           gen(partition);
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        auto begin = partition * partition_size;
        if(begin >= count)
            continue;
        auto end = std::min(begin + partition_size, count);

        for(auto d = hostBuf.begin() + begin; d != hostBuf.begin() + end; ++d)
        {
            d->x = dis(gen);
            d->y = dis(gen);
        }
    }
    return host_vec_to_dev(hostBuf);
}

// generate random real input
template <typename Treal>
gpubuf_t<Treal> random_real_device(unsigned int count)
{
    std::vector<Treal> hostBuf(count);

    auto partitions     = std::max<size_t>(std::thread::hardware_concurrency(), 32);
    auto partition_size = ceildiv(count, partitions);

#pragma omp parallel for
    for(unsigned int partition = 0; partition < partitions; ++partition)
    {
        std::mt19937                           gen(partition);
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        auto begin = partition * partition_size;
        if(begin >= count)
            continue;
        auto end = std::min(begin + partition_size, count);

        for(auto d = hostBuf.begin() + begin; d != hostBuf.begin() + end; ++d)
        {
            *d = dis(gen);
        }
    }
    return host_vec_to_dev(hostBuf);
}

// compile a function using hipRTC
std::unique_ptr<RTCKernel> compile(const std::string& name, const std::string& src)
{
    hiprtcProgram prog;
    if(hiprtcCreateProgram(&prog, src.c_str(), "rtc.cu", 0, nullptr, nullptr) != HIPRTC_SUCCESS)
    {
        throw std::runtime_error("unable to create program");
    }
    std::vector<const char*> options;
    options.reserve(2);
    options.push_back("-O3");
    options.push_back("-mcumode");

    auto compileResult = hiprtcCompileProgram(prog, options.size(), options.data());
    if(compileResult != HIPRTC_SUCCESS)
    {
        size_t logSize = 0;
        hiprtcGetProgramLogSize(prog, &logSize);

        if(logSize)
        {
            std::vector<char> log(logSize, '\0');
            if(hiprtcGetProgramLog(prog, log.data()) == HIPRTC_SUCCESS)
                throw std::runtime_error(log.data());
        }
        throw std::runtime_error("compile failed without log");
    }

    size_t codeSize;
    if(hiprtcGetCodeSize(prog, &codeSize) != HIPRTC_SUCCESS)
        throw std::runtime_error("failed to get code size");

    std::vector<char> code(codeSize);
    if(hiprtcGetCode(prog, code.data()) != HIPRTC_SUCCESS)
        throw std::runtime_error("failed to get code");
    hiprtcDestroyProgram(&prog);

    return std::make_unique<RTCKernel>(name, code);
}
