/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2023 Advanced Micro Devices, Inc.
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
#ifndef GUARD_MIOPEN_CAT_DRIVER_HPP
#define GUARD_MIOPEN_CAT_DRIVER_HPP

#include "InputFlags.hpp"
#include "driver.hpp"
#include "random.hpp"
#include "tensor_driver.hpp"
#include "timer.hpp"
#include <algorithm>
#include <cfloat>
#include <cstdlib>
#include <memory>
#include <miopen/miopen.h>
#include <miopen/tensor.hpp>
#include <numeric>
#include <vector>
#include <../test/tensor_holder.hpp>
#include <../test/verify.hpp>

#include <fstream>
#include <iostream>
#include <miopen/ford.hpp>

#ifndef MLO_CATHOST_H_
#define MLO_CATHOST_H_

template <typename Tgpu, typename Tcheck>
int32_t mloCatForwardRunHost(std::vector<miopenTensorDescriptor_t> inputDescs,
                             std::vector<Tgpu*> inputs,
                             miopenTensorDescriptor_t outputDesc,
                             Tcheck* outputhost,
                             uint32_t dim)
{
    // std::vector<size_t> copySizes(inputs.size());
    const auto t0          = std::chrono::high_resolution_clock::now();
    auto shape             = miopen::deref(outputDesc).GetLengths();
    size_t outer_size      = 1;
    size_t inner_size      = 1;
    size_t output_dim_size = shape[dim];
    for(size_t i = 0; i < dim; i++)
    {
        outer_size *= shape[i];
    }

    for(size_t i = dim + 1; i < shape.size(); i++)
    {
        inner_size *= shape[i];
    }

    // std::cout << "Shape: ";
    // for (const auto& i : shape)
    // {
    //     std::cout << i << " ";
    // }
    // std::cout << "\nouter_size: " << outer_size << ", inner_size: " << inner_size << std::endl;

    int32_t ret                = 0;
    size_t output_start_offset = 0;

    for(size_t i = 0; i < inputs.size(); i++)
    {
        auto input       = inputs[i];
        size_t dim_size  = miopen::deref(inputDescs[i]).GetLengths()[dim];
        size_t copy_size = inner_size * dim_size;
        // copySizes[i] = copy_size;
        // std::cout << "\ncopy_size: " << copy_size << ", dim_size: " << dim_size << std::endl;
        for(size_t o = 0; o < outer_size; o++)
        {
            size_t output_offset = output_start_offset + (o * inner_size * output_dim_size);
            for(size_t j = 0; j < copy_size; j++)
            {
                outputhost[output_offset + j] = input[copy_size * o + j];
            }
        }
        output_start_offset += copy_size;
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    std::cout << "Original CPU CAT solver: " << ns << "ns (" << (double(ns) / 1000000000.0)
              << " sec)" << std::endl;

    std::fstream of("/data/Dev/mloCatForwardRunHost.txt", std::ofstream::app);
    // of << ns << ";";

    // of << outer_size << ";";
    // for (const auto& s : copySizes)
    // {
    //     of << s << ";";
    // }
    // of << std::endl;

    of << ns << std::endl;

    return ret;
}

template <typename Tgpu, typename Tcheck>
int32_t mloCatForwardRunHost_upd(const std::vector<miopenTensorDescriptor_t>& inputDescs,
                                 const std::vector<Tgpu*>& inputs,
                                 miopenTensorDescriptor_t outputDesc,
                                 Tcheck* outputhost,
                                 uint32_t dim)
{
    // std::vector<size_t> copySizes(inputs.size());
    size_t total_data_size{0};
    const auto t0                = std::chrono::high_resolution_clock::now();
    const auto& shape            = miopen::deref(outputDesc).GetLengths();
    size_t outer_size            = 1;
    size_t inner_size            = 1;
    const size_t output_dim_size = shape[dim];
    for(size_t i = 0; i < dim; ++i)
    {
        outer_size *= shape[i];
    }

    for(size_t i = dim + 1; i < shape.size(); ++i)
    {
        inner_size *= shape[i];
    }

    int32_t ret                                = 0;
    size_t output_start_offset                 = 0;
    const size_t inner_size_by_output_dim_size = inner_size * output_dim_size;

    for(size_t i = 0; i < inputs.size(); ++i)
    {
        const auto input                = inputs[i];
        const size_t dim_size           = miopen::deref(inputDescs[i]).GetLengths()[dim];
        const size_t copy_size          = inner_size * dim_size;
        const size_t copy_size_in_bytes = copy_size * sizeof(*outputhost);
        // copySizes[i] = copy_size;
        for(size_t o = 0; o < outer_size; ++o)
        {
            const size_t input_offset  = copy_size * o;
            const size_t output_offset = output_start_offset + (o * inner_size_by_output_dim_size);
            total_data_size += copy_size_in_bytes;
            // std::copy_n(&input[input_offset], copy_size, &outputhost[output_offset]);
            if constexpr(std::is_same_v<Tgpu, Tcheck> && std::is_trivially_copyable_v<Tgpu>)
            {
                memcpy(&outputhost[output_offset], &input[input_offset], copy_size_in_bytes);
            }
            else
            {
                for(size_t j = 0; j < copy_size; ++j)
                {
                    outputhost[output_offset + j] = input[input_offset + j];
                }
            }
        }
        output_start_offset += copy_size;
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    std::cout << "Updated CPU CAT solver: " << ns << "ns (" << (double(ns) / 1000000000.0)
              << " sec)" << std::endl;

    std::fstream of("/data/Dev/mloCatForwardRunHost_upd.txt", std::ofstream::app);
    of << ns << ";" << total_data_size << ";" << std::endl;
    // of << ns << ";";// << outer_size << ";";
    // for (const auto& s : copySizes)
    // {
    //     of << s << ";";
    // }
    // of << std::endl;

    return ret;
}

template <typename Tgpu, typename Tcheck>
int32_t mloCatForwardRunHost_upd_mt(const std::vector<miopenTensorDescriptor_t>& inputDescs,
                                    const std::vector<Tgpu*>& inputs,
                                    miopenTensorDescriptor_t outputDesc,
                                    Tcheck* outputhost,
                                    uint32_t dim)
{
    // std::vector<size_t> copySizes(inputs.size());
    const auto t0                = std::chrono::high_resolution_clock::now();
    const auto& shape            = miopen::deref(outputDesc).GetLengths();
    size_t outer_size            = 1;
    size_t inner_size            = 1;
    const size_t output_dim_size = shape[dim];
    for(size_t i = 0; i < dim; ++i)
    {
        outer_size *= shape[i];
    }

    for(size_t i = dim + 1; i < shape.size(); ++i)
    {
        inner_size *= shape[i];
    }

    int32_t ret                                = 0;
    size_t output_start_offset                 = 0;
    const size_t inner_size_by_output_dim_size = inner_size * output_dim_size;

    miopen::par_ford(inputs.size())([&](size_t i) {
        const auto input       = inputs[i];
        const size_t dim_size  = miopen::deref(inputDescs[i]).GetLengths()[dim];
        const size_t copy_size = inner_size * dim_size;
        // copySizes[i] = copy_size;
        miopen::par_ford(outer_size)([&](size_t o) {
            const size_t input_offset  = copy_size * o;
            const size_t output_offset = output_start_offset + (o * inner_size_by_output_dim_size);
            // std::copy_n(&input[input_offset], copy_size, &outputhost[output_offset]);
            for(size_t j = 0; j < copy_size; j++)
            {
                outputhost[output_offset + j] = input[input_offset + j];
            }
        });
        output_start_offset += copy_size;
    });

    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    std::cout << "Updated multi-threaded CPU CAT solver: " << ns << "ns ("
              << (double(ns) / 1000000000.0) << " sec)" << std::endl;

    std::fstream of("/data/Dev/mloCatForwardRunHost_upd_mt.txt", std::ofstream::app);
    of << ns; // << ";";// << outer_size << ";";
    // for (const auto& s : copySizes)
    // {
    //     of << s << ";";
    // }
    of << std::endl;

    return ret;
}

template <typename Tgpu, typename Tcheck>
int32_t mloCatForwardRunHost_upd_mt_2(const std::vector<miopenTensorDescriptor_t>& inputDescs,
                                      const std::vector<Tgpu*>& inputs,
                                      miopenTensorDescriptor_t outputDesc,
                                      Tcheck* outputhost,
                                      uint32_t dim)
{
    // std::vector<size_t> copySizes(inputs.size());
    const auto t0                = std::chrono::high_resolution_clock::now();
    const auto& shape            = miopen::deref(outputDesc).GetLengths();
    size_t outer_size            = 1;
    size_t inner_size            = 1;
    const size_t output_dim_size = shape[dim];
    for(size_t i = 0; i < dim; ++i)
    {
        outer_size *= shape[i];
    }

    for(size_t i = dim + 1; i < shape.size(); ++i)
    {
        inner_size *= shape[i];
    }

    int32_t ret                                = 0;
    size_t output_start_offset                 = 0;
    const size_t inner_size_by_output_dim_size = inner_size * output_dim_size;

    miopen::par_ford(inputs.size())([&](size_t i) {
        const auto input       = inputs[i];
        const size_t dim_size  = miopen::deref(inputDescs[i]).GetLengths()[dim];
        const size_t copy_size = inner_size * dim_size;
        // copySizes[i] = copy_size;
        for(size_t o = 0; o < outer_size; ++o)
        {
            const size_t input_offset  = copy_size * o;
            const size_t output_offset = output_start_offset + (o * inner_size_by_output_dim_size);
            // std::copy_n(&input[input_offset], copy_size, &outputhost[output_offset]);
            for(size_t j = 0; j < copy_size; j++)
            {
                outputhost[output_offset + j] = input[input_offset + j];
            }
        }
        output_start_offset += copy_size;
    });

    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    std::cout << "Updated multi-threaded CPU CAT solver [2]: " << ns << "ns ("
              << (double(ns) / 1000000000.0) << " sec)" << std::endl;

    std::fstream of("/data/Dev/mloCatForwardRunHost_upd_mt_2.txt", std::ofstream::app);
    of << ns; // << ";";// << outer_size << ";";
    // for (const auto& s : copySizes)
    // {
    //     of << s << ";";
    // }
    of << std::endl;

    return ret;
}

template <typename Tgpu, typename Tcheck>
int32_t mloCatForwardRunHost_upd_2(std::vector<miopenTensorDescriptor_t> inputDescs,
                                   std::vector<Tgpu*> inputs,
                                   miopenTensorDescriptor_t outputDesc,
                                   Tcheck* outputhost,
                                   uint32_t dim)
{
    // std::vector<size_t> copySizes(inputs.size());
    const auto t0 = std::chrono::high_resolution_clock::now();
    // Estrai shape e calcola outer/inner come nell'originale
    auto shape             = miopen::deref(outputDesc).GetLengths();
    size_t outer_size      = 1;
    size_t inner_size      = 1;
    size_t output_dim_size = shape[dim];
    for(size_t i = 0; i < dim; i++)
    {
        outer_size *= shape[i];
    }

    for(size_t i = dim + 1; i < shape.size(); i++)
    {
        inner_size *= shape[i];
    }

    int32_t ret                = 0;
    size_t output_start_offset = 0;

    // Per ogni input, copiamo blocchi contigui:
    // - se Tgpu == Tcheck possiamo memcpyare whole-blocks (più veloce)
    // - altrimenti facciamo una copia elemento-per-elemento (possibile parallelizzazione)
    for(size_t i = 0; i < inputs.size(); i++)
    {
        const Tgpu* input = inputs[i];
        size_t dim_size   = miopen::deref(inputDescs[i]).GetLengths()[dim];
        size_t copy_elems = inner_size * dim_size; // elementi da copiare per ogni 'outer' slice
        size_t copy_bytes = copy_elems * sizeof(Tgpu);
        // copySizes[i] = copy_elems;

        // stride (in elementi) nell'output per la dimensione dim totale
        size_t output_stride_elems = inner_size * output_dim_size;

        // Caso ottimizzato: same type -> memcpy per blocco
        if(std::is_same<Tgpu, Tcheck>::value)
        {
            for(size_t o = 0; o < outer_size; ++o)
            {
                size_t out_offset = output_start_offset + (o * output_stride_elems);
                const void* src   = static_cast<const void*>(input + (o * copy_elems));
                void* dst         = static_cast<void*>(outputhost + out_offset);
                // copia contigua di copy_bytes
                std::memcpy(dst, src, copy_bytes);
            }
        }
        else
        {
            // I tipi sono diversi: dobbiamo convertire elemento-per-elemento.
            for(size_t o = 0; o < outer_size; ++o)
            {
                size_t out_offset = output_start_offset + (o * output_stride_elems);
                const Tgpu* src   = input + (o * copy_elems);
                Tcheck* dst       = outputhost + out_offset;
                // conversione elemento-per-elemento
                for(size_t j = 0; j < copy_elems; ++j)
                {
                    dst[j] = static_cast<Tcheck>(src[j]);
                }
            }
        }

        output_start_offset += copy_elems;
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    std::cout << "Updated CPU CAT solver [2]: " << ns << "ns (" << (double(ns) / 1000000000.0)
              << " sec)" << std::endl;

    std::fstream of("/data/Dev/mloCatForwardRunHost_upd_2.txt", std::ofstream::app);
    of << ns; // << ";";// << outer_size << ";";
    // for (const auto& s : copySizes)
    // {
    //     of << s << ";";
    // }
    of << std::endl;

    return ret;
}

template <typename Tgpu, typename Tcheck>
int32_t mloCatForwardRunHost_upd_3(std::vector<miopenTensorDescriptor_t> inputDescs,
                                   std::vector<Tgpu*> inputs,
                                   miopenTensorDescriptor_t outputDesc,
                                   Tcheck* __restrict outputhost,
                                   uint32_t dim)
{
    // std::vector<size_t> copySizes(inputs.size());
    const auto t0 = std::chrono::high_resolution_clock::now();

    const auto& shape = miopen::deref(outputDesc).GetLengths();

    // Calcola outer e inner size una sola volta
    size_t outer_size = 1;
    size_t inner_size = 1;
    const size_t rank = shape.size();
    for(size_t i = 0; i < dim; ++i)
        outer_size *= shape[i];
    for(size_t i = dim + 1; i < rank; ++i)
        inner_size *= shape[i];

    const size_t output_dim_size = shape[dim];
    const size_t output_stride   = inner_size * output_dim_size;

    size_t output_start_offset = 0;
    int32_t ret                = 0;

    // Scorri tutti gli input concatenati
    for(size_t i = 0; i < inputs.size(); ++i)
    {
        const Tgpu* __restrict input = inputs[i];
        const auto& in_shape         = miopen::deref(inputDescs[i]).GetLengths();
        const size_t dim_size        = in_shape[dim];
        const size_t copy_elems      = inner_size * dim_size;
        const size_t copy_bytes      = copy_elems * sizeof(Tgpu);
        // copySizes[i] = copy_elems;

        // Percorso veloce: stesso tipo → blocchi contigui, memcpy
        if constexpr(std::is_same<Tgpu, Tcheck>::value)
        {
            for(size_t o = 0; o < outer_size; ++o)
            {
                const Tgpu* src = input + o * copy_elems;
                Tcheck* dst     = outputhost + output_start_offset + o * output_stride;
                std::memcpy(dst, src, copy_bytes);
            }
        }
        else
        {
            // Percorso conversione: blocchi manuali per locality e prefetch
            constexpr size_t BLOCK = 16; // tuning: 8–32 va bene per cache L1/L2
            for(size_t o = 0; o < outer_size; ++o)
            {
                const Tgpu* src = input + o * copy_elems;
                Tcheck* dst     = outputhost + output_start_offset + o * output_stride;

                size_t j           = 0;
                const size_t limit = copy_elems - (copy_elems % BLOCK);

                // copia a blocchi
                for(; j < limit; j += BLOCK)
                {
                    // Prefetch futura cache line (aiuta per grandi buffer)
                    __builtin_prefetch(src + j + 64, 0, 0);

// copia blocco manuale
#pragma unroll
                    for(size_t k = 0; k < BLOCK; ++k)
                        dst[j + k] = static_cast<Tcheck>(src[j + k]);
                }
                // copia rimanente
                for(; j < copy_elems; ++j)
                    dst[j] = static_cast<Tcheck>(src[j]);
            }
        }

        output_start_offset += copy_elems;
    }

    const auto t1 = std::chrono::high_resolution_clock::now();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();

    std::cout << "Updated CPU CAT solver [3]: " << ns << "ns (" << (double(ns) / 1000000000.0)
              << " sec)" << std::endl;

    std::fstream of("/data/Dev/mloCatForwardRunHost_upd_3.txt", std::ofstream::app);
    of << ns; // << ";";// << outer_size << ";";
    // for (const auto& s : copySizes)
    // {
    //     of << s << ";";
    // }
    of << std::endl;

    return ret;
}

#endif

template <typename Tgpu, typename Tref = Tgpu>
class CatDriver : public Driver
{
public:
    CatDriver() : Driver()
    {
        miopenCreateTensorDescriptor(&outputDesc);

        data_type = miopen_type<Tgpu>{};
    }

    int AddCmdLineArgs() override;
    int ParseCmdLineArgs(int argc, char* argv[]) override;
    InputFlags& GetInputFlags() override { return inflags; }

    int GetandSetData() override;
    std::vector<std::vector<int>> GetInputTensorLengthsFromCmdLine();

    int AllocateBuffersAndCopy() override;

    int RunForwardGPU() override;
    int RunForwardCPU();

    int RunBackwardGPU() override;

    int VerifyBackward() override;
    int VerifyForward() override;
    ~CatDriver() override
    {
        for(auto inputDesc : inputDescs)
        {
            miopenDestroyTensorDescriptor(inputDesc);
        }
        miopenDestroyTensorDescriptor(outputDesc);
    }

private:
    InputFlags inflags;
    uint32_t dim;

    std::vector<miopenTensorDescriptor_t> inputDescs;
    miopenTensorDescriptor_t outputDesc;

    std::vector<std::unique_ptr<GPUMem>> in_devs;
    std::unique_ptr<GPUMem> out_dev;

    std::vector<std::vector<Tgpu>> ins;
    std::vector<Tgpu> out;
    std::vector<Tref> outhost;

    std::vector<Tref> outhost_upd;
    std::vector<Tref> outhost_upd_2;
    std::vector<Tref> outhost_upd_3;
    std::vector<Tref> outhost_upd_mt;
    std::vector<Tref> outhost_upd_mt_2;

    std::vector<void*> in_devs_ptr;
    std::vector<Tgpu*> ins_ptr;

    bool use_multithread = false;
};

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::ParseCmdLineArgs(int argc, char* argv[])
{
    inflags.Parse(argc, argv);

    if(inflags.GetValueInt("time") == 1)
    {
        miopenEnableProfiling(GetHandle(), true);
    }
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::GetandSetData()
{
    miopenTensorDescriptor_t inputDesc;
    size_t output_dim_size = 0;
    auto in_lens           = GetInputTensorLengthsFromCmdLine();
    dim                    = inflags.GetValueInt("dim");
    use_multithread        = (inflags.GetValueInt("mt") != 0);

    // int64_t t = 0;

    for(auto in_len : in_lens)
    {
        miopenCreateTensorDescriptor(&inputDesc);
        SetTensorNd(inputDesc, in_len, data_type);
        inputDescs.push_back(inputDesc);
        output_dim_size += in_len[dim];

        // int64_t s = 1;

        // std::cout << "in_len: ";

        // for (const auto i : in_len)
        // {
        //     s *= int64_t(i);
        //     std::cout << i << " ";
        // }

        // t += s;

        // std::cout << "-> " << s << std::endl;
    }
    // std::cout << "Tot: " << t << std::endl;
    auto out_len = in_lens[0];
    out_len[dim] = output_dim_size;

    // std::cout << "out_len: ";

    // for (const auto i : out_len)
    // {
    //     std::cout << i << " ";
    // }

    // std::cout << std::endl;

    SetTensorNd(outputDesc, out_len, data_type);

    return 0;
}

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::AddCmdLineArgs()
{
    inflags.AddInputFlag("forw", 'F', "1", "Run only Forward Cat (Default=1)", "int");
    inflags.AddTensorFlag("input1", '1', "2x32x128x128x128", "input1 tensor descriptor");
    inflags.AddTensorFlag("input2", '2', "2x32x128x128x128", "input2 tensor descriptor");
    inflags.AddTensorFlag("input3", '3', "", "input3 tensor descriptor");
    inflags.AddTensorFlag("input4", '4', "", "input4 tensor descriptor");
    inflags.AddTensorFlag("input5", '5', "", "input5 tensor descriptor");
    inflags.AddTensorFlag("input6", '6', "", "input6 tensor descriptor");
    inflags.AddTensorFlag("input7", '7', "", "input7 tensor descriptor");
    inflags.AddTensorFlag("input8", '8', "", "input8 tensor descriptor");
    inflags.AddInputFlag("dim", 'd', "0", "Concatenation dimension (Default=0)", "int");

    inflags.AddInputFlag("iter", 'i', "10", "Number of Iterations (Default=10)", "int");
    inflags.AddInputFlag("verify", 'V', "1", "Verify Each Layer (Default=1)", "int");
    inflags.AddInputFlag("time", 't', "0", "Time Each Layer (Default=0)", "int");
    inflags.AddInputFlag(
        "wall", 'w', "0", "Wall-clock Time Each Layer, Requires time == 1 (Default=0)", "int");

    inflags.AddInputFlag("mt", 'M', "0", "Use multithreaded version (Default=0)", "int");

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
std::vector<std::vector<int>> CatDriver<Tgpu, Tref>::GetInputTensorLengthsFromCmdLine()
{
    const int max_input_count = 8;
    std::vector<std::vector<int>> ret;
    std::string name = "input";
    for(int i = 1; i < max_input_count; i++)
    {
        auto tensor = inflags.GetValueTensor(name + std::to_string(i));
        if(!tensor.lengths.empty())
            ret.push_back(tensor.lengths);
    }
    return ret;
}

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::AllocateBuffersAndCopy()
{
    uint32_t ctx = 0;
    for(auto& inputDesc : inputDescs)
    {
        auto in_sz = GetTensorSize(inputDesc);
        in_devs.push_back(std::make_unique<GPUMem>(ctx, in_sz, sizeof(Tgpu)));
        ins.push_back(std::vector<Tgpu>(in_sz, static_cast<Tgpu>(0)));
        auto& in    = ins.back();
        auto in_dev = in_devs.back().get();

        for(int i = 0; i < in_sz; i++)
        {
            in[i] = prng::gen_A_to_B<Tgpu>(static_cast<Tgpu>(0.0), static_cast<Tgpu>(1.0));
        }
        if(in_dev->ToGPU(GetStream(), in.data()) != 0)
            std::cerr << "Error copying (in) to GPU, size: " << in_dev->GetSize() << std::endl;
        in_devs_ptr.push_back(in_dev->GetMem());
        ins_ptr.push_back(in.data());
    }

    size_t out_sz = GetTensorSize(outputDesc);

    out_dev = std::unique_ptr<GPUMem>(new GPUMem(ctx, out_sz, sizeof(Tgpu)));
    out     = std::vector<Tgpu>(out_sz, static_cast<Tgpu>(0));
    outhost = std::vector<Tref>(out_sz, static_cast<Tref>(0));

    outhost_upd      = std::vector<Tref>(out_sz, static_cast<Tref>(0));
    outhost_upd_2    = std::vector<Tref>(out_sz, static_cast<Tref>(0));
    outhost_upd_3    = std::vector<Tref>(out_sz, static_cast<Tref>(0));
    outhost_upd_mt   = std::vector<Tref>(out_sz, static_cast<Tref>(0));
    outhost_upd_mt_2 = std::vector<Tref>(out_sz, static_cast<Tref>(0));

    if(out_dev->ToGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out) to GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::RunForwardGPU()
{
    float kernel_total_time = 0;
    float kernel_first_time = 0;

    Timer t;
    START_TIME

    for(int i = 0; i < inflags.GetValueInt("iter"); i++)
    {
        miopenCatForward(GetHandle(),
                         inputDescs.size(),
                         inputDescs.data(),
                         in_devs_ptr.data(),
                         outputDesc,
                         out_dev->GetMem(),
                         dim);

        float time = 0.0;
        miopenGetKernelTime(GetHandle(), &time);
        kernel_total_time += time;
        if(i == 0)
            kernel_first_time = time;
    }

    if(inflags.GetValueInt("time") == 1)
    {
        STOP_TIME
        int iter = inflags.GetValueInt("iter");
        if(WALL_CLOCK)
            printf("Wall-clock Time Forward Cat Elapsed: %f ms\n", t.gettime_ms() / iter);

        float kernel_average_time =
            iter > 1 ? (kernel_total_time - kernel_first_time) / (iter - 1) : kernel_first_time;
        printf("GPU Kernel Time Forward Cat Elapsed: %f ms\n", kernel_average_time);
    }

    if(out_dev->FromGPU(GetStream(), out.data()) != 0)
        std::cerr << "Error copying (out_dev) from GPU, size: " << out_dev->GetSize() << std::endl;

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::RunForwardCPU()
{
    // if constexpr (std::is_same_v<Tgpu, Tref> && std::is_trivially_copyable_v<Tgpu>)
    // {
    //     for (size_t i : {10, 100, 1000, 5000, 10000, 50000, 100000, 1000000, 471954865})
    //     {
    //         std::vector<Tgpu> vSrc(i);
    //         std::vector<Tref> vDst(i);
    //         const size_t byteSize = i * sizeof(vDst[0]);
    //         const auto* pSrc = vSrc.data();
    //         auto* pDst = vDst.data();

    //         const auto t0 = std::chrono::high_resolution_clock::now();

    //         for (size_t a = 0; a < ins_ptr.size(); ++a)
    //         {
    //             memcpy(pDst, pSrc, byteSize);
    //         }

    //         const auto t1 = std::chrono::high_resolution_clock::now();

    //         vSrc = std::vector<Tgpu>(i);
    //         vDst = std::vector<Tref>(i);
    //         pSrc = vSrc.data();
    //         pDst = vDst.data();

    //         const auto t2 = std::chrono::high_resolution_clock::now();

    //         for (size_t a = 0; a < ins_ptr.size(); ++a)
    //         {
    //             for (auto j = 0; j < i; ++j)
    //             {
    //                 pDst[j] = pSrc[j];
    //             }
    //         }

    //         const auto t3 = std::chrono::high_resolution_clock::now();

    //         const auto nsMemcpy = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 -
    //         t0).count(); const auto nsFor =
    //         std::chrono::duration_cast<std::chrono::nanoseconds>(t3 - t2).count();

    //         std::cout
    //             << "Copia di " << i << " elementi:\n"
    //             << "    memcpy: " << nsMemcpy << "ns (" << (double(nsMemcpy) / 1000000000.0) <<
    //             "s)\n"
    //             << "    for: " << nsFor << "ns (" << (double(nsFor) / 1000000000.0) << "s)\n"
    //             << std::endl
    //             ;
    //     }
    // }

    mloCatForwardRunHost<Tgpu, Tref>(inputDescs, ins_ptr, outputDesc, outhost.data(), dim);

    mloCatForwardRunHost_upd<Tgpu, Tref>(inputDescs, ins_ptr, outputDesc, outhost_upd.data(), dim);

    mloCatForwardRunHost_upd_2<Tgpu, Tref>(
        inputDescs, ins_ptr, outputDesc, outhost_upd_2.data(), dim);

    mloCatForwardRunHost_upd_3<Tgpu, Tref>(
        inputDescs, ins_ptr, outputDesc, outhost_upd_3.data(), dim);

    mloCatForwardRunHost_upd_mt<Tgpu, Tref>(
        inputDescs, ins_ptr, outputDesc, outhost_upd_mt.data(), dim);

    mloCatForwardRunHost_upd_mt_2<Tgpu, Tref>(
        inputDescs, ins_ptr, outputDesc, outhost_upd_mt_2.data(), dim);

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::RunBackwardGPU()
{
    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::VerifyForward()
{
    RunForwardCPU();
    const auto error          = miopen::rms_range(outhost, out);
    const auto error_upd      = miopen::rms_range(outhost_upd, out);
    const auto error_upd_2    = miopen::rms_range(outhost_upd_2, out);
    const auto error_upd_3    = miopen::rms_range(outhost_upd_3, out);
    const auto error_upd_mt   = miopen::rms_range(outhost_upd_mt, out);
    const auto error_upd_mt_2 = miopen::rms_range(outhost_upd_mt_2, out);

    if(!std::isfinite(error) || error != 0)
    {
        std::cout << "Forward Cat FAILED against original CPU reference: " << error << " > 0"
                  << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Cat Verifies OK on CPU reference" << std::endl;
    }

    if(!std::isfinite(error_upd) || error_upd != 0)
    {
        std::cout << "Forward Cat FAILED against updated CPU reference: " << error_upd << " > 0"
                  << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Cat Verifies OK on updated CPU reference" << std::endl;
    }

    if(!std::isfinite(error_upd_2) || error_upd_2 != 0)
    {
        std::cout << "Forward Cat FAILED against updated CPU reference [2]: " << error_upd_2
                  << " > 0" << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Cat Verifies OK on updated CPU reference [2]" << std::endl;
    }

    if(!std::isfinite(error_upd_3) || error_upd_3 != 0)
    {
        std::cout << "Forward Cat FAILED against updated CPU reference [3]: " << error_upd_2
                  << " > 0" << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Cat Verifies OK on updated CPU reference [3]" << std::endl;
    }

    if(!std::isfinite(error_upd_mt) || error_upd_mt != 0)
    {
        std::cout << "Forward Cat FAILED against updated multi-threaded CPU reference: "
                  << error_upd_mt << " > 0" << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Cat Verifies OK on updated multi-threaded CPU reference" << std::endl;
    }

    if(!std::isfinite(error_upd_mt_2) || error_upd_mt_2 != 0)
    {
        std::cout << "Forward Cat FAILED against updated multi-threaded CPU reference [2]: "
                  << error_upd_mt_2 << " > 0" << std::endl;
        return EC_VerifyFwd;
    }
    else
    {
        std::cout << "Forward Cat Verifies OK on updated multi-threaded CPU reference [2]"
                  << std::endl;
    }

    return miopenStatusSuccess;
}

template <typename Tgpu, typename Tref>
int CatDriver<Tgpu, Tref>::VerifyBackward()
{
    return miopenStatusSuccess;
}

#endif // GUARD_MIOPEN_CAT_DRIVER_HPP
