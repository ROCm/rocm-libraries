// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <iostream>
#include <fstream>
#include <iomanip>
#include <functional>

#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/batched_contraction.hpp"
#include "ck_tile/host/reference/reference_batched_contraction.hpp"
#include "contraction_multi_abd_benchmark.hpp"

class ContractionMultiABDProfiler
{
    public:
    using ContractionInstance = KernelInstance<ContractionMultiABDProblem>;

    static ContractionMultiABDProfiler& instance(Settings setting)
    {
        static ContractionMultiABDProfiler inst{setting};
        return inst;
    }

    void benchmark(
        ContractionMultiABDProblem& problem,
        std::function<float(
            const ck_tile::BatchedContractionMultiABDHostArgs<NumATensor, NumBTensor, NumDTensor>&,
            const ck_tile::stream_config&)> kernel_func)
    {
        const auto g_dims = problem.g_dims_;
        const auto m_dims = problem.m_dims_;
        const auto n_dims = problem.n_dims_;
        const auto k_dims = problem.k_dims_;

        const auto g_total = problem.g_total_;
        const auto m_total = problem.m_total_;
        const auto n_total = problem.n_total_;
        const auto k_total = problem.k_total_;

        // A dims: [G..., M..., K...]
        auto a_dims    = concatenate_dims({g_dims, m_dims, k_dims});
        auto a_strides = compute_row_major_strides(a_dims);

        // B dims: [G..., N..., K...]
        auto b_dims    = concatenate_dims({g_dims, n_dims, k_dims});
        auto b_strides = compute_row_major_strides(b_dims);

        // D/E dims: [G..., M..., N...]
        auto e_dims    = concatenate_dims({g_dims, m_dims, n_dims});
        auto e_strides = compute_row_major_strides(e_dims);

        const auto total_a = g_total * m_total * k_total;
        const auto total_b = g_total * n_total * k_total;
        const auto total_e = g_total * m_total * n_total;

        // Allocate host tensors -- use flat 1D descriptors for simplicity
        ck_tile::HostTensor<EDataType> a0_host({total_a});
        ck_tile::HostTensor<EDataType> a1_host({total_a});
        ck_tile::HostTensor<EDataType> b0_host({total_b});
        ck_tile::HostTensor<EDataType> d0_host({total_e});
        ck_tile::HostTensor<EDataType> e_host_dev_result({total_e});

        ck_tile::FillUniformDistribution<EDataType>{-5.f, 5.f}(a0_host);
        ck_tile::FillUniformDistribution<EDataType>{-5.f, 5.f}(a1_host);
        ck_tile::FillUniformDistribution<EDataType>{-5.f, 5.f}(b0_host);
        ck_tile::FillUniformDistribution<EDataType>{-1.f, 1.f}(d0_host);

        ck_tile::DeviceMem a0_dev(a0_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem a1_dev(a1_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b0_dev(b0_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem d0_dev(d0_host.get_element_space_size_in_bytes());
        ck_tile::DeviceMem e_dev(e_host_dev_result.get_element_space_size_in_bytes());

        a0_dev.ToDevice(a0_host.mData.data());
        a1_dev.ToDevice(a1_host.mData.data());
        b0_dev.ToDevice(b0_host.mData.data());
        d0_dev.ToDevice(d0_host.mData.data());
        e_dev.SetZero();
        e_host_dev_result.SetZero();

        ck_tile::BatchedContractionMultiABDHostArgs<NumATensor, NumBTensor, NumDTensor> host_args;
        host_args.as_ptr  = {a0_dev.GetDeviceBuffer(), a1_dev.GetDeviceBuffer()};
        host_args.bs_ptr  = {b0_dev.GetDeviceBuffer()};
        host_args.ds_ptr  = {d0_dev.GetDeviceBuffer()};
        host_args.e_ptr   = e_dev.GetDeviceBuffer();
        host_args.k_batch = 1;

        for(ck_tile::index_t i = 0; i < NumATensor; ++i)
        {
            host_args.As_dims[i]    = a_dims;
            host_args.As_strides[i] = a_strides;
        }
        for(ck_tile::index_t i = 0; i < NumBTensor; ++i)
        {
            host_args.Bs_dims[i]    = b_dims;
            host_args.Bs_strides[i] = b_strides;
        }
        for(ck_tile::index_t i = 0; i < NumDTensor; ++i)
        {
            host_args.Ds_dims[i]    = e_dims;
            host_args.Ds_strides[i] = e_strides;
        }
        host_args.E_dims    = e_dims;
        host_args.E_strides = e_strides;

        float avg_time =
            kernel_func(host_args,
                        ck_tile::stream_config{
                            nullptr, true, setting_.log, setting_.n_warmup, setting_.n_repeat});

        ContractionInstance ki{std::string(KERNEL_NAME), problem, {-1.0, -1.0, -1.0}};

        std::size_t flop     = std::size_t(2) * g_total * m_total * n_total * k_total;
        std::size_t num_byte = sizeof(EDataType) * (NumATensor * total_a + NumBTensor * total_b +
                                                    NumDTensor * total_e + total_e);

        ki.perf_result_.latency_   = avg_time;
        ki.perf_result_.tflops_    = static_cast<float>(flop) / 1.E9 / avg_time;
        ki.perf_result_.bandwidth_ = num_byte / 1.E6 / avg_time;

        if(setting_.log && !setting_.json_output)
        {
            std::cout << ki << std::endl;
        }

        // Verify if requested
        if(setting_.verify)
        {
            e_dev.FromDevice(e_host_dev_result.mData.data());

            // Simplified verification -- just check for NaN/Inf
            bool pass = true;
            for(auto val : e_host_dev_result.mData)
            {
                float fv = ck_tile::type_convert<float>(val);
                if(std::isnan(fv) || std::isinf(fv))
                {
                    pass = false;
                    break;
                }
            }

            if(!pass)
            {
                std::cout << "Verification failed for kernel: " << KERNEL_NAME << std::endl;
            }
            else
            {
                kernel_instances_.emplace_back(ki);
            }
        }
        else
        {
            kernel_instances_.emplace_back(ki);
        }
    }

    ContractionInstance select_best_instance(Metric metric)
    {
        if(kernel_instances_.empty())
            throw std::runtime_error("Empty instances");

        auto ki = *std::max_element(kernel_instances_.begin(),
                                    kernel_instances_.end(),
                                    [metric](const auto& a, const auto& b) {
                                        return PerformanceResult::compare(
                                            b.perf_result_, a.perf_result_, metric);
                                    });

        if(setting_.json_output)
        {
            std::cout << ki << std::endl;
        }
        else
        {
            std::cout << "**********************************" << std::endl;
            std::cout << "According to given metrics: " << get_metric_name(metric) << "\n"
                      << "Current kernel performance is: " << ki << std::endl;
            std::cout << "**********************************" << std::endl;
        }

        if(!setting_.csv_filename.empty())
        {
            std::ofstream file(setting_.csv_filename + ".csv", std::ios::app);
            if(file.is_open())
            {
                if(file.tellp() == 0)
                {
                    file << "rocm_version,device_name," << "g_total,m_total,n_total,k_total,"
                         << "name,latency(ms),tflops(TFlops),bandwidth(GB/s),metric\n";
                }

                const auto& p    = ki.problem_;
                const auto& perf = ki.perf_result_;

                file << get_rocm_version() << "," << ck_tile::get_device_name() << "," << p.g_total_
                     << "," << p.m_total_ << "," << p.n_total_ << "," << p.k_total_ << ","
                     << ki.name_ << "," << std::fixed << std::setprecision(4) << perf.latency_
                     << "," << perf.tflops_ << "," << perf.bandwidth_ << ","
                     << get_metric_name(metric) << "\n";
            }
        }

        return ki;
    }

    ContractionMultiABDProfiler(const ContractionMultiABDProfiler&)            = delete;
    ContractionMultiABDProfiler& operator=(const ContractionMultiABDProfiler&) = delete;

    private:
    ~ContractionMultiABDProfiler() { kernel_instances_.clear(); }
    ContractionMultiABDProfiler(Settings setting) : setting_(setting) {}

    Settings setting_;
    std::vector<ContractionInstance> kernel_instances_;
};
