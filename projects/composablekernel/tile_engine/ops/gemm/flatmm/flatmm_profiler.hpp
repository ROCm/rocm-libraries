// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <fstream>
#include <iomanip>
#include <iostream>

#include "ck_tile/host/device_prop.hpp"
#include "ck_tile/ops/flatmm.hpp"
#include "flatmm_benchmark.hpp"

class FlatmmProfiler
{
    public:
    static FlatmmProfiler& instance(Setting setting)
    {
        static FlatmmProfiler profiler{setting};
        return profiler;
    }

    void benchmark(FlatmmProblem& flatmm_problem,
                   std::function<float(const ck_tile::FlatmmHostArgs<>&,
                                       const ck_tile::stream_config&)> kernel_func,
                   KernelConfig& config)
    {
        std::vector<std::function<std::tuple<std::string, float>(ck_tile::FlatmmHostArgs<>&,
                                                                 const ck_tile::stream_config&)>>
            callables;

        callables.push_back(
            [kernel_func](ck_tile::FlatmmHostArgs<>& args, const ck_tile::stream_config& stream) {
                const float time = kernel_func(args, stream);
                return std::make_tuple(std::string(KERNEL_NAME), time);
            });

        benchmark(flatmm_problem, callables, config);
    }

    void benchmark(FlatmmProblem& flatmm_problem,
                   std::vector<std::function<std::tuple<std::string, float>(
                       ck_tile::FlatmmHostArgs<>&, const ck_tile::stream_config&)>>& callables,
                   KernelConfig& config)
    {
        const ALayout layout_a = ALayout{};
        const BLayout layout_b = BLayout{};
        const CLayout layout_c = CLayout{};

        flatmm_problem.stride_a_ = ck_tile::get_default_stride(
            flatmm_problem.m_, flatmm_problem.k_, flatmm_problem.stride_a_, is_row_major(layout_a));
        flatmm_problem.stride_b_ = ck_tile::get_default_stride(
            flatmm_problem.k_, flatmm_problem.n_, flatmm_problem.stride_b_, is_row_major(layout_b));
        flatmm_problem.stride_c_ = ck_tile::get_default_stride(
            flatmm_problem.m_, flatmm_problem.n_, flatmm_problem.stride_c_, is_row_major(layout_c));

        ck_tile::HostTensor<ADataType> a_m_k(
            ck_tile::host_tensor_descriptor(flatmm_problem.m_,
                                            flatmm_problem.k_,
                                            flatmm_problem.stride_a_,
                                            is_row_major(layout_a)));
        ck_tile::HostTensor<BDataType> b_k_n(
            ck_tile::host_tensor_descriptor(flatmm_problem.k_,
                                            flatmm_problem.n_,
                                            flatmm_problem.stride_b_,
                                            is_row_major(layout_b)));
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(
            ck_tile::host_tensor_descriptor(flatmm_problem.m_,
                                            flatmm_problem.n_,
                                            flatmm_problem.stride_c_,
                                            is_row_major(layout_c)));

        if(setting_.init_method_ == 0)
        {
            ck_tile::FillUniformDistribution<ADataType>{0.0f, 1.0f}(a_m_k);
            ck_tile::FillUniformDistribution<BDataType>{-0.5f, 0.5f}(b_k_n);
        }
        else if(setting_.init_method_ == 1)
        {
            ck_tile::FillMonotonicSeq<ADataType>{}(a_m_k);
            ck_tile::FillMonotonicSeq<BDataType>{}(b_k_n);
        }
        else if(setting_.init_method_ == 2)
        {
            ck_tile::FillConstant<ADataType>{static_cast<ADataType>(1)}(a_m_k);
            ck_tile::FillConstant<BDataType>{static_cast<BDataType>(1)}(b_k_n);
        }
        else
        {
            a_m_k.SetZero();
            b_k_n.SetZero();
        }

        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

        a_m_k_dev_buf.ToDevice(a_m_k.data());
        c_m_n_dev_buf.SetZero();
        c_m_n_dev_result.SetZero();

        ck_tile::HostTensor<CDataType> c_m_n_ref(
            ck_tile::host_tensor_descriptor(flatmm_problem.m_,
                                            flatmm_problem.n_,
                                            flatmm_problem.stride_c_,
                                            is_row_major(layout_c)));
        c_m_n_ref.SetZero();

        if(setting_.verify_)
        {
            flatmm_host_reference(setting_.verify_,
                                  a_m_k,
                                  b_k_n,
                                  c_m_n_ref,
                                  a_m_k_dev_buf,
                                  b_k_n_dev_buf,
                                  flatmm_problem.m_,
                                  flatmm_problem.n_,
                                  flatmm_problem.k_,
                                  flatmm_problem.stride_a_,
                                  flatmm_problem.stride_b_,
                                  flatmm_problem.stride_c_);
        }

        for(const auto& callable : callables)
        {
            const auto [tile_m, tile_n, tile_k]                = config.tile_dims;
            const auto [warp_m, warp_n, warp_k]                = config.warp_dims;
            const auto [warp_tile_m, warp_tile_n, warp_tile_k] = config.warp_tile_dims;
            ck_tile::ignore                                    = tile_m;
            ck_tile::ignore                                    = tile_k;
            ck_tile::ignore                                    = warp_m;
            ck_tile::ignore                                    = warp_k;
            ck_tile::ignore                                    = warp_tile_m;

            ck_tile::HostTensor<BDataType> b_shuffle_host = [&]() {
                if(config.permute_n)
                {
                    return shuffle_b_v1(b_k_n, tile_n, warp_n, warp_tile_n, warp_tile_k);
                }
                else
                {
                    return shuffle_b_v0(b_k_n, warp_tile_n, warp_tile_k);
                }
            }();

            ck_tile::DeviceMem b_shuffle_dev_buf(b_shuffle_host.get_element_space_size_in_bytes());
            b_shuffle_dev_buf.ToDevice(b_shuffle_host.data());

            ck_tile::FlatmmHostArgs<> flatmm_args = {a_m_k_dev_buf.GetDeviceBuffer(),
                                                     b_shuffle_dev_buf.GetDeviceBuffer(),
                                                     {},
                                                     c_m_n_dev_buf.GetDeviceBuffer(),
                                                     flatmm_problem.split_k_,
                                                     flatmm_problem.m_,
                                                     flatmm_problem.n_,
                                                     flatmm_problem.k_,
                                                     flatmm_problem.stride_a_,
                                                     flatmm_problem.stride_b_,
                                                     {},
                                                     flatmm_problem.stride_c_};

            auto kernel_run_result = callable(flatmm_args,
                                              ck_tile::stream_config{nullptr,
                                                                     true,
                                                                     setting_.log_,
                                                                     setting_.n_warmup_,
                                                                     setting_.n_repeat_,
                                                                     setting_.is_gpu_timer_,
                                                                     setting_.flush_cache_,
                                                                     setting_.rotating_count_});

            process_result(
                flatmm_problem, c_m_n_dev_buf, c_m_n_ref, c_m_n_dev_result, kernel_run_result);
        }
    }

    void process_result(const FlatmmProblem& flatmm_problem,
                        ck_tile::DeviceMem& c_m_n_dev_buf,
                        ck_tile::HostTensor<CDataType>& c_m_n_ref,
                        ck_tile::HostTensor<CDataType>& c_m_n_dev_result,
                        const std::tuple<std::string, float>& kernel_run_result)
    {
        auto [name, avg_time] = kernel_run_result;
        KernelInstance kernel_instance{name, flatmm_problem, {-1.0f, -1.0f, -1.0f}};

        const std::size_t flop =
            std::size_t(2) * flatmm_problem.m_ * flatmm_problem.n_ * flatmm_problem.k_;
        const std::size_t num_byte = sizeof(ADataType) * flatmm_problem.m_ * flatmm_problem.k_ +
                                     sizeof(BDataType) * flatmm_problem.n_ * flatmm_problem.k_ +
                                     sizeof(CDataType) * flatmm_problem.m_ * flatmm_problem.n_;

        kernel_instance.perf_result_.latency_   = avg_time;
        kernel_instance.perf_result_.tflops_    = static_cast<float>(flop) / 1.E9 / avg_time;
        kernel_instance.perf_result_.bandwidth_ = num_byte / 1.E6 / avg_time;

        if(setting_.log_ > 0 && !setting_.json_output_)
        {
            std::cout << kernel_instance << std::endl;
        }

        c_m_n_dev_buf.FromDevice(c_m_n_dev_result.data());
        const bool verified_correct =
            !setting_.verify_ ||
            compare(name, flatmm_problem.k_, flatmm_problem.split_k_, c_m_n_dev_result, c_m_n_ref);

        if(verified_correct)
        {
            kernel_instances_.emplace_back(kernel_instance);
        }
        else
        {
            std::cout << "Verification failed, skip kernel: " << name << std::endl;
        }

        c_m_n_dev_buf.SetZero();
        c_m_n_dev_result.SetZero();
    }

    KernelInstance select_best_instance(Metric metric)
    {
        if(kernel_instances_.empty())
        {
            throw std::runtime_error("Empty instances");
        }

        auto kernel_instance = *std::max_element(
            kernel_instances_.begin(),
            kernel_instances_.end(),
            [metric](const auto& lhs, const auto& rhs) {
                return PerformanceResult::compare(rhs.perf_result_, lhs.perf_result_, metric);
            });

        if(setting_.json_output_)
        {
            std::cout << kernel_instance << std::endl;
        }
        else
        {
            std::cout << "**********************************" << std::endl;
            std::cout << "According to given metric: " << get_metric_name(metric) << "\n"
                      << "Current kernel performance is: " << kernel_instance << std::endl;
            std::cout << "**********************************" << std::endl;
        }

        if(!setting_.csv_filename_.empty())
        {
            std::ofstream file(setting_.csv_filename_ + ".csv", std::ios::app);
            if(!file.is_open())
            {
                std::cerr << "Warning: Failed to open CSV file for writing." << std::endl;
            }
            else
            {
                if(file.tellp() == 0)
                {
                    file << "rocm_version,device_name,"
                         << "split_k,m,n,k,stride_a,stride_b,stride_c,"
                         << "dtype_a,dtype_b,dtype_acc,dtype_c," << "layout_a,layout_b,layout_c,"
                         << "name,latency(ms),tflops(TFlops),bandwidth(GB/s),metric\n";
                }

                const auto& problem = kernel_instance.problem_;
                const auto& perf    = kernel_instance.perf_result_;

                file << get_rocm_version() << "," << ck_tile::get_device_name() << ","
                     << problem.split_k_ << "," << problem.m_ << "," << problem.n_ << ","
                     << problem.k_ << "," << problem.stride_a_ << "," << problem.stride_b_ << ","
                     << problem.stride_c_ << "," << problem.dtype_a_ << "," << problem.dtype_b_
                     << "," << problem.dtype_acc_ << "," << problem.dtype_c_ << ","
                     << problem.layout_a_ << "," << problem.layout_b_ << "," << problem.layout_c_
                     << "," << kernel_instance.name_ << "," << std::fixed << std::setprecision(4)
                     << perf.latency_ << "," << std::fixed << std::setprecision(4) << perf.tflops_
                     << "," << std::fixed << std::setprecision(4) << perf.bandwidth_ << ","
                     << get_metric_name(metric) << "\n";

                if(!file)
                {
                    std::cerr << "Warning: Error occurred while writing to CSV file." << std::endl;
                }
            }
        }

        return kernel_instance;
    }

    FlatmmProfiler(const FlatmmProfiler&)            = delete;
    FlatmmProfiler& operator=(const FlatmmProfiler&) = delete;

    private:
    ~FlatmmProfiler() { kernel_instances_.clear(); }
    FlatmmProfiler(Setting setting) : setting_(setting) {}

    Setting setting_;
    std::vector<KernelInstance> kernel_instances_;
};
