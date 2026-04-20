// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <memory>
#include <tuple>

#include "ck_tile/ops/gemm_quant.hpp"
#include "gemm/gemm_profiler.hpp"
#include "gemm_quant_benchmark.hpp"

class GemmQuantProfiler
    : public GemmProfiler<GemmQuantProfiler, GemmQuantProblem, ck_tile::QuantGemmHostArgs>
{
    public:
    using BaseGemm = GemmProfiler<GemmQuantProfiler, GemmQuantProblem, ck_tile::QuantGemmHostArgs>;
    using BaseGemm::benchmark;

    GemmQuantProfiler(Settings setting)
        : GemmProfiler<GemmQuantProfiler, GemmQuantProblem, ck_tile::QuantGemmHostArgs>(setting)
    {
    }

    void
    benchmark(GemmQuantProblem& gemm_problem,
              std::vector<std::function<std::tuple<std::string, float>(
                  ck_tile::QuantGemmHostArgs&, const ck_tile::stream_config&)>>& callables) override
    {
        const ALayout layout_a   = ALayout{};
        const AQLayout layout_aq = AQLayout{};
        const BLayout layout_b   = BLayout{};
        const BQLayout layout_bq = BQLayout{};
        const CLayout layout_c   = CLayout{};

        gemm_problem.stride_a_ = ck_tile::get_default_stride(
            gemm_problem.m_, gemm_problem.k_, gemm_problem.stride_a_, is_row_major(layout_a));
        gemm_problem.stride_b_ = ck_tile::get_default_stride(
            gemm_problem.k_, gemm_problem.n_, gemm_problem.stride_b_, is_row_major(layout_b));
        gemm_problem.stride_c_ = ck_tile::get_default_stride(
            gemm_problem.m_, gemm_problem.n_, gemm_problem.stride_c_, is_row_major(layout_c));

        const auto aq_group = parse_quant_group(gemm_problem.aq_group_);
        const auto bq_group = parse_quant_group(gemm_problem.bq_group_);

        ck_tile::index_t aq_rows = 0;
        ck_tile::index_t aq_cols = 0;
        ck_tile::index_t bq_rows = 0;
        ck_tile::index_t bq_cols = 0;

        if(gemm_problem.quant_mode_ == "AQuantGrouped")
        {
            gemm_problem.qk_a_ = ck_tile::integer_divide_ceil(gemm_problem.k_, aq_group[2]);
            gemm_problem.qk_b_ = 0;
            aq_rows            = gemm_problem.m_;
            aq_cols            = gemm_problem.qk_a_;
        }
        else if(gemm_problem.quant_mode_ == "BQuantGrouped")
        {
            gemm_problem.qk_a_ = 0;
            gemm_problem.qk_b_ = ck_tile::integer_divide_ceil(gemm_problem.k_, bq_group[2]);
            bq_rows            = gemm_problem.qk_b_;
            bq_cols            = ck_tile::integer_divide_ceil(gemm_problem.n_, bq_group[1]);
        }
        else if(gemm_problem.quant_mode_ == "RowColQuant")
        {
            gemm_problem.qk_a_ = 1;
            gemm_problem.qk_b_ = 1;
            aq_rows            = gemm_problem.m_;
            aq_cols            = 1;
            bq_rows            = 1;
            bq_cols            = gemm_problem.n_;
        }
        else if(gemm_problem.quant_mode_ == "TensorQuant")
        {
            gemm_problem.qk_a_ = 1;
            gemm_problem.qk_b_ = 1;
            aq_rows            = 1;
            aq_cols            = 1;
            bq_rows            = 1;
            bq_cols            = 1;
        }
        else
        {
            throw std::runtime_error("Unsupported gemm_quant mode");
        }

        if(aq_rows > 0)
        {
            gemm_problem.stride_aq_ = ck_tile::get_default_stride(
                aq_rows, aq_cols, gemm_problem.stride_aq_, is_row_major(layout_aq));
        }
        else
        {
            gemm_problem.stride_aq_ = 0;
        }

        if(bq_rows > 0)
        {
            gemm_problem.stride_bq_ = ck_tile::get_default_stride(
                bq_rows, bq_cols, gemm_problem.stride_bq_, is_row_major(layout_bq));
        }
        else
        {
            gemm_problem.stride_bq_ = 0;
        }

        ck_tile::HostTensor<ADataType> a_m_k(ck_tile::host_tensor_descriptor(
            gemm_problem.m_, gemm_problem.k_, gemm_problem.stride_a_, is_row_major(layout_a)));
        ck_tile::HostTensor<BDataType> b_k_n(ck_tile::host_tensor_descriptor(
            gemm_problem.k_, gemm_problem.n_, gemm_problem.stride_b_, is_row_major(layout_b)));
        ck_tile::HostTensor<CDataType> c_m_n_dev_result(ck_tile::host_tensor_descriptor(
            gemm_problem.m_, gemm_problem.n_, gemm_problem.stride_c_, is_row_major(layout_c)));

        std::unique_ptr<ck_tile::HostTensor<AQDataType>> aq_tensor_ptr = nullptr;
        if(aq_rows > 0)
        {
            aq_tensor_ptr =
                std::make_unique<ck_tile::HostTensor<AQDataType>>(ck_tile::host_tensor_descriptor(
                    aq_rows, aq_cols, gemm_problem.stride_aq_, is_row_major(layout_aq)));
        }

        std::unique_ptr<ck_tile::HostTensor<BQDataType>> bq_tensor_ptr = nullptr;
        if(bq_rows > 0)
        {
            bq_tensor_ptr =
                std::make_unique<ck_tile::HostTensor<BQDataType>>(ck_tile::host_tensor_descriptor(
                    bq_rows, bq_cols, gemm_problem.stride_bq_, is_row_major(layout_bq)));
        }

        if(setting_.init_method == 0)
        {
            ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k);
            ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n);
            if(aq_tensor_ptr)
            {
                ck_tile::FillUniformDistribution<AQDataType>{0.5f, 2.0f}(*aq_tensor_ptr);
            }
            if(bq_tensor_ptr)
            {
                ck_tile::FillUniformDistribution<BQDataType>{0.5f, 2.0f}(*bq_tensor_ptr);
            }
        }
        else if(setting_.init_method == 1)
        {
            ck_tile::FillMonotonicSeq<ADataType>{}(a_m_k);
            ck_tile::FillMonotonicSeq<BDataType>{}(b_k_n);
            if(aq_tensor_ptr)
            {
                ck_tile::FillMonotonicSeq<AQDataType>{}(*aq_tensor_ptr);
            }
            if(bq_tensor_ptr)
            {
                ck_tile::FillMonotonicSeq<BQDataType>{}(*bq_tensor_ptr);
            }
        }
        else if(setting_.init_method == 2)
        {
            ck_tile::FillConstant<ADataType>{static_cast<ADataType>(1)}(a_m_k);
            ck_tile::FillConstant<BDataType>{static_cast<BDataType>(1)}(b_k_n);
            if(aq_tensor_ptr)
            {
                ck_tile::FillConstant<AQDataType>{static_cast<AQDataType>(1)}(*aq_tensor_ptr);
            }
            if(bq_tensor_ptr)
            {
                ck_tile::FillConstant<BQDataType>{static_cast<BQDataType>(1)}(*bq_tensor_ptr);
            }
        }
        else
        {
            a_m_k.SetZero();
            b_k_n.SetZero();
            if(aq_tensor_ptr)
            {
                aq_tensor_ptr->SetZero();
            }
            if(bq_tensor_ptr)
            {
                bq_tensor_ptr->SetZero();
            }
        }

        ck_tile::DeviceMem a_m_k_dev_buf(a_m_k.get_element_space_size_in_bytes());
        ck_tile::DeviceMem b_k_n_dev_buf(b_k_n.get_element_space_size_in_bytes());
        ck_tile::DeviceMem c_m_n_dev_buf(c_m_n_dev_result.get_element_space_size_in_bytes());

        std::unique_ptr<ck_tile::DeviceMem> aq_dev_buf_ptr = nullptr;
        if(aq_tensor_ptr)
        {
            aq_dev_buf_ptr = std::make_unique<ck_tile::DeviceMem>(
                aq_tensor_ptr->get_element_space_size_in_bytes());
            aq_dev_buf_ptr->ToDevice(aq_tensor_ptr->data());
        }

        std::unique_ptr<ck_tile::DeviceMem> bq_dev_buf_ptr = nullptr;
        if(bq_tensor_ptr)
        {
            bq_dev_buf_ptr = std::make_unique<ck_tile::DeviceMem>(
                bq_tensor_ptr->get_element_space_size_in_bytes());
            bq_dev_buf_ptr->ToDevice(bq_tensor_ptr->data());
        }

        a_m_k_dev_buf.ToDevice(a_m_k.data());
        b_k_n_dev_buf.ToDevice(b_k_n.data());
        c_m_n_dev_buf.SetZero();
        c_m_n_dev_result.SetZero();

        ck_tile::QuantGemmHostArgs gemm_args = {
            a_m_k_dev_buf.GetDeviceBuffer(),
            b_k_n_dev_buf.GetDeviceBuffer(),
            c_m_n_dev_buf.GetDeviceBuffer(),
            aq_dev_buf_ptr ? aq_dev_buf_ptr->GetDeviceBuffer() : nullptr,
            bq_dev_buf_ptr ? bq_dev_buf_ptr->GetDeviceBuffer() : nullptr,
            gemm_problem.split_k_,
            gemm_problem.m_,
            gemm_problem.n_,
            gemm_problem.k_,
            gemm_problem.qk_a_,
            gemm_problem.qk_b_,
            gemm_problem.stride_a_,
            gemm_problem.stride_b_,
            gemm_problem.stride_c_,
            gemm_problem.stride_aq_,
            gemm_problem.stride_bq_};

        ck_tile::HostTensor<CDataType> c_m_n_host_result(ck_tile::host_tensor_descriptor(
            gemm_problem.m_, gemm_problem.n_, gemm_problem.stride_c_, is_row_major(layout_c)));

        if(setting_.verify)
        {
            gemm_quant_host_reference(setting_.verify,
                                      a_m_k,
                                      aq_tensor_ptr.get(),
                                      b_k_n,
                                      bq_tensor_ptr.get(),
                                      c_m_n_host_result);
        }

        for(auto& callable : callables)
        {
            auto kernel_run_result = callable(gemm_args,
                                              ck_tile::stream_config{nullptr,
                                                                     true,
                                                                     setting_.log,
                                                                     setting_.n_warmup,
                                                                     setting_.n_repeat,
                                                                     setting_.is_gpu_timer,
                                                                     setting_.flush_cache,
                                                                     setting_.rotating_count});
            process_result(gemm_problem,
                           c_m_n_dev_buf,
                           c_m_n_host_result,
                           c_m_n_dev_result,
                           kernel_run_result);
        }
    }

    protected:
    std::size_t get_byte_count(const GemmQuantProblem& problem) const override
    {
        std::size_t num_byte = BaseGemm::get_byte_count(problem);

        if(problem.quant_mode_ == "AQuantGrouped")
        {
            num_byte += sizeof(AQDataType) * problem.m_ * problem.qk_a_;
        }
        else if(problem.quant_mode_ == "BQuantGrouped")
        {
            const auto bq_group = parse_quant_group(problem.bq_group_);
            const auto bq_cols  = ck_tile::integer_divide_ceil(problem.n_, bq_group[1]);
            num_byte += sizeof(BQDataType) * problem.qk_b_ * bq_cols;
        }
        else if(problem.quant_mode_ == "RowColQuant")
        {
            num_byte += sizeof(AQDataType) * problem.m_;
            num_byte += sizeof(BQDataType) * problem.n_;
        }
        else if(problem.quant_mode_ == "TensorQuant")
        {
            num_byte += sizeof(AQDataType);
            num_byte += sizeof(BQDataType);
        }

        return num_byte;
    }

    void write_csv_header(std::ostream& os) const override
    {
        os << "rocm_version,device_name,"
           << "split_k,m,n,k,stride_a,stride_aq,stride_b,stride_bq,stride_c," << "qk_a,qk_b,"
           << "dtype_a,dtype_q,dtype_b,dtype_acc,dtype_c,"
           << "layout_a,layout_aq,layout_b,layout_bq,layout_c,"
           << "quant_mode,quant_profile,aq_group,bq_group,"
           << "name,latency(ms),tflops(TFlops),bandwidth(GB/s),metric\n";
    }

    void write_csv_row(std::ostream& os,
                       const KernelInstance<GemmQuantProblem>& kernel_instance,
                       Metric metric) const override
    {
        const auto& problem = kernel_instance.problem_;
        const auto& perf    = kernel_instance.perf_result_;

        os << get_rocm_version() << "," << ck_tile::get_device_name() << "," << problem.split_k_
           << "," << problem.m_ << "," << problem.n_ << "," << problem.k_ << ","
           << problem.stride_a_ << "," << problem.stride_aq_ << "," << problem.stride_b_ << ","
           << problem.stride_bq_ << "," << problem.stride_c_ << "," << problem.qk_a_ << ","
           << problem.qk_b_ << "," << problem.dtype_a_ << "," << problem.dtype_q_ << ","
           << problem.dtype_b_ << "," << problem.dtype_acc_ << "," << problem.dtype_c_ << ","
           << problem.layout_a_ << "," << problem.layout_aq_ << "," << problem.layout_b_ << ","
           << problem.layout_bq_ << "," << problem.layout_c_ << "," << problem.quant_mode_ << ","
           << problem.quant_profile_ << "," << problem.aq_group_ << "," << problem.bq_group_ << ","
           << kernel_instance.name_ << "," << std::fixed << std::setprecision(4) << perf.latency_
           << "," << std::fixed << std::setprecision(4) << perf.tflops_ << "," << std::fixed
           << std::setprecision(4) << perf.bandwidth_ << "," << get_metric_name(metric) << "\n";
    }
};
