// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include "tdm_fmha_v128_reference.hpp"
#include "fmha_fwd.hpp"

#include <hip/hip_runtime_api.h>

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <string>

namespace tdm_v128_test {
namespace reference {

struct DeviceValidation
{
    Metrics metrics;
    std::string report;
};

namespace runtime_detail {

inline void Require(bool valid, const char* message)
{
    if(!valid)
        throw std::invalid_argument(std::string("strict FP32 device oracle: ") + message);
}

inline void HipCheck(hipError_t result, const char* operation)
{
    if(result != hipSuccess)
        throw std::runtime_error(std::string("strict FP32 device oracle: ") + operation + ": " +
                                 hipGetErrorString(result));
}

inline std::size_t Nonnegative(ck_tile::index_t value, const char* name)
{
    Require(value >= 0, name);
    return static_cast<std::size_t>(value);
}

template <typename T>
std::vector<T> CopyDevice(const void* pointer, std::size_t count)
{
    if(count == 0)
        return {};
    Require(pointer != nullptr, "null required device pointer");
    const auto bytes = detail::AddExtent(0, count, sizeof(T));
    hipPointerAttribute_t attributes{};
    HipCheck(hipPointerGetAttributes(&attributes, pointer), "hipPointerGetAttributes");
    int device = 0;
    HipCheck(hipGetDevice(&device), "hipGetDevice");
    Require(attributes.type == hipMemoryTypeDevice && attributes.device == device,
            "expected a device allocation on the current device");
    hipDeviceptr_t allocation    = nullptr;
    std::size_t allocation_bytes = 0;
    HipCheck(hipMemGetAddressRange(&allocation,
                                   &allocation_bytes,
                                   static_cast<hipDeviceptr_t>(const_cast<void*>(pointer))),
             "hipMemGetAddressRange");
    const auto address = reinterpret_cast<std::uintptr_t>(pointer);
    const auto base    = reinterpret_cast<std::uintptr_t>(allocation);
    Require(address >= base && address - base <= allocation_bytes &&
                bytes <= allocation_bytes - (address - base),
            "metadata span exceeds device allocation");
    std::vector<T> result(count);
    HipCheck(hipMemcpy(result.data(), pointer, bytes, hipMemcpyDeviceToHost), "hipMemcpy D2H");
    return result;
}

inline std::vector<float> CopyBf16(const void* pointer, std::size_t count)
{
    const auto encoded = CopyDevice<std::uint16_t>(pointer, count);
    std::vector<float> decoded(count);
    static_assert(sizeof(float) == sizeof(std::uint32_t) && std::numeric_limits<float>::is_iec559);
    for(std::size_t i = 0; i < count; ++i)
    {
        const std::uint32_t bits = static_cast<std::uint32_t>(encoded[i]) << 16;
        std::memcpy(&decoded[i], &bits, sizeof(float));
    }
    return decoded;
}

inline void ValidateCumulative(const std::vector<std::int32_t>& values)
{
    if(values.empty())
        return;
    Require(values.front() >= 0 && std::is_sorted(values.begin(), values.end()),
            "negative or nonmonotonic cumulative sequence lengths");
}

struct Lengths
{
    std::vector<std::int32_t> starts;
    std::vector<std::int32_t> logical;
};

inline Lengths ReadLengths(bool group,
                           std::size_t batch,
                           ck_tile::index_t batch_length,
                           const void* physical_pointer,
                           const void* logical_pointer,
                           const void* cumulative_pointer)
{
    Require(!(logical_pointer && cumulative_pointer),
            "per-sequence and cumulative logical lengths are mutually exclusive");
    Require(group || (!physical_pointer && !logical_pointer),
            "batch mode must not supply group-only sequence metadata");
    Require(!group || physical_pointer, "group mode requires physical sequence starts");
    Lengths result;
    if(group)
    {
        result.starts = CopyDevice<std::int32_t>(physical_pointer, batch + 1);
        ValidateCumulative(result.starts);
    }
    if(logical_pointer)
        result.logical = CopyDevice<std::int32_t>(logical_pointer, batch);
    else if(cumulative_pointer)
    {
        const auto cumulative = CopyDevice<std::int32_t>(cumulative_pointer, batch + 1);
        ValidateCumulative(cumulative);
        result.logical.resize(batch);
        for(std::size_t b = 0; b < batch; ++b)
            result.logical[b] = cumulative[b + 1] - cumulative[b];
    }
    else if(group)
    {
        result.logical.resize(batch);
        for(std::size_t b = 0; b < batch; ++b)
            result.logical[b] = result.starts[b + 1] - result.starts[b];
    }
    else
    {
        Require(batch_length >= 0, "negative batch sequence length");
        result.logical.assign(batch, batch_length);
    }
    for(std::size_t b = 0; b < batch; ++b)
    {
        const auto physical = group ? result.starts[b + 1] - result.starts[b] : batch_length;
        Require(result.logical[b] >= 0 && result.logical[b] <= physical,
                "logical sequence length exceeds physical extent");
    }
    return result;
}

inline std::size_t
RequiredElements(const FloatView& view, std::size_t heads, std::size_t rows, std::size_t dimensions)
{
    if(heads == 0 || rows == 0 || dimensions == 0)
        return 0;
    auto last = detail::AddExtent(view.offset, heads - 1, view.head_stride);
    last      = detail::AddExtent(last, rows - 1, view.row_stride);
    last      = detail::AddExtent(last, dimensions - 1, view.dim_stride);
    return detail::AddExtent(last, 1, 1);
}

inline FloatView MakeView(bool group,
                          std::size_t batch,
                          const Lengths& lengths,
                          ck_tile::index_t head_stride,
                          ck_tile::index_t row_stride,
                          ck_tile::index_t batch_stride)
{
    FloatView view;
    view.head_stride = Nonnegative(head_stride, "negative head stride");
    view.row_stride  = Nonnegative(row_stride, "negative row stride");
    view.offset =
        group ? detail::AddExtent(0, lengths.starts[batch], view.row_stride)
              : detail::AddExtent(0, batch, Nonnegative(batch_stride, "negative batch stride"));
    return view;
}

inline void Bind(FloatView& view, const std::vector<float>& values)
{
    view.data = values.data();
    view.size = values.size();
}

template <typename Mask>
void BindMask(Sequence& sequence, const Mask& mask)
{
    sequence.is_masked = [mask](std::size_t query, std::size_t key) {
        return mask.IsOutOfSinkBound(static_cast<ck_tile::index_t>(query),
                                     static_cast<ck_tile::index_t>(key));
    };
}

inline void SelectRows(Sequence& sequence)
{
    if(sequence.query_length == 0 || std::max(sequence.query_length, sequence.key_length) < 32768)
        return;
    const auto last = sequence.query_length - 1;
    for(const std::size_t row : {std::size_t{0},
                                 std::size_t{1},
                                 std::size_t{31},
                                 std::size_t{32},
                                 std::size_t{63},
                                 std::size_t{64},
                                 std::size_t{127},
                                 std::size_t{128},
                                 sequence.query_length / 2,
                                 last == 0 ? 0 : last - 1,
                                 last})
        if(row <= last)
            sequence.sampled_query_rows.push_back(row);
    if(sequence.query_length > sequence.key_length)
    {
        const auto boundary = sequence.query_length - sequence.key_length;
        sequence.sampled_query_rows.push_back(boundary - 1);
        if(boundary <= last)
            sequence.sampled_query_rows.push_back(boundary);
    }
    auto& rows = sequence.sampled_query_rows;
    std::sort(rows.begin(), rows.end());
    rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
}

inline void PrintView(std::ostream& stream, const char* name, const FloatView& view)
{
    stream << ' ' << name << "={offset:" << view.offset << ",head:" << view.head_stride
           << ",row:" << view.row_stride << ",dim:" << view.dim_stride << '}';
}

} // namespace runtime_detail

// Test-only, BF16 Q/K/V/O, row V, no bias/dropout/quantization/softcap/sink/head slicing.
// This deliberately synchronizes and copies device allocations. Never call it in
// the timing binary. Generic-window enum is rejected because the current runner
// and kernel disagree on its alignment; TL/BR sliding windows are supported.
inline DeviceValidation ValidateFromDevice(const fmha_fwd_traits& traits, const fmha_fwd_args& args)
{
    using namespace runtime_detail;
    Require(traits.data_type == "bf16" && traits.is_v_rowmajor,
            "only BF16 Q/K/V/O with row-major V is supported");
    Require(!traits.has_logits_soft_cap && args.logits_soft_cap == 0 &&
                traits.bias_type == bias_enum::no_bias && !traits.has_dropout && args.p_drop == 0 &&
                traits.qscale_type == quant_scale_enum::no_scale && !traits.has_sink &&
                args.sink_ptr == nullptr && args.sink_size == 0 && !traits.skip_min_seqlen_q,
            "bias/dropout/quantization/softcap/sink/skip-min-sequence is unsupported");
    Require(args.head_start == 0 &&
                (args.num_head_q_total == 0 || args.num_head_q_total == args.nhead_q),
            "head slicing is unsupported");
    Require(args.batch > 0 && args.nhead_q > 0 && args.nhead_k > 0 &&
                args.nhead_q % args.nhead_k == 0 && args.hdim_q > 0 && args.hdim_v > 0 &&
                traits.hdim_q == args.hdim_q && traits.hdim_v == args.hdim_v &&
                std::isfinite(args.scale_s),
            "invalid dimensions, GQA ratio, or scale");
    Require(args.mask_type == static_cast<ck_tile::index_t>(traits.mask_type) &&
                (traits.mask_type == mask_enum::no_mask ||
                 traits.mask_type == mask_enum::mask_top_left ||
                 traits.mask_type == mask_enum::mask_bottom_right),
            "invalid mask metadata or unsupported generic-window alignment");
    HipCheck(hipDeviceSynchronize(), "hipDeviceSynchronize before validation");
    const auto batch     = static_cast<std::size_t>(args.batch);
    const auto q_lengths = ReadLengths(traits.is_group_mode,
                                       batch,
                                       args.seqlen_q,
                                       args.seqstart_q_ptr,
                                       args.seqlen_q_ptr,
                                       args.cu_seqlen_q_ptr);
    const auto k_lengths = ReadLengths(traits.is_group_mode,
                                       batch,
                                       args.seqlen_k,
                                       args.seqstart_k_ptr,
                                       args.seqlen_k_ptr,
                                       args.cu_seqlen_k_ptr);
    std::vector<Sequence> sequences(batch);
    std::vector<Output> outputs(batch);
    std::size_t q_size = 0, k_size = 0, v_size = 0, o_size = 0, lse_size = 0;
    for(std::size_t b = 0; b < batch; ++b)
    {
        auto& sequence        = sequences[b];
        auto& output          = outputs[b];
        sequence.query_length = q_lengths.logical[b];
        sequence.key_length   = k_lengths.logical[b];
        Require(args.max_seqlen_q >= q_lengths.logical[b],
                "query length exceeds maximum used for the launch grid");
        sequence.query_heads     = args.nhead_q;
        sequence.kv_heads        = args.nhead_k;
        sequence.query_dimension = args.hdim_q;
        sequence.value_dimension = args.hdim_v;
        sequence.score_scale     = args.scale_s;
        sequence.q               = MakeView(traits.is_group_mode,
                              b,
                              q_lengths,
                              args.nhead_stride_q,
                              args.stride_q,
                              args.batch_stride_q);
        sequence.k               = MakeView(traits.is_group_mode,
                              b,
                              k_lengths,
                              args.nhead_stride_k,
                              args.stride_k,
                              args.batch_stride_k);
        sequence.v               = MakeView(traits.is_group_mode,
                              b,
                              k_lengths,
                              args.nhead_stride_v,
                              args.stride_v,
                              args.batch_stride_v);
        output.o                 = MakeView(traits.is_group_mode,
                            b,
                            q_lengths,
                            args.nhead_stride_o,
                            args.stride_o,
                            args.batch_stride_o);
        q_size                   = std::max(
            q_size,
            RequiredElements(
                sequence.q, sequence.query_heads, sequence.query_length, sequence.query_dimension));
        k_size = std::max(
            k_size,
            RequiredElements(
                sequence.k, sequence.kv_heads, sequence.key_length, sequence.query_dimension));
        v_size = std::max(
            v_size,
            RequiredElements(
                sequence.v, sequence.kv_heads, sequence.key_length, sequence.value_dimension));
        o_size = std::max(
            o_size,
            RequiredElements(
                output.o, sequence.query_heads, sequence.query_length, sequence.value_dimension));
        if(traits.has_lse)
        {
            output.lse = MakeView(traits.is_group_mode,
                                  b,
                                  q_lengths,
                                  args.nhead_stride_lse,
                                  1,
                                  args.batch_stride_lse);
            lse_size   = std::max(
                lse_size,
                RequiredElements(*output.lse, sequence.query_heads, sequence.query_length, 1));
        }
        if(traits.mask_type != mask_enum::no_mask)
        {
            // Match fmha_fwd_runner.hpp: causal and local masks have distinct
            // CK types, with alignment evaluated using each sequence's lengths.
            const bool top_left = traits.mask_type == mask_enum::mask_top_left;
            if(args.window_size_left < 0)
                BindMask(sequence,
                         ck_tile::make_generic_attention_mask_from_lr_window<FmhaMasks::CausalMask>(
                             args.window_size_left,
                             args.window_size_right,
                             0,
                             q_lengths.logical[b],
                             k_lengths.logical[b],
                             top_left));
            else
                BindMask(
                    sequence,
                    ck_tile::make_generic_attention_mask_from_lr_window<FmhaMasks::GenericMask>(
                        args.window_size_left,
                        args.window_size_right,
                        0,
                        q_lengths.logical[b],
                        k_lengths.logical[b],
                        top_left));
        }
        SelectRows(sequence);
    }
    const auto q = CopyBf16(args.q_ptr, q_size);
    const auto k = CopyBf16(args.k_ptr, k_size);
    const auto v = CopyBf16(args.v_ptr, v_size);
    const auto o = CopyBf16(args.o_ptr, o_size);
    const auto lse =
        traits.has_lse ? CopyDevice<float>(args.lse_ptr, lse_size) : std::vector<float>{};
    for(std::size_t b = 0; b < batch; ++b)
    {
        Bind(sequences[b].q, q);
        Bind(sequences[b].k, k);
        Bind(sequences[b].v, v);
        Bind(outputs[b].o, o);
        if(outputs[b].lse)
            Bind(*outputs[b].lse, lse);
    }
    const auto expected = Compute(sequences);
    DeviceValidation validation{Compare(sequences, expected, outputs), {}};
    const auto& m = validation.metrics;
    std::ostringstream report;
    report << std::setprecision(10) << "\nstrict_fp32_oracle dtype=bf16 P=fp32 mode="
           << (traits.is_group_mode ? "group" : "batch") << " lse=" << traits.has_lse
           << " maxabs_limit=0.015625 mincos_limit=0.99998 meancos_limit=0.99999"
           << " lse_atol=0.02 lse_rtol=0.002\n";
    for(std::size_t b = 0; b < batch; ++b)
    {
        const auto& s = sequences[b];
        report << "strict_fp32_sequence batch=" << b << " Sq=" << s.query_length
               << " Sk=" << s.key_length << " Hq=" << s.query_heads << " Hkv=" << s.kv_heads
               << " Dq=" << s.query_dimension << " Dv=" << s.value_dimension
               << " score_scale=" << s.score_scale << " descales=1,1,1 mask=" << args.mask_type
               << " left=" << args.window_size_left << " right=" << args.window_size_right;
        PrintView(report, "Q", s.q);
        PrintView(report, "K", s.k);
        PrintView(report, "V", s.v);
        PrintView(report, "O", outputs[b].o);
        if(outputs[b].lse)
            PrintView(report, "LSE", *outputs[b].lse);
        report << " query_rows=";
        if(s.sampled_query_rows.empty())
            report << "all";
        else
            for(std::size_t i = 0; i < s.sampled_query_rows.size(); ++i)
                report << (i ? "," : "") << s.sampled_query_rows[i];
        report << '\n';
    }
    report << "strict_fp32_metrics passed=" << m.passed << " checked_rows=" << m.checked_rows
           << " checked_elements=" << m.checked_elements
           << " scanned_output_elements=" << m.scanned_output_elements
           << " checked_lse_rows=" << m.checked_lse_rows
           << " scanned_lse_rows=" << m.scanned_lse_rows << " maxabs=" << m.max_abs_error
           << " maxrelative=" << m.max_relative_error << " rms=" << m.rms_error
           << " relative_rms=" << m.relative_rms_error << " mincos=" << m.min_row_cosine
           << " meancos=" << m.mean_row_cosine << " lse_maxabs=" << m.max_lse_abs_error
           << " lse_maxrelative=" << m.max_lse_relative_error
           << " lse_mismatch=" << m.lse_mismatch_count
           << " nonfinite_output=" << m.nonfinite_output_count
           << " invalid_lse=" << m.invalid_lse_count << '\n';
    validation.report = report.str();
    return validation;
}

} // namespace reference
} // namespace tdm_v128_test
