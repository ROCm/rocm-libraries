// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#ifndef GUARD_MIOPEN_TEST_GTEST_POOLING_COMMON_HPP
#define GUARD_MIOPEN_TEST_GTEST_POOLING_COMMON_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include <miopen/logger.hpp>
#include <miopen/miopen.h>
#include <miopen/pooling.hpp>
#include <miopen/tensor.hpp>
#include <miopen/tensor_layout.hpp>

#include "../cpu_conv.hpp"
#include "../tensor_holder.hpp"
#include "../test.hpp"
#include "../verify.hpp"
#include "../workspace.hpp"
#include "get_handle.hpp"

namespace pooling_gtest {

struct PoolingTestCase
{
    std::vector<int> in_shape;
    std::vector<int> lens;
    std::vector<int> pads;
    std::vector<int> strides;
    miopenIndexType_t index_type;
    miopenPoolingMode_t mode;
    int wsidx;
    std::string in_layout{"NCHW"};
    std::string out_layout{"NCHW"};

    friend std::ostream& operator<<(std::ostream& os, const PoolingTestCase& tc)
    {
        os << "in_shape: ";
        miopen::LogRange(os << "[", tc.in_shape, ",") << "] ";
        os << "lens: ";
        miopen::LogRange(os << "[", tc.lens, ",") << "] ";
        os << "pads: ";
        miopen::LogRange(os << "[", tc.pads, ",") << "] ";
        os << "strides: ";
        miopen::LogRange(os << "[", tc.strides, ",") << "] ";
        os << "in_layout: " << tc.in_layout << " ";
        return os << "index_type: " << tc.index_type << ", mode: " << tc.mode
                  << ", wsidx: " << tc.wsidx;
    }
};

template <class T>
struct pooling_operators
{
    miopen::PoolingDescriptor filter;
    pooling_operators(miopen::PoolingDescriptor f) : filter(f) {}

    double start() const
    {
        if(filter.GetMode() == miopenPoolingMax)
            return std::numeric_limits<T>::lowest();
        else
            return 0.0;
    }

    double operator()(double x, double y) const
    {
        if(filter.GetMode() == miopenPoolingMax)
            return std::max(x, y);
        else
            return x + y;
    }

    double final(double x, double y)
    {
        if(filter.GetMode() == miopenPoolingMax)
            return x;
        else
            return x / y;
    }
};

template <int SptDim>
struct verify_forward_pooling
{
    template <class T, class Index>
    tensor<T>
    cpu(const tensor<T>& input, const miopen::PoolingDescriptor& filter, std::vector<Index>&) const
    {
        tensor<T> out{filter.GetForwardOutputTensor(input.desc)};

        const auto in_strides         = input.desc.GetStrides();
        const std::size_t in_n_stride = in_strides[0];
        const std::size_t in_c_stride = in_strides[1];
        std::array<std::size_t, SptDim> in_spatial_strides{};
        std::copy_n(in_strides.begin() + 2, SptDim, in_spatial_strides.begin());

        const auto out_strides         = out.desc.GetStrides();
        const std::size_t out_n_stride = out_strides[0];
        const std::size_t out_c_stride = out_strides[1];
        std::array<std::size_t, SptDim> out_spatial_strides{};
        std::copy_n(out_strides.begin() + 2, SptDim, out_spatial_strides.begin());

        const T* input_ptr = input.data.data();
        T* out_ptr         = out.data.data();

        std::array<int, SptDim> in_dim{};
        std::copy_n(input.desc.GetLengths().begin() + 2, SptDim, in_dim.begin());
        std::array<int, SptDim> strides{};
        std::copy_n(filter.GetStrides().begin(), SptDim, strides.begin());
        std::array<int, SptDim> pads{};
        std::copy_n(filter.GetPads().begin(), SptDim, pads.begin());
        std::array<int, SptDim> kers{};
        std::copy_n(filter.GetLengths().begin(), SptDim, kers.begin());
        auto op = pooling_operators<T>{filter};

        int b_n = out.desc.GetLengths()[0];
        int k_n = out.desc.GetLengths()[1];
        std::array<int, SptDim> out_spatial_len{};
        std::copy_n(out.desc.GetLengths().begin() + 2, SptDim, out_spatial_len.begin());

        auto par_ford_out =
            miopen::unpacker(miopen::prepender(miopen::par_ford, b_n, k_n))(out_spatial_len);

        par_ford_out([&](int o, int w, auto... out_spatial_id_pack) {
            auto out_spatial_id = make_array(out_spatial_id_pack...);

            std::array<int, SptDim> start_idx{};
            std::array<int, SptDim> win_sz{};
            for(int i = 0; i < SptDim; ++i)
            {
                start_idx[i] = out_spatial_id[i] * strides[i] - pads[i];
                int end_idx  = start_idx[i] + kers[i];
                end_idx      = std::min(end_idx, in_dim[i]);
                start_idx[i] = std::max(start_idx[i], 0);
                win_sz[i]    = std::max(end_idx - start_idx[i], 0);
            }

            int pool_size =
                filter.GetMode() == miopenPoolingAverageInclusive
                    ? std::accumulate(kers.begin(), kers.end(), 1, std::multiplies<int>())
                    : std::accumulate(win_sz.begin(), win_sz.end(), 1, std::multiplies<int>());
            pool_size = std::max(pool_size, 1); // Avoid division by zero when window is in padding

            double acc = op.start();
            miopen::unpacker(miopen::ford)(win_sz)([&](auto... in_spatial_id_pack) {
                auto in_spatial_id = make_array(in_spatial_id_pack...);
                std::array<std::size_t, SptDim + 2> idx{};
                idx[0] = o;
                idx[1] = w;

                bool in_cmp_idx = true;
                for(int i = 0; i < SptDim; ++i)
                {
                    idx[i + 2] = start_idx[i] + in_spatial_id[i];
                    in_cmp_idx &= (in_dim[i] > static_cast<int>(idx[i + 2]));
                }

                if(in_cmp_idx)
                {
                    // Compute input linear index
                    std::size_t in_linear_idx = o * in_n_stride + w * in_c_stride;
                    for(int i = 0; i < SptDim; ++i)
                        in_linear_idx += idx[i + 2] * in_spatial_strides[i];
                    acc = op(acc, input_ptr[in_linear_idx]);
                }
            });
            // Compute output linear index
            std::size_t out_linear_idx = o * out_n_stride + w * out_c_stride;
            for(int i = 0; i < SptDim; ++i)
                out_linear_idx += out_spatial_id[i] * out_spatial_strides[i];
            out_ptr[out_linear_idx] = T(op.final(acc, pool_size));
        });
        return out;
    }

    template <class T, class Index>
    tensor<T> gpu(const tensor<T>& input,
                  const miopen::PoolingDescriptor& filter,
                  std::vector<Index>& indices) const
    {
        auto&& handle = get_handle();
        tensor<T> out{filter.GetForwardOutputTensor(input.desc)};
        indices.resize(out.data.size(), 0);

        auto in_dev  = handle.Write(input.data);
        auto out_dev = handle.Create<T>(out.data.size());
        Workspace wspace{};
        wspace.Write(indices);

        float alpha = 1, beta = 0;
        filter.Forward(handle,
                       &alpha,
                       input.desc,
                       in_dev.get(),
                       &beta,
                       out.desc,
                       out_dev.get(),
                       true,
                       wspace.ptr(),
                       wspace.size());

        indices  = wspace.Read<std::vector<Index>>();
        out.data = handle.Read<T>(out_dev, out.data.size());
        return out;
    }
};

template <int SptDim>
struct verify_backward_pooling
{
    template <class T, class Index>
    tensor<T> cpu(const tensor<T>& input,
                  const tensor<T>& dout,
                  const tensor<T>& out,
                  const miopen::PoolingDescriptor& filter,
                  const std::vector<Index>& indices,
                  bool use_global_index,
                  bool verify_index) const
    {
        auto dinput = input;
        std::vector<double> din_vec(input.desc.GetElementSpace(), 0.0);
        CHECK(dout.desc == out.desc);

        const auto in_strides         = input.desc.GetStrides();
        const std::size_t in_n_stride = in_strides[0];
        const std::size_t in_c_stride = in_strides[1];
        std::array<std::size_t, SptDim> in_spatial_strides{};
        std::copy_n(in_strides.begin() + 2, SptDim, in_spatial_strides.begin());

        const auto out_strides         = dout.desc.GetStrides();
        const std::size_t out_n_stride = out_strides[0];
        const std::size_t out_c_stride = out_strides[1];
        std::array<std::size_t, SptDim> out_spatial_strides{};
        std::copy_n(out_strides.begin() + 2, SptDim, out_spatial_strides.begin());

        const T* input_ptr = input.data.data();
        const T* dout_ptr  = dout.data.data();
        const T* out_ptr   = out.data.data();

        std::array<int, SptDim + 2> in_dim{};
        std::copy_n(input.desc.GetLengths().begin(), SptDim + 2, in_dim.begin());
        std::array<int, SptDim + 2> in_str{};
        std::copy_n(input.desc.GetStrides().begin(), SptDim + 2, in_str.begin());
        std::array<int, SptDim> strides{};
        std::copy_n(filter.GetStrides().begin(), SptDim, strides.begin());
        std::array<int, SptDim> pads{};
        std::copy_n(filter.GetPads().begin(), SptDim, pads.begin());
        std::array<int, SptDim> kers{};
        std::copy_n(filter.GetLengths().begin(), SptDim, kers.begin());
        auto ford_ker = miopen::unpacker(miopen::ford)(kers);

        int out_n = out.desc.GetLengths()[0];
        int out_c = out.desc.GetLengths()[1];
        std::array<int, SptDim> out_spatial_len{};
        std::copy_n(out.desc.GetLengths().begin() + 2, SptDim, out_spatial_len.begin());
        auto ford_out = miopen::unpacker(miopen::ford)(out_spatial_len);

        miopen::par_ford(out_n, out_c)([&](int o, int w) {
            if(filter.GetMode() == miopenPoolingMax)
            {
                ford_out([&](auto... out_spatial_id_pack) {
                    auto mx_idx = indices.at(dout.desc.GetIndex(o, w, out_spatial_id_pack...));
                    std::array<std::size_t, SptDim + 2> idx{};
                    bool in_cmp_idx = true;
                    if(use_global_index)
                    {
                        auto total_spatial = std::accumulate(
                            in_dim.begin() + 2, in_dim.end(), 1ULL, std::multiplies<std::size_t>());
                        in_cmp_idx = (mx_idx < total_spatial);

                        // Match legacy ctest behavior: if pooling window is fully outside
                        // input for this output coordinate, ignore this workspace index.
                        auto out_spatial_id = make_array(out_spatial_id_pack...);
                        for(int i = 0; i < SptDim && in_cmp_idx; i++)
                        {
                            int win_start = out_spatial_id[i] * strides[i] - pads[i];
                            int win_end   = win_start + kers[i] - 1;
                            if(win_end < 0 || win_start >= in_dim[i + 2])
                            {
                                in_cmp_idx = false;
                            }
                        }

                        for(int i = 0; i < SptDim; i++)
                        {
                            std::size_t mx_idx_dim = mx_idx;
                            mx_idx_dim /= std::accumulate(in_dim.begin() + i + 3,
                                                          in_dim.end(),
                                                          1ULL,
                                                          std::multiplies<std::size_t>());
                            mx_idx_dim %= in_dim[i + 2];
                            idx[i + 2] = mx_idx_dim;
                        }
                    }
                    else
                    {
                        auto out_spatial_id = make_array(out_spatial_id_pack...);
                        for(int i = 0; i < SptDim; i++)
                        {
                            int mx_idx_dim = mx_idx;
                            mx_idx_dim /= std::accumulate(
                                kers.begin() + i + 1, kers.end(), 1, std::multiplies<int>());
                            mx_idx_dim %= kers[i];
                            mx_idx_dim += (out_spatial_id[i] * strides[i] - pads[i]);
                            in_cmp_idx &= (in_dim[i + 2] > mx_idx_dim && mx_idx_dim >= 0);
                            idx[i + 2] = std::size_t(mx_idx_dim);
                        }
                    }

                    if(in_cmp_idx)
                    {
                        idx[0] = o;
                        idx[1] = w;
                        if(verify_index)
                        {
                            // Compute input and output linear indices for verification
                            std::size_t in_verify_idx  = o * in_n_stride + w * in_c_stride;
                            std::size_t out_verify_idx = o * out_n_stride + w * out_c_stride;
                            auto out_spatial_id        = make_array(out_spatial_id_pack...);
                            for(int i = 0; i < SptDim; ++i)
                            {
                                in_verify_idx += idx[i + 2] * in_spatial_strides[i];
                                out_verify_idx += out_spatial_id[i] * out_spatial_strides[i];
                            }
                            CHECK(miopen::float_equal(input_ptr[in_verify_idx],
                                                      out_ptr[out_verify_idx]));
                        }
                        std::size_t din_idx = 0;
                        for(int i = 0; i < SptDim + 2; i++)
                            din_idx += idx[i] * in_str[i];

                        std::size_t dout_linear_idx = o * out_n_stride + w * out_c_stride;
                        auto out_spatial_id         = make_array(out_spatial_id_pack...);
                        for(int i = 0; i < SptDim; ++i)
                            dout_linear_idx += out_spatial_id[i] * out_spatial_strides[i];
                        din_vec.at(din_idx) += dout_ptr[dout_linear_idx];
                    }
                });
            }
            else
            {
                ford_out([&](auto... out_spatial_id_pack) {
                    auto out_spatial_id = make_array(out_spatial_id_pack...);
                    std::array<int, SptDim> start_idx{};
                    std::array<int, SptDim> win_sz{};
                    for(int i = 0; i < SptDim; ++i)
                    {
                        start_idx[i] = out_spatial_id[i] * strides[i] - pads[i];
                        int end_idx  = start_idx[i] + kers[i];
                        end_idx      = std::min(end_idx, in_dim[i + 2]);
                        win_sz[i]    = end_idx - std::max(start_idx[i], 0);
                        win_sz[i]    = std::max(win_sz[i], 0);
                    }
                    int pool_size =
                        filter.GetMode() == miopenPoolingAverageInclusive
                            ? std::accumulate(kers.begin(), kers.end(), 1, std::multiplies<int>())
                            : std::accumulate(
                                  win_sz.begin(), win_sz.end(), 1, std::multiplies<int>());
                    pool_size = std::max(pool_size, 1); // Avoid division by zero

                    ford_ker([&](auto... ker_id_pack) {
                        auto ker_id     = make_array(ker_id_pack...);
                        bool in_cmp_idx = true;
                        std::array<int, SptDim + 2> in_idx{};
                        in_idx[0] = o;
                        in_idx[1] = w;
                        for(int i = 0; i < SptDim; ++i)
                        {
                            in_idx[i + 2] = start_idx[i] + ker_id[i];
                            in_cmp_idx &= (in_dim[i + 2] > in_idx[i + 2] && in_idx[i + 2] >= 0);
                        }
                        if(in_cmp_idx)
                        {
                            std::size_t din_idx = 0;
                            for(int i = 0; i < SptDim + 2; i++)
                            {
                                din_idx += in_idx[i] * in_str[i];
                            }

                            // Compute dout linear index
                            std::size_t dout_linear_idx = o * out_n_stride + w * out_c_stride;
                            for(int i = 0; i < SptDim; ++i)
                            {
                                dout_linear_idx += out_spatial_id[i] * out_spatial_strides[i];
                            }
                            din_vec.at(din_idx) +=
                                static_cast<double>(dout_ptr[dout_linear_idx]) / pool_size;
                        }
                    });
                });
            }
        });

        miopen::unpacker(miopen::ford)(in_dim)([&](auto... in_id_pack) {
            auto in_id          = make_array(in_id_pack...);
            std::size_t din_idx = 0;
            for(int i = 0; i < SptDim + 2; i++)
                din_idx += in_id[i] * in_str[i];
            dinput(in_id_pack...) = din_vec.at(din_idx);
        });
        return dinput;
    }

    template <class T, class Index>
    tensor<T> gpu(const tensor<T>& input,
                  const tensor<T>& dout,
                  const tensor<T>& out,
                  const miopen::PoolingDescriptor& filter,
                  const std::vector<Index>& indices,
                  bool,
                  bool) const
    {
        auto&& handle = get_handle();
        auto dinput   = input;
        auto in_dev   = handle.Write(input.data);
        auto dout_dev = handle.Write(dout.data);
        auto out_dev  = handle.Write(out.data);
        auto din_dev  = handle.Create<T>(dinput.data.size());
        Workspace wspace{};
        wspace.Write(indices);

        float alpha = 1, beta = 0;
        filter.Backward(handle,
                        &alpha,
                        // y
                        out.desc,
                        out_dev.get(),
                        // dy
                        dout.desc,
                        dout_dev.get(),
                        // x
                        input.desc,
                        in_dev.get(),
                        &beta,
                        // dx
                        dinput.desc,
                        din_dev.get(),
                        wspace.ptr());

        dinput.data = handle.Read<T>(din_dev, dinput.data.size());
        return dinput;
    }
};

template <typename T, typename Index, int SptDim>
void RunPoolingTestWithIndexType(const PoolingTestCase& test_case)
{
    tensor<T> input{test_case.in_shape};
    // Set up tensor descriptor with correct layout BEFORE generating data.
    if(test_case.in_layout != (SptDim == 2 ? "NCHW" : "NCDHW"))
    {
        const std::vector<std::size_t> dim_lens = input.desc.GetLengths();
        std::vector<std::size_t> dim_strides;
        miopen::tensor_layout_to_strides(dim_lens,
                                         miopen::tensor_layout_get_default(input.desc.GetNumDims()),
                                         test_case.in_layout,
                                         dim_strides);
        input.desc = miopen::TensorDescriptor(miopen_type<T>{}, dim_lens, dim_strides);
    }
    input.generate(tensor_elem_gen_integer{
        (miopen_type<T>{} == miopenHalf || miopen_type<T>{} == miopenBFloat16) ? 5 : 17});

    miopen::PoolingDescriptor filter{
        test_case.mode, miopenPaddingDefault, test_case.lens, test_case.strides, test_case.pads};
    filter.SetIndexType(test_case.index_type);
    filter.SetWorkspaceIndexMode(miopenPoolingWorkspaceIndexMode_t(test_case.wsidx));

    std::vector<Index> indices;
    verify_forward_pooling<SptDim> forward_verifier;

    auto forward_result     = forward_verifier.cpu(input, filter, indices);
    auto forward_gpu_result = forward_verifier.gpu(input, filter, indices);

    EXPECT_EQ(miopen::range_distance(forward_result), miopen::range_distance(forward_gpu_result));
    const double threshold = std::numeric_limits<T>::epsilon() * 80.0;
    const double forward_rms_error =
        miopen::rms_range(forward_result.data, forward_gpu_result.data);
    EXPECT_LE(forward_rms_error, threshold);

    auto dout = forward_result;
    dout.generate(tensor_elem_gen_integer{2503});

    if(test_case.mode == miopenPoolingMax && indices.empty())
        GTEST_FAIL() << "Indices not populated for max pooling backward";

    verify_backward_pooling<SptDim> backward_verifier;
    const bool use_global_index = test_case.wsidx != 0;
    const bool verify_index     = false;
    auto backward_result        = backward_verifier.cpu(
        input, dout, forward_result, filter, indices, use_global_index, verify_index);
    auto backward_gpu_result = backward_verifier.gpu(
        input, dout, forward_result, filter, indices, use_global_index, verify_index);

    EXPECT_EQ(miopen::range_distance(backward_result), miopen::range_distance(backward_gpu_result));
    const double backward_rms_error =
        miopen::rms_range(backward_result.data, backward_gpu_result.data);
    EXPECT_LE(backward_rms_error, threshold);
}

template <typename T, int SptDim = 2>
void RunPoolingTest(const PoolingTestCase& test_case)
{
    try
    {
        switch(test_case.index_type)
        {
        case miopenIndexUint8: RunPoolingTestWithIndexType<T, uint8_t, SptDim>(test_case); break;
        case miopenIndexUint16: RunPoolingTestWithIndexType<T, uint16_t, SptDim>(test_case); break;
        case miopenIndexUint32: RunPoolingTestWithIndexType<T, uint32_t, SptDim>(test_case); break;
        case miopenIndexUint64: RunPoolingTestWithIndexType<T, uint64_t, SptDim>(test_case); break;
        default: GTEST_FAIL() << "Unsupported index type";
        }
    }
    catch(const std::exception& e)
    {
        std::string msg = e.what();
        if(msg.find("No solver found") != std::string::npos ||
           msg.find("exceeds the device limit") != std::string::npos)
            GTEST_SKIP() << msg;
        else
            GTEST_FAIL() << msg;
    }
}

template <typename T, int SptDim = 2>
struct PoolingCommon : public testing::TestWithParam<PoolingTestCase>
{
    void SetUp() override { prng::reset_seed(); }
    void RunTest() { RunPoolingTest<T, SptDim>(this->GetParam()); }
};

inline bool ShouldIncludeTestCase(const PoolingTestCase& test_case,
                                  int& num_uint16_case,
                                  int& num_uint32_case,
                                  int& num_uint32_case_imgidx,
                                  int& num_uint64_case,
                                  int& num_uint64_case_imgidx,
                                  bool skip_many_configs = true,
                                  bool is_wide_dataset   = false)
{
    int spt_dim = static_cast<int>(test_case.in_shape.size()) - 2;
    if(test_case.wsidx == 0 && spt_dim == 3 && test_case.mode == miopenPoolingMax)
        return false;
    if(test_case.wsidx == 0 && spt_dim == 2 && test_case.mode == miopenPoolingMax &&
       is_wide_dataset)
        return false;
    if(test_case.wsidx == 0 &&
       (test_case.mode == miopenPoolingAverage || test_case.mode == miopenPoolingAverageInclusive))
        return false;
    if(spt_dim == 3 && test_case.mode == miopenPoolingMax &&
       (test_case.index_type == miopenIndexUint8 || test_case.index_type == miopenIndexUint16))
        return false;

    /// The "index is too small" limitation is an approximation
    /// of the real limitation, and therefore applied only when
    /// the "full test" is run. See:
    /// \ref max_pooling_index_max_restriction
    switch(test_case.index_type)
    {
    case miopenIndexUint8:
        if((spt_dim == 3 || (spt_dim == 2 && test_case.wsidx == 1)) &&
           test_case.mode == miopenPoolingMax)
            return false;
        break;
    case miopenIndexUint16:
        // test_pooling_test --all only test 5 uint16 cases
        if((spt_dim == 3 || (spt_dim == 2 && test_case.wsidx == 1)) &&
           test_case.mode == miopenPoolingMax)
            return false;
        if(skip_many_configs && ++num_uint16_case > 5)
            return false;
        break;
    case miopenIndexUint32:
        if(skip_many_configs)
        {
            // test_pooling_test --all only test 5 uint32 cases
            if(test_case.wsidx == 0)
            {
                if(++num_uint32_case > 5)
                    return false;
            }
            else
            {
                if(++num_uint32_case_imgidx > 5)
                    return false;
            }
        }
        break;
    case miopenIndexUint64:
        if(skip_many_configs)
        {
            if(test_case.wsidx == 0)
            {
                if(++num_uint64_case > 5)
                    return false;
            }
            else
            {
                if(++num_uint64_case_imgidx > 5 && spt_dim == 2)
                    return false;
            }
        }
        break;
    }

    miopen::TensorDescriptor input_desc(miopenFloat, test_case.in_shape);
    for(int i = 0; i < spt_dim; i++)
        if(test_case.lens[i] > (input_desc.GetLengths()[i + 2] + 2 * test_case.pads[i]))
            return false;

    return true;
}

inline void AddTestCasesForInput(const std::vector<int>& in_shape,
                                 const std::vector<std::vector<int>>& lens_list,
                                 const std::vector<std::vector<int>>& strides_list,
                                 const std::vector<std::vector<int>>& pads_list,
                                 const std::vector<miopenIndexType_t>& index_types,
                                 const std::vector<miopenPoolingMode_t>& modes,
                                 const std::vector<int>& wsidx_values,
                                 std::vector<PoolingTestCase>& test_cases,
                                 int& num_uint16_case,
                                 int& num_uint32_case,
                                 int& num_uint32_case_imgidx,
                                 int& num_uint64_case,
                                 int& num_uint64_case_imgidx,
                                 bool skip_many_configs    = true,
                                 bool is_wide_dataset      = false,
                                 const std::string& layout = "NCHW")
{
    for(auto index_type : index_types)
        for(auto mode : modes)
            for(const auto& lens : lens_list)
                for(const auto& strides : strides_list)
                    for(const auto& pads : pads_list)
                        for(int wsidx : wsidx_values)
                        {
                            PoolingTestCase tc = {in_shape,
                                                  lens,
                                                  pads,
                                                  strides,
                                                  index_type,
                                                  mode,
                                                  wsidx,
                                                  layout,
                                                  layout};
                            if(ShouldIncludeTestCase(tc,
                                                     num_uint16_case,
                                                     num_uint32_case,
                                                     num_uint32_case_imgidx,
                                                     num_uint64_case,
                                                     num_uint64_case_imgidx,
                                                     skip_many_configs,
                                                     is_wide_dataset))
                                test_cases.push_back(tc);
                        }
}

inline std::string GetPoolingTestCaseName(const testing::TestParamInfo<PoolingTestCase>& info)
{
    std::ostringstream os;
    os << info.param;
    std::string s        = os.str(), name;
    bool last_underscore = false;
    for(char c : s)
    {
        if(isalnum(c))
        {
            name += c;
            last_underscore = false;
        }
        else if(!last_underscore)
        {
            name += '_';
            last_underscore = true;
        }
    }
    return name;
}

} // namespace pooling_gtest

#endif
