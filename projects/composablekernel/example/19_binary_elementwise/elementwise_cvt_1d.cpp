// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2024, Advanced Micro Devices, Inc. All rights reserved.

#include <iostream>
#include <cstdlib>

#include "ck/ck.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_elementwise_dynamic_vector_dims_impl.hpp"
#include "ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp"
#include "ck/library/utility/check_err.hpp"
#include "ck/library/utility/device_memory.hpp"
#include "ck/library/utility/host_tensor.hpp"
#include "ck/library/utility/host_tensor_generator.hpp"

using ::ck::DeviceMem;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;

using F8   = ck::f8_t;
using BF8  = ck::bf8_t;
using F16  = ck::half_t;
using BF16 = ck::bhalf_t;

using UnaryConvert = ck::tensor_operation::element_wise::UnaryConvert;
using ConvertF8SR  = ck::tensor_operation::element_wise::ConvertF8SR;
using ConvertF8RNE = ck::tensor_operation::element_wise::ConvertF8RNE;

template <typename HostTensorA, typename HostTensorC, typename Functor>
void unary_host_elementwise1D(HostTensorC& C, const HostTensorA& A, int M, Functor functor)
{
    using ctype = ck::remove_reference_t<decltype(C(0))>;

    for(int m = 0; m < M; ++m)
    {
        auto Am  = A(m);
        ctype Cm = {};
        functor(Cm, Am);
        C(m) = Cm;
    }
}

template <typename HostTensorA, typename HostTensorC, typename Functor>
void f8sr_host_elementwise1D(HostTensorC& C, const HostTensorA& A, int M, Functor functor)
{
    using ctype = ck::remove_reference_t<decltype(C(0))>;

    for(int m = 0; m < M; ++m)
    {
        auto Am  = A(m);
        ctype Cm = {};
        functor(Cm, Am);
        C(m) = Cm;
    }
}

template <typename HostTensorA, typename HostTensorC, typename Functor>
void f8rne_host_elementwise1D(HostTensorC& C, const HostTensorA& A, int M, Functor functor)
{
    using ctype = ck::remove_reference_t<decltype(C(0))>;

    for(int m = 0; m < M; ++m)
    {
        auto Am  = A(m);
        ctype Cm = {};
        functor(Cm, Am);
        C(m) = Cm;
    }
}

template <typename ADataType, typename CDataType>
bool verifyUnaryConvert()
{
    bool do_verification = true;
    bool time_kernel     = false;

    ck::index_t M = 1024;

    auto f_host_tensor_descriptor1d = [](std::size_t len, std::size_t stride) {
        return HostTensorDescriptor({len}, {stride});
    };

    Tensor<ADataType> a_m(f_host_tensor_descriptor1d(M, 1));
    Tensor<CDataType> c_m(f_host_tensor_descriptor1d(M, 1));

    a_m.GenerateTensorValue(GeneratorTensor_3<ADataType>{-5000, 5000});

    DeviceMem a_m_device_buf(sizeof(ADataType) * a_m.mDesc.GetElementSpaceSize());
    DeviceMem c_m_device_buf(sizeof(CDataType) * c_m.mDesc.GetElementSpaceSize());

    a_m_device_buf.ToDevice(a_m.mData.data());

    std::array<const void*, 1> input = {a_m_device_buf.GetDeviceBuffer()};
    std::array<void*, 1> output      = {c_m_device_buf.GetDeviceBuffer()};

    std::array<ck::index_t, 1> abc_lengths = {M};
    std::array<ck::index_t, 1> a_strides   = {1};
    std::array<ck::index_t, 1> c_strides   = {1};

    auto broadcastUnaryConvert =
        ck::tensor_operation::device::DeviceElementwiseImpl<ck::Tuple<ADataType>,
                                                            ck::Tuple<CDataType>,
                                                            UnaryConvert,
                                                            1,
                                                            64,
                                                            16,
                                                            16,
                                                            2,
                                                            2,
                                                            ck::Sequence<1, 0>,
                                                            ck::Sequence<1>,
                                                            ck::Sequence<1>>{};
    auto argument = broadcastUnaryConvert.MakeArgumentPointer(
        abc_lengths, {a_strides}, {c_strides}, input, output, UnaryConvert{});

    if(!broadcastUnaryConvert.IsSupportedArgument(argument.get()))
    {
        throw std::runtime_error(
            "The runtime parameters seems not supported by the device instance, exiting!");
    };

    auto broadcastUnaryConvert_invoker_ptr = broadcastUnaryConvert.MakeInvokerPointer();
    float ave_time =
        broadcastUnaryConvert_invoker_ptr->Run(argument.get(), StreamConfig{nullptr, time_kernel});

    std::cout << "Perf: " << ave_time << " ms" << std::endl;

    bool pass = true;
    if(do_verification)
    {
        c_m_device_buf.FromDevice(c_m.mData.data());
        Tensor<CDataType> host_c_m(f_host_tensor_descriptor1d(M, 1));

        unary_host_elementwise1D<Tensor<ADataType>, Tensor<CDataType>, UnaryConvert>(
            host_c_m, a_m, M, UnaryConvert{});

        pass &= ck::utils::check_err(c_m, host_c_m, "Error: Incorrect results c", 5e-3, 5e-3);
    }

    return pass;
}

template <typename ADataType, typename CDataType>
bool verifyF8SRConvert()
{
    bool do_verification = true;
    bool time_kernel     = false;

    ck::index_t M = 1024;

    auto f_host_tensor_descriptor1d = [](std::size_t len, std::size_t stride) {
        return HostTensorDescriptor({len}, {stride});
    };

    Tensor<ADataType> a_m(f_host_tensor_descriptor1d(M, 1));
    Tensor<CDataType> c_m(f_host_tensor_descriptor1d(M, 1));

    a_m.GenerateTensorValue(GeneratorTensor_3<ADataType>{-5000, 5000});

    DeviceMem a_m_device_buf(sizeof(ADataType) * a_m.mDesc.GetElementSpaceSize());
    DeviceMem c_m_device_buf(sizeof(CDataType) * c_m.mDesc.GetElementSpaceSize());

    a_m_device_buf.ToDevice(a_m.mData.data());

    std::array<const void*, 1> input = {a_m_device_buf.GetDeviceBuffer()};
    std::array<void*, 1> output      = {c_m_device_buf.GetDeviceBuffer()};

    std::array<ck::index_t, 1> abc_lengths = {M};
    std::array<ck::index_t, 1> a_strides   = {1};
    std::array<ck::index_t, 1> c_strides   = {1};

    auto broadcastF8SRConvert =
        ck::tensor_operation::device::DeviceElementwiseImpl<ck::Tuple<ADataType>,
                                                            ck::Tuple<CDataType>,
                                                            ConvertF8SR,
                                                            1,
                                                            64,
                                                            16,
                                                            16,
                                                            2,
                                                            2,
                                                            ck::Sequence<1, 0>,
                                                            ck::Sequence<1>,
                                                            ck::Sequence<1>>{};
    auto argument = broadcastF8SRConvert.MakeArgumentPointer(
        abc_lengths, {a_strides}, {c_strides}, input, output, ConvertF8SR{});

    if(!broadcastF8SRConvert.IsSupportedArgument(argument.get()))
    {
        throw std::runtime_error(
            "The runtime parameters seems not supported by the device instance, exiting!");
    };

    auto broadcastF8SRConvert_invoker_ptr = broadcastF8SRConvert.MakeInvokerPointer();
    float ave_time =
        broadcastF8SRConvert_invoker_ptr->Run(argument.get(), StreamConfig{nullptr, time_kernel});

    std::cout << "Perf: " << ave_time << " ms" << std::endl;

    bool pass = true;
    if(do_verification)
    {
        c_m_device_buf.FromDevice(c_m.mData.data());
        Tensor<CDataType> host_c_m(f_host_tensor_descriptor1d(M, 1));

        f8sr_host_elementwise1D<Tensor<ADataType>, Tensor<CDataType>, ConvertF8SR>(
            host_c_m, a_m, M, ConvertF8SR{});

        pass &= ck::utils::check_err(c_m, host_c_m, "Error: Incorrect results c", 1, 1);
    }

    return pass;
}

template <typename ADataType, typename CDataType>
bool verifyF8RNEConvert()
{
    bool do_verification = true;
    bool time_kernel     = false;

    ck::index_t M = 1024;

    auto f_host_tensor_descriptor1d = [](std::size_t len, std::size_t stride) {
        return HostTensorDescriptor({len}, {stride});
    };

    Tensor<ADataType> a_m(f_host_tensor_descriptor1d(M, 1));
    Tensor<CDataType> c_m(f_host_tensor_descriptor1d(M, 1));

    a_m.GenerateTensorValue(GeneratorTensor_3<ADataType>{-5000, 5000});

    DeviceMem a_m_device_buf(sizeof(ADataType) * a_m.mDesc.GetElementSpaceSize());
    DeviceMem c_m_device_buf(sizeof(CDataType) * c_m.mDesc.GetElementSpaceSize());

    a_m_device_buf.ToDevice(a_m.mData.data());

    std::array<const void*, 1> input = {a_m_device_buf.GetDeviceBuffer()};
    std::array<void*, 1> output      = {c_m_device_buf.GetDeviceBuffer()};

    std::array<ck::index_t, 1> abc_lengths = {M};
    std::array<ck::index_t, 1> a_strides   = {1};
    std::array<ck::index_t, 1> c_strides   = {1};

    auto broadcastF8RNEConvert =
        ck::tensor_operation::device::DeviceElementwiseImpl<ck::Tuple<ADataType>,
                                                            ck::Tuple<CDataType>,
                                                            ConvertF8RNE,
                                                            1,
                                                            64,
                                                            16,
                                                            16,
                                                            2,
                                                            2,
                                                            ck::Sequence<1, 0>,
                                                            ck::Sequence<1>,
                                                            ck::Sequence<1>>{};
    auto argument = broadcastF8RNEConvert.MakeArgumentPointer(
        abc_lengths, {a_strides}, {c_strides}, input, output, ConvertF8RNE{});

    if(!broadcastF8RNEConvert.IsSupportedArgument(argument.get()))
    {
        throw std::runtime_error(
            "The runtime parameters seems not supported by the device instance, exiting!");
    };

    auto broadcastF8RNEConvert_invoker_ptr = broadcastF8RNEConvert.MakeInvokerPointer();
    float ave_time =
        broadcastF8RNEConvert_invoker_ptr->Run(argument.get(), StreamConfig{nullptr, time_kernel});

    std::cout << "Perf: " << ave_time << " ms" << std::endl;

    bool pass = true;
    if(do_verification)
    {
        c_m_device_buf.FromDevice(c_m.mData.data());
        Tensor<CDataType> host_c_m(f_host_tensor_descriptor1d(M, 1));

        f8rne_host_elementwise1D<Tensor<ADataType>, Tensor<CDataType>, ConvertF8RNE>(
            host_c_m, a_m, M, ConvertF8RNE{});

        pass &= ck::utils::check_err(c_m, host_c_m, "Error: Incorrect results c", 1, 1);
    }

    return pass;
}

int main()
{
    bool pass = true;

    // Test case for v_cvt_f16_bf8
    pass = verifyUnaryConvert<BF8, F16>();
    // Test case for v_cvt_f16_fp8
    pass &= verifyUnaryConvert<F8, F16>();
    // Test case for v_cvt_sr_bf8_f16
    pass &= verifyF8SRConvert<F16, BF8>();
    // Test case for v_cvt_sr_fp8_f16
    pass &= verifyF8SRConvert<F16, F8>();
    // Test case for v_cvt_pk_bf8_f16
    pass &= verifyF8RNEConvert<F16, BF8>();
    // Test case for v_cvt_pk_fp8_f16
    pass &= verifyF8RNEConvert<F16, F8>();

    return pass ? 0 : 1;
}
