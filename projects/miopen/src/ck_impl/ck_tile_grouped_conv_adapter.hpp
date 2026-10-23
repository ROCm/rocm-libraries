// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <ck/library/tensor_operation_instance/device_operation_instance_factory.hpp>

#if defined(MIOPEN_USE_CK_TILE) && MIOPEN_USE_CK_TILE && defined(CK_IMPL_BUILDING_DLL)
#include "ck_tile_grouped_conv_common.hpp"
#include <ck/tensor_operation/gpu/device/device_grouped_conv_fwd_multiple_abd.hpp>
#include <ck/tensor_operation/gpu/device/device_grouped_conv_bwd_data_multiple_d.hpp>
#include <ck/tensor_operation/gpu/device/device_grouped_conv_bwd_weight.hpp>
#include <ck/tensor_operation/gpu/element/element_wise_operation.hpp>
#include <stdexcept>
#include <type_traits>

namespace miopen::solver::ck_tile_detail {

namespace device  = ck::tensor_operation::device;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

struct Argument : device::BaseArgument, Arguments
{
};

struct Invoker : device::BaseInvoker
{
    float Run(const device::BaseArgument* base, const StreamConfig& config) override
    {
        const auto* arg = dynamic_cast<const Argument*>(base);
        if(arg == nullptr)
            throw std::invalid_argument("CK Tile: mismatched convolution argument");
        auto args      = static_cast<const Arguments&>(*arg);
        args.workspace = arg->p_workspace_;
        return ck_tile_detail::Run(args, config.stream_id_);
    }
};

template <typename T>
inline constexpr bool IsTileDataType =
    std::is_same_v<T, ck::half_t> || std::is_same_v<T, ck::bhalf_t> || std::is_same_v<T, float>;

template <typename T>
constexpr DataType GetDataType()
{
    if constexpr(std::is_same_v<T, ck::half_t>)
        return DataType::Half;
    else if constexpr(std::is_same_v<T, ck::bhalf_t>)
        return DataType::BFloat16;
    else
        return DataType::Float;
}

template <typename Base, Direction Op, typename T, int NDim>
struct Operation : Base
{
    int config_index = 0;

    bool IsSupportedArgument(const device::BaseArgument* base) override
    {
        const auto* arg = dynamic_cast<const Argument*>(base);
        return arg != nullptr && arg->direction == Op && arg->data_type == GetDataType<T>() &&
               arg->spatial_dims == NDim && arg->config_index == config_index &&
               ck_tile_detail::IsSupported(*arg);
    }

    std::string GetTypeString() const override
    {
        return GetKernelId(Op, GetDataType<T>(), NDim, config_index);
    }

    size_t GetWorkSpaceSize(const device::BaseArgument* base) const override
    {
        const auto* arg = dynamic_cast<const Argument*>(base);
        if(arg == nullptr)
            throw std::invalid_argument("CK Tile: mismatched workspace argument");
        return ck_tile_detail::GetWorkspaceSize(*arg);
    }

    std::unique_ptr<device::BaseInvoker> MakeInvokerPointer() override
    {
        return std::make_unique<Invoker>();
    }

    template <typename Index>
    std::unique_ptr<device::BaseArgument> MakeArgument(const void* x,
                                                       const void* w,
                                                       const void* y,
                                                       const std::array<Index, NDim + 3>& xl,
                                                       const std::array<Index, NDim + 3>& xs,
                                                       const std::array<Index, NDim + 3>& wl,
                                                       const std::array<Index, NDim + 3>& ws,
                                                       const std::array<Index, NDim + 3>& yl,
                                                       const std::array<Index, NDim + 3>& ys,
                                                       const std::array<Index, NDim>& stride,
                                                       const std::array<Index, NDim>& dilation,
                                                       const std::array<Index, NDim>& left,
                                                       const std::array<Index, NDim>& right,
                                                       int split) const
    {
        auto arg          = std::make_unique<Argument>();
        arg->direction    = Op;
        arg->data_type    = GetDataType<T>();
        arg->spatial_dims = NDim;
        arg->config_index = config_index;
        arg->input        = x;
        arg->weight       = w;
        arg->output       = y;
        arg->split_k      = split;
        std::copy(xl.begin(), xl.end(), arg->input_lengths.begin());
        std::copy(xs.begin(), xs.end(), arg->input_strides.begin());
        std::copy(wl.begin(), wl.end(), arg->weight_lengths.begin());
        std::copy(ws.begin(), ws.end(), arg->weight_strides.begin());
        std::copy(yl.begin(), yl.end(), arg->output_lengths.begin());
        std::copy(ys.begin(), ys.end(), arg->output_strides.begin());
        std::copy(stride.begin(), stride.end(), arg->strides.begin());
        std::copy(dilation.begin(), dilation.end(), arg->dilations.begin());
        std::copy(left.begin(), left.end(), arg->left_pads.begin());
        std::copy(right.begin(), right.end(), arg->right_pads.begin());
        return arg;
    }
};

// Only the exact homogeneous, PassThrough, packed channels-last interfaces below
// receive Tile candidates. Fused/scaled/nonpacked and other compute contracts stay legacy.
template <typename Base>
struct Adapter;

template <int NDim, typename T, typename InLayout, typename WeiLayout, typename OutLayout>
    requires(IsTileDataType<T> && (NDim == 2 || NDim == 3))
struct Adapter<device::DeviceGroupedConvFwdMultipleABD<NDim,
                                                       InLayout,
                                                       WeiLayout,
                                                       ck::Tuple<>,
                                                       OutLayout,
                                                       T,
                                                       T,
                                                       ck::Tuple<>,
                                                       T,
                                                       PassThrough,
                                                       PassThrough,
                                                       PassThrough,
                                                       T,
                                                       T>>
    : Operation<device::DeviceGroupedConvFwdMultipleABD<NDim,
                                                        InLayout,
                                                        WeiLayout,
                                                        ck::Tuple<>,
                                                        OutLayout,
                                                        T,
                                                        T,
                                                        ck::Tuple<>,
                                                        T,
                                                        PassThrough,
                                                        PassThrough,
                                                        PassThrough,
                                                        T,
                                                        T>,
                Direction::Forward,
                T,
                NDim>
{
    using Base = Operation<device::DeviceGroupedConvFwdMultipleABD<NDim,
                                                                   InLayout,
                                                                   WeiLayout,
                                                                   ck::Tuple<>,
                                                                   OutLayout,
                                                                   T,
                                                                   T,
                                                                   ck::Tuple<>,
                                                                   T,
                                                                   PassThrough,
                                                                   PassThrough,
                                                                   PassThrough,
                                                                   T,
                                                                   T>,
                           Direction::Forward,
                           T,
                           NDim>;

    std::unique_ptr<device::BaseArgument>
    MakeArgumentPointer(const void* x,
                        const void* w,
                        const std::array<const void*, 0>&,
                        void* y,
                        const std::array<ck::index_t, NDim + 3>& xl,
                        const std::array<ck::index_t, NDim + 3>& xs,
                        const std::array<ck::index_t, NDim + 3>& wl,
                        const std::array<ck::index_t, NDim + 3>& ws,
                        const std::array<std::array<ck::index_t, NDim + 3>, 0>&,
                        const std::array<std::array<ck::index_t, NDim + 3>, 0>&,
                        const std::array<ck::index_t, NDim + 3>& yl,
                        const std::array<ck::index_t, NDim + 3>& ys,
                        const std::array<ck::index_t, NDim>& stride,
                        const std::array<ck::index_t, NDim>& dilation,
                        const std::array<ck::index_t, NDim>& left,
                        const std::array<ck::index_t, NDim>& right,
                        const PassThrough&,
                        const PassThrough&,
                        const PassThrough&) override
    {
        return Base::MakeArgument(
            x, w, y, xl, xs, wl, ws, yl, ys, stride, dilation, left, right, 1);
    }

    std::unique_ptr<device::BaseArgument>
    MakeArgumentPointer(const void* x,
                        const void* w,
                        const std::array<const void*, 0>&,
                        void* y,
                        const std::array<ck::long_index_t, NDim + 3>& xl,
                        const std::array<ck::long_index_t, NDim + 3>& xs,
                        const std::array<ck::long_index_t, NDim + 3>& wl,
                        const std::array<ck::long_index_t, NDim + 3>& ws,
                        const std::array<std::array<ck::long_index_t, NDim + 3>, 0>&,
                        const std::array<std::array<ck::long_index_t, NDim + 3>, 0>&,
                        const std::array<ck::long_index_t, NDim + 3>& yl,
                        const std::array<ck::long_index_t, NDim + 3>& ys,
                        const std::array<ck::long_index_t, NDim>& stride,
                        const std::array<ck::long_index_t, NDim>& dilation,
                        const std::array<ck::long_index_t, NDim>& left,
                        const std::array<ck::long_index_t, NDim>& right,
                        const PassThrough&,
                        const PassThrough&,
                        const PassThrough&) override
    {
        return Base::MakeArgument(
            x, w, y, xl, xs, wl, ws, yl, ys, stride, dilation, left, right, 1);
    }
};

template <int NDim, typename T, typename OutLayout, typename WeiLayout, typename InLayout>
    requires(IsTileDataType<T> && (NDim == 2 || NDim == 3))
struct Adapter<device::DeviceGroupedConvBwdDataMultipleD<NDim,
                                                         OutLayout,
                                                         WeiLayout,
                                                         ck::Tuple<>,
                                                         InLayout,
                                                         T,
                                                         T,
                                                         ck::Tuple<>,
                                                         T,
                                                         PassThrough,
                                                         PassThrough,
                                                         PassThrough,
                                                         T,
                                                         T>>
    : Operation<device::DeviceGroupedConvBwdDataMultipleD<NDim,
                                                          OutLayout,
                                                          WeiLayout,
                                                          ck::Tuple<>,
                                                          InLayout,
                                                          T,
                                                          T,
                                                          ck::Tuple<>,
                                                          T,
                                                          PassThrough,
                                                          PassThrough,
                                                          PassThrough,
                                                          T,
                                                          T>,
                Direction::BackwardData,
                T,
                NDim>
{
    using Base = Operation<device::DeviceGroupedConvBwdDataMultipleD<NDim,
                                                                     OutLayout,
                                                                     WeiLayout,
                                                                     ck::Tuple<>,
                                                                     InLayout,
                                                                     T,
                                                                     T,
                                                                     ck::Tuple<>,
                                                                     T,
                                                                     PassThrough,
                                                                     PassThrough,
                                                                     PassThrough,
                                                                     T,
                                                                     T>,
                           Direction::BackwardData,
                           T,
                           NDim>;

    std::unique_ptr<device::BaseArgument>
    MakeArgumentPointer(const void* y,
                        const void* w,
                        const std::array<const void*, 0>&,
                        void* x,
                        const std::array<ck::index_t, NDim + 3>& yl,
                        const std::array<ck::index_t, NDim + 3>& ys,
                        const std::array<ck::index_t, NDim + 3>& wl,
                        const std::array<ck::index_t, NDim + 3>& ws,
                        const std::array<std::array<ck::index_t, NDim + 3>, 0>&,
                        const std::array<std::array<ck::index_t, NDim + 3>, 0>&,
                        const std::array<ck::index_t, NDim + 3>& xl,
                        const std::array<ck::index_t, NDim + 3>& xs,
                        const std::array<ck::index_t, NDim>& stride,
                        const std::array<ck::index_t, NDim>& dilation,
                        const std::array<ck::index_t, NDim>& left,
                        const std::array<ck::index_t, NDim>& right,
                        const PassThrough&,
                        const PassThrough&,
                        const PassThrough&,
                        ck::index_t split = 1) override
    {
        return Base::MakeArgument(
            x, w, y, xl, xs, wl, ws, yl, ys, stride, dilation, left, right, split);
    }

    std::unique_ptr<device::BaseArgument>
    MakeArgumentPointer(const void* y,
                        const void* w,
                        const std::array<const void*, 0>&,
                        void* x,
                        const std::array<ck::long_index_t, NDim + 3>& yl,
                        const std::array<ck::long_index_t, NDim + 3>& ys,
                        const std::array<ck::long_index_t, NDim + 3>& wl,
                        const std::array<ck::long_index_t, NDim + 3>& ws,
                        const std::array<std::array<ck::long_index_t, NDim + 3>, 0>&,
                        const std::array<std::array<ck::long_index_t, NDim + 3>, 0>&,
                        const std::array<ck::long_index_t, NDim + 3>& xl,
                        const std::array<ck::long_index_t, NDim + 3>& xs,
                        const std::array<ck::long_index_t, NDim>& stride,
                        const std::array<ck::long_index_t, NDim>& dilation,
                        const std::array<ck::long_index_t, NDim>& left,
                        const std::array<ck::long_index_t, NDim>& right,
                        const PassThrough&,
                        const PassThrough&,
                        const PassThrough&,
                        ck::index_t split = 1) override
    {
        return Base::MakeArgument(
            x, w, y, xl, xs, wl, ws, yl, ys, stride, dilation, left, right, split);
    }
};

template <int NDim, typename T, typename InLayout, typename WeiLayout, typename OutLayout>
    requires(IsTileDataType<T> && (NDim == 2 || NDim == 3))
struct Adapter<device::DeviceGroupedConvBwdWeight<NDim,
                                                  InLayout,
                                                  WeiLayout,
                                                  OutLayout,
                                                  T,
                                                  T,
                                                  T,
                                                  PassThrough,
                                                  PassThrough,
                                                  PassThrough,
                                                  T,
                                                  T>>
    : Operation<device::DeviceGroupedConvBwdWeight<NDim,
                                                   InLayout,
                                                   WeiLayout,
                                                   OutLayout,
                                                   T,
                                                   T,
                                                   T,
                                                   PassThrough,
                                                   PassThrough,
                                                   PassThrough,
                                                   T,
                                                   T>,
                Direction::BackwardWeight,
                T,
                NDim>
{
    using Base = Operation<device::DeviceGroupedConvBwdWeight<NDim,
                                                              InLayout,
                                                              WeiLayout,
                                                              OutLayout,
                                                              T,
                                                              T,
                                                              T,
                                                              PassThrough,
                                                              PassThrough,
                                                              PassThrough,
                                                              T,
                                                              T>,
                           Direction::BackwardWeight,
                           T,
                           NDim>;

    std::unique_ptr<device::BaseArgument>
    MakeArgumentPointer(const void* x,
                        void* w,
                        const void* y,
                        const std::array<ck::index_t, NDim + 3>& xl,
                        const std::array<ck::index_t, NDim + 3>& xs,
                        const std::array<ck::index_t, NDim + 3>& wl,
                        const std::array<ck::index_t, NDim + 3>& ws,
                        const std::array<ck::index_t, NDim + 3>& yl,
                        const std::array<ck::index_t, NDim + 3>& ys,
                        const std::array<ck::index_t, NDim>& stride,
                        const std::array<ck::index_t, NDim>& dilation,
                        const std::array<ck::index_t, NDim>& left,
                        const std::array<ck::index_t, NDim>& right,
                        PassThrough,
                        PassThrough,
                        PassThrough,
                        ck::index_t split) override
    {
        return Base::MakeArgument(
            x, w, y, xl, xs, wl, ws, yl, ys, stride, dilation, left, right, split);
    }

    std::unique_ptr<device::BaseArgument>
    MakeArgumentPointer(const void* x,
                        void* w,
                        const void* y,
                        const std::array<ck::long_index_t, NDim + 3>& xl,
                        const std::array<ck::long_index_t, NDim + 3>& xs,
                        const std::array<ck::long_index_t, NDim + 3>& wl,
                        const std::array<ck::long_index_t, NDim + 3>& ws,
                        const std::array<ck::long_index_t, NDim + 3>& yl,
                        const std::array<ck::long_index_t, NDim + 3>& ys,
                        const std::array<ck::long_index_t, NDim>& stride,
                        const std::array<ck::long_index_t, NDim>& dilation,
                        const std::array<ck::long_index_t, NDim>& left,
                        const std::array<ck::long_index_t, NDim>& right,
                        PassThrough,
                        PassThrough,
                        PassThrough,
                        ck::index_t split) override
    {
        return Base::MakeArgument(
            x, w, y, xl, xs, wl, ws, yl, ys, stride, dilation, left, right, split);
    }
};

template <typename Base, typename = void>
struct Catalog
{
    static void Append(std::vector<std::unique_ptr<Base>>&) {}
};

template <typename Base>
struct Catalog<Base, std::void_t<decltype(sizeof(Adapter<Base>))>>
{
    static void Append(std::vector<std::unique_ptr<Base>>& instances)
    {
        // Only the six plain channels-last factory aliases use this wrapper.
        for(size_t i = 0; i < KernelConfigs.size(); ++i)
        {
            auto instance          = std::make_unique<Adapter<Base>>();
            instance->config_index = static_cast<int>(i);
            if(!instance->GetTypeString().empty())
                instances.push_back(std::move(instance));
        }
    }
};

} // namespace miopen::solver::ck_tile_detail
#endif

namespace miopen::solver {

template <typename DeviceOp>
struct CKTileOperationInstanceFactory
{
    static std::vector<std::unique_ptr<DeviceOp>> GetInstances()
    {
        auto instances = ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
            DeviceOp>::GetInstances();
#if defined(MIOPEN_USE_CK_TILE) && MIOPEN_USE_CK_TILE && defined(CK_IMPL_BUILDING_DLL)
        ck_tile_detail::Catalog<DeviceOp>::Append(instances);
#endif
        return instances;
    }
};

} // namespace miopen::solver
