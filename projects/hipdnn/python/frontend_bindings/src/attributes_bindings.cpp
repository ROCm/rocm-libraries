// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "bindings.hpp"

#include <hipdnn_frontend/attributes/BatchnormAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributes.hpp>
#include <hipdnn_frontend/attributes/BatchnormInferenceAttributesVarianceExt.hpp>
#include <hipdnn_frontend/attributes/BlockScaleDequantizeAttributes.hpp>
#include <hipdnn_frontend/attributes/BlockScaleQuantizeAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionDgradAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/ConvolutionWgradAttributes.hpp>
#include <hipdnn_frontend/attributes/CustomOpAttributes.hpp>
#include <hipdnn_frontend/attributes/LayernormAttributes.hpp>
#include <hipdnn_frontend/attributes/LayernormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/MatmulAttributes.hpp>
#include <hipdnn_frontend/attributes/MoeGroupedMatmulAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>
#include <hipdnn_frontend/attributes/RMSNormAttributes.hpp>
#include <hipdnn_frontend/attributes/RMSNormBackwardAttributes.hpp>
#include <hipdnn_frontend/attributes/ReductionAttributes.hpp>
#include <hipdnn_frontend/attributes/ResampleBwdAttributes.hpp>
#include <hipdnn_frontend/attributes/ResampleFwdAttributes.hpp>
#include <hipdnn_frontend/attributes/SdpaAttributes.hpp>
#include <hipdnn_frontend/attributes/SdpaBackwardAttributes.hpp>
#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;
using namespace hipdnn_frontend;

namespace
{
using Tensor = std::shared_ptr<graph::TensorAttributes>;

template <typename Attr>
using TensorSetter = Attr& (Attr::*)(const Tensor&);

template <typename Attr>
using TensorListSetter = Attr& (Attr::*)(const std::vector<Tensor>&);

template <typename Attr>
using TwoTensorSetter = Attr& (Attr::*)(const Tensor&, const Tensor&);

template <typename Attr>
using ThreeTensorSetter = Attr& (Attr::*)(const Tensor&, const Tensor&, const Tensor&);

template <typename Attr>
using Int64VectorSetter = Attr& (Attr::*)(const std::vector<int64_t>&);

template <typename Attr>
using Int32VectorSetter = Attr& (Attr::*)(const std::vector<int32_t>&);
} // namespace

void attributesBindings(nb::module_& m)
{
    // BatchnormAttributes
    nb::class_<graph::BatchnormAttributes>(m, "BatchnormAttributes")
        .def(nb::init<>())
        .def("set_name", &graph::BatchnormAttributes::set_name, nb::rv_policy::reference_internal)
        .def("get_name", &graph::BatchnormAttributes::get_name)
        .def("set_compute_data_type",
             &graph::BatchnormAttributes::set_compute_data_type,
             nb::rv_policy::reference_internal)
        .def("get_compute_data_type", &graph::BatchnormAttributes::get_compute_data_type)
        .def("set_x",
             static_cast<TensorSetter<graph::BatchnormAttributes>>(
                 &graph::BatchnormAttributes::set_x),
             nb::rv_policy::reference_internal)
        .def("set_scale",
             static_cast<TensorSetter<graph::BatchnormAttributes>>(
                 &graph::BatchnormAttributes::set_scale),
             nb::rv_policy::reference_internal)
        .def("set_bias",
             static_cast<TensorSetter<graph::BatchnormAttributes>>(
                 &graph::BatchnormAttributes::set_bias),
             nb::rv_policy::reference_internal)
        .def("set_epsilon",
             static_cast<TensorSetter<graph::BatchnormAttributes>>(
                 &graph::BatchnormAttributes::set_epsilon),
             nb::rv_policy::reference_internal)
        .def("set_peer_stats",
             static_cast<TensorListSetter<graph::BatchnormAttributes>>(
                 &graph::BatchnormAttributes::set_peer_stats),
             nb::rv_policy::reference_internal)
        .def("set_prev_running_mean",
             static_cast<TensorSetter<graph::BatchnormAttributes>>(
                 &graph::BatchnormAttributes::set_prev_running_mean),
             nb::rv_policy::reference_internal)
        .def("set_prev_running_variance",
             static_cast<TensorSetter<graph::BatchnormAttributes>>(
                 &graph::BatchnormAttributes::set_prev_running_variance),
             nb::rv_policy::reference_internal)
        .def("set_momentum",
             static_cast<TensorSetter<graph::BatchnormAttributes>>(
                 &graph::BatchnormAttributes::set_momentum),
             nb::rv_policy::reference_internal)
        .def("set_y",
             static_cast<TensorSetter<graph::BatchnormAttributes>>(
                 &graph::BatchnormAttributes::set_y),
             nb::rv_policy::reference_internal)
        .def("set_mean",
             static_cast<TensorSetter<graph::BatchnormAttributes>>(
                 &graph::BatchnormAttributes::set_mean),
             nb::rv_policy::reference_internal)
        .def("set_inv_variance",
             static_cast<TensorSetter<graph::BatchnormAttributes>>(
                 &graph::BatchnormAttributes::set_inv_variance),
             nb::rv_policy::reference_internal)
        .def("set_next_running_mean",
             static_cast<TensorSetter<graph::BatchnormAttributes>>(
                 &graph::BatchnormAttributes::set_next_running_mean),
             nb::rv_policy::reference_internal)
        .def("set_next_running_variance",
             static_cast<TensorSetter<graph::BatchnormAttributes>>(
                 &graph::BatchnormAttributes::set_next_running_variance),
             nb::rv_policy::reference_internal)
        .def("set_previous_running_stats",
             static_cast<ThreeTensorSetter<graph::BatchnormAttributes>>(
                 &graph::BatchnormAttributes::set_previous_running_stats),
             nb::rv_policy::reference_internal)
        .def("get_x", &graph::BatchnormAttributes::get_x)
        .def("get_scale", &graph::BatchnormAttributes::get_scale)
        .def("get_bias", &graph::BatchnormAttributes::get_bias)
        .def("get_epsilon", &graph::BatchnormAttributes::get_epsilon)
        .def("get_peer_stats", &graph::BatchnormAttributes::get_peer_stats)
        .def("get_prev_running_mean", &graph::BatchnormAttributes::get_prev_running_mean)
        .def("get_prev_running_variance", &graph::BatchnormAttributes::get_prev_running_variance)
        .def("get_momentum", &graph::BatchnormAttributes::get_momentum)
        .def("get_y", &graph::BatchnormAttributes::get_y)
        .def("get_mean", &graph::BatchnormAttributes::get_mean)
        .def("get_inv_variance", &graph::BatchnormAttributes::get_inv_variance)
        .def("get_next_running_mean", &graph::BatchnormAttributes::get_next_running_mean)
        .def("get_next_running_variance", &graph::BatchnormAttributes::get_next_running_variance);

    // BatchnormBackwardAttributes
    nb::class_<graph::BatchnormBackwardAttributes>(m, "BatchnormBackwardAttributes")
        .def(nb::init<>())
        .def("set_name",
             &graph::BatchnormBackwardAttributes::set_name,
             nb::rv_policy::reference_internal)
        .def("get_name", &graph::BatchnormBackwardAttributes::get_name)
        .def("set_compute_data_type",
             &graph::BatchnormBackwardAttributes::set_compute_data_type,
             nb::rv_policy::reference_internal)
        .def("get_compute_data_type", &graph::BatchnormBackwardAttributes::get_compute_data_type)
        .def("set_dy",
             static_cast<TensorSetter<graph::BatchnormBackwardAttributes>>(
                 &graph::BatchnormBackwardAttributes::set_dy),
             nb::rv_policy::reference_internal)
        .def("set_x",
             static_cast<TensorSetter<graph::BatchnormBackwardAttributes>>(
                 &graph::BatchnormBackwardAttributes::set_x),
             nb::rv_policy::reference_internal)
        .def("set_scale",
             static_cast<TensorSetter<graph::BatchnormBackwardAttributes>>(
                 &graph::BatchnormBackwardAttributes::set_scale),
             nb::rv_policy::reference_internal)
        .def("set_mean",
             static_cast<TensorSetter<graph::BatchnormBackwardAttributes>>(
                 &graph::BatchnormBackwardAttributes::set_mean),
             nb::rv_policy::reference_internal)
        .def("set_inv_variance",
             static_cast<TensorSetter<graph::BatchnormBackwardAttributes>>(
                 &graph::BatchnormBackwardAttributes::set_inv_variance),
             nb::rv_policy::reference_internal)
        .def("set_dx",
             static_cast<TensorSetter<graph::BatchnormBackwardAttributes>>(
                 &graph::BatchnormBackwardAttributes::set_dx),
             nb::rv_policy::reference_internal)
        .def("set_dscale",
             static_cast<TensorSetter<graph::BatchnormBackwardAttributes>>(
                 &graph::BatchnormBackwardAttributes::set_dscale),
             nb::rv_policy::reference_internal)
        .def("set_dbias",
             static_cast<TensorSetter<graph::BatchnormBackwardAttributes>>(
                 &graph::BatchnormBackwardAttributes::set_dbias),
             nb::rv_policy::reference_internal)
        .def("set_peer_stats",
             static_cast<TensorListSetter<graph::BatchnormBackwardAttributes>>(
                 &graph::BatchnormBackwardAttributes::set_peer_stats),
             nb::rv_policy::reference_internal)
        .def("set_saved_mean_and_inv_variance",
             static_cast<TwoTensorSetter<graph::BatchnormBackwardAttributes>>(
                 &graph::BatchnormBackwardAttributes::set_saved_mean_and_inv_variance),
             nb::rv_policy::reference_internal)
        .def("get_dy", &graph::BatchnormBackwardAttributes::get_dy)
        .def("get_x", &graph::BatchnormBackwardAttributes::get_x)
        .def("get_scale", &graph::BatchnormBackwardAttributes::get_scale)
        .def("get_mean", &graph::BatchnormBackwardAttributes::get_mean)
        .def("get_inv_variance", &graph::BatchnormBackwardAttributes::get_inv_variance)
        .def("get_dx", &graph::BatchnormBackwardAttributes::get_dx)
        .def("get_dscale", &graph::BatchnormBackwardAttributes::get_dscale)
        .def("get_dbias", &graph::BatchnormBackwardAttributes::get_dbias)
        .def("get_peer_stats", &graph::BatchnormBackwardAttributes::get_peer_stats);

    // BatchnormInferenceAttributes
    nb::class_<graph::BatchnormInferenceAttributes>(m, "BatchnormInferenceAttributes")
        .def(nb::init<>())
        .def("set_name",
             &graph::BatchnormInferenceAttributes::set_name,
             nb::rv_policy::reference_internal)
        .def("get_name", &graph::BatchnormInferenceAttributes::get_name)
        .def("set_compute_data_type",
             &graph::BatchnormInferenceAttributes::set_compute_data_type,
             nb::rv_policy::reference_internal)
        .def("get_compute_data_type", &graph::BatchnormInferenceAttributes::get_compute_data_type)
        .def("set_x",
             static_cast<TensorSetter<graph::BatchnormInferenceAttributes>>(
                 &graph::BatchnormInferenceAttributes::set_x),
             nb::rv_policy::reference_internal)
        .def("set_mean",
             static_cast<TensorSetter<graph::BatchnormInferenceAttributes>>(
                 &graph::BatchnormInferenceAttributes::set_mean),
             nb::rv_policy::reference_internal)
        .def("set_inv_variance",
             static_cast<TensorSetter<graph::BatchnormInferenceAttributes>>(
                 &graph::BatchnormInferenceAttributes::set_inv_variance),
             nb::rv_policy::reference_internal)
        .def("set_scale",
             static_cast<TensorSetter<graph::BatchnormInferenceAttributes>>(
                 &graph::BatchnormInferenceAttributes::set_scale),
             nb::rv_policy::reference_internal)
        .def("set_bias",
             static_cast<TensorSetter<graph::BatchnormInferenceAttributes>>(
                 &graph::BatchnormInferenceAttributes::set_bias),
             nb::rv_policy::reference_internal)
        .def("set_y",
             static_cast<TensorSetter<graph::BatchnormInferenceAttributes>>(
                 &graph::BatchnormInferenceAttributes::set_y),
             nb::rv_policy::reference_internal)
        .def("get_x", &graph::BatchnormInferenceAttributes::get_x)
        .def("get_mean", &graph::BatchnormInferenceAttributes::get_mean)
        .def("get_inv_variance", &graph::BatchnormInferenceAttributes::get_inv_variance)
        .def("get_scale", &graph::BatchnormInferenceAttributes::get_scale)
        .def("get_bias", &graph::BatchnormInferenceAttributes::get_bias)
        .def("get_y", &graph::BatchnormInferenceAttributes::get_y);

    // BatchnormInferenceAttributesVarianceExt
    nb::class_<graph::BatchnormInferenceAttributesVarianceExt>(
        m, "BatchnormInferenceAttributesVarianceExt")
        .def(nb::init<>())
        .def("set_name",
             &graph::BatchnormInferenceAttributesVarianceExt::set_name,
             nb::rv_policy::reference_internal)
        .def("get_name", &graph::BatchnormInferenceAttributesVarianceExt::get_name)
        .def("set_compute_data_type",
             &graph::BatchnormInferenceAttributesVarianceExt::set_compute_data_type,
             nb::rv_policy::reference_internal)
        .def("get_compute_data_type",
             &graph::BatchnormInferenceAttributesVarianceExt::get_compute_data_type)
        .def("set_x",
             static_cast<TensorSetter<graph::BatchnormInferenceAttributesVarianceExt>>(
                 &graph::BatchnormInferenceAttributesVarianceExt::set_x),
             nb::rv_policy::reference_internal)
        .def("set_mean",
             static_cast<TensorSetter<graph::BatchnormInferenceAttributesVarianceExt>>(
                 &graph::BatchnormInferenceAttributesVarianceExt::set_mean),
             nb::rv_policy::reference_internal)
        .def("set_variance",
             static_cast<TensorSetter<graph::BatchnormInferenceAttributesVarianceExt>>(
                 &graph::BatchnormInferenceAttributesVarianceExt::set_variance),
             nb::rv_policy::reference_internal)
        .def("set_scale",
             static_cast<TensorSetter<graph::BatchnormInferenceAttributesVarianceExt>>(
                 &graph::BatchnormInferenceAttributesVarianceExt::set_scale),
             nb::rv_policy::reference_internal)
        .def("set_bias",
             static_cast<TensorSetter<graph::BatchnormInferenceAttributesVarianceExt>>(
                 &graph::BatchnormInferenceAttributesVarianceExt::set_bias),
             nb::rv_policy::reference_internal)
        .def("set_epsilon",
             static_cast<TensorSetter<graph::BatchnormInferenceAttributesVarianceExt>>(
                 &graph::BatchnormInferenceAttributesVarianceExt::set_epsilon),
             nb::rv_policy::reference_internal)
        .def("set_y",
             static_cast<TensorSetter<graph::BatchnormInferenceAttributesVarianceExt>>(
                 &graph::BatchnormInferenceAttributesVarianceExt::set_y),
             nb::rv_policy::reference_internal)
        .def("get_x", &graph::BatchnormInferenceAttributesVarianceExt::get_x)
        .def("get_mean", &graph::BatchnormInferenceAttributesVarianceExt::get_mean)
        .def("get_variance", &graph::BatchnormInferenceAttributesVarianceExt::get_variance)
        .def("get_scale", &graph::BatchnormInferenceAttributesVarianceExt::get_scale)
        .def("get_bias", &graph::BatchnormInferenceAttributesVarianceExt::get_bias)
        .def("get_epsilon", &graph::BatchnormInferenceAttributesVarianceExt::get_epsilon)
        .def("get_y", &graph::BatchnormInferenceAttributesVarianceExt::get_y);

    // BlockScaleDequantizeAttributes
    nb::class_<graph::BlockScaleDequantizeAttributes>(m, "BlockScaleDequantizeAttributes")
        .def(nb::init<>())
        .def("set_name",
             &graph::BlockScaleDequantizeAttributes::set_name,
             nb::rv_policy::reference_internal)
        .def("get_name", &graph::BlockScaleDequantizeAttributes::get_name)
        .def("set_compute_data_type",
             &graph::BlockScaleDequantizeAttributes::set_compute_data_type,
             nb::rv_policy::reference_internal)
        .def("get_compute_data_type", &graph::BlockScaleDequantizeAttributes::get_compute_data_type)
        .def("set_x",
             static_cast<TensorSetter<graph::BlockScaleDequantizeAttributes>>(
                 &graph::BlockScaleDequantizeAttributes::set_x),
             nb::rv_policy::reference_internal)
        .def("set_scale",
             static_cast<TensorSetter<graph::BlockScaleDequantizeAttributes>>(
                 &graph::BlockScaleDequantizeAttributes::set_scale),
             nb::rv_policy::reference_internal)
        .def("set_y",
             static_cast<TensorSetter<graph::BlockScaleDequantizeAttributes>>(
                 &graph::BlockScaleDequantizeAttributes::set_y),
             nb::rv_policy::reference_internal)
        .def("set_block_size",
             static_cast<Int32VectorSetter<graph::BlockScaleDequantizeAttributes>>(
                 &graph::BlockScaleDequantizeAttributes::set_block_size),
             nb::rv_policy::reference_internal)
        .def("set_block_size",
             static_cast<graph::BlockScaleDequantizeAttributes& (
                 graph::BlockScaleDequantizeAttributes::*)(int32_t, int32_t)>(
                 &graph::BlockScaleDequantizeAttributes::set_block_size),
             nb::arg("value"),
             nb::arg("idx") = 0,
             nb::rv_policy::reference_internal)
        .def("set_is_negative_scale",
             &graph::BlockScaleDequantizeAttributes::set_is_negative_scale,
             nb::rv_policy::reference_internal)
        .def("get_x", &graph::BlockScaleDequantizeAttributes::get_x)
        .def("get_scale", &graph::BlockScaleDequantizeAttributes::get_scale)
        .def("get_y", &graph::BlockScaleDequantizeAttributes::get_y)
        .def("get_block_size", &graph::BlockScaleDequantizeAttributes::get_block_size)
        .def("get_is_negative_scale",
             &graph::BlockScaleDequantizeAttributes::get_is_negative_scale);

    // BlockScaleQuantizeAttributes
    nb::class_<graph::BlockScaleQuantizeAttributes>(m, "BlockScaleQuantizeAttributes")
        .def(nb::init<>())
        .def("set_name",
             &graph::BlockScaleQuantizeAttributes::set_name,
             nb::rv_policy::reference_internal)
        .def("get_name", &graph::BlockScaleQuantizeAttributes::get_name)
        .def("set_compute_data_type",
             &graph::BlockScaleQuantizeAttributes::set_compute_data_type,
             nb::rv_policy::reference_internal)
        .def("get_compute_data_type", &graph::BlockScaleQuantizeAttributes::get_compute_data_type)
        .def("set_x",
             static_cast<TensorSetter<graph::BlockScaleQuantizeAttributes>>(
                 &graph::BlockScaleQuantizeAttributes::set_x),
             nb::rv_policy::reference_internal)
        .def("set_y",
             static_cast<TensorSetter<graph::BlockScaleQuantizeAttributes>>(
                 &graph::BlockScaleQuantizeAttributes::set_y),
             nb::rv_policy::reference_internal)
        .def("set_scale",
             static_cast<TensorSetter<graph::BlockScaleQuantizeAttributes>>(
                 &graph::BlockScaleQuantizeAttributes::set_scale),
             nb::rv_policy::reference_internal)
        .def("set_block_size",
             &graph::BlockScaleQuantizeAttributes::set_block_size,
             nb::rv_policy::reference_internal)
        .def("set_axis",
             &graph::BlockScaleQuantizeAttributes::set_axis,
             nb::rv_policy::reference_internal)
        .def("set_transpose",
             &graph::BlockScaleQuantizeAttributes::set_transpose,
             nb::rv_policy::reference_internal)
        .def("get_x", &graph::BlockScaleQuantizeAttributes::get_x)
        .def("get_y", &graph::BlockScaleQuantizeAttributes::get_y)
        .def("get_scale", &graph::BlockScaleQuantizeAttributes::get_scale)
        .def("get_block_size", &graph::BlockScaleQuantizeAttributes::get_block_size)
        .def("get_axis", &graph::BlockScaleQuantizeAttributes::get_axis)
        .def("get_transpose", &graph::BlockScaleQuantizeAttributes::get_transpose);

    // Convolution attributes
    auto convDgradClass
        = nb::class_<graph::ConvDgradAttributes>(m, "ConvolutionDgradAttributes")
              .def(nb::init<>())
              .def("set_name",
                   &graph::ConvDgradAttributes::set_name,
                   nb::rv_policy::reference_internal)
              .def("get_name", &graph::ConvDgradAttributes::get_name)
              .def("set_compute_data_type",
                   &graph::ConvDgradAttributes::set_compute_data_type,
                   nb::rv_policy::reference_internal)
              .def("get_compute_data_type", &graph::ConvDgradAttributes::get_compute_data_type)
              .def("set_dy",
                   static_cast<TensorSetter<graph::ConvDgradAttributes>>(
                       &graph::ConvDgradAttributes::set_dy),
                   nb::rv_policy::reference_internal)
              .def("set_w",
                   static_cast<TensorSetter<graph::ConvDgradAttributes>>(
                       &graph::ConvDgradAttributes::set_w),
                   nb::rv_policy::reference_internal)
              .def("set_dx",
                   static_cast<TensorSetter<graph::ConvDgradAttributes>>(
                       &graph::ConvDgradAttributes::set_dx),
                   nb::rv_policy::reference_internal)
              .def("set_padding",
                   &graph::ConvDgradAttributes::set_padding,
                   nb::rv_policy::reference_internal)
              .def("set_pre_padding",
                   static_cast<Int64VectorSetter<graph::ConvDgradAttributes>>(
                       &graph::ConvDgradAttributes::set_pre_padding),
                   nb::rv_policy::reference_internal)
              .def("set_post_padding",
                   static_cast<Int64VectorSetter<graph::ConvDgradAttributes>>(
                       &graph::ConvDgradAttributes::set_post_padding),
                   nb::rv_policy::reference_internal)
              .def("set_stride",
                   static_cast<Int64VectorSetter<graph::ConvDgradAttributes>>(
                       &graph::ConvDgradAttributes::set_stride),
                   nb::rv_policy::reference_internal)
              .def("set_dilation",
                   static_cast<Int64VectorSetter<graph::ConvDgradAttributes>>(
                       &graph::ConvDgradAttributes::set_dilation),
                   nb::rv_policy::reference_internal)
              .def("set_convolution_mode",
                   &graph::ConvDgradAttributes::set_convolution_mode,
                   nb::rv_policy::reference_internal)
              .def("get_dy", &graph::ConvDgradAttributes::get_dy)
              .def("get_w", &graph::ConvDgradAttributes::get_w)
              .def("get_dx", &graph::ConvDgradAttributes::get_dx)
              .def("get_pre_padding", &graph::ConvDgradAttributes::get_pre_padding)
              .def("get_post_padding", &graph::ConvDgradAttributes::get_post_padding)
              .def("get_stride", &graph::ConvDgradAttributes::get_stride)
              .def("get_dilation", &graph::ConvDgradAttributes::get_dilation)
              .def("get_convolution_mode", &graph::ConvDgradAttributes::get_convolution_mode);
    m.attr("ConvDgradAttributes") = convDgradClass;

    auto convFpropClass
        = nb::class_<graph::ConvFpropAttributes>(m, "ConvolutionFpropAttributes")
              .def(nb::init<>())
              .def("set_name",
                   &graph::ConvFpropAttributes::set_name,
                   nb::rv_policy::reference_internal)
              .def("get_name", &graph::ConvFpropAttributes::get_name)
              .def("set_compute_data_type",
                   &graph::ConvFpropAttributes::set_compute_data_type,
                   nb::rv_policy::reference_internal)
              .def("get_compute_data_type", &graph::ConvFpropAttributes::get_compute_data_type)
              .def("set_x",
                   static_cast<TensorSetter<graph::ConvFpropAttributes>>(
                       &graph::ConvFpropAttributes::set_x),
                   nb::rv_policy::reference_internal)
              .def("set_w",
                   static_cast<TensorSetter<graph::ConvFpropAttributes>>(
                       &graph::ConvFpropAttributes::set_w),
                   nb::rv_policy::reference_internal)
              .def("set_y",
                   static_cast<TensorSetter<graph::ConvFpropAttributes>>(
                       &graph::ConvFpropAttributes::set_y),
                   nb::rv_policy::reference_internal)
              .def("set_padding",
                   &graph::ConvFpropAttributes::set_padding,
                   nb::rv_policy::reference_internal)
              .def("set_pre_padding",
                   static_cast<Int64VectorSetter<graph::ConvFpropAttributes>>(
                       &graph::ConvFpropAttributes::set_pre_padding),
                   nb::rv_policy::reference_internal)
              .def("set_post_padding",
                   static_cast<Int64VectorSetter<graph::ConvFpropAttributes>>(
                       &graph::ConvFpropAttributes::set_post_padding),
                   nb::rv_policy::reference_internal)
              .def("set_stride",
                   static_cast<Int64VectorSetter<graph::ConvFpropAttributes>>(
                       &graph::ConvFpropAttributes::set_stride),
                   nb::rv_policy::reference_internal)
              .def("set_dilation",
                   static_cast<Int64VectorSetter<graph::ConvFpropAttributes>>(
                       &graph::ConvFpropAttributes::set_dilation),
                   nb::rv_policy::reference_internal)
              .def("set_convolution_mode",
                   &graph::ConvFpropAttributes::set_convolution_mode,
                   nb::rv_policy::reference_internal)
              .def("get_x", &graph::ConvFpropAttributes::get_x)
              .def("get_w", &graph::ConvFpropAttributes::get_w)
              .def("get_y", &graph::ConvFpropAttributes::get_y)
              .def("get_pre_padding", &graph::ConvFpropAttributes::get_pre_padding)
              .def("get_post_padding", &graph::ConvFpropAttributes::get_post_padding)
              .def("get_stride", &graph::ConvFpropAttributes::get_stride)
              .def("get_dilation", &graph::ConvFpropAttributes::get_dilation)
              .def("get_convolution_mode", &graph::ConvFpropAttributes::get_convolution_mode);
    m.attr("ConvFpropAttributes") = convFpropClass;

    auto convWgradClass
        = nb::class_<graph::ConvWgradAttributes>(m, "ConvolutionWgradAttributes")
              .def(nb::init<>())
              .def("set_name",
                   &graph::ConvWgradAttributes::set_name,
                   nb::rv_policy::reference_internal)
              .def("get_name", &graph::ConvWgradAttributes::get_name)
              .def("set_compute_data_type",
                   &graph::ConvWgradAttributes::set_compute_data_type,
                   nb::rv_policy::reference_internal)
              .def("get_compute_data_type", &graph::ConvWgradAttributes::get_compute_data_type)
              .def("set_x",
                   static_cast<TensorSetter<graph::ConvWgradAttributes>>(
                       &graph::ConvWgradAttributes::set_x),
                   nb::rv_policy::reference_internal)
              .def("set_dy",
                   static_cast<TensorSetter<graph::ConvWgradAttributes>>(
                       &graph::ConvWgradAttributes::set_dy),
                   nb::rv_policy::reference_internal)
              .def("set_dw",
                   static_cast<TensorSetter<graph::ConvWgradAttributes>>(
                       &graph::ConvWgradAttributes::set_dw),
                   nb::rv_policy::reference_internal)
              .def("set_padding",
                   &graph::ConvWgradAttributes::set_padding,
                   nb::rv_policy::reference_internal)
              .def("set_pre_padding",
                   static_cast<Int64VectorSetter<graph::ConvWgradAttributes>>(
                       &graph::ConvWgradAttributes::set_pre_padding),
                   nb::rv_policy::reference_internal)
              .def("set_post_padding",
                   static_cast<Int64VectorSetter<graph::ConvWgradAttributes>>(
                       &graph::ConvWgradAttributes::set_post_padding),
                   nb::rv_policy::reference_internal)
              .def("set_stride",
                   static_cast<Int64VectorSetter<graph::ConvWgradAttributes>>(
                       &graph::ConvWgradAttributes::set_stride),
                   nb::rv_policy::reference_internal)
              .def("set_dilation",
                   static_cast<Int64VectorSetter<graph::ConvWgradAttributes>>(
                       &graph::ConvWgradAttributes::set_dilation),
                   nb::rv_policy::reference_internal)
              .def("set_convolution_mode",
                   &graph::ConvWgradAttributes::set_convolution_mode,
                   nb::rv_policy::reference_internal)
              .def("get_x", &graph::ConvWgradAttributes::get_x)
              .def("get_dy", &graph::ConvWgradAttributes::get_dy)
              .def("get_dw", &graph::ConvWgradAttributes::get_dw)
              .def("get_pre_padding", &graph::ConvWgradAttributes::get_pre_padding)
              .def("get_post_padding", &graph::ConvWgradAttributes::get_post_padding)
              .def("get_stride", &graph::ConvWgradAttributes::get_stride)
              .def("get_dilation", &graph::ConvWgradAttributes::get_dilation)
              .def("get_convolution_mode", &graph::ConvWgradAttributes::get_convolution_mode);
    m.attr("ConvWgradAttributes") = convWgradClass;

    // CustomOpAttributes
    nb::class_<graph::CustomOpAttributes>(m, "CustomOpAttributes")
        .def(nb::init<>())
        .def("set_name", &graph::CustomOpAttributes::set_name, nb::rv_policy::reference_internal)
        .def("get_name", &graph::CustomOpAttributes::get_name)
        .def("set_compute_data_type",
             &graph::CustomOpAttributes::set_compute_data_type,
             nb::rv_policy::reference_internal)
        .def("get_compute_data_type", &graph::CustomOpAttributes::get_compute_data_type)
        .def("set_custom_op_id",
             &graph::CustomOpAttributes::set_custom_op_id,
             nb::rv_policy::reference_internal)
        .def("get_custom_op_id", &graph::CustomOpAttributes::get_custom_op_id)
        .def(
            "set_inputs", &graph::CustomOpAttributes::set_inputs, nb::rv_policy::reference_internal)
        .def("get_inputs", &graph::CustomOpAttributes::get_inputs)
        .def("set_outputs",
             &graph::CustomOpAttributes::set_outputs,
             nb::rv_policy::reference_internal)
        .def("get_outputs", &graph::CustomOpAttributes::get_outputs)
        .def("set_data", &graph::CustomOpAttributes::set_data, nb::rv_policy::reference_internal)
        .def("get_data", &graph::CustomOpAttributes::get_data);

    // LayernormAttributes
    nb::class_<graph::LayernormAttributes>(m, "LayernormAttributes")
        .def(nb::init<>())
        .def("set_name", &graph::LayernormAttributes::set_name, nb::rv_policy::reference_internal)
        .def("get_name", &graph::LayernormAttributes::get_name)
        .def("set_compute_data_type",
             &graph::LayernormAttributes::set_compute_data_type,
             nb::rv_policy::reference_internal)
        .def("get_compute_data_type", &graph::LayernormAttributes::get_compute_data_type)
        .def("set_x",
             static_cast<TensorSetter<graph::LayernormAttributes>>(
                 &graph::LayernormAttributes::set_x),
             nb::rv_policy::reference_internal)
        .def("set_scale",
             static_cast<TensorSetter<graph::LayernormAttributes>>(
                 &graph::LayernormAttributes::set_scale),
             nb::rv_policy::reference_internal)
        .def("set_bias",
             static_cast<TensorSetter<graph::LayernormAttributes>>(
                 &graph::LayernormAttributes::set_bias),
             nb::rv_policy::reference_internal)
        .def("set_epsilon",
             static_cast<TensorSetter<graph::LayernormAttributes>>(
                 &graph::LayernormAttributes::set_epsilon),
             nb::rv_policy::reference_internal)
        .def("set_y",
             static_cast<TensorSetter<graph::LayernormAttributes>>(
                 &graph::LayernormAttributes::set_y),
             nb::rv_policy::reference_internal)
        .def("set_mean",
             static_cast<TensorSetter<graph::LayernormAttributes>>(
                 &graph::LayernormAttributes::set_mean),
             nb::rv_policy::reference_internal)
        .def("set_inv_variance",
             static_cast<TensorSetter<graph::LayernormAttributes>>(
                 &graph::LayernormAttributes::set_inv_variance),
             nb::rv_policy::reference_internal)
        .def("set_forward_phase",
             &graph::LayernormAttributes::set_forward_phase,
             nb::rv_policy::reference_internal)
        .def("get_x", &graph::LayernormAttributes::get_x)
        .def("get_scale", &graph::LayernormAttributes::get_scale)
        .def("get_bias", &graph::LayernormAttributes::get_bias)
        .def("get_epsilon", &graph::LayernormAttributes::get_epsilon)
        .def("get_y", &graph::LayernormAttributes::get_y)
        .def("get_mean", &graph::LayernormAttributes::get_mean)
        .def("get_inv_variance", &graph::LayernormAttributes::get_inv_variance)
        .def("get_forward_phase", &graph::LayernormAttributes::get_forward_phase);

    // LayernormBackwardAttributes
    nb::class_<graph::LayernormBackwardAttributes>(m, "LayernormBackwardAttributes")
        .def(nb::init<>())
        .def("set_name",
             &graph::LayernormBackwardAttributes::set_name,
             nb::rv_policy::reference_internal)
        .def("get_name", &graph::LayernormBackwardAttributes::get_name)
        .def("set_compute_data_type",
             &graph::LayernormBackwardAttributes::set_compute_data_type,
             nb::rv_policy::reference_internal)
        .def("get_compute_data_type", &graph::LayernormBackwardAttributes::get_compute_data_type)
        .def("set_dy",
             static_cast<TensorSetter<graph::LayernormBackwardAttributes>>(
                 &graph::LayernormBackwardAttributes::set_dy),
             nb::rv_policy::reference_internal)
        .def("set_x",
             static_cast<TensorSetter<graph::LayernormBackwardAttributes>>(
                 &graph::LayernormBackwardAttributes::set_x),
             nb::rv_policy::reference_internal)
        .def("set_scale",
             static_cast<TensorSetter<graph::LayernormBackwardAttributes>>(
                 &graph::LayernormBackwardAttributes::set_scale),
             nb::rv_policy::reference_internal)
        .def("set_mean",
             static_cast<TensorSetter<graph::LayernormBackwardAttributes>>(
                 &graph::LayernormBackwardAttributes::set_mean),
             nb::rv_policy::reference_internal)
        .def("set_inv_variance",
             static_cast<TensorSetter<graph::LayernormBackwardAttributes>>(
                 &graph::LayernormBackwardAttributes::set_inv_variance),
             nb::rv_policy::reference_internal)
        .def("set_epsilon",
             static_cast<TensorSetter<graph::LayernormBackwardAttributes>>(
                 &graph::LayernormBackwardAttributes::set_epsilon),
             nb::rv_policy::reference_internal)
        .def("set_dx",
             static_cast<TensorSetter<graph::LayernormBackwardAttributes>>(
                 &graph::LayernormBackwardAttributes::set_dx),
             nb::rv_policy::reference_internal)
        .def("set_dscale",
             static_cast<TensorSetter<graph::LayernormBackwardAttributes>>(
                 &graph::LayernormBackwardAttributes::set_dscale),
             nb::rv_policy::reference_internal)
        .def("set_dbias",
             static_cast<TensorSetter<graph::LayernormBackwardAttributes>>(
                 &graph::LayernormBackwardAttributes::set_dbias),
             nb::rv_policy::reference_internal)
        .def("set_normalized_dim_count",
             &graph::LayernormBackwardAttributes::set_normalized_dim_count,
             nb::rv_policy::reference_internal)
        .def("set_saved_mean_and_inv_variance",
             static_cast<TwoTensorSetter<graph::LayernormBackwardAttributes>>(
                 &graph::LayernormBackwardAttributes::set_saved_mean_and_inv_variance),
             nb::rv_policy::reference_internal)
        .def("get_dy", &graph::LayernormBackwardAttributes::get_dy)
        .def("get_x", &graph::LayernormBackwardAttributes::get_x)
        .def("get_scale", &graph::LayernormBackwardAttributes::get_scale)
        .def("get_mean", &graph::LayernormBackwardAttributes::get_mean)
        .def("get_inv_variance", &graph::LayernormBackwardAttributes::get_inv_variance)
        .def("get_epsilon", &graph::LayernormBackwardAttributes::get_epsilon)
        .def("get_dx", &graph::LayernormBackwardAttributes::get_dx)
        .def("get_dscale", &graph::LayernormBackwardAttributes::get_dscale)
        .def("get_dbias", &graph::LayernormBackwardAttributes::get_dbias)
        .def("get_normalized_dim_count",
             &graph::LayernormBackwardAttributes::get_normalized_dim_count);

    // MatmulAttributes
    nb::class_<graph::MatmulAttributes>(m, "MatmulAttributes")
        .def(nb::init<>())
        .def("set_name", &graph::MatmulAttributes::set_name, nb::rv_policy::reference_internal)
        .def("get_name", &graph::MatmulAttributes::get_name)
        .def("set_compute_data_type",
             &graph::MatmulAttributes::set_compute_data_type,
             nb::rv_policy::reference_internal)
        .def("get_compute_data_type", &graph::MatmulAttributes::get_compute_data_type)
        .def("set_a",
             static_cast<TensorSetter<graph::MatmulAttributes>>(&graph::MatmulAttributes::set_a),
             nb::rv_policy::reference_internal)
        .def("set_b",
             static_cast<TensorSetter<graph::MatmulAttributes>>(&graph::MatmulAttributes::set_b),
             nb::rv_policy::reference_internal)
        .def("set_c",
             static_cast<TensorSetter<graph::MatmulAttributes>>(&graph::MatmulAttributes::set_c),
             nb::rv_policy::reference_internal)
        .def("get_a", &graph::MatmulAttributes::get_a)
        .def("get_b", &graph::MatmulAttributes::get_b)
        .def("get_c", &graph::MatmulAttributes::get_c);

    // MoeGroupedMatmulAttributes
    nb::class_<graph::MoeGroupedMatmulAttributes>(m, "MoeGroupedMatmulAttributes")
        .def(nb::init<>())
        .def("set_name",
             &graph::MoeGroupedMatmulAttributes::set_name,
             nb::rv_policy::reference_internal)
        .def("get_name", &graph::MoeGroupedMatmulAttributes::get_name)
        .def("set_compute_data_type",
             &graph::MoeGroupedMatmulAttributes::set_compute_data_type,
             nb::rv_policy::reference_internal)
        .def("get_compute_data_type", &graph::MoeGroupedMatmulAttributes::get_compute_data_type)
        .def("set_token",
             static_cast<TensorSetter<graph::MoeGroupedMatmulAttributes>>(
                 &graph::MoeGroupedMatmulAttributes::set_token),
             nb::rv_policy::reference_internal)
        .def("set_weight",
             static_cast<TensorSetter<graph::MoeGroupedMatmulAttributes>>(
                 &graph::MoeGroupedMatmulAttributes::set_weight),
             nb::rv_policy::reference_internal)
        .def("set_first_token_offset",
             static_cast<TensorSetter<graph::MoeGroupedMatmulAttributes>>(
                 &graph::MoeGroupedMatmulAttributes::set_first_token_offset),
             nb::rv_policy::reference_internal)
        .def("set_token_index",
             static_cast<TensorSetter<graph::MoeGroupedMatmulAttributes>>(
                 &graph::MoeGroupedMatmulAttributes::set_token_index),
             nb::rv_policy::reference_internal)
        .def("set_token_ks",
             static_cast<TensorSetter<graph::MoeGroupedMatmulAttributes>>(
                 &graph::MoeGroupedMatmulAttributes::set_token_ks),
             nb::rv_policy::reference_internal)
        .def("set_output",
             static_cast<TensorSetter<graph::MoeGroupedMatmulAttributes>>(
                 &graph::MoeGroupedMatmulAttributes::set_output),
             nb::rv_policy::reference_internal)
        .def("set_mode",
             &graph::MoeGroupedMatmulAttributes::set_mode,
             nb::rv_policy::reference_internal)
        .def("set_top_k",
             &graph::MoeGroupedMatmulAttributes::set_top_k,
             nb::rv_policy::reference_internal)
        .def("get_token", &graph::MoeGroupedMatmulAttributes::get_token)
        .def("get_weight", &graph::MoeGroupedMatmulAttributes::get_weight)
        .def("get_first_token_offset", &graph::MoeGroupedMatmulAttributes::get_first_token_offset)
        .def("get_token_index", &graph::MoeGroupedMatmulAttributes::get_token_index)
        .def("get_token_ks", &graph::MoeGroupedMatmulAttributes::get_token_ks)
        .def("get_output", &graph::MoeGroupedMatmulAttributes::get_output)
        .def("get_mode", &graph::MoeGroupedMatmulAttributes::get_mode)
        .def("get_top_k", &graph::MoeGroupedMatmulAttributes::get_top_k);

    // PointwiseAttributes
    nb::class_<graph::PointwiseAttributes>(m, "PointwiseAttributes")
        .def(nb::init<>())
        .def("set_name", &graph::PointwiseAttributes::set_name, nb::rv_policy::reference_internal)
        .def("get_name", &graph::PointwiseAttributes::get_name)
        .def("set_compute_data_type",
             &graph::PointwiseAttributes::set_compute_data_type,
             nb::rv_policy::reference_internal)
        .def("get_compute_data_type", &graph::PointwiseAttributes::get_compute_data_type)
        .def("set_mode", &graph::PointwiseAttributes::set_mode, nb::rv_policy::reference_internal)
        .def("set_relu_lower_clip",
             &graph::PointwiseAttributes::set_relu_lower_clip,
             nb::rv_policy::reference_internal)
        .def("set_relu_upper_clip",
             &graph::PointwiseAttributes::set_relu_upper_clip,
             nb::rv_policy::reference_internal)
        .def("set_relu_lower_clip_slope",
             &graph::PointwiseAttributes::set_relu_lower_clip_slope,
             nb::rv_policy::reference_internal)
        .def("set_swish_beta",
             &graph::PointwiseAttributes::set_swish_beta,
             nb::rv_policy::reference_internal)
        .def("set_elu_alpha",
             &graph::PointwiseAttributes::set_elu_alpha,
             nb::rv_policy::reference_internal)
        .def("set_softplus_beta",
             &graph::PointwiseAttributes::set_softplus_beta,
             nb::rv_policy::reference_internal)
        .def("set_axis", &graph::PointwiseAttributes::set_axis, nb::rv_policy::reference_internal)
        .def("set_input_0",
             static_cast<TensorSetter<graph::PointwiseAttributes>>(
                 &graph::PointwiseAttributes::set_input_0),
             nb::rv_policy::reference_internal)
        .def("set_input_1",
             static_cast<TensorSetter<graph::PointwiseAttributes>>(
                 &graph::PointwiseAttributes::set_input_1),
             nb::rv_policy::reference_internal)
        .def("set_input_2",
             static_cast<TensorSetter<graph::PointwiseAttributes>>(
                 &graph::PointwiseAttributes::set_input_2),
             nb::rv_policy::reference_internal)
        .def("set_output_0",
             static_cast<TensorSetter<graph::PointwiseAttributes>>(
                 &graph::PointwiseAttributes::set_output_0),
             nb::rv_policy::reference_internal)
        .def("get_mode", &graph::PointwiseAttributes::get_mode)
        .def("get_relu_lower_clip", &graph::PointwiseAttributes::get_relu_lower_clip)
        .def("get_relu_upper_clip", &graph::PointwiseAttributes::get_relu_upper_clip)
        .def("get_relu_lower_clip_slope", &graph::PointwiseAttributes::get_relu_lower_clip_slope)
        .def("get_swish_beta", &graph::PointwiseAttributes::get_swish_beta)
        .def("get_elu_alpha", &graph::PointwiseAttributes::get_elu_alpha)
        .def("get_softplus_beta", &graph::PointwiseAttributes::get_softplus_beta)
        .def("get_axis", &graph::PointwiseAttributes::get_axis)
        .def("get_input_0", &graph::PointwiseAttributes::get_input_0)
        .def("get_input_1", &graph::PointwiseAttributes::get_input_1)
        .def("get_input_2", &graph::PointwiseAttributes::get_input_2)
        .def("get_output_0", &graph::PointwiseAttributes::get_output_0);

    // ReductionAttributes
    nb::class_<graph::ReductionAttributes>(m, "ReductionAttributes")
        .def(nb::init<>())
        .def("set_name", &graph::ReductionAttributes::set_name, nb::rv_policy::reference_internal)
        .def("get_name", &graph::ReductionAttributes::get_name)
        .def("set_compute_data_type",
             &graph::ReductionAttributes::set_compute_data_type,
             nb::rv_policy::reference_internal)
        .def("get_compute_data_type", &graph::ReductionAttributes::get_compute_data_type)
        .def("set_mode", &graph::ReductionAttributes::set_mode, nb::rv_policy::reference_internal)
        .def("get_mode", &graph::ReductionAttributes::get_mode)
        .def("set_is_deterministic",
             &graph::ReductionAttributes::set_is_deterministic,
             nb::rv_policy::reference_internal)
        .def("get_is_deterministic", &graph::ReductionAttributes::get_is_deterministic)
        .def("set_x",
             static_cast<TensorSetter<graph::ReductionAttributes>>(
                 &graph::ReductionAttributes::set_x),
             nb::rv_policy::reference_internal)
        .def("set_y",
             static_cast<TensorSetter<graph::ReductionAttributes>>(
                 &graph::ReductionAttributes::set_y),
             nb::rv_policy::reference_internal)
        .def("get_x", &graph::ReductionAttributes::get_x)
        .def("get_y", &graph::ReductionAttributes::get_y);

    // ResampleFwdAttributes
    nb::class_<graph::ResampleFwdAttributes>(m, "ResampleFwdAttributes")
        .def(nb::init<>())
        .def("set_name", &graph::ResampleFwdAttributes::set_name, nb::rv_policy::reference_internal)
        .def("get_name", &graph::ResampleFwdAttributes::get_name)
        .def("set_compute_data_type",
             &graph::ResampleFwdAttributes::set_compute_data_type,
             nb::rv_policy::reference_internal)
        .def("get_compute_data_type", &graph::ResampleFwdAttributes::get_compute_data_type)
        .def("set_x",
             static_cast<TensorSetter<graph::ResampleFwdAttributes>>(
                 &graph::ResampleFwdAttributes::set_x),
             nb::rv_policy::reference_internal)
        .def("set_y",
             static_cast<TensorSetter<graph::ResampleFwdAttributes>>(
                 &graph::ResampleFwdAttributes::set_y),
             nb::rv_policy::reference_internal)
        .def("set_index",
             static_cast<TensorSetter<graph::ResampleFwdAttributes>>(
                 &graph::ResampleFwdAttributes::set_index),
             nb::rv_policy::reference_internal)
        .def("set_pre_padding",
             &graph::ResampleFwdAttributes::set_pre_padding,
             nb::rv_policy::reference_internal)
        .def("set_post_padding",
             &graph::ResampleFwdAttributes::set_post_padding,
             nb::rv_policy::reference_internal)
        .def("set_stride",
             &graph::ResampleFwdAttributes::set_stride,
             nb::rv_policy::reference_internal)
        .def("set_window",
             &graph::ResampleFwdAttributes::set_window,
             nb::rv_policy::reference_internal)
        .def("set_resample_mode",
             &graph::ResampleFwdAttributes::set_resample_mode,
             nb::rv_policy::reference_internal)
        .def("set_padding_mode",
             &graph::ResampleFwdAttributes::set_padding_mode,
             nb::rv_policy::reference_internal)
        .def("set_generate_index",
             &graph::ResampleFwdAttributes::set_generate_index,
             nb::rv_policy::reference_internal)
        .def("get_x", &graph::ResampleFwdAttributes::get_x)
        .def("get_y", &graph::ResampleFwdAttributes::get_y)
        .def("get_index", &graph::ResampleFwdAttributes::get_index)
        .def("get_pre_padding", &graph::ResampleFwdAttributes::get_pre_padding)
        .def("get_post_padding", &graph::ResampleFwdAttributes::get_post_padding)
        .def("get_stride", &graph::ResampleFwdAttributes::get_stride)
        .def("get_window", &graph::ResampleFwdAttributes::get_window)
        .def("get_resample_mode", &graph::ResampleFwdAttributes::get_resample_mode)
        .def("get_padding_mode", &graph::ResampleFwdAttributes::get_padding_mode)
        .def("get_generate_index", &graph::ResampleFwdAttributes::get_generate_index);

    // ResampleBwdAttributes
    nb::class_<graph::ResampleBwdAttributes>(m, "ResampleBwdAttributes")
        .def(nb::init<>())
        .def("set_name", &graph::ResampleBwdAttributes::set_name, nb::rv_policy::reference_internal)
        .def("get_name", &graph::ResampleBwdAttributes::get_name)
        .def("set_compute_data_type",
             &graph::ResampleBwdAttributes::set_compute_data_type,
             nb::rv_policy::reference_internal)
        .def("get_compute_data_type", &graph::ResampleBwdAttributes::get_compute_data_type)
        .def("set_dy",
             static_cast<TensorSetter<graph::ResampleBwdAttributes>>(
                 &graph::ResampleBwdAttributes::set_dy),
             nb::rv_policy::reference_internal)
        .def("set_index",
             static_cast<TensorSetter<graph::ResampleBwdAttributes>>(
                 &graph::ResampleBwdAttributes::set_index),
             nb::rv_policy::reference_internal)
        .def("set_dx",
             static_cast<TensorSetter<graph::ResampleBwdAttributes>>(
                 &graph::ResampleBwdAttributes::set_dx),
             nb::rv_policy::reference_internal)
        .def("set_pre_padding",
             static_cast<Int64VectorSetter<graph::ResampleBwdAttributes>>(
                 &graph::ResampleBwdAttributes::set_pre_padding),
             nb::rv_policy::reference_internal)
        .def("set_post_padding",
             static_cast<Int64VectorSetter<graph::ResampleBwdAttributes>>(
                 &graph::ResampleBwdAttributes::set_post_padding),
             nb::rv_policy::reference_internal)
        .def("set_stride",
             static_cast<Int64VectorSetter<graph::ResampleBwdAttributes>>(
                 &graph::ResampleBwdAttributes::set_stride),
             nb::rv_policy::reference_internal)
        .def("set_window",
             static_cast<Int64VectorSetter<graph::ResampleBwdAttributes>>(
                 &graph::ResampleBwdAttributes::set_window),
             nb::rv_policy::reference_internal)
        .def("set_resample_mode",
             &graph::ResampleBwdAttributes::set_resample_mode,
             nb::rv_policy::reference_internal)
        .def("set_padding_mode",
             &graph::ResampleBwdAttributes::set_padding_mode,
             nb::rv_policy::reference_internal)
        .def("get_dy", &graph::ResampleBwdAttributes::get_dy)
        .def("get_index", &graph::ResampleBwdAttributes::get_index)
        .def("get_dx", &graph::ResampleBwdAttributes::get_dx)
        .def("get_pre_padding", &graph::ResampleBwdAttributes::get_pre_padding)
        .def("get_post_padding", &graph::ResampleBwdAttributes::get_post_padding)
        .def("get_stride", &graph::ResampleBwdAttributes::get_stride)
        .def("get_window", &graph::ResampleBwdAttributes::get_window)
        .def("get_resample_mode", &graph::ResampleBwdAttributes::get_resample_mode)
        .def("get_padding_mode", &graph::ResampleBwdAttributes::get_padding_mode);

    // RMSNormAttributes
    nb::class_<graph::RMSNormAttributes>(m, "RMSNormAttributes")
        .def(nb::init<>())
        .def("set_name", &graph::RMSNormAttributes::set_name, nb::rv_policy::reference_internal)
        .def("get_name", &graph::RMSNormAttributes::get_name)
        .def("set_compute_data_type",
             &graph::RMSNormAttributes::set_compute_data_type,
             nb::rv_policy::reference_internal)
        .def("get_compute_data_type", &graph::RMSNormAttributes::get_compute_data_type)
        .def("set_x",
             static_cast<TensorSetter<graph::RMSNormAttributes>>(&graph::RMSNormAttributes::set_x),
             nb::rv_policy::reference_internal)
        .def("set_scale",
             static_cast<TensorSetter<graph::RMSNormAttributes>>(
                 &graph::RMSNormAttributes::set_scale),
             nb::rv_policy::reference_internal)
        .def("set_epsilon",
             static_cast<TensorSetter<graph::RMSNormAttributes>>(
                 &graph::RMSNormAttributes::set_epsilon),
             nb::rv_policy::reference_internal)
        .def("set_bias",
             static_cast<TensorSetter<graph::RMSNormAttributes>>(
                 &graph::RMSNormAttributes::set_bias),
             nb::rv_policy::reference_internal)
        .def("set_y",
             static_cast<TensorSetter<graph::RMSNormAttributes>>(&graph::RMSNormAttributes::set_y),
             nb::rv_policy::reference_internal)
        .def("set_inv_rms",
             static_cast<TensorSetter<graph::RMSNormAttributes>>(
                 &graph::RMSNormAttributes::set_inv_rms),
             nb::rv_policy::reference_internal)
        .def("set_forward_phase",
             &graph::RMSNormAttributes::set_forward_phase,
             nb::rv_policy::reference_internal)
        .def("get_x", &graph::RMSNormAttributes::get_x)
        .def("get_scale", &graph::RMSNormAttributes::get_scale)
        .def("get_epsilon", &graph::RMSNormAttributes::get_epsilon)
        .def("get_bias", &graph::RMSNormAttributes::get_bias)
        .def("get_y", &graph::RMSNormAttributes::get_y)
        .def("get_inv_rms", &graph::RMSNormAttributes::get_inv_rms)
        .def("get_forward_phase", &graph::RMSNormAttributes::get_forward_phase);

    // RMSNormBackwardAttributes
    nb::class_<graph::RMSNormBackwardAttributes>(m, "RMSNormBackwardAttributes")
        .def(nb::init<>())
        .def("set_name",
             &graph::RMSNormBackwardAttributes::set_name,
             nb::rv_policy::reference_internal)
        .def("get_name", &graph::RMSNormBackwardAttributes::get_name)
        .def("set_compute_data_type",
             &graph::RMSNormBackwardAttributes::set_compute_data_type,
             nb::rv_policy::reference_internal)
        .def("get_compute_data_type", &graph::RMSNormBackwardAttributes::get_compute_data_type)
        .def("set_dy",
             static_cast<TensorSetter<graph::RMSNormBackwardAttributes>>(
                 &graph::RMSNormBackwardAttributes::set_dy),
             nb::rv_policy::reference_internal)
        .def("set_x",
             static_cast<TensorSetter<graph::RMSNormBackwardAttributes>>(
                 &graph::RMSNormBackwardAttributes::set_x),
             nb::rv_policy::reference_internal)
        .def("set_scale",
             static_cast<TensorSetter<graph::RMSNormBackwardAttributes>>(
                 &graph::RMSNormBackwardAttributes::set_scale),
             nb::rv_policy::reference_internal)
        .def("set_inv_rms",
             static_cast<TensorSetter<graph::RMSNormBackwardAttributes>>(
                 &graph::RMSNormBackwardAttributes::set_inv_rms),
             nb::rv_policy::reference_internal)
        .def("set_dx",
             static_cast<TensorSetter<graph::RMSNormBackwardAttributes>>(
                 &graph::RMSNormBackwardAttributes::set_dx),
             nb::rv_policy::reference_internal)
        .def("set_dscale",
             static_cast<TensorSetter<graph::RMSNormBackwardAttributes>>(
                 &graph::RMSNormBackwardAttributes::set_dscale),
             nb::rv_policy::reference_internal)
        .def("set_dbias",
             static_cast<TensorSetter<graph::RMSNormBackwardAttributes>>(
                 &graph::RMSNormBackwardAttributes::set_dbias),
             nb::rv_policy::reference_internal)
        .def("set_compute_dbias",
             &graph::RMSNormBackwardAttributes::set_compute_dbias,
             nb::rv_policy::reference_internal)
        .def("has_dbias",
             &graph::RMSNormBackwardAttributes::has_dbias,
             nb::rv_policy::reference_internal)
        .def("get_dy", &graph::RMSNormBackwardAttributes::get_dy)
        .def("get_x", &graph::RMSNormBackwardAttributes::get_x)
        .def("get_scale", &graph::RMSNormBackwardAttributes::get_scale)
        .def("get_inv_rms", &graph::RMSNormBackwardAttributes::get_inv_rms)
        .def("get_dx", &graph::RMSNormBackwardAttributes::get_dx)
        .def("get_dscale", &graph::RMSNormBackwardAttributes::get_dscale)
        .def("get_dbias", &graph::RMSNormBackwardAttributes::get_dbias)
        .def("get_compute_dbias", &graph::RMSNormBackwardAttributes::get_compute_dbias);

    // SdpaAttributes. SDPA node methods are feature-gated; these attributes are not.
    nb::class_<graph::SdpaAttributes>(m, "SdpaAttributes")
        .def(nb::init<>())
        .def("set_name", &graph::SdpaAttributes::set_name, nb::rv_policy::reference_internal)
        .def("get_name", &graph::SdpaAttributes::get_name)
        .def("set_compute_data_type",
             &graph::SdpaAttributes::set_compute_data_type,
             nb::rv_policy::reference_internal)
        .def("get_compute_data_type", &graph::SdpaAttributes::get_compute_data_type)
        .def("set_q",
             static_cast<TensorSetter<graph::SdpaAttributes>>(&graph::SdpaAttributes::set_q),
             nb::rv_policy::reference_internal)
        .def("set_k",
             static_cast<TensorSetter<graph::SdpaAttributes>>(&graph::SdpaAttributes::set_k),
             nb::rv_policy::reference_internal)
        .def("set_v",
             static_cast<TensorSetter<graph::SdpaAttributes>>(&graph::SdpaAttributes::set_v),
             nb::rv_policy::reference_internal)
        .def("set_bias",
             static_cast<TensorSetter<graph::SdpaAttributes>>(&graph::SdpaAttributes::set_bias),
             nb::rv_policy::reference_internal)
        .def("set_attn_scale",
             static_cast<TensorSetter<graph::SdpaAttributes>>(
                 &graph::SdpaAttributes::set_attn_scale),
             nb::rv_policy::reference_internal)
        .def(
            "set_seq_len_q",
            static_cast<TensorSetter<graph::SdpaAttributes>>(&graph::SdpaAttributes::set_seq_len_q),
            nb::rv_policy::reference_internal)
        .def("set_seq_len_kv",
             static_cast<TensorSetter<graph::SdpaAttributes>>(
                 &graph::SdpaAttributes::set_seq_len_kv),
             nb::rv_policy::reference_internal)
        .def("set_seed",
             static_cast<TensorSetter<graph::SdpaAttributes>>(&graph::SdpaAttributes::set_seed),
             nb::rv_policy::reference_internal)
        .def("set_offset",
             static_cast<TensorSetter<graph::SdpaAttributes>>(&graph::SdpaAttributes::set_offset),
             nb::rv_policy::reference_internal)
        .def("set_dropout_mask",
             static_cast<TensorSetter<graph::SdpaAttributes>>(
                 &graph::SdpaAttributes::set_dropout_mask),
             nb::rv_policy::reference_internal)
        .def("set_dropout_scale",
             static_cast<TensorSetter<graph::SdpaAttributes>>(
                 &graph::SdpaAttributes::set_dropout_scale),
             nb::rv_policy::reference_internal)
        .def("set_paged_attention_k_table",
             static_cast<TensorSetter<graph::SdpaAttributes>>(
                 &graph::SdpaAttributes::set_paged_attention_k_table),
             nb::rv_policy::reference_internal)
        .def("set_paged_attention_v_table",
             static_cast<TensorSetter<graph::SdpaAttributes>>(
                 &graph::SdpaAttributes::set_paged_attention_v_table),
             nb::rv_policy::reference_internal)
        .def("set_block_mask",
             static_cast<TensorSetter<graph::SdpaAttributes>>(
                 &graph::SdpaAttributes::set_block_mask),
             nb::rv_policy::reference_internal)
        .def("set_sink_token",
             static_cast<TensorSetter<graph::SdpaAttributes>>(
                 &graph::SdpaAttributes::set_sink_token),
             nb::rv_policy::reference_internal)
        .def(
            "set_descale_q",
            static_cast<TensorSetter<graph::SdpaAttributes>>(&graph::SdpaAttributes::set_descale_q),
            nb::rv_policy::reference_internal)
        .def(
            "set_descale_k",
            static_cast<TensorSetter<graph::SdpaAttributes>>(&graph::SdpaAttributes::set_descale_k),
            nb::rv_policy::reference_internal)
        .def(
            "set_descale_v",
            static_cast<TensorSetter<graph::SdpaAttributes>>(&graph::SdpaAttributes::set_descale_v),
            nb::rv_policy::reference_internal)
        .def(
            "set_descale_s",
            static_cast<TensorSetter<graph::SdpaAttributes>>(&graph::SdpaAttributes::set_descale_s),
            nb::rv_policy::reference_internal)
        .def("set_scale_s",
             static_cast<TensorSetter<graph::SdpaAttributes>>(&graph::SdpaAttributes::set_scale_s),
             nb::rv_policy::reference_internal)
        .def("set_scale_o",
             static_cast<TensorSetter<graph::SdpaAttributes>>(&graph::SdpaAttributes::set_scale_o),
             nb::rv_policy::reference_internal)
        .def("set_o",
             static_cast<TensorSetter<graph::SdpaAttributes>>(&graph::SdpaAttributes::set_o),
             nb::rv_policy::reference_internal)
        .def("set_stats",
             static_cast<TensorSetter<graph::SdpaAttributes>>(&graph::SdpaAttributes::set_stats),
             nb::rv_policy::reference_internal)
        .def(
            "set_logit_max",
            static_cast<TensorSetter<graph::SdpaAttributes>>(&graph::SdpaAttributes::set_logit_max),
            nb::rv_policy::reference_internal)
        .def("set_score_sum_exp",
             static_cast<TensorSetter<graph::SdpaAttributes>>(
                 &graph::SdpaAttributes::set_score_sum_exp),
             nb::rv_policy::reference_internal)
        .def("set_rng_dump",
             static_cast<TensorSetter<graph::SdpaAttributes>>(&graph::SdpaAttributes::set_rng_dump),
             nb::rv_policy::reference_internal)
        .def("set_amax_s",
             static_cast<TensorSetter<graph::SdpaAttributes>>(&graph::SdpaAttributes::set_amax_s),
             nb::rv_policy::reference_internal)
        .def("set_amax_o",
             static_cast<TensorSetter<graph::SdpaAttributes>>(&graph::SdpaAttributes::set_amax_o),
             nb::rv_policy::reference_internal)
        .def("set_generate_stats",
             &graph::SdpaAttributes::set_generate_stats,
             nb::rv_policy::reference_internal)
        .def("set_alibi_mask",
             &graph::SdpaAttributes::set_alibi_mask,
             nb::rv_policy::reference_internal)
        .def("set_padding_mask",
             &graph::SdpaAttributes::set_padding_mask,
             nb::rv_policy::reference_internal)
        .def("set_causal_mask",
             &graph::SdpaAttributes::set_causal_mask,
             nb::rv_policy::reference_internal)
        .def("set_causal_mask_bottom_right",
             &graph::SdpaAttributes::set_causal_mask_bottom_right,
             nb::rv_policy::reference_internal)
        .def("set_dropout",
             static_cast<graph::SdpaAttributes& (
                 graph::SdpaAttributes::*)(float, const Tensor&, const Tensor&)>(
                 &graph::SdpaAttributes::set_dropout),
             nb::rv_policy::reference_internal)
        .def("set_dropout_probability",
             &graph::SdpaAttributes::set_dropout_probability,
             nb::rv_policy::reference_internal)
        .def("set_attn_scale",
             static_cast<graph::SdpaAttributes& (graph::SdpaAttributes::*)(float)>(
                 &graph::SdpaAttributes::set_attn_scale),
             nb::rv_policy::reference_internal)
        .def("set_diagonal_band_left_bound",
             &graph::SdpaAttributes::set_diagonal_band_left_bound,
             nb::rv_policy::reference_internal)
        .def("set_diagonal_band_right_bound",
             &graph::SdpaAttributes::set_diagonal_band_right_bound,
             nb::rv_policy::reference_internal)
        .def("set_paged_attention_max_seq_len_kv",
             &graph::SdpaAttributes::set_paged_attention_max_seq_len_kv,
             nb::rv_policy::reference_internal)
        .def("set_diagonal_alignment",
             &graph::SdpaAttributes::set_diagonal_alignment,
             nb::rv_policy::reference_internal)
        .def("set_mma_core_mode",
             &graph::SdpaAttributes::set_mma_core_mode,
             nb::rv_policy::reference_internal)
        .def("set_implementation",
             &graph::SdpaAttributes::set_implementation,
             nb::rv_policy::reference_internal)
        .def("set_unfuse_fma",
             &graph::SdpaAttributes::set_unfuse_fma,
             nb::rv_policy::reference_internal)
        .def("get_q", &graph::SdpaAttributes::get_q)
        .def("get_k", &graph::SdpaAttributes::get_k)
        .def("get_v", &graph::SdpaAttributes::get_v)
        .def("get_bias", &graph::SdpaAttributes::get_bias)
        .def("get_attn_scale", &graph::SdpaAttributes::get_attn_scale)
        .def("get_seq_len_q", &graph::SdpaAttributes::get_seq_len_q)
        .def("get_seq_len_kv", &graph::SdpaAttributes::get_seq_len_kv)
        .def("get_seed", &graph::SdpaAttributes::get_seed)
        .def("get_offset", &graph::SdpaAttributes::get_offset)
        .def("get_dropout_mask", &graph::SdpaAttributes::get_dropout_mask)
        .def("get_dropout_scale", &graph::SdpaAttributes::get_dropout_scale)
        .def("get_page_table_k", &graph::SdpaAttributes::get_page_table_k)
        .def("get_page_table_v", &graph::SdpaAttributes::get_page_table_v)
        .def("get_block_mask", &graph::SdpaAttributes::get_block_mask)
        .def("get_sink_token", &graph::SdpaAttributes::get_sink_token)
        .def("get_descale_q", &graph::SdpaAttributes::get_descale_q)
        .def("get_descale_k", &graph::SdpaAttributes::get_descale_k)
        .def("get_descale_v", &graph::SdpaAttributes::get_descale_v)
        .def("get_descale_s", &graph::SdpaAttributes::get_descale_s)
        .def("get_scale_s", &graph::SdpaAttributes::get_scale_s)
        .def("get_scale_o", &graph::SdpaAttributes::get_scale_o)
        .def("get_o", &graph::SdpaAttributes::get_o)
        .def("get_stats", &graph::SdpaAttributes::get_stats)
        .def("get_max", &graph::SdpaAttributes::get_max)
        .def("get_sum_exp", &graph::SdpaAttributes::get_sum_exp)
        .def("get_rng_dump", &graph::SdpaAttributes::get_rng_dump)
        .def("get_amax_s", &graph::SdpaAttributes::get_amax_s)
        .def("get_amax_o", &graph::SdpaAttributes::get_amax_o)
        .def_rw("generate_stats", &graph::SdpaAttributes::generate_stats)
        .def_rw("alibi_mask", &graph::SdpaAttributes::alibi_mask)
        .def_rw("padding_mask", &graph::SdpaAttributes::padding_mask)
        .def_rw("causal_mask", &graph::SdpaAttributes::causal_mask)
        .def_rw("causal_mask_bottom_right", &graph::SdpaAttributes::causal_mask_bottom_right)
        .def_rw("dropout_probability", &graph::SdpaAttributes::dropout_probability)
        .def_rw("attn_scale_value", &graph::SdpaAttributes::attn_scale_value)
        .def_rw("left_bound", &graph::SdpaAttributes::left_bound)
        .def_rw("right_bound", &graph::SdpaAttributes::right_bound)
        .def_rw("max_seq_len_kv", &graph::SdpaAttributes::max_seq_len_kv)
        .def_rw("diagonal_alignment", &graph::SdpaAttributes::diagonal_alignment)
        .def_rw("mma_core_mode", &graph::SdpaAttributes::mma_core_mode)
        .def_rw("implementation", &graph::SdpaAttributes::implementation)
        .def_rw("unfuse_fma_hint", &graph::SdpaAttributes::unfuse_fma_hint);

    // SdpaBackwardAttributes. SDPA node methods are feature-gated; these attributes are not.
    nb::class_<graph::SdpaBackwardAttributes>(m, "SdpaBackwardAttributes")
        .def(nb::init<>())
        .def(
            "set_name", &graph::SdpaBackwardAttributes::set_name, nb::rv_policy::reference_internal)
        .def("get_name", &graph::SdpaBackwardAttributes::get_name)
        .def("set_compute_data_type",
             &graph::SdpaBackwardAttributes::set_compute_data_type,
             nb::rv_policy::reference_internal)
        .def("get_compute_data_type", &graph::SdpaBackwardAttributes::get_compute_data_type)
        .def("set_q",
             static_cast<TensorSetter<graph::SdpaBackwardAttributes>>(
                 &graph::SdpaBackwardAttributes::set_q),
             nb::rv_policy::reference_internal)
        .def("set_k",
             static_cast<TensorSetter<graph::SdpaBackwardAttributes>>(
                 &graph::SdpaBackwardAttributes::set_k),
             nb::rv_policy::reference_internal)
        .def("set_v",
             static_cast<TensorSetter<graph::SdpaBackwardAttributes>>(
                 &graph::SdpaBackwardAttributes::set_v),
             nb::rv_policy::reference_internal)
        .def("set_o",
             static_cast<TensorSetter<graph::SdpaBackwardAttributes>>(
                 &graph::SdpaBackwardAttributes::set_o),
             nb::rv_policy::reference_internal)
        .def("set_do",
             static_cast<TensorSetter<graph::SdpaBackwardAttributes>>(
                 &graph::SdpaBackwardAttributes::set_do),
             nb::rv_policy::reference_internal)
        .def("set_stats",
             static_cast<TensorSetter<graph::SdpaBackwardAttributes>>(
                 &graph::SdpaBackwardAttributes::set_stats),
             nb::rv_policy::reference_internal)
        .def("set_attn_scale",
             static_cast<TensorSetter<graph::SdpaBackwardAttributes>>(
                 &graph::SdpaBackwardAttributes::set_attn_scale),
             nb::rv_policy::reference_internal)
        .def("set_bias",
             static_cast<TensorSetter<graph::SdpaBackwardAttributes>>(
                 &graph::SdpaBackwardAttributes::set_bias),
             nb::rv_policy::reference_internal)
        .def("set_seq_len_q",
             static_cast<TensorSetter<graph::SdpaBackwardAttributes>>(
                 &graph::SdpaBackwardAttributes::set_seq_len_q),
             nb::rv_policy::reference_internal)
        .def("set_seq_len_kv",
             static_cast<TensorSetter<graph::SdpaBackwardAttributes>>(
                 &graph::SdpaBackwardAttributes::set_seq_len_kv),
             nb::rv_policy::reference_internal)
        .def("set_seed",
             static_cast<TensorSetter<graph::SdpaBackwardAttributes>>(
                 &graph::SdpaBackwardAttributes::set_seed),
             nb::rv_policy::reference_internal)
        .def("set_offset",
             static_cast<TensorSetter<graph::SdpaBackwardAttributes>>(
                 &graph::SdpaBackwardAttributes::set_offset),
             nb::rv_policy::reference_internal)
        .def("set_dropout_mask",
             static_cast<TensorSetter<graph::SdpaBackwardAttributes>>(
                 &graph::SdpaBackwardAttributes::set_dropout_mask),
             nb::rv_policy::reference_internal)
        .def("set_dropout_scale",
             static_cast<TensorSetter<graph::SdpaBackwardAttributes>>(
                 &graph::SdpaBackwardAttributes::set_dropout_scale),
             nb::rv_policy::reference_internal)
        .def("set_dropout_scale_inv",
             static_cast<TensorSetter<graph::SdpaBackwardAttributes>>(
                 &graph::SdpaBackwardAttributes::set_dropout_scale_inv),
             nb::rv_policy::reference_internal)
        .def("set_dq",
             static_cast<TensorSetter<graph::SdpaBackwardAttributes>>(
                 &graph::SdpaBackwardAttributes::set_dq),
             nb::rv_policy::reference_internal)
        .def("set_dk",
             static_cast<TensorSetter<graph::SdpaBackwardAttributes>>(
                 &graph::SdpaBackwardAttributes::set_dk),
             nb::rv_policy::reference_internal)
        .def("set_dv",
             static_cast<TensorSetter<graph::SdpaBackwardAttributes>>(
                 &graph::SdpaBackwardAttributes::set_dv),
             nb::rv_policy::reference_internal)
        .def("set_dbias",
             static_cast<TensorSetter<graph::SdpaBackwardAttributes>>(
                 &graph::SdpaBackwardAttributes::set_dbias),
             nb::rv_policy::reference_internal)
        .def("set_alibi_mask",
             &graph::SdpaBackwardAttributes::set_alibi_mask,
             nb::rv_policy::reference_internal)
        .def("set_padding_mask",
             &graph::SdpaBackwardAttributes::set_padding_mask,
             nb::rv_policy::reference_internal)
        .def("set_causal_mask",
             &graph::SdpaBackwardAttributes::set_causal_mask,
             nb::rv_policy::reference_internal)
        .def("set_causal_mask_bottom_right",
             &graph::SdpaBackwardAttributes::set_causal_mask_bottom_right,
             nb::rv_policy::reference_internal)
        .def("set_dropout",
             static_cast<graph::SdpaBackwardAttributes& (
                 graph::SdpaBackwardAttributes::*)(float, const Tensor&, const Tensor&)>(
                 &graph::SdpaBackwardAttributes::set_dropout),
             nb::rv_policy::reference_internal)
        .def("set_attn_scale",
             static_cast<graph::SdpaBackwardAttributes& (graph::SdpaBackwardAttributes::*)(float)>(
                 &graph::SdpaBackwardAttributes::set_attn_scale),
             nb::rv_policy::reference_internal)
        .def("set_diagonal_band_left_bound",
             &graph::SdpaBackwardAttributes::set_diagonal_band_left_bound,
             nb::rv_policy::reference_internal)
        .def("set_diagonal_band_right_bound",
             &graph::SdpaBackwardAttributes::set_diagonal_band_right_bound,
             nb::rv_policy::reference_internal)
        .def("set_diagonal_alignment",
             &graph::SdpaBackwardAttributes::set_diagonal_alignment,
             nb::rv_policy::reference_internal)
        .def("get_q", &graph::SdpaBackwardAttributes::get_q)
        .def("get_k", &graph::SdpaBackwardAttributes::get_k)
        .def("get_v", &graph::SdpaBackwardAttributes::get_v)
        .def("get_o", &graph::SdpaBackwardAttributes::get_o)
        .def("get_do", &graph::SdpaBackwardAttributes::get_do)
        .def("get_stats", &graph::SdpaBackwardAttributes::get_stats)
        .def("get_attn_scale", &graph::SdpaBackwardAttributes::get_attn_scale)
        .def("get_bias", &graph::SdpaBackwardAttributes::get_bias)
        .def("get_seq_len_q", &graph::SdpaBackwardAttributes::get_seq_len_q)
        .def("get_seq_len_kv", &graph::SdpaBackwardAttributes::get_seq_len_kv)
        .def("get_seed", &graph::SdpaBackwardAttributes::get_seed)
        .def("get_offset", &graph::SdpaBackwardAttributes::get_offset)
        .def("get_dropout_mask", &graph::SdpaBackwardAttributes::get_dropout_mask)
        .def("get_dropout_scale", &graph::SdpaBackwardAttributes::get_dropout_scale)
        .def("get_dropout_scale_inv", &graph::SdpaBackwardAttributes::get_dropout_scale_inv)
        .def("get_dq", &graph::SdpaBackwardAttributes::get_dq)
        .def("get_dk", &graph::SdpaBackwardAttributes::get_dk)
        .def("get_dv", &graph::SdpaBackwardAttributes::get_dv)
        .def("get_dbias", &graph::SdpaBackwardAttributes::get_dbias)
        .def_rw("alibi_mask", &graph::SdpaBackwardAttributes::alibi_mask)
        .def_rw("padding_mask", &graph::SdpaBackwardAttributes::padding_mask)
        .def_rw("causal_mask", &graph::SdpaBackwardAttributes::causal_mask)
        .def_rw("causal_mask_bottom_right",
                &graph::SdpaBackwardAttributes::causal_mask_bottom_right)
        .def_rw("dropout_probability", &graph::SdpaBackwardAttributes::dropout_probability)
        .def_rw("attn_scale_value", &graph::SdpaBackwardAttributes::attn_scale_value)
        .def_rw("left_bound", &graph::SdpaBackwardAttributes::left_bound)
        .def_rw("right_bound", &graph::SdpaBackwardAttributes::right_bound)
        .def_rw("diagonal_alignment", &graph::SdpaBackwardAttributes::diagonal_alignment);
}
