// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains facilities for converting hipDNN serialized graphs to
// fusilli graphs.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_PLUGIN_SRC_GRAPH_IMPORT_H
#define FUSILLI_PLUGIN_SRC_GRAPH_IMPORT_H

#include <fusilli.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_bwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_wrw_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/custom_op_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/matmul_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/norm_common_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/reduction_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/rmsnorm_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/sdpa_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include <cstdint>
#include <format>
#include <memory>
#include <optional>
#include <random>
#include <unordered_map>

#include "hipdnn_engine_plugin_execution_context.h"

// Convert from hipDNN DataType to fusilli DataType.
inline fusilli::ErrorOr<fusilli::DataType> hipDnnDataTypeToFusilliDataType(
    hipdnn_flatbuffers_sdk::data_objects::DataType hipdnnType) {
  switch (hipdnnType) {
  case hipdnn_flatbuffers_sdk::data_objects::DataType::HALF:
    return ok(fusilli::DataType::Half);
  case hipdnn_flatbuffers_sdk::data_objects::DataType::BFLOAT16:
    return ok(fusilli::DataType::BFloat16);
  case hipdnn_flatbuffers_sdk::data_objects::DataType::FLOAT:
    return ok(fusilli::DataType::Float);
  case hipdnn_flatbuffers_sdk::data_objects::DataType::DOUBLE:
    return ok(fusilli::DataType::Double);
  case hipdnn_flatbuffers_sdk::data_objects::DataType::INT8:
    return ok(fusilli::DataType::Int8);
  case hipdnn_flatbuffers_sdk::data_objects::DataType::UINT8:
    return ok(fusilli::DataType::Uint8);
  case hipdnn_flatbuffers_sdk::data_objects::DataType::INT32:
    return ok(fusilli::DataType::Int32);
  case hipdnn_flatbuffers_sdk::data_objects::DataType::INT4:
    return ok(fusilli::DataType::Int4);
  case hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET:
    return ok(fusilli::DataType::NotSet);
  default:
    return error(fusilli::ErrorCode::RuntimeFailure,
                 "Unknown type in hipdnn -> fusilli graph translation.");
  }
}

#define FUSILLI_PLUGIN_POINTWISE_CASE(CASE)                                    \
  case hipdnn_flatbuffers_sdk::data_objects::PointwiseMode::CASE:              \
    return fusilli::PointwiseAttr::Mode::CASE;

// Convert from hipDNN PointwiseMode to fusilli PointwiseAttr::Mode.
inline fusilli::ErrorOr<fusilli::PointwiseAttr::Mode>
hipDnnPointwiseModeToFusilliMode(
    hipdnn_flatbuffers_sdk::data_objects::PointwiseMode hipdnnMode) {
  switch (hipdnnMode) {
    FUSILLI_POINTWISE_OPS(FUSILLI_PLUGIN_POINTWISE_CASE)
  default:
    return error(fusilli::ErrorCode::NotImplemented,
                 "Unsupported pointwise mode.");
  }
}

// Convert from hipDNN NormFwdPhase to fusilli NormFwdPhase.
inline fusilli::ErrorOr<fusilli::NormFwdPhase> hipDnnNormFwdPhaseToFusilliPhase(
    hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase hipdnnPhase) {
  switch (hipdnnPhase) {
  case hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase::TRAINING:
    return ok(fusilli::NormFwdPhase::TRAINING);
  case hipdnn_flatbuffers_sdk::data_objects::NormFwdPhase::INFERENCE:
    return ok(fusilli::NormFwdPhase::INFERENCE);
  default:
    return error(fusilli::ErrorCode::NotImplemented,
                 "Unsupported norm forward phase.");
  }
}

// Convert from hipDNN ReductionMode to fusilli ReductionAttr::Mode.
inline fusilli::ErrorOr<fusilli::ReductionAttr::Mode>
hipDnnReductionModeToFusilliMode(
    hipdnn_flatbuffers_sdk::data_objects::ReductionMode hipdnnMode) {
  switch (hipdnnMode) {
  case hipdnn_flatbuffers_sdk::data_objects::ReductionMode::ADD:
    return ok(fusilli::ReductionAttr::Mode::SUM);
  case hipdnn_flatbuffers_sdk::data_objects::ReductionMode::MIN_OP:
    return ok(fusilli::ReductionAttr::Mode::MIN);
  case hipdnn_flatbuffers_sdk::data_objects::ReductionMode::MAX_OP:
    return ok(fusilli::ReductionAttr::Mode::MAX);
  default:
    return error(fusilli::ErrorCode::NotImplemented,
                 "Unsupported reduction mode.");
  }
}

// Graph import is done through importGraph function, this class exists for
// organization and is used by importGraph.
//
// Graph import is designed around individual Node import functions (such as
// importConvFPropAttr) which convert a given node type, and track input and
// output tensors in shared state (via importNodeInput and importNodeOutput
// functions). Graph nodes are processed in topological order to ensure that
// outputs of producer nodes are tracked and available for consuming nodes.
//
// NOTE: inputs should already be topologically sorted, hipDNN's
// Graph::validate() includes a topological sort.
class GraphImport {
private:
  friend fusilli::ErrorOr<HipdnnEnginePluginExecutionContext>
  importGraph(const hipdnnPluginConstData_t *opGraph);

  // The imported graph.
  fusilli::Graph fusilliGraph;

  // Maps hipDNN tensor UIDs to fusilli::TensorAttrs for graph boundary tensors
  // (inputs and outputs). Used by hipdnnEnginePluginExecuteOpGraph to match
  // incoming device buffers (identified by UID) to their corresponding
  // fusilli::TensorAttr.
  std::unordered_map<int64_t, std::shared_ptr<fusilli::TensorAttr>>
      uidToIOTensor;

  // Maps hipDNN tensor UIDs to fusilli::TensorAttrs for intermediate (virtual)
  // tensors. These are outputs of one node that serve as inputs to another.
  std::unordered_map<int64_t, std::shared_ptr<fusilli::TensorAttr>>
      uidToVirtualTensor;

  // Helper class for reading from flatbuffer.
  hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper opGraphWrapper;

  // Per-instance random nonce mixed into the fusilli graph name (see
  // importGraph below).
  uint64_t graphInstanceNonce;

  GraphImport(const hipdnnPluginConstData_t *opGraph)
      : opGraphWrapper(opGraph->ptr, opGraph->size),
        graphInstanceNonce(makeGraphInstanceNonce()) {}

  static uint64_t makeGraphInstanceNonce() {
    std::random_device rd;
    std::seed_seq seq{rd(), rd()};
    std::mt19937_64 rng(seq);
    return rng();
  }

  fusilli::ErrorObject importGraph() {
    const hipdnn_flatbuffers_sdk::data_objects::Graph &hipDnnGraph =
        opGraphWrapper.getGraph();

    // Import graph level properties.
    fusilli::DataType ioDataType;
    FUSILLI_ASSIGN_OR_RETURN(ioDataType, hipDnnDataTypeToFusilliDataType(
                                             hipDnnGraph.io_data_type()));
    fusilli::DataType intermediateDataType;
    FUSILLI_ASSIGN_OR_RETURN(
        intermediateDataType,
        hipDnnDataTypeToFusilliDataType(hipDnnGraph.intermediate_data_type()));
    fusilli::DataType computeDataType;
    FUSILLI_ASSIGN_OR_RETURN(
        computeDataType,
        hipDnnDataTypeToFusilliDataType(hipDnnGraph.compute_data_type()));
    // Mix the per-instance nonce into the fusilli graph name so each Graph
    // gets a unique compile-cache directory. Also covers the null-name case
    // that would otherwise segfault on name()->str().
    std::string graphName =
        hipDnnGraph.name()
            ? std::format("{}_{:016x}", hipDnnGraph.name()->str(),
                          graphInstanceNonce)
            : std::format("hipdnn_{:016x}", graphInstanceNonce);
    fusilliGraph.setName(graphName)
        .setIODataType(ioDataType)
        .setIntermediateDataType(intermediateDataType)
        .setComputeDataType(computeDataType);

    return importNodes();
  }

  // Import all graph nodes.
  fusilli::ErrorObject importNodes() {
    for (uint32_t i = 0; i < opGraphWrapper.nodeCount(); ++i) {
      const hipdnn_flatbuffers_sdk::data_objects::Node &node =
          opGraphWrapper.getNode(i);
      FUSILLI_CHECK_ERROR(importNode(node));
    }

    return fusilli::ok();
  }

  // Import single graph node.
  fusilli::ErrorObject
  importNode(const hipdnn_flatbuffers_sdk::data_objects::Node &node) {
    switch (node.attributes_type()) {
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
        ConvolutionFwdAttributes:
      FUSILLI_CHECK_ERROR(
          importConvFPropAttr(node.attributes_as_ConvolutionFwdAttributes()));
      break;
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
        ConvolutionWrwAttributes:
      FUSILLI_CHECK_ERROR(
          importConvWGradAttr(node.attributes_as_ConvolutionWrwAttributes()));
      break;
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
        ConvolutionBwdAttributes:
      FUSILLI_CHECK_ERROR(
          importConvDGradAttr(node.attributes_as_ConvolutionBwdAttributes()));
      break;
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
        PointwiseAttributes:
      FUSILLI_CHECK_ERROR(
          importPointwiseAttr(node.attributes_as_PointwiseAttributes()));
      break;
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::MatmulAttributes:
      FUSILLI_CHECK_ERROR(
          importMatmulAttr(node.attributes_as_MatmulAttributes()));
      break;
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::SdpaAttributes:
      FUSILLI_CHECK_ERROR(importSdpaAttr(node.attributes_as_SdpaAttributes()));
      break;
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
        ReductionAttributes:
      FUSILLI_CHECK_ERROR(
          importReductionAttr(node.attributes_as_ReductionAttributes()));
      break;
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
        RMSNormAttributes:
      FUSILLI_CHECK_ERROR(
          importRmsnormAttr(node.attributes_as_RMSNormAttributes()));
      break;
    case hipdnn_flatbuffers_sdk::data_objects::NodeAttributes::
        CustomOpAttributes:
      FUSILLI_CHECK_ERROR(
          importCustomOpAttr(node.attributes_as_CustomOpAttributes()));
      break;
    default:
      return fusilli::error(fusilli::ErrorCode::NotImplemented,
                            "Unsupported node type.");
    }
    return fusilli::ok();
  }

  fusilli::ErrorObject importConvFPropAttr(
      const hipdnn_flatbuffers_sdk::data_objects::ConvolutionFwdAttributes
          *hipDnnConvFwdAttr) {
    // Import node inputs.
    FUSILLI_ASSIGN_OR_RETURN(
        std::shared_ptr<fusilli::TensorAttr> x,
        importNodeInput(hipDnnConvFwdAttr->x_tensor_uid(), "x"));
    FUSILLI_ASSIGN_OR_RETURN(
        std::shared_ptr<fusilli::TensorAttr> w,
        importNodeInput(hipDnnConvFwdAttr->w_tensor_uid(), "w"));

    // Fusilli only supports symmetric padding.
    if (!std::ranges::equal(*hipDnnConvFwdAttr->pre_padding(),
                            *hipDnnConvFwdAttr->post_padding())) // C++20
      return fusilli::error(fusilli::ErrorCode::AttributeNotSet,
                            "Conv node with asymmetric padding found.");
    // Import node.
    auto fusilliConvFwdAttr =
        fusilli::ConvFPropAttr()
            .setPadding(*hipDnnConvFwdAttr->post_padding())
            .setStride(*hipDnnConvFwdAttr->stride())
            .setDilation(*hipDnnConvFwdAttr->dilation());
    std::shared_ptr<fusilli::TensorAttr> y =
        fusilliGraph.convFProp(x, w, fusilliConvFwdAttr);

    // Import node output.
    FUSILLI_CHECK_ERROR(
        importNodeOutput(hipDnnConvFwdAttr->y_tensor_uid(), "y", y));

    return fusilli::ok();
  }

  fusilli::ErrorObject importConvWGradAttr(
      const hipdnn_flatbuffers_sdk::data_objects::ConvolutionWrwAttributes
          *hipDnnConvWrwAttr) {
    // Import node inputs.
    FUSILLI_ASSIGN_OR_RETURN(
        std::shared_ptr<fusilli::TensorAttr> dy,
        importNodeInput(hipDnnConvWrwAttr->dy_tensor_uid(), "dy"));
    FUSILLI_ASSIGN_OR_RETURN(
        std::shared_ptr<fusilli::TensorAttr> x,
        importNodeInput(hipDnnConvWrwAttr->x_tensor_uid(), "x"));

    // Fusilli only supports symmetric padding.
    if (!std::ranges::equal(*hipDnnConvWrwAttr->pre_padding(),
                            *hipDnnConvWrwAttr->post_padding())) // C++20
      return fusilli::error(fusilli::ErrorCode::AttributeNotSet,
                            "Conv wgrad node with asymmetric padding found.");
    // Import node.
    auto fusilliConvWGradAttr =
        fusilli::ConvWGradAttr()
            .setPadding(*hipDnnConvWrwAttr->post_padding())
            .setStride(*hipDnnConvWrwAttr->stride())
            .setDilation(*hipDnnConvWrwAttr->dilation());
    std::shared_ptr<fusilli::TensorAttr> dw =
        fusilliGraph.convWGrad(dy, x, fusilliConvWGradAttr);

    // Import node output.
    FUSILLI_CHECK_ERROR(
        importNodeOutput(hipDnnConvWrwAttr->dw_tensor_uid(), "dw", dw));

    return fusilli::ok();
  }

  fusilli::ErrorObject importConvDGradAttr(
      const hipdnn_flatbuffers_sdk::data_objects::ConvolutionBwdAttributes
          *hipDnnConvBwdAttr) {
    // Import node inputs.
    FUSILLI_ASSIGN_OR_RETURN(
        std::shared_ptr<fusilli::TensorAttr> dy,
        importNodeInput(hipDnnConvBwdAttr->dy_tensor_uid(), "dy"));
    FUSILLI_ASSIGN_OR_RETURN(
        std::shared_ptr<fusilli::TensorAttr> w,
        importNodeInput(hipDnnConvBwdAttr->w_tensor_uid(), "w"));

    // Fusilli only supports symmetric padding.
    if (!std::ranges::equal(*hipDnnConvBwdAttr->pre_padding(),
                            *hipDnnConvBwdAttr->post_padding())) // C++20
      return fusilli::error(fusilli::ErrorCode::AttributeNotSet,
                            "Conv dgrad node with asymmetric padding found.");
    // Import node.
    auto fusilliConvDGradAttr =
        fusilli::ConvDGradAttr()
            .setPadding(*hipDnnConvBwdAttr->post_padding())
            .setStride(*hipDnnConvBwdAttr->stride())
            .setDilation(*hipDnnConvBwdAttr->dilation());
    std::shared_ptr<fusilli::TensorAttr> dx =
        fusilliGraph.convDGrad(dy, w, fusilliConvDGradAttr);

    // Import node output.
    FUSILLI_CHECK_ERROR(
        importNodeOutput(hipDnnConvBwdAttr->dx_tensor_uid(), "dx", dx));

    return fusilli::ok();
  }

  fusilli::ErrorObject importPointwiseAttr(
      const hipdnn_flatbuffers_sdk::data_objects::PointwiseAttributes
          *hipDnnPwAttr) {
    // Get mode and determine input count.
    FUSILLI_ASSIGN_OR_RETURN(
        fusilli::PointwiseAttr::Mode mode,
        hipDnnPointwiseModeToFusilliMode(hipDnnPwAttr->operation()));
    int requiredInputs =
        fusilli::PointwiseAttr::kModeToRequiredInputCount.at(mode);

    // Import first input (always present).
    FUSILLI_ASSIGN_OR_RETURN(
        std::shared_ptr<fusilli::TensorAttr> in0,
        importNodeInput(hipDnnPwAttr->in_0_tensor_uid(), "in0"));

    // Build fusilli pointwise node.
    std::shared_ptr<fusilli::TensorAttr> out;
    auto fusilliPwAttr = fusilli::PointwiseAttr().setMode(mode);

    switch (requiredInputs) {
    case 1:
      // Unary op (e.g., RELU_FWD).
      out = fusilliGraph.pointwise(in0, fusilliPwAttr);
      break;
    case 2: {
      // Binary op (e.g., ADD, MUL, SUB, DIV).
      auto in1Uid = hipDnnPwAttr->in_1_tensor_uid();
      if (!in1Uid.has_value())
        return fusilli::error(fusilli::ErrorCode::AttributeNotSet,
                              "Binary pointwise op missing second input.");
      FUSILLI_ASSIGN_OR_RETURN(std::shared_ptr<fusilli::TensorAttr> in1,
                               importNodeInput(in1Uid.value(), "in1"));
      out = fusilliGraph.pointwise(in0, in1, fusilliPwAttr);
      break;
    }
    default:
      return fusilli::error(fusilli::ErrorCode::RuntimeFailure,
                            "Unexpected number of inputs to pointwise op.");
    }

    // Import node output.
    FUSILLI_CHECK_ERROR(
        importNodeOutput(hipDnnPwAttr->out_0_tensor_uid(), "out0", out));

    return fusilli::ok();
  }

  fusilli::ErrorObject
  importMatmulAttr(const hipdnn_flatbuffers_sdk::data_objects::MatmulAttributes
                       *hipDnnMatmulAttr) {
    // Import node inputs.
    FUSILLI_ASSIGN_OR_RETURN(
        std::shared_ptr<fusilli::TensorAttr> a,
        importNodeInput(hipDnnMatmulAttr->a_tensor_uid(), "a"));
    FUSILLI_ASSIGN_OR_RETURN(
        std::shared_ptr<fusilli::TensorAttr> b,
        importNodeInput(hipDnnMatmulAttr->b_tensor_uid(), "b"));

    // Import node - matmul has no extra attributes.
    auto fusilliMatmulAttr = fusilli::MatmulAttr();
    std::shared_ptr<fusilli::TensorAttr> c =
        fusilliGraph.matmul(a, b, fusilliMatmulAttr);

    // Import node output.
    FUSILLI_CHECK_ERROR(
        importNodeOutput(hipDnnMatmulAttr->c_tensor_uid(), "c", c));

    return fusilli::ok();
  }

  fusilli::ErrorObject
  importSdpaAttr(const hipdnn_flatbuffers_sdk::data_objects::SdpaAttributes
                     *hipDnnSdpaAttr) {
    // Reject hipDNN features not supported by the fusilli SDPA path.
    if (hipDnnSdpaAttr->dropout_probability().has_value() &&
        *hipDnnSdpaAttr->dropout_probability() > 0.0f) {
      return fusilli::error(fusilli::ErrorCode::NotImplemented,
                            "SDPA with dropout not supported.");
    }
    if (hipDnnSdpaAttr->alibi_mask()) {
      return fusilli::error(fusilli::ErrorCode::NotImplemented,
                            "SDPA with alibi mask not supported.");
    }
    if (hipDnnSdpaAttr->padding_mask()) {
      return fusilli::error(fusilli::ErrorCode::NotImplemented,
                            "SDPA with padding mask not supported.");
    }
    if (hipDnnSdpaAttr->stats_tensor_uid().has_value()) {
      return fusilli::error(fusilli::ErrorCode::NotImplemented,
                            "SDPA with stats output not supported.");
    }
    if (hipDnnSdpaAttr->seed_tensor_uid().has_value() ||
        hipDnnSdpaAttr->offset_tensor_uid().has_value() ||
        hipDnnSdpaAttr->dropout_mask_tensor_uid().has_value() ||
        hipDnnSdpaAttr->dropout_scale_tensor_uid().has_value()) {

      return fusilli::error(fusilli::ErrorCode::NotImplemented,
                            "SDPA with dropout tensors not supported.");
    }
    if (hipDnnSdpaAttr->page_table_k_tensor_uid().has_value() ||
        hipDnnSdpaAttr->page_table_v_tensor_uid().has_value()) {
      return fusilli::error(fusilli::ErrorCode::NotImplemented,
                            "SDPA with paged attention not supported.");
    }
    if (hipDnnSdpaAttr->block_mask_tensor_uid().has_value() ||
        hipDnnSdpaAttr->sink_token_tensor_uid().has_value()) {
      return fusilli::error(fusilli::ErrorCode::NotImplemented,
                            "SDPA with block mask not supported.");
    }
    if (hipDnnSdpaAttr->left_bound().has_value() ||
        hipDnnSdpaAttr->right_bound().has_value()) {
      return fusilli::error(fusilli::ErrorCode::NotImplemented,
                            "SDPA with sliding window not supported.");
    }
    if (hipDnnSdpaAttr->descale_q_tensor_uid().has_value() ||
        hipDnnSdpaAttr->descale_k_tensor_uid().has_value() ||
        hipDnnSdpaAttr->descale_v_tensor_uid().has_value() ||
        hipDnnSdpaAttr->descale_s_tensor_uid().has_value() ||
        hipDnnSdpaAttr->scale_s_tensor_uid().has_value() ||
        hipDnnSdpaAttr->scale_o_tensor_uid().has_value() ||
        hipDnnSdpaAttr->amax_s_tensor_uid().has_value() ||
        hipDnnSdpaAttr->amax_o_tensor_uid().has_value()) {
      return fusilli::error(fusilli::ErrorCode::NotImplemented,
                            "SDPA with FP8 quantization not supported.");
    }
    if (hipDnnSdpaAttr->diagonal_alignment() !=
        hipdnn_flatbuffers_sdk::data_objects::DiagonalAlignment::TOP_LEFT) {
      return fusilli::error(
          fusilli::ErrorCode::NotImplemented,
          "SDPA with non-TOP_LEFT diagonal alignment not supported.");
    }
    // This is out of an over-abundance of caution, IREE doesn't need a backend
    // implementation hint so if one is given the user is likely targeting a
    // different backend.
    if (hipDnnSdpaAttr->implementation() !=
        hipdnn_flatbuffers_sdk::data_objects::AttentionImplementation::AUTO) {
      return fusilli::error(
          fusilli::ErrorCode::NotImplemented,
          "SDPA with explicit implementation strategy not supported.");
    }
    // Causal attention implies an explicit attn_mask, additional attention mask
    // doesn't make sense and torch dialect will reject it.
    if (hipDnnSdpaAttr->causal_mask() &&
        hipDnnSdpaAttr->attn_mask_tensor_uid().has_value()) {
      return fusilli::error(
          fusilli::ErrorCode::NotImplemented,
          "SDPA with both causal mask and attention mask not supported.");
    }
    // mma_core_mode requests a specific accumulator precision. Fusilli's
    // lowering path accumulates in the query element type, so reject if the
    // requested mode doesn't match. UNSET (the default) is always fine.
    hipdnn_flatbuffers_sdk::data_objects::DataType mmaCoreMode =
        hipDnnSdpaAttr->mma_core_mode();
    if (mmaCoreMode != hipdnn_flatbuffers_sdk::data_objects::DataType::UNSET) {
      hipdnn_flatbuffers_sdk::data_objects::DataType qDataType =
          opGraphWrapper.getTensorMap()
              .at(hipDnnSdpaAttr->q_tensor_uid())
              ->data_type();
      if (mmaCoreMode != qDataType) {
        return fusilli::error(
            fusilli::ErrorCode::NotImplemented,
            "SDPA mma_core_mode must match query tensor dtype.");
      }
    }

    bool hasAttnMask = hipDnnSdpaAttr->attn_mask_tensor_uid().has_value();
    bool isCausal = hipDnnSdpaAttr->causal_mask();

    // Import required tensor inputs.
    FUSILLI_ASSIGN_OR_RETURN(
        std::shared_ptr<fusilli::TensorAttr> q,
        importNodeInput(hipDnnSdpaAttr->q_tensor_uid(), "q"));
    FUSILLI_ASSIGN_OR_RETURN(
        std::shared_ptr<fusilli::TensorAttr> k,
        importNodeInput(hipDnnSdpaAttr->k_tensor_uid(), "k"));
    FUSILLI_ASSIGN_OR_RETURN(
        std::shared_ptr<fusilli::TensorAttr> v,
        importNodeInput(hipDnnSdpaAttr->v_tensor_uid(), "v"));

    // Import optional attn_mask tensor.
    std::shared_ptr<fusilli::TensorAttr> mask = nullptr;
    if (hasAttnMask) {
      FUSILLI_ASSIGN_OR_RETURN(
          mask, importNodeInput(*hipDnnSdpaAttr->attn_mask_tensor_uid(),
                                "attn_mask"));
    }

    // Read optional attention scale.
    std::optional<float> scaleValue;
    if (hipDnnSdpaAttr->attn_scale_value().has_value()) {
      // Scalar float attribute set directly on the SDPA node.
      scaleValue = *hipDnnSdpaAttr->attn_scale_value();
    } else if (hipDnnSdpaAttr->scale_tensor_uid().has_value()) {
      // Scale provided as a tensor UID — only pass-by-value scalars are
      // supported since the scale must be a compile-time constant.
      const auto *scaleTensor =
          opGraphWrapper.getTensorMap().at(*hipDnnSdpaAttr->scale_tensor_uid());
      if (!isPassByValue(scaleTensor))
        return fusilli::error(fusilli::ErrorCode::NotImplemented,
                              "SDPA scale must be a pass-by-value scalar, "
                              "not a device tensor.");
      if (scaleTensor->value_type() !=
          hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value)
        return fusilli::error(fusilli::ErrorCode::NotImplemented,
                              "SDPA scale tensor must be Float32.");
      scaleValue = scaleTensor->value_as_Float32Value()->value();
    }

    // GQA: enable when Q has more heads than K or V.
    bool enableGqa =
        q->getDim()[1] != k->getDim()[1] || q->getDim()[1] != v->getDim()[1];

    // #TODO(iree/issues/21858) GQA with f32 triggers an IREE distribution
    // failure. SdpaNode does not check this, so we must reject here.
    if (enableGqa && q->getDataType() == fusilli::DataType::Float)
      return fusilli::error(fusilli::ErrorCode::NotImplemented,
                            "SDPA GQA with f32 not supported.");

    // Import node.
    fusilli::SdpaAttr sdpaAttr;
    sdpaAttr.setName("sdpa_fprop")
        .setIsCausal(isCausal)
        .setEnableGqa(enableGqa);
    if (scaleValue.has_value())
      sdpaAttr.setScale(scaleValue);
    std::shared_ptr<fusilli::TensorAttr> o =
        fusilliGraph.sdpa(q, k, v, mask, sdpaAttr);

    // Import node output.
    FUSILLI_CHECK_ERROR(
        importNodeOutput(hipDnnSdpaAttr->o_tensor_uid(), "o", o));

    return fusilli::ok();
  }

  fusilli::ErrorObject importReductionAttr(
      const hipdnn_flatbuffers_sdk::data_objects::ReductionAttributes
          *hipDnnRedAttr) {
    // Import node input.
    FUSILLI_ASSIGN_OR_RETURN(
        std::shared_ptr<fusilli::TensorAttr> x,
        importNodeInput(hipDnnRedAttr->in_tensor_uid(), "x"));

    // Convert reduction mode.
    FUSILLI_ASSIGN_OR_RETURN(
        fusilli::ReductionAttr::Mode mode,
        hipDnnReductionModeToFusilliMode(hipDnnRedAttr->mode()));

    // Import node.
    auto fusilliRedAttr = fusilli::ReductionAttr().setMode(mode);
    std::shared_ptr<fusilli::TensorAttr> y =
        fusilliGraph.reduction(x, fusilliRedAttr);

    // Import node output.
    FUSILLI_CHECK_ERROR(
        importNodeOutput(hipDnnRedAttr->out_tensor_uid(), "y", y));

    return fusilli::ok();
  }

  fusilli::ErrorObject importRmsnormAttr(
      const hipdnn_flatbuffers_sdk::data_objects::RMSNormAttributes
          *hipDnnRmsnormAttr) {
    // Fusilli's rmsnorm does not support the optional bias input.
    if (hipDnnRmsnormAttr->bias_tensor_uid().has_value())
      return fusilli::error(fusilli::ErrorCode::NotImplemented,
                            "RmsNorm with bias input is not supported.");

    // Import node inputs.
    FUSILLI_ASSIGN_OR_RETURN(
        std::shared_ptr<fusilli::TensorAttr> x,
        importNodeInput(hipDnnRmsnormAttr->x_tensor_uid(), "x"));
    FUSILLI_ASSIGN_OR_RETURN(
        std::shared_ptr<fusilli::TensorAttr> scale,
        importNodeInput(hipDnnRmsnormAttr->scale_tensor_uid(), "scale"));
    FUSILLI_ASSIGN_OR_RETURN(
        std::shared_ptr<fusilli::TensorAttr> epsilon,
        importNodeInput(hipDnnRmsnormAttr->epsilon_tensor_uid(), "epsilon"));

    // Fusilli requires epsilon to be a scalar constant (pass-by-value).
    if (!epsilon->isScalar())
      return fusilli::error(
          fusilli::ErrorCode::NotImplemented,
          "RmsNorm epsilon must be a pass-by-value scalar, not a device "
          "tensor.");

    bool hasInvRms = hipDnnRmsnormAttr->inv_rms_tensor_uid().has_value();
    FUSILLI_ASSIGN_OR_RETURN(
        fusilli::NormFwdPhase forwardPhase,
        hipDnnNormFwdPhaseToFusilliPhase(hipDnnRmsnormAttr->forward_phase()));

    // The hipDNN frontend should have already rejected this mismatch; this
    // check exists as a defensive guard for the plugin boundary.
    if ((forwardPhase == fusilli::NormFwdPhase::TRAINING) != hasInvRms)
      return fusilli::error(
          fusilli::ErrorCode::InvalidAttribute,
          "RmsNorm inv_rms output must be set if and only if forward phase "
          "is TRAINING.");

    // Import node.
    auto fusilliRmsnormAttr =
        fusilli::RmsnormAttr().setEpsilon(epsilon).setForwardPhase(
            forwardPhase);
    auto [y, invRms] = fusilliGraph.rmsnorm(x, scale, fusilliRmsnormAttr);

    // Import node outputs.
    FUSILLI_CHECK_ERROR(
        importNodeOutput(hipDnnRmsnormAttr->y_tensor_uid(), "y", y));
    if (hasInvRms) {
      FUSILLI_CHECK_ERROR(importNodeOutput(
          *hipDnnRmsnormAttr->inv_rms_tensor_uid(), "inv_rms", invRms));
    }

    return fusilli::ok();
  }

  fusilli::ErrorObject importCustomOpAttr(
      const hipdnn_flatbuffers_sdk::data_objects::CustomOpAttributes
          *hipDnnAttr) {
    // Only import custom ops targeting this plugin.
    std::string customOpId = hipDnnAttr->custom_op_id()->str();
    if (!customOpId.starts_with("fusilli.")) {
      return fusilli::error(
          fusilli::ErrorCode::NotImplemented,
          std::format("Custom op id '{}' does not target the fusilli plugin. "
                      "Expected 'fusilli.<operation>' prefix.",
                      customOpId));
    }

    // The data byte array is the MLIR template string directly.
    std::string mlirString(
        reinterpret_cast<const char *>(hipDnnAttr->data()->data()),
        hipDnnAttr->data()->size());

    // Import input tensors (variable-length).
    std::vector<std::shared_ptr<fusilli::TensorAttr>> inputs;
    for (auto uid : *hipDnnAttr->input_tensor_uids()) {
      FUSILLI_ASSIGN_OR_RETURN(auto tensor, importNodeInput(uid, "custom_in"));
      inputs.push_back(std::move(tensor));
    }

    // Build fusilli CustomOpAttr. numOutputs derived from flatbuffer directly.
    auto numOutputs = hipDnnAttr->output_tensor_uids()->size();
    fusilli::CustomOpAttr fusilliAttr;
    fusilliAttr.setName(customOpId)
        .setMlir(mlirString)
        .setNumOutputs(numOutputs);

    // Create custom op node in fusilli graph.
    auto outputTensors = fusilliGraph.customOp(inputs, fusilliAttr);

    // Import output tensors.
    auto *outputUids = hipDnnAttr->output_tensor_uids();
    for (size_t i = 0; i < outputUids->size(); ++i) {
      FUSILLI_CHECK_ERROR(importNodeOutput(
          outputUids->Get(static_cast<flatbuffers::uoffset_t>(i)), "custom_out",
          outputTensors[i]));
    }

    return fusilli::ok();
  }

  // Import, and track, node input tensor. Node input tensor is created in the
  // case of a boundary tensor, and read from shared state otherwise.
  fusilli::ErrorOr<std::shared_ptr<fusilli::TensorAttr>>
  importNodeInput(int64_t uid, const char *name) {
    // Get hipDNN tensor. TensorMap is created from the graph that uid variable
    // is read from, so .at() call should be safe.
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes
        *hipDnnTensorAttr = opGraphWrapper.getTensorMap().at(uid);

    // A virtual tensor indicates an intermediate (non-boundary) tensor.
    if (hipDnnTensorAttr->virtual_()) {
      // Look up the output of a previously imported node.
      if (!uidToVirtualTensor.contains(uid))
        return fusilli::error(fusilli::ErrorCode::RuntimeFailure,
                              "Virtual tensor not found - graph may not be "
                              "topologically sorted.");
      return ok(uidToVirtualTensor.at(uid));
    }

    // Import new tensor.
    fusilli::TensorAttr fusilliTensorAttr;
    if (isPassByValue(hipDnnTensorAttr)) { // handle scalar tensors
      switch (hipDnnTensorAttr->value_type()) {
      case hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float32Value:
        fusilliTensorAttr = fusilli::TensorAttr(
            hipDnnTensorAttr->value_as_Float32Value()->value());
        break;
      case hipdnn_flatbuffers_sdk::data_objects::TensorValue::Float64Value:
        fusilliTensorAttr = fusilli::TensorAttr(
            hipDnnTensorAttr->value_as_Float64Value()->value());
        break;
      case hipdnn_flatbuffers_sdk::data_objects::TensorValue::Int32Value:
        fusilliTensorAttr = fusilli::TensorAttr(
            hipDnnTensorAttr->value_as_Int32Value()->value());
        break;
      default:
        return fusilli::error(
            fusilli::ErrorCode::NotImplemented,
            "Unsupported scalar type in hipdnn -> fusilli graph translation.");
      }
    }
    fusilliTensorAttr.setName(std::format("{}_{}", name, uid)); // C++20
    FUSILLI_CHECK_ERROR(importAttrs(fusilliTensorAttr, hipDnnTensorAttr));
    std::shared_ptr<fusilli::TensorAttr> graphInput =
        fusilliGraph.tensor(fusilliTensorAttr);

    // Scalar constants are embedded in the MLIR IR and don't need device
    // buffers, so exclude them from the IO tensor map that drives variant
    // pack construction at execution time.
    if (!graphInput->isScalar())
      uidToIOTensor[uid] = graphInput;

    return ok(graphInput);
  };

  // Import and track node output tensor.
  fusilli::ErrorObject
  importNodeOutput(int64_t uid, const char *name,
                   const std::shared_ptr<fusilli::TensorAttr> &nodeOutput) {
    // Get hipDNN tensor. TensorMap is created from the graph that uid variable
    // is read from, so .at() call should be safe.
    const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes
        *hipDnnTensorAttr = opGraphWrapper.getTensorMap().at(uid);

    // Import attrs.
    nodeOutput->setName(std::format("{}_{}", name, uid)); // C++20
    FUSILLI_CHECK_ERROR(importAttrs(*nodeOutput, hipDnnTensorAttr));

    // A virtual tensor indicates an intermediate (non-boundary) tensor.
    if (hipDnnTensorAttr->virtual_()) {
      // Check for duplicate UIDs.
      if (uidToVirtualTensor.contains(uid))
        return fusilli::error(
            fusilli::ErrorCode::RuntimeFailure,
            "Duplicate virtual tensor UID - UIDs must be unique.");
      // Track for use by downstream nodes.
      uidToVirtualTensor[uid] = nodeOutput;
      return fusilli::ok();
    }

    // Track boundary tensor.
    uidToIOTensor[uid] = nodeOutput;

    return fusilli::ok();
  };

  // Whether the hipDNN tensor carries a pass-by-value scalar (equivalent to
  // hipDNN frontend's TensorAttributes::get_pass_by_value()).
  static bool isPassByValue(
      const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes *src) {
    return src->value_type() !=
           hipdnn_flatbuffers_sdk::data_objects::TensorValue::NONE;
  }

  // Import tensor attrs (dims, strides, datatype) from hipDNN to fusilli.
  fusilli::ErrorObject importAttrs(
      fusilli::TensorAttr &dest,
      const hipdnn_flatbuffers_sdk::data_objects::TensorAttributes *src) {
    FUSILLI_ASSIGN_OR_RETURN(auto dataType,
                             hipDnnDataTypeToFusilliDataType(src->data_type()));
    dest.setIsVirtual(src->virtual_())
        .setDim(*src->dims())
        .setStride(*src->strides())
        .setDataType(dataType);
    return fusilli::ok();
  }
};

// Given a hipDNN serialized graph, return imported fusilli::Graph and UID ->
// fusilli::TensorAttr map for IO tensors.
//
// NOTE: HipdnnEnginePluginExecutionContext used as return type because it
// contains (only) the exact required fields. If it requires more members in
// the future it's probably worth creating a new data transmission type.
inline fusilli::ErrorOr<HipdnnEnginePluginExecutionContext>
importGraph(const hipdnnPluginConstData_t *opGraph) {
  auto gc = GraphImport(opGraph);
  FUSILLI_CHECK_ERROR(gc.importGraph());
  FUSILLI_CHECK_ERROR(gc.fusilliGraph.validate());
  return HipdnnEnginePluginExecutionContext{.graph = std::move(gc.fusilliGraph),
                                            .uidToFusilliTensorAttr =
                                                std::move(gc.uidToIOTensor)};
}

#endif // FUSILLI_PLUGIN_SRC_GRAPH_IMPORT_H
