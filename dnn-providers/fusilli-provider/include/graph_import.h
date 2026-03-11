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
#include <hipdnn_data_sdk/data_objects/convolution_wrw_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/data_types_generated.h>
#include <hipdnn_data_sdk/data_objects/matmul_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/pointwise_attributes_generated.h>
#include <hipdnn_data_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_data_sdk/flatbuffer_utilities/GraphWrapper.hpp>
#include <hipdnn_plugin_sdk/PluginApiDataTypes.h>

#include <format>
#include <memory>
#include <unordered_map>

#include "hipdnn_engine_plugin_execution_context.h"

// Convert from hipDNN DataType to fusilli DataType.
inline fusilli::ErrorOr<fusilli::DataType> hipDnnDataTypeToFusilliDataType(
    hipdnn_data_sdk::data_objects::DataType hipdnnType) {
  switch (hipdnnType) {
  case hipdnn_data_sdk::data_objects::DataType::HALF:
    return ok(fusilli::DataType::Half);
  case hipdnn_data_sdk::data_objects::DataType::BFLOAT16:
    return ok(fusilli::DataType::BFloat16);
  case hipdnn_data_sdk::data_objects::DataType::FLOAT:
    return ok(fusilli::DataType::Float);
  case hipdnn_data_sdk::data_objects::DataType::DOUBLE:
    return ok(fusilli::DataType::Double);
  case hipdnn_data_sdk::data_objects::DataType::UINT8:
    return ok(fusilli::DataType::Uint8);
  case hipdnn_data_sdk::data_objects::DataType::INT32:
    return ok(fusilli::DataType::Int32);
  case hipdnn_data_sdk::data_objects::DataType::UNSET:
    return ok(fusilli::DataType::NotSet);
  default:
    return error(fusilli::ErrorCode::RuntimeFailure,
                 "Unknown type in hipdnn -> fusilli graph translation.");
  }
}

#define FUSILLI_PLUGIN_POINTWISE_CASE(CASE)                                    \
  case hipdnn_data_sdk::data_objects::PointwiseMode::CASE:                     \
    return fusilli::PointwiseAttr::Mode::CASE;

// Convert from hipDNN PointwiseMode to fusilli PointwiseAttr::Mode.
inline fusilli::ErrorOr<fusilli::PointwiseAttr::Mode>
hipDnnPointwiseModeToFusilliMode(
    hipdnn_data_sdk::data_objects::PointwiseMode hipdnnMode) {
  switch (hipdnnMode) {
    FUSILLI_POINTWISE_OPS(FUSILLI_PLUGIN_POINTWISE_CASE)
  default:
    return error(fusilli::ErrorCode::NotImplemented,
                 "Unsupported pointwise mode.");
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
  hipdnn_plugin_sdk::GraphWrapper opGraphWrapper;

  GraphImport(const hipdnnPluginConstData_t *opGraph)
      : opGraphWrapper(opGraph->ptr, opGraph->size) {}

  fusilli::ErrorObject importGraph() {
    const hipdnn_data_sdk::data_objects::Graph &hipDnnGraph =
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
    fusilliGraph.setName(hipDnnGraph.name()->str())
        .setIODataType(ioDataType)
        .setIntermediateDataType(intermediateDataType)
        .setComputeDataType(computeDataType);

    return importNodes();
  }

  // Import all graph nodes.
  fusilli::ErrorObject importNodes() {
    for (uint32_t i = 0; i < opGraphWrapper.nodeCount(); ++i) {
      const hipdnn_data_sdk::data_objects::Node &node =
          opGraphWrapper.getNode(i);
      FUSILLI_CHECK_ERROR(importNode(node));
    }

    return fusilli::ok();
  }

  // Import single graph node.
  fusilli::ErrorObject
  importNode(const hipdnn_data_sdk::data_objects::Node &node) {
    switch (node.attributes_type()) {
    case hipdnn_data_sdk::data_objects::NodeAttributes::
        ConvolutionFwdAttributes:
      FUSILLI_CHECK_ERROR(
          importConvFPropAttr(node.attributes_as_ConvolutionFwdAttributes()));
      break;
    case hipdnn_data_sdk::data_objects::NodeAttributes::
        ConvolutionWrwAttributes:
      FUSILLI_CHECK_ERROR(
          importConvWGradAttr(node.attributes_as_ConvolutionWrwAttributes()));
      break;
    case hipdnn_data_sdk::data_objects::NodeAttributes::PointwiseAttributes:
      FUSILLI_CHECK_ERROR(
          importPointwiseAttr(node.attributes_as_PointwiseAttributes()));
      break;
    case hipdnn_data_sdk::data_objects::NodeAttributes::MatmulAttributes:
      FUSILLI_CHECK_ERROR(
          importMatmulAttr(node.attributes_as_MatmulAttributes()));
      break;
    default:
      return fusilli::error(fusilli::ErrorCode::NotImplemented,
                            "Unsupported node type.");
    }
    return fusilli::ok();
  }

  fusilli::ErrorObject importConvFPropAttr(
      const hipdnn_data_sdk::data_objects::ConvolutionFwdAttributes
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
      const hipdnn_data_sdk::data_objects::ConvolutionWrwAttributes
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

  fusilli::ErrorObject importPointwiseAttr(
      const hipdnn_data_sdk::data_objects::PointwiseAttributes *hipDnnPwAttr) {
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

  fusilli::ErrorObject importMatmulAttr(
      const hipdnn_data_sdk::data_objects::MatmulAttributes *hipDnnMatmulAttr) {
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

  // Import, and track, node input tensor. Node input tensor is created in the
  // case of a boundary tensor, and read from shared state otherwise.
  fusilli::ErrorOr<std::shared_ptr<fusilli::TensorAttr>>
  importNodeInput(int64_t uid, const char *name) {
    // Get hipDNN tensor. TensorMap is created from the graph that uid variable
    // is read from, so .at() call should be safe.
    const hipdnn_data_sdk::data_objects::TensorAttributes *hipDnnTensorAttr =
        opGraphWrapper.getTensorMap().at(uid);

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
      case hipdnn_data_sdk::data_objects::TensorValue::Float32Value:
        fusilliTensorAttr = fusilli::TensorAttr(
            hipDnnTensorAttr->value_as_Float32Value()->value());
        break;
      case hipdnn_data_sdk::data_objects::TensorValue::Float64Value:
        fusilliTensorAttr = fusilli::TensorAttr(
            hipDnnTensorAttr->value_as_Float64Value()->value());
        break;
      case hipdnn_data_sdk::data_objects::TensorValue::Int32Value:
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
    const hipdnn_data_sdk::data_objects::TensorAttributes *hipDnnTensorAttr =
        opGraphWrapper.getTensorMap().at(uid);

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
  static bool
  isPassByValue(const hipdnn_data_sdk::data_objects::TensorAttributes *src) {
    return src->value_type() !=
           hipdnn_data_sdk::data_objects::TensorValue::NONE;
  }

  // Import tensor attrs (dims, strides, datatype) from hipDNN to fusilli.
  fusilli::ErrorObject
  importAttrs(fusilli::TensorAttr &dest,
              const hipdnn_data_sdk::data_objects::TensorAttributes *src) {
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
