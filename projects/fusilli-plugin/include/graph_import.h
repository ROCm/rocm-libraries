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
#include <hipdnn_sdk/data_objects/data_types_generated.h>
#include <hipdnn_sdk/data_objects/tensor_attributes_generated.h>
#include <hipdnn_sdk/plugin/PluginApiDataTypes.h>
#include <hipdnn_sdk/plugin/flatbuffer_utilities/GraphWrapper.hpp>

#include <format>
#include <memory>
#include <unordered_map>

#include "hipdnn_engine_plugin_execution_context.h"

// Convert from hipDNN DataType to fusilli DataType.
inline fusilli::ErrorOr<fusilli::DataType>
hipDnnDataTypeToFusilliDataType(hipdnn_sdk::data_objects::DataType hipdnnType) {
  switch (hipdnnType) {
  case hipdnn_sdk::data_objects::DataType::HALF:
    return ok(fusilli::DataType::Half);
  case hipdnn_sdk::data_objects::DataType::BFLOAT16:
    return ok(fusilli::DataType::BFloat16);
  case hipdnn_sdk::data_objects::DataType::FLOAT:
    return ok(fusilli::DataType::Float);
  case hipdnn_sdk::data_objects::DataType::DOUBLE:
    return ok(fusilli::DataType::Double);
  case hipdnn_sdk::data_objects::DataType::UINT8:
    return ok(fusilli::DataType::Uint8);
  case hipdnn_sdk::data_objects::DataType::INT32:
    return ok(fusilli::DataType::Int32);
  case hipdnn_sdk::data_objects::DataType::UNSET:
    return ok(fusilli::DataType::NotSet);
  default:
    return error(fusilli::ErrorCode::RuntimeFailure,
                 "Unknown type in hipdnn -> fusilli graph translation.");
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
// NOTE: input hipDNN graph .node()s may not be in topological order. There's
// plans for a topological sort method but that's not available yet. As we're
// only handling single-node graphs currently, that's not a problem.
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
  //
  // All tensors in this map should be non-virtual boundary tensors---a virtual
  // tensor is an internal intermediate---internal tensors will be tracked
  // elsewhere. Currently we only support single node graphs, so there are no
  // virtual tensors to track.
  std::unordered_map<int64_t, std::shared_ptr<fusilli::TensorAttr>>
      uidToIOTensor;

  // Helper class for reading from flatbuffer.
  hipdnn_plugin::GraphWrapper opGraphWrapper;

  GraphImport(const hipdnnPluginConstData_t *opGraph)
      : opGraphWrapper(opGraph->ptr, opGraph->size) {}

  fusilli::ErrorObject importGraph() {
    const hipdnn_sdk::data_objects::Graph &hipDnnGraph =
        opGraphWrapper.getGraph();

    // Import graph level properties.
    fusilliGraph.setName(hipDnnGraph.name()->str())
        .setIODataType(
            FUSILLI_TRY(hipDnnDataTypeToFusilliDataType(hipDnnGraph.io_type())))
        .setIntermediateDataType(FUSILLI_TRY(
            hipDnnDataTypeToFusilliDataType(hipDnnGraph.intermediate_type())))
        .setComputeDataType(FUSILLI_TRY(
            hipDnnDataTypeToFusilliDataType(hipDnnGraph.compute_type())));

    return importNodes();
  }

  // Import all graph nodes.
  fusilli::ErrorObject importNodes() {
    if (opGraphWrapper.nodeCount() > 1)
      return fusilli::error(fusilli::ErrorCode::NotImplemented,
                            "Multi-node graphs not supported currently.");
    for (size_t i = 0; i < opGraphWrapper.nodeCount(); ++i) {
      const hipdnn_sdk::data_objects::Node &node = opGraphWrapper.getNode(i);
      FUSILLI_CHECK_ERROR(importNode(node));
    }

    return fusilli::ok();
  }

  // Import single graph node.
  fusilli::ErrorObject importNode(const hipdnn_sdk::data_objects::Node &node) {
    switch (node.attributes_type()) {
    case hipdnn_sdk::data_objects::NodeAttributes::ConvolutionFwdAttributes:
      FUSILLI_CHECK_ERROR(
          importConvFPropAttr(node.attributes_as_ConvolutionFwdAttributes()));
      break;
    default:
      return fusilli::error(fusilli::ErrorCode::NotImplemented,
                            "Unsupported node type.");
    }
    return fusilli::ok();
  }

  fusilli::ErrorObject
  importConvFPropAttr(const hipdnn_sdk::data_objects::ConvolutionFwdAttributes
                          *hipDnnConvFwdAttr) {
    // Import node inputs.
    std::shared_ptr<fusilli::TensorAttr> x =
        FUSILLI_TRY(importNodeInput(hipDnnConvFwdAttr->x_tensor_uid(), "x"));
    std::shared_ptr<fusilli::TensorAttr> w =
        FUSILLI_TRY(importNodeInput(hipDnnConvFwdAttr->w_tensor_uid(), "w"));

    // hipdnnEnginePluginGetApplicableEngineIds should have already eliminated
    // any nodes with asymmetric padding, this is just a double check.
    if (!std::ranges::equal(*hipDnnConvFwdAttr->pre_padding(),
                            *hipDnnConvFwdAttr->post_padding())) // C++ 20
      return fusilli::error(fusilli::ErrorCode::AttributeNotSet,
                            "Conv node with asymmetric padding found.");
    // Import node.
    auto fusilliConvFwdAttr =
        fusilli::ConvFPropAttr()
            .setPadding(*hipDnnConvFwdAttr->post_padding())
            .setStride(*hipDnnConvFwdAttr->stride())
            .setDilation(*hipDnnConvFwdAttr->dilation())
            .setName("conv_fprop");
    std::shared_ptr<fusilli::TensorAttr> y =
        fusilliGraph.convFProp(x, w, fusilliConvFwdAttr);

    // Import node output.
    FUSILLI_CHECK_ERROR(
        importNodeOutput(hipDnnConvFwdAttr->y_tensor_uid(), "y", y));

    return fusilli::ok();
  }

  // Import, and track, node input tensor. Node input tensor is created in the
  // case of a boundary tensor, and read from shared state otherwise.
  fusilli::ErrorOr<std::shared_ptr<fusilli::TensorAttr>>
  importNodeInput(int64_t uid, const char *name) {
    // Get hipDNN tensor. TensorMap is created from the graph that uid variable
    // is read from, so .at() call should be safe.
    const hipdnn_sdk::data_objects::TensorAttributes *hipDnnTensorAttr =
        opGraphWrapper.getTensorMap().at(uid);

    // A virtual node indicates a non-boundary node.
    if (hipDnnTensorAttr->virtual_()) {
      // When multi-op graphs are supported, we would look up the output of a
      // previously imported node and return it here.
      return fusilli::error(fusilli::ErrorCode::NotImplemented,
                            "Virtual inputs currently unsupported.");
    }

    // Import new tensor.
    auto fusilliTensorAttr = fusilli::TensorAttr().setName(
        std::format("{}_{}", name, uid)); // C++ 20
    FUSILLI_CHECK_ERROR(importAttrs(fusilliTensorAttr, hipDnnTensorAttr));
    std::shared_ptr<fusilli::TensorAttr> graphInput =
        fusilliGraph.tensor(fusilliTensorAttr);

    // Track boundary tensor.
    uidToIOTensor[uid] = graphInput;

    return ok(graphInput);
  };

  // Import and track node output tensor.
  fusilli::ErrorObject
  importNodeOutput(int64_t uid, const char *name,
                   const std::shared_ptr<fusilli::TensorAttr> &nodeOutput) {
    // Get hipDNN tensor. TensorMap is created from the graph that uid variable
    // is read from, so .at() call should be safe.
    const hipdnn_sdk::data_objects::TensorAttributes *hipDnnTensorAttr =
        opGraphWrapper.getTensorMap().at(uid);

    // Import attrs.
    nodeOutput->setName(std::format("{}_{}", name, uid)); // C++ 20
    FUSILLI_CHECK_ERROR(importAttrs(*nodeOutput, hipDnnTensorAttr));

    // A virtual node indicates a non-boundary node.
    if (hipDnnTensorAttr->virtual_()) {
      // This tensor is an input to nodes farther down the topological sort.
      // When multi-op graphs are supported, we would track UID -> node output
      // tensor in a non-IO tensor map.
      return fusilli::error(fusilli::ErrorCode::NotImplemented,
                            "Virtual outputs currently unsupported. An output "
                            "tensor may have not been marked as an output.");
    }

    // Track boundary node.
    uidToIOTensor[uid] = nodeOutput;

    return fusilli::ok();
  };

  // Import all tensor attrs src -> dest.
  fusilli::ErrorObject
  importAttrs(fusilli::TensorAttr &dest,
              const hipdnn_sdk::data_objects::TensorAttributes *src) {
    dest.setIsVirtual(src->virtual_())
        .setDim(*src->dims())
        .setStride(*src->strides())
        .setDataType(
            FUSILLI_TRY(hipDnnDataTypeToFusilliDataType(src->data_type())));
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
  return HipdnnEnginePluginExecutionContext{.graph = std::move(gc.fusilliGraph),
                                            .uidToFusilliTensorAttr =
                                                std::move(gc.uidToIOTensor)};
}

#endif // FUSILLI_PLUGIN_SRC_GRAPH_IMPORT_H
