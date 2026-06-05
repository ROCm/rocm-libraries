// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

/**
 * @file hipdnn_frontend.hpp
 * @brief Main entry point for the hipDNN Frontend C++ API
 *
 * Include this header (`#include <hipdnn_frontend.hpp>`) to access the hipDNN
 * frontend API. It provides a high-level C++ interface for building and executing
 * deep learning computational graphs on AMD GPUs.
 *
 * @section overview Overview
 *
 * The hipDNN Frontend provides a graph-based API for constructing deep learning
 * operations including:
 * - Convolution operations (forward, data gradient, weight gradient)
 * - Batch normalization (forward, backward, inference)
 * - Pointwise/element-wise operations
 * - Matrix multiplication
 *
 * @section usage Basic Usage
 *
 * @code{.cpp}
 * #include <hipdnn_frontend.hpp>
 *
 * using namespace hipdnn_frontend;
 * using namespace hipdnn_frontend::graph;
 *
 * // Create a graph
 * Graph graph;
 * graph.set_io_data_type(DataType::HALF)
 *      .set_compute_data_type(DataType::FLOAT);
 *
 * // Create tensors
 * auto x = Graph::tensor(TensorAttributes()
 *              .set_dim({1, 64, 28, 28})
 *              .set_stride({50176, 784, 28, 1})
 *              .set_data_type(DataType::HALF));
 *
 * // Add operations and build the graph
 * // ...
 *
 * hipdnnHandle_t handle;
 * hipdnnCreate(&handle);
 * graph.build(handle);
 *
 * // Execute
 * graph.execute(handle, tensorLookup, workspace);
 * @endcode
 *
 * @see hipdnn_frontend::graph::Graph The main Graph class for building operations
 * @see hipdnn_frontend::DataType Supported data types
 * @see hipdnn_frontend::Error Error handling types
 */

#pragma once

#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Handle.hpp>
#include <hipdnn_frontend/HeuristicPolicyInfo.hpp>
#include <hipdnn_frontend/PluginPaths.hpp>
#include <hipdnn_frontend/Types.hpp>
