// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file GraphAttributes.hpp
 * @brief Graph-level configuration attributes for hipDNN computational graphs
 *
 * This file defines the GraphAttributes class which holds configuration settings
 * that apply to an entire computational graph, such as default data types.
 */

#pragma once

#include <hipdnn_frontend/Types.hpp>
#include <memory>
#include <string>
#include <unordered_map>

namespace hipdnn_frontend::graph
{

/**
 * @class GraphAttributes
 * @brief Configuration settings that apply to an entire computational graph
 *
 * GraphAttributes holds default settings for a graph, including:
 * - **compute_data_type**: The data type used for computations (e.g., FLOAT for accumulation)
 * - **intermediate_data_type**: The data type for intermediate/virtual tensors
 * - **io_data_type**: The data type for input/output tensors
 *
 * These settings are inherited by tensors and operations that don't explicitly
 * specify their own data types.
 *
 * @code{.cpp}
 * Graph graph;
 * graph.set_io_data_type(DataType::HALF)
 *      .set_compute_data_type(DataType::FLOAT)
 *      .set_intermediate_data_type(DataType::HALF)
 *      .set_name("my_graph");
 * @endcode
 */
class GraphAttributes
{
public:
    /**
     * @brief Get the name of the graph
     * @return The graph name
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    const std::string& get_name() const
    {
        return _name;
    }
    /**
     * @brief Get the compute data type
     * @return The data type used for computations/accumulation
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    DataType get_compute_data_type() const
    {
        return _computeType;
    }

    /**
     * @brief Get the intermediate data type
     * @return The data type for virtual/intermediate tensors
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    DataType get_intermediate_data_type() const
    {
        return _intermediateType;
    }

    /**
     * @brief Get the I/O data type
     * @return The data type for input/output tensors
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    DataType get_io_data_type() const
    {
        return _ioType;
    }

    /**
     * @brief Set the compute data type
     * @param computeType The data type for computations/accumulation
     * @return Reference to this for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    GraphAttributes& set_compute_data_type(DataType computeType)
    {
        _computeType = computeType;
        return *this;
    }

    /**
     * @brief Set the intermediate data type
     * @param intermediateType The data type for virtual/intermediate tensors
     * @return Reference to this for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    GraphAttributes& set_intermediate_data_type(DataType intermediateType)
    {
        _intermediateType = intermediateType;
        return *this;
    }

    /**
     * @brief Set the I/O data type
     * @param ioType The data type for input/output tensors
     * @return Reference to this for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    GraphAttributes& set_io_data_type(DataType ioType)
    {
        _ioType = ioType;
        return *this;
    }

    /**
     * @brief Set the name of the graph
     * @param name The graph name (for debugging and logging)
     * @return Reference to this for method chaining
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    GraphAttributes& set_name(const std::string& name)
    {
        _name = name;
        return *this;
    }

    /**
     * @brief Fill unset properties from another GraphAttributes object
     * @param other The source to copy missing properties from
     * @return Reference to this for method chaining
     *
     * Only properties that are NOT_SET or empty will be copied from other.
     */
    // NOLINTNEXTLINE(readability-identifier-naming)
    GraphAttributes& fill_missing_properties(const GraphAttributes& other)
    {
        if(_computeType == DataType::NOT_SET)
        {
            _computeType = other._computeType;
        }

        if(_intermediateType == DataType::NOT_SET)
        {
            _intermediateType = other._intermediateType;
        }

        if(_ioType == DataType::NOT_SET)
        {
            _ioType = other._ioType;
        }

        if(_name.empty())
        {
            _name = other._name;
        }

        return *this;
    }

private:
    std::string _name; ///< Graph name for debugging
    DataType _computeType = DataType::NOT_SET; ///< Compute/accumulation data type
    DataType _intermediateType = DataType::NOT_SET; ///< Intermediate tensor data type
    DataType _ioType = DataType::NOT_SET; ///< Input/output tensor data type
};

typedef GraphAttributes Context; ///< @brief Type alias for GraphAttributes
} // namespace hipdnn_frontend::graph
