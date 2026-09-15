// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file Error.hpp
 * @brief Error handling types and macros for the hipDNN Frontend API
 *
 * This file defines the error handling infrastructure used throughout the hipDNN
 * frontend. It includes error codes, the Error struct for detailed error information,
 * and helper macros for error checking and propagation.
 */

#pragma once

#include <ostream>
#include <string>
#include <vector>

/**
 * @brief Check an Error return value and propagate if an error occurred
 * @param x Expression returning an Error object
 *
 * This macro evaluates the expression and returns immediately if the result
 * indicates an error (is_bad() returns true).
 */
#define HIPDNN_CHECK_ERROR(x) \
    do                        \
    {                         \
        const auto err = x;   \
        if(err.is_bad())      \
        {                     \
            return err;       \
        }                     \
    } while(0)

namespace hipdnn_frontend
{
/**
 * @enum ErrorCode
 * @brief Error codes returned by hipDNN Frontend operations
 *
 * Portions derived from NVIDIA cuDNN frontend, used under the MIT license.
 */
enum class ErrorCode
{
    OK, ///< Operation completed successfully
    INVALID_VALUE, ///< An invalid value was provided
    HIPDNN_BACKEND_ERROR, ///< An error occurred in the hipDNN backend
    ATTRIBUTE_NOT_SET, ///< A required attribute was not set
    /**
     * @brief No engine accepted this graph
     *
     * No engine has an applicable solution on the current device, likely
     * because the requested dtype/shape/operation pattern is not supported by
     * any engine. Distinct from validation errors -- the graph is well-formed,
     * just not runnable by the available engines.
     */
    GRAPH_NOT_SUPPORTED, ///< No engine accepted this graph (graph well-formed but unrunnable on available engines)
    SHAPE_DEDUCTION_FAILED, ///< Tensor shape/stride deduction failed
    INVALID_TENSOR_NAME, ///< A referenced tensor name is invalid
    INVALID_VARIANT_PACK, ///< The variant pack provided to execute() is invalid
    GRAPH_EXECUTION_PLAN_CREATION_FAILED, ///< Creating an execution plan for the graph failed
    GRAPH_EXECUTION_FAILED, ///< Executing the graph failed
    HEURISTIC_QUERY_FAILED, ///< Querying heuristics for the graph failed
    UNSUPPORTED_GRAPH_FORMAT, ///< The graph format is unsupported
    CUDA_API_FAILED, ///< A CUDA runtime API call failed (CUDA-specific; not produced by hipDNN)
    CUDNN_BACKEND_API_FAILED, ///< A cuDNN backend API call failed (CUDA-specific; not produced by hipDNN)
    INVALID_CUDA_DEVICE, ///< The CUDA device is invalid (CUDA-specific; not produced by hipDNN)
    HANDLE_ERROR, ///< A handle-related error occurred
    NVRTC_COMPILATION_FAILED ///< Runtime kernel compilation via NVRTC failed (CUDA-specific; not produced by hipDNN)
};

// NOLINTNEXTLINE(readability-identifier-naming)
inline std::string to_string(ErrorCode code)
{
    switch(code)
    {
    case ErrorCode::OK:
        return "OK";
    case ErrorCode::INVALID_VALUE:
        return "INVALID_VALUE";
    case ErrorCode::HIPDNN_BACKEND_ERROR:
        return "HIPDNN_BACKEND_ERROR";
    case ErrorCode::ATTRIBUTE_NOT_SET:
        return "ATTRIBUTE_NOT_SET";
    case ErrorCode::GRAPH_NOT_SUPPORTED:
        return "GRAPH_NOT_SUPPORTED";
    case ErrorCode::SHAPE_DEDUCTION_FAILED:
        return "SHAPE_DEDUCTION_FAILED";
    case ErrorCode::INVALID_TENSOR_NAME:
        return "INVALID_TENSOR_NAME";
    case ErrorCode::INVALID_VARIANT_PACK:
        return "INVALID_VARIANT_PACK";
    case ErrorCode::GRAPH_EXECUTION_PLAN_CREATION_FAILED:
        return "GRAPH_EXECUTION_PLAN_CREATION_FAILED";
    case ErrorCode::GRAPH_EXECUTION_FAILED:
        return "GRAPH_EXECUTION_FAILED";
    case ErrorCode::HEURISTIC_QUERY_FAILED:
        return "HEURISTIC_QUERY_FAILED";
    case ErrorCode::UNSUPPORTED_GRAPH_FORMAT:
        return "UNSUPPORTED_GRAPH_FORMAT";
    case ErrorCode::CUDA_API_FAILED:
        return "CUDA_API_FAILED";
    case ErrorCode::CUDNN_BACKEND_API_FAILED:
        return "CUDNN_BACKEND_API_FAILED";
    case ErrorCode::INVALID_CUDA_DEVICE:
        return "INVALID_CUDA_DEVICE";
    case ErrorCode::HANDLE_ERROR:
        return "HANDLE_ERROR";
    case ErrorCode::NVRTC_COMPILATION_FAILED:
        return "NVRTC_COMPILATION_FAILED";
    default:
        return "UNKNOWN_ERROR";
    }
}

inline std::ostream& operator<<(std::ostream& os, const ErrorCode& error)
{
    return os << to_string(error);
}

typedef ErrorCode error_code_t; ///< @brief Type alias for ErrorCode

/**
 * @struct Error
 * @brief Represents an error with a code and descriptive message
 *
 * The Error struct is the primary mechanism for reporting errors in hipDNN Frontend
 * operations. It combines an ErrorCode with a human-readable message describing
 * what went wrong.
 *
 * @code{.cpp}
 * Error result = graph.build(handle);
 * if(result.is_bad())
 * {
 *     std::cerr << "Build failed: " << result.get_message() << std::endl;
 * }
 * @endcode
 */
struct Error
{
    ErrorCode code; ///< The error code
    std::string err_msg; ///< Detailed error message  // NOLINT(readability-identifier-naming)

    /**
     * @brief Default constructor, creates a success result
     */
    Error()
        : code(ErrorCode::OK)
    {
    }

    /**
     * @brief Construct an Error with a specific code and message
     * @param errorCode The error code
     * @param message Descriptive message about the error
     */
    Error(ErrorCode errorCode, std::string message)
        : code(errorCode)
        , err_msg(std::move(message))
    {
    }

    /**
     * @brief Get the error message
     * @return The error message string
     */
    std::string get_message() const // NOLINT(readability-identifier-naming)
    {
        return err_msg;
    }

    /**
     * @brief Get the error code
     * @return The ErrorCode value
     */
    ErrorCode get_code() const // NOLINT(readability-identifier-naming)
    {
        return code;
    }

    /**
     * @brief Check if the operation succeeded
     * @return true if code == ErrorCode::OK, false otherwise
     */
    bool is_good() const // NOLINT(readability-identifier-naming)
    {
        return code == ErrorCode::OK;
    }

    /**
     * @brief Check if the operation failed
     * @return true if code != ErrorCode::OK, false otherwise
     */
    bool is_bad() const // NOLINT(readability-identifier-naming)
    {
        return code != ErrorCode::OK;
    }

    /// @brief Compare error code with an ErrorCode value
    bool operator==(ErrorCode otherCode) const
    {
        return code == otherCode;
    }
    /// @brief Compare error code with an ErrorCode value (inequality)
    bool operator!=(ErrorCode otherCode) const
    {
        return code != otherCode;
    }
    /// @brief Compare two Error objects by their error codes
    bool operator==(const Error& other) const
    {
        return code == other.code;
    }
    /// @brief Compare two Error objects by their error codes (inequality)
    bool operator!=(const Error& other) const
    {
        return code != other.code;
    }
};

inline std::ostream& operator<<(std::ostream& os, const Error& error)
{
    return os << "{" << error.code << ", " << error.get_message() << "}";
}

typedef Error error_object; ///< @brief Type alias for Error (compatibility)
typedef Error error_t; ///< @brief Type alias for Error

#define HIPDNN_RETURN_IF_NE(x, y, error_status, message) \
    do                                                   \
    {                                                    \
        if((x) != (y))                                   \
        {                                                \
            return {error_status, message};              \
        }                                                \
    } while(0)

#define HIPDNN_RETURN_IF_EQ(x, y, error_status, message) \
    do                                                   \
    {                                                    \
        if((x) == (y))                                   \
        {                                                \
            return {error_status, message};              \
        }                                                \
    } while(0)

#define HIPDNN_RETURN_IF_TRUE(x, error_status, message) \
    do                                                  \
    {                                                   \
        if(x)                                           \
        {                                               \
            return {error_status, message};             \
        }                                               \
    } while(0)

#define HIPDNN_RETURN_IF_FALSE(x, error_status, message) \
    do                                                   \
    {                                                    \
        if(!(x))                                         \
        {                                                \
            return {error_status, message};              \
        }                                                \
    } while(0)

#define HIPDNN_RETURN_IF_NULL(x, error_status, message) \
    do                                                  \
    {                                                   \
        if((x) == nullptr)                              \
        {                                               \
            return {error_status, message};             \
        }                                               \
    } while(0)

#define HIPDNN_RETURN_IF_LT(x, y, error_status, message) \
    do                                                   \
    {                                                    \
        if((x) < (y))                                    \
        {                                                \
            return {error_status, message};              \
        }                                                \
    } while(0)

#define HIPDNN_RETURN_IF_GE(x, y, error_status, message) \
    do                                                   \
    {                                                    \
        if((x) >= (y))                                   \
        {                                                \
            return {error_status, message};              \
        }                                                \
    } while(0)

#define HIPDNN_RETURN_IF_LE(x, y, error_status, message) \
    do                                                   \
    {                                                    \
        if((x) <= (y))                                   \
        {                                                \
            return {error_status, message};              \
        }                                                \
    } while(0)

/// Evaluate an expression that returns Error and early-return
/// \c {err, {}} when the error is bad. Works with any function whose
/// return type is \c std::pair<Error, T> because \c {} value-initialises
/// the second element (std::nullopt for optionals, default-constructed
/// pairs, etc.).
#define HIPDNN_FE_TRY(expr)           \
    do                                \
    {                                 \
        auto _fe_try_err = (expr);    \
        if(_fe_try_err.is_bad())      \
        {                             \
            return {_fe_try_err, {}}; \
        }                             \
    } while(0)
}
