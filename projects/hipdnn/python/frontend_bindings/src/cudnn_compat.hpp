// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_backend.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <nanobind/nanobind.h>

#include <cstdint>
#include <optional>
#include <vector>

namespace hipdnn_python
{

// Raises NotImplementedError for a cuDNN parameter hipDNN cannot honour. The
// cuDNN-shaped bindings keep such parameters in their cuDNN position so
// positional calls line up, and accept only one value: cuDNN's default where
// hipDNN can express it, otherwise None.
[[noreturn]] void throwUnsupportedParameter(const char* parameter, const char* accepted);

// throwUnsupportedParameter(parameter, "None") unless value is None.
void requireNone(const nanobind::handle& value, const char* parameter);

// Raises the Python exception cuDNN's Python frontend raises for this error
// code; does nothing for a good Error. Only the cuDNN-semantics surface
// (pygraph) raises; hipdnn.Graph keeps returning Error objects.
void throwIfBad(const hipdnn_frontend::Error& error);

// Resolves a handle argument: None maps to nullptr, anything else is an object
// whose get() returns the raw handle as an integer (hipdnn.Handle, or a
// duck-typed stand-in).
hipdnnHandle_t rawHandleOrNull(const nanobind::handle& handle);

// None maps to nullptr; anything else goes through toDataPointer().
void* workspacePointer(const nanobind::handle& workspace);

std::vector<uint8_t> bytesToVector(const nanobind::bytes& data);

// Maps a hipdnn.DataType or a torch or numpy dtype object (torch.float16,
// numpy.dtype("float32"), numpy.float32) to a DataType, as cuDNN's Python
// frontend does for data_type arguments. Returns nullopt for anything else, so
// callers choose the fallback (TypeError or nb::next_overload); raises
// ValueError for a torch/numpy dtype hipDNN has no DataType for.
std::optional<hipdnn_frontend::DataType> dataTypeFromObject(const nanobind::handle& value);

// Registers hipdnn.GraphNotSupportedError, the counterpart of
// cudnn.cudnnGraphNotSupportedError. Must run before any binding can raise it.
void compatBindings(nanobind::module_& m);

} // namespace hipdnn_python
