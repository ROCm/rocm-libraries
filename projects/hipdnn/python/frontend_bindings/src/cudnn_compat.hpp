// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <hipdnn_backend.h>
#include <hipdnn_frontend/Error.hpp>
#include <hipdnn_frontend/Types.hpp>
#include <nanobind/nanobind.h>

#include <optional>

namespace hipdnn_python
{

// Raises NotImplementedError for a cuDNN parameter hipDNN cannot honour. The
// cuDNN-shaped bindings keep such parameters in their cuDNN position so
// positional calls line up, and accept only cuDNN's default value.
[[noreturn]] void throwUnsupportedParameter(const char* parameter, const char* accepted);

// Raises the Python exception cuDNN's Python frontend raises for this error
// code; does nothing for a good Error. Only the cuDNN-semantics surface
// (pygraph) raises; hipdnn.Graph keeps returning Error objects.
void throwIfBad(const hipdnn_frontend::Error& error);

// Resolves a handle argument: None maps to nullptr, anything else is an object
// whose get() returns the raw handle as an integer (hipdnn.Handle, or a
// duck-typed stand-in).
hipdnnHandle_t rawHandleOrNull(const nanobind::handle& handle);

// Maps a torch or numpy dtype object (torch.float16, numpy.dtype("float32"),
// numpy.float32) to a DataType, as cuDNN's Python frontend does for
// data_type arguments. Returns nullopt for anything that is not a torch or
// numpy dtype, so overloads can fall back with nb::next_overload; raises
// ValueError for a torch/numpy dtype hipDNN has no DataType for.
std::optional<hipdnn_frontend::DataType> dataTypeFromDtypeObject(const nanobind::handle& dtype);

// Registers hipdnn.GraphNotSupportedError, the counterpart of
// cudnn.cudnnGraphNotSupportedError. Must run before any binding can raise it.
void compatBindings(nanobind::module_& m);

} // namespace hipdnn_python
