// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "bindings.hpp"
#include "handle_wrapper.hpp"

#include <hipdnn_backend.h>
#include <hipdnn_frontend.hpp>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;
using namespace hipdnn_frontend;

void handleBindings(nb::module_& m)
{
    nb::class_<EngineInfo>(m, "EngineInfo")
        .def_ro("engine_id", &EngineInfo::engineId)
        .def_ro("engine_name", &EngineInfo::engineName)
        .def_ro("plugin_name", &EngineInfo::pluginName)
        .def_ro("version", &EngineInfo::version)
        .def_ro("type", &EngineInfo::type);

    nb::class_<HandleWrapper>(m, "Handle")
        .def(nb::init<>(), "Create a new hipdnn handle")
        .def(nb::init<uintptr_t>(), nb::arg("stream"), "Create a handle with a stream")
        .def(
            "get",
            [](const HandleWrapper& h) { return reinterpret_cast<uintptr_t>(h.get()); },
            "Get the handle pointer as an integer")
        .def("set_stream",
             &HandleWrapper::setStream,
             nb::arg("stream"),
             "Set the HIP stream (as integer pointer)")
        .def("get_stream", &HandleWrapper::getStream, "Get the HIP stream (as integer pointer)")
        .def("get_engine_info",
             &HandleWrapper::getEngineInfo,
             nb::arg("engine_id"),
             "Return metadata for a loaded engine ID")
        .def("engine_id_to_name",
             &HandleWrapper::engineIdToName,
             nb::arg("engine_id"),
             "Resolve a loaded engine ID to the name that engine carries, including "
             "plugin-supplied engines absent from the built-in registry.\n\n"
             "Raises IndexError if no loaded engine carries the ID.")
        .def("__int__", [](const HandleWrapper& h) { return reinterpret_cast<uintptr_t>(h.get()); })
        .def("__index__",
             [](const HandleWrapper& h) { return reinterpret_cast<uintptr_t>(h.get()); })
        .def("__repr__", [](const HandleWrapper& h) {
            if(!h.isValid())
            {
                return std::string("<hipdnn_frontend.Handle (destroyed)>");
            }
            return "<hipdnn_frontend.Handle at "
                   + std::to_string(reinterpret_cast<uintptr_t>(h.get())) + ">";
        });

    m.def(
        "create_handle",
        []() { return std::make_shared<HandleWrapper>(); },
        "Create a new hipdnn handle");
    m.def(
        "create_handle",
        [](uintptr_t stream) { return std::make_shared<HandleWrapper>(stream); },
        nb::arg("stream"),
        "Create a new hipdnn handle with a stream");
    m.def(
        "destroy_handle",
        [](HandleWrapper& h) { h.destroy(); },
        nb::arg("handle"),
        "Destroy a hipdnn handle. The handle object should not be used after this call.");
    m.def(
        "set_stream",
        [](HandleWrapper& h, uintptr_t stream) { h.setStream(stream); },
        nb::arg("handle"),
        nb::arg("stream"),
        "Set the HIP stream on a handle");
    m.def(
        "get_stream",
        [](const HandleWrapper& h) { return h.getStream(); },
        nb::arg("handle"),
        "Get the HIP stream from a handle");
}
