// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "bindings.hpp"

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

struct EngineInfo
{
    int64_t engineId;
    std::string engineName;
    std::string pluginName;
    std::string version;
    std::string type;
};

class HandleWrapper
{
private:
    HipdnnHandlePtr _handle;

    void checkNotDestroyed() const
    {
        if(!_handle)
        {
            throw std::runtime_error("Handle has been destroyed");
        }
    }

public:
    HandleWrapper()
    {
        const auto error = createHipdnnHandle(_handle);
        if(error.is_bad())
        {
            throw std::runtime_error("Failed to create hipdnn handle: " + error.get_message());
        }
    }

    explicit HandleWrapper(uintptr_t streamPtr)
    {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        const auto error = createHipdnnHandle(_handle, reinterpret_cast<hipStream_t>(streamPtr));
        if(error.is_bad())
        {
            throw std::runtime_error("Failed to create hipdnn handle: " + error.get_message());
        }
    }

    hipdnnHandle_t get() const
    {
        checkNotDestroyed();
        return *_handle;
    }

    bool isValid() const
    {
        return _handle != nullptr;
    }

    void destroy()
    {
        _handle.reset();
    }

    void setStream(uintptr_t streamPtr)
    {
        checkNotDestroyed();
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        const auto error = setHipdnnHandleStream(_handle, reinterpret_cast<hipStream_t>(streamPtr));
        if(error.is_bad())
        {
            throw std::runtime_error("Failed to set stream on hipdnn handle: "
                                     + error.get_message());
        }
    }

    uintptr_t getStream() const
    {
        checkNotDestroyed();
        hipStream_t stream = nullptr;
        const auto error = getHipdnnHandleStream(_handle, &stream);
        if(error.is_bad())
        {
            throw std::runtime_error("Failed to get stream from hipdnn handle: "
                                     + error.get_message());
        }
        return reinterpret_cast<uintptr_t>(stream);
    }
    std::string engineIdToName(int64_t engineId) const
    {
        checkNotDestroyed();
        size_t engineNameLen = 0;
        if(hipdnnGetEngineNameById_ext(get(), engineId, nullptr, &engineNameLen)
           != HIPDNN_STATUS_SUCCESS)
        {
            throw std::out_of_range("Engine ID is not loaded");
        }
        std::vector<char> engineName(engineNameLen);
        if(hipdnnGetEngineNameById_ext(get(), engineId, engineName.data(), &engineNameLen)
           != HIPDNN_STATUS_SUCCESS)
        {
            throw std::runtime_error("Failed to resolve engine name");
        }
        return {engineName.data()};
    }

    EngineInfo getEngineInfo(int64_t engineId) const
    {
        checkNotDestroyed();
        size_t count = 0;
        if(hipdnnGetEngineCount_ext(get(), &count) != HIPDNN_STATUS_SUCCESS)
        {
            throw std::runtime_error("Failed to query loaded engine count");
        }
        for(size_t index = 0; index < count; ++index)
        {
            int64_t candidateId = 0;
            size_t engineNameLen = 0;
            size_t pluginNameLen = 0;
            size_t versionLen = 0;
            size_t typeLen = 0;
            if(hipdnnGetEngineInfo_ext(get(),
                                       index,
                                       &candidateId,
                                       nullptr,
                                       &engineNameLen,
                                       nullptr,
                                       &pluginNameLen,
                                       nullptr,
                                       &versionLen,
                                       nullptr,
                                       &typeLen)
               != HIPDNN_STATUS_SUCCESS)
            {
                throw std::runtime_error("Failed to query loaded engine metadata sizes");
            }
            if(candidateId != engineId)
            {
                continue;
            }
            std::vector<char> engineName(engineNameLen);
            std::vector<char> pluginName(pluginNameLen);
            std::vector<char> version(versionLen);
            std::vector<char> type(typeLen);
            if(hipdnnGetEngineInfo_ext(get(),
                                       index,
                                       &candidateId,
                                       engineName.data(),
                                       &engineNameLen,
                                       pluginName.data(),
                                       &pluginNameLen,
                                       version.data(),
                                       &versionLen,
                                       type.data(),
                                       &typeLen)
               != HIPDNN_STATUS_SUCCESS)
            {
                throw std::runtime_error("Failed to query loaded engine metadata");
            }

            return {candidateId,
                    std::string(engineName.data()),
                    std::string(pluginName.data()),
                    std::string(version.data()),
                    std::string(type.data())};
        }
        throw std::out_of_range("Engine ID is not loaded");
    }
};

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
