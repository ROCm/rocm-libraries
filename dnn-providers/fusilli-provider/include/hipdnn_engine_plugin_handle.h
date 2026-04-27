// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the fusilli plugin's definition of
// HipdnnEnginePluginHandle. To hipDNN this type is opaque, it deals in
// hipdnnEnginePluginHandle_t which is a pointer to the undefined
// HipdnnEnginePluginHandle. Each plugin must define HipdnnEnginePluginHandle in
// order to create something when hipDNN asks for an plugin handle.
//
// HipdnnEnginePluginHandle stores any persistent data associated with a
// particular engine plugin. In fusilli plugin that's the fusilli::Handle, and
// some temporary buffers that higher level APIs create and destroy at different
// times.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_PLUGIN_SRC_HIPDNN_ENGINE_PLUGIN_HANDLE_H
#define FUSILLI_PLUGIN_SRC_HIPDNN_ENGINE_PLUGIN_HANDLE_H

#include <flatbuffers/detached_buffer.h>
#include <fusilli.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>

struct HipdnnEnginePluginHandle {
public:
  const int deviceId;

  HipdnnEnginePluginHandle(int deviceId) : deviceId(deviceId) {}

  // Take ownership of a flatbuffers::DetachedBuffer and store it associated
  // with its memory address.
  void storeEngineDetailsBuffer(
      const void *ptr, std::unique_ptr<flatbuffers::DetachedBuffer> &&buffer) {
    _engineDetailsBuffers[ptr] = std::move(buffer);
  }

  // Destroy the flatbuffers::DetachedBuffer associated with ptr.
  void eraseEngineDetailsBuffer(const void *ptr) {
    _engineDetailsBuffers.erase(ptr);
  }

  // Get or create fusilli::Handle just in time. As the engine API may set the
  // stream (through `hipdnnEnginePluginSetStream`) after initial handle
  // creation (in `hipdnnEnginePluginCreate`) we defer the fusilli::Handle
  // creation until we know if a stream has been set.
  fusilli::ErrorOr<std::reference_wrapper<fusilli::Handle>> getFusilliHandle() {
    if (!_fusilliHandle.has_value()) {
      FUSILLI_ASSIGN_OR_RETURN(
          auto handle,
          fusilli::Handle::create(fusilli::Backend::AMDGPU, deviceId,
                                  reinterpret_cast<uintptr_t>(_stream)));
      _fusilliHandle = std::move(handle);
    }
    return fusilli::ok(
        std::reference_wrapper<fusilli::Handle>(*_fusilliHandle));
  }

  void setStream(hipStream_t stream) { _stream = stream; }

private:
  // Default to creating a handle on the null (default) stream.
  hipStream_t _stream = 0;

  // Fusilli handle, will be created on the first call to `getFusilliHandle`.
  std::optional<fusilli::Handle> _fusilliHandle;

  // Storage for engine details.
  std::unordered_map<const void *, std::unique_ptr<flatbuffers::DetachedBuffer>>
      _engineDetailsBuffers;
};

#endif // FUSILLI_PLUGIN_SRC_HIPDNN_ENGINE_PLUGIN_HANDLE_H
