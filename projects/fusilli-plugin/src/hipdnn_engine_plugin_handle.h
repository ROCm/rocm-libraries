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

#include "fusilli/backend/handle.h"
#include <flatbuffers/detached_buffer.h>
#include <memory>
#include <unordered_map>

struct HipdnnEnginePluginHandle {
public:
  fusilli::Handle fusilliHandle;

  HipdnnEnginePluginHandle(fusilli::Handle &&handle)
      : fusilliHandle(std::move(handle)) {}

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

private:
  std::unordered_map<const void *, std::unique_ptr<flatbuffers::DetachedBuffer>>
      _engineDetailsBuffers;
};

#endif // FUSILLI_PLUGIN_SRC_HIPDNN_ENGINE_PLUGIN_HANDLE_H
