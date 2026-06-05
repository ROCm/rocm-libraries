// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/flatbuffer_utilities/GraphWrapper.hpp>

#include <memory>
#include <string>
#include <utility>

struct GraphTest
{
    std::shared_ptr<flatbuffers::DetachedBuffer> buffer;
    std::string message;

    GraphTest(flatbuffers::FlatBufferBuilder&& builder, std::string inMessage)
        : buffer(std::make_shared<flatbuffers::DetachedBuffer>(builder.Release()))
        , message(std::move(inMessage))
    {
    }

    hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper graphWrapper() const
    {
        return hipdnn_flatbuffers_sdk::flatbuffer_utilities::GraphWrapper(buffer->data(),
                                                                          buffer->size());
    }
};
