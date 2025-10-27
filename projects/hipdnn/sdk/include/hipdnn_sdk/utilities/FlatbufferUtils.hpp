// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <flatbuffers/flatbuffers.h>
#include <vector>

namespace hipdnn_sdk::utilities
{

template <typename T>
inline std::vector<T> convertFlatBufferVectorToStdVector(const flatbuffers::Vector<T>* in)
{
    std::vector<T> out;

    if(in)
    {
        out.resize(in->size());
        for(::flatbuffers::uoffset_t i = 0; i < in->size(); i++)
        {
            out[i] = in->Get(i);
        }
    }

    return out;
}

}
