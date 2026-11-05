// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <hip/hip_runtime.h>

#include <hipdnn_data_sdk/utilities/Tensor.hpp>

namespace sdpa_case_flow
{

// Device copies of the tensors whose data starts past an aligned allocation start. Each copy
// starts `offset` bytes past a hipMalloc base, and the variant pack entry points at that start.
// copyOutputsBack returns the outputs that a run wrote to the copies to the original buffers.
class ShiftedStarts
{
public:
    ShiftedStarts(
        const std::unordered_map<int64_t, std::shared_ptr<hipdnn_data_sdk::utilities::ITensor>>&
            tensors,
        const std::map<int64_t, int64_t>& offsetBytes,
        std::unordered_map<int64_t, void*>& variantPack)
    {
        try
        {
            for(const auto& [uid, offset] : offsetBytes)
            {
                addCopy(*tensors.at(uid), uid, offset, variantPack);
            }
        }
        catch(...)
        {
            for(const auto& copy : _copies)
            {
                variantPack[copy.uid] = copy.original;
            }
            release();
            throw;
        }
    }

    ShiftedStarts(const ShiftedStarts&) = delete;
    ShiftedStarts& operator=(const ShiftedStarts&) = delete;

    ~ShiftedStarts()
    {
        release();
    }

    void copyOutputsBack(const std::vector<int64_t>& outputUids) const
    {
        for(const auto& copy : _copies)
        {
            if(std::find(outputUids.begin(), outputUids.end(), copy.uid) == outputUids.end())
            {
                continue;
            }
            if(hipMemcpy(copy.original, copy.shifted, copy.bytes, hipMemcpyDeviceToDevice)
               != hipSuccess)
            {
                throw std::runtime_error("hipMemcpy failed for a shifted output");
            }
        }
    }

private:
    struct Copy
    {
        int64_t uid = 0;
        size_t bytes = 0;
        void* original = nullptr;
        void* base = nullptr;
        void* shifted = nullptr;
    };

    void addCopy(hipdnn_data_sdk::utilities::ITensor& tensor,
                 int64_t uid,
                 int64_t offset,
                 std::unordered_map<int64_t, void*>& variantPack)
    {
        if(offset <= 0)
        {
            throw std::logic_error("shifted data start of tensor " + std::to_string(uid)
                                   + " must be a positive byte count");
        }
        Copy copy;
        copy.uid = uid;
        copy.bytes = tensor.elementSpace() * tensor.elementSize();
        copy.original = variantPack.at(uid);
        if(hipMalloc(&copy.base, copy.bytes + static_cast<size_t>(offset)) != hipSuccess)
        {
            throw std::runtime_error("hipMalloc failed for a shifted data start");
        }
        _copies.push_back(copy);
        void* shifted = static_cast<char*>(copy.base) + offset;
        if(hipMemcpy(shifted, copy.original, copy.bytes, hipMemcpyDeviceToDevice) != hipSuccess)
        {
            throw std::runtime_error("hipMemcpy failed for a shifted data start");
        }
        _copies.back().shifted = shifted;
        variantPack[uid] = shifted;
    }

    void release() noexcept
    {
        for(const auto& copy : _copies)
        {
            static_cast<void>(hipFree(copy.base));
        }
        _copies.clear();
    }

    std::vector<Copy> _copies;
};

} // namespace sdpa_case_flow
