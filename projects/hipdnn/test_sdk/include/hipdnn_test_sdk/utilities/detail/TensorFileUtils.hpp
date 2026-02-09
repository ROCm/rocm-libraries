// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <filesystem>
#include <fstream>
#include <hipdnn_data_sdk/utilities/Tensor.hpp>
#include <vector>

namespace hipdnn_test_sdk::detail
{

template <class T>
struct DatatypeFromTensor
{
};

template <class T>
struct DatatypeFromTensor<hipdnn_data_sdk::utilities::Tensor<T>>
{
    using Type = T;
};

inline void fillTensorFromFile(hipdnn_data_sdk::utilities::ITensor& tensor,
                               const std::filesystem::path& path)
{
    std::ifstream fileInputStream(path, std::ios::binary);
    if(!fileInputStream)
    {
        throw std::runtime_error("Error: could not load tensor " + path.string());
    }

    auto vec = std::vector<unsigned char>(std::istreambuf_iterator<char>(fileInputStream),
                                          std::istreambuf_iterator<char>{});

    tensor.fillWithData(vec.data(), vec.size());
}

} // namespace hipdnn_test_sdk::detail
