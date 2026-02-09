// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#include <ostream>
#include <vector>

namespace hipdnn_test_sdk::utilities
{

/**
 * @brief Wrapper for streaming std::vector to ostream
 *
 * This wrapper provides a safe way to log vectors without polluting
 * the global namespace with operator<< for std::vector.
 *
 * Usage: std::cout << StreamVec(myVector);
 * Usage: HIPDNN_LOG_INFO("dims: " << StreamVec(tensor.get_dim()));
 */
template <typename T>
class StreamVec
{
public:
    explicit StreamVec(const std::vector<T>& vec)
        : _vec(vec)
    {
    }

    friend std::ostream& operator<<(std::ostream& os, const StreamVec& wrapper)
    {
        os << "[";
        for(size_t i = 0; i < wrapper._vec.size(); ++i)
        {
            if(i > 0)
            {
                os << ", ";
            }
            os << wrapper._vec[i];
        }
        os << "]";
        return os;
    }

private:
    const std::vector<T>& _vec;
};

} // namespace hipdnn_test_sdk::utilities
