/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (C) 2021-2025 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#include "instance_params.hpp"

namespace ck::tensor_operation::device::instance
{
    std::vector<hiptensor::Uid>
        getHashCodeOfBestPerfInstances(std::vector<hipDataType> const&       typeIn,
                                       std::vector<hipDataType> const&       typeOut,
                                       hiptensor::PermutationOpId_t          scale,
                                       index_t                               numDim,
                                       hiptensor::InstanceHyperParams const& hyperParams)
    {
        std::vector<hiptensor::Uid> hashCodes;
        // - scalarPerVectorSeq is 0 when it is CPU reference instance.
        // - `hashCodes` may contain hash codes that not represent any instances. It is not a problem
        //      since these hash codes will be ignored.
        auto params = DeviceElementwiseParams::Gen(typeIn, typeOut, scale, numDim, hyperParams);
        hashCodes.push_back(hiptensor::Hash{}(params));
        // instances below are safe net
        // clang-format off
        if (typeIn.size() == 1)
        {
            if (numDim == 2) {
                if (typeIn[0] == HIP_R_16F) {
                    hashCodes.push_back(hiptensor::Hash{}( DeviceElementwiseParams::Gen(typeIn , typeOut , scale , numDim , {64  , 32  , 128 , 8 , 8 , {0 , 1} , {2} , {2}})));
                    hashCodes.push_back(hiptensor::Hash{}( DeviceElementwiseParams::Gen(typeIn , typeOut , scale , numDim , {64  , 32  , 128 , 8 , 8 , {0 , 1} , {1} , {1}})));
                } else if (typeIn[0] == HIP_R_32F) {
                    hashCodes.push_back(hiptensor::Hash{}( DeviceElementwiseParams::Gen(typeIn , typeOut , scale , numDim , {256 , 64  , 64  , 4 , 4 , {0 , 1} , {2} , {2}})));
                    hashCodes.push_back(hiptensor::Hash{}( DeviceElementwiseParams::Gen(typeIn , typeOut , scale , numDim , {256 , 64  , 64  , 4 , 4 , {0 , 1} , {1} , {1}})));
                }
            } else if (numDim == 3) {
                if (typeIn[0] == HIP_R_16F) {
                    hashCodes.push_back(hiptensor::Hash{}( DeviceElementwiseParams::Gen(typeIn , typeOut , scale , numDim , {256 , 128 , 128 , 8 , 8 , {0 , 1} , {2} , {2}})));
                    hashCodes.push_back(hiptensor::Hash{}( DeviceElementwiseParams::Gen(typeIn , typeOut , scale , numDim , {256 , 128 , 128 , 8 , 8 , {0 , 1} , {1} , {1}})));
                } else if (typeIn[0] == HIP_R_32F) {
                    hashCodes.push_back(hiptensor::Hash{}( DeviceElementwiseParams::Gen(typeIn , typeOut , scale , numDim , {256 , 64  , 64  , 4 , 4 , {0 , 1} , {2} , {2}})));
                    hashCodes.push_back(hiptensor::Hash{}( DeviceElementwiseParams::Gen(typeIn , typeOut , scale , numDim , {256 , 64  , 64  , 4 , 4 , {0 , 1} , {1} , {1}})));
                }
            } else if (numDim == 4) {
                if (typeIn[0] == HIP_R_16F) {
                    hashCodes.push_back(hiptensor::Hash{}( DeviceElementwiseParams::Gen(typeIn , typeOut , scale , numDim , {64  , 128 , 32  , 8  , 8  , {0 , 1} , {2}  , {2}})));
                    hashCodes.push_back(hiptensor::Hash{}( DeviceElementwiseParams::Gen(typeIn , typeOut , scale , numDim , {64  , 128 , 32  , 8  , 8  , {0 , 1} , {1}  , {1}})));
                } else if (typeIn[0] == HIP_R_32F) {
                    hashCodes.push_back(hiptensor::Hash{}( DeviceElementwiseParams::Gen(typeIn , typeOut , scale , numDim , {256 , 64  , 64  , 4 , 4 , {0 , 1} , {2} , {2}})));
                    hashCodes.push_back(hiptensor::Hash{}( DeviceElementwiseParams::Gen(typeIn , typeOut , scale , numDim , {256 , 64  , 64  , 4 , 4 , {0 , 1} , {1} , {1}})));
                }
            } else if (numDim == 5 || numDim == 6) {
                hashCodes.push_back(hiptensor::Hash{}( DeviceElementwiseParams::Gen(typeIn , typeOut , scale , numDim , {256 , 64  , 64  , 4 , 4 , {0 , 1} , {4} , {4}})));
                hashCodes.push_back(hiptensor::Hash{}( DeviceElementwiseParams::Gen(typeIn , typeOut , scale , numDim , {256 , 64  , 64  , 4 , 4 , {0 , 1} , {2} , {2}})));
                hashCodes.push_back(hiptensor::Hash{}( DeviceElementwiseParams::Gen(typeIn , typeOut , scale , numDim , {256 , 64  , 64  , 4 , 4 , {0 , 1} , {1} , {1}})));
            }
        } else if (typeIn.size() == 2) {
                hashCodes.push_back(hiptensor::Hash{}( DeviceElementwiseParams::Gen(typeIn , typeOut , scale , numDim , {256 , 64  , 64  , 4 , 4 , {0 , 1} , {4, 4} , {4}})));
                hashCodes.push_back(hiptensor::Hash{}( DeviceElementwiseParams::Gen(typeIn , typeOut , scale , numDim , {256 , 64  , 64  , 4 , 4 , {0 , 1} , {2, 2} , {2}})));
                hashCodes.push_back(hiptensor::Hash{}( DeviceElementwiseParams::Gen(typeIn , typeOut , scale , numDim , {256 , 64  , 64  , 4 , 4 , {0 , 1} , {1, 1} , {1}})));
        } else if (typeIn.size() == 3) {
                hashCodes.push_back(hiptensor::Hash{}( DeviceElementwiseParams::Gen(typeIn , typeOut , scale , numDim , {256 , 64  , 64  , 4 , 4 , {0 , 1} , {4, 4, 4} , {4}})));
                hashCodes.push_back(hiptensor::Hash{}( DeviceElementwiseParams::Gen(typeIn , typeOut , scale , numDim , {256 , 64  , 64  , 4 , 4 , {0 , 1} , {2, 2, 2} , {2}})));
                hashCodes.push_back(hiptensor::Hash{}( DeviceElementwiseParams::Gen(typeIn , typeOut , scale , numDim , {256 , 64  , 64  , 4 , 4 , {0 , 1} , {1, 1, 1} , {1}})));
        }
        // clang-format on

        return hashCodes;
    }
}
