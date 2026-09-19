/*
MIT License

Copyright (c) 2026 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#ifndef RPP_TEST_ND_CONFIG_PARAM_H
#define RPP_TEST_ND_CONFIG_PARAM_H

#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <ostream>
#include <string>
#include <vector>

#include "framework/backend_param.hpp"
#include "framework/config_param.hpp"

// Grid axes for the ND "generic tensor" ops (Misc domain).
// The shared axes (DType, the WithParams machinery, num_token) live in config_param.hpp.

namespace rpptest {

// The generic-tensor ops are gridded over rank instead of layout/ROI. The label maps onto the same
// structural slots as the image grammar: rank takes the Layout slot, the op param takes the Roi
// slot. Descriptor construction and traversal for these live in generic_tensor_setup.hpp.

// Extents including the leading batch axis, matching RpptGenericDesc::dims.
using NdDims = std::vector<Rpp32u>;

inline Rpp32u nd_rank(const NdDims& dims) {
    return static_cast<Rpp32u>(dims.size()) - 1;
}

// Which operand, if any, is broadcast: its trailing axis is collapsed to extent 1.
enum class Broadcast { None, Src1, Src2 };

inline std::string broadcast_name(Broadcast b) {
    switch (b) {
        case Broadcast::None:
            return "NoBroadcast";
        case Broadcast::Src1:
            return "BroadcastSrc1";
        case Broadcast::Src2:
            return "BroadcastSrc2";
    }
    return "UNK";
}

inline RpptBroadcastMode to_rpp_broadcast(Broadcast b) {
    return b == Broadcast::None ? RPP_BROADCAST_DISABLE : RPP_BROADCAST_ENABLE;
}

struct DTypeConv {
    DType in, out;
};

// The innermost extent decides whether the vector loops see a tail. The ND tensors are dense
// (unlike the image domain, whose rows are padded out), so a non-multiple innermost extent is the
// only way the tail path and the one-past-the-end of the last store are exercised here. This is
// the counterpart of the image grid's size axis.
enum class NdShape { VectorAligned, Tail };

struct NdConfig {
    RppBackend backend;
    DType dtypeIn, dtypeOut;
    Rpp32u nDim;  // per-sample rank: 2, 3 or 4
    NdShape shape;
};

// Every axis has a distinct extent, so an axis mix-up cannot pass by coincidence.
inline NdDims nd_extents(Rpp32u nDim, NdShape shape = NdShape::VectorAligned) {
    const bool tail = shape == NdShape::Tail;
    switch (nDim) {
        case 3:
            return tail ? NdDims{2, 5, 12, 19} : NdDims{2, 5, 12, 16};
        case 4:
            return tail ? NdDims{2, 2, 4, 10, 13} : NdDims{2, 2, 4, 10, 12};
        default:
            return tail ? NdDims{2, 24, 35} : NdDims{2, 24, 32};
    }
}

inline NdDims nd_extents(const NdConfig& c) {
    return nd_extents(c.nDim, c.shape);
}

inline NdDims nd_operand_dims(const NdConfig& c, Broadcast broadcast, int operand) {
    NdDims dims = nd_extents(c);
    if ((broadcast == Broadcast::Src1 && operand == 1) ||
        (broadcast == Broadcast::Src2 && operand == 2))
        dims.back() = 1;
    return dims;
}

// Per-axis max; the operands only ever differ where one of them is 1.
inline NdDims nd_broadcast_dims(const NdDims& a, const NdDims& b) {
    NdDims out(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) out[i] = a[i] > b[i] ? a[i] : b[i];
    return out;
}

inline std::vector<NdConfig> make_nd_configs(const std::vector<DTypeConv>& convs,
                                             const std::vector<Rpp32u>& ranks,
                                             const std::vector<NdShape>& shapes) {
    std::vector<NdConfig> configs;
    for (RppBackend backend : available_backends())
        for (DTypeConv conv : convs)
            for (Rpp32u nDim : ranks)
                for (NdShape shape : shapes)
                    configs.push_back({backend, conv.in, conv.out, nDim, shape});
    return configs;
}

inline std::vector<NdConfig> make_nd_configs(const std::vector<DType>& dtypes,
                                             const std::vector<Rpp32u>& ranks,
                                             const std::vector<NdShape>& shapes) {
    std::vector<DTypeConv> convs;
    for (DType d : dtypes) convs.push_back({d, d});
    return make_nd_configs(convs, ranks, shapes);
}

// "<Backend>_<DTypeConv>_<Rank>[_<opToken>]_<Shape>". The shape token spells the extents out, so
// NdShape needs no token of its own.
inline std::string nd_label(const NdConfig& c, const std::string& opToken) {
    const NdDims dims = nd_extents(c);
    std::string shape = std::to_string(dims[0]);
    for (std::size_t i = 1; i < dims.size(); ++i) shape += "x" + std::to_string(dims[i]);
    return backend_name(c.backend) + "_" + dtype_name(c.dtypeIn) + "to" + dtype_name(c.dtypeOut) +
           "_" + std::to_string(c.nDim) + "D_" + (opToken.empty() ? "" : opToken + "_") + shape;
}

inline std::string nd_config_name(const NdConfig& c) {
    return nd_label(c, "");
}

inline std::string nd_config_param_name(const ::testing::TestParamInfo<NdConfig>& info) {
    return nd_config_name(info.param);
}

// The ND counterpart of WithParams. P must provide std::string name() const.
template <typename P>
struct NdWithParams {
    NdConfig cfg;
    P op;
};

template <typename P>
inline std::vector<NdWithParams<P>> nd_with_params(const std::vector<NdConfig>& base,
                                                   const std::vector<P>& params) {
    std::vector<NdWithParams<P>> out;
    out.reserve(base.size() * params.size());
    for (const auto& c : base)
        for (const auto& p : params) out.push_back({c, p});
    return out;
}

template <typename P>
inline std::string nd_config_name(const NdWithParams<P>& p) {
    return nd_label(p.cfg, p.op.name());
}

template <typename P>
inline std::string nd_op_config_name(const ::testing::TestParamInfo<NdWithParams<P>>& info) {
    return nd_config_name(info.param);
}

struct BroadcastParams {
    Broadcast mode;
    std::string name() const {
        return broadcast_name(mode);
    }
};

inline void PrintTo(const NdConfig&, std::ostream*) {}

template <typename P>
void PrintTo(const NdWithParams<P>&, std::ostream*) {}

}  // namespace rpptest

#endif  // RPP_TEST_ND_CONFIG_PARAM_H
