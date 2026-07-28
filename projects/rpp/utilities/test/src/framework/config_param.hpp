#ifndef RPP_TEST_CONFIG_PARAM_H
#define RPP_TEST_CONFIG_PARAM_H

#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <string>
#include <vector>

#include "framework/backend_param.hpp"

namespace rpptest {

// The cross-cutting configuration axes each op is tested over. These map onto the
// {Backend}_{DTypeConv}_{Layout}_{Roi} tokens of the value-parameter label so every
// axis is independently greppable via --gtest_filter.

// I16 is only reachable through the ND (Misc) grid -- rppt_log1p documents i16->f32 as its
// only conversion. The image-domain helpers (to_unit / from_unit / quantize_stored) do not
// model it.
enum class DType { U8, F16, F32, I8, I16 };
enum class Layout { PKD3, PLN3, PLN1 };  // PKD3/PLN3 => 3 channels, PLN1 => 1 channel
enum class Roi { Full, Partial };

// Spatial extent of the test tensor. The channel count is not carried here: it is
// derived from Layout (PLN1 => 1, PKD3/PLN3 => 3) at descriptor-build time.
struct Size {
    Rpp32u n, h, w;
};

inline std::string dtype_name(DType d) {
    switch (d) {
        case DType::U8: return "U8";
        case DType::F16: return "F16";
        case DType::F32: return "F32";
        case DType::I8: return "I8";
        case DType::I16: return "I16";
    }
    return "UNK";
}

inline std::string layout_name(Layout l) {
    switch (l) {
        case Layout::PKD3: return "PKD3";
        case Layout::PLN3: return "PLN3";
        case Layout::PLN1: return "PLN1";
    }
    return "UNK";
}

inline std::string roi_name(Roi r) { return r == Roi::Full ? "FullRoi" : "PartialRoi"; }

inline std::string size_name(Size s) {
    return std::to_string(s.n) + "x" + std::to_string(s.h) + "x" + std::to_string(s.w);
}

// A single point in the test grid. dtypeIn == dtypeOut for now; kept as one field until
// mixed-precision conversions (e.g. U8->F32) are exercised.
struct TestConfig {
    RppBackend backend;
    DType dtype;
    Layout layout;
    Roi roi;
    Size size;
};

// Produces the value-parameter label, e.g. "HIP_U8toU8_PKD3_FullRoi_2x36x48".
inline std::string config_name(const TestConfig& c) {
    return backend_name(c.backend) + "_" + dtype_name(c.dtype) + "to" + dtype_name(c.dtype) +
           "_" + layout_name(c.layout) + "_" + roi_name(c.roi) + "_" + size_name(c.size);
}

// Cartesian product of the requested axes with every available backend. Pass the
// dtype/layout/roi/size sets an op supports; HIP is only present when the suite was built
// with the HIP backend (see available_backends()). Most ops take the default single size.
inline std::vector<TestConfig> make_configs(const std::vector<DType>& dtypes,
                                            const std::vector<Layout>& layouts,
                                            const std::vector<Roi>& rois,
                                            const std::vector<Size>& sizes = {{2, 36, 48}}) {
    std::vector<TestConfig> configs;
    for (RppBackend backend : available_backends())
        for (DType dtype : dtypes)
            for (Layout layout : layouts)
                for (Roi roi : rois)
                    for (Size size : sizes)
                        configs.push_back({backend, dtype, layout, roi, size});
    return configs;
}

// GTest name generator: turns each TestConfig into its filterable label.
inline std::string config_param_name(const ::testing::TestParamInfo<TestConfig>& info) {
    return config_name(info.param);
}

// ---- op-specific parameters -----------------------------------------------
//
// Universal axes live in TestConfig; scalar op inputs (blend alpha, brightness
// alpha/beta, ...) are carried alongside it via WithParams<P>, where P is a small
// per-op struct defined in that op's test. This keeps the shared grid uniform while
// letting each op bake its own values in at INSTANTIATE_TEST_SUITE_P time (and turn
// any of them into an axis just by passing more than one value).

template <typename P>
struct WithParams {
    TestConfig cfg;
    P op;
};

// Attaches each op-param set to every base config (op params as an extra grid axis).
template <typename P>
inline std::vector<WithParams<P>> with_params(const std::vector<TestConfig>& base,
                                              const std::vector<P>& params) {
    std::vector<WithParams<P>> out;
    out.reserve(base.size() * params.size());
    for (const auto& c : base)
        for (const auto& p : params) out.push_back({c, p});
    return out;
}

// GTest name generator for parameterized ops: config label + the op's own suffix,
// e.g. "HIP_U8toU8_PKD3_FullRoi_2x36x48_a0p75". P must provide std::string name() const.
template <typename P>
inline std::string op_config_name(const ::testing::TestParamInfo<WithParams<P>>& info) {
    const std::string suffix = info.param.op.name();
    return config_name(info.param.cfg) + (suffix.empty() ? "" : "_" + suffix);
}

// Renders a float as a gtest-legal token ([A-Za-z0-9_]): '.' -> 'p', leading '-' -> 'n',
// trailing zeros trimmed. 0.75 -> "0p75", -1.5 -> "n1p5", 50.0 -> "50".
inline std::string num_token(float v) {
    std::string s = std::to_string(v);
    if (s.find('.') != std::string::npos) {
        s.erase(s.find_last_not_of('0') + 1);
        if (!s.empty() && s.back() == '.') s.pop_back();
    }
    for (char& ch : s) {
        if (ch == '.') ch = 'p';
        else if (ch == '-') ch = 'n';
    }
    return s;
}

// ---- ND (Misc) grid axes ----------------------------------------------------
//
// The generic-tensor ops are gridded over rank instead of layout/ROI. The label maps onto the same
// structural slots as the image grammar: rank takes the Layout slot, the op param takes the Roi
// slot. Descriptor construction and traversal for these live in generic_tensor_setup.hpp.

// Extents including the leading batch axis, matching RpptGenericDesc::dims.
using NdDims = std::vector<Rpp32u>;

inline Rpp32u nd_rank(const NdDims& dims) { return static_cast<Rpp32u>(dims.size()) - 1; }

// Which operand, if any, is broadcast: its trailing axis is collapsed to extent 1.
enum class Broadcast { None, Src1, Src2 };

inline std::string broadcast_name(Broadcast b) {
    switch (b) {
        case Broadcast::None: return "NoBroadcast";
        case Broadcast::Src1: return "BroadcastSrc1";
        case Broadcast::Src2: return "BroadcastSrc2";
    }
    return "UNK";
}

inline RpptBroadcastMode to_rpp_broadcast(Broadcast b) {
    return b == Broadcast::None ? RPP_BROADCAST_DISABLE : RPP_BROADCAST_ENABLE;
}

struct DTypeConv {
    DType in, out;
};

struct NdConfig {
    RppBackend backend;
    DType dtypeIn, dtypeOut;
    Rpp32u nDim;  // per-sample rank: 2, 3 or 4
};

// Every axis has a distinct extent, so an axis mix-up cannot pass by coincidence.
inline NdDims nd_extents(Rpp32u nDim) {
    switch (nDim) {
        case 3:  return {2, 5, 12, 16};
        case 4:  return {2, 2, 4, 10, 12};
        default: return {2, 24, 32};
    }
}

inline NdDims nd_operand_dims(Rpp32u nDim, Broadcast broadcast, int operand) {
    NdDims dims = nd_extents(nDim);
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
                                             const std::vector<Rpp32u>& ranks) {
    std::vector<NdConfig> configs;
    for (RppBackend backend : available_backends())
        for (DTypeConv conv : convs)
            for (Rpp32u nDim : ranks) configs.push_back({backend, conv.in, conv.out, nDim});
    return configs;
}

inline std::vector<NdConfig> make_nd_configs(const std::vector<DType>& dtypes,
                                             const std::vector<Rpp32u>& ranks) {
    std::vector<DTypeConv> convs;
    for (DType d : dtypes) convs.push_back({d, d});
    return make_nd_configs(convs, ranks);
}

// "<Backend>_<DTypeConv>_<Rank>[_<opToken>]_<Shape>".
inline std::string nd_label(const NdConfig& c, const std::string& opToken) {
    const NdDims dims = nd_extents(c.nDim);
    std::string shape = std::to_string(dims[0]);
    for (std::size_t i = 1; i < dims.size(); ++i) shape += "x" + std::to_string(dims[i]);
    return backend_name(c.backend) + "_" + dtype_name(c.dtypeIn) + "to" + dtype_name(c.dtypeOut) +
           "_" + std::to_string(c.nDim) + "D_" + (opToken.empty() ? "" : opToken + "_") + shape;
}

inline std::string nd_config_name(const NdConfig& c) { return nd_label(c, ""); }

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
    std::string name() const { return broadcast_name(mode); }
};

}  // namespace rpptest

#endif  // RPP_TEST_CONFIG_PARAM_H
