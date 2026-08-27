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

#ifndef RPP_TEST_CONFIG_PARAM_H
#define RPP_TEST_CONFIG_PARAM_H

#include <gtest/gtest.h>
#include <rpp/rpp.h>

#include <ostream>
#include <string>
#include <vector>

#include "framework/backend_param.hpp"
// Pulled in for its PrintTo(RppStatus, ...) overload: nearly every test that includes
// this header also asserts on an RppStatus return, so this is the shared point to get it.
#include "framework/status_param.hpp"

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
        case DType::U8:
            return "U8";
        case DType::F16:
            return "F16";
        case DType::F32:
            return "F32";
        case DType::I8:
            return "I8";
        case DType::I16:
            return "I16";
    }
    return "UNK";
}

inline std::string layout_name(Layout l) {
    switch (l) {
        case Layout::PKD3:
            return "PKD3";
        case Layout::PLN3:
            return "PLN3";
        case Layout::PLN1:
            return "PLN1";
    }
    return "UNK";
}

inline std::string roi_name(Roi r) {
    return r == Roi::Full ? "FullRoi" : "PartialRoi";
}

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
    return backend_name(c.backend) + "_" + dtype_name(c.dtype) + "to" + dtype_name(c.dtype) + "_" +
           layout_name(c.layout) + "_" + roi_name(c.roi) + "_" + size_name(c.size);
}

// The default shape for images.
inline constexpr Size kDefaultSize{2, 36, 48};

// Cartesian product of the requested axes with every available backend. Pass the
// dtype/layout/roi/size sets an op supports; HIP is only present when the suite was built
// with the HIP backend (see available_backends()). Most ops take the default single size.
inline std::vector<TestConfig> make_configs(const std::vector<DType>& dtypes,
                                            const std::vector<Layout>& layouts,
                                            const std::vector<Roi>& rois,
                                            const std::vector<Size>& sizes = {kDefaultSize}) {
    std::vector<TestConfig> configs;
    for (RppBackend backend : available_backends())
        for (DType dtype : dtypes)
            for (Layout layout : layouts)
                for (Roi roi : rois)
                    for (Size size : sizes) configs.push_back({backend, dtype, layout, roi, size});
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
        if (ch == '.')
            ch = 'p';
        else if (ch == '-')
            ch = 'n';
    }
    return s;
}

// ---- value-parameter printing ----------------------------------------------
//
// See the note on PrintTo in backend_param.hpp: the label is already in the test name, so
// every param type prints as empty.

inline void PrintTo(const TestConfig&, std::ostream*) {}

template <typename P>
void PrintTo(const WithParams<P>&, std::ostream*) {}

}  // namespace rpptest

#endif  // RPP_TEST_CONFIG_PARAM_H
