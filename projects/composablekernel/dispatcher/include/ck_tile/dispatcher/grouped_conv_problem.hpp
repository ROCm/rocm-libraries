// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

/**
 * @file grouped_conv_problem.hpp
 * @brief Grouped Convolution problem definition
 */

#pragma once

#include <cstdint>
#include <array>
#include <stdexcept>
#include <string>
#include <utility>

namespace ck_tile {
namespace dispatcher {

/**
 * @brief Grouped Convolution operation type
 */
enum class GroupedConvOp
{
    Forward,       // Y = Conv(X, W)
    BackwardData,  // dX = ConvBwdData(dY, W)
    BackwardWeight // dW = ConvBwdWeight(X, dY)
};

/**
 * @brief Grouped Convolution problem specification
 */
struct GroupedConvProblem
{
    // Batch and channels
    std::int64_t N; // Batch size
    std::int64_t C; // Input channels
    std::int64_t K; // Output channels (filters)
    std::int64_t G; // Number of groups (1 for standard conv)

    // Spatial dimensions (supports 2D and 3D; unused slot is 1 for 2D)
    std::array<std::int64_t, 3> input_spatial;  // {D, H, W} or {1, H, W} for 2D
    std::array<std::int64_t, 3> filter_spatial; // {Z, Y, X} or {1, Y, X} for 2D
    std::array<std::int64_t, 3> output_spatial; // {Do, Ho, Wo} or {1, Ho, Wo} for 2D

    // Runtime signature (canonical values, not aliases)
    std::string dtype_in;
    std::string dtype_wei;
    std::string dtype_out;
    std::string layout;
    int ndim_spatial = 2;
    std::string arch;

    // Convolution parameters
    std::array<std::int64_t, 3> stride;
    std::array<std::int64_t, 3> padding_left;
    std::array<std::int64_t, 3> padding_right;
    std::array<std::int64_t, 3> dilation;

    // Operation type
    GroupedConvOp op = GroupedConvOp::Forward;

    // Split-K for backward weight (k_batch parameter in CK Tile).
    // Values > 1 split the reduction dimension across multiple thread blocks
    // and use atomic accumulation.
    int split_k = 1;

    // Default constructor for 2D convolution
    GroupedConvProblem()
        : N(1),
          C(64),
          K(64),
          G(1),
          input_spatial{1, 28, 28},
          filter_spatial{1, 3, 3},
          output_spatial{1, 26, 26},
          stride{1, 1, 1},
          padding_left{0, 0, 0},
          padding_right{0, 0, 0},
          dilation{1, 1, 1},
          op(GroupedConvOp::Forward)
    {
    }

    // Constructor for 2D convolution
    GroupedConvProblem(std::int64_t n,
                       std::int64_t c,
                       std::int64_t k,
                       std::int64_t hi,
                       std::int64_t wi,
                       std::int64_t y,
                       std::int64_t x,
                       std::int64_t stride_h   = 1,
                       std::int64_t stride_w   = 1,
                       std::int64_t pad_h      = 0,
                       std::int64_t pad_w      = 0,
                       std::int64_t dilation_h = 1,
                       std::int64_t dilation_w = 1)
        : N(n),
          C(c),
          K(k),
          G(1),
          input_spatial{1, hi, wi},
          filter_spatial{1, y, x},
          stride{1, stride_h, stride_w},
          padding_left{0, pad_h, pad_w},
          padding_right{0, pad_h, pad_w},
          dilation{1, dilation_h, dilation_w},
          op(GroupedConvOp::Forward)
    {
        compute_output_size();
    }

    /// Check if problem dimensions are valid
    bool is_valid() const
    {
        if(N <= 0 || C <= 0 || K <= 0 || G <= 0 || C % G != 0 || K % G != 0 ||
           dtype_in.empty() || dtype_wei.empty() || dtype_out.empty() || layout.empty() ||
           arch.empty() || (ndim_spatial != 2 && ndim_spatial != 3))
            return false;

        if(ndim_spatial == 2 &&
           (input_spatial[0] != 1 || filter_spatial[0] != 1 || output_spatial[0] != 1 ||
            stride[0] != 1 || dilation[0] != 1 || padding_left[0] != 0 ||
            padding_right[0] != 0))
            return false;

        for(int i = 0; i < 3; ++i)
        {
            if(input_spatial[i] <= 0 || filter_spatial[i] <= 0 || output_spatial[i] <= 0 ||
               stride[i] <= 0 || dilation[i] <= 0 || padding_left[i] < 0 ||
               padding_right[i] < 0)
                return false;
            const std::int64_t effective_filter =
                (filter_spatial[i] - 1) * dilation[i] + 1;
            const std::int64_t numerator = input_spatial[i] + padding_left[i] +
                                           padding_right[i] - effective_filter;
            if(numerator < 0 || output_spatial[i] != numerator / stride[i] + 1)
                return false;
        }
        return true;
    }

    /// Compute output dimensions when all spatial parameters are valid.
    /// Returns false without changing output_spatial when the inputs cannot describe
    /// a positive output.
    bool compute_output_size()
    {
        std::array<std::int64_t, 3> computed_output;
        for(int i = 0; i < 3; ++i)
        {
            if(input_spatial[i] <= 0 || filter_spatial[i] <= 0 || stride[i] <= 0 ||
               dilation[i] <= 0 || padding_left[i] < 0 || padding_right[i] < 0)
                return false;
            const std::int64_t effective_filter =
                (filter_spatial[i] - 1) * dilation[i] + 1;
            const std::int64_t numerator = input_spatial[i] + padding_left[i] +
                                           padding_right[i] - effective_filter;
            if(numerator < 0)
                return false;
            computed_output[i] = numerator / stride[i] + 1;
        }
        output_spatial = computed_output;
        return true;
    }

    /// Get 2D height/width accessors
    std::int64_t Hi() const { return input_spatial[1]; }
    std::int64_t Wi() const { return input_spatial[2]; }
    std::int64_t Ho() const { return output_spatial[1]; }
    std::int64_t Wo() const { return output_spatial[2]; }
    std::int64_t Y() const { return filter_spatial[1]; } // Filter height
    std::int64_t X() const { return filter_spatial[2]; } // Filter width

    /// Get total FLOPs for this convolution
    double get_flops() const
    {
        // Forward: 2 * N * K * Ho * Wo * C * Y * X / G
        double spatial_out = 1.0;
        double filter_size = 1.0;
        for(int i = 0; i < 3; ++i)
        {
            spatial_out *= output_spatial[i];
            filter_size *= filter_spatial[i];
        }
        return 2.0 * N * K * spatial_out * (C / G) * filter_size;
    }

    /// Check if this is a depthwise convolution
    bool is_depthwise() const { return G == C && G == K; }

    /// Check if this is a pointwise (1x1) convolution
    bool is_pointwise() const
    {
        return filter_spatial[0] == 1 && filter_spatial[1] == 1 && filter_spatial[2] == 1;
    }

    /// String representation
    std::string to_string() const
    {
        std::string s = "GroupedConvProblem(N=" + std::to_string(N);
        s += ", C=" + std::to_string(C) + ", K=" + std::to_string(K);
        s += ", G=" + std::to_string(G);
        s += ", Hi=" + std::to_string(Hi()) + ", Wi=" + std::to_string(Wi());
        s += ", Y=" + std::to_string(Y()) + ", X=" + std::to_string(X());
        s += ", Ho=" + std::to_string(Ho()) + ", Wo=" + std::to_string(Wo());
        s += ")";
        return s;
    }
};

// =============================================================================
// GroupedConvProblemBuilder
// =============================================================================

/// Builder pattern for Grouped Convolution problem configuration
class GroupedConvProblemBuilder
{
    public:
    GroupedConvProblemBuilder() = default;

    GroupedConvProblemBuilder& batch(std::int64_t n)
    {
        problem_.N = n;
        return *this;
    }

    GroupedConvProblemBuilder& channels(std::int64_t c, std::int64_t k)
    {
        problem_.C = c;
        problem_.K = k;
        return *this;
    }

    GroupedConvProblemBuilder& groups(std::int64_t g)
    {
        problem_.G = g;
        return *this;
    }

    GroupedConvProblemBuilder& ndim(int value)
    {
        problem_.ndim_spatial = value;
        return *this;
    }

    GroupedConvProblemBuilder& input_size(std::int64_t h, std::int64_t w)
    {
        problem_.input_spatial[0] = 1;
        problem_.input_spatial[1] = h;
        problem_.input_spatial[2] = w;
        return *this;
    }

    GroupedConvProblemBuilder& input_size(std::int64_t d, std::int64_t h, std::int64_t w)
    {
        problem_.input_spatial = {d, h, w};
        return *this;
    }

    GroupedConvProblemBuilder& filter_size(std::int64_t y, std::int64_t x)
    {
        problem_.filter_spatial[0] = 1;
        problem_.filter_spatial[1] = y;
        problem_.filter_spatial[2] = x;
        return *this;
    }

    GroupedConvProblemBuilder& filter_size(std::int64_t z, std::int64_t y, std::int64_t x)
    {
        problem_.filter_spatial = {z, y, x};
        return *this;
    }

    GroupedConvProblemBuilder& stride(std::int64_t sh, std::int64_t sw)
    {
        problem_.stride[0] = 1;
        problem_.stride[1] = sh;
        problem_.stride[2] = sw;
        return *this;
    }

    GroupedConvProblemBuilder& stride(std::int64_t sd, std::int64_t sh, std::int64_t sw)
    {
        problem_.stride = {sd, sh, sw};
        return *this;
    }

    /// Set symmetric 2D {H, W} padding; depth padding remains zero.
    GroupedConvProblemBuilder& padding(std::int64_t ph, std::int64_t pw)
    {
        problem_.padding_left  = {0, ph, pw};
        problem_.padding_right = {0, ph, pw};
        return *this;
    }

    /// Set symmetric 3D {D, H, W} padding.
    GroupedConvProblemBuilder& padding(std::int64_t pd, std::int64_t ph, std::int64_t pw)
    {
        problem_.padding_left  = {pd, ph, pw};
        problem_.padding_right = {pd, ph, pw};
        return *this;
    }

    /// Set asymmetric 2D {left H, left W, right H, right W} padding.
    GroupedConvProblemBuilder& padding(std::int64_t pad_h_left,
                                       std::int64_t pad_w_left,
                                       std::int64_t pad_h_right,
                                       std::int64_t pad_w_right)
    {
        problem_.padding_left  = {0, pad_h_left, pad_w_left};
        problem_.padding_right = {0, pad_h_right, pad_w_right};
        return *this;
    }

    /// Set asymmetric 3D {left D, H, W, right D, H, W} padding.
    GroupedConvProblemBuilder& padding(std::int64_t pad_d_left,
                                       std::int64_t pad_h_left,
                                       std::int64_t pad_w_left,
                                       std::int64_t pad_d_right,
                                       std::int64_t pad_h_right,
                                       std::int64_t pad_w_right)
    {
        problem_.padding_left  = {pad_d_left, pad_h_left, pad_w_left};
        problem_.padding_right = {pad_d_right, pad_h_right, pad_w_right};
        return *this;
    }

    GroupedConvProblemBuilder& dilation(std::int64_t dh, std::int64_t dw)
    {
        problem_.dilation[0] = 1;
        problem_.dilation[1] = dh;
        problem_.dilation[2] = dw;
        return *this;
    }

    GroupedConvProblemBuilder& data_types(std::string in, std::string wei, std::string out)
    {
        problem_.dtype_in  = std::move(in);
        problem_.dtype_wei = std::move(wei);
        problem_.dtype_out = std::move(out);
        return *this;
    }

    GroupedConvProblemBuilder& layout(std::string value)
    {
        problem_.layout = std::move(value);
        return *this;
    }

    GroupedConvProblemBuilder& arch(std::string value)
    {
        problem_.arch = std::move(value);
        return *this;
    }

    GroupedConvProblemBuilder& dilation(std::int64_t dd,
                                        std::int64_t dh,
                                        std::int64_t dw)
    {
        problem_.dilation = {dd, dh, dw};
        return *this;
    }

    GroupedConvProblemBuilder& operation(GroupedConvOp op)
    {
        problem_.op = op;
        return *this;
    }

    [[nodiscard]] GroupedConvProblem build() const
    {
        GroupedConvProblem p = problem_;
        if(!p.compute_output_size() || !p.is_valid())
        {
            throw std::invalid_argument("Invalid grouped convolution problem dimensions");
        }
        return p;
    }

    private:
    GroupedConvProblem problem_;
};

} // namespace dispatcher
} // namespace ck_tile
