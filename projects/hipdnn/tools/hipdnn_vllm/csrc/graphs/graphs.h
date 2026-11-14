// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Op entry points implemented by the graph builders. ops.cpp registers them.

#pragma once

#include "../decline.h"

#include <ATen/ATen.h>

#include <cstdint>
#include <optional>
#include <pybind11/pytypes.h>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace hipdnn_vllm
{

// Every op raises a RuntimeError whose message starts with this prefix when
// hipDNN declines, followed by the reason (STAGE[:detail]) and " :: " and the
// backend message. hipdnn_vllm.declines.parse_op_error() decodes it.
std::string declineErrorPrefix(const char* op);

// torch.ops.hipdnn_vllm.rms_norm
at::Tensor rmsNorm(const at::Tensor& x, const at::Tensor& weight, double eps);
// "" when hipDNN would run this signature, else the decline reason. Builds and
// caches the plan; does not execute or count.
std::string rmsNormProbe(const at::Tensor& x, const at::Tensor& weight, double eps);

// rms_norm building blocks for norm_rows.cpp (norm.cpp owns the graph).
std::optional<Decline> rmsNormPrefilter(const at::Tensor& x, const at::Tensor& weight);
// x [d0, ..., N] as the no-copy [R, N] view; nullopt when the leading dims do
// not collapse into one row stride.
std::optional<at::Tensor> rmsNormAsRows(const at::Tensor& x);
// [R, N] -> dims/strides of the [R, N, 1, 1] description.
void rmsNormRowLayout(const at::Tensor& t, std::vector<int64_t>& dims, std::vector<int64_t>& strides);
// Run the exact plan for the [R, N] views x -> y (y allocated by the caller).
// nullopt = ran.
std::optional<Decline>
    rmsNormInto(const at::Tensor& x, const at::Tensor& weight, const at::Tensor& y, double eps);
// Build (or fetch) the plan for x -> y without running it; x and y may be
// meta tensors, `anchor` names the device.
std::optional<Decline> rmsNormPrepare(
    const at::Tensor& x, const at::Tensor& weight, const at::Tensor& y, double eps, const at::Tensor& anchor);

// torch.ops.hipdnn_vllm.rms_norm_rows / fused_add_rms_norm_rows:
// any number of rows (dim 0) with the dynamic-row interim, and an in-op
// fallback to the native formula on a decline (raises when `strict`).
at::Tensor rmsNormRows(const at::Tensor& x, const at::Tensor& weight, double eps, bool strict);
std::tuple<at::Tensor, at::Tensor> fusedAddRmsNormRows(
    const at::Tensor& x, const at::Tensor& residual, const at::Tensor& weight, double eps, bool strict);

// Row counts that get an exact plan (vLLM's cudagraph capture sizes).
void normSetExactRows(std::vector<int64_t> rows);
std::optional<std::vector<int64_t>> normExactRows();
// "fused", "decomposed" or "" (not decided yet) for this signature.
std::string fusedAddRoute(const at::Tensor& x, const at::Tensor& residual, const at::Tensor& weight, double eps);
struct NormPrebuildTotals
{
    uint64_t signatures;
    uint64_t plans;
    uint64_t declines;
    uint64_t ns;
};
NormPrebuildTotals normPrebuildTotals();
// Forget per-signature row state and the exact-row configuration (plans stay
// in the plan cache).
void normRowsReset();

// torch.ops.hipdnn_vllm.linear: y = x · weightᵀ + bias.
at::Tensor
    linear(const at::Tensor& x, const at::Tensor& weight, const std::optional<at::Tensor>& bias);
// Same, without raising: (y, "", "") when hipDNN ran it, else (None, reason,
// backend message). The decline is already counted.
std::tuple<std::optional<at::Tensor>, std::string, std::string>
    linearTry(const at::Tensor& x, const at::Tensor& weight, const std::optional<at::Tensor>& bias);
// ("", engine name) when hipDNN would run this signature, else (reason, "").
// Builds and caches the plan; does not execute or count.
std::pair<std::string, std::string> linearProbe(const at::Tensor& x,
                                                const at::Tensor& weight,
                                                const std::optional<at::Tensor>& bias);

// torch.ops.hipdnn_vllm.sdpa_fwd: SDPA forward written into `out` (sdpa.cpp has
// the layout contract: dense, ragged Q/O, ragged or paged K/V, masks, GQA, and
// the optional per-head attention sink [1, Hq, 1, 1]).
void sdpaFwd(const at::Tensor& q,
             const at::Tensor& k,
             const at::Tensor& v,
             const at::Tensor& out,
             double scale,
             bool causal,
             bool bottomRight,
             int64_t leftBound,
             int64_t rightBound,
             const std::optional<at::Tensor>& raggedOffsetQ,
             const std::optional<at::Tensor>& seqLenQ,
             int64_t maxSeqLenQ,
             const std::optional<at::Tensor>& raggedOffsetKv,
             const std::optional<at::Tensor>& pageTable,
             const std::optional<at::Tensor>& seqLenKv,
             int64_t maxSeqLenKv,
             const std::optional<at::Tensor>& sink);
// (reason, detail): reason "" when hipDNN would run the signature and detail is
// the engine name; otherwise the decline reason and the backend message. Builds
// and caches the plan; does not execute or count.
std::pair<std::string, std::string> sdpaFwdProbe(const at::Tensor& q,
                                                 const at::Tensor& k,
                                                 const at::Tensor& v,
                                                 const at::Tensor& out,
                                                 double scale,
                                                 bool causal,
                                                 bool bottomRight,
                                                 int64_t leftBound,
                                                 int64_t rightBound,
                                                 const std::optional<at::Tensor>& raggedOffsetQ,
                                                 const std::optional<at::Tensor>& seqLenQ,
                                                 int64_t maxSeqLenQ,
                                                 const std::optional<at::Tensor>& raggedOffsetKv,
                                                 const std::optional<at::Tensor>& pageTable,
                                                 const std::optional<at::Tensor>& seqLenKv,
                                                 int64_t maxSeqLenKv,
                                                 const std::optional<at::Tensor>& sink);
// The described graph (tensors, uids, strides, ragged multipliers, slots, plan
// key) without building anything, for tests and probes.
pybind11::dict sdpaFwdLayout(const at::Tensor& q,
                             const at::Tensor& k,
                             const at::Tensor& v,
                             const at::Tensor& out,
                             double scale,
                             bool causal,
                             bool bottomRight,
                             int64_t leftBound,
                             int64_t rightBound,
                             const std::optional<at::Tensor>& raggedOffsetQ,
                             const std::optional<at::Tensor>& seqLenQ,
                             int64_t maxSeqLenQ,
                             const std::optional<at::Tensor>& raggedOffsetKv,
                             const std::optional<at::Tensor>& pageTable,
                             const std::optional<at::Tensor>& seqLenKv,
                             int64_t maxSeqLenKv,
                             const std::optional<at::Tensor>& sink);

// torch.ops.hipdnn_vllm.encoder_attention: dense non-causal SDPA over
// q [B, S, Hq, D], k/v [B, S_kv, Hk, D(v)], split into images by cu_seqlens.
// Returns [B, S, Hq, Dv] contiguous.
at::Tensor encoderAttention(const at::Tensor& q,
                            const at::Tensor& k,
                            const at::Tensor& v,
                            const std::optional<at::Tensor>& cuSeqlens,
                            double scale);
// torch.ops.hipdnn_vllm.layer_norm. On a decline it counts a fallback and
// returns the native result, or raises when `strict`.
at::Tensor layerNorm(const at::Tensor& x,
                     const std::optional<at::Tensor>& weight,
                     const std::optional<at::Tensor>& bias,
                     double eps,
                     bool strict);
std::string layerNormProbe(const at::Tensor& x,
                           const std::optional<at::Tensor>& weight,
                           const std::optional<at::Tensor>& bias,
                           double eps);

// torch.ops.hipdnn_vllm.conv_fwd. Same decline behaviour as layerNorm.
at::Tensor convFwd(const at::Tensor& x,
                   const at::Tensor& weight,
                   const std::optional<at::Tensor>& bias,
                   at::IntArrayRef stride,
                   at::IntArrayRef padding,
                   at::IntArrayRef dilation,
                   int64_t groups,
                   bool strict);
std::string convFwdProbe(const at::Tensor& x,
                         const at::Tensor& weight,
                         const std::optional<at::Tensor>& bias,
                         at::IntArrayRef stride,
                         at::IntArrayRef padding,
                         at::IntArrayRef dilation,
                         int64_t groups);

} // namespace hipdnn_vllm
