// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#pragma once

#ifdef HIPDNN_ENABLE_KERNEL_INGESTOR

#include <optional>
#include <string_view>

#include <hipdnn_plugin_sdk/ingestor/KernelDefinition.hpp>
#include <hipdnn_plugin_sdk/ingestor/MatchContext.hpp>

#include "engines/kernel_ingestor_engine/packs/ConvNaiveProblem.hpp"
#include "engines/kernel_ingestor_engine/packs/ConvNarrowPlan.hpp"

/**
 * @file ConvNarrowNative.hpp
 * @brief The graph half of CONV_NARROW_FWD / CONV_NARROW_BWD: which graphs they take (in the
 *        kernels' roles), a catalog config's metadata, and admission of a config on a device.
 */
namespace hip_kernel_provider::kernel_ingestor_engine::conv_narrow
{

/// A graph the engines take: one dense convolution node (groups 1), forward or backward data,
/// 1-D to 3-D, fp32 / fp16 / bf16 tensors of one element type, compute type unset or fp32,
/// both conv modes, any stride, padding and dilation, whose output side (K forward, C backward
/// data) has at most MAX_KO channels and fewer than the side it reduces; extents within 32 bits.
struct NarrowProblem
{
    conv_naive::ConvNaiveProblem base;
    NarrowShape shape;
};

/// The problem @p context's graph describes in @p direction, or nullopt.
std::optional<NarrowProblem>
    parseNarrowProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                       conv_naive::ConvDirection direction);

/// The config a catalog entry declares: strings direction, layout, atom; integers kmin, ko, fd,
/// fr, fs, sw, ds, cmin, cm, pd, sf, wr, rb, nw, op, atom_id (checkConfig). nullopt when a field is missing or malformed.
std::optional<Config> configOf(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel);

/// Whether the ship list @p archs (comma-separated: an arch, `gfx11`, `gfx12` or `any`) names
/// the device @p gcnArchName.
bool shipsTo(std::string_view archs, std::string_view gcnArchName);

/// A config admitted for a graph on a device.
struct Admitted
{
    NarrowProblem problem;
    Plan plan;
};

/// @p kernel admitted for @p context's graph and device: its direction and dtype are the graph's,
/// its ship list names the device, its atom runs there on the graph's element type (or is
/// `valu`), and the plan fits; nullopt otherwise.
std::optional<Admitted> admitted(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                                 const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel);

/// A backward-weights graph CONV_NARROW_WRW takes: one dense convolution node (groups 1), 1-D to
/// 3-D, fp32 / fp16 / bf16 tensors of one element type, compute type unset or fp32, both conv
/// modes, any stride, padding and dilation, one side (K: an output layer, or C: a stem) of at most
/// MAX_KO channels and narrower than the other; extents within 32 bits.
struct NarrowWrwProblem
{
    conv_naive::ConvNaiveProblem base;
    WrwShape shape;
};

/// The backward-weights problem @p context's graph describes, or nullopt.
std::optional<NarrowWrwProblem>
    parseNarrowWrwProblem(const hipdnn_plugin_sdk::ingestor::MatchContext& context);

/// The backward-weights config a catalog entry declares: strings side, layout, atom; integers
/// kmin, ko, fd, fr, fs, wr, rb, nw, mpw, atom_id, side_id (checkWrwConfig).
std::optional<WrwConfig> wrwConfigOf(const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel);

/// A backward-weights config admitted for a graph on a device.
struct WrwAdmitted
{
    NarrowWrwProblem problem;
    WrwPlan plan;
};

/// @p kernel admitted for @p context's backward-weights graph and device: its side and dtype are
/// the graph's, its ship list names the device, its atom runs there on the graph's element type
/// (or is `valu`), and the plan fits; nullopt otherwise.
std::optional<WrwAdmitted> admittedWrw(const hipdnn_plugin_sdk::ingestor::MatchContext& context,
                                       const hipdnn_plugin_sdk::ingestor::KernelDefinition& kernel);

} // namespace hip_kernel_provider::kernel_ingestor_engine::conv_narrow

#endif // HIPDNN_ENABLE_KERNEL_INGESTOR
