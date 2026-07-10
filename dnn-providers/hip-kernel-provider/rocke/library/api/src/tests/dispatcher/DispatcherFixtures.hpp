// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <string>

#include "dispatcher/AotInstance.hpp"
#include "dispatcher/SdpaProblem.hpp"

// Test-only builders for dispatcher fixtures. These mirror the checked-in smoke
// instance in PR #8866 (fp16 / BSHD / mask none / no dropout / default scale) so
// selection behavior is exercised against a realistic contract without shipping
// any catalog. They are NOT production data.
namespace rocke_client::dispatcher::test
{

// The attribute constraints shared by the #8866 SDPA smoke instances.
inline AttributeConstraints smokeAttributeConstraints()
{
    AttributeConstraints constraints;
    AttributeRule maskMode;
    maskMode.equals = AttrValue{std::string("none")};
    constraints.emplace("mask_mode", maskMode);

    AttributeRule dropout;
    dropout.equals = AttrValue{0.0};
    constraints.emplace("dropout_probability", dropout);

    AttributeRule scale;
    scale.equals = AttrValue{std::string("default_1_over_sqrt_d")};
    constraints.emplace("scale_policy", scale);

    AttributeRule padding;
    padding.equals = AttrValue{false};
    constraints.emplace("padding_mask", padding);

    AttributeRule alibi;
    alibi.equals = AttrValue{false};
    constraints.emplace("alibi_mask", alibi);

    return constraints;
}

struct InstanceParams
{
    std::string name = "sdpa_smoke";
    std::string arch = "gfx942";
    std::int64_t headSize = 64;
    std::int64_t seqlenQ = 64;
    std::int64_t seqlenK = 64;
    std::int64_t numQueryHeads = 4;
    std::int64_t numKvHeads = 4;
    std::int64_t batchMin = 1;
    std::int64_t batchMax = 64;
};

inline AotInstance makeInstance(const InstanceParams& params)
{
    AotInstance instance;
    instance.name = params.name;
    instance.op = "sdpa_fwd";
    instance.arch = params.arch;
    instance.compileSpec.dtype = "fp16";
    instance.compileSpec.canonicalLayout = "BSHD";
    instance.compileSpec.seqlenQ = params.seqlenQ;
    instance.compileSpec.seqlenK = params.seqlenK;
    instance.compileSpec.numQueryHeads = params.numQueryHeads;
    instance.compileSpec.numKvHeads = params.numKvHeads;
    instance.compileSpec.headSize = params.headSize;
    instance.compileSpec.blockSizeQ = 16;
    instance.compileSpec.blockSizeK = 64;
    instance.compileSpec.tileSize   = 64; // 2D tile width T (tuning; tie-break only)
    instance.compileSpec.numWarps   = 1;  // wavefronts per tile (tuning; tie-break only)
    instance.compileSpec.maskMode = "none";
    instance.batch.min = params.batchMin;
    instance.batch.max = params.batchMax;
    instance.attributeConstraints = smokeAttributeConstraints();
    return instance;
}

// A problem that satisfies an instance built from equivalent InstanceParams.
inline SdpaProblem makeMatchingProblem(const InstanceParams& params, std::int64_t batch = 2)
{
    SdpaProblem problem;
    problem.op = "sdpa_fwd";
    problem.arch = params.arch;
    problem.dtype = "fp16";
    problem.layout = TensorLayout::BSHD;
    problem.batch = batch;
    problem.seqlenQ = params.seqlenQ;
    problem.seqlenK = params.seqlenK;
    problem.numQueryHeads = params.numQueryHeads;
    problem.numKvHeads = params.numKvHeads;
    problem.headSize = params.headSize;
    problem.maskMode = "none";
    problem.dropoutProbability = 0.0;
    problem.paddingMask = false;
    problem.alibiMask = false;
    problem.scalePolicy = "default_1_over_sqrt_d";
    return problem;
}

} // namespace rocke_client::dispatcher::test
