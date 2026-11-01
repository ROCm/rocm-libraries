/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/tensor_gett_planner.h -- C++ mirror of the tensor contraction product route of
 * dispatch/families/tensor_gett.py (plan_tensor_gett): for any f16 / bf16 contraction request on
 * gfx950, the tuned route table entry, else the cost model's choice between the MN-layout routes
 * (single MN-layout GETT launch on the GEMM dispatcher's CRR pick or the B3 ring, the pack hybrid
 * with tensor_pack copies, the ring-body GETT) and the K-contiguous route (both operands packed
 * K-contiguous, the GETT on the dispatcher's RCR pick), using the GEMM dispatcher and its time
 * model mirrored in rocke/gemm_dispatch.h. Decision-identical to the Python planner; the GETT
 * specs are kept on the full TraitSpec mirror (gemm_dispatch::Trait) so the planner sees what
 * Python sees, and converted to the C++ kernel builders' rocke::gett::TensorGettSpec when the
 * chosen bodies use no Python-engine-only lever.
 */
#ifndef ROCKE_TENSOR_GETT_PLANNER_H
#define ROCKE_TENSOR_GETT_PLANNER_H

#include "rocke/gemm_dispatch.h"
#include "rocke/instance_tensor_gett.h"

#include <map>
#include <string>
#include <vector>

namespace rocke
{
namespace gett
{
namespace planner
{

/* a contraction request (TensorGettRequest): row-major mode lists (last mode fastest), element
 * strides per mode (has_strides_* false: packed row-major), dtype "f16" / "fp16" / "bf16" */
struct Problem
{
    std::vector<std::string> modes_a, modes_b, modes_d;
    std::map<std::string, int64_t> extents;
    bool has_strides_a = false, has_strides_b = false, has_strides_d = false;
    std::map<std::string, int64_t> strides_a, strides_b, strides_d;
    std::string dtype = "f16";
    bool bilinear = false;
};

/* TensorGettSpec on the full TraitSpec mirror */
struct GettSpec
{
    std::vector<std::string> modes_a, modes_b, modes_d;
    std::map<std::string, int64_t> extents;
    bool has_strides_a = false, has_strides_b = false, has_strides_d = false;
    std::map<std::string, int64_t> strides_a, strides_b, strides_d;
    std::string dtype = "fp16";
    Knobs knobs;
    int tile_k = 32;
    int dtl_stages = 3;
    int persistent_ctas = 0;
    int chiplet_wgm = 8;
    int chiplet_chunk_size = 64;
    bool has_mn = false; /* mn_tile + mn_trait */
    gemm_dispatch::Tile mn_tile;
    gemm_dispatch::Trait mn_trait;
    bool has_kc = false; /* kc_tile + kc_trait */
    gemm_dispatch::Tile kc_tile;
    gemm_dispatch::Trait kc_trait;

    /* TensorGettSpec.plan() (false with the ValueError) */
    bool plan(Plan* out, std::string* why) const;
    /* TensorGettSpec.body() (name: kernel_name()) */
    gemm_dispatch::UGemm body() const;
    /* the asdict JSON / kernel_name of the Python TensorGettSpec */
    std::string json_key() const;
    std::string kernel_name() const;
};

/* tensor GETT is_valid_spec on the mirror */
bool is_valid(const GettSpec& spec, const std::string& arch, std::string* why);

/* the C++ kernel builders' spec of `spec` (false: a Python-engine-only lever is set) */
bool to_builder_spec(const GettSpec& spec, TensorGettSpec* out, std::string* why);

/* the launches of a route (TensorGettPlan) */
struct RoutePlan
{
    /* "single" / "pack_hybrid" (TensorGettPlan.kind) */
    std::string kind;
    /* the GETT candidate: tensor_gett_mn / tensor_gett_t32s3 / tensor_gett_kc */
    std::string candidate;
    std::vector<PackStep> packs;
    GettSpec gett;
    int grid[3] = {1, 1, 1};
    int block = 256;
    int64_t workspace_bytes = 0;
    /* "tuned" (route table), "cost" (cost model), "mn" (no K-contiguous route or no time
     * prediction), "kc" (no MN-layout route) */
    std::string source;
    /* the predicted times (us) of the two routes when both exist and the model predicts them */
    bool has_t_mn = false, has_t_kc = false;
    double t_mn = 0.0, t_kc = 0.0;
};

/* whether the planner serves `arch` (TENSOR_GETT_ARCHS: every other target is refused); a
 * caller can skip loading the selection models elsewhere */
bool plans_arch(const std::string& arch);

/* plan_tensor_gett(TensorGettRequest(problem, arch)): false (with the reason) when the Python
 * planner raises (no route for the request) */
bool plan_tensor_gett(const Problem& problem,
                      const std::string& arch,
                      const gemm_dispatch::Context& ctx,
                      RoutePlan* out,
                      std::string* why);

/* the predicted time (us) of a route (_plan_cost; false: None) */
bool plan_cost(const RoutePlan& plan,
               const std::string& arch,
               const gemm_dispatch::Context& ctx,
               double* out);

} // namespace planner
} // namespace gett
} // namespace rocke

#endif /* ROCKE_TENSOR_GETT_PLANNER_H */
