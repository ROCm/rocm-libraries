/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * gemm_dispatch_internal.h -- the functions the C++ GEMM dispatcher (rocke/gemm_dispatch.h)
 * shares between its translation units. Each mirrors the Python function named in its comment.
 */
#ifndef ROCKE_GEMM_DISPATCH_INTERNAL_H
#define ROCKE_GEMM_DISPATCH_INTERNAL_H

#include "rocke/gemm_dispatch.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace rocke
{
namespace gemm_dispatch
{
namespace detail
{

/* ------------------------------------------------------------------- constants */

/* the rule thresholds and launch limits of the Python dispatcher (generated) */
#define ROCKE_GD_CONST(type, name, value) inline constexpr type name = value;
#include "gemm_dispatch_consts.inc"
#undef ROCKE_GD_CONST

/* -------------------------------------------------------------------- integers */

/* -(-a // b) for b > 0 (Python ceil division of ints) */
inline int64_t cdiv(int64_t a, int64_t b)
{
    int64_t q = a / b;
    int64_t r = a % b;
    return (r != 0 && ((r > 0) == (b > 0))) ? q + 1 : q;
}
/* Python a // b and a % b (floor semantics) */
inline int64_t fdiv(int64_t a, int64_t b)
{
    int64_t q = a / b;
    int64_t r = a % b;
    return (r != 0 && ((r < 0) != (b < 0))) ? q - 1 : q;
}
inline int64_t pmod(int64_t a, int64_t b)
{
    int64_t r = a % b;
    return (r != 0 && ((r < 0) != (b < 0))) ? r + b : r;
}
/* Python round(x) of a float: round half to even */
int64_t pround(double x);
/* math.floor(math.log2(max(1, x))) of a positive int */
int64_t floor_log2(int64_t x);
inline bool set_why(std::string* why, const std::string& m)
{
    if(why != nullptr)
        *why = m;
    return false;
}

/* -------------------------------------------------------------------- CU count */

/* helpers.split_k.arch_num_cus of the dispatch in flight (Context.cus; 256 outside one) */
int64_t current_cus();
struct CusScope
{
    explicit CusScope(int64_t cus);
    ~CusScope();
    int64_t prev;
};

/* ------------------------------------------------------------------- arch facts */

/* core.arch.ArchTarget facts of the dispatcher's targets (gfx950); false: unknown arch */
struct ArchFacts
{
    int wave_size;
    int async_lds_max_dwords;
    int64_t lds_capacity_bytes;
    int max_threads_per_block;
};
bool arch_facts(const std::string& arch, ArchFacts* out);
/* target.mma.has_shape / op_for_shape (family "mma", c_dtype fp32): the fragment lengths */
bool mma_shape(const std::string& arch,
               const std::string& dtype,
               int64_t m,
               int64_t n,
               int64_t k,
               int* a_frag,
               int* b_frag);
/* target.supports_dtype_combo(dtype, dtype, "fp32", family="mma") */
bool supports_dtype(const std::string& arch, const std::string& dtype);

/* ------------------------------------------------------------------- spec fields */

std::vector<FieldValue> trait_values(const Trait& t);
const std::vector<std::string>& trait_field_names();
/* the index of a TraitSpec field (-1: no such field) */
int trait_field_index(const std::string& name);
/* setattr(trait, name, value) (false: unknown field / value of another kind) */
bool set_trait_field(Trait* t, const std::string& name, const FieldValue& v);
FieldValue get_trait_field(const Trait& t, const std::string& name);
std::vector<std::pair<std::string, FieldValue>> skinny_fields(const Skinny& s);
/* str(v) as Python formats a dataclass field value ("True", "None", ...) */
std::string py_str(const FieldValue& v);
/* the value of one config-string item (selector._parse_value) */
FieldValue parse_value(const std::string& v);
bool field_eq(const FieldValue& a, const FieldValue& b);

/* ------------------------------------------------------------ gemm_universal rules */

/* helpers.spec.choose_load_vec (false: the Python ValueError) */
bool choose_load_vec(
    int64_t tile_m, int64_t tile_n, int64_t tile_k, int64_t block_size, int64_t* v);
/* gemm_universal.universal_lds_bytes (elem sizes 2) */
void universal_lds_bytes(const UGemm& s,
                         int64_t lds_cap,
                         int64_t eb_a,
                         int64_t eb_b,
                         int64_t eb_c,
                         int64_t* total,
                         int64_t* ab,
                         int64_t* c);
/* gemm_universal.ab_double_buffer_decision */
bool ab_double_buffer_decision(const UGemm& s, int64_t lds_cap, int64_t eb_a, int64_t eb_b);

/* gemm_mn_operand.MnLayout */
struct MnLayout
{
    int64_t tile_mn = 0, tile_k = 0, block_size = 0, wave_size = 0, warp_ext = 0;
    int64_t pad = 32, chunk = 0;
    bool sub_perm = false;
    int64_t W = 0, R = 0;
    std::vector<int64_t> row_bits, blk_bits;
    int64_t n_waves() const
    {
        return block_size / wave_size;
    }
    int64_t n_blocks() const
    {
        return tile_mn * tile_k * 2 / 1024;
    }
    int64_t n_pass() const
    {
        return n_blocks() / n_waves();
    }
    bool conflict_free() const;
};
/* MnLayout(...) with __post_init__ (false with the ValueError text) */
bool make_mn_layout(int64_t tile_mn,
                    int64_t tile_k,
                    int64_t block_size,
                    int64_t wave_size,
                    int64_t warp_ext,
                    int64_t pad,
                    int64_t chunk,
                    bool sub_perm,
                    MnLayout* out,
                    std::string* why);
bool uses_mn_dtl(const UGemm& s);
/* spec_mn_layouts(spec): (A, B) layouts (has_* false: None); false with the ValueError.
 * a_sub_perm: the GETT side channel's a_sub_perm (tensor GETT MN body with epi_v16). */
bool spec_mn_layouts(const UGemm& s,
                     bool a_sub_perm,
                     bool* has_a,
                     MnLayout* la,
                     bool* has_b,
                     MnLayout* lb,
                     std::string* why);
/* is_valid_spec with the GETT side channel's a_sub_perm / presence (the MN / K-contiguous GETT
 * bodies' spec._gett) */
bool is_valid_spec_x(
    const UGemm& s, const std::string& arch, bool gett, bool a_sub_perm, std::string* why);

/* dispatch.gemm.support */
bool universal_spec_supported(const UGemm& s, const std::string& arch, std::string* why);
bool request_shape_supported(const Request& r, const UGemm& s, std::string* why);

/* --------------------------------------------------------------------- gemm_skinny */

bool skinny_is_valid(const Skinny& s, const std::string& arch, std::string* why);
std::string skinny_launch_error(const Skinny& s, int64_t M, int64_t N, int64_t K, int64_t split);
int64_t skinny_steps(const Skinny& s, int64_t K);
int64_t skinny_kps(const Skinny& s, int64_t K, int64_t split);
int64_t skinny_lds_bytes(const Skinny& s);

/* ----------------------------------------------------------------------- rules */

/* common.native_layout_alternatives / native_layout_spec */
bool native_layout_alternatives(const Request& req,
                                const Context& ctx,
                                const UGemm& spec,
                                std::vector<std::pair<std::string, UGemm>>* out,
                                std::string* why);
bool native_layout_spec(
    const Request& req, const Context& ctx, const UGemm& spec, UGemm* out, std::string* why);
/* common.g1_adapt_spec */
UGemm g1_adapt_spec(const Request& req, const UGemm& spec);
/* common.agpr_ilv_big_grid_tile_trait (the B3 ring) */
void agpr_ilv_big_grid_tile_trait(const Context& ctx, Tile* tile, Trait* trait);

/* a registered candidate of the fp16 / bf16 registry */
struct Candidate
{
    std::string name;
    std::string algorithm;
    std::string spec_id;
    int priority = 0;
    bool opt_in = false;
    std::vector<std::string> arches;
    std::vector<std::string> layouts;
    int64_t max_m = -1; /* ShapeRange("M", max=...) (-1: none) */
    /* candidate.admits(request) */
    std::function<bool(const Request&, std::string*)> admits;
    /* candidate.select_spec(request) (requires admits) */
    std::function<bool(const Request&, Spec*, std::string*)> select;
    /* candidate.sweep_space(request) */
    std::function<std::vector<Spec>(const Request&)> sweep;
};

/* the registry of a dtype ("fp16" / "bf16"), candidates sorted by (priority, name) */
struct Registry
{
    std::string dtype;
    std::vector<Candidate> candidates;
    const Candidate* get(const std::string& name) const;
};
/* the registry of the request's dtype for a context (rules bound to the context) */
std::shared_ptr<Registry> make_registry(const std::string& dtype, const Context& ctx);
/* CandidateRegistry.select (false: "no candidate supports request") */
bool registry_select(const Registry& reg,
                     const Request& req,
                     const Candidate** out,
                     std::string* why);
/* CandidateRegistry.combos(req, include_opt_in=False) */
std::vector<std::pair<const Candidate*, Spec>> registry_combos(const Registry& reg,
                                                               const Request& req);

/* skinny.make_skinny_candidate (fallback: the region-free lowest-priority registration) */
Candidate make_skinny_candidate(const std::string& name,
                                const std::string& dtype,
                                bool fallback,
                                const Context& ctx);
/* skinny._ring_small_takes */
bool ring_small_takes(const Request& req);

/* the dispatcher state Python keeps in module globals (re-entry guards) */
struct DispatchState
{
    int kproxy_active = 0;
    int ring_split_bypass = 0;
};
DispatchState& dispatch_state();

/* ---------------------------------------------------------------------- skinny */

/* skinny.skinny_native / skinny_plan (skinny_select applied) */
Skinny skinny_native(const Skinny& spec, const Request& req);
Skinny skinny_plan(const Request& req, const Context& ctx);
bool skinny_plan_ok(const Skinny& s, const std::string& arch, int64_t M, int64_t N, int64_t K);
int64_t skinny_kalign(int64_t K);
int64_t skinny_split(const Skinny& spec, int64_t M, int64_t N, int64_t K, int64_t waves_target);
Skinny
    skinny_native_fit(const Skinny& spec, const std::string& arch, int64_t M, int64_t N, int64_t K);
/* skinny_select.select(req, plan) */
Skinny skinny_select(const Request& req, const Context& ctx, const Skinny& plan);
/* skinny_table.skinny_table_entry / skinny_native_table.skinny_native_entry
 * (false: no entry); 9 / 10 values */
bool skinny_table_entry(
    int64_t M, int64_t N, int64_t K, const std::string& dtype, std::array<int64_t, 9>* out);
bool skinny_native_entry(const std::string& arch,
                         int64_t M,
                         int64_t N,
                         int64_t K,
                         const std::string& layout,
                         bool ld_odd,
                         std::array<int64_t, 10>* out);

/* -------------------------------------------------------------------- selector */

/* selector.flatten_spec + feature_vector of one spec */
std::vector<double> feature_row(const SelectorModel& m,
                                const std::string& dtype,
                                int64_t M,
                                int64_t N,
                                int64_t K,
                                const Spec& spec);
const SelectorModel*
    load_model(const Context& ctx, const std::string& arch, const std::string& kind);
/* the model's feature layout (false: a feature this tree does not compute) */
bool make_feature_plan(SelectorModel* m, std::string* why);
/* selector.dispatch_override: the rule pick (cand, spec) replaced by the selector's pick */
void dispatch_override(const Request& req,
                       const Context& ctx,
                       const Registry& reg,
                       const Candidate*& cand,
                       Spec& spec);
/* selector.roofline_cycles(arch, M, N, K) */
double roofline_cycles(int64_t M, int64_t N, int64_t K);

} // namespace detail
} // namespace gemm_dispatch
} // namespace rocke

#endif /* ROCKE_GEMM_DISPATCH_INTERNAL_H */
