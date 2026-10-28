/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/instance_tensor_gett.h -- C++ mirror of the native tensor contraction
 * (GETT) family, rocke/instances/common/tensor/{gett_plan,gett}.py:
 *
 *   gett_plan.plan_gett / GettPlan        rocke::gett::plan_gett / Plan
 *   gett.GettKnobs / GettHooks            rocke::gett::Knobs / rocke_gett_hooks
 *   gett.TensorGettSpec, is_valid_spec,   rocke::gett::TensorGettSpec, is_valid_spec,
 *   build_tensor_gett, tensor_gett_grid   build_tensor_gett, tensor_gett_grid
 *
 * The hooks ride the universal GEMM spec as the `_gett` side channel (Python
 * spec._gett): NULL keeps every GEMM kernel unchanged. Mirrored bodies: the
 * ring body (the hook owns the operand addressing, the K-outer LDS layout, the
 * fragment reads and the epilogue), the MN-layout (CRR) bodies (the hook owns
 * the operand addressing and the epilogue; gemm_mn_operand owns the LDS
 * stages) and the K-contiguous (RCR) bodies on packed operands (the hook owns
 * the epilogue only: GettHooks.epilogue, or epilogue_rve on the row-interleave
 * vector epilogue). The store_nt epilogue is implemented by the Python engine
 * only: is_valid_spec refuses it with that reason. Also mirrored: the tensor
 * pack (tensor/pack.py) and the launches of the tuned and AOT route tables
 * (dispatch/families/tensor_gett.py _table_plan).
 */
#ifndef ROCKE_INSTANCE_TENSOR_GETT_H
#define ROCKE_INSTANCE_TENSOR_GETT_H

#include <stdbool.h>
#include <stdint.h>

#include "rocke/instance_gemm_mn_operand.h"
#include "rocke/instance_gemm_universal.h"
#include "rocke/ir.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The `_gett` side channel (opaque to C). */
struct rocke_gett_hooks;

/* GettHooks.a_sub_perm: A's transpose reads in QUAD_PERM order (MN body with
 * the 16-byte epilogue). */
bool rocke_gett_hooks_a_sub_perm(const struct rocke_gett_hooks* h);

/* The shipped tuned GETT route table (dispatch/families/tensor_gett_tuned.json,
 * compiled into the engine by CMake): NUL-terminated JSON text, the input of
 * rocke::gett::tuned_entries. */
const char* rocke_tensor_gett_tuned_json(void);

#ifdef __cplusplus
} /* extern "C" */

#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace rocke
{
namespace gett
{

using V = rocke_value_t;

/* (extent, stride) radix digit, inner first */
struct Digit
{
    int64_t e;
    int64_t s;
};

/* (radix, mode, mult) of the K split */
struct KDigit
{
    int64_t r;
    std::string m;
    int64_t mult;
};

/* A digit list: static digits, or a runtime digit space of the shape-generic
 * ring GETT (hiptensor_aot rgett: prefix + stride name, Python _Rt). */
struct Digits
{
    std::vector<Digit> d;
    bool rt = false;
    std::string prefix;
    std::string stride;
};

/* gett_plan.OpGeom */
struct OpGeom
{
    std::string name;
    std::vector<std::string> rows_order;
    int64_t rows = 0;
    Digits row_digits;
    int vec = 0;
    std::map<std::string, int64_t> strides;
    int64_t nbytes = 0;
};

/* gett_plan.GettPlan */
struct Plan
{
    std::vector<std::string> modes_a, modes_b, modes_d;
    std::map<std::string, int64_t> ext;
    bool swapped = false;
    int64_t M = 0, N = 0, K = 0;
    int tile_k = 0;
    std::vector<std::string> k_order;
    std::vector<KDigit> k_local;
    std::vector<KDigit> k_tile;
    OpGeom ops[2]; /* GEMM roles A, B */
    Digits d_row_digits;
    Digits d_col_digits;
    int vec_d = 1;
    std::map<std::string, int64_t> strides_d;
    bool vec_d8 = false;
    bool has_k_pad = false;
    std::string k_pad_mode;
    int64_t k_pad_real = 0;
    int64_t k_pad_padded = 0;
    bool kc = false;
    bool vec_dc8 = false;

    const OpGeom& op(char role) const
    {
        return ops[role == 'A' ? 0 : 1];
    }
    int64_t klocal(int64_t kl, char role) const;
    int64_t ktile(int64_t kt, char role) const;
};

/* gett_plan.plan_gett. Mode lists are row-major (last mode fastest); empty
 * stride maps are packed row-major. tile_k 0: try 64 then 32. False with the
 * ValueError message. */
bool plan_gett(const std::vector<std::string>& modes_a,
               const std::vector<std::string>& modes_b,
               const std::vector<std::string>& modes_d,
               const std::map<std::string, int64_t>& ext,
               const std::map<std::string, int64_t>* strides_a,
               const std::map<std::string, int64_t>* strides_b,
               const std::map<std::string, int64_t>* strides_d,
               int tile_k,
               bool kc,
               char d_fast_role,
               Plan* out,
               std::string* why);

/* gett_plan helpers */
std::map<std::string, int64_t> packed_strides(const std::vector<std::string>& modes,
                                              const std::map<std::string, int64_t>& ext);
std::vector<Digit> merge_digits(const std::vector<std::string>& order,
                                const std::map<std::string, int64_t>& ext,
                                const std::map<std::string, int64_t>& strides);

/* gett.GettKnobs */
struct Knobs
{
    int pad = 32;
    bool bilinear = false;
    std::string epi_order = "mi";
    bool epi_pair = false;
    bool lane_split = true;
    int acc_pin = 0;
    bool frag_pin = false;
    std::string llvm_opts;
    bool epi_v16 = false;
    int c_prefetch = 0;
    bool store_nt = false;
};

/* gett constants */
constexpr int64_t OOB = 0x7FFFFFF0;
constexpr int64_t NARROW_D_BYTES = OOB;
constexpr int64_t WIDE_OOB = 0xFFFFFF00;
constexpr int64_t MAX_D_BYTES = WIDE_OOB;
constexpr int64_t MAX_TENSOR_BYTES = 0x7FFF0000;

/* gett.BODY_TRAIT over the TraitSpec defaults, body_trait_errors,
 * mn_body_errors, mn_body_trait, body_spec */
rocke_gemm_trait_spec_t body_trait();
std::vector<std::string> body_trait_errors(const rocke_gemm_trait_spec_t& trait);
std::vector<std::string> mn_body_errors(const rocke_gemm_universal_spec_t& spec);
rocke_gemm_trait_spec_t mn_body_trait(const rocke_gemm_trait_spec_t& trait);
/* kc_body_errors, kc_body_trait: the K-contiguous (RCR) GETT bodies */
std::vector<std::string> kc_body_errors(const rocke_gemm_universal_spec_t& spec);
rocke_gemm_trait_spec_t kc_body_trait(const rocke_gemm_trait_spec_t& trait);
/* gemm_universal.launch_shape_error(spec, M, N, K) of an RCR body without an
 * operand prologue (the K-contiguous GETT bodies); empty when the launch is
 * inside the body's kernel contract. Another layout or a prologue is refused
 * with that reason. */
std::string gemm_launch_shape_error(const rocke_gemm_universal_spec_t& spec,
                                    int64_t M,
                                    int64_t N,
                                    int64_t K);
/* body_spec(dtype, tile_k, **trait_over): the overrides are applied by the
 * caller on the returned spec's trait (name "tensor_gett") */
rocke_gemm_universal_spec_t body_spec(const char* dtype, int tile_k);

/* (name, value) mode maps of a TensorGettSpec (Python ModeMap: sorted by name) */
using ModeMap = std::vector<std::pair<std::string, int64_t>>;

/* gett.TensorGettSpec (mode lists row-major: the last mode is the fastest).
 * kc_tile / kc_trait: a K-contiguous (RCR) body on packed operands
 * (kc_operand_modes). */
struct TensorGettSpec
{
    std::vector<std::string> modes_a, modes_b, modes_d;
    ModeMap extents;
    bool has_strides_a = false, has_strides_b = false, has_strides_d = false;
    ModeMap strides_a, strides_b, strides_d;
    std::string dtype = "fp16";
    Knobs knobs;
    int tile_k = 32;
    int dtl_stages = 3;
    int persistent_ctas = 0;
    int chiplet_wgm = 8;
    int chiplet_chunk_size = 64;
    bool has_mn_tile = false;
    rocke_gemm_tile_spec_t mn_tile{};
    bool has_mn_trait = false;
    rocke_gemm_trait_spec_t mn_trait{};
    bool has_kc_tile = false;
    rocke_gemm_tile_spec_t kc_tile{};
    bool has_kc_trait = false;
    rocke_gemm_trait_spec_t kc_trait{};

    /* TensorGettSpec.make: sorted mode maps */
    static TensorGettSpec make(const std::vector<std::string>& modes_a,
                               const std::vector<std::string>& modes_b,
                               const std::vector<std::string>& modes_d,
                               const std::map<std::string, int64_t>& extents);
    bool bilinear() const
    {
        return knobs.bilinear;
    }
    bool mn_body() const
    {
        return has_mn_tile;
    }
    bool kc_body() const
    {
        return has_kc_tile;
    }
    /* plan(); false with the ValueError message */
    bool plan(Plan* out, std::string* why) const;
    /* body(): the universal GEMM body spec (name = kernel_name(); the returned
     * spec's name points into `name_store`) */
    rocke_gemm_universal_spec_t body(std::string* name_store) const;
    std::string kernel_name() const;
    /* the asdict / json.dumps(sort_keys=True) text kernel_name() hashes */
    std::string json_key() const;
};

bool is_valid_spec(const TensorGettSpec& spec, const char* arch, std::string* why);

/* build_tensor_gett(spec, arch) into the (uninitialised) builder `b`; NULL
 * with the builder's error set on an invalid spec. The hooks are owned by
 * `hooks_out` (they must outlive the kernel's construction only). */
rocke_kernel_def_t*
    build_tensor_gett(rocke_ir_builder_t* b, const TensorGettSpec& spec, const char* arch);

void tensor_gett_grid(const TensorGettSpec& spec, int grid[3]);
int tensor_gett_block(const TensorGettSpec& spec);
/* launch_operands(spec): the user operands bound to the kernel's A and B */
std::pair<std::string, std::string> launch_operands(const TensorGettSpec& spec);
/* kc_operand_modes(spec): the row-major mode orders of the K-contiguous packed
 * copies the kernel's A and B params take ([GEMM rows, K modes]) */
std::pair<std::vector<std::string>, std::vector<std::string>>
    kc_operand_modes(const TensorGettSpec& spec);

/* dispatch/gemm/selector.parse_config(cfg) into a tile + trait (over the
 * TraitSpec defaults); false with the reason (unknown or Python-engine-only
 * trait fields included). */
bool parse_config(const std::string& cfg,
                  rocke_gemm_tile_spec_t* tile,
                  rocke_gemm_trait_spec_t* trait,
                  std::string* why);

/* dispatch/families/tensor_gett: one entry of the shipped tuned route table */
struct TunedEntry
{
    std::string key;
    std::string route;
    std::string body; /* the GEMM body as a selector config string */
    std::vector<std::string> modes_a, modes_b, modes_d;
    std::map<std::string, int64_t> extents;
    std::string dtype;
    bool bilinear = false;
};

/* the entries of `arch` in the tuned table file (JSON text), sorted by key */
bool tuned_entries(const std::string& json_text,
                   const std::string& arch,
                   std::vector<TunedEntry>* out,
                   std::string* why);

/* tensor_gett_route_key(TensorGettRequest(problem, arch)) */
std::string route_key(const TensorGettSpec& problem);

/* plan_tensor_gett(TensorGettRequest(problem, arch)) on the tuned table entry
 * `entry`: the single GETT launch the tuned route picks (no pack steps for the
 * "mn" route). False with the reason when the entry gives no plan. */
bool tuned_plan(const TunedEntry& entry,
                const std::string& arch,
                TensorGettSpec* out,
                std::string* why);

/* the request problem of a tuned entry (TensorGettSpec.make with the knobs'
 * bilinear) */
TensorGettSpec tuned_problem(const TunedEntry& entry);

/* SHA-1 / SHA-256 hex digests (kernel names and route keys) */
std::string sha1_hex(const std::string& s);
std::string sha256_hex(const std::string& s);

/* json.dumps text (ensure_ascii, sort_keys) of the values the kernel names
 * hash: a string, a list of strings, a ModeMap (list of [name, value] pairs) */
std::string json_str(const std::string& s);
std::string json_modes(const std::vector<std::string>& v);
std::string json_modemap(const ModeMap& m);

/* ------------------------------------------------------------- tensor pack
 * tensor/pack.py: Y[modes_dst] = X[modes_src], Y packed row-major in
 * modes_dst order, X with arbitrary element strides (forms pairs / tiled /
 * elements, "auto" = the first that applies). */

constexpr int64_t MAX_PACK_BYTES = 0x7FFF0000;

struct TensorPackSpec
{
    std::vector<std::string> modes_src, modes_dst;
    ModeMap extents;
    bool has_strides_src = false;
    ModeMap strides_src;
    std::string dtype = "fp16";
    int groups = 2;
    int block = 256;
    std::string form = "auto";

    /* TensorPackSpec.make: sorted mode maps; strides_src NULL = packed in
     * modes_src order */
    static TensorPackSpec make(const std::vector<std::string>& modes_src,
                               const std::vector<std::string>& modes_dst,
                               const std::map<std::string, int64_t>& extents,
                               const std::map<std::string, int64_t>* strides_src,
                               const std::string& dtype);
    std::map<std::string, int64_t> ext() const;
    /* false when a source mode has no stride (Python KeyError) */
    bool src_strides(std::map<std::string, int64_t>* out) const;
    int64_t numel() const;
    std::vector<Digit> src_digits() const;
    /* the source's unit-stride mode (extent > 1); false when there is none */
    bool unit_mode(std::string* out) const;
    bool pairs() const;
    /* (O modes, I modes) of the tiled form; false when it does not apply */
    bool tiled_groups(std::vector<std::string>* O, std::vector<std::string>* I) const;
    std::string resolved_form() const;
    std::string json_key() const;
    std::string kernel_name() const;
};

bool pack_is_valid_spec(const TensorPackSpec& spec, const char* arch, std::string* why);
/* build_tensor_pack(spec, arch) into the (uninitialised) builder `b`; NULL
 * with the builder's error set on an invalid spec. Params X (source), Y
 * (destination). */
rocke_kernel_def_t*
    build_tensor_pack(rocke_ir_builder_t* b, const TensorPackSpec& spec, const char* arch);
void tensor_pack_grid(const TensorPackSpec& spec, int grid[3]);
int tensor_pack_block(const TensorPackSpec& spec);

/* -------------------------------------------- route tables: the launches
 * dispatch/families/tensor_gett: TensorPackStep, TensorGettPlan and
 * _table_plan (the plan of a tuned or AOT route table entry: the MN-layout
 * GETT, single launch or pack hybrid, or the K-contiguous route with both
 * operands packed). The pack steps use the tensor_pack kernel
 * (TENSOR_GETT_PACK_KERNEL "tensor_pack") with the "local" pack order
 * (TENSOR_GETT_PACK_ORDER), the shipped defaults. */

struct PackStep
{
    std::string operand; /* the user operand copied: "A" or "B" */
    TensorPackSpec spec;
    int grid[3] = {1, 1, 1};
    int block = 256;
    int64_t workspace_bytes = 0;
};

struct TablePlan
{
    std::vector<PackStep> packs;
    TensorGettSpec gett;
    int64_t workspace_bytes = 0;
};

/* _table_plan(TensorGettRequest(tuned_problem(entry), arch), table): false
 * with the reason when the entry gives no plan (Python returns None). */
bool table_plan(const TunedEntry& entry, const std::string& arch, TablePlan* out, std::string* why);

/* gett_operand_sources(plan): for the GETT launch's A and B params, the pack
 * step index (>= 0) of a packed operand, or -1 for the user operand
 * (launch_operands names it). */
void gett_operand_sources(const TablePlan& plan, int src[2]);

} // namespace gett
} // namespace rocke

extern "C" {
/* The AOT route table (dispatch/families/tensor_gett_aot.json, compiled into
 * the engine by CMake): NUL-terminated JSON text in the tuned table's format,
 * the input of rocke::gett::tuned_entries. */
const char* rocke_tensor_gett_aot_json(void);
}

/* gett.GettHooks: the `spec._gett` side channel consumed by the universal GEMM
 * builder. Subclassed by the shape-generic ring GETT (rocke::htaot). */
struct rocke_gett_hooks
{
    using V = rocke_value_t;

    /* Python GettHooks(plan, knobs, spec, dtype). ok() false: the Python
     * constructor raised; error() is its message. */
    rocke_gett_hooks(const rocke::gett::Plan& plan,
                     const rocke::gett::Knobs& knobs,
                     const rocke_gemm_universal_spec_t& spec,
                     const char* dtype);
    virtual ~rocke_gett_hooks() = default;

    bool ok() const
    {
        return err_.empty();
    }
    const std::string& error() const
    {
        return err_;
    }

    rocke::gett::Plan plan;
    rocke::gett::Knobs k;
    const rocke_type_t* dt = nullptr;
    int bs = 0, ws = 0, T = 0, stages = 0;
    int bmn[2] = {0, 0}, wt[2] = {0, 0}, mf[2] = {0, 0}, wmn[2] = {0, 0};
    bool mn = false;
    bool kc = false;
    bool addr = true;
    bool a_sub_perm = false;

    /* geometry */
    bool v16_ok() const;
    int lane_bytes(char nm) const;
    int passes(char nm) const;
    int block_bytes() const;
    void lds_shape(char nm, int* shape) const;
    int wave_bytes(char nm) const;
    int pass_bytes(char nm) const;

    /* params / bind / addressing */
    virtual void add_params(rocke_ir_builder_t* b);
    void bind(rocke_ir_builder_t* b, V* tid, V* lane, V* warp_m_idx, V* warp_n_idx);
    virtual std::vector<V*> tile_voffs(rocke_ir_builder_t* b, char nm, V* mn0);
    virtual V* ktile_off_bytes(rocke_ir_builder_t* b, char nm, V* k_ld);
    int pass_soff_bytes(char nm, int p);
    V* frag(rocke_ir_builder_t* b, V* smem, char nm, int stage, int idx, int kk);
    std::vector<V*> tile_end(rocke_ir_builder_t* b, const std::vector<V*>& acc);
    void epilogue(rocke_ir_builder_t* b,
                  const std::vector<V*>& vals,
                  V* bm,
                  V* bn,
                  V* C,
                  V* warp_m_idx,
                  V* warp_n_idx,
                  V* lane,
                  const std::function<V*(V*)>& acc_out);
    /* epilogue_rve: the permuted alpha (/ beta) epilogue on the row-interleave
     * vector epilogue's lane layout (K-contiguous bodies): lane (q, cn) holds
     * rows row0 + v * mfmas_m + mi x columns col0 + ni (ni < mfmas_n = 8) as
     * f32 accumulators accf[mi * mfmas_n + ni][v] */
    void epilogue_rve(rocke_ir_builder_t* b,
                      const std::vector<V*>& accf,
                      V* row0,
                      V* col0,
                      int mfmas_m,
                      int mfmas_n,
                      int c_per_lane,
                      V* C);

protected:
    /* a D (C) offset part: the byte offset (narrow D) or the element offset
     * and its in-range predicate (wide D) */
    struct DPart
    {
        V* off = nullptr; /* narrow */
        V* e = nullptr; /* wide */
        V* ok = nullptr; /* wide (NULL: none) */
    };

    virtual V* digits(rocke_ir_builder_t* b, V* x, const rocke::gett::Digits& dg);
    V* digits_static(rocke_ir_builder_t* b, V* x, const std::vector<rocke::gett::Digit>& dg);
    virtual bool lane_split(char nm);
    virtual V* klocal_expr(rocke_ir_builder_t* b, char nm, V* k);
    virtual V* k_valid(rocke_ir_builder_t* b, V* k);
    virtual int64_t d_bytes() const;
    virtual bool wide_d() const;
    virtual DPart
        d_part(rocke_ir_builder_t* b, V* idx, const rocke::gett::Digits& dg, int64_t lim, V* big);
    V* d_off(rocke_ir_builder_t* b, const DPart& r, const DPart& c);
    std::pair<int64_t, int64_t> lane_geom(char nm, int p, int tid) const;
    std::pair<V*, V*> lane_expr(rocke_ir_builder_t* b, char nm, int p);
    std::vector<V*> tile_voffs_mn(rocke_ir_builder_t* b, char nm, V* mn0);
    void bind_mn(rocke_ir_builder_t* b);
    void epilogue_v16(rocke_ir_builder_t* b,
                      const std::vector<V*>& vals,
                      V* bm,
                      V* bn,
                      V* warp_m_idx,
                      V* warp_n_idx,
                      V* lane,
                      const std::function<V*(V*)>& acc_out,
                      V* rs_d,
                      V* rs_c,
                      V* big,
                      bool bf,
                      const std::vector<DPart>& cols16,
                      V* D_ptr);
    void epilogue_pair(rocke_ir_builder_t* b,
                       const std::vector<V*>& vals,
                       V* bm,
                       V* bn,
                       V* warp_m_idx,
                       V* warp_n_idx,
                       V* lane,
                       const std::function<V*(V*)>& acc_out,
                       V* rs_d,
                       V* rs_c,
                       V* big,
                       bool bf);
    /* the MN layouts of the body (mn mode) */
    bool has_lay[2] = {false, false};
    rocke_gemm_mn_layout_t lay[2];

    std::map<std::string, V*> p;
    V* tid_ = nullptr;
    V* lane_ = nullptr;
    V* widx[2] = {nullptr, nullptr};
    std::map<std::pair<char, int>, int> sh_;
    std::map<std::pair<char, int>, V*> fbase_;
    struct MnGeo
    {
        V* k;
        V* mn;
        V* kl;
        V* kok;
    };
    std::vector<MnGeo> mn_geo_[2];
    std::map<char, bool> split_cache_;
    std::string err_;
};

#endif /* __cplusplus */

#endif /* ROCKE_INSTANCE_TENSOR_GETT_H */
