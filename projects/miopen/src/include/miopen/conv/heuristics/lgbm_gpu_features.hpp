#ifndef GUARD_MIOPEN_CONV_HEURISTICS_LGBM_GPU_FEATURES_HPP
#define GUARD_MIOPEN_CONV_HEURISTICS_LGBM_GPU_FEATURES_HPP

#include <array>
#include <cstddef>
#include <string_view>

namespace miopen {
namespace ai {
namespace lgbm {

// Number of training-time spec_ids. Matches model_meta.json
// rank.categorical_vocab.spec_id.
inline constexpr std::size_t kNumSpecIds = 9;

// Per-spec_id GPU descriptor block consumed by the LGBM rank model (v5,
// 59-feature pruned). Field order MUST match gen_gpu_table.py NUMERIC_FIELDS +
// CAT_FIELDS; the categorical fields hold integer codes into the model's
// categorical vocabularies (or -1 for missing).
struct GpuFeatures
{
    // 20 numeric features (indices 33..52 in the v5 feature_order).
    double cu_count;
    double wave_size;
    double simds_per_cu;
    double lds_size_per_cu_kb;
    double lds_size_per_workgroup_kb;
    double l1_cache_kb_per_cu;
    double l2_cache_total_kb;
    double l3_infinity_cache_kb;
    double boost_clock_mhz;
    double xcd_count;
    double shader_engines;
    double cacheline_size_bytes;
    double vram_bytes;
    double peak_tflops_fp64;
    double peak_tflops_fp32;
    double peak_tflops_fp16;
    double peak_tflops_bf16;
    double peak_tflops_fp8;
    double peak_tflops_int8;
    double mfma_shape_count;

    // 5 categorical codes (indices 53..57). spec_id_code doubles as the
    // index into kGpuTable.
    int gfx_id_code;
    int arch_family_code;
    int winograd_support_code;
    int asm_implicit_gemm_support_code;
    int spec_id_code;
};

extern const std::array<GpuFeatures, kNumSpecIds> kGpuTable;
extern const std::array<std::string_view, kNumSpecIds> kSpecIdNames;

// Returns the index into kGpuTable for a given (gfx_id, cu_count, vram_bytes)
// triple. Returns -1 when the SKU is not in the trained spec_id vocab; the
// picker should abstain in that case.
int ResolveSpecId(std::string_view gfx_id, std::size_t cu_count, std::size_t vram_bytes);

} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // GUARD_MIOPEN_CONV_HEURISTICS_LGBM_GPU_FEATURES_HPP
