#include <miopen/config.h>
#if MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK

#include <miopen/conv/heuristics/lgbm_pick.hpp>
#include <miopen/conv/heuristics/lgbm_metadata.hpp>
#include <miopen/conv/heuristics/lgbm_gpu_features.hpp>
#include <miopen/conv/heuristics/lgbm_predict.hpp>

#include <miopen/conv/problem_description.hpp>
#include <miopen/conv_algo_name.hpp>
#include <miopen/handle.hpp>
#include <miopen/logger.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <string>

namespace miopen {
namespace ai {
namespace lgbm {

namespace {

// Indices into the model's v5 59-feature row, matching model_meta.json
// rank.feature_order. Keep in sync with Appendix A in
// ~/AutoResearchAllLGBM/CPP_PORT.md. v5 dropped 10 features vs the original
// 69-feature schema (tensor_vect_type/length, convolution_mode, pad_mode,
// max_waves_per_cu, vgpr_per_simd, sgpr_per_simd, peak_tflops_fp4,
// dtype_support_count, matrix_core_gen).
constexpr int kIdxNMiniBatchSize = 0;
constexpr int kIdxChannels       = 1;
constexpr int kIdxDepth          = 2;
constexpr int kIdxHeight         = 3;
constexpr int kIdxWidth          = 4;
constexpr int kIdxOutputChannels = 5;
constexpr int kIdxDepthOutput    = 6;
constexpr int kIdxSpatialDim     = 7;
constexpr int kIdxNumDimensions  = 8;
constexpr int kIdxFilterHeightY  = 9;
constexpr int kIdxFilterWidthX   = 10;
constexpr int kIdxFilterDepthZ   = 11;
constexpr int kIdxPadHeight      = 12;
constexpr int kIdxPadWidth       = 13;
constexpr int kIdxPadDepth       = 14;
constexpr int kIdxStrideHeight   = 15;
constexpr int kIdxStrideWidth    = 16;
constexpr int kIdxStrideDepth    = 17;
constexpr int kIdxDilationHeight = 18;
constexpr int kIdxDilationWidth  = 19;
constexpr int kIdxDilationDepth  = 20;
constexpr int kIdxGroups         = 21;
constexpr int kIdxFlopCnt        = 22;
constexpr int kIdxBytesRead      = 23;
constexpr int kIdxBytesWritten   = 24;
constexpr int kIdxBytesProcessed = 25;
constexpr int kIdxGflops         = 26;
constexpr int kIdxBandwidthGbps  = 27;
constexpr int kIdxDataType       = 28;
constexpr int kIdxDirection      = 29;
constexpr int kIdxInLayout       = 30;
constexpr int kIdxFilLayout      = 31;
constexpr int kIdxOutLayout      = 32;
constexpr int kIdxCuCount               = 33;
constexpr int kIdxWaveSize              = 34;
constexpr int kIdxSimdsPerCu            = 35;
constexpr int kIdxLdsSizePerCuKb        = 36;
constexpr int kIdxLdsSizePerWorkgroupKb = 37;
constexpr int kIdxL1CacheKbPerCu        = 38;
constexpr int kIdxL2CacheTotalKb        = 39;
constexpr int kIdxL3InfinityCacheKb     = 40;
constexpr int kIdxBoostClockMhz         = 41;
constexpr int kIdxXcdCount              = 42;
constexpr int kIdxShaderEngines         = 43;
constexpr int kIdxCachelineSizeBytes    = 44;
constexpr int kIdxVramBytes             = 45;
constexpr int kIdxPeakTflopsFp64        = 46;
constexpr int kIdxPeakTflopsFp32        = 47;
constexpr int kIdxPeakTflopsFp16        = 48;
constexpr int kIdxPeakTflopsBf16        = 49;
constexpr int kIdxPeakTflopsFp8         = 50;
constexpr int kIdxPeakTflopsInt8        = 51;
constexpr int kIdxMfmaShapeCount        = 52;
constexpr int kIdxGfxId                  = 53;
constexpr int kIdxArchFamily             = 54;
constexpr int kIdxWinogradSupport        = 55;
constexpr int kIdxAsmImplicitGemmSupport = 56;
constexpr int kIdxSpecId                 = 57;
constexpr int kIdxSolverName             = 58;

// Treelite missing-marker. Generated header sets missing = -1 to indicate
// "present"; we mirror that in our LgbmEntry union.
inline void SetNumeric(LgbmEntry& e, double v)
{
    if(std::isnan(v))
        e.missing = -1;
    else
    {
        e.missing = 0;
        e.fvalue  = v;
    }
}

// Categorical features are passed as integer codes encoded into fvalue.
// A code of -1 means "value not in vocabulary"; the Treelite generator
// expects missing for unknown categoricals.
inline void SetCategorical(LgbmEntry& e, int code)
{
    if(code < 0)
        e.missing = -1;
    else
    {
        e.missing = 0;
        e.fvalue  = static_cast<double>(code);
    }
}

// Map MIOpen's conv::Direction enum to the perf-DB convention used to train
// the model: 1=Forward, 2=BackwardData, 4=BackwardWeights.
int DirectionPerfDbCode(conv::Direction d)
{
    switch(d)
    {
    case conv::Direction::Forward: return 1;
    case conv::Direction::BackwardData: return 2;
    case conv::Direction::BackwardWeights: return 4;
    }
    return 1;
}

std::string DataTypeName(miopenDataType_t t)
{
    switch(t)
    {
    case miopenHalf: return "fp16";
    case miopenFloat: return "fp32";
    case miopenBFloat16: return "bf16";
    case miopenInt8: return "int8";
    default: return "";
    }
}

// Fill GPU feature block at indices 33..57 from the static lookup table.
void FillGpuFeatures(LgbmEntry* row, const GpuFeatures& g)
{
    SetNumeric(row[kIdxCuCount],               g.cu_count);
    SetNumeric(row[kIdxWaveSize],              g.wave_size);
    SetNumeric(row[kIdxSimdsPerCu],            g.simds_per_cu);
    SetNumeric(row[kIdxLdsSizePerCuKb],        g.lds_size_per_cu_kb);
    SetNumeric(row[kIdxLdsSizePerWorkgroupKb], g.lds_size_per_workgroup_kb);
    SetNumeric(row[kIdxL1CacheKbPerCu],        g.l1_cache_kb_per_cu);
    SetNumeric(row[kIdxL2CacheTotalKb],        g.l2_cache_total_kb);
    SetNumeric(row[kIdxL3InfinityCacheKb],     g.l3_infinity_cache_kb);
    SetNumeric(row[kIdxBoostClockMhz],         g.boost_clock_mhz);
    SetNumeric(row[kIdxXcdCount],              g.xcd_count);
    SetNumeric(row[kIdxShaderEngines],         g.shader_engines);
    SetNumeric(row[kIdxCachelineSizeBytes],    g.cacheline_size_bytes);
    SetNumeric(row[kIdxVramBytes],             g.vram_bytes);
    SetNumeric(row[kIdxPeakTflopsFp64],        g.peak_tflops_fp64);
    SetNumeric(row[kIdxPeakTflopsFp32],        g.peak_tflops_fp32);
    SetNumeric(row[kIdxPeakTflopsFp16],        g.peak_tflops_fp16);
    SetNumeric(row[kIdxPeakTflopsBf16],        g.peak_tflops_bf16);
    SetNumeric(row[kIdxPeakTflopsFp8],         g.peak_tflops_fp8);
    SetNumeric(row[kIdxPeakTflopsInt8],        g.peak_tflops_int8);
    SetNumeric(row[kIdxMfmaShapeCount],        g.mfma_shape_count);
    SetCategorical(row[kIdxGfxId],                  g.gfx_id_code);
    SetCategorical(row[kIdxArchFamily],             g.arch_family_code);
    SetCategorical(row[kIdxWinogradSupport],        g.winograd_support_code);
    SetCategorical(row[kIdxAsmImplicitGemmSupport], g.asm_implicit_gemm_support_code);
    SetCategorical(row[kIdxSpecId],                 g.spec_id_code);
}

// Derive textbook FLOP/byte counts. These are inputs the perf-DB stored;
// the model trained on them but the exact upstream formula isn't documented
// flop_cnt / bytes_* / gflops / bandwidth_gbps are perf-DB-instrumented
// workload features. The DB's exact formula is not documented and, crucially,
// is direction-aware (the byte counts swap read/write tensors for Bwd-data and
// Bwd-weights) in a way a textbook forward estimate does not reproduce.
//
// We deliberately feed NaN ("missing") for all six. LightGBM treats NaN as a
// real branch direction (~1/3 of training rows had these as NaN), and a
// validation replay over deploy/test_vectors.json showed NaN reproduces the
// reference pick on 217/225 vectors vs only 211/225 for a textbook-derived
// estimate — wrong byte values actively mislead the trees more than missing
// values do. The residual 8 mismatches require the exact DB workload numbers,
// which are unavailable at runtime.
void FillDerivedNumeric(LgbmEntry* row, const conv::ProblemDescription& /*p*/)
{
    const double nan_v = std::numeric_limits<double>::quiet_NaN();
    SetNumeric(row[kIdxFlopCnt], nan_v);
    SetNumeric(row[kIdxBytesRead], nan_v);
    SetNumeric(row[kIdxBytesWritten], nan_v);
    SetNumeric(row[kIdxBytesProcessed], nan_v);
    SetNumeric(row[kIdxGflops], nan_v);
    SetNumeric(row[kIdxBandwidthGbps], nan_v);
}

// Fill problem feature block at indices 0..32 from the ProblemDescription.
void FillProblemFeatures(LgbmEntry* row,
                         const conv::ProblemDescription& p,
                         const LgbmMetadata& meta)
{
    SetNumeric(row[kIdxNMiniBatchSize], static_cast<double>(p.GetBatchSize()));
    SetNumeric(row[kIdxChannels],       static_cast<double>(p.GetInChannels()));
    SetNumeric(row[kIdxDepth],          static_cast<double>(p.GetInDepth()));
    SetNumeric(row[kIdxHeight],         static_cast<double>(p.GetInHeight()));
    SetNumeric(row[kIdxWidth],          static_cast<double>(p.GetInWidth()));
    SetNumeric(row[kIdxOutputChannels], static_cast<double>(p.GetOutChannels()));
    SetNumeric(row[kIdxDepthOutput],    static_cast<double>(p.GetOutDepth()));
    SetNumeric(row[kIdxSpatialDim],     static_cast<double>(p.GetSpatialDims()));
    SetNumeric(row[kIdxNumDimensions],  static_cast<double>(p.GetSpatialDims()) + 2.0);
    SetNumeric(row[kIdxFilterHeightY],  static_cast<double>(p.GetWeightsHeight()));
    SetNumeric(row[kIdxFilterWidthX],   static_cast<double>(p.GetWeightsWidth()));
    SetNumeric(row[kIdxFilterDepthZ],   static_cast<double>(p.GetWeightsDepth()));
    SetNumeric(row[kIdxPadHeight],      static_cast<double>(p.GetPadH()));
    SetNumeric(row[kIdxPadWidth],       static_cast<double>(p.GetPadW()));
    SetNumeric(row[kIdxPadDepth],       static_cast<double>(p.GetPadD()));
    SetNumeric(row[kIdxStrideHeight],   static_cast<double>(p.GetKernelStrideH()));
    SetNumeric(row[kIdxStrideWidth],    static_cast<double>(p.GetKernelStrideW()));
    SetNumeric(row[kIdxStrideDepth],    static_cast<double>(p.GetKernelStrideD()));
    SetNumeric(row[kIdxDilationHeight], static_cast<double>(p.GetDilationH()));
    SetNumeric(row[kIdxDilationWidth],  static_cast<double>(p.GetDilationW()));
    SetNumeric(row[kIdxDilationDepth],  static_cast<double>(p.GetDilationD()));
    SetNumeric(row[kIdxGroups],         static_cast<double>(p.GetGroupCount()));

    FillDerivedNumeric(row, p);

    SetCategorical(row[kIdxDataType],
                   meta.CategoricalCode("data_type", DataTypeName(p.GetInDataType())));
    SetCategorical(row[kIdxDirection],
                   meta.CategoricalCode("direction",
                                        std::to_string(DirectionPerfDbCode(p.GetDirection()))));
    SetCategorical(row[kIdxInLayout],
                   meta.CategoricalCode("in_layout", p.GetInLayout()));
    SetCategorical(row[kIdxFilLayout],
                   meta.CategoricalCode("fil_layout", p.GetWeightsLayout()));
    SetCategorical(row[kIdxOutLayout],
                   meta.CategoricalCode("out_layout", p.GetOutLayout()));
}

solver::Id Pick(const conv::ProblemDescription& problem, int spec_id_code)
{
    const auto& meta = LgbmMetadata::Get();
    if(!meta.IsReady())
        return {};

    if(spec_id_code < 0 || spec_id_code >= static_cast<int>(kNumSpecIds))
        return {};

    const auto& gpu = kGpuTable[spec_id_code];
    const std::string spec_id_str{kSpecIdNames[spec_id_code]};

    // Candidate solver list via triple_vocab (spec_id, direction, dtype).
    // Falls back to the full 52-solver list when the triple isn't covered
    // (e.g., a never-trained dtype on a known spec).
    const int direction_code = DirectionPerfDbCode(problem.GetDirection());
    const std::string dtype  = DataTypeName(problem.GetInDataType());
    const std::string key    = spec_id_str + "|" + std::to_string(direction_code) + "|" + dtype;

    const auto& triple_vocab = meta.TripleVocab();
    const auto vocab_it      = triple_vocab.find(key);
    const std::vector<std::string>& candidates =
        vocab_it != triple_vocab.end() ? vocab_it->second : meta.Solvers();

    if(candidates.empty())
        return {};

    // Build the constant problem+GPU prefix once; only solver_name varies
    // per candidate.
    std::array<LgbmEntry, kNumFeatures> row{};
    FillProblemFeatures(row.data(), problem, meta);
    FillGpuFeatures(row.data(), gpu);

    // Score each candidate with the rank model. Higher raw score = predicted
    // faster (objective is lambdarank). v5 has no margin gate and no
    // applicability VETO: argmax of the candidate scores is the pick.
    double best_score = -std::numeric_limits<double>::infinity();
    std::size_t top   = 0;
    for(std::size_t i = 0; i < candidates.size(); ++i)
    {
        SetCategorical(row[kIdxSolverName], meta.SolverCode(candidates[i]));
        double s = 0.0;
        lgbm_rank_predict(row.data(), /*pred_margin=*/0, &s);
        if(s > best_score)
        {
            best_score = s;
            top        = i;
        }
    }

    if(solver::Id id{candidates[top].c_str()}; id.IsValid())
        return id;
    // Training/runtime solver-name drift: the model recommended a solver this
    // MIOpen build doesn't know. Abstain rather than guess.
    MIOPEN_LOG_I2("lgbm: solver \"" << candidates[top]
                                     << "\" unknown to this MIOpen build; abstain");
    return {};
}

} // namespace

solver::Id PickSolver(const conv::ProblemDescription& problem, const Handle& handle)
{
    const std::string device_name = handle.GetDeviceName();
    const std::size_t cu          = handle.GetMaxComputeUnits();
    const std::size_t vram        = handle.GetGlobalMemorySize();
    const int spec_id_code        = ResolveSpecId(device_name, cu, vram);
    if(spec_id_code < 0)
    {
        MIOPEN_LOG_I2("lgbm: abstain (unknown spec_id for " << device_name << ", cu=" << cu
                                                            << ", vram=" << vram << ")");
        return {};
    }
    return Pick(problem, spec_id_code);
}

solver::Id PickSolverForSpec(const conv::ProblemDescription& problem, int spec_id_code)
{
    return Pick(problem, spec_id_code);
}

} // namespace lgbm
} // namespace ai
} // namespace miopen

#endif // MIOPEN_ENABLE_AI_IMMED_MODE_FALLBACK
