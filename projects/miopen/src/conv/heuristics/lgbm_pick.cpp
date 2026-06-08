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
#include <cstring>
#include <limits>
#include <string>

namespace miopen {
namespace ai {
namespace lgbm {

namespace {

// Indices into the model's 69-feature row, matching model_meta.json
// rank.feature_order. Keep in sync with the appendix in
// ~/AutoResearchAllLGBM/CPP_PORT.md.
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
constexpr int kIdxTensorVectType   = 28;
constexpr int kIdxTensorVectLength = 29;
constexpr int kIdxDataType         = 30;
constexpr int kIdxDirection        = 31;
constexpr int kIdxInLayout         = 32;
constexpr int kIdxFilLayout        = 33;
constexpr int kIdxOutLayout        = 34;
constexpr int kIdxConvolutionMode  = 35;
constexpr int kIdxPadMode          = 36;
constexpr int kIdxCuCount               = 37;
constexpr int kIdxWaveSize              = 38;
constexpr int kIdxSimdsPerCu            = 39;
constexpr int kIdxMaxWavesPerCu         = 40;
constexpr int kIdxLdsSizePerCuKb        = 41;
constexpr int kIdxLdsSizePerWorkgroupKb = 42;
constexpr int kIdxL1CacheKbPerCu        = 43;
constexpr int kIdxL2CacheTotalKb        = 44;
constexpr int kIdxL3InfinityCacheKb     = 45;
constexpr int kIdxVgprPerSimd           = 46;
constexpr int kIdxSgprPerSimd           = 47;
constexpr int kIdxBoostClockMhz         = 48;
constexpr int kIdxXcdCount              = 49;
constexpr int kIdxShaderEngines         = 50;
constexpr int kIdxCachelineSizeBytes    = 51;
constexpr int kIdxVramBytes             = 52;
constexpr int kIdxPeakTflopsFp64        = 53;
constexpr int kIdxPeakTflopsFp32        = 54;
constexpr int kIdxPeakTflopsFp16        = 55;
constexpr int kIdxPeakTflopsBf16        = 56;
constexpr int kIdxPeakTflopsFp8         = 57;
constexpr int kIdxPeakTflopsFp4         = 58;
constexpr int kIdxPeakTflopsInt8        = 59;
constexpr int kIdxMfmaShapeCount        = 60;
constexpr int kIdxDtypeSupportCount     = 61;
constexpr int kIdxGfxId                 = 62;
constexpr int kIdxArchFamily            = 63;
constexpr int kIdxMatrixCoreGen         = 64;
constexpr int kIdxWinogradSupport       = 65;
constexpr int kIdxAsmImplicitGemmSupport= 66;
constexpr int kIdxSpecId                = 67;
constexpr int kIdxSolverName            = 68;

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

// Fill GPU feature block at indices 37..67 from the static lookup table.
void FillGpuFeatures(LgbmEntry* row, const GpuFeatures& g)
{
    SetNumeric(row[kIdxCuCount],               g.cu_count);
    SetNumeric(row[kIdxWaveSize],              g.wave_size);
    SetNumeric(row[kIdxSimdsPerCu],            g.simds_per_cu);
    SetNumeric(row[kIdxMaxWavesPerCu],         g.max_waves_per_cu);
    SetNumeric(row[kIdxLdsSizePerCuKb],        g.lds_size_per_cu_kb);
    SetNumeric(row[kIdxLdsSizePerWorkgroupKb], g.lds_size_per_workgroup_kb);
    SetNumeric(row[kIdxL1CacheKbPerCu],        g.l1_cache_kb_per_cu);
    SetNumeric(row[kIdxL2CacheTotalKb],        g.l2_cache_total_kb);
    SetNumeric(row[kIdxL3InfinityCacheKb],     g.l3_infinity_cache_kb);
    SetNumeric(row[kIdxVgprPerSimd],           g.vgpr_per_simd);
    SetNumeric(row[kIdxSgprPerSimd],           g.sgpr_per_simd);
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
    SetNumeric(row[kIdxPeakTflopsFp4],         g.peak_tflops_fp4);
    SetNumeric(row[kIdxPeakTflopsInt8],        g.peak_tflops_int8);
    SetNumeric(row[kIdxMfmaShapeCount],        g.mfma_shape_count);
    SetNumeric(row[kIdxDtypeSupportCount],     g.dtype_support_count);
    SetCategorical(row[kIdxGfxId],                  g.gfx_id_code);
    SetCategorical(row[kIdxArchFamily],             g.arch_family_code);
    SetCategorical(row[kIdxMatrixCoreGen],          g.matrix_core_gen_code);
    SetCategorical(row[kIdxWinogradSupport],        g.winograd_support_code);
    SetCategorical(row[kIdxAsmImplicitGemmSupport], g.asm_implicit_gemm_support_code);
    SetCategorical(row[kIdxSpecId],                 g.spec_id_code);
}

// Derive textbook FLOP/byte counts. These are inputs the perf-DB stored;
// the model trained on them but the exact upstream formula isn't documented
// in the cache. Standard conv definitions used here; precision differences
// vs the trained values may shift split decisions on a small fraction of
// problems near tree thresholds.
void FillDerivedNumeric(LgbmEntry* row, const conv::ProblemDescription& p)
{
    const double n  = static_cast<double>(p.GetBatchSize());
    const double c  = static_cast<double>(p.GetInChannels());
    const double k  = static_cast<double>(p.GetOutChannels());
    const double g  = static_cast<double>(p.GetGroupCount());
    const double od = static_cast<double>(p.GetOutDepth());
    const double oh = static_cast<double>(p.GetOutHeight());
    const double ow = static_cast<double>(p.GetOutWidth());
    const double kd = static_cast<double>(p.GetWeightsDepth());
    const double kh = static_cast<double>(p.GetWeightsHeight());
    const double kw = static_cast<double>(p.GetWeightsWidth());
    const double d  = static_cast<double>(p.GetInDepth());
    const double h  = static_cast<double>(p.GetInHeight());
    const double w  = static_cast<double>(p.GetInWidth());

    const double in_elems      = n * c * d * h * w;
    const double out_elems     = n * k * od * oh * ow;
    const double weights_elems = (g > 0 ? k * (c / g) : k * c) * kd * kh * kw;

    const double in_elem_sz  = static_cast<double>(p.GetInElementSize());
    const double w_elem_sz   = static_cast<double>(GetTypeSize(p.GetWeightsDataType()));
    const double out_elem_sz = static_cast<double>(GetTypeSize(p.GetOutDataType()));

    const double bytes_read      = in_elems * in_elem_sz + weights_elems * w_elem_sz;
    const double bytes_written   = out_elems * out_elem_sz;
    const double bytes_processed = bytes_read + bytes_written;
    const double flop_cnt        = 2.0 * g * (n * (k / std::max(g, 1.0)) * (c / std::max(g, 1.0)) *
                                       od * oh * ow) * kd * kh * kw;

    SetNumeric(row[kIdxFlopCnt],        flop_cnt);
    SetNumeric(row[kIdxBytesRead],      bytes_read);
    SetNumeric(row[kIdxBytesWritten],   bytes_written);
    SetNumeric(row[kIdxBytesProcessed], bytes_processed);
    SetNumeric(row[kIdxGflops],         flop_cnt / 1.0e9);
    SetNumeric(row[kIdxBandwidthGbps],  bytes_processed / 1.0e9);
}

// Fill problem feature block at indices 0..36 from the ProblemDescription.
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

    // tensor_vect_*: nonzero only for *_VECT_C layouts. GetVectorLength()
    // returns 1 for ordinary layouts.
    const auto in_layout = p.GetInLayout();
    const bool is_vect_c = in_layout.find("_VECT_C") != std::string::npos;
    SetNumeric(row[kIdxTensorVectType],   is_vect_c ? 1.0 : 0.0);
    SetNumeric(row[kIdxTensorVectLength], static_cast<double>(p.GetVectorLength()));

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
    // convolution_mode and pad_mode vocabularies in the trained model are
    // empty ([""]) — the training data never populated them. Always send
    // missing so the model takes its no-signal branch.
    SetCategorical(row[kIdxConvolutionMode], -1);
    SetCategorical(row[kIdxPadMode],         -1);
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
    FillGpuFeatures(row.data() + kIdxCuCount, gpu);

    // Score each candidate with the rank model. Higher raw score = predicted
    // faster (objective is lambdarank).
    std::vector<double> scores(candidates.size());
    for(std::size_t i = 0; i < candidates.size(); ++i)
    {
        SetCategorical(row[kIdxSolverName], meta.SolverCode(candidates[i]));
        double s = 0.0;
        lgbm_rank_predict(row.data(), /*pred_margin=*/0, &s);
        scores[i] = s;
    }

    // Top + runner-up indices.
    std::size_t top = 0;
    for(std::size_t i = 1; i < scores.size(); ++i)
        if(scores[i] > scores[top])
            top = i;
    std::size_t runner   = (top == 0 && scores.size() > 1) ? 1 : 0;
    for(std::size_t i = 0; i < scores.size(); ++i)
    {
        if(i == top)
            continue;
        if(scores[i] > scores[runner] || runner == top)
            runner = i;
    }

    // Margin gate (lambdarank objective): margin = 1 + max(top - runner, 0).
    if(scores.size() > 1)
    {
        const double margin = 1.0 + std::max(scores[top] - scores[runner], 0.0);
        if(margin < meta.MarginThresh(spec_id_str))
        {
            MIOPEN_LOG_I2("lgbm: abstain (margin " << margin << " < "
                                                    << meta.MarginThresh(spec_id_str) << ")");
            return {};
        }
    }

    // Applicability VETO on the top pick.
    SetCategorical(row[kIdxSolverName], meta.SolverCode(candidates[top]));
    double appl_prob = 0.0;
    lgbm_appl_predict(row.data(), /*pred_margin=*/0, &appl_prob);
    if(appl_prob < meta.ApplThresh(spec_id_str))
    {
        MIOPEN_LOG_I2("lgbm: abstain (appl_prob " << appl_prob << " < "
                                                   << meta.ApplThresh(spec_id_str) << ")");
        return {};
    }

    if(solver::Id id{candidates[top].c_str()}; id.IsValid())
        return id;
    // Training/runtime solver-name drift: rather than try the runner-up
    // (which would diverge from rules.py), abstain.
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
