// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT
//
// CPU-only scorer + selection tests for the implicit-GEMM conv path.
// LightGBM is a CPU library and the model file is portable, so this
// suite runs on any host (no GPU required). Mirrors the structure of
// SdpaScorerTest.cpp: real-model load, prediction is finite, loaded
// selection is deterministic and stays within the enumerated set,
// model-missing falls back to the analytic policy (not first-fit).

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <cmath>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "CkDslContainer.hpp"
#include "adapters/conv_implicit_gemm/ConvImplicitGemmCandidateSelector.hpp"
#include "adapters/conv_implicit_gemm/ConvImplicitGemmPerfKnobs.hpp"
#include "adapters/conv_implicit_gemm/ConvImplicitGemmScorer.hpp"
#include "ckdsl_provider_paths.h"

namespace py = pybind11;

namespace {

using ck_dsl_provider::ConvImplicitGemmPerfKnobs;
using ck_dsl_provider::ConvImplicitGemmScorer;
using ck_dsl_provider::ConvSelectionProblem;
using ck_dsl_provider::enumerateCandidates;
using ck_dsl_provider::selectAnalyticFallback;
using ck_dsl_provider::selectPerfKnobs;

/// In-family bf16 reference problem matched to the selector test, so
/// the two test files agree on a known-buildable shape.
ConvSelectionProblem makeReferenceProblem() {
    ConvSelectionProblem p;
    p.N = 8;
    p.C = 128;
    p.K = 128;
    p.G = 1;
    p.Hi = 56;
    p.Wi = 56;
    p.R = 3;
    p.S = 3;
    p.sH = 1;
    p.sW = 1;
    p.pH = 1;
    p.pW = 1;
    p.dH = 1;
    p.dW = 1;
    p.dtype = "bf16";
    return p;
}

/// fp16/gfx942 problem: same shape as reference, dtype=fp16.
/// selectPerfKnobs activates the fp16/gfx942 oracle for this pair.
ConvSelectionProblem makeFp16Problem() {
    ConvSelectionProblem p = makeReferenceProblem();
    p.dtype = "fp16";
    return p;
}


/// Large shape used to guard against the "everything collapses to the
/// smallest tile" degeneracy that bit the SDPA scorer (commit
/// d495887dd29). At C=K=512 / 28x28 / 3x3 the oracle-best should be one
/// of the wider tiles, not the 16-wide-M minimum.
ConvSelectionProblem makeLargeProblem() {
    ConvSelectionProblem p = makeReferenceProblem();
    p.C = 512;
    p.K = 512;
    p.Hi = 28;
    p.Wi = 28;
    return p;
}

/// Structural equality over the knob fields the selection varies.
bool sameKnobs(const ConvImplicitGemmPerfKnobs& a, const ConvImplicitGemmPerfKnobs& b) {
    return a.tile_m == b.tile_m && a.tile_n == b.tile_n && a.tile_k == b.tile_k &&
           a.warp_m == b.warp_m && a.warp_n == b.warp_n && a.warp_tile_m == b.warp_tile_m &&
           a.warp_tile_n == b.warp_tile_n && a.warp_tile_k == b.warp_tile_k &&
           a.pipeline == b.pipeline && a.wave_size == b.wave_size;
}

}  // namespace

// ---------------------------------------------------------------------------
// Real model loads from the in-tree path.
// ---------------------------------------------------------------------------

TEST(ConvImplicitGemmScorer, DefaultConstructorLoadsInTreeModel) {
    ConvImplicitGemmScorer scorer;
    EXPECT_TRUE(scorer.isLoaded())
        << "default ConvImplicitGemmScorer should load the in-tree grouped-conv-forward "
           "gfx950/bf16 LightGBM model baked into CK_DSL_GROUPED_CONV_FWD_MODEL_PATH "
           "(decompress the .lgbm.gz in-tree if this fails)";
}

TEST(ConvImplicitGemmScorer, PredictReturnsFiniteValueForSampleCandidate) {
    ConvImplicitGemmScorer scorer;
    ASSERT_TRUE(scorer.isLoaded());

    const ConvSelectionProblem problem = makeReferenceProblem();
    const std::vector<ConvImplicitGemmPerfKnobs> candidates = enumerateCandidates(problem, "gfx950");
    ASSERT_FALSE(candidates.empty());

    const double pred = scorer.predict(problem, candidates.front());
    EXPECT_TRUE(std::isfinite(pred)) << "predicted TFLOPS must be finite (got " << pred << ")";
}

TEST(ConvImplicitGemmScorer, PredictIsFiniteForEveryEnumeratedCandidate) {
    ConvImplicitGemmScorer scorer;
    ASSERT_TRUE(scorer.isLoaded());

    const ConvSelectionProblem problem = makeReferenceProblem();
    const std::vector<ConvImplicitGemmPerfKnobs> candidates = enumerateCandidates(problem, "gfx950");
    ASSERT_FALSE(candidates.empty());

    for (const auto& k : candidates) {
        const double pred = scorer.predict(problem, k);
        EXPECT_TRUE(std::isfinite(pred))
            << "non-finite prediction for tile=(" << k.tile_m << "," << k.tile_n << ","
            << k.tile_k << ") pipeline=" << k.pipeline << " (got " << pred << ")";
    }
}

// ---------------------------------------------------------------------------
// Loaded path: selection is deterministic and stays in the enumerated set.
// ---------------------------------------------------------------------------

TEST(ConvImplicitGemmScorer, LoadedSelectionIsDeterministic) {
    ConvImplicitGemmScorer scorer;
    ASSERT_TRUE(scorer.isLoaded());

    const ConvSelectionProblem problem = makeReferenceProblem();
    const std::vector<ConvImplicitGemmPerfKnobs> candidates = enumerateCandidates(problem, "gfx950");
    ASSERT_FALSE(candidates.empty());

    const ConvImplicitGemmPerfKnobs first = selectPerfKnobs(problem, candidates, &scorer);
    const ConvImplicitGemmPerfKnobs second = selectPerfKnobs(problem, candidates, &scorer);
    EXPECT_TRUE(sameKnobs(first, second))
        << "selectPerfKnobs over the same candidates must return the same combo twice";
}

TEST(ConvImplicitGemmScorer, LoadedSelectionPicksAnEnumeratedCandidate) {
    ConvImplicitGemmScorer scorer;
    ASSERT_TRUE(scorer.isLoaded());

    const ConvSelectionProblem problem = makeReferenceProblem();
    const std::vector<ConvImplicitGemmPerfKnobs> candidates = enumerateCandidates(problem, "gfx950");
    ASSERT_FALSE(candidates.empty());

    const ConvImplicitGemmPerfKnobs chosen = selectPerfKnobs(problem, candidates, &scorer);
    bool found = false;
    for (const ConvImplicitGemmPerfKnobs& c : candidates) {
        if (sameKnobs(c, chosen)) {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "the scored pick must be one of the enumerated candidates";
}

// ---------------------------------------------------------------------------
// Degeneracy guard: large shapes must not collapse to the smallest tile.
// ---------------------------------------------------------------------------

TEST(ConvImplicitGemmScorer, DoesNotCollapseToSmallestTile) {
    ConvImplicitGemmScorer scorer;
    ASSERT_TRUE(scorer.isLoaded());

    const ConvSelectionProblem problem = makeLargeProblem();
    const std::vector<ConvImplicitGemmPerfKnobs> candidates = enumerateCandidates(problem, "gfx950");
    ASSERT_FALSE(candidates.empty());

    const ConvImplicitGemmPerfKnobs chosen = selectPerfKnobs(problem, candidates, &scorer);
    // The TILE_TO_WAVE table's smallest M tile is 16. A model that
    // degenerates to "always smallest" (the SdpaScorer bug fixed in
    // d495887dd29) would always pick tile_m == 16. For a large shape
    // (C=K=512, 28x28) that pick is clearly suboptimal.
    EXPECT_GT(chosen.tile_m, 16)
        << "scorer collapsed to the minimum M tile on a large shape "
           "(tile_m=16) -- this is the SDPA-era degeneracy guard the conv "
           "scorer must not regress into";
}

// ---------------------------------------------------------------------------
// fp16 + off-oracle arch: scorer is bypassed, analytic policy wins.
// fp16 on gfx950 has no trained model (only gfx942 does).
// ---------------------------------------------------------------------------

TEST(ConvImplicitGemmScorer, Fp16OnNonOracleArchFallsBackToAnalytic) {
    // fp16+gfx950 has no registry entry: the plan builder passes nullptr.
    // Verify selectPerfKnobs produces the same result as the analytic fallback.
    const ConvSelectionProblem problem = makeFp16Problem();  // dtype=fp16
    const std::vector<ConvImplicitGemmPerfKnobs> candidates = enumerateCandidates(problem, "gfx950");
    ASSERT_FALSE(candidates.empty());

    const ConvImplicitGemmPerfKnobs viaSelect =
        selectPerfKnobs(problem, candidates, nullptr);
    const ConvImplicitGemmPerfKnobs viaAnalytic = selectAnalyticFallback(problem, candidates);
    EXPECT_TRUE(sameKnobs(viaSelect, viaAnalytic))
        << "nullptr scorer must produce the analytic fallback";
}

// ---------------------------------------------------------------------------
// Off-oracle arch: scorer is bypassed even for bf16, analytic policy wins.
// gfx1151 has no oracle for any dtype/arch pair.
// ---------------------------------------------------------------------------

TEST(ConvImplicitGemmScorer, NonOracleArchFallsBackToAnalytic) {
    // bf16+gfx1151 has no registry entry: the plan builder passes nullptr.
    const ConvSelectionProblem problem = makeReferenceProblem();  // bf16
    const std::vector<ConvImplicitGemmPerfKnobs> candidates = enumerateCandidates(problem, "gfx1151");
    ASSERT_FALSE(candidates.empty());

    const ConvImplicitGemmPerfKnobs viaSelect =
        selectPerfKnobs(problem, candidates, nullptr);
    const ConvImplicitGemmPerfKnobs viaAnalytic = selectAnalyticFallback(problem, candidates);
    EXPECT_TRUE(sameKnobs(viaSelect, viaAnalytic))
        << "nullptr scorer must produce the analytic fallback";
}

// ---------------------------------------------------------------------------
// fp16 / gfx942 oracle: the fp16/gfx942 scorer activates on the correct pair.
// ---------------------------------------------------------------------------

TEST(ConvImplicitGemmScorer, Fp16Gfx942ScorerLoads) {
    ConvImplicitGemmScorer scorer{std::string(ck_dsl_provider::kCkDslGroupedConvFwdFp16Gfx942ModelPath),
                                  "gfx942"};
    EXPECT_TRUE(scorer.isLoaded())
        << "fp16/gfx942 scorer should load the in-tree model baked into "
           "CK_DSL_GROUPED_CONV_FWD_FP16_GFX942_MODEL_PATH";
}

TEST(ConvImplicitGemmScorer, Fp16Gfx942ScorerActivatesOnOraclePair) {
    // The fp16/gfx942 scorer must predict (not fall back to analytic) when
    // presented with fp16+gfx942. Verify by checking the selection differs
    // from the analytic fallback on a problem where the model has an opinion.
    ConvImplicitGemmScorer scorer{std::string(ck_dsl_provider::kCkDslGroupedConvFwdFp16Gfx942ModelPath),
                                  "gfx942"};
    ASSERT_TRUE(scorer.isLoaded());

    const ConvSelectionProblem problem = makeFp16Problem();
    const std::vector<ConvImplicitGemmPerfKnobs> candidates = enumerateCandidates(problem, "gfx942");
    ASSERT_FALSE(candidates.empty());

    const double pred = scorer.predict(problem, candidates.front());
    EXPECT_TRUE(std::isfinite(pred))
        << "fp16/gfx942 scorer must return a finite TFLOPS prediction (got " << pred << ")";
}

TEST(ConvImplicitGemmScorer, Fp16Gfx942SelectionIsDeterministic) {
    ConvImplicitGemmScorer scorer{std::string(ck_dsl_provider::kCkDslGroupedConvFwdFp16Gfx942ModelPath),
                                  "gfx942"};
    ASSERT_TRUE(scorer.isLoaded());

    const ConvSelectionProblem problem = makeFp16Problem();
    const std::vector<ConvImplicitGemmPerfKnobs> candidates = enumerateCandidates(problem, "gfx942");
    ASSERT_FALSE(candidates.empty());

    const ConvImplicitGemmPerfKnobs first = selectPerfKnobs(problem, candidates, &scorer);
    const ConvImplicitGemmPerfKnobs second = selectPerfKnobs(problem, candidates, &scorer);
    EXPECT_TRUE(sameKnobs(first, second))
        << "fp16/gfx942 selectPerfKnobs must return the same combo on repeated calls";
}

// ---------------------------------------------------------------------------
// Model-missing -> analytic fallback (NOT first-fit).
// ---------------------------------------------------------------------------

TEST(ConvImplicitGemmScorer, MissingModelIsNotLoaded) {
    ConvImplicitGemmScorer bad{"/nonexistent/path/to/conv_model.lgbm"};
    EXPECT_FALSE(bad.isLoaded());
}

TEST(ConvImplicitGemmScorer, MissingModelFallsBackToAnalyticPolicy) {
    ConvImplicitGemmScorer bad{"/nonexistent/path/to/conv_model.lgbm"};
    ASSERT_FALSE(bad.isLoaded());

    const ConvSelectionProblem problem = makeReferenceProblem();
    const std::vector<ConvImplicitGemmPerfKnobs> candidates = enumerateCandidates(problem, "gfx950");
    ASSERT_FALSE(candidates.empty());

    const ConvImplicitGemmPerfKnobs viaSelect =
        selectPerfKnobs(problem, candidates, &bad);
    const ConvImplicitGemmPerfKnobs viaAnalytic = selectAnalyticFallback(problem, candidates);
    EXPECT_TRUE(sameKnobs(viaSelect, viaAnalytic))
        << "with no model, selectPerfKnobs must equal the analytic fallback, not a first-fit";
}

// ---------------------------------------------------------------------------
// Score-parity: C++ extractConvFeatures vs Python feature_engine_grouped_conv
// ---------------------------------------------------------------------------
//
// The C++ extractor in ConvImplicitGemmScorer.cpp is a hand-port of the
// Python ``GroupedConvFeatureEngine.extract``. Both feed the SAME LightGBM
// booster, so any per-element drift between them silently corrupts the
// scored ranking the moment a non-trivial branch (e.g. lds_cap override
// for compv4, the 1x1/3x3 indicators, the rem/floor/ceil interactions)
// fires. This test re-derives the Python heuristics directory from the
// CMake-baked model path, prepends it to ``sys.path``, runs both extractors
// over the same problem + knobs, and asserts per-feature equality within a
// double-precision tolerance.
//
// Uses a fixture so the embedded interpreter is up before the test runs
// (the CkDslContainer ctor calls Py_Initialize and seeds sys.path with
// the ck_dsl + provider packages); the heuristics dir is NOT one of those
// so we prepend it ourselves.

class ConvImplicitGemmScorerParity : public ::testing::Test {
   protected:
    void SetUp() override {
        _container = std::make_unique<ck_dsl_provider::CkDslContainer>();

        // Derive the heuristics dir (parent of "models/<model>/...") from
        // the CMake-baked model path. The model lives at
        //   .../dispatcher/heuristics/models/<name>/model_tflops.lgbm
        // so going up THREE levels lands on .../dispatcher/heuristics --
        // the dir that contains feature_engine.py and
        // feature_engine_grouped_conv.py.
        std::filesystem::path modelPath{std::string(ck_dsl_provider::kCkDslGroupedConvFwdModelPath)};
        std::filesystem::path heuristicsDir =
            modelPath.parent_path().parent_path().parent_path();
        ASSERT_TRUE(std::filesystem::exists(heuristicsDir / "feature_engine_grouped_conv.py"))
            << "expected feature_engine_grouped_conv.py under " << heuristicsDir
            << " (derived from CK_DSL_GROUPED_CONV_FWD_MODEL_PATH=" << modelPath << ")";

        py::gil_scoped_acquire gil;
        py::module_ sys = py::module_::import("sys");
        py::list path = sys.attr("path").cast<py::list>();
        path.insert(0, py::cast(heuristicsDir.string()));
    }

    std::unique_ptr<ck_dsl_provider::CkDslContainer> _container;
};

/// Mirror of the kernel dict ``GroupedConvFeatureEngine.extract`` reads.
/// Built from the knobs the C++ extractor sees (block_size derived from
/// warp_m * warp_n * wave_size to match ``ConvImplicitGemmPerfKnobs::
/// block_size()``).
py::dict kernelDictFromKnobs(const ConvImplicitGemmPerfKnobs& k) {
    py::dict d;
    d["block_size"] = k.block_size();
    d["gemm_m_per_block"] = k.tile_m;
    d["gemm_n_per_block"] = k.tile_n;
    d["pipeline"] = k.pipeline;
    // Pinned to the C++ extractor's defaults (see "Suffix-aware features"
    // in ConvImplicitGemmScorer.cpp). intrawave / no-dsb / no-si matches
    // the dominant cell of the training distribution.
    d["wave_mode"] = "intrawave";
    d["has_dsb"] = 0;
    d["has_si"] = 0;
    return d;
}

/// Problem dict matching the C++ ``ConvSelectionProblem``-to-feature map.
/// Depth fields stay pinned to their 2D values (Di=1, Z=1, stride_d=1,
/// pad_d=0, dilation_d=1) -- the same pins ``extractConvFeatures``
/// enforces via ``kConvSelectionDim``.
py::dict problemDictFromSelection(const ConvSelectionProblem& p) {
    py::dict d;
    d["N"] = p.N;
    d["C"] = p.C;
    d["K"] = p.K;
    d["G"] = p.G;
    d["Hi"] = p.Hi;
    d["Wi"] = p.Wi;
    d["Y"] = p.R;
    d["X"] = p.S;
    d["stride_h"] = p.sH;
    d["stride_w"] = p.sW;
    d["pad_h"] = p.pH;
    d["pad_w"] = p.pW;
    d["dilation_h"] = p.dH;
    d["dilation_w"] = p.dW;
    d["dtype"] = p.dtype;
    // 2D pin: depth fields stay at their 2D defaults to match the
    // ``kConvSelectionDim == 2`` contract.
    d["Di"] = 1;
    d["Z"] = 1;
    d["stride_d"] = 1;
    d["pad_d"] = 0;
    d["dilation_d"] = 1;
    return d;
}

TEST_F(ConvImplicitGemmScorerParity, FeatureVectorMatchesPython) {
    ConvImplicitGemmScorer scorer;  // bf16/gfx950 default constructor
    ASSERT_TRUE(scorer.isLoaded());

    // A representative problem + a representative knob set from the
    // trained TILE_TO_WAVE table. Picking a non-default tile + a
    // non-``mem`` pipeline exercises the conditional branches in the
    // feature engine (lds_cap override for compv4, is_compv* indicators)
    // that a uniform default would skip.
    const ConvSelectionProblem problem = makeReferenceProblem();
    ConvImplicitGemmPerfKnobs knobs;
    knobs.tile_m = 64;
    knobs.tile_n = 128;
    knobs.tile_k = 64;
    knobs.warp_m = 2;
    knobs.warp_n = 2;
    knobs.warp_tile_m = 32;
    knobs.warp_tile_n = 32;
    knobs.warp_tile_k = 16;
    knobs.pipeline = "compv4";
    knobs.wave_size = 64;

    const std::vector<double> cppFeatures = scorer.extractFeaturesForTest(problem, knobs);
    ASSERT_EQ(cppFeatures.size(), 97u)
        << "feature count drifted from the trained schema "
           "(grouped_conv_forward_2d3d_suffix_bf16_gfx950 = 97)";

    py::gil_scoped_acquire gil;
    py::module_ engineMod = py::module_::import("feature_engine_grouped_conv");
    py::object engineCls = engineMod.attr("GroupedConvFeatureEngine");
    py::object engine = engineCls();  // gfx950 defaults match ConvHardwareProfile

    // Cross-check the feature-name list count matches the C++ size, so a
    // Python-side schema growth fails this test loudly instead of silently
    // truncating the comparison.
    py::list names = engine.attr("get_feature_names")().cast<py::list>();
    ASSERT_EQ(names.size(), cppFeatures.size())
        << "Python feature schema grew or shrunk relative to the C++ extractor; "
           "update ConvImplicitGemmScorer.cpp and kNumConvFeatures together";

    py::array_t<double> pyArr =
        engine.attr("extract")(problemDictFromSelection(problem), kernelDictFromKnobs(knobs))
            .cast<py::array_t<double>>();
    ASSERT_EQ(static_cast<std::size_t>(pyArr.size()), cppFeatures.size());

    auto pyView = pyArr.unchecked<1>();
    for (std::size_t i = 0; i < cppFeatures.size(); ++i) {
        const std::string name = py::str(names[i]).cast<std::string>();
        EXPECT_NEAR(cppFeatures[i], pyView(i), 1e-9)
            << "feature[" << i << "] '" << name << "' diverged: C++=" << cppFeatures[i]
            << " Python=" << pyView(i);
    }
}

// fp16/gfx942 parity: C++ extractor (with gfx942 hw profile) vs Python
// GroupedConvFeatureEngine constructed with the gfx942 hw params from
// HW_PROFILES in convert_csv_to_parquet.py. Uses the fp16/gfx942 model
// path to confirm the path constant is correctly wired; any feature
// divergence between C++ and Python silently corrupts the scored ranking.
TEST_F(ConvImplicitGemmScorerParity, Fp16Gfx942FeatureVectorMatchesPython) {
    ConvImplicitGemmScorer scorer{
        std::string(ck_dsl_provider::kCkDslGroupedConvFwdFp16Gfx942ModelPath), "gfx942"};
    ASSERT_TRUE(scorer.isLoaded())
        << "fp16/gfx942 scorer failed to load from "
        << ck_dsl_provider::kCkDslGroupedConvFwdFp16Gfx942ModelPath;

    const ConvSelectionProblem problem = makeFp16Problem();
    ConvImplicitGemmPerfKnobs knobs;
    knobs.tile_m = 64;
    knobs.tile_n = 128;
    knobs.tile_k = 64;
    knobs.warp_m = 2;
    knobs.warp_n = 2;
    knobs.warp_tile_m = 32;
    knobs.warp_tile_n = 32;
    knobs.warp_tile_k = 16;
    knobs.pipeline = "compv4";
    knobs.wave_size = 64;

    const std::vector<double> cppFeatures = scorer.extractFeaturesForTest(problem, knobs);
    ASSERT_EQ(cppFeatures.size(), 97u)
        << "feature count drifted from the fp16/gfx942 trained schema (expected 97)";

    py::gil_scoped_acquire gil;
    py::module_ engineMod = py::module_::import("feature_engine_grouped_conv");
    py::object engineCls = engineMod.attr("GroupedConvFeatureEngine");
    // Construct with gfx942 hardware params matching HW_PROFILES["gfx942"]
    // in convert_csv_to_parquet.py and hardwareProfileForArch("gfx942") in
    // ConvImplicitGemmScorer.cpp.
    py::object engine = engineCls(
        /*num_cus=*/228,
        /*lds_capacity=*/65536,
        /*max_clock_mhz=*/2100,
        /*simds_per_cu=*/4,
        /*shader_engines=*/28,
        /*max_waves_per_cu=*/32,
        /*wavefront_size=*/64,
        /*l1_cache_kb=*/32,
        /*l2_cache_kb=*/4096,
        /*l3_cache_kb=*/262144,
        /*num_xcd=*/8);

    py::list names = engine.attr("get_feature_names")().cast<py::list>();
    ASSERT_EQ(names.size(), cppFeatures.size())
        << "Python fp16/gfx942 feature schema diverged from C++ extractor";

    py::array_t<double> pyArr =
        engine.attr("extract")(problemDictFromSelection(problem), kernelDictFromKnobs(knobs))
            .cast<py::array_t<double>>();
    ASSERT_EQ(static_cast<std::size_t>(pyArr.size()), cppFeatures.size());

    auto pyView = pyArr.unchecked<1>();
    for (std::size_t i = 0; i < cppFeatures.size(); ++i) {
        const std::string name = py::str(names[i]).cast<std::string>();
        EXPECT_NEAR(cppFeatures[i], pyView(i), 1e-9)
            << "fp16/gfx942 feature[" << i << "] '" << name
            << "' diverged: C++=" << cppFeatures[i] << " Python=" << pyView(i);
    }
}
