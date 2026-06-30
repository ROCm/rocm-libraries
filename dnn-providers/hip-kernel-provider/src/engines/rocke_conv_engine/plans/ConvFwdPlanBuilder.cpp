// Copyright © Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier:  MIT

#include "plans/ConvFwdPlanBuilder.hpp"
#include "core/Utils.hpp"
#include "plans/ConvFwdPlan.hpp"

#include <hip_kernel_provider_common/HipDeviceUtils.hpp>
#include <hipdnn_flatbuffers_sdk/data_objects/convolution_fwd_attributes_generated.h>
#include <hipdnn_flatbuffers_sdk/data_objects/data_types_generated.h>
#include <hipdnn_plugin_sdk/PluginLogging.hpp>

// rocKE C API — host-only headers; excluded from device compilation passes.
// Suppress warnings from rocKE headers (third-party, uses C-style casts).
#ifndef __HIP_DEVICE_COMPILE__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wold-style-cast"
#pragma clang diagnostic ignored "-Wcast-align"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wconversion"
#include <rocke/conv_ml_heuristic.h>
#include <rocke/instance_conv_implicit_gemm.h> // includes lower_llvm.h transitively
#pragma clang diagnostic pop
#endif

#include <hip/hiprtc.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <mutex>
#include <string>
#include <vector>

// The entire plan builder is host-only. Wrap in the HIP host guard so that
// hipcc's device compilation pass (which cannot see the rocKE headers) skips it.
#ifndef __HIP_DEVICE_COMPILE__

namespace rocke_conv_engine
{

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Returns the installed models directory for the rocKE conv heuristics.
// Set by cmake via -DROCKE_MODELS_DIR="..."; falls back to env var at runtime.
static std::string getModelsDir()
{
    if(const char* env = std::getenv("ROCKE_MODELS_DIR"))
    {
        return env;
    }
#ifdef ROCKE_MODELS_DIR
    return ROCKE_MODELS_DIR;
#else
    return "";
#endif
}

// NOLINTNEXTLINE(readability-function-cognitive-complexity)
static bool extractConvProblem(const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
                               rocke_conv_problem_t& prob,
                               int64_t& xUid,
                               int64_t& wUid,
                               int64_t& yUid,
                               std::string& logPrefix)
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;

    auto& node = opGraph.getNodeWrapper(0);
    auto& attrs = node.attributesAs<ConvolutionFwdAttributes>();
    auto& tensorMap = opGraph.getTensorMap();

    xUid = attrs.x_tensor_uid();
    wUid = attrs.w_tensor_uid();
    yUid = attrs.y_tensor_uid();

    auto* xT = tensorMap.at(xUid);
    auto* wT = tensorMap.at(wUid);

    // Expect 4-D NHWC tensors: x=[N,H,W,C], w=[K,Y,X,C]
    if(xT->dims()->size() != 4 || wT->dims()->size() != 4)
    {
        HIPDNN_PLUGIN_LOG_INFO(logPrefix << "tensors must be rank-4 (NCHW logical dims)");
        return false;
    }

    // Tensors use NCHW logical dim order (as required by the hipDNN frontend),
    // with NHWC-contiguous strides set separately. x: [N, C, Hi, Wi].
    int N = static_cast<int>(xT->dims()->Get(0));
    int C = static_cast<int>(xT->dims()->Get(1));
    int Hi = static_cast<int>(xT->dims()->Get(2));
    int Wi = static_cast<int>(xT->dims()->Get(3));
    // w: [K, C, Y, X]
    int K = static_cast<int>(wT->dims()->Get(0));
    int Y = static_cast<int>(wT->dims()->Get(2));
    int X = static_cast<int>(wT->dims()->Get(3));

    auto getFirst = [](const flatbuffers::Vector<int64_t>* v, int def) {
        return (v && v->size() > 0) ? static_cast<int>(v->Get(0)) : def;
    };
    int sH = getFirst(attrs.stride(), 1);
    int sW = (attrs.stride() && attrs.stride()->size() > 1)
                 ? static_cast<int>(attrs.stride()->Get(1))
                 : sH;
    int pH = getFirst(attrs.pre_padding(), 0);
    int pW = (attrs.pre_padding() && attrs.pre_padding()->size() > 1)
                 ? static_cast<int>(attrs.pre_padding()->Get(1))
                 : pH;
    int dH = getFirst(attrs.dilation(), 1);
    int dW = (attrs.dilation() && attrs.dilation()->size() > 1)
                 ? static_cast<int>(attrs.dilation()->Get(1))
                 : dH;

    prob = rocke_conv_problem_make(N, Hi, Wi, C, K, Y, X, sH, sW, pH, pW, dH, dW);
    return true;
}

// Patch make.buffer.rsrc intrinsic declarations for the container LLVM 23 build
// (ROCm 7.14, AMD clang 23.0.0git).
//
// This LLVM 23 build accepts ONLY:
//   declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1(
//       ptr addrspace(1), i16, i64, i32)
// with i64 num_records (verified by probe: p8.p1 i64 → OK, p8.p1 i32 → FAIL).
//
// rocKE LLVM20 emits: @...make.buffer.rsrc.p1, i32, with old attributes
// rocKE LLVM22 emits: @...make.buffer.rsrc.p8.p1, i64, with old attributes
//
// Target form: .p8.p1, i64, no parameter attributes on the pointer arg.
static std::string patchMakeBufferRsrc(const char* ir)
{
    std::string s(ir);
    auto replaceAll = [&](const std::string& from, const std::string& to) {
        for(size_t pos = 0; (pos = s.find(from, pos)) != std::string::npos; pos += to.size())
            s.replace(pos, from.size(), to);
    };

    // Step 1: LLVM22 path — already .p8.p1 with i64. Strip old attributes only.
    replaceAll("declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1("
               "ptr addrspace(1) nocapture readnone, i16, i64, i32)",
               "declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1("
               "ptr addrspace(1), i16, i64, i32)");

    // Step 2: LLVM20 path — .p1 with i32. Rename to .p8.p1 and widen i32 → i64.
    // Guard: only activate when the .p1 (non-.p8.p1) variant is present.
    const bool isLlvm20 = s.find("@llvm.amdgcn.make.buffer.rsrc.p1(") != std::string::npos;
    if(isLlvm20)
    {
        // Declare: rename + strip attributes + widen num_records.
        replaceAll("declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p1("
                   "ptr addrspace(1) nocapture readnone, i16, i32, i32)",
                   "declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1("
                   "ptr addrspace(1), i16, i64, i32)");
        replaceAll("declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p1("
                   "ptr addrspace(1), i16, i32, i32)",
                   "declare ptr addrspace(8) @llvm.amdgcn.make.buffer.rsrc.p8.p1("
                   "ptr addrspace(1), i16, i64, i32)");
        // Call sites: rename function.
        replaceAll("@llvm.amdgcn.make.buffer.rsrc.p1(", "@llvm.amdgcn.make.buffer.rsrc.p8.p1(");

        // Step A: swap i32->i64 at all make.buffer.rsrc call sites (covers literals and vars).
        replaceAll(", i16 0, i32 ", ", i16 0, i64 ");

        // Steps B+C: widen SSA variable args (%A_bytes etc.) and inject zext instructions.
        // Only needed if the IR actually uses these named params at the call site.
        // Rename call-site references to the widened alias to avoid type clash with
        // the still-i32 kernel params, then insert zext defs in the entry block.
        struct ByteParam
        {
            const char* name;
            const char* widened;
        };
        static constexpr ByteParam kParams[] = {
            {"%A_bytes", "%A_bytes_i64"},
            {"%B_bytes", "%B_bytes_i64"},
            {"%D_bytes", "%D_bytes_i64"},
        };
        std::string zexts;
        for(const auto& p : kParams)
        {
            const std::string callRef64Mid = std::string("i64 ") + p.name + ",";
            const std::string callRef64End = std::string("i64 ") + p.name + ")";
            if(s.find(callRef64Mid) != std::string::npos
               || s.find(callRef64End) != std::string::npos)
            {
                replaceAll(callRef64Mid, std::string("i64 ") + p.widened + ",");
                replaceAll(callRef64End, std::string("i64 ") + p.widened + ")");
                zexts += std::string("  ") + p.widened + " = zext i32 " + p.name + " to i64\n";
            }
        }

        if(!zexts.empty())
        {
            // Locate "entry:" followed by optional whitespace then newline.
            // LLVM IR guarantees "entry:" on its own line; we search for the newline
            // after the colon, tolerating trailing spaces or \r.
            const std::string entryLabel = "entry:";
            auto pos = s.find(entryLabel);
            if(pos != std::string::npos)
            {
                pos += entryLabel.size();
                // Skip any trailing whitespace (spaces, tabs, \r) to land on \n.
                while(pos < s.size() && s[pos] != '\n'
                      && (s[pos] == ' ' || s[pos] == '\t' || s[pos] == '\r'))
                    ++pos;
                if(pos < s.size() && s[pos] == '\n')
                    s.insert(pos + 1, zexts);
            }
        }
    }

    return s;
}

// Compile LLVM IR to a HIP module via the hipRTC linker API.
// HIPRTC_JIT_INPUT_LLVM_BITCODE accepts both LLVM bitcode and IR assembly text.
// Returns nullptr on failure.
static hipModule_t
    compileIrToModule(const char* llvmIr, const std::string& arch, const std::string& kernelName)
{
    // Normalise make.buffer.rsrc intrinsic to the form accepted by the container
    // LLVM 23 toolchain (AMD clang 23.0.0git / ROCm 7.14).
    const std::string patchedIr = patchMakeBufferRsrc(llvmIr);

    // Dump patched IR for diagnosis (first kernel only; path from ROCKE_CONV_IR_DUMP env var).
    {
        static std::once_flag dumpOnce;
        std::call_once(dumpOnce, [&] {
            const char* dumpPath = std::getenv("ROCKE_CONV_IR_DUMP");
            if(dumpPath && *dumpPath)
            {
                if(FILE* f = std::fopen(dumpPath, "w"))
                {
                    std::fwrite(patchedIr.data(), 1, patchedIr.size(), f);
                    std::fclose(f);
                    HIPDNN_PLUGIN_LOG_INFO("ConvFwdPlanBuilder: dumped patched IR to " << dumpPath);
                }
            }
        });
    }

    // Pass the target arch as IR→ISA compiler options via the hipRTC linker.
    // HIPRTC_JIT_IR_TO_ISA_OPT_EXT:       value is a const char** of compiler flags
    // HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT: value is the count cast to void* (JIT convention)
    const std::string gpuArchFlag = "--offload-arch=" + arch;
    const char* isaOpts[] = {gpuArchFlag.c_str()};

    hiprtcJIT_option optKeys[] = {HIPRTC_JIT_IR_TO_ISA_OPT_EXT, HIPRTC_JIT_IR_TO_ISA_OPT_COUNT_EXT};
    void* optVals[] = {const_cast<void*>(static_cast<const void*>(isaOpts)),
                       reinterpret_cast<void*>(static_cast<uintptr_t>(1))};

    hiprtcLinkState linkState = nullptr;
    hiprtcResult rr = hiprtcLinkCreate(2, optKeys, optVals, &linkState);
    if(rr != HIPRTC_SUCCESS)
    {
        HIPDNN_PLUGIN_LOG_ERROR(
            "ConvFwdPlanBuilder: hiprtcLinkCreate failed: " << hiprtcGetErrorString(rr));
        return nullptr;
    }

    // Feed IR assembly text through the LLVM_BITCODE path.
    // HIPRTC_JIT_INPUT_LLVM_BITCODE accepts both LLVM bitcode and IR assembly text.
    rr = hiprtcLinkAddData(linkState,
                           HIPRTC_JIT_INPUT_LLVM_BITCODE,
                           const_cast<void*>(static_cast<const void*>(patchedIr.data())),
                           patchedIr.size(),
                           kernelName.c_str(),
                           0,
                           nullptr,
                           nullptr);
    if(rr != HIPRTC_SUCCESS)
    {
        HIPDNN_PLUGIN_LOG_ERROR(
            "ConvFwdPlanBuilder: hiprtcLinkAddData failed: " << hiprtcGetErrorString(rr));
        hiprtcLinkDestroy(linkState);
        return nullptr;
    }

    void* hsacoData = nullptr;
    size_t hsacoSize = 0;
    rr = hiprtcLinkComplete(linkState, &hsacoData, &hsacoSize);
    if(rr != HIPRTC_SUCCESS)
    {
        HIPDNN_PLUGIN_LOG_ERROR(
            "ConvFwdPlanBuilder: hiprtcLinkComplete failed: " << hiprtcGetErrorString(rr));
        hiprtcLinkDestroy(linkState);
        return nullptr;
    }

    hipModule_t mod = nullptr;
    hipError_t herr = hipModuleLoadData(&mod, hsacoData);
    hiprtcLinkDestroy(linkState); // frees hsacoData
    if(herr != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR(
            "ConvFwdPlanBuilder: hipModuleLoadData failed: " << hipGetErrorString(herr));
        return nullptr;
    }
    return mod;
}

// ---------------------------------------------------------------------------
// isApplicable
// ---------------------------------------------------------------------------

bool ConvFwdPlanBuilder::isApplicable(
    const Handle& handle, const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph) const
{
    using namespace hipdnn_flatbuffers_sdk::data_objects;
    // NOLINTNEXTLINE(readability-identifier-naming)
    static const char* HIP_KERNEL_LOG_PREFIX = "[ConvFwdPlanBuilder::isApplicable] ";

    // Check arch
    std::string arch;
    try
    {
        arch = hip_kernel_provider_common::getDeviceString(handle.getStream());
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_ERROR("Could not query device string: " << e.what());
        return false;
    }
    HIP_KERNEL_RETURN_FALSE_IF(arch != "gfx942" && arch != "gfx950" && arch != "gfx90a",
                               "unsupported arch: " + arch);

    // Single conv-fwd node
    auto& nodes = opGraph.nodeWrappers();
    HIP_KERNEL_RETURN_FALSE_IF(nodes.size() != 1, "graph must have exactly one node");
    HIP_KERNEL_RETURN_FALSE_IF(nodes.front()->attributesType()
                                   != NodeAttributes::ConvolutionFwdAttributes,
                               "node is not ConvolutionFwdAttributes");

    // fp16 input only (rocKE conv heuristic is fp16-only)
    auto& tensorMap = opGraph.getTensorMap();
    auto& attrs = nodes.front()->attributesAs<ConvolutionFwdAttributes>();
    auto* xT = tensorMap.at(attrs.x_tensor_uid());
    HIP_KERNEL_RETURN_FALSE_IF(xT->data_type() != DataType::HALF,
                               "input tensor must be FP16 (got "
                                   + std::string(EnumNameDataType(xT->data_type())) + ")");

    // 2-D spatial only (rank-4 tensors)
    HIP_KERNEL_RETURN_FALSE_IF(xT->dims()->size() != 4,
                               "only 2-D (rank-4 NHWC) convolution is supported");

    // Model file must exist — lightweight path check to avoid parsing the full
    // LightGBM model here; the actual load and inference happen in buildPlan.
    const std::string modelsDir = getModelsDir();
    HIP_KERNEL_RETURN_FALSE_IF(modelsDir.empty(),
                               "ROCKE_MODELS_DIR not set; cannot locate conv heuristic model");

    const std::string modelPath
        = modelsDir + "/grouped_conv_forward_fp16_" + arch + "/model_tflops.lgbm";
    HIP_KERNEL_RETURN_FALSE_IF(!std::filesystem::exists(modelPath),
                               "conv heuristic model not found at " + modelPath
                                   + " (gunzip the .lgbm.gz first)");

    return true;
}

// ---------------------------------------------------------------------------
// getMaxWorkspaceSize
// ---------------------------------------------------------------------------

size_t ConvFwdPlanBuilder::getMaxWorkspaceSize(
    const Handle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    const Settings& /*executionSettings*/) const
{
    return 0;
}

// ---------------------------------------------------------------------------
// initializeExecutionSettings
// ---------------------------------------------------------------------------

void ConvFwdPlanBuilder::initializeExecutionSettings(
    const Handle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
    Settings& /*executionSettings*/) const
{
}

// ---------------------------------------------------------------------------
// buildPlan
// ---------------------------------------------------------------------------

void ConvFwdPlanBuilder::buildPlan(
    const Handle& handle,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& opGraph,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IEngineConfig& /*engineConfig*/,
    Context& executionContext) const
{
    // 1. Query arch and live device properties
    std::string arch;
    hipDeviceProp_t deviceProps{};
    try
    {
        deviceProps = hip_kernel_provider_common::getDeviceProperties(handle.getStream());
        const std::string archFull(deviceProps.gcnArchName);
        arch = archFull.substr(0, archFull.find(':'));
    }
    catch(const std::exception& e)
    {
        HIPDNN_PLUGIN_LOG_ERROR(
            "ConvFwdPlanBuilder::buildPlan: failed to query arch: " << e.what());
        return;
    }

    // 2. Extract problem from graph
    rocke_conv_problem_t prob{};
    int64_t xUid = 0, wUid = 0, yUid = 0;
    std::string prefix = "[ConvFwdPlanBuilder::buildPlan] ";
    if(!extractConvProblem(opGraph, prob, xUid, wUid, yUid, prefix))
    {
        return;
    }

    // 3. Load heuristic and select best tile config
    const std::string modelsDir = getModelsDir();
    const std::string modelDir = modelsDir + "/grouped_conv_forward_fp16_" + arch;
    rocke::ConvMLHeuristic heuristic(modelDir, deviceProps, "fp16");
    const bool useHeuristic = heuristic.is_loaded();
    if(!useHeuristic)
        HIPDNN_PLUGIN_LOG_WARN("ConvFwdPlanBuilder::buildPlan: heuristic not loaded from "
                               << modelDir << "; falling back to first valid tile config");

    // Enumerate candidate tile configs and rank them by predicted tflops.
    // We use a fixed candidate set covering the common tile sizes.
    struct TileCandidate
    {
        int tM, tN, tK, wM, wN;
    };
    static const TileCandidate kCandidates[] = {
        {128, 128, 32, 2, 2},
        {128, 64, 32, 2, 2},
        {64, 128, 32, 2, 2},
        {64, 64, 32, 2, 2},
        {128, 128, 64, 2, 4},
        {128, 256, 32, 2, 4},
        {256, 128, 32, 4, 2},
        {64, 256, 32, 2, 4},
        {256, 64, 32, 4, 2},
        {32, 64, 32, 1, 2},
        {64, 32, 32, 2, 1},
        {32, 32, 32, 1, 1},
    };

    // CDNA2 (gfx90a) and CDNA3 (gfx942) use the 32x32x8 MFMA atom for f16 (warp_tile_k=8).
    // CDNA4 (gfx950) and others use the 32x32x16 atom (warp_tile_k=16, the default).
    const int warpTileK = (arch == "gfx942" || arch == "gfx90a") ? 8 : 16;

    double bestScore = -1.0;
    TileCandidate best = kCandidates[0];

    for(const auto& c : kCandidates)
    {
        rocke_implicit_gemm_conv_spec_t spec = rocke_implicit_gemm_conv_spec_default();
        spec.problem = prob;
        spec.tile_m = c.tM;
        spec.tile_n = c.tN;
        spec.tile_k = c.tK;
        spec.warp_m = c.wM;
        spec.warp_n = c.wN;
        spec.warp_tile_k = warpTileK;

        char reason[256];
        if(!rocke_implicit_gemm_conv_is_valid_spec(&spec, arch.c_str(), reason, sizeof(reason)))
        {
            continue;
        }

        if(useHeuristic)
        {
            double score = heuristic.predict_tflops(prob, spec);
            if(score > bestScore)
            {
                bestScore = score;
                best = c;
            }
        }
        else if(bestScore < 0.0)
        {
            // No heuristic: take first valid candidate.
            bestScore = 0.0;
            best = c;
            break;
        }
    }

    if(bestScore < 0.0)
    {
        HIPDNN_PLUGIN_LOG_ERROR("ConvFwdPlanBuilder::buildPlan: no valid tile config found for "
                                << arch);
        return;
    }

    // 4. Build the chosen spec and lower to LLVM IR
    rocke_implicit_gemm_conv_spec_t spec = rocke_implicit_gemm_conv_spec_default();
    spec.problem = prob;
    spec.tile_m = best.tM;
    spec.tile_n = best.tN;
    spec.tile_k = best.tK;
    spec.warp_m = best.wM;
    spec.warp_n = best.wN;
    spec.warp_tile_k = warpTileK;

    char kNameBuf[512];
    rocke_implicit_gemm_conv_spec_kernel_name(&spec, kNameBuf, sizeof(kNameBuf));
    const std::string kernelName(kNameBuf);

    char* llText = nullptr;
    char err[512];
    rocke_status_t status = rocke_conv_implicit_gemm_lower_to_llvm(
        &spec, arch.c_str(), ROCKE_LLVM_FLAVOR_AUTO, &llText, err, sizeof(err));
    if(status != ROCKE_OK || llText == nullptr)
    {
        HIPDNN_PLUGIN_LOG_ERROR("ConvFwdPlanBuilder::buildPlan: rocKE lowering failed: " << err);
        return;
    }

    // 5. Compile IR to HIP module
    hipModule_t mod = compileIrToModule(llText, arch, kernelName);
    std::free(llText);
    if(mod == nullptr)
    {
        return;
    }

    hipFunction_t fn = nullptr;
    hipError_t herr = hipModuleGetFunction(&fn, mod, kernelName.c_str());
    if(herr != hipSuccess)
    {
        HIPDNN_PLUGIN_LOG_ERROR("ConvFwdPlanBuilder::buildPlan: hipModuleGetFunction failed for '"
                                << kernelName << "': " << hipGetErrorString(herr));
        (void)hipModuleUnload(mod);
        return;
    }

    // 6. Build params
    ConvFwdParams params;
    params.xUid = xUid;
    params.wUid = wUid;
    params.yUid = yUid;
    params.N = prob.N;
    params.C = prob.C;
    params.K = prob.K;
    params.Hi = prob.Hi;
    params.Wi = prob.Wi;
    params.Y = prob.Y;
    params.X = prob.X;
    params.Ho = rocke_conv_problem_ho(&prob);
    params.Wo = rocke_conv_problem_wo(&prob);
    params.strideH = prob.sH;
    params.strideW = prob.sW;
    params.padH = prob.pH;
    params.padW = prob.pW;
    params.dilH = prob.dH;
    params.dilW = prob.dW;
    params.tileM = best.tM;
    params.tileN = best.tN;
    params.tileK = best.tK;
    params.warpM = best.wM;
    params.warpN = best.wN;
    params.blockSize = static_cast<unsigned int>(rocke_implicit_gemm_conv_spec_block_size(&spec));

    // M = N * Ho * Wo (rows of the output), N_gemm = K (columns)
    const int M = rocke_conv_problem_m(&prob);
    params.gridM = static_cast<unsigned int>((M + best.tM - 1) / best.tM);
    params.gridN = static_cast<unsigned int>((prob.K + best.tN - 1) / best.tN);
    params.kernelName = kernelName;

    executionContext.setPlan(
        std::make_unique<ConvFwdPlan>(ConvModuleGuard{mod, fn}, std::move(params)));
}

// ---------------------------------------------------------------------------
// getCustomKnobs
// ---------------------------------------------------------------------------

std::vector<hipdnn_flatbuffers_sdk::data_objects::KnobT> ConvFwdPlanBuilder::getCustomKnobs(
    const Handle& /*handle*/,
    const hipdnn_flatbuffers_sdk::flatbuffer_utilities::IGraph& /*opGraph*/) const
{
    return {};
}

} // namespace rocke_conv_engine

#endif // __HIP_DEVICE_COMPILE__
