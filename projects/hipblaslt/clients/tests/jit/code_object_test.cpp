// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Builds code objects through comgr without a GPU, and loads and launches them
// with --gpu (device 0) or --ffm (the simulator selected by HSA_MODEL_*).
// --bundle names a TensileLite bundle: a source bundle (sources/*.s and
// sources/Kernels.cpp) or a legacy bundle kept with --keep-build-tmp, whose
// clang-built code objects are then compared with the comgr-built ones.

#include "hipblaslt-jit-code-object.hpp"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace co = hipblaslt_jit::code_object;
namespace fs = std::filesystem;

namespace
{
    struct Config
    {
        std::string              target;
        fs::path                 out;
        bool                     gpu = false;
        bool                     ffm = false;
        std::vector<fs::path>    bundles;
        int                      tensileCov = 4;
        std::string              rocm;
        std::vector<std::string> only;
    };

    Config cfg;

    struct Outcome
    {
        bool        pass = false;
        std::string detail;
    };

    std::string readText(const fs::path& path)
    {
        std::ifstream     file(path, std::ios::binary);
        std::stringstream ss;
        ss << file.rdbuf();
        return ss.str();
    }

    std::vector<char> readBytes(const fs::path& path)
    {
        const auto text = readText(path);
        return {text.begin(), text.end()};
    }

    void writeBytes(const fs::path& path, const void* data, size_t size)
    {
        fs::create_directories(path.parent_path());
        std::ofstream(path, std::ios::binary).write(static_cast<const char*>(data), size);
    }

    void writeText(const fs::path& path, const std::string& text)
    {
        writeBytes(path, text.data(), text.size());
    }

    std::string describe(const co::CodeObjectMetadata& md)
    {
        std::ostringstream os;
        os << "isa=" << md.isaName << " target=" << md.target << " cov=" << md.codeObjectVersion
           << " kernels=" << md.kernels.size() << "\n";
        auto kernels = md.kernels;
        std::sort(kernels.begin(), kernels.end(), [](auto& a, auto& b) { return a.name < b.name; });
        for(const auto& k : kernels)
        {
            os << "kernel " << k.name << " symbol=" << k.symbol << " kernarg=" << k.kernargSegmentSize
               << " kernarg_align=" << k.kernargSegmentAlign << " lds=" << k.groupSegmentFixedSize
               << " scratch=" << k.privateSegmentFixedSize << " vgpr=" << k.vgprCount
               << " agpr=" << k.agprCount << " sgpr=" << k.sgprCount
               << " vgpr_spill=" << k.vgprSpillCount << " sgpr_spill=" << k.sgprSpillCount
               << " wave=" << k.wavefrontSize << " max_wg=" << k.maxFlatWorkgroupSize
               << " args=" << k.arguments.size() << "\n";
            for(const auto& a : k.arguments)
                os << "  arg " << a.offset << "+" << a.size << " " << a.valueKind << " " << a.name
                   << "\n";
        }
        return os.str();
    }

    co::Target target()
    {
        return co::Target::fromTargetId(cfg.target);
    }

    bool gfx12()
    {
        return target().gfx.rfind("gfx12", 0) == 0;
    }

    // Minimal kernel: out[i] = op(in[i]) for i < n; kernarg {in, out, n}.
    std::string asmKernel(const std::string& name, const std::string& op, int cov)
    {
        const std::string wave = gfx12() ? "32" : "64";
        std::ostringstream s;
        s << ".amdgcn_target \"amdgcn-amd-amdhsa--" << cfg.target << "\"\n"
          << ".text\n.globl " << name << "\n.p2align 8\n.type " << name << ",@function\n"
          << name << ":\n";
        if(gfx12())
        {
            s << "  s_load_b128 s[4:7], s[0:1], 0x0\n"
                 "  s_load_b32 s8, s[0:1], 0x10\n"
                 "  s_lshl_b32 s9, ttmp9, 6\n"
                 "  v_and_b32 v0, 0x3ff, v0\n"
                 "  v_add_nc_u32 v1, s9, v0\n"
                 "  s_wait_kmcnt 0\n"
                 "  v_cmp_gt_u32 vcc_lo, s8, v1\n"
                 "  s_nop 3\n"
                 "  s_and_saveexec_b32 s10, vcc_lo\n"
              << "  s_cbranch_execz .L" << name << "_done\n"
              << "  v_lshlrev_b32 v2, 2, v1\n"
                 "  global_load_b32 v3, v2, s[4:5]\n"
                 "  s_wait_loadcnt 0\n"
              << op << "  global_store_b32 v2, v3, s[6:7]\n";
        }
        else
        {
            s << "  s_load_dwordx4 s[4:7], s[0:1], 0x0\n"
                 "  s_load_dword s8, s[0:1], 0x10\n"
                 "  s_lshl_b32 s9, s2, 6\n"
                 "  v_add_u32 v1, s9, v0\n"
                 "  s_waitcnt lgkmcnt(0)\n"
                 "  v_cmp_gt_u32 vcc, s8, v1\n"
                 "  s_nop 4\n"
                 "  s_and_saveexec_b64 s[10:11], vcc\n"
              << "  s_cbranch_execz .L" << name << "_done\n"
              << "  v_lshlrev_b32 v2, 2, v1\n"
                 "  global_load_dword v3, v2, s[4:5]\n"
                 "  s_waitcnt vmcnt(0)\n"
              << op << "  global_store_dword v2, v3, s[6:7]\n";
        }
        s << ".L" << name << "_done:\n  s_endpgm\n.L" << name << "_end:\n  .size " << name << ", .L"
          << name << "_end-" << name << "\n\n.rodata\n.p2align 6\n.amdhsa_kernel " << name << "\n"
          << "  .amdhsa_user_sgpr_kernarg_segment_ptr 1\n  .amdhsa_user_sgpr_count 2\n"
          << "  .amdhsa_system_sgpr_workgroup_id_x 1\n  .amdhsa_kernarg_size 24\n"
          << "  .amdhsa_next_free_vgpr 4\n  .amdhsa_next_free_sgpr 12\n"
          << (gfx12() ? "  .amdhsa_wavefront_size32 1\n" : "  .amdhsa_accum_offset 4\n")
          << ".end_amdhsa_kernel\n\n.amdgpu_metadata\n---\namdhsa.version: [ 1, "
          << (cov == 4 ? 1 : 2) << " ]\namdhsa.kernels:\n"
          << "  - .name: " << name << "\n    .symbol: " << name << ".kd\n"
          << "    .kernarg_segment_size: 24\n    .kernarg_segment_align: 8\n"
          << "    .group_segment_fixed_size: 0\n    .private_segment_fixed_size: 0\n"
          << "    .wavefront_size: " << wave << "\n    .sgpr_count: 14\n    .vgpr_count: 4\n"
          << "    .agpr_count: 0\n    .max_flat_workgroup_size: 256\n    .args:\n"
          << "      - { .name: in, .size: 8, .offset: 0, .value_kind: global_buffer, "
             ".address_space: global }\n"
          << "      - { .name: out, .size: 8, .offset: 8, .value_kind: global_buffer, "
             ".address_space: global }\n"
          << "      - { .name: n, .size: 4, .offset: 16, .value_kind: by_value }\n"
          << "...\n.end_amdgpu_metadata\n";
        return s.str();
    }

    std::string scaleOp()
    {
        return gfx12() ? "  v_lshlrev_b32 v3, 1, v3\n  v_add_nc_u32 v3, 1, v3\n"
                       : "  v_lshlrev_b32 v3, 1, v3\n  v_add_u32 v3, 1, v3\n";
    }

    std::string add7Op()
    {
        return gfx12() ? "  v_add_nc_u32 v3, 7, v3\n" : "  v_add_u32 v3, 7, v3\n";
    }

    const char* hipHelperSource = R"(#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include "helper_math.h"

extern "C" __global__ void reduce_partials(const float* ws, __half* d, int n, int splits, float alpha)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    float sum = 0.f;
    for(int k = 0; k < splits; ++k)
        sum = helper::accumulate(sum, ws[static_cast<size_t>(k) * n + i]);
    d[i] = __float2half(alpha * sum);
}
)";

    const char* hipHelperInclude = R"(#pragma once
namespace helper
{
    __device__ inline float accumulate(float acc, float x) { return acc + x; }
}
)";

    const char* hipTripleSource = R"(#include <hip/hip_runtime.h>
extern "C" __global__ void triple(const int* in, int* out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n)
        out[i] = 3 * in[i];
}
)";

    co::Options baseOptions()
    {
        co::Options o;
        o.rocmPath = cfg.rocm;
        return o;
    }

    co::Options hipOptions()
    {
        co::Options o = baseOptions();
        o.includes.push_back({"helper_math.h", hipHelperInclude});
        return o;
    }

    // The options the JIT builder uses for TensileLite sources.
    co::Options tensileOptions()
    {
        co::Options o       = baseOptions();
        o.codeObjectVersion = cfg.tensileCov;
        o.retargetAssembly  = true;
        o.compilerFlags     = {"-D__HIP_HCC_COMPAT_MODE__=1"};
        o.linkerFlags       = {"-Xlinker", "--build-id=sha1"};
        return o;
    }

    // ---- GPU helpers ---------------------------------------------------------

    std::string hipCheck(hipError_t e, const char* what)
    {
        return e == hipSuccess ? std::string() : std::string(what) + ": " + hipGetErrorString(e);
    }

    struct Module
    {
        hipModule_t module = nullptr;
        ~Module()
        {
            if(module)
                (void)hipModuleUnload(module);
        }
    };

    std::string loadAndResolve(const std::vector<char>&              elf,
                               const std::vector<std::string>&       symbols,
                               Module&                               m,
                               std::map<std::string, hipFunction_t>* functions = nullptr)
    {
        if(auto e = hipCheck(hipModuleLoadData(&m.module, elf.data()), "hipModuleLoadData");
           !e.empty())
            return e;
        for(const auto& s : symbols)
        {
            hipFunction_t f = nullptr;
            if(auto e = hipCheck(hipModuleGetFunction(&f, m.module, s.c_str()),
                                 ("hipModuleGetFunction(" + s + ")").c_str());
               !e.empty())
                return e;
            if(functions)
                (*functions)[s] = f;
        }
        return {};
    }

    // Launches a {in, out, n} int kernel and checks out[i] == expect(in[i]).
    std::string runIntKernel(hipFunction_t f, const std::function<int(int)>& expect)
    {
        const int        n = 5000;
        std::vector<int> in(n), out(n, -1);
        for(int i = 0; i < n; ++i)
            in[i] = (i * 37) % 1001 - 500;
        int *       dIn = nullptr, *dOut = nullptr;
        std::string err = hipCheck(hipMalloc(&dIn, n * sizeof(int)), "hipMalloc");
        if(err.empty())
            err = hipCheck(hipMalloc(&dOut, n * sizeof(int)), "hipMalloc");
        if(err.empty())
            err = hipCheck(hipMemcpy(dIn, in.data(), n * sizeof(int), hipMemcpyHostToDevice),
                           "hipMemcpy");
        if(err.empty())
            err = hipCheck(hipMemset(dOut, 0xff, n * sizeof(int)), "hipMemset");
        struct
        {
            const int* in;
            int*       out;
            int        n;
            int        pad;
        } args{dIn, dOut, n, 0};
        size_t argSize  = sizeof(args);
        void*  config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                           &args,
                           HIP_LAUNCH_PARAM_BUFFER_SIZE,
                           &argSize,
                           HIP_LAUNCH_PARAM_END};
        if(err.empty())
            err = hipCheck(
                hipModuleLaunchKernel(f, (n + 63) / 64, 1, 1, 64, 1, 1, 0, nullptr, nullptr, config),
                "hipModuleLaunchKernel");
        if(err.empty())
            err = hipCheck(hipDeviceSynchronize(), "hipDeviceSynchronize");
        if(err.empty())
            err = hipCheck(hipMemcpy(out.data(), dOut, n * sizeof(int), hipMemcpyDeviceToHost),
                           "hipMemcpy");
        if(dIn)
            (void)hipFree(dIn);
        if(dOut)
            (void)hipFree(dOut);
        if(!err.empty())
            return err;
        int bad = 0;
        for(int i = 0; i < n; ++i)
            bad += out[i] != expect(in[i]);
        return bad ? std::to_string(bad) + " of " + std::to_string(n) + " outputs wrong" : "";
    }

    float halfToFloat(uint16_t h)
    {
        const uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1f, man = h & 0x3ff;
        float          v;
        if(exp == 0)
            v = std::ldexp(static_cast<float>(man), -24);
        else if(exp == 31)
            v = man ? NAN : INFINITY;
        else
            v = std::ldexp(static_cast<float>(man | 0x400), static_cast<int>(exp) - 25);
        return sign ? -v : v;
    }

    std::string runReduce(hipFunction_t f)
    {
        const int          n = 4096, splits = 4;
        const float        alpha = 0.5f;
        std::vector<float> ws(static_cast<size_t>(n) * splits);
        for(size_t i = 0; i < ws.size(); ++i)
            ws[i] = static_cast<float>((i * 13) % 97) / 16.f - 3.f;
        float*      dWs = nullptr;
        uint16_t*   dD  = nullptr;
        std::string err = hipCheck(hipMalloc(&dWs, ws.size() * sizeof(float)), "hipMalloc");
        if(err.empty())
            err = hipCheck(hipMalloc(&dD, n * sizeof(uint16_t)), "hipMalloc");
        if(err.empty())
            err = hipCheck(
                hipMemcpy(dWs, ws.data(), ws.size() * sizeof(float), hipMemcpyHostToDevice),
                "hipMemcpy");
        struct
        {
            const float* ws;
            uint16_t*    d;
            int          n;
            int          splits;
            float        alpha;
            int          pad;
        } args{dWs, dD, n, splits, alpha, 0};
        size_t argSize  = sizeof(args);
        void*  config[] = {HIP_LAUNCH_PARAM_BUFFER_POINTER,
                           &args,
                           HIP_LAUNCH_PARAM_BUFFER_SIZE,
                           &argSize,
                           HIP_LAUNCH_PARAM_END};
        if(err.empty())
            err = hipCheck(
                hipModuleLaunchKernel(f, n / 256, 1, 1, 256, 1, 1, 0, nullptr, nullptr, config),
                "hipModuleLaunchKernel");
        if(err.empty())
            err = hipCheck(hipDeviceSynchronize(), "hipDeviceSynchronize");
        std::vector<uint16_t> d(n);
        if(err.empty())
            err = hipCheck(hipMemcpy(d.data(), dD, n * sizeof(uint16_t), hipMemcpyDeviceToHost),
                           "hipMemcpy");
        if(dWs)
            (void)hipFree(dWs);
        if(dD)
            (void)hipFree(dD);
        if(!err.empty())
            return err;
        double maxErr = 0;
        for(int i = 0; i < n; ++i)
        {
            float ref = 0;
            for(int k = 0; k < splits; ++k)
                ref += ws[static_cast<size_t>(k) * n + i];
            ref *= alpha;
            maxErr = std::max(maxErr,
                              std::fabs(static_cast<double>(halfToFloat(d[i]) - ref))
                                  / std::max(1.0, std::fabs(static_cast<double>(ref))));
        }
        return maxErr <= 1e-3 ? "" : "max relative error " + std::to_string(maxErr);
    }

    // ---- Tensile bundle discovery ----------------------------------------------

    struct Bundle
    {
        std::string           name;
        fs::path              assembly;
        fs::path              helperSource;
        std::vector<fs::path> headers;
        fs::path              referenceCo; // legacy bundles only
        fs::path              referenceHsaco; // legacy bundles with helpers only
    };

    std::vector<fs::path> filesIn(const fs::path& dir, const char* extension)
    {
        std::vector<fs::path> paths;
        if(fs::is_directory(dir))
            for(auto& e : fs::directory_iterator(dir))
                if(e.is_regular_file() && e.path().extension() == extension)
                    paths.push_back(e.path());
        std::sort(paths.begin(), paths.end());
        return paths;
    }

    Bundle discover(const fs::path& dir)
    {
        Bundle b;
        b.name             = dir.parent_path().filename().string();
        const auto gfx     = target().gfx;
        fs::path   sources = dir / "sources", assembly = sources;
        if(!fs::is_directory(sources))
        {
            sources = dir;
            assembly.clear();
            for(auto& e : fs::directory_iterator(dir / "build_tmp"))
                assembly = e.path() / "assembly";
            const auto library = dir / "library" / gfx;
            b.referenceCo      = library / ("TensileLibrary_" + gfx + ".co");
            for(const auto& path : filesIn(library, ".hsaco"))
                b.referenceHsaco = path;
        }
        const auto asmFiles = filesIn(assembly, ".s");
        if(asmFiles.size() != 1)
            throw std::runtime_error("expected one .s in " + assembly.string());
        b.assembly = asmFiles.front();
        if(fs::exists(sources / "Kernels.cpp"))
            b.helperSource = sources / "Kernels.cpp";
        b.headers = filesIn(sources, ".h");
        return b;
    }

    // The kernel named by the first ".amdhsa_kernel <name>" directive.
    std::string mainKernel(const std::string& assembly)
    {
        const std::string directive = ".amdhsa_kernel ";
        const auto        at        = assembly.find(directive);
        if(at == std::string::npos)
            return {};
        const auto begin = at + directive.size();
        return assembly.substr(begin, assembly.find_first_of(" \t\r\n", begin) - begin);
    }

    std::string compareMetadata(const co::CodeObjectMetadata& ref, const co::CodeObjectMetadata& got)
    {
        std::ostringstream diff;
        if(ref.kernels.size() != got.kernels.size())
            diff << "kernel count " << ref.kernels.size() << " vs " << got.kernels.size() << "; ";
        if(ref.codeObjectVersion != got.codeObjectVersion)
            diff << "cov " << ref.codeObjectVersion << " vs " << got.codeObjectVersion << "; ";
        if(ref.isaName != got.isaName)
            diff << "isa " << ref.isaName << " vs " << got.isaName << "; ";
        std::map<std::string, const co::KernelMetadata*> byName;
        for(const auto& k : got.kernels)
            byName[k.name] = &k;
        for(const auto& r : ref.kernels)
        {
            auto it = byName.find(r.name);
            if(it == byName.end())
            {
                diff << "missing " << r.name << "; ";
                continue;
            }
            const auto& g     = *it->second;
            auto        field = [&](const char* what, uint64_t a, uint64_t b) {
                if(a != b)
                    diff << r.name.substr(0, 40) << " " << what << " " << a << " vs " << b << "; ";
            };
            if(r.symbol != g.symbol)
                diff << r.name << " symbol differs; ";
            field("kernarg", r.kernargSegmentSize, g.kernargSegmentSize);
            field("lds", r.groupSegmentFixedSize, g.groupSegmentFixedSize);
            field("scratch", r.privateSegmentFixedSize, g.privateSegmentFixedSize);
            field("vgpr", r.vgprCount, g.vgprCount);
            field("agpr", r.agprCount, g.agprCount);
            field("sgpr", r.sgprCount, g.sgprCount);
            field("vgpr_spill", r.vgprSpillCount, g.vgprSpillCount);
            field("sgpr_spill", r.sgprSpillCount, g.sgprSpillCount);
            field("wave", r.wavefrontSize, g.wavefrontSize);
            field("args", r.arguments.size(), g.arguments.size());
        }
        return diff.str();
    }

    std::vector<std::string> kernelNames(const co::CodeObjectMetadata& md)
    {
        std::vector<std::string> names;
        for(const auto& k : md.kernels)
            names.push_back(k.name);
        return names;
    }

    // ---- tests -----------------------------------------------------------------

    using Test = std::function<Outcome(const fs::path&)>;
    std::vector<std::pair<std::string, Test>> tests;

    void addTest(std::string name, Test test)
    {
        tests.emplace_back(std::move(name), std::move(test));
    }

    Outcome fromResult(const std::string& error)
    {
        return {error.empty(), error.empty() ? "ok" : error};
    }

    co::Result
        compileTensileHelper(const Bundle& b, const co::Target& build, const co::Options& options)
    {
        auto o = options;
        for(const auto& h : b.headers)
            o.includes.push_back({h.filename().string(), readText(h)});
        return co::compileHip(
            {{b.helperSource.filename().string(), readText(b.helperSource)}}, build, o);
    }

    // The target a clang-built reference was built for, so comgr can reproduce it.
    co::Target referenceTarget(const co::CodeObjectMetadata& reference)
    {
        const std::string triple = "amdgcn-amd-amdhsa--";
        const auto&       isa    = reference.isaName;
        return co::Target::fromTargetId(isa.rfind(triple, 0) == 0 ? isa.substr(triple.size()) : isa);
    }

    void registerTensileTests(const Bundle& b)
    {
        addTest("a_tensile_assemble[" + b.name + "]", [b](const fs::path& dir) -> Outcome {
            const auto source    = readText(b.assembly);
            const auto reference = b.referenceCo.empty() ? std::vector<char>{}
                                                         : readBytes(b.referenceCo);
            const auto ref       = co::readMetadata(reference);
            if(!reference.empty() && !ref.ok())
                return {false, "readMetadata(clang): " + ref.log};
            const auto build  = reference.empty() ? target() : referenceTarget(ref.metadata);
            const auto result = co::assemble(
                {{b.assembly.filename().string(), source}}, build, tensileOptions());
            writeText(dir / "assemble.log", result.log);
            if(!result.ok())
                return {false, std::string(co::toString(result.status)) + ": " + result.log};
            writeBytes(dir / "comgr.co", result.bytes.data(), result.bytes.size());
            const auto got = co::readMetadata(result.bytes);
            if(!got.ok())
                return {false, "readMetadata: " + got.log};
            writeText(dir / "comgr.metadata.txt", describe(got.metadata));
            const auto name = mainKernel(source);
            if(got.metadata.kernels.size() != 1 || got.metadata.kernels.front().name != name
               || got.metadata.isaName != build.isaName())
                return {false, "unexpected metadata: " + describe(got.metadata).substr(0, 300)};
            if(reference.empty())
                return {true, "1 kernel, isa " + got.metadata.isaName};
            fs::copy_file(b.referenceCo, dir / "clang.co", fs::copy_options::overwrite_existing);
            writeText(dir / "clang.metadata.txt", describe(ref.metadata));
            const auto diff = compareMetadata(ref.metadata, got.metadata);
            return {diff.empty(),
                    diff.empty() ? std::string("metadata identical to clang; file ")
                                       + (reference == result.bytes ? "byte-identical"
                                                                    : "differs (compare the .co files)")
                                 : "metadata differs: " + diff};
        });

        if(gfx12() || target().gfx.rfind("gfx11", 0) == 0)
            addTest("e_tensile_real_true16[" + b.name + "]", [b](const fs::path& dir) -> Outcome {
                // Informational: shows whether the +real-true16 default matters for this kernel.
                auto o                 = tensileOptions();
                o.architectureDefaults = false;
                const auto result      = co::assemble(
                    {{b.assembly.filename().string(), readText(b.assembly)}}, target(), o);
                const auto with = co::assemble(
                    {{b.assembly.filename().string(), readText(b.assembly)}}, target(), tensileOptions());
                writeText(dir / "assemble-no-defaults.log", result.log);
                if(!result.ok())
                    return {true,
                            "without +real-true16: " + std::string(co::toString(result.status))};
                return {true,
                        result.bytes == with.bytes ? "without +real-true16: identical output"
                                                   : "without +real-true16: output differs"};
            });

        if(cfg.gpu)
            addTest("b_gpu_load_tensile[" + b.name + "]", [b](const fs::path& dir) -> Outcome {
                const auto result = co::assemble(
                    {{b.assembly.filename().string(), readText(b.assembly)}}, target(), tensileOptions());
                if(!result.ok())
                    return {false, result.log};
                const auto md = co::readMetadata(result.bytes);
                if(!md.ok())
                    return {false, md.log};
                Module     m;
                const auto err = loadAndResolve(result.bytes, kernelNames(md.metadata), m);
                writeText(dir / "result.txt", err.empty() ? "loaded and resolved\n" : err + "\n");
                return {err.empty(),
                        err.empty() ? "hipModuleLoadData + hipModuleGetFunction("
                                          + md.metadata.kernels.front().name.substr(0, 48) + "...)"
                                    : err};
            });

        if(b.helperSource.empty())
            return;

        addTest("c_tensile_helper_compile[" + b.name + "]", [b](const fs::path& dir) -> Outcome {
            const auto reference = b.referenceHsaco.empty() ? std::vector<char>{}
                                                            : readBytes(b.referenceHsaco);
            const auto ref       = co::readMetadata(reference);
            if(!reference.empty() && !ref.ok())
                return {false, "readMetadata(clang): " + ref.log};
            const auto build   = reference.empty() ? target() : referenceTarget(ref.metadata);
            auto       staged  = tensileOptions();
            staged.hipPipeline = co::HipPipeline::Staged;
            const auto oneShot = compileTensileHelper(b, build, tensileOptions());
            const auto twoStep = compileTensileHelper(b, build, staged);
            writeText(dir / "compile-oneshot.log", oneShot.log);
            writeText(dir / "compile-staged.log", twoStep.log);
            if(!oneShot.ok() || !twoStep.ok())
                return {false, "one-shot: " + std::string(co::toString(oneShot.status))
                                   + ", staged: " + co::toString(twoStep.status) + ": "
                                   + (oneShot.log + twoStep.log).substr(0, 4000)};
            writeBytes(dir / "comgr-oneshot.hsaco", oneShot.bytes.data(), oneShot.bytes.size());
            writeBytes(dir / "comgr-staged.hsaco", twoStep.bytes.data(), twoStep.bytes.size());
            const auto a = co::readMetadata(oneShot.bytes);
            const auto s = co::readMetadata(twoStep.bytes);
            if(!a.ok() || !s.ok())
                return {false, "readMetadata: " + a.log + s.log};
            writeText(dir / "comgr-oneshot.metadata.txt", describe(a.metadata));
            auto diff   = compareMetadata(a.metadata, s.metadata);
            auto detail = std::to_string(a.metadata.kernels.size()) + " kernels; staged "
                          + (diff.empty() ? "matches one-shot" : "differs: " + diff);
            if(!reference.empty())
            {
                fs::copy_file(
                    b.referenceHsaco, dir / "clang.hsaco", fs::copy_options::overwrite_existing);
                writeText(dir / "clang.metadata.txt", describe(ref.metadata));
                const auto clangDiff = compareMetadata(ref.metadata, a.metadata);
                detail += clangDiff.empty() ? "; metadata identical to clang"
                                            : "; differs from clang: " + clangDiff.substr(0, 600);
                diff += clangDiff;
            }
            return {diff.empty(), detail};
        });

        // The gate for one code object per solution: the assembled main kernel and
        // the compiled helpers link into one executable that exports both.
        addTest("g_tensile_mixed_link[" + b.name + "]", [b](const fs::path& dir) -> Outcome {
            const auto source = readText(b.assembly);
            const auto main   = mainKernel(source);
            auto       o      = tensileOptions();
            for(const auto& h : b.headers)
                o.includes.push_back({h.filename().string(), readText(h)});
            auto a = co::assembleRelocatables({{b.assembly.filename().string(), source}}, target(), o);
            auto h = co::compileHipRelocatables(
                {{b.helperSource.filename().string(), readText(b.helperSource)}}, target(), o);
            if(!a.ok() || !h.ok())
                return {false, (a.log + h.log).substr(0, 4000)};
            auto objects = a.objects;
            objects.insert(objects.end(), h.objects.begin(), h.objects.end());
            const auto linked = co::link(objects, target(), o);
            writeText(dir / "link.log", a.log + h.log + linked.log);
            if(!linked.ok())
                return {false, "link: " + linked.log.substr(0, 4000)};
            writeBytes(dir / "entry.co", linked.bytes.data(), linked.bytes.size());
            const auto md = co::readMetadata(linked.bytes);
            if(!md.ok())
                return {false, md.log};
            writeText(dir / "metadata.txt", describe(md.metadata));
            auto names = kernelNames(md.metadata);
            if(std::count(names.begin(), names.end(), main) != 1 || names.size() < 2)
                return {false,
                        "expected the main kernel and at least one helper, got "
                            + std::to_string(names.size()) + " kernels"};
            const auto helper = names.front() == main ? names.back() : names.front();
            std::string detail = "main + " + std::to_string(names.size() - 1)
                                 + " helper kernels in one executable";
            if(!cfg.gpu)
                return {true, detail + " (metadata only)"};
            Module m;
            if(auto e = loadAndResolve(linked.bytes, names, m); !e.empty())
                return {false, e};
            return {true, detail + "; loaded; resolved " + main.substr(0, 32) + "... and "
                              + helper.substr(0, 32) + "... among all "
                              + std::to_string(names.size())};
        });
    }

    void registerTests()
    {
        for(const auto& dir : cfg.bundles)
            registerTensileTests(discover(dir));

        if(cfg.gpu)
        {
            addTest("b_gpu_run_handwritten_asm", [](const fs::path& dir) -> Outcome {
                const auto result
                    = co::assemble({{"scale_add", asmKernel("scale_add", scaleOp(), 5)}}, target(), {});
                if(!result.ok())
                    return {false, result.log};
                writeBytes(dir / "scale_add.co", result.bytes.data(), result.bytes.size());
                Module                               m;
                std::map<std::string, hipFunction_t> f;
                if(auto e = loadAndResolve(result.bytes, {"scale_add"}, m, &f); !e.empty())
                    return {false, e};
                return fromResult(runIntKernel(f["scale_add"], [](int x) { return 2 * x + 1; }));
            });

            addTest("c_hip_helper_run", [](const fs::path& dir) -> Outcome {
                const auto result = co::compileHip(
                    {{"reduce_partials.hip", hipHelperSource}}, target(), hipOptions());
                writeText(dir / "compile.log", result.log);
                if(!result.ok())
                    return {false, std::string(co::toString(result.status)) + ": " + result.log};
                writeBytes(dir / "reduce_partials.co", result.bytes.data(), result.bytes.size());
                const auto md = co::readMetadata(result.bytes);
                if(!md.ok())
                    return {false, md.log};
                writeText(dir / "metadata.txt", describe(md.metadata));
                Module                               m;
                std::map<std::string, hipFunction_t> f;
                if(auto e = loadAndResolve(result.bytes, {"reduce_partials"}, m, &f); !e.empty())
                    return {false, e};
                auto err = runReduce(f["reduce_partials"]);
                return {err.empty(),
                        err.empty() ? "split-K style reduction, 4096 halves, rel err <= 1e-3"
                                    : err};
            });
        }

        addTest("t_concurrent_builds", [](const fs::path& dir) -> Outcome {
            constexpr int threads = 8, iterations = 3;
            const auto    asmSource = asmKernel("scale_add", scaleOp(), 5);
            std::vector<std::vector<char>> asmOut(threads * iterations),
                hipOut(threads * iterations);
            std::vector<std::string> errors(threads);
            std::vector<std::thread> pool;
            for(int t = 0; t < threads; ++t)
                pool.emplace_back([&, t] {
                    for(int i = 0; i < iterations; ++i)
                    {
                        auto a = co::assemble({{"scale_add", asmSource}}, target(), {});
                        auto h = co::compileHip(
                            {{"triple.hip", hipTripleSource}}, target(), hipOptions());
                        if(!a.ok() || !h.ok())
                        {
                            errors[t] = a.log + h.log;
                            return;
                        }
                        asmOut[t * iterations + i] = std::move(a.bytes);
                        hipOut[t * iterations + i] = std::move(h.bytes);
                    }
                });
            for(auto& thread : pool)
                thread.join();
            for(const auto& e : errors)
                if(!e.empty())
                    return {false, e.substr(0, 1000)};
            const bool sameAsm = std::all_of(
                asmOut.begin(), asmOut.end(), [&](const auto& b) { return b == asmOut.front(); });
            const bool sameHip = std::all_of(
                hipOut.begin(), hipOut.end(), [&](const auto& b) { return b == hipOut.front(); });
            writeText(dir / "result.txt",
                      "threads=" + std::to_string(threads) + " iterations="
                          + std::to_string(iterations) + " asm identical="
                          + std::to_string(sameAsm) + " hip identical=" + std::to_string(sameHip)
                          + "\n");
            return {sameAsm && sameHip,
                    std::to_string(threads) + " threads x " + std::to_string(iterations)
                        + " assemble+compileHip: all succeeded; outputs "
                        + (sameAsm && sameHip ? "byte-identical" : "DIFFER")};
        });

        addTest("d_multi_asm_link", [](const fs::path& dir) -> Outcome {
            const auto result = co::assemble({{"scale_add", asmKernel("scale_add", scaleOp(), 5)},
                                              {"add7", asmKernel("add7", add7Op(), 5)}},
                                             target(),
                                             {});
            if(!result.ok())
                return {false, result.log};
            writeBytes(dir / "two_kernels.co", result.bytes.data(), result.bytes.size());
            const auto md = co::readMetadata(result.bytes);
            if(!md.ok())
                return {false, md.log};
            writeText(dir / "metadata.txt", describe(md.metadata));
            if(md.metadata.kernels.size() != 2)
                return {false,
                        "expected 2 kernels in metadata, got "
                            + std::to_string(md.metadata.kernels.size())};
            if(!cfg.gpu)
                return {true, "2 kernels in one executable (metadata only)"};
            Module                               m;
            std::map<std::string, hipFunction_t> f;
            if(auto e = loadAndResolve(result.bytes, {"scale_add", "add7"}, m, &f); !e.empty())
                return {false, e};
            auto e1 = runIntKernel(f["scale_add"], [](int x) { return 2 * x + 1; });
            auto e2 = runIntKernel(f["add7"], [](int x) { return x + 7; });
            return fromResult(e1 + e2);
        });

        addTest("d_mixed_asm_hip_link", [](const fs::path& dir) -> Outcome {
            auto a = co::assembleRelocatables(
                {{"scale_add", asmKernel("scale_add", scaleOp(), 5)}}, target(), {});
            auto h = co::compileHipRelocatables(
                {{"triple.hip", hipTripleSource}}, target(), baseOptions());
            if(!a.ok() || !h.ok())
                return {false, a.log + h.log};
            std::vector<co::Relocatable> objects = a.objects;
            objects.insert(objects.end(), h.objects.begin(), h.objects.end());
            const auto result = co::link(objects, target(), {});
            if(!result.ok())
                return {false, result.log};
            writeBytes(dir / "mixed.co", result.bytes.data(), result.bytes.size());
            const auto md = co::readMetadata(result.bytes);
            if(!md.ok())
                return {false, md.log};
            writeText(dir / "metadata.txt", describe(md.metadata));
            if(md.metadata.kernels.size() != 2)
                return {false,
                        "expected 2 kernels, got " + std::to_string(md.metadata.kernels.size())};
            if(!cfg.gpu)
                return {true, "asm + HIP relocatables linked (metadata only)"};
            Module                               m;
            std::map<std::string, hipFunction_t> f;
            if(auto e = loadAndResolve(result.bytes, {"scale_add", "triple"}, m, &f); !e.empty())
                return {false, e};
            auto e1 = runIntKernel(f["scale_add"], [](int x) { return 2 * x + 1; });
            auto e2 = runIntKernel(f["triple"], [](int x) { return 3 * x; });
            return fromResult(e1 + e2);
        });

        addTest("d_multi_hip_sources", [](const fs::path& dir) -> Outcome {
            std::string detail, error;
            for(auto pipeline : {co::HipPipeline::Staged, co::HipPipeline::SourceToRelocatable})
            {
                co::Options o     = hipOptions();
                o.hipPipeline     = pipeline;
                const auto result = co::compileHip(
                    {{"reduce_partials.hip", hipHelperSource}, {"triple.hip", hipTripleSource}},
                    target(),
                    o);
                const char* tag = pipeline == co::HipPipeline::Staged ? "staged" : "oneshot";
                writeText(dir / (std::string(tag) + ".log"), result.log);
                if(!result.ok())
                {
                    error += std::string(tag) + ": " + result.log.substr(0, 600) + "; ";
                    continue;
                }
                writeBytes(
                    dir / (std::string(tag) + ".co"), result.bytes.data(), result.bytes.size());
                const auto md = co::readMetadata(result.bytes);
                if(!md.ok() || md.metadata.kernels.size() != 2)
                {
                    error += std::string(tag) + ": expected 2 kernels; ";
                    continue;
                }
                detail += std::string(tag) + " 2 kernels";
                if(cfg.gpu)
                {
                    Module                               m;
                    std::map<std::string, hipFunction_t> f;
                    auto e = loadAndResolve(result.bytes, {"reduce_partials", "triple"}, m, &f);
                    if(e.empty())
                        e = runReduce(f["reduce_partials"])
                            + runIntKernel(f["triple"], [](int x) { return 3 * x; });
                    if(!e.empty())
                        error += std::string(tag) + " GPU: " + e + "; ";
                    else
                        detail += " ran";
                }
                detail += "; ";
            }
            return {error.empty(), error.empty() ? detail : error};
        });

        addTest("d_shared_cuid_collides", [](const fs::path& dir) -> Outcome {
            co::Options o   = hipOptions();
            o.compilerFlags = {"-cuid=shared"};
            const auto result = co::compileHip(
                {{"reduce_partials.hip", hipHelperSource}, {"triple.hip", hipTripleSource}},
                target(),
                o);
            writeText(dir / "link.log", result.log);
            const bool ok = result.status == co::Status::ComgrError
                            && result.log.find("duplicate symbol: __hip_cuid_") != std::string::npos;
            return {ok,
                    std::string("one -cuid for two sources: ") + co::toString(result.status)
                        + (ok ? " (duplicate __hip_cuid_ symbol)" : ": " + result.log.substr(0, 300))};
        });

        addTest("options_code_object_versions", [](const fs::path& dir) -> Outcome {
            std::string detail, error;
            for(int v : {4, 5, 6})
            {
                co::Options o       = baseOptions();
                o.codeObjectVersion = v;
                const auto a = co::assemble({{"k", asmKernel("k", scaleOp(), v)}}, target(), o);
                const auto h = co::compileHip({{"triple.hip", hipTripleSource}}, target(), o);
                if(!a.ok() || !h.ok())
                {
                    error += "v" + std::to_string(v) + ": " + a.log + h.log;
                    continue;
                }
                writeBytes(
                    dir / ("asm_v" + std::to_string(v) + ".co"), a.bytes.data(), a.bytes.size());
                writeBytes(
                    dir / ("hip_v" + std::to_string(v) + ".co"), h.bytes.data(), h.bytes.size());
                const int av = co::readMetadata(a.bytes).metadata.codeObjectVersion;
                const int hv = co::readMetadata(h.bytes).metadata.codeObjectVersion;
                detail += "v" + std::to_string(v) + "->asm " + std::to_string(av) + "/hip "
                          + std::to_string(hv) + " ";
                if(av != v || hv != v)
                    error += "requested v" + std::to_string(v) + " got asm " + std::to_string(av)
                             + " hip " + std::to_string(hv) + "; ";
            }
            return {error.empty(), error.empty() ? detail : error};
        });

        addTest("options_linker_flags_forwarded", [](const fs::path& dir) -> Outcome {
            const auto  source = asmKernel("scale_add", scaleOp(), 5);
            co::Options bare;
            bare.linkerFlags    = {"--build-id=sha1"};
            const auto rejected = co::assemble({{"scale_add", source}}, target(), bare);
            co::Options o;
            o.linkerFlags     = {"-Xlinker", "--build-id=sha1"};
            const auto result = co::assemble({{"scale_add", source}}, target(), o);
            writeText(dir / "link.log", rejected.log + "\n----\n" + result.log);
            if(!result.ok())
                return {false, "link with -Xlinker --build-id=sha1 failed: " + result.log};
            writeBytes(dir / "build_id.co", result.bytes.data(), result.bytes.size());
            const std::string bytes(result.bytes.begin(), result.bytes.end());
            const bool        hasNote = bytes.find(".note.gnu.build-id") != std::string::npos;
            const bool        bareRejected
                = rejected.status == co::Status::ComgrError
                  && rejected.log.find("unknown argument") != std::string::npos;
            return {hasNote && bareRejected,
                    std::string("bare --build-id: ") + co::toString(rejected.status)
                        + " (unknown argument); -Xlinker --build-id=sha1: build-id note "
                        + (hasNote ? "present" : "absent")};
        });

        addTest("options_retarget_assembly", [](const fs::path&) -> Outcome {
            if(target().gfx.rfind("gfx9", 0) != 0)
                return {true, "skipped (gfx9 only)"};
            const auto text  = asmKernel("k", scaleOp(), 5);
            co::Target other = co::Target::fromTargetId("gfx942:xnack-");
            const auto plain = co::assemble({{"k", text}}, other, {});
            co::Options o;
            o.retargetAssembly = true;
            const auto moved   = co::assemble({{"k", text}}, other, o);
            const auto md      = co::readMetadata(moved.bytes);
            const bool ok      = !plain.ok() && moved.ok() && md.ok()
                            && md.metadata.isaName == other.isaName();
            return {ok,
                    std::string("without retarget: ") + co::toString(plain.status)
                        + "; with retarget: " + co::toString(moved.status)
                        + (md.ok() ? " isa=" + md.metadata.isaName : "")};
        });

        addTest("options_missing_rocm_path", [](const fs::path& dir) -> Outcome {
            if(std::getenv("HIP_PATH"))
                return {true, "skipped (HIP_PATH is set)"};
            const auto result = co::compileHip({{"triple.hip", hipTripleSource}}, target(), {});
            writeText(dir / "log.txt", result.log);
            const bool ok = result.status == co::Status::ComgrError
                            && result.log.find("hip/hip_runtime.h") != std::string::npos;
            return {ok,
                    std::string("no --rocm-path: ") + co::toString(result.status)
                        + (ok ? " ('hip/hip_runtime.h' not found)" : ": " + result.log.substr(0, 300))};
        });

        addTest("f_bad_assembly", [](const fs::path& dir) -> Outcome {
            auto       text   = asmKernel("k", "  v_bogus_instruction v3, v3\n", 5);
            const auto result = co::assemble({{"bad.s", text}}, target(), {});
            writeText(dir / "log.txt", result.log);
            const bool ok = result.status == co::Status::ComgrError && !result.bytes.size()
                            && result.log.find("v_bogus_instruction") != std::string::npos;
            const auto at = result.log.find("error");
            return {ok,
                    std::string(co::toString(result.status)) + ", log "
                        + std::to_string(result.log.size()) + " bytes: "
                        + (at == std::string::npos ? result.log : result.log.substr(at, 160))};
        });

        addTest("f_unknown_target", [](const fs::path& dir) -> Outcome {
            const auto r1 = co::assemble(
                {{"k", asmKernel("k", scaleOp(), 5)}}, co::Target{"gfx9999", {}, 0}, {});
            const auto r2
                = co::compileHip({{"t.hip", hipTripleSource}}, co::Target{"sm_90", {}, 0}, {});
            const auto r3
                = co::assemble({{"k", "s_endpgm\n"}}, co::Target{"gfx950", {"bogus"}, 0}, {});
            writeText(dir / "log.txt", r1.log + r2.log + r3.log);
            const bool ok = r1.status == co::Status::InvalidArgument && !r1.log.empty()
                            && r2.status == co::Status::InvalidArgument && !r2.log.empty()
                            && r3.status == co::Status::InvalidArgument;
            return {ok,
                    "gfx9999: " + std::string(co::toString(r1.status)) + " '"
                        + r1.log.substr(0, 90) + "'; sm_90: " + co::toString(r2.status)
                        + "; bad feature: " + co::toString(r3.status)};
        });

        addTest("f_bad_hip", [](const fs::path& dir) -> Outcome {
            const auto result = co::compileHip(
                {{"broken.hip", "extern \"C\" __global__ void k(int* p) { p[0] = undeclared; }\n"}},
                target(),
                baseOptions());
            writeText(dir / "log.txt", result.log);
            const bool ok = result.status == co::Status::ComgrError
                            && result.log.find("undeclared") != std::string::npos;
            return {ok,
                    std::string(co::toString(result.status)) + ", log mentions identifier: "
                        + (ok ? "yes" : "no: " + result.log.substr(0, 300))};
        });

        addTest("f_bad_inputs", [](const fs::path& dir) -> Outcome {
            std::vector<std::string> failures;
            auto expect = [&](const char* what, co::Status got, co::Status want) {
                if(got != want)
                    failures.push_back(std::string(what) + "=" + co::toString(got));
            };
            const std::vector<char> garbage(256, 'x');
            expect("garbage metadata", co::readMetadata(garbage).status, co::Status::MetadataError);
            std::string bundle = "__CLANG_OFFLOAD_BUNDLE__";
            bundle.resize(128, '\0');
            const auto bundled = co::readMetadata(bundle.data(), bundle.size());
            expect("bundle metadata", bundled.status, co::Status::MetadataError);
            expect(
                "null metadata", co::readMetadata(nullptr, 0).status, co::Status::InvalidArgument);
            expect("no sources", co::assemble({}, target(), {}).status, co::Status::InvalidArgument);
            expect("no objects", co::link({}, target(), {}).status, co::Status::InvalidArgument);
            co::Options bad;
            bad.codeObjectVersion = 3;
            expect("cov 3",
                   co::assemble({{"k", "s_endpgm"}}, target(), bad).status,
                   co::Status::InvalidArgument);
            expect("duplicate names",
                   co::assemble({{"k", "s_endpgm"}, {"k", "s_endpgm"}}, target(), {}).status,
                   co::Status::InvalidArgument);
            expect("empty relocatable",
                   co::link({{"x", {}}}, target(), {}).status,
                   co::Status::InvalidArgument);
            expect("garbage relocatable",
                   co::link({{"x", std::vector<char>(64, 'y')}}, target(), {}).status,
                   co::Status::ComgrError);
            co::Options badInclude;
            badInclude.includes = {{"../escape.h", ""}};
            badInclude.rocmPath = cfg.rocm;
            expect("include escape",
                   co::compileHip({{"t.hip", hipTripleSource}}, target(), badInclude).status,
                   co::Status::InvalidArgument);
            writeText(dir / "bundle.log", bundled.log);
            std::string detail;
            for(auto& f : failures)
                detail += f + "; ";
            return {failures.empty(),
                    failures.empty() ? "10 malformed inputs rejected with the expected status"
                                     : detail};
        });
    }

    void usage()
    {
        std::fprintf(stderr,
                     "usage: hipblaslt-jit-code-object-test --out DIR [--target ID] [--gpu | --ffm] "
                     "[--bundle BUNDLE_DIR]... [--tensile-cov N] [--rocm PREFIX] "
                     "[--only NAME]...\n");
    }
}

int main(int argc, char** argv)
{
    for(int i = 1; i < argc; ++i)
    {
        std::string a    = argv[i];
        auto        next = [&]() -> std::string {
            if(i + 1 >= argc)
            {
                usage();
                std::exit(2);
            }
            return argv[++i];
        };
        if(a == "--target")
            cfg.target = next();
        else if(a == "--out")
            cfg.out = next();
        else if(a == "--gpu")
            cfg.gpu = true;
        else if(a == "--ffm")
            cfg.gpu = cfg.ffm = true;
        else if(a == "--bundle")
            cfg.bundles.push_back(next());
        else if(a == "--tensile-cov")
            cfg.tensileCov = std::stoi(next());
        else if(a == "--rocm")
            cfg.rocm = next();
        else if(a == "--only")
            cfg.only.push_back(next());
        else
        {
            usage();
            return 2;
        }
    }
    if(cfg.out.empty() || (!cfg.gpu && cfg.target.empty()))
    {
        usage();
        return 2;
    }
    if(cfg.rocm.empty())
        cfg.rocm = co::rocmPath();
    if(cfg.gpu)
    {
        // The simulated topology exposes one GPU as device 0; model mode is
        // selected by HSA_MODEL_TOPOLOGY and never falls back to /dev/kfd.
        if(cfg.ffm && (!std::getenv("HSA_MODEL_TOPOLOGY") || !std::getenv("HSA_MODEL_LIB")))
        {
            std::fprintf(stderr, "--ffm requires HSA_MODEL_TOPOLOGY and HSA_MODEL_LIB\n");
            return 2;
        }
        hipDeviceProp_t props;
        if(hipGetDeviceProperties(&props, 0) != hipSuccess)
        {
            std::fprintf(stderr, "no HIP device\n");
            return 2;
        }
        if(cfg.target.empty())
            cfg.target = props.gcnArchName;
        if(co::Target::fromTargetId(props.gcnArchName).gfx != target().gfx)
        {
            std::fprintf(stderr,
                         "device 0 is %s, expected %s; refusing to run\n",
                         props.gcnArchName,
                         cfg.target.c_str());
            return 2;
        }
        std::printf("device: %s (%s), comgr %s\n",
                    props.name,
                    props.gcnArchName,
                    co::comgrIdentity().c_str());
    }
    else
        std::printf("target %s, comgr %s, no GPU tests\n",
                    cfg.target.c_str(),
                    co::comgrIdentity().c_str());

    try
    {
        registerTests();
    }
    catch(const std::exception& e)
    {
        std::fprintf(stderr, "%s\n", e.what());
        return 2;
    }
    fs::create_directories(cfg.out);
    std::ofstream summary(cfg.out / "summary.tsv");
    summary << "test\tresult\tdetail\n";
    int failures = 0;
    for(auto& entry : tests)
    {
        const auto& name = entry.first;
        auto&       test = entry.second;
        if(!cfg.only.empty()
           && std::none_of(cfg.only.begin(), cfg.only.end(), [&](const std::string& o) {
                  return name.find(o) != std::string::npos;
              }))
            continue;
        std::string dirName = name;
        std::replace(dirName.begin(), dirName.end(), '[', '_');
        dirName.erase(std::remove(dirName.begin(), dirName.end(), ']'), dirName.end());
        const fs::path dir = cfg.out / dirName;
        fs::create_directories(dir);
        Outcome outcome;
        try
        {
            outcome = test(dir);
        }
        catch(const std::exception& e)
        {
            outcome = {false, std::string("exception: ") + e.what()};
        }
        failures += !outcome.pass;
        std::string oneLine = outcome.detail;
        std::replace(oneLine.begin(), oneLine.end(), '\n', ' ');
        std::replace(oneLine.begin(), oneLine.end(), '\t', ' ');
        std::printf("%-4s %-44s %s\n",
                    outcome.pass ? "PASS" : "FAIL",
                    name.c_str(),
                    oneLine.substr(0, 400).c_str());
        summary << name << "\t" << (outcome.pass ? "PASS" : "FAIL") << "\t" << oneLine << "\n";
    }
    std::printf("%d test(s) failed\n", failures);
    return failures ? 1 : 0;
}
