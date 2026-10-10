/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2026 AIwork4me
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

// Regression test: runtime-compiled MIOpen kernels must stay self-contained
// when the HIPRTC toolchain cannot reach a host C++ standard library.
//
// Background (MIOpen#3956): on hosts whose clang has no configured C++ stdlib
// (AMD Windows pip wheels, or any HIPRTC without host STL paths), kernels
// pulling <type_traits>/<utility>/<limits>/<initializer_list> through
// miopen_type_traits.hpp / miopen_utility.hpp / radix.hpp / tensor_view.hpp
// fail at hiprtcCompileProgram with "'type_traits' file not found".
// The fix selects freestanding definitions via __has_include when no standard
// library is reachable. This test compiles the originally failing kernel
// (MIOpenBatchNormFwdTrainSpatial.cpp) through hiprtc with the STL isolated,
// so an unpatched source tree fails with the field signature and a patched
// tree produces a non-empty code object. Compile-only: no GPU is required.
//
// Usage:
//   hiprtc_selfcontained <kernels-src-dir> [--mode=M] [--arch=gfxNNNN]
//                        [--hip-flat=N] [--isolate=OPT]
//
//   --mode=positive   (default) BN kernel with the host STL isolated must
//                     compile and produce a code object. PATCHED-tree control.
//   --mode=negative   BN kernel with the host STL isolated must fail with the
//                     missing-STL signature 'type_traits' file not found.
//                     UNPATCHED-tree control; any other failure is reported
//                     as INCONCLUSIVE, not as a pass.
//   --mode=with-stl   BN kernel with the ambient include environment (no
//                     isolation) must compile: STL-present control for trees
//                     where a host stdlib is available.
//   --mode=ordinary   trivial one-line kernel through the same hiprtc calls:
//                     toolchain sanity control independent of MIOpen sources.
//
//   --arch=gfxNNNN    target architecture passed to hiprtc (required for
//                     codegen; compile-only still needs a valid target).
//                     Default: gfx1151 (the architecture the fix was
//                     validated on); override per CI leg as needed.
//   --hip-flat=N      value for HIP_PACKAGE_VERSION_FLAT. Default is derived
//                     from hiprtcVersion() as major*1000000000+minor*1000000,
//                     which preserves version ordering; CI wiring that knows
//                     the exact package value should pass it explicitly.
//                     Values >= 7000000000 exercise the affected HIP>=7 arm.
//   --isolate=OPT     include-isolation option for no-STL modes (default
//                     "-nostdinc"). Full -nostdinc is required on msvc-triple
//                     clang, whose automatic MSVC STL detection bypasses
//                     -nostdinc++; GNU-triple toolchains may prefer
//                     --isolate=-nostdinc++ to keep libc headers. Whatever
//                     option is used, the STL-unreachable probe verifies the
//                     isolation is real before verdicts count.
//
// Exit codes:
//   0  PASS (the mode's expectation held)
//   1  FAIL (the mode's expectation was violated)
//   2  SETUP ERROR (bad usage, unreadable source, hiprtc API misuse)
//   4  INCONCLUSIVE (expectation not testable in this environment, e.g. the
//                    host STL is still reachable in isolated mode, or a
//                    negative control failed for the wrong reason)
//
// Include-path design: kernel headers resolve through -I<kernels-src-dir>
// exactly as MIOpen's offline canaries do; hiprtc injects its own HIP headers
// internally. hiprtcCreateProgram named-header arguments are deliberately NOT
// used as a substitute for this search path (they are a virtual header map,
// not a filesystem include directory). Angle-bracket host includes are not
// resolvable through -I<kernels-src-dir> (the kernels directory contains no
// files named like standard library headers), so <type_traits> resolution
// depends only on the host toolchain environment, which is what this test
// controls.

// The hip headers require a platform selection even for host-side
// inclusion; MIOpen builds define this project-wide.
#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__ 1
#endif
#include <hip/hiprtc.h>

#include <cctype>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace {

constexpr int kExitPass         = 0;
constexpr int kExitFail         = 1;
constexpr int kExitSetup        = 2;
constexpr int kExitInconclusive = 4;

const char* kKernelName = "MIOpenBatchNormFwdTrainSpatial.cpp";

// Exact clang diagnostic lines for unresolvable host-library includes.
// Anchored to the full "fatal error: '<name>' file not found" form so that
// neither a missing KERNEL header ('miopen_type_traits.hpp' file not
// found) nor an injected #error directive whose text merely CONTAINS the
// phrase can satisfy the check (adversarial reviews F1, V1).
const char* kMissingStdlibSignatures[] = {
    "fatal error: 'type_traits' file not found",
    "fatal error: 'utility' file not found",
    "fatal error: 'limits' file not found",
    "fatal error: 'initializer_list' file not found",
};

// Number of clang diagnostic error lines: occurrences of "error:" as a
// diagnostic marker (covers "error:" and "fatal error:"). The echoed
// source line and the "N errors generated" footer do not contain this
// token, so this counts real diagnostics only.
std::size_t count_error_diagnostics(const std::string& log)
{
    std::size_t n = 0;
    for(std::size_t pos = 0; (pos = log.find("error:", pos)) != std::string::npos; pos += 6)
        ++n;
    return n;
}

// Error count from clang's footer ("N error generated" / "N errors
// generated"); returns 0 when no footer is present.
std::size_t footer_error_count(const std::string& log)
{
    for(const char* form : {" error generated", " errors generated"})
    {
        for(std::size_t pos = 0; (pos = log.find(form, pos)) != std::string::npos; pos += 6)
        {
            std::size_t dig = pos;
            while(dig > 0 && isdigit(static_cast<unsigned char>(log[dig - 1])))
                --dig;
            if(dig < pos)
            {
                try
                {
                    return static_cast<std::size_t>(std::stoull(log.substr(dig, pos - dig)));
                }
                catch(...)
                {
                }
            }
        }
    }
    return 0;
}

struct Args
{
    std::string kernels_dir;
    std::string mode            = "positive";
    std::string arch            = "gfx1151";
    std::string isolate         = "-nostdinc";
    unsigned long long hip_flat = 0; // 0 => derive from hiprtcVersion()
};

// RAII wrapper so every early exit still releases the hiprtc program.
struct Program
{
    hiprtcProgram prog = nullptr;
    ~Program()
    {
        if(prog)
        {
            hiprtcResult r = hiprtcDestroyProgram(&prog);
            if(r != HIPRTC_SUCCESS)
                std::fprintf(
                    stderr, "warning: hiprtcDestroyProgram: %s\n", hiprtcGetErrorString(r));
        }
    }
};

bool read_file(const std::string& path, std::string& out)
{
    std::ifstream f(path, std::ios::binary);
    if(!f)
        return false;
    std::stringstream ss;
    ss << f.rdbuf();
    out = ss.str();
    return true;
}

// Fetch the complete compiler log (both success and failure paths).
// Single named-object exit: mixed empty/value returns would defeat NRVO
// under MIOpen's -Wnrvo.
std::string get_log(hiprtcProgram prog)
{
    std::string log;
    std::size_t n = 0;
    if(hiprtcGetProgramLogSize(prog, &n) == HIPRTC_SUCCESS && n > 0)
    {
        log.resize(n, '\0');
        if(hiprtcGetProgramLog(prog, &log[0]) != HIPRTC_SUCCESS)
            log.clear();
        else
        {
            // The log is NUL-terminated within the reported size.
            const auto pos = log.find('\0');
            if(pos != std::string::npos)
                log.resize(pos);
        }
    }
    return log;
}

// Compile `src` as `name` with `options`; reports the compile result and,
// on success, retrieves the code object to prove one was generated.
// Returns HIPRTC_SUCCESS only if compilation succeeded AND a non-empty code
// object was produced (retrieve failures are returned as-is).
struct CompileOutcome
{
    hiprtcResult create_status  = HIPRTC_SUCCESS;
    hiprtcResult compile_status = HIPRTC_SUCCESS;
    hiprtcResult code_status    = HIPRTC_SUCCESS;
    std::size_t code_size       = 0;
    std::string log;
};

CompileOutcome compile(const std::string& name,
                       const std::string& src,
                       const std::vector<std::string>& options,
                       bool require_code)
{
    CompileOutcome out;
    Program p;
    out.create_status =
        hiprtcCreateProgram(&p.prog, src.c_str(), name.c_str(), 0, nullptr, nullptr);
    if(out.create_status != HIPRTC_SUCCESS)
        return out;

    std::vector<const char*> copts;
    copts.reserve(options.size());
    for(const auto& o : options)
        copts.push_back(o.c_str());

    out.compile_status = hiprtcCompileProgram(p.prog, static_cast<int>(copts.size()), copts.data());
    out.log            = get_log(p.prog);

    if(out.compile_status == HIPRTC_SUCCESS && require_code)
    {
        out.code_status = hiprtcGetCodeSize(p.prog, &out.code_size);
        if(out.code_status == HIPRTC_SUCCESS && out.code_size > 0)
        {
            // Materialize the code object: proves generation, not just a size query.
            std::vector<char> code(out.code_size);
            out.code_status = hiprtcGetCode(p.prog, code.data());
            if(out.code_status != HIPRTC_SUCCESS)
                out.code_size = 0;
        }
    }
    return out;
}

unsigned long long default_hip_flat()
{
    int major = 0, minor = 0;
    if(hiprtcVersion(&major, &minor) != HIPRTC_SUCCESS)
        return 0;
    return static_cast<unsigned long long>(major) * 1000000000ULL +
           static_cast<unsigned long long>(minor) * 1000000ULL;
}

// Common MIOpen BuildHip options for the BN spatial kernel (src/comgr.cpp)
// plus the production define set used when this kernel is runtime-compiled.
std::vector<std::string> kernel_options(const Args& a, bool isolate_stl)
{
    std::vector<std::string> opts = {
        "-D__HIP_PLATFORM_AMD__=1",
        "-DMIOPEN_USE_FP16=0",
        "-DMIOPEN_USE_FP32=1",
        "-DMIOPEN_USE_FPMIX=0",
        "-DMIOPEN_USE_BFPMIX=0",
        "-DMIOPEN_LAYER_NCHW=1",
        "-DMIOPEN_LAYER_NHWC=0",
        "-DMIO_BN_VARIANT=0",
        "-DMIO_BN_GRP0=1024",
        "-DMIO_BN_GRP1=1",
        "-DMIO_BN_GRP2=1",
        "-DHIP_PACKAGE_VERSION_FLAT=" + std::to_string(a.hip_flat),
        "-DMIOPEN_HIP_RUNTIME_COMPILE",
        "-Wno-cuda-compat",
        "-fno-gpu-rdc",
        "-O3",
        "-std=c++17",
        "--gpu-architecture=" + a.arch,
        "-I" + a.kernels_dir,
    };
    if(isolate_stl)
        opts.push_back(a.isolate); // host stdlib unreachable; probe verifies
    return opts;
}

// Prove the isolation is real before trusting isolated-mode verdicts: if the
// stdlib is still reachable, 'positive' would pass for the wrong reason (the
// __has_include arm took the real headers) and 'negative' could never fire.
// The probe covers all four standard-library headers the patch gates, not
// just <type_traits>, so a partial leak of any one of them is detected.
bool stdlib_is_reachable(const Args& a, CompileOutcome& probe)
{
    const std::string src = "#if __has_include(<type_traits>) || __has_include(<utility>) || "
                            "__has_include(<limits>) || __has_include(<initializer_list>)\n"
                            "#error STL_PROBE_REACHABLE\n"
                            "#endif\n"
                            "extern \"C\" __global__ void probe() {}\n";
    probe                 = compile("stl_probe.cu",
                    src,
                    kernel_options(a, /*isolate_stl=*/true),
                    /*require_code=*/false);
    return probe.compile_status != HIPRTC_SUCCESS &&
           probe.log.find("STL_PROBE_REACHABLE") != std::string::npos;
}

void print_outcome(const char* what, const CompileOutcome& o)
{
    std::fprintf(stderr,
                 "[%s] create=%d compile=%d code=%d code_size=%zu\n",
                 what,
                 static_cast<int>(o.create_status),
                 static_cast<int>(o.compile_status),
                 static_cast<int>(o.code_status),
                 o.code_size);
    if(!o.log.empty())
        std::fprintf(stderr, "---- compiler log ----\n%s\n----------------------\n", o.log.c_str());
}

int usage(const char* argv0)
{
    std::fprintf(stderr,
                 "usage: %s <kernels-src-dir> [--mode=positive|negative|with-stl|ordinary] "
                 "[--arch=gfxNNNN] [--hip-flat=N] [--isolate=OPT]\n",
                 argv0);
    return kExitSetup;
}

} // namespace

int main(int argc, char** argv)
{
    Args args;
    bool seen_mode = false, seen_arch = false, seen_hip_flat = false, seen_isolate = false;
    for(int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if(a.rfind("--mode=", 0) == 0)
        {
            if(seen_mode)
                return usage(argv[0]); // duplicate would silently downgrade a run
            seen_mode = true;
            args.mode = a.substr(7);
        }
        else if(a.rfind("--arch=", 0) == 0)
        {
            if(seen_arch)
                return usage(argv[0]);
            seen_arch = true;
            args.arch = a.substr(7);
        }
        else if(a.rfind("--hip-flat=", 0) == 0)
        {
            if(seen_hip_flat)
                return usage(argv[0]);
            seen_hip_flat = true;
            try
            {
                args.hip_flat = std::stoull(a.substr(11));
            }
            catch(...)
            {
                std::fprintf(stderr, "SETUP ERROR: bad --hip-flat value\n");
                return kExitSetup;
            }
        }
        else if(a.rfind("--isolate=", 0) == 0)
        {
            if(seen_isolate)
                return usage(argv[0]);
            seen_isolate = true;
            args.isolate = a.substr(10);
        }
        else if(!a.empty() && a[0] == '-')
            return usage(argv[0]);
        else if(args.kernels_dir.empty())
            args.kernels_dir = a;
        else
            return usage(argv[0]);
    }
    if(args.kernels_dir.empty())
        return usage(argv[0]);
    if(args.mode != "positive" && args.mode != "negative" && args.mode != "with-stl" &&
       args.mode != "ordinary")
        return usage(argv[0]);
    if(args.hip_flat == 0 && (args.hip_flat = default_hip_flat()) == 0)
    {
        std::fprintf(stderr, "SETUP ERROR: hiprtcVersion failed; pass --hip-flat=N\n");
        return kExitSetup;
    }
    if(args.hip_flat < 7000000000ULL)
        std::fprintf(stderr,
                     "note: HIP_PACKAGE_VERSION_FLAT=%llu (< 7): the affected HIP>=7 arm "
                     "is not exercised by this run\n",
                     args.hip_flat);

    if(args.mode == "ordinary")
    {
        const std::string src = "extern \"C\" __global__ void ordinary() {}\n";
        const CompileOutcome o =
            compile("ordinary.cu",
                    src,
                    {"-D__HIP_PLATFORM_AMD__=1", "--gpu-architecture=" + args.arch},
                    /*require_code=*/true);
        print_outcome("ordinary", o);
        const bool pass = o.compile_status == HIPRTC_SUCCESS && o.code_size > 0;
        std::printf("%s: ordinary hiprtc control %s\n",
                    pass ? "PASS" : "FAIL",
                    pass ? "(code object generated)" : "(see log)");
        return pass ? kExitPass : kExitFail;
    }

    std::string kernel_src;
    if(!read_file(args.kernels_dir + "/" + kKernelName, kernel_src))
    {
        std::fprintf(
            stderr, "SETUP ERROR: cannot read %s/%s\n", args.kernels_dir.c_str(), kKernelName);
        return kExitSetup;
    }
    // Identity markers: refuse a substituted/truncated kernels directory.
    // A trivial replacement kernel would otherwise satisfy the compile
    // without exercising the kernel's real include closure.
    if(kernel_src.find("MIOpenBatchNormFwdTrainSpatial") == std::string::npos ||
       kernel_src.find("__global__") == std::string::npos)
    {
        std::fprintf(stderr,
                     "SETUP ERROR: %s lacks the real BN spatial kernel identity markers; "
                     "refusing to test a substituted source\n",
                     kKernelName);
        return kExitSetup;
    }

    const bool isolate = (args.mode == "positive" || args.mode == "negative");

    if(isolate)
    {
        // Mandatory: prove the isolation is real before verdicts count. If
        // the stdlib is still reachable, 'positive' would pass through the
        // real-header arm and 'negative' could never fire (or worse, fire
        // for a mixed-environment reason) - no isolated verdict is issued.
        CompileOutcome probe;
        if(stdlib_is_reachable(args, probe))
        {
            print_outcome("stl-probe", probe);
            std::fprintf(stderr,
                         "INCONCLUSIVE: host C++ stdlib still reachable under "
                         "%s; isolation does not reproduce the no-STL "
                         "condition\n",
                         args.isolate.c_str());
            return kExitInconclusive;
        }
    }

    const CompileOutcome o =
        compile(kKernelName, kernel_src, kernel_options(args, isolate), /*require_code=*/true);
    print_outcome(args.mode.c_str(), o);

    if(args.mode == "negative")
    {
        if(o.compile_status == HIPRTC_SUCCESS)
        {
            std::fprintf(
                stderr,
                "FAIL: no-STL compile unexpectedly succeeded (self-containment regression?)\n");
            return kExitFail;
        }
        bool signature = false;
        for(const char* s : kMissingStdlibSignatures)
            if(o.log.find(s) != std::string::npos)
            {
                signature = true;
                break;
            }
        const std::size_t n_diag   = count_error_diagnostics(o.log);
        const std::size_t n_footer = footer_error_count(o.log);
        const bool single_error    = n_diag == 1 && (n_footer == 0 || n_footer == 1);
        if(!signature || !single_error)
        {
            std::fprintf(stderr,
                         "INCONCLUSIVE: compile failed without the exact missing-STL "
                         "signature as the single error (signature=%d, diagnostics=%zu, "
                         "footer=%zu); refusing to count arbitrary failures\n",
                         signature ? 1 : 0,
                         n_diag,
                         n_footer);
            return kExitInconclusive;
        }
        std::printf("PASS: unpatched kernel fails with the field signature "
                    "('type_traits' file not found) as the sole error, under "
                    "no-STL isolation\n");
        return kExitPass;
    }

    // positive / with-stl
    if(o.create_status != HIPRTC_SUCCESS)
    {
        std::fprintf(stderr, "SETUP ERROR: hiprtcCreateProgram failed\n");
        return kExitSetup;
    }
    const bool pass = o.compile_status == HIPRTC_SUCCESS && o.code_size > 0;
    if(!pass)
        std::fprintf(stderr, "FAIL: %s compile did not produce a code object\n", args.mode.c_str());
    else
        std::printf("PASS: kernel compiled with%s host STL (%zu-byte code object)\n",
                    args.mode == "positive" ? "out" : "",
                    o.code_size);
    return pass ? kExitPass : kExitFail;
}
