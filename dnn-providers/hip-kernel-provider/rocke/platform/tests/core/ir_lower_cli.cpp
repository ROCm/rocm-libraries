// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/ir_lower_cli.cpp -- family-agnostic IR-artifact lowering tool.
 *
 * Reads serialized `ck.dsl.ir/v1` text (from a file argument, or stdin when no
 * file is given), parses it back into a KernelDef with rocke_ir_parse, lowers it
 * with rocke_lower_kernel_to_llvm, and writes the resulting AMDGPU LLVM IR text
 * to stdout. This is the keystone of the single-lowerer path: an author can
 * serialize IR in any front end, and this tool reproduces the exact .ll the
 * engine would emit -- with no per-family C builder involved.
 *
 * Usage:
 *   ir_lower_cli [arch] [ir_file]
 *     arch     ISA target gfx string; defaults to "gfx950".
 *     ir_file  path to serialized IR; "-" or omitted reads stdin.
 *   ir_lower_cli --hip-batch
 *     Lowers many kernels to HIP source in one process, each with the defaults
 *     rocke_engine.lower_serialized_ir_hip uses. stdin holds one frame per
 *     kernel, "<arch> <byte count>\n" then that many bytes of serialized IR;
 *     stdout gets one frame per kernel in order, "<status> <byte count>\n" then
 *     the bytes: the HIP source when the status is 0, else the engine's
 *     refusal message, with the rocke_status_t code the engine returned, as
 *     rocke_engine.lower_serialized_ir_hip reports them. A refusal is data,
 *     not a failure: the process exits 0 once every frame is answered, and 2
 *     on a malformed or empty frame (the binding refuses empty IR outright)
 *     or when stdout cannot be written.
 *
 * All streams are binary, so the text is passed through unchanged on every
 * host. Without --hip-batch, any parse or lowering failure writes a clean
 * message to stderr and the process exits nonzero. The engine's extern "C"
 * boundary already converts every internal failure (including raised
 * exceptions) into a status code and an error string, so this tool never
 * aborts/terminates -- it only reports and exits.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif

#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_hip.h"
#include "rocke/lower_llvm.h"
#include "rocke/strbuf.h"

namespace
{

// Read an entire stream into a std::string. Used for stdin and file input.
std::string read_all(FILE* f)
{
    std::string text;
    char buf[65536];
    size_t n;
    while((n = fread(buf, 1, sizeof(buf), f)) > 0)
    {
        text.append(buf, n);
    }
    return text;
}

// Parse `ir_text` and lower it to HIP source for `arch`, as
// rocke_engine.lower_serialized_ir_hip does. Returns ROCKE_OK with the source
// in `out`, or the engine's status with its message in `out`, with the
// binding's fallback text when the engine set none. (A parse that returns no
// kernel and no error status, which the parser does not do, is reported as
// ROCKE_ERR_VALUE, since status 0 marks source here.)
rocke_status_t lower_to_hip(const std::string& ir_text, const char* arch, std::string* out)
{
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "ir_lower_cli") != ROCKE_OK)
    {
        *out = "builder init failed (OOM)";
        return ROCKE_ERR_OOM;
    }
    rocke_kernel_def_t* kernel = nullptr;
    rocke_status_t st = rocke_ir_parse(ir_text.c_str(), &b, &kernel);
    if(st != ROCKE_OK || !kernel)
    {
        const char* msg = rocke_ir_builder_error(&b);
        *out = (msg && *msg) ? msg : "unknown parse error";
        rocke_ir_builder_free(&b);
        return st != ROCKE_OK ? st : ROCKE_ERR_VALUE;
    }
    rocke_strbuf_t text;
    if(rocke_strbuf_init(&text, 0) != 0)
    {
        *out = "output allocation failed";
        rocke_ir_builder_free(&b);
        return ROCKE_ERR_OOM;
    }
    rocke_lower_hip_opts_t opts = {};
    opts.arch = arch;
    st = rocke_lower_kernel_to_hip(&b, kernel, &opts, &text);
    if(st == ROCKE_OK)
    {
        *out = rocke_strbuf_cstr(&text);
    }
    else
    {
        const char* msg = rocke_ir_builder_error(&b);
        *out = (msg && *msg) ? msg : "unknown lowering error";
    }
    rocke_strbuf_free(&text);
    rocke_ir_builder_free(&b);
    return st;
}

// The largest serialized kernel a --hip-batch frame may carry; a larger count
// is a malformed header rather than an allocation to attempt.
constexpr size_t kMaxFrameBytes = size_t(1) << 30;

// --hip-batch: answer one frame per kernel until stdin ends (see the header).
int lower_hip_batch()
{
    char arch[64];
    size_t size = 0;
    for(;;)
    {
        const int fields = std::fscanf(stdin, "%63s %zu", arch, &size);
        if(fields == EOF)
        {
            return std::fflush(stdout) == 0 && !std::ferror(stdout) ? 0 : 2;
        }
        if(fields != 2 || size > kMaxFrameBytes || std::fgetc(stdin) != '\n')
        {
            std::fprintf(stderr, "ir_lower_cli: malformed --hip-batch frame header\n");
            return 2;
        }
        if(size == 0)
        {
            std::fprintf(stderr, "ir_lower_cli: empty IR input in a --hip-batch frame\n");
            return 2;
        }
        std::string ir_text(size, '\0');
        if(std::fread(&ir_text[0], 1, size, stdin) != size)
        {
            std::fprintf(stderr, "ir_lower_cli: --hip-batch frame shorter than its count\n");
            return 2;
        }
        std::string result;
        const rocke_status_t st = lower_to_hip(ir_text, arch, &result);
        std::fprintf(stdout, "%d %zu\n", (int)st, result.size());
        if(std::fwrite(result.data(), 1, result.size(), stdout) != result.size())
        {
            std::fprintf(stderr, "ir_lower_cli: cannot write the --hip-batch reply\n");
            return 2;
        }
    }
}

} // namespace

int main(int argc, char** argv)
{
#ifdef _WIN32
    // The engine's text and messages pass through unchanged: no CRLF
    // translation on any stream.
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);
#endif
    if(argc == 2 && std::strcmp(argv[1], "--hip-batch") == 0)
    {
        return lower_hip_batch();
    }

    const char* arch = "gfx950";
    const char* ir_path = nullptr;

    // argv[1] = arch (optional), argv[2] = ir file (optional, "-" = stdin).
    if(argc >= 2 && std::strcmp(argv[1], "-") != 0)
    {
        arch = argv[1];
    }
    if(argc >= 3 && std::strcmp(argv[2], "-") != 0)
    {
        ir_path = argv[2];
    }

    std::string ir_text;
    if(ir_path)
    {
        FILE* f = std::fopen(ir_path, "rb");
        if(!f)
        {
            std::fprintf(stderr, "ir_lower_cli: cannot open '%s'\n", ir_path);
            return 1;
        }
        ir_text = read_all(f);
        std::fclose(f);
    }
    else
    {
        ir_text = read_all(stdin);
    }

    if(ir_text.empty())
    {
        std::fprintf(stderr, "ir_lower_cli: empty IR input\n");
        return 1;
    }

    // Parse the serialized IR back into a KernelDef. The builder owns the arena
    // the whole parsed graph lives in; free it before exit.
    rocke_ir_builder_t b;
    if(rocke_ir_builder_init(&b, "ir_lower_cli") != ROCKE_OK)
    {
        std::fprintf(stderr, "ir_lower_cli: builder init failed (OOM)\n");
        return 1;
    }

    rocke_kernel_def_t* kernel = nullptr;
    rocke_status_t st = rocke_ir_parse(ir_text.c_str(), &b, &kernel);
    if(st != ROCKE_OK || !kernel)
    {
        const char* msg = rocke_ir_builder_error(&b);
        std::fprintf(stderr,
                     "ir_lower_cli: parse failed (status %d): %s\n",
                     (int)st,
                     (msg && *msg) ? msg : "unknown parse error");
        rocke_ir_builder_free(&b);
        return 1;
    }

    // Lower the parsed kernel to LLVM IR for the requested arch. The _ex variant
    // fills a diagnostic buffer on failure so the caller gets a real message.
    char* out_ll = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    st = rocke_lower_kernel_to_llvm_ex(
        kernel, ROCKE_LLVM_FLAVOR_AUTO, arch, &out_ll, err, sizeof(err));
    if(st != ROCKE_OK || !out_ll)
    {
        std::fprintf(stderr,
                     "ir_lower_cli: lower failed for arch '%s' (status %d): %s\n",
                     arch,
                     (int)st,
                     err[0] ? err : "unknown lowering error");
        rocke_ir_builder_free(&b);
        return 1;
    }

    std::fwrite(out_ll, 1, std::strlen(out_ll), stdout);

    std::free(out_ll);
    rocke_ir_builder_free(&b);
    return 0;
}
