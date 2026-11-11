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
 * engine would emit -- with no per-family C builder involved. With --hip it
 * lowers with rocke_lower_kernel_to_hip instead and writes the HIP source, as
 * rocke_engine.lower_serialized_ir_hip returns it.
 *
 * Usage:
 *   ir_lower_cli [--hip] [arch] [ir_file]
 *     --hip    lower to HIP source rather than LLVM IR.
 *     arch     ISA target gfx string; defaults to "gfx950".
 *     ir_file  path to serialized IR; "-" or omitted reads stdin.
 *
 * On any parse or lowering failure a clean message is written to stderr and the
 * process exits nonzero. The message ends "(status <code>): <engine message>",
 * the form rocke_engine's errors take, so a caller reads the engine's status
 * and message the same way from either. The engine's extern "C" boundary already converts every
 * internal failure (including raised exceptions) into a status code and an error
 * string, so this tool never aborts/terminates -- it only reports and exits.
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

// Lower `kernel` to HIP source for `arch` with the defaults
// rocke_engine.lower_serialized_ir_hip uses, and write it to stdout.
int lower_to_hip(rocke_ir_builder_t* b, const rocke_kernel_def_t* kernel, const char* arch)
{
    rocke_strbuf_t text;
    if(rocke_strbuf_init(&text, 0) != 0)
    {
        std::fprintf(stderr, "ir_lower_cli: output allocation failed\n");
        return 1;
    }
    rocke_lower_hip_opts_t opts = {};
    opts.arch = arch;
    const rocke_status_t st = rocke_lower_kernel_to_hip(b, kernel, &opts, &text);
    if(st != ROCKE_OK)
    {
        const char* msg = rocke_ir_builder_error(b);
        std::fprintf(stderr,
                     "ir_lower_cli: lower failed for arch '%s' (status %d): %s\n",
                     arch,
                     (int)st,
                     (msg && *msg) ? msg : "unknown lowering error");
        rocke_strbuf_free(&text);
        return 1;
    }
    const char* source = rocke_strbuf_cstr(&text);
    std::fwrite(source, 1, std::strlen(source), stdout);
    rocke_strbuf_free(&text);
    return 0;
}

} // namespace

int main(int argc, char** argv)
{
    const char* arch = "gfx950";
    const char* ir_path = nullptr;
    bool hip = false;

#ifdef _WIN32
    // The engine's text and messages pass through unchanged: no CRLF
    // translation on any stream.
    _setmode(_fileno(stdin), _O_BINARY);
    _setmode(_fileno(stdout), _O_BINARY);
    _setmode(_fileno(stderr), _O_BINARY);
#endif

    // argv[1] = --hip (optional), then arch (optional), then ir file
    // (optional, "-" = stdin).
    int positional = 1;
    if(argc > positional && std::strcmp(argv[positional], "--hip") == 0)
    {
        hip = true;
        ++positional;
    }
    if(argc > positional && std::strcmp(argv[positional], "-") != 0)
    {
        arch = argv[positional];
    }
    if(argc > positional + 1 && std::strcmp(argv[positional + 1], "-") != 0)
    {
        ir_path = argv[positional + 1];
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

    if(hip)
    {
        const int exit_code = lower_to_hip(&b, kernel, arch);
        rocke_ir_builder_free(&b);
        return exit_code;
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
