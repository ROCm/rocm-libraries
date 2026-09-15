// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/ir_lower_cli.cpp -- family-agnostic IR-artifact lowering tool.
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
 *
 * On any parse or lowering failure a clean message is written to stderr and the
 * process exits nonzero. The engine's extern "C" boundary already converts every
 * internal failure (including raised exceptions) into a status code and an error
 * string, so this tool never aborts/terminates -- it only reports and exits.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "rocke/ir.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"

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

} // namespace

int main(int argc, char** argv)
{
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
