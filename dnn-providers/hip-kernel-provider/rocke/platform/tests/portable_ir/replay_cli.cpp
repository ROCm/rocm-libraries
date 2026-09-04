// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/portable_ir/replay_cli.cpp -- lower a portable-IR artifact to AMDGPU
 * LLVM IR, from a standalone binary.
 *
 * This is the deployment shape of the portable-IR path made concrete: an
 * executable linked against rocke_core that reads an artifact off disk and
 * writes .ll to stdout. There is no Python in the process. The online path
 * (cpp/portable_ir/online.cpp, driven over ctypes from
 * python/rocke/portable_ir/src/online.py) does exactly the same work in-process;
 * this CLI is the offline twin, and is what you reach for when bisecting a
 * parity failure without a Python stack in the way.
 *
 * It is a sibling of tests/core/ir_lower_cli.cpp, which lowers rocke's own
 * `ckdsl.ir/v1` serialized text. The two formats are different: this one reads
 * the artifacts the Python front end exports (`rocke.ir/v1` JSON) and the
 * recipes it records (`rocke.recipe/v1`, JSON or CBOR, bare or bundled).
 *
 * Usage:
 *   replay_cli --ir      <kernel.ir.json>            [common opts]
 *   replay_cli --recipe  <recipe.json>               [common opts] [spec opts]
 *   replay_cli --recipe  <recipe.cbor> --cbor        [common opts] [spec opts]
 *   replay_cli --bundle  <bundle.cbor> --key KEY     [common opts] [spec opts]
 *
 *   common opts: --arch GFX (default gfx950), --flavor llvm20|llvm22|llvm23
 *   spec opts  : --int NAME=VALUE, --str NAME=VALUE  (repeatable)
 *
 * Exits nonzero with a message on stderr for any read, parse, replay, or
 * lowering failure; the engine's extern "C" boundary turns internal faults into
 * status codes, so this never aborts.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rocke/ir.h"
#include "rocke/ir_import.h"
#include "rocke/lower_llvm.h"
#include "rocke/recipe_vm.h"

namespace
{

enum class Mode
{
    None,
    Ir,
    Recipe,
    Bundle
};

void usage(const char* argv0)
{
    std::fprintf(stderr,
                 "usage: %s (--ir FILE | --recipe FILE [--cbor] | --bundle FILE --key KEY)\n"
                 "          [--arch GFX] [--flavor llvm20|llvm22|llvm23]\n"
                 "          [--int NAME=VALUE]... [--str NAME=VALUE]...\n",
                 argv0);
}

bool read_file(const char* path, std::string* out)
{
    FILE* f = std::fopen(path, "rb");
    if(!f)
    {
        std::fprintf(stderr, "replay_cli: cannot open '%s'\n", path);
        return false;
    }
    char buf[65536];
    size_t n;
    while((n = std::fread(buf, 1, sizeof(buf), f)) > 0)
    {
        out->append(buf, n);
    }
    std::fclose(f);
    return true;
}

// Split "NAME=VALUE" at the first '='. Returns false when there is no '='.
bool split_kv(const char* arg, std::string* key, std::string* val)
{
    const char* eq = std::strchr(arg, '=');
    if(!eq)
    {
        return false;
    }
    key->assign(arg, static_cast<size_t>(eq - arg));
    val->assign(eq + 1);
    return true;
}

} // namespace

int main(int argc, char** argv)
{
    Mode mode = Mode::None;
    const char* path = nullptr;
    const char* key = nullptr;
    const char* arch = "gfx950";
    const char* flavor_name = nullptr;
    bool cbor = false;

    // The spec name/value strings must outlive the spec arrays handed to the VM,
    // so they are owned here and only pointed at below.
    std::vector<std::string> int_names, str_names, str_vals;
    std::vector<long> int_vals;

    for(int i = 1; i < argc; ++i)
    {
        const char* a = argv[i];
        const bool has_next = (i + 1) < argc;
        if(std::strcmp(a, "--ir") == 0 && has_next)
        {
            mode = Mode::Ir;
            path = argv[++i];
        }
        else if(std::strcmp(a, "--recipe") == 0 && has_next)
        {
            mode = Mode::Recipe;
            path = argv[++i];
        }
        else if(std::strcmp(a, "--bundle") == 0 && has_next)
        {
            mode = Mode::Bundle;
            path = argv[++i];
            cbor = true; // bundles only ship as CBOR
        }
        else if(std::strcmp(a, "--key") == 0 && has_next)
        {
            key = argv[++i];
        }
        else if(std::strcmp(a, "--arch") == 0 && has_next)
        {
            arch = argv[++i];
        }
        else if(std::strcmp(a, "--flavor") == 0 && has_next)
        {
            flavor_name = argv[++i];
        }
        else if(std::strcmp(a, "--cbor") == 0)
        {
            cbor = true;
        }
        else if((std::strcmp(a, "--int") == 0 || std::strcmp(a, "--str") == 0) && has_next)
        {
            const bool is_int = std::strcmp(a, "--int") == 0;
            std::string k, v;
            if(!split_kv(argv[++i], &k, &v))
            {
                std::fprintf(stderr, "replay_cli: expected NAME=VALUE after %s\n", a);
                return 2;
            }
            if(is_int)
            {
                int_names.push_back(k);
                int_vals.push_back(std::strtol(v.c_str(), nullptr, 10));
            }
            else
            {
                str_names.push_back(k);
                str_vals.push_back(v);
            }
        }
        else
        {
            std::fprintf(stderr, "replay_cli: unrecognized argument '%s'\n", a);
            usage(argv[0]);
            return 2;
        }
    }

    if(mode == Mode::None || !path)
    {
        usage(argv[0]);
        return 2;
    }
    if(mode == Mode::Bundle && !key)
    {
        std::fprintf(stderr, "replay_cli: --bundle requires --key\n");
        return 2;
    }

    rocke_llvm_flavor_t flavor = ROCKE_LLVM_FLAVOR_AUTO;
    if(flavor_name)
    {
        flavor = rocke_llvm_flavor_from_name(flavor_name);
        if(flavor == ROCKE_LLVM_FLAVOR_AUTO)
        {
            std::fprintf(stderr, "replay_cli: unknown --flavor '%s'\n", flavor_name);
            return 2;
        }
    }

    std::string blob;
    if(!read_file(path, &blob))
    {
        return 1;
    }
    if(blob.empty())
    {
        std::fprintf(stderr, "replay_cli: '%s' is empty\n", path);
        return 1;
    }

    std::vector<rocke_recipe_spec_int_t> ints;
    ints.reserve(int_names.size());
    for(size_t i = 0; i < int_names.size(); ++i)
    {
        ints.push_back({int_names[i].c_str(), int_vals[i]});
    }
    std::vector<rocke_recipe_spec_str_t> strs;
    strs.reserve(str_names.size());
    for(size_t i = 0; i < str_names.size(); ++i)
    {
        strs.push_back({str_names[i].c_str(), str_vals[i].c_str()});
    }

    const auto* data = reinterpret_cast<const unsigned char*>(blob.data());
    const int n_ints = static_cast<int>(ints.size());
    const int n_strs = static_cast<int>(strs.size());

    rocke_ir_builder_t b;
    rocke_kernel_def_t* kernel = nullptr;
    char err[ROCKE_ERR_MSG_CAP];
    err[0] = '\0';
    rocke_status_t st = ROCKE_OK;

    switch(mode)
    {
    case Mode::Ir:
        // Concrete portable-IR graph: replay the ops through the C builder.
        st = rocke_import_kernel_from_json(blob.c_str(), nullptr, &b, &kernel, err, sizeof(err));
        break;
    case Mode::Recipe:
        // Builder recipe: the VM re-runs the authoring algorithm against the spec.
        st = cbor ? rocke_recipe_run_from_cbor(data,
                                               blob.size(),
                                               ints.data(),
                                               n_ints,
                                               strs.data(),
                                               n_strs,
                                               &b,
                                               &kernel,
                                               err,
                                               sizeof(err))
                  : rocke_recipe_run_from_json(blob.c_str(),
                                               ints.data(),
                                               n_ints,
                                               strs.data(),
                                               n_strs,
                                               &b,
                                               &kernel,
                                               err,
                                               sizeof(err));
        break;
    case Mode::Bundle:
        st = rocke_recipe_run_from_bundle_cbor(data,
                                               blob.size(),
                                               key,
                                               arch,
                                               ints.data(),
                                               n_ints,
                                               strs.data(),
                                               n_strs,
                                               &b,
                                               &kernel,
                                               err,
                                               sizeof(err));
        break;
    case Mode::None:
        return 2;
    }

    if(st != ROCKE_OK || !kernel)
    {
        std::fprintf(stderr,
                     "replay_cli: replay of '%s' failed (status %d): %s\n",
                     path,
                     (int)st,
                     err[0] ? err : "unknown error");
        return 1;
    }

    char* out_ll = nullptr;
    char lerr[ROCKE_ERR_MSG_CAP];
    lerr[0] = '\0';
    st = rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, &out_ll, lerr, sizeof(lerr));
    if(st != ROCKE_OK || !out_ll)
    {
        std::fprintf(stderr,
                     "replay_cli: lower for arch '%s' failed (status %d): %s\n",
                     arch,
                     (int)st,
                     lerr[0] ? lerr : "unknown lowering error");
        rocke_ir_builder_free(&b);
        return 1;
    }

    std::fwrite(out_ll, 1, std::strlen(out_ll), stdout);
    std::free(out_ll);
    rocke_ir_builder_free(&b);
    return 0;
}
