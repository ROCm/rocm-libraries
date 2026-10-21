// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// The C++ engine keeps hand-written copies of facts rocKE's arch data files
// record. This test reads the files themselves and compares each copy with
// them, so an edited or regenerated file can never leave the C++ copy stale:
//
//   arch_specs.json "generic_targets"  <->  rocke_ati_generic_targets
//
// argv[1] is the data directory (python/rocke/core/arch/data); the test reads
// the source tree, so it runs from the build tree and is not installed.
#include "rocke/arch_target_internal.h"
#include "rocke/arena.h"
#include "rocke/json_dom.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

static int failures = 0;

// Parse one JSON file into the arena; NULL (counted as a failure) when it
// cannot be read or parsed.
static const jd_val_t* load_json(const fs::path& path, rocke_arena_t* arena)
{
    std::ifstream in(path, std::ios::binary);
    if(!in)
    {
        fprintf(stderr, "FAIL: cannot read %s\n", path.string().c_str());
        ++failures;
        return NULL;
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();
    char err[256];
    const jd_val_t* root = rocke_json_parse(text.c_str(), arena, err, sizeof(err));
    if(root == NULL)
    {
        fprintf(stderr, "FAIL: cannot parse %s: %s\n", path.string().c_str(), err);
        ++failures;
    }
    return root;
}

using generic_list = std::vector<std::pair<std::string, std::vector<std::string>>>;

static void print_generic_list(const char* source, const generic_list& list)
{
    fprintf(stderr, "  %s:\n", source);
    for(const auto& [generic, members] : list)
    {
        fprintf(stderr, "    %s:", generic.c_str());
        for(const std::string& member : members)
            fprintf(stderr, " %s", member.c_str());
        fprintf(stderr, "\n");
    }
}

// rocke_ati_generic_targets lists exactly the generic targets of
// arch_specs.json "generic_targets", with the same members in the same order.
static void check_generic_targets(const jd_val_t* specs)
{
    const jd_val_t* generic_targets = rocke_jget(specs, "generic_targets");
    if(generic_targets == NULL || generic_targets->kind != JD_OBJ)
    {
        fprintf(stderr, "FAIL: arch_specs.json has no generic_targets object\n");
        ++failures;
        return;
    }
    generic_list in_file;
    for(int i = 0; i < generic_targets->obj_len; ++i)
    {
        const jd_member_t& entry = generic_targets->obj[i];
        if(entry.key[0] == '_') // a comment, not a generic target
            continue;
        std::vector<std::string> members;
        for(int m = 0; entry.val->kind == JD_ARR && m < entry.val->arr_len; ++m)
        {
            const char* member = rocke_jstr(entry.val->arr[m]);
            members.emplace_back(member ? member : "(not a string)");
        }
        in_file.emplace_back(entry.key, members);
    }
    generic_list in_cpp;
    for(const rocke_ati_generic_row_t* row = rocke_ati_generic_targets; row->generic != NULL; ++row)
    {
        std::vector<std::string> members;
        for(const char* const* member = row->members; *member != NULL; ++member)
            members.emplace_back(*member);
        in_cpp.emplace_back(row->generic, members);
    }
    if(in_file != in_cpp)
    {
        fprintf(stderr,
                "FAIL: rocke_ati_generic_targets (cpp/core/arch/data.cpp) differs from "
                "arch_specs.json generic_targets\n");
        print_generic_list("arch_specs.json", in_file);
        print_generic_list("data.cpp", in_cpp);
        ++failures;
    }
}

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        fprintf(stderr, "usage: %s <python/rocke/core/arch/data directory>\n", argv[0]);
        return 2;
    }
    const fs::path data_dir = argv[1];
    rocke_arena_t arena;
    if(rocke_arena_init(&arena, 0) != 0)
    {
        fprintf(stderr, "FAIL: cannot allocate the JSON arena\n");
        return 1;
    }
    if(const jd_val_t* specs = load_json(data_dir / "arch_specs.json", &arena))
        check_generic_targets(specs);
    rocke_arena_destroy(&arena);
    return failures == 0 ? 0 : 1;
}
