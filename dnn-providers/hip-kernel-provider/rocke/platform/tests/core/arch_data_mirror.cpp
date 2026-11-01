// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// The C++ engine keeps hand-written copies of facts rocKE's arch data files
// record. This test reads the files themselves and compares each copy with
// them, so an edited or regenerated file can never leave the C++ copy stale:
//
//   arch_specs.json "generic_targets"  <->  rocke_ati_generic_targets
//   "processors" of every intrinsic_arch_domain.<flavor>.json column
//                                      <->  rocke_ati_rejected_processors
//
// argv[1] is the data directory (python/rocke/core/arch/data); the test reads
// the source tree, so it runs from the build tree and is not installed.
#include "rocke/arch_target_internal.h"
#include "rocke/arena.h"
#include "rocke/json_dom.h"

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
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

// rocke_ati_rejected_processors holds exactly the (flavor, processor) pairs
// that some committed arch-domain column records as not accepted, for every
// column in the directory.
static void check_rejected_processors(const fs::path& data_dir, rocke_arena_t* arena)
{
    const std::string prefix = "intrinsic_arch_domain.";
    const std::string suffix = ".json";
    std::set<std::pair<std::string, std::string>> in_columns;
    int columns = 0;
    for(const fs::directory_entry& entry : fs::directory_iterator(data_dir))
    {
        const std::string name = entry.path().filename().string();
        if(name.size() <= prefix.size() + suffix.size() || name.rfind(prefix, 0) != 0
           || name.compare(name.size() - suffix.size(), suffix.size(), suffix) != 0)
            continue;
        const std::string flavor
            = name.substr(prefix.size(), name.size() - prefix.size() - suffix.size());
        ++columns;
        const jd_val_t* column = load_json(entry.path(), arena);
        const jd_val_t* processors = rocke_jget(column, "processors");
        if(processors == NULL || processors->kind != JD_OBJ)
        {
            fprintf(stderr, "FAIL: %s has no processors object\n", name.c_str());
            ++failures;
            continue;
        }
        for(int i = 0; i < processors->obj_len; ++i)
        {
            const jd_val_t* accepted = rocke_jget(processors->obj[i].val, "accepted");
            if(accepted == NULL || accepted->kind != JD_BOOL)
            {
                fprintf(stderr,
                        "FAIL: %s processor %s has no boolean 'accepted'\n",
                        name.c_str(),
                        processors->obj[i].key);
                ++failures;
            }
            else if(!accepted->b)
                in_columns.emplace(flavor, processors->obj[i].key);
        }
    }
    if(columns == 0)
    {
        fprintf(stderr,
                "FAIL: no %s<flavor>%s column in %s\n",
                prefix.c_str(),
                suffix.c_str(),
                data_dir.string().c_str());
        ++failures;
    }
    std::set<std::pair<std::string, std::string>> in_cpp;
    for(const rocke_ati_rejected_processor_t* row = rocke_ati_rejected_processors;
        row->llvm_flavor != NULL;
        ++row)
        in_cpp.emplace(row->llvm_flavor, row->gfx);
    for(const auto& [flavor, gfx] : in_columns)
        if(in_cpp.count({flavor, gfx}) == 0)
        {
            fprintf(stderr,
                    "FAIL: the %s column records %s as not accepted, but "
                    "rocke_ati_rejected_processors (cpp/core/arch/data.cpp) does not list it\n",
                    flavor.c_str(),
                    gfx.c_str());
            ++failures;
        }
    for(const auto& [flavor, gfx] : in_cpp)
        if(in_columns.count({flavor, gfx}) == 0)
        {
            fprintf(stderr,
                    "FAIL: rocke_ati_rejected_processors (cpp/core/arch/data.cpp) lists %s at "
                    "%s, but no committed %s column records it as not accepted\n",
                    gfx.c_str(),
                    flavor.c_str(),
                    flavor.c_str());
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
    check_rejected_processors(data_dir, &arena);
    rocke_arena_destroy(&arena);
    return failures == 0 ? 0 : 1;
}
