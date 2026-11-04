// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/* Native mirror of test_f64_mma.py's fixed probe contract and valid controls. */
#include "f64_mma_probe_internal.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <vector>

#include "rocke/instance_f64_mma_probe.h"
#include "rocke/ir_serialize.h"
#include "rocke/lower_llvm.h"

#define CHECK(expr)                                                 \
    do                                                              \
    {                                                               \
        if(!(expr))                                                 \
        {                                                           \
            std::fprintf(stderr, "line %d: %s\n", __LINE__, #expr); \
            return 1;                                               \
        }                                                           \
    } while(0)

static int test_rejects_incompatible_contract(bool safe_witness_only)
{
    const auto* target = rocke_arch_target_from_gfx("gfx942");
    int rejected = 0;
    for(int m : {16, 4})
        for(int role : {0, 1, 2, 3, -1})
            for(const char* field : {"dtype",
                                     "width_short",
                                     "width_long",
                                     "missing_map",
                                     "no_fn",
                                     "wrong_map",
                                     "map_width",
                                     "map_wave",
                                     "missing_atom",
                                     "op_id",
                                     "family",
                                     "m",
                                     "n",
                                     "k",
                                     "wave",
                                     "both_width_short",
                                     "both_capacity",
                                     "both_wrong_map"})
            {
                bool descriptor
                    = !std::strcmp(field, "dtype") || !std::strcmp(field, "width_short")
                      || !std::strcmp(field, "width_long") || !std::strcmp(field, "missing_map")
                      || !std::strcmp(field, "no_fn") || !std::strcmp(field, "wrong_map")
                      || !std::strcmp(field, "map_width") || !std::strcmp(field, "map_wave");
                if(descriptor != (role >= 0))
                    continue;
                if(safe_witness_only && (role != 3 || std::strcmp(field, "width_short")))
                    continue;
                std::vector<rocke_mma_op_t> ops(target->mma.ops,
                                                target->mma.ops + target->mma.num_ops);
                rocke_mma_catalog_t catalog = {ops.data(), static_cast<int>(ops.size())};
                const auto* selected = rocke_mma_catalog_op_for_shape(
                    &catalog, "mma", "fp64", "fp64", "fp64", m, m, 4, nullptr);
                CHECK(selected);
                auto& atom = ops[selected - ops.data()];
                auto src_map = *atom.srcs[2].layout;
                auto dst_map = *atom.dst.layout;
                rocke_layout_map_t map = role < 0    ? dst_map
                                         : role == 3 ? dst_map
                                                     : *atom.srcs[role].layout;
                if(role >= 0)
                {
                    if(role == 3)
                        atom.dst.layout = &map;
                    else
                        atom.srcs[role].layout = &map;
                    // The dst descriptor is a different C type; mutate shared fields explicitly.
                    if(!std::strcmp(field, "dtype"))
                    {
                        if(role == 3)
                            atom.dst.dtype = "fp32";
                        else
                            atom.srcs[role].dtype = "fp32";
                    }
                    else if(!std::strncmp(field, "width_", 6))
                    {
                        int delta = !std::strcmp(field, "width_short") ? -1 : 1;
                        if(role == 3)
                            atom.dst.frag_len += delta;
                        else
                            atom.srcs[role].frag_len += delta;
                    }
                    else if(!std::strcmp(field, "missing_map"))
                    {
                        if(role == 3)
                            atom.dst.layout = nullptr;
                        else
                            atom.srcs[role].layout = nullptr;
                    }
                    else if(!std::strcmp(field, "no_fn"))
                        map.fn = nullptr;
                    else if(!std::strcmp(field, "wrong_map"))
                        map.fn = role == 0 ? atom.srcs[1].layout->fn : atom.srcs[0].layout->fn;
                    else if(!std::strcmp(field, "map_width"))
                        ++map.frag_len;
                    else
                        map.wave_size = 32;
                }
                else if(!std::strcmp(field, "missing_atom"))
                    catalog.num_ops = 0;
                else if(!std::strcmp(field, "op_id"))
                    atom.op_id = "wrong";
                else if(!std::strcmp(field, "family"))
                    atom.family = "wrong";
                else if(!std::strcmp(field, "m"))
                    ++atom.m;
                else if(!std::strcmp(field, "n"))
                    ++atom.n;
                else if(!std::strcmp(field, "k"))
                    atom.k = 8;
                else if(!std::strcmp(field, "wave"))
                    atom.wave_size = 32;
                else if(!std::strcmp(field, "both_wrong_map"))
                {
                    src_map.fn = dst_map.fn = atom.srcs[0].layout->fn;
                    atom.srcs[2].layout = &src_map;
                    atom.dst.layout = &dst_map;
                }
                else
                    atom.srcs[2].frag_len = atom.dst.frag_len
                        = !std::strcmp(field, "both_capacity") ? 5 : atom.dst.frag_len - 1;
                rocke_ir_builder_t b;
                auto* kernel = ckc::build_f64_mma_probe(&b, m, 4, &catalog);
                // Free before assertions: even an unexpectedly admitted contract owns an arena.
                bool correct
                    = kernel == nullptr && b.status == ROCKE_ERR_VALUE
                      && !std::strcmp(
                          b.err,
                          "F64 probe requires the fixed gfx942 instruction and layout contract")
                      && b.kernel->num_params == 0 && b.kernel->body->num_ops == 0;
                rocke_ir_builder_free(&b);
                if(!correct)
                    std::fprintf(stderr, "m=%d role=%d field=%s\n", m, role, field);
                CHECK(correct);
                ++rejected;
            }
    CHECK(rejected == (safe_witness_only ? 2 : 84));
    std::printf("F64 contract rejections: %d\n", rejected);
    return 0;
}

static int test_valid_copied_contracts()
{
    const auto* target = rocke_arch_target_from_gfx("gfx942");
    for(int m : {16, 4})
        for(int k : {4, 8, 64})
        {
            std::vector<rocke_mma_op_t> ops(target->mma.ops, target->mma.ops + target->mma.num_ops);
            rocke_mma_catalog_t catalog = {ops.data(), static_cast<int>(ops.size())};
            const auto* selected = rocke_mma_catalog_op_for_shape(
                &catalog, "mma", "fp64", "fp64", "fp64", m, m, 4, nullptr);
            CHECK(selected);
            auto& atom = ops[selected - ops.data()];
            rocke_layout_map_t maps[] = {
                *atom.srcs[0].layout, *atom.srcs[1].layout, *atom.srcs[2].layout, *atom.dst.layout};
            for(int role = 0; role < 3; ++role)
                atom.srcs[role].layout = &maps[role];
            atom.dst.layout = &maps[3];
            rocke_ir_builder_t public_builder, copied_builder;
            auto* expected = rocke_build_f64_mma_probe(&public_builder, m, k);
            auto* copied = ckc::build_f64_mma_probe(&copied_builder, m, k, &catalog);
            CHECK(expected && copied);
            char *public_ir = nullptr, *copied_ir = nullptr;
            CHECK(rocke_ir_serialize(expected, &public_ir) == ROCKE_OK);
            CHECK(rocke_ir_serialize(copied, &copied_ir) == ROCKE_OK);
            CHECK(!std::strcmp(public_ir, copied_ir));
            for(auto flavor : {ROCKE_LLVM_FLAVOR_LLVM20,
                               ROCKE_LLVM_FLAVOR_LLVM22,
                               ROCKE_LLVM_FLAVOR_LLVM23,
                               ROCKE_LLVM_FLAVOR_LLVM24})
            {
                char *public_ll = nullptr, *copied_ll = nullptr;
                CHECK(rocke_lower_kernel_to_llvm(expected, flavor, "gfx942", &public_ll)
                      == ROCKE_OK);
                CHECK(rocke_lower_kernel_to_llvm(copied, flavor, "gfx942", &copied_ll) == ROCKE_OK);
                CHECK(!std::strcmp(public_ll, copied_ll));
                std::free(public_ll);
                std::free(copied_ll);
            }
            std::free(public_ir);
            std::free(copied_ir);
            rocke_ir_builder_free(&public_builder);
            rocke_ir_builder_free(&copied_builder);
        }
    return 0;
}

int main(int argc, char** argv)
{
    bool safe = argc == 2 && !std::strcmp(argv[1], "--safe-witness-only");
    CHECK(test_rejects_incompatible_contract(safe) == 0);
    if(!safe)
        CHECK(test_valid_copied_contracts() == 0);
    return 0;
}
