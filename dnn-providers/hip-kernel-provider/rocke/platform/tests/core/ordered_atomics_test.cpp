// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// Independent C-builder proof and host contract tests; no GPU execution.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

#include "rocke/ir.h"
#include "rocke/lower_llvm.h"
#include "rocke/verify.h"

static_assert(ROCKE_OP_MEMREF_GLOBAL_ATOMIC_LOAD_I32 == ROCKE_OP_CF_RETURN + 1);
static_assert(ROCKE_OP_MEMREF_GLOBAL_FENCE == ROCKE_OP_CF_RETURN + 3);

struct builder_owner
{
    rocke_ir_builder_t b {};
    builder_owner()
    {
        if (rocke_ir_builder_init(&b, "ordered_atomics") != ROCKE_OK)
            throw std::runtime_error("builder init failed");
    }
    ~builder_owner() { rocke_ir_builder_free(&b); }
};

static void require(bool ok, const char* message)
{
    if (!ok)
        throw std::runtime_error(message);
}

static void verify_clean(rocke_kernel_def_t* kernel)
{
    rocke_diag_t* diagnostics = nullptr;
    size_t count = 0;
    require(rocke_verify(kernel, &diagnostics, &count) == ROCKE_OK, "verify failed");
    bool clean = count == 0;
    rocke_diags_free(diagnostics, count);
    require(clean, "valid atomic kernel did not verify cleanly");
}

static void verify_message(rocke_kernel_def_t* kernel, const char* message)
{
    rocke_diag_t* diagnostics = nullptr;
    size_t count = 0;
    require(rocke_verify(kernel, &diagnostics, &count) == ROCKE_OK, "verify failed");
    bool found = false;
    for (size_t i = 0; i < count; ++i)
        found |= strcmp(diagnostics[i].message, message) == 0;
    rocke_diags_free(diagnostics, count);
    require(found, "expected verifier diagnostic absent");
}

static std::string emit(const char* flavor, const char* arch, const char* index_type,
    const char* scope, const char* load_order, const char* store_order, const char* fence_order,
    bool mutate = false)
{
    builder_owner owner;
    auto* b = &owner.b;
    auto* kernel = rocke_ir_builder_kernel(b);
    rocke_attr_set_int(b, &kernel->attrs, "max_workgroup_size", 64);
    auto* flag = rocke_b_param(b, "flag", rocke_ptr_type(b, rocke_i32(), "global"), nullptr);
    auto* index = rocke_b_param(
        b, "index", strcmp(index_type, "i64") == 0 ? rocke_i64() : rocke_i32(), nullptr);
    auto* value = rocke_b_param(b, "value", rocke_i32(), nullptr);
    auto* loaded = rocke_b_global_atomic_load_i32(b, flag, index, load_order, scope);
    rocke_b_global_atomic_store_i32(b, flag, index, value, store_order, scope);
    rocke_b_global_fence(b, fence_order, scope);
    rocke_b_ret(b);
    verify_clean(kernel);
    require(!rocke_opcode_is_pure(loaded->op->opcode), "load must be effectful");
    require(!rocke_opcode_is_pure(ROCKE_OP_MEMREF_GLOBAL_ATOMIC_STORE_I32), "store is effectful");
    require(!rocke_opcode_is_pure(ROCKE_OP_MEMREF_GLOBAL_FENCE), "fence is effectful");
    char* text = nullptr;
    auto status
        = rocke_lower_kernel_to_llvm(kernel, rocke_llvm_flavor_from_name(flavor), arch, &text);
    require(status == ROCKE_OK && text, "atomic lowering failed");
    std::string result(text);
    free(text);
    require(result.find("no.remote.memory") == std::string::npos, "remote-exclusion metadata");
    require(result.find("no.fine.grained.memory") == std::string::npos, "local-only metadata");
    if (mutate)
    {
        rocke_attr_set_str(b, &loaded->op->attrs, "scope", "workgroup");
        verify_message(kernel, "memref.global_atomic_load_i32: invalid scope");
        rocke_attr_set_str(b, &loaded->op->attrs, "scope", "system");
        rocke_attr_set_str(b, &loaded->op->attrs, "ordering", "release");
        verify_message(kernel, "memref.global_atomic_load_i32: invalid ordering");
        rocke_attr_set_str(b, &loaded->op->attrs, "ordering", "acquire");
        loaded->type = rocke_i64();
        verify_message(kernel, "memref.global_atomic_load_i32: result must be i32");
    }
    return result;
}

static void rejection_tests()
{
    for (const char* bad : { "release", "acq_rel", "unordered" })
    {
        builder_owner owner;
        auto* b = &owner.b;
        auto* flag = rocke_b_param(b, "flag", rocke_ptr_type(b, rocke_i32(), "global"), nullptr);
        auto* index = rocke_b_param(b, "index", rocke_i32(), nullptr);
        int before = rocke_ir_builder_kernel(b)->body->num_ops;
        bool rejected = false;
        try
        {
            rocke_b_global_atomic_load_i32(b, flag, index, bad, "system");
        }
        catch (const std::exception& error)
        {
            rejected = std::string(error.what()).find("ordering") != std::string::npos;
        }
        require(rejected, "invalid load ordering accepted");
        require(rocke_ir_builder_kernel(b)->body->num_ops == before, "rejection emitted SSA");
    }
}

int main(int argc, char** argv)
{
    try
    {
        if (argc == 9 && strcmp(argv[1], "--emit") == 0)
        {
            auto text = emit(argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8]);
            std::fputs(text.c_str(), stdout);
            return 0;
        }
        require(argc == 1,
            "expected --emit flavor arch index_type scope load_order store_order fence_order");
        for (const char* arch : { "gfx942", "gfx950" })
            for (const char* flavor : { "llvm20", "llvm22", "llvm23" })
                for (const char* index : { "i32", "i64" })
                    for (const char* scope : { "system", "agent" })
                        for (const char* load : { "monotonic", "acquire", "seq_cst" })
                            for (const char* store : { "monotonic", "release", "seq_cst" })
                                for (const char* fence :
                                    { "acquire", "release", "acq_rel", "seq_cst" })
                                    emit(flavor, arch, index, scope, load, store, fence);
        emit("llvm20", "gfx950", "i32", nullptr, nullptr, nullptr, nullptr, true);
        rejection_tests();
        std::puts("ordered atomic host contracts passed");
        return 0;
    }
    catch (const std::exception& error)
    {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
