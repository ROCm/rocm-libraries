// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#include "rocke/instance_gfx950_attention_dense_builder.h"
#include "rocke/ir_serialize.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <latch>
#include <memory>
#include <string_view>
#include <thread>

namespace
{
using Spec = rocke_gfx950_attention_dense_spec_t;
using Text = std::unique_ptr<char, decltype(&std::free)>;

bool refused(rocke_ir_builder_t& b,
             const Spec* spec,
             const char* arch,
             rocke_status_t status,
             const char* prefix)
{
    auto* kernel = rocke_gfx950_build_attention_dense_new(&b, spec, arch);
    const auto actual = rocke_ir_builder_status(&b);
    const char* error = rocke_ir_builder_error(&b);
    const bool ok
        = !kernel && actual == status && error && std::string_view(error).starts_with(prefix);
    if(!ok)
        std::fprintf(stderr,
                     "expected refusal %d '%s...', got kernel=%p status=%d '%s'\n",
                     int(status),
                     prefix,
                     static_cast<void*>(kernel),
                     int(actual),
                     error ? error : "");
    rocke_ir_builder_free(&b);
    return ok;
}

// Preserve the serializer's own bytes independently of the arena.
Text snapshot(rocke_ir_builder_t& b, rocke_kernel_def_t* kernel, const char* label)
{
    char* text = nullptr;
    if(!kernel || rocke_ir_builder_status(&b) != ROCKE_OK || rocke_ir_builder_kernel(&b) != kernel)
        std::fprintf(stderr, "%s: build failed: %s\n", label, rocke_ir_builder_error(&b));
    else if(rocke_ir_serialize(kernel, &text) != ROCKE_OK || !text)
        std::fprintf(stderr, "%s: could not serialize returned kernel\n", label);
    return Text(text, &std::free);
}

bool scheduled_as(std::string_view ir, bool persistent)
{
    // Both variants do real attention math. Only the ordinary grid decodes
    // multiple block-id axes; persistent CTAs stride over work items from x.
    constexpr std::string_view id = "gpu.block_id";
    const auto first = ir.find(id);
    return ir.find("mfma_f32_32x32x16_bf16") != ir.npos && first != ir.npos
           && (ir.find(id, first + id.size()) == ir.npos) == persistent;
}
} // namespace

int main()
{
    Spec grid = rocke_gfx950_attention_dense_spec_default();
    grid.base.batch = 1;
    grid.base.num_query_heads = 4;
    grid.base.num_kv_heads = 2;
    grid.base.seqlen_q = grid.base.seqlen_kv = 256;
    grid.base.head_size = 64;
    grid.base.block_m = 128;
    grid.base.block_n = 32;
    grid.base.dtype = "bf16";
    grid.base.causal = false;
    Spec persistent = grid;
    persistent.base.persistent = true;
    persistent.base.num_persistent = 2;

    // The entry point initializes fresh storage, even when rejecting before
    // arena allocation. The caller frees after *each* call before reusing b.
    rocke_ir_builder_t b;
    Spec invalid = grid;
    invalid.base.head_size = 63;
    if(!refused(
           b, &invalid, "gfx942", ROCKE_ERR_NOTIMPL, "attention_dense is gfx950-only (got gfx942)")
       || !refused(
           b,
           &invalid,
           "gfx950",
           ROCKE_ERR_VALUE,
           "unsupported gfx950 attention_dense spec: invalid Gfx950AttentionDenseSpec: head_size")
       || !refused(b,
                   nullptr,
                   nullptr,
                   ROCKE_ERR_VALUE,
                   "unsupported gfx950 attention_dense spec: attention dense spec is NULL"))
        return 1;
    Spec unsupported = grid;
    unsupported.kv_segments = 0; // Constructible, rejected by the support gate.
    if(!refused(b,
                &unsupported,
                nullptr,
                ROCKE_ERR_VALUE,
                "unsupported gfx950 attention_dense spec: kv_segments"))
        return 1;

    auto* kernel = rocke_gfx950_build_attention_dense_new(&b, &grid, nullptr);
    const auto grid_ir = snapshot(b, kernel, "ordinary");
    rocke_ir_builder_free(&b);
    kernel = rocke_gfx950_build_attention_dense_new(&b, &persistent, "gfx950");
    const auto persistent_ir = snapshot(b, kernel, "persistent after ordinary");
    rocke_ir_builder_free(&b);
    if(!grid_ir || !persistent_ir || !scheduled_as(grid_ir.get(), false)
       || !scheduled_as(persistent_ir.get(), true))
    {
        std::fprintf(stderr,
                     "ordinary/persistent IR lacks expected attention math or grid schedule\n");
        return 1;
    }

    // Rebuild ordinary after persistent on the same thread while another
    // thread builds its first (persistent) kernel in independent storage.
    rocke_ir_builder_t peer;
    rocke_kernel_def_t* retained = nullptr;
    std::latch start(2);
    std::thread worker([&] {
        start.arrive_and_wait();
        retained = rocke_gfx950_build_attention_dense_new(&peer, &persistent, nullptr);
    });
    start.arrive_and_wait();
    kernel = rocke_gfx950_build_attention_dense_new(&b, &grid, "gfx950");
    worker.join();
    const auto rebuilt = snapshot(b, kernel, "concurrent ordinary");
    rocke_ir_builder_free(&b);
    // The peer's returned kernel must still be usable after both its thread
    // has exited and the ordinary arena has been destroyed.
    const auto surviving = snapshot(peer, retained, "retained persistent");
    rocke_ir_builder_free(&peer);
    if(!rebuilt || std::strcmp(grid_ir.get(), rebuilt.get()) != 0)
    {
        std::fprintf(stderr, "ordinary IR changed after arena reuse/concurrent build\n");
        return 1;
    }
    if(!surviving || std::strcmp(persistent_ir.get(), surviving.get()) != 0)
    {
        std::fprintf(stderr, "retained persistent IR differs from sequential build\n");
        return 1;
    }
    return 0;
}
