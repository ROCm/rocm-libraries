// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/core/isa_backend_targets.cpp -- host unit test for how both C++ backend
 * registries resolve a target.
 *
 * Each registry has a row for every target whose arch_specs.json row holds facts
 * of its own. A target whose row takes its facts from a generic target
 * (rocke_arch_facts_of) must get its generic's backend under its own name, so
 * its arch-domain cells and messages stay its own; Python's backend_for does the
 * same through listed_arch_for. A target neither registry knows is refused with
 * Python's message, whose "known" list is every known arch.
 *
 * Plain executable: returns non-zero when any check fails.
 */
#include <cstdio>
#include <cstring>
#include <string>

#include "rocke/arch_target.h"
#include "rocke/isa_backend.h"
#include "rocke/lower_llvm_internal.h"

static int g_failures = 0;

#define CHECK(cond, what, gfx)                                                               \
    do                                                                                       \
    {                                                                                        \
        if(!(cond))                                                                          \
        {                                                                                    \
            fprintf(stderr, "FAIL: %s for %s (%s:%d)\n", (what), (gfx), __FILE__, __LINE__); \
            ++g_failures;                                                                    \
        }                                                                                    \
    } while(0)

// The ISA-layer backend of `gfx` equals its generic's in everything but the name.
static void check_isa_backend(const char* gfx, const char* generic)
{
    const char* err = "unset";
    const rocke_isa_backend_t own = rocke_backend_for(gfx, &err);
    CHECK(own.valid && err == NULL, "rocke_backend_for resolves", gfx);
    CHECK(own.gfx != NULL && strcmp(own.gfx, gfx) == 0, "the ISA backend keeps the name", gfx);
    CHECK(rocke_backend_is_known(gfx), "rocke_backend_is_known", gfx);
    if(generic == NULL)
        return;
    const rocke_isa_backend_t shared = rocke_backend_for(generic, NULL);
    CHECK(own.kind == shared.kind && own.vmcnt_bits == shared.vmcnt_bits
              && own.buffer_rsrc_word3 == shared.buffer_rsrc_word3
              && own.waitcnt_layout == shared.waitcnt_layout && own.wave_size == shared.wave_size,
          "the ISA backend is its generic's",
          gfx);
}

// The lowerer's backend of `gfx` equals its generic's in everything but the name.
static void check_lowering_backend(const char* gfx, const char* generic)
{
    rocke_status_t status = ROCKE_ERR_VALUE;
    const ckc::rocke_isa_backend_t* own = ckc::rocke_ll_backend_for(gfx, &status);
    CHECK(own != NULL && status == ROCKE_OK, "rocke_ll_backend_for resolves", gfx);
    if(own == NULL)
        return;
    CHECK(strcmp(own->gfx, gfx) == 0, "the lowering backend keeps the name", gfx);
    CHECK(ckc::rocke_ll_backend_for(gfx, NULL) == own, "the lowering backend is one row", gfx);
    if(generic == NULL)
        return;
    const ckc::rocke_isa_backend_t* shared = ckc::rocke_ll_backend_for(generic, NULL);
    CHECK(shared != NULL && shared != own, "the generic has a row of its own", gfx);
    if(shared == NULL)
        return;
    CHECK(own->datalayout == shared->datalayout && own->triple == shared->triple
              && own->buffer_rsrc_word3 == shared->buffer_rsrc_word3
              && own->encode_waitcnt == shared->encode_waitcnt && own->kind == shared->kind
              && own->has_async_lds_counter == shared->has_async_lds_counter
              && own->blocks_ds_load_tr16 == shared->blocks_ds_load_tr16
              && own->emits_legacy_s_waitcnt == shared->emits_legacy_s_waitcnt
              && own->emit_lds_barrier_drain == shared->emit_lds_barrier_drain
              && own->ds_tr16_b128_spec == shared->ds_tr16_b128_spec
              && own->has_fp8_scale_conversion == shared->has_fp8_scale_conversion,
          "the lowering backend is its generic's",
          gfx);
}

// An unknown target is refused by both registries, with Python's message.
static void check_unknown_target(const char* gfx)
{
    std::string expected = std::string("no ISA backend registered for '") + gfx + "'; known: [";
    int count = 0;
    const char* const* arches = rocke_known_arches(&count);
    for(int i = 0; i < count; ++i)
        expected += std::string(i == 0 ? "'" : ", '") + arches[i] + "'";
    expected += "]";

    const char* err = NULL;
    const rocke_isa_backend_t be = rocke_backend_for(gfx, &err);
    CHECK(!be.valid && !rocke_backend_is_known(gfx), "rocke_backend_for refuses", gfx);
    CHECK(err != NULL && expected == err, "the refusal is Python's message", gfx);
    if(err != NULL && expected != err)
        fprintf(stderr, "  expected: %s\n  got:      %s\n", expected.c_str(), err);

    rocke_status_t status = ROCKE_OK;
    CHECK(ckc::rocke_ll_backend_for(gfx, &status) == NULL && status == ROCKE_ERR_KEY,
          "rocke_ll_backend_for refuses",
          gfx);
}

// The shared list writer reports the whole length, as snprintf does, and cuts
// at the buffer with a NUL, so a caller can tell a cut text from a whole one.
static void check_known_arch_list_repr(void)
{
    const size_t whole = rocke_known_arch_list_repr(NULL, 0, NULL);
    std::string full(whole + 1, '\0');
    CHECK(rocke_known_arch_list_repr(&full[0], full.size(), NULL) == whole
              && strlen(full.c_str()) == whole && full[0] == '[' && full[whole - 1] == ']',
          "the whole list is written",
          "rocke_known_arch_list_repr");
    char cut[8];
    CHECK(rocke_known_arch_list_repr(cut, sizeof(cut), NULL) == whole && whole >= sizeof(cut)
              && strlen(cut) == sizeof(cut) - 1 && full.compare(0, sizeof(cut) - 1, cut) == 0,
          "a short buffer gets a NUL-terminated prefix and the whole length",
          "rocke_known_arch_list_repr");
}

int main(void)
{
    int count = 0;
    const char* const* arches = rocke_known_arches(&count);
    int members = 0;
    for(int i = 0; i < count; ++i)
    {
        const char* generic = rocke_arch_facts_of(arches[i]);
        members += generic != NULL;
        check_isa_backend(arches[i], generic);
        check_lowering_backend(arches[i], generic);
    }
    // Without a member the generic checks above would pass on nothing.
    CHECK(members > 0, "a known arch takes its facts from a generic target", "the registry");
    check_unknown_target("gfx9999");
    check_known_arch_list_repr();
    if(g_failures == 0)
        printf("isa_backend_targets: %d targets, %d through their generic\n", count, members);
    return g_failures == 0 ? 0 : 1;
}
