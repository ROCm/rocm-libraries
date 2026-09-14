// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * lower_llvm_core.c -- BUCKET 0 (the SPINE) of the C99 port of
 * rocke.core.lower_llvm.
 *
 * This file owns the lowerer state plumbing that every other bucket calls:
 *   - the public entry points (rocke_lower_kernel_to_llvm[_ex]),
 *   - the rocke_ll_dispatch table + rocke_ll_set_handler,
 *   - the op / region walkers (rocke_ll_lower_op / rocke_ll_lower_region),
 *   - the _Block / CFG model (current/new_block/block_at/emit/...),
 *   - the smem pre-pass + smem-global lookup,
 *   - finalize (module assembly),
 *   - flavor helpers + the ISA backend resolver,
 *   - and ALL the rocke_ll_* operand / type / constant / fresh-name / need
 *     utility helpers the per-op buckets consume.
 *
 * Faithful translation of rocke.core.lower_llvm (_Lowerer + module helpers).
 * Every spine helper below is fully ported from its Python counterpart; no
 * stub bodies remain in this file.
 */
#include "rocke/lower_llvm_internal.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef _WIN32
#include <dirent.h> /* ll_scan_opt_rocm_version: enumerate /opt/rocm* roots */
#endif

#include "rocke/error.hpp" /* ckc::Error boundary translation */

#include <exception>
#include <new>

/* ====================================================================== */
/* Flavor ladder -- the single source for the C++ side                    */
/* ====================================================================== */

/* One rung: the flavor, its canonical name, and the oldest ROCm that ships
 * it. Mirrors the Python LLVM_FLAVORS tuple and _ROCM_FLAVOR_LADDER, which
 * are a single source enforced by test_no_hand_rolled_flavor_membership_lists.
 * Everything on this side that names, parses, validates, enumerates, or
 * version-maps a flavor reads this table, so adding a rung is one row here
 * plus the enumerator in lower_llvm.h -- previously it was five coordinated
 * edits across three files, which is exactly the drift the Python lint bans.
 *
 * Ordered oldest first to match LLVM_FLAVORS, so error messages listing the
 * set read the same in both engines. */
static const struct
{
    rocke_llvm_flavor_t flavor;
    const char* name;
    int min_rocm_major;
    int min_rocm_minor;
    /* Datalayout generation: false = the LLVM20 plain-p8 shape, true = the
     * LLVM21+ indexed-p8 shape. Python's _DATALAYOUT_KIND_FLAVORS partition,
     * as a column. */
    bool modern;
} ROCKE_LL_FLAVOR_LADDER[] = {
    {ROCKE_LLVM_FLAVOR_LLVM20, "llvm20", 0, 0, false},
    {ROCKE_LLVM_FLAVOR_LLVM22, "llvm22", 7, 2, true},
    {ROCKE_LLVM_FLAVOR_LLVM23, "llvm23", 7, 13, true},
};

static const int ROCKE_LL_FLAVOR_LADDER_COUNT
    = (int)(sizeof(ROCKE_LL_FLAVOR_LADDER) / sizeof(ROCKE_LL_FLAVOR_LADDER[0]));

int rocke_llvm_flavor_count(void)
{
    return ROCKE_LL_FLAVOR_LADDER_COUNT;
}

const char* rocke_llvm_flavor_at(int index)
{
    if(index < 0 || index >= ROCKE_LL_FLAVOR_LADDER_COUNT)
        return "";
    return ROCKE_LL_FLAVOR_LADDER[index].name;
}

const char* rocke_llvm_flavor_name(rocke_llvm_flavor_t flavor)
{
    int i;
    for(i = 0; i < ROCKE_LL_FLAVOR_LADDER_COUNT; ++i)
    {
        if(ROCKE_LL_FLAVOR_LADDER[i].flavor == flavor)
            return ROCKE_LL_FLAVOR_LADDER[i].name;
    }
    return ""; /* AUTO, or a value outside the enum */
}

rocke_llvm_flavor_t rocke_llvm_flavor_from_name(const char* name)
{
    int i;
    if(name)
    {
        for(i = 0; i < ROCKE_LL_FLAVOR_LADDER_COUNT; ++i)
        {
            if(strcmp(name, ROCKE_LL_FLAVOR_LADDER[i].name) == 0)
                return ROCKE_LL_FLAVOR_LADDER[i].flavor;
        }
    }
    return ROCKE_LLVM_FLAVOR_AUTO;
}

bool rocke_llvm_flavor_is_known(rocke_llvm_flavor_t flavor)
{
    int i;
    for(i = 0; i < ROCKE_LL_FLAVOR_LADDER_COUNT; ++i)
    {
        if(ROCKE_LL_FLAVOR_LADDER[i].flavor == flavor)
            return true;
    }
    return false;
}

/* The lowerer's private symbols live in namespace ckc; the public entry points
 * (rocke_llvm_flavor_name/from_name above, rocke_lower_kernel_to_llvm[_ex] below)
 * stay at global scope under their extern "C" header declarations. */
namespace ckc
{

bool rocke_ll_flavor_is_modern(rocke_llvm_flavor_t flavor)
{
    int i;
    for(i = 0; i < ROCKE_LL_FLAVOR_LADDER_COUNT; ++i)
    {
        if(ROCKE_LL_FLAVOR_LADDER[i].flavor == flavor)
            return ROCKE_LL_FLAVOR_LADDER[i].modern;
    }
    return false; /* AUTO / out of range: the legacy shape, as before */
}

/* Python _flavor_for_rocm, walking the shared ladder from the newest rung
 * down. Clamped at both ends and never an error: a ROCm newer than the newest
 * rung resolves to the newest flavor, and anything older to the first rung
 * (LLVM20 -- what pre-7.2 actually shipped). Callers wanting strictness pass
 * an explicit flavor, which IS validated in rocke_lower_kernel_to_llvm. */
static rocke_llvm_flavor_t ll_flavor_for_rocm(int major, int minor)
{
    int i;
    for(i = ROCKE_LL_FLAVOR_LADDER_COUNT - 1; i > 0; --i)
    {
        int rmaj = ROCKE_LL_FLAVOR_LADDER[i].min_rocm_major;
        int rmin = ROCKE_LL_FLAVOR_LADDER[i].min_rocm_minor;
        if(major > rmaj || (major == rmaj && minor >= rmin))
            return ROCKE_LL_FLAVOR_LADDER[i].flavor;
    }
    return ROCKE_LL_FLAVOR_LADDER[0].flavor;
}

/* Python comgr._parse_rocm_version: take the text before the first '-', then
 * the first two dot-separated fields. A single field means minor 0 -- "7"
 * parses as (7, 0), NOT a failure. Getting that right matters: a bare-major
 * version file used to make the C side fall through to the llvm22 default
 * while Python mapped (7, 0) down to llvm20. */
static bool ll_parse_rocm_version(const char* text, int* out_major, int* out_minor)
{
    const char* p = text;
    long major = 0, minor = 0;
    char* end = NULL;

    if(!p)
        return false;
    while(*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r')
        ++p;
    major = strtol(p, &end, 10);
    if(end == p)
        return false;
    if(*end == '.')
    {
        const char* q = end + 1;
        minor = strtol(q, &end, 10);
        if(end == q)
            minor = 0;
    }
    *out_major = (int)major;
    *out_minor = (int)minor;
    return true;
}

/* Python comgr._read_rocm_version_file: "<dir>/.info/version" -> (major,
 * minor), or false when absent/unparseable. */
static bool ll_read_rocm_version_file(const char* dir, int* out_major, int* out_minor)
{
    char path[1024];
    char buf[256];
    FILE* fp = NULL;
    size_t n = 0;

    if(!dir || !*dir)
        return false;
    if((size_t)snprintf(path, sizeof(path), "%s/.info/version", dir) >= sizeof(path))
        return false;
    fp = fopen(path, "r");
    if(!fp)
        return false;
    n = fread(buf, 1, sizeof(buf) - 1, fp);
    fclose(fp);
    buf[n] = '\0';
    return ll_parse_rocm_version(buf, out_major, out_minor);
}

static bool ll_path_exists(const char* path)
{
    FILE* fp = fopen(path, "rb");
    if(!fp)
        return false;
    fclose(fp);
    return true;
}

/* Is `name` a ROCm install-root directory name? Python checks
 * `base == "rocm" or base.startswith(("rocm-", "rocm_"))`. */
static bool ll_is_rocm_root_name(const char* name)
{
    return strcmp(name, "rocm") == 0 || strncmp(name, "rocm-", 5) == 0
           || strncmp(name, "rocm_", 5) == 0;
}

/* Python comgr.resolved_lib_rocm_version's climb: walk up from the directory
 * holding the resolved comgr lib, collecting every ".info/version" passed, and
 * prefer the one in a directory *named* like a ROCm install root; failing that
 * take the outermost (closest to "/") match.
 *
 * The climb is the point, and a fixed dirname(dirname(lib)) is wrong: a
 * packaged ROCm keeps comgr in a versioned "core-<X>/lib" subdir (e.g.
 * /opt/rocm-7.2.0/core-7.13/lib) whose own ".info/version" records the
 * *component* version 7.13.0, not the ROCm release 7.2.0. Picking the
 * component version there would read as ROCm 7.13 and select llvm23 for a
 * ROCm 7.2 / LLVM 22 toolchain. */
static bool ll_climb_rocm_version(const char* start_dir, int* out_major, int* out_minor)
{
    char dir[1024];
    int best_major = 0, best_minor = 0;
    bool found = false;

    if(!start_dir || !*start_dir)
        return false;
    if((size_t)snprintf(dir, sizeof(dir), "%s", start_dir) >= sizeof(dir))
        return false;

    for(;;)
    {
        char* slash = NULL;
        int major = 0, minor = 0;

        if(ll_read_rocm_version_file(dir, &major, &minor))
        {
            const char* base = strrchr(dir, '/');
            base = base ? base + 1 : dir;
            if(ll_is_rocm_root_name(base))
            {
                *out_major = major;
                *out_minor = minor;
                return true; /* named install root wins outright */
            }
            /* Otherwise keep climbing; the outermost match is the root. */
            best_major = major;
            best_minor = minor;
            found = true;
        }

        slash = strrchr(dir, '/');
        if(!slash || slash == dir)
            break;
        *slash = '\0';
    }

    if(found)
    {
        *out_major = best_major;
        *out_minor = best_minor;
    }
    return found;
}

/* Python runtime_coexistence._version_key, as a comparison: compare the runs of
 * digits in each name as an integer sequence so "rocm-7.10" sorts NEWER than
 * "rocm-7.2" (a plain strcmp gets this backwards, because '1' < '2'). Returns
 * >0 when `a` is newer than `b`. A name with no digits sorts oldest.
 *
 * Only the two dirent scans below order names, and both are POSIX-only, so the
 * definition carries their guard: unguarded it is dead code on Windows, which
 * that build rejects (-Werror,-Wunused-function). */
#ifndef _WIN32
static int ll_rocm_name_newer(const char* a, const char* b)
{
    const char* pa = a;
    const char* pb = b;
    for(;;)
    {
        long va = -1, vb = -1;
        while(*pa && (*pa < '0' || *pa > '9'))
            ++pa;
        while(*pb && (*pb < '0' || *pb > '9'))
            ++pb;
        if(*pa)
        {
            char* e = NULL;
            va = strtol(pa, &e, 10);
            pa = e;
        }
        if(*pb)
        {
            char* e = NULL;
            vb = strtol(pb, &e, 10);
            pb = e;
        }
        if(va < 0 && vb < 0)
            return 0; /* both exhausted: equal */
        if(va != vb)
            return (va > vb) ? 1 : -1;
    }
}
#endif /* !_WIN32 */

/* Does <libdir>/libamd_comgr.so[.3] exist? Python's _candidate_lib_paths gives
 * each discovered libdir the bare .so plus the SONAME-suffixed variants; we
 * only need to know whether comgr lives there, not to load it. */
static bool ll_libdir_has_comgr(const char* libdir)
{
    static const char* const SONAMES[] = {"libamd_comgr.so", "libamd_comgr.so.3"};
    size_t i;
    for(i = 0; i < sizeof(SONAMES) / sizeof(SONAMES[0]); ++i)
    {
        char path[1024];
        if((size_t)snprintf(path, sizeof(path), "%s/%s", libdir, SONAMES[i]) >= sizeof(path))
            continue;
        if(ll_path_exists(path))
            return true;
    }
    return false;
}

/* If <root> holds comgr -- directly in lib/, or in a packaged core-<X>/lib --
 * climb from that libdir to the install root's version. `core_tier` selects
 * which of Python's two globs we are serving: true = "<root>/core-<X>/lib",
 * false = "<root>/lib". */
static bool ll_root_comgr_version(const char* root, bool core_tier, int* out_major, int* out_minor)
{
    char libdir[1024];

    if(!core_tier)
    {
        if((size_t)snprintf(libdir, sizeof(libdir), "%s/lib", root) >= sizeof(libdir))
            return false;
        if(!ll_libdir_has_comgr(libdir))
            return false;
        return ll_climb_rocm_version(libdir, out_major, out_minor);
    }
#ifndef _WIN32
    {
        DIR* dir = opendir(root);
        struct dirent* ent = NULL;
        char best[256];
        bool found = false;

        if(!dir)
            return false;
        best[0] = '\0';
        while((ent = readdir(dir)) != NULL)
        {
            if(strncmp(ent->d_name, "core-", 5) != 0)
                continue;
            if((size_t)snprintf(libdir, sizeof(libdir), "%s/%s/lib", root, ent->d_name)
               >= sizeof(libdir))
                continue;
            if(!ll_libdir_has_comgr(libdir))
                continue;
            if(found && ll_rocm_name_newer(ent->d_name, best) <= 0)
                continue;
            snprintf(best, sizeof(best), "%s", ent->d_name);
            found = true;
        }
        closedir(dir);
        if(!found)
            return false;
        if((size_t)snprintf(libdir, sizeof(libdir), "%s/%s/lib", root, best) >= sizeof(libdir))
            return false;
        return ll_climb_rocm_version(libdir, out_major, out_minor);
    }
#else
    (void)root;
    return false;
#endif
}

/* Enumerate /opt/rocm* install roots, newest first (Python's
 * glob("/opt/rocm*") sorted by _version_key, reversed). Returns the count
 * written into `out`. */
static int ll_list_opt_rocm_roots(char out[][256], int max_roots)
{
    int n = 0;
#ifndef _WIN32
    DIR* dir = opendir("/opt");
    struct dirent* ent = NULL;

    if(!dir)
        return 0;
    while((ent = readdir(dir)) != NULL && n < max_roots)
    {
        int i, j;
        if(strncmp(ent->d_name, "rocm", 4) != 0)
            continue;
        /* Insertion sort, newest first. */
        for(i = 0; i < n; ++i)
        {
            if(ll_rocm_name_newer(ent->d_name, out[i]) > 0)
                break;
        }
        for(j = n; j > i; --j)
            snprintf(out[j], 256, "%s", out[j - 1]);
        snprintf(out[i], 256, "%s", ent->d_name);
        ++n;
    }
    closedir(dir);
#else
    (void)out;
    (void)max_roots;
#endif
    return n;
}

/* Python _detect_llvm_flavor, minus the one step that needs a live Python
 * interpreter.
 *
 * Python's order is: $ROCKE_LLVM_FLAVOR -> the ROCm vintage of the comgr
 * library that will actually compile the IR -> torch.version.hip -> an
 * installed ROCm's .info/version -> llvm22. The torch step cannot be
 * reproduced here, so callers needing exact parity on a torch-rocm box go
 * through python/rocke/core/backend.py, which resolves the flavor in Python
 * and passes it explicitly instead of handing the engine AUTO.
 *
 * Everything else is mirrored, and mirroring the comgr step rather than just
 * reading /opt/rocm is the point. The flavor MUST match the comgr that
 * compiles the IR, and Python finds that comgr via $ROCKE_COMGR_LIB, then
 * $ROCM_PATH/$ROCM_HOME, then globbed /opt/rocm* trees newest-first (each
 * possibly with a packaged core-<X>/lib subdir). Reading only the hardcoded
 * /opt/rocm/.info/version, as this used to, picks a different ROCm -- and so a
 * different flavor -- than Python on any host where /opt/rocm is absent, is a
 * stale symlink, or is not the install $ROCM_PATH points at. That is a silent
 * flavor split between the two engines with no test able to see it, because
 * the engines only disagree on hosts the gate does not run on. */
static rocke_llvm_flavor_t ll_resolve_flavor(void)
{
    static const char* const ROOT_ENVS[] = {"ROCM_PATH", "ROCM_HOME"};
    enum
    {
        LL_MAX_OPT_ROOTS = 16
    };
    char roots[LL_MAX_OPT_ROOTS][256];
    int major = 0, minor = 0;
    int nroots, r, tier;
    size_t i;

    const char* env = getenv("ROCKE_LLVM_FLAVOR");
    if(env)
    {
        rocke_llvm_flavor_t f = rocke_llvm_flavor_from_name(env);
        if(f != ROCKE_LLVM_FLAVOR_AUTO)
        {
            return f;
        }
    }

    /* Tier 1: $ROCKE_COMGR_LIB names the comgr to load outright. */
    {
        const char* override_lib = getenv("ROCKE_COMGR_LIB");
        if(override_lib && *override_lib && ll_path_exists(override_lib))
        {
            char dir[1024];
            char* slash = NULL;
            snprintf(dir, sizeof(dir), "%s", override_lib);
            slash = strrchr(dir, '/');
            if(slash)
            {
                *slash = '\0';
                if(ll_climb_rocm_version(dir, &major, &minor))
                    return ll_flavor_for_rocm(major, minor);
            }
        }
    }

    /* Tier 2: an operator-set root. Python's _rocm_root_libdirs gives an env
     * root ONLY "<root>/lib" -- the "core-<X>/lib" glob is applied to
     * /opt/rocm* alone -- so probing the core subdir here too would make the C
     * side accept a root Python skips. */
    for(i = 0; i < sizeof(ROOT_ENVS) / sizeof(ROOT_ENVS[0]); ++i)
    {
        const char* root = getenv(ROOT_ENVS[i]);
        if(root && *root && ll_root_comgr_version(root, /*core_tier=*/false, &major, &minor))
            return ll_flavor_for_rocm(major, minor);
    }

    /* Tier 3: discovered installs. Python runs its two globs as separate
     * passes, so every discovered root's "core-<X>/lib" is probed before ANY
     * root's plain "lib"; a packaged install keeps the real runtime in the
     * versioned subdir. Roots are ordered newest-first, which combined with
     * the newest-first core subdir gives the same order as Python sorting the
     * full glob by _version_key. */
    nroots = ll_list_opt_rocm_roots(roots, LL_MAX_OPT_ROOTS);
    for(tier = 0; tier < 2; ++tier)
    {
        bool core_tier = (tier == 0);
        for(r = 0; r < nroots; ++r)
        {
            char path[512];
            if((size_t)snprintf(path, sizeof(path), "/opt/%s", roots[r]) >= sizeof(path))
                continue;
            if(ll_root_comgr_version(path, core_tier, &major, &minor))
                return ll_flavor_for_rocm(major, minor);
        }
    }

    /* Tier 4: Python's _system_rocm_version -- no comgr found anywhere, but an
     * install still records a version. This deliberately reads /opt/rocm ONLY,
     * not $ROCM_PATH: Python's fallback is that exact hardcoded path, and
     * honouring the env root here would resolve a different flavor than Python
     * whenever $ROCM_PATH names a tree with no comgr in it. */
    if(ll_read_rocm_version_file("/opt/rocm", &major, &minor))
        return ll_flavor_for_rocm(major, minor);

    return ROCKE_LLVM_FLAVOR_LLVM22;
}

/* ====================================================================== */
/* ISA backend (lower_llvm_internal.h's own backend struct)               */
/* ====================================================================== */

/* Every backend the lowerer resolves. The waitcnt encoders, barrier drains and
 * tr16 opcode selections are defined in the control / crosslane buckets next to
 * the handlers that call them. Static storage: returned by pointer.
 *
 * Written with designated initializers so a row reads as the set of Python
 * ISABackend overrides it stands for -- the defaults (legacy s_waitcnt, no
 * async counter, type-agnostic tr16) are what the Python base class supplies. */
#define LL_BACKEND_CDNA_DEFAULTS                                                              \
    .datalayout = NULL, .triple = NULL, .buffer_rsrc_word3 = ROCKE_LL_BUFFER_RSRC_WORD3_CDNA, \
    .encode_waitcnt = rocke_ll_encode_waitcnt_gfx9_10, .kind = ROCKE_LL_ISA_CDNA,             \
    .has_async_lds_counter = false, .emits_legacy_s_waitcnt = true,                           \
    .emit_lds_barrier_drain = rocke_ll_emit_lds_barrier_drain_legacy,                         \
    .ds_tr16_b128_spec = rocke_ll_tr16_spec_b128_default

/* RDNA backends (Python Gfx11RdnaBackend / Gfx12RdnaBackend): same
 * datalayout/triple as CDNA on the ROCm releases we target, but the RDNA buffer
 * SRD word3 and the contiguous gfx11 s_waitcnt layout. gfx12 differs from gfx11
 * only in WMMA fragment width, which the op_id ("wmma_gfx12_*") encodes. */
#define LL_BACKEND_RDNA_DEFAULTS                                                              \
    .datalayout = NULL, .triple = NULL, .buffer_rsrc_word3 = ROCKE_LL_BUFFER_RSRC_WORD3_RDNA, \
    .encode_waitcnt = rocke_ll_encode_waitcnt_gfx11, .kind = ROCKE_LL_ISA_RDNA,               \
    .has_async_lds_counter = false, .emits_legacy_s_waitcnt = true,                           \
    .emit_lds_barrier_drain = rocke_ll_emit_lds_barrier_drain_legacy,                         \
    .ds_tr16_b128_spec = rocke_ll_tr16_spec_b128_default

static const rocke_isa_backend_t LL_BACKEND_GFX950 = {.gfx = "gfx950", LL_BACKEND_CDNA_DEFAULTS};
static const rocke_isa_backend_t LL_BACKEND_GFX942 = {.gfx = "gfx942", LL_BACKEND_CDNA_DEFAULTS};
static const rocke_isa_backend_t LL_BACKEND_GFX908 = {.gfx = "gfx908", LL_BACKEND_CDNA_DEFAULTS};
static const rocke_isa_backend_t LL_BACKEND_GFX90A = {.gfx = "gfx90a", LL_BACKEND_CDNA_DEFAULTS};
static const rocke_isa_backend_t LL_BACKEND_GFX1151 = {.gfx = "gfx1151", LL_BACKEND_RDNA_DEFAULTS};
static const rocke_isa_backend_t LL_BACKEND_GFX1201 = {.gfx = "gfx1201", LL_BACKEND_RDNA_DEFAULTS};
static const rocke_isa_backend_t LL_BACKEND_GFX11_GENERIC
    = {.gfx = "gfx11-generic", LL_BACKEND_RDNA_DEFAULTS};

/* gfx1250 (Python Gfx1250Backend, which derives from Gfx12RdnaBackend). It is a
 * CDNA part programmed on the GFX12 model: wave32, WMMA-only, with the K=32
 * f16/bf16 and K=64 fp8/bf8 atoms. Datalayout and triple are byte-identical to
 * gfx950/gfx1201 on the ROCm releases we target.
 *
 * The buffer SRD word3 and the gfx11 s_waitcnt *layout* are inherited
 * placeholders, adequate for the flat-global WMMA GEMM path; the gfx1250 57-bit
 * SRD is deferred, and the layout is unreachable anyway because
 * emits_legacy_s_waitcnt is false. What does diverge: split wait counters
 * (llvm.amdgcn.s.waitcnt is not selectable, and the pre-barrier drain is
 * s_wait_loadcnt / s_wait_dscnt), a dedicated async-DMA counter, and the
 * element-typed ds_load_tr16_b128 opcodes. */
static const rocke_isa_backend_t LL_BACKEND_GFX1250
    = {.gfx = "gfx1250",
       .datalayout = NULL,
       .triple = NULL,
       .buffer_rsrc_word3 = ROCKE_LL_BUFFER_RSRC_WORD3_RDNA,
       .encode_waitcnt = rocke_ll_encode_waitcnt_gfx11,
       .kind = ROCKE_LL_ISA_RDNA,
       .has_async_lds_counter = true,
       .blocks_ds_load_tr16 = true,
       .emits_legacy_s_waitcnt = false,
       .emit_lds_barrier_drain = rocke_ll_emit_lds_barrier_drain_split,
       .ds_tr16_b128_spec = rocke_ll_tr16_spec_b128_gfx1250};

#undef LL_BACKEND_CDNA_DEFAULTS
#undef LL_BACKEND_RDNA_DEFAULTS

/* Mutable copies so datalayout/triple (extern consts resolved at runtime) can
 * be patched in. backend_for fills them from ROCKE_LL_DATALAYOUT/TRIPLE. */
static rocke_isa_backend_t LL_BACKEND_RESOLVED;

const rocke_isa_backend_t* rocke_ll_backend_for(const char* arch, rocke_status_t* st)
{
    const rocke_isa_backend_t* base = NULL;
    if(arch == NULL || strcmp(arch, "gfx950") == 0)
    {
        base = &LL_BACKEND_GFX950;
    }
    else if(strcmp(arch, "gfx942") == 0)
    {
        base = &LL_BACKEND_GFX942;
    }
    else if(strcmp(arch, "gfx908") == 0)
    {
        base = &LL_BACKEND_GFX908;
    }
    else if(strcmp(arch, "gfx90a") == 0)
    {
        base = &LL_BACKEND_GFX90A;
    }
    else if(strcmp(arch, "gfx1151") == 0)
    {
        base = &LL_BACKEND_GFX1151;
    }
    else if(strcmp(arch, "gfx1201") == 0)
    {
        base = &LL_BACKEND_GFX1201;
    }
    else if(strcmp(arch, "gfx11-generic") == 0)
    {
        base = &LL_BACKEND_GFX11_GENERIC;
    }
    else if(strcmp(arch, "gfx1250") == 0)
    {
        base = &LL_BACKEND_GFX1250;
    }
    else
    {
        if(st)
        {
            *st = ROCKE_ERR_KEY;
        }
        return NULL;
    }
    LL_BACKEND_RESOLVED = *base;
    LL_BACKEND_RESOLVED.datalayout = ROCKE_LL_DATALAYOUT;
    LL_BACKEND_RESOLVED.triple = ROCKE_LL_TRIPLE;
    if(st)
    {
        *st = ROCKE_OK;
    }
    return &LL_BACKEND_RESOLVED;
}

/* ====================================================================== */
/* Error model                                                            */
/* ====================================================================== */

[[noreturn]] void rocke_ll_fail(rocke_lower_t* L, rocke_status_t st, const char* fmt, ...)
{
    /* Format the reason once (bounded exactly like the legacy sink), then raise.
     * This [[noreturn]]s via ckc::raise_status. The thrown exception is caught at
     * the lowerer boundary and translated back into the status code + caller
     * `err` buffer, so the extern "C" ABI is unchanged. */
    (void)L; /* the lowerer no longer carries a sticky error; we raise instead */
    char buf[ROCKE_ERR_MSG_CAP];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    buf[sizeof buf - 1] = '\0';
    ckc::raise_status(st, buf);
}

bool rocke_ll_live(const rocke_lower_t* L)
{
    return L != NULL;
}

/* ====================================================================== */
/* Block / CFG model (Python _Block)                                      */
/* ====================================================================== */

rocke_ll_block_t* rocke_ll_current(rocke_lower_t* L)
{
    if(!L || L->blocks.len == 0)
    {
        return NULL;
    }
    return L->blocks.data[L->blocks.len - 1];
}

static rocke_ll_block_t* ll_make_block(rocke_lower_t* L, const char* label)
{
    rocke_ll_block_t* blk = (rocke_ll_block_t*)rocke_arena_calloc(&L->arena, sizeof(*blk));
    if(!blk)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "block alloc");
    }
    blk->label = rocke_arena_strdup(&L->arena, label ? label : "");
    rocke_vec_init(&blk->lines);
    blk->terminated = false;
    int rc;
    rocke_vec_push(&L->arena, &L->blocks, blk, rc);
    if(rc != 0)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "blocks push");
    }
    return blk;
}

rocke_ll_block_t* rocke_ll_new_block(rocke_lower_t* L, const char* base)
{
    if(!L)
    {
        return NULL;
    }
    L->block_counter += 1;
    char* label = rocke_arena_printf(&L->arena, "%s.%d", base ? base : "", L->block_counter);
    if(!label)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "new_block label");
    }
    return ll_make_block(L, label);
}

rocke_ll_block_t* rocke_ll_block_at(rocke_lower_t* L, int idx)
{
    if(!L || idx < 0 || (size_t)idx >= L->blocks.len)
    {
        return NULL;
    }
    return L->blocks.data[idx];
}

int rocke_ll_block_count(const rocke_lower_t* L)
{
    return L ? (int)L->blocks.len : 0;
}

void rocke_ll_block_emit(rocke_lower_t* L, rocke_ll_block_t* blk, const char* line)
{
    if(!L || !blk)
    {
        return;
    }
    if(blk->terminated)
    {
        rocke_ll_fail(L, ROCKE_ERR_VALUE, "block %s already terminated", blk->label);
    }
    char* copy = rocke_arena_strdup(&L->arena, line ? line : "");
    if(!copy)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "emit strdup");
    }
    int rc;
    rocke_vec_push(&L->arena, &blk->lines, copy, rc);
    if(rc != 0)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "emit push");
    }
}

void rocke_ll_block_emitf(rocke_lower_t* L, rocke_ll_block_t* blk, const char* fmt, ...)
{
    if(!L || !blk)
    {
        return;
    }
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if(n >= 0 && (size_t)n < sizeof buf)
    {
        rocke_ll_block_emit(L, blk, buf);
        return;
    }
    /* Long line: format into the arena. */
    va_start(ap, fmt);
    char big[8192];
    vsnprintf(big, sizeof big, fmt, ap);
    va_end(ap);
    rocke_ll_block_emit(L, blk, big);
}

void rocke_ll_emit(rocke_lower_t* L, const char* line)
{
    rocke_ll_block_emit(L, rocke_ll_current(L), line);
}

void rocke_ll_emitf(rocke_lower_t* L, const char* fmt, ...)
{
    rocke_ll_block_t* blk = rocke_ll_current(L);
    if(!L || !blk)
    {
        return;
    }
    char buf[1024];
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    if(n >= 0 && (size_t)n < sizeof buf)
    {
        rocke_ll_block_emit(L, blk, buf);
        return;
    }
    va_start(ap, fmt);
    char big[8192];
    vsnprintf(big, sizeof big, fmt, ap);
    va_end(ap);
    rocke_ll_block_emit(L, blk, big);
}

/* ====================================================================== */
/* Fresh names (Python _fresh)                                            */
/* ====================================================================== */

const char* rocke_ll_fresh(rocke_lower_t* L, const char* hint)
{
    if(!L)
    {
        return "";
    }
    L->tmp_counter += 1;
    char* s = rocke_arena_printf(&L->arena, "%%%s.%d", hint ? hint : "t", L->tmp_counter);
    if(!s)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "fresh");
    }
    return s;
}

/* ====================================================================== */
/* Intrinsic need-tracking (Python _need / self._decls)                   */
/* ====================================================================== */

/* Resolve a decl key to its declaration text: dyn_decls override, then the
 * flavor override table (LLVM22), then the base table. NULL if unknown. */
static const char* ll_resolve_decl(rocke_lower_t* L, const char* key)
{
    if(!L || !key)
    {
        return NULL;
    }
    for(size_t i = 0; i < L->dyn_decls.len; i++)
    {
        if(strcmp(L->dyn_decls.data[i].key, key) == 0)
        {
            return L->dyn_decls.data[i].decl;
        }
    }
    {
        int novr = 0;
        const rocke_ll_decl_t* ovr = rocke_ll_flavor_overrides(L->flavor, &novr);
        for(int i = 0; i < novr; i++)
        {
            if(strcmp(ovr[i].key, key) == 0)
            {
                return ovr[i].decl;
            }
        }
    }
    for(int i = 0; i < ROCKE_LL_INTRINSIC_DECLS_COUNT; i++)
    {
        if(strcmp(ROCKE_LL_INTRINSIC_DECLS[i].key, key) == 0)
        {
            return ROCKE_LL_INTRINSIC_DECLS[i].decl;
        }
    }
    return NULL;
}

static bool ll_need_has(const rocke_lower_t* L, const char* key)
{
    for(size_t i = 0; i < L->needs.len; i++)
    {
        if(strcmp(L->needs.data[i].key, key) == 0)
        {
            return true;
        }
    }
    return false;
}

void rocke_ll_need(rocke_lower_t* L, const char* key)
{
    if(!L || !key)
    {
        return;
    }
    if(ll_need_has(L, key))
    {
        return;
    }
    rocke_ll_need_t rec;
    rec.key = rocke_arena_strdup(&L->arena, key);
    rec.decl = ll_resolve_decl(L, key); /* may be NULL; finalize tolerates */
    int rc;
    rocke_vec_push(&L->arena, &L->needs, rec, rc);
    if(rc != 0)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "need push");
    }
}

void rocke_ll_need_dynamic(rocke_lower_t* L, const char* key, const char* decl)
{
    if(!L || !key)
    {
        return;
    }
    /* Register/replace the dynamic decl text (Python self._decls[key] = decl). */
    for(size_t i = 0; i < L->dyn_decls.len; i++)
    {
        if(strcmp(L->dyn_decls.data[i].key, key) == 0)
        {
            L->dyn_decls.data[i].decl = rocke_arena_strdup(&L->arena, decl ? decl : "");
            rocke_ll_need(L, key);
            return;
        }
    }
    rocke_ll_decl_t d;
    d.key = rocke_arena_strdup(&L->arena, key);
    d.decl = rocke_arena_strdup(&L->arena, decl ? decl : "");
    int rc;
    rocke_vec_push(&L->arena, &L->dyn_decls, d, rc);
    if(rc != 0)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "dyn_decl push");
    }
    rocke_ll_need(L, key);
}

/* ====================================================================== */
/* Type rendering (Python _llvm_type / _llvm_type_from_name)              */
/* ====================================================================== */

const char* rocke_ll_llvm_type(rocke_lower_t* L, const rocke_type_t* t)
{
    if(!t)
    {
        if(L)
        {
            rocke_ll_fail(L, ROCKE_ERR_NOTIMPL, "no LLVM mapping for (null) type");
        }
        return "";
    }
    if(t->kind == ROCKE_TYPE_PTR)
    {
        const char* sp = t->space;
        if(sp && strcmp(sp, "global") == 0)
            return "ptr addrspace(1)";
        if(sp && strcmp(sp, "lds") == 0)
            return "ptr addrspace(3)";
        if(sp && strcmp(sp, "constant") == 0)
            return "ptr addrspace(4)";
        return "ptr";
    }
    if(t->kind == ROCKE_TYPE_VECTOR)
    {
        const char* elem = rocke_ll_llvm_type(L, t->elem);
        return rocke_arena_printf(&L->arena, "<%d x %s>", t->count, elem);
    }
    if(t->kind == ROCKE_TYPE_SMEM)
    {
        return "ptr addrspace(3)";
    }
    /* scalar */
    const char* n = t->name;
    if(n)
    {
        if(strcmp(n, "i1") == 0)
            return "i1";
        if(strcmp(n, "i8") == 0)
            return "i8";
        if(strcmp(n, "i16") == 0)
            return "i16";
        if(strcmp(n, "i32") == 0)
            return "i32";
        if(strcmp(n, "i64") == 0)
            return "i64";
        if(strcmp(n, "f16") == 0)
            return "half";
        if(strcmp(n, "bf16") == 0)
            return "bfloat";
        if(strcmp(n, "fp8e4m3") == 0)
            return "i8";
        if(strcmp(n, "bf8e5m2") == 0)
            return "i8";
        if(strcmp(n, "f32") == 0)
            return "float";
    }
    rocke_ll_fail(L, ROCKE_ERR_NOTIMPL, "no LLVM mapping for type %s", n ? n : "(null)");
}

const char* rocke_ll_param_llvm_type(rocke_lower_t* L, const rocke_param_t* p)
{
    const char* ovr;
    if(!p)
        return "";
    if(p->type && p->type->kind == ROCKE_TYPE_PTR)
    {
        /* addr_space override (P17): a pointer param can be pinned to a
         * different space than its IR type says. The function header and any
         * call site passing that param must name the same type. */
        ovr = rocke_attr_get_str(&p->attrs, "addr_space");
        if(ovr && strcmp(ovr, "constant") == 0)
            return "ptr addrspace(4)";
        if(ovr && strcmp(ovr, "global") == 0)
            return "ptr addrspace(1)";
    }
    return rocke_ll_llvm_type(L, p->type);
}

const char* rocke_ll_value_ptr_type(rocke_lower_t* L, const rocke_value_t* v)
{
    int i;
    if(!v)
        return "";
    /* Python _Lowerer._ptr_llvm_type: kernel params carry the header's type. */
    if(L && L->kernel && v->name)
    {
        for(i = 0; i < L->kernel->num_params; i++)
        {
            const rocke_param_t* p = L->kernel->params[i];
            if(p && p->name && v->name[0] == '%' && strcmp(v->name + 1, p->name) == 0)
                return rocke_ll_param_llvm_type(L, p);
        }
    }
    return rocke_ll_llvm_type(L, v->type);
}

int rocke_ll_anyptr_space(rocke_lower_t* L,
                          const char* op,
                          const rocke_value_t* ptr,
                          const rocke_ll_anyptr_space_t* allowed,
                          int count,
                          const char** out_ptr_ty)
{
    const char* ty = rocke_ll_value_ptr_type(L, ptr);
    char list[128];
    size_t used = 0;
    int i;
    for(i = 0; i < count; i++)
    {
        if(strcmp(ty, allowed[i].ptr_ty) == 0)
        {
            if(out_ptr_ty)
                *out_ptr_ty = allowed[i].ptr_ty;
            return allowed[i].space;
        }
    }
    list[0] = '\0';
    for(i = 0; i < count && used + 1 < sizeof(list); i++)
    {
        int n
            = snprintf(list + used, sizeof(list) - used, "%s%s", i ? ", " : "", allowed[i].ptr_ty);
        if(n < 0)
            break;
        used += (size_t)n;
    }
    rocke_ll_fail(L,
                  ROCKE_ERR_VALUE,
                  "%s: pointer operand is %s, but the intrinsic accepts only %s",
                  op,
                  ty,
                  list);
}

const char* rocke_ll_llvm_type_from_name(rocke_lower_t* L, const char* name)
{
    if(!name)
    {
        rocke_ll_fail(L, ROCKE_ERR_NOTIMPL, "no LLVM type for (null)");
    }
    if(strcmp(name, "i32") == 0)
        return "i32";
    if(strcmp(name, "i64") == 0)
        return "i64";
    if(strcmp(name, "i8") == 0)
        return "i8";
    if(strcmp(name, "f16") == 0)
        return "half";
    if(strcmp(name, "bf16") == 0)
        return "bfloat";
    if(strcmp(name, "f32") == 0)
        return "float";
    if(strcmp(name, "fp8e4m3") == 0)
        return "i8";
    if(strncmp(name, "vec<", 4) == 0)
    {
        /* vec<elemxN> -> "<N x llvm_elem>" */
        const char* inner = name + 4;
        const char* xpos = strchr(inner, 'x');
        const char* end = strrchr(name, '>');
        if(xpos && end && end > xpos)
        {
            char elem[32];
            size_t elen = (size_t)(xpos - inner);
            if(elen >= sizeof elem)
            {
                elen = sizeof elem - 1;
            }
            memcpy(elem, inner, elen);
            elem[elen] = '\0';
            int count = atoi(xpos + 1);
            const char* lelem = "i32";
            if(strcmp(elem, "f32") == 0)
                lelem = "float";
            else if(strcmp(elem, "f16") == 0)
                lelem = "half";
            else if(strcmp(elem, "bf16") == 0)
                lelem = "bfloat";
            else if(strcmp(elem, "i32") == 0)
                lelem = "i32";
            else
            {
                rocke_ll_fail(L, ROCKE_ERR_NOTIMPL, "no LLVM type for vec elem %s", elem);
            }
            return rocke_arena_printf(&L->arena, "<%d x %s>", count, lelem);
        }
    }
    rocke_ll_fail(L, ROCKE_ERR_NOTIMPL, "no LLVM type for %s", name);
}

const char* rocke_ll_smem_storage_type(rocke_lower_t* L, const rocke_type_t* smem)
{
    if(!L || !smem)
    {
        return "";
    }
    const char* out = rocke_ll_llvm_type(L, smem->elem);
    /* Wrap from innermost (last dim) to outermost (first dim). */
    for(int d = smem->rank - 1; d >= 0; d--)
    {
        out = rocke_arena_printf(&L->arena, "[%d x %s]", smem->shape[d], out);
        if(!out)
        {
            rocke_ll_fail(L, ROCKE_ERR_OOM, "smem_storage_type");
        }
    }
    return out;
}

/* ====================================================================== */
/* FP hex constants (Python _fp32_hex / _fp16_hex)                        */
/* ====================================================================== */

const char* rocke_ll_fp32_hex(rocke_lower_t* L, double x)
{
    /* LLVM spells a float hex constant as the 64-bit hex of the double value
     * of the rounded fp32 constant. */
    float f = (float)x;
    double rounded = (double)f;
    uint64_t bits;
    memcpy(&bits, &rounded, sizeof bits);
    return rocke_arena_printf(&L->arena, "0x%016llX", (unsigned long long)bits);
}

const char* rocke_ll_fp16_hex(rocke_lower_t* L, double x)
{
    /* LLVM IR: half 0xH<4 hex>. Convert double -> IEEE-754 binary16 (round to
     * nearest even) without relying on _Float16 support. */
    float f = (float)x;
    uint32_t fb;
    memcpy(&fb, &f, sizeof fb);
    uint32_t sign = (fb >> 16) & 0x8000u;
    int32_t exp = (int32_t)((fb >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = fb & 0x7FFFFFu;
    uint16_t h;
    if(((fb >> 23) & 0xFF) == 0xFF)
    {
        /* inf / nan */
        h = (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
    }
    else if(exp >= 0x1F)
    {
        h = (uint16_t)(sign | 0x7C00u); /* overflow -> inf */
    }
    else if(exp <= 0)
    {
        if(exp < -10)
        {
            h = (uint16_t)sign; /* underflow -> 0 */
        }
        else
        {
            mant |= 0x800000u;
            uint32_t shift = (uint32_t)(14 - exp);
            uint32_t halfmant = mant >> shift;
            /* round to nearest even */
            uint32_t rem = mant & ((1u << shift) - 1u);
            uint32_t halfway = 1u << (shift - 1);
            if(rem > halfway || (rem == halfway && (halfmant & 1u)))
            {
                halfmant += 1;
            }
            h = (uint16_t)(sign | halfmant);
        }
    }
    else
    {
        uint16_t hm = (uint16_t)(mant >> 13);
        uint32_t rem = mant & 0x1FFFu;
        h = (uint16_t)(sign | ((uint16_t)exp << 10) | hm);
        if(rem > 0x1000u || (rem == 0x1000u && (hm & 1u)))
        {
            h += 1; /* carries into exponent naturally */
        }
    }
    return rocke_arena_printf(&L->arena, "0xH%04X", (unsigned)h);
}

/* ====================================================================== */
/* asm string escaping (Python _escape_llvm_asm_string)                   */
/* ====================================================================== */

const char* rocke_ll_escape_asm_string(rocke_lower_t* L, const char* s)
{
    if(!L)
    {
        return "";
    }
    if(!s)
    {
        return rocke_arena_strdup(&L->arena, "");
    }
    /* worst case 4x ("\XX"+1) -- be generous. */
    size_t n = strlen(s);
    size_t cap = n * 4 + 1;
    char* out = (char*)rocke_arena_alloc(&L->arena, cap);
    if(!out)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "escape_asm");
    }
    size_t w = 0;
    for(size_t i = 0; i < n; i++)
    {
        unsigned char c = (unsigned char)s[i];
        if(c == '\\')
        {
            w += (size_t)snprintf(out + w, cap - w, "\\5C");
        }
        else if(c == '"')
        {
            w += (size_t)snprintf(out + w, cap - w, "\\22");
        }
        else if(c >= 0x20 && c <= 0x7E)
        {
            out[w++] = (char)c;
        }
        else
        {
            w += (size_t)snprintf(out + w, cap - w, "\\%02X", c);
        }
    }
    out[w] = '\0';
    return out;
}

/* ====================================================================== */
/* Constant helpers (Python _is_constant / _eval_constant / _operand)     */
/* ====================================================================== */

bool rocke_ll_is_constant(const rocke_value_t* v)
{
    return v && v->op && v->op->opcode == ROCKE_OP_ARITH_CONSTANT;
}

int64_t rocke_ll_eval_constant(rocke_lower_t* L, const rocke_value_t* v)
{
    if(!rocke_ll_is_constant(v))
    {
        rocke_ll_fail(L,
                      ROCKE_ERR_VALUE,
                      "Value %s is not a compile-time constant",
                      (v && v->name) ? v->name : "(null)");
    }
    int64_t iv = 0;
    if(rocke_attr_get_int(&v->op->attrs, "value", &iv))
    {
        return iv;
    }
    double fv = 0.0;
    if(rocke_attr_get_float(&v->op->attrs, "value", &fv))
    {
        return (int64_t)fv;
    }
    return 0;
}

const char* rocke_ll_operand(rocke_lower_t* L, const rocke_value_t* v)
{
    if(!v)
    {
        return "";
    }
    const rocke_op_t* op = v->op;
    if(op == NULL)
    {
        return v->name;
    }
    if(op->opcode == ROCKE_OP_ARITH_CONSTANT)
    {
        const char* ity = rocke_attr_get_str(&op->attrs, "ity");
        if(ity == NULL)
        {
            ity = "i32";
        }
        if(strcmp(ity, "f32") == 0)
        {
            double fv = 0.0;
            rocke_attr_get_float(&op->attrs, "value", &fv);
            return rocke_ll_fp32_hex(L, fv);
        }
        if(strcmp(ity, "f16") == 0)
        {
            double fv = 0.0;
            rocke_attr_get_float(&op->attrs, "value", &fv);
            return rocke_ll_fp16_hex(L, fv);
        }
        int64_t iv = 0;
        if(!rocke_attr_get_int(&op->attrs, "value", &iv))
        {
            double fv = 0.0;
            if(rocke_attr_get_float(&op->attrs, "value", &fv))
            {
                iv = (int64_t)fv;
            }
        }
        return rocke_arena_printf(&L->arena, "%lld", (long long)iv);
    }
    return v->name;
}

const char* rocke_ll_operand_with_type(rocke_lower_t* L, const rocke_value_t* v)
{
    if(!v)
    {
        return "";
    }
    const char* ty = rocke_ll_llvm_type(L, v->type);
    const char* op = rocke_ll_operand(L, v);
    return rocke_arena_printf(&L->arena, "%s %s", ty, op);
}

/* ====================================================================== */
/* Shared binary-op helpers (Python _binop / _vector_binop)               */
/* ====================================================================== */

void rocke_ll_binop(rocke_lower_t* L, const rocke_op_t* op, const char* llvm_op)
{
    if(!rocke_ll_live(L) || !op || op->num_results < 1 || op->num_operands < 2)
    {
        return;
    }
    const rocke_value_t* res = op->results[0];
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    rocke_ll_emitf(L,
                   "  %s = %s %s %s, %s",
                   res->name,
                   llvm_op,
                   rocke_ll_llvm_type(L, res->type),
                   rocke_ll_operand(L, a),
                   rocke_ll_operand(L, b));
}

void rocke_ll_vector_binop(rocke_lower_t* L, const rocke_op_t* op, const char* llvm_op)
{
    if(!rocke_ll_live(L) || !op || op->num_results < 1 || op->num_operands < 2)
    {
        return;
    }
    const rocke_value_t* res = op->results[0];
    const rocke_value_t* a = op->operands[0];
    const rocke_value_t* b = op->operands[1];
    rocke_ll_emitf(L,
                   "  %s = %s %s %s, %s",
                   res->name,
                   llvm_op,
                   rocke_ll_llvm_type(L, a->type),
                   rocke_ll_operand(L, a),
                   rocke_ll_operand(L, b));
}

/* ====================================================================== */
/* smem pre-pass (Python _collect_smem / _smem_global_name)               */
/* ====================================================================== */

void rocke_ll_collect_smem(rocke_lower_t* L, const rocke_region_t* region)
{
    if(!L || !region)
    {
        return;
    }
    for(int i = 0; i < region->num_ops; i++)
    {
        const rocke_op_t* op = region->ops[i];
        if(!op)
        {
            continue;
        }
        if(op->opcode == ROCKE_OP_TILE_SMEM_ALLOC && op->num_results > 0)
        {
            const rocke_value_t* res = op->results[0];
            const char* short_name = res->name;
            if(short_name && short_name[0] == '%')
            {
                short_name += 1;
            }
            const char* kname = L->kernel ? L->kernel->name : "";
            char* gname = rocke_arena_printf(
                &L->arena, "@%s.%s", short_name ? short_name : "", kname ? kname : "");
            rocke_ll_smem_global_t g;
            g.gname = gname;
            g.stype = res->type;
            int rc;
            rocke_vec_push(&L->arena, &L->smem_globals, g, rc);
            if(rc != 0)
            {
                rocke_ll_fail(L, ROCKE_ERR_OOM, "smem_globals push");
            }
            rocke_ll_smem_name_t nm;
            nm.value_name = res->name;
            nm.gname = gname;
            rocke_vec_push(&L->arena, &L->smem_names, nm, rc);
            if(rc != 0)
            {
                rocke_ll_fail(L, ROCKE_ERR_OOM, "smem_names push");
            }
        }
        for(int r = 0; r < op->num_regions; r++)
        {
            rocke_ll_collect_smem(L, op->regions[r]);
        }
    }
}

const char* rocke_ll_smem_global_name(rocke_lower_t* L,
                                      const rocke_value_t* smem,
                                      const rocke_type_t** out_stype)
{
    if(out_stype)
    {
        *out_stype = NULL;
    }
    if(!L || !smem)
    {
        return NULL;
    }
    for(size_t i = 0; i < L->smem_names.len; i++)
    {
        if(smem->name && L->smem_names.data[i].value_name
           && strcmp(L->smem_names.data[i].value_name, smem->name) == 0)
        {
            if(out_stype)
            {
                *out_stype = smem->type;
            }
            return L->smem_names.data[i].gname;
        }
    }
    rocke_ll_fail(
        L, ROCKE_ERR_KEY, "smem value %s never allocated", smem->name ? smem->name : "(null)");
}

/* ====================================================================== */
/* smem live-interval analysis + pool layout (_compute_smem_layout)       */
/* ====================================================================== */

/* Live-interval entry: (first_seq, last_seq) for one smem global. `used` is
 * set when the global appears as an operand of some op (i.e. is actually read,
 * written or address-taken); an allocation that is only defined but never used
 * is dead and must not consume pool space (mirrors Python's `used` set). */
typedef struct ll_live_interval
{
    const char* gname;
    int first_seq;
    int last_seq;
    int used;
} ll_live_interval_t;

/* Mutable state threaded through the DFS walk (mirrors Python closures). */
typedef struct ll_liveness_walk_ctx
{
    rocke_lower_t* L;
    ll_live_interval_t* intervals; /* flat array, len == smem_globals.len */
    size_t num_intervals;
    int counter; /* preorder sequence counter */
} ll_liveness_walk_ctx_t;

/* Return the index into intervals[] for gname, or -1 if not found. */
static int ll_interval_index(const ll_liveness_walk_ctx_t* ctx, const char* gname)
{
    for(size_t i = 0; i < ctx->num_intervals; i++)
    {
        if(strcmp(ctx->intervals[i].gname, gname) == 0)
            return (int)i;
    }
    return -1;
}

/* Look up the global name for an IR value name (%foo) via smem_names. */
static const char* ll_val_to_gname(const rocke_lower_t* L, const char* value_name)
{
    if(!value_name)
        return NULL;
    for(size_t i = 0; i < L->smem_names.len; i++)
    {
        if(L->smem_names.data[i].value_name
           && strcmp(L->smem_names.data[i].value_name, value_name) == 0)
            return L->smem_names.data[i].gname;
    }
    return NULL;
}

/* Number of preorder counter ticks for `op` and its descendants -- mirrors
 * Python `_subtree_size`.  Matches ll_liveness_walk exactly: one tick per op
 * slot (a NULL op still ticks the counter but does not recurse), plus every
 * op in each sub-region.  The subtree rooted at index `idx` therefore occupies
 * contiguous sequence indices [idx, idx + ll_subtree_size(op) - 1]. */
static int ll_subtree_size(const rocke_op_t* op)
{
    int n = 1;
    if(!op)
        return n;
    for(int r = 0; r < op->num_regions; r++)
    {
        const rocke_region_t* reg = op->regions[r];
        if(!reg)
            continue;
        for(int i = 0; i < reg->num_ops; i++)
            n += ll_subtree_size(reg->ops[i]);
    }
    return n;
}

/* DFS preorder walk -- mirrors Python `walk(ops, loop_end)`. */
static void
    ll_liveness_walk(ll_liveness_walk_ctx_t* ctx, const rocke_region_t* region, int loop_end)
{
    if(!region)
        return;
    for(int i = 0; i < region->num_ops; i++)
    {
        const rocke_op_t* op = region->ops[i];
        if(!op)
        {
            ctx->counter++;
            continue;
        }
        int idx = ctx->counter++;

        /* Definition point: tile.smem_alloc */
        if(op->opcode == ROCKE_OP_TILE_SMEM_ALLOC && op->num_results > 0)
        {
            const rocke_value_t* res = op->results[0];
            const char* gn = ll_val_to_gname(ctx->L, res ? res->name : NULL);
            if(gn)
            {
                int ii = ll_interval_index(ctx, gn);
                if(ii >= 0 && ctx->intervals[ii].first_seq < 0)
                {
                    ctx->intervals[ii].first_seq = idx;
                    ctx->intervals[ii].last_seq = idx;
                }
            }
        }

        /* Any operand that is an smem value extends its live range. */
        for(int j = 0; j < op->num_operands; j++)
        {
            const rocke_value_t* v = op->operands[j];
            if(!v || !v->name)
                continue;
            const char* gn = ll_val_to_gname(ctx->L, v->name);
            if(!gn)
                continue;
            int ii = ll_interval_index(ctx, gn);
            if(ii < 0)
                continue;
            ctx->intervals[ii].used = 1;
            int first = (ctx->intervals[ii].first_seq < 0) ? idx : ctx->intervals[ii].first_seq;
            int new_last = (loop_end >= 0) ? loop_end : idx;
            int last = ctx->intervals[ii].last_seq;
            ctx->intervals[ii].first_seq = (first < idx) ? first : idx;
            ctx->intervals[ii].last_seq = (last > new_last) ? last : new_last;
        }

        /* Recurse into sub-regions. scf.for gets a conservative loop_end: the
         * last preorder index of the loop subtree (idx + size - 1), not the
         * for-op's own (earlier) index, so allocations defined inside the loop
         * are extended across the whole loop and interfere as they must.
         *
         * For a nested loop, an allocation used only in the inner loop can be
         * re-read on a later outer iteration, so its live range must reach the
         * enclosing loop's end too -- take the max with any enclosing loop_end
         * (loop_end < 0 means "no enclosing loop", i.e. Python's None). */
        for(int r = 0; r < op->num_regions; r++)
        {
            int child_loop_end;
            if(op->opcode == ROCKE_OP_SCF_FOR)
            {
                int own_last = idx + ll_subtree_size(op) - 1;
                child_loop_end
                    = (loop_end < 0) ? own_last : (own_last > loop_end ? own_last : loop_end);
            }
            else
            {
                child_loop_end = loop_end;
            }
            ll_liveness_walk(ctx, op->regions[r], child_loop_end);
        }
    }
}

/* Size of one smem allocation in bytes (Python _seg_size). */
static int ll_smem_seg_size(const rocke_type_t* stype)
{
    if(!stype || !stype->elem || !stype->elem->name)
        return 0;
    const char* n = stype->elem->name;
    int eb;
    if(strcmp(n, "i8") == 0 || strcmp(n, "fp8e4m3") == 0 || strcmp(n, "bf8e5m2") == 0)
        eb = 1;
    else if(strcmp(n, "f16") == 0 || strcmp(n, "bf16") == 0)
        eb = 2;
    else if(strcmp(n, "i32") == 0 || strcmp(n, "f32") == 0)
        eb = 4;
    else if(strcmp(n, "i64") == 0)
        eb = 8;
    else
        eb = 2; /* default */
    int seg = eb;
    for(int d = 0; d < stype->rank; d++)
        seg *= stype->shape[d];
    return seg;
}

/* Alignment for one smem allocation (Python _align). */
static int ll_smem_align(const rocke_type_t* stype)
{
    if(!stype || !stype->elem || !stype->elem->name)
        return 4;
    const char* n = stype->elem->name;
    if(strcmp(n, "i8") == 0 || strcmp(n, "fp8e4m3") == 0 || strcmp(n, "bf8e5m2") == 0)
        return 16;
    return 4;
}

void rocke_ll_compute_smem_layout(rocke_lower_t* L)
{
    if(!L)
        return;

    /* Build the pool name unconditionally (needed even when empty). */
    const char* kname = L->kernel ? L->kernel->name : "";
    L->smem_pool_name = rocke_arena_printf(&L->arena, "@smem_pool.%s", kname ? kname : "");
    L->smem_pool_size = 0;

    /* Initialize offset vector to zero. */
    rocke_vec_init(&L->smem_offsets);

    size_t n = L->smem_globals.len;
    if(n == 0)
        return;

    /* ---- live-interval analysis ---- */
    ll_live_interval_t* intervals
        = (ll_live_interval_t*)rocke_arena_calloc(&L->arena, n * sizeof(*intervals));
    if(!intervals)
        rocke_ll_fail(L, ROCKE_ERR_OOM, "smem_layout: intervals alloc");

    for(size_t i = 0; i < n; i++)
    {
        intervals[i].gname = L->smem_globals.data[i].gname;
        intervals[i].first_seq = -1; /* unset */
        intervals[i].last_seq = -1;
        intervals[i].used = 0;
    }

    ll_liveness_walk_ctx_t ctx;
    ctx.L = L;
    ctx.intervals = intervals;
    ctx.num_intervals = n;
    ctx.counter = 0;
    ll_liveness_walk(&ctx, L->kernel ? L->kernel->body : NULL, -1);

    /* Fix up allocations with no uses: first_seq = last_seq = 0. */
    for(size_t i = 0; i < n; i++)
    {
        if(intervals[i].first_seq < 0)
        {
            intervals[i].first_seq = 0;
            intervals[i].last_seq = 0;
        }
    }

    /* ---- sort by live-interval start (stable: preserve declaration order for
     *      ties, like Python's sorted key) ---- */
    /* Build an index array and sort it. Simple insertion sort (few allocs). */
    int* order = (int*)rocke_arena_calloc(&L->arena, n * sizeof(int));
    if(!order)
        rocke_ll_fail(L, ROCKE_ERR_OOM, "smem_layout: order alloc");
    for(size_t i = 0; i < n; i++)
        order[i] = (int)i;
    /* Insertion sort by first_seq. */
    for(size_t i = 1; i < n; i++)
    {
        int key = order[i];
        int key_first = intervals[key].first_seq;
        int j = (int)i - 1;
        while(j >= 0 && intervals[order[j]].first_seq > key_first)
        {
            order[j + 1] = order[j];
            j--;
        }
        order[j + 1] = key;
    }

    /* ---- greedy interval packing ---- */
    /* Each slot: (offset, size, last_seq). */
    typedef struct ll_slot
    {
        int offset;
        int size;
        int last_seq;
    } ll_slot_t;

    ll_slot_t* slots = (ll_slot_t*)rocke_arena_calloc(&L->arena, n * sizeof(*slots));
    if(!slots)
        rocke_ll_fail(L, ROCKE_ERR_OOM, "smem_layout: slots alloc");
    int num_slots = 0;

    /* Offset array in declaration order (parallel to smem_globals). */
    int* offsets = (int*)rocke_arena_calloc(&L->arena, n * sizeof(int));
    if(!offsets)
        rocke_ll_fail(L, ROCKE_ERR_OOM, "smem_layout: offsets alloc");

    for(size_t si = 0; si < n; si++)
    {
        int gi = order[si]; /* smem_globals index */
        /* Dead allocations (never read/written/address-taken) consume no pool
         * space: the pre-pool lowering emitted them as their own addrspace(3)
         * globals that the AMDGPU backend dead-strips. Folding them into the
         * single referenced pool would make their bytes count and can overflow
         * the 64 KB LDS limit (the fp16 D128 nw=4 attention Acc_lds regression).
         * offsets[] is calloc'd to 0, so a dead alloc keeps a harmless offset 0
         * and is simply not given a slot. Mirrors Python's `used` filter. */
        if(!intervals[gi].used)
            continue;
        const rocke_type_t* stype = L->smem_globals.data[gi].stype;
        int seg = ll_smem_seg_size(stype);
        int aln = ll_smem_align(stype);
        int first_seq = intervals[gi].first_seq;
        int last_seq = intervals[gi].last_seq;
        /* Exclusive (cshuffle no-alias) allocations must not reuse another
         * allocation's slot and must never be reused, so they occupy their own
         * byte range. Skipping the reuse search forces a fresh slot; recording
         * it with the sentinel last_seq below keeps it permanently "live" so
         * nothing else packs onto it. (Mirrors Python's _EXCL_LAST_SEQ.) */
        int excl = stype ? stype->smem_exclusive : 0;
        const int ROCKE_LL_EXCL_LAST_SEQ = 1 << 30;

        /* Find the best (lowest aligned-offset) free slot. */
        int best = -1;
        int best_aligned = -1;
        for(int k = 0; excl ? 0 : (k < num_slots); k++)
        {
            if(slots[k].last_seq >= first_seq)
                continue; /* still live, interference */
            int aligned_off = (slots[k].offset + aln - 1) & ~(aln - 1);
            /* Reusing slot k places this allocation at [aligned_off,
             * aligned_off+seg). Because it may be larger than slot k's original
             * footprint, that range can spill upward into a DIFFERENT slot that
             * is still live while this allocation is live -- which would alias
             * two simultaneously-live allocations and corrupt data. Reject any
             * candidate whose placed range overlaps a still-live slot; slots
             * already dead before first_seq are safe to overlap. */
            int placed_end = aligned_off + seg;
            int conflict = 0;
            for(int j = 0; j < num_slots; j++)
            {
                if(j == k || slots[j].last_seq < first_seq)
                    continue;
                if(aligned_off < slots[j].offset + slots[j].size && slots[j].offset < placed_end)
                {
                    conflict = 1;
                    break;
                }
            }
            if(conflict)
                continue;
            if(best < 0 || aligned_off < best_aligned)
            {
                best = k;
                best_aligned = aligned_off;
            }
        }

        if(best >= 0)
        {
            int aligned_off = best_aligned;
            offsets[gi] = aligned_off;
            int new_size = slots[best].size;
            int end_needed = aligned_off - slots[best].offset + seg;
            if(end_needed > new_size)
                new_size = end_needed;
            slots[best].size = new_size;
            slots[best].last_seq = last_seq;
        }
        else
        {
            /* No reusable slot; open a new one at the end of the pool. */
            int current_end = 0;
            for(int k = 0; k < num_slots; k++)
            {
                int e = slots[k].offset + slots[k].size;
                if(e > current_end)
                    current_end = e;
            }
            int aligned_off = (current_end + aln - 1) & ~(aln - 1);
            offsets[gi] = aligned_off;
            slots[num_slots].offset = aligned_off;
            slots[num_slots].size = seg;
            slots[num_slots].last_seq = excl ? ROCKE_LL_EXCL_LAST_SEQ : last_seq;
            num_slots++;
        }
    }

    /* Compute pool size (max slot end, rounded to 16). */
    int pool_size = 0;
    for(int k = 0; k < num_slots; k++)
    {
        int e = slots[k].offset + slots[k].size;
        if(e > pool_size)
            pool_size = e;
    }
    L->smem_pool_size = (pool_size + 15) & ~15;

    /* Populate L->smem_offsets in declaration order. */
    for(size_t i = 0; i < n; i++)
    {
        int rc;
        rocke_vec_push(&L->arena, &L->smem_offsets, offsets[i], rc);
        if(rc != 0)
            rocke_ll_fail(L, ROCKE_ERR_OOM, "smem_layout: offsets push");
    }
}

const char*
    rocke_ll_emit_smem_base_ptr(rocke_lower_t* L, const char* gname, const rocke_type_t* stype)
{
    if(!L)
        return "";
    if(!gname)
        return L->smem_pool_name ? L->smem_pool_name : "";

    /* Find the offset for this global. */
    int offset = 0;
    for(size_t i = 0; i < L->smem_globals.len; i++)
    {
        if(L->smem_globals.data[i].gname && strcmp(L->smem_globals.data[i].gname, gname) == 0)
        {
            if(i < L->smem_offsets.len)
                offset = L->smem_offsets.data[i];
            break;
        }
    }

    if(offset == 0)
        return L->smem_pool_name;

    /* Reuse a base pointer already computed for this allocation in the current
     * block: the byte offset is a compile-time constant, so one GEP per
     * (block, allocation) suffices and dominates all later same-block uses
     * (instructions within a block execute sequentially). Mirrors Python
     * _smem_base_cache. */
    const rocke_ll_block_t* blk = rocke_ll_current(L);
    for(size_t i = 0; i < L->smem_base_cache.len; i++)
    {
        if(L->smem_base_cache.data[i].block == blk && L->smem_base_cache.data[i].gname
           && strcmp(L->smem_base_cache.data[i].gname, gname) == 0)
            return L->smem_base_cache.data[i].base;
    }

    const char* agg_ty = rocke_ll_smem_storage_type(L, stype);
    (void)agg_ty; /* not needed for the byte-GEP */
    const char* base = rocke_ll_fresh(L, "smem_base");
    rocke_ll_emitf(L,
                   "  %s = getelementptr inbounds i8, ptr addrspace(3) %s, i32 %d",
                   base,
                   L->smem_pool_name,
                   offset);

    rocke_ll_smem_base_cache_t ent;
    ent.block = blk;
    ent.gname = gname;
    ent.base = base;
    int rc;
    rocke_vec_push(&L->arena, &L->smem_base_cache, ent, rc);
    if(rc != 0)
        rocke_ll_fail(L, ROCKE_ERR_OOM, "smem_base_cache push");
    return base;
}

/* ====================================================================== */
/* yield-stack helpers DEFINED IN CONTROL bucket -- not here.             */
/* ====================================================================== */

/* ====================================================================== */
/* Op dispatch table                                                      */
/* ====================================================================== */

rocke_ll_op_fn rocke_ll_dispatch[ROCKE_OP__COUNT];

void rocke_ll_set_handler(rocke_opcode_t opcode, rocke_ll_op_fn fn)
{
    if(opcode > ROCKE_OP_INVALID && opcode < ROCKE_OP__COUNT)
    {
        rocke_ll_dispatch[(int)opcode] = fn;
    }
}

void rocke_ll_lower_op(rocke_lower_t* L, const rocke_op_t* op)
{
    if(!rocke_ll_live(L) || !op)
    {
        return;
    }
    rocke_opcode_t oc = op->opcode;
    rocke_ll_op_fn fn = NULL;
    if(oc > ROCKE_OP_INVALID && oc < ROCKE_OP__COUNT)
    {
        fn = rocke_ll_dispatch[(int)oc];
    }
    if(fn == NULL)
    {
        rocke_ll_fail(L,
                      ROCKE_ERR_NOTIMPL,
                      "no LLVM lowering for op %s",
                      op->name ? op->name : rocke_opcode_name(oc));
    }
    int dbg = (L->debug && op->loc) ? rocke_ll_debug_location_id(L, L->debug, op->loc) : -1;
    if(dbg < 0)
    {
        fn(L, op);
        return;
    }
    /* One op can append to several blocks and can create new ones (scf.for
     * builds a header/body/latch/exit diamond), so remember where each block
     * ended and label only what this op added. Ops with nested regions lower
     * their children first, and those keep their own tighter locations.
     * Keyed on the block pointer, not the index, because the CFG builders
     * back-patch earlier blocks. */
    size_t n_before = L->blocks.len;
    size_t marks_base = L->dbg_marks.len;
    for(size_t i = 0; i < n_before; i++)
    {
        rocke_ll_dbg_mark_t m;
        m.block = L->blocks.data[i];
        m.len = L->blocks.data[i]->lines.len;
        int rc;
        rocke_vec_push(&L->arena, &L->dbg_marks, m, rc);
        if(rc != 0)
        {
            rocke_ll_fail(L, ROCKE_ERR_OOM, "debug marks");
        }
    }
    fn(L, op);
    for(size_t i = 0; i < L->blocks.len; i++)
    {
        rocke_ll_block_t* blk = L->blocks.data[i];
        size_t start = 0;
        for(size_t j = marks_base; j < L->dbg_marks.len; j++)
        {
            if(L->dbg_marks.data[j].block == blk)
            {
                start = L->dbg_marks.data[j].len;
                break;
            }
        }
        rocke_ll_debug_annotate(L, blk, start, dbg);
    }
    L->dbg_marks.len = marks_base;
}

void rocke_ll_lower_region(rocke_lower_t* L, const rocke_region_t* region)
{
    if(!L || !region)
    {
        return;
    }
    for(int i = 0; i < region->num_ops && rocke_ll_live(L); i++)
    {
        rocke_ll_lower_op(L, region->ops[i]);
    }
}

/* Build the dispatch table once: call every per-bucket registration hook. */
static void ll_register_all(void)
{
    memset(rocke_ll_dispatch, 0, sizeof rocke_ll_dispatch);
    rocke_ll_register_arith();
    rocke_ll_register_convert();
    rocke_ll_register_mem();
    rocke_ll_register_mma();
    rocke_ll_register_crosslane();
    rocke_ll_register_vector();
}

/* ====================================================================== */
/* finalize trailers (Python _param_attrs / _format_agpr_alloc)           */
/* ====================================================================== */

const char* rocke_ll_param_attrs(rocke_lower_t* L, const rocke_param_t* p)
{
    if(!L || !p)
    {
        return "";
    }
    if(!p->type || p->type->kind != ROCKE_TYPE_PTR)
    {
        return "";
    }
    char buf[256];
    size_t w = 0;
    buf[0] = '\0';
#define LL_APPEND_ATTR(txt)                                                      \
    do                                                                           \
    {                                                                            \
        int _n = snprintf(buf + w, sizeof buf - w, "%s%s", w ? " " : "", (txt)); \
        if(_n > 0)                                                               \
        {                                                                        \
            w += (size_t)_n;                                                     \
        }                                                                        \
    } while(0)
    if(rocke_attr_get_bool(&p->attrs, "noalias", false))
        LL_APPEND_ATTR("noalias");
    if(rocke_attr_get_bool(&p->attrs, "readonly", false))
        LL_APPEND_ATTR("readonly");
    if(rocke_attr_get_bool(&p->attrs, "writeonly", false))
        LL_APPEND_ATTR("writeonly");
    if(rocke_attr_get_bool(&p->attrs, "nocapture", true))
        LL_APPEND_ATTR("nocapture");
    if(rocke_attr_get_bool(&p->attrs, "nonnull", false))
        LL_APPEND_ATTR("nonnull");
    int64_t align = 0;
    if(rocke_attr_get_int(&p->attrs, "align", &align))
    {
        char a[48];
        snprintf(a, sizeof a, "align %lld", (long long)align);
        LL_APPEND_ATTR(a);
    }
    int64_t deref = 0;
    if(rocke_attr_get_int(&p->attrs, "dereferenceable", &deref))
    {
        char d[64];
        snprintf(d, sizeof d, "dereferenceable(%lld)", (long long)deref);
        LL_APPEND_ATTR(d);
    }
#undef LL_APPEND_ATTR
    if(w == 0)
    {
        return "";
    }
    return rocke_arena_printf(&L->arena, " %s", buf);
}

const char* rocke_ll_format_agpr_alloc(rocke_lower_t* L, const rocke_attr_value_t* v)
{
    if(!L || !v)
    {
        rocke_ll_fail(L, ROCKE_ERR_VALUE, "agpr_alloc must be a (min, max) pair");
    }
    long lo = 0, hi = 0;
    if(v->kind == ROCKE_ATTR_STR)
    {
        const char* text = v->u.s;
        if(!text || !*text)
        {
            rocke_ll_fail(
                L, ROCKE_ERR_VALUE, "agpr_alloc string must be 'min,max', not empty/'none'");
        }
        /* skip leading ws */
        while(*text == ' ' || *text == '\t')
        {
            text++;
        }
        if(strncmp(text, "none", 4) == 0 || strncmp(text, "None", 4) == 0)
        {
            rocke_ll_fail(
                L, ROCKE_ERR_VALUE, "agpr_alloc string must be 'min,max', not empty/'none'");
        }
        const char* comma = strchr(text, ',');
        if(!comma)
        {
            rocke_ll_fail(L, ROCKE_ERR_VALUE, "agpr_alloc must contain two unsigned integers");
        }
        lo = strtol(text, NULL, 10);
        hi = strtol(comma + 1, NULL, 10);
    }
    else if(v->kind == ROCKE_ATTR_INT_LIST && v->u.ilist.count == 2)
    {
        /* A two-element list of bare ints (the (min, max) pair). */
        lo = (long)v->u.ilist.ints[0];
        hi = (long)v->u.ilist.ints[1];
    }
    else if(v->kind == ROCKE_ATTR_LIST && v->u.list.count == 2)
    {
        /* A two-element list of int maps; tolerate by reading [0]/[1] ints. */
        const rocke_attr_value_t* e0 = rocke_attr_get(v->u.list.items[0], "value");
        const rocke_attr_value_t* e1 = rocke_attr_get(v->u.list.items[1], "value");
        lo = (e0 && e0->kind == ROCKE_ATTR_INT) ? (long)e0->u.i : 0;
        hi = (e1 && e1->kind == ROCKE_ATTR_INT) ? (long)e1->u.i : 0;
    }
    else if(v->kind == ROCKE_ATTR_INT)
    {
        lo = hi = (long)v->u.i;
    }
    else
    {
        rocke_ll_fail(
            L, ROCKE_ERR_VALUE, "agpr_alloc must be a (min, max) pair or 'min,max' string");
    }
    if(lo < 0 || hi < 0)
    {
        rocke_ll_fail(L, ROCKE_ERR_VALUE, "agpr_alloc values must be unsigned");
    }
    if(lo > hi)
    {
        rocke_ll_fail(L, ROCKE_ERR_VALUE, "agpr_alloc min must be <= max");
    }
    return rocke_arena_printf(&L->arena, "%ld,%ld", lo, hi);
}

static const char* rocke_ll_scheduler_strategy(rocke_lower_t* L, const rocke_attr_value_t* v)
{
    if(!v || v->kind != ROCKE_ATTR_STR || !v->u.s)
    {
        rocke_ll_fail(L, ROCKE_ERR_VALUE, "scheduler_strategy must be a string");
    }
    const char* strategy = v->u.s;
    if(strcmp(strategy, "max-ilp") != 0 && strcmp(strategy, "max-memory-clause") != 0
       && strcmp(strategy, "iterative-ilp") != 0 && strcmp(strategy, "iterative-minreg") != 0
       && strcmp(strategy, "iterative-maxocc") != 0)
    {
        rocke_ll_fail(L,
                      ROCKE_ERR_VALUE,
                      "unsupported scheduler_strategy '%s'; expected one of: "
                      "iterative-ilp, iterative-maxocc, iterative-minreg, max-ilp, "
                      "max-memory-clause, or None",
                      strategy);
    }
    return strategy;
}

/* ====================================================================== */
/* finalize (Python finalize)                                             */
/* ====================================================================== */

void rocke_ll_finalize(rocke_lower_t* L, rocke_strbuf_t* out)
{
    if(!L || !out)
    {
        return;
    }
    /* Terminate the current block with ret void. */
    rocke_ll_block_t* cur = rocke_ll_current(L);
    if(cur && !cur->terminated)
    {
        rocke_ll_block_emit(L, cur, " ret void");
        cur->terminated = true;
    }

    const char* dl = L->backend ? L->backend->datalayout : ROCKE_LL_DATALAYOUT;
    const char* tr = L->backend ? L->backend->triple : ROCKE_LL_TRIPLE;
    rocke_strbuf_appendf(out, "target datalayout = \"%s\"\n", dl ? dl : "");
    rocke_strbuf_appendf(out, "target triple = \"%s\"\n", tr ? tr : "");
    rocke_strbuf_append(out, "\n");

    /* smem pool: a single unified addrspace(3) global backing all smem allocations.
     * align 16 satisfies all segment alignments (strictest: 16 B for ds_read_b64_tr_b8
     * on i8/fp8 tiles). */
    if(L->smem_globals.len > 0 && L->smem_pool_name && L->smem_pool_size > 0)
    {
        rocke_strbuf_appendf(out,
                             "%s = internal unnamed_addr addrspace(3) "
                             "global [%d x i8] poison, align 16\n",
                             L->smem_pool_name,
                             L->smem_pool_size);
        rocke_strbuf_append(out, "\n");
    }

    /* Needed intrinsic declarations, in canonical TABLE order (then dynamic
     * decls). This mirrors finalize iterating self._decls in insertion order.
     *
     * Python builds self._decls as dict(_INTRINSIC_DECLS) then .update(
     * _INTRINSIC_DECLS_LLVM22_OVERRIDES) for the LLVM22 flavor. dict.update
     * REPLACES the value text for an existing key but PRESERVES that key's
     * original insertion position. So we must iterate the base table in order
     * and, per needed key, emit the override text when the flavor is LLVM22 and
     * the key has an override -- NOT emit all overrides in a separate leading
     * loop (which would float overridden keys, e.g. make.buffer.rsrc, to the
     * front of the declare block). */
    bool any_need = false;
    int novr = 0;
    const rocke_ll_decl_t* ovr = rocke_ll_flavor_overrides(L->flavor, &novr);
    for(int i = 0; i < ROCKE_LL_INTRINSIC_DECLS_COUNT; i++)
    {
        const char* k = ROCKE_LL_INTRINSIC_DECLS[i].key;
        const char* decl_text = ROCKE_LL_INTRINSIC_DECLS[i].decl;
        for(int j = 0; j < novr; j++)
        {
            if(strcmp(ovr[j].key, k) == 0)
            {
                decl_text = ovr[j].decl;
                break;
            }
        }
        if(ll_need_has(L, k))
        {
            rocke_strbuf_appendf(out, "%s\n", decl_text);
            any_need = true;
        }
    }
    /* Dynamic decls not in either static table. */
    for(size_t i = 0; i < L->dyn_decls.len; i++)
    {
        const char* k = L->dyn_decls.data[i].key;
        if(!ll_need_has(L, k))
        {
            continue;
        }
        if(ll_resolve_decl(L, k) == L->dyn_decls.data[i].decl && ll_resolve_decl(L, k) != NULL)
        {
            /* only emit if not already covered by a static table row */
            bool in_static = false;
            for(int j = 0; j < ROCKE_LL_INTRINSIC_DECLS_COUNT; j++)
            {
                if(strcmp(ROCKE_LL_INTRINSIC_DECLS[j].key, k) == 0)
                {
                    in_static = true;
                    break;
                }
            }
            if(!in_static)
            {
                rocke_strbuf_appendf(out, "%s\n", L->dyn_decls.data[i].decl);
                any_need = true;
            }
        }
    }
    if(any_need)
    {
        rocke_strbuf_append(out, "\n");
    }

    /* Function header. */
    rocke_strbuf_appendf(out, "define amdgpu_kernel void @%s(", L->kernel ? L->kernel->name : "");
    if(L->kernel)
    {
        for(int i = 0; i < L->kernel->num_params; i++)
        {
            const rocke_param_t* p = L->kernel->params[i];
            const char* tstr = rocke_ll_param_llvm_type(L, p);
            const char* attrs = rocke_ll_param_attrs(L, p);
            rocke_strbuf_appendf(out, "%s%s%s %%%s", i ? ", " : "", tstr, attrs, p->name);
        }
    }
    if(rocke_ll_debug_has_locations(L->debug))
    {
        rocke_strbuf_appendf(out, ") #0 !dbg !%d {\n", L->debug->subprogram_id);
    }
    else
    {
        rocke_strbuf_append(out, ") #0 {\n");
    }

    for(size_t i = 0; i < L->blocks.len; i++)
    {
        const rocke_ll_block_t* blk = L->blocks.data[i];
        rocke_strbuf_appendf(out, "%s:\n", blk->label);
        for(size_t j = 0; j < blk->lines.len; j++)
        {
            rocke_strbuf_appendf(out, "%s\n", blk->lines.data[j]);
        }
    }
    rocke_strbuf_append(out, "}\n\n");

    /* attributes #0. */
    int max_wg = L->kernel ? rocke_kernel_max_workgroup_size(L->kernel) : 256;
    rocke_strbuf_append(out, "attributes #0 = { ");
    rocke_strbuf_append(out, "\"uniform-work-group-size\"=\"true\" ");
    rocke_strbuf_appendf(out, "\"amdgpu-flat-work-group-size\"=\"64,%d\"", max_wg);

    if(L->kernel)
    {
        const rocke_attr_value_t* scheduler
            = rocke_attr_get(&L->kernel->attrs, "scheduler_strategy");
        if(scheduler)
        {
            rocke_strbuf_appendf(out,
                                 " \"amdgpu-sched-strategy\"=\"%s\"",
                                 rocke_ll_scheduler_strategy(L, scheduler));
        }
        /* waves_per_eu mirrors the Python lowerer: a bare int N emits "N,N",
         * a 2-element tuple (lo,hi) -- serialized as the INT_LIST l:[ i:lo, i:hi ]
         * -- emits "lo,hi". */
        const rocke_attr_value_t* wpe_v = rocke_attr_get(&L->kernel->attrs, "waves_per_eu");
        if(wpe_v && wpe_v->kind == ROCKE_ATTR_INT)
        {
            long long n = (long long)wpe_v->u.i;
            rocke_strbuf_appendf(out, " \"amdgpu-waves-per-eu\"=\"%lld,%lld\"", n, n);
        }
        else if(wpe_v && wpe_v->kind == ROCKE_ATTR_INT_LIST && wpe_v->u.ilist.count == 2)
        {
            rocke_strbuf_appendf(out,
                                 " \"amdgpu-waves-per-eu\"=\"%lld,%lld\"",
                                 (long long)wpe_v->u.ilist.ints[0],
                                 (long long)wpe_v->u.ilist.ints[1]);
        }
        const rocke_attr_value_t* agpr = rocke_attr_get(&L->kernel->attrs, "agpr_alloc");
        if(agpr)
        {
            const char* fa = rocke_ll_format_agpr_alloc(L, agpr);
            rocke_strbuf_appendf(out, " \"amdgpu-agpr-alloc\"=\"%s\"", fa);
        }
    }
    rocke_strbuf_append(out, " norecurse nounwind }\n");

    /* Python finalize does "\n".join(out) with a trailing "" element, so the
     * file ends in a single newline; when the fp-atomic metadata is present it
     * is preceded by a blank line (out.append("") before "!1 = !{}"). */
    if(L->needs_fp_atomic_md)
    {
        rocke_strbuf_append(out, "\n!1 = !{}\n");
    }
    if(L->needs_av_scope_md)
    {
        rocke_strbuf_append(out, "\n!3 = !{!\"agent\"}\n");
    }
    if(L->debug)
    {
        rocke_ll_debug_render(L, L->debug, out);
    }
}

/* ====================================================================== */
/* Public entry points (Python lower_kernel_to_llvm)                      */
/* ====================================================================== */

/* Run the lowering against an initialized lowerer `L`. On any failure the
 * per-op handlers (and the spine helpers) raise via rocke_ll_fail -> throw, so
 * this body has no in-band error returns; success produces the heap-owned IR
 * text in *out_text. The caller owns L.arena and destroys it on both paths. */
static void ll_lower_into(rocke_lower_t* L,
                          const rocke_kernel_def_t* kernel,
                          rocke_llvm_flavor_t flavor,
                          const char* arch,
                          char** out_text)
{
    /* Resolve flavor. */
    L->flavor = (flavor == ROCKE_LLVM_FLAVOR_AUTO) ? ll_resolve_flavor() : flavor;
    if(!rocke_llvm_flavor_is_known(L->flavor))
    {
        rocke_ll_fail(L, ROCKE_ERR_VALUE, "unknown LLVM flavor");
    }

    /* Resolve ISA backend. */
    rocke_status_t bst = ROCKE_OK;
    L->backend = rocke_ll_backend_for(arch, &bst);
    if(L->backend == NULL || bst != ROCKE_OK)
    {
        rocke_ll_fail(L,
                      bst != ROCKE_OK ? bst : ROCKE_ERR_KEY,
                      "unknown arch backend %s",
                      arch ? arch : "(null)");
    }
    /* The AMDGPU datalayout is FLAVOR-KEYED (Python backend.datalayout(flavor)
     * via _datalayout_for_flavor): the p8 field drifts between LLVM20 and
     * LLVM22. backend_for installs the LLVM20 form by default; rebind the
     * resolved backend's datalayout to the form for the resolved flavor.
     * L->backend points at the static LL_BACKEND_RESOLVED scratch copy, so this
     * does not mutate the canonical per-arch descriptors. */
    LL_BACKEND_RESOLVED.datalayout = rocke_ll_datalayout_for_flavor(L->flavor);

    /* Debug metadata is opt-in: the kernel carries `debug_info` only when it was
     * built with location capture on, and an op's @loc rides the same
     * serialization, so the engine sees exactly what the Python lowerer sees. */
    const rocke_attr_value_t* dbg_attr = rocke_attr_get(&kernel->attrs, "debug_info");
    if(dbg_attr
       && ((dbg_attr->kind == ROCKE_ATTR_BOOL && dbg_attr->u.b)
           || (dbg_attr->kind == ROCKE_ATTR_INT && dbg_attr->u.i)))
    {
        L->debug = rocke_ll_debug_create(L, kernel->name);
    }

    /* Entry block (ll_make_block raises on OOM). */
    ll_make_block(L, "entry");

    /* Pre-pass + lowering. */
    rocke_ll_collect_smem(L, kernel->body);
    rocke_ll_compute_smem_layout(L);
    rocke_ll_lower_region(L, kernel->body);

    rocke_strbuf_t sb;
    if(rocke_strbuf_init(&sb, 4096) != 0)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "out buffer");
    }
    /* finalize raises (rocke_ll_fail -> throw) on OOM; free `sb` before the
     * exception propagates so the throw path does not leak its heap buffer.
     * Codegen-neutral: the success path is unchanged. */
    try
    {
        rocke_ll_finalize(L, &sb);
    }
    catch(...)
    {
        rocke_strbuf_free(&sb);
        throw;
    }
    if(sb.oom)
    {
        rocke_strbuf_free(&sb);
        rocke_ll_fail(L, ROCKE_ERR_OOM, "finalize OOM");
    }
    char* text = rocke_strbuf_detach(&sb);
    if(text == NULL)
    {
        /* empty builder -> hand back an empty heap string */
        text = (char*)malloc(1);
        if(text == NULL)
        {
            rocke_ll_fail(L, ROCKE_ERR_OOM, "detach");
        }
        text[0] = '\0';
    }
    *out_text = text;
}

static rocke_status_t ll_lower_kernel_to_llvm_ex_impl(const rocke_kernel_def_t* kernel,
                                                      rocke_llvm_flavor_t flavor,
                                                      const char* arch,
                                                      char** out_text,
                                                      char* err,
                                                      size_t err_cap)
{
    if(out_text)
    {
        *out_text = NULL;
    }
    if(err && err_cap > 0)
    {
        err[0] = '\0';
    }
    if(!kernel || !out_text)
    {
        return ROCKE_ERR_VALUE;
    }

    /* Build the dispatch table (idempotent across calls). */
    ll_register_all();

    rocke_lower_t L;
    memset(&L, 0, sizeof L);
    if(rocke_arena_init(&L.arena, 0) != 0)
    {
        return ROCKE_ERR_OOM;
    }
    L.kernel = kernel;
    L.status = ROCKE_OK;
    L.err = (char*)rocke_arena_calloc(&L.arena, ROCKE_ERR_MSG_CAP);
    L.unroll_elide_sync_op = NULL;
    L.needs_fp_atomic_md = false;
    L.needs_av_scope_md = false;
    rocke_vec_init(&L.blocks);
    rocke_vec_init(&L.needs);
    rocke_vec_init(&L.dyn_decls);
    rocke_vec_init(&L.smem_globals);
    rocke_vec_init(&L.smem_names);
    rocke_vec_init(&L.smem_offsets);
    L.smem_pool_size = 0;
    L.smem_pool_name = NULL;
    rocke_vec_init(&L.yield_stack);
    rocke_vec_init(&L.dbg_marks);

    /* A failure anywhere in lowering raises a ckc::Error; catch it here so the
     * arena is always destroyed, then translate it into the legacy status code +
     * caller `err` buffer (keeping the extern "C" ABI unchanged). */
    try
    {
        ll_lower_into(&L, kernel, flavor, arch, out_text);
        rocke_arena_destroy(&L.arena);
        return ROCKE_OK;
    }
    catch(const ckc::Error& e)
    {
        if(err && err_cap > 0)
        {
            snprintf(err, err_cap, "%s", e.what());
        }
        rocke_arena_destroy(&L.arena);
        return e.code();
    }
    catch(const std::bad_alloc& e)
    {
        if(err && err_cap > 0)
        {
            snprintf(err, err_cap, "%s", e.what());
        }
        rocke_arena_destroy(&L.arena);
        return ROCKE_ERR_OOM;
    }
    catch(const std::exception& e)
    {
        if(err && err_cap > 0)
        {
            snprintf(err, err_cap, "%s", e.what());
        }
        rocke_arena_destroy(&L.arena);
        return ROCKE_ERR_VALUE;
    }
    catch(...)
    {
        if(err && err_cap > 0)
        {
            snprintf(err, err_cap, "%s", "unknown C++ exception at extern \"C\" boundary");
        }
        rocke_arena_destroy(&L.arena);
        return ROCKE_ERR_VALUE;
    }
}

} /* namespace ckc */

rocke_status_t rocke_lower_kernel_to_llvm_ex(const rocke_kernel_def_t* kernel,
                                             rocke_llvm_flavor_t flavor,
                                             const char* arch,
                                             char** out_text,
                                             char* err,
                                             size_t err_cap)
{
    return ckc::ll_lower_kernel_to_llvm_ex_impl(kernel, flavor, arch, out_text, err, err_cap);
}

rocke_status_t rocke_lower_kernel_to_llvm(const rocke_kernel_def_t* kernel,
                                          rocke_llvm_flavor_t flavor,
                                          const char* arch,
                                          char** out_text)
{
    return rocke_lower_kernel_to_llvm_ex(kernel, flavor, arch, out_text, NULL, 0);
}
