/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * src/instance_gfx950_deep_fused_conv_pool_gfx950_dfcp_parity_test.c --
 * Self-contained C99 PARITY VERIFICATION HARNESS for the gfx950 (CDNA, wave64,
 * MFMA 32x32x16) arch shim over the deep-fused conv0 -> conv1 -> maxpool kernel,
 * the C mirror of rocke/instances/gfx950/deep_fused_conv_pool.py.
 *
 * WHAT THIS IS.
 *   This is the C-side coverage for tests/parity/deep_fused_conv_pool_emit.c on
 *   the gfx950 arch. Unlike the bare *_emit.c emitters (which only print the .ll
 *   to stdout for an external sha256 driver to compare), this harness builds each
 *   sampled spec through the gfx950 public re-export surface and BYTE-COMPARES
 *   FOUR ASPECTS against a Python rocke gfx950 reference golden, all inside C:
 *
 *     1. the emitted LLVM .ll text
 *     2. the manifest signature (the {name,type} entry list)
 *     3. the launch grid (1, pool_ho // pool_tile_h, pool_wo // pool_tile_w)
 *     4. the kernel name ("rocke_gfx950_deep_fused_conv_pool")
 *
 *   Every spec is built via rocke_gfx950_deep_fused_conv_pool_make_spec(...) (which
 *   stamps the wave64 MFMA geometry + gfx950 kernel name, byte-identically to the
 *   Python make_deep_fused_conv_pool_spec), lowered via
 *   rocke_gfx950_deep_fused_conv_pool_lower_to_llvm(...) with arch="gfx950", and the
 *   re-exported rocke_gfx950_deep_fused_conv_pool_{signature,grid,kernel_name}
 *   entries supply the other three aspects. NO NEW PUBLIC SYMBOLS are introduced;
 *   this is a pure verification translation unit (file-local statics + main()).
 *
 * CONFIG MAP (mirrors deep_fused_conv_pool_emit.py _spec for the gfx950 arch).
 *   The harness covers the gfx950 sampleConfigs from the shared Map -- indices
 *   0,1,2,4,5 -- plus the two distinguishing configs:
 *     - config 3 distinguishes cache_input_footprint=true
 *     - config 4 distinguishes direct_conv0_from_input_cache=true
 *   Indices line up 1:1 with the common emitter's gfx950 rows so the same Python
 *   golden generator feeds both. The gfx950 make_spec hard-codes wave_size=64 and
 *   warp_tile 32x32 (the common defaults), so the make_spec arg list omits those
 *   four geometry args -- matching the Python gfx950 factory keyword surface.
 *
 * GOLDEN FILE CONVENTION.
 *   For config index <i> and aspect <a> in {ll, sig, grid, name}, the golden is
 *   read from "<golden_dir>/gfx950_dfcp_cfg<i>.<a>". The Python golden generator
 *   writes the same four files per config. A missing/!= golden is a FAIL.
 *
 *   usage:  <prog> <golden_dir> [config_index]
 *   With no config_index every config in the map is checked; the exit code is 0
 *   iff all four aspects of every checked config match their golden.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/arena.h"
#include "rocke/helper_rocke.helpers.spec.h"
#include "rocke/instance_gfx950_deep_fused_conv_pool.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm.h"
/* The private gfx950 surface is bound only to keep this harness in lock-step with
 * the shim's internal contract (the ctx + closure phases the public driver wires
 * up); the harness itself drives only the public re-export entries. */
#include "rocke/instance_gfx950_deep_fused_conv_pool_internal.h"

/* ------------------------------------------------------------------ *
 * Config map -- gfx950 sampleConfigs 0,1,2,4,5 (+ 3 distinguishing
 * cache_input_footprint, 4 distinguishing direct_conv0_from_input_cache).
 * ------------------------------------------------------------------ */

/* Build the gfx950-pinned spec for config `idx`. Returns 0 on success, -1 if the
 * index is outside the map. The gfx950 factory stamps wave_size=64 / warp_tile
 * 32x32 / the gfx950 kernel name, so only the common-overridable fields are
 * passed (byte-identical to gfx950/deep_fused_conv_pool.py make_spec kwargs). */
static int make_cfg(int idx, rocke_gfx950_deep_fused_conv_pool_spec_t* spec)
{
    switch(idx)
    {
    case 0:
        *spec = rocke_gfx950_deep_fused_conv_pool_make_spec(
            /*n*/ 1,
            /*h*/ 112,
            /*w*/ 112,
            /*c*/ 64,
            /*k0*/ 64,
            /*k1*/ 64,
            /*r*/ 3,
            /*s*/ 3,
            /*pool_tile_h*/ 4,
            /*pool_tile_w*/ 8,
            /*tile_n*/ 32,
            /*tile_k*/ 16,
            /*conv1_tile_k*/ 0,
            /*warp_m*/ 2,
            /*warp_n*/ 1,
            /*pipeline*/ NULL,
            /*unroll_k*/ false,
            /*async_dma*/ false,
            /*cache_input_footprint*/ false,
            /*direct_conv0_from_input_cache*/ false);
        return 0;
    case 1:
        *spec = rocke_gfx950_deep_fused_conv_pool_make_spec(1,
                                                            56,
                                                            56,
                                                            128,
                                                            128,
                                                            128,
                                                            3,
                                                            3,
                                                            4,
                                                            8,
                                                            32,
                                                            16,
                                                            0,
                                                            2,
                                                            1,
                                                            NULL,
                                                            false,
                                                            false,
                                                            false,
                                                            false);
        return 0;
    case 2:
        *spec = rocke_gfx950_deep_fused_conv_pool_make_spec(1,
                                                            28,
                                                            28,
                                                            256,
                                                            256,
                                                            256,
                                                            3,
                                                            3,
                                                            4,
                                                            8,
                                                            32,
                                                            16,
                                                            0,
                                                            2,
                                                            1,
                                                            NULL,
                                                            false,
                                                            false,
                                                            false,
                                                            false);
        return 0;
    case 3:
        /* Distinguishing config: cache_input_footprint=true. */
        *spec
            = rocke_gfx950_deep_fused_conv_pool_make_spec(1,
                                                          56,
                                                          56,
                                                          32,
                                                          32,
                                                          32,
                                                          3,
                                                          3,
                                                          4,
                                                          8,
                                                          32,
                                                          16,
                                                          0,
                                                          2,
                                                          1,
                                                          NULL,
                                                          false,
                                                          false,
                                                          /*cache_input_footprint*/ true,
                                                          /*direct_conv0_from_input_cache*/ false);
        return 0;
    case 4:
        /* Distinguishing config: direct_conv0_from_input_cache=true. */
        *spec = rocke_gfx950_deep_fused_conv_pool_make_spec(1,
                                                            28,
                                                            28,
                                                            64,
                                                            64,
                                                            64,
                                                            3,
                                                            3,
                                                            4,
                                                            8,
                                                            32,
                                                            16,
                                                            0,
                                                            2,
                                                            1,
                                                            NULL,
                                                            false,
                                                            false,
                                                            /*cache_input_footprint*/ false,
                                                            /*direct_conv0_from_input_cache*/ true);
        return 0;
    case 5:
        /* Passing config that BOTH the gate accepts AND the emitter supports
         * (the 16x16x16-resolvable shape; proves the emit path is byte-faithful,
         * not merely a reject). */
        *spec = rocke_gfx950_deep_fused_conv_pool_make_spec(
            1, 64, 128, 8, 16, 16, 3, 3, 4, 8, 16, 16, 0, 2, 1, NULL, false, false, false, false);
        return 0;
    default:
        return -1;
    }
}

#define ROCKE_DFCP_MAP_SIZE 6

/* ------------------------------------------------------------------ *
 * Golden I/O + byte-compare helpers (file-local).
 * ------------------------------------------------------------------ */

/* Read the whole file at `path` into a malloc'd NUL-terminated buffer. On
 * success returns the buffer (caller frees) and sets *out_len to the byte length
 * (excluding the NUL). Returns NULL on any error. */
static char* read_file(const char* path, size_t* out_len)
{
    FILE* f = fopen(path, "rb");
    if(!f)
        return NULL;
    if(fseek(f, 0, SEEK_END) != 0)
    {
        fclose(f);
        return NULL;
    }
    long n = ftell(f);
    if(n < 0)
    {
        fclose(f);
        return NULL;
    }
    if(fseek(f, 0, SEEK_SET) != 0)
    {
        fclose(f);
        return NULL;
    }
    char* buf = (char*)malloc((size_t)n + 1);
    if(!buf)
    {
        fclose(f);
        return NULL;
    }
    size_t got = fread(buf, 1, (size_t)n, f);
    fclose(f);
    buf[got] = '\0';
    if(out_len)
        *out_len = got;
    return buf;
}

/* Byte-compare `actual` (length actual_len) against the golden file at `path`.
 * Returns 0 on byte-identical match, nonzero otherwise. Prints a short verdict
 * (and, on mismatch, the first differing offset) tagged with `tag`. */
static int
    cmp_golden(const char* tag, int idx, const char* path, const char* actual, size_t actual_len)
{
    size_t glen = 0;
    char* golden = read_file(path, &glen);
    if(!golden)
    {
        fprintf(stderr, "FAIL  cfg%d %-4s : golden missing (%s)\n", idx, tag, path);
        return 1;
    }
    int bad = (glen != actual_len) || (memcmp(golden, actual, actual_len) != 0);
    if(bad)
    {
        size_t off = 0, lim = glen < actual_len ? glen : actual_len;
        while(off < lim && golden[off] == actual[off])
            off++;
        fprintf(stderr,
                "FAIL  cfg%d %-4s : len C=%zu PY=%zu first-diff@%zu\n",
                idx,
                tag,
                actual_len,
                glen,
                off);
    }
    else
    {
        fprintf(stdout, "PASS  cfg%d %-4s : %zu bytes\n", idx, tag, actual_len);
    }
    free(golden);
    return bad ? 1 : 0;
}

/* ------------------------------------------------------------------ *
 * Per-aspect renderers -- produce the canonical text the golden stores.
 * ------------------------------------------------------------------ */

/* Render the signature as one "<name> <type>\n" line per entry, in order. The
 * Python golden writer emits the same line-per-entry form so a list reorder or a
 * type change is caught byte-for-byte. Writes into `out` (capacity out_cap) and
 * sets *out_len. Returns ROCKE_OK or an error status. */
static rocke_status_t render_signature(const rocke_gfx950_deep_fused_conv_pool_spec_t* spec,
                                       char* out,
                                       size_t out_cap,
                                       size_t* out_len)
{
    rocke_arena_t arena;
    if(rocke_arena_init(&arena, 4096) != 0)
        return ROCKE_ERR_OOM;

    const rocke_sig_entry_t* items = NULL;
    size_t count = 0;
    rocke_status_t st = rocke_gfx950_deep_fused_conv_pool_signature(&arena, spec, &items, &count);
    if(st != ROCKE_OK)
    {
        rocke_arena_destroy(&arena);
        return st;
    }

    size_t used = 0;
    for(size_t i = 0; i < count; i++)
    {
        const char* nm = items[i].name ? items[i].name : "";
        const char* ty = items[i].type ? items[i].type : "";
        int w = snprintf(out + used, out_cap - used, "%s %s\n", nm, ty);
        if(w < 0 || (size_t)w >= out_cap - used)
        {
            rocke_arena_destroy(&arena);
            return ROCKE_ERR_VALUE; /* buffer too small */
        }
        used += (size_t)w;
    }
    out[used] = '\0';
    if(out_len)
        *out_len = used;
    rocke_arena_destroy(&arena);
    return ROCKE_OK;
}

/* Render the grid as "x y z\n". */
static rocke_status_t render_grid(const rocke_gfx950_deep_fused_conv_pool_spec_t* spec,
                                  char* out,
                                  size_t out_cap,
                                  size_t* out_len)
{
    int g[3] = {0, 0, 0};
    rocke_status_t st = rocke_gfx950_deep_fused_conv_pool_grid(spec, g);
    if(st != ROCKE_OK)
        return st;
    int w = snprintf(out, out_cap, "%d %d %d\n", g[0], g[1], g[2]);
    if(w < 0 || (size_t)w >= out_cap)
        return ROCKE_ERR_VALUE;
    if(out_len)
        *out_len = (size_t)w;
    return ROCKE_OK;
}

/* Render the kernel name verbatim (no trailing newline -- it is an ABI string;
 * the Python golden writes it the same way). */
static rocke_status_t render_kernel_name(const rocke_gfx950_deep_fused_conv_pool_spec_t* spec,
                                         char* out,
                                         size_t out_cap,
                                         size_t* out_len)
{
    rocke_status_t st = rocke_gfx950_deep_fused_conv_pool_kernel_name(spec, out, out_cap);
    if(st != ROCKE_OK)
        return st;
    if(out_len)
        *out_len = strlen(out);
    return ROCKE_OK;
}

/* ------------------------------------------------------------------ *
 * Per-config verification: build -> lower -> render 4 aspects -> compare.
 * ------------------------------------------------------------------ */

/* Compose "<dir>/gfx950_dfcp_cfg<idx>.<aspect>" into `out`. */
static void golden_path(char* out, size_t cap, const char* dir, int idx, const char* aspect)
{
    snprintf(out, cap, "%s/gfx950_dfcp_cfg%d.%s", dir, idx, aspect);
}

/* Verify all four aspects of config `idx` against the golden in `dir`.
 * Returns the count of failing aspects (0 == config fully matches). */
static int verify_config(int idx, const char* dir)
{
    rocke_gfx950_deep_fused_conv_pool_spec_t spec;
    if(make_cfg(idx, &spec) != 0)
    {
        fprintf(stderr, "FAIL  cfg%d      : unknown config index\n", idx);
        return 1;
    }

    int fails = 0;
    char path[1024];

    /* --- aspect: kernel name --- */
    {
        char nm[256];
        size_t nlen = 0;
        rocke_status_t st = render_kernel_name(&spec, nm, sizeof(nm), &nlen);
        if(st != ROCKE_OK)
        {
            fprintf(stderr, "FAIL  cfg%d name : render status=%d\n", idx, (int)st);
            fails++;
        }
        else
        {
            golden_path(path, sizeof(path), dir, idx, "name");
            fails += cmp_golden("name", idx, path, nm, nlen);
        }
    }

    /* --- aspect: grid --- */
    {
        char gtxt[64];
        size_t glen = 0;
        rocke_status_t st = render_grid(&spec, gtxt, sizeof(gtxt), &glen);
        if(st != ROCKE_OK)
        {
            fprintf(stderr, "FAIL  cfg%d grid : render status=%d\n", idx, (int)st);
            fails++;
        }
        else
        {
            golden_path(path, sizeof(path), dir, idx, "grid");
            fails += cmp_golden("grid", idx, path, gtxt, glen);
        }
    }

    /* --- aspect: signature --- */
    {
        static char sigtxt[8192];
        size_t slen = 0;
        rocke_status_t st = render_signature(&spec, sigtxt, sizeof(sigtxt), &slen);
        if(st != ROCKE_OK)
        {
            fprintf(stderr, "FAIL  cfg%d sig  : render status=%d\n", idx, (int)st);
            fails++;
        }
        else
        {
            golden_path(path, sizeof(path), dir, idx, "sig");
            fails += cmp_golden("sig", idx, path, sigtxt, slen);
        }
    }

    /* --- aspect: lowered .ll --- *
     * Drive the gfx950 build->lower convenience entry exactly as the Python
     * gfx950 reference does (arch="gfx950", flavor AUTO). */
    {
        char* llvm = NULL;
        char err[ROCKE_ERR_MSG_CAP];
        err[0] = '\0';
        rocke_status_t st = rocke_gfx950_deep_fused_conv_pool_lower_to_llvm(
            &spec, /*arch*/ "gfx950", ROCKE_LLVM_FLAVOR_AUTO, &llvm, err, sizeof(err));
        if(st != ROCKE_OK || !llvm)
        {
            fprintf(stderr,
                    "FAIL  cfg%d ll   : lower status=%d (%s)\n",
                    idx,
                    (int)st,
                    err[0] ? err : "(no message)");
            fails++;
        }
        else
        {
            golden_path(path, sizeof(path), dir, idx, "ll");
            fails += cmp_golden("ll", idx, path, llvm, strlen(llvm));
            free(llvm);
        }
    }

    return fails;
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        fprintf(stderr,
                "usage: %s <golden_dir> [config_index]\n"
                "  Verifies the gfx950 deep-fused conv+pool emit against the\n"
                "  Python golden in <golden_dir> (.ll/.sig/.grid/.name per cfg).\n"
                "  With no config_index, all configs in the map are checked.\n",
                argv[0]);
        return 2;
    }
    const char* dir = argv[1];

    int total_fails = 0;
    if(argc >= 3)
    {
        int idx = atoi(argv[2]);
        total_fails = verify_config(idx, dir);
    }
    else
    {
        for(int idx = 0; idx < ROCKE_DFCP_MAP_SIZE; idx++)
        {
            total_fails += verify_config(idx, dir);
        }
    }

    if(total_fails == 0)
    {
        fprintf(stdout, "ALL PASS\n");
        return 0;
    }
    fprintf(stderr, "TOTAL FAILS: %d\n", total_fails);
    return 1;
}
