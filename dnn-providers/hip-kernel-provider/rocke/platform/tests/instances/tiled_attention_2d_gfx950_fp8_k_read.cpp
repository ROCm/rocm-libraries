// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * Host-only test that the C++ gfx950 tiled-2D builder reads fp8 K as fp8 on
 * every QK path (16x16, 32x32 non-transposed, 32x32 transposed).
 *
 * With an fp8 KV cache and use_fp8_mfma_qk the kernel keeps K in LDS as raw
 * fp8 bytes and multiplies each value by k_scale before the bf16 QK MFMA. A
 * QK path that reads that buffer as bf16 hands the MFMA fp8 byte pairs with
 * no k_scale, and the scores come out wrong; the IR verifier does not catch
 * it. The lowered LLVM IR addresses every LDS load, vector or scalar, through
 * a getelementptr over the buffer's own array type, so this test checks that
 * each LDS load reads the element type of the buffer it addresses and that K
 * is read as <N x i8> (the only i8 LDS buffer in these kernels is fp8 K). It
 * also checks that the kernel multiplies by k_scale exactly as many values as
 * it reads fp8 K elements; it does not trace which value each multiply takes.
 *
 * The parser reads the IR the lowering prints today: an LDS load whose pointer
 * is not a getelementptr in that form fails the test rather than passing it.
 *
 * The Python twin is library/tests/test_attn_tiled_2d_gfx950_k_lds_read.py,
 * which traces each fp8 K read to its k_scale multiply; its C++ case needs the
 * rocke_engine binding, which CI does not build, so this is the check of the
 * C++ half that CI runs. The spec and the three QK paths mirror the twin's
 * _BASE and library/tests/_attn_tiled_2d_gfx950_qk_paths.py.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rocke/instance_gfx950_attention_tiled_2d.h"
#include "rocke/lower_llvm.h"

typedef void (*select_qk_path_fn)(rocke_attention_tiled_2d_spec_t* spec);

typedef struct qk_path
{
    const char* name;
    select_qk_path_fn select;
} qk_path_t;

/* Counts over the LDS loads of one lowered kernel. */
typedef struct lds_load_census
{
    int loads;
    int i8_loads;
    int i8_lanes;
    int unchecked_loads;
} lds_load_census_t;

static void select_16x16(rocke_attention_tiled_2d_spec_t* spec)
{
    (void)spec;
}

static void select_32x32_non_transposed(rocke_attention_tiled_2d_spec_t* spec)
{
    spec->use_mfma_32x32 = true;
    spec->block_m_per_warp = 32;
}

static void select_32x32_transposed(rocke_attention_tiled_2d_spec_t* spec)
{
    select_32x32_non_transposed(spec);
    spec->use_transposed_qk_32x32 = true;
}

static rocke_attention_tiled_2d_spec_t fp8_k_in_lds_spec(const qk_path_t* path)
{
    rocke_attention_tiled_2d_spec_t s = rocke_attention_tiled_2d_spec_default();
    s.head_size = 64;
    s.block_size = 32;
    s.num_query_heads = 32;
    s.num_kv_heads = 32;
    s.dtype = "bf16";
    s.use_sinks = false;
    s.sliding_window = 0;
    s.has_softcap = false;
    s.has_tile_size = true;
    s.tile_size = 64;
    s.kv_storage_dtype = "fp8e4m3";
    s.use_fp8_mfma_qk = true;
    path->select(&s);
    return s;
}

/* Copies the innermost element type of the LDS getelementptr that defines
 * `%pointer` into `elem` ("i8" for "[2 x [64 x [64 x i8]]]"). Returns 0 when
 * `%pointer` is not defined by an LDS getelementptr. */
static int lds_buffer_element_type(const char* ll, const char* pointer, char* elem, size_t elem_cap)
{
    char definition[160];
    snprintf(definition, sizeof(definition), "  %%%s = getelementptr inbounds ", pointer);
    const char* source_type = strstr(ll, definition);
    if(source_type == NULL)
        return 0;
    source_type += strlen(definition);
    const char* source_end = strstr(source_type, ", ptr addrspace(3) ");
    const char* line_end = strchr(source_type, '\n');
    if(source_end == NULL || (line_end != NULL && source_end > line_end))
        return 0;

    /* The innermost element follows the last "x " of a nested array type. */
    const char* innermost = source_type;
    for(const char* p = source_type; p + 1 < source_end; ++p)
    {
        if(p[0] == 'x' && p[1] == ' ')
            innermost = p + 2;
    }
    size_t length = 0;
    while(innermost + length < source_end && innermost[length] != ']')
        ++length;
    if(length == 0 || length >= elem_cap)
        return 0;
    memcpy(elem, innermost, length);
    elem[length] = '\0';
    return 1;
}

/* Reads the element type, lane count (1 for a scalar) and pointer name of an
 * LDS load from the text after "load [volatile ]". Returns 0 when it is not
 * "<N x T>, ptr addrspace(3) %p" or "T, ptr addrspace(3) %p". */
static int parse_lds_load(const char* operands, int* lanes, char* read_type, char* pointer)
{
    static const char VOLATILE[] = "volatile ";
    if(strncmp(operands, VOLATILE, sizeof(VOLATILE) - 1) == 0)
        operands += sizeof(VOLATILE) - 1;
    if(sscanf(operands, "<%d x %31[^>]>, ptr addrspace(3) %%%127[^,\n]", lanes, read_type, pointer)
       == 3)
        return 1;
    *lanes = 1;
    return operands[0] != '<'
           && sscanf(operands, "%31[^ ,], ptr addrspace(3) %%%127[^,\n]", read_type, pointer) == 2;
}

/* Walks every LDS load ("= load ... ptr addrspace(3) ...") and compares the
 * type it reads with the element type of the buffer it addresses. A load the
 * parser cannot read counts as unchecked, so a new IR form fails the test. */
static lds_load_census_t census_lds_loads(const char* ll, const char* path_name)
{
    static const char LOAD[] = " = load ";
    lds_load_census_t census = {0, 0, 0, 0};
    for(const char* at = strstr(ll, LOAD); at != NULL; at = strstr(at + 1, LOAD))
    {
        const char* operands = at + strlen(LOAD);
        const char* line_end = strchr(operands, '\n');
        const char* lds_pointer = strstr(operands, "ptr addrspace(3) ");
        if(lds_pointer == NULL || (line_end != NULL && lds_pointer > line_end))
            continue;
        ++census.loads;

        int lanes = 0;
        char read_type[32];
        char pointer[128];
        char stored_type[32];
        if(!parse_lds_load(operands, &lanes, read_type, pointer))
        {
            fprintf(stderr,
                    "%s: cannot read the LDS load \"%.*s\"\n",
                    path_name,
                    (int)(line_end != NULL ? line_end - operands : 80),
                    operands);
            ++census.unchecked_loads;
            continue;
        }
        if(strcmp(read_type, "i8") == 0)
        {
            ++census.i8_loads;
            census.i8_lanes += lanes;
        }
        if(!lds_buffer_element_type(ll, pointer, stored_type, sizeof(stored_type)))
        {
            fprintf(stderr, "%s: no LDS getelementptr defines %%%s\n", path_name, pointer);
            ++census.unchecked_loads;
        }
        else if(strcmp(stored_type, read_type) != 0)
        {
            fprintf(stderr,
                    "%s: %%%s reads a %s buffer as %d x %s\n",
                    path_name,
                    pointer,
                    stored_type,
                    lanes,
                    read_type);
            ++census.unchecked_loads;
        }
    }
    return census;
}

/* Counts the fmul instructions that take the %k_scale kernel argument. */
static int count_k_scale_multiplies(const char* ll)
{
    static const char K_SCALE[] = "%k_scale";
    int count = 0;
    for(const char* at = strstr(ll, K_SCALE); at != NULL; at = strstr(at + 1, K_SCALE))
    {
        const char after = at[sizeof(K_SCALE) - 1];
        if(after != ',' && after != '\n')
            continue;
        const char* line = at;
        while(line > ll && line[-1] != '\n')
            --line;
        const char* fmul = strstr(line, " = fmul ");
        if(fmul != NULL && fmul < at)
            ++count;
    }
    return count;
}

static int check_path(const qk_path_t* path)
{
    const rocke_attention_tiled_2d_spec_t spec = fp8_k_in_lds_spec(path);
    char err[1024];
    char* ll = NULL;
    err[0] = '\0';
    const rocke_status_t st = rocke_gfx950_attention_tiled_2d_lower_to_llvm(
        &spec, "gfx950", ROCKE_LLVM_FLAVOR_LLVM20, &ll, err, sizeof(err));
    if(st != ROCKE_OK || ll == NULL)
    {
        fprintf(stderr, "%s: lowering failed: status=%d err=%s\n", path->name, (int)st, err);
        free(ll);
        return 1;
    }

    const lds_load_census_t census = census_lds_loads(ll, path->name);
    const int k_scale_multiplies = count_k_scale_multiplies(ll);
    free(ll);

    int failures = 0;
    if(census.loads == 0)
    {
        fprintf(stderr, "%s: no LDS load in the kernel\n", path->name);
        ++failures;
    }
    if(census.unchecked_loads != 0)
    {
        fprintf(stderr,
                "%s: %d LDS load(s) not shown to read their buffer's element type\n",
                path->name,
                census.unchecked_loads);
        ++failures;
    }
    if(census.i8_loads == 0)
    {
        fprintf(stderr, "%s: K is never read from LDS as fp8 (<N x i8>)\n", path->name);
        ++failures;
    }
    if(k_scale_multiplies != census.i8_lanes)
    {
        fprintf(stderr,
                "%s: %d fp8 K elements read but %d multiplies by k_scale\n",
                path->name,
                census.i8_lanes,
                k_scale_multiplies);
        ++failures;
    }
    if(failures == 0)
    {
        printf("%-22s %d LDS loads, %d fp8 K reads, %d k_scale multiplies\n",
               path->name,
               census.loads,
               census.i8_loads,
               k_scale_multiplies);
    }
    return failures;
}

int main(void)
{
    static const qk_path_t paths[] = {
        {"16x16", select_16x16},
        {"32x32 non-transposed", select_32x32_non_transposed},
        {"32x32 transposed", select_32x32_transposed},
    };
    int failures = 0;
    for(size_t i = 0; i < sizeof(paths) / sizeof(paths[0]); ++i)
        failures += check_path(&paths[i]);
    if(failures != 0)
    {
        fprintf(stderr, "%d GFX950 TILED-2D FP8 K READ CHECK(S) FAILED\n", failures);
        return 1;
    }
    printf("ALL GFX950 TILED-2D QK PATHS READ FP8 K AS FP8\n");
    return 0;
}
