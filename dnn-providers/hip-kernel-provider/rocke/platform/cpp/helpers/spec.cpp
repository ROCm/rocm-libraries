// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C99 port of rocke/helpers/spec.py: WarpTileBlockSizeMixin,
 * derive_block_size, choose_load_vec, kernel_name_join, SignatureBuilder
 * (with ptr_type_str / sig_param / sig_scalar), and ceil_div_grid.
 *
 * Faithful, builder-free value producers. See the header for the original
 * Python and the contract. The goal is byte-identical return values so the
 * downstream IR (const_i32 of the load_vec, the kernel name string baked into
 * the manifest, the derived block_size) is byte-identical to the Python.
 */
#include "rocke/helper_rocke.helpers.spec.h"

#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------------ *
 * derive_block_size / WarpTileBlockSizeMixin
 * ------------------------------------------------------------------ */

int rocke_warp_tile_block_size(int warp_m, int warp_n, int warp_k, int wave_size)
{
    /* Python: int(warp_m) * int(warp_n) * int(warp_k) * int(wave_size).
     * The fields originate as Python ints; in every real GEMM-family spec
     * they are small positive integers whose product fits in 32 bits, so a
     * plain int multiply reproduces the value exactly. */
    return warp_m * warp_n * warp_k * wave_size;
}

int rocke_warp_tile_init_block_size(
    int current_block_size, int warp_m, int warp_n, int warp_k, int wave_size)
{
    /* Python: if getattr(self, "block_size", 0) == 0: derive; else keep. */
    if(current_block_size == 0)
    {
        return rocke_warp_tile_block_size(warp_m, warp_n, warp_k, wave_size);
    }
    return current_block_size;
}

/* ------------------------------------------------------------------ *
 * IOSpecRule / validate_io
 * ------------------------------------------------------------------ */

/* The Python dataclass defaults. */
static const char* const rocke_io_default_dtypes[3] = {"f16", "fp16", "bf16"};
static const int rocke_io_default_block_sizes[5] = {64, 128, 256, 512, 1024};
static const int rocke_io_default_vecs[3] = {2, 4, 8};

void rocke_io_spec_rule_init(rocke_io_spec_rule_t* rule, const char* dtype, int block_size, int vec)
{
    if(rule == NULL)
    {
        return;
    }
    rule->dtype = dtype;
    rule->block_size = block_size;
    rule->vec = vec;
    rule->n_per_block_set = 0; /* None */
    rule->n_per_block = 0;
    rule->max_elems_per_thread_set = 0; /* None */
    rule->max_elems_per_thread = 0;
    /* NULL allowed_* => validate_io substitutes the Python default tuples. */
    rule->allowed_dtypes = NULL;
    rule->num_allowed_dtypes = 0;
    rule->allowed_block_sizes = NULL;
    rule->num_allowed_block_sizes = 0;
    rule->allowed_vecs = NULL;
    rule->num_allowed_vecs = 0;
}

/* Format a Python `set(ints)` literal "{a, b, c}" into the arena. For the two
 * default tuples this reproduces CPython's exact hash-bucket iteration order
 * byte-for-byte; for any other (custom) tuple it falls back to the order the
 * caller supplied. The set spelling only ever surfaces through a ValueError
 * message and never enters the IR, so this fallback cannot perturb emitted
 * code. Returns NULL on OOM. */
static const char* rocke_format_int_set(rocke_arena_t* arena, const int* vals, size_t n)
{
    /* CPython: list(set((64,128,256,512,1024))) == [64,256,128,512,1024]
     *          list(set((2,4,8)))               == [8,2,4] */
    static const int def_bs[5] = {64, 128, 256, 512, 1024};
    static const int def_bs_order[5] = {64, 256, 128, 512, 1024};
    static const int def_vec[3] = {2, 4, 8};
    static const int def_vec_order[3] = {8, 2, 4};

    const int* order = vals;
    size_t i;
    char* buf;
    size_t cap;
    size_t pos;

    /* Detect the default block_size / vec tuples (in their declared order) and
     * substitute CPython's set-iteration order. */
    if(n == 5 && memcmp(vals, def_bs, sizeof(def_bs)) == 0)
    {
        order = def_bs_order;
    }
    else if(n == 3 && memcmp(vals, def_vec, sizeof(def_vec)) == 0)
    {
        order = def_vec_order;
    }

    /* "{" + each "<int>" (<=11 chars) + ", " between + "}" + NUL. */
    cap = 2 + n * (11 + 2) + 1;
    buf = (char*)rocke_arena_alloc(arena, cap);
    if(buf == NULL)
    {
        return NULL;
    }
    pos = 0;
    buf[pos++] = '{';
    for(i = 0; i < n; ++i)
    {
        int wrote;
        if(i > 0)
        {
            buf[pos++] = ',';
            buf[pos++] = ' ';
        }
        wrote = snprintf(buf + pos, cap - pos, "%d", order[i]);
        if(wrote < 0 || (size_t)wrote >= cap - pos)
        {
            return NULL;
        }
        pos += (size_t)wrote;
    }
    buf[pos++] = '}';
    buf[pos] = '\0';
    return buf;
}

int rocke_validate_io(rocke_arena_t* arena,
                      const rocke_io_spec_rule_t* rule,
                      const char** out_reason)
{
    const char* const* dtypes;
    size_t n_dtypes;
    const int* block_sizes;
    size_t n_block_sizes;
    const int* vecs;
    size_t n_vecs;
    size_t i;
    int found;
    const char* set_str;

    if(rule == NULL)
    {
        if(out_reason != NULL)
        {
            *out_reason = "invalid rule";
        }
        return 0;
    }

    /* Resolve the allowed_* tuples, substituting the Python defaults on NULL. */
    if(rule->allowed_dtypes != NULL)
    {
        dtypes = rule->allowed_dtypes;
        n_dtypes = rule->num_allowed_dtypes;
    }
    else
    {
        dtypes = rocke_io_default_dtypes;
        n_dtypes = 3;
    }
    if(rule->allowed_block_sizes != NULL)
    {
        block_sizes = rule->allowed_block_sizes;
        n_block_sizes = rule->num_allowed_block_sizes;
    }
    else
    {
        block_sizes = rocke_io_default_block_sizes;
        n_block_sizes = 5;
    }
    if(rule->allowed_vecs != NULL)
    {
        vecs = rule->allowed_vecs;
        n_vecs = rule->num_allowed_vecs;
    }
    else
    {
        vecs = rocke_io_default_vecs;
        n_vecs = 3;
    }

    /* if rule.dtype not in rule.allowed_dtypes:
     *     return False, f"unsupported dtype {rule.dtype!r}" */
    found = 0;
    for(i = 0; i < n_dtypes; ++i)
    {
        if(rule->dtype != NULL && dtypes[i] != NULL && strcmp(rule->dtype, dtypes[i]) == 0)
        {
            found = 1;
            break;
        }
    }
    if(!found)
    {
        if(out_reason != NULL)
        {
            /* {x!r} on a str => single-quoted repr. The dtype names in use
             * contain no quotes/backslashes, so 'name' is the exact repr. */
            const char* r = rocke_arena_printf(
                arena, "unsupported dtype '%s'", rule->dtype ? rule->dtype : "");
            *out_reason = (r != NULL) ? r : "unsupported dtype";
        }
        return 0;
    }

    /* if rule.block_size not in rule.allowed_block_sizes:
     *     return False, f"block_size {bs} not in {set(allowed_block_sizes)}" */
    found = 0;
    for(i = 0; i < n_block_sizes; ++i)
    {
        if(rule->block_size == block_sizes[i])
        {
            found = 1;
            break;
        }
    }
    if(!found)
    {
        if(out_reason != NULL)
        {
            set_str = rocke_format_int_set(arena, block_sizes, n_block_sizes);
            if(set_str != NULL)
            {
                const char* r = rocke_arena_printf(
                    arena, "block_size %d not in %s", rule->block_size, set_str);
                *out_reason = (r != NULL) ? r : "block_size not allowed";
            }
            else
            {
                *out_reason = "block_size not allowed";
            }
        }
        return 0;
    }

    /* if rule.vec not in rule.allowed_vecs:
     *     return False, f"vec {vec} not in {set(allowed_vecs)}" */
    found = 0;
    for(i = 0; i < n_vecs; ++i)
    {
        if(rule->vec == vecs[i])
        {
            found = 1;
            break;
        }
    }
    if(!found)
    {
        if(out_reason != NULL)
        {
            set_str = rocke_format_int_set(arena, vecs, n_vecs);
            if(set_str != NULL)
            {
                const char* r = rocke_arena_printf(arena, "vec %d not in %s", rule->vec, set_str);
                *out_reason = (r != NULL) ? r : "vec not allowed";
            }
            else
            {
                *out_reason = "vec not allowed";
            }
        }
        return 0;
    }

    /* if rule.n_per_block is not None: */
    if(rule->n_per_block_set)
    {
        int chunk = rule->block_size * rule->vec; /* block_size * vec */
        /* if rule.n_per_block % chunk:
         *     return False, "n_per_block (X) must be divisible by
         *                    block_size*vec (chunk)" */
        if(chunk == 0 || (rule->n_per_block % chunk) != 0)
        {
            if(out_reason != NULL)
            {
                const char* r = rocke_arena_printf(
                    arena,
                    "n_per_block (%d) must be divisible by block_size*vec (%d)",
                    rule->n_per_block,
                    chunk);
                *out_reason = (r != NULL) ? r : "n_per_block not divisible";
            }
            return 0;
        }
        /* if rule.max_elems_per_thread is not None: */
        if(rule->max_elems_per_thread_set)
        {
            /* elems = rule.n_per_block // rule.block_size */
            int elems = (rule->block_size != 0) ? (rule->n_per_block / rule->block_size) : 0;
            /* if elems > rule.max_elems_per_thread: ... */
            if(elems > rule->max_elems_per_thread)
            {
                if(out_reason != NULL)
                {
                    const char* r = rocke_arena_printf(
                        arena,
                        "elems_per_thread %d > %d; pick a larger block_size or "
                        "a multi-pass kernel",
                        elems,
                        rule->max_elems_per_thread);
                    *out_reason = (r != NULL) ? r : "elems_per_thread too large";
                }
                return 0;
            }
        }
    }

    /* return True, "ok" */
    if(out_reason != NULL)
    {
        *out_reason = "ok";
    }
    return 1;
}

/* ------------------------------------------------------------------ *
 * choose_load_vec
 * ------------------------------------------------------------------ */

rocke_status_t
    rocke_choose_load_vec(int tile_m, int tile_n, int tile_k, int block_size, int* out_vec)
{
    /* Python:
     *   threads = block_size
     *   for v in (8, 4, 2, 1):
     *       if tile_k % v: continue
     *       a_vecs = (tile_m * tile_k) // v
     *       b_vecs = (tile_n * tile_k) // v
     *       if a_vecs < threads or b_vecs < threads: continue
     *       if a_vecs % threads or b_vecs % threads: continue
     *       return v
     *   raise ValueError(...)
     */
    static const int candidates[4] = {8, 4, 2, 1};
    int threads = block_size;
    int i;

    if(out_vec == NULL)
    {
        return ROCKE_ERR_VALUE;
    }

    for(i = 0; i < 4; ++i)
    {
        int v = candidates[i];
        int a_vecs;
        int b_vecs;

        if(tile_k % v)
        {
            continue;
        }
        a_vecs = (tile_m * tile_k) / v;
        b_vecs = (tile_n * tile_k) / v;
        if(a_vecs < threads || b_vecs < threads)
        {
            continue;
        }
        if((threads != 0) && (a_vecs % threads || b_vecs % threads))
        {
            continue;
        }
        /* threads==0 would make Python raise ZeroDivisionError on the modulo;
         * guard above leaves that path to fail through to ValueError, matching
         * "no usable load_vec" rather than crashing. */
        if(threads == 0)
        {
            continue;
        }
        *out_vec = v;
        return ROCKE_OK;
    }

    /* Python raises ValueError("no usable load_vec ..."). */
    return ROCKE_ERR_VALUE;
}

/* ------------------------------------------------------------------ *
 * kernel_name_join
 * ------------------------------------------------------------------ */

/* Append a NUL-terminated string into out at *pos, bounded by out_cap.
 * Returns 0 on success, -1 if it would overflow. Always keeps out
 * NUL-terminated on success. */
static int knj_append(char* out, size_t out_cap, size_t* pos, const char* s)
{
    size_t n = strlen(s);
    /* need n bytes + the trailing NUL */
    if(*pos + n + 1 > out_cap)
    {
        return -1;
    }
    memcpy(out + *pos, s, n);
    *pos += n;
    out[*pos] = '\0';
    return 0;
}

rocke_status_t rocke_kernel_name_join(const char* prefix,
                                      const char* const* parts,
                                      size_t num_parts,
                                      const char* const* flag_names,
                                      const int* flag_on,
                                      size_t num_flags,
                                      char* out,
                                      size_t out_cap,
                                      size_t* out_len)
{
    size_t pos = 0;
    int wrote_any = 0; /* tracks whether a separator is needed before next */
    size_t i;

    if(out == NULL || out_cap == 0)
    {
        return ROCKE_ERR_VALUE;
    }
    out[0] = '\0';

    /* body = "_".join(p for p in (prefix, *parts) if p)
     * Python drops empty / falsy strings (empty string is falsy; a None
     * would also be skipped). We skip NULL and "" identically. */
    if(prefix != NULL && prefix[0] != '\0')
    {
        if(knj_append(out, out_cap, &pos, prefix) != 0)
        {
            return ROCKE_ERR_VALUE;
        }
        wrote_any = 1;
    }

    for(i = 0; i < num_parts; ++i)
    {
        const char* p = parts ? parts[i] : NULL;
        if(p == NULL || p[0] == '\0')
        {
            continue;
        }
        if(wrote_any)
        {
            if(knj_append(out, out_cap, &pos, "_") != 0)
            {
                return ROCKE_ERR_VALUE;
            }
        }
        if(knj_append(out, out_cap, &pos, p) != 0)
        {
            return ROCKE_ERR_VALUE;
        }
        wrote_any = 1;
    }

    /* if flags: for name, on in flags.items(): if on: body += f"_{name}"
     * Iteration order is the caller-provided order (== dict insertion order
     * in CPython 3.7+). Note: the flag suffix is appended with "_<name>"
     * unconditionally -- it does NOT participate in the empty-string-drop of
     * the join above, so even an otherwise-empty body would gain a leading
     * "_<name>". We replicate that exactly. */
    for(i = 0; i < num_flags; ++i)
    {
        const char* name;
        if(flag_on == NULL || !flag_on[i])
        {
            continue;
        }
        name = flag_names ? flag_names[i] : NULL;
        if(knj_append(out, out_cap, &pos, "_") != 0)
        {
            return ROCKE_ERR_VALUE;
        }
        if(name != NULL)
        {
            if(knj_append(out, out_cap, &pos, name) != 0)
            {
                return ROCKE_ERR_VALUE;
            }
        }
    }

    /* return body.replace("/", "_") -- rewrite every '/' to '_' in place. */
    {
        size_t j;
        for(j = 0; j < pos; ++j)
        {
            if(out[j] == '/')
            {
                out[j] = '_';
            }
        }
    }

    if(out_len != NULL)
    {
        *out_len = pos;
    }
    return ROCKE_OK;
}

/* ------------------------------------------------------------------ *
 * ptr_type_str / sig_param / sig_scalar
 * ------------------------------------------------------------------ */

const char* rocke_ptr_type_str(rocke_arena_t* arena, const char* dtype, const char* addr_space)
{
    /* Python:
     *   canon = "f16" if dtype in ("f16", "fp16") else dtype
     *   return f"ptr<{canon}, {addr_space}>"
     * NOTE the manifest spacing: "ptr<f16, global>" (a single space after the
     * comma) -- distinct from the IR-internal PtrType.name "ptr<f16,global>".
     * We reproduce the manifest spelling byte-for-byte. */
    const char* canon;

    if(arena == NULL || dtype == NULL)
    {
        return NULL;
    }
    if(strcmp(dtype, "f16") == 0 || strcmp(dtype, "fp16") == 0)
    {
        canon = "f16";
    }
    else
    {
        canon = dtype;
    }
    if(addr_space == NULL)
    {
        addr_space = "global"; /* the Python default argument */
    }
    return rocke_arena_printf(arena, "ptr<%s, %s>", canon, addr_space);
}

rocke_status_t rocke_sig_param(rocke_arena_t* arena,
                               const char* name,
                               const char* dtype,
                               const char* addr_space,
                               rocke_sig_entry_t* out)
{
    /* Python: {"name": name, "type": ptr_type_str(dtype, addr_space)} */
    const char* type_str;
    char* name_copy;

    if(arena == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    type_str = rocke_ptr_type_str(arena, dtype, addr_space);
    if(type_str == NULL)
    {
        return ROCKE_ERR_OOM;
    }
    name_copy = rocke_arena_strdup(arena, name);
    if(name_copy == NULL)
    {
        return ROCKE_ERR_OOM;
    }
    out->name = name_copy;
    out->type = type_str;
    return ROCKE_OK;
}

rocke_status_t
    rocke_sig_scalar(rocke_arena_t* arena, const char* name, const char* ty, rocke_sig_entry_t* out)
{
    /* Python:
     *   if ty not in ("i32", "i64", "f32"):
     *       raise ValueError(f"unsupported scalar arg type {ty!r}")
     *   return {"name": name, "type": ty} */
    char* name_copy;
    char* ty_copy;

    if(arena == NULL || out == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    if(ty == NULL || (strcmp(ty, "i32") != 0 && strcmp(ty, "i64") != 0 && strcmp(ty, "f32") != 0))
    {
        return ROCKE_ERR_VALUE; /* the Python ValueError */
    }
    name_copy = rocke_arena_strdup(arena, name);
    if(name_copy == NULL)
    {
        return ROCKE_ERR_OOM;
    }
    ty_copy = rocke_arena_strdup(arena, ty);
    if(ty_copy == NULL)
    {
        return ROCKE_ERR_OOM;
    }
    out->name = name_copy;
    out->type = ty_copy;
    return ROCKE_OK;
}

/* ------------------------------------------------------------------ *
 * SignatureBuilder
 * ------------------------------------------------------------------ */

/* Ensure room for one more entry. The Python list grows amortised-O(1); we
 * mirror that with arena reallocation (the old slab is leaked into the arena,
 * which is fine -- the arena frees en masse). Returns 0 on success, -1 on OOM. */
static int sb_reserve_one(rocke_signature_builder_t* sb)
{
    size_t new_cap;
    rocke_sig_entry_t* grown;

    if(sb->count < sb->cap)
    {
        return 0;
    }
    new_cap = (sb->cap == 0) ? 4 : sb->cap * 2;
    grown = (rocke_sig_entry_t*)rocke_arena_alloc(sb->arena, new_cap * sizeof(rocke_sig_entry_t));
    if(grown == NULL)
    {
        return -1;
    }
    if(sb->count > 0 && sb->items != NULL)
    {
        memcpy(grown, sb->items, sb->count * sizeof(rocke_sig_entry_t));
    }
    sb->items = grown;
    sb->cap = new_cap;
    return 0;
}

rocke_status_t rocke_signature_builder_init(rocke_signature_builder_t* sb, rocke_arena_t* arena)
{
    if(sb == NULL || arena == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    sb->arena = arena;
    sb->items = NULL;
    sb->count = 0;
    sb->cap = 0;
    sb->status = ROCKE_OK; /* empty _items list, no failure yet */
    return ROCKE_OK;
}

rocke_signature_builder_t* rocke_signature_builder_ptr(rocke_signature_builder_t* sb,
                                                       const char* name,
                                                       const char* dtype,
                                                       const char* addr_space)
{
    rocke_sig_entry_t entry;
    rocke_status_t st;

    if(sb == NULL)
    {
        return sb;
    }
    /* sticky-error: a prior failed call aborts the chain (Python exception). */
    if(sb->status != ROCKE_OK)
    {
        return sb;
    }
    st = rocke_sig_param(sb->arena, name, dtype, addr_space, &entry);
    if(st != ROCKE_OK)
    {
        sb->status = st;
        return sb;
    }
    if(sb_reserve_one(sb) != 0)
    {
        sb->status = ROCKE_ERR_OOM;
        return sb;
    }
    sb->items[sb->count++] = entry; /* self._items.append(...) */
    return sb; /* return self */
}

rocke_signature_builder_t*
    rocke_signature_builder_scalar(rocke_signature_builder_t* sb, const char* name, const char* ty)
{
    rocke_sig_entry_t entry;
    rocke_status_t st;

    if(sb == NULL)
    {
        return sb;
    }
    if(sb->status != ROCKE_OK)
    {
        return sb;
    }
    st = rocke_sig_scalar(sb->arena, name, ty, &entry);
    if(st != ROCKE_OK)
    {
        sb->status = st;
        return sb;
    }
    if(sb_reserve_one(sb) != 0)
    {
        sb->status = ROCKE_ERR_OOM;
        return sb;
    }
    sb->items[sb->count++] = entry;
    return sb;
}

rocke_signature_builder_t* rocke_signature_builder_extend(rocke_signature_builder_t* sb,
                                                          const rocke_sig_entry_t* items,
                                                          size_t n)
{
    size_t i;

    if(sb == NULL)
    {
        return sb;
    }
    if(sb->status != ROCKE_OK)
    {
        return sb;
    }
    /* self._items.extend(items): append each element in order. The Python
     * stores the same dict objects; we copy the two pointers per entry. */
    for(i = 0; i < n; ++i)
    {
        if(sb_reserve_one(sb) != 0)
        {
            sb->status = ROCKE_ERR_OOM;
            return sb;
        }
        sb->items[sb->count++] = items[i];
    }
    return sb;
}

rocke_status_t rocke_signature_builder_build(const rocke_signature_builder_t* sb,
                                             const rocke_sig_entry_t** out_items,
                                             size_t* out_count)
{
    if(sb == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    if(sb->status != ROCKE_OK)
    {
        return sb->status; /* surface the aborting failure */
    }
    if(out_items != NULL)
    {
        *out_items = sb->items;
    }
    if(out_count != NULL)
    {
        *out_count = sb->count;
    }
    return ROCKE_OK;
}

/* ------------------------------------------------------------------ *
 * ceil_div_grid
 * ------------------------------------------------------------------ */

rocke_status_t rocke_ceil_div_grid(const int* totals, const int* tiles, size_t num_dims, int out[3])
{
    /* Python:
     *   if not (1 <= len(dims) <= 3): raise ValueError(...)
     *   for total, tile in dims:
     *       if int(tile) <= 0: raise ValueError("tile must be positive...")
     *       out.append((int(total) + int(tile) - 1) // int(tile))
     *   while len(out) < 3: out.append(1)
     *   return (out[0], out[1], out[2]) */
    int scratch[3];
    size_t i;

    if(out == NULL || totals == NULL || tiles == NULL)
    {
        return ROCKE_ERR_VALUE;
    }
    if(!(num_dims >= 1 && num_dims <= 3))
    {
        return ROCKE_ERR_VALUE; /* "ceil_div_grid takes 1-3 pairs" */
    }

    scratch[0] = 1;
    scratch[1] = 1;
    scratch[2] = 1;

    for(i = 0; i < num_dims; ++i)
    {
        int total = totals[i];
        int tile = tiles[i];
        if(tile <= 0)
        {
            return ROCKE_ERR_VALUE; /* "tile must be positive" */
        }
        /* Python floor-division on these non-negative values == truncation.
         * Compute the ceiling without the `total + tile - 1` sum so a total near
         * INT_MAX cannot overflow signed int (matches arbitrary-precision Python). */
        scratch[i] = total / tile + ((total % tile != 0) ? 1 : 0);
    }

    out[0] = scratch[0];
    out[1] = scratch[1];
    out[2] = scratch[2];
    return ROCKE_OK;
}

/* ------------------------------------------------------------------ *
 * spec reason-string copy
 * ------------------------------------------------------------------ */

void rocke_spec_set_reason(char* reason, size_t cap, const char* msg)
{
    if(reason == NULL || cap == 0)
    {
        return;
    }
    size_t n = strlen(msg);
    if(n >= cap)
    {
        n = cap - 1;
    }
    memcpy(reason, msg, n);
    reason[n] = '\0';
}
