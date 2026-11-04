// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * cpp/core/lower_llvm/debug.cpp -- DWARF line-table metadata for the C++ port
 * of rocke.core.lower_llvm. Mirrors the Python `_DebugInfo` class.
 *
 * The kernel attr `debug_info` (set by IRBuilder when location capture is on)
 * travels through the ck.dsl.ir/v1 serialization along with each op's `@loc`,
 * so the engine receives everything it needs; this file is what turns that back
 * into `!dbg` attachments and a `!DICompileUnit` chain. Without it the engine
 * would silently drop the locations, and a profile taken through the default
 * backend would have no source mapping at all.
 *
 * Byte-identity: the emitted metadata must match the Python lowerer exactly,
 * which is why the ids are allocated in the same order, the caches are keyed on
 * the same tuples, and the nodes are rendered in allocation order. The
 * ROCKE_BACKEND=both gate compares the two engines' text directly, so any drift
 * here fails loudly rather than producing a subtly different profile.
 */

#include <string.h>

#include "rocke/arena.h"
#include "rocke/ir.h"
#include "rocke/lower_llvm_internal.h"
#include "rocke/py_path_split.h"
#include "rocke/strbuf.h"
#include "rocke/vec.h"

namespace ckc
{

/* ---------------------------------------------------------------------- */
/* String helpers (Python _escape_md_string / _di_file)                    */
/* ---------------------------------------------------------------------- */

/* Escape for an LLVM metadata string literal. Same ``\XX`` encoding as
 * ``rocke_ll_escape_asm_string`` / Python ``_escape_md_string``: printable
 * ASCII verbatim, backslash / quote / non-printable as hex. */
static const char* dbg_escape(rocke_lower_t* L, const char* text)
{
    if(text == NULL)
    {
        return "";
    }
    return rocke_ll_escape_asm_string(L, text);
}

/* "!DIFile(filename: ..., directory: ...)" -- split exactly as the Python
 * lowerer's os.path.split does, including on Windows, where a native path
 * carries backslashes and a drive that posixpath would leave in the filename. */
static const char* dbg_di_file(rocke_lower_t* L, const char* path)
{
    const rocke_py_path_split_t cut
        = rocke_py_path_split(path, strlen(path), ROCKE_PY_PATH_WINDOWS);
    char* directory = (char*)rocke_arena_alloc(&L->arena, cut.head_len + 1);
    if(directory == NULL)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "debug difile");
    }
    memcpy(directory, path, cut.head_len);
    directory[cut.head_len] = '\0';
    return rocke_arena_printf(&L->arena,
                              "!DIFile(filename: \"%s\", directory: \"%s\")",
                              dbg_escape(L, path + cut.tail_off),
                              dbg_escape(L, directory));
}

/* ---------------------------------------------------------------------- */
/* loc parsing (Python _parse_frame / _parse_loc)                          */
/* ---------------------------------------------------------------------- */

static bool dbg_all_digits(const char* s, size_t n)
{
    if(n == 0) /* Python "".isdigit() is False */
    {
        return false;
    }
    for(size_t i = 0; i < n; i++)
    {
        if(s[i] < '0' || s[i] > '9')
        {
            return false;
        }
    }
    return true;
}

/* Digits-only (the caller validated), saturating rather than wrapping: a line
 * number that large is nonsense either way, and wrapping could turn it
 * negative. */
static int dbg_atoi(const char* s, size_t n)
{
    long long v = 0;
    for(size_t i = 0; i < n; i++)
    {
        if(v <= 214748364LL)
        {
            v = v * 10 + (s[i] - '0');
        }
    }
    return v > 2147483647LL ? 2147483647 : (int)v;
}

/* Parse "<path>:<line>[:<col>[:<func>]]". Fields are taken from the right and
 * validated rather than split positionally, so a path containing a colon still
 * parses. Returns false when the text carries no usable line number. */
static bool
    dbg_parse_frame(rocke_lower_t* L, const char* text, size_t len, rocke_ll_dbg_frame_t* out)
{
    /* Split on ':' into (offset, length) parts, mirroring str.split(":"). */
    size_t starts[64];
    size_t lens[64];
    int count = 0;
    size_t begin = 0;
    for(size_t i = 0; i <= len; i++)
    {
        if(i == len || text[i] == ':')
        {
            if(count < 64)
            {
                starts[count] = begin;
                lens[count] = i - begin;
                count++;
            }
            else
            {
                /* More colons than any real path has; treat as unparseable
                 * rather than silently truncating the path. */
                return false;
            }
            begin = i + 1;
        }
    }
    if(count < 2)
    {
        return false;
    }
    const char* func = "";
    size_t func_len = 0;
    if(!dbg_all_digits(text + starts[count - 1], lens[count - 1]))
    {
        func = text + starts[count - 1];
        func_len = lens[count - 1];
        count--;
    }
    int col = 0;
    if(count >= 3 && dbg_all_digits(text + starts[count - 1], lens[count - 1])
       && dbg_all_digits(text + starts[count - 2], lens[count - 2]))
    {
        col = dbg_atoi(text + starts[count - 1], lens[count - 1]);
        count--;
    }
    if(count < 2 || !dbg_all_digits(text + starts[count - 1], lens[count - 1]))
    {
        return false;
    }
    int line = dbg_atoi(text + starts[count - 1], lens[count - 1]);
    count--;
    /* path = ":".join(remaining) -- contiguous in the source text. */
    size_t path_len = starts[count - 1] + lens[count - 1] - starts[0];
    if(path_len == 0)
    {
        return false;
    }
    char* path = (char*)rocke_arena_alloc(&L->arena, path_len + 1);
    char* fn = (char*)rocke_arena_alloc(&L->arena, func_len + 1);
    if(path == NULL || fn == NULL)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "debug frame");
    }
    memcpy(path, text + starts[0], path_len);
    path[path_len] = '\0';
    memcpy(fn, func, func_len);
    fn[func_len] = '\0';
    out->path = path;
    out->line = line;
    out->col = col;
    out->func = fn;
    return true;
}

/* Parse a whole Op.loc into frames, innermost first. Unparseable frames are
 * skipped, exactly as the Python does, so a hand-written location cannot make
 * the lowering fail. Splits on unescaped ';' and unescapes '\\' / '\;',
 * matching Python split_loc. */
static int dbg_parse_loc(rocke_lower_t* L, const char* loc, rocke_ll_dbg_frame_t* out, int cap)
{
    int n = 0;
    const char* p = loc;
    for(;;)
    {
        if(n >= cap)
        {
            break;
        }
        size_t cap_buf = strlen(p) + 1;
        char* buf = (char*)rocke_arena_alloc(&L->arena, cap_buf);
        if(buf == NULL)
        {
            rocke_ll_fail(L, ROCKE_ERR_OOM, "debug loc");
        }
        size_t w = 0;
        const char* q = p;
        while(*q)
        {
            if(q[0] == '\\' && q[1] != '\0')
            {
                char nxt = q[1];
                if(nxt == '\\' || nxt == ';')
                {
                    buf[w++] = nxt;
                }
                else
                {
                    buf[w++] = '\\';
                    buf[w++] = nxt;
                }
                q += 2;
                continue;
            }
            if(*q == ';')
            {
                break;
            }
            buf[w++] = *q++;
        }
        buf[w] = '\0';
        if(dbg_parse_frame(L, buf, w, &out[n]))
        {
            n++;
        }
        if(*q != ';')
        {
            break;
        }
        p = q + 1;
    }
    return n;
}

/* ---------------------------------------------------------------------- */
/* Metadata node allocation (Python _DebugInfo)                            */
/* ---------------------------------------------------------------------- */

static int dbg_alloc(rocke_ll_debug_t* D)
{
    return D->next_id++;
}

static void dbg_node(rocke_lower_t* L, rocke_ll_debug_t* D, const char* text)
{
    int rc;
    rocke_vec_push(&L->arena, &D->nodes, rocke_arena_strdup(&L->arena, text), rc);
    if(rc != 0)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "debug node");
    }
}

static bool dbg_is_primary(const rocke_ll_debug_t* D, const char* path)
{
    return D->primary_file != NULL && strcmp(path, D->primary_file) == 0;
}

static int dbg_file_id(rocke_lower_t* L, rocke_ll_debug_t* D, const char* path)
{
    if(dbg_is_primary(D, path))
    {
        return D->primary_file_id;
    }
    for(size_t i = 0; i < D->file_ids.len; i++)
    {
        if(strcmp(D->file_ids.data[i].key, path) == 0)
        {
            return D->file_ids.data[i].id;
        }
    }
    int file_id = dbg_alloc(D);
    dbg_node(L, D, rocke_arena_printf(&L->arena, "!%d = %s", file_id, dbg_di_file(L, path)));
    rocke_ll_dbg_str_id_t ent;
    ent.key = rocke_arena_strdup(&L->arena, path);
    ent.id = file_id;
    int rc;
    rocke_vec_push(&L->arena, &D->file_ids, ent, rc);
    if(rc != 0)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "debug file id");
    }
    return file_id;
}

/* Scope for a lone frame in a file other than the kernel's own (Python
 * _lexical_block_id): DILexicalBlockFile is what clang emits for #line. */
static int dbg_lexical_block_id(rocke_lower_t* L, rocke_ll_debug_t* D, const char* path)
{
    if(dbg_is_primary(D, path))
    {
        return D->subprogram_id;
    }
    for(size_t i = 0; i < D->block_ids.len; i++)
    {
        if(strcmp(D->block_ids.data[i].key, path) == 0)
        {
            return D->block_ids.data[i].id;
        }
    }
    int file_id = dbg_file_id(L, D, path);
    int scope_id = dbg_alloc(D);
    dbg_node(
        L,
        D,
        rocke_arena_printf(&L->arena,
                           "!%d = !DILexicalBlockFile(scope: !%d, file: !%d, discriminator: 0)",
                           scope_id,
                           D->subprogram_id,
                           file_id));
    rocke_ll_dbg_str_id_t ent;
    ent.key = rocke_arena_strdup(&L->arena, path);
    ent.id = scope_id;
    int rc;
    rocke_vec_push(&L->arena, &D->block_ids, ent, rc);
    if(rc != 0)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "debug block id");
    }
    return scope_id;
}

/* One DISubprogram per (file, function) rather than per call site, so repeated
 * calls share it (Python _inlined_subprogram_id). */
static int dbg_inlined_subprogram_id(rocke_lower_t* L,
                                     rocke_ll_debug_t* D,
                                     const rocke_ll_dbg_frame_t* frame)
{
    for(size_t i = 0; i < D->inlined.len; i++)
    {
        if(strcmp(D->inlined.data[i].path, frame->path) == 0
           && strcmp(D->inlined.data[i].func, frame->func) == 0)
        {
            return D->inlined.data[i].id;
        }
    }
    int file_id = dbg_file_id(L, D, frame->path);
    int mid = dbg_alloc(D);
    rocke_ll_dbg_func_id_t ent;
    ent.path = rocke_arena_strdup(&L->arena, frame->path);
    ent.func = rocke_arena_strdup(&L->arena, frame->func);
    ent.id = mid;
    int rc;
    rocke_vec_push(&L->arena, &D->inlined, ent, rc);
    if(rc != 0)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "debug inlined id");
    }
    const char* name = (frame->func && frame->func[0]) ? frame->func : "<anonymous>";
    dbg_node(L,
             D,
             rocke_arena_printf(&L->arena,
                                "!%d = distinct !DISubprogram(name: \"%s\", scope: !%d, "
                                "file: !%d, line: %d, type: !%d, scopeLine: %d, "
                                "spFlags: DISPFlagDefinition, unit: !%d)",
                                mid,
                                dbg_escape(L, name),
                                file_id,
                                file_id,
                                frame->line,
                                D->subroutine_id,
                                frame->line,
                                D->cu_id));
    return mid;
}

/* ---------------------------------------------------------------------- */
/* Public entry points                                                     */
/* ---------------------------------------------------------------------- */

rocke_ll_debug_t* rocke_ll_debug_create(rocke_lower_t* L, const char* kernel_name)
{
    rocke_ll_debug_t* D = (rocke_ll_debug_t*)rocke_arena_calloc(&L->arena, sizeof(*D));
    if(D == NULL)
    {
        rocke_ll_fail(L, ROCKE_ERR_OOM, "debug info");
    }
    D->kernel_name = rocke_arena_strdup(&L->arena, kernel_name ? kernel_name : "");
    D->flag_id = ROCKE_LL_DEBUG_MD_BASE;
    D->empty_id = ROCKE_LL_DEBUG_MD_BASE + 1;
    D->subroutine_id = ROCKE_LL_DEBUG_MD_BASE + 2;
    D->primary_file_id = ROCKE_LL_DEBUG_MD_BASE + 3;
    D->cu_id = ROCKE_LL_DEBUG_MD_BASE + 4;
    D->subprogram_id = ROCKE_LL_DEBUG_MD_BASE + 5;
    D->next_id = ROCKE_LL_DEBUG_MD_BASE + 6;
    D->primary_file = NULL;
    D->primary_line = 0;
    rocke_vec_init(&D->file_ids);
    rocke_vec_init(&D->block_ids);
    rocke_vec_init(&D->inlined);
    rocke_vec_init(&D->locations);
    rocke_vec_init(&D->nodes);
    return D;
}

bool rocke_ll_debug_has_locations(const rocke_ll_debug_t* D)
{
    return D != NULL && D->primary_file != NULL;
}

/* Intern `loc` and return the id of the innermost !DILocation, or -1 when the
 * location carries no usable frame (Python location_id returning None).
 *
 * A captured call stack becomes a chain of DILocations linked by inlinedAt,
 * exactly how LLVM represents an inlined C++ call. The outermost frame is
 * scoped to the kernel's own subprogram, which is what LLVM requires of the end
 * of an inlining chain.
 */
int rocke_ll_debug_location_id(rocke_lower_t* L, rocke_ll_debug_t* D, const char* loc)
{
    if(D == NULL || loc == NULL || loc[0] == '\0')
    {
        return -1;
    }
    rocke_ll_dbg_frame_t frames[ROCKE_LL_DEBUG_MAX_FRAMES];
    int n = dbg_parse_loc(L, loc, frames, ROCKE_LL_DEBUG_MAX_FRAMES);
    if(n == 0)
    {
        return -1;
    }
    const rocke_ll_dbg_frame_t* outermost = &frames[n - 1];
    if(D->primary_file == NULL)
    {
        /* The outermost frame is the kernel builder's own entry point, so it is
         * the natural file for the compile unit and the subprogram. */
        D->primary_file = rocke_arena_strdup(&L->arena, outermost->path);
        D->primary_line = outermost->line;
    }
    else if(strcmp(outermost->path, D->primary_file) == 0)
    {
        if(outermost->line < D->primary_line)
        {
            D->primary_line = outermost->line;
        }
    }

    int parent = -1;
    for(int depth = 0; depth < n; depth++)
    {
        const rocke_ll_dbg_frame_t* frame = &frames[n - 1 - depth];
        int scope = (depth == 0) ? dbg_lexical_block_id(L, D, frame->path)
                                 : dbg_inlined_subprogram_id(L, D, frame);
        int cached = -1;
        for(size_t i = 0; i < D->locations.len; i++)
        {
            const rocke_ll_dbg_loc_t* e = &D->locations.data[i];
            if(e->scope == scope && e->parent == parent && e->line == frame->line
               && e->col == frame->col && strcmp(e->path, frame->path) == 0
               && strcmp(e->func, frame->func) == 0)
            {
                cached = e->id;
                break;
            }
        }
        if(cached < 0)
        {
            cached = dbg_alloc(D);
            if(parent >= 0)
            {
                dbg_node(L,
                         D,
                         rocke_arena_printf(
                             &L->arena,
                             "!%d = !DILocation(line: %d, column: %d, scope: !%d, inlinedAt: !%d)",
                             cached,
                             frame->line,
                             frame->col,
                             scope,
                             parent));
            }
            else
            {
                dbg_node(L,
                         D,
                         rocke_arena_printf(&L->arena,
                                            "!%d = !DILocation(line: %d, column: %d, scope: !%d)",
                                            cached,
                                            frame->line,
                                            frame->col,
                                            scope));
            }
            rocke_ll_dbg_loc_t ent;
            ent.path = rocke_arena_strdup(&L->arena, frame->path);
            ent.func = rocke_arena_strdup(&L->arena, frame->func);
            ent.line = frame->line;
            ent.col = frame->col;
            ent.scope = scope;
            ent.parent = parent;
            ent.id = cached;
            int rc;
            rocke_vec_push(&L->arena, &D->locations, ent, rc);
            if(rc != 0)
            {
                rocke_ll_fail(L, ROCKE_ERR_OOM, "debug location");
            }
        }
        parent = cached;
    }
    return parent;
}

/* Attach ", !dbg !<id>" to the lines block `blk` grew from index `start` on
 * (Python _DebugInfo.annotate). A line that already carries a !dbg was claimed
 * by a nested op with a tighter location; the innermost one is the useful one. */
void rocke_ll_debug_annotate(rocke_lower_t* L, rocke_ll_block_t* blk, size_t start, int dbg)
{
    for(size_t i = start; i < blk->lines.len; i++)
    {
        char* line = blk->lines.data[i];
        if(line == NULL)
        {
            continue;
        }
        const char* first = line;
        while(*first == ' ' || *first == '\t' || *first == '\n' || *first == '\r' || *first == '\f'
              || *first == '\v')
        {
            first++;
        }
        if(*first == '\0' || *first == ';')
        {
            continue;
        }
        if(strstr(line, ", !dbg !") != NULL)
        {
            continue;
        }
        char* updated = rocke_arena_printf(&L->arena, "%s, !dbg !%d", line, dbg);
        if(updated == NULL)
        {
            rocke_ll_fail(L, ROCKE_ERR_OOM, "debug annotate");
        }
        blk->lines.data[i] = updated;
    }
}

/* Render the module-level debug metadata (Python _DebugInfo.render). Emits
 * nothing when no location was ever interned, so a kernel built with capture on
 * but no usable locations stays byte-identical to one built without it. */
void rocke_ll_debug_render(rocke_lower_t* L, const rocke_ll_debug_t* D, rocke_strbuf_t* out)
{
    if(!rocke_ll_debug_has_locations(D))
    {
        return;
    }
    /* The subprogram stands in for the kernel builder function, so anchor it at
     * the earliest line seen in the primary file. */
    int line = D->primary_line > 1 ? D->primary_line : 1;
    rocke_strbuf_appendf(out, "\n!llvm.module.flags = !{!%d}\n", D->flag_id);
    rocke_strbuf_appendf(out, "!llvm.dbg.cu = !{!%d}\n\n", D->cu_id);
    rocke_strbuf_appendf(out,
                         "!%d = !{i32 2, !\"Debug Info Version\", i32 %d}\n",
                         D->flag_id,
                         ROCKE_LL_DEBUG_INFO_VERSION);
    rocke_strbuf_appendf(out, "!%d = !{}\n", D->empty_id);
    rocke_strbuf_appendf(
        out, "!%d = !DISubroutineType(types: !%d)\n", D->subroutine_id, D->empty_id);
    rocke_strbuf_appendf(out, "!%d = %s\n", D->primary_file_id, dbg_di_file(L, D->primary_file));
    rocke_strbuf_appendf(out,
                         "!%d = distinct !DICompileUnit(language: DW_LANG_Python, file: !%d, "
                         "producer: \"rocke\", isOptimized: true, runtimeVersion: 0, "
                         "emissionKind: LineTablesOnly)\n",
                         D->cu_id,
                         D->primary_file_id);
    rocke_strbuf_appendf(
        out,
        "!%d = distinct !DISubprogram(name: \"%s\", scope: !%d, file: !%d, "
        "line: %d, type: !%d, scopeLine: %d, spFlags: DISPFlagDefinition, unit: !%d)\n",
        D->subprogram_id,
        dbg_escape(L, D->kernel_name),
        D->primary_file_id,
        D->primary_file_id,
        line,
        D->subroutine_id,
        line,
        D->cu_id);
    for(size_t i = 0; i < D->nodes.len; i++)
    {
        rocke_strbuf_appendf(out, "%s\n", D->nodes.data[i]);
    }
}

} /* namespace ckc */
