// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * lower_cktile.c -- C99 port of rocke.core.lower_cktile.
 *
 * Faithful translation of the Python string-emitter. See rocke/lower_cktile.h
 * for the spec-struct contract and error model.
 *
 * The Python module joins a Python list with "\n" and returns the result
 * WITHOUT a trailing newline (the last element of every `parts` list is "",
 * so the join produces a trailing "\n" but no final "\n" after it -- i.e. the
 * output ends with exactly one "\n"). We reproduce that byte-for-byte: every
 * logical line is appended followed by '\n', and the final "" element is
 * emitted as a bare line, yielding the same single trailing newline.
 */

#include "rocke/lower_cktile.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------------ small helpers */

static void rocke__err(char err[ROCKE_ERR_MSG_CAP], const char* fmt, ...)
{
    if(!err)
    {
        return;
    }
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, ROCKE_ERR_MSG_CAP, fmt, ap);
    va_end(ap);
}

/* Append one line + '\n' (the Python "\n".join(parts) semantics, where every
 * part is a complete line). */
static void L(rocke_strbuf_t* sb, const char* line)
{
    rocke_strbuf_append(sb, line);
    rocke_strbuf_append_char(sb, '\n');
}

/* ----------------------------------------------------------- spec-field maps */

/* _PIPELINE_MAP */
static const char* pipeline_to_cktile(const char* p)
{
    if(!p)
    {
        return NULL;
    }
    if(strcmp(p, "mem") == 0)
        return "ck_tile::GemmPipeline::MEMORY";
    if(strcmp(p, "compv3") == 0)
        return "ck_tile::GemmPipeline::COMPUTE_V3";
    if(strcmp(p, "compv4") == 0)
        return "ck_tile::GemmPipeline::COMPUTE_V4";
    return NULL;
}

/* sorted(_PIPELINE_MAP) -> "['compv3', 'compv4', 'mem']" */
#define PIPELINE_CHOICES_STR "['compv3', 'compv4', 'mem']"

/* _SCHEDULER_MAP */
static const char* scheduler_to_cktile(const char* s)
{
    if(!s)
    {
        return NULL;
    }
    if(strcmp(s, "intrawave") == 0)
        return "ck_tile::GemmPipelineScheduler::Intrawave";
    if(strcmp(s, "interwave") == 0)
        return "ck_tile::GemmPipelineScheduler::Interwave";
    return NULL;
}

/* sorted(_SCHEDULER_MAP) -> "['interwave', 'intrawave']" */
#define SCHEDULER_CHOICES_STR "['interwave', 'intrawave']"

/* _DTYPE_MAP */
static const char* dtype_to_cktile(const char* d)
{
    if(!d)
    {
        return NULL;
    }
    if(strcmp(d, "fp16") == 0)
        return "ck_tile::half_t";
    if(strcmp(d, "f16") == 0)
        return "ck_tile::half_t";
    if(strcmp(d, "bf16") == 0)
        return "ck_tile::bf16_t";
    if(strcmp(d, "bfloat16") == 0)
        return "ck_tile::bf16_t";
    if(strcmp(d, "fp32") == 0)
        return "float";
    if(strcmp(d, "f32") == 0)
        return "float";
    if(strcmp(d, "fp8e4m3") == 0)
        return "ck_tile::fp8_t";
    return NULL;
}

/* _layout_3: validate and map each of three R/C letters. Writes the three
 * CK Tile layout strings to out[0..2]. Returns false (and fills err) on the
 * Python ValueError path. */
static bool layout_3(const char* layout, const char* out[3], char err[ROCKE_ERR_MSG_CAP])
{
    if(!layout || strlen(layout) != 3)
    {
        rocke__err(err,
                   "unsupported gemm layout '%s'; expected three letters from "
                   "{'R', 'C'}, e.g. 'RCR'",
                   layout ? layout : "");
        return false;
    }
    for(int i = 0; i < 3; ++i)
    {
        char c = layout[i];
        if(c != 'R' && c != 'C')
        {
            rocke__err(err,
                       "unsupported gemm layout '%s'; expected three letters from "
                       "{'R', 'C'}, e.g. 'RCR'",
                       layout);
            return false;
        }
        out[i] = (c == 'R') ? "ck_tile::tensor_layout::gemm::RowMajor"
                            : "ck_tile::tensor_layout::gemm::ColumnMajor";
    }
    return true;
}

/* _is_fp16_path: A/B/C all fp16/f16 and acc fp32/f32. */
static bool is_fp16_path(const rocke_cktile_data_spec_t* d)
{
    const char* xs[3] = {d->dtype_a, d->dtype_b, d->dtype_c};
    for(int i = 0; i < 3; ++i)
    {
        if(!xs[i] || (strcmp(xs[i], "fp16") != 0 && strcmp(xs[i], "f16") != 0))
        {
            return false;
        }
    }
    return d->dtype_acc && (strcmp(d->dtype_acc, "fp32") == 0 || strcmp(d->dtype_acc, "f32") == 0);
}

static const char* bool_lit(bool v)
{
    return v ? "true" : "false";
}
static const char* py_bool(bool v)
{
    return v ? "True" : "False";
}

/* ---------------------------------------------- kernel_name mangling */

/* kernel_name_join(prefix, *parts, flags): join non-empty parts with '_',
 * append '_<name>' for each truthy flag in order, then replace '/' -> '_'.
 * Appends the result to `sb`. The `parts`/`flag_*` arrays are NULL-terminated
 * by count. */
static void mangle_replace_slash(char* s)
{
    for(; *s; ++s)
    {
        if(*s == '/')
        {
            *s = '_';
        }
    }
}

/* UniversalGemmSpec.kernel_name(). Writes into `buf` (cap bytes). */
static void gemm_kernel_name(const rocke_cktile_gemm_spec_t* spec, char* buf, size_t cap)
{
    const rocke_cktile_tile_spec_t* t = &spec->tile;
    const rocke_cktile_trait_spec_t* tr = &spec->trait;
    int n = 0;
    /* parts joined by '_'; spec->name is the prefix and is assumed non-empty
     * (matches Python which drops only empty strings). */
    n += snprintf(buf + n,
                  (n < (int)cap) ? cap - (size_t)n : 0,
                  "%s_%s_t%dx%dx%d_w%dx%dx%d_wt%dx%dx%d_%s_%s_%s",
                  spec->name,
                  spec->data.dtype_a,
                  t->tile_m,
                  t->tile_n,
                  t->tile_k,
                  t->warp_m,
                  t->warp_n,
                  t->warp_k,
                  t->warp_tile_m,
                  t->warp_tile_n,
                  t->warp_tile_k,
                  tr->pipeline,
                  tr->scheduler,
                  tr->epilogue);

    /* flags, in Python dict insertion order (UniversalGemmSpec.kernel_name):
     *   {"pad", "pers", "bat", "preb", "dtl", "pref", "actt"}
     * kernel_name_join appends "_<name>" for each truthy flag in iteration
     * order. The four trailing flags map to TraitSpec fields that this v1
     * CK-Tile path does not model (preshuffle_b / direct_to_lds /
     * dtl_prefetch / active_tile_skip); they all default to False in
     * gemm_universal.TraitSpec and the emitted CK-Tile body here never
     * enables them, so for every spec this struct can express they are
     * always False and contribute no suffix. Holding them at their default
     * reproduces Python's kernel_name() byte-for-byte across this path's
     * representable domain while preserving the exact suffix ordering. */
    bool pad = (tr->pad_m || tr->pad_n || tr->pad_k);
    bool pers = tr->persistent;
    bool bat = spec->batched;
    bool preb = false; /* TraitSpec.preshuffle_b    (default False) */
    bool dtl = false; /* TraitSpec.direct_to_lds   (default False) */
    bool pref = false; /* TraitSpec.dtl_prefetch    (default False) */
    bool actt = false; /* TraitSpec.active_tile_skip(default False) */
    if(pad)
        n += snprintf(buf + n, (n < (int)cap) ? cap - (size_t)n : 0, "_pad");
    if(pers)
        n += snprintf(buf + n, (n < (int)cap) ? cap - (size_t)n : 0, "_pers");
    if(bat)
        n += snprintf(buf + n, (n < (int)cap) ? cap - (size_t)n : 0, "_bat");
    if(preb)
        n += snprintf(buf + n, (n < (int)cap) ? cap - (size_t)n : 0, "_preb");
    if(dtl)
        n += snprintf(buf + n, (n < (int)cap) ? cap - (size_t)n : 0, "_dtl");
    if(pref)
        n += snprintf(buf + n, (n < (int)cap) ? cap - (size_t)n : 0, "_pref");
    if(actt)
        n += snprintf(buf + n, (n < (int)cap) ? cap - (size_t)n : 0, "_actt");
    (void)n;
    mangle_replace_slash(buf);
}

/* ImplicitGemmConvSpec.kernel_name(). Writes into `buf` (cap bytes). */
static void conv_kernel_name(const rocke_cktile_conv_spec_t* spec, char* buf, size_t cap)
{
    const rocke_cktile_conv_problem_t* p = &spec->problem;
    int n = 0;
    n += snprintf(buf + n,
                  (n < (int)cap) ? cap - (size_t)n : 0,
                  "%s_N%dH%dW%dC%d_K%dR%dS%d_t%dx%dx%d_w%dx%d_a%dx%dx%d_%s_%s",
                  spec->name,
                  p->N,
                  p->Hi,
                  p->Wi,
                  p->C,
                  p->K,
                  p->R,
                  p->S,
                  spec->tile_m,
                  spec->tile_n,
                  spec->tile_k,
                  spec->warp_m,
                  spec->warp_n,
                  spec->warp_tile_m,
                  spec->warp_tile_n,
                  spec->warp_tile_k,
                  spec->pipeline,
                  spec->epilogue);
    /* acc_epilogue.tag(): empty for identity. */
    if(spec->acc_epilogue_tag && spec->acc_epilogue_tag[0])
    {
        n += snprintf(buf + n, (n < (int)cap) ? cap - (size_t)n : 0, "_%s", spec->acc_epilogue_tag);
    }
    if(spec->async_dma)
    {
        n += snprintf(buf + n, (n < (int)cap) ? cap - (size_t)n : 0, "_async");
    }
    (void)n;
    mangle_replace_slash(buf);
}

/* ------------------------------------------------- PipelineTypeTraits block */

static void emit_pipeline_type_traits(rocke_strbuf_t* sb)
{
    L(sb, "// Inlined from example/ck_tile/03_gemm/gemm_utils.hpp so the emitted");
    L(sb, "// source is self-contained. Maps the ``GemmPipeline`` enum to the");
    L(sb, "// pipeline template instantiation -- one specialisation per supported");
    L(sb, "// pipeline variant. Class names match CK Tile's public surface in");
    L(sb, "// ck_tile/ops/gemm/pipeline/* (``AgBgCr`` = ``AGmemBGmemCReg``).");
    L(sb, "template <ck_tile::GemmPipeline>");
    L(sb, "struct PipelineTypeTraits;");
    L(sb, "");
    L(sb, "template <>");
    L(sb, "struct PipelineTypeTraits<ck_tile::GemmPipeline::MEMORY>");
    L(sb, "{");
    L(sb, "    template <typename PipelineProblem>");
    L(sb, "    using GemmPipeline = ck_tile::GemmPipelineAgBgCrMem<PipelineProblem>;");
    L(sb, "};");
    L(sb, "");
    L(sb, "template <>");
    L(sb, "struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V3>");
    L(sb, "{");
    L(sb, "    template <typename PipelineProblem>");
    L(sb, "    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<PipelineProblem>;");
    L(sb, "};");
    L(sb, "");
    L(sb, "template <>");
    L(sb, "struct PipelineTypeTraits<ck_tile::GemmPipeline::COMPUTE_V4>");
    L(sb, "{");
    L(sb, "    template <typename PipelineProblem>");
    L(sb, "    using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV4<PipelineProblem>;");
    L(sb, "};");
    /* The Python _PIPELINE_TYPE_TRAITS triple-quoted constant ends with a
     * trailing newline (the closing """ is on its own line after "};"). That
     * newline is part of the element's content, so when joined it yields one
     * extra blank line after the block. Reproduce it. */
    L(sb, "");
}

/* ===================================================================== GEMM */

rocke_status_t rocke_lower_universal_gemm_to_cktile(const rocke_cktile_gemm_spec_t* spec,
                                                    const char* kernel_name_override,
                                                    rocke_strbuf_t* out,
                                                    char err[ROCKE_ERR_MSG_CAP])
{
    if(err)
    {
        err[0] = '\0';
    }
    if(!spec || !out)
    {
        rocke__err(err, "lower_universal_gemm_to_cktile: NULL spec/out");
        return ROCKE_ERR_VALUE;
    }

    if(!is_fp16_path(&spec->data))
    {
        rocke__err(err,
                   "CK Tile lowering currently supports the fp16 dtype path only; "
                   "extend _DTYPE_MAP and the dtype aliases below to add bf16 / "
                   "fp8 support");
        return ROCKE_ERR_NOTIMPL;
    }

    const rocke_cktile_tile_spec_t* t = &spec->tile;
    const rocke_cktile_trait_spec_t* trait = &spec->trait;

    const char* sched = scheduler_to_cktile(trait->scheduler);
    const char* pipe = pipeline_to_cktile(trait->pipeline);
    if(!pipe)
    {
        rocke__err(err,
                   "unsupported pipeline '%s'; supported: %s",
                   trait->pipeline ? trait->pipeline : "",
                   PIPELINE_CHOICES_STR);
        return ROCKE_ERR_NOTIMPL;
    }
    if(!sched)
    {
        rocke__err(err,
                   "unsupported scheduler '%s'; supported: %s",
                   trait->scheduler ? trait->scheduler : "",
                   SCHEDULER_CHOICES_STR);
        return ROCKE_ERR_NOTIMPL;
    }

    const char* layouts[3];
    if(!layout_3(spec->data.layout, layouts, err))
    {
        return ROCKE_ERR_VALUE;
    }
    const char* a_layout = layouts[0];
    const char* b_layout = layouts[1];
    const char* c_layout = layouts[2];

    /* name override precedence: explicit arg > spec->kernel_name > computed. */
    char namebuf[ROCKE_ERR_MSG_CAP];
    const char* name;
    if(kernel_name_override)
    {
        name = kernel_name_override;
    }
    else if(spec->kernel_name)
    {
        name = spec->kernel_name;
    }
    else
    {
        gemm_kernel_name(spec, namebuf, sizeof namebuf);
        name = namebuf;
    }

    const char* double_smem_buffer = (strcmp(trait->pipeline, "compv4") == 0) ? "true" : "false";
    const char* persistent = bool_lit(trait->persistent);

    const int tp_group_num = 8;
    const int tp_m01 = 4;
    const int num_wave_groups = 1;
    const int block_per_cu = 1;

    const char* acc_ct = dtype_to_cktile(spec->data.dtype_acc);
    const char* a_ct = dtype_to_cktile(spec->data.dtype_a);
    const char* b_ct = dtype_to_cktile(spec->data.dtype_b);
    const char* c_ct = dtype_to_cktile(spec->data.dtype_c);

    char line[512];

    L(out, "// =========================================================");
    L(out, "// Auto-generated by rocke.core.lower_cktile from");
    L(out, "//   UniversalGemmSpec(");
    snprintf(line, sizeof line, "//     name='%s',", spec->name);
    L(out, line);
    snprintf(line,
             sizeof line,
             "//     tile_m=%d, tile_n=%d, tile_k=%d,",
             t->tile_m,
             t->tile_n,
             t->tile_k);
    L(out, line);
    snprintf(line,
             sizeof line,
             "//     warp_m=%d, warp_n=%d, warp_k=%d,",
             t->warp_m,
             t->warp_n,
             t->warp_k);
    L(out, line);
    snprintf(line,
             sizeof line,
             "//     warp_tile_m=%d, warp_tile_n=%d,",
             t->warp_tile_m,
             t->warp_tile_n);
    L(out, line);
    snprintf(line, sizeof line, "//     warp_tile_k=%d,", t->warp_tile_k);
    L(out, line);
    snprintf(line,
             sizeof line,
             "//     pipeline='%s', scheduler='%s',",
             trait->pipeline,
             trait->scheduler);
    L(out, line);
    snprintf(line,
             sizeof line,
             "//     epilogue='%s', pad=(%s,%s,%s),",
             trait->epilogue,
             py_bool(trait->pad_m),
             py_bool(trait->pad_n),
             py_bool(trait->pad_k));
    L(out, line);
    snprintf(line, sizeof line, "//     persistent=%s,", py_bool(trait->persistent));
    L(out, line);
    snprintf(line, sizeof line, "//     layout='%s')", spec->data.layout);
    L(out, line);
    L(out, "//");
    L(out, "// Mirrors example/ck_tile/03_gemm/universal_gemm_invoker.hpp.");
    L(out, "// =========================================================");
    L(out, "");
    L(out, "#include <ck_tile/core.hpp>");
    L(out, "#include <ck_tile/host.hpp>");
    L(out, "#include <ck_tile/ops/gemm.hpp>");
    L(out, "#include <ck_tile/ops/epilogue.hpp>");
    L(out, "");
    emit_pipeline_type_traits(out);
    L(out, "");
    snprintf(line, sizeof line, "namespace rocke_emit_%s {", name);
    L(out, line);
    L(out, "");
    L(out, "// -- per-spec GemmConfig: a single struct of static constexpr,");
    L(out, "// matching GemmConfigBase + per-config overrides in");
    L(out, "// example/ck_tile/03_gemm/gemm_utils.hpp.");
    L(out, "struct GemmConfig {");
    snprintf(line, sizeof line, "    static constexpr ck_tile::index_t M_Tile = %d;", t->tile_m);
    L(out, line);
    snprintf(line, sizeof line, "    static constexpr ck_tile::index_t N_Tile = %d;", t->tile_n);
    L(out, line);
    snprintf(line, sizeof line, "    static constexpr ck_tile::index_t K_Tile = %d;", t->tile_k);
    L(out, line);
    snprintf(line, sizeof line, "    static constexpr ck_tile::index_t M_Warp = %d;", t->warp_m);
    L(out, line);
    snprintf(line, sizeof line, "    static constexpr ck_tile::index_t N_Warp = %d;", t->warp_n);
    L(out, line);
    snprintf(line, sizeof line, "    static constexpr ck_tile::index_t K_Warp = %d;", t->warp_k);
    L(out, line);
    snprintf(line,
             sizeof line,
             "    static constexpr ck_tile::index_t M_Warp_Tile = %d;",
             t->warp_tile_m);
    L(out, line);
    snprintf(line,
             sizeof line,
             "    static constexpr ck_tile::index_t N_Warp_Tile = %d;",
             t->warp_tile_n);
    L(out, line);
    snprintf(line,
             sizeof line,
             "    static constexpr ck_tile::index_t K_Warp_Tile = %d;",
             t->warp_tile_k);
    L(out, line);
    L(out, "");
    snprintf(line, sizeof line, "    static constexpr bool kPadM = %s;", bool_lit(trait->pad_m));
    L(out, line);
    snprintf(line, sizeof line, "    static constexpr bool kPadN = %s;", bool_lit(trait->pad_n));
    L(out, line);
    snprintf(line, sizeof line, "    static constexpr bool kPadK = %s;", bool_lit(trait->pad_k));
    L(out, line);
    L(out, "");
    snprintf(
        line, sizeof line, "    static constexpr bool DoubleSmemBuffer = %s;", double_smem_buffer);
    L(out, line);
    L(out, "    static constexpr bool TransposeC = false;");
    L(out, "    static constexpr bool UseStructuredSparsity = false;");
    L(out, "    static constexpr bool Preshuffle = false;");
    L(out, "    static constexpr bool PermuteA = false;");
    L(out, "    static constexpr bool PermuteB = false;");
    L(out, "");
    snprintf(line,
             sizeof line,
             "    static constexpr ck_tile::index_t NumWaveGroups = %d;",
             num_wave_groups);
    L(out, line);
    snprintf(line, sizeof line, "    static constexpr int kBlockPerCu = %d;", block_per_cu);
    L(out, line);
    snprintf(line,
             sizeof line,
             "    static constexpr ck_tile::index_t TileParitionerGroupNum = %d;",
             tp_group_num);
    L(out, line);
    snprintf(
        line, sizeof line, "    static constexpr ck_tile::index_t TileParitionerM01 = %d;", tp_m01);
    L(out, line);
    L(out, "");
    snprintf(line, sizeof line, "    static constexpr auto Scheduler = %s;", sched);
    L(out, line);
    snprintf(line, sizeof line, "    static constexpr ck_tile::GemmPipeline Pipeline = %s;", pipe);
    L(out, line);
    L(out, "};");
    L(out, "");
    L(out, "// -- spec.data.{dtype_a,b,c,acc,layout} -> CK Tile aliases.");
    snprintf(line, sizeof line, "using ADataType = %s;", a_ct);
    L(out, line);
    snprintf(line, sizeof line, "using BDataType = %s;", b_ct);
    L(out, line);
    snprintf(line, sizeof line, "using AccDataType = %s;", acc_ct);
    L(out, line);
    snprintf(line, sizeof line, "using CDataType = %s;", c_ct);
    L(out, line);
    L(out, "using DsDataType = ck_tile::tuple<>;");
    snprintf(line, sizeof line, "using ALayout = %s;", a_layout);
    L(out, line);
    snprintf(line, sizeof line, "using BLayout = %s;", b_layout);
    L(out, line);
    L(out, "using DsLayout = ck_tile::tuple<>;");
    snprintf(line, sizeof line, "using ELayout = %s;", c_layout);
    L(out, line);
    L(out, "using CDEElementWise = ck_tile::element_wise::PassThrough;");
    L(out, "");
    L(out, "// -- Kernel composition (one-to-one with universal_gemm_invoker.hpp).");
    L(out, "using GemmShape = ck_tile::TileGemmShape<");
    L(out, "    ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,");
    L(out, "    ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,");
    L(out,
      "    ck_tile::sequence<GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, "
      "GemmConfig::K_Warp_Tile>,");
    L(out, "    GemmConfig::PermuteA,");
    L(out, "    GemmConfig::PermuteB>;");
    L(out, "");
    L(out, "using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<");
    L(out, "    GemmShape,");
    L(out, "    GemmConfig::TileParitionerGroupNum,");
    L(out, "    GemmConfig::TileParitionerM01>;");
    L(out, "");
    L(out, "using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<");
    L(out, "    GemmConfig::kPadM, GemmConfig::kPadN, GemmConfig::kPadK,");
    L(out, "    GemmConfig::DoubleSmemBuffer,");
    L(out, "    ALayout, BLayout, ELayout,");
    L(out, "    GemmConfig::TransposeC,");
    L(out, "    GemmConfig::UseStructuredSparsity,");
    snprintf(line, sizeof line, "    /*Persistent=*/%s,", persistent);
    L(out, line);
    L(out, "    GemmConfig::NumWaveGroups,");
    L(out, "    GemmConfig::Preshuffle>;");
    L(out, "");
    L(out, "using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<");
    L(out, "    ADataType, BDataType, AccDataType,");
    L(out, "    GemmShape, GemmUniversalTraits, GemmConfig::Scheduler>;");
    L(out, "");
    L(out, "using GemmPipeline =");
    L(out,
      "    typename ::PipelineTypeTraits<GemmConfig::Pipeline>::template "
      "GemmPipeline<UniversalGemmProblem>;");
    L(out, "");
    L(out, "using GemmEpilogue = ck_tile::CShuffleEpilogue<");
    L(out, "    ck_tile::CShuffleEpilogueProblem<");
    L(out, "        ADataType, BDataType, DsDataType, AccDataType, CDataType,");
    L(out, "        DsLayout, ELayout, CDEElementWise,");
    L(out, "        TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,");
    L(out, "        GemmConfig::M_Warp, GemmConfig::N_Warp,");
    L(out, "        GemmConfig::M_Warp_Tile, GemmConfig::N_Warp_Tile, GemmConfig::K_Warp_Tile,");
    L(out, "        UniversalGemmProblem::TransposeC,");
    L(out, "        GemmConfig::NumWaveGroups,");
    L(out, "        /*FixedVectorSize=*/false,");
    L(out, "        /*VectorSizeC=*/1,");
    L(out, "        /*BlockedXDLN_PerWarp=*/1,");
    L(out, "        GemmConfig::DoubleSmemBuffer>>;");
    L(out, "");
    L(out, "using Kernel = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;");
    L(out, "");
    snprintf(line, sizeof line, "} // namespace rocke_emit_%s", name);
    L(out, line);
    L(out, "");
    L(out, "// -- Host launcher. ABI: void*-friendly so a C / Python ctypes shim");
    L(out, "// can call this directly. Returns the elapsed kernel time in ms,");
    L(out, "// or -1.0f if the kernel rejects the arguments (mirrors how");
    L(out, "// universal_gemm_invoker.hpp's std::runtime_error path would surface");
    L(out, "// via a non-throwing return).");
    snprintf(line, sizeof line, "extern \"C\" float launch_%s(", name);
    L(out, line);
    L(out, "    const void* a_ptr,");
    L(out, "    const void* b_ptr,");
    L(out, "    void* c_ptr,");
    L(out, "    ck_tile::index_t M,");
    L(out, "    ck_tile::index_t N,");
    L(out, "    ck_tile::index_t K,");
    L(out, "    ck_tile::index_t stride_a,");
    L(out, "    ck_tile::index_t stride_b,");
    L(out, "    ck_tile::index_t stride_c,");
    L(out, "    ck_tile::index_t k_batch,");
    L(out, "    hipStream_t stream)");
    L(out, "{");
    snprintf(line, sizeof line, "    using namespace rocke_emit_%s;", name);
    L(out, line);
    L(out, "    ck_tile::GemmHostArgs args;");
    L(out, "    args.a_ptr = a_ptr;");
    L(out, "    args.b_ptr = b_ptr;");
    L(out, "    args.e_ptr = c_ptr;");
    L(out, "    args.M = M;");
    L(out, "    args.N = N;");
    L(out, "    args.K = K;");
    L(out, "    args.stride_A = stride_a;");
    L(out, "    args.stride_B = stride_b;");
    L(out, "    args.stride_E = stride_c;");
    L(out, "    args.k_batch = k_batch;");
    L(out, "");
    L(out, "    ck_tile::stream_config s;");
    L(out, "    s.stream_id_ = stream;");
    L(out, "    s.time_kernel_ = false;");
    L(out, "");
    L(out, "    auto kargs = Kernel::MakeKernelArgs(args);");
    L(out, "    const dim3 grids = Kernel::GridSize(args.M, args.N, args.k_batch);");
    L(out, "    const dim3 blocks = Kernel::BlockSize();");
    L(out, "    if (!Kernel::IsSupportedArgument(kargs)) { return -1.0f; }");
    L(out, "");
    L(out, "    return ck_tile::launch_kernel(");
    L(out, "        s,");
    L(out, "        ck_tile::make_kernel<GemmConfig::kBlockPerCu>(");
    L(out, "            Kernel{}, grids, blocks, 0, kargs));");
    L(out, "}");
    /* Python's `parts` ends with a trailing "" element; "\n".join collapses it
     * into a single terminating newline, which the L() on "}" already emitted.
     * So we do NOT append another blank line here. */

    if(out->oom)
    {
        rocke__err(err, "lower_universal_gemm_to_cktile: out of memory");
        return ROCKE_ERR_OOM;
    }
    return ROCKE_OK;
}

/* ===================================================================== CONV */

rocke_status_t rocke_lower_implicit_gemm_conv_to_cktile(const rocke_cktile_conv_spec_t* spec,
                                                        const char* kernel_name_override,
                                                        rocke_strbuf_t* out,
                                                        char err[ROCKE_ERR_MSG_CAP])
{
    if(err)
    {
        err[0] = '\0';
    }
    if(!spec || !out)
    {
        rocke__err(err, "lower_implicit_gemm_conv_to_cktile: NULL spec/out");
        return ROCKE_ERR_VALUE;
    }

    const char* pipe = pipeline_to_cktile(spec->pipeline);
    if(!pipe)
    {
        rocke__err(err,
                   "conv pipeline '%s'; supported %s",
                   spec->pipeline ? spec->pipeline : "",
                   PIPELINE_CHOICES_STR);
        return ROCKE_ERR_NOTIMPL;
    }
    if(!spec->epilogue
       || (strcmp(spec->epilogue, "cshuffle") != 0 && strcmp(spec->epilogue, "default") != 0))
    {
        rocke__err(err,
                   "conv epilogue '%s'; supported 'cshuffle' / 'default'",
                   spec->epilogue ? spec->epilogue : "");
        return ROCKE_ERR_NOTIMPL;
    }

    char namebuf[ROCKE_ERR_MSG_CAP];
    const char* name;
    if(kernel_name_override)
    {
        name = kernel_name_override;
    }
    else if(spec->kernel_name)
    {
        name = spec->kernel_name;
    }
    else
    {
        conv_kernel_name(spec, namebuf, sizeof namebuf);
        name = namebuf;
    }

    const char* in_dtype = "ck_tile::half_t";
    const char* wei_dtype = "ck_tile::half_t";
    const char* out_dtype = "ck_tile::half_t";
    const char* acc_dtype = "float";

    const int block_per_cu = 1;
    const int num_wave_groups = 1;
    const int num_groups_to_merge = 1;
    const int vector_size_a = 8;
    const int vector_size_b = 8;
    const int vector_size_c = 8;
    const char* double_smem_buffer = (strcmp(spec->pipeline, "compv4") == 0) ? "true" : "false";

    const rocke_cktile_conv_problem_t* p = &spec->problem;
    char line[512];

    L(out, "// =========================================================");
    L(out, "// Auto-generated by rocke.core.lower_cktile from");
    L(out, "//   ImplicitGemmConvSpec(");
    /* problem={spec.problem!r}: dataclass repr emits every field in order. */
    snprintf(line,
             sizeof line,
             "//     name='%s', problem=ConvProblem(N=%d, Hi=%d, Wi=%d, C=%d, "
             "K=%d, R=%d, S=%d, sH=%d, sW=%d, pH=%d, pW=%d, dH=%d, dW=%d),",
             spec->name,
             p->N,
             p->Hi,
             p->Wi,
             p->C,
             p->K,
             p->R,
             p->S,
             p->sH,
             p->sW,
             p->pH,
             p->pW,
             p->dH,
             p->dW);
    L(out, line);
    snprintf(line,
             sizeof line,
             "//     tile_m=%d, tile_n=%d, tile_k=%d,",
             spec->tile_m,
             spec->tile_n,
             spec->tile_k);
    L(out, line);
    snprintf(line, sizeof line, "//     warp_m=%d, warp_n=%d,", spec->warp_m, spec->warp_n);
    L(out, line);
    snprintf(line,
             sizeof line,
             "//     warp_tile=(%d,%d,%d),",
             spec->warp_tile_m,
             spec->warp_tile_n,
             spec->warp_tile_k);
    L(out, line);
    snprintf(
        line, sizeof line, "//     pipeline='%s', epilogue='%s')", spec->pipeline, spec->epilogue);
    L(out, line);
    L(out, "//");
    /* Python concatenates two adjacent string literals into one element:
     * "// Mirrors example/ck_tile/20_grouped_convolution/"
     * "grouped_convolution_forward_invoker.hpp." */
    L(out,
      "// Mirrors example/ck_tile/20_grouped_convolution/grouped_convolution_forward_invoker.hpp.");
    L(out, "// =========================================================");
    L(out, "");
    L(out, "#include <ck_tile/core.hpp>");
    L(out, "#include <ck_tile/host.hpp>");
    L(out, "#include <ck_tile/ops/gemm.hpp>");
    L(out, "#include <ck_tile/ops/grouped_convolution.hpp>");
    L(out, "#include <ck_tile/ops/epilogue.hpp>");
    L(out, "");
    emit_pipeline_type_traits(out);
    L(out, "");
    snprintf(line, sizeof line, "namespace rocke_emit_%s {", name);
    L(out, line);
    L(out, "");
    L(out, "struct ConvConfig {");
    snprintf(line, sizeof line, "    static constexpr ck_tile::index_t M_Tile = %d;", spec->tile_m);
    L(out, line);
    snprintf(line, sizeof line, "    static constexpr ck_tile::index_t N_Tile = %d;", spec->tile_n);
    L(out, line);
    snprintf(line, sizeof line, "    static constexpr ck_tile::index_t K_Tile = %d;", spec->tile_k);
    L(out, line);
    snprintf(line, sizeof line, "    static constexpr ck_tile::index_t M_Warp = %d;", spec->warp_m);
    L(out, line);
    snprintf(line, sizeof line, "    static constexpr ck_tile::index_t N_Warp = %d;", spec->warp_n);
    L(out, line);
    L(out, "    static constexpr ck_tile::index_t K_Warp = 1;");
    snprintf(line,
             sizeof line,
             "    static constexpr ck_tile::index_t M_Warp_Tile = %d;",
             spec->warp_tile_m);
    L(out, line);
    snprintf(line,
             sizeof line,
             "    static constexpr ck_tile::index_t N_Warp_Tile = %d;",
             spec->warp_tile_n);
    L(out, line);
    snprintf(line,
             sizeof line,
             "    static constexpr ck_tile::index_t K_Warp_Tile = %d;",
             spec->warp_tile_k);
    L(out, line);
    L(out, "");
    snprintf(
        line, sizeof line, "    static constexpr bool DoubleSmemBuffer = %s;", double_smem_buffer);
    L(out, line);
    snprintf(line,
             sizeof line,
             "    static constexpr ck_tile::index_t NumWaveGroups = %d;",
             num_wave_groups);
    L(out, line);
    snprintf(line,
             sizeof line,
             "    static constexpr ck_tile::index_t NumGroupsToMerge = %d;",
             num_groups_to_merge);
    L(out, line);
    snprintf(line, sizeof line, "    static constexpr int kBlockPerCu = %d;", block_per_cu);
    L(out, line);
    snprintf(line,
             sizeof line,
             "    static constexpr ck_tile::index_t VectorSizeA = %d;",
             vector_size_a);
    L(out, line);
    snprintf(line,
             sizeof line,
             "    static constexpr ck_tile::index_t VectorSizeB = %d;",
             vector_size_b);
    L(out, line);
    snprintf(line,
             sizeof line,
             "    static constexpr ck_tile::index_t VectorSizeC = %d;",
             vector_size_c);
    L(out, line);
    L(out, "");
    L(out, "    // Scheduler/Pipeline mirror the GEMM lowering's choices.");
    L(out, "    static constexpr auto Scheduler = ck_tile::GemmPipelineScheduler::Intrawave;");
    snprintf(line, sizeof line, "    static constexpr ck_tile::GemmPipeline Pipeline = %s;", pipe);
    L(out, line);
    L(out, "};");
    L(out, "");
    L(out, "// Conv spatial layouts: NHWGC input, GKYXC weights, NHWGK output, 2D spatial.");
    L(out, "// These are the only layout triples ``GroupedConvolutionForwardKernel``'s");
    L(out, "// internal ``MakeADescriptor_M_K`` / ``MakeBDescriptor_N_K`` /");
    L(out, "// ``MakeCDescriptor_M_N`` overloads accept for NDimSpatial==2 (see");
    L(out, "// ck_tile/ops/grouped_convolution/utils/transform_conv_fwd_to_gemm.hpp).");
    L(out, "static constexpr ck_tile::index_t NDimSpatial = 2;");
    snprintf(line, sizeof line, "using InDataType = %s;", in_dtype);
    L(out, line);
    snprintf(line, sizeof line, "using WeiDataType = %s;", wei_dtype);
    L(out, line);
    snprintf(line, sizeof line, "using AccDataType = %s;", acc_dtype);
    L(out, line);
    snprintf(line, sizeof line, "using OutDataType = %s;", out_dtype);
    L(out, line);
    L(out, "using InLayout = ck_tile::tensor_layout::convolution::NHWGC;");
    L(out, "using WeiLayout = ck_tile::tensor_layout::convolution::GKYXC;");
    L(out, "using OutLayout = ck_tile::tensor_layout::convolution::NHWGK;");
    L(out, "using DsDataType = ck_tile::tuple<>;");
    L(out, "using DsLayout = ck_tile::tuple<>;");
    L(out, "using CDElementWise = ck_tile::element_wise::PassThrough;");
    L(out, "");
    L(out, "// Implicit-GEMM tile shape -- same TileGemmShape as a pure GEMM, but");
    L(out, "// the M / N / K axes are the implicit-GEMM packings of the conv.");
    L(out, "using GemmShape = ck_tile::TileGemmShape<");
    L(out, "    ck_tile::sequence<ConvConfig::M_Tile, ConvConfig::N_Tile, ConvConfig::K_Tile>,");
    L(out, "    ck_tile::sequence<ConvConfig::M_Warp, ConvConfig::N_Warp, ConvConfig::K_Warp>,");
    L(out,
      "    ck_tile::sequence<ConvConfig::M_Warp_Tile, ConvConfig::N_Warp_Tile, "
      "ConvConfig::K_Warp_Tile>>;");
    L(out, "");
    L(out, "static constexpr auto ConvSpec = ck_tile::ConvolutionSpecialization::Default;");
    L(out, "using GroupedConvTraitsType = ck_tile::GroupedConvTraits<");
    L(out, "    NDimSpatial,");
    L(out, "    ConvSpec,");
    L(out, "    InLayout, WeiLayout, DsLayout, OutLayout,");
    L(out, "    ConvConfig::VectorSizeA, ConvConfig::VectorSizeB, ConvConfig::VectorSizeC,");
    L(out, "    ConvConfig::NumGroupsToMerge>;");
    L(out, "");
    L(out, "using TilePartitioner = ck_tile::GemmSpatiallyLocalTilePartitioner<");
    L(out, "    GemmShape,");
    L(out, "    GroupedConvTraitsType::FixedGemmParams::TilePartitionerGroupNum,");
    L(out, "    GroupedConvTraitsType::FixedGemmParams::TilePartitionerM01>;");
    L(out, "");
    L(out, "using GemmUniversalTraits = ck_tile::TileGemmUniversalTraits<");
    L(out, "    GroupedConvTraitsType::FixedGemmParams::kPadM,");
    L(out, "    GroupedConvTraitsType::FixedGemmParams::kPadN,");
    L(out, "    GroupedConvTraitsType::FixedGemmParams::kPadK,");
    L(out, "    ConvConfig::DoubleSmemBuffer,");
    L(out, "    typename GroupedConvTraitsType::AsLayoutFwd,");
    L(out, "    typename GroupedConvTraitsType::BsLayoutFwd,");
    L(out, "    typename GroupedConvTraitsType::CLayoutFwd,");
    L(out, "    GroupedConvTraitsType::FixedGemmParams::TransposeC,");
    L(out, "    GroupedConvTraitsType::FixedGemmParams::UseStructuredSparsity,");
    L(out, "    GroupedConvTraitsType::FixedGemmParams::Persistent,");
    L(out, "    ConvConfig::NumWaveGroups>;");
    L(out, "");
    L(out, "using UniversalGemmProblem = ck_tile::UniversalGemmPipelineProblem<");
    L(out, "    InDataType, WeiDataType, AccDataType,");
    L(out, "    GemmShape, GemmUniversalTraits, ConvConfig::Scheduler,");
    L(out, "    ck_tile::element_wise::PassThrough,");
    L(out, "    ck_tile::element_wise::PassThrough,");
    L(out, "    OutDataType,");
    L(out, "    GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,");
    L(out, "    GroupedConvTraitsType::VectorSizeA,");
    L(out, "    GroupedConvTraitsType::VectorSizeB>;");
    L(out, "");
    L(out, "using GemmPipeline =");
    L(out,
      "    typename ::PipelineTypeTraits<ConvConfig::Pipeline>::template "
      "GemmPipeline<UniversalGemmProblem>;");
    L(out, "");
    L(out, "using ConvEpilogue = ck_tile::CShuffleEpilogue<");
    L(out, "    ck_tile::CShuffleEpilogueProblem<");
    L(out, "        InDataType, WeiDataType, DsDataType, AccDataType, OutDataType,");
    L(out, "        typename GroupedConvTraitsType::ImplicitGemmDsLayout,");
    L(out, "        typename GroupedConvTraitsType::FixedGemmParams::ELayout,");
    L(out, "        CDElementWise,");
    L(out, "        TilePartitioner::MPerBlock, TilePartitioner::NPerBlock,");
    L(out, "        ConvConfig::M_Warp, ConvConfig::N_Warp,");
    L(out, "        ConvConfig::M_Warp_Tile, ConvConfig::N_Warp_Tile, ConvConfig::K_Warp_Tile,");
    L(out, "        GroupedConvTraitsType::FixedGemmParams::TransposeC,");
    L(out, "        ConvConfig::NumWaveGroups,");
    L(out, "        GroupedConvTraitsType::FixedGemmParams::FixedVectorSize,");
    L(out, "        GroupedConvTraitsType::VectorSizeC>>;");
    L(out, "");
    L(out, "// GroupedConvolutionForwardKernel takes 4 type parameters; the");
    L(out, "// NDimSpatial value lives inside GroupedConvTraitsType already.");
    L(out, "// Wiring exactly matches example/ck_tile/20_grouped_convolution/");
    L(out, "// grouped_convolution_forward_invoker.hpp.");
    L(out, "using Kernel = ck_tile::GroupedConvolutionForwardKernel<");
    L(out, "    GroupedConvTraitsType,");
    L(out, "    TilePartitioner,");
    L(out, "    GemmPipeline,");
    L(out, "    ConvEpilogue>;");
    L(out, "");
    snprintf(line, sizeof line, "} // namespace rocke_emit_%s", name);
    L(out, line);
    L(out, "");
    L(out, "// Host launcher: takes a ``GroupedConvFwdHostArgs<PassThrough>``");
    L(out, "// (the public host-side args struct in CK Tile) and forwards to");
    L(out, "// ``Kernel::MakeKernelArgs``, exactly as");
    L(out,
      "// ``example/ck_tile/20_grouped_convolution/grouped_convolution_forward_invoker.hpp`` "
      "does.");
    snprintf(line, sizeof line, "extern \"C\" float launch_%s(", name);
    L(out, line);
    L(out, "    const ck_tile::GroupedConvFwdHostArgs<ck_tile::element_wise::PassThrough>& args,");
    L(out, "    hipStream_t stream)");
    L(out, "{");
    snprintf(line, sizeof line, "    using namespace rocke_emit_%s;", name);
    L(out, line);
    L(out, "    ck_tile::stream_config s;");
    L(out, "    s.stream_id_ = stream;");
    L(out, "    s.time_kernel_ = false;");
    L(out, "");
    L(out, "    auto kargs = Kernel::MakeKernelArgs(args);");
    L(out, "    const dim3 grids = Kernel::GridSize(kargs);");
    L(out, "    const dim3 blocks = Kernel::BlockSize();");
    L(out, "    if (!Kernel::IsSupportedArgument(kargs)) { return -1.0f; }");
    L(out, "");
    L(out, "    return ck_tile::launch_kernel(");
    L(out, "        s,");
    L(out, "        ck_tile::make_kernel<ConvConfig::kBlockPerCu>(");
    L(out, "            Kernel{}, grids, blocks, 0, kargs));");
    L(out, "}");
    /* Trailing "" element: see note in the GEMM emitter. */

    if(out->oom)
    {
        rocke__err(err, "lower_implicit_gemm_conv_to_cktile: out of memory");
        return ROCKE_ERR_OOM;
    }
    return ROCKE_OK;
}

/* ================================================================ dispatch */

rocke_status_t rocke_lower_spec_to_cktile(const rocke_cktile_spec_t* spec,
                                          const char* kernel_name_override,
                                          rocke_strbuf_t* out,
                                          char err[ROCKE_ERR_MSG_CAP])
{
    if(err)
    {
        err[0] = '\0';
    }
    if(!spec || !out)
    {
        rocke__err(err, "lower_spec_to_cktile: NULL spec/out");
        return ROCKE_ERR_VALUE;
    }
    switch(spec->kind)
    {
    case ROCKE_CKTILE_SPEC_GEMM:
        return rocke_lower_universal_gemm_to_cktile(spec->u.gemm, kernel_name_override, out, err);
    case ROCKE_CKTILE_SPEC_CONV:
        return rocke_lower_implicit_gemm_conv_to_cktile(
            spec->u.conv, kernel_name_override, out, err);
    default:
        rocke__err(err,
                   "no CK Tile lowering for spec kind %d; supported: "
                   "UniversalGemmSpec, ImplicitGemmConvSpec",
                   (int)spec->kind);
        return ROCKE_ERR_NOTIMPL;
    }
}
