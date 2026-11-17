// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * tests/instances/wgrad_streamk_host.cpp -- C parity for the stream-K wgrad
 * host helpers and validator gates of instance_conv_implicit_gemm_wgrad.h.
 *
 * The launch grid, workspace layout and auto CTA pool these helpers compute
 * never reach the emitted IR, so the byte-identity gate cannot see a C/Python
 * divergence in them; it would surface as skipped work or an undersized
 * buffer. The Python-vs-C pytest (library/tests/test_conv_wgrad_streamk.py)
 * needs the rocke_engine binding, which CI does not build, so the vectors are
 * pinned here as well.
 *
 * Every expected value below was produced by the Python implementation
 * (kernels.common.conv_implicit_gemm_wgrad: wgrad_streamk_default_pool,
 * wgrad_streamk_launch(...).pool, wgrad_streamk_partition, wgrad_streamk_grid,
 * wgrad_streamk_workspace_layout, wgrad_streamk_workspace_nbytes and
 * is_valid_wgrad_spec) for the same specs; regenerate them from there if the
 * Python reference changes.
 */
#include <cstdio>
#include <cstring>

#include "rocke/instance_conv_implicit_gemm_wgrad.h"

namespace
{

int g_failures = 0;

#define CHECK(cond)                                                           \
    do                                                                        \
    {                                                                         \
        if(!(cond))                                                           \
        {                                                                     \
            fprintf(stderr, "FAIL: %s (%s:%d)\n", #cond, __FILE__, __LINE__); \
            ++g_failures;                                                     \
        }                                                                     \
    } while(0)

/* ConvProblem(N=2, Hi=12, Wi=12, C=32, K=96, Y=3, X=3, pH=1, pW=1): 10 tiles. */
rocke_conv_problem_t dense()
{
    rocke_conv_problem_t p = rocke_conv_problem_default(2, 12, 12, 32, 96, 3, 3);
    p.pH = 1;
    p.pW = 1;
    return p;
}

/* ConvProblem(N=2, Hi=12, Wi=12, C=192, K=1056, Y=3, X=3, pH=1, pW=1):
 * 17 x 27 = 459 tiles, more than either arch's default pool, so an auto-pool
 * launch splits. */
rocke_conv_problem_t splits()
{
    rocke_conv_problem_t p = rocke_conv_problem_default(2, 12, 12, 192, 1056, 3, 3);
    p.pH = 1;
    p.pW = 1;
    return p;
}

/* The library test's _spec(): 64x64x32 tiles, 2x2 warps, 32x32x16 atoms,
 * dp_sk, linear, streamk_ctas=6. */
rocke_implicit_gemm_conv_wgrad_spec_t spec(const rocke_conv_problem_t& p)
{
    rocke_implicit_gemm_conv_wgrad_spec_t s = rocke_implicit_gemm_conv_wgrad_spec_default();
    s.problem = p;
    s.tile_m = 64;
    s.tile_n = 64;
    s.tile_k = 32;
    s.warp_m = 2;
    s.warp_n = 2;
    s.warp_tile_m = 32;
    s.warp_tile_n = 32;
    s.warp_tile_k = 16;
    s.streamk = "dp_sk";
    s.streamk_ctas = 6;
    return s;
}

void check_default_pool()
{
    static const struct
    {
        const char* arch;
        int linear, atomic;
    } k_pools[] = {
        {"gfx950", 256, 1024},
        {"gfx942", 304, 1216},
        {"gfx1151", 64, 256},
        {"gfx90a", 256, 1024}, /* not in the CU table: 256-CU fallback */
        {"gfx9999", 256, 1024},
        {NULL, 256, 1024}, /* NULL arch is gfx950 */
    };
    for(const auto& e : k_pools)
    {
        CHECK(rocke_wgrad_conv_streamk_default_pool("linear", e.arch) == e.linear);
        CHECK(rocke_wgrad_conv_streamk_default_pool("tree", e.arch) == e.linear);
        CHECK(rocke_wgrad_conv_streamk_default_pool("atomic", e.arch) == e.atomic);
        CHECK(rocke_wgrad_conv_streamk_default_pool("workspace", e.arch) == e.atomic);
    }
    CHECK(rocke_wgrad_conv_streamk_default_pool(NULL, "gfx942") == 304); /* linear */
}

void check_ctas()
{
    rocke_implicit_gemm_conv_wgrad_spec_t s = spec(dense());
    CHECK(rocke_wgrad_conv_streamk_ctas(&s, "gfx942") == 6);
    s.streamk_ctas = -1;
    CHECK(rocke_wgrad_conv_streamk_ctas(&s, NULL) == 256);
    CHECK(rocke_wgrad_conv_streamk_ctas(&s, "gfx950") == 256);
    CHECK(rocke_wgrad_conv_streamk_ctas(&s, "gfx942") == 304);
    s.streamk_reduction = "workspace";
    CHECK(rocke_wgrad_conv_streamk_ctas(&s, "gfx942") == 1216);
    /* Python wgrad_streamk_launch raises for these; C reports no pool. */
    s.streamk_ctas = 0;
    CHECK(rocke_wgrad_conv_streamk_ctas(&s, "gfx950") == 0);
    s.streamk_ctas = -3;
    CHECK(rocke_wgrad_conv_streamk_ctas(&s, "gfx950") == 0);
    s.streamk_ctas = 6;
    s.streamk = "off";
    CHECK(rocke_wgrad_conv_streamk_ctas(&s, "gfx950") == 0);
    CHECK(rocke_wgrad_conv_streamk_ctas(NULL, "gfx950") == 0);
}

struct PlanVector
{
    const char* arch;
    const char* mode;
    int ctas;
    int sk_tiles, sk_ctas, dp_tiles, grid, ipsc, extra;
    size_t flags_bytes, partials_bytes;
};

void check_plans()
{
    /* linear reduction on the 459-tile problem. */
    static const PlanVector k_plans[] = {
        {"gfx950", "dp_sk", -1, 459, 256, 0, 256, 16, 35, 4096, 4194304},
        {"gfx950", "persistent", -1, 459, 256, 0, 256, 16, 35, 4096, 4194304},
        {"gfx942", "dp_sk", -1, 459, 304, 0, 304, 13, 179, 4864, 4980736},
        {"gfx942", "persistent", -1, 459, 304, 0, 304, 13, 179, 4864, 4980736},
        {"gfx950", "dp_sk", 7, 11, 7, 448, 455, 14, 1, 256, 114688},
        {"gfx942", "persistent", 7, 11, 7, 448, 7, 14, 1, 256, 114688},
        {NULL, "dp_sk", -1, 459, 256, 0, 256, 16, 35, 4096, 4194304},
    };
    for(const PlanVector& v : k_plans)
    {
        rocke_implicit_gemm_conv_wgrad_spec_t s = spec(splits());
        s.streamk = v.mode;
        s.streamk_ctas = v.ctas;
        rocke_status_t st = ROCKE_OK;
        const rocke_streamk_iter_partition_t part
            = rocke_wgrad_conv_streamk_partition(&s, v.arch, &st);
        CHECK(st == ROCKE_OK);
        CHECK(part.m_tiles == 17);
        CHECK(part.n_tiles == 27);
        CHECK(part.iters_per_tile == 9);
        const rocke_streamk_iter_plan_t r = rocke_streamk_iter_plan(&part);
        CHECK(r.sk_tiles == v.sk_tiles);
        CHECK(r.sk_ctas == v.sk_ctas);
        CHECK(r.dp_tiles == v.dp_tiles);
        CHECK(r.grid_size == v.grid);
        CHECK(r.iters_per_sk_cta == v.ipsc);
        CHECK(r.extra_iters == v.extra);

        int grid[3] = {0, 0, 0};
        CHECK(rocke_wgrad_conv_streamk_grid(&s, v.arch, grid) == ROCKE_OK);
        CHECK(grid[0] == v.grid && grid[1] == 1 && grid[2] == 1);

        size_t flags = 1, partials = 1;
        CHECK(rocke_wgrad_conv_streamk_workspace_layout(&s, v.arch, &flags, &partials) == ROCKE_OK);
        CHECK(flags == v.flags_bytes);
        CHECK(partials == v.partials_bytes);
        CHECK(rocke_wgrad_conv_streamk_workspace_bytes(&s, v.arch)
              == v.flags_bytes + v.partials_bytes);
    }

    /* The other reductions on the same shape, auto pool. */
    static const struct
    {
        const char* arch;
        const char* reduction;
        const char* dtype_d;
        int grid;
        size_t workspace_bytes;
    } k_reductions[] = {
        {"gfx950", "tree", "fp16", 256, 4198400},
        {"gfx942", "tree", "fp16", 304, 4985600},
        {"gfx950", "atomic", "fp32", 1024, 0},
        {"gfx942", "atomic", "fp32", 1216, 0},
        {"gfx950", "workspace", "fp16", 1024, 58392576},
        {"gfx942", "workspace", "fp16", 1216, 58392576},
    };
    for(const auto& e : k_reductions)
    {
        rocke_implicit_gemm_conv_wgrad_spec_t s = spec(splits());
        s.streamk_ctas = -1;
        s.streamk_reduction = e.reduction;
        s.dtype_d = e.dtype_d;
        int grid[3] = {0, 0, 0};
        CHECK(rocke_wgrad_conv_streamk_grid(&s, e.arch, grid) == ROCKE_OK);
        CHECK(grid[0] == e.grid);
        size_t flags = 1, partials = 1;
        CHECK(rocke_wgrad_conv_streamk_workspace_layout(&s, e.arch, &flags, &partials) == ROCKE_OK);
        const bool fixup = strcmp(e.reduction, "tree") == 0;
        CHECK(fixup ? flags + partials == e.workspace_bytes : flags == 0 && partials == 0);
        CHECK(rocke_wgrad_conv_streamk_workspace_bytes(&s, e.arch) == e.workspace_bytes);
        CHECK(rocke_wgrad_conv_spec_is_deterministic(&s) == fixup);
    }

    /* 10 tiles over a pool of 5: all data-parallel, no flags or slots. */
    rocke_implicit_gemm_conv_wgrad_spec_t s = spec(dense());
    s.streamk_ctas = 5;
    size_t flags = 1, partials = 1;
    CHECK(rocke_wgrad_conv_streamk_workspace_layout(&s, "gfx950", &flags, &partials) == ROCKE_OK);
    CHECK(flags == 0 && partials == 0);
    /* 10 tiles over 7: 256 flag bytes, 7 partial tiles. */
    s.streamk_ctas = 7;
    CHECK(rocke_wgrad_conv_streamk_workspace_layout(&s, "gfx950", &flags, &partials) == ROCKE_OK);
    CHECK(flags == 256 && partials == 7 * 64 * 64 * 4);
}

void check_invalid_inputs()
{
    rocke_status_t st = ROCKE_OK;
    int grid[3] = {0, 0, 0};
    size_t flags = 0, partials = 0;

    rocke_streamk_iter_partition_t part = rocke_wgrad_conv_streamk_partition(NULL, NULL, &st);
    CHECK(st == ROCKE_ERR_VALUE);
    CHECK(part.m_tiles == 0 && part.max_active_wgs == 0);
    CHECK(rocke_wgrad_conv_streamk_grid(NULL, NULL, grid) == ROCKE_ERR_VALUE);
    CHECK(rocke_wgrad_conv_streamk_workspace_layout(NULL, NULL, &flags, &partials)
          == ROCKE_ERR_VALUE);
    CHECK(rocke_wgrad_conv_streamk_workspace_bytes(NULL, NULL) == 0);

    rocke_implicit_gemm_conv_wgrad_spec_t s = spec(dense());
    CHECK(rocke_wgrad_conv_streamk_grid(&s, NULL, NULL) == ROCKE_ERR_VALUE);
    CHECK(rocke_wgrad_conv_streamk_workspace_layout(&s, NULL, NULL, &partials) == ROCKE_ERR_VALUE);
    CHECK(rocke_wgrad_conv_streamk_workspace_layout(&s, NULL, &flags, NULL) == ROCKE_ERR_VALUE);

    const int bad_ctas[] = {0, -2, -3};
    for(int c : bad_ctas)
    {
        s.streamk_ctas = c;
        st = ROCKE_OK;
        (void)rocke_wgrad_conv_streamk_partition(&s, "gfx950", &st);
        CHECK(st == ROCKE_ERR_VALUE);
        CHECK(rocke_wgrad_conv_streamk_grid(&s, "gfx950", grid) == ROCKE_ERR_VALUE);
        CHECK(rocke_wgrad_conv_streamk_workspace_layout(&s, "gfx950", &flags, &partials)
              == ROCKE_ERR_VALUE);
        CHECK(rocke_wgrad_conv_streamk_workspace_bytes(&s, "gfx950") == 0);
    }
    s.streamk_ctas = 6;
    s.streamk = "off";
    st = ROCKE_OK;
    (void)rocke_wgrad_conv_streamk_partition(&s, "gfx950", &st);
    CHECK(st == ROCKE_ERR_VALUE);
    CHECK(rocke_wgrad_conv_streamk_workspace_bytes(&s, "gfx950") == 0);
}

/* is_valid_wgrad_spec accept/reject reasons, verbatim from Python. */
void check_validator()
{
    char why[1024];
    auto valid = [&](const rocke_implicit_gemm_conv_wgrad_spec_t& s, const char* arch) {
        why[0] = '\0';
        return rocke_implicit_gemm_conv_wgrad_is_valid_spec(&s, arch, why, sizeof(why));
    };
    auto rejects =
        [&](const rocke_implicit_gemm_conv_wgrad_spec_t& s, const char* arch, const char* reason) {
            const bool ok = valid(s, arch);
            if(ok || strcmp(why, reason) != 0)
            {
                fprintf(stderr, "  want: %s\n  got:  %s\n", reason, ok ? "(accepted)" : why);
                return false;
            }
            return true;
        };

    rocke_implicit_gemm_conv_wgrad_spec_t s = spec(dense());
    CHECK(valid(s, "gfx950"));
    CHECK(valid(s, NULL));
    s.streamk_ctas = -1;
    CHECK(valid(s, "gfx950"));
    s.streamk = "persistent";
    s.streamk_reduction = "tree";
    CHECK(valid(s, "gfx950"));
    s = spec(dense());
    s.warp_tile_m = 16;
    s.warp_tile_n = 16;
    CHECK(valid(s, "gfx942"));
    s = spec(dense());
    s.dtype_d = "fp32";
    s.streamk_reduction = "atomic";
    CHECK(valid(s, "gfx950"));

    s = spec(dense());
    s.streamk_ctas = 0;
    CHECK(rejects(s, "gfx950", "streamk_ctas must be -1 (auto) or > 0 (got 0)"));
    s.streamk_ctas = -3;
    CHECK(rejects(s, "gfx950", "streamk_ctas must be -1 (auto) or > 0 (got -3)"));

    s = spec(dense());
    s.streamk = "sideways";
    CHECK(rejects(s, "gfx950", "streamk must be 'off', 'dp_sk' or 'persistent' (got 'sideways')"));
    s = spec(dense());
    s.streamk_reduction = "ring";
    CHECK(rejects(s,
                  "gfx950",
                  "streamk_reduction must be 'linear', 'tree', 'atomic' or 'workspace' "
                  "(got 'ring')"));
    s = spec(dense());
    s.split_k = 4;
    CHECK(rejects(s,
                  "gfx950",
                  "streamk and split_k both partition K_wg; use split_k=1 with streamk "
                  "(got split_k=4)"));
    s = spec(dense());
    s.async_dma = true;
    CHECK(rejects(s,
                  "gfx950",
                  "streamk runs one plain K loop per work item; async_dma/unroll_k "
                  "pipeline the K loop across items"));
    s = spec(dense());
    s.streamk_reduction = "atomic";
    CHECK(rejects(s,
                  "gfx950",
                  "streamk_reduction='atomic' needs dtype_d='fp32' (got 'fp16'); use "
                  "'linear', 'tree' or 'workspace'"));
    s = spec(dense());
    CHECK(rejects(s, "gfx90a", "stream-K wgrad supports only gfx942, gfx950 (got gfx90a)"));

    /* Multi-cause rejects: the first gate in Python's order wins. */
    s = spec(dense());
    s.streamk_ctas = 0;
    s.tile_m = 96;
    CHECK(rejects(s, "gfx950", "tile_m not divisible by warp_m * warp_tile_m"));
    s = spec(dense());
    s.warp_m = 8;
    s.warp_n = 8;
    s.tile_m = 256;
    s.tile_n = 256;
    s.streamk_ctas = 0;
    CHECK(rejects(s, "gfx950", "block_size 4096 > 1024 (hardware cap) on gfx950"));
    s = spec(dense());
    s.wave_size = 32;
    s.streamk_ctas = 0;
    CHECK(rejects(s, "gfx950", "spec wave_size 32 != gfx950 wave_size 64"));
    s = spec(dense());
    CHECK(rejects(s, "gfx1250", "spec wave_size 64 != gfx1250 wave_size 32"));
    s = spec(dense());
    s.has_vector_size_c = true;
    s.vector_size_c = 4;
    s.streamk_ctas = -5;
    CHECK(rejects(s, "gfx950", "default epilogue is not supported with vector size c: 4"));
    s = spec(dense());
    s.async_dma = true;
    s.streamk = "off";
    CHECK(rejects(s,
                  "gfx950",
                  "wgrad async_dma requires lds_k_outer=True: the direct global->LDS load "
                  "needs a stride-1 reduction axis, which wgrad only has once the tile is "
                  "stored K-outer"));
    s = spec(dense());
    s.tile_m = 96;
    CHECK(rejects(s,
                  "gfx9999",
                  "\"unknown gfx target 'gfx9999'; known: ['gfx11-generic', 'gfx1151', "
                  "'gfx12-generic', 'gfx1201', 'gfx1250', 'gfx90a', 'gfx942', 'gfx950']. "
                  "Add a row to "
                  "arch_specs.json.\""));

    /* Depthwise ConvProblem(N=2, Hi=8, Wi=8, C=16, K=16, Y=3, X=3, pH=1, pW=1,
     * groups=16): the shape group merging admits. */
    rocke_conv_problem_t dw = rocke_conv_problem_default(2, 8, 8, 16, 16, 3, 3);
    dw.pH = 1;
    dw.pW = 1;
    dw.groups = 16;
    s = spec(dw);
    s.group_merge = 3;
    CHECK(rejects(s, "gfx950", "group_merge must be 1 or one of (2, 4, 8, 16, 32, 64); got 3"));
    s.group_merge = 2;
    CHECK(rejects(s, "gfx950", "streamk folds conv groups into GEMM-M; use group_merge=1 (got 2)"));
    s.tile_m = 96;
    CHECK(rejects(s, "gfx950", "tile_m not divisible by warp_m * warp_tile_m"));
}

} // namespace

int main()
{
    check_default_pool();
    check_ctas();
    check_plans();
    check_invalid_inputs();
    check_validator();
    if(g_failures)
    {
        fprintf(stderr, "rocke_wgrad_streamk_host: %d failure(s)\n", g_failures);
        return 1;
    }
    printf("rocke_wgrad_streamk_host: OK\n");
    return 0;
}
