/* Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 * SPDX-License-Identifier: MIT
 *
 * rocke/helper_rocke.instances.common._attention_bwd_tiles.h -- C port of the
 * small-tile warp grids and the occupancy bound of the attention backward
 * (library/kernels/common/_attention_bwd_tiles.py). Host-side decisions only;
 * nothing here emits IR.
 *
 *   Python (_attention_bwd_tiles.py)    C (this header)
 *   ---------------------------------   ---------------------------------------
 *   (rows, cols) grid tuple             rocke_attn_bwd_grid_t
 *   grid_waves                          rocke_attn_bwd_grid_waves
 *   natural_grids                       rocke_attn_bwd_natural_grids
 *   g02_grids / g13_grids               rocke_attn_bwd_g02_grids / _g13_grids
 *   replicated_g02                      rocke_attn_bwd_replicated_g02
 *   check_lds_route_grid                rocke_attn_bwd_check_lds_route_grid
 *   kv_operand_split                    rocke_attn_bwd_kv_operand_split
 *   small_tile_grids                    rocke_attn_bwd_small_tile_grids
 *   AttnBwdOccupancy                    rocke_attn_bwd_occupancy_t
 *     .waves / .hint_reachable          rocke_attn_bwd_occupancy_waves / _hint_reachable
 *   attn_bwd_occupancy                  rocke_attn_bwd_occupancy
 *
 * Grid lists are written to caller arrays of ROCKE_ATTN_BWD_GRIDS_MAX (g02 /
 * g13) or ROCKE_ATTN_BWD_TRIPLES_MAX (small_tile_grids) entries: Python's
 * lists are bounded by its fixed wave counts (1, 2, 4, 8), and a longer list
 * (a wave count beyond them) is refused with ROCKE_ERR_VALUE.
 *
 * Everything in _attention_bwd_tiles.py is ported.
 */
#ifndef ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_TILES_H
#define ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_TILES_H

#include <stdbool.h>

#include "rocke/helper_rocke.instances.common._attention_bwd_caps.h" /* rocke_attn_bwd_err_t */

#ifdef __cplusplus
extern "C" {
#endif

/* A warp grid (rows, cols). */
typedef struct rocke_attn_bwd_grid
{
    int rows;
    int cols;
} rocke_attn_bwd_grid_t;

#define ROCKE_ATTN_BWD_GRIDS_MAX 16
#define ROCKE_ATTN_BWD_TRIPLES_MAX 64

int rocke_attn_bwd_grid_waves(rocke_attn_bwd_grid_t grid);

/* natural_grids(waves): (1, W) and (W, 1). */
void rocke_attn_bwd_natural_grids(int waves,
                                  rocke_attn_bwd_grid_t* g02,
                                  rocke_attn_bwd_grid_t* g13);

/* g02_grids / g13_grids: count written to *n_out. */
rocke_status_t rocke_attn_bwd_g02_grids(int waves,
                                        rocke_attn_bwd_grid_t out[ROCKE_ATTN_BWD_GRIDS_MAX],
                                        int* n_out,
                                        rocke_attn_bwd_err_t* err);
rocke_status_t rocke_attn_bwd_g13_grids(int waves,
                                        rocke_attn_bwd_grid_t out[ROCKE_ATTN_BWD_GRIDS_MAX],
                                        int* n_out,
                                        rocke_attn_bwd_err_t* err);

bool rocke_attn_bwd_replicated_g02(rocke_attn_bwd_grid_t grid, int waves);

/* check_lds_route_grid(gname, grid, natural, waves, rows, cols, pt_route). */
rocke_status_t rocke_attn_bwd_check_lds_route_grid(const char* gname,
                                                   rocke_attn_bwd_grid_t grid,
                                                   rocke_attn_bwd_grid_t natural,
                                                   int waves,
                                                   int rows,
                                                   int cols,
                                                   const char* pt_route,
                                                   rocke_attn_bwd_err_t* err);

int rocke_attn_bwd_kv_operand_split(rocke_attn_bwd_grid_t grid_g02, int waves);

/* One (G0 / G2, G1 / G3, G4) grid triple. */
typedef struct rocke_attn_bwd_grid_triple
{
    rocke_attn_bwd_grid_t g02, g13, g4;
} rocke_attn_bwd_grid_triple_t;

/* small_tile_grids(head_size, block_m, block_n, waves). */
rocke_status_t
    rocke_attn_bwd_small_tile_grids(int head_size,
                                    int block_m,
                                    int block_n,
                                    int waves,
                                    rocke_attn_bwd_grid_triple_t out[ROCKE_ATTN_BWD_TRIPLES_MAX],
                                    int* n_out,
                                    rocke_attn_bwd_err_t* err);

/* AttnBwdOccupancy; requested is the spec's waves_per_eu, or
 * ROCKE_ATTN_BWD_NONE for None. */
typedef struct rocke_attn_bwd_occupancy
{
    int by_registers;
    int by_lds;
    int requested;
} rocke_attn_bwd_occupancy_t;

int rocke_attn_bwd_occupancy_waves(const rocke_attn_bwd_occupancy_t* occ);
bool rocke_attn_bwd_occupancy_hint_reachable(const rocke_attn_bwd_occupancy_t* occ);

struct rocke_attn_bwd_spec; /* instance_attention_bwd.h */

/* attn_bwd_occupancy(spec, arch) of a resolved spec. */
rocke_status_t rocke_attn_bwd_occupancy(const struct rocke_attn_bwd_spec* spec,
                                        const char* arch,
                                        rocke_attn_bwd_occupancy_t* out,
                                        rocke_attn_bwd_err_t* err);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <string>
#include <vector>

namespace rocke_attn_bwd
{
using Grid = rocke_attn_bwd_grid_t;
inline bool operator==(const Grid& a, const Grid& b)
{
    return a.rows == b.rows && a.cols == b.cols;
}
std::string grid_repr(const Grid& g);
std::string grids_repr(const std::vector<Grid>& gs);
int grid_waves(const Grid& g);
std::vector<Grid> g02_grids(int waves);
std::vector<Grid> g13_grids(int waves);
void check_lds_route_grid(const std::string& gname,
                          const Grid& grid,
                          const Grid& natural,
                          int waves,
                          int rows,
                          int cols,
                          const std::string& pt_route);
int kv_operand_split(const Grid& g02, int waves);
std::vector<rocke_attn_bwd_grid_triple_t>
    small_tile_grids(int head_size, int block_m, int block_n, int waves);
} // namespace rocke_attn_bwd
#endif

#endif /* ROCKE_HELPER_ROCKE_INSTANCES_COMMON__ATTENTION_BWD_TILES_H */
