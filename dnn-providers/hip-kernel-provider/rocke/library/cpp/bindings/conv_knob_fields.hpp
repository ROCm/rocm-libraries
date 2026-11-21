// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * library/cpp/bindings/conv_knob_fields.hpp -- the K-loop / epilogue knob fields of the
 * implicit-GEMM conv specs (fwd, dgrad, wgrad) and of the wgrad Stage 2 fold that the
 * rocke_library_engine spec builders read from the Python dict by name.
 *
 * Each list is an X-macro: X_INT(field) for an int field, X_BOOL(field) for a
 * bool field. The field name is the C struct member and the Python dataclass
 * field alike. A field absent from the dict keeps the engine default (the
 * dataclass default), so a dict without knobs builds the same spec as before.
 */
#ifndef ROCKE_BINDINGS_CONV_KNOB_FIELDS_HPP
#define ROCKE_BINDINGS_CONV_KNOB_FIELDS_HPP

/* ImplicitGemmConvSpec knobs. */
#define ROCKE_CONV_FWD_KNOB_FIELDS(X_INT, X_BOOL) \
    X_INT(xcd_remap)                              \
    X_INT(m_band)                                 \
    X_INT(reg_prefetch)                           \
    X_BOOL(lds_xor)                               \
    X_BOOL(lds_kpack_read)                        \
    X_BOOL(lds_xor_rowkey)                        \
    X_BOOL(uniform_k_decode)                      \
    X_BOOL(uk_tap_fast)                           \
    X_BOOL(uk_nopad)                              \
    X_INT(uk_chunk_group)                         \
    X_BOOL(fb_skip_empty)                         \
    X_INT(vmem_interleave)                        \
    X_BOOL(split_atomic_agent)                    \
    X_BOOL(split_atomic_fast)                     \
    X_BOOL(epi_store_fast)                        \
    X_BOOL(split_flag_store)                      \
    X_BOOL(split_zero_first)                      \
    X_BOOL(split_tail_compact)                    \
    X_BOOL(lds_ahead)                             \
    X_INT(defer_mfma)                             \
    X_INT(defer_mfma_per_write)                   \
    X_INT(rpf_setprio)                            \
    X_INT(rpf_pingpong)                           \
    X_INT(pp_pre_mfma)                            \
    X_BOOL(pp_b_double)                           \
    X_INT(pp_split_fetch)                         \
    X_BOOL(pp_end_guard)                          \
    X_BOOL(split_addr_k_decode)                   \
    X_BOOL(split_tail)                            \
    X_BOOL(split_atomic)                          \
    X_BOOL(split_atomic_pairs)                    \
    X_INT(split_det)                              \
    X_INT(cshuffle_m_chunks)                      \
    X_BOOL(epi_lds_barriers)                      \
    X_INT(split_xcd_remap)                        \
    X_BOOL(split_streamk)                         \
    X_INT(split_hybrid)                           \
    X_INT(uk_xhalo)                               \
    X_BOOL(xh_wait_smem)                          \
    X_INT(epi_quad_xpose)                         \
    X_BOOL(no_agpr)                               \
    X_BOOL(xh_stride_m)                           \
    X_INT(fxh_dma)                                \
    X_INT(xh_yring)                               \
    X_BOOL(xh_pad)                                \
    X_BOOL(rpf_lds_barrier)                       \
    X_BOOL(rpf_fetch_late)                        \
    X_INT(sched_strategy)

/* DgradConvSpec knobs. */
#define ROCKE_CONV_DGRAD_KNOB_FIELDS(X_INT, X_BOOL) \
    X_INT(tapk_prefetch)                            \
    X_BOOL(tapk_xhalo)                              \
    X_INT(b_kpack)                                  \
    X_INT(tapk_ilv)                                 \
    X_BOOL(xhalo_exact)                             \
    X_INT(xcd_remap)                                \
    X_INT(m_band)                                   \
    X_BOOL(xh_stride_m)                             \
    X_INT(xh_dlds)                                  \
    X_BOOL(xh_epi_lin)                              \
    X_BOOL(epi_lds_barriers)                        \
    X_BOOL(xh_tap_rsrc)                             \
    X_INT(xh_yring)                                 \
    X_BOOL(no_agpr)                                 \
    X_INT(cshuffle_m_chunks)                        \
    X_INT(tapk_lds_ahead)                           \
    X_BOOL(tapk_lds_barrier)                        \
    X_INT(tapk_setprio)                             \
    X_BOOL(lds_xor)                                 \
    X_BOOL(xh_b_wide)                               \
    X_INT(xh_b_kpad)                                \
    X_INT(tapk_pingpong)                            \
    X_INT(tapk_pp_sched)                            \
    X_INT(tapk_pp_epi)                              \
    X_BOOL(split_atomic_pairs)                      \
    X_INT(tapk_pp_persist)                          \
    X_INT(tapk_pp_xhalo)                            \
    X_INT(tapk_ztrim)                               \
    X_INT(tapk_pp_bslab)                            \
    X_BOOL(xh_a_k4)                                 \
    X_INT(xh_order)                                 \
    X_INT(xh_defer)                                 \
    X_INT(xh_fast_a)                                \
    X_INT(xh_epi)                                   \
    X_INT(xh_persist)                               \
    X_INT(pp_streamk)                               \
    X_INT(pp_sk_mode)                               \
    X_BOOL(pp_zskip)                                \
    X_INT(pp_boxes)                                 \
    X_INT(pp_box_cus)                               \
    X_INT(pp_box_order)                             \
    X_INT(pp_box_det)                               \
    X_BOOL(xh_dx_wmask)

/* WgradConvSpec knobs. */
#define ROCKE_CONV_WGRAD_KNOB_FIELDS(X_INT, X_BOOL) \
    X_INT(prefetch_regs)                            \
    X_INT(lds_kpack)                                \
    X_INT(xcd_remap)                                \
    X_INT(m_band)                                   \
    X_INT(sep_addr)                                 \
    X_INT(pfr_sched)                                \
    X_INT(lds_bswz)                                 \
    X_INT(kp_line)                                  \
    X_INT(xwin)                                     \
    X_INT(xwin_rows)                                \
    X_INT(xwin_groups)                              \
    X_INT(xwin_mrow)                                \
    X_INT(xwin_gn)                                  \
    X_INT(xwin_pp)                                  \
    X_INT(ws_fuse)                                  \
    X_INT(ws_store)                                 \
    X_INT(ws_tree)                                  \
    X_INT(epi_quad_xpose)                           \
    X_BOOL(epi_store_fast)                          \
    X_INT(xwin_kband)                               \
    X_INT(plain_pp)                                 \
    X_INT(xwin_bq)                                  \
    X_INT(xwin_aq)                                  \
    X_INT(xwin_pps)                                 \
    X_INT(xwin_rot)                                 \
    X_INT(xwin_kord)                                \
    X_INT(ws_lc)                                    \
    X_INT(ws_gred)

/* WgradReduceSpec (the two-stage wgrad Stage 2 fold) knobs. */
#define ROCKE_CONV_WGRAD_REDUCE_KNOB_FIELDS(X_INT, X_BOOL) \
    X_INT(vec)                                             \
    X_INT(rezero)                                          \
    X_BOOL(ws_bf16)                                        \
    X_INT(fold_loop)                                       \
    X_INT(slab_par)

#endif /* ROCKE_BINDINGS_CONV_KNOB_FIELDS_HPP */
