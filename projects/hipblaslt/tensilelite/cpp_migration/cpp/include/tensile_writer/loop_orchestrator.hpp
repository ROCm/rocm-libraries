// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// C++ port of the structural loop-emission layer from LogicalScheduler.py
// and the VGPR tile zero-init utility from Kernel.py:
//
//   emit_loop                 (LogicalScheduler._emitLoop)
//   emit_main_and_exit_loops  (LogicalScheduler.emitMainAndExitLoops)
//   emit_tail_loop            (LogicalScheduler.emitTailLoop — structural part)
//   init_vgpr_tiles_to_zero   (Kernel.initVgprTilesToZero + _zeroRegRange)
//
// Boundary contract:
//   The orchestrator owns NO writer state. Per-leaf instruction emission
//   stays in Python: the two callables (emit_fn, schedule_fn) are passed in
//   from the Python LogicalScheduler wrapper and are never stored longer than
//   one emit call. All rocisa object construction goes through ModuleBuilder.
//
// Dependency: rocisa_module_builder.hpp (for ModuleBuilder).
//
// This header is intentionally header-only (inline implementations) to match
// the build pattern of the other tensile_writer C++ headers.

#pragma once

#include "rocisa_module_builder.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace tw::subtile::loop_orch {

namespace nb = nanobind;
using namespace nb::literals;

// ═══════════════════════════════════════════════════════════════════════
// LoopOrchestrator
//
// Caches the rocisa instruction class handles needed for loop control-flow
// (scalar branches, counters) and delegates per-leaf emission to the two
// Python callables supplied at construction:
//
//   emit_fn    : callable(em: EmittedModule, unroll_iter: int) -> list[Item]
//                (InstructionEmitter.emit_module)
//   schedule_fn: callable(em_list, inst_lists) -> Module
//                (instructionScheduleFromLists from InstructionScheduler.py)
//
// For init_vgpr_tiles_to_zero (no emit/schedule callables needed), pass
// nb::none() for both callables.
// ═══════════════════════════════════════════════════════════════════════
class LoopOrchestrator {
 public:
  LoopOrchestrator(rocisa_builder::ModuleBuilder& builder,
                   nb::object emit_fn,
                   nb::object schedule_fn)
      : builder_(builder), emit_fn_(emit_fn), schedule_fn_(schedule_fn) {
    nb::module_ inst = nb::module_::import_("rocisa.instruction");
    nb::module_ container = nb::module_::import_("rocisa.container");

    sgpr_fn_          = container.attr("sgpr");
    vgpr_fn_          = container.attr("vgpr");
    accvgpr_fn_       = container.attr("accvgpr");

    ssub_u32_cls_     = inst.attr("SSubU32");
    scmp_eq_u32_cls_  = inst.attr("SCmpEQU32");
    scbranch_scc0_    = inst.attr("SCBranchSCC0");
    scbranch_scc1_    = inst.attr("SCBranchSCC1");
    sbranch_cls_      = inst.attr("SBranch");

    // For init_vgpr_tiles_to_zero
    vmov_b64_cls_     = inst.attr("VMovB64");
    vmov_b32_cls_     = inst.attr("VMovB32");
    vaccvgpr_write_   = inst.attr("VAccvgprWrite");
    snop_cls_         = inst.attr("SNop");
    mfma_inst_cls_    = inst.attr("MFMAInstruction");
    inst_type_enum_   = nb::module_::import_("rocisa.enum").attr("InstType");
  }

  // ── emit_loop ──────────────────────────────────────────────────────────
  // Port of LogicalScheduler._emitLoop.
  //
  // emitted_3d: Python list[list[list[EmittedModule]]]
  //   outer = partitions, middle = subIterK groups, inner = EmittedModules
  //
  // Returns a rocisa Module.
  nb::object emit_loop(nb::object emitted_3d, const std::string& label,
                       int unroll_iter = 0, bool schedule = true) const {
    nb::object mod = builder_.module(label);
    builder_.add_comment0(mod, label + " start");

    nb::list part_list = nb::cast<nb::list>(emitted_3d);
    for (size_t pi = 0; pi < part_list.size(); ++pi) {
      nb::list k_list = nb::cast<nb::list>(part_list[pi]);
      for (size_t k = 0; k < k_list.size(); ++k) {
        builder_.add_comment0(mod, "partition=" + std::to_string(pi) +
                                       " subIterK=" + std::to_string(k));
        nb::list em_list = nb::cast<nb::list>(k_list[k]);
        if (schedule && em_list.size() > 0) {
          // Build per-module instruction lists via Python emit callable, then
          // pass both em_list and inst_lists to instructionScheduleFromLists.
          nb::list inst_lists;
          for (size_t i = 0; i < em_list.size(); ++i) {
            inst_lists.append(emit_fn_(em_list[i], unroll_iter));
          }
          nb::object scheduled = schedule_fn_(em_list, inst_lists);
          builder_.add(mod, scheduled);
        } else {
          for (size_t i = 0; i < em_list.size(); ++i) {
            nb::list items = nb::cast<nb::list>(emit_fn_(em_list[i], unroll_iter));
            for (size_t j = 0; j < items.size(); ++j) {
              builder_.add(mod, items[j]);
            }
          }
        }
      }
    }

    builder_.add_comment0(mod, label + " end");
    return mod;
  }

  // ── emit_main_and_exit_loops ───────────────────────────────────────────
  // Port of LogicalScheduler.emitMainAndExitLoops.
  //
  // Emits: optional K<DepthU skip test, PRELOOP, MAINLOOP (with unrolling),
  // NGLL (when pgr>=2), NLL exit paths, and the SkipToEnd label target.
  // The tail loop is emitted separately by the Python caller (emitTailLoop).
  //
  // Parameters:
  //   preloop_emitted / emitted / ngll_emitted / nll_emitted:
  //     Python [partition][group][EmittedModule] lists from LogicalScheduler.
  //   no_tail_loop  : kernel["NoTailLoop"] — controls K<DepthU skip branch.
  //   pgr           : scheduler config pgr (0/1/2).
  //   unroll_factor : scheduler.unroll_factor.
  nb::object emit_main_and_exit_loops(
      nb::object preloop_emitted, nb::object emitted,
      nb::object ngll_emitted, nb::object nll_emitted,
      bool no_tail_loop, int pgr, int unroll_factor) const {

    nb::object mod = builder_.module("MainAndExitLoops");
    int uf = unroll_factor;
    const std::string end_label = "SkipToEnd";

    // ── Skip preloop/mainloop/NGLL/NLL when K < DepthU ──
    // (only when a tail loop exists — mirrors Python condition)
    if (!no_tail_loop) {
      builder_.add(mod, scmp_eq_u32_cls_(
          "src0"_a = sgpr_fn_("LoopCounterL"), "src1"_a = 0,
          "comment"_a = "K < DepthU? skip to tail loop"));
      builder_.add(mod, scbranch_scc1_(
          "labelName"_a = end_label,
          "comment"_a = "K < DepthU: only tail loop runs"));
    }

    // ── Preloop ──
    builder_.add(mod, emit_loop(preloop_emitted, "PRELOOP", 0, false));

    // ── Mainloop ──
    builder_.add_comment0(mod, "MAINLOOP");
    const std::string loop_begin = "LoopBeginL";
    builder_.add(mod, builder_.label(loop_begin, ""));

    int exitValue = pgr;

    for (int ui = 0; ui < uf; ++ui) {
      builder_.add(mod,
                   emit_loop(emitted, "MAINLOOP_C" + std::to_string(ui), ui,
                             /*schedule=*/true));
      builder_.add(mod,
                   ssub_u32_cls_(
                       "dst"_a = sgpr_fn_("LoopCounterL"),
                       "src0"_a = sgpr_fn_("LoopCounterL"), "src1"_a = 1,
                       "comment"_a = "dec counterL (copy " + std::to_string(ui) + ")"));
      builder_.add(mod,
                   scmp_eq_u32_cls_(
                       "src0"_a = sgpr_fn_("LoopCounterL"), "src1"_a = exitValue,
                       "comment"_a = "counterL == " + std::to_string(exitValue) +
                                     "? (copy " + std::to_string(ui) + " exit)"));
      if (ui < uf - 1) {
        const std::string exit_label = "ExitC" + std::to_string(ui);
        builder_.add(mod,
                     scbranch_scc1_(
                         "labelName"_a = exit_label,
                         "comment"_a = "copy " + std::to_string(ui) +
                                       " exit \u2192 NGLL_C" + std::to_string(ui)));
      } else {
        builder_.add(mod,
                     scbranch_scc0_(
                         "labelName"_a = loop_begin,
                         "comment"_a = "restart mainloop"));
      }
    }

    // ── NGLL + NLL exit paths ──
    bool hasNGLL = (pgr >= 2);
    builder_.add(mod, builder_.label("SkipMainloop", ""));
    if (hasNGLL) {
      builder_.add(mod, builder_.label("SkipToNGLL", ""));
    }

    // After mainloop C{ui}, data in LDS/vgprs corresponds to
    //   unroll_iter = (ui + pgr) % uf  for NLL
    //   unroll_iter = (ui + 1)  % uf  for NGLL
    // SkipToNLL is placed before whichever NLL block uses unroll_iter=0.
    int last    = uf - 1;
    int nll_ft  = (last + pgr) % uf;

    // Fall-through from last mainloop copy
    if (hasNGLL) {
      builder_.add_comment0(mod, "NGLL_C" + std::to_string(last));
      builder_.add(mod, emit_loop(ngll_emitted,
                                   "NGLL_C" + std::to_string(last),
                                   (last + 1) % uf,
                                   /*schedule=*/true));
    }
    if (nll_ft == 0) {
      builder_.add(mod, builder_.label("SkipToNLL", ""));
    }
    builder_.add_comment0(mod, "NLL_C" + std::to_string(last));
    builder_.add(mod, emit_loop(nll_emitted,
                                 "NLL_C" + std::to_string(last), nll_ft,
                                 /*schedule=*/true));
    builder_.add(mod, sbranch_cls_("labelName"_a = end_label,
                                    "comment"_a = "skip other exit paths"));

    for (int ui = 0; ui < uf - 1; ++ui) {
      int nll_idx = (ui + pgr) % uf;
      // Define the exit label target
      builder_.add(mod, builder_.label("ExitC" + std::to_string(ui), ""));
      if (hasNGLL) {
        builder_.add_comment0(mod, "NGLL_C" + std::to_string(ui));
        builder_.add(mod, emit_loop(ngll_emitted,
                                     "NGLL_C" + std::to_string(ui),
                                     (ui + 1) % uf,
                                     /*schedule=*/true));
      }
      if (nll_idx == 0) {
        builder_.add(mod, builder_.label("SkipToNLL", ""));
      }
      builder_.add_comment0(mod, "NLL_C" + std::to_string(ui));
      builder_.add(mod, emit_loop(nll_emitted,
                                   "NLL_C" + std::to_string(ui), nll_idx,
                                   /*schedule=*/true));
      if (ui < uf - 2) {
        builder_.add(mod, sbranch_cls_("labelName"_a = end_label,
                                        "comment"_a = "skip other exit paths"));
      }
    }

    // SkipToEnd label target (preloop-skip branches land here; tail loop
    // wrapper adds the tail loop body after this module).
    builder_.add(mod, builder_.label(end_label, ""));

    return mod;
  }

  // ── emit_tail_loop ─────────────────────────────────────────────────────
  // Port of the structural (non-VGPR-realloc) part of LogicalScheduler.emitTailLoop.
  //
  // The Python caller is responsible for:
  //   1. kernel["NoTailLoop"] check (returns early if true)
  //   2. _realloc_tail_tiles_flat (reallocates VGPR tiles to flat layout)
  //   3. Pre-collecting mask_k_init_items via list(emitter.emit_mask_k_init())
  //      — this allocates the mask VGPRs as a side-effect
  //   4. Pre-collecting mask_k_done_items via list(emitter.emit_mask_k_done())
  //
  // Returns a rocisa Module named "TailLoop".
  nb::object emit_tail_loop(nb::object tailloop_emitted,
                            nb::object mask_k_init_items,
                            nb::object mask_k_done_items) const {
    nb::object mod = builder_.module("TailLoop");
    builder_.add_comment0(mod, "TAILLOOP");

    nb::list init_items = nb::cast<nb::list>(mask_k_init_items);
    for (size_t i = 0; i < init_items.size(); ++i) {
      builder_.add(mod, init_items[i]);
    }

    builder_.add(mod, emit_loop(tailloop_emitted, "TAILLOOP",
                                 /*unroll_iter=*/0,
                                 /*schedule=*/false));

    nb::list done_items = nb::cast<nb::list>(mask_k_done_items);
    for (size_t i = 0; i < done_items.size(); ++i) {
      builder_.add(mod, done_items[i]);
    }

    return mod;
  }

  // ── init_vgpr_tiles_to_zero ───────────────────────────────────────────
  // Port of Kernel._zeroRegRange + initVgprTilesToZero.
  //
  // The Python caller groups consecutive vgprTiles by pool type (AGPR vs VGPR)
  // and passes those groups here as a flat list of (firstReg, totalRegs,
  // isAgpr, tmpVgpr) tuples. tmpVgpr is a Python-allocated 2-register aligned
  // VGPR base index for MFMA-based zero-init; pass -1 to fall back to scalar
  // VMovB32/VAccvgprWrite for the entire group.
  //
  // Returns a rocisa Module.
  nb::object init_vgpr_tiles_to_zero(
      const std::string& tc,
      const std::vector<std::tuple<int, int, bool, int>>& reg_groups) const {
    nb::object mod = builder_.module();
    builder_.add_comment0(mod, "Init " + tc + " vgprTiles to zero");
    for (auto& [firstReg, totalRegs, isAgpr, tmpVgpr] : reg_groups) {
      zero_reg_range(mod, tc, firstReg, totalRegs, isAgpr, tmpVgpr);
    }
    return mod;
  }

 private:
  // Port of Kernel._zeroRegRange.
  // Uses MFMA (32x32x16 I8) for blocks of 16 registers, scalar
  // VMovB32/VAccvgprWrite for the remainder.
  void zero_reg_range(nb::object mod, const std::string& tc,
                      int firstReg, int totalRegs, bool isAgpr,
                      int tmpVgpr) const {
    constexpr int regsPerMfma = 16;
    int numMfma = (tmpVgpr >= 0) ? (totalRegs / regsPerMfma) : 0;

    if (numMfma > 0) {
      // Zero two VGPRs and issue a NOP so they're ready before MFMA
      builder_.add(mod, vmov_b64_cls_(
          "dst"_a = vgpr_fn_(tmpVgpr, 2), "src"_a = 0, "comment"_a = ""));
      builder_.add(mod, snop_cls_(
          "waitState"_a = 1,
          "comment"_a = "wait for vgpr to be ready before MFMA"));

      nb::list variant;
      variant.append(32); variant.append(32); variant.append(16); variant.append(1);

      for (int i = 0; i < numMfma; ++i) {
        int r = firstReg + i * regsPerMfma;
        nb::object tile_reg = isAgpr ? accvgpr_fn_(r, regsPerMfma)
                                     : vgpr_fn_(r, regsPerMfma);
        std::string comment = "init" + tc + ": [" + std::to_string(r) + ":" +
                              std::to_string(r + regsPerMfma - 1) + "]";
        builder_.add(mod, mfma_inst_cls_(
            "instType"_a  = inst_type_enum_.attr("INST_I8"),
            "accType"_a   = inst_type_enum_.attr("INST_I32"),
            "variant"_a   = variant,
            "mfma1k"_a    = false,
            "acc"_a       = tile_reg,
            "a"_a         = vgpr_fn_(tmpVgpr, 2),
            "b"_a         = vgpr_fn_(tmpVgpr, 2),
            "acc2"_a      = 0,
            "comment"_a   = comment));
      }
    }

    // Scalar writes for remaining registers (or all when no MFMA path)
    for (int i = numMfma * regsPerMfma; i < totalRegs; ++i) {
      nb::object tile_reg = isAgpr ? accvgpr_fn_(firstReg + i)
                                   : vgpr_fn_(firstReg + i);
      nb::object write_cls = isAgpr ? vaccvgpr_write_ : vmov_b32_cls_;
      builder_.add(mod, write_cls("dst"_a = tile_reg, "src"_a = 0,
                                  "comment"_a = "init" + tc));
    }
  }

  rocisa_builder::ModuleBuilder& builder_;
  nb::object emit_fn_;
  nb::object schedule_fn_;

  // rocisa.container
  nb::object sgpr_fn_;
  nb::object vgpr_fn_;
  nb::object accvgpr_fn_;

  // rocisa.instruction — control flow
  nb::object ssub_u32_cls_;
  nb::object scmp_eq_u32_cls_;
  nb::object scbranch_scc0_;
  nb::object scbranch_scc1_;
  nb::object sbranch_cls_;

  // rocisa.instruction — VGPR zero-init
  nb::object vmov_b64_cls_;
  nb::object vmov_b32_cls_;
  nb::object vaccvgpr_write_;
  nb::object snop_cls_;
  nb::object mfma_inst_cls_;
  nb::object inst_type_enum_;
};

}  // namespace tw::subtile::loop_orch
