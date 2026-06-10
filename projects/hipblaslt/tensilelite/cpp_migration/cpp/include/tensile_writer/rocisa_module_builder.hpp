// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
//
// rocisa module-builder foundation for the subtile emit slices.
//
// Until now the C++ migration layer has been *data-only*: it computes geometry,
// offset-assignment plans, scheduling orders, and instType decisions, and the
// Python emit functions in Tensile.Components.Subtile turn those decisions into
// rocisa Module objects. This header is the foundation that lets the C++ side
// *construct the rocisa Module itself*, so subsequent slices can move the large
// per-tile emit loops (InstructionEmitter, SubtileGREmit, SubtileLREmit,
// SubtileScaleEmit) out of Python without first re-implementing rocisa in C++.
//
// Strategy (no new C++ link dependency):
//   rocisa is its own nanobind extension (_rocisa) with the Python-visible
//   submodules rocisa.code / rocisa.instruction / rocisa.container. Rather than
//   link _tensile_writer against the rocisa C++ library (which would drag the
//   whole ISA/HIP build surface into the dependency-light migration wheel, and
//   force cross-extension nanobind type sharing), the builder holds cached
//   nb::object handles to the rocisa Python API and constructs genuine rocisa
//   objects through it. The returned object IS a real rocisa Module — callers
//   (and the existing rocisa pass pipeline) cannot tell it was assembled from
//   C++. The rocisa dependency is therefore explicit but runtime-only (an
//   import), exactly like Kernel.py's own `from rocisa.code import Module`.
//
// Boundary contract (see docs/rocisa_module_builder_boundary.md):
//   The builder owns NO writer state. Every value that lives on the Python
//   writer — VGPR/SGPR indices from the register pools, sgpr/label *names*,
//   writer.states scalars (laneSGPRCount, unrollIdx, ...), and tail-loop helper
//   vgprs — is resolved on the Python side and passed into the builder as plain
//   ints / strings. The builder never references the writer, its pools, or its
//   labels; it only assembles rocisa Items from those primitive inputs. This
//   keeps register allocation and label minting authoritative in Python while
//   the *shape* of the emitted module moves to C++.
//
// StinkyTofu: gfx950 subtile emits through the rocisa string/Module path, so
// this foundation targets rocisa. The builder API is backend-neutral (it speaks
// in "items" and "modules"), leaving room for a StinkyTofu construction backend
// later without changing call sites.

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <cassert>
#include <optional>
#include <string>
#include <vector>

namespace tw::subtile::rocisa_builder {

namespace nb = nanobind;
using namespace nb::literals;  // enables the "name"_a keyword-argument syntax

// Thin C++ facade over the rocisa Python construction API.
//
// One ModuleBuilder caches the rocisa class/function handles it needs; reuse a
// single instance for a whole emit pass. All methods return real rocisa objects
// as nb::object so they flow back to Python (and into the rocisa pass pipeline)
// transparently.
class ModuleBuilder {
 public:
  ModuleBuilder() {
    // Import the Python-visible rocisa submodules. These are the same imports
    // Kernel.py / InstructionEmitter.py use; importing them here makes the
    // rocisa dependency explicit and fails loudly (ImportError) if rocisa is
    // not installed, instead of silently degrading.
    nb::module_ code = nb::module_::import_("rocisa.code");
    nb::module_ container = nb::module_::import_("rocisa.container");
    nb::module_ enum_ = nb::module_::import_("rocisa.enum");
    inst_ = nb::module_::import_("rocisa.instruction");

    module_cls_ = code.attr("Module");
    textblock_cls_ = code.attr("TextBlock");
    label_cls_ = code.attr("Label");
    vgpr_fn_ = container.attr("vgpr");
    sgpr_fn_ = container.attr("sgpr");
    accvgpr_fn_ = container.attr("accvgpr");
    mgpr_fn_ = container.attr("mgpr");
    ds_modifiers_cls_ = container.attr("DSModifiers");
    mubuf_modifiers_cls_ = container.attr("MUBUFModifiers");
    vop3p_modifiers_cls_ = container.attr("VOP3PModifiers");
    inst_type_enum_ = enum_.attr("InstType");

    // Instruction classes consumed by the ported GR/LR data-movement leaves.
    sadd_u32_cls_ = inst_.attr("SAddU32");
    sadd_cu32_cls_ = inst_.attr("SAddCU32");
    sxor_b32_cls_ = inst_.attr("SXorB32");
    vxor_b32_cls_ = inst_.attr("VXorB32");
    buffer_load_b128_cls_ = inst_.attr("BufferLoadB128");
    ds_load_b128_cls_ = inst_.attr("DSLoadB128");

    // Instruction classes consumed by the ported MX scale GR/LR leaves.
    smov_b32_cls_ = inst_.attr("SMovB32");
    ds_load_b32_cls_ = inst_.attr("DSLoadB32");

    // Instruction classes for MFMA emission.
    mfma_instruction_cls_ = inst_.attr("MFMAInstruction");
    mxmfma_instruction_cls_ = inst_.attr("MXMFMAInstruction");
    swait_cnt_cls_ = inst_.attr("SWaitCnt");
  }

  // ---- Module / container factories ------------------------------------

  // Fresh rocisa Code.Module. `name` mirrors rocisa's Module(name="").
  nb::object module(const std::string& name = "") const {
    return module_cls_(name);
  }

  // rocisa Code.TextBlock — raw, unstructured text.
  nb::object text_block(const std::string& text) const {
    return textblock_cls_(text);
  }

  // rocisa Code.Label(label, comment, alignment). `label` is the writer-minted
  // label *name* (a string crosses the boundary; the builder does not mint it).
  nb::object label(const std::string& label_name, const std::string& comment = "",
                   int alignment = 1) const {
    return label_cls_(label_name, comment, alignment);
  }

  // rocisa container.vgpr / .sgpr. The index/name is supplied by the caller
  // (resolved from the writer's register pools on the Python side).
  nb::object vgpr(nb::handle reg, int size = 1) const {
    return vgpr_fn_(reg, size);
  }
  nb::object sgpr(nb::handle reg, int size = 1) const {
    return sgpr_fn_(reg, size);
  }

  // rocisa container.DSModifiers(na=..., offset=...). na is the number of
  // addresses (1 for single-address DS ops, 2 for dual-address DS ops used by
  // the subtile LR/GR emit paths); defaults to 1.
  nb::object ds_modifiers(int offset = 0, int na = 1) const {
    return ds_modifiers_cls_("na"_a = na, "offset"_a = offset);
  }

  // Generic instruction factory: construct rocisa.instruction.<class_name> with
  // arbitrary positional/keyword args. This is the open-ended hook future emit
  // slices use to build any instruction the subtile path needs, with the C++
  // side supplying the (writer-resolved) operands and immediates. Raises
  // AttributeError if the class does not exist.
  nb::object instruction(const std::string& class_name, nb::args args,
                         nb::kwargs kwargs) const {
    nb::object cls = nb::getattr(inst_, class_name.c_str());
    return cls(*args, **kwargs);
  }

  // ---- Module mutation helpers -----------------------------------------

  // Append a pre-built rocisa Item (instruction / module / text / label).
  // Returns the added item to support one-line add patterns, matching
  // rocisa Module.add().
  nb::object add(nb::handle mod, nb::handle item) const {
    return mod.attr("add")(item);
  }

  void add_comment(nb::handle mod, const std::string& comment) const {
    mod.attr("addComment")(comment);
  }
  void add_comment_align(nb::handle mod, const std::string& comment) const {
    mod.attr("addCommentAlign")(comment);
  }
  // rocisa Module.addComment0 — the "//<comment>" leading-newline comment form
  // used by the GR/LR LDS-swap emit leaves.
  void add_comment0(nb::handle mod, const std::string& comment) const {
    mod.attr("addComment0")(comment);
  }

  // Flatten a module to its leaf items (rocisa Module.flatitems()). The subtile
  // emit leaves return list(module.flatitems()); exposing it here lets a ported
  // C++ leaf hand back the same flattened list.
  nb::object flatitems(nb::handle mod) const { return mod.attr("flatitems")(); }

  // ---- Representative typed leaf builders (foundation demonstration) ----
  //
  // These reproduce two self-contained production emit leaves
  // (InstructionEmitter.emit_sync / emit_wait_lr) entirely in C++ to prove the
  // builder yields byte-identical rocisa output. They take no writer state, so
  // they are the natural first leaves to live on the C++ side. Richer leaves
  // (which consume register indices / labels) follow the same pattern: scalars
  // in, rocisa Module out.

  // == InstructionEmitter.emit_sync(): a standalone barrier. ==
  nb::object barrier(const std::string& comment = "Barrier") const {
    return inst_.attr("SBarrier")("comment"_a = comment);
  }

  // == InstructionEmitter.emit_wait_lr(): wait for local reads to drain. ==
  nb::object wait_lr(
      const std::string& comment = "Wait for LR to complete") const {
    return inst_.attr("SWaitCnt")("dscnt"_a = 0, "vlcnt"_a = -1, "vscnt"_a = -1,
                                  "comment"_a = comment);
  }

  // Build a Module wrapping a single item — the common shape of an emit leaf
  // that produces one instruction. Useful for parity smoke tests and as a
  // template for single-instruction leaves.
  nb::object single_item_module(nb::handle item,
                                const std::string& name = "") const {
    nb::object mod = module(name);
    add(mod, item);
    return mod;
  }

  // ---- GR / LR data-movement emit leaves (ported from Python) ----------
  //
  // These reproduce the rocisa construction of the subtile GR/LR data-movement
  // leaves (SubtileGREmit.emitSingleBufferLoad / _emitGRPtrUpdate_TLU0 /
  // _emitGRLDSSwap_TLU0, SubtileLREmit.emitSingleDsRead / _emitLRLDSSwap_1x2).
  // The instruction-shape decision (skip flag, m0/DS offsets, per-read map) is
  // computed by the data-only C++ plans (tile_info.hpp); register state
  // (soffset/voff/DS-address VGPRs, sharedVgprLROffset) and the GR/LR tag
  // dispatch stay authoritative in Python and cross the boundary here as
  // primitive ints / strings / already-resolved rocisa operand objects.

  // == SubtileGREmit.emitSingleBufferLoad ==
  // For each m0 offset: stage m0 (SAddU32 into mgpr(0)) then emit the
  // direct-to-LDS buffer_load_b128. `soffset` is the (Python-resolved) shared
  // SGPR soffset object or the int 0; `voffs[i]` is the per-load VGPR offset
  // index. The plan's skip case is handled on the Python side (returns an empty
  // Module) so this is only called for emitting loads.
  nb::object single_buffer_load(const std::string& tc, bool is_glc, bool is_slc,
                                bool is_nt, long offsetK, long grBaseId,
                                const std::vector<long>& m0Offsets,
                                nb::object soffset,
                                const std::vector<int>& voffs) const {
    nb::object mod = module();
    nb::object srd = sgpr_fn_(std::string("Srd") + tc, 4);
    std::string wba = std::string("LocalWriteBaseAddr") + tc;
    for (size_t i = 0; i < m0Offsets.size(); ++i) {
      add(mod, sadd_u32_cls_("dst"_a = mgpr_fn_(0), "src0"_a = sgpr_fn_(wba),
                             "src1"_a = (m0Offsets[i] - offsetK)));
      nb::object mubuf = mubuf_modifiers_cls_(
          "offen"_a = true, "offset12"_a = offsetK, "glc"_a = is_glc,
          "slc"_a = is_slc, "nt"_a = is_nt, "lds"_a = true);
      std::string comment = "grBaseId = " + std::to_string(grBaseId) +
                            ", i= " + std::to_string(i);
      add(mod, buffer_load_b128_cls_(
                   "dst"_a = nb::none(), "vaddr"_a = vgpr_fn_(voffs[i]),
                   "saddr"_a = srd, "soffset"_a = soffset, "mubuf"_a = mubuf,
                   "comment"_a = comment));
    }
    return mod;
  }

  // == SubtileLREmit.emitSingleDsRead ==
  // Emit one ds_read_b128 per entry. `dstVgpr` is the destination tile's base
  // VGPR index; `dstRegOffsets[r]` / `addrVgprs[r]` are the (Python-resolved)
  // per-read destination VGPR offset and sharedVgprLROffset address index.
  nb::object single_ds_read(const std::string& tc, long sId0, long sId1,
                            long subIterK, int dstVgpr, int regsPerDsRead,
                            long offset, const std::vector<int>& dstRegOffsets,
                            const std::vector<int>& addrVgprs) const {
    nb::object mod = module();
    for (size_t r = 0; r < dstRegOffsets.size(); ++r) {
      std::string comment = "Subtile" + tc + "[" + std::to_string(sId0) + ", " +
                            std::to_string(sId1) +
                            "] subIterK=" + std::to_string(subIterK) +
                            " read=" + std::to_string(r);
      add(mod, ds_load_b128_cls_(
                   "dst"_a = vgpr_fn_(dstVgpr + dstRegOffsets[r], regsPerDsRead),
                   "src"_a = vgpr_fn_(addrVgprs[r]),
                   "ds"_a = ds_modifiers(offset, 1), "comment"_a = comment));
    }
    return mod;
  }

  // == SubtileGREmit._emitGRPtrUpdate_TLU0 ==
  // Advance the SRD base pointer by one depthU iteration (`inc` bytes), with
  // carry into Srd+1.
  nb::object gr_ptr_update(const std::string& tc, long inc) const {
    nb::object mod = module("GR Ptr Update (" + tc + ")");
    std::string srd = std::string("Srd") + tc;
    add(mod, sadd_u32_cls_(
                 "dst"_a = sgpr_fn_(srd), "src0"_a = sgpr_fn_(srd),
                 "src1"_a = inc,
                 "comment"_a = tc + ": advance SRD by " + std::to_string(inc) +
                               " bytes"));
    add(mod, sadd_cu32_cls_("dst"_a = sgpr_fn_(srd + "+1"),
                            "src0"_a = sgpr_fn_(srd + "+1"), "src1"_a = 0,
                            "comment"_a = tc + ": carry"));
    return mod;
  }

  // == SubtileGREmit._emitGRLDSSwap_TLU0 ==
  // Toggle the GR DTL write target between double-buffer halves
  // (LocalWriteBaseAddr ^= Swap).
  nb::object gr_lds_buffer_swap(const std::string& tc) const {
    nb::object mod = module();
    add_comment0(mod, "Emit code to swap " + tc + " GR m0 offsets");
    std::string wba = std::string("LocalWriteBaseAddr") + tc;
    add(mod, sxor_b32_cls_("dst"_a = sgpr_fn_(wba), "src0"_a = sgpr_fn_(wba),
                           "src1"_a = sgpr_fn_(std::string("Swap") + tc),
                           "comment"_a = std::string("")));
    return mod;
  }

  // == SubtileLREmit._emitLRLDSSwap_1x2 ==
  // Toggle each LR read offset between double-buffer halves (offset ^= swap).
  // `voffs` / `vswaps` are the (Python-resolved) sharedVgprLROffset and
  // sharedVgprLROffsetSwap VGPR index lists.
  nb::object lr_lds_buffer_swap(const std::string& tc,
                                const std::vector<int>& voffs,
                                const std::vector<int>& vswaps) const {
    assert(voffs.size() == vswaps.size() &&
           "lr_lds_buffer_swap requires voffs and vswaps to have equal length");
    nb::object mod = module();
    add_comment0(mod, "Emit code to swap " + tc + " LR vgpr offsets");
    for (size_t i = 0; i < voffs.size(); ++i) {
      add(mod, vxor_b32_cls_("dst"_a = vgpr_fn_(voffs[i]),
                             "src0"_a = vgpr_fn_(voffs[i]),
                             "src1"_a = vgpr_fn_(vswaps[i]),
                             "comment"_a = std::string("")));
    }
    return mod;
  }

  // ---- MX scale GR / LR data-movement emit leaves (ported from Python) -----
  //
  // These reproduce the rocisa construction of the MX scale-factor (MXSA/MXSB)
  // GR/LR data-movement leaves (SubtileScaleEmit.globalReadDoScaleSubtile /
  // emitSubtileScaleDsRead / emitScaleGRPtrUpdate). Scale factors use a simpler
  // access pattern than data tiles: GR is a single direct-to-LDS buffer_load
  // per wave, LR is a ds_read_b32 per scale group. The GR/LR LDS-swap leaves are
  // byte-identical to the AB leaves above and reuse gr_lds_buffer_swap /
  // lr_lds_buffer_swap with the MXSA/MXSB component tag. Register state
  // (sharedVgprGROffset / sharedVgprLROffset, destination tile VGPRs) and the
  // MXBlock guard stay authoritative in Python and cross the boundary here as
  // primitive ints / strings.

  // == SubtileScaleEmit.globalReadDoScaleSubtile ==
  // Set M0 to the scale LDS base then emit the direct-to-LDS buffer_load_b128
  // (vaddr serves as both the global read offset and the LDS write offset).
  // `tc` is the full scale component name ("MXSA" / "MXSB"); `voff` is the
  // (Python-resolved) sharedVgprGROffset[0] index.
  nb::object scale_gr_load(const std::string& tc, bool is_glc, bool is_slc,
                           bool is_nt, int voff) const {
    nb::object mod = module();
    add_comment0(mod, "Scale GR: " + tc + " (DTL: BufferLoadB128 -> LDS)");
    add(mod, smov_b32_cls_(
                 "dst"_a = mgpr_fn_(0),
                 "src"_a = sgpr_fn_(std::string("LocalWriteBaseAddr") + tc),
                 "comment"_a = "scale" + tc + ": M0 = scaleLdsBase"));
    nb::object mubuf = mubuf_modifiers_cls_(
        "offen"_a = true, "offset12"_a = 0, "glc"_a = is_glc, "slc"_a = is_slc,
        "nt"_a = is_nt, "lds"_a = true);
    add(mod, buffer_load_b128_cls_(
                 "dst"_a = nb::none(), "vaddr"_a = vgpr_fn_(voff),
                 "saddr"_a = sgpr_fn_(std::string("Srd") + tc, 4),
                 "soffset"_a = 0, "mubuf"_a = mubuf,
                 "comment"_a = "scale" + tc + ": DTL b128 load"));
    return mod;
  }

  // == SubtileScaleEmit.emitSubtileScaleDsRead ==
  // Emit one ds_read_b32 (4 bytes = 4 E8M0 scale values) into one VGPR for a
  // single scale group. `vdst` is the destination tile VGPR index; `addrVgpr`
  // is the sharedVgprLROffset[0] address index. When `k >= 0` the comment
  // carries the K index (scheduler emit path); k < 0 omits it (PGR=0 path).
  nb::object scale_ds_read(const std::string& tc, int vdst, int addrVgpr,
                           long dsOffset, long scaleGroupIdx,
                           long k = -1) const {
    nb::object mod = module();
    std::string comment =
        "scale" + tc + "[group" + std::to_string(scaleGroupIdx);
    if (k >= 0) comment += ",K=" + std::to_string(k);
    comment += "]: load 4B from LDS";
    add(mod, ds_load_b32_cls_("dst"_a = vgpr_fn_(vdst),
                              "src"_a = vgpr_fn_(addrVgpr),
                              "ds"_a = ds_modifiers(dsOffset, 1),
                              "comment"_a = comment));
    return mod;
  }

  // == SubtileScaleEmit.emitScaleGRPtrUpdate ==
  // Advance the scale SRD base pointer by one depthU iteration (`inc` bytes),
  // with carry into Srd<tc>+1.
  nb::object scale_gr_ptr_update(const std::string& tc, long inc) const {
    nb::object mod = module();
    add_comment0(mod, "Scale SRD update: " + tc + " += " + std::to_string(inc));
    std::string srd = std::string("Srd") + tc;
    add(mod, sadd_u32_cls_("dst"_a = sgpr_fn_(srd), "src0"_a = sgpr_fn_(srd),
                           "src1"_a = inc));
    add(mod, sadd_cu32_cls_("dst"_a = sgpr_fn_(srd + "+1"),
                            "src0"_a = sgpr_fn_(srd + "+1"), "src1"_a = 0));
    return mod;
  }

  // ---- MFMA instruction emission (ported from Kernel.emitMfmaInstruction) ---
  //
  // Builds one MFMA rocisa Module: MFMAInstruction for BF16 (miK != 128) or
  // MXMFMAInstruction for the MX FP4/FP8 family (miK == 128).
  //
  // Boundary contract: all register indices and boolean flags are resolved by
  // the Python caller (from writer.vgprPool, kernel dict, and tile pool identity
  // checks) and passed here as plain ints/booleans. The instTypeName string for
  // miK==128 cases comes from mfma_f8f6f4_inst_type() (emit_leaves.hpp) — the
  // Python caller calls that function and passes the resulting string here.
  // For miK!=128 (BF16), instTypeName should be empty (""); INST_BF16 is used.
  //
  // Parameters:
  //   vgprAStart/opASize   — A tile: base VGPR index and register count
  //   vgprBStart/opBSize   — B tile: base VGPR index and register count
  //   vgprCStart/opCSize   — C tile: base VGPR index and register count
  //   vgprDStart/opDSize   — D tile: base VGPR index and register count
  //   dIsVgpr/cIsVgpr      — true if D/C tile's pool is vgprPool (not agprPool)
  //   miArchVgpr           — kernel["MIArchVgpr"]: if true, D/C always use vgpr()
  //   sourceSwap            — kernel["SourceSwap"]: swap A/B operand positions
  //   miK                  — kernel["MatrixInstK"]
  //   instTypeName         — F8/F6/F4 instType member name ("INST_F8" etc.);
  //                          empty string for BF16 path (miK != 128)
  //   scaleAVgpr/scaleBVgpr — real scale VGPR indices, or -1 for fallback path
  //   unitScaleVgpr        — kernel["_subtileUnitScaleVgpr"] for fallback path;
  //                          only accessed when scaleAVgpr < 0 and miK == 128
  //   scaleAsel/scaleBsel  — op_sel indices for the VOP3P modifier
  //   comment              — instruction comment string
  nb::object emit_mfma(int vgprAStart, int opASize, int vgprBStart, int opBSize,
                       int vgprCStart, int opCSize, int vgprDStart, int opDSize,
                       bool dIsVgpr, bool cIsVgpr, bool miArchVgpr, bool sourceSwap,
                       int miK, const std::string& instTypeName,
                       int scaleAVgpr = -1, int scaleBVgpr = -1,
                       int unitScaleVgpr = -1, int scaleAsel = 0, int scaleBsel = 0,
                       const std::string& comment = "") const {
    nb::object mod = module();

    // D/C register type: vgpr() when the tile's pool is vgprPool or when
    // MIArchVgpr forces vgpr-only accumulation.
    bool useVgprD = dIsVgpr || miArchVgpr;
    bool useVgprC = cIsVgpr || miArchVgpr;

    // A/B operands: SourceSwap physically exchanges which tile goes in which
    // operand slot (this is what the Python code does before calling emitMfma).
    int aStart = sourceSwap ? vgprBStart : vgprAStart;
    int aSize  = sourceSwap ? opBSize    : opASize;
    int bStart = sourceSwap ? vgprAStart : vgprBStart;
    int bSize  = sourceSwap ? opASize    : opBSize;

    nb::object aOp  = vgpr_fn_(aStart, aSize);
    nb::object bOp  = vgpr_fn_(bStart, bSize);
    nb::object dReg = useVgprD ? vgpr_fn_(vgprDStart, opDSize)
                               : accvgpr_fn_(vgprDStart, opDSize);
    nb::object cReg = useVgprC ? vgpr_fn_(vgprCStart, opCSize)
                               : accvgpr_fn_(vgprCStart, opCSize);

    nb::object inst_f32 = inst_type_enum_.attr("INST_F32");

    // variant list [M, N, K, batch] — constant for gfx950 subtile.
    nb::list variant;
    variant.append(16);
    variant.append(16);
    variant.append(miK);
    variant.append(1);

    if (miK == 128) {
      // MX FP4/FP8 path: V_MFMA_SCALE_F32_16x16x128_F8F6F4
      nb::object mxInstType = inst_type_enum_.attr(instTypeName.c_str());

      if (scaleAVgpr >= 0 && scaleBVgpr >= 0) {
        // Real scale VGPRs supplied — build op_sel / op_sel_hi modifiers.
        nb::list op_sel, op_sel_hi;
        op_sel.append(scaleAsel % 2);
        op_sel.append(scaleBsel % 2);
        op_sel_hi.append((scaleAsel >> 1) % 2);
        op_sel_hi.append((scaleBsel >> 1) % 2);
        nb::object vop3 = vop3p_modifiers_cls_(
            "op_sel"_a = op_sel, "op_sel_hi"_a = op_sel_hi);
        add(mod, mxmfma_instruction_cls_(
                     "instType"_a = mxInstType, "accType"_a = inst_f32,
                     "variant"_a = variant,
                     "acc"_a = dReg, "a"_a = aOp, "b"_a = bOp, "acc2"_a = cReg,
                     "mxsa"_a = vgpr_fn_(scaleAVgpr),
                     "mxsb"_a = vgpr_fn_(scaleBVgpr),
                     "vop3"_a = vop3, "comment"_a = comment));
      } else {
        // No real scale — use the pre-initialized unit-scale VGPR (0x7f7f7f7f
        // = scale 1.0 in E8M0).  The caller asserts unitScaleVgpr >= 0.
        add(mod, mxmfma_instruction_cls_(
                     "instType"_a = mxInstType, "accType"_a = inst_f32,
                     "variant"_a = variant,
                     "acc"_a = dReg, "a"_a = aOp, "b"_a = bOp, "acc2"_a = cReg,
                     "mxsa"_a = vgpr_fn_(unitScaleVgpr),
                     "mxsb"_a = vgpr_fn_(unitScaleVgpr),
                     "comment"_a = comment));
      }
    } else {
      // BF16 path: V_MFMA_F32_16x16x<miK>_BF16
      add(mod, mfma_instruction_cls_(
                   "instType"_a = inst_type_enum_.attr("INST_BF16"),
                   "accType"_a = inst_f32, "variant"_a = variant,
                   "mfma1k"_a = false,
                   "acc"_a = dReg, "a"_a = aOp, "b"_a = bOp, "acc2"_a = cReg,
                   "comment"_a = comment));
    }
    return mod;
  }

 private:
  nb::object module_cls_;
  nb::object textblock_cls_;
  nb::object label_cls_;
  nb::object vgpr_fn_;
  nb::object sgpr_fn_;
  nb::object accvgpr_fn_;
  nb::object mgpr_fn_;
  nb::object ds_modifiers_cls_;
  nb::object mubuf_modifiers_cls_;
  nb::object vop3p_modifiers_cls_;
  nb::object inst_type_enum_;
  nb::object sadd_u32_cls_;
  nb::object sadd_cu32_cls_;
  nb::object sxor_b32_cls_;
  nb::object vxor_b32_cls_;
  nb::object buffer_load_b128_cls_;
  nb::object ds_load_b128_cls_;
  nb::object smov_b32_cls_;
  nb::object ds_load_b32_cls_;
  nb::object mfma_instruction_cls_;
  nb::object mxmfma_instruction_cls_;
  nb::object swait_cnt_cls_;
  nb::object inst_;
};

}  // namespace tw::subtile::rocisa_builder
