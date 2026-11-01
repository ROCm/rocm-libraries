"""Mutation check of the rust-track lowerings: each mutant breaks one piece; its gate must FAIL on it (and pass on
the restored tree).

    python rt_mutants.py cpu <rust-tree> <cargo-target-dir> [--only NAME]
        apply each CPU mutant, `cargo test --lib` (2 jobs, 2 test threads, deep_nesting skipped) must fail;
        every file is restored in a finally
    python rt_mutants.py build <rust-tree> <cargo-target-dir> <out-dir>
        cluster CPU job: for each GPU mutant, `cargo build --release --lib` and keep `<out-dir>/mut_<name>.so`
    python rt_mutants.py list

GPU mutants run in the GPU job (``rt_gpu.sh`` step ``mut``): the arm ``D`` of the named harness with
``ROCKE_CODEOBJ_DIRECT_LIB=mut_<name>.so`` must fail its numerics (bitwise / tolerance / determinism), under a
watchdog. Python-side mutants (``py`` kind) patch the job's copy of ``rocke/runtime/direct_lift.py`` instead.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

RS = "platform/rust/rocke-codeobj/src/"
ISEL = RS + "rir/isel.rs"
MFMA = RS + "target/gfx9/mfma.rs"
Q38 = RS + "target/gfx9/q38.rs"
ASM9 = RS + "target/gfx9/asm.rs"
LIFT = "platform/python/rocke/runtime/direct_lift.py"

# (name, file, old, new, kind, harness that must fail / test that must fail)
MUTANTS = [
    # ---- hardware gates
    # (rt-g3: dropping exp's low part of log2e left B3 bitwise: below f32 rounding for its gate inputs, an
    # equivalent mutant there; the fraction left unrounded is not)
    ("exp_round_skipped", ISEL, "self.valu(Opc::VRndneF32, [r(e)], [ph], 0);", "self.copy(e, ph);", "gpu", "b3:r1:tuned"),
    ("dpp_bank_mask_ignored", ISEL,
     'let (row_mask, bank_mask) = (op.int_attr("row_mask").unwrap_or(15), op.int_attr("bank_mask").unwrap_or(15));',
     'let (row_mask, bank_mask) = (op.int_attr("row_mask").unwrap_or(15), 15);', "cpu", "q38::tests::masked_dpp_keeps_old_lanes"),
    # (rt-g3: bound_ctrl flipped is equivalent for the u32 max scan: max with 0 or with itself; a reversed row
    # shift is not)
    ("dpp_row_shift_reversed", ISEL,
     'let imm = crate::target::gfx9::q38::dpp_imm(op.int_attr("ctrl").unwrap_or(0), row_mask, bank_mask, bound);',
     'let c = op.int_attr("ctrl").unwrap_or(0);\n                let imm = crate::target::gfx9::q38::dpp_imm(if (0x111..=0x11f).contains(&c) { c - 0x10 } else { c }, row_mask, bank_mask, bound);',
     "gpu", "fmoe:4:delta"),
    ("cvt_fp8_hi_ignored", ISEL, 'let mods = if op.bool_attr("hi") == Some(true) { M_OPSEL_DST } else { 0 };',
     "let mods = 0;", "gpu", "fmoe:1:delta"),
    ("mx_scale_select_ignored", MFMA,
     "let ctl = fmt_a as i64 | (fmt_b as i64) << 7 | (opsel_a as i64) << 10 | (opsel_b as i64) << 12;",
     "let ctl = fmt_a as i64 | (fmt_b as i64) << 7;", "gpu", "fmoe:4:delta"),
    ("maximum3_abs_dropped", ISEL, 'let abs = (op.int_attr("abs").unwrap_or(0) as u32 & 7) * crate::target::gfx9::q38::M_ABS0;',
     "let abs = 0;", "gpu", "fmoe:1:delta"),
    # ---- the R2 levers the toy d8w0 WIN rests on (R2 verifier caveat 2): the toy replay with the lever arm must fail
    # (rt-g3: half mirror as full mirror and a dropped bound_ctrl were equivalent for the toy's complete butterfly
    # sums; a partner of itself and swapped operands are not)
    ("dppmirror_half_mirror_as_self", RS + "target/gfx9/isel_hooks.rs", "7 if mirror => XorShuffle::QuadPerm(0x141),",
     "7 if mirror => XorShuffle::QuadPerm(0xe4),", "cpu", "dpp::tests::mirror_masks_take_dpp_and_moves_fold_into_adds"),
    ("dppcombine_operands_swapped", RS + "target/gfx9/dpp.rs",
     "let fused = MInst { op: fused_op, defs: add.defs, uses: Uses::from([src, other]), imm: mv.imm, mods: 0, mem: add.mem };",
     "let fused = MInst { op: fused_op, defs: add.defs, uses: Uses::from([other, src]), imm: mv.imm, mods: 0, mem: add.mem };",
     "cpu", "dpp::tests::a_folded_move_reads_its_source_in_src0"),
    # (rt-g3: the reversed match formed no quads at all, the lever off; a shifted quad offset reads other dwords)
    ("ldsquads_wrong_offset", ISEL, "self.emit(MInst::new(Opc::DsLoadB128, [r(dst)], [addr]).with_imm(offset));",
     "self.emit(MInst::new(Opc::DsLoadB128, [r(dst)], [addr]).with_imm(offset + 4));",
     "gpu", "toy:d4w8l2:dppmirror+dppcombine+ldsquads"),
    # ---- the rust track's levers: the lever arm's numerics must fail (harness:<..>:<levers>)
    ("minmax_max_min_swapped", RS + "target/gfx9/minmax.rs", "found.push((k, if larger { max } else { min }));",
     "found.push((k, if larger { min } else { max }));", "gpu", "fmoe:1:delta:minmax+xorcmp"),
    ("xorcmp_order_reversed", RS + "rir/opt.rs", 'let (p, q) = if pred == "lt" { (x, y) } else { (y, x) };',
     'let (p, q) = if pred == "lt" { (y, x) } else { (x, y) };', "gpu", "fmoe:1:delta:xorcmp"),
    ("ldspairs_load_offsets_swapped", RS + "target/gfx9/lds_pairs.rs",
     "let mut merged = MInst::new(Opc::Gfx9(Gfx9Op::DsLoad2addrB32), [Opnd::R(pair)], [addr]).with_imm(imm);",
     "let mut merged = MInst::new(Opc::Gfx9(Gfx9Op::DsLoad2addrB32), [Opnd::R(pair)], [addr]).with_imm(imm >> 8 | (imm & 255) << 8);",
     "gpu", "b3:r1:tuned:ldspairs"),
    ("dppcombinef_mul_as_add", RS + "target/gfx9/dpp.rs", "Opc::VMulF32 => g(Gfx9Op::VMulF32Dpp),",
     "Opc::VMulF32 => g(Gfx9Op::VAddF32Dpp),", "cpu", "dpp::tests::dppcombinef_folds_each_op_into_its_own_dpp_form"),
    ("dppcombinef_add_as_mul", RS + "target/gfx9/dpp.rs", "Opc::VAddF32 => g(Gfx9Op::VAddF32Dpp),",
     "Opc::VAddF32 => g(Gfx9Op::VMulF32Dpp),", "cpu", "dpp::tests::dppcombinef_folds_each_op_into_its_own_dpp_form"),
    ("dppcombinef_max_as_add", RS + "target/gfx9/dpp.rs", "Opc::VMaxF32 => g(Gfx9Op::VMaxF32Dpp),",
     "Opc::VMaxF32 => g(Gfx9Op::VAddF32Dpp),", "cpu", "dpp::tests::dppcombinef_folds_each_op_into_its_own_dpp_form"),
        # (wgfence_waits_nothing on fmoe:1 survived rt-m6 / rt-m7: a race the numerics do not hit; the CPU twin
    # wgfence_test_waits_nothing gates the wait.)
    ("wgfence_test_waits_nothing", RS + "target/gfx9/waitcnt.rs",
     "    if inst.op == Opc::Gfx9(super::gen_isa::Gfx9Op::WaitRelease) {\n        need_vm = need_vm.min(state.vm.get(STORE_KEY));",
     "    if inst.op == Opc::Gfx9(super::gen_isa::Gfx9Op::WaitRelease) {\n        need_vm = need_vm.min(NONE);",
     "cpu", "q38::workgroup_releases_wait_for_stores_and_dma_not_loads"),
    # ---- v7 levers: bitop3, selxor, subadd (GPU: the lever arm must fail), litreuse (CPU: its unit test)
    ("bitop3_table_bit0_flipped", RS + "target/gfx9/bitop3.rs", "let table = tree.eval(insts, k);",
     "let table = tree.eval(insts, k) ^ 1;", "gpu", "fmoe:1:delta:selxor+bitop3"),
    ("bitop3_xor_as_or", RS + "target/gfx9/bitop3.rs", "        Opc::VXorB32 => x[0] ^ x[1],\n        Opc::VNotB32 => !x[0],\n        Opc::VAndOrB32",
     "        Opc::VXorB32 => x[0] | x[1],\n        Opc::VNotB32 => !x[0],\n        Opc::VAndOrB32", "cpu", "bitop3::tests::router_key_is_one_bitop3"),
    ("selxor_arms_swapped", RS + "rir/opt.rs", "let m = self.emit(OpKind::Select, ty.clone(), [cond, ka, kb], [], out);",
     "let m = self.emit(OpKind::Select, ty.clone(), [cond, kb, ka], [], out);", "gpu", "fmoe:1:delta:selxor"),
    ("selxor_mask_dropped", RS + "rir/opt.rs",
     "let (ka, kb) = (self.int_const(&ty, ka & mask), self.int_const(&ty, kb & mask));",
     "let (ka, kb) = (self.int_const(&ty, ka), self.int_const(&ty, kb));", "cpu", "q38::tests::router_key_levers_save_the_xor_and_the_add"),
    ("subadd_sign_flipped", RS + "rir/opt.rs", "let k = self.int_const(&ty, k1 - k2);",
     "let k = self.int_const(&ty, k1 + k2);", "gpu", "fmoe:1:delta:subadd"),
    ("litreuse_ignored", RS + "target/gfx9/legalize.rs", "let here = block.loop_depth > 0 || straight;",
     "let here = block.loop_depth > 0;", "cpu", "legalize::tests::litreuse_moves_a_literal_once_in_straight_line_code"),
    ("loadorder_counts_copies", RS + "rir/post.rs", "                    insts[j].op != Opc::Copy\n                        && insts[j].uses.iter()",
     "                    insts[j].uses.iter()", "cpu", "q38::tests::guarded_loads_issue_in_the_order_they_are_read"),
    ("dsrebase_field_truncated", ISEL, "let (hi, lo) = (total & !0xffff, total & 0xffff);",
     "let (hi, lo) = (total & !0xffff, total & 0x7fff);", "cpu", "q38::tests::dsrebase_shares_one_base_past_the_offset_field"),
    ("scalesep_b_as_adjacent", ASM9, "src1: t::SRC_VGPR + s1, op_sel, op_sel_hi",
     "src1: t::SRC_VGPR + s + 1, op_sel, op_sel_hi", "cpu", "q38::tests::scalesep_reads_the_scales_where_they_are"),
    # ---- the inline asm assembler (rir/isel_asm.rs) and i64 lshr
    ("asm_waitcnt_vm_high_bits_dropped", RS + "rir/isel_asm.rs", "Ok((vm & 15) | (vm >> 4 & 3) << 14 | (exp & 7) << 4 | (lgkm & 15) << 8)",
     "Ok((vm & 15) | (exp & 7) << 4 | (lgkm & 15) << 8)", "cpu", "q38::tests::a_bounded_poll_template_assembles_as_written"),
    ("asm_branch_to_the_block_after", RS + "rir/isel_asm.rs", "self.f.blocks[block].insts[at].imm = b as i64;",
     "self.f.blocks[block].insts[at].imm = b as i64 + 1;", "cpu", "q38::tests::a_bounded_poll_template_assembles_as_written"),
    ("asm_load_sc1_dropped", RS + "rir/isel_asm.rs", '"sc0" | "glc" => m |= M_GLC,\n                    "sc1" => m |= M_SC1,',
     '"sc0" | "glc" => m |= M_GLC,\n                    "sc1" => {}', "cpu",
     "q38::tests::a_bounded_poll_template_assembles_as_written"),
    ("i64_lshr_carry_dropped", ISEL, "let lo = self.int_bin(OpKind::Or, low, carry, uniform, false)?;",
     "let lo = low;", "cpu", "q38::tests::i64_shifts_right_by_constants"),
    ("bufoffany_offset_dropped", RS + "rir/opt.rs", "op.operands[at + 1] = base;\n                            offset = k;",
     "op.operands[at + 1] = base;\n                            offset = 0;", "cpu", "q38::tests::bufoffany_moves_the_constant_into_the_offset_field"),
    ("asm_atomic_returns_dropped", RS + "rir/isel_asm.rs", "Some(d) => MInst::new(opc, [d], uses).with_mods(m | M_GLC),",
     "Some(d) => MInst::new(opc, [d], uses).with_mods(m),", "cpu", "q38::tests::inline_asm_cache_atomics_scalar_loads_and_fixed_vgprs"),
    ("asm_buffer_inv_bits_swapped", RS + "rir/isel_asm.rs", '"sc1" => cpol |= 2,', '"sc1" => cpol |= 1,', "cpu", "q38::tests::inline_asm_cache_atomics_scalar_loads_and_fixed_vgprs"),
    ("asm_sload_glc_ignored", RS + "rir/isel_asm.rs", '"glc" | "sc0" => m |= M_GLC,', '"glc" | "sc0" => {}', "cpu", "q38::tests::inline_asm_cache_atomics_scalar_loads_and_fixed_vgprs"),
    ("asm_fixed_dword_off_by_one", RS + "rir/isel_asm.rs", "Some(&(first, _, reg)) => Ok(Opnd::R(VRef { off: reg.off + (a - first) as u8, n, ..reg })),",
     "Some(&(first, _, reg)) => Ok(Opnd::R(VRef { off: reg.off + (a - first) as u8 + 1, n, ..reg })),", "cpu", "q38::tests::inline_asm_cache_atomics_scalar_loads_and_fixed_vgprs"),
    ("wpecap_no_margin", RS + "rir/mod.rs", "let floor = (512 / (max + 1) + 8).next_multiple_of(8);",
     "let floor = (512 / (max + 1)).next_multiple_of(8);", "cpu", "q38::tests::wpecap_caps_the_occupancy_in_the_descriptor"),
    # ---- Python lifting (the job patches its copy of the tree)
    ("lift_poll_payload_from_tag_word", LIFT,
     'lo = c.op1("vector.extract", [vecs[k]], I32, index=2 * q)', 'lo = c.op1("vector.extract", [vecs[k]], I32, index=2 * q + 1)',
     "py", "b3:r1:mono"),
    ("lift_rowshift_banks_swapped", LIFT, 'row_mask=int(m.group(2), 16), bank_mask=int(m.group(3), 16), bound_ctrl=False,\n        )  # fmt: skip\n        if',
     'row_mask=int(m.group(2), 16), bank_mask=int(m.group(6), 16), bound_ctrl=False,\n        )  # fmt: skip\n        if', "py", "b3:r1:mono"),
    # ---- crate tests (cargo test --lib)
    ("abs_not_encoded", ASM9, "let abs = (mods / super::q38::M_ABS0) & 0x7;", "let abs = 0;", "cpu", "q38::maximum3_takes_abs_modifiers"),
    ("ld_scale_selects_swapped", ASM9,
     "let (op_sel, op_sel_hi) = (sel & 1 | (sel >> 2 & 1) << 1, (sel >> 1 & 1) | (sel >> 3 & 1) << 1);",
     "let (op_sel, op_sel_hi) = ((sel >> 1 & 1) | (sel >> 3 & 1) << 1, sel & 1 | (sel >> 2 & 1) << 1);", "cpu",
     "q38::mx_mfma_takes_formats_and_scale_selects"),
    ("dpp_imm_masks_swapped", Q38, "(ctrl & 0x1ff) | (bound_ctrl as i64) << 11 | (bank_mask & 15) << 16 | (row_mask & 15) << 20",
     "(ctrl & 0x1ff) | (bound_ctrl as i64) << 11 | (row_mask & 15) << 16 | (bank_mask & 15) << 20", "cpu",
     "q38::masked_dpp_keeps_old_lanes"),
    ("constant_loads_not_invariant", RS + "rir/opt.rs", "if invariant {\n                        attrs.push((\"invariant\".into(), Attr::Bool(true)));",
     "if false && invariant {\n                        attrs.push((\"invariant\".into(), Attr::Bool(true)));", "cpu",
     "q38::constant_pointers_load_through_smem"),
    ("sqrt_one_refinement_dropped", ISEL, "let ret = self.v64(Opc::VFmaF64, [r(d1), r(h1), r(s2)], 0);",
     "let ret = s2;", "cpu", "q38::f64_sqrt_is_rsq_and_goldschmidt"),
    ("global_cp_sc0_dropped", ISEL, "    if aux & 1 != 0 {\n        mods |= M_GLC;\n    }", "    if aux & 1 != 0 {\n        mods |= 0;\n    }",
     "cpu", "q38::global_accesses_carry_sc0_sc1_nt"),
    # ---- lever scopes (dlevers)
    ("scope_ignored_by_mir_passes", RS + "rir/mod.rs",
     "s => scopes.get(s as usize).and_then(|d| d.get(name)).unwrap_or_else(|| kernel_lever(name)),",
     "_ => kernel_lever(name),", "cpu", "q38::scopes_reach_what_passes_make"),
    ("scope_not_set_for_isel", ISEL, "let prev = super::set_scope(Some(self.f.scopes[id as usize].clone()));",
     "let prev = super::set_scope(None);", "cpu", "q38::dsrebase_shares_one_base_past_the_offset_field"),
    ("scope_not_entered_by_opt", RS + "rir/opt.rs", "let attr = op.attr(SCOPE_ATTR)?.clone();",
     "let attr = op.attr(\"no-scope\")?.clone();", "cpu", "q38::scoped_levers_apply_to_their_ops_only"),
    ("scopes_not_filled", RS + "rir/mir.rs", "        if self.scopes.is_empty() {\n            return;\n        }\n        let mut last",
     "        if true {\n            return;\n        }\n        let mut last", "cpu", "q38::scopes_reach_what_passes_make"),
    ("loadfirst_never_urgent", RS + "rir/post.rs", "urgent[i] = true;", "urgent[i] = false;", "cpu",
     "q38::loadfirst_issues_loads_once_their_address_is_ready"),
    ("loadfirst_address_not_urgent", RS + "rir/post.rs", "urgent[d] = true;", "urgent[d] = false;", "cpu",
     "q38::loadfirst_issues_loads_once_their_address_is_ready"),
    ("loadfirst_behind_mfma", RS + "rir/post.rs", "return now | (key[i] & 0xffff_ffff) >> 2;", "return now | (key[i] & 0xffff_ffff);",
     "cpu", "post::tests::loadfirst_ranks_a_ready_load_ahead_of_a_ready_mfma"),
    ("source_ops_not_marked", RS + "rir/opt.rs", "vec![(&mut kernel.body, Some(Attr::Str(\"\".into())))]",
     "vec![(&mut kernel.body, None)]", "cpu", "q38::unscoped_source_ops_keep_the_kernels_levers"),
]


def _apply(tree: str, m) -> str:
    name, f, old, new = m[:4]
    path = os.path.join(tree, f)
    text = open(path, encoding="utf-8").read()
    if text.count(old) != 1:
        raise SystemExit(f"mutant {name}: anchor found {text.count(old)} times in {f}")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text.replace(old, new, 1))
    return text


def cpu(tree: str, target: str, only: str | None) -> int:
    crate = os.path.join(tree, "platform/rust/rocke-codeobj")
    env = dict(os.environ, CARGO_TARGET_DIR=target)
    bad = 0
    for m in MUTANTS:
        if m[4] != "cpu" or (only and m[0] != only):
            continue
        path = os.path.join(tree, m[1])
        orig = _apply(tree, m)
        try:
            p = subprocess.run(["cargo", "test", "--release", "--offline", "-j", "2", "--lib", "--", "--test-threads=2",
                                "--skip", "deep_nesting"], cwd=crate, env=env, capture_output=True, text=True)  # fmt: skip
            killed = p.returncode != 0 and m[5].split("::")[-1] in (p.stdout + p.stderr)
            print(f"{m[0]}: {'KILLED' if killed else 'SURVIVED'} ({m[5]})", flush=True)
            bad += not killed
        finally:
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(orig)
    return 1 if bad else 0


def build(tree: str, target: str, out: str) -> int:
    crate = os.path.join(tree, "platform/rust/rocke-codeobj")
    env = dict(os.environ, CARGO_TARGET_DIR=target)
    os.makedirs(out, exist_ok=True)
    bad = 0
    for m in MUTANTS:
        if m[4] != "gpu":
            continue
        path = os.path.join(tree, m[1])
        orig = _apply(tree, m)
        try:
            p = subprocess.run(["cargo", "build", "--release", "--locked", "--offline", "--lib"], cwd=crate, env=env,
                               capture_output=True, text=True)  # fmt: skip
            ok = p.returncode == 0
            if ok:
                shutil.copy(os.path.join(target, "release", "librocke_codeobj_ffi.so"), os.path.join(out, f"mut_{m[0]}.so"))
            print(f"{m[0]}: {'built' if ok else 'BUILD FAILED ' + p.stderr[-400:]}", flush=True)
            bad += not ok
        finally:
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(orig)
    return 1 if bad else 0


def py_patch(tree: str, name: str) -> None:
    """Apply the Python mutant ``name`` to ``tree`` (the GPU job's own copy)."""
    for m in MUTANTS:
        if m[0] == name and m[4] == "py":
            _apply(tree, m)
            return
    raise SystemExit(f"no py mutant {name}")


def main(argv: list[str]) -> int:
    if argv[:1] == ["list"]:
        for m in MUTANTS:
            print(f"{m[0]}\t{m[4]}\t{m[5]}")
        return 0
    if argv[:1] == ["cpu"]:
        only = argv[argv.index("--only") + 1] if "--only" in argv else None
        return cpu(argv[1], argv[2], only)
    if argv[:1] == ["build"]:
        return build(argv[1], argv[2], argv[3])
    if argv[:1] == ["py"]:
        py_patch(argv[1], argv[2])
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
