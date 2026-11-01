"""Mutation check of the R2 gates: each mutant breaks one piece of the R2
lowering; its gate must FAIL on it (and pass on the restored tree).

    python r2_mutants.py cpu <rust-tree> <cargo-target-dir> [--only NAME]
        local: apply each CPU mutant, `cargo test --lib` (2 jobs, 2 test
        threads) must fail; restores every file in a finally
    python r2_mutants.py build <rust-tree> <cargo-target-dir> <out-dir>
        cluster CPU job: for each GPU mutant, `cargo build --lib` and keep
        `<out-dir>/mut_<name>.so` (restored after each)
    python r2_mutants.py list

GPU mutants are run by the GPU job (``r2_gpu.py numerics`` with
``ROCKE_CODEOBJ_DIRECT_LIB=mut_<name>.so`` must fail, under a watchdog: a
broken poll can spin forever).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

RS = "platform/rust/rocke-codeobj/src/"
ISEL = RS + "rir/isel.rs"

# (name, file, old, new, kind, what fails)
MUTANTS = [
    # ---- hardware gates (numerics: bitwise vs COMGR / host model)
    ("poll_compares_payload", ISEL, "            for lane in (1..w).step_by(2) {", "            for lane in (0..w).step_by(2) {",
     "gpu", "toy replay / head fused argmax (the poll compares payload words with the tag)"),
    ("cvt_byte_swapped", ISEL,
     "let mods = if byte & 1 != 0 { M_OPSEL_SRC0 } else { 0 } | if byte & 2 != 0 { M_OPSEL_SRC1 } else { 0 };",
     "let mods = if byte & 1 != 0 { M_OPSEL_SRC1 } else { 0 } | if byte & 2 != 0 { M_OPSEL_SRC0 } else { 0 };",
     "gpu", "head logits bitwise (bytes 1 and 2 swapped)"),
    ("permlane16_results_swapped", ISEL,
     "                self.set(op.results[0], Val::W(r(x)));\n                self.set(op.results[1], Val::W(r(y)));\n            }\n            SGetreg => {",
     "                self.set(op.results[0], Val::W(r(y)));\n                self.set(op.results[1], Val::W(r(x)));\n            }\n            SGetreg => {",
     "gpu", "head logits bitwise"),
    ("preload_sgprs_shifted", ISEL, "isel.f.vregs[fixed.reg as usize].fixed = Some(2 + first as u16);",
     "isel.f.vregs[fixed.reg as usize].fixed = Some(4 + first as u16);", "gpu", "toy replay (preloaded arguments read from the wrong SGPRs)"),
    ("int_to_ptr_halves_swapped", ISEL,
     "            IntToPtr => {\n                let (lo, hi) = self.d(op.operands[0])?;\n                self.set(res.unwrap(), Val::D(lo, hi));",
     "            IntToPtr => {\n                let (lo, hi) = self.d(op.operands[0])?;\n                self.set(res.unwrap(), Val::D(hi, lo));",
     "gpu", "toy replay"),
    ("poll_exec_not_narrowed", ISEL,
     "        self.salu(self.lm(Opc::SAndB32), [self.exec()], [self.exec(), waiting]);\n        self.emit(MInst::new(Opc::SCbranchExecz, [], []));",
     "        let any = self.mreg();\n        self.salu(self.lm(Opc::SAndB32), [r(any)], [self.exec(), waiting]);\n        self.emit(MInst::new(Opc::SCbranchExecz, [], []));",
     "gpu", "toy replay / head (the poll never leaves: watchdog)"),
    # ---- crate tests (cargo test --lib)
    ("cache_policy_sc0_sc1_swapped", ISEL, "    if aux & 1 != 0 {\n        mods |= M_GLC;\n    }",
     "    if aux & 1 != 0 {\n        mods |= crate::target::gfx9::cpol::M_SC1;\n    }", "cpu", "q38::buffer_accesses_carry_their_cache_policy"),
    ("atomic_system_scope_without_sc1", ISEL,
     '        if op.str_attr("scope") == Some("system") {\n            mods |= crate::target::gfx9::cpol::M_SC1;\n        }',
     '        if op.str_attr("scope") == Some("system") {\n            mods |= 0;\n        }', "cpu", "q38::scoped_atomics_return_with_sc0_and_scope_sc1"),
    ("storewait_counts_loads", RS + "target/gfx9/waitcnt.rs", "if lds_dma::is_dma(inst.op) || flags & F_VMEM_STORE != 0 {",
     "if lds_dma::is_dma(inst.op) || flags & (F_VMEM_STORE | F_VMEM_LOAD) != 0 {", "cpu", "q38::storewait_barriers_wait_for_stores_only"),
    ("l2_touch_not_held", ISEL,
     "                    self.emit(MInst::new(Opc::Gfx9(crate::target::gfx9::gen_isa::Gfx9Op::KeepAlive), [], Uses::from(chunk)));",
     "                    let _ = chunk;", "cpu", "q38::l2_touch_loads_stay_in_flight_until_the_drain"),
    ("preload_entry_offsets", RS + "target/gfx9/mod.rs", ".with_imm(4 * k as i64);", ".with_imm(4 * k as i64 + 4);",
     "cpu", "q38::the_preload_entry_is_comgrs"),
    ("descriptor_without_preload", RS + "hsaco_writer.rs", "    put_u16(&mut descriptor, 58, spec.kernarg_preload);",
     "    let _ = spec.kernarg_preload;", "cpu", "q38::preloaded_arguments_live_in_user_sgprs"),
    ("sgpr_pin_of_divergent_accepted", RS + "rir/validate.rs",
     "            OpKind::InlineAsmIdentity if op.asm_pin() == Some('s') && !op.operands.first().is_some_and(uni) => {",
     "            OpKind::InlineAsmIdentity if op.asm_pin() == Some('s') && false => {", "cpu",
     "validate::qwen38_device_ops_lower_on_gfx950_only"),
    ("poll_refused_again", RS + "rir/ast.rs", '            "tile.tagged_poll" => OpKind::TaggedPoll,\n', "", "cpu",
     "validate::qwen38_device_ops_lower_on_gfx950_only"),
]

CARGO = os.path.expandvars(r"%USERPROFILE%\.cargo\bin\cargo.exe") if os.name == "nt" else "cargo"


def apply(tree: str, m) -> tuple[str, bytes]:
    _, rel, old, new, *_ = m
    path = os.path.join(tree, rel)
    raw = open(path, "rb").read()
    text = raw.decode("utf-8")
    if text.count(old) != 1:
        raise SystemExit(f"{m[0]}: anchor occurs {text.count(old)} times in {rel}")
    open(path, "wb").write(text.replace(old, new).encode("utf-8"))
    return path, raw


def main(argv: list[str]) -> int:
    cmd = argv[0]
    if cmd == "list":
        for m in MUTANTS:
            print(f"{m[0]:<34} {m[4]:<4} {m[5]}")
        return 0
    tree, target = argv[1], argv[2]
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    crate = os.path.join(tree, "platform/rust/rocke-codeobj")
    env = dict(os.environ, CARGO_TARGET_DIR=target, RUST_TEST_THREADS="2")
    results = {}
    for m in MUTANTS:
        name, kind = m[0], m[4]
        if only and name != only:
            continue
        if (cmd == "cpu") != (kind == "cpu"):
            continue
        path, raw = apply(tree, m)
        try:
            if cmd == "cpu":
                p = subprocess.run([CARGO, "test", "--release", "--locked", "--offline", "--lib", "-j", "2", "-q"],
                                   cwd=crate, env=env, capture_output=True, text=True)  # fmt: skip
                failed = p.returncode != 0
                hit = m[5].split("::")[-1] in (p.stdout + p.stderr)
                results[name] = "KILLED" if failed and hit else ("KILLED (other test)" if failed else "SURVIVED")
            else:
                p = subprocess.run([CARGO, "build", "--release", "--locked", "--offline", "--lib"], cwd=crate, env=env,
                                   capture_output=True, text=True)  # fmt: skip
                if p.returncode != 0:
                    results[name] = "BUILD FAILED: " + p.stderr[-300:]
                else:
                    lib = "rocke_codeobj_ffi.dll" if os.name == "nt" else "librocke_codeobj_ffi.so"
                    shutil.copy(os.path.join(target, "release", lib), os.path.join(argv[3], f"mut_{name}.so"))
                    results[name] = "built"
        finally:
            open(path, "wb").write(raw)
        print(f"{name}: {results[name]}", flush=True)
    if cmd == "cpu":
        p = subprocess.run([CARGO, "test", "--release", "--locked", "--offline", "--lib", "-j", "2", "-q"], cwd=crate, env=env,
                           capture_output=True, text=True)  # fmt: skip
        print("restored tree:", "PASS" if p.returncode == 0 else "FAIL")
        return 0 if p.returncode == 0 and all(v.startswith("KILLED") for v in results.values()) else 1
    return 0 if all(v == "built" for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
