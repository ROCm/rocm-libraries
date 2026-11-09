// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! `tile.iglp_opt`, LLVM's instruction-ordering strategies for MFMA loops
//! (G2 design section 8). The op only orders instructions, so instruction
//! selection may emit nothing for it.

#[cfg(test)]
mod tests {
    use crate::rir;
    use crate::target::Target;

    /// A loop of `n` iterations reading X and accumulating into Y, with
    /// `directive` first in its body.
    fn kernel(directive: &str) -> String {
        format!(
            "ckdsl.ir v1\nkernel @k {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    %X : ptr<f32,global>\n    \
             %Y : ptr<f32,global>\n    %n : i32\n  }}\n  region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
             %c0 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:0 }}\n    \
             %c1 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:1 }}\n    \
             %c64 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:64 }}\n    \
             %z : f32 = arith.constant (  ) {{ ity = s:\"f32\", value = f:0.0 }}\n    \
             %s : f32 = scf.for ( %c0, %n, %c1, %z ) {{ iter_args = l:[ {{ name = s:\"%a\", type = s:\"f32\" }} ], iv = s:\"%k\", \
             iv_type = s:\"i32\", num_iter_args = i:1, unroll = b:false }}\n      region @body {{\n{directive}        \
             %o : i32 = arith.mul ( %k, %c64 )\n        %p : i32 = arith.add ( %o, %tid )\n        \
             %x : f32 = memref.global_load_typed ( %X, %p ) {{ align = i:4, elem_type = s:\"f32\" }}\n        \
             %b : f32 = arith.fadd ( %a, %x )\n        scf.yield ( %b ) {{ num = i:1 }}\n      }}\n    \
             memref.global_store_typed ( %Y, %tid, %s ) {{ align = i:4, elem_type = s:\"f32\" }}\n    cf.return (  )\n  }}\n}}\n"
        )
    }

    #[test]
    fn iglp_opt_compiles_to_what_the_kernel_gives_without_it() {
        for target in Target::ALL {
            let plain = rir::compile_any_target(&kernel(""), target).unwrap_or_else(|e| panic!("{e}"));
            for level in 0..=3 {
                let text = kernel(&format!("        tile.iglp_opt (  ) {{ level = i:{level} }}\n"));
                let with = rir::compile_any_target(&text, target).unwrap_or_else(|e| panic!("{e}"));
                assert!(with.hsaco == plain.hsaco, "{target:?} level {level}");
            }
        }
    }

    #[test]
    fn iglp_opt_needs_a_strategy_level() {
        for directive in [
            "        tile.iglp_opt (  )\n",
            "        tile.iglp_opt (  ) { level = i:4 }\n",
            "        tile.iglp_opt (  ) { level = i:-1 }\n",
            "        tile.iglp_opt (  ) { level = s:\"0\" }\n",
            "        tile.iglp_opt ( %k ) { level = i:0 }\n",
        ] {
            let err = rir::compile_any_target(&kernel(directive), Target::Gfx950).err().unwrap_or_else(|| panic!("accepted: {directive}"));
            assert_eq!(err.kind, "ROCKE_DIRECT_MALFORMED", "{}\n{directive}", err.message);
        }
    }
}
