// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Constants used before their definition (gfx950): some census kernels
//! (dsa_indexer) define an `arith.constant` after the region that reads it.
//! The LLVM lowering writes a constant's value into each use instead of an
//! instruction, so COMGR never sees the order; the parser here resolves names
//! in text order and refuses it. When it does, `reorder` moves each such
//! constant's line to the start of the kernel's body, which every use follows,
//! and the text parses again. A constant has no operands, so the move changes
//! no value.

/// `text` with each `arith.constant` line that follows a use of its value
/// moved to the start of the kernel's body; `None` when there is none.
pub fn reorder(text: &str) -> Option<String> {
    let lines: Vec<&str> = text.split_inclusive('\n').collect();
    let entry = lines.iter().position(|l| l.trim() == "region @entry {")?;
    let mut moved = Vec::new();
    for (i, line) in lines.iter().enumerate().skip(entry + 1) {
        let Some((name, rest)) = line.trim_start().split_once(" : ") else { continue };
        if !name.starts_with('%') || !rest.split_once(" = ").is_some_and(|(_, op)| op.starts_with("arith.constant (")) {
            continue;
        }
        if lines[entry + 1..i].iter().any(|l| uses(l, name)) {
            moved.push(i);
        }
    }
    if moved.is_empty() {
        return None;
    }
    let mut out = String::with_capacity(text.len());
    for (i, line) in lines.iter().enumerate() {
        if moved.contains(&i) {
            continue;
        }
        out.push_str(line);
        if i == entry {
            for &m in &moved {
                out.push_str("    ");
                out.push_str(lines[m].trim_start());
            }
        }
    }
    Some(out)
}

/// Whether `line` names `value` as a whole token.
fn uses(line: &str, value: &str) -> bool {
    let token = |c: char| c.is_alphanumeric() || matches!(c, '_' | '.' | '$' | '-');
    line.match_indices(value).any(|(at, _)| !line[at + value.len()..].starts_with(token))
}

#[cfg(test)]
mod tests {
    use crate::rir;
    use crate::target::Target;

    /// `%c` read in a guarded region; defined first, or (`late`) after it.
    fn kernel(late: bool) -> String {
        let c = "    %c : i32 = arith.constant (  ) { ity = s:\"i32\", value = i:7 }\n";
        format!(
            "ckdsl.ir v1\nkernel @early {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    %Y : ptr<i32,global>\n    %n : i32\n  }}\n  \
             region @entry {{\n{}    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n    \
             %lt : i1 = arith.cmp ( %tid, %n ) {{ pred = s:\"lt\" }}\n    scf.if ( %lt )\n      region @then {{\n        \
             %s : i32 = arith.select ( %lt, %tid, %c )\n        \
             memref.global_store_typed ( %Y, %tid, %s ) {{ align = i:4, elem_type = s:\"i32\" }}\n      }}\n{}    \
             %c70 : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:70 }}\n    cf.return (  )\n  }}\n}}\n",
            if late { "" } else { c },
            if late { c } else { "" }
        )
    }

    #[test]
    fn constants_used_before_their_definition_move_to_the_start() {
        assert_eq!(super::reorder(&kernel(true)).as_deref(), Some(kernel(false).as_str()));
        assert_eq!(super::reorder(&kernel(false)), None);
        let first = rir::compile_any_target(&kernel(false), Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        let late = rir::compile_any_target(&kernel(true), Target::Gfx950).unwrap_or_else(|e| panic!("{e}"));
        assert!(late.hsaco == first.hsaco);
        let e = rir::compile_any_target(&kernel(true), Target::Gfx1151).err().map(|e| e.to_string()).unwrap_or_default();
        assert!(e.contains("used before definition"), "{e}");
    }
}
