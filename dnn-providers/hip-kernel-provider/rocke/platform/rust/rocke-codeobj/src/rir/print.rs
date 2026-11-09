// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Debug printer for the direct backend's AST.

use std::fmt::Write;

use super::ast::{Attr, Kernel, Region};

fn attr(value: &Attr) -> String {
    match value {
        Attr::Bool(v) => v.to_string(),
        Attr::Int(v) => v.to_string(),
        Attr::Float(v) => format!("{v:?}"),
        Attr::Str(v) => format!("{v:?}"),
        Attr::List(items) => format!("[{}]", items.iter().map(attr).collect::<Vec<_>>().join(", ")),
        Attr::Map(items) => format!(
            "{{{}}}",
            items
                .iter()
                .map(|(k, v)| format!("{k}={}", attr(v)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

fn region(kernel: &Kernel, body: &Region, depth: usize, out: &mut String) {
    for op in &body.ops {
        let pad = "  ".repeat(depth);
        let results = op
            .results
            .iter()
            .map(|r| format!("{}#{} : {:?}", kernel.name_of(*r), r.0, kernel.ty(*r)))
            .collect::<Vec<_>>()
            .join(", ");
        let operands = op
            .operands
            .iter()
            .map(|v| format!("{}#{}", kernel.name_of(*v), v.0))
            .collect::<Vec<_>>()
            .join(", ");
        let attrs = op
            .attrs
            .iter()
            .filter(|(k, _)| k != "iter_args")
            .map(|(k, v)| format!("{k}={}", attr(v)))
            .collect::<Vec<_>>()
            .join(" ");
        let lhs = if results.is_empty() { String::new() } else { format!("{results} = ") };
        let _ = writeln!(out, "{pad}{lhs}{:?}({operands}) {attrs}", op.kind);
        if !op.block_args.is_empty() {
            let args = op
                .block_args
                .iter()
                .map(|v| format!("{}#{}", kernel.name_of(*v), v.0))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = writeln!(out, "{pad}  ^args({args})");
        }
        for inner in &op.regions {
            let _ = writeln!(out, "{pad}  {{");
            region(kernel, inner, depth + 2, out);
            let _ = writeln!(out, "{pad}  }}");
        }
    }
}

pub fn dump(kernel: &Kernel) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "kernel {} (max_wg {})", kernel.name, kernel.max_workgroup_size);
    for param in &kernel.params {
        let _ = writeln!(out, "  param {}#{} : {:?}", param.name, param.value.0, param.ty);
    }
    region(kernel, &kernel.body, 1, &mut out);
    out
}

pub fn op_count(body: &Region) -> usize {
    body.ops.iter().map(|op| 1 + op.regions.iter().map(op_count).sum::<usize>()).sum()
}
