// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Builds the proposed, unwired `src/rir/model.rs` against the crate's machine
//! IR and prints its class table: `cargo run --release --example model_check`.
//! `cargo test --release --example model_check` runs the model's own tests.
//! Nothing in the compile path uses the model.

use rocke_codeobj_ffi::rir::mir;

#[allow(dead_code)]
#[path = "../src/rir/model.rs"]
mod model;

use mir::{MInst, Opc};
use model::{issue_cycles, result_latency, unit};

fn main() {
    let ops = [
        Opc::SAddU32,
        Opc::SLoadB64,
        Opc::VAddF32,
        Opc::VFmaF32,
        Opc::VMulLoU32,
        Opc::VLshlrevB64,
        Opc::VFmaF64,
        Opc::VExpF32,
        Opc::VMovB32Dpp,
        Opc::VWmmaF32F16,
        Opc::DsLoadB32,
        Opc::DsLoadB128,
        Opc::DsBpermuteB32,
        Opc::GLoadB128,
        Opc::GStoreB128,
    ];
    for op in ops {
        let inst = MInst::new(op, [], []);
        println!("{:<28} {:<12} latency {:>4} issue {:>3}", op.name(), format!("{:?}", unit(op)),
                 result_latency(&inst), issue_cycles(&inst));
    }
}
