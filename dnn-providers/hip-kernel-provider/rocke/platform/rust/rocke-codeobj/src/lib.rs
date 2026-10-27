// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! Direct rocKE IR to HSACO compilation.
//!
//! The gfx1151 WMMA path performs IR matching, AMDGPU encoding, and ELF/AMDHSA
//! writing in Rust. No LLVM, Clang, LLD, COMGR, or external tool participates.

pub mod direct_gemm;
pub mod elf;
pub mod error;
mod gfx1151_isa;
mod hsaco_writer;
pub mod native;
pub mod rir;
pub mod target;

pub use direct_gemm::{
    compile_gfx1151_wmma_gemm_ir, emit_gfx1151_wmma_gemm, parse_gfx1151_wmma_gemm_ir,
    DirectCompilation, DIRECT_BACKEND_VERSION, DIRECT_GFX1151_WMMA_GEMM_SYMBOL,
};
pub use elf::{validate_hsaco, ElfFacts};
pub use error::CompileError;
