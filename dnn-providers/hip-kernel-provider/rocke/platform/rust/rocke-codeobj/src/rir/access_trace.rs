// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! The memory accesses one lane of an optimized kernel makes, for checking
//! that two optimizations of one kernel compute the same addresses. `trace`
//! evaluates the kernel's integer and address arithmetic for one lane and
//! lists every global, buffer and LDS access with its address and the name
//! of the value it loads or stores. Floats and vectors are not evaluated:
//! an access is identified by the name its value has in the IR text.
//!
//! Not part of a compile: the unit test below and the `addr_check` example
//! use it.

use super::ast::{Kernel, Op, OpKind, Region, Scalar, Type, ValueId};

/// One lane's view of a launch.
pub struct Inputs {
    /// A value per kernel parameter, in order: an address for a pointer.
    pub params: Vec<i64>,
    /// `gpu.thread_id` by axis (x, y, z).
    pub thread: [i64; 3],
    /// `gpu.block_id` by axis (x, y, z).
    pub block: [i64; 3],
    /// Varies the integers the trace does not compute. An integer from an
    /// op it does not evaluate (a load, a float compare) is a small value
    /// fixed by the result's IR name and this seed: the same in every
    /// optimization of the kernel, whatever memory would hold.
    pub seed: u64,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum AccessKind {
    GlobalLoad,
    GlobalStore,
    BufferLoad,
    LdsLoad,
    LdsStore,
}

/// A memory access: where, and the IR name of the value loaded or stored.
/// `space` is empty for a global access, the buffer's base address and
/// record count for a buffer access, and the allocation's IR name for an LDS
/// access.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct Access {
    pub kind: AccessKind,
    pub space: String,
    pub address: u64,
    pub value: String,
}

#[derive(Clone, Copy)]
enum Value {
    /// Sign-extended from the value's type.
    Int(i64),
    /// A buffer resource: base address and record count.
    Resource(u64, u32),
    Opaque,
}

struct Tracer<'k> {
    kernel: &'k Kernel,
    inputs: &'k Inputs,
    env: Vec<Value>,
    out: Vec<Access>,
    steps: u64,
}

fn bits(ty: &Type) -> Option<u32> {
    match ty {
        Type::Scalar(Scalar::I64) => Some(64),
        Type::Scalar(Scalar::I32) => Some(32),
        Type::Scalar(Scalar::I16) => Some(16),
        Type::Scalar(Scalar::I8) => Some(8),
        Type::Scalar(Scalar::I1) => Some(1),
        _ => None,
    }
}

/// `value` wrapped to `bits` and sign-extended (an i1 is 0 or 1).
fn wrap(value: i128, bits: u32) -> i64 {
    match bits {
        1 => (value & 1) as i64,
        64 => value as i64,
        _ => ((value << (128 - bits)) >> (128 - bits)) as i64,
    }
}

fn unsigned(value: i64, bits: u32) -> u64 {
    if bits == 64 {
        value as u64
    } else {
        value as u64 & ((1 << bits) - 1)
    }
}

fn axis(op: &Op) -> usize {
    match op.str_attr("axis") {
        Some("y") => 1,
        Some("z") => 2,
        _ => 0,
    }
}

impl Tracer<'_> {
    fn int(&self, v: ValueId) -> Result<i64, String> {
        match self.env[v.0 as usize] {
            Value::Int(x) => Ok(x),
            _ => Err(format!("{} is not an evaluated integer", self.kernel.name_of(v))),
        }
    }

    fn width(&self, v: ValueId) -> Result<u32, String> {
        bits(self.kernel.ty(v)).ok_or_else(|| format!("{} is not an integer", self.kernel.name_of(v)))
    }

    fn access(&mut self, kind: AccessKind, space: String, address: u64, value: ValueId) {
        self.out.push(Access { kind, space, address, value: self.kernel.name_of(value).to_string() });
    }

    /// Runs `region`; the operands of its `scf.yield`, if it ends in one.
    fn region(&mut self, region: &Region) -> Result<Option<Vec<Value>>, String> {
        for op in &region.ops {
            self.steps += 1;
            if self.steps > 50_000_000 {
                return Err("the trace runs too long".into());
            }
            if op.kind == OpKind::Yield {
                return Ok(Some(op.operands.iter().map(|v| self.env[v.0 as usize]).collect()));
            }
            self.op(op)?;
        }
        Ok(None)
    }

    /// The value of integer `r` of `bits` bits that no evaluated op gives
    /// (`Inputs::seed`): a bit, or 0..=15.
    fn stand_in(&self, r: ValueId, bits: u32) -> i64 {
        let mut hash = 0xcbf2_9ce4_8422_2325 ^ self.inputs.seed;
        for byte in self.kernel.name_of(r).bytes() {
            hash = (hash ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
        ((hash >> 32) & if bits == 1 { 1 } else { 15 }) as i64
    }

    fn op(&mut self, op: &Op) -> Result<(), String> {
        let k = self.kernel;
        let o = &op.operands;
        let offset = op.int_attr("offset").unwrap_or(0);
        let result = op.results.first().copied();
        let value = match op.kind {
            OpKind::Constant => match (result.and_then(|r| bits(k.ty(r))), op.int_attr("value")) {
                (Some(w), Some(v)) => Value::Int(wrap(v as i128, w)),
                _ => Value::Opaque,
            },
            OpKind::KernArg => {
                let p = op.int_attr("param").unwrap_or(-1);
                Value::Int(*self.inputs.params.get(p as usize).ok_or("a kernel argument without an input")?)
            }
            OpKind::ThreadId => Value::Int(self.inputs.thread[axis(op)]),
            OpKind::BlockId => Value::Int(self.inputs.block[axis(op)]),
            OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Shl => {
                let (r, w) = (result.ok_or("no result")?, self.width(o[0])?);
                let (a, b) = (self.int(o[0])? as i128, self.int(o[1])? as i128);
                let exact = match op.kind {
                    OpKind::Add => a + b,
                    OpKind::Sub => a - b,
                    OpKind::Mul => a * b,
                    _ => a << (b as u32 & 63),
                };
                let wrapped = wrap(exact, w);
                // The result of an overflowing nsw op is poison: no trace.
                let nsw = if op.kind == OpKind::Shl { op.attr("nsw").is_some() } else { op.attr("wrap").is_none() };
                if nsw && wrapped as i128 != exact {
                    return Err(format!("{UNDEFINED}signed overflow at {}", k.name_of(r)));
                }
                Value::Int(wrapped)
            }
            OpKind::Lshr | OpKind::Ashr | OpKind::And | OpKind::Or | OpKind::Xor | OpKind::Smin | OpKind::Smax => {
                let w = self.width(o[0])?;
                let (a, b) = (self.int(o[0])?, self.int(o[1])?);
                Value::Int(match op.kind {
                    OpKind::Lshr => wrap((unsigned(a, w) >> (b as u32 & 63)) as i128, w),
                    OpKind::Ashr => a >> (b as u32 & 63),
                    OpKind::And => a & b,
                    OpKind::Or => a | b,
                    OpKind::Smin => a.min(b),
                    OpKind::Smax => a.max(b),
                    _ => a ^ b,
                })
            }
            // The high word of the 32-bit product.
            OpKind::SmulHi => Value::Int((self.int(o[0])? * self.int(o[1])?) >> 32),
            OpKind::UmulHi => {
                let (a, b) = (unsigned(self.int(o[0])?, 32), unsigned(self.int(o[1])?, 32));
                Value::Int(wrap(((a * b) >> 32) as i128, 32))
            }
            OpKind::Div | OpKind::Mod => {
                let w = self.width(o[0])?;
                let (a, b) = (self.int(o[0])? as i128, self.int(o[1])? as i128);
                if b == 0 {
                    return Err(format!("{UNDEFINED}division by zero"));
                }
                Value::Int(wrap(if op.kind == OpKind::Div { a / b } else { a % b }, w))
            }
            OpKind::Cmp => {
                let w = self.width(o[0])?;
                let (a, b) = (self.int(o[0])?, self.int(o[1])?);
                let (ua, ub) = (unsigned(a, w), unsigned(b, w));
                Value::Int(match op.str_attr("pred").unwrap_or("lt") {
                    "lt" => a < b,
                    "le" => a <= b,
                    "gt" => a > b,
                    "ge" => a >= b,
                    "eq" => a == b,
                    "ne" => a != b,
                    "ult" => ua < ub,
                    "ule" => ua <= ub,
                    "ugt" => ua > ub,
                    "uge" => ua >= ub,
                    other => return Err(format!("compare predicate {other}")),
                } as i64)
            }
            OpKind::Select => self.env[o[if self.int(o[0])? & 1 != 0 { 1 } else { 2 }].0 as usize],
            OpKind::Sext | OpKind::Trunc => {
                let w = self.width(result.ok_or("no result")?)?;
                Value::Int(wrap(self.int(o[0])? as i128, w))
            }
            OpKind::Zext => {
                let w = self.width(result.ok_or("no result")?)?;
                Value::Int(wrap(unsigned(self.int(o[0])?, self.width(o[0])?) as i128, w))
            }
            // The kernels read the first lane of values every lane agrees on.
            OpKind::ReadFirstLane => self.env[o[0].0 as usize],
            OpKind::GlobalPtrAdd | OpKind::GlobalPtrOffset => {
                let step = self.int(o[1])?;
                // An i32 byte offset of `global_ptr_add` is unsigned.
                let step = if *k.ty(o[1]) == Type::Scalar(Scalar::I64) { step } else { unsigned(step, 32) as i64 };
                Value::Int(self.int(o[0])?.wrapping_add(step))
            }
            OpKind::BufferRsrc => Value::Resource(self.int(o[0])? as u64, self.int(o[1])? as u32),
            OpKind::BufferLoadVec => {
                let Value::Resource(base, records) = self.env[o[0].0 as usize] else {
                    return Err("a buffer load without a resource".into());
                };
                // The hardware adds the offsets in 32 bits.
                let address = (self.int(o[1])? as u32).wrapping_add(offset as u32).wrapping_add(self.int(o[2])? as u32);
                self.access(AccessKind::BufferLoad, format!("{base:#x}+{records}"), address as u64, result.ok_or("no result")?);
                Value::Opaque
            }
            OpKind::GLoad | OpKind::GStore => {
                let store = op.kind == OpKind::GStore;
                let address = &o[..o.len() - store as usize];
                let mut sum = self.int(address[0])?.wrapping_add(offset);
                if let Some(&v) = address.get(1) {
                    sum = sum.wrapping_add(unsigned(self.int(v)?, 32) as i64);
                }
                let (kind, value) = if store {
                    (AccessKind::GlobalStore, *o.last().ok_or("a store without a value")?)
                } else {
                    (AccessKind::GlobalLoad, result.ok_or("no result")?)
                };
                self.access(kind, String::new(), sum as u64, value);
                Value::Opaque
            }
            OpKind::LLoad | OpKind::LStore => {
                let address = (self.int(o[1])? as u32).wrapping_add(offset as u32) as u64;
                let (kind, value) = if op.kind == OpKind::LStore {
                    (AccessKind::LdsStore, *o.last().ok_or("a store without a value")?)
                } else {
                    (AccessKind::LdsLoad, result.ok_or("no result")?)
                };
                self.access(kind, k.name_of(o[0]).to_string(), address, value);
                Value::Opaque
            }
            OpKind::If | OpKind::IfElse => {
                let taken = if self.int(o[0])? & 1 != 0 { op.regions.first() } else { op.regions.get(1) };
                if let Some(region) = taken {
                    self.region(region)?;
                }
                return Ok(());
            }
            OpKind::For => {
                let (mut iv, end, step) = (self.int(o[0])?, self.int(o[1])?, self.int(o[2])?);
                if step <= 0 {
                    return Err("a loop that does not count up".into());
                }
                let mut carried: Vec<Value> = o[3..].iter().map(|v| self.env[v.0 as usize]).collect();
                let body = op.regions.first().ok_or("a loop without a body")?;
                while iv < end {
                    self.env[op.block_args[0].0 as usize] = Value::Int(iv);
                    for (arg, value) in op.block_args[1..].iter().zip(&carried) {
                        self.env[arg.0 as usize] = *value;
                    }
                    if let Some(yielded) = self.region(body)? {
                        carried = yielded;
                    }
                    iv += step;
                }
                for (result, value) in op.results.iter().zip(&carried) {
                    self.env[result.0 as usize] = *value;
                }
                return Ok(());
            }
            _ => {
                if !op.regions.is_empty() {
                    return Err(format!("{:?} is not traced", op.kind));
                }
                Value::Opaque
            }
        };
        for &r in &op.results {
            self.env[r.0 as usize] = match (value, bits(k.ty(r))) {
                (Value::Opaque, Some(w)) => Value::Int(self.stand_in(r, w)),
                _ => value,
            };
        }
        Ok(())
    }
}

/// Starts the error of a launch the IR gives no meaning: an nsw op
/// overflows (its result is poison), or a division is by zero.
pub const UNDEFINED: &str = "undefined: ";

/// The accesses of one lane of `kernel`, an optimized kernel
/// (`opt::optimize`), sorted: optimizations keep every access, and may
/// order the loads differently. `Err` for a launch without a meaning
/// (`UNDEFINED`), or when an address depends on an op this module does not
/// evaluate.
pub fn trace(kernel: &Kernel, inputs: &Inputs) -> Result<Vec<Access>, String> {
    let mut tracer = Tracer { kernel, inputs, env: vec![Value::Opaque; kernel.values.len()], out: Vec::new(), steps: 0 };
    for (param, &value) in kernel.params.iter().zip(&inputs.params) {
        tracer.env[param.value.0 as usize] = Value::Int(value);
    }
    tracer.region(&kernel.body)?;
    tracer.out.sort();
    Ok(tracer.out)
}

/// `text` optimized for a wave of `wave` lanes, with or without the address
/// reductions (`Kernel::address_reduction`).
pub fn optimized(text: &str, wave: u32, address_reduction: bool) -> Result<Kernel, String> {
    let mut kernel = super::parse::parse(text).map_err(|e| e.to_string())?;
    kernel.wave = wave;
    kernel.address_reduction = address_reduction;
    let (mut lds, _) = super::analysis::lds_layout(&kernel);
    super::opt::optimize(&mut kernel, &mut lds).map_err(|e| e.to_string())?;
    Ok(kernel)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tile store as the GEMM kernels write it: four rows of two column
    /// tiles, through a buffer-sized i32 index in one branch and through
    /// `global_ptr_offset` with i64 arithmetic in the other, after a load
    /// whose buffer offset is the low word of an i64 product.
    fn tile_kernel() -> String {
        let mut body = String::from(
            r#"    %c0 : i32 = arith.constant (  ) { ity = s:"i32", value = i:0 }
    %c4 : i64 = arith.constant (  ) { ity = s:"i64", value = i:4 }
    %c16 : i32 = arith.constant (  ) { ity = s:"i32", value = i:16 }
    %c1024 : i32 = arith.constant (  ) { ity = s:"i32", value = i:1024 }
    %row : i32 = arith.div ( %tid, %c16 )
    %col : i32 = arith.mod ( %tid, %c16 )
    %sn : i64 = arith.sext ( %N )
    %srow : i64 = arith.sext ( %row )
    %scol : i64 = arith.sext ( %col )
    %wide : i64 = arith.mul ( %scol, %sn )
    %wide2 : i64 = arith.add ( %wide, %srow )
    %bytes : i64 = arith.mul ( %wide2, %c4 )
    %voff : i32 = arith.trunc ( %bytes )
    %small : i1 = arith.cmp ( %N, %c1024 ) { pred = s:"le" }
    scf.if_else ( %small )
      region @then {
        %rs : vec<i32x4> = tile.buffer_rsrc ( %A, %c1024 )
        %bl : vec<f32x4> = tile.buffer_load_vN ( %rs, %voff, %c0 ) { dwords = i:4, elem_type = s:"f32" }
"#,
        );
        for tile in 0..2 {
            for j in 0..4 {
                let n = tile * 4 + j;
                body += &format!(
                    "        %j{n} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{j} }}\n        \
                     %t{n} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n        \
                     %r{n} : i32 = arith.add ( %row, %j{n} )\n        %m{n} : i32 = arith.mul ( %r{n}, %N )\n        \
                     %k{n} : i32 = arith.add ( %col, %t{n} )\n        %i{n} : i32 = arith.add ( %m{n}, %k{n} )\n        \
                     %e{n} : f32 = vector.extract ( %bl ) {{ index = i:{j} }}\n        \
                     memref.global_store_typed ( %C, %i{n}, %e{n} ) {{ align = i:4, elem_type = s:\"f32\" }}\n",
                    tile * 16
                );
            }
        }
        body += "      }\n      region @else {\n        %f : f32 = arith.bitcast ( %tid )\n";
        for tile in 0..2 {
            for j in 0..4 {
                let n = tile * 4 + j;
                body += &format!(
                    "        %j{n} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{j} }}\n        \
                     %t{n} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n        \
                     %r{n} : i32 = arith.add ( %row, %j{n} )\n        %x{n} : i64 = arith.sext ( %r{n} )\n        \
                     %m{n} : i64 = arith.mul ( %x{n}, %sn )\n        %k{n} : i32 = arith.add ( %col, %t{n} )\n        \
                     %y{n} : i64 = arith.sext ( %k{n} )\n        %i{n} : i64 = arith.add ( %m{n}, %y{n} )\n        \
                     %b{n} : i64 = arith.mul ( %i{n}, %c4 )\n        \
                     %p{n} : ptr<f32,global> = tile.global_ptr_offset ( %C, %b{n} )\n        \
                     memref.global_store_typed ( %p{n}, %c0, %f ) {{ align = i:4, elem_type = s:\"f32\" }}\n",
                    tile * 16
                );
            }
        }
        body += "      }\n";
        format!(
            "ckdsl.ir v1\nkernel @probe {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    %A : ptr<f32,global>\n    \
             %C : ptr<f32,global>\n    %N : i32\n  }}\n  region @entry {{\n    \
             %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n{body}    cf.return (  )\n  }}\n}}\n"
        )
    }

    #[test]
    fn address_reductions_keep_every_access() {
        let text = tile_kernel();
        let plain = optimized(&text, 32, false).unwrap();
        let reduced = optimized(&text, 32, true).unwrap();
        // Both branches: the i32 index up to N = 1024, i64 offsets above.
        for n in [100, 1024, 40_000, 3_000_000] {
            for tid in [0, 1, 17, 63] {
                let params = vec![0x1000_0000_0000, 0x7fff_f000_0000, n];
                let inputs = Inputs { params, thread: [tid, 0, 0], block: [0; 3], seed: 0 };
                let want = trace(&plain, &inputs).unwrap_or_else(|e| panic!("N = {n}, lane {tid}: {e}"));
                assert_eq!(trace(&reduced, &inputs).as_ref(), Ok(&want), "N = {n}, lane {tid}");
                assert_eq!(want.iter().filter(|a| a.kind == AccessKind::GlobalStore).count(), 8);
            }
        }
        // The check compares two different kernels: the reductions take
        // 64-bit multiplies out (one per row, and the one under the buffer
        // offset).
        let wide_multiplies = |k: &Kernel| {
            let mut n = 0;
            super::super::analysis::preorder(&k.body, |op| {
                let wide = op.results.first().is_some_and(|&r| *k.ty(r) == Type::Scalar(Scalar::I64));
                n += (wide && op.kind == OpKind::Mul) as usize;
            });
            n
        };
        assert!(wide_multiplies(&reduced) < wide_multiplies(&plain), "{} of {}", wide_multiplies(&reduced), wide_multiplies(&plain));
    }

    #[test]
    fn an_overflowing_index_has_no_trace() {
        // With the branches swapped, row 3 of 2^30 columns overflows the
        // i32 index.
        let kernel = optimized(&tile_kernel().replace("pred = s:\"le\"", "pred = s:\"ge\""), 32, false).unwrap();
        let inputs = Inputs { params: vec![0, 0, 1 << 30], thread: [63, 0, 0], block: [0; 3], seed: 0 };
        assert!(trace(&kernel, &inputs).is_err_and(|e| e.starts_with(UNDEFINED)));
    }

    /// LDS loads of four rows 16 apart, as a GEMM kernel reads a staged
    /// tile: each column is xor'ed with a swizzle term of its row.
    fn swizzled_kernel() -> String {
        let mut body = String::from(
            r#"    %c1 : i32 = arith.constant (  ) { ity = s:"i32", value = i:1 }
    %c3 : i32 = arith.constant (  ) { ity = s:"i32", value = i:3 }
    %c4 : i32 = arith.constant (  ) { ity = s:"i32", value = i:4 }
    %row : i32 = arith.div ( %tid, %c4 )
    %col : i32 = arith.mod ( %tid, %c4 )
    %S : smem<f16, [128x32]> = tile.smem_alloc (  )
"#,
        );
        for k in 0..4 {
            body += &format!(
                "    %o{k} : i32 = arith.constant (  ) {{ ity = s:\"i32\", value = i:{} }}\n    \
                 %r{k} : i32 = arith.add ( %row, %o{k} )\n    \
                 %h{k} : i32 = arith.lshr ( %r{k}, %c1 )\n    %m{k} : i32 = arith.mod ( %h{k}, %c4 )\n    \
                 %s{k} : i32 = arith.shl ( %m{k}, %c3 )\n    %x{k} : i32 = arith.xor ( %col, %s{k} )\n    \
                 %v{k} : vec<f16x8> = tile.smem_load_vN ( %S, %r{k}, %x{k} ) {{ elem_type = s:\"f16\", rank = i:2, vec = i:8 }}\n    \
                 %e{k} : f16 = vector.extract ( %v{k} ) {{ index = i:0 }}\n    %i{k} : i32 = arith.add ( %tid, %o{k} )\n    \
                 memref.global_store_typed ( %Y, %i{k}, %e{k} ) {{ align = i:2, elem_type = s:\"f16\" }}\n",
                16 * k
            );
        }
        format!(
            "ckdsl.ir v1\nkernel @probe {{\n  attrs {{ max_workgroup_size = i:64 }}\n  params {{\n    %Y : ptr<f16,global>\n  }}\n  \
             region @entry {{\n    %tid : i32 = gpu.thread_id (  ) {{ axis = s:\"x\" }}\n{body}    cf.return (  )\n  }}\n}}\n"
        )
    }

    #[test]
    fn rows_sixteen_apart_share_their_swizzle_term_and_lds_base() {
        let text = swizzled_kernel();
        let plain = optimized(&text, 32, false).unwrap();
        let reduced = optimized(&text, 32, true).unwrap();
        for tid in [0, 3, 9, 22, 63] {
            let inputs = Inputs { params: vec![0x1000_0000], thread: [tid, 0, 0], block: [0; 3], seed: 0 };
            let want = trace(&plain, &inputs).unwrap_or_else(|e| panic!("lane {tid}: {e}"));
            assert_eq!(trace(&reduced, &inputs).as_ref(), Ok(&want), "lane {tid}");
            assert_eq!(want.iter().filter(|a| a.kind == AccessKind::LdsLoad).count(), 4);
        }
        // The swizzle terms, and the base addresses the loads add their
        // offsets to.
        let shape = |k: &Kernel| {
            let (mut terms, mut bases) = (0, Vec::new());
            super::super::analysis::preorder(&k.body, |op| {
                terms += (op.kind == OpKind::Xor) as usize;
                if op.kind == OpKind::LLoad {
                    bases.push(op.operands[1]);
                }
            });
            bases.sort();
            bases.dedup();
            (terms, bases.len())
        };
        assert_eq!((shape(&plain), shape(&reduced)), ((4, 4), (1, 1)));
    }
}
