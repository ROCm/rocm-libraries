// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

//! The i32 ring form shared by `reassoc` and `voff_split`: the value of an i32 expression as a polynomial of
//! degree <= 2 over its leaves, modulo 2^32 (add, sub, mul and shl are ring operations), the walker that reads it
//! off the definitions, and the emitter that rebuilds a sum of its terms as `wrap = "modular"` ops.
//!
//! The walker's `Limits` pick the use: `reassoc` reads linear forms (a product of two non-constants is a leaf) and
//! gives up past a few leaves; `voff_split` reads degree-2 forms (a leaf times a uniform stride) and turns a
//! too-large sub-expression into a leaf.

use super::ast::*;

#[derive(Clone)]
pub(super) struct Def {
    pub(super) kind: OpKind,
    pub(super) operands: Vec<ValueId>,
    pub(super) konst: Option<i64>,
}

/// A monomial: the sorted leaf ids of one product (one leaf for a linear term, two for a degree-2 one).
pub(super) type Mono = Vec<u32>;

/// A polynomial of degree <= 2 over leaves: (monomial, coefficient) terms sorted by monomial, and a constant, modulo
/// 2^32.
#[derive(Clone, Debug, Default, PartialEq)]
pub(super) struct Form {
    pub(super) terms: Vec<(Mono, u32)>,
    pub(super) konst: u32,
}

impl Form {
    pub(super) fn leaf(v: ValueId) -> Form {
        Form { terms: vec![(vec![v.0], 1)], konst: 0 }
    }

    fn norm(mut self) -> Form {
        self.terms.sort();
        let mut out: Vec<(Mono, u32)> = Vec::new();
        for (k, c) in self.terms {
            match out.last_mut() {
                Some(last) if last.0 == k => last.1 = last.1.wrapping_add(c),
                _ => out.push((k, c)),
            }
        }
        out.retain(|t| t.1 != 0);
        self.terms = out;
        self
    }

    pub(super) fn add(mut self, other: &Form, sign: u32) -> Form {
        self.terms.extend(other.terms.iter().map(|(k, c)| (k.clone(), c.wrapping_mul(sign))));
        self.konst = self.konst.wrapping_add(other.konst.wrapping_mul(sign));
        self.norm()
    }

    pub(super) fn scale(mut self, k: u32) -> Form {
        for t in &mut self.terms {
            t.1 = t.1.wrapping_mul(k);
        }
        self.konst = self.konst.wrapping_mul(k);
        self.norm()
    }

    /// The product, or None past degree 2.
    pub(super) fn mul(&self, other: &Form) -> Option<Form> {
        let mut terms: Vec<(Mono, u32)> = Vec::new();
        for (ka, ca) in &self.terms {
            for (kb, cb) in &other.terms {
                let mut k: Mono = ka.iter().chain(kb.iter()).copied().collect();
                if k.len() > 2 {
                    return None;
                }
                k.sort_unstable();
                terms.push((k, ca.wrapping_mul(*cb)));
            }
            terms.push((ka.clone(), ca.wrapping_mul(other.konst)));
        }
        for (kb, cb) in &other.terms {
            terms.push((kb.clone(), cb.wrapping_mul(self.konst)));
        }
        Some(Form { terms, konst: self.konst.wrapping_mul(other.konst) }.norm())
    }
}

/// What the walker reads.
#[derive(Clone, Copy)]
pub(super) struct Limits {
    /// Whether a product of two non-constants is a degree-2 term (else the product is a leaf).
    pub(super) products: bool,
    /// The most terms of a form.
    pub(super) terms: usize,
    /// A form past `terms` is: the whole walk fails (false), or that sub-expression is one leaf (true).
    pub(super) leaf_on_overflow: bool,
}

pub(super) struct Cx<'a> {
    pub(super) defs: &'a [Option<Def>],
    pub(super) kernel: &'a Kernel,
    pub(super) limits: Limits,
}

impl Cx<'_> {
    pub(super) fn konst(&self, v: ValueId) -> Option<u32> {
        self.defs.get(v.0 as usize)?.as_ref()?.konst.map(|k| k as u32)
    }

    pub(super) fn form(&self, v: ValueId, depth: u32) -> Option<Form> {
        if *self.kernel.ty(v) != Type::Scalar(Scalar::I32) {
            return None;
        }
        let Some(Some(d)) = self.defs.get(v.0 as usize) else { return Some(Form::leaf(v)) };
        if let Some(k) = d.konst {
            return Some(Form { terms: Vec::new(), konst: k as u32 });
        }
        if depth == 0 {
            return Some(Form::leaf(v));
        }
        let sub = |x: ValueId| self.form(x, depth - 1);
        let f = match (d.kind, d.operands.as_slice()) {
            (OpKind::Add, &[a, b]) => sub(a)?.add(&sub(b)?, 1),
            (OpKind::Sub, &[a, b]) => sub(a)?.add(&sub(b)?, u32::MAX),
            (OpKind::Mul, &[a, b]) => match (self.konst(a), self.konst(b)) {
                (_, Some(k)) => sub(a)?.scale(k),
                (Some(k), _) => sub(b)?.scale(k),
                _ if self.limits.products => match sub(a)?.mul(&sub(b)?) {
                    Some(p) => p,
                    None => Form::leaf(v),
                },
                _ => Form::leaf(v),
            },
            (OpKind::Shl, &[a, b]) => match self.konst(b) {
                Some(k) if k < 32 => sub(a)?.scale(1u32 << k),
                _ => Form::leaf(v),
            },
            _ => Form::leaf(v),
        };
        if f.terms.len() > self.limits.terms {
            return self.limits.leaf_on_overflow.then(|| Form::leaf(v));
        }
        Some(f)
    }
}

pub(super) fn collect(region: &Region, defs: &mut [Option<Def>], home: &mut [u32], loops: &mut Vec<(u32, u32)>, at: u32) {
    for op in &region.ops {
        let konst = (op.kind == OpKind::Constant).then(|| op.int_attr("value")).flatten();
        for r in &op.results {
            if let Some(slot) = defs.get_mut(r.0 as usize) {
                *slot = Some(Def { kind: op.kind, operands: op.operands.iter().copied().collect(), konst });
                home[r.0 as usize] = at;
            }
        }
        let inner = if op.kind == OpKind::For {
            loops.push((at, 0));
            let id = loops.len() as u32 - 1;
            for v in &op.block_args {
                if let Some(h) = home.get_mut(v.0 as usize) {
                    *h = id;
                }
            }
            id
        } else {
            at
        };
        for r in &op.regions {
            collect(r, defs, home, loops, inner);
        }
        if op.kind == OpKind::For {
            // The last loop number inside this one.
            let last = loops.len() as u32 - 1;
            loops[inner as usize].1 = last;
        }
    }
}

/// Emits the ops of a rebuilt sum into `out`; the implementor keeps `defs` / `home` in step with the new values.
pub(super) trait Emit {
    /// The name prefix of the values it creates.
    const PREFIX: &'static str;

    /// The kernel and its definition and home-loop tables.
    fn tables(&mut self) -> (&mut Kernel, &mut Vec<Option<Def>>, &mut Vec<u32>);

    fn value(&mut self) -> ValueId {
        let (kernel, defs, home) = self.tables();
        let v = kernel.new_value(Self::PREFIX, Type::Scalar(Scalar::I32));
        defs.push(None);
        home.push(0);
        v
    }

    fn konst(&mut self, k: u32, out: &mut Vec<Op>) -> ValueId {
        let v = self.value();
        out.push(Op {
            kind: OpKind::Constant,
            results: [v].into(),
            operands: Operands::new(),
            attrs: vec![("ity".into(), Attr::Str("i32".into())), ("value".into(), Attr::Int(k as i32 as i64))],
            regions: Vec::new(),
            block_args: Vec::new(),
            loc: None,
        });
        v
    }

    fn bin(&mut self, kind: OpKind, a: ValueId, b: ValueId, out: &mut Vec<Op>) -> ValueId {
        let v = self.value();
        out.push(Op {
            kind,
            results: [v].into(),
            operands: [a, b].into(),
            attrs: vec![("wrap".into(), Attr::Str("modular".into()))],
            regions: Vec::new(),
            block_args: Vec::new(),
            loc: None,
        });
        v
    }

    /// `coeff * leaves...` as ops.
    fn term(&mut self, leaves: &[u32], coeff: u32, out: &mut Vec<Op>) -> ValueId {
        let mut acc = ValueId(leaves[0]);
        for &l in &leaves[1..] {
            acc = self.bin(OpKind::Mul, acc, ValueId(l), out);
        }
        if coeff != 1 {
            let k = self.konst(coeff, out);
            acc = self.bin(OpKind::Mul, acc, k, out);
        }
        acc
    }

    /// The sum of `terms` (None when there are none).
    fn sum<'t>(&mut self, terms: impl IntoIterator<Item = &'t (Mono, u32)>, out: &mut Vec<Op>) -> Option<ValueId> {
        let mut acc: Option<ValueId> = None;
        for (k, c) in terms {
            let t = self.term(k, *c, out);
            acc = Some(match acc {
                None => t,
                Some(a) => self.bin(OpKind::Add, a, t, out),
            });
        }
        acc
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forms_multiply_and_cancel() {
        let (a, b) = (Form::leaf(ValueId(1)), Form::leaf(ValueId(2)));
        let k = Form { terms: Vec::new(), konst: 3 };
        // (a + 3) * b = a*b + 3*b
        let p = a.clone().add(&k, 1).mul(&b).unwrap();
        assert_eq!(p.terms, vec![(vec![1, 2], 1), (vec![2], 3)]);
        assert_eq!(p.konst, 0);
        assert!(a.clone().add(&a, u32::MAX).terms.is_empty());
        // Degree three is refused.
        assert!(p.mul(&a).is_none());
    }
}
