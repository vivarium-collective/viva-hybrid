//! Rate-law compiler: an `Ast` is compiled once into a boxed closure tree
//! behind the `RateLaw` trait (the codegen/JIT seam).
//!
//! Unknown species or parameter names panic at compile time with a clear
//! message; the IR is validated upstream so this signals a programming error.
use crate::ir::Ast;
use std::collections::BTreeMap;

/// A compiled kinetic law. `state` holds species amounts in `species_index` order.
pub trait RateLaw {
    fn eval(&self, state: &[f64]) -> f64;
}

type Node = Box<dyn Fn(&[f64]) -> f64>;

struct Closure(Node);

impl RateLaw for Closure {
    fn eval(&self, state: &[f64]) -> f64 {
        (self.0)(state)
    }
}

fn build(
    ast: &Ast,
    species_index: &BTreeMap<String, usize>,
    params: &BTreeMap<String, f64>,
) -> Node {
    match ast {
        Ast::Const(c) => {
            let c = *c;
            Box::new(move |_| c)
        }
        Ast::Param(p) => {
            let v = *params
                .get(p)
                .unwrap_or_else(|| panic!("unknown param '{p}' in rate law"));
            Box::new(move |_| v)
        }
        Ast::Species(s) => {
            let i = *species_index
                .get(s)
                .unwrap_or_else(|| panic!("unknown species '{s}' in rate law"));
            Box::new(move |st| st[i])
        }
        Ast::Add(a, b) => bin(a, b, species_index, params, |x, y| x + y),
        Ast::Sub(a, b) => bin(a, b, species_index, params, |x, y| x - y),
        Ast::Mul(a, b) => bin(a, b, species_index, params, |x, y| x * y),
        Ast::Div(a, b) => bin(a, b, species_index, params, |x, y| x / y),
        Ast::Pow(a, b) => bin(a, b, species_index, params, f64::powf),
    }
}

fn bin(
    a: &Ast,
    b: &Ast,
    si: &BTreeMap<String, usize>,
    p: &BTreeMap<String, f64>,
    op: fn(f64, f64) -> f64,
) -> Node {
    let (l, r) = (build(a, si, p), build(b, si, p));
    Box::new(move |st| op(l(st), r(st)))
}

/// Compile `ast` into an evaluator. Params fold to constants; species become
/// `state[index]` reads. Panics on an unknown species or param name.
pub fn compile(
    ast: &Ast,
    species_index: &BTreeMap<String, usize>,
    params: &BTreeMap<String, f64>,
) -> Box<dyn RateLaw> {
    Box::new(Closure(build(ast, species_index, params)))
}
