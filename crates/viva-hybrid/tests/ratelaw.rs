use std::collections::BTreeMap;
use viva_hybrid::ir::Ast;
use viva_hybrid::ratelaw::compile;

fn b(a: Ast) -> Box<Ast> {
    Box::new(a)
}
fn idx() -> BTreeMap<String, usize> {
    BTreeMap::from([("S".to_string(), 0), ("T".to_string(), 1)])
}
fn params() -> BTreeMap<String, f64> {
    BTreeMap::from([("k".to_string(), 0.5)])
}

#[test]
fn mul_param_species() {
    let ast = Ast::Mul(b(Ast::Param("k".into())), b(Ast::Species("S".into())));
    let rl = compile(&ast, &idx(), &params());
    assert_eq!(rl.eval(&[10.0, 0.0]), 5.0);
}

#[test]
fn pow_and_div() {
    let pow = Ast::Pow(b(Ast::Species("S".into())), b(Ast::Const(2.5)));
    let rl = compile(&pow, &idx(), &params());
    assert!((rl.eval(&[4.0, 0.0]) - 32.0).abs() < 1e-12);

    let div = Ast::Div(b(Ast::Species("S".into())), b(Ast::Species("T".into())));
    let rl = compile(&div, &idx(), &params());
    assert_eq!(rl.eval(&[9.0, 3.0]), 3.0);
}

#[test]
fn add_sub_nested() {
    // (S + T) - k
    let ast = Ast::Sub(
        b(Ast::Add(b(Ast::Species("S".into())), b(Ast::Species("T".into())))),
        b(Ast::Param("k".into())),
    );
    let rl = compile(&ast, &idx(), &params());
    assert_eq!(rl.eval(&[1.0, 2.0]), 2.5);
}

#[test]
#[should_panic(expected = "unknown species")]
fn unknown_species_panics() {
    compile(&Ast::Species("Z".into()), &idx(), &params());
}
