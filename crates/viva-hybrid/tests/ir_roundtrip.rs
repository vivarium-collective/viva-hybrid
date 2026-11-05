use viva_hybrid::ir::{Ast, Model, Partition};

const FIX: &str = include_str!("../../../tests/fixtures/mini_ir.json");

#[test]
fn parses_and_roundtrips() {
    let m = Model::from_json(FIX).unwrap();
    assert_eq!(m.species.len(), 2);
    assert_eq!(m.species[0].initial, 100.0);
    assert_eq!(m.params["k"], 0.5);
    assert_eq!(m.reactions[0].partition, Partition::Continuous);
    assert_eq!(m.reactions[1].partition, Partition::Stochastic);
    assert_eq!(
        m.reactions[0].rate,
        Ast::Mul(Box::new(Ast::Param("k".into())), Box::new(Ast::Species("S".into())))
    );
    let again = Model::from_json(&serde_json::to_string(&m).unwrap()).unwrap();
    assert_eq!(m, again);
}

#[test]
fn bad_partition_errors() {
    let bad = FIX.replace("\"stochastic\"", "\"hybrid\"");
    assert!(Model::from_json(&bad).is_err());
}

#[test]
fn bad_op_errors() {
    let bad = FIX.replace("\"mul\"", "\"mod\"");
    assert!(Model::from_json(&bad).is_err());
}
