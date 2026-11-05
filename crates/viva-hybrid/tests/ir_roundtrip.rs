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

fn rate_json(rate: &str) -> String {
    format!(
        r#"{{"species":[{{"id":"S","initial":1.0}}],"params":{{}},"reactions":[{{"id":"R","partition":"continuous","stoichiometry":{{"S":-1}},"rate":{rate}}}]}}"#
    )
}

#[test]
fn multi_key_ast_errors() {
    let bad = rate_json(r#"{"add":[{"const":1},{"const":2}],"mul":[{"const":1},{"const":2}]}"#);
    assert!(Model::from_json(&bad).is_err());
}

#[test]
fn wrong_arity_errors() {
    assert!(Model::from_json(&rate_json(r#"{"add":[{"const":1}]}"#)).is_err());
    assert!(Model::from_json(&rate_json(r#"{"add":[{"const":1},{"const":2},{"const":3}]}"#)).is_err());
    assert!(Model::from_json(&rate_json(r#"{"add":[{"const":1},{"const":2}]}"#)).is_ok());
}

#[test]
fn unknown_fields_error() {
    assert!(Model::from_json(&FIX.replace("\"params\"", "\"extra\": 1, \"params\"")).is_err());
    assert!(Model::from_json(&FIX.replace("\"initial\": 100.0", "\"initial\": 100.0, \"x\": 1")).is_err());
    assert!(Model::from_json(&FIX.replace("\"id\": \"R1\"", "\"id\": \"R1\", \"x\": 1")).is_err());
}

#[test]
fn bad_types_error() {
    assert!(Model::from_json(&FIX.replace("\"initial\": 100.0", "\"initial\": \"100\"")).is_err());
    assert!(Model::from_json(&FIX.replace("\"S\": -1", "\"S\": -1.5")).is_err());
    assert!(Model::from_json(&FIX.replace("\"id\": \"R1\"", "\"id\": 7")).is_err());
    assert!(Model::from_json(&rate_json(r#"{"const":"1"}"#)).is_err());
    assert!(Model::from_json(&rate_json(r#"{"param":3}"#)).is_err());
}
