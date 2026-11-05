//! Shared IR types. The JSON shape is the cross-language contract (Python
//! `viva_hybrid_ref/ir.py` mirrors it exactly).
//!
//! Ast JSON encoding (externally tagged, one key per node):
//!   {"const": 1.0} {"species": "S"} {"param": "k"}
//!   {"add": [a, b]} {"sub": [a, b]} {"mul": [a, b]} {"div": [a, b]} {"pow": [a, b]}
//! Partition is the string "continuous" or "stochastic". Unknown tags are errors.

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Species {
    pub id: String,
    pub initial: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Partition {
    Continuous,
    Stochastic,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Ast {
    Const(f64),
    Param(String),
    Species(String),
    Add(Box<Ast>, Box<Ast>),
    Sub(Box<Ast>, Box<Ast>),
    Mul(Box<Ast>, Box<Ast>),
    Div(Box<Ast>, Box<Ast>),
    Pow(Box<Ast>, Box<Ast>),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Reaction {
    pub id: String,
    pub partition: Partition,
    pub stoichiometry: BTreeMap<String, i64>,
    pub rate: Ast,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Model {
    pub species: Vec<Species>,
    pub params: BTreeMap<String, f64>,
    pub reactions: Vec<Reaction>,
}

impl Model {
    pub fn from_json(s: &str) -> Result<Model, serde_json::Error> {
        serde_json::from_str(s)
    }
}
