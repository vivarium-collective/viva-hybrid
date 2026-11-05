"""Shared IR types; JSON shape mirrors crates/viva-hybrid/src/ir.rs.

Ast JSON encoding (one key per node):
  {"const": 1.0} {"species": "S"} {"param": "k"}
  {"add": [a, b]} {"sub": [a, b]} {"mul": [a, b]} {"div": [a, b]} {"pow": [a, b]}
Partition is "continuous" or "stochastic". Unknown tags raise ValueError.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

PARTITIONS = ("continuous", "stochastic")
LEAF_OPS = ("const", "param", "species")
BIN_OPS = ("add", "sub", "mul", "div", "pow")


@dataclass(frozen=True)
class Ast:
    """op in LEAF_OPS -> arg is float/str; op in BIN_OPS -> arg is (Ast, Ast)."""
    op: str
    arg: Union[float, str, tuple]


@dataclass
class Species:
    id: str
    initial: float


@dataclass
class Reaction:
    id: str
    partition: str
    stoichiometry: dict
    rate: Ast


@dataclass
class Model:
    species: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    reactions: list = field(default_factory=list)


def _num(v, what) -> float:
    # Rust f64: accepts JSON ints/floats, rejects strings and bools.
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"{what}: expected number, got {v!r}")
    return float(v)


def _int(v, what) -> int:
    # Rust i64: rejects floats (even 1.0), strings, bools, out-of-range.
    if isinstance(v, bool) or not isinstance(v, int) or not -(2**63) <= v < 2**63:
        raise ValueError(f"{what}: expected i64 integer, got {v!r}")
    return v


def _str(v, what) -> str:
    if not isinstance(v, str):
        raise ValueError(f"{what}: expected string, got {v!r}")
    return v


def _obj(o, keys, what) -> dict:
    # Rust deny_unknown_fields + required fields.
    if not isinstance(o, dict):
        raise ValueError(f"{what}: expected object, got {o!r}")
    extra, missing = set(o) - set(keys), set(keys) - set(o)
    if extra or missing:
        raise ValueError(f"{what}: unknown fields {sorted(extra)}, missing {sorted(missing)}")
    return o


def ast_from_obj(o) -> Ast:
    if not isinstance(o, dict) or len(o) != 1:
        raise ValueError(f"bad ast node: {o!r}")
    (op, v), = o.items()
    if op == "const":
        return Ast(op, _num(v, "const"))
    if op in ("param", "species"):
        return Ast(op, _str(v, op))
    if op in BIN_OPS:
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError(f"{op} needs exactly two operands")
        return Ast(op, (ast_from_obj(v[0]), ast_from_obj(v[1])))
    raise ValueError(f"unknown ast op: {op!r}")


def ast_to_obj(a: Ast):
    if a.op in LEAF_OPS:
        return {a.op: a.arg}
    if a.op in BIN_OPS:
        return {a.op: [ast_to_obj(a.arg[0]), ast_to_obj(a.arg[1])]}
    raise ValueError(f"unknown ast op: {a.op!r}")


def _map(o, conv, what) -> dict:
    if not isinstance(o, dict):
        raise ValueError(f"{what}: expected object, got {o!r}")
    return {k: conv(v, f"{what}.{k}") for k, v in o.items()}


def _list(o, what) -> list:
    if not isinstance(o, list):
        raise ValueError(f"{what}: expected array, got {o!r}")
    return o


def model_from_dict(d: dict) -> Model:
    d = _obj(d, ("species", "params", "reactions"), "model")
    reactions = []
    for r in _list(d["reactions"], "reactions"):
        r = _obj(r, ("id", "partition", "stoichiometry", "rate"), "reaction")
        if r["partition"] not in PARTITIONS:
            raise ValueError(f"unknown partition: {r['partition']!r}")
        reactions.append(Reaction(
            id=_str(r["id"], "reaction.id"),
            partition=r["partition"],
            stoichiometry=_map(r["stoichiometry"], _int, "stoichiometry"),
            rate=ast_from_obj(r["rate"]),
        ))
    species = []
    for s in _list(d["species"], "species"):
        s = _obj(s, ("id", "initial"), "species")
        species.append(Species(_str(s["id"], "species.id"), _num(s["initial"], "species.initial")))
    return Model(species=species, params=_map(d["params"], _num, "params"), reactions=reactions)


def model_to_dict(m: Model) -> dict:
    return {
        "species": [{"id": s.id, "initial": s.initial} for s in m.species],
        "params": dict(m.params),
        "reactions": [
            {"id": r.id, "partition": r.partition,
             "stoichiometry": dict(r.stoichiometry), "rate": ast_to_obj(r.rate)}
            for r in m.reactions
        ],
    }


def load_ir(path) -> Model:
    return model_from_dict(json.loads(Path(path).read_text()))


def dump_ir(model: Model) -> str:
    return json.dumps(model_to_dict(model), indent=2, sort_keys=True)
