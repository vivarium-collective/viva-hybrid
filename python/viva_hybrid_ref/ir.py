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


def ast_from_obj(o) -> Ast:
    if not isinstance(o, dict) or len(o) != 1:
        raise ValueError(f"bad ast node: {o!r}")
    (op, v), = o.items()
    if op == "const":
        return Ast(op, float(v))
    if op in ("param", "species"):
        return Ast(op, str(v))
    if op in BIN_OPS:
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError(f"{op} needs two operands")
        return Ast(op, (ast_from_obj(v[0]), ast_from_obj(v[1])))
    raise ValueError(f"unknown ast op: {op!r}")


def ast_to_obj(a: Ast):
    if a.op in LEAF_OPS:
        return {a.op: a.arg}
    if a.op in BIN_OPS:
        return {a.op: [ast_to_obj(a.arg[0]), ast_to_obj(a.arg[1])]}
    raise ValueError(f"unknown ast op: {a.op!r}")


def model_from_dict(d: dict) -> Model:
    reactions = []
    for r in d["reactions"]:
        if r["partition"] not in PARTITIONS:
            raise ValueError(f"unknown partition: {r['partition']!r}")
        reactions.append(Reaction(
            id=r["id"],
            partition=r["partition"],
            stoichiometry={k: int(v) for k, v in r["stoichiometry"].items()},
            rate=ast_from_obj(r["rate"]),
        ))
    return Model(
        species=[Species(s["id"], float(s["initial"])) for s in d["species"]],
        params={k: float(v) for k, v in d["params"].items()},
        reactions=reactions,
    )


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
