"""SBML + partition sidecar YAML -> shared IR. Runs once, outside the hot loop."""
from __future__ import annotations

import argparse
from pathlib import Path

import libsbml
import yaml

from .ir import PARTITIONS, Ast, Model, Reaction, Species, dump_ir

_BIN = {
    libsbml.AST_PLUS: "add",
    libsbml.AST_MINUS: "sub",
    libsbml.AST_TIMES: "mul",
    libsbml.AST_DIVIDE: "div",
    libsbml.AST_POWER: "pow",
    libsbml.AST_FUNCTION_POWER: "pow",
}
_NUM = (libsbml.AST_INTEGER, libsbml.AST_REAL, libsbml.AST_REAL_E, libsbml.AST_RATIONAL)


def _fold(op, kids):
    """n-ary add/mul -> left-nested binary."""
    acc = kids[0]
    for k in kids[1:]:
        acc = Ast(op, (acc, k))
    return acc


def _ast(node, species, params, where) -> Ast:
    t = node.getType()
    n = node.getNumChildren()
    if t in _NUM:
        return Ast("const", float(node.getValue()))
    if t == libsbml.AST_NAME:
        name = node.getName()
        if name in species:
            return Ast("species", name)
        if name in params:
            return Ast("param", name)
        raise ValueError(f"{where}: unknown symbol {name!r}")
    if t in _BIN:
        op = _BIN[t]
        kids = [_ast(node.getChild(i), species, params, where) for i in range(n)]
        if op == "sub" and n == 1:  # unary minus -> Mul(Const(-1), x)
            return Ast("mul", (Ast("const", -1.0), kids[0]))
        if n == 1 and op in ("add", "mul"):
            return kids[0]
        if n < 2:
            raise ValueError(f"{where}: operator {op!r} needs two operands, got {n}")
        if op in ("add", "mul"):
            return _fold(op, kids)
        if n != 2:
            raise ValueError(f"{where}: operator {op!r} needs exactly two operands, got {n}")
        return Ast(op, (kids[0], kids[1]))
    name = node.getName() or f"ast-type-{t}"
    raise ValueError(f"{where}: unsupported MathML operator {name!r}")


def _load_partition(path) -> dict:
    d = yaml.safe_load(Path(path).read_text())
    if not isinstance(d, dict):
        raise ValueError(f"{path}: partition sidecar must be a mapping reaction -> tag")
    return d


def sbml_to_ir(sbml_path, partition_path) -> Model:
    doc = libsbml.readSBMLFromFile(str(sbml_path))
    if doc.getNumErrors(libsbml.LIBSBML_SEV_ERROR) or doc.getModel() is None:
        msgs = "; ".join(doc.getError(i).getMessage() for i in range(doc.getNumErrors()))
        raise ValueError(f"cannot parse SBML {sbml_path}: {msgs}")
    m = doc.getModel()
    part = _load_partition(partition_path)

    species = []
    for s in m.getListOfSpecies():
        if not s.isSetInitialAmount():
            raise ValueError(
                f"species {s.getId()!r}: no initialAmount (concentration-initialized "
                "species are not supported)")
        species.append(Species(s.getId(), float(s.getInitialAmount())))
    sids = {s.id for s in species}
    params = {p.getId(): float(p.getValue()) for p in m.getListOfParameters()}

    reactions = []
    for r in m.getListOfReactions():
        rid = r.getId()
        tag = part.get(rid)
        if tag is None:
            raise ValueError(f"reaction {rid!r} missing from partition sidecar")
        if tag not in PARTITIONS:
            raise ValueError(f"reaction {rid!r}: invalid partition tag {tag!r}")
        stoich: dict = {}
        for sr in r.getListOfReactants():
            stoich[sr.getSpecies()] = stoich.get(sr.getSpecies(), 0) - _int_stoich(sr, rid)
        for sr in r.getListOfProducts():
            stoich[sr.getSpecies()] = stoich.get(sr.getSpecies(), 0) + _int_stoich(sr, rid)
        stoich = {k: v for k, v in stoich.items() if v != 0}
        kl = r.getKineticLaw()
        if kl is None or kl.getMath() is None:
            raise ValueError(f"reaction {rid!r}: no kinetic law")
        rate = _ast(kl.getMath(), sids, params, f"reaction {rid!r}")
        reactions.append(Reaction(rid, tag, stoich, rate))
    unmatched = sorted(set(map(str, part)) - {r.id for r in reactions})
    if unmatched:
        raise ValueError(f"partition sidecar keys match no reaction: {unmatched}")
    return Model(species=species, params=params, reactions=reactions)


def _int_stoich(sr, rid) -> int:
    v = sr.getStoichiometry()
    if v != int(v):
        raise ValueError(f"reaction {rid!r}: non-integer stoichiometry {v}")
    return int(v)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m viva_hybrid_ref.sbml_to_ir")
    ap.add_argument("sbml")
    ap.add_argument("partition")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    Path(a.out).write_text(dump_ir(sbml_to_ir(a.sbml, a.partition)) + "\n")


if __name__ == "__main__":
    main()
