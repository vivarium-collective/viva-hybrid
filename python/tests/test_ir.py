import json
from pathlib import Path

import pytest

from viva_hybrid_ref.ir import Ast, load_ir, dump_ir, Model

FIX = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "mini_ir.json"


def test_load_fields():
    m = load_ir(FIX)
    assert [s.initial for s in m.species] == [100.0, 0.0]
    assert m.params == {"k": 0.5}
    assert m.reactions[0].partition == "continuous"
    assert m.reactions[1].partition == "stochastic"
    assert m.reactions[0].rate == Ast("mul", (Ast("param", "k"), Ast("species", "S")))


def test_dump_roundtrip():
    m = load_ir(FIX)
    assert json.loads(dump_ir(m)) == json.loads(FIX.read_text())
    assert dump_ir(load_ir_str(dump_ir(m))) == dump_ir(m)


def load_ir_str(s):
    from viva_hybrid_ref.ir import model_from_dict
    return model_from_dict(json.loads(s))


def test_bad_partition():
    d = json.loads(FIX.read_text())
    d["reactions"][0]["partition"] = "hybrid"
    from viva_hybrid_ref.ir import model_from_dict
    with pytest.raises(ValueError):
        model_from_dict(d)


def test_bad_op():
    d = json.loads(FIX.read_text())
    d["reactions"][0]["rate"] = {"mod": [{"const": 1.0}, {"const": 2.0}]}
    from viva_hybrid_ref.ir import model_from_dict
    with pytest.raises(ValueError):
        model_from_dict(d)


def _model(rate=None, **over):
    d = json.loads(FIX.read_text())
    if rate is not None:
        d["reactions"][0]["rate"] = rate
    d.update(over)
    return d


def _bad(d):
    from viva_hybrid_ref.ir import model_from_dict
    with pytest.raises(ValueError):
        model_from_dict(d)


C = {"const": 1}


def test_multi_key_ast():
    _bad(_model({"add": [C, C], "mul": [C, C]}))


def test_wrong_arity():
    _bad(_model({"add": [C]}))
    _bad(_model({"add": [C, C, C]}))


def test_unknown_fields():
    _bad(_model(extra=1))
    d = _model(); d["species"][0]["x"] = 1; _bad(d)
    d = _model(); d["reactions"][0]["x"] = 1; _bad(d)


def test_bad_types():
    d = _model(); d["species"][0]["initial"] = "100"; _bad(d)
    d = _model(); d["species"][0]["initial"] = True; _bad(d)
    d = _model(); d["reactions"][0]["stoichiometry"]["S"] = -1.5; _bad(d)
    d = _model(); d["reactions"][0]["stoichiometry"]["S"] = -1.0; _bad(d)
    d = _model(); d["reactions"][0]["id"] = 7; _bad(d)
    d = _model(); d["params"]["k"] = "0.5"; _bad(d)
    _bad(_model({"const": "1"}))
    _bad(_model({"param": 3}))
