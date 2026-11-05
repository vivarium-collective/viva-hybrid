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
