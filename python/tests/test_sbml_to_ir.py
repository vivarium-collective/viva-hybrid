from pathlib import Path

import pytest

from viva_hybrid_ref.ir import load_ir
from viva_hybrid_ref.sbml_to_ir import sbml_to_ir

FIX = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def test_mini_matches_ir_fixture():
    got = sbml_to_ir(FIX / "mini.xml", FIX / "mini.partition.yaml")
    assert got == load_ir(FIX / "mini_ir.json")


def test_missing_partition_tag_raises(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("R1: continuous\n")
    with pytest.raises(ValueError, match="R2"):
        sbml_to_ir(FIX / "mini.xml", p)


def test_invalid_partition_value_raises(tmp_path):
    p = tmp_path / "p.yaml"
    p.write_text("R1: continuous\nR2: hybrid\n")
    with pytest.raises(ValueError, match="hybrid"):
        sbml_to_ir(FIX / "mini.xml", p)


def test_unsupported_op_raises_naming_op(tmp_path):
    x = tmp_path / "m.xml"
    x.write_text((FIX / "mini.xml").read_text().replace(
        "<apply><times/><ci> k </ci><ci> S </ci></apply>",
        "<apply><sin/><ci> S </ci></apply>"))
    with pytest.raises(ValueError, match="sin"):
        sbml_to_ir(x, FIX / "mini.partition.yaml")


def test_unary_minus_and_power(tmp_path):
    x = tmp_path / "m.xml"
    x.write_text((FIX / "mini.xml").read_text().replace(
        "<apply><times/><ci> k </ci><ci> S </ci></apply>",
        "<apply><minus/><apply><power/><ci> S </ci><cn type=\"integer\"> 2 </cn></apply></apply>"))
    r = sbml_to_ir(x, FIX / "mini.partition.yaml").reactions[0].rate
    assert r.op == "mul" and r.arg[0].op == "const" and r.arg[0].arg == -1.0
    assert r.arg[1].op == "pow"
