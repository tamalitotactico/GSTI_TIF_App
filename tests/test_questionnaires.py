"""Los JSON de cuestionarios (contenido de la app) deben coincidir con el catálogo del backend."""
import json
from pathlib import Path

import pytest

from app.catalog import INSTRUMENTS_BY_CODE
from app.services.scoring import validate_and_score

QDIR = Path(__file__).resolve().parents[1] / "questionnaires"
FILES = sorted(QDIR.glob("*.json"))


def _items(q):
    """Todos los ítems enviables del JSON: principales, sub-ítems e impacto funcional."""
    out = []
    for it in q["items"]:
        out.append(it)
        out += it.get("sub_items", [])
    if "functional_impact_item" in q:
        out.append(q["functional_impact_item"])
    return out


def _range(q, it):
    if it.get("type") == "boolean" or q["items"][0].get("type") == "boolean":
        return 0, 1
    opts = it.get("options") or q["scale"]["options"]
    vals = [o["value"] for o in opts]
    return min(vals), max(vals)


@pytest.mark.parametrize("path", FILES, ids=lambda p: p.name)
def test_json_coincide_con_catalogo(path):
    q = json.loads(path.read_text(encoding="utf-8"))
    ins = INSTRUMENTS_BY_CODE[q["backend"]["instrument"]]
    cat = {i.item_no: i for i in ins.items}
    nums = [it["backend_item"] for it in _items(q)]
    assert sorted(nums) == sorted(cat), f"{path.name}: ítems JSON {sorted(nums)} vs catálogo {sorted(cat)}"
    for it in _items(q):
        c = cat[it["backend_item"]]
        assert _range(q, it) == (c.min_value, c.max_value), f"{path.name} ítem {it['backend_item']}"


def test_cuatro_cuestionarios_del_articulo():
    codes = {json.loads(p.read_text(encoding="utf-8"))["backend"]["instrument"] for p in FILES}
    assert codes == {"PHQ9", "GAD7", "ULS6", "CSSRS_SCREEN"}


def test_impacto_funcional_opcional_y_bandas():
    r = validate_and_score(INSTRUMENTS_BY_CODE["PHQ9"], {i: 1 for i in range(1, 10)})
    assert r.total == 9 and r.subscores["severity"] == "mild"
    r = validate_and_score(INSTRUMENTS_BY_CODE["PHQ9"], {**{i: 2 for i in range(1, 10)}, 10: 3})
    assert r.total == 18 and r.subscores == {"severity": "moderately_severe", "phq9_impact": 3}
    r = validate_and_score(INSTRUMENTS_BY_CODE["GAD7"], {**{i: 3 for i in range(1, 8)}, 8: 1})
    assert r.total == 21 and r.subscores["severity"] == "severe"


def test_endpoint_entrega_cuestionario(app_client):
    lst = app_client.get("/api/v1/catalog/questionnaires").json()
    assert {x["instrument"] for x in lst} == {"PHQ9", "GAD7", "ULS6", "CSSRS_SCREEN"}
    q = app_client.get("/api/v1/catalog/questionnaires/PHQ-9").json()
    assert q["backend"]["instrument"] == "PHQ9" and len(q["items"]) == 9
    assert app_client.get("/api/v1/catalog/questionnaires/XYZ").status_code == 404
