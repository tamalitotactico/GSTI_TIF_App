"""Pruebas del módulo de ML: etiquetas, entrenamiento, construcción de predicciones e
integración del microservicio de inferencia contra la API real."""
import asyncio
from datetime import timedelta

import httpx
import numpy as np
import pandas as pd
import pytest

from scripts import ml_service_client as client_mod
from scripts.entrenamiento_riesgo import (
    FEATURES_VECTOR, entrenar_y_guardar, generar_ground_truth, mascara_bypass, normalizar_columnas,
)
from scripts.generar_datos_sinteticos import generar_dataset_sintetico
from tests.conftest import new_id, now_iso


# ------------------------------------------------------------------ unitarias
def test_normalizacion_de_columnas_alias():
    raw_df = pd.DataFrame([{
        "participant_code": "P0001", "steps_daily_mean": 5000, "sleep_minutes_mean": 400,
        "home_minutes_mean": 600, "app_use_minutes_mean": 120, "phq9_total": 12, "gad7_total": 8,
        "uls6_total": 15, "cssrs_level": 1, "phq9_item9": 0,
    }])
    df_std = normalizar_columnas(raw_df)
    assert set(FEATURES_VECTOR) <= set(df_std.columns)
    assert df_std.loc[0, "pasos"] == 5000 and df_std.loc[0, "estudiante_id"] == "P0001"


def test_clinical_bypass_fuerza_riesgo_alto():
    # Una sola fila con gatillo clínico: antes fallaba por percentil de un conjunto vacío
    bypass_df = pd.DataFrame([{
        "pasos": 10000, "sueno": 480, "tiempo_casa": 400, "uso_apps": 50,
        "PHQ9": 2, "GAD7": 1, "ULS6": 6, "CSSRS": 0, "phq9_item9": 1,
    }])
    assert generar_ground_truth(bypass_df).iloc[0] == 2


def test_bypass_mismo_criterio_que_backend():
    df = normalizar_columnas(pd.DataFrame({"CSSRS": [2, 3, 0, np.nan], "phq9_item9": [0, 0, 1, np.nan]}))
    assert mascara_bypass(df).tolist() == [False, True, True, False]


@pytest.fixture(scope="module")
def modelo_entrenado(tmp_path_factory):
    d = tmp_path_factory.mktemp("ml")
    csv, out = d / "datos.csv", d / "modelo.joblib"
    generar_dataset_sintetico(n=400, seed=7).to_csv(csv, index=False)
    metricas = entrenar_y_guardar(str(csv), str(out))
    return out, metricas


def test_entrenamiento_excluye_gatillo_y_reporta_metricas(modelo_entrenado):
    out, m = modelo_entrenado
    assert out.exists()
    assert m["excluidas_por_gatillo"] > 0 and m["n"] + m["excluidas_por_gatillo"] == 400
    for k in ("sensibilidad", "especificidad", "f1", "auc_roc"):
        assert m[k] is not None and 0 <= m[k] <= 1


def test_construir_predicciones(modelo_entrenado):
    art = client_mod.cargar_modelo(str(modelo_entrenado[0]))
    filas = [
        {"participant_code": "P0001", "week_start": "2026-10-12", "steps_daily_mean": 9000,
         "sleep_minutes_mean": 450, "home_minutes_mean": 500, "app_use_minutes_mean": 90,
         "phq9_total": 2, "gad7_total": 1, "uls6_total": 7, "cssrs_level": 0, "phq9_item9": 0},
        {"participant_code": "P0002", "week_start": "2026-10-12", "steps_daily_mean": None,
         "sleep_minutes_mean": None, "home_minutes_mean": None, "app_use_minutes_mean": None,
         "phq9_total": 5, "gad7_total": None, "uls6_total": None, "cssrs_level": 3, "phq9_item9": 0},
    ]
    payload, n_bypass = client_mod.construir_predicciones(filas, art)
    assert n_bypass == 1 and payload[1]["risk_level"] == "alto"
    assert all(0 <= p["score"] <= 1 and p["score"] != 1.0 or p["risk_level"] == "alto" for p in payload)
    assert payload[0]["model_name"] == "random_forest" and payload[0]["threshold"] == 0.35


# ------------------------------------------------------------------ integración con la API
def test_ciclo_inferencia_contra_api(app_client, staff_tokens, enrolled, modelo_entrenado, monkeypatch, sql):
    from tests.test_api import blocks, lima_midnight, phq9

    code, dev, _ = enrolled
    d = lima_midnight(1)
    app_client.post("/api/v1/ingest/sensors", headers=dev, json={
        "batch_id": new_id(), "observations": blocks("steps", "health_connect", d, 48, 50)})
    app_client.post("/api/v1/questionnaires/responses", headers=dev, json={
        "client_response_id": new_id(), "instrument": "PHQ9", "context": "weekly",
        "completed_at": (d + timedelta(hours=10)).isoformat(), "answers": phq9(0)})

    monkeypatch.setattr(client_mod, "MODEL_PATH", str(modelo_entrenado[0]))
    monkeypatch.setenv("SERVICE_EMAIL", "service@test.pe")
    monkeypatch.setenv("SERVICE_PASSWORD", "una-clave-larga-123")
    transport = httpx.ASGITransport(app=app_client.app)
    cli = httpx.AsyncClient(transport=transport, base_url="http://testserver/api/v1")
    resumen = asyncio.run(client_mod.ejecutar_ciclo_inferencia(client=cli))

    assert resumen["semanas"] >= 1 and resumen["stored"] + resumen["updated"] == resumen["semanas"]
    row = sql("SELECT mp.model_name, mp.risk_level, mp.score FROM core.model_predictions mp "
              "JOIN core.participants p ON p.id = mp.participant_id WHERE p.code = %s", code)
    assert row and row[0][0] == "random_forest" and row[0][1] in ("bajo", "moderado", "alto")
