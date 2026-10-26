"""Microservicio de inferencia (artículo 4.3 y 4.6).

Ciclo: login con la cuenta de rol 'service' -> GET /features/person-week -> predicción ->
POST /predictions. Pensado para ejecutarse de forma programada (p. ej., semanal).

- El gatillo clínico se aplica como cláusula de guarda: esas semanas se informan como
  "alto" y se marcan en el resumen. El servidor vuelve a aplicar la misma regla y no
  duplica la alerta (ya se emitió al recibir el cuestionario).
- Para todas las semanas se guarda la probabilidad REAL del modelo (nunca un 1.0
  inventado), para que las métricas de evaluación (H1, H2) se calculen sin sesgo.

Variables de entorno: API_BASE_URL, SERVICE_EMAIL, SERVICE_PASSWORD (obligatoria),
MODEL_PATH, MODEL_VERSION.

Uso:
    SERVICE_PASSWORD='...' python -m scripts.ml_service_client [--since 2026-10-01] [--dry-run]
"""
import argparse
import asyncio
import os
import sys

import httpx
import joblib
import pandas as pd

from scripts.entrenamiento_riesgo import CSSRS_BYPASS, UMBRAL_ALTO, indice_clase, normalizar_columnas

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
SERVICE_EMAIL = os.getenv("SERVICE_EMAIL", "service@unsa.edu.pe")
MODEL_PATH = os.getenv("MODEL_PATH", "scripts/models_saved/modelo.joblib")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0")
LOTE_MAX = 1000  # límite de la API por solicitud


def cargar_modelo(path: str | None = None) -> dict:
    art = joblib.load(path or MODEL_PATH)
    if not isinstance(art, dict):  # compatibilidad con modelos guardados como Pipeline suelto
        art = {"pipeline": art, "features": list(art.feature_names_in_), "modelo": "rf", "umbral": UMBRAL_ALTO}
    return art


def construir_predicciones(features: list[dict], art: dict) -> tuple[list[dict], int]:
    """Convierte filas de /features/person-week en el cuerpo de POST /predictions."""
    if not features:
        return [], 0
    pipeline, cols, umbral = art["pipeline"], art["features"], art.get("umbral", UMBRAL_ALTO)
    df = normalizar_columnas(pd.DataFrame(features))
    probas = pipeline.predict_proba(df[cols].astype(float))
    i_alto, i_mod = indice_clase(pipeline, 2), indice_clase(pipeline, 1)
    nombre = {"rf": "random_forest", "xgb": "xgboost"}.get(art.get("modelo"), art.get("modelo", "random_forest"))

    payload, n_bypass = [], 0
    for row, p in zip(features, probas):
        bypass = (row.get("phq9_item9") or 0) > 0 or (row.get("cssrs_level") or 0) >= CSSRS_BYPASS
        n_bypass += bypass
        score = float(p[i_alto])
        if bypass or score >= umbral:
            nivel = "alto"
        else:
            nivel = "moderado" if p[i_mod] >= umbral else "bajo"
        payload.append({
            "participant_code": row["participant_code"], "week_start": row["week_start"],
            "model_name": nombre, "model_version": MODEL_VERSION,
            "score": round(score, 4), "threshold": umbral, "risk_level": nivel,
        })
    return payload, n_bypass


async def ejecutar_ciclo_inferencia(since: str | None = None, dry_run: bool = False,
                                    client: httpx.AsyncClient | None = None) -> dict:
    password = os.getenv("SERVICE_PASSWORD")
    if not password:
        sys.exit("Define SERVICE_PASSWORD con la contraseña de la cuenta de servicio")
    art = cargar_modelo()
    own = client is None
    client = client or httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0)
    try:
        r = await client.post("/auth/login", json={"email": os.getenv("SERVICE_EMAIL", SERVICE_EMAIL),
                                                   "password": password})
        r.raise_for_status()
        headers = {"Authorization": f"Bearer {r.json()['access_token']}"}

        params = {"since": since} if since else {}
        r = await client.get("/features/person-week", headers=headers, params=params)
        r.raise_for_status()
        payload, n_bypass = construir_predicciones(r.json(), art)
        resumen = {"semanas": len(payload), "gatillo_clinico": n_bypass,
                   "alto": sum(p["risk_level"] == "alto" for p in payload)}
        if not payload or dry_run:
            print("Sin envío (dry-run o sin datos):", resumen)
            return resumen

        for i in range(0, len(payload), LOTE_MAX):
            r = await client.post("/predictions", headers=headers, json={"predictions": payload[i:i + LOTE_MAX]})
            r.raise_for_status()
            for k, v in r.json().items():
                resumen[k] = resumen.get(k, 0) + v
        print("Predicciones enviadas:", resumen)
        return resumen
    finally:
        if own:
            await client.aclose()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ciclo de inferencia del microservicio ML.")
    ap.add_argument("--since", help="Solo semanas desde esta fecha (AAAA-MM-DD)")
    ap.add_argument("--dry-run", action="store_true", help="Calcula sin enviar a la API")
    a = ap.parse_args()
    asyncio.run(ejecutar_ciclo_inferencia(a.since, a.dry_run))
