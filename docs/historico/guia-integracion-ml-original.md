# **Guía de Integración Técnica y Tutorial de Despliegue: Módulo de ML en GSTI Backend** 

Esta guía técnica sirve como tutorial paso a paso para el desarrollador responsable de la Base de Datos y Backend en el proyecto GSTI (Sistema Móvil Inteligente de Monitoreo Comportamental para la Detección Temprana de Riesgo Suicida - EPIS UNSA). Detalla la arquitectura de integración, la ubicación exacta de los scripts en la estructura del repositorio tamalitotactico-gsti_back_datos, los flujos de interacción con la API FastAPI y PostgreSQL, y el protocolo de pruebas automatizadas con pytest. 

## **1. Ubicación de Archivos en la Estructura del Repositorio** 

Los scripts de Machine Learning se ubican dentro del repositorio tamalitotactico-gsti_back_datos siguiendo la convención modular del proyecto: 

|Ruta Relativa del Archivo|Propósito / Función|Dependencias Principales|
|---|---|---|
|scripts/generar_datos_sinteti<br>cos.py|Genera datasets sintéticos<br>multimodales para staging y<br>desarrollo.|pandas, numpy|
|scripts/entrenamiento_riesgo<br>.py|Entrena el Pipeline de<br>Scikit-Learn (Imputación<br>KNN + Z-score + Random<br>Forest) y exporta el modelo.|scikit-learn, joblib, pandas|
|scripts/ml_service_client.py|Cliente/Microservicio de<br>inferencia que consume<br>/features/person-week y<br>envía /predictions a FastAPI.|httpx, joblib, pandas|
|scripts/models_saved/model<br>o.joblib|Artefacto binario serializado<br>del pipeline entrenado<br>(Random Forest).|Generado por entrenamiento|
|tests/test_ml_inference.py|Suite de pruebas unitarias e<br>integración para el flujo de|pytest, httpx, scikit-learn|



|Ruta Relativa del Archivo|Propósito / Función|Dependencias Principales|
|---|---|---|
||ML y la regla de Clinical<br>Bypass.||



## **2. Arquitectura de Integración y Flujo de Datos** 

El motor de inteligencia artificial opera de manera desacoplada consumiendo la API mediante la cuenta de rol service. El flujo de información se ejecuta de la siguiente manera: 

1. **Extracción de Características (GET):** El microservicio de ML consulta el endpoint GET /api/v1/features/person-week. Este endpoint lee de la vista SQL 

analytics.v_person_week_features los 8 indicadores por persona-semana: steps_daily_mean, sleep_minutes_mean, home_minutes_mean, 

app_use_minutes_mean, phq9_total, gad7_total, uls6_total y cssrs_level. 

2. **Preprocesamiento e Inferencia:** 

   - Si se activa la regla de **Clinical Bypass** (Ítem 9 de PHQ-9 > 0 o C-SSRS ≥ 3), la instancia se etiqueta automáticamente como **Riesgo Alto** (probabilidad 1.0) sin pasar por el clasificador. 

   - Si no hay bypass, el vector procesa los faltantes mediante KNNImputer(k=5), escala con StandardScaler y calcula probabilidades con RandomForestClassifier. 

3. **Persistencia de Predicciones y Alertas (POST):** El microservicio envía los resultados mediante POST /api/v1/predictions. La API registra la predicción en core.model_predictions. Si el riesgo es **Alto** (probabilidad ≥ 0.35 o Bypass), la API 

genera automáticamente un registro en core.alerts con el evento model_high_risk y emite una notificación WebSocket en tiempo real hacia el dashboard (/ws/dashboard). 

## **3. Scripts de Implementación Técnica** 

### **3.1 Script de Generación de Datos Sintéticos (scripts/generar_datos_sinteticos.py)** 

<mark>import argparse import logging import sys from pathlib import Path from typing import Optional import numpy as np import pandas as pd</mark> 

<mark>RANGO_PASOS = (1000, 25000) RANGO_SUENO = (180, 600) RANGO_CASA = (300, 1200) RANGO_APPS = (10, 400)</mark> 

<mark>def configurar_logging() -> None: logger = logging.getLogger() if not logger.handlers: logging.basicConfig(</mark> 

<mark>level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[logging.StreamHandler(sys.stdout)], )</mark> 

<mark>def generar_dataset_sintetico(n: int = 500, seed: int = 42) -> pd.DataFrame: np.random.seed(seed)</mark> 

<mark># Formato de seudónimo P0001 alineado a la base de datos core.participants estudiantes_ids = [f"P{i+1:04d}" for i in range(n)]</mark> 

<mark>semanas = np.random.randint(1, 17, size=n)</mark> 

<mark>z = np.random.normal(0, 1, n)</mark> 

<mark>phq9 = np.clip(np.round(np.exp(1.5 + 0.6 * z) - 1), 0, 27).astype(int)</mark> 

<mark>gad7 = np.clip(np.round(np.exp(1.3 + 0.55 * z) - 1), 0, 21).astype(int) uls6 = np.clip(np.round(6 + np.exp(1.8 + 0.45 * z) - 1), 6, 24).astype(int)</mark> 

<mark>pasos = np.clip(np.round(8500 - 2500 * z + np.random.normal(0, 1200, n)), RANGO_PASOS[0], RANGO_PASOS[1])</mark> 

<mark>sueno = np.clip(np.round(420 - 55 * z + np.random.normal(0, 40, n)), RANGO_SUENO[0], RANGO_SUENO[1])</mark> 

<mark>tiempo_casa = np.clip(np.round(550 + 160 * z + np.random.normal(0, 80, n)), RANGO_CASA[0], RANGO_CASA[1])</mark> 

<mark>uso_apps = np.clip(np.round(100 + 65 * z + np.random.normal(0, 25, n)), RANGO_APPS[0], RANGO_APPS[1])</mark> 

<mark>prob_bypass = 1 / (1 + np.exp(-(z - 1.8))) * 0.35</mark> 

<mark>es_bypass = np.random.rand(n) < prob_bypass</mark> 

<mark>cssrs = np.zeros(n, dtype=int)</mark> 

<mark>phq9_item9 = np.zeros(n, dtype=int)</mark> 

<mark>for i in range(n):</mark> 

<mark>if es_bypass[i]: if np.random.rand() > 0.5: cssrs[i] = np.random.choice([3, 4, 5]) phq9_item9[i] = np.random.choice([0, 1, 2, 3]) else: cssrs[i] = np.random.choice([0, 1, 2]) phq9_item9[i] = np.random.choice([1, 2, 3]) else: cssrs[i] = np.random.choice([0, 1, 2], p=[0.80, 0.15, 0.05]) phq9_item9[i] = 0 df = pd.DataFrame({ "estudiante_id": estudiantes_ids, "semana": semanas, "pasos": pasos, "sueno": sueno, "tiempo_casa": tiempo_casa, "uso_apps": uso_apps, "PHQ9": phq9, "GAD7": gad7, "ULS6": uls6, "CSSRS": cssrs, "phq9_item9": phq9_item9, }) for col in ["pasos", "sueno", "tiempo_casa", "uso_apps"]: mask_nulos = np.random.rand(n) < np.random.uniform(0.05, 0.08) df.loc[mask_nulos, col] = np.nan</mark> 

<mark>return df</mark> 

<mark>def main(args_list: Optional[list] = None) -> pd.DataFrame: configurar_logging()</mark> 

<mark>parser = argparse.ArgumentParser(description="Generador de datos sintéticos.") parser.add_argument("--n", type=int, default=2000) parser.add_argument("--output", type=str, default="scripts/datos_estudiantes.csv") parser.add_argument("--seed", type=int, default=42)</mark> 

<mark>if args_list is not None: args, _ = parser.parse_known_args(args_list) else: args, _ = parser.parse_known_args()</mark> 

<mark>path_output = Path(args.output) path_output.parent.mkdir(parents=True, exist_ok=True) df = generar_dataset_sintetico(args.n, args.seed) df.to_csv(path_output, index=False)</mark> 

<mark>logging.info(f"Dataset generado exitosamente con {len(df)} registros en {path_output}") return df</mark> 

<mark>if __name__ == "__main__": main()</mark> 

### **3.2 Script de Entrenamiento del Pipeline ML (scripts/entrenamiento_riesgo.py)** 

<mark>import json import logging from pathlib import Path import joblib import numpy as np import pandas as pd from sklearn.experimental import enable_iterative_imputer  # noqa: F401 from sklearn.impute import KNNImputer, IterativeImputer from sklearn.preprocessing import StandardScaler from sklearn.ensemble import RandomForestClassifier from sklearn.pipeline import Pipeline from sklearn.model_selection import train_test_split from sklearn.metrics import classification_report, accuracy_score</mark> 

<mark>FEATURES_VECTOR = ["pasos", "sueno", "tiempo_casa", "uso_apps", "PHQ9", "GAD7", "ULS6", "CSSRS", "phq9_item9"]</mark> 

<mark>def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame: alias_map = { "pasos": ["pasos", "pasos_diarios", "steps", "steps_daily_mean"], "sueno": ["sueno", "minutos_sueno", "sleep_minutes", "sleep_minutes_mean"], "tiempo_casa": ["tiempo_casa", "tiempo_casa_min", "home_minutes", "home_minutes_mean"], "uso_apps": ["uso_apps", "bloques_uso_apps", "app_use_minutes", "app_use_minutes_mean"], "PHQ9": ["PHQ9", "phq9_total", "phq9"],</mark> 

<mark>"GAD7": ["GAD7", "gad7_total", "gad7"], "ULS6": ["ULS6", "uls6_total", "uls6"], "CSSRS": ["CSSRS", "cssrs_level", "cssrs_nivel"], "phq9_item9": ["phq9_item9", "phq9_i9", "item9"], } df_std = df.copy() for col_std, aliases in alias_map.items(): encontrado = next((a for a in aliases if a in df_std.columns), None) if encontrado: df_std.rename(columns={encontrado: col_std}, inplace=True) else: df_std[col_std] = 0.0 if col_std == "phq9_item9" else np.nan return df_std def generar_ground_truth(df: pd.DataFrame) -> pd.Series: df_std = normalizar_columnas(df) bypass_mask = (df_std["phq9_item9"] > 0) | (df_std["CSSRS"] >= 3) phq_norm = df_std["PHQ9"].fillna(0) / 27.0 gad_norm = df_std["GAD7"].fillna(0) / 21.0 uls_norm = (df_std["ULS6"].fillna(6) - 6) / 18.0 pasos_inv = 1.0 - (df_std["pasos"].fillna(8500) / 25000.0) sueno_inv = 1.0 - (df_std["sueno"].fillna(420) / 600.0) casa_norm = df_std["tiempo_casa"].fillna(550) / 1200.0 apps_norm = df_std["uso_apps"].fillna(100) / 400.0</mark> 

<mark>score = ( 0.30 * phq_norm + 0.25 * gad_norm + 0.20 * uls_norm + 0.10 * pasos_inv + 0.05 * sueno_inv + 0.05 * casa_norm + 0.05 * apps_norm ) p75 = np.percentile(score[~bypass_mask], 75) p50 = np.percentile(score[~bypass_mask], 50) target = pd.Series(0, index=df_std.index, dtype=int) target[bypass_mask] = 2</mark> 

<mark>for idx in df_std[~bypass_mask].index: s = score.loc[idx] target.loc[idx] = 2 if s >= p75 else (1 if s >= p50 else 0)</mark> 

<mark>return target</mark> 

<mark>def entrenar_y_guardar(csv_path: str = "scripts/datos_estudiantes.csv", model_path: str = "scripts/models_saved/modelo.joblib") -> None: df = pd.read_csv(csv_path) df_std = normalizar_columnas(df) y = generar_ground_truth(df_std) X = df_std[FEATURES_VECTOR].copy()</mark> 

<mark>X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)</mark> 

<mark>pipeline = Pipeline([</mark> 

<mark>("knn_imputer", KNNImputer(n_neighbors=5)), ("scaler", StandardScaler()), ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)) ])</mark> 

<mark>pipeline.fit(X_train, y_train) y_pred = pipeline.predict(X_test) acc = accuracy_score(y_test, y_pred)</mark> 

<mark>print(f"Entrenamiento finalizado. Accuracy Test: {acc:.4f}")</mark> 

<mark>Path(model_path).parent.mkdir(parents=True, exist_ok=True) joblib.dump(pipeline, model_path)</mark> 

<mark>print(f"Modelo guardado exitosamente en: {model_path}")</mark> 

<mark>if __name__ == "__main__": entrenar_y_guardar()</mark> 

### **3.3 Cliente de Inferencia y Conexión con FastAPI (scripts/ml_service_client.py)** 

<mark>import asyncio import os import joblib import httpx import pandas as pd</mark> 

<mark>API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1") SERVICE_EMAIL = os.getenv("SERVICE_EMAIL", "service@unsa.edu.pe") SERVICE_PASSWORD = os.getenv("SERVICE_PASSWORD", "clave-de-servicio-123")</mark> 

<mark>MODEL_PATH = os.getenv("MODEL_PATH", "scripts/models_saved/modelo.joblib")</mark> 

<mark>FEATURES_ORDER = ["pasos", "sueno", "tiempo_casa", "uso_apps", "PHQ9", "GAD7", "ULS6", "CSSRS", "phq9_item9"]</mark> 

<mark>async def ejecutar_ciclo_inferencia(): pipeline = joblib.load(MODEL_PATH)</mark> 

<mark>async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30.0) as client: # 1. Autenticación con rol 'service' resp_login = await client.post("/auth/login", json={"email": SERVICE_EMAIL, "password": SERVICE_PASSWORD}) resp_login.raise_for_status() token = resp_login.json()["access_token"] headers = {"Authorization": f"Bearer {token}"}</mark> 

<mark># 2. Lectura de características persona-semana desde la API resp_feat = await client.get("/features/person-week", headers=headers) resp_feat.raise_for_status() features_data = resp_feat.json()</mark> 

<mark>if not features_data:</mark> 

<mark>print("No se encontraron registros de persona-semana para evaluar.") return</mark> 

<mark># 3. Mapeo e Inferencia predicciones_payload = [] for row in features_data: code = row["participant_code"] week_start = row["week_start"] p9 = row.get("phq9_item9") or 0 cssrs = row.get("cssrs_level") or 0 # Evaluación de Clinical Bypass is_bypass = (p9 > 0) or (cssrs >= 3)</mark> 

<mark>if is_bypass: score = 1.0 risk_level = "alto" else: vector_df = pd.DataFrame([{ "pasos": row.get("steps_daily_mean"), "sueno": row.get("sleep_minutes_mean"),</mark> 

<mark>"tiempo_casa": row.get("home_minutes_mean"), "uso_apps": row.get("app_use_minutes_mean"), "PHQ9": row.get("phq9_total"), "GAD7": row.get("gad7_total"), "ULS6": row.get("uls6_total"), "CSSRS": cssrs, "phq9_item9": p9, }])[FEATURES_ORDER] probas = pipeline.predict_proba(vector_df)[0] score = float(probas[2])  # Probabilidad de clase 'Alto' risk_level = "alto" if score >= 0.35 else ("moderado" if probas[1] >= 0.35 else "bajo") predicciones_payload.append({ "participant_code": code, "week_start": week_start, "model_name": "random_forest", "model_version": "1.0", "score": round(score, 4), "threshold": 0.35, "risk_level": risk_level }) # 4. Enviar predicciones a la API resp_pred = await client.post("/predictions", headers=headers, json={"predictions": predicciones_payload}) resp_pred.raise_for_status() print("Resultado de la inserción de predicciones:", resp_pred.json()) if __name__ == "__main__": asyncio.run(ejecutar_ciclo_inferencia())</mark> 

## **4. Pruebas Automatizadas y Validación (Testing)** 

Crea el archivo tests/test_ml_inference.py para verificar que el pipeline de ML respeta el gatillo clínico y se integra correctamente con las clases de riesgo del backend: 

<mark>import pytest import pandas as pd</mark> 

<mark>from scripts.entrenamiento_riesgo import normalizar_columnas, generar_ground_truth</mark> 

<mark>def test_normalizacion_de_columnas_alias(): raw_df = pd.DataFrame([{ "steps_daily_mean": 5000, "sleep_minutes_mean": 400, "home_minutes_mean": 600, "app_use_minutes_mean": 120, "phq9_total": 12, "gad7_total": 8, "uls6_total": 15, "cssrs_level": 1, "phq9_item9": 0 }]) df_std = normalizar_columnas(raw_df) assert "pasos" in df_std.columns assert "sueno" in df_std.columns assert df_std.loc[0, "pasos"] == 5000</mark> 

<mark>def test_clinical_bypass_fuerza_riesgo_alto(): bypass_df = pd.DataFrame([{ "pasos": 10000, "sueno": 480, "tiempo_casa": 400, "uso_apps": 50, "PHQ9": 2, "GAD7": 1, "ULS6": 6, "CSSRS": 0, "phq9_item9": 1 }]) target = generar_ground_truth(bypass_df) assert target.iloc[0] == 2  # Código 2 = Riesgo Alto por Bypass</mark> 

#### **Para ejecutar la suite de pruebas del backend y de ML:** 

<mark>pytest tests/test_ml_inference.py tests/test_scoring.py</mark> 

## **5. Pasos para el Despliegue en Docker / Servidor** 

1. **Crear Usuario de Servicio en la BD:** Ejecutar el script para dar de alta al usuario con rol service: 

<mark>docker compose exec api python -m scripts.create_admin --email service@unsa.edu.pe --name "Microservicio ML" --role service</mark> 

2. **Generar Datos y Entrenar Modelo Inicial:** 

<mark>python -m scripts.generar_datos_sinteticos --n 500</mark> 

<mark>python -m scripts.entrenamiento_riesgo</mark> 

3. **Ejecutar Inferencia Programada (Cron/Task):** Programar la ejecución semanal de scripts/ml_service_client.py. 

