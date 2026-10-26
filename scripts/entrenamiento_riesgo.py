"""Entrenamiento del clasificador de riesgo (artículo 3.10, 3.11 y 4.6).

Pipeline: imputación k-NN (k = 5) -> estandarización Z-score -> Random Forest (o XGBoost).

Decisiones de diseño:
- Gatillo clínico como cláusula de guarda (4.6.2): las semanas con PHQ-9 ítem 9 > 0 o
  C-SSRS >= 3 se clasifican "alto" por regla y nunca llegan al modelo. Por eso el
  clasificador se entrena SOLO con semanas sin gatillo: aprender la regla sería trivial
  y inflaría las métricas.
- Validación agrupada por participante (GroupKFold) para que un mismo estudiante no
  aparezca en entrenamiento y evaluación (aproximación práctica a LOSO-CV, 3.10).
- Métricas de la clase "alto" con el umbral de 0.35: sensibilidad, especificidad, F1, AUC.

ADVERTENCIA: `generar_ground_truth` produce una etiqueta HEURÍSTICA a partir de las mismas
variables de entrada. Sirve para probar el flujo con datos sintéticos; sus métricas NO
son resultados del estudio. Con datos reales, la etiqueta debe venir de una referencia
independiente (ver docs/integracion-ml.md).

Uso:
    python -m scripts.entrenamiento_riesgo --csv scripts/datos_estudiantes.csv --model rf
"""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Ocho indicadores del artículo (3.10). El ítem 9 del PHQ-9 no entra: en las semanas
# que evalúa el modelo siempre vale 0 (si fuera > 0 habría actuado el gatillo clínico).
FEATURES_VECTOR = ["pasos", "sueno", "tiempo_casa", "uso_apps", "PHQ9", "GAD7", "ULS6", "CSSRS"]
CLASES = {0: "bajo", 1: "moderado", 2: "alto"}
UMBRAL_ALTO = 0.35
CSSRS_BYPASS = 3

ALIAS = {
    "pasos": ["pasos", "pasos_diarios", "steps", "steps_daily_mean"],
    "sueno": ["sueno", "minutos_sueno", "sleep_minutes", "sleep_minutes_mean"],
    "tiempo_casa": ["tiempo_casa", "tiempo_casa_min", "home_minutes", "home_minutes_mean"],
    "uso_apps": ["uso_apps", "bloques_uso_apps", "app_use_minutes", "app_use_minutes_mean"],
    "PHQ9": ["PHQ9", "phq9_total", "phq9"],
    "GAD7": ["GAD7", "gad7_total", "gad7"],
    "ULS6": ["ULS6", "uls6_total", "uls6"],
    "CSSRS": ["CSSRS", "cssrs_level", "cssrs_nivel"],
    "phq9_item9": ["phq9_item9", "phq9_i9", "item9"],
    "estudiante_id": ["estudiante_id", "participant_code"],
}


def normalizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Unifica los nombres del CSV sintético y de la API (/features/person-week)."""
    df_std = df.copy()
    for col_std, aliases in ALIAS.items():
        encontrado = next((a for a in aliases if a in df_std.columns), None)
        if encontrado and encontrado != col_std:
            df_std = df_std.rename(columns={encontrado: col_std})
        elif not encontrado:
            df_std[col_std] = 0.0 if col_std == "phq9_item9" else np.nan
    return df_std


def mascara_bypass(df_std: pd.DataFrame) -> pd.Series:
    """Gatillo clínico directo: PHQ-9 ítem 9 > 0 o C-SSRS >= 3 (mismo criterio que el backend)."""
    return (df_std["phq9_item9"].fillna(0) > 0) | (df_std["CSSRS"].fillna(0) >= CSSRS_BYPASS)


def generar_ground_truth(df: pd.DataFrame) -> pd.Series:
    """Etiqueta heurística SOLO para datos sintéticos: 0 = bajo, 1 = moderado, 2 = alto."""
    df_std = normalizar_columnas(df)
    bypass = mascara_bypass(df_std)

    score = (
        0.30 * df_std["PHQ9"].fillna(0) / 27.0
        + 0.25 * df_std["GAD7"].fillna(0) / 21.0
        + 0.20 * (df_std["ULS6"].fillna(6) - 6) / 18.0
        + 0.10 * (1.0 - df_std["pasos"].fillna(8500) / 25000.0)
        + 0.05 * (1.0 - df_std["sueno"].fillna(420) / 600.0)
        + 0.05 * df_std["tiempo_casa"].fillna(800) / 1440.0
        + 0.05 * df_std["uso_apps"].fillna(540) / 1080.0
    )

    target = pd.Series(0, index=df_std.index, dtype=int)
    target[bypass] = 2
    resto = score[~bypass]
    if len(resto):  # antes fallaba si todas las filas eran de gatillo clínico
        p50, p75 = np.percentile(resto, [50, 75])
        target[~bypass] = np.where(resto >= p75, 2, np.where(resto >= p50, 1, 0))
    return target


def construir_pipeline(modelo: str = "rf", seed: int = 42) -> Pipeline:
    """Patrón strategy (4.6.2): mismo pipeline, clasificador intercambiable."""
    if modelo == "xgb":
        try:
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise SystemExit("Instala xgboost (pip install xgboost) para usar --model xgb") from exc
        clf = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, random_state=seed,
                            eval_metric="mlogloss")
    else:
        clf = RandomForestClassifier(n_estimators=100, random_state=seed, class_weight="balanced")
    return Pipeline([
        ("knn_imputer", KNNImputer(n_neighbors=5)),
        ("scaler", StandardScaler()),
        ("classifier", clf),
    ])


def indice_clase(pipeline: Pipeline, clase: int) -> int:
    return list(pipeline.classes_).index(clase)


def evaluar_agrupado(X: pd.DataFrame, y: pd.Series, grupos: pd.Series, modelo: str, k: int = 5) -> dict:
    """Validación cruzada agrupada por participante; métricas de la clase 'alto' (umbral 0.35)."""
    k = min(k, grupos.nunique())
    prob = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=k).split(X, y, grupos):
        p = construir_pipeline(modelo).fit(X.iloc[tr], y.iloc[tr])
        prob[te] = p.predict_proba(X.iloc[te])[:, indice_clase(p, 2)]
    real = (y.values == 2).astype(int)
    pred = (prob >= UMBRAL_ALTO).astype(int)
    vp, fn = int(((pred == 1) & (real == 1)).sum()), int(((pred == 0) & (real == 1)).sum())
    vn, fp = int(((pred == 0) & (real == 0)).sum()), int(((pred == 1) & (real == 0)).sum())
    return {
        "pliegues": k, "n": int(len(y)), "positivos_alto": int(real.sum()),
        "sensibilidad": round(vp / (vp + fn), 4) if vp + fn else None,
        "especificidad": round(vn / (vn + fp), 4) if vn + fp else None,
        "f1": round(f1_score(real, pred), 4),
        "auc_roc": round(roc_auc_score(real, prob), 4) if 0 < real.sum() < len(real) else None,
        "umbral": UMBRAL_ALTO,
    }


def entrenar_y_guardar(csv_path: str = "scripts/datos_estudiantes.csv",
                       model_path: str = "scripts/models_saved/modelo.joblib",
                       modelo: str = "rf") -> dict:
    df = normalizar_columnas(pd.read_csv(csv_path))
    y_total = generar_ground_truth(df)
    sin_gatillo = ~mascara_bypass(df)
    X = df.loc[sin_gatillo, FEATURES_VECTOR]
    y = y_total[sin_gatillo]
    ids = df.loc[sin_gatillo, "estudiante_id"]
    grupos = ids.where(ids.notna(), X.index.to_series().astype(str))  # sin id: cada fila es su propio grupo

    metricas = evaluar_agrupado(X, y, grupos, modelo)
    metricas["excluidas_por_gatillo"] = int((~sin_gatillo).sum())
    print("Validación agrupada por participante (clase 'alto'):")
    print(json.dumps(metricas, indent=2, ensure_ascii=False))

    final = construir_pipeline(modelo).fit(X, y)
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final, "features": FEATURES_VECTOR, "modelo": modelo,
                 "umbral": UMBRAL_ALTO, "metricas_cv": metricas}, model_path)
    print(f"Modelo guardado en: {model_path}")
    return metricas


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Entrena el clasificador de riesgo GSTI.")
    ap.add_argument("--csv", default="scripts/datos_estudiantes.csv")
    ap.add_argument("--out", default="scripts/models_saved/modelo.joblib")
    ap.add_argument("--model", choices=["rf", "xgb"], default="rf")
    a = ap.parse_args()
    entrenar_y_guardar(a.csv, a.out, a.model)
