"""Genera un dataset sintético persona-semana para desarrollo y pruebas del modelo.

Datos 100 % sintéticos. Las columnas usan los nombres del equipo de ML; el entrenamiento
los unifica con los de la API mediante normalizar_columnas().

Uso:
    python -m scripts.generar_datos_sinteticos --n 2000 --output scripts/datos_estudiantes.csv
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

# Rangos en las MISMAS unidades que analytics.v_person_week_features del backend:
# pasos/día; minutos de sueño/noche; minutos en casa/día y minutos con uso de apps/día,
# ambos calculados como (bloques de 30 min con valor 1) x 30.
RANGO_PASOS = (1000, 25000)
RANGO_SUENO = (180, 600)
RANGO_CASA = (300, 1440)
RANGO_APPS = (60, 1080)

def configurar_logging() -> None:
    logger = logging.getLogger()
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.StreamHandler(sys.stdout)],
        )

def generar_dataset_sintetico(n: int = 500, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)

    # Formato de seudónimo P0001 alineado a la base de datos core.participants
    estudiantes_ids = [f"P{i+1:04d}" for i in range(n)]
    semanas = np.random.randint(1, 17, size=n)
    z = np.random.normal(0, 1, n)

    phq9 = np.clip(np.round(np.exp(1.5 + 0.6 * z) - 1), 0, 27).astype(int)
    gad7 = np.clip(np.round(np.exp(1.3 + 0.55 * z) - 1), 0, 21).astype(int)
    uls6 = np.clip(np.round(6 + np.exp(1.8 + 0.45 * z) - 1), 6, 24).astype(int)

    pasos = np.clip(np.round(8500 - 2500 * z + np.random.normal(0, 1200, n)), RANGO_PASOS[0], RANGO_PASOS[1])
    sueno = np.clip(np.round(420 - 55 * z + np.random.normal(0, 40, n)), RANGO_SUENO[0], RANGO_SUENO[1])
    tiempo_casa = np.clip(np.round(800 + 160 * z + np.random.normal(0, 80, n)), RANGO_CASA[0], RANGO_CASA[1])
    uso_apps = np.clip(np.round(540 + 130 * z + np.random.normal(0, 60, n)), RANGO_APPS[0], RANGO_APPS[1])

    prob_bypass = 1 / (1 + np.exp(-(z - 1.8))) * 0.35
    es_bypass = np.random.rand(n) < prob_bypass

    cssrs = np.zeros(n, dtype=int)
    phq9_item9 = np.zeros(n, dtype=int)

    for i in range(n):
        if es_bypass[i]:
            if np.random.rand() > 0.5:
                cssrs[i] = np.random.choice([3, 4, 5])
                phq9_item9[i] = np.random.choice([0, 1, 2, 3])
            else:
                cssrs[i] = np.random.choice([0, 1, 2])
                phq9_item9[i] = np.random.choice([1, 2, 3])
        else:
            cssrs[i] = np.random.choice([0, 1, 2], p=[0.80, 0.15, 0.05])
            phq9_item9[i] = 0

    df = pd.DataFrame({
        "estudiante_id": estudiantes_ids,
        "semana": semanas,
        "pasos": pasos,
        "sueno": sueno,
        "tiempo_casa": tiempo_casa,
        "uso_apps": uso_apps,
        "PHQ9": phq9,
        "GAD7": gad7,
        "ULS6": uls6,
        "CSSRS": cssrs,
        "phq9_item9": phq9_item9,
    })

    for col in ["pasos", "sueno", "tiempo_casa", "uso_apps"]:
        mask_nulos = np.random.rand(n) < np.random.uniform(0.05, 0.08)
        df.loc[mask_nulos, col] = np.nan

    return df

def main(args_list: Optional[list] = None) -> pd.DataFrame:
    configurar_logging()

    parser = argparse.ArgumentParser(description="Generador de datos sintéticos.")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--output", type=str, default="scripts/datos_estudiantes.csv")
    parser.add_argument("--seed", type=int, default=42)

    if args_list is not None:
        args, _ = parser.parse_known_args(args_list)
    else:
        args, _ = parser.parse_known_args()

    path_output = Path(args.output)
    path_output.parent.mkdir(parents=True, exist_ok=True)
    df = generar_dataset_sintetico(args.n, args.seed)
    df.to_csv(path_output, index=False)
    logging.info(f"Dataset generado exitosamente con {len(df)} registros en {path_output}")
    return df

if __name__ == "__main__":
    main()