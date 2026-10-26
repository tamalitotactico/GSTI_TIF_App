# Integración del módulo de ML con el backend

Versión unificada y corregida de la *Guía de Integración Técnica y Tutorial de Despliegue: Modelo ML en Backend*. El código vive en `scripts/` y las pruebas en `tests/`; esta guía no lo duplica.

## 1. Archivos

| Archivo | Función |
|---|---|
| `scripts/generar_datos_sinteticos.py` | Dataset sintético persona-semana en las mismas unidades que la API |
| `scripts/entrenamiento_riesgo.py` | Pipeline k-NN (k = 5) → Z-score → Random Forest (o XGBoost); validación agrupada por participante; guarda el modelo |
| `scripts/ml_service_client.py` | Microservicio de inferencia: `GET /features/person-week` → predicción → `POST /predictions` |
| `scripts/models_saved/modelo.joblib` | Modelo entrenado (generado, no se versiona) |
| `tests/test_ml_inference.py` | Pruebas de etiquetas, entrenamiento, construcción de predicciones e integración con la API |
| `requirements-ml.txt` | Dependencias del módulo (fuera del contenedor de la API) |

## 2. Flujo

1. **Extracción:** el microservicio inicia sesión con una cuenta de rol `service` y llama a `GET /api/v1/features/person-week`. La API lee la vista `analytics.v_person_week_features`: pasos diarios, minutos de sueño, minutos en casa, minutos con uso de apps, PHQ-9, GAD-7, ULS-6 y nivel C-SSRS (los 8 indicadores del modelo), además de `phq9_item9` y la etiqueta `label_high_risk`. Los faltantes llegan como `null`.
2. **Inferencia:** el pipeline imputa los faltantes, estandariza y devuelve la probabilidad de cada clase. El nivel se asigna así:
   - gatillo clínico (PHQ-9 ítem 9 > 0 o C-SSRS ≥ 3) → **alto**;
   - probabilidad de "alto" ≥ 0.35 → **alto**;
   - si no, probabilidad de "moderado" ≥ 0.35 → **moderado**;
   - en cualquier otro caso → **bajo**.

   Se envía siempre la probabilidad real del modelo, nunca un 1.0 fijo para los casos de gatillo: así las métricas de H1 y H2 no quedan sesgadas.
3. **Persistencia:** `POST /api/v1/predictions` guarda en `core.model_predictions`. El servidor es la autoridad: vuelve a aplicar el gatillo clínico y el umbral. Cuando una semana pasa a "alto" **por el modelo**, crea una alerta `model_high_risk` y la envía al tablero por `/ws/dashboard`. Si la semana es "alto" **por gatillo clínico**, no crea otra alerta, porque ya se emitió al recibir el cuestionario.

## 3. Puesta en marcha

```bash
pip install -r requirements-ml.txt

# 1. Cuenta de servicio (una vez)
docker compose exec api python -m scripts.create_admin --email service@unsa.edu.pe --name "Microservicio ML" --role service

# 2. Datos sintéticos y entrenamiento inicial
python -m scripts.generar_datos_sinteticos --n 2000
python -m scripts.entrenamiento_riesgo            # --model xgb para el modelo de contraste

# 3. Inferencia (programarla semanalmente con cron o el Programador de tareas de Windows)
SERVICE_PASSWORD='...' API_BASE_URL=http://localhost:8000/api/v1 python -m scripts.ml_service_client
#    --dry-run calcula sin enviar; --since AAAA-MM-DD limita las semanas
```

Pruebas: `pytest` (backend y ML juntos, 42 pruebas).

## 4. Cambios respecto de la guía original

| Problema | Corrección |
|---|---|
| El cliente enviaba `risk_level: "alto"` y la API lo rechazaba (error 422) | La API acepta los tres niveles y decide con el umbral |
| `generar_ground_truth` fallaba si todas las filas eran de gatillo clínico (la prueba incluida no pasaba) | Se calcula el percentil solo si quedan filas sin gatillo |
| El modelo se entrenaba con los casos de gatillo, cuya etiqueta es la propia regla: aprendía algo trivial e inflaba las métricas | El modelo se entrena y evalúa solo con semanas sin gatillo, que son las únicas que decide |
| `phq9_item9` como predictor (9 variables) | Se quitó: en las semanas que evalúa el modelo siempre vale 0. Quedan los 8 indicadores del artículo |
| División aleatoria 80/20 con `accuracy` | `GroupKFold` por participante, con sensibilidad, especificidad, F1 y AUC de la clase "alto" a umbral 0.35 (3.10) |
| Índice fijo `probas[2]` | Se usa `pipeline.classes_` |
| Probabilidad 1.0 inventada para los casos de gatillo | Se envía la probabilidad real; el servidor marca `clinical_bypass` |
| Uso de apps y tiempo en casa en escalas distintas de las del backend (10–400 min) | Rangos sintéticos en las unidades de la API: bloques de 30 min × 30 |
| Contraseña de servicio con valor por defecto en el código | `SERVICE_PASSWORD` obligatoria por variable de entorno |
| Sin XGBoost (4.6 lo menciona) | `--model xgb` usa el mismo pipeline (patrón strategy) |
| Dependencias de ML no declaradas; modelo y CSV sin ignorar en git | `requirements-ml.txt`; artefactos agregados a `.gitignore` |

## 5. Advertencias metodológicas (para el grupo)

1. **La etiqueta sintética es circular.** `generar_ground_truth` calcula la clase con una fórmula ponderada de las mismas variables que recibe el modelo. Por eso la validación da métricas cercanas a 0.99. **Esos números prueban que el flujo funciona, no son resultados del estudio** y no deben reportarse en la sección 5.
2. **Con datos reales, la etiqueta del artículo no sirve para entrenar.** El artículo define la etiqueta como C-SSRS ≥ 3 o ítem 9 del PHQ-9 > 0, que es exactamente la regla del gatillo clínico. Esas semanas nunca llegan al modelo, y en las demás la etiqueta siempre es negativa. El modelo no tendría nada que aprender. Opciones:
   - **(a) Predicción anticipada:** usar los indicadores de la semana *t* para predecir la etiqueta de la semana *t + 1*. Esto encaja con el objetivo de detección temprana y la vista persona-semana ya lo permite.
   - **(b) Referencia independiente:** una evaluación clínica estructurada hecha por un profesional, o un instrumento semanal distinto de los predictores.

   Esta decisión cambia 3.10 y 3.11 y conviene tomarla antes de recolectar datos.
3. **LOSO-CV.** El artículo menciona *Leave-One-Subject-Out*. `GroupKFold` con 5 pliegues es la aproximación práctica: un participante nunca está a la vez en entrenamiento y evaluación. Con la muestra real (100–150 participantes) también puede correrse LOSO completo.
