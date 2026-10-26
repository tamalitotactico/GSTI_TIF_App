# Pendientes para consolidar el backend GSTI (Claude Code)

Contexto: repo `tamalitotactico/gsti_back_datos`, rama `main`. Stack: FastAPI + PostgreSQL 16 + Alembic + Docker Compose; módulo de ML en `scripts/`. Lee primero `README.md`, `docs/contrato-api.md` y `docs/integracion-ml.md`. Hoy pasan 42 pruebas (`pytest`).

Reglas generales:
- Trabaja en una rama `consolidacion`; un commit por tarea; `pytest` en verde antes de cada commit.
- No cambies el contrato JSON de la app (docs/contrato-api.md) sin actualizar el documento y las pruebas.
- No subas `.env`, secretos, el CSV ni el `.joblib`.
- Ningún texto ni salida debe presentar una alerta como diagnóstico.

## 1. Levantar y verificar el entorno local (Windows + Docker Desktop)
- [ ] Copiar `.env.example` a `.env` y generar los secretos con los comandos indicados en el propio archivo: `JWT_SECRET`, `IDENTITY_ENCRYPTION_KEY`, `IDENTITY_HMAC_KEY` y las contraseñas.
- [ ] `docker compose up -d --build`; comprobar que `GET /health` responde `ok` y que `/docs` carga.
- [ ] Crear el usuario admin y el usuario `service` con `scripts.create_admin`.
- [ ] Correr `scripts.simulate_device` y, después, el ciclo ML (`generar_datos_sinteticos` → `entrenamiento_riesgo` → `ml_service_client`).
- [ ] Revisar `/api/v1/metrics/latency` y `/api/v1/predictions`.
- [ ] Si algo falla en Windows (finales de línea CRLF en `db/init/01_roles.sh`, puertos, etc.), corregirlo en el repo. Por ejemplo, agregar `.gitattributes` con `*.sh text eol=lf`.
- Criterio: el flujo completo corre en la PC sin pasos manuales fuera del README.

## 2. Pruebas contra Docker
- [ ] Documentar en el README cómo correr `pytest` contra el contenedor `db`, usando `TEST_ADMIN_DSN` con `gsti_owner`. Verificar que funciona.
- [ ] Agregar un workflow de GitHub Actions (`.github/workflows/tests.yml`) con un servicio `postgres:16` que ejecute `pip install -r requirements-dev.txt` y luego `pytest`.

## 3. Endurecimiento antes del piloto
- [ ] Limitar los intentos en `POST /auth/login` y `POST /devices/register`, por ejemplo con `slowapi` o un middleware propio: 5 intentos/min por IP en login y 10 por IP en register. Agregar pruebas.
- [ ] Al iniciar con `ENVIRONMENT=production`, la app debe fallar si `JWT_SECRET`, `IDENTITY_HMAC_KEY` o `IDENTITY_ENCRYPTION_KEY` están vacíos o tienen el valor de desarrollo.
- [ ] Agregar Caddy al `docker-compose.yml` (perfil `prod`) como proxy inverso con HTTPS automático. En producción, el puerto 8000 no debe quedar expuesto.
- [ ] Respaldo: script `scripts/backup_db.sh` con `pg_dump` y un ejemplo de tarea programada. Documentar que la clave Fernet se respalda aparte.

## 4. Microservicio ML en Docker
- [ ] Crear `Dockerfile.ml` y un servicio `ml` en compose (perfil `ml`) que ejecute `scripts.ml_service_client` contra `http://api:8000/api/v1`, usando `SERVICE_PASSWORD` desde `.env`.
- [ ] Documentar cómo programarlo cada semana: cron en Linux o el Programador de tareas en Windows.

## 5. Etiqueta para predicción anticipada (preparar, sin cambiar la metodología)
- [ ] Agregar en la vista o en el endpoint de features la columna `label_next_week`: la etiqueta de referencia de la semana t+1 del mismo participante, o `null` si no existe. Hacerlo con una migración nueva (0005), sin editar las anteriores.
- [ ] En `scripts/entrenamiento_riesgo.py`, agregar `--label next_week` para entrenar con datos de la API (`--source api`) usando esa columna. Mantener la etiqueta sintética como opción por defecto.
- [ ] Agregar pruebas. La decisión final la toma el grupo (ver `docs/integracion-ml.md` §5).

## 6. Limpieza
- [ ] Reemplazar `datetime.utcnow`/avisos de deprecación si aparecen. Revisar que `ruff check .` pase, si se agrega ruff.
- [ ] Actualizar en el README el conteo de pruebas y la sección «Limitaciones conocidas».

## Fuera de alcance (no hacer)
- App móvil, chatbot y tablero.
- Puntos de corte clínicos: los define el equipo clínico mediante `.env`.
- Redacción del artículo.

## Entrega
Al terminar, abre un PR de `consolidacion` hacia `main` con un resumen por tarea, el resultado de `pytest` y lo que quedó pendiente con su motivo.
