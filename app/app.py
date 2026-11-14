"""
API y servidor web del prototipo «¿Héroe o villano?» (ACIF104).

Endpoints
  GET  /                      Front end (static/index.html)
  GET  /api/salud             Estado del servicio y del modelo
  GET  /api/opciones          Valores válidos del formulario, modelo y métricas de prueba
  POST /api/predecir          Predicción + explicación por variable (SHAP o, si no está instalado, oclusión)
  POST /api/retroalimentacion Registra la alineación real de una predicción (para medir el desempeño en uso)
  GET  /api/monitoreo         Métricas de uso, latencia, deriva de datos y exactitud con retroalimentación
"""
import csv
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

BASE = Path(__file__).resolve().parent
MODELO = Path(os.environ.get("MODELO_PATH", BASE / "modelo" / "app_modelo.joblib"))
DIR_MON = Path(os.environ.get("MONITOREO_DIR", BASE / "monitoreo"))
DIR_MON.mkdir(parents=True, exist_ok=True)
LOG_PRED = DIR_MON / "predicciones.csv"
LOG_FB = DIR_MON / "retroalimentacion.csv"
UMBRAL_DERIVA_PP = 15.0   # alerta si el % de faltantes de una variable se aleja más de 15 puntos de la referencia
_lock = Lock()

paquete = joblib.load(MODELO)
PREP, MOD = paquete["prep"], paquete["modelo"]
CAT, NUM = paquete["categoricas"], paquete["numericas"]
VARIABLES = paquete["variables"]
OPCIONES = paquete["opciones"]
REF = paquete["referencia"]
COLUMNAS = paquete["nombres_columnas"]

try:  # SHAP es opcional: si no está instalado se usa explicación por oclusión
    import shap
    if hasattr(MOD, "estimators_") or MOD.__class__.__name__.startswith("DecisionTree"):
        _EXPLICADOR = shap.TreeExplainer(MOD)
    else:
        _f = (lambda Z: MOD.predict_proba(Z)[:, 1]) if hasattr(MOD, "predict_proba") else MOD.decision_function
        _EXPLICADOR = shap.Explainer(_f, paquete["fondo_shap"], algorithm="permutation")
    METODO_EXPLICACION = "SHAP"
except Exception:  # noqa: BLE001
    _EXPLICADOR = None
    METODO_EXPLICACION = "Oclusión (reemplazo por valor típico)"

app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024  # las solicitudes son pequeñas: se rechaza lo que exceda 16 KB


def variable_original(col):
    for v in VARIABLES:
        if col == v or col.startswith(v + "_") or col == f"missingindicator_{v}":
            return v
    return col


def probabilidad(df):
    Z = PREP.transform(df)
    if hasattr(MOD, "predict_proba"):
        return MOD.predict_proba(Z)[:, 1]
    d = MOD.decision_function(Z)
    return 1 / (1 + np.exp(-d))


def validar(entrada):
    """Valida y normaliza la entrada. Devuelve (fila, errores)."""
    errores, fila = [], {}
    for c in CAT:
        v = entrada.get(c)
        if v in (None, "", "No informado"):
            fila[c] = np.nan
        elif v in OPCIONES[c]:
            fila[c] = v
        else:
            errores.append(f"Valor no válido para {c}")
    for c, (mn, mx) in {"APPEARANCES": (1, 10000), "YEAR": (1935, 2030)}.items():
        if c not in NUM:
            continue
        v = entrada.get(c)
        if v in (None, ""):
            fila[c] = np.nan
            continue
        try:
            v = float(v)
        except (TypeError, ValueError):
            errores.append(f"{c} debe ser numérico")
            continue
        if not mn <= v <= mx:
            errores.append(f"{c} debe estar entre {mn} y {mx}")
        fila[c] = v
    if errores:
        return None, errores
    return pd.DataFrame([fila])[VARIABLES], errores


def explicar(df, p):
    if _EXPLICADOR is not None:
        sv = _EXPLICADOR(pd.DataFrame(PREP.transform(df), columns=COLUMNAS))
        vals = sv.values[0]
        if vals.ndim == 2:  # (variables, clases)
            vals = vals[:, 1]
        contrib = pd.Series(vals, index=COLUMNAS).groupby([variable_original(c) for c in COLUMNAS]).sum()
    else:
        contrib = {}
        for v in VARIABLES:
            alt = df.copy()
            alt[v] = REF["moda"][v] if v in CAT else (np.expm1(REF["mediana_log_apariciones"]) if v == "APPEARANCES" else REF["mediana_anio"])
            contrib[v] = float(p - probabilidad(alt)[0])
        contrib = pd.Series(contrib)
    contrib = contrib.reindex(VARIABLES).fillna(0.0)
    orden = contrib.abs().sort_values(ascending=False).index
    return [{"variable": v, "aporte": round(float(contrib[v]), 4)} for v in orden]


def registrar(path, fila):
    path.parent.mkdir(parents=True, exist_ok=True)
    nuevo = not path.exists()
    with _lock, open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fila))
        if nuevo:
            w.writeheader()
        w.writerow(fila)


@app.get("/")
def inicio():
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api/salud")
def salud():
    return jsonify({"estado": "ok", "modelo": paquete["nombre_modelo"], "explicacion": METODO_EXPLICACION})


@app.get("/api/opciones")
def opciones():
    return jsonify({"opciones": OPCIONES, "modelo": paquete["nombre_modelo"],
                    "metricas_prueba": paquete["metricas_prueba"], "explicacion": METODO_EXPLICACION})


@app.post("/api/predecir")
def predecir():
    t0 = time.perf_counter()
    entrada = request.get_json(silent=True) or {}
    df, errores = validar(entrada)
    if errores:
        return jsonify({"errores": errores}), 400
    p = float(probabilidad(df)[0])
    explicacion = explicar(df, p)
    latencia = (time.perf_counter() - t0) * 1000
    pid = uuid.uuid4().hex[:12]
    registrar(LOG_PRED, {"id": pid, "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         **{c: ("" if pd.isna(df.iloc[0][c]) else df.iloc[0][c]) for c in VARIABLES},
                         "prob_heroe": round(p, 4), "clase": "Héroe" if p >= 0.5 else "Villano",
                         "latencia_ms": round(latencia, 1)})
    return jsonify({"id": pid, "prob_heroe": round(p, 4), "clase": "Héroe" if p >= 0.5 else "Villano",
                    "explicacion": explicacion, "metodo_explicacion": METODO_EXPLICACION,
                    "modelo": paquete["nombre_modelo"], "latencia_ms": round(latencia, 1)})


@app.post("/api/retroalimentacion")
def retroalimentacion():
    d = request.get_json(silent=True) or {}
    if d.get("real") not in ("Héroe", "Villano") or not str(d.get("id", "")).isalnum():
        return jsonify({"error": "Se requiere id y real = 'Héroe' o 'Villano'"}), 400
    registrar(LOG_FB, {"id": d["id"], "real": d["real"], "fecha": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    return jsonify({"ok": True})


@app.get("/api/monitoreo")
def monitoreo():
    if not LOG_PRED.exists():
        return jsonify({"n_predicciones": 0, "referencia": REF, "metricas_prueba": paquete["metricas_prueba"], "alertas": []})
    log = pd.read_csv(LOG_PRED)
    alertas, deriva = [], []
    for c in VARIABLES:
        actual = 100 * log[c].isna().mean()
        ref = REF["faltantes_pct"][c]
        deriva.append({"variable": c, "faltantes_ref_pct": ref, "faltantes_uso_pct": round(actual, 1)})
        if len(log) >= 30 and abs(actual - ref) > UMBRAL_DERIVA_PP:
            alertas.append(f"Deriva en {c}: {actual:.0f} % de faltantes en uso vs {ref:.0f} % en entrenamiento")
    prop_heroe = float((log["clase"] == "Héroe").mean())
    if len(log) >= 30 and abs(prop_heroe - REF["prop_buenos_predicha_prueba"]) > 0.20:
        alertas.append("La proporción de héroes predichos se alejó más de 20 puntos de la observada en prueba")
    exactitud_uso, n_fb = None, 0
    if LOG_FB.exists():
        fb = pd.read_csv(LOG_FB).drop_duplicates("id", keep="last").merge(log[["id", "clase"]], on="id")
        n_fb = len(fb)
        if n_fb:
            exactitud_uso = round(float((fb["real"] == fb["clase"]).mean()), 4)
            if n_fb >= 30 and exactitud_uso < paquete["metricas_prueba"]["Exactitud"] - 0.10:
                alertas.append("La exactitud en uso cayó más de 10 puntos respecto de la prueba: revisar o reentrenar")
    return jsonify({
        "n_predicciones": int(len(log)),
        "prop_heroe_predicha": round(prop_heroe, 4),
        "latencia_media_ms": round(float(log["latencia_ms"].mean()), 1),
        "latencia_p95_ms": round(float(log["latencia_ms"].quantile(0.95)), 1),
        "deriva_faltantes": deriva,
        "mediana_log_apariciones_uso": round(float(np.log1p(log["APPEARANCES"]).median()), 3) if log["APPEARANCES"].notna().any() else None,
        "referencia": {"mediana_log_apariciones": round(REF["mediana_log_apariciones"], 3), "prop_buenos_predicha_prueba": round(REF["prop_buenos_predicha_prueba"], 4)},
        "n_retroalimentacion": n_fb, "exactitud_en_uso": exactitud_uso,
        "metricas_prueba": paquete["metricas_prueba"],
        "ultimas": log.tail(10).iloc[::-1].fillna("").to_dict(orient="records"),
        "alertas": alertas,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
