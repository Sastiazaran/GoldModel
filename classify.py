"""Clasifica el oro a 5 y 20 días: alza, plano o baja.

La banda plana es media desviación de la volatilidad ya conocida, multiplicada
por la raíz del horizonte. Un movimiento chico, dentro de esa banda, no cuenta
como acierto de dirección.

En cada reentreno se ajustan una logística, un bosque y LightGBM, solo con
filas purgadas. La probabilidad de alza se calibra con una isotónica en el
tramo final de ese entrenamiento. Qué modelo se publica se decide en ese
tramo, no mirando el examen final.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, brier_score_loss, roc_auc_score

from evaluacion import indices_purgados, limite_pasado
from modelos import _tomar_no_solapadas, retorno_log_pasado

K_BANDA = 0.5
MODELOS = ("logistica", "bosque", "lightgbm")


def volatilidad_conocida(precios: pd.Series, ventana: int = 20) -> pd.Series:
    """Desviación de los retornos diarios ya ocurridos. No mira hacia adelante."""
    ret1 = retorno_log_pasado(precios.astype(float), 1)
    return ret1.rolling(ventana, min_periods=ventana).std(ddof=0)


def etiquetar_direccion(
    y: np.ndarray,
    vol: np.ndarray,
    horizonte: int,
    k: float = K_BANDA,
) -> tuple[np.ndarray, np.ndarray]:
    """1 alza, 0 plano, -1 baja. NaN si falta el retorno o la volatilidad."""
    y = np.asarray(y, dtype=float)
    vol = np.asarray(vol, dtype=float)
    banda = k * vol * np.sqrt(horizonte)
    clase = np.full(len(y), np.nan)
    m = np.isfinite(y) & np.isfinite(banda) & (banda >= 0)
    clase[m] = np.where(y[m] > banda[m], 1.0, np.where(y[m] < -banda[m], -1.0, 0.0))
    return clase, banda


def armar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Columnas conocidas en t. La brecha de Kalman entra si ya está calculada."""
    datos: dict[str, pd.Series] = {}

    def poner(nombre: str, serie) -> None:
        if serie is None:
            return
        valores = pd.Series(serie, index=df.index, dtype=float)
        if int(np.isfinite(valores.to_numpy(dtype=float)).sum()) < 30:
            return
        datos[nombre] = valores

    for nombre in ("ret_5", "ret_20", "ret_60", "d_dolar_20", "d_brent_20", "d_vix_5", "vix", "rendimiento_real_10"):
        if nombre in df.columns:
            poner(nombre, df[nombre])
    if "gvz" in df.columns:
        poner("gvz", df["gvz"])
        poner("d_gvz_5", df["gvz"].astype(float) - df["gvz"].astype(float).shift(5))
    if "breakeven_10" in df.columns:
        poner("breakeven_10", df["breakeven_10"])
        poner("d_be_20", df["breakeven_10"].astype(float) - df["breakeven_10"].astype(float).shift(20))
    if "cot_neto_oi" in df.columns:
        poner("cot_neto_oi", df["cot_neto_oi"])
    if "residuo_kalman" in df.columns and int(df["residuo_kalman"].notna().sum()) >= 30:
        poner("brecha", df["residuo_kalman"])
    elif "residuo_vf" in df.columns:
        poner("brecha", df["residuo_vf"])
    if "post_feb2022" in df.columns:
        poner("post_feb2022", df["post_feb2022"])
    if "regimen" in df.columns:
        regimen = df["regimen"]
        for estado in (0, 1, 2):
            marca = (regimen == estado).astype(float)
            marca = marca.where(regimen.notna())
            poner(f"regimen_{estado}", marca)
    return pd.DataFrame(datos, index=df.index)


def _estimador(nombre: str):
    if nombre == "logistica":
        # La logística se estandariza con la media del propio entrenamiento.
        # Sin eso, el VIX (cerca de 20) aplasta a un retorno (cerca de 0.01)
        # y el ajuste no cierra.
        return Pipeline(
            [
                ("escala", StandardScaler()),
                (
                    "modelo",
                    LogisticRegression(max_iter=300, class_weight="balanced", random_state=0),
                ),
            ]
        )
    if nombre == "bosque":
        return RandomForestClassifier(
            n_estimators=40,
            max_depth=4,
            min_samples_leaf=25,
            class_weight="balanced_subsample",
            random_state=0,
            n_jobs=1,
        )
    return LGBMClassifier(
        n_estimators=40,
        max_depth=3,
        learning_rate=0.08,
        min_child_samples=25,
        class_weight="balanced",
        verbosity=-1,
        random_state=0,
        n_jobs=1,
        force_col_wise=True,
    )


def _clases_de(modelo) -> list[float]:
    clases = getattr(modelo, "classes_", None)
    if clases is None and hasattr(modelo, "named_steps"):
        clases = modelo.named_steps["modelo"].classes_
    return [float(valor) for valor in clases]


def _proba_de(modelo, x: np.ndarray, clase: float) -> float:
    clases = _clases_de(modelo)
    proba = modelo.predict_proba(np.asarray(x, dtype=float).reshape(1, -1))[0]
    if clase not in clases:
        return 0.0
    return float(proba[clases.index(clase)])


def _repartir(modelo, x: np.ndarray, isotonic) -> tuple[float, float, float, float]:
    """P(alza) calibrada, y plano/baja reescalados para que las tres sumen 1."""
    p_alza_cruda = _proba_de(modelo, x, 1.0)
    p_plano = _proba_de(modelo, x, 0.0)
    p_baja = _proba_de(modelo, x, -1.0)
    if isotonic is None:
        p_alza = p_alza_cruda
    else:
        p_alza = float(isotonic.predict([p_alza_cruda])[0])
    p_alza = float(np.clip(p_alza, 0.0, 1.0))
    resto = p_plano + p_baja
    if resto <= 1e-8:
        p_plano = p_baja = (1.0 - p_alza) / 2.0
    else:
        p_plano = (1.0 - p_alza) * p_plano / resto
        p_baja = (1.0 - p_alza) * p_baja / resto
    clase = float(modelo.predict(np.asarray(x, dtype=float).reshape(1, -1))[0])
    return p_alza, p_plano, p_baja, clase


def _ajustar_bloque(
    x: np.ndarray,
    clase: np.ndarray,
    idx: np.ndarray,
    horizonte: int,
    embargo: int,
) -> tuple[dict, str | None, int | None]:
    """Ajusta los tres modelos. Elige el de menor Brier en el tramo de calibración.

    `idx` son posiciones de la serie original. El ajuste termina antes de que
    su etiqueta se cruce con la primera fila de calibración.
    """
    n = len(idx)
    cal_n = max(30, int(round(n * 0.2)))
    cal_n = min(cal_n, n // 3)
    if cal_n < 20:
        return {}, None, None
    calib = idx[-cal_n:]
    limite = limite_pasado(int(calib[0]), horizonte, embargo)
    ent = idx[idx <= limite]
    if len(ent) < 40:
        ent = idx[:-cal_n]
    if len(ent) < 40 or len(np.unique(clase[ent])) < 2:
        return {}, None, None
    vivos = {}
    brier_cal = {}
    for nombre in MODELOS:
        try:
            est = _estimador(nombre)
            est.fit(x[ent], clase[ent])
        except Exception:
            continue
        if len(np.unique(clase[calib])) < 2:
            vivos[nombre] = (est, None)
            continue
        crudas = np.array([_proba_de(est, x[i], 1.0) for i in calib])
        y_up = (clase[calib] == 1).astype(float)
        try:
            brier_cal[nombre] = float(brier_score_loss(y_up, np.clip(crudas, 0, 1)))
        except ValueError:
            brier_cal[nombre] = None
        iso = None
        if len(np.unique(y_up)) >= 2:
            try:
                iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
                iso.fit(crudas, y_up)
            except Exception:
                iso = None
        vivos[nombre] = (est, iso)
    if not vivos:
        return {}, None, None
    candidatos = {nombre: valor for nombre, valor in brier_cal.items() if valor is not None}
    if candidatos:
        elegido = min(candidatos, key=candidatos.get)
    else:
        elegido = next(iter(vivos))
    return vivos, elegido, int(ent.max())


def clasificar_walk_forward(
    x: np.ndarray,
    y: np.ndarray,
    vol: np.ndarray,
    horizonte: int,
    min_entrenamiento: int = 756,
    paso: int = 21,
    embargo: int | None = None,
) -> dict:
    """Caminata expansiva. La fila t no entra a su propio entrenamiento."""
    if embargo is None:
        embargo = horizonte
    n = len(y)
    clase, banda = etiquetar_direccion(y, vol, horizonte)
    fila_x = np.isfinite(x).all(axis=1)
    fila_y = np.isfinite(clase)
    prob_up = np.full(n, np.nan)
    prob_plano = np.full(n, np.nan)
    prob_baja = np.full(n, np.nan)
    pred = np.full(n, np.nan)
    por_modelo = {
        nombre: {
            "prob_up": np.full(n, np.nan),
            "pred": np.full(n, np.nan),
        }
        for nombre in MODELOS
    }
    elegido_en = np.array([""] * n, dtype=object)
    vivos: dict = {}
    elegido = None
    siguiente = 0
    indice_max = None
    t_ajuste = None
    for t in range(n):
        if not fila_x[t]:
            continue
        if not vivos or t >= siguiente:
            validas = fila_x & fila_y
            idx = indices_purgados(n, t, horizonte, validas, embargo=embargo)
            if len(idx) >= min_entrenamiento:
                vivos, elegido, tope = _ajustar_bloque(x, clase, idx, horizonte, embargo)
                if vivos:
                    indice_max = tope
                    t_ajuste = int(t)
                    siguiente = t + paso
            if not vivos:
                continue
        if elegido is None or elegido not in vivos:
            continue
        for nombre, (est, iso) in vivos.items():
            try:
                p_alza, p_plano, p_baja, clase_pred = _repartir(est, x[t], iso)
            except Exception:
                continue
            por_modelo[nombre]["prob_up"][t] = p_alza
            por_modelo[nombre]["pred"][t] = clase_pred
            if nombre == elegido:
                prob_up[t] = p_alza
                prob_plano[t] = p_plano
                prob_baja[t] = p_baja
                pred[t] = clase_pred
                elegido_en[t] = nombre
    return {
        "clase": clase,
        "banda": banda,
        "prob_up": prob_up,
        "prob_plano": prob_plano,
        "prob_baja": prob_baja,
        "pred": pred,
        "por_modelo": por_modelo,
        "elegido_en": elegido_en,
        "indice_max_entreno": indice_max,
        "t_ultimo_ajuste": t_ajuste,
        "embargo": int(embargo),
        "horizonte": int(horizonte),
    }


def _brier_base(y_up: np.ndarray) -> float:
    tasa = float(np.mean(y_up))
    return float(brier_score_loss(y_up, np.full(len(y_up), tasa)))


def calibracion_por_bins(y_up: np.ndarray, prob: np.ndarray, n_bins: int = 8) -> list[dict]:
    """Bins de frecuencia parecida: probabilidad media contra frecuencia realizada."""
    if len(y_up) < n_bins:
        return []
    orden = np.argsort(prob)
    cortes = np.array_split(orden, n_bins)
    filas = []
    for grupo in cortes:
        if len(grupo) == 0:
            continue
        filas.append(
            {
                "n": int(len(grupo)),
                "p_media": float(np.mean(prob[grupo])),
                "y_media": float(np.mean(y_up[grupo])),
            }
        )
    return filas


def metricas_clasificacion(corrida: dict, horizonte: int) -> dict:
    """Balanced accuracy, ROC-AUC y Brier, sin solapar, contra siempre-alza."""
    clase = np.asarray(corrida["clase"], dtype=float)
    pred = np.asarray(corrida["pred"], dtype=float)
    prob = np.asarray(corrida["prob_up"], dtype=float)
    comun = np.isfinite(clase) & np.isfinite(pred) & np.isfinite(prob)
    indices = _tomar_no_solapadas(comun, horizonte)

    def bloque(yy: np.ndarray, pp: np.ndarray, pr: np.ndarray) -> dict:
        if len(yy) < 15 or len(np.unique(yy)) < 2:
            return {
                "n": int(len(yy)),
                "balanced_accuracy": None,
                "balanced_accuracy_siempre_alza": None,
                "roc_auc": None,
                "brier": None,
                "brier_siempre_alza": None,
                "brier_tasa_base": None,
            }
        y_up = (yy == 1).astype(int)
        siempre = np.ones(len(yy))
        try:
            auc = float(roc_auc_score(y_up, pr))
        except ValueError:
            auc = None
        try:
            brier = float(brier_score_loss(y_up, np.clip(pr, 0, 1)))
            brier_siempre = float(brier_score_loss(y_up, np.ones(len(yy))))
            brier_tasa = _brier_base(y_up)
        except ValueError:
            brier = brier_siempre = brier_tasa = None
        return {
            "n": int(len(yy)),
            "balanced_accuracy": float(balanced_accuracy_score(yy, pp)),
            "balanced_accuracy_siempre_alza": float(balanced_accuracy_score(yy, siempre)),
            "roc_auc": auc,
            "brier": brier,
            "brier_siempre_alza": brier_siempre,
            "brier_tasa_base": brier_tasa,
            "calibracion": calibracion_por_bins(y_up.astype(float), np.clip(pr, 0, 1)),
        }

    salida = {"elegido": bloque(clase[indices], pred[indices], prob[indices])}
    for nombre, serie in corrida["por_modelo"].items():
        pred_m = np.asarray(serie["pred"], dtype=float)
        prob_m = np.asarray(serie["prob_up"], dtype=float)
        comun_m = np.isfinite(clase) & np.isfinite(pred_m) & np.isfinite(prob_m)
        idx = _tomar_no_solapadas(comun_m, horizonte)
        salida[nombre] = bloque(clase[idx], pred_m[idx], prob_m[idx])
    return salida


def backtest_compra_en_partes(
    precios: np.ndarray,
    prob_up: np.ndarray,
    paso: int = 20,
    umbral_alto: float = 0.60,
    umbral_bajo: float = 0.35,
) -> dict:
    """DCA plano contra adelantar la compra cuando P(alza) es alta.

    Cada fecha de decisión tiene un dólar programado. Si P(alza) pasa el
    umbral alto, se adelanta también el dólar de la fecha siguiente. Si queda
    por debajo del umbral bajo, se aplaza la mitad a la fecha siguiente.
    Al final se gasta el mismo total. Gana quien compra más onzas.
    """
    precios = np.asarray(precios, dtype=float)
    prob = np.asarray(prob_up, dtype=float)
    validos = [i for i in range(len(precios)) if np.isfinite(prob[i]) and precios[i] > 0]
    if len(validos) < paso * 2:
        return {"disponible": False, "nota": "No hay suficientes fechas con probabilidad."}
    decisiones = validos[::paso]
    n = len(decisiones)
    programado = np.ones(n, dtype=float)
    gastado_partes = 0.0
    onzas_partes = 0.0
    onzas_dca = 0.0
    adelantos = 0
    aplazos = 0
    for j, i in enumerate(decisiones):
        monto = float(programado[j])
        if monto <= 0:
            onzas_dca += 1.0 / precios[i]
            continue
        if j + 1 < n and prob[i] >= umbral_alto and programado[j + 1] > 0:
            tomar = float(programado[j + 1])
            programado[j + 1] = 0.0
            monto += tomar
            adelantos += 1
        elif j + 1 < n and prob[i] <= umbral_bajo and monto > 0:
            aplazar = 0.5 * monto
            monto -= aplazar
            programado[j + 1] += aplazar
            aplazos += 1
        if monto > 0:
            onzas_partes += monto / precios[i]
            gastado_partes += monto
        onzas_dca += 1.0 / precios[i]
    gastado_dca = float(n)
    if gastado_partes <= 0 or onzas_dca <= 0 or onzas_partes <= 0:
        return {"disponible": False, "nota": "La regla no llegó a comprar."}
    precio_dca = gastado_dca / onzas_dca
    precio_partes = gastado_partes / onzas_partes
    return {
        "disponible": True,
        "n": int(n),
        "dolares": gastado_dca,
        "dolares_partes": gastado_partes,
        "onzas_dca": onzas_dca,
        "onzas_partes": onzas_partes,
        "precio_promedio_dca": precio_dca,
        "precio_promedio_partes": precio_partes,
        "ventaja_onzas_pct": (onzas_partes / onzas_dca - 1.0) * 100.0,
        "adelantos": int(adelantos),
        "aplazos": int(aplazos),
        "umbral_alto": umbral_alto,
        "umbral_bajo": umbral_bajo,
        "nota": (
            "Misma caja en los dos caminos. Ventaja positiva: la regla compró más onzas "
            "(un precio promedio más bajo). No es una recomendación."
        ),
    }
