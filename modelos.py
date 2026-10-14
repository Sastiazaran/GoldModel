"""Dos regresiones lineales, a propósito simples.

1. Valor justo: explica el *nivel* del oro (en logaritmo) con tasas reales,
   dólar, petróleo, VIX e inflación. La brecha es lo que el precio se aleja
   de esa relación.
2. Pronóstico: intenta el *retorno futuro* a 5 y a 20 días de mercado, usando
   solo lo que ya se sabía en el día t. Se evalúa caminando hacia adelante,
   contra dos ideas tontas: "el retorno es cero" y "el retorno es el promedio
   del pasado".

No es asesoría financiera. A corto plazo el oro se parece mucho a una
caminata aleatoria: el mejor apuesta simple es que el cambio esperado es chico
frente al ruido.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller

# Una onza troy son 31.1034768 gramos. El oro se cotiza por onza troy.
GRAMOS_POR_ONZA_TROY = 31.1034768

# Unos 3 años de sesiones. Menos que eso, el coeficiente baila demasiado.
MIN_OBS_VALOR = 756
MIN_ENTRENAMIENTO = 756
# Reajustar más o menos cada mes de mercado, no todos los días.
PASO_REENTRENO = 21
HORIZONTES = (5, 20)
# Penalización de Ridge. El programa elige una dentro de cada ventana,
# mirando solo el pasado.
ALPHAS = (0.1, 1.0, 10.0, 100.0, 1000.0)
REZAGOS_HAC = 21
CORTE_SUBMUESTRA = "2022-01-01"

NOMBRES_VALOR = [
    "constante",
    "rendimiento_real_10",
    "log_dolar",
    "log_brent",
    "vix",
    "cpi_anual",
]

ETIQUETAS_VALOR = {
    "rendimiento_real_10": "Real 10 años",
    "log_dolar": "Dólar (log)",
    "log_brent": "Petróleo (log)",
    "vix": "VIX",
    "cpi_anual": "Inflación IPC",
}

FEATURES_PRONOSTICO = [
    "ret_5",
    "ret_20",
    "ret_60",
    "d_real_20",
    "d_dolar_20",
    "d_brent_20",
    "d_vix_5",
    "d_spx_20",
    "rendimiento_real_10",
    "vix",
    "cpi_anual",
    "residuo_vf",
]

ETIQUETAS_PRONOSTICO = {
    "ret_5": "retorno del oro, 5 días",
    "ret_20": "retorno del oro, 20 días",
    "ret_60": "retorno del oro, 60 días",
    "d_real_20": "cambio de la tasa real, 20 días",
    "d_dolar_20": "cambio del dólar, 20 días",
    "d_brent_20": "cambio del petróleo, 20 días",
    "d_vix_5": "cambio del VIX, 5 días",
    "d_spx_20": "cambio del S&P 500, 20 días",
    "rendimiento_real_10": "nivel de la tasa real",
    "vix": "nivel del VIX",
    "cpi_anual": "inflación anual conocida",
    "residuo_vf": "brecha del valor justo",
}


def onza_a_mxn_por_gramo(usd_por_onza: float, mxn_por_usd: float) -> float:
    """Pasa dólares por onza a pesos por gramo.

    Multiplica por el tipo de cambio (pesos por dólar) y divide entre los
    gramos de una onza troy. Es el valor del metal en el mercado mayorista,
    no el precio de una joyería.
    """
    return float(usd_por_onza) * float(mxn_por_usd) / GRAMOS_POR_ONZA_TROY


def retorno_log_pasado(precios: pd.Series, dias: int) -> pd.Series:
    """Log(precio de hoy) - log(precio de hace `dias` sesiones).

    `shift(dias)` positivo mira al pasado. Este número sí se puede usar
    como dato de entrada el día de hoy.
    """
    log_precio = np.log(precios.astype(float))
    return log_precio - log_precio.shift(dias)


def retorno_log_adelante(precios: pd.Series, dias: int) -> pd.Series:
    """Log(precio de dentro de `dias` sesiones) - log(precio de hoy).

    `shift(-dias)` mira al futuro. Sirve solo como la respuesta que queremos
    predecir. Nunca entra como columna explicativa.
    """
    log_precio = np.log(precios.astype(float))
    return log_precio.shift(-dias) - log_precio


def preparar_muestra(panel: pd.DataFrame) -> pd.DataFrame:
    """Agrega retornos, cambios y las respuestas futuras. No tira filas."""
    requeridas = [
        "oro_usd",
        "rendimiento_real_10",
        "dolar_index",
        "brent",
        "vix",
        "cpi_anual",
        "sp500",
        "usd_mxn",
    ]
    faltan = [columna for columna in requeridas if columna not in panel.columns]
    if faltan:
        raise RuntimeError(
            "Faltan series para armar el modelo: " + ", ".join(faltan) + ". "
            "Sin ellas la regresión no se puede interpretar."
        )
    df = panel.copy()
    oro = df["oro_usd"].astype(float)
    df["ret_5"] = retorno_log_pasado(oro, 5)
    df["ret_20"] = retorno_log_pasado(oro, 20)
    df["ret_60"] = retorno_log_pasado(oro, 60)
    df["y_5"] = retorno_log_adelante(oro, 5)
    df["y_20"] = retorno_log_adelante(oro, 20)
    df["d_real_20"] = df["rendimiento_real_10"] - df["rendimiento_real_10"].shift(20)
    df["d_dolar_20"] = retorno_log_pasado(df["dolar_index"], 20)
    df["d_brent_20"] = retorno_log_pasado(df["brent"], 20)
    df["d_vix_5"] = df["vix"] - df["vix"].shift(5)
    if "sp500" in df.columns:
        df["d_spx_20"] = retorno_log_pasado(df["sp500"], 20)
    else:
        df["d_spx_20"] = np.nan
    return df


def matriz_valor_justo(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Arma X y y del valor justo.

    y es el logaritmo del oro: un coeficiente de -0.10 quiere decir, en
    corto, cerca de -10% en el precio. La tasa real y el VIX van en su
    propia unidad (puntos). El dólar y el petróleo van en logaritmo para
    que el coeficiente sea una elasticidad.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        log_dolar = np.log(df["dolar_index"].to_numpy(dtype=float))
        log_brent = np.log(df["brent"].to_numpy(dtype=float))
        log_oro = np.log(df["oro_usd"].to_numpy(dtype=float))
    x = pd.DataFrame(
        {
            "constante": 1.0,
            "rendimiento_real_10": df["rendimiento_real_10"].to_numpy(dtype=float),
            "log_dolar": log_dolar,
            "log_brent": log_brent,
            "vix": df["vix"].to_numpy(dtype=float),
            "cpi_anual": df["cpi_anual"].to_numpy(dtype=float),
        },
        index=df.index,
    )
    y = pd.Series(log_oro, index=df.index, name="log_oro")
    return x, y


def _filas_completas(x: pd.DataFrame, y: pd.Series) -> pd.Series:
    return np.isfinite(x.to_numpy(dtype=float)).all(axis=1) & np.isfinite(y.to_numpy(dtype=float))


def valor_justo_expandiendo(
    x: pd.DataFrame,
    y: pd.Series,
    min_obs: int = MIN_OBS_VALOR,
    paso: int = PASO_REENTRENO,
) -> pd.DataFrame:
    """Valor justo de cada día, estimado solo con días anteriores.

    Así la brecha de 2018 no usa coeficientes que ya vieron 2024. El ajuste
    se refresca cada `paso` sesiones. La primera estimación espera `min_obs`
    días completos (unos tres años).
    """
    matriz = x.to_numpy(dtype=float)
    objetivo = y.to_numpy(dtype=float)
    n = len(objetivo)
    completo = np.isfinite(matriz).all(axis=1) & np.isfinite(objetivo)
    valor = np.full(n, np.nan)
    beta = None
    siguiente = -1
    for t in range(n):
        if not completo[t]:
            continue
        if beta is None or t >= siguiente:
            idx = np.flatnonzero(completo[:t])
            if len(idx) >= min_obs:
                beta = np.linalg.lstsq(matriz[idx], objetivo[idx], rcond=None)[0]
                siguiente = t + paso
            elif beta is None:
                continue
        valor[t] = float(matriz[t] @ beta)
    salida = pd.DataFrame(
        {
            "log_valor_justo_vf": valor,
            "residuo_vf": objetivo - valor,
        },
        index=x.index,
    )
    return salida


def _ajustar_ols(y: pd.Series, x: pd.DataFrame):
    return sm.OLS(y, x).fit(
        cov_type="HAC",
        cov_kwds={"maxlags": REZAGOS_HAC, "use_correction": True},
    )


def _vif(x: pd.DataFrame) -> dict[str, float]:
    matriz = x.to_numpy(dtype=float)
    salida = {}
    for i, nombre in enumerate(x.columns):
        if nombre == "constante":
            continue
        try:
            salida[nombre] = float(variance_inflation_factor(matriz, i))
        except Exception:  # noqa: BLE001
            salida[nombre] = float("nan")
    return salida


def _efecto_submuestra(y: pd.Series, x: pd.DataFrame, mascara: pd.Series) -> dict | None:
    if int(mascara.sum()) < 250:
        return None
    modelo = _ajustar_ols(y.loc[mascara], x.loc[mascara])
    return {
        "n": int(mascara.sum()),
        "r2": float(modelo.rsquared),
        "coef_real": float(modelo.params["rendimiento_real_10"]),
        "ee_real": float(modelo.bse["rendimiento_real_10"]),
        "coef_dolar": float(modelo.params["log_dolar"]),
        "ee_dolar": float(modelo.bse["log_dolar"]),
    }


def ajustar_valor_justo(df: pd.DataFrame) -> dict:
    """Regresión de niveles en toda la muestra, más la brecha sin adelanto."""
    x, y = matriz_valor_justo(df)
    completo = _filas_completas(x, y)
    if int(completo.sum()) < MIN_OBS_VALOR:
        raise RuntimeError(
            "No hay suficientes días completos para el valor justo "
            f"({int(completo.sum())}). Revisa si faltó alguna descarga."
        )
    modelo = _ajustar_ols(y.loc[completo], x.loc[completo])
    expandido = valor_justo_expandiendo(x, y)
    df = df.copy()
    df["log_valor_justo_vf"] = expandido["log_valor_justo_vf"]
    df["residuo_vf"] = expandido["residuo_vf"]
    df["valor_justo_vf"] = np.exp(df["log_valor_justo_vf"])

    # Coeficientes en la misma escala: 1 desviación estándar de cada variable.
    x_std = x.loc[completo].copy()
    for columna in x_std.columns:
        if columna == "constante":
            continue
        desviacion = float(x_std[columna].std(ddof=0))
        media = float(x_std[columna].mean())
        if desviacion > 0:
            x_std[columna] = (x_std[columna] - media) / desviacion
        else:
            x_std[columna] = 0.0
    modelo_std = sm.OLS(y.loc[completo], x_std).fit()

    anios = (x.index[completo] - x.index[completo][0]).days / 365.25
    tendencia = sm.OLS(y.loc[completo].to_numpy(), sm.add_constant(np.asarray(anios))).fit()

    residuos = np.asarray(modelo.resid, dtype=float)
    try:
        adf_stat, adf_p, *_ = adfuller(
            residuos,
            maxlag=REZAGOS_HAC,
            autolag="AIC",
            result_object=False,
        )
        adf_stat, adf_p = float(adf_stat), float(adf_p)
    except Exception:  # noqa: BLE001
        adf_stat, adf_p = None, None

    fecha = x.index[completo][-1]
    log_justo = float(np.dot(x.loc[fecha].to_numpy(dtype=float), modelo.params.to_numpy(dtype=float)))
    precio = float(df.loc[fecha, "oro_usd"])
    valor = float(math.exp(log_justo))
    log_vf = df.loc[fecha, "log_valor_justo_vf"]
    valor_vf = float(math.exp(log_vf)) if np.isfinite(log_vf) else float("nan")

    coeficientes = []
    for nombre in NOMBRES_VALOR:
        if nombre == "constante":
            continue
        coef = float(modelo.params[nombre])
        coef_std = float(modelo_std.params[nombre])
        coeficientes.append(
            {
                "nombre": nombre,
                "etiqueta": ETIQUETAS_VALOR[nombre],
                "coef": coef,
                "ee": float(modelo.bse[nombre]),
                "t": float(modelo.tvalues[nombre]),
                "p": float(modelo.pvalues[nombre]),
                "coef_std": coef_std,
                "efecto_std_pct": float((math.exp(coef_std) - 1.0) * 100.0),
            }
        )
    coeficientes.sort(key=lambda fila: abs(fila["efecto_std_pct"]), reverse=True)

    pre = completo & (x.index < CORTE_SUBMUESTRA)
    post = completo & (x.index >= CORTE_SUBMUESTRA)
    return {
        "df": df,
        "fecha": fecha,
        "n": int(completo.sum()),
        "inicio": x.index[completo][0],
        "fin": x.index[completo][-1],
        "r2": float(modelo.rsquared),
        "r2_tendencia": float(tendencia.rsquared),
        "adf_stat": adf_stat,
        "adf_p": adf_p,
        "vif": _vif(x.loc[completo]),
        "precio": precio,
        "valor_justo": valor,
        "brecha_pct": (precio / valor - 1.0) * 100.0,
        "valor_justo_vf": valor_vf,
        "brecha_vf_pct": (precio / valor_vf - 1.0) * 100.0 if np.isfinite(valor_vf) else float("nan"),
        "coeficientes": coeficientes,
        "antes_2022": _efecto_submuestra(y, x, pre),
        "desde_2022": _efecto_submuestra(y, x, post),
        "params": {nombre: float(modelo.params[nombre]) for nombre in modelo.params.index},
    }


def indices_de_entrenamiento(n: int, t: int, horizonte: int, fila_valida: np.ndarray) -> np.ndarray:
    """Filas cuyo resultado ya se conoce el día t.

    El resultado del día i usa el precio de i+horizonte. Para conocerlo en t
    hace falta i+horizonte <= t, o sea i <= t-horizonte. La fila t no entra.
    """
    if t - horizonte < 0:
        return np.array([], dtype=int)
    candidatos = np.arange(0, t - horizonte + 1)
    validos = np.asarray(fila_valida, dtype=bool)[candidatos]
    return candidatos[validos]


def _estandarizar(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    media = x.mean(axis=0)
    desviacion = x.std(axis=0, ddof=0)
    desviacion = np.where(desviacion < 1e-8, 1.0, desviacion)
    return media, desviacion


def elegir_alpha(x: np.ndarray, y: np.ndarray, alphas: tuple[float, ...] = ALPHAS) -> float:
    """Elige la penalización con el último tramo del *entrenamiento*.

    No baraja las filas. `train_test_split` las revolvería y el futuro se
    colaría en el pasado. Aquí el bloque final de la ventana es la prueba
    interna; lo anterior es el ajuste.
    """
    n = len(y)
    n_val = max(60, int(round(n * 0.20)))
    n_val = min(n_val, n // 3)
    if n_val < 30 or n - n_val < 30:
        return 10.0
    x_ent, x_val = x[:-n_val], x[-n_val:]
    y_ent, y_val = y[:-n_val], y[-n_val:]
    media, desviacion = _estandarizar(x_ent)
    x_ent_z = (x_ent - media) / desviacion
    x_val_z = (x_val - media) / desviacion
    mejor_alpha = alphas[0]
    mejor_rmse = None
    for alpha in alphas:
        modelo = Ridge(alpha=float(alpha), fit_intercept=True)
        modelo.fit(x_ent_z, y_ent)
        pred = modelo.predict(x_val_z)
        rmse = float(np.sqrt(np.mean((y_val - pred) ** 2)))
        if mejor_rmse is None or rmse < mejor_rmse:
            mejor_rmse = rmse
            mejor_alpha = float(alpha)
    return mejor_alpha


def _tomar_no_solapadas(mascara: np.ndarray, horizonte: int) -> np.ndarray:
    """Se queda con fechas separadas por al menos `horizonte` sesiones.

    Un retorno de 20 días comparte 19 días con el del día siguiente. Contarlos
    como 20 victorias independientes infla la seguridad. Esta selección deja
    una observación por ventana.
    """
    elegidos: list[int] = []
    ultimo = -10**9
    for i in np.flatnonzero(mascara):
        if i - ultimo >= horizonte:
            elegidos.append(int(i))
            ultimo = int(i)
    return np.asarray(elegidos, dtype=int)


def _r2(y: np.ndarray, pred: np.ndarray, base: np.ndarray) -> float:
    denom = float(np.sum((y - base) ** 2))
    if denom <= 0:
        return float("nan")
    return float(1.0 - np.sum((y - pred) ** 2) / denom)


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def _acierto(y: np.ndarray, pred: np.ndarray) -> float:
    mascara = np.isfinite(y) & np.isfinite(pred) & (y != 0)
    if int(mascara.sum()) == 0:
        return float("nan")
    return float(np.mean(np.sign(pred[mascara]) == np.sign(y[mascara])))


def metricas_pronostico(
    y: np.ndarray,
    pred: np.ndarray,
    media: np.ndarray,
    horizonte: int,
) -> dict:
    """R², RMSE y acierto de dirección, solapado y sin solapar.

    El R² contra cero pregunta si le ganamos a "el precio no cambia".
    El R² contra la media pregunta si le ganamos a "el oro rinde lo de siempre".
    El segundo es más exigente: un modelo que siempre dice "sube un poquito"
    puede ganarle al cero solo porque el oro, en promedio, ha subido.
    """
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    media = np.asarray(media, dtype=float)
    comun = np.isfinite(y) & np.isfinite(pred) & np.isfinite(media)
    posiciones = _tomar_no_solapadas(comun, horizonte)

    def bloque(indices: np.ndarray) -> dict:
        if len(indices) == 0:
            return {
                "n": 0,
                "r2_cero": float("nan"),
                "r2_media": float("nan"),
                "rmse": float("nan"),
                "rmse_cero": float("nan"),
                "rmse_media": float("nan"),
                "acierto": float("nan"),
                "acierto_siempre_alza": float("nan"),
                "acierto_signo_media": float("nan"),
                "pct_pred_positivas": float("nan"),
            }
        yy = y[indices]
        pp = pred[indices]
        mm = media[indices]
        con_signo = yy != 0
        siempre_alza = float(np.mean(yy[con_signo] > 0)) if con_signo.any() else float("nan")
        signo_media = (
            float(np.mean(np.sign(mm[con_signo]) == np.sign(yy[con_signo])))
            if con_signo.any()
            else float("nan")
        )
        return {
            "n": int(len(indices)),
            "r2_cero": _r2(yy, pp, np.zeros(len(yy))),
            "r2_media": _r2(yy, pp, mm),
            "rmse": _rmse(yy, pp),
            "rmse_cero": _rmse(yy, np.zeros(len(yy))),
            "rmse_media": _rmse(yy, mm),
            "acierto": _acierto(yy, pp),
            "acierto_siempre_alza": siempre_alza,
            "acierto_signo_media": signo_media,
            "pct_pred_positivas": float(np.mean(pp > 0)),
        }

    solapado = bloque(np.flatnonzero(comun))
    limpio = bloque(posiciones)
    return {"solapado": solapado, "sin_solapar": limpio}


def pronostico_walk_forward(
    x: np.ndarray,
    y: np.ndarray,
    fila_x: np.ndarray,
    fila_y: np.ndarray,
    horizonte: int,
    min_entrenamiento: int = MIN_ENTRENAMIENTO,
    paso: int = PASO_REENTRENO,
) -> dict:
    """Predice cada día con un Ridge ajustado solo con el pasado ya cerrado."""
    n = len(y)
    pred = np.full(n, np.nan)
    media_entreno = np.full(n, np.nan)
    alpha_usado = np.full(n, np.nan)
    modelo = None
    media_beta = None
    desviacion_beta = None
    media_y = None
    alpha = None
    coef = None
    siguiente = 0
    n_entreno = 0
    for t in range(n):
        if not fila_x[t]:
            continue
        if modelo is None or t >= siguiente:
            idx = indices_de_entrenamiento(n, t, horizonte, fila_x & fila_y)
            if len(idx) >= min_entrenamiento:
                x_ent = x[idx]
                y_ent = y[idx]
                alpha = elegir_alpha(x_ent, y_ent)
                media_beta, desviacion_beta = _estandarizar(x_ent)
                modelo = Ridge(alpha=float(alpha), fit_intercept=True)
                modelo.fit((x_ent - media_beta) / desviacion_beta, y_ent)
                media_y = float(y_ent.mean())
                coef = modelo.coef_.copy()
                n_entreno = int(len(idx))
                siguiente = t + paso
            elif modelo is None:
                continue
        pred[t] = float(modelo.predict(((x[t] - media_beta) / desviacion_beta).reshape(1, -1))[0])
        media_entreno[t] = media_y
        alpha_usado[t] = alpha
    return {
        "pred": pred,
        "media": media_entreno,
        "alpha": alpha_usado,
        "coef": coef,
        "alpha_ultimo": None if alpha is None else float(alpha),
        "n_entreno_ultimo": n_entreno,
    }


def ajustar_pronosticos(df: pd.DataFrame) -> dict:
    """Walk-forward para 5 y 20 días. Escribe las predicciones en el DataFrame."""
    if "residuo_vf" not in df.columns:
        raise RuntimeError("Falta la brecha del valor justo. Ajusta ese modelo primero.")
    faltan = [columna for columna in FEATURES_PRONOSTICO if columna not in df.columns]
    if faltan:
        raise RuntimeError(f"Faltan columnas para el pronóstico: {', '.join(faltan)}")

    x = df[FEATURES_PRONOSTICO].to_numpy(dtype=float)
    fila_x = np.isfinite(x).all(axis=1)
    if int(fila_x.sum()) < MIN_ENTRENAMIENTO + 60:
        raise RuntimeError("No hay historia suficiente para el pronóstico fuera de muestra.")

    resultados = {}
    for horizonte in HORIZONTES:
        columna_y = f"y_{horizonte}"
        y = df[columna_y].to_numpy(dtype=float)
        fila_y = np.isfinite(y)
        corrida = pronostico_walk_forward(x, y, fila_x, fila_y, horizonte)
        df[f"pred_{horizonte}"] = corrida["pred"]
        df[f"media_{horizonte}"] = corrida["media"]
        metricas = metricas_pronostico(y, corrida["pred"], corrida["media"], horizonte)
        con_pred = np.isfinite(corrida["pred"])
        fechas = df.index[con_pred]
        # El último día con datos de entrada, aunque su retorno futuro aún no exista.
        ultimos = np.flatnonzero(fila_x)
        t_ultimo = int(ultimos[-1])
        pred_ultimo = float(corrida["pred"][t_ultimo])
        sigma_base = metricas["sin_solapar"]
        if sigma_base["n"] < 30:
            sigma_base = metricas["solapado"]
            sigma_origen = "solapado"
        else:
            sigma_origen = "sin solapar"
        # Error típico fuera de muestra, en la misma unidad que la predicción.
        sigma = sigma_base["rmse"]
        pesos = []
        if corrida["coef"] is not None:
            for nombre, peso in zip(FEATURES_PRONOSTICO, corrida["coef"]):
                pesos.append({"nombre": nombre, "etiqueta": ETIQUETAS_PRONOSTICO[nombre], "peso": float(peso)})
            pesos.sort(key=lambda fila: abs(fila["peso"]), reverse=True)
        resultados[horizonte] = {
            "metricas": metricas,
            "pred_ultimo": pred_ultimo,
            "fecha_ultimo": df.index[t_ultimo],
            "sigma": float(sigma) if sigma is not None else float("nan"),
            "sigma_origen": sigma_origen,
            "alpha_ultimo": corrida["alpha_ultimo"],
            "n_entreno_ultimo": corrida["n_entreno_ultimo"],
            "pesos": pesos,
            "oos_inicio": fechas.min() if len(fechas) else None,
            "oos_fin_realizado": df.index[np.isfinite(corrida["pred"]) & np.isfinite(y)].max()
            if np.any(np.isfinite(corrida["pred"]) & np.isfinite(y))
            else None,
        }
    return {"df": df, "por_horizonte": resultados}
