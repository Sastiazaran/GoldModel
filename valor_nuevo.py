"""Valor justo con breakeven, quiebre de 2022, betas que se mueven y Markov.

El modelo viejo (tasas reales, dólar, petróleo, VIX, inflación) se queda.
Este archivo arma otra recta, solo para comparar. El día t se valora con
betas estimados antes de ver el precio de t.

La dummie de quiebre vale 1 desde el 24 de febrero de 2022, el día en que
empezó la invasión de Ucrania. Es una fecha de calendario, no un dato que
haya que esperar. El breakeven de 10 años (FRED T10YIE, o nominal menos real
si FRED no contestó) es un precio de mercado y se usa el mismo día.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

FECHA_QUIEBRE = pd.Timestamp("2022-02-24")
Q_CANDIDATOS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)


def dummy_post_feb2022(indice) -> pd.Series:
    """1 a partir del 24 de febrero de 2022, inclusive. 0 antes."""
    fechas = pd.DatetimeIndex(pd.to_datetime(indice)).tz_localize(None).normalize()
    valores = (fechas >= FECHA_QUIEBRE).astype(float)
    return pd.Series(valores, index=indice, name="post_feb2022")


def matriz_valor_nuevo(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Diseño del valor justo nuevo. Una columna ausente no entra.

    Lleva la recta base más, si existen, el breakeven, el GPR, las compras
    de bancos centrales y la dummie de febrero de 2022. El breakeven y la
    tasa real no son la misma serie (la real no es el nominal menos este
    breakeven en todos los días), pero se parecen: el VIF se reporta aparte.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        log_oro = np.log(df["oro_usd"].to_numpy(dtype=float))
        log_dolar = np.log(df["dolar_index"].to_numpy(dtype=float))
        log_brent = np.log(df["brent"].to_numpy(dtype=float))
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
    usadas = ["rendimiento_real_10", "log_dolar", "log_brent", "vix", "cpi_anual"]
    if "breakeven_10" in df.columns:
        breakeven = df["breakeven_10"].to_numpy(dtype=float)
        if int(np.isfinite(breakeven).sum()) >= 250 and float(np.nanstd(breakeven)) > 0:
            x["breakeven_10"] = breakeven
            usadas.append("breakeven_10")
    if "gpr" in df.columns:
        gpr = df["gpr"].to_numpy(dtype=float)
        if int(np.isfinite(gpr).sum()) >= 250 and float(np.nanstd(gpr)) > 0:
            x["gpr"] = gpr
            usadas.append("gpr")
    if "compras_bc_ton" in df.columns:
        compras = df["compras_bc_ton"].to_numpy(dtype=float)
        if int(np.isfinite(compras).sum()) >= 250 and float(np.nanstd(compras)) > 0:
            x["compras_bc_ton"] = compras
            usadas.append("compras_bc_ton")
    x["post_feb2022"] = dummy_post_feb2022(df.index).to_numpy(dtype=float)
    usadas.append("post_feb2022")
    y = pd.Series(log_oro, index=df.index, name="log_oro")
    return x, y, usadas


def _loglik_predictiva(x: np.ndarray, y: np.ndarray, varianza: float, q_rel: float, desde: int) -> float:
    """Verosimilitud de un paso, solo dentro de la ventana inicial."""
    n, k = x.shape
    beta = np.linalg.lstsq(x[:desde], y[:desde], rcond=None)[0]
    try:
        p = varianza * np.linalg.inv(x[:desde].T @ x[:desde])
    except np.linalg.LinAlgError:
        p = np.eye(k) * varianza
    q = (q_rel * varianza) * np.eye(k)
    total = 0.0
    cuenta = 0
    for t in range(desde, n):
        p = p + q
        xt = x[t]
        pred = float(xt @ beta)
        err = float(y[t] - pred)
        s = float(xt @ p @ xt) + varianza
        if s <= 1e-12:
            continue
        total += -0.5 * (math_log(s) + err * err / s)
        k_gan = (p @ xt) / s
        beta = beta + k_gan * err
        p = (np.eye(k) - np.outer(k_gan, xt)) @ p
        p = 0.5 * (p + p.T)
        cuenta += 1
    if cuenta == 0:
        return -np.inf
    return total


def math_log(valor: float) -> float:
    return float(np.log(valor))


def elegir_q(x: np.ndarray, y: np.ndarray, varianza: float) -> float:
    """Elige el ruido de los betas solo con la ventana inicial, en una cuadrícula."""
    n = len(y)
    desde = n // 2
    if desde < 40 or n - desde < 30:
        return 1e-4
    mejor = 1e-4
    mejor_ll = -np.inf
    for q_rel in Q_CANDIDATOS:
        ll = _loglik_predictiva(x, y, varianza, q_rel, desde)
        if ll > mejor_ll:
            mejor_ll = ll
            mejor = float(q_rel)
    return mejor


def filtrar_kalman(
    x: np.ndarray,
    y: np.ndarray,
    min_obs: int = 756,
) -> dict:
    """Filtro de Kalman con betas en caminata aleatoria.

    El valor de t es x_t por el beta predicho antes de ver y_t. La varianza
    del error y el tamaño del ruido de los betas se estiman en las primeras
    `min_obs` filas completas y no se vuelven a mirar.
    """
    n, k = x.shape
    completo = np.isfinite(x).all(axis=1) & np.isfinite(y)
    yhat = np.full(n, np.nan)
    idx = np.flatnonzero(completo)
    if len(idx) < min_obs + 5:
        return {"yhat": yhat, "q_rel": None, "varianza": None, "n_inicial": int(len(idx))}
    i0 = idx[:min_obs]
    beta = np.linalg.lstsq(x[i0], y[i0], rcond=None)[0]
    residuo = y[i0] - x[i0] @ beta
    varianza = float(np.var(residuo))
    if not np.isfinite(varianza) or varianza < 1e-10:
        varianza = 1e-10
    q_rel = elegir_q(x[i0], y[i0], varianza)
    try:
        p = varianza * np.linalg.inv(x[i0].T @ x[i0])
    except np.linalg.LinAlgError:
        p = np.eye(k) * varianza
    q = (q_rel * varianza) * np.eye(k)
    ojo = np.eye(k)
    for t in idx[min_obs:]:
        p = p + q
        xt = x[t]
        pred = float(xt @ beta)
        yhat[t] = pred
        err = float(y[t] - pred)
        s = float(xt @ p @ xt) + varianza
        if s <= 1e-12:
            continue
        k_gan = (p @ xt) / s
        beta = beta + k_gan * err
        p = (ojo - np.outer(k_gan, xt)) @ p
        p = 0.5 * (p + p.T)
    return {"yhat": yhat, "q_rel": float(q_rel), "varianza": varianza, "n_inicial": int(min_obs)}


def _probabilidad_siguiente(resultado) -> np.ndarray | None:
    """P(régimen de la observación siguiente) a partir del filtro, no del suavizado.

    El filtro del último día de la muestra usa esa muestra y nada posterior.
    Un paso de la matriz de transición lo lleva al día que todavía no se vio.
    """
    filtrada = np.asarray(resultado.filtered_marginal_probabilities, dtype=float)
    if filtrada.ndim != 2 or not np.isfinite(filtrada[-1]).all():
        return None
    trans = np.asarray(resultado.regime_transition, dtype=float)
    if trans.ndim == 3:
        trans = trans[:, :, -1]
    if trans.shape[0] != trans.shape[1]:
        return filtrada[-1]
    fila = filtrada[-1]
    if np.nanmax(np.abs(trans.sum(axis=1) - 1.0)) < 0.05:
        pred = fila @ trans
    elif np.nanmax(np.abs(trans.sum(axis=0) - 1.0)) < 0.05:
        pred = trans @ fila
    else:
        pred = fila
    pred = np.clip(pred, 0.0, None)
    suma = float(pred.sum())
    if suma <= 0 or not np.isfinite(suma):
        return None
    return pred / suma


def _ajustar_markov(y: np.ndarray, x: np.ndarray, nombres: list[str]) -> dict | None:
    """Un ajuste de MarkovRegression. None si no sale un número usable."""
    from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression

    # Una columna constante (la dummie de 2022, antes del quiebre o mucho
    # después) rompe el ajuste. Se queda fuera de esta ventana.
    variar = x.std(axis=0) > 1e-8
    if int(variar.sum()) < 1:
        return None
    x = x[:, variar]
    nombres = [nombre for nombre, queda in zip(nombres, variar) if queda]
    media = x.mean(axis=0)
    escala = x.std(axis=0)
    escala = np.where(escala < 1e-8, 1.0, escala)
    x_std = (x - media) / escala
    exog = pd.DataFrame(x_std, columns=[f"x{i + 1}" for i in range(x.shape[1])])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            modelo = MarkovRegression(
                y,
                k_regimes=2,
                trend="c",
                exog=exog,
                switching_trend=True,
                switching_exog=True,
                # La varianza que cambia de régimen deja la cadena sin una
                # distribución estacionaria en estos datos (SVD). Sin ese
                # cambio el ajuste sí cierra.
                switching_variance=False,
            )
            resultado = modelo.fit(maxiter=30, disp=False, em_iter=10)
    except Exception:
        return None
    nombres_param = list(resultado.model.param_names)
    theta = np.asarray(resultado.params, dtype=float)
    if len(nombres_param) != len(theta) or not np.isfinite(theta).all():
        return None
    params = dict(zip(nombres_param, theta))
    # statsmodels nombra las columnas exógenas x1, x2, ... en el orden en que
    # entraron, y la constante del régimen como const[r]. No usa el nombre de la columna.
    betas = []
    for regimen in (0, 1):
        constante = params.get(f"const[{regimen}]")
        if constante is None:
            return None
        coef = []
        for i in range(x.shape[1]):
            clave = f"x{i + 1}[{regimen}]"
            if clave not in params:
                return None
            coef.append(params[clave])
        betas.append((float(constante), np.asarray(coef, dtype=float)))
    prob = _probabilidad_siguiente(resultado)
    if prob is None or len(prob) != 2:
        return None
    convergio = bool(getattr(resultado, "mle_retvals", {}).get("converged", False))
    return {
        "betas": betas,
        "prob_siguiente": prob,
        "media": media,
        "escala": escala,
        "columnas": variar,
        "convergio": convergio,
    }


def camino_markov(
    x: np.ndarray,
    y: np.ndarray,
    nombres: list[str],
    min_obs: int = 756,
    paso: int = 126,
) -> dict:
    """Valor justo de Markov reestimado solo con el pasado.

    Entre reentrenos se dejan quietos los betas y la probabilidad de régimen
    predicha en el último ajuste. Ninguno de los dos usa el precio del día t.
    """
    n = len(y)
    yhat = np.full(n, np.nan)
    probs = np.full((n, 2), np.nan)
    completo = np.isfinite(x).all(axis=1) & np.isfinite(y)
    ajuste = None
    siguiente = 0
    n_ajustes = 0
    convergio_ultimo = None
    for t in range(n):
        if not completo[t]:
            continue
        if ajuste is None or t >= siguiente:
            idx = np.flatnonzero(completo[:t])
            if len(idx) >= min_obs:
                nuevo = _ajustar_markov(y[idx], x[idx], nombres)
                siguiente = t + paso
                if nuevo is not None:
                    ajuste = nuevo
                    n_ajustes += 1
                    convergio_ultimo = nuevo["convergio"]
            if ajuste is None:
                continue
        xz = (x[t, ajuste["columnas"]] - ajuste["media"]) / ajuste["escala"]
        medias = [constante + float(xz @ coef) for constante, coef in ajuste["betas"]]
        yhat[t] = float(np.dot(ajuste["prob_siguiente"], medias))
        probs[t] = ajuste["prob_siguiente"]
    return {
        "yhat": yhat,
        "probs": probs,
        "n_ajustes": n_ajustes,
        "convergio": convergio_ultimo,
    }
