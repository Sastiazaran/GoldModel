"""Intervalos de predicción: cuantiles de LightGBM, conformal con MAPIE y, si se puede, GARCH.

El cuantil se entrena solo con el pasado purgado. MAPIE recalibra el ancho
con un tramo posterior a ese entrenamiento, también ya realizado. La
cobertura que se reporta es la de fuera de muestra, no la del ajuste.
"""

from __future__ import annotations

import warnings

import numpy as np
from lightgbm import LGBMRegressor
from mapie.regression import ConformalizedQuantileRegressor
from scipy import stats

from evaluacion import cobertura_intervalo, indices_purgados

NIVEL = 0.80


def _cuantil(alpha: float) -> LGBMRegressor:
    return LGBMRegressor(
        objective="quantile",
        alpha=float(alpha),
        n_estimators=40,
        max_depth=3,
        learning_rate=0.08,
        min_child_samples=25,
        verbosity=-1,
        random_state=0,
        n_jobs=1,
        force_col_wise=True,
    )


def _leer_intervalo(punto, intervalo) -> tuple[float, float, float]:
    punto = np.asarray(punto, dtype=float).reshape(-1)
    banda = np.asarray(intervalo, dtype=float)
    if banda.ndim == 3:
        bajo = float(banda[0, 0, 0])
        alto = float(banda[0, 1, 0])
    else:
        bajo = float(banda[0, 0])
        alto = float(banda[0, 1])
    if bajo > alto:
        bajo, alto = alto, bajo
    return float(punto[0]), bajo, alto


def caminar_intervalos(
    x: np.ndarray,
    y: np.ndarray,
    horizonte: int,
    min_entrenamiento: int = 756,
    paso: int = 21,
    embargo: int | None = None,
    nivel: float = NIVEL,
) -> dict:
    """Mediana y banda conformal. La mediana es el pronóstico de punto nuevo."""
    if embargo is None:
        embargo = horizonte
    n = len(y)
    mediana = np.full(n, np.nan)
    bajo = np.full(n, np.nan)
    alto = np.full(n, np.nan)
    media = np.full(n, np.nan)
    fila_x = np.isfinite(x).all(axis=1)
    fila_y = np.isfinite(y)
    modelo = None
    siguiente = 0
    indice_max = None
    t_ajuste = None
    alpha_bajo = (1.0 - nivel) / 2.0
    alpha_alto = 1.0 - alpha_bajo
    for t in range(n):
        if not fila_x[t]:
            continue
        if modelo is None or t >= siguiente:
            idx = indices_purgados(n, t, horizonte, fila_x & fila_y, embargo=embargo)
            if len(idx) >= min_entrenamiento:
                cal_n = max(40, int(round(len(idx) * 0.2)))
                cal_n = min(cal_n, len(idx) // 3)
                if cal_n >= 30:
                    calib = idx[-cal_n:]
                    limite = int(calib[0]) - horizonte - embargo - 1
                    ent = idx[idx <= limite] if limite >= min_entrenamiento // 2 else idx[:-cal_n]
                    if len(ent) >= min_entrenamiento // 2:
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                lo = _cuantil(alpha_bajo)
                                hi = _cuantil(alpha_alto)
                                mid = _cuantil(0.5)
                                lo.fit(x[ent], y[ent])
                                hi.fit(x[ent], y[ent])
                                mid.fit(x[ent], y[ent])
                                cqr = ConformalizedQuantileRegressor(
                                    estimator=[lo, hi, mid],
                                    confidence_level=nivel,
                                    prefit=True,
                                )
                                cqr.conformalize(x[calib], y[calib])
                            modelo = cqr
                            media_y = float(y[ent].mean())
                            indice_max = int(ent.max())
                            t_ajuste = int(t)
                            siguiente = t + paso
                        except Exception:
                            modelo = modelo
            if modelo is None:
                continue
        try:
            punto, intervalo = modelo.predict_interval(x[t].reshape(1, -1))
            med, lo_t, hi_t = _leer_intervalo(punto, intervalo)
        except Exception:
            continue
        mediana[t] = med
        bajo[t] = lo_t
        alto[t] = hi_t
        media[t] = media_y
    return {
        "mediana": mediana,
        "bajo": bajo,
        "alto": alto,
        "media": media,
        "nivel": nivel,
        "cobertura": cobertura_intervalo(y, bajo, alto, horizonte),
        "indice_max_entreno": indice_max,
        "t_ultimo_ajuste": t_ajuste,
        "metodo": "lightgbm_cuantil_mapie",
    }


def caminar_garch(
    retornos: np.ndarray,
    horizonte: int,
    nivel: float = NIVEL,
    min_obs: int = 252,
    paso: int = 42,
    ventana: int = 504,
) -> dict | None:
    """Banda de un GARCH(1,1) para la suma de los próximos `horizonte` días.

    Se reestima cada `paso` sesiones con los retornos ya observados, como
    máximo `ventana` días. Si `arch` no está o un ajuste falla, se conserva
    el anterior. None si no hubo ni un ajuste.
    """
    try:
        from arch import arch_model
    except Exception:
        return None
    retornos = np.asarray(retornos, dtype=float)
    n = len(retornos)
    bajo = np.full(n, np.nan)
    alto = np.full(n, np.nan)
    media = np.full(n, np.nan)
    z = float(stats.norm.ppf(0.5 + nivel / 2.0))
    vigente = None
    siguiente = min_obs
    for t in range(n):
        if not np.isfinite(retornos[t]):
            continue
        if vigente is None or t >= siguiente:
            inicio = max(0, t - ventana + 1)
            muestra = retornos[inicio : t + 1]
            muestra = muestra[np.isfinite(muestra)]
            if len(muestra) >= min_obs:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        modelo = arch_model(
                            muestra * 100.0,
                            mean="Constant",
                            vol="Garch",
                            p=1,
                            q=1,
                            dist="normal",
                            rescale=False,
                        )
                        ajuste = modelo.fit(disp="off", show_warning=False)
                        pronostico = ajuste.forecast(horizon=horizonte, reindex=False)
                    varianzas = np.asarray(pronostico.variance, dtype=float).reshape(-1)[:horizonte]
                    medias = np.asarray(pronostico.mean, dtype=float).reshape(-1)[:horizonte]
                    sigma = float(np.sqrt(np.sum(varianzas)) / 100.0)
                    mu = float(np.sum(medias) / 100.0)
                    if np.isfinite(sigma) and sigma > 0 and np.isfinite(mu):
                        vigente = (mu, sigma)
                        siguiente = t + paso
                except Exception:
                    if vigente is not None:
                        siguiente = t + paso
            if vigente is None:
                continue
        mu, sigma = vigente
        media[t] = mu
        bajo[t] = mu - z * sigma
        alto[t] = mu + z * sigma
    if not np.isfinite(bajo).any():
        return None
    return {
        "bajo": bajo,
        "alto": alto,
        "media": media,
        "nivel": nivel,
        "metodo": "garch11",
    }
