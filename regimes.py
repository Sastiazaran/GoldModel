"""Regímenes con asignación filtrada: lo de hoy no usa el día de mañana.

K-means, una mezcla gaussiana y un HMM se estiman con una ventana que termina
antes del día t. La etiqueta de t sale del filtro de un paso (en el HMM) o del
centro ya estimado (k-means y mezcla). No se usa el suavizado, que recorrería
la serie hacia atrás con datos futuros.

Las columnas entran estandarizadas con la media y la desviación de los últimos
252 días, también cerrados en t. Una media de toda la muestra adelantaría el
nivel de 2024 a una fila de 2016.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from modelos import retorno_log_pasado

N_REGIMENES = 3
VENTANA_Z = 252
VENTANA_AJUSTE = 756


def matriz_regimen(df: pd.DataFrame, ventana: int = VENTANA_Z) -> pd.DataFrame:
    """Retorno, volatilidad, tasa real, dólar y petróleo, en puntajes móviles."""
    oro = df["oro_usd"].astype(float)
    ret1 = retorno_log_pasado(oro, 1)
    columnas = {
        "ret_20": retorno_log_pasado(oro, 20),
        "vol_20": ret1.rolling(20, min_periods=20).std(ddof=0),
        "real": df["rendimiento_real_10"].astype(float),
        "dolar": retorno_log_pasado(df["dolar_index"].astype(float), 20),
        "petroleo": retorno_log_pasado(df["brent"].astype(float), 20),
    }
    marco = pd.DataFrame(columnas, index=df.index)
    salida = pd.DataFrame(index=df.index)
    for nombre, serie in marco.items():
        media = serie.rolling(ventana, min_periods=ventana).mean()
        desv = serie.rolling(ventana, min_periods=ventana).std(ddof=0)
        salida[nombre] = (serie - media) / desv.where(desv > 1e-8)
    return salida


def _distancia_medias(viejas: np.ndarray, nuevas: np.ndarray) -> np.ndarray:
    """Índice original que corresponde a cada estado ya alineado (0..k-1)."""
    costo = np.linalg.norm(viejas[:, None, :] - nuevas[None, :, :], axis=2)
    filas, cols = linear_sum_assignment(costo)
    orden = np.empty(len(filas), dtype=int)
    orden[filas] = cols
    return orden


def _permutar_hmm(modelo, orden: np.ndarray) -> None:
    modelo.startprob_ = modelo.startprob_[orden]
    modelo.transmat_ = modelo.transmat_[np.ix_(orden, orden)]
    modelo.means_ = modelo.means_[orden]
    cov = np.asarray(modelo.covars_)[orden]
    # En 'diag' el getter a veces devuelve la matriz llena. El setter pide
    # (n_estados, n_columnas): nos quedamos con la diagonal, que es la que se estimó.
    if modelo.covariance_type == "diag" and cov.ndim == 3:
        cov = np.diagonal(cov, axis1=1, axis2=2)
    modelo.covars_ = cov


def _paso_hmm(modelo, log_alpha: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Un paso del filtro hacia adelante. log_alpha es el estado anterior."""
    log_b = np.asarray(modelo._compute_log_likelihood(x.reshape(1, -1))[0], dtype=float)
    log_trans = np.log(np.clip(modelo.transmat_, 1e-12, 1.0))
    predicho = logsumexp(log_alpha[:, None] + log_trans, axis=0)
    nuevo = predicho + log_b
    return nuevo - logsumexp(nuevo)


def _nombres_desde_medias(medias: np.ndarray) -> list[str]:
    """Pone un nombre distinto a cada estado según su media, en el orden de columnas."""
    k = medias.shape[0]
    nombres = [""] * k
    usados: set[int] = set()

    def tomar(estado: int, texto: str) -> None:
        if estado in usados or nombres[estado]:
            return
        nombres[estado] = texto
        usados.add(estado)

    tomar(int(np.argmax(medias[:, 1])), "Volatilidad alta")
    tomar(int(np.argmax(medias[:, 2])), "Tasas reales altas")
    for estado in np.argsort(-medias[:, 0]):
        if int(estado) not in usados:
            tomar(int(estado), "Oro al alza")
            break
    for estado in range(k):
        if not nombres[estado]:
            nombres[estado] = "Oro débil"
    return nombres


def asignar_regimenes(
    x: np.ndarray,
    min_obs: int = 504,
    paso: int = 63,
    ventana: int = VENTANA_AJUSTE,
    k: int = N_REGIMENES,
    semilla: int = 0,
) -> dict:
    """Etiquetas de HMM, k-means y mezcla. La de HMM es la que sigue el modelo.

    En cada reentreno el HMM nuevo se reordena para que el estado 0 siga siendo
    el mismo de antes, comparando las medias. Sin eso, el color de la gráfica
    saltaría de nombre en cada ajuste.
    """
    from hmmlearn.hmm import GaussianHMM

    n, p = x.shape
    finito = np.isfinite(x).all(axis=1)
    estado = np.full(n, np.nan)
    prob = np.full((n, k), np.nan)
    kmeans = np.full(n, np.nan)
    gmm = np.full(n, np.nan)
    modelo = None
    centros_k = None
    mezcla = None
    medias_previas = None
    log_alpha = None
    siguiente = min_obs
    nombres = [f"Régimen {i}" for i in range(k)]
    for t in range(n):
        if not finito[t]:
            log_alpha = None
            continue
        if modelo is None or t >= siguiente:
            inicio = max(0, t - ventana)
            bloque = x[inicio:t]
            filas = np.isfinite(bloque).all(axis=1)
            if int(filas.sum()) >= min_obs:
                muestra = bloque[filas]
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        nuevo = GaussianHMM(
                            n_components=k,
                            covariance_type="diag",
                            n_iter=20,
                            random_state=semilla,
                            tol=1e-3,
                        )
                        nuevo.fit(muestra)
                    if medias_previas is not None:
                        orden = _distancia_medias(medias_previas, nuevo.means_)
                        _permutar_hmm(nuevo, orden)
                    medias_previas = np.array(nuevo.means_, copy=True)
                    modelo = nuevo
                    nombres = _nombres_desde_medias(modelo.means_)
                    posterior = modelo.predict_proba(muestra)[-1]
                    log_alpha = np.log(np.clip(posterior, 1e-12, 1.0))
                except Exception:
                    # Un reentreno fallido no borra el filtro que ya venía sirviendo.
                    if modelo is None:
                        log_alpha = None
                try:
                    centros_k = KMeans(n_clusters=k, n_init=5, random_state=semilla).fit(muestra)
                except Exception:
                    centros_k = None
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mezcla = GaussianMixture(
                            n_components=k, covariance_type="diag", random_state=semilla, n_init=1
                        ).fit(muestra)
                except Exception:
                    mezcla = None
                siguiente = t + paso
            elif modelo is None:
                continue
        if modelo is not None and log_alpha is not None:
            log_alpha = _paso_hmm(modelo, log_alpha, x[t])
            peso = np.exp(log_alpha)
            peso = peso / peso.sum()
            prob[t] = peso
            estado[t] = int(np.argmax(peso))
        if centros_k is not None:
            kmeans[t] = int(centros_k.predict(x[t].reshape(1, -1))[0])
        if mezcla is not None:
            gmm[t] = int(mezcla.predict(x[t].reshape(1, -1))[0])
    return {
        "estado": estado,
        "prob": prob,
        "kmeans": kmeans,
        "gmm": gmm,
        "nombres": nombres,
        "columnas": ["ret_20", "vol_20", "real", "dolar", "petroleo"][:p],
    }


def buscar_analogos(
    z: pd.DataFrame,
    precios: pd.Series,
    ventana: int = 60,
    k: int = 5,
    separacion: int = 60,
    paso: int = 5,
) -> dict:
    """Ventanas pasadas parecidas a la más reciente, y qué hizo el oro después.

    El análogo termina antes de la ventana reciente, y su retorno posterior
    también cabe antes de esa ventana. No se usa el futuro de hoy.
    """
    x = z.to_numpy(dtype=float)
    precio = precios.to_numpy(dtype=float)
    n = len(x)
    if n < ventana + separacion + ventana:
        return {"disponible": False, "periodos": [], "nota": "Historia corta para buscar análogos."}
    reciente = x[-ventana:]
    if not np.isfinite(reciente).all():
        return {"disponible": False, "periodos": [], "nota": "La ventana reciente tiene huecos."}
    tope = n - ventana - separacion
    candidatos = []
    for fin in range(ventana, tope, paso):
        trozo = x[fin - ventana : fin]
        if not np.isfinite(trozo).all():
            continue
        dist = float(np.sqrt(np.mean((trozo - reciente) ** 2)))
        candidatos.append((dist, fin))
    candidatos.sort(key=lambda par: par[0])
    elegidos: list[tuple[float, int]] = []
    for dist, fin in candidatos:
        if any(abs(fin - otro) < ventana for _, otro in elegidos):
            continue
        elegidos.append((dist, fin))
        if len(elegidos) >= k:
            break
    periodos = []
    fechas = z.index
    for dist, fin in elegidos:
        ultimo = fin - 1
        def retorno(horizonte: int) -> float | None:
            destino = ultimo + horizonte
            if destino >= n - ventana or destino >= len(precio):
                return None
            if not (np.isfinite(precio[ultimo]) and np.isfinite(precio[destino]) and precio[ultimo] > 0):
                return None
            return float(np.log(precio[destino] / precio[ultimo]))

        periodos.append(
            {
                "desde": fechas[fin - ventana],
                "hasta": fechas[ultimo],
                "distancia": dist,
                "retorno_20": retorno(20),
                "retorno_60": retorno(60),
            }
        )
    return {
        "disponible": len(periodos) > 0,
        "ventana": ventana,
        "reciente_desde": fechas[-ventana],
        "reciente_hasta": fechas[-1],
        "periodos": periodos,
        "nota": (
            "La distancia es la raíz del error cuadrático medio entre puntajes. "
            "El retorno posterior es historia de ese análogo, no un pronóstico."
        ),
    }
