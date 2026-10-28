"""Evaluación sin adelanto para horizontes que se solapan.

Una caminata expansiva reentrena solo con el pasado. La purga quita del
entrenamiento cualquier fila cuyo retorno de 5 o 20 días comparta fechas con
el bloque de prueba. El embargo deja, además, otro horizonte de por medio.
Diebold-Mariano compara el error del modelo con el de "el precio no cambia"
sin tratar los días solapados como si fueran independientes.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy import stats

from modelos import _tomar_no_solapadas


def limite_pasado(t: int, horizonte: int, embargo: int) -> int:
    """Último índice de entrenamiento permitido para predecir el día t.

    La etiqueta del día i usa el precio hasta i+horizonte. La del día t usa
    el precio desde t. Para que no se toquen, y para dejar `embargo` sesiones
    extra, hace falta i + horizonte + embargo < t.
    """
    return int(t) - int(horizonte) - int(embargo) - 1


def indices_purgados(
    n: int,
    t: int,
    horizonte: int,
    fila_valida: np.ndarray,
    embargo: int | None = None,
) -> np.ndarray:
    """Filas de entrenamiento para predecir t. Solo el pasado, con purga y embargo.

    `embargo` por defecto es el propio horizonte: además de no solapar la
    etiqueta, se deja otra ventana igual de ancha. Con embargo 0 la regla
    coincide con conocer el retorno (i + horizonte < t), un día más estricta
    que la caminata vieja, que admite la fila cuyo retorno cierra exactamente en t.
    """
    if embargo is None:
        embargo = horizonte
    limite = limite_pasado(t, horizonte, embargo)
    if limite < 0 or n <= 0:
        return np.array([], dtype=int)
    candidatos = np.arange(0, limite + 1)
    validos = np.asarray(fila_valida, dtype=bool)[candidatos]
    return candidatos[validos]


def bloques_cv_purgada(
    n: int,
    horizonte: int,
    n_bloques: int = 5,
    embargo: int | None = None,
    min_entrenamiento: int = 50,
) -> list[dict]:
    """Bloques de una validación expansiva, purgada y con embargo.

    Cada bloque de prueba es un tramo posterior al anterior. El entrenamiento
    es solo lo anterior a ese tramo, menos la purga y el embargo. No se entrena
    con días posteriores al bloque: en un pronóstico eso sería mirar el futuro.
    El pliegue de libro que sí usa el futuro, ya purgado, está en
    `mascara_entrenamiento_ambos_lados` y no se usa para la nota.
    """
    if embargo is None:
        embargo = horizonte
    if n_bloques < 1 or n <= min_entrenamiento + horizonte:
        return []
    inicio = int(min_entrenamiento)
    ancho = n - inicio
    n_bloques = min(n_bloques, max(1, ancho // max(horizonte, 1)))
    bordes = np.linspace(inicio, n, n_bloques + 1, dtype=int)
    bloques = []
    for k in range(n_bloques):
        a = int(bordes[k])
        b = int(bordes[k + 1])
        if b <= a:
            continue
        limite = limite_pasado(a, horizonte, embargo)
        if limite + 1 < max(20, min_entrenamiento // 5):
            continue
        bloques.append(
            {
                "train": np.arange(0, limite + 1),
                "test": np.arange(a, b),
                "test_inicio": a,
                "test_fin": b - 1,
            }
        )
    return bloques


def mascara_entrenamiento_ambos_lados(
    n: int,
    test_inicio: int,
    test_fin: int,
    horizonte: int,
    embargo: int | None = None,
) -> np.ndarray:
    """Máscara de un pliegue purgado clásico, con datos a los dos lados.

    Sirve para comprobar la purga. La nota del pronóstico no la usa: entrenar
    con lo que vino después del bloque de prueba adelanta información.
    """
    if embargo is None:
        embargo = horizonte
    ok = np.ones(n, dtype=bool)
    izq = limite_pasado(test_inicio, horizonte, embargo)
    # A la derecha, la primera fila usable es la que ya dejó el embargo
    # después de que cerró la última etiqueta del bloque.
    der = int(test_fin) + int(horizonte) + int(embargo) + 1
    ok[: izq + 1] = True
    if izq + 1 < n:
        ok[izq + 1 :] = False
    if der < n:
        ok[der:] = True
    ok[max(0, test_inicio) : min(n, test_fin + 1)] = False
    return ok


def _newey_west_varianza(d: np.ndarray, rezagos: int) -> float:
    """Varianza de largo plazo de la media de d, con pesos de Bartlett."""
    n = len(d)
    d_c = d - d.mean()
    gamma0 = float(np.dot(d_c, d_c) / n)
    var = gamma0
    rezagos = int(max(0, min(rezagos, n - 2)))
    for j in range(1, rezagos + 1):
        gamma = float(np.dot(d_c[j:], d_c[:-j]) / n)
        peso = 1.0 - j / (rezagos + 1.0)
        var += 2.0 * peso * gamma
    return float(var)


def diebold_mariano(
    y: np.ndarray,
    pred_modelo: np.ndarray,
    pred_base: np.ndarray,
    horizonte: int,
    perdida: str = "se",
) -> dict:
    """Diebold-Mariano del modelo contra un baseline, con corrección de Harvey-Ley-Newbold.

    La diferencial es pérdida(modelo) − pérdida(base). Negativa: el modelo
    se equivoca menos. Con días solapados la varianza es Newey-West con
    `horizonte - 1` rezagos. El p-valor sale de una t de Student con n−1
    grados de libertad, que es la corrección de muestra chica de HLN.
    """
    y = np.asarray(y, dtype=float)
    modelo = np.asarray(pred_modelo, dtype=float)
    base = np.asarray(pred_base, dtype=float)
    comun = np.isfinite(y) & np.isfinite(modelo) & np.isfinite(base)
    vacio = {
        "n": 0,
        "perdida": perdida,
        "media_diferencial": None,
        "estadistico": None,
        "estadistico_sin_correccion": None,
        "p": None,
        "modelo_mejor": None,
        "rezagos": int(max(horizonte - 1, 0)),
    }
    if int(comun.sum()) < 10:
        vacio["n"] = int(comun.sum())
        return vacio

    def armar(indices: np.ndarray, rezagos: int) -> dict:
        if len(indices) < 10:
            return {"n": int(len(indices)), "media_diferencial": None, "estadistico": None, "p": None, "modelo_mejor": None}
        yy = y[indices]
        e_m = yy - modelo[indices]
        e_b = yy - base[indices]
        if perdida == "ae":
            d = np.abs(e_m) - np.abs(e_b)
        else:
            d = e_m**2 - e_b**2
        n = len(d)
        media = float(d.mean())
        var = _newey_west_varianza(d, rezagos)
        if not math.isfinite(var) or var <= 1e-18:
            stat = 0.0 if abs(media) < 1e-15 else math.copysign(math.inf, media)
            stat_hln = stat
            p = None if math.isinf(stat) else 1.0
        else:
            stat = media / math.sqrt(var / n)
            h = max(int(horizonte), 1)
            factor = (n + 1 - 2 * h + h * (h - 1) / n) / n
            stat_hln = stat * math.sqrt(factor) if factor > 0 else stat
            p = float(2 * stats.t.sf(abs(stat_hln), df=n - 1))
        return {
            "n": int(n),
            "media_diferencial": media,
            "estadistico": None if not math.isfinite(stat_hln) else float(stat_hln),
            "estadistico_sin_correccion": None if not math.isfinite(stat) else float(stat),
            "p": p,
            "modelo_mejor": bool(media < 0),
            "rezagos": int(rezagos),
            "perdida": perdida,
        }

    solapado = armar(np.flatnonzero(comun), max(int(horizonte) - 1, 0))
    limpio = armar(_tomar_no_solapadas(comun, max(int(horizonte), 1)), 0)
    return {"solapado": solapado, "sin_solapar": limpio, "perdida": perdida}


def cobertura_intervalo(
    y: np.ndarray,
    bajo: np.ndarray,
    alto: np.ndarray,
    horizonte: int,
) -> dict:
    """Fracción de retornos realizados que cayeron dentro del intervalo."""
    y = np.asarray(y, dtype=float)
    bajo = np.asarray(bajo, dtype=float)
    alto = np.asarray(alto, dtype=float)
    comun = np.isfinite(y) & np.isfinite(bajo) & np.isfinite(alto) & (alto >= bajo)

    def bloque(indices: np.ndarray) -> dict:
        if len(indices) == 0:
            return {"n": 0, "cobertura": None}
        dentro = (y[indices] >= bajo[indices]) & (y[indices] <= alto[indices])
        return {"n": int(len(indices)), "cobertura": float(np.mean(dentro))}

    return {
        "sin_solapar": bloque(_tomar_no_solapadas(comun, max(int(horizonte), 1))),
        "solapado": bloque(np.flatnonzero(comun)),
    }


def fecha_no_adelanta(fecha_fila, fecha_disponible) -> bool:
    """True si el dato ya estaba publicado en esa fila."""
    if pd.isna(fecha_fila) or pd.isna(fecha_disponible):
        return False
    return pd.Timestamp(fecha_disponible).normalize() <= pd.Timestamp(fecha_fila).normalize()
