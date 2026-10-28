"""Junta la evaluación, el valor justo nuevo, los regímenes, la clase y los intervalos.

El modelo viejo no se toca. Este módulo corre al lado y solo reemplaza el
pronóstico titular si, en las mismas fechas y sin solapar el horizonte, el
error cuadrático medio es menor. Si no, el titular sigue siendo el Ridge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from classify import (
    armar_features,
    backtest_compra_en_partes,
    clasificar_walk_forward,
    metricas_clasificacion,
    volatilidad_conocida,
)
from evaluacion import cobertura_intervalo, diebold_mariano
from intervalos import caminar_garch, caminar_intervalos
from modelos import (
    MIN_ENTRENAMIENTO,
    PASO_REENTRENO,
    _rmse,
    _tomar_no_solapadas,
    _vif,
    metricas_pronostico,
    retorno_log_pasado,
    valor_justo_expandiendo,
)
from regimes import asignar_regimenes, buscar_analogos, matriz_regimen
from valor_nuevo import camino_markov, dummy_post_feb2022, filtrar_kalman, matriz_valor_nuevo


def _ultimo(serie: pd.Series):
    validos = serie.replace([np.inf, -np.inf], np.nan).dropna()
    if validos.empty:
        return None, None
    return validos.index[-1], float(validos.iloc[-1])


def _rmse_log(y: np.ndarray, pred: np.ndarray, indices: np.ndarray) -> float | None:
    if len(indices) == 0:
        return None
    return _rmse(y[indices], pred[indices])


def _fechas_comunes(y, *series, horizonte: int) -> np.ndarray:
    comun = np.isfinite(np.asarray(y, dtype=float))
    for serie in series:
        comun = comun & np.isfinite(np.asarray(serie, dtype=float))
    return _tomar_no_solapadas(comun, horizonte)


def correr_estudio(
    df: pd.DataFrame,
    paso: int | None = None,
    min_obs: int | None = None,
    hacer_markov: bool = True,
    hacer_garch: bool = True,
    paso_markov: int = 126,
    paso_regimen: int = 63,
) -> dict:
    """Corre las piezas nuevas y devuelve números listos para el informe y el tablero.

    Escribe columnas en `df`. Si una pieza falla, queda anotada en `avisos`
    y no se inventa su métrica.
    """
    paso = PASO_REENTRENO if paso is None else int(paso)
    min_obs = MIN_ENTRENAMIENTO if min_obs is None else int(min_obs)
    avisos: list[str] = []
    df["post_feb2022"] = dummy_post_feb2022(df.index)

    x_df, y_nivel, usadas = matriz_valor_nuevo(df)
    x = x_df.to_numpy(dtype=float)
    yv = y_nivel.to_numpy(dtype=float)
    kalman = filtrar_kalman(x, yv, min_obs=min_obs)
    df["log_valor_kalman"] = kalman["yhat"]
    df["valor_kalman"] = np.exp(kalman["yhat"])
    df["residuo_kalman"] = yv - kalman["yhat"]
    # La recta nueva, con los mismos mínimos cuadrados expansivos del modelo
    # base. Esta es la que puede reemplazar al titular. El Kalman no: su error
    # de un paso persigue el precio y no es una brecha estructural.
    ols_nuevo = valor_justo_expandiendo(x_df, y_nivel, min_obs=min_obs)
    df["log_valor_ols_nuevo"] = ols_nuevo["log_valor_justo_vf"]
    df["valor_ols_nuevo"] = np.exp(ols_nuevo["log_valor_justo_vf"])

    markov = None
    preferidas = [
        nombre
        for nombre in (
            "rendimiento_real_10",
            "log_dolar",
            "breakeven_10",
            "gpr",
            "compras_bc_ton",
            "post_feb2022",
        )
        if nombre in x_df.columns
    ]
    if hacer_markov and len(preferidas) >= 2:
        try:
            markov = camino_markov(
                x_df[preferidas].to_numpy(dtype=float),
                yv,
                preferidas,
                min_obs=min_obs,
                paso=paso_markov,
            )
            df["log_valor_markov"] = markov["yhat"]
            df["valor_markov"] = np.exp(markov["yhat"])
        except Exception as error:  # noqa: BLE001
            avisos.append(f"El Markov del valor justo no corrió ({error}).")
            markov = None
    elif hacer_markov:
        avisos.append("Markov no entró: no hubo suficientes columnas nuevas.")

    regimen = None
    analogos = {"disponible": False, "periodos": []}
    try:
        z = matriz_regimen(df)
        regimen = asignar_regimenes(
            z.to_numpy(dtype=float),
            min_obs=max(180, min_obs // 2),
            paso=paso_regimen,
        )
        df["regimen"] = regimen["estado"]
        for i in range(regimen["prob"].shape[1]):
            df[f"prob_regimen_{i}"] = regimen["prob"][:, i]
        analogos = buscar_analogos(z, df["oro_usd"].astype(float))
    except Exception as error:  # noqa: BLE001
        avisos.append(f"Los regímenes no corrieron ({error}).")

    features = armar_features(df)
    matriz = features.to_numpy(dtype=float)
    vol = volatilidad_conocida(df["oro_usd"]).to_numpy(dtype=float)

    clasificacion = {}
    bandas = {}
    garch = {}
    for horizonte in (5, 20):
        objetivo = df[f"y_{horizonte}"].to_numpy(dtype=float)
        try:
            corrida = clasificar_walk_forward(
                matriz,
                objetivo,
                vol,
                horizonte,
                min_entrenamiento=min_obs,
                paso=paso,
            )
            df[f"prob_up_{horizonte}"] = corrida["prob_up"]
            df[f"prob_plano_{horizonte}"] = corrida["prob_plano"]
            df[f"prob_baja_{horizonte}"] = corrida["prob_baja"]
            df[f"clase_pred_{horizonte}"] = corrida["pred"]
            clasificacion[horizonte] = {
                "corrida": corrida,
                "metricas": metricas_clasificacion(corrida, horizonte),
                "columnas": list(features.columns),
            }
        except Exception as error:  # noqa: BLE001
            avisos.append(f"La clasificación a {horizonte} días no corrió ({error}).")
        try:
            banda = caminar_intervalos(
                matriz,
                objetivo,
                horizonte,
                min_entrenamiento=min_obs,
                paso=paso,
            )
            df[f"pred_mediana_{horizonte}"] = banda["mediana"]
            df[f"banda_baja_{horizonte}"] = banda["bajo"]
            df[f"banda_alta_{horizonte}"] = banda["alto"]
            df[f"media_mediana_{horizonte}"] = banda["media"]
            bandas[horizonte] = banda
        except Exception as error:  # noqa: BLE001
            avisos.append(f"Los intervalos a {horizonte} días no corrieron ({error}).")
        if hacer_garch:
            try:
                ret1 = retorno_log_pasado(df["oro_usd"].astype(float), 1).to_numpy(dtype=float)
                g = caminar_garch(ret1, horizonte, paso=max(paso, 42), min_obs=min(252, max(80, min_obs // 3)))
                if g is not None:
                    g["cobertura"] = cobertura_intervalo(objetivo, g["bajo"], g["alto"], horizonte)
                    df[f"garch_bajo_{horizonte}"] = g["bajo"]
                    df[f"garch_alto_{horizonte}"] = g["alto"]
                garch[horizonte] = g
            except Exception as error:  # noqa: BLE001
                avisos.append(f"GARCH a {horizonte} días no corrió ({error}).")
                garch[horizonte] = None

    comparacion = _comparar_pronosticos(df, bandas)
    valor = _resumen_valor(df, x_df, y_nivel, usadas, kalman, markov)
    reg_resumen = _resumen_regimen(df, regimen)
    dca = {"disponible": False}
    if "prob_up_20" in df.columns:
        dca = backtest_compra_en_partes(
            df["oro_usd"].to_numpy(dtype=float),
            df["prob_up_20"].to_numpy(dtype=float),
            paso=20,
        )
    return {
        "df": df,
        "avisos": avisos,
        "variables_valor": usadas,
        "columnas_features": list(features.columns),
        "valor": valor,
        "comparacion": comparacion,
        "clasificacion": {h: _empaquetar_clase(df, bloque) for h, bloque in clasificacion.items()},
            "intervalos": {
                h: _empaquetar_intervalo(df, h, bloque, garch.get(h)) for h, bloque in bandas.items()
            },
        "regimen": reg_resumen,
        "analogos": analogos,
        "dca": dca,
        "esquema": (
            "Caminata expansiva. El entrenamiento de un día t solo usa filas cuya "
            "etiqueta ya cerró y deja, además, un embargo de otro horizonte. "
            "Los días que se solapan no se cuentan como pruebas distintas en la nota principal."
        ),
    }


def _comparar_pronosticos(df: pd.DataFrame, bandas: dict) -> dict:
    """RMSE en las mismas fechas. El nuevo pasa a titular solo si el RMSE baja."""
    filas = []
    diebold = {}
    promovido = {}
    for horizonte in (5, 20):
        y = df[f"y_{horizonte}"].to_numpy(dtype=float)
        viejo = df[f"pred_{horizonte}"].to_numpy(dtype=float) if f"pred_{horizonte}" in df.columns else np.full(len(df), np.nan)
        media_vieja = df[f"media_{horizonte}"].to_numpy(dtype=float) if f"media_{horizonte}" in df.columns else np.full(len(df), np.nan)
        nuevo = df[f"pred_mediana_{horizonte}"].to_numpy(dtype=float) if f"pred_mediana_{horizonte}" in df.columns else np.full(len(df), np.nan)
        media_nueva = df[f"media_mediana_{horizonte}"].to_numpy(dtype=float) if f"media_mediana_{horizonte}" in df.columns else np.full(len(df), np.nan)
        indices = _fechas_comunes(y, viejo, nuevo, media_vieja, media_nueva, horizonte=horizonte)
        rmse_viejo = _rmse_log(y, viejo, indices)
        rmse_nuevo = _rmse_log(y, nuevo, indices)
        gana = (
            rmse_viejo is not None
            and rmse_nuevo is not None
            and rmse_nuevo < rmse_viejo
        )
        promovido[horizonte] = bool(gana)
        met_viejo = metricas_pronostico(y, viejo, media_vieja, horizonte)["sin_solapar"]
        met_nuevo = metricas_pronostico(y, nuevo, media_nueva, horizonte)["sin_solapar"]
        ceros = np.zeros(len(y))
        diebold[str(horizonte)] = {
            "ridge_vs_cero": diebold_mariano(y, viejo, ceros, horizonte),
            "nuevo_vs_cero": diebold_mariano(y, nuevo, ceros, horizonte),
            "nuevo_vs_ridge": diebold_mariano(y, nuevo, viejo, horizonte),
        }
        for modelo, metrica, rmse_comun in (
            ("ridge", met_viejo, rmse_viejo),
            ("mediana_cuantil", met_nuevo, rmse_nuevo),
        ):
            filas.append(
                {
                    "horizonte": horizonte,
                    "modelo": modelo,
                    "n": int(len(indices)),
                    "n_metrica": metrica.get("n"),
                    "rmse": metrica.get("rmse"),
                    "rmse_mismas_fechas": rmse_comun,
                    "rmse_cero": metrica.get("rmse_cero"),
                    "r2_cero": metrica.get("r2_cero"),
                    "r2_media": metrica.get("r2_media"),
                    "acierto": metrica.get("acierto"),
                    "acierto_siempre_alza": metrica.get("acierto_siempre_alza"),
                    "es_titular": bool(gana) if modelo == "mediana_cuantil" else not bool(gana),
                }
            )
        if horizonte in bandas:
            filas[-1]["cobertura"] = bandas[horizonte]["cobertura"]["sin_solapar"]
    return {
        "criterio": (
            "El pronóstico nuevo (mediana del cuantil, calibrada con MAPIE) reemplaza al Ridge "
            "en un horizonte solo si su RMSE es menor en las mismas fechas, una cada horizonte."
        ),
        "filas": filas,
        "diebold_mariano": diebold,
        "promovido": {str(h): promovido.get(h, False) for h in (5, 20)},
    }


def _resumen_valor(df, x_df, y_nivel, usadas, kalman, markov) -> dict:
    log_precio = y_nivel.to_numpy(dtype=float)
    kalman_hat = df["log_valor_kalman"].to_numpy(dtype=float)
    ols_hat = (
        df["log_valor_ols_nuevo"].to_numpy(dtype=float)
        if "log_valor_ols_nuevo" in df.columns
        else np.full(len(df), np.nan)
    )
    base_hat = (
        np.log(df["valor_justo_vf"].to_numpy(dtype=float))
        if "valor_justo_vf" in df.columns
        else np.full(len(df), np.nan)
    )
    # Una observación cada 20 sesiones: el error de nivel de días seguidos es casi el mismo.
    # El Kalman no entra a esta cuenta: un paso del filtro persigue el precio.
    idx_k = _tomar_no_solapadas(
        np.isfinite(log_precio) & np.isfinite(ols_hat) & np.isfinite(base_hat),
        20,
    )
    rmse_ols = _rmse_log(log_precio, ols_hat, idx_k)
    rmse_base = _rmse_log(log_precio, base_hat, idx_k)
    idx_kalman = _tomar_no_solapadas(
        np.isfinite(log_precio) & np.isfinite(kalman_hat) & np.isfinite(base_hat),
        20,
    )
    rmse_kalman = _rmse_log(log_precio, kalman_hat, idx_kalman)
    rmse_markov = None
    rmse_base_markov = None
    if markov is not None and "log_valor_markov" in df.columns:
        markov_hat = df["log_valor_markov"].to_numpy(dtype=float)
        idx_m = _tomar_no_solapadas(
            np.isfinite(log_precio) & np.isfinite(markov_hat) & np.isfinite(base_hat),
            20,
        )
        rmse_markov = _rmse_log(log_precio, markov_hat, idx_m)
        rmse_base_markov = _rmse_log(log_precio, base_hat, idx_m)
    mejora = {}
    if rmse_ols is not None and rmse_base is not None:
        mejora["ols_nuevo"] = rmse_base - rmse_ols
    if rmse_markov is not None and rmse_base_markov is not None and len(idx_m) >= 30:
        mejora["markov"] = rmse_base_markov - rmse_markov
    positivas = {nombre: delta for nombre, delta in mejora.items() if delta > 0}
    if positivas:
        candidato_titular = max(positivas, key=positivas.get)
        promovido = True
    else:
        candidato_titular = "base"
        promovido = False
    indices = idx_k
    fecha_ols, valor_ols = _ultimo(df["valor_ols_nuevo"]) if "valor_ols_nuevo" in df.columns else (None, None)
    brecha_ols = None
    if fecha_ols is not None and valor_ols and valor_ols > 0:
        brecha_ols = (float(df.loc[fecha_ols, "oro_usd"]) / valor_ols - 1.0) * 100.0
    fecha_k, valor_k = _ultimo(df["valor_kalman"]) if "valor_kalman" in df.columns else (None, None)
    precio = float(df.loc[fecha_k, "oro_usd"]) if fecha_k is not None else float("nan")
    brecha_k = (precio / valor_k - 1.0) * 100.0 if valor_k and valor_k > 0 else float("nan")
    fecha_m, valor_m = _ultimo(df["valor_markov"]) if "valor_markov" in df.columns else (None, None)
    brecha_m = None
    if fecha_m is not None and valor_m and valor_m > 0:
        brecha_m = (float(df.loc[fecha_m, "oro_usd"]) / valor_m - 1.0) * 100.0
    completo = np.isfinite(x_df.to_numpy(dtype=float)).all(axis=1) & np.isfinite(log_precio)
    vif = _vif(x_df.loc[completo]) if int(completo.sum()) > x_df.shape[1] + 5 else {}
    if candidato_titular == "markov":
        serie = df["valor_markov"]
    elif candidato_titular == "ols_nuevo":
        serie = df["valor_ols_nuevo"]
    else:
        serie = df["valor_justo_vf"] if "valor_justo_vf" in df.columns else df["valor_ols_nuevo"]
    df["valor_titular"] = serie
    return {
        "variables": usadas,
        "q_rel": kalman.get("q_rel"),
        "rmse_log_base": rmse_base,
        "rmse_log_ols": rmse_ols,
        "rmse_log_kalman": rmse_kalman,
        "rmse_log_markov": rmse_markov,
        "n_rmse": int(len(indices)),
        "promovido": bool(promovido),
        "titular": candidato_titular,
        "fecha": fecha_ols or fecha_k,
        "valor_ols": valor_ols,
        "brecha_ols_pct": brecha_ols,
        "valor_kalman": valor_k,
        "brecha_kalman_pct": brecha_k,
        "valor_markov": valor_m,
        "brecha_markov_pct": brecha_m,
        "markov_ajustes": None if markov is None else markov.get("n_ajustes"),
        "markov_convergio": None if markov is None else markov.get("convergio"),
        "vif": vif,
        "nota": (
            "La dummie vale 1 desde el 24 de febrero de 2022. "
            "El RMSE que decide el titular es el de la recta expansiva (mínimos cuadrados, "
            "solo con el pasado), una observación cada 20 sesiones. "
            "El Kalman se reporta aparte: su error de un paso persigue el precio y no compite."
        ),
    }


def _resumen_regimen(df: pd.DataFrame, regimen: dict | None) -> dict:
    if regimen is None or "regimen" not in df.columns:
        return {"disponible": False}
    fecha, estado = _ultimo(df["regimen"])
    if fecha is None:
        return {"disponible": False}
    estado = int(estado)
    nombres = list(regimen.get("nombres") or [])
    probs = []
    for i in range(len(nombres)):
        columna = f"prob_regimen_{i}"
        if columna in df.columns and np.isfinite(df.loc[fecha, columna]):
            probs.append(float(df.loc[fecha, columna]))
        else:
            probs.append(None)
    ari_k = None
    ari_g = None
    estado_arr = df["regimen"].to_numpy(dtype=float)
    if "kmeans" in regimen:
        ambos = np.isfinite(estado_arr) & np.isfinite(regimen["kmeans"])
        if int(ambos.sum()) > 30:
            ari_k = float(adjusted_rand_score(estado_arr[ambos], regimen["kmeans"][ambos]))
    if "gmm" in regimen:
        ambos = np.isfinite(estado_arr) & np.isfinite(regimen["gmm"])
        if int(ambos.sum()) > 30:
            ari_g = float(adjusted_rand_score(estado_arr[ambos], regimen["gmm"][ambos]))
    return {
        "disponible": True,
        "fecha": fecha,
        "id": estado,
        "nombre": nombres[estado] if estado < len(nombres) else f"Régimen {estado}",
        "nombres": nombres,
        "probabilidades": probs,
        "metodo": "hmm_filtrado",
        "ari_kmeans": ari_k,
        "ari_gmm": ari_g,
        "nota": (
            "El estado de hoy sale del filtro del HMM, con parámetros estimados antes de hoy. "
            "No es el suavizado, que usaría el futuro. K-means y la mezcla gaussiana se estiman "
            "igual, solo con el pasado; el ARI dice cuánto coinciden las etiquetas, no cuál acierta."
        ),
    }


def _empaquetar_clase(df: pd.DataFrame, bloque: dict) -> dict:
    corrida = bloque["corrida"]
    horizonte = int(corrida["horizonte"])
    fecha, p_up = _ultimo(df[f"prob_up_{horizonte}"]) if f"prob_up_{horizonte}" in df.columns else (None, None)
    p_plano = p_baja = None
    if fecha is not None:
        p_plano = float(df.loc[fecha, f"prob_plano_{horizonte}"]) if np.isfinite(df.loc[fecha, f"prob_plano_{horizonte}"]) else None
        p_baja = float(df.loc[fecha, f"prob_baja_{horizonte}"]) if np.isfinite(df.loc[fecha, f"prob_baja_{horizonte}"]) else None
    elegido = None
    if fecha is not None and f"prob_up_{horizonte}" in df.columns:
        # El nombre del modelo elegido ese día está en la corrida.
        pos = df.index.get_loc(fecha)
        if isinstance(pos, slice):
            pos = pos.start
        nombre = corrida["elegido_en"][int(pos)]
        elegido = nombre or None
    return {
        "fecha": fecha,
        "prob_up": p_up,
        "prob_plano": p_plano,
        "prob_baja": p_baja,
        "modelo": elegido,
        "metricas": bloque["metricas"],
        "indice_max_entreno": corrida.get("indice_max_entreno"),
        "t_ultimo_ajuste": corrida.get("t_ultimo_ajuste"),
        "embargo": corrida.get("embargo"),
        "columnas": bloque.get("columnas"),
    }


def _empaquetar_intervalo(df: pd.DataFrame, horizonte: int, bloque: dict, garch) -> dict:
    fecha, mediana = _ultimo(df[f"pred_mediana_{horizonte}"])
    bajo = alto = None
    if fecha is not None:
        bajo = float(df.loc[fecha, f"banda_baja_{horizonte}"])
        alto = float(df.loc[fecha, f"banda_alta_{horizonte}"])
    precio = float(df.loc[fecha, "oro_usd"]) if fecha is not None else float("nan")
    salida = {
        "fecha": fecha,
        "nivel": bloque.get("nivel"),
        "metodo": bloque.get("metodo"),
        "retorno_mediana": mediana,
        "retorno_bajo": bajo,
        "retorno_alto": alto,
        "nivel_bajo_usd": None if bajo is None or not np.isfinite(precio) else float(precio * np.exp(bajo)),
        "nivel_alto_usd": None if alto is None or not np.isfinite(precio) else float(precio * np.exp(alto)),
        "cobertura": bloque.get("cobertura"),
        "indice_max_entreno": bloque.get("indice_max_entreno"),
        "t_ultimo_ajuste": bloque.get("t_ultimo_ajuste"),
    }
    if garch is not None:
        salida["garch"] = {
            "metodo": garch.get("metodo"),
            "cobertura": garch.get("cobertura"),
            "nivel": garch.get("nivel"),
        }
    return salida
