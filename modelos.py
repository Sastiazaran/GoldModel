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
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# Una onza troy son 31.1034768 gramos. El oro se cotiza por onza troy.
GRAMOS_POR_ONZA_TROY = 31.1034768

# Unos 3 años de sesiones. Menos que eso, el coeficiente baila demasiado.
MIN_OBS_VALOR = 756
MIN_ENTRENAMIENTO = 756
# Reajustar más o menos cada mes de mercado, no todos los días.
PASO_REENTRENO = 21
# Ventanas móviles: cerca de 2 y de 3 años de sesiones.
VENTANA_2A = 504
VENTANA_3A = 756
# Bosque chico a propósito. Se reentrena cientos de veces; un bosque grande
# no cambiaría la lectura y haría la corrida inútilmente lenta.
GBR_PARAMS = {
    "n_estimators": 40,
    "max_depth": 2,
    "learning_rate": 0.08,
    "subsample": 0.8,
    "min_samples_leaf": 30,
    "random_state": 0,
}
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
    "gpr": "Riesgo geopolítico",
    "log_gld_ton": "Tenencias GLD (log)",
    "compras_bc_ton": "Compras bancos centrales",
    "log_usd_mxn": "Tipo de cambio (log)",
}

# Entran al valor justo solo si la columna existe y tiene historia.
EXTRAS_VALOR = ("gpr", "log_gld_ton", "compras_bc_ton")

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

# Columnas del pronóstico que solo se usan si llegaron en los datos.
EXTRAS_PRONOSTICO = (
    "cot_neto_oi",
    "rendimiento_nominal_2",
    "d_nominal2_20",
)

ETIQUETAS_EXTRAS = {
    "cot_neto_oi": "neto no comercial del COT, sobre el interés abierto",
    "rendimiento_nominal_2": "tasa nominal a 2 años",
    "d_nominal2_20": "cambio de la tasa a 2 años, 20 días",
    "ret_mxn_5": "retorno del oro en pesos, 5 días",
    "ret_mxn_20": "retorno del oro en pesos, 20 días",
    "ret_mxn_60": "retorno del oro en pesos, 60 días",
    "d_mxn_20": "cambio del peso, 20 días",
    "residuo_mxn": "brecha del valor justo en pesos",
}

FEATURES_PRONOSTICO_MXN = [
    "ret_mxn_5",
    "ret_mxn_20",
    "ret_mxn_60",
    "d_real_20",
    "d_mxn_20",
    "d_brent_20",
    "d_vix_5",
    "rendimiento_real_10",
    "vix",
    "cpi_anual",
    "residuo_mxn",
]


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
    if "rendimiento_nominal_2" in df.columns:
        df["d_nominal2_20"] = df["rendimiento_nominal_2"] - df["rendimiento_nominal_2"].shift(20)
    if "gld_toneladas" in df.columns:
        with np.errstate(divide="ignore", invalid="ignore"):
            toneladas = df["gld_toneladas"].astype(float)
            df["log_gld_ton"] = np.where(toneladas > 0, np.log(toneladas), np.nan)
    # Oro en pesos por gramo. Su propio retorno, no el del dólar convertido
    # al cierre: el tipo de cambio también se mueve entre hoy y el horizonte.
    if "usd_mxn" in df.columns:
        oro_mxn = df["oro_usd"].astype(float) * df["usd_mxn"].astype(float) / GRAMOS_POR_ONZA_TROY
        df["oro_mxn_g"] = oro_mxn
        df["ret_mxn_5"] = retorno_log_pasado(oro_mxn, 5)
        df["ret_mxn_20"] = retorno_log_pasado(oro_mxn, 20)
        df["ret_mxn_60"] = retorno_log_pasado(oro_mxn, 60)
        df["y_mxn_5"] = retorno_log_adelante(oro_mxn, 5)
        df["y_mxn_20"] = retorno_log_adelante(oro_mxn, 20)
        df["d_mxn_20"] = retorno_log_pasado(df["usd_mxn"].astype(float), 20)
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


def columnas_pronostico(
    df: pd.DataFrame,
    base: list[str] | None = None,
    extras: tuple[str, ...] = EXTRAS_PRONOSTICO,
) -> list[str]:
    """Lista de entradas del pronóstico. Las extras solo si la columna existe."""
    if base is None:
        base = FEATURES_PRONOSTICO
    columnas = [columna for columna in base if columna in df.columns]
    for columna in extras:
        if columna in df.columns and int(df[columna].notna().sum()) >= 30:
            columnas.append(columna)
    return columnas


def matriz_valor_extendida(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """La misma regresión de niveles, más GPR, GLD y compras oficiales si están.

    Una columna ausente no rompe el modelo base: simplemente no entra.
    Devuelve también los nombres extra que sí se usaron.
    """
    x, y = matriz_valor_justo(df)
    usadas: list[str] = []
    if "gpr" in df.columns:
        gpr = df["gpr"].to_numpy(dtype=float)
        if int(np.isfinite(gpr).sum()) >= 250 and float(np.nanstd(gpr)) > 0:
            x["gpr"] = gpr
            usadas.append("gpr")
    if "log_gld_ton" in df.columns:
        log_gld = df["log_gld_ton"].to_numpy(dtype=float)
        if int(np.isfinite(log_gld).sum()) >= 250 and float(np.nanstd(log_gld)) > 0:
            x["log_gld_ton"] = log_gld
            usadas.append("log_gld_ton")
    if "compras_bc_ton" in df.columns:
        compras = df["compras_bc_ton"].to_numpy(dtype=float)
        if int(np.isfinite(compras).sum()) >= 250 and float(np.nanstd(compras)) > 0:
            x["compras_bc_ton"] = compras
            usadas.append("compras_bc_ton")
    return x, y, usadas


def matriz_valor_mxn(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Valor justo del oro en pesos por gramo.

    Lleva el tipo de cambio (pesos por dólar) en lugar del índice dólar,
    para no contar el dólar dos veces. GPR, GLD y compras oficiales entran
    si están, igual que en el modelo en dólares.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        log_mxn = np.log(df["usd_mxn"].to_numpy(dtype=float))
        log_brent = np.log(df["brent"].to_numpy(dtype=float))
        log_oro = np.log(df["oro_mxn_g"].to_numpy(dtype=float))
    x = pd.DataFrame(
        {
            "constante": 1.0,
            "rendimiento_real_10": df["rendimiento_real_10"].to_numpy(dtype=float),
            "log_usd_mxn": log_mxn,
            "log_brent": log_brent,
            "vix": df["vix"].to_numpy(dtype=float),
            "cpi_anual": df["cpi_anual"].to_numpy(dtype=float),
        },
        index=df.index,
    )
    usadas: list[str] = []
    for nombre in EXTRAS_VALOR:
        if nombre not in df.columns:
            continue
        valores = df[nombre].to_numpy(dtype=float)
        if int(np.isfinite(valores).sum()) >= 250 and float(np.nanstd(valores)) > 0:
            x[nombre] = valores
            usadas.append(nombre)
    y = pd.Series(log_oro, index=df.index, name="log_oro_mxn")
    return x, y, usadas


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


def valor_justo_ventana(
    x: pd.DataFrame,
    y: pd.Series,
    ventana: int,
    paso: int = PASO_REENTRENO,
) -> pd.DataFrame:
    """Valor justo con los últimos `ventana` días completos, sin el día t.

    La ventana es de observaciones, no de calendario: 504 sesiones son cerca
    de dos años de mercado, 756 cerca de tres. El día t no entra. Entre
    reentrenos se deja quieto el último vector de coeficientes.
    """
    matriz = x.to_numpy(dtype=float)
    objetivo = y.to_numpy(dtype=float)
    n = len(objetivo)
    completo = np.isfinite(matriz).all(axis=1) & np.isfinite(objetivo)
    valor = np.full(n, np.nan)
    coefs = np.full((n, matriz.shape[1]), np.nan)
    beta = None
    siguiente = -1
    for t in range(n):
        if not completo[t]:
            continue
        if beta is None or t >= siguiente:
            previos = np.flatnonzero(completo[:t])
            if len(previos) >= ventana:
                idx = previos[-ventana:]
                beta = np.linalg.lstsq(matriz[idx], objetivo[idx], rcond=None)[0]
                siguiente = t + paso
            elif beta is None:
                continue
        valor[t] = float(matriz[t] @ beta)
        coefs[t] = beta
    salida = pd.DataFrame({"log_valor": valor}, index=x.index)
    for i, nombre in enumerate(x.columns):
        salida[nombre] = coefs[:, i]
    return salida


def vida_media(alpha: float) -> float | None:
    """Periodos hasta que la brecha baje a la mitad, si es que se encoge.

    En el modelo de corrección, el rezago del error entra con coeficiente
    alpha. La brecha se multiplica cada periodo por (1 + alpha). Eso solo
    describe un regreso a la mitad si alpha está entre -1 y 0: más negativo
    que -1 la voltearía de un salto, y cero o positivo no la encoge.
    """
    if alpha is None or not np.isfinite(alpha):
        return None
    if alpha <= -1.0 or alpha >= 0.0:
        return None
    return float(math.log(2.0) / -math.log(1.0 + alpha))


def _adf_p(serie: pd.Series, maxlag: int = 4) -> float | None:
    valores = serie.dropna().to_numpy(dtype=float)
    if len(valores) < 40:
        return None
    try:
        _stat, pvalor, *_resto = adfuller(
            valores,
            maxlag=maxlag,
            autolag="AIC",
            result_object=False,
        )
        return float(pvalor)
    except Exception:  # noqa: BLE001
        return None


def _vida_media_ar1(residuo: pd.Series) -> dict:
    """Vida media descriptiva de un residuo, sin afirmar que sea equilibrio.

    Ajusta e_t = c + rho e_{t-1}. Si rho está entre 0 y 1, la vida media
    es log(2) / -log(rho). No es una prueba de cointegración.
    """
    serie = residuo.dropna().astype(float)
    if len(serie) < 50:
        return {"rho": None, "vida_media": None}
    y = serie.iloc[1:].to_numpy()
    x = sm.add_constant(serie.iloc[:-1].to_numpy())
    try:
        modelo = sm.OLS(y, x).fit()
        rho = float(modelo.params[1])
    except Exception:  # noqa: BLE001
        return {"rho": None, "vida_media": None}
    if not np.isfinite(rho) or rho <= 0 or rho >= 1:
        return {"rho": rho, "vida_media": None}
    return {"rho": rho, "vida_media": float(math.log(2.0) / -math.log(rho))}


def evaluar_cointegracion(df: pd.DataFrame, columnas: list[str] | None = None) -> dict:
    """Engle-Granger y Johansen, en datos semanales, y corrección de errores si aplica.

    Se usa el viernes (o el último día de la semana) para no tratar 5.000
    días seguidos como si fueran observaciones independientes. Primero se
    mira qué series parecen tener raíz unitaria. La cointegración se prueba
    entre el log del oro y esas series. Si alguna prueba rechaza 'no hay
    relación' al 5%, se estima el modelo de corrección y la vida media.
    Si no, la vida media de un AR(1) sobre la brecha se reporta aparte y
    no se presenta como equilibrio.
    """
    vacio = {
        "frecuencia": "semanal",
        "n": 0,
        "variables": [],
        "adf": {},
        "eg_stat": None,
        "eg_p": None,
        "johansen_trace": None,
        "johansen_crit_5": None,
        "johansen_rechaza_r0": None,
        "hay_relacion": False,
        "prueba_que_rechaza": [],
        "alpha_ecm": None,
        "ee_alpha": None,
        "t_alpha": None,
        "vida_media_semanas": None,
        "vida_media_dias": None,
        "ar1_rho": None,
        "ar1_vida_media_dias": None,
        "nota": "No hubo suficientes semanas completas para la prueba.",
    }
    if "oro_usd" not in df.columns:
        vacio["nota"] = "Falta el precio del oro."
        return vacio
    with np.errstate(divide="ignore", invalid="ignore"):
        log_oro = np.log(df["oro_usd"].astype(float))
    candidatos = {
        "log_dolar": np.log(df["dolar_index"].astype(float)) if "dolar_index" in df.columns else None,
        "log_brent": np.log(df["brent"].astype(float)) if "brent" in df.columns else None,
        "rendimiento_real_10": df["rendimiento_real_10"] if "rendimiento_real_10" in df.columns else None,
        "log_gld_ton": df["log_gld_ton"] if "log_gld_ton" in df.columns else None,
        "gpr": df["gpr"] if "gpr" in df.columns else None,
    }
    if columnas:
        candidatos = {nombre: serie for nombre, serie in candidatos.items() if nombre in columnas or nombre in ("log_dolar", "log_brent", "rendimiento_real_10")}
    marco = pd.DataFrame({"log_oro": log_oro}, index=df.index)
    for nombre, serie in candidatos.items():
        if serie is None:
            continue
        marco[nombre] = pd.Series(np.asarray(serie, dtype=float), index=df.index)
    semanal = marco.resample("W-FRI").last().dropna(how="any")
    if len(semanal) < 120:
        return vacio

    adf = {"log_oro": _adf_p(semanal["log_oro"])}
    integradas = []
    for nombre in semanal.columns:
        if nombre == "log_oro":
            continue
        pvalor = _adf_p(semanal[nombre])
        adf[nombre] = pvalor
        # No rechazar raíz unitaria al 10%: candidata a entrar al sistema.
        if pvalor is None or pvalor > 0.10:
            integradas.append(nombre)
    if not integradas:
        # Sin conductores integrados la prueba no tiene con qué relacionar el oro.
        # Se dejan dólar y petróleo, que son el núcleo del modelo, y se dice.
        for nombre in ("log_dolar", "log_brent"):
            if nombre in semanal.columns:
                integradas.append(nombre)
        nota_integracion = (
            "Ningún conductor rechazó quedarse fuera de la raíz unitaria con holgura. "
            "La prueba se hizo igual con dólar y petróleo, y hay que leerla con esa reserva."
        )
    else:
        nota_integracion = (
            "Entraron las series que, en la prueba ADF semanal, no rechazan raíz unitaria al 10%."
        )
    # Como máximo cuatro conductores: con más, Johansen se queda sin grados de libertad.
    integradas = integradas[:4]
    sistema = semanal[["log_oro", *integradas]].dropna()
    n = len(sistema)
    resultado = {
        "frecuencia": "semanal",
        "n": int(n),
        "variables": ["log_oro", *integradas],
        "adf": adf,
        "eg_stat": None,
        "eg_p": None,
        "johansen_trace": None,
        "johansen_crit_5": None,
        "johansen_rechaza_r0": None,
        "hay_relacion": False,
        "prueba_que_rechaza": [],
        "alpha_ecm": None,
        "ee_alpha": None,
        "t_alpha": None,
        "vida_media_semanas": None,
        "vida_media_dias": None,
        "ar1_rho": None,
        "ar1_vida_media_dias": None,
        "nota": nota_integracion,
    }
    y = sistema["log_oro"]
    x = sm.add_constant(sistema[integradas])
    try:
        eg_stat, eg_p, _crit = coint(y, sistema[integradas], trend="c", maxlag=4, autolag="aic")
        resultado["eg_stat"] = float(eg_stat)
        resultado["eg_p"] = float(eg_p)
    except Exception as error:  # noqa: BLE001
        resultado["nota"] += f" Engle-Granger no corrió ({error})."
    try:
        # Con series que sí se mueven juntas, un autovalor puede salir
        # apenas por encima de 1 por redondeo y statsmodels avisa. El
        # estadístico de traza sigue siendo usable si es un número real.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
            johansen = coint_johansen(sistema.to_numpy(dtype=float), det_order=0, k_ar_diff=2)
        traza = float(np.real(johansen.lr1[0]))
        critico = float(johansen.cvt[0, 1])
        if np.isfinite(traza) and np.isfinite(critico):
            resultado["johansen_trace"] = traza
            resultado["johansen_crit_5"] = critico
            resultado["johansen_rechaza_r0"] = bool(traza > critico)
        else:
            resultado["nota"] += " Johansen no dio un estadístico finito."
    except Exception as error:  # noqa: BLE001
        resultado["nota"] += f" Johansen no corrió ({error})."

    rechaza = []
    if resultado["eg_p"] is not None and resultado["eg_p"] < 0.05:
        rechaza.append("Engle-Granger")
    if resultado["johansen_rechaza_r0"]:
        rechaza.append("Johansen")
    resultado["prueba_que_rechaza"] = rechaza
    resultado["hay_relacion"] = len(rechaza) > 0

    # AR(1) de la brecha diaria, siempre, y etiquetado como descriptivo.
    if "residuo_vf" in df.columns:
        ar1 = _vida_media_ar1(df["residuo_vf"])
        resultado["ar1_rho"] = ar1["rho"]
        resultado["ar1_vida_media_dias"] = ar1["vida_media"]

    if not resultado["hay_relacion"]:
        resultado["nota"] += (
            " Ninguna de las dos pruebas rechaza, al 5%, que no haya relación de largo plazo. "
            "No se estima un modelo de corrección de errores."
        )
        return resultado

    try:
        niveles = sm.OLS(y, x).fit()
        ect = pd.Series(niveles.resid, index=sistema.index)
        dy = y.diff()
        regresores = pd.DataFrame({"ect_lag": ect.shift(1)}, index=sistema.index)
        regresores["dy_lag"] = dy.shift(1)
        for nombre in integradas:
            regresores[f"d_{nombre}"] = sistema[nombre].diff().shift(1)
        muestra = pd.concat([dy.rename("dy"), regresores], axis=1).dropna()
        modelo = sm.OLS(muestra["dy"], sm.add_constant(muestra.drop(columns="dy"))).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": 4, "use_correction": True},
        )
        alpha = float(modelo.params["ect_lag"])
        resultado["alpha_ecm"] = alpha
        resultado["ee_alpha"] = float(modelo.bse["ect_lag"])
        resultado["t_alpha"] = float(modelo.tvalues["ect_lag"])
        media = vida_media(alpha)
        resultado["vida_media_semanas"] = media
        if media is not None:
            resultado["vida_media_dias"] = media * 5.0
        else:
            resultado["nota"] += (
                " Hay señal de relación, pero el coeficiente de corrección no está entre -1 y 0, "
                "así que no hay una vida media de regreso a la mitad."
            )
    except Exception as error:  # noqa: BLE001
        resultado["nota"] += f" El modelo de corrección no se pudo estimar ({error})."
    return resultado


def _efecto_una_desviacion(y: pd.Series, x: pd.DataFrame) -> dict[str, float]:
    completo = _filas_completas(x, y)
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
    modelo = sm.OLS(y.loc[completo], x_std).fit()
    return {nombre: float(modelo.params[nombre]) for nombre in modelo.params.index}


def _bloque_niveles(df: pd.DataFrame, x: pd.DataFrame, y: pd.Series, prefijo: str) -> dict:
    """Ajuste de niveles, brecha de toda la muestra y brecha caminando hacia adelante."""
    completo = _filas_completas(x, y)
    if int(completo.sum()) < MIN_OBS_VALOR:
        return {"disponible": False, "n": int(completo.sum())}
    modelo = _ajustar_ols(y.loc[completo], x.loc[completo])
    expandido = valor_justo_expandiendo(x, y)
    df[f"log_valor_{prefijo}"] = expandido["log_valor_justo_vf"]
    df[f"residuo_{prefijo}"] = expandido["residuo_vf"]
    df[f"valor_{prefijo}"] = np.exp(df[f"log_valor_{prefijo}"])
    fecha = x.index[completo][-1]
    log_justo = float(np.dot(x.loc[fecha].to_numpy(dtype=float), modelo.params.to_numpy(dtype=float)))
    precio_log = float(y.loc[fecha])
    precio = float(math.exp(precio_log))
    valor = float(math.exp(log_justo))
    log_vf = df.loc[fecha, f"log_valor_{prefijo}"]
    valor_vf = float(math.exp(log_vf)) if np.isfinite(log_vf) else float("nan")
    efectos = _efecto_una_desviacion(y, x)
    coeficientes = []
    for nombre in x.columns:
        if nombre == "constante":
            continue
        coef_std = float(efectos.get(nombre, float("nan")))
        coeficientes.append(
            {
                "nombre": nombre,
                "etiqueta": ETIQUETAS_VALOR.get(nombre, nombre),
                "coef": float(modelo.params[nombre]),
                "ee": float(modelo.bse[nombre]),
                "t": float(modelo.tvalues[nombre]),
                "p": float(modelo.pvalues[nombre]),
                "coef_std": coef_std,
                "efecto_std_pct": float((math.exp(coef_std) - 1.0) * 100.0) if np.isfinite(coef_std) else float("nan"),
            }
        )
    coeficientes.sort(key=lambda fila: abs(fila["efecto_std_pct"]) if np.isfinite(fila["efecto_std_pct"]) else -1, reverse=True)
    return {
        "disponible": True,
        "fecha": fecha,
        "n": int(completo.sum()),
        "inicio": x.index[completo][0],
        "fin": x.index[completo][-1],
        "r2": float(modelo.rsquared),
        "precio": precio,
        "valor_justo": valor,
        "brecha_pct": (precio / valor - 1.0) * 100.0,
        "valor_justo_vf": valor_vf,
        "brecha_vf_pct": (precio / valor_vf - 1.0) * 100.0 if np.isfinite(valor_vf) and valor_vf > 0 else float("nan"),
        "coeficientes": coeficientes,
        "variables": [nombre for nombre in x.columns if nombre != "constante"],
        "vif": _vif(x.loc[completo]),
    }


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
        "extendido": _ajustar_extendido(df, y),
        "movil": _ajustar_movil(df),
        "cointegracion": evaluar_cointegracion(df),
    }


def _ajustar_extendido(df: pd.DataFrame, y_base: pd.Series) -> dict:
    """Modelo de niveles con las variables nuevas, y cuánto mueve la brecha."""
    x_ext, y_ext, usadas = matriz_valor_extendida(df)
    if not usadas:
        return {
            "disponible": False,
            "variables": [],
            "n": 0,
            "cierre_abs_pp": None,
            "nota": "No llegó ninguna variable nueva con historia suficiente.",
        }
    bloque = _bloque_niveles(df, x_ext, y_ext, "ext")
    if not bloque.get("disponible"):
        bloque["variables"] = usadas
        bloque["cierre_abs_pp"] = None
        bloque["nota"] = "Las variables nuevas no alcanzan para una muestra de tres años."
        return bloque
    # La brecha base caminando hacia adelante, el mismo día, para comparar.
    fecha = bloque["fecha"]
    precio = float(df.loc[fecha, "oro_usd"])
    valor_base = df.loc[fecha, "valor_justo_vf"] if "valor_justo_vf" in df.columns else np.nan
    if np.isfinite(valor_base) and valor_base > 0 and np.isfinite(bloque["brecha_vf_pct"]):
        brecha_base = (precio / float(valor_base) - 1.0) * 100.0
        bloque["brecha_base_vf_pct"] = brecha_base
        bloque["cierre_abs_pp"] = abs(brecha_base) - abs(bloque["brecha_vf_pct"])
        bloque["cierre_pp"] = brecha_base - bloque["brecha_vf_pct"]
    else:
        bloque["brecha_base_vf_pct"] = float("nan")
        bloque["cierre_abs_pp"] = None
        bloque["cierre_pp"] = None
    bloque["nota"] = "Variables nuevas: " + ", ".join(usadas) + "."
    # y_base queda para que el llamador no tenga que repetir el log del oro.
    _ = y_base
    return bloque


def _ajustar_movil(df: pd.DataFrame) -> dict:
    """Coeficientes en ventanas de 2 y 3 años, y el valor justo con esos pesos."""
    x_ext, y_ext, usadas = matriz_valor_extendida(df)
    if usadas:
        x, y = x_ext, y_ext
        especificacion = ["rendimiento_real_10", "log_dolar", "log_brent", "vix", "cpi_anual", *usadas]
    else:
        x, y = matriz_valor_justo(df)
        especificacion = ["rendimiento_real_10", "log_dolar", "log_brent", "vix", "cpi_anual"]
    salida = {"especificacion": especificacion, "ventanas": {}}
    for clave, ventana in (("2a", VENTANA_2A), ("3a", VENTANA_3A)):
        camino = valor_justo_ventana(x, y, ventana)
        df[f"log_valor_movil_{clave}"] = camino["log_valor"]
        df[f"valor_movil_{clave}"] = np.exp(camino["log_valor"])
        coefs = camino.drop(columns="log_valor")
        ultimo = {}
        for nombre in coefs.columns:
            validos = coefs[nombre].dropna()
            ultimo[nombre] = float(validos.iloc[-1]) if len(validos) else float("nan")
        fecha_ultimo = coefs.dropna(how="all").index[-1] if coefs.notna().any().any() else None
        precio = float("nan")
        valor = float("nan")
        brecha = float("nan")
        if fecha_ultimo is not None and np.isfinite(df.loc[fecha_ultimo, f"valor_movil_{clave}"]):
            # La brecha que se reporta es la del último día del oro con valor móvil,
            # que es el mismo calendario del modelo si ese día está completo.
            fecha_precio = df.index[np.isfinite(df[f"valor_movil_{clave}"].to_numpy(dtype=float))][-1]
            precio = float(df.loc[fecha_precio, "oro_usd"])
            valor = float(df.loc[fecha_precio, f"valor_movil_{clave}"])
            brecha = (precio / valor - 1.0) * 100.0 if valor > 0 else float("nan")
            fecha_ultimo = fecha_precio
        salida["ventanas"][clave] = {
            "ventana": ventana,
            "coeficientes": coefs,
            "ultimo": ultimo,
            "fecha": fecha_ultimo,
            "precio": precio,
            "valor_justo": valor,
            "brecha_pct": brecha,
        }
    return salida


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
    pred_boosting = np.full(n, np.nan)
    media_entreno = np.full(n, np.nan)
    alpha_usado = np.full(n, np.nan)
    modelo = None
    bosque = None
    media_beta = None
    desviacion_beta = None
    media_y = None
    alpha = None
    coef = None
    importancias = None
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
                # El bosque no se estandariza: un árbol no cambia si una
                # columna se multiplica por una constante. Mismo embargo,
                # mismas filas y el mismo calendario de reentreno que Ridge.
                bosque = GradientBoostingRegressor(**GBR_PARAMS)
                bosque.fit(x_ent, y_ent)
                media_y = float(y_ent.mean())
                coef = modelo.coef_.copy()
                importancias = bosque.feature_importances_.copy()
                n_entreno = int(len(idx))
                siguiente = t + paso
            elif modelo is None:
                continue
        pred[t] = float(modelo.predict(((x[t] - media_beta) / desviacion_beta).reshape(1, -1))[0])
        pred_boosting[t] = float(bosque.predict(x[t].reshape(1, -1))[0])
        media_entreno[t] = media_y
        alpha_usado[t] = alpha
    return {
        "pred": pred,
        "pred_boosting": pred_boosting,
        "media": media_entreno,
        "alpha": alpha_usado,
        "coef": coef,
        "importancias": importancias,
        "alpha_ultimo": None if alpha is None else float(alpha),
        "n_entreno_ultimo": n_entreno,
    }


def _empaquetar_horizonte(
    df: pd.DataFrame,
    y: np.ndarray,
    corrida: dict,
    fila_x: np.ndarray,
    horizonte: int,
    columnas: list[str],
    etiquetas: dict[str, str],
) -> dict:
    """Arma el bloque de un horizonte: Ridge, boosting y el mismo baseline."""
    metricas = metricas_pronostico(y, corrida["pred"], corrida["media"], horizonte)
    metricas_b = metricas_pronostico(y, corrida["pred_boosting"], corrida["media"], horizonte)
    con_pred = np.isfinite(corrida["pred"])
    fechas = df.index[con_pred]
    ultimos = np.flatnonzero(fila_x)
    t_ultimo = int(ultimos[-1])
    sigma_base = metricas["sin_solapar"]
    if sigma_base["n"] < 30:
        sigma_base = metricas["solapado"]
        sigma_origen = "solapado"
    else:
        sigma_origen = "sin solapar"
    sigma_b = metricas_b["sin_solapar"]
    if sigma_b["n"] < 30:
        sigma_b = metricas_b["solapado"]
    pesos = []
    if corrida["coef"] is not None:
        for nombre, peso in zip(columnas, corrida["coef"]):
            pesos.append({"nombre": nombre, "etiqueta": etiquetas.get(nombre, nombre), "peso": float(peso)})
        pesos.sort(key=lambda fila: abs(fila["peso"]), reverse=True)
    importancias = []
    if corrida["importancias"] is not None:
        for nombre, peso in zip(columnas, corrida["importancias"]):
            importancias.append(
                {"nombre": nombre, "etiqueta": etiquetas.get(nombre, nombre), "peso": float(peso)}
            )
        importancias.sort(key=lambda fila: abs(fila["peso"]), reverse=True)
    realizados = np.isfinite(corrida["pred"]) & np.isfinite(y)
    return {
        "metricas": metricas,
        "metricas_boosting": metricas_b,
        "pred_ultimo": float(corrida["pred"][t_ultimo]),
        "pred_ultimo_boosting": float(corrida["pred_boosting"][t_ultimo]),
        "fecha_ultimo": df.index[t_ultimo],
        "sigma": float(sigma_base["rmse"]) if sigma_base["rmse"] is not None else float("nan"),
        "sigma_boosting": float(sigma_b["rmse"]) if sigma_b["rmse"] is not None else float("nan"),
        "sigma_origen": sigma_origen,
        "alpha_ultimo": corrida["alpha_ultimo"],
        "n_entreno_ultimo": corrida["n_entreno_ultimo"],
        "pesos": pesos,
        "importancias": importancias,
        "oos_inicio": fechas.min() if len(fechas) else None,
        "oos_fin_realizado": df.index[realizados].max() if np.any(realizados) else None,
    }


def ajustar_pronosticos(
    df: pd.DataFrame,
    min_entrenamiento: int = MIN_ENTRENAMIENTO,
) -> dict:
    """Walk-forward para 5 y 20 días. Ridge y boosting, mismo embargo."""
    if "residuo_vf" not in df.columns:
        raise RuntimeError("Falta la brecha del valor justo. Ajusta ese modelo primero.")
    faltan = [columna for columna in FEATURES_PRONOSTICO if columna not in df.columns]
    if faltan:
        raise RuntimeError(f"Faltan columnas para el pronóstico: {', '.join(faltan)}")

    columnas = columnas_pronostico(df)
    etiquetas = {**ETIQUETAS_PRONOSTICO, **ETIQUETAS_EXTRAS}
    x = df[columnas].to_numpy(dtype=float)
    fila_x = np.isfinite(x).all(axis=1)
    if int(fila_x.sum()) < min_entrenamiento + 40:
        raise RuntimeError("No hay historia suficiente para el pronóstico fuera de muestra.")

    resultados = {}
    for horizonte in HORIZONTES:
        columna_y = f"y_{horizonte}"
        y = df[columna_y].to_numpy(dtype=float)
        fila_y = np.isfinite(y)
        corrida = pronostico_walk_forward(
            x, y, fila_x, fila_y, horizonte, min_entrenamiento=min_entrenamiento
        )
        df[f"pred_{horizonte}"] = corrida["pred"]
        df[f"pred_gb_{horizonte}"] = corrida["pred_boosting"]
        df[f"media_{horizonte}"] = corrida["media"]
        resultados[horizonte] = _empaquetar_horizonte(
            df, y, corrida, fila_x, horizonte, columnas, etiquetas
        )
    return {"df": df, "por_horizonte": resultados, "columnas": columnas}


def ajustar_oro_mxn(
    df: pd.DataFrame,
    min_entrenamiento: int = MIN_ENTRENAMIENTO,
) -> dict:
    """Valor justo y pronóstico del oro en pesos por gramo.

    No es el valor justo en dólares pasado al tipo de cambio de hoy: eso
    dejaría la brecha en el mismo porcentaje. Aquí la regresión y el
    pronóstico se estiman sobre el precio en pesos.
    """
    if "oro_mxn_g" not in df.columns:
        raise RuntimeError("Falta el oro en pesos por gramo. Prepara la muestra primero.")
    x, y, usadas = matriz_valor_mxn(df)
    bloque = _bloque_niveles(df, x, y, "mxn")
    if not bloque.get("disponible"):
        raise RuntimeError("No hay historia suficiente para el valor justo en pesos.")
    columnas = columnas_pronostico(df, base=FEATURES_PRONOSTICO_MXN, extras=EXTRAS_PRONOSTICO)
    faltan = [columna for columna in FEATURES_PRONOSTICO_MXN if columna not in df.columns]
    if faltan:
        raise RuntimeError(f"Faltan columnas para el pronóstico en pesos: {', '.join(faltan)}")
    matriz = df[columnas].to_numpy(dtype=float)
    fila_x = np.isfinite(matriz).all(axis=1)
    if int(fila_x.sum()) < min_entrenamiento + 40:
        raise RuntimeError("No hay historia suficiente para el pronóstico en pesos.")
    etiquetas = {**ETIQUETAS_PRONOSTICO, **ETIQUETAS_EXTRAS}
    pronostico = {}
    for horizonte in HORIZONTES:
        objetivo = df[f"y_mxn_{horizonte}"].to_numpy(dtype=float)
        fila_y = np.isfinite(objetivo)
        corrida = pronostico_walk_forward(
            matriz,
            objetivo,
            fila_x,
            fila_y,
            horizonte,
            min_entrenamiento=min_entrenamiento,
        )
        df[f"pred_mxn_{horizonte}"] = corrida["pred"]
        df[f"pred_mxn_gb_{horizonte}"] = corrida["pred_boosting"]
        df[f"media_mxn_{horizonte}"] = corrida["media"]
        pronostico[horizonte] = _empaquetar_horizonte(
            df, objetivo, corrida, fila_x, horizonte, columnas, etiquetas
        )
    bloque["usadas_extra"] = usadas
    return {"valor": bloque, "por_horizonte": pronostico, "columnas": columnas, "df": df}
