"""Texto y gráficas del informe.

Este archivo no vuelve a estimar nada: solo cuenta, en español, lo que
ya calcularon `datos` y `modelos`, y guarda las figuras en outputs/.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from modelos import onza_a_mxn_por_gramo

RAIZ = Path(__file__).resolve().parent
RUTA_SNAPSHOT = RAIZ / "referencia" / "snapshot_2026-10-05.csv"

MESES = [
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
]

COLOR_ORO = "#A67C2D"
COLOR_MODELO = "#1D4E89"
COLOR_CARO = "#A33B20"
COLOR_BARATO = "#2E6B4F"
COLOR_TEXTO = "#243042"


def _sufijo_fuente(datos: dict, clave: str, fecha_modelo) -> str:
    """Avisa cuando el número del informe es anterior al día del oro."""
    cruda = datos.get("fechas", {}).get(clave)
    if not cruda:
        return ""
    fecha = pd.Timestamp(cruda)
    if fecha.normalize() < pd.Timestamp(fecha_modelo).normalize():
        return f" (publicado el {fecha.date()})"
    return ""


def fecha_larga(fecha) -> str:
    fecha = pd.Timestamp(fecha)
    return f"{fecha.day} de {MESES[fecha.month - 1]} de {fecha.year}"


def num(valor, decimales: int = 2) -> str:
    if valor is None or not math.isfinite(float(valor)):
        return "n/d"
    return f"{float(valor):,.{decimales}f}"


def pct(valor, decimales: int = 1, signo: bool = False) -> str:
    if valor is None or not math.isfinite(float(valor)):
        return "n/d"
    if signo:
        return f"{float(valor):+.{decimales}f}%"
    return f"{float(valor):.{decimales}f}%"


def r2_txt(valor) -> str:
    if valor is None or not math.isfinite(float(valor)):
        return "n/d"
    return f"{float(valor):+.3f}"


def parrafo(texto: str) -> str:
    return textwrap.fill(" ".join(texto.split()), width=78)


def _prob_alza(pred: float, sigma: float) -> float:
    if not math.isfinite(pred) or not math.isfinite(sigma) or sigma <= 0:
        return float("nan")
    z = pred / sigma
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def lectura_coeficiente(fila: dict) -> str:
    nombre = fila["nombre"]
    coef = fila["coef"]
    if nombre in {"rendimiento_real_10", "cpi_anual"}:
        cambio = (math.exp(coef) - 1.0) * 100.0
        unidad = "punto porcentual de tasa real" if nombre == "rendimiento_real_10" else "punto de inflación anual"
        return f"+1 {unidad} se asocia con {cambio:+.1f}% en el oro"
    if nombre == "vix":
        cambio = (math.exp(coef) - 1.0) * 100.0
        return f"+1 punto de VIX se asocia con {cambio:+.2f}% en el oro"
    if nombre == "log_dolar":
        return f"un dólar 1% más alto se asocia con {coef:+.2f}% en el oro"
    if nombre == "log_brent":
        return f"un petróleo 1% más alto se asocia con {coef:+.2f}% en el oro"
    return ""


def _valor_en_fecha(serie: pd.Series, fecha: str):
    fecha = pd.Timestamp(fecha)
    serie = serie.dropna()
    if serie.empty:
        return None, None
    if fecha in serie.index and pd.notna(serie.loc[fecha]):
        return fecha, float(serie.loc[fecha])
    previos = serie.loc[:fecha]
    if previos.empty:
        return None, None
    ultima = previos.index[-1]
    if (fecha - ultima).days > 4:
        return None, None
    return ultima, float(previos.iloc[-1])


def _a_float(texto) -> float | None:
    try:
        return float(str(texto).replace("~", "").replace("%", "").replace(",", "").strip())
    except (TypeError, ValueError):
        return None


def _bloque_snapshot(df: pd.DataFrame, datos: dict | None = None) -> list[str]:
    if not RUTA_SNAPSHOT.exists():
        return ["No está el archivo de referencia del 5 de octubre de 2026."]
    tabla = pd.read_csv(RUTA_SNAPSHOT)
    ref = dict(zip(tabla["metric"], tabla["value"]))
    lineas = [
        "La nota del 5 de octubre de 2026 es solo un chequeo. No entra al modelo.",
        "El modelo usa el cierre del futuro GC=F; la nota mezcla spot y un contrato COMEX.",
    ]
    comparaciones = [
        ("spot_gold_close", "oro_usd", "Oro", "USD/oz", 2),
        ("usdmxn_spot_close", "usd_mxn", "USD/MXN", "pesos por dólar", 4),
        ("dxy", "dolar_index", "Índice dólar", "puntos", 2),
    ]
    for clave, columna, titulo, unidad, dec in comparaciones:
        if columna not in df.columns or clave not in ref:
            continue
        esperado = _a_float(ref[clave])
        if esperado is None:
            continue
        fecha, nuestro = _valor_en_fecha(df[columna], "2026-10-05")
        if nuestro is None:
            lineas.append(f"{titulo}: no tengo dato cerca del 5 de octubre.")
            continue
        dif = (nuestro / esperado - 1.0) * 100.0
        lineas.append(
            f"{titulo} el {pd.Timestamp(fecha).date()}: este proyecto {num(nuestro, dec)} {unidad}; "
            f"la nota {num(esperado, dec)}. Diferencia {dif:+.2f}%."
        )
    if "rendimiento_nominal_10" in df.columns and "us10y_yield" in ref:
        esperado = _a_float(ref["us10y_yield"])
        fecha, nuestro = _valor_en_fecha(df["rendimiento_nominal_10"], "2026-10-05")
        if esperado is not None and nuestro is not None:
            lineas.append(
                f"Tasa nominal a 10 años el {pd.Timestamp(fecha).date()}: "
                f"{nuestro:.2f}% aquí, cerca de {esperado:.2f}% en la nota."
            )
    if "oro_usd" in df.columns and "comex_front_week_change" in ref:
        try:
            retorno = float(df.loc["2026-10-02", "oro_usd"]) / float(df.loc["2026-09-25", "oro_usd"]) - 1.0
            nota = _a_float(ref["comex_front_week_change"])
            if nota is not None:
                lineas.append(
                    "Semana del 25 de septiembre al 2 de octubre: "
                    f"GC=F {retorno * 100:+.2f}%. La nota cita {nota:+.2f}% para el COMEX. "
                    "El movimiento semanal debería parecerse, aunque el nivel no sea idéntico."
                )
        except KeyError:
            pass
    if datos is not None:
        atrasados = []
        for clave, etiqueta in (
            ("rendimiento_real_10", "tasa real"),
            ("rendimiento_nominal_10", "tasa nominal"),
            ("brent", "Brent"),
            ("vix", "VIX"),
        ):
            cruda = datos.get("fechas", {}).get(clave)
            if cruda and pd.Timestamp(cruda) < pd.Timestamp("2026-10-05"):
                atrasados.append(f"{etiqueta} hasta {pd.Timestamp(cruda).date()}")
        if atrasados:
            lineas.append(
                "Estas series no llegan al 5 de octubre, así que un número de ese día en la tabla "
                "puede ser el último dato arrastrado: " + "; ".join(atrasados) + "."
            )
    return lineas


def _texto_pronostico(horizonte: int, info: dict) -> list[str]:
    pred = float(info["pred_ultimo"])
    sigma = float(info["sigma"])
    esperado = (math.exp(pred) - 1.0) * 100.0
    bajo = (math.exp(pred - sigma) - 1.0) * 100.0 if math.isfinite(sigma) else float("nan")
    alto = (math.exp(pred + sigma) - 1.0) * 100.0 if math.isfinite(sigma) else float("nan")
    limpio = info["metricas"]["sin_solapar"]
    solapado = info["metricas"]["solapado"]
    sin_habilidad = math.isfinite(limpio["r2_media"]) and limpio["r2_media"] < 0
    senal_chica = (not math.isfinite(sigma)) or abs(pred) < 0.25 * sigma
    if pred > 0:
        hacia = "alza"
    elif pred < 0:
        hacia = "baja"
    else:
        hacia = "un cambio de cero"
    lineas = [
        f"Horizonte de {horizonte} días de mercado (no de calendario).",
        f"La fórmula apunta a {hacia}: {pct(esperado, 2, signo=True)}.",
        f"Banda de un error típico fuera de muestra ({info['sigma_origen']}): "
        f"{pct(bajo, 2, signo=True)} a {pct(alto, 2, signo=True)}.",
        f"Traducción aproximada, si el error fuera normal: "
        f"P(alza) ≈ {pct(_prob_alza(pred, sigma) * 100, 0)}. "
        "Eso solo reparte la señal frente al ruido; no es una probabilidad de mercado.",
    ]
    if sin_habilidad:
        lineas.append(
            "Fuera de muestra, este horizonte no le gana al promedio histórico del retorno. "
            "La dirección de arriba es lo que sale de la cuenta, y la evidencia no alcanza "
            "para tratarla como una señal."
        )
    elif senal_chica:
        lineas.append(
            "La señal cabe varias veces dentro del error típico. La dirección es débil."
        )
    lineas.append(
        f"Evaluación sin solapar (una observación cada {horizonte} sesiones, "
        f"n={limpio['n']}):"
    )
    lineas.append(
        f"  R² contra retorno cero: {r2_txt(limpio['r2_cero'])}.  "
        f"R² contra el promedio del entrenamiento: {r2_txt(limpio['r2_media'])}."
    )
    lineas.append(
        f"  RMSE modelo {pct(limpio['rmse'] * 100, 2)} | "
        f"baseline cero {pct(limpio['rmse_cero'] * 100, 2)} | "
        f"baseline promedio {pct(limpio['rmse_media'] * 100, 2)}."
    )
    lineas.append(
        f"  Acierto de dirección {pct(limpio['acierto'] * 100, 1)} | "
        f"siempre alza {pct(limpio['acierto_siempre_alza'] * 100, 1)} | "
        f"signo del promedio {pct(limpio['acierto_signo_media'] * 100, 1)} | "
        f"predicciones positivas {pct(limpio['pct_pred_positivas'] * 100, 1)}."
    )
    lineas.append(
        f"Con días solapados (n={solapado['n']}, no son independientes): "
        f"R² contra cero {r2_txt(solapado['r2_cero'])}, "
        f"R² contra promedio {r2_txt(solapado['r2_media'])}, "
        f"acierto {pct(solapado['acierto'] * 100, 1)}."
    )
    if info["oos_inicio"] is not None:
        fin_realizado = info["oos_fin_realizado"]
        fin_txt = "n/d" if fin_realizado is None or pd.isna(fin_realizado) else str(pd.Timestamp(fin_realizado).date())
        lineas.append(
            f"Periodo fuera de muestra con predicción: {pd.Timestamp(info['oos_inicio']).date()} "
            f"a {pd.Timestamp(info['fecha_ultimo']).date()}. "
            f"El último retorno ya realizado llega a {fin_txt}. "
            f"Último Ridge con alpha {info['alpha_ultimo']} y "
            f"{info['n_entreno_ultimo']} filas de entrenamiento."
        )
    if info["alpha_ultimo"] is not None and float(info["alpha_ultimo"]) >= 100:
        lineas.append(
            f"El alpha de Ridge quedó en {float(info['alpha_ultimo']):g}. "
            "Una penalización alta aplasta los pesos hacia cero, así que el pronóstico se acerca "
            "al promedio histórico y es difícil que le gane."
        )
    if info["pesos"]:
        top = info["pesos"][:3]
        pesos = ", ".join(f"{fila['etiqueta']} ({fila['peso']:+.3f})" for fila in top)
        lineas.append(f"Mayores pesos estandarizados del último ajuste: {pesos}.")
    return lineas


def construir_informe(datos: dict, valor: dict, pronostico: dict) -> str:
    df = pronostico["df"]
    fecha_modelo = pd.Timestamp(valor["fecha"])
    fila = df.loc[fecha_modelo]
    ultima_oro = df["oro_usd"].dropna().index[-1]
    precio_ultimo = float(df.loc[ultima_oro, "oro_usd"])
    mxn = float(fila["usd_mxn"]) if pd.notna(fila.get("usd_mxn")) else float("nan")
    if pd.notna(df.loc[ultima_oro].get("usd_mxn")):
        mxn_precio = float(df.loc[ultima_oro, "usd_mxn"])
        fecha_mxn = ultima_oro
    else:
        mxn_precio = mxn
        fecha_mxn = fecha_modelo
    pesos_gramo = (
        onza_a_mxn_por_gramo(precio_ultimo, mxn_precio) if math.isfinite(mxn_precio) else float("nan")
    )

    lineas: list[str] = []
    sep = "=" * 72
    lineas.append(sep)
    lineas.append("ORO — INFORME DEL MODELO")
    lineas.append(f"Dato del modelo: {fecha_larga(fecha_modelo)}")
    lineas.append("Esto no es asesoría financiera.")
    lineas.append("Pronosticar el oro a corto plazo es difícil: el ruido le gana a la señal.")
    lineas.append(sep)
    lineas.append("")
    lineas.append("1) Precio")
    lineas.append(
        f"Futuro COMEX (Yahoo GC=F) al {fecha_larga(ultima_oro)}: "
        f"{num(precio_ultimo)} USD por onza troy."
    )
    if ultima_oro.normalize() == pd.Timestamp.today().normalize():
        lineas.append(
            "Esa fecha es hoy. Si el mercado sigue abierto, el cierre del día todavía puede moverse."
        )
    if math.isfinite(mxn_precio):
        lineas.append(
            f"Tipo de cambio al {fecha_larga(fecha_mxn)}: {num(mxn_precio, 4)} pesos por dólar."
        )
        lineas.append(
            f"{num(precio_ultimo)} × {num(mxn_precio, 4)} ÷ 31.1034768 = "
            f"{num(pesos_gramo)} pesos por gramo."
        )
        lineas.append(
            "Eso es el valor del metal en el futuro y el tipo de cambio de mercado. "
            "Una joyería o un banco en México cobra una prima encima, y a veces usa el FIX de Banxico."
        )
    if "plata" in df.columns and pd.notna(df.loc[ultima_oro, "plata"]) and df.loc[ultima_oro, "plata"] > 0:
        razon = precio_ultimo / float(df.loc[ultima_oro, "plata"])
        lineas.append(
            f"Relación oro/plata (onzas de plata por onza de oro): {num(razon, 1)}. "
            "Es contexto. La plata no entra a la regresión, porque se mueve con el oro."
        )
    try:
        cierre_2025 = float(df.loc[: "2025-12-31", "oro_usd"].dropna().iloc[-1])
        fecha_2025 = df.loc[: "2025-12-31", "oro_usd"].dropna().index[-1]
        ytd = precio_ultimo / cierre_2025 - 1.0
        lineas.append(
            f"En lo que va de 2026 el futuro va {pct(ytd * 100, 2, signo=True)} "
            f"frente al {fecha_larga(fecha_2025)} ({num(cierre_2025)} USD/oz). Eso ya pasó; no es un pronóstico."
        )
    except (KeyError, IndexError):
        pass
    lineas.append("")
    lineas.append("2) Con qué se armó el valor justo, ese día")
    lineas.append(
        "Si la fecha del dato es anterior a la del oro, el valor se arrastró hacia adelante "
        "(como máximo 5 días de mercado). No es una publicación de hoy."
    )
    lineas.append(
        f"Tasa real 10 años: {num(fila['rendimiento_real_10'], 2)}%"
        f"{_sufijo_fuente(datos, 'rendimiento_real_10', fecha_modelo)}"
    )
    lineas.append(
        f"Dólar (ICE): {num(fila['dolar_index'], 2)}"
        f"{_sufijo_fuente(datos, 'dolar_index', fecha_modelo)}"
    )
    lineas.append(
        f"Brent: {num(fila['brent'], 2)} USD"
        f"{_sufijo_fuente(datos, 'brent', fecha_modelo)}"
    )
    lineas.append(
        f"VIX: {num(fila['vix'], 2)}"
        f"{_sufijo_fuente(datos, 'vix', fecha_modelo)}"
    )
    lineas.append(f"IPC anual conocido: {pct(float(fila['cpi_anual']), 2)}")
    if "rendimiento_nominal_10" in df.columns and pd.notna(fila.get("rendimiento_nominal_10")):
        lineas.append(
            f"Tasa nominal 10 años (no entra al modelo): {num(fila['rendimiento_nominal_10'], 2)}%"
            f"{_sufijo_fuente(datos, 'rendimiento_nominal_10', fecha_modelo)}"
        )
    if "breakeven_10" in df.columns and pd.notna(fila.get("breakeven_10")):
        etiqueta_be = datos["fuentes"].get("breakeven_10", {}).get("etiqueta", "breakeven")
        lineas.append(
            f"Breakeven 10 años ({etiqueta_be}): {num(fila['breakeven_10'], 2)}%"
            f"{_sufijo_fuente(datos, 'breakeven_10', fecha_modelo)}. "
            "No entra al modelo junto con la tasa real."
        )
    if "fed_funds" in df.columns and pd.notna(fila.get("fed_funds")):
        lineas.append(f"Fondos federales (contexto, no entra al modelo): {num(fila['fed_funds'], 2)}%")
    if pd.notna(fila.get("cpi_mes")):
        lineas.append(
            f"IPC de referencia: {fecha_larga(fila['cpi_mes'])}, tratado como conocido desde "
            f"el {fecha_larga(fila['cpi_fecha_publicacion'])} (regla del día 15 del mes siguiente, "
            "o el lunes si cae en fin de semana)."
        )
    if pd.notna(fila.get("ret_20")) and pd.notna(fila.get("d_real_20")) and pd.notna(fila.get("d_dolar_20")):
        lineas.append(
            f"Ya ocurrió, últimos 20 días de mercado: oro {pct((math.exp(fila['ret_20']) - 1) * 100, 2, signo=True)}, "
            f"tasa real {float(fila['d_real_20']):+.2f} puntos, "
            f"dólar {pct((math.exp(fila['d_dolar_20']) - 1) * 100, 2, signo=True)}."
        )
    lineas.append("")
    lineas.append("3) Valor justo (regresión en niveles)")
    lineas.append("log(oro) = constante + real 10 años + log(dólar) + log(Brent) + VIX + inflación.")
    lineas.append(
        f"Muestra completa: {fecha_larga(valor['inicio'])} a {fecha_larga(valor['fin'])}, n={valor['n']} días."
    )
    lineas.append(f"Precio ese día: {num(valor['precio'])} USD/oz.")
    lineas.append(
        f"Valor justo con coeficientes de toda la historia: {num(valor['valor_justo'])} USD/oz. "
        f"Brecha: {pct(valor['brecha_pct'], 1, signo=True)} "
        f"({num(valor['precio'] - valor['valor_justo'], 0)} USD/oz)."
    )
    lineas.append(
        f"Valor justo caminando hacia adelante (coeficientes solo con el pasado): "
        f"{num(valor['valor_justo_vf'])} USD/oz. "
        f"Brecha: {pct(valor['brecha_vf_pct'], 1, signo=True)}."
    )
    if math.isfinite(valor["brecha_pct"]) and math.isfinite(valor["brecha_vf_pct"]):
        if abs(valor["brecha_pct"] - valor["brecha_vf_pct"]) > 5:
            lineas.append(
                "Las dos brechas se separan más de 5 puntos. La relación estimada con todo el historial "
                "no es la misma que la estimada solo con datos anteriores: la relación se ha movido."
            )
    if abs(valor["brecha_pct"]) > 15:
        lineas.append(
            "Una brecha ancha también puede querer decir que al modelo le faltan piezas. "
            "No incluye compras de bancos centrales ni primas geopolíticas, y en estos años han pesado."
        )
        lineas.append(
            "Tampoco es una orden de venta ni de compra. La gráfica muestra que el precio puede "
            "pasar años por encima del modelo. El pronóstico de 5 y 20 días, más abajo, es el que "
            "dice si el modelo ve un regreso pronto, y hay que leerlo con su error."
        )
    lineas.append(
        f"R² dentro de muestra: {valor['r2']:.3f}. "
        f"R² de una simple tendencia en el tiempo, misma muestra: {valor['r2_tendencia']:.3f}."
    )
    lineas.append(
        "El R² en niveles, con series que suben durante años, se ve alto con facilidad. "
        "No mide el pronóstico. El pronóstico está en la sección 4 y es fuera de muestra."
    )
    if valor["adf_p"] is None:
        lineas.append("No pude calcular la prueba ADF sobre la brecha.")
    elif valor["adf_p"] > 0.05:
        lineas.append(
            f"Prueba ADF sobre el residuo de la muestra completa: estadístico {valor['adf_stat']:.2f}, "
            f"p={valor['adf_p']:.3f}. No rechaza raíz unitaria al 5%. "
            "La brecha puede irse lejos y tardar en volver. No es un imán de corto plazo."
        )
    else:
        lineas.append(
            f"Prueba ADF sobre el residuo: estadístico {valor['adf_stat']:.2f}, p={valor['adf_p']:.3f}. "
            "Rechaza raíz unitaria al 5%. Aun así, la prueba no dice cuándo vuelve la brecha, "
            "y con datos diarios hay que leerla con calma."
        )
    lineas.append("")
    lineas.append("Coeficientes (error estándar de Newey-West, 21 rezagos).")
    lineas.append("EE es el error estándar. t = coeficiente / EE. Si |t| es menor que 2, el dato no separa el coeficiente de cero.")
    lineas.append(f"{'Variable':<18}{'Coef':>10}{'EE':>10}{'t':>8}   Lectura")
    for fila_coef in valor["coeficientes"]:
        lineas.append(
            f"{fila_coef['etiqueta']:<18}"
            f"{fila_coef['coef']:>10.3f}"
            f"{fila_coef['ee']:>10.3f}"
            f"{fila_coef['t']:>8.2f}   "
            f"{lectura_coeficiente(fila_coef)}"
        )
    lineas.append(
        "La constante no se lee sola: sería el log del oro si todo lo demás fuera cero, y ese cero no ocurre."
    )
    lineas.append("Misma historia, partida en 2022, para ver si el signo de la tasa real y del dólar se quedó quieto:")
    for titulo, bloque in (("Hasta 2021", valor["antes_2022"]), ("Desde 2022", valor["desde_2022"])):
        if not bloque:
            lineas.append(f"  {titulo}: muestra corta.")
            continue
        lineas.append(
            f"  {titulo}: n={bloque['n']}, R² {bloque['r2']:.2f}, "
            f"tasa real {bloque['coef_real']:+.3f} (EE {bloque['ee_real']:.3f}), "
            f"dólar {bloque['coef_dolar']:+.3f} (EE {bloque['ee_dolar']:.3f})."
        )
    vifs_altos = [f"{nombre} ({vif:.1f})" for nombre, vif in valor["vif"].items() if math.isfinite(vif) and vif >= 10]
    if vifs_altos:
        lineas.append(
            "Inflación de varianza alta (VIF ≥ 10) en "
            + ", ".join(vifs_altos)
            + ". Esas columnas se parecen entre sí y su coeficiente individual es menos firme."
        )
    lineas.append("")
    lineas.append("4) Pronóstico a 5 y 20 días, fuera de muestra")
    lineas.append(
        "Cada predicción usa solo información ya conocida ese día. El entrenamiento deja fuera "
        "las últimas observaciones cuyo retorno futuro todavía no había cerrado (embargo del horizonte). "
        "Se reentrena más o menos cada 21 sesiones. Ridge estandariza las columnas con la media y la "
        "desviación del propio entrenamiento."
    )
    lineas.append(
        "Baseline cero: predecir que el retorno logarítmico es 0 (caminata aleatoria, sin deriva). "
        "Baseline promedio: predecir el retorno medio del entrenamiento de esa ventana."
    )
    for horizonte in (5, 20):
        lineas.append("")
        lineas.extend(_texto_pronostico(horizonte, pronostico["por_horizonte"][horizonte]))
    lineas.append("")
    lineas.append("5) Qué variables pesan más en el nivel del oro")
    lineas.append(
        "Ordenadas por el efecto de subir una desviación estándar de la muestra "
        "(un cambio grande, de un nivel típico a otro, no un salto de un solo día)."
    )
    for i, fila_coef in enumerate(valor["coeficientes"], start=1):
        firme = "el |t| pasa de 2" if abs(fila_coef["t"]) >= 2 else "el |t| no llega a 2: magnitud imprecisa"
        lineas.append(
            f"  {i}. {fila_coef['etiqueta']}: {pct(fila_coef['efecto_std_pct'], 1, signo=True)} "
            f"en el precio. t={fila_coef['t']:+.2f} ({firme})."
        )
    antes = valor["antes_2022"]
    despues = valor["desde_2022"]
    if antes and despues and antes["coef_dolar"] * despues["coef_dolar"] < 0:
        lineas.append(parrafo(
            "El dólar encabeza esa lista en la muestra completa, con signo positivo: "
            "un dólar más fuerte se asocia, en el promedio 2006-2026, con un oro más caro. "
            f"El signo no se queda quieto: {antes['coef_dolar']:+.2f} hasta 2021 y "
            f"{despues['coef_dolar']:+.2f} desde 2022. No lo uses como regla."
        ))
    if antes and despues and antes["coef_real"] * despues["coef_real"] < 0:
        lineas.append(parrafo(
            "La tasa real tiene el signo de manual en la muestra completa (más tasa real, menos oro), "
            f"pero el periodo se parte: {antes['coef_real']:+.2f} hasta 2021 y "
            f"{despues['coef_real']:+.2f} desde 2022. El promedio esconde ese quiebre."
        ))
    lineas.append("")
    lineas.append("6) Para explicarle a tu papá, en corto")
    brecha = valor["brecha_pct"]
    if brecha >= 8:
        donde = f"el futuro está cerca de {brecha:.0f}% por encima de lo que tasas reales, dólar, petróleo, VIX e inflación explicarían con la historia desde 2006"
    elif brecha <= -8:
        donde = f"el futuro está cerca de {abs(brecha):.0f}% por debajo de esa referencia histórica"
    else:
        donde = f"el futuro está cerca de esa referencia histórica (brecha de {brecha:+.1f}%)"
    info5 = pronostico["por_horizonte"][5]
    info20 = pronostico["por_horizonte"][20]
    r2_5 = info5["metricas"]["sin_solapar"]["r2_media"]
    r2_20 = info20["metricas"]["sin_solapar"]["r2_media"]
    if (math.isfinite(r2_5) and r2_5 < 0) and (math.isfinite(r2_20) and r2_20 < 0):
        futuro = (
            "Para las próximas una a cuatro semanas el modelo no demuestra, fuera de muestra, "
            "que sepa más que el promedio histórico. No es un motivo para comprar ni para vender."
        )
    else:
        futuro = (
            "El detalle de 5 y 20 días está arriba, con su banda de error. "
            "Si la banda es mucho más ancha que el número esperado, la dirección no alcanza para una decisión."
        )
    lineas.append(parrafo(f"Con los datos de este informe, {donde}. {futuro}"))
    lineas.append("")
    lineas.append("7) Chequeo contra la nota del 5 de octubre de 2026")
    lineas.extend(_bloque_snapshot(df, datos))
    lineas.append("")
    lineas.append("8) De dónde salió cada serie")
    for nombre, info in datos["fuentes"].items():
        if nombre == "cpi":
            continue
        fecha = datos["fechas"].get(nombre, "")
        extra = f", último dato {fecha}" if fecha else ""
        lineas.append(f"  {nombre}: {info['etiqueta']}{extra}")
    if datos["fechas"].get("cpi_mes"):
        lineas.append(
            f"  cpi: {datos['fuentes'].get('cpi', {}).get('etiqueta', '')}, "
            f"mes {datos['fechas'].get('cpi_mes')}, publicación usada {datos['fechas'].get('cpi_fecha_publicacion')}"
        )
    if datos["avisos"]:
        lineas.append("")
        lineas.append("Avisos de la descarga:")
        for aviso in datos["avisos"]:
            lineas.append(parrafo(f"- {aviso}"))
    lineas.append("")
    lineas.append(
        "El IPC histórico es el dato revisado que publica hoy el BLS, no la primera cifra de cada mes. "
        "La regla del día 15 evita usar el mes antes de tiempo; no reconstruye esas revisiones."
    )
    lineas.append(sep)
    return "\n".join(lineas) + "\n"


def _estilo() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#D5DBE3",
            "axes.labelcolor": COLOR_TEXTO,
            "xtick.color": COLOR_TEXTO,
            "ytick.color": COLOR_TEXTO,
            "text.color": COLOR_TEXTO,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#E6EAF0",
            "grid.linewidth": 0.8,
            "font.size": 11,
            "axes.titlesize": 14,
            "font.family": "DejaVu Sans",
        }
    )


def _pie(fig, texto: str) -> None:
    fig.text(0.01, 0.012, texto, fontsize=8, color="#5C6B7A")
    fig.tight_layout(rect=(0, 0.04, 1, 1))


def guardar_graficas(df: pd.DataFrame, valor: dict, carpeta: Path) -> list[Path]:
    carpeta.mkdir(parents=True, exist_ok=True)
    _estilo()
    rutas = [
        _grafica_precio(df, valor, carpeta / "precio_vs_valor_justo.png"),
        _grafica_brecha(df, valor, carpeta / "brecha_valor_justo.png"),
        _grafica_coeficientes(valor, carpeta / "coeficientes_valor_justo.png"),
    ]
    return rutas


def _grafica_precio(df: pd.DataFrame, valor: dict, ruta: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    ax.plot(df.index, df["oro_usd"], color=COLOR_ORO, linewidth=1.4, label="Precio del futuro (USD/oz)")
    ax.plot(
        df.index,
        df["valor_justo_vf"],
        color=COLOR_MODELO,
        linewidth=1.3,
        label="Valor justo con datos anteriores",
    )
    ultimo = df["oro_usd"].dropna()
    ax.scatter([ultimo.index[-1]], [ultimo.iloc[-1]], color=COLOR_ORO, s=28, zorder=4)
    ax.annotate(
        f"{ultimo.iloc[-1]:,.0f}",
        xy=(ultimo.index[-1], ultimo.iloc[-1]),
        xytext=(-70, 12),
        textcoords="offset points",
        fontsize=10,
        color=COLOR_ORO,
    )
    ax.set_title("Precio del oro y valor justo del modelo")
    ax.set_ylabel("Dólares por onza troy")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlabel(
        f"Brecha de hoy con toda la historia: {valor['brecha_pct']:+.1f}%. "
        f"Con solo el pasado: {valor['brecha_vf_pct']:+.1f}%."
    )
    _pie(fig, "No es asesoría financiera. GC=F y regresión lineal. El valor justo no usa el futuro de cada día.")
    fig.savefig(ruta, dpi=140)
    plt.close(fig)
    return ruta


def _grafica_brecha(df: pd.DataFrame, valor: dict, ruta: Path) -> Path:
    brecha = (np.exp(df["residuo_vf"].astype(float)) - 1.0) * 100.0
    brecha = brecha.replace([np.inf, -np.inf], np.nan)
    valores = brecha.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    ax.axhline(0, color="#7B8794", linewidth=1)
    desviacion = float(np.nanstd(valores))
    if math.isfinite(desviacion):
        ax.axhline(desviacion, color="#C5CED8", linewidth=0.8, linestyle="--")
        ax.axhline(-desviacion, color="#C5CED8", linewidth=0.8, linestyle="--")
    ax.fill_between(
        df.index,
        np.where(np.isfinite(valores), valores, 0.0),
        0,
        where=np.isfinite(valores) & (valores >= 0),
        color=COLOR_CARO,
        alpha=0.25,
        interpolate=True,
    )
    ax.fill_between(
        df.index,
        np.where(np.isfinite(valores), valores, 0.0),
        0,
        where=np.isfinite(valores) & (valores < 0),
        color=COLOR_BARATO,
        alpha=0.25,
        interpolate=True,
    )
    ax.plot(df.index, brecha, color=COLOR_TEXTO, linewidth=1.0)
    ultima = brecha.dropna()
    if len(ultima):
        ax.scatter([ultima.index[-1]], [ultima.iloc[-1]], color=COLOR_TEXTO, s=28, zorder=4)
        ax.annotate(
            f"{ultima.iloc[-1]:+.1f}%",
            xy=(ultima.index[-1], ultima.iloc[-1]),
            xytext=(-64, 12),
            textcoords="offset points",
            fontsize=10,
        )
    ax.set_title("Brecha: cuánto se aleja el precio del valor justo")
    ax.set_ylabel("Porcentaje sobre el valor justo")
    ax.set_xlabel("Arriba de cero: el oro está más caro que el modelo. Las líneas punteadas son ±1 desviación de esta brecha.")
    _pie(fig, "No es asesoría financiera. Brecha calculada con coeficientes estimados solo con información anterior.")
    fig.savefig(ruta, dpi=140)
    plt.close(fig)
    return ruta


def _grafica_coeficientes(valor: dict, ruta: Path) -> Path:
    filas = list(valor["coeficientes"])
    filas = sorted(filas, key=lambda fila: abs(fila["efecto_std_pct"]))
    etiquetas = [fila["etiqueta"] for fila in filas]
    efectos = [fila["efecto_std_pct"] for fila in filas]
    colores = [COLOR_BARATO if efecto < 0 else COLOR_CARO for efecto in efectos]
    fig, ax = plt.subplots(figsize=(11.2, 5.6))
    ax.barh(etiquetas, efectos, color=colores, height=0.62)
    ax.axvline(0, color="#7B8794", linewidth=1)
    ax.set_title("Cuánto mueve al oro cada variable")
    ax.set_xlabel("Cambio asociado en el precio si la variable sube 1 desviación estándar (%)")
    _pie(fig, "No es asesoría financiera. Magnitud dentro de muestra. El signo negativo asocia esa subida con un oro más barato.")
    fig.savefig(ruta, dpi=140)
    plt.close(fig)
    return ruta


def guardar_tablas(valor: dict, pronostico: dict, carpeta: Path) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)
    coef = pd.DataFrame(valor["coeficientes"])
    coef.to_csv(carpeta / "coeficientes_valor_justo.csv", index=False)
    filas = []
    for horizonte, info in pronostico["por_horizonte"].items():
        for esquema, bloque in info["metricas"].items():
            fila = {"horizonte": horizonte, "esquema": esquema}
            fila.update(bloque)
            filas.append(fila)
    pd.DataFrame(filas).to_csv(carpeta / "metricas_pronostico.csv", index=False)
    df = pronostico["df"]
    columnas = [c for c in ("pred_5", "y_5", "pred_20", "y_20", "residuo_vf", "valor_justo_vf", "oro_usd") if c in df.columns]
    salida = df.loc[df["pred_5"].notna() | df["pred_20"].notna(), columnas].copy()
    salida.index.name = "fecha"
    salida.to_csv(carpeta / "predicciones_fuera_de_muestra.csv")
