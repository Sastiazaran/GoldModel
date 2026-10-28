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

ETIQUETA_CORTA = {
    "rendimiento_real_10": "Tasa real",
    "log_dolar": "Dólar",
    "log_brent": "Petróleo",
    "vix": "VIX",
    "cpi_anual": "Inflación",
    "gpr": "GPR",
    "log_gld_ton": "GLD",
    "compras_bc_ton": "Bancos centrales",
    "log_usd_mxn": "Tipo de cambio",
}


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
    if nombre == "log_gld_ton":
        return f"unas tenencias de GLD 1% más altas se asocian con {coef:+.2f}% en el oro"
    if nombre == "log_usd_mxn":
        return f"un tipo de cambio 1% más alto (más pesos por dólar) se asocia con {coef:+.2f}% en el oro en pesos"
    if nombre == "gpr":
        cambio = (math.exp(coef) - 1.0) * 100.0
        return f"+1 punto del índice GPR se asocia con {cambio:+.3f}% en el oro"
    if nombre == "compras_bc_ton":
        cambio = (math.exp(coef * 100.0) - 1.0) * 100.0
        return f"+100 toneladas netas en el trimestre se asocian con {cambio:+.2f}% en el oro"
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


def _texto_boosting(horizonte: int, info: dict) -> list[str]:
    """Misma ventana y mismo embargo que Ridge. Otro método, no otra muestra."""
    if "metricas_boosting" not in info:
        return []
    limpio = info["metricas_boosting"]["sin_solapar"]
    pred = float(info.get("pred_ultimo_boosting", float("nan")))
    esperado = (math.exp(pred) - 1.0) * 100.0 if math.isfinite(pred) else float("nan")
    ridge = info["metricas"]["sin_solapar"]
    lineas = [
        f"Gradient boosting, mismo horizonte de {horizonte} días, misma caminata y el mismo embargo.",
        f"Apunta a {pct(esperado, 2, signo=True)} (el número de hoy, no la nota histórica).",
        f"Sin solapar, n={limpio['n']}: "
        f"R² contra cero {r2_txt(limpio['r2_cero'])} "
        f"(Ridge {r2_txt(ridge['r2_cero'])}), "
        f"R² contra el promedio {r2_txt(limpio['r2_media'])} "
        f"(Ridge {r2_txt(ridge['r2_media'])}).",
        f"  RMSE boosting {pct(limpio['rmse'] * 100, 2)} | "
        f"Ridge {pct(ridge['rmse'] * 100, 2)} | "
        f"baseline cero {pct(limpio['rmse_cero'] * 100, 2)} | "
        f"baseline promedio {pct(limpio['rmse_media'] * 100, 2)}.",
        f"  Acierto boosting {pct(limpio['acierto'] * 100, 1)} | "
        f"Ridge {pct(ridge['acierto'] * 100, 1)} | "
        f"siempre alza {pct(limpio['acierto_siempre_alza'] * 100, 1)}.",
    ]
    if info.get("importancias"):
        top = info["importancias"][:3]
        texto = ", ".join(f"{fila['etiqueta']} ({fila['peso']:.2f})" for fila in top)
        lineas.append(f"Mayor peso del último bosque (importancia, suma 1): {texto}.")
    gana_cero = math.isfinite(limpio["r2_cero"]) and limpio["r2_cero"] > 0
    gana_media = math.isfinite(limpio["r2_media"]) and limpio["r2_media"] > 0
    if not gana_cero and not gana_media:
        lineas.append(
            "Tampoco le gana a los dos baselines. Cambiar de recta a árboles no sacó una señal."
        )
    elif gana_cero and not gana_media:
        lineas.append(
            "Le gana al cero y no al promedio: alcanza para decir 'el oro suele subir', "
            "no para decir cuándo."
        )
    return lineas


def _seccion_variables_nuevas(valor: dict) -> list[str]:
    ext = valor.get("extendido") or {}
    lineas = ["", "4) Variables nuevas: bancos centrales, GLD y riesgo geopolítico"]
    lineas.append(
        "Al modelo de la sección 3 se le agregan, si la serie llegó, tres columnas: "
        "compras netas de oro de bancos centrales (toneladas en el trimestre, World Gold Council, "
        "usables a fin del mes siguiente al cierre del trimestre), "
        "el logaritmo de las toneladas en GLD (el dato de una sesión entra al día hábil siguiente) "
        "y el índice GPR de Caldara e Iacoviello (el diario se usa el mismo día). "
        "La brecha de esta sección también camina hacia adelante: el día t no entra a sus propios coeficientes."
    )
    if not ext.get("disponible"):
        lineas.append(ext.get("nota") or "No se pudo armar el modelo ampliado.")
        return lineas
    lineas.append("Entraron: " + ", ".join(ext.get("variables") or []) + f". n={ext['n']}.")
    lineas.append(
        f"Valor justo ampliado, con toda la historia: {num(ext['valor_justo'])} USD/oz. "
        f"Brecha: {pct(ext['brecha_pct'], 1, signo=True)}."
    )
    lineas.append(
        f"Valor justo ampliado, solo con el pasado: {num(ext['valor_justo_vf'])} USD/oz. "
        f"Brecha: {pct(ext['brecha_vf_pct'], 1, signo=True)}."
    )
    base = ext.get("brecha_base_vf_pct")
    cierre = ext.get("cierre_abs_pp")
    if base is not None and cierre is not None and math.isfinite(base) and math.isfinite(cierre):
        lineas.append(
            f"La brecha del modelo base, el mismo día y también solo con el pasado, era {pct(base, 1, signo=True)}. "
            f"En valor absoluto, las variables nuevas la mueven {cierre:+.1f} puntos porcentuales "
            "(positivo quiere decir que la brecha se acerca a cero)."
        )
        if cierre > 1 and base * ext["brecha_vf_pct"] < 0:
            lineas.append(
                "La distancia a cero baja, pero el signo cambia: el modelo base decía que el oro estaba "
                "caro y el ampliado dice que está barato. No es un cierre limpio. Es otra recta."
            )
            lineas.append(
                "Ojo con GLD: las toneladas del fondo suben en los mismos años en que sube el precio, "
                "porque entran inversionistas. Un coeficiente grande ahí puede estar contando esa subida, "
                "no una causa que empuje el oro desde afuera. El cambio de signo no se lea como "
                "'ahora está barato y hay que comprar'."
            )
        elif cierre > 1:
            lineas.append("Sí cierran parte de la brecha, y el signo se queda igual. No la hacen desaparecer.")
        elif cierre < -1:
            lineas.append(
                "No la cierran: con estas columnas el precio queda aún más lejos del valor justo. "
                "Eso también es un resultado."
            )
        else:
            lineas.append("Casi no mueven la brecha: el hueco no era, sobre todo, estas tres series.")
    lineas.append(
        f"R² dentro de muestra del modelo ampliado: {ext['r2']:.3f}. "
        f"El del modelo base está en la sección 3. Ninguno de los dos es la nota del pronóstico."
    )
    lineas.append("Coeficientes del modelo ampliado (Newey-West, 21 rezagos):")
    lineas.append(f"{'Variable':<28}{'Coef':>10}{'EE':>10}{'t':>8}   Lectura")
    for fila_coef in ext.get("coeficientes") or []:
        lineas.append(
            f"{fila_coef['etiqueta']:<28}"
            f"{fila_coef['coef']:>10.3f}"
            f"{fila_coef['ee']:>10.3f}"
            f"{fila_coef['t']:>8.2f}   "
            f"{lectura_coeficiente(fila_coef)}"
        )
    return lineas


def _seccion_movil(valor: dict) -> list[str]:
    movil = valor.get("movil") or {}
    lineas = ["", "5) Coeficientes que cambian (ventanas de 2 y 3 años)"]
    espec = movil.get("especificacion") or []
    lineas.append(
        "Cada día, la recta se estima solo con las últimas 504 sesiones (cerca de 2 años) "
        "o las últimas 756 (cerca de 3). El día de hoy no entra. Se reestima más o menos cada 21 sesiones. "
        "Especificación: " + ", ".join(espec) + "."
    )
    lineas.append(
        "La gráfica coeficientes_moviles.png muestra el camino de cada coeficiente. "
        "Si una línea cruza el cero, el signo que salía en la muestra completa no era una ley."
    )
    for clave, titulo in (("2a", "2 años"), ("3a", "3 años")):
        bloque = (movil.get("ventanas") or {}).get(clave)
        if not bloque or bloque.get("fecha") is None:
            lineas.append(f"Ventana de {titulo}: todavía no hay suficientes sesiones.")
            continue
        lineas.append(
            f"Ventana de {titulo}, último día {fecha_larga(bloque['fecha'])}: "
            f"valor justo {num(bloque['valor_justo'])} USD/oz, "
            f"brecha {pct(bloque['brecha_pct'], 1, signo=True)}."
        )
        ultimo = bloque.get("ultimo") or {}
        if "rendimiento_real_10" in ultimo or "log_dolar" in ultimo:
            lineas.append(
                f"  Últimos coeficientes: tasa real {ultimo.get('rendimiento_real_10', float('nan')):+.3f}, "
                f"dólar {ultimo.get('log_dolar', float('nan')):+.3f}."
            )
    return lineas


def _seccion_cointegracion(valor: dict) -> list[str]:
    coint = valor.get("cointegracion") or {}
    lineas = ["", "6) Cointegración y vida media de la brecha"]
    lineas.append(
        "Cointegración pregunta si el oro y sus causas se mueven juntos en el largo plazo, "
        "aunque cada uno por su lado parezca no volver a un promedio. "
        "Se prueba en datos semanales (viernes), para no contar cinco días seguidos como cinco pruebas. "
        "La prueba ADF de la sección 3 no es esta prueba: sus valores críticos no sirven cuando el residuo "
        "sale de una regresión estimada."
    )
    variables = coint.get("variables") or []
    if coint.get("n"):
        lineas.append(
            f"Semanas usadas: {coint['n']}. Variables: {', '.join(variables)}."
        )
    adf = coint.get("adf") or {}
    if adf:
        partes = []
        for nombre, pvalor in adf.items():
            if pvalor is None:
                partes.append(f"{nombre} sin ADF")
            else:
                partes.append(f"{nombre} p={pvalor:.3f}")
        lineas.append("ADF de cada serie en niveles (p chico: se rechaza raíz unitaria): " + "; ".join(partes) + ".")
    if coint.get("eg_p") is None:
        lineas.append("Engle-Granger no quedó calculado.")
    else:
        decision = "rechaza" if coint["eg_p"] < 0.05 else "no rechaza"
        lineas.append(
            f"Engle-Granger: estadístico {coint['eg_stat']:.2f}, p={coint['eg_p']:.3f}. "
            f"{decision.capitalize()} 'no hay relación', al 5%."
        )
    if coint.get("johansen_trace") is None:
        lineas.append("Johansen no quedó calculado.")
    else:
        decision = "sí" if coint.get("johansen_rechaza_r0") else "no"
        lineas.append(
            f"Johansen, traza para cero relaciones: {coint['johansen_trace']:.2f}. "
            f"Valor crítico al 5%: {coint['johansen_crit_5']:.2f}. "
            f"{decision.capitalize()} se rechaza que no haya ninguna relación."
        )
    if coint.get("hay_relacion"):
        cuales = " y ".join(coint.get("prueba_que_rechaza") or [])
        lineas.append(f"Hay señal de relación según {cuales}.")
        if coint.get("alpha_ecm") is not None:
            lineas.append(
                f"Modelo de corrección de errores, en semanas: el rezago de la brecha entra con "
                f"coeficiente {coint['alpha_ecm']:+.3f} "
                f"(EE {coint['ee_alpha']:.3f}, t={coint['t_alpha']:.2f}). "
                "Negativo y entre -1 y 0 quiere decir que una brecha positiva empuja el precio hacia abajo."
            )
        if coint.get("vida_media_semanas") is not None:
            lineas.append(
                f"Vida media de la brecha: {coint['vida_media_semanas']:.1f} semanas "
                f"(cerca de {coint['vida_media_dias']:.0f} días de mercado). "
                "Es el tiempo para que, si el modelo de corrección fuera cierto, la brecha baje a la mitad. "
                "No es una fecha en la que el oro vaya a caer."
            )
        else:
            lineas.append(
                "No hay vida media: el coeficiente de corrección no encoge la brecha hacia la mitad."
            )
    else:
        lineas.append(
            "No hay relación de largo plazo según estas dos pruebas, al 5%. "
            "No se reporta un modelo de corrección ni una vida media de equilibrio."
        )
    if coint.get("ar1_vida_media_dias") is not None:
        lineas.append(
            f"Aparte, y sin llamarlo equilibrio: un AR(1) sobre la brecha diaria del modelo base "
            f"tiene rho {coint['ar1_rho']:.3f} y una vida media descriptiva de "
            f"{coint['ar1_vida_media_dias']:.0f} días de mercado. "
            "Eso solo describe qué tan lento se mueve la brecha. No prueba que vaya a volver."
        )
    if coint.get("nota"):
        lineas.append(parrafo(coint["nota"]))
    return lineas


def _seccion_mxn(mxn: dict | None) -> list[str]:
    lineas = ["", "9) Oro en pesos por gramo"]
    if not mxn:
        lineas.append("No se estimó el modelo en pesos.")
        return lineas
    valor = mxn["valor"]
    lineas.append(
        "Esta regresión explica el logaritmo del oro en pesos por gramo. "
        "Lleva el tipo de cambio (pesos por dólar) en lugar del índice dólar, "
        "para no contar el dólar dos veces. No es el valor justo en dólares convertido al tipo de cambio de hoy: "
        "esa conversión dejaría el mismo porcentaje de brecha."
    )
    lineas.append(
        f"Precio: {num(valor['precio'])} pesos por gramo. "
        f"Valor justo con toda la historia: {num(valor['valor_justo'])}. "
        f"Brecha: {pct(valor['brecha_pct'], 1, signo=True)}."
    )
    lineas.append(
        f"Valor justo solo con el pasado: {num(valor['valor_justo_vf'])} pesos por gramo. "
        f"Brecha: {pct(valor['brecha_vf_pct'], 1, signo=True)}. "
        f"R² dentro de muestra: {valor['r2']:.3f}. n={valor['n']}."
    )
    if valor.get("usadas_extra"):
        lineas.append("También entraron: " + ", ".join(valor["usadas_extra"]) + ".")
    lineas.append(
        "Pronóstico del retorno en pesos, a 5 y 20 días, con la misma caminata, el mismo embargo "
        "y los dos baselines (cero y promedio). Momentum del propio precio en pesos, cambio del peso, "
        "tasa real, petróleo, VIX, la brecha en pesos y, si llegaron, el COT y la tasa a 2 años."
    )
    for horizonte in (5, 20):
        info = mxn["por_horizonte"][horizonte]
        lineas.append("")
        lineas.append(f"Pesos, {horizonte} días.")
        lineas.extend(_texto_pronostico(horizonte, info))
        lineas.extend(_texto_boosting(horizonte, info))
    return lineas


def _seccion_papa(valor: dict, pronostico: dict, mxn: dict | None) -> list[str]:
    """Cierre en lenguaje llano. Usa los números de esta corrida, no un discurso fijo."""
    ext = valor.get("extendido") or {}
    coint = valor.get("cointegracion") or {}
    brecha = valor.get("brecha_vf_pct", valor.get("brecha_pct"))
    lineas = ["", "10) Para tu papá"]
    frases = []
    frases.append(
        f"El oro, en el futuro de Nueva York, está en {num(valor['precio'])} dólares la onza."
    )
    if mxn and mxn.get("valor"):
        frases.append(
            f"En pesos, el metal en el mercado mayorista sale en {num(mxn['valor']['precio'])} pesos el gramo. "
            "Una joyería cobra más que eso: es el precio del metal, no el de un anillo."
        )
        frases.append(
            f"La cuenta hecha directo en pesos, con su propia fórmula, lo deja "
            f"{pct(mxn['valor']['brecha_vf_pct'], 0, signo=True)} "
            "respecto de lo que tasas, tipo de cambio, petróleo e inflación explicarían con el pasado."
        )
    if math.isfinite(brecha):
        if brecha > 8:
            donde = f"cerca de {brecha:.0f}% por encima"
        elif brecha < -8:
            donde = f"cerca de {abs(brecha):.0f}% por debajo"
        else:
            donde = f"cerca (brecha de {brecha:+.0f}%)"
        frases.append(
            f"La cuenta en dólares, la de siempre (tasas reales, dólar, petróleo, nerviosismo e inflación), "
            f"dice que el precio está {donde} de esa historia. "
            "Esa brecha usa solo datos anteriores a hoy."
        )
    if ext.get("disponible") and ext.get("cierre_abs_pp") is not None and math.isfinite(ext["cierre_abs_pp"]):
        cierre = ext["cierre_abs_pp"]
        if cierre > 1:
            que = f"la acercan a cero en {cierre:.0f} puntos"
        elif cierre < -1:
            que = f"no la cierran: la alejan {abs(cierre):.0f} puntos más"
        else:
            que = "casi no la mueven"
        if math.isfinite(brecha) and brecha * ext["brecha_vf_pct"] < 0 and cierre > 1:
            que = (
                f"acercan el número a cero en {cierre:.0f} puntos, pero le cambian el signo: "
                "pasa de 'caro' a 'barato'. Parte de eso es el fondo GLD, que guarda más oro "
                "justo cuando el precio ya subió. No lo leas como una oferta"
            )
        frases.append(
            f"Le agregué lo que compran los bancos centrales, cuánto oro guarda el fondo GLD "
            f"y un índice de noticias de riesgo geopolítico. Esas tres {que}. "
            f"La brecha ampliada, también solo con el pasado, queda en {pct(ext['brecha_vf_pct'], 0, signo=True)}."
        )
    frases.append(
        "Los pesos de esa fórmula no están quietos. En ventanas de dos y de tres años, "
        "el efecto de la tasa y del dólar cambia. No es una regla para siempre. "
        "La gráfica de los coeficientes móviles es esa historia."
    )
    if coint.get("hay_relacion") and coint.get("vida_media_semanas") is not None:
        frases.append(
            f"Una prueba de largo plazo (cointegración) sí encuentra que el precio y esas causas se mueven juntos. "
            f"Si eso fuera cierto, la brecha tardaría cerca de {coint['vida_media_semanas']:.0f} semanas "
            "en bajar a la mitad. No quiere decir que el mes que viene baje."
        )
    elif coint.get("hay_relacion"):
        frases.append(
            "La prueba de largo plazo encuentra alguna relación, pero el ajuste no da una vida media clara "
            "de regreso a la mitad."
        )
    else:
        frases.append(
            "La prueba de largo plazo (si el oro y esas causas vuelven a caminar juntos) "
            "no encuentra esa relación con la confianza que pide el 5%. "
            "Por eso no te doy una fecha en la que 'tiene que' regresar."
        )
    info5 = pronostico["por_horizonte"][5]
    info20 = pronostico["por_horizonte"][20]
    r2s = [
        info5["metricas"]["sin_solapar"]["r2_media"],
        info20["metricas"]["sin_solapar"]["r2_media"],
        info5.get("metricas_boosting", {}).get("sin_solapar", {}).get("r2_media", float("nan")),
        info20.get("metricas_boosting", {}).get("sin_solapar", {}).get("r2_media", float("nan")),
    ]
    if all(math.isfinite(valor_r2) and valor_r2 < 0 for valor_r2 in r2s):
        frases.append(
            "Para las próximas una y cuatro semanas comparé dos métodos, una recta penalizada y un bosque de árboles, "
            "con el mismo examen: solo datos ya conocidos, y contra 'el precio no cambia' y contra 'el oro rinde lo de siempre'. "
            "Ninguno le gana al promedio. No es un motivo para comprar ni para vender."
        )
    else:
        frases.append(
            "El detalle de una y de cuatro semanas está arriba, método por método, con su error. "
            "Si el error es más ancho que el número esperado, la flecha no alcanza para una decisión."
        )
    frases.append("Esto no es un consejo. Es una cuenta con datos públicos, y a unos días el oro es casi ruido.")
    lineas.append(parrafo(" ".join(frases)))
    return lineas


def _seccion_estudio(estudio: dict | None) -> list[str]:
    """Cuenta el modelo nuevo con los números de esta corrida. No rellena huecos."""
    lineas = ["", "13) Modelo nuevo: evaluación, regímenes y dirección"]
    if not estudio:
        lineas.append("No se corrió el modelo nuevo.")
        return lineas
    lineas.append(parrafo(estudio.get("esquema") or ""))
    valor = estudio.get("valor") or {}
    if valor:
        lineas.append(
            f"Valor justo de Kalman: {num(valor.get('valor_kalman'))} USD/oz, "
            f"brecha {pct(valor.get('brecha_kalman_pct'), 1, signo=True)}. "
            f"RMSE del log, una observación cada 20 sesiones: "
            f"base {num(valor.get('rmse_log_base'), 4)}, "
            f"Kalman {num(valor.get('rmse_log_kalman'), 4)}, "
            f"Markov {num(valor.get('rmse_log_markov'), 4)}."
        )
        if valor.get("promovido"):
            lineas.append(f"El valor justo titular pasa a {valor.get('titular')}: le bajó el RMSE a la base.")
        else:
            lineas.append("El valor justo titular se queda en el modelo base: el nuevo no le bajó el RMSE.")
        if valor.get("markov_convergio") is False:
            lineas.append("El Markov no llegó a converger. El número es el último intento, no una solución cerrada.")
        lineas.append("Variables del valor nuevo: " + ", ".join(valor.get("variables") or []) + ".")
    comp = estudio.get("comparacion") or {}
    if comp.get("criterio"):
        lineas.append(parrafo(comp["criterio"]))
    for fila in comp.get("filas") or []:
        lineas.append(
            f"  {fila['modelo']} a {fila['horizonte']} días, n={fila['n']}: "
            f"RMSE {pct((fila.get('rmse_mismas_fechas') or float('nan')) * 100, 2)} "
            f"en las mismas fechas, R² contra cero {r2_txt(fila.get('r2_cero'))}, "
            f"acierto {pct((fila.get('acierto') or float('nan')) * 100, 1)}. "
            f"{'Titular.' if fila.get('es_titular') else 'Comparación.'}"
        )
    for horizonte, bloque in (estudio.get("clasificacion") or {}).items():
        elegido = (bloque.get("metricas") or {}).get("elegido") or {}
        lineas.append(
            f"Dirección a {horizonte} días ({bloque.get('modelo') or 'sin modelo'}): "
            f"P(alza) {pct((bloque.get('prob_up') or float('nan')) * 100, 1)}. "
            f"Balanced accuracy {pct((elegido.get('balanced_accuracy') or float('nan')) * 100, 1)} "
            f"frente a siempre alza {pct((elegido.get('balanced_accuracy_siempre_alza') or float('nan')) * 100, 1)}. "
            f"ROC-AUC {num(elegido.get('roc_auc'), 3)}. "
            f"Brier {num(elegido.get('brier'), 3)} "
            f"(siempre alza {num(elegido.get('brier_siempre_alza'), 3)}, "
            f"tasa base {num(elegido.get('brier_tasa_base'), 3)})."
        )
    for horizonte, bloque in (estudio.get("intervalos") or {}).items():
        cob = (bloque.get("cobertura") or {}).get("sin_solapar") or {}
        lineas.append(
            f"Intervalo {int((bloque.get('nivel') or 0) * 100)}% a {horizonte} días "
            f"({bloque.get('metodo')}): cobertura fuera de muestra "
            f"{pct((cob.get('cobertura') or float('nan')) * 100, 1)} con n={cob.get('n')}."
        )
        garch = bloque.get("garch") or {}
        cob_g = (garch.get("cobertura") or {}).get("sin_solapar") or {}
        if cob_g.get("cobertura") is not None:
            lineas.append(
                f"  GARCH(1,1), misma cobertura nominal: {pct(cob_g['cobertura'] * 100, 1)} con n={cob_g.get('n')}."
            )
    regimen = estudio.get("regimen") or {}
    if regimen.get("disponible"):
        lineas.append(
            f"Régimen de hoy, filtro HMM: {regimen.get('nombre')} "
            f"(estado {regimen.get('id')})."
        )
    dca = estudio.get("dca") or {}
    if dca.get("disponible"):
        lineas.append(
            f"Compra en partes contra DCA plano, misma caja: "
            f"{dca.get('onzas_partes'):.4f} onzas contra {dca.get('onzas_dca'):.4f} "
            f"({pct(dca.get('ventaja_onzas_pct'), 2, signo=True)}). "
            f"Adelantos {dca.get('adelantos')}, aplazos {dca.get('aplazos')}."
        )
    elif dca.get("nota"):
        lineas.append(dca["nota"])
    for aviso in estudio.get("avisos") or []:
        lineas.append(parrafo(f"- {aviso}"))
    lineas.append("Nada de esto es una orden de comprar o vender.")
    return lineas


def construir_informe(
    datos: dict,
    valor: dict,
    pronostico: dict,
    mxn: dict | None = None,
    estudio: dict | None = None,
) -> str:
    df = pronostico["df"]
    fecha_modelo = pd.Timestamp(valor["fecha"])
    fila = df.loc[fecha_modelo]
    ultima_oro = df["oro_usd"].dropna().index[-1]
    precio_ultimo = float(df.loc[ultima_oro, "oro_usd"])
    tipo_cambio = float(fila["usd_mxn"]) if pd.notna(fila.get("usd_mxn")) else float("nan")
    if pd.notna(df.loc[ultima_oro].get("usd_mxn")):
        mxn_precio = float(df.loc[ultima_oro, "usd_mxn"])
        fecha_mxn = ultima_oro
    else:
        mxn_precio = tipo_cambio
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
            "Una brecha ancha también puede querer decir que al modelo de esta sección le faltan piezas. "
            "Las compras de bancos centrales, el oro en GLD y el riesgo geopolítico se agregan en la sección 4, "
            "y ahí se mide si cierran esta brecha."
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
        "No mide el pronóstico. El pronóstico está en la sección 7 y es fuera de muestra."
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
    lineas.extend(_seccion_variables_nuevas(valor))
    lineas.extend(_seccion_movil(valor))
    lineas.extend(_seccion_cointegracion(valor))
    lineas.append("")
    lineas.append("7) Pronóstico a 5 y 20 días, fuera de muestra")
    lineas.append(
        "Cada predicción usa solo información ya conocida ese día. El entrenamiento deja fuera "
        "las últimas observaciones cuyo retorno futuro todavía no había cerrado (embargo del horizonte). "
        "Se reentrena más o menos cada 21 sesiones. Ridge estandariza las columnas con la media y la "
        "desviación del propio entrenamiento. En la misma caminata, con las mismas filas y el mismo embargo, "
        "se ajusta también un gradient boosting (árboles chicos). "
        "Si llegaron, entran además el neto no comercial del COT (como fracción del interés abierto, "
        "usable el viernes en que la CFTC publica) y la tasa nominal a 2 años, como aproximación a lo que "
        "el mercado espera de la Fed. El momentum de 5, 20 y 60 días ya estaba."
    )
    lineas.append(
        "Baseline cero: predecir que el retorno logarítmico es 0 (caminata aleatoria, sin deriva). "
        "Baseline promedio: predecir el retorno medio del entrenamiento de esa ventana."
    )
    if pronostico.get("columnas"):
        lineas.append("Columnas de esta caminata: " + ", ".join(pronostico["columnas"]) + ".")
    for horizonte in (5, 20):
        lineas.append("")
        lineas.extend(_texto_pronostico(horizonte, pronostico["por_horizonte"][horizonte]))
        lineas.extend(_texto_boosting(horizonte, pronostico["por_horizonte"][horizonte]))
    lineas.append("")
    lineas.append("8) Qué variables pesan más en el nivel del oro")
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
    lineas.extend(_seccion_mxn(mxn))
    lineas.extend(_seccion_papa(valor, pronostico, mxn))
    lineas.extend(_seccion_estudio(estudio))
    lineas.append("")
    lineas.append("11) Chequeo contra la nota del 5 de octubre de 2026")
    lineas.extend(_bloque_snapshot(df, datos))
    lineas.append("")
    lineas.append("12) De dónde salió cada serie")
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


def guardar_graficas(
    df: pd.DataFrame,
    valor: dict,
    carpeta: Path,
    mxn: dict | None = None,
) -> list[Path]:
    carpeta.mkdir(parents=True, exist_ok=True)
    _estilo()
    rutas = [
        _grafica_precio(df, valor, carpeta / "precio_vs_valor_justo.png"),
        _grafica_brecha(df, valor, carpeta / "brecha_valor_justo.png"),
        _grafica_coeficientes(valor, carpeta / "coeficientes_valor_justo.png"),
        _grafica_coeficientes_moviles(valor, carpeta / "coeficientes_moviles.png"),
        _grafica_mxn(df, mxn, carpeta / "oro_mxn_vs_valor_justo.png"),
    ]
    return [ruta for ruta in rutas if ruta is not None]


def _grafica_precio(df: pd.DataFrame, valor: dict, ruta: Path) -> Path:
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    ax.plot(df.index, df["oro_usd"], color=COLOR_ORO, linewidth=1.4, label="Precio del futuro (USD/oz)")
    ax.plot(
        df.index,
        df["valor_justo_vf"],
        color=COLOR_MODELO,
        linewidth=1.3,
        label="Valor justo base, solo el pasado",
    )
    if "valor_ext" in df.columns:
        ax.plot(
            df.index,
            df["valor_ext"],
            color="#5C4B8A",
            linewidth=1.15,
            label="Con bancos centrales, GLD y GPR",
        )
    if "valor_movil_3a" in df.columns:
        ax.plot(
            df.index,
            df["valor_movil_3a"],
            color="#8A5A2B",
            linewidth=1.05,
            linestyle="--",
            label="Pesos móviles, ventana de 3 años",
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
    ax.plot(df.index, brecha, color=COLOR_TEXTO, linewidth=1.0, label="Modelo base")
    if "residuo_ext" in df.columns:
        brecha_ext = (np.exp(df["residuo_ext"].astype(float)) - 1.0) * 100.0
        ax.plot(df.index, brecha_ext, color="#5C4B8A", linewidth=1.05, label="Con las variables nuevas")
        ax.legend(frameon=False, loc="upper left")
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


def _grafica_coeficientes_moviles(valor: dict, ruta: Path) -> Path | None:
    movil = valor.get("movil") or {}
    bloque = (movil.get("ventanas") or {}).get("3a")
    if not bloque or bloque.get("coeficientes") is None:
        return None
    coefs = bloque["coeficientes"]
    nombres = [nombre for nombre in coefs.columns if nombre != "constante" and coefs[nombre].notna().any()]
    if not nombres:
        return None
    fig, ejes = plt.subplots(len(nombres), 1, figsize=(11.2, 2.1 * len(nombres)), sharex=True)
    if len(nombres) == 1:
        ejes = [ejes]
    ventana_2 = (movil.get("ventanas") or {}).get("2a", {}).get("coeficientes")
    for ax, nombre in zip(ejes, nombres):
        ax.axhline(0, color="#7B8794", linewidth=0.8)
        ax.plot(coefs.index, coefs[nombre], color=COLOR_MODELO, linewidth=1.1, label="3 años")
        if ventana_2 is not None and nombre in ventana_2.columns:
            ax.plot(ventana_2.index, ventana_2[nombre], color=COLOR_ORO, linewidth=1.0, label="2 años")
        ax.set_ylabel(ETIQUETA_CORTA.get(nombre, nombre))
        ax.legend(frameon=False, loc="upper left", fontsize=8)
    ejes[0].set_title("Coeficientes en ventanas móviles, estimados solo con el pasado")
    ejes[-1].set_xlabel("El día de la estimación no entra en la ventana. Se reestima más o menos cada mes de mercado.")
    _pie(fig, "No es asesoría financiera. Una línea que cruza el cero cambió de signo: el promedio de toda la historia no es una ley.")
    fig.savefig(ruta, dpi=140)
    plt.close(fig)
    return ruta


def _grafica_mxn(df: pd.DataFrame, mxn: dict | None, ruta: Path) -> Path | None:
    if mxn is None or "oro_mxn_g" not in df.columns or "valor_mxn" not in df.columns:
        return None
    fig, ax = plt.subplots(figsize=(11.2, 6.2))
    ax.plot(df.index, df["oro_mxn_g"], color=COLOR_ORO, linewidth=1.4, label="Oro, pesos por gramo")
    ax.plot(
        df.index,
        df["valor_mxn"],
        color=COLOR_MODELO,
        linewidth=1.3,
        label="Valor justo en pesos, solo el pasado",
    )
    ultimo = df["oro_mxn_g"].dropna()
    if len(ultimo):
        ax.scatter([ultimo.index[-1]], [ultimo.iloc[-1]], color=COLOR_ORO, s=28, zorder=4)
        ax.annotate(
            f"{ultimo.iloc[-1]:,.0f}",
            xy=(ultimo.index[-1], ultimo.iloc[-1]),
            xytext=(-70, 12),
            textcoords="offset points",
            fontsize=10,
            color=COLOR_ORO,
        )
    brecha = mxn["valor"].get("brecha_vf_pct", float("nan"))
    ax.set_title("Oro en pesos por gramo y su propio valor justo")
    ax.set_ylabel("Pesos por gramo")
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlabel(f"Brecha con coeficientes solo del pasado: {brecha:+.1f}%. No es el precio de una joyería.")
    _pie(fig, "No es asesoría financiera. Regresión propia del precio en pesos, no el valor en dólares convertido al tipo de cambio.")
    fig.savefig(ruta, dpi=140)
    plt.close(fig)
    return ruta


def _filas_metricas(por_horizonte: dict, mercado: str, modelo: str, clave: str) -> list[dict]:
    filas = []
    for horizonte, info in por_horizonte.items():
        metricas = info.get(clave) or {}
        for esquema, bloque in metricas.items():
            fila = {"mercado": mercado, "modelo": modelo, "horizonte": horizonte, "esquema": esquema}
            fila.update(bloque)
            filas.append(fila)
    return filas


def guardar_tablas(valor: dict, pronostico: dict, carpeta: Path, mxn: dict | None = None) -> None:
    carpeta.mkdir(parents=True, exist_ok=True)
    coef = pd.DataFrame(valor["coeficientes"])
    coef.to_csv(carpeta / "coeficientes_valor_justo.csv", index=False)
    ext = valor.get("extendido") or {}
    if ext.get("coeficientes"):
        pd.DataFrame(ext["coeficientes"]).to_csv(carpeta / "coeficientes_valor_justo_ampliado.csv", index=False)
    movil = (valor.get("movil") or {}).get("ventanas") or {}
    if "3a" in movil:
        camino = movil["3a"]["coeficientes"].dropna(how="all")
        camino.to_csv(carpeta / "coeficientes_moviles_3a.csv")
    filas = _filas_metricas(pronostico["por_horizonte"], "usd", "ridge", "metricas")
    filas.extend(_filas_metricas(pronostico["por_horizonte"], "usd", "boosting", "metricas_boosting"))
    if mxn is not None:
        filas.extend(_filas_metricas(mxn["por_horizonte"], "mxn", "ridge", "metricas"))
        filas.extend(_filas_metricas(mxn["por_horizonte"], "mxn", "boosting", "metricas_boosting"))
    pd.DataFrame(filas).to_csv(carpeta / "metricas_pronostico.csv", index=False)
    df = pronostico["df"]
    columnas = [
        c
        for c in (
            "pred_5",
            "pred_gb_5",
            "y_5",
            "pred_20",
            "pred_gb_20",
            "y_20",
            "residuo_vf",
            "valor_justo_vf",
            "valor_ext",
            "valor_movil_3a",
            "oro_usd",
            "oro_mxn_g",
            "valor_mxn",
            "pred_mxn_5",
            "pred_mxn_20",
        )
        if c in df.columns
    ]
    mascara = df["pred_5"].notna() | df["pred_20"].notna()
    salida = df.loc[mascara, columnas].copy()
    salida.index.name = "fecha"
    salida.to_csv(carpeta / "predicciones_fuera_de_muestra.csv")
