"""Arma el JSON que lee el tablero.

Uso, desde la raíz del repositorio:

    python scripts/export_dashboard_data.py
    python scripts/export_dashboard_data.py --solo-reportes
    python scripts/export_dashboard_data.py --sin-descarga

`--solo-reportes` solo junta `data/reports/YYYY-MM-DD.json` en
`dashboard/data/reportes.json`. No toca el modelo.

Sin esa bandera, el script baja el panel (o lee `data/panel_diario.csv`
si se pide `--sin-descarga`), corre el valor justo y el pronóstico
walk-forward de 5 y 20 días, y escribe `dashboard/data/snapshot.json`.

No inventa precios. Un hueco del reporte se queda en null. Un pronóstico
cuyo horizonte todavía no cierra no entra al acierto en vivo.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import numpy as np
import pandas as pd

from modelos import (
    _tomar_no_solapadas,
    metricas_pronostico,
    onza_a_mxn_por_gramo,
)
CARPETA_REPORTES = RAIZ / "data" / "reports"
CARPETA_TABLERO = RAIZ / "dashboard" / "data"
RUTA_PANEL = RAIZ / "data" / "panel_diario.csv"
RUTA_ESQUEMA = CARPETA_REPORTES / "schema.json"

# La gráfica del tablero no necesita los 20 años día por día.
DESDE_GRAFICA = "2020-01-01"
PATRON_FECHA = re.compile(r"^\d{4}-\d{2}-\d{2}$")
CAMPOS_REPORTE = (
    "fecha",
    "sesion",
    "precio_usd_oz",
    "precio_aproximado",
    "precio_fuente",
    "precio_url",
    "cierre_previo_usd_oz",
    "cierre_previo_aproximado",
    "cambio_usd",
    "cambio_pct",
    "cambio_texto",
    "usd_mxn",
    "usd_mxn_tipo",
    "usd_mxn_fecha",
    "usd_mxn_fuente",
    "usd_mxn_url",
    "mxn_por_gramo",
    "mxn_por_gramo_aproximado",
    "drivers",
    "lectura",
    "notion_url",
    "notas",
)
CAMPOS_DRIVER = ("titulo", "detalle", "fuente", "url")
SESIONES = {"con_registro", "sin_registro"}
# En un día sin registro estos campos no pueden traer un número inventado.
CAMPOS_VACIOS_SIN_REGISTRO = (
    "precio_usd_oz",
    "precio_aproximado",
    "precio_fuente",
    "precio_url",
    "cierre_previo_usd_oz",
    "cierre_previo_aproximado",
    "cambio_usd",
    "cambio_pct",
    "cambio_texto",
    "usd_mxn",
    "usd_mxn_tipo",
    "usd_mxn_fecha",
    "usd_mxn_fuente",
    "usd_mxn_url",
    "mxn_por_gramo",
    "mxn_por_gramo_aproximado",
    "lectura",
    "notion_url",
)


def _num(valor):
    if valor is None or isinstance(valor, bool):
        return None
    try:
        numero = float(valor)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(numero):
        return None
    return numero


def _texto(valor) -> str | None:
    if valor is None:
        return None
    if not isinstance(valor, str):
        return None
    limpio = valor.strip()
    return limpio or None


def _es_url(valor) -> bool:
    return isinstance(valor, str) and valor.startswith(("https://", "http://"))


def validar_reporte(reporte: dict) -> list[str]:
    """Devuelve la lista de errores. Vacía si el reporte cumple el esquema."""
    errores: list[str] = []
    if not isinstance(reporte, dict):
        return ["El reporte no es un objeto."]
    faltan = [campo for campo in CAMPOS_REPORTE if campo not in reporte]
    if faltan:
        errores.append("Faltan campos: " + ", ".join(faltan))
    sobran = [campo for campo in reporte if campo not in CAMPOS_REPORTE]
    if sobran:
        errores.append("Campos fuera del esquema: " + ", ".join(sobran))
    fecha = reporte.get("fecha")
    if not isinstance(fecha, str) or PATRON_FECHA.match(fecha) is None:
        errores.append("fecha tiene que ser YYYY-MM-DD.")
    sesion = reporte.get("sesion")
    if sesion not in SESIONES:
        errores.append("sesion tiene que ser con_registro o sin_registro.")
    for campo in (
        "precio_usd_oz",
        "cierre_previo_usd_oz",
        "cambio_usd",
        "cambio_pct",
        "usd_mxn",
        "mxn_por_gramo",
    ):
        valor = reporte.get(campo)
        if valor is None:
            continue
        if isinstance(valor, bool) or _num(valor) is None:
            errores.append(f"{campo} tiene que ser número o null.")
    for campo in ("precio_aproximado", "cierre_previo_aproximado", "mxn_por_gramo_aproximado"):
        valor = reporte.get(campo)
        if valor is not None and not isinstance(valor, bool):
            errores.append(f"{campo} tiene que ser true, false o null.")
    if reporte.get("usd_mxn_tipo") not in (None, "banxico_fix"):
        errores.append("usd_mxn_tipo tiene que ser banxico_fix o null.")
    fecha_fx = reporte.get("usd_mxn_fecha")
    if fecha_fx is not None and (not isinstance(fecha_fx, str) or PATRON_FECHA.match(fecha_fx) is None):
        errores.append("usd_mxn_fecha tiene que ser YYYY-MM-DD o null.")
    for campo in ("precio_url", "usd_mxn_url", "notion_url"):
        valor = reporte.get(campo)
        if valor is not None and not _es_url(valor):
            errores.append(f"{campo} tiene que ser una URL http(s) o null.")
    for campo in ("precio_fuente", "cambio_texto", "usd_mxn_fuente", "lectura", "notas"):
        valor = reporte.get(campo)
        if valor is not None and not isinstance(valor, str):
            errores.append(f"{campo} tiene que ser texto o null.")
    drivers = reporte.get("drivers")
    if not isinstance(drivers, list):
        errores.append("drivers tiene que ser una lista.")
    else:
        for i, driver in enumerate(drivers):
            if not isinstance(driver, dict):
                errores.append(f"drivers[{i}] no es un objeto.")
                continue
            for campo in CAMPOS_DRIVER:
                if not isinstance(driver.get(campo), str) or not driver.get(campo, "").strip():
                    errores.append(f"drivers[{i}].{campo} tiene que ser texto.")
            if "url" in driver and isinstance(driver.get("url"), str) and not _es_url(driver["url"]):
                errores.append(f"drivers[{i}].url tiene que ser http(s).")
    if sesion == "con_registro":
        precio = _num(reporte.get("precio_usd_oz"))
        if precio is None or precio <= 0:
            errores.append("Un día con registro necesita precio_usd_oz positivo.")
        if not _texto(reporte.get("lectura")):
            errores.append("Un día con registro necesita lectura.")
    if sesion == "sin_registro":
        for campo in CAMPOS_VACIOS_SIN_REGISTRO:
            if reporte.get(campo) is not None:
                errores.append(f"Un día sin registro deja {campo} en null.")
        if drivers:
            errores.append("Un día sin registro deja drivers vacío.")
    return errores


def cargar_reportes(carpeta: Path = CARPETA_REPORTES) -> list[dict]:
    """Lee los JSON diarios, valida y los devuelve ordenados por fecha."""
    if not carpeta.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de reportes: {carpeta}")
    reportes = []
    for ruta in sorted(carpeta.glob("????-??-??.json")):
        reporte = json.loads(ruta.read_text(encoding="utf-8"))
        errores = validar_reporte(reporte)
        if ruta.stem != reporte.get("fecha"):
            errores.append(f"La fecha interna no coincide con el archivo {ruta.name}.")
        if errores:
            raise ValueError(f"{ruta.name}: " + " ".join(errores))
        reportes.append(reporte)
    return reportes


def direccion_pronostico(retorno_log: float) -> str:
    """alza, baja o sin_cambio a partir del retorno logarítmico pronosticado."""
    if retorno_log > 0:
        return "alza"
    if retorno_log < 0:
        return "baja"
    return "sin_cambio"


def nivel_objetivo(precio: float, retorno_log: float) -> float:
    """Precio de hoy llevado por el retorno logarítmico pronosticado."""
    return float(precio) * math.exp(float(retorno_log))


def niveles_alineados(precios: np.ndarray, pred: np.ndarray, horizonte: int) -> np.ndarray:
    """Nivel que el pronóstico del día t apuntaba para t+horizonte.

    El valor queda en la fecha en que el horizonte vence, para poder
    dibujarlo encima del precio realizado. No rellena días futuros que
    todavía no están en la serie.
    """
    precios = np.asarray(precios, dtype=float)
    pred = np.asarray(pred, dtype=float)
    salida = np.full(len(precios), np.nan)
    for origen in range(len(precios)):
        destino = origen + horizonte
        if destino >= len(precios):
            break
        if np.isfinite(pred[origen]) and np.isfinite(precios[origen]) and precios[origen] > 0:
            salida[destino] = precios[origen] * math.exp(pred[origen])
    return salida


def _mae(y: np.ndarray, pred: np.ndarray) -> float | None:
    if len(y) == 0:
        return None
    return float(np.mean(np.abs(y - pred)))


def _rmse(y: np.ndarray, pred: np.ndarray) -> float | None:
    if len(y) == 0:
        return None
    return float(np.sqrt(np.mean((y - pred) ** 2)))


def errores_contra_cero(y: np.ndarray, pred: np.ndarray, horizonte: int) -> dict:
    """MAE y RMSE del modelo y del baseline que dice que el retorno es cero.

    El esquema sin solapar deja una observación cada `horizonte` sesiones,
    la misma regla que la nota del modelo. Las unidades son el retorno
    logarítmico. `mae_pct` lo multiplica por 100 para leerlo como puntos
    porcentuales aproximados, igual que el RMSE del informe.
    """
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    comun = np.isfinite(y) & np.isfinite(pred)

    def bloque(indices: np.ndarray) -> dict:
        if len(indices) == 0:
            vacio = {"n": 0, "mae": None, "rmse": None, "mae_pct": None, "rmse_pct": None,
                     "mae_cero": None, "rmse_cero": None, "mae_cero_pct": None, "rmse_cero_pct": None}
            return vacio
        yy = y[indices]
        pp = pred[indices]
        cero = np.zeros(len(yy))
        mae = _mae(yy, pp)
        rmse = _rmse(yy, pp)
        mae_cero = _mae(yy, cero)
        rmse_cero = _rmse(yy, cero)
        return {
            "n": int(len(indices)),
            "mae": mae,
            "rmse": rmse,
            "mae_pct": None if mae is None else mae * 100.0,
            "rmse_pct": None if rmse is None else rmse * 100.0,
            "mae_cero": mae_cero,
            "rmse_cero": rmse_cero,
            "mae_cero_pct": None if mae_cero is None else mae_cero * 100.0,
            "rmse_cero_pct": None if rmse_cero is None else rmse_cero * 100.0,
        }

    return {
        "sin_solapar": bloque(_tomar_no_solapadas(comun, horizonte)),
        "solapado": bloque(np.flatnonzero(comun)),
    }


def acierto_en_vivo(
    fechas,
    precios: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    horizonte: int,
) -> dict:
    """Acierto de los pronósticos cuyo horizonte ya se realizó.

    Un día con predicción y sin retorno realizado (el horizonte todavía
    no llega) no entra. Tampoco entra un retorno realizado de cero exacto,
    igual que el acierto de dirección del modelo. La tasa que se acumula
    usa observaciones separadas por al menos `horizonte` sesiones.
    """
    fechas = list(fechas)
    precios = np.asarray(precios, dtype=float)
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if not (len(fechas) == len(precios) == len(y) == len(pred)):
        raise ValueError("fechas, precios, y y pred tienen que medir lo mismo.")
    # Primero se elige la rejilla, igual que las métricas del modelo.
    # Un retorno de cero exacto no vota, y no cambia qué días entran.
    comun = np.isfinite(y) & np.isfinite(pred)
    posiciones = _tomar_no_solapadas(comun, horizonte)
    aciertos = 0
    serie = []
    detalle = []
    cuenta = 0
    for posicion in posiciones:
        if y[posicion] == 0:
            continue
        cuenta += 1
        golpe = bool(np.sign(pred[posicion]) == np.sign(y[posicion]))
        aciertos += int(golpe)
        fecha = _fecha_texto(fechas[posicion])
        destino = int(posicion) + int(horizonte)
        precio = _num(precios[posicion])
        nivel = None
        if precio is not None and precio > 0 and np.isfinite(pred[posicion]):
            nivel = nivel_objetivo(precio, float(pred[posicion]))
        precio_realizado = _num(precios[destino]) if destino < len(precios) else None
        fila = {
            "fecha": fecha,
            "fecha_realizada": _fecha_texto(fechas[destino]) if destino < len(fechas) else None,
            "direccion": direccion_pronostico(float(pred[posicion])),
            "direccion_realizada": direccion_pronostico(float(y[posicion])),
            "acierto": golpe,
            "retorno_pronosticado": float(pred[posicion]),
            "retorno_realizado": float(y[posicion]),
            "precio": precio,
            "nivel_objetivo": nivel,
            "precio_realizado": precio_realizado,
        }
        detalle.append(fila)
        serie.append({"fecha": fecha, "n": cuenta, "tasa": aciertos / cuenta})
    n = len(detalle)
    return {
        "n": n,
        "aciertos": aciertos,
        "tasa": None if n == 0 else aciertos / n,
        "serie": serie,
        "ultimos": detalle[-12:],
    }


def _fecha_texto(valor) -> str:
    return pd.Timestamp(valor).strftime("%Y-%m-%d")


def _redondear(valor, decimales: int):
    numero = _num(valor)
    if numero is None:
        return None
    return round(numero, decimales)


def sanitizar(valor):
    """Deja el objeto listo para JSON: sin NaN, sin timestamps, sin numpy."""
    if isinstance(valor, dict):
        return {str(clave): sanitizar(contenido) for clave, contenido in valor.items()}
    if isinstance(valor, (list, tuple)):
        return [sanitizar(contenido) for contenido in valor]
    if isinstance(valor, (np.bool_, bool)):
        return bool(valor)
    if isinstance(valor, (np.integer,)):
        return int(valor)
    if isinstance(valor, (np.floating, float)):
        numero = float(valor)
        if not math.isfinite(numero):
            return None
        return numero
    if isinstance(valor, pd.Timestamp):
        if pd.isna(valor):
            return None
        return valor.strftime("%Y-%m-%d")
    if valor is pd.NaT:
        return None
    try:
        if valor is not None and pd.isna(valor):
            return None
    except (TypeError, ValueError):
        pass
    return valor


def _bloque_metricas(metricas: dict) -> dict:
    """Se queda con los campos que el tablero enseña, en los dos esquemas."""
    campos = (
        "n",
        "r2_cero",
        "r2_media",
        "rmse",
        "rmse_cero",
        "rmse_media",
        "acierto",
        "acierto_siempre_alza",
        "acierto_signo_media",
    )
    salida = {}
    for esquema in ("sin_solapar", "solapado"):
        bloque = metricas.get(esquema) or {}
        salida[esquema] = {campo: bloque.get(campo) for campo in campos}
    return salida


def bloque_pronostico_actual(precio: float, info: dict, horizonte: int) -> dict:
    pred = float(info["pred_ultimo"])
    sigma = _num(info.get("sigma"))
    senal_debil = sigma is None or sigma <= 0 or abs(pred) < 0.25 * sigma
    banda_baja = None
    banda_alta = None
    if sigma is not None and sigma > 0:
        banda_baja = (math.exp(pred - sigma) - 1.0) * 100.0
        banda_alta = (math.exp(pred + sigma) - 1.0) * 100.0
    return {
        "horizonte_dias_mercado": horizonte,
        "fecha": _fecha_texto(info["fecha_ultimo"]),
        "retorno_log": round(pred, 6),
        "retorno_pct": round((math.exp(pred) - 1.0) * 100.0, 3),
        "direccion": direccion_pronostico(pred),
        "nivel_objetivo_usd": round(nivel_objetivo(precio, pred), 2) if precio and precio > 0 else None,
        "banda_baja_pct": None if banda_baja is None else round(banda_baja, 2),
        "banda_alta_pct": None if banda_alta is None else round(banda_alta, 2),
        "sigma_log": sigma,
        "senal_debil": senal_debil,
        "modelo": "ridge",
    }


def serie_grafica(df: pd.DataFrame, desde: str = DESDE_GRAFICA) -> list[dict]:
    """Precio, valor justo, banda de 20 días y régimen. El pronóstico va en el día que vence."""
    precios = df["oro_usd"].to_numpy(dtype=float)
    pred_20 = df["pred_20"].to_numpy(dtype=float) if "pred_20" in df.columns else np.full(len(df), np.nan)
    if "pred_mediana_20" in df.columns and "es_titular_20" in df.attrs and df.attrs.get("es_titular_20"):
        pred_20 = df["pred_mediana_20"].to_numpy(dtype=float)
    alineado = niveles_alineados(precios, pred_20, 20)
    bajo = df["banda_baja_20"].to_numpy(dtype=float) if "banda_baja_20" in df.columns else np.full(len(df), np.nan)
    alto = df["banda_alta_20"].to_numpy(dtype=float) if "banda_alta_20" in df.columns else np.full(len(df), np.nan)
    banda_baja = niveles_alineados(precios, bajo, 20)
    banda_alta = niveles_alineados(precios, alto, 20)
    if "valor_titular" in df.columns:
        justo = df["valor_titular"].to_numpy(dtype=float)
    elif "valor_justo_vf" in df.columns:
        justo = df["valor_justo_vf"].to_numpy(dtype=float)
    else:
        justo = np.full(len(df), np.nan)
    ampliado = df["valor_ext"].to_numpy(dtype=float) if "valor_ext" in df.columns else np.full(len(df), np.nan)
    nuevo = df["valor_kalman"].to_numpy(dtype=float) if "valor_kalman" in df.columns else np.full(len(df), np.nan)
    base = df["valor_justo_vf"].to_numpy(dtype=float) if "valor_justo_vf" in df.columns else np.full(len(df), np.nan)
    regimen = df["regimen"].to_numpy(dtype=float) if "regimen" in df.columns else np.full(len(df), np.nan)
    limite = pd.Timestamp(desde)
    puntos = []
    for i, fecha in enumerate(df.index):
        if pd.Timestamp(fecha) < limite:
            continue
        if not np.isfinite(precios[i]):
            continue
        regimen_i = int(regimen[i]) if np.isfinite(regimen[i]) else None
        puntos.append(
            {
                "fecha": _fecha_texto(fecha),
                "precio": _redondear(precios[i], 2),
                "valor_justo": _redondear(justo[i], 2),
                "valor_ampliado": _redondear(ampliado[i], 2),
                "valor_nuevo": _redondear(nuevo[i], 2),
                "valor_base": _redondear(base[i], 2),
                "pronostico_20": _redondear(alineado[i], 2),
                "intervalo_bajo": _redondear(banda_baja[i], 2),
                "intervalo_alto": _redondear(banda_alta[i], 2),
                "regimen": regimen_i,
            }
        )
    return puntos


MESES = (
    "enero", "febrero", "marzo", "abril", "mayo", "junio",
    "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre",
)
EXTRAS_BRECHA = (
    ("compras_bc_ton", "las compras de bancos centrales"),
    ("log_gld_ton", "el oro guardado en GLD"),
    ("gpr", "el riesgo geopolítico"),
)


def _usd_texto(valor) -> str | None:
    numero = _num(valor)
    if numero is None:
        return None
    if abs(numero - round(numero)) < 0.005:
        return f"{numero:,.0f} dólares"
    return f"{numero:,.2f} dólares"


def _fecha_hablada(iso: str | None) -> str | None:
    if not iso or not isinstance(iso, str) or len(iso) < 10:
        return None
    try:
        anio, mes, dia = (int(parte) for parte in iso[:10].split("-"))
    except ValueError:
        return None
    if not 1 <= mes <= 12:
        return None
    return f"{dia} de {MESES[mes - 1]} de {anio}"


def _frase_cambio(pct) -> str | None:
    numero = _num(pct)
    if numero is None:
        return None
    if numero > 0.05:
        return f"subió {numero:.1f}%"
    if numero < -0.05:
        return f"bajó {abs(numero):.1f}%"
    return "quedó casi igual"


def _frase_brecha(pct) -> str | None:
    numero = _num(pct)
    if numero is None:
        return None
    if numero > 0.05:
        return f"está {numero:.1f}% por encima"
    if numero < -0.05:
        return f"está {abs(numero):.1f}% por debajo"
    return "está pegado"


def _puntos_historia(modelo: dict | None) -> list[dict]:
    if not modelo:
        return []
    puntos = []
    for punto in modelo.get("historia") or []:
        precio = _num(punto.get("precio"))
        fecha = punto.get("fecha")
        if precio is None or precio <= 0 or not fecha:
            continue
        puntos.append({"fecha": str(fecha)[:10], "precio": precio})
    return puntos


def _hace(puntos: list[dict], sesiones: int) -> dict | None:
    if len(puntos) <= sesiones:
        return None
    return puntos[-1 - sesiones]


def _texto_relato(relato: dict) -> str:
    partes = [relato.get("intro") or ""]
    for seccion in relato.get("secciones") or []:
        partes.extend(seccion.get("parrafos") or [])
    return "\n".join(partes)


def _enlaces_drivers(reportes: list[dict]) -> tuple[str, list[dict]]:
    """Arma la frase de temas y las citas literales de los reportes."""
    con_registro = [reporte for reporte in reportes if reporte.get("sesion") == "con_registro"]
    if not con_registro:
        return "No hay reportes con registro en esta exportación.", []
    temas: dict[str, dict] = {}
    for reporte in con_registro:
        for driver in reporte.get("drivers") or []:
            titulo = (driver.get("titulo") or "").strip()
            if not titulo:
                continue
            actual = temas.setdefault(
                titulo,
                {"titulo": titulo, "n": 0, "fecha": "", "detalle": "", "fuente": "", "url": ""},
            )
            actual["n"] += 1
            fecha = str(reporte.get("fecha") or "")
            if fecha >= actual["fecha"]:
                actual["fecha"] = fecha
                actual["detalle"] = (driver.get("detalle") or "").strip()
                actual["fuente"] = (driver.get("fuente") or "").strip()
                actual["url"] = driver.get("url") or ""
    if not temas:
        return "No hay drivers en los reportes de estos días.", []
    ordenados = sorted(temas.values(), key=lambda tema: (tema["n"], tema["fecha"]), reverse=True)
    frases = []
    enlaces = []
    for tema in ordenados[:3]:
        cuando = _fecha_hablada(tema["fecha"]) or tema["fecha"]
        veces = "1 día" if tema["n"] == 1 else f"{tema['n']} días"
        frases.append(f"«{tema['titulo']}» sale en {veces}, la última el {cuando}")
        if tema["detalle"]:
            enlaces.append(
                {
                    "texto": tema["titulo"],
                    "detalle": tema["detalle"],
                    "fuente": tema["fuente"] or "Fuente del log",
                    "url": tema["url"] or None,
                    "fecha": tema["fecha"],
                }
            )
    return "En los reportes de estos días: " + "; ".join(frases) + ".", enlaces


def _parrafos_donde(modelo: dict | None, reportes: list[dict], puntos: list[dict]) -> tuple[list[str], dict | None, list[dict]]:
    parrafos: list[str] = []
    notas: list[dict] = []
    ultimo_reporte = None
    for reporte in reversed(reportes):
        if reporte.get("sesion") == "con_registro" and _num(reporte.get("precio_usd_oz")):
            ultimo_reporte = reporte
            break
    if ultimo_reporte is None:
        parrafos.append("No hay lectura de la mañana en los reportes.")
    else:
        precio = _usd_texto(ultimo_reporte["precio_usd_oz"])
        cuando = _fecha_hablada(ultimo_reporte.get("fecha")) or "ese día"
        aprox = " La cifra del log está marcada como aproximada." if ultimo_reporte.get("precio_aproximado") else ""
        parrafos.append(f"Por la mañana del {cuando} el spot del log anda en {precio}.{aprox} Es la lectura del reporte, no el cierre del futuro.")
        if _num(ultimo_reporte.get("mxn_por_gramo")) is not None:
            gramos = f"{float(ultimo_reporte['mxn_por_gramo']):,.0f}"
            extra = ""
            if _num(ultimo_reporte.get("usd_mxn")) is not None:
                extra = f", con el dólar a {float(ultimo_reporte['usd_mxn']):,.4f}"
                if ultimo_reporte.get("usd_mxn_fecha"):
                    extra += f" ({_fecha_hablada(ultimo_reporte['usd_mxn_fecha'])})"
            parrafos.append(f"En pesos el log lo deja en {gramos} el gramo{extra}.")
        cambio = _frase_cambio(ultimo_reporte.get("cambio_pct"))
        if cambio and _num(ultimo_reporte.get("cambio_usd")) is not None:
            parrafos.append(
                f"Contra el cierre previo {cambio}, unos {abs(float(ultimo_reporte['cambio_usd'])):,.2f} dólares."
            )
        elif cambio:
            parrafos.append(f"Contra el cierre previo {cambio}.")
        else:
            parrafos.append("El reporte no trae el cambio contra el cierre previo.")

    if not modelo or _num((modelo or {}).get("precio_usd_oz")) is None:
        parrafos.append("No tengo el cierre del futuro en esta exportación.")
    else:
        cuando = _fecha_hablada(modelo.get("fecha")) or "el último día del modelo"
        parrafos.append(
            f"El futuro {modelo.get('instrumento') or 'GC=F'} cerró el {cuando} en {_usd_texto(modelo['precio_usd_oz'])}."
        )
        notas.append({"texto": "Cierre del futuro", "valor": _usd_texto(modelo["precio_usd_oz"])})

    if len(puntos) <= 20:
        parrafos.append("No tengo 20 días de mercado en la serie para contar el tramo reciente.")
    else:
        antes = _hace(puntos, 20)
        ultimo = puntos[-1]
        pct = (ultimo["precio"] / antes["precio"] - 1.0) * 100.0
        parrafos.append(
            f"En los últimos 20 días de mercado el futuro {_frase_cambio(pct)}: "
            f"de {_usd_texto(antes['precio'])} el {_fecha_hablada(antes['fecha'])} "
            f"a {_usd_texto(ultimo['precio'])} el {_fecha_hablada(ultimo['fecha'])}."
        )
    if len(puntos) <= 60:
        parrafos.append("No tengo 60 días de mercado en la serie para ese tramo.")
    else:
        antes = _hace(puntos, 60)
        ultimo = puntos[-1]
        pct = (ultimo["precio"] / antes["precio"] - 1.0) * 100.0
        parrafos.append(
            f"En 60 días de mercado {_frase_cambio(pct)}, desde {_usd_texto(antes['precio'])} "
            f"el {_fecha_hablada(antes['fecha'])}."
        )

    if puntos:
        anio = int(puntos[-1]["fecha"][:4])
        del_anio = [punto for punto in puntos if punto["fecha"].startswith(f"{anio}-")]
        anteriores = [punto for punto in puntos if punto["fecha"] < f"{anio}-01-01"]
        if not anteriores:
            parrafos.append(f"No tengo el cierre de {anio - 1} en la serie, así que no hay variación del año.")
        else:
            base = anteriores[-1]
            pct = (puntos[-1]["precio"] / base["precio"] - 1.0) * 100.0
            parrafos.append(
                f"En lo que va de {anio} el futuro {_frase_cambio(pct)} frente al {_fecha_hablada(base['fecha'])} "
                f"({_usd_texto(base['precio'])})."
            )
        if del_anio:
            alto = max(del_anio, key=lambda punto: punto["precio"])
            bajo = min(del_anio, key=lambda punto: punto["precio"])
            parrafos.append(
                f"En {anio}, dentro de esta serie, el futuro tocó {_usd_texto(alto['precio'])} "
                f"el {_fecha_hablada(alto['fecha'])} y {_usd_texto(bajo['precio'])} "
                f"el {_fecha_hablada(bajo['fecha'])}."
            )
    else:
        parrafos.append("No tengo serie de precios para contar cómo llegó hasta aquí.")

    grafica = None
    if len(puntos) >= 2:
        ventana = puntos[-60:]
        grafica = {
            "tipo": "linea",
            "titulo": "Futuro, últimos días de mercado",
            "puntos": [{"fecha": punto["fecha"], "valor": round(punto["precio"], 2)} for punto in ventana],
            "marcas": [
                {"fecha": ventana[0]["fecha"], "etiqueta": "inicio del tramo"},
                {"fecha": ventana[-1]["fecha"], "etiqueta": "último cierre"},
            ],
        }
    return parrafos, grafica, notas


def _parrafos_modelo(modelo: dict | None) -> tuple[list[str], dict | None, list[dict]]:
    parrafos: list[str] = []
    notas: list[dict] = []
    barras = []
    if not modelo or _num((modelo or {}).get("valor_justo_usd")) is None or _num((modelo or {}).get("precio_usd_oz")) is None:
        parrafos.append("No tengo valor justo en esta exportación.")
        return parrafos, None, notas
    precio = float(modelo["precio_usd_oz"])
    justo = float(modelo["valor_justo_usd"])
    brecha = _frase_brecha(modelo.get("brecha_pct"))
    if brecha is None:
        parrafos.append(
            f"El cierre del futuro está en {_usd_texto(precio)} y el valor justo, solo con el pasado, en {_usd_texto(justo)}. "
            "No tengo la brecha en porcentaje."
        )
    else:
        if modelo.get("valor_justo_modelo") == "ols_nuevo":
            parrafos.append(
                f"El cierre del futuro está en {_usd_texto(precio)}. El valor justo titular, "
                f"armado solo con lo ya conocido, queda en {_usd_texto(justo)}. "
                f"El precio {brecha} de esa cuenta."
            )
            base = _usd_texto(modelo.get("valor_justo_base_usd"))
            extra = (
                f"La recta base, también solo con el pasado, queda en {base}. "
                if base
                else ""
            )
            parrafos.append(
                "Ese titular no es la recta de siempre. Suma el breakeven, las compras de bancos centrales, "
                "el índice GPR y una dummie desde el 24 de febrero de 2022. "
                + extra
                + "Se usa como titular porque su error fuera de muestra es menor, aunque hoy la brecha sea más ancha."
            )
        else:
            parrafos.append(
                f"El cierre del futuro está en {_usd_texto(precio)}. El valor justo, armado solo con lo ya conocido "
                f"(tasas, dólar, petróleo, nerviosismo e inflación), queda en {_usd_texto(justo)}. "
                f"El precio {brecha} de esa cuenta."
            )
        if _num(modelo.get("brecha_pct")) is not None and abs(float(modelo["brecha_pct"])) > 15:
            parrafos.append("Una brecha así de ancha puede durar. No es una orden de vender ni de comprar.")
    notas.append({"texto": "Valor justo", "valor": _usd_texto(justo)})
    barras.append({"etiqueta": "Cierre", "valor": round(precio, 2)})
    barras.append({"etiqueta": "Valor justo", "valor": round(justo, 2)})

    ampliado = modelo.get("ampliado")
    if not ampliado:
        parrafos.append(
            "No hay valor justo ampliado en esta exportación, así que no puedo decir si "
            "bancos centrales o geopolítica explican el hueco."
        )
    else:
        variables = set(ampliado.get("variables") or [])
        nombres = [frase for clave, frase in EXTRAS_BRECHA if clave in variables]
        justo_ext = _usd_texto(ampliado.get("valor_justo_usd"))
        brecha_ext = _num(ampliado.get("brecha_pct"))
        cierre = _num(ampliado.get("cierre_abs_pp"))
        if not nombres:
            parrafos.append(
                "Hay un valor justo ampliado, pero no trae compras de bancos centrales, GLD ni riesgo geopolítico. "
                "No voy a decir que el hueco es por eso."
            )
        else:
            if len(nombres) == 1:
                lista = nombres[0]
            else:
                lista = ", ".join(nombres[:-1]) + " y " + nombres[-1]
            if cierre is None:
                efecto = "no tengo en cuántos puntos mueven la brecha"
            elif cierre > 1:
                efecto = f"acercan la brecha a cero en {cierre:.1f} puntos"
            elif cierre < -1:
                efecto = f"alejan la brecha de cero en {abs(cierre):.1f} puntos"
            else:
                efecto = f"casi no mueven la brecha ({cierre:.1f} puntos)"
            frase_brecha = (
                f" Con esas series la brecha queda en {brecha_ext:+.1f}%."
                if brecha_ext is not None
                else " No tengo la brecha del modelo ampliado."
            )
            if cierre is not None and cierre > 1:
                cola = " Cierran una parte. El resto del hueco sigue ahí."
            elif cierre is not None and cierre < -1:
                cola = " No explican el hueco: lo dejan más lejos."
            else:
                cola = " Casi no cambia la historia."
            nivel = f" El valor justo ampliado queda en {justo_ext}." if justo_ext else " No tengo el nivel del valor justo ampliado."
            parrafos.append(
                f"El modelo base no ve {lista}. En esta corrida sí entran, y {efecto}.{frase_brecha}{nivel}{cola}"
            )
        if _num(ampliado.get("valor_justo_usd")) is not None:
            barras.append({"etiqueta": "Con lo que el base no ve", "valor": round(float(ampliado["valor_justo_usd"]), 2)})
            if justo_ext:
                notas.append({"texto": "Valor justo ampliado", "valor": justo_ext})
    grafica = {"tipo": "barras", "titulo": "Dólares por onza", "puntos": barras} if len(barras) >= 2 else None
    return parrafos, grafica, notas


def _frase_pronostico(info: dict | None, horizonte: int) -> str:
    if not info or _num(info.get("retorno_pct")) is None:
        return f"No tengo el pronóstico a {horizonte} días."
    hacia = {"alza": "al alza", "baja": "a la baja"}.get(info.get("direccion"), "a un cambio de cero")
    frase = f"A {horizonte} días de mercado la cuenta apunta {hacia}: {float(info['retorno_pct']):+.2f}%"
    if _num(info.get("nivel_objetivo_usd")) is not None:
        frase += f", hacia {_usd_texto(info['nivel_objetivo_usd'])}."
    else:
        frase += ". No tengo el nivel objetivo."
    bajo = _num(info.get("banda_baja_pct"))
    alto = _num(info.get("banda_alta_pct"))
    if bajo is None or alto is None:
        frase += " No tengo la banda de error de este horizonte."
    else:
        frase += f" La banda de un error típico va de {bajo:+.1f}% a {alto:+.1f}%."
        if bajo < 0 < alto:
            frase += " Esa banda incluye subida y bajada."
    if info.get("senal_debil"):
        frase += " La señal cabe en el error típico."
    return frase


def _frase_confianza(horizonte: int, bloque: dict | None) -> str:
    if not bloque:
        return f"No tengo la precisión a {horizonte} días."
    errores = (bloque.get("errores") or {}).get("sin_solapar") or {}
    metricas = (bloque.get("metricas") or {}).get("sin_solapar") or {}
    mae = _num(errores.get("mae_pct"))
    mae_base = _num(errores.get("mae_cero_pct"))
    partes = [f"A {horizonte} días,"]
    if mae is None or mae_base is None:
        partes.append("no tengo el error contra el baseline de que el precio no cambia.")
    elif mae > mae_base + 0.005:
        partes.append(
            f"el modelo se equivoca más que asumir que el precio no cambia: {mae:.2f}% de error contra {mae_base:.2f}%."
        )
    elif mae < mae_base - 0.005:
        partes.append(
            f"el modelo se equivoca menos que asumir que el precio no cambia: {mae:.2f}% de error contra {mae_base:.2f}%."
        )
    else:
        partes.append(
            f"el error queda casi igual que el de no mover el precio ({mae:.2f}% contra {mae_base:.2f}%)."
        )
    r2_cero = _num(metricas.get("r2_cero"))
    if r2_cero is None:
        partes.append("No tengo el R² contra cero.")
    elif r2_cero < 0:
        partes.append(f"El R² contra cero es {r2_cero:+.3f}: no le gana a la idea de que el precio no cambia.")
    else:
        partes.append(f"El R² contra cero es {r2_cero:+.3f}.")
    acierto = _num(metricas.get("acierto"))
    siempre = _num(metricas.get("acierto_siempre_alza"))
    if acierto is None:
        partes.append("No tengo el acierto de dirección.")
    elif siempre is None:
        partes.append(f"La dirección coincidió el {acierto * 100:.1f}%. No tengo el acierto de siempre alza.")
    else:
        partes.append(
            f"La dirección coincidió el {acierto * 100:.1f}%. Siempre decir que sube habría acertado el {siempre * 100:.1f}%."
        )
    vivo = bloque.get("en_vivo") or {}
    n = vivo.get("n")
    if not n:
        partes.append("Todavía no hay pronósticos vencidos para contar un acierto en vivo.")
    else:
        tasa = _num(vivo.get("tasa"))
        tasa_txt = f" ({tasa * 100:.1f}%)" if tasa is not None else ""
        partes.append(f"Ya con el precio realizado, acertó {vivo.get('aciertos')} de {n}{tasa_txt}.")
    return " ".join(partes)


def _parrafos_confianza(modelo: dict | None) -> tuple[list[str], dict | None]:
    if not modelo:
        return ["No tengo pronósticos ni precisión en esta exportación."], None
    pronosticos = modelo.get("pronosticos") or {}
    precision = modelo.get("precision") or {}
    parrafos = [
        _frase_pronostico(pronosticos.get("5"), 5),
        _frase_pronostico(pronosticos.get("20"), 20),
        _frase_confianza(5, precision.get("5")),
        _frase_confianza(20, precision.get("20")),
    ]
    if precision.get("5") and precision.get("20"):
        gana = precision["5"].get("le_gana_al_cero") or precision["20"].get("le_gana_al_cero")
        if not gana:
            parrafos.append("Ni a 5 ni a 20 días esta cuenta le gana a pensar que el precio no cambia. La flecha es la cuenta, no una señal para irse de lleno.")
    filas = []
    for horizonte in ("5", "20"):
        bloque = precision.get(horizonte) or {}
        errores = (bloque.get("errores") or {}).get("sin_solapar") or {}
        metricas = (bloque.get("metricas") or {}).get("sin_solapar") or {}
        if _num(errores.get("mae_pct")) is None and _num(errores.get("mae_cero_pct")) is None:
            continue
        filas.append(
            {
                "etiqueta": f"{horizonte} días",
                "modelo": _redondear(errores.get("mae_pct"), 2),
                "base": _redondear(errores.get("mae_cero_pct"), 2),
                "acierto": _redondear(metricas.get("acierto"), 3),
            }
        )
    grafica = {"tipo": "comparacion", "titulo": "Error típico, en % del retorno", "filas": filas} if filas else None
    return parrafos, grafica


def _cierre(modelo: dict | None) -> str:
    partes = ["Para cerrar, con calma. Esto no es asesoría financiera."]
    precision = (modelo or {}).get("precision") or {}
    if not precision:
        partes.append("Hoy no tengo la nota del modelo para decir si la flecha merece confianza.")
    else:
        gana = any((precision.get(horizonte) or {}).get("le_gana_al_cero") for horizonte in ("5", "20"))
        if gana:
            partes.append("En algún horizonte la cuenta le gana al baseline de no cambiar, y aun así la banda de error manda.")
        else:
            partes.append("A unos días el modelo no le gana a pensar que el precio no cambia.")
    brecha = _num((modelo or {}).get("brecha_pct"))
    if brecha is not None and abs(brecha) > 15:
        partes.append("Que el precio esté lejos del valor justo no te dice el día en que va a regresar.")
    partes.append("Si vas a comprar, compra en partes: no tiene caso jugártela toda en un solo día.")
    return " ".join(partes)


def construir_relato(modelo: dict | None, reportes: list[dict] | None) -> dict:
    """Arma «La historia del oro» solo con números que ya vienen en el export.

    Las frases son plantillas. Si un campo no está, el texto lo dice y no
    rellena la cifra.
    """
    reportes = list(reportes or [])
    puntos = _puntos_historia(modelo)
    donde, grafica_donde, notas_donde = _parrafos_donde(modelo, reportes, puntos)
    temas, enlaces = _enlaces_drivers(reportes)
    parrafos_impulso = [temas]
    movido = None
    for reporte in reportes:
        if reporte.get("sesion") != "con_registro" or _num(reporte.get("cambio_pct")) is None:
            continue
        if movido is None or abs(float(reporte["cambio_pct"])) > abs(float(movido["cambio_pct"])):
            movido = reporte
    if movido is not None:
        cuando = _fecha_hablada(movido.get("fecha")) or movido.get("fecha")
        parrafos_impulso.append(
            f"El día con más movimiento en el log fue el {cuando}: {_frase_cambio(movido['cambio_pct'])}."
        )
        if movido.get("cambio_texto"):
            parrafos_impulso.append(f"El log lo escribió así: «{movido['cambio_texto']}».")
    else:
        parrafos_impulso.append("Ningún reporte trae el cambio del día, así que no hay un movimiento destacado.")
    lectura = None
    for reporte in reversed(reportes):
        if reporte.get("sesion") == "con_registro" and (reporte.get("lectura") or "").strip():
            lectura = reporte
            break
    if lectura:
        parrafos_impulso.append(
            f"La lectura del {_fecha_hablada(lectura.get('fecha')) or 'último reporte'}, tal como quedó en el log: «{lectura['lectura'].strip()}»."
        )
    barras_dias = []
    for reporte in reportes:
        if _num(reporte.get("cambio_pct")) is None:
            continue
        barras_dias.append(
            {
                "etiqueta": (reporte.get("fecha") or "")[5:10],
                "valor": round(float(reporte["cambio_pct"]), 2),
            }
        )
    grafica_impulso = (
        {"tipo": "barras", "titulo": "Cambio del día en el log, %", "puntos": barras_dias}
        if barras_dias
        else None
    )
    modelo_pars, grafica_modelo, notas_modelo = _parrafos_modelo(modelo)
    confianza, grafica_confianza = _parrafos_confianza(modelo)
    cierre = _cierre(modelo)
    fecha = None
    if modelo and modelo.get("fecha"):
        fecha = str(modelo["fecha"])[:10]
    elif reportes:
        fecha = reportes[-1].get("fecha")
    return {
        "titulo": "La historia del oro",
        "fecha": fecha,
        "intro": (
            "Te la platico con los números de esta exportación. No está escrita a mano. "
            "Si falta un dato, se dice, y no se rellena."
        ),
        "secciones": [
            {
                "id": "donde",
                "paso": "01",
                "kicker": "Dónde está",
                "titulo": "El precio, y cómo llegó hasta aquí",
                "parrafos": donde,
                "notas": notas_donde,
                "enlaces": [],
                "grafica": grafica_donde,
            },
            {
                "id": "impulso",
                "paso": "02",
                "kicker": "Qué lo empuja",
                "titulo": "Lo que dicen los reportes de estos días",
                "parrafos": parrafos_impulso,
                "notas": [],
                "enlaces": enlaces,
                "grafica": grafica_impulso,
            },
            {
                "id": "modelo",
                "paso": "03",
                "kicker": "Qué dice la cuenta",
                "titulo": "El valor justo, y por qué queda hueco",
                "parrafos": modelo_pars,
                "notas": notas_modelo,
                "enlaces": [],
                "grafica": grafica_modelo,
            },
            {
                "id": "confianza",
                "paso": "04",
                "kicker": "Cuánto creerle",
                "titulo": "A dónde apunta, y qué tan seguido le atina",
                "parrafos": confianza,
                "notas": [],
                "enlaces": [],
                "grafica": grafica_confianza,
            },
            {
                "id": "cierre",
                "paso": "05",
                "kicker": "Con calma",
                "titulo": "Si vas a hacer algo, hazlo en partes",
                "parrafos": [cierre],
                "notas": [],
                "enlaces": [],
                "grafica": None,
            },
        ],
    }


def _bloque_desde_cuantil(precio: float, fecha, retorno: float, bajo, alto, horizonte: int) -> dict:
    """Tarjeta del pronóstico nuevo: mediana y banda conformal, no el RMSE del Ridge."""
    banda_baja = None if bajo is None else (math.exp(bajo) - 1.0) * 100.0
    banda_alta = None if alto is None else (math.exp(alto) - 1.0) * 100.0
    senal_debil = (
        bajo is None
        or alto is None
        or (banda_baja is not None and banda_alta is not None and banda_baja < 0 < banda_alta and abs(retorno) < 0.25 * max(abs(bajo), abs(alto)))
    )
    return {
        "horizonte_dias_mercado": horizonte,
        "fecha": _fecha_texto(fecha),
        "retorno_log": round(float(retorno), 6),
        "retorno_pct": round((math.exp(retorno) - 1.0) * 100.0, 3),
        "direccion": direccion_pronostico(float(retorno)),
        "nivel_objetivo_usd": round(nivel_objetivo(precio, float(retorno)), 2) if precio and precio > 0 else None,
        "banda_baja_pct": None if banda_baja is None else round(banda_baja, 2),
        "banda_alta_pct": None if banda_alta is None else round(banda_alta, 2),
        "sigma_log": None,
        "senal_debil": senal_debil,
        "modelo": "mediana_cuantil",
    }


def _prob_publica(estudio: dict | None, horizonte: int):
    if not estudio:
        return None
    bloque = (estudio.get("clasificacion") or {}).get(horizonte)
    if not bloque:
        return None
    return bloque.get("prob_up")


def _valor_titular(valor: dict, estudio: dict | None) -> tuple[float | None, float | None, str]:
    """Precio justo, brecha y nombre del modelo que se muestra como titular."""
    base = valor.get("valor_justo_vf")
    brecha = valor.get("brecha_vf_pct")
    if not estudio or not (estudio.get("valor") or {}).get("promovido"):
        return base, brecha, "base"
    bloque = estudio["valor"]
    if bloque.get("titular") == "markov" and bloque.get("valor_markov") is not None:
        return bloque.get("valor_markov"), bloque.get("brecha_markov_pct"), "markov"
    if bloque.get("titular") == "ols_nuevo" and bloque.get("valor_ols") is not None:
        return bloque.get("valor_ols"), bloque.get("brecha_ols_pct"), "ols_nuevo"
    return base, brecha, "base"


def _estudio_publico(estudio: dict | None) -> dict | None:
    """Quita las corridas internas y deja lo que el tablero puede enseñar."""
    if not estudio:
        return None
    clasificacion = {}
    for horizonte, bloque in (estudio.get("clasificacion") or {}).items():
        clasificacion[str(horizonte)] = {
            "fecha": bloque.get("fecha"),
            "prob_up": bloque.get("prob_up"),
            "prob_plano": bloque.get("prob_plano"),
            "prob_baja": bloque.get("prob_baja"),
            "modelo": bloque.get("modelo"),
            "metricas": bloque.get("metricas"),
        }
    intervalos = {}
    for horizonte, bloque in (estudio.get("intervalos") or {}).items():
        intervalos[str(horizonte)] = bloque
    return {
        "esquema": estudio.get("esquema"),
        "avisos": list(estudio.get("avisos") or []),
        "valor": estudio.get("valor"),
        "comparacion": estudio.get("comparacion"),
        "clasificacion": clasificacion,
        "intervalos": intervalos,
        "regimen": estudio.get("regimen"),
        "analogos": estudio.get("analogos"),
        "dca": estudio.get("dca"),
        "columnas": estudio.get("columnas_features"),
    }


def construir_snapshot(datos: dict, valor: dict, pronostico: dict, reportes: list[dict], estudio: dict | None = None) -> dict:
    """Junta el modelo ya estimado y los reportes diarios en un solo objeto."""
    df = pronostico["df"]
    fecha_modelo = pd.Timestamp(valor["fecha"])
    precio = float(df.loc[fecha_modelo, "oro_usd"])
    tipo_cambio = _num(df.loc[fecha_modelo].get("usd_mxn")) if "usd_mxn" in df.columns else None
    mxn_gramo = None
    if tipo_cambio is not None and tipo_cambio > 0:
        mxn_gramo = onza_a_mxn_por_gramo(precio, tipo_cambio)

    promovidos = {}
    if estudio is not None:
        promovidos = (estudio.get("comparacion") or {}).get("promovido") or {}
    pronosticos = {}
    pronosticos_ridge = {}
    precision = {}
    for horizonte in (5, 20):
        info = pronostico["por_horizonte"][horizonte]
        pronosticos_ridge[str(horizonte)] = bloque_pronostico_actual(precio, info, horizonte)
        y = df[f"y_{horizonte}"].to_numpy(dtype=float)
        usa_nuevo = bool(promovidos.get(str(horizonte))) and f"pred_mediana_{horizonte}" in df.columns
        if usa_nuevo:
            df.attrs[f"es_titular_{horizonte}"] = True
            pred = df[f"pred_mediana_{horizonte}"].to_numpy(dtype=float)
            media = df[f"media_mediana_{horizonte}"].to_numpy(dtype=float)
            fecha_pred = pd.Timestamp(info["fecha_ultimo"])
            pos = int(df.index.get_indexer([fecha_pred])[0])
            if pos < 0 or not np.isfinite(pred[pos]):
                usa_nuevo = False
            else:
                bajo_v = df[f"banda_baja_{horizonte}"].to_numpy(dtype=float)[pos]
                alto_v = df[f"banda_alta_{horizonte}"].to_numpy(dtype=float)[pos]
                bajo = float(bajo_v) if np.isfinite(bajo_v) else None
                alto = float(alto_v) if np.isfinite(alto_v) else None
                pronosticos[str(horizonte)] = _bloque_desde_cuantil(
                    precio, fecha_pred, float(pred[pos]), bajo, alto, horizonte
                )
        if not usa_nuevo:
            pred = df[f"pred_{horizonte}"].to_numpy(dtype=float)
            media = df[f"media_{horizonte}"].to_numpy(dtype=float)
            pronosticos[str(horizonte)] = pronosticos_ridge[str(horizonte)]
        metricas = _bloque_metricas(metricas_pronostico(y, pred, media, horizonte))
        limpio = metricas["sin_solapar"]
        precision[str(horizonte)] = {
            "metricas": metricas,
            "errores": errores_contra_cero(y, pred, horizonte),
            "en_vivo": acierto_en_vivo(df.index, df["oro_usd"].to_numpy(dtype=float), y, pred, horizonte),
            "le_gana_al_cero": limpio.get("r2_cero") is not None and limpio["r2_cero"] > 0,
            "le_gana_al_promedio": limpio.get("r2_media") is not None and limpio["r2_media"] > 0,
            "modelo": "mediana_cuantil" if usa_nuevo else "ridge",
        }

    extendido = valor.get("extendido") or {}
    ampliado = None
    if extendido.get("disponible"):
        ampliado = {
            "valor_justo_usd": _redondear(extendido.get("valor_justo_vf"), 2),
            "brecha_pct": _redondear(extendido.get("brecha_vf_pct"), 2),
            "cierre_abs_pp": _redondear(extendido.get("cierre_abs_pp"), 2),
            "variables": list(extendido.get("variables") or []),
        }

    fuentes = []
    for nombre, info in (datos.get("fuentes") or {}).items():
        if nombre in {"oro_usd", "usd_mxn", "rendimiento_real_10", "dolar_index"}:
            fuentes.append(
                {
                    "serie": nombre,
                    "etiqueta": info.get("etiqueta"),
                    "url": info.get("url"),
                    "ultimo_dato": (datos.get("fechas") or {}).get(nombre),
                }
            )

    snapshot = {
        "generado_en": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "aviso": (
            "No es asesoría financiera. El pronóstico a 5 y 20 días se compara con "
            "predecir que el precio no cambia. Si no le gana, la flecha es la cuenta, no una señal."
        ),
        "grafica_desde": DESDE_GRAFICA,
        "reportes": reportes,
        "relato": None,
        "modelo": {
            "fecha": _fecha_texto(fecha_modelo),
            "instrumento": "GC=F",
            "instrumento_nota": (
                "Futuro continuo del COMEX (Yahoo Finance). No es el spot de Kitco "
                "del reporte de la mañana, y el nivel puede diferir cerca de un 1%."
            ),
            "precio_usd_oz": round(precio, 2),
            "usd_mxn": None if tipo_cambio is None else round(tipo_cambio, 4),
            "usd_mxn_nota": (
                "Tipo de cambio del panel (Yahoo MXN=X, o el respaldo que haya usado la descarga), "
                "no el FIX de Banxico del reporte diario."
            ),
            "mxn_por_gramo": None if mxn_gramo is None else round(mxn_gramo, 2),
            "valor_justo_usd": _redondear(_valor_titular(valor, estudio)[0], 2),
            "brecha_pct": _redondear(_valor_titular(valor, estudio)[1], 2),
            "valor_justo_modelo": _valor_titular(valor, estudio)[2],
            "valor_justo_base_usd": _redondear(valor.get("valor_justo_vf"), 2),
            "brecha_base_pct": _redondear(valor.get("brecha_vf_pct"), 2),
            "valor_justo_muestra_usd": _redondear(valor.get("valor_justo"), 2),
            "brecha_muestra_pct": _redondear(valor.get("brecha_pct"), 2),
            "n": valor.get("n"),
            "inicio": valor.get("inicio"),
            "fin": valor.get("fin"),
            "ampliado": ampliado,
            "pronosticos": pronosticos,
            "pronosticos_ridge": pronosticos_ridge,
            "precision": precision,
            "prob_up_5d": _prob_publica(estudio, 5),
            "prob_up_20d": _prob_publica(estudio, 20),
            "estudio": _estudio_publico(estudio),
            "historia": serie_grafica(df),
            "fuentes": fuentes,
            "avisos": list(datos.get("avisos") or []),
        },
    }
    snapshot["relato"] = construir_relato(snapshot["modelo"], reportes)
    return sanitizar(snapshot)


def escribir_json(ruta: Path, objeto) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(objeto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def escribir_reportes(reportes: list[dict], carpeta: Path = CARPETA_TABLERO) -> Path:
    ruta = carpeta / "reportes.json"
    escribir_json(ruta, reportes)
    return ruta


def escribir_snapshot(snapshot: dict, carpeta: Path = CARPETA_TABLERO) -> Path:
    ruta = carpeta / "snapshot.json"
    escribir_json(ruta, snapshot)
    return ruta


def _panel_guardado() -> pd.DataFrame:
    if not RUTA_PANEL.exists():
        raise FileNotFoundError(
            "No está data/panel_diario.csv. Corre sin --sin-descarga para bajarlo."
        )
    panel = pd.read_csv(RUTA_PANEL, parse_dates=["fecha"])
    panel = panel.set_index("fecha").sort_index()
    return panel


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Exporta el JSON del tablero del oro.")
    parser.add_argument(
        "--solo-reportes",
        action="store_true",
        help="Solo junta los reportes diarios. No corre el modelo.",
    )
    parser.add_argument(
        "--sin-descarga",
        action="store_true",
        help="Usa data/panel_diario.csv en lugar de volver a bajar las series.",
    )
    args = parser.parse_args(argv)

    reportes = cargar_reportes()
    ruta_reportes = escribir_reportes(reportes)
    print(f"Reportes: {len(reportes)} días → {ruta_reportes}", file=sys.stderr)
    if args.solo_reportes:
        ruta_snapshot = CARPETA_TABLERO / "snapshot.json"
        if ruta_snapshot.exists():
            previo = json.loads(ruta_snapshot.read_text(encoding="utf-8"))
            previo["reportes"] = reportes
            previo["relato"] = construir_relato(previo.get("modelo"), reportes)
            escribir_snapshot(sanitizar(previo))
            print(f"Historia actualizada en {ruta_snapshot}", file=sys.stderr)
        return

    if args.sin_descarga:
        print("Leyendo data/panel_diario.csv", file=sys.stderr)
        datos = {
            "panel": _panel_guardado(),
            "fuentes": {},
            "fechas": {},
            "avisos": ["Sin descarga: se usó data/panel_diario.csv."],
        }
    else:
        from datos import actualizar_datos

        print("Descargando series y armando el panel.", file=sys.stderr)
        datos = actualizar_datos()

    from estudio import correr_estudio
    from modelos import ajustar_pronosticos, ajustar_valor_justo, preparar_muestra

    print("Ajustando valor justo.", file=sys.stderr)
    valor = ajustar_valor_justo(preparar_muestra(datos["panel"]))
    print("Ajustando pronósticos a 5 y 20 días. Esta parte es la lenta.", file=sys.stderr)
    pronostico = ajustar_pronosticos(valor["df"])
    print("Ajustando regímenes, clasificación e intervalos.", file=sys.stderr)
    estudio = correr_estudio(pronostico["df"])
    snapshot = construir_snapshot(datos, valor, pronostico, reportes, estudio)
    ruta = escribir_snapshot(snapshot)
    modelo = snapshot["modelo"]
    print(
        f"Snapshot: {ruta} · {modelo['fecha']} · {modelo['precio_usd_oz']} USD/oz · "
        f"valor justo {modelo['valor_justo_usd']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
