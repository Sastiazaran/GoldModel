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
    """Precio, valor justo y el pronóstico de 20 días ya alineado a su vencimiento."""
    precios = df["oro_usd"].to_numpy(dtype=float)
    pred_20 = df["pred_20"].to_numpy(dtype=float) if "pred_20" in df.columns else np.full(len(df), np.nan)
    alineado = niveles_alineados(precios, pred_20, 20)
    justo = df["valor_justo_vf"].to_numpy(dtype=float) if "valor_justo_vf" in df.columns else np.full(len(df), np.nan)
    ampliado = df["valor_ext"].to_numpy(dtype=float) if "valor_ext" in df.columns else np.full(len(df), np.nan)
    limite = pd.Timestamp(desde)
    puntos = []
    for i, fecha in enumerate(df.index):
        if pd.Timestamp(fecha) < limite:
            continue
        if not np.isfinite(precios[i]):
            continue
        puntos.append(
            {
                "fecha": _fecha_texto(fecha),
                "precio": _redondear(precios[i], 2),
                "valor_justo": _redondear(justo[i], 2),
                "valor_ampliado": _redondear(ampliado[i], 2),
                "pronostico_20": _redondear(alineado[i], 2),
            }
        )
    return puntos


def construir_snapshot(datos: dict, valor: dict, pronostico: dict, reportes: list[dict]) -> dict:
    """Junta el modelo ya estimado y los reportes diarios en un solo objeto."""
    df = pronostico["df"]
    fecha_modelo = pd.Timestamp(valor["fecha"])
    precio = float(df.loc[fecha_modelo, "oro_usd"])
    tipo_cambio = _num(df.loc[fecha_modelo].get("usd_mxn")) if "usd_mxn" in df.columns else None
    mxn_gramo = None
    if tipo_cambio is not None and tipo_cambio > 0:
        mxn_gramo = onza_a_mxn_por_gramo(precio, tipo_cambio)

    pronosticos = {}
    precision = {}
    for horizonte in (5, 20):
        info = pronostico["por_horizonte"][horizonte]
        pronosticos[str(horizonte)] = bloque_pronostico_actual(precio, info, horizonte)
        y = df[f"y_{horizonte}"].to_numpy(dtype=float)
        pred = df[f"pred_{horizonte}"].to_numpy(dtype=float)
        media = df[f"media_{horizonte}"].to_numpy(dtype=float)
        metricas = _bloque_metricas(metricas_pronostico(y, pred, media, horizonte))
        limpio = metricas["sin_solapar"]
        precision[str(horizonte)] = {
            "metricas": metricas,
            "errores": errores_contra_cero(y, pred, horizonte),
            "en_vivo": acierto_en_vivo(df.index, df["oro_usd"].to_numpy(dtype=float), y, pred, horizonte),
            "le_gana_al_cero": limpio.get("r2_cero") is not None and limpio["r2_cero"] > 0,
            "le_gana_al_promedio": limpio.get("r2_media") is not None and limpio["r2_media"] > 0,
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
            "valor_justo_usd": _redondear(valor.get("valor_justo_vf"), 2),
            "brecha_pct": _redondear(valor.get("brecha_vf_pct"), 2),
            "valor_justo_muestra_usd": _redondear(valor.get("valor_justo"), 2),
            "brecha_muestra_pct": _redondear(valor.get("brecha_pct"), 2),
            "n": valor.get("n"),
            "inicio": valor.get("inicio"),
            "fin": valor.get("fin"),
            "ampliado": ampliado,
            "pronosticos": pronosticos,
            "precision": precision,
            "historia": serie_grafica(df),
            "fuentes": fuentes,
            "avisos": list(datos.get("avisos") or []),
        },
    }
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

    from modelos import ajustar_pronosticos, ajustar_valor_justo, preparar_muestra

    print("Ajustando valor justo.", file=sys.stderr)
    valor = ajustar_valor_justo(preparar_muestra(datos["panel"]))
    print("Ajustando pronósticos a 5 y 20 días. Esta parte es la lenta.", file=sys.stderr)
    pronostico = ajustar_pronosticos(valor["df"])
    snapshot = construir_snapshot(datos, valor, pronostico, reportes)
    ruta = escribir_snapshot(snapshot)
    modelo = snapshot["modelo"]
    print(
        f"Snapshot: {ruta} · {modelo['fecha']} · {modelo['precio_usd_oz']} USD/oz · "
        f"valor justo {modelo['valor_justo_usd']}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
