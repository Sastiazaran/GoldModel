"""Descarga y alinea los datos del oro y de sus posibles causas.

Cada función hace un paso. `actualizar_datos` las llama en orden:
bajar series, ponerlas en los días en que el oro negoció, y cuidar
de no usar un dato antes de que existiera (sobre todo el IPC mensual).

Fuentes, en este orden:
1. FRED, con el CSV público (no pide llave).
2. Si FRED no contesta, un respaldo gratuito de la misma idea:
   Tesoro de EE.UU., BLS, Fed de Nueva York, CBOE o Yahoo Finance.
"""

from __future__ import annotations

import io
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# Carpeta del proyecto, no la carpeta desde la que escribiste el comando.
RAIZ = Path(__file__).resolve().parent
CARPETA_DATOS = RAIZ / "data"
CARPETA_CRUDOS = CARPETA_DATOS / "raw"

# 2006 alcanza para ver 2008, el máximo de 2011, la subida de tasas de 2022
# y el ciclo reciente. Son más de 15 años de días de mercado.
INICIO = "2006-01-01"
# El IPC necesita 12 meses previos para la variación anual.
INICIO_IPC = "2004-01-01"
# Un feriado en EE.UU. puede dejar un hueco de unos pocos días hábiles.
LIMITE_RELLENO = 5

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; estudio-oro/1.0; "
        "proyecto de estudio, no comercial)"
    ),
    "Accept": "text/csv,application/json,text/plain,*/*",
}

# Tickers de Yahoo Finance. Son los mismos que pedirías con yfinance.
TICKERS_YAHOO = {
    "oro_usd": "GC=F",
    "gld": "GLD",
    "dolar_index": "DX-Y.NYB",
    "tnx_yahoo": "^TNX",
    "vix_yahoo": "^VIX",
    "brent": "BZ=F",
    "sp500": "^GSPC",
    "plata": "SI=F",
    "usd_mxn": "MXN=X",
    # GVZ: volatilidad implícita del oro en el CBOE. Es un cierre de mercado,
    # usable el mismo día, igual que el VIX. Si Yahoo no lo trae, el modelo sigue.
    "gvz": "^GVZ",
}

SERIES_FRED = {
    "rendimiento_real_10": "DFII10",
    "rendimiento_nominal_10": "DGS10",
    "rendimiento_nominal_2": "DGS2",
    "breakeven_10": "T10YIE",
    "fed_funds": "DFF",
    "vix": "VIXCLS",
    "brent_spot": "DCOILBRENTEU",
    "dolar_amplio": "DTWEXBGS",
    "usd_mxn_fred": "DEXMXUS",
    "cpi": "CPIAUCSL",
}

# Precios que no pueden ser cero o negativos. Las tasas sí pueden ser negativas:
# la tasa real a 10 años lo fue varios años. No la filtres aquí.
SERIES_POSITIVAS = {
    "oro_usd",
    "gld",
    "dolar_index",
    "dolar_amplio",
    "brent",
    "brent_spot",
    "sp500",
    "plata",
    "usd_mxn",
    "usd_mxn_fred",
    "vix",
    "vix_yahoo",
    "gvz",
    "cpi",
}


def _aviso(mensaje: str) -> None:
    print(mensaje, file=sys.stderr)


def fecha_publicacion_cpi(mes: pd.Timestamp) -> pd.Timestamp:
    """Día en que tratamos como conocido el IPC de un mes.

    El IPC de enero no se publica el 1 de enero. Sale hacia mediados de
    febrero. Usamos el día 15 del mes siguiente. Si cae en sábado o domingo,
    pasa al lunes. Es una regla fija y un poco tardía (a veces el dato sale
    el día 10), así que no se adelanta. No es el calendario oficial del BLS
    día por día: en un cierre del gobierno de EE.UU. el dato puede salir aún
    más tarde.
    """
    mes = pd.Timestamp(mes).normalize().replace(day=1)
    if mes.month == 12:
        dia = pd.Timestamp(year=mes.year + 1, month=1, day=15)
    else:
        dia = pd.Timestamp(year=mes.year, month=mes.month + 1, day=15)
    while dia.weekday() >= 5:
        dia += pd.Timedelta(days=1)
    return dia


def variacion_anual_ipc(indice: pd.Series) -> pd.Series:
    """Variación de 12 meses del índice, en fracción (0.03 = 3%).

    `shift(12)` mira 12 meses hacia atrás. El primer año queda vacío a
    propósito: todavía no hay con qué comparar.
    """
    indice = indice.sort_index()
    return indice / indice.shift(12) - 1.0


def tabla_publicacion_cpi(indice_mensual: pd.Series) -> pd.DataFrame:
    """Arma la tabla mensual con la fecha en que el dato ya se puede usar."""
    indice = pd.to_numeric(indice_mensual, errors="coerce").dropna().sort_index()
    indice.index = pd.to_datetime(indice.index).to_period("M").to_timestamp()
    indice = indice.groupby(level=0).last().sort_index()
    completo = pd.date_range(indice.index.min(), indice.index.max(), freq="MS")
    indice = indice.reindex(completo)
    tabla = pd.DataFrame(
        {
            "mes_referencia": indice.index,
            "cpi_indice": indice.to_numpy(dtype=float),
            "cpi_anual": (variacion_anual_ipc(indice) * 100.0).to_numpy(dtype=float),
        }
    )
    tabla["fecha_disponible"] = tabla["mes_referencia"].map(fecha_publicacion_cpi)
    tabla = tabla.dropna(subset=["cpi_indice", "cpi_anual"])
    tabla = tabla.drop_duplicates("fecha_disponible", keep="last")
    return tabla.sort_values("fecha_disponible").reset_index(drop=True)


def pegar_cpi_sin_adelanto(fechas, tabla: pd.DataFrame) -> pd.DataFrame:
    """Para cada día, pega el último IPC cuya publicación ya ocurrió.

    `direction='backward'` quiere decir: usar el dato más reciente entre
    los que tienen fecha de publicación menor o igual a ese día. Un IPC de
    agosto publicado el 15 de septiembre no entra en un día de agosto.
    """
    base = pd.DataFrame({"fecha": _fechas_ns(fechas)})
    base = base.sort_values("fecha")
    tab = tabla.copy()
    tab["fecha_disponible"] = _fechas_ns(tab["fecha_disponible"])
    if "mes_referencia" in tab.columns:
        tab["mes_referencia"] = _fechas_ns(tab["mes_referencia"])
    tab = tab.sort_values("fecha_disponible").drop_duplicates(
        "fecha_disponible", keep="last"
    )
    return pd.merge_asof(
        base,
        tab,
        left_on="fecha",
        right_on="fecha_disponible",
        direction="backward",
    )


def rellenar_hacia_adelante(serie: pd.Series, limite: int = LIMITE_RELLENO) -> pd.Series:
    """Copia el último valor conocido, hacia adelante, como máximo `limite` días.

    No rellena hacia atrás. Rellenar hacia atrás usaría un dato que en ese
    día todavía no existía.
    """
    return serie.ffill(limit=limite)


def _fechas_ns(valores) -> pd.DatetimeIndex:
    """Una sola resolución de fecha.

    Pandas 3 no deja unir, con merge_asof, un día en segundos con un día en
    microsegundos. Todo el proyecto usa nanosegundos, sin hora.
    """
    indice = pd.to_datetime(valores, errors="coerce")
    if getattr(indice, "tz", None) is not None:
        indice = indice.tz_convert("UTC").tz_localize(None)
    if not isinstance(indice, pd.DatetimeIndex):
        indice = pd.DatetimeIndex(indice)
    return indice.as_unit("ns").normalize()


def normalizar_serie(serie: pd.Series, solo_positivos: bool = False) -> pd.Series:
    """Deja una columna con fechas (sin hora) y números, ordenada, sin duplicados."""
    valores = pd.to_numeric(serie, errors="coerce")
    indice = _fechas_ns(serie.index)
    salida = pd.Series(valores.to_numpy(dtype=float), index=indice)
    salida = salida[~salida.index.isna()]
    salida = salida.groupby(level=0).last().sort_index()
    salida = salida.replace([np.inf, -np.inf], np.nan).dropna()
    if solo_positivos:
        salida = salida[salida > 0]
    salida.index.name = "fecha"
    return salida


def extraer_cierre(datos: pd.DataFrame) -> pd.Series:
    """Saca la columna Close de la tabla que devuelve yfinance.

    yfinance 1.x a veces entrega columnas de dos pisos, por ejemplo
    ('Close', 'GC=F'). Preferimos Close y no Adj Close: en un futuro del
    oro el ajuste no cambia la lectura, y Close es el precio del día.
    """
    if not isinstance(datos.columns, pd.MultiIndex):
        if "Close" in datos.columns:
            return datos["Close"]
        raise KeyError("La tabla de Yahoo no trae columna Close.")

    def texto(par: tuple) -> str:
        return " ".join(str(parte) for parte in par)

    pares = list(datos.columns)
    cierres = [par for par in pares if "Close" in texto(par) and "Adj Close" not in texto(par)]
    if not cierres:
        cierres = [par for par in pares if "Adj Close" in texto(par)]
    if not cierres:
        raise KeyError(f"No encontré Close en {pares[:8]}")
    serie = datos[cierres[0]]
    if isinstance(serie, pd.DataFrame):
        serie = serie.iloc[:, 0]
    return serie


def _sesion() -> requests.Session:
    sesion = requests.Session()
    sesion.headers.update(HEADERS)
    return sesion


def _guardar_serie(nombre: str, serie: pd.Series) -> None:
    CARPETA_CRUDOS.mkdir(parents=True, exist_ok=True)
    ruta = CARPETA_CRUDOS / f"{nombre}.csv"
    serie.rename("valor").to_csv(ruta, index_label="fecha")


def _leer_serie_cache(nombre: str, solo_positivos: bool = False) -> pd.Series | None:
    ruta = CARPETA_CRUDOS / f"{nombre}.csv"
    if not ruta.exists():
        return None
    tabla = pd.read_csv(ruta, parse_dates=["fecha"])
    serie = pd.Series(tabla["valor"].to_numpy(), index=pd.to_datetime(tabla["fecha"]))
    return normalizar_serie(serie, solo_positivos=solo_positivos)


def descargar_yahoo_uno(ticker: str, inicio: str, fin: str) -> pd.Series:
    """Baja el cierre diario de un ticker. Reintenta porque Yahoo a veces falla."""
    ultimo_error: Exception | None = None
    for intento in range(3):
        try:
            # auto_adjust=False deja el precio de cierre, sin un ajuste que
            # en el futuro del oro no cambia la lectura. threads=False evita
            # varias descargas a la vez: es más fácil de seguir y falla menos.
            kwargs = dict(
                start=inicio,
                end=fin,
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            try:
                datos = yf.download(ticker, timeout=30, **kwargs)
            except TypeError:
                datos = yf.download(ticker, **kwargs)
            if datos is None or len(datos) == 0:
                raise RuntimeError(f"Yahoo devolvió una tabla vacía para {ticker}.")
            return normalizar_serie(extraer_cierre(datos), solo_positivos=True)
        except Exception as error:  # noqa: BLE001 - reintento deliberado
            ultimo_error = error
            time.sleep(1.5 * (intento + 1))
    raise RuntimeError(f"No pude bajar {ticker} desde Yahoo Finance: {ultimo_error}")


def _fred_a_serie(texto: str, serie_id: str, solo_positivos: bool) -> pd.Series | None:
    if "DATE" not in texto[:80].upper() and "observation_date" not in texto[:80]:
        return None
    tabla = pd.read_csv(io.StringIO(texto))
    tabla.columns = [str(columna).strip() for columna in tabla.columns]
    columna_fecha = "DATE" if "DATE" in tabla.columns else tabla.columns[0]
    if serie_id not in tabla.columns:
        return None
    fechas = pd.to_datetime(tabla[columna_fecha], errors="coerce")
    valores = pd.to_numeric(tabla[serie_id], errors="coerce")
    serie = pd.Series(valores.to_numpy(), index=fechas)
    serie = normalizar_serie(serie, solo_positivos=solo_positivos)
    if serie.empty:
        return None
    return serie


def descargar_fred(sesion: requests.Session, serie_id: str, inicio: str, fin: str) -> pd.Series | None:
    """Intenta el CSV público de FRED. Si el servidor no contesta, devuelve None."""
    url = (
        "https://fred.stlouisfed.org/graph/fredgraph.csv"
        f"?id={serie_id}&cosd={inicio}&coed={fin}"
    )
    try:
        respuesta = sesion.get(url, timeout=(5, 12))
    except requests.RequestException:
        return None
    if respuesta.status_code != 200 or len(respuesta.content) < 40:
        return None
    return _fred_a_serie(respuesta.text, serie_id, solo_positivos=serie_id in {"VIXCLS", "DCOILBRENTEU", "DTWEXBGS", "DEXMXUS", "CPIAUCSL"})


def _columna_plazo(columnas, anios: int) -> str | None:
    objetivo = f"{anios}yr"
    for columna in columnas:
        limpio = "".join(caracter for caracter in str(columna).lower() if caracter.isalnum())
        if limpio == objetivo:
            return columna
    return None


def _serie_desde_csv_tesoro(texto: str, anios: int) -> pd.Series:
    tabla = pd.read_csv(io.StringIO(texto))
    tabla.columns = [str(columna).strip() for columna in tabla.columns]
    columna = _columna_plazo(tabla.columns, anios)
    if columna is None:
        raise ValueError(f"El CSV del Tesoro no trae el plazo {anios} años: {list(tabla.columns)}")
    fechas = pd.to_datetime(tabla.iloc[:, 0], format="%m/%d/%Y", errors="coerce")
    valores = pd.to_numeric(tabla[columna], errors="coerce")
    return normalizar_serie(pd.Series(valores.to_numpy(), index=fechas), solo_positivos=False)


def descargar_tesoro(
    sesion: requests.Session,
    real: bool,
    anios: int,
    inicio: str,
    fin: str,
    avisos: list[str],
) -> pd.Series:
    """Curva del Tesoro de EE.UU., un año por archivo.

    Es el respaldo de DFII10, DGS10 y DGS2. Los años viejos se guardan en
    data/raw y no se vuelven a bajar. El año en curso y el anterior sí,
    porque ahí están los datos nuevos.
    """
    tipo = "daily_treasury_real_yield_curve" if real else "daily_treasury_yield_curve"
    anio_inicio = int(inicio[:4])
    anio_fin = int(fin[:4])
    pedazos: list[pd.Series] = []
    for anio in range(anio_inicio, anio_fin + 1):
        nombre = f"tesoro_{'real' if real else 'nominal'}_{anio}.csv"
        ruta = CARPETA_CRUDOS / nombre
        refrescar = anio >= anio_fin - 1
        url = (
            "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
            f"daily-treasury-rates.csv/{anio}/all?type={tipo}"
            f"&field_tdr_date_value={anio}&page&_format=csv"
        )
        texto = None
        if ruta.exists() and not refrescar:
            texto = ruta.read_text(encoding="utf-8")
        else:
            try:
                respuesta = sesion.get(url, timeout=40)
                respuesta.raise_for_status()
                texto = respuesta.text
                if "Date" not in texto[:80]:
                    raise RuntimeError("La respuesta no parece un CSV del Tesoro.")
                CARPETA_CRUDOS.mkdir(parents=True, exist_ok=True)
                ruta.write_text(texto, encoding="utf-8")
                time.sleep(0.15)
            except Exception as error:  # noqa: BLE001
                if ruta.exists():
                    avisos.append(
                        f"No pude refrescar el Tesoro {anio} ({error}). Uso la copia guardada."
                    )
                    texto = ruta.read_text(encoding="utf-8")
                else:
                    avisos.append(f"Año {anio} del Tesoro no disponible: {error}")
                    continue
        try:
            pedazos.append(_serie_desde_csv_tesoro(texto, anios))
        except Exception as error:  # noqa: BLE001
            avisos.append(f"No pude leer el Tesoro {anio}: {error}")
    if not pedazos:
        raise RuntimeError("No hubo datos del Tesoro de EE.UU.")
    serie = pd.concat(pedazos)
    serie = normalizar_serie(serie, solo_positivos=False)
    return serie.loc[inicio:fin]


def descargar_cpi_bls(sesion: requests.Session, avisos: list[str]) -> pd.Series:
    """IPC de Estados Unidos, ajustado por estacionalidad.

    CUSR0000SA0 es la serie del BLS que corresponde a CPIAUCSL de FRED.
    La API pública admite unos 10 años por consulta, así que partimos el
    historial. Guardamos el índice mensual; la fecha de publicación se
    aplica después.
    """
    anio_hoy = pd.Timestamp.today().year
    bloques = []
    anio = 2004
    while anio <= anio_hoy:
        fin_bloque = min(anio + 9, anio_hoy)
        bloques.append((str(anio), str(fin_bloque)))
        anio = fin_bloque + 1
    filas: list[tuple[pd.Timestamp, float]] = []
    try:
        for inicio_bloque, fin_bloque in bloques:
            respuesta = sesion.post(
                "https://api.bls.gov/publicAPI/v2/timeseries/data/",
                json={
                    "seriesid": ["CUSR0000SA0"],
                    "startyear": inicio_bloque,
                    "endyear": fin_bloque,
                },
                timeout=40,
            )
            respuesta.raise_for_status()
            cuerpo = respuesta.json()
            if cuerpo.get("status") != "REQUEST_SUCCEEDED":
                raise RuntimeError(cuerpo.get("message"))
            for punto in cuerpo["Results"]["series"][0]["data"]:
                periodo = str(punto.get("period", ""))
                if not periodo.startswith("M") or periodo == "M13":
                    continue
                mes = int(periodo[1:])
                if not 1 <= mes <= 12:
                    continue
                filas.append(
                    (
                        pd.Timestamp(year=int(punto["year"]), month=mes, day=1),
                        float(punto["value"]),
                    )
                )
            time.sleep(0.2)
    except Exception as error:  # noqa: BLE001
        cache = _leer_serie_cache("cpi_bls", solo_positivos=True)
        if cache is None:
            raise RuntimeError(f"No pude bajar el IPC del BLS: {error}") from error
        avisos.append(f"No pude refrescar el IPC ({error}). Uso la copia guardada.")
        return cache

    serie = normalizar_serie(
        pd.Series([valor for _, valor in filas], index=[fecha for fecha, _ in filas]),
        solo_positivos=True,
    )
    _guardar_serie("cpi_bls", serie)
    return serie


def descargar_vix_cboe(sesion: requests.Session, avisos: list[str]) -> pd.Series | None:
    """VIX oficial de CBOE. Es la misma idea que VIXCLS en FRED."""
    url = "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    try:
        respuesta = sesion.get(url, timeout=40)
        respuesta.raise_for_status()
        tabla = pd.read_csv(io.BytesIO(respuesta.content))
        tabla.columns = [str(columna).strip().upper() for columna in tabla.columns]
        fechas = pd.to_datetime(tabla["DATE"], format="%m/%d/%Y", errors="coerce")
        serie = normalizar_serie(
            pd.Series(pd.to_numeric(tabla["CLOSE"], errors="coerce").to_numpy(), index=fechas),
            solo_positivos=True,
        )
        _guardar_serie("vix_cboe", serie)
        return serie
    except Exception as error:  # noqa: BLE001
        avisos.append(f"No pude bajar el VIX de CBOE ({error}).")
        return _leer_serie_cache("vix_cboe", solo_positivos=True)


def descargar_fed_funds(sesion: requests.Session, inicio: str, fin: str, avisos: list[str]) -> pd.Series | None:
    """Tasa efectiva de fondos federales (EFFR), el equivalente de FRED DFF."""
    url = (
        "https://markets.newyorkfed.org/api/rates/unsecured/effr/search.json"
        f"?startDate={inicio}&endDate={fin}"
    )
    try:
        respuesta = sesion.get(url, timeout=40)
        respuesta.raise_for_status()
        registros = respuesta.json().get("refRates", [])
        fechas = [registro.get("effectiveDate") for registro in registros]
        valores = [registro.get("percentRate") for registro in registros]
        serie = normalizar_serie(
            pd.Series(pd.to_numeric(valores, errors="coerce"), index=pd.to_datetime(fechas)),
            solo_positivos=False,
        )
        _guardar_serie("fed_funds_nyfed", serie)
        return serie
    except Exception as error:  # noqa: BLE001
        avisos.append(f"No pude bajar la tasa de fondos federales ({error}).")
        return _leer_serie_cache("fed_funds_nyfed", solo_positivos=False)


def _elegir(
    principal: pd.Series | None,
    respaldo: pd.Series | None,
    minimo: int,
) -> pd.Series | None:
    if principal is not None and principal.dropna().shape[0] >= minimo:
        return principal
    if respaldo is not None and respaldo.dropna().shape[0] >= minimo:
        return respaldo
    return principal if principal is not None else respaldo


def _etiqueta_fred(serie_id: str, uso_fred: bool, respaldo: str) -> str:
    if uso_fred:
        return f"FRED {serie_id}"
    return respaldo


def fecha_publicacion_cot(fecha_corte: pd.Timestamp) -> pd.Timestamp:
    """Día en que el COT de la CFTC ya se puede usar.

    El informe legacy corta el martes y se publica el viernes de esa semana
    (tres días después). Si la fecha no es martes —los informes viejos, de
    antes del calendario semanal, caían el 15 o a fin de mes— se usan los
    mismos tres días y, si caen en sábado o domingo, el lunes.
    """
    fecha = pd.Timestamp(fecha_corte).normalize()
    salida = fecha + pd.Timedelta(days=3)
    while salida.weekday() >= 5:
        salida += pd.Timedelta(days=1)
    return salida


def fecha_publicacion_gld(fecha_corte: pd.Timestamp) -> pd.Timestamp:
    """Las tenencias de un día se publican al día hábil siguiente.

    El archivo de GLD trae el cierre de la sesión D. Ese archivo no está
    la misma tarde: aparece a la mañana siguiente. Por eso el dato de un
    viernes entra el lunes, y el de un lunes entra el martes.
    """
    dia = pd.Timestamp(fecha_corte).normalize() + pd.Timedelta(days=1)
    while dia.weekday() >= 5:
        dia += pd.Timedelta(days=1)
    return dia


def fecha_publicacion_compras_bc(fin_trimestre: pd.Timestamp) -> pd.Timestamp:
    """El trimestre oficial se trata como conocido a fin del mes siguiente.

    El World Gold Council dice que el dato trimestral se actualiza alrededor
    de un mes después de que cierra el trimestre. Fin de julio para el
    segundo trimestre, fin de enero para el cuarto. Es una regla fija, un
    poco tardía: mejor entrar tarde que usar el trimestre antes de tiempo.
    `fin_trimestre` tiene que ser el último día del trimestre (31 mar, 30 jun,
    30 sep o 31 dic).
    """
    fin = pd.Timestamp(fin_trimestre).normalize()
    return (fin + pd.offsets.MonthEnd(1)).normalize()


def fecha_publicacion_mes_siguiente(mes: pd.Timestamp) -> pd.Timestamp:
    """Primer día hábil del mes siguiente. Respaldo si el GPR solo es mensual."""
    mes = pd.Timestamp(mes).normalize().replace(day=1)
    if mes.month == 12:
        dia = pd.Timestamp(year=mes.year + 1, month=1, day=1)
    else:
        dia = pd.Timestamp(year=mes.year, month=mes.month + 1, day=1)
    while dia.weekday() >= 5:
        dia += pd.Timedelta(days=1)
    return dia


def _fecha_etiqueta_trimestre(texto: str) -> pd.Timestamp | None:
    """Convierte 'Q2 26' en el último día de ese trimestre (2026-06-30)."""
    partes = str(texto).strip().split()
    if len(partes) != 2 or not partes[0].startswith("Q"):
        return None
    try:
        trimestre = int(partes[0][1:])
        anio = int(partes[1])
    except ValueError:
        return None
    if trimestre < 1 or trimestre > 4:
        return None
    if anio < 100:
        anio = 2000 + anio
    mes = trimestre * 3
    return (pd.Timestamp(year=anio, month=mes, day=1) + pd.offsets.MonthEnd(0)).normalize()


def pegar_publicado(fechas, tabla: pd.DataFrame, columnas: list[str]) -> pd.DataFrame:
    """Pega, en cada día, el último dato cuya fecha de publicación ya pasó.

    Misma idea que el IPC: `merge_asof` hacia atrás. Una fila con
    `fecha_disponible` en el futuro no entra.
    """
    base = pd.DataFrame({"fecha": _fechas_ns(fechas)})
    base = base.sort_values("fecha")
    tab = tabla.copy()
    tab["fecha_disponible"] = _fechas_ns(tab["fecha_disponible"])
    for columna in columnas:
        if columna not in tab.columns:
            tab[columna] = np.nan
    tab = tab.dropna(subset=["fecha_disponible"])
    tab = tab.sort_values("fecha_disponible").drop_duplicates("fecha_disponible", keep="last")
    return pd.merge_asof(
        base,
        tab[["fecha_disponible", *columnas]],
        left_on="fecha",
        right_on="fecha_disponible",
        direction="backward",
    )


CARPETA_SNAPSHOTS = CARPETA_DATOS / "snapshots"


def _guardar_tabla(nombre: str, tabla: pd.DataFrame) -> None:
    CARPETA_CRUDOS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(CARPETA_CRUDOS / f"{nombre}.csv", index=False)
    # Copia en el repositorio. data/raw está ignorado; si la fuente falla
    # la próxima vez, esta copia (con fecha y URL en MANIFEST.json) es el respaldo.
    CARPETA_SNAPSHOTS.mkdir(parents=True, exist_ok=True)
    tabla.to_csv(CARPETA_SNAPSHOTS / f"{nombre}.csv", index=False)


def _leer_tabla_en(ruta: Path) -> pd.DataFrame | None:
    if not ruta.exists():
        return None
    tabla = pd.read_csv(ruta)
    if "fecha_disponible" not in tabla.columns or tabla.empty:
        return None
    tabla["fecha_disponible"] = pd.to_datetime(tabla["fecha_disponible"])
    return tabla


def _leer_tabla_cache(nombre: str) -> pd.DataFrame | None:
    tabla = _leer_tabla_en(CARPETA_CRUDOS / f"{nombre}.csv")
    if tabla is not None:
        return tabla
    return _leer_tabla_en(CARPETA_SNAPSHOTS / f"{nombre}.csv")


def _get_bytes(sesion: requests.Session, url: str, segundos: int = 60) -> bytes:
    respuesta = sesion.get(url, timeout=(10, segundos))
    respuesta.raise_for_status()
    return respuesta.content


def descargar_gpr(sesion: requests.Session, avisos: list[str]) -> tuple[pd.DataFrame | None, str]:
    """Índice GPR de Caldara e Iacoviello.

    El diario (GPRD) se puede usar el mismo día: es un conteo de noticias
    de esa fecha, como el VIX. Si el archivo diario no responde, se usa el
    mensual y entra el primer día hábil del mes siguiente.
    """
    url_diaria = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
    url_mensual = "https://www.matteoiacoviello.com/gpr_files/data_gpr_export.xls"
    try:
        contenido = _get_bytes(sesion, url_diaria, segundos=90)
        tabla = pd.read_excel(io.BytesIO(contenido))
        if "date" not in tabla.columns or "GPRD" not in tabla.columns:
            raise RuntimeError("el archivo diario no trae columnas date y GPRD")
        fechas = pd.to_datetime(tabla["date"], errors="coerce")
        valores = pd.to_numeric(tabla["GPRD"], errors="coerce")
        salida = pd.DataFrame({"fecha_disponible": fechas, "gpr": valores})
        salida = salida.dropna()
        if len(salida) < 1000:
            raise RuntimeError("el GPR diario llegó casi vacío")
        _guardar_tabla("gpr_publicado", salida)
        return salida.sort_values("fecha_disponible"), (
            "GPR diario de Caldara-Iacoviello (GPRD), usable el mismo día"
        )
    except Exception as error:  # noqa: BLE001
        avisos.append(
            f"El GPR diario (Caldara-Iacoviello) no respondió ({error}). "
            "Intento el índice mensual."
        )
    try:
        contenido = _get_bytes(sesion, url_mensual, segundos=90)
        tabla = pd.read_excel(io.BytesIO(contenido))
        columnas = {str(columna).strip().lower(): columna for columna in tabla.columns}
        col_fecha = columnas.get("month") or columnas.get("date")
        col_gpr = columnas.get("gpr")
        if col_fecha is None or col_gpr is None:
            raise RuntimeError("el archivo mensual no trae month y GPR")
        fechas = pd.to_datetime(tabla[col_fecha], errors="coerce")
        valores = pd.to_numeric(tabla[col_gpr], errors="coerce")
        salida = pd.DataFrame({"mes": fechas, "gpr": valores}).dropna()
        salida["fecha_disponible"] = salida["mes"].map(fecha_publicacion_mes_siguiente)
        salida = salida[["fecha_disponible", "gpr"]]
        if len(salida) < 100:
            raise RuntimeError("el GPR mensual llegó casi vacío")
        _guardar_tabla("gpr_publicado", salida)
        return salida.sort_values("fecha_disponible"), (
            "GPR mensual de Caldara-Iacoviello (respaldo), "
            "usable el primer día hábil del mes siguiente"
        )
    except Exception as error:  # noqa: BLE001
        avisos.append(f"Tampoco respondió el GPR mensual ({error}).")
        cache = _leer_tabla_cache("gpr_publicado")
        if cache is not None:
            avisos.append("Uso la copia guardada del GPR.")
            return cache, "GPR, copia guardada en data/raw"
        return None, ""


def descargar_gld_tenencias(sesion: requests.Session, avisos: list[str]) -> tuple[pd.DataFrame | None, str]:
    """Toneladas de oro en el fondo GLD.

    Primero el archivo histórico de SPDR, que trae las toneladas. Si no
    responde, el historial de NAV de SSGA: ahí no vienen toneladas, así que
    se usan las acciones en circulación como respaldo de las tenencias.
    En los dos casos el dato de la sesión D entra al día hábil siguiente.
    """
    url_archivo = (
        "https://api.spdrgoldshares.com/api/v1/historical-archive"
        "?product=gld&exchange=NYSE&lang=en"
    )
    url_nav = "https://www.ssga.com/library-content/products/fund-data/etfs/us/navhist-us-en-gld.xlsx"
    try:
        contenido = _get_bytes(sesion, url_archivo, segundos=90)
        if contenido[:2] != b"PK":
            raise RuntimeError("la respuesta no es un archivo de Excel")
        tabla = pd.read_excel(io.BytesIO(contenido), sheet_name="US GLD Historical Archive")
        fechas = pd.to_datetime(tabla["Date"], format="%d-%b-%Y", errors="coerce")
        if fechas.isna().mean() > 0.5:
            fechas = pd.to_datetime(tabla["Date"], errors="coerce")
        toneladas = pd.to_numeric(tabla["Tonnes of Gold"], errors="coerce")
        salida = pd.DataFrame(
            {
                "fecha_corte": fechas,
                "gld_toneladas": toneladas,
            }
        ).dropna()
        salida = salida.loc[salida["gld_toneladas"] > 0]
        if len(salida) < 1000:
            raise RuntimeError("el archivo de GLD llegó casi vacío")
        salida["fecha_disponible"] = salida["fecha_corte"].map(fecha_publicacion_gld)
        salida = salida[["fecha_disponible", "fecha_corte", "gld_toneladas"]]
        _guardar_tabla("gld_publicado", salida)
        return salida.sort_values("fecha_disponible"), (
            "SPDR Gold Shares, archivo histórico (toneladas en el fideicomiso), "
            "usable al día hábil siguiente"
        )
    except Exception as error:  # noqa: BLE001
        avisos.append(
            f"El archivo histórico de GLD (SPDR) no respondió ({error}). "
            "Intento el historial de NAV de SSGA."
        )
    try:
        contenido = _get_bytes(sesion, url_nav, segundos=90)
        libro = pd.read_excel(io.BytesIO(contenido), header=None)
        fila_titulo = None
        for i, fila in libro.iterrows():
            textos = [str(valor).strip().lower() for valor in fila.tolist()]
            if "date" in textos and any("shares" in texto for texto in textos):
                fila_titulo = i
                break
        if fila_titulo is None:
            raise RuntimeError("no encontré la fila de títulos en el NAV de SSGA")
        tabla = pd.read_excel(io.BytesIO(contenido), header=int(fila_titulo))
        columnas = {str(columna).strip().lower(): columna for columna in tabla.columns}
        col_fecha = columnas.get("date")
        col_acciones = None
        for nombre, original in columnas.items():
            if "shares" in nombre:
                col_acciones = original
                break
        if col_fecha is None or col_acciones is None:
            raise RuntimeError("el NAV de SSGA no trae fecha y acciones")
        fechas = pd.to_datetime(tabla[col_fecha], errors="coerce")
        acciones = pd.to_numeric(tabla[col_acciones], errors="coerce")
        # Las acciones en circulación se mueven casi igual que las toneladas
        # (cada acción representa cerca de una décima de onza, y esa fracción
        # baja muy despacio). Se guardan en la misma columna para no dejar
        # el modelo sin tenencias, y el informe dice que es el respaldo.
        salida = pd.DataFrame({"fecha_corte": fechas, "gld_toneladas": acciones / 1_000_000.0})
        salida = salida.dropna()
        salida = salida.loc[salida["gld_toneladas"] > 0]
        if len(salida) < 500:
            raise RuntimeError("el NAV de SSGA llegó casi vacío")
        salida["fecha_disponible"] = salida["fecha_corte"].map(fecha_publicacion_gld)
        salida = salida[["fecha_disponible", "fecha_corte", "gld_toneladas"]]
        _guardar_tabla("gld_publicado", salida)
        avisos.append(
            "GLD quedó en millones de acciones en circulación (respaldo de SSGA), "
            "no en toneladas. El archivo de toneladas no respondió."
        )
        return salida.sort_values("fecha_disponible"), (
            "SSGA, acciones en circulación de GLD (respaldo; no son toneladas), "
            "usable al día hábil siguiente"
        )
    except Exception as error:  # noqa: BLE001
        avisos.append(f"Tampoco respondió el NAV de GLD ({error}).")
        cache = _leer_tabla_cache("gld_publicado")
        if cache is not None:
            avisos.append("Uso la copia guardada de las tenencias de GLD.")
            return cache, "Tenencias de GLD, copia guardada en data/raw"
        return None, ""


def descargar_cot_oro(sesion: requests.Session, avisos: list[str]) -> tuple[pd.DataFrame | None, str]:
    """Posición neta no comercial del oro en el COMEX, informe legacy de la CFTC.

    Neta = largos no comerciales − cortos no comerciales. También se guarda
    esa neta dividida por el interés abierto, para que un contrato de 2006
    y uno de 2026 se puedan comparar. La fecha del JSON es el martes de
    corte. Aquí se cambia por el viernes de publicación.
    """
    url = "https://publicreporting.cftc.gov/resource/6dca-aqww.json"
    params = {
        "$where": "cftc_contract_market_code='088691'",
        "$select": (
            "report_date_as_yyyy_mm_dd,noncomm_positions_long_all,"
            "noncomm_positions_short_all,open_interest_all"
        ),
        "$order": "report_date_as_yyyy_mm_dd",
        "$limit": 50000,
    }
    etiqueta = (
        "CFTC COT legacy, futuros de oro COMEX (código 088691), "
        "neto no comercial, usable el viernes posterior al martes de corte"
    )
    try:
        respuesta = sesion.get(url, params=params, timeout=(10, 90))
        respuesta.raise_for_status()
        filas = respuesta.json()
        if not isinstance(filas, list) or len(filas) < 100:
            raise RuntimeError("la CFTC no devolvió el historial del oro")
        registros = []
        for fila in filas:
            corte = pd.to_datetime(fila.get("report_date_as_yyyy_mm_dd"), errors="coerce")
            if pd.isna(corte):
                continue
            largo = pd.to_numeric(fila.get("noncomm_positions_long_all"), errors="coerce")
            corto = pd.to_numeric(fila.get("noncomm_positions_short_all"), errors="coerce")
            abierto = pd.to_numeric(fila.get("open_interest_all"), errors="coerce")
            if pd.isna(largo) or pd.isna(corto):
                continue
            neto = float(largo) - float(corto)
            ratio = float("nan")
            if pd.notna(abierto) and float(abierto) > 0:
                ratio = neto / float(abierto)
            registros.append(
                {
                    "fecha_disponible": fecha_publicacion_cot(corte),
                    "fecha_corte": pd.Timestamp(corte).normalize(),
                    "cot_neto": neto,
                    "cot_neto_oi": ratio,
                }
            )
        salida = pd.DataFrame(registros)
        if len(salida) < 100:
            raise RuntimeError("el COT del oro quedó casi vacío")
        _guardar_tabla("cot_publicado", salida)
        return salida.sort_values("fecha_disponible"), etiqueta
    except Exception as error:  # noqa: BLE001
        avisos.append(f"El COT de la CFTC no respondió ({error}).")
        cache = _leer_tabla_cache("cot_publicado")
        if cache is not None:
            avisos.append("Uso la copia guardada del COT.")
            return cache, "COT del oro, copia guardada en data/raw"
        return None, ""


def _sumar_toneladas_wgc(tabla_metrica: dict) -> pd.DataFrame:
    """Compra neta del trimestre, solo con países que reportaron los dos cortes.

    Si un país todavía no mandó el trimestre nuevo, su celda viene vacía.
    Sumar los que sí reportaron y restar el total anterior contaría a los
    faltantes como si hubieran vendido todo su oro. Aquí el cambio se arma
    país por país, y solo entra quien tiene los dos números.
    """
    encabezados = tabla_metrica["headers"][0]
    fechas = [None]
    for celda in encabezados[1:]:
        fechas.append(_fecha_etiqueta_trimestre(celda.get("val", "")))
    registros = []
    for i, fin in enumerate(fechas):
        if i == 0 or fin is None or fechas[i - 1] is None:
            continue
        compra = 0.0
        paises = 0
        for fila in tabla_metrica["rows"]:
            if i >= len(fila) or i - 1 >= len(fila):
                continue
            anterior = pd.to_numeric(fila[i - 1].get("val"), errors="coerce")
            actual = pd.to_numeric(fila[i].get("val"), errors="coerce")
            if pd.notna(anterior) and pd.notna(actual):
                compra += float(actual) - float(anterior)
                paises += 1
        if paises >= 20:
            registros.append(
                {"fin_trimestre": fin, "compras_bc_ton": compra, "paises": paises}
            )
    salida = pd.DataFrame(registros).drop_duplicates("fin_trimestre").sort_values("fin_trimestre")
    salida["fecha_disponible"] = salida["fin_trimestre"].map(fecha_publicacion_compras_bc)
    return salida[["fecha_disponible", "fin_trimestre", "compras_bc_ton", "paises"]]


def descargar_compras_bancos(sesion: requests.Session, avisos: list[str]) -> tuple[pd.DataFrame | None, str]:
    """Compras netas oficiales de oro, en toneladas por trimestre.

    El Excel que publica el World Gold Council en la página de reservas
    pide un permiso y aquí responde 403. El respaldo es la API pública del
    mismo consejo (el tablero de bancos centrales): suma de toneladas
    reportadas al FMI, con los ajustes que el consejo documenta. La compra
    del trimestre es el cambio de esa suma. Entra a fin del mes siguiente.
    """
    url_pagina = "https://www.gold.org/goldhub/data/gold-reserves-by-country"
    url_api = "https://fsapi.gold.org/api/cbd/v11/charts/getPage"
    etiqueta = (
        "World Gold Council, suma de reservas oficiales en toneladas "
        "(API pública del tablero; el Excel de la página respondió 403), "
        "cambio trimestral solo entre países que reportaron los dos cortes, "
        "usable a fin del mes siguiente"
    )
    try:
        pagina = sesion.get(url_pagina, timeout=(8, 20))
        enlace = None
        if pagina.ok:
            hallados = re.findall(r'href="(/download/file/\d+/Quarterly_gold[^"]+)"', pagina.text)
            if hallados:
                enlace = "https://www.gold.org" + hallados[0].replace(" ", "")
        if enlace:
            try:
                archivo = sesion.get(enlace, timeout=(8, 25))
                if archivo.status_code != 200 or archivo.content[:2] != b"PK":
                    avisos.append(
                        "El Excel trimestral del World Gold Council no se pudo bajar "
                        f"(respuesta {archivo.status_code}). Uso la API pública del mismo sitio."
                    )
                else:
                    avisos.append(
                        "El Excel del World Gold Council sí bajó, pero el modelo usa la API "
                        "pública del tablero, que trae el mismo agregado país por país."
                    )
            except Exception as error:  # noqa: BLE001
                avisos.append(
                    f"El Excel trimestral del World Gold Council no respondió ({error}). "
                    "Uso la API pública del mismo sitio."
                )
        else:
            avisos.append(
                "No encontré el enlace al Excel de reservas del World Gold Council. "
                "Uso la API pública del tablero de bancos centrales."
            )
        respuesta = sesion.get(
            url_api,
            params={
                "page": "date_range",
                "periodicity": "QTD_FULL",
                "startDate": "2000-03-31",
                "endDate": "2026-12-31",
            },
            timeout=(10, 90),
        )
        respuesta.raise_for_status()
        cuerpo = respuesta.json()
        metrica = cuerpo["chartData"]["table"]["QTD_FULL"]["gold_reserves_tns"]
        salida = _sumar_toneladas_wgc(metrica)
        if len(salida) < 40:
            raise RuntimeError("la suma trimestral de reservas quedó corta")
        _guardar_tabla("compras_bc_publicado", salida)
        return salida.sort_values("fecha_disponible"), etiqueta
    except Exception as error:  # noqa: BLE001
        avisos.append(f"La API de reservas del World Gold Council no respondió ({error}).")
        cache = _leer_tabla_cache("compras_bc_publicado")
        if cache is not None:
            avisos.append("Uso la copia guardada de las compras de bancos centrales.")
            return cache, "Compras de bancos centrales, copia guardada en data/raw"
        return None, ""


def construir_panel(
    series: dict[str, pd.Series],
    tabla_cpi: pd.DataFrame,
    publicadas: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """Pone todas las columnas en los días en que hubo precio del oro.

    El oro manda el calendario. Un dato diario que falta (feriado) se copia
    hacia adelante hasta 5 días de mercado. El IPC no se rellena así: entra
    solo desde su fecha de publicación.
    """
    if "oro_usd" not in series or series["oro_usd"].empty:
        raise RuntimeError("No hay precio del oro para armar el calendario.")
    oro = series["oro_usd"]
    panel = pd.DataFrame(index=oro.index.copy())
    panel.index.name = "fecha"
    for nombre, serie in series.items():
        if nombre == "cpi":
            continue
        panel[nombre] = serie.reindex(panel.index)

    cpi_diario = pegar_cpi_sin_adelanto(panel.index, tabla_cpi)
    cpi_diario = cpi_diario.drop_duplicates("fecha", keep="last").set_index("fecha")
    panel["cpi_indice"] = cpi_diario["cpi_indice"].reindex(panel.index)
    panel["cpi_anual"] = cpi_diario["cpi_anual"].reindex(panel.index)
    panel["cpi_mes"] = cpi_diario["mes_referencia"].reindex(panel.index)
    panel["cpi_fecha_publicacion"] = cpi_diario["fecha_disponible"].reindex(panel.index)

    # Series que no son diarias: entran por su fecha de publicación y se
    # quedan vigentes hasta el siguiente dato. No pasan por el relleno de
    # 5 días, porque un trimestre o un COT vive más que eso, y porque
    # rellenar desde la fecha de corte (y no desde la de publicación)
    # adelantaría el dato.
    no_rellenar = {
        "oro_usd",
        "cpi_indice",
        "cpi_anual",
        "cpi_mes",
        "cpi_fecha_publicacion",
    }
    for tabla in (publicadas or {}).values():
        if tabla is None or len(tabla) == 0 or "fecha_disponible" not in tabla.columns:
            continue
        columnas = [columna for columna in tabla.columns if columna != "fecha_disponible"]
        pegado = pegar_publicado(panel.index, tabla, columnas)
        pegado = pegado.drop_duplicates("fecha", keep="last").set_index("fecha")
        for columna in columnas:
            panel[columna] = pegado[columna].reindex(panel.index)
            no_rellenar.add(columna)

    # El breakeven de mercado es, en corto, tasa nominal menos tasa real.
    # Si FRED no dio T10YIE, lo calculamos. No lo uses junto con las dos
    # tasas en la misma regresión: sería contar lo mismo dos veces.
    if panel.get("breakeven_10") is None or panel["breakeven_10"].isna().all():
        if {"rendimiento_nominal_10", "rendimiento_real_10"} <= set(panel.columns):
            panel["breakeven_10"] = (
                panel["rendimiento_nominal_10"] - panel["rendimiento_real_10"]
            )

    columnas_diarias = [columna for columna in panel.columns if columna not in no_rellenar]
    for columna in columnas_diarias:
        panel[columna] = rellenar_hacia_adelante(panel[columna], LIMITE_RELLENO)

    panel = panel.sort_index()
    panel = panel.loc[panel["oro_usd"].notna()].copy()
    return panel


def _corregir_escala_tnx(serie: pd.Series, avisos: list[str]) -> pd.Series:
    """^TNX a veces se cotiza como el rendimiento × 10 (45 en vez de 4.5)."""
    if serie is None or serie.empty:
        return serie
    if float(serie.median()) > 20:
        avisos.append("Yahoo ^TNX venía multiplicado por 10. Lo dividí para dejarlo en por ciento.")
        return serie / 10.0
    return serie


def actualizar_datos() -> dict:
    """Baja todo, arma el panel diario y lo guarda en data/panel_diario.csv."""
    CARPETA_DATOS.mkdir(parents=True, exist_ok=True)
    CARPETA_CRUDOS.mkdir(parents=True, exist_ok=True)
    fin = (pd.Timestamp.today().normalize() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    avisos: list[str] = []
    fuentes: dict[str, dict[str, str]] = {}
    sesion = _sesion()

    _aviso("Descargando precios de Yahoo Finance (yfinance)...")
    yahoo: dict[str, pd.Series] = {}
    for nombre, ticker in TICKERS_YAHOO.items():
        try:
            serie = descargar_yahoo_uno(ticker, INICIO, fin)
            _guardar_serie(f"yahoo_{nombre}", serie)
            yahoo[nombre] = serie
            _aviso(f"  {ticker}: {len(serie)} días, hasta {serie.index.max().date()}")
        except Exception as error:  # noqa: BLE001
            cache = _leer_serie_cache(f"yahoo_{nombre}", solo_positivos=True)
            if cache is None:
                avisos.append(f"Falló {ticker}: {error}")
                _aviso(f"  {ticker}: falló ({error})")
            else:
                yahoo[nombre] = cache
                avisos.append(f"Falló {ticker}. Uso la copia guardada. {error}")
                _aviso(f"  {ticker}: uso copia guardada")

    if "tnx_yahoo" in yahoo:
        yahoo["tnx_yahoo"] = _corregir_escala_tnx(yahoo["tnx_yahoo"], avisos)

    if "oro_usd" not in yahoo or yahoo["oro_usd"].empty:
        raise RuntimeError(
            "No pude obtener el precio del oro (GC=F) ni hay una copia en data/raw. "
            "Revisa la conexión y vuelve a correr python main.py."
        )

    _aviso("Probando FRED (un intento corto)...")
    fred: dict[str, pd.Series] = {}
    prueba = descargar_fred(sesion, "DFII10", INICIO, fin)
    fred_vivo = prueba is not None
    if fred_vivo:
        fred["rendimiento_real_10"] = prueba
        _aviso("  FRED respondió. Bajo el resto de las series.")
        for nombre, serie_id in SERIES_FRED.items():
            if nombre == "rendimiento_real_10":
                continue
            inicio_serie = INICIO_IPC if nombre == "cpi" else INICIO
            bajada = descargar_fred(sesion, serie_id, inicio_serie, fin)
            if bajada is not None:
                fred[nombre] = bajada
                _aviso(f"  FRED {serie_id}: {len(bajada)} observaciones")
    else:
        avisos.append(
            "FRED no contestó a tiempo. Uso respaldos públicos: Tesoro de EE.UU., "
            "BLS, Fed de Nueva York, CBOE y Yahoo Finance."
        )
        _aviso("  FRED no contestó. Sigo con los respaldos.")

    _aviso("Descargando respaldos macro (Tesoro, BLS, Fed de NY, CBOE)...")
    real_tesoro = descargar_tesoro(sesion, real=True, anios=10, inicio=INICIO, fin=fin, avisos=avisos)
    nominal_10 = descargar_tesoro(sesion, real=False, anios=10, inicio=INICIO, fin=fin, avisos=avisos)
    nominal_2 = descargar_tesoro(sesion, real=False, anios=2, inicio=INICIO, fin=fin, avisos=avisos)
    cpi = fred.get("cpi")
    cpi_es_fred = cpi is not None
    if cpi is None:
        cpi = descargar_cpi_bls(sesion, avisos)
    vix = _elegir(fred.get("vix"), descargar_vix_cboe(sesion, avisos), minimo=100)
    if vix is None:
        vix = yahoo.get("vix_yahoo")
    fed = _elegir(fred.get("fed_funds"), descargar_fed_funds(sesion, INICIO, fin, avisos), minimo=100)

    series: dict[str, pd.Series] = {}

    def poner(nombre: str, serie: pd.Series | None, etiqueta: str, url: str) -> None:
        if serie is None or serie.empty:
            avisos.append(f"Sin datos para {nombre}.")
            return
        serie = serie.loc[INICIO:] if nombre != "cpi" else serie
        series[nombre] = serie
        fuentes[nombre] = {"etiqueta": etiqueta, "url": url}
        _guardar_serie(nombre, serie)

    poner("oro_usd", yahoo.get("oro_usd"), "Yahoo Finance GC=F (futuro COMEX)", "https://finance.yahoo.com/quote/GC=F")
    poner("gld", yahoo.get("gld"), "Yahoo Finance GLD", "https://finance.yahoo.com/quote/GLD")
    poner(
        "dolar_index",
        yahoo.get("dolar_index"),
        "Yahoo Finance DX-Y.NYB (índice dólar ICE)",
        "https://finance.yahoo.com/quote/DX-Y.NYB",
    )
    if "dolar_amplio" in fred:
        poner("dolar_amplio", fred["dolar_amplio"], "FRED DTWEXBGS", "https://fred.stlouisfed.org/series/DTWEXBGS")

    real = _elegir(fred.get("rendimiento_real_10"), real_tesoro, minimo=100)
    poner(
        "rendimiento_real_10",
        real,
        _etiqueta_fred("DFII10", fred.get("rendimiento_real_10") is not None, "Tesoro de EE.UU., curva real 10 años"),
        "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/",
    )
    poner(
        "rendimiento_nominal_10",
        _elegir(fred.get("rendimiento_nominal_10"), nominal_10, minimo=100),
        _etiqueta_fred("DGS10", fred.get("rendimiento_nominal_10") is not None, "Tesoro de EE.UU., nominal 10 años"),
        "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/",
    )
    poner(
        "rendimiento_nominal_2",
        _elegir(fred.get("rendimiento_nominal_2"), nominal_2, minimo=100),
        _etiqueta_fred("DGS2", fred.get("rendimiento_nominal_2") is not None, "Tesoro de EE.UU., nominal 2 años"),
        "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/",
    )
    if fred.get("breakeven_10") is not None:
        poner("breakeven_10", fred["breakeven_10"], "FRED T10YIE", "https://fred.stlouisfed.org/series/T10YIE")

    poner(
        "fed_funds",
        fed,
        _etiqueta_fred("DFF", fred.get("fed_funds") is not None, "Fed de Nueva York, EFFR"),
        "https://www.newyorkfed.org/markets/reference-rates/effr",
    )
    vix_etiqueta = "FRED VIXCLS" if fred.get("vix") is not None else "CBOE VIX History"
    if fred.get("vix") is None and vix is yahoo.get("vix_yahoo"):
        vix_etiqueta = "Yahoo Finance ^VIX"
    poner("vix", vix, vix_etiqueta, "https://www.cboe.com/tradable_products/vix/")
    if "gvz" in yahoo:
        poner(
            "gvz",
            yahoo.get("gvz"),
            "Yahoo Finance ^GVZ (volatilidad implícita del oro, CBOE)",
            "https://www.cboe.com/us/indices/dashboard/gvz/",
        )

    if fred.get("brent_spot") is not None:
        poner("brent", fred["brent_spot"], "FRED DCOILBRENTEU (Brent spot)", "https://fred.stlouisfed.org/series/DCOILBRENTEU")
    else:
        poner("brent", yahoo.get("brent"), "Yahoo Finance BZ=F (futuro Brent)", "https://finance.yahoo.com/quote/BZ=F")

    poner("sp500", yahoo.get("sp500"), "Yahoo Finance ^GSPC", "https://finance.yahoo.com/quote/%5EGSPC")
    poner("plata", yahoo.get("plata"), "Yahoo Finance SI=F", "https://finance.yahoo.com/quote/SI=F")
    poner("usd_mxn", yahoo.get("usd_mxn"), "Yahoo Finance MXN=X (pesos por dólar)", "https://finance.yahoo.com/quote/MXN=X")
    if fred.get("usd_mxn_fred") is not None:
        poner("usd_mxn_fred", fred["usd_mxn_fred"], "FRED DEXMXUS", "https://fred.stlouisfed.org/series/DEXMXUS")
    if "tnx_yahoo" in yahoo:
        poner("tnx_yahoo", yahoo["tnx_yahoo"], "Yahoo Finance ^TNX (chequeo de la tasa a 10 años)", "https://finance.yahoo.com/quote/%5ETNX")

    _aviso("Descargando GPR, tenencias de GLD, COT y compras de bancos centrales...")
    publicadas: dict[str, pd.DataFrame] = {}
    gpr_tabla, gpr_etiqueta = descargar_gpr(sesion, avisos)
    if gpr_tabla is not None:
        publicadas["gpr"] = gpr_tabla
        fuentes["gpr"] = {
            "etiqueta": gpr_etiqueta,
            "url": "https://www.matteoiacoviello.com/gpr.htm",
        }
        _aviso(f"  GPR: {len(gpr_tabla)} publicaciones, hasta {pd.Timestamp(gpr_tabla['fecha_disponible'].max()).date()}")
    gld_tabla, gld_etiqueta = descargar_gld_tenencias(sesion, avisos)
    if gld_tabla is not None:
        publicadas["gld_toneladas"] = gld_tabla
        fuentes["gld_toneladas"] = {
            "etiqueta": gld_etiqueta,
            "url": "https://www.spdrgoldshares.com/usa/historical-data/",
        }
        _aviso(
            f"  GLD: {len(gld_tabla)} publicaciones, "
            f"hasta {pd.Timestamp(gld_tabla['fecha_disponible'].max()).date()}"
        )
    cot_tabla, cot_etiqueta = descargar_cot_oro(sesion, avisos)
    if cot_tabla is not None:
        publicadas["cot_neto"] = cot_tabla
        fuentes["cot_neto"] = {
            "etiqueta": cot_etiqueta,
            "url": "https://publicreporting.cftc.gov/resource/6dca-aqww.json",
        }
        _aviso(
            f"  COT: {len(cot_tabla)} publicaciones, "
            f"hasta {pd.Timestamp(cot_tabla['fecha_disponible'].max()).date()}"
        )
    bc_tabla, bc_etiqueta = descargar_compras_bancos(sesion, avisos)
    if bc_tabla is not None:
        publicadas["compras_bc_ton"] = bc_tabla
        fuentes["compras_bc_ton"] = {
            "etiqueta": bc_etiqueta,
            "url": "https://fsapi.gold.org/api/cbd/v11/charts/getPage",
        }
        _aviso(
            f"  Bancos centrales: {len(bc_tabla)} trimestres, "
            f"hasta {pd.Timestamp(bc_tabla['fecha_disponible'].max()).date()}"
        )

    if not cpi_es_fred:
        fuentes["cpi"] = {
            "etiqueta": "BLS CUSR0000SA0 (IPC-U ajustado; equivalente a FRED CPIAUCSL)",
            "url": "https://api.bls.gov/publicAPI/v2/timeseries/data/",
        }
    else:
        fuentes["cpi"] = {"etiqueta": "FRED CPIAUCSL", "url": "https://fred.stlouisfed.org/series/CPIAUCSL"}

    tabla_cpi = tabla_publicacion_cpi(cpi)
    panel = construir_panel(series, tabla_cpi, publicadas)
    if "breakeven_10" in panel.columns and "breakeven_10" not in fuentes:
        fuentes["breakeven_10"] = {
            "etiqueta": "Nominal 10 años menos real 10 años (aproxima a T10YIE)",
            "url": "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/",
        }

    fechas = {}
    for nombre, serie in series.items():
        if len(serie):
            fechas[nombre] = str(serie.index.max().date())
    if len(tabla_cpi):
        ultima_cpi = tabla_cpi.iloc[-1]
        fechas["cpi_mes"] = str(pd.Timestamp(ultima_cpi["mes_referencia"]).date())
        fechas["cpi_fecha_publicacion"] = str(pd.Timestamp(ultima_cpi["fecha_disponible"]).date())
    for nombre, tabla in publicadas.items():
        if tabla is None or tabla.empty:
            continue
        fechas[nombre] = str(pd.Timestamp(tabla["fecha_disponible"].max()).date())

    panel.to_csv(CARPETA_DATOS / "panel_diario.csv", index_label="fecha")
    if CARPETA_SNAPSHOTS.exists():
        manifest = {
            "nota": (
                "Copias de series que a veces no responden (GPR, GLD, COT, bancos centrales). "
                "No son datos inventados: salen de la descarga de esta corrida. "
                "La fecha de cada fila es fecha_disponible, no la fecha de corte."
            ),
            "generado_en": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"),
            "fuentes": {
                nombre: fuentes[nombre]
                for nombre in ("gpr", "gld_toneladas", "cot_neto", "compras_bc_ton", "gvz")
                if nombre in fuentes
            },
        }
        (CARPETA_SNAPSHOTS / "MANIFEST.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (CARPETA_DATOS / "fuentes.json").write_text(
        json.dumps({"fuentes": fuentes, "fechas": fechas, "avisos": avisos}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _aviso(
        f"Panel listo: {len(panel)} días de oro, "
        f"de {panel.index.min().date()} a {panel.index.max().date()}."
    )
    return {"panel": panel, "fuentes": fuentes, "fechas": fechas, "avisos": avisos}
