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


def filas_desde_puntos_bls(puntos: list[dict]) -> list[tuple[pd.Timestamp, float]]:
    """Lee los meses del IPC que el BLS ya publicó.

    El mes que todavía no sale llega con ``value`` igual a ``"-"`` (a veces
    vacío). No es un fallo de la descarga: se salta y se queda el resto.
    ``M13`` es el promedio anual y no entra al índice mensual.
    """
    filas: list[tuple[pd.Timestamp, float]] = []
    for punto in puntos:
        periodo = str(punto.get("period", ""))
        if not periodo.startswith("M") or periodo == "M13":
            continue
        try:
            mes = int(periodo[1:])
            anio = int(punto["year"])
        except (TypeError, ValueError, KeyError):
            continue
        if not 1 <= mes <= 12:
            continue
        valor = pd.to_numeric(str(punto.get("value", "")).strip(), errors="coerce")
        if pd.isna(valor) or float(valor) <= 0:
            continue
        filas.append((pd.Timestamp(year=anio, month=mes, day=1), float(valor)))
    return filas


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
            filas.extend(filas_desde_puntos_bls(cuerpo["Results"]["series"][0]["data"]))
            time.sleep(0.2)
        if not filas:
            raise RuntimeError("El BLS respondió, pero ningún mes traía un IPC numérico.")
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


def construir_panel(
    series: dict[str, pd.Series],
    tabla_cpi: pd.DataFrame,
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

    # El breakeven de mercado es, en corto, tasa nominal menos tasa real.
    # Si FRED no dio T10YIE, lo calculamos. No lo uses junto con las dos
    # tasas en la misma regresión: sería contar lo mismo dos veces.
    if panel.get("breakeven_10") is None or panel["breakeven_10"].isna().all():
        if {"rendimiento_nominal_10", "rendimiento_real_10"} <= set(panel.columns):
            panel["breakeven_10"] = (
                panel["rendimiento_nominal_10"] - panel["rendimiento_real_10"]
            )

    columnas_diarias = [
        columna
        for columna in panel.columns
        if columna not in {"oro_usd", "cpi_indice", "cpi_anual", "cpi_mes", "cpi_fecha_publicacion"}
    ]
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

    if not cpi_es_fred:
        fuentes["cpi"] = {
            "etiqueta": "BLS CUSR0000SA0 (IPC-U ajustado; equivalente a FRED CPIAUCSL)",
            "url": "https://api.bls.gov/publicAPI/v2/timeseries/data/",
        }
    else:
        fuentes["cpi"] = {"etiqueta": "FRED CPIAUCSL", "url": "https://fred.stlouisfed.org/series/CPIAUCSL"}

    tabla_cpi = tabla_publicacion_cpi(cpi)
    panel = construir_panel(series, tabla_cpi)
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

    panel.to_csv(CARPETA_DATOS / "panel_diario.csv", index_label="fecha")
    (CARPETA_DATOS / "fuentes.json").write_text(
        json.dumps({"fuentes": fuentes, "fechas": fechas, "avisos": avisos}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _aviso(
        f"Panel listo: {len(panel)} días de oro, "
        f"de {panel.index.min().date()} a {panel.index.max().date()}."
    )
    return {"panel": panel, "fuentes": fuentes, "fechas": fechas, "avisos": avisos}
