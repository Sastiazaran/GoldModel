"""Comprobaciones que no necesitan internet.

Corre: python pruebas.py

Sirven para ver que el IPC no se adelanta, que el pronóstico no entrena con
el futuro, y que los pesos por gramo salen de la fórmula que dice el informe.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from datos import (
    extraer_cierre,
    fecha_publicacion_compras_bc,
    fecha_publicacion_cot,
    fecha_publicacion_cpi,
    fecha_publicacion_gld,
    pegar_cpi_sin_adelanto,
    pegar_publicado,
    rellenar_hacia_adelante,
    tabla_publicacion_cpi,
    variacion_anual_ipc,
)
from modelos import (
    ajustar_pronosticos,
    ajustar_valor_justo,
    columnas_pronostico,
    evaluar_cointegracion,
    indices_de_entrenamiento,
    metricas_pronostico,
    onza_a_mxn_por_gramo,
    preparar_muestra,
    pronostico_walk_forward,
    retorno_log_adelante,
    retorno_log_pasado,
    valor_justo_ventana,
    vida_media,
)


def probar_publicacion_cae_en_dia_habil() -> None:
    # 15 de febrero de 2020 fue sábado → lunes 17.
    assert fecha_publicacion_cpi(pd.Timestamp("2020-01-01")) == pd.Timestamp("2020-02-17")
    # 15 de marzo de 2020 fue domingo → lunes 16.
    assert fecha_publicacion_cpi(pd.Timestamp("2020-02-01")) == pd.Timestamp("2020-03-16")
    # 15 de abril de 2020 fue miércoles y se queda.
    assert fecha_publicacion_cpi(pd.Timestamp("2020-03-01")) == pd.Timestamp("2020-04-15")


def probar_variacion_anual_mira_atras() -> None:
    indice = pd.date_range("2019-01-01", periods=14, freq="MS")
    serie = pd.Series(np.arange(100, 114), index=indice, dtype=float)
    variacion = variacion_anual_ipc(serie)
    assert pd.isna(variacion.iloc[11])
    assert abs(variacion.iloc[12] - (112 / 100 - 1)) < 1e-12


def probar_pegar_cpi_aunque_las_fechas_traigan_otra_resolucion() -> None:
    # Pandas 3 rechaza merge_asof si un lado está en segundos y el otro en microsegundos.
    tabla = pd.DataFrame(
        {
            "mes_referencia": pd.to_datetime(["2019-12-01"]).as_unit("us"),
            "cpi_indice": [100.0],
            "cpi_anual": [2.0],
            "fecha_disponible": pd.to_datetime(["2020-01-15"]).as_unit("us"),
        }
    )
    fechas = pd.to_datetime(["2020-01-14", "2020-01-15"]).as_unit("s")
    unido = pegar_cpi_sin_adelanto(fechas, tabla)
    assert pd.isna(unido.loc[unido["fecha"] == "2020-01-14", "cpi_indice"].iloc[0])
    assert unido.loc[unido["fecha"] == "2020-01-15", "cpi_indice"].iloc[0] == 100.0


def probar_ipc_no_se_adelanta() -> None:
    meses = pd.date_range("2018-01-01", periods=30, freq="MS")
    serie = pd.Series(np.linspace(100, 130, len(meses)), index=meses)
    tabla = tabla_publicacion_cpi(serie)
    fila = tabla.loc[tabla["mes_referencia"] == pd.Timestamp("2020-01-01")].iloc[0]
    assert fila["fecha_disponible"] == pd.Timestamp("2020-02-17")
    fechas = pd.to_datetime(["2020-02-01", "2020-02-16", "2020-02-17"])
    unido = pegar_cpi_sin_adelanto(fechas, tabla)
    # El 1 y el 16 de febrero todavía no salió el IPC de enero.
    assert unido.loc[unido["fecha"] == "2020-02-01", "mes_referencia"].iloc[0] == pd.Timestamp("2019-12-01")
    assert unido.loc[unido["fecha"] == "2020-02-16", "mes_referencia"].iloc[0] == pd.Timestamp("2019-12-01")
    assert unido.loc[unido["fecha"] == "2020-02-17", "mes_referencia"].iloc[0] == pd.Timestamp("2020-01-01")


def probar_relleno_no_mira_al_futuro() -> None:
    serie = pd.Series([1.0, np.nan, np.nan, np.nan])
    rellena = rellenar_hacia_adelante(serie, 2)
    assert rellena.iloc[1] == 1 and rellena.iloc[2] == 1 and pd.isna(rellena.iloc[3])
    futura = pd.Series([np.nan, 5.0])
    rellena_futura = rellenar_hacia_adelante(futura, 5)
    assert pd.isna(rellena_futura.iloc[0]) and rellena_futura.iloc[1] == 5


def probar_retornos_pasan_y_adelante() -> None:
    precios = pd.Series([100.0, 101.0, 102.0, 104.0, 108.0, 100.0])
    pasado = retorno_log_pasado(precios, 2)
    adelante = retorno_log_adelante(precios, 2)
    assert pd.isna(pasado.iloc[0]) and pd.isna(pasado.iloc[1])
    assert abs(pasado.iloc[2] - np.log(102 / 100)) < 1e-12
    assert pd.isna(adelante.iloc[-1]) and pd.isna(adelante.iloc[-2])
    assert abs(adelante.iloc[0] - np.log(102 / 100)) < 1e-12
    assert abs(adelante.iloc[3] - np.log(100 / 104)) < 1e-12


def probar_entrenamiento_no_incluye_el_futuro() -> None:
    validas = np.ones(100, dtype=bool)
    indices = indices_de_entrenamiento(100, t=50, horizonte=20, fila_valida=validas)
    assert indices.max() == 30
    assert 50 not in indices
    assert 31 not in indices
    vacio = indices_de_entrenamiento(100, t=10, horizonte=20, fila_valida=validas)
    assert len(vacio) == 0


def probar_pesos_por_gramo() -> None:
    # 4131.02 USD/oz × 18.0899 MXN/USD ÷ 31.1034768 ≈ 2402.6 MXN/g (nota del 5 oct 2026).
    pesos = onza_a_mxn_por_gramo(4131.02, 18.0899)
    assert abs(pesos - 2402.6) < 0.15


def probar_r2_de_la_caminata_aleatoria() -> None:
    y = np.array([0.10, -0.20, 0.05, 0.02])
    pred = np.zeros(4)
    media = np.full(4, y.mean())
    metricas = metricas_pronostico(y, pred, media, horizonte=1)
    assert abs(metricas["sin_solapar"]["r2_cero"]) < 1e-12
    perfecto = metricas_pronostico(y, y.copy(), media, horizonte=1)
    assert abs(perfecto["sin_solapar"]["r2_cero"] - 1) < 1e-12
    assert abs(perfecto["sin_solapar"]["rmse"]) < 1e-12


def probar_acierto_contra_siempre_alza() -> None:
    y = np.array([0.1, -0.2, 0.3])
    pred = np.array([0.01, 0.01, 0.01])
    media = np.array([0.02, 0.02, 0.02])
    metricas = metricas_pronostico(y, pred, media, horizonte=1)["sin_solapar"]
    assert abs(metricas["acierto"] - 2 / 3) < 1e-12
    assert abs(metricas["acierto_siempre_alza"] - 2 / 3) < 1e-12


def probar_valor_justo_con_serie_falsa() -> None:
    n = 900
    indice = pd.bdate_range("2016-01-04", periods=n)
    azar = np.random.default_rng(0)
    real = azar.normal(1.0, 0.3, n)
    dolar = np.full(n, 100.0) + azar.normal(0, 0.2, n).cumsum()
    brent = np.clip(60 + azar.normal(0, 0.4, n).cumsum(), 20, None)
    vix = np.clip(azar.normal(18, 2, n), 10, 50)
    log_oro = 7.2 - 0.04 * real - 0.2 * np.log(dolar) + azar.normal(0, 0.01, n)
    panel = pd.DataFrame(
        {
            "oro_usd": np.exp(log_oro),
            "rendimiento_real_10": real,
            "dolar_index": dolar,
            "brent": brent,
            "vix": vix,
            "cpi_anual": 2.0 + 0.4 * np.sin(np.linspace(0, 8, n)),
            "sp500": np.linspace(2000, 4000, n),
            "usd_mxn": np.full(n, 18.0),
        },
        index=indice,
    )
    valor = ajustar_valor_justo(preparar_muestra(panel))
    assert valor["n"] > 800
    assert math.isfinite(valor["brecha_pct"])
    assert math.isfinite(valor["r2"])
    assert "rendimiento_real_10" in {fila["nombre"] for fila in valor["coeficientes"]}


def probar_pronostico_del_ultimo_dia_aunque_no_haya_respuesta() -> None:
    azar = np.random.default_rng(1)
    n = 1800
    x = azar.normal(size=(n, 3))
    y = azar.normal(size=n)
    y[-20:] = np.nan
    salida = pronostico_walk_forward(
        x,
        y,
        np.ones(n, dtype=bool),
        np.isfinite(y),
        horizonte=20,
    )
    assert math.isfinite(salida["pred"][-1])
    assert math.isfinite(salida["pred"][-20])
    # El día final todavía no tiene respuesta, y aun así hay pronóstico.
    assert not math.isfinite(y[-1])


def probar_pipeline_corto_no_inventa_habilidad() -> None:
    """Con ruido puro, el ajuste tiene que correr y devolver métricas finitas."""
    azar = np.random.default_rng(2)
    n = 2600
    indice = pd.bdate_range("2014-01-02", periods=n)
    real = azar.normal(0.5, 0.2, n)
    dolar = np.clip(100 + azar.normal(0, 0.3, n).cumsum(), 70, 140)
    brent = np.clip(50 + azar.normal(0, 0.3, n).cumsum(), 20, 150)
    vix = np.clip(azar.normal(18, 3, n), 9, 60)
    oro = np.exp(7 + azar.normal(0, 0.01, n).cumsum())
    panel = pd.DataFrame(
        {
            "oro_usd": oro,
            "rendimiento_real_10": real,
            "dolar_index": dolar,
            "brent": brent,
            "vix": vix,
            "cpi_anual": 1.5 + 0.8 * np.sin(np.linspace(0, 10, n)),
            "sp500": np.exp(8 + azar.normal(0, 0.01, n).cumsum()),
            "usd_mxn": np.full(n, 17.5),
        },
        index=indice,
    )
    pronostico = ajustar_pronosticos(ajustar_valor_justo(preparar_muestra(panel))["df"])
    bloque = pronostico["por_horizonte"][20]["metricas"]["sin_solapar"]
    assert bloque["n"] > 10
    assert math.isfinite(bloque["r2_cero"])
    assert math.isfinite(pronostico["por_horizonte"][5]["pred_ultimo"])


def probar_cot_no_se_usa_antes_del_viernes() -> None:
    # 13 de octubre de 2026 fue martes. El COT de ese corte sale el viernes 16.
    assert fecha_publicacion_cot(pd.Timestamp("2026-10-13")) == pd.Timestamp("2026-10-16")
    # Un corte que no es martes (jueves 15) cae tres días después, y el domingo pasa al lunes.
    assert fecha_publicacion_cot(pd.Timestamp("2026-10-15")) == pd.Timestamp("2026-10-19")
    tabla = pd.DataFrame(
        {"fecha_disponible": [pd.Timestamp("2026-10-16")], "cot_neto": [100.0]}
    )
    fechas = pd.to_datetime(["2026-10-15", "2026-10-16"])
    pegado = pegar_publicado(fechas, tabla, ["cot_neto"])
    assert math.isnan(pegado.loc[0, "cot_neto"])
    assert pegado.loc[1, "cot_neto"] == 100.0


def probar_gld_entra_al_dia_habil_siguiente() -> None:
    # Viernes 16 de octubre de 2026: el archivo sale el lunes 19.
    assert fecha_publicacion_gld(pd.Timestamp("2026-10-16")) == pd.Timestamp("2026-10-19")
    # Lunes 12: sale el martes 13. No el mismo lunes.
    assert fecha_publicacion_gld(pd.Timestamp("2026-10-12")) == pd.Timestamp("2026-10-13")


def probar_un_pais_sin_dato_no_cuenta_como_venta() -> None:
    """Una celda vacía no es una venta de todo el oro de ese país."""
    from datos import _sumar_toneladas_wgc

    metrica = {
        "headers": [[{"val": "Country"}, {"val": "Q1 20"}, {"val": "Q2 20"}]],
        "rows": [
            [{"iso3": "AAA", "val": "Uno"}, {"val": "100"}, {"val": "110"}],
            [{"iso3": "BBB", "val": "Dos"}, {"val": "50"}, {"val": ""}],
        ],
    }
    # Con un solo país pareado la función pide al menos 20, así que se replica.
    metrica["rows"] = metrica["rows"][:1] * 20 + [metrica["rows"][1]]
    tabla = _sumar_toneladas_wgc(metrica)
    assert len(tabla) == 1
    assert abs(float(tabla["compras_bc_ton"].iloc[0]) - 200.0) < 1e-6


def probar_compras_de_bancos_entran_al_mes_siguiente() -> None:
    # Segundo trimestre de 2026 cierra el 30 de junio y se usa desde el 31 de julio.
    assert fecha_publicacion_compras_bc(pd.Timestamp("2026-06-30")) == pd.Timestamp("2026-07-31")
    assert fecha_publicacion_compras_bc(pd.Timestamp("2025-12-31")) == pd.Timestamp("2026-01-31")
    tabla = pd.DataFrame(
        {
            "fecha_disponible": [pd.Timestamp("2026-07-31")],
            "compras_bc_ton": [50.0],
        }
    )
    fechas = pd.to_datetime(["2026-07-30", "2026-07-31"])
    pegado = pegar_publicado(fechas, tabla, ["compras_bc_ton"])
    assert math.isnan(pegado.loc[0, "compras_bc_ton"])
    assert pegado.loc[1, "compras_bc_ton"] == 50.0


def probar_vida_media_solo_si_la_brecha_se_encoge() -> None:
    # alpha = -0.5 multiplica la brecha por 0.5 cada periodo: vida media = 1.
    assert abs(vida_media(-0.5) - 1.0) < 1e-9
    assert vida_media(0.0) is None
    assert vida_media(-1.0) is None
    assert vida_media(0.2) is None
    assert vida_media(float("nan")) is None


def probar_ventana_movil_no_mira_el_futuro() -> None:
    n = 800
    indice = pd.bdate_range("2018-01-02", periods=n)
    x1 = np.linspace(-1.0, 1.0, n)
    y = 0.2 + 0.5 * x1
    y_shock = y.copy()
    y_shock[-100:] += 40.0
    marco = pd.DataFrame({"constante": 1.0, "x1": x1}, index=indice)
    objetivo = pd.Series(y, index=indice)
    objetivo_shock = pd.Series(y_shock, index=indice)
    limpio = valor_justo_ventana(marco, objetivo, ventana=200, paso=40)
    sucio = valor_justo_ventana(marco, objetivo_shock, ventana=200, paso=40)
    # En la fila 600 la ventana es [400, 600). El shock vive en [700, 800).
    assert abs(limpio["log_valor"].iloc[600] - sucio["log_valor"].iloc[600]) < 1e-8
    assert abs(limpio["x1"].iloc[600] - sucio["x1"].iloc[600]) < 1e-8
    # Antes de completar la ventana no hay valor.
    assert math.isnan(limpio["log_valor"].iloc[199])
    assert math.isfinite(limpio["log_valor"].iloc[200])


def probar_variables_nuevas_no_rompen_el_valor_justo() -> None:
    n = 900
    indice = pd.bdate_range("2016-01-04", periods=n)
    azar = np.random.default_rng(4)
    real = azar.normal(1.0, 0.3, n)
    dolar = np.full(n, 100.0) + azar.normal(0, 0.2, n).cumsum()
    brent = np.clip(60 + azar.normal(0, 0.4, n).cumsum(), 20, None)
    vix = np.clip(azar.normal(18, 2, n), 10, 50)
    log_oro = 7.2 - 0.04 * real - 0.2 * np.log(dolar) + azar.normal(0, 0.01, n)
    panel = pd.DataFrame(
        {
            "oro_usd": np.exp(log_oro),
            "rendimiento_real_10": real,
            "dolar_index": dolar,
            "brent": brent,
            "vix": vix,
            "cpi_anual": 2.0 + 0.4 * np.sin(np.linspace(0, 8, n)),
            "sp500": np.linspace(2000, 4000, n),
            "usd_mxn": np.full(n, 18.0),
            "gpr": 100 + azar.normal(0, 5, n).cumsum() * 0.05,
            "gld_toneladas": np.linspace(400, 900, n),
            "compras_bc_ton": azar.normal(20, 30, n),
        },
        index=indice,
    )
    valor = ajustar_valor_justo(preparar_muestra(panel))
    ext = valor["extendido"]
    assert ext["disponible"]
    assert {"gpr", "log_gld_ton", "compras_bc_ton"} <= set(ext["variables"])
    assert math.isfinite(ext["brecha_vf_pct"])
    assert ext["cierre_abs_pp"] is not None and math.isfinite(ext["cierre_abs_pp"])
    assert math.isfinite(valor["movil"]["ventanas"]["3a"]["brecha_pct"])
    assert valor["cointegracion"]["n"] > 50


def probar_cointegracion_encuentra_una_relacion_obvia() -> None:
    azar = np.random.default_rng(5)
    n = 2200
    indice = pd.bdate_range("2012-01-02", periods=n)
    caminata = np.cumsum(azar.normal(0, 0.01, n))
    error = np.zeros(n)
    # Persistente también de una semana a la otra. Si se apaga en unos días,
    # el dato semanal ya no ve la corrección.
    for t in range(1, n):
        error[t] = 0.98 * error[t - 1] + azar.normal(0, 0.001)
    log_oro = 7.0 + 1.2 * caminata + error
    df = pd.DataFrame(
        {
            "oro_usd": np.exp(log_oro),
            "dolar_index": np.exp(4.6 + caminata),
            "brent": 60 + azar.normal(0, 0.4, n),
            "rendimiento_real_10": azar.normal(1.0, 0.05, n),
            "vix": azar.normal(18, 1, n),
        },
        index=indice,
    )
    resultado = evaluar_cointegracion(df)
    assert resultado["hay_relacion"]
    assert resultado["vida_media_semanas"] is not None
    assert resultado["vida_media_semanas"] > 0


def probar_pronostico_usa_cot_y_boosting() -> None:
    azar = np.random.default_rng(6)
    n = 420
    indice = pd.bdate_range("2020-01-02", periods=n)
    df = pd.DataFrame(index=indice)
    for columna in (
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
        "cot_neto_oi",
        "rendimiento_nominal_2",
        "d_nominal2_20",
        "y_5",
        "y_20",
    ):
        df[columna] = azar.normal(0, 0.01, n)
    assert "cot_neto_oi" in columnas_pronostico(df)
    assert "rendimiento_nominal_2" in columnas_pronostico(df)
    salida = ajustar_pronosticos(df, min_entrenamiento=120)
    bloque = salida["por_horizonte"][5]
    assert "cot_neto_oi" in salida["columnas"]
    assert math.isfinite(bloque["metricas_boosting"]["sin_solapar"]["r2_cero"])
    assert math.isfinite(bloque["pred_ultimo_boosting"])
    assert math.isfinite(bloque["metricas"]["sin_solapar"]["r2_media"])


def probar_extraer_cierre_de_yfinance() -> None:
    columnas = pd.MultiIndex.from_tuples([("Adj Close", "GC=F"), ("Close", "GC=F")])
    tabla = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=columnas)
    serie = extraer_cierre(tabla)
    assert list(serie.values) == [2.0, 4.0]


def _modulo_export():
    if getattr(_modulo_export, "cache", None) is None:
        ruta = Path(__file__).resolve().parent / "scripts" / "export_dashboard_data.py"
        spec = importlib.util.spec_from_file_location("export_dashboard_data", ruta)
        modulo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(modulo)
        _modulo_export.cache = modulo
    return _modulo_export.cache


def probar_esquema_de_reportes_coincide_con_el_validador() -> None:
    modulo = _modulo_export()
    esquema = json.loads((Path(__file__).resolve().parent / "data" / "reports" / "schema.json").read_text(encoding="utf-8"))
    assert tuple(esquema["required"]) == modulo.CAMPOS_REPORTE
    assert esquema["version"] == 1


def probar_reportes_de_octubre_no_inventan_huecos() -> None:
    modulo = _modulo_export()
    reportes = modulo.cargar_reportes()
    por_fecha = {reporte["fecha"]: reporte for reporte in reportes}
    assert list(por_fecha) == [
        "2026-10-21",
        "2026-10-22",
        "2026-10-23",
        "2026-10-24",
        "2026-10-25",
        "2026-10-26",
        "2026-10-27",
        "2026-10-28",
    ]
    hoy = por_fecha["2026-10-28"]
    assert hoy["precio_usd_oz"] == 4047
    assert hoy["mxn_por_gramo"] == 2427
    assert hoy["usd_mxn"] == 18.6525
    assert hoy["cambio_usd"] == -12.3
    assert hoy["usd_mxn_fecha"] == "2026-10-27"
    assert hoy["lectura"]
    assert hoy["drivers"][0]["url"].startswith("https://")
    for fecha in ("2026-10-24", "2026-10-25"):
        vacio = por_fecha[fecha]
        assert vacio["sesion"] == "sin_registro"
        assert vacio["precio_usd_oz"] is None
        assert vacio["usd_mxn"] is None
        assert vacio["mxn_por_gramo"] is None
        assert vacio["lectura"] is None
        assert vacio["drivers"] == []
    sucio = dict(vacio)
    sucio["precio_usd_oz"] = 4000
    assert modulo.validar_reporte(sucio)


def probar_mae_rmse_contra_el_baseline_de_no_cambio() -> None:
    modulo = _modulo_export()
    y = np.array([0.10, -0.20, 0.30])
    perfecto = modulo.errores_contra_cero(y, y.copy(), horizonte=1)
    assert perfecto["sin_solapar"]["n"] == 3
    assert abs(perfecto["sin_solapar"]["mae"]) < 1e-12
    assert abs(perfecto["sin_solapar"]["rmse"]) < 1e-12
    assert abs(perfecto["sin_solapar"]["mae_cero"] - float(np.mean(np.abs(y)))) < 1e-12
    ceros = modulo.errores_contra_cero(y, np.zeros(3), horizonte=1)
    assert abs(ceros["sin_solapar"]["mae"] - ceros["sin_solapar"]["mae_cero"]) < 1e-12
    assert abs(ceros["sin_solapar"]["rmse"] - ceros["sin_solapar"]["rmse_cero"]) < 1e-12


def probar_acierto_en_vivo_ignora_pronosticos_abiertos() -> None:
    modulo = _modulo_export()
    fechas = pd.bdate_range("2024-01-02", periods=6)
    precios = np.array([100.0, 110.0, 90.0, 120.0, 100.0, 130.0])
    y = np.array([0.10, -0.20, np.nan, 0.05, np.nan, np.nan])
    pred = np.array([0.01, -0.01, 0.02, -0.03, 0.04, 0.05])
    vivo = modulo.acierto_en_vivo(fechas, precios, y, pred, horizonte=2)
    assert vivo["n"] == 2
    assert vivo["aciertos"] == 1
    assert abs(vivo["tasa"] - 0.5) < 1e-12
    assert vivo["ultimos"][0]["acierto"] is True
    assert vivo["ultimos"][0]["direccion"] == "alza"
    assert vivo["ultimos"][1]["acierto"] is False
    assert vivo["ultimos"][1]["direccion"] == "baja"
    assert vivo["ultimos"][1]["direccion_realizada"] == "alza"
    fechas_contadas = {fila["fecha"] for fila in vivo["ultimos"]}
    assert "2024-01-08" not in fechas_contadas
    assert vivo["serie"][-1]["tasa"] == vivo["tasa"]
    # Un cero exacto no vota y no recorre la rejilla: el día siguiente se queda.
    y_cero = np.array([0.10, 0.0, -0.20])
    pred_cero = np.array([0.05, 0.05, 0.05])
    precios_cero = np.array([100.0, 100.0, 90.0])
    fechas_cero = pd.bdate_range("2024-02-01", periods=3)
    con_cero = modulo.acierto_en_vivo(fechas_cero, precios_cero, y_cero, pred_cero, horizonte=1)
    assert con_cero["n"] == 2
    assert con_cero["aciertos"] == 1
    assert con_cero["ultimos"][1]["fecha"] == "2024-02-05"


def probar_direccion_y_nivel_objetivo() -> None:
    modulo = _modulo_export()
    assert modulo.direccion_pronostico(0.01) == "alza"
    assert modulo.direccion_pronostico(-0.02) == "baja"
    assert modulo.direccion_pronostico(0.0) == "sin_cambio"
    assert abs(modulo.nivel_objetivo(100.0, math.log(1.1)) - 110.0) < 1e-9


def probar_pronostico_alineado_cae_en_el_dia_que_vence() -> None:
    modulo = _modulo_export()
    precios = np.array([100.0, 100.0, 100.0, 100.0])
    pred = np.array([math.log(1.1), np.nan, np.nan, np.nan])
    alineado = modulo.niveles_alineados(precios, pred, horizonte=2)
    assert math.isnan(alineado[0]) and math.isnan(alineado[1])
    assert abs(alineado[2] - 110.0) < 1e-9
    assert math.isnan(alineado[3])


def main() -> None:
    funciones = [valor for nombre, valor in sorted(globals().items()) if nombre.startswith("probar_")]
    fallos = 0
    for funcion in funciones:
        try:
            funcion()
        except Exception as error:  # noqa: BLE001
            fallos += 1
            print(f"FALLO  {funcion.__name__}: {error}")
        else:
            print(f"ok    {funcion.__name__}")
    if fallos:
        raise SystemExit(f"{fallos} prueba(s) fallaron")
    print(f"Todas las pruebas pasaron ({len(funciones)}).")


if __name__ == "__main__":
    main()
