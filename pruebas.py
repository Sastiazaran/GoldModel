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
        "2026-10-29",
    ]
    hoy = por_fecha["2026-10-28"]
    assert hoy["precio_usd_oz"] == 4047
    assert hoy["mxn_por_gramo"] == 2427
    assert hoy["usd_mxn"] == 18.6525
    assert hoy["cambio_usd"] == -12.3
    assert hoy["usd_mxn_fecha"] == "2026-10-27"
    assert hoy["lectura"]
    assert hoy["drivers"][0]["url"].startswith("https://")
    nuevo = por_fecha["2026-10-29"]
    assert nuevo["sesion"] == "con_registro"
    assert nuevo["precio_usd_oz"] == 4031.3
    assert nuevo["precio_aproximado"] is False
    assert nuevo["cierre_previo_usd_oz"] == 4066.6
    assert nuevo["cierre_previo_aproximado"] is True
    assert nuevo["cambio_usd"] == -35.3
    assert nuevo["cambio_pct"] == -0.87
    assert nuevo["usd_mxn"] == 18.6673
    assert nuevo["usd_mxn_fecha"] == "2026-10-28"
    assert nuevo["mxn_por_gramo"] == 2419.5
    assert nuevo["mxn_por_gramo_aproximado"] is True
    assert nuevo["notion_url"] is None
    assert nuevo["lectura"]
    assert nuevo["drivers"][0]["url"].startswith("https://")
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


def _texto_relato(relato: dict) -> str:
    return _modulo_export()._texto_relato(relato)


def _historia_de_prueba() -> list[dict]:
    fechas = pd.bdate_range("2026-01-02", periods=21)
    return [
        {"fecha": fecha.strftime("%Y-%m-%d"), "precio": round(100 + i * 0.5, 2)}
        for i, fecha in enumerate(fechas)
    ]


def _modelo_de_prueba(**cambios) -> dict:
    modelo = {
        "fecha": "2026-01-30",
        "instrumento": "GC=F",
        "precio_usd_oz": 110.0,
        "valor_justo_usd": 100.0,
        "brecha_pct": 10.0,
        "ampliado": None,
        "historia": _historia_de_prueba(),
        "pronosticos": {
            "5": {
                "direccion": "alza",
                "retorno_pct": 1.0,
                "nivel_objetivo_usd": 111.1,
                "banda_baja_pct": -1.0,
                "banda_alta_pct": 3.0,
                "senal_debil": False,
            },
            "20": {
                "direccion": "baja",
                "retorno_pct": -2.0,
                "nivel_objetivo_usd": 107.8,
                "banda_baja_pct": -6.0,
                "banda_alta_pct": 2.0,
                "senal_debil": True,
            },
        },
        "precision": {
            "5": {
                "le_gana_al_cero": False,
                "errores": {"sin_solapar": {"mae_pct": 2.5, "mae_cero_pct": 1.2}},
                "metricas": {"sin_solapar": {"r2_cero": -0.02, "acierto": 0.5, "acierto_siempre_alza": 0.6}},
                "en_vivo": {"n": 10, "aciertos": 5, "tasa": 0.5},
            },
            "20": {
                "le_gana_al_cero": False,
                "errores": {"sin_solapar": {"mae_pct": 4.0, "mae_cero_pct": 3.5}},
                "metricas": {"sin_solapar": {"r2_cero": -0.04, "acierto": 0.48, "acierto_siempre_alza": 0.55}},
                "en_vivo": {"n": 4, "aciertos": 2, "tasa": 0.5},
            },
        },
    }
    modelo.update(cambios)
    return modelo


def probar_relato_cuenta_con_los_numeros_que_recibe() -> None:
    modulo = _modulo_export()
    reportes = [
        {
            "fecha": "2026-01-30",
            "sesion": "con_registro",
            "precio_usd_oz": 108.0,
            "precio_aproximado": True,
            "cambio_usd": -1.5,
            "cambio_pct": -1.37,
            "cambio_texto": "bajó frente al cierre previo",
            "mxn_por_gramo": 70,
            "usd_mxn": 20.0,
            "usd_mxn_fecha": "2026-01-29",
            "lectura": "Sin prisas.",
            "drivers": [
                {
                    "titulo": "Petróleo",
                    "detalle": "Brent en 110 dólares según el log.",
                    "fuente": "Reuters",
                    "url": "https://example.com/brent",
                }
            ],
        }
    ]
    relato = modulo.construir_relato(_modelo_de_prueba(), reportes)
    texto = _texto_relato(relato)
    assert "110 dólares" in texto
    assert "108 dólares" in texto
    assert "subió 10.0%" in texto
    assert "9,999" not in texto
    assert "Brent en 110 dólares según el log." in json.dumps(relato, ensure_ascii=False)
    assert relato["secciones"][0]["grafica"]["puntos"][-1]["valor"] == 110.0
    assert "No tengo el cierre de 2025" in texto
    assert "No tengo 60 días de mercado" in texto


def probar_relato_no_inventa_cuando_falta_el_modelo() -> None:
    modulo = _modulo_export()
    relato = modulo.construir_relato(None, [])
    texto = _texto_relato(relato)
    assert "No tengo el cierre del futuro" in texto
    assert "No tengo valor justo" in texto
    assert "No hay reportes con registro" in texto
    assert "No tengo pronósticos ni precisión" in texto
    assert "4,051" not in texto
    assert "dólares" not in texto.split("No tengo el cierre del futuro")[0] or "108" not in texto


def probar_relato_solo_habla_de_bancos_si_el_modelo_los_trae() -> None:
    modulo = _modulo_export()
    sin_extra = modulo.construir_relato(_modelo_de_prueba(ampliado=None), [])
    texto_sin = _texto_relato(sin_extra)
    assert "No hay valor justo ampliado" in texto_sin
    assert "acercan la brecha" not in texto_sin
    con_extra = modulo.construir_relato(
        _modelo_de_prueba(
            ampliado={
                "valor_justo_usd": 105.0,
                "brecha_pct": 4.8,
                "cierre_abs_pp": 10.1,
                "variables": ["rendimiento_real_10", "compras_bc_ton", "gpr", "log_gld_ton"],
            }
        ),
        [],
    )
    texto_con = _texto_relato(con_extra)
    assert "las compras de bancos centrales" in texto_con
    assert "el riesgo geopolítico" in texto_con
    assert "acercan la brecha a cero en 10.1 puntos" in texto_con
    assert "105 dólares" in texto_con


def probar_relato_confianza_contra_el_baseline() -> None:
    modulo = _modulo_export()
    relato = modulo.construir_relato(_modelo_de_prueba(), [])
    texto = _texto_relato(relato)
    assert "se equivoca más que asumir que el precio no cambia" in texto
    assert "2.50% de error contra 1.20%" in texto
    assert "no le gana a la idea de que el precio no cambia" in texto
    assert "A 5 días de mercado la cuenta apunta al alza: +1.00%" in texto
    assert "hacia 111.10 dólares" in texto
    assert "A 20 días de mercado la cuenta apunta a la baja: -2.00%" in texto
    assert "Esa banda incluye subida y bajada." in texto
    vacio = dict(_modelo_de_prueba())
    vacio["precision"] = {
        "5": {"le_gana_al_cero": False, "errores": {"sin_solapar": {}}, "metricas": {"sin_solapar": {}}, "en_vivo": {"n": 0}},
        "20": {"le_gana_al_cero": False, "errores": {"sin_solapar": {}}, "metricas": {"sin_solapar": {}}, "en_vivo": {"n": 0}},
    }
    vacio["pronosticos"] = {}
    texto_vacio = _texto_relato(modulo.construir_relato(vacio, []))
    assert "No tengo el pronóstico a 5 días." in texto_vacio
    assert "no tengo el error contra el baseline" in texto_vacio


def probar_relato_cierra_sin_venderlo_como_consejo() -> None:
    modulo = _modulo_export()
    relato = modulo.construir_relato(_modelo_de_prueba(), [])
    cierre = relato["secciones"][-1]["parrafos"][0].lower()
    assert "no es asesoría financiera" in cierre
    assert "compra en partes" in cierre
    assert relato["secciones"][-1]["id"] == "cierre"
    assert "no le gana a pensar que el precio no cambia" in cierre


def probar_purga_deja_un_horizonte_de_embargo() -> None:
    from evaluacion import bloques_cv_purgada, indices_purgados, limite_pasado, mascara_entrenamiento_ambos_lados

    assert limite_pasado(50, 20, 20) == 9
    indices = indices_purgados(100, t=50, horizonte=20, fila_valida=np.ones(100, dtype=bool), embargo=20)
    assert indices.max() == 9
    assert 10 not in indices
    bloques = bloques_cv_purgada(400, horizonte=10, n_bloques=4, embargo=10, min_entrenamiento=80)
    assert bloques
    for bloque in bloques:
        assert bloque["train"].max() <= limite_pasado(bloque["test_inicio"], 10, 10)
        assert bloque["train"].max() < bloque["test"][0]
    mascara = mascara_entrenamiento_ambos_lados(200, 80, 99, horizonte=5, embargo=5)
    assert not mascara[90]
    assert mascara[0]
    assert mascara[199]
    # 109 todavía está dentro del embargo; 110 ya quedó del otro lado.
    assert not mascara[109]
    assert mascara[110]


def probar_diebold_mariano_prefiere_al_pronostico_perfecto() -> None:
    from evaluacion import diebold_mariano

    y = np.array([0.2, -0.1, 0.05, -0.3, 0.15, 0.4, -0.2, 0.1, -0.05, 0.3, 0.12, -0.18])
    perfecto = diebold_mariano(y, y.copy(), np.zeros(len(y)), horizonte=1)
    assert perfecto["sin_solapar"]["modelo_mejor"] is True
    assert perfecto["sin_solapar"]["p"] is not None and perfecto["sin_solapar"]["p"] < 0.05
    igual = diebold_mariano(y, np.zeros(len(y)), np.zeros(len(y)), horizonte=1)
    assert abs(igual["sin_solapar"]["media_diferencial"]) < 1e-12
    # Con horizonte 5 la varianza de Newey-West tiene que salir un número, no tronar.
    largo = np.linspace(-0.02, 0.03, 40)
    solapado = diebold_mariano(largo, largo * 0.2, np.zeros(len(largo)), horizonte=5)
    assert solapado["solapado"]["n"] == 40
    assert solapado["solapado"]["estadistico"] is not None


def probar_dummie_de_febrero_2022_no_se_adelanta() -> None:
    from evaluacion import fecha_no_adelanta
    from valor_nuevo import FECHA_QUIEBRE, dummy_post_feb2022

    fechas = pd.to_datetime(["2022-02-23", "2022-02-24", "2022-03-01"])
    dummie = dummy_post_feb2022(fechas)
    assert list(dummie) == [0.0, 1.0, 1.0]
    assert FECHA_QUIEBRE == pd.Timestamp("2022-02-24")
    assert fecha_no_adelanta("2026-07-31", "2026-07-31")
    assert not fecha_no_adelanta("2026-07-30", "2026-07-31")


def probar_kalman_no_usa_el_precio_de_hoy_ni_el_futuro() -> None:
    from valor_nuevo import filtrar_kalman

    azar = np.random.default_rng(7)
    n = 420
    x = np.column_stack([np.ones(n), azar.normal(size=(n, 3))])
    y = x @ np.array([1.0, -0.4, 0.2, 0.1]) + azar.normal(0, 0.05, n)
    base = filtrar_kalman(x, y, min_obs=80)
    futuro = y.copy()
    futuro[300:] += 1.5
    con_futuro = filtrar_kalman(x, futuro, min_obs=80)
    assert np.isfinite(base["yhat"][200])
    assert abs(base["yhat"][200] - con_futuro["yhat"][200]) < 1e-8
    hoy = y.copy()
    hoy[200] += 4.0
    con_hoy = filtrar_kalman(x, hoy, min_obs=80)
    assert abs(base["yhat"][200] - con_hoy["yhat"][200]) < 1e-8
    assert abs(base["yhat"][230] - con_hoy["yhat"][230]) > 1e-4


def probar_markov_no_usa_el_futuro() -> None:
    from valor_nuevo import camino_markov

    azar = np.random.default_rng(8)
    n = 260
    x = np.column_stack([azar.normal(size=n), azar.normal(size=n), np.linspace(0, 1, n)])
    y = 0.4 * x[:, 0] - 0.2 * x[:, 1] + azar.normal(0, 0.15, n)
    y[180:] += 0.3
    nombres = ["real", "dolar", "post_feb2022"]
    base = camino_markov(x, y, nombres, min_obs=90, paso=200)
    futuro = y.copy()
    futuro[200:] += 2.0
    con_futuro = camino_markov(x, futuro, nombres, min_obs=90, paso=200)
    assert base["n_ajustes"] >= 1
    assert np.isfinite(base["yhat"][120])
    assert abs(base["yhat"][120] - con_futuro["yhat"][120]) < 1e-8


def probar_regimen_filtrado_ignora_el_futuro() -> None:
    from regimes import asignar_regimenes

    azar = np.random.default_rng(9)
    n = 360
    x = azar.normal(size=(n, 5))
    x[200:, 1] += 2.5
    chocante = x.copy()
    chocante[250:] += 4.0
    base = asignar_regimenes(x, min_obs=80, paso=60, ventana=160, semilla=0)
    otro = asignar_regimenes(chocante, min_obs=80, paso=60, ventana=160, semilla=0)
    assert np.allclose(base["estado"][:240], otro["estado"][:240], equal_nan=True)
    assert np.isfinite(base["estado"][200:240]).mean() > 0.8
    assert len(base["nombres"]) == 3


def probar_clasificador_no_entrena_con_la_etiqueta_abierta() -> None:
    from classify import clasificar_walk_forward, etiquetar_direccion
    from evaluacion import limite_pasado

    azar = np.random.default_rng(10)
    n = 360
    x = azar.normal(size=(n, 4))
    y = azar.normal(0, 0.02, n)
    vol = np.full(n, 0.01)
    corrida = clasificar_walk_forward(x, y, vol, horizonte=5, min_entrenamiento=80, paso=120, embargo=5)
    assert corrida["indice_max_entreno"] is not None
    assert corrida["indice_max_entreno"] <= limite_pasado(corrida["t_ultimo_ajuste"], 5, 5)
    futuro = y.copy()
    futuro[300:] += 0.2
    otra = clasificar_walk_forward(x, futuro, vol, horizonte=5, min_entrenamiento=80, paso=120, embargo=5)
    assert np.allclose(corrida["prob_up"][:280], otra["prob_up"][:280], equal_nan=True)
    clase, banda = etiquetar_direccion(np.array([0.05, 0.001, -0.05]), np.array([0.01, 0.01, 0.01]), horizonte=4)
    # Banda = 0.5 * 0.01 * 2 = 0.01. 0.05 alza, 0.001 plano, -0.05 baja.
    assert list(clase) == [1, 0, -1]
    assert abs(banda[0] - 0.01) < 1e-12


def probar_intervalo_conformal_no_entrena_con_el_futuro() -> None:
    from evaluacion import cobertura_intervalo, limite_pasado
    from intervalos import caminar_intervalos

    azar = np.random.default_rng(11)
    n = 320
    x = azar.normal(size=(n, 3))
    y = 0.1 * x[:, 0] + azar.normal(0, 0.05, n)
    banda = caminar_intervalos(x, y, horizonte=5, min_entrenamiento=90, paso=140, embargo=5)
    assert banda["indice_max_entreno"] is not None
    assert banda["indice_max_entreno"] <= limite_pasado(banda["t_ultimo_ajuste"], 5, 5)
    assert np.isfinite(banda["bajo"]).sum() > 20
    futuro = y.copy()
    futuro[280:] += 1.0
    otra = caminar_intervalos(x, futuro, horizonte=5, min_entrenamiento=90, paso=140, embargo=5)
    assert np.allclose(banda["mediana"][:260], otra["mediana"][:260], equal_nan=True)
    cob = cobertura_intervalo(y, banda["bajo"], banda["alto"], horizonte=5)
    assert cob["sin_solapar"]["n"] > 5
    assert 0.0 <= cob["sin_solapar"]["cobertura"] <= 1.0


def probar_compra_en_partes_adelanta_la_caja() -> None:
    from classify import backtest_compra_en_partes

    precios = np.array([100.0, 200.0])
    alta = np.array([0.9, 0.9])
    partes = backtest_compra_en_partes(precios, alta, paso=1, umbral_alto=0.6, umbral_bajo=0.35)
    assert partes["disponible"]
    assert abs(partes["dolares"] - 2) < 1e-9
    assert abs(partes["dolares_partes"] - 2) < 1e-9
    assert abs(partes["onzas_partes"] - 0.02) < 1e-9
    assert partes["onzas_partes"] > partes["onzas_dca"]
    baja = backtest_compra_en_partes(precios, np.array([0.1, 0.1]), paso=1, umbral_alto=0.6, umbral_bajo=0.35)
    assert baja["onzas_partes"] < baja["onzas_dca"]


def probar_snapshot_publico_trae_probabilidades() -> None:
    modulo = _modulo_export()
    estudio = {
        "esquema": "prueba",
        "avisos": [],
        "clasificacion": {5: {"prob_up": 0.42, "prob_plano": 0.2, "prob_baja": 0.38, "modelo": "logistica", "metricas": {}}},
        "comparacion": {"promovido": {"5": False, "20": False}, "filas": []},
        "valor": {"promovido": False},
        "intervalos": {},
        "regimen": {"disponible": False},
        "analogos": {"disponible": False, "periodos": []},
        "dca": {"disponible": False},
    }
    publico = modulo._estudio_publico(estudio)
    assert publico["clasificacion"]["5"]["prob_up"] == 0.42
    assert modulo._prob_publica(estudio, 5) == 0.42
    assert modulo._valor_titular({"valor_justo_vf": 100.0, "brecha_vf_pct": 1.0}, estudio)[2] == "base"


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
