"""Comprobaciones que no necesitan internet.

Corre: python pruebas.py

Sirven para ver que el IPC no se adelanta, que el pronóstico no entrena con
el futuro, y que los pesos por gramo salen de la fórmula que dice el informe.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from datos import (
    extraer_cierre,
    fecha_publicacion_cpi,
    filas_desde_puntos_bls,
    pegar_cpi_sin_adelanto,
    rellenar_hacia_adelante,
    tabla_publicacion_cpi,
    variacion_anual_ipc,
)
from modelos import (
    ajustar_pronosticos,
    ajustar_valor_justo,
    indices_de_entrenamiento,
    metricas_pronostico,
    onza_a_mxn_por_gramo,
    preparar_muestra,
    pronostico_walk_forward,
    retorno_log_adelante,
    retorno_log_pasado,
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


def probar_cpi_bls_ignora_mes_sin_publicar() -> None:
    # El BLS manda "-" (o vacío) en el mes que todavía no publicó.
    puntos = [
        {"year": "2026", "period": "M09", "value": "324.368"},
        {"year": "2026", "period": "M10", "value": "-"},
        {"year": "2026", "period": "M11", "value": ""},
        {"year": "2026", "period": "M13", "value": "300.0"},
        {"year": "2026", "period": "M08", "value": "323.976"},
    ]
    filas = filas_desde_puntos_bls(puntos)
    assert filas == [
        (pd.Timestamp("2026-09-01"), 324.368),
        (pd.Timestamp("2026-08-01"), 323.976),
    ]


def probar_extraer_cierre_de_yfinance() -> None:
    columnas = pd.MultiIndex.from_tuples([("Adj Close", "GC=F"), ("Close", "GC=F")])
    tabla = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], columns=columnas)
    serie = extraer_cierre(tabla)
    assert list(serie.values) == [2.0, 4.0]


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
