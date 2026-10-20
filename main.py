"""Un solo comando: python main.py

Baja datos, ajusta las dos regresiones e imprime el informe.
Las gráficas quedan en la carpeta outputs/.

Esto no es asesoría financiera.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from datos import actualizar_datos
from modelos import ajustar_oro_mxn, ajustar_pronosticos, ajustar_valor_justo, preparar_muestra
from reporte import construir_informe, guardar_graficas, guardar_tablas

CARPETA_SALIDAS = Path(__file__).resolve().parent / "outputs"
CARPETA_ARTEFACTOS = Path("/opt/cursor/artifacts")


def _copiar_artefactos(rutas: list[Path]) -> None:
    if not CARPETA_ARTEFACTOS.exists():
        return
    for ruta in rutas:
        if ruta.exists():
            shutil.copy2(ruta, CARPETA_ARTEFACTOS / ruta.name)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass

    print("Bajando datos y ajustando el modelo. La primera vez puede tardar unos minutos.", file=sys.stderr)
    datos = actualizar_datos()
    muestra = preparar_muestra(datos["panel"])
    valor = ajustar_valor_justo(muestra)
    pronostico = ajustar_pronosticos(valor["df"])
    mxn = ajustar_oro_mxn(pronostico["df"])
    texto = construir_informe(datos, valor, pronostico, mxn)
    print(texto, end="")

    CARPETA_SALIDAS.mkdir(parents=True, exist_ok=True)
    ruta_informe = CARPETA_SALIDAS / "informe.txt"
    ruta_informe.write_text(texto, encoding="utf-8")
    graficas = guardar_graficas(pronostico["df"], valor, CARPETA_SALIDAS, mxn)
    guardar_tablas(valor, pronostico, CARPETA_SALIDAS, mxn)
    _copiar_artefactos([ruta_informe, *graficas])

    print("\nArchivos:", file=sys.stderr)
    print(f"  {ruta_informe}", file=sys.stderr)
    for grafica in graficas:
        print(f"  {grafica}", file=sys.stderr)


if __name__ == "__main__":
    main()
