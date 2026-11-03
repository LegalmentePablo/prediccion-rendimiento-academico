"""Ejecuta el análisis completo y deja tablas y figuras en `reports/`.

Uso: `prediccion-rendimiento [--datos data/raw] [--salida reports]`
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from . import graficos
from .config import DIRECTORIO_DATOS, SEMILLA
from .datos import cargar_datos, reporte_calidad
from .evaluacion import analizar


def ejecutar(datos: Path, salida: Path, semilla: int = SEMILLA) -> dict:
    """Corre todo el flujo y devuelve las tablas producidas."""
    figuras_dir = salida / "figuras"
    figuras_dir.mkdir(parents=True, exist_ok=True)

    df = cargar_datos(datos)
    calidad = reporte_calidad(df)
    r = analizar(df, semilla)

    tablas = {
        "calidad_datos": calidad,
        "validacion_cruzada": r.validacion_cruzada,
        "prueba": r.prueba,
    }
    for nombre, tabla in tablas.items():
        tabla.round(4).to_csv(salida / f"{nombre}.csv")

    figuras = {
        "coherencia_etiquetas": graficos.coherencia_etiquetas(calidad),
        "riesgo_por_nota": graficos.riesgo_por_nota(df),
        "comparacion_modelos": graficos.comparar_modelos(
            r.prueba, ["AUC ROC", "Exactitud balanceada", "Sensibilidad"]
        ),
        "curvas_roc": graficos.curvas_roc(r.ajustados, r.particion),
        "matriz_confusion": graficos.matriz_confusion(r.ajustados[r.mejor], r.particion, r.mejor),
    }
    for nombre, figura in figuras.items():
        figura.savefig(figuras_dir / f"{nombre}.png", dpi=120)
        plt.close(figura)

    return tablas | {"mejor_modelo": r.mejor}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--datos", type=Path, default=DIRECTORIO_DATOS)
    parser.add_argument("--salida", type=Path, default=Path("reports"))
    parser.add_argument("--semilla", type=int, default=SEMILLA)
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):  # tildes legibles también en consolas de Windows
        sys.stdout.reconfigure(encoding="utf-8")

    resultado = ejecutar(args.datos, args.salida, args.semilla)
    print("Métricas en el conjunto de prueba:\n")
    print(resultado["prueba"].round(3).to_string())
    print(f"\nMejor modelo en validación cruzada: {resultado['mejor_modelo']}")
    print(f"Tablas y figuras en: {args.salida}/")


if __name__ == "__main__":
    main()
