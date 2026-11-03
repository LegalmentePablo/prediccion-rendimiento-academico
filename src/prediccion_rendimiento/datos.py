"""Carga, validación y diagnóstico de calidad de las notas."""

from pathlib import Path

import pandas as pd

from .config import DIRECTORIO_DATOS, ESCALA, MATERIAS, NOTA_APROBATORIA

COLUMNAS_ORIGINALES = {
    "Estudiantes": "estudiante",
    "Periodo1": "nota_p1",
    "Periodo2": "nota_p2",
    "PierdePeriodo1": "pierde_p1_reportado",
    "PierdePeriodo2": "pierde_p2_reportado",
}


def pierde(nota: pd.Series) -> pd.Series:
    """1 si la nota está por debajo de la aprobatoria, 0 si no."""
    return (nota < NOTA_APROBATORIA).astype(int)


def validar(df: pd.DataFrame, origen: str = "datos") -> pd.DataFrame:
    """Falla con un mensaje claro si faltan columnas, hay vacíos o notas fuera de escala."""
    faltan = set(COLUMNAS_ORIGINALES) - set(df.columns)
    if faltan:
        raise ValueError(f"{origen}: faltan columnas {sorted(faltan)}")
    if df[list(COLUMNAS_ORIGINALES)].isna().any().any():
        raise ValueError(f"{origen}: hay valores vacíos")
    if not df[["Periodo1", "Periodo2"]].stack().between(*ESCALA).all():
        raise ValueError(f"{origen}: hay notas fuera de la escala {ESCALA}")
    if df["Estudiantes"].duplicated().any():
        raise ValueError(f"{origen}: hay estudiantes repetidos")
    return df


def cargar_datos(directorio: Path | str = DIRECTORIO_DATOS) -> pd.DataFrame:
    """Une las cinco materias en una tabla larga: una fila por estudiante y materia.

    Las etiquetas `pierde_p*` se recalculan con la nota (`nota < 3`); las que trae el archivo
    se conservan como `pierde_p*_reportado` porque no siempre son coherentes (ver
    `reporte_calidad`).
    """
    directorio = Path(directorio)
    partes = [
        validar(pd.read_csv(directorio / archivo), archivo)
        .rename(columns=COLUMNAS_ORIGINALES)
        .assign(materia=materia)
        for materia, archivo in MATERIAS.items()
    ]
    df = pd.concat(partes, ignore_index=True)
    df["materia"] = pd.Categorical(df["materia"], categories=list(MATERIAS))
    return df.assign(pierde_p1=pierde(df["nota_p1"]), pierde_p2=pierde(df["nota_p2"]))


def reporte_calidad(df: pd.DataFrame) -> pd.DataFrame:
    """Por materia: tamaño, tasa de pérdida, coherencia de las etiquetas y relación P1–P2."""

    def resumen(g: pd.DataFrame) -> pd.Series:
        return pd.Series(
            {
                "estudiantes": len(g),
                "pierde_p1": g["pierde_p1"].mean(),
                "pierde_p2": g["pierde_p2"].mean(),
                "etiqueta_p1_coherente": (g["pierde_p1"] == g["pierde_p1_reportado"]).mean(),
                "etiqueta_p2_coherente": (g["pierde_p2"] == g["pierde_p2_reportado"]).mean(),
                "correlacion_p1_p2": g["nota_p1"].corr(g["nota_p2"]),
            }
        )

    calidad = df.groupby("materia", observed=True).apply(resumen, include_groups=False)
    return calidad.astype({"estudiantes": int})
