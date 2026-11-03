"""Figuras del análisis. Cada función devuelve la figura para mostrarla o guardarla."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from sklearn.base import BaseEstimator
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

from .config import MATERIAS, NOTA_APROBATORIA
from .evaluacion import Particion


def coherencia_etiquetas(calidad: pd.DataFrame) -> Figure:
    """Qué tanto coinciden las etiquetas del archivo con la regla `nota < 3`."""
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    columnas = {"etiqueta_p1_coherente": "Periodo 1", "etiqueta_p2_coherente": "Periodo 2"}
    calidad[list(columnas)].rename(columns=columnas).mul(100).plot.barh(ax=ax)
    ax.axvline(50, color="grey", ls="--", lw=1, label="Azar (50 %)")
    ax.invert_yaxis()
    ax.set(xlabel="% de filas donde la etiqueta coincide con la nota", ylabel="", xlim=(0, 100))
    ax.legend(loc="lower left", bbox_to_anchor=(1, 0))
    ax.set_title("Coherencia de las etiquetas «Pierde» del archivo original")
    return fig


def riesgo_por_nota(df: pd.DataFrame, minimo: int = 30) -> Figure:
    """Proporción real de estudiantes que pierden el periodo 2 según su nota del periodo 1.

    Se omiten los tramos con menos de `minimo` estudiantes, cuyo porcentaje es puro ruido.
    """
    tramos = pd.cut(df["nota_p1"], bins=np.arange(0, 5.5, 0.5), include_lowest=True)
    resumen = df.groupby(["materia", tramos], observed=True)["pierde_p2"].agg(["mean", "size"])
    tasa = resumen["mean"].where(resumen["size"] >= minimo).unstack(0)
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    centros = [intervalo.mid for intervalo in tasa.index]
    for materia in MATERIAS:
        ax.plot(centros, tasa[materia] * 100, marker="o", label=materia)
    ax.axvline(NOTA_APROBATORIA, color="grey", ls="--", lw=1)
    ax.set(
        xlabel="Nota del periodo 1",
        ylabel="% que pierde el periodo 2",
        title="Riesgo de perder el periodo 2 según la nota del periodo 1",
        ylim=(0, 100),
    )
    ax.legend()
    return fig


def comparar_modelos(metricas: pd.DataFrame, columnas: list[str]) -> Figure:
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    metricas[columnas].plot.bar(ax=ax, rot=20)
    ax.set(ylim=(0, 1), xlabel="", ylabel="Valor", title="Modelos en el conjunto de prueba")
    ax.legend(loc="upper right", ncols=len(columnas))
    return fig


def curvas_roc(modelos: dict[str, BaseEstimator], particion: Particion) -> Figure:
    fig, ax = plt.subplots(figsize=(7, 6), layout="constrained")
    for nombre, modelo in modelos.items():
        RocCurveDisplay.from_estimator(
            modelo, particion.X_prueba, particion.y_prueba, name=nombre, ax=ax
        )
    ax.plot([0, 1], [0, 1], color="grey", ls="--", lw=1)
    ax.set(
        xlabel="Tasa de falsos positivos",
        ylabel="Tasa de verdaderos positivos (sensibilidad)",
        title="Curvas ROC en el conjunto de prueba",
    )
    return fig


def matriz_confusion(modelo: BaseEstimator, particion: Particion, nombre: str) -> Figure:
    fig, ax = plt.subplots(figsize=(5, 4.5), layout="constrained")
    ConfusionMatrixDisplay.from_estimator(
        modelo,
        particion.X_prueba,
        particion.y_prueba,
        display_labels=["Aprueba", "Pierde"],
        cmap="Blues",
        colorbar=False,
        ax=ax,
    )
    ax.set(xlabel="Predicción", ylabel="Real", title=f"Matriz de confusión — {nombre}")
    return fig
