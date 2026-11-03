"""División de datos, métricas y flujo de evaluación, siempre agrupando por estudiante."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import StratifiedGroupKFold, cross_validate

from .config import OBJETIVO, RASGOS, SEMILLA
from .modelos import crear_modelos

METRICAS = {
    "roc_auc": "AUC ROC",
    "balanced_accuracy": "Exactitud balanceada",
    "recall": "Sensibilidad",
    "precision": "Precisión",
    "f1": "F1",
}
"""Métricas de scikit-learn y su nombre en los reportes. La clase positiva es «pierde»."""


@dataclass
class Particion:
    X_entrenamiento: pd.DataFrame
    X_prueba: pd.DataFrame
    y_entrenamiento: pd.Series
    y_prueba: pd.Series
    grupos_entrenamiento: pd.Series


@dataclass
class Resultado:
    particion: Particion
    validacion_cruzada: pd.DataFrame
    prueba: pd.DataFrame
    ajustados: dict[str, BaseEstimator]

    @property
    def mejor(self) -> str:
        """Modelo con mayor AUC media en validación cruzada; nunca se elige mirando la prueba."""
        return self.validacion_cruzada["AUC ROC media"].idxmax()


def _pliegues(n: int, semilla: int) -> StratifiedGroupKFold:
    return StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=semilla)


def dividir(df: pd.DataFrame, semilla: int = SEMILLA) -> Particion:
    """Separa ~20 % para la prueba final sin repartir a un mismo estudiante entre los dos lados.

    Varias materias comparten casi las mismas notas por estudiante; si sus filas quedaran a
    ambos lados, la prueba estaría contaminada.
    """
    grupos = df["estudiante"]
    entrenamiento, prueba = next(_pliegues(5, semilla).split(df, df[OBJETIVO], grupos))
    ent, pru = df.iloc[entrenamiento], df.iloc[prueba]
    return Particion(
        ent[RASGOS], pru[RASGOS], ent[OBJETIVO], pru[OBJETIVO], grupos.iloc[entrenamiento]
    )


def validacion_cruzada(
    modelos: dict[str, BaseEstimator], particion: Particion, semilla: int = SEMILLA
) -> pd.DataFrame:
    """Media y desviación de cada métrica en 5 pliegues del conjunto de entrenamiento."""
    filas = {}
    for nombre, modelo in modelos.items():
        r = cross_validate(
            modelo,
            particion.X_entrenamiento,
            particion.y_entrenamiento,
            groups=particion.grupos_entrenamiento,
            cv=_pliegues(5, semilla),
            scoring=list(METRICAS),
            error_score="raise",
        )
        filas[nombre] = {
            f"{METRICAS[m]} {estadistico}": f(r[f"test_{m}"])
            for m in METRICAS
            for estadistico, f in (("media", np.mean), ("desv.", np.std))
        }
    return pd.DataFrame(filas).T.sort_values("AUC ROC media", ascending=False)


def evaluar_en_prueba(
    modelos: dict[str, BaseEstimator], particion: Particion
) -> tuple[pd.DataFrame, dict[str, BaseEstimator]]:
    """Entrena cada modelo con todo el entrenamiento y lo mide una sola vez en la prueba."""
    ajustados = {
        nombre: clone(m).fit(particion.X_entrenamiento, particion.y_entrenamiento)
        for nombre, m in modelos.items()
    }
    tabla = pd.DataFrame(
        {
            nombre: {
                etiqueta: get_scorer(m)(modelo, particion.X_prueba, particion.y_prueba)
                for m, etiqueta in METRICAS.items()
            }
            for nombre, modelo in ajustados.items()
        }
    ).T.sort_values("AUC ROC", ascending=False)
    return tabla, ajustados


def analizar(df: pd.DataFrame, semilla: int = SEMILLA) -> Resultado:
    """Flujo completo: división, validación cruzada y evaluación final de todos los modelos."""
    particion = dividir(df, semilla)
    modelos = crear_modelos(semilla)
    prueba, ajustados = evaluar_en_prueba(modelos, particion)
    return Resultado(particion, validacion_cruzada(modelos, particion, semilla), prueba, ajustados)
