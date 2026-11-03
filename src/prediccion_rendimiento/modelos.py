"""Catálogo de modelos. Todos reciben `nota_p1` y `materia` y predicen si se pierde el periodo 2."""

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from imblearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import ESCALA, NOTA_APROBATORIA, SEMILLA


class ReglaPeriodoAnterior(ClassifierMixin, BaseEstimator):
    """Línea base sin aprendizaje: «si perdió el periodo 1, perderá el periodo 2».

    Como puntaje de riesgo usa la distancia a la nota máxima, de modo que también tiene
    curva ROC y se puede comparar con los modelos. Ese puntaje solo ordena estudiantes: su
    corte de 0,5 (nota 2,5) no coincide con el de `predict` (nota 3,0), y no hace falta.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ReglaPeriodoAnterior":
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        riesgo = (ESCALA[1] - X["nota_p1"].to_numpy()) / (ESCALA[1] - ESCALA[0])
        return np.column_stack([1 - riesgo, riesgo])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (X["nota_p1"].to_numpy() < NOTA_APROBATORIA).astype(int)


class TasaPorMateria(ClassifierMixin, BaseEstimator):
    """Línea base que ignora la nota: el riesgo es la tasa de pérdida histórica de la materia.

    Mide cuánto se gana solo por saber la materia. Marca como «pierde» las materias cuya tasa
    supera la tasa global.
    """

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "TasaPorMateria":
        self.classes_ = np.array([0, 1])
        self.tasas_ = y.groupby(X["materia"], observed=True).mean()
        self.corte_ = y.mean()
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        riesgo = X["materia"].map(self.tasas_).astype(float).fillna(self.corte_).to_numpy()
        return np.column_stack([1 - riesgo, riesgo])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] > self.corte_).astype(int)


def _con_balanceo(clasificador: BaseEstimator, semilla: int) -> Pipeline:
    """SMOTENC → codificación → clasificador.

    Al ir dentro del pipeline, el sobremuestreo solo toca los datos de entrenamiento de cada
    pliegue; las métricas se calculan siempre sobre datos reales que el modelo no vio.
    """
    codificacion = ColumnTransformer(
        [
            ("nota", StandardScaler(), ["nota_p1"]),
            ("materia", OneHotEncoder(handle_unknown="ignore"), ["materia"]),
        ]
    )
    return Pipeline(
        [
            ("balanceo", SMOTENC(categorical_features=["materia"], random_state=semilla)),
            ("codificacion", codificacion),
            ("clasificador", clasificador),
        ]
    )


def crear_modelos(semilla: int = SEMILLA) -> dict[str, BaseEstimator]:
    """Modelos a comparar, con hiperparámetros moderados para no sobreajustar dos variables.

    Los hiperparámetros se fijaron de antemano y no se ajustaron mirando el conjunto de prueba.
    """
    clasificadores = {
        "Regresión logística": LogisticRegression(max_iter=1000),
        "Árbol de decisión": DecisionTreeClassifier(max_depth=4, random_state=semilla),
        "KNN": KNeighborsClassifier(n_neighbors=25),
        "Bosque aleatorio": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=20, random_state=semilla, n_jobs=-1
        ),
        "Red neuronal (MLP)": MLPClassifier(
            hidden_layer_sizes=(32, 16), early_stopping=True, max_iter=2000, random_state=semilla
        ),
    }
    lineas_base = {
        "Regla: nota P1 < 3": ReglaPeriodoAnterior(),
        "Tasa por materia": TasaPorMateria(),
    }
    return lineas_base | {
        nombre: _con_balanceo(clf, semilla) for nombre, clf in clasificadores.items()
    }
