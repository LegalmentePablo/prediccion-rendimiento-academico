import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from prediccion_rendimiento.config import RASGOS
from prediccion_rendimiento.modelos import ReglaPeriodoAnterior, crear_modelos

MODELOS = crear_modelos()


def test_los_rasgos_no_usan_informacion_del_periodo_2():
    assert not any("p2" in rasgo for rasgo in RASGOS)


def test_la_regla_base_predice_perdida_si_la_nota_p1_es_menor_a_tres():
    X = pd.DataFrame({"nota_p1": [1.0, 2.9, 3.0, 4.5], "materia": ["Física"] * 4})
    regla = ReglaPeriodoAnterior().fit(X, None)
    assert regla.predict(X).tolist() == [1, 1, 0, 0]
    riesgo = regla.predict_proba(X)[:, 1]
    assert np.all(np.diff(riesgo) < 0), "a mayor nota, menor riesgo"


@pytest.mark.parametrize("nombre", MODELOS)
def test_cada_modelo_entrena_y_da_una_prediccion_por_fila(nombre, particion):
    modelo = clone(MODELOS[nombre]).fit(particion.X_entrenamiento, particion.y_entrenamiento)
    prediccion = modelo.predict(particion.X_prueba)
    probabilidad = modelo.predict_proba(particion.X_prueba)
    # SMOTENC solo actúa al entrenar: en la predicción no se inventan filas.
    assert len(prediccion) == len(particion.X_prueba)
    assert set(np.unique(prediccion)) <= {0, 1}
    assert np.allclose(probabilidad.sum(axis=1), 1)
