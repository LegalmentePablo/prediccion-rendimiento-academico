import pandas as pd
import pytest

from prediccion_rendimiento.config import MATERIAS
from prediccion_rendimiento.datos import pierde, reporte_calidad, validar


def test_carga_las_cinco_materias_en_formato_largo(df):
    assert list(df["materia"].cat.categories) == list(MATERIAS)
    assert len(df) == 13_587
    assert {"estudiante", "nota_p1", "nota_p2", "pierde_p1", "pierde_p2"} <= set(df.columns)


def test_la_perdida_se_define_por_nota_menor_a_tres():
    assert pierde(pd.Series([0.0, 2.99, 3.0, 5.0])).tolist() == [1, 1, 0, 0]


def test_el_reporte_detecta_las_etiquetas_incoherentes_de_ciencias_naturales(df):
    calidad = reporte_calidad(df)
    assert calidad.loc["Ciencias naturales", "etiqueta_p1_coherente"] < 0.6
    assert calidad.loc["Ciencias sociales", "etiqueta_p1_coherente"] == 1.0


@pytest.mark.parametrize(
    ("cambio", "mensaje"),
    [
        (lambda d: d.drop(columns="Periodo2"), "faltan columnas"),
        (lambda d: d.assign(Periodo1=[None, 3.0]), "vacíos"),
        (lambda d: d.assign(Periodo1=[6.0, 3.0]), "fuera de la escala"),
        (lambda d: d.assign(Estudiantes=[1, 1]), "repetidos"),
    ],
)
def test_validar_rechaza_datos_invalidos(cambio, mensaje):
    base = pd.DataFrame(
        {
            "Estudiantes": [1, 2],
            "Periodo1": [2.0, 3.0],
            "Periodo2": [4.0, 1.0],
            "PierdePeriodo1": [1, 0],
            "PierdePeriodo2": [0, 1],
        }
    )
    with pytest.raises(ValueError, match=mensaje):
        validar(cambio(base))
