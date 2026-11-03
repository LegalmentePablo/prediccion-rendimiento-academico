from prediccion_rendimiento import dividir
from prediccion_rendimiento.cli import ejecutar
from prediccion_rendimiento.evaluacion import METRICAS


def test_ningun_estudiante_queda_en_entrenamiento_y_prueba(df, particion):
    estudiantes_prueba = set(df.loc[particion.X_prueba.index, "estudiante"])
    assert estudiantes_prueba.isdisjoint(set(particion.grupos_entrenamiento))
    assert 0.15 < len(particion.X_prueba) / len(df) < 0.25


def test_la_division_es_reproducible(df, particion):
    assert dividir(df).X_prueba.index.equals(particion.X_prueba.index)


def test_el_flujo_completo_genera_tablas_y_figuras(directorio_datos, tmp_path):
    resultado = ejecutar(directorio_datos, tmp_path)

    assert list(resultado["prueba"].columns) == list(METRICAS.values())
    assert resultado["prueba"]["AUC ROC"].between(0.5, 1).all()
    assert {p.name for p in tmp_path.glob("*.csv")} == {
        "calidad_datos.csv",
        "validacion_cruzada.csv",
        "prueba.csv",
    }
    assert len(list((tmp_path / "figuras").glob("*.png"))) == 5
