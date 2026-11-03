from pathlib import Path

import pytest

from prediccion_rendimiento import cargar_datos, dividir


@pytest.fixture(scope="session")
def directorio_datos():
    return Path(__file__).parents[1] / "data" / "raw"


@pytest.fixture(scope="session")
def df(directorio_datos):
    return cargar_datos(directorio_datos)


@pytest.fixture(scope="session")
def particion(df):
    return dividir(df)
