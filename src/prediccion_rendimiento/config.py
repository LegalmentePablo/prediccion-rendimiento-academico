"""Constantes del proyecto: escala de notas, materias y semilla."""

from pathlib import Path

NOTA_APROBATORIA = 3.0
"""Nota mínima para aprobar en la escala colombiana de 0 a 5."""

ESCALA = (0.0, 5.0)
SEMILLA = 42
DIRECTORIO_DATOS = Path("data/raw")

MATERIAS = {
    "Matemáticas": "matematicas.csv",
    "Inglés": "ingles.csv",
    "Física": "Fisica.csv",
    "Ciencias sociales": "C_SOCIALES.csv",
    "Ciencias naturales": "C_Naturales.csv",
}
"""Nombre legible de cada materia y su archivo dentro de `data/raw`."""

RASGOS = ["nota_p1", "materia"]
"""Lo que se sabe al cerrar el periodo 1. Nunca incluye nada del periodo 2."""

OBJETIVO = "pierde_p2"
