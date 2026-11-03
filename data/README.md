# Datos

Cinco archivos CSV en `raw/`, uno por materia. Se conservan **tal como se usaron en el proyecto
original**; el código no los modifica.

| Archivo | Materia | Filas |
|---|---|---:|
| `matematicas.csv` | Matemáticas | 2.718 |
| `ingles.csv` | Inglés | 2.717 |
| `Fisica.csv` | Física | 2.717 |
| `C_SOCIALES.csv` | Ciencias sociales | 2.717 |
| `C_Naturales.csv` | Ciencias naturales | 2.718 |

## Columnas

| Columna | Significado |
|---|---|
| `Estudiantes` | Número correlativo (1, 2, 3…). No contiene datos personales; se asume que el mismo número es el mismo estudiante en los cinco archivos (ver hallazgo 2). |
| `Periodo1`, `Periodo2` | Nota de cada periodo, escala de 0 a 5. Se aprueba con 3,0. |
| `PierdePeriodo1`, `PierdePeriodo2` | 1 si perdió ese periodo, 0 si no (según el archivo). |

Los datos son **anónimos**: no hay nombres, documentos ni ningún otro dato personal.

## Hallazgos de calidad

`reporte_calidad()` los calcula y `reports/calidad_datos.csv` los guarda.

1. **Etiquetas aleatorias en Ciencias naturales.** `PierdePeriodo*` coincide con `nota < 3` solo en
   ~51 % de las filas, lo mismo que lanzar una moneda. En las demás materias coincide entre 97,8 % y
   100 %. Por eso el código recalcula la etiqueta a partir de la nota y guarda la original como
   `pierde_p*_reportado`.
2. **Columnas copiadas.** `Periodo1` es idéntica fila a fila en Inglés, Física y Ciencias sociales,
   y sus `Periodo2` tienen una correlación de entre 0,78 y 0,94. Matemáticas y Ciencias naturales no
   se relacionan con las demás. Esto respalda que `Estudiantes` designa al mismo estudiante entre
   archivos, y por eso la evaluación agrupa por ese número: así un estudiante nunca queda a la vez
   en entrenamiento y en prueba.
3. **Bloque repetido en Matemáticas.** Unas 300 filas reaparecen idénticas (`Periodo1` y
   `Periodo2`) unas 814 posiciones más abajo, con otro número de estudiante. El agrupamiento no las
   une. Como el modelo solo ve la nota del periodo 1 y la materia, una copia equivale a otro
   estudiante con la misma nota, así que el efecto sobre la evaluación es mínimo.
4. **Inferencia:** por (1), (2) y (3), los datos parecen sintéticos o armados a partir de una base
   menor. No se pudo confirmar su origen, así que los resultados deben leerse como un ejercicio
   metodológico, no como una conclusión sobre estudiantes reales.
