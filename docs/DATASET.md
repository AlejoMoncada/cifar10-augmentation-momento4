# Conjunto de datos: CIFAR-10

CIFAR-10 fue presentado por Alex Krizhevsky, Vinod Nair y Geoffrey Hinton en el informe técnico de 2009 *Learning Multiple Layers of Features from Tiny Images*. Es un subconjunto de **80 Million Tiny Images**. La distribución oficial y su descripción se encuentran en [la página de CIFAR-10 de la Universidad de Toronto](https://www.cs.toronto.edu/~kriz/cifar.html).

Contiene 60 000 imágenes a color de 32 × 32 píxeles, en 10 clases con 6 000 imágenes por clase. La partición oficial tiene 50 000 imágenes de entrenamiento y 10 000 de prueba. Las clases son avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo, barco y camión. En la taxonomía original, **automóvil** y **camión** son categorías mutuamente excluyentes, aunque ambas representan vehículos terrestres.

## Particiones del experimento

Se reserva el 20 % del entrenamiento oficial para validación mediante una división estratificada con `random_state=42`; la prueba oficial no interviene en entrenamiento ni selección de hiperparámetros. Los conteos observados por clase y partición se escriben en [`results/dataset_summary.csv`](../results/dataset_summary.csv).

Conteos observados en `results/dataset_summary.csv`:

| Clase | Entrenamiento | Validación | Prueba |
| --- | ---: | ---: | ---: |
| Avión | 4 000 | 1 000 | 1 000 |
| Automóvil | 4 000 | 1 000 | 1 000 |
| Pájaro | 4 000 | 1 000 | 1 000 |
| Gato | 4 000 | 1 000 | 1 000 |
| Ciervo | 4 000 | 1 000 | 1 000 |
| Perro | 4 000 | 1 000 | 1 000 |
| Rana | 4 000 | 1 000 | 1 000 |
| Caballo | 4 000 | 1 000 | 1 000 |
| Barco | 4 000 | 1 000 | 1 000 |
| Camión | 4 000 | 1 000 | 1 000 |
| **Total** | **40 000** | **10 000** | **10 000** |

## Limitaciones

La resolución de 32 × 32 limita los detalles discriminativos y algunas transformaciones, como el reflejo vertical o el desenfoque, pueden alterar señales útiles o la plausibilidad de la etiqueta. También se han **reportado** duplicados próximos entre particiones en la literatura: Barz y Denzler (2020), *Do we train on test data? Purging CIFAR of near-duplicates*. Este repositorio realiza un cribado con pHash; sus pares son candidatos y requieren inspección. No se atribuye aquí ninguna cantidad publicada no verificada. El aumento de datos no reemplaza datos representativos.

El cribado local de los 50 000 ejemplos oficiales de entrenamiento frente a los 10 000 de prueba encontró **117 pares candidatos** con distancia Hamming pHash ≤ 4. Este conteo es una observación del procedimiento local, no una cantidad confirmada de duplicados ni una cifra atribuida al artículo citado.
