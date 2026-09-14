# Convolución 2D y filtrado de imágenes por lotes

Patrón stencil: Secuencial → OpenMP → CUDA

## Descripción

Este proyecto implementa la convolución 2D para el procesamiento y filtrado de
imágenes por lotes, comparando tres estrategias de ejecución: secuencial,
paralela en CPU con OpenMP y paralela en GPU con CUDA.

## Problema

El filtrado de imágenes mediante convolución 2D es una operación fundamental en
visión por computadora (desenfoque, realce de bordes, nitidez, detección de
características para redes neuronales, etc.). En escenarios reales rara vez se
filtra una sola imagen: aplicaciones de preprocesamiento de datasets, cámaras
de vigilancia o pipelines de entrenamiento de modelos necesitan aplicar el
mismo kernel a lotes de cientos o miles de imágenes. La versión ingenua, con
cuatro ciclos anidados (imagen, fila, columna, kernel), tiene un costo de
O(N · H · W · K²) y se vuelve impráctica en CPU secuencial cuando el número de
imágenes (N), la resolución o el tamaño del kernel crecen. El proyecto aborda
esta limitación implementando y comparando tres estrategias de ejecución para
determinar cuánto y bajo qué condiciones cada una acelera el filtrado por
lotes.

## Datos

Se usará un conjunto de imágenes en escala de grises y RGB tomadas de un
dataset público de acceso libre (por ejemplo, un subconjunto de USC-SIPI o de
CIFAR-10/DIV2K), agrupado en lotes de distinto tamaño (10, 100 y 500 imágenes)
y en al menos tres resoluciones (256×256, 512×512 y 1024×1024) para medir
escalabilidad. Sobre cada imagen se aplicarán tres kernels representativos:
promedio/blur (3×3 y 5×5, filtro separable), Sobel (detección de bordes) y
sharpen (realce), lo que permite evaluar el efecto del tamaño del kernel
además del tamaño del lote.

## Por qué es paralelizable

La convolución 2D es un patrón stencil clásico: el valor de cada píxel de
salida depende únicamente de un vecindario fijo y acotado de píxeles de
entrada (definido por el kernel), y es completamente independiente del
cálculo de cualquier otro píxel de salida. Esto elimina las dependencias de
datos entre iteraciones y permite paralelismo en varios niveles: (a) a nivel
de píxel/fila dentro de una misma imagen, (b) a nivel de imagen dentro de un
lote (cada imagen se procesa de forma independiente, un caso
"embarazosamente paralelo"), y (c) a nivel de kernel/filtro si se aplican
varios a la vez. Además, el acceso a memoria sigue un patrón regular y
predecible (ventana deslizante), lo que favorece tanto la vectorización y el
uso de caché en CPU como el uso de memoria compartida y el acceso coalescente
en GPU.

## Versiones

### V0 — Secuencial

Implementación de referencia en C/C++ con ciclos anidados sobre imagen y
kernel. Sirve como línea base de correctitud y tiempo de ejecución.

### V1 — OpenMP

Paralelización con `#pragma omp parallel for` a nivel de filas de la imagen
y/o a nivel de lote, probando distintas políticas de reparto de carga
(`static`, `dynamic`) y número de hilos.

### V2 — CUDA

Un hilo por píxel (o por bloque de píxeles) de salida; uso de memoria
compartida para cachear el vecindario del stencil y reducir accesos a memoria
global, con manejo explícito de bordes (padding).

Las tres versiones se validarán contra la misma salida de referencia y se
compararán mediante tiempo de ejecución, speedup y eficiencia, variando
resolución de imagen y tamaño de lote.

## Objetivo

Comparar el tiempo de ejecución y el speedup obtenido entre las versiones
secuencial, OpenMP y CUDA sobre el filtrado por lotes de imágenes.

## Integrantes

- Alcántara Ascencio Leticia
- Najar Carbajal Santiago
- De la Cruz Fuentes Diego
