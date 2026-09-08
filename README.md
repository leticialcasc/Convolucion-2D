# Convolución 2D y filtrado de imágenes por lotes

## Descripción

Este proyecto implementa la convolución 2D para el procesamiento y
filtrado de imágenes por lotes utilizando diferentes estrategias
de ejecución.

Se desarrollarán tres versiones:

- Secuencial
- OpenMP
- CUDA

El objetivo es comparar el rendimiento de las diferentes
implementaciones y analizar los beneficios de la paralelización.

## Problema

El procesamiento de grandes cantidades de imágenes mediante
convolución puede requerir un tiempo considerable. Debido a que
el cálculo de diferentes píxeles puede realizarse de manera
independiente, este problema puede ser paralelizado.

## Versiones

### Secuencial
Implementación de referencia utilizando CPU.

### OpenMP
Implementación paralela utilizando múltiples hilos de CPU.

### CUDA
Implementación paralela utilizando GPU.

## Datos

Se utilizará un conjunto de imágenes como entrada y se aplicarán
diferentes filtros mediante convolución 2D.

## Objetivo

Comparar el tiempo de ejecución y el speedup obtenido entre las
versiones secuencial, OpenMP y CUDA.

## Integrantes

- Alcántara Ascencio Leticia
- Najar Carbajal Santiago
