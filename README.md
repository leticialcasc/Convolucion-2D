# Convolución 2D y filtrado de imágenes por lotes
**Patrón stencil: Secuencial → OpenMP → CUDA**

Cómputo Paralelo · 6BM1 · ESCOM-IPN · Proyecto integrador

## Descripción
Implementación de la convolución 2D para filtrado de imágenes por lotes, comparando tres estrategias de ejecución: secuencial, paralela en CPU con OpenMP y paralela en GPU con CUDA.

## Problema
El filtrado de imágenes mediante convolución 2D es una operación fundamental en visión por computadora (desenfoque, realce de bordes, nitidez, detección de características para redes neuronales, etc.). En escenarios reales rara vez se filtra una sola imagen: el preprocesamiento de datasets, las cámaras de vigilancia o los pipelines de entrenamiento necesitan aplicar el mismo kernel a lotes de cientos o miles de imágenes. La versión ingenua (cuatro ciclos anidados) cuesta O(N · H · W · K²) y se vuelve impráctica en CPU secuencial cuando N, la resolución o el kernel crecen.

## Datos
Lotes de 10, 100 y 500 imágenes en tres resoluciones (256×256, 512×512, 1024×1024), con cuatro kernels: `blur3`, `blur5` (promedio), `sobel` (bordes, Sobel-X) y `sharpen` (realce).

> **Nota:** los benchmarks usan lotes sintéticos pseudoaleatorios y deterministas (misma semilla en V0, V1 y V2), generados por `generar_lote()` en `src/common.h`. Esto aísla el costo de cómputo de la lectura de disco.

## Por qué es paralelizable
Cada píxel de salida depende solo de un vecindario fijo de la entrada y es independiente de los demás píxeles de salida → paralelismo a nivel de píxel/fila, de imagen (embarazosamente paralelo) y de kernel. El acceso a memoria es regular (ventana deslizante), lo que favorece caché/vectorización en CPU y memoria compartida/acceso coalescente en GPU.

## Versiones
| Versión | Archivo | Descripción |
|---|---|---|
| V0 – Secuencial | `src/v0_secuencial.cpp` | Línea base y referencia de correctitud |
| V1 – OpenMP | `src/v1_openmp.cpp` | `parallel for` por **lote** (imagen completa por hilo) o por **filas** (`collapse(2)` sobre imagen×fila); `schedule` static/dynamic/guided y número de hilos configurables |
| V2 – CUDA | `src/v2_cuda.cu` | Un hilo por píxel, bloques de 16×16, malla 3D (z = imagen). Variante `global` y variante `shared` (tile + halo en memoria compartida); kernel en memoria constante; bordes con clamping |

Todas las versiones usan el mismo manejo de bordes (clamping) y validan su salida contra V0 (`max_error` en el CSV).

## Estructura
```
Convolucion-2D/
├── src/            # common.h, v0_secuencial.cpp, v1_openmp.cpp, v2_cuda.cu
├── scripts/        # run_benchmarks.sh
├── analisis/       # resumen.py (speedup, eficiencia, Karp-Flatt, gráficas)
├── resultados/     # CSVs por máquina (resultados/<maquina>/)
├── Makefile
└── README.md
```

## Requisitos
- `g++` ≥ 9 con OpenMP (Linux/WSL2/macOS con `libomp`, o MinGW-w64)
- Para V2: GPU NVIDIA y CUDA Toolkit (`nvcc`) ≥ 11.5, o Google Colab/Kaggle
- Python 3 (opcional `matplotlib`) para `analisis/resumen.py`

> Versiones usadas: g++ 13.3.0 (Ubuntu 24.04), nvcc 13.0 (CUDA 13.0, driver 580.82.07), Python 3.12.3 (laptop) y 3.13.16 (Colab), GNU Make 4.3, Git 2.43.0.

## Compilación
```bash
make cpu                 # V0 y V1
make gpu                 # V2 (si hace falta otra arquitectura: make gpu ARCH=sm_75)
```

## Ejecución
```bash
./bin/v0_secuencial  N ancho alto kernel reps
./bin/v1_openmp      N ancho alto kernel modo sched hilos reps [chunk]
./bin/v2_cuda        N ancho alto kernel reps
```
Ejemplos:
```bash
./bin/v0_secuencial 100 512 512 sharpen 5
./bin/v1_openmp     100 512 512 sharpen lote static 8 5
./bin/v1_openmp     100 512 512 sharpen filas dynamic 8 5
./bin/v2_cuda       100 512 512 sharpen 5
```
Cada ejecución imprime una fila CSV:
`version,variante,medida,imagenes,ancho,alto,kernel,config,reps,media_ms,desv_ms,max_error`
- `medida=total` incluye (en CUDA) copias H2D + kernel + D2H; `medida=kernel` es solo el kernel.
- Cada medición se repite `reps` veces (≥5) tras un calentamiento; se reporta media y desviación estándar.

### Benchmarks completos
```bash
./scripts/run_benchmarks.sh                                      # barrido completo -> resultados/local/
RES=512 LOTES=100 KERNELS=sharpen ./scripts/run_benchmarks.sh    # prueba rápida
python3 analisis/resumen.py resultados/local                     # speedup, eficiencia, Karp-Flatt, gráficas
```
**Importante:** el speedup se calcula contra V0 *de la misma máquina*. Si V2 se corre en Colab, corre también V0 allí (`MAQUINA=colab-t4 ./scripts/run_benchmarks.sh`).

## Hardware de medición
| Máquina | CPU (núcleos/hilos) | RAM | GPU | SO |
|---|---|---|---|---|
| Laptop (`resultados/local`) | Intel Core i5-1035G1 @ 1.00 GHz (4 núcleos / 8 hilos) | 5.7 GiB visibles para WSL2 | — | Windows + WSL2 (Ubuntu 24.04) |
| Google Colab (`resultados/colab-t4`) | Intel Xeon @ 2.00 GHz (2 vCPU) | 12 GiB | NVIDIA Tesla T4 (15 GB) | Ubuntu 24.04 |

**Notas de medición**
- V0 y V1 se midieron en la laptop; V0 y V2 en Colab. El speedup de cada versión se calcula contra el V0 de su misma máquina.
- En la laptop se omitió el caso de 500 imágenes de 1024×1024 (≈6 GiB de RAM, no cabe en los 5.7 GiB de WSL2). En Colab sí se corrió.
- Todas las mediciones usan 1 calentamiento y 5 repeticiones, reportando media y desviación estándar.
- En V2, `total` incluye las copias CPU↔GPU y `kernel` mide solo el cómputo en GPU.

## Resultados
_(Se completan con `resultados/<maquina>/resumen.csv` y las gráficas generadas.)_

## Objetivo
Comparar el tiempo de ejecución y el speedup obtenido entre las versiones secuencial, OpenMP y CUDA sobre el filtrado por lotes de imágenes.

## Integrantes
* Alcántara Ascencio Leticia
* Najar Carbajal Santiago
* De la Cruz Fuentes Diego
