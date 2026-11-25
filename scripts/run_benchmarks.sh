#!/usr/bin/env bash
# Corre V0, V1 y V2 (si existen los binarios) y guarda CSVs en resultados/<MAQUINA>/
# Ejemplos:
#   ./scripts/run_benchmarks.sh                       # todo (tarda)
#   RES="512" LOTES="100" KERNELS="sharpen" ./scripts/run_benchmarks.sh   # prueba rapida
#   MAQUINA=colab-t4 ./scripts/run_benchmarks.sh      # en Colab (corre tambien V0 ahi)
set -e
cd "$(dirname "$0")/.."
RES=${RES:-"256 512 1024"}
LOTES=${LOTES:-"10 100 500"}
KERNELS=${KERNELS:-"blur3 blur5 sobel sharpen"}
HILOS=${HILOS:-"1 2 4 8"}
MODOS=${MODOS:-"lote filas"}
SCHEDS=${SCHEDS:-"static dynamic"}
REPS=${REPS:-5}
MAQUINA=${MAQUINA:-local}
OUT=resultados/$MAQUINA
HEADER="version,variante,medida,imagenes,ancho,alto,kernel,config,reps,media_ms,desv_ms,max_error"
mkdir -p "$OUT"

[ -x bin/v0_secuencial ] && echo "$HEADER" > "$OUT/v0.csv"
[ -x bin/v1_openmp ]     && echo "$HEADER" > "$OUT/v1.csv"
[ -x bin/v2_cuda ]       && echo "$HEADER" > "$OUT/v2.csv"

for k in $KERNELS; do for r in $RES; do for n in $LOTES; do
  if [ $((n*r*r)) -gt ${MAX_PIXELES:-200000000} ]; then echo "saltando N=$n ${r}x${r} (RAM)" >&2; continue; fi
  echo ">> kernel=$k res=${r}x${r} N=$n" >&2
  [ -x bin/v0_secuencial ] && ./bin/v0_secuencial $n $r $r $k $REPS >> "$OUT/v0.csv"
  if [ -x bin/v1_openmp ]; then
    for m in $MODOS; do for s in $SCHEDS; do for h in $HILOS; do
      ./bin/v1_openmp $n $r $r $k $m $s $h $REPS >> "$OUT/v1.csv"
    done; done; done
  fi
  [ -x bin/v2_cuda ] && ./bin/v2_cuda $n $r $r $k $REPS >> "$OUT/v2.csv"
done; done; done
echo "Listo: $OUT" >&2
