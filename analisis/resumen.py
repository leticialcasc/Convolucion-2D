#!/usr/bin/env python3
"""Calcula speedup, eficiencia y Karp-Flatt a partir de resultados/<maquina>/*.csv
Uso: python3 analisis/resumen.py resultados/local
El baseline (V0) debe venir de la MISMA maquina que V1/V2."""
import csv, glob, os, sys

d = sys.argv[1] if len(sys.argv) > 1 else "resultados/local"
filas = []
for f in sorted(glob.glob(os.path.join(d, "v[0-9].csv"))):
    filas += list(csv.DictReader(open(f)))

clave = lambda r: (r["imagenes"], r["ancho"], r["alto"], r["kernel"])
base = {clave(r): float(r["media_ms"]) for r in filas
        if r["version"] == "V0_Secuencial" and r["medida"] == "total"}

out = []
for r in filas:
    if clave(r) not in base:
        continue
    t = float(r["media_ms"])
    s = base[clave(r)] / t
    ef = kf = ""
    if r["version"] == "V1_OpenMP":
        p = int(r["config"])
        ef = s / p
        if p > 1:
            kf = (1 / s - 1 / p) / (1 - 1 / p)   # fraccion serial experimental
    out.append({**r, "speedup": round(s, 4),
                "eficiencia": round(ef, 4) if ef != "" else "",
                "karp_flatt": round(kf, 4) if kf != "" else ""})

if not out:
    sys.exit("Sin datos (revisa que exista V0 en la misma carpeta)")
with open(os.path.join(d, "resumen.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0].keys()))
    w.writeheader(); w.writerows(out)
print(f"Escrito {os.path.join(d, 'resumen.csv')} ({len(out)} filas)")

try:  # graficas opcionales
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(os.path.join(d, "graficas"), exist_ok=True)
    casos = sorted({clave(r) for r in out if r["version"] == "V1_OpenMP"})
    for c in casos:
        plt.figure()
        for v in sorted({r["variante"] for r in out if r["version"] == "V1_OpenMP"}):
            pts = sorted((int(r["config"]), r["speedup"]) for r in out
                         if r["version"] == "V1_OpenMP" and r["variante"] == v and clave(r) == c)
            if pts: plt.plot(*zip(*pts), marker="o", label=v)
        hs = sorted({int(r["config"]) for r in out if r["version"] == "V1_OpenMP"})
        plt.plot(hs, hs, "k--", label="ideal")
        plt.xlabel("Hilos"); plt.ylabel("Speedup"); plt.legend()
        plt.title("N=%s %sx%s kernel=%s" % c)
        plt.savefig(os.path.join(d, "graficas", "speedup_N%s_%sx%s_%s.png" % c), dpi=120)
        plt.close()
except ImportError:
    print("(matplotlib no instalado: se omiten graficas)")
