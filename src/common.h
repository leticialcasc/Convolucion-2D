// common.h - utilidades compartidas por V0, V1 y V2
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

// ---------------------------------------------------------------- kernels
inline std::vector<float> obtener_kernel(const std::string& nombre, int& tam) {
    if (nombre == "blur3")   { tam = 3; return std::vector<float>(9, 1.0f / 9.0f); }
    if (nombre == "blur5")   { tam = 5; return std::vector<float>(25, 1.0f / 25.0f); }
    if (nombre == "sobel")   { tam = 3; return {-1, 0, 1, -2, 0, 2, -1, 0, 1}; }   // Sobel-X
    if (nombre == "sharpen") { tam = 3; return {0, -1, 0, -1, 5, -1, 0, -1, 0}; }
    std::fprintf(stderr, "Kernel desconocido: %s (usa blur3|blur5|sobel|sharpen)\n", nombre.c_str());
    std::exit(1);
}

// ------------------------------------------------- V0: referencia secuencial
// (tu implementacion original, sin cambios de logica)
inline void convolucion2D_secuencial(const float* entrada, float* salida, int ancho, int alto,
                                     const float* kernel, int tam_kernel) {
    int radio = tam_kernel / 2;
    for (int y = 0; y < alto; ++y) {
        for (int x = 0; x < ancho; ++x) {
            float suma = 0.0f;
            for (int ky = -radio; ky <= radio; ++ky) {
                for (int kx = -radio; kx <= radio; ++kx) {
                    int px = std::min(std::max(x + kx, 0), ancho - 1);
                    int py = std::min(std::max(y + ky, 0), alto - 1);
                    suma += entrada[py * ancho + px] * kernel[(ky + radio) * tam_kernel + (kx + radio)];
                }
            }
            salida[y * ancho + x] = suma;
        }
    }
}

inline void procesar_lote_secuencial(const std::vector<float>& lote_entrada, std::vector<float>& lote_salida,
                                     int num_imagenes, int ancho, int alto,
                                     const float* kernel, int tam_kernel) {
    size_t tam_imagen = (size_t)ancho * alto;
    for (int i = 0; i < num_imagenes; ++i)
        convolucion2D_secuencial(&lote_entrada[i * tam_imagen], &lote_salida[i * tam_imagen],
                                 ancho, alto, kernel, tam_kernel);
}

// ------------------------------------------------------------- datos / metricas
// Datos sinteticos pseudoaleatorios (0..255) y deterministas: la misma semilla
// produce el mismo lote en V0, V1 y V2. (Un lote constante haria que Sobel dé
// todo 0 y no validaria nada.)
inline void generar_lote(std::vector<float>& v, uint32_t semilla = 42) {
    uint32_t s = semilla;
    for (auto& x : v) {
        s = s * 1664525u + 1013904223u;
        x = (float)((s >> 16) & 0xFF);
    }
}

inline double max_error(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

inline void media_desv(const std::vector<double>& v, double& media, double& desv) {
    media = 0.0;
    for (double x : v) media += x;
    media /= v.size();
    desv = 0.0;
    if (v.size() > 1) {
        for (double x : v) desv += (x - media) * (x - media);
        desv = std::sqrt(desv / (v.size() - 1));   // desviacion muestral
    }
}

// ------------------------------------------------------------------- salida CSV
// version,variante,medida,imagenes,ancho,alto,kernel,config,reps,media_ms,desv_ms,max_error
inline void imprimir_fila(const std::string& version, const std::string& variante, const std::string& medida,
                          int n, int ancho, int alto, const std::string& kernel, const std::string& config,
                          int reps, double media, double desv, double err) {
    std::printf("%s,%s,%s,%d,%d,%d,%s,%s,%d,%.4f,%.4f,%.6g\n", version.c_str(), variante.c_str(),
                medida.c_str(), n, ancho, alto, kernel.c_str(), config.c_str(), reps, media, desv, err);
}

inline std::string arg_s(int argc, char** argv, int i, const std::string& def) {
    return argc > i ? std::string(argv[i]) : def;
}
inline int arg_i(int argc, char** argv, int i, int def) { return argc > i ? std::atoi(argv[i]) : def; }
