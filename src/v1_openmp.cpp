// V1 - OpenMP (memoria compartida)
// Uso: ./bin/v1_openmp [N] [ancho] [alto] [kernel] [modo=lote|filas] [sched=static|dynamic|guided] [hilos] [reps] [chunk]
//   modo=lote  -> cada hilo procesa imagenes completas (paralelismo a nivel de imagen)
//   modo=filas -> se reparten pares (imagen, fila) entre hilos (collapse(2))
#include <omp.h>
#include "common.h"

static void conv_fila(const float* in, float* out, int ancho, int alto, const float* k, int tk, int y) {
    int radio = tk / 2;
    for (int x = 0; x < ancho; ++x) {
        float suma = 0.0f;
        for (int ky = -radio; ky <= radio; ++ky)
            for (int kx = -radio; kx <= radio; ++kx) {
                int px = std::min(std::max(x + kx, 0), ancho - 1);
                int py = std::min(std::max(y + ky, 0), alto - 1);
                suma += in[py * ancho + px] * k[(ky + radio) * tk + (kx + radio)];
            }
        out[y * ancho + x] = suma;
    }
}

static void lote_por_imagen(const float* in, float* out, int n, int ancho, int alto, const float* k, int tk) {
    size_t tam = (size_t)ancho * alto;
    #pragma omp parallel for schedule(runtime)
    for (int i = 0; i < n; ++i)
        convolucion2D_secuencial(in + i * tam, out + i * tam, ancho, alto, k, tk);
}

static void lote_por_filas(const float* in, float* out, int n, int ancho, int alto, const float* k, int tk) {
    size_t tam = (size_t)ancho * alto;
    #pragma omp parallel for collapse(2) schedule(runtime)
    for (int i = 0; i < n; ++i)
        for (int y = 0; y < alto; ++y)
            conv_fila(in + i * tam, out + i * tam, ancho, alto, k, tk, y);
}

int main(int argc, char** argv) {
    int n = arg_i(argc, argv, 1, 100), ancho = arg_i(argc, argv, 2, 512), alto = arg_i(argc, argv, 3, 512);
    std::string nombre = arg_s(argc, argv, 4, "sharpen");
    std::string modo = arg_s(argc, argv, 5, "lote");
    std::string sched = arg_s(argc, argv, 6, "static");
    int hilos = arg_i(argc, argv, 7, omp_get_max_threads());
    int reps = arg_i(argc, argv, 8, 5);
    int chunk = arg_i(argc, argv, 9, 0);

    if (modo != "lote" && modo != "filas") { std::fprintf(stderr, "modo: lote|filas\n"); return 1; }
    omp_sched_t kind = omp_sched_static;
    if (sched == "dynamic") kind = omp_sched_dynamic;
    else if (sched == "guided") kind = omp_sched_guided;
    else if (sched != "static") { std::fprintf(stderr, "sched: static|dynamic|guided\n"); return 1; }
    omp_set_schedule(kind, chunk);
    omp_set_num_threads(hilos);

    int tk;
    std::vector<float> kernel = obtener_kernel(nombre, tk);
    size_t tam = (size_t)ancho * alto;
    std::vector<float> entrada(n * tam), ref(n * tam, 0.0f), salida(n * tam, 0.0f);
    generar_lote(entrada);

    // Referencia (V0) para validar correctitud
    procesar_lote_secuencial(entrada, ref, n, ancho, alto, kernel.data(), tk);

    auto ejecutar = [&]() {
        if (modo == "lote") lote_por_imagen(entrada.data(), salida.data(), n, ancho, alto, kernel.data(), tk);
        else                lote_por_filas(entrada.data(), salida.data(), n, ancho, alto, kernel.data(), tk);
    };

    ejecutar();   // calentamiento
    std::vector<double> t;
    for (int r = 0; r < reps; ++r) {
        double ini = omp_get_wtime();
        ejecutar();
        t.push_back((omp_get_wtime() - ini) * 1000.0);
    }
    double m, d;
    media_desv(t, m, d);
    imprimir_fila("V1_OpenMP", modo + "_" + sched, "total", n, ancho, alto, nombre, std::to_string(hilos),
                  reps, m, d, max_error(ref, salida));
    return 0;
}
