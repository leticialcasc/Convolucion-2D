// V0 - Secuencial (linea base)
// Uso: ./bin/v0_secuencial [N=100] [ancho=512] [alto=512] [kernel=sharpen] [reps=5]
#include <chrono>
#include "common.h"

int main(int argc, char** argv) {
    int n = arg_i(argc, argv, 1, 100), ancho = arg_i(argc, argv, 2, 512), alto = arg_i(argc, argv, 3, 512);
    std::string nombre = arg_s(argc, argv, 4, "sharpen");
    int reps = arg_i(argc, argv, 5, 5);

    int tam_k;
    std::vector<float> kernel = obtener_kernel(nombre, tam_k);
    size_t tam_imagen = (size_t)ancho * alto;

    std::vector<float> entrada(n * tam_imagen), salida(n * tam_imagen, 0.0f);
    generar_lote(entrada);

    procesar_lote_secuencial(entrada, salida, n, ancho, alto, kernel.data(), tam_k);   // calentamiento

    std::vector<double> t;
    for (int r = 0; r < reps; ++r) {
        auto ini = std::chrono::steady_clock::now();
        procesar_lote_secuencial(entrada, salida, n, ancho, alto, kernel.data(), tam_k);
        auto fin = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double, std::milli>(fin - ini).count());
    }
    double m, d;
    media_desv(t, m, d);
    imprimir_fila("V0_Secuencial", "secuencial", "total", n, ancho, alto, nombre, "1", reps, m, d, 0.0);
    return 0;
}
