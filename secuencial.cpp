#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

// Función para convolución 2D secuencial sobre una sola imagen (Escala de Grises)
void convolucion2D_secuencial(const float* entrada, float* salida, int ancho, int alto, 
                             const float* kernel, int tam_kernel) {
    int radio = tam_kernel / 2;

    for (int y = 0; y < alto; ++y) {
        for (int x = 0; x < ancho; ++x) {
            float suma = 0.0f;

            for (int ky = -radio; ky <= radio; ++ky) {
                for (int kx = -radio; kx <= radio; ++kx) {
                    // Posición con manejo de bordes (clamping)
                    int px = std::min(std::max(x + kx, 0), ancho - 1);
                    int py = std::min(std::max(y + ky, 0), alto - 1);

                    float pixel = entrada[py * ancho + px];
                    float peso_kernel = kernel[(ky + radio) * tam_kernel + (kx + radio)];

                    suma += pixel * peso_kernel;
                }
            }
            salida[y * ancho + x] = suma;
        }
    }
}

// Procesamiento por lote (Batch)
void procesar_lote_secuencial(const std::vector<float>& lote_entrada, std::vector<float>& lote_salida, 
                              int num_imagenes, int ancho, int alto, 
                              const float* kernel, int tam_kernel) {
    size_t tam_imagen = ancho * alto;
    
    for (int i = 0; i < num_imagenes; ++i) {
        const float* ptr_in = &lote_entrada[i * tam_imagen];
        float* ptr_out = &lote_salida[i * tam_imagen];
        
        convolucion2D_secuencial(ptr_in, ptr_out, ancho, alto, kernel, tam_kernel);
    }
}

int main() {
    // Configuración de prueba
    int num_imagenes = 100;
    int ancho = 512;
    int alto = 512;
    int tam_kernel = 3; // Kernel 3x3
    size_t tam_imagen = ancho * alto;

    // Reservar memoria para el lote de entrada y salida
    std::vector<float> lote_entrada(num_imagenes * tam_imagen, 128.0f); // Datos sintéticos
    std::vector<float> lote_salida(num_imagenes * tam_imagen, 0.0f);

    // Kernel de prueba (Sharpen / Realce)
    float kernel[9] = {
         0, -1,  0,
        -1,  5, -1,
         0, -1,  0
    };

    // --- MEDICIÓ N DE TIEMPO ---
    auto inicio = std::chrono::high_resolution_clock::now();

    procesar_lote_secuencial(lote_entrada, lote_salida, num_imagenes, ancho, alto, kernel, tam_kernel);

    auto fin = std::chrono::high_resolution_clock::now();
    // ---------------------------

    std::chrono::duration<double, std::milli> tiempo_ms = fin - inicio;

    std::cout << "--- Resultados Version Secuencial (V0) ---" << std::endl;
    std::cout << "Imagenes procesadas: " << num_imagenes << std::endl;
    std::cout << "Resolucion: " << ancho << "x" << alto << std::endl;
    std::cout << "Tiempo de ejecucion: " << tiempo_ms.count() << " ms" << std::endl;

    return 0;
}