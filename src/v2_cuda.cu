// V2 - CUDA
// Uso: ./bin/v2_cuda [N] [ancho] [alto] [kernel] [reps]
// Ejecuta dos variantes: "global" (solo memoria global) y "shared" (tile en memoria compartida).
// Reporta dos mediciones: "total" (H2D + kernel + D2H) y "kernel" (solo el kernel).
#include <cuda_runtime.h>
#include "common.h"

#define BX 16
#define BY 16
#define MAX_K 49   // hasta 7x7

#define CUDA_CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); std::exit(1); } } while (0)

__constant__ float c_kernel[MAX_K];

// Un hilo por pixel de salida, lecturas directas de memoria global (con clamping)
__global__ void conv_global(const float* in, float* out, int w, int h, int radio) {
    int x = blockIdx.x * BX + threadIdx.x;
    int y = blockIdx.y * BY + threadIdx.y;
    if (x >= w || y >= h) return;
    size_t off = (size_t)blockIdx.z * w * h;
    int tk = 2 * radio + 1;
    float suma = 0.0f;
    for (int ky = -radio; ky <= radio; ++ky)
        for (int kx = -radio; kx <= radio; ++kx) {
            int px = min(max(x + kx, 0), w - 1);
            int py = min(max(y + ky, 0), h - 1);
            suma += in[off + (size_t)py * w + px] * c_kernel[(ky + radio) * tk + (kx + radio)];
        }
    out[off + (size_t)y * w + x] = suma;
}

// El bloque carga a memoria compartida un tile de (BX+2r)x(BY+2r) (halo con clamping)
// y cada hilo calcula su pixel leyendo solo de memoria compartida.
__global__ void conv_shared(const float* in, float* out, int w, int h, int radio) {
    extern __shared__ float tile[];
    size_t off = (size_t)blockIdx.z * w * h;
    int tw = BX + 2 * radio, th = BY + 2 * radio;
    int x0 = blockIdx.x * BX - radio;
    int y0 = blockIdx.y * BY - radio;

    for (int idx = threadIdx.y * BX + threadIdx.x; idx < tw * th; idx += BX * BY) {
        int tx = idx % tw, ty = idx / tw;
        int gx = min(max(x0 + tx, 0), w - 1);
        int gy = min(max(y0 + ty, 0), h - 1);
        tile[idx] = in[off + (size_t)gy * w + gx];
    }
    __syncthreads();

    int x = blockIdx.x * BX + threadIdx.x;
    int y = blockIdx.y * BY + threadIdx.y;
    if (x >= w || y >= h) return;
    int tk = 2 * radio + 1;
    float suma = 0.0f;
    for (int ky = 0; ky < tk; ++ky)
        for (int kx = 0; kx < tk; ++kx)
            suma += tile[(threadIdx.y + ky) * tw + (threadIdx.x + kx)] * c_kernel[ky * tk + kx];
    out[off + (size_t)y * w + x] = suma;
}

int main(int argc, char** argv) {
    int n = arg_i(argc, argv, 1, 100), ancho = arg_i(argc, argv, 2, 512), alto = arg_i(argc, argv, 3, 512);
    std::string nombre = arg_s(argc, argv, 4, "sharpen");
    int reps = arg_i(argc, argv, 5, 5);

    int tk;
    std::vector<float> kernel = obtener_kernel(nombre, tk);
    if (tk * tk > MAX_K) { std::fprintf(stderr, "kernel demasiado grande\n"); return 1; }
    int radio = tk / 2;
    size_t tam = (size_t)ancho * alto, total = n * tam, bytes = total * sizeof(float);

    std::vector<float> entrada(total), ref(total, 0.0f), salida(total, 0.0f);
    generar_lote(entrada);
    procesar_lote_secuencial(entrada, ref, n, ancho, alto, kernel.data(), tk);   // referencia V0

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemcpyToSymbol(c_kernel, kernel.data(), tk * tk * sizeof(float)));

    dim3 bloque(BX, BY);
    dim3 grid((ancho + BX - 1) / BX, (alto + BY - 1) / BY, n);   // z = indice de imagen
    size_t smem = (size_t)(BX + 2 * radio) * (BY + 2 * radio) * sizeof(float);

    cudaEvent_t e0, e1, e2, e3;
    cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventCreate(&e2); cudaEventCreate(&e3);

    const char* variantes[2] = {"global", "shared"};
    for (int v = 0; v < 2; ++v) {
        auto lanzar = [&]() {
            if (v == 0) conv_global<<<grid, bloque>>>(d_in, d_out, ancho, alto, radio);
            else        conv_shared<<<grid, bloque, smem>>>(d_in, d_out, ancho, alto, radio);
        };
        // calentamiento
        CUDA_CHECK(cudaMemcpy(d_in, entrada.data(), bytes, cudaMemcpyHostToDevice));
        lanzar();
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<double> t_total, t_kernel;
        for (int r = 0; r < reps; ++r) {
            cudaEventRecord(e0);
            cudaMemcpy(d_in, entrada.data(), bytes, cudaMemcpyHostToDevice);
            cudaEventRecord(e1);
            lanzar();
            cudaEventRecord(e2);
            cudaMemcpy(salida.data(), d_out, bytes, cudaMemcpyDeviceToHost);
            cudaEventRecord(e3);
            CUDA_CHECK(cudaEventSynchronize(e3));
            CUDA_CHECK(cudaGetLastError());
            float ms_total, ms_kernel;
            cudaEventElapsedTime(&ms_total, e0, e3);
            cudaEventElapsedTime(&ms_kernel, e1, e2);
            t_total.push_back(ms_total);
            t_kernel.push_back(ms_kernel);
        }
        double err = max_error(ref, salida);
        double m, d;
        std::string var = variantes[v];
        std::string cfg = std::to_string(BX) + "x" + std::to_string(BY);
        media_desv(t_total, m, d);
        imprimir_fila("V2_CUDA", var, "total", n, ancho, alto, nombre, cfg, reps, m, d, err);
        media_desv(t_kernel, m, d);
        imprimir_fila("V2_CUDA", var, "kernel", n, ancho, alto, nombre, cfg, reps, m, d, err);
    }
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
