#include <cuda_runtime.h>
#include <assert.h>
#define CUDA(_s) assert((_s) == cudaSuccess)
#define BLK 128
#define THR 128
#define KCALL(f,...) {f<<<BLK,THR>>>(__VA_ARGS__); CUDA(cudaGetLastError()); }

__global__ void _update32(int n, float* px, float* pv, float* pbest, float* gbest,
    double w, double l, double h, double vmin, double vmax, double r1, double r2) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < n){
        pv[i] = pv[i] * w + r1 * (pbest[i] - px[i]) + r2 * (gbest[i] - px[i]);
        pv[i] = pv[i] < vmin ? vmin : pv[i] > vmax ? vmax : pv[i];
        px[i] += pv[i];
        px[i] = px[i] < l ? l : px[i] > h ? h : px[i];
    }
}

__global__ void _update64(int n, double* px, double* pv, double* pbest, double* gbest,
    double w, double l, double h, double vmin, double vmax, double r1, double r2) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if(i < n){
        pv[i] = pv[i] * w + r1 * (pbest[i] - px[i]) + r2 * (gbest[i] - px[i]);
        pv[i] = pv[i] < vmin ? vmin : pv[i] > vmax ? vmax : pv[i];
        px[i] += pv[i];
        px[i] = px[i] < l ? l : px[i] > h ? h : px[i];
    }
}


extern "C" {
    void update64(int n, double* px, double* pv, double* pbest, double* gbest, double w, double l, double h, double vmin, double vmax, double r1, double r2) KCALL(_update64, n, px, pv, pbest, gbest, w, l, h, vmin, vmax, r1, r2);
    void update32(int n, float* px, float* pv, float* pbest, float* gbest, double w, double l, double h, double vmin, double vmax, double r1, double r2) KCALL(_update32, n, px, pv, pbest, gbest, w, l, h, vmin, vmax, r1, r2);    
}
