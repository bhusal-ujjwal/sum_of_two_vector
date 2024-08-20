#include <stdio.h>
#include <iostream>
#include <chrono>

#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

void sumVectorsCPU(int *a, int *b, int *c, int size) {
    for (int i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

__global__ void sumVectorsGPUManual(int *a, int *b, int *c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

__global__ void sumVectorsGPUAuto(int *a, int *b, int *c, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        c[tid] = a[tid] + b[tid];
    }
}

#define vectorDim 1000

int main() {
    for (int size = 100000; size < 1000000; size += 10000) {
        int *dataA = new int[size];
        int *dataB = new int[size];
        int *dataC = new int[size];

        // Fill dataA and dataB with values
        for (int i = 0; i < size; ++i) {
            dataA[i] = i + 1;
            dataB[i] = i + 2;
        }

        // CPU version
        auto startCPU = std::chrono::steady_clock::now();
        sumVectorsCPU(dataA, dataB, dataC, size);
        auto endCPU = std::chrono::steady_clock::now();
        auto timeCPU = std::chrono::duration_cast<std::chrono::microseconds>(endCPU - startCPU).count();

        // GPU version with manual memory management
        int *d_dataA, *d_dataB, *d_dataC;
        cudaMalloc((void**)&d_dataA, size * sizeof(int));
        cudaMalloc((void**)&d_dataB, size * sizeof(int));
        cudaMalloc((void**)&d_dataC, size * sizeof(int));

        auto startGPUManual = std::chrono::steady_clock::now();
        cudaMemcpy(d_dataA, dataA, size * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_dataB, dataB, size * sizeof(int), cudaMemcpyHostToDevice);

        int threadsPerBlock = 256;
        int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
        sumVectorsGPUManual<<<blocksPerGrid, threadsPerBlock>>>(d_dataA, d_dataB, d_dataC, size);

        cudaMemcpy(dataC, d_dataC, size * sizeof(int), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        auto endGPUManual = std::chrono::steady_clock::now();
        auto timeGPUManual = std::chrono::duration_cast<std::chrono::microseconds>(endGPUManual - startGPUManual).count();

        // Free GPU memory
        cudaFree(d_dataA);
        cudaFree(d_dataB);
        cudaFree(d_dataC);

        // GPU version with automatic memory management (Unified Memory)
        int *d_dataA_auto, *d_dataB_auto, *d_dataC_auto;
        cudaMallocManaged(&d_dataA_auto, size * sizeof(int));
        cudaMallocManaged(&d_dataB_auto, size * sizeof(int));
        cudaMallocManaged(&d_dataC_auto, size * sizeof(int));

        for (int i = 0; i < size; ++i) {
            d_dataA_auto[i] = dataA[i];
            d_dataB_auto[i] = dataB[i];
        }

        auto startGPUAuto = std::chrono::steady_clock::now();
        int blocksPerGridAuto = (size + threadsPerBlock - 1) / threadsPerBlock;
        sumVectorsGPUAuto<<<blocksPerGridAuto, threadsPerBlock>>>(d_dataA_auto, d_dataB_auto, d_dataC_auto, size);
        cudaDeviceSynchronize();
        auto endGPUAuto = std::chrono::steady_clock::now();
        auto timeGPUAuto = std::chrono::duration_cast<std::chrono::microseconds>(endGPUAuto - startGPUAuto).count();

        // Free GPU memory (Unified Memory)
        cudaFree(d_dataA_auto);
        cudaFree(d_dataB_auto);
        cudaFree(d_dataC_auto);

        // Print results
        std::cout << size << "," << timeCPU << "," << timeGPUManual << "," << timeGPUAuto << std::endl;

        delete[] dataA;
        delete[] dataB;
        delete[] dataC;
    }

    return 0;
}