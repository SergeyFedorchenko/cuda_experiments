#include <stdio.h>
#include <cuda.h>
#include <chrono>  // For CPU timing

#define N (1 << 24)  // Size of the array
#define THREADS_PER_BLOCK 256


__global__ void sumReductionGPU1(float *input, float *output, int n) {
    __shared__ float shared[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    shared[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    for (unsigned int s=1; s < blockDim.x; s *= 2) {
        if (tid % (2*s) == 0) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }
    // Write result of this block to output
    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}


__global__ void sumReductionGPU2(float *input, float *output, int n) {
    __shared__ float shared[THREADS_PER_BLOCK];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load elements into shared memory
    shared[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Perform reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared[tid] += shared[tid + s];
        }
        __syncthreads();
    }

    // Write result of this block to output
    if (tid == 0) {
        output[blockIdx.x] = shared[0];
    }
}

__host__ float sumReductionCPU(float *input, int n) {
    float result = 0;
    for (int i = 0; i < n; i++){
        result += input[i];
    }
    return result;
}


int main() {
    float *h_input, *h_output;
    float *d_input, *d_output;
    int size = N * sizeof(float);
    int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // Allocate host memory
    h_input = (float*)malloc(size);
    h_output = (float*)malloc(blocks * sizeof(float));

    // Initialize input array
    for (int i = 0; i < N; ++i) {
        h_input[i] = 1.0f; // Easy to verify, should sum to N
    }

    auto start_cpu = std::chrono::high_resolution_clock::now();
    float result_cpu = sumReductionCPU(h_input, N);
    auto end_cpu = std::chrono::high_resolution_clock::now();
    double cpu_time = std::chrono::duration<double, std::milli>(end_cpu - start_cpu).count();
    printf("Sum cpu = %f\n", result_cpu);

    
    // Allocate device memory
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, blocks * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Launch reduction kernel
    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    sumReductionGPU1<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, N);
    cudaEventRecord(stop_gpu);

    // Wait for GPU to finish
    cudaEventSynchronize(stop_gpu);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);

    // Copy partial sums back
    cudaMemcpy(h_output, d_output, blocks * sizeof(float), cudaMemcpyDeviceToHost);

    // Final sum on CPU
    float gpu_result = 0.0f;
    for (int i = 0; i < blocks; ++i) {
        gpu_result += h_output[i];
    }


    printf("CPU Result: %.2f, Time: %.4f ms\n", result_cpu, cpu_time);
    printf("GPU1 Result: %.2f, Time: %.4f ms\n", gpu_result, milliseconds);

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);

    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    // Allocate device memory
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, blocks * sizeof(float));

    // Copy input to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Launch reduction kernel
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    cudaEventRecord(start_gpu);
    sumReductionGPU2<<<blocks, THREADS_PER_BLOCK>>>(d_input, d_output, N);
    cudaEventRecord(stop_gpu);

    // Wait for GPU to finish
    cudaEventSynchronize(stop_gpu);

    // float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start_gpu, stop_gpu);

    // Copy partial sums back
    cudaMemcpy(h_output, d_output, blocks * sizeof(float), cudaMemcpyDeviceToHost);


    printf("GPU2 Result: %.2f, Time: %.4f ms\n", gpu_result, milliseconds);
    free(h_input);
    free(h_output);
    // Final sum on CPU
    // float gpu_result = 0.0f;
    // for (int i = 0; i < blocks; ++i) {
    //     gpu_result += h_output[i];
    // }


    return 0;
}
